"""Real-GPU smoke for the single inference arbiter (do-not-repeat #50).

ONE server thread owns the Metal device and serves TWO DIFFERENT checkpoints
while two worker processes issue interleaved requests. Answers: does the arbiter
route correctly under real device load, with exactly one owner?

Two different checkpoints are deliberate. Identical weights would make a crossed
response INVISIBLE -- both models would return the same digest. Distinct weights
are what make routing falsifiable.

Dose (frozen by the card): 402 synchronous GPU calls = 2 reference calls + 400
routed requests (100 per model per worker, 2 workers). B=14, M=64, C=30,
active_size=24, max_batch_rows=14, flush_ms=2, seed 20260811. Workers issue in
OPPOSED order: worker 0 does A/B, worker 1 does B/A.

Reference digests are computed BY THE ARBITER THREAD -- they are sent through the
server like any other request. Computing them on the harness thread would put a
second thread on the device while trying to prove there is only one.

Because B equals max_batch_rows, every request is its own device batch, so this
smoke exercises DEVICE OWNERSHIP AND ROUTING. Mixed-model grouping inside one
flush is discharged by the deterministic non-GPU tests.

Exit: 0 pass · 1 verification fail · 3 artifact exists · 4 sha mismatch ·
142 timeout (SIGALRM) · -6/134 SIGABRT · anything else invalid.

    python -m scripts.GPU.alphazero.smoke_inference_arbiter
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .ipc_messages import InferenceRequest, StopSignal

# ---------------------------------------------------------------- frozen dose
MODEL_A = "model_a"
MODEL_B = "model_b"
CHECKPOINTS = {
    MODEL_A: ("checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors",
              "209cf2d4fd24a48553d259dd71b4954867b9473e"),
    MODEL_B: ("checkpoints/alphazero-v2-staged/model_iter_0379.safetensors",
              "8ad62ac432c35c6ea9b0630b8a2b8c572a0b03a1"),
}
HIDDEN, BLOCKS = 128, 6
ACTIVE_SIZE, BATCH, MOVES, CHANNELS = 24, 14, 64, 30
MAX_BATCH_ROWS, FLUSH_MS = 14, 2
N_WORKERS = 2
REQUESTS_PER_MODEL_PER_WORKER = 100
INPUT_SEED = 20260811
TIMEOUT_S = 900
OUT_JSON = Path("logs/eval/arbiter_smoke.json")
OUT_EXIT = Path("logs/eval/arbiter_smoke.exit")

EXPECTED_REQUESTS = N_WORKERS * REQUESTS_PER_MODEL_PER_WORKER      # 200 per model
EXPECTED_ROWS = EXPECTED_REQUESTS * BATCH                          # 2,800 per model
EXPECTED_BATCHES = EXPECTED_REQUESTS                               # 200: B == row cap


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_inputs():
    """Deterministic batch, IDENTICAL for both models and every call.

    Regenerated from the seed in each process rather than pickled, so workers and
    the reference calls provably use the same bytes.
    """
    rng = np.random.default_rng(INPUT_SEED)
    boards = rng.random((BATCH, ACTIVE_SIZE, ACTIVE_SIZE, CHANNELS), dtype=np.float32)
    rows = rng.integers(0, ACTIVE_SIZE, size=(BATCH, MOVES)).astype(np.int32)
    cols = rng.integers(0, ACTIVE_SIZE, size=(BATCH, MOVES)).astype(np.int32)
    mask = np.ones((BATCH, MOVES), dtype=np.float32)
    mask[:, MOVES // 2:] = 0.0
    return boards, rows, cols, mask


def digest(priors: np.ndarray, values: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.round(priors.astype(np.float64), 6).tobytes())
    h.update(np.round(values.astype(np.float64), 6).tobytes())
    return h.hexdigest()[:16]


def _shapes_ok(priors, values) -> bool:
    return list(priors.shape) == [BATCH, MOVES] and list(values.shape) == [BATCH]


def smoke_worker_main(worker_id, request_queue, response_queues, result_queue,
                      references, model_order):
    """Issue interleaved requests for both models and verify every response.

    `model_order` is this worker's opposed schedule, e.g. (A, B) or (B, A).
    """
    from .remote_evaluator import RemoteEvaluator
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    boards, rows, cols, mask = make_inputs()
    evaluators = {
        m: RemoteEvaluator(worker_id, request_queue, response_queues[m], model_id=m)
        for m in model_order
    }

    completed = {m: 0 for m in model_order}
    digest_ok = {m: True for m in model_order}
    shape_ok = {m: True for m in model_order}
    finite_ok = {m: True for m in model_order}
    err = None
    try:
        for _ in range(REQUESTS_PER_MODEL_PER_WORKER):
            for m in model_order:                    # opposed alternation
                priors, values = evaluators[m].infer(
                    boards, rows, cols, mask, ACTIVE_SIZE)
                completed[m] += 1
                if not _shapes_ok(priors, values):
                    shape_ok[m] = False
                if not (np.isfinite(priors).all() and np.isfinite(values).all()):
                    finite_ok[m] = False
                if digest(priors, values) != references[m]:
                    digest_ok[m] = False             # a crossed or drifted reply
    except Exception as exc:                          # noqa: BLE001 - reported, not swallowed
        err = f"{type(exc).__name__}: {exc}"

    result_queue.put({
        "worker_id": worker_id,
        "model_order": list(model_order),
        "completed": completed,
        "digest_ok": digest_ok,
        "shape_ok": shape_ok,
        "finite_ok": finite_ok,
        "error": err,
    })


def run() -> dict:
    from .network import create_network
    from .local_evaluator import LocalGPUEvaluator
    from .inference_server import InferenceServer

    boards, rows, cols, mask = make_inputs()
    evaluators = {}
    for model_id, (path, _sha) in CHECKPOINTS.items():
        net = create_network(hidden=HIDDEN, n_blocks=BLOCKS)
        net.load_weights(path)
        net.eval()
        evaluators[model_id] = LocalGPUEvaluator(net, compile=False)

    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue(maxsize=256)
    # (worker_id, model_id) addressing, plus a reference channel per model.
    REF_WORKER = -1
    response_queues = {
        (w, m): ctx.Queue(maxsize=64)
        for w in list(range(N_WORKERS)) + [REF_WORKER]
        for m in (MODEL_A, MODEL_B)
    }

    server = InferenceServer(
        evaluators=evaluators,
        request_queue=request_queue,
        response_queues=response_queues,
        max_batch_rows=MAX_BATCH_ROWS,
        flush_ms=FLUSH_MS,
    )
    server_thread = threading.Thread(target=server.run_forever, daemon=True)
    server_thread.start()

    t0 = time.perf_counter()

    # --- 2 reference calls, computed BY THE ARBITER THREAD ---
    references = {}
    for model_id in (MODEL_A, MODEL_B):
        request_queue.put(InferenceRequest(
            worker_id=REF_WORKER, request_id=1, boards=boards, move_rows=rows,
            move_cols=cols, move_mask=mask, active_size=ACTIVE_SIZE,
            model_id=model_id))
        resp = response_queues[(REF_WORKER, model_id)].get(timeout=120)
        references[model_id] = digest(resp.priors, resp.values)

    # Telemetry baseline AFTER the references, so the routed delta is exactly
    # the pinned 200 / 2,800 / 200 rather than 201 / 2,814 / 201.
    tel_before = server.model_telemetry()

    # --- 400 routed requests from two workers, opposed schedules ---
    result_queue = ctx.Queue()
    orders = {0: (MODEL_A, MODEL_B), 1: (MODEL_B, MODEL_A)}
    procs = []
    for wid in range(N_WORKERS):
        p = ctx.Process(target=smoke_worker_main, kwargs={
            "worker_id": wid,
            "request_queue": request_queue,
            "response_queues": {m: response_queues[(wid, m)] for m in (MODEL_A, MODEL_B)},
            "result_queue": result_queue,
            "references": references,
            "model_order": orders[wid],
        })
        p.start()
        procs.append(p)

    worker_reports = [result_queue.get(timeout=TIMEOUT_S) for _ in procs]
    for p in procs:
        p.join(timeout=30)

    tel_after = server.model_telemetry()
    elapsed = time.perf_counter() - t0

    try:
        request_queue.put(StopSignal(), timeout=0.5)
    except Exception:
        pass
    server.stop()
    server_thread.join(timeout=5.0)
    server_alive = server_thread.is_alive()
    for q in [request_queue, result_queue, *response_queues.values()]:
        try:
            q.cancel_join_thread()
            q.close()
        except Exception:
            pass

    routed = {
        m: {k: tel_after[m][k] - tel_before[m][k] for k in ("requests", "rows", "batches")}
        for m in (MODEL_A, MODEL_B)
    }
    per_model_completed = {
        m: sum(r["completed"][m] for r in worker_reports) for m in (MODEL_A, MODEL_B)
    }

    report = {
        "checkpoints": {m: {"path": p, "sha1": s} for m, (p, s) in CHECKPOINTS.items()},
        "dose": {"reference_calls": 2, "routed_requests": 400, "total_gpu_calls": 402,
                 "workers": N_WORKERS, "per_model_per_worker": REQUESTS_PER_MODEL_PER_WORKER},
        "batch": {"B": BATCH, "M": MOVES, "C": CHANNELS, "active_size": ACTIVE_SIZE,
                  "max_batch_rows": MAX_BATCH_ROWS, "flush_ms": FLUSH_MS},
        "input_seed": INPUT_SEED,
        "schedules": {str(w): list(o) for w, o in orders.items()},
        "references": references,
        "references_computed_on_arbiter_thread": True,
        "references_differ": references[MODEL_A] != references[MODEL_B],
        "per_model_completed": per_model_completed,
        "per_model_expected": {MODEL_A: EXPECTED_REQUESTS, MODEL_B: EXPECTED_REQUESTS},
        "routed_telemetry": routed,
        "expected_telemetry": {"requests": EXPECTED_REQUESTS, "rows": EXPECTED_ROWS,
                               "batches": EXPECTED_BATCHES},
        "worker_reports": worker_reports,
        "server_thread_stopped": not server_alive,
        "elapsed_s": round(elapsed, 2),
    }

    errors = [r["error"] for r in worker_reports if r["error"]]
    report["worker_errors"] = errors
    report["PASS"] = bool(
        not errors
        and all(per_model_completed[m] == EXPECTED_REQUESTS for m in (MODEL_A, MODEL_B))
        and all(r["digest_ok"][m] for r in worker_reports for m in (MODEL_A, MODEL_B))
        and all(r["shape_ok"][m] for r in worker_reports for m in (MODEL_A, MODEL_B))
        and all(r["finite_ok"][m] for r in worker_reports for m in (MODEL_A, MODEL_B))
        and report["references_differ"]                # models genuinely distinct
        and all(routed[m] == {"requests": EXPECTED_REQUESTS, "rows": EXPECTED_ROWS,
                              "batches": EXPECTED_BATCHES} for m in (MODEL_A, MODEL_B))
        and not server_alive
    )
    return report


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    for path in (OUT_JSON, OUT_EXIT):
        if path.exists():
            print(f"REFUSE: {path} already exists")
            return 3

    # Fail closed BEFORE any evaluator is built or Metal is touched.
    for model_id, (path, expected) in CHECKPOINTS.items():
        actual = _sha1_file(path)
        if actual != expected:
            print(f"REFUSE: {model_id} sha1 {actual} != expected {expected}")
            return 4

    signal.alarm(TIMEOUT_S)
    report = run()
    signal.alarm(0)

    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nPASS={report['PASS']}  written={OUT_JSON}")
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
