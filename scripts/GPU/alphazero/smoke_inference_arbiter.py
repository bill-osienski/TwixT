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

Reference digests are computed BY THE ARBITER THREAD, but by calling the two
evaluator instances DIRECTLY, before that same thread enters run_forever(). They
must not travel through the server: routing them would make the oracle circular,
because a systematic model-selection swap would swap the references too and every
later swapped response would match its swapped reference. Computing them on the
harness thread instead would put a second thread on the device while trying to
prove there is only one -- so it is the same thread, but not the same path.

Both evaluators are wrapped in a thread-ID recorder, so "exactly one thread ever
touched a model" is evidence rather than code inspection.

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

from .ipc_messages import StopSignal

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


class _ThreadRecordingEvaluator:
    """Wraps a real evaluator and records which thread called into it.

    Turns the one-owner claim into measured evidence: any second thread that
    reaches a model shows up here.
    """

    def __init__(self, inner, seen: set):
        self._inner = inner
        self._seen = seen

    def build_input_tensor(self, state):
        return self._inner.build_input_tensor(state)

    def infer(self, *args, **kwargs):
        self._seen.add(threading.get_ident())
        return self._inner.infer(*args, **kwargs)


def run() -> dict:
    import queue as _queue

    from .network import create_network
    from .local_evaluator import LocalGPUEvaluator
    from .inference_server import InferenceServer

    boards, rows, cols, mask = make_inputs()
    seen_threads: set = set()
    evaluators = {}
    for model_id, (path, _sha) in CHECKPOINTS.items():
        net = create_network(hidden=HIDDEN, n_blocks=BLOCKS)
        net.load_weights(path)
        net.eval()
        evaluators[model_id] = _ThreadRecordingEvaluator(
            LocalGPUEvaluator(net, compile=False), seen_threads)

    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue(maxsize=256)
    response_queues = {
        (w, m): ctx.Queue(maxsize=64)
        for w in range(N_WORKERS) for m in (MODEL_A, MODEL_B)
    }

    server = InferenceServer(
        evaluators=evaluators,
        request_queue=request_queue,
        response_queues=response_queues,
        max_batch_rows=MAX_BATCH_ROWS,
        flush_ms=FLUSH_MS,
    )

    # The arbiter thread computes the references by calling each evaluator
    # DIRECTLY -- same thread, different path -- then hands them to the harness
    # over a CPU-only queue and begins serving.
    ref_channel: "_queue.Queue" = _queue.Queue()

    def _arbiter() -> None:
        refs = {}
        for model_id in (MODEL_A, MODEL_B):
            priors, values = evaluators[model_id].infer(
                boards, rows, cols, mask, ACTIVE_SIZE)
            refs[model_id] = digest(priors, values)
        ref_channel.put(refs)          # CPU-only handoff, no device work
        server.run_forever()

    t0 = time.perf_counter()
    server_thread = threading.Thread(target=_arbiter, daemon=True)
    server_thread.start()
    references = ref_channel.get(timeout=300)

    # --- 400 routed requests from two workers, opposed schedules ---
    result_queue = ctx.Queue()
    orders = {0: (MODEL_A, MODEL_B), 1: (MODEL_B, MODEL_A)}
    procs = {}
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
        procs[wid] = p

    worker_reports = [result_queue.get(timeout=TIMEOUT_S) for _ in procs]

    # Worker lifecycle is a pass condition: a worker may publish a valid-looking
    # report and then hang or exit non-zero.
    worker_exit = {}
    for wid, p in procs.items():
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)
            worker_exit[wid] = "ALIVE_AFTER_JOIN"
        else:
            worker_exit[wid] = p.exitcode
    workers_clean = all(v == 0 for v in worker_exit.values())

    # Stop and JOIN the server before reading telemetry: responses are queued
    # before the counters increment, so reading earlier races the final flush.
    try:
        request_queue.put(StopSignal(), timeout=0.5)
    except Exception:
        pass
    server.stop()
    server_thread.join(timeout=30.0)
    server_alive = server_thread.is_alive()

    telemetry = server.model_telemetry()
    elapsed = time.perf_counter() - t0

    for q in [request_queue, result_queue, *response_queues.values()]:
        try:
            q.cancel_join_thread()
            q.close()
        except Exception:
            pass

    per_model_completed = {
        m: sum(r["completed"][m] for r in worker_reports) for m in (MODEL_A, MODEL_B)
    }
    expected_tel = {"requests": EXPECTED_REQUESTS, "rows": EXPECTED_ROWS,
                    "batches": EXPECTED_BATCHES}

    report = {
        "checkpoints": {m: {"path": p, "sha1": s} for m, (p, s) in CHECKPOINTS.items()},
        "dose": {"reference_calls": 2, "routed_requests": 400, "total_gpu_calls": 402,
                 "workers": N_WORKERS, "per_model_per_worker": REQUESTS_PER_MODEL_PER_WORKER},
        "batch": {"B": BATCH, "M": MOVES, "C": CHANNELS, "active_size": ACTIVE_SIZE,
                  "max_batch_rows": MAX_BATCH_ROWS, "flush_ms": FLUSH_MS},
        "input_seed": INPUT_SEED,
        "schedules": {str(w): list(o) for w, o in orders.items()},
        "references": references,
        "reference_oracle": "direct evaluator calls on the arbiter thread, not routed",
        "references_differ": references[MODEL_A] != references[MODEL_B],
        "per_model_completed": per_model_completed,
        "per_model_expected": {MODEL_A: EXPECTED_REQUESTS, MODEL_B: EXPECTED_REQUESTS},
        "telemetry": telemetry,
        "expected_telemetry": expected_tel,
        "worker_reports": worker_reports,
        "worker_exit_codes": worker_exit,
        "workers_clean": workers_clean,
        "server_thread_stopped": not server_alive,
        "inference_threads_observed": len(seen_threads),
        "elapsed_s": round(elapsed, 2),
    }

    errors = [r["error"] for r in worker_reports if r["error"]]
    report["worker_errors"] = errors
    report["PASS"] = bool(
        not errors
        and workers_clean
        and all(per_model_completed[m] == EXPECTED_REQUESTS for m in (MODEL_A, MODEL_B))
        and all(r["digest_ok"][m] for r in worker_reports for m in (MODEL_A, MODEL_B))
        and all(r["shape_ok"][m] for r in worker_reports for m in (MODEL_A, MODEL_B))
        and all(r["finite_ok"][m] for r in worker_reports for m in (MODEL_A, MODEL_B))
        and report["references_differ"]
        and all(telemetry[m] == expected_tel for m in (MODEL_A, MODEL_B))
        and len(seen_threads) == 1          # exactly one device owner, measured
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
