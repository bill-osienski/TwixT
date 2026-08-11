"""Single-arbiter Metal feasibility probe (do-not-repeat #50).

Answers ONE question: can a single Metal-owning thread serve two distinct
network instances under sustained interleaved load without a driver abort?

That is the untested premise of the arbiter design #50 prescribes. This probe is
deliberately smaller than the arbiter itself: if the premise is false, the
refactor is dead before it is written.

Two arms, identical total device work, ONE thread in both. There is no
two-thread arm: #50 forbids two owners submitting concurrently "in any
experiment, for any purpose", and a non-reproduction would not falsify a
timing-sensitive race anyway.

    control    one evaluator,  2N calls
    treatment  two evaluators, N calls each, strictly alternating

`LocalGPUEvaluator.infer()` calls `mx.eval` (local_evaluator.py:115,119), so
every call forces synchronous Metal execution rather than queuing lazy graphs.

Exit codes: 0 completed · -6/134 SIGABRT (the failure under test) · 142 timeout
(SIGALRM) · anything else invalid. Writes one JSON report per arm.

Run one arm per process:  python -m scripts.GPU.alphazero.probe_single_arbiter_metal --arm control
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- frozen dose
CHECKPOINT = "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
CHECKPOINT_SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"
HIDDEN = 128
BLOCKS = 6
ACTIVE_SIZE = 24
BATCH = 14            # matches --mcts-eval-batch-size
MOVES = 64            # padded legal-move slots per position
CHANNELS = 30         # NUM_CHANNELS
CALLS_PER_EVALUATOR = 200   # control does 2 * this on one evaluator
INPUT_SEED = 20260810
TIMEOUT_S = 900
COMPILE = False       # explicit: the training path never compiles
OUT_DIR = Path("logs/eval")


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_inputs(rng: np.random.Generator):
    """Deterministic batch, identical for every call and every arm."""
    boards = rng.random((BATCH, ACTIVE_SIZE, ACTIVE_SIZE, CHANNELS), dtype=np.float32)
    rows = rng.integers(0, ACTIVE_SIZE, size=(BATCH, MOVES)).astype(np.int32)
    cols = rng.integers(0, ACTIVE_SIZE, size=(BATCH, MOVES)).astype(np.int32)
    mask = np.ones((BATCH, MOVES), dtype=np.float32)
    mask[:, MOVES // 2:] = 0.0          # half padding, like a real late-game batch
    return boards, rows, cols, mask


def _digest(priors: np.ndarray, values: np.ndarray) -> str:
    """Small stable digest, so a pass cannot be vacuous."""
    h = hashlib.sha1()
    h.update(np.round(priors.astype(np.float64), 6).tobytes())
    h.update(np.round(values.astype(np.float64), 6).tobytes())
    return h.hexdigest()[:16]


def _build_evaluator():
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    net = create_network(hidden=HIDDEN, n_blocks=BLOCKS)
    net.load_weights(CHECKPOINT)
    net.eval()
    return LocalGPUEvaluator(net, compile=COMPILE)


def _check(priors: np.ndarray, values: np.ndarray) -> dict:
    return {
        "priors_shape": list(priors.shape),
        "values_shape": list(values.shape),
        "priors_shape_ok": list(priors.shape) == [BATCH, MOVES],
        "values_shape_ok": list(values.shape) == [BATCH],
        "all_finite": bool(np.isfinite(priors).all() and np.isfinite(values).all()),
    }


def run(arm: str) -> dict:
    rng = np.random.default_rng(INPUT_SEED)
    boards, rows, cols, mask = _make_inputs(rng)

    if arm == "control":
        evaluators = [_build_evaluator()]
        schedule = [0] * (2 * CALLS_PER_EVALUATOR)
    else:
        # Two INDEPENDENT instances of the same checkpoint: separate networks,
        # separate evaluators, both resident. Identical weights means their
        # digests must match -- a free correctness check on top of survival.
        evaluators = [_build_evaluator(), _build_evaluator()]
        schedule = [i % 2 for i in range(2 * CALLS_PER_EVALUATOR)]

    completed = [0] * len(evaluators)
    last = [None] * len(evaluators)
    checks = [None] * len(evaluators)

    t0 = time.perf_counter()
    for idx in schedule:
        priors, values = evaluators[idx].infer(boards, rows, cols, mask, ACTIVE_SIZE)
        completed[idx] += 1
        last[idx] = _digest(priors, values)
        if checks[idx] is None:
            checks[idx] = _check(priors, values)
    elapsed = time.perf_counter() - t0

    expected = [2 * CALLS_PER_EVALUATOR] if arm == "control" \
        else [CALLS_PER_EVALUATOR, CALLS_PER_EVALUATOR]

    report = {
        "arm": arm,
        "checkpoint": CHECKPOINT,
        "checkpoint_sha1_actual": _sha1_file(CHECKPOINT),
        "checkpoint_sha1_expected": CHECKPOINT_SHA1,
        "network": {"hidden": HIDDEN, "blocks": BLOCKS, "compile": COMPILE},
        "batch": {"B": BATCH, "M": MOVES, "C": CHANNELS, "active_size": ACTIVE_SIZE},
        "input_seed": INPUT_SEED,
        "n_evaluators": len(evaluators),
        "calls_expected": expected,
        "calls_completed": completed,
        "calls_match": completed == expected,
        "total_calls": sum(completed),
        "digests": last,
        "checks": checks,
        "elapsed_s": round(elapsed, 2),
        "mx_eval_forces_sync": True,   # local_evaluator.py:115,119
    }
    report["sha1_match"] = report["checkpoint_sha1_actual"] == CHECKPOINT_SHA1
    report["digests_agree"] = (len(set(last)) == 1) if len(evaluators) > 1 else None
    report["all_shapes_ok"] = all(c["priors_shape_ok"] and c["values_shape_ok"] for c in checks)
    report["all_finite"] = all(c["all_finite"] for c in checks)
    report["PASS"] = bool(
        report["calls_match"] and report["sha1_match"]
        and report["all_shapes_ok"] and report["all_finite"]
        and all(d is not None for d in last)
        and (report["digests_agree"] in (True, None))
    )
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("control", "treatment"), required=True)
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"arbiter_probe_{args.arm}.json"
    if out.exists():
        print(f"REFUSE: {out} already exists")
        return 3

    signal.alarm(TIMEOUT_S)          # SIGALRM -> 142, distinguishable from abort
    report = run(args.arm)
    signal.alarm(0)

    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nPASS={report['PASS']}  arm={args.arm}  written={out}")
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
