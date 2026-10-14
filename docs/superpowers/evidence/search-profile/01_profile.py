"""SEARCH PROFILE -- where does play-path search time go? (prepared 2026-10-14; the real run needs approval)

One run on the play path, unchanged:
* the pinned incumbent, loaded through eval_runner's own evaluator factory (LocalGPUEvaluator, compile=True);
* cfg_from(EvalConfig(**PROTOCOL), 400);
* MCTS.search_with_root(state, add_noise=False), the synchronous path play uses (eval_runner.play_eval_game).

The work is three frozen hand-built 24x24 positions at 400 simulations each (1,200 in all), plus ONE explicit
warm-up infer on P1's root that touches no MCTS random state. Time goes into coarse EXCLUSIVE buckets, each
measured by a wrapper installed in this process only; no repository file is edited. The wrappers' own cost is
estimated by calibrating against a no-op, and it is reported as an estimate.

  --dry       validate the fixtures, the seeds and the checkpoint hash. No model, no MLX, no files written.
  --out DIR   the real profile. DIR must exist (the launch creates it) and hold no record.json; writes
              DIR/record.json, create-only.

Exit 0 COMPLETED / 3 STOP_ERROR / 4 REFUSED.
"""
import hashlib
import json
import os
import pathlib
import platform
import random
import subprocess
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.GPU.alphazero import e4_screen_reference as REF  # noqa: E402
from scripts.GPU.alphazero import search_distillation_stage_b as B  # noqa: E402
from scripts.GPU.alphazero.eval_runner import EvalConfig, cfg_from  # noqa: E402
from scripts.GPU.alphazero.game.twixt_state import TwixtState  # noqa: E402
from scripts.GPU.alphazero.mcts import MCTS  # noqa: E402

CKPT = ROOT / "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"          # calib020_0001
SIMS = 400
EXIT = {"COMPLETED": 0, "STOP_ERROR": 3, "REFUSED": 4}

# Hand-built, red moves first, alternating. No corpus or held-out material. Red never on rows 0/23 and
# black never on columns 0/23, so neither side can have won. Frozen: the move list, the state digest
# (B.state_digest), the peg and link counts.
FIXTURES = {
    "P1": {"moves": [[11, 11], [11, 13], [13, 12], [9, 12], [9, 10], [12, 15]],
           "digest": "2fb43ca7011e62f31a20024f9bdcc4653cc099b072a0cdd056c8eab6f3757059", "pegs": 6, "links": 4},
    "P2": {"moves": [[3, 5], [2, 10], [5, 6], [3, 12], [7, 5], [2, 14], [9, 6], [3, 16], [11, 5], [2, 18],
                     [13, 6], [3, 20], [15, 5], [21, 2], [4, 16], [20, 4], [6, 17], [21, 6], [8, 16], [20, 8],
                     [10, 17], [12, 12], [14, 10], [13, 14], [16, 9], [12, 16], [18, 10], [13, 18], [20, 11],
                     [12, 20]],
           "digest": "d953b5533eedd722ab93b832ffba08382ba82b8716ce855ff5f0a120ba092b45", "pegs": 30, "links": 24},
    "P3": {"moves": [[3, 5], [2, 10], [5, 6], [3, 12], [7, 5], [2, 14], [9, 6], [3, 16], [11, 5], [2, 18],
                     [13, 6], [3, 20], [15, 5], [21, 2], [4, 16], [20, 4], [6, 17], [21, 6], [8, 16], [20, 8],
                     [10, 17], [12, 12], [14, 10], [13, 14], [16, 9], [12, 16], [18, 10], [13, 18], [20, 11],
                     [12, 20], [15, 14], [16, 12], [17, 15], [17, 14], [19, 14], [16, 16], [21, 15], [17, 18],
                     [15, 18], [16, 20], [17, 19], [17, 22], [19, 18], [7, 13], [21, 19], [8, 15], [5, 9],
                     [7, 17], [7, 10], [8, 19], [9, 9], [7, 21], [11, 10], [19, 5], [14, 3], [18, 7], [16, 2],
                     [19, 9], [18, 3], [22, 9], [20, 2], [21, 11], [4, 21], [22, 13], [6, 20], [5, 1], [8, 21],
                     [6, 3], [10, 20], [7, 1]],
           "digest": "5d67975c99a26f0b19e677f500c7deea145782f3af5b76ebbf9497f9dd5ed160", "pegs": 70, "links": 54},
}

# Bucket -> what its exclusive time covers. "expansion_other" is _expand_batch minus its wrapped children:
# array packing, prior construction, the prior shuffle and other bookkeeping, not input building alone.
BUCKETS = {
    "selection": "MCTS._select_child (PUCT over a node's children)",
    "board": "TwixtState.apply_move / legal_moves / is_terminal / winner",
    "input": "LocalGPUEvaluator.build_input_tensor",
    "infer": "LocalGPUEvaluator.infer, combined (forward, GPU sync, softmax, conversion)",
    "expansion_other": "MCTS._expand_batch exclusive: packing, priors, shuffle, bookkeeping",
    "backup": "MCTS._backup",
}


class Refused(Exception):
    pass


def in_any(seed, intervals):
    return any(lo <= seed < hi for lo, hi in intervals)


def seeds():
    """One seed per fixture, the first three of the test-only namespace; membership verified, not assumed."""
    lo, hi = REF.TEST_ONLY_SEED_INTERVALS[0]
    out = {name: lo + i for i, name in enumerate(FIXTURES)}
    for name, s in out.items():
        if not in_any(s, REF.TEST_ONLY_SEED_INTERVALS):
            raise Refused(f"{name}: seed {s} is not in TEST_ONLY_SEED_INTERVALS")
        for reg in ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS"):
            if in_any(s, getattr(REF, reg)):
                raise Refused(f"{name}: seed {s} is also in {reg}")
    return out


def build(name):
    f = FIXTURES[name]
    s = B.replay_state([])
    for m in f["moves"]:
        if tuple(m) not in s.legal_moves():
            raise Refused(f"{name}: move {m} is illegal for {s.to_move} at ply {s.ply}")
        s = s.apply_move(tuple(m))
    got = {"digest": B.state_digest(s), "pegs": len(s.pegs), "links": len(s.bridges)}
    for k, v in got.items():
        if v != f[k]:
            raise Refused(f"{name}: {k} is {v}, frozen {f[k]}")
    if s.is_terminal() or s.winner() is not None:
        raise Refused(f"{name}: the position is terminal")
    if (s.board_size, s.active_size, s.max_plies_limit) != (24, 24, 280):
        raise Refused(f"{name}: not the 24x24, 280-ply protocol")
    return s


def preflight():
    """Everything checkable without MLX or the model."""
    if (B.PROTOCOL["board_size"], B.PROTOCOL["max_moves"], B.PROTOCOL["mcts_sims"]) != (24, 280, SIMS):
        raise Refused(f"PROTOCOL is not 24x24 / 280 plies / {SIMS} simulations")
    states = {name: build(name) for name in FIXTURES}
    sd = seeds()
    sha1 = hashlib.sha1(CKPT.read_bytes()).hexdigest()
    if sha1 != SHA1:
        raise Refused(f"checkpoint SHA-1 {sha1} is not the pinned calib020_0001")
    return states, sd, sha1


class Timers:
    """Exclusive time per bucket: a wrapped call's elapsed time less its wrapped children's. Active only
    between start() and stop(), so fixture building and the warm-up are not counted."""

    def __init__(self):
        self.on, self.stack, self.infer_calls = False, [], []
        self.ns = {b: 0 for b in BUCKETS}
        self.calls = {b: 0 for b in BUCKETS}

    def wrap(self, owner, attr, bucket, infer=False):
        fn = getattr(owner, attr)

        def w(*a, **k):
            if not self.on:
                return fn(*a, **k)
            self.stack.append(0)
            t = time.perf_counter_ns()
            try:
                return fn(*a, **k)
            finally:
                el = time.perf_counter_ns() - t
                child = self.stack.pop()
                self.ns[bucket] += el - child
                self.calls[bucket] += 1
                if self.stack:
                    self.stack[-1] += el
                if infer:                                  # args: self, boards, rows, cols, mask, active_size
                    self.infer_calls.append((list(a[1].shape[:1]) + list(a[2].shape[1:]), el))

        setattr(owner, attr, w)

    def start(self):
        self.__init__()
        self.on = True

    def stop(self):
        self.on = False


def overhead_ns_per_call():
    """The wrapper's own cost per call, on a no-op method. An ESTIMATE for this process."""
    class Noop:
        def f(self):
            return None

    n, x = 200_000, Noop()
    t = time.perf_counter_ns()
    for _ in range(n):
        x.f()
    bare = time.perf_counter_ns() - t
    tm = Timers()
    tm.ns["probe"], tm.calls["probe"] = 0, 0
    tm.wrap(Noop, "f", "probe")
    tm.on = True
    t = time.perf_counter_ns()
    for _ in range(n):
        x.f()
    return max(0.0, ((time.perf_counter_ns() - t) - bare) / n)


def warmup(ev, state):
    """ONE infer on P1's root, packed as _expand_batch packs a single leaf. No MCTS object, no random state."""
    import numpy as np

    legal = state.legal_moves()
    board = np.transpose(ev.build_input_tensor(state), (1, 2, 0))[None].astype(np.float32)
    rows = np.array([[r for r, _ in legal]], dtype=np.int32)
    cols = np.array([[c for _, c in legal]], dtype=np.int32)
    mask = np.ones((1, len(legal)), dtype=np.float32)
    t = time.perf_counter()
    ev.infer(board, rows, cols, mask, state.active_size)
    return time.perf_counter() - t


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def search(ev, tm, name, state, seed):
    mcts = MCTS(ev, cfg_from(EvalConfig(**B.PROTOCOL), SIMS), random.Random(seed))
    tm.start()
    w0, c0 = time.perf_counter_ns(), time.process_time_ns()
    counts, value, root = mcts.search_with_root(state, add_noise=False)
    wall, cpu = time.perf_counter_ns() - w0, time.process_time_ns() - c0
    tm.stop()
    legal = state.legal_moves()
    if root.visit_count != SIMS or sum(counts.values()) != SIMS:
        raise RuntimeError(f"{name}: {root.visit_count} root visits / {sum(counts.values())} counted, "
                           f"not {SIMS} completed simulations")
    if sorted(counts) != sorted(legal):
        raise RuntimeError(f"{name}: the visit map is not over the full legal list")
    full = [[r, c, counts[(r, c)]] for r, c in sorted(legal)]
    top = max(counts.values())
    ties = sorted([r, c] for (r, c), v in counts.items() if v == top)
    attributed = sum(tm.ns.values())
    infer_ns = [ns for _, ns in tm.infer_calls]
    return {
        "seed": seed, "wall_ms": wall / 1e6, "process_cpu_ms": cpu / 1e6,
        "buckets_ms": {b: tm.ns[b] / 1e6 for b in BUCKETS}, "calls": dict(tm.calls),
        "unattributed_ms": (wall - attributed) / 1e6,
        "infer": {"calls": len(infer_ns), "median_ms": pct(infer_ns, 0.5) / 1e6, "p95_ms": pct(infer_ns, 0.95) / 1e6,
                  "max_ms": max(infer_ns) / 1e6,
                  "distinct_shapes": len({tuple(s) for s, _ in tm.infer_calls}),
                  "batch_sizes": sorted({s[0] for s, _ in tm.infer_calls})},
        "baseline": {"visit_counts": full, "visit_counts_sha256": B._sha(full), "root_value": repr(value),
                     "most_visited": ties[0], "most_visited_rule": "max visits; ties -> smallest (row, col)",
                     "max_visit_ties": ties,
                     "rng_state_sha256": hashlib.sha256(repr(mcts.rng.getstate()).encode()).hexdigest()},
    }


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def profile(rec, states, sd):
    t = time.perf_counter()
    import mlx.core
    import numpy

    from scripts.GPU.alphazero import local_evaluator as LE
    from scripts.GPU.alphazero.eval_runner import _default_evaluator_factory
    rec["phases_s"] = {"imports": time.perf_counter() - t}
    rec["environment"].update({"mlx": mlx.core.__version__, "numpy": numpy.__version__})
    t = time.perf_counter()
    ev = _default_evaluator_factory(str(CKPT))
    rec["phases_s"]["load_and_construct"] = time.perf_counter() - t
    rec["phases_s"]["first_call_warmup_infer"] = warmup(ev, states["P1"])

    tm = Timers()
    tm.wrap(MCTS, "_select_child", "selection")
    for attr in ("apply_move", "legal_moves", "is_terminal", "winner"):
        tm.wrap(TwixtState, attr, "board")
    tm.wrap(LE.LocalGPUEvaluator, "build_input_tensor", "input")
    tm.wrap(LE.LocalGPUEvaluator, "infer", "infer", infer=True)
    tm.wrap(MCTS, "_expand_batch", "expansion_other")
    tm.wrap(MCTS, "_backup", "backup")

    rec["positions"] = {name: search(ev, tm, name, states[name], sd[name]) for name in FIXTURES}
    per = rec["positions"].values()
    wall = sum(p["wall_ms"] for p in per)
    tot = {b: sum(p["buckets_ms"][b] for p in per) for b in BUCKETS}
    tot["unattributed"] = sum(p["unattributed_ms"] for p in per)
    calls = sum(sum(p["calls"].values()) for p in per)
    oh = overhead_ns_per_call()
    rec["overall"] = {"wall_ms": wall, "process_cpu_ms": sum(p["process_cpu_ms"] for p in per),
                      "simulations": SIMS * len(FIXTURES), "ms_per_simulation": wall / (SIMS * len(FIXTURES)),
                      "buckets_ms": tot, "share": {b: v / wall for b, v in tot.items()},
                      "wrapped_calls": calls,
                      "instrumentation_estimate": {"ns_per_call_noop_calibration": oh,
                                                   "estimated_ms": oh * calls / 1e6,
                                                   "note": "an estimate from a no-op calibration, not a measurement of the run"}}


def main(argv):
    dry = "--dry" in argv
    out = None if dry else (pathlib.Path(argv[argv.index("--out") + 1]) if "--out" in argv else None)
    if not dry:
        if out is None or not out.is_dir() or (out / "record.json").exists():
            print("REFUSED: --out must name an existing directory without record.json (or pass --dry)",
                  file=sys.stderr)
            return EXIT["REFUSED"]
    rec = {"kind": "search-profile/1", "script": "docs/superpowers/evidence/search-profile/01_profile.py",
           "buckets": BUCKETS, "strength_verdict": None}
    t0 = time.perf_counter()
    try:
        states, sd, sha1 = preflight()
        if dry:
            for name, s in states.items():
                print(f"{name}: plies {s.ply} pegs {len(s.pegs)} links {len(s.bridges)} to_move {s.to_move} "
                      f"legal {len(s.legal_moves())} digest {B.state_digest(s)} seed {sd[name]} (test-only)")
            print(f"checkpoint SHA-1 {sha1} verified; DRY OK in {time.perf_counter() - t0:.2f} s; nothing written")
            return EXIT["COMPLETED"]
        rec.update({"head": git("rev-parse", "HEAD"), "dirty_paths": git("status", "--porcelain").splitlines(),
                    "checkpoint": {"path": str(CKPT), "sha1_verified": sha1},
                    "protocol": dict(B.PROTOCOL), "sims_per_position": SIMS,
                    "seeds": {"values": sd, "rule": "first three of TEST_ONLY_SEED_INTERVALS[0], membership verified"},
                    "fixtures": FIXTURES, "loadavg_before": os.getloadavg(),
                    "environment": {"python": sys.version, "platform": platform.platform(),
                                    "executable": sys.executable}})
        profile(rec, states, sd)
        rec["loadavg_after"] = os.getloadavg()
        rec["verdict"], code = "COMPLETED", EXIT["COMPLETED"]
    except Refused as e:
        rec["verdict"], rec["error"], code = "REFUSED", str(e), EXIT["REFUSED"]
        print(f"REFUSED: {e}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 -- the record boundary covers every failure
        rec["verdict"], rec["error"], code = "STOP_ERROR", traceback.format_exc(), EXIT["STOP_ERROR"]
        print(f"STOP_ERROR: {e!r}", file=sys.stderr)
    rec["total_seconds"] = time.perf_counter() - t0
    if not dry:
        with open(out / "record.json", "x") as fh:
            fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        if code == 0:
            o = rec["overall"]
            print(f"COMPLETED: {o['simulations']} simulations, {o['wall_ms']:.0f} ms search wall, "
                  f"{o['ms_per_simulation']:.2f} ms/simulation")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
