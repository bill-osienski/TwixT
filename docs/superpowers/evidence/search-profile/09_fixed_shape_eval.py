"""FIXED-SHAPE BATCHING UNDER EVALUATION MODE -- does fixed-shape B1600 fit within S400's time now that the factory
puts the network in evaluation mode (c8cb2aa6)? A speed screen only. (2026-10-15)

A thin adaptation of 05_fixed_shape.py (which stays unrun): the same fixed-shape evaluator (every call padded to
14 boards x 528 move slots, production infer on the padded arrays, only the real rows and slots returned), but
with no training-mode reproduction targets. The historical baselines were measured in training mode, so S400 is
measured fresh here and checked for accounting only.

Reuses 04_batch_feasibility.py (and through it 02 and 01): the three frozen fixtures, the pinned incumbent, the
test-only seeds, the frozen batch settings (asserted), the evaluator-call wrapper and B1600's accounting checks.
* one load through eval_runner's factory; the network must report evaluation mode, every BatchNorm included;
  the fixed-shape evaluator shares that network;
* six warm-up infers, fixed in advance and recorded apart: one per fixture root on the production evaluator,
  then one per fixture root on the fixed-shape evaluator;
* per fixture: S400 (production evaluator, search_with_root), then B1600 (fixed-shape evaluator,
  search_from_root). 6 searches, 6,000 simulations.

Checks (any failure is a STOP): evaluation mode; every evaluator output finite; S400 root and summed visits 400
over exactly the legal list with a finite root value; B1600 04's accounting checks plus evaluated rows = 1 root +
expanded leaf nodes. Criterion, unchanged: FEASIBLE if B1600's combined search time is at most S400's (ratio <=
1.00), else NOT_FEASIBLE. Every timed call is kept; nothing is subtracted. No strength claim.

  --out DIR   DIR must exist (the launch creates it) and hold no record.json; writes DIR/record.json.

Exit 0 COMPLETED (whatever the verdict) / 2 STOP / 3 STOP_ERROR / 4 REFUSED.
"""
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import platform
import random
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("search_profile_04", HERE / "04_batch_feasibility.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)                      # definitions only; it loads 02 as D.C and 01 as D.P
_spec6 = importlib.util.spec_from_file_location("search_profile_06", HERE / "06_bn_mode.py")
M = importlib.util.module_from_spec(_spec6)
_spec6.loader.exec_module(M)                     # definitions only (modes helper; reads no running statistic)
C, P = D.C, D.P

PAD_B, PAD_M = 14, 528
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}


def fixed_class(LE):
    import numpy as np

    class FixedShapeEvaluator(LE.LocalGPUEvaluator):
        def infer(self, boards, move_rows, move_cols, move_mask, active_size):
            b, m = boards.shape[0], move_rows.shape[1]
            if b > PAD_B or m > PAD_M:
                raise RuntimeError(f"call of {b} boards x {m} moves exceeds the fixed {PAD_B} x {PAD_M}")
            pb = np.zeros((PAD_B,) + boards.shape[1:], boards.dtype)
            pr = np.zeros((PAD_B, PAD_M), move_rows.dtype)
            pc = np.zeros((PAD_B, PAD_M), move_cols.dtype)
            pm = np.zeros((PAD_B, PAD_M), move_mask.dtype)
            pb[:b], pr[:b, :m], pc[:b, :m], pm[:b, :m] = boards, move_rows, move_cols, move_mask
            priors, values = super().infer(pb, pr, pc, pm, active_size)
            return np.ascontiguousarray(priors[:b, :m]), np.ascontiguousarray(values[:b])

    return FixedShapeEvaluator


def sequential(ev, name, state, seed):
    mcts = D.MCTS(ev, D.cfg_from(D.EvalConfig(**D.B.PROTOCOL), P.SIMS), random.Random(seed))
    w0, c0 = time.perf_counter_ns(), time.process_time_ns()
    counts, value, root = mcts.search_with_root(state, add_noise=False)
    wall, cpu = time.perf_counter_ns() - w0, time.process_time_ns() - c0
    legal = state.legal_moves()
    checks = {"root_visits": root.visit_count == P.SIMS, "summed_visits": sum(counts.values()) == P.SIMS,
              "legal_support": sorted(counts) == sorted(legal), "finite_root_value": math.isfinite(value)}
    full = [[r, c, counts[(r, c)]] for r, c in sorted(legal)]
    top = max(counts.values())
    out = {"wall_ms": wall / 1e6, "process_cpu_ms": cpu / 1e6, "root_value": repr(value),
           "visit_counts_sha256": D.B._sha(full),
           "most_visited": sorted([r, c] for (r, c), v in counts.items() if v == top)[0],
           "rng_state_sha256": hashlib.sha256(repr(mcts.rng.getstate()).encode()).hexdigest(),
           "root_visits": root.visit_count, "checks": checks}
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise C.Stop(f"{name}/S400: failed {bad}", {"fixture": name, "arm": "S400", "fields": bad, **out})
    return out


def calls_kept(log):
    """Every timed call: [boards, move slots] as the caller passed them, elapsed ms, first-seen (caller's shape)."""
    return [[c["shape"][0], c["shape"][1], c["ns"] / 1e6, c["first_seen"]] for c in log]


def run(rec, states, sd):
    t = time.perf_counter()
    from scripts.GPU.alphazero import local_evaluator as LE
    from scripts.GPU.alphazero.eval_runner import _default_evaluator_factory
    rec["phases_s"] = {"imports": time.perf_counter() - t}
    cfg = D.batched_cfg()
    if cfg.eval_batch_size != PAD_B:
        raise P.Refused(f"eval_batch_size {cfg.eval_batch_size} is not the padded batch {PAD_B}")
    t = time.perf_counter()
    ev = _default_evaluator_factory(str(P.CKPT))
    rec["phases_s"]["factory"] = time.perf_counter() - t
    if type(ev) is not LE.LocalGPUEvaluator or ev._use_compile is not True:
        raise P.Refused("the factory did not return the production compile=True evaluator")
    rec["modes_at_load"] = m = M.modes(ev.network)
    if m["network_training"] or not m["batchnorm"] or any(x["training"] for x in m["batchnorm"]):
        raise C.Stop("the factory did not return the network in evaluation mode",
                     {"fields": ["evaluation_mode"], "modes": m})
    fx = fixed_class(LE)(ev.network, compile=True)
    calls = {"production": D.Calls(ev), "fixed_shape": D.Calls(fx)}

    rec["warmups"] = []
    for arm, e in (("production", ev), ("fixed_shape", fx)):
        for n in P.FIXTURES:
            seconds = P.warmup(e, states[n])
            log = calls[arm].take(f"warmup {arm} {n}")
            rec["warmups"].append({"evaluator": arm, "fixture": n, "seconds": seconds, "calls": calls_kept(log)})
    rec["searches"] = []
    for n in P.FIXTURES:
        r = sequential(ev, n, states[n], sd[n])
        log = calls["production"].take(f"{n}/S400")
        rec["searches"].append({"fixture": n, "arm": "S400", "evaluator": "production", "simulations": P.SIMS,
                                **r, **D.describe(log), "calls": calls_kept(log)})
        r = D.batched(fx, cfg, n, states[n], sd[n], rec["searches"][-1]["most_visited"])
        log = calls["fixed_shape"].take(f"{n}/B1600")
        s = {"fixture": n, "arm": "B1600", "evaluator": "fixed_shape", "simulations": D.SIMS_B, **r,
             **D.describe(log), "calls": calls_kept(log)}
        rec["searches"].append(s)
        if s["evaluated_rows"] != 1 + s["counters"]["_unique_leaves_expanded"]:
            raise C.Stop(f"{n}/B1600: evaluated rows are not 1 root + expanded leaf nodes",
                         {"fixture": n, "arm": "B1600", "fields": ["evaluated_rows"],
                          "evaluated_rows": s["evaluated_rows"],
                          "expanded_leaf_nodes": s["counters"]["_unique_leaves_expanded"]})
    if M.modes(ev.network)["modules_in_training_mode"]:
        raise C.Stop("a module left evaluation mode during the run", {"fields": ["evaluation_mode"],
                                                                       "modes": M.modes(ev.network)})

    def total(arm, fixture=None):
        return sum(x["wall_ms"] for x in rec["searches"] if x["arm"] == arm and fixture in (None, x["fixture"]))

    s400, b = total("S400"), total("B1600")
    rec["combined"] = {"timing": "all timed calls kept, nothing subtracted", "s400_ms": s400, "b1600_fixed_ms": b,
                       "ratio": b / s400, "criterion_max_ratio": D.CRITERION,
                       "simulations": sum(x["simulations"] for x in rec["searches"]),
                       "per_fixture_ratio": {n: total("B1600", n) / total("S400", n) for n in P.FIXTURES}}
    rec["feasibility"] = "FEASIBLE" if b / s400 <= D.CRITERION else "NOT_FEASIBLE"


def main(argv):
    out = pathlib.Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists():
        print("REFUSED: --out must name an existing directory without record.json", file=sys.stderr)
        return EXIT["REFUSED"]
    rec = {"kind": "search-fixed-shape-eval/1",
           "script": "docs/superpowers/evidence/search-profile/09_fixed_shape_eval.py",
           "fixed_shape": {"boards": PAD_B, "move_slots": PAD_M}, "inference_mode": "evaluation (factory)",
           "strength_verdict": None,
           "limitation": "speed screen only; one search per arm per fixture; first_seen refers to the caller's "
                         "unpadded shape, the fixed-shape evaluator always runs 14 x 528; not steady-state latency"}
    t0 = time.perf_counter()
    try:
        states, sd, sha1 = P.preflight()           # fixtures, test-only seeds and checkpoint hash, before MLX
        import mlx.core
        import numpy
        rec.update({"head": P.git("rev-parse", "HEAD"), "dirty_paths": P.git("status", "--porcelain").splitlines(),
                    "checkpoint": {"path": str(P.CKPT), "sha1_verified": sha1},
                    "seeds": sd, "loadavg_before": os.getloadavg(),
                    "environment": {"python": sys.version, "platform": platform.platform(),
                                    "executable": sys.executable, "mlx": mlx.core.__version__,
                                    "numpy": numpy.__version__}})
        run(rec, states, sd)
        rec["loadavg_after"] = os.getloadavg()
        rec["verdict"], code = "COMPLETED", EXIT["COMPLETED"]
    except P.Refused as e:
        rec["verdict"], rec["error"], code = "REFUSED", str(e), EXIT["REFUSED"]
        print(f"REFUSED: {e}", file=sys.stderr)
    except C.Stop as e:
        rec["verdict"], rec["error"], rec["mismatch"], code = "STOP", str(e), e.details, EXIT["STOP"]
        print(f"STOP: {e}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 -- the record boundary covers every failure
        rec["verdict"], rec["error"], code = "STOP_ERROR", traceback.format_exc(), EXIT["STOP_ERROR"]
        print(f"STOP_ERROR: {e!r}", file=sys.stderr)
    rec["total_seconds"] = time.perf_counter() - t0
    with open(out / "record.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    if code == 0:
        m = rec["combined"]
        print(f"COMPLETED: {m['simulations']} simulations, evaluation mode, all checks passed; S400 "
              f"{m['s400_ms']:.0f} ms, fixed-shape B1600 {m['b1600_fixed_ms']:.0f} ms, ratio {m['ratio']:.3f} "
              f"-> {rec['feasibility']}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
