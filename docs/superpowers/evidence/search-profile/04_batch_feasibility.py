"""BATCH FEASIBILITY -- does the existing batched search at 1,600 simulations fit within the time sequential search
takes at 400? (prepared 2026-10-15; the run needs approval)

Production files are unchanged. Two arms on the same weights, rules, fixtures and seeds:

    S400   MCTS.search_with_root(state, add_noise=False), cfg_from(EvalConfig(**PROTOCOL), 400) -- the play path
    B1600  MCTS.search_from_root(MCTSNode(state), add_noise=False, ply=state.ply),
           cfg_from(EvalConfig(**PROTOCOL), 1600) -- the existing batched path, frozen settings asserted below

The run reuses the frozen setup of 01_profile.py and the checks of 02_sync_compare.py: the same three fixtures,
the pinned incumbent and each fixture's test-only seed (a fresh random.Random(seed) per search, both arms).
It goes as follows:
* one network load through eval_runner's factory; both arms share that evaluator;
* three explicit warm-up infers, one per fixture root;
* per fixture: S400, then B1600. That is 6 searches, 6,000 simulations.

Checks (any failure is a STOP, with details in the record):
* S400 must EXACTLY match the committed baseline (run/record.json), as in 02_sync_compare.py;
* B1600: root visits and summed visit counts are 1,600, over exactly the legal list; finite root value;
  evaluated rows = 1 root + expanded leaf nodes; terminal backups + waiter backups = 1,600 (both counted here,
  independently of the MCTS counters, which must agree);
* every evaluator output in the run is finite.
B1600's choices may differ from S400's; its most-visited move is recorded as description only.

Instrumentation lives in this process only: a wrapper on the evaluator instance (per-call shape, time, finiteness)
and, for B1600 only, counting wrappers on that MCTS instance's _backup and _flush_pending_batch. Waiter and
unique-leaf counters describe the batched path; for S400 they are recorded as not applicable. Expanded leaf nodes
are tree nodes, not necessarily distinct board positions.

Criterion, frozen in advance: FEASIBLE if B1600's combined search time over the three fixtures is at most S400's
(ratio <= 1.00), else NOT_FEASIBLE. The times are FIRST-SEEN-SHAPE-INCLUSIVE: shapes are tracked across warm-ups
and all searches; a first-seen shape does not prove recompilation, and these timings do not establish
steady-state gameplay latency. One search per arm per fixture: a practical feasibility check, not a strength
result.

  --out DIR   DIR must exist (the launch creates it) and hold no record.json; writes DIR/record.json.

Exit 0 COMPLETED (whatever the verdict) / 2 STOP / 3 STOP_ERROR / 4 REFUSED.
"""
import collections
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
_spec = importlib.util.spec_from_file_location("search_profile_02", HERE / "02_sync_compare.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)                      # definitions only; it loads 01_profile.py as C.P
P = C.P

from scripts.GPU.alphazero import search_distillation_stage_b as B  # noqa: E402
from scripts.GPU.alphazero.eval_runner import EvalConfig, cfg_from  # noqa: E402
from scripts.GPU.alphazero.mcts import MCTS, MCTSNode  # noqa: E402

SIMS_B = 1600
CRITERION = 1.00
EXIT = {"COMPLETED": 0, "STOP": 2, "STOP_ERROR": 3, "REFUSED": 4}
# n_simulations, eval_batch_size, pending_virtual_visits, stall_flush_sims, closeout forcing, c_puct, FPU mass
FROZEN_B = (SIMS_B, 14, 8, 48, False, 1.5, None)
BATCH_COUNTERS = ("_nn_batches", "_nn_call_count", "_unique_leaves_expanded", "_total_backups",
                  "_total_waiters_backed_up", "_max_waiters_on_any_leaf", "_flush_full", "_flush_stall", "_flush_tail")


def batched_cfg():
    cfg = cfg_from(EvalConfig(**B.PROTOCOL), SIMS_B)
    got = (cfg.n_simulations, cfg.eval_batch_size, cfg.pending_virtual_visits, cfg.stall_flush_sims,
           cfg.closeout_td1_visit_forcing_enabled, cfg.c_puct, cfg.fpu_policy_mass_reduction)
    if got != FROZEN_B:
        raise P.Refused(f"batched settings are {got}, frozen {FROZEN_B}")
    return cfg


class Calls:
    """Wraps ev.infer on the instance: per call, its (batch, padded moves) shape, whether that shape was first seen
    in this process (across warm-ups and searches), elapsed ns and output finiteness."""

    def __init__(self, ev):
        import numpy as np

        self.seen, self.log, real = set(), [], ev.infer

        def infer(boards, rows, cols, mask, active_size):
            t = time.perf_counter_ns()
            priors, values = real(boards, rows, cols, mask, active_size)
            el = time.perf_counter_ns() - t
            shape = (int(boards.shape[0]), int(rows.shape[1]))
            first = shape not in self.seen
            self.seen.add(shape)
            self.log.append({"shape": shape, "first_seen": first, "ns": el,
                             "finite": bool(np.isfinite(priors).all() and np.isfinite(values).all())})
            return priors, values

        ev.infer = infer

    def take(self, where):
        log, self.log = self.log, []
        bad = [i for i, c in enumerate(log) if not c["finite"]]
        if bad:
            raise C.Stop(f"{where}: non-finite evaluator output", {"where": where, "fields": ["finite_outputs"],
                                                                    "call_indices": bad})
        return log


def describe(log):
    sizes = collections.Counter(c["shape"][0] for c in log)
    return {"infer_calls": len(log), "evaluated_rows": sum(c["shape"][0] for c in log),
            "batch_size_histogram": {str(k): sizes[k] for k in sorted(sizes)},
            "infer_ms": sum(c["ns"] for c in log) / 1e6,
            "distinct_shapes": len({c["shape"] for c in log}),
            "first_seen_shape_calls": sum(c["first_seen"] for c in log),
            "first_seen_shape_infer_ms": sum(c["ns"] for c in log if c["first_seen"]) / 1e6,
            "per_batch_size_median_ms": {str(k): P.pct([c["ns"] for c in log if c["shape"][0] == k], 0.5) / 1e6
                                         for k in sorted(sizes)}}


def batched(ev, cfg, name, state, seed, s400_top):
    mcts = MCTS(ev, cfg, random.Random(seed))
    n = {"terminal_backups": 0, "waiter_backups": 0, "in_flush": False}
    real_backup, real_flush = mcts._backup, mcts._flush_pending_batch

    def backup(path, value):
        n["waiter_backups" if n["in_flush"] else "terminal_backups"] += 1
        return real_backup(path, value)

    def flush(nodes, waiters):
        n["in_flush"] = True
        try:
            return real_flush(nodes, waiters)
        finally:
            n["in_flush"] = False

    mcts._backup, mcts._flush_pending_batch = backup, flush
    w0, c0 = time.perf_counter_ns(), time.process_time_ns()
    counts, value, root = mcts.search_from_root(MCTSNode(state=state), add_noise=False, ply=state.ply)
    wall, cpu = time.perf_counter_ns() - w0, time.process_time_ns() - c0
    legal = state.legal_moves()
    ctr = {k: getattr(mcts, k) for k in BATCH_COUNTERS}
    checks = {"root_visits": root.visit_count == SIMS_B, "summed_visits": sum(counts.values()) == SIMS_B,
              "legal_support": sorted(counts) == sorted(legal), "finite_root_value": math.isfinite(value),
              "backups_sum": n["terminal_backups"] + n["waiter_backups"] == SIMS_B,
              "waiter_counter_agrees": n["waiter_backups"] == ctr["_total_waiters_backed_up"],
              "backup_counter_agrees": n["terminal_backups"] + n["waiter_backups"] == ctr["_total_backups"]}
    full = [[r, c, counts[(r, c)]] for r, c in sorted(legal)]
    top = max(counts.values())
    most = sorted([r, c] for (r, c), v in counts.items() if v == top)[0]
    out = {"wall_ms": wall / 1e6, "process_cpu_ms": cpu / 1e6, "root_value": repr(value),
           "visit_counts_sha256": B._sha(full), "most_visited": most, "same_most_visited_as_s400": most == s400_top,
           "root_visits": root.visit_count, "backups": {k: n[k] for k in ("terminal_backups", "waiter_backups")},
           "counters": ctr, "checks": checks}
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise C.Stop(f"{name}/B1600: failed {bad}", {"fixture": name, "arm": "B1600", "fields": bad, **out})
    return out


def run(rec, states, sd, base):
    t = time.perf_counter()
    from scripts.GPU.alphazero import local_evaluator as LE
    from scripts.GPU.alphazero.eval_runner import _default_evaluator_factory
    rec["phases_s"] = {"imports": time.perf_counter() - t}
    cfg = batched_cfg()
    t = time.perf_counter()
    ev = _default_evaluator_factory(str(P.CKPT))
    rec["phases_s"]["factory"] = time.perf_counter() - t
    if type(ev) is not LE.LocalGPUEvaluator or ev._use_compile is not True:
        raise P.Refused("the factory did not return the production compile=True evaluator")
    calls = Calls(ev)

    rec["warmups"] = []
    for n in P.FIXTURES:
        seconds = P.warmup(ev, states[n])
        rec["warmups"].append({"fixture": n, "seconds": seconds, **describe(calls.take(f"warmup {n}"))})
    rec["searches"] = []
    for n in P.FIXTURES:
        want = base["positions"][n]["baseline"]
        r = C.search(ev, n, "S400", states[n], sd[n], want)
        rec["searches"].append({"fixture": n, "arm": "S400", "simulations": P.SIMS, "matched_baseline": True, **r,
                                **describe(calls.take(f"{n}/S400")),
                                "batched_path_counters": "not applicable (sequential path)"})
        r = batched(ev, cfg, n, states[n], sd[n], want["most_visited"])
        rec["searches"].append({"fixture": n, "arm": "B1600", "simulations": SIMS_B, **r,
                                **describe(calls.take(f"{n}/B1600"))})
        b = rec["searches"][-1]
        if b["evaluated_rows"] != 1 + b["counters"]["_unique_leaves_expanded"]:
            raise C.Stop(f"{n}/B1600: evaluated rows are not 1 root + expanded leaf nodes",
                         {"fixture": n, "arm": "B1600", "fields": ["evaluated_rows"],
                          "evaluated_rows": b["evaluated_rows"],
                          "expanded_leaf_nodes": b["counters"]["_unique_leaves_expanded"]})

    def total(arm, fixture=None):
        return sum(s["wall_ms"] for s in rec["searches"] if s["arm"] == arm and fixture in (None, s["fixture"]))

    s, b = total("S400"), total("B1600")
    rec["combined"] = {"timing": "first-seen-shape-inclusive", "s400_ms": s, "b1600_ms": b, "ratio": b / s,
                       "criterion_max_ratio": CRITERION, "simulations": sum(x["simulations"] for x in rec["searches"]),
                       "per_fixture_ratio": {n: total("B1600", n) / total("S400", n) for n in P.FIXTURES},
                       "first_seen_shape_infer_ms": {a: sum(x["first_seen_shape_infer_ms"] for x in rec["searches"]
                                                            if x["arm"] == a) for a in ("S400", "B1600")}}
    rec["feasibility"] = "FEASIBLE" if b / s <= CRITERION else "NOT_FEASIBLE"


def main(argv):
    out = pathlib.Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    if out is None or not out.is_dir() or (out / "record.json").exists():
        print("REFUSED: --out must name an existing directory without record.json", file=sys.stderr)
        return EXIT["REFUSED"]
    rec = {"kind": "search-batch-feasibility/1",
           "script": "docs/superpowers/evidence/search-profile/04_batch_feasibility.py",
           "arms": {"S400": "search_with_root, 400", "B1600": "search_from_root, 1600, frozen batch settings"},
           "frozen_batched_settings": dict(zip(("n_simulations", "eval_batch_size", "pending_virtual_visits",
                                                "stall_flush_sims", "closeout_td1_visit_forcing_enabled", "c_puct",
                                                "fpu_policy_mass_reduction"), FROZEN_B)),
           "strength_verdict": None,
           "limitation": "first-seen-shape-inclusive times; a first-seen shape does not prove recompilation; "
                         "not steady-state gameplay latency; one search per arm per fixture"}
    t0 = time.perf_counter()
    try:
        states, sd, sha1 = P.preflight()           # fixtures, test-only seeds and checkpoint hash, before MLX
        base = C.load_baseline()
        if base["seeds"]["values"] != sd:
            raise P.Refused("the baseline seeds are not this run's seeds")
        import mlx.core
        import numpy
        rec.update({"head": P.git("rev-parse", "HEAD"), "dirty_paths": P.git("status", "--porcelain").splitlines(),
                    "checkpoint": {"path": str(P.CKPT), "sha1_verified": sha1},
                    "baseline": {"path": str(C.BASELINE.relative_to(P.ROOT)), "sha256": C.BASELINE_SHA256},
                    "seeds": sd, "loadavg_before": os.getloadavg(),
                    "environment": {"python": sys.version, "platform": platform.platform(),
                                    "executable": sys.executable, "mlx": mlx.core.__version__,
                                    "numpy": numpy.__version__}})
        run(rec, states, sd, base)
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
        print(f"COMPLETED: {m['simulations']} simulations, S400 matched the baseline 3/3, B1600 checks passed 3/3; "
              f"S400 {m['s400_ms']:.0f} ms, B1600 {m['b1600_ms']:.0f} ms, ratio {m['ratio']:.3f} "
              f"(first-seen-shape-inclusive) -> {rec['feasibility']}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
