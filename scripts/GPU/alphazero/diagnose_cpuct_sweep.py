"""READ-ONLY falsification diagnostic: does changing c_puct materially reduce
the 400-sim A predrop inflation?

THE CLAIM UNDER TEST. The A gate's overvaluation is not a value-head defect: at
400 sims against ~500 legal moves, root.q_value is the unweighted mean of the
raw leaf evaluations, and 89% of the expansion frontier is single-visit
depth-2 nodes. `_select_child` scores an unvisited child `0.0 + c*p*sqrt(N)`
(non-negative for any c >= 0) and a visited child `-child.q_value + u`. Measured
on the real trees: 99.8% of depth-2 children are bad for the opponent, and in
19 of 27 branches EVERY visited reply has a negative opponent-perspective q
(median best: -0.099). So the unvisited child wins for any c >= 0, including
c = 0, and the opponent scans hundreds of fresh replies instead of revisiting.
=> c_puct cannot reach the pathology. FPU (the hardcoded 0.0) is the only knob
that can. This script tries to falsify that.

PREDICTION, REGISTERED BEFORE THE RUN:
  1. mean_black_value remains roughly +0.20 to +0.30 across c_puct values
     (nowhere near the 6400-sim reference of -0.045).
  2. gate over (>= 0.25) remains near 50%.
  3. gate severe (>= 0.50) remains near 43%.
  4. root_n_visited_children DECREASES as c_puct decreases (the root
     concentrates, as it should -- proving the knob is doing something).
  5. top_child_n_visited_children INCREASES or stays high as c_puct decreases
     (a more concentrated root funnels more sims into one child, and that child
     still expands a fresh reply nearly every simulation).
  6. FALSIFIER: if top_child_n_visited_children falls materially as c_puct
     falls, the claim is WRONG -- c_puct does reach the pathology, and an FPU
     change may be unnecessary.

Without columns 4 and 5 a null result would be ambiguous: it could not
distinguish "c_puct did nothing" from "c_puct did exactly what it should and
the pathology is one ply below where it acts."

READ-ONLY: reads one checkpoint, the A probe manifest, the replay JSONs, and the
Phase-0 concentration CSV; writes two CSVs. No MCTSConfig field is added --
c_puct already exists. No mcts.py change, no FPU, no prior pruning, no trainer,
manifest, loader, or calibration change.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import random
from pathlib import Path

from .continuation_extraction import _best_child
from .diagnose_v15_a_continuation_concentration import search_for_row
from .eval_raw_nn_position_rows import to_black
from .mcts import decode_move
from .position_probe_cases import (OVERVALUE_THRESHOLD,
                                   SEVERE_OVERVALUE_THRESHOLD,
                                   load_csv_manifest)

DEFAULT_A_MANIFEST = (
    "logs/eval/loss_analysis_v2_calib020_0001_vs_0379_black/"
    "0001_black_post_opening_top30_predrop_probe_manifest.csv")
DEFAULT_CHECKPOINT = (
    "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors")
DEFAULT_PHASE0_CSV = "logs/eval/v15prep_a_continuation_concentration.csv"
DEFAULT_OUT = "logs/eval/cpuct_check/a_predrop_cpuct_sweep_cases.csv"
DEFAULT_SUMMARY_OUT = "logs/eval/cpuct_check/a_predrop_cpuct_sweep_summary.csv"
DEFAULT_CPUCTS = "1.5,1.0,0.75,0.5,0.25"
BASELINE_CPUCT = 1.5
SIMS = 400
TOLERANCE = 1e-6

FIELDNAMES = [
    "c_puct", "case_id", "root_mcts_black_value", "gate_over_ge_0_25",
    "gate_severe_ge_0_50", "root_n_visited_children", "top_child_move",
    "top_child_visit_share", "top_child_q_black",
    "top_child_n_visited_children",
]
SUMMARY_FIELDNAMES = [
    "c_puct", "n", "mean_black_value", "over_pct_ge_0_25",
    "severe_pct_ge_0_50", "positive_pct_gt_0", "min", "max",
    "mean_root_n_visited_children", "mean_top_child_n_visited_children",
]


def gate_flags(value: float) -> tuple[bool, bool]:
    """(over, severe) using the GATE's own inclusive thresholds -- 0.25 / 0.50,
    never `> 0`. An earlier ad-hoc summarizer used `> 0` for `over`, which made
    the gate's overvalue rate look flat across sim budgets when it collapses."""
    return (value >= OVERVALUE_THRESHOLD, value >= SEVERE_OVERVALUE_THRESHOLD)


def n_visited_children(node) -> int:
    """Children with at least one visit. MCTS creates a child for every legal
    move at expansion, so `len(node.children)` counts moves, not exploration;
    this counts what the search actually sampled."""
    return sum(1 for c in node.children.values() if c.visit_count > 0)


def summarize(rows) -> dict:
    """Per-c_puct aggregate. `over`/`severe` use the gate thresholds;
    `positive` is the separate `> 0` statistic, reported alongside so the two
    can never again be confused."""
    vals = [r["root_mcts_black_value"] for r in rows]
    n = len(vals)
    return {
        "n": n,
        "mean_black_value": sum(vals) / n,
        "over_pct_ge_0_25": 100.0 * sum(
            1 for v in vals if v >= OVERVALUE_THRESHOLD) / n,
        "severe_pct_ge_0_50": 100.0 * sum(
            1 for v in vals if v >= SEVERE_OVERVALUE_THRESHOLD) / n,
        "positive_pct_gt_0": 100.0 * sum(1 for v in vals if v > 0) / n,
        "min": min(vals),
        "max": max(vals),
        "mean_root_n_visited_children": sum(
            r["root_n_visited_children"] for r in rows) / n,
        "mean_top_child_n_visited_children": sum(
            r["top_child_n_visited_children"] for r in rows) / n,
    }


def _search_fns(checkpoint: str, cpucts, eval_batch_size: int,
                stall_flush_sims: int) -> dict:
    """One evaluator, reused across all c_puct values. `cfg_from` builds the
    gate's exact MCTSConfig; `dataclasses.replace` changes c_puct and nothing
    else. The `cfg=cfg` default argument binds per value -- without it every
    entry would close over the LAST cfg and the sweep would show a spurious
    flat line."""
    from .eval_runner import EvalConfig, cfg_from, _default_evaluator_factory
    from .mcts import MCTS
    evaluator = _default_evaluator_factory(checkpoint)
    base = cfg_from(EvalConfig(mcts_sims=SIMS,
                               mcts_eval_batch_size=eval_batch_size,
                               mcts_stall_flush_sims=stall_flush_sims))
    fns = {}
    for c in cpucts:
        cfg = dataclasses.replace(base, c_puct=c)

        def fn(state, seed, cfg=cfg):
            return MCTS(evaluator, cfg, random.Random(seed)).search_with_root(
                state, add_noise=False)

        fns[c] = fn
    return fns


def _phase0_baseline(csv_path) -> dict:
    """{case_id: root_mcts_black_value} from the Phase-0 concentration CSV
    (the value is repeated on every child row; take the first per case)."""
    out = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["root_case_id"],
                           float(r["root_mcts_black_value"]))
    return out


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="READ-ONLY falsification diagnostic: does changing c_puct "
                    "materially reduce the 400-sim A predrop inflation? Sweeps "
                    "c_puct over the A probe roots and records the gate metric "
                    "plus two tree-shape counters. No mcts.py change, no FPU.")
    ap.add_argument("--a-manifest", default=DEFAULT_A_MANIFEST)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--phase0-csv", default=DEFAULT_PHASE0_CSV,
                    help="baseline for the mandatory c_puct=1.5 integrity check")
    ap.add_argument("--c-pucts", default=DEFAULT_CPUCTS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    ap.add_argument("--eval-batch-size", type=int, default=14)
    ap.add_argument("--stall-flush-sims", type=int, default=48)
    ap.add_argument("--position-probe-base-seed", type=int, default=20260616)
    ap.add_argument("--goal-line-base-seed", type=int, default=20260614)
    ap.add_argument("--limit-cases", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    cpucts = [float(c) for c in args.c_pucts.split(",") if c.strip()]
    cases = load_csv_manifest(args.a_manifest)["cases"]
    if args.limit_cases is not None:
        cases = cases[:args.limit_cases]
    baseline = _phase0_baseline(args.phase0_csv)
    search_fns = _search_fns(args.checkpoint, cpucts, args.eval_batch_size,
                             args.stall_flush_sims)

    out_rows, summary_rows = [], []
    for c in cpucts:
        rows = []
        for case in cases:
            cid = case["case_id"]
            _state, side, root_value_stm, root = search_for_row(
                case, search_fns[c],
                pos_base_seed=args.position_probe_base_seed,
                goal_base_seed=args.goal_line_base_seed)

            if root.visit_count != SIMS:
                raise SystemExit(
                    f"c_puct={c} {cid}: search ran {root.visit_count} sims, "
                    f"expected {SIMS} -- the per-value MCTSConfig did not take "
                    f"effect (late-binding closure in _search_fns?)")

            black = to_black(root_value_stm, side)

            # MANDATORY integrity check: c_puct=1.5 IS the gate's config, so it
            # must reproduce Phase 0 exactly. If not, the sweep is worthless.
            if c == BASELINE_CPUCT:
                if cid not in baseline:
                    raise SystemExit(f"{cid} missing from {args.phase0_csv}")
                if abs(black - baseline[cid]) > TOLERANCE:
                    raise SystemExit(
                        f"INTEGRITY CHECK FAILED at c_puct=1.5 on {cid}: "
                        f"fresh root_mcts_black_value={black:+.6f} != Phase-0 "
                        f"{baseline[cid]:+.6f} -- the baseline config drifted; "
                        f"DO NOT INTERPRET THE SWEEP")

            over, severe = gate_flags(black)
            top = _best_child(root)
            rows.append({
                "c_puct": c,
                "case_id": cid,
                "root_mcts_black_value": black,
                "gate_over_ge_0_25": over,
                "gate_severe_ge_0_50": severe,
                "root_n_visited_children": n_visited_children(root),
                "top_child_move": "" if top is None else
                                  "{}:{}".format(*decode_move(top.move)),
                "top_child_visit_share": "" if top is None else
                                         top.visit_count / root.visit_count,
                "top_child_q_black": "" if top is None else
                                     to_black(top.q_value, top.state.to_move),
                "top_child_n_visited_children": 0 if top is None else
                                                n_visited_children(top),
            })
        if c == BASELINE_CPUCT:
            print(f"[cpuct] integrity check PASSED at c_puct=1.5 on "
                  f"{len(rows)} cases (reproduces Phase 0 within {TOLERANCE})")
        s = summarize(rows)
        s["c_puct"] = c
        summary_rows.append(s)
        out_rows.extend(rows)
        print(f"[cpuct] c={c:<5} mean={s['mean_black_value']:+.4f} "
              f"over={s['over_pct_ge_0_25']:.1f}% "
              f"severe={s['severe_pct_ge_0_50']:.1f}% "
              f"root_children={s['mean_root_n_visited_children']:.1f} "
              f"top_child_children={s['mean_top_child_n_visited_children']:.1f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(out_rows)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nwrote {len(out_rows)} case rows -> {args.out}")
    print(f"wrote {len(summary_rows)} summary rows -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
