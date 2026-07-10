"""READ-ONLY falsification diagnostic: does making an unvisited child's
assumed value pessimistic for the mover (First-Play Urgency, i.e.
MCTSConfig.fpu_value) materially reduce the 400-sim A predrop inflation?

THE CLAIM UNDER TEST. The c_puct falsification test upheld the mechanism: the
400-sim A metric is driven by the opponent's FIRST-TOUCH evaluation of
unvisited replies. `_select_child` scores an unvisited child `q = 0.0 + u` --
an even game, in the mover's perspective, and non-negative for any c_puct >= 0
-- so the opponent scans hundreds of distinct fresh replies once each instead
of revisiting known-good ones, and root.q_value is the unweighted mean of
those single-visit blunder evaluations. Measured on the c_puct sweep:
top_child_n_visited_children and root_mcts_black_value correlate at r=+0.943,
and lowering c_puct (which funnels MORE sims into that first-touch scan)
RAISED the metric monotonically -- c_puct cannot reach the pathology; only the
hardcoded first-play value (the old q=0.0) can. fpu_value is the direct lever:
it replaces that hardcoded 0.0 in the unvisited branch of `_select_child`,
in the MOVER's perspective. This script tries to falsify that a negative
fpu_value pulls the 400-sim A metric toward the 6400-sim reference.

DECISION RULE, REGISTERED BEFORE THE RUN (design doc
docs/superpowers/specs/2026-07-10-fpu-first-play-value-sweep-design.md §4).
FPU is promising only if, relative to the fpu_value=0.0 baseline, it moves the
400-sim A metric toward the 6400-sim reference:
  1. mean_black_value falls materially below +0.2570.
  2. gate over (>= 0.25) falls materially below 50.0%.
  3. gate severe (>= 0.50) falls materially below 43.3%.
  4. top_child_n_visited_children falls materially -- the mechanism actually
     changing, not merely the metric moving for an unrelated reason.
  5. NOT a degenerate collapse: top_child_visit_share and top_child_move must
     not show the root move choice being wrecked to hide the inflation rather
     than resolve it.
A result that moves 1-3 without moving 4, or that does so only by collapsing
the root onto a single move regardless of quality, does not support adoption.

fpu_value is an ABSOLUTE CONSTANT in the mover's perspective -- adequate for
this homogeneous (contested/losing-for-black) A set only. A positive result
here does NOT pre-commit the constant form into self-play; the ship-form
(constant vs. parent-relative reduction) is a separate stage-3 decision
(design doc §2).

READ-ONLY, OPT-IN, BYTE-IDENTICAL OFF: reads one checkpoint, the A probe
manifest, the replay JSONs, and the Phase-0 concentration CSV; writes two
CSVs. `fpu_value` defaults to 0.0 on `MCTSConfig` and is read at exactly one
site in `_select_child`; this script only ever builds configs explicitly via
`dataclasses.replace`, so the default path is never touched. No mcts.py
change (Task 1 already finished it), no prior pruning, no top-k, no trainer,
network, manifest, loader, or calibration change.
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
DEFAULT_OUT = "logs/eval/fpu_check/a_predrop_fpu_sweep_cases.csv"
DEFAULT_SUMMARY_OUT = "logs/eval/fpu_check/a_predrop_fpu_sweep_summary.csv"
DEFAULT_FPUS = "0.0,-0.05,-0.10,-0.20,-0.35,-0.50"
BASELINE_FPU = 0.0
SIMS = 400
TOLERANCE = 1e-6

FIELDNAMES = [
    "fpu_value", "case_id", "root_mcts_black_value", "gate_over_ge_0_25",
    "gate_severe_ge_0_50", "root_n_visited_children", "top_child_move",
    "top_child_visit_share", "top_child_q_black",
    "top_child_n_visited_children",
]
SUMMARY_FIELDNAMES = [
    "fpu_value", "n", "mean_black_value", "over_pct_ge_0_25",
    "severe_pct_ge_0_50", "positive_pct_gt_0", "root_children_mean",
    "top_child_children_mean", "top_child_visit_share_mean", "min", "max",
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
    """Per-fpu_value aggregate. `over`/`severe` are derived from `gate_flags`
    -- the same function the per-case rows use -- so this can never drift out
    of sync with the gate's own thresholds; `positive` is the separate `> 0`
    statistic, reported alongside so the two can never again be confused.
    `top_child_visit_share_mean` is required by the FPU decision rule (a fall
    in the overvalue rate that is really a degenerate collapse onto one move
    would show up here); blank/non-numeric shares are skipped defensively,
    though every A root has a visited top child."""
    vals = [r["root_mcts_black_value"] for r in rows]
    n = len(vals)
    flags = [gate_flags(v) for v in vals]
    shares = [r["top_child_visit_share"] for r in rows
              if isinstance(r["top_child_visit_share"], (int, float))]
    return {
        "n": n,
        "mean_black_value": sum(vals) / n,
        "over_pct_ge_0_25": 100.0 * sum(1 for over, _ in flags if over) / n,
        "severe_pct_ge_0_50": 100.0 * sum(
            1 for _, severe in flags if severe) / n,
        "positive_pct_gt_0": 100.0 * sum(1 for v in vals if v > 0) / n,
        "root_children_mean": sum(
            r["root_n_visited_children"] for r in rows) / n,
        "top_child_children_mean": sum(
            r["top_child_n_visited_children"] for r in rows) / n,
        "top_child_visit_share_mean": (
            sum(shares) / len(shares) if shares else 0.0),
        "min": min(vals),
        "max": max(vals),
    }


def _make_search_fn(evaluator, cfg):
    """Own scope per config: late binding is structurally impossible here."""
    from .mcts import MCTS

    def fn(state, seed):
        return MCTS(evaluator, cfg, random.Random(seed)).search_with_root(
            state, add_noise=False)

    return fn


def _search_fns(checkpoint: str, fpus, eval_batch_size: int,
                stall_flush_sims: int) -> dict:
    """One evaluator, reused across all fpu_value values. `cfg_from` builds
    the gate's exact MCTSConfig; `dataclasses.replace` changes fpu_value and
    nothing else. Each value's closure is built by `_make_search_fn`, which
    gives it its own scope -- late binding is structurally impossible, not
    merely guarded against by a default-argument trick."""
    from .eval_runner import EvalConfig, cfg_from, _default_evaluator_factory
    evaluator = _default_evaluator_factory(checkpoint)
    base = cfg_from(EvalConfig(mcts_sims=SIMS,
                               mcts_eval_batch_size=eval_batch_size,
                               mcts_stall_flush_sims=stall_flush_sims))
    fns = {}
    for x in fpus:
        fns[x] = _make_search_fn(evaluator, dataclasses.replace(base, fpu_value=x))
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
        description="READ-ONLY falsification diagnostic: does making an "
                    "unvisited child's assumed value pessimistic for the "
                    "mover (fpu_value) materially reduce the 400-sim A "
                    "predrop inflation? Sweeps fpu_value over the A probe "
                    "roots and records the gate metric plus two tree-shape "
                    "counters. No mcts.py change (Task 1 already finished "
                    "it).")
    ap.add_argument("--a-manifest", default=DEFAULT_A_MANIFEST)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--phase0-csv", default=DEFAULT_PHASE0_CSV,
                    help="baseline for the mandatory fpu_value=0.0 integrity check")
    ap.add_argument("--fpu-values", default=DEFAULT_FPUS)
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
    fpus = [float(x) for x in args.fpu_values.split(",") if x.strip()]
    if BASELINE_FPU not in fpus:
        raise SystemExit(
            f"--fpu-values must include the baseline {BASELINE_FPU} (the "
            f"gate's own value): it is the only check that the per-value "
            f"config binding took effect and that this sweep reproduces the "
            f"gate. Got {fpus}")
    cases = load_csv_manifest(args.a_manifest)["cases"]
    if args.limit_cases is not None:
        cases = cases[:args.limit_cases]
    baseline = _phase0_baseline(args.phase0_csv)
    search_fns = _search_fns(args.checkpoint, fpus, args.eval_batch_size,
                             args.stall_flush_sims)

    out_rows, summary_rows = [], []
    for x in fpus:
        rows = []
        for case in cases:
            cid = case["case_id"]
            _state, side, root_value_stm, root = search_for_row(
                case, search_fns[x],
                pos_base_seed=args.position_probe_base_seed,
                goal_base_seed=args.goal_line_base_seed)

            if root.visit_count != SIMS:
                raise SystemExit(
                    f"fpu_value={x} {cid}: search ran {root.visit_count} "
                    f"sims, expected {SIMS} -- the MCTSConfig's "
                    f"n_simulations did not take effect")

            black = to_black(root_value_stm, side)

            # MANDATORY integrity check: fpu_value=0.0 IS the gate's config,
            # so it must reproduce Phase 0 exactly. If not, the sweep is
            # worthless. This check -- not the visit-count guard above, which
            # only catches a wrong n_simulations -- is what actually verifies
            # the per-value MCTSConfig binding took effect.
            if x == BASELINE_FPU:
                if cid not in baseline:
                    raise SystemExit(f"{cid} missing from {args.phase0_csv}")
                if abs(black - baseline[cid]) > TOLERANCE:
                    raise SystemExit(
                        f"INTEGRITY CHECK FAILED at fpu_value=0.0 on {cid}: "
                        f"fresh root_mcts_black_value={black:+.6f} != Phase-0 "
                        f"{baseline[cid]:+.6f} -- the baseline config drifted; "
                        f"DO NOT INTERPRET THE SWEEP")

            over, severe = gate_flags(black)
            top = _best_child(root)
            rows.append({
                "fpu_value": x,
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
        if x == BASELINE_FPU:
            print(f"[fpu] integrity check PASSED at fpu_value=0.0 on "
                  f"{len(rows)} cases (reproduces Phase 0 within {TOLERANCE})")
        s = summarize(rows)
        s["fpu_value"] = x
        summary_rows.append(s)
        out_rows.extend(rows)
        print(f"[fpu] fpu={x:<6} mean={s['mean_black_value']:+.4f} "
              f"over={s['over_pct_ge_0_25']:.1f}% "
              f"severe={s['severe_pct_ge_0_50']:.1f}% "
              f"root_children={s['root_children_mean']:.1f} "
              f"top_child_children={s['top_child_children_mean']:.1f} "
              f"top_child_share={s['top_child_visit_share_mean']:.3f}")

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
