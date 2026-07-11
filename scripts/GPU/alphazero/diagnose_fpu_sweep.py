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
import math
import random
from pathlib import Path
from statistics import median

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


DEFAULT_STRATA_SUMMARY_OUT = "logs/eval/fpu_check/a_predrop_fpu_sweep_summary_by_stratum.csv"
PROTOCOL_FPUS = [0.0, -0.20]   # frozen v16a validation candidate set (0.0 = control)


def manifest_is_neutral(cases) -> bool:
    """Neutral / v16a iff EVERY row has a non-empty ply_bucket; legacy iff none
    do. A mixed or empty manifest is a construction error -> raise (a first-row
    check would silently misclassify a malformed manifest)."""
    if not cases:
        raise ValueError("empty manifest: cannot determine legacy/neutral mode")
    flags = [bool(c.get("ply_bucket")) for c in cases]
    if all(flags):
        return True
    if not any(flags):
        return False
    raise ValueError(
        f"mixed manifest: {sum(flags)}/{len(flags)} rows carry ply_bucket; "
        "all-or-none is required")


def resolve_integrity_csv(integrity_csv, skip, neutral, default_csv):
    """fpu=0.0 exact-reproduction baseline, or None to skip. --skip wins; explicit
    csv next; neutral-unspecified skips; legacy-unspecified uses default_csv (so a
    bare legacy run still checks)."""
    if skip:
        return None
    if integrity_csv:
        return integrity_csv
    return None if neutral else default_csv


def _parse_fpu_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def resolve_fpu_values(fpu_values_arg, neutral, allow_non_protocol):
    """Frozen-protocol default: a neutral run with no explicit values uses
    PROTOCOL_FPUS (0.0, -0.20). Any other value set on a neutral manifest needs
    --allow-non-protocol-fpu (screening extra candidates on the holdout is tuning
    on the holdout). 0.0 must be present (delta baseline + integrity)."""
    if fpu_values_arg is None:
        values = list(PROTOCOL_FPUS) if neutral else _parse_fpu_list(DEFAULT_FPUS)
    else:
        values = _parse_fpu_list(fpu_values_arg)
    if BASELINE_FPU not in values:
        raise SystemExit(f"--fpu-values must include the baseline {BASELINE_FPU}")
    if neutral and set(values) != set(PROTOCOL_FPUS) and not allow_non_protocol:
        raise SystemExit(
            f"neutral (held-out) manifest: the frozen v16a protocol is "
            f"{PROTOCOL_FPUS} only. Screening other values on the holdout is "
            f"tuning on the holdout. Pass --allow-non-protocol-fpu to override.")
    return values


def resolve_output_paths(out, summary_out, strata_out, manifest, neutral):
    """Legacy -> the exact A defaults. Neutral -> beside the manifest, so held-out
    results never land in the selected-A directory. Explicit paths always win."""
    if not neutral:
        return (out or DEFAULT_OUT, summary_out or DEFAULT_SUMMARY_OUT,
                strata_out or DEFAULT_STRATA_SUMMARY_OUT)
    base = Path(manifest).parent
    return (out or str(base / "neutral_fpu_sweep_cases.csv"),
            summary_out or str(base / "neutral_fpu_sweep_summary.csv"),
            strata_out or str(base / "neutral_fpu_sweep_by_stratum.csv"))


GENERIC_CASE_FIELDNAMES = [
    "fpu_value", "case_id", "game_id", "ply", "ply_bucket", "side_to_move",
    "root_mcts_stm_value", "root_mcts_black_value", "top_move",
    "top_child_visit_share", "root_visit_entropy", "root_effective_children",
    "root_collapsed_ge_0_95", "root_n_visited_children",
    "top_child_n_visited_children",
    "root_value_delta_stm_vs_fpu0", "root_value_delta_black_vs_fpu0",
    "top_move_changed_vs_fpu0", "root_children_delta_vs_fpu0",
    "top_child_children_delta_vs_fpu0", "top_child_visit_share_delta_vs_fpu0",
    "root_effective_children_delta_vs_fpu0", "root_visit_entropy_delta_vs_fpu0",
    "new_collapse_vs_fpu0", "resolved_collapse_vs_fpu0",
]


def visit_entropy(visit_counts) -> float:
    """Shannon entropy (nats) of the root children's visit distribution.
    exp(entropy) = effective children; it falls as search concentrates even when
    the raw visited-children COUNT stays flat (the c_puct result). Empty -> 0."""
    total = sum(visit_counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in visit_counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def _num_delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a - b
    return ""


def enrich_with_deltas(rows):
    """Attach fpu=0.0-relative deltas per case_id, in place. Mover (_stm_) and
    black (_black_) value deltas both recorded (summaries lead with mover, which
    does not cancel across colors). Paired search-shape deltas + collapse
    accounting (new = candidate collapses where baseline did not; resolved =
    the reverse)."""
    baseline = {r["case_id"]: r for r in rows if r["fpu_value"] == BASELINE_FPU}
    for r in rows:
        b = baseline.get(r["case_id"])
        if b is None:
            raise ValueError(f"no fpu={BASELINE_FPU} baseline for {r['case_id']!r}")
        r["root_value_delta_stm_vs_fpu0"] = r["root_mcts_stm_value"] - b["root_mcts_stm_value"]
        r["root_value_delta_black_vs_fpu0"] = r["root_mcts_black_value"] - b["root_mcts_black_value"]
        r["top_move_changed_vs_fpu0"] = r["top_child_move"] != b["top_child_move"]
        r["root_children_delta_vs_fpu0"] = r["root_n_visited_children"] - b["root_n_visited_children"]
        r["top_child_children_delta_vs_fpu0"] = r["top_child_n_visited_children"] - b["top_child_n_visited_children"]
        r["top_child_visit_share_delta_vs_fpu0"] = _num_delta(
            r["top_child_visit_share"], b["top_child_visit_share"])
        r["root_effective_children_delta_vs_fpu0"] = _num_delta(
            r["root_effective_children"], b["root_effective_children"])
        r["root_visit_entropy_delta_vs_fpu0"] = r["root_visit_entropy"] - b["root_visit_entropy"]
        r["new_collapse_vs_fpu0"] = bool(r["root_collapsed_ge_0_95"]) and not bool(b["root_collapsed_ge_0_95"])
        r["resolved_collapse_vs_fpu0"] = bool(b["root_collapsed_ge_0_95"]) and not bool(r["root_collapsed_ge_0_95"])
    return rows


BUCKET_ORDER = ["opening", "early_mid", "midgame", "late"]
_METRIC_FIELDS = [
    "num_positions",
    "mean_root_value_delta_stm_vs_fpu0", "median_abs_root_value_delta_stm_vs_fpu0",
    "p90_abs_root_value_delta_stm_vs_fpu0", "p95_abs_root_value_delta_stm_vs_fpu0",
    "mean_root_value_delta_black_vs_fpu0", "top_move_flip_rate_vs_fpu0",
    "mean_root_visit_entropy_delta_vs_fpu0",
    "mean_root_effective_children_delta_vs_fpu0",
    "mean_root_children_delta_vs_fpu0", "mean_top_child_children_delta_vs_fpu0",
    "mean_top_child_children_delta_stable_top_vs_fpu0",
    "mean_top_child_visit_share_delta_vs_fpu0",
    "new_collapse_count", "new_collapse_rate", "resolved_collapse_count",
    "mean_root_effective_children", "mean_root_n_visited_children",
    "mean_top_child_n_visited_children", "mean_top_child_visit_share",
    "collapsed_ge_0_95_rate",
]
GENERIC_SUMMARY_FIELDNAMES = ["fpu_value"] + _METRIC_FIELDS
STRATA_SUMMARY_FIELDNAMES = ["fpu_value", "group_kind", "group"] + _METRIC_FIELDS


def _percentile(values, q):
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        raise ValueError("percentile of empty sequence")
    if n == 1:
        return float(xs[0])
    rank = (q / 100.0) * (n - 1)
    lo = int(rank)
    if lo + 1 >= n:
        return float(xs[-1])
    return float(xs[lo] + (rank - lo) * (xs[lo + 1] - xs[lo]))


def _mean_num(rows, key):
    vals = [r[key] for r in rows if isinstance(r[key], (int, float))
            and not isinstance(r[key], bool)]
    return sum(vals) / len(vals) if vals else 0.0


def _delta_metrics(rows):
    """Metric dict over ENRICHED rows of one group. Mover deltas are primary;
    black mean is continuity (cancels across colors). Search-shape and reply
    metrics are PAIRED (vs fpu=0.0); reply is also reported over unchanged-top
    rows. Collapse is counted as newly-introduced vs resolved."""
    n = len(rows)
    if n == 0:
        raise ValueError("no rows to summarize")
    stm = [r["root_value_delta_stm_vs_fpu0"] for r in rows]
    abs_stm = [abs(d) for d in stm]
    stable = [r for r in rows if not r["top_move_changed_vs_fpu0"]]
    stable_d = [r["top_child_children_delta_vs_fpu0"] for r in stable
                if isinstance(r["top_child_children_delta_vs_fpu0"], (int, float))]
    new_c = sum(1 for r in rows if r["new_collapse_vs_fpu0"])
    return {
        "num_positions": n,
        "mean_root_value_delta_stm_vs_fpu0": sum(stm) / n,
        "median_abs_root_value_delta_stm_vs_fpu0": median(abs_stm),
        "p90_abs_root_value_delta_stm_vs_fpu0": _percentile(abs_stm, 90),
        "p95_abs_root_value_delta_stm_vs_fpu0": _percentile(abs_stm, 95),
        "mean_root_value_delta_black_vs_fpu0":
            sum(r["root_value_delta_black_vs_fpu0"] for r in rows) / n,
        "top_move_flip_rate_vs_fpu0":
            sum(1 for r in rows if r["top_move_changed_vs_fpu0"]) / n,
        "mean_root_visit_entropy_delta_vs_fpu0": _mean_num(rows, "root_visit_entropy_delta_vs_fpu0"),
        "mean_root_effective_children_delta_vs_fpu0": _mean_num(rows, "root_effective_children_delta_vs_fpu0"),
        "mean_root_children_delta_vs_fpu0": _mean_num(rows, "root_children_delta_vs_fpu0"),
        "mean_top_child_children_delta_vs_fpu0": _mean_num(rows, "top_child_children_delta_vs_fpu0"),
        "mean_top_child_children_delta_stable_top_vs_fpu0":
            (sum(stable_d) / len(stable_d) if stable_d else ""),
        "mean_top_child_visit_share_delta_vs_fpu0": _mean_num(rows, "top_child_visit_share_delta_vs_fpu0"),
        "new_collapse_count": new_c,
        "new_collapse_rate": new_c / n,
        "resolved_collapse_count": sum(1 for r in rows if r["resolved_collapse_vs_fpu0"]),
        "mean_root_effective_children": _mean_num(rows, "root_effective_children"),
        "mean_root_n_visited_children": _mean_num(rows, "root_n_visited_children"),
        "mean_top_child_n_visited_children": _mean_num(rows, "top_child_n_visited_children"),
        "mean_top_child_visit_share": _mean_num(rows, "top_child_visit_share"),
        "collapsed_ge_0_95_rate": sum(1 for r in rows if r["root_collapsed_ge_0_95"]) / n,
    }


def _ordered(values, canonical):
    return [v for v in canonical if v in values] + sorted(v for v in values if v not in canonical)


def summarize_grouped(rows, group_kind):
    if group_kind == "all":
        groups = [("all", rows)]
    elif group_kind == "bucket":
        groups = [(b, [r for r in rows if r["ply_bucket"] == b])
                  for b in _ordered({r["ply_bucket"] for r in rows}, BUCKET_ORDER)]
    elif group_kind == "side":
        groups = [(s, [r for r in rows if r["side_to_move"] == s])
                  for s in _ordered({r["side_to_move"] for r in rows}, ["red", "black"])]
    elif group_kind == "bucket_x_side":
        groups = []
        for b in _ordered({r["ply_bucket"] for r in rows}, BUCKET_ORDER):
            for s in ["red", "black"]:
                sub = [r for r in rows if r["ply_bucket"] == b and r["side_to_move"] == s]
                if sub:
                    groups.append((f"{b}|{s}", sub))
    else:
        raise ValueError(f"unknown group_kind {group_kind!r}")
    out = []
    for gname, grows in groups:
        if not grows:
            continue
        m = _delta_metrics(grows)
        m["group_kind"], m["group"] = group_kind, gname
        out.append(m)
    return out


def _legacy_case_row(r):
    return {k: r[k] for k in FIELDNAMES}


def _generic_case_row(r):
    m = {"fpu_value": r["fpu_value"], "case_id": r["case_id"],
         "game_id": r["game_idx"], "ply": r["position_ply"],
         "ply_bucket": r["ply_bucket"], "side_to_move": r["side_to_move"],
         "root_mcts_stm_value": r["root_mcts_stm_value"],
         "root_mcts_black_value": r["root_mcts_black_value"],
         "top_move": r["top_child_move"],
         "top_child_visit_share": r["top_child_visit_share"],
         "root_visit_entropy": r["root_visit_entropy"],
         "root_effective_children": r["root_effective_children"],
         "root_collapsed_ge_0_95": r["root_collapsed_ge_0_95"],
         "root_n_visited_children": r["root_n_visited_children"],
         "top_child_n_visited_children": r["top_child_n_visited_children"]}
    for k in ("root_value_delta_stm_vs_fpu0", "root_value_delta_black_vs_fpu0",
              "top_move_changed_vs_fpu0", "root_children_delta_vs_fpu0",
              "top_child_children_delta_vs_fpu0", "top_child_visit_share_delta_vs_fpu0",
              "root_effective_children_delta_vs_fpu0", "root_visit_entropy_delta_vs_fpu0",
              "new_collapse_vs_fpu0", "resolved_collapse_vs_fpu0"):
        m[k] = r[k]
    return m


def _write_csv(path, fieldnames, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


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
    ap.add_argument("--manifest", "--a-manifest", dest="manifest",
                    default=DEFAULT_A_MANIFEST)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--integrity-csv", "--phase0-csv", dest="integrity_csv",
                    default=None)
    ap.add_argument("--skip-integrity-check", action="store_true")
    ap.add_argument("--fpu-values", default=None,
                    help="comma list; neutral manifests default to the frozen "
                         "protocol 0.0,-0.20.")
    ap.add_argument("--allow-non-protocol-fpu", action="store_true",
                    help="permit non-protocol --fpu-values on a neutral manifest.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--strata-summary-out", default=None)
    ap.add_argument("--eval-batch-size", type=int, default=14)
    ap.add_argument("--stall-flush-sims", type=int, default=48)
    ap.add_argument("--position-probe-base-seed", type=int, default=20260616)
    ap.add_argument("--goal-line-base-seed", type=int, default=20260614)
    ap.add_argument("--limit-cases", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    cases = load_csv_manifest(args.manifest)["cases"]
    if args.limit_cases is not None:
        cases = cases[:args.limit_cases]
    neutral = manifest_is_neutral(cases)
    fpus = resolve_fpu_values(args.fpu_values, neutral, args.allow_non_protocol_fpu)
    if neutral and set(fpus) != set(PROTOCOL_FPUS):
        print(f"[fpu] WARNING: non-protocol values {fpus} on a held-out manifest "
              f"(--allow-non-protocol-fpu); this is not the frozen v16a protocol.")
    out_path, summary_path, strata_path = resolve_output_paths(
        args.out, args.summary_out, args.strata_summary_out, args.manifest, neutral)
    resolved_integrity = resolve_integrity_csv(
        args.integrity_csv, args.skip_integrity_check, neutral, DEFAULT_PHASE0_CSV)
    run_integrity = resolved_integrity is not None
    baseline = _phase0_baseline(resolved_integrity) if run_integrity else {}
    if not run_integrity:
        why = ("--skip-integrity-check" if args.skip_integrity_check
               else "neutral manifest, no baseline" if neutral else "no baseline")
        print(f"[fpu] integrity check SKIPPED ({why}); fpu=0.0 remains the delta baseline.")
    search_fns = _search_fns(args.checkpoint, fpus, args.eval_batch_size,
                             args.stall_flush_sims)

    all_rows = []
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
                    f"fpu_value={x} {cid}: {root.visit_count} sims != {SIMS}")
            black = to_black(root_value_stm, side)
            if run_integrity and x == BASELINE_FPU:
                if cid not in baseline:
                    raise SystemExit(f"{cid} missing from {resolved_integrity}")
                if abs(black - baseline[cid]) > TOLERANCE:
                    raise SystemExit(
                        f"INTEGRITY CHECK FAILED at fpu=0.0 on {cid}: "
                        f"{black:+.6f} != Phase-0 {baseline[cid]:+.6f}")
            over, severe = gate_flags(black)
            top = _best_child(root)
            top_share = "" if top is None else top.visit_count / root.visit_count
            child_visits = [c.visit_count for c in root.children.values()
                            if c.visit_count > 0]
            entropy = visit_entropy(child_visits)
            rows.append({
                "fpu_value": x, "case_id": cid,
                "game_idx": case["game_idx"], "position_ply": case["position_ply"],
                "ply_bucket": case.get("ply_bucket", ""), "side_to_move": side,
                "root_mcts_stm_value": root_value_stm, "root_mcts_black_value": black,
                "gate_over_ge_0_25": over, "gate_severe_ge_0_50": severe,
                "root_n_visited_children": n_visited_children(root),
                "root_visit_entropy": entropy,
                "root_effective_children": math.exp(entropy) if child_visits else 0.0,
                "root_collapsed_ge_0_95": isinstance(top_share, float) and top_share >= 0.95,
                "top_child_move": "" if top is None else "{}:{}".format(*decode_move(top.move)),
                "top_child_visit_share": top_share,
                "top_child_q_black": "" if top is None else to_black(top.q_value, top.state.to_move),
                "top_child_n_visited_children": 0 if top is None else n_visited_children(top),
            })
        if run_integrity and x == BASELINE_FPU:
            print(f"[fpu] integrity check PASSED at fpu=0.0 on {len(rows)} cases")
        all_rows.extend(rows)

    if neutral:
        enrich_with_deltas(all_rows)
        _write_csv(out_path, GENERIC_CASE_FIELDNAMES,
                   [_generic_case_row(r) for r in all_rows])
        overall = []
        for x in fpus:
            g = summarize_grouped([r for r in all_rows if r["fpu_value"] == x], "all")[0]
            g["fpu_value"] = x
            overall.append({k: g[k] for k in GENERIC_SUMMARY_FIELDNAMES})
        _write_csv(summary_path, GENERIC_SUMMARY_FIELDNAMES, overall)
        strata = []
        for x in fpus:
            xr = [r for r in all_rows if r["fpu_value"] == x]
            for kind in ("bucket", "side", "bucket_x_side"):
                for g in summarize_grouped(xr, kind):
                    g["fpu_value"] = x
                    strata.append({k: g[k] for k in STRATA_SUMMARY_FIELDNAMES})
        _write_csv(strata_path, STRATA_SUMMARY_FIELDNAMES, strata)
        for row in overall:
            print(f"[fpu] fpu={row['fpu_value']:<6} "
                  f"mover_dmean={row['mean_root_value_delta_stm_vs_fpu0']:+.4f} "
                  f"flip={row['top_move_flip_rate_vs_fpu0']*100:.1f}% "
                  f"new_collapse={row['new_collapse_rate']*100:.1f}% "
                  f"eff_child_d={row['mean_root_effective_children_delta_vs_fpu0']:+.2f}")
        print(f"\nwrote {len(all_rows)} case rows -> {out_path}")
        print(f"wrote {len(overall)} overall + {len(strata)} stratified summary rows")
    else:
        _write_csv(out_path, FIELDNAMES, [_legacy_case_row(r) for r in all_rows])
        summary_rows = []
        for x in fpus:
            s = summarize([r for r in all_rows if r["fpu_value"] == x])
            s["fpu_value"] = x
            summary_rows.append(s)
        _write_csv(summary_path, SUMMARY_FIELDNAMES, summary_rows)
        for s in summary_rows:
            print(f"[fpu] fpu={s['fpu_value']:<6} mean={s['mean_black_value']:+.4f} "
                  f"over={s['over_pct_ge_0_25']:.1f}% severe={s['severe_pct_ge_0_50']:.1f}%")
        print(f"\nwrote {len(all_rows)} case rows -> {out_path}")
        print(f"wrote {len(summary_rows)} summary rows -> {summary_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
