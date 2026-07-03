"""Deterministic v6 searched-continuation retention manifest builder.

Source = the v5 manifest (targeted_calibration_v5_mcts_root_from_calib020_0001.csv).
Every source row passes through UNCHANGED. Additionally, each B/C/D
mcts_root_retention row is re-searched with the gate-faithful config
(search_with_root: same synchronous path as the gates; NEVER search_from_root)
and continuation rows are extracted per the spec's tag-based rules and appended
immediately after their parent row.

Each continuation row anchors a fresh EVAL-mode raw teacher value at the
continuation state (_teacher_infer on a separate eval() network — the tree's
train-mode nn_value is provenance only). Policy columns stay blank unless
--emit-continuation-policy.

Cross-checks (all hard failures):
  - fresh root_black_value vs the SOURCE row's stored root_black_value
    (--source-root-tolerance) — proves the v6 rebuild reproduces v5's search;
  - optional --gate-cases-csv cross-check (reuses the v5 builder's
    cross_check_gate_values with the fresh values);
  - per-root and total continuation caps; case_id uniqueness.

See docs/superpowers/specs/2026-07-02-targeted-value-calibration-v6-searched-
continuation-retention-design.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from .position_probe_cases import load_csv_manifest
from .goal_line_trigger_probe_cases import position_state
from .build_teacher_calibration_manifest import _teacher_infer
from .build_mcts_root_retention_manifest import (
    cross_check_gate_values, output_fieldnames, row_seed, _to_black)
from .calibration_pool import CONTINUATION_LOSS_MODE, legal_moves_sha1
from .continuation_extraction import (
    CONTINUATION_TAG_BY_SOURCE_TAG, FAMILY_BY_SOURCE_TAG, continuation_case_id,
    extract_continuations, format_path_moves, root_max_visit_share)

CORRECTION_TAG = "black_predrop_correction"
D_ROOT_VALUE_TAG = "red_predrop_root_value_retention"
D_ROOT_VALUE_SOURCE_TAG = "red_predrop_retention"
NEW_COLUMNS_V6 = [
    "extra_moves_json", "continuation_side_to_move",
    "continuation_legal_moves_sha1", "continuation_depth",
    "continuation_parent_case_id", "continuation_source",
    "continuation_path_moves", "continuation_tree_visits",
    "continuation_tree_nn_value", "teacher_value_source",
]
# Expected family shapes for the D2 telemetry warnings (not row gates):
_SHARP_WARN = {"old_post_opening_retention": 0.65}   # C expected sharp (>=)
_DIFFUSE_WARN = {"red_predrop_retention": 0.65}      # D expected diffuse (<)


def classify_row(r: dict) -> str:
    tag = r.get("tag", "")
    mode = r.get("loss_mode") or "hard_value"
    if mode == "hard_value" and tag == CORRECTION_TAG:
        return "passthrough"
    if mode == "mcts_root_retention" and tag in FAMILY_BY_SOURCE_TAG:
        return "extract"
    if (mode == CONTINUATION_LOSS_MODE
            and tag in CONTINUATION_TAG_BY_SOURCE_TAG.values()):
        return "passthrough"                        # rerun on a v6 output
    if mode == CONTINUATION_LOSS_MODE and tag == D_ROOT_VALUE_TAG:
        return "passthrough"                        # rerun on a v6c output
    raise ValueError(f"{r.get('case_id')}: unknown loss_mode/tag combination "
                     f"({mode!r}, {tag!r})")


def _continuation_row(parent: dict, spec, raw_evaluator, emit_policy: bool) -> dict:
    row = dict(parent)                              # inherit ALL parent columns
    for c in NEW_COLUMNS_V6:
        row.setdefault(c, "")
    legal_c, priors_c, raw_value = _teacher_infer(spec.state, raw_evaluator)
    row["case_id"] = continuation_case_id(parent["case_id"], spec)
    row["tag"] = CONTINUATION_TAG_BY_SOURCE_TAG[parent["tag"]]
    row["loss_mode"] = CONTINUATION_LOSS_MODE
    row["teacher_value"] = repr(float(raw_value))
    row["teacher_value_source"] = "base_raw_continuation"
    row["extra_moves_json"] = json.dumps(
        [{"row": r, "col": c} for (r, c) in spec.path_moves])
    row["continuation_side_to_move"] = spec.state.to_move
    row["continuation_legal_moves_sha1"] = legal_moves_sha1(legal_c)
    row["continuation_depth"] = str(spec.depth)
    row["continuation_parent_case_id"] = parent["case_id"]
    row["continuation_source"] = spec.source
    row["continuation_path_moves"] = format_path_moves(spec.path_moves)
    row["continuation_tree_visits"] = str(spec.tree_visits)
    row["continuation_tree_nn_value"] = (
        "" if spec.tree_nn_value is None else repr(float(spec.tree_nn_value)))
    row["target_black_value"] = ""                  # never a hard target
    row["root_visits_json"] = ""                    # not a root-policy row
    if emit_policy:
        total = sum(priors_c) or 1.0
        row["teacher_policy_json"] = json.dumps([p / total for p in priors_c])
        row["teacher_legal_moves_sha1"] = legal_moves_sha1(legal_c)
    else:
        row["teacher_policy_json"] = ""
        row["teacher_legal_moves_sha1"] = ""
    return row


def _root_value_row(parent: dict, root_state) -> dict:
    """Depth-0 value-only clone of a D root row (v6c): anchors the raw
    eval-mode teacher_value at the ROOT state with no policy signal.
    teacher_value is INHERITED from the source row (raw eval anchor) —
    NEVER root_black_value/root_value_stm (the MCTS root scalar; experiment
    ledger do-not-repeat #9)."""
    if (parent.get("teacher_value") or "") == "":
        raise ValueError(
            f"{parent.get('case_id')}: D root row lacks teacher_value; cannot "
            f"emit a root-value clone")
    row = dict(parent)                              # inherit ALL parent columns
    for c in NEW_COLUMNS_V6:
        row.setdefault(c, "")
    legal = root_state.legal_moves()
    row["case_id"] = f"{parent['case_id']}__root_value"
    row["tag"] = D_ROOT_VALUE_TAG
    row["loss_mode"] = CONTINUATION_LOSS_MODE
    row["teacher_value_source"] = "base_raw_root_clone"
    row["extra_moves_json"] = "[]"
    row["continuation_side_to_move"] = root_state.to_move
    row["continuation_legal_moves_sha1"] = legal_moves_sha1(legal)
    row["continuation_depth"] = "0"
    row["continuation_parent_case_id"] = parent["case_id"]
    row["continuation_source"] = "root_value"
    row["continuation_path_moves"] = ""
    row["continuation_tree_visits"] = ""
    row["continuation_tree_nn_value"] = ""
    row["target_black_value"] = ""                  # never a hard target
    row["root_visits_json"] = ""                    # NO policy signal
    row["teacher_policy_json"] = ""
    row["teacher_legal_moves_sha1"] = ""
    return row


def build_rows_v6(rows, raw_evaluator, search_fn, *, pos_base_seed,
                  goal_base_seed, b_pv_depth, c_pv_depth, d_top_k,
                  d_child_pv_depth, d_child_pv_min_visits, max_per_root,
                  max_total, emit_policy, source_root_tolerance,
                  limit_cases, only_case_ids, emit_d_root_value=False):
    # NOTE: continuation rows inherit their parent's root_* provenance stamps
    # (sims/seed/checkpoint/batch/stall) via dict(parent) — the search config
    # itself is proven equivalent by the source-root cross-check, so this
    # function does not take sims/base_checkpoint/batch/stall parameters.
    out, fresh_root_black = [], {}
    stats = {"n_continuation": 0, "n_root_value": 0, "by_tag": {}, "excluded": []}
    n_extracted_roots = 0
    for r in rows:
        row = dict(r)
        for c in NEW_COLUMNS_V6:
            row.setdefault(c, "")
        out.append(row)
        if classify_row(r) != "extract":
            continue
        cid = r["case_id"]
        if only_case_ids is not None and cid not in only_case_ids:
            stats["excluded"].append(f"{cid}: not in --only-case-id")
            continue
        if limit_cases is not None and n_extracted_roots >= limit_cases:
            stats["excluded"].append(f"{cid}: past --limit-cases {limit_cases}")
            continue
        n_extracted_roots += 1
        replay = json.loads(Path(r["replay_path"]).read_text())
        ply = int(float(r["position_ply"]))
        side = r["side_to_move"]
        state = position_state(replay, ply, side)
        if emit_d_root_value and r["tag"] == D_ROOT_VALUE_SOURCE_TAG:
            rv_row = _root_value_row(r, state)
            out.append(rv_row)
            stats["n_root_value"] += 1
            stats["by_tag"][D_ROOT_VALUE_TAG] = (
                stats["by_tag"].get(D_ROOT_VALUE_TAG, 0) + 1)
        seed = row_seed(r.get("tag", ""), r["game_idx"], ply,
                        pos_base_seed, goal_base_seed)
        counts, root_value_stm, root = search_fn(state, seed)
        fresh_black = _to_black(root_value_stm, side)
        fresh_root_black[cid] = fresh_black
        stored = r.get("root_black_value")
        if stored not in (None, "") and (
                abs(fresh_black - float(stored)) > source_root_tolerance):
            raise ValueError(
                f"{cid}: source root value mismatch — recomputed "
                f"{fresh_black:+.4f} vs stored {float(stored):+.4f} "
                f"(wrong seeds / BN mode / config?)")
        share = root_max_visit_share(root)
        tag = r["tag"]
        if tag in _SHARP_WARN and share < _SHARP_WARN[tag]:
            print(f"WARNING: {cid}: C root diffuse (max share {share:.3f})")
        if tag in _DIFFUSE_WARN and share >= _DIFFUSE_WARN[tag]:
            print(f"WARNING: {cid}: D root sharp (max share {share:.3f})")
        print(f"{cid}: root max-visit-share {share:.3f}")
        specs = extract_continuations(
            root, tag, b_pv_depth=b_pv_depth, c_pv_depth=c_pv_depth,
            d_top_k=d_top_k, d_child_pv_depth=d_child_pv_depth,
            d_child_pv_min_visits=d_child_pv_min_visits,
            max_per_root=max_per_root)
        for spec in specs:
            crow = _continuation_row(r, spec, raw_evaluator, emit_policy)
            out.append(crow)
            stats["n_continuation"] += 1
            ctag = crow["tag"]
            stats["by_tag"][ctag] = stats["by_tag"].get(ctag, 0) + 1
        if stats["n_continuation"] > max_total:
            raise ValueError(
                f"continuation rows {stats['n_continuation']} exceed max_total "
                f"{max_total} — raise the cap or tighten thresholds "
                f"(operator tuning point, see spec §1 D4)")
    ids = [r["case_id"] for r in out]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate case_ids in output: {dupes}")
    stats["fresh_root_black"] = fresh_root_black
    return out, stats


def _real_search_fn(base_checkpoint: str, sims: int,
                    eval_batch_size: int, stall_flush_sims: int):
    """Gate-faithful root-returning search. Heavy imports deferred (fakes in
    tests). Same evaluator/config as the v5 builder and the gate probes."""
    from .eval_runner import EvalConfig, cfg_from, _default_evaluator_factory
    from .mcts import MCTS
    evaluator = _default_evaluator_factory(base_checkpoint)
    cfg = cfg_from(EvalConfig(mcts_sims=sims,
                              mcts_eval_batch_size=eval_batch_size,
                              mcts_stall_flush_sims=stall_flush_sims))

    def search_fn(state, seed):
        return MCTS(evaluator, cfg, random.Random(seed)).search_with_root(
            state, add_noise=False)

    return search_fn


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the v6 searched-continuation retention manifest.")
    ap.add_argument("--source", required=True, help="the v5 manifest CSV")
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--position-probe-base-seed", type=int, default=20260616)
    ap.add_argument("--goal-line-base-seed", type=int, default=20260614)
    ap.add_argument("--eval-batch-size", type=int, default=14)
    ap.add_argument("--stall-flush-sims", type=int, default=48)
    ap.add_argument("--b-pv-depth", type=int, default=2)
    ap.add_argument("--c-pv-depth", type=int, default=3)
    ap.add_argument("--d-top-k", type=int, default=3)
    ap.add_argument("--d-child-pv-depth", type=int, default=1)
    ap.add_argument("--d-child-pv-min-visits", type=int, default=40)
    ap.add_argument("--max-continuations-per-root", type=int, default=6)
    ap.add_argument("--max-total-continuation-rows", type=int, default=250)
    ap.add_argument("--emit-continuation-policy", action="store_true",
                    help="also write dense eval-mode teacher policy on "
                         "continuation rows (v6b variant; default OFF = value-only)")
    ap.add_argument("--source-root-tolerance", type=float, default=1e-3)
    ap.add_argument("--gate-cases-csv", action="append", default=[])
    ap.add_argument("--gate-tolerance", type=float, default=1e-3)
    ap.add_argument("--gate-checkpoint-label", default=None)
    ap.add_argument("--limit-cases", type=int, default=None,
                    help="extract from only the first N eligible roots")
    ap.add_argument("--only-case-id", action="append", default=None,
                    help="extract only from these root case_ids (repeatable)")
    ap.add_argument("--emit-d-root-value-rows", action="store_true",
                    help="v6c: for each red_predrop_retention source row, also "
                         "emit a depth-0 value-only red_predrop_root_value_retention "
                         "clone (default OFF = byte-identical v6 output)")
    args = ap.parse_args(argv)

    from .local_evaluator import LocalGPUEvaluator
    from .probe_eval import load_network_for_scoring
    rows = load_csv_manifest(args.source)["cases"]
    network, *_ = load_network_for_scoring(args.base_checkpoint)
    network.eval()                       # raw anchors: EVAL-mode BN
    raw_evaluator = LocalGPUEvaluator(network)
    search_fn = _real_search_fn(args.base_checkpoint, args.sims,
                                args.eval_batch_size, args.stall_flush_sims)

    out_rows, stats = build_rows_v6(
        rows, raw_evaluator, search_fn,
        pos_base_seed=args.position_probe_base_seed,
        goal_base_seed=args.goal_line_base_seed,
        b_pv_depth=args.b_pv_depth, c_pv_depth=args.c_pv_depth,
        d_top_k=args.d_top_k, d_child_pv_depth=args.d_child_pv_depth,
        d_child_pv_min_visits=args.d_child_pv_min_visits,
        max_per_root=args.max_continuations_per_root,
        max_total=args.max_total_continuation_rows,
        emit_policy=args.emit_continuation_policy,
        source_root_tolerance=args.source_root_tolerance,
        limit_cases=args.limit_cases,
        only_case_ids=set(args.only_case_id) if args.only_case_id else None,
        emit_d_root_value=args.emit_d_root_value_rows)

    if args.gate_cases_csv:
        check_rows = [{"loss_mode": "mcts_root_retention", "case_id": cid,
                       "root_black_value": repr(v)}
                      for cid, v in stats["fresh_root_black"].items()]
        gs = cross_check_gate_values(check_rows, args.gate_cases_csv,
                                     args.gate_tolerance,
                                     checkpoint_label=args.gate_checkpoint_label)
        print(f"gate cross-check PASS: {gs['checked']} matched, "
              f"{gs['unmatched']} roots without a gate row")
    else:
        print("WARNING: no --gate-cases-csv given; fresh root values checked "
              "only against the source manifest")

    for line in stats["excluded"]:
        print(f"excluded: {line}")
    base_columns = list(rows[0].keys()) if rows else []
    fieldnames = output_fieldnames(base_columns, out_rows)
    for c in NEW_COLUMNS_V6:
        if c not in fieldnames:
            fieldnames.append(c)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows ({stats['n_continuation']} continuation, "
          f"{stats['n_root_value']} root-value: {stats['by_tag']}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
