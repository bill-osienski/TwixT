"""Raw NN-only diagnostic scorer for fixed calibration/probe positions.

DIAGNOSTIC ONLY. Imports existing helpers; changes no manifest, checkpoint, or
training path. NO MCTS: it runs the shared single-position raw forward
(_teacher_infer) in eval-mode BatchNorm across one or more checkpoints and
reports per-position value drift from the teacher (the BASE checkpoint) plus the
top-1 policy move.

Answers: on the shared severe C/D gate rows, did the candidate raw network still
match the teacher, or did it drift before MCTS? See
docs/superpowers/plans/2026-07-01-eval-raw-nn-position-rows-diagnostic.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .position_probe_cases import (
    OVERVALUE_THRESHOLD,
    SEVERE_OVERVALUE_THRESHOLD,
    load_csv_manifest,
)
from .goal_line_trigger_probe_cases import position_state
from .build_teacher_calibration_manifest import _teacher_infer


def to_black(value_stm: float, side_to_move: str) -> float:
    """Express a side-to-move value in the black perspective.

    Involution matching calibration_pool.target_in_to_move and
    eval_position_probe.py:86-91: black as-is, red negated.
    """
    if side_to_move == "black":
        return float(value_stm)
    if side_to_move == "red":
        return float(-value_stm)
    raise ValueError(f"unexpected side_to_move {side_to_move!r}")


def _format_move(move) -> str:
    r, c = move
    return f"{r}:{c}"


def score_row(evaluator, case: dict) -> dict:
    """Raw NN score of one reconstructed position. NN-only (no MCTS)."""
    replay = json.loads(Path(case["replay_path"]).read_text())
    state = position_state(replay, int(float(case["position_ply"])), case["side_to_move"])
    legal, policy, value_stm = _teacher_infer(state, evaluator)
    raw_black = to_black(value_stm, case["side_to_move"])
    if legal:
        i = max(range(len(policy)), key=lambda j: policy[j])
        top1_move = _format_move(legal[i])
        top1_prob = policy[i]
    else:                                     # non-terminal in practice; guard against empty
        top1_move, top1_prob = "", ""
    return {
        "raw_value_stm": value_stm,
        "raw_black_value": raw_black,
        "top1_move": top1_move,
        "top1_prob": top1_prob,
        "overvalue": raw_black >= OVERVALUE_THRESHOLD,
        "severe_overvalue": raw_black >= SEVERE_OVERVALUE_THRESHOLD,
    }


def resolve_deltas(rows: list, base_checkpoint: str) -> None:
    """Second pass: resolve each row's teacher_value and delta, in place.

    teacher_value = manifest teacher_value if present/non-empty, else the BASE
    checkpoint's raw_value_stm for the same case (BASE = the v4 teacher/anchor).
    Delta is computed in side-to-move space (raw_value_stm - teacher_value); NO flip.
    """
    base_raw = {
        r["case_id"]: r["raw_value_stm"]
        for r in rows
        if r["checkpoint"] == base_checkpoint
    }
    for r in rows:
        manifest_tv = r.get("teacher_value", "")
        if manifest_tv not in (None, ""):
            tv, source = float(manifest_tv), "manifest"
        else:
            tv, source = base_raw.get(r["case_id"]), "base_checkpoint"
        r["teacher_value"] = "" if tv is None else tv
        r["teacher_value_source"] = source
        if tv is None:
            r["value_delta_vs_teacher"] = ""
            r["abs_value_delta_vs_teacher"] = ""
        else:
            delta = r["raw_value_stm"] - tv
            r["value_delta_vs_teacher"] = delta
            r["abs_value_delta_vs_teacher"] = abs(delta)


def load_and_filter_cases(manifest_paths, case_ids=None, tags=None, limit=None) -> list:
    """Union rows across manifests; dedup; filter by case_id/tag; cap at limit."""
    seen, cases = set(), []
    for path in manifest_paths:
        for case in load_csv_manifest(path)["cases"]:
            cid = case["case_id"]
            if case_ids and cid not in case_ids:
                continue
            if tags and case.get("tag", "") not in tags:
                continue
            key = (cid, case.get("replay_path"), case["position_ply"], case["side_to_move"])
            if key in seen:
                continue
            seen.add(key)
            cases.append(case)
            if limit and len(cases) >= limit:
                return cases
    return cases


PASSTHROUGH_COLUMNS = (
    "case_id", "tag", "source", "source_rank", "side_to_move", "position_ply",
    "replay_path", "loss_mode", "target_black_value", "teacher_value",
    "teacher_legal_moves_sha1",
)

OUTPUT_COLUMNS = (
    "checkpoint", "checkpoint_label", "case_id", "tag", "source", "source_rank",
    "side_to_move", "position_ply", "replay_path", "loss_mode", "target_black_value",
    "teacher_value", "teacher_value_source", "raw_value_stm", "raw_black_value",
    "value_delta_vs_teacher", "abs_value_delta_vs_teacher",
    "top1_move", "top1_prob", "overvalue", "severe_overvalue", "teacher_legal_moves_sha1",
)


def checkpoint_label(path: str) -> str:
    """Display label (parent dir name). Cosmetic only — deltas key on the path."""
    return Path(path).parent.name


def score_all(cases, checkpoints, evaluator_factory) -> list:
    """Score every (checkpoint, case): passthrough columns + raw NN scores."""
    rows = []
    for ckpt in checkpoints:
        label = checkpoint_label(ckpt)
        evaluator = evaluator_factory(ckpt)
        for case in cases:
            row = {k: case.get(k, "") for k in PASSTHROUGH_COLUMNS}
            row["checkpoint"] = ckpt
            row["checkpoint_label"] = label
            row.update(score_row(evaluator, case))
            rows.append(row)
    return rows


def build_evaluator(checkpoint_path: str):
    """Real factory: load checkpoint, eval-mode BatchNorm, wrap in LocalGPUEvaluator."""
    from .local_evaluator import LocalGPUEvaluator
    from .probe_eval import load_network_for_scoring

    network, *_ = load_network_for_scoring(checkpoint_path)
    network.eval()                              # running stats, batch-independent (matches teacher cache)
    return LocalGPUEvaluator(network)


def write_rows(out_path: str, rows: list) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(OUTPUT_COLUMNS), extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Raw NN-only diagnostic scorer for fixed calibration/probe positions (no MCTS)."
    )
    p.add_argument("--manifest", action="append", default=[], required=True, dest="manifests",
                   metavar="PATH", help="position manifest CSV (repeatable; rows unioned).")
    p.add_argument("--checkpoint", action="append", default=[], required=True, dest="checkpoints",
                   metavar="PATH")
    p.add_argument("--base-checkpoint", default=None, metavar="PATH",
                   help="teacher reference (default: first --checkpoint). Rows without a manifest "
                        "teacher_value use this checkpoint's raw value as the teacher.")
    p.add_argument("--case-id", action="append", default=[], dest="case_ids", metavar="CASE_ID")
    p.add_argument("--tag", action="append", default=[], dest="tags", metavar="TAG")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True, metavar="PATH")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None, evaluator_factory=None) -> int:
    args = parse_args(argv)
    factory = evaluator_factory or build_evaluator

    for ckpt in args.checkpoints:
        if not Path(ckpt).exists():
            print(f"error: checkpoint not found: {ckpt}", file=sys.stderr)
            return 2
    for man in args.manifests:
        if not Path(man).exists():
            print(f"error: manifest not found: {man}", file=sys.stderr)
            return 2

    base_ckpt = args.base_checkpoint or args.checkpoints[0]
    checkpoints = list(args.checkpoints)
    if base_ckpt not in checkpoints:            # ensure base is scored for the teacher fallback
        checkpoints = [base_ckpt] + checkpoints

    cases = load_and_filter_cases(
        args.manifests,
        set(args.case_ids) or None,
        set(args.tags) or None,
        args.limit,
    )
    if not cases:
        print("error: no cases matched filters", file=sys.stderr)
        return 2

    rows = score_all(cases, checkpoints, factory)
    resolve_deltas(rows, base_ckpt)
    write_rows(args.out, rows)

    print(f"wrote {len(rows)} rows ({len(checkpoints)} checkpoints x {len(cases)} cases) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
