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
