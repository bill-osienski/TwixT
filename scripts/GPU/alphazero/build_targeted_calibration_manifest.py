"""Deterministic builder for the Targeted Value Calibration v2 mixed manifest.

Correction rows (hard target) + retention rows (anchored to a checkpoint's own
probe_black_root_value) are merged into one CSV the calibration pool can load.
See docs/superpowers/specs/2026-06-23-targeted-value-calibration-v2-design.md.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def resolve_anchor_rows(rows: list, anchor_label: str) -> list:
    """Rows whose checkpoint == anchor_label (exact); else the unique set whose
    checkpoint endswith ':' + anchor_label. Raise on ambiguous or missing."""
    exact = [r for r in rows if r.get("checkpoint") == anchor_label]
    if exact:
        return exact
    suffix = [r for r in rows if str(r.get("checkpoint", "")).endswith(":" + anchor_label)]
    labels = sorted({r["checkpoint"] for r in suffix})
    if len(labels) == 1:
        return suffix
    if len(labels) > 1:
        raise ValueError(
            f"ambiguous anchor label {anchor_label!r}; candidates: {labels}; "
            f"pass an exact --*-anchor-label")
    raise ValueError(f"no checkpoint matches anchor label {anchor_label!r}")


UNIFIED_COLUMNS = [
    "case_rank", "tag", "source", "source_rank", "target_black_value", "weight_scale",
    "game_idx", "case_id", "replay_path", "position_ply", "side_to_move",
    "anchor_checkpoint", "drop_ply", "largest_drop_phase", "collapse_type",
]


def _unified_row(**kw) -> dict:
    row = {c: "" for c in UNIFIED_COLUMNS}
    row.update(kw)
    extra = set(kw) - set(UNIFIED_COLUMNS)
    if extra:
        raise KeyError(f"unknown unified columns: {sorted(extra)}")
    return row


def _read_csv(path) -> list:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _ply_key(replay_path: str, position_ply) -> tuple:
    return (replay_path, str(int(float(position_ply))))


def _validate_target_str(value: str, case_id: str) -> str:
    t = float(value)
    if not math.isfinite(t) or not (-1.0 <= t <= 1.0):
        raise ValueError(f"target_black_value {t!r} out of [-1,1] (case {case_id!r})")
    return value


def correction_rows(manifest_path, target: float, weight: float) -> list:
    target_str = _validate_target_str(str(target), "correction-target")
    out = []
    for r in _read_csv(manifest_path):
        out.append(_unified_row(
            tag="black_predrop_correction",
            source=Path(manifest_path).name,
            source_rank=r.get("case_rank", ""),
            target_black_value=target_str,
            weight_scale=str(weight),
            game_idx=r["game_idx"], case_id=r["case_id"], replay_path=r["replay_path"],
            position_ply=r["position_ply"], side_to_move=r["side_to_move"],
            drop_ply=r.get("drop_ply", ""),
            largest_drop_phase=r.get("largest_drop_phase", ""),
            collapse_type=r.get("collapse_type", "")))
    return out


def assert_no_holdout_overlap(correction: list, holdout_path) -> None:
    holdout_keys = {_ply_key(r["replay_path"], r["position_ply"])
                    for r in _read_csv(holdout_path)}
    leaks = [r for r in correction
             if _ply_key(r["replay_path"], r["position_ply"]) in holdout_keys]
    if leaks:
        raise ValueError(
            f"correction train leaks {len(leaks)} frozen-eval positions: "
            f"{[r['case_id'] for r in leaks]}")
