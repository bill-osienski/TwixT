"""Post-opening sharp-drop calibration pool (design Mechanism B).

A fixed set of external replay positions where the checkpoint (as black)
overvalued a losing position. Each becomes a value-only training sample whose
target is a soft negative (black perspective). The pool is sampled each train
step; the value-only MSE term is added to total_loss in alphazero_loss_batch.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .goal_line_trigger_probe_cases import position_state
from .position_probe_cases import load_csv_manifest
from .self_play import PositionRecord


def target_in_to_move(side_to_move: str, calibration_target: float) -> float:
    """Express the black-perspective target in the side-to-move perspective.

    The value head outputs side-to-move perspective. For black-to-move the
    target is used as-is; for red-to-move it is negated.
    """
    if side_to_move == "black":
        return float(calibration_target)
    if side_to_move == "red":
        return float(-calibration_target)
    raise ValueError(f"unexpected side_to_move {side_to_move!r}")


@dataclass(frozen=True)
class CalibrationSample:
    """A calibration position plus its per-row weight/tag/target metadata.

    The loss reads the target from record.outcome (already in side-to-move
    perspective); target_black_value is retained as black-perspective metadata.
    """
    record: PositionRecord
    weight_scale: float = 1.0
    tag: str = ""
    target_black_value: float | None = None


def _resolve_target_black(case: dict, fallback: float) -> float:
    """Per-row black-perspective target, falling back to the global value.
    Validates finite and in [-1.0, +1.0]."""
    raw = case.get("target_black_value")
    target = float(fallback) if raw in (None, "") else float(raw)
    if not math.isfinite(target) or not (-1.0 <= target <= 1.0):
        raise ValueError(
            f"target_black_value {target!r} must be finite in [-1.0, 1.0] "
            f"(case {case.get('case_id')!r})")
    return target


def _parse_weight_scale(case: dict) -> tuple[float, bool]:
    """Return (weight_scale, was_explicit). Default 1.0; validate finite and >= 0."""
    raw = case.get("weight_scale")
    if raw in (None, ""):
        return 1.0, False
    w = float(raw)
    if not math.isfinite(w) or w < 0.0:
        raise ValueError(
            f"weight_scale {w!r} must be finite and >= 0 (case {case.get('case_id')!r})")
    return w, True


def build_calibration_position(case: dict, calibration_target: float) -> PositionRecord:
    """Reconstruct a case to a board and build a value-only PositionRecord.

    visit_counts is a zero vector (policy is never supervised here); outcome
    carries the soft target in side-to-move perspective.
    """
    replay_path = Path(case["replay_path"])
    if not replay_path.exists():
        raise FileNotFoundError(
            f"{case.get('case_id')}: replay not found: {replay_path}")
    replay = json.loads(replay_path.read_text())
    position_ply = int(case["position_ply"])
    side = case["side_to_move"]
    state = position_state(replay, position_ply, side)

    board_chw = state.to_tensor()                       # (30, 24, 24) CHW
    board_hwc = np.transpose(board_chw, (1, 2, 0)).astype(np.float32)  # (24,24,30)
    legal = state.legal_moves()

    return PositionRecord(
        board_tensor=board_hwc,
        to_move=state.to_move,
        legal_moves=legal,
        visit_counts=[0] * len(legal),
        outcome=target_in_to_move(state.to_move, _resolve_target_black(case, calibration_target)),
        active_size=state.active_size,
        ply=position_ply,
        game_n_moves=None,
    )


def build_calibration_sample(case: dict, calibration_target: float) -> CalibrationSample:
    """Wrap a value-only PositionRecord with per-row weight/tag/target metadata."""
    record = build_calibration_position(case, calibration_target)
    weight_scale, _ = _parse_weight_scale(case)
    tag = case.get("tag") or ""
    target_black = _resolve_target_black(case, calibration_target)
    return CalibrationSample(record=record, weight_scale=weight_scale,
                             tag=tag, target_black_value=target_black)


class CalibrationPool:
    """Fixed pool of calibration PositionRecords; sampled with replacement."""

    def __init__(self, records):
        if not records:
            raise ValueError("CalibrationPool requires at least one record")
        self._records = list(records)

    def __len__(self):
        return len(self._records)

    def sample(self, k: int, rng):
        if k <= 0:
            return []
        return [rng.choice(self._records) for _ in range(k)]

    @classmethod
    def from_manifest(cls, manifest_path, calibration_target: float):
        manifest = load_csv_manifest(manifest_path)
        records = [build_calibration_position(c, calibration_target)
                   for c in manifest["cases"]]
        return cls(records)


def build_post_opening_calibration_block(config: dict, enabled: bool,
                                         loss_accumulator: dict) -> dict:
    """Per-iteration calibration telemetry for the training stats sidecar.

    calib_mean_value_pred is the headline signal: it should drift from ~+0.6
    toward the target (~-0.5) over the run.
    """
    steps = max(int(loss_accumulator.get("steps_done", 0)), 1)
    n_drawn = int(loss_accumulator.get("sum_calib_n_drawn", 0))
    return {
        "version": 1,
        "enabled": bool(enabled),
        "config": dict(config),
        "loss": {
            "calib_loss_avg_iter":
                float(loss_accumulator.get("sum_calib_loss", 0.0)) / steps,
            "calib_mean_value_pred":
                float(loss_accumulator.get("sum_calib_value_pred", 0.0)) / steps,
            "calib_n_drawn_total": n_drawn,
            "calib_n_drawn_per_step": n_drawn / steps,
        },
    }
