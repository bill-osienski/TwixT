"""Post-opening sharp-drop calibration pool (design Mechanism B).

A fixed set of external replay positions where the checkpoint (as black)
overvalued a losing position. Each becomes a value-only training sample whose
target is a soft negative (black perspective). The pool is sampled each train
step; the value-only MSE term is added to total_loss in alphazero_loss_batch.
"""
from __future__ import annotations

import json
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
        outcome=target_in_to_move(state.to_move, calibration_target),
        active_size=state.active_size,
        ply=position_ply,
        game_n_moves=None,
    )


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
