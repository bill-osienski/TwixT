import csv
import json
import random

import numpy as np
import pytest

from scripts.GPU.alphazero.calibration_pool import (
    target_in_to_move, build_calibration_position, CalibrationPool,
)
from scripts.GPU.alphazero.self_play import PositionRecord
from tests.goal_line_probe_fixtures import legal_replay


def test_target_in_to_move_perspective():
    assert target_in_to_move("black", -0.5) == -0.5
    assert target_in_to_move("red", -0.5) == 0.5
    with pytest.raises(ValueError):
        target_in_to_move("green", -0.5)


def _write_case(tmp_path, game_idx=0, position_ply=5):
    # legal_replay alternates from red; odd ply => black to move.
    assert position_ply % 2 == 1
    replay = legal_replay(position_ply + 3, game_idx=game_idx)
    rpath = tmp_path / f"game_{game_idx:06d}.json"
    rpath.write_text(json.dumps(replay))
    return {
        "game_idx": game_idx,
        "case_id": f"game_{game_idx:06d}_ply_{position_ply:03d}",
        "replay_path": str(rpath),
        "position_ply": position_ply,
        "side_to_move": "black",
    }


def test_build_calibration_position_black(tmp_path):
    case = _write_case(tmp_path, game_idx=1, position_ply=5)
    rec = build_calibration_position(case, calibration_target=-0.5)
    assert isinstance(rec, PositionRecord)
    assert rec.to_move == "black"
    assert rec.outcome == -0.5
    assert rec.active_size == 24
    assert rec.board_tensor.shape == (24, 24, 30)
    assert rec.board_tensor.dtype == np.float32
    assert len(rec.legal_moves) > 0
    assert rec.visit_counts == [0] * len(rec.legal_moves)


def test_missing_replay_raises(tmp_path):
    case = {"replay_path": str(tmp_path / "nonexistent.json"),
            "case_id": "x", "position_ply": 5, "side_to_move": "black"}
    with pytest.raises(FileNotFoundError):
        build_calibration_position(case, calibration_target=-0.5)


def test_empty_pool_raises():
    with pytest.raises(ValueError):
        CalibrationPool([])


def test_from_manifest_loads_all_cases(tmp_path):
    manifest = tmp_path / "train.csv"
    cases = [_write_case(tmp_path, game_idx=i, position_ply=5) for i in (1, 2, 3)]
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["game_idx", "case_id", "replay_path",
                           "position_ply", "side_to_move"])
        w.writeheader()
        w.writerows(cases)
    pool = CalibrationPool.from_manifest(str(manifest), calibration_target=-0.5)
    assert len(pool) == 3
    drawn = pool.sample(7, random.Random(0))
    assert len(drawn) == 7
    assert all(r.outcome == -0.5 for r in drawn)
