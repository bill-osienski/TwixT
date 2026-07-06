"""v12 guardrail loss_mode: value-only rows whose target_black_value is the
BASE black-perspective value, emitting a per-row black-perspective sign."""
import json

import numpy as np
import pytest

from scripts.GPU.alphazero import calibration_pool as cp
from scripts.GPU.alphazero.calibration_pool import (
    GUARDRAIL_LOSS_MODE, build_calibration_sample, split_samples_with_guardrail,
    target_in_to_move, VALID_LOSS_MODES, RETENTION_POLICY_LOSS_MODES,
    TEACHER_MODE_LOSS_MODES)
from tests.goal_line_probe_fixtures import legal_replay


def test_guardrail_mode_registered_value_only():
    assert GUARDRAIL_LOSS_MODE == "asymmetric_guardrail_retention"
    assert GUARDRAIL_LOSS_MODE in VALID_LOSS_MODES
    assert GUARDRAIL_LOSS_MODE not in RETENTION_POLICY_LOSS_MODES
    assert GUARDRAIL_LOSS_MODE not in TEACHER_MODE_LOSS_MODES


def _case(**over):
    c = dict(
        case_id="c1", tag="goal_line_guardrail_retention",
        loss_mode=GUARDRAIL_LOSS_MODE,
        replay_path="MISSING.json", position_ply="0", side_to_move="black",
        target_black_value="0.20", teacher_value="0.20",
        teacher_policy_json="", root_visits_json="", extra_moves_json="",
        continuation_source="", continuation_depth="")
    c.update(over)
    return c


def test_guardrail_row_is_value_only(monkeypatch):
    # Stub position building so we test metadata, not board reconstruction.
    import scripts.GPU.alphazero.calibration_pool as m
    from scripts.GPU.alphazero.self_play import PositionRecord
    def fake_pos(case, target):
        # mirror the default (hard_value) branch: outcome = target in stm
        return PositionRecord(
            board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
            to_move=case["side_to_move"], legal_moves=[(0, 0)],
            visit_counts=[0], outcome=target_in_to_move(
                case["side_to_move"], float(case["target_black_value"])),
            active_size=24, ply=0, game_n_moves=None)
    monkeypatch.setattr(m, "build_calibration_position", fake_pos)
    s = build_calibration_sample(_case(), calibration_target=-0.35)
    assert s.loss_mode == GUARDRAIL_LOSS_MODE
    assert s.has_policy_target is False
    assert s.target_black_value == pytest.approx(0.20)


def test_guardrail_validation_rejects_policy_and_root(monkeypatch):
    import scripts.GPU.alphazero.calibration_pool as m
    monkeypatch.setattr(m, "build_calibration_position", lambda c, t: None)
    with pytest.raises(ValueError, match="teacher_policy_json"):
        build_calibration_sample(_case(teacher_policy_json="[0.5,0.5]"), -0.35)
    with pytest.raises(ValueError, match="root_visits_json"):
        build_calibration_sample(_case(root_visits_json="[0.5,0.5]"), -0.35)
    with pytest.raises(ValueError, match="target_black_value"):
        build_calibration_sample(_case(target_black_value=""), -0.35)


def test_split_emits_black_perspective_sign():
    from scripts.GPU.alphazero.self_play import PositionRecord
    from scripts.GPU.alphazero.calibration_pool import CalibrationSample
    def rec(side):
        return PositionRecord(
            board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
            to_move=side, legal_moves=[(0, 0)], visit_counts=[0],
            outcome=0.0, active_size=24, ply=0, game_n_moves=None)
    samples = [
        CalibrationSample(record=rec("black"), loss_mode=GUARDRAIL_LOSS_MODE,
                          tag="g", target_black_value=0.2),
        CalibrationSample(record=rec("red"), loss_mode=GUARDRAIL_LOSS_MODE,
                          tag="g", target_black_value=0.2),
        CalibrationSample(record=rec("black"), loss_mode="hard_value",
                          tag="a", target_black_value=-0.35),
    ]
    _records, _weights, sign = split_samples_with_guardrail(samples, False)
    assert list(sign) == [1.0, -1.0, 0.0]     # black=+1, red=-1, non-guardrail=0


def _real_guardrail_case(tmp_path, side, position_ply, target_black):
    # legal_replay alternates from red: odd ply => black to move, even => red.
    replay = legal_replay(position_ply + 3, game_idx=1)
    rpath = tmp_path / "game_000001.json"
    rpath.write_text(json.dumps(replay))
    return {"game_idx": 1, "case_id": f"g_ply_{position_ply}",
            "replay_path": str(rpath), "position_ply": position_ply,
            "side_to_move": side, "loss_mode": GUARDRAIL_LOSS_MODE,
            "target_black_value": repr(target_black),
            "teacher_value": repr(target_black),
            "teacher_policy_json": "", "root_visits_json": ""}


def test_guardrail_correctness_triple_real_pool(tmp_path):
    """The core correctness triple through the REAL pool path (no stubs):
    (1) cb_targets = target_black converted to side-to-move (record.outcome);
    (2) guardrail_sign = +1 black-to-move / -1 red-to-move; (3) policy mask 0."""
    # (1)+(3) black-to-move (odd ply): stm outcome == +target_black
    sb = build_calibration_sample(
        _real_guardrail_case(tmp_path, "black", 5, 0.20), calibration_target=-0.35)
    assert sb.record.outcome == pytest.approx(target_in_to_move("black", 0.20))  # +0.20
    assert sb.has_policy_target is False
    # (1) red-to-move (even ply): stm outcome == -target_black
    sr = build_calibration_sample(
        _real_guardrail_case(tmp_path, "red", 4, 0.20), calibration_target=-0.35)
    assert sr.record.outcome == pytest.approx(target_in_to_move("red", 0.20))    # -0.20
    # (2) sign emission for the two real rows
    _r, _w, sign = split_samples_with_guardrail([sb, sr], False)
    assert list(sign) == [1.0, -1.0]
