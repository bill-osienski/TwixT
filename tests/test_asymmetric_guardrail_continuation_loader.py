"""v12b: a guardrail row carrying extra_moves_json reconstructs the CONTINUATION
state (not the root), with the target/sign in continuation-side perspective.
Root guardrail rows (blank extra_moves_json) still reconstruct the root."""
import json

import pytest

from scripts.GPU.alphazero.calibration_pool import (
    GUARDRAIL_LOSS_MODE, build_calibration_sample, split_samples_with_guardrail,
    target_in_to_move, legal_moves_sha1)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from tests.goal_line_probe_fixtures import legal_replay


def _root_state(replay):
    return position_state(replay, 5, "black")   # plies 0-4 applied, black to move


def _apply_n(replay, n):
    """Apply n legal moves from the ply-5 root; return (extra_moves, final_state)."""
    state = _root_state(replay)
    moves = []
    for _ in range(n):
        m = state.legal_moves()[0]
        moves.append({"row": m[0], "col": m[1]})
        state = state.apply_move(m)
    return moves, state


def _cont_guardrail_case(tmp_path, n_moves, target_black):
    rp = tmp_path / "game_000001.json"
    replay = legal_replay(9, game_idx=1)
    rp.write_text(json.dumps(replay))
    moves, final = _apply_n(replay, n_moves)
    case = {
        "game_idx": "1", "case_id": "game_000001_ply_005__cont__guardrail",
        "replay_path": str(rp), "position_ply": "5", "side_to_move": "black",
        "tag": "old_post_opening_continuation_guardrail_retention",
        "loss_mode": GUARDRAIL_LOSS_MODE,
        "target_black_value": repr(target_black),
        "teacher_value": repr(target_black),      # provenance only
        "extra_moves_json": json.dumps(moves),
        "continuation_side_to_move": final.to_move,
        "continuation_legal_moves_sha1": legal_moves_sha1(final.legal_moves()),
        "continuation_depth": str(n_moves), "continuation_source": "pv",
        "teacher_policy_json": "", "root_visits_json": "",
    }
    return case, final


def test_continuation_guardrail_reconstructs_even_depth(tmp_path):
    # 2 moves from black root -> black to move (even depth)
    case, final = _cont_guardrail_case(tmp_path, 2, 0.30)
    s = build_calibration_sample(case, calibration_target=-0.35)
    assert s.loss_mode == GUARDRAIL_LOSS_MODE
    assert s.has_policy_target is False
    rec = s.record
    assert rec.ply == 7                                       # 5 + 2
    assert rec.to_move == final.to_move                       # continuation side
    assert rec.legal_moves == final.legal_moves()
    assert rec.outcome == pytest.approx(target_in_to_move(final.to_move, 0.30))
    _r, _w, sign = split_samples_with_guardrail([s], False)
    assert list(sign) == [1.0 if final.to_move == "black" else -1.0]


def test_continuation_guardrail_reconstructs_odd_depth(tmp_path):
    # 1 move from black root -> red to move (odd depth): sign flips to -1
    case, final = _cont_guardrail_case(tmp_path, 1, 0.30)
    s = build_calibration_sample(case, calibration_target=-0.35)
    rec = s.record
    assert rec.ply == 6                                       # 5 + 1
    assert rec.to_move == final.to_move                       # red
    assert rec.outcome == pytest.approx(target_in_to_move(final.to_move, 0.30))
    _r, _w, sign = split_samples_with_guardrail([s], False)
    assert list(sign) == [-1.0]                               # red-to-move continuation


def test_root_guardrail_still_reconstructs_root(tmp_path):
    # blank extra_moves_json -> falls through to the root branch (v12 behavior)
    rp = tmp_path / "game_000001.json"
    replay = legal_replay(9, game_idx=1)
    rp.write_text(json.dumps(replay))
    root = _root_state(replay)
    case = {
        "game_idx": "1", "case_id": "game_000001_ply_005__guardrail",
        "replay_path": str(rp), "position_ply": "5", "side_to_move": "black",
        "tag": "goal_line_guardrail_retention", "loss_mode": GUARDRAIL_LOSS_MODE,
        "target_black_value": "0.20", "teacher_value": "0.20",
        "teacher_policy_json": "", "root_visits_json": "", "extra_moves_json": "",
    }
    s = build_calibration_sample(case, calibration_target=-0.35)
    rec = s.record
    assert rec.ply == 5                                       # no extra moves
    assert rec.to_move == root.to_move                        # black (root side)
    assert rec.outcome == pytest.approx(target_in_to_move("black", 0.20))   # +0.20


def test_continuation_guardrail_bad_sha1_fails_loud(tmp_path):
    case, _ = _cont_guardrail_case(tmp_path, 2, 0.30)
    case["continuation_legal_moves_sha1"] = "deadbeef"
    with pytest.raises(ValueError, match="sha1"):
        build_calibration_sample(case, calibration_target=-0.35)
