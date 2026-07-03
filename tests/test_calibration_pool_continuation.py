import json

import pytest

from scripts.GPU.alphazero.calibration_pool import (
    CONTINUATION_LOSS_MODE, RETENTION_POLICY_LOSS_MODES, TEACHER_MODE_LOSS_MODES,
    VALID_LOSS_MODES, CalibrationPool, build_calibration_sample,
    legal_moves_sha1, split_samples_with_modes)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from tests.goal_line_probe_fixtures import legal_replay


def _root_state(replay):
    return position_state(replay, 5, "black")   # plies 0-4 applied, black to move


def _continuation_fields(replay):
    """Two legal extra moves from ply 5; returns (fields, final_state)."""
    state = _root_state(replay)
    m1 = state.legal_moves()[0]
    s1 = state.apply_move(m1)
    m2 = s1.legal_moves()[0]
    s2 = s1.apply_move(m2)
    fields = {
        "extra_moves_json": json.dumps(
            [{"row": m1[0], "col": m1[1]}, {"row": m2[0], "col": m2[1]}]),
        "continuation_side_to_move": s2.to_move,
        "continuation_legal_moves_sha1": legal_moves_sha1(s2.legal_moves()),
    }
    return fields, s2


def _case(tmp_path, **overrides):
    rp = tmp_path / "game_000001.json"
    replay = legal_replay(9, game_idx=1)
    rp.write_text(json.dumps(replay))
    fields, final_state = _continuation_fields(replay)
    case = {
        "game_idx": "1",
        "case_id": "game_000001_ply_005__cont_pv2_x",
        "replay_path": str(rp), "position_ply": "5", "side_to_move": "black",
        "tag": "old_post_opening_continuation_retention",
        "loss_mode": CONTINUATION_LOSS_MODE,
        "teacher_value": "-0.25", "weight_scale": "1.0",
        **fields,
    }
    case.update(overrides)
    return case, final_state


def test_mode_sets():
    assert CONTINUATION_LOSS_MODE == "searched_continuation_retention"
    assert CONTINUATION_LOSS_MODE in VALID_LOSS_MODES
    assert CONTINUATION_LOSS_MODE in TEACHER_MODE_LOSS_MODES
    # backward-compat: the always-policy set is unchanged
    assert RETENTION_POLICY_LOSS_MODES == frozenset(
        {"teacher_retention", "mcts_root_retention"})
    assert CONTINUATION_LOSS_MODE not in RETENTION_POLICY_LOSS_MODES


def test_continuation_row_reconstructs_and_is_value_only(tmp_path):
    case, final_state = _case(tmp_path)
    sample = build_calibration_sample(case, calibration_target=-0.35)
    assert sample.loss_mode == CONTINUATION_LOSS_MODE
    assert sample.has_policy_target is False
    rec = sample.record
    assert rec.outcome == pytest.approx(-0.25)          # teacher_value, stm, direct
    assert rec.to_move == final_state.to_move           # side flipped twice from black
    assert rec.legal_moves == final_state.legal_moves() # continuation legal set
    assert rec.visit_counts == [0] * len(rec.legal_moves)
    assert rec.ply == 7                                  # position_ply 5 + 2 extra


def test_continuation_row_with_policy_sets_mask(tmp_path):
    case, final_state = _case(tmp_path)
    legal = final_state.legal_moves()
    policy = [1.0 / len(legal)] * len(legal)
    case["teacher_policy_json"] = json.dumps(policy)
    case["teacher_legal_moves_sha1"] = legal_moves_sha1(legal)
    sample = build_calibration_sample(case, calibration_target=-0.35)
    assert sample.has_policy_target is True
    assert sample.record.visit_counts == pytest.approx(policy)
    _, _, mask = split_samples_with_modes([sample], has_weight_scale=False)
    assert mask.tolist() == [1.0]


def test_value_only_mask_is_zero_and_v5_mask_unchanged(tmp_path):
    case, _ = _case(tmp_path)
    cont = build_calibration_sample(case, calibration_target=-0.35)
    _, _, mask = split_samples_with_modes([cont], has_weight_scale=False)
    assert mask.tolist() == [0.0]


@pytest.mark.parametrize("break_field,break_value,match", [
    ("extra_moves_json", "", "extra_moves_json"),
    ("extra_moves_json", "[]", "extra_moves_json"),
    ("extra_moves_json", "not json", "extra_moves_json"),
    # (99,99) is off the active board -> never in legal_moves()
    ("extra_moves_json", json.dumps([{"row": 99, "col": 99}]), "illegal"),
    ("continuation_side_to_move", "", "continuation_side_to_move"),
    ("continuation_legal_moves_sha1", "deadbeef", "sha1"),
    ("teacher_value", "", "teacher_value"),
])
def test_continuation_row_fails_loud(tmp_path, break_field, break_value, match):
    case, _ = _case(tmp_path, **{break_field: break_value})
    with pytest.raises(ValueError, match=match):
        build_calibration_sample(case, calibration_target=-0.35)


def test_wrong_continuation_side_fails(tmp_path):
    case, final_state = _case(tmp_path)
    wrong = "red" if final_state.to_move == "black" else "black"
    case["continuation_side_to_move"] = wrong
    with pytest.raises(ValueError, match="continuation_side_to_move"):
        build_calibration_sample(case, calibration_target=-0.35)


def test_continuation_row_rejects_root_visits_json(tmp_path):
    case, _ = _case(tmp_path, root_visits_json=json.dumps([1.0]))
    with pytest.raises(ValueError, match="root_visits_json"):
        build_calibration_sample(case, calibration_target=-0.35)


def test_hard_value_row_rejects_continuation_columns(tmp_path):
    case, _ = _case(tmp_path)
    case["loss_mode"] = "hard_value"
    case["teacher_value"] = ""
    case["target_black_value"] = "-0.35"
    with pytest.raises(ValueError, match="hard_value"):
        build_calibration_sample(case, calibration_target=-0.35)


def _write_manifest(tmp_path, rows):
    import csv
    fields = sorted({k for r in rows for k in r})
    p = tmp_path / "manifest.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return p


def test_from_manifest_allows_root_plus_continuation_mix(tmp_path):
    cont_case, _ = _case(tmp_path)
    replay = legal_replay(9, game_idx=1)
    state = _root_state(replay)
    legal = state.legal_moves()
    dense = [0.0] * len(legal); dense[0] = 1.0
    root_case = {
        "game_idx": "1", "case_id": "game_000001_ply_005",
        "replay_path": cont_case["replay_path"],
        "position_ply": "5", "side_to_move": "black",
        "tag": "old_post_opening_retention",
        "loss_mode": "mcts_root_retention", "teacher_value": "-0.11",
        "root_visits_json": json.dumps(dense),
        "root_legal_moves_sha1": legal_moves_sha1(legal),
    }
    p = _write_manifest(tmp_path, [root_case, cont_case])
    pool = CalibrationPool.from_manifest(p, calibration_target=-0.35)
    assert pool.schema == "searched_continuation_retention"
    assert len(pool) == 2


def test_from_manifest_still_rejects_teacher_plus_root_mix(tmp_path):
    cont_case, _ = _case(tmp_path)
    teach = dict(cont_case)
    teach["case_id"] = "t1"
    teach["loss_mode"] = "teacher_retention"
    root = dict(cont_case)
    root["case_id"] = "r1"
    root["loss_mode"] = "mcts_root_retention"
    p = _write_manifest(tmp_path, [teach, root])
    with pytest.raises(ValueError, match="retention loss_modes"):
        CalibrationPool.from_manifest(p, calibration_target=-0.35)


def _root_value_case(tmp_path, **overrides):
    """Depth-0 D root-value clone: continuation state IS the root state."""
    rp = tmp_path / "game_000002.json"
    replay = legal_replay(9, game_idx=2)
    rp.write_text(json.dumps(replay))
    state = _root_state(replay)
    case = {
        "game_idx": "2",
        "case_id": "game_000002_ply_005__root_value",
        "replay_path": str(rp), "position_ply": "5", "side_to_move": "black",
        "tag": "red_predrop_root_value_retention",
        "loss_mode": CONTINUATION_LOSS_MODE,
        "teacher_value": "-0.4173", "weight_scale": "1.0",
        "extra_moves_json": "[]",
        "continuation_source": "root_value",
        "continuation_depth": "0",
        "continuation_side_to_move": state.to_move,
        "continuation_legal_moves_sha1": legal_moves_sha1(state.legal_moves()),
    }
    case.update(overrides)
    return case, state


def test_root_value_row_loads_at_root_state(tmp_path):
    case, root = _root_value_case(tmp_path)
    sample = build_calibration_sample(case, calibration_target=-0.35)
    assert sample.loss_mode == CONTINUATION_LOSS_MODE
    assert sample.tag == "red_predrop_root_value_retention"
    assert sample.has_policy_target is False
    rec = sample.record
    assert rec.outcome == pytest.approx(-0.4173)         # teacher_value, stm, direct
    assert rec.to_move == root.to_move                   # root side, no moves applied
    assert rec.legal_moves == root.legal_moves()         # root legal set
    assert rec.visit_counts == [0] * len(rec.legal_moves)
    assert rec.ply == 5                                  # position_ply + 0
    _, _, mask = split_samples_with_modes([sample], has_weight_scale=False)
    assert mask.tolist() == [0.0]                        # value-only: never policy


def test_root_value_row_blank_extra_moves_also_accepted(tmp_path):
    case, root = _root_value_case(tmp_path, extra_moves_json="")
    sample = build_calibration_sample(case, calibration_target=-0.35)
    assert sample.record.legal_moves == root.legal_moves()


def test_root_value_row_rejects_nonempty_extra_moves(tmp_path):
    case, root = _root_value_case(tmp_path)
    m = root.legal_moves()[0]
    case["extra_moves_json"] = json.dumps([{"row": m[0], "col": m[1]}])
    with pytest.raises(ValueError, match="must have empty extra_moves_json"):
        build_calibration_sample(case, calibration_target=-0.35)


def test_root_value_row_still_verifies_sha1(tmp_path):
    case, _ = _root_value_case(tmp_path, continuation_legal_moves_sha1="deadbeef")
    with pytest.raises(ValueError, match="sha1"):
        build_calibration_sample(case, calibration_target=-0.35)


def test_empty_extra_moves_without_root_value_marker_still_fails(tmp_path):
    # non-root_value continuation rows keep today's fail-loud behavior
    case, _ = _case(tmp_path, extra_moves_json="[]")
    with pytest.raises(ValueError, match="extra_moves_json"):
        build_calibration_sample(case, calibration_target=-0.35)
    case2, _ = _case(tmp_path, extra_moves_json="")
    with pytest.raises(ValueError, match="extra_moves_json"):
        build_calibration_sample(case2, calibration_target=-0.35)


def test_root_value_row_rejects_root_visits_json(tmp_path):
    case, _ = _root_value_case(tmp_path, root_visits_json=json.dumps([1.0]))
    with pytest.raises(ValueError, match="root_visits_json"):
        build_calibration_sample(case, calibration_target=-0.35)
