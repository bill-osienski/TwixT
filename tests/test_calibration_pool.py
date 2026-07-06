import csv
import json
import math
import random

import numpy as np
import pytest

from scripts.GPU.alphazero.calibration_pool import (
    target_in_to_move, build_calibration_position, CalibrationPool,
    CalibrationSample, build_calibration_sample,
    _resolve_target_black, _parse_weight_scale, split_samples,
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


def _write_case_side(tmp_path, side, position_ply, game_idx=1, **extra):
    """legal_replay alternates from red: odd ply => black to move, even => red."""
    replay = legal_replay(position_ply + 3, game_idx=game_idx)
    rpath = tmp_path / f"game_{game_idx:06d}.json"
    rpath.write_text(json.dumps(replay))
    case = {
        "game_idx": game_idx,
        "case_id": f"game_{game_idx:06d}_ply_{position_ply:03d}",
        "replay_path": str(rpath),
        "position_ply": position_ply,
        "side_to_move": side,
    }
    case.update(extra)
    return case


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
    assert all(s.record.outcome == -0.5 for s in drawn)


def test_build_post_opening_calibration_block():
    from scripts.GPU.alphazero.calibration_pool import (
        build_post_opening_calibration_block,
    )
    block = build_post_opening_calibration_block(
        config={"enabled": True, "target": -0.5, "effective_weight": 0.02,
                "pool_size": 134},
        enabled=True,
        loss_accumulator={"sum_calib_loss": 4.0, "sum_calib_n_drawn": 60,
                          "sum_calib_value_pred": 3.0, "steps_done": 10},
    )
    assert block["enabled"] is True
    assert block["version"] == 1
    assert block["config"]["pool_size"] == 134
    np.testing.assert_allclose(block["loss"]["calib_loss_avg_iter"], 0.4)
    np.testing.assert_allclose(block["loss"]["calib_mean_value_pred"], 0.3)
    assert block["loss"]["calib_n_drawn_total"] == 60
    assert block["draws_by_tag"] == {}  # absent in accumulator -> empty dict


def test_per_row_target_overrides_global(tmp_path):
    case = _write_case_side(tmp_path, "black", 5, target_black_value="-0.35")
    rec = build_calibration_position(case, calibration_target=-0.5)
    assert rec.outcome == -0.35  # per-row wins over global -0.5


def test_red_side_to_move_sign_flip(tmp_path):
    case = _write_case_side(tmp_path, "red", 4, target_black_value="-0.30")
    rec = build_calibration_position(case, calibration_target=-0.5)
    assert rec.outcome == 0.30  # black-perspective -0.30 → side-to-move (red) = +0.30


def test_parse_weight_scale_default_and_explicit():
    assert _parse_weight_scale({}) == (1.0, False)
    assert _parse_weight_scale({"weight_scale": ""}) == (1.0, False)
    assert _parse_weight_scale({"weight_scale": "0.5"}) == (0.5, True)


def test_invalid_target_raises():
    with pytest.raises(ValueError):
        _resolve_target_black({"target_black_value": "1.5"}, fallback=-0.5)
    with pytest.raises(ValueError):
        _resolve_target_black({"target_black_value": "nan"}, fallback=-0.5)


def test_invalid_weight_raises():
    with pytest.raises(ValueError):
        _parse_weight_scale({"weight_scale": "-0.1"})
    with pytest.raises(ValueError):
        _parse_weight_scale({"weight_scale": "inf"})


def test_build_calibration_sample_carries_metadata(tmp_path):
    case = _write_case_side(tmp_path, "black", 5,
                            target_black_value="-0.35", weight_scale="0.5", tag="correction")
    s = build_calibration_sample(case, calibration_target=-0.5)
    assert isinstance(s, CalibrationSample)
    assert s.weight_scale == 0.5
    assert s.tag == "correction"
    assert s.target_black_value == -0.35
    assert s.record.outcome == -0.35


def _write_manifest(tmp_path, rows, name="m.csv"):
    fieldnames = sorted({k for r in rows for k in r})
    path = tmp_path / name
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


def test_from_manifest_detects_global_schema_no_weights(tmp_path):
    cases = [_write_case(tmp_path, game_idx=i, position_ply=5) for i in (1, 2)]
    path = _write_manifest(tmp_path, cases)
    pool = CalibrationPool.from_manifest(str(path), calibration_target=-0.5)
    assert pool.schema == "global_target"
    assert pool.has_weight_scale is False


def test_from_manifest_detects_per_row_schema_and_weights(tmp_path):
    cases = []
    for i in (1, 2):
        c = _write_case(tmp_path, game_idx=i, position_ply=5)
        c["target_black_value"] = "-0.35"
        c["weight_scale"] = "0.5"
        c["tag"] = "correction" if i == 1 else "retention"
        cases.append(c)
    path = _write_manifest(tmp_path, cases)
    pool = CalibrationPool.from_manifest(str(path), calibration_target=-0.5)
    assert pool.schema == "per_row_target"
    assert pool.has_weight_scale is True
    assert pool.tag_counts() == {"correction": 1, "retention": 1}


def test_split_samples_gating(tmp_path):
    # has_weight_scale=False → weights None; True → full array incl. 1.0 defaults
    s_explicit = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, weight_scale="0.5"), -0.5)
    s_default = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=2), -0.5)  # omitted → 1.0
    records, weights = split_samples([s_explicit, s_default], has_weight_scale=False)
    assert weights is None
    assert [type(r).__name__ for r in records] == ["PositionRecord", "PositionRecord"]
    records, weights = split_samples([s_explicit, s_default], has_weight_scale=True)
    assert weights is not None
    assert list(weights) == [0.5, 1.0]


def test_pool_rejects_raw_position_records(tmp_path):
    rec = build_calibration_position(
        _write_case(tmp_path, game_idx=1, position_ply=5), calibration_target=-0.5)
    with pytest.raises(TypeError):
        CalibrationPool([rec])


def test_sidecar_block_passes_through_v2_config_fields():
    from scripts.GPU.alphazero.calibration_pool import build_post_opening_calibration_block
    block = build_post_opening_calibration_block(
        config={"enabled": True, "schema": "per_row_target", "has_weight_scale": True,
                "tags": {"black_predrop_correction": 50, "red_predrop_retention": 30}},
        enabled=True,
        loss_accumulator={"sum_calib_loss": 4.0, "sum_calib_n_drawn": 60,
                          "sum_calib_value_pred": 3.0, "steps_done": 10})
    assert block["config"]["schema"] == "per_row_target"
    assert block["config"]["has_weight_scale"] is True
    assert block["config"]["tags"]["black_predrop_correction"] == 50


def test_sample_by_tag_draws_requested_counts(tmp_path):
    s_corr_a = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, tag="correction"), -0.5)
    s_corr_b = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=2, tag="correction"), -0.5)
    s_ret = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=3, tag="retention"), -0.5)
    pool = CalibrationPool([s_corr_a, s_corr_b, s_ret])
    drawn = pool.sample_by_tag({"correction": 2, "retention": 1}, random.Random(0))
    tags = [s.tag for s in drawn]
    assert len(drawn) == 3
    assert tags.count("correction") == 2
    assert tags.count("retention") == 1


def test_sample_by_tag_samples_with_replacement(tmp_path):
    s = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, tag="correction"), -0.5)
    pool = CalibrationPool([s])  # single-member bucket
    drawn = pool.sample_by_tag({"correction": 4}, random.Random(0))
    assert len(drawn) == 4
    assert all(d.tag == "correction" for d in drawn)


def test_sample_by_tag_zero_count_skips(tmp_path):
    s_corr = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, tag="correction"), -0.5)
    s_ret = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=2, tag="retention"), -0.5)
    pool = CalibrationPool([s_corr, s_ret])
    drawn = pool.sample_by_tag({"correction": 2, "retention": 0}, random.Random(0))
    assert len(drawn) == 2
    assert all(d.tag == "correction" for d in drawn)


def test_sample_by_tag_unknown_tag_raises(tmp_path):
    s = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, tag="correction"), -0.5)
    pool = CalibrationPool([s])
    with pytest.raises(ValueError):
        pool.sample_by_tag({"nonexistent": 1}, random.Random(0))


def test_validate_tag_schedule_passes_for_known_tags(tmp_path):
    s = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, tag="correction"), -0.5)
    pool = CalibrationPool([s])
    pool.validate_tag_schedule({"correction": 2})  # no raise


def test_validate_tag_schedule_raises_for_missing_tag(tmp_path):
    s = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, tag="correction"), -0.5)
    pool = CalibrationPool([s])
    with pytest.raises(ValueError):
        pool.validate_tag_schedule({"correction": 1, "typo_tag": 1})


def test_validate_tag_schedule_ignores_zero_count_missing_tag(tmp_path):
    s = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1, tag="correction"), -0.5)
    pool = CalibrationPool([s])
    pool.validate_tag_schedule({"correction": 1, "absent": 0})  # 0-count tag skipped


def test_sidecar_block_surfaces_draws_by_tag():
    from scripts.GPU.alphazero.calibration_pool import (
        build_post_opening_calibration_block,
    )
    block = build_post_opening_calibration_block(
        config={"enabled": True},
        enabled=True,
        loss_accumulator={"sum_calib_loss": 4.0, "sum_calib_n_drawn": 60,
                          "sum_calib_value_pred": 3.0, "steps_done": 10,
                          "sum_calib_n_drawn_by_tag": {"correction": 40,
                                                       "retention": 20}})
    assert block["draws_by_tag"] == {"correction": 40, "retention": 20}


def test_legal_moves_sha1_stable_and_order_sensitive():
    from scripts.GPU.alphazero.calibration_pool import legal_moves_sha1
    a = legal_moves_sha1([(0, 0), (1, 2), (3, 4)])
    b = legal_moves_sha1([(0, 0), (1, 2), (3, 4)])
    c = legal_moves_sha1([(1, 2), (0, 0), (3, 4)])  # same length, reordered
    assert a == b                       # deterministic
    assert a != c                       # catches a same-length reorder
    assert len(a) == 40
    assert all(ch in "0123456789abcdef" for ch in a)


def _teacher_case(tmp_path, position_ply=5, game_idx=7):
    """A teacher_retention row: black to move (odd ply), with a teacher policy
    aligned to the reconstructed legal_moves order."""
    from scripts.GPU.alphazero.calibration_pool import legal_moves_sha1
    from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
    import json as _json
    case = _write_case_side(tmp_path, "black", position_ply, game_idx=game_idx)
    replay = _json.loads((tmp_path / f"game_{game_idx:06d}.json").read_text())
    state = position_state(replay, position_ply, "black")
    legal = state.legal_moves()
    n = len(legal)
    policy = [1.0 / n] * n                       # uniform teacher policy
    case.update({
        "loss_mode": "teacher_retention",
        "teacher_value": "0.20",                 # side-to-move
        "teacher_policy_json": _json.dumps(policy),
        "teacher_legal_moves_sha1": legal_moves_sha1(legal),
    })
    return case, n


def test_teacher_retention_row_uses_teacher_value_and_policy(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import build_calibration_position
    case, n = _teacher_case(tmp_path)
    rec = build_calibration_position(case, calibration_target=-0.5)
    assert rec.outcome == 0.20                   # teacher_value, NOT through target_in_to_move
    assert len(rec.visit_counts) == n
    assert abs(sum(rec.visit_counts) - 1.0) < 1e-6
    assert rec.to_move == "black"


def test_from_manifest_detects_teacher_schema(tmp_path):
    import csv as _csv
    from scripts.GPU.alphazero.calibration_pool import CalibrationPool
    case, _ = _teacher_case(tmp_path)
    manifest = tmp_path / "v4.csv"
    with manifest.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(case.keys()))
        w.writeheader()
        w.writerow(case)
    pool = CalibrationPool.from_manifest(str(manifest), calibration_target=-0.35)
    assert pool.schema == "teacher_retention"
    assert pool._samples[0].loss_mode == "teacher_retention"
    assert pool._samples[0].teacher_value == 0.20


def test_teacher_policy_length_mismatch_rejected(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import build_calibration_position
    case, n = _teacher_case(tmp_path)
    case["teacher_policy_json"] = json.dumps([1.0 / (n + 1)] * (n + 1))  # wrong length
    with pytest.raises(ValueError, match="length"):
        build_calibration_position(case, calibration_target=-0.5)


def test_teacher_policy_sha1_reorder_rejected(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import build_calibration_position
    case, n = _teacher_case(tmp_path)
    case["teacher_legal_moves_sha1"] = "0" * 40            # same length, wrong hash
    with pytest.raises(ValueError, match="sha1|alignment"):
        build_calibration_position(case, calibration_target=-0.5)


def test_teacher_policy_not_normalized_rejected(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import build_calibration_position
    case, n = _teacher_case(tmp_path)
    case["teacher_policy_json"] = json.dumps([2.0 / n] * n)  # sums to 2.0
    with pytest.raises(ValueError, match="sum|normal"):
        build_calibration_position(case, calibration_target=-0.5)


def test_teacher_value_out_of_range_rejected(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import build_calibration_position
    case, n = _teacher_case(tmp_path)
    case["teacher_value"] = "1.5"
    with pytest.raises(ValueError, match="teacher_value"):
        build_calibration_position(case, calibration_target=-0.5)


def test_hard_value_row_with_teacher_column_rejected(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import build_calibration_sample
    case = _write_case_side(tmp_path, "black", 5, game_idx=9)
    case["loss_mode"] = "hard_value"
    case["teacher_value"] = "0.1"                          # must be blank
    with pytest.raises(ValueError, match="hard_value|blank"):
        build_calibration_sample(case, calibration_target=-0.35)


def test_split_samples_with_modes_builds_mask(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import (
        build_calibration_sample, split_samples_with_modes)
    hard = build_calibration_sample(
        _write_case_side(tmp_path, "black", 5, game_idx=1), calibration_target=-0.35)
    tcase, _ = _teacher_case(tmp_path, position_ply=5, game_idx=2)
    teach = build_calibration_sample(tcase, calibration_target=-0.35)
    records, weights, mask = split_samples_with_modes([hard, teach, hard],
                                                      has_weight_scale=False)
    assert len(records) == 3
    assert mask.tolist() == [0.0, 1.0, 0.0]
    assert mask.dtype == np.float32


def test_calibration_block_includes_teacher_telemetry():
    from scripts.GPU.alphazero.calibration_pool import build_post_opening_calibration_block
    block = build_post_opening_calibration_block(
        config={"enabled": True, "schema": "teacher_retention"},
        enabled=True,
        loss_accumulator={
            "sum_calib_loss": 4.0, "sum_calib_n_drawn": 60,
            "sum_calib_value_pred": 3.0, "steps_done": 10,
            "sum_calib_value_term": 2.0, "sum_calib_policy_ce": 5.0,
            "sum_calib_policy_kl_est": 0.1, "sum_n_teacher_retention": 20,
        },
    )
    np.testing.assert_allclose(block["loss"]["calib_value_term_avg_iter"], 0.2)
    np.testing.assert_allclose(block["loss"]["calib_policy_ce_avg_iter"], 0.5)
    np.testing.assert_allclose(block["loss"]["calib_policy_kl_est_avg_iter"], 0.01)
    assert block["loss"]["n_teacher_retention_drawn"] == 20


def test_retention_policy_loss_modes_registry():
    from scripts.GPU.alphazero.calibration_pool import (
        RETENTION_POLICY_LOSS_MODES, VALID_LOSS_MODES)
    assert RETENTION_POLICY_LOSS_MODES == frozenset({"teacher_retention", "mcts_root_retention"})
    # v6 adds searched_continuation_retention as a fourth valid (teacher-mode)
    # loss mode; it is intentionally NOT in RETENTION_POLICY_LOSS_MODES (see
    # tests/test_calibration_pool_continuation.py::test_mode_sets).
    # v12 adds asymmetric_guardrail_retention as a fifth valid, value-only
    # loss mode; also intentionally NOT in RETENTION_POLICY_LOSS_MODES (see
    # tests/test_asymmetric_guardrail_pool.py::test_guardrail_mode_registered_value_only).
    assert VALID_LOSS_MODES == frozenset({
        "hard_value", "teacher_retention", "mcts_root_retention",
        "searched_continuation_retention", "asymmetric_guardrail_retention"})


def _sample_with_mode(loss_mode):
    """Direct CalibrationSample construction (no manifest parse needed) to test
    the mask predicate in isolation."""
    from scripts.GPU.alphazero.calibration_pool import CalibrationSample
    from scripts.GPU.alphazero.self_play import PositionRecord
    import numpy as _np
    rec = PositionRecord(
        board_tensor=_np.zeros((24, 24, 30), dtype=_np.float32), to_move="black",
        legal_moves=[(0, 0), (1, 1)], visit_counts=[0.5, 0.5], outcome=0.1,
        active_size=24, ply=5, game_n_moves=None)
    return CalibrationSample(record=rec, loss_mode=loss_mode)


def test_split_samples_with_modes_masks_all_retention_modes():
    from scripts.GPU.alphazero.calibration_pool import split_samples_with_modes
    samples = [_sample_with_mode("hard_value"),
               _sample_with_mode("teacher_retention"),
               _sample_with_mode("mcts_root_retention")]
    _, _, mask = split_samples_with_modes(samples, has_weight_scale=False)
    assert mask.tolist() == [0.0, 1.0, 1.0]   # root rows MUST be 1.0 (the v5 make-or-break)
    assert mask.dtype.name == "float32"


def test_unknown_loss_mode_rejected(tmp_path):
    case = _write_case_side(tmp_path, "black", 5)          # existing helper in this file
    case["loss_mode"] = "typo_mode"
    from scripts.GPU.alphazero.calibration_pool import build_calibration_position
    with pytest.raises(ValueError, match="loss_mode"):
        build_calibration_position(case, calibration_target=-0.35)


def _root_case(tmp_path, **overrides):
    """A valid mcts_root_retention case dict with matching sha1/policy computed
    from the actually reconstructed position."""
    import json as _json
    from scripts.GPU.alphazero.calibration_pool import legal_moves_sha1
    from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
    from tests.goal_line_probe_fixtures import legal_replay
    replay = legal_replay(9, game_idx=1)
    rp = tmp_path / "game_000001.json"
    rp.write_text(_json.dumps(replay))
    state = position_state(replay, 5, "black")
    legal = state.legal_moves()
    n = len(legal)
    case = {
        "game_idx": "1", "case_id": "root1", "replay_path": str(rp),
        "position_ply": "5", "side_to_move": "black",
        "tag": "old_post_opening_retention", "weight_scale": "1.0",
        "loss_mode": "mcts_root_retention",
        "teacher_value": "0.2",
        "root_visits_json": _json.dumps([1.0 / n] * n),
        "root_legal_moves_sha1": legal_moves_sha1(legal),
    }
    case.update(overrides)
    return case


def test_root_retention_row_parses(tmp_path):
    from scripts.GPU.alphazero.calibration_pool import build_calibration_sample
    sample = build_calibration_sample(_root_case(tmp_path), calibration_target=-0.35)
    assert sample.loss_mode == "mcts_root_retention"
    rec = sample.record
    assert abs(rec.outcome - 0.2) < 1e-9                       # raw teacher anchor, stm, DIRECT
    assert len(rec.visit_counts) == len(rec.legal_moves)       # dense root policy
    assert abs(sum(rec.visit_counts) - 1.0) < 1e-6
    assert abs(sample.teacher_value - 0.2) < 1e-9              # metadata reused


def test_root_retention_requires_teacher_value(tmp_path):
    import pytest as _pytest
    from scripts.GPU.alphazero.calibration_pool import build_calibration_sample
    with _pytest.raises(ValueError, match="teacher_value"):
        build_calibration_sample(_root_case(tmp_path, teacher_value=""),
                                 calibration_target=-0.35)


def test_root_retention_rejects_bad_policy(tmp_path):
    import json as _json
    import pytest as _pytest
    from scripts.GPU.alphazero.calibration_pool import build_calibration_sample
    # wrong length
    with _pytest.raises(ValueError, match="length"):
        build_calibration_sample(_root_case(tmp_path, root_visits_json=_json.dumps([1.0])),
                                 calibration_target=-0.35)
    # bad sum
    base = _root_case(tmp_path)
    n = len(_json.loads(base["root_visits_json"]))
    with _pytest.raises(ValueError, match="normal"):
        build_calibration_sample(_root_case(tmp_path, root_visits_json=_json.dumps([2.0 / n] * n)),
                                 calibration_target=-0.35)
    # sha1 mismatch (alignment)
    with _pytest.raises(ValueError, match="alignment|sha1"):
        build_calibration_sample(_root_case(tmp_path, root_legal_moves_sha1="0" * 40),
                                 calibration_target=-0.35)


def test_hard_value_rejects_populated_root_columns(tmp_path):
    import pytest as _pytest
    from scripts.GPU.alphazero.calibration_pool import build_calibration_sample
    case = _root_case(tmp_path, loss_mode="hard_value", target_black_value="-0.35",
                      teacher_value="")
    # root_visits_json / root_legal_moves_sha1 still populated -> must fail loudly
    with _pytest.raises(ValueError, match="blank"):
        build_calibration_sample(case, calibration_target=-0.35)


def test_root_retention_rejects_populated_teacher_policy(tmp_path):
    import json as _json
    import pytest as _pytest
    from scripts.GPU.alphazero.calibration_pool import build_calibration_sample
    case = _root_case(tmp_path, teacher_policy_json=_json.dumps([0.5, 0.5]))
    with _pytest.raises(ValueError, match="teacher_policy_json"):
        build_calibration_sample(case, calibration_target=-0.35)


def test_from_manifest_detects_root_schema_and_rejects_mixed(tmp_path):
    import csv as _csv
    import pytest as _pytest
    from scripts.GPU.alphazero.calibration_pool import CalibrationPool
    root = _root_case(tmp_path)
    hard = dict(_root_case(tmp_path), case_id="corr1", loss_mode="hard_value",
                teacher_value="", root_visits_json="", root_legal_moves_sha1="",
                target_black_value="-0.35")
    cols = sorted(set(root) | set(hard))
    man = tmp_path / "v5.csv"
    with man.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader(); w.writerows([root, hard])
    pool = CalibrationPool.from_manifest(str(man), calibration_target=-0.35)
    assert pool.schema == "mcts_root_retention"

    # mixed retention modes in one manifest -> loud error
    teacher = dict(_root_case(tmp_path), case_id="t1", loss_mode="teacher_retention")
    with (tmp_path / "mixed.csv").open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=sorted(set(root) | set(teacher)), restval="")
        w.writeheader(); w.writerows([root, teacher])
    with _pytest.raises(ValueError, match="mixes"):
        CalibrationPool.from_manifest(str(tmp_path / "mixed.csv"), calibration_target=-0.35)
