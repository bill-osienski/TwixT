import json

import numpy as np
import pytest

from scripts.GPU.alphazero.build_mcts_root_retention_manifest import (
    build_rows, cross_check_gate_values, dense_normalized_visits,
    output_fieldnames, row_seed)
from scripts.GPU.alphazero.calibration_pool import (
    build_calibration_sample, legal_moves_sha1)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from tests.goal_line_probe_fixtures import legal_replay


class _FakeRawEval:
    """Eval-mode raw-anchor stand-in (LocalGPUEvaluator API)."""
    def build_input_tensor(self, state):
        return state.to_tensor()
    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        b, m = move_mask.shape
        priors = move_mask / np.maximum(move_mask.sum(axis=1, keepdims=True), 1.0)
        return priors.astype(np.float32), np.full((b,), 0.2, dtype=np.float32)


def _fake_search(state, seed):
    """Deterministic fake gate search: all visits on the first legal move except one."""
    legal = state.legal_moves()
    counts = {m: 0 for m in legal}
    counts[legal[0]] = 399
    counts[legal[-1]] = 1
    return counts, -0.1389        # root value, stm


def _rows(tmp_path):
    rp = tmp_path / "game_000001.json"
    rp.write_text(json.dumps(legal_replay(9, game_idx=1)))
    return [
        {"game_idx": "1", "case_id": "corr1", "replay_path": str(rp),
         "position_ply": "5", "side_to_move": "black",
         "tag": "black_predrop_correction", "target_black_value": "-0.35",
         "weight_scale": "1.0"},
        {"game_idx": "1", "case_id": "game_000001_ply_005", "replay_path": str(rp),
         "position_ply": "5", "side_to_move": "black",
         "tag": "old_post_opening_retention", "target_black_value": "-0.11",
         "weight_scale": "1.0"},
    ]


def _build(tmp_path):
    return build_rows(_rows(tmp_path), _FakeRawEval(), _fake_search,
                      sims=400, base_checkpoint="ckpt/base.safetensors",
                      pos_base_seed=20260616, goal_base_seed=20260614,
                      eval_batch_size=14, stall_flush_sims=48)


def test_row_seed_matches_gate_probe_schemes():
    # position-probe families: base ^ game ^ ply (eval_position_probe.py:76)
    assert row_seed("old_post_opening_retention", 7, 51, 20260616, 20260614) == (20260616 ^ 7 ^ 51)
    assert row_seed("red_predrop_retention", 7, 51, 20260616, 20260614) == (20260616 ^ 7 ^ 51)
    # goal-line family: base ^ game only (eval_goal_line_trigger_probe.py:69)
    assert row_seed("goal_line_retention", 7, 51, 20260616, 20260614) == (20260614 ^ 7)


def test_dense_normalized_visits_aligned_and_zero_total_rejected(tmp_path):
    rp = tmp_path / "game_000001.json"
    rp.write_text(json.dumps(legal_replay(9, game_idx=1)))
    state = position_state(legal_replay(9, game_idx=1), 5, "black")
    legal = state.legal_moves()
    counts = {m: 0 for m in legal}
    counts[legal[0]] = 3
    counts[legal[1]] = 1
    dense = dense_normalized_visits(counts, legal, "c1")
    assert len(dense) == len(legal)
    assert dense[0] == pytest.approx(0.75) and dense[1] == pytest.approx(0.25)
    assert sum(dense) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="zero"):
        dense_normalized_visits({m: 0 for m in legal}, legal, "c1")


def test_builder_blanks_correction_and_fills_root_retention(tmp_path):
    out = _build(tmp_path)
    corr = next(r for r in out if r["case_id"] == "corr1")
    ret = next(r for r in out if r["case_id"] == "game_000001_ply_005")
    assert corr["loss_mode"] == "hard_value"
    assert corr["teacher_value"] == "" and corr["root_visits_json"] == ""
    assert corr["root_legal_moves_sha1"] == "" and corr["root_value_stm"] == ""
    assert corr["target_black_value"] == "-0.35"          # A hard target PRESERVED

    assert ret["loss_mode"] == "mcts_root_retention"
    assert abs(float(ret["teacher_value"]) - 0.2) < 1e-6  # raw eval-mode anchor
    assert abs(float(ret["root_value_stm"]) - (-0.1389)) < 1e-9
    assert abs(float(ret["root_black_value"]) - (-0.1389)) < 1e-9  # black to move: no flip
    assert ret["target_black_value"] == ""                # stale v3 MCTS scalar blanked
    policy = json.loads(ret["root_visits_json"])
    assert abs(sum(policy) - 1.0) < 1e-6
    assert max(policy) == pytest.approx(399 / 400)
    # provenance stamps
    assert ret["root_sims"] == "400"
    assert ret["root_base_checkpoint"] == "ckpt/base.safetensors"
    assert ret["root_seed"] == str(20260616 ^ 1 ^ 5)
    assert ret["root_mcts_eval_batch_size"] == "14"
    assert ret["root_mcts_stall_flush_sims"] == "48"
    # sha1 matches the actually reconstructed legal order
    state = position_state(legal_replay(9, game_idx=1), 5, "black")
    assert ret["root_legal_moves_sha1"] == legal_moves_sha1(state.legal_moves())


def test_builder_output_passes_v5_parser(tmp_path):
    out = _build(tmp_path)
    ret = next(r for r in out if r["loss_mode"] == "mcts_root_retention")
    sample = build_calibration_sample(ret, calibration_target=-0.35)
    assert sample.loss_mode == "mcts_root_retention"
    assert abs(sample.record.outcome - 0.2) < 1e-6        # value = RAW anchor, not root value
    assert abs(sum(sample.record.visit_counts) - 1.0) < 1e-6


def test_cross_check_gate_values(tmp_path):
    out = _build(tmp_path)
    gate_csv = tmp_path / "position_probe_cases.csv"
    gate_csv.write_text(
        "checkpoint,case_id,probe_black_root_value\n"
        "calib020_0001,game_000001_ply_005,-0.1389\n")
    stats = cross_check_gate_values(out, [str(gate_csv)], tol=1e-3)
    assert stats["checked"] == 1 and stats["unmatched"] == 0

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        "checkpoint,case_id,probe_black_root_value\n"
        "calib020_0001,game_000001_ply_005,0.9\n")
    with pytest.raises(ValueError, match="cross-check"):
        cross_check_gate_values(out, [str(bad_csv)], tol=1e-3)


def test_cross_check_rejects_multi_checkpoint_without_label(tmp_path):
    out = _build(tmp_path)
    multi_csv = tmp_path / "multi_checkpoint.csv"
    multi_csv.write_text(
        "checkpoint,case_id,probe_black_root_value\n"
        "0001,game_000001_ply_005,-0.1389\n"
        "0002,game_000001_ply_005,0.9\n")
    with pytest.raises(ValueError, match="ambiguous"):
        cross_check_gate_values(out, [str(multi_csv)], tol=1e-3)


def test_cross_check_filters_by_checkpoint_label(tmp_path):
    out = _build(tmp_path)
    multi_csv = tmp_path / "multi_checkpoint.csv"
    multi_csv.write_text(
        "checkpoint,case_id,probe_black_root_value\n"
        "0001,game_000001_ply_005,-0.1389\n"
        "0002,game_000001_ply_005,0.9\n")
    stats = cross_check_gate_values(out, [str(multi_csv)], tol=1e-3,
                                    checkpoint_label="0001")
    assert stats == {"checked": 1, "unmatched": 0}

    suffix_csv = tmp_path / "suffix_checkpoint.csv"
    suffix_csv.write_text(
        "checkpoint,case_id,probe_black_root_value\n"
        "alphazero-v2:0001,game_000001_ply_005,-0.1389\n")
    stats = cross_check_gate_values(out, [str(suffix_csv)], tol=1e-3,
                                    checkpoint_label="0001")
    assert stats == {"checked": 1, "unmatched": 0}

    with pytest.raises(ValueError):
        cross_check_gate_values(out, [str(multi_csv)], tol=1e-3,
                                checkpoint_label="9999")


def test_output_fieldnames_includes_builder_added_columns(tmp_path):
    out = _build(tmp_path)
    fields = output_fieldnames(["game_idx", "case_id"], out)
    assert fields[:2] == ["game_idx", "case_id"]
    assert fields.count("target_black_value") == 1


def test_builder_module_defers_heavy_imports():
    """MLX/MCTS must not load at import time (tests run with fakes)."""
    import importlib
    # NOTE: never pop mlx from sys.modules here — re-importing the native module later in the same process aborts (Metal re-init).
    m = importlib.import_module(
        "scripts.GPU.alphazero.build_mcts_root_retention_manifest")
    src = open(m.__file__).read()
    head = src.split("def ", 1)[0]                 # module-level import block
    assert "eval_runner" not in head and "local_evaluator" not in head
    assert "probe_eval" not in head
