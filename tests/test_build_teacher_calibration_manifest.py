import importlib
import json
import numpy as np

from scripts.GPU.alphazero.build_teacher_calibration_manifest import build_rows
from scripts.GPU.alphazero.calibration_pool import (
    build_calibration_sample, legal_moves_sha1)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from tests.goal_line_probe_fixtures import legal_replay


class _FakeEval:
    """Stand-in for LocalGPUEvaluator: deterministic uniform priors + fixed value.
    Records that infer() was called (no MCTS)."""
    def build_input_tensor(self, state):
        return state.to_tensor()
    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        b, m = move_mask.shape
        priors = move_mask / np.maximum(move_mask.sum(axis=1, keepdims=True), 1.0)
        values = np.full((b,), 0.2, dtype=np.float32)
        return priors.astype(np.float32), values


def _rows(tmp_path):
    rp = tmp_path / "game_000001.json"
    rp.write_text(json.dumps(legal_replay(9, game_idx=1)))
    return [
        {"game_idx": "1", "case_id": "corr1", "replay_path": str(rp),
         "position_ply": "5", "side_to_move": "black",
         "tag": "black_predrop_correction", "target_black_value": "-0.35",
         "weight_scale": "1.0"},
        {"game_idx": "1", "case_id": "ret1", "replay_path": str(rp),
         "position_ply": "5", "side_to_move": "black",
         "tag": "old_post_opening_retention", "target_black_value": "-0.11",  # stale MCTS scalar
         "weight_scale": "1.0"},
    ]


def test_builder_blanks_correction_and_fills_retention(tmp_path):
    out = build_rows(_rows(tmp_path), _FakeEval())
    corr = next(r for r in out if r["case_id"] == "corr1")
    ret = next(r for r in out if r["case_id"] == "ret1")
    assert corr["loss_mode"] == "hard_value"
    assert corr["teacher_value"] == "" and corr["teacher_policy_json"] == ""
    assert corr["target_black_value"] == "-0.35"    # correction hard target PRESERVED (not blanked)
    assert ret["loss_mode"] == "teacher_retention"
    assert abs(float(ret["teacher_value"]) - 0.2) < 1e-6
    assert ret["target_black_value"] == ""          # retention-only: stale MCTS scalar blanked
    policy = json.loads(ret["teacher_policy_json"])
    assert abs(sum(policy) - 1.0) < 1e-6


def test_builder_output_passes_parser(tmp_path):
    out = build_rows(_rows(tmp_path), _FakeEval())
    ret = next(r for r in out if r["case_id"] == "ret1")
    # round-trip: the built row must satisfy the v4 loader/validation.
    sample = build_calibration_sample(ret, calibration_target=-0.35)
    assert sample.loss_mode == "teacher_retention"
    assert abs(sample.teacher_value - 0.2) < 1e-6


def test_builder_module_does_not_import_mcts():
    mod = importlib.import_module(
        "scripts.GPU.alphazero.build_teacher_calibration_manifest")
    src = open(mod.__file__).read()
    assert "import mcts" not in src.lower()
    assert "from .mcts" not in src and "MCTS(" not in src


def test_self_distillation_holds_for_matching_teacher(tmp_path):
    import csv as _csv
    from scripts.GPU.alphazero.smoke_teacher_calibration_v4 import assert_self_distillation
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator

    # Build a 1-retention-row manifest whose teacher == THIS network's own outputs.
    rp = tmp_path / "game_000001.json"
    rp.write_text(json.dumps(legal_replay(9, game_idx=1)))
    net = create_network(hidden=64, n_blocks=2)
    net.eval()                                           # cache in eval mode (mirrors builder main())
    rows = [{"game_idx": "1", "case_id": "ret1", "replay_path": str(rp),
             "position_ply": "5", "side_to_move": "black",
             "tag": "old_post_opening_retention", "weight_scale": "1.0"}]
    built = build_rows(rows, LocalGPUEvaluator(net))     # teacher = net itself
    manifest = tmp_path / "v4.csv"
    with manifest.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(built[0].keys()))
        w.writeheader(); w.writerows(built)

    stats = assert_self_distillation(net, str(manifest), tol=1e-3)
    assert abs(stats["value_mse"]) < 1e-3
    assert abs(stats["kl_est"]) < 1e-3


def test_self_distillation_holds_for_multi_position_batch(tmp_path):
    """Multi-position guard: the network uses BatchNorm, so a TRAIN-mode batched
    forward is batch-statistics-dependent. Self-distillation over >1 retention
    row therefore only holds when the teacher cache AND the smoke forward both
    use EVAL mode (running stats, batch-independent). The single-row test above
    cannot catch this (batch==1 is trivially batch-independent)."""
    import csv as _csv
    from scripts.GPU.alphazero.smoke_teacher_calibration_v4 import assert_self_distillation
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator

    net = create_network(hidden=64, n_blocks=2)
    net.eval()                                   # cache uses base running stats (mirrors builder main())
    rp = tmp_path / "game_000001.json"
    rp.write_text(json.dumps(legal_replay(9, game_idx=1)))
    # 3 distinct boards (different plies) + both sides → a real padded batch.
    rows = [
        {"game_idx": "1", "case_id": "r1", "replay_path": str(rp),
         "position_ply": "5", "side_to_move": "black",
         "tag": "old_post_opening_retention", "weight_scale": "1.0"},
        {"game_idx": "1", "case_id": "r2", "replay_path": str(rp),
         "position_ply": "6", "side_to_move": "red",
         "tag": "red_predrop_retention", "weight_scale": "1.0"},
        {"game_idx": "1", "case_id": "r3", "replay_path": str(rp),
         "position_ply": "7", "side_to_move": "black",
         "tag": "goal_line_retention", "weight_scale": "1.0"},
    ]
    built = build_rows(rows, LocalGPUEvaluator(net))     # teacher = net itself
    manifest = tmp_path / "v4_multi.csv"
    with manifest.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(built[0].keys()))
        w.writeheader(); w.writerows(built)

    net.train()                                  # net in TRAIN at smoke time (production-faithful)
    stats = assert_self_distillation(net, str(manifest), tol=1e-4)
    assert abs(stats["value_mse"]) < 1e-4
    assert abs(stats["kl_est"]) < 1e-4
