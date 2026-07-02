import json

import numpy as np
import pytest

from scripts.GPU.alphazero.build_searched_continuation_retention_manifest import (
    NEW_COLUMNS_V6, build_rows_v6, classify_row)
from scripts.GPU.alphazero.calibration_pool import (
    CONTINUATION_LOSS_MODE, build_calibration_sample, legal_moves_sha1)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move
from tests.goal_line_probe_fixtures import legal_replay


class _FakeRawEval:
    def build_input_tensor(self, state):
        return state.to_tensor()
    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        b, m = move_mask.shape
        priors = move_mask / np.maximum(move_mask.sum(axis=1, keepdims=True), 1.0)
        return priors.astype(np.float32), np.full((b,), -0.25, dtype=np.float32)


def _child(parent, move_rc, visits, nn_value=0.1):
    node = MCTSNode(state=parent.state.apply_move(move_rc), parent=parent,
                    move=encode_move(*move_rc), visit_count=visits,
                    nn_value=nn_value, priors={})
    parent.children[node.move] = node
    return node


def _fake_search(state, seed):
    """Deterministic tree: PV chain depth 3 + one sibling. Root value -0.1389 stm."""
    root = MCTSNode(state=state, visit_count=400, priors={})
    legal = state.legal_moves()
    a = _child(root, legal[0], 300, nn_value=-0.4)
    _child(root, legal[1], 99, nn_value=0.2)
    b = _child(a, a.state.legal_moves()[0], 200, nn_value=0.3)
    _child(b, b.state.legal_moves()[0], 120, nn_value=-0.2)
    counts = {m: 0 for m in legal}
    counts[legal[0]] = 300
    counts[legal[1]] = 99
    return counts, -0.1389, root


def _rows(tmp_path):
    rp = tmp_path / "game_000001.json"
    replay = legal_replay(9, game_idx=1)
    rp.write_text(json.dumps(replay))
    state = position_state(replay, 5, "black")
    legal = state.legal_moves()
    dense = [0.0] * len(legal); dense[0] = 300 / 399; dense[1] = 99 / 399
    common = {"replay_path": str(rp), "position_ply": "5", "side_to_move": "black",
              "weight_scale": "1.0"}
    return [
        {"game_idx": "1", "case_id": "corr1",
         "tag": "black_predrop_correction", "loss_mode": "hard_value",
         "target_black_value": "-0.35", **common},
        {"game_idx": "1", "case_id": "game_000001_ply_005",
         "tag": "old_post_opening_retention", "loss_mode": "mcts_root_retention",
         "teacher_value": "-0.11", "target_black_value": "",
         "root_visits_json": json.dumps(dense),
         "root_legal_moves_sha1": legal_moves_sha1(legal),
         "root_value_stm": "-0.1389", "root_black_value": "-0.1389",
         "root_sims": "400", "root_seed": str(20260616 ^ 1 ^ 5),
         "root_base_checkpoint": "ckpt/base.safetensors",
         "root_mcts_eval_batch_size": "14", "root_mcts_stall_flush_sims": "48",
         **common},
    ]


def _build(tmp_path, **kw):
    params = dict(pos_base_seed=20260616, goal_base_seed=20260614,
                  b_pv_depth=2, c_pv_depth=3, d_top_k=3, d_child_pv_depth=1,
                  d_child_pv_min_visits=40, max_per_root=6, max_total=250,
                  emit_policy=False, source_root_tolerance=1e-3,
                  limit_cases=None, only_case_ids=None)
    params.update(kw)
    return build_rows_v6(_rows(tmp_path), _FakeRawEval(), _fake_search, **params)


def test_classify_row():
    assert classify_row({"loss_mode": "hard_value",
                         "tag": "black_predrop_correction"}) == "passthrough"
    assert classify_row({"loss_mode": "mcts_root_retention",
                         "tag": "old_post_opening_retention"}) == "extract"
    assert classify_row({"loss_mode": CONTINUATION_LOSS_MODE,
                         "tag": "old_post_opening_continuation_retention"}) == "passthrough"
    with pytest.raises(ValueError, match="unknown"):
        classify_row({"loss_mode": "mcts_root_retention", "tag": "mystery"})


def test_passthrough_and_continuation_rows(tmp_path):
    out, stats = _build(tmp_path)
    # source rows unchanged and first, C continuations appended after their parent
    assert out[0]["case_id"] == "corr1" and out[0]["loss_mode"] == "hard_value"
    assert out[1]["case_id"] == "game_000001_ply_005"
    assert out[1]["root_visits_json"] != ""            # untouched passthrough
    conts = out[2:]
    assert len(conts) == 3                             # C family: PV depth 1-3
    for depth, row in enumerate(conts, start=1):
        assert row["loss_mode"] == CONTINUATION_LOSS_MODE
        assert row["tag"] == "old_post_opening_continuation_retention"
        assert row["continuation_parent_case_id"] == "game_000001_ply_005"
        assert row["continuation_source"] == "pv"
        assert int(row["continuation_depth"]) == depth
        assert row["teacher_value_source"] == "base_raw_continuation"
        assert abs(float(row["teacher_value"]) - (-0.25)) < 1e-6   # fresh eval fwd
        assert row["target_black_value"] == "" and row["root_visits_json"] == ""
        assert row["teacher_policy_json"] == ""
        assert row["weight_scale"] == "1.0"
        moves = json.loads(row["extra_moves_json"])
        assert len(moves) == depth
    # tree provenance recorded
    assert int(conts[0]["continuation_tree_visits"]) == 300
    assert abs(float(conts[0]["continuation_tree_nn_value"]) - (-0.4)) < 1e-9
    assert stats["n_continuation"] == 3
    assert stats["by_tag"]["old_post_opening_continuation_retention"] == 3


def test_continuation_rows_load_through_the_pool(tmp_path):
    out, _ = _build(tmp_path)
    for row in out[2:]:
        sample = build_calibration_sample(row, calibration_target=-0.35)
        assert sample.loss_mode == CONTINUATION_LOSS_MODE
        assert sample.has_policy_target is False
        assert sample.record.outcome == pytest.approx(-0.25)


def test_case_ids_unique_and_deterministic(tmp_path):
    out1, _ = _build(tmp_path)
    out2, _ = _build(tmp_path)
    assert out1 == out2                                # byte-identical rebuild
    ids = [r["case_id"] for r in out1]
    assert len(ids) == len(set(ids))


def test_source_root_value_mismatch_fails(tmp_path):
    rows = _rows(tmp_path)
    rows[1]["root_black_value"] = "0.9"                # stored v5 value disagrees
    with pytest.raises(ValueError, match="source root value"):
        build_rows_v6(rows, _FakeRawEval(), _fake_search,
                      pos_base_seed=20260616, goal_base_seed=20260614,
                      b_pv_depth=2, c_pv_depth=3,
                      d_top_k=3, d_child_pv_depth=1, d_child_pv_min_visits=40,
                      max_per_root=6, max_total=250, emit_policy=False,
                      source_root_tolerance=1e-3, limit_cases=None,
                      only_case_ids=None)


def test_total_cap_hard_fails(tmp_path):
    with pytest.raises(ValueError, match="max_total"):
        _build(tmp_path, max_total=2)


def test_emit_policy_writes_normalized_teacher_policy(tmp_path):
    out, _ = _build(tmp_path, emit_policy=True)
    row = out[2]
    policy = json.loads(row["teacher_policy_json"])
    assert abs(sum(policy) - 1.0) < 1e-6
    sample = build_calibration_sample(row, calibration_target=-0.35)
    assert sample.has_policy_target is True


def test_limit_and_only_case_id_filters(tmp_path):
    out, stats = _build(tmp_path, limit_cases=0)
    assert stats["n_continuation"] == 0 and len(out) == 2   # passthrough only
    out, stats = _build(tmp_path, only_case_ids={"game_000001_ply_005"})
    assert stats["n_continuation"] == 3
    out, stats = _build(tmp_path, only_case_ids={"nope"})
    assert stats["n_continuation"] == 0


def test_module_defers_heavy_imports():
    import importlib
    # NOTE: never pop mlx from sys.modules — native re-import SIGABRTs.
    m = importlib.import_module(
        "scripts.GPU.alphazero.build_searched_continuation_retention_manifest")
    head = open(m.__file__).read().split("def ", 1)[0]
    assert "eval_runner" not in head and "local_evaluator" not in head
    assert "probe_eval" not in head and "import mlx" not in head
