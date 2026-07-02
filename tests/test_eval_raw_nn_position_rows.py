import csv
import json

import numpy as np
import pytest

from scripts.GPU.alphazero import eval_raw_nn_position_rows as R
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from tests.goal_line_probe_fixtures import legal_replay


class _FakeEval:
    """Deterministic stand-in for LocalGPUEvaluator: uniform priors + fixed value. No MCTS."""

    def __init__(self, value=0.2):
        self._value = value

    def build_input_tensor(self, state):
        return state.to_tensor()

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        b, m = move_mask.shape
        priors = move_mask / np.maximum(move_mask.sum(axis=1, keepdims=True), 1.0)
        values = np.full((b,), self._value, dtype=np.float32)
        return priors.astype(np.float32), values


def _replay_file(tmp_path, n=9, game_idx=1):
    rp = tmp_path / f"game_{game_idx:06d}.json"
    rp.write_text(json.dumps(legal_replay(n, game_idx=game_idx)))
    return rp


def _case(rp, case_id, ply, side, **extra):
    base = {
        "game_idx": "1", "case_id": case_id, "replay_path": str(rp),
        "position_ply": str(ply), "side_to_move": side,
    }
    base.update(extra)
    return base


def test_to_black_flips_red_to_move():
    assert R.to_black(0.7, "black") == pytest.approx(0.7)
    assert R.to_black(0.7, "red") == pytest.approx(-0.7)
    with pytest.raises(ValueError):
        R.to_black(0.1, "green")


def test_score_row_red_to_move_flips_black_value(tmp_path):
    rp = _replay_file(tmp_path)
    row = R.score_row(_FakeEval(value=0.2), _case(rp, "red1", 4, "red"))  # 4 moves -> red to move
    assert row["raw_value_stm"] == pytest.approx(0.2)
    assert row["raw_black_value"] == pytest.approx(-0.2)     # red-to-move: black = -stm
    assert row["overvalue"] is False and row["severe_overvalue"] is False


def test_score_row_black_overvalue_flags_and_top1(tmp_path):
    rp = _replay_file(tmp_path)
    row = R.score_row(_FakeEval(value=0.6), _case(rp, "b1", 5, "black"))  # 5 moves -> black
    assert row["raw_black_value"] == pytest.approx(0.6)
    assert row["overvalue"] is True and row["severe_overvalue"] is True   # 0.6 >= 0.50
    assert ":" in row["top1_move"] and 0.0 < row["top1_prob"] <= 1.0


def test_raw_nn_rows_scores_with_local_evaluator(tmp_path):
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator

    rp = _replay_file(tmp_path)
    net = create_network(hidden=64, n_blocks=2)
    net.eval()
    ev = LocalGPUEvaluator(net)
    row = R.score_row(ev, _case(rp, "ret1", 5, "black"))
    # score_row must apply NO transform to the stm value beyond the shared infer wrapper.
    state = position_state(legal_replay(9, game_idx=1), 5, "black")
    _, _, value = R._teacher_infer(state, ev)   # _teacher_infer wraps evaluator.infer (the "direct infer")
    assert row["raw_value_stm"] == pytest.approx(value, abs=1e-6)
    assert row["raw_black_value"] == pytest.approx(value, abs=1e-6)   # black-to-move: no flip


def test_score_row_side_to_move_mismatch_raises(tmp_path):
    rp = _replay_file(tmp_path)
    with pytest.raises(ValueError, match="side_to_move"):
        R.score_row(_FakeEval(), _case(rp, "bad", 4, "black"))   # ply 4 -> red; claims black
