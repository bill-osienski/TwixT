import pytest

from scripts.GPU.alphazero.eval_runner import (
    EvalGameTask, EvalGameResult, EvalConfig,
    cfg_from, play_eval_game, make_result, run_game_tasks,
)
from tests.eval_fakes import FakeEvaluator, fake_evaluator_factory, counting_factory


def _tiny_cfg(**kw):
    base = dict(board_size=8, mcts_sims=8, mcts_eval_batch_size=4,
                mcts_stall_flush_sims=4, opening_temp_plies=4,
                temp_high=1.0, temp_low=0.1, max_moves=12)
    base.update(kw)
    return EvalConfig(**base)


def test_cfg_from_maps_fields():
    cfg = cfg_from(_tiny_cfg())
    assert cfg.n_simulations == 8
    assert cfg.eval_batch_size == 4
    assert cfg.stall_flush_sims == 4
    assert cfg.temp_threshold_ply == 4
    assert cfg.temp_high == 1.0 and cfg.temp_low == 0.1


def test_cfg_from_argmax_zeroes_temps():
    cfg = cfg_from(_tiny_cfg(selection_mode="argmax"))
    assert cfg.temp_high == 0.0 and cfg.temp_low == 0.0


def test_cfg_from_rejects_unknown_mode():
    with pytest.raises(ValueError):
        cfg_from(_tiny_cfg(selection_mode="bogus"))


def test_play_eval_game_is_deterministic_by_seed():
    cfg = _tiny_cfg()
    r1 = play_eval_game(FakeEvaluator(), FakeEvaluator(), cfg, seed=123)
    r2 = play_eval_game(FakeEvaluator(), FakeEvaluator(), cfg, seed=123)
    assert r1 == r2


def test_play_eval_game_reason_is_valid():
    winner, reason, n, _records = play_eval_game(FakeEvaluator(), FakeEvaluator(),
                                                 _tiny_cfg(), seed=1)
    assert reason in {"win", "state_cap", "board_full", "unknown_error"}
    assert reason != "unknown_error"
    assert n >= 1


def test_make_result_red_win_credits_red_checkpoint():
    task = EvalGameTask(0, "p", 0, "A.safetensors", "B.safetensors", 7)
    res = make_result(task, "red", "win", 40)
    assert res.winner_checkpoint == "A.safetensors"
    assert res.red_score == 1.0 and res.black_score == 0.0


def test_make_result_black_win_credits_black_checkpoint():
    task = EvalGameTask(0, "p", 1, "B.safetensors", "A.safetensors", 7)
    res = make_result(task, "black", "win", 40)
    assert res.winner_checkpoint == "A.safetensors"
    assert res.red_score == 0.0 and res.black_score == 1.0


def test_make_result_state_cap_is_half_each():
    task = EvalGameTask(0, "p", 0, "A.safetensors", "B.safetensors", 7)
    res = make_result(task, None, "state_cap", 12)
    assert res.winner_checkpoint is None
    assert res.red_score == 0.5 and res.black_score == 0.5


def test_run_game_tasks_workers1_sorted_and_complete():
    tasks = [
        EvalGameTask(5, "p", 5, "A", "B", 105),
        EvalGameTask(0, "p", 0, "A", "B", 100),
        EvalGameTask(2, "p", 2, "B", "A", 102),
    ]
    out = run_game_tasks(tasks, workers=1, config=_tiny_cfg(),
                         evaluator_factory=fake_evaluator_factory)
    assert [r.game_idx for r in out] == [0, 2, 5]  # sorted by (pairing_id, game_idx)
    assert len(out) == 3


def test_run_game_tasks_empty_returns_empty():
    assert run_game_tasks([], workers=4, config=_tiny_cfg(),
                          evaluator_factory=fake_evaluator_factory) == []


def test_worker_cache_loads_each_checkpoint_once_sequential():
    counting_factory.calls.clear()
    tasks = [
        EvalGameTask(0, "p", 0, "A", "B", 200),
        EvalGameTask(1, "p", 1, "B", "A", 201),  # reuses A and B
        EvalGameTask(2, "p", 2, "A", "B", 202),
    ]
    run_game_tasks(tasks, workers=1, config=_tiny_cfg(),
                   evaluator_factory=counting_factory)
    assert counting_factory.calls == {"A": 1, "B": 1}


def test_play_eval_game_capture_off_returns_none():
    *_head, records = play_eval_game(FakeEvaluator(), FakeEvaluator(),
                                     _tiny_cfg(), seed=1)
    assert records is None


def test_play_eval_game_capture_records_one_per_ply():
    winner, reason, n, records = play_eval_game(
        FakeEvaluator(), FakeEvaluator(), _tiny_cfg(), seed=1, capture=True)
    assert records is not None
    assert len(records) == n
    assert [r["ply"] for r in records] == list(range(n))
    players = [r["player"] for r in records]
    assert players[0] == "red"                                  # red moves first
    assert all(players[i] != players[i + 1] for i in range(len(players) - 1))
    for r in records:
        assert set(r) == {"ply", "player", "row", "col", "root_value",
                          "root_top1_share", "selected_visit_rank",
                          "selected_visit_count", "root_total_visits", "n_legal"}
        assert 0.0 <= r["root_top1_share"] <= 1.0
        assert r["selected_visit_rank"] >= 1
        assert r["selected_visit_count"] >= 1
        assert r["n_legal"] >= 1


def test_play_eval_game_capture_does_not_change_outcome():
    off = play_eval_game(FakeEvaluator(), FakeEvaluator(), _tiny_cfg(), seed=7)
    on = play_eval_game(FakeEvaluator(), FakeEvaluator(), _tiny_cfg(),
                        seed=7, capture=True)
    assert off[:3] == on[:3]   # winner, reason, n_moves identical regardless of capture


def test_make_result_default_replay_path_is_none():
    task = EvalGameTask(0, "p", 0, "A.safetensors", "B.safetensors", 7)
    assert make_result(task, "red", "win", 40).replay_path is None


def test_make_result_sets_replay_path():
    task = EvalGameTask(0, "p", 0, "A.safetensors", "B.safetensors", 7)
    res = make_result(task, "red", "win", 40, "logs/x_replays/game_000000.json")
    assert res.replay_path == "logs/x_replays/game_000000.json"
