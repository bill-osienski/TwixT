import pytest

from scripts.GPU.alphazero.eval_runner import EvalGameTask, EvalConfig, run_game_tasks
from scripts.GPU.alphazero.eval_checkpoint_tournament import build_tournament_tasks
from tests.eval_fakes import fake_evaluator_factory, raising_factory


def _tiny_cfg():
    return EvalConfig(board_size=8, mcts_sims=8, mcts_eval_batch_size=4,
                      mcts_stall_flush_sims=4, opening_temp_plies=4,
                      temp_high=1.0, temp_low=0.1, max_moves=12)


def _key(results):
    # Identity + outcome tuple per result, in sorted order, for comparison.
    return [(r.task_id, r.pairing_id, r.game_idx, r.winner, r.reason,
             r.n_moves, r.red_score) for r in results]


def test_workers1_vs_workers2_identical_results():
    tasks = build_tournament_tasks([("A", "B"), ("A", "C")], games=6, base_seed=42)
    seq = run_game_tasks(tasks, workers=1, config=_tiny_cfg(),
                         evaluator_factory=fake_evaluator_factory)
    par = run_game_tasks(tasks, workers=2, config=_tiny_cfg(),
                         evaluator_factory=fake_evaluator_factory)
    assert _key(seq) == _key(par)


def test_parallel_returns_all_results_sorted():
    tasks = build_tournament_tasks([("A", "B")], games=8, base_seed=7)
    out = run_game_tasks(tasks, workers=3, config=_tiny_cfg(),
                         evaluator_factory=fake_evaluator_factory)
    assert len(out) == 8
    assert [r.game_idx for r in out] == sorted(r.game_idx for r in out)


def test_parallel_surfaces_worker_crash():
    # A factory that raises must surface promptly as a RuntimeError, not hang.
    tasks = build_tournament_tasks([("A", "B")], games=4, base_seed=1)
    with pytest.raises(RuntimeError):
        run_game_tasks(tasks, workers=2, config=_tiny_cfg(),
                       evaluator_factory=raising_factory)
