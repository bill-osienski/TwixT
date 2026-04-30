"""Tests for per-game stats persistence (spec 2026-04-29).

Covers:
  - MCTS final-root instrumentation (final_root_value, final_top1_share)
  - JSON schema written by save_game_replay
  - Trainer routing helpers _save_game_from_ipc / _save_game_from_record
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_mcts_with_small_net(n_simulations: int = 50):
    """Construct a real MCTS with a small MLX net for instrumentation tests.

    Mirrors the pattern in tests/test_mcts.py — use the actual evaluator
    rather than a stub, so we exercise the same code paths self-play uses.
    """
    import random
    import numpy as np
    import mlx.core as mx
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator

    np.random.seed(42)
    mx.random.seed(42)
    net = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(net)
    config = MCTSConfig(n_simulations=n_simulations)
    mcts = MCTS(evaluator, config, rng=random.Random(42))
    return mcts


def test_mcts_capture_final_root_stats_after_search_from_root():
    """search_from_root sets _final_root_value (finite) and _final_top1_share in (0, 1]."""
    from scripts.GPU.alphazero.mcts import MCTSNode
    from scripts.GPU.alphazero.game import TwixtState

    mcts = _make_mcts_with_small_net(n_simulations=50)
    state = TwixtState()
    root = MCTSNode(state=state)

    mcts.search_from_root(root, add_noise=False)

    assert mcts._final_root_value is not None, "final_root_value should be set after search"
    # Helper coerces to Python float — exact-type check is safe.
    assert isinstance(mcts._final_root_value, float)
    # Under MCTS numeric invariants this is finite; we don't assert range
    # tightly because spec keeps the bound informal.
    assert mcts._final_top1_share is not None, "final_top1_share should be set after search"
    assert isinstance(mcts._final_top1_share, float)
    assert 0.0 < mcts._final_top1_share <= 1.0, (
        f"final_top1_share out of range: {mcts._final_top1_share}"
    )


def test_mcts_search_vanilla_also_captures():
    """Vanilla MCTS.search() also populates the final-root snapshot."""
    from scripts.GPU.alphazero.game import TwixtState

    mcts = _make_mcts_with_small_net(n_simulations=50)
    state = TwixtState()

    mcts.search(state, add_noise=False)

    assert mcts._final_root_value is not None
    assert isinstance(mcts._final_root_value, float)
    assert mcts._final_top1_share is not None
    assert isinstance(mcts._final_top1_share, float)
    assert 0.0 < mcts._final_top1_share <= 1.0


def test_mcts_capture_final_root_stats_no_searches_run():
    """Fresh MCTS that never ran a search has both attributes as None."""
    mcts = _make_mcts_with_small_net(n_simulations=10)
    assert mcts._final_root_value is None
    assert mcts._final_top1_share is None


def test_mcts_capture_final_root_stats_zero_visits_returns_none_share():
    """Root with children but zero visits → top1_share is None, root_value still set."""
    from scripts.GPU.alphazero.mcts import MCTSNode
    from scripts.GPU.alphazero.game import TwixtState

    mcts = _make_mcts_with_small_net(n_simulations=10)

    state = TwixtState()
    root = MCTSNode(state=state)
    # Two children with zero visits (degenerate edge of the helper).
    # Note: TwixT forbids corners for first player, so we apply a legal
    # non-corner move to construct the child state. The helper only reads
    # child.visit_count, so the specific child state doesn't matter.
    child_state = state.apply_move((0, 1))
    root.children = {
        0: MCTSNode(state=child_state, parent=root, move=0),
        1: MCTSNode(state=child_state, parent=root, move=1),
    }
    # root.q_value defaults to 0.0 when visit_count == 0 (see MCTSNode.q_value)

    mcts._capture_final_root_stats(root)

    assert mcts._final_top1_share is None
    assert mcts._final_root_value == 0.0  # root.q_value with visit_count==0


def test_game_record_has_new_optional_fields_with_none_defaults():
    """GameRecord schema gains wall_time_s, final_root_value, final_top1_share."""
    from scripts.GPU.alphazero.self_play import GameRecord

    record = GameRecord(positions=[], winner=None, n_moves=0)

    assert hasattr(record, "wall_time_s")
    assert hasattr(record, "final_root_value")
    assert hasattr(record, "final_top1_share")
    assert record.wall_time_s is None
    assert record.final_root_value is None
    assert record.final_top1_share is None
