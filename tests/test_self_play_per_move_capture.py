"""Tests for per-move root-value and top1-share capture during self-play (spec 2026-05-03 §5).

Note on test fidelity: the plan-supplied test code referenced a `SelfPlayConfig`
class and a `play_game(state, mcts, cfg, ...)` signature that do not exist in
this repository. The actual `play_game` in scripts/GPU/alphazero/self_play.py
takes an evaluator + kwargs (it constructs the MCTS and the starting TwixtState
internally from `active_size`). These tests preserve the original assertions
(GameRecord has the two new fields, both length-equal to move_history on the
normal and resign code paths) but invoke `play_game` via its real signature.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_in_process_play_game_returns_per_move_lists_aligned_with_history():
    """play_game's GameRecord has move_root_values and move_top1_shares
    aligned with move_history (same length, no None except where MCTS produced None)."""
    import random
    import numpy as np
    import mlx.core as mx
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.mcts import MCTSConfig
    from scripts.GPU.alphazero.self_play import play_game

    np.random.seed(7)
    mx.random.seed(7)
    net = create_network(hidden=32, n_blocks=2)
    evaluator = LocalGPUEvaluator(net)

    mcts_config = MCTSConfig(n_simulations=20)
    rng = random.Random(7)

    record = play_game(
        evaluator,
        mcts_config=mcts_config,
        rng=rng,
        max_moves=16,
        add_noise=False,
        active_size=8,
        game_id=0,
    )

    assert hasattr(record, "move_root_values"), "GameRecord must carry move_root_values"
    assert hasattr(record, "move_top1_shares"), "GameRecord must carry move_top1_shares"
    n = len(record.move_history)
    assert len(record.move_root_values) == n, (
        f"move_root_values length {len(record.move_root_values)} != move_history {n}"
    )
    assert len(record.move_top1_shares) == n
    for v in record.move_root_values:
        assert v is None or isinstance(v, float)
    for v in record.move_top1_shares:
        assert v is None or (isinstance(v, float) and 0.0 < v <= 1.0)


def test_resign_path_does_not_append_phantom_per_move_entries():
    """When the loser resigns, the per-move accumulators must remain length-equal
    to move_history (no phantom entry for the resign-decision ply)."""
    import random
    import numpy as np
    import mlx.core as mx
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.mcts import MCTSConfig
    from scripts.GPU.alphazero.self_play import play_game

    np.random.seed(13)
    mx.random.seed(13)
    net = create_network(hidden=32, n_blocks=2)
    evaluator = LocalGPUEvaluator(net)

    mcts_config = MCTSConfig(n_simulations=10)
    rng = random.Random(13)

    record = play_game(
        evaluator,
        mcts_config=mcts_config,
        rng=rng,
        max_moves=32,
        add_noise=False,
        active_size=8,
        game_id=0,
        # Lenient resign so it may trigger in a small synthetic game; the test
        # only cares about list/history alignment, not whether resign actually fires.
        resign_enabled=True,
        resign_threshold=-0.2,
        resign_min_ply=1,
        resign_min_visits=1,
        resign_min_top1_share=0.0,
        resign_k=1,
        resign_window=1,
    )

    # The critical invariant: even if resign fired (or didn't), the lists
    # are length-equal to move_history.
    assert len(record.move_root_values) == len(record.move_history)
    assert len(record.move_top1_shares) == len(record.move_history)
