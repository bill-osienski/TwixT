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


def test_ipc_game_complete_pickle_roundtrip_preserves_per_move_lists():
    """GameComplete pickle/unpickle preserves move_root_values + move_top1_shares."""
    import pickle
    from scripts.GPU.alphazero.ipc_messages import GameComplete

    msg = GameComplete(
        worker_id=0,
        winner="red",
        draw_reason=0,
        n_moves=3,
        n_positions=3,
        wall_time_s=1.5,
        nn_calls=10,
        expand_calls=10,
        nn_batches=1,
        total_backups=10,
        total_waiters=0,
        unique_leaves=10,
        max_waiters=0,
        flush_full=0,
        flush_stall=0,
        flush_tail=0,
        move_history=((0, 1), (5, 5), (1, 2)),
        start_player="red",
        move_root_values=(0.1, -0.2, 0.9),
        move_top1_shares=(0.4, 0.18, 0.77),
    )
    rt = pickle.loads(pickle.dumps(msg))
    assert rt.move_root_values == (0.1, -0.2, 0.9)
    assert rt.move_top1_shares == (0.4, 0.18, 0.77)


def test_save_game_from_record_writes_per_move_fields(tmp_path):
    """End-to-end: GameRecord with per-move lists -> saved JSON has populated fields."""
    import json
    from scripts.GPU.alphazero.game_saver import GameSaver
    from scripts.GPU.alphazero.self_play import GameRecord
    from scripts.GPU.alphazero.trainer import _save_game_from_record

    record = GameRecord(
        positions=[],
        winner="red",
        n_moves=3,
        move_history=[(0, 1), (5, 5), (1, 2)],
        start_player="red",
        draw_reason=None,
        resigned_by=None,
        nn_calls=10,
        nn_batches=1,
        total_backups=10,
        adj_blocked_by=None,
        opening_diagnostics=[],
        opening_diagnostics_meta=None,
        wall_time_s=1.0,
        final_root_value=0.7,
        final_top1_share=0.5,
        move_root_values=[0.1, 0.2, 0.3],
        move_top1_shares=[0.4, 0.5, 0.6],
    )
    saver = GameSaver(games_dir=tmp_path, max_games_per_iter=5, simulations=400, active_size=8)
    saver.set_iteration(7)
    _save_game_from_record(saver, record)

    saved = json.loads((tmp_path / "iter_0007_game_000.json").read_text())
    assert saved["moves"][0]["search_score"] == 0.1
    assert saved["moves"][2]["root_top1_share"] == 0.6
