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


def test_game_complete_has_new_optional_fields_with_none_defaults():
    """GameComplete IPC message gains final_root_value, final_top1_share."""
    from scripts.GPU.alphazero.ipc_messages import GameComplete

    msg = GameComplete(
        worker_id=0,
        winner="red",
        draw_reason=0,
        n_moves=10,
        n_positions=10,
        wall_time_s=1.5,
        nn_calls=100,
        expand_calls=100,
        nn_batches=10,
        total_backups=100,
        total_waiters=0,
        unique_leaves=100,
        max_waiters=0,
        flush_full=0,
        flush_stall=0,
        flush_tail=0,
    )

    assert hasattr(msg, "final_root_value")
    assert hasattr(msg, "final_top1_share")
    assert msg.final_root_value is None
    assert msg.final_top1_share is None


def test_save_record_with_all_new_fields_populated(tmp_path):
    """save_game_replay writes all new fields under meta in the documented schema."""
    import json
    from scripts.GPU.alphazero.game_saver import save_game_replay

    filepath = save_game_replay(
        games_dir=tmp_path,
        iteration=12,
        game_idx=3,
        winner="red",
        move_history=((0, 0), (1, 1), (2, 2)),
        n_moves=3,
        active_size=24,
        simulations=200,
        start_player="red",
        # New per-game stats kwargs
        worker_id=2,
        wall_time_s=14.27,
        adjudication_block_reason="ply",
        final_root_value=0.83,
        final_top1_share=0.62,
        leaf_evals=17400,
        backups=17400,
        nn_batches=850,
    )

    record = json.loads(filepath.read_text())
    meta = record["meta"]

    # New flat diagnostic fields
    assert meta["worker_id"] == 2
    assert meta["wall_time_s"] == 14.27
    assert meta["adjudication_block_reason"] == "ply"
    assert meta["final_root_value"] == 0.83
    assert meta["final_top1_share"] == 0.62

    # New compute block
    assert meta["compute"] == {"leaf_evals": 17400, "backups": 17400, "nn_batches": 850}

    # Pre-existing meta keys still present and unchanged
    for key in ("board_size", "mode", "reason", "iteration", "game_idx",
               "simulations", "n_moves", "starting_player"):
        assert key in meta, f"pre-existing meta key {key!r} missing"
    assert meta["iteration"] == 12
    assert meta["game_idx"] == 3


def test_save_record_with_no_new_fields_uses_safe_defaults(tmp_path):
    """When new kwargs are unspecified, compute is zeros and flat fields are null."""
    import json
    from scripts.GPU.alphazero.game_saver import save_game_replay

    filepath = save_game_replay(
        games_dir=tmp_path,
        iteration=0,
        game_idx=0,
        winner=None,
        move_history=((0, 0),),
        n_moves=1,
    )

    meta = json.loads(filepath.read_text())["meta"]

    assert meta["compute"] == {"leaf_evals": 0, "backups": 0, "nn_batches": 0}
    assert meta["worker_id"] is None
    assert meta["wall_time_s"] is None
    assert meta["adjudication_block_reason"] is None
    assert meta["final_root_value"] is None
    assert meta["final_top1_share"] is None


def test_compute_counter_none_coerces_to_zero(tmp_path):
    """leaf_evals=None / backups=None / nn_batches=None must coerce to 0, not crash."""
    import json
    from scripts.GPU.alphazero.game_saver import save_game_replay

    filepath = save_game_replay(
        games_dir=tmp_path,
        iteration=0,
        game_idx=0,
        winner=None,
        move_history=((0, 0),),
        n_moves=1,
        leaf_evals=None,
        backups=None,
        nn_batches=None,
    )

    meta = json.loads(filepath.read_text())["meta"]
    assert meta["compute"] == {"leaf_evals": 0, "backups": 0, "nn_batches": 0}


def test_float_zero_preserved_distinct_from_null(tmp_path):
    """wall_time_s=0.0 and final_root_value=0.0 must be preserved as 0.0, not null.

    Catches `or 0.0` truthiness regressions on floats. final_top1_share=None
    is used because 0.0 is outside the documented (0, 1] range.
    """
    import json
    from scripts.GPU.alphazero.game_saver import save_game_replay

    filepath = save_game_replay(
        games_dir=tmp_path,
        iteration=0,
        game_idx=0,
        winner=None,
        move_history=((0, 0),),
        n_moves=1,
        wall_time_s=0.0,
        final_root_value=0.0,
        final_top1_share=None,
    )

    meta = json.loads(filepath.read_text())["meta"]
    assert meta["wall_time_s"] == 0.0
    assert meta["final_root_value"] == 0.0
    assert meta["final_top1_share"] is None


def _make_saver(tmp_path):
    """Construct a fresh GameSaver bound to tmp_path for routing tests."""
    from scripts.GPU.alphazero.game_saver import GameSaver

    saver = GameSaver(
        games_dir=tmp_path,
        max_games_per_iter=10,
        simulations=200,
        active_size=24,
    )
    saver.set_iteration(0)
    return saver


def test_save_game_from_ipc_routes_all_new_fields(tmp_path):
    """_save_game_from_ipc translates all GameComplete fields onto save kwargs."""
    import json
    from scripts.GPU.alphazero.trainer import _save_game_from_ipc
    from scripts.GPU.alphazero.ipc_messages import GameComplete

    saver = _make_saver(tmp_path)

    msg = GameComplete(
        worker_id=2,
        winner="red",
        draw_reason=0,           # 0 → None (no draw)
        n_moves=3,
        n_positions=3,
        wall_time_s=14.27,
        nn_calls=17400,
        expand_calls=17400,
        nn_batches=850,
        total_backups=17400,
        total_waiters=0,
        unique_leaves=17400,
        max_waiters=0,
        flush_full=0,
        flush_stall=0,
        flush_tail=0,
        move_history=((0, 0), (1, 1), (2, 2)),
        start_player="red",
        adj_blocked_by="ply",
        final_root_value=0.83,
        final_top1_share=0.62,
    )

    filepath = _save_game_from_ipc(saver, msg)
    assert filepath is not None

    meta = json.loads(filepath.read_text())["meta"]
    assert meta["worker_id"] == 2
    assert meta["wall_time_s"] == 14.27
    assert meta["adjudication_block_reason"] == "ply"   # adj_blocked_by → adjudication_block_reason
    assert meta["final_root_value"] == 0.83
    assert meta["final_top1_share"] == 0.62
    assert meta["compute"] == {
        "leaf_evals": 17400,                            # nn_calls → leaf_evals
        "backups": 17400,                               # total_backups → backups
        "nn_batches": 850,
    }


def test_save_game_from_ipc_handles_optional_fields_as_null(tmp_path):
    """When GameComplete optional fields are at defaults, JSON has nulls and zeros."""
    import json
    from scripts.GPU.alphazero.trainer import _save_game_from_ipc
    from scripts.GPU.alphazero.ipc_messages import GameComplete

    saver = _make_saver(tmp_path)

    msg = GameComplete(
        worker_id=0,
        winner="draw",
        draw_reason=1,
        n_moves=2,
        n_positions=2,
        wall_time_s=0.5,
        nn_calls=0,
        expand_calls=0,
        nn_batches=0,
        total_backups=0,
        total_waiters=0,
        unique_leaves=0,
        max_waiters=0,
        flush_full=0,
        flush_stall=0,
        flush_tail=0,
        move_history=((0, 0), (1, 1)),
        start_player="red",
        # adj_blocked_by, final_root_value, final_top1_share at defaults (None)
    )

    filepath = _save_game_from_ipc(saver, msg)
    assert filepath is not None

    meta = json.loads(filepath.read_text())["meta"]
    assert meta["adjudication_block_reason"] is None
    assert meta["final_root_value"] is None
    assert meta["final_top1_share"] is None
    assert meta["compute"] == {"leaf_evals": 0, "backups": 0, "nn_batches": 0}


def test_save_game_from_record_routes_all_new_fields(tmp_path):
    """_save_game_from_record translates all GameRecord fields onto save kwargs."""
    import json
    from scripts.GPU.alphazero.trainer import _save_game_from_record
    from scripts.GPU.alphazero.self_play import GameRecord

    saver = _make_saver(tmp_path)

    game = GameRecord(
        positions=[],
        winner="black",
        n_moves=3,
        move_history=[(0, 0), (1, 1), (2, 2)],
        start_player="red",
        nn_calls=17400,
        nn_batches=850,
        total_backups=17400,
        adj_blocked_by="threshold",
        wall_time_s=12.5,
        final_root_value=-0.41,
        final_top1_share=0.55,
    )

    filepath = _save_game_from_record(saver, game)
    assert filepath is not None

    meta = json.loads(filepath.read_text())["meta"]
    assert meta["worker_id"] is None    # in-process path has no worker
    assert meta["wall_time_s"] == 12.5
    assert meta["adjudication_block_reason"] == "threshold"
    assert meta["final_root_value"] == -0.41
    assert meta["final_top1_share"] == 0.55
    assert meta["compute"] == {
        "leaf_evals": 17400,
        "backups": 17400,
        "nn_batches": 850,
    }
