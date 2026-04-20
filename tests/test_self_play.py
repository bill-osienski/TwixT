#!/usr/bin/env python3
"""Tests for AlphaZero self-play game generation.

Run with: python3 tests/test_self_play.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_position_record():
    """Test PositionRecord serialization round-trip."""
    import numpy as np
    from scripts.GPU.alphazero.self_play import PositionRecord

    # Create a position record with NHWC format (H, W, C) = (24, 24, 24)
    board = np.random.randn(24, 24, 24).astype(np.float32)
    record = PositionRecord(
        board_tensor=board,
        to_move="red",
        legal_moves=[(1, 1), (2, 2), (3, 3)],
        visit_counts=[100, 50, 25],
        outcome=1.0,
    )

    # Round-trip through dict
    d = record.to_dict()
    restored = PositionRecord.from_dict(d)

    assert restored.to_move == "red"
    assert restored.legal_moves == [(1, 1), (2, 2), (3, 3)]
    assert restored.visit_counts == [100, 50, 25]
    assert restored.outcome == 1.0
    assert np.allclose(restored.board_tensor, board)

    print("PASS: PositionRecord serialization")


def test_game_record():
    """Test GameRecord serialization round-trip."""
    import numpy as np
    from scripts.GPU.alphazero.self_play import PositionRecord, GameRecord

    # Create a game with 2 positions
    pos1 = PositionRecord(
        board_tensor=np.zeros((24, 24, 24), dtype=np.float32),
        to_move="red",
        legal_moves=[(5, 5)],
        visit_counts=[100],
        outcome=1.0,
    )
    pos2 = PositionRecord(
        board_tensor=np.ones((24, 24, 24), dtype=np.float32),
        to_move="black",
        legal_moves=[(6, 6)],
        visit_counts=[80],
        outcome=-1.0,
    )

    game = GameRecord(
        positions=[pos1, pos2],
        winner="red",
        n_moves=2,
        move_history=[(5, 5), (6, 6)],
    )

    # Round-trip
    d = game.to_dict()
    restored = GameRecord.from_dict(d)

    assert restored.winner == "red"
    assert restored.n_moves == 2
    assert len(restored.positions) == 2
    assert restored.positions[0].to_move == "red"
    assert restored.positions[1].to_move == "black"
    assert restored.move_history == [(5, 5), (6, 6)]

    print("PASS: GameRecord serialization")


def test_play_single_game():
    """Test playing a single self-play game."""
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.mcts import MCTSConfig
    from scripts.GPU.alphazero.self_play import play_game

    # Create small network for speed
    network = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(network)

    # Play one game with low simulations for speed
    config = MCTSConfig(n_simulations=10)
    game = play_game(
        evaluator,
        mcts_config=config,
        max_moves=20,  # Short game
        add_noise=True,
    )

    # Basic checks
    assert game.n_moves > 0, "Game should have at least 1 move"
    # With mirror augmentation (default 50%), positions >= n_moves
    assert len(game.positions) >= game.n_moves, "Position count should be >= move count"
    assert game.winner in ("red", "black", None), f"Invalid winner: {game.winner}"

    # Check position records
    for i, pos in enumerate(game.positions):
        assert pos.to_move in ("red", "black"), f"Position {i}: invalid to_move"
        assert len(pos.legal_moves) > 0, f"Position {i}: no legal moves"
        assert len(pos.visit_counts) == len(pos.legal_moves), f"Position {i}: count mismatch"
        assert pos.outcome is not None, f"Position {i}: outcome not set"
        assert pos.outcome in (-1.0, 0.0, 1.0), f"Position {i}: invalid outcome"

    # Check outcomes are consistent with winner
    for pos in game.positions:
        if game.winner is None:
            assert pos.outcome == 0.0, "Draw should have outcome 0"
        elif game.winner == pos.to_move:
            assert pos.outcome == 1.0, "Winner position should have outcome +1"
        else:
            assert pos.outcome == -1.0, "Loser position should have outcome -1"

    print(f"PASS: Single game ({game.n_moves} moves, winner={game.winner})")


def test_play_multiple_games():
    """Test playing multiple games with seeded RNG."""
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.mcts import MCTSConfig
    from scripts.GPU.alphazero.self_play import play_games

    network = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(network)
    config = MCTSConfig(n_simulations=10)

    # Play 3 games with seed
    games = play_games(
        evaluator,
        n_games=3,
        mcts_config=config,
        seed=42,
        max_moves=15,
        add_noise=True,
    )

    assert len(games) == 3, f"Expected 3 games, got {len(games)}"

    # All games should have positions and valid outcomes
    for i, game in enumerate(games):
        assert game.n_moves > 0, f"Game {i}: no moves"
        assert len(game.positions) > 0, f"Game {i}: no positions"
        assert game.winner in ("red", "black", None), f"Game {i}: invalid winner"

    print(f"PASS: Multiple games (3 games generated)")


def test_reproducibility():
    """Test that same seed produces same game."""
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.mcts import MCTSConfig
    from scripts.GPU.alphazero.self_play import play_games

    network = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(network)
    config = MCTSConfig(n_simulations=10)

    # Play same game twice with same seed
    games1 = play_games(
        evaluator,
        n_games=1,
        mcts_config=config,
        seed=12345,
        max_moves=10,
        add_noise=False,  # No noise for reproducibility
    )

    games2 = play_games(
        evaluator,
        n_games=1,
        mcts_config=config,
        seed=12345,
        max_moves=10,
        add_noise=False,
    )

    # Should produce identical games
    g1, g2 = games1[0], games2[0]
    assert g1.n_moves == g2.n_moves, "Move counts should match"
    assert g1.winner == g2.winner, "Winners should match"
    assert g1.move_history == g2.move_history, "Move histories should match"

    print("PASS: Reproducibility with same seed")


def test_to_move_explicit():
    """Test that to_move is stored explicitly, not inferred from ply."""
    import os
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.mcts import MCTSConfig
    from scripts.GPU.alphazero.self_play import play_game
    import scripts.GPU.alphazero.self_play as _sp

    network = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(network)
    config = MCTSConfig(n_simulations=10)

    # Disable mirror augmentation so positions alternate 1:1 with plies
    saved_mirror = _sp._MIRROR_PROB
    _sp._MIRROR_PROB = 0.0
    try:
        game = play_game(
            evaluator,
            mcts_config=config,
            max_moves=10,
            add_noise=False,
            start_player="red",
        )
    finally:
        _sp._MIRROR_PROB = saved_mirror

    # Check to_move alternates correctly (red starts)
    expected_player = "red"
    for i, pos in enumerate(game.positions):
        assert pos.to_move == expected_player, (
            f"Position {i}: expected {expected_player}, got {pos.to_move}"
        )
        expected_player = "black" if expected_player == "red" else "red"

    print("PASS: to_move stored explicitly and alternates correctly")


def test_visit_counts_raw():
    """Test that visit counts are raw (not normalized)."""
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.mcts import MCTSConfig
    from scripts.GPU.alphazero.self_play import play_game

    network = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(network)
    config = MCTSConfig(n_simulations=50)  # More sims for meaningful counts

    game = play_game(
        evaluator,
        mcts_config=config,
        max_moves=5,
        add_noise=True,
    )

    # Check first position has visit counts that sum to ~n_simulations
    pos = game.positions[0]
    total_visits = sum(pos.visit_counts)

    # Should be close to n_simulations (may be slightly less due to terminal expansions)
    assert total_visits >= config.n_simulations * 0.5, (
        f"Total visits {total_visits} too low for {config.n_simulations} simulations"
    )
    assert total_visits <= config.n_simulations * 1.5, (
        f"Total visits {total_visits} too high"
    )

    # Counts should be integers (raw, not normalized)
    for count in pos.visit_counts:
        assert isinstance(count, int), f"Visit count should be int, got {type(count)}"
        assert count >= 0, "Visit count should be non-negative"

    print(f"PASS: Visit counts are raw integers (total={total_visits})")


def test_adjudication_wiring():
    """Test that adjudication parameters are accepted and produce valid outcomes."""
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.mcts import MCTSConfig
    from scripts.GPU.alphazero.self_play import play_game, ADJUDICATED, DRAW_TIMEOUT

    network = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(network)
    config = MCTSConfig(n_simulations=10)

    # Play with adjudication enabled + permissive gates
    # Random untrained network may or may not cross threshold, so we test
    # structural correctness, not "must adjudicate"
    game = play_game(
        evaluator,
        mcts_config=config,
        max_moves=10,
        add_noise=False,
        adjudicate_enabled=True,
        adjudicate_min_ply=0,
        adjudicate_threshold=0.10,  # Very loose
        adjudicate_min_visits=1,
        adjudicate_min_top1_share=0.0,
    )

    # Game should complete without error
    assert game.n_moves > 0
    assert game.winner in ("red", "black", None)

    # Validate structural correctness based on what actually happened
    if game.draw_reason == ADJUDICATED:
        # Adjudicated: must have a decisive winner and +/-1 outcomes
        assert game.winner in ("red", "black"), "Adjudicated game must have a winner"
        for pos in game.positions:
            assert pos.outcome in (1.0, -1.0), f"Adjudicated game should have +/-1 outcomes, got {pos.outcome}"
    elif game.draw_reason == DRAW_TIMEOUT:
        # Timeout draw: winner must be None, outcomes must be 0.0
        assert game.winner is None, "Timeout draw should have no winner"
        for pos in game.positions:
            assert pos.outcome == 0.0, f"Timeout draw should have 0.0 outcomes, got {pos.outcome}"
    else:
        # Early terminal win is also valid (game ended before cap)
        assert game.winner in ("red", "black", None)

    print(f"PASS: Adjudication wiring (moves={game.n_moves}, winner={game.winner}, reason={game.draw_reason})")


def test_adjudication_disabled_by_default():
    """Test that adjudication is off by default (no change to existing behavior)."""
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    from scripts.GPU.alphazero.mcts import MCTSConfig
    from scripts.GPU.alphazero.self_play import play_game, ADJUDICATED

    network = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(network)
    config = MCTSConfig(n_simulations=10)

    # Play without adjudication (default)
    game = play_game(
        evaluator,
        mcts_config=config,
        max_moves=10,
        add_noise=False,
    )

    # Should never have adjudicated reason when disabled
    assert game.draw_reason != ADJUDICATED, "Adjudication should not fire when disabled"

    print(f"PASS: Adjudication disabled by default (reason={game.draw_reason})")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SELF-PLAY TESTS")
    print("=" * 60)
    print()

    tests = [
        test_position_record,
        test_game_record,
        test_play_single_game,
        test_play_multiple_games,
        test_reproducibility,
        test_to_move_explicit,
        test_visit_counts_raw,
        test_adjudication_wiring,
        test_adjudication_disabled_by_default,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()  # passes if no exception; assertions raise on failure
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("Gate PASSED: Self-play generates valid training data")
        return 0
    else:
        print("Gate FAILED: Self-play tests failed")
        return 1


def test_position_record_has_ply_and_game_n_moves():
    """After play_game, each position carries ply + game_n_moves."""
    import numpy as np
    from scripts.GPU.alphazero.self_play import PositionRecord
    from scripts.GPU.alphazero.game.twixt_state import NUM_CHANNELS
    # Check dataclass has the new fields
    pos = PositionRecord(
        board_tensor=np.zeros((24, 24, NUM_CHANNELS), dtype=np.float32),
        to_move="red", legal_moves=[(0, 0)], visit_counts=[1], active_size=24,
        ply=5, game_n_moves=100,
    )
    assert pos.ply == 5
    assert pos.game_n_moves == 100


if __name__ == "__main__":
    sys.exit(main())
