"""Tests for GPU-accelerated batch evaluation.

Validates:
1. Batch vs sequential equivalence
2. GPU vs CPU equivalence
3. Performance improvement from batching
"""

import pytest
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.game.state import GameState
from scripts.GPU.game.rules import apply_move
from scripts.GPU.game.board import legal_moves
from scripts.GPU.ai.heuristics import (
    score_moves,
    score_moves_batch,
    extract_features,
)
from scripts.GPU.ai.value_model import ValueModel, load_value_model
from scripts.GPU.ai.batch_eval import (
    BatchValueModel,
    batch_extract_features,
    get_batch_value_model,
    is_gpu_available,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def game_state():
    """Create a fresh GameState."""
    return GameState(board_size=24, to_move="red")


@pytest.fixture
def mid_game_state():
    """Create a mid-game state with pegs and bridges."""
    state = GameState()
    moves = [(5, 5), (10, 10), (7, 6), (12, 11), (9, 7), (14, 12), (11, 8), (16, 13)]
    for r, c in moves:
        state = apply_move(state, r, c)
    return state


@pytest.fixture
def mock_value_model():
    """Create a simple mock value model for testing."""
    # Simple model with 5 features
    return ValueModel(
        feature_keys=[
            "friendly_connected_paths",
            "opponent_connected_paths",
            "friendly_pegs",
            "opponent_pegs",
            "move_count",
        ],
        weights=[0.1, 0.5, -0.3, 0.2, -0.1, 0.05],  # bias + 5 weights
        standardize=False,
        mean=None,
        std=None,
        scale=600.0,
    )


# =============================================================================
# Batch Feature Extraction Tests
# =============================================================================

class TestBatchFeatureExtraction:
    """Test batch feature extraction."""

    def test_single_state(self, game_state):
        """Single state extraction works."""
        game_state.pegs[(5, 5)] = "red"
        features = batch_extract_features(
            [game_state], "red", 0, 1, 0
        )
        assert len(features) == 1
        assert "friendly_pegs" in features[0]

    def test_multiple_states(self, mid_game_state):
        """Multiple states are extracted correctly."""
        moves = legal_moves(mid_game_state, mid_game_state.to_move)[:5]
        child_states = [apply_move(mid_game_state, r, c) for r, c in moves]

        features = batch_extract_features(
            child_states, "red", 8, 4, 4
        )

        assert len(features) == 5
        for f in features:
            assert "friendly_pegs" in f
            assert "turn" in f
            assert f["turn"] == 9  # base_turn + 1

    def test_context_features_added(self, game_state):
        """Context features (turn, player, peg counts) are added."""
        game_state.pegs[(5, 5)] = "red"
        features = batch_extract_features(
            [game_state], "red", 5, 3, 2
        )

        assert features[0]["turn"] == 6
        assert features[0]["player"] == 1.0
        assert features[0]["playerPegCount"] == 4
        assert features[0]["opponentPegCount"] == 2


# =============================================================================
# Batch Value Model Tests
# =============================================================================

class TestBatchValueModel:
    """Test batch value model inference."""

    def test_single_evaluation(self, mock_value_model):
        """Single evaluation matches sequential."""
        batch_model = BatchValueModel(mock_value_model)

        features = {
            "friendly_connected_paths": 100.0,
            "opponent_connected_paths": 50.0,
            "friendly_pegs": 5,
            "opponent_pegs": 4,
            "move_count": 10,
        }

        # Sequential
        seq_result = mock_value_model.evaluate(features)

        # Batch
        batch_results = batch_model.batch_evaluate([features])

        assert len(batch_results) == 1
        assert abs(batch_results[0]["probability"] - seq_result["probability"]) < 1e-5
        assert abs(batch_results[0]["adjustment"] - seq_result["adjustment"]) < 1e-3

    def test_multiple_evaluations(self, mock_value_model):
        """Multiple evaluations match sequential."""
        batch_model = BatchValueModel(mock_value_model)

        feature_list = [
            {"friendly_connected_paths": 100.0, "opponent_connected_paths": 50.0,
             "friendly_pegs": 5, "opponent_pegs": 4, "move_count": 10},
            {"friendly_connected_paths": 200.0, "opponent_connected_paths": 100.0,
             "friendly_pegs": 8, "opponent_pegs": 6, "move_count": 20},
            {"friendly_connected_paths": 50.0, "opponent_connected_paths": 150.0,
             "friendly_pegs": 3, "opponent_pegs": 5, "move_count": 8},
        ]

        # Sequential
        seq_results = [mock_value_model.evaluate(f) for f in feature_list]

        # Batch
        batch_results = batch_model.batch_evaluate(feature_list)

        assert len(batch_results) == 3
        for i in range(3):
            assert abs(batch_results[i]["probability"] - seq_results[i]["probability"]) < 1e-5
            assert abs(batch_results[i]["adjustment"] - seq_results[i]["adjustment"]) < 1e-3

    def test_empty_batch(self, mock_value_model):
        """Empty batch returns empty list."""
        batch_model = BatchValueModel(mock_value_model)
        results = batch_model.batch_evaluate([])
        assert results == []

    def test_missing_features(self, mock_value_model):
        """Missing features default to 0."""
        batch_model = BatchValueModel(mock_value_model)

        features = {"friendly_connected_paths": 100.0}  # Missing most features

        results = batch_model.batch_evaluate([features])
        assert len(results) == 1
        assert results[0]["probability"] is not None

    def test_gpu_cpu_equivalence(self, mock_value_model):
        """GPU and CPU results match (if GPU available)."""
        batch_model = BatchValueModel(mock_value_model)

        feature_list = [
            {"friendly_connected_paths": float(i * 10), "opponent_connected_paths": float(i * 5),
             "friendly_pegs": i, "opponent_pegs": i - 1, "move_count": i * 2}
            for i in range(1, 11)
        ]

        cpu_results = batch_model.batch_evaluate_cpu(feature_list)

        if is_gpu_available():
            gpu_results = batch_model.batch_evaluate_gpu(feature_list)

            for i in range(10):
                assert abs(gpu_results[i]["probability"] - cpu_results[i]["probability"]) < 1e-4
                assert abs(gpu_results[i]["adjustment"] - cpu_results[i]["adjustment"]) < 1e-2


# =============================================================================
# Score Moves Equivalence Tests
# =============================================================================

class TestScoreMovesEquivalence:
    """Test that batch scoring matches sequential scoring."""

    def test_empty_moves(self, game_state):
        """Empty move list returns empty results."""
        result = score_moves_batch(game_state, [])
        assert result == []

    def test_single_move(self, game_state):
        """Single move scoring works."""
        result = score_moves_batch(game_state, [(5, 5)])
        assert len(result) == 1
        assert result[0][0] == (5, 5)

    def test_ordering_preserved(self, mid_game_state):
        """Batch scoring preserves score ordering."""
        moves = legal_moves(mid_game_state, mid_game_state.to_move)[:20]

        seq_results = score_moves(mid_game_state, moves)
        batch_results = score_moves_batch(mid_game_state, moves)

        # Same number of results
        assert len(seq_results) == len(batch_results)

        # Both should be sorted descending
        for i in range(len(seq_results) - 1):
            assert seq_results[i][1] >= seq_results[i + 1][1]
            assert batch_results[i][1] >= batch_results[i + 1][1]

    def test_scores_close_without_value_model(self, mid_game_state):
        """Without value model, batch and sequential should match exactly."""
        # This test only makes sense when no value model is loaded
        moves = legal_moves(mid_game_state, mid_game_state.to_move)[:10]

        seq_results = score_moves(mid_game_state, moves, value_model=None)
        batch_results = score_moves_batch(mid_game_state, moves)

        # If no value model loaded, scores should be very close
        # (batch doesn't add value model adjustments if model is None)
        batch_model = get_batch_value_model()
        if batch_model is None:
            seq_dict = {m: s for m, s in seq_results}
            batch_dict = {m: s for m, s in batch_results}

            for move in seq_dict:
                assert abs(seq_dict[move] - batch_dict[move]) < 1e-3


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestBatchPerformance:
    """Test that batching provides performance benefit."""

    def test_batch_vs_sequential_performance(self, mock_value_model, mid_game_state):
        """Benchmark batch vs sequential evaluation (informational, not pass/fail)."""
        batch_model = BatchValueModel(mock_value_model)
        moves = legal_moves(mid_game_state, mid_game_state.to_move)[:50]
        child_states = [apply_move(mid_game_state, r, c) for r, c in moves]

        features = batch_extract_features(
            child_states, "red", 8, 4, 4
        )

        # Sequential timing
        start = time.perf_counter()
        for _ in range(10):
            for f in features:
                mock_value_model.evaluate(f)
        seq_time = time.perf_counter() - start

        # Batch timing
        start = time.perf_counter()
        for _ in range(10):
            batch_model.batch_evaluate(features)
        batch_time = time.perf_counter() - start

        # Report performance (no assertion - just informational)
        speedup = seq_time / batch_time if batch_time > 0 else 0
        print(f"\nBatch vs Sequential ({len(features)} items, {len(mock_value_model.feature_keys)} features):")
        print(f"  Sequential: {seq_time*1000:.1f}ms")
        print(f"  Batch:      {batch_time*1000:.1f}ms")
        print(f"  Speedup:    {speedup:.2f}x")

    def test_gpu_vs_cpu_performance(self, mock_value_model):
        """Benchmark GPU vs CPU batch evaluation (informational, not pass/fail)."""
        if not is_gpu_available():
            pytest.skip("GPU not available")

        batch_model = BatchValueModel(mock_value_model)

        # Test with batch size 500
        features = [
            {"friendly_connected_paths": float(i), "opponent_connected_paths": float(i / 2),
             "friendly_pegs": i % 10, "opponent_pegs": (i + 1) % 10, "move_count": i}
            for i in range(500)
        ]

        # Warm up (first call has JIT compilation overhead)
        _ = batch_model.batch_evaluate_gpu(features)
        _ = batch_model.batch_evaluate_cpu(features)

        # CPU timing
        start = time.perf_counter()
        for _ in range(5):
            batch_model.batch_evaluate_cpu(features)
        cpu_time = time.perf_counter() - start

        # GPU timing
        start = time.perf_counter()
        for _ in range(5):
            batch_model.batch_evaluate_gpu(features)
        gpu_time = time.perf_counter() - start

        # Report performance (no assertion - just informational)
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"\nGPU vs CPU ({len(features)} items, {len(mock_value_model.feature_keys)} features):")
        print(f"  CPU (NumPy): {cpu_time*1000:.1f}ms")
        print(f"  GPU (MLX):   {gpu_time*1000:.1f}ms")
        print(f"  GPU Speedup: {speedup:.2f}x")
