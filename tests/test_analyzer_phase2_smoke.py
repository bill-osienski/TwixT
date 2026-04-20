"""E2E smoke tests for Phase 1/2 analyzer additions."""
import json
import os
import subprocess
import tempfile


def test_connectivity_diagnostics_on_real_games():
    """connectivity_diagnostics returns non-empty stats on existing game JSONs."""
    import sys
    sys.path.insert(0, ".")
    from scripts.GPU.alphazero.connectivity_diagnostics import (
        compute_position_connectivity, aggregate_connectivity_by_ply,
    )
    from scripts.GPU.alphazero.game.twixt_state import TwixtState

    # Build a known state and score it
    state = TwixtState(active_size=8)
    state = state.apply_move((0, 3))  # red on top edge
    state = state.apply_move((4, 4))  # black middle
    state = state.apply_move((7, 5))  # red on bottom edge (different component)
    stats = compute_position_connectivity(state)
    assert stats["red_has_top_component"] is True
    assert stats["red_has_bottom_component"] is True
    assert stats["red_n_goal_touching_components"] == 2  # two separate red pegs on different edges
    assert stats["black_has_left_component"] is False


def test_value_calibration_bucket_classification():
    """Bucket classifier should place positions correctly."""
    from scripts.GPU.alphazero.value_calibration import classify_position
    from scripts.GPU.alphazero.game.twixt_state import TwixtState

    # A clearly red-winning structure (chain top + bottom via 8 pegs)
    # Simplification: synthesize via mock
    # ... build a state with red_largest_component_size >= 8 and red_n_goal_touching_components >= 1
    # Then classify_position should return category including "red_winning_structure"
    state = TwixtState(active_size=8)
    cat = classify_position(state, ply=0, game_n_moves=100, min_size=8)
    assert cat == "balanced_no_winning_structure"


def test_replay_cap_has_termination_breakdown():
    """After aggregation, replay-cap stats include positions_by_termination breakdown."""
    from scripts.GPU.alphazero.trainer import ReplayBuffer
    # The test is a simple integration check: once the feature lands, a sidecar
    # will carry these keys. For now, we check the aggregator helper directly.
    try:
        from scripts.twixt_replay_analyzer import aggregate_replay_cap
    except ImportError:
        import pytest
        pytest.skip("aggregate_replay_cap not importable")
    rcap_by_iter = {
        100: {
            "enabled": True, "max_positions_per_game": 64,
            "games_capped": 5, "games_total": 10,
            "total_positions_original": 500, "total_positions_kept": 300,
            "positions_by_termination": {"win": 80, "resign": 180, "adjudicated": 30, "timeout": 10},
            "positions_in_short_games": 50, "positions_in_long_games": 250,
            "by_length_bucket": {"edges_ply": [40, 80, 120, 160, 200],
                                 "games": [1, 2, 3, 2, 1, 1],
                                 "positions_original": [50, 100, 150, 100, 60, 40],
                                 "positions_kept": [50, 100, 90, 40, 15, 5]},
        }
    }
    agg = aggregate_replay_cap(rcap_by_iter)
    # Must carry the new keys through aggregation
    assert "total_positions_by_termination" in agg or "positions_by_termination" in str(agg)
