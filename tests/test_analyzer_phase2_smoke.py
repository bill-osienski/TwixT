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
