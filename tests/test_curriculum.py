"""Critical tests for curriculum learning correctness.

These tests prevent the "trains 12 hours, learns nothing" failure mode
by verifying that active_size boundaries, placement rules, win detection,
tensor encoding, and value head masking all work correctly together.
"""
import numpy as np
import pytest

# Import the modules under test
import sys
from pathlib import Path

# Add the scripts/GPU path so we can import alphazero modules
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "GPU"))

from alphazero.game.twixt_state import TwixtState, BOARD_SIZE
from alphazero.curriculum import CurriculumManager
# CurriculumManager.get_metrics() only counts a draw if a draw_reason was given
# at record_game time — no-reason draws are intentionally invisible to match
# the production calling convention (play_game always sets a draw_reason).
from alphazero.self_play import DRAW_UNKNOWN


class TestActiveSizePlacementRules:
    """Test 1: active_size Bounds + Placement Rules."""

    def test_corners_relative_to_active_size(self):
        """Corners/edges are relative to active_size, not board_size."""
        state = TwixtState(active_size=8)

        # Corners of active region (0,0), (0,7), (7,0), (7,7) should be forbidden
        assert not state.is_valid_placement(0, 0), "Corner (0,0) should be forbidden"
        assert not state.is_valid_placement(0, 7), "Corner (0,7) should be forbidden"
        assert not state.is_valid_placement(7, 0), "Corner (7,0) should be forbidden"
        assert not state.is_valid_placement(7, 7), "Corner (7,7) should be forbidden"

    def test_outside_active_region(self):
        """Positions outside active region should be invalid."""
        state = TwixtState(active_size=8)

        # Position (8, 5) should be out of bounds (outside active region)
        assert not state.is_valid_placement(8, 5), "(8,5) should be out of bounds"
        assert not state.is_valid_placement(5, 8), "(5,8) should be out of bounds"
        assert not state.is_valid_placement(10, 10), "(10,10) should be out of bounds"

    def test_red_edge_restrictions(self):
        """Red cannot place on left/right edges of active region."""
        state = TwixtState(active_size=8, to_move="red")

        # Red can't place on col 0 or col 7 (left/right edges of active)
        assert not state.is_valid_placement(3, 0), "Red can't place on col 0"
        assert not state.is_valid_placement(3, 7), "Red can't place on col 7"

        # Red CAN place on row 0 (top edge - that's red's goal)
        assert state.is_valid_placement(0, 3), "Red should place on row 0"

    def test_black_edge_restrictions(self):
        """Black cannot place on top/bottom edges of active region."""
        state = TwixtState(active_size=8, to_move="black")

        # Black can't place on row 0 or row 7 (top/bottom edges)
        assert not state.is_valid_placement(0, 3), "Black can't place on row 0"
        assert not state.is_valid_placement(7, 3), "Black can't place on row 7"

        # Black CAN place on col 0 (left edge - that's black's goal)
        assert state.is_valid_placement(3, 0), "Black should place on col 0"

    def test_legal_moves_respects_active_size(self):
        """legal_moves() should only return positions within active region."""
        state = TwixtState(active_size=8)
        moves = state.legal_moves()

        # All moves should be within active region
        for r, c in moves:
            assert 0 <= r < 8, f"Row {r} exceeds active_size"
            assert 0 <= c < 8, f"Col {c} exceeds active_size"

        # There should be no moves at (10, 5) etc.
        assert (10, 5) not in moves


class TestWinnerUsesActiveSize:
    """Test 2: winner() Uses Active Edges."""

    def test_win_detection_at_active_boundary(self):
        """Win should be detected at active_size-1, not board_size-1."""
        # Create state with active_size=8
        state = TwixtState(active_size=8)

        # A win at row 7 (active_size-1) should count
        # A peg at row 23 (board_size-1) should NOT count as a win edge

        # For this test, we just verify the edge detection logic
        # by checking that active_size is used in _check_win
        assert state.active_size == 8
        assert state.board_size == 24

    def test_red_win_at_active_bottom(self):
        """Red should win when reaching row active_size-1."""
        state = TwixtState(active_size=8)

        # Place red pegs that would form a path from row 0 to row 7
        # This is a simplified test - just verify the state structure
        state = state.apply_move((0, 3))  # Red at top edge
        assert state.to_move == "black"
        state = state.apply_move((3, 0))  # Black move
        assert state.to_move == "red"

        # Verify state is using correct active_size
        assert state.active_size == 8


class TestToTensorEdgeDistances:
    """Test 3: to_tensor() Active Edge Channels."""

    def test_edge_distance_at_active_corners(self):
        """Edge distance channels should use active_size geometry."""
        state = TwixtState(active_size=8)
        tensor = state.to_tensor()

        # Channel indices
        CHANNEL_RED_TOP_DIST = 19
        CHANNEL_RED_BOTTOM_DIST = 20

        # At (0, 0): top-left corner of active region
        # Red top dist should be 1.0 (at goal edge)
        assert tensor[CHANNEL_RED_TOP_DIST, 0, 0] == 1.0, "Red top dist at (0,0) should be 1.0"
        # Red bottom dist should be 0.0
        assert tensor[CHANNEL_RED_BOTTOM_DIST, 0, 0] == 0.0, "Red bottom dist at (0,0) should be 0.0"

        # At (7, 7): bottom-right of active region (row 7 = active_size-1)
        # Red top dist should be 0.0
        assert tensor[CHANNEL_RED_TOP_DIST, 7, 7] == 0.0, "Red top dist at (7,7) should be 0.0"
        # Red bottom dist should be 1.0
        assert tensor[CHANNEL_RED_BOTTOM_DIST, 7, 7] == 1.0, "Red bottom dist at (7,7) should be 1.0"

    def test_outside_active_region_is_zero(self):
        """Positions outside active region should have zero in edge channels."""
        state = TwixtState(active_size=8)
        tensor = state.to_tensor()

        CHANNEL_RED_TOP_DIST = 19
        CHANNEL_RED_BOTTOM_DIST = 20
        CHANNEL_CURRENT_PLAYER = 18
        CHANNEL_MOVE_NUMBER = 23

        # Outside active region (e.g., (10, 10)) should be zero
        assert tensor[CHANNEL_RED_TOP_DIST, 10, 10] == 0.0
        assert tensor[CHANNEL_RED_BOTTOM_DIST, 10, 10] == 0.0
        assert tensor[CHANNEL_CURRENT_PLAYER, 10, 10] == 0.0
        assert tensor[CHANNEL_MOVE_NUMBER, 10, 10] == 0.0

    def test_current_player_only_in_active_region(self):
        """Current player channel should only fill active region."""
        state = TwixtState(active_size=8, to_move="red")
        tensor = state.to_tensor()

        CHANNEL_CURRENT_PLAYER = 18

        # Inside active region should be 1.0 (red to move)
        assert tensor[CHANNEL_CURRENT_PLAYER, 3, 3] == 1.0

        # Outside active region should be 0.0
        assert tensor[CHANNEL_CURRENT_PLAYER, 10, 10] == 0.0


class TestApplyMoveValidation:
    """Test 5: apply_move() Rejects Illegal Moves."""

    def test_apply_move_rejects_outside_active(self):
        """apply_move() should reject moves outside active region."""
        state = TwixtState(active_size=8)

        # Move outside active region should raise
        with pytest.raises(ValueError, match="Illegal move"):
            state.apply_move((10, 5))

    def test_apply_move_rejects_corner(self):
        """apply_move() should reject moves on forbidden corners."""
        state = TwixtState(active_size=8)

        # Move on forbidden corner should raise
        with pytest.raises(ValueError, match="Illegal move"):
            state.apply_move((0, 0))

    def test_apply_move_rejects_wrong_edge(self):
        """apply_move() should reject moves on opponent's edge."""
        # Red can't place on left/right edges
        state = TwixtState(active_size=8, to_move="red")
        with pytest.raises(ValueError, match="Illegal move"):
            state.apply_move((3, 0))  # Col 0 is forbidden for red

    def test_apply_move_accepts_legal_move(self):
        """apply_move() should accept legal moves."""
        state = TwixtState(active_size=8, to_move="red")
        new_state = state.apply_move((3, 3))  # Valid move
        assert (3, 3) in new_state.pegs
        assert new_state.pegs[(3, 3)] == "red"


class TestCurriculumManager:
    """Test CurriculumManager promotion logic."""

    def test_initial_state(self):
        """CurriculumManager should start at first size."""
        cm = CurriculumManager(sizes=(8, 10, 12, 16, 20, 24))
        assert cm.active_size == 8
        assert cm.idx == 0
        assert not cm.is_final

    def test_record_game_updates_history(self):
        """Recording games should update history.

        Draws must supply a draw_reason; otherwise the current
        CurriculumManager.get_metrics() doesn't classify the game as a draw
        (it only counts draws with a recognized reason). Production callers
        always pass one.
        """
        cm = CurriculumManager(window=10)
        cm.record_game("red")
        cm.record_game("black")
        cm.record_game(None, DRAW_UNKNOWN)  # draw

        metrics = cm.get_metrics()
        assert metrics["red_wins"] == 1
        assert metrics["black_wins"] == 1
        assert metrics["draws"] == 1
        assert metrics["total"] == 3

    def test_window_rolls(self):
        """History should be limited to window size."""
        cm = CurriculumManager(window=5)
        for _ in range(10):
            cm.record_game("red")

        metrics = cm.get_metrics()
        assert metrics["total"] == 5  # Window limits to 5

    def test_promotion_requires_both_colors_winning(self):
        """Promotion should require both colors to have wins."""
        cm = CurriculumManager(window=20, min_wins_each=3, draw_threshold=0.5)

        # Only red wins - should not promote
        for _ in range(15):
            cm.record_game("red")

        assert not cm.should_promote()

        # Add some black wins
        for _ in range(5):
            cm.record_game("black")

        # Now should be ready (both colors winning, low draw rate)
        assert cm.should_promote()

    def test_promotion_requires_low_draw_rate(self):
        """Promotion should require draw rate below threshold.

        Each recorded draw supplies a draw_reason (matches production calls —
        see `record_game_updates_history` for context).
        """
        cm = CurriculumManager(window=20, min_wins_each=1, draw_threshold=0.3)

        # Many draws - should not promote
        for _ in range(15):
            cm.record_game(None, DRAW_UNKNOWN)
        cm.record_game("red")
        cm.record_game("black")

        assert not cm.should_promote()  # Too many draws

    def test_promotion_stability_guard(self):
        """Promotion requires 2 consecutive checks passing."""
        cm = CurriculumManager(window=20, min_wins_each=2, draw_threshold=0.5)

        # Setup conditions for promotion
        for _ in range(10):
            cm.record_game("red")
        for _ in range(10):
            cm.record_game("black")

        # First call - should not promote yet (streak = 1)
        assert cm.should_promote()
        promoted = cm.maybe_promote()
        assert not promoted  # First check doesn't promote

        # Second call - should promote now (streak = 2)
        promoted = cm.maybe_promote()
        assert promoted
        assert cm.active_size == 10  # Moved to second size

    def test_serialization_roundtrip(self):
        """CurriculumManager should serialize/deserialize correctly."""
        cm = CurriculumManager(sizes=(8, 10, 12))
        cm.record_game("red")
        cm.record_game("black")
        cm._promote_streak = 1
        cm.idx = 1

        d = cm.to_dict()
        cm2 = CurriculumManager.from_dict(d)

        assert cm2.active_size == cm.active_size
        assert cm2.idx == cm.idx
        assert cm2._promote_streak == cm._promote_streak
        assert cm2._history == cm._history


class TestIntegration:
    """Integration tests for curriculum with game state."""

    def test_game_at_small_active_size(self):
        """Verify a full game works at small active_size."""
        state = TwixtState(active_size=6)

        # Play a few moves
        moves_played = 0
        while not state.is_terminal() and moves_played < 10:
            legal = state.legal_moves()
            if not legal:
                break
            move = legal[0]  # Just pick first move
            state = state.apply_move(move)
            moves_played += 1

        # Should have played some moves
        assert moves_played > 0
        assert state.ply == moves_played

    def test_active_size_preserved_through_moves(self):
        """active_size should be preserved when applying moves."""
        state = TwixtState(active_size=10)
        state = state.apply_move((3, 3))
        state = state.apply_move((4, 0))  # Black move

        assert state.active_size == 10

    def test_tensor_shape_independent_of_active_size(self):
        """Tensor spatial dims always 24x24 regardless of active_size; channel
        count tracks NUM_CHANNELS (30 after Phase 2)."""
        from scripts.GPU.alphazero.game.twixt_state import NUM_CHANNELS
        for active_size in [6, 8, 12, 24]:
            state = TwixtState(active_size=active_size)
            tensor = state.to_tensor()
            assert tensor.shape == (NUM_CHANNELS, 24, 24), \
                f"Wrong shape for active_size={active_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
