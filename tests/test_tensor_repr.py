"""Tests for tensor representation module.

Verifies:
1. Legal mask matches generate_moves() exactly
2. Bridge direction encoding is correct and reversible
3. Goal distance channels are turn-invariant
4. Channel dimensions are correct
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.GPU.game.state import GameState
from scripts.GPU.game.rules import apply_move, generate_moves
from scripts.GPU.ai.tensor_repr import (
    state_to_numpy,
    verify_legal_mask,
    get_legal_moves_from_mask,
    KNIGHT_DIRS,
    DIR_TO_IDX,
    opposite_dir,
    CHANNEL_NAMES,
)


class TestLegalMask:
    """Test that legal mask matches generate_moves() exactly."""

    def test_empty_board_red(self):
        """Empty board, red to move."""
        state = GameState(board_size=24)
        assert state.to_move == "red"
        assert verify_legal_mask(state)

    def test_empty_board_black(self):
        """Empty board, black to move."""
        state = GameState(board_size=24)
        state.to_move = "black"
        assert verify_legal_mask(state)

    def test_with_pegs(self):
        """Board with some pegs placed."""
        state = GameState(board_size=24)
        state = apply_move(state, 5, 5)   # Red
        state = apply_move(state, 10, 10)  # Black
        state = apply_move(state, 6, 7)   # Red
        state = apply_move(state, 11, 12)  # Black

        # Should match for both players
        assert verify_legal_mask(state)

        # Switch player and check again
        state.to_move = "red" if state.to_move == "black" else "black"
        # Note: This changes the mask, so we rebuild
        tensor = state_to_numpy(state)
        mask_moves = set(get_legal_moves_from_mask(tensor[:, :, 18]))
        actual_moves = set(generate_moves(state))
        assert mask_moves == actual_moves

    def test_edge_restrictions_red(self):
        """Red cannot place on columns 0 and 23."""
        state = GameState(board_size=24)
        state.to_move = "red"
        tensor = state_to_numpy(state)
        legal_mask = tensor[:, :, 18]

        # Left and right edges should be forbidden
        assert np.all(legal_mask[:, 0] == 0.0)
        assert np.all(legal_mask[:, 23] == 0.0)

        # Interior should mostly be legal (except corners)
        assert legal_mask[5, 5] == 1.0
        assert legal_mask[10, 10] == 1.0

    def test_edge_restrictions_black(self):
        """Black cannot place on rows 0 and 23."""
        state = GameState(board_size=24)
        state.to_move = "black"
        tensor = state_to_numpy(state)
        legal_mask = tensor[:, :, 18]

        # Top and bottom edges should be forbidden
        assert np.all(legal_mask[0, :] == 0.0)
        assert np.all(legal_mask[23, :] == 0.0)

        # Interior should mostly be legal (except corners)
        assert legal_mask[5, 5] == 1.0
        assert legal_mask[10, 10] == 1.0

    def test_corners_forbidden(self):
        """All 4 corners are forbidden for both players."""
        for player in ["red", "black"]:
            state = GameState(board_size=24)
            state.to_move = player
            tensor = state_to_numpy(state)
            legal_mask = tensor[:, :, 18]

            assert legal_mask[0, 0] == 0.0
            assert legal_mask[0, 23] == 0.0
            assert legal_mask[23, 0] == 0.0
            assert legal_mask[23, 23] == 0.0

    def test_occupied_cells_forbidden(self):
        """Occupied cells should be forbidden."""
        state = GameState(board_size=24)
        state = apply_move(state, 5, 5)  # Red at (5,5)

        tensor = state_to_numpy(state)
        legal_mask = tensor[:, :, 18]

        # (5,5) is now occupied
        assert legal_mask[5, 5] == 0.0


class TestBridgeEncoding:
    """Test bridge direction encoding."""

    def test_opposite_dir(self):
        """Test opposite direction calculation."""
        for i, (dr, dc) in enumerate(KNIGHT_DIRS):
            opp = opposite_dir(i)
            opp_dr, opp_dc = KNIGHT_DIRS[opp]
            assert opp_dr == -dr
            assert opp_dc == -dc

    def test_all_knight_directions_covered(self):
        """Ensure all 8 knight directions are indexed."""
        assert len(KNIGHT_DIRS) == 8
        assert len(DIR_TO_IDX) == 8

        # Check symmetry
        for dr, dc in KNIGHT_DIRS:
            assert (-dr, -dc) in DIR_TO_IDX

    def test_bridge_encoding_simple(self):
        """Test that a bridge is encoded in both endpoint channels."""
        state = GameState(board_size=24)
        # Place two red pegs that form a bridge
        state = apply_move(state, 5, 5)   # Red
        state = apply_move(state, 10, 10)  # Black (to switch turns)
        state = apply_move(state, 7, 6)   # Red - knight move from (5,5)

        # Should have a bridge between (5,5) and (7,6)
        # Direction: (7-5, 6-5) = (2, 1)
        assert (2, 1) in DIR_TO_IDX

        tensor = state_to_numpy(state)

        # Check that bridge is encoded
        # Red bridges are channels 2-9
        dir_idx = DIR_TO_IDX[(2, 1)]
        opp_idx = opposite_dir(dir_idx)

        # At (5,5): direction toward (7,6) = (2,1)
        assert tensor[5, 5, 2 + dir_idx] == 1.0

        # At (7,6): direction toward (5,5) = (-2,-1)
        assert tensor[7, 6, 2 + opp_idx] == 1.0

    def test_bridge_encoding_black(self):
        """Test bridge encoding for black player."""
        state = GameState(board_size=24)
        state = apply_move(state, 5, 5)   # Red
        state = apply_move(state, 10, 10)  # Black
        state = apply_move(state, 6, 7)   # Red
        state = apply_move(state, 12, 11)  # Black - knight move from (10,10)

        # Direction: (12-10, 11-10) = (2, 1)
        dir_idx = DIR_TO_IDX[(2, 1)]
        opp_idx = opposite_dir(dir_idx)

        tensor = state_to_numpy(state)

        # Black bridges are channels 10-17
        # At (10,10): direction toward (12,11)
        assert tensor[10, 10, 10 + dir_idx] == 1.0

        # At (12,11): direction toward (10,10)
        assert tensor[12, 11, 10 + opp_idx] == 1.0


class TestGoalDistance:
    """Test goal distance channels are turn-invariant."""

    def test_goal_distance_shape(self):
        """Goal distance channels have correct shape."""
        state = GameState(board_size=24)
        tensor = state_to_numpy(state)

        assert tensor.shape == (24, 24, 24)

    def test_goal_distance_turn_invariant(self):
        """Goal distances don't change with player to move."""
        state = GameState(board_size=24)

        # Get tensor for red's turn
        state.to_move = "red"
        tensor_red = state_to_numpy(state)

        # Get tensor for black's turn
        state.to_move = "black"
        tensor_black = state_to_numpy(state)

        # Goal distance channels (20-23) should be identical
        np.testing.assert_array_equal(
            tensor_red[:, :, 20:24],
            tensor_black[:, :, 20:24]
        )

    def test_goal_distance_values(self):
        """Check goal distance values at specific positions."""
        state = GameState(board_size=24)
        tensor = state_to_numpy(state)

        # Channel 20: distance from top (row 0)
        assert tensor[0, 10, 20] == 0.0  # At top edge
        assert tensor[23, 10, 20] == 1.0  # At bottom edge
        assert abs(tensor[12, 10, 20] - 12/23) < 0.01  # Middle

        # Channel 21: distance from bottom (row 23)
        assert tensor[23, 10, 21] == 0.0  # At bottom edge
        assert tensor[0, 10, 21] == 1.0  # At top edge

        # Channel 22: distance from left (col 0)
        assert tensor[10, 0, 22] == 0.0  # At left edge
        assert tensor[10, 23, 22] == 1.0  # At right edge

        # Channel 23: distance from right (col 23)
        assert tensor[10, 23, 23] == 0.0  # At right edge
        assert tensor[10, 0, 23] == 1.0  # At left edge


class TestPlayerChannel:
    """Test player-to-move channel."""

    def test_player_channel_red(self):
        """Red to move = all 1s in channel 19."""
        state = GameState(board_size=24)
        state.to_move = "red"
        tensor = state_to_numpy(state)

        assert np.all(tensor[:, :, 19] == 1.0)

    def test_player_channel_black(self):
        """Black to move = all 0s in channel 19."""
        state = GameState(board_size=24)
        state.to_move = "black"
        tensor = state_to_numpy(state)

        assert np.all(tensor[:, :, 19] == 0.0)


class TestPegEncoding:
    """Test peg position encoding."""

    def test_red_pegs_channel(self):
        """Red pegs are in channel 0."""
        state = GameState(board_size=24)
        state = apply_move(state, 5, 5)  # Red

        tensor = state_to_numpy(state)
        assert tensor[5, 5, 0] == 1.0
        assert tensor[5, 5, 1] == 0.0  # Not black

    def test_black_pegs_channel(self):
        """Black pegs are in channel 1."""
        state = GameState(board_size=24)
        state = apply_move(state, 5, 5)   # Red
        state = apply_move(state, 10, 10)  # Black

        tensor = state_to_numpy(state)
        assert tensor[10, 10, 1] == 1.0
        assert tensor[10, 10, 0] == 0.0  # Not red


class TestChannelCount:
    """Verify we have exactly 24 channels."""

    def test_channel_count(self):
        """Tensor has 24 channels."""
        state = GameState(board_size=24)
        tensor = state_to_numpy(state)
        assert tensor.shape[2] == 24

    def test_channel_names_complete(self):
        """All 24 channels are named."""
        assert len(CHANNEL_NAMES) == 24
        for i in range(24):
            assert i in CHANNEL_NAMES


class TestMidgamePosition:
    """Test with a realistic mid-game position."""

    def test_midgame_legal_mask(self):
        """Legal mask matches in mid-game."""
        state = GameState(board_size=24)

        # Play several moves
        moves = [
            (5, 10), (10, 5),   # Red, Black
            (6, 12), (11, 6),   # Red, Black
            (7, 11), (12, 4),   # Red, Black
            (8, 13), (13, 5),   # Red, Black
        ]
        for r, c in moves:
            state = apply_move(state, r, c)

        # Verify legal mask matches
        assert verify_legal_mask(state)

    def test_midgame_bridge_count(self):
        """Bridges are encoded correctly in mid-game."""
        state = GameState(board_size=24)

        # Create positions that form bridges
        state = apply_move(state, 5, 5)    # Red
        state = apply_move(state, 10, 10)   # Black
        state = apply_move(state, 7, 6)    # Red - bridge with (5,5)
        state = apply_move(state, 12, 11)   # Black - bridge with (10,10)
        state = apply_move(state, 9, 7)    # Red - bridge with (7,6)

        tensor = state_to_numpy(state)

        # Count non-zero bridge channels for red (2-9)
        red_bridge_sum = np.sum(tensor[:, :, 2:10])
        # Each bridge marks 2 endpoints, so should be even
        assert red_bridge_sum % 2 == 0
        assert red_bridge_sum > 0

        # Count for black (10-17)
        black_bridge_sum = np.sum(tensor[:, :, 10:18])
        assert black_bridge_sum % 2 == 0
        assert black_bridge_sum > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
