"""Tests for canonicalize_batch() to prevent rotation/coordinate bugs.

These tests would have caught the active-region rotation bug immediately:
- Board rotation was using full 24x24: (r,c) -> (c, 23-r)
- Move rotation was using active_size: (r,c) -> (c, S-1-r)
- Result: policy gathered features at wrong locations for black-to-move
"""

import pytest
import numpy as np
import mlx.core as mx

from scripts.GPU.alphazero.network import (
    canonicalize_batch,
    CH_RED_PEG,
    CH_BLACK_PEG,
    CH_TO_MOVE,
    NUM_CHANNELS,
)

BOARD_SIZE = 24


def make_empty_board(active_size: int, to_move: str = "black") -> mx.array:
    """Create an empty board tensor with specified to_move."""
    board = np.zeros((1, BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), dtype=np.float32)
    # CH_TO_MOVE: 1.0 = red, 0.0 = black
    if to_move == "red":
        board[0, :active_size, :active_size, CH_TO_MOVE] = 1.0
    # else: already 0.0 for black
    return mx.array(board)


def place_peg(board: mx.array, row: int, col: int, color: str) -> mx.array:
    """Place a peg on the board (returns new array)."""
    board_np = np.array(board)
    ch = CH_RED_PEG if color == "red" else CH_BLACK_PEG
    board_np[0, row, col, ch] = 1.0
    return mx.array(board_np)


class TestBlackRotationSinglePeg:
    """Test that single peg maps correctly under black canonicalization."""

    @pytest.mark.parametrize("r,c,S", [
        (4, 2, 8),   # center-ish (the exact bug case)
        (0, 5, 8),   # top edge
        (6, 1, 8),   # near corner
        (3, 3, 10),  # different active_size
        (9, 4, 12),  # larger board
    ])
    def test_black_rotation_single_peg_maps_correctly(self, r, c, S):
        """Peg at (r,c) must appear at (c, S-1-r) after black canonicalization."""
        H = W = BOARD_SIZE

        # Create black-to-move board with single red peg
        board = make_empty_board(S, to_move="black")
        board = place_peg(board, r, c, "red")

        # Dummy moves (not used for this test, but required by API)
        move_rows = mx.array([[r]], dtype=mx.int32)
        move_cols = mx.array([[c]], dtype=mx.int32)
        move_mask = mx.array([[1.0]], dtype=mx.float32)

        # Canonicalize
        boards_out, _, _, _ = canonicalize_batch(
            board, move_rows, move_cols, move_mask, S
        )

        # Expected correct location: (r, c) -> (c, S-1-r)
        correct_r = c
        correct_c = S - 1 - r

        # Expected wrong location (old bug): (r, c) -> (c, H-1-r)
        wrong_r = c
        wrong_c = H - 1 - r

        # After swap: red peg becomes opponent (channel 1)
        # Check correct location has the peg
        opp_at_correct = float(boards_out[0, correct_r, correct_c, 1].item())
        assert opp_at_correct == 1.0, (
            f"Peg at ({r},{c}) should map to ({correct_r},{correct_c}) "
            f"but opp_peg={opp_at_correct}"
        )

        # Check wrong location is empty (regression guard)
        if wrong_c < BOARD_SIZE:  # only check if in bounds
            opp_at_wrong = float(boards_out[0, wrong_r, wrong_c, 1].item())
            assert opp_at_wrong == 0.0, (
                f"Old bug location ({wrong_r},{wrong_c}) should be empty "
                f"but opp_peg={opp_at_wrong}"
            )


class TestMovesRotateInBounds:
    """Test that rotated move coordinates stay within active region."""

    @pytest.mark.parametrize("S", [8, 10, 12, 16])
    def test_moves_rotate_with_board_in_bounds(self, S):
        """All rotated moves must satisfy 0 <= r' < S and 0 <= c' < S."""
        # Create black-to-move board
        board = make_empty_board(S, to_move="black")

        # Generate moves covering the active region
        moves = []
        for r in range(S):
            for c in range(S):
                # Skip corners (illegal in TwixT)
                if (r == 0 or r == S-1) and (c == 0 or c == S-1):
                    continue
                moves.append((r, c))

        # Take first 50 moves to keep test fast
        moves = moves[:50]

        move_rows = mx.array([[r for r, c in moves]], dtype=mx.int32)
        move_cols = mx.array([[c for r, c in moves]], dtype=mx.int32)
        move_mask = mx.ones((1, len(moves)), dtype=mx.float32)

        # Canonicalize
        _, rows_out, cols_out, _ = canonicalize_batch(
            board, move_rows, move_cols, move_mask, S
        )

        # Check all rotated coords are in bounds
        rows_np = np.array(rows_out[0])
        cols_np = np.array(cols_out[0])

        assert np.all(rows_np >= 0), f"Some rows < 0: {rows_np[rows_np < 0]}"
        assert np.all(rows_np < S), f"Some rows >= S: {rows_np[rows_np >= S]}"
        assert np.all(cols_np >= 0), f"Some cols < 0: {cols_np[cols_np < 0]}"
        assert np.all(cols_np < S), f"Some cols >= S: {cols_np[cols_np >= S]}"


class TestPegCountPreserved:
    """Test that peg count in active region is preserved."""

    @pytest.mark.parametrize("S", [8, 10, 12])
    def test_active_region_peg_count_preserved_black(self, S):
        """Peg count in [0:S,0:S] must match before/after canonicalization."""
        # Create black-to-move board with several pegs
        board = make_empty_board(S, to_move="black")

        # Place some pegs (avoiding corners)
        peg_positions = [
            (2, 3, "red"),
            (4, 5, "black"),
            (1, 4, "red"),
            (S-2, 2, "black"),
        ]
        for r, c, color in peg_positions:
            if r < S and c < S:
                board = place_peg(board, r, c, color)

        # Count pegs before
        board_np = np.array(board)
        red_before = np.sum(board_np[0, :S, :S, CH_RED_PEG])
        black_before = np.sum(board_np[0, :S, :S, CH_BLACK_PEG])
        total_before = red_before + black_before

        # Canonicalize
        move_rows = mx.array([[1]], dtype=mx.int32)
        move_cols = mx.array([[1]], dtype=mx.int32)
        move_mask = mx.array([[1.0]], dtype=mx.float32)

        boards_out, _, _, _ = canonicalize_batch(
            board, move_rows, move_cols, move_mask, S
        )

        # Count pegs after (channels are swapped: 0=cur, 1=opp)
        out_np = np.array(boards_out)
        cur_after = np.sum(out_np[0, :S, :S, 0])  # was black
        opp_after = np.sum(out_np[0, :S, :S, 1])  # was red
        total_after = cur_after + opp_after

        assert total_after == total_before, (
            f"Peg count changed: {total_before} -> {total_after}"
        )
        # Also check individual colors swapped correctly
        assert cur_after == black_before, (
            f"Black pegs should become cur: {black_before} -> {cur_after}"
        )
        assert opp_after == red_before, (
            f"Red pegs should become opp: {red_before} -> {opp_after}"
        )


class TestToMoveForced:
    """Test that CH_TO_MOVE is forced to 1 after canonicalization."""

    @pytest.mark.parametrize("to_move", ["red", "black"])
    @pytest.mark.parametrize("S", [8, 12])
    def test_to_move_forced_to_one(self, to_move, S):
        """After canonicalization, CH_TO_MOVE must be 1.0 in active region."""
        board = make_empty_board(S, to_move=to_move)

        move_rows = mx.array([[1]], dtype=mx.int32)
        move_cols = mx.array([[1]], dtype=mx.int32)
        move_mask = mx.array([[1.0]], dtype=mx.float32)

        boards_out, _, _, _ = canonicalize_batch(
            board, move_rows, move_cols, move_mask, S
        )

        # Check CH_TO_MOVE is 1.0 throughout active region
        out_np = np.array(boards_out)
        to_move_vals = out_np[0, :S, :S, CH_TO_MOVE]

        assert np.all(to_move_vals == 1.0), (
            f"CH_TO_MOVE should be 1.0 everywhere in active region, "
            f"but got min={to_move_vals.min()}, max={to_move_vals.max()}"
        )
