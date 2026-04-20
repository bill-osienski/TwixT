"""Tests for horizontal mirror augmentation correctness.

Verifies that _mirror_position_lr correctly:
1. Spatially flips columns within the active square
2. Remaps directional link channels (dc -> -dc)
3. Swaps BLACK_LEFT_DIST <-> BLACK_RIGHT_DIST
4. Mirrors legal move coordinates
5. Leaves regions outside the active square untouched
"""
import numpy as np
import pytest

from scripts.GPU.alphazero.self_play import (
    _mirror_position_lr,
    _MIRROR_DIR_PERM,
)
from scripts.GPU.alphazero.game import (
    DIRECTION_TO_CHANNEL,
    CHANNEL_RED_LINKS_START,
    CHANNEL_BLACK_LINKS_START,
    CHANNEL_BLACK_LEFT_DIST,
    CHANNEL_BLACK_RIGHT_DIST,
)


S = 8  # active_size for all tests


class TestMirrorDirPerm:
    """Verify the permutation table itself is self-consistent."""

    def test_perm_is_valid_permutation(self):
        """_MIRROR_DIR_PERM must be a permutation of 0..7."""
        assert sorted(_MIRROR_DIR_PERM) == list(range(8))

    def test_perm_is_involution(self):
        """Mirroring twice must return to the original direction."""
        for i in range(8):
            assert _MIRROR_DIR_PERM[_MIRROR_DIR_PERM[i]] == i, (
                f"dir {i} -> {_MIRROR_DIR_PERM[i]} -> "
                f"{_MIRROR_DIR_PERM[_MIRROR_DIR_PERM[i]]}, expected {i}"
            )

    def test_perm_matches_dc_negation(self):
        """Each mapping must correspond to negating the dc component."""
        idx_to_dir = {v: k for k, v in DIRECTION_TO_CHANNEL.items()}
        for src_idx in range(8):
            dr, dc = idx_to_dir[src_idx]
            expected_idx = DIRECTION_TO_CHANNEL[(dr, -dc)]
            assert _MIRROR_DIR_PERM[src_idx] == expected_idx, (
                f"dir ({dr},{dc}) idx={src_idx}: perm gives "
                f"{_MIRROR_DIR_PERM[src_idx]}, expected {expected_idx} "
                f"for ({dr},{-dc})"
            )


class TestMirrorSpatialFlip:
    """Verify spatial column flip within active square."""

    def test_peg_channel_flips_columns(self):
        """A hot pixel at (r, c) in a peg channel should land at (r, S-1-c)."""
        board = np.zeros((24, 24, 24), dtype=np.float32)
        r, c = 3, 2
        board[r, c, 0] = 1.0  # red peg channel

        out, _, _ = _mirror_position_lr(board, [(r, c)], [1], S)

        assert out[r, S - 1 - c, 0] == 1.0, "Peg should move to mirrored column"
        assert out[r, c, 0] == 0.0, "Original location should be empty"

    def test_outside_active_square_untouched(self):
        """Pixels outside [0:S, 0:S) must not be modified."""
        board = np.zeros((24, 24, 24), dtype=np.float32)
        # Place values outside the active square
        board[S + 1, 3, 0] = 7.0
        board[3, S + 1, 0] = 8.0
        board[S + 2, S + 2, 5] = 9.0

        out, _, _ = _mirror_position_lr(board, [(1, 1)], [1], S)

        assert out[S + 1, 3, 0] == 7.0
        assert out[3, S + 1, 0] == 8.0
        assert out[S + 2, S + 2, 5] == 9.0


class TestMirrorRedLinkChannels:
    """Verify red directional link channels are remapped correctly."""

    @pytest.mark.parametrize("src_dir_idx", range(8))
    def test_red_link_channel_remap(self, src_dir_idx):
        """A hot pixel in red link dir src_dir_idx should land in the
        mirrored dir channel at the mirrored column."""
        board = np.zeros((24, 24, 24), dtype=np.float32)
        r, c = 4, 2
        src_ch = CHANNEL_RED_LINKS_START + src_dir_idx
        board[r, c, src_ch] = 1.0

        out, _, _ = _mirror_position_lr(board, [(1, 1)], [1], S)

        dst_dir_idx = _MIRROR_DIR_PERM[src_dir_idx]
        dst_ch = CHANNEL_RED_LINKS_START + dst_dir_idx
        mc = S - 1 - c  # mirrored column

        assert out[r, mc, dst_ch] == 1.0, (
            f"Red dir {src_dir_idx}->ch{src_ch} at ({r},{c}) should map to "
            f"dir {dst_dir_idx}->ch{dst_ch} at ({r},{mc})"
        )
        # Original location in original channel should be zero
        # (unless src==dst and c==mc, which can't happen with c=2, S=8)
        assert out[r, c, src_ch] == 0.0, "Original location should be cleared"


class TestMirrorBlackLinkChannels:
    """Verify black directional link channels are remapped correctly."""

    @pytest.mark.parametrize("src_dir_idx", range(8))
    def test_black_link_channel_remap(self, src_dir_idx):
        """A hot pixel in black link dir src_dir_idx should land in the
        mirrored dir channel at the mirrored column."""
        board = np.zeros((24, 24, 24), dtype=np.float32)
        r, c = 5, 1
        src_ch = CHANNEL_BLACK_LINKS_START + src_dir_idx
        board[r, c, src_ch] = 1.0

        out, _, _ = _mirror_position_lr(board, [(1, 1)], [1], S)

        dst_dir_idx = _MIRROR_DIR_PERM[src_dir_idx]
        dst_ch = CHANNEL_BLACK_LINKS_START + dst_dir_idx
        mc = S - 1 - c

        assert out[r, mc, dst_ch] == 1.0, (
            f"Black dir {src_dir_idx}->ch{src_ch} at ({r},{c}) should map to "
            f"dir {dst_dir_idx}->ch{dst_ch} at ({r},{mc})"
        )
        assert out[r, c, src_ch] == 0.0, "Original location should be cleared"


class TestMirrorDistanceSwap:
    """Verify BLACK_LEFT_DIST and BLACK_RIGHT_DIST are swapped."""

    def test_left_right_swap(self):
        """Channels 21 and 22 should swap within the active square."""
        board = np.zeros((24, 24, 24), dtype=np.float32)
        # Fill left dist with 0.25, right dist with 0.75 inside active square
        board[:S, :S, CHANNEL_BLACK_LEFT_DIST] = 0.25
        board[:S, :S, CHANNEL_BLACK_RIGHT_DIST] = 0.75

        out, _, _ = _mirror_position_lr(board, [(1, 1)], [1], S)

        # After mirror: left should have old right's VALUES (spatially flipped),
        # and right should have old left's VALUES (spatially flipped).
        # Since both were uniform constants, spatial flip doesn't change the value.
        assert np.allclose(out[:S, :S, CHANNEL_BLACK_LEFT_DIST], 0.75), (
            "LEFT_DIST should now contain old RIGHT_DIST values"
        )
        assert np.allclose(out[:S, :S, CHANNEL_BLACK_RIGHT_DIST], 0.25), (
            "RIGHT_DIST should now contain old LEFT_DIST values"
        )

    def test_left_right_swap_with_gradient(self):
        """Non-uniform dist values: verify both spatial flip and channel swap."""
        board = np.zeros((24, 24, 24), dtype=np.float32)
        # left_dist = column index / S (increases left to right)
        for c in range(S):
            board[:S, c, CHANNEL_BLACK_LEFT_DIST] = c / S
        # right_dist = 1 - column index / S (decreases left to right)
        for c in range(S):
            board[:S, c, CHANNEL_BLACK_RIGHT_DIST] = 1.0 - c / S

        out, _, _ = _mirror_position_lr(board, [(1, 1)], [1], S)

        # After mirror:
        # - spatial flip reverses columns
        # - channel swap exchanges left<->right
        # So out[:S, c, LEFT] = old_right[:S, S-1-c] = 1 - (S-1-c)/S
        for c in range(S):
            expected_left = 1.0 - (S - 1 - c) / S
            expected_right = (S - 1 - c) / S
            assert np.allclose(out[:S, c, CHANNEL_BLACK_LEFT_DIST], expected_left), (
                f"col {c}: LEFT_DIST expected {expected_left}, "
                f"got {out[0, c, CHANNEL_BLACK_LEFT_DIST]}"
            )
            assert np.allclose(out[:S, c, CHANNEL_BLACK_RIGHT_DIST], expected_right), (
                f"col {c}: RIGHT_DIST expected {expected_right}, "
                f"got {out[0, c, CHANNEL_BLACK_RIGHT_DIST]}"
            )


class TestMirrorLegalMoves:
    """Verify legal move coordinate mirroring."""

    def test_moves_mirror_columns(self):
        """Each (r, c) should become (r, S-1-c)."""
        moves = [(0, 0), (3, 2), (7, 7), (5, 4)]
        board = np.zeros((24, 24, 24), dtype=np.float32)
        counts = [10, 20, 30, 40]

        _, mirrored, out_counts = _mirror_position_lr(board, moves, counts, S)

        expected = [(0, 7), (3, 5), (7, 0), (5, 3)]
        assert mirrored == expected, f"Expected {expected}, got {mirrored}"
        assert out_counts == counts, "Visit counts should be unchanged"

    def test_center_move_stays_or_shifts(self):
        """For odd S, center column maps to itself; for even S it shifts."""
        # S=8 (even): col 3 -> col 4, col 4 -> col 3
        _, mirrored, _ = _mirror_position_lr(
            np.zeros((24, 24, 24), dtype=np.float32),
            [(4, 3), (4, 4)], [1, 1], S
        )
        assert mirrored == [(4, 4), (4, 3)]


class TestMirrorInvolution:
    """Mirroring twice should return the original board and moves."""

    def test_double_mirror_is_identity(self):
        """Apply mirror twice — result should match the original."""
        rng = np.random.RandomState(42)
        board = rng.randn(24, 24, 24).astype(np.float32)
        moves = [(1, 2), (3, 5), (6, 0), (7, 7)]
        counts = [10, 20, 30, 40]

        m1_board, m1_moves, m1_counts = _mirror_position_lr(board, moves, counts, S)
        m2_board, m2_moves, m2_counts = _mirror_position_lr(m1_board, m1_moves, m1_counts, S)

        assert np.allclose(m2_board, board, atol=1e-6), (
            "Double mirror should recover original board"
        )
        assert m2_moves == moves, (
            f"Double mirror moves: expected {moves}, got {m2_moves}"
        )
        assert m2_counts == counts
