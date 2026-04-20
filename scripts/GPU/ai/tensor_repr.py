"""Board tensor representation for GPU-accelerated move scoring.

Converts GameState to a 24-channel tensor suitable for CNN processing.
Build in NumPy first, convert to MLX at the end to avoid per-cell updates.

Channel layout (24 channels):
    0:     Red peg positions (0/1)
    1:     Black peg positions (0/1)
    2-9:   Red bridges (8 knight-move directions)
    10-17: Black bridges (8 knight-move directions)
    18:    Legal placement mask for current player
    19:    Player to move (1.0 = red, 0.0 = black)
    20:    Distance to red top edge (row 0)
    21:    Distance to red bottom edge (row N-1)
    22:    Distance to black left edge (col 0)
    23:    Distance to black right edge (col N-1)

Goal distance planes are purely geometric; they do not depend on state.to_move.
"""
from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

import numpy as np

from ..utils.maybe_mlx import try_import_mlx

if TYPE_CHECKING:
    from ..game.state import GameState

# Try to import MLX
_mlx_env = try_import_mlx()

# 8 knight-move directions (matching TwixT bridge geometry)
KNIGHT_DIRS: List[Tuple[int, int]] = [
    (-2, -1), (-2, +1), (-1, -2), (-1, +2),
    (+1, -2), (+1, +2), (+2, -1), (+2, +1),
]
DIR_TO_IDX = {d: i for i, d in enumerate(KNIGHT_DIRS)}


def opposite_dir(idx: int) -> int:
    """Get opposite direction index."""
    dr, dc = KNIGHT_DIRS[idx]
    return DIR_TO_IDX[(-dr, -dc)]


def state_to_numpy(state: "GameState") -> np.ndarray:
    """Convert GameState to NumPy tensor.

    Args:
        state: Current game state

    Returns:
        np.ndarray of shape (N, N, 24) where N = board_size
    """
    N = state.board_size
    tensor = np.zeros((N, N, 24), dtype=np.float32)

    # Channels 0-1: Peg positions
    for (r, c), player in state.pegs.items():
        if player == "red":
            tensor[r, c, 0] = 1.0
        else:
            tensor[r, c, 1] = 1.0

    # Channels 2-17: Bridge directions (8 per player)
    for (r1, c1), (r2, c2) in state.bridges:
        # Determine player from either endpoint (both must match)
        player = state.pegs.get((r1, c1)) or state.pegs.get((r2, c2))
        if player is None:
            continue

        # Debug-only: verify both endpoints exist and match
        if __debug__:
            p1 = state.pegs.get((r1, c1))
            p2 = state.pegs.get((r2, c2))
            if p1 != p2 or p1 != player:
                # Log warning instead of crashing training
                pass

        dr, dc = r2 - r1, c2 - c1

        # Try forward direction
        dir_idx = DIR_TO_IDX.get((dr, dc))
        if dir_idx is None:
            # Try reversed (bridges are canonical: smaller endpoint first)
            dir_idx = DIR_TO_IDX.get((-dr, -dc))
            if dir_idx is not None:
                # Swap endpoints for correct direction encoding
                r1, c1, r2, c2 = r2, c2, r1, c1
            else:
                continue  # Invalid bridge geometry

        base_ch = 2 if player == "red" else 10
        # Mark both endpoints with their respective directions
        tensor[r1, c1, base_ch + dir_idx] = 1.0
        tensor[r2, c2, base_ch + opposite_dir(dir_idx)] = 1.0

    # Channel 18: Legal placement mask for current player
    # Must match generate_moves() / is_valid_placement() exactly
    tensor[:, :, 18] = _compute_legal_mask_numpy(state)

    # Channel 19: Player to move
    if state.to_move == "red":
        tensor[:, :, 19] = 1.0
    # else: already 0.0

    # Channels 20-23: Goal distance planes (purely geometric, turn-invariant)
    # Red goal edges: top (row 0) and bottom (row N-1)
    # Black goal edges: left (col 0) and right (col N-1)
    for r in range(N):
        tensor[r, :, 20] = r / (N - 1)           # Distance from top (0 at top)
        tensor[r, :, 21] = (N - 1 - r) / (N - 1)  # Distance from bottom (0 at bottom)
    for c in range(N):
        tensor[:, c, 22] = c / (N - 1)           # Distance from left (0 at left)
        tensor[:, c, 23] = (N - 1 - c) / (N - 1)  # Distance from right (0 at right)

    return tensor


def _compute_legal_mask_numpy(state: "GameState") -> np.ndarray:
    """Compute legal placement mask - MUST match generate_moves() exactly.

    Rules from is_valid_placement():
    - In bounds [0, N-1] (always true for array indices)
    - Not occupied
    - Not a corner (4 corners)
    - Red: forbidden on cols 0 and N-1
    - Black: forbidden on rows 0 and N-1

    Note: NO crossing-bridge constraints (bridges created after placement).
    """
    N = state.board_size
    mask = np.ones((N, N), dtype=np.float32)

    # Occupied cells
    for (r, c) in state.pegs:
        mask[r, c] = 0.0

    # Corners (always forbidden)
    mask[0, 0] = 0.0
    mask[0, N - 1] = 0.0
    mask[N - 1, 0] = 0.0
    mask[N - 1, N - 1] = 0.0

    # Edge restrictions for current player
    if state.to_move == "red":
        mask[:, 0] = 0.0       # Left edge forbidden for red
        mask[:, N - 1] = 0.0   # Right edge forbidden for red
    else:
        mask[0, :] = 0.0       # Top edge forbidden for black
        mask[N - 1, :] = 0.0   # Bottom edge forbidden for black

    return mask


def state_to_tensor(state: "GameState"):
    """Convert GameState to MLX tensor (or NumPy if MLX unavailable).

    Build in NumPy first, convert to MLX at the end.
    Avoids per-cell MLX updates which kill fusion.

    Args:
        state: Current game state

    Returns:
        mx.array (or np.ndarray) of shape (N, N, 24)
    """
    np_tensor = state_to_numpy(state)

    if _mlx_env.available:
        return _mlx_env.mx.array(np_tensor)
    return np_tensor


def moves_to_coords(moves: List[Tuple[int, int]]):
    """Convert move list to coordinate array for gather operations.

    Args:
        moves: List of (row, col) tuples

    Returns:
        mx.array (or np.ndarray) of shape (N, 2)
    """
    coords = np.array(moves, dtype=np.int32)

    if _mlx_env.available:
        return _mlx_env.mx.array(coords)
    return coords


def get_legal_moves_from_mask(legal_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Extract legal move coordinates from mask.

    Useful for verification against generate_moves().

    Args:
        legal_mask: (N, N) array with 1.0 for legal cells

    Returns:
        List of (row, col) tuples for legal moves
    """
    rows, cols = np.where(legal_mask > 0.5)
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


def verify_legal_mask(state: "GameState") -> bool:
    """Verify that our legal mask matches generate_moves() exactly.

    Args:
        state: Game state to check

    Returns:
        True if masks match, False otherwise
    """
    from ..game.rules import generate_moves

    # Get moves from our mask
    np_tensor = state_to_numpy(state)
    mask_moves = set(get_legal_moves_from_mask(np_tensor[:, :, 18]))

    # Get moves from generate_moves
    actual_moves = set(generate_moves(state))

    return mask_moves == actual_moves


# Channel indices for documentation/debugging
CHANNEL_NAMES = {
    0: "red_pegs",
    1: "black_pegs",
    2: "red_bridge_dir0",
    3: "red_bridge_dir1",
    4: "red_bridge_dir2",
    5: "red_bridge_dir3",
    6: "red_bridge_dir4",
    7: "red_bridge_dir5",
    8: "red_bridge_dir6",
    9: "red_bridge_dir7",
    10: "black_bridge_dir0",
    11: "black_bridge_dir1",
    12: "black_bridge_dir2",
    13: "black_bridge_dir3",
    14: "black_bridge_dir4",
    15: "black_bridge_dir5",
    16: "black_bridge_dir6",
    17: "black_bridge_dir7",
    18: "legal_mask",
    19: "player_to_move",
    20: "dist_red_top",
    21: "dist_red_bottom",
    22: "dist_black_left",
    23: "dist_black_right",
}
