from __future__ import annotations

from typing import Iterable, List, Tuple

from .state import GameState


def is_in_bounds(state: GameState, row: int, col: int) -> bool:
    """Check if (row, col) is within the board."""
    return 0 <= row < state.board_size and 0 <= col < state.board_size


def is_corner(state: GameState, row: int, col: int) -> bool:
    """Check if (row, col) is a corner position (forbidden for all players)."""
    last = state.board_size - 1
    return (row == 0 or row == last) and (col == 0 or col == last)


def is_valid_placement(state: GameState, player: str, row: int, col: int) -> bool:
    """Check if a player can legally place a peg at (row, col).

    Rules:
    - Must be in bounds
    - Must not be occupied
    - Corners are forbidden for all players
    - Red cannot place on cols 0 or 23 (black's goal edges)
    - Black cannot place on rows 0 or 23 (red's goal edges)
    """
    if not is_in_bounds(state, row, col):
        return False

    if (row, col) in state.pegs:
        return False

    if is_corner(state, row, col):
        return False

    last = state.board_size - 1

    if player == "red":
        # Red connects top↔bottom; cannot place on left/right edges
        if col == 0 or col == last:
            return False
    else:
        # Black connects left↔right; cannot place on top/bottom edges
        if row == 0 or row == last:
            return False

    return True


def place_peg(state: GameState, player: str, row: int, col: int) -> None:
    """Place a peg. Raises ValueError if placement is invalid.

    Note: This only places the peg. Bridge creation is handled separately
    in the rules module.
    """
    if not is_valid_placement(state, player, row, col):
        raise ValueError(f"Invalid placement: {player} at ({row}, {col})")

    state.pegs[(row, col)] = player
    state.move_history.append((player, row, col))
    state.invalidate_cc_cache()  # Invalidate after mutation


def legal_moves(state: GameState, player: str) -> List[Tuple[int, int]]:
    """Return all legal moves for a player as a list of (row, col) tuples.

    Uses the current player from state.to_move if player is not specified.
    """
    moves = []
    for r in range(state.board_size):
        for c in range(state.board_size):
            if is_valid_placement(state, player, r, c):
                moves.append((r, c))
    return moves


def legal_moves_for_current(state: GameState) -> List[Tuple[int, int]]:
    """Return all legal moves for the current player (state.to_move)."""
    return legal_moves(state, state.to_move)
