from __future__ import annotations

from collections import deque
from typing import List, Optional, Set, Tuple

from .board import is_valid_placement, legal_moves
from .bridge import add_bridges_for_new_peg
from .state import GameState


def apply_move(state: GameState, row: int, col: int) -> GameState:
    """Return a new state with the move applied.

    Places peg, creates bridges, switches player.
    """
    s2 = state.copy()
    player = s2.to_move

    # Place peg
    if not is_valid_placement(s2, player, row, col):
        raise ValueError(f"Invalid move: {player} at ({row}, {col})")

    s2.pegs[(row, col)] = player
    s2.move_history.append((player, row, col))

    # Create bridges
    add_bridges_for_new_peg(s2, player, row, col)

    # Invalidate CC cache AFTER all mutations to pegs/bridges/mask
    s2.invalidate_cc_cache()

    # Switch player
    s2.to_move = "red" if player == "black" else "black"

    return s2


def get_connected_component(
    state: GameState,
    start_row: int,
    start_col: int,
    player: str
) -> Set[Tuple[int, int]]:
    """BFS to find all positions connected to (start_row, start_col) via same-player bridges.

    Returns set of (row, col) tuples in the connected component.
    """
    if (start_row, start_col) not in state.pegs:
        return set()
    if state.pegs[(start_row, start_col)] != player:
        return set()

    visited: Set[Tuple[int, int]] = set()
    queue: deque[Tuple[int, int]] = deque()
    queue.append((start_row, start_col))

    while queue:
        row, col = queue.popleft()

        if (row, col) in visited:
            continue
        if (row, col) not in state.pegs:
            continue
        if state.pegs[(row, col)] != player:
            continue

        visited.add((row, col))

        # Find neighbors through bridges
        for bridge in state.bridges:
            (r1, c1), (r2, c2) = bridge

            # Check if either endpoint of bridge is our current position
            # and the other endpoint belongs to same player
            if r1 == row and c1 == col:
                other = (r2, c2)
            elif r2 == row and c2 == col:
                other = (r1, c1)
            else:
                continue

            # Bridge must connect same-player pegs
            if other in state.pegs and state.pegs[other] == player:
                if other not in visited:
                    queue.append(other)

    return visited


def check_winner(state: GameState) -> Optional[str]:
    """Check if either player has won.

    Red wins: connected path from row 0 to row 23 via red bridges
    Black wins: connected path from col 0 to col 23 via black bridges

    Returns 'red', 'black', or None.
    """
    last = state.board_size - 1

    # Check red (connects top to bottom)
    for start_col in range(state.board_size):
        if (0, start_col) not in state.pegs:
            continue
        if state.pegs[(0, start_col)] != "red":
            continue

        component = get_connected_component(state, 0, start_col, "red")
        for row, col in component:
            if row == last:
                return "red"

    # Check black (connects left to right)
    for start_row in range(state.board_size):
        if (start_row, 0) not in state.pegs:
            continue
        if state.pegs[(start_row, 0)] != "black":
            continue

        component = get_connected_component(state, start_row, 0, "black")
        for row, col in component:
            if col == last:
                return "black"

    return None


def is_game_over(state: GameState) -> bool:
    """Check if the game has ended (someone won or no moves available)."""
    winner = check_winner(state)
    if winner is not None:
        return True

    # Check if current player has any legal moves
    moves = legal_moves(state, state.to_move)
    return len(moves) == 0


def generate_moves(state: GameState) -> List[Tuple[int, int]]:
    """Return all legal moves for the current player."""
    return legal_moves(state, state.to_move)
