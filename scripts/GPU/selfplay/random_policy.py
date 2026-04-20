"""Random and greedy move policies for fuzz testing.

These simple policies help find rule bugs quickly by running many games
without needing the full heuristics/search implementation.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from ..game.state import GameState
from ..game.rules import generate_moves


def random_move(state: GameState, rng: Optional[random.Random] = None) -> Optional[Tuple[int, int]]:
    """Pick a random legal move.

    Args:
        state: Current game state
        rng: Random number generator (uses global if not provided)

    Returns:
        (row, col) tuple or None if no legal moves
    """
    moves = generate_moves(state)
    if not moves:
        return None

    if rng is None:
        return random.choice(moves)
    return rng.choice(moves)


def search_move(state: GameState, depth: int = 2, top_n: int = 12) -> Optional[Tuple[int, int]]:
    """Pick the best move using minimax search.

    Args:
        state: Current game state
        depth: Search depth (2-4 recommended)
        top_n: Number of moves to consider at each level

    Returns:
        (row, col) tuple or None if no legal moves
    """
    from ..ai.search import get_best_move

    moves = generate_moves(state)
    if not moves:
        return None

    return get_best_move(state, depth=depth, top_n=top_n)


def greedy_move(state: GameState, rng: Optional[random.Random] = None) -> Optional[Tuple[int, int]]:
    """Pick a greedy move using full heuristics.

    Uses the heuristic move scoring from ai/heuristics.py which considers:
    - Bridge connections
    - Goal distance
    - Opponent blocking
    - Span bonuses
    - Center bias (early game)

    Args:
        state: Current game state
        rng: Random number generator for tiebreaking

    Returns:
        (row, col) tuple or None if no legal moves
    """
    from ..ai.heuristics import score_moves

    moves = generate_moves(state)
    if not moves:
        return None

    # Score all moves using full heuristics
    scored = score_moves(state, moves)

    # Get best score
    best_score = scored[0][1]

    # Get all moves with best score (for tiebreaking)
    best_moves = [m for m, s in scored if s == best_score]

    # Random tiebreak among best moves
    if rng is None:
        return random.choice(best_moves)
    return rng.choice(best_moves)


def play_random_game(
    seed: int,
    max_moves: int = 220,
    stall_limit: int = 40
) -> dict:
    """Play a complete game with random moves.

    Args:
        seed: Random seed for reproducibility
        max_moves: Maximum moves before declaring draw
        stall_limit: Max moves without progress before declaring stall

    Returns:
        Dict with game info:
        - winner: "red", "black", or "draw"
        - moves: list of (player, row, col)
        - total_moves: number of moves made
        - reason: "win", "stall", "max_moves", or "no_moves"
    """
    from ..game.rules import apply_move, check_winner

    rng = random.Random(seed)
    state = GameState()

    moves_made: List[Tuple[str, int, int]] = []
    stagnation = 0

    # Track progress (simplified: just count pegs near goal edges)
    last_progress = {"red": 0, "black": 0}

    for turn in range(max_moves):
        player = state.to_move
        move = random_move(state, rng)

        if move is None:
            # No legal moves
            return {
                "winner": "draw",
                "moves": moves_made,
                "total_moves": len(moves_made),
                "reason": "no_moves",
                "seed": seed,
            }

        row, col = move
        moves_made.append((player, row, col))
        state = apply_move(state, row, col)

        # Check for winner
        winner = check_winner(state)
        if winner:
            return {
                "winner": winner,
                "moves": moves_made,
                "total_moves": len(moves_made),
                "reason": "win",
                "seed": seed,
            }

        # Simple progress check: count pegs on goal edges
        progress = 0
        for (r, c), p in state.pegs.items():
            if p == "red" and (r == 0 or r == state.board_size - 1):
                progress += 1
            elif p == "black" and (c == 0 or c == state.board_size - 1):
                progress += 1

        current = {"red": 0, "black": 0}
        for (r, c), p in state.pegs.items():
            if p == "red" and (r == 0 or r == state.board_size - 1):
                current["red"] += 1
            elif p == "black" and (c == 0 or c == state.board_size - 1):
                current["black"] += 1

        if current["red"] > last_progress["red"] or current["black"] > last_progress["black"]:
            stagnation = 0
            last_progress = current
        else:
            stagnation += 1

        if stagnation >= stall_limit:
            return {
                "winner": "draw",
                "moves": moves_made,
                "total_moves": len(moves_made),
                "reason": "stall",
                "seed": seed,
            }

    return {
        "winner": "draw",
        "moves": moves_made,
        "total_moves": len(moves_made),
        "reason": "max_moves",
        "seed": seed,
    }


def play_search_game(
    seed: int,
    depth: int = 2,
    max_moves: int = 220,
    stall_limit: int = 40
) -> dict:
    """Play a complete game with minimax search.

    Args:
        seed: Random seed (for reproducibility, though search is deterministic)
        depth: Search depth
        max_moves: Maximum moves before declaring draw
        stall_limit: Max moves without progress before declaring stall

    Returns:
        Dict with game info
    """
    from ..game.rules import apply_move, check_winner

    state = GameState()
    moves_made: List[Tuple[str, int, int]] = []
    stagnation = 0
    last_progress = {"red": 0, "black": 0}

    for turn in range(max_moves):
        player = state.to_move
        move = search_move(state, depth=depth)

        if move is None:
            return {
                "winner": "draw",
                "moves": moves_made,
                "total_moves": len(moves_made),
                "reason": "no_moves",
                "seed": seed,
                "depth": depth,
            }

        row, col = move
        moves_made.append((player, row, col))
        state = apply_move(state, row, col)

        winner = check_winner(state)
        if winner:
            return {
                "winner": winner,
                "moves": moves_made,
                "total_moves": len(moves_made),
                "reason": "win",
                "seed": seed,
                "depth": depth,
            }

        # Progress check
        current = {"red": 0, "black": 0}
        for (r, c), p in state.pegs.items():
            if p == "red" and (r == 0 or r == state.board_size - 1):
                current["red"] += 1
            elif p == "black" and (c == 0 or c == state.board_size - 1):
                current["black"] += 1

        if current["red"] > last_progress["red"] or current["black"] > last_progress["black"]:
            stagnation = 0
            last_progress = current
        else:
            stagnation += 1

        if stagnation >= stall_limit:
            return {
                "winner": "draw",
                "moves": moves_made,
                "total_moves": len(moves_made),
                "reason": "stall",
                "seed": seed,
                "depth": depth,
            }

    return {
        "winner": "draw",
        "moves": moves_made,
        "total_moves": len(moves_made),
        "reason": "max_moves",
        "seed": seed,
        "depth": depth,
    }


def play_greedy_game(
    seed: int,
    max_moves: int = 220,
    stall_limit: int = 40
) -> dict:
    """Play a complete game with greedy moves.

    Same as play_random_game but uses greedy_move instead of random_move.
    """
    from ..game.rules import apply_move, check_winner

    rng = random.Random(seed)
    state = GameState()

    moves_made: List[Tuple[str, int, int]] = []
    stagnation = 0
    last_progress = {"red": 0, "black": 0}

    for turn in range(max_moves):
        player = state.to_move
        move = greedy_move(state, rng)

        if move is None:
            return {
                "winner": "draw",
                "moves": moves_made,
                "total_moves": len(moves_made),
                "reason": "no_moves",
                "seed": seed,
            }

        row, col = move
        moves_made.append((player, row, col))
        state = apply_move(state, row, col)

        winner = check_winner(state)
        if winner:
            return {
                "winner": winner,
                "moves": moves_made,
                "total_moves": len(moves_made),
                "reason": "win",
                "seed": seed,
            }

        # Progress check
        current = {"red": 0, "black": 0}
        for (r, c), p in state.pegs.items():
            if p == "red" and (r == 0 or r == state.board_size - 1):
                current["red"] += 1
            elif p == "black" and (c == 0 or c == state.board_size - 1):
                current["black"] += 1

        if current["red"] > last_progress["red"] or current["black"] > last_progress["black"]:
            stagnation = 0
            last_progress = current
        else:
            stagnation += 1

        if stagnation >= stall_limit:
            return {
                "winner": "draw",
                "moves": moves_made,
                "total_moves": len(moves_made),
                "reason": "stall",
                "seed": seed,
            }

    return {
        "winner": "draw",
        "moves": moves_made,
        "total_moves": len(moves_made),
        "reason": "max_moves",
        "seed": seed,
    }
