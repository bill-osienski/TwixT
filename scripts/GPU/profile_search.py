#!/usr/bin/env python3
"""Profile search to identify remaining bottlenecks.

Usage:
    python3.14 scripts/GPU/profile_search.py
"""

import cProfile
import pstats
import io
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.GPU.game.state import GameState
from scripts.GPU.game.rules import apply_move
from scripts.GPU.game.board import is_valid_placement
from scripts.GPU.ai.search import choose_move
from scripts.GPU.ai.heuristics import DEFAULT_KNOBS


def setup_midgame_state() -> GameState:
    """Create a mid-game state with ~20 pegs per side."""
    state = GameState(board_size=24)

    red_positions = [
        (5, 10), (6, 12), (7, 11), (8, 13), (9, 12),
        (10, 14), (11, 13), (12, 15), (13, 14), (14, 16),
        (6, 8), (7, 9), (8, 7), (9, 8), (10, 6),
    ]
    black_positions = [
        (10, 5), (11, 6), (12, 4), (13, 5), (14, 3),
        (10, 18), (11, 17), (12, 19), (13, 18), (14, 20),
        (8, 10), (9, 11), (10, 9), (11, 10), (12, 8),
    ]

    for i in range(max(len(red_positions), len(black_positions))):
        if i < len(red_positions):
            r, c = red_positions[i]
            if is_valid_placement(state, state.to_move, r, c):
                state = apply_move(state, r, c)
        if i < len(black_positions):
            r, c = black_positions[i]
            if is_valid_placement(state, state.to_move, r, c):
                state = apply_move(state, r, c)

    return state


def main():
    print("=" * 60)
    print("SEARCH PROFILER")
    print("=" * 60)

    state = setup_midgame_state()
    print(f"\nState: {len(state.pegs)} pegs, player={state.to_move}")

    # Profile depth 2 search (3 iterations)
    print("\n--- Profiling depth 2 search (3 iterations) ---\n")

    pr = cProfile.Profile()
    pr.enable()

    for _ in range(3):
        _ = choose_move(state, DEFAULT_KNOBS, depth=2, top_n=20, use_value_model=False)

    pr.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

    print("\n" + "=" * 60)
    print("TOP FUNCTIONS BY TOTAL TIME")
    print("=" * 60 + "\n")

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
    ps2.print_stats(20)
    print(s2.getvalue())


if __name__ == "__main__":
    main()
