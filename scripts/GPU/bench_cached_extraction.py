#!/usr/bin/env python3
"""Benchmark cached vs uncached feature extraction.

Compares:
1. Original batch_extract_features (no caching)
2. New batch_extract_features_cached (opponent features cached)

Usage:
    python3.14 scripts/GPU/bench_cached_extraction.py
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.GPU.game.state import GameState
from scripts.GPU.game.rules import apply_move, generate_moves
from scripts.GPU.game.board import is_valid_placement
from scripts.GPU.ai.batch_eval import batch_extract_features, batch_extract_features_cached


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


def bench_original(parent_state: GameState, child_states: list, player: str, n_iters: int = 20):
    """Benchmark original batch_extract_features."""
    base_turn = len(parent_state.move_history)
    friendly_count = sum(1 for p in parent_state.pegs.values() if p == player)
    opponent_count = len(parent_state.pegs) - friendly_count

    # Warm up
    for _ in range(2):
        _ = batch_extract_features(child_states, player, base_turn, friendly_count, opponent_count)

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = batch_extract_features(child_states, player, base_turn, friendly_count, opponent_count)
    elapsed = time.perf_counter() - start

    return elapsed / n_iters


def bench_cached(parent_state: GameState, child_states: list, player: str, n_iters: int = 20):
    """Benchmark new batch_extract_features_cached."""
    base_turn = len(parent_state.move_history)
    friendly_count = sum(1 for p in parent_state.pegs.values() if p == player)
    opponent_count = len(parent_state.pegs) - friendly_count

    # Warm up
    for _ in range(2):
        _ = batch_extract_features_cached(
            parent_state, child_states, player, base_turn, friendly_count, opponent_count
        )

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = batch_extract_features_cached(
            parent_state, child_states, player, base_turn, friendly_count, opponent_count
        )
    elapsed = time.perf_counter() - start

    return elapsed / n_iters


def main():
    print("=" * 60)
    print("CACHED FEATURE EXTRACTION BENCHMARK")
    print("=" * 60)

    parent_state = setup_midgame_state()
    moves = generate_moves(parent_state)
    player = parent_state.to_move

    print(f"\nState: {len(parent_state.pegs)} pegs, player={player}")
    print(f"Total moves: {len(moves)}")

    # Test with different batch sizes (simulating top-k)
    batch_sizes = [10, 25, 50, 100]

    print(f"\n{'Batch Size':<12} {'Original (ms)':<15} {'Cached (ms)':<15} {'Speedup':>10}")
    print("-" * 55)

    for k in batch_sizes:
        sample_moves = moves[:k]
        child_states = [apply_move(parent_state, r, c) for r, c in sample_moves]

        t_original = bench_original(parent_state, child_states, player)
        t_cached = bench_cached(parent_state, child_states, player)

        speedup = t_original / t_cached if t_cached > 0 else 0
        print(f"{k:<12} {t_original*1000:>13.2f} {t_cached*1000:>13.2f} {speedup:>10.2f}x")

    # Verify feature parity
    print("\n" + "=" * 60)
    print("FEATURE PARITY CHECK")
    print("=" * 60)

    sample_moves = moves[:5]
    child_states = [apply_move(parent_state, r, c) for r, c in sample_moves]
    base_turn = len(parent_state.move_history)
    friendly_count = sum(1 for p in parent_state.pegs.values() if p == player)
    opponent_count = len(parent_state.pegs) - friendly_count

    original_features = batch_extract_features(
        child_states, player, base_turn, friendly_count, opponent_count
    )
    cached_features = batch_extract_features_cached(
        parent_state, child_states, player, base_turn, friendly_count, opponent_count
    )

    all_match = True
    for i, (orig, cached) in enumerate(zip(original_features, cached_features)):
        for key in orig:
            if key not in cached:
                print(f"  Move {i}: Missing key '{key}' in cached")
                all_match = False
            elif abs(orig[key] - cached[key]) > 1e-6:
                print(f"  Move {i}: Mismatch on '{key}': {orig[key]} vs {cached[key]}")
                all_match = False

    if all_match:
        print("\n  All features match between original and cached versions.")
    else:
        print("\n  WARNING: Feature mismatch detected!")


if __name__ == "__main__":
    main()
