#!/usr/bin/env python3
"""Profile extract_features() to identify expensive components.

Usage:
    python3.14 scripts/GPU/profile_features.py
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.GPU.game.state import GameState
from scripts.GPU.game.rules import apply_move, generate_moves
from scripts.GPU.game.board import is_valid_placement
from scripts.GPU.ai.heuristics import (
    evaluate_connected_paths,
    evaluate_potential_connections,
    evaluate_edge_progress,
    component_metrics,
    compute_frontier,
    DEFAULT_KNOBS,
)


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


def profile_component(name: str, func, state: GameState, player: str, n_iters: int = 1000):
    """Profile a single feature extraction component."""
    # Warm up
    for _ in range(10):
        func(state, player)

    start = time.perf_counter()
    for _ in range(n_iters):
        func(state, player)
    elapsed = time.perf_counter() - start

    per_call_us = (elapsed / n_iters) * 1_000_000
    return per_call_us


def profile_component_with_knobs(name: str, func, state: GameState, player: str, knobs: dict, n_iters: int = 1000):
    """Profile a component that takes knobs parameter."""
    # Warm up
    for _ in range(10):
        func(state, player, knobs)

    start = time.perf_counter()
    for _ in range(n_iters):
        func(state, player, knobs)
    elapsed = time.perf_counter() - start

    per_call_us = (elapsed / n_iters) * 1_000_000
    return per_call_us


def main():
    print("=" * 60)
    print("FEATURE EXTRACTION PROFILER")
    print("=" * 60)

    state = setup_midgame_state()
    player = "red"
    opponent = "black"
    k = DEFAULT_KNOBS
    n_iters = 1000

    print(f"\nState: {len(state.pegs)} pegs, {len(state.bridges)} bridges")
    print(f"Iterations per component: {n_iters}")

    results = []

    # Profile each component
    print("\n--- Profiling individual components ---\n")

    # component_metrics (called twice in extract_features)
    t = profile_component("component_metrics(player)", component_metrics, state, player, n_iters)
    results.append(("component_metrics(player)", t))
    print(f"  component_metrics(player):     {t:8.2f} µs")

    t = profile_component("component_metrics(opponent)", component_metrics, state, opponent, n_iters)
    results.append(("component_metrics(opponent)", t))
    print(f"  component_metrics(opponent):   {t:8.2f} µs")

    # evaluate_connected_paths (called twice)
    t = profile_component_with_knobs("connected_paths(player)", evaluate_connected_paths, state, player, k, n_iters)
    results.append(("connected_paths(player)", t))
    print(f"  connected_paths(player):       {t:8.2f} µs")

    t = profile_component_with_knobs("connected_paths(opponent)", evaluate_connected_paths, state, opponent, k, n_iters)
    results.append(("connected_paths(opponent)", t))
    print(f"  connected_paths(opponent):     {t:8.2f} µs")

    # evaluate_potential_connections (called twice)
    t = profile_component_with_knobs("potential_connections(player)", evaluate_potential_connections, state, player, k, n_iters)
    results.append(("potential_connections(player)", t))
    print(f"  potential_connections(player): {t:8.2f} µs")

    t = profile_component_with_knobs("potential_connections(opp)", evaluate_potential_connections, state, opponent, k, n_iters)
    results.append(("potential_connections(opp)", t))
    print(f"  potential_connections(opp):    {t:8.2f} µs")

    # evaluate_edge_progress (called twice)
    t = profile_component_with_knobs("edge_progress(player)", evaluate_edge_progress, state, player, k, n_iters)
    results.append(("edge_progress(player)", t))
    print(f"  edge_progress(player):         {t:8.2f} µs")

    t = profile_component_with_knobs("edge_progress(opponent)", evaluate_edge_progress, state, opponent, k, n_iters)
    results.append(("edge_progress(opponent)", t))
    print(f"  edge_progress(opponent):       {t:8.2f} µs")

    # compute_frontier (called once)
    t = profile_component("compute_frontier", compute_frontier, state, player, n_iters)
    results.append(("compute_frontier", t))
    print(f"  compute_frontier:              {t:8.2f} µs")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (sorted by time)")
    print("=" * 60)

    results.sort(key=lambda x: x[1], reverse=True)
    total = sum(t for _, t in results)

    print(f"\n{'Component':<35} {'Time (µs)':>12} {'% of total':>12}")
    print("-" * 60)
    for name, t in results:
        pct = (t / total) * 100
        print(f"{name:<35} {t:>12.2f} {pct:>11.1f}%")
    print("-" * 60)
    print(f"{'TOTAL':<35} {total:>12.2f} {'100.0':>11}%")

    # Estimate for full extract_features()
    print(f"\n--- Estimated extract_features() time: {total:.0f} µs ({total/1000:.2f} ms) ---")

    # Profile with apply_move overhead
    print("\n" + "=" * 60)
    print("WITH APPLY_MOVE OVERHEAD")
    print("=" * 60)

    moves = generate_moves(state)[:50]  # Sample 50 moves
    print(f"\nProfiling apply_move + component_metrics for {len(moves)} moves...")

    # Just apply_move
    start = time.perf_counter()
    for r, c in moves:
        child = apply_move(state, r, c)
    elapsed = time.perf_counter() - start
    apply_time = (elapsed / len(moves)) * 1_000_000
    print(f"  apply_move alone:              {apply_time:8.2f} µs")

    # apply_move + component_metrics (the expensive combo)
    start = time.perf_counter()
    for r, c in moves:
        child = apply_move(state, r, c)
        _ = component_metrics(child, player)
        _ = component_metrics(child, opponent)
    elapsed = time.perf_counter() - start
    combo_time = (elapsed / len(moves)) * 1_000_000
    print(f"  apply + 2x component_metrics:  {combo_time:8.2f} µs")

    print(f"\n  component_metrics overhead:    {combo_time - apply_time:8.2f} µs")


if __name__ == "__main__":
    main()
