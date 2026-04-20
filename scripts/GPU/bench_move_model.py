#!/usr/bin/env python3
"""Benchmark GPU move scoring model.

Compares:
1. GPU model forward pass timing
2. CPU heuristic scoring timing

Usage:
    python3 scripts/GPU/bench_move_model.py
"""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.GPU.game.state import GameState
from scripts.GPU.game.rules import apply_move, generate_moves
from scripts.GPU.game.board import is_valid_placement
from scripts.GPU.ai.tensor_repr import state_to_numpy
from scripts.GPU.utils.maybe_mlx import try_import_mlx

# Check MLX availability
_mlx_env = try_import_mlx()


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


def bench_tensor_conversion(state: GameState, n_iters: int = 100):
    """Benchmark tensor conversion."""
    # Warm up
    for _ in range(5):
        _ = state_to_numpy(state)

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = state_to_numpy(state)
    elapsed = time.perf_counter() - start

    return elapsed / n_iters


def bench_gpu_model(state: GameState, moves: list, n_iters: int = 100):
    """Benchmark GPU model forward pass."""
    if not _mlx_env.available:
        return None, "MLX not available"

    from scripts.GPU.ai.move_model import create_model
    from scripts.GPU.ai.tensor_repr import state_to_tensor

    mx = _mlx_env.mx

    model = create_model()
    board_tensor = state_to_tensor(state)

    # Warm up
    for _ in range(5):
        _ = model.score_all_moves(board_tensor, moves)
        mx.eval(_)  # Force evaluation

    start = time.perf_counter()
    for _ in range(n_iters):
        scores = model.score_all_moves(board_tensor, moves)
        mx.eval(scores)  # Force evaluation
    elapsed = time.perf_counter() - start

    return elapsed / n_iters, None


def bench_gpu_model_with_encoding(state: GameState, moves: list, n_iters: int = 100):
    """Benchmark GPU model including tensor encoding."""
    if not _mlx_env.available:
        return None, "MLX not available"

    from scripts.GPU.ai.move_model import MoveRanker

    mx = _mlx_env.mx

    model = MoveRanker()

    # Warm up
    for _ in range(5):
        _ = model.score_moves_from_state(state, moves)
        mx.eval(_)

    start = time.perf_counter()
    for _ in range(n_iters):
        scores = model.score_moves_from_state(state, moves)
        mx.eval(scores)
    elapsed = time.perf_counter() - start

    return elapsed / n_iters, None


def bench_cpu_heuristics(state: GameState, moves: list, n_iters: int = 20):
    """Benchmark CPU heuristic scoring."""
    from scripts.GPU.ai.heuristics import score_moves_batch, DEFAULT_KNOBS

    # Warm up
    for _ in range(2):
        _ = score_moves_batch(state, moves, knobs=DEFAULT_KNOBS)

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = score_moves_batch(state, moves, knobs=DEFAULT_KNOBS)
    elapsed = time.perf_counter() - start

    return elapsed / n_iters


def main():
    print("=" * 60)
    print("MOVE MODEL BENCHMARK")
    print("=" * 60)

    state = setup_midgame_state()
    moves = generate_moves(state)

    print(f"\nState: {len(state.pegs)} pegs, {len(moves)} moves, player={state.to_move}")
    print(f"MLX available: {_mlx_env.available}")

    print("\n" + "-" * 60)
    print("TENSOR CONVERSION (NumPy)")
    print("-" * 60)

    t_tensor = bench_tensor_conversion(state)
    print(f"  state_to_numpy: {t_tensor*1000:.2f}ms")

    print("\n" + "-" * 60)
    print("GPU MODEL FORWARD PASS")
    print("-" * 60)

    if _mlx_env.available:
        # Test forward pass only (pre-encoded tensor)
        t_forward, err = bench_gpu_model(state, moves)
        if err:
            print(f"  Error: {err}")
        else:
            print(f"  Forward pass only: {t_forward*1000:.2f}ms ({len(moves)} moves)")

        # Test with encoding included
        t_full, err = bench_gpu_model_with_encoding(state, moves)
        if err:
            print(f"  Error: {err}")
        else:
            print(f"  With encoding:     {t_full*1000:.2f}ms ({len(moves)} moves)")
    else:
        print("  Skipped (MLX not available)")

    print("\n" + "-" * 60)
    print("CPU HEURISTIC SCORING")
    print("-" * 60)

    t_cpu = bench_cpu_heuristics(state, moves)
    print(f"  score_moves_batch: {t_cpu*1000:.2f}ms ({len(moves)} moves)")

    print("\n" + "-" * 60)
    print("COMPARISON")
    print("-" * 60)

    if _mlx_env.available and t_full is not None:
        speedup = t_cpu / t_full
        print(f"  GPU (with encoding) vs CPU heuristics: {speedup:.1f}x speedup")

        if t_forward is not None:
            speedup_forward = t_cpu / t_forward
            print(f"  GPU (forward only) vs CPU heuristics:  {speedup_forward:.1f}x speedup")
    else:
        print("  Cannot compare (GPU not available)")

    # Test different batch sizes
    print("\n" + "-" * 60)
    print("SCALING WITH MOVE COUNT")
    print("-" * 60)

    if _mlx_env.available:
        from scripts.GPU.ai.move_model import create_model
        from scripts.GPU.ai.tensor_repr import state_to_tensor

        mx = _mlx_env.mx
        model = create_model()
        board_tensor = state_to_tensor(state)

        for n_moves in [50, 100, 200, 500]:
            subset = moves[:min(n_moves, len(moves))]

            # Warm up
            for _ in range(3):
                _ = model.score_all_moves(board_tensor, subset)
                mx.eval(_)

            start = time.perf_counter()
            for _ in range(50):
                scores = model.score_all_moves(board_tensor, subset)
                mx.eval(scores)
            elapsed = (time.perf_counter() - start) / 50

            print(f"  {len(subset):3d} moves: {elapsed*1000:.2f}ms")


if __name__ == "__main__":
    main()
