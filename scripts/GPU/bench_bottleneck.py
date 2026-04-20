#!/usr/bin/env python3
"""Benchmark to identify GPU batch evaluation bottlenecks.

Measures:
1. Feature extraction time for N moves
2. MLX matmul time for [N,F] @ [F,1]
3. Apply/undo + extraction combined

Usage:
    python3.14 scripts/GPU/bench_bottleneck.py
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.GPU.game.state import GameState
from scripts.GPU.game.rules import apply_move, generate_moves
from scripts.GPU.game.board import is_valid_placement
from scripts.GPU.ai.heuristics import extract_features, DEFAULT_KNOBS
from scripts.GPU.utils.maybe_mlx import try_import_mlx

import numpy as np

# Try MLX
mlx_env = try_import_mlx()
if mlx_env.available:
    mx = mlx_env.mx
    print(f"MLX available: {mx.default_device()}")
else:
    mx = None
    print("MLX not available, skipping GPU benchmarks")


def setup_midgame_state() -> GameState:
    """Create a mid-game state with ~20 pegs per side."""
    state = GameState(board_size=24)

    # Place some pegs to simulate mid-game (alternating red/black turns)
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

    # Interleave red and black moves to maintain turn order
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


def bench_feature_extraction(state: GameState, moves: list, player: str, n_iters: int = 100):
    """Benchmark feature extraction for N moves."""
    print(f"\n=== Feature Extraction Benchmark ===")
    print(f"Moves: {len(moves)}, Iterations: {n_iters}")

    # Warm up
    for r, c in moves[:5]:
        child = apply_move(state, r, c)
        _ = extract_features(child, player)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        for r, c in moves:
            child = apply_move(state, r, c)
            _ = extract_features(child, player)
    elapsed = time.perf_counter() - start

    total_extractions = n_iters * len(moves)
    print(f"Total time: {elapsed:.3f}s")
    print(f"Per extraction: {elapsed / total_extractions * 1000:.3f}ms")
    print(f"Extractions/sec: {total_extractions / elapsed:.0f}")

    return elapsed, len(moves)


def bench_apply_undo_extract(state: GameState, moves: list, player: str, n_iters: int = 100):
    """Benchmark apply/undo with feature extraction.

    Note: If your engine supports in-place apply/undo, switch this to use it.
    For now, this uses apply_move() which returns a new state.
    """
    print(f"\n=== Apply/Undo + Extract Benchmark ===")
    print(f"Moves: {len(moves)}, Iterations: {n_iters}")

    start = time.perf_counter()
    for _ in range(n_iters):
        for r, c in moves:
            child = apply_move(state, r, c)
            _ = extract_features(child, player)
    elapsed = time.perf_counter() - start

    total = n_iters * len(moves)
    print(f"Total time: {elapsed:.3f}s")
    print(f"Per move cycle: {elapsed / total * 1000:.3f}ms")
    print(f"Cycles/sec: {total / elapsed:.0f}")

    return elapsed


def bench_numpy_buffer(state: GameState, moves: list, player: str, n_iters: int = 100):
    """Benchmark filling a preallocated numpy buffer."""
    print(f"\n=== NumPy Buffer Benchmark ===")

    # Get feature count from one extraction
    r0, c0 = moves[0]
    child = apply_move(state, r0, c0)
    sample_features = extract_features(child, player)
    n_features = len(sample_features)
    print(f"Moves: {len(moves)}, Features: {n_features}, Iterations: {n_iters}")

    # Preallocate buffer
    buffer = np.zeros((len(moves), n_features), dtype=np.float32)
    feature_keys = list(sample_features.keys())

    start = time.perf_counter()
    for _ in range(n_iters):
        for i, (r, c) in enumerate(moves):
            child = apply_move(state, r, c)
            features = extract_features(child, player)
            for j, k in enumerate(feature_keys):
                buffer[i, j] = features.get(k, 0.0)
    elapsed = time.perf_counter() - start

    total = n_iters * len(moves)
    print(f"Total time: {elapsed:.3f}s")
    print(f"Per move: {elapsed / total * 1000:.3f}ms")

    return elapsed, buffer, feature_keys


def bench_mlx_matmul(buffer: np.ndarray, n_iters: int = 1000):
    """Benchmark MLX matmul on [N,F] @ [F,1]."""
    if mx is None:
        print("\n=== MLX Matmul Benchmark: SKIPPED (no MLX) ===")
        return None

    print(f"\n=== MLX Matmul Benchmark ===")
    N, F = buffer.shape
    print(f"Shape: [{N}, {F}] @ [{F}, 1], Iterations: {n_iters}")

    # Create GPU tensors
    X_gpu = mx.array(buffer)
    W_gpu = mx.random.normal((F, 1))

    # Warm up
    for _ in range(10):
        result = mx.matmul(X_gpu, W_gpu)
        mx.eval(result)  # Force sync

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        result = mx.matmul(X_gpu, W_gpu)
        mx.eval(result)
    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.3f}s")
    print(f"Per matmul: {elapsed / n_iters * 1000:.4f}ms")
    print(f"Matmuls/sec: {n_iters / elapsed:.0f}")

    return elapsed


def bench_transfer_overhead(buffer: np.ndarray, n_iters: int = 100):
    """Benchmark CPU->GPU transfer overhead."""
    if mx is None:
        print("\n=== Transfer Overhead Benchmark: SKIPPED (no MLX) ===")
        return None

    print(f"\n=== CPU->GPU Transfer Benchmark ===")
    N, F = buffer.shape
    print(f"Shape: [{N}, {F}], Iterations: {n_iters}")

    # Warm up
    for _ in range(10):
        X_gpu = mx.array(buffer)
        mx.eval(X_gpu)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        X_gpu = mx.array(buffer)
        mx.eval(X_gpu)
    elapsed = time.perf_counter() - start

    print(f"Total time: {elapsed:.3f}s")
    print(f"Per transfer: {elapsed / n_iters * 1000:.4f}ms")

    return elapsed


def main():
    print("=" * 60)
    print("BOTTLENECK BENCHMARK")
    print("=" * 60)

    # Setup
    state = setup_midgame_state()
    moves = generate_moves(state)
    player = state.to_move
    print(f"\nSetup: {len(moves)} valid moves, player={player}")

    # 1. Feature extraction
    t_extract, n_moves = bench_feature_extraction(state, moves, player, n_iters=50)

    # 2. Apply/undo + extract
    _ = bench_apply_undo_extract(state, moves, player, n_iters=50)

    # 3. NumPy buffer approach
    t_buffer, buffer, _ = bench_numpy_buffer(state, moves, player, n_iters=50)

    # 4. MLX matmul
    t_matmul = bench_mlx_matmul(buffer, n_iters=1000)

    # 5. Transfer overhead
    t_transfer = bench_transfer_overhead(buffer, n_iters=100)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Normalize to per-batch (scoring all moves once)
    t_extract_batch = t_extract / 50  # per batch
    t_buffer_batch = t_buffer / 50

    print(f"\nPer-batch times (scoring {n_moves} moves):")
    print(f"  Feature extraction:     {t_extract_batch * 1000:.2f}ms")
    print(f"  NumPy buffer fill:      {t_buffer_batch * 1000:.2f}ms")

    if t_matmul is not None:
        t_matmul_one = t_matmul / 1000
        print(f"  MLX matmul:             {t_matmul_one * 1000:.4f}ms")

    if t_transfer is not None:
        t_transfer_one = t_transfer / 100
        print(f"  CPU->GPU transfer:      {t_transfer_one * 1000:.4f}ms")

    print("\n" + "-" * 40)
    print("BOTTLENECK ANALYSIS:")

    if (t_matmul is not None) and (t_transfer is not None):
        gpu_time = t_matmul_one + t_transfer_one
        cpu_time = t_buffer_batch
        ratio = cpu_time / max(1e-12, gpu_time)
        print(f"  CPU (extract+buffer):   {cpu_time * 1000:.2f}ms")
        print(f"  GPU (transfer+matmul):  {gpu_time * 1000:.4f}ms")
        print(f"  Ratio CPU/GPU:          {ratio:.1f}x")

        if ratio > 10:
            print("\n  >> BOTTLENECK: Feature extraction (CPU-bound)")
            print("     Optimize extract_features() before GPU work")
        elif ratio > 2:
            print("\n  >> BOTTLENECK: Mixed - extraction + GPU both matter")
            print("     Worth optimizing both")
        else:
            print("\n  >> BOTTLENECK: GPU matmul or transfer")
            print("     Focus on reducing transfers, batching larger")


if __name__ == "__main__":
    main()
