#!/usr/bin/env python3
"""Verify ONNX export matches MLX model outputs.

Run on multiple boards with different move counts to catch edge cases.

Usage:
    python -m scripts.GPU.alphazero.verify_export \\
        --weights checkpoints/alphazero/model_iter_0100.safetensors \\
        --onnx model.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def verify_forward_parity(
    mlx_model,
    onnx_path: str,
    test_cases: List[Tuple[np.ndarray, List[Tuple[int, int]]]],
    tolerance: float = 1e-4,
) -> bool:
    """Compare MLX and ONNX outputs on test positions.

    IMPORTANT: mlx_model must be in eval mode (mlx_model.eval()) to match
    ONNX behavior. In eval mode, BatchNorm uses running statistics instead
    of batch statistics.

    Args:
        mlx_model: MLX AlphaZeroNetwork (must be in eval mode)
        onnx_path: Path to exported ONNX model
        test_cases: List of (board_tensor_hwc, legal_moves) tuples
                   board_tensor is (H, W, C) in NHWC format
        tolerance: Maximum allowed difference

    Returns:
        True if all tests pass

    Raises:
        AssertionError if outputs differ by more than tolerance
    """
    import mlx.core as mx
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)

    all_passed = True
    for i, (board_hwc, moves) in enumerate(test_cases):
        # MLX forward (expects NHWC: B, H, W, C)
        board_mlx = mx.array(board_hwc[None, ...])  # (1, H, W, C)
        policy_mlx, value_mlx = mlx_model(board_mlx, moves)
        mx.eval(policy_mlx, value_mlx)

        # ONNX forward (expects NCHW: B, C, H, W)
        # board_hwc is (H, W, C), need to transpose to (C, H, W) then add batch
        board_chw = np.transpose(board_hwc, (2, 0, 1))  # (C, H, W)
        board_onnx = board_chw[None, ...].astype(np.float32)  # (1, C, H, W)

        # Prepare padded move inputs
        move_rows = np.zeros(512, dtype=np.int64)
        move_cols = np.zeros(512, dtype=np.int64)
        move_mask = np.zeros(512, dtype=np.float32)

        for j, (r, c) in enumerate(moves):
            move_rows[j] = r
            move_cols[j] = c
            move_mask[j] = 1.0

        # Run ONNX inference
        policy_onnx, value_onnx = session.run(
            None,
            {
                "board": board_onnx,
                "move_rows": move_rows,
                "move_cols": move_cols,
                "move_mask": move_mask,
            }
        )

        # Compare (only valid moves for policy)
        policy_mlx_np = np.array(policy_mlx)
        policy_onnx_valid = policy_onnx[:len(moves)]

        policy_diff = np.max(np.abs(policy_mlx_np - policy_onnx_valid))
        value_diff = abs(float(value_mlx) - float(value_onnx))

        passed = policy_diff < tolerance and value_diff < tolerance
        status = "PASS" if passed else "FAIL"

        print(f"  Board {i} ({len(moves)} moves): policy_diff={policy_diff:.6f}, "
              f"value_diff={value_diff:.6f} [{status}]")

        if not passed:
            all_passed = False

        # Additional check: masked positions should be -1e9
        if len(moves) < 512:
            masked_values = policy_onnx[len(moves):]
            if not np.allclose(masked_values, -1e9, rtol=1e-3):
                print(f"    WARNING: Masked logits not -1e9: {masked_values[:5]}...")

    return all_passed


def generate_test_cases(n_cases: int = 10, seed: int = 42) -> List[Tuple[np.ndarray, List[Tuple[int, int]]]]:
    """Generate random test cases for verification.

    Returns list of (board_hwc, moves) tuples where board is (H, W, C) format.
    """
    from scripts.GPU.alphazero.game import TwixtState

    rng = np.random.default_rng(seed)
    test_cases = []

    for i in range(n_cases):
        # Create game state with some random moves
        state = TwixtState()
        n_moves = rng.integers(0, 30)

        for _ in range(n_moves):
            legal = state.legal_moves()
            if not legal or state.is_terminal():
                break
            move = legal[rng.integers(len(legal))]
            state = state.apply_move(move)

        if state.is_terminal():
            # Reset to a simpler state if we hit terminal
            state = TwixtState()
            for _ in range(min(5, n_moves)):
                legal = state.legal_moves()
                if not legal:
                    break
                move = legal[rng.integers(len(legal))]
                state = state.apply_move(move)

        # Get board tensor (C, H, W) and transpose to (H, W, C)
        board_chw = state.to_tensor()  # (C, H, W)
        board_hwc = np.transpose(board_chw, (1, 2, 0)).astype(np.float32)  # (H, W, C)

        # Get subset of legal moves (varying counts)
        legal = state.legal_moves()
        n_test_moves = min(len(legal), rng.integers(5, 100))
        moves = legal[:n_test_moves]

        test_cases.append((board_hwc, moves))

    return test_cases


def main():
    parser = argparse.ArgumentParser(
        description="Verify ONNX export matches MLX model"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to MLX weights (.safetensors)",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="Network hidden channels (default: 128)",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=6,
        help="Network residual blocks (default: 6)",
    )
    parser.add_argument(
        "--n-tests",
        type=int,
        default=10,
        help="Number of test cases (default: 10)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Maximum allowed difference (default: 1e-4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for test generation (default: 42)",
    )

    args = parser.parse_args()

    # Import MLX modules
    from scripts.GPU.alphazero.network import create_network

    print("=" * 60)
    print("ONNX EXPORT VERIFICATION")
    print("=" * 60)
    print()

    # Load MLX model
    print(f"Loading MLX model from {args.weights}...")
    mlx_model = create_network(hidden=args.hidden, n_blocks=args.blocks)
    mlx_model.load_weights(args.weights)
    mlx_model.eval()  # Use eval mode to match ONNX (running stats, not batch stats)

    # Generate test cases
    print(f"Generating {args.n_tests} test cases...")
    test_cases = generate_test_cases(n_cases=args.n_tests, seed=args.seed)

    # Verify
    print(f"\nVerifying against {args.onnx}...")
    print(f"Tolerance: {args.tolerance}")
    print()

    passed = verify_forward_parity(
        mlx_model,
        args.onnx,
        test_cases,
        tolerance=args.tolerance,
    )

    print()
    if passed:
        print("=" * 60)
        print(f"PASSED: All {args.n_tests} test cases match within tolerance")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("FAILED: Some test cases exceeded tolerance")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
