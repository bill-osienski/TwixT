#!/usr/bin/env python3
"""Tests for ONNX export functionality.

Run with: python3 tests/test_onnx_export.py
"""
import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Channel count used by ONNX test shapes. Follows NUM_CHANNELS so the
# tests exercise whatever the current architecture uses (24 pre-Phase-2,
# 30 after) without hardcoding.
from scripts.GPU.alphazero.game.twixt_state import NUM_CHANNELS as _ONNX_IN_CH

# Move-tensor size baked into the ONNX export. Tracks the OnnxAlphaZero
# default (576 = 24*24, the true max legal moves on a 24x24 board) so
# bumps to that constant don't silently drift the tests.
import inspect as _inspect
from scripts.GPU.alphazero.export_onnx import OnnxAlphaZero as _OnnxAZ
_MAX_MOVES = _inspect.signature(_OnnxAZ).parameters["max_moves"].default


def test_pytorch_model_forward():
    """Test PyTorch model forward pass works."""
    import torch
    from scripts.GPU.alphazero.export_onnx import OnnxAlphaZero

    model = OnnxAlphaZero(hidden=64, n_blocks=2)
    model.eval()

    # Create inputs
    board = torch.randn(1, _ONNX_IN_CH, 24, 24)  # NCHW
    move_rows = torch.zeros(_MAX_MOVES, dtype=torch.long)
    move_cols = torch.zeros(_MAX_MOVES, dtype=torch.long)
    move_mask = torch.zeros(_MAX_MOVES)

    # Set some valid moves
    move_rows[:5] = torch.tensor([1, 2, 3, 4, 5])
    move_cols[:5] = torch.tensor([1, 2, 3, 4, 5])
    move_mask[:5] = 1.0

    # Forward pass
    with torch.no_grad():
        policy, value = model(board, move_rows, move_cols, move_mask)

    # Check shapes
    assert policy.shape == (_MAX_MOVES,), f"Policy shape: {policy.shape}"
    assert value.ndim == 0, f"Value shape: {value.shape}"

    # Check value range
    assert -1.0 <= float(value) <= 1.0, f"Value out of range: {float(value)}"

    # Check masked values are -1e9
    assert torch.allclose(policy[5:], torch.full((_MAX_MOVES - 5,), -1e9)), "Masked logits should be -1e9"

    print("PASS: PyTorch model forward pass")


def test_conv_weight_conversion():
    """Test conv weight layout conversion."""
    import numpy as np
    from scripts.GPU.alphazero.export_onnx import convert_conv_weight

    # MLX Conv2d weight shape: (out, kH, kW, in)
    mlx_weight = np.arange(24).reshape(2, 3, 2, 2).astype(np.float32)

    # Convert to PyTorch: (out, in, kH, kW)
    torch_weight = convert_conv_weight(mlx_weight)

    assert torch_weight.shape == (2, 2, 3, 2), f"Wrong shape: {torch_weight.shape}"

    # Verify a specific element
    # mlx_weight[out, kh, kw, in] should map to torch_weight[out, in, kh, kw]
    for o in range(2):
        for kh in range(3):
            for kw in range(2):
                for i in range(2):
                    assert mlx_weight[o, kh, kw, i] == torch_weight[o, i, kh, kw], (
                        f"Mismatch at [{o},{kh},{kw},{i}]"
                    )

    print("PASS: Conv weight conversion")


def test_weight_transfer():
    """Test weight transfer from MLX to PyTorch."""
    import torch
    import mlx.core as mx
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.export_onnx import (
        OnnxAlphaZero, flatten_mlx_params, convert_weights
    )

    # Create MLX model
    mlx_model = create_network(hidden=64, n_blocks=2)

    # Create PyTorch model
    pytorch_model = OnnxAlphaZero(hidden=64, n_blocks=2)

    # Transfer weights
    mlx_params = flatten_mlx_params(mlx_model.parameters())
    convert_weights(mlx_params, pytorch_model)

    # Check that weights were transferred (not all zeros)
    state_dict = pytorch_model.state_dict()

    # Check encoder conv1
    assert "encoder_conv1.weight" in state_dict
    assert torch.any(state_dict["encoder_conv1.weight"] != 0), "Weights should not be all zeros"

    # Check policy head
    assert "policy_fc.weight" in state_dict
    assert "policy_out.weight" in state_dict

    # Check value head
    assert "value_fc1.weight" in state_dict
    assert "value_fc2.weight" in state_dict

    print("PASS: Weight transfer")


def test_export_and_load():
    """Test full export and ONNX load."""
    import onnxruntime as ort
    import numpy as np
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.export_onnx import export_to_onnx

    # Create MLX model
    mlx_model = create_network(hidden=64, n_blocks=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "test_model.onnx"

        # Export
        export_to_onnx(mlx_model, str(onnx_path), hidden=64, n_blocks=2)

        assert onnx_path.exists(), "ONNX file should exist"

        # Load with ONNX Runtime
        session = ort.InferenceSession(str(onnx_path))

        # Check inputs
        inputs = {inp.name for inp in session.get_inputs()}
        assert inputs == {"board", "move_rows", "move_cols", "move_mask"}, f"Wrong inputs: {inputs}"

        # Check outputs
        outputs = {out.name for out in session.get_outputs()}
        assert outputs == {"policy_logits", "value"}, f"Wrong outputs: {outputs}"

        # Run inference
        board = np.random.randn(1, _ONNX_IN_CH, 24, 24).astype(np.float32)
        move_rows = np.zeros(_MAX_MOVES, dtype=np.int64)
        move_cols = np.zeros(_MAX_MOVES, dtype=np.int64)
        move_mask = np.zeros(_MAX_MOVES, dtype=np.float32)
        move_mask[:10] = 1.0

        policy, value = session.run(
            None,
            {
                "board": board,
                "move_rows": move_rows,
                "move_cols": move_cols,
                "move_mask": move_mask,
            }
        )

        assert policy.shape == (_MAX_MOVES,), f"Policy shape: {policy.shape}"
        assert -1.0 <= float(value) <= 1.0, f"Value out of range: {float(value)}"

    print("PASS: Export and load")


def test_parity_simple():
    """Test MLX vs ONNX parity on simple case."""
    import numpy as np
    import mlx.core as mx
    import onnxruntime as ort
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.export_onnx import export_to_onnx

    # Create and export model
    mlx_model = create_network(hidden=64, n_blocks=2)
    mlx_model.eval()  # Put in eval mode to use running stats

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "test_model.onnx"
        export_to_onnx(mlx_model, str(onnx_path), hidden=64, n_blocks=2)

        session = ort.InferenceSession(str(onnx_path))

        # Create test input (H, W, C format for MLX)
        board_hwc = np.random.randn(24, 24, _ONNX_IN_CH).astype(np.float32)
        moves = [(5, 5), (6, 6), (7, 7)]

        # MLX forward
        board_mlx = mx.array(board_hwc[None, ...])  # (1, H, W, C)
        policy_mlx, value_mlx = mlx_model(board_mlx, moves)
        mx.eval(policy_mlx, value_mlx)

        # ONNX forward
        board_chw = np.transpose(board_hwc, (2, 0, 1))  # (C, H, W)
        board_onnx = board_chw[None, ...].astype(np.float32)  # (1, C, H, W)

        move_rows = np.zeros(_MAX_MOVES, dtype=np.int64)
        move_cols = np.zeros(_MAX_MOVES, dtype=np.int64)
        move_mask = np.zeros(_MAX_MOVES, dtype=np.float32)

        for i, (r, c) in enumerate(moves):
            move_rows[i] = r
            move_cols[i] = c
            move_mask[i] = 1.0

        policy_onnx, value_onnx = session.run(
            None,
            {
                "board": board_onnx,
                "move_rows": move_rows,
                "move_cols": move_cols,
                "move_mask": move_mask,
            }
        )

        # Compare
        policy_mlx_np = np.array(policy_mlx)
        policy_onnx_valid = policy_onnx[:len(moves)]

        policy_diff = np.max(np.abs(policy_mlx_np - policy_onnx_valid))
        value_diff = abs(float(value_mlx) - float(value_onnx))

        assert policy_diff < 1e-4, f"Policy diff too large: {policy_diff}"
        assert value_diff < 1e-4, f"Value diff too large: {value_diff}"

    print(f"PASS: Parity check (policy_diff={policy_diff:.6f}, value_diff={value_diff:.6f})")


def test_parity_multiple_boards():
    """Test parity on multiple game positions."""
    import numpy as np
    import mlx.core as mx
    import onnxruntime as ort
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.game import TwixtState
    from scripts.GPU.alphazero.export_onnx import export_to_onnx

    mlx_model = create_network(hidden=64, n_blocks=2)
    mlx_model.eval()  # Put in eval mode to use running stats

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "test_model.onnx"
        export_to_onnx(mlx_model, str(onnx_path), hidden=64, n_blocks=2)

        session = ort.InferenceSession(str(onnx_path))

        rng = np.random.default_rng(42)
        max_policy_diff = 0.0
        max_value_diff = 0.0

        for test_idx in range(10):
            # Create game state
            state = TwixtState()
            for _ in range(rng.integers(0, 20)):
                legal = state.legal_moves()
                if not legal or state.is_terminal():
                    break
                move = legal[rng.integers(len(legal))]
                state = state.apply_move(move)

            if state.is_terminal():
                state = TwixtState()

            # Get board and moves
            board_chw = state.to_tensor()
            board_hwc = np.transpose(board_chw, (1, 2, 0)).astype(np.float32)
            moves = state.legal_moves()[:50]  # Cap at 50 moves

            # MLX forward
            board_mlx = mx.array(board_hwc[None, ...])
            policy_mlx, value_mlx = mlx_model(board_mlx, moves)
            mx.eval(policy_mlx, value_mlx)

            # ONNX forward
            board_onnx = np.transpose(board_hwc, (2, 0, 1))[None, ...].astype(np.float32)
            move_rows = np.zeros(_MAX_MOVES, dtype=np.int64)
            move_cols = np.zeros(_MAX_MOVES, dtype=np.int64)
            move_mask = np.zeros(_MAX_MOVES, dtype=np.float32)

            for i, (r, c) in enumerate(moves):
                move_rows[i] = r
                move_cols[i] = c
                move_mask[i] = 1.0

            policy_onnx, value_onnx = session.run(
                None,
                {
                    "board": board_onnx,
                    "move_rows": move_rows,
                    "move_cols": move_cols,
                    "move_mask": move_mask,
                }
            )

            # Compare
            policy_mlx_np = np.array(policy_mlx)
            policy_onnx_valid = policy_onnx[:len(moves)]

            policy_diff = np.max(np.abs(policy_mlx_np - policy_onnx_valid))
            value_diff = abs(float(value_mlx) - float(value_onnx))

            max_policy_diff = max(max_policy_diff, policy_diff)
            max_value_diff = max(max_value_diff, value_diff)

            assert policy_diff < 1e-4, f"Board {test_idx}: policy diff {policy_diff}"
            assert value_diff < 1e-4, f"Board {test_idx}: value diff {value_diff}"

    print(f"PASS: Multiple boards (max policy_diff={max_policy_diff:.6f}, "
          f"max value_diff={max_value_diff:.6f})")


def test_masked_logits():
    """Test that invalid positions get -1e9 masking."""
    import numpy as np
    import onnxruntime as ort
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.export_onnx import export_to_onnx

    mlx_model = create_network(hidden=64, n_blocks=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "test_model.onnx"
        export_to_onnx(mlx_model, str(onnx_path), hidden=64, n_blocks=2)

        session = ort.InferenceSession(str(onnx_path))

        # Only 3 valid moves
        board = np.random.randn(1, _ONNX_IN_CH, 24, 24).astype(np.float32)
        move_rows = np.zeros(_MAX_MOVES, dtype=np.int64)
        move_cols = np.zeros(_MAX_MOVES, dtype=np.int64)
        move_mask = np.zeros(_MAX_MOVES, dtype=np.float32)

        move_rows[:3] = [5, 6, 7]
        move_cols[:3] = [5, 6, 7]
        move_mask[:3] = 1.0

        policy, value = session.run(
            None,
            {
                "board": board,
                "move_rows": move_rows,
                "move_cols": move_cols,
                "move_mask": move_mask,
            }
        )

        # First 3 should be valid (not -1e9)
        assert not np.allclose(policy[:3], -1e9), "Valid logits should not be -1e9"

        # Rest should be -1e9
        assert np.allclose(policy[3:], -1e9, rtol=1e-3), (
            f"Invalid logits should be -1e9, got {policy[3:6]}"
        )

    print("PASS: Masked logits")


def test_move_order_invariance():
    """Test that move order doesn't affect logits (catches sorting bugs).

    This test runs ONNX twice with same moves in different order,
    re-associates logits by (row, col), and verifies they match.
    """
    import numpy as np
    import onnxruntime as ort
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.export_onnx import export_to_onnx

    mlx_model = create_network(hidden=64, n_blocks=2)
    mlx_model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "test_model.onnx"
        export_to_onnx(mlx_model, str(onnx_path), hidden=64, n_blocks=2)

        session = ort.InferenceSession(str(onnx_path))

        # Fixed board
        np.random.seed(123)
        board = np.random.randn(1, _ONNX_IN_CH, 24, 24).astype(np.float32)

        # Original move order
        moves_original = [(5, 5), (10, 10), (15, 3), (3, 15), (12, 8)]

        move_rows1 = np.zeros(_MAX_MOVES, dtype=np.int64)
        move_cols1 = np.zeros(_MAX_MOVES, dtype=np.int64)
        move_mask1 = np.zeros(_MAX_MOVES, dtype=np.float32)

        for i, (r, c) in enumerate(moves_original):
            move_rows1[i] = r
            move_cols1[i] = c
            move_mask1[i] = 1.0

        policy1, value1 = session.run(
            None,
            {
                "board": board,
                "move_rows": move_rows1,
                "move_cols": move_cols1,
                "move_mask": move_mask1,
            }
        )

        # Shuffled move order
        moves_shuffled = [(15, 3), (5, 5), (12, 8), (3, 15), (10, 10)]

        move_rows2 = np.zeros(_MAX_MOVES, dtype=np.int64)
        move_cols2 = np.zeros(_MAX_MOVES, dtype=np.int64)
        move_mask2 = np.zeros(_MAX_MOVES, dtype=np.float32)

        for i, (r, c) in enumerate(moves_shuffled):
            move_rows2[i] = r
            move_cols2[i] = c
            move_mask2[i] = 1.0

        policy2, value2 = session.run(
            None,
            {
                "board": board,
                "move_rows": move_rows2,
                "move_cols": move_cols2,
                "move_mask": move_mask2,
            }
        )

        # Re-associate logits by (row, col)
        logits_by_move_1 = {
            moves_original[i]: policy1[i] for i in range(len(moves_original))
        }
        logits_by_move_2 = {
            moves_shuffled[i]: policy2[i] for i in range(len(moves_shuffled))
        }

        # Compare
        for move in moves_original:
            diff = abs(logits_by_move_1[move] - logits_by_move_2[move])
            assert diff < 1e-5, (
                f"Move {move}: logit differs by {diff} between orderings"
            )

        # Value should also match
        assert abs(float(value1) - float(value2)) < 1e-5, (
            f"Value differs: {value1} vs {value2}"
        )

    print("PASS: Move-order invariance")


def main():
    """Run all tests."""
    print("=" * 60)
    print("ONNX EXPORT TESTS")
    print("=" * 60)
    print()

    tests = [
        test_pytorch_model_forward,
        test_conv_weight_conversion,
        test_weight_transfer,
        test_export_and_load,
        test_parity_simple,
        test_parity_multiple_boards,
        test_masked_logits,
        test_move_order_invariance,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()  # passes if no exception; assertions raise on failure
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("Gate PASSED: ONNX export works, parity verified")
        return 0
    else:
        print("Gate FAILED: ONNX export tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
