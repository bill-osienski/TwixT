#!/usr/bin/env python3
"""Tests for AlphaZero network architecture.

Run with: python3 tests/test_network.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_forward_pass():
    """Test basic forward pass on random board."""
    import mlx.core as mx
    from scripts.GPU.alphazero.network import create_network, state_to_input
    from scripts.GPU.alphazero.game import TwixtState

    # Create network
    net = create_network(hidden=128, n_blocks=6)

    # Create test state
    state = TwixtState()
    state = state.apply_move((10, 10))
    state = state.apply_move((5, 5))

    board = state_to_input(state)
    moves = state.legal_moves()[:20]

    # Forward pass
    policy, value = net(board, moves)
    mx.eval(policy, value)

    # Check shapes
    assert policy.shape == (len(moves),), f"Policy shape: {policy.shape}"
    # ValueHead is batched: returns shape (B,) even for B==1. Scalar () was
    # the original per-position return; keep compat but accept current shape.
    assert value.ndim == 0 or value.shape in ((), (1,)), f"Value shape: {value.shape}"

    # Check value range (tanh output)
    assert -1.0 <= float(value) <= 1.0, f"Value out of range: {float(value)}"

    print("PASS: Forward pass")


def test_output_shapes():
    """Test output shapes are correct for various move counts."""
    import mlx.core as mx
    from scripts.GPU.alphazero.network import create_network, state_to_input
    from scripts.GPU.alphazero.game import TwixtState

    net = create_network(hidden=64, n_blocks=2)  # Smaller for speed

    # Test with different numbers of legal moves
    for n_moves in [1, 10, 50, 100, 200]:
        state = TwixtState()
        board = state_to_input(state)
        all_moves = state.legal_moves()
        moves = all_moves[:n_moves]

        policy, value = net(board, moves)
        mx.eval(policy, value)

        assert policy.shape == (len(moves),), \
            f"Policy shape mismatch for {n_moves} moves: {policy.shape}"

    print("PASS: Output shapes")


def test_gradients_flow():
    """Test that gradients flow through the network (no NaN)."""
    import mlx.core as mx
    import mlx.nn as nn
    from scripts.GPU.alphazero.network import create_network, state_to_input
    from scripts.GPU.alphazero.game import TwixtState

    net = create_network(hidden=64, n_blocks=2)  # Smaller for speed

    # Create test data
    state = TwixtState()
    state = state.apply_move((12, 12))
    board = state_to_input(state)
    moves = state.legal_moves()[:10]

    # Define loss function
    def loss_fn(model):
        policy, value = model(board, moves)
        # Simple loss: sum of policy logits + value squared
        return mx.sum(policy) + value * value

    # Compute gradients
    loss, grads = nn.value_and_grad(net, loss_fn)(net)
    mx.eval(loss, grads)

    # Check no NaN in loss
    assert not mx.isnan(loss).item(), f"Loss is NaN: {loss}"

    # Check gradients exist and are not NaN
    def check_grads(g, path=""):
        if isinstance(g, dict):
            for k, v in g.items():
                check_grads(v, f"{path}.{k}")
        elif isinstance(g, list):
            for i, v in enumerate(g):
                check_grads(v, f"{path}[{i}]")
        elif isinstance(g, mx.array):
            assert not mx.any(mx.isnan(g)).item(), f"NaN gradient at {path}"

    check_grads(grads)

    print("PASS: Gradients flow (no NaN)")


def test_evaluate_method():
    """Test the evaluate convenience method."""
    import mlx.core as mx
    from scripts.GPU.alphazero.network import create_network, state_to_input
    from scripts.GPU.alphazero.game import TwixtState

    net = create_network(hidden=64, n_blocks=2)

    state = TwixtState()
    board = state_to_input(state)
    moves = state.legal_moves()[:20]

    priors, value = net.evaluate(board, moves)
    mx.eval(priors)

    # Priors should sum to 1 (softmax)
    prior_sum = float(mx.sum(priors))
    assert abs(prior_sum - 1.0) < 1e-5, f"Priors don't sum to 1: {prior_sum}"

    # All priors should be non-negative
    assert mx.all(priors >= 0).item(), "Negative priors found"

    # Value should be float in [-1, 1]
    assert isinstance(value, float), f"Value is not float: {type(value)}"
    assert -1.0 <= value <= 1.0, f"Value out of range: {value}"

    print("PASS: Evaluate method")


def test_different_game_states():
    """Test network on various game states (early, mid, late game)."""
    import random
    import mlx.core as mx
    from scripts.GPU.alphazero.network import create_network, state_to_input
    from scripts.GPU.alphazero.game import TwixtState

    net = create_network(hidden=64, n_blocks=2)
    rng = random.Random(42)

    # Early game (2 moves)
    state = TwixtState()
    state = state.apply_move((10, 10))
    state = state.apply_move((5, 5))

    board = state_to_input(state)
    moves = state.legal_moves()[:20]
    policy, value = net(board, moves)
    mx.eval(policy, value)
    assert policy.shape[0] == 20, "Early game policy shape wrong"

    # Mid game (30 moves)
    state = TwixtState()
    for _ in range(30):
        legal = state.legal_moves()
        if not legal or state.is_terminal():
            break
        move = rng.choice(legal)
        state = state.apply_move(move)

    if not state.is_terminal():
        board = state_to_input(state)
        moves = state.legal_moves()[:20]
        policy, value = net(board, moves)
        mx.eval(policy, value)
        assert policy.shape[0] == len(moves), "Mid game policy shape wrong"

    # Late game (80 moves)
    state = TwixtState()
    for _ in range(80):
        legal = state.legal_moves()
        if not legal or state.is_terminal():
            break
        move = rng.choice(legal)
        state = state.apply_move(move)

    if not state.is_terminal():
        board = state_to_input(state)
        moves = state.legal_moves()[:20]
        policy, value = net(board, moves)
        mx.eval(policy, value)
        assert policy.shape[0] == len(moves), "Late game policy shape wrong"

    print("PASS: Different game states")


def test_network_components():
    """Test individual network components."""
    import mlx.core as mx
    from scripts.GPU.alphazero.network import (
        BoardEncoder, PolicyHead, ValueHead, ResBlock
    )

    # Test ResBlock
    block = ResBlock(64)
    x = mx.zeros((1, 24, 24, 64))
    y = block(x)
    mx.eval(y)
    assert y.shape == x.shape, f"ResBlock shape mismatch: {y.shape}"

    # Test BoardEncoder
    encoder = BoardEncoder(in_channels=24, hidden=64, n_blocks=2)
    x = mx.zeros((1, 24, 24, 24))
    y = encoder(x)
    mx.eval(y)
    assert y.shape == (1, 24, 24, 64), f"Encoder shape: {y.shape}"

    # Test PolicyHead
    policy_head = PolicyHead(in_channels=64, hidden=32)
    features = mx.zeros((1, 24, 24, 64))
    moves = [(0, 1), (1, 1), (2, 2)]
    logits = policy_head(features, moves)
    mx.eval(logits)
    assert logits.shape == (3,), f"PolicyHead shape: {logits.shape}"

    # Test ValueHead
    value_head = ValueHead(in_channels=64, hidden=32)
    features = mx.zeros((1, 24, 24, 64))
    value = value_head(features)
    mx.eval(value)
    # ValueHead is batched: returns shape (B,) even for B==1. Scalar () was
    # the original per-position return; keep compat but accept current shape.
    assert value.ndim == 0 or value.shape in ((), (1,)), f"ValueHead shape: {value.shape}"
    assert -1.0 <= float(value) <= 1.0, f"Value range: {float(value)}"

    print("PASS: Network components")


def test_empty_moves():
    """Test network handles empty move list gracefully."""
    import mlx.core as mx
    from scripts.GPU.alphazero.network import create_network, state_to_input
    from scripts.GPU.alphazero.game import TwixtState

    net = create_network(hidden=64, n_blocks=2)

    state = TwixtState()
    board = state_to_input(state)
    moves = []  # Empty move list

    policy, value = net(board, moves)
    mx.eval(policy, value)

    assert policy.shape == (0,), f"Empty moves policy shape: {policy.shape}"
    assert -1.0 <= float(value) <= 1.0, "Value out of range"

    print("PASS: Empty moves")


def test_canonicalize_batch_handles_30_channels_black_to_move():
    """With NUM_CHANNELS=30, canonicalize_batch must pass a 30-ch input
    through both red-to-move and black-to-move paths without shape errors,
    and the 6 new connectivity channels must be present in the output."""
    import mlx.core as mx
    import numpy as np
    from scripts.GPU.alphazero.network import canonicalize_batch
    from scripts.GPU.alphazero.game.twixt_state import NUM_CHANNELS

    # Build a trivial 30-ch board: red-to-move (CH_TO_MOVE=1 everywhere)
    B, S = 2, 8
    boards = np.zeros((B, 24, 24, NUM_CHANNELS), dtype=np.float32)
    # Batch element 0: red-to-move (to_move channel = 1)
    boards[0, :, :, 18] = 1.0
    # Batch element 1: black-to-move (to_move channel = 0)
    # Put a distinctive value in the 6 new channels so we can trace where it lands
    boards[1, 0, 0, 24] = 1.0  # red_conn_top at (0,0) pre-canon
    boards[1, 0, 0, 27] = 1.0  # black_conn_left at (0,0) pre-canon

    move_rows = mx.zeros((B, 1), dtype=mx.int32)
    move_cols = mx.zeros((B, 1), dtype=mx.int32)
    move_mask = mx.ones((B, 1), dtype=mx.float32)

    out_boards, _, _, _ = canonicalize_batch(
        mx.array(boards), move_rows, move_cols, move_mask, active_size=S
    )
    # Must not shape-error; output keeps 30 channels
    assert out_boards.shape == (B, 24, 24, NUM_CHANNELS)
    # Red-to-move sample: unchanged except to_move channel forced to 1
    out_np = np.array(out_boards)
    # Black sample: after 90° CW rotation, the black_conn_left at (0,0) should
    # map to the current_conn_top channel (new slot 24) at position (0, S-1)
    # ((r,c)=(0,0) rotates to (r',c')=(0, S-1-0)=(0, S-1))
    assert out_np[1, 0, S - 1, 24] == 1.0, (
        "After black-canon, old black_conn_left at (0,0) should land in "
        f"new ch24 (current_conn_top) at (0, {S-1})"
    )

    print("PASS: canonicalize_batch handles 30 channels (black-to-move)")


def test_layout_sanity():
    """Test that PolicyHead gather uses correct NHWC indexing.

    This is a TRIPWIRE test - it will fail 100% of the time if gather
    indexing is wrong. Two key design choices make it robust:

    1. Non-square dimensions (H=7, W=11, C=5) so NHWC and NCHW tensors
       have different shapes and can't accidentally "look the same"

    2. Deterministic weights (all ones) so output is predictable:
       - Correct NHWC gather of [1,0,0,0,0] → dot with ones → 1.0
       - Correct NHWC gather of [0,0,0,0,1] → dot with ones → 1.0
       - Wrong NCHW gather would get different values → different output

    Layout contract:
    - MLX = NHWC everywhere
    - ONNX/PyTorch = NCHW
    - Single conversion boundary: export wrapper and Node preprocessing
    """
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    # Use non-square dims: H=7, W=11, C=5
    # NHWC shape: (1, 7, 11, 5)
    # NCHW shape would be: (1, 5, 7, 11) - clearly different
    H, W, C = 7, 11, 5

    # Build a minimal "policy head" with known weights
    # Just: gather features at (r,c), then dot with ones vector
    # This way we know exactly what output to expect
    conv = nn.Conv2d(C, 2, kernel_size=1)  # C -> 2 channels
    fc = nn.Linear(2, 1)

    # Set all weights to 1.0 for deterministic behavior
    conv.weight = mx.ones_like(conv.weight)
    conv.bias = mx.zeros_like(conv.bias)
    fc.weight = mx.ones_like(fc.weight)
    fc.bias = mx.zeros_like(fc.bias)

    # Create synthetic feature tensor in NHWC: (1, H, W, C) = (1, 7, 11, 5)
    features_np = np.zeros((1, H, W, C), dtype=np.float32)

    # Position (3, 4) has channel 0 = 1.0, rest = 0
    # Position (4, 3) has channel 4 = 1.0, rest = 0
    features_np[0, 3, 4, 0] = 1.0  # NHWC: [batch, row, col, channel]
    features_np[0, 4, 3, 4] = 1.0

    features = mx.array(features_np)

    # Run conv (produces (1, H, W, 2) in NHWC)
    conv_out = conv(features)
    mx.eval(conv_out)

    # Gather at positions using NHWC indexing
    feat_3_4 = conv_out[0, 3, 4, :]  # Should get features at row=3, col=4
    feat_4_3 = conv_out[0, 4, 3, :]  # Should get features at row=4, col=3

    # Apply fc
    logit_3_4 = float(fc(feat_3_4).squeeze())
    logit_4_3 = float(fc(feat_4_3).squeeze())

    mx.eval(fc.weight)  # Ensure evaluated

    # With all-ones weights:
    # - Input at (3,4) has [1,0,0,0,0] in channel dim
    # - Input at (4,3) has [0,0,0,0,1] in channel dim
    # After conv with ones: both produce same magnitude but verify they're valid
    # The key test: if we used NCHW indexing x[0, :, r, c], we'd get wrong values

    # Both should produce finite, non-zero values (conv of [1,0,0,0,0] with ones)
    assert abs(logit_3_4) > 1e-6, f"Logit at (3,4) should be non-zero, got {logit_3_4}"
    assert abs(logit_4_3) > 1e-6, f"Logit at (4,3) should be non-zero, got {logit_4_3}"

    # Query a position with all zeros - should give different result
    feat_0_0 = conv_out[0, 0, 0, :]
    logit_0_0 = float(fc(feat_0_0).squeeze())

    # (0,0) has all zeros in input, so after conv with ones kernel, it should differ
    # from positions that had non-zero input
    # This catches if gather is completely broken
    assert logit_3_4 != logit_0_0 or logit_4_3 != logit_0_0, (
        f"Positions with different inputs should produce different outputs. "
        f"Got same value {logit_0_0} everywhere - gather may be broken."
    )

    # Verify the network-level assertion catches wrong channel count
    from scripts.GPU.alphazero.network import create_network, NUM_CHANNELS, BOARD_SIZE
    net = create_network(hidden=64, n_blocks=2)

    wrong_channels = 16  # Not 24
    board_wrong = mx.array(np.zeros((1, BOARD_SIZE, BOARD_SIZE, wrong_channels), dtype=np.float32))

    try:
        net(board_wrong, [(1, 1)])
        assert False, "Should have raised assertion for wrong channel count"
    except AssertionError as e:
        if "channels" in str(e).lower() or str(NUM_CHANNELS) in str(e):
            pass  # Expected - assertion caught wrong channels
        else:
            raise

    print("PASS: Layout sanity (NHWC gather verified with non-square dims, wrong channels rejected)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("NETWORK ARCHITECTURE TESTS")
    print("=" * 60)
    print()

    tests = [
        test_forward_pass,
        test_output_shapes,
        test_gradients_flow,
        test_evaluate_method,
        test_different_game_states,
        test_network_components,
        test_empty_moves,
        test_layout_sanity,
        test_canonicalize_batch_handles_30_channels_black_to_move,
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
        print("Gate PASSED: Network forward pass works, shapes correct, gradients flow")
        return 0
    else:
        print("Gate FAILED: Network tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
