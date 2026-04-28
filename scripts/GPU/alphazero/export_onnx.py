#!/usr/bin/env python3
"""Export MLX model to ONNX for Node.js inference.

Strategy: Pad moves to OnnxAlphaZero.max_moves (576 = 24*24, the true maximum
legal moves on a 24x24 board) with -1e9 masking for invalid positions.

Layout conversion:
- MLX uses NHWC: (B, H, W, C)
- PyTorch/ONNX use NCHW: (B, C, H, W)
- Conv weights: MLX stores (out, kH, kW, in), PyTorch expects (out, in, kH, kW)

Usage:
    python -m scripts.GPU.alphazero.export_onnx \\
        --weights checkpoints/alphazero/model_iter_0100.safetensors \\
        --output model.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as tnn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class OnnxResBlock(tnn.Module):
    """Residual block for ONNX export."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = tnn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = tnn.BatchNorm2d(channels)
        self.conv2 = tnn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = tnn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)


# =====================================================================
# Canonicalization constants (mirror scripts/GPU/alphazero/network.py)
# =====================================================================
# Channel layout constants
_CH_TO_MOVE = 18
_ACTIVE_SIZE = 24  # Inference always runs on the full 24x24 board

# Permutation that maps new-channel -> old-channel for the black-to-move
# canonicalized board, matching canonicalize_batch() in network.py:
#   new 0   <- old 1    (black pegs -> current pegs)
#   new 1   <- old 0    (red pegs   -> opponent pegs)
#   new 2-9 <- old 10-17 (black links, link-channel permuted by INV_LINK_PERM_CW=[2,3,4,5,6,7,0,1])
#   new 10-17 <- old 2-9 (red links, same permutation)
#   new 18  <- PLACEHOLDER (will be overwritten with ones)
#   new 19  <- old 21    (black-left  -> current-top)
#   new 20  <- old 22    (black-right -> current-bottom)
#   new 21  <- old 20    (red-bottom  -> opp-left)
#   new 22  <- old 19    (red-top     -> opp-right)
#   new 23  <- old 23    (phase unchanged)
#   new 24  <- old 27    (black_conn_left  -> current_conn_top)
#   new 25  <- old 28    (black_conn_right -> current_conn_bottom)
#   new 26  <- old 29    (black_conn_both  -> current_conn_both)
#   new 27  <- old 25    (red_conn_bottom  -> opp_conn_left)
#   new 28  <- old 24    (red_conn_top     -> opp_conn_right)
#   new 29  <- old 26    (red_conn_both    -> opp_conn_both)
_BLACK_CANON_CHANNEL_PERM = [
    1, 0,
    12, 13, 14, 15, 16, 17, 10, 11,
    4, 5, 6, 7, 8, 9, 2, 3,
    18,
    21, 22, 20, 19,
    23,
    27, 28, 29, 25, 24, 26,
]


class OnnxAlphaZero(tnn.Module):
    """PyTorch model matching MLX architecture for ONNX export.

    Uses pad-to-512 strategy with masking for variable move counts.

    IMPORTANT: PyTorch uses NCHW layout. Node.js feeds RAW (non-canonicalized)
    (1, C, 24, 24) tensors; this module canonicalizes internally inside
    forward() so it matches MLX AlphaZeroNetwork.forward_padded end-to-end.
    """

    def __init__(self, hidden: int = 128, n_blocks: int = 6, max_moves: int = 576,
                 in_channels: int = None):
        super().__init__()
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.max_moves = max_moves
        # Default to current NUM_CHANNELS (30 post-Phase 2); callers can override
        # with in_channels=24 to export 24-channel legacy checkpoints.
        if in_channels is None:
            from .game.twixt_state import NUM_CHANNELS
            in_channels = NUM_CHANNELS
        self.in_channels = in_channels

        # Register channel permutation as a buffer so it ships inside the ONNX
        # graph (rather than being embedded as a traced Python list, which would
        # not survive torch.onnx.export cleanly).
        self.register_buffer(
            "_black_channel_perm",
            torch.tensor(_BLACK_CANON_CHANNEL_PERM, dtype=torch.long),
            persistent=False,
        )

        self.encoder_conv1 = tnn.Conv2d(in_channels, hidden, 3, padding=1)
        self.encoder_bn1 = tnn.BatchNorm2d(hidden)

        self.res_blocks = tnn.ModuleList([
            OnnxResBlock(hidden) for _ in range(n_blocks)
        ])

        # Policy head
        self.policy_conv = tnn.Conv2d(hidden, 2, 1)
        self.policy_bn = tnn.BatchNorm2d(2)
        self.policy_fc = tnn.Linear(2, 64)
        self.policy_out = tnn.Linear(64, 1)

        # Value head (global pooling - layout agnostic)
        # 2*hidden because we concat avg and max pooled features
        self.value_fc1 = tnn.Linear(2 * hidden, 256)
        self.value_fc2 = tnn.Linear(256, 1)

    def _canonicalize_board(self, board: torch.Tensor, is_black: torch.Tensor) -> torch.Tensor:
        """Port of network.canonicalize_batch for NCHW board tensors.

        active_size is hardcoded to 24 (full board) because the ONNX export
        is only used for inference, which always runs on the full 24x24 board.

        Args:
            board: (B, C, H, W) raw input (from JS toTensorHWC reshaped to NCHW)
            is_black: (B, 1, 1, 1) bool selector; True for samples where
                the raw input's CH_TO_MOVE encodes black-to-move.

        Returns:
            (B, C, H, W) canonicalized board:
              - is_black==True  -> board rotated 90° CW, channels swapped to
                current/opponent order, dist/connectivity swapped,
                CH_TO_MOVE forced to 1.
              - is_black==False -> only CH_TO_MOVE forced to 1 (no-op in
                practice for red-to-move inputs); no rotation or channel swap.
        """
        # --- Red-fixed branch: CH_TO_MOVE forced to 1, everything else unchanged ---
        ones_to_move = torch.ones_like(board[:, _CH_TO_MOVE:_CH_TO_MOVE + 1, :, :])
        red_fixed = torch.cat(
            [
                board[:, :_CH_TO_MOVE, :, :],
                ones_to_move,
                board[:, _CH_TO_MOVE + 1:, :, :],
            ],
            dim=1,
        )

        # --- Black-canonical branch ---
        # Step 1: 90° CW spatial rotation.
        # NHWC CW rotation is transpose(H,W) then reverse W.
        # NCHW equivalent: transpose(2,3) then flip(dims=[3]).
        board_rot = board.transpose(2, 3).flip(dims=[3])

        # Step 2: channel re-index via _black_channel_perm, then force ch 18=1.
        black_canon_raw = torch.index_select(board_rot, 1, self._black_channel_perm)
        black_canon_ones = torch.ones_like(
            black_canon_raw[:, _CH_TO_MOVE:_CH_TO_MOVE + 1, :, :]
        )
        black_canon = torch.cat(
            [
                black_canon_raw[:, :_CH_TO_MOVE, :, :],
                black_canon_ones,
                black_canon_raw[:, _CH_TO_MOVE + 1:, :, :],
            ],
            dim=1,
        )

        return torch.where(is_black, black_canon, red_fixed)

    def _canonicalize_moves(
        self,
        is_black: torch.Tensor,
        move_rows: torch.Tensor,
        move_cols: torch.Tensor,
        move_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate move coordinates for black-to-move (matches canonicalize_batch).

        CW rotation on coords: (r, c) -> (c, active_size - 1 - r).
        Applied only where move_mask > 0.5 AND the sample is black-to-move.

        Assumes B=1 (the ONNX export has fixed batch size 1).

        Args:
            is_black: (B, 1, 1, 1) bool — same selector used for the board.
        """
        # Collapse is_black to (B,) = (1,) so it broadcasts over (M,) moves.
        is_black_scalar = is_black.view(-1)

        rows_rot = move_cols
        cols_rot = (_ACTIVE_SIZE - 1) - move_rows

        rotate = (move_mask > 0.5) & is_black_scalar
        move_rows_out = torch.where(rotate, rows_rot, move_rows)
        move_cols_out = torch.where(rotate, cols_rot, move_cols)
        return move_rows_out, move_cols_out

    def forward(
        self,
        board: torch.Tensor,      # (1, C, H, W) NCHW, RAW (uncanonicalized) input
        move_rows: torch.Tensor,  # (512,) padded row indices in RAW coords
        move_cols: torch.Tensor,  # (512,) padded col indices in RAW coords
        move_mask: torch.Tensor,  # (512,) 1.0 for valid, 0.0 for padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for ONNX.

        Canonicalizes the input so the network always sees "current player
        connecting top↔bottom", matching MLX AlphaZeroNetwork.forward_padded.
        Callers (JS server, parity tests) pass raw tensors and raw move coords.

        Returns:
            policy_logits: (512,) with -1e9 for invalid moves (in RAW move order)
            value: scalar in [-1, 1] from the current player's perspective
        """
        # === CANONICALIZE (matches MLX forward_padded) ===
        # Compute is_black from the RAW board before any channel rewrite —
        # _canonicalize_board forces CH_TO_MOVE=1 in both branches, so sampling
        # it after would collapse the selector to a constant.
        is_black = board[:, _CH_TO_MOVE:_CH_TO_MOVE + 1, 0:1, 0:1] < 0.5  # (B,1,1,1)
        board = self._canonicalize_board(board, is_black)
        move_rows, move_cols = self._canonicalize_moves(
            is_black, move_rows, move_cols, move_mask
        )

        # Encode
        x = torch.relu(self.encoder_bn1(self.encoder_conv1(board)))
        for block in self.res_blocks:
            x = block(x)

        # Policy: gather features at (canonicalized) move locations
        policy_feat = torch.relu(self.policy_bn(self.policy_conv(x)))  # (1, 2, 24, 24)

        # policy_feat[0, :, row, col] -> (2,) per move; stack to (2, 512) then transpose.
        gathered = policy_feat[0, :, move_rows, move_cols]  # (2, 512)
        gathered = gathered.T  # (512, 2)

        h = torch.relu(self.policy_fc(gathered))  # (512, 64)
        raw_logits = self.policy_out(h).squeeze(-1)  # (512,)

        masked_logits = torch.where(
            move_mask > 0.5,
            raw_logits,
            torch.full_like(raw_logits, -1e9),
        )

        # Value head (global avg+max pool; active_size=24 makes MLX's masked
        # pooling equivalent to this unmasked pool).
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool = torch.amax(x, dim=(2, 3))
        v = torch.cat([avg_pool, max_pool], dim=-1)

        v = torch.relu(self.value_fc1(v))
        pre = self.value_fc2(v)

        # Soft pretanh clamp (matches ValueHead in network.py):
        #   pre_clamped = 10 * tanh(pre / 10);  value = tanh(pre_clamped)
        PRETANH_CLAMP = 10.0
        pre_clamped = PRETANH_CLAMP * torch.tanh(pre / PRETANH_CLAMP)
        value = torch.tanh(pre_clamped)

        return masked_logits, value.squeeze()


def flatten_mlx_params(params, prefix="") -> Dict[str, np.ndarray]:
    """Flatten nested MLX parameter dict to flat dict with dot-separated keys."""
    result = {}
    if isinstance(params, dict):
        for k, v in params.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            result.update(flatten_mlx_params(v, new_prefix))
    elif isinstance(params, list):
        for i, v in enumerate(params):
            new_prefix = f"{prefix}.{i}"
            result.update(flatten_mlx_params(v, new_prefix))
    else:
        # Assume it's an mx.array or similar - convert to numpy
        result[prefix] = np.array(params)
    return result


def convert_conv_weight(mlx_weight: np.ndarray) -> np.ndarray:
    """Convert MLX Conv2d weight to PyTorch format.

    MLX Conv2d: (out_channels, kH, kW, in_channels) - OIHW with channels last in kernel
    PyTorch Conv2d: (out_channels, in_channels, kH, kW) - OIHW standard
    """
    # MLX shape: (out, kH, kW, in) -> PyTorch shape: (out, in, kH, kW)
    return np.transpose(mlx_weight, (0, 3, 1, 2))


def convert_weights(mlx_params: Dict[str, np.ndarray], pytorch_model: OnnxAlphaZero) -> None:
    """Copy weights from MLX to PyTorch model.

    Handles:
    - Conv weight layout conversion (MLX channels-last kernel to PyTorch)
    - BatchNorm parameters
    - Linear weights (same layout)
    """
    state_dict = pytorch_model.state_dict()

    # Flatten MLX params
    flat_params = mlx_params

    # Direct mappings for encoder
    # encoder.conv1
    if "encoder.conv1.weight" in flat_params:
        state_dict["encoder_conv1.weight"] = torch.from_numpy(
            convert_conv_weight(flat_params["encoder.conv1.weight"])
        )
    if "encoder.conv1.bias" in flat_params:
        state_dict["encoder_conv1.bias"] = torch.from_numpy(flat_params["encoder.conv1.bias"])

    # encoder.bn1
    for suffix in ["weight", "bias", "running_mean", "running_var"]:
        key = f"encoder.bn1.{suffix}"
        if key in flat_params:
            state_dict[f"encoder_bn1.{suffix}"] = torch.from_numpy(flat_params[key])

    # ResBlocks
    for i in range(pytorch_model.n_blocks):
        mlx_prefix = f"encoder.blocks.{i}"
        torch_prefix = f"res_blocks.{i}"

        # conv1, bn1, conv2, bn2
        for j in [1, 2]:
            # Conv
            conv_key = f"{mlx_prefix}.conv{j}.weight"
            if conv_key in flat_params:
                state_dict[f"{torch_prefix}.conv{j}.weight"] = torch.from_numpy(
                    convert_conv_weight(flat_params[conv_key])
                )
            bias_key = f"{mlx_prefix}.conv{j}.bias"
            if bias_key in flat_params:
                state_dict[f"{torch_prefix}.conv{j}.bias"] = torch.from_numpy(flat_params[bias_key])

            # BatchNorm
            for suffix in ["weight", "bias", "running_mean", "running_var"]:
                bn_key = f"{mlx_prefix}.bn{j}.{suffix}"
                if bn_key in flat_params:
                    state_dict[f"{torch_prefix}.bn{j}.{suffix}"] = torch.from_numpy(flat_params[bn_key])

    # Policy head
    if "policy_head.conv.weight" in flat_params:
        state_dict["policy_conv.weight"] = torch.from_numpy(
            convert_conv_weight(flat_params["policy_head.conv.weight"])
        )
    if "policy_head.conv.bias" in flat_params:
        state_dict["policy_conv.bias"] = torch.from_numpy(flat_params["policy_head.conv.bias"])

    for suffix in ["weight", "bias", "running_mean", "running_var"]:
        key = f"policy_head.bn.{suffix}"
        if key in flat_params:
            state_dict[f"policy_bn.{suffix}"] = torch.from_numpy(flat_params[key])

    if "policy_head.fc.weight" in flat_params:
        state_dict["policy_fc.weight"] = torch.from_numpy(flat_params["policy_head.fc.weight"])
    if "policy_head.fc.bias" in flat_params:
        state_dict["policy_fc.bias"] = torch.from_numpy(flat_params["policy_head.fc.bias"])

    if "policy_head.out.weight" in flat_params:
        state_dict["policy_out.weight"] = torch.from_numpy(flat_params["policy_head.out.weight"])
    if "policy_head.out.bias" in flat_params:
        state_dict["policy_out.bias"] = torch.from_numpy(flat_params["policy_head.out.bias"])

    # Value head
    if "value_head.fc1.weight" in flat_params:
        state_dict["value_fc1.weight"] = torch.from_numpy(flat_params["value_head.fc1.weight"])
    if "value_head.fc1.bias" in flat_params:
        state_dict["value_fc1.bias"] = torch.from_numpy(flat_params["value_head.fc1.bias"])

    if "value_head.fc2.weight" in flat_params:
        state_dict["value_fc2.weight"] = torch.from_numpy(flat_params["value_head.fc2.weight"])
    if "value_head.fc2.bias" in flat_params:
        state_dict["value_fc2.bias"] = torch.from_numpy(flat_params["value_head.fc2.bias"])

    pytorch_model.load_state_dict(state_dict)


def export_to_onnx(
    mlx_model,
    output_path: str,
    hidden: int = 128,
    n_blocks: int = 6,
    in_channels: int = None,
) -> None:
    """Export MLX model to ONNX.

    Args:
        mlx_model: Trained MLX AlphaZeroNetwork
        output_path: Path for .onnx file
        hidden: Hidden channels (must match mlx_model)
        n_blocks: Number of residual blocks (must match mlx_model)
        in_channels: Input channel count (defaults to NUM_CHANNELS from twixt_state).
            Override for dual-format exports (e.g. in_channels=24 for pre-Phase-2 weights).
    """
    import mlx.core as mx
    if in_channels is None:
        from .game.twixt_state import NUM_CHANNELS
        in_channels = NUM_CHANNELS

    # Create PyTorch model with matching architecture
    pytorch_model = OnnxAlphaZero(hidden=hidden, n_blocks=n_blocks, in_channels=in_channels)
    pytorch_model.eval()

    # Flatten and convert MLX parameters
    mlx_params = flatten_mlx_params(mlx_model.parameters())

    # Copy weights
    convert_weights(mlx_params, pytorch_model)

    # Create dummy inputs for tracing — shape matches the constructed network.
    # Move-tensor length must follow the model's max_moves (576 = 24*24, the
    # true maximum legal moves on a 24x24 board); the previous 512 cap was
    # smaller than openings can produce, which made callers over-read the
    # output buffer and contaminate priors with NaN.
    max_moves = pytorch_model.max_moves
    board = torch.randn(1, in_channels, 24, 24)  # NCHW
    move_rows = torch.zeros(max_moves, dtype=torch.long)
    move_cols = torch.zeros(max_moves, dtype=torch.long)
    move_mask = torch.zeros(max_moves, dtype=torch.float32)
    move_mask[:10] = 1.0  # Pretend 10 valid moves

    # Export to ONNX
    torch.onnx.export(
        pytorch_model,
        (board, move_rows, move_cols, move_mask),
        output_path,
        input_names=["board", "move_rows", "move_cols", "move_mask"],
        output_names=["policy_logits", "value"],
        dynamic_axes=None,  # Fixed sizes for simplicity
        opset_version=18,  # Use opset 18 for better compatibility
    )

    print(f"Exported ONNX model to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export MLX AlphaZero model to ONNX"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to MLX weights (.safetensors)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="model.onnx",
        help="Output ONNX file path (default: model.onnx)",
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

    args = parser.parse_args()

    # Import MLX modules
    from scripts.GPU.alphazero.network import create_network

    print("=" * 60)
    print("ONNX EXPORT")
    print("=" * 60)
    print()

    # Load MLX model
    print(f"Loading MLX model from {args.weights}...")
    mlx_model = create_network(hidden=args.hidden, n_blocks=args.blocks)
    mlx_model.load_weights(args.weights)

    # Export
    print(f"Exporting to {args.output}...")
    export_to_onnx(mlx_model, args.output, hidden=args.hidden, n_blocks=args.blocks)

    print()
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
