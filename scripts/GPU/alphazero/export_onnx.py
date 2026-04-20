#!/usr/bin/env python3
"""Export MLX model to ONNX for Node.js inference.

Strategy: Pad moves to 512 with -1e9 masking for invalid positions.

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


class OnnxAlphaZero(tnn.Module):
    """PyTorch model matching MLX architecture for ONNX export.

    Uses pad-to-512 strategy with masking for variable move counts.
    24-channel input to match training encoding.

    IMPORTANT: PyTorch uses NCHW layout. Node.js must feed (1, 24, 24, 24)
    where dim order is (batch, channels, height, width).
    """

    def __init__(self, hidden: int = 128, n_blocks: int = 6, max_moves: int = 512,
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

    def forward(
        self,
        board: torch.Tensor,      # (1, C, H, W) = (1, 24, 24, 24) NCHW format!
        move_rows: torch.Tensor,  # (512,) padded row indices
        move_cols: torch.Tensor,  # (512,) padded col indices
        move_mask: torch.Tensor,  # (512,) 1.0 for valid, 0.0 for padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for ONNX.

        Returns:
            policy_logits: (512,) with -1e9 for invalid moves
            value: scalar in [-1, 1]
        """
        # Encode - expects NCHW input
        x = torch.relu(self.encoder_bn1(self.encoder_conv1(board)))
        for block in self.res_blocks:
            x = block(x)

        # Policy: gather features at move locations
        policy_feat = torch.relu(self.policy_bn(self.policy_conv(x)))  # (1, 2, 24, 24)

        # Vectorized gather for better ONNX performance
        # policy_feat is (1, 2, H, W) in NCHW format
        # Gather at all 512 positions, then apply FC
        # Index as policy_feat[0, :, row, col] to get (2,) per move
        gathered = policy_feat[0, :, move_rows, move_cols]  # (2, 512)
        gathered = gathered.T  # (512, 2)

        # Batch FC through all moves
        h = torch.relu(self.policy_fc(gathered))  # (512, 64)
        raw_logits = self.policy_out(h).squeeze(-1)  # (512,)

        # Apply mask: -1e9 for invalid positions
        masked_logits = torch.where(
            move_mask > 0.5,
            raw_logits,
            torch.full_like(raw_logits, -1e9)
        )

        # Value head (global pooling - layout agnostic)
        # x is (1, C, H, W) from encoder
        # Pool over spatial dims (2, 3) in NCHW format
        avg_pool = torch.mean(x, dim=(2, 3))  # (B, C)
        max_pool = torch.amax(x, dim=(2, 3))  # (B, C)
        v = torch.cat([avg_pool, max_pool], dim=-1)  # (B, 2*C)

        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

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

    # Create dummy inputs for tracing — shape matches the constructed network
    board = torch.randn(1, in_channels, 24, 24)  # NCHW
    move_rows = torch.zeros(512, dtype=torch.long)
    move_cols = torch.zeros(512, dtype=torch.long)
    move_mask = torch.zeros(512, dtype=torch.float32)
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
