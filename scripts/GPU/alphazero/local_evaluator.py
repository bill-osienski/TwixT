"""LocalGPUEvaluator: MLX-based evaluator for single-process training.

This evaluator runs neural network inference locally on the GPU using MLX.
It implements the Evaluator protocol, taking numpy arrays and returning numpy arrays.

Cache clearing: Handled by trainer.py per-game (mx.clear_cache() after each game).
Do NOT add periodic clearing here - it becomes a perf tax with hot inference.
"""
from typing import Tuple

import numpy as np
import mlx.core as mx

from .network import AlphaZeroNetwork


class LocalGPUEvaluator:
    """Local GPU evaluator using MLX.

    Wraps an AlphaZeroNetwork and handles:
    - numpy <-> MLX array conversion
    - Batched forward pass
    - Stable softmax with mask and renormalization
    - GPU synchronization
    """

    def __init__(self, network: AlphaZeroNetwork):
        """Initialize evaluator.

        Args:
            network: AlphaZeroNetwork instance for inference
        """
        self.network = network

    def build_input_tensor(self, state) -> np.ndarray:
        """Build the (C, H, W) input tensor in the format matching the network.

        For 30-channel networks: delegates to state.to_tensor().
        For 24-channel networks (legacy iter-0999 format): uses to_tensor_v1
        from twixt_state to produce the pre-Phase-2 layout.
        """
        from .game.twixt_state import to_tensor_v1
        in_channels = getattr(self.network, "in_channels", None)
        if in_channels == 24:
            return to_tensor_v1(state)
        # Default: current (30-channel) format
        return state.to_tensor()

    def infer(
        self,
        boards: np.ndarray,
        move_rows: np.ndarray,
        move_cols: np.ndarray,
        move_mask: np.ndarray,
        active_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate batch of positions.

        Args:
            boards: Board states as (B, H, W, C) float32 array
            move_rows: Row indices of legal moves, (B, M) int32
            move_cols: Column indices of legal moves, (B, M) int32
            move_mask: Mask for valid moves, (B, M) float32 (1.0 = valid, 0.0 = padding)
            active_size: Current curriculum board size

        Returns:
            priors: (B, M) float32 - masked and renormalized probability distribution
            values: (B,) float32 - value estimates for each position
        """
        # Convert numpy to MLX arrays
        boards_mx = mx.array(boards)
        move_rows_mx = mx.array(move_rows)
        move_cols_mx = mx.array(move_cols)
        move_mask_mx = mx.array(move_mask)

        # Forward pass
        policy_logits, values_mx, _ = self.network.forward_padded(
            boards_mx, move_rows_mx, move_cols_mx, move_mask_mx, active_size
        )

        # Sync GPU (known stable at eval_batch <= 14)
        mx.eval(policy_logits, values_mx)

        # Stable softmax with mask and renormalization
        priors_mx = self._masked_softmax_batch(policy_logits, move_mask_mx)
        mx.eval(priors_mx)

        # Convert to numpy (bulk transfer)
        priors_list = priors_mx.tolist()
        values_list = values_mx.tolist()

        # Handle MLX returning scalar for B=1
        if not isinstance(values_list, list):
            values_list = [values_list]

        # Handle priors being 1D for B=1
        B = boards.shape[0]
        if B == 1 and priors_list and not isinstance(priors_list[0], list):
            priors_list = [priors_list]

        priors_np = np.array(priors_list, dtype=np.float32)
        values_np = np.array(values_list, dtype=np.float32)

        # Release MLX arrays
        del boards_mx, move_rows_mx, move_cols_mx, move_mask_mx
        del policy_logits, values_mx, priors_mx

        # Note: No cache clearing here - trainer.py handles it per-game

        return priors_np, values_np

    def _masked_softmax_batch(self, logits: mx.array, mask: mx.array) -> mx.array:
        """Masked softmax: apply mask and renormalize to ensure valid distribution.

        This ensures padded moves get exactly 0 probability and valid moves
        sum to 1.0, preventing probability mass from leaking to padding.

        Args:
            logits: (B, M) with NEG_INF for padded positions
            mask: (B, M) with 1.0 for valid moves, 0.0 for padding

        Returns:
            (B, M) probabilities (padded = 0, valid sum to 1)
        """
        # Subtract max per row for numerical stability
        max_logits = mx.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits

        # Replace non-finite values (from -inf - -inf = nan) with large negative
        shifted = mx.where(mx.isfinite(shifted), shifted, -1e9)

        exp_shifted = mx.exp(shifted)

        # Apply mask to zero out padding
        masked_exp = exp_shifted * mask

        # Renormalize over valid moves only
        denom = mx.sum(masked_exp, axis=1, keepdims=True)
        denom = mx.maximum(denom, 1e-9)  # Avoid divide-by-zero

        return masked_exp / denom
