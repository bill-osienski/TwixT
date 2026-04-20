"""Evaluator protocol for MCTS leaf evaluation.

This abstraction decouples MCTS from the specific neural network implementation,
enabling:
1. Worker processes to import MCTS without GPU/MLX access
2. Different evaluator backends (local GPU, remote server, etc.)
3. Easier testing with mock evaluators
"""
from typing import Protocol, Tuple

import numpy as np


class Evaluator(Protocol):
    """Interface for MCTS leaf evaluation.

    Implementations must provide an `infer` method that takes batched
    board states and move information, returning policy priors and values.
    """

    def infer(
        self,
        boards: np.ndarray,      # (B, H, W, C) float32
        move_rows: np.ndarray,   # (B, M) int32
        move_cols: np.ndarray,   # (B, M) int32
        move_mask: np.ndarray,   # (B, M) float32
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
            priors: (B, M) float32 - probability distribution over moves
            values: (B,) float32 - value estimates for each position
        """
        ...
