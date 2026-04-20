"""Neural network for GPU-accelerated move scoring.

Architecture:
1. BoardEncoder: Conv stack that runs ONCE per position → feature map F(H,W,D) + global G
2. MoveHead: For each candidate move, gather F[r,c] and score via MLP

Key design: Encode once, gather for all moves. Avoids 500x tensor copies.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING
import math

import numpy as np

from ..utils.maybe_mlx import try_import_mlx
from .tensor_repr import state_to_tensor, moves_to_coords

if TYPE_CHECKING:
    from ..game.state import GameState

# Import MLX
_mlx_env = try_import_mlx()

if _mlx_env.available:
    import mlx.core as mx
    import mlx.nn as nn
else:
    # Stub for type checking when MLX not available
    mx = None
    nn = None


def _check_mlx():
    """Raise if MLX is not available."""
    if not _mlx_env.available:
        raise RuntimeError("MLX is required for move_model but not available")


class BoardEncoder(nn.Module):
    """Encode board state to spatial feature map + global vector.

    Runs ONCE per position. Output is used to score all candidate moves.
    """

    def __init__(
        self,
        in_channels: int = 24,
        hidden_channels: List[int] = None,
        feature_dim: int = 128,
    ):
        """Initialize BoardEncoder.

        Args:
            in_channels: Number of input channels (24 for our representation)
            hidden_channels: Conv layer channel sizes (default [64, 128])
            feature_dim: Output feature dimension per cell
        """
        _check_mlx()
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [64, 128]

        layers = []
        prev_ch = in_channels

        # Build conv stack
        for i, out_ch in enumerate(hidden_channels):
            layers.append(nn.Conv2d(prev_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm(out_ch))
            prev_ch = out_ch

        # Final conv to feature_dim
        layers.append(nn.Conv2d(prev_ch, feature_dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm(feature_dim))

        self.layers = layers
        self.feature_dim = feature_dim

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            x: (H, W, C) single board tensor (channels-last)

        Returns:
            F: (H, W, D) spatial feature map
            G: (2*D,) global feature vector (avg + max pool concatenated)
        """
        # Add batch dim: (H, W, C) -> (1, H, W, C)
        # MLX Conv2d uses channels-last by default, no transpose needed
        x = x[None]

        # Apply conv layers with ReLU
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, nn.BatchNorm):
                x = nn.relu(x)

        # Remove batch dim: (1, H, W, D) -> (H, W, D)
        F = x[0]

        # Global pooling: (H, W, D) -> (D,) for each of avg and max
        avg_pool = mx.mean(F, axis=(0, 1))
        max_pool = mx.max(F, axis=(0, 1))
        G = mx.concatenate([avg_pool, max_pool])  # (2*D,)

        return F, G


class MoveHead(nn.Module):
    """Score moves from gathered local + global features.

    For each candidate move (r, c):
    1. Gather F[r, c, :] for local features
    2. Concatenate with global features, coord embedding, legality
    3. Pass through MLP to get score
    """

    def __init__(
        self,
        local_dim: int = 128,
        global_dim: int = 256,
        coord_embed_dim: int = 32,
        hidden_dim: int = 128,
        board_size: int = 24,
    ):
        """Initialize MoveHead.

        Args:
            local_dim: Dimension of local features from BoardEncoder
            global_dim: Dimension of global features (2 * local_dim typically)
            coord_embed_dim: Dimension for coordinate embeddings
            hidden_dim: Hidden layer dimension
            board_size: Board size for coordinate embedding
        """
        _check_mlx()
        super().__init__()

        # Coordinate embeddings (row and col each get half)
        self.row_embed = nn.Embedding(board_size, coord_embed_dim // 2)
        self.col_embed = nn.Embedding(board_size, coord_embed_dim // 2)

        # Input: local + global + coord_embed + legality
        input_dim = local_dim + global_dim + coord_embed_dim + 1

        # MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

    def __call__(
        self,
        local_features: mx.array,
        global_features: mx.array,
        coords: mx.array,
        legality: mx.array,
    ) -> mx.array:
        """Score multiple moves.

        Args:
            local_features: (N, D) gathered from feature map at move positions
            global_features: (G,) global features (broadcast to all moves)
            coords: (N, 2) row, col for each move (int32)
            legality: (N,) 0/1 legality for each move

        Returns:
            (N,) scores/logits for each move
        """
        N = local_features.shape[0]

        # Coordinate embeddings
        row_emb = self.row_embed(coords[:, 0])  # (N, coord_dim//2)
        col_emb = self.col_embed(coords[:, 1])  # (N, coord_dim//2)
        coord_emb = mx.concatenate([row_emb, col_emb], axis=1)  # (N, coord_dim)

        # Broadcast global features to all moves
        global_broadcast = mx.broadcast_to(
            global_features[None, :], (N, global_features.shape[0])
        )

        # Concatenate all features
        x = mx.concatenate([
            local_features,           # (N, local_dim)
            global_broadcast,         # (N, global_dim)
            coord_emb,                # (N, coord_dim)
            legality[:, None],        # (N, 1)
        ], axis=1)

        # MLP
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze(-1)  # (N,)


class MoveRanker(nn.Module):
    """Full model: encode board once, score all moves via gather.

    Usage:
        model = MoveRanker()
        board_tensor = state_to_tensor(state)
        scores = model.score_all_moves(board_tensor, moves)
    """

    def __init__(
        self,
        in_channels: int = 24,
        feature_dim: int = 128,
        hidden_channels: List[int] = None,
        coord_embed_dim: int = 32,
        mlp_hidden: int = 128,
        board_size: int = 24,
    ):
        """Initialize MoveRanker.

        Args:
            in_channels: Number of input channels
            feature_dim: Feature dimension from encoder
            hidden_channels: Conv layer channel sizes
            coord_embed_dim: Coordinate embedding dimension
            mlp_hidden: MLP hidden dimension
            board_size: Board size
        """
        _check_mlx()
        super().__init__()

        self.encoder = BoardEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            feature_dim=feature_dim,
        )
        self.head = MoveHead(
            local_dim=feature_dim,
            global_dim=2 * feature_dim,  # avg + max pool
            coord_embed_dim=coord_embed_dim,
            hidden_dim=mlp_hidden,
            board_size=board_size,
        )
        self.board_size = board_size

    def score_all_moves(
        self,
        board_tensor: mx.array,
        moves: List[Tuple[int, int]],
    ) -> mx.array:
        """Score all candidate moves in one efficient pass.

        This is the main API. Encodes board once, then scores all moves
        via gather + MLP.

        Args:
            board_tensor: (H, W, 24) from state_to_tensor()
            moves: List of (row, col) candidate moves

        Returns:
            (N,) logits for each move
        """
        if len(moves) == 0:
            return mx.array([], dtype=mx.float32)

        # Step 1: Encode board ONCE
        F, G = self.encoder(board_tensor)  # F: (H,W,D), G: (2D,)

        # Step 2: Convert moves to coords array
        coords = mx.array(moves, dtype=mx.int32)  # (N, 2)

        # Step 3: Vectorized gather for all moves
        local_features = F[coords[:, 0], coords[:, 1], :]  # (N, D)

        # Step 4: Get legality for each move from channel 18
        legal_mask = board_tensor[:, :, 18]  # (H, W)
        legality = legal_mask[coords[:, 0], coords[:, 1]]  # (N,)

        # Step 5: Score via MLP
        logits = self.head(local_features, G, coords, legality)

        return logits

    def score_moves_from_state(
        self,
        state: "GameState",
        moves: Optional[List[Tuple[int, int]]] = None,
    ) -> mx.array:
        """Convenience method: score moves directly from GameState.

        Args:
            state: Current game state
            moves: Candidate moves (if None, uses all legal moves)

        Returns:
            (N,) logits for each move
        """
        from ..game.rules import generate_moves

        if moves is None:
            moves = generate_moves(state)

        board_tensor = state_to_tensor(state)
        return self.score_all_moves(board_tensor, moves)


def create_model(
    feature_dim: int = 128,
    hidden_channels: List[int] = None,
    coord_embed_dim: int = 32,
    mlp_hidden: int = 128,
) -> MoveRanker:
    """Factory function to create a MoveRanker with default settings.

    Args:
        feature_dim: Feature dimension from encoder
        hidden_channels: Conv layer sizes (default [64, 128])
        coord_embed_dim: Coordinate embedding dimension
        mlp_hidden: MLP hidden dimension

    Returns:
        MoveRanker model
    """
    _check_mlx()

    if hidden_channels is None:
        hidden_channels = [64, 128]

    return MoveRanker(
        in_channels=24,
        feature_dim=feature_dim,
        hidden_channels=hidden_channels,
        coord_embed_dim=coord_embed_dim,
        mlp_hidden=mlp_hidden,
        board_size=24,
    )


def save_model(model: MoveRanker, path: str) -> None:
    """Save model parameters to file.

    Args:
        model: MoveRanker to save
        path: File path (.safetensors format)
    """
    _check_mlx()
    model.save_weights(path)


def load_model(path: str, **kwargs) -> MoveRanker:
    """Load model from file.

    Args:
        path: File path to load from (.safetensors format)
        **kwargs: Additional arguments for create_model

    Returns:
        Loaded MoveRanker
    """
    _check_mlx()

    model = create_model(**kwargs)
    model.load_weights(path)
    return model


# For testing without MLX
class NumpyMoveRanker:
    """NumPy-based fallback for testing when MLX is unavailable.

    Uses random weights - for shape/interface testing only.
    """

    def __init__(self, board_size: int = 24, feature_dim: int = 128):
        self.board_size = board_size
        self.feature_dim = feature_dim
        # Random "weights" for testing
        self._rng = np.random.default_rng(42)

    def score_all_moves(
        self,
        board_tensor: np.ndarray,
        moves: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Score moves with random values (for testing)."""
        if len(moves) == 0:
            return np.array([], dtype=np.float32)

        # Generate pseudo-random scores based on position
        scores = []
        for r, c in moves:
            # Use board features to generate deterministic pseudo-score
            score = float(board_tensor[r, c, :].sum())
            score += self._rng.random() * 0.01  # Small noise
            scores.append(score)

        return np.array(scores, dtype=np.float32)
