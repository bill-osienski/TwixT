"""Batched evaluation for GPU-accelerated move scoring.

Uses MLX for batch matrix operations on Apple Silicon.
Falls back to NumPy/CPU if MLX unavailable.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..game.state import GameState
from ..utils.maybe_mlx import try_import_mlx
from .heuristics import extract_features
from .value_model import ValueModel

# Try to import MLX
_mlx_env = try_import_mlx()

# Try to import NumPy as fallback
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False


def batch_extract_features(
    states: List[GameState],
    player: str,
    base_turn: int,
    friendly_peg_count: int,
    opponent_peg_count: int,
) -> List[Dict[str, float]]:
    """Extract features for multiple states in batch.

    Args:
        states: List of game states (typically child states after moves)
        player: Player perspective for feature extraction
        base_turn: Turn number before moves were applied
        friendly_peg_count: Friendly peg count before moves
        opponent_peg_count: Opponent peg count before moves

    Returns:
        List of feature dicts, one per state
    """
    results = []
    for i, state in enumerate(states):
        features = extract_features(state, player)
        # Add context features that JS uses
        features["turn"] = base_turn + 1
        features["player"] = 1.0 if player == "red" else 0.0
        features["playerPegCount"] = friendly_peg_count + 1
        features["opponentPegCount"] = opponent_peg_count
        results.append(features)
    return results


def batch_extract_features_cached(
    parent_state: GameState,
    child_states: List[GameState],
    player: str,
    base_turn: int,
    friendly_peg_count: int,
    opponent_peg_count: int,
    *,
    parent_opponent_cache: Optional[Dict[str, float]] = None,
) -> List[Dict[str, float]]:
    """Extract features for child states with cached opponent features.

    Key optimizations:
    1. Opponent features computed ONCE from parent (invariant under our move)
    2. Valid cells precomputed once, used for O(1) lookups
    3. Uses _fast versions of expensive functions

    For typical mid-game with ~500 candidate moves and top-k=50:
    - Before: 50 * 114µs = 5.7ms feature extraction
    - After:  50 * 45µs + 49µs = 2.3ms
    - Speedup: ~2.5x

    Args:
        parent_state: State before any move was applied
        child_states: States after each candidate move
        player: Player who just moved (feature extraction perspective)
        base_turn: Turn number before moves were applied
        friendly_peg_count: Friendly peg count before moves
        opponent_peg_count: Opponent peg count before moves
        parent_opponent_cache: Pre-computed opponent features from parent
                               (if None, computed once internally)

    Returns:
        List of feature dicts, one per child state
    """
    from .heuristics import (
        evaluate_connected_paths,
        evaluate_potential_connections,
        evaluate_potential_connections_fast,
        evaluate_edge_progress,
        component_metrics,
        compute_frontier,
        compute_frontier_fast,
        precompute_valid_cells,
        DEFAULT_KNOBS,
    )

    opponent = "red" if player == "black" else "black"
    k = DEFAULT_KNOBS

    # Precompute valid cells for player ONCE (used in fast functions)
    # Note: After a move, only one cell changes (the new peg)
    # So we compute from parent and will remove the new peg position per child
    parent_valid_player = precompute_valid_cells(parent_state, player)

    # Compute opponent features ONCE from parent state (invariant under our move)
    if parent_opponent_cache is not None:
        opp_cache = parent_opponent_cache
    else:
        opp_metrics = component_metrics(parent_state, opponent)
        # For opponent, use parent valid cells (doesn't change when player moves)
        parent_valid_opponent = precompute_valid_cells(parent_state, opponent)
        opp_frontier = compute_frontier_fast(parent_state, opponent, opp_metrics, parent_valid_opponent)
        opp_cache = {
            "opponent_connected_paths": evaluate_connected_paths(parent_state, opponent, k),
            "opponent_potential": evaluate_potential_connections_fast(parent_state, opponent, parent_valid_opponent),
            "opponent_edge_progress": evaluate_edge_progress(parent_state, opponent, k),
            "opponent_pegs": sum(1 for p in parent_state.pegs.values() if p == opponent),
            "opponent_max_row_span": opp_metrics["max_row_span"],
            "opponent_max_col_span": opp_metrics["max_col_span"],
            "opponent_component_count": len(opp_metrics["components"]),
            "opponent_largest_size": len(opp_metrics["largest_component"]),
            "opponent_touches_top": 1.0 if opp_metrics["touches_top"] else 0.0,
            "opponent_touches_bottom": 1.0 if opp_metrics["touches_bottom"] else 0.0,
            "opponent_touches_left": 1.0 if opp_metrics["touches_left"] else 0.0,
            "opponent_touches_right": 1.0 if opp_metrics["touches_right"] else 0.0,
        }

    results = []
    for child in child_states:
        # Compute player-specific features (these change per move)
        friendly_metrics = component_metrics(child, player)

        # Get valid cells for this child (parent minus the new peg)
        # The new peg is the last move in move_history
        if child.move_history:
            _, new_r, new_c = child.move_history[-1]
            child_valid = parent_valid_player - {(new_r, new_c)}
        else:
            child_valid = parent_valid_player

        # Use fast versions with precomputed valid cells
        friendly_frontier = compute_frontier_fast(child, player, friendly_metrics, child_valid)

        features: Dict[str, float] = {}

        # Player features (computed per child with fast functions)
        features["friendly_connected_paths"] = evaluate_connected_paths(child, player, k)
        features["friendly_potential"] = evaluate_potential_connections_fast(child, player, child_valid)
        features["friendly_edge_progress"] = evaluate_edge_progress(child, player, k)
        features["friendly_pegs"] = sum(1 for p in child.pegs.values() if p == player)

        features["friendly_max_row_span"] = friendly_metrics["max_row_span"]
        features["friendly_max_col_span"] = friendly_metrics["max_col_span"]
        features["friendly_component_count"] = len(friendly_metrics["components"])
        features["friendly_largest_size"] = len(friendly_metrics["largest_component"])

        features["friendly_touches_top"] = 1.0 if friendly_metrics["touches_top"] else 0.0
        features["friendly_touches_bottom"] = 1.0 if friendly_metrics["touches_bottom"] else 0.0
        features["friendly_touches_left"] = 1.0 if friendly_metrics["touches_left"] else 0.0
        features["friendly_touches_right"] = 1.0 if friendly_metrics["touches_right"] else 0.0

        features["frontier_size"] = len(friendly_frontier["frontier"])
        features["connector_count"] = len(friendly_frontier["connectors"])
        features["trailing_count"] = len(friendly_frontier["trailing"])

        # Opponent features (cached, invariant)
        features.update(opp_cache)

        # Shared features
        features["move_count"] = len(child.move_history)
        features["total_bridges"] = len(child.bridges)

        # Context features for value model
        features["turn"] = base_turn + 1
        features["player"] = 1.0 if player == "red" else 0.0
        features["playerPegCount"] = friendly_peg_count + 1
        features["opponentPegCount"] = opponent_peg_count

        results.append(features)

    return results


class BatchValueModel:
    """GPU-accelerated batch value model inference.

    Wraps a ValueModel and provides batch evaluation using MLX or NumPy.
    """

    def __init__(self, model: ValueModel):
        self.model = model
        self._weights_gpu = None
        self._weights_np = None
        self._mean_gpu = None
        self._std_gpu = None
        self._mean_np = None
        self._std_np = None
        self._setup_done = False

    def _setup_gpu(self) -> bool:
        """Setup GPU tensors for batch inference."""
        if self._setup_done:
            return self._weights_gpu is not None

        self._setup_done = True

        if not _mlx_env.available:
            return False

        mx = _mlx_env.mx
        try:
            # Convert weights to MLX array: [bias, w1, w2, ...]
            # We'll separate bias and weights for matmul
            self._bias_gpu = mx.array([self.model.weights[0]], dtype=mx.float32)
            self._weights_gpu = mx.array(self.model.weights[1:], dtype=mx.float32)

            # Preprocessing params
            if self.model.standardize and self.model.mean and self.model.std:
                self._mean_gpu = mx.array(self.model.mean, dtype=mx.float32)
                self._std_gpu = mx.array(self.model.std, dtype=mx.float32)
                # Replace zeros in std to avoid division by zero
                self._std_gpu = mx.where(
                    self._std_gpu == 0,
                    mx.ones_like(self._std_gpu),
                    self._std_gpu
                )
            return True
        except Exception:
            self._weights_gpu = None
            return False

    def _setup_numpy(self) -> bool:
        """Setup NumPy arrays for batch inference."""
        if not _HAS_NUMPY:
            return False

        if self._weights_np is not None:
            return True

        try:
            self._bias_np = np.array([self.model.weights[0]], dtype=np.float32)
            self._weights_np = np.array(self.model.weights[1:], dtype=np.float32)

            if self.model.standardize and self.model.mean and self.model.std:
                self._mean_np = np.array(self.model.mean, dtype=np.float32)
                self._std_np = np.array(self.model.std, dtype=np.float32)
                self._std_np = np.where(self._std_np == 0, 1.0, self._std_np)
            return True
        except Exception:
            self._weights_np = None
            return False

    def _build_feature_matrix(
        self, feature_dicts: List[Dict[str, float]]
    ) -> List[List[float]]:
        """Build NxD feature matrix from feature dicts."""
        n = len(feature_dicts)
        d = len(self.model.feature_keys)
        matrix = []

        for features in feature_dicts:
            row = []
            for key in self.model.feature_keys:
                val = features.get(key, 0.0)
                if val is None or (isinstance(val, float) and val != val):  # NaN check
                    val = 0.0
                row.append(float(val))
            matrix.append(row)

        return matrix

    def batch_evaluate_gpu(
        self, feature_dicts: List[Dict[str, float]]
    ) -> List[Dict[str, Optional[float]]]:
        """Batch evaluate using MLX GPU acceleration.

        Args:
            feature_dicts: List of feature dicts

        Returns:
            List of evaluation results with probability, logit, adjustment
        """
        if not self._setup_gpu():
            return self.batch_evaluate_cpu(feature_dicts)

        mx = _mlx_env.mx
        n = len(feature_dicts)

        if n == 0:
            return []

        # Build feature matrix
        X = mx.array(self._build_feature_matrix(feature_dicts), dtype=mx.float32)

        # Apply standardization if needed
        if self._mean_gpu is not None and self._std_gpu is not None:
            X = (X - self._mean_gpu) / self._std_gpu

        # Compute logits: z = X @ W + bias (batch matmul)
        z = mx.matmul(X, self._weights_gpu) + self._bias_gpu

        # Sigmoid with numerical stability
        z_clipped = mx.clip(z, -35.0, 35.0)
        p = 1.0 / (1.0 + mx.exp(-z_clipped))

        # Compute adjustments
        adjustments = (p - 0.5) * self.model.scale

        # Convert to Python lists
        z_list = z.tolist()
        p_list = p.tolist()
        adj_list = adjustments.tolist()

        results = []
        for i in range(n):
            results.append({
                "probability": p_list[i] if isinstance(p_list[i], float) else p_list[i][0],
                "logit": z_list[i] if isinstance(z_list[i], float) else z_list[i][0],
                "adjustment": adj_list[i] if isinstance(adj_list[i], float) else adj_list[i][0],
            })

        return results

    def batch_evaluate_cpu(
        self, feature_dicts: List[Dict[str, float]]
    ) -> List[Dict[str, Optional[float]]]:
        """Batch evaluate using NumPy (CPU fallback).

        Args:
            feature_dicts: List of feature dicts

        Returns:
            List of evaluation results
        """
        if self._setup_numpy() and _HAS_NUMPY:
            return self._batch_evaluate_numpy(feature_dicts)

        # Ultimate fallback: sequential evaluation
        return [self.model.evaluate(f) for f in feature_dicts]

    def _batch_evaluate_numpy(
        self, feature_dicts: List[Dict[str, float]]
    ) -> List[Dict[str, Optional[float]]]:
        """NumPy-accelerated batch evaluation."""
        n = len(feature_dicts)
        if n == 0:
            return []

        X = np.array(self._build_feature_matrix(feature_dicts), dtype=np.float32)

        # Apply standardization if needed
        if self._mean_np is not None and self._std_np is not None:
            X = (X - self._mean_np) / self._std_np

        # Compute logits: z = X @ W + bias
        z = X @ self._weights_np + self._bias_np

        # Sigmoid with numerical stability
        z_clipped = np.clip(z, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z_clipped))

        # Compute adjustments
        adjustments = (p - 0.5) * self.model.scale

        results = []
        for i in range(n):
            results.append({
                "probability": float(p[i]),
                "logit": float(z[i]),
                "adjustment": float(adjustments[i]),
            })

        return results

    def batch_evaluate(
        self, feature_dicts: List[Dict[str, float]]
    ) -> List[Dict[str, Optional[float]]]:
        """Batch evaluate with automatic GPU/CPU selection.

        Tries GPU first, falls back to CPU.
        """
        if _mlx_env.available:
            return self.batch_evaluate_gpu(feature_dicts)
        return self.batch_evaluate_cpu(feature_dicts)


# Cached batch model
_cached_batch_model: Optional[BatchValueModel] = None


def get_batch_value_model(model: Optional[ValueModel] = None) -> Optional[BatchValueModel]:
    """Get or create a BatchValueModel.

    Args:
        model: ValueModel to wrap (uses cached model if None)

    Returns:
        BatchValueModel if model available, None otherwise
    """
    global _cached_batch_model

    if model is None:
        from .value_model import get_cached_model, try_load_value_model
        model = get_cached_model()
        # Auto-load value model if not cached
        if model is None:
            model = try_load_value_model()

    if model is None:
        return None

    if _cached_batch_model is not None and _cached_batch_model.model is model:
        return _cached_batch_model

    _cached_batch_model = BatchValueModel(model)
    return _cached_batch_model


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return _mlx_env.available
