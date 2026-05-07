"""Conversion auxiliary loss helpers (Spec 2).

Pure functions — no MLX state, no I/O. Predicates and target builders for
the policy-side closeout correction.
"""
from __future__ import annotations
from typing import Optional
import numpy as np


def is_conversion_eligible(
    gc_state_full: Optional[dict],
    *,
    max_total_goal_distance: int,
    min_component_size: int,
) -> bool:
    """Determines whether a pre-move state qualifies the side-to-move's
    PositionRecord for conversion auxiliary loss.

    Pure dict math — no BFS. Defends against missing/None fields.
    """
    if gc_state_full is None:
        return False
    total = gc_state_full.get("total_goal_distance")
    comp_size = gc_state_full.get("largest_component_size")
    if total is None or comp_size is None:
        return False
    if total > max_total_goal_distance:
        return False
    if comp_size < min_component_size:
        return False
    completion = gc_state_full.get("endpoint_completion_moves") or []
    reducing = gc_state_full.get("distance_reducing_moves") or []
    if not completion and not reducing:
        return False
    return True


def build_conversion_target(
    legal_moves: list,
    completion_moves: set,
    reducing_moves: set,
    *,
    completion_weight: float,
    reducer_weight: float,
) -> Optional[np.ndarray]:
    """Build a normalized auxiliary target distribution over legal_moves.

    Returns a length-len(legal_moves) np.float32 array summing to 1.0,
    or None if the target is empty after legal-move alignment.

    Disjoint-mass rule: a move that is both endpoint-completing AND
    distance-reducing receives completion_weight (the larger), not the sum.
    """
    weights = np.zeros(len(legal_moves), dtype=np.float32)
    for i, m in enumerate(legal_moves):
        if m in completion_moves:
            weights[i] = completion_weight
        elif m in reducing_moves:
            weights[i] = reducer_weight
    total = float(weights.sum())
    if total <= 0.0:
        return None
    return weights / total
