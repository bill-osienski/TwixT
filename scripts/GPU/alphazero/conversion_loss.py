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


def make_conversion_aux_tensors(
    positions: list,
    legal_moves_padded: list,        # per-position list ordered like target_pi columns
    max_moves_cap: int,
    *,
    completion_weight: float = 1.0,
    reducer_weight: float = 0.35,
) -> tuple:
    """Return (aux_target, aux_mask) np arrays.

    aux_target shape: (B, max_moves_cap), float32
    aux_mask shape:   (B,), float32

    Padding entries in legal_moves_padded[i] (None values) are skipped.
    legal_moves_padded[i] is ordered exactly like target_pi[i] columns
    and move_mask[i] — same indexing as the policy CE computation.
    """
    B = len(positions)
    aux_target = np.zeros((B, max_moves_cap), dtype=np.float32)
    aux_mask   = np.zeros((B,), dtype=np.float32)

    for i, p in enumerate(positions):
        conv = getattr(p, "conversion", None)
        if conv is None:
            continue
        completion = {tuple(m) for m in conv.get("endpoint_completion_moves") or ()}
        reducing   = {tuple(m) for m in conv.get("distance_reducing_moves")   or ()}

        weights = np.zeros(max_moves_cap, dtype=np.float32)
        for j, m in enumerate(legal_moves_padded[i]):
            if m is None:                  # padding entry — skip
                continue
            if m in completion:
                weights[j] = completion_weight
            elif m in reducing:
                weights[j] = reducer_weight

        total = float(weights.sum())
        if total <= 0.0:
            continue
        aux_target[i] = weights / total
        aux_mask[i]   = 1.0

    return aux_target, aux_mask
