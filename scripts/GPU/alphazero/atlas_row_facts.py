"""Derive the atlas row facts from FROZEN measured fields -- design sections 3
and 8. Introduces no new constant and no new predicate.

Stage 4 accepted `phase`, `flat_policy` and `near_even` as caller-supplied
booleans, so the one seam it could not qualify was the one that computes them.
This module is that producer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .atlas_readout_c import is_flat
from .corpus_geometry import phase_for_ply, side_for_ply

# Section 8's existing near-even definition, verbatim. NOT a new threshold.
NEAR_EVEN_ABS_VALUE = 0.30


def _root_edge_priors(snapshots: Dict[str, Any]) -> Optional[Dict[int, float]]:
    """The merged deep line's ROOT edge priors, or None.

    `merge_reference_lines` asserts a parent's priors are identical at both deep
    rungs, and the ladder runs add_noise=False, so these ARE the root's priors
    rather than one rung's reading of them.

    `capture_tree_state` records `policy_entropy` and `n_legal` but NOT the top
    prior, so the frozen two-part predicate is not computable from a capture --
    which is why the reference line is the source here.
    """
    merged = ((snapshots or {}).get("reference_lines") or {}).get("merged") or {}
    for edge in merged.get("required_edges", ()):
        if edge.get("depth") == 0:
            return edge.get("parent_priors")
    return None


def derive_row_facts(legs: Sequence[Any], snapshots: Dict[str, Any],
                     target_ply: int, n_moves: int, start_player: str,
                     assigned_phase: Optional[str] = None,
                     assigned_side: Optional[str] = None) -> Dict[str, Any]:
    """The three row facts, plus the two the assignment already knows.

    `n_moves` sits immediately after `target_ply` because amendment 5 makes the
    two meaningless apart: a ply carries no phase without the trajectory it
    belongs to.

    `assigned_phase` / `assigned_side` turn the derivation into a CROSS-CHECK:
    disagreement means the assignment and the ply have drifted, which fails the
    row rather than being silently overwritten by either side.
    """
    phase = phase_for_ply(target_ply, n_moves)
    side = side_for_ply(target_ply, start_player)
    if assigned_phase is not None and assigned_phase != phase:
        raise ValueError(
            f"ply {target_ply} derives phase {phase!r} but the assignment says "
            f"{assigned_phase!r}; the assignment and the ply have drifted")
    if assigned_side is not None and assigned_side != side:
        raise ValueError(
            f"ply {target_ply} with start_player {start_player!r} derives side "
            f"{side!r} but the assignment says {assigned_side!r}")

    undefined: List[str] = []

    priors = _root_edge_priors(snapshots)
    if priors:
        flat_policy: Optional[bool] = is_flat(priors)
    else:
        # None, never False: no root edge means no root priors, which is not
        # the same fact as a concentrated policy.
        flat_policy = None
        undefined.append("flat_policy")

    by_b = {l.nominal_B: l for l in legs}
    if 400 in by_b:
        near_even: Optional[bool] = abs(by_b[400].root_value) <= NEAR_EVEN_ABS_VALUE
    else:
        near_even = None
        undefined.append("near_even")

    return {"phase": phase, "side": side, "flat_policy": flat_policy,
            "near_even": near_even, "undefined": undefined}
