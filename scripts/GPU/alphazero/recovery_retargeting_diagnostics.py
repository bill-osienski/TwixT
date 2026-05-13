"""Spec 4 — Recovery / re-targeting diagnostic.

Per-side, per-game tracker that detects collapse/re-targeting failure
patterns from MCTS root values and visit-share concentration. Diagnostic
only: does not affect MCTS, selection, or training targets.

Spec: docs/superpowers/specs/2026-05-12-recovery-retargeting-diagnostic-design.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RecoveryRetargetingConfig:
    enabled: bool = True
    collapse_value_threshold: float = -0.75
    severe_collapse_value_threshold: float = -0.90
    diffuse_root_top1_threshold: float = 0.20
    very_diffuse_root_top1_threshold: float = 0.15
    delta_threshold: float = 0.50
    delta_max_current_score: float = -0.30
    alternate_component_min_size: int = 4
    classify_defense: bool = True
    max_sampled_moves_per_side: int = 32
    sample_all_moves: bool = False


def validate_config(cfg: RecoveryRetargetingConfig) -> None:
    """Raise ValueError on out-of-band config. Called once at startup."""
    if not (cfg.collapse_value_threshold < cfg.delta_max_current_score):
        raise ValueError(
            f"collapse_value_threshold ({cfg.collapse_value_threshold}) must be "
            f"strictly less than delta_max_current_score ({cfg.delta_max_current_score}) "
            f"so the delta path doesn't subsume the steady-state path"
        )
    if not (cfg.severe_collapse_value_threshold <= cfg.collapse_value_threshold):
        raise ValueError(
            f"severe_collapse_value_threshold ({cfg.severe_collapse_value_threshold}) "
            f"must be <= collapse_value_threshold ({cfg.collapse_value_threshold})"
        )
    if not (cfg.very_diffuse_root_top1_threshold <= cfg.diffuse_root_top1_threshold):
        raise ValueError(
            f"very_diffuse_root_top1_threshold ({cfg.very_diffuse_root_top1_threshold}) "
            f"must be <= diffuse_root_top1_threshold ({cfg.diffuse_root_top1_threshold})"
        )
    if not (0.0 <= cfg.diffuse_root_top1_threshold <= 1.0):
        raise ValueError(
            f"diffuse_root_top1_threshold ({cfg.diffuse_root_top1_threshold}) must be in [0, 1]"
        )
    if not (0.0 <= cfg.very_diffuse_root_top1_threshold <= 1.0):
        raise ValueError(
            f"very_diffuse_root_top1_threshold ({cfg.very_diffuse_root_top1_threshold}) must be in [0, 1]"
        )
    if not (cfg.delta_threshold > 0):
        raise ValueError(f"delta_threshold ({cfg.delta_threshold}) must be > 0")
    if not (cfg.alternate_component_min_size >= 1):
        raise ValueError(
            f"alternate_component_min_size ({cfg.alternate_component_min_size}) must be >= 1"
        )
    if not (cfg.max_sampled_moves_per_side >= 0):
        raise ValueError(
            f"max_sampled_moves_per_side ({cfg.max_sampled_moves_per_side}) must be >= 0"
        )


# ---------------------------------------------------------------------------
# Component analysis helpers
# ---------------------------------------------------------------------------

_KNIGHT_OFFSETS = ((1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1))


def knight_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
    """The 8 TwixT knight-distance offsets from (r, c). No bounds check."""
    return [(r + dr, c + dc) for dr, dc in _KNIGHT_OFFSETS]


def find_components(state, side: str) -> List[frozenset]:
    """All same-color bridge-connected components for `side` on the current state.

    Uses state._get_connected_component which respects enemy-blocking of bridges
    on real states. The _StubState fixture in tests provides a simpler
    knight-neighbor walk; that's sufficient for unit testing the classifier
    logic without a full Twixt board.
    """
    pegs_of = [p for p, color in state.pegs.items() if color == side]
    seen: set = set()
    components: List[frozenset] = []
    for peg in pegs_of:
        if peg in seen:
            continue
        comp = frozenset(state._get_connected_component(peg, side))
        if not comp:
            comp = frozenset({peg})
        seen.update(comp)
        components.append(comp)
    return components


def is_local_to_existing(state, side: str, move: Tuple[int, int]) -> bool:
    """True iff `move` is at TwixT knight distance of at least one same-color peg.

    Bridge-formability is NOT required; the flag is about proximity to
    bridge-able structure, per spec §3.1.
    """
    r, c = move
    for (nr, nc) in knight_neighbors(r, c):
        if state.pegs.get((nr, nc)) == side:
            return True
    return False


def selected_component_after(state_after, side: str, move: Tuple[int, int]) -> frozenset:
    """The component containing `move` in the POST-MOVE state.

    Caller is responsible for constructing `state_after` (via state.apply_move
    or equivalent). This helper performs no state mutation — making it safe
    to call inside the per-ply hook without copying or restoring board state.
    """
    comp = frozenset(state_after._get_connected_component(move, side))
    if not comp:
        comp = frozenset({move})
    return comp


# ---------------------------------------------------------------------------
# Trigger evaluation
# ---------------------------------------------------------------------------

def evaluate_trigger(
    *,
    current_search_score: Optional[float],
    root_top1_share: Optional[float],
    previous_own_search_score: Optional[float],
    config: RecoveryRetargetingConfig,
) -> dict:
    """Per-ply trigger decision. Pure function. Spec §2.

    Returns:
        {
          "triggered": bool,
          "trigger_reason": None | "delta_precursor" | "steady_state" | "both",
          "is_severe_collapse": bool,
          "is_very_diffuse": bool,
          "missing_search_score": bool,
          "missing_root_top1_share": bool,
          "search_score_delta": Optional[float],
        }
    """
    missing_search_score = current_search_score is None
    missing_root_top1_share = root_top1_share is None
    if missing_search_score or missing_root_top1_share:
        return {
            "triggered": False,
            "trigger_reason": None,
            "is_severe_collapse": False,
            "is_very_diffuse": False,
            "missing_search_score": missing_search_score,
            "missing_root_top1_share": missing_root_top1_share,
            "search_score_delta": None,
        }

    diffuse_root = root_top1_share <= config.diffuse_root_top1_threshold

    delta_value = (
        previous_own_search_score - current_search_score
        if previous_own_search_score is not None else None
    )
    delta_precursor = (
        previous_own_search_score is not None
        and delta_value is not None
        and delta_value >= config.delta_threshold
        and current_search_score <= config.delta_max_current_score
        and diffuse_root
    )

    steady_state = (
        current_search_score <= config.collapse_value_threshold
        and diffuse_root
    )

    if delta_precursor and steady_state:
        trigger_reason = "both"
        triggered = True
    elif delta_precursor:
        trigger_reason = "delta_precursor"
        triggered = True
    elif steady_state:
        trigger_reason = "steady_state"
        triggered = True
    else:
        trigger_reason = None
        triggered = False

    return {
        "triggered": triggered,
        "trigger_reason": trigger_reason,
        "is_severe_collapse": current_search_score <= config.severe_collapse_value_threshold,
        "is_very_diffuse": root_top1_share <= config.very_diffuse_root_top1_threshold,
        "missing_search_score": False,
        "missing_root_top1_share": False,
        "search_score_delta": delta_value,
    }


# ---------------------------------------------------------------------------
# Primary-class classifier
# ---------------------------------------------------------------------------

PRIMARY_CLASSES = (
    "blocks_opponent_closeout",
    "reduces_own_goal_distance",
    "starts_or_extends_alternate_component",
    "connects_to_existing_component",
    "improves_own_largest_component",
    "redundant_local_reinforcement",
    "off_plan_or_unclear",
)


def _dominant_component(components: List[frozenset]) -> Optional[frozenset]:
    """Largest component by size; tie-break by lexicographically-smallest peg."""
    if not components:
        return None
    return max(components, key=lambda c: (len(c), -min(c)[0] if c else 0, -min(c)[1] if c else 0))


def classify_move(
    *,
    state_before,
    state_after,
    side: str,
    move: Tuple[int, int],
    own_total_goal_distance_before: Optional[int],
    own_total_goal_distance_after: Optional[int],
    opponent_total_goal_distance_before: Optional[int],
    opponent_total_goal_distance_after: Optional[int],
    classify_defense: bool,
    alternate_component_min_size: int,
) -> dict:
    """Classify a single move into one of PRIMARY_CLASSES. Spec §3.

    Both `state_before` (pre-move) and `state_after` (post-move) are passed
    in; the classifier never mutates either. Caller computes state_after
    once via state.apply_move(move) and reuses it.

    Returns:
        {
          "primary_class": str,
          "flags": {
            "opens_new_component": bool,
            "merges_components": bool,
            "merges_dominant_with_alternate": bool,
            "extends_dominant_component": bool,
            "local_to_existing": bool,
            "blocked_opponent_closeout": bool,
          },
          "own_largest_component_size_before": int,
          "own_largest_component_size_after": int,
        }
    """
    own_components_before = find_components(state_before, side)
    dominant_before = _dominant_component(own_components_before)
    selected_after = selected_component_after(state_after, side, move)
    local_flag = is_local_to_existing(state_before, side, move)

    prior_components_extended = [c for c in own_components_before if c <= selected_after]
    opens_new = len(prior_components_extended) == 0
    merges = len(prior_components_extended) >= 2
    extends_dominant = dominant_before is not None and (dominant_before <= selected_after)
    merges_dom_alt = extends_dominant and merges
    extends_only_non_dominant = (
        len(prior_components_extended) == 1
        and not extends_dominant
    )

    largest_before = max((len(c) for c in own_components_before), default=0)
    own_components_after = find_components(state_after, side)
    largest_after = max((len(c) for c in own_components_after), default=0)

    # Defense check (priority 1, only when classify_defense=True).
    blocked_opp = False
    if classify_defense and opponent_total_goal_distance_before is not None and opponent_total_goal_distance_before <= 2:
        if (opponent_total_goal_distance_after is None
                or opponent_total_goal_distance_after > opponent_total_goal_distance_before):
            blocked_opp = True

    flags = {
        "opens_new_component":            opens_new,
        "merges_components":              merges,
        "merges_dominant_with_alternate": merges_dom_alt,
        "extends_dominant_component":     extends_dominant,
        "local_to_existing":              local_flag,
        "blocked_opponent_closeout":      blocked_opp,
    }

    # Priority-ordered classification.
    if blocked_opp:
        primary = "blocks_opponent_closeout"
    elif (own_total_goal_distance_before is not None
          and own_total_goal_distance_after is not None
          and own_total_goal_distance_after < own_total_goal_distance_before):
        primary = "reduces_own_goal_distance"
    elif (
        not extends_dominant
        and (opens_new or extends_only_non_dominant or merges)
        and len(selected_after) >= alternate_component_min_size
    ):
        primary = "starts_or_extends_alternate_component"
    elif len(prior_components_extended) >= 1:
        primary = "connects_to_existing_component"
    elif largest_after > largest_before:
        primary = "improves_own_largest_component"
    elif (local_flag
          and (own_total_goal_distance_before is None
               or own_total_goal_distance_after is None
               or own_total_goal_distance_after >= own_total_goal_distance_before)):
        primary = "redundant_local_reinforcement"
    else:
        primary = "off_plan_or_unclear"

    return {
        "primary_class": primary,
        "flags": flags,
        "own_largest_component_size_before": largest_before,
        "own_largest_component_size_after": largest_after,
    }
