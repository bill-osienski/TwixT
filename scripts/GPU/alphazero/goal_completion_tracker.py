"""Per-game inline goal-completion tracker (spec 2026-05-05).

Observes per-ply self-play events; emits one compact goal_completion_record
per game. Replaces the analyzer's replay-side BFS aggregation as the
canonical source of goal-completion telemetry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from scripts.GPU.alphazero.connectivity_diagnostics import (
    classify_selected_conversion_move,
)


def _zero_class_counts() -> dict:
    return {
        "completes_endpoint": 0,
        "reduces_total_goal_distance": 0,
        "redundant_reinforcement": 0,
        "off_chain": 0,
        "other": 0,
    }


@dataclass
class _SideAccumulator:
    """Per-side per-game accumulator. Both sides tracked in parallel
    during play; finalize_game picks the focal side based on outcome."""
    detected: bool = False
    first_dominant_unclosed_ply: Optional[int] = None
    first_total_goal_distance: Optional[int] = None
    first_category: Optional[str] = None
    first_largest_component_size: Optional[int] = None
    first_endpoint_distances: Optional[dict] = None
    primary_class_counts: dict = field(default_factory=_zero_class_counts)
    moves_after_detection: int = 0
    moves_with_dominant_component: int = 0
    moves_with_dominant_unavailable: int = 0
    search_scores_after_detection: list = field(default_factory=list)
    high_value_after_detection_plies: int = 0
    min_total_goal_distance: Optional[int] = None
    ever_distance_le_2: bool = False
    ever_distance_le_3: bool = False


@dataclass
class GoalCompletionGameTracker:
    enabled: bool = True
    detection_threshold: int = 2
    high_value_threshold: float = 0.9
    high_value_delay_threshold_plies: int = 6
    max_depth: int = 3
    min_component_size: int = 8
    red: _SideAccumulator = field(default_factory=_SideAccumulator)
    black: _SideAccumulator = field(default_factory=_SideAccumulator)

    def is_detected(self, side: str) -> bool:
        if side == "red":
            return self.red.detected
        if side == "black":
            return self.black.detected
        return False

    def observe_pre_move(
        self,
        *,
        state,                      # TwixtState; unused in Task 1, used in Task 2
        ply: int,
        side_to_move: str,
        selected_move: Tuple[int, int],
        search_score: Optional[float],
        gc_state_cheap: Optional[dict],
        gc_state_full: Optional[dict],
    ) -> None:
        if not self.enabled:
            return
        if side_to_move not in ("red", "black"):
            return
        acc = self.red if side_to_move == "red" else self.black

        # 1. Coverage flags from cheap state.
        if gc_state_cheap is not None:
            total = gc_state_cheap.get("total_goal_distance")
            if total is not None:
                if acc.min_total_goal_distance is None or total < acc.min_total_goal_distance:
                    acc.min_total_goal_distance = total
                if total <= 2:
                    acc.ever_distance_le_2 = True
                if total <= 3:
                    acc.ever_distance_le_3 = True

        # 2. Detection update (only first event).
        if not acc.detected and gc_state_cheap is not None:
            total = gc_state_cheap.get("total_goal_distance")
            if total is not None and total <= self.detection_threshold:
                acc.detected = True
                acc.first_dominant_unclosed_ply = ply
                acc.first_total_goal_distance = total
                acc.first_category = gc_state_cheap.get("category")
                acc.first_endpoint_distances = (
                    dict(gc_state_cheap.get("endpoint_distances") or {})
                    if gc_state_cheap.get("endpoint_distances") is not None else None
                )
                comp = gc_state_cheap.get("component_pegs")
                acc.first_largest_component_size = len(comp) if comp else None

        # 3. Watch-window: if detected (either before or just now),
        # the selected move counts as post-detection. Classify when full
        # state is available; otherwise log as dominant_unavailable.
        if acc.detected:
            acc.moves_after_detection += 1
            if gc_state_cheap is None:
                acc.moves_with_dominant_unavailable += 1
            elif gc_state_full is None:
                acc.moves_with_dominant_unavailable += 1
            else:
                acc.moves_with_dominant_component += 1
                cls = classify_selected_conversion_move(
                    state, side_to_move, selected_move, gc_state_full,
                    max_depth=self.max_depth,
                    min_component_size=self.min_component_size,
                )
                primary = cls.get("primary_class", "other")
                if primary in acc.primary_class_counts:
                    acc.primary_class_counts[primary] += 1
                else:
                    acc.primary_class_counts["other"] += 1

            if search_score is not None:
                ss = float(search_score)
                acc.search_scores_after_detection.append(ss)
                if ss >= self.high_value_threshold:
                    acc.high_value_after_detection_plies += 1
