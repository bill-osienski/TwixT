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

    def finalize_game(
        self,
        *,
        winner: Optional[str],
        reason: str,
        n_moves: int,
        starting_player: str,
        iteration: int,
        game_idx: int,
        game_id: str,
    ) -> Optional[dict]:
        if not self.enabled:
            return None

        outcome_class = _classify_outcome(winner, reason)
        common_id = {
            "version": 1,
            "game_id": game_id,
            "iteration": iteration,
            "game_idx": game_idx,
            "starting_player": starting_player,
            "n_moves": n_moves,
            "reason": reason,
            "outcome_class": outcome_class,
        }

        if outcome_class == 1:
            focal = self.red if winner == "red" else self.black
            return _build_class1_record(
                common_id=common_id, winner=winner, focal=focal,
                actual_terminal_ply=n_moves,
                high_value_delay_threshold_plies=self.high_value_delay_threshold_plies,
            )
        if outcome_class == 2:
            focal_side, focal = _pick_class2_focal(self.red, self.black)
            return _build_class2_record(
                common_id=common_id, focal=focal, focal_side=focal_side,
                actual_terminal_ply=n_moves,
            )
        return _build_class3_record(common_id=common_id)


# ---------------------------------------------------------------------------
# Module-level helpers (not inside the class)
# ---------------------------------------------------------------------------


def _classify_outcome(winner: Optional[str], reason: str) -> int:
    if winner in ("red", "black"):
        return 1
    if reason in ("state_cap", "timeout", "board_full"):
        return 2
    return 3


def _pick_class2_focal(
    red_acc: _SideAccumulator, black_acc: _SideAccumulator,
) -> Tuple[str, _SideAccumulator]:
    """Tie-break: earliest first_dominant_unclosed_ply ->
    lower first_total_goal_distance -> red before black.

    If neither side is detected, returns ('red', red_acc) — caller will
    populate the record as detected=false / detected_player=null."""
    candidates = []
    if red_acc.detected:
        candidates.append(("red", red_acc))
    if black_acc.detected:
        candidates.append(("black", black_acc))
    if not candidates:
        return "red", red_acc
    candidates.sort(key=lambda c: (
        c[1].first_dominant_unclosed_ply if c[1].first_dominant_unclosed_ply is not None else 10**9,
        c[1].first_total_goal_distance if c[1].first_total_goal_distance is not None else 10**9,
        0 if c[0] == "red" else 1,
    ))
    return candidates[0]


def _build_class1_record(
    *, common_id: dict, winner: str, focal: _SideAccumulator,
    actual_terminal_ply: int, high_value_delay_threshold_plies: int,
) -> dict:
    detected = focal.detected
    first_ply = focal.first_dominant_unclosed_ply
    conversion_delay_plies = (
        actual_terminal_ply - first_ply if (detected and first_ply is not None) else None
    )
    # Conversion delay in winner-only moves: focal.moves_after_detection
    # already counts this side's post-detection moves only.
    conversion_delay_winner_moves = (
        focal.moves_after_detection if detected else None
    )
    if focal.search_scores_after_detection:
        max_ss = max(focal.search_scores_after_detection)
        mean_ss = sum(focal.search_scores_after_detection) / len(focal.search_scores_after_detection)
        coverage = len(focal.search_scores_after_detection)
    else:
        max_ss, mean_ss, coverage = None, None, 0

    if detected and conversion_delay_plies is not None:
        root_high_delayed = (
            focal.high_value_after_detection_plies >= 1
            and conversion_delay_plies >= high_value_delay_threshold_plies
        )
    else:
        root_high_delayed = False

    out = dict(common_id)
    out.update({
        "winner": winner,
        "detected_player": winner,
        "scope": "winner",
        "ever_distance_le_2": focal.ever_distance_le_2,
        "ever_distance_le_3": focal.ever_distance_le_3,
        "min_total_goal_distance": focal.min_total_goal_distance,
        "detected": detected,
        "first_dominant_unclosed_ply": first_ply,
        "first_total_goal_distance": focal.first_total_goal_distance,
        "first_category": focal.first_category,
        "first_largest_component_size": focal.first_largest_component_size,
        "first_endpoint_distances": focal.first_endpoint_distances,
        "actual_terminal_ply": actual_terminal_ply,
        "actual_win_ply": actual_terminal_ply,
        "conversion_delay_plies": conversion_delay_plies,
        "conversion_delay_winner_moves": conversion_delay_winner_moves,
        "cap_delay_proxy_plies": None,
        "winner_moves_in_watch_window": focal.moves_after_detection if detected else 0,
        "winner_moves_with_dominant_component": focal.moves_with_dominant_component if detected else 0,
        "winner_moves_with_dominant_unavailable": focal.moves_with_dominant_unavailable if detected else 0,
        "primary_class_counts": dict(focal.primary_class_counts),
        "max_search_score_after_detection": max_ss,
        "mean_search_score_after_detection": mean_ss,
        "high_value_after_detection_plies": focal.high_value_after_detection_plies,
        "root_value_high_but_delayed": root_high_delayed,
        "search_score_coverage_in_watch_window": coverage,
    })
    return out


def _build_class2_record(
    *, common_id: dict, focal: _SideAccumulator, focal_side: str,
    actual_terminal_ply: int,
) -> dict:
    detected = focal.detected
    first_ply = focal.first_dominant_unclosed_ply
    cap_delay = (
        actual_terminal_ply - first_ply if (detected and first_ply is not None) else None
    )
    out = dict(common_id)
    out.update({
        "winner": None,
        "detected_player": focal_side if detected else None,
        "scope": "both_sides",
        "ever_distance_le_2": focal.ever_distance_le_2,
        "ever_distance_le_3": focal.ever_distance_le_3,
        "min_total_goal_distance": focal.min_total_goal_distance,
        "detected": detected,
        "first_dominant_unclosed_ply": first_ply,
        "first_total_goal_distance": focal.first_total_goal_distance,
        "first_category": focal.first_category,
        "first_largest_component_size": focal.first_largest_component_size,
        "first_endpoint_distances": focal.first_endpoint_distances,
        "actual_terminal_ply": actual_terminal_ply,
        "actual_win_ply": None,
        "conversion_delay_plies": None,
        "conversion_delay_winner_moves": None,
        "cap_delay_proxy_plies": cap_delay,
        "winner_moves_in_watch_window": None,
        "winner_moves_with_dominant_component": None,
        "winner_moves_with_dominant_unavailable": None,
        "primary_class_counts": None,
        "max_search_score_after_detection": None,
        "mean_search_score_after_detection": None,
        "high_value_after_detection_plies": None,
        "root_value_high_but_delayed": None,
        "search_score_coverage_in_watch_window": None,
    })
    return out


def _build_class3_record(*, common_id: dict) -> dict:
    out = dict(common_id)
    out.update({
        "winner": None,
        "detected_player": None,
        "scope": "excluded",
        "detected": False,
        "actual_terminal_ply": common_id["n_moves"],
        "actual_win_ply": None,
        "conversion_delay_plies": None,
        "cap_delay_proxy_plies": None,
    })
    return out
