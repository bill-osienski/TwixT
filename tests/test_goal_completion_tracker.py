"""Tracker unit tests (spec 2026-05-05 §6).

Pre-move detection semantics: detection fires when the side to move
already has a closeout-shaped position pre-move. The selected move on
the detection ply IS counted as a post-detection move (classification
arrives in Task 2).
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.goal_completion_tracker import (
    GoalCompletionGameTracker,
    _SideAccumulator,
)


def _gc_state(total, category="two_endpoint_closeout_2ply",
              endpoint_distances=None, component_pegs=None):
    """Build a minimal gc_state dict for the tracker's coverage/detection path."""
    return {
        "total_goal_distance": total,
        "category": category,
        "endpoint_distances": endpoint_distances or {"top": 0, "bottom": 1},
        "component_pegs": component_pegs or frozenset({(0, 0), (2, 1), (4, 2)}),
    }


def test_tracker_disabled_observe_is_noop():
    t = GoalCompletionGameTracker(enabled=False)
    t.observe_pre_move(
        state=None, ply=1, side_to_move="red",
        selected_move=(5, 5), search_score=None,
        gc_state_cheap=_gc_state(2),
        gc_state_full=None,
    )
    assert t.red.detected is False
    assert t.red.first_dominant_unclosed_ply is None


def test_tracker_coverage_flags_update_per_side_to_move():
    t = GoalCompletionGameTracker()
    # Red's pre-move state at ply 5: total=4. Not below threshold yet but
    # min/ever flags should advance.
    t.observe_pre_move(
        state=None, ply=5, side_to_move="red",
        selected_move=(0, 0), search_score=None,
        gc_state_cheap=_gc_state(4), gc_state_full=None,
    )
    assert t.red.min_total_goal_distance == 4
    assert t.red.ever_distance_le_2 is False
    assert t.red.ever_distance_le_3 is False
    # Black's accumulator is untouched.
    assert t.black.min_total_goal_distance is None


def test_tracker_premove_detection_fires_at_first_eligible_side_move():
    """Detection ply equals the first ply where the side to move already
    has total_goal_distance <= detection_threshold pre-move. Detection
    threshold defaults to 2."""
    t = GoalCompletionGameTracker(detection_threshold=2)
    # Red's first three moves: total decreasing 5 -> 3 -> 2.
    # Detection fires at the THIRD red move (ply 5, 1-indexed).
    t.observe_pre_move(state=None, ply=1, side_to_move="red",
                       selected_move=(0,0), search_score=None,
                       gc_state_cheap=_gc_state(5), gc_state_full=None)
    t.observe_pre_move(state=None, ply=3, side_to_move="red",
                       selected_move=(1,1), search_score=None,
                       gc_state_cheap=_gc_state(3), gc_state_full=None)
    assert t.red.detected is False
    t.observe_pre_move(state=None, ply=5, side_to_move="red",
                       selected_move=(2,2), search_score=None,
                       gc_state_cheap=_gc_state(2, category="two_endpoint_closeout_2ply"),
                       gc_state_full=None)
    assert t.red.detected is True
    assert t.red.first_dominant_unclosed_ply == 5
    assert t.red.first_total_goal_distance == 2
    assert t.red.first_category == "two_endpoint_closeout_2ply"
    assert t.red.first_endpoint_distances == {"top": 0, "bottom": 1}


def test_tracker_first_largest_component_size_recorded_at_detection():
    t = GoalCompletionGameTracker()
    component = frozenset({(0, 0), (2, 1), (4, 2), (6, 3), (8, 4)})
    t.observe_pre_move(state=None, ply=11, side_to_move="black",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2, category="one_endpoint_distance_2",
                                                component_pegs=component),
                       gc_state_full=None)
    assert t.black.detected is True
    assert t.black.first_largest_component_size == 5


def test_tracker_detection_records_only_first_event():
    """Once detected, subsequent observations on the same side do not
    overwrite first-detection metadata."""
    t = GoalCompletionGameTracker()
    t.observe_pre_move(state=None, ply=7, side_to_move="red",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2, category="two_endpoint_closeout_2ply"),
                       gc_state_full=None)
    t.observe_pre_move(state=None, ply=9, side_to_move="red",
                       selected_move=(1, 1), search_score=None,
                       gc_state_cheap=_gc_state(1, category="one_move_win"),
                       gc_state_full=None)
    assert t.red.first_dominant_unclosed_ply == 7
    assert t.red.first_total_goal_distance == 2
    assert t.red.first_category == "two_endpoint_closeout_2ply"
    # min should track the lower value though.
    assert t.red.min_total_goal_distance == 1


def test_tracker_dual_side_independent():
    """Both sides reach detection independently in the same game."""
    t = GoalCompletionGameTracker()
    t.observe_pre_move(state=None, ply=11, side_to_move="red",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    t.observe_pre_move(state=None, ply=14, side_to_move="black",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    assert t.red.detected is True and t.red.first_dominant_unclosed_ply == 11
    assert t.black.detected is True and t.black.first_dominant_unclosed_ply == 14


def test_tracker_is_detected_helper():
    t = GoalCompletionGameTracker()
    assert t.is_detected("red") is False
    t.observe_pre_move(state=None, ply=3, side_to_move="red",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    assert t.is_detected("red") is True
    assert t.is_detected("black") is False


# ---------------------------------------------------------------------------
# Task 2: Watch-window classification tests
# ---------------------------------------------------------------------------
from unittest.mock import patch


def _gc_state_full(total, completion_moves=(), reducing_moves=()):
    return {
        "total_goal_distance": total,
        "category": "two_endpoint_closeout_2ply",
        "endpoint_distances": {"top": 0, "bottom": 1},
        "component_pegs": frozenset({(0, 0), (2, 1), (4, 2)}),
        "endpoint_completion_moves": list(completion_moves),
        "distance_reducing_moves": list(reducing_moves),
        "moves_enumerated": True,
    }


def test_tracker_premove_detection_classifies_detection_ply_move():
    """ANCHOR: Pre-move semantics — the move on the detection ply itself
    counts as a post-detection move and IS classified."""
    t = GoalCompletionGameTracker()
    full = _gc_state_full(total=2, completion_moves=[(7, 7)])

    fake_cls = {"primary_class": "completes_endpoint",
                "completes_endpoint": True,
                "reduces_total_goal_distance": False,
                "is_redundant_reinforcement": False,
                "is_off_chain": False,
                "total_goal_distance_before": 2,
                "total_goal_distance_after": 0}

    with patch(
        "scripts.GPU.alphazero.goal_completion_tracker.classify_selected_conversion_move",
        return_value=fake_cls,
    ) as mock_cls:
        t.observe_pre_move(
            state="<state>", ply=11, side_to_move="red",
            selected_move=(7, 7), search_score=0.99,
            gc_state_cheap=full, gc_state_full=full,
        )

    assert t.red.detected is True
    assert t.red.first_dominant_unclosed_ply == 11
    assert t.red.moves_after_detection == 1
    assert t.red.moves_with_dominant_component == 1
    assert t.red.moves_with_dominant_unavailable == 0
    assert t.red.primary_class_counts["completes_endpoint"] == 1
    assert t.red.search_scores_after_detection == [0.99]
    assert t.red.high_value_after_detection_plies == 1
    assert mock_cls.call_count == 1


def test_tracker_classification_each_primary_class():
    """Each primary_class string maps to its own counter."""
    t = GoalCompletionGameTracker()
    full = _gc_state_full(total=2)

    cases = [
        ("completes_endpoint", 13),
        ("reduces_total_goal_distance", 15),
        ("redundant_reinforcement", 17),
        ("off_chain", 19),
        ("other", 21),
    ]
    for primary, ply in cases:
        with patch(
            "scripts.GPU.alphazero.goal_completion_tracker.classify_selected_conversion_move",
            return_value={"primary_class": primary},
        ):
            t.observe_pre_move(
                state="<state>", ply=ply, side_to_move="red",
                selected_move=(0, 0), search_score=None,
                gc_state_cheap=full, gc_state_full=full,
            )

    assert t.red.primary_class_counts == {
        "completes_endpoint": 1,
        "reduces_total_goal_distance": 1,
        "redundant_reinforcement": 1,
        "off_chain": 1,
        "other": 1,
    }
    assert t.red.moves_after_detection == 5


def test_tracker_unknown_primary_class_falls_to_other():
    t = GoalCompletionGameTracker()
    full = _gc_state_full(total=2)
    with patch(
        "scripts.GPU.alphazero.goal_completion_tracker.classify_selected_conversion_move",
        return_value={"primary_class": "garbled_string"},
    ):
        t.observe_pre_move(
            state="<state>", ply=11, side_to_move="red",
            selected_move=(0, 0), search_score=None,
            gc_state_cheap=full, gc_state_full=full,
        )
    assert t.red.primary_class_counts["other"] == 1


def test_tracker_dominant_unavailable_when_cheap_state_none_post_detection():
    """If the focal side already detected but a later ply has no dominant
    component (cheap state is None), count as dominant_unavailable."""
    t = GoalCompletionGameTracker()
    # First, get red detected.
    t.observe_pre_move(state="<state>", ply=11, side_to_move="red",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    # Next red ply: cheap state is None.
    t.observe_pre_move(state="<state>", ply=13, side_to_move="red",
                       selected_move=(1, 1), search_score=None,
                       gc_state_cheap=None, gc_state_full=None)
    assert t.red.moves_after_detection == 2  # detection ply + this one
    # Both plies fall under "dominant_unavailable":
    #   ply 11: cheap state present but no gc_state_full -> classification
    #           skipped, defensive count -> +1
    #   ply 13: cheap state is None -> +1
    assert t.red.moves_with_dominant_unavailable == 2
    assert sum(t.red.primary_class_counts.values()) == 0


def test_tracker_no_full_state_treated_as_dominant_unavailable():
    """When cheap state exists post-detection but full was not provided,
    we cannot classify; treat as dominant_unavailable defensively."""
    t = GoalCompletionGameTracker()
    t.observe_pre_move(state="<state>", ply=11, side_to_move="red",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    assert t.red.detected is True
    assert t.red.moves_after_detection == 1
    assert t.red.moves_with_dominant_unavailable == 1
    assert sum(t.red.primary_class_counts.values()) == 0


def test_tracker_search_score_high_value_count():
    """high_value_after_detection_plies counts post-detection plies where
    search_score >= high_value_threshold (default 0.9)."""
    t = GoalCompletionGameTracker(high_value_threshold=0.9)
    full = _gc_state_full(total=2)
    with patch(
        "scripts.GPU.alphazero.goal_completion_tracker.classify_selected_conversion_move",
        return_value={"primary_class": "redundant_reinforcement"},
    ):
        for ply, score in [(11, 0.95), (13, 0.5), (15, 0.91), (17, None)]:
            t.observe_pre_move(state="<state>", ply=ply, side_to_move="red",
                               selected_move=(0, 0), search_score=score,
                               gc_state_cheap=full, gc_state_full=full)
    assert t.red.search_scores_after_detection == [0.95, 0.5, 0.91]
    assert t.red.high_value_after_detection_plies == 2


def test_tracker_opponent_side_unaffected_by_focal_classification():
    t = GoalCompletionGameTracker()
    full = _gc_state_full(total=2)
    with patch(
        "scripts.GPU.alphazero.goal_completion_tracker.classify_selected_conversion_move",
        return_value={"primary_class": "completes_endpoint"},
    ):
        t.observe_pre_move(state="<state>", ply=11, side_to_move="red",
                           selected_move=(0, 0), search_score=0.95,
                           gc_state_cheap=full, gc_state_full=full)
    assert t.black.detected is False
    assert t.black.moves_after_detection == 0
    assert sum(t.black.primary_class_counts.values()) == 0


# ---------------------------------------------------------------------------
# Task 3: finalize_game tests
# ---------------------------------------------------------------------------


def test_finalize_class3_when_disabled_returns_none():
    t = GoalCompletionGameTracker(enabled=False)
    rec = t.finalize_game(
        winner="red", reason="win", n_moves=20,
        starting_player="red", iteration=110, game_idx=5,
        game_id="iter_0110_game_005",
    )
    assert rec is None


def test_finalize_class1_basic_decisive_winner():
    """Class 1: winner == detected side. All winner-perspective fields populated."""
    t = GoalCompletionGameTracker(high_value_delay_threshold_plies=6)
    full = _gc_state_full(total=2)
    # Detect red at ply 11 with two_endpoint_closeout shape.
    with patch(
        "scripts.GPU.alphazero.goal_completion_tracker.classify_selected_conversion_move",
        return_value={"primary_class": "completes_endpoint"},
    ):
        t.observe_pre_move(state="<state>", ply=11, side_to_move="red",
                           selected_move=(0, 0), search_score=0.99,
                           gc_state_cheap=full, gc_state_full=full)

    rec = t.finalize_game(
        winner="red", reason="win", n_moves=21,
        starting_player="red", iteration=112, game_idx=34,
        game_id="iter_0112_game_034",
    )
    assert rec["version"] == 1
    assert rec["outcome_class"] == 1
    assert rec["scope"] == "winner"
    assert rec["winner"] == "red"
    assert rec["detected_player"] == "red"
    assert rec["reason"] == "win"
    assert rec["detected"] is True
    assert rec["first_dominant_unclosed_ply"] == 11
    assert rec["first_total_goal_distance"] == 2
    assert rec["first_category"] == "two_endpoint_closeout_2ply"
    assert rec["actual_terminal_ply"] == 21
    assert rec["actual_win_ply"] == 21
    assert rec["conversion_delay_plies"] == 10  # 21 - 11
    # winner_moves_in_watch_window mirrors moves_after_detection for Class 1.
    assert rec["winner_moves_in_watch_window"] == 1
    assert rec["winner_moves_with_dominant_component"] == 1
    assert rec["winner_moves_with_dominant_unavailable"] == 0
    assert rec["primary_class_counts"]["completes_endpoint"] == 1
    assert rec["max_search_score_after_detection"] == 0.99
    assert rec["mean_search_score_after_detection"] == 0.99
    assert rec["high_value_after_detection_plies"] == 1
    # delay_plies 10 >= high_value_delay_threshold 6 AND high_value >= 1
    assert rec["root_value_high_but_delayed"] is True
    assert rec["search_score_coverage_in_watch_window"] == 1
    assert rec["cap_delay_proxy_plies"] is None


def test_finalize_class1_winner_never_detected():
    """Edge: winner exists but their side never reached detection threshold.
    detected=false, all post-detection fields null/0."""
    t = GoalCompletionGameTracker()
    # Red plays, total stays at 4 throughout — no detection.
    t.observe_pre_move(state="<state>", ply=1, side_to_move="red",
                       selected_move=(0, 0), search_score=0.5,
                       gc_state_cheap=_gc_state(4), gc_state_full=None)
    rec = t.finalize_game(
        winner="red", reason="win", n_moves=20,
        starting_player="red", iteration=110, game_idx=0,
        game_id="iter_0110_game_000",
    )
    assert rec["outcome_class"] == 1
    assert rec["winner"] == "red"
    assert rec["detected"] is False
    assert rec["first_dominant_unclosed_ply"] is None
    assert rec["conversion_delay_plies"] is None
    assert rec["winner_moves_in_watch_window"] == 0
    assert rec["primary_class_counts"] == {
        "completes_endpoint": 0,
        "reduces_total_goal_distance": 0,
        "redundant_reinforcement": 0,
        "off_chain": 0,
        "other": 0,
    }
    assert rec["root_value_high_but_delayed"] is False


def test_finalize_class2_capped_focal_earliest_detector():
    """Class 2: no winner; focal = earliest detector. cap_delay_proxy_plies =
    actual_terminal_ply - first_dominant_unclosed_ply."""
    t = GoalCompletionGameTracker()
    # Red detects at ply 41, Black at ply 60 — Red is focal.
    t.observe_pre_move(state="<state>", ply=41, side_to_move="red",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    t.observe_pre_move(state="<state>", ply=60, side_to_move="black",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)

    rec = t.finalize_game(
        winner=None, reason="state_cap", n_moves=120,
        starting_player="red", iteration=112, game_idx=7,
        game_id="iter_0112_game_007",
    )
    assert rec["outcome_class"] == 2
    assert rec["scope"] == "both_sides"
    assert rec["winner"] is None
    assert rec["detected_player"] == "red"
    assert rec["detected"] is True
    assert rec["first_dominant_unclosed_ply"] == 41
    assert rec["first_total_goal_distance"] == 2
    assert rec["actual_terminal_ply"] == 120
    assert rec["actual_win_ply"] is None
    assert rec["conversion_delay_plies"] is None
    assert rec["cap_delay_proxy_plies"] == 79  # 120 - 41
    assert rec["primary_class_counts"] is None
    assert rec["max_search_score_after_detection"] is None


def test_finalize_class2_tiebreak_lower_first_total_then_red():
    """Tie-break: same ply -> lower first_total_goal_distance -> red."""
    # Equal-ply, equal-distance -> red wins.
    t = GoalCompletionGameTracker()
    t.observe_pre_move(state="<state>", ply=50, side_to_move="red",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    t.observe_pre_move(state="<state>", ply=50, side_to_move="black",
                       selected_move=(0, 0), search_score=None,
                       gc_state_cheap=_gc_state(2), gc_state_full=None)
    rec = t.finalize_game(
        winner=None, reason="timeout", n_moves=80,
        starting_player="red", iteration=110, game_idx=0,
        game_id="iter_0110_game_000",
    )
    assert rec["detected_player"] == "red"

    # Equal-ply, black has lower distance -> black wins.
    t2 = GoalCompletionGameTracker()
    t2.observe_pre_move(state="<state>", ply=50, side_to_move="red",
                        selected_move=(0, 0), search_score=None,
                        gc_state_cheap=_gc_state(2), gc_state_full=None)
    t2.observe_pre_move(state="<state>", ply=50, side_to_move="black",
                        selected_move=(0, 0), search_score=None,
                        gc_state_cheap=_gc_state(1), gc_state_full=None)
    rec2 = t2.finalize_game(
        winner=None, reason="timeout", n_moves=80,
        starting_player="red", iteration=110, game_idx=0,
        game_id="iter_0110_game_000",
    )
    assert rec2["detected_player"] == "black"


def test_finalize_class2_no_detection_either_side():
    """Class 2 with neither side detected: detected=false, proxy null."""
    t = GoalCompletionGameTracker()
    rec = t.finalize_game(
        winner=None, reason="state_cap", n_moves=200,
        starting_player="red", iteration=110, game_idx=0,
        game_id="iter_0110_game_000",
    )
    assert rec["outcome_class"] == 2
    assert rec["detected"] is False
    assert rec["detected_player"] is None
    assert rec["first_dominant_unclosed_ply"] is None
    assert rec["cap_delay_proxy_plies"] is None


def test_finalize_class3_excluded():
    """Unhandled outcome -> Class 3 minimal record."""
    t = GoalCompletionGameTracker()
    rec = t.finalize_game(
        winner=None, reason="unknown", n_moves=0,
        starting_player="red", iteration=110, game_idx=0,
        game_id="iter_0110_game_000",
    )
    assert rec["outcome_class"] == 3
    assert rec["scope"] == "excluded"
    assert rec["winner"] is None
    assert rec["detected_player"] is None
    assert rec["detected"] is False
    assert rec["actual_terminal_ply"] == 0
    assert rec["actual_win_ply"] is None
    assert rec["conversion_delay_plies"] is None
    assert rec["cap_delay_proxy_plies"] is None
