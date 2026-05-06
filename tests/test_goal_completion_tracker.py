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
