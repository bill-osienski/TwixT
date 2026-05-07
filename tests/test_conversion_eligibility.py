"""Eligibility predicate tests (Spec 2 §4)."""
from scripts.GPU.alphazero.conversion_loss import is_conversion_eligible


def _gc(total=2, comp=12, completion=None, reducing=None):
    return {
        "total_goal_distance": total,
        "largest_component_size": comp,
        "endpoint_completion_moves": completion if completion is not None else [(0, 8)],
        "distance_reducing_moves":   reducing   if reducing   is not None else [(22, 4)],
    }


def test_eligible_with_two_endpoint_closeout():
    gc = _gc(total=2, comp=12)
    assert is_conversion_eligible(gc, max_total_goal_distance=2, min_component_size=8) is True


def test_ineligible_when_total_distance_above_threshold():
    gc = _gc(total=4)
    assert is_conversion_eligible(gc, max_total_goal_distance=3, min_component_size=8) is False


def test_ineligible_when_component_too_small():
    gc = _gc(comp=6)
    assert is_conversion_eligible(gc, max_total_goal_distance=2, min_component_size=8) is False


def test_ineligible_when_no_completion_or_reducer_moves():
    gc = _gc(completion=[], reducing=[])
    assert is_conversion_eligible(gc, max_total_goal_distance=2, min_component_size=8) is False


def test_ineligible_when_gc_state_full_is_none():
    assert is_conversion_eligible(None, max_total_goal_distance=2, min_component_size=8) is False


def test_ineligible_when_total_distance_is_none():
    gc = _gc()
    gc["total_goal_distance"] = None
    assert is_conversion_eligible(gc, max_total_goal_distance=2, min_component_size=8) is False
