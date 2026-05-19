"""Tests for the long-tail bucket classifier.

Spec: docs/superpowers/specs/2026-05-19-long-tail-bucket-classifier-design.md
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.long_tail_bucket_classifier import (
    matches_long_tail_filter,
    classify_long_tail_bucket,
    aggregate_long_tail_buckets,
    LONG_TAIL_BUCKETS,
    NOT_LONG_TAIL,
)


def _rec(**overrides):
    """Per-game goal_completion_record fixture with sensible defaults
    that do NOT match the long-tail filter."""
    base = {
        "iteration": 200,
        "game_idx": 0,
        "game_id": "game_000",
        "winner": "red",
        "n_moves": 80,
        "reason": "win",
        "first_total_goal_distance": 2,
        "first_category": "two_endpoint_closeout",
        "conversion_delay_plies": 5,
        "winner_moves_with_dominant_unavailable": 0,
    }
    base.update(overrides)
    return base


def _redund_ply(*, has_top5_alt: bool):
    """A goal_completion_diagnostics entry classified as redundant_reinforcement."""
    ranking = {"any_in_visit_top5": has_top5_alt}
    return {
        "ply": 50, "side_to_move": "red",
        "goal_completion": {"total_goal_distance_before": 2, "category": "one_endpoint_distance_2"},
        "endpoint_completion_ranking": ranking,
        "distance_reducing_ranking": ranking,
        "selected_move": [10, 10],
        "selected_move_classification": {"primary_class": "redundant_reinforcement"},
    }


# ---------------------------------------------------------------------------
# matches_long_tail_filter
# ---------------------------------------------------------------------------

def test_matches_long_tail_filter_delay_threshold():
    assert matches_long_tail_filter(_rec(conversion_delay_plies=19)) is False
    assert matches_long_tail_filter(_rec(conversion_delay_plies=20)) is True


def test_matches_long_tail_filter_state_cap():
    assert matches_long_tail_filter(_rec(reason="state_cap")) is True


def test_matches_long_tail_filter_dom_unavail_threshold():
    assert matches_long_tail_filter(_rec(winner_moves_with_dominant_unavailable=19)) is False
    assert matches_long_tail_filter(_rec(winner_moves_with_dominant_unavailable=20)) is True


def test_matches_long_tail_filter_n_moves_280():
    assert matches_long_tail_filter(_rec(n_moves=279)) is False
    assert matches_long_tail_filter(_rec(n_moves=280)) is True


# ---------------------------------------------------------------------------
# classify_long_tail_bucket — single-bucket cases
# ---------------------------------------------------------------------------

def test_bucket_marathon_or_state_cap_state_cap_reason():
    rec = _rec(reason="state_cap", n_moves=280, winner=None)
    assert classify_long_tail_bucket(rec, []) == "marathon_or_state_cap"


def test_bucket_marathon_or_state_cap_280_ply_with_winner():
    rec = _rec(reason="win", n_moves=280, conversion_delay_plies=210)
    assert classify_long_tail_bucket(rec, []) == "marathon_or_state_cap"


def test_bucket_dominant_unavailable_contested_threshold():
    rec_in = _rec(winner_moves_with_dominant_unavailable=10, conversion_delay_plies=25)
    assert classify_long_tail_bucket(rec_in, []) == "dominant_unavailable_contested"
    rec_out = _rec(winner_moves_with_dominant_unavailable=9, conversion_delay_plies=30)
    # Not marathon (n_moves=80), dom_un=9 below threshold, so falls through to td2 buckets.
    assert classify_long_tail_bucket(rec_out, []) != "dominant_unavailable_contested"


def test_bucket_td3_drift():
    rec = _rec(first_total_goal_distance=3, conversion_delay_plies=25)
    assert classify_long_tail_bucket(rec, []) == "td3_drift"


def test_bucket_td2_alt_in_top5_majority_have_top5_alt():
    rec = _rec(first_total_goal_distance=2, conversion_delay_plies=25)
    diagnostics = [_redund_ply(has_top5_alt=True) for _ in range(3)]
    assert classify_long_tail_bucket(rec, diagnostics) == "td2_alt_in_top5"


def test_bucket_td2_reducer_buried_majority_have_no_top5_alt():
    rec = _rec(first_total_goal_distance=2, conversion_delay_plies=25)
    diagnostics = [_redund_ply(has_top5_alt=False) for _ in range(3)]
    assert classify_long_tail_bucket(rec, diagnostics) == "td2_reducer_buried"


def test_bucket_td2_exactly_50_percent_alt_goes_to_top5():
    """Boundary: 2 of 4 redundant picks have top-5 alt — uses >=, so bucket 4."""
    rec = _rec(first_total_goal_distance=2, conversion_delay_plies=25)
    diagnostics = [
        _redund_ply(has_top5_alt=True),
        _redund_ply(has_top5_alt=True),
        _redund_ply(has_top5_alt=False),
        _redund_ply(has_top5_alt=False),
    ]
    assert classify_long_tail_bucket(rec, diagnostics) == "td2_alt_in_top5"


# ---------------------------------------------------------------------------
# Priority order
# ---------------------------------------------------------------------------

def test_priority_marathon_over_contested():
    """State-cap + high dom_un → marathon wins."""
    rec = _rec(reason="state_cap", n_moves=280, winner=None,
               winner_moves_with_dominant_unavailable=50)
    assert classify_long_tail_bucket(rec, []) == "marathon_or_state_cap"


def test_priority_contested_over_td_buckets():
    """td=2 game with dom_un=20 → contested wins, not td2_alt_in_top5."""
    rec = _rec(first_total_goal_distance=2, conversion_delay_plies=25,
               winner_moves_with_dominant_unavailable=20)
    diagnostics = [_redund_ply(has_top5_alt=True) for _ in range(3)]
    assert classify_long_tail_bucket(rec, diagnostics) == "dominant_unavailable_contested"


def test_unclassified_defensive_fallback():
    """A long-tail game (delay>=20) with first_td=1 and no diagnostics
    doesn't match any of the five buckets — falls through to unclassified."""
    rec = _rec(first_total_goal_distance=1, conversion_delay_plies=25)
    # No marathon, no contested, not td>=3, not td==2. Falls through.
    assert classify_long_tail_bucket(rec, []) == "unclassified"


def test_not_long_tail_returns_sentinel():
    """A game that doesn't match the long-tail filter returns NOT_LONG_TAIL."""
    rec = _rec(conversion_delay_plies=5)  # below threshold
    assert classify_long_tail_bucket(rec, []) == NOT_LONG_TAIL


# ---------------------------------------------------------------------------
# aggregate_long_tail_buckets
# ---------------------------------------------------------------------------

def test_aggregate_long_tail_buckets_per_iter_and_range_totals():
    """3 long-tail games across 2 iters → correct per-iter and range totals."""
    records = [
        (_rec(iteration=200, reason="state_cap", n_moves=280, winner=None), []),
        (_rec(iteration=200, first_total_goal_distance=2, conversion_delay_plies=25),
         [_redund_ply(has_top5_alt=True) for _ in range(3)]),
        (_rec(iteration=201, first_total_goal_distance=3, conversion_delay_plies=25), []),
        (_rec(iteration=201, conversion_delay_plies=5), []),  # not long-tail, excluded
    ]
    out = aggregate_long_tail_buckets(records)
    assert out["per_iter"][200]["marathon_or_state_cap"] == 1
    assert out["per_iter"][200]["td2_alt_in_top5"] == 1
    assert out["per_iter"][201]["td3_drift"] == 1
    assert out["total_long_tail_games_per_iter"][200] == 2
    assert out["total_long_tail_games_per_iter"][201] == 1
    assert out["total_long_tail_games_range"] == 3
    assert out["range_total"]["marathon_or_state_cap"] == 1
    assert out["range_total"]["td2_alt_in_top5"] == 1
    assert out["range_total"]["td3_drift"] == 1


def test_aggregate_long_tail_buckets_share_computation():
    """Per-iter share = bucket_count / total_long_tail rounded to 3 places."""
    records = [
        (_rec(iteration=200, reason="state_cap", n_moves=280, winner=None), []),
        (_rec(iteration=200, reason="state_cap", n_moves=280, winner=None), []),
        (_rec(iteration=200, first_total_goal_distance=3, conversion_delay_plies=25), []),
    ]
    out = aggregate_long_tail_buckets(records)
    shares = out["per_iter_shares"][200]
    assert abs(shares["marathon_or_state_cap"] - 0.667) < 1e-9
    assert abs(shares["td3_drift"] - 0.333) < 1e-9
    # Range-level shares.
    range_shares = out["range_total_shares"]
    assert abs(range_shares["marathon_or_state_cap"] - 0.667) < 1e-9
