"""Aggregator unit tests (spec 2026-05-05 §7)."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.goal_completion_aggregator import (
    aggregate_goal_completion_records,
    _normalize_record,
    _zero_class_counts,
)


def _decisive_record(**overrides):
    base = {
        "version": 1,
        "outcome_class": 1,
        "winner": "red",
        "detected_player": "red",
        "reason": "win",
        "detected": True,
        "ever_distance_le_2": True,
        "ever_distance_le_3": True,
        "min_total_goal_distance": 2,
        "first_dominant_unclosed_ply": 11,
        "first_total_goal_distance": 2,
        "first_category": "two_endpoint_closeout_2ply",
        "actual_terminal_ply": 21,
        "actual_win_ply": 21,
        "conversion_delay_plies": 10,
        "conversion_delay_winner_moves": 5,
        "cap_delay_proxy_plies": None,
        "primary_class_counts": {
            "completes_endpoint": 1,
            "reduces_total_goal_distance": 0,
            "redundant_reinforcement": 3,
            "off_chain": 1,
            "other": 0,
        },
        "max_search_score_after_detection": 0.99,
        "mean_search_score_after_detection": 0.95,
        "high_value_after_detection_plies": 4,
        "root_value_high_but_delayed": True,
        "search_score_coverage_in_watch_window": 5,
        "winner_moves_in_watch_window": 5,
        "winner_moves_with_dominant_component": 5,
        "winner_moves_with_dominant_unavailable": 0,
    }
    base.update(overrides)
    return base


def test_aggregator_empty_records_returns_skeleton():
    result = aggregate_goal_completion_records([], config={"detection_threshold": 2}, games_total=0)
    assert result["version"] == 1
    assert result["config"] == {"detection_threshold": 2}
    assert result["diagnostics_coverage"] == {
        "games_total": 0,
        "games_with_record": 0,
        "coverage_rate": 0.0,
        "games_class_1": 0,
        "games_class_2": 0,
        "games_class_3": 0,
    }
    assert result["main_population"]["n"] == 0
    assert result["capped_population"]["n"] == 0
    assert result["excluded_population"] == {"n": 0}


def test_aggregator_mixed_nones_real_coverage():
    """coverage_rate uses games_total (caller-supplied), not len(valid)."""
    rec = _decisive_record()
    result = aggregate_goal_completion_records(
        [rec, None, rec, None],
        config={"detection_threshold": 2},
        games_total=4,
    )
    assert result["diagnostics_coverage"]["games_total"] == 4
    assert result["diagnostics_coverage"]["games_with_record"] == 2
    assert result["diagnostics_coverage"]["coverage_rate"] == 0.5
    assert result["diagnostics_coverage"]["games_class_1"] == 2


def test_aggregator_default_games_total_is_record_count():
    rec = _decisive_record()
    result = aggregate_goal_completion_records([rec], config={})
    assert result["diagnostics_coverage"]["games_total"] == 1
    assert result["diagnostics_coverage"]["coverage_rate"] == 1.0


def test_aggregator_class_split_counts():
    cap = _decisive_record(
        outcome_class=2, winner=None, reason="state_cap",
        actual_win_ply=None, conversion_delay_plies=None,
        conversion_delay_winner_moves=None,
        cap_delay_proxy_plies=42,
        primary_class_counts=None,
        max_search_score_after_detection=None,
        mean_search_score_after_detection=None,
        high_value_after_detection_plies=None,
        root_value_high_but_delayed=None,
        search_score_coverage_in_watch_window=None,
        winner_moves_in_watch_window=None,
        winner_moves_with_dominant_component=None,
        winner_moves_with_dominant_unavailable=None,
    )
    excl = {"version": 1, "outcome_class": 3, "winner": None,
            "detected": False, "reason": "unknown"}
    decisive = _decisive_record()
    result = aggregate_goal_completion_records(
        [decisive, decisive, cap, excl],
        config={}, games_total=4,
    )
    assert result["diagnostics_coverage"]["games_class_1"] == 2
    assert result["diagnostics_coverage"]["games_class_2"] == 1
    assert result["diagnostics_coverage"]["games_class_3"] == 1


def test_aggregator_zero_games_total_zero_rate():
    result = aggregate_goal_completion_records([None, None], config={}, games_total=0)
    assert result["diagnostics_coverage"]["coverage_rate"] == 0.0


def test_normalize_record_fills_defaults():
    out = _normalize_record({"version": 1, "outcome_class": 1, "winner": "red"})
    assert out["version"] == 1
    assert out["outcome_class"] == 1
    assert out["winner"] == "red"
    assert out["detected"] is False  # default
    assert out["primary_class_counts"] == _zero_class_counts()
    assert out["reason"] == "unknown"
    assert out["min_total_goal_distance"] is None


def test_normalize_record_handles_unknown_version_defensively():
    out = _normalize_record({"version": 99, "outcome_class": 1, "winner": "red"})
    assert out["version"] == 99  # passes through; aggregator may warn
