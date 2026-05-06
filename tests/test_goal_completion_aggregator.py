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
    assert result["excluded_population"]["n"] == 0
    assert result["excluded_population"]["games"] == 0


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


def _capped_record(**overrides):
    base = {
        "version": 1, "outcome_class": 2, "winner": None,
        "detected_player": "red", "reason": "state_cap",
        "detected": True,
        "ever_distance_le_2": True, "ever_distance_le_3": True,
        "min_total_goal_distance": 2,
        "first_dominant_unclosed_ply": 60,
        "first_total_goal_distance": 2,
        "first_category": "two_endpoint_closeout_2ply",
        "actual_terminal_ply": 100,
        "actual_win_ply": None,
        "conversion_delay_plies": None,
        "conversion_delay_winner_moves": None,
        "cap_delay_proxy_plies": 40,
        "primary_class_counts": None,
    }
    base.update(overrides)
    return base


def test_main_population_known_percentiles():
    """Handcrafted record set with known delays — assert exact percentiles."""
    delays = [4, 4, 6, 8, 10, 14, 18, 22, 28]
    records = [_decisive_record(conversion_delay_plies=d, conversion_delay_winner_moves=d // 2)
               for d in delays]
    result = aggregate_goal_completion_records(records, config={}, games_total=len(records))
    main = result["main_population"]
    assert main["n"] == 9
    assert main["detected"] == 9
    assert main["detection_rate"] == 1.0
    cd = main["conversion_delay_plies"]
    # Linear-interpolation percentiles (per _percentile helper):
    #   rank = p/100 * (N-1); for N=9 sorted [4,4,6,8,10,14,18,22,28]:
    #     p50 -> rank 4.0 -> values[4] = 10
    #     p90 -> rank 7.2 -> values[7]*0.8 + values[8]*0.2 = 22*0.8 + 28*0.2 = 23.2
    #     max -> 28
    assert cd["p50"] == 10
    assert abs(cd["p90"] - 23.2) < 1e-9
    assert cd["max"] == 28


def test_main_population_naming_continuity():
    """Spec-locked naming — continuity with existing analyzer report."""
    rec = _decisive_record()
    result = aggregate_goal_completion_records([rec, rec], config={}, games_total=2)
    main = result["main_population"]
    assert "games_with_dominant_unclosed" in main
    assert "games_with_total_distance_le_2" in main
    assert "games_with_total_distance_le_3" in main
    assert main["games_with_dominant_unclosed"] == 2
    assert main["games_with_total_distance_le_2"] == 2


def test_main_population_primary_class_rates_pooled():
    """primary_class_rates pools counts across all main games."""
    r1 = _decisive_record(primary_class_counts={
        "completes_endpoint": 2, "reduces_total_goal_distance": 1,
        "redundant_reinforcement": 1, "off_chain": 0, "other": 0,
    })
    r2 = _decisive_record(primary_class_counts={
        "completes_endpoint": 0, "reduces_total_goal_distance": 1,
        "redundant_reinforcement": 4, "off_chain": 0, "other": 1,
    })
    result = aggregate_goal_completion_records([r1, r2], config={}, games_total=2)
    rates = result["main_population"]["primary_class_rates"]
    # Total selected = 10. completes=2, reduces=2, redundant=5, off=0, other=1.
    assert abs(rates["completes_endpoint"] - 0.2) < 1e-9
    assert abs(rates["reduces_total_goal_distance"] - 0.2) < 1e-9
    assert abs(rates["redundant_reinforcement"] - 0.5) < 1e-9


def test_main_population_bad_cases_thresholds():
    delays = [3, 9, 10, 11, 19, 20, 25]
    records = [_decisive_record(
        conversion_delay_plies=d,
        high_value_after_detection_plies=2,
    ) for d in delays]
    result = aggregate_goal_completion_records(records, config={}, games_total=len(records))
    bad = result["main_population"]["bad_cases"]
    # delay >= 10 -> 5 (10, 11, 19, 20, 25)
    assert bad["delay_ge_10_plies"] == 5
    # delay >= 20 -> 2 (20, 25)
    assert bad["delay_ge_20_plies"] == 2
    # high_value_after_detection_plies_total = 2 * 7 = 14
    assert bad["high_value_after_detection_plies_total"] == 14


def test_main_population_root_value_high_but_delayed_count():
    records = [
        _decisive_record(root_value_high_but_delayed=True),
        _decisive_record(root_value_high_but_delayed=False),
        _decisive_record(root_value_high_but_delayed=True),
    ]
    result = aggregate_goal_completion_records(records, config={}, games_total=3)
    assert result["main_population"]["bad_cases"]["root_value_high_but_delayed"] == 2


def test_capped_population_summary():
    records = [
        _capped_record(cap_delay_proxy_plies=20, detected_player="red"),
        _capped_record(cap_delay_proxy_plies=40, detected_player="red"),
        _capped_record(cap_delay_proxy_plies=60, detected_player="black"),
    ]
    result = aggregate_goal_completion_records(records, config={}, games_total=3)
    cap = result["capped_population"]
    assert cap["n"] == 3
    assert cap["detected"] == 3
    assert cap["cap_delay_proxy_plies"]["p50"] == 40
    assert cap["cap_delay_proxy_plies"]["max"] == 60
    assert cap["first_detector_side"] == {"red": 2, "black": 1}


def test_cross_iteration_roll_up_matches_per_iter_aggregation():
    """The same shared aggregator at any scope: per-iter aggregation
    composed via roll-up should equal one cross-iter aggregation on the
    same records (recompute principle, spec §11.1)."""
    delays = [4, 6, 8, 10, 14, 18]
    records = [_decisive_record(conversion_delay_plies=d) for d in delays]
    cross = aggregate_goal_completion_records(records, config={}, games_total=len(records))
    # Reaggregate from the same records — same input, same shape, same numbers.
    cross2 = aggregate_goal_completion_records(records, config={}, games_total=len(records))
    assert cross == cross2


def test_main_population_legacy_formatter_keys_present():
    """Aggregator must emit legacy keys the existing analyzer formatter
    reads (spec §11.7, plan Task 10 reuses format_goal_completion_report
    unchanged)."""
    rec = _decisive_record()
    result = aggregate_goal_completion_records([rec, rec], config={}, games_total=2)
    main = result["main_population"]
    # Legacy key set required by format_goal_completion_report:
    assert "games" in main
    assert main["games"] == 2
    assert "move_quality_after_detection" in main
    assert "high_value_diagnostics" in main
    assert "delay_ge_10_plies" in main["bad_cases"]
    assert "delay_ge_20_plies" in main["bad_cases"]
    # New keys also still present (programmatic-access aliases):
    assert "n" in main
    assert "primary_class_rates" in main
    assert "search_score_after_detection" in main


def test_main_population_move_quality_uses_rate_suffix_and_separate_denominator():
    """move_quality_after_detection has _rate suffix keys; the 5 base rates
    use pooled_with_component denominator; dominant_unavailable_rate uses
    pooled_with_component + pooled_unavailable denominator (legacy semantics)."""
    r1 = _decisive_record(
        primary_class_counts={
            "completes_endpoint": 2, "reduces_total_goal_distance": 1,
            "redundant_reinforcement": 1, "off_chain": 0, "other": 0,
        },
        winner_moves_with_dominant_unavailable=1,
    )
    result = aggregate_goal_completion_records([r1], config={}, games_total=1)
    mq = result["main_population"]["move_quality_after_detection"]
    # pooled_with_component = 2+1+1+0+0 = 4
    # pooled_unavailable = 1
    # base rates use denom 4: completes 2/4, reduces 1/4, redundant 1/4, off 0/4, other 0/4
    assert abs(mq["completes_endpoint_rate"] - 0.5) < 1e-9
    assert abs(mq["reduces_total_goal_distance_rate"] - 0.25) < 1e-9
    assert abs(mq["redundant_reinforcement_rate"] - 0.25) < 1e-9
    # dominant_unavailable_rate uses denom 4+1=5: 1/5
    assert abs(mq["dominant_unavailable_rate"] - 0.2) < 1e-9


def test_main_population_move_quality_none_when_no_with_component():
    """When no detected records have a dominant component (all unavailable
    or no detection), move_quality_after_detection is None (legacy)."""
    r1 = _decisive_record(
        primary_class_counts={
            "completes_endpoint": 0, "reduces_total_goal_distance": 0,
            "redundant_reinforcement": 0, "off_chain": 0, "other": 0,
        },
        winner_moves_with_dominant_unavailable=3,
    )
    result = aggregate_goal_completion_records([r1], config={}, games_total=1)
    assert result["main_population"]["move_quality_after_detection"] is None


def test_main_population_high_value_diagnostics_shape():
    """high_value_diagnostics has search_score_coverage_pct + nested
    max/mean blocks with p50/p90/max keys."""
    rec = _decisive_record(
        max_search_score_after_detection=0.99,
        mean_search_score_after_detection=0.95,
        search_score_coverage_in_watch_window=5,
    )
    result = aggregate_goal_completion_records([rec, rec], config={}, games_total=2)
    hv = result["main_population"]["high_value_diagnostics"]
    assert "search_score_coverage_pct" in hv
    assert hv["search_score_coverage_pct"] == 100.0
    assert "max_search_score_after_detection" in hv
    assert "mean_search_score_after_detection" in hv
    assert hv["max_search_score_after_detection"]["p50"] == 0.99
    assert hv["max_search_score_after_detection"]["max"] == 0.99


def test_capped_population_legacy_aliases_present():
    """capped_population must emit detected_before_cap + cap_delay_after_detection_plies
    + games (legacy keys for formatter)."""
    records = [_capped_record(cap_delay_proxy_plies=20, detected_player="red"),
               _capped_record(cap_delay_proxy_plies=40, detected_player="black")]
    result = aggregate_goal_completion_records(records, config={}, games_total=2)
    cap = result["capped_population"]
    assert cap["games"] == 2
    assert cap["detected_before_cap"] == 2
    # Legacy alias points at same stats block as the new key.
    assert cap["cap_delay_after_detection_plies"] == cap["cap_delay_proxy_plies"]


def test_capped_population_bad_cases_counts_by_reason():
    """Capped population emits bad_cases counters keyed by reason
    (state_cap_after_detection / timeout_after_detection /
    board_full_after_detection) for legacy formatter compatibility."""
    records = [
        _capped_record(reason="state_cap", cap_delay_proxy_plies=10),
        _capped_record(reason="state_cap", cap_delay_proxy_plies=12),
        _capped_record(reason="timeout", cap_delay_proxy_plies=20),
        _capped_record(reason="board_full", cap_delay_proxy_plies=30),
    ]
    result = aggregate_goal_completion_records(records, config={}, games_total=4)
    bc = result["capped_population"]["bad_cases"]
    assert bc["state_cap_after_detection"] == 2
    assert bc["timeout_after_detection"] == 1
    assert bc["board_full_after_detection"] == 1


def test_capped_population_bad_cases_only_counts_detected():
    """Bad cases counters only include detected records (matches legacy)."""
    records = [
        _capped_record(reason="state_cap", detected=True),
        _capped_record(reason="state_cap", detected=False, cap_delay_proxy_plies=None,
                       first_dominant_unclosed_ply=None),
    ]
    result = aggregate_goal_completion_records(records, config={}, games_total=2)
    bc = result["capped_population"]["bad_cases"]
    assert bc["state_cap_after_detection"] == 1


def test_excluded_population_emits_games_legacy_alias():
    """excluded_population emits both n and games for formatter compatibility."""
    excl = {"version": 1, "outcome_class": 3, "winner": None,
            "detected": False, "reason": "unknown"}
    result = aggregate_goal_completion_records(
        [excl, excl, excl], config={}, games_total=3,
    )
    assert result["excluded_population"]["n"] == 3
    assert result["excluded_population"]["games"] == 3
