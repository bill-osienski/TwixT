"""Tests for Spec 4 analyzer surface."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import format_recovery_retargeting_report


def _summary(**overrides):
    base = {
        "version": 1,
        "enabled": True,
        "config": {
            "collapse_value_threshold": -0.75,
            "severe_collapse_value_threshold": -0.90,
            "diffuse_root_top1_threshold": 0.20,
            "very_diffuse_root_top1_threshold": 0.15,
            "delta_threshold": 0.50,
            "delta_max_current_score": -0.30,
            "alternate_component_min_size": 4,
            "classify_defense": True,
        },
        "games_total": 1000, "games_triggered": 143,
        "trigger_rate": 0.143,
        "triggered_loser_side": 136, "triggered_winner_side": 9,
        "triggered_loser_side_per_triggered_game": 0.951,
        "triggered_winner_side_per_triggered_game": 0.063,
        "in_window_own_moves_total": 1284,
        "triggered_own_moves_total": 1108,
        "non_triggered_in_window_moves_total": 176,
        "missing_signal_moves_total": 0,
        "severe_collapse_moves_total": 522,
        "very_diffuse_moves_total": 914,
        "trigger_reason_counts_total": {"delta_precursor": 177, "steady_state": 859, "both": 72},
        "classified_in_window_moves_total": 1284,
        "selected_class_counts_total": {
            "blocks_opponent_closeout": 104, "reduces_own_goal_distance": 55,
            "starts_or_extends_alternate_component": 41,
            "connects_to_existing_component": 231, "improves_own_largest_component": 159,
            "redundant_local_reinforcement": 548, "off_plan_or_unclear": 146,
        },
        "constructive_recovery_rate": 0.075,
        "defensive_rate": 0.081,
        "structural_connection_rate": 0.304,
        "local_drift_rate": 0.540,
        "schema_integrity": {
            "skipped_unknown_version_count": 0,
            "skipped_config_mismatch_count": 0,
            "classifier_error_count_total": 0,
        },
        "iters_covered": [170, 179],
    }
    base.update(overrides)
    return base


def test_format_emits_section_header_and_key_lines():
    lines = format_recovery_retargeting_report(_summary())
    body = "\n".join(lines)
    assert "Recovery / Re-targeting Diagnostics" in body
    assert "Triggered games:" in body
    assert "constructive recovery:" in body
    assert "local drift / unclear:" in body


def test_format_warns_when_classify_defense_off():
    s = _summary()
    s["config"] = dict(s["config"])
    s["config"]["classify_defense"] = False
    body = "\n".join(format_recovery_retargeting_report(s))
    assert "defense classification disabled" in body


def test_format_returns_empty_when_summary_is_none_or_empty():
    assert format_recovery_retargeting_report(None) == []
    assert format_recovery_retargeting_report({}) == []
