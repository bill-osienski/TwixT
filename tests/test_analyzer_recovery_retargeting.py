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


import csv

from scripts.twixt_replay_analyzer import write_recovery_retargeting_by_iter_csv


def test_by_iter_csv_one_row_per_iter(tmp_path):
    per_iter = {
        170: _summary(games_total=100, games_triggered=14),
        171: _summary(games_total=100, games_triggered=20),
    }
    out = tmp_path / "recovery_retargeting_by_iter.csv"
    write_recovery_retargeting_by_iter_csv(str(out), per_iter)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    assert int(rows[0]["iteration"]) == 170
    assert int(rows[0]["games_triggered"]) == 14
    assert int(rows[1]["games_triggered"]) == 20
    assert "local_drift_rate" in rows[0]


from scripts.twixt_replay_analyzer import write_recovery_retargeting_worst_cases_csv


def _per_game_rec(iteration, game_idx, sides_triggered, local_drift_moves, in_window):
    side_records = {"red": {"triggered": False}, "black": {"triggered": False}}
    for side in sides_triggered:
        side_records[side] = {
            "triggered": True,
            "first_trigger_ply": 44, "first_trigger_reason": "steady_state",
            "in_window_own_moves": in_window, "triggered_own_moves": in_window,
            "severe_collapse_moves": 0, "very_diffuse_moves": 0,
            "classified_in_window_moves": in_window, "missing_signal_moves": 0,
            "selected_class_counts": {
                "blocks_opponent_closeout": 0, "reduces_own_goal_distance": 0,
                "starts_or_extends_alternate_component": 0,
                "connects_to_existing_component": 0, "improves_own_largest_component": 0,
                "redundant_local_reinforcement": local_drift_moves,
                "off_plan_or_unclear": 0,
            },
            "constructive_recovery_moves": 0, "defensive_moves": 0,
            "structural_connection_moves": 0, "local_drift_moves": local_drift_moves,
            "local_drift_rate": 1.0, "constructive_recovery_rate": 0.0,
            "mean_search_score_triggered_plies": -0.85,
            "min_search_score_triggered_plies": -0.99,
            "max_search_score_triggered_plies": -0.75,
            "mean_root_top1_share_triggered_plies": 0.12,
        }
    return {
        "iteration": iteration, "game_idx": game_idx, "game_id": f"game_{game_idx:03d}",
        "winner": "red", "loser": "black", "n_moves": 65, "reason": "win",
        "triggered_sides": sides_triggered, "side_records": side_records,
    }


def test_worst_cases_csv_sort_order_and_topk(tmp_path):
    out = tmp_path / "recovery_retargeting_worst_cases.csv"
    records = [
        _per_game_rec(170, 0, ["black"], local_drift_moves=2, in_window=2),
        _per_game_rec(170, 1, ["black"], local_drift_moves=15, in_window=15),
        _per_game_rec(170, 2, ["black"], local_drift_moves=8, in_window=8),
    ]
    write_recovery_retargeting_worst_cases_csv(str(out), records, top_k=2)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    assert int(rows[0]["local_drift_moves"]) == 15
    assert int(rows[1]["local_drift_moves"]) == 8


def test_worst_cases_csv_two_rows_for_dual_triggered_game(tmp_path):
    out = tmp_path / "recovery_retargeting_worst_cases.csv"
    records = [_per_game_rec(170, 0, ["black", "red"], local_drift_moves=5, in_window=5)]
    write_recovery_retargeting_worst_cases_csv(str(out), records, top_k=25)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    sides = sorted(r["triggered_side"] for r in rows)
    assert sides == ["black", "red"]


# Spec 2026-05-13 — filtered side-split view tests.

def _empty_split_bucket(*, sides=0, in_window=0, triggered=0, mean_score=None,
                        constructive=0.0, defensive=0.0, structural=0.0, local=0.0):
    return {
        "sides": sides,
        "in_window_own_moves_total": in_window,
        "triggered_own_moves_total": triggered,
        "non_triggered_in_window_moves_total": in_window - triggered,
        "missing_signal_moves_total": 0,
        "severe_collapse_moves_total": triggered // 2,
        "very_diffuse_moves_total": triggered,
        "classified_in_window_moves_total": triggered,
        "selected_class_counts_total": {
            "blocks_opponent_closeout": 0, "reduces_own_goal_distance": 0,
            "starts_or_extends_alternate_component": 0,
            "connects_to_existing_component": 0, "improves_own_largest_component": 0,
            "redundant_local_reinforcement": 0, "off_plan_or_unclear": 0,
        },
        "selected_class_rates_total": {
            "blocks_opponent_closeout": 0.0, "reduces_own_goal_distance": 0.0,
            "starts_or_extends_alternate_component": 0.0,
            "connects_to_existing_component": 0.0, "improves_own_largest_component": 0.0,
            "redundant_local_reinforcement": 0.0, "off_plan_or_unclear": 0.0,
        },
        "trigger_reason_counts_total": {"delta_precursor": 0, "steady_state": triggered, "both": 0},
        "constructive_recovery_rate": constructive,
        "defensive_rate":             defensive,
        "structural_connection_rate": structural,
        "local_drift_rate":           local,
        "mean_search_score_triggered_plies":    mean_score,
        "min_search_score_triggered_plies":     (mean_score - 0.05) if mean_score is not None else None,
        "max_search_score_triggered_plies":     (mean_score + 0.05) if mean_score is not None else None,
        "mean_root_top1_share_triggered_plies": 0.15 if mean_score is not None else None,
    }


def _split_summary(**overrides):
    """Minimal raw_side_split summary fixture."""
    base = {
        "version": 1,
        "view": "raw_side_split",
        "enabled": True,
        "config": _summary()["config"],
        "games_total": 1000, "games_triggered": 1000,
        "eventual_loser":    _empty_split_bucket(sides=989, in_window=10000, triggered=8000,
                                                 mean_score=-0.92, constructive=0.20, defensive=0.01,
                                                 structural=0.55, local=0.24),
        "eventual_winner":   _empty_split_bucket(sides=342, in_window=2500, triggered=1500,
                                                 mean_score=-0.78, constructive=0.30, defensive=0.00,
                                                 structural=0.50, local=0.20),
        "state_cap_or_draw": _empty_split_bucket(sides=8, in_window=200, triggered=120,
                                                 mean_score=-0.93, constructive=0.18, defensive=0.02,
                                                 structural=0.55, local=0.25),
        "schema_integrity": {
            "skipped_unknown_version_count": 0,
            "skipped_config_mismatch_count": 0,
            "classifier_error_count_total": 0,
        },
    }
    base.update(overrides)
    return base


def _filtered_summary(**overrides):
    """Minimal filtered_actionable_collapse summary fixture."""
    base = _split_summary()
    base["view"] = "filtered_actionable_collapse"
    base["filter_summary"] = {
        "filter_config": {
            "min_in_window_own_moves": 20,
            "min_triggered_own_moves": 3,
            "max_mean_search_score_triggered_plies": -0.85,
            "max_constructive_recovery_rate": 0.30,
            "min_structural_plus_local_rate": 0.60,
        },
        "side_views_total":  1339,
        "side_views_passed": 412,
        "side_views_failed": 927,
        "failed_reason_counts": {
            "in_window_below_min":             120,
            "triggered_below_min":             210,
            "mean_score_above_max":            300,
            "constructive_recovery_above_max": 250,
            "structural_plus_local_below_min": 90,
        },
    }
    base.update(overrides)
    return base


def test_format_report_renders_three_sections():
    pooled = _summary()
    split = _split_summary()
    filtered = _filtered_summary()
    lines = format_recovery_retargeting_report(pooled, split, filtered)
    text = "\n".join(lines)
    assert "Raw side-outcome split" in text
    assert "Filtered actionable-collapse view" in text
    assert "Filter summary" in text
    assert "eventual_loser" in text
    assert "eventual_winner" in text
    assert "state_cap_or_draw" in text


def test_format_report_backward_compat_pooled_only():
    """When split and filtered are None, render only the existing pooled section."""
    pooled = _summary()
    lines = format_recovery_retargeting_report(pooled, None, None)
    text = "\n".join(lines)
    assert "Recovery / Re-targeting Diagnostics" in text
    assert "Raw side-outcome split" not in text
    assert "Filtered actionable-collapse view" not in text
