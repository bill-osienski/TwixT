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
