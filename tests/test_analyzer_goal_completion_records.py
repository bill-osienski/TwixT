"""Analyzer record-consumption default path (spec §11)."""
import csv
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _replay_with_record(*, iteration, game_idx, winner, outcome_class,
                       conversion_delay_plies=None, cap_delay_proxy_plies=None,
                       detected=True, primary_class_counts=None, **rec_overrides):
    """Build a minimal in-memory replay dict carrying a goal_completion_record."""
    record = {
        "version": 1,
        "outcome_class": outcome_class,
        "iteration": iteration,
        "game_idx": game_idx,
        "game_id": f"iter_{iteration:04d}_game_{game_idx:03d}",
        "winner": winner,
        "detected_player": winner if outcome_class == 1 else "red",
        "starting_player": "red",
        "n_moves": 21,
        "reason": "win" if outcome_class == 1 else "state_cap",
        "scope": "winner" if outcome_class == 1 else "both_sides",
        "ever_distance_le_2": True,
        "ever_distance_le_3": True,
        "min_total_goal_distance": 2,
        "detected": detected,
        "first_dominant_unclosed_ply": 11,
        "first_total_goal_distance": 2,
        "first_category": "two_endpoint_closeout_2ply",
        "actual_terminal_ply": 21,
        "actual_win_ply": 21 if outcome_class == 1 else None,
        "conversion_delay_plies": conversion_delay_plies,
        "conversion_delay_winner_moves": (conversion_delay_plies // 2 if conversion_delay_plies else None),
        "cap_delay_proxy_plies": cap_delay_proxy_plies,
        "primary_class_counts": primary_class_counts or (
            {"completes_endpoint": 1, "reduces_total_goal_distance": 0,
             "redundant_reinforcement": 3, "off_chain": 1, "other": 0}
            if outcome_class == 1 else None
        ),
        "max_search_score_after_detection": 0.99 if outcome_class == 1 else None,
        "mean_search_score_after_detection": 0.95 if outcome_class == 1 else None,
        "high_value_after_detection_plies": 4 if outcome_class == 1 else None,
        "root_value_high_but_delayed": False,
        "search_score_coverage_in_watch_window": 5 if outcome_class == 1 else None,
        "winner_moves_in_watch_window": 5 if outcome_class == 1 else None,
        "winner_moves_with_dominant_component": 5 if outcome_class == 1 else None,
        "winner_moves_with_dominant_unavailable": 0 if outcome_class == 1 else None,
        "first_largest_component_size": 12,
        "first_endpoint_distances": {"top": 0, "bottom": 1},
    }
    record.update(rec_overrides)
    return {
        "iteration": iteration,
        "game_idx": game_idx,
        "winner": winner,
        "starting_player": "red",
        "moves": [{"player": "red", "row": r, "col": c, "turn": i + 1}
                  for i, (r, c) in enumerate([(0, 0)])],
        "meta": {"reason": "win" if outcome_class == 1 else "state_cap",
                 "n_moves": 21, "board_size": 24,
                 "starting_player": "red"},
        "goal_completion_record": record,
    }


def test_analyzer_default_path_uses_records_no_recompute():
    """Default path consumes per-game records via the shared aggregator."""
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics_from_records,
    )
    replays = [
        _replay_with_record(iteration=110, game_idx=i, winner="red",
                            outcome_class=1, conversion_delay_plies=10)
        for i in range(3)
    ]
    summary = aggregate_goal_completion_diagnostics_from_records(
        replays,
        sidecar_summaries={},
        config={"detection_threshold": 2},
    )
    assert summary["main_population"]["n"] == 3
    assert summary["diagnostics_coverage"]["games_with_record"] == 3
    assert summary["diagnostics_coverage"]["coverage_rate"] == 1.0


def test_worst_cases_csv_from_records_class1():
    from scripts.twixt_replay_analyzer import write_goal_completion_worst_cases_csv

    replays = [
        _replay_with_record(iteration=110, game_idx=i, winner="red",
                            outcome_class=1, conversion_delay_plies=d)
        for i, d in enumerate([3, 22, 8, 28, 14])
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "worst.csv"
        write_goal_completion_worst_cases_csv(
            str(out_path), replays, top_k=3,
        )
        with open(out_path) as f:
            rows = list(csv.DictReader(f))
    assert len(rows) == 3
    delays = [int(r["conversion_delay_plies"]) for r in rows]
    assert delays == [28, 22, 14]


def test_worst_cases_csv_mixed_class1_class2_unified_sort():
    from scripts.twixt_replay_analyzer import write_goal_completion_worst_cases_csv

    replays = [
        _replay_with_record(iteration=110, game_idx=0, winner="red",
                            outcome_class=1, conversion_delay_plies=10),
        _replay_with_record(iteration=110, game_idx=1, winner=None,
                            outcome_class=2, cap_delay_proxy_plies=50),
        _replay_with_record(iteration=110, game_idx=2, winner="black",
                            outcome_class=1, conversion_delay_plies=30),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "worst.csv"
        write_goal_completion_worst_cases_csv(
            str(out_path), replays, top_k=3,
        )
        with open(out_path) as f:
            rows = list(csv.DictReader(f))
    # Sort by delay/proxy descending: 50 (Class 2) > 30 (Class 1) > 10 (Class 1).
    ranked = [r["scope"] for r in rows]
    assert ranked[0] == "both_sides"   # Class 2 first


def test_worst_cases_csv_skips_replays_without_record():
    from scripts.twixt_replay_analyzer import write_goal_completion_worst_cases_csv

    r_with = _replay_with_record(iteration=110, game_idx=0, winner="red",
                                 outcome_class=1, conversion_delay_plies=20)
    r_without = {"iteration": 110, "game_idx": 1, "winner": "red",
                 "starting_player": "red", "moves": [],
                 "meta": {"reason": "win", "n_moves": 0, "board_size": 24}}
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "worst.csv"
        write_goal_completion_worst_cases_csv(
            str(out_path), [r_with, r_without], top_k=5,
        )
        with open(out_path) as f:
            rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["game_idx"] == "0"


def test_analyzer_missing_record_warning_aggregated(capsys):
    """One summary warning per missing-record bucket, with up to 3 examples."""
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics_from_records,
    )
    replays = []
    # 5 missing replays.
    for i in range(5):
        replays.append({
            "iteration": 110, "game_idx": i, "winner": "red",
            "starting_player": "red", "moves": [],
            "meta": {"n_moves": 0, "board_size": 24},
        })
    # 1 with record.
    replays.append(_replay_with_record(iteration=110, game_idx=99, winner="red",
                                       outcome_class=1, conversion_delay_plies=10))

    aggregate_goal_completion_diagnostics_from_records(
        replays, sidecar_summaries={}, config={},
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "5/6" in out
    assert "missing goal_completion_record" in out
    assert "Examples:" in out


def test_analyzer_all_missing_warning(capsys):
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics_from_records,
    )
    replays = [
        {"iteration": 110, "game_idx": i, "winner": "red",
         "starting_player": "red", "moves": [],
         "meta": {"n_moves": 0, "board_size": 24}}
        for i in range(3)
    ]
    aggregate_goal_completion_diagnostics_from_records(
        replays, sidecar_summaries={}, config={},
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "3/3" in out
    assert "Goal-completion report skipped" in out


def test_analyzer_sidecar_mismatch_warning(capsys):
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics_from_records,
    )
    replays = [
        _replay_with_record(iteration=110, game_idx=0, winner="red",
                            outcome_class=1, conversion_delay_plies=10),
    ]
    sidecar_summaries = {
        110: {"diagnostics_coverage": {"games_with_record": 100,
                                       "games_total": 100,
                                       "coverage_rate": 1.0,
                                       "games_class_1": 100,
                                       "games_class_2": 0,
                                       "games_class_3": 0}}
    }
    aggregate_goal_completion_diagnostics_from_records(
        replays, sidecar_summaries=sidecar_summaries, config={},
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "sidecar/replay mismatch" in out
    assert "iter 0110" in out


def test_analyzer_version_mismatch_warning(capsys):
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics_from_records,
    )
    replays = [
        _replay_with_record(iteration=110, game_idx=0, winner="red",
                            outcome_class=1, conversion_delay_plies=10),
    ]
    sidecar_summaries = {
        110: {
            "version": 2,
            "diagnostics_coverage": {"games_with_record": 1, "games_total": 1,
                                     "coverage_rate": 1.0,
                                     "games_class_1": 1, "games_class_2": 0,
                                     "games_class_3": 0},
        }
    }
    aggregate_goal_completion_diagnostics_from_records(
        replays, sidecar_summaries=sidecar_summaries, config={},
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "version mismatch" in out
    assert "records canonical" in out


def test_analyzer_default_path_does_not_recompute_goal_completion():
    """ANCHOR: Structural guard — default path must not call BFS helpers
    on the analyzer side. Monkeypatch and assert zero calls."""
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics_from_records,
    )
    replays = [
        _replay_with_record(iteration=110, game_idx=i, winner="red",
                            outcome_class=1, conversion_delay_plies=10)
        for i in range(5)
    ]

    with patch(
        "scripts.GPU.alphazero.connectivity_diagnostics.compute_goal_completion_state",
    ) as mock_compute, patch(
        "scripts.twixt_replay_analyzer._build_class1_per_game_record",
    ) as mock_build1, patch(
        "scripts.twixt_replay_analyzer._build_class2_per_game_record",
    ) as mock_build2:
        aggregate_goal_completion_diagnostics_from_records(
            replays, sidecar_summaries={}, config={},
        )

    assert mock_compute.call_count == 0, \
        "Default analyzer path must not call compute_goal_completion_state"
    assert mock_build1.call_count == 0, \
        "Default analyzer path must not call _build_class1_per_game_record"
    assert mock_build2.call_count == 0, \
        "Default analyzer path must not call _build_class2_per_game_record"
