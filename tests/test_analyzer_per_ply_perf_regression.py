"""Generous wall-clock perf guard for the default analyzer path.

This test is a secondary smoke guard — the structural test in
test_analyzer_goal_completion_records.py is the primary regression
guard. Both must pass for the perf fix to remain stable.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_fixture_replay(iteration: int, game_idx: int) -> dict:
    """Tiny in-memory replay carrying a goal_completion_record."""
    return {
        "iteration": iteration,
        "game_idx": game_idx,
        "winner": "red",
        "starting_player": "red",
        "moves": [{"player": "red", "row": r, "col": c, "turn": i + 1}
                  for i, (r, c) in enumerate([(0, 0), (1, 1), (2, 2)])],
        "meta": {"reason": "win", "n_moves": 3, "board_size": 24,
                 "starting_player": "red"},
        "goal_completion_record": {
            "version": 1,
            "outcome_class": 1,
            "iteration": iteration, "game_idx": game_idx,
            "game_id": f"iter_{iteration:04d}_game_{game_idx:03d}",
            "winner": "red", "detected_player": "red",
            "starting_player": "red",
            "n_moves": 3, "reason": "win", "scope": "winner",
            "ever_distance_le_2": True, "ever_distance_le_3": True,
            "min_total_goal_distance": 2,
            "detected": True,
            "first_dominant_unclosed_ply": 1,
            "first_total_goal_distance": 2,
            "first_category": "two_endpoint_closeout_2ply",
            "actual_terminal_ply": 3, "actual_win_ply": 3,
            "conversion_delay_plies": 2, "conversion_delay_winner_moves": 1,
            "cap_delay_proxy_plies": None,
            "primary_class_counts": {
                "completes_endpoint": 1, "reduces_total_goal_distance": 0,
                "redundant_reinforcement": 0, "off_chain": 0, "other": 0,
            },
            "max_search_score_after_detection": 0.99,
            "mean_search_score_after_detection": 0.99,
            "high_value_after_detection_plies": 1,
            "root_value_high_but_delayed": False,
            "search_score_coverage_in_watch_window": 1,
            "winner_moves_in_watch_window": 1,
            "winner_moves_with_dominant_component": 1,
            "winner_moves_with_dominant_unavailable": 0,
            "first_largest_component_size": 8,
            "first_endpoint_distances": {"top": 0, "bottom": 1},
        },
    }


def test_default_path_under_5s_for_50_fixture_games():
    """Guard against re-introducing per-ply BFS in the default path."""
    from scripts.twixt_replay_analyzer import (
        aggregate_goal_completion_diagnostics_from_records,
        write_goal_completion_worst_cases_csv,
    )
    replays = [_make_fixture_replay(110, i) for i in range(50)]
    t0 = time.perf_counter()
    summary = aggregate_goal_completion_diagnostics_from_records(
        replays, sidecar_summaries={}, config={},
    )
    with tempfile.TemporaryDirectory() as tmp:
        write_goal_completion_worst_cases_csv(
            str(Path(tmp) / "worst.csv"),
            replays, top_k=10,
        )
    elapsed = time.perf_counter() - t0
    assert summary["main_population"]["n"] == 50
    assert elapsed < 5.0, f"Default path took {elapsed:.2f}s on 50 games"
