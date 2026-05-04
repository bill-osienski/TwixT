"""Tests for analyzer surfacing of inline closeout diagnostics (spec 2026-05-03 §8.6-8.7)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import (
    aggregate_goal_completion_diagnostics,
    format_policy_mcts_closeout_report,
)


def _replay_with_diag(diag_records, meta=None, n_moves=20, reason="win"):
    """Build a replay with a goal_completion_diagnostics array."""
    return {
        "id": "iter_0050_game_001",
        "winner": "red", "starting_player": "red",
        "moves": [{"turn": i+1, "player": "red" if i % 2 == 0 else "black",
                   "row": 0, "col": i, "bridges_created": [], "heuristics": {},
                   "search_score": None, "root_top1_share": None}
                  for i in range(n_moves)],
        "meta": {"board_size": 24, "iteration": 50, "game_idx": 1,
                 "n_moves": n_moves, "reason": reason, "starting_player": "red"},
        "goal_completion_diagnostics": diag_records,
        "goal_completion_diagnostics_meta": meta or {
            "enabled": True, "diagnostic_version": 1, "error_count": 0,
            "resign_dropped_partial_count": 0,
            "skipped_missing_priors_count": 0,
            "records_dropped_by_cap": 0,
        },
    }


def test_aggregate_diagnostics_coverage_counts_games_with_records():
    """Replays with goal_completion_diagnostics array → coverage counts populated."""
    diag1 = [{
        "ply": 10, "side_to_move": "red",
        "goal_completion": {"total_goal_distance_before": 2},
        "endpoint_completion_ranking": {"any_in_policy_top5": True, "any_in_visit_top5": False,
                                         "best_visit_rank": 8, "best_policy_rank": 4},
        "selected_move_classification": {"primary_class": "off_chain"},
    }]
    replays = [_replay_with_diag(diag1), _replay_with_diag([])]
    r = aggregate_goal_completion_diagnostics(replays, min_component_size=1)
    assert r["diagnostics_coverage"]["games_with_diagnostics"] == 1
    assert r["diagnostics_coverage"]["total_records"] == 1
    assert r["diagnostics_coverage"]["error_count"] == 0
    assert r["diagnostics_coverage"]["version"] == 1


def test_aggregate_policy_mcts_summary_pools_records_correctly_by_distance():
    """Records pool into policy_mcts_summary; by_distance buckets le_2 / eq_3 correctly."""
    diag_le2 = {
        "ply": 10, "side_to_move": "red",
        "goal_completion": {"total_goal_distance_before": 2},
        "root_summary": {"q_value": 0.95},
        "endpoint_completion_ranking": {"any_in_policy_top5": True, "any_in_visit_top5": True,
                                         "best_visit_rank": 1, "best_policy_rank": 1},
        "distance_reducing_ranking": {"any_in_policy_top5": True, "any_in_visit_top5": True,
                                       "best_visit_rank": 1, "best_policy_rank": 1},
        "selected_move_classification": {"primary_class": "completes_endpoint"},
    }
    diag_eq3 = dict(diag_le2)
    diag_eq3["goal_completion"] = {"total_goal_distance_before": 3}
    diag_eq3["selected_move_classification"] = {"primary_class": "redundant_reinforcement"}

    replays = [_replay_with_diag([diag_le2, diag_eq3])]
    r = aggregate_goal_completion_diagnostics(replays, min_component_size=1)
    pms = r["policy_mcts_summary"]
    assert pms is not None
    assert pms["n_records"] == 2
    assert pms["by_distance"]["distance_le_2"]["n"] == 1
    assert pms["by_distance"]["distance_eq_3"]["n"] == 1
    assert pms["selected_primary_class_rates"]["completes_endpoint"] == 0.5
    assert pms["selected_primary_class_rates"]["redundant_reinforcement"] == 0.5
    # high_value_delayed: requires q_value >= 0.9 + redundant/off_chain/other primary_class + total <= 2.
    # The le2 record has primary_class=completes_endpoint (excluded).
    # The eq3 record has total=3 (excluded).
    # So neither qualifies → 0.
    assert pms["high_value_delayed_closeouts"] == 0
