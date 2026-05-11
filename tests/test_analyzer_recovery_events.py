"""Tests for Fix 3: recovery event classification (spec 2026-05-10 §6)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import aggregate_recovery_events


def _fixture(rec_overrides=None, meta_overrides=None, diag=None):
    rec = {
        "winner": "black",
        "detected_player": "black",
        "first_dominant_unclosed_ply": 50,
        "actual_terminal_ply": 90,
        "conversion_delay_winner_moves": 15,
        "winner_moves_in_watch_window": 20,
        "winner_moves_with_dominant_unavailable": 12,
    }
    if rec_overrides:
        rec.update(rec_overrides)
    meta = {"reason": "win", "iteration": 130, "game_idx": 1, "final_root_value": 0.95}
    if meta_overrides:
        meta.update(meta_overrides)
    return {
        "goal_completion_record": rec,
        "meta": meta,
        "goal_completion_diagnostics": diag or [],
    }


def test_lost_then_state_cap_classified():
    g = _fixture(rec_overrides={"winner_moves_with_dominant_unavailable": 15},
                 meta_overrides={"reason": "state_cap", "final_root_value": 0.92})
    events = aggregate_recovery_events([g])
    assert len(events) == 1
    assert events[0]["recovery_class"] == "lost_then_state_cap"
    assert events[0]["eventual_outcome"] == "state_cap"


def test_lost_and_value_collapsed():
    g = _fixture(meta_overrides={"final_root_value": 0.2},
                 diag=[{"ply": 60, "side_to_move": "black",
                        "root_summary": {"q_value": 0.95},
                        "goal_completion": {"total_goal_distance_before": 5}}])
    events = aggregate_recovery_events([g])
    assert events[0]["recovery_class"] == "lost_and_value_collapsed"


def test_lost_but_value_stayed_high():
    g = _fixture(meta_overrides={"final_root_value": 0.99},
                 diag=[{"ply": 60, "side_to_move": "black",
                        "root_summary": {"q_value": 0.95},
                        "goal_completion": {"total_goal_distance_before": 5}}])
    events = aggregate_recovery_events([g])
    assert events[0]["recovery_class"] == "lost_but_value_stayed_high"


def test_lost_then_won_late():
    g = _fixture(rec_overrides={"conversion_delay_winner_moves": 50})
    events = aggregate_recovery_events([g])
    assert events[0]["recovery_class"] in ("lost_then_recovered", "lost_then_won_late")


def test_below_event_threshold_excluded():
    g = _fixture(rec_overrides={"winner_moves_with_dominant_unavailable": 2})
    assert aggregate_recovery_events([g]) == []
