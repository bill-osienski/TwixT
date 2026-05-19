"""Tests for the marathon-termination diagnostics module.

Spec: docs/superpowers/specs/2026-05-19-marathon-termination-tuning-design.md
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.marathon_termination_diagnostics import (
    detect_no_progress_windows,
    classify_adjudication_coverage,
    ADJUDICATION_GATE_BUCKETS,
    NO_PROGRESS_WINDOW_SIZE,
)


def _ply_entry(*, primary_class: str,
               own_td_before: int = 5, own_td_after: int = 5,
               opp_td_before: int = 5, opp_td_after: int = 5):
    """A goal_completion_diagnostics entry, minimally shaped for the detector."""
    return {
        "ply": 50, "side_to_move": "red",
        "goal_completion": {
            "total_goal_distance_before": own_td_before,
            "category": "one_endpoint_distance_2",
            "_own_td_after": own_td_after,
            "_opp_td_before": opp_td_before,
            "_opp_td_after": opp_td_after,
        },
        "selected_move": [10, 10],
        "selected_move_classification": {
            "primary_class": primary_class,
            "total_goal_distance_before": own_td_before,
            "total_goal_distance_after": own_td_after,
        },
    }


def test_no_progress_window_detects_pure_structural_run():
    """Spec §7 test 1. 15 consecutive redundant_reinforcement moves
    with no goal-distance progress → 1 window detected."""
    entries = [
        _ply_entry(primary_class="redundant_reinforcement",
                   own_td_before=5, own_td_after=5,
                   opp_td_before=5, opp_td_after=5)
        for _ in range(15)
    ]
    assert detect_no_progress_windows(entries, side="red") == 1


def test_no_progress_window_breaks_on_distance_reduction():
    """Spec §7 test 2. 14 redundant + 1 reduces_total_goal_distance → 0 windows."""
    entries = [
        _ply_entry(primary_class="redundant_reinforcement",
                   own_td_before=5, own_td_after=5)
        for _ in range(14)
    ]
    entries.append(_ply_entry(
        primary_class="reduces_total_goal_distance",
        own_td_before=5, own_td_after=4,
    ))
    assert detect_no_progress_windows(entries, side="red") == 0


def test_no_progress_window_breaks_on_endpoint_completion():
    """Spec §7 test 3. 14 redundant + 1 completes_endpoint → 0 windows."""
    entries = [
        _ply_entry(primary_class="redundant_reinforcement")
        for _ in range(14)
    ]
    entries.append(_ply_entry(primary_class="completes_endpoint"))
    assert detect_no_progress_windows(entries, side="red") == 0


def test_no_progress_window_breaks_on_opponent_block():
    """Spec §7 test 4. 14 redundant + 1 blocks_opponent_closeout → 0 windows."""
    entries = [
        _ply_entry(primary_class="redundant_reinforcement")
        for _ in range(14)
    ]
    entries.append(_ply_entry(primary_class="blocks_opponent_closeout"))
    assert detect_no_progress_windows(entries, side="red") == 0


def test_no_progress_window_window_size_15():
    """Spec §7 test 5. Exactly 14 redundant → 0 windows; 15 → 1."""
    e14 = [_ply_entry(primary_class="redundant_reinforcement") for _ in range(14)]
    e15 = [_ply_entry(primary_class="redundant_reinforcement") for _ in range(15)]
    assert detect_no_progress_windows(e14, side="red") == 0
    assert detect_no_progress_windows(e15, side="red") == 1
    # Sanity-check the exported constant.
    assert NO_PROGRESS_WINDOW_SIZE == 15


def test_no_progress_window_opponent_block_uses_primary_class_only():
    """Spec §7 test 6. The opponent-block test uses the
    primary_class == 'blocks_opponent_closeout' marker (Spec 4 vocabulary)
    — confirms we are NOT applying a stricter local recomputation. If
    Spec 4's defense classifier flagged the move, we trust it."""
    # A move classified as blocks_opponent_closeout (per Spec 4) — even if
    # the inline distance fields look ambiguous — must count as a block.
    entries = [
        _ply_entry(primary_class="redundant_reinforcement")
        for _ in range(14)
    ]
    entries.append({
        "ply": 50, "side_to_move": "red",
        "goal_completion": {"total_goal_distance_before": 5, "category": "x"},
        "selected_move": [10, 10],
        "selected_move_classification": {
            "primary_class": "blocks_opponent_closeout",
        },
    })
    # Run of 14 followed by a block → 0 no-progress windows.
    assert detect_no_progress_windows(entries, side="red") == 0


def _gc_record_state_cap(**meta_overrides):
    """A per-game (record, meta, diagnostics) triple for a 280-ply state_cap game."""
    record = {
        "iteration": 220, "game_idx": 0,
        "winner": None, "reason": "state_cap", "n_moves": 280,
        "first_total_goal_distance": 2,
        "winner_moves_with_dominant_unavailable": 0,
        "conversion_delay_plies": 0,
    }
    meta = {"reason": "state_cap", "n_moves": 280}
    meta.update(meta_overrides)
    return record, meta, []


def test_adjudication_coverage_blocked_by_min_top1():
    """Spec §7 test 7. self-play 'top1' → bucket 'min_top1_share'."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason="top1")
    assert classify_adjudication_coverage(rec, meta, diag) == "min_top1_share"


def test_adjudication_coverage_blocked_by_value_below_threshold():
    """Spec §7 test 8. self-play 'threshold' → bucket 'value_below_threshold'."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason="threshold")
    assert classify_adjudication_coverage(rec, meta, diag) == "value_below_threshold"


def test_adjudication_coverage_blocked_by_min_visits():
    """Spec §7 test 9. self-play 'visits' → bucket 'min_visits'."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason="visits")
    assert classify_adjudication_coverage(rec, meta, diag) == "min_visits"


def test_adjudication_coverage_not_attempted_when_ply_blocked():
    """Spec §7 test 10. self-play 'ply' → 'not_attempted' (adjudication
    couldn't run because the ply gate fired)."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason="ply")
    assert classify_adjudication_coverage(rec, meta, diag) == "not_attempted"


def test_adjudication_coverage_would_have_passed_when_none_on_state_cap():
    """Spec §7 test 11. state_cap game with adjudication_block_reason
    PRESENT as None (key exists, value is None) → 'would_have_passed'
    (bug indicator: game state-capped, key was set but no blocking gate
    recorded → an attempt should have happened). MUST NOT silently
    collapse to 'not_attempted'."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason=None)
    # Explicit assertion: the key IS in meta, just None-valued.
    assert "adjudication_block_reason" in meta
    assert meta["adjudication_block_reason"] is None
    assert classify_adjudication_coverage(rec, meta, diag) == "would_have_passed"


def test_adjudication_coverage_missing_signal_when_key_absent():
    """Spec §7 test 12. Old-format per-game JSON where the key isn't
    present at all → 'missing_signal' (observability gap, not a bug)."""
    rec, meta, diag = _gc_record_state_cap()
    assert "adjudication_block_reason" not in meta
    assert classify_adjudication_coverage(rec, meta, diag) == "missing_signal"


def test_adjudication_coverage_skipped_for_non_state_cap_games():
    """Spec §7 test 14. Game ending in win → returns None
    (excluded from §3.2 scope)."""
    record = {
        "iteration": 220, "game_idx": 0,
        "winner": "red", "reason": "win", "n_moves": 80,
    }
    meta = {"reason": "win", "adjudication_block_reason": None}
    assert classify_adjudication_coverage(record, meta, []) is None


def test_adjudication_gate_buckets_export_matches_spec_taxonomy():
    """Spec §3.2 enumeration. The exported tuple lists exactly the six
    bucket names so the analyzer and tests share one source of truth."""
    assert set(ADJUDICATION_GATE_BUCKETS) == {
        "not_attempted",
        "value_below_threshold",
        "min_top1_share",
        "min_visits",
        "missing_signal",
        "would_have_passed",
    }
