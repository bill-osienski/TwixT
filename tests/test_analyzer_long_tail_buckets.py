"""Analyzer-side integration tests for the long-tail bucket classifier.

Spec: docs/superpowers/specs/2026-05-19-long-tail-bucket-classifier-design.md
"""
import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import (
    write_goal_completion_worst_cases_csv,
    write_goal_completion_long_tail_buckets_csv,
    format_long_tail_bucket_report,
)
from scripts.GPU.alphazero.long_tail_bucket_classifier import aggregate_long_tail_buckets


def _gc_record(**overrides):
    base = {
        "iteration": 200, "game_idx": 0, "game_id": "game_000",
        "winner": "red", "starting_player": "red",
        "n_moves": 80, "reason": "win", "outcome_class": 1, "scope": "broader",
        "detected_player": "red", "first_dominant_unclosed_ply": 30,
        "first_total_goal_distance": 2, "first_category": "two_endpoint_closeout",
        "actual_terminal_ply": 80, "actual_win_ply": 80,
        "conversion_delay_plies": 25, "conversion_delay_winner_moves": 12,
        "cap_delay_proxy_plies": 0,
        "primary_class_counts": {
            "completes_endpoint": 2, "reduces_total_goal_distance": 0,
            "redundant_reinforcement": 8, "off_chain": 1, "other": 1,
        },
        "winner_moves_in_watch_window": 12, "winner_moves_with_dominant_component": 12,
        "winner_moves_with_dominant_unavailable": 0,
        "max_search_score_after_detection": 0.95, "mean_search_score_after_detection": 0.6,
        "high_value_after_detection_plies": 3, "root_value_high_but_delayed": True,
        "search_score_coverage_in_watch_window": 1.0,
    }
    base.update(overrides)
    return base


def _replay(record, diagnostics=None):
    return {
        "goal_completion_record": record,
        "goal_completion_diagnostics": diagnostics or [],
    }


def _redund_ply(*, has_top5_alt: bool):
    return {
        "ply": 50, "side_to_move": "red",
        "endpoint_completion_ranking": {"any_in_visit_top5": has_top5_alt},
        "distance_reducing_ranking":   {"any_in_visit_top5": has_top5_alt},
        "selected_move": [10, 10],
        "selected_move_classification": {"primary_class": "redundant_reinforcement"},
    }


def test_worst_cases_csv_has_long_tail_bucket_column(tmp_path):
    """Existing CSV gains the new long_tail_bucket column; rows have correct labels."""
    replays = [
        _replay(_gc_record(iteration=200, game_idx=0, reason="state_cap", n_moves=280, winner=None,
                           conversion_delay_plies=0)),
        _replay(_gc_record(iteration=200, game_idx=1, first_total_goal_distance=2,
                           conversion_delay_plies=30),
                [_redund_ply(has_top5_alt=True) for _ in range(3)]),
        _replay(_gc_record(iteration=200, game_idx=2, first_total_goal_distance=2,
                           conversion_delay_plies=30),
                [_redund_ply(has_top5_alt=False) for _ in range(3)]),
    ]
    out_path = tmp_path / "wc.csv"
    write_goal_completion_worst_cases_csv(str(out_path), replays, top_k=25)
    rows = list(csv.DictReader(open(out_path)))
    assert all("long_tail_bucket" in r for r in rows)
    by_idx = {int(r["game_idx"]): r["long_tail_bucket"] for r in rows}
    assert by_idx[0] == "marathon_or_state_cap"
    assert by_idx[1] == "td2_alt_in_top5"
    assert by_idx[2] == "td2_reducer_buried"


def test_analyzer_writes_long_tail_buckets_csv(tmp_path):
    """Synthetic records produce expected per-iter + range_total rows.
    Per-iter rows have iteration >= 0; range-total rows have iteration = -1."""
    replays = [
        _replay(_gc_record(iteration=200, game_idx=0, reason="state_cap", n_moves=280, winner=None)),
        _replay(_gc_record(iteration=200, game_idx=1, first_total_goal_distance=3,
                           conversion_delay_plies=25)),
        _replay(_gc_record(iteration=201, game_idx=0, first_total_goal_distance=2,
                           conversion_delay_plies=25),
                [_redund_ply(has_top5_alt=True) for _ in range(3)]),
    ]
    pairs = [(r["goal_completion_record"], r["goal_completion_diagnostics"]) for r in replays]
    agg = aggregate_long_tail_buckets(pairs)
    out_path = tmp_path / "buckets.csv"
    write_goal_completion_long_tail_buckets_csv(str(out_path), agg)
    rows = list(csv.DictReader(open(out_path)))
    # Six buckets x 2 iters + 6 range-total rows = 18 rows.
    assert len(rows) == 18
    iters = sorted({int(r["iteration"]) for r in rows})
    assert iters == [-1, 200, 201]
    # Range-total marathon == 1.
    range_marathon = next(r for r in rows if int(r["iteration"]) == -1 and r["bucket"] == "marathon_or_state_cap")
    assert int(range_marathon["games"]) == 1
    assert int(range_marathon["total_long_tail_games"]) == 3


def test_format_long_tail_bucket_report_renders_section_with_hints():
    """Report section has header, totals, per-bucket rows with next-action hints."""
    pairs = [
        (_gc_record(iteration=200, reason="state_cap", n_moves=280, winner=None), []),
        (_gc_record(iteration=200, first_total_goal_distance=2, conversion_delay_plies=25),
         [_redund_ply(has_top5_alt=True) for _ in range(3)]),
    ]
    agg = aggregate_long_tail_buckets(pairs)
    lines = format_long_tail_bucket_report(agg, range_label="200-209")
    text = "\n".join(lines)
    assert "Long-tail bucket counts (200-209)" in text
    assert "Long-tail filter: delay >= 20 OR state_cap" in text
    assert "marathon_or_state_cap" in text
    assert "td2_alt_in_top5" in text
    assert "Fix 2 calibration" in text  # next-action hint
    assert "Per-iter trend:" in text


def test_format_long_tail_bucket_report_empty_range():
    """Empty range emits the header + 'no long-tail games' line."""
    agg = aggregate_long_tail_buckets([])
    lines = format_long_tail_bucket_report(agg, range_label="empty")
    text = "\n".join(lines)
    assert "Total long-tail games in range: 0" in text
    assert "(no long-tail games in this range)" in text
