"""Tests for analyzer per-move stats aggregation (spec 2026-05-03 §5.4-5.5)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import (
    aggregate_per_move_stats,
    format_per_move_stats_report,
)


def _replay(moves):
    return {"moves": moves, "meta": {"n_moves": len(moves)}}


def _move(search_score=None, top1=None):
    return {
        "turn": 1, "player": "red", "row": 0, "col": 0,
        "bridges_created": [], "heuristics": {},
        "search_score": search_score, "root_top1_share": top1,
    }


def test_aggregate_per_move_stats_zero_coverage_for_old_replays():
    """Old replays (moves without search_score / root_top1_share keys) → coverage 0,
    distributions null."""
    old_replays = [
        {"moves": [{"turn": 1, "player": "red", "row": 0, "col": 0,
                    "bridges_created": [], "heuristics": {}}]}
    ] * 5
    result = aggregate_per_move_stats(old_replays)
    assert result["n_games_total"] == 5
    assert result["n_moves_total"] == 5
    assert result["coverage"]["search_score"] == 0
    assert result["coverage"]["root_top1_share"] == 0
    assert result["search_score"] is None
    assert result["root_top1_share"] is None


def test_aggregate_per_move_stats_full_coverage_distributions_correct():
    """Synthetic replay set with known scores → percentiles correct."""
    replays = [
        _replay([_move(0.10, 0.40), _move(0.20, 0.30), _move(0.30, 0.50)]),
        _replay([_move(-0.10, 0.20), _move(0.50, 0.60)]),
    ]
    r = aggregate_per_move_stats(replays)
    assert r["n_games_total"] == 2
    assert r["n_moves_total"] == 5
    assert r["coverage"]["search_score"] == 5
    assert r["search_score"]["min"] == -0.1
    assert r["search_score"]["max"] == 0.5
    # Mean of [0.1, 0.2, 0.3, -0.1, 0.5] == 0.2
    assert abs(r["search_score"]["mean"] - 0.2) < 1e-9
    # mean_abs of [0.1, 0.2, 0.3, 0.1, 0.5] == 0.24
    assert abs(r["search_score"]["mean_abs"] - 0.24) < 1e-9
    # Mean of [0.4, 0.3, 0.5, 0.2, 0.6] == 0.4
    assert abs(r["root_top1_share"]["mean"] - 0.4) < 1e-9


def test_aggregate_per_move_stats_partial_coverage_excludes_missing_not_zero():
    """Mixed coverage: replays with some moves carrying scores, others not.
    Distributions only over present values; coverage counts at move level."""
    replays = [
        _replay([_move(0.5, 0.5), _move(None, None)]),         # 2 moves, 1 covered
        _replay([_move(None, None), _move(None, None)]),       # 2 moves, 0 covered
        _replay([_move(0.9, 0.9)]),                            # 1 move,  1 covered
    ]
    r = aggregate_per_move_stats(replays)
    assert r["n_games_total"] == 3
    assert r["n_moves_total"] == 5
    assert r["coverage"]["search_score"] == 2
    # Average = (0.5 + 0.9) / 2 = 0.7 (NOT depressed by the 3 missing zeros)
    assert abs(r["search_score"]["mean"] - 0.7) < 1e-9


def test_format_per_move_stats_report_uniform_coverage_suppresses_coverage_line():
    per_move = aggregate_per_move_stats(
        [_replay([_move(0.5, 0.5), _move(0.6, 0.6)])]
    )
    out = format_per_move_stats_report(per_move)
    text = "\n".join(out)
    assert "Per-move stats" in text
    assert "Coverage:" not in text  # uniform full coverage


def test_format_per_move_stats_report_zero_coverage_short_message():
    per_move = aggregate_per_move_stats(
        [{"moves": [{"turn": 1, "player": "red", "row": 0, "col": 0,
                     "bridges_created": [], "heuristics": {}}]}]
    )
    out = format_per_move_stats_report(per_move)
    text = "\n".join(out)
    assert "no moves carry new fields" in text
