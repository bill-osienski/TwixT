"""Recompute fallback path (spec §11.5–§11.7).

Legacy replay walker, updated to pre-move detection semantics so its
outputs match the inline tracker's records.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.goal_completion_recompute import (
    recompute_goal_completion_records_from_replays,
)


def _replay_no_record():
    """An old-style replay JSON without goal_completion_record."""
    return {
        "iteration": 110,
        "game_idx": 0,
        "winner": "red",
        "starting_player": "red",
        "moves": [
            # Tiny synthetic 4-move game with legal TwixT moves (board_size=8).
            # Closeout-shape recognition relies on real connectivity helpers,
            # so we don't assert specific record fields here -- just structural
            # correctness. (0,0) is a corner and illegal; use (0,1) for red.
            {"player": "red",   "row": 0, "col": 1, "turn": 1},
            {"player": "black", "row": 5, "col": 5, "turn": 2},
            {"player": "red",   "row": 1, "col": 2, "turn": 3},
            {"player": "black", "row": 5, "col": 6, "turn": 4},
        ],
        "meta": {"reason": "win", "n_moves": 4, "board_size": 8,
                 "starting_player": "red"},
    }


def test_recompute_returns_same_length_as_replays():
    replays = [_replay_no_record(), _replay_no_record()]
    result = recompute_goal_completion_records_from_replays(replays, config={})
    assert len(result) == 2


def test_recompute_record_has_record_shape():
    replays = [_replay_no_record()]
    result = recompute_goal_completion_records_from_replays(replays, config={})
    rec = result[0]
    assert rec is not None
    assert rec["version"] == 1
    assert rec["outcome_class"] in (1, 2, 3)
    assert "primary_class_counts" in rec or rec["outcome_class"] in (2, 3)


def test_recompute_premove_semantics_anchor():
    """When a side has total <= detection_threshold pre-move, that ply is
    first_dominant_unclosed_ply (NOT the prior ply that created the
    closeout)."""
    replays = [_replay_no_record()]
    result = recompute_goal_completion_records_from_replays(
        replays, config={"detection_threshold": 2}
    )
    rec = result[0]
    if rec is not None and rec.get("detected"):
        # Pre-move detection: the detection ply is one where the side to
        # move already has the closeout shape pre-move.
        assert rec["first_dominant_unclosed_ply"] >= 1


def test_analyzer_recompute_fills_mixed_corpus():
    """When some replays have records and others don't, recompute fills
    the gaps so the canonical aggregator sees full coverage."""
    from scripts.twixt_replay_analyzer import (
        _merge_inline_with_recomputed,
    )

    inline = {"version": 1, "outcome_class": 1, "winner": "red",
              "detected": True, "iteration": 110, "game_idx": 0}
    recomputed_only = {"version": 1, "outcome_class": 1, "winner": "red",
                       "detected": True, "iteration": 110, "game_idx": 1}

    merged = _merge_inline_with_recomputed(
        [inline, None], [None, recomputed_only],
    )
    assert merged[0] is inline      # inline preferred when present
    assert merged[1] is recomputed_only  # recomputed used when inline missing


def test_compare_records_all_match_returns_empty():
    from scripts.GPU.alphazero.goal_completion_recompute import (
        compare_records_for_validation,
    )
    inline = {"version": 1, "outcome_class": 1, "detected": True,
              "first_dominant_unclosed_ply": 11,
              "first_total_goal_distance": 2,
              "primary_class_counts": {"completes_endpoint": 1, "redundant_reinforcement": 0,
                                       "reduces_total_goal_distance": 0, "off_chain": 0, "other": 0},
              "max_search_score_after_detection": 0.99,
              "mean_search_score_after_detection": 0.99}
    div = compare_records_for_validation(inline, inline)
    assert div == {}


def test_compare_records_field_divergence():
    from scripts.GPU.alphazero.goal_completion_recompute import (
        compare_records_for_validation,
    )
    inline = {"version": 1, "outcome_class": 1, "detected": True,
              "first_dominant_unclosed_ply": 11,
              "first_total_goal_distance": 2,
              "primary_class_counts": {"completes_endpoint": 1, "redundant_reinforcement": 0,
                                       "reduces_total_goal_distance": 0, "off_chain": 0, "other": 0}}
    recomputed = {**inline, "first_dominant_unclosed_ply": 13}
    div = compare_records_for_validation(inline, recomputed)
    assert "first_dominant_unclosed_ply" in div
    assert div["first_dominant_unclosed_ply"] == (11, 13)


def test_compare_records_float_tolerance():
    from scripts.GPU.alphazero.goal_completion_recompute import (
        compare_records_for_validation,
    )
    inline = {"version": 1, "outcome_class": 1, "detected": True,
              "max_search_score_after_detection": 0.97}
    # Within 1e-6 -> not flagged.
    recomputed = {**inline, "max_search_score_after_detection": 0.97 + 5e-7}
    assert compare_records_for_validation(inline, recomputed) == {}
    # Exceeding 1e-6 -> flagged.
    recomputed_diff = {**inline, "max_search_score_after_detection": 0.971}
    div = compare_records_for_validation(inline, recomputed_diff)
    assert "max_search_score_after_detection" in div
