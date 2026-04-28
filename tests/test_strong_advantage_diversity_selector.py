"""Tests for the diversity-aware selector in scripts/build_probe_suite.py.

The selector replaces the simple `admitted = admitted[: args.max_probes]`
slice with category round-robin + near-duplicate / ply-separation /
per-game cap rules. See spec
docs/superpowers/specs/2026-04-28-strong-advantage-diversity-selector-design.md.

Each test constructs a synthetic admitted list and asserts on the
selector's output and audit deltas. No live MCTS labeling.
"""
from __future__ import annotations

from copy import deepcopy

import pytest


def _make_candidate(
    *,
    source_game: str,
    source_ply: int,
    category: str,
    ply: int | None = None,
    cc_size: int = 18,
    axis_span_margin: float = 0.30,
    cc_axis_span: float = 0.74,
    min_top1_share: float = 0.25,
    value_stability: float = 0.05,
    mean_root_value: float = 0.62,
):
    """Construct a synthetic admitted candidate matching the fields the
    selector reads. Defaults pass every Phase-1 and Phase-2 gate so the
    only thing exercised is selection logic.

    Represents the candidate shape AFTER Phase-2 label injection (i.e.,
    after `cand['phase2_label']['label_checkpoint'] = ...` runs in
    `_run_strong_advantage`), which is what the selector sees at call time.

    `ply` defaults to `source_ply` when not provided; pass an explicit
    value if a future test needs to distinguish them.
    """
    return {
        "source_game": source_game,
        "source_ply": source_ply,
        "category": category,
        "winner": "red" if category.endswith("_red") else "black",
        "ply": ply if ply is not None else source_ply,
        "starting_player": "red",
        "move_history": [],  # selector doesn't touch this
        "phase1_features": {
            "cc_size": cc_size,
            "cc_axis_span": cc_axis_span,
            "cc_touches_own_goal": True,
            "axis_span_margin": axis_span_margin,
            "centroid_chebyshev_from_center": 4 if "central" in category else 10,
            "forced_within_2": False,
        },
        "phase2_label": {
            "mean_root_value": mean_root_value,
            "value_per_run": [mean_root_value, mean_root_value],
            "value_stability": value_stability,
            "min_top1_share": min_top1_share,
            "label_mcts_sims": 2000,
            "label_mcts_repeats": 2,
            "rng_seed_base": 1,
            "label_checkpoint": "test_ckpt.safetensors",
        },
    }


def test_diversity_sort_key_orders_by_cc_size_desc_first():
    """Stage-2 sort: larger cc_size sorts before smaller, all else equal."""
    from scripts.build_probe_suite import _diversity_sort_key

    a = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red", cc_size=20)
    b = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red", cc_size=15)

    # Lower sort-key tuple sorts first, so larger cc_size (negated) wins.
    assert _diversity_sort_key(a) < _diversity_sort_key(b)


def test_diversity_sort_key_axis_span_margin_breaks_cc_size_tie():
    """When cc_size matches, larger axis_span_margin wins."""
    from scripts.build_probe_suite import _diversity_sort_key

    a = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red",
                        cc_size=20, axis_span_margin=0.40)
    b = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red",
                        cc_size=20, axis_span_margin=0.20)

    assert _diversity_sort_key(a) < _diversity_sort_key(b)


def test_diversity_sort_key_min_top1_share_breaks_structural_ties():
    """When all structural fields tie, higher min_top1_share wins."""
    from scripts.build_probe_suite import _diversity_sort_key

    a = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red",
                        min_top1_share=0.40)
    b = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red",
                        min_top1_share=0.20)

    assert _diversity_sort_key(a) < _diversity_sort_key(b)


def test_diversity_sort_key_total_order_via_source_tiebreak():
    """Every field equal except source — final _sort_key tiebreak applies."""
    from scripts.build_probe_suite import _diversity_sort_key

    a = _make_candidate(source_game="iter_0099_game_001", source_ply=50,
                        category="chain_advantage_central_red")
    b = _make_candidate(source_game="iter_0050_game_001", source_ply=50,
                        category="chain_advantage_central_red")

    # Higher iter (-iter is smaller) wins → a sorts before b.
    assert _diversity_sort_key(a) < _diversity_sort_key(b)


def test_diversity_sort_key_cc_axis_span_breaks_margin_tie():
    """When cc_size and axis_span_margin both match, larger cc_axis_span wins.
    Pins down the position of cc_axis_span in the sort tuple."""
    from scripts.build_probe_suite import _diversity_sort_key

    a = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red",
                        cc_size=20, axis_span_margin=0.30, cc_axis_span=0.90)
    b = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red",
                        cc_size=20, axis_span_margin=0.30, cc_axis_span=0.60)

    assert _diversity_sort_key(a) < _diversity_sort_key(b)


def test_diversity_sort_key_value_stability_breaks_top1_share_tie():
    """When all structural fields and min_top1_share match, lower
    value_stability wins (more stable = better, so ascending). Pins
    down the position of value_stability in the sort tuple."""
    from scripts.build_probe_suite import _diversity_sort_key

    a = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red",
                        min_top1_share=0.30, value_stability=0.02)
    b = _make_candidate(source_game="iter_0001_game_001", source_ply=10,
                        category="chain_advantage_central_red",
                        min_top1_share=0.30, value_stability=0.10)

    assert _diversity_sort_key(a) < _diversity_sort_key(b)


def test_rule_a_near_duplicate_matches_same_game_same_category_close_features():
    from scripts.build_probe_suite import _find_near_duplicate_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red",
                             cc_size=20, axis_span_margin=0.30)
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                           category="chain_advantage_central_red",
                           cc_size=21, axis_span_margin=0.31)  # Δcc=1, Δasm=0.01

    assert _find_near_duplicate_keeper(cand, [keeper]) is keeper


def test_rule_a_near_duplicate_skips_different_game():
    from scripts.build_probe_suite import _find_near_duplicate_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red",
                             cc_size=20, axis_span_margin=0.30)
    cand = _make_candidate(source_game="iter_0058_game_999", source_ply=51,
                           category="chain_advantage_central_red",
                           cc_size=21, axis_span_margin=0.31)

    assert _find_near_duplicate_keeper(cand, [keeper]) is None


def test_rule_a_near_duplicate_skips_different_category():
    """Cross-category same-game pair is NOT a near-duplicate, even when
    structural deltas are below thresholds. Spec §5.6."""
    from scripts.build_probe_suite import _find_near_duplicate_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red",
                             cc_size=20, axis_span_margin=0.30)
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                           category="chain_advantage_edge_red",
                           cc_size=21, axis_span_margin=0.31)

    assert _find_near_duplicate_keeper(cand, [keeper]) is None


def test_rule_a_near_duplicate_skips_when_cc_size_delta_at_threshold():
    """|Δcc_size| < 2 is strict: delta == 2 is NOT a duplicate."""
    from scripts.build_probe_suite import _find_near_duplicate_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red",
                             cc_size=20, axis_span_margin=0.30)
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                           category="chain_advantage_central_red",
                           cc_size=22, axis_span_margin=0.30)  # Δcc=2

    assert _find_near_duplicate_keeper(cand, [keeper]) is None


def test_rule_a_near_duplicate_returns_smallest_source_ply_when_multiple_match():
    """Tie-break: when multiple kept candidates match, return the one
    with the smallest source_ply."""
    from scripts.build_probe_suite import _find_near_duplicate_keeper

    keeper_low = _make_candidate(source_game="iter_0058_game_040", source_ply=48,
                                 category="chain_advantage_central_red",
                                 cc_size=20, axis_span_margin=0.30)
    keeper_high = _make_candidate(source_game="iter_0058_game_040", source_ply=52,
                                  category="chain_advantage_central_red",
                                  cc_size=21, axis_span_margin=0.31)
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                           category="chain_advantage_central_red",
                           cc_size=20, axis_span_margin=0.30)

    # Both keepers match cand; smallest source_ply wins.
    assert _find_near_duplicate_keeper(cand, [keeper_high, keeper_low]) is keeper_low


def test_rule_a_near_duplicate_skips_when_axis_span_margin_delta_at_threshold():
    """|Δaxis_span_margin| < 0.05 is strict: delta == 0.05 is NOT a duplicate.
    Symmetric to the cc_size threshold test."""
    from scripts.build_probe_suite import _find_near_duplicate_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red",
                             cc_size=20, axis_span_margin=0.30)
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                           category="chain_advantage_central_red",
                           cc_size=20, axis_span_margin=0.36)  # Δasm=0.06 > 0.05

    assert _find_near_duplicate_keeper(cand, [keeper]) is None


def test_rule_a_near_duplicate_requires_both_thresholds_satisfied():
    """Predicate uses AND, not OR: cc_size in range but axis_span_margin
    out of range → NOT a duplicate. Catches accidental and→or change."""
    from scripts.build_probe_suite import _find_near_duplicate_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red",
                             cc_size=20, axis_span_margin=0.30)
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                           category="chain_advantage_central_red",
                           cc_size=21, axis_span_margin=0.36)  # Δcc=1 (pass), Δasm=0.06 (fail)

    assert _find_near_duplicate_keeper(cand, [keeper]) is None
