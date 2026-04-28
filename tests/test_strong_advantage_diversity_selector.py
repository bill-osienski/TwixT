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


def test_rule_a_near_duplicate_skips_when_axis_span_margin_delta_above_threshold():
    """|Δaxis_span_margin| < 0.05: a delta strictly above threshold is NOT
    a duplicate. Catches accidental swap of cc_size/axis_span_margin
    threshold lines or relaxation of the threshold value.

    Note: testing the exact-threshold case (Δ == 0.05) cleanly with
    floats is unreliable (e.g., abs(0.35 - 0.30) == 0.0499...), so this
    test uses Δ = 0.06 instead."""
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


def test_rule_b_ply_too_close_matches_same_game_within_separation():
    from scripts.build_probe_suite import _find_ply_too_close_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                           category="chain_advantage_central_red")
    rank_index = {id(keeper): 0, id(cand): 1}

    assert _find_ply_too_close_keeper(cand, [keeper], rank_index) is keeper


def test_rule_b_ply_too_close_admits_at_separation_boundary():
    """|Δsource_ply| < MIN_PLY_SEPARATION_SAME_GAME=3 is strict: Δ=3
    is admissible, not too close."""
    from scripts.build_probe_suite import _find_ply_too_close_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=53,  # Δ=3
                           category="chain_advantage_central_red")
    rank_index = {id(keeper): 0, id(cand): 1}

    assert _find_ply_too_close_keeper(cand, [keeper], rank_index) is None


def test_rule_b_ply_too_close_skips_different_game():
    from scripts.build_probe_suite import _find_ply_too_close_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_999", source_ply=51,
                           category="chain_advantage_central_red")
    rank_index = {id(keeper): 0, id(cand): 1}

    assert _find_ply_too_close_keeper(cand, [keeper], rank_index) is None


def test_rule_b_ply_too_close_ignores_category_only_game_matters():
    """Rule B is category-agnostic: same-game cross-category pair within
    separation still triggers Rule B."""
    from scripts.build_probe_suite import _find_ply_too_close_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                           category="chain_advantage_edge_red")
    rank_index = {id(keeper): 0, id(cand): 1}

    assert _find_ply_too_close_keeper(cand, [keeper], rank_index) is keeper


def test_rule_b_tie_break_prefers_closest_keeper():
    """Two keepers, candidate at ply 51: keeper at 50 (Δ=1) wins over
    keeper at 49 (Δ=2). The closer keeper has WORSE Stage-2 rank
    (rank 5 vs rank 0) so this test specifically exercises that
    distance dominates rank in the tie-break (would catch a swap of
    fields 0 and 1 in the sort-key lambda)."""
    from scripts.build_probe_suite import _find_ply_too_close_keeper

    closest = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                              category="chain_advantage_central_red")
    farther = _make_candidate(source_game="iter_0058_game_040", source_ply=49,
                              category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                           category="chain_advantage_central_red")
    # Closer keeper has WORSE rank (5) than farther keeper (0).
    # Distance must win — otherwise farther would be returned.
    rank_index = {id(closest): 5, id(farther): 0, id(cand): 99}

    assert _find_ply_too_close_keeper(cand, [farther, closest], rank_index) is closest


def test_rule_b_tie_break_uses_better_rank_when_equidistant():
    """Two keepers equidistant from candidate (both Δ=1): the one with
    better Stage-2 rank (lower rank_index value) wins."""
    from scripts.build_probe_suite import _find_ply_too_close_keeper

    higher_rank = _make_candidate(source_game="iter_0058_game_040", source_ply=49,
                                  category="chain_advantage_central_red")
    lower_rank = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                                 category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                           category="chain_advantage_central_red")
    # higher_rank has rank 0 (better); lower_rank has rank 5 (worse).
    rank_index = {id(higher_rank): 0, id(lower_rank): 5, id(cand): 99}

    assert _find_ply_too_close_keeper(cand, [lower_rank, higher_rank], rank_index) is higher_rank


def test_rule_b_tie_break_falls_back_to_smallest_source_ply():
    """When equidistant AND same rank_index value (synthetic edge case
    only achievable via test setup), smallest source_ply wins."""
    from scripts.build_probe_suite import _find_ply_too_close_keeper

    later_ply = _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                                category="chain_advantage_central_red")
    earlier_ply = _make_candidate(source_game="iter_0058_game_040", source_ply=49,
                                  category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                           category="chain_advantage_central_red")
    # Force same rank_index for both keepers to exercise the final tie-break.
    rank_index = {id(later_ply): 0, id(earlier_ply): 0, id(cand): 99}

    assert _find_ply_too_close_keeper(cand, [later_ply, earlier_ply], rank_index) is earlier_ply


def test_rule_c_per_game_cap_returns_none_when_under_cap():
    from scripts.build_probe_suite import _find_per_game_cap_keeper

    keeper = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                             category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=53,
                           category="chain_advantage_central_red")

    # 1 keeper, cap=2 → not exceeded.
    assert _find_per_game_cap_keeper(cand, [keeper], cap=2) is None


def test_rule_c_per_game_cap_fires_when_at_cap():
    from scripts.build_probe_suite import _find_per_game_cap_keeper

    keeper_a = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                               category="chain_advantage_central_red")
    keeper_b = _make_candidate(source_game="iter_0058_game_040", source_ply=53,
                               category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=56,
                           category="chain_advantage_central_red")

    # 2 keepers, cap=2 → exceeded for the next candidate.
    keeper = _find_per_game_cap_keeper(cand, [keeper_a, keeper_b], cap=2)
    assert keeper is keeper_a  # smallest source_ply


def test_rule_c_per_game_cap_counts_across_categories():
    """Cap is total per game, not per (game, category). One central +
    one edge from the same game already fills cap=2."""
    from scripts.build_probe_suite import _find_per_game_cap_keeper

    central = _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                              category="chain_advantage_central_red")
    edge = _make_candidate(source_game="iter_0058_game_040", source_ply=53,
                           category="chain_advantage_edge_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=56,
                           category="chain_advantage_central_black")

    keeper = _find_per_game_cap_keeper(cand, [central, edge], cap=2)
    assert keeper is central


def test_rule_c_per_game_cap_ignores_other_games():
    from scripts.build_probe_suite import _find_per_game_cap_keeper

    keeper_other = _make_candidate(source_game="iter_0058_game_999", source_ply=50,
                                   category="chain_advantage_central_red")
    cand = _make_candidate(source_game="iter_0058_game_040", source_ply=53,
                           category="chain_advantage_central_red")

    assert _find_per_game_cap_keeper(cand, [keeper_other], cap=1) is None


def test_selector_returns_all_when_under_max_probes_and_no_rules_fire():
    """Three structurally distinct candidates from three different games
    in three different categories. cap=2, max_probes=10. All three kept;
    audit gets 3 admitted rows; no diversity drops."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    cands = [
        _make_candidate(source_game=f"iter_0001_game_{i:03d}", source_ply=50,
                        category=cat, cc_size=20 + i)
        for i, cat in enumerate([
            "chain_advantage_central_red",
            "chain_advantage_central_black",
            "chain_advantage_edge_red",
        ])
    ]
    audit = []

    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=10, max_probes_per_game=2,
    )

    assert len(kept) == 3
    admitted_rows = [r for r in audit if r["reason"] == "admitted"]
    diversity_rows = [r for r in audit if r["reason"].startswith("diversity_")]
    assert len(admitted_rows) == 3
    assert len(diversity_rows) == 0


def test_selector_per_game_cap_test():
    """5 probes from one game (no near-dupes, well-separated plies) plus
    1 from another. cap=2 → 2 from clustered game survive; 3 dropped
    with diversity_per_game_cap; 1 from the other game survives."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    # Same game, 5 plies far enough apart to clear ply-separation,
    # cc_size descending so they don't trip near-duplicate.
    clustered = [
        _make_candidate(source_game="iter_0058_game_040", source_ply=ply,
                        category="chain_advantage_central_red", cc_size=cs)
        for ply, cs in [(40, 28), (44, 24), (48, 20), (52, 16), (56, 12)]
    ]
    other = _make_candidate(source_game="iter_0058_game_041", source_ply=50,
                            category="chain_advantage_central_red", cc_size=22)
    audit = []

    kept = _select_diverse_admitted_candidates(
        clustered + [other], audit, max_probes=10, max_probes_per_game=2,
    )

    kept_from_clustered = [k for k in kept if k["source_game"] == "iter_0058_game_040"]
    assert len(kept_from_clustered) == 2
    assert other in kept

    cap_drops = [r for r in audit if r["reason"] == "diversity_per_game_cap"]
    assert len(cap_drops) == 3
    for row in cap_drops:
        assert row["source_game"] == "iter_0058_game_040"
        assert row["kept_instead_source_ply"] == 40  # smallest kept ply


def test_selector_near_duplicate_suppression():
    """3 same-game same-category probes with cc_size=(20,21,25) and
    axis_span_margin=(0.20,0.21,0.40). Probes with cc_size=20 and 21 are
    duplicates; rank-2 of those is dropped with diversity_near_duplicate;
    cc_size=25 is kept as structurally distinct."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    cands = [
        _make_candidate(source_game="iter_0058_game_040", source_ply=40,
                        category="chain_advantage_central_red",
                        cc_size=20, axis_span_margin=0.20),
        _make_candidate(source_game="iter_0058_game_040", source_ply=44,
                        category="chain_advantage_central_red",
                        cc_size=21, axis_span_margin=0.21),
        _make_candidate(source_game="iter_0058_game_040", source_ply=48,
                        category="chain_advantage_central_red",
                        cc_size=25, axis_span_margin=0.40),
    ]
    audit = []

    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=10, max_probes_per_game=3,
    )

    # cc_size=25 is rank-1 (sorts first), kept. Then cc_size=21 and 20
    # are siblings of 25 — but Δcc_size from 25 is 4 and 5 (≥ 2), so
    # they're NOT duplicates of 25. Walking in rank order: 25 is kept;
    # then 21 (rank 2) is checked — Δcc from 25 is 4, not a duplicate;
    # 21 is kept. Then 20 (rank 3) is checked — Δcc from 21 is 1 AND
    # Δasm = 0.01 → IS a duplicate → dropped.
    kept_cc_sizes = sorted(k["phase1_features"]["cc_size"] for k in kept)
    assert kept_cc_sizes == [21, 25]

    dup_drops = [r for r in audit if r["reason"] == "diversity_near_duplicate"]
    assert len(dup_drops) == 1
    assert dup_drops[0]["source_ply"] == 40  # cc_size=20 was the dropped one
    assert dup_drops[0]["kept_instead_source_ply"] == 44  # ply of cc_size=21 keeper


def test_selector_ply_separation():
    """3 same-game probes, structurally distinct (no near-dupes),
    source_ply ∈ {50, 51, 54}. cap=3 (so cap doesn't bind). Plies 50
    and 54 kept; 51 dropped with diversity_ply_too_close."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    # Stage-2 sort wants larger cc_size first. Order them so that 50
    # has the largest cc, then 54, then 51 — exercising the sort.
    cands = [
        _make_candidate(source_game="iter_0058_game_040", source_ply=50,
                        category="chain_advantage_central_red", cc_size=28),
        _make_candidate(source_game="iter_0058_game_040", source_ply=54,
                        category="chain_advantage_central_red", cc_size=22),
        _make_candidate(source_game="iter_0058_game_040", source_ply=51,
                        category="chain_advantage_central_red", cc_size=18),
    ]
    audit = []

    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=10, max_probes_per_game=3,
    )

    kept_plies = sorted(k["source_ply"] for k in kept)
    assert kept_plies == [50, 54]

    too_close = [r for r in audit if r["reason"] == "diversity_ply_too_close"]
    assert len(too_close) == 1
    assert too_close[0]["source_ply"] == 51
    assert too_close[0]["kept_instead_source_ply"] == 50  # closest keeper


def test_selector_drop_reason_precedence():
    """Synthetic candidate that triggers BOTH diversity_near_duplicate
    AND diversity_per_game_cap: audit reason must be near_duplicate
    (more specific wins)."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    # Two keepers from same game at cap=2, then a third candidate that
    # is also a near-duplicate of one of them.
    a = _make_candidate(source_game="iter_0058_game_040", source_ply=40,
                        category="chain_advantage_central_red",
                        cc_size=28, axis_span_margin=0.40)
    b = _make_candidate(source_game="iter_0058_game_040", source_ply=44,
                        category="chain_advantage_central_red",
                        cc_size=22, axis_span_margin=0.30)
    # c is a near-duplicate of b (Δcc=1, Δasm=0.01) AND would exceed
    # cap=2 if kept. Rule A fires first.
    c = _make_candidate(source_game="iter_0058_game_040", source_ply=48,
                        category="chain_advantage_central_red",
                        cc_size=21, axis_span_margin=0.31)
    audit = []

    kept = _select_diverse_admitted_candidates(
        [a, b, c], audit, max_probes=10, max_probes_per_game=2,
    )

    drop_rows = [r for r in audit if r["source_ply"] == 48]
    assert len(drop_rows) == 1
    assert drop_rows[0]["reason"] == "diversity_near_duplicate"


def test_cli_accepts_max_probes_per_game_flag():
    """argparse accepts --max-probes-per-game with int, default 2.

    Inspects --help output via subprocess rather than importing the
    parser directly, since the parser is constructed inside main()
    and isn't exposed as a module-level object.
    """
    import subprocess
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "build_probe_suite.py"),
         "--tier", "strong_advantage", "--help"],
        capture_output=True, text=True, cwd=project_root,
    )
    assert "--max-probes-per-game" in result.stdout, (
        f"--max-probes-per-game flag not found in --help output:\n{result.stdout}"
    )
    assert "default: 2" in result.stdout.lower() or "default 2" in result.stdout.lower() \
        or "(default: 2)" in result.stdout, (
        f"Expected default of 2 documented in --help:\n{result.stdout}"
    )


def test_end_to_end_strong_advantage_runs_selector_and_writes_meta(tmp_path, monkeypatch):
    """Full _run_strong_advantage with stubbed labeler, network loader,
    and admission filter. Asserts:
    - draft file is written
    - audit file is written
    - admitted audit rows count == probes in draft (no double-counting)
    - meta.selection_rules contains the new diversity keys
    - per-game cap is upheld

    Stubs are applied to the module BEFORE main_with_args runs, so the
    function-local `from scripts.GPU.alphazero.probe_eval import ...`
    inside _run_strong_advantage picks up the stubbed callables.

    INVARIANT: this test depends on _run_strong_advantage doing its
    probe_eval imports lazily (inside the function body, not at module
    scope). If those imports are ever hoisted, this monkeypatch pattern
    silently bypasses the stubs and the test starts using real MCTS
    labeling. If you hoist the imports, switch to patching
    `scripts.build_probe_suite.label_candidate_with_mcts` etc. instead.
    """
    import json
    from pathlib import Path
    from unittest.mock import MagicMock

    import scripts.GPU.alphazero.probe_eval as pe
    from scripts.build_probe_suite import main_with_args

    project_root = Path(__file__).resolve().parent.parent

    # Stub the labeler to return a sign-positive passing label. The
    # generator post-normalizes to red-perspective per STM, so a positive
    # raw value is fine for whichever side moved last.
    def stub_label(state, sims, repeats, rng_seed_base, labeler=None):
        return {
            "mean_root_value": 0.95,
            "value_per_run": [0.95, 0.95],
            "value_stability": 0.0,
            "min_top1_share": 0.30,
            "label_mcts_sims": sims,
            "label_mcts_repeats": repeats,
            "rng_seed_base": rng_seed_base,
        }
    monkeypatch.setattr(pe, "label_candidate_with_mcts", stub_label)

    # Stub admission filter to always pass. This decouples the
    # integration test from the (real) sign-checking logic — the
    # selector's behavior is the focus of this test, and the admission
    # filter is independently covered in test_strong_advantage_probe_suite.py.
    monkeypatch.setattr(pe, "apply_admission_filter",
                        lambda cand, **kwargs: (True, "admitted"))

    # Stub network loader. MagicMock so the subsequent .eval() call works.
    mock_network = MagicMock()
    monkeypatch.setattr(pe, "load_network_for_scoring",
                        lambda path: (mock_network, 24, 128, 6))
    monkeypatch.setattr(pe, "_set_default_labeler_network", lambda net: None)

    # Fake checkpoint file: must exist (existence check), never read.
    fake_ckpt = tmp_path / "fake.safetensors"
    fake_ckpt.write_bytes(b"fake")

    # Run against the same source range the committed file uses.
    out_path = tmp_path / "strong_advantage_probes.json"
    rc = main_with_args([
        "--tier", "strong_advantage",
        "--input", str(project_root / "scripts" / "GPU" / "logs" / "games"),
        "--source-iter-range", "57", "58",
        "--label-checkpoint", str(fake_ckpt),
        "--out", str(out_path),
        "--max-probes", "30",
        "--max-probes-per-game", "2",
        "--label-mcts-sims", "100",
        "--label-mcts-repeats", "1",
        "--force",
    ])
    assert rc == 0, f"generator exited {rc}"

    draft_path = out_path.with_suffix(".draft.json")
    audit_path = out_path.parent / "candidates_strong_advantage.json"
    assert draft_path.exists(), f"draft file missing: {draft_path}"
    assert audit_path.exists(), f"audit file missing: {audit_path}"

    draft = json.loads(draft_path.read_text())
    audit = json.loads(audit_path.read_text())["audit"]

    # Sanity: at least one probe survived all stages — otherwise the
    # equality assertion below would pass trivially.
    assert len(draft["probes"]) > 0, (
        "no probes produced — check iter_0057-58 fixture in "
        "scripts/GPU/logs/games/"
    )

    # No audit double-counting: admitted rows == probes in draft.
    admitted_rows = [r for r in audit if r["reason"] == "admitted"]
    assert len(admitted_rows) == len(draft["probes"]), (
        f"audit admitted count ({len(admitted_rows)}) != probes "
        f"({len(draft['probes'])})"
    )

    # Per-game cap upheld.
    from collections import Counter
    per_game = Counter(p["source_game"] for p in draft["probes"])
    assert all(n <= 2 for n in per_game.values()), (
        f"per-game cap of 2 violated: {per_game.most_common(5)}"
    )

    # meta.selection_rules has the new keys.
    rules = draft["meta"]["selection_rules"]
    assert rules["max_probes_per_game"] == 2
    assert rules["min_ply_separation_same_game"] == 3
    assert rules["category_iteration_order"] == [
        "chain_advantage_central_red",
        "chain_advantage_central_black",
        "chain_advantage_edge_red",
        "chain_advantage_edge_black",
    ]
    assert "diversity_quality_key_order" in rules
    assert isinstance(rules["diversity_quality_key_order"], list)
    assert len(rules["diversity_quality_key_order"]) >= 6


def test_selector_determinism_under_input_shuffle():
    """Same admitted list in different orders must produce byte-identical
    selector output AND identical audit deltas."""
    from copy import deepcopy
    import random

    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    base_cands = [
        _make_candidate(source_game=f"iter_{50 + g:04d}_game_{g:03d}",
                        source_ply=40 + g * 3,
                        category=cat, cc_size=20 + g)
        for g in range(8)
        for cat in ["chain_advantage_central_red", "chain_advantage_central_black"]
    ]

    rng = random.Random(42)
    shuffled = deepcopy(base_cands)
    rng.shuffle(shuffled)

    audit_a = []
    kept_a = _select_diverse_admitted_candidates(
        deepcopy(base_cands), audit_a, max_probes=10, max_probes_per_game=2,
    )
    audit_b = []
    kept_b = _select_diverse_admitted_candidates(
        shuffled, audit_b, max_probes=10, max_probes_per_game=2,
    )

    # Compare by stable identity (source_game, source_ply, category).
    def _identity(c):
        return (c["source_game"], c["source_ply"], c["category"])

    assert [_identity(k) for k in kept_a] == [_identity(k) for k in kept_b], (
        "kept order differs between original and shuffled input"
    )

    audit_keys_a = sorted(
        (r["source_game"], r["source_ply"], r["reason"]) for r in audit_a
    )
    audit_keys_b = sorted(
        (r["source_game"], r["source_ply"], r["reason"]) for r in audit_b
    )
    assert audit_keys_a == audit_keys_b, "audit deltas differ between runs"


def test_round_robin_skips_empty_without_reordering_nonempty():
    """Only central_black (position 2) and edge_red (position 3) populated;
    central_red (1) and edge_black (4) empty. Selector walks the canonical
    4-tuple, skips empties, and the relative order between the two
    non-empty categories matches [central_black, edge_red] repeating —
    NOT alphabetical or yield-based. Spec §5.4."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    # 4 candidates each in central_black and edge_red, all from different
    # games so per-game cap doesn't bind.
    cands = []
    for i in range(4):
        cands.append(_make_candidate(
            source_game=f"iter_{60 + i:04d}_game_001",
            source_ply=40, category="chain_advantage_central_black",
            cc_size=30 - i,  # decreasing so insertion-order != rank-order
        ))
        cands.append(_make_candidate(
            source_game=f"iter_{70 + i:04d}_game_001",
            source_ply=40, category="chain_advantage_edge_red",
            cc_size=30 - i,
        ))

    audit = []
    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=8, max_probes_per_game=2,
    )

    # First 4 picks should alternate central_black, edge_red, central_black,
    # edge_red because the round-robin walks canonical order (1=skip empty,
    # 2=central_black, 3=edge_red, 4=skip empty), and after one pass takes
    # one from each non-empty bucket.
    cats = [k["category"] for k in kept]
    expected_pattern = ["chain_advantage_central_black", "chain_advantage_edge_red"] * 4
    assert cats == expected_pattern, (
        f"Expected canonical-order alternation, got: {cats}"
    )


def test_cross_category_same_game_not_deduped():
    """Two same-game probes with structural deltas BELOW thresholds but in
    DIFFERENT categories. Rule A's same-category requirement prevents
    dedupe; both kept (under cap). Spec §5.6."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    a = _make_candidate(source_game="iter_0058_game_040", source_ply=40,
                        category="chain_advantage_central_red",
                        cc_size=20, axis_span_margin=0.30)
    b = _make_candidate(source_game="iter_0058_game_040", source_ply=44,  # > sep
                        category="chain_advantage_edge_red",
                        cc_size=21, axis_span_margin=0.31)  # Δcc=1, Δasm=0.01
    audit = []

    kept = _select_diverse_admitted_candidates(
        [a, b], audit, max_probes=10, max_probes_per_game=2,
    )

    assert len(kept) == 2
    assert {k["category"] for k in kept} == {
        "chain_advantage_central_red", "chain_advantage_edge_red"
    }
    near_dup = [r for r in audit if r["reason"] == "diversity_near_duplicate"]
    assert near_dup == []


def test_sparse_category_backfill():
    """Only central_red populated (10 candidates from 5 games, 2 per game).
    max_probes=10, cap=2. All 10 kept; no errors; round-robin gracefully
    skips empties."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    cands = []
    for game_idx in range(5):
        for ply_offset in [0, 4]:  # > separation
            cands.append(_make_candidate(
                source_game=f"iter_0099_game_{game_idx:03d}",
                source_ply=40 + ply_offset,
                category="chain_advantage_central_red",
                cc_size=28 - game_idx - ply_offset,  # all distinct
                axis_span_margin=0.40 - game_idx * 0.06 - ply_offset * 0.01,
            ))
    audit = []

    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=10, max_probes_per_game=2,
    )

    assert len(kept) == 10
    assert all(k["category"] == "chain_advantage_central_red" for k in kept)


def test_edge_categories_empty_alternates_two_centrals():
    """central_red and central_black populated, edge_* empty. Round-robin
    alternates central_red ↔ central_black; output respects max_probes."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    cands = []
    for game_idx in range(6):
        for cat in ["chain_advantage_central_red", "chain_advantage_central_black"]:
            cands.append(_make_candidate(
                source_game=f"iter_0099_game_{game_idx:03d}_{cat[-3:]}",
                source_ply=40,
                category=cat,
                cc_size=28 - game_idx,
            ))
    audit = []

    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=6, max_probes_per_game=2,
    )

    assert len(kept) == 6
    cats = [k["category"] for k in kept]
    # First 6 picks alternate central_red, central_black, central_red, ...
    expected = ["chain_advantage_central_red", "chain_advantage_central_black"] * 3
    assert cats == expected


def test_audit_kept_instead_field_present_on_diversity_drops_only():
    """diversity_* rows carry kept_instead_source_ply pointing at a real
    keeper. admitted rows do NOT have this field."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    # Setup: clustered game forces per-game cap drops.
    cands = [
        _make_candidate(source_game="iter_0058_game_040", source_ply=ply,
                        category="chain_advantage_central_red", cc_size=cs)
        for ply, cs in [(40, 28), (44, 24), (48, 20)]
    ]
    audit = []
    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=10, max_probes_per_game=2,
    )
    kept_plies = {k["source_ply"] for k in kept}

    for row in audit:
        if row["reason"].startswith("diversity_"):
            assert "kept_instead_source_ply" in row
            assert row["kept_instead_source_ply"] in kept_plies, (
                f"kept_instead_source_ply={row['kept_instead_source_ply']} "
                f"not in actually-kept plies {kept_plies}"
            )
        else:
            assert row["reason"] == "admitted"
            assert "kept_instead_source_ply" not in row


def test_audit_admitted_count_equals_kept_count():
    """Selector writes exactly one reason='admitted' audit row per kept
    candidate. No double-counting."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    cands = [
        _make_candidate(source_game=f"iter_0099_game_{i:03d}",
                        source_ply=40,
                        category="chain_advantage_central_red",
                        cc_size=28 - i)
        for i in range(5)
    ]
    audit = []
    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=3, max_probes_per_game=1,
    )

    admitted_rows = [r for r in audit if r["reason"] == "admitted"]
    assert len(admitted_rows) == len(kept) == 3


def test_quality_key_structural_priority():
    """Two same-game same-category candidates: one with higher cc_size
    but lower min_top1_share, the other with lower cc_size but higher
    min_top1_share, structurally far enough apart that the near-duplicate
    rule does not fire (cc_size = (15, 25)). With max_probes_per_game=1,
    the higher cc_size wins (structural beats Phase-2 fields)."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    high_struct = _make_candidate(
        source_game="iter_0058_game_040", source_ply=40,
        category="chain_advantage_central_red",
        cc_size=25, min_top1_share=0.20,
    )
    high_phase2 = _make_candidate(
        source_game="iter_0058_game_040", source_ply=44,
        category="chain_advantage_central_red",
        cc_size=15, min_top1_share=0.50,
    )
    audit = []

    kept = _select_diverse_admitted_candidates(
        [high_phase2, high_struct], audit,  # input order shouldn't matter
        max_probes=10, max_probes_per_game=1,
    )

    assert len(kept) == 1
    assert kept[0] is high_struct


def test_category_round_robin_canonical_order_all_four_populated():
    """Spec §9 item 4: all 4 categories populated with 4 candidates each
    (16 total, all from distinct games so per-game cap doesn't bind AND
    Rule B does not collide cross-category). With max_probes=8, the
    round-robin walks the canonical 4-tuple twice and admits 2 from each
    category in canonical order: central_red, central_black, edge_red,
    edge_black, central_red, ..., for a total of 8 probes split 2/2/2/2."""
    from scripts.build_probe_suite import _select_diverse_admitted_candidates

    # Use the FULL category name as a suffix so source_game is unique per
    # (category, game_idx). Rule B is category-agnostic, so a same-source_game
    # collision across categories at ply 40 would trigger ply-too-close drops
    # even with disjoint per-category indices.
    cands = []
    for game_idx in range(4):
        for cat in [
            "chain_advantage_central_red",
            "chain_advantage_central_black",
            "chain_advantage_edge_red",
            "chain_advantage_edge_black",
        ]:
            cands.append(_make_candidate(
                source_game=f"iter_0099_game_{game_idx:03d}_{cat}",
                source_ply=40, category=cat,
                cc_size=28 - game_idx,  # rank decreases within each bucket
            ))

    audit = []
    kept = _select_diverse_admitted_candidates(
        cands, audit, max_probes=8, max_probes_per_game=2,
    )

    assert len(kept) == 8

    # Iteration order must be the canonical 4-tuple repeating twice.
    expected_cats = [
        "chain_advantage_central_red",
        "chain_advantage_central_black",
        "chain_advantage_edge_red",
        "chain_advantage_edge_black",
    ] * 2
    assert [k["category"] for k in kept] == expected_cats

    # Each category contributes exactly 2 probes (8 / 4).
    from collections import Counter
    by_cat = Counter(k["category"] for k in kept)
    assert all(by_cat[c] == 2 for c in [
        "chain_advantage_central_red",
        "chain_advantage_central_black",
        "chain_advantage_edge_red",
        "chain_advantage_edge_black",
    ])
