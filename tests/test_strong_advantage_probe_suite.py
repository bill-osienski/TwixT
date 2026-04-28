"""Tests for the strong_advantage probe tier: structural features,
admission filter, ID determinism, category assignment, and the promotion
workflow.

Labeling is mocked: tests inject a stub labeler. The opt-in live smoke
test lives separately in tests/test_strong_advantage_smoke_live.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_state(moves, starting_player="red"):
    """Build a TwixtState by applying the given (row, col) moves in order."""
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    s = TwixtState(active_size=24, to_move=starting_player)
    for r, c in moves:
        s = s.apply_move((r, c))
    return s


def test_phase1_features_red_chain_top_to_mid_board():
    """Red builds a knight-connected chain from row 0 down through the
    middle. cc_size, cc_axis_span, cc_touches_own_goal must reflect the
    chain.
    """
    from scripts.GPU.alphazero.probe_eval import compute_phase1_features

    # Red knight chain: (0,12) -> (2,11) -> (4,12) -> (6,11) -> (8,12)
    # Black filler so plies alternate; black pegs placed away from red's chain.
    moves = [
        (0, 12), (1, 0),
        (2, 11), (1, 1),
        (4, 12), (1, 2),
        (6, 11), (1, 3),
        (8, 12), (1, 4),
    ]
    state = _make_state(moves)
    feats = compute_phase1_features(state, winner="red")
    assert feats["cc_size"] >= 5
    assert feats["cc_axis_span"] >= 0.30  # spans rows 0..8 of 23
    assert feats["cc_touches_own_goal"] is True  # (0, 12) touches row 0
    assert feats["forced_within_2"] is False
    # axis_span_margin = winner_span - loser_span; loser is black with no chain
    assert feats["axis_span_margin"] >= 0.20
    # centroid around row 4, col ~12; center is (11.5, 11.5) so Chebyshev
    # distance is ~7-8 (row 4 is 7.5 from center)
    assert feats["centroid_chebyshev_from_center"] <= 9


def _make_decisive_game_dict(winner_color, terminal_ply, moves):
    """Build the minimal game-record dict that probe_eval ingests."""
    return {
        "meta": {"iteration": 70},
        "winner": winner_color,
        "winner_reason": "win",
        "moves": [{"row": r, "col": c} for r, c in moves],
        "starting_player": "red",
    }


def test_extract_strong_advantage_candidates_drops_midband():
    """Mid-band centroid (Chebyshev 7-8) candidates are excluded with the
    category_midband audit reason; central and edge candidates survive.
    """
    from scripts.GPU.alphazero.probe_eval import extract_strong_advantage_candidates

    games = [
        _make_decisive_game_dict("red", 30, _central_red_chain()),
        _make_decisive_game_dict("red", 30, _edge_red_chain()),
        _make_decisive_game_dict("red", 30, _midband_red_chain()),
    ]
    candidates, audit = extract_strong_advantage_candidates(
        games, k_plies_range=(3, 8), category_min_count=0
    )
    cats = sorted(c["category"] for c in candidates)
    assert "chain_advantage_central_red" in cats
    assert "chain_advantage_edge_red" in cats
    assert all("midband" not in c["category"] for c in candidates)
    midband_drops = [a for a in audit if a["reason"] == "category_midband"]
    assert len(midband_drops) >= 1


def test_extract_strong_advantage_candidates_drops_low_axis_span_margin():
    """A candidate where the loser's chain is as long as the winner's is
    rejected via axis_span_margin < 0.10.
    """
    from scripts.GPU.alphazero.probe_eval import extract_strong_advantage_candidates

    games = [_make_decisive_game_dict("red", 20, _both_strong_chain())]
    candidates, audit = extract_strong_advantage_candidates(
        games, k_plies_range=(3, 8), category_min_count=0
    )
    assert candidates == []
    assert any(a["reason"] == "phase1_axis_span_margin" for a in audit)


def _central_red_chain():
    # Red knight chain alternating cols 11/12 across rows 0..22.
    # Centroid ≈ (11.0, 11.5), Chebyshev from (11.5, 11.5) = 0.5 → 0
    # → central (≤6).
    base = [(0, 12), (2, 11), (4, 12), (6, 11), (8, 12), (10, 11), (12, 12),
            (14, 11), (16, 12), (18, 11), (20, 12), (22, 11)]
    return _interleave_with_filler(base, filler_col=22)


def _edge_red_chain():
    # Red knight chain alternating cols 1/2 across rows 0..22 (col 0 is
    # black's goal edge — red can't play there). Centroid ≈ (11.0, 1.5),
    # Chebyshev from (11.5, 11.5) = 10 → edge (≥9).
    base = [(0, 1), (2, 2), (4, 1), (6, 2), (8, 1), (10, 2), (12, 1),
            (14, 2), (16, 1), (18, 2), (20, 1), (22, 2)]
    return _interleave_with_filler(base, filler_col=15)


def _midband_red_chain():
    # Red knight chain alternating cols 3/4 across rows 0..22.
    # Centroid ≈ (11.0, 3.5), Chebyshev from (11.5, 11.5) = 8 → midband
    # (7-8 → dropped at category assignment, audit reason category_midband).
    # Spans rows 0..22 (axis_span ≈ 0.96), cc_size = 12 — passes Phase-1
    # gates so the midband-drop path is what actually rejects it.
    base = [(0, 3), (2, 4), (4, 3), (6, 4), (8, 3), (10, 4), (12, 3),
            (14, 4), (16, 3), (18, 4), (20, 3), (22, 4)]
    return _interleave_with_filler(base, filler_col=18)


def _both_strong_chain():
    # Red knight chain cols 20/21 rows 0..22 (12 pegs, span ≈ 0.96).
    # Black knight chain rows 12..21, cols 1..19 (10 pegs, col span ≈ 0.78).
    # Chains are placed in non-overlapping column regions so no bridge
    # crossing occurs. At the k=3 sample point red has 10 pegs (cc_size ≥ 10,
    # span ≥ 0.55) but margin ≈ 0.087 < 0.10 → fails phase1_axis_span_margin.
    # Remaining sample points fail phase1_cc_size, so candidates == [].
    red = [(0, 20), (2, 21), (4, 20), (6, 21), (8, 20), (10, 21),
           (12, 20), (14, 21), (16, 20), (18, 21), (20, 20), (22, 21)]
    black = [(12, 1), (13, 3), (14, 5), (15, 7), (16, 9), (17, 11),
             (18, 13), (19, 15), (20, 17), (21, 19)]
    out = []
    for i in range(max(len(red), len(black))):
        if i < len(red):
            out.append(red[i])
        if i < len(black):
            out.append(black[i])
    return out


def _interleave_with_filler(red_moves, filler_col):
    out = []
    for i, rm in enumerate(red_moves):
        out.append(rm)
        out.append((1 + (i % 22), filler_col))  # black filler in safe column
    return out


def test_extract_strong_advantage_candidates_reads_canonical_schema():
    """Regression: real on-disk game records use meta.reason (not top-level
    winner_reason) and have an `id` field. Function must extract candidates
    from records with that schema, not just from synthetic test fixtures.
    """
    from scripts.GPU.alphazero.probe_eval import extract_strong_advantage_candidates

    # Mirror the schema scripts/GPU/alphazero/game_saver.py emits:
    # - top-level `id`, `winner`, `moves`, `starting_player`
    # - `meta.reason`, `meta.iteration`, `meta.game_idx`, `meta.board_size`
    # NO top-level `winner_reason`.
    canonical_game = {
        "id": "iter_0070_game_042",
        "winner": "red",
        "starting_player": "red",
        "moves": [{"row": r, "col": c} for r, c in _central_red_chain()],
        "meta": {
            "iteration": 70,
            "game_idx": 42,
            "reason": "win",
            "board_size": 24,
        },
    }
    candidates, audit = extract_strong_advantage_candidates(
        [canonical_game], k_plies_range=(3, 8), category_min_count=0
    )
    assert len(candidates) >= 1, (
        f"Expected at least one candidate from canonical-schema game; got {len(candidates)}. "
        f"Audit: {[(a.get('source_ply'), a.get('reason')) for a in audit][:10]}"
    )
    # source_game must be the explicit `id`, not a fallback-derived placeholder
    assert candidates[0]["source_game"] == "iter_0070_game_042"


def test_extract_strong_advantage_candidates_skips_non_decisive_canonical():
    """Regression: a canonical-schema game with meta.reason != 'win' must
    skip cleanly (zero candidates), not crash.
    """
    from scripts.GPU.alphazero.probe_eval import extract_strong_advantage_candidates

    draw_game = {
        "id": "iter_0070_game_099",
        "winner": None,
        "starting_player": "red",
        "moves": [{"row": r, "col": c} for r, c in _central_red_chain()],
        "meta": {
            "iteration": 70,
            "game_idx": 99,
            "reason": "draw",
            "board_size": 24,
        },
    }
    candidates, audit = extract_strong_advantage_candidates(
        [draw_game], k_plies_range=(3, 8), category_min_count=0
    )
    assert candidates == []


def test_label_candidate_with_mcts_uses_injected_labeler():
    """The labeler signature must be (state, sims, seed) -> (root_value,
    top1_share). Test that a stub labeler produces the expected aggregate
    (mean_root_value, value_per_run, value_stability, min_top1_share).
    """
    from scripts.GPU.alphazero.probe_eval import label_candidate_with_mcts

    state = _make_state([(0, 12), (1, 0), (2, 11)])

    canned = [(0.6, 0.30), (0.7, 0.25), (0.5, 0.40)]
    calls = []

    def stub_labeler(state, sims, seed):
        calls.append((sims, seed))
        return canned[len(calls) - 1]

    label = label_candidate_with_mcts(
        state, sims=10000, repeats=3,
        rng_seed_base=12345, labeler=stub_labeler,
    )
    assert calls == [(10000, 12345 ^ 0), (10000, 12345 ^ 1), (10000, 12345 ^ 2)]
    assert label["mean_root_value"] == pytest.approx(0.6)
    assert label["value_per_run"] == [0.6, 0.7, 0.5]
    assert label["value_stability"] == pytest.approx(0.2)
    assert label["min_top1_share"] == pytest.approx(0.25)


@pytest.fixture
def passing_candidate():
    """A candidate whose default Phase-2 evaluation passes every clause."""
    return {
        "winner": "red",
        "phase1_features": {
            "cc_size": 14,
            "cc_axis_span": 0.74,
            "cc_touches_own_goal": True,
            "axis_span_margin": 0.20,
            "centroid_chebyshev_from_center": 4,
            "forced_within_2": False,
        },
        "phase2_label": {
            "mean_root_value": 0.62,
            "value_per_run": [0.60, 0.65, 0.61],
            "value_stability": 0.05,
            "min_top1_share": 0.25,
            "label_mcts_sims": 10000,
            "label_mcts_repeats": 3,
            "rng_seed_base": 1,
        },
    }


def test_admission_passes_when_all_clauses_satisfied(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    admitted, reason = apply_admission_filter(
        passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15,
    )
    assert admitted is True
    assert reason == "admitted"


def test_admission_rejects_sign_mismatch(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase2_label"]["mean_root_value"] = -0.62
    passing_candidate["phase2_label"]["value_per_run"] = [-0.60, -0.65, -0.61]
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "sign_mismatch"


def test_admission_rejects_low_magnitude(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase2_label"]["mean_root_value"] = 0.30
    passing_candidate["phase2_label"]["value_per_run"] = [0.28, 0.32, 0.30]
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "magnitude_below_threshold"


def test_admission_rejects_low_top1_share(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase2_label"]["min_top1_share"] = 0.10
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "low_top1_share"


def test_admission_rejects_unstable_value(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase2_label"]["value_stability"] = 0.30
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "unstable_value"


def test_admission_rejects_already_forced(passing_candidate):
    from scripts.GPU.alphazero.probe_eval import apply_admission_filter
    passing_candidate["phase1_features"]["forced_within_2"] = True
    admitted, reason = apply_admission_filter(passing_candidate,
        magnitude_threshold=0.45, top1_share_floor=0.15, stability_cap=0.15)
    assert admitted is False
    assert reason == "position_already_forced"
