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
