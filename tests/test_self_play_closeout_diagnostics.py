"""Tests for inline closeout-diagnostics capture (spec 2026-05-03 §8)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_visit_counts_and_priors(active_size=8):
    """Build minimal visit_counts dict and priors_raw/priors_adjusted arrays
    suitable for build_closeout_diagnostic_partial."""
    import numpy as np
    visit_counts = {(0, 5): 100, (1, 5): 80, (2, 5): 60, (3, 4): 40, (5, 5): 20}
    n = active_size * active_size
    priors = np.zeros(n, dtype=np.float32)
    priors[0 * active_size + 5] = 0.30  # (0, 5)
    priors[1 * active_size + 5] = 0.20  # (1, 5)
    priors[2 * active_size + 5] = 0.15  # (2, 5)
    priors[3 * active_size + 4] = 0.10  # (3, 4)
    priors[5 * active_size + 5] = 0.05  # (5, 5)
    return visit_counts, priors, priors


def _decode_move(active_size):
    return lambda mid: (mid // active_size, mid % active_size)


def test_build_closeout_diagnostic_partial_includes_root_summary_and_goal_completion():
    """Partial record carries root_summary, goal_completion sub-block, and
    completion-move-ranking fields."""
    from scripts.GPU.alphazero.closeout_diagnostics import (
        build_closeout_diagnostic_partial,
    )
    visit_counts, priors_raw, priors_adj = _make_visit_counts_and_priors(active_size=8)

    class _StubRoot:
        visit_count = 100
        q_value = 0.95
        nn_value = 0.92

    gc_state = {
        "max_depth": 3,
        "total_goal_distance": 2,
        "endpoint_distances": {"top": 1, "bottom": 1},
        "largest_component_size": 11,
        "category": "two_endpoint_closeout_2ply",
        "endpoint_completion_moves": [(0, 5), (7, 5)],
        "distance_reducing_moves": [(0, 5), (7, 5), (3, 4)],
        "component_pegs": frozenset({(2, 5), (4, 5), (6, 5)}),
    }
    rec = build_closeout_diagnostic_partial(
        ply=10,
        side_to_move="red",
        visit_counts=visit_counts,
        priors_raw=priors_raw,
        priors_adjusted=priors_adj,
        root=_StubRoot(),
        goal_completion_state=gc_state,
        board_size=8,
        skip_distance_reducing=False,
        decode_fn=_decode_move(8),
    )
    assert rec["ply"] == 10
    assert rec["side_to_move"] == "red"
    assert rec["root_summary"]["q_value"] == 0.95
    assert rec["goal_completion"]["total_goal_distance_before"] == 2
    assert rec["goal_completion"]["category"] == "two_endpoint_closeout_2ply"
    assert rec["endpoint_completion_ranking"] is not None
    # (0, 5) is the top by both visit and policy.
    assert rec["endpoint_completion_ranking"]["best_visit_rank"] == 1
    assert rec["endpoint_completion_ranking"]["any_in_visit_top5"] is True


def test_build_closeout_diagnostic_partial_no_endpoint_completion_moves_yields_null_ranking():
    from scripts.GPU.alphazero.closeout_diagnostics import build_closeout_diagnostic_partial
    visit_counts, priors_raw, priors_adj = _make_visit_counts_and_priors(active_size=8)

    class _StubRoot:
        visit_count = 100
        q_value = 0.5
        nn_value = 0.5

    gc_state = {
        "max_depth": 3, "total_goal_distance": 3,
        "endpoint_distances": {"top": 2, "bottom": 1},
        "largest_component_size": 8,
        "category": "broader_conversion",
        "endpoint_completion_moves": [],
        "distance_reducing_moves": [(3, 4)],
        "component_pegs": frozenset({(2, 5), (4, 5)}),
    }
    rec = build_closeout_diagnostic_partial(
        ply=10, side_to_move="red", visit_counts=visit_counts,
        priors_raw=priors_raw, priors_adjusted=priors_adj,
        root=_StubRoot(), goal_completion_state=gc_state,
        board_size=8, skip_distance_reducing=False, decode_fn=_decode_move(8),
    )
    assert rec["endpoint_completion_ranking"] is None
    assert rec["distance_reducing_ranking"] is not None


def test_build_closeout_diagnostic_partial_skip_flag_nulls_distance_reducing():
    from scripts.GPU.alphazero.closeout_diagnostics import build_closeout_diagnostic_partial
    visit_counts, priors_raw, priors_adj = _make_visit_counts_and_priors(active_size=8)

    class _StubRoot:
        visit_count = 100
        q_value = 0.5
        nn_value = 0.5

    gc_state = {
        "max_depth": 3, "total_goal_distance": 2,
        "endpoint_distances": {"top": 1, "bottom": 1},
        "largest_component_size": 11,
        "category": "two_endpoint_closeout_2ply",
        "endpoint_completion_moves": [(0, 5), (7, 5)],
        "distance_reducing_moves": [(0, 5), (7, 5), (3, 4)],
        "component_pegs": frozenset({(2, 5), (4, 5)}),
    }
    rec = build_closeout_diagnostic_partial(
        ply=10, side_to_move="red", visit_counts=visit_counts,
        priors_raw=priors_raw, priors_adjusted=priors_adj,
        root=_StubRoot(), goal_completion_state=gc_state,
        board_size=8, skip_distance_reducing=True, decode_fn=_decode_move(8),
    )
    assert rec["distance_reducing_ranking"] is None
    assert rec["goal_completion"]["distance_reducing_moves"] is None


def test_finalize_closeout_diagnostic_adds_selected_and_classification():
    """finalize() adds selected_move + classification to a partial record without mutating it."""
    from scripts.GPU.alphazero.closeout_diagnostics import (
        build_closeout_diagnostic_partial, finalize_closeout_diagnostic,
    )
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.connectivity_diagnostics import compute_goal_completion_state

    # Build a real state where Red has a dominant unclosed structure.
    # Use the existing curated fixture pattern from test_connectivity_goal_completion.py.
    moves = [(2, 5), (10, 15), (0, 4), (10, 16)]  # Red(2,5), Black(10,15), Red(0,4), Black(10,16)
    s = TwixtState(active_size=24, to_move="red")
    for m in moves:
        s = s.apply_move(m)
    gc_state = compute_goal_completion_state(s, "red", max_depth=3, min_component_size=1)

    # Synthetic visit_counts / priors for the rank computation.
    visit_counts = {(0, 6): 80, (1, 6): 60}
    import numpy as np
    priors = np.zeros(24 * 24, dtype=np.float32)
    priors[0 * 24 + 6] = 0.4
    priors[1 * 24 + 6] = 0.3

    class _StubRoot:
        visit_count = 200
        q_value = 0.8
        nn_value = 0.7

    decode_fn = lambda mid: (mid // 24, mid % 24)
    partial = build_closeout_diagnostic_partial(
        ply=4, side_to_move="red", visit_counts=visit_counts,
        priors_raw=priors, priors_adjusted=priors, root=_StubRoot(),
        goal_completion_state=gc_state or {
            "max_depth": 3, "total_goal_distance": 1,
            "endpoint_distances": {"top": 0, "bottom": 1},
            "largest_component_size": 2,
            "category": "one_move_win",
            "endpoint_completion_moves": [(0, 6)],
            "distance_reducing_moves": [(0, 6)],
            "component_pegs": frozenset({(2, 5), (0, 4)}),
        },
        board_size=24, skip_distance_reducing=False, decode_fn=decode_fn,
    )
    assert "selected_move" not in partial
    assert "selected_move_classification" not in partial

    # Finalize with a synthetic selected_move. classify_selected_conversion_move
    # will call apply_move internally; just check that the keys are added.
    if gc_state is not None:
        # Use a real move from the gc_state for legality.
        completion_moves = gc_state.get("endpoint_completion_moves") or [(0, 6)]
        sel = tuple(completion_moves[0]) if completion_moves else (0, 6)
        full = finalize_closeout_diagnostic(partial, s, "red", sel, gc_state)
        assert full["selected_move"] == list(sel)
        assert "selected_move_classification" in full
        assert "primary_class" in full["selected_move_classification"]
        # Original partial dict not mutated.
        assert "selected_move" not in partial
