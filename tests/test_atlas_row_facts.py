"""Atlas Stage 5, Task 0 -- row facts derived from frozen measured fields.

The seam Stage 4 could not qualify: `phase`, `flat_policy` and `near_even` were
caller-supplied booleans and every Stage 4 test hardcoded them.

Pure: synthetic dicts only. No reservoir, no checkpoint, no MLX.
"""
import pytest

from scripts.GPU.alphazero.atlas_readout_c import classify_edge_strata, is_flat
from scripts.GPU.alphazero.atlas_row_facts import (
    NEAR_EVEN_ABS_VALUE, derive_row_facts,
)
from scripts.GPU.alphazero.warm_prefix_replay import LegResult

FLAT = {i: 1.0 / 500 for i in range(500)}
SHARP = {0: 0.9, 1: 0.05, 2: 0.05}


def _legs(v400=0.10):
    return [LegResult(nominal_B=b, inherited_I=10, effective=10 + b,
                      root_value=(v400 if b == 400 else 0.05),
                      selected_move=3, selected_move_prior_rank=1,
                      top_share=0.5, top_two_margin=0.2,
                      effective_children=12.0, n_visited_children=20,
                      visit_counts={3: 100})
            for b in (400, 1600, 3200, 6400)]


def _snaps(root_priors=FLAT, with_root_edge=True):
    edges = ([{"parent_path": (), "move": 0, "depth": 0,
               "parent_priors": root_priors, "sources": (3200, 6400)}]
             if with_root_edge else [])
    return {"reference_lines": {"merged": {"required_edges": edges,
                                           "agreement": {}}}}


def test_the_frozen_near_even_bound_is_pinned_and_not_a_new_number():
    assert NEAR_EVEN_ABS_VALUE == 0.30          # design section 8, verbatim


def test_phase_is_TRAJECTORY_RELATIVE_and_side_alternates():
    """Amendment 5: ply 30 is `late` in a 40-move game and `opening` in a
    200-move one, so the cross-check must agree with the game, not the ply."""
    f = derive_row_facts(_legs(), _snaps(), 30, 40, "red")
    assert f["phase"] == "late" and f["side"] == "red"      # even ply
    f = derive_row_facts(_legs(), _snaps(), 30, 200, "red")
    assert f["phase"] == "opening"
    f = derive_row_facts(_legs(), _snaps(), 95, 100, "red")
    assert f["phase"] == "late" and f["side"] == "black"    # odd ply


def test_n_moves_is_required_here_too():
    """A default would let a stale call site keep absolute-like behaviour."""
    with pytest.raises(TypeError):
        derive_row_facts(_legs(), _snaps(), 30, start_player="red")


def test_near_even_uses_the_B400_root_value_in_stm_perspective():
    assert derive_row_facts(_legs(0.10), _snaps(), 12, 200, "red")["near_even"] is True
    assert derive_row_facts(_legs(-0.29), _snaps(), 12, 200, "red")["near_even"] is True
    assert derive_row_facts(_legs(0.31), _snaps(), 12, 200, "red")["near_even"] is False
    # The bound is inclusive, exactly as section 8 states it.
    assert derive_row_facts(_legs(0.30), _snaps(), 12, 200, "red")["near_even"] is True


def test_near_even_is_None_when_the_400_rung_is_absent():
    legs = [l for l in _legs() if l.nominal_B != 400]
    f = derive_row_facts(legs, _snaps(), 12, 200, "red")
    assert f["near_even"] is None               # None, never False
    assert "near_even" in f["undefined"]


def test_flat_policy_applies_the_frozen_predicate_to_the_ROOT_EDGE_priors():
    assert derive_row_facts(_legs(), _snaps(FLAT), 12, 200, "red")["flat_policy"] is True
    assert derive_row_facts(_legs(), _snaps(SHARP), 12, 200, "red")["flat_policy"] is False


def test_flat_policy_is_None_when_the_merged_line_has_no_root_edge():
    """Undefined, never False. The row is KEPT and the gap is reported."""
    f = derive_row_facts(_legs(), _snaps(with_root_edge=False), 12, 200, "red")
    assert f["flat_policy"] is None
    assert "flat_policy" in f["undefined"]


def test_root_and_edge_flatness_use_THE_SAME_predicate():
    """One frozen definition, one implementation. A second copy is how the
    root stratum and the local strata drift apart."""
    assert is_flat(FLAT) is True and is_flat(SHARP) is False
    # The edge-level classifier is built on the same function.
    assert classify_edge_strata({"depth": 1, "parent_priors": FLAT}) == {
        "locally_flat_depth1"}
    assert classify_edge_strata({"depth": 1, "parent_priors": SHARP}) == set()


def test_a_derived_phase_that_contradicts_the_assignment_FAILS_the_row():
    """assign_corpus already emits phase and side, so re-deriving them is a
    CROSS-CHECK. Disagreement means the assignment and the ply have drifted."""
    with pytest.raises(ValueError, match="phase"):
        derive_row_facts(_legs(), _snaps(), 95, 100, "red", assigned_phase="opening")
    with pytest.raises(ValueError, match="side"):
        derive_row_facts(_legs(), _snaps(), 95, 100, "red", assigned_side="red")
    # Agreement passes silently.
    derive_row_facts(_legs(), _snaps(), 95, 100, "red",
                     assigned_phase="late", assigned_side="black")
