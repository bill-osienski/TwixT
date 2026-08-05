"""Atlas Stage 4, Task 1 -- labelling and capacity sizing (design section 5, 3).

Pure: synthetic LegResult rows only. No reservoir, no checkpoint, no MLX.
"""
import pytest

from scripts.GPU.alphazero.atlas_labelling import (
    class_counts, classify_row, final_capacity_gate, size_from_pilot,
    stable_reference,
)
from scripts.GPU.alphazero.warm_prefix_replay import LegResult


def _legs(v400, v1600, v3200, v6400, m400, m3200, m6400, margin=0.20):
    """Four rungs with only the fields labelling reads."""
    vals = (v400, v1600, v3200, v6400)
    moves = (m400, m400, m3200, m6400)
    out = []
    for i, (b, v, m) in enumerate(zip((400, 1600, 3200, 6400), vals, moves)):
        out.append(LegResult(
            nominal_B=b, inherited_I=10, effective=10 + b, root_value=v,
            selected_move=m, selected_move_prior_rank=1, top_share=0.5,
            top_two_margin=(margin if b == 6400 else 0.30),
            effective_children=12.0, n_visited_children=20,
            visit_counts={m: 100}))
    return out


def test_stable_reference_requires_all_three_conditions():
    ok = _legs(0.9, 0.5, 0.10, 0.05, m400=7, m3200=3, m6400=3)
    assert stable_reference(ok)["stable"] is True

    moves_disagree = _legs(0.9, 0.5, 0.10, 0.05, m400=7, m3200=3, m6400=9)
    assert stable_reference(moves_disagree)["stable"] is False

    values_apart = _legs(0.9, 0.5, 0.90, 0.05, m400=7, m3200=3, m6400=3)
    assert stable_reference(values_apart)["stable"] is False

    thin_margin = _legs(0.9, 0.5, 0.10, 0.05, m400=7, m3200=3, m6400=3, margin=0.01)
    assert stable_reference(thin_margin)["stable"] is False


def test_misleading_is_an_OR_of_value_and_move():
    by_value = _legs(0.9, 0.5, 0.10, 0.05, m400=3, m3200=3, m6400=3)
    assert classify_row(by_value) == "misleading"       # |0.9-0.05| >= 0.25
    by_move = _legs(0.06, 0.05, 0.05, 0.05, m400=7, m3200=3, m6400=3)
    assert classify_row(by_move) == "misleading"        # different 400 move


def test_stable_negative_is_an_AND():
    r = _legs(0.06, 0.05, 0.05, 0.05, m400=3, m3200=3, m6400=3)
    assert classify_row(r) == "stable_negative"


def test_the_ambiguous_band_is_kept_not_forced():
    """Same move, value gap in (0.10, 0.25) -- neither class."""
    r = _legs(0.20, 0.10, 0.05, 0.05, m400=3, m3200=3, m6400=3)
    assert classify_row(r) == "ambiguous"


def test_rows_without_a_stable_reference_are_their_own_class():
    r = _legs(0.9, 0.5, 0.10, 0.05, m400=7, m3200=3, m6400=9)
    assert classify_row(r) == "no_stable_reference"


def test_class_counts_report_components_separately():
    rows = [_legs(0.9, 0.5, 0.10, 0.05, 3, 3, 3),          # misleading by value
            _legs(0.06, 0.05, 0.05, 0.05, 7, 3, 3),        # misleading by move
            _legs(0.06, 0.05, 0.05, 0.05, 3, 3, 3)]        # stable negative
    c = class_counts(rows)
    assert c["misleading"] == 2 and c["stable_negative"] == 1
    # Section 5: value and move components reported separately -- a detector that
    # predicts value correction but not move error is weaker evidence.
    assert c["misleading_by_value"] == 1 and c["misleading_by_move"] == 1


def test_sizing_matches_the_frozen_formula():
    counts = {"misleading": 8, "stable_negative": 9}       # of 24 pilot rows
    r = size_from_pilot(counts)
    # N_required = max(24/(0.4*p_m), 30/(0.4*p_s)); rounded up to a multiple of 40
    assert r["p_m"] == pytest.approx(8 / 24)
    assert r["p_s"] == pytest.approx(9 / 24)
    assert r["N"] % 40 == 0 and 200 <= r["N"] <= 400
    assert r["verdict"] == "OK"


def test_sizing_fails_closed_on_a_zero_class_frequency():
    r = size_from_pilot({"misleading": 0, "stable_negative": 9})
    assert r["verdict"] == "PROJECTED_CAPACITY_NO_GO"
    assert r["N"] is None                # None, never a defaulted number


def test_sizing_fails_closed_when_the_requirement_exceeds_400():
    r = size_from_pilot({"misleading": 1, "stable_negative": 1})
    assert r["verdict"] == "PROJECTED_CAPACITY_NO_GO"


def test_final_capacity_gate_needs_20_misleading_and_25_stable_negative():
    assert final_capacity_gate({"misleading": 20, "stable_negative": 25})["verdict"] == "OK"
    short = final_capacity_gate({"misleading": 19, "stable_negative": 25})
    assert short["verdict"] == "CAPACITY_FAILURE"
    assert "misleading" in short["short_of"]
