"""Atlas Stage 4, Task 4 -- Read-out B, old-gate calibration (section 7).

Pure: synthetic LegResult rows only. No reservoir, no checkpoint, no MLX.
"""
import pytest

from scripts.GPU.alphazero.atlas_labelling import stable_reference
from scripts.GPU.alphazero.atlas_readout_b import (
    BASE_RATE_MARGIN, MIN_CONVERGENT_RATE, MIN_ELIGIBLE_TRIGGERS,
    by_stratum_summary, calibrate_gate, closes_half, compound_narrowing,
    convergent, gate_triggers, natural_convergence_report,
)
from scripts.GPU.alphazero.warm_prefix_replay import LegResult


def _leg(b, value, move, top_share=0.5, eff=12.0, rank=1, margin=0.20):
    return LegResult(nominal_B=b, inherited_I=10, effective=10 + b,
                     root_value=value, selected_move=move,
                     selected_move_prior_rank=rank, top_share=top_share,
                     top_two_margin=margin, effective_children=eff,
                     n_visited_children=20, visit_counts={move: 100})


def _legs(v=(0.9, 0.4, 0.05, 0.05), m=(7, 3, 3, 3), shares=(0.5, 0.6, 0.7, 0.7),
          effs=(20.0, 16.0, 12.0, 12.0), ranks=(1, 1, 1, 1)):
    return [_leg(b, v[i], m[i], shares[i], effs[i], ranks[i])
            for i, b in enumerate((400, 1600, 3200, 6400))]


def _row(legs=None, phase="late", flat=False, near_even=False):
    return {"legs": legs or _legs(), "phase": phase,
            "flat_policy": flat, "near_even": near_even}


def test_frozen_thresholds_are_pinned():
    assert MIN_ELIGIBLE_TRIGGERS == 10
    assert MIN_CONVERGENT_RATE == 0.75
    assert BASE_RATE_MARGIN == 0.15


def test_closes_half_needs_a_real_gap_and_half_closure():
    assert closes_half(1.0, 0.4, 0.0) is True
    assert closes_half(1.0, 0.8, 0.0) is False
    assert closes_half(0.5, 0.5, 0.5) is False       # no gap: vacuous


def test_convergent_requires_persistence_as_a_JOINT_condition():
    legs = _legs()
    assert convergent(legs, stable_reference(legs))["convergent"] is True
    broken = _legs(m=(7, 9, 3, 3))
    r = convergent(broken, stable_reference(broken))
    assert r["persistent"] is False and r["convergent"] is False


def test_dist_convergent_needs_the_SAME_metric_at_both_deep_rungs():
    """The disjunction is over METRICS, not rungs.

    The `mixed` fixture is built so that top_share closes half its gap toward
    3,200 but NOT toward 6,400, while effective_children closes toward 6,400 but
    NOT toward 3,200. Every individual closure is real, so an implementation
    that ORed across rungs would score this as convergence; requiring the SAME
    metric on both sides is what rejects it.
    """
    ok = _legs(v=(0.06, 0.05, 0.05, 0.05), m=(3, 3, 3, 3),
               shares=(0.20, 0.60, 0.62, 0.62))
    assert convergent(ok, stable_reference(ok))["dist_convergent"] is True

    mixed = _legs(v=(0.06, 0.05, 0.05, 0.05), m=(3, 3, 3, 3),
                  # top_share: 0.20 -> 0.60. Toward 0.90 closes (0.30 <= 0.35);
                  # toward 0.30 does not (0.30 > 0.05).
                  shares=(0.20, 0.60, 0.90, 0.30),
                  # effective_children: 20 -> 16. Toward 12 closes (4 <= 4);
                  # toward 30 does not (14 > 5).
                  effs=(20.0, 16.0, 30.0, 12.0))
    r = convergent(mixed, stable_reference(mixed))
    assert closes_half(0.20, 0.60, 0.90) is True      # each half is real...
    assert closes_half(20.0, 16.0, 12.0) is True
    assert r["dist_convergent"] is False              # ...but not the same metric


def test_gate_triggers_take_the_upper_rung_as_a_parameter():
    legs = _legs(shares=(0.90, 0.96, 0.97, 0.97))
    assert gate_triggers(legs, hi=1600)["new_collapse"] is True
    assert gate_triggers(legs, hi=6400)["new_collapse"] is True
    quiet = _legs(shares=(0.90, 0.91, 0.97, 0.97))
    assert gate_triggers(quiet, hi=1600)["new_collapse"] is False


def test_lower_prior_flip_uses_the_prior_RANK():
    legs = _legs(m=(3, 9, 9, 9), ranks=(1, 7, 7, 7))
    assert gate_triggers(legs, hi=1600)["lower_prior_flip"] is True


def test_compound_narrowing_is_an_AGGREGATE_not_a_per_row_boolean():
    """Mean effective-children reduction >= 0.50 AND mean top-share increase
    >= 0.15, over the cohort."""
    strong = [_row(_legs(shares=(0.50, 0.90, 0.90, 0.90),
                         effs=(20.0, 8.0, 8.0, 8.0))) for _ in range(4)]
    assert compound_narrowing(strong) is True          # 60% and +0.40
    # Narrowed, but nowhere near the aggregate thresholds -- a per-row
    # directional test would wrongly call this compound narrowing.
    slight = [_row(_legs(shares=(0.50, 0.52, 0.52, 0.52),
                         effs=(20.0, 19.0, 19.0, 19.0))) for _ in range(4)]
    assert compound_narrowing(slight) is False


def test_compound_narrowing_is_None_where_inapplicable():
    assert compound_narrowing([_row(_legs(shares=(None, None, None, None)))]) is None
    assert compound_narrowing([]) is None


def test_compound_narrowing_is_None_when_ANY_row_is_partially_missing():
    """Both means must describe the SAME cohort. A row contributing to one mean
    but not the other would make them summarize different row sets."""
    good = _row(_legs(shares=(0.50, 0.90, 0.90, 0.90),
                      effs=(20.0, 8.0, 8.0, 8.0)))
    partial = _row(_legs(shares=(0.50, 0.90, 0.90, 0.90),
                         effs=(None, None, None, None)))   # share only
    assert compound_narrowing([good] * 4) is True
    assert compound_narrowing([good] * 4 + [partial]) is None


def test_natural_convergence_report_covers_400_to_6400():
    rows = [_row() for _ in range(4)]
    r = natural_convergence_report(rows)
    assert set(r["trigger_rates"]) >= {"new_collapse", "top_share_increase"}
    # Reported as the reference distribution, NOT causal evidence that a
    # same-budget intervention is safe.
    assert r["is_causal_evidence"] is False
    assert r["transition"] == "400->6400"


def test_calibration_uses_the_ELIGIBLE_denominator():
    rows = [_row() for _ in range(12)] + [_row(_legs(m=(7, 3, 3, 9)))] * 5
    r = calibrate_gate(rows, "top_share_increase")
    assert r["eligible_triggers"] <= r["total_triggers"]
    assert r["eligible_trigger_fraction"] is not None


def test_needs_review_requires_all_three_conditions():
    """Each condition falsified individually -- accepting either verdict would
    prove nothing."""
    conv = [_row() for _ in range(12)]
    assert calibrate_gate(conv[:5], "top_share_increase")["verdict"] == "no finding"
    mixed = conv[:6] + [_row(_legs(m=(7, 9, 3, 3))) for _ in range(6)]
    rm = calibrate_gate(mixed, "top_share_increase")
    assert (rm["convergent_rate"] or 0) < MIN_CONVERGENT_RATE
    assert rm["verdict"] == "no finding"
    ra = calibrate_gate(conv, "top_share_increase")
    margin = ((ra["convergent_rate"] - ra["base_convergent_rate"])
              if ra["convergent_rate"] is not None else None)
    assert (ra["verdict"] == "needs review") == (
        ra["eligible_triggers"] >= MIN_ELIGIBLE_TRIGGERS
        and (ra["convergent_rate"] or 0) >= MIN_CONVERGENT_RATE
        and (margin or 0) >= BASE_RATE_MARGIN)
    assert "invalid" not in ra["verdict"]


def test_calibration_is_None_not_zero_with_no_eligible_triggers():
    rows = [_row(_legs(m=(7, 3, 3, 9)))] * 4
    r = calibrate_gate(rows, "new_collapse")
    assert r["convergent_rate"] is None and r["verdict"] == "no finding"


def test_stratum_summary_uses_the_ROW_schema_and_does_not_gate():
    rows = [_row(phase="late"), _row(phase="midgame", flat=True),
            _row(phase="late", near_even=True)]
    s = by_stratum_summary(rows, "top_share_increase")
    assert set(s) >= {"overall", "late", "flat_policy", "near_even"}
    # Section 7: no per-stratum acceptance gate.
    for k, v in s.items():
        if k != "overall":
            assert "verdict" not in v
