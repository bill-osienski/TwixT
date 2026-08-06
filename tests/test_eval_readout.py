"""Frozen-rule tests for the eval readout module.

The constants under test are FROZEN (design spec §7.4, 2026-08-06). These
tests exist to make a silent drift in any of them fail loudly.
"""
import math
import random

import pytest

from scripts.GPU.alphazero import eval_readout as R


def test_frozen_constants():
    assert R.VALUE_RANGE == 2.0
    assert R.DELTA == 0.05
    assert R.MIN_CHILD_VISITS == 8


def test_hoeffding_radius_is_the_exact_frozen_formula():
    """eps(n) = R*sqrt(ln(2/delta)/(2n)), computed here from the frozen R and
    delta rather than from the spec's rounded 1.84444 display constant.

    A loose comparison against the rounded numerator would tolerate a real
    drift in R or delta; this one cannot.
    """
    for n in (1, 7, 8, 40, 100, 190, 400):
        expected = R.VALUE_RANGE * math.sqrt(
            math.log(2.0 / R.DELTA) / (2.0 * n))
        assert R.hoeffding_radius(n) == pytest.approx(expected, rel=1e-12)


def test_the_rounded_display_constant_still_describes_the_formula():
    # 1.84444 is what the spec prints; it must remain a faithful 5dp rounding
    # of ln(2/delta)/2, or the spec text has drifted from the code.
    assert math.log(2.0 / R.DELTA) / 2.0 == pytest.approx(1.84444, abs=5e-6)


def test_hoeffding_worked_magnitudes():
    # Exact values, tighter than the spec's 3dp display figures. The spec
    # originally printed eps(40) as 0.430; the true value is 0.42947, which
    # rounds to 0.429. Corrected there; the RULE is unchanged.
    assert R.hoeffding_radius(190) == pytest.approx(0.19705, abs=5e-5)
    assert R.hoeffding_radius(100) == pytest.approx(0.27162, abs=5e-5)
    assert R.hoeffding_radius(40) == pytest.approx(0.42947, abs=5e-5)
    assert R.hoeffding_radius(8) == pytest.approx(0.96032, abs=5e-5)


def test_the_worked_override_gap_is_what_the_spec_states():
    # A 190-visit leader vs a 40-visit challenger: the challenger must exceed
    # the leader by this much in root-perspective Q to override.
    gap = R.hoeffding_radius(40) - R.hoeffding_radius(190)
    assert gap == pytest.approx(0.232, abs=5e-4)


def test_min_child_visits_is_the_boundary_of_the_frozen_requirement():
    # n_min follows from the preregistered requirement eps(n) <= 1.0.
    # Constructed boundary: 8 satisfies it, 7 does not.
    assert R.hoeffding_radius(R.MIN_CHILD_VISITS) <= 1.0
    assert R.hoeffding_radius(R.MIN_CHILD_VISITS - 1) > 1.0


def test_hoeffding_radius_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        R.hoeffding_radius(0)


def test_top_two_orders_by_visits_then_canonical_move():
    stats = {(1, 1): (10, 0.1), (2, 2): (50, -0.2), (0, 5): (10, 0.3)}
    t2 = R.top_two(stats)
    assert [c.move for c in t2] == [(2, 2), (0, 5)]  # tie 10/10 -> (0,5) first


def test_top_two_maps_zero_visit_children_to_none_not_zero():
    stats = {(1, 1): (0, 0.0), (2, 2): (50, -0.2)}
    t2 = R.top_two(stats)
    zero = [c for c in t2 if c.move == (1, 1)][0]
    assert zero.q_child is None
    assert zero.q_root is None


def test_root_perspective_is_the_negation_of_child_perspective():
    stats = {(1, 1): (30, 0.25), (2, 2): (50, -0.20)}
    t2 = R.top_two(stats)
    for c in t2:
        assert c.q_root == pytest.approx(-c.q_child)


def test_lcb_override_fires_when_challenger_lcb_is_higher():
    # leader 190 visits, q_root -0.30 -> LCB -0.497
    # challenger 40 visits, q_root  0.00 -> LCB -0.429  (higher -> override)
    top2 = [
        R.ChildStat((2, 2), 190, 0.30, -0.30),
        R.ChildStat((1, 1), 40, -0.00, 0.00),
    ]
    assert R.lcb_override(top2) == (1, 1)


def test_lcb_override_declines_when_the_gap_is_too_small():
    # Same visits; challenger only 0.10 better, needs > 0.232.
    top2 = [
        R.ChildStat((2, 2), 190, 0.30, -0.30),
        R.ChildStat((1, 1), 40, -0.20, -0.20),
    ]
    assert R.lcb_override(top2) is None


def test_lcb_override_declines_below_min_visits():
    top2 = [
        R.ChildStat((2, 2), 190, 0.30, -0.30),
        R.ChildStat((1, 1), 7, -0.90, 0.90),
    ]
    assert R.lcb_override(top2) is None


def test_lcb_override_declines_on_undefined_q():
    top2 = [
        R.ChildStat((2, 2), 190, None, None),
        R.ChildStat((1, 1), 40, -0.90, 0.90),
    ]
    assert R.lcb_override(top2) is None


def test_lcb_override_declines_with_fewer_than_two_children():
    assert R.lcb_override([R.ChildStat((2, 2), 190, 0.3, -0.3)]) is None
    assert R.lcb_override([]) is None


def test_argmax_mode_is_deterministic_and_ignores_rng():
    counts = {(1, 1): 10, (2, 2): 50, (0, 5): 10}
    cfg = R.ReadoutConfig(mode=R.MODE_ARGMAX)
    for _ in range(5):
        move, overrode = R.select(counts, ply=3, readout=cfg, rng=random.Random(1))
        assert move == (2, 2)
        assert overrode is False


def test_argmax_ties_break_in_canonical_numeric_order():
    counts = {(2, 2): 50, (0, 5): 50}
    cfg = R.ReadoutConfig(mode=R.MODE_ARGMAX)
    move, _ = R.select(counts, ply=3, readout=cfg, rng=random.Random(1))
    assert move == (0, 5)


def test_opening_temperature_mode_samples_early_and_argmaxes_late_when_temp_low_is_zero():
    counts = {(1, 1): 10, (2, 2): 50}
    cfg = R.ReadoutConfig(mode=R.MODE_OPENING_TEMPERATURE, temp_high=1.0, temp_low=0.0)
    late, _ = R.select(counts, ply=20, readout=cfg, rng=random.Random(1))
    assert late == (2, 2)
    seen = {R.select(counts, ply=0, readout=cfg, rng=random.Random(s))[0]
            for s in range(40)}
    assert seen == {(1, 1), (2, 2)}  # opening genuinely samples both


def test_hoeffding_mode_samples_the_opening_then_overrides_post_opening():
    counts = {(1, 1): 40, (2, 2): 190}
    top2 = [
        R.ChildStat((2, 2), 190, 0.30, -0.30),
        R.ChildStat((1, 1), 40, -0.00, 0.00),
    ]
    cfg = R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB, temp_high=1.0)
    move, overrode = R.select(counts, ply=20, readout=cfg,
                              rng=random.Random(1), top2=top2)
    assert move == (1, 1)
    assert overrode is True


def test_hoeffding_mode_reports_no_override_when_the_rule_declines():
    counts = {(1, 1): 40, (2, 2): 190}
    top2 = [
        R.ChildStat((2, 2), 190, 0.30, -0.30),
        R.ChildStat((1, 1), 40, -0.20, -0.20),
    ]
    cfg = R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB, temp_high=1.0)
    move, overrode = R.select(counts, ply=20, readout=cfg,
                              rng=random.Random(1), top2=top2)
    assert move == (2, 2)
    assert overrode is False


def test_hoeffding_mode_requires_top2_post_opening():
    cfg = R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB)
    with pytest.raises(ValueError):
        R.select({(1, 1): 5}, ply=20, readout=cfg, rng=random.Random(1))


def test_unknown_mode_is_rejected_at_construction():
    with pytest.raises(ValueError):
        R.ReadoutConfig(mode="wishful")


def test_select_rejects_empty_counts():
    cfg = R.ReadoutConfig(mode=R.MODE_ARGMAX)
    with pytest.raises(ValueError):
        R.select({}, ply=0, readout=cfg, rng=random.Random(1))
