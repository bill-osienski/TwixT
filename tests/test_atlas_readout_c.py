"""Atlas Stage 4, Task 5 -- Read-out C, progressive-widening coverage.

Edge-union retention at BOTH instants (amendment 4), per-stratum and depth
bucket aggregation, the directional lag bound, lexicographic shape selection,
and the frozen three-way validation verdict.

Pure: every input is a plain dict. No reservoir, no checkpoint, no MLX.
"""
import pytest

from scripts.GPU.alphazero.atlas_readout_c import (
    GATING_INSTANT, INSTANTS, MISLEADING_INTERVENTION_BAR, REQUIRED_RATES,
    RETENTION_DEPTH1_BAR, RETENTION_ROOT_BAR, STABLE_INTERVENTION_CEILING,
    STABLE_REFERENCE_LABELS, STRATA, aggregate_shape, classify_edge_strata,
    classify_strata, edge_retention, intervention_from_snapshots,
    select_on_discovery_validate_on_selected, select_shape, static_retention,
    validation_verdict,
)
from scripts.GPU.alphazero.selection_tracer import WIDENING_SHAPES

SHAPE = ("c4a05", 4.0, 0.5)


def _priors(n, best=0):
    """Descending, so prior rank == move id + 1."""
    return {i: (1.0 if i == best else 0.5 - i * 1e-4) for i in range(n)}


def _edge(path, move, priors=None, sources=(3200, 6400)):
    return {"parent_path": path, "move": move, "depth": len(path),
            "parent_priors": priors if priors is not None else _priors(500),
            "sources": sources}


def _merged(edges=None, agreement=None):
    """`merge_reference_lines` output: the union plus the agreement report."""
    return {
        "required_edges": edges if edges is not None else [
            _edge((), 0), _edge((0,), 1, _priors(400))],
        "agreement": agreement or {
            "root": {"in_3200": True, "in_6400": True, "state": "agree"},
            "reply": {"in_3200": True, "in_6400": True, "state": "agree"},
            "two_ply": {"in_3200": False, "in_6400": False,
                        "state": "absent_both"}},
    }


def _cell(elig=200, outside=30, ft=100, ft_out=15, lagged=12, mass=80.0):
    """One tracer cell, shaped exactly like SelectionTracer.snapshot emits."""
    return {"eligible_events": elig, "outside_events": outside,
            "first_touch_events": ft, "first_touch_outside_events": ft_out,
            "lagged_first_touch_outside_events": lagged,
            "excluded_prior_mass": mass,
            "outside_rate": (outside / elig) if elig else None,
            "first_touch_outside_rate": (ft_out / ft) if ft else None,
            "mean_excluded_prior_mass": (mass / elig) if elig else None}


def _tracer(overall=None, within_forced=5, bypass=2, bypass_out=1):
    o = overall if overall is not None else _cell()
    block = {**{k: dict(o) for k in ("overall", "0", "1", "2+")},
             "forced_root_bypass_events": bypass,
             "forced_root_bypass_outside_events": bypass_out,
             "forced_root_bypass_outside_rate": ((bypass_out / bypass)
                                                 if bypass else None),
             "meaningfully_affected": (o["first_touch_outside_rate"] is not None
                                       and o["first_touch_outside_rate"] >= 0.10)}
    return {"by_shape": {n: dict(block) for n, _c, _a in WIDENING_SHAPES},
            "within_forced_events": within_forced}


def _pv(root=463, reply=90):
    return {(): root, (0,): reply}


def _snaps(boundary=None, at400=None, merged=None, pv=None):
    """The ladder's snapshots dict, verbatim -- one producer document."""
    return {"at_boundary": _tracer() if boundary is None else boundary,
            "at_400": _tracer() if at400 is None else at400,
            "reference_lines": {"at_3200": None, "at_6400": None,
                                "merged": merged or _merged()},
            "parent_visits": {"at_boundary": pv or _pv(),
                              "at_400": pv or _pv()}}


def _row(label="misleading", phase="late", flat=False, near_even=False,
         snaps=None):
    return {"snapshots": snaps or _snaps(), "label": label, "phase": phase,
            "flat_policy": flat, "near_even": near_even}


def test_frozen_bars_and_strata_are_pinned():
    assert RETENTION_ROOT_BAR == 0.95 and RETENTION_DEPTH1_BAR == 0.90
    assert MISLEADING_INTERVENTION_BAR == 0.50
    assert STABLE_INTERVENTION_CEILING == 0.25
    assert set(STRATA) == {"late", "near_even", "root_flat",
                           "locally_flat_depth1", "locally_flat_depth2"}
    assert INSTANTS == ("at_boundary", "at_400")
    assert GATING_INSTANT == "at_400"           # amendment 4: B=400 drives bars
    # An ALLOW-list: a label added later must not be admitted by default.
    assert set(STABLE_REFERENCE_LABELS) == {"misleading", "stable_negative",
                                            "ambiguous"}


def test_static_retention_uses_EFFECTIVE_parent_visits():
    """K(n) keys on completed visits, which at a warm root include I."""
    wide = static_retention(_priors(500), [80], n_at_selection=463, shape=SHAPE)
    narrow = static_retention(_priors(500), [80], n_at_selection=320, shape=SHAPE)
    assert wide["k"] > narrow["k"]
    assert wide["retained"] == 1 and narrow["retained"] == 0


def test_static_retention_of_nothing_is_None():
    assert static_retention(_priors(10), [], 400, SHAPE)["rate"] is None


def test_edge_retention_reads_the_INSTANT_parent_visit_map():
    """n comes from THAT instant's map, never a nominal budget and never the
    6,400 tree. An ABSENT path has zero visits, where K(0) = 1 admits rank 1
    only."""
    edge = _edge((), 80)                              # prior rank 81
    wide = edge_retention(edge, {(): 463}, SHAPE)     # K = 87
    narrow = edge_retention(edge, {(): 320}, SHAPE)   # K = 72
    assert wide["k"] > narrow["k"]
    assert wide["retained"] is True and narrow["retained"] is False
    absent = edge_retention(edge, {}, SHAPE)
    assert absent["n"] == 0 and absent["k"] == 1 and absent["retained"] is False
    assert edge_retention(_edge((), 0), {}, SHAPE)["retained"] is True


def test_retention_covers_the_deduplicated_union_of_edges():
    """When the deep lines disagree BOTH replies are required, so both count
    toward the depth-1 denominator. Neither is truth."""
    merged = _merged([_edge((), 0),
                      _edge((0,), 1, _priors(400), sources=(3200,)),
                      _edge((0,), 300, _priors(400), sources=(6400,))])
    a = aggregate_shape([_row(snaps=_snaps(merged=merged))], SHAPE)
    reply = a["instants"]["at_400"]["by_role"]["reply"]
    assert reply["required"] == 2                 # both replies retained
    assert reply["retained"] == 1                 # rank 301 is far outside K(90)


def test_a_retention_floor_must_pass_at_BOTH_instants():
    """Amendment 4. The hoisted number is the WORSE instant, so `>= bar` is
    exactly "passed at both"."""
    snaps = _snaps(merged=_merged([_edge((), 80)]))          # rank 81
    snaps["parent_visits"] = {"at_boundary": {(): 320},      # K = 72 -> missed
                              "at_400": {(): 463}}           # K = 87 -> retained
    a = aggregate_shape([_row(snaps=snaps)], SHAPE)
    assert a["instants"]["at_400"]["root_retention"] == 1.0
    assert a["instants"]["at_boundary"]["root_retention"] == 0.0
    assert a["root_retention"] == 0.0            # the worse one, not the better


def test_the_bars_use_the_B400_intervention_and_report_the_boundary_one():
    snaps = _snaps(boundary=_tracer(_cell(ft=100, ft_out=2, lagged=1)),
                   at400=_tracer(_cell(ft=100, ft_out=40, lagged=35)))
    a = aggregate_shape([_row(label="misleading", snaps=snaps)], SHAPE)
    assert a["gated_on"] == "at_400"
    assert a["instants"]["at_400"]["misleading_intervention"] == 1.0
    assert a["instants"]["at_boundary"]["misleading_intervention"] == 0.0
    assert a["misleading_intervention"] == 1.0          # the B=400 number


def test_a_missing_snapshot_is_NO_SNAPSHOT_not_zero():
    """A row whose boundary never fired has no snapshot. That is not an
    intervention rate of zero."""
    snaps = _snaps()
    snaps["at_boundary"] = None
    r = intervention_from_snapshots(snaps, "c4a05", instant="at_boundary")
    assert r["verdict"] == "NO_SNAPSHOT" and r["meaningfully_affected"] is None


def test_intervention_requires_the_PRODUCED_lagged_bound():
    snaps = _snaps(at400=_tracer(_cell(ft=100, ft_out=12, lagged=8)))
    r = intervention_from_snapshots(snaps, "c4a05", instant="at_400")
    assert r["meaningfully_affected"] is None      # None, not False
    assert r["verdict"] == "INCONCLUSIVE"


def test_intervention_passes_when_both_bounds_clear():
    snaps = _snaps(at400=_tracer(_cell(ft=100, ft_out=15, lagged=12)))
    r = intervention_from_snapshots(snaps, "c4a05", instant="at_400")
    assert r["meaningfully_affected"] is True and r["verdict"] == "OK"


def test_classify_strata_reads_the_row_not_a_bare_leg_list():
    s = classify_strata(_row(phase="late", flat=True, near_even=True))
    assert {"late", "root_flat", "near_even"} <= s


def test_local_flat_strata_are_EDGE_level_not_row_level():
    """A row can hold both flat and non-flat reference parents; pooling them
    would hide the contrast the stratum exists to expose."""
    flat_priors = {i: 1.0 / 500 for i in range(500)}
    assert "locally_flat_depth1" in classify_edge_strata(
        {"depth": 1, "parent_priors": flat_priors})
    assert "locally_flat_depth2" in classify_edge_strata(
        {"depth": 2, "parent_priors": flat_priors})
    assert classify_edge_strata(
        {"depth": 1, "parent_priors": {0: 0.9, 1: 0.05, 2: 0.05}}) == set()


def test_a_flat_ROOT_edge_gets_no_local_stratum():
    """Depth 0 is the root, not a local parent. An `else depth2` fallthrough
    would invent a stratum membership the edge does not have."""
    flat_priors = {i: 1.0 / 500 for i in range(500)}
    assert classify_edge_strata({"depth": 0, "parent_priors": flat_priors}) == set()
    assert classify_edge_strata({"parent_priors": flat_priors}) == set()
    assert classify_edge_strata({"depth": 7, "parent_priors": flat_priors}) == set()


def test_per_stratum_retention_uses_edge_level_flatness():
    """One row, one flat reference parent and one concentrated one. `_priors`
    is itself flat under the frozen definition -- normalized entropy ~1.0 and a
    top prior of ~0.005 -- so the non-flat case must be built explicitly."""
    flat = {i: 1.0 / 500 for i in range(500)}
    sharp = {0: 0.5, 1: 0.3, 2: 0.2}                      # NOT flat
    merged = _merged([_edge((), 0),
                      _edge((0,), 1, flat),               # locally flat, depth 1
                      _edge((0, 1), 2, sharp)])
    snaps = _snaps(merged=merged, pv={(): 463, (0,): 90, (0, 1): 20})
    a = aggregate_shape([_row(phase="late", snaps=snaps)], SHAPE)
    st = a["instants"]["at_400"]["by_stratum"]
    assert st["locally_flat_depth1"]["required"] == 1
    assert st["locally_flat_depth2"]["required"] == 0
    assert st["locally_flat_depth2"]["rate"] is None      # None, never 0.0
    assert st["late"]["required"] == 3                    # row-level: all edges


def test_aggregate_excludes_INCONCLUSIVE_rows_from_the_denominator():
    """Folding them in as either outcome would invent a measurement."""
    rows = [_row(snaps=_snaps(at400=_tracer(_cell(ft=100, ft_out=15, lagged=12)))),
            _row(snaps=_snaps(at400=_tracer(_cell(ft=100, ft_out=12, lagged=8))))]
    a = aggregate_shape(rows, SHAPE)
    assert a["misleading_denominator"] == 1
    assert a["inconclusive"] == 1


def test_aggregate_rate_is_None_when_the_denominator_empties():
    rows = [_row(snaps=_snaps(at400=_tracer(_cell(ft=100, ft_out=12, lagged=8))))]
    a = aggregate_shape(rows, SHAPE)
    assert a["misleading_intervention"] is None


def test_aggregate_reports_depth_buckets_forced_counts_and_agreement():
    """Section 8's online aggregates, pooled across the cohort."""
    a = aggregate_shape([_row(), _row()], SHAPE)
    c = a["counters"]["at_400"]
    assert set(c["by_depth"]) == {"0", "1", "2+"}
    assert c["eligible_events"] == 400                    # 2 rows x 200
    # Forced-root bypasses are reported SEPARATELY, never in the primary
    # intervention denominator.
    assert c["forced_root_bypass_events"] == 4
    assert c["forced_root_bypass_outside_rate"] == 0.5
    assert c["within_forced_events"] == 10
    # Agreement is reported and adds NO gate. The two missingness states are
    # counted separately, and neither is in the denominator.
    assert a["agreement"]["reply"]["agree_rate"] == 1.0
    assert a["agreement"]["two_ply"]["absent_both"] == 2
    assert a["agreement"]["two_ply"]["single_line"] == 0
    assert a["agreement"]["two_ply"]["agree_rate"] is None


def test_retention_bars_exclude_rows_without_a_stable_reference():
    """Section 8's floors are about STABLE deep moves. A row whose 3,200 and
    6,400 rungs never agreed has none, so it contributes no required edges --
    but its selection events still count, because those describe what widening
    would have done regardless of the label."""
    rows = [_row(label="misleading"), _row(label="no_stable_reference")]
    a = aggregate_shape(rows, SHAPE)
    at400 = a["instants"]["at_400"]
    assert at400["retention_rows"] == 1                 # not 2
    assert at400["by_role"]["root"]["required"] == 1    # one row's edges only
    assert a["rows_without_stable_reference"] == 1
    # Event counters cover EVERY row.
    assert a["counters"]["at_400"]["eligible_events"] == 400
    # ...and the excluded row is in neither intervention denominator.
    assert at400["misleading_denominator"] == 1
    assert at400["stable_denominator"] == 0


def test_excluded_prior_mass_pools_event_wise_across_rows():
    """Sum the mass and sum the events. A mean of per-row means would weight a
    10-event row the same as a 990-event one."""
    small = _snaps(at400=_tracer(_cell(elig=10, mass=1.0)))       # row mean 0.10
    large = _snaps(at400=_tracer(_cell(elig=990, mass=495.0)))    # row mean 0.50
    a = aggregate_shape([_row(snaps=small), _row(snaps=large)], SHAPE)
    # Pooled 496/1000; a mean of per-row means would have given 0.30.
    assert a["counters"]["at_400"]["mean_excluded_prior_mass"] == pytest.approx(0.496)


def test_a_shape_with_a_None_rate_cannot_pass():
    per = {"c4a05": {"root_retention": 0.99, "depth1_retention": 0.95,
                     "misleading_intervention": None, "stable_intervention": 0.10,
                     "descendant_retention": 0.90}}
    assert select_shape(per)["selected"] is None


def test_shape_selection_is_lexicographic():
    a = {"root_retention": 0.99, "depth1_retention": 0.95,
         "misleading_intervention": 0.60, "stable_intervention": 0.20,
         "descendant_retention": 0.80}
    b = dict(a, misleading_intervention=0.55, stable_intervention=0.10,
             descendant_retention=0.99)
    assert select_shape({"c4a05": a, "c13a03": b})["selected"] == "c4a05"


def test_a_retention_floor_excludes_a_shape_however_good_otherwise():
    a = {"root_retention": 0.90, "depth1_retention": 0.95,
         "misleading_intervention": 0.99, "stable_intervention": 0.01,
         "descendant_retention": 0.99}
    b = {"root_retention": 0.96, "depth1_retention": 0.91,
         "misleading_intervention": 0.51, "stable_intervention": 0.24,
         "descendant_retention": 0.70}
    assert select_shape({"c4a05": a, "c13a03": b})["selected"] == "c13a03"


def test_no_shape_passing_is_a_named_failure():
    bad = {"root_retention": 0.10, "depth1_retention": 0.10,
           "misleading_intervention": 0.99, "stable_intervention": 0.99,
           "descendant_retention": 0.10}
    r = select_shape({"c4a05": bad, "c13a03": bad})
    assert r["selected"] is None and r["verdict"] == "NO_SHAPE_PASSES"


def test_ties_break_on_descendant_retention():
    a = {"root_retention": 0.99, "depth1_retention": 0.95,
         "misleading_intervention": 0.60, "stable_intervention": 0.20,
         "descendant_retention": 0.70}
    b = dict(a, descendant_retention=0.90)
    assert select_shape({"c4a05": a, "c13a03": b})["selected"] == "c13a03"


def test_selection_happens_on_discovery_and_only_that_shape_is_validated():
    disc = [_row() for _ in range(4)]
    val = [_row() for _ in range(4)]
    r = select_on_discovery_validate_on_selected(disc, val)
    assert r["selected_on"] == "discovery"
    assert set(r["validated"]) <= {r["selected"]}      # never both shapes


# -- amendment 6a: the validation aggregate is JUDGED, not merely computed ----

def test_validation_verdict_precedence_is_FAIL_then_INCONCLUSIVE_then_PASS():
    """A DEFINED miss is evidence and outranks a gap in the evidence."""
    good = {"root_retention": 0.99, "depth1_retention": 0.95,
            "misleading_intervention": 0.60, "stable_intervention": 0.10}
    assert validation_verdict(good)["verdict"] == "PASS"
    assert validation_verdict(
        dict(good, misleading_intervention=None))["verdict"] == "INCONCLUSIVE"
    assert validation_verdict(dict(good, root_retention=0.50))["verdict"] == "FAIL"

    # BOTH at once -- the case the precedence exists for. Without an ordering
    # this result satisfies two verdicts simultaneously.
    r = validation_verdict(dict(good, root_retention=0.50,
                                misleading_intervention=None))
    assert r["verdict"] == "FAIL"
    assert r["failed"] == ["root_retention"]
    assert r["undefined"] == ["misleading_intervention"]


def test_the_ceiling_is_judged_as_a_ceiling_not_a_floor():
    good = {"root_retention": 0.99, "depth1_retention": 0.95,
            "misleading_intervention": 0.60, "stable_intervention": 0.10}
    assert validation_verdict(dict(good, stable_intervention=0.90))["verdict"] == "FAIL"


def test_the_tie_break_retention_is_not_a_required_rate():
    """descendant_retention breaks exact ties in shape selection; it is not a
    bar, so an undefined one must not turn a passing aggregate INCONCLUSIVE."""
    assert "descendant_retention" not in REQUIRED_RATES
    a = {"root_retention": 0.99, "depth1_retention": 0.95,
         "misleading_intervention": 0.60, "stable_intervention": 0.10,
         "descendant_retention": None}
    assert validation_verdict(a)["verdict"] == "PASS"


def _cohort():
    """Misleading rows that widening would intervene on, stable-negative rows
    it would leave alone -- the shape a passing feasibility result has."""
    hot = _tracer(_cell(ft=100, ft_out=40, lagged=35))
    cold = _tracer(_cell(ft=100, ft_out=2, lagged=1))
    return ([_row(label="misleading", snaps=_snaps(boundary=hot, at400=hot))
             for _ in range(2)]
            + [_row(label="stable_negative", snaps=_snaps(boundary=cold,
                                                          at400=cold))
               for _ in range(2)])


def test_the_selected_shape_receives_the_three_way_verdict():
    rows = _cohort()
    r = select_on_discovery_validate_on_selected(rows, rows)
    assert r["selected"] is not None
    assert set(r["validated"]) == {r["selected"]}
    assert r["validation_verdict"]["verdict"] == "PASS"


def test_no_selected_shape_means_no_verdict_to_give():
    """NO_SHAPE_PASSES is not an INCONCLUSIVE validation -- nothing was
    validated, so there is no validation aggregate to judge."""
    r = select_on_discovery_validate_on_selected([_row()], [_row()])
    assert r["selected"] is None and r["verdict"] == "NO_SHAPE_PASSES"
    assert r["validation_verdict"] is None
