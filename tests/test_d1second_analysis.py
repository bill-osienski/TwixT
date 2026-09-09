"""D1'' search-suppression analysis -- the FROZEN plan
(docs/superpowers/2026-09-08-t1j-d1second-search-readout-proposal.md, as amended
at 0b2934c) implemented against SYNTHETIC fixtures only.

🔴 NOTHING HERE READS THE ACQUIRED D1 RECORD. No value of `raw_policy`,
`root_visits`, `selected_visit_rank` or `readout_overrode_leader` from
2026-09-08-t1j-d1-acquisition is inspected, and no D1'' quantity is computed from
it. The disjointness precondition is implemented and tested on synthetic rows;
running it on the real record is a separate authorization.

No model, no JVM, no query, no game, no seed, no registry edit, no gate change.
The only randomness is the plan's analysis-only PRNG (PCG64, seed 20260908) on
synthetic fixtures.
"""
import inspect

import pytest

from scripts.GPU.alphazero import d1prime_analysis as DP
from scripts.GPU.alphazero import d1second_analysis as DS

from tests.test_d1prime_analysis import (            # synthetic fixtures only
    _cohort, _d1_position, _design, _policy_with_ranks, _strata, canonical,
    _complete_report, _report_for, _positions_cached, _canonical_cached, OPENINGS, PHASES, ARMS)


# ────────────────────────── the frozen constants ────────────────────────────

def test_the_frozen_constants_match_the_plan_and_are_BOUND_not_retyped():
    """Every shared constant is the SAME OBJECT as D1''s, so the two analyses
    cannot drift apart silently. Only the bootstrap seed differs, and it differs
    on purpose."""
    assert DS.K_SS is DP.K_LPRD == 5
    assert DS.T_THRESHOLD is DP.T_THRESHOLD == 0.15
    assert DS.B_REPLICATES is DP.B_REPLICATES == 10_000
    assert DS.FLOOR is DP.FLOOR
    assert DS.PRIMARY_COHORT is DP.PRIMARY_COHORT
    assert DS.INDICATOR == "ss"
    # 🔑 A DIFFERENT SEED, deliberately: reusing 20260907 would make D1'''s
    # replicate draw a deterministic repeat of D1''s over the same games.
    assert DS.BOOTSTRAP_SEED == 20260908 != DP.BOOTSTRAP_SEED


# ─────────────────────────────── rank_visit ─────────────────────────────────

def test_rank_visit_orders_by_visits_then_breaks_ties_by_row_col_only():
    ranks = DS.rank_visit({(0, 0): 3, (0, 1): 9, (1, 0): 3})
    assert ranks[(0, 1)] == 1                       # most visits leads
    assert ranks[(0, 0)] == 2 and ranks[(1, 0)] == 3  # tie -> ascending (row, col)


def test_rank_visit_takes_the_VISITS_AND_NOTHING_ELSE():
    """🔑 THE POINT OF THE WHOLE METRIC. If the visit ranking could see the
    policy, it would inherit the ordering it is being compared against and `ss`
    would measure agreement with itself. The signature is the guarantee."""
    params = list(inspect.signature(DS.rank_visit).parameters)
    assert params == ["visits"], params


def test_rank_visit_ranks_zero_visit_moves_and_does_not_drop_them():
    """At 400 simulations most legal moves get zero visits. They are RANKED, not
    discarded: dropping them would make `rank_visit` undefined exactly where the
    metric needs it."""
    ranks = DS.rank_visit({(0, 0): 5, (2, 2): 0, (1, 1): 0})
    assert ranks[(0, 0)] == 1
    assert ranks[(1, 1)] == 2 and ranks[(2, 2)] == 3
    assert len(ranks) == 3


def test_rank_visit_handles_a_single_legal_move_root():
    assert DS.rank_visit({(4, 4): 400}) == {(4, 4): 1}


def test_rank_visit_refuses_an_empty_root():
    with pytest.raises(DS.D1SecondError, match="no legal move"):
        DS.rank_visit({})


# ───────────────────────── the ss indicator itself ──────────────────────────

def _pos(*, t1j_rank, t1j_visit_rank, n=10, **labels):
    """A position whose T1j move sits at `t1j_rank` by policy and
    `t1j_visit_rank` by visits, both constructed explicitly."""
    pol = _policy_with_ranks(n)                      # rank == col + 1
    t1j = (0, t1j_rank - 1)
    # visits: give the move at visit-rank r the count (n - r); ties impossible
    order = [(0, i) for i in range(n)]
    order.remove(t1j)
    order.insert(t1j_visit_rank - 1, t1j)
    visits = {f"{m[0]},{m[1]}": n - i for i, m in enumerate(order)}
    return _d1_position(policy=pol, our_move=(0, 0), t1j_move_6=t1j, visits=visits, **labels)


def test_ss_is_true_only_when_the_policy_ranks_it_top_five_and_the_visits_do_not():
    assert DS.suppression_row(_pos(t1j_rank=3, t1j_visit_rank=7))["ss"] is True
    assert DS.suppression_row(_pos(t1j_rank=3, t1j_visit_rank=2))["ss"] is False   # search kept it
    assert DS.suppression_row(_pos(t1j_rank=8, t1j_visit_rank=9))["ss"] is False   # policy never had it


@pytest.mark.parametrize("rank,visit_rank,ss", [(5, 6, True), (5, 5, False), (6, 7, False)])
def test_ss_is_inclusive_at_five_on_the_policy_side_and_exclusive_on_the_visit_side(
        rank, visit_rank, ss):
    """AT the boundary, both sides. `rank_raw <= 5` and `rank_visit > 5`."""
    assert DS.suppression_row(_pos(t1j_rank=rank, t1j_visit_rank=visit_rank))["ss"] is ss


def test_ss_and_lprd_are_DISJOINT_by_construction_over_every_rank_combination():
    """The plan's central claim about this metric, checked exhaustively over the
    ranks a 10-move root can produce -- not asserted in prose."""
    for rank in range(1, 11):
        for visit_rank in range(1, 11):
            row = DS.suppression_row(_pos(t1j_rank=rank, t1j_visit_rank=visit_rank))
            assert not (row["ss"] and row["lprd"]), (rank, visit_rank, row)


def test_the_row_carries_the_D1prime_fields_too_so_the_two_analyses_share_labels():
    row = DS.suppression_row(_pos(t1j_rank=3, t1j_visit_rank=7))
    for f in DP.COHORT_BINDING_FIELDS:
        assert f in row
    assert row["rank_t1j"] == 3 and row["rank_visit_t1j"] == 7


def test_the_strict_variant_needs_no_tiebreak_and_fires_only_on_zero_visits():
    pol = _policy_with_ranks(6)
    zero = _d1_position(policy=pol, our_move=(0, 0), t1j_move_6=(0, 2),
                        visits={"0,0": 400, "0,1": 0, "0,2": 0, "0,3": 0, "0,4": 0, "0,5": 0})
    assert DS.suppression_row(zero)["ss0"] is True
    # 🔴 MY FIRST VERSION OF THIS ASSERTED THE WRONG THING and the code was right:
    # in a SIX-move root, a move with 4 visits ranks SECOND, so it is not
    # suppressed by rank at all. `ss` needs enough better-visited moves to push it
    # past fifth, which takes a bigger root -- so the case is built that way.
    pol10 = _policy_with_ranks(10)
    some = _d1_position(policy=pol10, our_move=(0, 0), t1j_move_6=(0, 2),
                        visits={"0,0": 100, "0,1": 90, "0,3": 80, "0,4": 70, "0,5": 60,
                                "0,6": 50, "0,2": 1, "0,7": 0, "0,8": 0, "0,9": 0})
    row = DS.suppression_row(some)
    assert row["ss0"] is False                 # it WAS visited, once
    assert row["rank_visit_t1j"] == 7 and row["rank_t1j"] == 3
    assert row["ss"] is True                   # still suppressed, by rank


def test_a_position_whose_visits_do_not_cover_the_policy_is_REFUSED():
    """D1 VOIDs such a record at acquisition; the analysis refuses it too rather
    than ranking over a different legal set than it scored."""
    p = _pos(t1j_rank=3, t1j_visit_rank=7)
    p["incumbent"]["root_visits"].pop(next(iter(p["incumbent"]["root_visits"])))
    with pytest.raises(DS.D1SecondError, match="same legal set"):
        DS.suppression_row(p)


def test_a_position_with_no_root_visits_at_all_is_REFUSED_not_scored_as_False():
    """The refusal has ONE owner -- `rank_visit` -- and the message is asserted,
    so a second guard cannot be added that no control can tell apart."""
    p = _pos(t1j_rank=3, t1j_visit_rank=7)
    p["incumbent"]["root_visits"] = {}
    with pytest.raises(DS.D1SecondError, match="no legal move"):
        DS.suppression_row(p)


# ───────────── the shared statistic, now indicator-parameterised ────────────

def test_the_matched_statistic_reads_the_REQUESTED_indicator():
    """D1'' reuses D1''s statistic through an explicit indicator, so the two
    analyses cannot silently score each other's field."""
    rows = [{"task_id": "g", "opening": "o1_center", "colour_arm": "t1j_red", "phase": "middle",
             "signature": "mover_fragmentation", "role": role, "lprd": lprd, "ss": ss}
            for role, lprd, ss in (("position", True, False), ("control", False, True))]
    assert DP.matched_statistic(rows, cohort="mover_fragmentation")["T"] == 1.0
    assert DP.matched_statistic(rows, cohort="mover_fragmentation", indicator="ss")["T"] == -1.0


def test_the_indicator_DEFAULTS_to_lprd_so_D1primes_result_is_untouched():
    cells = [(o, a, p) for o in OPENINGS[:4] for a in ARMS for p in PHASES[:2]]
    rows = _cohort(cells, pos_rate=1 / 3, ctl_rate=0.0)
    assert DP.matched_statistic(rows, cohort="mover_fragmentation") == \
        DP.matched_statistic(rows, cohort="mover_fragmentation", indicator="lprd")


def test_the_statistic_names_its_rates_after_the_indicator():
    rows = _cohort([(OPENINGS[0], ARMS[0], PHASES[0])], pos_rate=1.0, ctl_rate=0.0)
    for r in rows:
        r["ss"] = r["lprd"]
    out = DP.matched_statistic(rows, cohort="mover_fragmentation", indicator="ss")
    assert "ss_positions" in out["cells"][0] and "ss_controls" in out["cells"][0]
    assert "lprd_positions" not in out["cells"][0]


def test_a_row_missing_the_indicator_field_is_REFUSED_not_read_as_False():
    """🔴 `pytest.raises(Exception)` did not discriminate: with the guard removed
    the row lookup raises KeyError, which is also an Exception, so the control
    was NOT CAUGHT. The refusal must be the NAMED one, with the field in it."""
    rows = _cohort([(OPENINGS[0], ARMS[0], PHASES[0])], pos_rate=1.0, ctl_rate=0.0)
    with pytest.raises(DP.D1PrimeError, match="no 'ss' field"):
        DP.matched_statistic(rows, cohort="mover_fragmentation", indicator="ss")


# ──────────────────────── the decision rule, on ss ──────────────────────────

def _ss_cohort(cells, *, pos_rate, ctl_rate, per_role=3):
    rows = _cohort(cells, pos_rate=pos_rate, ctl_rate=ctl_rate, per_role=per_role)
    for r in rows:                       # the synthetic indicator IS ss here
        r["ss"] = r.pop("lprd")
        r["lprd"] = False                # disjointness holds in the fixture too
        # The module REFUSES a row missing a field it reports on, rather than
        # scoring the absence as False -- so the fixture carries them, and the
        # secondary test overrides them to prove they change no outcome.
        r.setdefault("ss0", False)
        r.setdefault("overrode_leader", False)
        # The frozen readout summary histograms these and counts the complement,
        # and REFUSES a row that omits any of them. `rank_t1j = 3` keeps the
        # fixture internally consistent: <= K, so `lprd` is False as set above.
        r.setdefault("selected_visit_rank", 1)
        r.setdefault("selected_policy_rank", 1)
        r.setdefault("rank_t1j", 3)
    return rows


def _decide(rows, **kw):
    return DS._suppression_decision(rows, _strata(), cohort=DS.PRIMARY_COHORT,
                                    B=kw.pop("B", 200), seed=kw.pop("seed", DS.BOOTSTRAP_SEED))


ALL_CELLS = [(o, a, p) for o in OPENINGS for a in ARMS for p in PHASES[:2]]


def test_a_symmetric_synthetic_cohort_is_NO_GO():
    out = _decide(_ss_cohort(ALL_CELLS, pos_rate=1 / 3, ctl_rate=1 / 3))
    assert out["outcome"] == "NO_GO" and abs(out["T"]) < 1e-12


def test_an_injected_suppression_excess_of_one_third_is_GO():
    out = _decide(_ss_cohort(ALL_CELLS, pos_rate=1.0, ctl_rate=2 / 3))
    assert out["outcome"] == "GO" and out["T"] > DS.T_THRESHOLD
    assert out["interval"][0] > 0


def test_the_same_excess_at_CONTROLS_is_NO_GO_direction_binds():
    out = _decide(_ss_cohort(ALL_CELLS, pos_rate=2 / 3, ctl_rate=1.0))
    assert out["outcome"] == "NO_GO" and out["T"] < 0


def test_insufficient_support_is_its_own_NO_GO_and_computes_no_interval():
    out = _decide(_ss_cohort(ALL_CELLS[:4], pos_rate=1.0, ctl_rate=0.0, per_role=2))
    assert out["outcome"] == "NO_GO — insufficient support"
    assert out["interval"] is None


def test_the_secondary_reports_are_present_and_CANNOT_produce_GO():
    """The strict variant, the readout summary and the second cohort are
    reported. None of them is consulted by the decision."""
    rows = _ss_cohort(ALL_CELLS, pos_rate=1 / 3, ctl_rate=1 / 3)
    for r in rows:                     # a huge strict-variant and readout signal
        r["ss0"] = r["role"] == "position"
        r["overrode_leader"] = r["role"] == "position"
    out = _decide(rows)
    assert out["outcome"] == "NO_GO"                        # unmoved
    assert out["secondary"]["ss0"]["T"] > 0.9
    # the frozen shape is a COUNT histogram, not a rate: every position row
    assert out["secondary"]["readout"]["position"]["overrode_leader"] == \
        [[False, 0], [True, out["secondary"]["readout"]["position"]["n"]]]
    assert "created_threat" in out["secondary"]


# ─────────────── the disjointness PRECONDITION, at execution time ───────────

def test_the_disjointness_precondition_accepts_rows_that_obey_it():
    DS.check_disjoint(_ss_cohort(ALL_CELLS, pos_rate=1 / 3, ctl_rate=1 / 3))


def test_the_disjointness_precondition_REFUSES_a_row_scoring_BOTH():
    """NEGATIVE CONTROL. `ss` requires rank_raw <= 5 and `lprd` requires
    rank_raw > 5, so a row with both did not come from these definitions -- and
    the statistic would silently double-count it."""
    rows = _ss_cohort(ALL_CELLS, pos_rate=1 / 3, ctl_rate=1 / 3)
    rows[0]["ss"] = rows[0]["lprd"] = True
    with pytest.raises(DS.D1SecondError, match="disjoint"):
        DS.check_disjoint(rows)


def test_the_public_entry_RUNS_the_disjointness_precondition(monkeypatch, canonical):
    """It is a precondition of the ANALYSIS, not a test-only assertion: the entry
    must call it, and a forged row must stop the run."""
    called = {}
    real = DS.check_disjoint
    monkeypatch.setattr(DS, "check_disjoint", lambda rows: called.setdefault("n", len(rows)) or real(rows))
    DS.analyse_search_suppression(_ss_report(canonical))
    assert called["n"] == 221


# ────────────────────────── the checked public entry ────────────────────────

def _ss_report(canon, **override):
    """A COMPLETE D1 report whose readouts carry visits summing to the FROZEN
    simulation budget, so `ss` is defined and the record describes the search the
    hypothesis is about. The D1′ fixtures predate the budget rule and give one
    visit per move, which D1″ now refuses -- correctly."""
    return _budgeted(_complete_report(canon, **override))


def test_the_public_entry_takes_the_REPORT_AND_NOTHING_ELSE():
    """🔴 No rows, no cohort, no tasks, no reps, no B, no seed. Every one of
    those was a knob a forged input could turn."""
    params = list(inspect.signature(DS.analyse_search_suppression).parameters)
    assert params == ["d1_report"], params


def test_the_entry_resolves_the_canonical_cohort_and_the_frozen_prng(canonical):
    out = DS.analyse_search_suppression(_ss_report(canonical))
    assert out["resolved"]["n_positions"] == 221
    assert out["resolved"]["cohort_source"] == "d1_selection.select_all"
    assert out["prng"] == {"bit_generator": "PCG64", "seed": 20260908, "B": 10_000}
    assert out["indicator"] == "ss"


def test_the_entry_refuses_a_report_that_is_not_a_completed_D1_run(canonical):
    """🔴 MY FIRST VERSION PASSED FOR THE WRONG REASON and a control proved it:
    with `{"positions": []}`, deleting the contract check entirely still raised,
    because a zero-row cohort fails the LATER cohort binding. The report below is
    a COMPLETE 221-row one missing exactly one acquisition field, so only the
    contract can refuse it -- and the message must name that field."""
    rep = _ss_report(canonical)
    rep.pop("queries_spent")
    with pytest.raises(DP.D1PrimeError, match="queries_spent"):
        DS.analyse_search_suppression(rep)


def test_the_entry_reports_the_acquisition_it_analysed(canonical):
    acq = DS.analyse_search_suppression(_ss_report(canonical))["acquisition"]
    assert acq["n_positions"] == 221 and acq["queries_spent"] == 1105
    assert acq["seed_interval"] == [202615000, 202615221]


def test_the_claim_and_meaning_are_carried_in_the_OUTPUT_not_only_the_plan(canonical):
    out = DS.analyse_search_suppression(_ss_report(canonical))
    assert "stability interval" in out["claim"] and "not" in out["claim"]
    assert "confirmation" in out["meaning"]


# ─────────────────── named refusals, never a silent zero ────────────────────

def test_an_undefined_statistic_is_NAMED_and_never_scored_as_zero():
    """No cell with both roles means T does not exist. It must be None under a
    named outcome -- a zero would read as "no suppression, measured"."""
    rows = _ss_cohort([(o, a, PHASES[0]) for o in OPENINGS for a in ARMS],
                      pos_rate=1.0, ctl_rate=0.0)
    rows = [r for r in rows if r["role"] == "position"]      # controls removed
    out = _decide(rows)
    assert out["outcome"] == "NO_GO — insufficient support"
    assert out["T"] is None and out["interval"] is None
    assert out["floor"]["cells"] == 0


def test_a_replicate_that_cannot_be_scored_makes_the_interval_UNDEFINED_by_name():
    """One cell carried by one game: some replicates lose a role entirely, and an
    undefined replicate is counted and kept, never dropped or imputed."""
    cells = [(o, a, p) for o in OPENINGS for a in ARMS for p in PHASES[:2]]
    rows = _ss_cohort(cells, pos_rate=1.0, ctl_rate=0.0)
    keep = {r["task_id"] for r in rows if r["role"] == "control"}
    rows = [r for r in rows
            if r["role"] == "position" or r["task_id"] == sorted(keep)[0]]
    out = _decide(rows)
    assert out["outcome"].startswith("NO_GO")
    if out["interval"] is not None:
        assert out["interval"] == "UNDEFINED" or out["stability"]["undefined"] >= 0


def test_the_readout_summary_REFUSES_a_row_that_does_not_carry_the_field():
    rows = _ss_cohort([(OPENINGS[0], ARMS[0], PHASES[0])], pos_rate=1.0, ctl_rate=0.0)
    del rows[0]["overrode_leader"]
    with pytest.raises(DS.D1SecondError, match="overrode_leader"):
        DS._readout_summary(rows, cohort=DS.PRIMARY_COHORT)


def test_the_entry_refuses_a_row_whose_LABEL_disagrees_with_the_frozen_manifest(canonical):
    """The rows are bound to the frozen cohort BEFORE anything is computed, so a
    relabelled control cannot sit in the wrong cell."""
    rep = _ss_report(canonical)
    rep["positions"][0]["role"] = ("control" if rep["positions"][0]["role"] == "position"
                                   else "position")
    with pytest.raises(DP.D1PrimeError):
        DS.analyse_search_suppression(rep)


def test_ss_and_lprd_are_disjoint_across_a_WHOLE_synthetic_report(canonical):
    """The precondition, exercised end to end on 221 synthetic rows. The same
    check runs on the real record when the analysis is authorized to."""
    rows = DS.rows_from_d1_report(_ss_report(canonical))
    assert len(rows) == 221
    assert not [r for r in rows if r["ss"] and r["lprd"]]
    DS.check_disjoint(rows)


def test_the_module_reaches_NO_executing_or_registry_machinery():
    """🔴 A DIAGNOSTIC THAT CAN ACQUIRE IS NOT A DIAGNOSTIC. The source may not
    name the evaluator loader, a subprocess, a seed registry, or a gate."""
    import pathlib
    src = pathlib.Path(DS.__file__).read_text()
    for forbidden in ("subprocess", "_default_load_evaluator", "ACCOUNTED_SEED_INTERVALS",
                      "EXPOSED_SEED_INTERVALS", "AUTHORIZED", "t1j_toolchain",
                      "e4_screen_reference", "compile_helper"):
        assert forbidden not in src, forbidden


def test_the_module_hardcodes_no_path_to_the_acquired_record():
    """The record is passed IN. A module that knows where the record lives can
    read it without being asked, and the plan forbids that until authorized."""
    import pathlib
    src = pathlib.Path(DS.__file__).read_text()
    assert "2026-09-08-t1j-d1-acquisition" not in src
    assert "open(" not in src and "read_text" not in src


# ═══════ pre-execution repair 1: a full-record validation pass, FIRST ════════

def _budgeted(rep):
    """Rewrite every readout so the root visits sum to the FROZEN simulation
    budget, through the REAL writer (`eval_replay.ply_record`), so every derived
    observable stays consistent. The D1′ fixtures predate the budget rule and
    give one visit per move; a D1″ record must spend the whole search.
    """
    from tests.test_d1prime_analysis import _incumbent_record
    budget = DS.frozen_sim_budget()
    for pos in rep["positions"]:
        inc = pos["incumbent"]
        policy = inc["raw_policy"]
        ours = (inc["row"], inc["col"])
        others = [k for k in policy if _mv(k) != ours]
        counts = {f"{ours[0]},{ours[1]}": budget - len(others)}
        counts.update({k: 1 for k in others})
        pos["incumbent"] = {**_incumbent_record(policy=policy, our_move=ours, visits=counts),
                            "seed": inc.get("seed", 1), "streams": inc.get("streams", {})}
    return rep


def _mv(key):
    r, c = str(key).split(",")
    return (int(r), int(c))


def _valid_report(canon):
    """A complete, well-formed report -- the baseline every corruption below
    starts from, so each test changes exactly one thing."""
    return _ss_report(canon)


def _corrupt(canon, **changes):
    rep = _valid_report(canon)
    rep["positions"][3]["incumbent"].update(changes)
    return rep


def test_the_validation_pass_accepts_a_well_formed_record(canonical):
    DS.validate_record(_valid_report(canonical))


def test_validation_runs_over_the_WHOLE_record_BEFORE_a_single_row_is_scored(canonical, monkeypatch):
    """🔑 A record that fails at row 200 must fail before row 0 is scored, so a
    partially computed analysis never exists."""
    rep = _valid_report(canonical)
    rep["positions"][-1]["incumbent"]["n_legal"] = 99
    monkeypatch.setattr(DS, "suppression_row",
                        lambda p: pytest.fail("a row was scored before validation finished"))
    with pytest.raises(DS.D1SecondError, match="n_legal"):
        DS.analyse_search_suppression(rep)


@pytest.mark.parametrize("field", [
    "raw_policy", "root_visits", "n_legal", "root_total_visits", "selected_visit_rank",
    "selected_visit_count", "selected_policy_rank", "selected_policy_mass",
    "root_top1_share", "readout_overrode_leader"])
def test_every_required_observable_is_REQUIRED_by_name(canonical, field):
    rep = _valid_report(canonical)
    rep["positions"][3]["incumbent"].pop(field)
    with pytest.raises(DS.D1SecondError, match=field):
        DS.validate_record(rep)


def test_a_BOOLEAN_visit_count_is_refused_because_True_is_not_one_visit(canonical):
    rep = _valid_report(canonical)
    inc = rep["positions"][3]["incumbent"]
    inc["root_visits"] = {k: (True if i == 0 else v)
                          for i, (k, v) in enumerate(inc["root_visits"].items())}
    with pytest.raises(DS.D1SecondError, match="root_visits"):
        DS.validate_record(rep)


@pytest.mark.parametrize("bad", [-1, 2.5, "3"])
def test_a_visit_count_that_is_not_a_NON_NEGATIVE_INT_is_refused(canonical, bad):
    rep = _valid_report(canonical)
    inc = rep["positions"][3]["incumbent"]
    k = next(iter(inc["root_visits"]))
    inc["root_visits"] = {**inc["root_visits"], k: bad}
    with pytest.raises(DS.D1SecondError, match="root_visits"):
        DS.validate_record(rep)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.5, True])
def test_a_policy_mass_that_is_not_a_FINITE_NON_NEGATIVE_REAL_is_refused(canonical, bad):
    rep = _valid_report(canonical)
    inc = rep["positions"][3]["incumbent"]
    k = next(iter(inc["raw_policy"]))
    inc["raw_policy"] = {**inc["raw_policy"], k: bad}
    with pytest.raises(DS.D1SecondError, match="raw_policy"):
        DS.validate_record(rep)


def test_an_all_zero_policy_is_refused_because_its_ranking_would_be_arbitrary(canonical):
    rep = _valid_report(canonical)
    inc = rep["positions"][3]["incumbent"]
    inc["raw_policy"] = {k: 0.0 for k in inc["raw_policy"]}
    with pytest.raises(DS.D1SecondError, match="raw_policy"):
        DS.validate_record(rep)


def test_an_empty_root_is_refused_by_the_validation_pass_too(canonical):
    with pytest.raises(DS.D1SecondError, match="raw_policy|root_visits"):
        DS.validate_record(_corrupt(canonical, raw_policy={}, root_visits={}))


@pytest.mark.parametrize("field,value", [
    ("n_legal", 99), ("root_total_visits", 4242), ("selected_visit_count", 77),
    ("selected_visit_rank", 9), ("selected_policy_rank", 9), ("selected_policy_mass", 0.99),
    ("root_top1_share", 0.99)])
def test_an_observable_that_CONTRADICTS_the_maps_it_describes_is_refused(canonical, field, value):
    """PRESENCE IS NOT AGREEMENT. Each of these is recomputed from `raw_policy`
    and `root_visits` and must match what the record claims."""
    with pytest.raises(DS.D1SecondError, match=field):
        DS.validate_record(_corrupt(canonical, **{field: value}))


def test_a_BOOLEAN_rank_is_refused_even_when_it_EQUALS_the_right_number(canonical):
    """🔴 A control proved the earlier test blind here: it changed the value to a
    WRONG one, which a loose `got != want` also refuses. `True == 1` is the case
    only a type-strict comparison catches, and rank 1 is the common case."""
    rep = _valid_report(canonical)
    inc = rep["positions"][3]["incumbent"]
    assert inc["selected_visit_rank"] == 1                  # so True would compare equal
    inc["selected_visit_rank"] = True
    with pytest.raises(DS.D1SecondError, match="selected_visit_rank"):
        DS.validate_record(rep)


def test_a_non_boolean_override_flag_is_refused(canonical):
    with pytest.raises(DS.D1SecondError, match="readout_overrode_leader"):
        DS.validate_record(_corrupt(canonical, readout_overrode_leader=1))


def test_the_policy_and_the_visits_must_cover_the_SAME_legal_set(canonical):
    rep = _valid_report(canonical)
    inc = rep["positions"][3]["incumbent"]
    inc["root_visits"] = {**inc["root_visits"], "23,23": 0}
    with pytest.raises(DS.D1SecondError, match="same legal set"):
        DS.validate_record(rep)


def test_a_depth6_move_that_is_not_a_PAIR_OF_INTS_is_refused(canonical):
    rep = _valid_report(canonical)
    # 🔴 The move below is ALSO illegal, so matching "depth" passed even with the
    # type check deleted -- the legality message says "depth-6" too. The TYPE
    # refusal must be named.
    rep["positions"][3]["depths"] = [{"depth": 6, "move": ["0", "0"]}]
    with pytest.raises(DS.D1SecondError, match="pair of ints"):
        DS.validate_record(rep)


# ════ pre-execution repair 2: the complement is DESCRIPTIVE, not a floor ════

def test_a_TINY_complement_yields_an_ORDINARY_NO_GO_not_insufficient_support():
    """🔴 THE PLAN SAID THE OPPOSITE AND WAS WRONG. The floors count rows in
    common-support cells over the unconditional denominator; they never count
    rows with rank_raw <= 5. A cohort with ample rows and almost no complement
    passes every floor, and a complement that small mechanically holds `ss` near
    zero -- which is an ordinary NO_GO."""
    rows = _ss_cohort(ALL_CELLS, pos_rate=0.0, ctl_rate=0.0)
    for i, r in enumerate(rows):
        r["rank_t1j"] = 3 if i == 0 else 9        # one single eligible row
        r["ss"] = i == 0
    out = _decide(rows)
    assert out["outcome"] == "NO_GO"
    assert out["outcome"] != "NO_GO — insufficient support"
    assert out["floor"]["positions"] >= DS.FLOOR["positions"]


def test_the_complement_is_REPORTED_per_role_and_decides_nothing():
    rows = _ss_cohort(ALL_CELLS, pos_rate=1 / 3, ctl_rate=1 / 3)
    for i, r in enumerate(rows):
        r["rank_t1j"] = 3 if i % 2 == 0 else 9
    out = _decide(rows)
    comp = out["secondary"]["readout"]["position"]["complement_rank_raw_le_k"]
    n, rate = comp
    assert n == sum(1 for r in rows if r["role"] == "position" and r["rank_t1j"] <= DS.K_SS)
    assert 0.0 <= rate <= 1.0


# ═════ pre-execution repair 3: the frozen descriptive representation ════════

def test_the_row_carries_BOTH_rank_fields_the_summary_needs():
    """🔴 My first version asserted both were 1 -- exactly what a control that
    HARDCODES 1 produces, so it caught nothing. Both values are now non-trivial:
    the T1j move takes visit rank 1, pushing ours to 2, and the policy rank is
    set to a value no plausible constant would match."""
    pos = _pos(t1j_rank=3, t1j_visit_rank=1)
    pos["incumbent"]["selected_policy_rank"] = 4
    row = DS.suppression_row(pos)
    assert row["selected_visit_rank"] == 2       # T1j's move led the visits, not ours
    assert row["selected_policy_rank"] == 4


def test_the_readout_summary_has_EXACTLY_the_frozen_shape():
    rows = _ss_cohort(ALL_CELLS, pos_rate=1 / 3, ctl_rate=1 / 3)
    for i, r in enumerate(rows):
        r["selected_visit_rank"] = 1 + (i % 3)
        r["selected_policy_rank"] = 1 + (i % 2)
        r["rank_t1j"] = 3
    got = DS._readout_summary(rows, cohort=DS.PRIMARY_COHORT)
    assert set(got) == {"position", "control"}
    for role in ("position", "control"):
        assert set(got[role]) == {"n", "overrode_leader", "selected_visit_rank",
                                  "selected_policy_rank", "ss_rate",
                                  "complement_rank_raw_le_k"}
        assert got[role]["overrode_leader"] == [[False, got[role]["n"]], [True, 0]]
        hist = got[role]["selected_visit_rank"]
        assert hist == sorted(hist)                       # ascending by rank
        assert all(c > 0 for _r, c in hist)               # zero counts omitted
        assert sum(c for _r, c in hist) == got[role]["n"]  # every row counted once


def test_the_histogram_counts_ranks_and_not_something_averaged():
    rows = _ss_cohort([(OPENINGS[0], ARMS[0], PHASES[0])], pos_rate=0.0, ctl_rate=0.0, per_role=3)
    for r, vr in zip([x for x in rows if x["role"] == "position"], (1, 1, 4)):
        r["selected_visit_rank"] = vr
        r["selected_policy_rank"] = 1
        r["rank_t1j"] = 9
    for r in [x for x in rows if x["role"] == "control"]:
        r["selected_visit_rank"] = 2
        r["selected_policy_rank"] = 1
        r["rank_t1j"] = 9
    got = DS._readout_summary(rows, cohort=DS.PRIMARY_COHORT)
    assert got["position"]["selected_visit_rank"] == [[1, 2], [4, 1]]   # not a mean of 2.0


def test_the_histogram_is_ordered_by_RANK_even_when_the_counts_disagree():
    """🔴 Two controls were blind until this existed. With counts {1:2, 4:1},
    ordering by descending COUNT gives the same list as ordering by rank, and
    ranks 1..4 with no gap hides a zero-count filler. Here rank 1 is the RAREST
    and the ranks are non-contiguous, so both defects change the answer."""
    rows = _ss_cohort([(OPENINGS[0], ARMS[0], PHASES[0])], pos_rate=0.0, ctl_rate=0.0,
                      per_role=4)
    for r, vr in zip([x for x in rows if x["role"] == "position"], (1, 5, 5, 9)):
        r["selected_visit_rank"] = vr
    got = DS._readout_summary(rows, cohort=DS.PRIMARY_COHORT)["position"]["selected_visit_rank"]
    assert got == [[1, 1], [5, 2], [9, 1]]      # ascending by rank; ranks 2-4, 6-8 absent


# ═══════════ pre-execution repair 2: four residual validation gaps ══════════

# ── P1: the depth records are validated type-strictly, and FIRST ────────────

@pytest.mark.parametrize("depths,msg", [
    ("not-a-list", "depths"),
    ([["depth", 6]], "depth record"),
    # 🔴 "depth" alone did NOT discriminate: with the type check deleted, "6" != 6
    # so no depth-6 record is found and the fallback message ("expected exactly
    # one depth-6 record") contains the word too. The TYPE refusal is named.
    ([{"depth": "6", "move": [0, 0]}], "a depth is an int"),
    ([{"depth": 6.0, "move": [0, 0]}], "a depth is an int"),
    ([{"move": [0, 0]}], "carries no 'depth'"),
])
def test_a_malformed_depths_container_or_entry_is_REFUSED_BY_NAME(canonical, depths, msg):
    """🔑 A string or float depth contradicts the type-strict contract, and a
    non-mapping entry must not escape as AttributeError or ValueError -- every
    refusal here is a named D1SecondError."""
    rep = _valid_report(canonical)
    rep["positions"][3]["depths"] = depths
    with pytest.raises(DS.D1SecondError, match=msg):
        DS.validate_record(rep)


def test_TWO_depth_six_records_are_refused(canonical):
    rep = _valid_report(canonical)
    d6 = [d for d in rep["positions"][3]["depths"] if d["depth"] == 6][0]
    rep["positions"][3]["depths"] = rep["positions"][3]["depths"] + [dict(d6)]
    with pytest.raises(DS.D1SecondError, match="exactly one depth-6"):
        DS.validate_record(rep)


# ── P1: the search budget is bound to the frozen configuration ──────────────

def test_the_simulation_budget_is_READ_from_the_frozen_identity_not_retyped():
    """The number lives in the frozen L0 plan's configuration, reached through
    `d1_probe.frozen_incumbent_identity`. This module must not carry a literal."""
    from scripts.GPU.alphazero import d1_probe as D1P
    want = D1P.frozen_incumbent_identity()["eval_config"]["mcts_sims"]
    assert DS.frozen_sim_budget() == want
    # An AST scan, not a substring search: the docstrings SAY "400-simulation
    # search" on purpose, and a text match would forbid explaining the rule while
    # allowing `if total != 400` written as `4 * 100`. What must not exist is the
    # integer CONSTANT.
    import ast, pathlib as _p
    tree = ast.parse(_p.Path(DS.__file__).read_text())
    literals = [n for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and type(n.value) is int and n.value == want]
    assert not literals, f"the budget is retyped as a literal on line {literals[0].lineno}"


def test_a_SELF_CONSISTENT_but_under_searched_root_is_REFUSED(canonical):
    """🔴 THE GAP THIS CLOSES. A forged map totalling 1 passed as long as
    `root_total_visits` was changed to agree: every internal check was satisfied
    and nothing bound the total to the search that was supposed to have run. The
    hypothesis is about treatment AFTER the frozen search, so a root that did not
    run it is not evidence about it."""
    from tests.test_d1prime_analysis import _incumbent_record
    rep = _valid_report(canonical)
    inc = rep["positions"][3]["incumbent"]
    policy = inc["raw_policy"]
    ours = (inc["row"], inc["col"])
    tiny = {k: (1 if _mv(k) == ours else 0) for k in policy}
    rep["positions"][3]["incumbent"] = _incumbent_record(policy=policy, our_move=ours,
                                                         visits=tiny)
    with pytest.raises(DS.D1SecondError, match="root_total_visits"):
        DS.validate_record(rep)


def test_the_budgeted_baseline_itself_is_ACCEPTED(canonical):
    """So the refusal above is about the budget and not about the fixture."""
    DS.validate_record(_valid_report(canonical))


# ── P2: top2 is inventory-only, and the code says so where the plan does ────

def test_top2_is_declared_UNUSED_rather_than_silently_omitted():
    """The plan's schema inventory lists `top2`; D1″ reads it nowhere. The
    promise the documentation makes and the promise the code keeps must be the
    same one, so the exclusion is DECLARED, not left as an absence."""
    assert "top2" in DS.UNUSED_OBSERVABLES
    assert "top2" not in DS.REQUIRED_OBSERVABLES
    assert not set(DS.UNUSED_OBSERVABLES) & set(DS.REQUIRED_OBSERVABLES)


def test_a_record_whose_top2_is_anything_at_all_is_still_accepted(canonical):
    """The declared consequence of "inventory-only": D1″ neither reads nor
    refuses it."""
    rep = _valid_report(canonical)
    rep["positions"][3]["incumbent"]["top2"] = "not even a list"
    DS.validate_record(rep)


# ── P2: derived values must agree EXACTLY ──────────────────────────────────

@pytest.mark.parametrize("field", ["selected_policy_mass", "root_top1_share"])
def test_a_derived_value_perturbed_BELOW_any_tolerance_is_still_refused(canonical, field):
    """🔴 `math.isclose` admitted an altered record. Both values are written from
    the very maps the record preserves, so recomputing them reproduces the same
    float exactly; no tolerance is needed and none is frozen."""
    rep = _valid_report(canonical)
    inc = rep["positions"][3]["incumbent"]
    inc[field] = inc[field] * (1 + 1e-13) + 1e-18
    with pytest.raises(DS.D1SecondError, match=field):
        DS.validate_record(rep)


@pytest.mark.parametrize("field", ["selected_policy_mass", "root_top1_share"])
def test_a_derived_value_of_the_WRONG_TYPE_is_refused(canonical, field):
    rep = _valid_report(canonical)
    rep["positions"][3]["incumbent"][field] = True
    with pytest.raises(DS.D1SecondError, match=field):
        DS.validate_record(rep)
