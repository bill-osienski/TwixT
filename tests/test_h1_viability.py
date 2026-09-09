"""H1 head-to-head viability screen -- plan, rules and verdict. NO EXECUTION.

No model is loaded, no JVM started, no game played, no seed drawn or registered.
Building a task's stream integers is XOR arithmetic on already-qualified masks.
"""
import pytest

from scripts.GPU.alphazero import l0_match_rules as L0RULES


# ───────────── the binding boundary must name WHICH design it binds ──────────

def test_bind_results_REQUIRES_a_design_and_has_no_default():
    """H1 has 224 games, L0 has 64, and `bind_results` checked L0's count, L0's
    reps and L0's frozen digest directly. Parameterising it with a DEFAULT would
    let a caller bind an H1 result set against the 64-game design by omission --
    the defaultable-switch failure this codebase keeps finding, and the same
    reason `validate_schedule_executable` is a separate function rather than a
    `require_unspent=` keyword. `design` is keyword-only and required.
    """
    with pytest.raises(TypeError):
        L0RULES.bind_results([], [])


def test_the_L0_design_is_still_exactly_what_L0_froze():
    d = L0RULES.L0_DESIGN
    assert (d.n_openings, d.n_arms, d.n_reps, d.n_games) == (8, 2, 4, 64)
    assert d.task_digest == L0RULES.L0_TASK_DIGEST


from scripts.GPU.alphazero import h1_viability_rules as R      # noqa: E402


# ────────────────────────── the design and its provenance ───────────────────

def test_the_design_is_8_openings_2_arms_14_reps():
    assert (R.N_OPENINGS, R.N_ARMS, R.N_REPS) == (8, 2, 14)
    assert R.N_GAMES == R.N_OPENINGS * R.N_ARMS * R.N_REPS == 224
    assert R.H1_DESIGN.n_games == R.N_GAMES and R.H1_DESIGN.name == "H1"


def test_H1_REDECLARES_ONLY_WHAT_DIFFERS_FROM_L0():
    """The shared parameters are READ from L0's rules, not restated here.

    ⚠ HONEST LIMIT, stated rather than papered over: for a small int, equality
    cannot distinguish "read from L0" from "retyped as 8" -- CPython interns
    both. So H1 does not declare the shared scalars at all; it re-exports L0's
    objects, and the real binding for the values that matter (the openings and
    the reference identity) is the sha-pinned plan file plus the task digest,
    which ARE checkable. See test_the_openings_come_from_the_pinned_plan.
    """
    for name in ("T1J_MDPLY", "PLY_CAP", "ALPHA", "Z_95", "WINNERS",
                 "TERMINAL_REASONS", "SCORE_DRAW", "N_OPENINGS", "N_ARMS"):
        assert getattr(R, name) is getattr(L0RULES, name), name


def test_there_is_no_early_stop():
    assert R.EARLY_STOP is None
    assert R.may_stop_early() is False
    assert R.may_stop_early(1, 2, three=3) is False


# ──────────────────────────── the viability verdict ─────────────────────────

def test_the_threshold_is_the_predeclared_0_75():
    assert R.VIABILITY_THRESHOLD == 0.75


@pytest.mark.parametrize("lo,hi,want", [
    (0.40, 0.68, "VIABLE"),           # upper below the threshold
    (0.75, 0.95, "NOT_VIABLE"),       # lower AT the threshold -- inclusive
    (0.80, 0.95, "NOT_VIABLE"),
    (0.70, 0.80, "INCONCLUSIVE"),     # straddles
    # BOUNDARY: the rule is "upper < 0.75", so an upper bound exactly AT the
    # threshold does NOT exclude it. My first expectation here said VIABLE and
    # contradicted its own comment; the code was right.
    (0.60, 0.75, "INCONCLUSIVE"),
])
def test_the_verdict_is_where_0_75_falls_relative_to_the_interval(lo, hi, want):
    assert R.viability_verdict(lo, hi) == want


def test_the_decisive_bands_are_DERIVED_not_typed():
    """The card's 0.6593 / 0.8407 must come from the interval, so they cannot
    drift from it. Recomputed here from the primary interval itself."""
    lo_band, hi_band = R.decisive_bands(R.N_GAMES)
    hw = (L0RULES.hoeffding_interval(R.N_GAMES / 2, R.N_GAMES)[1] - 0.5)
    assert abs(lo_band - (R.VIABILITY_THRESHOLD - hw)) < 1e-12
    assert abs(hi_band - (R.VIABILITY_THRESHOLD + hw)) < 1e-12
    assert round(lo_band, 4) == 0.6593 and round(hi_band, 4) == 0.8407


def test_a_rate_just_inside_each_band_gets_the_verdict_the_card_promises():
    lo_band, hi_band = R.decisive_bands(R.N_GAMES)
    for rate, want in ((lo_band - 1e-6, "VIABLE"), (hi_band + 1e-6, "NOT_VIABLE"),
                       ((lo_band + hi_band) / 2, "INCONCLUSIVE")):
        lo, hi = L0RULES.hoeffding_interval(rate * R.N_GAMES, R.N_GAMES)
        assert R.viability_verdict(lo, hi) == want, rate


# ─────────────────────────────── reporting rules ────────────────────────────

def test_H1_inherits_every_COUNT_FREE_claim_recounts_two_and_adds_two():
    """⚠ CORRECTED: H1 does NOT inherit L0's list whole, and an earlier version
    of this test asserted it did. Two of L0's prohibitions state denominators
    that are false for a 224-game match."""
    assert set(L0RULES.COUNT_FREE_FORBIDDEN_CLAIMS) <= set(R.FORBIDDEN_CLAIMS)
    assert not set(L0RULES.FORBIDDEN_CLAIMS) <= set(R.FORBIDDEN_CLAIMS)
    assert len(R.FORBIDDEN_CLAIMS) == len(L0RULES.FORBIDDEN_CLAIMS) + 2
    # 🔴 EACH PROHIBITION IS BOUND BY ITS OPENING WORDS, not by a substring of the
    # joined list. A control that replaced the first line of the pooling claim with
    # "POOLING IS FINE ACTUALLY" passed the earlier version of this test: the word
    # "pooling" still appeared further down the same implicitly-concatenated
    # string, and the COUNT was unchanged. A search over the whole text cannot see
    # a claim being inverted in place.
    assert any(c.startswith("any pooling of H1 with L0") for c in R.FORBIDDEN_CLAIMS)
    assert any(c.startswith("any use of the ~191 Elo figure")
               for c in R.FORBIDDEN_CLAIMS)


def test_the_independence_caveat_is_about_H1s_own_448_streams():
    assert "448" in R.INDEPENDENCE_CAVEAT
    assert "224" in R.INDEPENDENCE_CAVEAT
    assert "modelled" in R.INDEPENDENCE_CAVEAT.lower()


from scripts.GPU.alphazero import h1_viability_plan as P        # noqa: E402
from scripts.GPU.alphazero import e4_screen_reference as REF    # noqa: E402
from scripts.GPU.alphazero import l0_match_plan as L0PLAN       # noqa: E402


@pytest.fixture(scope="module")
def tasks():
    return P.build_tasks(P.load_source_plan())


def test_the_task_digest_constant_IS_the_digest_of_the_built_tasks(tasks):
    """🔴 THE ANTI-FABRICATION CONTROL. While this module was being written,
    H1_TASK_DIGEST held a hand-typed 64-hex string that looked entirely
    plausible and matched nothing. A pinned digest that is never recomputed
    against the artifact it pins is decoration."""
    assert R.H1_TASK_DIGEST == L0RULES.l0_task_digest(tasks)
    assert R.H1_DESIGN.task_digest == R.H1_TASK_DIGEST


def test_the_schedule_is_224_games_in_16_balanced_cells(tasks):
    assert len(tasks) == 224
    cells = {}
    for t in tasks:
        cells[(t["opening"], t["colour_arm"])] = cells.get((t["opening"], t["colour_arm"]), 0) + 1
    assert len(cells) == 16
    assert set(cells.values()) == {14}


def test_colours_are_RECIPROCAL_WITHIN_each_opening(tasks):
    """Not merely balanced in aggregate. L0 saw t1j_red 0.781 against t1j_black
    0.406; an opening played more often in one arm would let that asymmetry into
    the overall rate, which is the one quantity H1 decides on."""
    for opening in {t["opening"] for t in tasks}:
        per_arm = {}
        for t in tasks:
            if t["opening"] == opening:
                per_arm[t["colour_arm"]] = per_arm.get(t["colour_arm"], 0) + 1
        assert per_arm == {"t1j_red": 14, "t1j_black": 14}, opening


def test_seeds_are_the_ATTEMPT2_block_one_per_task_in_order(tasks):
    """🔴 ATTEMPT 2. The 2026-09-05 match VOIDed at game 60 and retired
    [202616000, 202616224) WHOLE; a retry needs a FRESH interval. This is it:
    the same one-thousand stride, a 776-seed gap from the spent block's end (so
    an off-by-one at the boundary cannot masquerade as a reservation),
    collision-proved 2026-09-07 against every registry AND D1's paper block."""
    assert P.H1_SEED_BLOCK == R.H1_SEED_BLOCK == (202617000, 202617224)
    assert R.H1_ATTEMPT1_SEED_BLOCK == (202616000, 202616224)
    assert [t["seed"] for t in tasks] == list(range(*P.H1_SEED_BLOCK))


def test_the_ATTEMPT2_block_is_ACCOUNTED_EXPOSED_224_and_RETIRED_WHOLE():
    """⚠ INVERTED THREE TIMES, the last by the match itself: paper-only (prep) ->
    ACCOUNTED (registration) -> ACCOUNTED + EXPOSED 224 + RETIRED WHOLE (the
    match ran once on 2026-09-07 and COMPLETED all 224 games).

    🔑 224 EXPOSED, counted from the RECORDS: every task has a task_result and
    at least one reference-agent ply, so every seed was drawn (attempt 1 had 60
    for the same reason it did not have 61). RETIRED WHOLE because a
    preregistered one-shot schedule completed; replaying any part would be
    selection after seeing the verdict."""
    lo, hi = R.H1_SEED_BLOCK
    st = [REF.seed_status(s) for s in range(lo, hi)]
    assert all(x["accounted"] for x in st)
    assert sum(x["exposed"] for x in st) == 224
    assert all(x["retired"] for x in st), "the block retires WHOLE"
    assert all(REF.seed_is_unavailable(s) for s in range(lo, hi))
    assert not any(x["test_only"] for x in st)


def test_the_ATTEMPT2_block_keeps_a_load_bearing_GAP_from_every_prior_boundary():
    lo, hi = R.H1_SEED_BLOCK
    from scripts.GPU.alphazero import d1_selection as SEL
    # every PRIOR boundary: the block's own interval (registered now) is excluded
    ends = {b for reg in (REF.ACCOUNTED_SEED_INTERVALS, REF.EXPOSED_SEED_INTERVALS,
                          REF.RETIRED_SEED_INTERVALS, REF.TEST_ONLY_SEED_INTERVALS,
                          (SEL.SEED_INTERVAL,)) for a, b_ in reg
            if (a, b_) != (lo, hi) for b in (a, b_)}
    assert min(min(abs(lo - b), abs(hi - b)) for b in ends) >= 224


def test_the_ATTEMPT1_block_is_ACCOUNTED_60_EXPOSED_and_RETIRED_WHOLE():
    """⚠ INVERTED TWICE, and the second time by the match itself.

    Paper-reserved -> ACCOUNTED (2026-09-04 seed preparation) -> ACCOUNTED +
    partly EXPOSED + wholly RETIRED (2026-09-05, when the single authorized match
    VOIDED at game 60 of 224).

    🔑 SIXTY EXPOSED, NOT 224 AND NOT 61. Exposure is a claim about DRAWS: 60
    games built real agents and were recorded. Game 60 began and bound its
    opening, but the arm is `t1j_red` and T1j was to move at ply 6 -- its query
    failed there, so our reference agent (black) was never constructed and seed
    202616060 was never drawn. Marking all 224 exposed would assert 164 draws
    that never happened, which is the overstatement these lists are kept apart to
    prevent.

    🔑 RETIRED WHOLE, drawn and undrawn alike. A preregistered one-shot schedule
    was started and did not complete, so replaying any part of it would select
    games after seeing where it failed.
    """
    lo, hi = R.H1_ATTEMPT1_SEED_BLOCK
    st = [REF.seed_status(s) for s in range(lo, hi)]
    assert all(x["accounted"] for x in st)
    assert sum(x["exposed"] for x in st) == 60
    assert all(REF.seed_is_exposed(s) for s in range(lo, lo + 60))
    assert not any(REF.seed_is_exposed(s) for s in range(lo + 60, hi))
    assert all(x["retired"] for x in st), "the block retires WHOLE"
    assert all(REF.seed_is_unavailable(s) for s in range(lo, hi))
    assert not any(x["test_only"] for x in st)


def test_D1s_reservation_is_SPENT_and_DISJOINT_from_both_H1_blocks():
    """INVERTED 2026-09-08, because the fact it asserted stopped being true.

    It used to say D1's §14 interval was in NO registry: the H1 authorizations
    registered one block each and swept nothing else in, which is also why the
    H1 collision proof had to name D1's paper reservation explicitly rather than
    enumerate registries. D1's OWN seed-preparation authorization has since
    registered it, so the paper-only claim is now false and this asserts what is
    still true and still H1's business:

    * D1's block is ACCOUNTED and nothing else -- no H1 round exposed, retired or
      test-marked seeds outside its own block, and neither did D1's registration;
    * it is disjoint from BOTH H1 blocks, so no H1 seed is a D1 seed.

    Keeping this in the H1 file is deliberate: it fails if a future H1 round
    reaches outside its own reservation.
    """
    from scripts.GPU.alphazero import d1_selection as SEL
    for seed in range(*SEL.SEED_INTERVAL):
        st = REF.seed_status(seed)
        # accounted -> +exposed +retired, as D1 ran once on 2026-09-08 and
        # completed. What stays H1's business is that no H1 round put it there.
        assert st["accounted"] and st["exposed"] and st["retired"], (seed, st)
        assert not st["test_only"], (seed, st)
    d1 = set(range(*SEL.SEED_INTERVAL))
    for block in (R.H1_ATTEMPT1_SEED_BLOCK, R.H1_SEED_BLOCK):
        assert not (d1 & set(range(*block))), block


def test_the_openings_come_from_the_pinned_plan_and_match_what_L0_PLAYED(tasks):
    """The values that matter are bound to a sha256-pinned file, and then
    cross-checked against the schedule L0 actually played -- so H1 is comparable
    to L0 by construction rather than by assertion."""
    l0_tasks = L0PLAN.build_tasks(L0PLAN.load_source_plan())
    assert {t["opening"] for t in tasks} == {t["opening"] for t in l0_tasks}
    assert {t["reference_sha256"] for t in tasks} == {t["reference_sha256"] for t in l0_tasks}
    assert {t["t1j_mdPly"] for t in tasks} == {6}


def test_a_tampered_source_plan_is_refused(tmp_path):
    bad = tmp_path / "plan.json"
    bad.write_text('{"openings": {}, "reference": {}}')
    with pytest.raises(P.H1PlanError, match="sha256"):
        P.load_source_plan(str(bad))


def test_the_schedule_validates_and_reports_its_own_shape(tasks):
    got = P.validate_h1_schedule(tasks)
    assert got["n_tasks"] == 224 and got["cells"] == 16 and got["reps_per_cell"] == 14
    assert got["task_digest"] == R.H1_TASK_DIGEST


@pytest.mark.parametrize("mutate,match", [
    (lambda ts: ts[:-1], "expected exactly"),
    (lambda ts: [dict(t, colour_arm="t1j_red") for t in ts], "cells"),
    # DISTINCT but outside the block: setting them all to one value tripped the
    # injective-streams guard first, so the block check was never reached.
    (lambda ts: [dict(t, seed=t["seed"] + 1_000_000) for t in ts], "outside"),
    (lambda ts: [dict(t, t1j_mdPly=3) for t in ts], "mdPly"),
])
def test_validation_refuses_a_schedule_that_is_not_the_design(tasks, mutate, match):
    with pytest.raises(P.H1PlanError, match=match):
        P.validate_h1_schedule(mutate(list(tasks)))


# ─────────────────────────── the report and its verdict ─────────────────────

def _results(tasks, t1j_wins):
    """A complete, valid result set in which T1j wins exactly `t1j_wins` games."""
    out = []
    for i, t in enumerate(tasks):
        t1j_won = i < t1j_wins
        anchor = t["anchor_colour"]
        winner = anchor if t1j_won else ("black" if anchor == "red" else "red")
        out.append({"task_id": t["task_id"], "winner": winner,
                    "terminal_reason": "win", "t1j_points": 1.0 if t1j_won else 0.0,
                    "plies": 41, "seed": t["seed"]})
    return out


def test_a_complete_result_set_reports_and_carries_its_verdict(tasks):
    rep = R.viability_report(_results(tasks, 133), tasks)          # 133/224 = 0.5938
    assert rep["reported"] is True
    assert rep["overall"]["games"] == 224
    assert round(rep["overall"]["t1j_rate"], 4) == 0.5938
    assert rep["verdict"] == "VIABLE"
    assert set(rep["forbidden_claims"]) == set(R.FORBIDDEN_CLAIMS)


@pytest.mark.parametrize("wins,want", [
    (133, "VIABLE"),          # L0's own observed rate, 0.5938
    (147, "VIABLE"),          # 0.65625, just inside the decisive band
    (149, "INCONCLUSIVE"),    # 0.66518, just outside it
    (170, "INCONCLUSIVE"),    # 0.75893 -- inside the inconclusive band
    # 0.89286: lower bound 0.8021, which IS >= 0.75. My first expectation here
    # said INCONCLUSIVE while its own comment quoted a lower bound above the
    # threshold. The code was right; the expectation contradicted itself.
    (200, "NOT_VIABLE"),
    (224, "NOT_VIABLE"),      # 1.0; lower bound 0.909 >= 0.75
])
def test_the_verdict_follows_the_predeclared_bands(tasks, wins, want):
    rep = R.viability_report(_results(tasks, wins), tasks)
    assert rep["verdict"] == want, (wins, rep["overall"]["t1j_rate"],
                                    rep["overall"]["ci95_hoeffding"])


def test_the_verdict_reads_the_HOEFFDING_interval_and_nothing_else(tasks):
    """Wilson is reported and decides nothing. Asserted by recomputing the
    verdict from the primary bounds alone and requiring agreement."""
    for wins in (100, 149, 180, 224):
        rep = R.viability_report(_results(tasks, wins), tasks)
        lo, hi = rep["overall"]["ci95_hoeffding"]
        assert rep["verdict"] == R.viability_verdict(lo, hi)
        wlo, whi = rep["overall"]["ci95_wilson"]
        assert (wlo, whi) != (lo, hi)          # they really are different numbers


def test_an_incomplete_match_is_REFUSED_not_reported(tasks):
    """A truncated match is an early stop wearing different clothes."""
    rep = R.viability_report(_results(tasks, 100)[:-1], tasks)
    assert rep["reported"] is False and "unplayed" in rep["reason"]


def test_the_L0_DESIGN_cannot_be_used_to_report_an_H1_match(tasks):
    """The design argument is what stops 224 results being bound to the 64-game
    protocol, and it must refuse rather than silently truncate."""
    pairs, why = L0RULES.bind_results(_results(tasks, 100), tasks,
                                      design=L0RULES.L0_DESIGN)
    assert pairs is None and "expected 64" in why


def test_cap_saturation_yields_NO_RATE_AND_NO_VERDICT(tasks):
    rows = _results(tasks, 0)
    for r in rows[:113]:                       # more than half of 224
        r.update(winner=None, terminal_reason="cap", t1j_points=0.5,
                 plies=R.PLY_CAP)
    rep = R.viability_report(rows, tasks)
    assert rep["reported"] is False
    assert rep["outcome"] == "CAP_SATURATED_NO_RATE"
    assert "verdict" not in rep


def test_the_report_states_the_bands_and_the_rule_it_used(tasks):
    rep = R.viability_report(_results(tasks, 133), tasks)
    lo_band, hi_band = R.decisive_bands(R.N_GAMES)
    assert rep["decisive_bands"] == {"viable_below": lo_band,
                                     "not_viable_at_or_above": hi_band}
    assert "directional" in rep["verdict_rule"].lower()


def test_the_H1_block_is_disjoint_from_the_D1_paper_reservation():
    """🔴 The §14 D1 block is in NO registry, so `seed_status` cannot see it and
    the registry test above would pass on an overlapping block. Checked directly
    against the constant."""
    from scripts.GPU.alphazero import d1_selection as SEL
    lo, hi = P.H1_SEED_BLOCK
    d_lo, d_hi = SEL.SEED_INTERVAL
    assert hi <= d_lo or lo >= d_hi, (P.H1_SEED_BLOCK, SEL.SEED_INTERVAL)


def test_a_cell_with_the_wrong_repetition_LABELS_is_refused(tasks):
    """Counts alone do not fix the labels: this cell has 14 rows and repeats r0."""
    bad = [dict(t, rep=0) if t["opening"] == tasks[0]["opening"] else t for t in tasks]
    with pytest.raises(P.H1PlanError, match="repetitions"):
        P.validate_h1_schedule(bad)


def test_the_frozen_plan_loads_and_IS_the_built_schedule(tasks):
    plan = P.load_h1_plan()
    assert plan["n_games"] == 224 and plan["seed_block"] == [202617000, 202617224]
    assert plan["attempt"] == 2
    assert plan["shape"]["task_digest"] == R.H1_TASK_DIGEST
    assert [t["task_id"] for t in plan["tasks"]] == [t["task_id"] for t in tasks]
    assert plan["seed_block_status"].startswith("PAPER-RESERVED, UNREGISTERED")
    assert plan["early_stop"] is None


def test_the_ATTEMPT1_plan_v3_is_PRESERVED_and_still_loads_under_its_own_pins():
    """Evidence is create-only: v3 is the record of what attempt 1 froze and ran.
    It loads structurally under its OWN pins -- and is refused for EXECUTION,
    because every one of its seeds is retired."""
    v3 = P.load_h1_plan(P.H1_ATTEMPT1_PLAN_REL, sha256=P.H1_ATTEMPT1_PLAN_SHA256,
                        task_digest=R.H1_ATTEMPT1_TASK_DIGEST)
    assert v3["seed_block"] == [202616000, 202616224]
    assert R.L0.l0_task_digest(v3["tasks"]) == R.H1_ATTEMPT1_TASK_DIGEST != R.H1_TASK_DIGEST
    with pytest.raises(REF.E4ReferenceError, match="EXPOSED .* cannot be scheduled"):
        REF.validate_schedule_executable(v3["tasks"])


def test_the_ATTEMPT2_plan_differs_from_v3_ONLY_in_seeds_streams_and_provenance():
    """The 224-game design, settings, threshold, bands, rules, estimand and
    forbidden claims are UNCHANGED; only the seed block (and therefore each
    task's seed and derived streams) and the provenance fields differ."""
    import json as _json
    v3 = _json.loads(open(P.H1_ATTEMPT1_PLAN_REL, "rb").read())
    v4 = P.load_h1_plan()
    changed = {"seed_block", "seed_block_status", "supersedes", "shape", "tasks",
               "attempt", "attempt_1"}
    for k in set(v3) | set(v4):
        if k in changed:
            continue
        assert v3.get(k) == v4.get(k), k
    for a, b in zip(v3["tasks"], v4["tasks"]):
        for k in a:
            if k in ("seed", "rng_streams"):
                assert a[k] != b[k], (a["task_id"], k)
            else:
                assert a[k] == b[k], (a["task_id"], k)
        assert b["seed"] - a["seed"] == 1000


def test_a_plan_whose_TASKS_were_swapped_is_refused_even_if_the_FILE_hashes(tmp_path,
                                                                            monkeypatch):
    """The file hash alone would accept a correctly-hashed file whose tasks came
    from a different design; the task digest is what closes that."""
    import json as _json
    plan = _json.loads(open(P.H1_PLAN_REL, "rb").read())
    plan["tasks"] = plan["tasks"][:-1] + [dict(plan["tasks"][-1], opening="invented")]
    bad = tmp_path / "p.json"
    raw = _json.dumps(plan, indent=2, sort_keys=False).encode()
    bad.write_bytes(raw)
    import hashlib as _h
    monkeypatch.setattr(P, "H1_PLAN_SHA256", _h.sha256(raw).hexdigest())
    with pytest.raises(P.H1PlanError, match="task digest"):
        P.load_h1_plan(str(bad))


def test_a_plan_with_the_SAME_TASKS_but_a_REWRITTEN_THRESHOLD_is_refused(tmp_path):
    """🔴 THE FILE HASH AND THE TASK DIGEST CATCH DIFFERENT ATTACKS, and only this
    case reaches the file hash alone.

    Here the 224 tasks are byte-identical, so the task digest still matches --
    but `viability_threshold` has been rewritten from 0.75 to 0.5, which would
    turn an INCONCLUSIVE result into a VIABLE one. A control that removed the
    file-hash check was NOT CAUGHT until this test existed, because every other
    loader test hands the loader the correct file.
    """
    import json as _json
    plan = _json.loads(open(P.H1_PLAN_REL, "rb").read())
    assert plan["viability_threshold"] == 0.75
    plan["viability_threshold"] = 0.5                  # tasks untouched
    bad = tmp_path / "p.json"
    bad.write_bytes(_json.dumps(plan, indent=2, sort_keys=False).encode())
    assert R.L0.l0_task_digest(plan["tasks"]) == R.H1_TASK_DIGEST      # digest agrees
    with pytest.raises(P.H1PlanError, match="sha256"):
        P.load_h1_plan(str(bad))


# ═══════ review corrections: composed abort rules, H1's own denominators ═════

def test_L0s_ABORT_RULES_are_unchanged_by_the_factoring():
    """L0 is a completed, published measurement. Factoring the seed rule out so
    H1 can name its OWN block must leave L0's tuple identical, ORDER included --
    the seed rule sits fifth, not last."""
    assert L0RULES.L0_ABORT_RULES == (
        "any per-ply state divergence between the two engines",
        "any T1j query that does not complete its requested depth",
        "any illegal move, or the null sentinel, from either side",
        "any postcondition failure: a Window/Frame, a non-headless jvm, a mutated host "
        "preference store, or an unauthorized reflective access",
        "any artifact identity mismatch: jar, JDK component, or checkpoint sha",
        "any seed outside the reserved L0 block, or any seed used twice",
        "any failure to write or fsync a durable record",
    )


def test_H1s_abort_rules_name_H1s_BLOCK_AND_ONLY_H1s():
    """🔴 APPENDING H1's seed rule to L0's list left BOTH active, and every valid
    H1 seed is outside the L0 block -- so the composed rule set aborted every H1
    game. The seed rule is now composed, not appended."""
    joined = " ".join(R.H1_ABORT_RULES)
    assert "reserved H1 block" in joined
    assert "L0 block" not in joined
    assert sum("seed outside the reserved" in r for r in R.H1_ABORT_RULES) == 1


def test_an_H1_SEED_is_accepted_and_an_L0_BLOCK_seed_is_refused(tasks):
    """The rule as a PREDICATE, not only as prose. Prose cannot be run."""
    for t in tasks:
        assert R.seed_is_outside_the_reserved_block(t["seed"]) is False
    for seed in (202613000, 202613063, 202614000, 202615000, 0):
        assert R.seed_is_outside_the_reserved_block(seed) is True


def test_L0s_FORBIDDEN_CLAIMS_are_unchanged_by_the_factoring():
    """PINNED BY CONTENT DIGEST, not by three fragments.

    🔴 The first version asserted only that two count-bearing phrases were still
    present and that the list held 8 entries. A control that reworded the Elo
    prohibition ("figure" -> "number") passed it: the fragments it checked were
    untouched. "Unchanged" has to be checked over the whole thing.
    """
    import hashlib as _h, json as _j
    assert len(L0RULES.FORBIDDEN_CLAIMS) == 8
    got = _h.sha256(_j.dumps(list(L0RULES.FORBIDDEN_CLAIMS)).encode()).hexdigest()
    assert got == "809f0ed8fa630a363c36f2fcb4a7277ac48f31fe453028927c7c8de54e0b2d7a"
    # kept for legibility: the digest says WHETHER it changed, these say WHAT it holds
    joined = " ".join(L0RULES.FORBIDDEN_CLAIMS)
    assert "8 games per opening and 32 per colour arm" in joined
    assert "the 64 games ARE independent" in joined


def test_NO_H1_forbidden_claim_carries_an_L0_DENOMINATOR():
    """🔴 TWO inherited claims were count-bearing, not one: the per-cell
    prohibition says '8 games per opening and 32 per colour arm', and the
    independence prohibition says 'the 64 games'. H1 has 28, 112 and 224. A
    prohibition that states a false denominator puts that denominator in the
    report, which is exactly what the prohibition exists to prevent."""
    # ⚠ EXACT MEMBERSHIP, not a substring search. My first version asserted
    # `"8 games per opening" not in joined` and failed on the CORRECT list,
    # because "28 games per opening" CONTAINS "8 games per opening". The same
    # loose-substring trap as grepping for a seed prefix.
    assert L0RULES.per_cell_prohibition(8, 32) not in R.FORBIDDEN_CLAIMS
    assert L0RULES.independence_prohibition(64) not in R.FORBIDDEN_CLAIMS
    assert L0RULES.per_cell_prohibition(28, 112) in R.FORBIDDEN_CLAIMS
    assert L0RULES.independence_prohibition(224) in R.FORBIDDEN_CLAIMS


def test_H1_still_inherits_every_COUNT_FREE_L0_prohibition():
    count_free = [c for c in L0RULES.FORBIDDEN_CLAIMS
                  if "8 games per opening" not in c and "the 64 games" not in c]
    assert len(count_free) == 6
    assert set(count_free) <= set(R.FORBIDDEN_CLAIMS)
    assert len(R.FORBIDDEN_CLAIMS) == 10           # 6 inherited + 2 recounted + 2 H1


def test_the_frozen_plan_STATES_the_cap_saturation_branch():
    """[P2] The card said cap terminations are always scored, counted and
    reported, and stopped there -- but past 112 of 224 the code correctly reports
    no rate AND NO VERDICT. An artifact that omits the branch describes a
    protocol the code does not implement."""
    cs = P.load_h1_plan()["cap_saturation"]
    assert cs["threshold"] == R.CAP_NO_RATE_THRESHOLD == 112
    assert cs["outcome"] == "CAP_SATURATED_NO_RATE"
    assert cs["is_a_void"] is False
    assert "NO VIABILITY VERDICT" in cs["rule"]
    assert f"all {R.N_GAMES} games are played" in cs["rule"]


def test_the_frozen_plan_carries_H1s_OWN_rules_not_L0s(tasks):
    plan = P.load_h1_plan()
    assert "reserved H1 block" in " ".join(plan["abort_rules"])
    assert "L0 block" not in " ".join(plan["abort_rules"])
    assert L0RULES.per_cell_prohibition(28, 112) in plan["forbidden_claims"]
    assert L0RULES.per_cell_prohibition(8, 32) not in plan["forbidden_claims"]


def test_the_card_states_the_cap_saturation_branch_too():
    """The card and the artifact must not disagree about what happens."""
    card = open("docs/superpowers/2026-08-31-t1j-h1-h2h-viability-card.md").read()
    assert "CAP_SATURATED_NO_RATE" in card
    assert "no rate and no\nviability verdict" in card.lower()


# ═══════ [P2] H1's non-abort rules must describe H1's protocol, not L0's ══════

def test_L0s_NOT_ABORT_RULES_are_unchanged():
    assert L0RULES.NOT_ABORT_RULES == (
        "cap-termination saturation: caps never stop an L0 match; see CAP_NO_RATE_THRESHOLD",
        "score saturation: L0 measures a rate and has no band to saturate",
        "any early stop of any kind",
    )


def test_H1s_NON_ABORT_RULES_CONTAIN_NO_L0_WORDING():
    """🔴 Inherited whole, they imported two statements that are FALSE of H1: that
    caps never stop an "L0 match", and that the design "has no band to saturate".
    H1 HAS a band -- the 0.75 viability threshold -- and a >112 cap-saturated
    no-rate branch. The prohibitions are right; the reasons named the wrong match.
    """
    for rule in R.NOT_ABORT_RULES:
        assert "L0" not in rule, rule


def test_H1s_non_abort_rules_state_H1s_ACTUAL_protocol():
    joined = " ".join(R.NOT_ABORT_RULES)
    assert "H1 match" in joined
    assert "CAP_SATURATED_NO_RATE" in joined
    assert str(R.CAP_NO_RATE_THRESHOLD) in joined
    assert str(R.VIABILITY_THRESHOLD) in joined
    assert "H1 HAS a band" in joined


def test_the_substantive_early_stop_PROHIBITION_survives_verbatim():
    """Rewording the reasons must not weaken the rule itself."""
    assert L0RULES.EARLY_STOP_NOT_ABORT_RULE == "any early stop of any kind"
    assert L0RULES.EARLY_STOP_NOT_ABORT_RULE in R.NOT_ABORT_RULES
    assert L0RULES.EARLY_STOP_NOT_ABORT_RULE in L0RULES.NOT_ABORT_RULES


def test_NO_L0_ONLY_WORDING_REACHES_THE_FROZEN_H1_ARTIFACT():
    """The artifact is what a reader of the protocol actually sees. Scoped to the
    rule lists: `forbidden_claims` names L0 legitimately, in the no-pooling
    prohibition."""
    plan = P.load_h1_plan()
    for key in ("abort_rules", "not_abort_rules"):
        for rule in plan[key]:
            assert "L0" not in rule, (key, rule)
    assert any("H1 HAS a band" in r for r in plan["not_abort_rules"])
    assert any("CAP_SATURATED_NO_RATE" in r for r in plan["not_abort_rules"])
    # the legitimate mention, still present
    assert any(c.startswith("any pooling of H1 with L0") for c in plan["forbidden_claims"])


def test_the_DESIGN_layer_still_declares_no_gate_and_no_barrier():
    """⚠ THE DELIBERATE REPLACEMENT the previous version of this test demanded.

    It pinned the state in which H1 had NEITHER barrier and said, in its own
    docstring, that when the runner added them it "must be REPLACED by one
    asserting they bind -- deliberately, not by the claim quietly becoming true".
    Both now exist and the block is registered, so this is that replacement.

    What is still true, and worth holding: the DESIGN layer declares neither. The
    gate and the registration precondition live in the RUNNER, which is the only
    module that can execute anything. A gate in the rules or the plan would be a
    switch on a module that runs no games.
    """
    import scripts.GPU.alphazero.h1_viability_rules as _r
    import scripts.GPU.alphazero.h1_viability_plan as _p
    from scripts.GPU.alphazero import h1_viability_runner as _run
    for mod in (_r, _p):
        assert not any(n.endswith("_AUTHORIZED") for n in vars(mod))
        # BOTH SPELLINGS. A control adding the PRIVATE `_check_seed_registration`
        # was NOT CAUGHT while this looked only for the public name -- a barrier
        # appearing in the design layer is the defect whatever it is called.
        assert not any(n.lstrip("_") == "check_seed_registration" for n in vars(mod)), \
            f"a registration barrier appeared in {mod.__name__}"
    # and the runner holds BOTH, and they bind
    assert _run.H1_EXECUTION_AUTHORIZED is False
    with pytest.raises(_run.H1Error, match="H1_EXECUTION_AUTHORIZED is False"):
        _run.check_gate()
    _run.check_seed_registration()          # satisfied: attempt 2's block IS registered (2026-09-07)


def test_validate_task_executable_STILL_does_not_ask_the_accounted_question():
    """The reason the registration barrier had to exist SEPARATELY, unchanged by
    registration: `validate_task_executable` asks about consumed / exposed /
    retired seeds and NOT accounted ones. It would have accepted an unregistered
    H1 seed, and it accepts a registered one for the same reason -- it never
    asked. Shown against a seed in no registry at all."""
    task = dict(P.build_tasks(P.load_source_plan())[0], seed=202618000)
    assert not any(REF.seed_status(202618000).values())     # in NO registry
    REF.validate_task_executable(task)                      # and still accepted


def test_the_FROZEN_artifacts_seed_status_is_a_FREEZE_TIME_record_not_a_LIVE_one():
    """⚠ A DISCREPANCY THAT IS DELIBERATE, AND RECORDED RATHER THAN PAPERED OVER.

    v3 (attempt 1) says `seed_block_status: "PAPER-RESERVED, DELIBERATELY
    UNREGISTERED"`. That was true when it was frozen, FALSE after registration
    (2026-09-04), and the block is now spent and RETIRED (2026-09-05). The
    artifact is NOT rewritten: a preregistration edited whenever the world moves
    is not frozen. The LIVE status lives in the registry, which this test reads.

    v4 (attempt 2) says PAPER-RESERVED, UNREGISTERED. That was true when it was
    frozen (2026-09-07 morning) and is FALSE since the same day's registration
    and then the match itself: the block is ACCOUNTED, EXPOSED 224 and RETIRED
    WHOLE. The artifact is not rewritten, for the same reason. The field is a FREEZE-TIME record, and
    nothing in the runtime reads it to decide anything.
    """
    v3 = P.load_h1_plan(P.H1_ATTEMPT1_PLAN_REL, sha256=P.H1_ATTEMPT1_PLAN_SHA256,
                        task_digest=R.H1_ATTEMPT1_TASK_DIGEST)
    assert v3["seed_block_status"] == "PAPER-RESERVED, DELIBERATELY UNREGISTERED"
    lo, hi = R.H1_ATTEMPT1_SEED_BLOCK
    assert all(REF.seed_is_accounted(s) and REF.seed_is_retired(s) for s in range(lo, hi)), \
        "the live registry disagrees with this test, not with the frozen artifact"
    v4 = P.load_h1_plan()
    assert v4["seed_block_status"].startswith("PAPER-RESERVED, UNREGISTERED")
    lo, hi = R.H1_SEED_BLOCK
    # live: registered 2026-09-07, then the match ran and COMPLETED the same day
    assert all(REF.seed_is_accounted(s) and REF.seed_is_exposed(s) and REF.seed_is_retired(s)
               for s in range(lo, hi)), \
        "the live registry disagrees with this test, not with the frozen artifact"
    # and nothing in the RUNTIME reads that field to decide anything
    import inspect
    from scripts.GPU.alphazero import h1_viability_runner as _run
    for mod in (P, _run):
        assert "seed_block_status" not in inspect.getsource(mod)
