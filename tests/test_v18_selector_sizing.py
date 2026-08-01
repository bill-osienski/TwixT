"""Task 8 Step 3 -- the v18 role predicates, the sizing ladder, the record.

The module under test owns no selector and no threshold. It owns the TRIAL
LOOP -- because role assignment is per reservoir and the Sec 9.2.2 side geometry
is verified after selection -- and calls `fpu_dev_corpus_v2.sample_v2_rows` and
`v18_preflight_criteria.classify_role` for everything else. These tests are
written to fail if either of those is quietly replaced by a local copy.
"""
import collections
import json

import pytest

from scripts.GPU.alphazero import v18_preflight_criteria as criteria
from scripts.GPU.alphazero import v18_selector_sizing as M

CUTOFF = 0.50

# Base plies per cell, 20 apart. The red row sits on an even ply and the black
# row 13 later (odd), so a game's two rows are SIDE-OPPOSED, >= MIN_PLY_GAP
# apart, and honour the real "red opens, even plies are red" rule.
CELL_PLY = {
    ("representative", "opening"): 10,
    ("representative", "early_mid"): 40,
    ("representative", "midgame"): 70,
    ("representative", "late"): 100,
    ("target", "late"): 130,
    ("identity", "late"): 160,
    ("flip", "late"): 190,
}

# Measured values that make a row classify as each role at CUTOFF. `would_clip`
# is written for every grid cap and is non-increasing in the cap, because
# clipping is strict.
RECIPE = {
    "target": {"exposure_primary_0.50": 0.90, "sign_dominance": 0.90,
               "root_value_stm": 0.10, "eligible_depth2_leaves": 60,
               "would_clip_0.5": 2, "would_clip_0.75": 1, "would_clip_1.0": 0,
               "would_clip_1.25": 0, "clipped_amount_1.25": 0.0},
    "representative": {"exposure_primary_0.50": 0.10, "sign_dominance": 0.90,
                       "root_value_stm": 0.10, "eligible_depth2_leaves": 60,
                       "would_clip_0.5": 0, "would_clip_0.75": 0,
                       "would_clip_1.0": 0, "would_clip_1.25": 0,
                       "clipped_amount_1.25": 0.0},
    # abs(stm) > NEAR_EVEN, so these are NOT representative candidates and fall
    # through to step 3 -- the frozen precedence, not a convenience.
    "identity": {"exposure_primary_0.50": 0.10, "sign_dominance": 0.50,
                 "root_value_stm": 0.50, "eligible_depth2_leaves": 60,
                 "would_clip_0.5": 0, "would_clip_0.75": 0, "would_clip_1.0": 0,
                 "would_clip_1.25": 0, "clipped_amount_1.25": 0.0},
    "flip": {"exposure_primary_0.50": 0.10, "sign_dominance": 0.50,
             "root_value_stm": 0.50, "eligible_depth2_leaves": 60,
             "would_clip_0.5": 7, "would_clip_0.75": 6, "would_clip_1.0": 5,
             "would_clip_1.25": 4, "clipped_amount_1.25": 0.80},
}


def census_row(role, phase, ply, game, **over):
    row = {
        "population": "census",
        "game_content_sha1": f"{game:040x}",
        "game_idx": game,
        "position_ply": ply,
        "side_to_move": "red" if ply % 2 == 0 else "black",
        "canonical_state_sha1": f"{game:020x}{ply:020x}",
        "phase": phase,
        "n_legal": 450,
        **RECIPE[role],
    }
    row.update(over)
    return row


def cell_games():
    """Games per cell -- each a large multiple of that cell's demand.

    Sized from a real failure, not by eye: at 6 games per representative cell,
    one draw in 299 kept a single game and starved `representative|late` at
    capacity 2 against demand 4, so the tier scored 298/299 and the ladder was
    flaky. A cell needs only 2 games to meet a 4-row demand; at 12, losing ten
    of twelve in an 80% draw is not a thing that happens.
    """
    return {("target", "late"): 18, ("identity", "late"): 12,
            ("flip", "late"): 12, ("representative", "opening"): 12,
            ("representative", "early_mid"): 12,
            ("representative", "midgame"): 12, ("representative", "late"): 12}


def pool(skip=()):
    """A 90-game census. Each game serves ONE cell with a side-opposed pair, so
    every whole-game pick is side-neutral by construction."""
    rows, game = [], 0
    for (role, phase), n in cell_games().items():
        for _ in range(n):
            game += 1
            if (role, phase) in skip:
                continue
            base = CELL_PLY[(role, phase)]
            rows.append(census_row(role, phase, base, game))
            rows.append(census_row(role, phase, base + 13, game))
    return rows


def all_game_ids(rows=None):
    """Every universe game, INCLUDING any that yielded no classified row."""
    total = sum(cell_games().values())
    return [f"{g:040x}" for g in range(1, total + 1)]


def predicates():
    return M.role_predicates(CUTOFF)


def classify(rows=None):
    return M.classify_rows(pool() if rows is None else rows, predicates())


def role_of(probe):
    """The role the FULL frozen assignment gives one probe row.

    `classify_rows` is the whole procedure, quota and all, so a predicate cannot
    be probed with a one-row list -- step 2 would STOP on the shortfall. The
    probe is embedded in a pool that satisfies the quota instead, which is also
    the only way to observe a predicate the way the corpus will.

    Probes default to a canonical hash of "f"*40, which sorts LAST, so the
    quota draw takes the pool's own candidates and the probe falls through to
    step 3 -- unless a test deliberately gives it a hash that sorts first.
    """
    rows = pool() + [probe]
    result = M.classify_rows(rows, predicates())
    for label, members in result["by_role"].items():
        if any(member is probe for member in members):
            return label
    raise AssertionError("probe row was not classified at all")


def probe(role, phase, ply, sha="f" * 40, **over):
    row = census_row(role, phase, ply, 9999, **over)
    row["canonical_state_sha1"] = sha
    return row


# --- the exposure cutoff -----------------------------------------------------

def control_cohort(exposures):
    return [census_row("representative", "late", 100 + 2 * i, i,
                       **{"exposure_primary_0.50": e})
            for i, e in enumerate(exposures)]


def test_exposure_cutoff_uses_control_rows_only():
    rows = control_cohort([float(i) for i in range(30)])
    assert M.exposure_cutoff(rows) == 26.0
    rows[7]["population"] = "selected_a"
    with pytest.raises(ValueError, match="A rows"):
        M.exposure_cutoff(rows)


def test_exposure_cutoff_needs_the_frozen_cohort_size():
    with pytest.raises(ValueError, match="exactly 30"):
        M.exposure_cutoff(control_cohort([1.0] * 29))


def test_exposure_cutoff_is_nearest_rank_not_interpolated():
    """ceil(0.90 * 30) = 27, so the cutoff is the 27th smallest -- an OBSERVED
    datum. A linear-interpolation quantile would return 26.1, which no row
    attains and which therefore selects nothing at the tie."""
    exposures = [float(i) for i in range(30)]
    cutoff = M.exposure_cutoff(control_cohort(exposures))
    assert cutoff == 26.0
    assert cutoff in exposures
    # A linear-interpolation 0.90 quantile of 0..29 is 26.1 -- a value no row
    # attains, so the tie the rule promises to admit would not exist.
    assert cutoff != 26.1


def test_exposure_cutoff_is_order_independent():
    """The frozen total ordering is applied inside, so a shuffled cohort cannot
    move a tie."""
    exposures = [float(i % 7) for i in range(30)]
    forward = M.exposure_cutoff(control_cohort(exposures))
    backward = M.exposure_cutoff(list(reversed(control_cohort(exposures))))
    assert forward == backward


def test_cutoff_ties_are_admitted_by_the_predicate():
    """`>` would drop the row sitting exactly on a cutoff that nearest-rank
    guarantees exists."""
    assert criteria.meets_exposure_cutoff(CUTOFF, CUTOFF) is True
    assert role_of(probe("target", "late", 130,
                         **{"exposure_primary_0.50": CUTOFF})) == "target"


def test_matched_control_rows_joins_on_content_sha1_and_ply():
    rows = pool()
    cohort = [{"game_content_sha1": r["game_content_sha1"],
               "position_ply": r["position_ply"],
               "canonical_state_sha1": r["canonical_state_sha1"]}
              for r in rows[:3]]
    assert M.matched_control_rows(cohort, rows) == rows[:3]
    cohort[1]["position_ply"] = 9999
    with pytest.raises(ValueError, match="not in the census"):
        M.matched_control_rows(cohort, rows)


def test_matched_control_rows_refuses_a_mismatched_canonical_state():
    rows = pool()
    cohort = [{"game_content_sha1": rows[0]["game_content_sha1"],
               "position_ply": rows[0]["position_ply"],
               "canonical_state_sha1": "0" * 40}]
    with pytest.raises(ValueError, match="canonical state"):
        M.matched_control_rows(cohort, rows)


# --- the role partition ------------------------------------------------------

def test_role_predicates_resolve_the_cutoff_placeholder():
    resolved = predicates()["roles"]["target"]["conditions"]
    values = [c["value"] for c in resolved]
    assert CUTOFF in values
    assert "EXPOSURE_CUTOFF" not in values
    # The rest of the frozen conditions are carried through untouched.
    assert [c["variable"] for c in resolved] == [
        c["variable"] for c in
        criteria.ROLE_ASSIGNMENT["roles"]["target"]["conditions"]]


def test_role_predicates_import_the_flip_rule_rather_than_restating_it():
    """Revision 3 restated it here as `and/or` and weakened a frozen AND."""
    assert predicates()["flip_control_exposure"] is criteria.FLIP_CONTROL_EXPOSURE
    assert criteria.FLIP_CONTROL_EXPOSURE["operator"] == "AND"


def test_role_assignment_is_total_and_exclusive():
    result = classify()
    rows = pool()
    assert sum(result["counts"].values()) == len(rows)
    seen = [id(r) for label in criteria.ROLE_LABELS for r in result["by_role"][label]]
    assert len(seen) == len(set(seen)) == len(rows)
    assert result["counts"]["target"] == 36
    # EXACTLY the quota -- step 2 is a draw, not a label.
    assert result["counts"]["representative"] == 16
    # The 80 near-even rows the quota did NOT take fall through to step 3 and
    # join the 24 that were never near-even. Labelling all 96 as
    # representatives would have hidden every one of them from identity.
    assert result["counts"]["identity"] == 104
    assert result["counts"]["flip"] == 24


def test_the_representative_quota_is_four_per_phase_and_eight_eight_by_side():
    """Spec Sec 9.2.1 (four per phase) and Sec 9.2.2 (8/8) at the point the
    quota is drawn, not merely on the manifest."""
    reps = [r for r in classify()["selector_rows"] if r["role"] == "representative"]
    assert len(reps) == 16
    assert collections.Counter(r["phase"] for r in reps) == {
        "opening": 4, "early_mid": 4, "midgame": 4, "late": 4}
    assert collections.Counter(r["side"] for r in reps) == {"red": 8, "black": 8}


def test_the_quota_draw_reaches_eight_eight_from_a_side_clustered_pool():
    """The draw steers on the side it is BEHIND on, across phases.

    Taking candidates in bare hash order works only while each phase's hash
    order happens to alternate. Here `opening` offers its four red candidates
    ahead of any black one, so an unsteered draw would take 4/0 there and land
    the role at 10/6 -- a manifest the Sec 9.2.2 check would then reject for no
    reason other than the draw order.
    """
    rows, game = [], 0
    for phase in ("opening", "early_mid", "midgame", "late"):
        base = CELL_PLY[("representative", phase)]
        for k in range(4):                      # 4 games -> 8 candidates/phase
            game += 1
            for ply in (base, base + 13):       # even -> red, odd -> black
                row = census_row("representative", phase, ply, game)
                side = "red" if ply % 2 == 0 else "black"
                # `opening` sorts every red candidate ahead of every black one;
                # the other phases alternate by game as usual.
                rank = (("a" if side == "red" else "z") if phase == "opening"
                        else f"{game:03d}")
                row["canonical_state_sha1"] = f"{phase[:3]}{rank}{ply:030d}"
                rows.append(row)
    assert len([r for r in rows if r["phase"] == "opening"]) == 8
    drawn = M._draw_representatives(rows)
    assert len(drawn) == 16
    assert collections.Counter(r["side_to_move"] for r in drawn) == {
        "red": 8, "black": 8}


def test_a_representative_shortfall_stops_rather_than_truncating():
    """`on_shortfall` is STOP: a phase that cannot fill its quota is a failure,
    not a corpus with three representatives in it."""
    opening = sorted({r["game_idx"] for r in pool() if r["phase"] == "opening"})
    keep = opening[0]                       # ONE game -> two rows against a quota of four
    thin = [r for r in pool()
            if r["phase"] != "opening" or r["game_idx"] == keep]
    assert len([r for r in thin if r["phase"] == "opening"]) == 2
    with pytest.raises(ValueError, match="representative shortfall.*STOP"):
        M.classify_rows(thin, predicates())


def test_surplus_representative_candidates_remain_available_to_step_three():
    """The blocker this closes: consuming every near-even non-target row as a
    representative leaves identity and flip drawing only from rows the near-even
    rule excludes, which measures a role geometry the corpus will never have."""
    result = classify()
    candidates = [r for r in pool()
                  if abs(r["root_value_stm"]) <= criteria.NEAR_EVEN["value"]
                  and r not in result["by_role"]["target"]]
    assert len(candidates) == 96                      # eligible
    assert result["counts"]["representative"] == 16   # drawn
    surplus = [r for r in candidates if r in result["by_role"]["identity"]]
    assert len(surplus) == 80                         # still reachable by step 3


def test_representatives_are_chosen_before_identity_and_flip_exist():
    """The representative candidate set is conditioned on TARGET STATUS ONLY.

    This row is identity-eligible (`would_clip_0.5 == 0`) and near-even, and it
    still classifies as a representative -- which is exactly what proves the
    candidate set was not first stripped of identity-eligible rows. A
    residual-conditioned candidate set would not be residual-independent.
    """
    row = probe("representative", "late", 100, sha="0" * 40)
    assert row["would_clip_0.5"] == criteria.IDENTITY_WITNESS["value"]
    assert row["side_to_move"] == "red"
    assert role_of(row) == "representative"


def test_target_flip_overlap_resolves_to_flip_control():
    """A row can satisfy both target exposure and flip exposure; the frozen
    `not is_flip` condition gives the scarcer role priority."""
    row = probe("target", "late", 130,
                **{"would_clip_1.25": 4, "would_clip_1.0": 5,
                   "would_clip_0.75": 6, "would_clip_0.5": 7,
                   "clipped_amount_1.25": 0.80, "root_value_stm": 0.50})
    assert role_of(row) == "flip"


def test_representative_ordering_key_is_residual_independent():
    key = predicates()["representative_ordering_key"]
    assert key == "canonical_state_sha1"
    assert key not in criteria.ABSOLUTE_RESIDUAL_VARIABLES
    assert key != criteria.PRIMARY_EXPOSURE_COLUMN
    assert not key.startswith(("would_clip", "clipped_amount", "exposure"))


def test_role_assignment_refuses_rather_than_reordering():
    """Step 2 can starve step 3, and the frozen answer is to STOP -- never to
    revisit representatives to free identity/flip supply."""
    from scripts.GPU.alphazero import fpu_dev_corpus_v2 as V
    assert criteria.ROLE_ASSIGNMENT["on_shortfall"] == "STOP"
    assert predicates()["revisit_representatives_after_residual_roles"] is False
    starved = M.classify_rows(pool(skip=[("flip", "late")]), predicates())
    profile = M.schema5_profile()
    report = V.post_screen_qualification_report(starved["selector_rows"], profile)
    assert report["status"] == "GATE_FAIL"
    assert "flip_control|late" in report["binding_constraint"]
    # The exact selector RAISES rather than emitting a manifest that reassigns
    # another role's rows to cover the shortfall.
    with pytest.raises(ValueError, match="capacity|shortfall"):
        V.sample_v2_rows(starved["selector_rows"], seed=1, alloc=profile)


def test_identity_witness_predicate_cannot_bind_at_any_grid_cap():
    """`would_clip_0.5 == 0` says no eligible residual exceeds 0.50, so the tree
    is byte-identical to shipped at EVERY grid cap."""
    row = probe("identity", "late", 160)
    assert role_of(row) == "identity"
    for cap in criteria.CAP_GRID:
        assert row[f"would_clip_{cap}"] == 0


def test_a_census_row_that_clips_more_at_a_stronger_cap_is_refused():
    """Without this the identity and flip claims are vacuous: a corrupt row
    could report zero clipping at 0.50 and three above 1.25 at once."""
    row = probe("identity", "late", 160, **{"would_clip_1.25": 3})
    with pytest.raises(ValueError, match="monotone in the cap"):
        M.classify_rows(pool() + [row], predicates())


def test_flip_control_predicate_requires_material_exposure_at_1_25():
    """The operator is AND: a count of three with a trivial clipped amount is
    not a flip control."""
    row = probe("flip", "late", 190, **{"clipped_amount_1.25": 0.10})
    assert role_of(row) == "unassigned"


def test_flip_control_exposed_at_1_25_is_exposed_at_every_stronger_cap():
    row = probe("flip", "late", 190)
    assert role_of(row) == "flip"
    weakest = row["would_clip_1.25"]
    for cap in criteria.CAP_GRID:
        assert row[f"would_clip_{cap}"] >= weakest


def test_selector_rows_carry_the_v18_cell_vocabulary_and_content_identity():
    result = classify()
    assert {r["role"] for r in result["selector_rows"]} == {
        "target", "identity_witness", "flip_control", "representative"}
    row = result["selector_rows"][0]
    assert len(row["game_idx"]) == 40        # replay CONTENT sha1, not an index
    assert set(row) == {"game_idx", "ply", "side", "role", "phase", "band",
                        "canonical_sha1"}


def test_classify_refuses_a_row_missing_a_required_census_field():
    row = probe("target", "late", 130)
    row.pop("sign_dominance")
    with pytest.raises(ValueError, match="missing required field"):
        M.classify_rows(pool() + [row], predicates())


# --- the ladder --------------------------------------------------------------

def test_schema5_profile_is_the_frozen_forty_row_geometry():
    profile = M.schema5_profile()
    assert profile.schema_version == 5
    assert profile.corpus_size == 40
    assert profile.splits == ("all",)
    assert profile.band_floor_cell is None
    assert profile.run_kind == "v18_preflight_sizing"


def test_sizing_ladder_reports_the_smallest_qualifying_tier():
    """A tier passes only on an EXACT-SELECTOR witness, and only if every one of
    the 299 trials produced one."""
    ladder = M.sizing_ladder(pool(), predicates(), all_game_ids=all_game_ids(),
                             tiers=(12, 72))
    by_tier = {t["n_games"]: t for t in ladder}
    assert by_tier[12]["meets_criterion"] is False
    assert by_tier[12]["n_successes"] == 0
    assert by_tier[72]["meets_criterion"] is True
    assert by_tier[72]["n_successes"] == by_tier[72]["n_trials"] == 299
    assert M.smallest_qualifying_tier(ladder) == 72
    assert by_tier[72]["witness_trial_index"] == 0
    assert by_tier[12]["witness_trial_index"] is None


def test_the_witness_is_a_trial_index_not_a_content_sha_prefix():
    """A content-SHA prefix can fail even when every random trial passes, so it
    is not the promised witness."""
    ladder = M.sizing_ladder(pool(), predicates(), all_game_ids=all_game_ids(),
                             tiers=(72,))
    assert ladder[0]["witness_trial_index"] == 0
    assert isinstance(ladder[0]["witness_trial_index"], int)


def test_the_full_universe_tier_is_one_degenerate_trial():
    """Drawing every game from the universe returns the same set each time, so
    299 repetitions cannot estimate a success probability."""
    ladder = M.sizing_ladder(pool(), predicates(), all_game_ids=all_game_ids(),
                             tiers=(90,), trials=299)
    assert ladder[0]["n_trials"] == 1
    assert ladder[0]["degenerate_full_universe"] is True
    assert ladder[0]["meets_criterion"] is False
    assert M.smallest_qualifying_tier(ladder) is None


def test_sizing_ladder_is_deterministic_under_a_frozen_seed():
    kwargs = dict(all_game_ids=all_game_ids(), tiers=(24,), trials=9)
    first = M.sizing_ladder(pool(), predicates(), seed=20260729, **kwargs)
    second = M.sizing_ladder(pool(), predicates(), seed=20260729, **kwargs)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert criteria.SIZING["seed"] == 20260729


def test_the_frozen_ladder_defaults_come_from_the_criteria_module():
    assert criteria.SIZING["probabilistic_tiers"] == (200, 300, 400, 500, 600, 700)
    assert criteria.SIZING["degenerate_tier"] == 800
    assert criteria.SIZING["trials_per_probabilistic_tier"] == 299


def test_recommended_operational_size_is_the_next_tier_up():
    ladder = [{"n_games": n, "meets_criterion": n >= 400,
               "degenerate_full_universe": n == 800}
              for n in (200, 300, 400, 500, 600, 700, 800)]
    assert M.smallest_qualifying_tier(ladder) == 400
    assert M.recommended_operational_size(ladder) == 500


def test_the_degenerate_tier_may_serve_as_the_next_tier_up():
    ladder = [{"n_games": n, "meets_criterion": n == 700,
               "degenerate_full_universe": n == 800}
              for n in (200, 300, 400, 500, 600, 700, 800)]
    assert M.recommended_operational_size(ladder) == 800


def test_a_degenerate_tier_cannot_be_the_qualifying_evidence():
    """Even if the full-universe tier reported a pass, ONE trial cannot estimate
    a success probability -- it may serve as an operational size, never as the
    evidence that a size qualifies."""
    ladder = [{"n_games": n, "meets_criterion": n == 800,
               "degenerate_full_universe": n == 800}
              for n in (200, 300, 400, 500, 600, 700, 800)]
    assert M.smallest_qualifying_tier(ladder) is None
    assert M.recommended_operational_size(ladder) is None


def test_no_qualifying_tier_reports_no_size_rather_than_the_largest():
    ladder = [{"n_games": n, "meets_criterion": False,
               "degenerate_full_universe": n == 800}
              for n in (200, 300, 400, 500, 600, 700, 800)]
    assert M.smallest_qualifying_tier(ladder) is None
    assert M.recommended_operational_size(ladder) is None


def test_sizing_reports_operating_characteristics_per_count_gate():
    ladder = M.sizing_ladder(pool(), predicates(), all_game_ids=all_game_ids(),
                             tiers=(24, 90), trials=9)
    table = M.operating_characteristics(ladder)
    assert set(table) == {"24", "90"}
    # 9 trials cannot clear the 0.99 rule even at 9/9, so the gate reports that
    # it is unattainable rather than pretending to a shortfall.
    assert table["24"]["successes_required_to_pass"] is None
    assert table["24"]["shortfall"] is None
    assert table["90"]["degenerate_full_universe"] is True


def test_a_partial_pass_refuses_to_name_trial_zero_as_the_witness():
    """Trial 0 is a witness only because passing REQUIRES all 299 successes. If
    the rule ever gains slack, the successful index must come from the trial
    loop -- inventing one here would name a trial that failed."""
    with pytest.raises(ValueError, match="no longer guaranteed"):
        M._witness_trial_index({"meets_criterion": True, "n_successes": 298,
                                "n_trials": 299})
    assert M._witness_trial_index(
        {"meets_criterion": True, "n_successes": 299, "n_trials": 299}) == 0


def test_the_count_gate_requires_every_one_of_299_trials():
    """298 of 299 gives 0.98999 and fails; the rule has no slack, which is what
    makes trial 0 a guaranteed witness at a passing tier."""
    assert M._successes_required(299) == 299
    assert M._successes_required(1) is None


def test_infeasible_geometry_fails_rather_than_relaxing():
    """A pool that cannot fill a cell produces zero successes -- never a
    smaller manifest and never a relaxed quota.

    Flip is the genuinely scarce role: nothing else in the pool clips three
    leaves above 1.25 with a material clipped amount. Starving IDENTITY no
    longer starves the cell, because the representative quota's surplus falls
    through to it -- which is exactly the geometry correction under test above.
    """
    starved = pool(skip=[("flip", "late")])
    ladder = M.sizing_ladder(starved, predicates(), all_game_ids=all_game_ids(),
                             tiers=(72,), trials=3)
    assert ladder[0]["n_successes"] == 0
    assert ladder[0]["meets_criterion"] is False
    assert ladder[0]["failure_reasons"]
    assert M.smallest_qualifying_tier(ladder) is None


# --- corrupt evidence is not a failed trial ----------------------------------

@pytest.mark.parametrize("corrupt, err", [
    (lambda r: r.pop("sign_dominance"), "missing required field"),
    (lambda r: r.update({"would_clip_1.25": 99}), "monotone in the cap"),
])
def test_corrupt_census_evidence_invalidates_the_measurement(corrupt, err):
    """A missing field or an impossible clip series is bad EVIDENCE, not a
    reservoir that failed to yield a corpus. Counting it as a failed trial would
    quietly move the sizing probability instead of stopping the run."""
    rows = pool()
    corrupt(rows[0])
    with pytest.raises(ValueError, match=err):
        M.sizing_ladder(rows, predicates(), all_game_ids=all_game_ids(),
                        tiers=(72,), trials=3)


def test_the_census_is_validated_before_any_sampling():
    """Validation precedes the trial loop, so a corrupt census cannot produce a
    partial ladder that was already counting trials when it stopped."""
    rows = pool()
    rows[-1].pop("would_clip_0.5")
    with pytest.raises(ValueError, match="missing required field"):
        M.sizing_ladder(rows, predicates(), all_game_ids=all_game_ids(),
                        tiers=(72,), trials=299)


def test_only_a_representative_shortfall_becomes_a_trial_failure():
    assert issubclass(M.RepresentativeShortfall, ValueError)
    starved = pool(skip=[("flip", "late")])
    ladder = M.sizing_ladder(starved, predicates(), all_game_ids=all_game_ids(),
                             tiers=(72,), trials=3)
    assert ladder[0]["n_successes"] == 0          # a real geometry failure ...
    assert sum(ladder[0]["failure_reasons"].values()) == 3   # ... fully accounted


def test_a_trial_swallows_only_a_representative_shortfall(monkeypatch):
    """The CATCH BOUNDARY, asserted directly.

    Widening it to `except ValueError` cannot be seen through the ladder --
    pre-validation means no corrupt row ever reaches a trial -- so the boundary
    is probed where it lives. Any other error must propagate and end the run;
    only a shortfall becomes a counted failure.
    """
    profile = M.schema5_profile()

    def other_defect(_candidates):
        raise ValueError("a defect that is not a shortfall")

    monkeypatch.setattr(M, "_draw_representatives", other_defect)
    with pytest.raises(ValueError, match="not a shortfall"):
        M._run_trial(pool(), predicates(), profile, selection_seed=1)

    def shortfall(_candidates):
        raise M.RepresentativeShortfall("representative shortfall: phase late")

    monkeypatch.setattr(M, "_draw_representatives", shortfall)
    reason = M._run_trial(pool(), predicates(), profile, selection_seed=1)
    assert reason.startswith("classify: representative shortfall")


# --- the representative draw carries the corpus geometry ---------------------

def test_the_draw_respects_max_per_game():
    """The sixteen drawn rows are the ONLY representatives the selector sees --
    zero slack -- so a draw over the per-game cap hands it an inadmissible set.

    One game is made to offer FIVE well-separated late rows that sort ahead of
    everything else. An uncapped draw takes four of them and the corpus would
    then hold four positions from one game.
    """
    from scripts.GPU.alphazero.fpu_dev_corpus_v2 import MAX_PER_GAME
    rows, game = [], 0
    for phase in ("opening", "early_mid", "midgame", "late"):
        base = CELL_PLY[("representative", phase)]
        for _ in range(6):
            game += 1
            rows.append(census_row("representative", phase, base, game))
            rows.append(census_row("representative", phase, base + 13, game))
    greedy = 9999
    # ALTERNATING sides, so side steering never has a reason to look past this
    # game: whichever side the draw is behind on, this game offers it. Only the
    # per-game cap stops all four late picks coming from here.
    for k, ply in enumerate((100, 113, 126, 139, 152)):
        row = census_row("representative", "late", ply, greedy)
        # All five sort ahead of every pool row: the distinguishing digit is at
        # the END, so "0001..." cannot fall behind a pool hash the way a
        # leading-digit scheme does.
        row["canonical_state_sha1"] = f"0000{k:036d}"
        rows.append(row)

    drawn = M._draw_representatives(rows)
    per_game = collections.Counter(r["game_content_sha1"] for r in drawn)
    assert max(per_game.values()) <= MAX_PER_GAME
    assert per_game[f"{greedy:040x}"] == MAX_PER_GAME


def test_the_draw_respects_min_ply_gap():
    """Two rows drawn from one game must be at least MIN_PLY_GAP apart. Here
    every game offers a pair only 4 plies apart, so the draw must take at most
    one row per game and reach across more games instead."""
    from scripts.GPU.alphazero.fpu_dev_corpus_v2 import MIN_PLY_GAP
    rows, game = [], 0
    for phase in ("opening", "early_mid", "midgame", "late"):
        base = CELL_PLY[("representative", phase)]
        for _ in range(8):
            game += 1
            rows.append(census_row("representative", phase, base, game))
            rows.append(census_row("representative", phase, base + 4, game))
    drawn = M._draw_representatives(rows)
    assert len(drawn) == 16
    by_game = collections.defaultdict(list)
    for row in drawn:
        by_game[row["game_content_sha1"]].append(row["position_ply"])
    assert all(len(plies) == 1 for plies in by_game.values())
    for plies in by_game.values():
        for a, b in zip(sorted(plies), sorted(plies)[1:]):
            assert b - a >= MIN_PLY_GAP


def test_the_draw_orders_by_a_total_key_not_a_bare_hash():
    """A bare canonical-hash sort leaves ties to input order -- which is
    whatever the census reader happened to yield."""
    a = census_row("representative", "late", 100, 1)
    b = census_row("representative", "late", 114, 2)
    for row in (a, b):
        row["canonical_state_sha1"] = "c" * 40      # a deliberate tie
    assert M._representative_order_key(a) < M._representative_order_key(b)
    assert M._representative_order_key(a)[0] == M._representative_order_key(b)[0]
    assert len(M._representative_order_key(a)) == 3


# --- exact per-role side geometry, spec Sec 9.2.2 ----------------------------

# No representative SURPLUS (exactly the quota) and a black-only identity cell.
# The selector steers the SPLIT's aggregate, so a cell it can only fill in black
# forces it to compensate in red elsewhere -- producing a manifest that fills
# every cell, is 20/20 overall, and is skewed inside two roles.
SKEW_GAMES = {("target", "late"): 18, ("identity", "late"): 12,
              ("flip", "late"): 12, ("representative", "opening"): 2,
              ("representative", "early_mid"): 2,
              ("representative", "midgame"): 2, ("representative", "late"): 2}


def skewed_pool():
    rows, game = [], 0
    for (role, phase), n in SKEW_GAMES.items():
        for _ in range(n):
            game += 1
            base = CELL_PLY[(role, phase)]
            plies = (base + 1, base + 15) if role == "identity" else (base, base + 13)
            for ply in plies:
                rows.append(census_row(role, phase, ply, game))
    return rows


def test_role_side_geometry_is_derived_and_matches_the_spec_table():
    assert M.ROLE_SIDE_GEOMETRY == {"target": (8, 8), "identity_witness": (2, 2),
                                    "flip_control": (2, 2),
                                    "representative": (8, 8)}
    assert sum(sum(pair) for pair in M.ROLE_SIDE_GEOMETRY.values()) == 40


def test_a_globally_balanced_but_role_skewed_manifest_is_refused():
    """The defect this closes: `side_tol` constrains only the split aggregate,
    so 20/20 overall says nothing about the geometry inside each role."""
    from scripts.GPU.alphazero import fpu_dev_corpus_v2 as V
    profile = M.schema5_profile()
    rows = M.classify_rows(skewed_pool(), predicates())["selector_rows"]
    assert V.post_screen_qualification_report(rows, profile)["status"] == "PASS"
    manifest, _stats = V.sample_v2_rows(rows, seed=criteria.SIZING["seed"],
                                        alloc=profile)

    reds = sum(1 for r in manifest if r["side"] == "red")
    assert len(manifest) == 40 and reds == 20              # 20/20 overall ...
    assert abs(reds - (40 - reds)) <= profile.side_tol     # ... and side_tol PASSES
    assert collections.Counter(
        r["side"] for r in manifest if r["role"] == "target") == {
            "red": 10, "black": 6}                          # ... but roles skew

    with pytest.raises(ValueError, match="Sec 9.2.2"):
        M.assert_role_side_balance(manifest)


def test_a_role_skewed_manifest_is_not_a_sizing_success():
    """End to end: the trial must not count a filled-but-skewed manifest."""
    ladder = M.sizing_ladder(
        skewed_pool(), predicates(),
        all_game_ids=[f"{g:040x}" for g in range(1, sum(SKEW_GAMES.values()) + 1)],
        tiers=(50,), trials=1)
    assert ladder[0]["n_successes"] == 0
    assert any(reason.startswith("side:")
               for reason in ladder[0]["failure_reasons"])


def test_a_correctly_balanced_manifest_passes_the_role_geometry():
    manifest = [{"role": role, "side": side}
                for role, (red, black) in M.ROLE_SIDE_GEOMETRY.items()
                for side, n in (("red", red), ("black", black))
                for _ in range(n)]
    assert len(manifest) == 40
    M.assert_role_side_balance(manifest)


# --- the record --------------------------------------------------------------

def frozen_ladder(passing=400):
    """A ladder whose derived fields are COMPUTED, not written by hand -- the
    emitter recomputes each one, so a hand-rounded bound would be refused."""
    from scripts.GPU.alphazero.fpu_dev_corpus_v2 import _binomial_lower_bound
    ladder = []
    for n in (200, 300, 400, 500, 600, 700, 800):
        trials = 1 if n == 800 else 299
        successes = trials if (n >= passing and n != 800) else 0
        bound = _binomial_lower_bound(successes, trials, 0.05)
        floor = criteria.SIZING["minimum_lower_bound"]
        failures = trials - successes
        ladder.append({
            "n_games": n, "n_trials": trials, "n_successes": successes,
            "success_rate": successes / trials,
            "lower_bound_95": bound,
            "meets_criterion": bound >= floor,
            "degenerate_full_universe": n == 800,
            # Every failed trial has exactly one recorded reason.
            "failure_reasons": ({"qualify: target|late capacity 0 < demand 16":
                                 failures} if failures else {}),
            "witness_trial_index": 0 if bound >= floor else None})
    return ladder


def bindings():
    return {"criteria_sha1": "a" * 40, "universe_sha1": "b" * 40,
            "census_sha1": "c" * 40, "matched_cohort_sha1": "d" * 40}


def emit(tmp_path, ladder=None, **over):
    kwargs = dict(cutoff=CUTOFF, classification=classify(),
                  ladder=frozen_ladder() if ladder is None else ladder,
                  predicates=predicates(), bindings=bindings())
    kwargs.update(over)
    return M.emit_sizing_record(str(tmp_path / "sizing.json"), **kwargs)


def test_sizing_record_is_byte_reproducible_and_forbids_interpretation(tmp_path):
    first = emit(tmp_path)
    payload = json.loads((tmp_path / "sizing.json").read_bytes())
    second = emit(tmp_path)
    assert first == second
    assert payload["scientific_interpretation_forbidden"] is True
    assert payload["run_kind"] == "v18_preflight_sizing"
    assert payload["exposure_cutoff"] == CUTOFF
    assert payload["smallest_qualifying_tier"] == 400
    assert payload["recommended_operational_size"] == 500
    assert payload["sizing_status"] == "SIZING_PASSES"
    assert payload["role_counts"]["target"] == 36
    assert set(payload["operating_characteristics"]) == {
        "200", "300", "400", "500", "600", "700", "800"}
    assert payload["profile"]["schema_version"] == 5
    assert payload["criteria_sha1"] == "a" * 40


def test_a_failed_sizing_is_recorded_as_a_failure(tmp_path):
    emit(tmp_path, ladder=frozen_ladder(passing=9999))
    payload = json.loads((tmp_path / "sizing.json").read_bytes())
    assert payload["sizing_status"] == "SIZING_FAILS"
    assert payload["smallest_qualifying_tier"] is None
    assert payload["recommended_operational_size"] is None


@pytest.mark.parametrize("name", M._REQUIRED_BINDINGS)
def test_every_input_binding_must_be_a_real_sha1(tmp_path, name):
    dropped = {k: v for k, v in bindings().items() if k != name}
    with pytest.raises(ValueError, match="missing input binding"):
        emit(tmp_path, bindings=dropped)


@pytest.mark.parametrize("value, why", [
    ("short", "too short"),
    ("z" * 40, "forty characters, not one of them hex"),
    ("A" * 40, "uppercase is not the canonical form"),
    ("a" * 39 + "!", "right length, wrong alphabet"),
    (40 * 1, "not a string at all"),
])
def test_a_binding_must_be_canonical_lowercase_hex(tmp_path, value, why):
    """Length is not authentication: `"z" * 40` identifies no artifact that can
    exist, and an uppercase digest is not the form every other v18 record
    writes."""
    bad = bindings()
    bad["census_sha1"] = value
    with pytest.raises(ValueError, match="lowercase hex sha1"):
        emit(tmp_path, bindings=bad)


def test_the_record_refuses_a_cutoff_the_classification_did_not_run_at(tmp_path):
    with pytest.raises(ValueError, match="classification ran at cutoff"):
        emit(tmp_path, cutoff=0.99)


def test_the_record_refuses_predicates_resolved_at_another_cutoff(tmp_path):
    """The classification agrees with the record; only the published predicate
    set disagrees -- which is precisely the case a cutoff-only check misses."""
    with pytest.raises(ValueError, match="predicates resolve cutoff"):
        emit(tmp_path, predicates=M.role_predicates(0.99))


def test_the_record_refuses_a_ladder_that_is_not_the_frozen_one(tmp_path):
    short = [t for t in frozen_ladder() if t["n_games"] != 600]
    with pytest.raises(ValueError, match="frozen ladder"):
        emit(tmp_path, ladder=short)


@pytest.mark.parametrize("field, value, err", [
    # The headline fabrication: zero successes, claimed as a pass.
    ("meets_criterion", True, "claims meets_criterion"),
    ("success_rate", 1.0, "reports success_rate"),
    ("lower_bound_95", 0.995, "reports lower_bound_95"),
    ("witness_trial_index", 0, "names witness trial"),
    ("n_successes", 400, "successes in 299 trials"),
    ("n_trials", 5, "ran 5 trials"),
    ("degenerate_full_universe", True, "claims degenerate"),
])
def test_the_record_recomputes_every_derived_ladder_field(tmp_path, field, value,
                                                          err):
    """A ladder is two primitive counts and six functions of them. Trusting any
    of the six lets a hand-edited artifact report SIZING_PASSES over a tier that
    never produced a single selector witness."""
    ladder = frozen_ladder(passing=9999)          # every tier failed, honestly
    failing = next(t for t in ladder if t["n_games"] == 200)
    failing[field] = value
    with pytest.raises(ValueError, match=err):
        emit(tmp_path, ladder=ladder)


def test_a_fabricated_pass_cannot_reach_the_artifact(tmp_path):
    """The whole fabrication at once: 0/299 dressed as a passing tier."""
    ladder = frozen_ladder(passing=9999)
    forged = next(t for t in ladder if t["n_games"] == 400)
    forged.update({"meets_criterion": True, "success_rate": 1.0,
                   "lower_bound_95": 0.99003, "witness_trial_index": 0})
    with pytest.raises(ValueError, match="reports success_rate"):
        emit(tmp_path, ladder=ladder)
    assert not (tmp_path / "sizing.json").exists()


@pytest.mark.parametrize("reasons, err", [
    ({}, "failure reasons totalling 0"),
    ({"qualify: x": 5}, "failure reasons totalling 5"),
    ({"qualify: x": 298, "select: y": 2}, "failure reasons totalling 300"),
    ({"qualify: x": 0}, "failure reasons totalling 0"),
    ({"qualify: x": -1}, "failure reasons"),
    ({"qualify: x": True}, "failure reasons"),
    # SUM-CORRECT but malformed: a sum-only check would pass all three.
    ({"qualify: x": 300, "select: y": -1}, "failure reasons"),
    ({"qualify: x": 298, "select: y": True}, "failure reasons"),
    ({"qualify: x": 299, "select: y": 0}, "failure reasons"),
])
def test_published_failure_counts_must_account_for_every_failed_trial(
        tmp_path, reasons, err):
    """0/299 with an empty reason map says every trial failed for no recorded
    reason, which is not a thing a measurement can report."""
    ladder = frozen_ladder(passing=9999)
    next(t for t in ladder if t["n_games"] == 200)["failure_reasons"] = reasons
    with pytest.raises(ValueError, match=err):
        emit(tmp_path, ladder=ladder)


def test_a_passing_tier_carries_no_failure_reasons(tmp_path):
    ladder = frozen_ladder()
    passing = next(t for t in ladder if t["n_games"] == 400)
    assert passing["failure_reasons"] == {}
    passing["failure_reasons"] = {"qualify: x": 1}
    with pytest.raises(ValueError, match="failure reasons totalling 1"):
        emit(tmp_path, ladder=ladder)


def test_the_pass_rule_numbers_come_from_the_criteria_module():
    """alpha and the lower-bound floor are read, not restated -- the module
    docstring's "no restated thresholds" claim has to be true of these two."""
    import inspect
    source = inspect.getsource(M)
    assert 'criteria.SIZING["alpha"]' in source or 'sizing["alpha"]' in source
    for literal in ("0.05", "0.99"):
        bare = [line for line in source.splitlines()
                if literal in line and not line.lstrip().startswith("#")]
        assert not bare, f"{literal} is restated at: {bare}"


def test_a_rounded_lower_bound_is_not_close_enough(tmp_path):
    """0.99003 is the docstring's ROUNDED form; the recomputation is exact."""
    ladder = frozen_ladder()
    passing = next(t for t in ladder if t["n_games"] == 400)
    assert passing["lower_bound_95"] != 0.99003
    passing["lower_bound_95"] = 0.99003
    with pytest.raises(ValueError, match="reports lower_bound_95"):
        emit(tmp_path, ladder=ladder)
