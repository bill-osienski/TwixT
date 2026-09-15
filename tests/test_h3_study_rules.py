"""H3 FULL STUDY — the frozen constants, the strata, the segments, the schedule.

Binds `docs/superpowers/2026-09-15-t1j-h3-full-study-design.md` as amended twice.
Nothing here plays, loads, draws a match seed or generates stratum B.
"""
import math

import pytest

from scripts.GPU.alphazero import h3_study_rules as R


# ───────────────────────── the card's numbers ──────────────────────────────

def test_the_card_numbers_are_the_module_numbers():
    assert R.N_PAIRS == 296
    assert R.N_GAMES == 592 == 2 * R.N_PAIRS
    assert R.N_SEGMENTS == 4
    assert R.PAIRS_PER_SEGMENT == 74
    assert R.GAMES_PER_SEGMENT == 148
    assert R.PAIRS_PER_STRATUM == 148
    assert R.STRATUM_PAIRS_PER_SEGMENT == 37
    assert R.OPENING_PLIES == 6
    assert R.BOARD_SIZE == 24


def test_EVERY_BALANCE_HOLDS_WITHOUT_REMAINDER():
    """🔴 296 was chosen over 289 precisely so nothing has a remainder. If any of
    these divisions left one, the composition would drift between segments."""
    assert R.PAIRS_PER_STRATUM * 2 == R.N_PAIRS
    assert R.PAIRS_PER_SEGMENT * R.N_SEGMENTS == R.N_PAIRS
    assert R.STRATUM_PAIRS_PER_SEGMENT * 2 == R.PAIRS_PER_SEGMENT
    assert R.STRATUM_PAIRS_PER_SEGMENT * R.N_SEGMENTS == R.PAIRS_PER_STRATUM
    assert R.GAMES_PER_SEGMENT * R.N_SEGMENTS == R.N_GAMES


def test_THE_PRECISION_TARGET_IS_ACTUALLY_MET():
    """🔴 THE DEFECT AMENDMENT 1 FIXED, BOUND AS A TEST. The first draft rounded
    288.194 DOWN to 288 and reported h = 0.08003 -- above the target the
    arithmetic existed to enforce. A gate that does not bind."""
    h = math.sqrt(math.log(2 / R.ALPHA) / (2 * R.N_PAIRS))
    assert R.half_width(R.N_PAIRS) == pytest.approx(h)
    assert h <= R.PRECISION_TARGET, f"h={h} exceeds the declared target"
    assert R.half_width(288) > R.PRECISION_TARGET, (
        "288 must NOT satisfy the target -- if it did, this test would not be "
        "binding the correction it exists for")
    assert math.ceil(math.log(2 / R.ALPHA) / (2 * R.PRECISION_TARGET ** 2)) == 289


def test_the_half_width_shrinks_with_n_and_refuses_nonsense():
    assert R.half_width(592) < R.half_width(296) < R.half_width(148)
    for bad in (0, -1):
        with pytest.raises(R.H3StudyError, match="positive"):
            R.half_width(bad)


# ───────────────────────── the strata ──────────────────────────────────────

def test_the_two_strata_are_named_and_fixed_5050():
    assert R.STRATA == (R.STRATUM_UNIFORM, R.STRATUM_CO_PRODUCED)
    assert R.STRATUM_UNIFORM == "uniform"
    assert R.STRATUM_CO_PRODUCED == "co_produced"
    assert all(R.PAIRS_PER_STRATUM == 148 for _ in R.STRATA)


def test_the_alternating_order_allocation_is_DECLARED_not_derived():
    """🔑 37 per segment is ODD, so the order split cannot be even and must not be
    left to emerge at run time (card §1.7.6)."""
    inc = R.INCUMBENT_FIRST_PER_SEGMENT
    t1j = R.T1J_FIRST_PER_SEGMENT
    assert inc == (19, 18, 19, 18)
    assert t1j == (18, 19, 18, 19)
    assert sum(inc) == sum(t1j) == 74
    assert sum(inc) + sum(t1j) == R.PAIRS_PER_STRATUM
    for k in range(R.N_SEGMENTS):
        assert inc[k] + t1j[k] == R.STRATUM_PAIRS_PER_SEGMENT


def test_the_generation_seeds_are_DECLARED_and_in_NO_registry():
    """Generation must never consume a drawable seed -- the pilot's rule. The
    WHOLE attempt range is checked, not just the base, because attempt j uses
    GEN_SEED + i*MAX_ATTEMPTS + j."""
    from scripts.GPU.alphazero import e4_screen_reference as REF
    assert R.MAX_ATTEMPTS >= 1
    for base in (R.GEN_SEED_UNIFORM, R.GEN_SEED_CO_PRODUCED):
        lo, hi = R.generation_seed_range(base)
        assert hi - lo == R.PAIRS_PER_STRATUM * R.MAX_ATTEMPTS
        for s in (lo, lo + 1, (lo + hi) // 2, hi - 1):
            st = REF.seed_status(s)
            assert not any(st.values()), (s, st)
            assert s not in REF.CONSUMED_SEEDS


def test_the_two_generation_RANGES_ARE_SEPARATED_BY_MORE_THAN_THEIR_OWN_SIZE():
    """🔴 DISJOINT IS NOT SEPARATED. The second attempt at these constants put the
    ranges 40,800 apart -- no overlap, but inside the gap floor this programme
    sets at the candidate's OWN SIZE, and collision proof v11 rejected them. The
    floor exists so that extending either range later cannot silently collide."""
    a = R.generation_seed_range(R.GEN_SEED_UNIFORM)
    b = R.generation_seed_range(R.GEN_SEED_CO_PRODUCED)
    assert a[1] <= b[0] or b[1] <= a[0], (a, b)
    gap = b[0] - a[1] if a[1] <= b[0] else a[0] - b[1]
    floor = max(a[1] - a[0], b[1] - b[0])
    assert gap >= floor, f"gap {gap} is inside the floor {floor}"


def test_the_attempt_seed_rule_never_reuses_a_REJECTED_seed():
    """🔑 A rejection must advance to a FRESH seed; re-drawing the seed that
    produced the rejected position would loop forever."""
    seen = set()
    for i in range(R.PAIRS_PER_STRATUM):
        for j in range(R.MAX_ATTEMPTS):
            s = R.attempt_seed(R.GEN_SEED_UNIFORM, i, j)
            assert s not in seen, (i, j, s)
            seen.add(s)
    assert len(seen) == R.PAIRS_PER_STRATUM * R.MAX_ATTEMPTS
    with pytest.raises(R.H3StudyError, match="attempt"):
        R.attempt_seed(R.GEN_SEED_UNIFORM, 0, R.MAX_ATTEMPTS)


# ───────────────────────── the generating configuration ────────────────────

def test_THE_GENERATOR_S_INCUMBENT_IS_NOT_THE_MATCH_INCUMBENT():
    """🔴 THE ENTROPY FINDING, BOUND AS A TEST (card §1.7.2).

    Under argmax the incumbent's move is a deterministic function of the
    position, and E3a proved T1j deterministic at fixed depth. An alternating
    generator built from those two players has NO ENTROPY: stratum B would be two
    positions repeated 74 times each.
    """
    from scripts.GPU.alphazero import h2_match_rules as H2R
    gen = R.generation_config()
    assert gen.selection_mode == "opening_temperature"
    assert gen.selection_mode != H2R.SELECTION_MODE == "argmax", (
        "if the generator played argmax it would produce ONE opening per order")
    # all six generated plies must fall inside the sampling window
    assert gen.opening_temp_plies >= R.OPENING_PLIES
    assert gen.temp_high > 0, "temperature zero is argmax by another name"


def test_generation_config_REFUSES_an_argmax_frozen_configuration(monkeypatch):
    """🔴 THE GUARD IS FOR A FUTURE CHANGE, so only a driven test can see it.

    Today's frozen configuration is `opening_temperature`, so the refusal never
    fires and asserting the mode alone cannot tell whether the guard exists. If
    the frozen research config ever became argmax, the generator would silently
    produce ONE opening per order -- the finding this whole design turns on.
    """
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    real = G3.eval_config()
    argmaxed = real.__class__(**{**real.__dict__, "selection_mode": "argmax"})
    monkeypatch.setattr(G3, "eval_config", lambda: argmaxed)
    with pytest.raises(R.H3StudyError, match="no entropy|two positions"):
        R.generation_config()


def test_generation_config_REFUSES_a_sampling_window_that_is_too_short(monkeypatch):
    """The other half of the same guard: if the window stopped covering all six
    generated plies, the later ones would be deterministic."""
    from scripts.GPU.alphazero import twixtbot_g3_reference as G3
    real = G3.eval_config()
    short = real.__class__(**{**real.__dict__, "opening_temp_plies": 2})
    monkeypatch.setattr(G3, "eval_config", lambda: short)
    with pytest.raises(R.H3StudyError, match="deterministic"):
        R.generation_config()


def test_the_generator_and_the_match_share_EVERYTHING_ELSE():
    """Only the readout differs. A generator that also changed the search would
    be producing positions from a different player altogether."""
    from scripts.GPU.alphazero import h2_match_rules as H2R
    from scripts.GPU.alphazero import h3_study_runner as RUN
    gen, match = R.generation_config(), RUN.frozen_argmax_config()
    assert gen.mcts_sims == match.mcts_sims == H2R.MCTS_SIMS
    assert gen.board_size == match.board_size == R.BOARD_SIZE
    differing = [f for f in vars(gen) if getattr(gen, f) != getattr(match, f)]
    assert differing == ["selection_mode"], differing


# ───────────────────────── stratum A, generated here ───────────────────────

@pytest.fixture(scope="module")
def uniform():
    return R.generate_uniform_openings()


def test_stratum_A_generates_the_right_number_at_the_right_depth(uniform):
    assert len(uniform) == R.PAIRS_PER_STRATUM
    for op in uniform:
        assert len(op["moves"]) == R.OPENING_PLIES
        assert op["stratum"] == R.STRATUM_UNIFORM
        assert op["order"] is None, "the uniform stratum has no alternating order"
        assert 1 <= op["attempts"] <= R.MAX_ATTEMPTS


def test_stratum_A_is_REPRODUCIBLE(uniform):
    again = R.generate_uniform_openings()
    assert [o["digest"] for o in again] == [o["digest"] for o in uniform]
    assert [o["attempts"] for o in again] == [o["attempts"] for o in uniform]


def test_every_opening_is_DISTINCT_and_excludes_EVERY_PRIOR_EXPERIMENT(uniform):
    digests = [o["digest"] for o in uniform]
    assert len(set(digests)) == len(digests)
    assert not set(digests) & R.excluded_digests()


def test_the_exclusion_set_actually_contains_the_pilot_AND_H1H2():
    """A vacuous exclusion set would make the test above prove nothing."""
    excluded = R.excluded_digests()
    assert len(excluded) == 20 + 8, len(excluded)


def test_no_generated_position_is_TERMINAL(uniform):
    for op in uniform:
        assert not op["state"].is_terminal(), op["index"]


def test_the_attempt_seeds_used_are_RECORDED(uniform):
    """A high rejection rate changes the population's conditioning, so it is
    evidence, not a private detail of the loop (card §1.7.5)."""
    for op in uniform:
        assert op["seed"] == R.attempt_seed(
            R.GEN_SEED_UNIFORM, op["index"], op["attempts"] - 1)


def test_exhausting_MAX_ATTEMPTS_ABORTS_rather_than_yielding_fewer(monkeypatch):
    """It never returns a short set and never relaxes a filter."""
    monkeypatch.setattr(R, "MAX_ATTEMPTS", 0)
    with pytest.raises(R.H3StudyError, match="attempts"):
        R.generate_uniform_openings(n=1)


# ───────────────────────── the segmented schedule ──────────────────────────

@pytest.fixture(scope="module")
def openings():
    """Both strata. Stratum B is NOT generated here — generating it is a run —
    so a declared STUB set stands in, carrying the same shape and provenance
    fields the real generator will emit."""
    return R.stub_opening_set()


def test_the_stub_set_is_MARKED_as_a_stub_and_refuses_to_be_pinned(openings):
    """🔴 A stub must never be mistaken for the artifact. It cannot satisfy the
    real pin, and it says so in every opening."""
    assert len(openings) == R.N_PAIRS
    b = [o for o in openings if o["stratum"] == R.STRATUM_CO_PRODUCED]
    assert all(o["stub"] is True for o in b)
    assert all(o.get("stub", False) is False for o in openings
               if o["stratum"] == R.STRATUM_UNIFORM)
    with pytest.raises(R.H3StudyError, match="stub"):
        R.check_opening_set(openings)


def test_the_set_is_148_of_each_stratum_and_74_of_each_ORDER(openings):
    from collections import Counter
    assert Counter(o["stratum"] for o in openings) == {
        R.STRATUM_UNIFORM: 148, R.STRATUM_CO_PRODUCED: 148}
    orders = Counter(o["order"] for o in openings
                     if o["stratum"] == R.STRATUM_CO_PRODUCED)
    assert orders == {R.ORDER_INCUMBENT_FIRST: 74, R.ORDER_T1J_FIRST: 74}


def test_EVERY_SEGMENT_CARRIES_THE_DECLARED_COMPOSITION(openings):
    """🔑 37 + 37 in every segment, and the declared 19/18 order split — so a
    VOIDed segment costs a balanced slice, not a skewed one."""
    from collections import Counter
    for k in range(R.N_SEGMENTS):
        seg = [o for o in openings if R.segment_of(o["index"]) == k]
        assert len(seg) == R.PAIRS_PER_SEGMENT
        assert Counter(o["stratum"] for o in seg) == {
            R.STRATUM_UNIFORM: 37, R.STRATUM_CO_PRODUCED: 37}
        orders = Counter(o["order"] for o in seg
                         if o["stratum"] == R.STRATUM_CO_PRODUCED)
        assert orders[R.ORDER_INCUMBENT_FIRST] == R.INCUMBENT_FIRST_PER_SEGMENT[k]
        assert orders[R.ORDER_T1J_FIRST] == R.T1J_FIRST_PER_SEGMENT[k]


def test_segment_of_partitions_every_index_exactly_once():
    got = [R.segment_of(i) for i in range(R.N_PAIRS)]
    assert sorted(set(got)) == list(range(R.N_SEGMENTS))
    assert all(got.count(k) == R.PAIRS_PER_SEGMENT for k in range(R.N_SEGMENTS))
    for bad in (-1, R.N_PAIRS):
        with pytest.raises(R.H3StudyError, match="index"):
            R.segment_of(bad)


# ───────────────────────── tasks ───────────────────────────────────────────

def test_NO_SEED_IS_ASSIGNED_because_no_block_is_reserved(openings):
    tasks = R.build_tasks(openings)
    assert len(tasks) == R.N_GAMES
    assert all(t["seed"] is None for t in tasks)


def test_each_pair_plays_the_SAME_opening_BOTH_WAYS(openings):
    by_pair = {}
    for t in R.build_tasks(openings):
        by_pair.setdefault(t["pair_id"], []).append(t)
    assert len(by_pair) == R.N_PAIRS
    for pid, two in by_pair.items():
        assert {t["incumbent_colour"] for t in two} == {"red", "black"}, pid
        assert len({t["opening_digest"] for t in two}) == 1, pid
        assert len({t["stratum"] for t in two}) == 1, pid


def test_every_task_CARRIES_ITS_STRATUM_AND_OPENING_DIGEST(openings):
    """🔴 Card §6.2: the record must name the position it was played from, so
    the pair identity is computable FROM A SINGLE RECORD."""
    for t in R.build_tasks(openings):
        assert t["stratum"] in R.STRATA
        assert len(t["opening_digest"]) == 64
        assert t["segment"] == R.segment_of(t["pair_id"])


def test_tasks_carry_the_REFERENCE_IDENTITY_the_builder_reads(openings):
    """The pilot's hardest-won lesson: without these the real builder refuses
    every task and `make_agent_factory` raises on the first move."""
    from scripts.GPU.alphazero import e4_screen_reference as REF
    for t in R.build_tasks(openings):
        for f in ("reference", "reference_sha1", "reference_colour"):
            assert f in t, f
        assert t["reference_colour"] == REF.reference_colour(t)
        assert t["reference_colour"] == t["incumbent_colour"]


def test_a_SUPPLIED_interval_is_assigned_POSITIONALLY(openings):
    lo = 777000000
    seeded = R.build_tasks(openings, seed_interval=(lo, lo + R.N_GAMES))
    assert [t["seed"] for t in seeded] == list(range(lo, lo + R.N_GAMES))


def test_a_SEGMENT_takes_a_CONTIGUOUS_quarter_of_the_block(openings):
    lo = 777000000
    seeded = R.build_tasks(openings, seed_interval=(lo, lo + R.N_GAMES))
    for k in range(R.N_SEGMENTS):
        seg = [t for t in seeded if t["segment"] == k]
        assert len(seg) == R.GAMES_PER_SEGMENT
        seeds = sorted(t["seed"] for t in seg)
        assert seeds == list(range(lo + k * R.GAMES_PER_SEGMENT,
                                   lo + (k + 1) * R.GAMES_PER_SEGMENT))


@pytest.mark.parametrize("iv,what", [
    (("777000000", 777000592), "a string endpoint"),
    ((777000000.0, 777000592), "a float endpoint"),
    ((True, 592), "a bool endpoint"),
])
def test_a_TYPE_DIFFERENT_seed_endpoint_is_REFUSED(openings, iv, what):
    with pytest.raises(R.H3StudyError, match="int"):
        R.build_tasks(openings, seed_interval=iv)


def test_an_interval_of_the_WRONG_SIZE_is_refused(openings):
    with pytest.raises(R.H3StudyError, match="592"):
        R.build_tasks(openings, seed_interval=(777000000, 777000591))


def test_a_SPENT_interval_is_REFUSED_by_STATUS(openings):
    """Exactly 592 seeds so it clears the length check and reaches the guard
    alone — the pilot's trap, where a too-long block was refused by size first."""
    with pytest.raises(R.H3StudyError, match="spent|EXPOSED|RETIRED"):
        R.build_tasks(openings, seed_interval=(202624000, 202624000 + R.N_GAMES))


def test_the_schedule_is_PINNED_BY_A_FULL_FIELD_DIGEST(openings):
    tasks = R.build_tasks(openings)
    d = R.task_digest(tasks)
    assert len(d) == 64
    tampered = [dict(t) for t in tasks]
    tampered[0]["ply_cap"] = tampered[0]["ply_cap"] + 1
    assert R.task_digest(tampered) != d, "a full-field digest must see any field"
