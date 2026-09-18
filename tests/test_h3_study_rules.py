"""H3 FULL STUDY — the frozen constants, the strata, the segments, the schedule.

Binds `docs/superpowers/2026-09-15-t1j-h3-full-study-design.md` as amended twice.
Nothing here plays, loads, draws a match seed or generates stratum B.
"""
import math

import pytest

from scripts.GPU.alphazero import h3_study_rules as R


#: 🔴 FOUR BLOCKS, ONE PER SEGMENT. The study no longer uses a single interval,
#: so a test schedule must not either.
FRESH_BLOCKS = tuple(
    (777000000 + R.GAMES_PER_SEGMENT * k,
     777000000 + R.GAMES_PER_SEGMENT * (k + 1))
    for k in range(R.N_SEGMENTS))


# ───────────────────────── the card's numbers ──────────────────────────────





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













# ───────────────────────── the generating configuration ────────────────────









# ───────────────────────── stratum A, generated here ───────────────────────

@pytest.fixture(scope="module")
def uniform():
    return R.generate_uniform_openings()




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
    """It never returns a short set and never relaxes a filter.

    🔴 THIS TEST NEVER REACHED THE PATH IT NAMES. It set `MAX_ATTEMPTS = 0` and
    matched "attempts" in the refusal -- but with the ceiling at zero the SEED
    ALLOCATION refuses first ("attempt 0 is beyond MAX_ATTEMPTS=0"), the loop is
    never entered, and the exhaustion abort never runs. The message happened to
    contain the same word. The injected-defect harness found it: replacing the
    abort with a `break` that returns a short set changed nothing this test saw.

    It now leaves the ceiling usable and makes every candidate INADMISSIBLE, so
    the loop really does exhaust its attempts.
    """
    from scripts.GPU.alphazero import h3_generation_protocol as PROTO
    monkeypatch.setattr(PROTO, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(PROTO, "admissible", lambda st, moves: False)
    with pytest.raises(R.H3StudyError, match="exhausted MAX_ATTEMPTS"):
        R.generate_uniform_openings(n=1)


def test_the_ABORT_MESSAGE_NAMES_THE_OPENING_AND_THE_SHORTFALL(monkeypatch):
    """A refusal that does not say WHICH opening and HOW FAR it got is a refusal
    nobody can act on."""
    from scripts.GPU.alphazero import h3_generation_protocol as PROTO
    monkeypatch.setattr(PROTO, "MAX_ATTEMPTS", 2)
    monkeypatch.setattr(PROTO, "admissible", lambda st, moves: False)
    with pytest.raises(R.H3StudyError) as ei:
        R.generate_uniform_openings(n=4)
    msg = str(ei.value)
    assert "opening 0" in msg and "0 of 4" in msg


def test_THE_RULES_RE_EXPORTS_ARE_THE_PROTOCOL_S_OWN_VALUES():
    """🔴 A RE-EXPORT THAT DRIFTS IS WORSE THAN A SECOND COPY, because it looks
    authoritative. The artifact pins the protocol; if this module's copies said
    something else, the schedule and the pinned walk would disagree."""
    from scripts.GPU.alphazero import h3_generation_protocol as PROTO
    for name in ("BOARD_SIZE", "OPENING_PLIES", "N_PAIRS", "MAX_ATTEMPTS",
                 "GEN_SEED_UNIFORM"):
        assert getattr(R, name) == getattr(PROTO, name), name


# ───────────────────────── the segmented schedule ──────────────────────────









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


















def test_the_schedule_is_PINNED_BY_A_FULL_FIELD_DIGEST(openings):
    tasks = R.build_tasks(openings)
    d = R.task_digest(tasks)
    assert len(d) == 64
    tampered = [dict(t) for t in tasks]
    tampered[0]["ply_cap"] = tampered[0]["ply_cap"] + 1
    assert R.task_digest(tampered) != d, "a full-field digest must see any field"


@pytest.fixture(scope="module")
def openings():
    """THE population: 296 uniform openings in study order. One call, reused."""
    return R.assemble_opening_set(R.generate_uniform_openings())


# ═══════ FOUR SEGMENT BLOCKS, AND STATUS IS A LAUNCH QUESTION ══════════════
# 🔴 `build_tasks` took ONE 592-seed interval and refused it if ANY seed in it
# was spent. Segment 0's quarter was retired on its VOID and the whole study
# became unbuildable -- segments 1-3 too, though their seeds were untouched.
# Segmenting exists so one segment's failure costs ONE segment.

def test_FOUR_BLOCKS_ARE_ASSIGNED_POSITIONALLY_WITHIN_EACH_SEGMENT():
    ops = R.assemble_opening_set(R.generate_uniform_openings())
    t = R.build_tasks(ops, seed_blocks=FRESH_BLOCKS)
    assert len(t) == R.N_GAMES
    for k, (lo, hi) in enumerate(FRESH_BLOCKS):
        seg = [x["seed"] for x in t if x["segment"] == k]
        assert seg == list(range(lo, hi)), k
        assert len(seg) == R.GAMES_PER_SEGMENT == 148


def test_build_tasks_DOES_NOT_CONSULT_SEED_STATUS():
    """🔴 THE DECOUPLING. A frozen schedule must exist whether or not a segment
    has since been spent -- otherwise retiring one segment deletes the plan for
    the other three. Status is checked at LAUNCH, per segment."""
    ops = R.assemble_opening_set(R.generate_uniform_openings())
    retired = (202_626_000, 202_626_148)          # segment 0's VOIDed quarter
    from scripts.GPU.alphazero import e4_screen_reference as REF
    assert all(REF.seed_status(s)["retired"] for s in range(*retired))
    t = R.build_tasks(ops, seed_blocks=(retired,) + FRESH_BLOCKS[1:])
    assert len(t) == R.N_GAMES, "construction must not care that a block is spent"


@pytest.mark.parametrize("blocks,match", [
    (FRESH_BLOCKS[:3], "3 seed blocks for 4 segments"),
    (FRESH_BLOCKS + FRESH_BLOCKS[:1], "5 seed blocks for 4 segments"),
    (((777000000, 777000010),) + FRESH_BLOCKS[1:], "holds 10 seeds"),
    ((("x", 777000148),) + FRESH_BLOCKS[1:], "an int is required"),
    (((777000000.0, 777000148),) + FRESH_BLOCKS[1:], "an int is required"),
    (((True, 777000148),) + FRESH_BLOCKS[1:], "an int is required"),
    ((FRESH_BLOCKS[0], FRESH_BLOCKS[0]) + FRESH_BLOCKS[2:], "overlaps an earlier"),
])
def test_THE_SEED_BLOCKS_ARE_REFUSED_WHEN_MALFORMED(blocks, match):
    ops = R.assemble_opening_set(R.generate_uniform_openings())
    with pytest.raises(R.H3StudyError, match=match):
        R.build_tasks(ops, seed_blocks=blocks)


def test_A_GOOD_SET_OF_BLOCKS_IS_ACCEPTED_OR_THE_REFUSALS_PROVE_NOTHING():
    ops = R.assemble_opening_set(R.generate_uniform_openings())
    assert len(R.build_tasks(ops, seed_blocks=FRESH_BLOCKS)) == R.N_GAMES
