"""H3 FULL STUDY — the frozen constants, the strata, the segments, the schedule.

Binds `docs/superpowers/2026-09-15-t1j-h3-full-study-design.md` as amended twice.
Nothing here plays, loads, draws a match seed or generates stratum B.
"""
import math

import pytest

from scripts.GPU.alphazero import h3_study_rules as R


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

    🔑 PATCHED ON THE PROTOCOL, not on the rules module's re-export. The loop
    moved into `h3_generation_protocol` (so the artifact can pin it without
    pinning `OPENING_SET_DIGEST`), and `R.MAX_ATTEMPTS` is now a copy taken at
    import. Patching the copy changed nothing and this test silently stopped
    exercising the abort.
    """
    from scripts.GPU.alphazero import h3_generation_protocol as PROTO
    monkeypatch.setattr(PROTO, "MAX_ATTEMPTS", 0)
    with pytest.raises(R.H3StudyError, match="attempts"):
        R.generate_uniform_openings(n=1)


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


@pytest.fixture(scope="module")
def openings():
    """THE population: 296 uniform openings in study order. One call, reused."""
    return R.assemble_opening_set(R.generate_uniform_openings())
