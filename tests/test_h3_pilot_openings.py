"""H3 PILOT — the opening set: generated deterministically, pinned by a digest.

The card (`docs/superpowers/2026-09-13-t1j-h3-pilot-card.md` §2) freezes the
procedure. These tests bind it, because a generation rule that drifts silently
would let the pilot answer a different question than the one it was scoped for.

Nothing here plays a game, loads a model, starts a JVM or draws a match seed. The
generator's PRNG constant is NOT a match seed and is not drawn from any block.
"""
import hashlib

import pytest

from scripts.GPU.alphazero import h3_pilot_rules as R
from scripts.GPU.alphazero import d1_selection as SEL


# ───────────────────────── the frozen constants ─────────────────────────

def test_the_card_numbers_are_the_module_numbers():
    assert R.N_OPENINGS == 20
    assert R.N_GAMES == 40, "20 openings x 2 colour assignments"
    assert R.OPENING_PLIES == 6, "the opening_bound every H1/H2 game used"
    assert R.BOARD_SIZE == 24
    assert R.OPENING_SEED == 20260913
    assert R.RUN_DEADLINE_S == 7200
    assert R.PER_CALL_TIMEOUT_S == 120


def test_the_generation_seed_is_NOT_a_match_seed():
    """🔑 Generation is not play. The card says so and the number must not sit in
    any seed block -- registered, reserved on paper, or spent."""
    from scripts.GPU.alphazero import e4_screen_reference as REF
    st = REF.seed_status(R.OPENING_SEED)
    assert not any(st.values()), (R.OPENING_SEED, st)


def test_the_stop_rule_thresholds_are_the_cards():
    assert R.MAX_DUPLICATE_PAIRS == 2            # S1: > 2 closes
    assert R.MAX_CAPPED_GAMES == 8               # S2: > 8 closes
    assert R.MAX_WITHIN_PAIR_IDENTICAL == 2      # S3: > 2 closes
    assert R.MAX_TOTAL_ELAPSED_S == 3600         # S4a
    assert R.MAX_P90_OVER_MEDIAN == 4.0          # S4b
    assert R.REPORT_FLOOR_PAIRS == 10


# ──────────────────────── the generated positions ───────────────────────

@pytest.fixture(scope="module")
def openings():
    return R.generate_openings()


def test_it_produces_exactly_the_frozen_number(openings):
    assert len(openings) == R.N_OPENINGS == 20


def test_every_position_is_at_the_frozen_depth(openings):
    for op in openings:
        assert len(op["moves"]) == R.OPENING_PLIES == 6, op["index"]
        assert len(op["state"].pegs) == 6, op["index"]


def test_every_move_was_LEGAL_when_it_was_played(openings):
    """Replayed from an empty board through the ENGINE's own rules, not asserted
    of the finished position: a position can look legal and have been reached
    illegally."""
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    for op in openings:
        st = TwixtState()
        for m in op["moves"]:
            assert tuple(m) in st.legal_moves(), (op["index"], m)
            st = st.apply_move(tuple(m))
        assert SEL.canonical_digest(st) == op["digest"]


def test_no_position_is_already_won(openings):
    for op in openings:
        assert not op["state"].is_terminal(), op["index"]
        assert op["state"].winner() is None, op["index"]


def test_every_position_is_DISTINCT_by_canonical_digest(openings):
    digests = [op["digest"] for op in openings]
    assert len(set(digests)) == len(digests) == 20


def test_none_duplicates_an_H1_or_H2_OPENING(openings):
    """The pilot must be independent of the eight the earlier matches used."""
    prior = R.prior_opening_digests()
    assert len(prior) == 8, "vacuous: the prior openings did not resolve"
    assert not (set(op["digest"] for op in openings) & prior)


def test_GENERATION_IS_DETERMINISTIC(openings):
    """Same constant, same set -- or the digest pins nothing."""
    again = R.generate_openings()
    assert [o["digest"] for o in again] == [o["digest"] for o in openings], \
        "GENERATION IS NOT REPRODUCIBLE: the same constant gave a different set"
    assert [o["moves"] for o in again] == [o["moves"] for o in openings], \
        "GENERATION IS NOT REPRODUCIBLE: the same constant gave different moves"


def test_A_DIFFERENT_SEED_GIVES_A_DIFFERENT_SET(openings):
    """The negative half: if the constant did not drive it, determinism above
    would be vacuous."""
    other = R.generate_openings(seed=R.OPENING_SEED + 1)
    assert [o["digest"] for o in other] != [o["digest"] for o in openings]


def test_THE_SET_IS_PINNED_BY_A_DIGEST(openings):
    """Recomputed from the openings, so a pin that drifts from the set fails."""
    want = hashlib.sha256(
        "\n".join(o["digest"] for o in openings).encode()).hexdigest()
    assert R.opening_set_digest(openings) == want
    assert R.opening_set_digest(openings) == R.OPENING_SET_DIGEST, (
        "the frozen pin does not match the generated set")


def test_the_pin_CHANGES_when_the_set_changes(openings):
    """A pin nothing can contradict is decoration."""
    tampered = list(openings)
    tampered[0] = dict(tampered[0], digest="0" * 64)
    assert R.opening_set_digest(tampered) != R.OPENING_SET_DIGEST


# ═══════════════════ the schedule: 20 pairs, 40 tasks, digest-pinned ═══════

@pytest.fixture(scope="module")
def tasks(openings):
    return R.build_tasks(openings)


def test_it_builds_exactly_40_tasks_in_20_PAIRS(tasks):
    assert len(tasks) == R.N_GAMES == 40
    pairs = {t["pair_id"] for t in tasks}
    assert len(pairs) == R.N_OPENINGS == 20
    for pid in pairs:
        assert sum(1 for t in tasks if t["pair_id"] == pid) == 2


def test_each_pair_plays_the_SAME_opening_BOTH_WAYS(tasks):
    """The pairing is the whole design: same position, colours reversed."""
    by_pair = {}
    for t in tasks:
        by_pair.setdefault(t["pair_id"], []).append(t)
    for pid, two in by_pair.items():
        assert {t["incumbent_colour"] for t in two} == {"red", "black"}, pid
        assert len({t["opening_digest"] for t in two}) == 1, pid
        assert len({tuple(map(tuple, t["opening_moves"])) for t in two}) == 1, pid


def test_task_ids_are_UNIQUE_and_carry_the_pair(tasks):
    assert len({t["task_id"] for t in tasks}) == 40
    for t in tasks:
        assert f"p{t['pair_id']:02d}" in t["task_id"], t["task_id"]


def test_NO_SEED_IS_ASSIGNED_because_no_block_is_reserved(tasks):
    """The card reserves nothing. A plan that hands out seeds before a block is
    authorized has spent it on paper."""
    assert all(t["seed"] is None for t in tasks)


def test_a_SUPPLIED_interval_is_assigned_POSITIONALLY(openings):
    """Row i carries lo + i, so a count of seeds drawn identifies WHICH."""
    lo = 777000000
    seeded = R.build_tasks(openings, seed_interval=(lo, lo + 40))
    assert [t["seed"] for t in seeded] == list(range(lo, lo + 40))


def test_an_interval_of_the_WRONG_SIZE_is_refused(openings):
    with pytest.raises(R.H3PilotError, match="40"):
        R.build_tasks(openings, seed_interval=(777000000, 777000039))


@pytest.mark.parametrize("lo,what", [
    (202622000, "H2 attempt 3: exposed AND retired"),
    (202622700, "H2 attempt 3: retired but never drawn"),
    (202618000, "H2 attempt 1: retired, exposed 0"),
])
def test_a_RETIRED_or_EXPOSED_interval_is_REFUSED(openings, lo, what):
    """A spent block may not be revived.

    🔑 EXACTLY 40 SEEDS, so it CLEARS THE LENGTH CHECK and reaches the spent-seed
    guard alone. My first version passed H2's whole 736-seed block, which was
    refused by size first and proved only that -- the same trap
    `test_selection_still_REFUSES_the_retired_block` records for D1.
    """
    with pytest.raises(R.H3PilotError, match="RETIRED|EXPOSED|spent"):
        R.build_tasks(openings, seed_interval=(lo, lo + 40))


def test_a_FRESH_interval_of_the_right_size_is_ACCEPTED(openings):
    """The other half: if nothing were ever accepted the guard above would be an
    outage rather than a check."""
    seeded = R.build_tasks(openings, seed_interval=(777000000, 777000040))
    assert [t["seed"] for t in seeded] == list(range(777000000, 777000040))


def test_THE_SCHEDULE_IS_PINNED_BY_A_DIGEST(tasks):
    assert R.task_digest(tasks) == R.TASK_DIGEST
    assert len(R.TASK_DIGEST) == 64


def test_the_schedule_pin_CHANGES_when_any_field_changes(tasks):
    tampered = [dict(t) for t in tasks]
    tampered[0]["incumbent_colour"] = "black" if tampered[0]["incumbent_colour"] == "red" else "red"
    assert R.task_digest(tampered) != R.TASK_DIGEST


# ══════════ the reference identity the REAL harness reads off a task ════════
# 🔴 THE GAP THE PRE-RUN VERIFICATION FOUND. H3's tasks carried none of the
# fields the qualified construction path reads, so `validate_schedule_executable`
# refused all 40 -- and `make_agent_factory` subscripts `reference_colour`
# DIRECTLY, so the first move of the first game would have raised KeyError. Every
# seam test supplied its own agent factory, which is precisely why none of them
# could see it.

def test_THE_REGISTERED_SCHEDULE_IS_EXECUTABLE_through_the_REAL_registry(openings):
    """Not `validate_schedule_structure`: the executable question, against the
    registered block, which is the question the runner asks before it plays."""
    from scripts.GPU.alphazero import e4_screen_reference as REF
    from scripts.GPU.alphazero import h3_pilot_runner as RUN
    tasks = R.build_tasks(openings, seed_interval=RUN.PILOT_SEED_BLOCK)
    out = REF.validate_schedule_executable(tasks)
    assert out["n_tasks"] == 40
    assert out["distinct_seeds"] == 40 and out["distinct_stream_pairs"] == 40


def test_the_REFERENCE_IDENTITY_AGREES_WITH_BOTH_PINS(tasks):
    """Two independently pinned sources saying the same thing.

    🔑 The sha1 is read from the sha256-VERIFIED source plan, never from
    `REFERENCE_CHECKPOINTS` -- because `validate_task_structure` compares the task
    against that registry, and sourcing it there would leave the check comparing a
    value with itself.
    """
    from scripts.GPU.alphazero import e4_screen_reference as REF
    from scripts.GPU.alphazero import h2_match_plan as H2PLAN
    ref = H2PLAN.load_source_plan()["reference"]
    for t in tasks:
        assert t["reference"] == ref["name"]
        assert t["reference_sha1"] == ref["sha1"]
        assert t["reference_sha1"] == REF.REFERENCE_CHECKPOINTS[t["reference"]]["sha1"]


def test_every_task_NAMES_the_colour_OUR_SIDE_plays(tasks):
    """`make_agent_factory` routes on this field and `_enforce_evaluator` refuses
    a task without it; both read it, neither derives it."""
    from scripts.GPU.alphazero import e4_screen_reference as REF
    for t in tasks:
        # 🔑 WITH A MESSAGE, so the failure names the defect rather than printing
        # `assert 'black' == 'red'` -- a reason that could be almost anything.
        assert t["reference_colour"] == t["incumbent_colour"], \
            f"{t['task_id']}: reference_colour is not OUR side"
        assert t["reference_colour"] == REF.reference_colour(t)
        assert t["reference_colour"] != t["anchor_colour"]


def test_the_task_carries_NEITHER_reference_sha256_NOR_rng_streams(tasks):
    """H2's tasks carry both; H3's deliberately do not, and this records why.

    Nothing reads either one: `build_reference_agent` checks the sha1, and
    `_injective_streams` DERIVES the streams from the seed rather than trusting a
    recorded copy. An unread field under a full-field digest is a value that can
    disagree with the truth without anything noticing. The seed->stream binding
    for this block is proved in the registration evidence instead.
    """
    for t in tasks:
        assert "reference_sha256" not in t
        assert "rng_streams" not in t


def test_the_incumbent_configuration_is_H2s_FROZEN_ONE(tasks):
    """Unchanged, deliberately: H3 changes the evidence structure, not the player."""
    from scripts.GPU.alphazero import h2_match_rules as H2R
    for t in tasks:
        assert t["selection_mode"] == H2R.SELECTION_MODE == "argmax"
        assert t["mcts_sims"] == H2R.MCTS_SIMS == 400
        assert t["t1j_mdPly"] == H2R.T1J_MDPLY == 6


@pytest.mark.parametrize("iv,what", [
    (("777000000", 777000040), "a string endpoint"),
    ((777000000.0, 777000040), "a float endpoint"),
    ((777000000, 777000040.0), "a float endpoint"),
    ((True, 41), "a bool endpoint"),
])
def test_a_TYPE_DIFFERENT_seed_endpoint_is_REFUSED(openings, iv, what):
    """🔴 `int()` coercion accepted "777000000" and 777000000.0 silently. The
    programme is type-strict about seeds everywhere else -- a seed that arrives as
    a float has been through arithmetic that an int would not survive."""
    with pytest.raises(R.H3PilotError, match="int"):
        R.build_tasks(openings, seed_interval=iv)


# ══════════ the rejection branches, REACHED ALONE ═══════════════════════════
# 🔴 Three injected-defect controls went NOT CAUGHT against the generated set:
# disabling deduplication, disabling the prior-opening exclusion and admitting
# terminal positions all changed NOTHING observable, because at 6 plies on a 24x24
# board a collision never occurs and a game is never over. The checks are real and
# the SAMPLE cannot exercise them. Each is driven directly here, the way
# `test_a_cohort_that_departs_from_the_frozen_table_is_refused` drives D1's.

def test_the_DEDUPLICATION_branch_is_reached_alone(monkeypatch):
    """Every candidate made to collide: the generator accepts the FIRST and can
    never accept a second, so it gives up rather than emit a duplicate.

    ⚠ The prior-opening set is emptied too. `prior_opening_digests` computes its
    eight digests with the SAME function, so patching the digest alone turned all
    eight into the constant and the first candidate was rejected as a PRIOR
    opening -- the other branch, and the test would have passed for the wrong
    reason.
    """
    monkeypatch.setattr(SEL, "canonical_digest", lambda st: "c" * 64)
    monkeypatch.setattr(R, "prior_opening_digests", frozenset)
    with pytest.raises(RuntimeError, match="only 1 of 2"):
        R.generate_openings(n=2)


def test_the_PRIOR_OPENING_exclusion_is_reached_alone(monkeypatch):
    """Every candidate made to equal an H1/H2 opening: nothing may be accepted."""
    monkeypatch.setattr(SEL, "canonical_digest", lambda st: "p" * 64)
    monkeypatch.setattr(R, "prior_opening_digests", lambda: frozenset({"p" * 64}))
    with pytest.raises(RuntimeError, match="only 0 of 1"):
        R.generate_openings(n=1)


def test_the_TERMINAL_rejection_is_reached_alone(monkeypatch):
    """A candidate terminal ONLY ONCE IT IS COMPLETE: a finished game is not an
    opening.

    ⚠ `is_terminal -> True` outright does NOT reach this branch. The walk checks
    it each ply, so it breaks at move 0, the position comes back short, and the
    LENGTH check rejects it -- with or without the terminal check, so a control
    aimed here went NOT CAUGHT. Terminal at exactly `OPENING_PLIES` pegs lets the
    walk finish and leaves the terminal check as the only thing that can refuse.
    """
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    monkeypatch.setattr(TwixtState, "is_terminal",
                        lambda self: len(self.pegs) >= R.OPENING_PLIES)
    with pytest.raises(RuntimeError, match="only 0 of 1"):
        R.generate_openings(n=1)


# ═══════════ the seeded pin: it cannot exist yet, and that is enforced ══════

def test_ASSIGNING_SEEDS_CHANGES_THE_FULL_FIELD_DIGEST(openings):
    """🔴 The reason a second pin must exist at all. `task_digest` covers EVERY
    field, so the unseeded pin does NOT describe the schedule that would run."""
    unseeded = R.task_digest(R.build_tasks(openings))
    seeded = R.task_digest(R.build_tasks(openings,
                                         seed_interval=(777000000, 777000040)))
    assert unseeded == R.TASK_DIGEST
    assert seeded != unseeded


def test_THE_SEEDED_PIN_IS_SET_now_that_a_block_is_REGISTERED():
    """🔴 INVERTED 2026-09-14 by the registration. It read "is None" while no
    block existed; a block is reserved and accounted now, so the pin that names
    its schedule must exist."""
    assert isinstance(R.SEEDED_TASK_DIGEST, str)
    assert len(R.SEEDED_TASK_DIGEST) == 64


def test_an_UNSEEDED_schedule_expects_the_unseeded_pin(openings):
    assert R.expected_task_digest(R.build_tasks(openings)) == R.TASK_DIGEST


def test_a_SEEDED_schedule_is_REFUSED_while_the_seeded_pin_is_unset(
        openings, monkeypatch):
    """THE GUARD, REACHED ALONE. The pin is set now, so the branch that refuses an
    unpinned seeded schedule is only reachable by unsetting it -- and it must stay
    reachable, because that is the state every future block starts in."""
    monkeypatch.setattr(R, "SEEDED_TASK_DIGEST", None)
    seeded = R.build_tasks(openings, seed_interval=(777000000, 777000040))
    with pytest.raises(R.H3PilotError, match="SEEDED_TASK_DIGEST is None"):
        R.expected_task_digest(seeded)


def test_a_HALF_SEEDED_schedule_matches_NEITHER_pin(openings):
    tasks = R.build_tasks(openings, seed_interval=(777000000, 777000040))
    tasks[0] = dict(tasks[0], seed=None)
    with pytest.raises(R.H3PilotError, match="half-seeded"):
        R.expected_task_digest(tasks)


def test_once_the_seeded_pin_IS_set_the_seeded_schedule_matches(openings, monkeypatch):
    """The other half: if nothing were ever accepted the check would be an outage."""
    seeded = R.build_tasks(openings, seed_interval=(777000000, 777000040))
    monkeypatch.setattr(R, "SEEDED_TASK_DIGEST", R.task_digest(seeded))
    assert R.expected_task_digest(seeded) == R.task_digest(seeded)


# ═══════════ the REGISTERED schedule and its pin ════════════════════════════

def test_THE_SEEDED_PIN_IS_RECOMPUTED_FROM_THE_REGISTERED_BLOCK(openings):
    """The pin names a schedule; rebuild that schedule and it must agree."""
    from scripts.GPU.alphazero import h3_pilot_runner as RUN
    tasks = R.build_tasks(openings, seed_interval=RUN.PILOT_SEED_BLOCK)
    assert R.task_digest(tasks) == R.SEEDED_TASK_DIGEST
    assert R.expected_task_digest(tasks) == R.SEEDED_TASK_DIGEST
    assert R.SEEDED_TASK_DIGEST != R.TASK_DIGEST, "seeds change the full-field digest"


def test_THE_REGISTERED_BLOCK_IS_ACCOUNTED_ONLY_and_the_barrier_is_SATISFIED():
    """THE SEED-PREPARATION STEP, and only that. Registering is bookkeeping, not
    permission: ACCOUNTED so the barrier is satisfied, and NOT exposed and NOT
    retired because a reservation is not a draw. The gate is the separate review
    and it is still shut."""
    from scripts.GPU.alphazero import e4_screen_reference as REF
    from scripts.GPU.alphazero import h3_pilot_runner as RUN
    lo, hi = RUN.PILOT_SEED_BLOCK
    assert (lo, hi) == (202624000, 202624040)
    assert hi - lo == R.N_GAMES == 40
    for name in ("ACCOUNTED_SEED_INTERVALS", "EXPOSED_SEED_INTERVALS",
                 "RETIRED_SEED_INTERVALS", "TEST_ONLY_SEED_INTERVALS"):
        assert getattr(REF, name), f"vacuous: {name} is empty"
    for seed in range(lo, hi):
        st = REF.seed_status(seed)
        assert st["accounted"], (seed, st)
        assert not (st["exposed"] or st["retired"] or st["test_only"]), (seed, st)
        assert seed not in REF.CONSUMED_SEEDS, seed
    RUN.check_seed_registration()                      # the barrier is down
    assert RUN.H3_PILOT_EXECUTION_AUTHORIZED is False, (
        "registering a block ALSO opened the execution gate -- registration is "
        "bookkeeping, and permission is a separate review")


def test_the_registered_block_is_DISJOINT_from_every_spent_one():
    """Recomputed here, not taken from the proof's output."""
    from scripts.GPU.alphazero import h2_match_rules as H2R
    from scripts.GPU.alphazero import h3_pilot_runner as RUN
    from scripts.GPU.alphazero.d1_selection import SEED_INTERVAL as D1
    ours = set(range(*RUN.PILOT_SEED_BLOCK))
    for name, blk in (("H2 a1", H2R.H2_ATTEMPT1_SEED_BLOCK),
                      ("H2 a2", H2R.H2_ATTEMPT2_SEED_BLOCK),
                      ("H2 a3", H2R.H2_SEED_BLOCK), ("D1 §14", D1)):
        assert not (ours & set(range(*blk))), name
