"""Checks for the E4 screen's reference-agent construction.

No model is loaded and no game is played: these cover the parts that decide
whether a scheduled seed reaches the generators that pick our moves.
"""
import pytest

from scripts.GPU.alphazero import e4_screen_reference as E
from scripts.GPU.alphazero.twixtbot_g3_reference import SeededReferenceAgent
from scripts.GPU.alphazero.twixtbot_g3_schedule import CONSUMED_SEEDS, REFERENCE_CHECKPOINTS

REF = "calib020_0001"
SHA1 = REFERENCE_CHECKPOINTS[REF]["sha1"]

#: THE SEED TESTS DRAW FROM. Inside TEST_ONLY_SEED_INTERVALS, so it is ineligible
#: for any schedule by construction and drawing from it as often as we like costs
#: nothing. It replaces the old ad-hoc `SYNTHETIC = 90000001`, which was drawn
#: from twice in preserved evidence and left out of the registry anyway.
SYNTHETIC = 90009001

#: A seed that MAY be scheduled: unreserved, not a test seed, and NEVER drawn from
#: -- `rng_witness` refuses it, so it cannot be spent by accident. Used only where
#: a test needs `validate_schedule_executable` to accept.
SCHEDULABLE = 91000001


def task(seed=SYNTHETIC, anchor="red", ref=REF, sha1=SHA1, **kw):
    t = {"seed": seed, "reference": ref, "reference_sha1": sha1, "anchor_colour": anchor}
    t.update(kw)
    return t


def test_reference_colour_is_the_opposite_of_the_anchor():
    assert E.reference_colour(task(anchor="red")) == "black"
    assert E.reference_colour(task(anchor="black")) == "red"


def test_reference_colour_rejects_a_bad_anchor():
    for bad in ("purple", None, "RED"):
        with pytest.raises(E.E4ReferenceError):
            E.reference_colour(task(anchor=bad))


def test_stream_seeds_use_the_qualified_masks_not_a_copy():
    t = task(seed=SYNTHETIC + 5, anchor="black")      # our side is red
    s = E.rng_stream_seeds(t)
    assert s["colour"] == "red"
    assert s["search_seed"] == (SYNTHETIC + 5) ^ SeededReferenceAgent.SEARCH_MASK["red"]
    assert s["readout_seed"] == (SYNTHETIC + 5) ^ SeededReferenceAgent.READOUT_MASK["red"]


def test_search_and_readout_streams_differ():
    s = E.rng_stream_seeds(task())
    assert s["search_seed"] != s["readout_seed"]
    w = E.rng_witness(task())
    assert w["search_first"] != w["readout_first"]
    assert len(w["search_first"]) == len(w["readout_first"]) == 4


def test_rng_witness_is_reproducible():
    assert E.rng_witness(task(seed=SYNTHETIC + 7)) == E.rng_witness(task(seed=SYNTHETIC + 7))
    assert E.rng_witness(task(seed=SYNTHETIC + 7)) != E.rng_witness(task(seed=SYNTHETIC + 8))


@pytest.mark.parametrize("check", ["validate_task_structure", "validate_task_executable"])
def test_validate_task_requires_every_field_build_reference_agent_reads(check):
    fn = getattr(E, check)
    fn(task())                                               # baseline accepted
    for field in E.REQUIRED_TASK_FIELDS:
        t = task()
        del t[field]
        with pytest.raises(E.E4ReferenceError):
            fn(t)


@pytest.mark.parametrize("check", ["validate_task_structure", "validate_task_executable"])
def test_validate_task_rejects_a_wrong_sha1_and_an_unknown_reference(check):
    fn = getattr(E, check)
    with pytest.raises(E.E4ReferenceError):
        fn(task(sha1="0" * 40))
    with pytest.raises(E.E4ReferenceError):
        fn(task(ref="not_a_checkpoint"))


def test_only_the_executable_check_rejects_a_consumed_seed():
    """THE SPLIT, on one task: structure passes, execution does not."""
    t = task(seed=CONSUMED_SEEDS[0])
    E.validate_task_structure(t)                             # still well formed
    with pytest.raises(E.E4ReferenceError):
        E.validate_task_executable(t)                        # but not runnable


def test_validate_schedule_rejects_duplicate_seeds_and_empty():
    good = [task(seed=SCHEDULABLE + i, anchor="red" if i % 2 else "black") for i in range(4)]
    for fn in (E.validate_schedule_structure, E.validate_schedule_executable):
        summary = fn(good)
        assert summary["n_tasks"] == 4 and summary["distinct_stream_pairs"] == 4
        with pytest.raises(E.E4ReferenceError):
            fn([])
        with pytest.raises(E.E4ReferenceError):
            fn(good + [task(seed=SCHEDULABLE, anchor="black")])


def test_a_test_seed_may_be_built_on_but_never_scheduled():
    """The band's whole contract, both halves.

    Building an agent on a test seed is fine -- unit tests do it constantly. What
    must never happen is one appearing in a SCHEDULE, because these are drawn from
    by design, and a schedule of already-drawn seeds is the reuse the registry
    exists to prevent.
    """
    t = task(seed=SYNTHETIC)
    E.validate_task_structure(t)
    E.validate_task_executable(t)                        # buildable
    E.validate_schedule_structure([t])                   # well formed
    with pytest.raises(E.E4ReferenceError, match="never appear in a schedule"):
        E.validate_schedule_executable([t])              # NOT schedulable
    assert E.seed_is_test_only(SYNTHETIC)
    assert not E.seed_is_test_only(SCHEDULABLE)
    E.validate_schedule_executable([task(seed=SCHEDULABLE)])   # the control


def test_there_is_no_switch_that_turns_the_seed_check_off():
    """The two questions are two REQUIRED functions, never one optional keyword.

    A `require_unspent=` switch would default, and a default is a switch-off. So
    the structural entry points must take no argument that could relax the
    executable one -- asserted against the real signatures.
    """
    import inspect
    for name in ("validate_task_structure", "validate_task_executable",
                 "validate_schedule_structure", "validate_schedule_executable"):
        params = list(inspect.signature(getattr(E, name)).parameters.values())
        assert len(params) == 1, f"{name} takes {len(params)} parameters, expected exactly 1"
        assert params[0].default is inspect.Parameter.empty, f"{name} has a defaulted parameter"


def test_frozen_settings_report_the_real_noise_mechanism():
    fs = E.frozen_settings()
    assert fs["noise_suppression"] == "search_with_root(state, add_noise=False)"
    assert "dirichlet_eps" not in fs["eval_config"]
    assert fs["eval_config"]["mcts_sims"] == 400
    assert fs["eval_config"]["max_moves"] == 280


def test_cfg_from_really_does_not_propagate_dirichlet_eps():
    """The claim in the docstring, checked against the code rather than asserted."""
    from scripts.GPU.alphazero.eval_runner import cfg_from
    from scripts.GPU.alphazero.mcts import MCTSConfig
    cfg = cfg_from(E.eval_config())
    assert cfg.dirichlet_eps == MCTSConfig().dirichlet_eps != 0.0


# --- the seed-accounting guard ---------------------------------------------
# rng_witness DRAWS from the generators, which strikes a SCHEDULABLE seed off even
# with no model and no game. Attempt 3 took witnesses over its own 32 scheduled
# seeds and burnt [202612000, 202612032). The guard exists so that cannot recur.
# Draws inside TEST_ONLY_SEED_INTERVALS strike nothing off: no schedule may
# contain one, so there is nothing to lose.

def test_rng_witness_refuses_any_accounted_seed():
    for seed in (202612000, 202612128, 202611000, 202608100):
        assert E.seed_is_accounted(seed)
        with pytest.raises(E.E4ReferenceError):
            E.rng_witness(task(seed=seed))


def test_rng_witness_allows_a_test_only_seed():
    assert not E.seed_is_accounted(SYNTHETIC) and E.seed_is_test_only(SYNTHETIC)
    w = E.rng_witness(task(seed=SYNTHETIC))
    assert len(w["search_first"]) == 4 and w["search_first"] != w["readout_first"]


def test_rng_witness_is_an_ALLOWLIST_so_an_unknown_seed_is_refused():
    """THE CLASS FIX. A denylist draws from whatever nobody thought to list.

    That is precisely how 90000001 was witnessed into the frozen plan and used for
    the first real agent call while still absent from the registry. An arbitrary
    SCHEDULABLE seed -- one no interval mentions, in either direction -- must now be
    REFUSED rather than drawn from and struck off in silence.
    """
    for seed in (777777, 12345678, SCHEDULABLE, 90000000, 90009100):
        assert not E.seed_is_accounted(seed), seed
        assert not E.seed_is_unavailable(seed), seed
        assert not E.seed_is_test_only(seed), seed
        with pytest.raises(E.E4ReferenceError, match="would strike it off"):
            E.rng_witness(task(seed=seed))


def test_burnt_seeds_can_never_be_scheduled_again():
    for seed in range(202612000, 202612032, 7):
        assert E.seed_is_exposed(seed)
        E.validate_task_structure(task(seed=seed))           # still parses
        with pytest.raises(E.E4ReferenceError):
            E.validate_task_executable(task(seed=seed))      # never runs
    assert not E.seed_is_exposed(202612032)


def test_stream_derivation_does_not_draw():
    """rng_stream_seeds is XOR only, so it may be used on scheduled seeds."""
    s = E.rng_stream_seeds(task(seed=202612128))
    assert s["search_seed"] == 202612128 ^ SeededReferenceAgent.SEARCH_MASK["black"]


# --- the real construction path, exercised without a model ------------------

class _FakeEvaluator:
    """Carries the identity tags load_reference_evaluator sets. Never called."""

    def __init__(self, reference=REF, sha1=SHA1):
        self._g3_reference = reference
        self._g3_sha1 = sha1


def test_build_calls_the_qualified_path_and_seeds_both_streams():
    """Exercises build() itself: a witness computed alongside proves nothing if
    the construction path stops delegating or seeds different generators."""
    import random

    t = task(seed=SYNTHETIC, anchor="red")          # our side is black
    agent = E.build(t, evaluator=_FakeEvaluator())

    assert agent.colour == "black" == E.reference_colour(t)
    assert agent.seed == SYNTHETIC
    want_search = random.Random(SYNTHETIC ^ SeededReferenceAgent.SEARCH_MASK["black"])
    want_readout = random.Random(SYNTHETIC ^ SeededReferenceAgent.READOUT_MASK["black"])
    assert agent.mcts.rng.getstate() == want_search.getstate()
    assert agent.readout_rng.getstate() == want_readout.getstate()
    assert agent.mcts.rng.getstate() != agent.readout_rng.getstate()
    assert agent.moves_made == 0


def test_build_refuses_an_evaluator_from_a_different_checkpoint():
    from scripts.GPU.alphazero.twixtbot_g3_reference import ReferenceError
    t = task(seed=SYNTHETIC)
    with pytest.raises((ReferenceError, E.E4ReferenceError)):
        E.build(t, evaluator=_FakeEvaluator(reference="0379", sha1="0" * 40))
    with pytest.raises((ReferenceError, E.E4ReferenceError)):
        E.build(t, evaluator=object())          # no identity tags at all


def test_build_refuses_a_task_missing_a_required_field():
    t = task(seed=SYNTHETIC)
    del t["anchor_colour"]
    with pytest.raises(E.E4ReferenceError):
        E.build(t, evaluator=_FakeEvaluator())


def test_build_seeds_differ_by_colour():
    import random
    a = E.build(task(seed=SYNTHETIC, anchor="red"), evaluator=_FakeEvaluator())    # black
    b = E.build(task(seed=SYNTHETIC, anchor="black"), evaluator=_FakeEvaluator())  # red
    assert a.colour != b.colour
    assert a.mcts.rng.getstate() != b.mcts.rng.getstate()


@pytest.mark.parametrize("lo", [90001000, 90002000])
def test_the_integration_qualification_seeds_are_recorded_as_consumed(lo):
    """Both E4 integration attempts drew from both generators on their seeds."""
    for seed in range(lo, lo + 4):
        assert E.seed_is_exposed(seed)
        with pytest.raises(E.E4ReferenceError):
            E.validate_task_executable(task(seed=seed))
        with pytest.raises(E.E4ReferenceError):
            E.rng_witness(task(seed=seed))
    assert not E.seed_is_exposed(lo - 1) and not E.seed_is_exposed(lo + 4)
    assert not E.seed_is_exposed(SYNTHETIC), "the test band records no history"


def test_the_canonical_screen_block_records_drawn_and_undrawn_separately():
    """The screen ran once from a8b3994. 24 tasks played; 8 were skipped undrawn.

    Exposure is a claim about what HAPPENED, so it covers exactly the 24 seeds a
    generator was built from. Retirement is a claim about what MAY happen, and it
    covers all 32: the one-shot schedule completed, and replaying only the 8 the
    early stop declined to play would be choosing tasks after seeing the result.
    Both are refused for execution; neither is refused for reading.
    """
    played = list(range(202612128, 202612136)) + list(range(202612144, 202612160))
    skipped = list(range(202612136, 202612144))
    assert len(played) == 24 and len(skipped) == 8

    for seed in played:
        assert E.seed_is_exposed(seed), f"{seed} drove a real generator"
    for seed in skipped:
        assert not E.seed_is_exposed(seed), (
            f"{seed} was never drawn from; calling it exposed overstates the record")

    for seed in played + skipped:
        assert E.seed_is_retired(seed) and E.seed_is_unavailable(seed)
        E.validate_task_structure(task(seed=seed))           # parses forever
        with pytest.raises(E.E4ReferenceError):
            E.validate_task_executable(task(seed=seed))      # runs never
        with pytest.raises(E.E4ReferenceError):
            E.rng_witness(task(seed=seed))

    assert not E.seed_is_unavailable(202612127)
    assert not E.seed_is_unavailable(202612160)


def test_a_skipped_seed_is_refused_for_retirement_not_for_exposure():
    """The two registries must not be collapsed: the REASON is part of the record."""
    with pytest.raises(E.E4ReferenceError, match="RETIRED"):
        E.validate_task_executable(task(seed=202612136))     # skipped, undrawn
    with pytest.raises(E.E4ReferenceError, match="EXPOSED"):
        E.validate_task_executable(task(seed=202612128))     # played, drawn
    assert E.seed_status(202612136) == {
        "exposed": False, "retired": True, "accounted": True, "test_only": False}
    assert E.seed_status(202612128) == {
        "exposed": True, "retired": True, "accounted": True, "test_only": False}
    assert E.seed_status(SYNTHETIC) == {
        "exposed": False, "retired": False, "accounted": False, "test_only": True}
    assert E.seed_status(90000001) == {
        "exposed": True, "retired": False, "accounted": False, "test_only": False}


def test_the_canonical_plan_still_loads_and_verifies_after_its_seeds_are_gone():
    """The spent schedule remains immutable historical evidence, not a dead file."""
    from scripts.GPU.alphazero import e4_screen_runner as H
    plan = H.load_canonical_plan(
        "docs/superpowers/evidence/2026-08-25-t1j-e4-preflight-attempt4/"
        "06_endpoint_screen_plan.json")
    assert len(plan["tasks"]) == H.CANONICAL_N_TASKS
    H.verify_tasks(plan["tasks"])                            # digest still binds
    summary = E.validate_schedule_structure(plan["tasks"])
    assert summary["n_tasks"] == 32 and summary["search_readout_disjoint"]
    with pytest.raises(E.E4ReferenceError):
        E.validate_schedule_executable(plan["tasks"])        # but it may not run


def test_the_old_synthetic_seed_90000001_is_recorded_as_exposed():
    """It was SCHEDULABLE and drawn from TWICE; both draws are preserved.

    Draw 1: the preflight froze an `rng_witness` on it into the canonical plan.
    Draw 2: the harness qualification's one real reference-agent call.

    The second is asserted against the CAPTURED RUN, not the script that
    configured it -- a source line proves which seed was written down, and the
    run's own record is what proves the call happened and both generators moved.
    """
    import json
    assert E.seed_is_exposed(90000001)
    assert E.seed_is_unavailable(90000001)
    with pytest.raises(E.E4ReferenceError, match="EXPOSED"):
        E.validate_task_executable(task(seed=90000001))
    with pytest.raises(E.E4ReferenceError):
        E.rng_witness(task(seed=90000001))
    with pytest.raises(E.E4ReferenceError):
        E.validate_schedule_executable([task(seed=90000001)])
    # only that one seed: its neighbours were never drawn from
    assert not E.seed_is_exposed(90000000) and not E.seed_is_exposed(90000002)

    # DRAW 1 -- the witness, still in the frozen plan
    plan = json.load(open("docs/superpowers/evidence/2026-08-25-t1j-e4-preflight-attempt4/"
                          "06_endpoint_screen_plan.json"))
    demo = plan["seed_accounting"]["witness_demonstration"]
    assert demo["synthetic_seed"] == 90000001
    assert len(demo["witness"]["search_first"]) == 4
    assert len(demo["witness"]["readout_first"]) == 4

    # DRAW 2 -- configuration, then the RUN that consumed it
    qual_dir = "docs/superpowers/evidence/2026-08-25-t1j-e4-harness-qualification/"
    src = open(qual_dir + "04_qualify.py.txt").read().splitlines()
    assert src[25].strip() == "SYNTHETIC = 90000001"           # :26
    assert '"seed": SYNTHETIC + i' in src[49]                  # :50
    assert 'REF.build(task(0, "black"), evaluator=EV)' in src[69]   # :70 -> +0
    run = open(qual_dir + "02_qualification.txt").read()
    assert "THE ONE REAL CALL" in run
    assert "the single real call completed (validate_ply passed)" in run
    assert "returned move (14, 13) is legal in our engine" in run
    assert "exactly one move made by this agent" in run
    assert "search RNG advanced" in run and "readout RNG advanced" in run


def test_the_test_band_never_overlaps_a_recorded_or_retired_seed():
    """An invariant, not an observation: the bands must stay disjoint.

    A test seed that was also exposed would be drawn from forever while claiming
    to record history; one that was also retired would be unusable by tests. Both
    are silent, so they are asserted rather than assumed.
    """
    def spread(intervals):
        return {s for lo, hi in intervals for s in range(lo, hi)}

    test_only = spread(E.TEST_ONLY_SEED_INTERVALS)
    assert test_only, "the test band may not be empty; tests must draw from something"
    assert not (test_only & spread(E.EXPOSED_SEED_INTERVALS))
    assert not (test_only & spread(E.RETIRED_SEED_INTERVALS))
    assert not (test_only & spread(E.ACCOUNTED_SEED_INTERVALS))
    assert not (test_only & set(CONSUMED_SEEDS))
    assert SYNTHETIC in test_only and SCHEDULABLE not in test_only
    for lo, hi in E.TEST_ONLY_SEED_INTERVALS:
        assert lo < hi, "half-open, non-empty"
