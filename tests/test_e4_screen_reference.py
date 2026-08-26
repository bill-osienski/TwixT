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

#: Deliberately outside every accounted seed interval. Behavioural tests must
#: never touch a scheduled seed: rng_witness DRAWS, and drawing spends it.
SYNTHETIC = 90000001


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


def test_validate_task_requires_every_field_build_reference_agent_reads():
    E.validate_task(task())                                  # baseline accepted
    for field in E.REQUIRED_TASK_FIELDS:
        t = task()
        del t[field]
        with pytest.raises(E.E4ReferenceError):
            E.validate_task(t)


def test_validate_task_rejects_a_wrong_sha1_and_an_unknown_reference():
    with pytest.raises(E.E4ReferenceError):
        E.validate_task(task(sha1="0" * 40))
    with pytest.raises(E.E4ReferenceError):
        E.validate_task(task(ref="not_a_checkpoint"))


def test_validate_task_rejects_a_consumed_seed():
    with pytest.raises(E.E4ReferenceError):
        E.validate_task(task(seed=CONSUMED_SEEDS[0]))


def test_validate_schedule_rejects_duplicate_seeds_and_empty():
    good = [task(seed=SYNTHETIC + i, anchor="red" if i % 2 else "black") for i in range(4)]
    summary = E.validate_schedule(good)
    assert summary["n_tasks"] == 4 and summary["distinct_stream_pairs"] == 4
    with pytest.raises(E.E4ReferenceError):
        E.validate_schedule([])
    with pytest.raises(E.E4ReferenceError):
        E.validate_schedule(good + [task(seed=SYNTHETIC, anchor="black")])


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
# rng_witness DRAWS from the generators, which spends the seed even with no model
# and no game. Attempt 3 took witnesses over its own 32 scheduled seeds and burnt
# [202612000, 202612032). The guard exists so that cannot recur.

def test_rng_witness_refuses_any_accounted_seed():
    for seed in (202612000, 202612128, 202611000, 202608100):
        assert E.seed_is_accounted(seed)
        with pytest.raises(E.E4ReferenceError):
            E.rng_witness(task(seed=seed))


def test_rng_witness_allows_a_synthetic_seed():
    assert not E.seed_is_accounted(SYNTHETIC)
    w = E.rng_witness(task(seed=SYNTHETIC))
    assert len(w["search_first"]) == 4 and w["search_first"] != w["readout_first"]


def test_burnt_seeds_can_never_be_scheduled_again():
    for seed in range(202612000, 202612032, 7):
        assert E.seed_is_exposed(seed)
        with pytest.raises(E.E4ReferenceError):
            E.validate_task(task(seed=seed))
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
            E.validate_task(task(seed=seed))
        with pytest.raises(E.E4ReferenceError):
            E.rng_witness(task(seed=seed))
    assert not E.seed_is_exposed(lo - 1) and not E.seed_is_exposed(lo + 4)
    assert not E.seed_is_exposed(SYNTHETIC), "the designated test seed stays usable"


def test_the_canonical_screen_block_is_now_spent():
    """The E4 screen ran once from a8b3994; its seeds can never be scheduled again."""
    for seed in range(202612128, 202612160):
        assert E.seed_is_exposed(seed)
        with pytest.raises(E.E4ReferenceError):
            E.validate_task(task(seed=seed))
    assert not E.seed_is_exposed(202612127) and not E.seed_is_exposed(202612160)
