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


def task(seed=202612000, anchor="red", ref=REF, sha1=SHA1, **kw):
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
    t = task(seed=202612005, anchor="black")          # our side is red
    s = E.rng_stream_seeds(t)
    assert s["colour"] == "red"
    assert s["search_seed"] == 202612005 ^ SeededReferenceAgent.SEARCH_MASK["red"]
    assert s["readout_seed"] == 202612005 ^ SeededReferenceAgent.READOUT_MASK["red"]


def test_search_and_readout_streams_differ():
    s = E.rng_stream_seeds(task())
    assert s["search_seed"] != s["readout_seed"]
    w = E.rng_witness(task())
    assert w["search_first"] != w["readout_first"]
    assert len(w["search_first"]) == len(w["readout_first"]) == 4


def test_rng_witness_is_reproducible():
    assert E.rng_witness(task(seed=202612007)) == E.rng_witness(task(seed=202612007))
    assert E.rng_witness(task(seed=202612007)) != E.rng_witness(task(seed=202612008))


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
    good = [task(seed=202612000 + i, anchor="red" if i % 2 else "black") for i in range(4)]
    summary = E.validate_schedule(good)
    assert summary["n_tasks"] == 4 and summary["distinct_stream_pairs"] == 4
    with pytest.raises(E.E4ReferenceError):
        E.validate_schedule([])
    with pytest.raises(E.E4ReferenceError):
        E.validate_schedule(good + [task(seed=202612000, anchor="black")])


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
