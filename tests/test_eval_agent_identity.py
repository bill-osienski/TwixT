"""Agent identity: colour binding, winner attribution, legacy preservation."""
import pytest

from scripts.GPU.alphazero import eval_readout as R
from scripts.GPU.alphazero.eval_runner import (
    AGENT_COMPARISON_UNIT, AgentSpec, build_agent_pairing_tasks,
    build_pairing_tasks, make_result,
)

CKPT = "checkpoints/x/model_iter_0001.safetensors"
CONTROL = AgentSpec("control", CKPT, R.ReadoutConfig(mode=R.MODE_ARGMAX))
CANDIDATE = AgentSpec("candidate", CKPT,
                      R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB))


def test_agent_tasks_alternate_colours_by_game_index():
    tasks = build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 4, 100)
    assert [t.red_agent.agent_id for t in tasks] == [
        "control", "candidate", "control", "candidate"]
    assert [t.black_agent.agent_id for t in tasks] == [
        "candidate", "control", "candidate", "control"]


def test_agent_tasks_carry_the_readout_with_the_agent_across_the_swap():
    tasks = build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 2, 100)
    for t in tasks:
        assert t.red_agent.readout.mode != t.black_agent.readout.mode
        red_is_control = t.red_agent.agent_id == "control"
        assert (t.red_agent.readout.mode == R.MODE_ARGMAX) is red_is_control


def test_agent_tasks_still_fill_the_checkpoint_fields():
    tasks = build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 2, 100)
    assert all(t.red_checkpoint == CKPT and t.black_checkpoint == CKPT
               for t in tasks)


def test_agent_task_seeds_are_task_derived_and_contiguous():
    tasks = build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 4, 100)
    assert [t.seed for t in tasks] == [100, 101, 102, 103]
    assert [t.task_id for t in tasks] == [0, 1, 2, 3]


def test_agent_tasks_reject_odd_game_counts():
    with pytest.raises(ValueError):
        build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 3, 100)


def test_agent_tasks_reject_fewer_than_two_games():
    with pytest.raises(ValueError):
        build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 0, 100)


def test_winner_agent_id_follows_the_colour_that_won():
    tasks = build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 2, 100)
    r0 = make_result(tasks[0], "red", "win", 50)      # game 0: red == control
    assert r0.winner_agent_id == "control"
    r1 = make_result(tasks[1], "red", "win", 50)      # game 1: red == candidate
    assert r1.winner_agent_id == "candidate"


def test_draws_leave_winner_agent_id_null_not_empty():
    tasks = build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 2, 100)
    r = make_result(tasks[0], None, "state_cap", 280)
    assert r.winner_agent_id is None
    assert r.red_score == 0.5 and r.black_score == 0.5


def test_agent_results_are_labelled_as_agent_comparisons():
    tasks = build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 2, 100)
    r = make_result(tasks[0], "black", "win", 50)
    assert r.comparison_unit == AGENT_COMPARISON_UNIT
    assert r.same_checkpoint is True
    assert r.red_readout["mode"] == R.MODE_ARGMAX
    assert r.black_readout["mode"] == R.MODE_HOEFFDING_LCB


def test_results_carry_the_COMPLETE_readout_config_not_just_the_mode():
    """`tournament` and `opening_then_argmax` are BOTH mode
    'opening_temperature' and differ only in temp_low. Recording the mode
    alone would make the two experiments indistinguishable in the artifact.
    """
    tournament = AgentSpec("control", CKPT, R.ReadoutConfig(
        mode=R.MODE_OPENING_TEMPERATURE, temp_high=1.0, temp_low=0.1))
    then_argmax = AgentSpec("candidate", CKPT, R.ReadoutConfig(
        mode=R.MODE_OPENING_TEMPERATURE, temp_high=1.0, temp_low=0.0))
    task = build_agent_pairing_tasks("p", then_argmax, tournament, 2, 100)[0]
    r = make_result(task, "red", "win", 50)
    assert r.red_readout["mode"] == r.black_readout["mode"]
    assert r.red_readout["temp_low"] == 0.0
    assert r.black_readout["temp_low"] == 0.1
    assert r.red_readout["opening_temp_plies"] == 20
    assert r.red_readout["temp_high"] == 1.0


def test_different_checkpoints_are_reported_as_not_same_checkpoint():
    other = AgentSpec("control", "checkpoints/y/model_iter_0002.safetensors",
                      R.ReadoutConfig(mode=R.MODE_ARGMAX))
    task = build_agent_pairing_tasks("p", CANDIDATE, other, 2, 100)[0]
    r = make_result(task, "red", "win", 50)
    assert r.same_checkpoint is False


def test_legacy_checkpoint_tasks_carry_no_agent_fields():
    # NEGATIVE CASE, constructed: the legacy path must stay unlabelled, so a
    # consumer can tell the two artifact kinds apart.
    tasks = build_pairing_tasks("p", "a.safetensors", "b.safetensors", 2, 100, 0)
    r = make_result(tasks[0], "red", "win", 50)
    assert r.comparison_unit is None
    assert r.winner_agent_id is None
    assert r.red_agent_id is None
    assert r.red_readout is None
    assert r.same_checkpoint is None
    assert r.winner_checkpoint == "a.safetensors"
