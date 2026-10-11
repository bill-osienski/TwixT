"""A0: per-agent search budgets in the evaluation harness.

CPU fixtures only (FakeEvaluator, no model), on the official 24x24 board at the
real 280 cap; only the simulation counts are small, to keep the games short.
"""
import dataclasses
import hashlib
import json

import pytest

from scripts.GPU.alphazero import eval_readout as R
from scripts.GPU.alphazero import eval_runner as ER
from scripts.GPU.alphazero.eval_integrity import IntegrityError, validate_game_binding
from scripts.GPU.alphazero.eval_runner import (
    AgentSpec, EvalConfig, agent_sims, build_agent_pairing_tasks, build_pairing_tasks,
    cfg_from, make_result, play_eval_game,
)
from tests.eval_fakes import FakeEvaluator

CKPT = "fake.safetensors"
ARGMAX = R.ReadoutConfig(mode=R.MODE_ARGMAX)
CFG = EvalConfig(board_size=24, max_moves=280, mcts_sims=4, selection_mode="argmax")
LOW = AgentSpec("low", CKPT, ARGMAX, mcts_sims=3)
HIGH = AgentSpec("high", CKPT, ARGMAX, mcts_sims=7)
INHERIT = AgentSpec("inherit", CKPT, ARGMAX)

#: sha256 of json.dumps(play_eval_game(..., CFG, seed=5, capture=True), sort_keys=True)
#: computed on the code BEFORE A0 (plan Task 1, Step 1: state_cap, 280 plies).
LEGACY_DIGEST = "9f30fbceacd6cbeace9ef5c16ee7dcccf40e3bbc5039da0e6313c0ccbed27afc"


def _visits_by_player(records):
    out = {"red": set(), "black": set()}
    for r in records:
        out[r["player"]].add(r["root_total_visits"])
    return out


def test_an_agent_without_a_budget_uses_the_run_budget():
    assert INHERIT.mcts_sims is None
    assert agent_sims(INHERIT, CFG) == 4
    assert agent_sims(None, CFG) == 4
    assert agent_sims(LOW, CFG) == 3


@pytest.mark.parametrize("bad", [0, -1, 2.0, True, "4"])
def test_an_agent_budget_must_be_a_positive_int(bad):
    with pytest.raises(ValueError, match="mcts_sims"):
        AgentSpec("x", CKPT, ARGMAX, mcts_sims=bad)


def test_the_cfg_override_changes_only_the_simulation_count():
    base, over = cfg_from(CFG), cfg_from(CFG, n_simulations=7)
    assert (base.n_simulations, over.n_simulations) == (4, 7)
    assert dataclasses.replace(over, n_simulations=4) == base


def test_each_colour_searches_at_its_own_budget():
    _w, _r, n, records = play_eval_game(
        FakeEvaluator(), FakeEvaluator(), CFG, seed=11, capture=True,
        red_sims=3, black_sims=7)
    assert n >= 2
    assert _visits_by_player(records) == {"red": {3}, "black": {7}}


def test_the_legacy_game_is_byte_identical_to_before_a0():
    out = play_eval_game(FakeEvaluator(), FakeEvaluator(), CFG, seed=5, capture=True)
    digest = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    assert digest == LEGACY_DIGEST
    assert _visits_by_player(out[3]) == {"red": {4}, "black": {4}}


def test_explicit_run_budgets_reproduce_the_legacy_game():
    legacy = play_eval_game(FakeEvaluator(), FakeEvaluator(), CFG, seed=5, capture=True)
    explicit = play_eval_game(FakeEvaluator(), FakeEvaluator(), CFG, seed=5, capture=True,
                              red_sims=4, black_sims=4)
    assert explicit == legacy


def test_the_integrity_check_is_per_side(monkeypatch):
    """If a side's override is lost, that side's first ply must stop the game."""
    real = ER.cfg_from
    monkeypatch.setattr(ER, "cfg_from", lambda config, n_simulations=None: real(config))
    with pytest.raises(IntegrityError, match=(
            r"ply 0: simulation budget mismatch -- root completed 4 visits, "
            r"expected exactly 3")):
        play_eval_game(FakeEvaluator(), FakeEvaluator(), CFG, seed=11, red_sims=3)
    with pytest.raises(IntegrityError, match=(
            r"ply 1: simulation budget mismatch -- root completed 4 visits, "
            r"expected exactly 7")):
        play_eval_game(FakeEvaluator(), FakeEvaluator(), CFG, seed=11, black_sims=7)


def _play(task, tmp_path):
    return ER._play_and_build_result(task, FakeEvaluator(), FakeEvaluator(), CFG,
                                     capture=True, replay_dir=str(tmp_path))


def _records(result):
    with open(result.replay_path) as fh:
        return json.load(fh)["moves"]


def test_the_budget_travels_with_the_agent_across_the_colour_swap(tmp_path):
    tasks = build_agent_pairing_tasks("p", LOW, HIGH, 2, base_seed=100)
    assert [t.red_agent.agent_id for t in tasks] == ["low", "high"]
    for task in tasks:
        result = _play(task, tmp_path)
        red, black = task.red_agent.mcts_sims, task.black_agent.mcts_sims
        assert _visits_by_player(_records(result)) == {"red": {red}, "black": {black}}
        assert (result.red_mcts_sims, result.black_mcts_sims) == (red, black)


def test_an_agent_without_a_budget_records_the_run_budget(tmp_path):
    task = build_agent_pairing_tasks("p", LOW, INHERIT, 2, base_seed=100)[0]
    result = _play(task, tmp_path)
    assert (result.red_mcts_sims, result.black_mcts_sims) == (3, 4)


def test_legacy_results_record_no_budget(tmp_path):
    task = build_pairing_tasks("p", "a.safetensors", "b.safetensors", 2, 100,
                               pairing_index=0)[0]
    result = _play(task, tmp_path)
    assert (result.red_mcts_sims, result.black_mcts_sims) == (None, None)
    assert _visits_by_player(_records(result)) == {"red": {4}, "black": {4}}


def test_rows_written_before_a0_still_load():
    task = build_pairing_tasks("p", "a.safetensors", "b.safetensors", 2, 100,
                               pairing_index=0)[0]
    row = dataclasses.asdict(make_result(task, "red", "win", 9))
    del row["red_mcts_sims"], row["black_mcts_sims"]
    assert ER.EvalGameResult(**row).red_mcts_sims is None


def test_the_binding_check_refuses_a_budget_that_is_not_the_agents(tmp_path):
    task = build_agent_pairing_tasks("p", LOW, HIGH, 2, base_seed=100)[0]
    result = _play(task, tmp_path)
    validate_game_binding(result, task)
    for field in ("red_mcts_sims", "black_mcts_sims"):
        with pytest.raises(IntegrityError, match="search budget mismatch"):
            validate_game_binding(dataclasses.replace(result, **{field: 5}), task)


def test_the_binding_check_asks_no_budget_of_agents_without_one():
    other = AgentSpec("other", CKPT, ARGMAX)
    task = build_agent_pairing_tasks("p", INHERIT, other, 2, base_seed=100)[0]
    validate_game_binding(make_result(task, "red", "win", 9), task)
