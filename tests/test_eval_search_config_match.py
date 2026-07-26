"""Asymmetric same-checkpoint match support (v17 Task 6).

Two agents may share identical checkpoint bytes and still have to search
differently — v17 runs one net against itself at two FPU coefficients. The
checkpoint-swap color balance cannot express that on its own, because swapping
a path for itself is a no-op, so the search config has to travel with the agent.

What these tests establish:

  * the default (no per-agent config) path is unchanged, against a golden
    captured BEFORE the change — see `tests/golden/eval_runner_default_path.json`;
  * an agent is the pair (checkpoint, search config) and the color swap moves
    both together, so no config can end up pinned to a color;
  * the per-agent config is genuinely in effect on its own side, proven by
    observing search behaviour rather than by reading source;
  * agents that differ outside the declared field under study are refused, and
    refused before any evaluator is loaded;
  * a match that uses the feature records both complete effective configs.

These tests deliberately draw NO conclusion about playing strength. A fake
uniform evaluator plays both sides; who wins is an artifact of the fake and
means nothing. Every assertion here is about configuration plumbing.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile

import pytest

from scripts.GPU.alphazero import fpu_v17_provenance as prov
from scripts.GPU.alphazero.eval_checkpoint_match import build_match_tasks, run_match
from scripts.GPU.alphazero.eval_runner import (
    EvalConfig, build_pairing_tasks, cfg_from, make_result, play_eval_game,
    require_agent_config_consistency, run_game_tasks,
)
from scripts.GPU.alphazero.mcts import MCTSConfig
from tests.eval_fakes import FakeEvaluator, fake_evaluator_factory

GOLDEN = os.path.join(os.path.dirname(__file__), "golden",
                      "eval_runner_default_path.json")

# Added by this task; absent from the pre-change golden by definition.
NEW_TASK_FIELDS = ("red_mcts", "black_mcts", "red_agent", "black_agent")

# The field v17 varies between the two agents. Everything else must match.
V17_FIELD = "fpu_shipped_policy_mass_reduction"

A_CKPT, B_CKPT = "A.safetensors", "B.safetensors"
SAME_CKPT = "same.safetensors"


def tiny_cfg(**kw) -> EvalConfig:
    """The established tiny eval config (matches tests/test_eval_runner.py)."""
    base = dict(board_size=8, mcts_sims=8, mcts_eval_batch_size=4,
                mcts_stall_flush_sims=4, opening_temp_plies=4,
                temp_high=1.0, temp_low=0.1, max_moves=12)
    base.update(kw)
    return EvalConfig(**base)


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN) as fh:
        return json.load(fh)


def _task_dict(task, *, drop_new_fields: bool):
    d = dataclasses.asdict(task)
    if drop_new_fields:
        for name in NEW_TASK_FIELDS:
            d.pop(name)
    return d


# --------------------------------------------------------------------------
# 1. The default path is unchanged
# --------------------------------------------------------------------------

def test_default_pairing_tasks_match_pre_change_golden(golden):
    """Ignoring only the two additive fields, tasks are exactly as before."""
    for idx in (0, 2):
        tasks = build_pairing_tasks("pair", A_CKPT, B_CKPT, games=6,
                                    base_seed=1000, pairing_index=idx)
        got = [_task_dict(t, drop_new_fields=True) for t in tasks]
        assert got == golden["pairing_tasks"][str(idx)]


def test_default_tasks_carry_no_per_agent_config():
    """The additive fields are None unless a caller asks for them."""
    tasks = build_pairing_tasks("pair", A_CKPT, B_CKPT, games=6,
                                base_seed=1000, pairing_index=0)
    assert all(t.red_mcts is None and t.black_mcts is None for t in tasks)


def test_default_play_eval_game_matches_pre_change_golden(golden):
    """Same outcomes AND the same per-ply move sequences and search records."""
    for mode, rows in golden["play_eval_game"].items():
        cfg = tiny_cfg(selection_mode=mode)
        for row in rows:
            winner, reason, n, records = play_eval_game(
                FakeEvaluator(), FakeEvaluator(), cfg, seed=row["seed"],
                capture=True)
            assert winner == row["winner"]
            assert reason == row["reason"]
            assert n == row["n_moves"]
            assert [[r["row"], r["col"]] for r in records] == row["moves"]
            assert json.loads(json.dumps(records)) == row["records"]


def test_golden_move_sequences_actually_diverge(golden):
    """Guard the guard.

    With this tiny config every game hits the ply cap, so winner/reason/n_moves
    are constant across seeds. If the move sequences were also constant the
    golden above would pass no matter what the code did.
    """
    for mode, rows in golden["play_eval_game"].items():
        seqs = {tuple(map(tuple, r["moves"])) for r in rows}
        assert len(seqs) == len(rows), f"{mode}: seeds do not diverge"


def test_default_run_match_artifacts_are_byte_identical(golden):
    """The written summary and per-game JSONL bytes are unchanged."""
    with tempfile.TemporaryDirectory() as td:
        output = os.path.join(td, "match.json")
        summary = run_match(
            a_ckpt=A_CKPT, b_ckpt=B_CKPT, games=4, base_seed=777,
            config=tiny_cfg(), workers=1, output=output,
            evaluator_factory=fake_evaluator_factory)
        stem, _ = os.path.splitext(output)
        with open(f"{stem}_games.jsonl", "rb") as fh:
            games_bytes = fh.read()
    assert games_bytes.decode() == golden["run_match"]["games_jsonl"]
    stripped = {k: v for k, v in summary.items()
                if k not in ("git_commit", "generated_at")}
    assert stripped == golden["run_match"]["summary"]


def test_default_summary_has_no_agent_mcts_key():
    """The provenance key appears only when the feature is used."""
    summary = run_match(a_ckpt=A_CKPT, b_ckpt=B_CKPT, games=2, base_seed=1,
                        config=tiny_cfg(), workers=1, output=None,
                        evaluator_factory=fake_evaluator_factory)
    assert "agent_mcts" not in summary["config"]


# --------------------------------------------------------------------------
# 2. An agent is (checkpoint, config), and the color swap moves both
# --------------------------------------------------------------------------

def _v17_pair(base: MCTSConfig, r: float):
    """The v17 agent pair: identical configs apart from the coefficient."""
    return (dataclasses.replace(base, **{V17_FIELD: 0.0}),
            dataclasses.replace(base, **{V17_FIELD: r}))


def test_color_swap_moves_checkpoint_and_config_together():
    a_cfg, b_cfg = _v17_pair(cfg_from(tiny_cfg()), 0.25)
    tasks = build_pairing_tasks("p", A_CKPT, B_CKPT, games=6, base_seed=0,
                                pairing_index=0, a_mcts=a_cfg, b_mcts=b_cfg)
    for t in tasks:
        # Each side is a whole agent, never a mix of one agent's checkpoint
        # with the other agent's config.
        assert (t.red_checkpoint, t.red_mcts) in ((A_CKPT, a_cfg), (B_CKPT, b_cfg))
        assert (t.black_checkpoint, t.black_mcts) in ((A_CKPT, a_cfg), (B_CKPT, b_cfg))
        assert t.red_checkpoint != t.black_checkpoint
        assert t.red_mcts is not t.black_mcts


def test_exact_color_balance_of_configs_on_same_checkpoint():
    """The case the checkpoint swap alone cannot express."""
    a_cfg, b_cfg = _v17_pair(cfg_from(tiny_cfg()), 0.25)
    games = 8
    tasks = build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=games,
                                base_seed=0, pairing_index=0,
                                a_mcts=a_cfg, b_mcts=b_cfg)
    # Checkpoints are indistinguishable; the configs carry the whole balance.
    assert {t.red_checkpoint for t in tasks} == {SAME_CKPT}
    assert sum(t.red_mcts == a_cfg for t in tasks) == games // 2
    assert sum(t.black_mcts == a_cfg for t in tasks) == games // 2
    assert sum(t.red_mcts == b_cfg for t in tasks) == games // 2
    assert sum(t.black_mcts == b_cfg for t in tasks) == games // 2
    # Even game_idx -> A is red, matching the documented checkpoint rule.
    for t in tasks:
        expected_red = a_cfg if t.game_idx % 2 == 0 else b_cfg
        assert t.red_mcts == expected_red
        assert t.black_mcts == (b_cfg if t.game_idx % 2 == 0 else a_cfg)


def test_tasks_stay_deterministic_with_per_agent_configs():
    a_cfg, b_cfg = _v17_pair(cfg_from(tiny_cfg()), 0.25)
    kw = dict(games=4, base_seed=1000, pairing_index=0,
              a_mcts=a_cfg, b_mcts=b_cfg)
    t1 = build_pairing_tasks("p", A_CKPT, B_CKPT, **kw)
    t2 = build_pairing_tasks("p", A_CKPT, B_CKPT, **kw)
    assert t1 == t2
    # Ids and seeds are unaffected by the new fields.
    plain = build_pairing_tasks("p", A_CKPT, B_CKPT, games=4, base_seed=1000,
                                pairing_index=0)
    assert [(t.task_id, t.seed) for t in t1] == [(t.task_id, t.seed) for t in plain]


def test_symmetric_relabeling_swaps_configs(pairing_index=0):
    """Task/config symmetry: naming the same two agents in the opposite order
    produces the mirror-image assignment, nothing else."""
    a_cfg, b_cfg = _v17_pair(cfg_from(tiny_cfg()), 0.25)
    forward = build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=4, base_seed=0,
                                  pairing_index=pairing_index,
                                  a_mcts=a_cfg, b_mcts=b_cfg)
    reverse = build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=4, base_seed=0,
                                  pairing_index=pairing_index,
                                  a_mcts=b_cfg, b_mcts=a_cfg)
    for f, r in zip(forward, reverse):
        assert f.red_mcts == r.black_mcts
        assert f.black_mcts == r.red_mcts
        assert (f.task_id, f.seed) == (r.task_id, r.seed)


# --------------------------------------------------------------------------
# 3. The per-agent config is actually in effect, on its own side only
# --------------------------------------------------------------------------
#
# Observable: a root's total visit count tracks that side's n_simulations. This
# reads real search behaviour per ply, so it fails if the config is dropped,
# applied to the wrong side, or applied to both.

def _visits_by_player(records):
    out = {}
    for r in records:
        out.setdefault(r["player"], set()).add(r["root_total_visits"])
    return out


def test_play_eval_game_applies_each_side_its_own_config():
    cfg = tiny_cfg(max_moves=6)
    base = cfg_from(cfg)
    red_cfg = dataclasses.replace(base, n_simulations=8)
    black_cfg = dataclasses.replace(base, n_simulations=40)
    _w, _r, _n, records = play_eval_game(
        FakeEvaluator(), FakeEvaluator(), cfg, seed=5, capture=True,
        red_mcts=red_cfg, black_mcts=black_cfg)
    assert _visits_by_player(records) == {"red": {8}, "black": {40}}


def test_overriding_one_side_leaves_the_other_on_the_base_config():
    """No leakage: an override given for red must not reach black."""
    cfg = tiny_cfg(max_moves=6)
    black_cfg = dataclasses.replace(cfg_from(cfg), n_simulations=40)
    _w, _r, _n, records = play_eval_game(
        FakeEvaluator(), FakeEvaluator(), cfg, seed=5, capture=True,
        black_mcts=black_cfg)
    assert _visits_by_player(records) == {"red": {8}, "black": {40}}


def test_per_agent_config_survives_the_runner_and_follows_the_color_swap():
    """End to end through run_game_tasks: in even games A is red, in odd games A
    is black, and A's config goes with it."""
    cfg = tiny_cfg(max_moves=6)
    base = cfg_from(cfg)
    a_cfg = dataclasses.replace(base, n_simulations=8)
    b_cfg = dataclasses.replace(base, n_simulations=40)
    tasks = build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=4, base_seed=3,
                                pairing_index=0, a_mcts=a_cfg, b_mcts=b_cfg)
    with tempfile.TemporaryDirectory() as td:
        results = run_game_tasks(tasks, workers=1, config=cfg,
                                 evaluator_factory=fake_evaluator_factory,
                                 replay_dir=td, allow_differ={"n_simulations"})
        for res in results:
            with open(res.replay_path) as fh:
                records = json.load(fh)["moves"]
            a_is_red = res.game_idx % 2 == 0
            expected = ({"red": {8}, "black": {40}} if a_is_red
                        else {"red": {40}, "black": {8}})
            assert _visits_by_player(records) == expected


def test_default_path_leaves_both_sides_on_the_base_config():
    """The same observable, with no per-agent config: both sides identical."""
    cfg = tiny_cfg(max_moves=6)
    _w, _r, _n, records = play_eval_game(
        FakeEvaluator(), FakeEvaluator(), cfg, seed=5, capture=True)
    assert _visits_by_player(records) == {"red": {8}, "black": {8}}


# --------------------------------------------------------------------------
# 4. Agents must be comparable
# --------------------------------------------------------------------------

def test_consistency_accepts_a_difference_in_the_declared_field():
    base = cfg_from(tiny_cfg())
    a_cfg, b_cfg = _v17_pair(base, 0.25)
    require_agent_config_consistency(base, a_cfg, b_cfg,
                                     allow_differ={V17_FIELD})


def test_consistency_rejects_an_undeclared_difference():
    base = cfg_from(tiny_cfg())
    a_cfg, b_cfg = _v17_pair(base, 0.25)
    b_cfg = dataclasses.replace(b_cfg, n_simulations=999)
    with pytest.raises(ValueError, match="n_simulations"):
        require_agent_config_consistency(base, a_cfg, b_cfg,
                                         allow_differ={V17_FIELD})


@pytest.mark.parametrize("field,value", [
    ("eval_batch_size", 16),
    ("stall_flush_sims", 16),
    ("pending_virtual_visits", 4),
])
def test_consistency_rejects_a_batching_difference(field, value):
    """The frozen batching triple is protected without this module naming it:
    batching is simply one more field that must match the base."""
    base = cfg_from(tiny_cfg())
    bad = dataclasses.replace(base, **{field: value})
    with pytest.raises(ValueError, match=field):
        require_agent_config_consistency(base, base, bad,
                                         allow_differ={V17_FIELD})


def test_consistency_reports_every_offending_field():
    base = cfg_from(tiny_cfg())
    bad = dataclasses.replace(base, n_simulations=999, c_puct=2.0)
    with pytest.raises(ValueError) as exc:
        require_agent_config_consistency(base, None, bad)
    assert "c_puct" in str(exc.value) and "n_simulations" in str(exc.value)


def test_consistency_names_the_offending_agent():
    base = cfg_from(tiny_cfg())
    bad = dataclasses.replace(base, c_puct=2.0)
    with pytest.raises(ValueError, match="agent b"):
        require_agent_config_consistency(base, base, bad)
    with pytest.raises(ValueError, match="agent a"):
        require_agent_config_consistency(base, bad, base)


def test_consistency_rejects_an_unknown_allow_differ_field():
    """A typo in the field under study must not silently allow everything."""
    base = cfg_from(tiny_cfg())
    with pytest.raises(ValueError, match="not on MCTSConfig"):
        require_agent_config_consistency(base, base, base,
                                         allow_differ={"fpu_sipped_reduction"})


def test_consistency_rejects_a_non_config_object():
    base = cfg_from(tiny_cfg())
    with pytest.raises(TypeError, match="MCTSConfig"):
        require_agent_config_consistency(base, {"n_simulations": 8}, None)


def test_consistency_ignores_a_none_agent():
    base = cfg_from(tiny_cfg())
    require_agent_config_consistency(base, None, None)


# --------------------------------------------------------------------------
# 5. Refusal happens before any evaluator is loaded
# --------------------------------------------------------------------------

def _recording_factory(path):
    _recording_factory.calls.append(path)
    return FakeEvaluator()


_recording_factory.calls = []


def test_run_match_refuses_before_loading_any_evaluator():
    _recording_factory.calls.clear()
    base = cfg_from(tiny_cfg())
    bad = dataclasses.replace(base, n_simulations=999)
    with pytest.raises(ValueError, match="n_simulations"):
        run_match(a_ckpt=SAME_CKPT, b_ckpt=SAME_CKPT, games=2, base_seed=1,
                  config=tiny_cfg(), workers=1, output=None,
                  evaluator_factory=_recording_factory,
                  a_mcts=base, b_mcts=bad, allow_differ={V17_FIELD})
    assert _recording_factory.calls == [], "an evaluator was loaded before refusal"


def test_run_match_loads_evaluators_on_the_accepted_path():
    """Counterpart to the above: proves the empty call list means 'refused
    early', not 'this factory is never used'."""
    _recording_factory.calls.clear()
    base = cfg_from(tiny_cfg())
    a_cfg, b_cfg = _v17_pair(base, 0.25)
    run_match(a_ckpt=SAME_CKPT, b_ckpt=SAME_CKPT, games=2, base_seed=1,
              config=tiny_cfg(), workers=1, output=None,
              evaluator_factory=_recording_factory,
              a_mcts=a_cfg, b_mcts=b_cfg, allow_differ={V17_FIELD})
    assert _recording_factory.calls, "factory never called on the accepted path"


def test_same_checkpoint_is_loaded_once_despite_differing_configs():
    """_make_cache stays an NN-evaluator cache keyed by path. Two agents on one
    checkpoint share one evaluator; the search config is not part of the key
    because the cache does not hold search configuration."""
    _recording_factory.calls.clear()
    base = cfg_from(tiny_cfg())
    a_cfg, b_cfg = _v17_pair(base, 0.25)
    run_match(a_ckpt=SAME_CKPT, b_ckpt=SAME_CKPT, games=4, base_seed=1,
              config=tiny_cfg(), workers=1, output=None,
              evaluator_factory=_recording_factory,
              a_mcts=a_cfg, b_mcts=b_cfg, allow_differ={V17_FIELD})
    assert _recording_factory.calls == [SAME_CKPT]


# --------------------------------------------------------------------------
# 6. Provenance records both complete effective configs
# --------------------------------------------------------------------------

def test_summary_records_both_complete_effective_configs():
    base = cfg_from(tiny_cfg())
    a_cfg, b_cfg = _v17_pair(base, 0.25)
    summary = run_match(a_ckpt=SAME_CKPT, b_ckpt=SAME_CKPT, games=2, base_seed=1,
                        config=tiny_cfg(), workers=1, output=None,
                        evaluator_factory=fake_evaluator_factory,
                        a_mcts=a_cfg, b_mcts=b_cfg, allow_differ={V17_FIELD})
    recorded = summary["config"]["agent_mcts"]
    assert recorded["agents"]["A"] == dataclasses.asdict(a_cfg)
    assert recorded["agents"]["B"] == dataclasses.asdict(b_cfg)
    assert recorded["allow_differ"] == [V17_FIELD]
    # "Complete" means every field, not a chosen subset.
    assert (set(recorded["agents"]["A"])
            == {f.name for f in dataclasses.fields(MCTSConfig)})
    assert recorded["agents"]["A"][V17_FIELD] == 0.0
    assert recorded["agents"]["B"][V17_FIELD] == 0.25


@pytest.mark.parametrize("agent_id", ["allow_differ", "agents"])
def test_agent_named_like_provenance_metadata_keeps_its_config(agent_id):
    """Agent ids are caller-controlled and must not be able to collide with the
    metadata recorded beside them."""
    base = cfg_from(tiny_cfg())
    a_cfg, b_cfg = _v17_pair(base, 0.25)
    summary = run_match(a_ckpt=SAME_CKPT, b_ckpt=SAME_CKPT, games=2, base_seed=1,
                        config=tiny_cfg(), workers=1, output=None,
                        evaluator_factory=fake_evaluator_factory,
                        a_mcts=a_cfg, b_mcts=b_cfg,
                        a_agent=agent_id, b_agent="other",
                        allow_differ={V17_FIELD})
    recorded = summary["config"]["agent_mcts"]
    assert recorded["agents"][agent_id] == dataclasses.asdict(a_cfg)
    assert recorded["agents"]["other"] == dataclasses.asdict(b_cfg)
    assert recorded["allow_differ"] == [V17_FIELD]
    assert summary["agent_a"] == agent_id
    assert summary["a_score_rate"] is not None


def test_agent_provenance_separates_ids_from_metadata():
    """Structural: every agent id lives under `agents`, never beside it."""
    a_cfg, b_cfg = _v17_pair(cfg_from(tiny_cfg()), 0.25)
    summary = run_match(a_ckpt=SAME_CKPT, b_ckpt=SAME_CKPT, games=2, base_seed=1,
                        config=tiny_cfg(), workers=1, output=None,
                        evaluator_factory=fake_evaluator_factory,
                        a_mcts=a_cfg, b_mcts=b_cfg, allow_differ={V17_FIELD})
    recorded = summary["config"]["agent_mcts"]
    assert set(recorded) == {"agents", "allow_differ"}
    assert set(recorded["agents"]) == {"A", "B"}


@pytest.mark.parametrize("bad_id", [1, "", ("A",), 0.5])
def test_agent_ids_must_be_non_empty_strings(bad_id):
    """Ids become JSON keys; a non-string would be coerced and stop matching
    the id recorded on the task. `None` is excluded deliberately: it is the
    'not supplied' sentinel and correctly falls back to the default id."""
    a_cfg, b_cfg = _v17_pair(cfg_from(tiny_cfg()), 0.25)
    with pytest.raises(ValueError, match="non-empty string"):
        build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=2, base_seed=0,
                            pairing_index=0, a_mcts=a_cfg, b_mcts=b_cfg,
                            a_agent=bad_id, b_agent="B")


def test_omitted_agent_ids_fall_back_to_the_defaults():
    """The counterpart: None is not a bad id, it means 'use the default'."""
    a_cfg, b_cfg = _v17_pair(cfg_from(tiny_cfg()), 0.25)
    tasks = build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=2, base_seed=0,
                                pairing_index=0, a_mcts=a_cfg, b_mcts=b_cfg,
                                a_agent=None, b_agent=None)
    assert (tasks[0].red_agent, tasks[0].black_agent) == ("A", "B")


def test_recorded_agent_ids_round_trip_through_json():
    """The provenance keys must still equal the ids carried on the results."""
    a_cfg, b_cfg = _v17_pair(cfg_from(tiny_cfg()), 0.25)
    with tempfile.TemporaryDirectory() as td:
        output = os.path.join(td, "m.json")
        run_match(a_ckpt=SAME_CKPT, b_ckpt=SAME_CKPT, games=2, base_seed=1,
                  config=tiny_cfg(), workers=1, output=output,
                  evaluator_factory=fake_evaluator_factory,
                  a_mcts=a_cfg, b_mcts=b_cfg,
                  a_agent="shipped", b_agent="r0.25",
                  allow_differ={V17_FIELD})
        stem, _ = os.path.splitext(output)
        with open(output) as fh:
            written = json.load(fh)
        with open(f"{stem}_games.jsonl") as fh:
            rows = [json.loads(ln) for ln in fh]
    recorded_ids = set(written["config"]["agent_mcts"]["agents"])
    assert recorded_ids == {"shipped", "r0.25"}
    for row in rows:
        assert {row["red_agent"], row["black_agent"]} == recorded_ids


def test_agent_mode_requires_both_configs_explicitly():
    """Leaving one side unset would mean 'the base' — indistinguishable from a
    config that was dropped by mistake, so it is refused."""
    b_cfg = dataclasses.replace(cfg_from(tiny_cfg()), **{V17_FIELD: 0.25})
    with pytest.raises(ValueError, match="requires BOTH"):
        run_match(a_ckpt=SAME_CKPT, b_ckpt=SAME_CKPT, games=2, base_seed=1,
                  config=tiny_cfg(), workers=1, output=None,
                  evaluator_factory=fake_evaluator_factory,
                  b_mcts=b_cfg, allow_differ={V17_FIELD})


def test_naming_an_agent_alone_still_requires_both_configs():
    with pytest.raises(ValueError, match="requires BOTH"):
        build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=2, base_seed=0,
                            pairing_index=0, a_agent="shipped")


def test_agents_must_have_distinct_ids():
    a_cfg, b_cfg = _v17_pair(cfg_from(tiny_cfg()), 0.25)
    with pytest.raises(ValueError, match="distinct ids"):
        build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=2, base_seed=0,
                            pairing_index=0, a_mcts=a_cfg, b_mcts=b_cfg,
                            a_agent="X", b_agent="X")


def test_recorded_configs_survive_json_round_trip():
    """The summary is written as JSON; the provenance must survive that."""
    base = cfg_from(tiny_cfg())
    a_cfg, b_cfg = _v17_pair(base, 0.25)
    with tempfile.TemporaryDirectory() as td:
        output = os.path.join(td, "m.json")
        run_match(a_ckpt=SAME_CKPT, b_ckpt=SAME_CKPT, games=2, base_seed=1,
                  config=tiny_cfg(), workers=1, output=output,
                  evaluator_factory=fake_evaluator_factory,
                  a_mcts=a_cfg, b_mcts=b_cfg, allow_differ={V17_FIELD})
        with open(output) as fh:
            written = json.load(fh)
    assert (written["config"]["agent_mcts"]["agents"]["B"]
            == dataclasses.asdict(b_cfg))


# --------------------------------------------------------------------------
# 7. The v17 shape specifically
# --------------------------------------------------------------------------

def test_v17_agent_configs_carry_the_frozen_batching_triple():
    """Both per-agent configs explicitly carry (14, 48, 8).

    Checked with the v17 module's own validator against its own frozen
    constant, so this test cannot drift from the specification by restating it.
    """
    production = EvalConfig(mcts_eval_batch_size=prov.BATCHING[0],
                            mcts_stall_flush_sims=prov.BATCHING[1],
                            mcts_sims=prov.MCTS_SIMS)
    base = cfg_from(production)
    a_cfg, b_cfg = _v17_pair(base, 0.25)
    for cfg in (a_cfg, b_cfg):
        prov.validate_batching([getattr(cfg, f) for f in prov.BATCHING_FIELDS])
        assert tuple(getattr(cfg, f) for f in prov.BATCHING_FIELDS) == prov.BATCHING
    require_agent_config_consistency(base, a_cfg, b_cfg,
                                     allow_differ={V17_FIELD})


def test_v17_baseline_agent_is_the_shipped_branch():
    """The r=0 agent must be the shipped path, not an enabled mode."""
    base = cfg_from(tiny_cfg())
    a_cfg, _b = _v17_pair(base, 0.25)
    assert getattr(a_cfg, V17_FIELD) == 0.0
    assert a_cfg.fpu_value == 0.0
    assert a_cfg.fpu_policy_mass_reduction is None


def test_match_tasks_forward_per_agent_configs():
    a_cfg, b_cfg = _v17_pair(cfg_from(tiny_cfg()), 0.25)
    tasks = build_match_tasks(SAME_CKPT, SAME_CKPT, games=2, base_seed=0,
                              pairing_id="p", a_mcts=a_cfg, b_mcts=b_cfg)
    assert tasks[0].red_mcts == a_cfg and tasks[0].black_mcts == b_cfg


def test_tasks_pickle_for_spawned_workers():
    """workers>1 sends tasks to spawned processes; the new fields must pickle."""
    import pickle
    a_cfg, b_cfg = _v17_pair(cfg_from(tiny_cfg()), 0.25)
    tasks = build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=2, base_seed=0,
                                pairing_index=0, a_mcts=a_cfg, b_mcts=b_cfg)
    assert pickle.loads(pickle.dumps(tasks)) == tasks


def test_spawned_workers_apply_the_per_agent_configs():
    """Pickling the tasks is not the same as the worker USING them. _worker_main
    is a separate loop from the sequential one, so exercise it for real."""
    cfg = tiny_cfg(max_moves=6)
    base = cfg_from(cfg)
    a_cfg = dataclasses.replace(base, n_simulations=8)
    b_cfg = dataclasses.replace(base, n_simulations=40)
    tasks = build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=4, base_seed=3,
                                pairing_index=0, a_mcts=a_cfg, b_mcts=b_cfg)
    with tempfile.TemporaryDirectory() as td:
        results = run_game_tasks(tasks, workers=2, config=cfg,
                                 evaluator_factory=fake_evaluator_factory,
                                 replay_dir=td, allow_differ={"n_simulations"})
        assert len(results) == len(tasks)
        for res in results:
            with open(res.replay_path) as fh:
                records = json.load(fh)["moves"]
            expected = ({"red": {8}, "black": {40}} if res.game_idx % 2 == 0
                        else {"red": {40}, "black": {8}})
            assert _visits_by_player(records) == expected


# --------------------------------------------------------------------------
# 8. The backstop for callers that bypass run_match
# --------------------------------------------------------------------------

def test_run_game_tasks_refuses_inconsistent_agents_before_loading():
    """A caller that builds tasks itself is still checked, and still before any
    evaluator load."""
    _recording_factory.calls.clear()
    cfg = tiny_cfg()
    base = cfg_from(cfg)
    bad = dataclasses.replace(base, c_puct=9.0)
    tasks = build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=2, base_seed=0,
                                pairing_index=0, a_mcts=base, b_mcts=bad)
    with pytest.raises(ValueError, match="c_puct"):
        run_game_tasks(tasks, workers=1, config=cfg,
                       evaluator_factory=_recording_factory)
    assert _recording_factory.calls == []


def test_run_game_tasks_default_refuses_any_undeclared_variation():
    """Omitting allow_differ must not mean 'anything goes'."""
    cfg = tiny_cfg()
    a_cfg, b_cfg = _v17_pair(cfg_from(cfg), 0.25)
    tasks = build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=2, base_seed=0,
                                pairing_index=0, a_mcts=a_cfg, b_mcts=b_cfg)
    with pytest.raises(ValueError, match=V17_FIELD):
        run_game_tasks(tasks, workers=1, config=cfg,
                       evaluator_factory=fake_evaluator_factory)
    # ... and is accepted once the caller declares the field under study.
    results = run_game_tasks(tasks, workers=1, config=cfg,
                             evaluator_factory=fake_evaluator_factory,
                             allow_differ={V17_FIELD})
    assert len(results) == len(tasks)


def _agent_tasks(games=4, **kw):
    """A well-formed agent-mode task list, for tests that then break one thing."""
    base = cfg_from(tiny_cfg())
    a_cfg, b_cfg = _v17_pair(base, 0.25)
    kw.setdefault("a_mcts", a_cfg)
    kw.setdefault("b_mcts", b_cfg)
    return build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=games,
                               base_seed=0, pairing_index=0, **kw)


def _expect_refusal(tasks, pattern, allow_differ=frozenset({V17_FIELD})):
    _recording_factory.calls.clear()
    with pytest.raises(ValueError, match=pattern):
        run_game_tasks(tasks, workers=1, config=tiny_cfg(),
                       evaluator_factory=_recording_factory,
                       allow_differ=allow_differ)
    assert _recording_factory.calls == [], "refused only after loading"


def test_wellformed_agent_task_list_is_accepted():
    """The control for every refusal below: unmodified, this list runs."""
    assert len(run_game_tasks(_agent_tasks(), workers=1, config=tiny_cfg(),
                              evaluator_factory=fake_evaluator_factory,
                              allow_differ={V17_FIELD})) == 4


def test_one_unconfigured_task_is_refused():
    """The silent base-vs-base game. Both configs None on a single task made it
    skip validation entirely under the old per-task check."""
    tasks = _agent_tasks()
    tasks[2] = dataclasses.replace(tasks[2], red_mcts=None, black_mcts=None,
                                   red_agent=None, black_agent=None)
    _expect_refusal(tasks, "carry no agent fields")


def test_half_configured_task_is_refused():
    tasks = _agent_tasks()
    tasks[1] = dataclasses.replace(tasks[1], black_mcts=None)
    _expect_refusal(tasks, "half-configured")


def test_agent_config_swapped_on_one_task_is_refused():
    """Permitted configs, wrong hands: agent A keeps its id but picks up B's
    config for one game."""
    tasks = _agent_tasks()
    tasks[0] = dataclasses.replace(tasks[0], red_mcts=tasks[0].black_mcts)
    _expect_refusal(tasks, "not defined consistently")


def test_agent_checkpoint_drift_is_refused():
    tasks = _agent_tasks()
    tasks[3] = dataclasses.replace(tasks[3], red_checkpoint="other.safetensors")
    _expect_refusal(tasks, "not defined consistently")


def test_broken_color_alternation_is_refused():
    """Both agents keep their definitions; only the color assignment is wrong."""
    tasks = _agent_tasks()
    t = tasks[1]
    tasks[1] = dataclasses.replace(
        t, red_agent=t.black_agent, black_agent=t.red_agent,
        red_mcts=t.black_mcts, black_mcts=t.red_mcts)
    _expect_refusal(tasks, "exact color balance requires")


def test_same_agent_on_both_sides_is_refused():
    tasks = _agent_tasks()
    t = tasks[0]
    tasks[0] = dataclasses.replace(t, black_agent=t.red_agent,
                                   black_mcts=t.red_mcts)
    _expect_refusal(tasks, "on both sides")


def test_third_agent_in_a_pairing_is_refused():
    tasks = _agent_tasks()
    t = tasks[2]
    tasks[2] = dataclasses.replace(t, red_agent="C")
    _expect_refusal(tasks, "exactly 2 agents")


def test_consistent_but_undeclared_agent_config_is_refused():
    """All the structural checks pass; the configs themselves are the problem."""
    base = cfg_from(tiny_cfg())
    bad = dataclasses.replace(base, c_puct=9.0)
    _expect_refusal(_agent_tasks(a_mcts=base, b_mcts=bad), "c_puct",
                    allow_differ=frozenset())


def test_run_game_tasks_default_path_is_unaffected_by_the_backstop():
    """Tasks with no per-agent config skip the check entirely."""
    cfg = tiny_cfg()
    tasks = build_pairing_tasks("p", A_CKPT, B_CKPT, games=2, base_seed=0,
                                pairing_index=0)
    results = run_game_tasks(tasks, workers=1, config=cfg,
                             evaluator_factory=fake_evaluator_factory)
    assert len(results) == len(tasks)


# --------------------------------------------------------------------------
# 9. Two agents on ONE checkpoint are scored against each other
# --------------------------------------------------------------------------

def _same_ckpt_match(output=None, **kw):
    base = cfg_from(tiny_cfg())
    a_cfg, b_cfg = _v17_pair(base, 0.25)
    return run_match(a_ckpt=SAME_CKPT, b_ckpt=SAME_CKPT, games=4, base_seed=11,
                     config=tiny_cfg(), workers=1, output=output,
                     evaluator_factory=fake_evaluator_factory,
                     a_mcts=a_cfg, b_mcts=b_cfg, allow_differ={V17_FIELD}, **kw)


def test_same_checkpoint_agents_are_not_a_self_match():
    summary = _same_ckpt_match()
    assert summary["self_match"] is False
    assert summary["agent_a"] == "A" and summary["agent_b"] == "B"


def test_same_checkpoint_agents_get_real_strength_statistics():
    """The endpoint the later strength comparison depends on. Under the old
    behaviour every one of these was None."""
    summary = _same_ckpt_match()
    for key in ("a_wins", "b_wins", "a_score", "a_score_rate", "elo_estimate",
                "elo_ci95", "score_rate_ci95", "verdict", "a_as_red",
                "a_as_black"):
        assert summary[key] is not None, f"{key} is None on a same-checkpoint match"
    assert summary["a_wins"] + summary["b_wins"] <= summary["games"]


def test_identical_checkpoints_without_agents_remain_a_self_match():
    """The legacy guarantee is untouched: with no agent identity the two sides
    really are indistinguishable, and nulling the statistics stays correct."""
    summary = run_match(a_ckpt=SAME_CKPT, b_ckpt=SAME_CKPT, games=2, base_seed=1,
                        config=tiny_cfg(), workers=1, output=None,
                        evaluator_factory=fake_evaluator_factory)
    assert summary["self_match"] is True
    assert summary["a_score_rate"] is None and summary["verdict"] is None
    assert "agent_a" not in summary


def test_results_carry_winner_agent_identity():
    tasks = _agent_tasks()
    results = run_game_tasks(tasks, workers=1, config=tiny_cfg(),
                             evaluator_factory=fake_evaluator_factory,
                             allow_differ={V17_FIELD})
    for res in results:
        assert {res.red_agent, res.black_agent} == {"A", "B"}
        if res.winner == "red":
            assert res.winner_agent == res.red_agent
        elif res.winner == "black":
            assert res.winner_agent == res.black_agent
        else:
            assert res.winner_agent is None
        # Checkpoint identity alone cannot distinguish them; agent identity can.
        assert res.red_checkpoint == res.black_checkpoint


def _decided(tasks, winners):
    """Results with FORCED winners.

    The fake evaluator draws every game on this tiny board, so a match played
    here exercises only the draw branch. Agent attribution on a win is the
    whole point of the feature, so those results are built explicitly.
    """
    return [make_result(t, w, "win" if w else "state_cap", 10)
            for t, w in zip(tasks, winners)]


@pytest.mark.parametrize("winner,expected", [
    ("red", "A"), ("black", "B"), (None, None),
])
def test_make_result_attributes_the_win_to_the_right_agent(winner, expected):
    task = _agent_tasks(games=2)[0]          # game_idx 0 => A is red
    assert task.red_agent == "A" and task.black_agent == "B"
    res = make_result(task, winner, "win" if winner else "state_cap", 10)
    assert res.winner_agent == expected
    assert res.red_agent == "A" and res.black_agent == "B"


def test_make_result_follows_the_agent_across_the_color_swap():
    task = _agent_tasks(games=2)[1]          # game_idx 1 => B is red
    assert task.red_agent == "B"
    assert make_result(task, "red", "win", 10).winner_agent == "B"


def test_agent_scoring_counts_wins_the_checkpoint_path_cannot_see():
    """The core of the gap: on one checkpoint, winner_checkpoint is identical
    for every game, so checkpoint-based counting yields nothing. Agent identity
    recovers the real 3-1 split."""
    from scripts.GPU.alphazero.eval_summary import summarize_match
    tasks = _agent_tasks(games=4)
    # A is red in games 0 and 2, black in 1 and 3. Give A three wins.
    results = _decided(tasks, ["red", "red", "red", "black"])
    assert {r.winner_checkpoint for r in results} == {SAME_CKPT}

    summary = summarize_match(results, SAME_CKPT, SAME_CKPT, "p", {},
                              a_agent="A", b_agent="B")
    assert (summary["a_wins"], summary["b_wins"]) == (3, 1)
    assert summary["a_score"] == 3.0
    assert summary["a_score_rate"] == 0.75
    assert summary["self_match"] is False
    assert summary["verdict"] is not None
    # A won both of its red games and one of its two black games.
    assert summary["a_as_red"]["wins"] == 2
    assert summary["a_as_black"]["wins"] == 1
    assert summary["a_as_red"]["games"] == summary["a_as_black"]["games"] == 2


def test_agent_scoring_is_symmetric_under_relabelling():
    from scripts.GPU.alphazero.eval_summary import summarize_match
    results = _decided(_agent_tasks(games=4), ["red", "red", "red", "black"])
    forward = summarize_match(results, SAME_CKPT, SAME_CKPT, "p", {},
                              a_agent="A", b_agent="B")
    reverse = summarize_match(results, SAME_CKPT, SAME_CKPT, "p", {},
                              a_agent="B", b_agent="A")
    assert forward["a_wins"] == reverse["b_wins"] == 3
    assert forward["b_wins"] == reverse["a_wins"] == 1
    assert reverse["a_score_rate"] == 0.25


def test_agent_color_stats_split_the_two_agents():
    """a_as_red must count only A's red games, not every red game — which is
    what checkpoint matching would have done on a shared checkpoint."""
    summary = _same_ckpt_match()
    assert summary["a_as_red"]["games"] == 2
    assert summary["a_as_black"]["games"] == 2


def test_agent_scoring_refuses_results_without_identity():
    from scripts.GPU.alphazero.eval_summary import summarize_match
    tasks = build_pairing_tasks("p", A_CKPT, B_CKPT, games=2, base_seed=0,
                                pairing_index=0)
    results = run_game_tasks(tasks, workers=1, config=tiny_cfg(),
                             evaluator_factory=fake_evaluator_factory)
    with pytest.raises(ValueError, match="no agent identity"):
        summarize_match(results, A_CKPT, B_CKPT, "p", {}, a_agent="A",
                        b_agent="B")


def test_agent_scoring_requires_both_ids():
    from scripts.GPU.alphazero.eval_summary import summarize_match
    tasks = _agent_tasks(games=2)
    results = run_game_tasks(tasks, workers=1, config=tiny_cfg(),
                             evaluator_factory=fake_evaluator_factory,
                             allow_differ={V17_FIELD})
    with pytest.raises(ValueError, match="supplied together"):
        summarize_match(results, SAME_CKPT, SAME_CKPT, "p", {}, a_agent="A")


def test_legacy_games_jsonl_omits_the_agent_fields(golden):
    """Byte preservation: the new result fields must not appear at all on the
    default path."""
    with tempfile.TemporaryDirectory() as td:
        output = os.path.join(td, "match.json")
        run_match(a_ckpt=A_CKPT, b_ckpt=B_CKPT, games=4, base_seed=777,
                  config=tiny_cfg(), workers=1, output=output,
                  evaluator_factory=fake_evaluator_factory)
        stem, _ = os.path.splitext(output)
        with open(f"{stem}_games.jsonl", "rb") as fh:
            games_bytes = fh.read()
    assert games_bytes.decode() == golden["run_match"]["games_jsonl"]
    for line in games_bytes.decode().splitlines():
        row = json.loads(line)
        assert not ({"red_agent", "black_agent", "winner_agent"} & set(row))


def test_agent_mode_games_jsonl_includes_the_agent_fields():
    with tempfile.TemporaryDirectory() as td:
        output = os.path.join(td, "m.json")
        _same_ckpt_match(output=output)
        stem, _ = os.path.splitext(output)
        with open(f"{stem}_games.jsonl") as fh:
            rows = [json.loads(ln) for ln in fh]
    assert len(rows) == 4
    for row in rows:
        assert {row["red_agent"], row["black_agent"]} == {"A", "B"}
        assert "winner_agent" in row  # present even when null (a draw)


# --------------------------------------------------------------------------
# 10. The frozen batching triple, enforced at runtime
# --------------------------------------------------------------------------

def _v17_production_cfg(**kw) -> EvalConfig:
    base = dict(mcts_eval_batch_size=prov.BATCHING[0],
                mcts_stall_flush_sims=prov.BATCHING[1],
                mcts_sims=prov.MCTS_SIMS, board_size=8, max_moves=4)
    base.update(kw)
    return EvalConfig(**base)


def test_wrong_base_batching_is_refused_before_any_evaluator_loads():
    """The gap a base-relative check alone leaves open: both agents can agree
    with an EvalConfig that is itself wrong."""
    _recording_factory.calls.clear()
    wrong = _v17_production_cfg(mcts_eval_batch_size=16,
                                mcts_stall_flush_sims=16)
    base = cfg_from(wrong)
    a_cfg, b_cfg = _v17_pair(base, 0.25)
    tasks = build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=2, base_seed=0,
                                pairing_index=0, a_mcts=a_cfg, b_mcts=b_cfg)
    # Agents are mutually consistent -- the base-relative check is satisfied.
    require_agent_config_consistency(base, a_cfg, b_cfg,
                                     allow_differ={V17_FIELD})
    with pytest.raises(prov.ProtocolViolation, match="batching triple"):
        run_game_tasks(tasks, workers=1, config=wrong,
                       evaluator_factory=_recording_factory,
                       allow_differ={V17_FIELD},
                       config_validator=prov.validate_batching)
    assert _recording_factory.calls == []


def test_correct_base_batching_passes_the_validator():
    _recording_factory.calls.clear()
    good = _v17_production_cfg()
    a_cfg, b_cfg = _v17_pair(cfg_from(good), 0.25)
    tasks = build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=2, base_seed=0,
                                pairing_index=0, a_mcts=a_cfg, b_mcts=b_cfg)
    run_game_tasks(tasks, workers=1, config=good,
                   evaluator_factory=_recording_factory,
                   allow_differ={V17_FIELD},
                   config_validator=prov.validate_batching)
    assert _recording_factory.calls == [SAME_CKPT]


def test_wrong_agent_batching_is_refused_even_with_a_correct_base():
    """The agent config is validated in its own right, not only relative to a
    base that happens to be right."""
    _recording_factory.calls.clear()
    good = _v17_production_cfg()
    base = cfg_from(good)
    a_cfg, b_cfg = _v17_pair(base, 0.25)
    b_cfg = dataclasses.replace(b_cfg, pending_virtual_visits=4)
    tasks = build_pairing_tasks("p", SAME_CKPT, SAME_CKPT, games=2, base_seed=0,
                                pairing_index=0, a_mcts=a_cfg, b_mcts=b_cfg)
    with pytest.raises((ValueError, prov.ProtocolViolation)):
        run_game_tasks(tasks, workers=1, config=good,
                       evaluator_factory=_recording_factory,
                       allow_differ={V17_FIELD, "pending_virtual_visits"},
                       config_validator=prov.validate_batching)
    assert _recording_factory.calls == []


def test_validator_runs_even_with_no_agent_tasks():
    """A wrong base is wrong whether or not the run is asymmetric."""
    _recording_factory.calls.clear()
    wrong = _v17_production_cfg(mcts_eval_batch_size=16)
    tasks = build_pairing_tasks("p", A_CKPT, B_CKPT, games=2, base_seed=0,
                                pairing_index=0)
    with pytest.raises(prov.ProtocolViolation, match="batching triple"):
        run_game_tasks(tasks, workers=1, config=wrong,
                       evaluator_factory=_recording_factory,
                       config_validator=prov.validate_batching)
    assert _recording_factory.calls == []


def test_run_match_forwards_the_config_validator():
    _recording_factory.calls.clear()
    wrong = _v17_production_cfg(mcts_stall_flush_sims=16)
    a_cfg, b_cfg = _v17_pair(cfg_from(wrong), 0.25)
    with pytest.raises(prov.ProtocolViolation, match="batching triple"):
        run_match(a_ckpt=SAME_CKPT, b_ckpt=SAME_CKPT, games=2, base_seed=1,
                  config=wrong, workers=1, output=None,
                  evaluator_factory=_recording_factory,
                  a_mcts=a_cfg, b_mcts=b_cfg, allow_differ={V17_FIELD},
                  config_validator=prov.validate_batching)
    assert _recording_factory.calls == []


# --------------------------------------------------------------------------
# 11. Task 6 must not touch the search engine
# --------------------------------------------------------------------------

def test_mcts_py_is_unchanged_since_the_golden_was_captured(golden):
    """The golden records the source hashes at capture time. mcts.py was frozen
    by Tasks 2-3; Task 6 is an eval-harness change and must not have altered
    it. eval_runner.py and eval_checkpoint_match.py DID legitimately change,
    which is why only mcts.py is pinned here."""
    import hashlib
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "scripts", "GPU", "alphazero", "mcts.py")
    with open(path, "rb") as fh:
        current = hashlib.sha1(fh.read()).hexdigest()
    assert current == golden["pre_change_source_sha1"]["mcts.py"]
