"""Agent identity: colour binding, winner attribution, legacy preservation."""
import json
import pathlib

import pytest

from scripts.GPU.alphazero import eval_readout as R
from scripts.GPU.alphazero.eval_runner import (
    AGENT_COMPARISON_UNIT, AgentSpec, build_agent_pairing_tasks,
    build_pairing_tasks, make_result,
)

# Task B6's provenance hashes the checkpoint, so this must be a REAL file.
# fake_evaluator_factory still ignores its contents.
CKPT = str(pathlib.Path(__file__).parent / "eval_fakes.py")
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


# --- Task B6: two-agent match CLI ------------------------------------------

from scripts.GPU.alphazero.eval_readout_match import (  # noqa: E402
    readout_config_from_name, run_readout_match,
)
from scripts.GPU.alphazero.eval_runner import EvalConfig  # noqa: E402
from tests.eval_fakes import fake_evaluator_factory  # noqa: E402

TINY = EvalConfig(board_size=8, mcts_sims=16, mcts_eval_batch_size=4,
                  mcts_stall_flush_sims=4, max_moves=16, opening_temp_plies=2)


def _stub_provenance(monkeypatch):
    """TEST-ONLY: satisfy provenance so match MECHANICS can be exercised while
    the working tree is legitimately dirty during development.

    This is a reach into the module, NOT a parameter -- no production caller
    can aim provenance elsewhere or skip it.
    """
    from scripts.GPU.alphazero import eval_readout_match as M
    monkeypatch.setattr(M, "_git_provenance",
                        lambda _d: {"git_commit": "0" * 40,
                                    "worktree_clean": True})


def test_readout_names_map_to_the_frozen_configs():
    control = readout_config_from_name("tournament", 20, 1.0, 0.1)
    assert control.mode == R.MODE_OPENING_TEMPERATURE and control.temp_low == 0.1

    c2_control = readout_config_from_name("opening_then_argmax", 20, 1.0, 0.1)
    assert c2_control.mode == R.MODE_OPENING_TEMPERATURE
    assert c2_control.temp_low == 0.0

    assert readout_config_from_name("argmax", 20, 1.0, 0.1).mode == R.MODE_ARGMAX
    assert readout_config_from_name(
        "hoeffding_lcb", 20, 1.0, 0.1).mode == R.MODE_HOEFFDING_LCB


def test_unknown_readout_name_is_rejected():
    with pytest.raises(ValueError):
        readout_config_from_name("wishful", 20, 1.0, 0.1)


def test_identical_readouts_are_refused():
    from scripts.GPU.alphazero import eval_readout_match as M
    with pytest.raises(ValueError, match="identical"):
        M.run_readout_match(
            checkpoint=CKPT,
            candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
            control_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
            games=2, base_seed=900, config=TINY, workers=1, output=None,
            evaluator_factory=fake_evaluator_factory,
        )


def test_provenance_anchors_to_the_executing_source_repository():
    """The anchor must be THIS repository, not the process CWD and not a
    caller-supplied path. Otherwise a run could execute dirty engine code
    while recording a pristine unrelated repository.
    """
    import inspect
    from scripts.GPU.alphazero import eval_readout_match as M

    anchor = pathlib.Path(M._source_repo_dir()).resolve()
    module_dir = pathlib.Path(M.__file__).resolve().parent
    assert anchor == module_dir

    params = inspect.signature(M.run_readout_match).parameters
    assert "repo_dir" not in params and "allow_dirty" not in params


def test_provenance_runs_before_any_game(monkeypatch):
    """CONSTRUCTED: make provenance fail and assert no game was played."""
    from scripts.GPU.alphazero import eval_readout_match as M

    def _boom(_repo_dir):
        raise RuntimeError("worktree is dirty")

    calls = []

    def _counting_factory(path):
        calls.append(path)
        return fake_evaluator_factory(path)

    monkeypatch.setattr(M, "_git_provenance", _boom)
    with pytest.raises(RuntimeError, match="dirty"):
        M.run_readout_match(
            checkpoint=CKPT,
            candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
            control_readout=readout_config_from_name("tournament", 2, 1.0, 0.1),
            games=2, base_seed=902, config=TINY, workers=1, output=None,
            evaluator_factory=_counting_factory,
        )
    assert calls == [], "a game ran despite failing provenance"


def test_run_readout_match_produces_a_real_score_rate(monkeypatch, tmp_path):
    from scripts.GPU.alphazero import eval_readout_match as M
    _stub_provenance(monkeypatch)
    out = tmp_path / "m.json"
    summary = M.run_readout_match(
        checkpoint=CKPT,
        candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
        control_readout=readout_config_from_name("tournament", 2, 1.0, 0.1),
        games=4, base_seed=900, config=TINY, workers=1, output=str(out),
        evaluator_factory=fake_evaluator_factory,
    )
    # The whole point of agent identity: this is NOT None.
    assert summary["a_score_rate"] is not None
    assert summary["score_rate_ci95"] is not None
    assert summary["comparison_unit"] == "agent"
    assert summary["same_checkpoint"] is True
    assert json.loads(out.read_text())["agent_a"] == "candidate"


def test_run_readout_match_writes_per_game_rows_with_agent_ids(monkeypatch, tmp_path):
    from scripts.GPU.alphazero import eval_readout_match as M
    _stub_provenance(monkeypatch)
    out = tmp_path / "m.json"
    M.run_readout_match(
        checkpoint=CKPT,
        candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
        control_readout=readout_config_from_name("tournament", 2, 1.0, 0.1),
        games=4, base_seed=901, config=TINY, workers=1, output=str(out),
        evaluator_factory=fake_evaluator_factory,
    )
    rows = [json.loads(line) for line in
            (tmp_path / "m_games.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert {r["red_agent_id"] for r in rows} == {"candidate", "control"}
    assert all(r["comparison_unit"] == "agent" for r in rows)


def test_summary_records_the_full_provenance_block(monkeypatch, tmp_path):
    from scripts.GPU.alphazero import eval_readout_match as M
    _stub_provenance(monkeypatch)
    out = tmp_path / "m.json"
    s = M.run_readout_match(
        checkpoint=CKPT,
        candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
        control_readout=readout_config_from_name("tournament", 2, 1.0, 0.1),
        games=2, base_seed=902, config=TINY, workers=1, output=str(out),
        evaluator_factory=fake_evaluator_factory,
        prior_seed_intervals=[[100, 200]],
    )
    assert s["git_commit"]                       # never None
    assert s["worktree_clean"] is True
    assert s["config"]["checkpoint_sha1"]        # never None
    assert s["wall_clock_seconds"] >= 0
    assert s["config"]["seed_interval"] == [902, 904]
    assert s["config"]["seed_interval_convention"] == "half_open_[start,end)"
    assert s["config"]["prior_seed_intervals"] == [[100, 200]]
    assert set(s["config"]["rng_derivation"]) == {
        "search_red", "search_black", "readout_red", "readout_black",
        "game_seed"}


def test_prior_seed_intervals_default_to_empty_not_none(monkeypatch):
    from scripts.GPU.alphazero import eval_readout_match as M
    _stub_provenance(monkeypatch)
    s = M.run_readout_match(
        checkpoint=CKPT,
        candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
        control_readout=readout_config_from_name("tournament", 2, 1.0, 0.1),
        games=2, base_seed=905, config=TINY, workers=1, output=None,
        evaluator_factory=fake_evaluator_factory,
    )
    assert s["config"]["prior_seed_intervals"] == []


def test_overlapping_seed_interval_is_refused_before_any_game(monkeypatch):
    """Closes the B6/B7 boundary. The overlap must be refused BEFORE any game,
    so the evaluator factory is never called."""
    from scripts.GPU.alphazero import eval_readout_match as M
    _stub_provenance(monkeypatch)
    calls = []

    def _counting_factory(path):
        calls.append(path)
        return fake_evaluator_factory(path)

    with pytest.raises(ValueError, match="overlap"):
        M.run_readout_match(
            checkpoint=CKPT,
            candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
            control_readout=readout_config_from_name("tournament", 2, 1.0, 0.1),
            games=4, base_seed=1000, config=TINY, workers=1, output=None,
            evaluator_factory=_counting_factory,
            prior_seed_intervals=[[1002, 1010]],      # overlaps [1000, 1004)
        )
    assert calls == [], "a game ran despite an overlapping seed interval"


def test_disjoint_prior_intervals_are_accepted(monkeypatch):
    from scripts.GPU.alphazero import eval_readout_match as M
    _stub_provenance(monkeypatch)
    s = M.run_readout_match(
        checkpoint=CKPT,
        candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
        control_readout=readout_config_from_name("tournament", 2, 1.0, 0.1),
        games=4, base_seed=1000, config=TINY, workers=1, output=None,
        evaluator_factory=fake_evaluator_factory,
        prior_seed_intervals=[[0, 64], [64, 128]],   # adjacent, not overlapping
    )
    assert s["config"]["prior_seed_intervals"] == [[0, 64], [64, 128]]


def test_an_unreadable_checkpoint_aborts_instead_of_recording_a_null_hash(
        monkeypatch, tmp_path):
    from scripts.GPU.alphazero import eval_readout_match as M
    _stub_provenance(monkeypatch)
    with pytest.raises(RuntimeError, match="hash checkpoint"):
        M.run_readout_match(
            checkpoint=str(tmp_path / "does_not_exist.safetensors"),
            candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
            control_readout=readout_config_from_name("tournament", 2, 1.0, 0.1),
            games=2, base_seed=903, config=TINY, workers=1, output=None,
            evaluator_factory=fake_evaluator_factory,
        )
