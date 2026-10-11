# A0 — Per-agent search budgets in the evaluation harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status: plan only. Implementing it, running its tests and committing anything each need the user's explicit approval.**

**Goal:** Let each agent in an evaluation match search with its own simulation budget. The budget is bound to the agent across the colour swap, checked per side at every ply, and recorded per colour in each result. Legacy behaviour stays unchanged.

**Architecture:** `AgentSpec` gains an optional `mcts_sims`, where `None` means the run's `EvalConfig.mcts_sims`. `cfg_from` gains an optional simulation override. `play_eval_game` builds each side's search from that side's budget and validates every ply against it. `_play_and_build_result` resolves each colour's budget from the task's agents. Agent results record the effective budgets, and `validate_game_binding` refuses a recorded budget that differs from an agent's explicit one.

**Tech Stack:** Python 3.14 under `.venv`; pytest; `tests/eval_fakes.FakeEvaluator` (CPU, no model).

**Spec:** `docs/superpowers/2026-10-11-search-distillation-proposal.md` (§4 harness audit, §5 step A0), committed at `103a9222`.

## Global Constraints

- Scope is limited to: per-agent simulation budgets, correct colour binding, per-side integrity checks, and recorded budgets.
- Deferred: full-visit export, worker changes or worker tests, training, and any runner or CLI for Stage A.
- Legacy defaults stay unchanged. With no agent budgets, a game is byte-identical to the game before A0 (pinned digest, Task 1).
- New fixtures run on the official 24×24 board at the real 280 cap. Only the simulation counts are small.
- Write no nine-digit `2026…` seed-band literals into `scripts/`, `tests/` or `docs/superpowers/`. Check with the pinned scanner `docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py` `S.scan`.
- The only permitted executions are **CPU games with the fake evaluator**: `tests/eval_fakes.FakeEvaluator` driving the real rules engine and the real search, on 24×24 at the 280 cap. No model loading, no 8×8 games, no worker execution, no A1. A1 needs its own approval.
- **Execution limits (user, 2026-10-11).** The Task 1 pre-change baseline command runs once, under `timeout 30`. `tests/test_eval_agent_budget.py` runs at most twice per task (the red and green runs), each under `timeout 60` with `-q -p no:cacheprovider`. Stop on a timeout, an unexpected failure or a legacy-digest mismatch, and report it without replacing the pinned digest or weakening any assertion. The run stops for review after each task, with no commit.
- **Result-row fields accepted (user, 2026-10-11).** The two nullable budget fields appear on every result row, legacy rows included. Loading old rows must stay compatible. The legacy digest checks game behaviour, not result-row bytes.
- No commits unless the user asks. The uncommitted capture work (`self_play.py` and the capture files) is not touched and not staged.

## Review Focus

1. **Legacy JSONL rows gain two `null` keys** (`red_mcts_sims`, `black_mcts_sims`), as the agent fields did before them. A consumer that pins row bytes or digests would notice. Old rows without the keys must still load through `EvalGameResult(**row)`; Task 2 tests that. Task 2 Step 5 also runs two data-only tests from `tests/test_fpu_dev_reservoir_protocol.py`, the known `EvalGameResult(**row)` consumer.
2. **A budget dropped on one side only.** If the override is lost for one colour, the run must stop on that colour's first ply, not pass silently. Task 1 tests each side separately.
3. **Malformed budgets** (`0`, negative, `float`, `bool`, `str`) must be refused when the `AgentSpec` is built, not at the first search. Tested in Task 1.
4. **Match summaries describe one budget.** `eval_checkpoint_match` and `eval_readout_match` summaries record only `config.mcts_sims`, so for an asymmetric match the per-game result rows are the authoritative budget record. This is not changed here; a Stage A runner must record per-agent budgets in its own summary.
5. **Workers > 1.** Budgets reach workers inside the pickled task, and workers call the same `_play_and_build_result`, so the code path is shared. No worker test is added or run; worker qualification is deferred.

---

### Task 1: Per-side budgets in `play_eval_game`

**Files:**
- Modify: `scripts/GPU/alphazero/eval_runner.py` — `AgentSpec` (around line 37), a new `agent_sims` beside it, `cfg_from` (around line 111), `play_eval_game` (around line 160)
- Test (create): `tests/test_eval_agent_budget.py`

**Interfaces:**
- Produces:
  - `AgentSpec(agent_id: str, checkpoint: str, readout: ReadoutConfig, mcts_sims: Optional[int] = None)`
  - `agent_sims(agent: Optional[AgentSpec], config: EvalConfig) -> int`
  - `cfg_from(config: EvalConfig, n_simulations: Optional[int] = None) -> MCTSConfig`
  - `play_eval_game(red_eval, black_eval, config, seed, capture=False, red_readout=None, black_readout=None, red_sims: Optional[int] = None, black_sims: Optional[int] = None)`

- [ ] **Step 1: Pin the pre-A0 legacy game digest (run on the unmodified code)**

Run from the repository root:

```bash
timeout 30 env PYTHONPATH=. .venv/bin/python -c "
import hashlib, json
from scripts.GPU.alphazero.eval_runner import EvalConfig, play_eval_game
from tests.eval_fakes import FakeEvaluator
cfg = EvalConfig(board_size=24, max_moves=280, mcts_sims=4, selection_mode='argmax')
out = play_eval_game(FakeEvaluator(), FakeEvaluator(), cfg, seed=5, capture=True)
print(out[1], out[2], hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest())"
```

Run it **once**. Expected: a reason, a ply count and a 64-hex digest. Record all three in the step's notes; the digest becomes `LEGACY_DIGEST` in Step 2. On a timeout, stop and report: the fixture cost is then too high for a focused test.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_eval_agent_budget.py`, replacing `PASTE_STEP_1_DIGEST` with the digest from Step 1:

```python
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
from scripts.GPU.alphazero.eval_integrity import IntegrityError
from scripts.GPU.alphazero.eval_runner import (
    AgentSpec, EvalConfig, agent_sims, cfg_from, play_eval_game,
)
from tests.eval_fakes import FakeEvaluator

CKPT = "fake.safetensors"
ARGMAX = R.ReadoutConfig(mode=R.MODE_ARGMAX)
CFG = EvalConfig(board_size=24, max_moves=280, mcts_sims=4, selection_mode="argmax")
LOW = AgentSpec("low", CKPT, ARGMAX, mcts_sims=3)
HIGH = AgentSpec("high", CKPT, ARGMAX, mcts_sims=7)
INHERIT = AgentSpec("inherit", CKPT, ARGMAX)

#: sha256 of json.dumps(play_eval_game(..., CFG, seed=5, capture=True), sort_keys=True)
#: computed on the code BEFORE A0 (plan Task 1, Step 1).
LEGACY_DIGEST = "PASTE_STEP_1_DIGEST"


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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run (red run, 1 of at most 2): `timeout 60 env PYTHONPATH=. .venv/bin/python -m pytest tests/test_eval_agent_budget.py -q -p no:cacheprovider`
Expected: collection fails with `ImportError: cannot import name 'agent_sims'`.

- [ ] **Step 4: Implement**

In `scripts/GPU/alphazero/eval_runner.py`, replace `AgentSpec` and add `agent_sims` directly after it:

```python
@dataclass(frozen=True)
class AgentSpec:
    """One competitor. `agent_id` is the experimental identity and is
    independent of `checkpoint` -- two agents may share a checkpoint and
    differ only in readout or search budget.

    `mcts_sims` None means the run's EvalConfig.mcts_sims (legacy behaviour).
    Like the readout, the budget travels WITH the agent across the colour swap.
    """
    agent_id: str
    checkpoint: str
    readout: ReadoutConfig
    mcts_sims: Optional[int] = None

    def __post_init__(self):
        s = self.mcts_sims
        if s is not None and (type(s) is not int or s < 1):
            raise ValueError(
                f"agent {self.agent_id!r}: mcts_sims must be a positive int "
                f"or None, got {s!r}")


def agent_sims(agent: Optional[AgentSpec], config: EvalConfig) -> int:
    """The simulation budget `agent` searches with under `config`."""
    if agent is None or agent.mcts_sims is None:
        return config.mcts_sims
    return agent.mcts_sims
```

`agent_sims` refers to `EvalConfig`, which is defined further down the module. That is fine because the annotation is a string under `from __future__ import annotations`.

Replace the signature and the `n_simulations=` line of `cfg_from`:

```python
def cfg_from(config: EvalConfig, n_simulations: Optional[int] = None) -> MCTSConfig:
    """Map EvalConfig -> MCTSConfig. argmax mode zeroes temps to hit
    select_move's deterministic argmax branch. `n_simulations` overrides
    config.mcts_sims for one side; None keeps the run budget."""
```

```python
        n_simulations=(config.mcts_sims if n_simulations is None
                       else n_simulations),
```

In `play_eval_game`, add the two keyword parameters after `black_readout`:

```python
                   black_readout: Optional[ReadoutConfig] = None,
                   red_sims: Optional[int] = None,
                   black_sims: Optional[int] = None):
```

Add this paragraph to its docstring, after the `red_readout`/`black_readout` paragraph:

```
    `red_sims`/`black_sims` set each side's search budget; None keeps
    config.mcts_sims. Every ply is validated against the MOVER's budget.
```

Replace the two `MCTS(...)` lines:

```python
    red_n = config.mcts_sims if red_sims is None else red_sims
    black_n = config.mcts_sims if black_sims is None else black_sims
    mcts_red = MCTS(red_eval, cfg_from(config, red_n), random.Random(seed ^ 0xA5A5A5))
    mcts_black = MCTS(black_eval, cfg_from(config, black_n), random.Random(seed ^ 0x5A5A5A))
```

Replace the `validate_ply` line:

```python
        validate_ply(ply, red_n if is_red else black_n, root.visit_count,
                     root_value, top2)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run (green run, 2 of at most 2): `timeout 60 env PYTHONPATH=. .venv/bin/python -m pytest tests/test_eval_agent_budget.py -q -p no:cacheprovider`
Expected: 11 passed (the parametrized refusal counts as 5). Record the wall time.

- [ ] **Step 6: Stop for review.** Report the results and the diff. Do not commit.

> **Task 1 result (2026-10-11).**
> * Baseline (once, pre-change, `timeout 30`): `state_cap`, 280 plies, 1.07 s, digest `9f30fbce…27afc`, pinned as `LEGACY_DIGEST`.
> * Red run: collection `ImportError` on `agent_sims`, exit 2, as expected.
> * Green run: 11 passed in 3.90 s, exit 0, legacy digest matched.
> * Seed-literal scan: `{}`. Two of two runs used. Nothing committed.

---

### Task 2: Colour binding and recorded budgets

**Files:**
- Modify: `scripts/GPU/alphazero/eval_runner.py` — `EvalGameResult` (around line 60), `make_result` (around line 214), `_play_and_build_result` (around line 371)
- Modify: `scripts/GPU/alphazero/eval_integrity.py` — `validate_game_binding` (around line 85)
- Test (extend): `tests/test_eval_agent_budget.py`

**Interfaces:**
- Consumes (Task 1): `AgentSpec.mcts_sims`, `agent_sims`, `play_eval_game(..., red_sims=, black_sims=)`
- Produces:
  - `EvalGameResult.red_mcts_sims: Optional[int] = None` and `EvalGameResult.black_mcts_sims: Optional[int] = None`. These hold the EFFECTIVE budgets on agent results and stay `None` on legacy results.
  - `make_result(task, winner, reason, n_moves, replay_path=None, config: Optional[EvalConfig] = None)`
  - `validate_game_binding(result, task)`, which now also raises `IntegrityError("... search budget mismatch ...")`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_agent_budget.py`, and add `build_agent_pairing_tasks`, `build_pairing_tasks` and `make_result` to its `eval_runner` import, plus `from scripts.GPU.alphazero.eval_integrity import validate_game_binding`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run (red run, 1 of at most 2): `timeout 60 env PYTHONPATH=. .venv/bin/python -m pytest tests/test_eval_agent_budget.py -q -p no:cacheprovider`
Expected: **5 failed, 12 passed** (user correction, 2026-10-11). `test_the_binding_check_asks_no_budget_of_agents_without_one` already passes before Task 2, and the 11 Task 1 tests still pass. The five budget tests fail with `AttributeError: 'EvalGameResult' object has no attribute 'red_mcts_sims'`, a `KeyError` on the deleted keys, or a visit-set mismatch, because `_play_and_build_result` does not pass budgets yet.

- [ ] **Step 3: Implement**

At the end of `EvalGameResult`, after `comparison_unit`:

```python
    # Each colour's EFFECTIVE search budget (agent results only). None on
    # legacy results; per-game rows are the authoritative budget record.
    red_mcts_sims: Optional[int] = None
    black_mcts_sims: Optional[int] = None
```

In `make_result`, change the signature to
`def make_result(task: EvalGameTask, winner, reason, n_moves, replay_path=None, config: Optional[EvalConfig] = None) -> EvalGameResult:`.
Inside the existing `agent_fields = {...}` branch, after the dict literal and still inside that branch, add:

```python
        if config is not None:
            agent_fields["red_mcts_sims"] = agent_sims(task.red_agent, config)
            agent_fields["black_mcts_sims"] = agent_sims(task.black_agent, config)
```

In `_play_and_build_result`, replace the `play_eval_game(...)` call and the `make_result` line:

```python
    winner, reason, nm, records = play_eval_game(
        red, black, config, task.seed, capture=capture,
        red_readout=red_rd, black_readout=black_rd,
        red_sims=agent_sims(task.red_agent, config),
        black_sims=agent_sims(task.black_agent, config))
    result = make_result(task, winner, reason, nm, config=config)
```

In `eval_integrity.validate_game_binding`, inside the per-colour `for` loop, after the readout check, add:

```python
        want = spec.mcts_sims
        got = getattr(result, f"{colour}_mcts_sims", None)
        if want is not None and got != want:
            raise IntegrityError(
                f"task {result.task_id}: search budget mismatch for {colour} "
                f"agent {agent_id!r} -- recorded {got!r}, task specifies {want}")
```

- [ ] **Step 4: Run the new tests**

Run (green run, 2 of at most 2): `timeout 60 env PYTHONPATH=. .venv/bin/python -m pytest tests/test_eval_agent_budget.py -q -p no:cacheprovider`
Expected: 17 passed.

- [ ] **Step 5: Run the data-only compatibility checks** (once, under `timeout 60`)

These existing tests build tasks and results and check binding, summaries and the loading of old-shaped rows. None of them plays a game. The reservoir tests rebuild `EvalGameResult(**row)` from fabricated rows that lack the new keys. 8×8 games and worker execution are excluded: `tests/test_eval_runner.py` and `tests/test_eval_runner_parallel.py` are not run, and worker qualification stays deferred.

```bash
timeout 60 env PYTHONPATH=. .venv/bin/python -m pytest -q -p no:cacheprovider \
  "tests/test_eval_agent_identity.py::test_winner_agent_id_follows_the_colour_that_won" \
  "tests/test_eval_agent_identity.py::test_draws_leave_winner_agent_id_null_not_empty" \
  "tests/test_eval_agent_identity.py::test_agent_results_are_labelled_as_agent_comparisons" \
  "tests/test_eval_agent_identity.py::test_results_carry_the_COMPLETE_readout_config_not_just_the_mode" \
  "tests/test_eval_agent_identity.py::test_different_checkpoints_are_reported_as_not_same_checkpoint" \
  "tests/test_eval_agent_identity.py::test_legacy_checkpoint_tasks_carry_no_agent_fields" \
  "tests/test_eval_integrity.py::test_game_binding_accepts_a_correct_result" \
  "tests/test_eval_integrity.py::test_game_binding_is_validated_immediately_not_at_the_end" \
  "tests/test_eval_integrity.py::test_game_binding_rejects_unknown_error_on_its_own" \
  "tests/test_eval_integrity.py::test_game_binding_rejects_a_swapped_agent_id" \
  "tests/test_eval_integrity.py::test_game_binding_rejects_one_agent_holding_both_colours" \
  "tests/test_eval_integrity.py::test_game_binding_leaves_LEGACY_results_alone" \
  "tests/test_eval_integrity.py::test_valid_result_set_passes" \
  "tests/test_eval_integrity.py::test_a_SYSTEMATIC_config_leak_is_caught" \
  "tests/test_eval_integrity.py::test_colour_imbalance_raises" \
  "tests/test_eval_summary_agent_mode.py::test_agent_summary_rejects_legacy_checkpoint_results" \
  "tests/test_eval_summary_agent_mode.py::test_legacy_summary_still_works_for_checkpoint_results" \
  "tests/test_fpu_dev_reservoir_protocol.py::test_summarize_match_produces_no_generated_at_or_git_commit_keys" \
  "tests/test_fpu_dev_reservoir_protocol.py::test_check_summary_binding_ok_on_faithful_reservoir"
```

Expected: 19 passed. Any failure stops the run and is reported. None is fixed by editing an existing test.

- [ ] **Step 6: Seed-literal scan of the changed and new files**

```bash
PYTHONPATH=. .venv/bin/python -c "
import importlib.util as u
s = u.spec_from_file_location('S', 'docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py')
S = u.module_from_spec(s); s.loader.exec_module(S)
for p in ('scripts/GPU/alphazero/eval_runner.py', 'scripts/GPU/alphazero/eval_integrity.py',
          'tests/test_eval_agent_budget.py', 'docs/superpowers/plans/2026-10-11-a0-per-agent-search-budget.md'):
    print(p, S.scan(open(p).read()))"
```

Expected: `{}` for every file.

- [ ] **Step 7: Stop for review.** Report the counts, timings and the diff. Do not commit, and do not stage the capture files.

> **Task 2 result (2026-10-11).**
> * Red run: 5 failed, 12 passed in 6.43 s, exit 1, as corrected.
> * Green run: 17 passed in 7.52 s (wall 12.85 s), exit 0, legacy digest still matching.
> * 19 data-only checks: 19 passed in 0.11 s, exit 0.
> * Seed-literal scan: `{}` for all four files. Nothing committed.

---

## After A0 (not part of this plan)

- **A1:** one colour-reversed pair, the incumbent at 1,600 against the incumbent at 400. This is a planning measurement only. A runtime cap with headroom is frozen afterwards. It needs separate approval for model loading and games.
- A Stage A runner, with per-agent budgets in its summary, fresh seed accounting and a frozen card, is separate work.
