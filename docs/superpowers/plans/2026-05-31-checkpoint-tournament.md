# Checkpoint Tournament System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone two-model checkpoint evaluator (match + tournament) that estimates relative strength via balanced, exploration-free games, for training-plateau detection.

**Architecture:** A shared flat task-queue + single global worker pool (`eval_runner.py`) executes A-vs-B games with two in-process `LocalGPUEvaluator`s. `eval_checkpoint_match.py` builds one pairing's tasks; `eval_checkpoint_tournament.py` builds all pairings into one flat list; both call the same `run_game_tasks`. Pure stats (`eval_elo.py`) and aggregation (`eval_summary.py`) are GPU-free and unit-tested. Zero changes to training code (`self_play.py`, `mcts.py`).

**Tech Stack:** Python 3.14, MLX (Apple GPU, via existing `LocalGPUEvaluator`), `multiprocessing` spawn context, pytest. Reuses `probe_eval.load_network_for_scoring`, `MCTS`/`MCTSConfig`, `TwixtState`.

**Reference spec:** `docs/superpowers/specs/2026-05-31-checkpoint-tournament-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `scripts/GPU/alphazero/eval_elo.py` | Pure stats: score-rate, clamped Elo, draw-aware trinomial CI, verdict. No game/MLX deps. |
| `scripts/GPU/alphazero/eval_runner.py` | Dataclasses (`EvalGameTask`/`EvalGameResult`/`EvalConfig`), `cfg_from`, `play_eval_game`, `make_result`, `build_pairing_tasks`, `run_game_tasks` (workers=1 + spawn pool), per-worker checkpoint cache, default real-evaluator factory. |
| `scripts/GPU/alphazero/eval_summary.py` | Aggregate `list[EvalGameResult]` → match/tournament summary dicts (by-color, color-bias, Elo, CI, verdict). Pure (no MLX/time/git). |
| `scripts/GPU/alphazero/eval_checkpoint_match.py` | One-pairing task builder + argparse CLI + writes per-game JSONL + summary JSON. |
| `scripts/GPU/alphazero/eval_checkpoint_tournament.py` | Multi-pairing flat task builder + argparse CLI + groups by `pairing_id` + combined output + printed table. |
| `tests/eval_fakes.py` | Module-level `FakeEvaluator` + picklable `fake_evaluator_factory` / counting factory for spawn-safe tests (NOT collected by pytest). |
| `tests/test_eval_elo.py` | Stats unit tests. |
| `tests/test_eval_runner.py` | Determinism, workers=1, workers=1-vs-2 equivalence, cache, bookkeeping. |
| `tests/test_eval_builders.py` | Match + tournament task builders: balance, seed stability/independence, global uniqueness. |
| `tests/test_eval_summary.py` | Aggregation correctness on synthetic results. |
| `tests/test_eval_cli.py` | Fake 2-game end-to-end match smoke + tournament "single `run_game_tasks` call" test. |
| `tests/test_eval_real_smoke.py` | `@pytest.mark.integration` real-checkpoint smokes (0419-vs-0419 sanity, first match). |

**Determinism rule (used throughout):** `seed = base_seed + pairing_index*1_000_000 + game_idx`; `task_id = pairing_index*1_000_000 + game_idx`. Same schedule + same `base_seed` ⇒ same games regardless of `--workers`.

---

## Task 1: `eval_elo.py` — pure statistics

**Files:**
- Create: `scripts/GPU/alphazero/eval_elo.py`
- Test: `tests/test_eval_elo.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_elo.py`:

```python
import math
import pytest

from scripts.GPU.alphazero.eval_elo import (
    score_rate, elo_diff, score_ci_trinomial, elo_ci, verdict,
)


def test_score_rate_counts_draws_half():
    # 6 wins, 2 draws, 2 losses out of 10 -> (6 + 1)/10 = 0.7
    assert score_rate(6, 2, 10) == pytest.approx(0.7)


def test_score_rate_rejects_zero_total():
    with pytest.raises(ValueError):
        score_rate(0, 0, 0)


def test_elo_diff_60pct_is_about_plus_70():
    assert elo_diff(0.6, 400) == pytest.approx(70.4, abs=1.0)


def test_elo_diff_is_antisymmetric():
    # A scoring p and B scoring 1-p must give opposite Elo.
    assert elo_diff(0.6, 400) == pytest.approx(-elo_diff(0.4, 400), abs=1e-9)


def test_elo_diff_clamps_clean_sweep_to_finite():
    # p == 1.0 must not be +inf; clamp at 1 - 1/(2N).
    val = elo_diff(1.0, 400)
    assert math.isfinite(val)
    assert val == pytest.approx(elo_diff(1.0 - 1.0 / 800, 400))


def test_score_ci_trinomial_brackets_mean():
    lo, hi = score_ci_trinomial(223, 8, 169)
    m = (223 + 0.5 * 8) / 400
    assert lo < m < hi
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_elo_ci_endpoints_ordered():
    lo, hi = elo_ci(223, 8, 169)
    assert lo < hi


def test_verdict_thresholds():
    assert verdict(0.60) == "stronger"
    assert verdict(0.55) == "stronger"
    assert verdict(0.53) == "weak_signal"
    assert verdict(0.50) == "tied"
    assert verdict(0.40) == "worse"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_elo.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.GPU.alphazero.eval_elo'`

- [ ] **Step 3: Write the implementation**

Create `scripts/GPU/alphazero/eval_elo.py`:

```python
"""Pure statistics for checkpoint-tournament results.

No game-engine or MLX dependencies — unit-testable in isolation. Scoring
counts a decisive win as 1 and a draw (state-cap or board-full) as 0.5.
"""
from __future__ import annotations

import math


def score_rate(wins: float, draws_plus_caps: float, total: int) -> float:
    """Score rate with draws/caps counting half. total must be > 0."""
    if total <= 0:
        raise ValueError("total must be > 0")
    return (wins + 0.5 * draws_plus_caps) / total


def _clamp_p(p: float, n: int) -> float:
    """Clamp a score rate away from {0, 1} so Elo stays finite.

    Bound is 1/(2N): a clean sweep maps to a large-but-finite Elo.
    """
    lo = 1.0 / (2 * n)
    hi = 1.0 - lo
    return min(max(p, lo), hi)


def elo_diff(p: float, n: int) -> float:
    """Elo difference implied by score rate p over n games (clamped)."""
    p = _clamp_p(p, n)
    return 400.0 * math.log10(p / (1.0 - p))


def score_ci_trinomial(w: int, d: int, l: int, z: float = 1.96) -> tuple[float, float]:
    """Draw-aware 95% CI on the score rate.

    Outcomes are {0, 0.5, 1}, so a Bernoulli/Wilson interval is the wrong
    model. Uses the trinomial score variance:
        var = [w(1-m)^2 + d(0.5-m)^2 + l(0-m)^2] / N,  SE = sqrt(var/N).
    w = wins, d = draws+caps, l = losses.
    """
    n = w + d + l
    if n <= 0:
        raise ValueError("no games")
    m = (w + 0.5 * d) / n
    var = (w * (1 - m) ** 2 + d * (0.5 - m) ** 2 + l * (0.0 - m) ** 2) / n
    se = math.sqrt(var / n)
    lo = max(0.0, m - z * se)
    hi = min(1.0, m + z * se)
    return lo, hi


def elo_ci(w: int, d: int, l: int, z: float = 1.96) -> tuple[float, float]:
    """95% Elo CI: trinomial score-rate endpoints mapped through elo_diff."""
    n = w + d + l
    lo, hi = score_ci_trinomial(w, d, l, z)
    return elo_diff(lo, n), elo_diff(hi, n)


def verdict(rate: float) -> str:
    """Strength verdict from score rate (spec thresholds)."""
    if rate >= 0.55:
        return "stronger"
    if rate >= 0.52:
        return "weak_signal"
    if rate >= 0.48:
        return "tied"
    return "worse"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_elo.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_elo.py tests/test_eval_elo.py
git commit -m "$(printf 'feat(eval): pure Elo/score/CI stats for checkpoint tournament\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 2: `eval_runner.py` core — dataclasses, per-game loop, sequential runner

**Files:**
- Create: `scripts/GPU/alphazero/eval_runner.py`
- Create: `tests/eval_fakes.py`
- Test: `tests/test_eval_runner.py`

- [ ] **Step 0: Confirm import paths (no code change)**

These five symbols are imported by the new modules. All were verified to
resolve against the repo on 2026-05-31; re-run if the tree has changed:

Run:
```bash
.venv/bin/python -c "
from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
from scripts.GPU.alphazero.probe_eval import load_network_for_scoring
print('imports OK')
"
```
Expected: `imports OK`. If any import fails, fix the path in `eval_runner.py`
before proceeding (this would be a non-design failure).

- [ ] **Step 1: Write the fake evaluator helper**

Create `tests/eval_fakes.py` (NOT a `test_*.py` file, so pytest won't collect it; module-level so it pickles under spawn):

```python
"""GPU-free deterministic fakes for eval tests.

FakeEvaluator implements the Evaluator protocol MCTS expects
(build_input_tensor + infer) with uniform priors and a fixed value, so
games are deterministic per seed and need no checkpoint or GPU.

Factories are module-level functions so they pickle under the spawn
multiprocessing context (lambdas do not).
"""
from __future__ import annotations

import numpy as np


class FakeEvaluator:
    def __init__(self, value: float = 0.0):
        self._value = float(value)

    def build_input_tensor(self, state) -> np.ndarray:
        # (C, H, W); contents ignored by infer. Minimal C=1.
        return np.zeros((1, state.active_size, state.active_size), dtype=np.float32)

    def infer(self, boards, move_rows, move_cols, move_mask, active_size):
        mask = move_mask.astype(np.float32)
        row_sums = mask.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        priors = (mask / row_sums).astype(np.float32)
        values = np.full((mask.shape[0],), self._value, dtype=np.float32)
        return priors, values


def fake_evaluator_factory(path: str) -> FakeEvaluator:
    """Picklable factory: ignores path, returns a fresh FakeEvaluator."""
    return FakeEvaluator(value=0.0)


def counting_factory(path: str) -> FakeEvaluator:
    """Sequential-only counting factory (process-local dict)."""
    counting_factory.calls[path] = counting_factory.calls.get(path, 0) + 1
    return FakeEvaluator(value=0.0)


counting_factory.calls = {}
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_eval_runner.py`:

```python
import pytest

from scripts.GPU.alphazero.eval_runner import (
    EvalGameTask, EvalGameResult, EvalConfig,
    cfg_from, play_eval_game, make_result, run_game_tasks,
)
from tests.eval_fakes import FakeEvaluator, fake_evaluator_factory, counting_factory


def _tiny_cfg(**kw):
    base = dict(board_size=8, mcts_sims=8, mcts_eval_batch_size=4,
                mcts_stall_flush_sims=4, opening_temp_plies=4,
                temp_high=1.0, temp_low=0.1, max_moves=12)
    base.update(kw)
    return EvalConfig(**base)


def test_cfg_from_maps_fields():
    cfg = cfg_from(_tiny_cfg())
    assert cfg.n_simulations == 8
    assert cfg.eval_batch_size == 4
    assert cfg.stall_flush_sims == 4
    assert cfg.temp_threshold_ply == 4
    assert cfg.temp_high == 1.0 and cfg.temp_low == 0.1


def test_cfg_from_argmax_zeroes_temps():
    cfg = cfg_from(_tiny_cfg(selection_mode="argmax"))
    assert cfg.temp_high == 0.0 and cfg.temp_low == 0.0


def test_cfg_from_rejects_unknown_mode():
    with pytest.raises(ValueError):
        cfg_from(_tiny_cfg(selection_mode="bogus"))


def test_play_eval_game_is_deterministic_by_seed():
    cfg = _tiny_cfg()
    r1 = play_eval_game(FakeEvaluator(), FakeEvaluator(), cfg, seed=123)
    r2 = play_eval_game(FakeEvaluator(), FakeEvaluator(), cfg, seed=123)
    assert r1 == r2


def test_play_eval_game_reason_is_valid():
    winner, reason, n = play_eval_game(FakeEvaluator(), FakeEvaluator(),
                                       _tiny_cfg(), seed=1)
    assert reason in {"win", "state_cap", "board_full", "unknown_error"}
    assert reason != "unknown_error"
    assert n >= 1


def test_make_result_red_win_credits_red_checkpoint():
    task = EvalGameTask(0, "p", 0, "A.safetensors", "B.safetensors", 7)
    res = make_result(task, "red", "win", 40)
    assert res.winner_checkpoint == "A.safetensors"
    assert res.red_score == 1.0 and res.black_score == 0.0


def test_make_result_black_win_credits_black_checkpoint():
    task = EvalGameTask(0, "p", 1, "B.safetensors", "A.safetensors", 7)
    res = make_result(task, "black", "win", 40)
    assert res.winner_checkpoint == "A.safetensors"
    assert res.red_score == 0.0 and res.black_score == 1.0


def test_make_result_state_cap_is_half_each():
    task = EvalGameTask(0, "p", 0, "A.safetensors", "B.safetensors", 7)
    res = make_result(task, None, "state_cap", 12)
    assert res.winner_checkpoint is None
    assert res.red_score == 0.5 and res.black_score == 0.5


def test_run_game_tasks_workers1_sorted_and_complete():
    tasks = [
        EvalGameTask(5, "p", 5, "A", "B", 105),
        EvalGameTask(0, "p", 0, "A", "B", 100),
        EvalGameTask(2, "p", 2, "B", "A", 102),
    ]
    out = run_game_tasks(tasks, workers=1, config=_tiny_cfg(),
                         evaluator_factory=fake_evaluator_factory)
    assert [r.game_idx for r in out] == [0, 2, 5]  # sorted by (pairing_id, game_idx)
    assert len(out) == 3


def test_run_game_tasks_empty_returns_empty():
    assert run_game_tasks([], workers=4, config=_tiny_cfg(),
                          evaluator_factory=fake_evaluator_factory) == []


def test_worker_cache_loads_each_checkpoint_once_sequential():
    counting_factory.calls.clear()
    tasks = [
        EvalGameTask(0, "p", 0, "A", "B", 200),
        EvalGameTask(1, "p", 1, "B", "A", 201),  # reuses A and B
        EvalGameTask(2, "p", 2, "A", "B", 202),
    ]
    run_game_tasks(tasks, workers=1, config=_tiny_cfg(),
                   evaluator_factory=counting_factory)
    assert counting_factory.calls == {"A": 1, "B": 1}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_runner.py -q`
Expected: FAIL — `ImportError` / `cannot import name` from `eval_runner`

- [ ] **Step 4: Write the implementation (core + sequential path)**

Create `scripts/GPU/alphazero/eval_runner.py`:

```python
"""Shared executor for checkpoint-tournament games.

A flat list of EvalGameTask is drained by run_game_tasks, which both the
match and tournament scripts call. workers==1 runs in-process; workers>1
uses a spawn worker pool with a shared atomic task counter (the trainer
idiom). Determinism is task-derived: same base_seed + schedule => same
games regardless of worker count.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import random
from dataclasses import dataclass
from typing import Callable, Optional

from .game.twixt_state import TwixtState
from .mcts import MCTS, MCTSConfig

# game_idx and pairing offsets share this stride; games-per-pairing must
# stay below it so task_ids/seeds never collide across pairings.
GAMES_PER_PAIRING_LIMIT = 1_000_000

EvaluatorFactory = Callable[[str], object]


@dataclass(frozen=True)
class EvalGameTask:
    task_id: int
    pairing_id: str
    game_idx: int
    red_checkpoint: str
    black_checkpoint: str
    seed: int


@dataclass
class EvalGameResult:
    task_id: int
    pairing_id: str
    game_idx: int
    red_checkpoint: str
    black_checkpoint: str
    winner: Optional[str]            # "red" | "black" | None
    winner_checkpoint: Optional[str]
    reason: str                      # "win"|"state_cap"|"board_full"|"unknown_error"
    n_moves: int
    red_score: float
    black_score: float


@dataclass(frozen=True)
class EvalConfig:
    board_size: int = 24
    mcts_sims: int = 400              # SIMS_TABLE[24]
    mcts_eval_batch_size: int = 14
    mcts_stall_flush_sims: int = 48
    selection_mode: str = "opening_temperature"   # or "argmax"
    opening_temp_plies: int = 20
    temp_high: float = 1.0
    temp_low: float = 0.1
    max_moves: int = 280             # MAX_MOVES_TABLE[24]


@dataclass(frozen=True)
class _WorkerDone:
    worker_id: int


def cfg_from(config: EvalConfig) -> MCTSConfig:
    """Map EvalConfig -> MCTSConfig. argmax mode zeroes temps to hit
    select_move's deterministic argmax branch."""
    if config.selection_mode == "argmax":
        th, tl = 0.0, 0.0
    elif config.selection_mode == "opening_temperature":
        th, tl = config.temp_high, config.temp_low
    else:
        raise ValueError(f"unknown selection_mode {config.selection_mode!r}")
    return MCTSConfig(
        n_simulations=config.mcts_sims,
        eval_batch_size=config.mcts_eval_batch_size,
        stall_flush_sims=config.mcts_stall_flush_sims,
        temp_threshold_ply=config.opening_temp_plies,
        temp_high=th,
        temp_low=tl,
    )


def play_eval_game(red_eval, black_eval, config: EvalConfig, seed: int):
    """Play one A-vs-B game. Returns (winner, reason, n_moves).

    Independent per-side child RNGs (seed-derived) avoid cross-player
    coupling. Classification is explicit via winner()/ply/legal_moves --
    is_terminal() conflates win/cap/board-full (twixt_state.py:549-574).
    """
    mcts_red = MCTS(red_eval, cfg_from(config), random.Random(seed ^ 0xA5A5A5))
    mcts_black = MCTS(black_eval, cfg_from(config), random.Random(seed ^ 0x5A5A5A))
    state = TwixtState(active_size=config.board_size, to_move="red",
                       max_plies_limit=config.max_moves)
    ply = 0
    # Explicit loop condition -- do NOT gate on is_terminal(), which conflates
    # win / cap / board-full. Continue while there is no real winner, the ply
    # cap is not reached, and legal moves remain.
    while state.winner() is None and ply < config.max_moves and state.legal_moves():
        mcts = mcts_red if state.to_move == "red" else mcts_black
        counts, _ = mcts.search(state, add_noise=False)
        move = mcts.select_move(counts, ply)
        state = state.apply_move(move)
        ply += 1
    winner = state.winner()
    if winner is not None:
        reason = "win"
    elif ply >= config.max_moves:
        reason = "state_cap"
    elif not state.legal_moves():
        reason = "board_full"
    else:
        reason = "unknown_error"
    return winner, reason, ply


def make_result(task: EvalGameTask, winner, reason, n_moves) -> EvalGameResult:
    """Build a result, mapping winner color -> checkpoint and 0/0.5/1 scores."""
    if winner == "red":
        red_score, black_score, winner_ckpt = 1.0, 0.0, task.red_checkpoint
    elif winner == "black":
        red_score, black_score, winner_ckpt = 0.0, 1.0, task.black_checkpoint
    else:
        red_score, black_score, winner_ckpt = 0.5, 0.5, None
    return EvalGameResult(
        task_id=task.task_id, pairing_id=task.pairing_id, game_idx=task.game_idx,
        red_checkpoint=task.red_checkpoint, black_checkpoint=task.black_checkpoint,
        winner=winner, winner_checkpoint=winner_ckpt, reason=reason,
        n_moves=n_moves, red_score=red_score, black_score=black_score,
    )


def build_pairing_tasks(pairing_id, a_ckpt, b_ckpt, games, base_seed, pairing_index):
    """Balanced-color tasks for one pairing. Even game_idx -> red=A; odd -> red=B.
    task_id and seed are task-derived (stable across worker counts)."""
    if games < 2:
        raise ValueError("games must be >= 2")
    if games % 2 != 0:
        # Color balancing assigns A=red on even game_idx, A=black on odd.
        # An odd count gives one model an extra red game -> biased.
        raise ValueError("games must be even for balanced colors")
    if games >= GAMES_PER_PAIRING_LIMIT:
        raise ValueError(f"games must be < {GAMES_PER_PAIRING_LIMIT}")
    offset = pairing_index * GAMES_PER_PAIRING_LIMIT
    tasks = []
    for g in range(games):
        red, black = (a_ckpt, b_ckpt) if g % 2 == 0 else (b_ckpt, a_ckpt)
        tasks.append(EvalGameTask(
            task_id=offset + g, pairing_id=pairing_id, game_idx=g,
            red_checkpoint=red, black_checkpoint=black, seed=base_seed + offset + g,
        ))
    return tasks


def short_id(token: str) -> str:
    """Derive a short iter id from a checkpoint path, or pass a bare id through.

    Lives here (low-level shared module) so eval_summary and the tournament/
    match scripts all import it from one place — avoids a circular import.
    """
    base = os.path.basename(token)
    if base.startswith("model_iter_") and base.endswith(".safetensors"):
        return base[len("model_iter_"):-len(".safetensors")]
    return token


def resolve_checkpoint(token: str, checkpoints_dir: str) -> str:
    """Resolve a token to a checkpoint path.

    A path (contains os.sep or ends with .safetensors) passes through; a bare
    iter id resolves to <dir>/model_iter_<id>.safetensors.
    """
    if os.sep in token or token.endswith(".safetensors"):
        return token
    return os.path.join(checkpoints_dir, f"model_iter_{token}.safetensors")


def _default_evaluator_factory(path: str):
    """Real loader: auto-detects 24/30-channel, wraps in LocalGPUEvaluator.
    Imported lazily so fake-evaluator tests need no MLX."""
    from .probe_eval import load_network_for_scoring
    from .local_evaluator import LocalGPUEvaluator
    net, _in_ch, _hidden, _blocks = load_network_for_scoring(path, verbose=False)
    return LocalGPUEvaluator(net)


def _sorted(results):
    return sorted(results, key=lambda r: (r.pairing_id, r.game_idx))


def _make_cache(factory):
    cache: dict = {}

    def get_eval(path):
        ev = cache.get(path)
        if ev is None:
            ev = factory(path)
            cache[path] = ev
        return ev

    return get_eval


def _run_sequential(tasks, config, factory):
    get_eval = _make_cache(factory)
    results = []
    for task in tasks:
        red = get_eval(task.red_checkpoint)
        black = get_eval(task.black_checkpoint)
        winner, reason, nm = play_eval_game(red, black, config, task.seed)
        results.append(make_result(task, winner, reason, nm))
    return _sorted(results)


def run_game_tasks(tasks, workers: int, config: EvalConfig,
                   evaluator_factory: Optional[EvaluatorFactory] = None):
    """Execute tasks; return results sorted by (pairing_id, game_idx).

    workers<=1 runs in-process. workers>1 (spawn pool) is added in Task 5;
    until then it raises NotImplementedError so the intermediate commit has
    no live unresolved reference.

    NOTE: when workers>1, evaluator_factory must be a MODULE-LEVEL picklable
    callable (it is sent to spawned workers). Lambdas/closures will fail to
    pickle. The default real loader and the test fakes satisfy this.
    """
    factory = evaluator_factory or _default_evaluator_factory
    if not tasks:
        return []
    workers = min(workers, len(tasks))
    if workers <= 1:
        return _run_sequential(tasks, config, factory)
    raise NotImplementedError("workers > 1 added in Task 5")
```

> NOTE: the `workers>1` branch raises `NotImplementedError` in this task. All
> Task 2/3/4 tests use `workers=1`, so this is safe; Task 5 replaces the raise
> with the real `_run_parallel` call and adds its tests.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_runner.py -q`
Expected: PASS (11 passed)

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/eval_runner.py tests/eval_fakes.py tests/test_eval_runner.py
git commit -m "$(printf 'feat(eval): runner core + sequential path + fake evaluator\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 3: Task builders (match + tournament)

**Files:**
- Modify: `scripts/GPU/alphazero/eval_checkpoint_match.py` (create with `build_match_tasks`)
- Modify: `scripts/GPU/alphazero/eval_checkpoint_tournament.py` (create with `build_tournament_tasks` + helpers)
- Test: `tests/test_eval_builders.py`

> The CLI `main()` for each script is added in Tasks 7-8. This task adds only
> the pure, importable builder functions so they can be tested without MLX.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_builders.py`:

```python
import pytest

from scripts.GPU.alphazero.eval_checkpoint_match import build_match_tasks
from scripts.GPU.alphazero.eval_checkpoint_tournament import (
    build_tournament_tasks, short_id, resolve_checkpoint,
)


def test_match_balanced_colors():
    tasks = build_match_tasks("A.sft", "B.sft", games=4, base_seed=1000,
                              pairing_id="A_vs_B")
    reds = [t.red_checkpoint for t in tasks]
    assert reds == ["A.sft", "B.sft", "A.sft", "B.sft"]
    assert sum(1 for t in tasks if t.red_checkpoint == "A.sft") == 2


def test_match_seed_is_task_derived_and_stable():
    t1 = build_match_tasks("A", "B", games=4, base_seed=1000, pairing_id="p")
    t2 = build_match_tasks("A", "B", games=4, base_seed=1000, pairing_id="p")
    assert [t.seed for t in t1] == [t.seed for t in t2]
    assert [t.seed for t in t1] == [1000, 1001, 1002, 1003]


def test_match_rejects_too_few_games():
    with pytest.raises(ValueError):
        build_match_tasks("A", "B", games=1, base_seed=0, pairing_id="p")


def test_match_rejects_odd_games():
    # Odd count would give one model an extra red game -> color imbalance.
    with pytest.raises(ValueError, match="even"):
        build_match_tasks("A", "B", games=3, base_seed=0, pairing_id="p")


def test_tournament_flat_list_unique_task_ids():
    pairings = [("A", "B"), ("A", "C")]
    tasks = build_tournament_tasks(pairings, games=4, base_seed=500)
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids)) == 8


def test_tournament_pairing_ids_and_grouping():
    pairings = [("A", "B"), ("A", "C")]
    tasks = build_tournament_tasks(pairings, games=2, base_seed=0)
    pids = {t.pairing_id for t in tasks}
    assert pids == {"A_vs_B", "A_vs_C"}


def test_tournament_seeds_independent_across_pairings():
    # Pairing 0 and pairing 1 must not share seeds (offset by stride).
    tasks = build_tournament_tasks([("A", "B"), ("A", "C")], games=2, base_seed=0)
    seeds = [t.seed for t in tasks]
    assert len(seeds) == len(set(seeds))


def test_short_id_from_path():
    assert short_id("checkpoints/x/model_iter_0419.safetensors") == "0419"
    assert short_id("0419") == "0419"


def test_resolve_checkpoint_short_id_uses_dir():
    path = resolve_checkpoint("0419", "checkpoints/alphazero-v2-staged")
    assert path == "checkpoints/alphazero-v2-staged/model_iter_0419.safetensors"


def test_resolve_checkpoint_passthrough_full_path():
    p = "checkpoints/x/model_iter_0419.safetensors"
    assert resolve_checkpoint(p, "ignored") == p
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_builders.py -q`
Expected: FAIL — `ModuleNotFoundError` for the two new modules

- [ ] **Step 3: Write `eval_checkpoint_match.py` builder**

Create `scripts/GPU/alphazero/eval_checkpoint_match.py`:

```python
"""Checkpoint match: one A-vs-B pairing. Builder + CLI (CLI added later)."""
from __future__ import annotations

from .eval_runner import build_pairing_tasks


def build_match_tasks(a_ckpt: str, b_ckpt: str, games: int, base_seed: int,
                      pairing_id: str):
    """Tasks for a single pairing (pairing_index fixed at 0)."""
    return build_pairing_tasks(pairing_id, a_ckpt, b_ckpt, games, base_seed,
                               pairing_index=0)
```

- [ ] **Step 4: Write `eval_checkpoint_tournament.py` builder + helpers**

Create `scripts/GPU/alphazero/eval_checkpoint_tournament.py`:

```python
"""Checkpoint tournament: many pairings in one flat task list. Builder + CLI."""
from __future__ import annotations

import os

# short_id / resolve_checkpoint live in eval_runner (shared low-level module)
# and are re-exported here so callers/tests can import them from either place.
# This avoids a circular import (eval_summary also needs short_id).
from .eval_runner import build_pairing_tasks, short_id, resolve_checkpoint


def build_tournament_tasks(pairings, games: int, base_seed: int):
    """Flat task list across all pairings. pairings: list[(a_ckpt, b_ckpt)].

    Each pairing gets a distinct pairing_index, so task_ids and seeds never
    collide (stride = GAMES_PER_PAIRING_LIMIT).
    """
    tasks = []
    for idx, (a_ckpt, b_ckpt) in enumerate(pairings):
        pairing_id = f"{short_id(a_ckpt)}_vs_{short_id(b_ckpt)}"
        tasks.extend(build_pairing_tasks(pairing_id, a_ckpt, b_ckpt, games,
                                         base_seed, pairing_index=idx))
    return tasks
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_builders.py -q`
Expected: PASS (10 passed)

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/eval_checkpoint_match.py scripts/GPU/alphazero/eval_checkpoint_tournament.py tests/test_eval_builders.py
git commit -m "$(printf 'feat(eval): match + tournament task builders\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 4: `eval_summary.py` — aggregation

**Files:**
- Create: `scripts/GPU/alphazero/eval_summary.py`
- Test: `tests/test_eval_summary.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_summary.py`:

```python
from scripts.GPU.alphazero.eval_runner import EvalGameResult
from scripts.GPU.alphazero.eval_summary import summarize_match, summarize_tournament


def _res(game_idx, red, black, winner, reason, n=50):
    if winner == "red":
        rs, bs, wc = 1.0, 0.0, red
    elif winner == "black":
        rs, bs, wc = 0.0, 1.0, black
    else:
        rs, bs, wc = 0.5, 0.5, None
    return EvalGameResult(game_idx, "A_vs_B", game_idx, red, black,
                          winner, wc, reason, n, rs, bs)


def _match_results():
    # 4 games, balanced colors. A wins both as red; split as black.
    return [
        _res(0, "A", "B", "red", "win"),     # A(red) win
        _res(1, "B", "A", "black", "win"),   # A(black) win
        _res(2, "A", "B", "black", "win"),   # B(black) win -> A loss
        _res(3, "B", "A", None, "state_cap"),  # draw
    ]


def test_summarize_match_counts_and_score():
    s = summarize_match(_match_results(), "A", "B", "A_vs_B", config={})
    assert s["games"] == 4
    assert s["a_wins"] == 2 and s["b_wins"] == 1 and s["state_caps"] == 1
    assert s["a_score"] == 2.5
    assert abs(s["a_score_rate"] - 0.625) < 1e-9
    assert s["verdict"] == "stronger"


def test_summarize_match_by_color_blocks():
    s = summarize_match(_match_results(), "A", "B", "A_vs_B", config={})
    assert s["a_as_red"]["games"] == 2 and s["a_as_red"]["wins"] == 1
    assert s["a_as_black"]["games"] == 2 and s["a_as_black"]["wins"] == 1


def test_summarize_match_color_bias_and_avg_plies():
    s = summarize_match(_match_results(), "A", "B", "A_vs_B", config={})
    # decisive winners by color: red wins in g0; black wins in g1,g2 -> red 1/3
    assert abs(s["color_bias"]["red_win_rate_decisive"] - (1 / 3)) < 1e-9
    assert s["avg_plies"] == 50.0
    assert s["draw_score_policy"] == "state_cap_and_board_full_score_0.5"


def test_summarize_tournament_groups_by_pairing():
    r = _match_results()
    out = summarize_tournament(r, [("A", "B")], config={})
    assert len(out["pairings"]) == 1
    assert out["pairings"][0]["pairing_id"] == "A_vs_B"
    assert len(out["table"]) == 1
    assert out["table"][0]["verdict"] == "stronger"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_summary.py -q`
Expected: FAIL — `ModuleNotFoundError: ... eval_summary`

- [ ] **Step 3: Write the implementation**

Create `scripts/GPU/alphazero/eval_summary.py`:

```python
"""Aggregate EvalGameResults into match / tournament summary dicts.

Pure: no MLX, no time, no git (the CLI stamps generated_at / git_commit).
"""
from __future__ import annotations

from statistics import mean

from .eval_elo import score_rate, elo_diff, score_ci_trinomial, elo_ci, verdict
from .eval_runner import short_id   # shared low-level module (no import cycle)

DRAW_SCORE_POLICY = "state_cap_and_board_full_score_0.5"


def _color_stats(results, model_ckpt, color):
    if color == "red":
        sub = [r for r in results if r.red_checkpoint == model_ckpt]
        wins = sum(1 for r in sub if r.winner == "red")
        losses = sum(1 for r in sub if r.winner == "black")
    else:
        sub = [r for r in results if r.black_checkpoint == model_ckpt]
        wins = sum(1 for r in sub if r.winner == "black")
        losses = sum(1 for r in sub if r.winner == "red")
    caps = sum(1 for r in sub if r.winner is None)
    n = len(sub)
    return {
        "games": n, "wins": wins, "losses": losses, "caps": caps,
        "score_rate": (score_rate(wins, caps, n) if n else None),
    }


def summarize_match(results, a_ckpt, b_ckpt, pairing_id, config) -> dict:
    if not results:
        # Empty here means a grouping bug (callers reject empty pairings
        # before running). Fail loud rather than emit a 0.0 placeholder.
        raise ValueError(f"no results for pairing {pairing_id}")
    games = len(results)
    a_wins = sum(1 for r in results if r.winner_checkpoint == a_ckpt)
    b_wins = sum(1 for r in results if r.winner_checkpoint == b_ckpt)
    state_caps = sum(1 for r in results if r.reason == "state_cap")
    board_full = sum(1 for r in results if r.reason == "board_full")
    draws = state_caps + board_full
    a_score = a_wins + 0.5 * draws
    rate = score_rate(a_wins, draws, games)
    s_lo, s_hi = score_ci_trinomial(a_wins, draws, b_wins)
    e_lo, e_hi = elo_ci(a_wins, draws, b_wins)

    red_wins = sum(1 for r in results if r.winner == "red")
    black_wins = sum(1 for r in results if r.winner == "black")
    decisive = red_wins + black_wins

    return {
        "pairing_id": pairing_id,
        "checkpoint_a": a_ckpt,
        "checkpoint_b": b_ckpt,
        "games": games,
        "a_wins": a_wins, "b_wins": b_wins,
        "state_caps": state_caps, "board_full": board_full,
        "a_score": a_score,
        "a_score_rate": rate,
        "elo_estimate": elo_diff(rate, games),
        "elo_ci95": [e_lo, e_hi],
        "score_rate_ci95": [s_lo, s_hi],
        "verdict": verdict(rate),
        "a_as_red": _color_stats(results, a_ckpt, "red"),
        "a_as_black": _color_stats(results, a_ckpt, "black"),
        "color_bias": {
            "red_win_rate_decisive": (red_wins / decisive) if decisive else None,
        },
        "avg_plies": mean(r.n_moves for r in results),
        "selection_mode": config.get("selection_mode") if config else None,
        "draw_score_policy": DRAW_SCORE_POLICY,
        "config": config,
    }


def summarize_tournament(results, pairings, config) -> dict:
    by_pairing: dict = {}
    for r in results:
        by_pairing.setdefault(r.pairing_id, []).append(r)

    pairing_summaries = []
    for a_ckpt, b_ckpt in pairings:
        pid = f"{short_id(a_ckpt)}_vs_{short_id(b_ckpt)}"
        group = by_pairing.get(pid, [])
        pairing_summaries.append(
            summarize_match(group, a_ckpt, b_ckpt, pid, config)
        )

    table = [
        {
            "pairing_id": s["pairing_id"],
            "a_score_rate": s["a_score_rate"],
            "elo_estimate": s["elo_estimate"],
            "elo_ci95": s["elo_ci95"],
            "verdict": s["verdict"],
        }
        for s in pairing_summaries
    ]
    table.sort(key=lambda t: t["pairing_id"])
    return {"pairings": pairing_summaries, "table": table, "config": config}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_summary.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_summary.py tests/test_eval_summary.py
git commit -m "$(printf 'feat(eval): match/tournament result aggregation\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 5: `run_game_tasks` parallel spawn path

**Files:**
- Modify: `scripts/GPU/alphazero/eval_runner.py` (add `_run_parallel` + `_worker_main`)
- Test: `tests/test_eval_runner_parallel.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_runner_parallel.py`:

```python
import pytest

from scripts.GPU.alphazero.eval_runner import EvalGameTask, EvalConfig, run_game_tasks
from scripts.GPU.alphazero.eval_checkpoint_tournament import build_tournament_tasks
from tests.eval_fakes import fake_evaluator_factory


def _tiny_cfg():
    return EvalConfig(board_size=8, mcts_sims=8, mcts_eval_batch_size=4,
                      mcts_stall_flush_sims=4, opening_temp_plies=4,
                      temp_high=1.0, temp_low=0.1, max_moves=12)


def _key(results):
    # Identity + outcome tuple per result, in sorted order, for comparison.
    return [(r.task_id, r.pairing_id, r.game_idx, r.winner, r.reason,
             r.n_moves, r.red_score) for r in results]


def test_workers1_vs_workers2_identical_results():
    tasks = build_tournament_tasks([("A", "B"), ("A", "C")], games=6, base_seed=42)
    seq = run_game_tasks(tasks, workers=1, config=_tiny_cfg(),
                         evaluator_factory=fake_evaluator_factory)
    par = run_game_tasks(tasks, workers=2, config=_tiny_cfg(),
                         evaluator_factory=fake_evaluator_factory)
    assert _key(seq) == _key(par)


def test_parallel_returns_all_results_sorted():
    tasks = build_tournament_tasks([("A", "B")], games=8, base_seed=7)
    out = run_game_tasks(tasks, workers=3, config=_tiny_cfg(),
                         evaluator_factory=fake_evaluator_factory)
    assert len(out) == 8
    assert [r.game_idx for r in out] == sorted(r.game_idx for r in out)
```

> These spawn real processes; they are slow-ish but unmarked so they run in CI.
> The fake factory is a module-level function so it pickles under spawn.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_runner_parallel.py -q`
Expected: FAIL — `NameError: name '_run_parallel' is not defined`

- [ ] **Step 3a: Activate the parallel branch in `run_game_tasks`**

In `scripts/GPU/alphazero/eval_runner.py`, replace the placeholder line inside
`run_game_tasks`:

```python
    raise NotImplementedError("workers > 1 added in Task 5")
```
with:
```python
    return _run_parallel(tasks, workers, config, factory)
```

- [ ] **Step 3b: Append the parallel path to `eval_runner.py`**

Append to `scripts/GPU/alphazero/eval_runner.py` (after `_run_sequential`).
`factory` is sent to spawned workers, so it must be a module-level picklable
callable (see `run_game_tasks` docstring):

```python
def _worker_main(worker_id, tasks, config, factory, next_idx, result_q):
    """Pull tasks via the shared atomic counter; per-process checkpoint cache."""
    get_eval = _make_cache(factory)
    n = len(tasks)
    while True:
        with next_idx.get_lock():
            i = next_idx.value
            if i >= n:
                break
            next_idx.value = i + 1
        task = tasks[i]
        red = get_eval(task.red_checkpoint)
        black = get_eval(task.black_checkpoint)
        winner, reason, nm = play_eval_game(red, black, config, task.seed)
        result_q.put(make_result(task, winner, reason, nm))
    result_q.put(_WorkerDone(worker_id))


def _run_parallel(tasks, workers, config, factory):
    """Spawn pool (macOS-mandatory). Shared next-task counter, results via
    queue, explicit WorkerDone, parent joins with timeout (no silent hang)."""
    ctx = mp.get_context("spawn")
    next_idx = ctx.Value("i", 0)
    result_q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker_main,
                    args=(wid, tasks, config, factory, next_idx, result_q))
        for wid in range(workers)
    ]
    for p in procs:
        p.start()

    GET_TIMEOUT = 600  # seconds without progress => assume stall
    results = []
    done = 0
    while done < workers:
        try:
            msg = result_q.get(timeout=GET_TIMEOUT)
        except queue.Empty:
            dead = [(p.pid, p.exitcode) for p in procs
                    if p.exitcode not in (None, 0)]
            for p in procs:
                p.terminate()
            raise RuntimeError(
                f"eval workers stalled (>{GET_TIMEOUT}s, no result); "
                f"crashed={dead}"
            )
        if isinstance(msg, _WorkerDone):
            done += 1
        else:
            results.append(msg)

    for p in procs:
        p.join(timeout=GET_TIMEOUT)

    if len(results) != len(tasks):
        raise RuntimeError(
            f"expected {len(tasks)} results, collected {len(results)}"
        )
    return _sorted(results)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_runner_parallel.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full eval suite to confirm no regressions**

Run: `.venv/bin/python -m pytest tests/test_eval_elo.py tests/test_eval_runner.py tests/test_eval_runner_parallel.py tests/test_eval_builders.py tests/test_eval_summary.py -q`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/eval_runner.py tests/test_eval_runner_parallel.py
git commit -m "$(printf 'feat(eval): spawn worker pool with shared task counter\n\nDynamic work-stealing, per-worker checkpoint cache, explicit WorkerDone,\nparent join-with-timeout. workers=1 vs 2 produce identical results.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 6: Match CLI + fake end-to-end smoke

**Files:**
- Modify: `scripts/GPU/alphazero/eval_checkpoint_match.py` (add `run_match`, `main`)
- Test: `tests/test_eval_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_cli.py`:

```python
import json

from scripts.GPU.alphazero.eval_runner import EvalConfig
from scripts.GPU.alphazero.eval_checkpoint_match import run_match
from tests.eval_fakes import fake_evaluator_factory


def _tiny_cfg():
    return EvalConfig(board_size=8, mcts_sims=8, mcts_eval_batch_size=4,
                      mcts_stall_flush_sims=4, opening_temp_plies=4,
                      temp_high=1.0, temp_low=0.1, max_moves=12)


def test_run_match_two_games_writes_outputs(tmp_path):
    out = tmp_path / "m.json"
    summary = run_match(
        a_ckpt="A", b_ckpt="B", games=2, base_seed=1, config=_tiny_cfg(),
        workers=1, output=str(out), evaluator_factory=fake_evaluator_factory,
    )
    # Summary fields present and internally consistent.
    assert summary["games"] == 2
    assert summary["a_wins"] + summary["b_wins"] + summary["state_caps"] \
        + summary["board_full"] == 2
    assert "elo_estimate" in summary and "a_as_red" in summary

    # Files written: summary JSON + per-game JSONL.
    assert out.exists()
    games_file = tmp_path / "m_games.jsonl"
    assert games_file.exists()
    lines = games_file.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert {"task_id", "pairing_id", "game_idx", "winner", "reason"} <= rec.keys()


def test_run_match_pairing_id_default(tmp_path):
    out = tmp_path / "m.json"
    s = run_match(a_ckpt="checkpoints/x/model_iter_0419.safetensors",
                  b_ckpt="checkpoints/x/model_iter_0379.safetensors",
                  games=2, base_seed=0, config=_tiny_cfg(), workers=1,
                  output=str(out), evaluator_factory=fake_evaluator_factory)
    assert s["pairing_id"] == "0419_vs_0379"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_cli.py::test_run_match_two_games_writes_outputs -q`
Expected: FAIL — `cannot import name 'run_match'`

- [ ] **Step 3: Add `run_match` + `main` to `eval_checkpoint_match.py`**

Replace the contents of `scripts/GPU/alphazero/eval_checkpoint_match.py` with:

```python
"""Checkpoint match: one A-vs-B pairing.

Builds one pairing's balanced-color tasks, runs them through the shared
eval_runner pool, aggregates a summary, and writes per-game JSONL + summary
JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone

from .eval_runner import EvalConfig, build_pairing_tasks, run_game_tasks, short_id
from .eval_summary import summarize_match


def build_match_tasks(a_ckpt, b_ckpt, games, base_seed, pairing_id):
    """Tasks for a single pairing (pairing_index fixed at 0)."""
    return build_pairing_tasks(pairing_id, a_ckpt, b_ckpt, games, base_seed,
                               pairing_index=0)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _write_outputs(output, summary, results):
    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    stem, _ext = os.path.splitext(output)
    games_path = f"{stem}_games.jsonl"
    with open(games_path, "w") as fh:
        for r in results:  # already sorted by (pairing_id, game_idx)
            fh.write(json.dumps(asdict(r)) + "\n")
    with open(output, "w") as fh:
        json.dump(summary, fh, indent=2)


def run_match(a_ckpt, b_ckpt, games, base_seed, config, workers, output,
              pairing_id=None, evaluator_factory=None):
    """Run a full match and write outputs. Returns the summary dict."""
    if pairing_id is None:
        pairing_id = f"{short_id(a_ckpt)}_vs_{short_id(b_ckpt)}"
    tasks = build_match_tasks(a_ckpt, b_ckpt, games, base_seed, pairing_id)
    results = run_game_tasks(tasks, workers=workers, config=config,
                             evaluator_factory=evaluator_factory)
    config_dict = {**asdict(config), "base_seed": base_seed, "workers": workers}
    summary = summarize_match(results, a_ckpt, b_ckpt, pairing_id, config_dict)
    summary["git_commit"] = _git_commit()
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    if output:
        _write_outputs(output, summary, results)
    return summary


def _build_arg_parser():
    ap = argparse.ArgumentParser(description="Run a checkpoint A-vs-B match.")
    ap.add_argument("--checkpoint-a", required=True)
    ap.add_argument("--checkpoint-b", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--board-size", type=int, default=24)
    ap.add_argument("--mcts-sims", type=int, default=400)
    ap.add_argument("--mcts-eval-batch-size", type=int, default=14)
    ap.add_argument("--mcts-stall-flush-sims", type=int, default=48)
    ap.add_argument("--selection-mode", default="opening_temperature",
                    choices=["opening_temperature", "argmax"])
    ap.add_argument("--opening-temp-plies", type=int, default=20)
    ap.add_argument("--temp-high", type=float, default=1.0)
    ap.add_argument("--temp-low", type=float, default=0.1)
    ap.add_argument("--max-moves", type=int, default=280)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--base-seed", type=int, default=12345)
    ap.add_argument("--output", required=True)
    return ap


def _config_from_args(args) -> EvalConfig:
    return EvalConfig(
        board_size=args.board_size, mcts_sims=args.mcts_sims,
        mcts_eval_batch_size=args.mcts_eval_batch_size,
        mcts_stall_flush_sims=args.mcts_stall_flush_sims,
        selection_mode=args.selection_mode,
        opening_temp_plies=args.opening_temp_plies,
        temp_high=args.temp_high, temp_low=args.temp_low,
        max_moves=args.max_moves,
    )


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    for path in (args.checkpoint_a, args.checkpoint_b):
        if not os.path.exists(path):
            raise SystemExit(f"checkpoint not found: {path}")
    summary = run_match(
        a_ckpt=args.checkpoint_a, b_ckpt=args.checkpoint_b, games=args.games,
        base_seed=args.base_seed, config=_config_from_args(args),
        workers=args.workers, output=args.output,
    )
    print(f"{summary['pairing_id']}: a_score_rate={summary['a_score_rate']:.4f} "
          f"elo={summary['elo_estimate']:.1f} "
          f"CI95={summary['elo_ci95']} verdict={summary['verdict']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval_cli.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_checkpoint_match.py tests/test_eval_cli.py
git commit -m "$(printf 'feat(eval): match CLI + run_match + fake e2e smoke\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 7: Tournament CLI + "single pool" test

**Files:**
- Modify: `scripts/GPU/alphazero/eval_checkpoint_tournament.py` (add `run_tournament`, `parse_pairings`, `main`)
- Test: `tests/test_eval_cli.py` (append tournament tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_cli.py`:

```python
import scripts.GPU.alphazero.eval_checkpoint_tournament as tourney_mod
from scripts.GPU.alphazero.eval_checkpoint_tournament import (
    run_tournament, parse_pairings,
)


def test_parse_pairings_resolves_ids():
    pairs = parse_pairings("0419:0379,0419:0339", "checkpoints/x")
    assert pairs == [
        ("checkpoints/x/model_iter_0419.safetensors",
         "checkpoints/x/model_iter_0379.safetensors"),
        ("checkpoints/x/model_iter_0419.safetensors",
         "checkpoints/x/model_iter_0339.safetensors"),
    ]


def test_run_tournament_calls_run_game_tasks_once(tmp_path, monkeypatch):
    calls = {"n": 0}
    real = tourney_mod.run_game_tasks

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(tourney_mod, "run_game_tasks", counting)

    pairings = [("A", "B"), ("A", "C")]
    out = run_tournament(
        pairings=pairings, games=2, base_seed=0, config=_tiny_cfg(),
        workers=1, output_dir=str(tmp_path),
        evaluator_factory=fake_evaluator_factory,
    )
    assert calls["n"] == 1                       # ONE pool, no nested calls
    assert len(out["pairings"]) == 2
    assert (tmp_path / "tournament.json").exists()


def test_run_tournament_writes_per_pairing_files(tmp_path):
    out = run_tournament(
        pairings=[("A", "B")], games=2, base_seed=0, config=_tiny_cfg(),
        workers=1, output_dir=str(tmp_path),
        evaluator_factory=fake_evaluator_factory,
    )
    assert (tmp_path / "A_vs_B.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_cli.py -k tournament -q`
Expected: FAIL — `cannot import name 'run_tournament'`

- [ ] **Step 3: Add tournament runtime to `eval_checkpoint_tournament.py`**

Append to `scripts/GPU/alphazero/eval_checkpoint_tournament.py`:

```python
import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone

from .eval_runner import EvalConfig, run_game_tasks
from .eval_summary import summarize_tournament


def parse_pairings(spec: str, checkpoints_dir: str):
    """Parse "A:B,A:C" into [(pathA, pathB), ...] resolving short ids."""
    pairings = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        a, b = token.split(":")
        pairings.append((resolve_checkpoint(a.strip(), checkpoints_dir),
                         resolve_checkpoint(b.strip(), checkpoints_dir)))
    if not pairings:
        raise ValueError("no pairings parsed")
    return pairings


def round_robin_pairings(checkpoints):
    """All C(n,2) pairs, preserving input order."""
    pairs = []
    for i in range(len(checkpoints)):
        for j in range(i + 1, len(checkpoints)):
            pairs.append((checkpoints[i], checkpoints[j]))
    return pairs


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def run_tournament(pairings, games, base_seed, config, workers, output_dir,
                   evaluator_factory=None):
    """Run all pairings through ONE flat task list / ONE pool. Returns the
    combined tournament summary and writes per-pairing + combined JSON."""
    tasks = build_tournament_tasks(pairings, games, base_seed)
    results = run_game_tasks(tasks, workers=workers, config=config,
                             evaluator_factory=evaluator_factory)
    config_dict = {**asdict(config), "base_seed": base_seed, "workers": workers}
    summary = summarize_tournament(results, pairings, config_dict)
    summary["git_commit"] = _git_commit()
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(output_dir, exist_ok=True)
    for ps in summary["pairings"]:
        with open(os.path.join(output_dir, f"{ps['pairing_id']}.json"), "w") as fh:
            json.dump(ps, fh, indent=2)
    with open(os.path.join(output_dir, "tournament.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _build_arg_parser():
    ap = argparse.ArgumentParser(description="Run a checkpoint tournament.")
    ap.add_argument("--checkpoints-dir", default="checkpoints/alphazero-v2-staged")
    ap.add_argument("--pairings", default=None,
                    help='e.g. "0419:0379,0419:0339"')
    ap.add_argument("--checkpoints", default=None,
                    help="comma list for --round-robin, e.g. 0419,0379,0339")
    ap.add_argument("--round-robin", action="store_true")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--board-size", type=int, default=24)
    ap.add_argument("--mcts-sims", type=int, default=400)
    ap.add_argument("--mcts-eval-batch-size", type=int, default=14)
    ap.add_argument("--mcts-stall-flush-sims", type=int, default=48)
    ap.add_argument("--selection-mode", default="opening_temperature",
                    choices=["opening_temperature", "argmax"])
    ap.add_argument("--opening-temp-plies", type=int, default=20)
    ap.add_argument("--temp-high", type=float, default=1.0)
    ap.add_argument("--temp-low", type=float, default=0.1)
    ap.add_argument("--max-moves", type=int, default=280)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--base-seed", type=int, default=12345)
    ap.add_argument("--output-dir", required=True)
    return ap


def _config_from_args(args) -> EvalConfig:
    return EvalConfig(
        board_size=args.board_size, mcts_sims=args.mcts_sims,
        mcts_eval_batch_size=args.mcts_eval_batch_size,
        mcts_stall_flush_sims=args.mcts_stall_flush_sims,
        selection_mode=args.selection_mode,
        opening_temp_plies=args.opening_temp_plies,
        temp_high=args.temp_high, temp_low=args.temp_low,
        max_moves=args.max_moves,
    )


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if args.round_robin:
        if not args.checkpoints:
            raise SystemExit("--round-robin requires --checkpoints")
        ckpts = [resolve_checkpoint(t.strip(), args.checkpoints_dir)
                 for t in args.checkpoints.split(",") if t.strip()]
        pairings = round_robin_pairings(ckpts)
    elif args.pairings:
        pairings = parse_pairings(args.pairings, args.checkpoints_dir)
    else:
        raise SystemExit("provide --pairings or --round-robin --checkpoints")

    for a, b in pairings:
        for p in (a, b):
            if not os.path.exists(p):
                raise SystemExit(f"checkpoint not found: {p}")

    summary = run_tournament(
        pairings=pairings, games=args.games, base_seed=args.base_seed,
        config=_config_from_args(args), workers=args.workers,
        output_dir=args.output_dir,
    )
    print(f"{'pairing':<16} {'a_rate':>7} {'elo':>7}  verdict")
    for row in summary["table"]:
        print(f"{row['pairing_id']:<16} {row['a_score_rate']:>7.4f} "
              f"{row['elo_estimate']:>7.1f}  {row['verdict']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_cli.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Run the full eval suite**

Run: `.venv/bin/python -m pytest tests/test_eval_elo.py tests/test_eval_runner.py tests/test_eval_runner_parallel.py tests/test_eval_builders.py tests/test_eval_summary.py tests/test_eval_cli.py -q`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/eval_checkpoint_tournament.py tests/test_eval_cli.py
git commit -m "$(printf 'feat(eval): tournament CLI, one flat pool, grouped output\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 8: Real-checkpoint smokes (integration-marked) + first runs

**Files:**
- Create: `tests/test_eval_real_smoke.py`

> These hit MLX + real checkpoints. They are `@pytest.mark.integration` so the
> default suite (`-m "not integration"`) skips them. Run explicitly with `-m
> integration`.

- [ ] **Step 1: Write the integration smoke test**

Create `tests/test_eval_real_smoke.py`:

```python
import json
import os

import pytest

from scripts.GPU.alphazero.eval_runner import EvalConfig
from scripts.GPU.alphazero.eval_checkpoint_match import run_match

CKPT_DIR = "checkpoints/alphazero-v2-staged"
CKPT_0419 = os.path.join(CKPT_DIR, "model_iter_0419.safetensors")

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.path.exists(CKPT_0419), reason="0419 checkpoint absent")
def test_self_match_color_balance_is_near_even(tmp_path):
    """Sanity gate: a model vs itself, validating color-balanced bookkeeping.

    For a self-match the per-CHECKPOINT score is meaningless (a_ckpt == b_ckpt,
    so winner_checkpoint matches both and a_wins counts every decisive game).
    And red_win_rate is NOT 0.5 either — TwixT has a real first-move (red)
    advantage, so red legitimately wins >50% of decisive games.

    The meaningful, bug-catching metric is A's SIDE-AWARE score: A plays red on
    even game_idx and black on odd, so if color-balancing correctly cancels the
    first-move advantage, A's combined score over both roles must sit near 0.5.
    A gross deviation means a color-assignment / seed / bookkeeping bug.

    Deterministic (fixed base_seed) — pass/fail is stable run-to-run; the wide
    band only tolerates the 20-game sample size. This is a smoke gate, NOT proof
    (see the note below for the 100-200 game manual validation).
    """
    cfg = EvalConfig(board_size=24, mcts_sims=64, max_moves=280)
    out = tmp_path / "self.json"
    summary = run_match(
        a_ckpt=CKPT_0419, b_ckpt=CKPT_0419, games=20, base_seed=12345,
        config=cfg, workers=2, output=str(out),
    )
    assert summary["games"] == 20

    recs = [json.loads(line) for line
            in (tmp_path / "self_games.jsonl").read_text().splitlines()]
    assert len(recs) == 20
    # A's side: red on even game_idx, black on odd.
    a_side_score = sum(
        (r["red_score"] if r["game_idx"] % 2 == 0 else r["black_score"])
        for r in recs
    ) / len(recs)
    assert 0.2 <= a_side_score <= 0.8, (
        f"self-match side-aware score {a_side_score:.3f} far from 0.5 — "
        f"suspect a color-assignment / seed / bookkeeping bug"
    )
```

- [ ] **Step 2: Run the sanity gate explicitly**

Run: `.venv/bin/python -m pytest tests/test_eval_real_smoke.py -m integration -q`
Expected: PASS (1 passed) — A's side-aware self-match score lands in [0.2, 0.8]
(near 0.5 once red's first-move advantage cancels across balanced colors).

> **This is a sanity gate only, not proof of correctness.** With only 20 games
> the CI is wide, so it can pass even with moderate issues. Before trusting any
> tournament output for a real decision, run a same-checkpoint validation of
> **100–200 games manually** and confirm the rate sits near 0.50 with a tight CI.
>
> If this gate FAILS, stop and debug before any real match: the most likely
> causes are a color-assignment bug (A not balanced across red/black) or a
> winner→checkpoint mapping bug. Re-check `build_pairing_tasks` parity and
> `make_result`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_eval_real_smoke.py
git commit -m "$(printf 'test(eval): real-checkpoint 0419-vs-0419 sanity gate (integration)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

- [ ] **Step 4: First real match (manual, not a test)**

Run the first real strength measurement (this is a real run, minutes-to-hours
depending on `--workers`; tune `--workers` to your core count):

```bash
mkdir -p logs/eval
.venv/bin/python -m scripts.GPU.alphazero.eval_checkpoint_match \
  --checkpoint-a checkpoints/alphazero-v2-staged/model_iter_0419.safetensors \
  --checkpoint-b checkpoints/alphazero-v2-staged/model_iter_0379.safetensors \
  --games 400 --board-size 24 \
  --mcts-sims 400 --mcts-eval-batch-size 14 --mcts-stall-flush-sims 48 \
  --selection-mode opening_temperature \
  --opening-temp-plies 20 --temp-high 1.0 --temp-low 0.1 --max-moves 280 \
  --workers 6 --base-seed 12345 \
  --output logs/eval/0419_vs_0379.json
```

Expected: prints `0419_vs_0379: a_score_rate=... elo=... CI95=[...] verdict=...`
and writes `logs/eval/0419_vs_0379.json` + `logs/eval/0419_vs_0379_games.jsonl`.

- [ ] **Step 5: Run the four-pairing tournament (manual)**

```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_checkpoint_tournament \
  --checkpoints-dir checkpoints/alphazero-v2-staged \
  --pairings 0419:0379,0419:0339,0419:0299,0379:0339 \
  --games 400 --board-size 24 \
  --mcts-sims 400 --mcts-eval-batch-size 14 --mcts-stall-flush-sims 48 \
  --selection-mode opening_temperature \
  --opening-temp-plies 20 --temp-high 1.0 --temp-low 0.1 --max-moves 280 \
  --workers 6 --base-seed 12345 \
  --output-dir logs/eval/tournament_0419_anchor/
```

Interpretation (from the spec):
- 0419 beats all clearly → keep training.
- 0419 ≈ 0379 but beats 0339/0299 → plateauing around 0379–0419.
- 0419 loses to 0379 → freeze 0379 as current best.

---

## Parallel MLX worker viability (best-effort)

**Stance:** Parallel MLX worker support is **best-effort**. The evaluator is
REQUIRED to pass and produce correct results with `--workers 1`. `workers>1` is
enabled only after a real-checkpoint viability smoke on the target Mac. If Metal
resource limits appear *at load time*, run tournaments with `--workers 1` — do
NOT block v1. A clean failure (RuntimeError suggesting `--workers 1`) is the
fallback; there is no automatic downgrade.

**On `compile=True`:** the eval path sets `LocalGPUEvaluator(..., compile=True)`,
which fixes/reduces *per-game* MLX/Metal resource churn (the unbounded
graph-retrace that exhausted the ~499k Metal-resource limit after ~2 games). It
does **not** by itself guarantee that multiple MLX worker processes can coexist
— that is a separate, hardware-dependent question answered only by the probe.

**Two distinct failure modes:**
- *At load time* — multiple MLX processes + Metal contexts cannot coexist.
  `compile=True` does NOT fix this. → run `--workers 1`.
- *After a few games* — resource accumulation / repeated graph compilation.
  `compile=True` fixes this. → parallel is viable.

**Viability probe matrix (run before any large `--workers N` tournament):**
```bash
# 1-worker baseline
.venv/bin/python -m scripts.GPU.alphazero.eval_checkpoint_match \
  --checkpoint-a checkpoints/alphazero-v2-staged/model_iter_0419.safetensors \
  --checkpoint-b checkpoints/alphazero-v2-staged/model_iter_0419.safetensors \
  --games 4 --board-size 24 --mcts-sims 64 --mcts-eval-batch-size 14 \
  --mcts-stall-flush-sims 48 --selection-mode argmax --max-moves 280 \
  --workers 1 --base-seed 12345 --output logs/eval/parallel_probe_w1.json
# 2-worker viability probe (same, --workers 2)
#   ... --workers 2 --output logs/eval/parallel_probe_w2.json
```
If `--workers 2` fails immediately at checkpoint load → parallel MLX is not
viable on this machine; run sequentially. If it completes 4 games → step up to
`workers=2, games=20, mcts_sims=400`, then `workers=4`/`6`.

**Empirical result on this Mac (2026-05-31, compile=True in place):** the probe
matrix passed — `--workers 1` finished 4 games in ~41s, `--workers 2` in ~24s
(~1.7× speedup), no Metal exhaustion. The implementer's earlier `workers=2`
failure predated the compile fix (it was the accumulation mode). Parallel is
viable here; other Macs must re-run the probe. A `workers=2` real-checkpoint
viability smoke is included in `tests/test_eval_real_smoke.py` (integration).

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Standalone loop, no training-code changes → Tasks 2/6/7 (new files only). ✓
- Flat task queue + single pool + dynamic counter + spawn → Task 5. ✓
- Per-worker checkpoint cache, loaded once → Task 2 (`_make_cache`) + cache test; Task 5 worker reuse. ✓
- Task-derived seeds, worker-independent → Task 3 builders + Task 5 equivalence test. ✓
- No adjudication / no resign; state-cap = draw, scored 0.5 → `play_eval_game` reasons + `summarize_match`. ✓
- Opening-temperature default + argmax smoke mode → `cfg_from` + `--selection-mode`. ✓
- Red-first, balance-the-model colors → `build_pairing_tasks` parity. ✓
- Draw-aware trinomial Elo CI + verdict + by-color + color-bias → `eval_elo` + `summarize_match`. ✓
- Match + tournament CLIs, JSONL + summary outputs, cross-format (auto-detect in loader) → Tasks 6/7. ✓
- Defaults 280 / 400 → `EvalConfig` + CLI defaults. ✓
- selection_mode + draw_score_policy output fields → `summarize_match`. ✓
- All 9 spec tests → elo (T1), determinism/cache/bookkeeping (T2), builders balance/seed (T3), summary (T4), workers=1-vs-2 + no-nested-pool (T5/T7), fake e2e smoke (T6), real sanity (T8). ✓

**Placeholder scan:** No TBD/TODO; every code/test step has complete content. ✓

**Type consistency:** `EvalGameTask`/`EvalGameResult`/`EvalConfig` field names and `cfg_from`/`play_eval_game`/`make_result`/`build_pairing_tasks`/`run_game_tasks`/`summarize_match`/`summarize_tournament` signatures are used identically across tasks. `short_id`/`resolve_checkpoint` defined in Task 3, imported by `eval_summary` (T4) and match CLI (T6). ✓

**Known forward-reference:** Task 2 ships `run_game_tasks` calling `_run_parallel`, which Task 5 defines. Documented inline; all intervening tests use `workers=1`. ✓
