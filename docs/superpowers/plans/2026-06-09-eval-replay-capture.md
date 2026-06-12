# Eval Replay Capture (V2 Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in per-ply replay capture to the checkpoint-eval match path (moves + value/search stats written to a sidecar, linked from each `*_games.jsonl` row), without changing game outcomes or scoring.

**Architecture:** A new pure `eval_replay.py` (record/dict construction + one sidecar writer) feeds an extended `eval_runner.play_eval_game` (optional `capture`) and an extended `EvalGameResult` (`replay_path`). `run_game_tasks` threads a `replay_dir` through the sequential and spawn-worker paths; each worker writes its own per-game sidecar (move records never cross the queue). `eval_checkpoint_match.py` gains `--save-eval-replays` / `--replay-dir`.

**Tech Stack:** Python 3.14, stdlib (`json`, `os`), pytest. No MLX in the pure module or its tests. Reuses `tests.eval_fakes` (`FakeEvaluator`, `fake_evaluator_factory`).

**Spec:** `docs/superpowers/specs/2026-06-09-eval-replay-capture-design.md`

**Run tests with:** `.venv/bin/python -m pytest <path> -v`

**Pre-commit note:** the git hook prints a wall of unrelated vendored-JS ESLint warnings, then proceeds — pre-existing noise, not caused by these changes. Commits succeed. Don't touch ESLint.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `scripts/GPU/alphazero/eval_replay.py` | **New.** Pure per-ply record + replay-dict builders, filename helper, and the single sidecar writer. No game engine, no MLX. |
| `scripts/GPU/alphazero/eval_runner.py` | **Modify.** `EvalGameResult.replay_path`; `make_result` param; `play_eval_game(capture=…)` → 4-tuple; thread `replay_dir` through `run_game_tasks`/`_run_sequential`/`_run_parallel`/`_worker_main`. |
| `scripts/GPU/alphazero/eval_checkpoint_match.py` | **Modify.** `--save-eval-replays` / `--replay-dir`, `replay_dir_for` helper, `run_match(replay_dir=…)`. |
| `tests/test_eval_replay.py` | **New.** Pure-module unit tests. |
| `tests/test_eval_runner.py` | **Modify.** Capture + replay_dir tests; fix the one 3-tuple unpack. |
| `tests/test_eval_match_replay.py` | **New.** Match-level plumbing tests (run_match + replay_dir_for). |

---

## Task 1: `eval_replay.ply_record` (pure)

**Files:**
- Create: `scripts/GPU/alphazero/eval_replay.py`
- Test: `tests/test_eval_replay.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_replay.py`:

```python
import pytest

from scripts.GPU.alphazero.eval_replay import ply_record, REPLAY_SCHEMA_VERSION


def test_ply_record_fields():
    counts = {(4, 19): 124, (5, 5): 76, (1, 1): 200}
    rec = ply_record(0, "red", (4, 19), counts, root_value=0.12)
    assert rec == {
        "ply": 0, "player": "red", "row": 4, "col": 19,
        "root_value": 0.12,
        "root_top1_share": 200 / 400,
        "selected_visit_rank": 2,        # 200 > 124 > 76 -> (4,19) is rank 2
        "selected_visit_count": 124,
        "root_total_visits": 400,
        "n_legal": 3,
    }


def test_ply_record_rank_tiebreak_by_rowcol():
    # two moves tie at 100 visits; ascending (row,col) breaks the tie
    counts = {(2, 2): 100, (1, 9): 100, (0, 0): 50}
    # (1,9) and (2,2) tie at 100; (1,9) sorts before (2,2) -> ranks 1 and 2
    assert ply_record(0, "red", (1, 9), counts, 0.0)["selected_visit_rank"] == 1
    assert ply_record(0, "red", (2, 2), counts, 0.0)["selected_visit_rank"] == 2


def test_ply_record_top1_and_totals():
    counts = {(0, 0): 3, (0, 1): 7}
    rec = ply_record(5, "black", (0, 0), counts, -0.4)
    assert rec["root_total_visits"] == 10
    assert rec["root_top1_share"] == 0.7
    assert rec["selected_visit_count"] == 3
    assert rec["selected_visit_rank"] == 2


def test_ply_record_fails_on_empty_counts():
    with pytest.raises(ValueError, match="empty"):
        ply_record(0, "red", (4, 19), {}, 0.0)


def test_ply_record_fails_when_move_not_in_counts():
    with pytest.raises(ValueError, match="not in"):
        ply_record(0, "red", (9, 9), {(4, 19): 10}, 0.0)


def test_schema_version_is_one():
    assert REPLAY_SCHEMA_VERSION == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.GPU.alphazero.eval_replay'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/GPU/alphazero/eval_replay.py`:

```python
"""Replay capture for checkpoint-eval games.

Pure per-ply / per-game record construction plus a single sidecar writer. No
game engine, no MLX. Coordinates are engine-native (row, col) — no x/y
conversion is performed in Phase A. A replay sidecar links from each
*_games.jsonl row via replay_path.
"""
from __future__ import annotations

import json
import os

REPLAY_SCHEMA_VERSION = 1


def ply_record(ply, player, move, counts, root_value):
    """One per-ply replay record.

    `move` is the selected (row, col). `counts` is the MCTS visit-count dict
    {(row, col): visits} over all legal moves at this root. `root_value` is
    root.q_value from the perspective of `player` (the side about to move),
    before the move is applied. Fail loud rather than emit a corrupt record.
    """
    if not counts:
        raise ValueError(f"ply {ply}: empty visit counts")
    if move not in counts:
        raise ValueError(f"ply {ply}: selected move {move} not in visit counts")
    total = sum(counts.values())
    # rank: descending visit count, ties broken by ascending (row, col).
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    rank = 1 + next(i for i, (m, _c) in enumerate(ranked) if m == move)
    row, col = move
    return {
        "ply": ply,
        "player": player,
        "row": row,
        "col": col,
        "root_value": root_value,
        "root_top1_share": max(counts.values()) / total,
        "selected_visit_rank": rank,
        "selected_visit_count": counts[move],
        "root_total_visits": total,
        "n_legal": len(counts),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_replay.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_replay.py tests/test_eval_replay.py
git commit -m "feat(eval): per-ply replay record builder"
```

---

## Task 2: `build_replay_dict` + `replay_filename` (pure)

**Files:**
- Modify: `scripts/GPU/alphazero/eval_replay.py`
- Test: `tests/test_eval_replay.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_replay.py`:

```python
from dataclasses import dataclass

from scripts.GPU.alphazero.eval_replay import build_replay_dict, replay_filename


@dataclass
class _FakeResult:
    pairing_id: str
    game_idx: int
    task_id: int
    red_checkpoint: str
    black_checkpoint: str
    winner: str
    winner_checkpoint: str
    reason: str
    n_moves: int


def test_build_replay_dict_shape():
    result = _FakeResult("0399_vs_0379", 3, 7, "A.safetensors", "B.safetensors",
                         "red", "A.safetensors", "win", 2)
    records = [
        {"ply": 0, "player": "red", "row": 4, "col": 19, "root_value": 0.1,
         "root_top1_share": 0.5, "selected_visit_rank": 1,
         "selected_visit_count": 5, "root_total_visits": 10, "n_legal": 3},
        {"ply": 1, "player": "black", "row": 1, "col": 1, "root_value": -0.1,
         "root_top1_share": 0.6, "selected_visit_rank": 1,
         "selected_visit_count": 6, "root_total_visits": 10, "n_legal": 2},
    ]
    d = build_replay_dict(result, seed=35791, board_size=24, records=records)
    assert d == {
        "schema_version": 1,
        "pairing_id": "0399_vs_0379",
        "game_idx": 3, "task_id": 7, "seed": 35791, "board_size": 24,
        "red_checkpoint": "A.safetensors", "black_checkpoint": "B.safetensors",
        "winner": "red", "winner_checkpoint": "A.safetensors",
        "reason": "win", "n_moves": 2,
        "moves": records,
    }


def test_replay_filename_zero_padded():
    assert replay_filename(0) == "game_000000.json"
    assert replay_filename(42) == "game_000042.json"
    assert replay_filename(123456) == "game_123456.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_replay.py -k "build_replay_dict or replay_filename" -v`
Expected: FAIL with `ImportError: cannot import name 'build_replay_dict'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/GPU/alphazero/eval_replay.py`:

```python
def build_replay_dict(result, seed, board_size, records):
    """Assemble the replay sidecar dict from a finished EvalGameResult plus the
    per-ply records. Reads identity/outcome from `result`; `seed` and
    `board_size` complete the contract."""
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "pairing_id": result.pairing_id,
        "game_idx": result.game_idx,
        "task_id": result.task_id,
        "seed": seed,
        "board_size": board_size,
        "red_checkpoint": result.red_checkpoint,
        "black_checkpoint": result.black_checkpoint,
        "winner": result.winner,
        "winner_checkpoint": result.winner_checkpoint,
        "reason": result.reason,
        "n_moves": result.n_moves,
        "moves": records,
    }


def replay_filename(game_idx):
    return f"game_{game_idx:06d}.json"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_replay.py -k "build_replay_dict or replay_filename" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_replay.py tests/test_eval_replay.py
git commit -m "feat(eval): replay-dict assembly + filename helper"
```

---

## Task 3: `write_replay` (IO)

**Files:**
- Modify: `scripts/GPU/alphazero/eval_replay.py`
- Test: `tests/test_eval_replay.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_replay.py`:

```python
import json as _json
import os as _os

from scripts.GPU.alphazero.eval_replay import write_replay


def test_write_replay_roundtrip_and_relative_path(tmp_path):
    replay_dir = tmp_path / "m_replays"
    d = {"schema_version": 1, "game_idx": 5, "moves": []}
    path = write_replay(str(replay_dir), d)
    # returns a path relative to CWD, not absolute
    assert not _os.path.isabs(path)
    # file exists where expected and round-trips
    abs_path = replay_dir / "game_000005.json"
    assert abs_path.exists()
    assert _json.loads(abs_path.read_text()) == d


def test_write_replay_creates_dir_idempotently(tmp_path):
    replay_dir = tmp_path / "nested" / "replays"
    write_replay(str(replay_dir), {"game_idx": 0, "moves": []})
    # second write into the same (now-existing) dir must not raise
    write_replay(str(replay_dir), {"game_idx": 1, "moves": []})
    assert (replay_dir / "game_000000.json").exists()
    assert (replay_dir / "game_000001.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_replay.py -k write_replay -v`
Expected: FAIL with `ImportError: cannot import name 'write_replay'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/GPU/alphazero/eval_replay.py`:

```python
def write_replay(replay_dir, replay_dict):
    """Write one game sidecar; return its path relative to the process CWD.

    Worker-safe: makedirs(exist_ok=True) tolerates concurrent creation by other
    worker processes writing into the same replay_dir.
    """
    os.makedirs(replay_dir, exist_ok=True)
    path = os.path.join(replay_dir, replay_filename(replay_dict["game_idx"]))
    with open(path, "w") as fh:
        json.dump(replay_dict, fh)
    return os.path.relpath(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_replay.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_replay.py tests/test_eval_replay.py
git commit -m "feat(eval): worker-safe replay sidecar writer"
```

---

## Task 4: `play_eval_game` capture + `EvalGameResult.replay_path` + `make_result`

**Files:**
- Modify: `scripts/GPU/alphazero/eval_runner.py:38-51` (dataclass), `:96-142` (play_eval_game, make_result), imports
- Test: `tests/test_eval_runner.py`

- [ ] **Step 1: Write the failing tests (and fix the 3-tuple unpack)**

In `tests/test_eval_runner.py`, **change** the existing line that unpacks `play_eval_game` (currently `winner, reason, n = play_eval_game(...)`) to a 4-tuple:

```python
def test_play_eval_game_reason_is_valid():
    winner, reason, n, _records = play_eval_game(FakeEvaluator(), FakeEvaluator(),
                                                 _tiny_cfg(), seed=1)
    assert reason in {"win", "state_cap", "board_full", "unknown_error"}
    assert reason != "unknown_error"
    assert n >= 1
```

Then append these tests to `tests/test_eval_runner.py`:

```python
def test_play_eval_game_capture_off_returns_none():
    *_head, records = play_eval_game(FakeEvaluator(), FakeEvaluator(),
                                     _tiny_cfg(), seed=1)
    assert records is None


def test_play_eval_game_capture_records_one_per_ply():
    winner, reason, n, records = play_eval_game(
        FakeEvaluator(), FakeEvaluator(), _tiny_cfg(), seed=1, capture=True)
    assert records is not None
    assert len(records) == n
    assert [r["ply"] for r in records] == list(range(n))
    players = [r["player"] for r in records]
    assert players[0] == "red"                                  # red moves first
    assert all(players[i] != players[i + 1] for i in range(len(players) - 1))
    for r in records:
        assert set(r) == {"ply", "player", "row", "col", "root_value",
                          "root_top1_share", "selected_visit_rank",
                          "selected_visit_count", "root_total_visits", "n_legal"}
        assert 0.0 <= r["root_top1_share"] <= 1.0
        assert r["selected_visit_rank"] >= 1
        assert r["selected_visit_count"] >= 1
        assert r["n_legal"] >= 1


def test_play_eval_game_capture_does_not_change_outcome():
    off = play_eval_game(FakeEvaluator(), FakeEvaluator(), _tiny_cfg(), seed=7)
    on = play_eval_game(FakeEvaluator(), FakeEvaluator(), _tiny_cfg(),
                        seed=7, capture=True)
    assert off[:3] == on[:3]   # winner, reason, n_moves identical regardless of capture


def test_make_result_default_replay_path_is_none():
    task = EvalGameTask(0, "p", 0, "A.safetensors", "B.safetensors", 7)
    assert make_result(task, "red", "win", 40).replay_path is None


def test_make_result_sets_replay_path():
    task = EvalGameTask(0, "p", 0, "A.safetensors", "B.safetensors", 7)
    res = make_result(task, "red", "win", 40, "logs/x_replays/game_000000.json")
    assert res.replay_path == "logs/x_replays/game_000000.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_runner.py -k "capture or replay_path" -v`
Expected: FAIL — `play_eval_game()` got an unexpected keyword `capture` / `make_result()` takes 4 positional args / `EvalGameResult` has no `replay_path`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/GPU/alphazero/eval_runner.py`, add the import near the other relative imports (after `from .mcts import MCTS, MCTSConfig`):

```python
from .eval_replay import ply_record, build_replay_dict, write_replay
```

Add the field to `EvalGameResult` (after `black_score: float`):

```python
    replay_path: Optional[str] = None
```

Replace `play_eval_game` (the function body at `eval_runner.py:96`) with:

```python
def play_eval_game(red_eval, black_eval, config: EvalConfig, seed: int,
                   capture: bool = False):
    """Play one A-vs-B game. Returns (winner, reason, n_moves, records).

    `records` is None unless capture=True, in which case it is a list of
    ply_record dicts (one per ply). Capturing reads already-computed search
    outputs only — no extra search calls, no RNG draws — so game outcomes are
    identical with capture on or off.
    """
    mcts_red = MCTS(red_eval, cfg_from(config), random.Random(seed ^ 0xA5A5A5))
    mcts_black = MCTS(black_eval, cfg_from(config), random.Random(seed ^ 0x5A5A5A))
    state = TwixtState(active_size=config.board_size, to_move="red",
                       max_plies_limit=config.max_moves)
    ply = 0
    records = [] if capture else None
    while state.winner() is None and ply < config.max_moves and state.legal_moves():
        mcts = mcts_red if state.to_move == "red" else mcts_black
        counts, root_value = mcts.search(state, add_noise=False)
        move = mcts.select_move(counts, ply)
        if capture:
            records.append(ply_record(ply, state.to_move, move, counts, root_value))
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
    return winner, reason, ply, records
```

Change the `make_result` signature/return (at `eval_runner.py:129`) to thread `replay_path`:

```python
def make_result(task: EvalGameTask, winner, reason, n_moves,
                replay_path=None) -> EvalGameResult:
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
        replay_path=replay_path,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_runner.py -v`
Expected: PASS (all — the new capture/replay tests plus the unchanged determinism test at the 4-tuple).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_runner.py tests/test_eval_runner.py
git commit -m "feat(eval): optional per-ply capture in play_eval_game; replay_path on result"
```

---

## Task 5: thread `replay_dir` through `run_game_tasks` and write sidecars

**Files:**
- Modify: `scripts/GPU/alphazero/eval_runner.py:223-345` (sequential, worker, parallel, run_game_tasks)
- Test: `tests/test_eval_runner.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_runner.py`:

```python
def test_run_game_tasks_no_replay_dir_leaves_replay_path_none():
    tasks = [EvalGameTask(0, "p", 0, "A", "B", 100),
             EvalGameTask(1, "p", 1, "B", "A", 101)]
    out = run_game_tasks(tasks, workers=1, config=_tiny_cfg(),
                         evaluator_factory=fake_evaluator_factory)
    assert all(r.replay_path is None for r in out)


def test_run_game_tasks_replay_dir_writes_one_sidecar_per_game(tmp_path):
    rd = tmp_path / "replays"
    tasks = [EvalGameTask(0, "p", 0, "A", "B", 100),
             EvalGameTask(1, "p", 1, "B", "A", 101)]
    out = run_game_tasks(tasks, workers=1, config=_tiny_cfg(),
                         evaluator_factory=fake_evaluator_factory,
                         replay_dir=str(rd))
    assert all(r.replay_path is not None for r in out)
    for r in out:
        assert (rd / f"game_{r.game_idx:06d}.json").exists()


def test_run_game_tasks_capture_does_not_change_results(tmp_path):
    tasks = [EvalGameTask(0, "p", 0, "A", "B", 100),
             EvalGameTask(1, "p", 1, "B", "A", 101)]
    off = run_game_tasks(tasks, workers=1, config=_tiny_cfg(),
                         evaluator_factory=fake_evaluator_factory)
    on = run_game_tasks(tasks, workers=1, config=_tiny_cfg(),
                        evaluator_factory=fake_evaluator_factory,
                        replay_dir=str(tmp_path / "r"))

    def fields(r):  # every pre-replay field
        return (r.game_idx, r.task_id, r.pairing_id, r.winner, r.winner_checkpoint,
                r.reason, r.n_moves, r.red_score, r.black_score,
                r.red_checkpoint, r.black_checkpoint)

    assert [fields(r) for r in off] == [fields(r) for r in on]


def test_run_game_tasks_replay_dir_parallel_writes_sidecars(tmp_path):
    rd = tmp_path / "replays_par"
    tasks = [EvalGameTask(0, "p", 0, "A", "B", 100),
             EvalGameTask(1, "p", 1, "B", "A", 101)]
    out = run_game_tasks(tasks, workers=2, config=_tiny_cfg(),
                         evaluator_factory=fake_evaluator_factory,
                         replay_dir=str(rd))
    assert all(r.replay_path is not None for r in out)
    for r in out:
        assert (rd / f"game_{r.game_idx:06d}.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_runner.py -k "replay_dir or capture_does_not_change" -v`
Expected: FAIL — `run_game_tasks()` got an unexpected keyword `replay_dir`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/GPU/alphazero/eval_runner.py`, replace `_run_sequential` (`:223`) with:

```python
def _run_sequential(tasks, config, factory, replay_dir=None):
    import gc

    import mlx.core as mx

    capture = replay_dir is not None
    get_eval = _make_cache(factory)
    results = []
    for task in tasks:
        red = get_eval(task.red_checkpoint)
        black = get_eval(task.black_checkpoint)
        winner, reason, nm, records = play_eval_game(
            red, black, config, task.seed, capture=capture)
        result = make_result(task, winner, reason, nm)
        if records is not None:
            result.replay_path = write_replay(
                replay_dir,
                build_replay_dict(result, task.seed, config.board_size, records))
        results.append(result)
        # Flush pending MLX lazy ops and release cached Metal buffers between
        # games to stay within Metal's resource limit (trainer.py:3169-3173).
        mx.eval()
        gc.collect()
        mx.clear_cache()
    return _sorted(results)
```

Replace `_worker_main` (`:243`) with (adds the trailing `replay_dir` arg + the write):

```python
def _worker_main(worker_id, tasks, config, factory, next_idx, result_q,
                 replay_dir=None):
    """Pull tasks via the shared atomic counter; per-process checkpoint cache.

    On any exception, send a _WorkerFailed sentinel so the parent fails
    promptly instead of waiting out the stall timeout.
    """
    import traceback
    capture = replay_dir is not None
    get_eval = _make_cache(factory)
    n = len(tasks)
    try:
        while True:
            with next_idx.get_lock():
                i = next_idx.value
                if i >= n:
                    break
                next_idx.value = i + 1
            task = tasks[i]
            red = get_eval(task.red_checkpoint)
            black = get_eval(task.black_checkpoint)
            winner, reason, nm, records = play_eval_game(
                red, black, config, task.seed, capture=capture)
            result = make_result(task, winner, reason, nm)
            if records is not None:
                result.replay_path = write_replay(
                    replay_dir,
                    build_replay_dict(result, task.seed, config.board_size, records))
            result_q.put(result)
    except Exception as e:
        result_q.put(_WorkerFailed(worker_id, f"{e!r}\n{traceback.format_exc()}"))
        return
    result_q.put(_WorkerDone(worker_id))
```

Replace `_run_parallel`'s signature and the `Process` args (`:270-281`) so `replay_dir` reaches each worker:

```python
def _run_parallel(tasks, workers, config, factory, replay_dir=None):
    """Spawn pool (macOS-mandatory). Shared next-task counter, results via
    queue, explicit WorkerDone, parent joins with timeout (no silent hang).
    A _WorkerFailed sentinel surfaces a crashed worker promptly."""
    ctx = mp.get_context("spawn")
    next_idx = ctx.Value("i", 0)
    result_q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker_main,
                    args=(wid, tasks, config, factory, next_idx, result_q,
                          replay_dir))
        for wid in range(workers)
    ]
```

(Leave the rest of `_run_parallel`'s body — `p.start()` onward — unchanged.)

Replace the `run_game_tasks` signature + dispatch (`:328-345`) with:

```python
def run_game_tasks(tasks, workers: int, config: EvalConfig,
                   evaluator_factory: Optional[EvaluatorFactory] = None,
                   replay_dir: Optional[str] = None):
    """Execute tasks; return results sorted by (pairing_id, game_idx).

    workers<=1 runs in-process. workers>1 uses a spawn worker pool with a
    shared atomic task counter (dynamic work-stealing).

    When replay_dir is set, each game writes a per-ply replay sidecar into it
    (worker-safe: each game writes its own file) and the result's replay_path
    is filled in; otherwise replay_path stays None.

    NOTE: when workers>1, evaluator_factory must be a MODULE-LEVEL picklable
    callable (it is sent to spawned workers). Lambdas/closures will fail to
    pickle. The default real loader and the test fakes satisfy this.
    """
    factory = evaluator_factory or _default_evaluator_factory
    if not tasks:
        return []
    workers = min(workers, len(tasks))
    if workers <= 1:
        return _run_sequential(tasks, config, factory, replay_dir)
    return _run_parallel(tasks, workers, config, factory, replay_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_runner.py -v`
Expected: PASS (all, including the workers=2 sidecar test).

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_runner.py tests/test_eval_runner.py
git commit -m "feat(eval): thread replay_dir through runner; write per-game sidecars"
```

---

## Task 6: `eval_checkpoint_match` CLI flags + `run_match` plumbing

**Files:**
- Modify: `scripts/GPU/alphazero/eval_checkpoint_match.py:47-61` (run_match), `:64-85` (parser), `:100-109` (main)
- Test: `tests/test_eval_match_replay.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_match_replay.py`:

```python
import json

from scripts.GPU.alphazero.eval_checkpoint_match import run_match, replay_dir_for
from scripts.GPU.alphazero.eval_runner import EvalConfig
from tests.eval_fakes import fake_evaluator_factory


def _tiny_cfg():
    return EvalConfig(board_size=8, mcts_sims=8, mcts_eval_batch_size=4,
                      mcts_stall_flush_sims=4, opening_temp_plies=4,
                      temp_high=1.0, temp_low=0.1, max_moves=12)


def test_replay_dir_for_off_returns_none():
    assert replay_dir_for("logs/eval/m.json", None, False) is None


def test_replay_dir_for_default_derives_from_output_stem():
    assert replay_dir_for("logs/eval/m.json", None, True) == "logs/eval/m_replays"


def test_replay_dir_for_explicit_overrides_default():
    assert replay_dir_for("logs/eval/m.json", "/tmp/rr", True) == "/tmp/rr"


def test_run_match_without_replays_leaves_replay_path_null(tmp_path):
    out = tmp_path / "m.json"
    run_match("A", "B", games=2, base_seed=5, config=_tiny_cfg(), workers=1,
              output=str(out), evaluator_factory=fake_evaluator_factory)
    rows = [json.loads(line) for line in (tmp_path / "m_games.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(r["replay_path"] is None for r in rows)
    assert not (tmp_path / "m_replays").exists()


def test_run_match_with_replays_writes_sidecars_and_links(tmp_path):
    out = tmp_path / "m.json"
    rd = str(tmp_path / "m_replays")
    run_match("A", "B", games=2, base_seed=5, config=_tiny_cfg(), workers=1,
              output=str(out), evaluator_factory=fake_evaluator_factory,
              replay_dir=rd)
    rows = [json.loads(line) for line in (tmp_path / "m_games.jsonl").read_text().splitlines()]
    assert all(r["replay_path"] for r in rows)
    for r in rows:
        assert (tmp_path / "m_replays" / f"game_{r['game_idx']:06d}.json").exists()
    rep = json.loads((tmp_path / "m_replays" / "game_000000.json").read_text())
    assert rep["schema_version"] == 1
    assert len(rep["moves"]) == rep["n_moves"]
    assert rep["seed"] == 5  # base_seed + offset(0) + game_idx(0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_eval_match_replay.py -v`
Expected: FAIL — `ImportError: cannot import name 'replay_dir_for'` / `run_match()` got an unexpected keyword `replay_dir`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/GPU/alphazero/eval_checkpoint_match.py`, add the helper (above `run_match`):

```python
def replay_dir_for(output, replay_dir_arg, save_enabled):
    """Resolve the replay output dir. None when capture is off; else the
    explicit --replay-dir, else <output-stem>_replays."""
    if not save_enabled:
        return None
    if replay_dir_arg:
        return replay_dir_arg
    stem, _ext = os.path.splitext(output)
    return f"{stem}_replays"
```

Change `run_match` to accept and forward `replay_dir`:

```python
def run_match(a_ckpt, b_ckpt, games, base_seed, config, workers, output,
              pairing_id=None, evaluator_factory=None, replay_dir=None):
    """Run a full match and write outputs. Returns the summary dict."""
    if pairing_id is None:
        pairing_id = f"{short_id(a_ckpt)}_vs_{short_id(b_ckpt)}"
    tasks = build_match_tasks(a_ckpt, b_ckpt, games, base_seed, pairing_id)
    results = run_game_tasks(tasks, workers=workers, config=config,
                             evaluator_factory=evaluator_factory,
                             replay_dir=replay_dir)
    config_dict = {**asdict(config), "base_seed": base_seed, "workers": workers}
    summary = summarize_match(results, a_ckpt, b_ckpt, pairing_id, config_dict)
    summary["git_commit"] = _git_commit()
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    if output:
        _write_outputs(output, summary, results)
    return summary
```

Add the two args in `_build_arg_parser` (after `--base-seed`, before `--output`):

```python
    ap.add_argument("--save-eval-replays", action="store_true",
                    help="write a per-ply replay sidecar per game and link it "
                         "from each *_games.jsonl row (default off).")
    ap.add_argument("--replay-dir", default=None,
                    help="replay output dir (default <output-stem>_replays); "
                         "only used with --save-eval-replays.")
```

Wire it into `main` (replace the `run_match(...)` call):

```python
    replay_dir = replay_dir_for(args.output, args.replay_dir, args.save_eval_replays)
    summary = run_match(
        a_ckpt=args.checkpoint_a, b_ckpt=args.checkpoint_b, games=args.games,
        base_seed=args.base_seed, config=_config_from_args(args),
        workers=args.workers, output=args.output, replay_dir=replay_dir,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_match_replay.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_checkpoint_match.py tests/test_eval_match_replay.py
git commit -m "feat(eval): --save-eval-replays / --replay-dir in checkpoint match"
```

---

## Task 7: Full-suite verification + operator capture run

**Files:** none (verification only)

- [ ] **Step 1: Run the full eval/replay test set**

Run: `.venv/bin/python -m pytest tests/test_eval_replay.py tests/test_eval_runner.py tests/test_eval_match_replay.py -v`
Expected: PASS (all). Then a broader no-regression check:
Run: `.venv/bin/python -m pytest tests/ -q -k "eval or replay"`
Expected: no NEW failures vs the pre-existing baseline.

- [ ] **Step 2: Confirm backward-compat of the V1 analyzer on a capture-disabled file**

The V1 analyzer must still accept rows that now carry a `replay_path` key. Quick check (uses an existing non-replay file, which has no replay_path key — confirms `validate_rows` tolerates the schema either way):

Run:
```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_loss_analyzer \
  --games-jsonl logs/eval/eps035_0399_vs_0379_800g_w4_games.jsonl \
  --output-dir logs/eval/loss_analysis
```
Expected: runs clean (no `validate_rows` error).

- [ ] **Step 3: Operator-run the eps035 capture (manual — real MLX, long-running)**

This step is operator-executed, not an automated test. It reuses base_seed 35791 so the engine reproduces the exact 800 games V1 analyzed, now with replays:

```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_checkpoint_match \
  --checkpoint-a checkpoints/alphazero-v2-eps035-from0379/model_iter_0399.safetensors \
  --checkpoint-b checkpoints/alphazero-v2-staged/model_iter_0379.safetensors \
  --games 800 --board-size 24 \
  --mcts-sims 400 --mcts-eval-batch-size 14 --mcts-stall-flush-sims 48 \
  --selection-mode opening_temperature \
  --opening-temp-plies 20 --temp-high 1.0 --temp-low 0.1 --max-moves 280 \
  --workers 4 --base-seed 35791 \
  --save-eval-replays \
  --output logs/eval/eps035_0399_vs_0379_800g_w4_seed35791_replay.json
```
(`--workers 4` is hardware-dependent per the MLX/Metal resource gotcha; fall back to `--workers 1` on a stall.) Expected artifacts: `..._seed35791_replay_games.jsonl` (rows carry `replay_path`), `..._seed35791_replay_replays/game_000000.json …`, and the summary JSON.

- [ ] **Step 4: Spot-check the captured data**

Run:
```bash
.venv/bin/python -c "
import json, glob, os
stem='logs/eval/eps035_0399_vs_0379_800g_w4_seed35791_replay'
rows=[json.loads(l) for l in open(stem+'_games.jsonl')]
print('rows', len(rows), 'with replay_path', sum(bool(r.get('replay_path')) for r in rows))
files=glob.glob(stem+'_replays/game_*.json')
print('replay files', len(files))
rep=json.load(open(sorted(files)[0]))
print('schema', rep['schema_version'], 'moves==n_moves', len(rep['moves'])==rep['n_moves'])
m=rep['moves'][0]
print('first ply keys', sorted(m))
print('seed', rep['seed'])
"
```
Expected: 800 rows all with `replay_path`; 800 replay files; `schema 1`, `moves==n_moves True`; per-ply keys include `root_value`, `root_top1_share`, `selected_visit_rank`, `selected_visit_count`, `root_total_visits`, `n_legal`; `seed 35791`.

---

## Self-Review

**Spec coverage:**
- Pure `eval_replay.py` (ply_record / build_replay_dict / replay_filename / write_replay) → Tasks 1–3. ✓
- `EvalGameResult.replay_path`, `make_result`, `play_eval_game(capture=…)` 4-tuple → Task 4. ✓
- Thread `replay_dir` through `run_game_tasks`/sequential/parallel/worker; per-game sidecar; worker-safe → Task 5. ✓
- Rich-minus-policy_rank schema incl. `selected_visit_count` + `root_total_visits`; `(-count,(row,col))` tie-break; fail-loud edges → Tasks 1 (record) + 2 (dict). ✓
- `--save-eval-replays` / `--replay-dir` (default `<stem>_replays`) → Task 6. ✓
- `replay_path` relative to CWD → Task 3 (`os.path.relpath`), asserted in Task 3 test. ✓
- Determinism (all pre-replay fields, except replay_path) → Task 4 (`play_eval_game`) + Task 5 (`run_game_tasks`). ✓
- Scoring unchanged → Task 5 result-equality test; V1-analyzer backward-compat → Task 7 Step 2. ✓
- eps035 seed-35791 capture run + output name → Task 7 Step 3. ✓
- Out of scope (analyzer / tournament / selected_policy_rank) → none built. ✓

**Placeholder scan:** none — every code/command step is complete.

**Type/name consistency:** `ply_record`, `build_replay_dict`, `replay_filename`, `write_replay`, `REPLAY_SCHEMA_VERSION`, `replay_dir_for`, `replay_path`, `replay_dir`, `capture` used consistently across module, runner, and CLI. `play_eval_game` returns a 4-tuple everywhere (both internal call sites in Task 5, the test fix in Task 4). `make_result(..., replay_path=None)` matches its callers. `build_replay_dict(result, seed, board_size, records)` matches both call sites.
