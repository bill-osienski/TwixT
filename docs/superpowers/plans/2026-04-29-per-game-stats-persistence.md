# Per-Game Stats Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist `worker_id`, `wall_time_s`, MCTS compute counters, `adjudication_block_reason`, and two new MCTS end-state fields (`final_root_value`, `final_top1_share`) per game in `scripts/GPU/logs/games/iter_NNNN_game_NNN.json`, so post-hoc analysis no longer depends on consuming `iter_NNNN_stats.json` aggregates or trainer in-memory state.

**Architecture:** Three planes touched in narrow, well-bounded places. **Plane 1 — MCTS observation:** add two `__init__` attributes and one private helper to `MCTS`; call the helper at the end of both `search()` and `search_from_root()`. **Plane 2 — record types:** add new fields to `self_play.GameRecord` (in-process path) and `ipc_messages.GameComplete` (worker-IPC path), populated from the MCTS instance. **Plane 3 — JSON write:** extend `save_game_replay` and `GameSaver.maybe_save_game` with eight new kwargs; extract two private routing helpers (`_save_game_from_ipc`, `_save_game_from_record`) in `trainer.py` to give the two save call sites a testable seam.

**Tech Stack:** Python 3.14, MLX (Apple Silicon GPU), pytest, dataclasses, multiprocessing IPC. Tests use `tmp_path` fixture and the existing `LocalGPUEvaluator` + `create_network` MCTS test pattern.

**Spec:** `docs/superpowers/specs/2026-04-29-per-game-stats-persistence-design.md`

---

## File Structure

| File | Type | Responsibility |
|---|---|---|
| `scripts/GPU/alphazero/mcts.py` | modify | Add `_final_root_value` / `_final_top1_share` instance attrs + `_capture_final_root_stats` helper. Call helper at end of both search methods. |
| `scripts/GPU/alphazero/self_play.py` | modify | Add `wall_time_s`, `final_root_value`, `final_top1_share` to `GameRecord`. Measure wall time in `play_game()`. Add `import time`. |
| `scripts/GPU/alphazero/ipc_messages.py` | modify | Add two new optional fields (`final_root_value`, `final_top1_share`) to `GameComplete`. |
| `scripts/GPU/alphazero/self_play_worker.py` | modify | Pass the two new fields into `GameComplete(...)` construction. |
| `scripts/GPU/alphazero/game_saver.py` | modify | Extend `save_game_replay` and `GameSaver.maybe_save_game` with eight new kwargs; write per `meta` schema in spec §4. |
| `scripts/GPU/alphazero/trainer.py` | modify | Extract `_save_game_from_ipc` + `_save_game_from_record` helpers. Replace inline blocks at `:1502-1523` and `:2491-2502`. |
| `tests/test_game_saver_per_game_fields.py` | create | All 11 tests (A + C scope). Single home for this feature's unit tests. |

---

## Task 1: MCTS final-root instrumentation

**Files:**
- Modify: `scripts/GPU/alphazero/mcts.py:200-225` (`__init__`), `:286-299` (end of `search()`), `:416-429` (end of `search_from_root()`), and add a private helper method below `_init__`.
- Test: `tests/test_game_saver_per_game_fields.py` (create)

### Step 1: Create the test file with the first MCTS instrumentation test

- [ ] Create `tests/test_game_saver_per_game_fields.py` with this content:

```python
"""Tests for per-game stats persistence (spec 2026-04-29).

Covers:
  - MCTS final-root instrumentation (final_root_value, final_top1_share)
  - JSON schema written by save_game_replay
  - Trainer routing helpers _save_game_from_ipc / _save_game_from_record
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_mcts_with_small_net(n_simulations: int = 50):
    """Construct a real MCTS with a small MLX net for instrumentation tests.

    Mirrors the pattern in tests/test_mcts.py — use the actual evaluator
    rather than a stub, so we exercise the same code paths self-play uses.
    """
    import random
    import numpy as np
    import mlx.core as mx
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator

    np.random.seed(42)
    mx.random.seed(42)
    net = create_network(hidden=64, n_blocks=2)
    evaluator = LocalGPUEvaluator(net)
    config = MCTSConfig(n_simulations=n_simulations)
    mcts = MCTS(evaluator, config, rng=random.Random(42))
    return mcts


def test_mcts_capture_final_root_stats_after_search_from_root():
    """search_from_root sets _final_root_value (finite) and _final_top1_share in (0, 1]."""
    from scripts.GPU.alphazero.mcts import MCTSNode
    from scripts.GPU.alphazero.game import TwixtState

    mcts = _make_mcts_with_small_net(n_simulations=50)
    state = TwixtState()
    root = MCTSNode(state=state)

    mcts.search_from_root(root, add_noise=False)

    assert mcts._final_root_value is not None, "final_root_value should be set after search"
    # Helper coerces to Python float — exact-type check is safe.
    assert isinstance(mcts._final_root_value, float)
    # Under MCTS numeric invariants this is finite; we don't assert range
    # tightly because spec keeps the bound informal.
    assert mcts._final_top1_share is not None, "final_top1_share should be set after search"
    assert isinstance(mcts._final_top1_share, float)
    assert 0.0 < mcts._final_top1_share <= 1.0, (
        f"final_top1_share out of range: {mcts._final_top1_share}"
    )
```

### Step 2: Run the test to verify it fails

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py::test_mcts_capture_final_root_stats_after_search_from_root -v`

Expected: **FAIL** with `AttributeError: 'MCTS' object has no attribute '_final_root_value'`.

### Step 3: Add MCTS instance attributes and the capture helper

- [ ] In `scripts/GPU/alphazero/mcts.py`, locate `MCTS.__init__` (around line 200). After the existing `self._flush_tail = 0` line (line 225), add:

```python
        # Final-root snapshot for per-game stats persistence (spec 2026-04-29).
        # Updated at the end of every successful root search; trainer reads
        # these after the game's last move.
        self._final_root_value: Optional[float] = None
        self._final_top1_share: Optional[float] = None
```

- [ ] Below `__init__`, before `def search(`, add the helper method:

```python
    def _capture_final_root_stats(self, root: MCTSNode) -> None:
        """Snapshot root.q_value and top child visit share after a search.

        Pure observation — does not mutate the tree, RNG, or counters.
        Sets self._final_root_value and self._final_top1_share for the trainer
        to read after the game's last move. Both values are coerced to Python
        float so JSON serialization downstream is straightforward.
        """
        value = getattr(root, "q_value", None)
        self._final_root_value = float(value) if value is not None else None
        children = list(getattr(root, "children", {}).values())
        if not children:
            self._final_top1_share = None
            return
        total_visits = sum(getattr(c, "visit_count", 0) for c in children)
        if total_visits <= 0:
            self._final_top1_share = None
            return
        top_visits = max(getattr(c, "visit_count", 0) for c in children)
        self._final_top1_share = float(top_visits / total_visits)
```

### Step 4: Call the helper at the end of search_from_root

- [ ] In `scripts/GPU/alphazero/mcts.py`, locate the end of `search_from_root` (around line 427). The current code ends with:

```python
        # Debug sanity check (catches encoding bugs)
        if __debug__:
            active = root.state.active_size
            for (r, c) in visit_counts.keys():
                assert 0 <= r < active and 0 <= c < active, f"Bad move {(r,c)} for active_size={active}"

        return visit_counts, root.q_value, root
```

Insert the helper call between the debug-assert block and the return:

```python
        # Debug sanity check (catches encoding bugs)
        if __debug__:
            active = root.state.active_size
            for (r, c) in visit_counts.keys():
                assert 0 <= r < active and 0 <= c < active, f"Bad move {(r,c)} for active_size={active}"

        # Snapshot final-root stats for per-game persistence (spec 2026-04-29).
        self._capture_final_root_stats(root)

        return visit_counts, root.q_value, root
```

### Step 5: Run the test to verify it passes

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py::test_mcts_capture_final_root_stats_after_search_from_root -v`

Expected: **PASS**.

### Step 6: Add the vanilla-search instrumentation test

- [ ] Append to `tests/test_game_saver_per_game_fields.py`:

```python
def test_mcts_search_vanilla_also_captures():
    """Vanilla MCTS.search() also populates the final-root snapshot."""
    from scripts.GPU.alphazero.game import TwixtState

    mcts = _make_mcts_with_small_net(n_simulations=50)
    state = TwixtState()

    mcts.search(state, add_noise=False)

    assert mcts._final_root_value is not None
    assert isinstance(mcts._final_root_value, float)
    assert mcts._final_top1_share is not None
    assert isinstance(mcts._final_top1_share, float)
    assert 0.0 < mcts._final_top1_share <= 1.0
```

### Step 7: Run the new test to verify it fails

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py::test_mcts_search_vanilla_also_captures -v`

Expected: **FAIL** — `mcts._final_root_value is None` because `search()` is not yet instrumented.

### Step 8: Call the helper at the end of vanilla search

- [ ] In `scripts/GPU/alphazero/mcts.py`, locate the end of `search` (around line 297). The current code ends with:

```python
        # Debug sanity check (catches encoding bugs)
        if __debug__:
            active = root.state.active_size
            for (r, c) in visit_counts.keys():
                assert 0 <= r < active and 0 <= c < active, f"Bad move {(r,c)} for active_size={active}"

        return visit_counts, root.q_value
```

Insert the helper call between the debug-assert block and the return:

```python
        # Debug sanity check (catches encoding bugs)
        if __debug__:
            active = root.state.active_size
            for (r, c) in visit_counts.keys():
                assert 0 <= r < active and 0 <= c < active, f"Bad move {(r,c)} for active_size={active}"

        # Snapshot final-root stats for per-game persistence (spec 2026-04-29).
        self._capture_final_root_stats(root)

        return visit_counts, root.q_value
```

### Step 9: Run the test to verify it passes

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py::test_mcts_search_vanilla_also_captures -v`

Expected: **PASS**.

### Step 10: Add the no-search and zero-visits edge tests

- [ ] Append to `tests/test_game_saver_per_game_fields.py`:

```python
def test_mcts_capture_final_root_stats_no_searches_run():
    """Fresh MCTS that never ran a search has both attributes as None."""
    mcts = _make_mcts_with_small_net(n_simulations=10)
    assert mcts._final_root_value is None
    assert mcts._final_top1_share is None


def test_mcts_capture_final_root_stats_zero_visits_returns_none_share():
    """Root with children but zero visits → top1_share is None, root_value still set."""
    from scripts.GPU.alphazero.mcts import MCTSNode
    from scripts.GPU.alphazero.game import TwixtState

    mcts = _make_mcts_with_small_net(n_simulations=10)

    state = TwixtState()
    root = MCTSNode(state=state)
    # Two children with zero visits (degenerate edge of the helper).
    child_state = state.apply_move((0, 0))
    root.children = {
        0: MCTSNode(state=child_state, parent=root, move=0),
        1: MCTSNode(state=child_state, parent=root, move=1),
    }
    # root.q_value defaults to 0.0 when visit_count == 0 (see MCTSNode.q_value)

    mcts._capture_final_root_stats(root)

    assert mcts._final_top1_share is None
    assert mcts._final_root_value == 0.0  # root.q_value with visit_count==0
```

### Step 11: Run the edge tests to verify they pass

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py -v -k "no_searches or zero_visits"`

Expected: **PASS** for both tests.

### Step 12: Run the full new test file

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py -v`

Expected: **4 tests pass** (the four MCTS-instrumentation tests).

### Step 13: Run the existing MCTS test suite for regression

Run: `.venv/bin/python -m pytest tests/test_mcts.py -v`

Expected: All existing MCTS tests pass — instrumentation does not change search behavior.

### Step 14: Commit

```bash
git add scripts/GPU/alphazero/mcts.py tests/test_game_saver_per_game_fields.py
git commit -m "feat(mcts): capture final root value and top1 share

Add MCTS._final_root_value / _final_top1_share instance attributes,
populated by a private _capture_final_root_stats(root) helper called at
the end of both search() and search_from_root(). Pure observation —
no effect on visit counts, move selection, RNG, or returned tuples.

Trainer reads these after the game's last move for per-game JSON
persistence (spec docs/superpowers/specs/2026-04-29-per-game-stats-persistence-design.md).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: GameRecord additions and wall-time measurement

**Files:**
- Modify: `scripts/GPU/alphazero/self_play.py:14-18` (imports), `:345-406` (`GameRecord` dataclass), `:430` (`play_game` function entry), `:813-866` (`return GameRecord(...)`)
- Test: `tests/test_game_saver_per_game_fields.py` (append)

### Step 1: Write the GameRecord schema test

- [ ] Append to `tests/test_game_saver_per_game_fields.py`:

```python
def test_game_record_has_new_optional_fields_with_none_defaults():
    """GameRecord schema gains wall_time_s, final_root_value, final_top1_share."""
    from scripts.GPU.alphazero.self_play import GameRecord

    record = GameRecord(positions=[], winner=None, n_moves=0)

    assert hasattr(record, "wall_time_s")
    assert hasattr(record, "final_root_value")
    assert hasattr(record, "final_top1_share")
    assert record.wall_time_s is None
    assert record.final_root_value is None
    assert record.final_top1_share is None
```

### Step 2: Run the test to verify it fails

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py::test_game_record_has_new_optional_fields_with_none_defaults -v`

Expected: **FAIL** with `AttributeError: 'GameRecord' object has no attribute 'wall_time_s'` (or similar).

### Step 3: Add the three new fields to GameRecord

- [ ] In `scripts/GPU/alphazero/self_play.py`, locate the `GameRecord` dataclass (line 345). Find the existing block near line 405:

```python
    # Phase 4: per-game replay cap diagnostics
    # n_positions_original = positions produced before cap (includes mirrors)
    # n_positions_kept     = positions retained after cap (what trainer sees)
    n_positions_original: int = 0
    n_positions_kept: int = 0
```

Add immediately after these lines (still inside the dataclass):

```python
    # Per-game stats persistence (spec 2026-04-29):
    # wall_time_s: per-game wall-clock duration; trainer/IPC paths both populate.
    # final_root_value / final_top1_share: snapshot from the last completed
    # MCTS root search before the game ended (mcts._final_root_value /
    # mcts._final_top1_share). None only in degenerate cases (no search ran,
    # or root had no children with visits).
    wall_time_s: Optional[float] = None
    final_root_value: Optional[float] = None
    final_top1_share: Optional[float] = None
```

### Step 4: Run the test to verify it passes

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py::test_game_record_has_new_optional_fields_with_none_defaults -v`

Expected: **PASS**.

### Step 5: Add `import time` to self_play.py

- [ ] In `scripts/GPU/alphazero/self_play.py`, locate the import block (lines 12-21). Find:

```python
import gc
import os
import random
```

Change to:

```python
import gc
import os
import random
import time
```

### Step 6: Add wall-time measurement at start of play_game

- [ ] In `scripts/GPU/alphazero/self_play.py`, locate the start of `play_game` body (just after the docstring closes, around line ~470 — find the first executable statement). Add at the **very top** of the function body:

```python
    game_t0 = time.perf_counter()
```

(Place it before any existing logic so the timer covers the full game.)

### Step 7: Populate the three new fields in the GameRecord(...) construction

- [ ] In `scripts/GPU/alphazero/self_play.py`, locate the `return GameRecord(...)` statement (line 813). Find the existing block:

```python
    return GameRecord(
        positions=positions,
        winner=winner,
        n_moves=ply,  # ply is authoritative (not state.ply)
        move_history=move_history,
        start_player=start_player,  # Needed for correct replay attribution
        draw_reason=draw_reason,
        resigned_by=resigned_by,  # Who resigned (or None)
        nn_calls=mcts._nn_call_count,
        ...
```

Add the three new fields anywhere inside the constructor call (place them after the closing existing block for grouping). Find the last fields:

```python
        n_positions_original=n_positions_original,
        n_positions_kept=n_positions_kept,
    )
```

Wait — those names may not match the local variables; the spec called these `n_positions_original` and `n_positions_kept` and they exist as locals. Inspect the actual return statement and append the three new kwargs immediately before the closing `)`:

```python
        n_positions_original=n_positions_original,
        n_positions_kept=n_positions_kept,
        # Per-game stats persistence (spec 2026-04-29)
        wall_time_s=time.perf_counter() - game_t0,
        final_root_value=mcts._final_root_value,
        final_top1_share=mcts._final_top1_share,
    )
```

If the actual local variable names for `n_positions_original` / `n_positions_kept` differ in the existing code, just append the three new lines at the bottom of the existing field list — order does not matter for keyword args.

### Step 8: Run existing self_play regression

Run: `.venv/bin/python -m pytest tests/test_self_play.py -v`

Expected: All existing tests pass — adding optional fields with defaults does not break callers.

### Step 9: Run the new test file in full

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py -v`

Expected: **5 tests pass** (4 from Task 1 + the new GameRecord schema test).

### Step 10: Commit

```bash
git add scripts/GPU/alphazero/self_play.py tests/test_game_saver_per_game_fields.py
git commit -m "feat(self-play): record in-process wall time and final-root diagnostics on GameRecord

Add wall_time_s, final_root_value, final_top1_share to GameRecord with
None defaults. Populate them in play_game() from a time.perf_counter()
bracket and from mcts._final_root_value / mcts._final_top1_share after
the game's last move. Per spec 2026-04-29, wall_time_s is now measured
in both IPC and in-process paths so per-game JSON is uniformly useful.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: GameComplete IPC additions and worker plumbing

**Files:**
- Modify: `scripts/GPU/alphazero/ipc_messages.py:99-103` (`GameComplete` end of fields)
- Modify: `scripts/GPU/alphazero/self_play_worker.py:219-254` (`GameComplete` construction)
- Test: `tests/test_game_saver_per_game_fields.py` (append)

### Step 1: Verify GameComplete required-fields shape, then write schema test

Before writing the test, **inspect** `scripts/GPU/alphazero/ipc_messages.py` to confirm the current set of required (no-default) fields on `GameComplete`. The plan below assumes the shape recorded in the spec at the time of writing (16 required fields ending at `flush_tail`). If new required fields have been added since, include neutral defaults for them in the test constructor too.

- [ ] Append to `tests/test_game_saver_per_game_fields.py`:

```python
def test_game_complete_has_new_optional_fields_with_none_defaults():
    """GameComplete IPC message gains final_root_value, final_top1_share."""
    from scripts.GPU.alphazero.ipc_messages import GameComplete

    msg = GameComplete(
        worker_id=0,
        winner="red",
        draw_reason=0,
        n_moves=10,
        n_positions=10,
        wall_time_s=1.5,
        nn_calls=100,
        expand_calls=100,
        nn_batches=10,
        total_backups=100,
        total_waiters=0,
        unique_leaves=100,
        max_waiters=0,
        flush_full=0,
        flush_stall=0,
        flush_tail=0,
    )

    assert hasattr(msg, "final_root_value")
    assert hasattr(msg, "final_top1_share")
    assert msg.final_root_value is None
    assert msg.final_top1_share is None
```

### Step 2: Run the test to verify it fails

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py::test_game_complete_has_new_optional_fields_with_none_defaults -v`

Expected: **FAIL** with `AttributeError`.

### Step 3: Add the two new fields to GameComplete

- [ ] In `scripts/GPU/alphazero/ipc_messages.py`, locate the end of the `GameComplete` dataclass (line 103, the last field). Current end:

```python
    # Phase 4: per-game replay cap diagnostics
    n_positions_original: int = 0
    n_positions_kept: int = 0
```

Add immediately after (still inside the dataclass body):

```python
    # Per-game stats persistence (spec 2026-04-29): final-root MCTS snapshot
    # at the last completed root search before the game ended. None when no
    # MCTS search ran or root had no children with visits.
    final_root_value: Optional[float] = None
    final_top1_share: Optional[float] = None
```

### Step 4: Run the test to verify it passes

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py::test_game_complete_has_new_optional_fields_with_none_defaults -v`

Expected: **PASS**.

### Step 5: Update the worker to forward the new fields

- [ ] In `scripts/GPU/alphazero/self_play_worker.py`, locate the `GameComplete(...)` construction (line 219). Find the existing last fields:

```python
                opening_diagnostics=tuple(game.opening_diagnostics),
                opening_diagnostics_meta=game.opening_diagnostics_meta,
                n_positions_original=game.n_positions_original,
                n_positions_kept=game.n_positions_kept,
            ))
```

Insert the two new kwargs immediately before the closing `))`:

```python
                opening_diagnostics=tuple(game.opening_diagnostics),
                opening_diagnostics_meta=game.opening_diagnostics_meta,
                n_positions_original=game.n_positions_original,
                n_positions_kept=game.n_positions_kept,
                # Per-game stats persistence (spec 2026-04-29)
                final_root_value=game.final_root_value,
                final_top1_share=game.final_top1_share,
            ))
```

### Step 6: Run the new test file in full

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py -v`

Expected: **6 tests pass**.

### Step 7: Run broader regression to confirm IPC pickling still works

Run: `.venv/bin/python -m pytest tests/ -k "self_play or mcts" -v`

Expected: All existing tests pass — adding optional fields with `None` defaults preserves pickle compatibility.

### Step 8: Commit

```bash
git add scripts/GPU/alphazero/ipc_messages.py scripts/GPU/alphazero/self_play_worker.py tests/test_game_saver_per_game_fields.py
git commit -m "feat(ipc): add final_root_value + final_top1_share to GameComplete

Two new Optional[float] fields with None defaults — pickle-safe and
backward-compatible with existing trainer consumers. Worker constructs
them from GameRecord, which itself reads from the worker's MCTS instance.

Per spec 2026-04-29.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: game_saver kwargs and JSON write

**Files:**
- Modify: `scripts/GPU/alphazero/game_saver.py:16-120` (`save_game_replay`), `:154-196` (`GameSaver.maybe_save_game`)
- Test: `tests/test_game_saver_per_game_fields.py` (append)

### Step 1: Write the "all fields populated" test

- [ ] Append to `tests/test_game_saver_per_game_fields.py`:

```python
def test_save_record_with_all_new_fields_populated(tmp_path):
    """save_game_replay writes all new fields under meta in the documented schema."""
    import json
    from scripts.GPU.alphazero.game_saver import save_game_replay

    filepath = save_game_replay(
        games_dir=tmp_path,
        iteration=12,
        game_idx=3,
        winner="red",
        move_history=((0, 0), (1, 1), (2, 2)),
        n_moves=3,
        active_size=24,
        simulations=200,
        start_player="red",
        # New per-game stats kwargs
        worker_id=2,
        wall_time_s=14.27,
        adjudication_block_reason="ply",
        final_root_value=0.83,
        final_top1_share=0.62,
        leaf_evals=17400,
        backups=17400,
        nn_batches=850,
    )

    record = json.loads(filepath.read_text())
    meta = record["meta"]

    # New flat diagnostic fields
    assert meta["worker_id"] == 2
    assert meta["wall_time_s"] == 14.27
    assert meta["adjudication_block_reason"] == "ply"
    assert meta["final_root_value"] == 0.83
    assert meta["final_top1_share"] == 0.62

    # New compute block
    assert meta["compute"] == {"leaf_evals": 17400, "backups": 17400, "nn_batches": 850}

    # Pre-existing meta keys still present and unchanged
    for key in ("board_size", "mode", "reason", "iteration", "game_idx",
               "simulations", "n_moves", "starting_player"):
        assert key in meta, f"pre-existing meta key {key!r} missing"
    assert meta["iteration"] == 12
    assert meta["game_idx"] == 3
```

### Step 2: Run the test to verify it fails

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py::test_save_record_with_all_new_fields_populated -v`

Expected: **FAIL** with `TypeError: save_game_replay() got an unexpected keyword argument 'worker_id'`.

### Step 3: Extend save_game_replay with the eight new kwargs and write them

- [ ] In `scripts/GPU/alphazero/game_saver.py`, locate the `save_game_replay` signature (line 16-30). Current signature:

```python
def save_game_replay(
    games_dir: Path,
    iteration: int,
    game_idx: int,
    winner: Optional[str],
    move_history: Tuple[Tuple[int, int], ...],
    n_moves: int,
    active_size: int = 24,
    simulations: int = 0,
    draw_reason: Optional[str] = None,
    start_player: str = "red",
    resigned_by: Optional[str] = None,
    opening_diagnostics: Optional[list] = None,
    opening_diagnostics_meta: Optional[dict] = None,
) -> Path:
```

Replace with:

```python
def save_game_replay(
    games_dir: Path,
    iteration: int,
    game_idx: int,
    winner: Optional[str],
    move_history: Tuple[Tuple[int, int], ...],
    n_moves: int,
    active_size: int = 24,
    simulations: int = 0,
    draw_reason: Optional[str] = None,
    start_player: str = "red",
    resigned_by: Optional[str] = None,
    opening_diagnostics: Optional[list] = None,
    opening_diagnostics_meta: Optional[dict] = None,
    # Per-game stats persistence (spec 2026-04-29)
    worker_id: Optional[int] = None,
    wall_time_s: Optional[float] = None,
    adjudication_block_reason: Optional[str] = None,
    final_root_value: Optional[float] = None,
    final_top1_share: Optional[float] = None,
    leaf_evals: int = 0,
    backups: int = 0,
    nn_batches: int = 0,
) -> Path:
```

- [ ] In the same function, locate the `meta = { ... }` dict construction (line 83-92). After the existing `"starting_player": start_player,` line and the optional `resigned_by` block, add the new fields. The existing block:

```python
    meta = {
        "board_size": active_size,
        "mode": "alphazero",
        "reason": reason,
        "iteration": iteration,
        "game_idx": game_idx,
        "simulations": simulations,
        "n_moves": n_moves,
        "starting_player": start_player,
    }
    # Add resigned_by only for resign games
    if reason == "resign" and resigned_by:
        meta["resigned_by"] = resigned_by
```

Insert immediately after the `if reason == "resign"` block:

```python
    # Per-game stats persistence (spec 2026-04-29).
    # Nullable flat fields use explicit None checks so 0.0 is preserved.
    # Compute counters always present; None upstream → 0 (counters are non-negative).
    meta["worker_id"] = int(worker_id) if worker_id is not None else None
    meta["wall_time_s"] = float(wall_time_s) if wall_time_s is not None else None
    meta["adjudication_block_reason"] = adjudication_block_reason
    meta["final_root_value"] = float(final_root_value) if final_root_value is not None else None
    meta["final_top1_share"] = float(final_top1_share) if final_top1_share is not None else None
    meta["compute"] = {
        "leaf_evals": int(leaf_evals or 0),
        "backups": int(backups or 0),
        "nn_batches": int(nn_batches or 0),
    }
```

### Step 4: Run the test to verify it passes

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py::test_save_record_with_all_new_fields_populated -v`

Expected: **PASS**.

### Step 5: Add the safe-defaults test

- [ ] Append to `tests/test_game_saver_per_game_fields.py`:

```python
def test_save_record_with_no_new_fields_uses_safe_defaults(tmp_path):
    """When new kwargs are unspecified, compute is zeros and flat fields are null."""
    import json
    from scripts.GPU.alphazero.game_saver import save_game_replay

    filepath = save_game_replay(
        games_dir=tmp_path,
        iteration=0,
        game_idx=0,
        winner=None,
        move_history=((0, 0),),
        n_moves=1,
    )

    meta = json.loads(filepath.read_text())["meta"]

    assert meta["compute"] == {"leaf_evals": 0, "backups": 0, "nn_batches": 0}
    assert meta["worker_id"] is None
    assert meta["wall_time_s"] is None
    assert meta["adjudication_block_reason"] is None
    assert meta["final_root_value"] is None
    assert meta["final_top1_share"] is None
```

### Step 6: Run the test to verify it passes

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py::test_save_record_with_no_new_fields_uses_safe_defaults -v`

Expected: **PASS** (defaults already wired in step 3).

### Step 7: Add the None-counter coercion test

- [ ] Append to `tests/test_game_saver_per_game_fields.py`:

```python
def test_compute_counter_none_coerces_to_zero(tmp_path):
    """leaf_evals=None / backups=None / nn_batches=None must coerce to 0, not crash."""
    import json
    from scripts.GPU.alphazero.game_saver import save_game_replay

    filepath = save_game_replay(
        games_dir=tmp_path,
        iteration=0,
        game_idx=0,
        winner=None,
        move_history=((0, 0),),
        n_moves=1,
        leaf_evals=None,
        backups=None,
        nn_batches=None,
    )

    meta = json.loads(filepath.read_text())["meta"]
    assert meta["compute"] == {"leaf_evals": 0, "backups": 0, "nn_batches": 0}
```

### Step 8: Run the test to verify it passes

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py::test_compute_counter_none_coerces_to_zero -v`

Expected: **PASS** (the `int(x or 0)` form already handles None safely).

### Step 9: Add the float-zero-preserved test

- [ ] Append to `tests/test_game_saver_per_game_fields.py`:

```python
def test_float_zero_preserved_distinct_from_null(tmp_path):
    """wall_time_s=0.0 and final_root_value=0.0 must be preserved as 0.0, not null.

    Catches `or 0.0` truthiness regressions on floats. final_top1_share=None
    is used because 0.0 is outside the documented (0, 1] range.
    """
    import json
    from scripts.GPU.alphazero.game_saver import save_game_replay

    filepath = save_game_replay(
        games_dir=tmp_path,
        iteration=0,
        game_idx=0,
        winner=None,
        move_history=((0, 0),),
        n_moves=1,
        wall_time_s=0.0,
        final_root_value=0.0,
        final_top1_share=None,
    )

    meta = json.loads(filepath.read_text())["meta"]
    assert meta["wall_time_s"] == 0.0
    assert meta["final_root_value"] == 0.0
    assert meta["final_top1_share"] is None
```

### Step 10: Run the test to verify it passes

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py::test_float_zero_preserved_distinct_from_null -v`

Expected: **PASS** (the `if x is not None` form preserves 0.0).

### Step 11: Extend GameSaver.maybe_save_game with the eight new kwargs

- [ ] In `scripts/GPU/alphazero/game_saver.py`, locate `GameSaver.maybe_save_game` (line 154). Current signature:

```python
    def maybe_save_game(
        self,
        winner: Optional[str],
        move_history: Optional[Tuple[Tuple[int, int], ...]],
        n_moves: int,
        draw_reason: Optional[str] = None,
        start_player: str = "red",
        resigned_by: Optional[str] = None,
        opening_diagnostics: Optional[list] = None,
        opening_diagnostics_meta: Optional[dict] = None,
    ) -> Optional[Path]:
```

Replace with:

```python
    def maybe_save_game(
        self,
        winner: Optional[str],
        move_history: Optional[Tuple[Tuple[int, int], ...]],
        n_moves: int,
        draw_reason: Optional[str] = None,
        start_player: str = "red",
        resigned_by: Optional[str] = None,
        opening_diagnostics: Optional[list] = None,
        opening_diagnostics_meta: Optional[dict] = None,
        # Per-game stats persistence (spec 2026-04-29)
        worker_id: Optional[int] = None,
        wall_time_s: Optional[float] = None,
        adjudication_block_reason: Optional[str] = None,
        final_root_value: Optional[float] = None,
        final_top1_share: Optional[float] = None,
        leaf_evals: int = 0,
        backups: int = 0,
        nn_batches: int = 0,
    ) -> Optional[Path]:
```

- [ ] In the same method, locate the `save_game_replay(...)` call (line 179-193). Add the eight new kwargs to the forwarded call. Current call:

```python
        filepath = save_game_replay(
            games_dir=self.games_dir,
            iteration=self._current_iter,
            game_idx=self._games_saved_this_iter,
            winner=winner,
            move_history=move_history,
            n_moves=n_moves,
            active_size=self.active_size,
            simulations=self.simulations,
            draw_reason=draw_reason,
            start_player=start_player,
            resigned_by=resigned_by,
            opening_diagnostics=opening_diagnostics,
            opening_diagnostics_meta=opening_diagnostics_meta,
        )
```

Replace with:

```python
        filepath = save_game_replay(
            games_dir=self.games_dir,
            iteration=self._current_iter,
            game_idx=self._games_saved_this_iter,
            winner=winner,
            move_history=move_history,
            n_moves=n_moves,
            active_size=self.active_size,
            simulations=self.simulations,
            draw_reason=draw_reason,
            start_player=start_player,
            resigned_by=resigned_by,
            opening_diagnostics=opening_diagnostics,
            opening_diagnostics_meta=opening_diagnostics_meta,
            # Per-game stats persistence (spec 2026-04-29)
            worker_id=worker_id,
            wall_time_s=wall_time_s,
            adjudication_block_reason=adjudication_block_reason,
            final_root_value=final_root_value,
            final_top1_share=final_top1_share,
            leaf_evals=leaf_evals,
            backups=backups,
            nn_batches=nn_batches,
        )
```

### Step 12: Run the new test file in full

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py -v`

Expected: **10 tests pass** (4 MCTS + GameRecord schema + GameComplete schema + 4 saver-JSON tests).

### Step 13: Commit

```bash
git add scripts/GPU/alphazero/game_saver.py tests/test_game_saver_per_game_fields.py
git commit -m "feat(saver): persist per-game compute, timing, and adjudication fields in JSON

Extend save_game_replay() and GameSaver.maybe_save_game() with eight new
kwargs (worker_id, wall_time_s, adjudication_block_reason,
final_root_value, final_top1_share, leaf_evals, backups, nn_batches).
Write per spec §4 schema A: nullable flat fields under meta, plus a
meta.compute block that mirrors iter_NNNN_stats.json.

compute block is always present (zeros if upstream None). Float fields
use explicit None checks so 0.0 is preserved. Existing on-disk JSON
files remain readable; consumers should use .get() with defaults.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Trainer routing helpers

**Files:**
- Modify: `scripts/GPU/alphazero/trainer.py:1502-1523` (worker-IPC save block), `:2491-2502` (in-process save block); add two private helpers.
- Test: `tests/test_game_saver_per_game_fields.py` (append)

### Step 1: Read the existing inline save blocks for accurate refactoring

- [ ] `Read scripts/GPU/alphazero/trainer.py:1490-1525` and `scripts/GPU/alphazero/trainer.py:2480-2510` to confirm the current inline structure (draw_reason translation, resigned_by derivation, opening_diagnostics handling) before extracting helpers. Also re-check `ipc_messages.GameComplete` for the current required-field shape — the routing tests in steps 5 and 9 must include neutral defaults for any required field added since this plan was written.

### Step 2: Extract `_save_game_from_ipc` helper

- [ ] In `scripts/GPU/alphazero/trainer.py`, find a sensible top-level location near other save-related helpers (or near the top of the file, after imports). Add this private helper:

```python
def _save_game_from_ipc(game_saver, msg):
    """Persist one finished game to JSON from a GameComplete IPC message.

    Internal seam for trainer.py wiring (spec 2026-04-29). Does
    draw-reason translation, resigned_by derivation, and field-name
    translation, then forwards to game_saver.maybe_save_game.

    Returns the saved Path or None if the saver skipped it.
    """
    if game_saver is None or msg.move_history is None:
        return None

    # Map draw_reason int back to string
    draw_reason_str = {
        0: None, 1: "timeout", 2: "board_full", 3: "state_cap",
        4: "unknown", 5: "resign", 6: "adjudicated",
    }.get(msg.draw_reason)

    # Derive resigned_by from msg (resign means loser resigned)
    resigned_by = None
    if draw_reason_str == "resign" and msg.winner and msg.winner != "draw":
        resigned_by = "black" if msg.winner == "red" else "red"

    return game_saver.maybe_save_game(
        winner=msg.winner if msg.winner != "draw" else None,
        move_history=msg.move_history,
        n_moves=msg.n_moves,
        draw_reason=draw_reason_str,
        start_player=msg.start_player,
        resigned_by=resigned_by,
        opening_diagnostics=list(msg.opening_diagnostics) if msg.opening_diagnostics else None,
        opening_diagnostics_meta=msg.opening_diagnostics_meta,
        # Per-game stats persistence (spec 2026-04-29)
        worker_id=msg.worker_id,
        wall_time_s=msg.wall_time_s,
        adjudication_block_reason=msg.adj_blocked_by,
        final_root_value=msg.final_root_value,
        final_top1_share=msg.final_top1_share,
        leaf_evals=msg.nn_calls,
        backups=msg.total_backups,
        nn_batches=msg.nn_batches,
    )
```

### Step 3: Replace the inline IPC save block with a helper call

- [ ] In `scripts/GPU/alphazero/trainer.py`, locate the inline block at line 1501-1523:

```python
            # Save game replay if enabled
            if game_saver is not None and msg.move_history is not None:
                # Map draw_reason int back to string (0=None, 1-4=draw reasons, 5=resign)
                # Note: resign has winner but also has draw_reason=5 for metadata
                draw_reason_str = {
                    0: None, 1: "timeout", 2: "board_full", 3: "state_cap", 4: "unknown", 5: "resign", 6: "adjudicated"
                }.get(msg.draw_reason)

                # Derive resigned_by from msg (resign means loser resigned)
                resigned_by = None
                if draw_reason_str == "resign" and msg.winner and msg.winner != "draw":
                    resigned_by = "black" if msg.winner == "red" else "red"

                game_saver.maybe_save_game(
                    winner=msg.winner if msg.winner != "draw" else None,
                    move_history=msg.move_history,
                    n_moves=msg.n_moves,
                    draw_reason=draw_reason_str,
                    start_player=msg.start_player,
                    resigned_by=resigned_by,
                    opening_diagnostics=list(msg.opening_diagnostics) if msg.opening_diagnostics else None,
                    opening_diagnostics_meta=msg.opening_diagnostics_meta,
                )
```

Replace with:

```python
            # Save game replay if enabled (spec 2026-04-29: routes per-game stats too)
            _save_game_from_ipc(game_saver, msg)
```

### Step 4: Run existing trainer/self_play regression

Run: `.venv/bin/python -m pytest tests/ -k "self_play or mcts or trainer" -v`

Expected: All existing tests pass.

### Step 5: Write the IPC-routing test

- [ ] Append to `tests/test_game_saver_per_game_fields.py`:

```python
def _make_saver(tmp_path):
    """Construct a fresh GameSaver bound to tmp_path for routing tests."""
    from scripts.GPU.alphazero.game_saver import GameSaver

    saver = GameSaver(
        games_dir=tmp_path,
        max_games_per_iter=10,
        simulations=200,
        active_size=24,
    )
    saver.set_iteration(0)
    return saver


def test_save_game_from_ipc_routes_all_new_fields(tmp_path):
    """_save_game_from_ipc translates all GameComplete fields onto save kwargs."""
    import json
    from scripts.GPU.alphazero.trainer import _save_game_from_ipc
    from scripts.GPU.alphazero.ipc_messages import GameComplete

    saver = _make_saver(tmp_path)

    msg = GameComplete(
        worker_id=2,
        winner="red",
        draw_reason=0,           # 0 → None (no draw)
        n_moves=3,
        n_positions=3,
        wall_time_s=14.27,
        nn_calls=17400,
        expand_calls=17400,
        nn_batches=850,
        total_backups=17400,
        total_waiters=0,
        unique_leaves=17400,
        max_waiters=0,
        flush_full=0,
        flush_stall=0,
        flush_tail=0,
        move_history=((0, 0), (1, 1), (2, 2)),
        start_player="red",
        adj_blocked_by="ply",
        final_root_value=0.83,
        final_top1_share=0.62,
    )

    filepath = _save_game_from_ipc(saver, msg)
    assert filepath is not None

    meta = json.loads(filepath.read_text())["meta"]
    assert meta["worker_id"] == 2
    assert meta["wall_time_s"] == 14.27
    assert meta["adjudication_block_reason"] == "ply"   # adj_blocked_by → adjudication_block_reason
    assert meta["final_root_value"] == 0.83
    assert meta["final_top1_share"] == 0.62
    assert meta["compute"] == {
        "leaf_evals": 17400,                            # nn_calls → leaf_evals
        "backups": 17400,                               # total_backups → backups
        "nn_batches": 850,
    }


def test_save_game_from_ipc_handles_optional_fields_as_null(tmp_path):
    """When GameComplete optional fields are at defaults, JSON has nulls and zeros."""
    import json
    from scripts.GPU.alphazero.trainer import _save_game_from_ipc
    from scripts.GPU.alphazero.ipc_messages import GameComplete

    saver = _make_saver(tmp_path)

    msg = GameComplete(
        worker_id=0,
        winner="draw",
        draw_reason=1,
        n_moves=2,
        n_positions=2,
        wall_time_s=0.5,
        nn_calls=0,
        expand_calls=0,
        nn_batches=0,
        total_backups=0,
        total_waiters=0,
        unique_leaves=0,
        max_waiters=0,
        flush_full=0,
        flush_stall=0,
        flush_tail=0,
        move_history=((0, 0), (1, 1)),
        start_player="red",
        # adj_blocked_by, final_root_value, final_top1_share at defaults (None)
    )

    filepath = _save_game_from_ipc(saver, msg)
    assert filepath is not None

    meta = json.loads(filepath.read_text())["meta"]
    assert meta["adjudication_block_reason"] is None
    assert meta["final_root_value"] is None
    assert meta["final_top1_share"] is None
    assert meta["compute"] == {"leaf_evals": 0, "backups": 0, "nn_batches": 0}
```

### Step 6: Run the IPC-routing tests to verify they pass

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py -v -k "save_game_from_ipc"`

Expected: **2 tests pass**.

### Step 7: Extract `_save_game_from_record` helper

- [ ] In `scripts/GPU/alphazero/trainer.py`, add (alongside `_save_game_from_ipc`):

```python
def _save_game_from_record(game_saver, game):
    """Persist one finished game to JSON from a GameRecord (in-process path).

    Internal seam for trainer.py wiring (spec 2026-04-29). Mirrors
    _save_game_from_ipc but reads from a GameRecord instead of a
    GameComplete IPC message.
    """
    if game_saver is None or not game.move_history:
        return None

    move_history_tuple = tuple(tuple(m) for m in game.move_history)

    return game_saver.maybe_save_game(
        winner=game.winner,
        move_history=move_history_tuple,
        n_moves=game.n_moves,
        draw_reason=game.draw_reason,
        start_player=game.start_player,
        resigned_by=game.resigned_by,
        opening_diagnostics=game.opening_diagnostics if game.opening_diagnostics else None,
        opening_diagnostics_meta=game.opening_diagnostics_meta,
        # Per-game stats persistence (spec 2026-04-29).
        # In-process path has no worker_id; record has wall_time_s now.
        worker_id=None,
        wall_time_s=game.wall_time_s,
        adjudication_block_reason=game.adj_blocked_by,
        final_root_value=game.final_root_value,
        final_top1_share=game.final_top1_share,
        leaf_evals=game.nn_calls,
        backups=game.total_backups,
        nn_batches=game.nn_batches,
    )
```

### Step 8: Replace the inline in-process save block with a helper call

- [ ] In `scripts/GPU/alphazero/trainer.py`, locate the inline block at line 2490-2502:

```python
                    # Save game replay if enabled
                    if game_saver is not None and game.move_history:
                        move_history_tuple = tuple(tuple(m) for m in game.move_history)
                        game_saver.maybe_save_game(
                            winner=game.winner,
                            move_history=move_history_tuple,
                            n_moves=game.n_moves,
                            draw_reason=game.draw_reason,
                            start_player=game.start_player,
                            resigned_by=game.resigned_by,
                            opening_diagnostics=game.opening_diagnostics if game.opening_diagnostics else None,
                            opening_diagnostics_meta=game.opening_diagnostics_meta,
                        )
```

Replace with:

```python
                    # Save game replay if enabled (spec 2026-04-29: routes per-game stats too)
                    _save_game_from_record(game_saver, game)
```

### Step 9: Write the record-routing test

- [ ] Append to `tests/test_game_saver_per_game_fields.py`:

```python
def test_save_game_from_record_routes_all_new_fields(tmp_path):
    """_save_game_from_record translates all GameRecord fields onto save kwargs."""
    import json
    from scripts.GPU.alphazero.trainer import _save_game_from_record
    from scripts.GPU.alphazero.self_play import GameRecord

    saver = _make_saver(tmp_path)

    game = GameRecord(
        positions=[],
        winner="black",
        n_moves=3,
        move_history=[(0, 0), (1, 1), (2, 2)],
        start_player="red",
        nn_calls=17400,
        nn_batches=850,
        total_backups=17400,
        adj_blocked_by="threshold",
        wall_time_s=12.5,
        final_root_value=-0.41,
        final_top1_share=0.55,
    )

    filepath = _save_game_from_record(saver, game)
    assert filepath is not None

    meta = json.loads(filepath.read_text())["meta"]
    assert meta["worker_id"] is None    # in-process path has no worker
    assert meta["wall_time_s"] == 12.5
    assert meta["adjudication_block_reason"] == "threshold"
    assert meta["final_root_value"] == -0.41
    assert meta["final_top1_share"] == 0.55
    assert meta["compute"] == {
        "leaf_evals": 17400,
        "backups": 17400,
        "nn_batches": 850,
    }
```

### Step 10: Run the new test to verify it passes

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py::test_save_game_from_record_routes_all_new_fields -v`

Expected: **PASS**.

### Step 11: Run the full new test file

Run: `.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py -v`

Expected: **13 tests pass** (4 MCTS + 1 GameRecord schema + 1 GameComplete schema + 4 saver-JSON + 3 routing).

### Step 12: Run the Phase 1 candidate-mining regression

Run: `.venv/bin/python -m pytest tests/test_strong_advantage_probe_suite.py -v`

Expected: All probe-suite tests pass — Phase 1 mining doesn't care about the new keys.

### Step 13: Run broader regression

Run: `.venv/bin/python -m pytest tests/ -k "mcts or self_play or trainer" -v`

Expected: All tests pass.

### Step 14: Commit

```bash
git add scripts/GPU/alphazero/trainer.py tests/test_game_saver_per_game_fields.py
git commit -m "feat(trainer): route per-game stats through new save helpers

Extract two private helpers _save_game_from_ipc and _save_game_from_record
that translate GameComplete / GameRecord fields onto game_saver kwargs:
  - msg.adj_blocked_by      → adjudication_block_reason
  - msg.nn_calls            → leaf_evals
  - msg.total_backups       → backups
  - msg.nn_batches          → nn_batches
  - msg.final_root_value    → final_root_value
  - msg.final_top1_share    → final_top1_share
  - msg.worker_id, msg.wall_time_s → unchanged

Replace inline save blocks at trainer.py:1501-1523 (worker-IPC path) and
trainer.py:2490-2502 (in-process path) with one-line calls to the
helpers. Helpers are private but tests import them directly — this is
an intentional internal seam.

Per spec 2026-04-29, both call paths now produce identical JSON shape
with the new meta fields and meta.compute block.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

After all five tasks are committed, run the full verification sequence from spec §9:

```bash
# 1. New + targeted tests
.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py -v

# 2. Phase 1 candidate mining regression
.venv/bin/python -m pytest tests/test_strong_advantage_probe_suite.py -v

# 3. Broader MCTS / self-play / trainer regression
.venv/bin/python -m pytest tests/ -k "mcts or self_play or trainer" -v

# 4. Manual inspection of one saved game JSON from a short live run.
#    Run a 1-iteration trainer config and inspect:
#    cat scripts/GPU/logs/games/iter_0000_game_000.json | python -m json.tool | head -40
#    Verify meta.worker_id, meta.wall_time_s, meta.adjudication_block_reason,
#    meta.final_root_value, meta.final_top1_share, meta.compute are all present.
```

Steps 1–3 green and step 4 showing the eight new fields/values populated → implementation complete.
