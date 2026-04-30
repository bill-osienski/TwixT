# Per-Game Stats Persistence in Saved Game JSON

**Date:** 2026-04-29
**Author:** brainstormed with Bill
**Status:** Design approved, awaiting implementation plan
**Touches:** `scripts/GPU/alphazero/mcts.py`, `scripts/GPU/alphazero/self_play.py`, `scripts/GPU/alphazero/ipc_messages.py`, `scripts/GPU/alphazero/self_play_worker.py`, `scripts/GPU/alphazero/game_saver.py`, `scripts/GPU/alphazero/trainer.py`, `tests/test_game_saver_per_game_fields.py` (new)

## 1. Problem

Per-game JSON records under `scripts/GPU/logs/games/iter_NNNN_game_NNN.json` are the primary artifact for post-hoc analysis (worker imbalance, adjudication tuning, value-calibration audits, replay debugging). Today they capture six of the ten MUST-HAVE fields and one of the four NICE-TO-HAVE fields the analysis workflow needs. The remaining fields are transmitted at game-completion time via `GameComplete` IPC messages, then aggregated into `iter_NNNN_stats.json` and discarded — the per-game granularity is lost. Two more fields (`final_root_value`, `final_top1_share`) are not captured anywhere today.

## 2. Goals

1. Persist all ten MUST-HAVE and four NICE-TO-HAVE fields per game in `iter_NNNN_game_NNN.json`, including two new MCTS end-state fields not currently captured anywhere.
2. Add the new fields under `meta` without renaming, removing, or reorganizing any existing key.
3. Keep both self-play execution paths (worker-IPC and in-process) producing identical JSON shape.
4. Preserve readability of existing on-disk game files (consumers use `.get()` with safe defaults).
5. No behavior change in MCTS, self-play, adjudication, resign gating, or curriculum.

## 3. Non-goals

- **No new aggregations** in `iter_NNNN_stats.json`. Aggregation already exists; this change is per-game persistence only.
- **No migration** of existing on-disk JSON files to the new schema. Old files stay readable as-is.
- **No UI / Replay.html changes.** New `meta.*` keys are invisible to the existing replay viewer.
- **No work on the `feature/probe-phase2-parallel-labeling` branch.** This branch bases on `main` and is independent.
- **No widening of public APIs** beyond what the new fields require.

## 4. JSON schema (final)

Existing top-level keys (`id`, `timestamp`, `config_hash`, `depth`, `seed`, `winner`, `starting_player`, `moves`, `meta`, optional `opening_diagnostics` / `opening_diagnostics_meta`) are unchanged. Only `meta` gains keys.

```jsonc
"meta": {
  // existing — unchanged
  "board_size": 24,
  "mode": "alphazero",
  "reason": "win",          // "win" | "draw" | "resign" | "adjudicated" | "timeout" | "board_full" | "state_cap" | "unknown"
  "iteration": 12,
  "game_idx": 3,
  "simulations": 200,
  "n_moves": 87,
  "starting_player": "red",
  "resigned_by": "black",   // present only when reason == "resign" (existing behavior)

  // NEW — flat diagnostic fields
  "worker_id": 2,                       // int >= 0, or null when not from a worker (in-process path)
  "wall_time_s": 14.27,                 // float >= 0, normally present for both IPC and in-process paths; null only when unknown
  "adjudication_block_reason": "ply",   // "ply" | "threshold" | "visits" | "top1" | null
  "final_root_value": 0.83,             // float, last root.q_value; under MCTS numeric invariants finite and usually in [-1, 1]
  "final_top1_share": 0.62,             // float in (0, 1], or null when total_root_visits == 0 / no children

  // NEW — compute counters (always present)
  "compute": {
    "leaf_evals": 17400,                // int >= 0  (= mcts._nn_call_count)
    "backups":    17400,                // int >= 0  (= mcts._total_backups)
    "nn_batches": 850                   // int >= 0  (= mcts._nn_batches)
  }
}
```

### 4.1 Field contracts

| Field | Type | None semantics |
|---|---|---|
| `worker_id` | int \| null | in-process self-play (no worker) |
| `wall_time_s` | float \| null | null only for legacy files, failed/aborted construction paths, or unusual callers that do not measure timing |
| `adjudication_block_reason` | "ply" \| "threshold" \| "visits" \| "top1" \| null | null means no adjudication block reason was recorded — usually adjudication did not run, was not checked, or passed its gates |
| `final_root_value` | float \| null | MCTS instance ran zero searches in the game (degenerate; normal played games have ≥1 search) |
| `final_top1_share` | float \| null | root had no children with visits; only null in degenerate edge cases |
| `compute.leaf_evals` | int | n/a — always present, zero only if zero searches ran |
| `compute.backups` | int | n/a — always present, zero only if zero searches ran |
| `compute.nn_batches` | int | n/a — always present, zero only if zero searches ran |

### 4.2 Schema rationale

The five flat diagnostic fields answer "what happened" questions (filtering, grouping, sorting per-game summaries). The nested `compute` block groups counters that belong together and mirrors the existing `iter_NNNN_stats.json` `compute` block (`trainer.py:2892`). This separation lets consumers compare per-game vs iteration vs worker-level compute totals using the same schema shape.

## 5. Architecture & data flow

The change runs through three planes; each plane is touched in exactly one well-bounded place.

### 5.1 Plane 1 — MCTS observation (`mcts.py`)

Add two `__init__` attributes on `MCTS`:

```python
self._final_root_value: Optional[float] = None
self._final_top1_share: Optional[float] = None
```

Add one private helper:

```python
def _capture_final_root_stats(self, root: MCTSNode) -> None:
    self._final_root_value = getattr(root, "q_value", None)
    children = list(getattr(root, "children", {}).values())
    if not children:
        self._final_top1_share = None
        return
    total_visits = sum(getattr(c, "visit_count", 0) for c in children)
    if total_visits <= 0:
        self._final_top1_share = None
        return
    top_visits = max(getattr(c, "visit_count", 0) for c in children)
    self._final_top1_share = top_visits / total_visits
```

Call `self._capture_final_root_stats(root)` at the end of **both** `MCTS.search()` (`mcts.py:299`) and `MCTS.search_from_root()` (`mcts.py:429`), after the `visit_counts` debug-assert block but before the return. Pure observation; reads `root.q_value` and `root.children` after the search has finished. No effect on visit counts, move selection, RNG, batching, or returned tuples.

Both entry points are instrumented for symmetry: production self-play uses `search_from_root`, but instrumenting `search` keeps the MCTS instance contract uniform for any one-off eval/probe caller.

### 5.2 Plane 2 — game-completion records

Two record types hold the new fields, one per call path:

- **`self_play.GameRecord`** (`self_play.py:345`) — used by the in-process trainer flow. Add:
  ```python
  wall_time_s: Optional[float] = None
  final_root_value: Optional[float] = None
  final_top1_share: Optional[float] = None
  ```
  In `self_play.run_game(...)` where `GameRecord` is constructed (`self_play.py:813`), populate `final_root_value` / `final_top1_share` from `mcts._final_root_value` / `mcts._final_top1_share`. Measure `wall_time_s` with a `time.perf_counter()` bracket around the game loop.

- **`ipc_messages.GameComplete`** (`ipc_messages.py:59`) — used by the worker-IPC path. Add:
  ```python
  final_root_value: Optional[float] = None
  final_top1_share: Optional[float] = None
  ```
  In `self_play_worker.py:219`, populate them from the worker's `mcts` instance the same way. `worker_id`, `wall_time_s`, `nn_calls`, `nn_batches`, `total_backups`, `adj_blocked_by` already exist on `GameComplete`.

### 5.3 Plane 3 — JSON write (`game_saver.py`)

Extend `save_game_replay(...)` (`game_saver.py:16`) and `GameSaver.maybe_save_game(...)` (`game_saver.py:154`) with eight new kwargs (defaults shown):

```python
worker_id: Optional[int] = None,
wall_time_s: Optional[float] = None,
adjudication_block_reason: Optional[str] = None,
final_root_value: Optional[float] = None,
final_top1_share: Optional[float] = None,
leaf_evals: int = 0,
backups: int = 0,
nn_batches: int = 0,
```

Write into `meta` per the schema in §4. Nullable flat fields are written as `null` when None; `json.dump` handles this automatically. The `compute` block is **always** written, even with zeros.

### 5.4 Trainer wiring (`trainer.py`) — testable seam

Extract the inline save logic at `trainer.py:1502-1523` and `trainer.py:2491-2502` into two private helpers in the same module:

```python
def _save_game_from_ipc(game_saver, msg: GameComplete) -> Optional[Path]: ...
def _save_game_from_record(game_saver, game: GameRecord) -> Optional[Path]: ...
```

Each helper does draw-reason translation, `resigned_by` derivation, and the new field-name translation, then calls `game_saver.maybe_save_game(...)`. Helpers are private (underscore-prefixed) but tests import them directly — this is an intentional internal seam. Trainer call sites become one-liners.

### 5.5 Field-name mapping (record → save kwargs)

Existing record fields map to save kwargs as:

| Source field (GameComplete / GameRecord) | Save kwarg |
|---|---|
| `worker_id` | `worker_id` |
| `wall_time_s` | `wall_time_s` |
| `adj_blocked_by` | `adjudication_block_reason` |
| `nn_calls` *(GameComplete)* / `nn_calls` *(GameRecord)* | `leaf_evals` |
| `total_backups` | `backups` |
| `nn_batches` | `nn_batches` |
| `final_root_value` | `final_root_value` |
| `final_top1_share` | `final_top1_share` |

Translation is a flat kwargs pass-through — no business logic.

## 6. Edge cases, invariants, error handling

### 6.1 MCTS instrumentation invariants

| Case | Behavior |
|---|---|
| Game ends with ≥1 MCTS search executed | `mcts._final_root_value` reflects the last `root.q_value` (under normal MCTS numeric invariants this is finite and usually in [-1, 1]); `mcts._final_top1_share` ∈ (0, 1] |
| Game ends with zero MCTS searches (pathological / unreachable in TwixT but possible in tests) | Both attributes remain `None` (init values); saved JSON has explicit `null`s |
| Root has children but none have visits (degenerate edge of `_capture_final_root_stats`) | `final_top1_share = None`; `final_root_value = root.q_value` (still meaningful) |
| `q_value` is NaN/Inf | Out of scope — pre-existing MCTS numeric invariant. We do not add `isfinite` guards in this patch. If strict JSON compliance is desired later, switch `json.dump` to `allow_nan=False` as a separate hardening change. |

### 6.2 JSON serialization safety

- Python's default `json.dump` allows NaN/Inf as non-standard JSON tokens; it does not raise. We keep current behavior — no `allow_nan=False` in this patch.
- `worker_id` is cast to `int` defensively (numpy ints don't always serialize cleanly):
  `int(worker_id) if worker_id is not None else None`.
- `wall_time_s`, `final_root_value`, `final_top1_share` use **explicit `None` checks** (not `or 0.0`) so that `0.0` is preserved as `0.0`, not `null`:
  `float(wall_time_s) if wall_time_s is not None else None`.
- `compute` counters use `int(x or 0)` — these are non-negative, so `None → 0` is safe:
  ```python
  "compute": {
      "leaf_evals": int(leaf_evals or 0),
      "backups": int(backups or 0),
      "nn_batches": int(nn_batches or 0),
  }
  ```

### 6.3 Concurrency / partial-save behavior

- A worker that crashes mid-game never sends `GameComplete`. The trainer's IPC loop simply doesn't receive a record for that game, and `maybe_save_game` is never called. **Pre-existing behavior; unchanged.**
- `GameSaver.maybe_save_game` is single-threaded inside the trainer's main event loop. No file-locking concerns.

### 6.4 Backward-compat invariants

- Existing on-disk games under `scripts/GPU/logs/games/` lack the new keys. Consumers must use `meta.get("compute", {}).get("leaf_evals", 0)` style access. We do not migrate old files — they stay readable as-is, and downstream tools degrade to "field absent" rather than crashing.
- All new save-kwargs default to `None` / `0`. A future caller that passes none of them produces a JSON record with `compute = {leaf_evals: 0, backups: 0, nn_batches: 0}` and the five flat fields all `null`. The record is still well-formed and Replay-compatible.
- No existing `meta` key is renamed, removed, or moved.
- Both new `GameComplete` fields default to `None`, so the existing IPC pickle protocol is unbroken.

## 7. Test plan (A + C)

New file: **`tests/test_game_saver_per_game_fields.py`**. Uses `tmp_path` for filesystem isolation.

### 7.1 Test A — JSON contract & MCTS instrumentation

1. **`test_save_record_with_all_new_fields_populated`** — call `save_game_replay(...)` with every new kwarg set; load JSON; assert each new field at its expected `meta` location with the correct value; assert all pre-existing meta keys still present and unchanged.
2. **`test_save_record_with_no_new_fields_uses_safe_defaults`** — call `save_game_replay(...)` with only the existing required kwargs; assert `meta.compute == {"leaf_evals": 0, "backups": 0, "nn_batches": 0}` and the five flat fields are `None` (explicit `null` in JSON).
3. **`test_compute_counter_none_coerces_to_zero`** — call with `leaf_evals=None, backups=None, nn_batches=None`; assert each becomes `0` in the compute block. Catches `int(None)` regressions.
4. **`test_float_zero_preserved_distinct_from_null`** — call with `wall_time_s=0.0, final_root_value=0.0, final_top1_share=None` (None is used here since `0.0` is outside the documented `(0, 1]` range for `final_top1_share`); assert JSON has `0.0` for the first two (not `null`) and `null` for `final_top1_share`. Catches `or 0.0` truthiness regressions.
5. **`test_mcts_capture_final_root_stats_after_search_from_root`** — construct `MCTS` with a deterministic stub evaluator (small uniform priors, value=0.0); run `search_from_root(root, ...)` once with a small `n_simulations`; assert `mcts._final_root_value` is finite and `mcts._final_top1_share` is in `(0.0, 1.0]`.
6. **`test_mcts_capture_final_root_stats_no_searches_run`** — construct `MCTS`, never call any search; assert both attributes are `None`.
7. **`test_mcts_capture_final_root_stats_zero_visits_returns_none_share`** — hand-construct an `MCTSNode` root with two children whose `visit_count == 0`; call `_capture_final_root_stats(root)` directly; assert `_final_top1_share is None` and `_final_root_value == root.q_value`.
8. **`test_mcts_search_vanilla_also_captures`** — same as #5 but using `MCTS.search()` (vanilla entry point); confirms both methods are instrumented.

### 7.2 Test C — IPC and in-process routing helpers

9. **`test_save_game_from_ipc_routes_all_new_fields`** — construct a `GameComplete` with every new field populated plus `move_history` and `start_player`; call `_save_game_from_ipc(saver, msg)`; load JSON; assert each new field arrived at its meta location with correct name translation (e.g., `msg.adj_blocked_by` → `meta.adjudication_block_reason`, `msg.nn_calls` → `meta.compute.leaf_evals`, `msg.total_backups` → `meta.compute.backups`).
10. **`test_save_game_from_record_routes_all_new_fields`** — same as #9 but constructing a `GameRecord` and calling `_save_game_from_record(saver, game)`; verifies the in-process path's translation.
11. **`test_save_game_from_ipc_handles_optional_fields_as_null`** — construct a `GameComplete` with the new fields at their defaults (None); assert JSON has `null`s for the five flat fields and zeros in `compute`.

### 7.3 Regression run

```
.venv/bin/python -m pytest tests/test_strong_advantage_probe_suite.py -v
```

Confirms Phase 1 candidate mining (`probe_eval.py:~375`) still reads existing fields correctly. The probe suite shouldn't care about the new keys, but a regression run is cheap insurance.

## 8. Implementation sequencing

Five commits, each independently runnable + tested. Branch: `feature/per-game-stats-persistence`, based on `main`.

1. **`feat(mcts): capture final root value and top1 share`**
   `mcts.py` only. Adds two `__init__` attributes, the `_capture_final_root_stats(root)` helper, and one call site at the end of each search method. Includes tests #5–#8 (MCTS-only).
   *Self-contained; rest of repo unchanged.*

2. **`feat(self-play): record in-process wall time and final-root diagnostics on GameRecord`**
   `self_play.py` only. Adds `wall_time_s`, `final_root_value`, `final_top1_share` to `GameRecord` and populates them from a `time.perf_counter()` measurement plus `mcts._final_root_value` / `mcts._final_top1_share`. No JSON persistence yet.
   *No callers break (all new fields default to None).*

3. **`feat(ipc): add final_root_value + final_top1_share to GameComplete`**
   `ipc_messages.py`: two new optional fields. `self_play_worker.py`: populate them from the worker's `mcts` instance when constructing the `GameComplete`.
   *Pickle-safe. Existing trainer code ignores the new fields gracefully.*

4. **`feat(saver): persist per-game compute, timing, and adjudication fields in JSON`**
   `game_saver.py`: extend `save_game_replay` and `GameSaver.maybe_save_game` with the eight new save kwargs; write per the schema (compute block always present; nullable flat fields). Includes tests #1–#4 (saver-only).
   *No call sites updated yet — tests verify the saver in isolation.*

5. **`feat(trainer): route per-game stats through new save helpers`**
   `trainer.py`: extract `_save_game_from_ipc` and `_save_game_from_record` private helpers. Replace the inline blocks at `trainer.py:1502-1523` and `trainer.py:2491-2502` with calls to the helpers. Includes tests #9–#11 (routing tests).
   *End-to-end working. Both call paths now produce the new JSON shape.*

## 9. Verification

```bash
# 1. New + targeted tests
.venv/bin/python -m pytest tests/test_game_saver_per_game_fields.py -v

# 2. Phase 1 candidate mining regression
.venv/bin/python -m pytest tests/test_strong_advantage_probe_suite.py -v

# 3. Broader MCTS / self-play / trainer regression
.venv/bin/python -m pytest tests/ -k "mcts or self_play or trainer" -v

# 4. Manual inspection of one saved game JSON from a short live run
#    (sanity check that the schema looks right end-to-end)
```

Steps 1–3 green and step 4 showing a JSON record with the eight new fields/values populated → done.

## 10. Out of scope

- Migration script for existing `iter_NNNN_game_NNN.json` files (consumers use `.get()`).
- New aggregations in `iter_NNNN_stats.json`.
- Behavior changes to MCTS, self-play, adjudication, resign gating, or curriculum.
- Changes to Replay.html or other UI consumers.
- Work on the `feature/probe-phase2-parallel-labeling` branch.
- `allow_nan=False` hardening of `json.dump`.
