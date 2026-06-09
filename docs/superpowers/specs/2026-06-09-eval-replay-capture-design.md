# Eval Replay Capture (V2 Phase A) — Design

**Date:** 2026-06-09
**Status:** Approved, pending implementation plan
**Author:** bill + Claude
**Predecessor:** V1 game-level analyzer (`docs/superpowers/specs/2026-06-09-eval-loss-analyzer-design.md`)

## Purpose

Add **opt-in replay capture** to the checkpoint-eval match path so each eval game
can write a per-ply replay sidecar (moves + value/search stats), and each
`*_games.jsonl` row can link to it via `replay_path`. Then run **one** match —
eps035 `0399` vs staged `0379` — with capture enabled to produce a real replay
dataset for V2 Phase B.

This exists because the per-ply **value trajectory is effectively free to capture**
(the eval loop already computes it and throws it away) but **expensive to recreate
later** (a second 800-game run). Capturing it now, in the same pass as the moves,
unlocks Phase B's "why / when does 0399 collapse as black" analysis at ~zero extra
runtime cost.

### Scope

- **In:** replay capture in `eval_checkpoint_match` (+ the shared `eval_runner` it
  calls), the replay sidecar schema, and the eps035 capture run.
- **Out (deferred):** the replay-aware **analyzer** (Phase B — its own spec→plan
  cycle, designed against this captured data); **tournament** capture (Phase C).

The V1 game-level analyzer is **not** modified.

## Background: what the eval loop already computes

`eval_runner.play_eval_game` (`eval_runner.py:96`) plays a game move-by-move but
keeps only the ply count. Each loop iteration already has everything we need:

```python
counts, root_value = mcts.search(state, add_noise=False)  # root_value = root.q_value
move = mcts.select_move(counts, ply)                       # (row, col)
state = state.apply_move(move)
```

- `root_value` is `root.q_value` (`mcts.py:455` returns `visit_counts, root.q_value`).
- `counts` is a dict `{(row,col): visit_count}` built from **all** legal moves
  (`mcts.py:441`), so `len(counts)` is the legal-move count.
- `move` is the engine-native `(row, col)` (`select_move -> Tuple[int,int]`,
  `apply_move(move: Pos)`).

Recording these makes **no extra search calls and consumes no RNG**, so it cannot
change game outcomes (see the determinism acceptance test).

## Replay sidecar schema

One JSON file per game: `<replay_dir>/game_{game_idx:06d}.json`.

```json
{
  "schema_version": 1,
  "pairing_id": "0399_vs_0379",
  "game_idx": 0,
  "task_id": 0,
  "seed": 35791,
  "board_size": 24,
  "red_checkpoint": "checkpoints/.../model_iter_0399.safetensors",
  "black_checkpoint": "checkpoints/.../model_iter_0379.safetensors",
  "winner": "red",
  "winner_checkpoint": "checkpoints/.../model_iter_0399.safetensors",
  "reason": "win",
  "n_moves": 75,
  "moves": [
    {"ply": 0, "player": "red", "row": 4, "col": 19,
     "root_value": 0.12, "root_top1_share": 0.31,
     "selected_visit_rank": 1, "n_legal": 520}
  ]
}
```

### Field definitions (locked)

- **Coordinates are engine-native `row`/`col`.** No x/y conversion is performed in
  Phase A. Phase B may add presentation-layer aliases later; the captured contract
  matches the engine.
- **`root_value`** — `root.q_value` after the search for this ply, **from the
  perspective of the player about to move** (`player`), *before* the selected move
  is applied. Consumers must respect this:
  - For A's value trajectory / collapse timing, use `root_value` **only on A's
    plies** (cleanest signal).
  - A board-advantage-from-A series would sign-flip on B-to-move plies
    (`a_value = root_value if player==A else -root_value`), but **avoid leaning on
    flipped values until the value-head sign convention is confirmed** against the
    training/eval code. Phase A only records; Phase B decides.
- **`root_top1_share`** = `max(counts.values()) / sum(counts.values())`. Range
  0.0–1.0; higher means MCTS concentrated visits into one top move.
- **`selected_visit_rank`** — 1-based rank of the selected move among legal moves
  sorted by descending visit count. Ties broken deterministically by move ordering
  (stable iteration). `1` means the selected move was the most-visited.
- **`n_legal`** — the number of legal actions represented in the MCTS visit-count
  dict for this root (i.e. `len(counts)`). Documented precisely so a future masked/
  pruned-move change doesn't silently shift its meaning.
- **`player`** — `"red"` / `"black"`, the side to move at that ply.

Only `schema_version` is carried (no separate `capture_version`). `mcts_sims` is not
embedded — the match's summary sidecar already records the full config.

### Fail-loud edges (protect the contract)

In the per-ply record builder:
- if `counts` is empty → raise (should be impossible; `select_move` already guards).
- if the selected move is not a key in `counts` → raise.

## Architecture

### New file — `scripts/GPU/alphazero/eval_replay.py` (pure schema + IO)

```python
REPLAY_SCHEMA_VERSION = 1

def ply_record(ply, player, move, counts, root_value) -> dict: ...   # pure; fail-loud edges
def build_replay_dict(result, seed, board_size, records) -> dict: ...  # pure; reads result fields
def replay_filename(game_idx) -> str:        ...                     # "game_000000.json"
def write_replay(replay_dir, replay_dict) -> str: ...               # mkdir(exist_ok) + write; returns relative path
```

`ply_record` and `build_replay_dict` are pure (return dicts) and unit-tested without
IO. `write_replay` is the only IO; it returns the `replay_path` (see below).

### Modified — `scripts/GPU/alphazero/eval_runner.py`

- `EvalGameResult` gains `replay_path: Optional[str] = None` (last field; default
  keeps `asdict` serialization and all existing consumers working).
- `play_eval_game(red_eval, black_eval, config, seed, capture=False)` now returns
  `(winner, reason, ply, records)`, where `records` is `None` unless `capture` is
  True, else a list of `ply_record(...)` dicts. The per-ply append happens in the
  existing loop.
- `make_result(task, winner, reason, n_moves, replay_path=None)` sets `replay_path`.
- `replay_dir` is threaded through: `run_game_tasks(..., replay_dir=None)` →
  `_run_sequential(tasks, config, factory, replay_dir)` and
  `_run_parallel(tasks, workers, config, factory, replay_dir)` →
  `_worker_main(..., replay_dir)`. `capture = replay_dir is not None`.
- In both the sequential loop and the worker loop, build the result first, then
  build the replay dict **from the result** (it already carries `winner_checkpoint`)
  and set `replay_path` on it — no duplicated color→checkpoint mapping:
  ```python
  winner, reason, nm, records = play_eval_game(red, black, config, task.seed,
                                               capture=capture)
  result = make_result(task, winner, reason, nm)        # replay_path defaults to None
  if records is not None:
      result.replay_path = write_replay(
          replay_dir,
          build_replay_dict(result, task.seed, config.board_size, records))
  <append/put> result
  ```
  (`EvalGameResult` is a non-frozen `@dataclass`, so setting `result.replay_path`
  after construction is fine. `build_replay_dict` reads `pairing_id`, `game_idx`,
  `task_id`, both checkpoints, `winner`, `winner_checkpoint`, `reason`, `n_moves`
  from the result, plus `seed` and `board_size` passed in.)
- The one direct test caller `tests/test_eval_runner.py:45` is updated to the
  4-tuple (`winner, reason, n, _ = play_eval_game(...)`).

### Worker-safety

Move records **never cross the multiprocessing queue** — each worker process writes
its own per-game sidecar to disk (unique `game_{game_idx:06d}.json`) and only the
small `EvalGameResult` (carrying a `replay_path` string) is put on the queue.
`os.makedirs(replay_dir, exist_ok=True)` inside `write_replay` is race-safe across
processes.

### Modified — `scripts/GPU/alphazero/eval_checkpoint_match.py`

- New args: `--save-eval-replays` (store_true, default **off** → current behavior
  byte-unchanged) and `--replay-dir` (default `<output-stem>_replays/`, derived the
  same way `_write_outputs` derives `<stem>_games.jsonl`).
- `run_match(..., replay_dir=None)` passes `replay_dir` to `run_game_tasks`. When
  `--save-eval-replays` is off, `replay_dir` is `None` and nothing changes.

### `replay_path` value

Stored **relative to the process working directory** (repo root for `python -m`
runs), e.g. `logs/eval/eps035_..._seed35791_replay_replays/game_000000.json`.
Absolute paths are avoided for portability; if `--replay-dir`/`--output` is given as
an absolute path, `write_replay` normalizes via `os.path.relpath(path)` against the
CWD. Phase B will resolve `replay_path` relative to the CWD.

## Backward compatibility

- `asdict(result)` (`eval_checkpoint_match.py:42`) auto-includes the new
  `replay_path` field in every JSONL row (value `null` when capture is off).
- `eval_summary` reads only specific fields — unaffected.
- The V1 analyzer's `validate_rows` checks `REQUIRED_KEYS - r.keys()` (missing keys);
  an **extra** `replay_path` key does not break it. (V1's `REQUIRED_KEYS` does not
  include `replay_path`, so the existing non-replay files keep validating too.)

## Determinism — hard acceptance test

For the same config and seed, **capture off vs capture on must yield identical games**
on the pre-replay fields:

- same `winner`, `reason`, `n_moves`, `red_score`, `black_score`, `winner_checkpoint`.

The whole JSONL is **not** compared byte-for-byte, because capture-on adds the
`replay_path` field. Compare only the pre-existing fields. (Capture adds no RNG draws
and no extra search calls, so this must hold.)

## The capture run (Phase A deliverable)

Re-run eps035 `0399` vs staged `0379` with capture, **reusing base_seed 35791** —
the original eps035 run's seed — so the engine (unchanged since that run) reproduces
the **exact 800 games V1 already characterized**, now with replays. This gives Phase B
continuity with the trusted loss-shape sample rather than a fresh equivalent one. The
output name carries the seed to avoid confusion with the prior non-replay sidecar:

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

This run is operator-executed (real MLX checkpoints, long-running, `--workers 4` is
hardware-dependent per the MLX/Metal resource-limit gotcha documented for the
checkpoint-eval system); the plan treats producing this dataset as a verification
step, not an automated test.

## Testing

Pure (`tests/test_eval_replay.py`, no MLX):
- `ply_record` builds the documented fields; `selected_visit_rank` ranks by
  descending visits with deterministic tie-break; `root_top1_share` math; `n_legal`.
- fail-loud on empty `counts` and on a selected move absent from `counts`.
- `build_replay_dict` shape + `schema_version`; `replay_filename` zero-padding.
- `write_replay` roundtrip in `tmp_path`; returns a relative path; creates the dir.

Capture integration (`tests/test_eval_runner.py`, with the `FakeEvaluator`):
- `play_eval_game(capture=False)` returns `records is None`.
- `play_eval_game(capture=True)` returns one record per ply with the right `player`
  alternation and in-range fields.
- **determinism**: same seed, capture off vs on → identical pre-replay result fields.
- `run_game_tasks(replay_dir=...)` (workers=1 and a small workers=2 case) writes one
  sidecar per game and every result has a non-null `replay_path`; with
  `replay_dir=None`, all `replay_path` are `None` and no files are written.
- scoring (`summarize_match`) is unchanged whether or not capture is on.

## Out of scope (recorded for Phase B / C)

- **Phase B** (separate spec): replay-aware analyzer keyed on the known signal
  (A-as-black, decisive A losses, 41–80 moves) — focus-game sampler, manual-review
  queue, value-trajectory / collapse-timing, opening clusters. Built against this
  captured eps035 dataset.
- **Phase C** (separate spec): `--save-eval-replays` in
  `eval_checkpoint_tournament.py` with per-pairing replay dirs.
- **`selected_policy_rank`** (needs `root.priors` plumbing) — deliberately excluded
  from Phase A's "rich-minus-policy_rank" schema.
