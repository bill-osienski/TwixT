# Plan: MCTS Stats Reporting in Parallel Self-Play

**Status: COMPLETE** (Implemented January 2026)

## Problem

Parallel self-play mode shows zeros for all MCTS stats:
```
Backups: 0, Leaf evals: 0, NN batches: 0
Avg batch: 0.0, Avg waiters: 0.0, Max waiters: 0
Flushes: full=0, stall=0, tail=0
```

Workers generate these stats via `GameRecord` but don't transmit them back.

## Solution

Extend `GameComplete` IPC message to include MCTS stats, aggregate in trainer.

---

## Implementation Steps

### Step 1: Update `ipc_messages.py` - Add MCTS stats to GameComplete

**File:** `scripts/GPU/alphazero/ipc_messages.py`

Add MCTS stats fields to `GameComplete`:

```python
@dataclass(frozen=True)
class GameComplete:
    worker_id: int
    winner: str  # "red", "black", or "draw"
    draw_reason: int  # 0=none, 1=timeout, 2=board_full, 3=state_cap, 4=unknown
    n_moves: int
    n_positions: int

    # MCTS stats (per game)
    nn_calls: int
    expand_calls: int  # Added for sanity signal alongside nn_calls
    nn_batches: int
    total_backups: int
    total_waiters: int
    unique_leaves: int
    max_waiters: int
    flush_full: int
    flush_stall: int
    flush_tail: int
```

**Note:** `winner: str` uses "draw" as a winner value (semantic shift from `winner=None` in GameRecord, but workable). Trainer interprets draws via `winner == "draw"` OR `draw_reason != 0`.

**Alternative (optional):** Use `winner: int` (0=draw, 1=red, 2=black) to avoid string mistakes across processes. Current string approach is fine if consistent.

---

### Step 2: Update `self_play_worker.py` - Send stats with consistency guards

**File:** `scripts/GPU/alphazero/self_play_worker.py`

1. Import draw reason constants
2. Add mapping from string reasons to ints
3. Add consistency guard: winner/draw_reason must be coherent
4. Send full MCTS stats in GameComplete

```python
from .self_play import play_game, DRAW_TIMEOUT, DRAW_BOARD_FULL, DRAW_STATE_CAP, DRAW_UNKNOWN

_DRAW_REASON_TO_INT = {
    None: 0,
    DRAW_TIMEOUT: 1,
    DRAW_BOARD_FULL: 2,
    DRAW_STATE_CAP: 3,
    DRAW_UNKNOWN: 4,
}

# After play_game():
winner = game.winner if game.winner is not None else "draw"

# Consistency guard: winner/draw_reason must agree
if winner != "draw":
    draw_reason_int = 0  # Wins never have draw reasons
else:
    draw_reason_int = _DRAW_REASON_TO_INT.get(game.draw_reason, 4)
    if draw_reason_int == 0:
        draw_reason_int = 4  # "draw but reason=0" -> unknown

stats_queue.put(GameComplete(
    worker_id=worker_id,
    winner=winner,
    draw_reason=draw_reason_int,
    n_moves=game.n_moves,
    n_positions=len(game.positions),
    nn_calls=game.nn_calls,
    expand_calls=game.expand_calls,
    nn_batches=game.nn_batches,
    total_backups=game.total_backups,
    total_waiters=game.total_waiters,
    unique_leaves=game.unique_leaves,
    max_waiters=game.max_waiters,
    flush_full=game.flush_full,
    flush_stall=game.flush_stall,
    flush_tail=game.flush_tail,
))
```

---

### Step 3: Update `trainer.py` - Aggregate stats with continuous draining

**File:** `scripts/GPU/alphazero/trainer.py`

1. Add MCTS stat accumulators (initialized to 0)
2. Add non-blocking drain helper function
3. Drain stats_queue FIRST in main loop (before blocking on position_queue)
4. In `process_stats_message()`, aggregate from GameComplete
5. Populate final stats dict with real values

**Critical:** Must drain stats_queue every loop iteration to avoid blocking workers (stats_queue maxsize=128).

**Non-blocking drain helper:**
```python
import queue as py_queue  # stdlib queue

def _drain_stats_queue_nonblocking(stats_queue, process_stats_message, limit: int = 256):
    """Drain up to `limit` messages per call to avoid starving position consumption."""
    drained = 0
    while drained < limit:
        try:
            msg = stats_queue.get_nowait()
        except py_queue.Empty:
            break
        process_stats_message(msg)
        drained += 1
```

**Main loop pattern:**
```python
import queue as py_queue  # stdlib queue (for Empty exception)

workers_done = 0
while workers_done < n_workers:
    # Drain stats FIRST so workers don't block if stats queue fills
    _drain_stats_queue_nonblocking(stats_queue, process_stats_message)

    try:
        item = position_queue.get(timeout=0.5)
    except py_queue.Empty:  # NOTE: use py_queue.Empty, not queue.Empty
        continue

    if isinstance(item, WorkerDone):
        workers_done += 1
        continue
    buffer.add_positions(item)

# Final drain to catch last GameComplete messages
_drain_stats_queue_nonblocking(stats_queue, process_stats_message, limit=10_000)
```

**Accumulators to add:**
```python
total_backups = 0
total_nn_calls = 0
total_expand_calls = 0
total_nn_batches = 0
total_waiters = 0
total_unique_leaves = 0
max_waiters = 0
total_flush_full = 0
total_flush_stall = 0
total_flush_tail = 0
```

**In `process_stats_message()` - handle server errors + aggregate GameComplete:**
```python
def process_stats_message(msg):
    nonlocal games_completed, total_plies, ...  # all accumulators

    # Server error check (fail fast)
    if isinstance(msg, dict) and msg.get("type") == "server_error":
        raise RuntimeError(f"InferenceServer crashed: {msg.get('error')}")

    if isinstance(msg, GameComplete):
        # Existing outcome accounting...
        games_completed += 1
        total_plies += msg.n_moves
        # winner/draw tracking...

        # MCTS stats aggregation
        total_backups += msg.total_backups
        total_nn_calls += msg.nn_calls
        total_expand_calls += msg.expand_calls
        total_nn_batches += msg.nn_batches
        total_waiters += msg.total_waiters
        total_unique_leaves += msg.unique_leaves
        max_waiters = max(max_waiters, msg.max_waiters)  # max, not sum!
        total_flush_full += msg.flush_full
        total_flush_stall += msg.flush_stall
        total_flush_tail += msg.flush_tail
```

Final stats dict uses real values instead of zeros.

---

### Step 4: Remove MLX from self_play.py entirely

**File:** `scripts/GPU/alphazero/self_play.py`

**Problem:** Module-level `import mlx.core as mx` causes warning in workers even though they don't use it.

**Best fix:** Remove MLX from self_play.py entirely. Move cache clearing to caller.

1. Remove module-level `import mlx.core as mx`
2. In `play_games()`, remove the `mx.clear_cache()` call
3. Caller (trainer) already handles cache clearing per-game

**Rationale:** Workers use `play_game()` which doesn't need MLX. Main process calls through trainer which already has `mx.eval(); gc.collect(); mx.clear_cache()` after each game in sequential mode. For `play_games()`, the caller can handle cache clearing if needed.

**Import chain separation rule:**
- **Workers import:** mcts.py, remote_evaluator.py, self_play.py, game/state code (CPU only)
- **Main imports MLX:** local_evaluator.py, model/network code, trainer.py (GPU-ish)

---

## Files Modified

1. `scripts/GPU/alphazero/ipc_messages.py` - Add 10 fields to GameComplete
2. `scripts/GPU/alphazero/self_play_worker.py` - Send MCTS stats, fix draw_reason mapping with guards
3. `scripts/GPU/alphazero/trainer.py` - Aggregate stats in run_parallel_selfplay, continuous draining
4. `scripts/GPU/alphazero/self_play.py` - Remove module-level MLX import and mx.clear_cache()

---

## Verification

```bash
# Run parallel mode and verify non-zero MCTS stats
python -m scripts.GPU.alphazero.train \
  --iterations 1 --games-per-iter 8 --simulations 100 \
  --curriculum-sizes 8 --train-steps 0 --n-workers 4
```

**Success criteria:**
- No MLX warning from workers
- MCTS stats are non-zero and reasonable:
  - `Backups` ~ games * avg_plies * simulations (same order of magnitude)
  - `Leaf evals` > 0
  - `NN batches` > 0
  - `Avg batch` ~ 10-14 (near eval_batch_size)
  - `Flushes: full > 0`

**Debug if still zeros:**
If stats remain zero after implementation:
1. Add worker-side print: `print(f"Worker {worker_id} game: nn_calls={game.nn_calls}, nn_batches={game.nn_batches}")`
2. If worker prints non-zero but trainer shows zero → aggregation plumbing issue
3. If worker prints zero → MCTS search isn't running (different bug)

---

## Notes

**Versioning/migration:** Changing GameComplete dataclass must happen everywhere at once (worker + trainer). Running with stale worker code will cause pickle/unpickle errors.

**Sanity invariant:** `expand_calls` should be close to `nn_calls` (both incremented similarly in `_expand_batch`).
