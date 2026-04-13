# Plan: Persist Training + Self-Play + MCTS Metrics Without Affecting Performance

## Goals

1. Persist the exact per-iteration metrics you already compute/print.
2. Add a small set of derived metrics useful for regression detection.
3. Support resume cleanly (append metrics; no duplication surprises).
4. Make the metrics schema forward-compatible (`schema_version`).
5. Avoid performance impact: no new work inside hot loops; write once per iteration.

---

## Phase 0 — Define Stable Schema + Run Identity

### 0.1 Add `schema_version`, `run_id`, and `timestamp`

**Where:** `trainer.py` (near top; module scope)

```python
import time
import uuid
from datetime import datetime

METRICS_SCHEMA_VERSION = 1

def generate_run_id() -> str:
    """Generate unique run ID: ISO timestamp + short random suffix."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{ts}_{suffix}"
```

**At process start (inside `train()` function, before iteration loop):**
```python
run_id = generate_run_id()
```

**Per iteration:**
```python
timestamp = datetime.now().isoformat()
```

**Why:**
- `schema_version` prevents future parsing headaches.
- `run_id` lets you append across resumes and still distinguish runs.
- `timestamp` correlates with machine events ("what happened at 3am").

---

## Phase 1 — Capture Timing (Iteration-Scoped Only)

### 1.1 Add timing around self-play, training, whole iteration

**Where:** `trainer.py` iteration loop

```python
for iteration in range(start_iteration, n_iterations):
    iter_start = time.perf_counter()

    # ... get curriculum params ...

    # 1. Self-play
    selfplay_start = time.perf_counter()
    # ... existing self-play code ...
    selfplay_end = time.perf_counter()

    # 2. Training
    train_start = time.perf_counter()
    # ... existing training code ...
    train_end = time.perf_counter()

    iter_end = time.perf_counter()

    # Compute timing metrics
    self_play_wall_s = selfplay_end - selfplay_start
    train_wall_s = train_end - train_start if train_steps_per_iteration > 0 else 0.0
    iter_wall_s = iter_end - iter_start
    positions_per_sec = positions_added / self_play_wall_s if self_play_wall_s > 0 else 0.0
```

**No perf impact:** 3–5 timer calls per iteration.

---

## Phase 2 — Persist the Exact Iteration Summary Metrics You Already Print

You already compute these at iteration scope — we persist them verbatim.

### 2.1 Build a single metrics dict at end of iteration

**Where:** `trainer.py` right before checkpoint save / right after summary print

```python
iteration_metrics = {
    # Identity
    "schema_version": METRICS_SCHEMA_VERSION,
    "run_id": run_id,
    "row_id": f"{run_id}:{iteration + 1}",  # For dedupe on crash-restart
    "timestamp": datetime.now().isoformat(),
    "iteration": iteration + 1,
    "active_size": active_size,
    "max_moves": scaled_max_moves,

    # Config snapshot (repeat each row; cheap and worth it)
    "games_per_iter": games_per_iteration,
    "simulations": mcts_simulations,
    "train_steps_per_iteration": train_steps_per_iteration,
    "batch_size": batch_size,
    "buffer_size_limit": buffer.max_size,  # CLI --buffer-size limit, not the object
    "mcts_eval_batch_size": mcts_eval_batch_size,
    "mcts_pending_virtual_visits": mcts_pending_virtual_visits,
    "mcts_stall_flush_sims": mcts_stall_flush_sims,
    "network_hidden": hidden,
    "network_blocks": n_blocks,
    # "seed": seed,  # Only include if available as variable

    # Self-play outputs
    "games_generated": games_generated,
    "positions_added": positions_added,
    "buffer_size_end": len(buffer),
    "avg_plies": avg_plies,

    # Results + draw breakdown
    "red_wins": red_wins,
    "black_wins": black_wins,
    "draws": draws,
    "timeout_draws": timeout_draws,
    "board_full_draws": board_full_draws,
    "state_cap_draws": state_cap_draws,
    "unknown_draws": unknown_draws,

    # MCTS rollup
    "total_backups": total_backups,
    "leaf_evals": total_nn_calls,
    "nn_batches": total_nn_batches,
    "avg_batch": avg_batch,
    "avg_waiters": avg_waiters,
    "max_waiters": max_waiters,
    "flush_full": total_flush_full,
    "flush_stall": total_flush_stall,
    "flush_tail": total_flush_tail,

    # Training (Phase 3 adds split)
    "avg_total_loss": avg_total_loss,
    "avg_policy_loss": avg_policy_loss,
    "avg_value_loss": avg_value_loss,
    "avg_l2_loss": avg_l2_loss,

    # Curriculum
    "draw_rate_true": curriculum_metrics["draw_rate_true"],
    "draw_rate_timeout": curriculum_metrics["draw_rate_timeout"],
    "promoted_this_iter": promoted,

    # Timing (Phase 1)
    "self_play_wall_s": self_play_wall_s,
    "train_wall_s": train_wall_s,
    "iter_wall_s": iter_wall_s,
    "positions_per_sec": positions_per_sec,

    # Derived regression detectors (Phase 4)
    "stall_flush_rate": stall_flush_rate,
    "backups_per_game": backups_per_game,
    "leaf_evals_per_game": leaf_evals_per_game,
}
```

---

## Phase 3 — Loss Split (Policy / Value / L2) with Correct MLX Grad Handling

### 3.1 Modify `alphazero_loss_batch()` to return a tuple with total first

**Where:** `trainer.py` `alphazero_loss_batch()` section (lines 149-195)

```python
def alphazero_loss_batch(
    network: AlphaZeroNetwork,
    positions: List["PositionRecord"],
    l2_weight: float = 1e-4,
    max_moves_cap: int = 512,
    active_size: int = 24,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Batched policy + value + L2 loss.

    Returns:
        Tuple of (total_loss, policy_loss, value_loss, l2_loss)

    IMPORTANT: total_loss MUST be first element because nn.value_and_grad()
    only differentiates the first returned value.
    """
    # ... existing code to compute logits, values ...

    # Policy loss: cross entropy
    log_probs = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    policy_loss = -mx.sum(target_pi * log_probs, axis=1)
    policy_loss = mx.mean(policy_loss)

    # Value loss: MSE
    value_loss = mx.mean((values - outcomes) ** 2)

    # L2 regularization
    l2_loss = mx.array(0.0)
    for _, param in flatten_params(network.parameters()):
        l2_loss = l2_loss + mx.sum(param ** 2)
    l2_loss = l2_weight * l2_loss

    total_loss = policy_loss + value_loss + l2_loss

    # CRITICAL: total_loss must be first for nn.value_and_grad()
    return total_loss, policy_loss, value_loss, l2_loss
```

### 3.2 Update `train_step()` to unpack correctly

**Where:** `trainer.py` `train_step()` function (lines 198-229)

```python
def train_step(
    network: AlphaZeroNetwork,
    optimizer: optim.Optimizer,
    batch: List["PositionRecord"],
    l2_weight: float = 1e-4,
    max_moves_cap: int = 512,
    active_size: int = 24,
) -> Tuple[float, float, float, float]:
    """Single training step with batched loss.

    Returns:
        Tuple of (total_loss, policy_loss, value_loss, l2_loss) as floats
    """
    def loss_fn(model):
        # Returns (total, policy, value, l2) - total is first for grad
        return alphazero_loss_batch(
            model, batch, l2_weight=l2_weight, max_moves_cap=max_moves_cap,
            active_size=active_size
        )

    # value_and_grad differentiates first element (total_loss)
    loss_tuple, grads = nn.value_and_grad(network, loss_fn)(network)

    # Unpack losses
    total_loss, policy_loss, value_loss, l2_loss = loss_tuple

    # Optimizer step uses total_loss (already computed via grads)
    optimizer.update(network, grads)

    # Evaluate all arrays before extracting Python floats
    # (Keep consistent with existing mx.eval pattern in codebase)
    mx.eval(network.parameters(), optimizer.state, loss_tuple)

    return (
        float(total_loss.item()),
        float(policy_loss.item()),
        float(value_loss.item()),
        float(l2_loss.item()),
    )
```

### 3.3 Trainer aggregation: maintain iteration averages for each component

**Where:** Training loop in `trainer.py:545-567`

```python
if positions_available >= batch_size and train_steps_per_iteration > 0:
    print(f"\nTraining: {train_steps_per_iteration} steps...")

    # Four accumulators
    sum_total = 0.0
    sum_policy = 0.0
    sum_value = 0.0
    sum_l2 = 0.0

    train_rng = random.Random(master_rng.randint(0, 2**31))

    for step in range(train_steps_per_iteration):
        batch = buffer.sample(batch_size, rng=train_rng, active_size=active_size)

        loss_total, loss_policy, loss_value, loss_l2 = train_step(
            network, optimizer, batch, l2_weight,
            active_size=active_size
        )

        sum_total += loss_total
        sum_policy += loss_policy
        sum_value += loss_value
        sum_l2 += loss_l2

        if (step + 1) % 20 == 0:
            avg = sum_total / (step + 1)
            print(f"  Step {step+1}/{train_steps_per_iteration}, Loss: {avg:.4f}")

    # Compute averages
    avg_total_loss = sum_total / train_steps_per_iteration
    avg_policy_loss = sum_policy / train_steps_per_iteration
    avg_value_loss = sum_value / train_steps_per_iteration
    avg_l2_loss = sum_l2 / train_steps_per_iteration

    print(f"  Average loss: {avg_total_loss:.4f} "
          f"(policy={avg_policy_loss:.4f}, value={avg_value_loss:.4f}, l2={avg_l2_loss:.4f})")
else:
    avg_total_loss = None
    avg_policy_loss = None
    avg_value_loss = None
    avg_l2_loss = None
```

**No perf impact:** adding 3 float accumulators is negligible.

---

## Phase 4 — Derived Metrics Based on Log Analysis

### 4.1 Optional: Persist self-play progress snapshots (safe)

**Where:** Same place you already print `Games: 5/25 ...`

**Guardrail:** Only collect snapshots at the same cadence as existing prints (every 5 games). This keeps both behavior and overhead aligned with existing logs.

```python
selfplay_progress = []

for g in range(games_per_iteration):
    # ... play game ...

    if (g + 1) % 5 == 0 or g == games_per_iteration - 1:
        # Existing print
        print(f"  Games: {g+1}/{games_per_iteration}, ...")

        # Capture progress snapshot (same cadence as print)
        selfplay_progress.append({
            "games_done": g + 1,
            "buffer_size": len(buffer),
            "elapsed_s": time.perf_counter() - selfplay_start,
        })
```

Store in JSON checkpoint only (not CSV) because it's nested.

### 4.2 Add derived regression detectors (cheap, high signal)

**Where:** Compute at iteration end, from existing totals

```python
# Derived metrics for regression detection
total_flushes = total_flush_full + total_flush_stall + total_flush_tail
stall_flush_rate = total_flush_stall / total_flushes if total_flushes > 0 else 0.0
backups_per_game = total_backups / games_generated if games_generated > 0 else 0.0
leaf_evals_per_game = total_nn_calls / games_generated if games_generated > 0 else 0.0
```

---

## Phase 5 — Write Metrics to Disk (Append-Only, Resume-Safe)

### 5.1 CSV: append one row per iteration

**Intent (confirmed):** append on resumed runs.

**Where:** `trainer.py` at iteration end, after metrics dict is built

```python
import csv

def append_metrics_csv(metrics_path: str, metrics: dict, fieldnames: List[str]):
    """Append one row to metrics CSV. Create with header if doesn't exist."""
    file_exists = os.path.exists(metrics_path)

    try:
        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
    except Exception as e:
        print(f"WARNING: failed to write metrics CSV: {e}")

# Define stable field order (authoritative list)
CSV_FIELDNAMES = [
    # Identity
    "schema_version", "run_id", "row_id", "timestamp", "iteration", "active_size", "max_moves",
    # Config
    "games_per_iter", "simulations", "train_steps_per_iteration", "batch_size", "buffer_size_limit",
    # Self-play
    "games_generated", "positions_added", "buffer_size_end", "avg_plies",
    # Results
    "red_wins", "black_wins", "draws",
    "timeout_draws", "board_full_draws", "state_cap_draws", "unknown_draws",
    # MCTS
    "total_backups", "leaf_evals", "nn_batches", "avg_batch",
    "avg_waiters", "max_waiters",
    "flush_full", "flush_stall", "flush_tail",
    # Training
    "avg_total_loss", "avg_policy_loss", "avg_value_loss", "avg_l2_loss",
    # Curriculum
    "draw_rate_true", "draw_rate_timeout", "promoted_this_iter",
    # Timing
    "self_play_wall_s", "train_wall_s", "iter_wall_s", "positions_per_sec",
    # Derived
    "stall_flush_rate", "backups_per_game", "leaf_evals_per_game",
]

# At iteration end:
metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
append_metrics_csv(metrics_path, iteration_metrics, CSV_FIELDNAMES)
```

### 5.2 JSON checkpoint: expand existing `model_iter_XXXX.json`

**Where:** Existing checkpoint `state = {...}` dict

```python
# Build expanded state dict
state = {
    **iteration_metrics,  # All metrics from Phase 2/3/4

    # Curriculum state for resume (existing)
    "curriculum": curriculum.to_dict(),

    # Self-play progress snapshots (nested, JSON only)
    "selfplay_progress": selfplay_progress,
}

# Write with guardrail
state_path = ckpt_path.replace(".safetensors", ".json")
try:
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
except Exception as e:
    print(f"WARNING: failed to write checkpoint JSON: {e}")
```

---

## Phase 6 — Verification (Correctness + No Performance Regression)

### 6.1 One-iteration smoke test

```bash
.venv/bin/python -u -m scripts.GPU.alphazero.train \
    --iterations 1 --games-per-iter 2 --train-steps 20 \
    --simulations 100 --curriculum-sizes 16 \
    --checkpoint-dir checkpoints/test-metrics
```

**Confirm:**
1. `model_iter_0001.json` contains:
   - `schema_version`, `run_id`, `timestamp`
   - `avg_plies`
   - Draw breakdown fields
   - `avg_total_loss`, `avg_policy_loss`, `avg_value_loss`, `avg_l2_loss`
   - `self_play_wall_s`, `train_wall_s`, `iter_wall_s`, `positions_per_sec`
   - `selfplay_progress` array

2. `metrics.csv` exists with header + one row

3. Training output shows loss split: `Average loss: X.XX (policy=X.XX, value=X.XX, l2=X.XX)`

### 6.2 Resume test

```bash
# Run 2 iterations
.venv/bin/python -u -m scripts.GPU.alphazero.train \
    --iterations 2 --games-per-iter 2 --train-steps 10 \
    --checkpoint-dir checkpoints/test-resume

# Resume (same command but higher iteration count)
.venv/bin/python -u -m scripts.GPU.alphazero.train \
    --iterations 4 --games-per-iter 2 --train-steps 10 \
    --checkpoint-dir checkpoints/test-resume \
    --resume checkpoints/test-resume/model_iter_0002.safetensors
```

**Verify:**
- CSV has 4 rows total (2 from first run + 2 from resume)
- `run_id` differs between process restarts (expected)
- No duplicate iteration rows for same `run_id`

---

## Invariants / Guardrails

1. **No new heavy ops** in MCTS or self-play inner loops
2. **All file I/O** happens once per iteration
3. **MLX grad:** total loss must be first element returned from `alphazero_loss_batch()`
4. **`schema_version`** always present
5. **CSV append** always includes `run_id`, `row_id`, `timestamp`
6. **try/except** around all file writes so metrics don't crash overnight runs
7. **curriculum_metrics timing:** capture `curriculum.get_metrics()` AFTER calling `curriculum.record_game(...)` for all games in the iteration
8. **Naming consistency:** use `avg_total_loss` everywhere (not `avg_loss`); same for policy/value/l2
9. **row_id for dedupe:** `row_id = f"{run_id}:{iteration+1}"` enables downstream dedupe on crash-restart

---

## Implementation Checklist

- [ ] Phase 0: Add `METRICS_SCHEMA_VERSION`, `generate_run_id()`, create `run_id` at train start
- [ ] Phase 0: Add `row_id = f"{run_id}:{iteration+1}"` for dedupe
- [ ] Phase 1: Add timing around self-play, training, iteration
- [ ] Phase 2: Build `iteration_metrics` dict with all fields
- [ ] Phase 2: Use `buffer.max_size` for buffer_size_limit (not buffer object)
- [ ] Phase 2: Standardize on `avg_total_loss` naming (not `avg_loss`)
- [ ] Phase 3.1: Update `alphazero_loss_batch()` to return 4-tuple (total first!)
- [ ] Phase 3.2: Update `train_step()` to unpack and return 4-tuple
- [ ] Phase 3.2: Ensure `mx.eval(loss_tuple)` before `.item()` calls
- [ ] Phase 3.3: Update training loop with 4 accumulators
- [ ] Phase 4.1: Add `selfplay_progress` capture (optional, same cadence as prints)
- [ ] Phase 4.2: Add derived metrics (stall_flush_rate, backups_per_game, leaf_evals_per_game)
- [ ] Phase 5.1: Add `append_metrics_csv()` helper + `CSV_FIELDNAMES` (include row_id)
- [ ] Phase 5.2: Expand checkpoint JSON with all metrics
- [ ] Phase 5.2: Capture `curriculum.get_metrics()` AFTER all `record_game()` calls
- [ ] Phase 6.1: Run smoke test
- [ ] Phase 6.2: Run resume test
- [ ] Cleanup: Remove test checkpoint directories
