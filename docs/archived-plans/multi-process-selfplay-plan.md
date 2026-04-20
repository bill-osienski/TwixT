# Plan: CPU Hot-Path Optimization + Multi-Core Self-Play

**Status: COMPLETE** (Implemented January 2026)

## Overview

**Problem:** Training is CPU-bound (92–93% one core). Profiling showed:
- 45% Python eval overhead
- 13% tuple comparisons (MCTS dict operations)
- 12% GPU wait (not the bottleneck)

**Solution:** Two-stage optimization (in this order):
1. **Stage 4 (do first):** Eliminate tuple overhead in MCTS (move_id:int instead of (row,col)) ✅ COMPLETE
2. **Stage 3 (do second):** Multi-core CPU workers with single GPU inference server ✅ COMPLETE

---

## Completed Stages

- **Stage 4.1** — Move encoding (move_id:int) ✅
- **Stage 3.1** — Evaluator abstraction ✅
- **Stage 3.2** — LocalGPUEvaluator ✅
- **Stage 3.3** — Multi-Process Workers ✅

---

## Stage 3.3 — Multi-Process Workers

### Configuration

```python
n_workers = max(1, os.cpu_count() - 2)  # Leave cores for main + OS
max_batch_size = 14                      # Match existing eval_batch_size
flush_ms = 2                             # Batch collection timeout (ms)
position_queue_maxsize = 128             # Backpressure on results
chunk_size = 32                          # Positions per queue put
request_q_max = 256
response_q_max = 64
```

### Architecture

```
Main Process (GPU):
├─ InferenceServer thread: pulls from request_queue, batches, GPU inference
├─ Training loop: consumes positions from position_queue
└─ Manages worker lifecycle

Worker 1..N (CPU):
├─ MCTS with RemoteEvaluator (pushes to shared request_queue)
├─ Gets responses from dedicated response_queue[worker_id]
└─ Streams positions in chunks to position_queue
```

### Implementation Files

1. `scripts/GPU/alphazero/ipc_messages.py` — IPC message types (InferenceRequest, InferenceResponse, WorkerStats, StopSignal, WorkerDone, GameComplete)
2. `scripts/GPU/alphazero/inference_server.py` — GPU inference batching server
3. `scripts/GPU/alphazero/remote_evaluator.py` — CPU-only evaluator for workers
4. `scripts/GPU/alphazero/self_play_worker.py` — Worker process entry point
5. `scripts/GPU/alphazero/trainer.py` — Added `run_parallel_selfplay()` function
6. `scripts/GPU/alphazero/train.py` — Added `--n-workers` CLI argument

### Key Design Decisions

**A) RAM from parallel position production:**
- Keep position_queue small (128 chunks max)
- Chunk size modest (32)
- Training loop must consume fast

**B) GPU becomes bottleneck (expected):**
- With N workers, GPU inference saturates — this is healthy

**C) Out-of-order responses:**
- RemoteEvaluator has `_mailbox` dict to handle responses arriving out of order

**D) Mixed active_size:**
- InferenceServer groups by active_size before batching

**E) Explicit WorkerDone signal:**
- Don't rely on `Queue.empty()` — workers send explicit `WorkerDone`

**F) Shutdown ordering:**
1. Wait for all WorkerDone signals
2. Join worker processes
3. Send StopSignal to server
4. Join server thread

**G) max_batch_rows caps TOTAL rows, not request count:**
- Prevents GPU batch exceeding known-stable size (14)

**H) Server crash safety:**
- `run_forever()` wrapped in try/except
- Workers timeout after 60s with useful error

**I) Variable M (legal moves) padding:**
- InferenceServer pads requests to common max_M before batching
- Trims priors back to original M when sending responses
- Fast path skips padding when all requests have same M

---

### CLI Usage

```bash
# Sequential (default)
python -m scripts.GPU.alphazero.train --n-workers 1

# Parallel (4 workers)
python -m scripts.GPU.alphazero.train --n-workers 4
```

---

### Verification Results

**Test 1: Sequential (n_workers=1)**
- 5 games in 16.7s = ~3.3s per game

**Test 2: Parallel (n_workers=4)**
- 8 games in 20.0s = ~2.5s per game
- **24% faster per game**

**Success criteria met:**
- ✅ Games complete without deadlock
- ✅ Results tracked correctly
- ✅ Wall time reduction with more workers

---

### Known Limitations

1. **MCTS stats not available in parallel mode** — Workers don't report backups, leaf evals, etc.
2. **MLX warning in workers** — Warning about MLX import appears but doesn't affect functionality (workers use RemoteEvaluator, not LocalGPUEvaluator)
