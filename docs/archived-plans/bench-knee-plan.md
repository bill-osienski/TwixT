# Plan: Benchmark Script for Finding Optimal Worker Count (Knee Detection)

## Goal

Create a script that automatically finds the optimal `--n-workers` value by:
1. Running training with different worker counts
2. Measuring **selfplay** positions/sec (not iter time)
3. Detecting the "knee" where diminishing returns start
4. Logging GPU pressure proxies to explain the curve

---

## Key Design Decisions

### 1. Primary Metric: Selfplay Positions/Sec

```python
throughput = positions / selfplay_seconds  # NOT iter_seconds
```

Even with `--train-steps 0`, iter time includes startup, checkpointing, shutdown overhead.

### 2. Secondary Metrics (explain the curve)

Parse from output:
- `Avg batch` - GPU batching efficiency (closer to eval_batch_size = better)
- `NN batches` - total GPU calls
- `Avg waiters`, `Max waiters` - contention/dogpile
- `Flushes: full=X, stall=Y, tail=Z` - batching behavior
- `Avg plies` - game length (affects position count)
- Draw/timeout breakdown

Compute:
- `stall_flush_ratio = stall / (full + stall + tail)` - high = batching under pressure
- `nn_batches_per_pos = nn_batches / positions` - lower = better batching efficiency
- `timeout_rate = timeout_draws / games` (not / draws)
- `draw_rate = draws / games`

### 3. Variance Control

**Warmup run:**
- Must match workload shape (same board, sims, eval_batch, resume)
- Use fewer games (2) to keep it quick
- Run at max worker count in sweep (to compile all kernels)
- Discard results
- **Parse failures are non-fatal** (warmup still compiles kernels even if output format varies)
- **Implementation:** Warmup calls exact same `build_train_cmd(...)` but with `games=2` and `workers=max(sweep)` — ensures shape-complete compilation

**Fixed seeds:** Set `PYTHONHASHSEED` env var. Pass `--seed` to training if available.

**Variance gate:** If CV (stdev/mean) > 8%, warn "results unstable; increase repeats/games".

### 4. Knee Detection Algorithm (improved)

**Definition (locked):** Knee = first index where a run of `consecutive` gains falls below `min_gain`, and we return the point immediately before that run began (`i - consecutive`).

**Edge cases:**
- If fewer than 2 valid points → return the only/best worker
- If no knee found → return best throughput worker

```python
def find_knee(worker_points, min_gain=0.07, consecutive=2, slack=0.05):
    """
    1. Apply monotonic envelope (smoothed[i] = max(raw[i], smoothed[i-1]))
    2. Find knee via consecutive below-threshold gains
    3. Require knee throughput within slack% of best
    """
    # Monotonic smoothing prevents single bad repeat from faking early knee
    # Keep BOTH raw and smoothed for visibility (smoothing can hide real regression)
    smoothed = []
    for w, t in worker_points:
        if smoothed:
            t = max(t, smoothed[-1][1])
        smoothed.append((w, t))
    # NOTE: Output table shows raw median; smoothed used only for knee detection

    # Find knee: "last good point before diminishing returns run began"
    best_w, best_t = max(smoothed, key=lambda x: x[1])
    below = 0
    knee_w = best_w  # Default to best if no knee found

    for i in range(1, len(smoothed)):
        prev_t = smoothed[i-1][1]
        curr_t = smoothed[i][1]
        gain = (curr_t - prev_t) / max(prev_t, 1e-9)

        if gain < min_gain:
            below += 1
        else:
            below = 0

        if below >= consecutive:
            # Knee is at i-consecutive (last point BEFORE the diminishing run)
            # Example: consecutive=2, gains below at i=2,3 → trigger at i=3
            # → knee is at smoothed[3-2][0] = smoothed[1][0] (the last good point)
            knee_w = smoothed[i - consecutive][0]
            break

    # Check knee is within slack% of best
    knee_t = dict(smoothed)[knee_w]
    if knee_t < (1 - slack) * best_t:
        return best_w  # Knee too far from best, use best instead

    return knee_w
```

### 5. Auto-Detect Worker Sweep

```python
def default_worker_sweep() -> list[int]:
    cpu = os.cpu_count() or 4
    candidates = [
        1, 2, 3, 4, 5,        # small integers (5 is useful for M-series P-core behavior)
        6, 8,                  # round numbers
        max(1, cpu - 4),       # near saturation
        max(1, cpu - 2),
        max(1, cpu - 1),
        cpu,
    ]
    # Add 7 for Apple Silicon (P-core sweet spot)
    if cpu >= 10:
        candidates.append(7)
    return sorted({w for w in candidates if 1 <= w <= cpu})

# 12-core M-series: [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
```

CLI: `--workers auto` (default) or `--workers 1,2,4,8` (manual).

### 6. Run Validity Checks

- Parsing fails → mark run failed
- `positions == 0` or `selfplay_seconds <= 0` → fail
- Timeout rate > `--max-timeout-rate` (default 30%) → warn "workload not representative"

**Parse failure handling:**
1. If a repeat fails parsing: retry once (same worker count)
2. If retry still fails: mark repeat failed, continue to next repeat
3. If ALL repeats for a worker fail: mark `ok=0`, exclude from knee detection
4. If timeout_rate > max_timeout_rate: also exclude from knee detection (workload shape changed)

### 7. Artifact Management

- Use temp checkpoint dir
- Delete in `try/finally` (even on errors)
- Note: checkpoint save time unavoidable without `--no-save-checkpoint` flag (future improvement)

### 8. Timeout Handling (kill process tree with SIGKILL fallback)

On timeout, must kill the entire process group (not just parent PID) to avoid leaving workers/inference server hanging.

```python
import os
import signal

# Launch in own process group
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    start_new_session=True,  # Creates new process group
)

try:
    stdout, stderr = proc.communicate(timeout=timeout_s)
except subprocess.TimeoutExpired:
    pgid = os.getpgid(proc.pid)
    # Try graceful SIGTERM first
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Force kill if SIGTERM ignored (MLX hangs, etc.)
        os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=5)
    # ... handle timeout
```

### 9. Failed Run Logging

On parse failure, save stdout+stderr to a log file for debugging:

```
Benchmarks/logs/bench_knee_{timestamp}_w{workers}_r{rep}.log
```

### 10. --max-moves Handling

**v1 approach:** Do NOT pass `--max-moves` to train (avoids "unrecognized args" risk).
- Store in CSV as "benchmark intent" for logging/comparability
- The actual max_moves used is whatever train.py defaults to (200)

### 11. Explicit build_train_cmd (benchmark-pure)

Force these every time to guarantee only `--n-workers` is swept:

```python
def build_train_cmd(board, sims, games, eval_batch, workers, tmp_dir, resume=None):
    cmd = [
        sys.executable, "-m", "scripts.GPU.alphazero.train",
        "--iterations", "1",
        "--train-steps", "0",
        "--games-per-iter", str(games),
        "--curriculum-sizes", str(board),
        "--simulations", str(sims),
        "--mcts-eval-batch-size", str(eval_batch),
        "--n-workers", str(workers),
        "--checkpoint-dir", tmp_dir,
    ]
    if resume:
        cmd.extend(["--resume", resume])
    return cmd
```

### 12. Parsing from combined stdout+stderr

Some runtimes print timing to stderr. Concatenate for parsing:
```python
combined = stdout + "\n" + stderr
# Parse from combined, but save both separately to log files
```

### 13. Shape-change exclusion from knee detection

Exclude runs where workload shape changed (prevents "knee driven by longer games"):
- Compute baseline `pos_per_game` from smallest valid worker count:
  ```python
  baseline_w = min(valid_workers)  # Not necessarily w=1 if it failed
  baseline_pos_per_game = median(pos_per_game at baseline_w)
  ```
- If later point deviates by >30% from baseline: mark "shape-changed", exclude from knee
- Still log the data (just don't use for knee detection)

### 14. Timeout cleanup delay

After SIGKILL, sleep 0.3s before starting next run:
```python
try:
    pgid = os.getpgid(proc.pid)
    os.killpg(pgid, signal.SIGKILL)
except (ProcessLookupError, OSError):
    pass  # Process already exited
proc.wait(timeout=5)
time.sleep(0.3)  # Let MLX/multiprocessing release resources
```

### 15. Pin process environment (repeatability)

Set these env vars for subprocess to prevent surprise thread storms:
```python
env = os.environ.copy()
env["PYTHONHASHSEED"] = "0"
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
# Pass env=env to subprocess
```

### 16. Validity check: assert parallel mode

For workers > 1, verify output contains `Workers: {n} (parallel)`:
```python
RE_WORKERS_MODE = re.compile(r"Workers:\s*(\d+)\s*\((parallel|sequential)\)")
# If workers > 1 but mode != "parallel", mark run invalid
```

### 17. Logging: include full command and metadata

**Log file header:**
- Full command: `" ".join(cmd)`
- Start timestamp, end timestamp
- Return code

**CSV additions:**
- `cmd_hash = hashlib.sha1(" ".join(cmd).encode()).hexdigest()[:8]` (for correlation)

### 18. Warmup specifics

- Uses same `build_train_cmd()` with `games=2`, `workers=max(sweep)`
- Uses same `--checkpoint-dir` pattern (temp dir, deleted after)
- Shorter timeout: `timeout_s // 2` or `min(timeout_s, 300)`
- Parse failures are non-fatal (warmup is "best effort")

---

## CLI Arguments

```
--board INT          Curriculum/board size (default: 16)
--sims INT           MCTS simulations (default: 200)
--games INT          Games per iteration (default: 8)
--max-moves INT      Max moves per game - logged only, NOT passed to train (default: 200)
--eval-batch INT     MCTS eval batch size (default: 14)
--workers VALUE      "auto" or comma-separated list (default: auto)
--repeats INT        Runs per worker count (default: 2)
--min-gain FLOAT     Knee threshold (default: 0.07 = 7%)
--consecutive INT    Below-threshold steps for knee (default: 2)
--timeout-s INT      Per-run timeout (default: 1800)
--max-timeout-rate   Warn if timeout rate exceeds this (default: 0.3)
--resume PATH        Resume from checkpoint (passed to train)
--keep-logs          Keep temp checkpoint dirs
--csv PATH           CSV output (default: Benchmarks/bench_knee_results.csv)
```

**Median calculation:** For each worker W, compute `pos_per_s` for each repeat, then take median (not "median positions / median seconds").

---

## Output Format

### Console Summary (per worker count)

```
[w=4] median throughput = 45.23 positions/sec (21.3 pos/game)
      avg_batch=12.1, avg_waiters=1.5, max_waiters=8
      stall_ratio=0.02, timeout_rate=0.00
```

### Summary Table (ASCII)

Shows median with spread (min..max) for variance visibility:

```
workers  pos/s [min..max]    pos/game  avg_batch  stall_ratio  timeout_rate
---------------------------------------------------------------------------
1        12.4 [11.8..13.0]   21.0      13.2       0.01         0.00
2        20.1 [19.5..20.8]   21.1      13.6       0.02         0.00
3        28.5 [27.9..29.1]   21.3      13.1       0.03         0.00
4        35.2 [34.1..36.2]   21.0      12.8       0.04         0.00
6        45.3 [44.0..46.5]   21.2      12.1       0.06         0.00   <- knee
8        48.1 [46.8..49.5]   21.1      11.5       0.08         0.00
...
```

### Final Recommendation

```
=== Knee recommendation ===
  recommended --n-workers 6
  (throughput: 52.1 pos/s, 94% of best)
  best throughput at workers=10 (55.3 pos/s)
```

### CSV Schema

```csv
timestamp,git_hash,cpu_count,board,sims,games,max_moves,eval_batch,workers,
positions,selfplay_s,pos_per_s,pos_per_game,nn_batches,nn_batches_per_pos,
avg_batch,avg_waiters,max_waiters,flush_full,flush_stall,flush_tail,stall_ratio,
red_wins,black_wins,draws,timeout_draws,timeout_rate,draw_rate,avg_plies,
resume_model,cmd_hash,ok
```

**Git hash handling:** Try `git rev-parse --short HEAD`. If git unavailable or not a repo, write `git_hash="unknown"`.

**CSV writing safety:**
- Create directories (`Benchmarks/`, `Benchmarks/logs/`) if missing
- Write header only if file doesn't exist OR is empty
- Use `newline=""` for csv module (avoids blank lines on macOS)

---

## Regex Patterns

**Important:** Use the LAST match found in stdout for each metric (in case of multiple prints).

```python
RE_POSITIONS = re.compile(r"Generated\s+(\d+)\s+games,\s+(\d+)\s+positions")
RE_SELFPLAY = re.compile(r"\bselfplay=(\d+(?:\.\d+)?)s\b")
RE_AVG_BATCH = re.compile(r"Avg batch:\s*(\d+(?:\.\d+)?)")
RE_WAITERS = re.compile(r"Avg waiters:\s*(\d+(?:\.\d+)?),\s*Max waiters:\s*(\d+)")
RE_FLUSHES = re.compile(r"Flushes:\s*full=(\d+),\s*stall=(\d+),\s*tail=(\d+)")
RE_AVG_PLIES = re.compile(r"Avg plies:\s*(\d+(?:\.\d+)?)")
RE_NN_BATCHES = re.compile(r"NN batches:\s*(\d+)")

# Anchored result/draw patterns (avoid matching other "timeout=" text)
RE_RESULTS = re.compile(r"Results:\s*Red=(\d+),\s*Black=(\d+),\s*Draws=(\d+)")
RE_DRAW_BREAKDOWN = re.compile(
    r"Draw breakdown:\s*timeout=(\d+),\s*board_full=(\d+),\s*state_cap=(\d+),\s*unknown=(\d+)"
)
```

**Computed metric:** `positions_per_game = positions / games` (detects if longer games skewed throughput)

---

## MLX Warning Fix

The warning still appears because MLX is imported before mcts.py in the main process.

**Fix:** Make warning conditional on being in a worker process, with env var override for debugging:

```python
# In mcts.py (at module level, after existing imports)
import os
import sys
import multiprocessing

if "mlx" in sys.modules:
    # Only warn in worker processes (main process imports MLX intentionally)
    # Set TWIXT_WARN_MLX_IMPORT_ORDER=1 to force-enable for debugging
    force_warn = os.getenv("TWIXT_WARN_MLX_IMPORT_ORDER", "0") == "1"
    is_worker = multiprocessing.current_process().name != "MainProcess"
    if force_warn or is_worker:
        import warnings
        warnings.warn(
            "MLX was imported before mcts.py - this may cause issues in worker processes. "
            "Ensure evaluator handles all GPU operations.",
            RuntimeWarning,
            stacklevel=2,
        )
```

This is a separate small fix to include.

---

## Files to Create/Modify

1. **`scripts/bench_knee.py`** - Main benchmark script
2. **`scripts/GPU/alphazero/mcts.py`** - Fix MLX warning (conditional check)

---

## Verification

```bash
# Quick sanity check
python scripts/bench_knee.py --board 8 --sims 100 --games 4 --repeats 1

# Real benchmark
python scripts/bench_knee.py --board 16 --sims 200 --games 8 --repeats 2

# With trained model
python scripts/bench_knee.py --board 16 --sims 200 --games 8 --repeats 2 \
  --resume checkpoints/alphazero/model_iter_0100.safetensors

# Check CSV
cat Benchmarks/bench_knee_results.csv
```

**Success criteria:**
- Warmup run completes and is discarded
- Variance gate triggers if results unstable
- Knee recommendation is reasonable (typically 4-8 for 12-core)
- CSV contains all metadata columns
- No MLX warnings from workers
