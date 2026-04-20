# AlphaZero TwixT Performance & Correctness Test Log (Human + Agent Digest)

_Last updated: 2026-01-31 (America/New_York)_

This document consolidates the **human-readable** narrative and the **AI/agent-readable digest** for the AlphaZero (TwixT) training performance investigations on an **M3 Pro MacBook Pro (12 CPU / 18 GPU cores)**.

---

## Table of Contents

1. [Context](#1-context)  
2. [System & Constraints](#2-system--constraints)  
3. [Correctness / Label Integrity Investigations](#3-correctness--label-integrity-investigations)  
4. [Training Behavior Notes](#4-training-behavior-notes)  
5. [Major Performance Optimization: `stall_flush_sims`](#5-major-performance-optimization-stall_flush_sims)  
6. [Eval Batch Size Experiments](#6-eval-batch-size-experiments)  
7. [Worker Count Experiments](#7-worker-count-experiments)  
8. [Variance / Repeatability Observations](#8-variance--repeatability-observations)  
9. [Current Best Known Settings](#9-current-best-known-settings)  
10. [Next Steps](#10-next-steps)  
11. [AI / Agent Digest (Parse-Friendly)](#11-ai--agent-digest-parse-friendly)

---

## 1) Context

You are running AlphaZero-style self-play for TwixT using:
- **Parallel self-play workers** feeding an **inference server batching loop**
- **MCTS** with virtual visits + stall flushing
- **Metal/MPS** acceleration on macOS

You observed:
- earlier concerns about potential parallel label corruption (value targets / perspective mismatch)
- a CLI bug where `--train-steps 0` still performed training
- performance bottlenecks in inference batching (avg batch < eval batch max, high stall/tail flushes)
- large run-to-run **variance** in throughput due to random game lengths, system load, and scheduling effects

---

## 2) System & Constraints

**Hardware**
- MacBook Pro **M3 Pro**
- CPU: **12 cores** (6 performance + 6 efficiency)
- GPU: **18 cores**
- OS: macOS (Metal/MPS backend)

**Key Constraints / Known Stability Limits**
- `eval_batch_size=16` caused GPU/Metal instability/crashes.
- `eval_batch_size=14` is stable.
- `eval_batch_size=15` was stable but **performed poorly** in the measured run (see experiments).

**MCTS simulations**
- CLI requested `--simulations 800`, but logs show **effective=400** due to a **SIMS_TABLE cap**:
  - `Sims: cli=800, table=400, factor=1.00, effective=400 (table)`

**Recommended environment variables (to reduce oversubscription & benchmark poisoning)**
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=0
export OMP_WAIT_POLICY=PASSIVE
# Optional: only if used consistently across ALL comparisons
export PYTHONHASHSEED=0
```

---

## 3) Correctness / Label Integrity Investigations

### 3.1 Initial Concern
An early A/B comparison suggested that **12-worker self-play** produced value labels/perspectives that degraded correlation vs 1-worker, implying a possible parallel aggregation bug.

### 3.2 Key Code Facts (as provided)
**sign_agree (non-draws only):**
```python
non_draw_mask = z_np != 0
non_draw_n = int(non_draw_mask.sum())
sign_agree = float(np.mean(np.sign(v_np[non_draw_mask]) == np.sign(z_np[non_draw_mask]))) if non_draw_n > 0 else 0.0
```

**z label source + mismatch check:**
```python
z_from_recs = np.array([float(rec.outcome) for rec in sample], dtype=np.float32)
z_from_batch = np.array(outcomes.tolist(), dtype=np.float32).reshape(-1)
batch_mismatch = int(np.sum(np.abs(z_from_recs - z_from_batch) > 1e-6))

z_np = np.array(outcomes.tolist(), dtype=np.float32).reshape(-1)
```

**z-by-to_move computation:**
```python
for rec in positions:
    z = float(rec.outcome)
    zs_all.append(z)
    (zs_red if rec.to_move == "red" else zs_black).append(z)
```

### 3.3 Revised Outcome After Fix + More Diagnostics
After fixing the `--train-steps 0` bug and re-running with stronger diagnostics + larger sample, the evidence indicated:
- **No label/flip bug detected**
- Parallel mode was not corrupting `z` targets
- Earlier “parallel is worse” result was likely **sample noise/variance**

A later A/B run showed **12 workers** performing *better* than 1 worker on several value metrics (MCC/bal_acc/zv_corr), supporting label integrity.

---

## 4) Training Behavior Notes

### 4.1 Replay buffer persistence
- Replay buffer is **not persisted to disk**.
- On `--resume`:
  - weights restored from `.safetensors`
  - training state restored from `.json`
  - replay buffer starts **empty**
- Therefore every resume effectively causes a “replay reset” and refills from fresh self-play.

### 4.2 Bug: `--train-steps 0` ignored (fixed)
You observed that passing `--train-steps 0` still performed training steps (e.g., printing `Training: 160 steps...`). This was confirmed as a bug and later fixed.  
This bug primarily affected **benchmark/test runs**, where you intended to measure *pure self-play throughput* without training overhead.

---

## 5) Major Performance Optimization: `stall_flush_sims`

### 5.1 What `stall_flush_sims` does (intuitive)
During MCTS, workers accumulate pending leaf evaluations. When the search fails to discover new leaves for a while, **stall flush** forces pending requests to be sent to the inference server rather than waiting indefinitely.

- **Too low**: flushes happen too frequently → excess synchronization + premature flushing → lower throughput.
- **Too high**: MCTS waits too long before flush triggers → inference server hits `flush_ms` timeout more often → batches flush due to timeouts instead of “naturally full” → throughput can drop.
- Hence there is usually a **sweet spot** where:
  - stall flushes are reduced (less pathological sync),
  - timeouts don’t dominate,
  - and the queue stays healthy.

### 5.2 First stall_flush sweep (12 workers, eval=14)
This sweep produced a dramatic improvement, with `stall_flush=48` emerging as the best performer among tested values, and `stall_flush=64` showing the “too high” effect.

Summary (as measured):
- Baseline `stall_flush=16`: **1.13 pos/sec**
- Best `stall_flush=48`: **2.82 pos/sec** (**+150%**)

---

## 6) Eval Batch Size Experiments

### 6.1 Known stability
- `eval_batch=16`: unstable/crashes (Metal/MPS)
- `eval_batch=14`: stable

### 6.2 `eval_batch=15` test (with `stall_flush=48`)
Measured outcome (on that run):
- Throughput **dropped** heavily vs `eval_batch=14`
- Conclusion: keep `eval_batch=14` for now

Important note:
- `eval_batch` and `stall_flush` do interact because both affect *how quickly inference batches are formed and flushed*.
- However in the measured run, `eval_batch=15` regressed strongly even with a known-good `stall_flush`.
- Given stability constraints and strong regression, the current best is **eval=14** and tune other levers first.

---

## 7) Worker Count Experiments

### 7.1 Why worker count matters on a 12-core (6P+6E) CPU
Even though the CPU has 12 cores, each worker is not “1 core”:
- Each worker does Python work + MCTS bookkeeping + queue waits
- If libraries spawn extra threads (BLAS/OpenMP), you can get **thread oversubscription**
- macOS scheduling + background load can strongly affect outcomes

### 7.2 Worker sweep (stall=48, eval=14)
Initial sweep suggested:
- **10 workers** best in that run
- **12 workers** performed worst (consistent with oversubscription or contention)

But a rerun showed extremely different throughput, implying that worker “best” is sensitive to:
- system load,
- thermals,
- background activity,
- variance in game lengths.

---

## 8) Variance / Repeatability Observations

You observed large run-to-run variance even with “identical” settings:
- 3.2× throughput swing between consecutive runs
- Different random games → different average plies → different positions produced → different pos/sec
- System usage during benchmarks can easily dominate in macOS

Practical implication:
- Prefer **25-game** benchmarks (or repeat 10-game tests multiple times and use median)
- Benchmark when system is quiet and plugged in (power/thermal consistency)

---

## 9) Current Best Known Settings

### Recommended **training** settings (as of 2026-01-31)

These are the settings that have been consistently stable and performant on **M3 Pro (12 CPU / 18 GPU)**:

- **n-workers:** 10  
- **mcts-eval-batch-size:** 14 (Metal-stable; 16 previously crashed; 15 regressed badly in one probe)
- **mcts-stall-flush-sims:** 48 (confirmed sweet spot in multiple sweeps)
- **simulations:** leave your CLI at `--simulations 800` if you want, but **board size 24 is capped by the sims table** to **effective=400** unless you explicitly force it (see “400 vs 800 effective sims” below).

**Always set these env vars before long runs** (prevents thread oversubscription / “thread soup”):

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=0
export OMP_WAIT_POLICY=PASSIVE
```

### Latest benchmark snapshot (w=10, stall=48, eval=14)

**(A) Normal size-24 behavior (effective=400 via sims table)**

- Sims: cli=800, table=400, factor=1.00, **effective=400 (table)**
- Generated 25 games, **2505 positions**
- NN batches: **37,797**
- Avg batch: **11.4**, Avg waiters: **2.3**, Max waiters: **49**
- Flushes: full=**24,359**, stall=**10,997**, tail=**2,416**
- Timing: selfplay=**1468.0s**
- **Throughput:** 2505 / 1468.0 = **1.71 pos/sec**

**(B) Forced effective=800 (overrides sims table)**

- Sims: cli=800, table=400, factor=1.00, **effective=800 (cli)**
- Generated 25 games, **2450 positions**
- NN batches: **66,667**
- Avg batch: **11.4**, Avg waiters: **2.5**, Max waiters: **49**
- Flushes: full=**40,666**, stall=**23,624**, tail=**2,352**
- Timing: selfplay=**2231.8s**
- **Throughput:** 2450 / 2231.8 = **1.10 pos/sec**

**400 vs 800 effective sims (this exact A/B)**

- Time: 1468s → 2232s (**1.52× slower**)
- NN batches / Leaf evals: ~**1.76× higher** (expected, more search)
- Positions were similar (2505 vs 2450), so the extra work mostly increased compute per position rather than producing more data.

**Practical conclusion:** for board size 24, **effective=400 is the throughput sweet spot** for your current goal (fast iteration). Only force 800 if you explicitly want higher search quality and are OK paying ~1.5×+ in wall time.

### Training command to continue from iter 150

```bash
.venv/bin/python -m scripts.GPU.alphazero.train   --iterations 170   --games-per-iter 25   --train-steps 100   --checkpoint-dir checkpoints/alphazero-fresh   --resume checkpoints/alphazero-fresh/model_iter_0150.safetensors   --curriculum-sizes 24   --value-lr-scale 0.0025   --value-grad-max-norm 0.05   --n-workers 10   --mcts-eval-batch-size 14   --mcts-stall-flush-sims 48
```


## 10) Next Steps

### Step 1 — Lock in the “fast & stable” production config

Use this as your *default* training baseline going forward:

- **n-workers=10**
- **stall_flush=48**
- **eval_batch=14**
- **effective sims = 400 (table) on size 24**

This is already a large speedup vs your original stall=16 / workers=12 defaults.

### Step 2 — Reduce benchmark variance (so we can trust deltas)

Your variance swings (sometimes 3×) are plausible on a laptop when:
- other workloads steal CPU/GPU,
- thermals / power management change mid-run,
- game lengths vary a lot.

For future “is X faster than Y?” questions, use this protocol:

- Run **3× 10-game** benchmarks per config (train_steps=0)
- Take the **median** pos/sec as the score
- Only trust differences ≥ ~10–15% unless you confirm with 25 games

(25 games is great for *validation*, but 3×10 is often a better cost/variance trade.)

### Step 3 — Only if we still want more speed: expose and tune `flush_ms`

At stall_flush=48, you’re already avoiding most pathological “stall flush” behavior.
The next lever is the inference server’s **batch timeout** (`flush_ms`, currently hardcoded at 2ms).

If we expose it as a CLI flag, the most likely useful sweep is:

- `flush_ms = 2 (baseline), 3, 5`

**Success looks like:**
- fewer tail flushes,
- higher avg batch / fewer NN batches per second,
- higher pos/sec without instability.

(We should not touch this until we’re measuring with the variance protocol above.)

### Step 4 — Leave `eval_batch` alone for now

- **16** was unstable on Metal.
- **15** regressed badly in your probe (likely a stability/shape/caching interaction rather than pure “batch is better”).

So treat **eval_batch=14** as the “do not mess with it” ceiling unless you’re ready for deeper MPS debugging.


## 11) AI / Agent Digest (Parse-Friendly)

### 11.1 System + Hard Constraints

```yaml
system:
  machine: "macbook pro m3 pro"
  cpu_cores: {performance: 6, efficiency: 6, total: 12}
  gpu_cores: 18
  backend: "pytorch mps / metal"
constraints:
  eval_batch_16: "unstable/crashes on metal"
  eval_batch_14: "stable"
  buffer_persistence: "replay buffer is NOT persisted; --resume starts with empty buffer"
```

### 11.2 Key Findings (High Signal)

```yaml
key_findings:
  - "Biggest throughput win came from increasing mcts_stall_flush_sims away from 16; sweet spot observed around 48."
  - "Too-high stall_flush (64) regressed throughput; likely shifts bottleneck to inference_server flush timeout (flush_ms=2)."
  - "eval_batch=15 probe regressed badly vs eval_batch=14; treat eval_batch=14 as the safe ceiling."
  - "Worker count is sensitive and highly variable on a laptop; initial sweep suggested 10>8 and 10>>12, but reruns showed multi-x variance when the system was in use."
  - "On board size 24, sims table caps effective sims to 400 unless you override; forcing effective=800 costs ~1.5x wall time in the latest A/B."
```

### 11.3 Benchmark Ledger

```yaml
benchmarks:
  - id: "stall_sweep_w12_eval14_size24"
    config: {n_workers: 12, eval_batch: 14, games: 25, train_steps: 0, sims_cli: 800, effective_sims: 400}
    results:
      stall16: {pos: 2663, s: 2354.1, pos_s: 1.13}
      stall24: {pos: 2669, s: 1607.3, pos_s: 1.66}
      stall32: {pos: 2466, s: 1080.9, pos_s: 2.28}
      stall48: {pos: 2797, s:  992.4, pos_s: 2.82}
      stall64: {pos: 2356, s: 1144.6, pos_s: 2.06}
    conclusion: "stall48 best in this sweep"

  - id: "eval_probe_w12_stall48"
    config: {n_workers: 12, stall_flush: 48, games: 25, train_steps: 0, sims_cli: 800, effective_sims: 400}
    results:
      eval14: {pos: 2797, s:  992.4, pos_s: 2.82}
      eval15: {pos: 3498, s: 2357.9, pos_s: 1.48}
    conclusion: "eval15 regressed; keep eval14"

  - id: "worker_sweep_size24_eval14_stall48"
    config: {eval_batch: 14, stall_flush: 48, games: 25, train_steps: 0, sims_cli: 800}
    results:
      w8:  {pos: 2380, s: 1509.5, pos_s: 1.58, nn_batches: 35238}
      w10: {pos: 2910, s: 1345.3, pos_s: 2.16, nn_batches: 41081}
      w12: {pos: 2890, s: 2816.5, pos_s: 1.03, nn_batches: 43816}
      w10_rerun_bad: {pos: 2773, s: 4541.1, pos_s: 0.61, note: "system was in use; shows laptop variance"}
    conclusion: "tentative best=10 workers, but require controlled variance protocol"

  - id: "latest_w10_stall48_eval14_sims_table_vs_forced"
    config: {n_workers: 10, eval_batch: 14, stall_flush: 48, games: 25, train_steps: 0}
    results:
      effective400_table: {pos: 2505, s: 1468.0, pos_s: 1.71, nn_batches: 37797}
      effective800_forced: {pos: 2450, s: 2231.8, pos_s: 1.10, nn_batches: 66667}
    conclusion: "effective=400 is throughput sweet spot on size 24"
```

### 11.4 Current Best Recommendation

```yaml
recommended_training_config:
  env:
    OMP_NUM_THREADS: 1
    MKL_NUM_THREADS: 1
    VECLIB_MAXIMUM_THREADS: 1
    NUMEXPR_NUM_THREADS: 1
    PYTORCH_ENABLE_MPS_FALLBACK: 0
    OMP_WAIT_POLICY: "PASSIVE"
  train_args:
    n_workers: 10
    mcts_eval_batch_size: 14
    mcts_stall_flush_sims: 48
    value_lr_scale: 0.0025
    value_grad_max_norm: 0.05
    curriculum_sizes: 24
```

### 11.5 Next Tests (If We Want More Speed)

```yaml
next_tests:
  - name: "variance_protocol"
    goal: "make benchmark deltas trustworthy"
    method: ["3x 10-game runs per config", "use median pos/sec", "avoid other workloads", "run after warmup"]
  - name: "flush_ms_sweep"
    goal: "reduce tail flush / improve batching efficiency after stall=48 is locked"
    requires: "expose flush_ms as CLI flag (currently hardcoded at 2ms)"
    sweep: [2, 3, 5]
    success_metrics: ["pos/sec up", "tail flush down", "nn_batches/sec down without instability"]
```

