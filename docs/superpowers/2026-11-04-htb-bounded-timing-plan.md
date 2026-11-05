# H-TB bounded timing plan (2026-11-04) — PLAN ONLY

**Status:** a paper plan for card §10 step 3 (card sha256 `96269e17…`). Nothing here is built or authorized. It
needs its own build (gate, entry, tests) and approval, and then a run approval.

**Purpose:** measure the time per unit of the label and fit runs on this machine, then set their ceilings at 1.5× the
projection. **Train-arm rows only. Time only:** no teacher output, label, prediction or loss value is written,
compared or inspected.

## 1. Preconditions for any timing run

- Environments:
  - The teacher environment from `docs/superpowers/evidence/htb-teacher-setup/`. Python 3.13.15 (standalone, archive
    sha256 `003d459a…`), the baseline freeze exactly (TensorFlow 2.21.0), and the checkout re-verified at launch by
    `twixtbot_g3_runner.verify_anchor_runtime`.
  - The project `.venv` (Python 3.14.10, MLX 0.30.3, NumPy 2.4.1) for the fit, with live versions read before launch.
- Machine state: an idle gate that the launch depends on, a BOINC check before and after, Unreal Editor closed, and a
  panic/reboot check before launch.
- Gate: a new closed gate, `HTB_TIMING_AUTHORIZED` (the 64th). It is opened for one run in a clean worktree and
  restored after. Fresh, create-only destinations: `docs/superpowers/evidence/htb-timing/{label,fit}/` and
  `~/twixt-cache/htb-timing/`. The run refuses if any exists.
- **Arm guard:** the timing entry receives only C0's train arm. It refuses any game whose `split_of` is val or test.
  It never calls `select` (which reads val keys), so no val or test position is replayed or keyed.

## 2. T-label — teacher environment, CPU TensorFlow (ceiling: TERM 1,200 s, KILL +10 s)

| Step | What is timed | Size (fixed in advance) |
|---|---|---|
| L1 | Corpus authentication, `eligible_games(arms=("train",))`. It reads, hashes and parses every file, as in the real run | the whole corpus, once |
| L2 | Replay and `state_key` throughput | every position of the first 200 train-arm games in sorted-id order (about 10.8 K positions) |
| L3 | `label_game`, both orientations, lockstep in both engines: per-call `eval_one` time and per-game replay time | plies with ply % 5 == 0 of the first 40 train-arm games in sorted-id order (about 430 rows, about 860 calls); the first 20 calls are reported but excluded from the rate |
| L4 | Writing shards | one 8,192-row shard of **zeros** (synthetic, not teacher output) to `~/twixt-cache/htb-timing/` |

- **What the record keeps:** counts, wall and CPU seconds, versions, and the idle and BOINC checks. Teacher outputs
  are discarded in memory; a test asserts the record has no target or value fields.
- **Isolation check:** the process asserts at exit that `mlx` was never imported.
- **Projection.**
  - The terms:
    - L1;
    - replay and keying of all train and val positions (C0 counts: 1,676,453 + 207,336) at L2's rate;
    - 528,000 calls at L3's steady per-call time;
    - lockstep replays for the games the selection will touch, two orientations each, at L3's per-game time (bounded
      above by C0's 34,849 train and val games);
    - 33 shards at L4's rate.
  - **Label ceiling** = 1.5 × the sum, rounded up to the next 10 minutes.

## 3. T-fit — project `.venv`, MLX on the CPU, `MLX_ENABLE_TF32=0` (ceiling: TERM 1,800 s, KILL +10 s)

| Step | What is timed | Size (fixed in advance) |
|---|---|---|
| F1 | `build_examples` throughput | 640 rows: plies with ply % 5 == 1 of train-arm games from game 41 on, in sorted-id order, until 640. The targets are **synthetic uniform** distributions over the legal moves and the values are 0, because the timing does not depend on them and no teacher output exists |
| F2 | Loading `calib020_0001` twice, after its SHA-1 check | 2 loads |
| F3 | Updates of the frozen shape: 32 rows × 2 orientations, Adam 1e-4, clip 1.0, BatchNorm in evaluation mode | 60 updates; the first 10 are excluded as warm-up. The student is discarded, not saved |
| F4 | Mirror-averaged scoring | the 640 rows, two networks |

- **What the record keeps:** counts and seconds only. No loss, KL or other metric value is recorded.
- **Projection.**
  - The terms:
    - examples for 264,000 rows at F1's rate;
    - two loads;
    - 15,000 updates at F3's steady s/update;
    - scoring of 2,048 + 2 × 24,000 rows at F4's rate;
    - one save.
  - **Fit ceiling** = 1.5 × the sum, rounded up to the next 10 minutes.

## 4. Outputs, interpretation and cost

- Two create-only records (`record.json`, each with `started.json` and a terminal status), plus a short projection
  note proposing both ceilings for review. The ceilings are not adopted until reviewed.
- The timing rows are train-arm positions and some may later be selected for training. That is harmless, because
  nothing but time is recorded. Timings depend on idleness, so an idle-gate deviation is disclosed, not discounted.
- **Cost:** at most 50 minutes of wall time including both ceilings.
- **Build needed first,** for review:
  - a `time label|time fit` entry in `htb_pilot.py` that reuses the run code paths;
  - the gate;
  - two wrappers;
  - focused fake-entry tests: arm guard, no teacher outputs in the record, no MLX in the label process, refusals
    before any load, terminal records.

## 5. Revision after the user's review (2026-11-04). This supersedes §§2–4 where they differ

**Build:** `scripts/GPU/alphazero/htb_timing.py`, gate `htb_timing_authorization.HTB_TIMING_AUTHORIZED` (the 64th,
closed), wrappers `docs/superpowers/evidence/htb-timing-{label,fit}-run/01_wrapper.sh`. The pilot's shard reload now
materializes each shard's arrays once (`load_label_shards`). Before that fix, indexing inside the row loop re-read
the whole array per row: about 4.94 TB at full size (calculated, not measured).

1. **Exact ceilings.**

   | Run | TERM | KILL | Cleanup observed through |
   |---|---|---|---|
   | T-label | 1,190 s | 1,200 s | 1,210 s |
   | T-fit | 1,790 s | 1,800 s | 1,810 s |

   - The **execution allowance** is 3,000 s (50 minutes), the sum of the two KILLs.
   - **Idle waiting is separate**: before each launch the wrapper's idle gate may wait up to `idle_gate.MAX_WAIT_S`
     (1,800 s), so up to 3,600 s of waiting in all, which is not counted in the 50 minutes. Each wrapper reports its
     wall time, and the idle samples are kept in `idle_check.txt`.
2. **The complete runs are priced.** Each term is measured in the timing, or derived from a measurement by a stated
   rule. Teacher-call time is measured per call and is **not** included in the replay overhead.

   **`run label`:**

   | Term | Basis |
   |---|---|
   | teacher runtime load | measured |
   | corpus authentication | measured |
   | selection (replay, keys, dedup, order preparation) | measured on 200 train games with `select(n_val=0)`, scaled by C0's train+val position count |
   | teacher calls | the measured warm-up, plus (528,000 − warm-up) × the steady mean |
   | lockstep replay overhead | label_game time minus teacher time, per ply replayed; an upper bound of every train+val position in both orientations |
   | 33 shard writes | one measured zero shard × 33 |
   | selection.json serialization | measured in memory on 264,000 synthetic rows |

   **`run fit`:**

   | Term | Basis |
   |---|---|
   | corpus authentication | measured again, in the fit environment |
   | label reload | two synthetic full-size shards reloaded through the fixed loader, × 33 |
   | example construction | per row × 264,000 |
   | the two model loads | measured |
   | the start check | per-row, per-network scoring × 2,048 |
   | validation scoring | per row, per network × 2 × 24,000 |
   | updates | the 10 measured warm-up updates, plus 14,990 × the steady mean of updates 11–60 |
   | the student save | measured |

   **Remaining allowances, unpriced and named in each record:** KB-scale record and manifest writes; the selection
   file's disk write beyond its serialization; the fit's label-binding read (tens of MB of JSON); the step log.
   **Proposed ceilings** are 1.5 × each total, rounded up to 10 minutes; they are adopted only after review.
3. **Memory.** T-fit records:
   - physical and available memory (`sysctl hw.memsize`; free + inactive + speculative pages from `vm_stat`);
   - exact board bytes;
   - the traced bytes per built example (`tracemalloc` over 64 examples);
   - per-row label bytes and the networks' bytes.

   From these it estimates the full fit's memory. **Arithmetic before any run:** the boards alone are
   2 × 69,120 B × 264,000 rows ≈ **36.5 GB**. This machine has **48 GiB** of physical memory (51,539,607,552 B), and
   about 19.8 GB was available when this was written. **As built, the full fit is not expected to fit.**

   T-fit also prices a **lazy-board** alternative: each batch's boards rebuilt per update, with replay, and not held.
   That would be an implementation change and needs its own approval; the recipe would be unchanged.

   `run_fit` now refuses unless `Config.memory_required_bytes` is set by a reviewed edit (it is `None`, so it refuses
   today) and the available memory at launch is at least that. No stress test.
4. Train arm only, time only, no outputs recorded, no MLX in the label process: as in §§1–4, enforced in code
   (arm guard, recursive output-field check, module check).
