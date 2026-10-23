# Search distillation D2 — historical exposure audit (read-only)

**Date:** 2026-10-23

**Status:** read-only audit. Approved alongside the synthetic row-count probe. Nothing was trained, run on a GPU, modified or loaded as a model.

**What was read:** source and git history, checkpoint JSON sidecars, run logs, run records, `.venv` metadata, self-play game logs, and D2 training games (through the gradient diagnostic's training-only reader).

**Disclosure:** for one width figure, a delegated read also read the legal-move counts of the D2 held-out games' records; see §6. That figure is not used here.

## 1. The question and the three labels

**The defect** (ledger entries 26–27):
- **Where it was seen:** on the Apple M5 Max GPU under MLX 0.30.3.
- **What fails:** the policy head's narrow float32 products give grossly wrong values in the first N − 32,736 rows.
- **Sizes measured:** N = 33,600 and 33,728 rows. Nothing was gross at 512–32,768 rows.
- **For any other machine, MLX version or row count, nothing is known.**

**The row count.** The policy head runs `fc` on a (B·M) × 2 array, where B is the batch size and M is the batch's padded move width (`network.py`, `PolicyHead.forward_padded`). So the relevant shape is **B × M**.

**The labels:**
- **Confirmed matching shapes:** records show GPU row counts above 32,768 in that run, and its MLX version is recorded. This is still not proven corruption unless the version is 0.30.3 on the M5 Max, where the defect was reproduced.
- **Possible exposure:** the structure (B, the rule for M, the GPU) makes more than 32,768 rows likely or possible, but the actual widths or the runtime version were not recorded.
- **Insufficient records:** not enough is recorded to say.

## 2. How M was formed (facts)

- **The trainer** (`scripts/GPU/alphazero/trainer.py`, `make_padded_batch`) sets `M = min(max legal-move count in the batch, max_moves_cap)` (trainer.py:1149-1150).
  - The rule is unchanged since the tree was first tracked (`1e7da926`, 2026-04-20).
  - The default `max_moves_cap` was **512** in every version until **`ffe6b342`** (2026-10-20), which raised it to 576. I verified both by `git grep` on `ffe6b342~1` and by its diff.
  - No caller in `train.py` or `train()` overrides the cap.
- **Batch size:** all 542 checkpoint sidecars under `checkpoints/` record `batch_size = 64`.
- **So every `checkpoints/` training ran at most 64 × 512 = 32,768 policy-head rows.**
- **Widths:** a 24×24 position has at most 528 legal moves. In 3,500 self-play games from the calib020 lineage (220,864 positions), 25.4% of positions have more than 512.
  - **INFERENCE:** with M capped at 512, essentially every 64-position batch ran at exactly 32,768 rows. 32,768 is the largest count that entry 27 found clean, and only on the M5 Max under 0.30.3.
  - The trainer's own batches were not recorded.
- **Value-only and partial trainings** (v8/v8b, v9–v13d, v14–v14d):
  - `train_step` always takes `value_and_grad` over the whole network, with the policy loss included (trainer.py:1371ff, 1587).
  - For v8 and v14, the policy-head and encoder gradients were computed, clipped and evaluated, then not applied.
  - For v9–v13, gradients through the policy head reach the trained final block and share its clipping norm (trainer.py:1691-1726).
  - So the policy head's forward and backward ran on the GPU in every one of these runs.
- **D2:**
  - `train_student` and `forward` use cap 576 (`search_distillation_d2.py:41, 140`), and `forward` evaluates in chunks of 64 records (:146).
  - **External learning stage 2:** B = 32, cap 576.
  - **Stage B:** B = 32, cap 576. At most 32 × 528 = 16,896 rows.
- **Device:** neither `trainer.py` nor `train.py` sets a device, so they ran on MLX's default, the GPU (INFERENCE). D2 records `Device(gpu, 0)`.

## 3. MLX versions and machines (what is recorded)

- **Requirements:** `requirements.txt` has never pinned MLX (`mlx>=0.20.0`), and there is no lock file. No sidecar, `metrics.csv` or safetensors header records the MLX version, device, host, commit, M or B×M.
- **The current `.venv`:**
  - It was created 2026-05-09 18:01:27.
  - `mlx-0.30.3.dist-info` dates from 18:01:44 the same day.
  - INFERENCE: trainer runs after that moment used 0.30.3.
  - The MLX version of the earlier `.venv`, which covers v2-staged up to about iteration 129, is **not recorded**.
- **Recorded `mlx: 0.30.3`:** every run record from 2026-10-09 onward: external learning stages 2–3, Stages A/A1/B, the search profile and D2.
- **Machine:**
  - Up to 2026-10-13: INFERENCE, the old M3 Pro. Evidence: the hardware change was 2026-10-21; one abort log names an AGX G15X command buffer; Stage B's record shows macOS 26.6.2.
  - D2 (2026-10-22) ran on the M5 Max: macOS 27.0.1 is recorded.
- **Not measured anywhere:** whether 32,768 rows, or any count, is affected on the M3 Pro GPU or under any MLX version other than 0.30.3.

## 4. Classification

| Run(s) | Dates | B | M rule | Max rows | Recorded widths | MLX / machine | Label |
|---|---|---|---|---|---|---|---|
| v2-staged iters 1–399; eps035, lr0003, lr0002 continuations | 2026-04-20 → 06-19 | 64 | batch max, cap 512 | 32,768 | none | not recorded (iters ≤ ~129 unknown; 0.30.3 after 05-09, inferred); M3 Pro, inferred | **insufficient records** |
| calib020 family, **incl. the incumbent calib020_0001**; INVALID_defaults; smoke | 06-19 → 06-21 | 64 | same | 32,768 | none | same | **insufficient records** |
| v2/v2b calibrations, v3–v7 | 06-22 → 07-03 | 64 | same | 32,768 | none | same | **insufficient records** |
| v8/v8b, v14–v14d (policy gradients computed, not applied) | 07-03 → 07-09 | 64 | same | 32,768 | none | same | **insufficient records** |
| v9–v13d (gradient through the policy head reaches the final block) | 07-04 → 07-08 | 64 | same | 32,768 | none | same | **insufficient records** |
| cont5, warm5, fp6 | 08-08 → 08-11 | 64 | same | 32,768 | none | same | **insufficient records** |
| External learning stage 2; Stage B | 10-09; 10-13 | 32 | batch max, cap 576 | 16,896 | none | 0.30.3 recorded; M3 Pro, inferred | **insufficient records** (structure ≤ 16,896) |
| **D2 Stage 3 student, updates 0–21** | 10-22 | 64 | batch max, cap 576 | 33,792 | **derived**: the recorded batch identities (gradient diagnostic `trace.jsonl`) × training-game widths give 64 × M = 33,600 (update 0), 33,792 (updates 1 and 3–20), 33,664 (update 2), 33,728 (update 21) | 0.30.3 and the M5 Max GPU, recorded | **confirmed matching shapes**, on the version and machine where the defect reproduced; the operands of updates 0 and 21 reproduce it in isolation (entry 27) |
| D2 Stage 3, updates 22–400 | 10-22 | 64 | same | 33,792 | batch identities not recorded (the diagnostic stopped at its mismatch at update 21) | 0.30.3, M5 Max, recorded | **possible exposure** (27.5% of training positions exceed 512 legal moves, 1,759 of 6,403) |
| D2 evaluation `forward()` (baseline and final policy metrics) | 10-22 | 64-record chunks | batch max, cap 576 | 33,792 | not derived (the evaluated records include held-out games, which this audit did not use) | 0.30.3, M5 Max, recorded | **possible exposure** |

**The "max rows" column for the first rows:**
- None of the three labels fits a run whose code caps rows **at or below** 32,768. Those runs are labelled **insufficient records**, because what is missing is the behaviour of that exact count on that machine and MLX version, not the shapes.
- Their exposure to the *observed* failure (N > 32,768) is excluded by the code, given the default cap and no recorded override. Uncommitted local edits cannot be ruled out, because no run records its commit or a dirty-tree flag.

**Incumbent lineage**, every ancestor **insufficient records (≤ 32,768 rows)**:
1. v2-staged iteration 0001 (2026-04-20; weight source not recorded).
2. v2-staged iteration 0379 (2026-05-30).
3. lr0003-eps035-from0379 iteration 0409 (2026-06-16).
4. calib020_0001 (2026-06-20). Its source is recorded only in planning text: `--load-weights-from …/model_iter_0409`.

**Inference paths (note only):**
- **Self-play:** rows are capped at eval batch × padded width, and every sidecar records eval batch 14, so at most 7,392 rows.
- **Play and matches:** use the batch-1 path, so at most 528 rows.
- **Unaffected:** none of these exceeds 32,768 rows.
- **Exception:** D2's evaluation `forward()`, listed above.

## 5. What is not recorded (needed to upgrade any label)

1. The MLX version per training run, and the version in the pre-2026-05-09 `.venv`.
2. The machine or GPU per run.
3. The code commit and dirty-tree state per trainer run.
4. Per-batch M or B×M for any trainer run, and for D2 Stage 3 beyond update 21.
5. The device for trainer, external-learning and Stage B runs.
6. Whether exactly 32,768 rows, or any count, is affected on the M3 Pro or under any other MLX version.
7. The exact command or weight source for calib020_0001 and staged iteration 0001.

## 6. Disclosure: one read beyond the training-only boundary

The width count was delegated. The delegated read computed "27.85% of 7,020 D2 records above 512" from **every** D2 corpus game file, including the 617 held-out records.
- **What it read:** only `len(legal_moves)` per record, plus one structural look at `phase_00/games/game_0.json`. That look printed truncated fields, including a cut-off `outcome` string.
- **What was printed:** aggregate counts only. No visit counts, values or `z` were read in the counting pass.
- **What this audit uses instead:** the training-only figure (1,759 of 6,403, 27.5%), computed through the gradient diagnostic's reader, which opens no held-out file.
- **What follows:** no held-out evaluation or metric was computed. The held-out set's role as untouched evaluation data should still be judged with this read known.

## 7. What this audit does not say

- **Not a finding of corruption** in any checkpoint. The defect is reproduced only for N > 32,768 rows on the M5 Max under 0.30.3.
- **Not a statement on measured playing strength.** Checkpoints' measured match results remain evidence about those checkpoints.
- **Not a cause for Stage 3's value drift:** whether the defect caused it remains unresolved.
- **The cap change is a forward risk:** since `ffe6b342`, the trainer's default cap is 576. Any trainer run with B = 64 on 24×24 data would now run above 32,768 rows on the GPU. No trainer run has happened since then: the only weights written after 2026-10-20 under `checkpoints/`, `runs/` or `~/twixt-cache/` are D2's Stage 3 student, a file-time check.

## 8. Dated correction (2026-10-23, later; ledger entry 30)

- **§7, "Any trainer run with B = 64 on 24×24 data would now run above 32,768 rows on the GPU":** overstated. The
  576 cap **permits** batches above 32,768 rows; it does not make every batch exceed it. At B = 64 a batch exceeds
  32,768 rows only when its padded width M is greater than 512, that is, when some position in the batch has more than
  512 legal moves. The forward risk stands in that form.
- **§6:** the user's assessment is that the held-out access **exceeded this audit's training-only scope**. The
  disclosure stands; no further corpus access was made.
- **What the synthetic probe adds** (entry 30, MLX 0.30.3 on the M5 Max only): the failure is not limited to narrow
  operands (a 64 → 64 product fails too), and at 32,769 rows the band is 32 rows, not N − 32,736. Nothing here
  changes a label in §4.
