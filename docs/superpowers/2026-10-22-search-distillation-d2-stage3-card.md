# Search distillation D2 — Stage 3 training card

**Date:** 2026-10-22

**Status:** FROZEN before execution.
- **Approved by the user** after accepting Stage 2 at `b189a4ae`: the entry point, the gate, this card and its command, focused tests and one training run.
- **Not approved:** Stages 4–5, promotion and pushing. Every other gate stays closed.

## 1. Purpose

Train one student copy of `calib020_0001` on the Stage 2 corpus, then check on the held-out games that it fits the search targets better than the incumbent without moving its value output beyond fixed limits.

A pass establishes fit and bounded value drift. It does not establish stronger play.

## 2. Inputs (pinned)

- **Corpus:** `docs/superpowers/evidence/search-distillation-d2-generate/corpus.json`, SHA-256 `a9a425f28c296e36ba334e55bcd26c56164e982145f11a09427436f5351edeb0`. It names the corpus at `~/twixt-cache/search-distillation-d2/corpus-20261022T162940Z/`.
  - Each phase's `context.json` and `finalized.json` must match the manifest's hashes.
  - Each phase is reopened through the strict capture reader, which checks the receipts and every game file by hash.
  - The rebuilt record count per phase must equal the manifest's.
- **Incumbent:** `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`, SHA-1 `209cf2d4fd24a48553d259dd71b4954867b9473e`, model state digest (`named-tensors/1`, all parameters including BatchNorm running statistics) `ae0e71fc395ce6b95d24f9e696f02870c7b3bfc6f8ea83f9d5d0a85c4979816b`. The corpus manifest and every phase's capture context must name the same digest.
- **Split:** the corpus's own, unchanged.
  - Held out: the 12 games with `slot % 10 == 9`, giving **617 records**.
  - Training: the other 108 games, giving **6,403 records**.
  - Any other count stops the run.
- **Records:** one per ply, as the Stage 0 reader rebuilds them. The target is each row's after-search child visits; there is no mirroring. Training records are in corpus order: phase, then game, then ply.

## 3. Networks

- The incumbent is loaded twice, as the **teacher** and the **student**. Both run in evaluation mode, with no module in training mode.
- Both must equal the pinned digest before training.
- **Update surface:** encoder and policy-head parameters only (`trainer.MainModule`'s trainable parameters).
- **Never updated:** the teacher, the value-head weights and every BatchNorm running statistic. After training, every tensor off the surface must be bitwise unchanged, at least one surface tensor must have changed, and the teacher's digest must be unchanged.

## 4. Recipe (frozen)

| Field | Value |
|---|---|
| Updates | **401** = ceil(4 × 6,403 / 64), i.e. four passes |
| Batch | 64 |
| Draw order | `numpy.random.default_rng(0)` permutations, concatenated as needed |
| Optimizer | MLX `optim.Adam`, learning rate 1e-4. The resolved defaults are read back from the optimizer object and must be betas (0.9, 0.999), eps 1e-8 and no bias correction; they are recorded |
| Gradient clipping | Global norm 1.0 (`trainer.clip_grad_norm`) |
| Loss | Policy cross-entropy to the normalized counts + 1.0 × mean (v_student − v_teacher)² + 1e-4 × sum of squares over the update surface |
| Teacher values | The teacher's own values on the training records, computed once before training |

The training loop is `search_distillation_d2.train_student`. It runs on the GPU in one process.

## 5. Use of the held-out records

The held-out records are read twice and only twice:
1. **Before training:** the incumbent baseline, consisting of the teacher's held-out policy cross-entropy and its held-out values (the reference for drift).
2. **After training:** the final check, run on the reloaded candidate.

There is no intermediate evaluation, early stopping, checkpoint selection or retuning. Training-set metrics for the teacher and the candidate are recorded as diagnostics and judge nothing.

## 6. Candidate

- One final candidate is saved create-only, outside the repository, at `~/twixt-cache/search-distillation-d2/student-stage3-<UTC stamp>.safetensors`. Its SHA-256 and state digest are recorded.
- It is reloaded strictly into a freshly built network, which must be in evaluation mode with a state digest equal to the saved one and to the trained student's.
- The candidate is saved even when training stops. A failed candidate is kept as a research artifact, never used.

## 7. Verdict

**PASS** requires all of:
- training finished with every check in §3;
- every baseline and final metric is finite;
- held-out policy cross-entropy is **strictly lower** than the teacher's;
- held-out absolute value drift |v_candidate − v_teacher| has a mean **≤ 0.05**;
- held-out absolute value drift has a 95th percentile **≤ 0.15**.

The maximum drift and the KL divergence are reported, not gated. If any requirement fails, the verdict is **STOP**, with every failed criterion named. Exits: PASS 0, STOP 6.

## 8. Stops, ceiling and record

- **VOID** (exit 3):
  - a corpus, split, identity, optimizer-default or reload mismatch;
  - any exception;
  - SIGTERM.
- **Ceiling:** 30 minutes in all. The wrapper sends TERM to the run's process group at 1,790 s and KILL 10 s later.
- **Gate and cleanup:** the wrapper restores the gate on every exit, reads it back against the pre-open commit and reports surviving processes.
- **No retry:** no retry, rerun or extension, whatever the outcome.
- **Diagnostics kept:** in every case, `record.json`, `progress.jsonl` (a line every 25 updates and on the last), stdout, stderr, `outer.json` and any candidate.
- **Recorded:**
  - the corpus verification and the recipe;
  - the resolved Adam settings, the device and the MLX version;
  - every update's loss, policy term, anchor term and gradient norm;
  - the surface checks and every digest;
  - the baseline and final metrics, the criteria and the verdict;
  - the loaded-module hashes, the environment and the load average.

No game seeds are drawn; the draw-order seed 0 is not a game seed.

## 9. Gate and command

- **Gate:** `SEARCH_DISTILLATION_D2_TRAIN_AUTHORIZED` in `scripts/GPU/alphazero/search_distillation_d2_train_authorization.py`. A one-line commit opens it.
- **Command:** run from a clean linked worktree whose HEAD is that gate-opening commit:

  `bash docs/superpowers/evidence/search-distillation-d2-train-run/01_wrapper.sh`

  The wrapper runs `env PYTHONPATH=. <main>/.venv/bin/python -B -m scripts.GPU.alphazero.search_distillation_d2_train run --checkpoint <main>/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`.
