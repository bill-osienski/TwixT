# Search distillation D2 — gradient diagnostic card

**Date:** 2026-10-22

**Status:** FROZEN before execution.
- **Approved by the user** after accepting Stage 3's STOP at `1b52d3d5`: routine preparation and ONE instrumented replay, as a **diagnostic-only exception to do-not-repeat #69's replay restriction**.
- The exception does not reopen the Stage 3 candidate, which stays a research artifact, never used.
- **Not approved:** Stages 4–5, further learning experiments, promotion and pushing. Every other gate stays closed.

## 1. Purpose

Stage 3 logged a global pre-clip gradient norm above 10,000 on 86 of its 401 updates, with ordinary loss values on those updates. The log could not show which tensors, records or loss terms carried those norms.

This diagnostic replays the same 401 updates, from the same incumbent, in the same draw order, and records what the log could not.

It is **training computation recorded as a diagnostic replay**. It is not a learning experiment:
- it judges nothing about the student;
- it saves no checkpoint;
- it names no cause.

Raw gradient peaks, especially before clipping, do not by themselves establish the cause of Stage 3's value drift. The ledger keeps **measured attribution** (what the replay recorded) separate from **causal hypotheses** (what might explain it).

## 2. Pre-run inspection (read-only, before this card was frozen)

These are measurements from Stage 3's committed log and from reading the checkpoint's tensors. Nothing was trained to obtain them.
- **Spikes are bimodal.** Of the 401 logged norms, 303 are below 10, 10 are between 10 and 100, none between 100 and 1,000, 2 between 1,000 and 10,000, and 86 above 10,000.
  - The loss terms on spike updates are ordinary. Median policy CE is 4.13 on spike updates and 4.10 on the others; median anchor term is 0.0158 against 0.0152.
- **The spikes are not explained by a fixed set of records.** The draw order (`default_rng(0)` permutations of 6,403 indices) is known without training.
  - 18 records appear only in spike batches. They cover 50 of the 86 spike updates.
  - The other 36 spike batches contain no record that is absent from every non-spike batch.
  - So no fixed set of records, each spiking whenever drawn, accounts for all 86. A record's effect may still depend on the parameters at that update.
- **Zero-variance BatchNorm channels.** Of the 13 encoder BatchNorm layers, 12 have channels whose stored `running_var` is exactly 0. There are 297 such channels in all, and 437 below 1e-6. Their stored means are below 1e-20.
  - In evaluation mode, MLX normalizes with `(x − running_mean) / sqrt(running_var + 1e-5)`. A zero-variance channel is therefore multiplied by about 316 before its learned scale.
  - The policy head's BatchNorm has no such channel.
  - This is a property of the checkpoint. Whether it is connected to the spikes is what §5's BatchNorm taps measure.

## 3. Inputs (pinned)

- **Corpus:** `docs/superpowers/evidence/search-distillation-d2-generate/corpus.json`, SHA-256 `a9a425f28c296e36ba334e55bcd26c56164e982145f11a09427436f5351edeb0`, with the split unchanged.
- **Training games only.** Each phase's `context.json` and `finalized.json` must match the manifest's hashes. The receipts must match the completion marker and be one per scheduled game.
  - Only the 108 training games' files are opened, each checked against its receipt's size and hash and validated as Stage 3's reader validates it.
  - **No held-out game file is opened.** The record lists every file opened. The 12 held-out games are named only by the manifest's split, and the run refuses if any held-out game path is among the files opened.
- **Training records:** 6,403, in Stage 3's order (phase, then game, then ply). The SHA-256 of their row identities in that order is pinned: `e90df4d31214eb95d496eb1f30a3945f1aa38b938e44ae0e139600d859cfe07d`.
  - That digest was computed during preparation with this diagnostic's own training-only reader.
- **Reference trace:** Stage 3's record, `docs/superpowers/evidence/search-distillation-d2-train/record.json`, SHA-256 `30b1d3ad6adb4dc575c5c33c4d5ae590324a5e9f1ade4eb0bb89f77ecd667a94`.
  - It must be COMPLETED, of this recipe, with the pinned teacher digest, Stage 3's student digest `9c73feb215961b074cdf775b485f7c910152b5a4c20ba28bd2875a73b833ffdc`, and steps 0–400 in its log.
- **Incumbent:** `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`, SHA-1 `209cf2d4fd24a48553d259dd71b4954867b9473e`, state digest `ae0e71fc395ce6b95d24f9e696f02870c7b3bfc6f8ea83f9d5d0a85c4979816b`.
  - It is loaded twice, as teacher and student, both in evaluation mode, both equal to the pinned digest.

## 4. The replay

- **Recipe and draw order:** exactly Stage 3's: 401 updates, batch 64, `default_rng(0)` permutations, MLX `optim.Adam` at 1e-4 with its resolved defaults read back and checked, clip 1.0, anchor 1.0, L2 1e-4.
  - Nothing else changes: no alternative recipe, and no change to normalization, evaluation mode or update surface.
- **Anchor targets:** the teacher's values on the training records, computed as Stage 3 computed them (`search_distillation_d2.forward`, chunks of 64).
- **Loop:** a step-for-step copy of `search_distillation_d2.train_student` with read-only instrumentation.
  - The synthetic checks show that `train_student`, the bare replay and the replay attributing every update leave the same student, to the bit.
  - They also show that the replay reproduces a Stage 3 run's trace and student to the bit, on one device.
- **Not saved:** the student exists only in memory.
- **Kept unchanged:** the teacher, every off-surface tensor and every BatchNorm statistic. This is checked after the replay, as in Stage 3.

## 5. Recorded

**Every update** (`trace.jsonl`, one line per update):
- the batch's 64 training-record indices; the record maps each index to its game and ply;
- the loss, policy and anchor terms and the global pre-clip norm, beside the reference's;
- each of the 60 surface tensors' pre-clip gradient norm;
- the clip scale and the clipped global norm;
- each surface tensor's applied update: the L2 norm and the largest absolute element of after − before.

**Steps 20, 21 and 361 only:** detailed attribution, at the parameters just before that step's update, read-only.
- Step 20 is the nearby control (norm 4.86), step 21 the first spike (11,404) and step 361 the largest (43,779).
- The batch's gradient of each loss component (policy, anchor, L2) and of their sum. For each, the global norm and per-tensor norms are recorded, along with how closely the components add up to the total.
- For each of the 64 records:
  - its policy CE, squared and absolute value difference from the teacher, and number of legal moves;
  - the global and per-tensor norms of its own policy gradient and its own anchor gradient;
  - the check that the per-record gradients sum to the batch's.
- **BatchNorm taps.** For each record and each of the 14 BatchNorm layers, the largest normalized input `|x − running_mean| / sqrt(running_var + eps)`. This is recorded separately for channels with `running_var < 1e-6` (with the channel) and for the others.
  - The encoder is re-run layer by layer for this, and the largest difference from the module's own output is recorded.

**Summary (measured):** for spike updates (reference norm above 10,000) and for the others, which surface tensor carries the largest share of the squared pre-clip norm, how often, and the smallest and median share.

## 6. Tolerance and outcome

- **Trace comparison:** at every update, each of loss, policy, anchor and grad_norm must satisfy `|replayed − reference| ≤ 1e-8 + 1e-5 × |reference|`. A non-finite value never satisfies it.
- **MISMATCH (exit 6):** the first update outside the tolerance stops the replay there. The record names the update, the fields and both values. Attribution for later steps is not run.
- **REPRODUCED (exit 0):** all 401 updates are within the tolerance. Whether the final student state equals Stage 3's candidate digest bit for bit is recorded, not gated.
- **VOID (exit 3):**
  - a corpus, split, reference, record-identity, checkpoint, network or optimizer-default mismatch;
  - any exception;
  - SIGTERM.
- **REFUSED (exit 4)** and **USAGE (exit 2)**, as in Stage 3.

## 7. Predictions on record

- **P1:** the replay reproduces the trace within the tolerance. MLX's Metal kernels are expected to be deterministic for fixed shapes.
- **P2:** on spike updates, a single surface tensor carries more than half the squared pre-clip norm. This is suggested by how bimodal the logged norms are.
- **None stated** on which tensor, which records or which loss component, nor on any link to the value drift.

## 8. Ceiling, cleanup and record

- **Ceiling:** 10 minutes in all. The wrapper sends TERM to the run's process group at 590 s and KILL 10 s later.
- **Gate and cleanup:** the wrapper restores the gate on every exit, reads it back against the pre-open commit and reports surviving processes.
- **No retry:** no retry, rerun or extension, whatever the outcome.
- **Kept in every case:** `record.json`, `progress.jsonl` (a line every 25 updates and on the last), `trace.jsonl` (partial if stopped), stdout, stderr and `outer.json`.
- No game seeds are drawn; the draw-order seed 0 is not a game seed.

## 9. Gate and command

- **Gate:** `SEARCH_DISTILLATION_D2_GRADTRACE_AUTHORIZED` in `scripts/GPU/alphazero/search_distillation_d2_gradtrace_authorization.py`. A one-line commit opens it.
- **Command:** run from a clean linked worktree whose HEAD is that gate-opening commit:

  `bash docs/superpowers/evidence/search-distillation-d2-gradtrace-run/01_wrapper.sh`

  The wrapper runs `env PYTHONPATH=. <main>/.venv/bin/python -B -m scripts.GPU.alphazero.search_distillation_d2_gradtrace run --checkpoint <main>/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`.
