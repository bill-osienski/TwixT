# Search distillation D2 — gradient correctness check card

**Date:** 2026-10-22

**Status:** FROZEN before execution.
- **Approved by the user** after the reproducibility diagnostic (ledger entry 25, `7dfdec03`): preparation and ONE bounded invocation, with a 10-minute ceiling.
- This is a diagnostic-only exception to the closed replay restriction.
- **Not approved:** Stages 4–5, further learning experiments, candidate scoring, recipe changes, promotion, pushing, backend upgrades and production fixes. Every other gate stays closed.

## 1. Purpose and limits

**What entry 25 measured:**
- gradients that differ from identical state and inputs;
- at update 21, a severe failure of the linearity identity: each record's own policy gradient had norm ≈ 3,867, and their sum was about 64 times the batch gradient.

It did not establish which computation is wrong. This check compares devices, compares the policy head against an independent reference, and locates the first stage whose output or vector-Jacobian product disagrees when given identical inputs.

- **CPU MLX is a comparison backend, not a correctness oracle:** it shares MLX's autodiff implementation. The independent reference is NumPy (§4), and only for the policy head.
- **Unresolved, and this check does not settle it:**
  - whether the defect affected the training of earlier checkpoints;
  - whether it caused Stage 3's value drift.

  Earlier checkpoints' measured match results remain evidence about those checkpoints.

## 2. Two fixed states

- **S0:** the unchanged incumbent, `calib020_0001` (state digest `ae0e71fc…`). Evaluated on Stage 3's **update-0 batch**.
- **S21:** a **newly reconstructed** pre-update-21 state, evaluated on Stage 3's **update-21 batch**.
  - It comes from ONE prefix of Stage 3's loop over updates 0–20 on the GPU, from a fresh load of the incumbent, with nothing inside the loop. Its values are written per update beside Stage 3's.
  - Its digest is recorded and compared with the pre-update-21 digests that entry 25 recorded: pass A, and passes B and C. **It is not described as those states.** Entry 25 kept digests, not state files.
  - S21 is held in memory only and never saved.
- **Batches:** both are from Stage 3's draw order, over the 6,403 training records. These are read by entry 24's training-only reader; no held-out game file is opened, and the identity digest is pinned.
- **Teacher values:** computed once on the GPU, as Stage 3 computed them. Their digest is compared with entry 25's.

## 3. CPU/GPU comparison of the training loss (no optimizer update)

- **Each evaluation is set up fresh.** At each state, on each device (`cpu`, then `gpu`), **three times**:
  - a fresh network is loaded and set from the state's arrays, and its digest must equal the state's;
  - fresh batch arrays (boards, move rows and columns, mask, targets, teacher values) are built, and their digest must equal the batch's;
  - the training loss of `search_distillation_d2.train_student` is evaluated: policy CE, plus 1.0 × anchor, plus 1e-4 × L2.
- **Recorded per evaluation:**
  - the loss terms;
  - the logits and values, compared element by element with the CPU's first evaluation;
  - **every one of the 60 gradient tensors**, each compared element by element with the CPU's first evaluation and between repeats, alongside the gradient norm.
- **Weighted-record additivity** (first evaluation per device): the gradient over the first half of the batch plus the gradient over the second half, against the gradient over all of it, all without the L2 term.

## 4. The independent reference: NumPy, float64, analytic

- **Scope:** the policy head, from its input features to the masked cross-entropy, in float64 with hand-derived derivatives.
  - The 1×1 convolution, then BatchNorm in evaluation mode, then ReLU.
  - The gather at the legal moves, including its scatter-add derivative, where repeated indices accumulate.
  - The two linear layers with ReLU between them, the −1e9 masks, the masked log-softmax and the weighted mean cross-entropy.
- **Derivatives taken with respect to:** every trainable policy-head tensor and the input features.
- **Validated independently of MLX:** fixed directional central differences, with ε = 1e-5 and the direction drawn with seed 0. These are used only where neither ReLU's pattern changes between the two points: ε is halved up to three times, otherwise the check is recorded as skipped. The tests also check every coordinate, on a small case.
- **Compared against MLX's gradients** of the same loss on each device, three times, with fresh copies of the tensors on each.
- **Cases:**
  - synthetic, on an 8-channel head:
    - repeated gather indices: (3, 4) three times in one record and twice in another;
    - different batch members;
    - padding at (0, 0), which another record also plays;
    - weighted-record additivity with weights 0.3, 1.7, 0 and 1.0;
  - the **real batch** at S0 and at S21, on the real head, with the CPU chain's encoder output as input.
  - In every case, MLX's single-record gradients are also summed and compared with its full gradient.

## 5. Localization: identical inputs, stage by stage

- **Stages:** the training loss's forward is split into canonicalization, encoder, policy conv, policy BatchNorm, policy ReLU, gather, fc, fc ReLU, out, head mask, masked log-softmax, policy CE, value head and anchor.
- **Reference chain:** the CPU computes the canonical stage inputs (its own forward chain) and the canonical cotangents (its own backward chain). The policy CE's cotangent is 1, the anchor's is 1.0, and the encoder's output receives the policy and value branches' sum.
- **Per stage, on each device, three times, from those identical inputs and cotangents:**
  - the stage's output;
  - its vector-Jacobian product with respect to its input and to every tensor it reads.
- **Decomposition check:** the chain's tensor gradients, composed, are compared with autodiff of the same loss (policy CE + 1.0 × anchor, no L2) through the module's own forward on the CPU. This shows the stages reproduce the computation they split.
- **First disagreement:** recorded per device, in forward order and in backward order. A disagreement is outside the tolerance (§6) or not bitwise repeatable on that device.

## 6. Declared tolerances (frozen)

- **Device or reference comparison of a float32 tensor:** `max|a − b| ≤ 1e-3 × max|b| + 1e-6`. Bitwise equality is reported alongside.
- **Repeats on one device:** bitwise. Any difference is reported.
- **Finite differences:**
  - against the analytic derivative: `|fd − an| ≤ 1e-6 × (|an| + 1)`;
  - against MLX: `|fd − mlx| ≤ 1e-3 × (|fd| + 1e-6)`.

## 7. Evidence, written as soon as it exists

- **In the repository** (`docs/superpowers/evidence/search-distillation-d2-gradcheck/`):
  - `progress.jsonl`, `reconstruction.jsonl`, `states.json` and `full.jsonl`;
  - `full/<state>.json`, `stages/<state>.json` and `reference/<case>.json`;
  - `fixtures.jsonl` and `record.json`.
- **Fixtures** for every component that disagrees are saved create-only as `.npz`, **outside the repository** at `~/twixt-cache/search-distillation-d2/gradcheck-fixtures-<UTC stamp>/`. `fixtures.jsonl` holds each one's SHA-256, array shapes and dtypes, and the declared tolerances.
  - **Contents:** the stage's input, cotangent, reference output, indices, mask, targets and teacher values, and the tensors of that stage only.
  - **Never saved:** encoder weights; for an encoder-stage fixture, the state digest identifies them.
  - **No playable checkpoint** and no full model state is saved.

## 8. Stops and outcome

- **A difference is a diagnostic observation and never stops the run.** That covers a difference between devices, between repeats, from the reference, from Stage 3 or from entry 25.
- **COMPLETED (exit 0):** every comparison in §3–5 ran.
- **VOID (exit 3):**
  - an input, reference, identity, state, batch, plan or optimizer-default mismatch;
  - a non-finite result;
  - an existing fixture directory;
  - any exception;
  - SIGTERM.
- **REFUSED (exit 4)** and **USAGE (exit 2)**, as before.
- **Not done:** no held-out game file is read, no new games, no checkpoint saved, no recipe change, no retry, no push.

## 9. Predictions on record

**None stated in advance.**

## 10. Ceiling, cleanup and gate

- **Ceiling:** 10 minutes in all. The wrapper sends TERM to the run's process group at 590 s and KILL 10 s later.
- **Gate and cleanup:** the wrapper restores the gate on every exit, reads it back against the pre-open commit and reports surviving processes.
- No game seeds are drawn.
- **Gate:** `SEARCH_DISTILLATION_D2_GRADCHECK_AUTHORIZED` in `scripts/GPU/alphazero/search_distillation_d2_gradcheck_authorization.py`. A one-line commit opens it.
- **Command:** run from a clean linked worktree whose HEAD is that gate-opening commit:

  `bash docs/superpowers/evidence/search-distillation-d2-gradcheck-run/01_wrapper.sh`

  The wrapper runs `env PYTHONPATH=. <main>/.venv/bin/python -B -m scripts.GPU.alphazero.search_distillation_d2_gradcheck run --checkpoint <main>/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`.
