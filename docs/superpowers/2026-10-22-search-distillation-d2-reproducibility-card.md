# Search distillation D2 — reproducibility diagnostic card

**Date:** 2026-10-22

**Status:** FROZEN before execution.
- **Approved by the user** after the gradient diagnostic's MISMATCH (ledger entry 24, `a5abb7bd`): preparation and ONE invocation, capped at 10 minutes.
- The Stage 3 candidate stays closed, and `calib020_0001` stays the incumbent.
- **Not approved:** Stages 4–5, further learning experiments, promotion and pushing. Every other gate stays closed.

## 1. Purpose

The gradient diagnostic reproduced Stage 3's logged values bit for bit through update 20, and diverged at update 21, the first spike. That establishes where the trace diverged, not why. Matching scalar logs through update 20 does not establish identical weights or Adam state.

This diagnostic asks whether the update path repeats. It replays updates 0–21 three times in one process and compares state, inputs and outputs exactly.

Three passes can establish **observed repeatability** and whether instrumentation **accompanies** a difference. They cannot conclusively identify a GPU cause, and they cannot explain Stage 3's value drift. The record holds measurements only.

## 2. Inputs (pinned, as the gradient diagnostic pins them)

- **Corpus:** `corpus.json` SHA-256 `a9a425f28c296e36ba334e55bcd26c56164e982145f11a09427436f5351edeb0`. Read by `search_distillation_d2_gradtrace.read_training`, unchanged:
  - only the 108 training games' files are opened, and no held-out game file;
  - the 6,403 training records' identities must hash to `e90df4d31214eb95d496eb1f30a3945f1aa38b938e44ae0e139600d859cfe07d`.
- **References:**
  - Stage 3's record, SHA-256 `30b1d3ad6adb4dc575c5c33c4d5ae590324a5e9f1ade4eb0bb89f77ecd667a94`;
  - the gradient diagnostic's `trace.jsonl`, SHA-256 `04a69366864e1fa9c2d7a2d6ad126297e04c424be069500c832c4c47927b316a`, with updates 0–21.
- **Incumbent:** `calib020_0001`, SHA-1 `209cf2d4fd24a48553d259dd71b4954867b9473e`, state digest `ae0e71fc395ce6b95d24f9e696f02870c7b3bfc6f8ea83f9d5d0a85c4979816b`.

## 3. Protocol (frozen)

- **Teacher:** loaded once, evaluation mode, equal to the pinned digest. Its values on the training records are computed once, as Stage 3 computed them (`search_distillation_d2.forward`, chunks of 64), and their digest is recorded. All three passes use those same values.
- **Three passes, in this order, each on updates 0–21:**

| Pass | Attribution |
|---|---|
| **A** | none |
| **B** | before updates **20 and 21**, as the gradient diagnostic placed it (`search_distillation_d2_gradtrace.attribute`, unchanged) |
| **C** | none |

  B runs at update 20 as well as 21 because update 20 is where the suspected interference in the gradient diagnostic began. Running C after B shows whether two uninstrumented passes agree across an instrumented one.
- **Identical inputs for every pass:**
  - a fresh load of the incumbent, required to equal the pinned digest;
  - a fresh MLX `optim.Adam` at 1e-4, with resolved defaults checked;
  - the same training records and the same `default_rng(0)` draw order;
  - the same teacher values.
- **Recipe:** unchanged: batch 64, clip 1.0, anchor 1.0, L2 1e-4.
- **Loop:** a step-for-step copy of `search_distillation_d2.train_student`. Nothing runs inside it beyond two things:
  - the digests at update 21, in every pass;
  - pass B's attribution.

## 4. Recorded, each written and flushed to disk as soon as it exists

- **`updates.jsonl`:** one line per pass and update. It holds the batch indices; the loss, policy, anchor and pre-clip norm; and Stage 3's and the gradient diagnostic's values for that update.
- **`digests.jsonl`:** for every pass, at update 21, at three points. The model digest is `named-tensors/1` over every parameter and BatchNorm statistic; the Adam digest covers the step count, learning rate and every first and second moment.
  - **before the attribution point:** model state, Adam state, and the update's input and target arrays (boards, move rows and columns, mask, targets, teacher values), each digested;
  - **just before the update:** the same again. In pass B, attribution runs between the two points, so equal digests show the attribution left the state unchanged;
  - **after the update:** model and Adam state.
- **`attribution/pass_B_update_20.json` and `_21.json`:** each written create-only and fsynced **before that update is computed**, and so before any comparison. A later stop cannot discard it, and a focused test proves this.
- **`measurements/pass_<A|B|C>.json`:** written **after** that pass's 22 updates, so the measurements cannot perturb the pass.
  - The pass's state just before update 21 is reloaded into a fresh network, and its digest must equal the recorded one. Update 21's batch is rebuilt from the same records, and its input digest must equal the recorded one.
  - Measured per record:
    - the encoder output's largest magnitude;
    - the **policy-head inputs** (the gathered features at the legal moves): largest magnitude and mean per channel;
    - the hidden layer's largest magnitude;
    - the **legal-move logits**: maximum, minimum, mean, standard deviation, spread and largest magnitude;
    - the softmax maximum and entropy, the policy CE and the target-weighted logit.
  - Also measured: the BatchNorm taps (as in the gradient diagnostic), and a recomputation of update 21's loss terms and per-tensor gradient norms at that state.
- **`record.json`:** the inputs, pass timings, file list and an exact comparison. The comparison covers:
  - each pass against Stage 3 and against the gradient diagnostic: the updates differing bit for bit, and those outside the gradient diagnostic's tolerance, `|d| ≤ 1e-8 + 1e-5 × |ref|`;
  - each pair of passes: updates differing; whether the digests are equal at each point; whether the measurements are equal;
  - whether pass B's attribution left model and Adam state unchanged.

## 5. Stops and outcome

- **A difference is an observation and never stops the run.** That covers a difference between passes, from Stage 3 or from the gradient diagnostic.
- **COMPLETED (exit 0):** all three passes and their measurements ran.
- **VOID (exit 3):**
  - an input, reference, identity, plan or optimizer-default mismatch;
  - a measurement network or batch not matching its recorded digest;
  - a non-finite update (its line is written first);
  - any exception;
  - SIGTERM.
- **REFUSED (exit 4)** and **USAGE (exit 2)**, as before.
- **Not done:** no held-out game file is read, no new games, no checkpoint saved, no parameter or recipe change, no retry, no push.

## 6. Predictions on record

- **None stated in advance.** The gradient diagnostic's own prediction P1 failed, and either outcome here is plausible.

## 7. Ceiling, cleanup and record

- **Ceiling:** 10 minutes in all. The wrapper sends TERM to the run's process group at 590 s and KILL 10 s later.
- **Gate and cleanup:** the wrapper restores the gate on every exit, reads it back against the pre-open commit and reports surviving processes.
- **Kept in every case:** every file in §4 (partial if stopped), `progress.jsonl`, stdout, stderr and `outer.json`.
- No game seeds are drawn.

## 8. Gate and command

- **Gate:** `SEARCH_DISTILLATION_D2_GRADREPEAT_AUTHORIZED` in `scripts/GPU/alphazero/search_distillation_d2_gradrepeat_authorization.py`. A one-line commit opens it.
- **Command:** run from a clean linked worktree whose HEAD is that gate-opening commit:

  `bash docs/superpowers/evidence/search-distillation-d2-gradrepeat-run/01_wrapper.sh`

  The wrapper runs `env PYTHONPATH=. <main>/.venv/bin/python -B -m scripts.GPU.alphazero.search_distillation_d2_gradrepeat run --checkpoint <main>/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`.
