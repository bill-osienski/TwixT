# Search distillation D2 — full-float32 setting: 21-update diagnostic prefix, qualification card

**Date:** 2026-10-23. **Status:** FROZEN, with the user's amendments after review of the draft (`8b255801`). **Gate:**
`SEARCH_DISTILLATION_D2_PREFIX_AUTHORIZED` (the forty-fourth), created closed with the build.

**Approved:** freezing this card, a minimal implementation and focused tests. **Not approved yet: execution.** The
frozen implementation comes back for review first; opening the gate for the one invocation below needs its own
approval.

**The invocation, if approved:** one, capped at 10 minutes, in one fresh process launched with `MLX_ENABLE_TF32=0`.
It includes **42 in-memory diagnostic optimizer updates** (two prefixes of 21, §4), with no state saved. **Never
approved here:** held-out files, parameter or package changes, any other environment switch, persistent configuration
changes, checkpoint or model-state output, retries, follow-up investigation during the run, training, a candidate
checkpoint, any strength claim, bug-report submission, pushing.

## 1. Purpose and limits

Entry 32's phase 3, the updated-state check, never ran: phase 2 failed first. Entries 33 and 34 explained that
failure as a numerical branch change. This card completes the prefix under a **prospective** criterion (§6), stated
before any run at the updated state.

- **Entry 32 stays a FAIL under its original rule.** The new criterion is not applied to the incumbent; §3 only
  reproduces entry 32's digests there.
- **The tolerance is entry 26's, unchanged:** `max|a − b| ≤ 1e-3 × max|b| + 1e-6`. A failing native comparison can
  be accepted only through the explicit branch accounting of §5, never through a wider tolerance, and there is no
  blanket exemption for near-zero activations.
- **The prefix stays diagnostic:** training records only, no candidate checkpoint, no strength claim. A pass would
  qualify the setting's numerics at one updated state on two batches. It would not qualify training.

## 2. The process and inputs

- **Launch, refusals and packages:** as entries 32–34. `MLX_ENABLE_TF32` exactly `"0"` at start, with no `mlx`
  module imported; then the gate, card and pins; the checkpoint by absolute path; no evidence directory. MLX 0.30.3,
  NumPy 2.4.1, Python 3.14.9, the M5 Max.
- **Inputs, verified first** (a mismatch is VOID):
  - the training corpus through entry 24's training-only reader;
  - the checkpoint's SHA-1 and the incumbent's digest;
  - MLX's Adam defaults;
  - Stage 3's per-update record, by its pinned hash (read only to write Stage 3's values beside the prefix's);
  - Stage 3's draws for updates 0 and 21 (seed 0), which must equal entry 32's.
- **Reused code, pinned by SHA-256, unchanged:**
  - `search_distillation_d2_tf32path.py`: the evaluation block (`full_block`), the comparison and the sink;
  - `search_distillation_d2_gradcheck.py`: the policy-head NumPy reference (`np_policy_head`, `reference_case`) and
    the batch draws;
  - `search_distillation_d2_branchtrace.py`: the evaluation and instrumented trace, the local checks, the flips, the
    float64 transposed convolution and BatchNorm;
  - `search_distillation_d2_convref.py`: the float64 convolution weight gradient;
  - `search_distillation_d2_gradtrace.py` and `search_distillation_d2.py`, as before.
- **New code, in `search_distillation_d2_prefix.py`:**
  - the prefix loop: `gradcheck.reconstruct`'s loop with per-update digests added. Its final state must equal
    `reconstruct`'s bitwise (a test);
  - the pre-tanh capture (§5);
  - float64 references for the encoder's conv-bias and BatchNorm weight and bias gradients;
  - the native policy-head check (§5);
  - the accounting generalized from entry 34's two conv2 targets to every encoder parameter tensor;
  - the judge (§6).
- **Pinned evidence:** entry 32's `record.json` and `full.jsonl`.

## 3. Step 1 — reproduce at the incumbent (a precondition, not a judgment)

- The teacher values, computed once on the GPU as before. Their digest must equal entry 32's (`abe2873a…`).
- On each device, one uninstrumented evaluation at `calib020_0001` on Stage 3's update-0 and update-21 batches. The
  loss terms and the gradient, logits and values digests must equal entry 32's first evaluation of that device and
  batch, bitwise.
- Otherwise, a non-finite evaluation included (entry 32's were finite), the run stops **`NOT_REPRODUCED`**: no
  optimizer update, no judgment.

## 4. Step 2 — the prefix, twice

- **P1 and P2,** in this process, one after the other. Each starts from **a fresh load of the incumbent** (digest
  checked) and **a fresh Adam**. Each runs Stage 3's updates 0–20 on the GPU with **identical batches** (draw seed 0)
  and **identical teacher targets** (the array above): Adam (lr 1e-4, MLX's defaults otherwise), clip 1.0, batch 64.
  Nothing else runs inside the loop.
- **Recorded after every update**, written as it finishes:
  - the loss terms and the pre-clip gradient norm, with Stage 3's beside them (an observation; the setting changes
    the path);
  - **the model digest** (every parameter and BatchNorm statistic);
  - **the Adam-state digest** (step, learning rate and every first and second moment).
- **Compared, update by update:** P1 against P2, the scalars, the model digest and the Adam digest; then U21's digest.
  Matching losses and a matching final digest alone are not enough.
- **U21** is P1's state after update 20. It is held in memory and never saved.
- **This tests entry 25's finding directly.** There, passes over the same inputs in one process differed from update 0.

## 5. Step 3 — checks at U21

On Stage 3's update-0 and update-21 batches. Batch 21 is the one whose gradient Stage 3 spiked on.

- **Native evaluations:** `full_block` unchanged. On each device, three evaluations from a fresh network set from U21
  and fresh batch arrays, both digests checked. Each gives the loss terms, logits, values and all 60 gradient tensors,
  plus half + half additivity.
- **Values:** the tanh'd predictions the network returns and the loss uses. They are the acceptance comparison, as in
  entry 32.
- **Pre-tanh values (supplementary, not judged):**
  - on each device, one more evaluation reads them through the network's existing `return_value_pretanh=True`;
  - its loss terms and its gradient, logits and values digests must equal evaluation 1's, bitwise (criterion 5);
  - the GPU's against the CPU's is recorded.
- **Instrumented trace:** branchtrace's zero-probe trace on each device. Its loss terms and its gradient, logits and
  values digests must equal evaluation 1's, bitwise.
- **Independent local checks, in float64 from each device's own captured operands. Each must pass, not only be
  recorded:**
  - **encoder stages:** branchtrace's forward and backward checks of every convolution, BatchNorm, residual add and
    ReLU (ReLU bitwise);
  - **encoder parameters, all 52 tensors,** each reference plus its L2 term (`2 × 1e-4 × θ`), against that device's
    native gradient (evaluation 1):
    - conv weights, by convref's weight-gradient reference;
    - conv biases, as the sum of the upstream gradient;
    - BatchNorm weights and biases, as `Σ g_a ⊙ (z − mean)/√(var + eps)` and `Σ g_a`;
  - **policy head, native, its 8 tensors:**
    - `np_policy_head` on **that device's own encoder output** (`relu(s)` of the last block, from its trace);
    - the canonicalized move rows and columns, the batch's mask and targets, every record weighted 1;
    - each gradient plus its L2 term, against that device's native gradient of the full loss (evaluation 1);
  - **policy head, common input (a separate comparison):** entry 26's `reference_case` at U21, with the CPU's encoder
    output as input on both devices, judged as entry 32 judged phase 1. On a real batch, a skipped directional check
    is recorded, not a failure.
- **Flips:** every ReLU mask difference between the devices, counted and quantified as in entry 34.
- **The encoder-output gradient:** the last block's output gradient, GPU against CPU.
- **Accounting, only for a tensor whose native GPU-against-CPU comparison fails:**
  - entry 34's linear float64 propagation of the encoder's flips alone, with `Δ = 0` at the encoder's output, through
    every block and the stem: `Δpre = Δpost ⊙ m_gpu + g_post_cpu ⊙ (m_gpu − m_cpu)`;
  - the predicted GPU gradient is that tensor's float64 formula from the GPU's operands, with the CPU's upstream
    gradient plus Δ, plus L2;
  - it is compared with the GPU's native gradient, with Δ and without it.

## 6. The criterion (prospective)

The checks run in this order: the two prefixes (criteria 1 and 2); the native evaluations on both batches (criteria
1–4); then, batch by batch with the update-0 batch first, the captures, local checks and accounting (criteria 5–7). At
each point the criteria are judged in number order. The first failing condition names the outcome, and the run stops
there.

1. **Finite:** every prefix update and every evaluation finite. Otherwise `NOT_QUALIFIED: non_finite`.
2. **Repeatable:**
   - P1 and P2 equal at every update: the scalars, the model digest and the Adam digest; and U21's digest;
   - on each device, evaluations 2 and 3 bitwise equal to evaluation 1.

   Otherwise `NOT_QUALIFIED: not_repeatable`.
3. **Additive:** on each device, half + half within the tolerance of the full gradient, for every tensor. Otherwise
   `NOT_QUALIFIED: not_additive`.
4. **Forward agreement**, GPU against the CPU's first evaluation, on both batches:
   - the loss terms;
   - logits at **legal moves only** (padding bitwise equal);
   - the **values** (tanh'd, as returned).

   Otherwise `NOT_QUALIFIED: forward_disagreement`.
5. **Instrumentation fidelity:** the trace and the pre-tanh capture, bitwise on both devices. Otherwise
   `NOT_QUALIFIED: trace_not_faithful`.
6. **Local derivatives:** on both devices,
   - every backward local check of the encoder's stages;
   - every one of the 52 encoder parameter references;
   - the 8 native policy-head references;
   - the common-input policy-head check.

   Otherwise `NOT_QUALIFIED: local_derivative_error`. Then every forward local check; otherwise `NOT_QUALIFIED:
   local_forward_error`.
7. **Native gradients:** every GPU gradient tensor within the tolerance of the CPU's first evaluation. A tensor outside
   it is **explained** only if all of these hold:
   - it is an encoder parameter;
   - the encoder-output gradient is within the tolerance, GPU against CPU. Internal encoder flips cannot excuse a
     discrepancy entering from the encoder-output gradient;
   - at least one flip lies on the backward path that reaches it: at a ReLU after the tensor in the forward pass;
   - its prediction is within the tolerance of the GPU's native gradient **with** Δ, and outside it **without** Δ.

   Any unexplained tensor gives `NOT_QUALIFIED: unexplained_gradient`, a policy-head tensor included (no accounting is
   defined there).

**`QUALIFIED_AT_U21`:** all seven hold. Explained tensors, and every flip, are listed in the record.

**`NOT_REPRODUCED`** (§3) stops before the prefix, with no optimizer update and no judgment.

**VOID:** an input, pin, identity, state, batch, plan or optimizer-default mismatch; the transposed convolution's
adjoint self-check failing (1e-9 relative); an exception; SIGTERM (the wrapper's ceiling). Under this criterion a
non-finite result is a failure (1), not a VOID.

**After any `NOT_QUALIFIED`:** a report, and nothing further runs. No automatic follow-up investigation. In particular,
the accounting is not extended to additivity, the policy head or the encoder-output gradient during this run.

## 7. Evidence, written as it exists

- **In the repository**, under `docs/superpowers/evidence/search-distillation-d2-prefix/`, all create-only:
  - `progress.jsonl`, `reproduce.json`;
  - `prefix.jsonl`: both prefixes, one line per update;
  - `full.jsonl` and `full/U21_b<batch>.json`: the native evaluations;
  - `checks/b<batch>_<device>.json`: fidelity, the pre-tanh capture and every local check;
  - `reference/U21_b<batch>.json`: the common-input policy-head check;
  - `stages/b<batch>_<relu>.json`: the flips;
  - `accounting/b<batch>.json`, when a native comparison fails;
  - `record.json`, on every exit the script survives.
- **Nothing outside it:** no fixtures, no arrays, no model state, no checkpoint. Flip operands are recorded in the
  stage files.
- **Elapsed time:** an observation, not a benchmark.

## 8. Predictions on record (stated before freezing)

1. §3 reproduces bitwise, and the teacher values' digest equals entry 32's.
2. Everything else, the outcome included: **None stated in advance.**

## 9. What a pass would not establish

- That training on the GPU is reliable, or that any branch change is harmless.
- Which device is more accurate.
- Anything for other states, batches, more updates, the failed candidate or Stage 3's value drift.
- Anything for another MLX version or machine.

## 10. The review's answers (the user, 2026-10-23)

1. **Two prefixes, 42 diagnostic updates,** each from a fresh incumbent and fresh Adam, with identical batches and
   teacher targets. Per-update model and Adam-state digests are compared, not only the scalars and the final digest.
2. **"Raw values" are the tanh'd predictions** the network returns and the loss uses: the acceptance comparison, as in
   entry 32. Pre-tanh values are supplementary, and their capture must leave the outputs and gradients unchanged.
3. **Unexplained failures stop qualification.** No accounting for additivity, the policy head or the encoder-output
   gradient during this run.
4. **Added:** every independent parameter reference gates, L2 included; and the native policy-head gradients are
   validated per device from that device's own encoder output, apart from the common-input check.

## 11. Ceiling, cleanup and gate

- **Gate:** created closed with the build; opened by a one-line commit only for an approved run.
- **Ceiling:** 10 minutes, with TERM at 590 s and KILL 10 s later.
- **Restored and read back:** the gate, on every exit.
- **After the evidence commit:** surviving processes reported; the linked worktree and branch removed.
- **No retry, whatever the result.**
- **Command:** from a clean linked worktree whose HEAD is the gate-opening commit:

  `bash docs/superpowers/evidence/search-distillation-d2-prefix-run/01_wrapper.sh`
