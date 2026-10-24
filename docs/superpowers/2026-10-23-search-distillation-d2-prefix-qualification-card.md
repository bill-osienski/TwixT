# Search distillation D2 — full-float32 setting: 21-update diagnostic prefix, qualification card

**Date:** 2026-10-23. **Status:** DRAFT FOR REVIEW — design only. Not frozen; nothing built; no gate exists for it.
Every gate stays closed until the user has reviewed this criterion. After review, freezing it, building it and running
it each need their own approval.

**Would approve, once frozen:** preparation and one bounded invocation, capped at 10 minutes, in one fresh process
launched with `MLX_ENABLE_TF32=0`. It includes **42 in-memory diagnostic optimizer updates** (two prefixes of 21, §4),
with no state saved. **Not approved:** held-out files, parameter or package changes, any other environment switch,
persistent configuration changes, checkpoint or model-state output, retries, follow-up investigation during the run,
training, a candidate checkpoint, any strength claim, bug-report submission, pushing.

## 1. Purpose and limits

Entry 32's phase 3, the updated-state check, never ran: phase 2 failed first. Entries 33 and 34 explained that
failure as a numerical branch change. This card completes the prefix under a **prospective** criterion (§6), stated
before any run at the updated state.

- **Entry 32 stays a FAIL under its original rule.** The new criterion is not applied to the incumbent; §3 only
  reproduces entry 32's digests there.
- **The tolerance is entry 26's, unchanged:** `max|a − b| ≤ 1e-3 × max|b| + 1e-6`. A failing native comparison can
  be accepted only through the explicit branch accounting of §5, never through a wider tolerance.
- **The prefix stays diagnostic:** training records only, no candidate checkpoint, no strength claim. A pass would
  qualify the setting's numerics at one updated state on two batches. It would not qualify training.

## 2. The process and inputs

- **Launch, refusals and packages:** as entries 32–34. `MLX_ENABLE_TF32` exactly `"0"` at start, with no `mlx`
  module imported; then the gate, card and pins. MLX 0.30.3, NumPy 2.4.1, Python 3.14.9, the M5 Max.
- **Inputs, verified first** (a mismatch is VOID): the training corpus through entry 24's training-only reader; the
  checkpoint's SHA-1 and the incumbent's digest; MLX's Adam defaults; Stage 3's draw order (seed 0); the teacher
  values, computed once on the GPU, whose digest must equal entry 32's (`abe2873a…`).
- **Reused code, pinned by SHA-256, unchanged:**
  - `search_distillation_d2_tf32path.py`: the evaluation block (`full_block`), the comparison and the sink;
  - `search_distillation_d2_gradcheck.py`: the prefix loop (`reconstruct`) and the policy-head NumPy reference
    (`reference_case`);
  - `search_distillation_d2_branchtrace.py`: the instrumented trace, the local checks, the flips and the accounting;
  - `search_distillation_d2_convref.py`: the float64 convolution weight gradient;
  - `search_distillation_d2_gradtrace.py` and `search_distillation_d2.py`, as before.
- **New code, small and reviewed with the build:**
  - float64 references for the encoder's conv-bias and BatchNorm weight and bias gradients;
  - the accounting generalized from the two conv2 targets to any encoder parameter tensor;
  - the pre-tanh value read (§5);
  - the judge (§6).
- **Pinned evidence:** entry 32's `record.json` and `full.jsonl`; entry 34's `record.json`.

## 3. Step 1 — reproduce at the incumbent (a precondition, not a judgment)

On each device, one uninstrumented evaluation at `calib020_0001` on Stage 3's update-0 and update-21 batches. The
loss terms and the gradient, logits and values digests must equal entry 32's first evaluation of that device and
batch, bitwise. Otherwise the run stops `NOT_REPRODUCED`.

## 4. Step 2 — the prefix, twice

- **P1:** a fresh load of the incumbent, then Stage 3's updates 0–20 on the GPU, through `reconstruct` unchanged:
  Adam (lr 1e-4), clip 1.0, batch 64, draw seed 0, the teacher values above. Each update's loss terms and pre-clip
  norm are written as it finishes, beside Stage 3's (an observation; the setting changes the path).
- **P2:** a second fresh load and the same 21 updates, in the same process.
- **Recorded:** U21's digest after each prefix. U21 is held in memory and never saved.
- **This tests entry 25's finding directly.** There, passes over the same inputs in one process differed from update 0.

## 5. Step 3 — checks at U21 (P1's state)

On Stage 3's update-0 and update-21 batches. Batch 21 is the one whose gradient Stage 3 spiked on.

- **Native evaluations:** `full_block` unchanged. On each device, three evaluations from a fresh network set from U21
  and fresh batch arrays, both digests checked. Each gives the loss terms, logits, values and all 60 gradient tensors,
  plus half + half additivity.
- **Raw values:** in each evaluation, the value head's pre-tanh output is read through the network's existing
  `return_value_pretanh=True`. Its tanh'd values and gradients must equal the default call's, bitwise.
- **Instrumented trace:** branchtrace's zero-probe trace on each device. The loss terms, logits, values and all 60
  gradients must equal the native evaluation's, bitwise.
- **Independent local checks, in float64 from each device's own captured operands:**
  - **encoder stages:** branchtrace's forward and backward checks of every convolution, BatchNorm, residual add and
    ReLU (ReLU bitwise);
  - **encoder parameters, all 52 tensors:**
    - conv weights, by convref's weight-gradient reference;
    - conv biases, as the sum of the upstream gradient;
    - BatchNorm weights and biases, as `Σ g_a ⊙ (z − mean)/√(var + eps)` and `Σ g_a`;
    - each plus its L2 term;
  - **policy head, its 8 tensors:** entry 26's NumPy reference (`reference_case`) at U21, with the CPU's encoder
    output as its input, as in entry 26.
- **Flips:** every ReLU mask difference between the devices, counted and quantified as in entry 34.
- **Accounting, only for a tensor whose native GPU-against-CPU comparison fails:**
  - entry 34's linear float64 propagation of the encoder's flips alone (`Δpre = Δpost ⊙ m_gpu + g_post_cpu ⊙ (m_gpu −
    m_cpu)`);
  - the predicted GPU gradient is that tensor's float64 formula from the GPU's operands, with the CPU's upstream
    gradient plus Δ, plus L2;
  - it is compared with the GPU's native gradient, with Δ and without it.

## 6. The criterion (prospective)

Judged in this order. The first failing condition names the outcome, and the run stops there.

1. **Finite:** every prefix update and every evaluation finite. Otherwise `NOT_QUALIFIED: non_finite`.
2. **Repeatable:**
   - P1 and P2 bitwise equal, at every update and in U21's digest;
   - on each device, evaluations 2 and 3 bitwise equal to evaluation 1.

   Otherwise `NOT_QUALIFIED: not_repeatable`.
3. **Additive:** on each device, half + half within the tolerance of the full gradient, for every tensor. Otherwise
   `NOT_QUALIFIED: not_additive`.
4. **Forward agreement**, GPU against the CPU's first evaluation, on both batches:
   - the loss terms;
   - logits at **legal moves only** (padding bitwise equal);
   - the **pre-tanh values** (the tanh'd values recorded beside them).

   Otherwise `NOT_QUALIFIED: forward_disagreement`, and the local checks do not run.
5. **Trace fidelity:** bitwise on both devices. Otherwise `NOT_QUALIFIED: trace_not_faithful`.
6. **Local derivatives:** every backward local check passes on both devices. Otherwise `NOT_QUALIFIED:
   local_derivative_error`, then `local_forward_error` for the forward checks.
7. **Native gradients:** every GPU gradient tensor within the tolerance of the CPU's first evaluation, on both
   batches. A tensor outside it is **explained** only if all of these hold:
   - it is an encoder parameter;
   - at least one flip lies in the backward path that reaches it;
   - its prediction is within the tolerance of the GPU's native gradient **with** Δ, and outside it **without** Δ.

   Any unexplained tensor gives `NOT_QUALIFIED: unexplained_gradient`. In particular:
   - a policy-head tensor (no accounting is defined there);
   - a failure caused by the encoder-output gradient, which the value head carries in and which has no independent
     reference.

**`QUALIFIED_AT_U21`:** all seven hold. Explained tensors, and every flip, are listed in the record.

**`NOT_REPRODUCED`** (§3) stops before the prefix, with no optimizer update and no judgment.

**VOID:** an input, pin, identity, state, batch, plan or optimizer-default mismatch; an exception; SIGTERM (the
wrapper's ceiling). Under this criterion a non-finite result is a failure (1), not a VOID.

**After any `NOT_QUALIFIED`:** a report, and nothing further runs. No automatic follow-up investigation.

## 7. Evidence, written as it exists

- **In the repository**, under `docs/superpowers/evidence/search-distillation-d2-prefix/`, all create-only:
  `progress.jsonl`, `reproduce.json`, `prefix.jsonl` (both prefixes, one line per update), `full.jsonl`,
  `full/U21_b<batch>.json`, `stages/<batch>_<stage>.json`, `accounting.json` and `record.json`.
- **Nothing outside it:** no fixtures, no arrays, no model state, no checkpoint. Flip operands are recorded in the
  stage files.
- **Elapsed time:** an observation, not a benchmark.

## 8. Predictions (to be stated before freezing)

Proposed:
1. §3 reproduces bitwise, and the teacher values' digest equals entry 32's.
2. Everything else, the outcome included: **None stated in advance.**

## 9. What a pass would not establish

- That training on the GPU is reliable, or that any branch change is harmless.
- Which device is more accurate.
- Anything for other states, batches, more updates, the failed candidate or Stage 3's value drift.
- Anything for another MLX version or machine.

## 10. Questions for review before freezing

1. **Two prefixes (42 updates) rather than one (21).** The second tests entry 25's in-process non-repeatability
   directly. With one prefix, criterion 2 covers only the evaluations.
2. **"Raw values" read as the pre-tanh output.** The alternative is the model's tanh'd values as returned, which is
   entry 32's comparison.
3. **No accounting for additivity, the policy head or the encoder-output gradient.** A failure in any of them stops
   qualification as unexplained. Extending the accounting there would be new design.

## 11. Ceiling, cleanup and gate (for the build)

- **Gate:** a new gate, the forty-fourth, created closed with the build, and opened only for the approved run.
- **Ceiling:** 10 minutes, with TERM at 590 s and KILL 10 s later.
- **Restored and read back:** the gate, on every exit.
- **After the evidence commit:** surviving processes reported; the linked worktree and branch removed.
- **No retry, whatever the result.**
