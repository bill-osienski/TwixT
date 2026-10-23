# Search distillation D2 — full-float32 setting: convolution-reference diagnostic card

**Date:** 2026-10-23. **Status:** FROZEN before any run. **Gate:** `SEARCH_DISTILLATION_D2_CONVREF_AUTHORIZED` (the
forty-second), opened for ONE approved invocation by a one-line commit and restored by the wrapper.

**Approved:** preparation and one bounded invocation, capped at 10 minutes, of the diagnostic below, at the incumbent on
Stage 3's update-0 batch, in one fresh process launched with `MLX_ENABLE_TF32=0`. **No optimizer update.**
**Not approved:** held-out files, parameter changes, package changes, any other environment switch, checkpoint output,
retries, training or any learning run, bug-report submission, pushing.

## 1. Purpose and limits

Ledger entry 32 **FAILED at phase 2**. At the incumbent on the update-0 batch, two of the 60 gradient tensors were
outside the frozen tolerance on the GPU against the CPU, in all three evaluations, repeatably and additively:
`encoder.blocks.1.conv2.weight` (2.05× the tolerance, 11 elements) and `encoder.blocks.2.conv2.weight` (1.57×, 5
elements). **That verdict stands.** Nothing here changes it, reopens training or relaxes the failed criterion.

This diagnostic asks whether the remaining difference is:

- **local:** made by one backend's convolution weight-gradient calculation itself, given the same operands; or
- **inherited:** carried into that calculation by different operands from the rest of the network (the convolution's
  input, or the upstream gradient arriving at its output).

**The limit, stated in advance:** if the CPU and GPU supply different upstream gradients, a NumPy calculation of the
convolution alone **cannot establish which device computed the more accurate whole-model gradient**. An `INHERITED`
result says where the difference enters, not which backend is right.

## 2. The process

- **Launch:** the wrapper runs `python -B -m scripts.GPU.alphazero.search_distillation_d2_convref run --checkpoint
  <calib020_0001>` with `MLX_ENABLE_TF32=0` in that process's environment only. Nothing is installed or persisted. No
  other environment switch is set.
- **The entry refuses** unless:
  - `MLX_ENABLE_TF32` is exactly `"0"` when it starts, and no `mlx` module is imported at that moment;
  - then the gate is open, the card and every pinned file match, the checkpoint names an existing file by absolute
    path, and neither the evidence directory nor the fixture directory exists.

  It records the variable's value at the start and at the end.
- **Same packages and application code:** MLX 0.30.3, NumPy 2.4.1, Python 3.14.9, the M5 Max.
- **Inputs, all verified before step 1** (any mismatch is VOID):
  - the training corpus through entry 24's **training-only reader** (identity digest pinned). It opens no held-out
    game file;
  - the checkpoint's SHA-1, the loaded incumbent's state digest, and the update-0 batch's draw (it must equal the one
    entry 32 recorded).
- **Reused code, pinned by SHA-256, unchanged:**
  - `search_distillation_d2_tf32path.py` (entry 32): `full_block`, its comparison and its judge;
  - `search_distillation_d2_gradcheck.py`: the device context, the stage VJP helper (`_vjp`), the batch arrays and
    the batch draw;
  - `search_distillation_d2_gradtrace.py`: the training-only reader;
  - `search_distillation_d2.py`: the batch builder, the forward pass and the state digest.
- **Entry 32's evidence, pinned by SHA-256:** its `record.json`, `full.jsonl` and `full/S0_b0.json`. They are read
  for the integrity checks, the reproduction and the predictions only.

## 3. Step 1 — reproduce the two failing comparisons

- **Targets:** the anchor's teacher values, computed once on the GPU in this process, as in entry 32. The same array
  goes to both devices.
- **Run entry 32's `full_block` unchanged** at the incumbent on the update-0 batch: on each device (`cpu`, then
  `gpu`), three evaluations, each from a fresh network and fresh batch arrays (digests checked), plus its additivity
  check. Entry 32's judge is applied unchanged.
- **Reproduced** only if:
  - the GPU's failures are exactly entry 32's six: `encoder.blocks.1.conv2.weight` and
    `encoder.blocks.2.conv2.weight` in each of the three evaluations, and nothing else;
  - the CPU has no failure of its own (repeats, additivity).
- **If not reproduced:** the outcome is `NOT_REPRODUCED` and the run **stops**. No capture is made.

## 4. Step 2 — capture the operands, and verify the capture

On each device (`cpu`, then `gpu`), a fresh network is loaded and set from the incumbent (digest checked), with fresh
batch arrays (digest checked).

- **(a) Uninstrumented evaluation:** the training loss and its gradient, exactly as `full_block` computes them. Its
  gradient digest must equal the digest step 1 recorded for that device's first evaluation, and its loss terms must
  be equal.
- **(b) Instrumented evaluation:** the same, with the two target convolutions instrumented. A zero tensor (a "probe")
  is added to each convolution's output, after its bias, and the gradient is taken with respect to the parameters and
  the probes together. The gradient with respect to a probe is **the upstream gradient `G` arriving at that
  convolution's output**. The convolution's input `X` is returned beside the loss.
  - The instrumented loss terms, logits, values and **all 60 gradient tensors** must equal (a)'s **bitwise**.
- **(c) Local reconstruction:** for each target, on the same device:
  - the convolution's local weight gradient from the captured `X` and `G` (the stage VJP helper, on that network's
    convolution module); plus
  - **the separately calculated L2 contribution:** MLX's gradient of `1e-4 × Σ w²` for that weight on that device.

  Their sum must reconstruct (a)'s gradient for that tensor: `max|recon − orig| ≤ 1e-6 × max|orig|`. Bitwise equality
  is recorded alongside. The L2 contribution's size is recorded.
- **Capture is faithful** only if (a), (b) and (c) hold for both targets on both devices.
- **If not faithful:** the outcome is `CAPTURE_NOT_FAITHFUL` and the run **stops** before step 3. Fixtures already
  saved are kept.

## 5. Step 3 — the NumPy float64 reference

**The reference:** for a 3×3 convolution with stride 1 and padding 1, in MLX's layouts (input NHWC, weight
`(out, kh, kw, in)`), the weight gradient is

`dW[o, a, b, c] = Σ_{n, y, x} G[n, y, x, o] · Xpad[n, y + a, x + b, c]`,

computed in float64 from the float32 operands by NumPy, one matrix product per kernel offset. The L2 contribution is
`2 × 1e-4 × W` in float64.

**The reference must validate itself first** (if either check fails, VOID at step `reference`):

- **Layout:** a NumPy float64 forward convolution of the CPU's captured `X` with the incumbent's weight and bias,
  against the CPU MLX convolution module's output on the same `X`, within the comparison tolerance (§6).
- **Adjoint identity,** for each operand set: for one fixed random direction `V` (NumPy seed 0, standard normal, the
  weight's shape), `|⟨dW, V⟩ − ⟨G, conv(X, V)⟩| ≤ 1e-9 × (|⟨dW, V⟩| + |⟨G, conv(X, V)⟩|)`, with `conv` the
  bias-free float64 forward convolution.

**The comparisons, per target:**

- **Identical operands:** for each operand set (the CPU's captured `X` and `G`, then the GPU's), copied to both
  devices:
  - each device's local weight gradient, three times, against the float64 reference of **that same operand set**;
  - the GPU's local gradient against the CPU's, on the same operands;
  - repeats are recorded (bitwise or not).
- **Native operands, examined separately:**
  - each device's original whole-model gradient (step 2a) against the float64 reference of **its own** operands,
    plus the L2 contribution;
  - the GPU's `X` against the CPU's, and the GPU's `G` against the CPU's;
  - **the operand-inherited difference:** the float64 reference of the GPU's operands against the float64 reference
    of the CPU's operands;
  - the L2 contribution, GPU against CPU.
- **At the elements that failed:** for each element outside the tolerance in the GPU-against-CPU comparison of the
  original gradients (all of them are counted; up to 64 are recorded, largest difference first), both devices'
  original values, both float64 references (with the L2 contribution), and the float64 sum of the absolute terms
  from the CPU's operands, `Σ |G| · |Xpad|`, as a measure of cancellation.

## 6. Tolerances (frozen, not relaxed)

- **Comparison of a tensor:** entry 26's `max|a − b| ≤ 1e-3 × max|b| + 1e-6`, with `b` the reference (the float64
  NumPy result, or the CPU, or the CPU's operand). Bitwise equality is reported alongside.
- **Capture fidelity:**
  - instrumented against uninstrumented evaluation: bitwise;
  - local reconstruction against the original gradient: `max|recon − orig| ≤ 1e-6 × max|orig|`. That is a thousandth
    of the comparison tolerance, so any slack in the capture is too small to account for a 1.57× or 2.05× miss.
- **Reference self-checks:** layout within the comparison tolerance; adjoint identity at `1e-9` relative.

## 7. Classification (per target, then overall)

Three flags per target, each over both operand sets and all three repeats:

- **`local_gpu`:** the GPU's local gradient is outside the tolerance of the float64 reference for **either** operand
  set.
- **`local_cpu`:** the same for the CPU.
- **`inherited`:** the float64 reference of the GPU's operands is outside the tolerance of the float64 reference of
  the CPU's operands.

| local_gpu | local_cpu | inherited | class |
|---|---|---|---|
| yes | no | no | `LOCAL_GPU` |
| no | yes | no | `LOCAL_CPU` |
| no | no | yes | `INHERITED` |
| yes | no | yes | `LOCAL_GPU_AND_INHERITED` |
| no | yes | yes | `LOCAL_CPU_AND_INHERITED` |
| yes | yes | either | `BOTH_KERNELS` |
| no | no | no | `NEITHER_ALONE` (each part is within the tolerance; the miss accumulates across them) |

**Overall:** the shared class if both targets have the same class; otherwise `MIXED`, with each target's class named.

**The outcome:** `CLASSIFIED`, `NOT_REPRODUCED` or `CAPTURE_NOT_FAITHFUL` (status COMPLETED); or VOID.

- **VOID:** an input, pin, identity, state, batch or plan mismatch; the reference failing its self-checks; a non-finite
  result; any exception; SIGTERM (the wrapper's ceiling).
- **Entry 32's verdict is unchanged** by any outcome: FAIL at phase 2.

## 8. Evidence, written as it exists

- **In the repository**, under `docs/superpowers/evidence/search-distillation-d2-convref/`, all create-only:
  - `progress.jsonl`;
  - `full.jsonl` and `full/S0_b0.json`: step 1, written by `full_block` as in entry 32;
  - `capture/<device>.json`: step 2, per device, as it finishes;
  - `reference/<target>.json`: step 3, per target, as it finishes;
  - `fixtures.jsonl`: each fixture's path, SHA-256, shapes and the frozen tolerances;
  - `record.json`: written on every exit the script survives, with the setting at the start and end, each step's
    result, the classification and the comparisons with entry 32.
- **Operand fixtures**, outside the repository in a new directory
  `~/twixt-cache/search-distillation-d2/convref-fixtures-<UTC time>/`, create-only, one compressed `.npz` per device
  and target, written as soon as that device's capture exists:
  - `X` and `G` (float32, 64 × 24 × 24 × 128 each), and the MLX local weight gradient from them (step 2c).
  - **Not saved:** the weights, the L2 contribution, the original gradient, any model state or checkpoint. Their
    digests are recorded; the weights are the incumbent's.
- **A compact result per comparison:** the largest absolute difference and the scale; when outside the tolerance, the
  number of elements outside it and the worst element's index and both values.

## 9. Predictions on record (stated before the run)

1. **Reproduction:** the teacher values' digest equals entry 32's (`abe2873a…`); on both devices, every evaluation's
   loss terms, logits digest and gradient digest equal entry 32's `full.jsonl` for `S0_b0`; and step 1 reproduces
   (the GPU's six failures, nothing else).
2. **Capture fidelity:** faithful on both devices, with step 2(c) **bitwise**. The basis: a CPU check of the same
   instrumentation on a random 3-block network (not the incumbent, not the GPU).
3. **The reference's self-checks pass.**
4. **Each target's flags and class, and the overall class:** **None stated in advance.**

## 10. What this cannot establish

- Which device computed the more accurate **whole-model** gradient, when the operands differ (§1).
- The cause of any difference inside a backend's kernel.
- Anything about the other 58 tensors, other batches or states, other MLX versions or machines.
- Anything for training, the failed candidate, Stage 3's value drift or measured playing strength.
- A relaxed or re-judged entry 32: its verdict stands.

## 11. Ceiling, cleanup and gate

- **Ceiling:** 10 minutes in all. The wrapper sends TERM to the run's process group at 590 s and KILL 10 s later.
- **Gate:** the wrapper restores it on every exit and reads it back against the pre-open commit.
- **Cleanup:** surviving processes are reported. The linked worktree and its branch are removed after the evidence
  commit.
- **No retry, whatever the result.**
- **Command:** from a clean linked worktree whose HEAD is the gate-opening commit:

  `bash docs/superpowers/evidence/search-distillation-d2-convref-run/01_wrapper.sh`
