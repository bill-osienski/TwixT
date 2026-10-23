# Search distillation D2 — full-float32 setting: encoder nonlinear-stage (branch) trace card

**Date:** 2026-10-23. **Status:** FROZEN before any run. **Gate:** `SEARCH_DISTILLATION_D2_BRANCHTRACE_AUTHORIZED` (the
forty-third), opened for ONE approved invocation by a one-line commit and restored by the wrapper.

**Approved:** preparation and one bounded invocation, capped at 10 minutes, of the trace below, at the unchanged
incumbent on Stage 3's update-0 batch, in one fresh process launched with `MLX_ENABLE_TF32=0`. **No optimizer update.**
**Not approved:** held-out files, parameter or package changes, any other environment switch, checkpoint output,
retries, recursive expansion during the run, training, bug-report submission, pushing.

## 1. Purpose and limits

Ledger entry 33 classified both of entry 32's failing encoder gradients as **INHERITED**. The upstream gradient
arriving at `encoder.blocks.{1,2}.conv2` differs between the CPU and GPU, and both blocks' largest difference is at
`[1, 13, 4, 42]`: record 1, row 13, column 4, channel 42. There block 2's CPU value is exactly −0.0.

**The hypothesis (the user's):** a ReLU takes a different branch on the two devices at that element. It is a
hypothesis, not an established backend defect.

This trace asks one question: is the difference a **numerical branch change**, or an **incorrect local derivative**
(or local forward arithmetic)? It does not try to force CPU/GPU equality, and it does not relax entry 32's criterion.
**Entry 32's FAIL at phase 2 stands, whatever the outcome.**

## 2. The process

- **Launch:** the wrapper runs `python -B -m scripts.GPU.alphazero.search_distillation_d2_branchtrace run --checkpoint
  <calib020_0001>` with `MLX_ENABLE_TF32=0` in that process's environment only. No other switch is set.
- **The entry refuses** unless:
  - `MLX_ENABLE_TF32` is exactly `"0"` when it starts, and no `mlx` module is imported at that moment;
  - then the gate is open, the card and every pinned file match, the checkpoint names an existing file by absolute
    path, and neither the evidence directory nor the fixture directory exists.
- **Same packages and application code:** MLX 0.30.3, NumPy 2.4.1, Python 3.14.9, the M5 Max.
- **Inputs, verified first** (any mismatch is VOID):
  - the training corpus, through entry 24's training-only reader;
  - the checkpoint's SHA-1, and the incumbent's state digest;
  - the update-0 draw, which must equal entry 32's.
- **Reused code, pinned by SHA-256:**
  - `search_distillation_d2_convref.py` (entry 33): the float64 weight-gradient and forward convolutions, the
    comparison, the sink and seams;
  - `search_distillation_d2_tf32path.py`, `search_distillation_d2_gradcheck.py`,
    `search_distillation_d2_gradtrace.py`, `search_distillation_d2.py`, as in entry 33.
- **Entry 33's evidence, pinned:** `record.json`, `full.jsonl`, `capture/cpu.json`, `capture/gpu.json`.

## 3. Step 1 — reproduce

- The teacher values are computed once on the GPU, as before. Their digest must equal entry 33's (`abe2873a…`).
- On each device (`cpu`, then `gpu`), one fresh network is set from the incumbent (digest checked), with fresh batch
  arrays (digest checked). The training loss and its gradient are evaluated **uninstrumented**, exactly as entries
  32 and 33 compute them.
- **Reproduced** only if, on each device:
  - the gradient digest and loss terms equal entry 33's `capture/<device>.json`;
  - the logits and values digests equal entry 33's `full.jsonl` (that device, evaluation 0).

  Bitwise equality implies entry 32's two failures. They are recomputed and recorded.
- **If not reproduced:** the outcome is `NOT_REPRODUCED`, and the run **stops**.

## 4. Step 2 — the instrumented trace

On each device, the same network and arrays. The encoder and its six residual blocks are instrumented by swapping
each instance's class, for the duration, to a subclass that performs **the same operations in the same order**:

- `conv → BN → ReLU` in the stem;
- `conv1 → BN1 → ReLU → conv2 → BN2 → + residual → ReLU` in each block.

It calls the same modules and MLX's compiled `nn.relu`. A **zero probe** is added at each traced point, and the
gradient is taken with respect to the parameters and all probes together. A probe's gradient is the loss's gradient at
that point.

- **Traced points, forward value and gradient:**
  - stem: `z0` (conv output), `a0` (BN output, the ReLU's input), `x0` (ReLU output);
  - each block `b` (0–5):
    - `z1` (conv1 output), `a1` (BN1 output = the inner ReLU's input), `h` (inner ReLU output);
    - `z2` (conv2 output), `s` (BN2 output + residual = the outer ReLU's input), `out` (outer ReLU output).
  - Also each block's input `x_in`, and the encoder's input (the canonicalized boards).
- **Fidelity:** the instrumented loss terms, logits, values and **all 60 gradient tensors** must equal the
  uninstrumented evaluation's **bitwise**. If not, the outcome is `TRACE_NOT_FAITHFUL`, and the run stops before
  step 3.
- The traced arrays are held in memory, converted to NumPy. Only the summaries and the fixtures (§8) are written.

## 5. Step 3 — analysis, in NumPy float64

**Self-check of the new reference first:** the float64 transposed convolution (a convolution's input gradient), by the
adjoint identity `⟨convᵀ(G, W), X⟩ = ⟨G, conv(X, W)⟩`. It is checked on block 2's CPU operands, at 1e-9 relative. If
it fails, the run is VOID at step `reference`.

**Per stage and device: local checks against float64 from that device's own captured operands.**

- **Forward:**
  - `z1` against `conv(x_in, W1) + b1`;
  - `a1` against `BN1(z1)`;
  - `z2` against `conv(h, W2) + b2`;
  - `s` against `BN2(z2) + x_in`;
  - the stem's `z0` and `a0` likewise.

  `BN` here is evaluation mode: `(z − mean) / √(var + eps) × weight + bias`, with the network's statistics.
- **Backward (local derivatives):**
  - **ReLU:** `g_a1 = g_h ⊙ [a1 > 0]`, `g_s = g_out ⊙ [s > 0]` and the stem's likewise, **bitwise**. MLX's ReLU
    derivative is 1 only where the input is > 0, on both devices (checked during design).
  - **BN:** `g_z = g_a × weight / √(var + eps)`, for BN1, BN2 and the stem's BN.
  - **The convolutions' input gradients and the residual add:**
    - `g_h = convᵀ(g_z2, W2)`;
    - each block's input gradient (the previous block's `g_out`, or the stem's `g_x0`) `= g_s + convᵀ(g_z1, W1)`.
  - The encoder output's gradient (block 5's `g_out`) comes from the heads, and is not checked.
- **Masks:** inner `[a1 > 0]`, outer `[s > 0]` and stem `[a0 > 0]`, per device. A **flip** is an element where the
  two devices' masks differ.
- **Every flip is quantified,** up to 64 per ReLU and largest injected gradient first; all are counted. For each:
  - the pre-activation on each device;
  - its float64 recomputation from that device's own operands, and whether its sign agrees with that device's branch;
  - the operands themselves (BN input, residual, the channel's BN constants);
  - the GPU−CPU difference of the pre-activation, against the stage's largest such difference;
  - the gradient arriving at the ReLU's output on each device, and the gradient leaving it.

  **A small pre-activation is reported as a number.** It does not establish harmless rounding.
- **The focus element `[1, 13, 4, 42]`:** every traced forward value, mask and gradient at it, on both devices, and
  every flip at that spatial position in any channel.
- **The backward walk (an observation):** the traced gradients in backward order (block 5 down to the stem, and within
  a block `out → s → z2 → h → a1 → z1`), each compared GPU against CPU. The first point outside the tolerance is the
  **entry point**; its outside elements are listed.
- **The accounting:** a float64, linear propagation of the flips' effect alone, from block 5 down to block 1. At each
  ReLU, using the GPU's masks:

  `Δpre = Δpost ⊙ m_gpu + g_post_cpu ⊙ (m_gpu − m_cpu)`

  with `Δ = 0` at the encoder's output, BN as its constant per-channel factor, and the transposed convolutions above.
  This is an exact identity per ReLU; only the flips act as sources.
  - For each target (`blocks.1.conv2.weight`, `blocks.2.conv2.weight`), the predicted GPU gradient is
    `dW(X_gpu, G_cpu + ΔG) + 2 × 1e-4 × W` (with `X = h` and `G = g_z2` of that block).
  - It is compared with the GPU's original gradient, **with** the flips' `ΔG` and **without** it (`ΔG = 0`).

## 6. Tolerances (frozen, not relaxed)

- **Every tensor comparison:** entry 26's `max|a − b| ≤ 1e-3 × max|b| + 1e-6`, with `b` the float64 value, the CPU,
  or the GPU's original gradient (in the accounting).
- **Instrumented against uninstrumented:** bitwise. The ReLU local derivative: bitwise.
- **The adjoint self-check:** 1e-9 relative.

## 7. Outcome

Taken in this order:

1. **`LOCAL_DERIVATIVE_ERROR`:** any backward local check (§5) fails on either device. The stage and device are named.
2. **`LOCAL_FORWARD_ERROR`:** otherwise, any forward local check fails on either device.
3. **`BRANCH_CHANGE`:** otherwise, if there is at least one flip in blocks 1–5, and for **both** targets the
   accounting's prediction is within the tolerance of the GPU's original gradient with the flips' `ΔG`, and outside it
   without. That is a numerical branch change that explains both discrepancies, with every traced local derivative and
   forward step correct on both devices. It does not say the branch change is harmless, or which device's branch is
   right.
4. **`NOT_EXPLAINED`:** anything else. Reported, and the run stops; no expansion.

`NOT_REPRODUCED` and `TRACE_NOT_FAITHFUL` stop earlier (status COMPLETED).

**VOID:** an input, pin, identity, state, batch or plan mismatch; the adjoint self-check failing; a non-finite result;
any exception; SIGTERM (the wrapper's ceiling).

**Entry 32's verdict is unchanged** by any outcome.

## 8. Evidence, written as it exists

- **In the repository**, under `docs/superpowers/evidence/search-distillation-d2-branchtrace/`, all create-only:
  - `progress.jsonl`;
  - `trace/<device>.json`: reproduction and fidelity;
  - `stages/<stage>.json`: each stage's local checks, masks, flips and comparisons, as it is analysed;
  - `walk.json`, `focus.json`, `accounting.json`, `fixtures.jsonl`;
  - `record.json`, on every exit the script survives.
- **Fixtures**, outside the repository, in a new directory
  `~/twixt-cache/search-distillation-d2/branchtrace-fixtures-<UTC time>/`, create-only, one compressed `.npz` per
  device, written as soon as that device's trace exists:
  - record 1's slice of every traced forward value and gradient, and of the encoder's input;
  - about 21 MB per device before compression.

  The operands of every recorded flip are in `stages/<stage>.json`. No weights, model state or checkpoint is saved.
- **Elapsed time** is recorded as an observation, not a benchmark.

## 9. Predictions on record (stated before the run)

1. **Reproduction:** bitwise on both devices, with the teacher values' digest equal to entry 33's.
2. **Fidelity:** the instrumentation is bitwise faithful on both devices.
3. **The branch (the user's hypothesis, made concrete):** block 2's outer ReLU flips at `[1, 13, 4, 42]`, with the
   CPU's mask 0 and the GPU's 1.
4. **The outcome:** **None stated in advance.**

## 10. What this cannot establish

- Which device's branch, or whole-model gradient, is the more accurate.
- Why the pre-activations differ upstream of an implicated branch (no recursive trace).
- That a branch change is harmless, or anything for other batches, states, training, the failed candidate or Stage
  3's value drift.
- A relaxed or re-judged entry 32.

## 11. Ceiling, cleanup and gate

- **Ceiling:** 10 minutes in all. The wrapper sends TERM at 590 s and KILL 10 s later.
- **The gate** is restored on every exit and read back. Surviving processes are reported. The worktree and branch are
  removed after the evidence commit.
- **No retry, whatever the result.**
- **Command:** `bash docs/superpowers/evidence/search-distillation-d2-branchtrace-run/01_wrapper.sh`, from a clean
  linked worktree whose HEAD is the gate-opening commit.
