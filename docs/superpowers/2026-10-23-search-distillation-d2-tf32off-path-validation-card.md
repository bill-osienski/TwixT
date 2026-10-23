# Search distillation D2 — full-float32 setting: policy-head and full-gradient-path validation card

**Date:** 2026-10-23. **Status:** FROZEN before any run. **Gate:** `SEARCH_DISTILLATION_D2_TF32PATH_AUTHORIZED` (the
forty-first), opened for ONE approved invocation by a one-line commit and restored by the wrapper.

**Approved:** preparation and one bounded invocation, capped at 10 minutes, of the three sequential checks below.
This explicitly includes **21 in-memory diagnostic optimizer updates** (phase 3), with no candidate saved.
**Not approved:** held-out files, retries, checkpoint output, persistent configuration changes, package changes,
chunking, bug-report submission, pushing, training or any learning run.

## 1. Purpose and limits

Ledger entry 31 found `MLX_ENABLE_TF32=0` a **candidate workaround for the isolated linear operations only**. This
checks it through the policy head and the full model's gradient path, in one fresh process launched with the
setting.

- **If all three phases pass,** the setting can be considered for future work.
- **A pass does not:**
  - rehabilitate the failed Stage 3 candidate;
  - establish the cause of Stage 3's value drift, or of either GPU defect;
  - authorize another learning run.
- **CPU/GPU agreement is a backend comparison.** CPU MLX shares MLX's autodiff, so it is not an oracle. The NumPy
  check (phase 1) validates the policy head independently.

## 2. The process

- **Launch:** the wrapper runs `python -B -m scripts.GPU.alphazero.search_distillation_d2_tf32path run --checkpoint
  <calib020_0001>` with `MLX_ENABLE_TF32=0` in that process's environment only. Nothing is installed or persisted.
- **The entry refuses** unless:
  - `MLX_ENABLE_TF32` is exactly `"0"` when it starts;
  - no `mlx` module is imported at that moment;
  - then the gate is open, the card and every pinned file match, the checkpoint names an existing file by absolute
    path, entry 26's fixture directory exists, and no output exists.

  It records the variable's value at the start and at the end.
- **Same packages and application code:** MLX 0.30.3, NumPy 2.4.1, Python 3.14.9, the M5 Max.
- **Inputs, all verified before phase 1** (any mismatch is VOID):
  - entry 26's saved fixtures (§3), through its committed `fixtures.jsonl`;
  - the training corpus through entry 24's **training-only reader** (`search_distillation_d2_gradtrace.read_training`,
    identity digest pinned). It opens no held-out game file;
  - Stage 3's per-update record (pinned), read only to write Stage 3's values beside phase 3's;
  - the checkpoint's SHA-1, and MLX's Adam defaults.
- **Reused code, pinned by SHA-256:**
  - `search_distillation_d2_gradcheck.py`: the NumPy reference, its finite-difference check, the policy-head case
    runner, the comparison function and the prefix loop;
  - `search_distillation_d2_gradtrace.py`: the training-only reader and the Stage 3 record reader;
  - `search_distillation_d2.py`: the batch builder, the forward pass and the state digest.
- **Entry 26's evidence, pinned by SHA-256:** its `fixtures.jsonl`, `record.json`, `full.jsonl` and six
  `reference/<case>.json`. They are read for the integrity checks and the predictions only.

## 3. Phase 1 — policy head against the independent NumPy reference

- **Cases:** the six saved by entry 26 as `reference_<case>.npz` in
  `~/twixt-cache/search-distillation-d2/gradcheck-fixtures-20261023T034401Z/`:
  - the four synthetic cases: `repeated_indices`, `batch_members`, `padding`, `weighted_additivity`;
  - the real batch at S0 (`real_S0`: 64 records × 525 padded move slots, so 33,600 fc rows) and at entry 26's S21
    (`real_S21`: 64 × 527, so 33,728 rows), with the CPU chain's encoder output as input.
- **Integrity:**
  - each file's SHA-256 must equal the one in entry 26's committed `fixtures.jsonl` (that file pinned), at the
    recorded path;
  - a policy head is rebuilt from the saved float64 parameters and BatchNorm statistics. Read back, its parameters
    must equal the saved ones exactly, eps included.

  Either failure is VOID.
- **Computation:** entry 26's case runner, unchanged (`gradcheck.reference_case`):
  - NumPy float64 forward and hand-derived analytic gradient of the masked cross-entropy, with respect to every
    trainable policy-head tensor and the input features;
  - MLX's gradient of the same loss, three times per device, from fresh copies;
  - **additivity:** MLX's single-record gradients, summed, against its full gradient (every case with more than one
    non-zero weight);
  - the fixed directional central difference (ε = 1e-5, direction seed 0, ε halved up to three times to avoid a
    ReLU pattern change).
- **The reference must validate itself on the four synthetic cases.** There, the run is VOID if the directional check
  is skipped or the analytic derivative misses the central difference.
  - Entry 26 validated it on all four.
  - Its unit test also checks every coordinate against central differences on a small case.
- **On the two real batches, entry 26's directional check was skipped:** every step down to ε = 6.25e-7 crossed a ReLU
  boundary (ledger entry 26).
  - NumPy is deterministic, so it is expected to be skipped again. A skip there is recorded, not a VOID.
  - On those two cases the GPU is judged on the other three conditions below.
- **Phase 1 PASSES** only if, in **every case, on the GPU**:
  - every gradient tensor, the input-feature gradient and the loss are within the tolerance (§6) of NumPy;
  - repeats 2 and 3 equal repeat 1 bitwise, for the loss and every gradient;
  - the summed single-record gradients are within the tolerance of the full gradient;
  - wherever the directional check ran, MLX's directional derivative is within the finite-difference tolerance of the
    central difference.

## 4. Phase 2 — full model at the incumbent (only if phase 1 passes)

- **State:** the unchanged incumbent `calib020_0001` (digest `ae0e71fc…`).
- **Batches:** Stage 3's **update-0** and **update-21** batches, from its draw order (seed 0) over the 6,403 training
  records.
- **Targets:** the anchor's teacher values, computed once on the GPU in this process, as Stage 3 computed them. The
  same array goes to both devices. Its digest is recorded and compared with entry 26's (`817aa7bf…`), as an
  observation only.
- **Per batch, on each device (`cpu`, then `gpu`), three times:**
  - a fresh network is loaded and set from the state's arrays (every parameter and BatchNorm statistic), and its
    digest must equal the state's;
  - fresh batch arrays are built (boards, moves, mask, targets, teacher values), and their digest must equal the
    batch's;
  - the training loss of `search_distillation_d2.train_student` is evaluated (policy CE + 1.0 × anchor + 1e-4 × L2),
    with no optimizer update.
- **Recorded per evaluation, as it happens:**
  - the loss terms;
  - the logits and values;
  - **all 60 gradient tensors** of the trainable surface;
  - their norm and digests.
- **Additivity:** for the first evaluation on each device, the gradient over the batch's first half plus the
  gradient over its second half, against the gradient over all of it, all without the L2 term.
- **Comparison:** each evaluation against the CPU's first evaluation of the same batch.
  - Logits are compared at **legal moves only**. The padded slots hold the −1e9 mask constant, and must be bitwise
    equal.
  - **This is stricter than entry 26.** Entry 26 compared the whole logits array, so the −1e9 constant set the scale
    and that comparison could not fail.
- **Phase 2 PASSES** only if, for **both batches, on the GPU**:
  - in every evaluation, the three loss terms, the legal-move logits, the values and every one of the 60 gradient
    tensors are within the tolerance of the CPU's first evaluation, and the padded logits are equal;
  - evaluations 2 and 3 equal evaluation 1 bitwise, in the loss terms, logits, values and every gradient;
  - the additivity check is within the tolerance for every tensor.

## 5. Phase 3 — updated state (only if phases 1 and 2 pass)

- **The prefix:** a fresh load of the incumbent, then **updates 0–20** of Stage 3's loop on the GPU with the setting
  enabled: Adam (lr 1e-4, MLX's defaults otherwise), clip 1.0, batch 64, draw seed 0, the teacher values above.
  Nothing runs inside the loop.
  - Each update's loss terms and pre-clip gradient norm are written as it finishes, beside Stage 3's.
  - **This is a new state.** It is not required to reproduce Stage 3's trace, nor entry 26's S21 digest
    (`91816a99…`); both are recorded as observations.
  - The state after update 20 (**U21**) is held in memory only and never saved.
- **Then phase 2's checks, unchanged,** at U21 on the same two batches (update 0's and update 21's).
- **Phase 3 PASSES** under phase 2's rule, at U21.

## 6. Tolerances (entry 26's card §6, frozen, not relaxed)

- **Device or reference comparison of a tensor:** `max|a − b| ≤ 1e-3 × max|b| + 1e-6`, with b the reference (NumPy in
  phase 1, the CPU's first evaluation in phases 2 and 3). Bitwise equality is reported alongside.
- **Repeats on one device:** bitwise.
- **Finite differences:**
  - against the analytic derivative: `|fd − an| ≤ 1e-6 × (|an| + 1)`;
  - against MLX: `|fd − mlx| ≤ 1e-3 × (|fd| + 1e-6)`.
- **Stated before the run:**
  - This tolerance is looser than entry 31's (1e-4). Entry 26's GPU results failed even this one.
  - The policy CE's gradient for the `out` layer's bias is zero analytically, since the softmax over legal moves
    sums to one. Its comparison therefore rests on the 1e-6 absolute term.

## 7. Verdict

- **Each phase is judged on the GPU only.**
  - The CPU's own results (against NumPy, its repeats, its additivity) are reported alongside. They do not enter the
    verdict.
  - If the CPU fails any of them, the record says so. The report must then say that the comparison base failed.
- **PASS:** all three phases pass.
- **FAIL:** the first failing phase is named, with every failing comparison. The later phases do not run:
  - after a phase-1 failure, no network is loaded and no teacher value is computed. Only the checkpoint's SHA-1 has
    been read, before phase 1;
  - after a phase-2 failure, no optimizer update runs.
- **VOID:** an input, pin, identity, state, batch, plan or optimizer-default mismatch; a fixture failing its checks;
  the reference failing its own validation; a non-finite result; any exception; SIGTERM (the wrapper's ceiling).
- **Removing gross errors alone is not a pass:** every comparison must be within the tolerance and every repeat
  bitwise.

## 8. Evidence, written as it exists

- **In the repository**, under `docs/superpowers/evidence/search-distillation-d2-tf32path/`, all create-only:
  - `progress.jsonl`;
  - `reference/<case>.json`: phase 1, one file per case as it finishes;
  - `full.jsonl`: one line per full-model evaluation, flushed and fsynced;
  - `full/<state>_b<batch>.json`: each comparison block as it finishes;
  - `reconstruction.jsonl`: phase 3's prefix, one line per update;
  - `record.json`: written on every exit the script survives. It holds the setting at the start and end, each
    phase's result, the verdict and the comparisons with entry 26.
- **Compact failure evidence:**
  - phase 1: entry 26's case runner records, for every tensor, the largest absolute difference from NumPy and the
    scale;
  - phases 2 and 3: every comparison records the largest absolute difference and the scale. When a comparison is
    outside the tolerance, it also records the number of elements outside it, and the worst element's index and
    both values.
- **No arrays are saved:**
  - phase 1's inputs are already entry 26's fixtures;
  - no model state, gradient or checkpoint is written anywhere.

## 9. Predictions on record (stated before the run)

1. **CPU unaffected, phase 1:** in all six cases, the CPU's three losses and its largest absolute difference from
   NumPy for every gradient tensor equal entry 26's (`reference/<case>.json`), exactly. NumPy's losses equal entry
   26's too.
2. **CPU unaffected, phase 2:** at S0 on the update-0 batch, the CPU's logits digest and policy term equal entry 26's
   S0 CPU values (`full.jsonl`), exactly.
   - The anchor term and the gradients are not predicted: they depend on the teacher values, which come from the GPU.
3. **The teacher values' digest:** **None stated in advance.**
4. **Each phase's result, and the verdict:** **None stated in advance.**
5. **U21's digest and the prefix's trace:** **None stated in advance.**

## 10. What this cannot establish

- The cause of either GPU defect, or of Stage 3's value drift.
- That the failed candidate was sound, or that any earlier checkpoint's training was or was not affected.
- Anything for training beyond these two batches, these two states and 21 updates.
- Anything for another MLX version or machine.
- A cost: the timings are recorded, but this is not a benchmark.
- Anything about measured playing strength.

## 11. Ceiling, cleanup and gate

- **Ceiling:** 10 minutes in all. The wrapper sends TERM to the run's process group at 590 s and KILL 10 s later.
- **Gate:** the wrapper restores it on every exit and reads it back against the pre-open commit.
- **Cleanup:** surviving processes are reported. The linked worktree and its branch are removed after the evidence
  commit.
- **No retry, and no other setting, whatever the result.**
- **Command:** from a clean linked worktree whose HEAD is the gate-opening commit:

  `bash docs/superpowers/evidence/search-distillation-d2-tf32path-run/01_wrapper.sh`
