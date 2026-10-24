# Search distillation D2 — bounded learning trial under the corrected numerical configuration (successor card)

**Date:** 2026-10-23

**Status:** FROZEN. The draft (`25b5b3b8`) was amended with the user's review (§11) and frozen with the build.
- **Authorized:** this card, the build (two thin entries, two closed gates, two wrappers, focused tests), and the
  seed procedure through the post-registration proof for exactly 12 fresh game seeds.
- **Not authorized:** generation, training, evaluation games, promotion, submission and pushing. Every gate stays
  closed.

## 1. Question

Can the fixed D2 distillation recipe, unchanged, produce a candidate that passes Stage 3's held-out checks when every
process runs with `MLX_ENABLE_TF32=0`? If it can, play is tested next (§9).

**One change:** only the numerical configuration changes; the recipe is not tuned at the same time. The trial produces
at most one candidate. A STOP or VOID at any step closes the trial, with no retry, top-up, rerun or selection.

**Basis:**
- Entries 31–35 support this configuration only within their tested scope: the isolated operations, the policy head,
  the incumbent's local derivatives, and one updated state (U21) on two batches.
- This card does not assume GPU training is reliable in general.
- It does not explain the old candidate's value drift.

## 2. Numerical configuration (every process in this trial)

| Step | Configuration | Flattened policy-head rows |
|---|---|---|
| Held-out generation (§4) | `MLX_ENABLE_TF32=0` in that process only; GPU inference through `LocalGPUEvaluator`, eval batch 14 | ≤ 14 × 528 = 7,392 |
| Training and held-out check (§5) | `MLX_ENABLE_TF32=0` in that process only; GPU | ≤ 64 × 528 = 33,792 (Stage 3's updates 0–21 ran 33,600–33,792) |
| Strength evaluation (§9, outline) | `MLX_ENABLE_TF32=0` in every evaluating process, workers included, for **both** arms | ≤ 528 (batch-1 play path) |

**About the row counts:**
- They are the rows of the policy head's flattened (position × move) matmuls. They do not bound every matrix
  operation in the network.
- Entries 31–35 checked particular sizes and states, not every batch.

**The setting:**
- **Enforced per process.** Each entry refuses unless it reads exactly `"0"` before MLX is imported, and records the
  value at start and at end, as the prefix entry does.
- **Nothing persists:** no shell profile, launch agent or package changes.

**Versions:** MLX 0.30.3, NumPy 2.4.1 and Python 3.14.9 on this M5 Max. Each entry checks the three versions; any other
version is a VOID.

**Provenance difference, stated, not corrected:**
- The 108 training games' labels (Stage 2, corpus `a9a425f2…`) came from processes whose `MLX_ENABLE_TF32` value was
  **not recorded**: the Stage 2 wrapper did not set it, and the record did not capture it.
- The value in those processes is therefore unknown. If it was unset, MLX's default (1) permits reduced-precision
  (TF32) matmuls.
- Their policy-head inference ran at ≤ 7,392 flattened rows, below the gross-defect range.
- The labels are reused unchanged and not regenerated. The fresh held-out labels (§4) use the corrected setting, so the
  training and held-out targets may differ in numerical provenance.
- Teacher values for the value anchor and the held-out baseline are recomputed in the training process under the
  corrected setting. Stage 3's values are not reused.

## 3. Inputs

- **Incumbent:** `calib020_0001`, unchanged (SHA-1 `209cf2d4…`, state digest `ae0e71fc…`). Loaded twice, as the
  teacher and the student, both in evaluation mode.
- **Training corpus:** the original 108 training games and 6,403 records. They are read through the gradient
  diagnostic's training-only reader, `search_distillation_d2_gradtrace.read_training`, as in entries 24–35:
  - each phase is verified by hash against `corpus.json`;
  - the training row identities must hash to `e90df4d3…`.
- **Original held-out games:** spent. They are never opened. The record reports **0 original held-out files opened and
  0 original held-out records read**, separately from any access to the new set. The corpus is not modified.
- **Old Stage 3 candidate:** not loaded and not used.

## 4. Fresh held-out set (12 games): `search_distillation_d2_heldout`

- **Games:** exactly 12 fresh self-play games of the incumbent at Stage 2's configuration, unchanged except for §2's
  setting:
  - 1,600 simulations, eval batch 14, `train()`'s other MCTS defaults;
  - root noise on, mirror off (`TWIXT_MIRROR_PROB=0`);
  - both goal-completion diagnostics off;
  - one serial process, network loaded once.
- **Slots and phase:** slot `s` (0–11) is game id `s`, with seed `plan.base + s`. All 12 are held out, in a single
  capture phase.
- **Purpose:** used only for §5's baseline and final check. They are never trained on, and never used for strength
  games or any other selection.
- **Storage:**
  - Games go in one capture directory, `~/twixt-cache/search-distillation-d2/heldout-tf32off-<UTC stamp>/phase_00/`,
    separate from the original corpus and create-only. It is finalized only when all 12 games are present and the
    model digest is unchanged.
  - The evidence directory `docs/superpowers/evidence/search-distillation-d2-heldout/` holds:
    - `record.json` and `progress.jsonl`;
    - the phase's three small files;
    - on completion, the manifest `heldout.json` (directory, games, hashes, record count, model digest, seeds,
      setting).
- **Checks:** after capture, every record the reader rebuilds must equal the learner row `play_game` returned
  (Stage 2's alignment check).
- **Eligibility:** all 12 games must complete. There are no top-ups, replacements or partial sets.
- **Stops (VOID; the trial closes):**
  - a capture failure or refusal;
  - a digest change, a module in training mode, or an alignment failure;
  - a seed that is not ACCOUNTED, available and schedulable;
  - mirror augmentation on;
  - a checkpoint or version mismatch;
  - any exception, SIGTERM, or the ceiling.
- **Ceiling: 2 hours in all.** TERM at 7,190 s, KILL 10 s later.
  - Basis: Stage 2 averaged 1.19 s per ply. At its mean length of 58.5 plies, 12 games take about 14 minutes; 12
    games at the 280-ply cap, at 1.43 s per ply, take about 80 minutes.
  - These are estimates, not guarantees. The setting's effect on inference speed is unmeasured.
  - A timeout is a VOID that closes the trial, with no extension or top-up.
- **Exits:** COMPLETED 0, VOID 3, REFUSED 4, USAGE 2.

## 5. Training and held-out check (one run): `search_distillation_d2_trial`

- **Recipe:** Stage 3's, frozen unchanged (Stage 3 card §3–4), through `search_distillation_d2.train_student`:
  - 401 updates at batch 64, draw-order seed 0;
  - Adam at learning rate 1e-4, with resolved defaults checked;
  - global-norm clip 1.0;
  - policy cross-entropy + 1.0 × value anchor + 1e-4 × L2;
  - update surface: the encoder and policy head;
  - evaluation-mode BatchNorm;
  - value head and running statistics never updated.
- **The new held-out set:**
  - **Admission:** it is admitted only if its generation record is COMPLETED under this card's hash, with the setting
    `"0"` at start and end and the teacher's digest before and after, and its `heldout.json` is the bytes that record
    names. The public entry refuses until both files are committed unchanged.
  - **Verification:** its phase is reopened through the strict capture reader, checked against the manifest's hashes,
    games and record count.
  - **Scoring:** it is scored **only twice**: the incumbent baseline (teacher values and metrics) before training, and
    the final check on the reloaded candidate. The record lists the scorings, which must be exactly `baseline`, then
    `final`. There is no intermediate evaluation, early stopping or retuning.
- **Verdict (Stage 3 card §7, unchanged):**
  - **PASS** requires all of:
    - every training check passes (surface changed, off-surface tensors bitwise unchanged, teacher unchanged);
    - every metric is finite;
    - held-out policy cross-entropy is strictly below the teacher's;
    - held-out mean |v_candidate − v_teacher| ≤ 0.05, and its 95th percentile ≤ 0.15.
  - Otherwise **STOP**, naming every failed criterion. The maximum drift and the KL divergence are reported, not gated.
- **Candidate:** exactly one.
  - Saved create-only to `~/twixt-cache/search-distillation-d2/student-tf32off-<UTC stamp>.safetensors`, reloaded
    strictly, with its digest verified.
  - Kept even on STOP, as a research artifact that is never used.
- **Recorded, not gated:**
  - every update's loss terms and pre-clip gradient norm (as Stage 3), with the count above 10,000. They are not
    compared to Stage 3's: the path differs (entry 35);
  - training-set metrics for the teacher and the candidate (diagnostics only).
- **VOID (the trial closes):**
  - a corpus, split, identity, setting, version, held-out-set, optimizer-default or reload mismatch;
  - an original held-out game file opened;
  - any exception, SIGTERM, or the ceiling.
- **Ceiling: 30 minutes in all.** TERM at 1,790 s, KILL 10 s later.
  - Stage 3 took 34 s. That is an estimate, not a guarantee.
  - A timeout is a VOID that closes the trial, with no extension.
- **Exits:** PASS 0, STOP 6, VOID 3, REFUSED 4, USAGE 2.
- **What a PASS shows:** fit and bounded value drift on 12 games under this configuration. It does not show stronger
  play.

## 6. Seeds

- **The block:** one fresh 12-seed block, chosen by the established rule (`seed_selection.candidate`), through the
  survey script `docs/superpowers/evidence/search-distillation-d2-heldout-seed-survey/01_seed_survey.py`. It is a copy
  of Stage 2's, with the tactical line's survey and collision-proof method imported by pinned hash.
- **The steps, each run once on a clean committed tree:**
  1. survey;
  2. pre-registration collision proof;
  3. registration of the block, appended last to ACCOUNTED, with
     `docs/superpowers/evidence/search-distillation-d2-heldout-run/plan.json` (base, 12 slots, survey hash);
  4. post-registration collision proof.
- **Afterwards:** after any generation run, K1 and K2 closing expose the started prefix and retire the whole block.
- A survey that is not CLEAN is a STOP with no adjustment.

## 7. Gates, commands, order

- **Gates (created closed):**
  - `SEARCH_DISTILLATION_D2_HELDOUT_AUTHORIZED` (45th) in
    `scripts/GPU/alphazero/search_distillation_d2_heldout_authorization.py`;
  - `SEARCH_DISTILLATION_D2_TRIAL_AUTHORIZED` (46th) in
    `scripts/GPU/alphazero/search_distillation_d2_trial_authorization.py`.

  Each is opened by a one-line commit in a clean linked worktree. Its wrapper restores it on every exit and reads it
  back against the pre-open commit.
- **Commands:**
  - `bash docs/superpowers/evidence/search-distillation-d2-heldout-run/01_wrapper.sh` runs
    `env MLX_ENABLE_TF32=0 PYTHONPATH=. TWIXT_MIRROR_PROB=0 <main>/.venv/bin/python -B -m scripts.GPU.alphazero.search_distillation_d2_heldout run --checkpoint <main>/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`.
  - `bash docs/superpowers/evidence/search-distillation-d2-trial-run/01_wrapper.sh` runs
    `env MLX_ENABLE_TF32=0 PYTHONPATH=. <main>/.venv/bin/python -B -m scripts.GPU.alphazero.search_distillation_d2_trial run --checkpoint <same>`.
- **Order, each step with its own approval:**
  1. preparation (this card, build, seeds through step 4);
  2. generation;
  3. seed closing;
  4. training;
  5. screen;
  6. strength comparison.

  The machine must be idle for each run.

## 8. Reuse

- **Generation:** `search_distillation_d2_measure`'s seams, loader, evaluator and MCTS configuration, and the serial
  capture driver and reader, as Stage 2 used them. The entry is a copy of Stage 2's generation loop for one 12-game
  phase.
- **Training:** `search_distillation_d2_train`'s recipe, criteria and save/reload path, and gradtrace's training-only
  reader. The entry is a copy of Stage 3's loop, with the readers replaced and the setting guard added.
- **Pins:** the reused modules are pinned by hash in each entry.
- **Unchanged:** `trainer.py`, `network.py`, `mcts.py` and the evaluator.

## 9. After a PASS: fresh strength evaluation (outline; each step needs its own card and approval)

Lower loss is not enough: a PASS only qualifies the candidate for these games.
- **Configuration:**
  - both arms in evaluation mode at 400 simulations, with `MLX_ENABLE_TF32=0` in every process;
  - the established evaluation-runner protocol, colour-reversed pairs, fresh seeds;
  - the 12 held-out games never used.
- **Baselines:** no strength number measured under the old setting is reused as a baseline (cf. the inference-mode
  rule).
- **Screen (gross regression only):**
  - 32 pairs; STOP if the mean pair score is below 0.40;
  - the screen claims nothing about strength, and its pairs are excluded from the comparison;
  - proposed ceiling 3 hours.
- **Strength comparison (D2 plan §6):**
  - 509 fresh pairs, fixed in advance, with no early look or extension;
  - pass only if a Hoeffding 95% lower bound on the mean pair score exceeds 0.5. **The exact formula and threshold are
    fixed in that card.** The plan's 0.56020 is the rounded outline figure only;
  - proposed ceiling 24 hours, re-derived from the screen's measured rate before that card is frozen.
- **Promotion** is a separate decision.

## 10. Prediction on record

**None stated in advance** (the user's decision). The draft's proposed figures are withdrawn and are not a prediction on
record.

## 11. The review's decisions (the user, 2026-10-23)

1. **Held-out labels:** generated with `MLX_ENABLE_TF32=0`. The training labels' provenance difference stays explicit
   (§2).
2. **Ceilings:** 2 hours for generation and 30 minutes for training, approved. The projections are estimates, not
   guarantees, and no extension or top-up follows a timeout.
3. **Outcome prediction:** None stated in advance.
4. **Wording fixes:**
   - zero original held-out files and records read is recorded separately from access to the new set, and the new set
     is scored only for the baseline and the final check (§3, §5);
   - the old processes' setting is unknown, not asserted (§2);
   - the row counts are flattened policy-head rows (§2);
   - the strength threshold's exact formula belongs to its future card (§9).
5. **The recipe, held-out thresholds and original corpus stay fixed.** This is one bounded trial of the corrected
   numerical configuration, with no simultaneous tuning.
