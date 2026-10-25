# Search distillation D2 — bounded learning trial, attempt 2 (successor card)

**Date:** 2026-10-25

**Status:** FROZEN, with the build that pins it.
- **Authorized:** this card; minimal changes to the existing implementation for the new records and paths; fresh seed
  accounting through the post-registration proof for exactly 12 fresh game seeds; a read-only check for kernel panics.
- **Not authorized:** generation, training, evaluation games, promotion, submission, pushing, stress tests and system
  changes. Gates 45 and 46 stay closed.

## 1. Why a second attempt

Attempt 1 (ledger Post-closure entry 36) closed at its generation step. A macOS kernel panic rebooted the machine
during held-out game 1 of 12. It is closed as interrupted and incomplete, with no learning result: the reboot prevented
the learning test and gave no evidence for or against the recipe. The user decided on 2026-10-25 that exactly one fresh
attempt is justified.

This is **not** a retry, top-up or extension of attempt 1:
- attempt 1's evidence, its partial capture
  (`~/twixt-cache/search-distillation-d2/heldout-tf32off-20261025T133651Z/`) and its retired seed block stay unchanged;
- neither entry opens them;
- nothing from them is reused.

## 2. Unchanged: the original card is the authority

`docs/superpowers/2026-10-23-search-distillation-d2-corrected-trial-card.md`, sha256
`9f2c60967792bc8d970f00472b50bbc91f8c3ab599acb7c59aa417815dc10062`, governs everything this card does not change.
That includes:
- the question and the one-change rule (§1);
- the numerical configuration, process-local `MLX_ENABLE_TF32=0` for every process, and the provenance statement
  (§2);
- the inputs: unchanged `calib020_0001`, the 108 immutable training games, and zero original held-out files and records
  read (§3);
- the held-out games' configuration, checks, eligibility, stops and **2-hour ceiling** (§4);
- the training recipe (401 updates, batch 64, Adam 1e-4, clip 1.0, anchor 1.0, L2 1e-4, seed 0, encoder and policy
  head), the baseline and final scorings only, the acceptance criteria (held-out CE strictly below the teacher's; mean
  |Δv| ≤ 0.05; p95 ≤ 0.15), one candidate and the **30-minute ceiling** (§5);
- the seed method (§6), the gates, commands' environment and per-step approvals (§7), the reuse (§8) and the strength
  outline (§9).

A STOP, VOID or timeout at any step closes attempt 2, with no retry, top-up, rerun or selection.

## 3. Changed: records and paths only

| What | Attempt 1 (closed) | Attempt 2 |
|---|---|---|
| Card pinned by both entries | the original card | this card |
| Held-out evidence directory | `docs/superpowers/evidence/search-distillation-d2-heldout/` | `docs/superpowers/evidence/search-distillation-d2-heldout-attempt2/` |
| Held-out wrapper, plan and wrapper outputs | `.../search-distillation-d2-heldout-run/` | `.../search-distillation-d2-heldout-attempt2-run/` |
| Held-out capture directory | `~/twixt-cache/search-distillation-d2/heldout-tf32off-<stamp>/` | `~/twixt-cache/search-distillation-d2/heldout-tf32off-attempt2-<stamp>/` |
| Seed survey and proofs | `.../search-distillation-d2-heldout-seed-survey/` | `.../search-distillation-d2-heldout-attempt2-seed-survey/` |
| Seed-usage derivation for closing | `.../search-distillation-d2-heldout-seed-closing/` | `.../search-distillation-d2-heldout-attempt2-seed-closing/` |
| Seed block | retired (entry 36) | one fresh 12-seed block, chosen by the same rule |

- **Unchanged and reused:**
  - the record and manifest kinds;
  - gates 45 (`SEARCH_DISTILLATION_D2_HELDOUT_AUTHORIZED`) and 46 (`SEARCH_DISTILLATION_D2_TRIAL_AUTHORIZED`), each to
    be opened at most once for attempt 2;
  - the training entry's evidence directory (`.../search-distillation-d2-trial/`) and its wrapper
    (`.../search-distillation-d2-trial-run/01_wrapper.sh`), which attempt 1 never used;
  - the candidate's name pattern.
- **Admission:** the training entry admits the new set only from a COMPLETED generation record whose `card_sha256`
  is this card's. Attempt 1's records cannot be admitted.
- **The new wrapper** is a copy of attempt 1's, differing only in the paths and its header.
- **The new survey and closing derivation** are copies of attempt 1's, differing only in the paths, titles and the
  most recent EXPOSED block's label.

## 4. System stability

- **Before scheduling** generation and before scheduling training, run a read-only check for kernel panics and reboots
  since the last check: the files in `/Library/Logs/DiagnosticReports/` and `last reboot`. A new panic is reported to
  the user before anything is scheduled.
- **If another panic interrupts attempt 2:**
  - stop all further GPU experiments pending the user's system-stability decision;
  - there is no automatic third attempt;
  - the closing still exposes the started prefix and retires the whole block.

  A reboot leaves no wrapper record. In that case the closing follows the pattern of attempt 1's amended derivation
  (`.../search-distillation-d2-heldout-seed-closing/03_derive_usage_reboot.py`) and needs the user's approval.
- **No stress test, panic reproduction or system change** is authorized.

## 5. Order, each step with its own approval

1. Preparation: this card, the build changes, seeds S1–S4.
2. Generation.
3. Seed closing.
4. Training.
5. Screen.
6. Strength comparison.

The machine must be idle for each run.

## 6. Prediction on record

**None stated in advance.**

## 7. The user's decision (2026-10-25)

- One fresh attempt is justified. The reboot prevented the learning test; it supplied no evidence for or against the
  recipe.
- Preserve the closed trial and partial capture unchanged. Use 12 fresh seeds and a separate held-out set.
- Keep the training corpus, recipe, `MLX_ENABLE_TF32=0`, acceptance criteria and compute ceilings unchanged.
- Reuse the existing implementation with minimal changes for the new records and paths.
- If another panic interrupts the new attempt, stop further GPU experiments pending a system-stability decision, with
  no automatic third attempt.
- Do not turn this interruption into another learning-method investigation, or tune the recipe before it has had a
  completed test.
