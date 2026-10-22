# Search distillation D2 — Stage 2 corpus generation card

**Date:** 2026-10-22

**Status:** FROZEN before execution.
- **Approved by the user** after accepting Stages 0–1: the build, focused tests, this card and its command, fresh seed accounting, one bounded generation run, evidence commits and seed closing.
- **Not approved:** Stage 3 (training), Stages 4–5, promotion and pushing. Every other gate stays closed.

## 1. Purpose

Produce the D2 feasibility corpus: the incumbent's own 1,600-simulation search targets, with full provenance, for a later and separately approved training stage. This stage makes no claim about target quality or strength.

## 2. What runs

- **Games:** 120 fresh self-play games of `calib020_0001` (SHA-1 `209cf2d4fd24a48553d259dd71b4954867b9473e`).
- **Configuration:** Stage 1's exact configuration (card `2026-10-22-search-distillation-d2-stage1-card.md` §2–3, unchanged):
  - the same loader, evaluation mode and evaluator (`LocalGPUEvaluator`, network inference on the GPU);
  - `MCTSConfig(n_simulations=1600)` with every other field at `train()`'s defaults;
  - root noise on, mirror augmentation off (`TWIXT_MIRROR_PROB=0`), and both goal-completion diagnostics off.
- **Execution:** one process, serial. The network is loaded once.
- **Phases:** 12 ordered phases of 10 games. Slot `s` (0–119) is game id `s` and uses seed `base + s`; its phase is `s // 10`. Each phase is one capture directory, finalized only when all 10 of its games are present and the model digest is unchanged.
- **After each phase:**
  - every record the Stage 0 reader rebuilds must equal the learner `PositionRecord` that `play_game` returned;
  - the phase's `context.json`, `receipts.jsonl` and `finalized.json`, which bind every game file by SHA-256, are copied into the evidence directory.

## 3. Split (fixed before generation)

- **Held out:** the 12 games with `slot % 10 == 9`, one per phase.
- **Training:** the other 108 games.

Held-out games are reserved for Stage 3's stop checks. They are never trained on and never used for strength games. The two Stage 1 games (seeds `[202691900, 202691902)`, EXPOSED and RETIRED) are excluded, and the fresh block cannot contain them.

## 4. Storage

- **The corpus:** written outside the repository, create-only, under `~/twixt-cache/search-distillation-d2/corpus-<UTC stamp>/phase_NN/`. At Stage 1's length the games total about 47 MB, and up to 8 MiB each is allowed.
- **The evidence directory** `docs/superpowers/evidence/search-distillation-d2-generate/` holds:
  - `record.json` and `progress.jsonl`;
  - each finalized phase's three small files;
  - on completion, `corpus.json`, which lists the phases, their hashes, and the training and held-out ids.

## 5. Eligibility, stops and ceiling

- **Eligible for training** only if all 12 phases finalize: status COMPLETED, with `corpus.json` written. There is no partial fallback.
- **Stops:** each of the following ends the stage at once as a VOID:
  - the first capture failure or refusal;
  - a model-digest change, a module in training mode, or an alignment failure;
  - a seed that is not ACCOUNTED and available;
  - any exception, or SIGTERM.

  There is no retry, replacement or top-up. Completed and partial phases stay on disk as diagnostics, and `record.json` is written in every case.
- **Ceiling:** 12 hours in all, including capture and validation. The wrapper sends TERM to the run's process group at 43,190 s and KILL 10 s later.

## 6. Seeds

A fresh 120-seed block follows the repository's procedure: survey, then pre-registration proof, then registration in ACCOUNTED with `plan.json`, then post-registration proof. After the run comes closing: K1 derives the usage, exposing the started prefix and retiring the whole block; K2 makes the registry edit.

## 7. Gate and command

- **Gate:** `SEARCH_DISTILLATION_D2_GENERATE_AUTHORIZED` in `scripts/GPU/alphazero/search_distillation_d2_generate_authorization.py`. A one-line commit opens it, and the wrapper restores it on every exit.
- **Command:** run from a clean linked worktree whose HEAD is that gate-opening commit:

  `bash docs/superpowers/evidence/search-distillation-d2-generate-run/01_wrapper.sh`

  The wrapper runs `env PYTHONPATH=. TWIXT_MIRROR_PROB=0 <main>/.venv/bin/python -B -m scripts.GPU.alphazero.search_distillation_d2_generate run --checkpoint <main>/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`.

## 8. Recorded

- **Per game:** phase, slot, seed, held-out flag, start player, plies, winner or draw reason, seconds, and seconds per ply.
- **Per phase:** bytes, write and reopen times, context and completion-marker hashes, records, and rows with inherited child visits.
- **Totals.**
- **Model:** digest before and after, and evaluation mode at load and at the end.
- **Run:** loaded-module hashes, environment and load average.

## 9. What the timing numbers are

Stage 1's projection of about 2.1 hours rests on two games. The 10.9 hours computed for 120 capped games is not a demonstrated bound, because the time per ply beyond ply 54 was never measured. The 12-hour ceiling is the only bound.
