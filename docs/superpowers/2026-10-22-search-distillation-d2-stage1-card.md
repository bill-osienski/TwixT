# Search distillation D2 — Stage 1 live measurement card

**Date:** 2026-10-22

**Status:** FROZEN before execution. The user approved Stage 0 (offline trainer and synthetic checks) and this one two-game measurement. Stages 2–5, promotion and pushing are NOT approved. The later-stage plan lives in `2026-10-22-search-distillation-d2-plan.md`; this card does not depend on it.

## 1. Purpose

Measure, once, what live corpus generation would cost and whether the capture path holds with the real network:
- seconds per ply at 1,600 simulations on this machine (Mac Studio M5 Max);
- the length of these two games;
- capture bytes, and write and reopen times;
- that the live model's digest, evaluation mode and every capture check hold through two real games;
- that the Stage 0 reader reopens the live phase.

**Measurement-only material.** The two games are never training, held-out or evaluation data for any later stage. Closing retires their seeds.

Two games cannot estimate the distribution of game lengths. The result prices Stage 2; it says nothing about target quality or strength.

## 2. What runs

Exactly two self-play games of `calib020_0001`:
- **Checkpoint:** `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`, SHA-1 `209cf2d4fd24a48553d259dd71b4954867b9473e`.
- **Network:** loaded by `probe_eval.load_network_for_scoring`, then `eval()`, with no module in training mode.
- **Evaluator:** `LocalGPUEvaluator(network)`, built as `train()` builds it (no compile).
- **Driver:** `training_capture_serial.run_serial_capture`. It runs serially in one process, with one Metal owner.

## 3. Frozen configuration (the settings intended for corpus generation)

| Setting | Value |
|---|---|
| Search | `MCTSConfig(n_simulations=1600)`. Every other field is the dataclass default, which equals `train()`'s defaults: `c_puct` 1.5, FPU 0, Dirichlet α 0.3 and ε 0.25, temperature 1.0 to ply 20 then 0.1, eval batch 14, pending virtual visits 8, stall flush 16, no prior penalties, no td1 forcing, no closeout tie-break |
| Play | `train()`'s serial call: 24×24, 280-ply cap, root noise on, no resignation, no adjudication, no position cap |
| Start player | `train()`'s draw: `random.Random(seed)`, then one `random()` |
| Mirror augmentation | Off: `TWIXT_MIRROR_PROB=0`, set by the wrapper and refused otherwise |
| Diagnostic switches | `goal_completion_emit_enabled` False and `goal_completion_record_enabled` False. They feed only closeout features that V1 refuses. Their effect on play is not asserted; the same setting applies to the corpus |
| Capture | Per-game ceiling 8 MiB; phase ceiling 20 MiB |

## 4. Seeds

A fresh two-seed block follows the repository's procedure. Each step runs once and is committed:
1. Survey.
2. Pre-registration collision proof.
3. Registration in ACCOUNTED, appended last, with `plan.json`.
4. Post-registration proof.

Game `g` uses seed `base + g`. After the run comes the closing: K1 seed-usage derivation, then K2, which marks the block EXPOSED and RETIRED as a whole.

## 5. Gate, command, ceiling

- **Gate:** `SEARCH_DISTILLATION_D2_MEASURE_AUTHORIZED` in `scripts/GPU/alphazero/search_distillation_d2_measure_authorization.py`. A one-line commit opens it. The wrapper restores it on every exit and reads it back.
- **Command:** run from a clean linked worktree whose HEAD is that gate-opening commit:

  `bash docs/superpowers/evidence/search-distillation-d2-measure-run/01_wrapper.sh`

  The wrapper runs `env PYTHONPATH=. TWIXT_MIRROR_PROB=0 <main>/.venv/bin/python -B -m scripts.GPU.alphazero.search_distillation_d2_measure run --checkpoint <main>/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`.
- **Ceiling:** 30 minutes in all. The wrapper sends TERM to the run's process group at 1,790 s and KILL 10 s later.

## 6. Stops

Each of these is a VOID: no retry, no extension, no second run.
- A capture refusal or error.
- A model-digest change, or a module in training mode.
- A seed that is not ACCOUNTED and available.
- Any exception, or SIGTERM (the ceiling).

An incomplete run is a VOID. Diagnostics are kept whatever happens: `progress.jsonl`, the partial capture, `record.json`, stdout and stderr, and `outer.json`.

## 7. Recorded

- **Per game:** slot, seed, start player, plies, winner or draw reason, wall seconds and seconds per ply.
- **Capture:** bytes, write time, reopen-and-validate time, and the number of records the Stage 0 reader rebuilds.
- **Target alignment, live:** every record the Stage 0 reader rebuilds must equal the learner `PositionRecord` that `play_game` returned for that ply (board tensor, mover, legal order, visit counts, outcome); any difference is a VOID at step `alignment`. The number of rows whose searches started with inherited child visits is recorded, so it is known whether the after-search versus before-search distinction was exercised; the fake-evaluator tests cannot exercise it.
- **Model and identity:**
  - the model digest before and after;
  - evaluation mode after load and after the last game;
  - the hashes of the loaded repository modules;
  - the environment;
  - the load average before and after.

## 8. Use of the result

The measured seconds per ply and plies per game price Stage 2. A projected corpus time above 12 hours stops Stage 2 as proposed; it does not prompt a rerun. The plan must be revised and approved again.
