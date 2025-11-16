# Baseline Tuning Guide

This guide explains how to iterate on the handcrafted TwixT heuristics while keeping red and black balanced. The current workflow centers on `autoTune.py`, which coordinates sweeps, validation runs, and result logging.

## Current Baseline Snapshot

- The active heuristic configuration lives in `assets/js/ai/search.json`. The latest validated baseline (hash `1a9fd23f…`) uses:
  - `firstEdgeTouchRed = 420`, `firstEdgeTouchBlack = 455`
  - `finishPenaltyBase = 1181`
  - `redFinishPenaltyFactor = 0.60`, `blackFinishScaleMultiplier = 1.0`
  - `redSpanGainMultiplier = 1.0`, `blackSpanGainMultiplier = 1.0`
  - `redDoubleCoverageBonus = 1500`, `blackDoubleCoverageScale = 0.60`

Check the file directly before making changes—autoTune keeps hashes in sync with every update. The engine’s instrumentation still emits `[TwixTAI] heuristic stats …` lines during play; they are useful sanity checks after each batch.

## Auto-Tuning Workflow

Run all commands from the project root with Python 3 available on `PATH`.

1. **Generate a sweep plan**

   ```
   python3 autoTune.py suggest [--count N] [--exploit M] [--seed S]
   ```

   Writes `logs/next-sweep.json` and records the plan in `logs/autoTune-state.json`. The script blends best historical configs with neighborhood exploration. Customize the batch size or RNG seed as needed.
   The planner now mines the entire sweep + validation history. Validation runs carry extra weight (depth‑3 parity counts double, draws add a small penalty), so knob trends reflect long-run parity rather than short 10/10 noise. You can disable replay of the “best 12” by starting the loop with `--exploit 0` when you only want validation-weighted suggestions.

2. **Run the sweep**

   ```
   python3 autoTune.py sweep
   ```

   Delegates to `scripts/tuneBaseline.js`, which consumes `logs/next-sweep.json`, runs short self-play batches, and appends results to `logs/sweep-results.json`.

3. **Process new results**

   ```
   python3 autoTune.py update [--limit K]
   ```

   Scans for unprocessed sweeps, computes quick parity metrics, and writes the top `K` config hashes into `logs/pending-validation.json`. Add `--rebuild-telemetry` if you need to recompute the lifetime bucket stats from the entire sweep history (helpful after telemetry/schema changes); the command rebuilds the summary before examining new sweeps.

4. **Validate promising configs**

   ```
   python3 autoTune.py validate --hash <configHash> [--depth-config=2:60,3:60] [--workers=10] [--log=custom.log] [--persist]
   ```

   Temporarily installs the candidate in `assets/js/ai/search.json`, runs `scripts/runValidation.js`, appends the aggregate split (with heuristic counters) to `logs/validation-results.json`, then restores the previous config unless `--persist` is set. Omitting `--hash` uses the first pending recommendation.
   The automation filters out hashes that already failed a long run, prioritises candidates that have already passed once (streak ≥ 1), and schedules up to five validations per sweep (default 10 workers). Validation candidates are ranked by streak → depth‑3 parity → depth‑2 parity → weighted parity → sweep score, so depth‑3 balance takes precedence. If a hash completes two consecutive 60/60 runs with acceptable parity and draws, autoTune persists that configuration to `assets/js/ai/search.json` automatically and stops the loop.

5. **Review progress**

   ```
   python3 autoTune.py report
   ```

   Prints the best-scoring sweep entries, validation balances per config hash, and any outstanding validation queue.
   Once a config records two consecutive 60/60 validations with per-depth parity ≤ 3 wins (and ≤ 6 draws), autoTune marks it as meeting the balance goal and removes it from future validation queues.

Repeat the cycle: suggest → sweep → update → validate → report. For unattended runs, launch `python3 autoTune.py loop` with your preferred flags. Helpful options:

- `--exploit 0` to skip replaying historical hashes and rely entirely on the trend/validation weighting.
- `--reset-stall` to zero the plateau counters and thaw any knobs that previously froze themselves after a plateau.
- `--workers 10` (default) to keep 60/60 batches fast; plateau backlogs also use 10 workers automatically.

Each 24-combo sweep is split across a soft-best distribution sampler (builds a probability distribution over the top configs), a niche hill-climb stage that perturbs several elite “styles” while enforcing minimum knob distance and a ±5 % knob-span cap per tweak (coarse knobs get a single-step exception and each slot retries several times before giving up), classic best-value recombinations (hard-capped at four per sweep), validation-weighted trend moves, under-sampled exploration, and a mutate/random tail for diversity. The soft-best pool always combines recent elites (best hashes from the latest sweeps) with a handful of all-time champions, and it only admits configs that have logged enough sweep games or at least one long validation, so noisy one-off batches don’t distort the distribution. The loop also tracks lifetime promotion stats per bucket (top‑10/top‑25 hit rates, wins, best rank), prints them after every `update`, and persists them to `logs/autoTune-state.json`, giving you a quick feel for which samplers deserve more or fewer slots. Each bucket now records average depth‑2 parity, depth‑3 parity, and draw rates as well, so you can immediately spot which samplers are pushing depth‑3 toward balance. That mix keeps multiple TwixT playstyles alive while still following the strongest parity signals from the logs.

The loop now:

- Finishes the current cycle on Ctrl+C without writing partial sweep data.
- Validates up to eight high-scoring hashes per cycle, prioritising any hash that already has one successful 60/60 run.
- Automatically persists the first configuration that achieves the validation streak goal (two consecutive ≤ 3 splits) to `assets/js/ai/search.json`, prints its knob values, and exits.
- Detects plateaus after five cycles with no score/streak improvement (ties at score 0 reset the counter automatically), validates the remaining score ≤ 2 candidates using 10 workers, and then exits cleanly if no hash passes.

### Hash lifecycle & scheduling guardrails

Every configuration hash now carries a lifecycle status inside `logs/autoTune-state.json`:

| Status | Meaning | Scheduler behavior |
|--------|---------|--------------------|
| `UNTESTED` | Seen in sweeps but not yet validated. | Eligible for all buckets and can appear in validation queue once its sweep score ≤ 2. |
| `SHORTLIST` | High-performing hash awaiting validation. | Allowed up to 3 exploitation sweeps (`MAX_TEN10_SWEEPS_PER_SHORTLIST`) before it must be validated or dropped. |
| `VALIDATING` | Currently queued/running a 60/60. | Removed from sweeps until validation finishes. |
| `STABLE` | Passed the streak goal (two consecutive ≤ 3/≤ 3 runs). | Never re-run except via explicit anchor slots after a cool-off window. |
| `RETIRED` | Failed validation or exceeded the retry limit with poor scores. | Never scheduled again (still tracked for telemetry). |

`command_update` rebuilds the validation plan after each sweep, and every `validate` call updates the streak/registry immediately so streaked hashes jump to the front of the queue without waiting a full cycle:

1. All hashes with streak ≥ 1 are queued first.
2. Remaining hashes with sweep score ≤ 2 fill the rest of the `--limit` slots (default 5).
3. When both stall counters hit 5 cycles with no improvement, the loop drains *all* remaining score ≤ 2 hashes (plateau mode) before exiting.

Each knob is tracked independently; if a knob’s best sweep score fails to improve for five cycles it is frozen at its best value so future searches concentrate on the remaining degrees of freedom.

Bucket telemetry persists to `logs/autoTune-state.json`; if you ever need to regenerate the historical averages after a change, run `python3 autoTune.py update --rebuild-telemetry` once to backfill from every sweep log. The `update` command prints the per-bucket averages after each run so you can track which sampler family is paying off.

## Manual Fallback (Legacy)

If you need to run the Node sweep directly—for example when debugging new knobs—you can still invoke:

```
node scripts/tuneBaseline.js | tee sweep.log
```

Copy interesting combos into `assets/js/ai/search.json`, then confirm with:

```
node scripts/selfPlayParallel.js --depth-config "2:24,3:24" --workers 12 --verbose
```

For a longer 60/60 check that matches the automated flow, use:

```
node scripts/runValidation.js --depth-config=2:60,3:60 --workers=10 --log=validation-manual.log
```

`runValidation.js` snapshots the current heuristics, runs `selfPlayParallel`, parses the `[TwixTAI] heuristic stats …` output, and appends the win/draw splits to `logs/validation-results.json`.

After any manual experiments, re-run `python3 autoTune.py update` so the Python tooling picks up the new results.
