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

   Scans for unprocessed sweeps, computes quick parity metrics, and writes the top `K` config hashes into `logs/pending-validation.json`.

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
- `--reset-stall` to zero the plateau counters when restarting after a manual cleanup.
- `--workers 10` (default) to keep 60/60 batches fast; plateau backlogs also use 10 workers automatically.

The loop now:

- Finishes the current cycle on Ctrl+C without writing partial sweep data.
- Validates up to eight high-scoring hashes per cycle, prioritising any hash that already has one successful 60/60 run.
- Automatically persists the first configuration that achieves the validation streak goal (two consecutive ≤ 3 splits) to `assets/js/ai/search.json`, prints its knob values, and exits.
- Detects plateaus after five cycles with no score/streak improvement (ties at score 0 reset the counter automatically), validates the remaining score ≤ 2 candidates using 10 workers, and then exits cleanly if no hash passes.

Each knob is tracked independently; if a knob’s best sweep score fails to improve for five cycles it is frozen at its best value so future searches concentrate on the remaining degrees of freedom.

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
