# TwixT Game AI

This workspace hosts the TwixT board game implementation and its search heuristics. The AI defaults to medium-depth play with a blend of handcrafted scoring and a lightweight value model.

## Sealed-Lane Performance Instrumentation

The edge-finish heuristics now support an optional performance probe that measures how often and how long the sealed-lane search runs.

- Toggle the probe in `assets/js/ai/search.json` by setting:

  ```json
  {
    "debug": {
      "performance": {
        "sealedLane": true,
        "sealedLaneLogEvery": 0
      }
    }
  }
  ```

  `sealedLaneLogEvery` (optional) prints a summary to the console every _n_ invocations; leave it at `0` to suppress automatic logging.

- When enabled, runtime stats accumulate on `globalThis.__TwixTSealedLaneStats` (also available via `window` in the browser). The object exposes counters for total calls, open vs sealed results, time spent, bridge-cross checks, and traversal sizes. Call `__TwixTSealedLaneStats.reset()` to zero the counters between batches.

The instrumentation is disabled by default and adds only minimal overhead when turned on. Use it to compare depth-2/3 self-play timings before and after heuristic changes.

## Heuristic Auto-Tuning

Use `autoTune.py` to manage sweep batches and validation runs end-to-end:

- `python3 autoTune.py suggest` → write the next sweep plan to `logs/next-sweep.json`.
- `python3 autoTune.py sweep` → invoke `scripts/tuneBaseline.js` to process that plan.
- `python3 autoTune.py update` → fold the new sweep data into `logs/sweep-results.json` and queue validation targets.
- `python3 autoTune.py validate --hash <configHash>` → swap `assets/js/ai/search.json`, run `scripts/runValidation.js`, and append results to `logs/validation-results.json`.
- `python3 autoTune.py report` → summarize top sweep scores, validation balance, and pending work.
- `python3 autoTune.py loop` → run the whole cycle continuously. The loop now prioritises previously successful hashes, schedules up to eight validations per sweep (using eight workers), automatically persists the first configuration that achieves two consecutive ≤ 3 splits to `assets/js/ai/search.json`, and exits either on success or after plateau validations clear the remaining score ≤ 2 backlog. Ctrl+C cancels the current sweep without writing partial data.
- Individual knobs that fail to improve their best sweep score for five consecutive cycles are frozen automatically; the loop keeps the best value found and focuses exploration on the remaining knobs.

The legacy `node scripts/tuneBaseline.js` entry point still works for one-off manual sweeps, but the Python wrapper keeps history consistent and tracks config hashes automatically. More detail lives in `docs/baseline-tuning.md`.

## Development Notes

- Search parameters live in `assets/js/ai/search.json`.
- `autoTune.py suggest` analyzes your sweep history to find knobs with strong score trends, prioritizes under-tested ranges, and only falls back to random exploration when it runs out of data-informed candidates.
- The validation loop considers a config “balanced” once it records two consecutive 60/60 runs where each depth stays within ±3 wins (with ≤6 draws). The first such hash is written to `assets/js/ai/search.json` automatically and the loop exits; failed hashes are excluded from future queues.
- Validation batches can be run directly with:
  ```
  node scripts/runValidation.js --depth-config=2:60,3:60 --workers=10 --log=validation.log
  ```
  The script snapshots `search.json`, runs `selfPlayParallel`, parses the console heuristic stats, and appends a summary to `logs/validation-results.json`. Override depth plans, worker count, or log filename as needed.
- Self-play batches can be driven via `node scripts/selfPlayParallel.js`.
- GPU training utilities reside under `scripts/GPU_Training/` if you want to generate or evaluate value-model data.
- Launch the browser client locally with:
  ```
  node scripts/startServer.js
  ```
  then open `http://localhost:8080/TwixT.html`.
- Train the logistic-regression value model from self-play traces:
  ```
  python scripts/train_value.py --help
  ```
