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
- `python3 autoTune.py loop` → run the whole cycle continuously. The planner now mines the entire sweep + validation history, weighting validation parity (depth 3 counts double) when proposing new knobs. The loop prioritises hashes with existing streaks, queues up to five 60/60 runs per cycle (default 10 workers), finishes the in-flight sweep on Ctrl+C, automatically persists the first configuration that achieves two consecutive ≤ 3 splits to `assets/js/ai/search.json`, and exits when either a winner is found or the plateau backlog (score ≤ 2) is cleared. Add `--reset-stall` if you want to zero the stall counters and thaw previously frozen knobs at startup, and drop the exploit quota with `--exploit 0` if you only want trend-driven combos. Each sweep’s 24 combos are now split across
  - a soft-best “distributional” sampler (roughly a cross-entropy method over the top configs),
  - a niche hill-climb stage that perturbs several different elite styles while enforcing a minimum knob-distance between them, keeping each tweak within roughly ±5 % of the parent knob values (single-step exceptions for coarse knobs), and retrying each slot several times before yielding it back to the general planner,
  - classic best-value recombinations (capped at four per sweep so they don’t crowd out diversity),
  - trend-based nudges,
  - targeted exploration of under-sampled ranges,
  - and a mutate/random tail for diversity.
    The loop also keeps lifetime promotion stats per bucket (top‑10/top‑25 hit rates, wins, best rank), printed after every `update` **and** persisted to `logs/autoTune-state.json`, so you can see which samplers are contributing the most and adjust their weights later if needed.
    The soft-best pool itself is half “recent elites” (best hashes from the last few sweeps) and half “all-time champions,” with a stability gate (minimum sweep games or at least one long validation) so lucky single batches don’t dominate the distribution.
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
- GPU training utilities reside under `scripts/GPU_Training/`; they are ignored by default in Git. If you need to keep local experiments, remove the ignore rule or copy the files elsewhere.
- Launch the browser client locally with:
  ```
  node scripts/startServer.js
  ```
  then open `http://localhost:8080/TwixT.html`.
- CI runs `npm run lint`, `npm run typecheck`, and `npm test` on every push. Locally mirror that plus the Python hygiene checks (`ruff check autoTune.py`, `black --check autoTune.py`). Prettier ignores the large log JSONs (`logs/*.json`, `selfplay.json`) to avoid heap issues; format code/docs only.
- Train the logistic-regression value model from self-play traces:

```
  python scripts/train_value.py --help
```

## Standalone Game Bundle

If you want to ship the playable game (including the 1-player AI) without any of the automation scripts, keep only the browser assets and the tuned configuration:

- `TwixT.html`
- `assets/js/**` (all AI and game modules)
- `assets/modernWood1.jpg`
- `assets/value-model.json` (optional, but recommended once you have trained one; the AI falls back to heuristics if it is missing)
- `assets/js/ai/search.json` should already contain your desired knob settings before you copy the files.

Everything else (logs/, scripts/, docs/, autoTune.py, etc.) can be omitted. Because `TwixT.html` loads ES modules, host the bundle behind any static server (for example, from the bundle root run `npx serve .` or drop it into GitHub Pages/S3). Opening the HTML file directly from disk will hit CORS restrictions in most browsers.
