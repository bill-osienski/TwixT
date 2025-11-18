# TwixT Game AI

This workspace hosts the TwixT board game implementation and its search heuristics. The AI defaults to medium-depth play with a blend of handcrafted scoring and a lightweight value model.

## Quick Start

**Requirements:**

- Node.js (v14 or later)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Supported platforms: Windows, macOS, Linux

**Technical Stack:**

- **Frontend:** Three.js r128 (3D graphics & texture loading, MIT license), OrbitControls (camera manipulation), vanilla JavaScript (ES modules)
- **Assets:** Wood texture by ForKotLow (CC0 license, OpenGameArt)
- **Backend:** Node.js native HTTP server (no external web framework dependencies)
- **Dependencies:** commander ^11.1.0 (CLI parsing for training scripts)
- **Dev Tools:** ESLint ^9.39.1, Prettier ^3.6.2

**To play the game:**

```bash
npm install
npm start
```

The server will start on port 5500 and automatically open your browser to the game.

## AI Training Pipeline

The game includes a computer opponent for 1-player mode, powered by a hybrid AI combining traditional game-tree search with machine learning. The training pipeline optimizes this AI through automated self-play and parallel hyperparameter tuning.

**Training Requirements:**

- Python 3.7+ (uses standard library only - no external ML frameworks)
- Dev tools: black >=24.10.0, ruff >=0.6.7 (optional, for code quality)

**Architecture:**

The AI uses **minimax search** with **alpha-beta pruning** at configurable depths (typically 2-3 plies). Evaluation is based on:

- Handcrafted heuristics (bridge connectivity, lane control, edge threats)
- A lightweight **logistic regression value model** trained on self-play game states

**Training Flow:**

```
autoTune.py → tuneBaseline.js → selfPlayParallel.js
```

1. **autoTune.py** - Orchestrates hyperparameter optimization using:
   - **Cross-Entropy Method (CEM)** for sampling promising parameter distributions
   - **Hill-climbing** for local optimization around elite configurations
   - **Trend analysis** to identify which parameters correlate with stronger play
   - Parallel validation across multiple search depths

2. **tuneBaseline.js** - Executes parameter sweeps:
   - Tests 24 candidate configurations per sweep
   - Uses multi-process parallelization to run games concurrently
   - Tracks win rates, game lengths, and heuristic performance metrics

3. **selfPlayParallel.js** - Game engine:
   - Runs AI vs AI matches with different parameter sets
   - Supports parallel worker pools for efficient batch processing
   - Generates game state data for training the value model

The system automatically identifies winning configurations and persists them to `assets/js/ai/search.json`. The process runs thousands of games to statistically validate improvements before adopting new parameters.

**Quick Commands:**

```bash
# Run automated training (recommended)
python3 autoTune.py loop

# Train the logistic regression value model
python scripts/train_value.py --help

# Manual parameter sweep
node scripts/tuneBaseline.js

# Direct self-play batch
node scripts/selfPlayParallel.js
```

See `docs/baseline-tuning.md` for detailed documentation.

## Development

- **AI Parameters:** `assets/js/ai/search.json`
- **Testing:** `npm test` - Runs smoke tests
- **Linting:** `npm run lint` or `npm run lint:fix`
- **Type Checking:** `npm run typecheck`
- **CI:** Runs lint, typecheck, and tests on every push. Also checks Python code with `ruff` and `black`.

## Advanced Topics

**Sealed-Lane Performance Instrumentation**

Toggle performance profiling in `assets/js/ai/search.json`:

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

Access stats via `window.__TwixTSealedLaneStats` in the browser console. Use for comparing search performance before/after heuristic changes.
