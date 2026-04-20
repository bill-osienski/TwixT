# TwixT GPU AutoTune (Python + MLX)

Python implementation of TwixT game engine and AI for GPU-accelerated auto-tuning.

- **Apple Silicon (M1/M2/M3+)** optimized
- **MLX** for GPU-accelerated numerical work (Metal backend)
- **Unified CLI** replacing the old 4-script workflow
- **Replayable self-play games** as first-class output

## Quick Start

```bash
# Play against the AI
python3 -m scripts.GPU.cli play

# Watch AI vs AI
python3 -m scripts.GPU.cli play --ai-vs-ai

# Run fuzz tests
python3 -m scripts.GPU.cli fuzz --games 100 --verbose

# Initialize tuning logs
python3 -m scripts.GPU.cli init
```

---

## CLI Commands

### `play` - Interactive Game Mode

Play TwixT interactively to test game logic.

```bash
python3 -m scripts.GPU.cli play [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--human {red,black}` | `black` | Your color |
| `--ai-vs-ai` | off | Watch AI play itself |
| `--depth N` | `2` | AI search depth (2-4 recommended) |
| `--board N` | `24` | Board size (global option, goes before `play`) |

**In-game controls:**
- Enter moves as `row,col` (e.g., `11,12`)
- `m` - Show legal moves
- `u` - Undo last move
- `q` - Quit

**Examples:**
```bash
# Play as black (default) against depth-2 AI
python3 -m scripts.GPU.cli play

# Play as red (you move first)
python3 -m scripts.GPU.cli play --human red

# Harder AI
python3 -m scripts.GPU.cli play --depth 3

# Watch AI vs AI on smaller board
python3 -m scripts.GPU.cli --board 12 play --ai-vs-ai

# Quick test on 8x8 board
python3 -m scripts.GPU.cli --board 8 play --ai-vs-ai --depth 2
```

---

### `fuzz` - Fuzz Testing

Run many random/greedy games to test game rules.

```bash
python3 -m scripts.GPU.cli fuzz [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--games N` | `1000` | Number of games to run |
| `--mode {random,greedy}` | `random` | Move selection policy |
| `--max_moves N` | `220` | Max moves before draw |
| `--stall_limit N` | `40` | Moves without progress before stall |
| `--verbose` | off | Show progress |

**Examples:**
```bash
# Quick test
python3 -m scripts.GPU.cli fuzz --games 100 --verbose

# Test with heuristic-based moves (finds more wins)
python3 -m scripts.GPU.cli fuzz --games 100 --mode greedy --verbose
```

---

### `init` - Initialize Logs

Create the logs directory for auto-tuning.

```bash
python3 -m scripts.GPU.cli init
```

---

### `status` - Show Tuning Status

Display current auto-tuning state.

```bash
python3 -m scripts.GPU.cli status
```

---

### `sweep` - Run Sweep Games

Run games with candidate configurations.

```bash
python3 -m scripts.GPU.cli sweep [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--depths` | `2,3` | Search depths (comma-separated) |
| `--games N` | `10` | Games per configuration |
| `--total N` | `24` | Total configurations to test |
| `--seed N` | random | Random seed |

---

### `rank` - Rank Candidates

Rank sweep results and queue top candidates for validation.

```bash
python3 -m scripts.GPU.cli rank [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--depths` | `2,3` | Search depths to consider |
| `--top N` | `8` | Number of top candidates |

---

### `validate` - Validate Candidates

Run validation games on queued candidates.

```bash
python3 -m scripts.GPU.cli validate [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--depths` | `2,3` | Search depths |
| `--games N` | `60` | Games per validation |
| `--pass_score` | `0.02` | Max bias to pass |
| `--streak_needed` | `2` | Consecutive passes needed |
| `--write_search PATH` | none | Write winning config to file |

---

### `tune` - Full Tuning Loop

Run complete sweep → rank → validate cycle.

```bash
python3 -m scripts.GPU.cli tune [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--cycles N` | `3` | Number of tuning cycles |

Combines all sweep/rank/validate options.

---

### `replay` - View Game Replay

Replay a saved game file.

```bash
python3 -m scripts.GPU.cli replay <game.json>
```

---

## Global Options

These go **before** the subcommand:

```bash
python3 -m scripts.GPU.cli [global options] <command> [command options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--root PATH` | auto | Project root directory |
| `--search PATH` | auto | Path to search.json config |
| `--board N` | `24` | Board size |

**Example:**
```bash
python3 -m scripts.GPU.cli --board 12 play --ai-vs-ai
```

---

## Game Rules

**TwixT** is a connection game on a 24×24 board:

- **Red** connects top edge (row 0) to bottom edge (row 23)
- **Black** connects left edge (col 0) to right edge (col 23)
- Players alternate placing pegs (red moves first)
- Bridges form automatically between your pegs a knight's move apart
- **Bridges cannot cross** (yours or opponent's)
- Corners are forbidden for all players
- Red cannot place on columns 0 or 23 (black's goal edges)
- Black cannot place on rows 0 or 23 (red's goal edges)

**Board legend:**
```
R = Red peg          B = Black peg
= = Red goal edge    | = Black goal edge
x = Corner           . = Empty
```

---

## Architecture

```
scripts/GPU/
├── cli.py              # Command-line interface
├── game/
│   ├── state.py        # GameState dataclass
│   ├── board.py        # Board validation, legal moves
│   ├── bridge.py       # Bridge creation, crossing detection
│   └── rules.py        # Apply move, win detection (BFS)
├── ai/
│   ├── heuristics.py   # Position/move evaluation (28+ features)
│   └── search.py       # Minimax with alpha-beta pruning
├── selfplay/
│   ├── engine.py       # TwixtSimulator for batch games
│   └── random_policy.py # Random/greedy/search policies
├── tuning/
│   ├── loop.py         # Sweep/rank/validate cycle
│   ├── state.py        # Tuning state persistence
│   └── hasher.py       # Config hashing
├── replay/
│   ├── format.py       # GameRecord, Move dataclasses
│   └── viewer.py       # ASCII replay viewer
└── config/
    └── search_config.py # Load/save search.json
```

---

## Performance

| Mode | Speed | Use Case |
|------|-------|----------|
| Random moves | ~20 games/sec | Rule fuzz testing |
| Greedy (heuristics) | ~1.2 games/sec | Quick bias estimation |
| Depth-2 search | ~120ms/move | Standard play |
| Depth-3 search | ~1.3s/move | Stronger play |

Benchmarks:
- JS latency: `node scripts/bench_js_latency.js`
- Python self-play: `python3 scripts/GPU/bench_selfplay.py`
- Training bias check: `python3 scripts/GPU/bench_bias.py`
- Determinism check: `python3 scripts/GPU/check_deterministic.py`

Self-play benchmark output includes:
- Win/draw counts
- End reasons (win/stall/max_moves/no_moves)
- Average moves per game
- Appends a JSON trend log to `logs/bench-selfplay.json`

**Optimizations applied:**
- Connected components caching with revision-based invalidation
- Opponent CC invariance (opponent components passed through, not recomputed per move)
- Batch move scoring with `return_children` (avoids double `apply_move` calls)
- Adjacency cache shared across players

---

## Tuning Artifacts

After running tuning commands:

```
scripts/GPU/logs/
├── sweep-results.jsonl       # Raw sweep game results
├── validation-results.jsonl  # Validation pass/fail records
├── pending-validation.json   # Queue of candidates to validate
├── tuning_state.json         # Persistent tuning state
└── games/<hash>/             # Game replays by config hash
    └── game-<uuid>.json
```

---

## GPU Acceleration (MLX)

**Implemented:**
- `ai/batch_eval.py` - Batch feature extraction + value model inference
- `ai/value_model.py` - Batched matrix multiplication (GPU crossover at ~5000 items)

**Future targets:**
- `tuning/ridge.py` - Ridge regression via `mx.linalg.solve`

Note: For typical batch sizes (~500 moves), CPU (NumPy) is faster due to GPU transfer overhead. GPU acceleration shines at larger batch sizes used in training.
