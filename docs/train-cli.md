# AlphaZero Training CLI Reference

```
python -m scripts.GPU.alphazero.train [OPTIONS]
```

## Quick Start

```bash
# Minimal smoke test
.venv/bin/python -m scripts.GPU.alphazero.train \
  --iterations 2 --games-per-iter 4 --train-steps 1 \
  --simulations 50 --n-workers 1 --seed 42

# Full training run
.venv/bin/python -m scripts.GPU.alphazero.train \
  --iterations 100 --games-per-iter 25 --train-steps 100 \
  --simulations 800 --checkpoint-dir checkpoints/alphazero

# Resume from checkpoint
.venv/bin/python -m scripts.GPU.alphazero.train \
  --resume checkpoints/alphazero/model_iter_0050.safetensors
```

---

## Training Loop

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--iterations` | int | 100 | Total training iterations |
| `--games-per-iter` | int | 25 | Self-play games per iteration |
| `--train-steps` | int | auto | Gradient updates per iteration. `None` = auto from internal table per board size. `0` = skip training (self-play only) |
| `--batch-size` | int | 64 | Positions per training step |
| `--buffer-size` | int | 100000 | Replay buffer capacity (positions) |

## MCTS Core

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--simulations` | int | per-size table | MCTS simulations per move. If omitted, uses `SIMS_TABLE` (8:400, 10:400, 12:300, 16:200, 20:150, 24:400). CLI value overrides the table for all sizes |
| `--max-moves` | int | 200 | Base max moves per game. In practice, curriculum overrides this with `MAX_MOVES_TABLE` (8:90, 10:110, 12:160, 16:200, 20:250, 24:340) |
| `--mcts-eval-batch-size` | int | 14 | Leaves per NN batch. Reduced from 16 to prevent Metal GPU hangs |
| `--mcts-pending-virtual-visits` | int | 8 | Virtual visits added to pending leaves (prevents dogpiling) |
| `--mcts-stall-flush-sims` | int | 16 | Flush pending batch if no new leaf found in N sims. `0` = disabled |

## Exploration Tuning

These override the `MCTSConfig` defaults. Omitting a flag keeps the built-in default.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dirichlet-alpha` | float | 0.3 | Dirichlet noise concentration. Lower = more uniform noise, higher = more peaked |
| `--dirichlet-eps` | float | 0.25 | Noise mixing weight. `prior = (1-eps)*prior + eps*noise`. `0` = no noise |
| `--temp-high` | float | 1.0 | Temperature for early-game moves (proportional sampling) |
| `--temp-low` | float | 0.1 | Temperature for late-game moves (nearly greedy) |
| `--temp-threshold-ply` | int | 20 | Ply at which temperature drops from high to low |

**Validation**: `--dirichlet-alpha` must be > 0. `--dirichlet-eps` must be in [0, 1]. Both temps must be > 0 and `--temp-low` <= `--temp-high`. `--temp-threshold-ply` must be >= 0.

## Opening Exploration Boost

Applies stronger Dirichlet noise during the first N plies to increase opening diversity.
Disabled by default (`opening_noise_ply=0`).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--opening-noise-ply` | int | 0 (disabled) | Plies to boost. Ply 0..N-1 use boosted alpha/eps; ply >= N use standard |
| `--opening-dirichlet-alpha` | float | 1.0 | Dirichlet alpha during boosted plies (higher = flatter noise) |
| `--opening-dirichlet-eps` | float | 0.5 | Noise mixing weight during boosted plies (higher = more exploration) |

**Example**: `--opening-noise-ply 4 --opening-dirichlet-alpha 1.5 --opening-dirichlet-eps 0.6` boosts noise for plies 0-3.

**Validation**: `--opening-noise-ply` must be >= 0. `--opening-dirichlet-alpha` must be > 0. `--opening-dirichlet-eps` must be in [0, 1].

## Edge-Band Prior Penalty

Applies a multiplicative penalty to edge-band moves in the root prior for plies < N.
Disabled by default.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--root-edge-band-penalty` | float | None (MCTSConfig: 0) | Penalty strength λ. Prior is multiplied by exp(-λ) for edge-band moves |
| `--root-edge-band-penalty-ply` | int | None (MCTSConfig: 0) | Apply penalty for plies < this value |
| `--root-edge-band-width` | int | None (MCTSConfig: 2) | Edge-band width B. A cell (r,c) is in edge-band if r<B or r≥S-B or c<B or c≥S-B |

**Example**: `--root-edge-band-penalty 0.25 --root-edge-band-penalty-ply 4 --root-edge-band-width 2`

**Validation**: penalty ≥ 0, ply ≥ 0, 1 ≤ width < 12

## Near-Corner Prior Penalty

Applies a multiplicative penalty to near-corner moves in the root prior for plies < N.
Uses Chebyshev distance (max of row/col distance) to determine corner proximity.
Disabled by default.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--root-near-corner-penalty` | float | None (MCTSConfig: 0) | Penalty strength λ. Prior is multiplied by exp(-λ) for near-corner moves |
| `--root-near-corner-penalty-ply` | int | None (MCTSConfig: 0) | Apply penalty for plies < this value |
| `--root-near-corner-radius` | int | None (MCTSConfig: 2) | Chebyshev radius R. A cell is near-corner if max(|r-corner_r|, |c-corner_c|) <= R for any corner |

**Example**: `--root-near-corner-penalty 0.25 --root-near-corner-penalty-ply 6 --root-near-corner-radius 2`

**Validation**: penalty ≥ 0, ply ≥ 0, 1 ≤ radius < 12

**Note**: When both edge-band and near-corner penalties are active, overlapping cells receive the **max penalty** (not double-penalized).

## Resign

Allows self-play games to end early when the position is hopeless.
Uses a sliding-window approach: resign fires when K of the last W checks meet the resign condition.
Disabled by default (conservative).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--resign-enabled` | flag | off | Enable automatic resignation |
| `--resign-min-ply` | int | 80 | Don't resign before this ply |
| `--resign-threshold` | float | -0.97 | Resign when root value ≤ this |
| `--resign-window` | int | 12 | Sliding window size W |
| `--resign-k` | int | 8 | Resign if K of last W checks meet condition |
| `--resign-min-visits` | int | 200 | Require root visits ≥ this |
| `--resign-min-top1-share` | float | 0.0 | Require top move's visit share ≥ this (0 = disabled) |

**Example**: `--resign-enabled --resign-threshold -0.95 --resign-window 12 --resign-k 6`

**Validation**: min_ply ≥ 0, threshold ≤ 0, window ≥ 1, 1 ≤ k ≤ window, min_visits ≥ 1, top1_share ∈ [0, 1]

**Note**: Resign is from current player's perspective. Each ply after `min_ply`, if root_value ≤ threshold (and visit/share conditions pass), a "hit" is recorded in the sliding window. When K hits accumulate in any W-sized window, the current player resigns.

**Debug output**: When resign is enabled, each game prints:
```
RESIGN_DEBUG: checks=172 hits=0 maxW=4 min_root=-0.86
```
- `checks`: total plies where resign was evaluated (plies ≥ min_ply)
- `hits`: total plies where the resign condition was met
- `maxW`: maximum number of hits in any single W-sized window (use this to tune `--resign-k`)
- `min_root`: lowest root value observed after min_ply

**Training impact**: Resigned games produce decisive outcomes (winner ≠ draw), so positions get ±1 value labels instead of 0. This strengthens the value target distribution and reduces timeout draws.

## Mirror Augmentation

Horizontally mirrors positions (left-right flip) during self-play recording to double training data symmetry.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mirror-prob` | float | 0.5 | Per-position probability of appending a mirrored copy. `0` = off, `1` = always mirror |

The mirror correctly handles:
- Spatial column flip within the active square
- Directional link channel remapping (dc -> -dc for all 8 knight-move directions)
- `BLACK_LEFT_DIST` / `BLACK_RIGHT_DIST` channel swap
- Legal move coordinate mirroring

**Validation**: Must be in [0, 1].

## Network Architecture

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--hidden` | int | 128 | Residual block hidden channels |
| `--blocks` | int | 6 | Number of residual blocks |

## Optimizer

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--lr` | float | 1e-3 | Learning rate (encoder + policy head) |
| `--l2` | float | 1e-4 | L2 regularization weight |
| `--value-lr-scale` | float | 0.1 | Value head LR = `lr * value-lr-scale` |
| `--value-grad-max-norm` | float | 0.5 | Gradient clipping max norm for value head |

## Curriculum Learning

Training starts on small boards and promotes to larger sizes when win-rate criteria are met.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--curriculum-sizes` | str | `8,10,12,16,20,24` | Comma-separated board sizes to progress through |
| `--curriculum-window` | int | 200 | Rolling games window for promotion metrics |
| `--curriculum-draw-threshold` | float | 0.3 | Max draw rate to qualify for promotion |
| `--curriculum-min-wins` | int | 5 | Min wins per color (in window) for promotion |

## Checkpointing and Resume

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint-dir` | str | `checkpoints/alphazero` | Directory for model checkpoints |
| `--resume` | str | None | Resume from checkpoint (full state: weights + optimizer + curriculum + freeze state) |
| `--load-weights` | str | None | Load weights only (no training state). Mutually exclusive with `--resume` |

## Parallelism and Reproducibility

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--n-workers` | int | 1 | Parallel self-play workers. `1` = single-process (most stable) |
| `--seed` | int | None | Master RNG seed. Per-game seeds are derived deterministically |

## Diagnostics

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--opening-debug` | flag | off | Log opening move diagnostics (top-K moves, visit counts, priors, tie info). Output appears in self-play workers for the first 16 games, plies 0-1. Also enables `[OPENNOISE] Boosted` log lines when opening noise boost is active |
| `--no-save-games` | flag | off | Disable saving game replays to `scripts/GPU/logs/games/` |

## Environment Variables

These are set automatically by the CLI flags but can also be set manually for worker processes.

| Variable | Set by | Description |
|----------|--------|-------------|
| `TWIXT_OPENING_DEBUG` | `--opening-debug` | `1` = enable opening diagnostics in self-play and MCTS |
| `TWIXT_MIRROR_PROB` | `--mirror-prob` | Float 0-1, read at module import time by self-play workers |
| `TWIXT_WARN_MLX_IMPORT_ORDER` | (manual only) | `1` = warn if MLX imported before mcts.py (debugging aid) |

## Example Runs

```bash
# Smoke test with all new features
.venv/bin/python -m scripts.GPU.alphazero.train \
  --iterations 2 \
  --games-per-iter 4 \
  --train-steps 1 \
  --checkpoint-dir checkpoints/smoke-test \
  --simulations 50 \
  --n-workers 1 \
  --seed 42 \
  --opening-debug \
  --dirichlet-alpha 0.3 --dirichlet-eps 0.25 \
  --opening-noise-ply 4 \
  --opening-dirichlet-alpha 1.5 --opening-dirichlet-eps 0.6 \
  --mirror-prob 0.5

# Conservative exploration (less noise, greedy earlier)
.venv/bin/python -m scripts.GPU.alphazero.train \
  --dirichlet-alpha 0.15 --dirichlet-eps 0.15 \
  --temp-threshold-ply 10 --temp-low 0.05

# Aggressive opening diversity
.venv/bin/python -m scripts.GPU.alphazero.train \
  --opening-noise-ply 6 \
  --opening-dirichlet-alpha 2.0 --opening-dirichlet-eps 0.7 \
  --mirror-prob 1.0

# Disable all augmentation and noise (baseline comparison)
.venv/bin/python -m scripts.GPU.alphazero.train \
  --mirror-prob 0 --dirichlet-eps 0 --opening-noise-ply 0
```
