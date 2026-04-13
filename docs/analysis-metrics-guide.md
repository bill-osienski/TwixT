# TwixT AlphaZero — Analysis Metrics Guide

Reference for all metrics produced by the training pipeline and replay analyzer. Each metric includes what it measures, how to interpret it, and which CLI knobs affect it.

---

## Output Files

| File | Produced By | Content |
|------|------------|---------|
| `iter_NNNN_game_NNN.json` | Trainer | Per-game replay with moves, metadata, opening diagnostics |
| `iter_NNNN_stats.json` | Trainer | Per-iteration aggregate stats (sidecar) |
| `summary.json` | Replay Analyzer | Full analysis summary (all metrics) |
| `replay_summary.csv` | Replay Analyzer | One row per game (winner, reason, opening geometry) |
| `opening_summary.csv` | Replay Analyzer | One row: all-ply opening diagnostics rollup per color |
| `opening_by_ply.csv` | Replay Analyzer | One row per (ply, color): per-ply opening diagnostics |
| `opening_per_game.csv` | Replay Analyzer | One row per (game, ply): raw per-root diagnostics |
| `report.txt` | Replay Analyzer | Human-readable summary of all metrics |

---

## Game Outcome Metrics

### results

| Field | Description |
|-------|-------------|
| `red_wins` | Games won by red |
| `black_wins` | Games won by black |
| `draws` | Games ending in draw (timeout, board_full, state_cap) |

**Purpose:** Track overall win balance. Persistent red dominance suggests a symmetry problem.

### draw_breakdown

| Field | Description |
|-------|-------------|
| `timeout` | Game hit max_moves limit |
| `board_full` | No legal moves remain |
| `state_cap` | Internal state repetition limit |
| `unknown` | Unclassified draw |

**Purpose:** Distinguish why games draw. High `timeout` means games aren't resolving — consider adjusting `max_moves` or enabling adjudication.

**Knobs:**
- `--max-moves` — maximum plies per game (per-size table: `MAX_MOVES_TABLE` in trainer.py)
- `--adjudicate-enabled` — convert timeout draws to decisive outcomes

### termination

| Field | Description |
|-------|-------------|
| `win` | Natural board win (connection completed) |
| `resign` | Losing player resigned |
| `adjudicated` | Timeout converted to decisive via root value evaluation |
| `timeout` | Draw by max_moves |

**Purpose:** Track how games end. A healthy late-training run has mostly resign + win, few timeouts.

### termination_by_winner

Same as `termination` but split by which player won (red/black/draw). Useful for detecting asymmetric resign or adjudication behavior.

### balance

| Field | Description |
|-------|-------------|
| `red_pct` | Red win percentage among decisive games |
| `black_pct` | Black win percentage among decisive games |
| `draw_pct` | Draw percentage among all games |
| `decisive_games` | Total non-draw games |
| `window` | Rolling red-dominant count / window size (e.g., "9/20") |

**Purpose:** Detect first-player advantage drift. `red_pct` persistently above 60% or `window` showing consistent red dominance signals a balance problem.

**Knobs:** No direct knob. Balance is an emergent property of the network + search. Curriculum progression, penalty tuning, and training duration all affect it indirectly.

### avg_plies

Average game length in plies (half-moves). Shorter games generally indicate stronger, more decisive play. Watch for sudden increases — may indicate the model has regressed or the board size increased via curriculum.

**Knobs:**
- `--max-moves` — caps game length
- Curriculum progression — larger boards naturally produce longer games

---

## Resign Metrics

### resign

| Field | Description |
|-------|-------------|
| `total` | Total games ending by resignation |
| `by_red` | Games where red resigned (red lost) |
| `by_black` | Games where black resigned (black lost) |

**Purpose:** Track resign frequency. Increasing resigns generally means the model is learning to evaluate positions accurately.

**Knobs:**
- `--resign-enabled` — enable/disable resign
- `--resign-threshold` — how negative the root value must be (default: -0.97)
- `--resign-min-ply` — earliest ply resign is allowed (default: 80)
- `--resign-window` / `--resign-k` — require K of last W evaluations to meet threshold

### resign_gate

Detailed resign gate diagnostics showing how many positions were checked, how many met the value threshold, and how many were blocked by the top-1 share requirement.

| Field | Description |
|-------|-------------|
| `checks` | Total resign evaluations (one per move after min_ply) |
| `value_hits` | Evaluations where root_value <= threshold |
| `blocked_by_top1` | Value hits blocked because top move's visit share was too low |
| `eligible_hits` | Value hits that passed all gates (actual resigns come from window/k logic) |
| `top1_share_on_value_hits` | Percentiles (p50/p90/p99) of top-1 visit share when value threshold was met |
| `min_top1_share` | The configured minimum top-1 share required |

**Purpose:** Diagnose resign gate behavior. If `value_hits` is high but `eligible_hits` is low, `blocked_by_top1` is filtering too aggressively — consider lowering `--resign-min-top1-share`.

**Knobs:**
- `--resign-min-visits` — minimum root visits before resign is considered (default: 200)
- `--resign-min-top1-share` — minimum top move visit share (default: 0.0 = disabled)

---

## Adjudication Metrics

### adjudication

| Field | Description |
|-------|-------------|
| `attempts` | Timeout games where adjudication was attempted |
| `adjudicated` | Games successfully adjudicated (converted to decisive) |
| `red_wins` / `black_wins` | Adjudicated outcomes by winner |
| `remaining_timeouts` | Timeout games that could not be adjudicated |

### adjudication.blocks

| Field | Description |
|-------|-------------|
| `ply` | Blocked: game didn't reach min_ply |
| `threshold` | Blocked: |root_value| below threshold |
| `visits` | Blocked: insufficient root visits |
| `top1` | Blocked: top move visit share too low |

### adjudication.stats

| Field | Description |
|-------|-------------|
| `abs_root_value.p50/p90` | Median and 90th percentile of |root_value| at adjudication attempts |
| `top1_share.p50/p10` | Median and 10th percentile of top-1 visit share |

**Purpose:** Tune adjudication gates. If `remaining_timeouts` is high, check which gate is blocking — lower that threshold.

**Knobs:**
- `--adjudicate-enabled` — enable/disable
- `--adjudicate-min-ply` — earliest ply for adjudication (default: 120)
- `--adjudicate-threshold` — minimum |root_value| (default: 0.90)
- `--adjudicate-min-visits` — minimum root visits (default: 200)
- `--adjudicate-min-top1-share` — minimum top-1 share (default: 0.0)

---

## Training Targets

### targets

| Field | Description |
|-------|-------------|
| `z_pos` | Positions where the outcome was +1 (to-move player won) |
| `z_zero` | Positions where the outcome was 0 (draw) |
| `z_neg` | Positions where the outcome was -1 (to-move player lost) |

**Purpose:** Monitor training signal quality. High `z_zero` relative to `z_pos + z_neg` means too many draws — the value head has weak signal. Healthy ratio is roughly balanced `z_pos` and `z_neg` with low `z_zero`.

**Knobs:** Affected by draw rate (see `draw_breakdown`). Enabling resign and adjudication reduces draws and improves the z-target mix.

---

## Compute Metrics

### compute

| Field | Description |
|-------|-------------|
| `buffer_size` | Current replay buffer size (positions) |
| `backups` | Total MCTS backups across all games |
| `leaf_evals` | Total neural network leaf evaluations |
| `nn_batches` | Total NN inference batches |

**Purpose:** Performance monitoring. `leaf_evals / nn_batches` = average batch size (should be close to `--mcts-eval-batch-size`). Low average batch size suggests tree is too narrow.

**Knobs:**
- `--mcts-eval-batch-size` — leaves per NN batch (default: 14)
- `--mcts-pending-virtual-visits` — virtual visits for pending leaves (default: 8)
- `--mcts-stall-flush-sims` — flush threshold when tree narrows (default: 16)
- `--buffer-size` — replay buffer capacity (default: 100000)

---

## Opening Geometry Metrics (Replay Analyzer)

These are computed by the replay analyzer from actual move positions in game files.

### opening.first_move_corner_rate

Fraction of games where the first move was in the exact corner cells.

### opening.first_move_entropy_nats

Shannon entropy of the first-move distribution per color. Higher = more diverse openings. Low entropy + high top-1 share = "stuck opening" risk.

### opening.first_move_top

Most frequent first move per color and its share of all first moves.

### opening.early_edge_corner

Edge and corner rates for early moves (within the first ply bucket). Includes `early_moves_considered` count.

### opening_geometry

Detailed per-ply rates for near-corner and edge-band move frequency, computed with configurable radius and band width.

**Analyzer knobs:**
- `--near-corner-radius` — Chebyshev radius for near-corner (default: 2)
- `--edge-band-width` — edge band width in cells (default: 2)
- `--opening-geom-kmax` — compute rates for k=1..K plies (default: 4)

### opening_drift

KL divergence between consecutive windows of first-move distributions. Low KL + low entropy = openings are stuck. High KL = healthy variation.

**Analyzer knob:**
- `--window` — window size for KL drift calculation (default: 50 games)

---

## Opening Penalty Diagnostics

These metrics diagnose **why** edge/corner play occurs by comparing what the neural network wants (raw priors), what penalties change (penalized priors), and what MCTS search ultimately chooses (visit distribution).

### How it works

At each ply within the diagnostic window, the system captures:
1. **Raw priors** (`raw_mass`) — neural network output before any adjustments
2. **Penalized priors** (`penalized_mass`) — after Dirichlet noise + edge/corner penalties are applied (post-root-adjustment)
3. **Visit distribution** (`visit_mass`) — what MCTS search chose after running all simulations

Each legal move is classified into an exclusive region: `near_corner` > `edge_band` > `interior`. Mass is summed per region and should total ~1.0.

### Diagnostic window

Controlled by: `max(edge_penalty_ply, corner_penalty_ply, 4) + 2`

Records all plies where penalties are active, plus 2 extra plies after penalties end (for rebound detection). Floor of 4 ensures diagnostics even when penalties are disabled.

### Per-game fields (in game JSON `opening_diagnostics`)

| Field | Description |
|-------|-------------|
| `ply` | Ply number (0-based) |
| `side_to_move` | "red" or "black" |
| `penalties_active.edge_band` | Whether edge-band penalty was active at this ply |
| `penalties_active.near_corner` | Whether near-corner penalty was active at this ply |
| `config` | Penalty config in effect (width, radius, penalty strengths) |
| `legal_moves_total` | Total legal moves at this position |
| `legal_move_counts` | Legal moves per region (near_corner, edge_band, interior) |
| `raw_mass` | Prior probability mass per region (from neural network) |
| `penalized_mass` | Prior mass per region after penalties + noise |
| `visit_mass` | Visit share per region after MCTS search |
| `raw_top1` | Top-1 move by raw prior: move coordinates, share, region |
| `penalized_top1` | Top-1 move by penalized prior |
| `visit_top1` | Top-1 move by visit count |

### Sidecar / analyzer aggregate fields

Aggregated across all games in an iteration, broken out by ply and color.

| Field | Description |
|-------|-------------|
| `mean_raw_mass` | Average raw prior mass per region |
| `mean_penalized_mass` | Average penalized prior mass per region |
| `mean_visit_mass` | Average visit mass per region |
| `mean_penalty_shift` | `penalized - raw` per region (negative = penalty moved mass away) |
| `raw_top1_region_pct` | Fraction of games where raw top-1 was in each region |
| `penalized_top1_region_pct` | Fraction of games where penalized top-1 was in each region |
| `visit_top1_region_pct` | Fraction of games where final top-1 visit move was in each region |
| `mean_legal_counts` | Average legal moves per region at that ply |
| `rebound_vs_last_active` | For post-penalty plies: visit_mass increase vs last penalty-active ply |

### all_diagnostic_plies

Weighted average across all diagnostic plies per color. Quick summary for iteration-to-iteration comparison.

### Diagnostic interpretation

| Observation | Diagnosis | Action |
|------------|-----------|--------|
| `raw_mass` heavily edge/corner | Model itself prefers edge/corner | Strengthen penalties |
| `raw_mass` OK, `penalized_mass` still edge/corner | Penalties too weak or too narrow | Raise `--root-edge-band-penalty` / `--root-near-corner-penalty`, widen band/radius |
| `penalized_mass` OK, `visit_mass` collapses back to edge/corner | Search/value interaction undoes penalties | Increase penalties further (not surviving search) |
| Active plies OK, post-penalty plies rebound (positive `rebound_vs_last_active`) | Penalty window too short | Increase `--root-edge-band-penalty-ply` / `--root-near-corner-penalty-ply` |
| Only certain plies regress | Ply-specific issue | Tune penalty window start/end |
| `mean_penalty_shift` near zero | Penalties are not moving mass | Check that penalties are enabled and strong enough |

### Coverage metadata

| Field | Description |
|-------|-------------|
| `source` | "sidecar" (from trainer stats) or "replay_fallback" (computed from game files) |
| `aggregation_impl` | "canonical" (shared code) or "analyzer_fallback" (local copy) |
| `coverage.games_with_diagnostics` | How many games had opening diagnostics |
| `coverage.coverage_pct` | Percentage of analyzed games with diagnostics |

**Knobs affecting opening diagnostics:**
- `--root-edge-band-penalty` — penalty strength lambda for edge-band moves (prior *= exp(-lambda))
- `--root-edge-band-penalty-ply` — apply edge-band penalty for plies < this value
- `--root-edge-band-width` — how many cells from each edge count as edge-band (default: 2)
- `--root-near-corner-penalty` — penalty strength lambda for near-corner moves
- `--root-near-corner-penalty-ply` — apply near-corner penalty for plies < this value
- `--root-near-corner-radius` — Chebyshev distance from corners (default: 2)

---

## Tuning Workflow

1. **Run training** with current settings
2. **Run replay analyzer** on the game files
3. **Check report.txt** for the quick summary
4. **Review opening_by_ply.csv** for per-ply mass shifts and rebound
5. **Identify the failure mode** using the diagnostic interpretation table
6. **Change one knob at a time** — do not adjust multiple penalty settings simultaneously
7. **Run another block** (200-500 games is enough for opening diagnostics)
8. **Compare** the new opening_summary.csv to the previous run

### What to check first

1. `balance` — is the game fair between red and black?
2. `avg_plies` — are games getting shorter (healthier) or longer?
3. `draw_breakdown.timeout` — are timeouts decreasing?
4. `opening_diagnostics_summary.all_diagnostic_plies` — is edge/corner mass decreasing?
5. `rebound_vs_last_active` — do openings snap back after penalties end?

---

## CLI Quick Reference

### Training knobs most commonly tuned

| Knob | What it does | Typical range | Current tuned value |
|------|-------------|---------------|---------------------|
| `--root-edge-band-penalty` | Edge penalty strength (lambda) | 0.3 - 2.0 | 0.75 |
| `--root-edge-band-penalty-ply` | Plies where edge penalty applies | 8 - 20 | 16 |
| `--root-edge-band-width` | Edge band width in cells | 1 - 3 | 2 |
| `--root-near-corner-penalty` | Corner penalty strength (lambda) | 0.1 - 1.5 | 0.30 |
| `--root-near-corner-penalty-ply` | Plies where corner penalty applies | 8 - 20 | 14 |
| `--root-near-corner-radius` | Corner region Chebyshev radius | 2 - 4 | 3 |
| `--resign-threshold` | How negative before resign | -0.99 to -0.90 | -0.945 |
| `--resign-min-ply` | Earliest resign ply | 40 - 120 | 80 |
| `--resign-k` | K of last W checks must meet threshold | 2 - 8 | 4 |
| `--resign-min-top1-share` | Min top-1 visit share to resign | 0.0 - 0.15 | 0.102 |
| `--adjudicate-threshold` | Min \|root_value\| for adjudication | 0.20 - 0.95 | 0.25 |
| `--adjudicate-min-ply` | Earliest adjudication ply | 120 - 380 | 340 |
| `--adjudicate-min-top1-share` | Min top-1 share to adjudicate | 0.0 - 0.20 | 0.13 |
| `--value-lr-scale` | Value head LR multiplier | 0.001 - 0.1 | 0.0025 |
| `--value-grad-max-norm` | Max gradient norm for value head | 0.05 - 0.5 | 0.05 |
| `--simulations` | MCTS sims per move (or per-size table) | 150 - 800 | 400 |
| `--max-moves` | Game length cap (per-size table) | Via `MAX_MOVES_TABLE` | 380 (size 24) |

### Analyzer knobs

| Knob | What it does | Default |
|------|-------------|---------|
| `--near-corner-radius` | Corner region for geometry analysis | 2 |
| `--edge-band-width` | Edge band for geometry analysis | 2 |
| `--opening-geom-kmax` | Plies for geometry rates | 4 |
| `--window` | Games per KL drift window | 50 |
| `--no-plots` | Skip heatmap PNG generation | off |
