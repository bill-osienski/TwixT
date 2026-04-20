# TwixT Heuristics Constants Reference

This document maps all constants used in the TwixT AI evaluation system.

## Architecture Overview

**JS has TWO separate scoring systems:**

```
JS:
├── heuristics.js (HARDCODED constants: 100, 20, 30, 10, 5, etc.)
│   ├── evaluatePosition() → board state evaluation (used at leaf nodes)
│   ├── evaluateMove() → basic move scoring
│   ├── scoreComponent() → component quality scoring
│   └── evaluateWinningThreats() → threat detection
│
└── search.js (TUNABLE knobs from search.json: 12, 35, 180, 3332, etc.)
    └── movePriority() → advanced move ordering for alpha-beta pruning
```

**Python mirrors this in heuristics.py:**

```
scripts/GPU/ai/heuristics.py:
├── HARDCODED CONSTANTS (lines 27-73)
│   └── CONNECTED_PATHS_MULTIPLIER=100, SPAN_MULT=20, FULL_SPAN_BONUS=500, etc.
│
├── TUNABLE KNOBS / DEFAULT_KNOBS (lines 77-127)
│   └── friendlyConnection=12, spanGainBase=180, finishBonusBase=3332, etc.
│
└── Functions:
    ├── evaluate_position() → uses HARDCODED (minimax leaf evaluation)
    ├── evaluate_move() → uses HARDCODED (basic move scoring)
    ├── score_component() → uses HARDCODED
    ├── evaluate_winning_threats() → uses HARDCODED
    │
    └── move_priority() → uses KNOBS (move ordering - ported from search.js)
        └── Called by score_moves() for alpha-beta move ordering
```

**How they work together in search:**
1. `score_moves()` calls `move_priority()` with KNOBS to order candidate moves
2. Alpha-beta search explores moves in this order
3. At leaf nodes, `evaluate_position()` uses HARDCODED constants
4. `choose_move()` combines minimax score + immediate score + position score

**Source files:**
- JS heuristics: `assets/js/ai/heuristics.js`
- JS search/movePriority: `assets/js/ai/search.js`
- JS knobs: `assets/js/ai/search.json`
- Python implementation: `scripts/GPU/ai/heuristics.py`

## Training Mode Knobs (Python)

Training mode uses a normalized knob set to avoid parity-driven bias and to
increase opening diversity. These knobs live in `scripts/GPU/ai/heuristics.py`
and are applied via `normalize_knobs_for_mode()`.

**Training-only knobs:**
- `opening_random_plies` (default 8): Early plies sampled from top-K heuristic moves.
- `opening_random_top_k` (default 0): 0 = uniform random over legal moves.
- `opening_random_plies_d2` (default 2): Random opening plies when depth >= 2.
- `opening_random_top_k_d2` (default 6): Sample from top-K heuristic moves.
- `opening_uniform_plies_d2` (default 0): Fully random opening plies when depth >= 2.
- `opening_uniform_margin_d2` (default 3): Avoid edge rows/cols for uniform openings.
- `opening_forbid_edge_plies_d2` (default 2): Forbid edge zones for early plies.
- `opening_forbid_edge_margin_d2` (default 3): Edge margin to forbid in opening.
- `opening_quadrant_plies_d2` (default 0): Bias opening toward a random quadrant.
- `opening_quadrant_margin_d2` (default 2): Margin for quadrant bias.
- `enforce_symmetry` (default 1): When enabled, neutralizes color-specific biases
  (red/black multipliers, bonuses, and scaling) to reduce structural advantage.
- `training_max_moves` (default 50): Cap training game length.
- `training_stall_limit` (default 15): Early stall detection for throughput.
- `training_max_moves_d2` (default 80): Depth-2 max moves.
- `training_stall_limit_d2` (default 30): Depth-2 stall limit.
- `training_span_win_margin` (default 1): Span margin to choose winner on stall/max-moves.
- `training_black_span_multiplier` (default 1.1): Training-only boost to black span gain.
- `training_goal_distance_mult` (default 1.1): Boost goal distance reward.
- `training_center_bias_mult` (default 0.8): Keep some center preference.
- `training_span_gain_mult` (default 1.3): Boost span growth reward.
- `training_connector_bonus_scale` (default 1.2): Boost bridge/connectivity reward.
- `training_finish_bonus_scale` (default 1.2): Boost finish pressure.
- `training_adjacent_penalty` (default 60.0): Penalize orthogonal clumping in training.
- `training_edge_push_weight` (default 25.0): Pull moves toward missing goal edges.
- `training_edge_push_max` (default 12.0): Max distance considered for edge push.
- `training_missing_edge_half_penalty` (default 80.0): Penalize moves on the wrong half before touching both edges.
- `training_second_edge_push_scale` (default 1.2): Extra push after first edge is touched.
- `training_global_span_gain_weight` (default 300.0): Reward overall span growth across all pegs.
- `training_first_edge_bonus_scale` (default 1.4): Scale first-edge touch bonus in training.
- `training_defense_scale` (default 1.3): Boost block/pressure terms in training.
- `training_edge_push_decay_after_both` (default 1): Disable edge-push after both edges are touched.
- `training_opening_center_plies` (default 8): Opening phase length for center bias.
- `training_opening_center_bias_mult` (default 2.0): Extra center pull during opening.
- `training_opening_goal_distance_scale` (default 0.5): Reduce goal-edge pull during opening.
- `training_opening_edge_push_scale` (default 0.2): Suppress edge push during opening.
- `training_opening_missing_half_scale` (default 0.2): Suppress wrong-half penalty during opening.
- `training_edge_push_ramp_plies` (default 6): Ramp edge-push after opening.
- `training_second_edge_bonus` (default 400.0): Bonus when a move touches the second edge.
- `training_edge_touch_requires_bridge` (default 1): Require bridge/connection for edge bonus.
- `training_goal_isolation_scale` (default 0.4): Reduce goal bonus on isolated moves.
- `training_edge_touch_min_component` (default 3): Require min component size for edge bonuses.
- `training_edge_progress_weight` (default 140.0): Reward progress toward missing edges.
- `training_edge_progress_min_component` (default 3): Use overall bounds until component is this size.
- `training_bridge_bonus` (default 80.0): Reward creating new bridges.
- `training_component_growth_bonus` (default 50.0): Reward growing largest component.
- `training_isolated_penalty` (default 120.0): Penalize isolated placements.
- `training_new_component_penalty` (default 150.0): Penalize creating new components.
- `training_bridge_bonus_after_both_scale` (default 0.3): Downscale bridge bonus after both edges.
- `training_growth_bonus_after_both_scale` (default 0.3): Downscale growth bonus after both edges.
- `training_redundant_bridge_penalty` (default 120.0): Penalize bridges without progress.
- `training_ladder_bonus` (default 180.0): Reward straight chain extension.
- `training_ladder_max_dev` (default 2.0): Max lateral deviation for ladder bonus.
- `training_ladder_requires_progress` (default 1): Require span/gap progress for ladder bonus.
- `training_sealed_lane_penalty` (default 1600.0): Penalize chasing a sealed lane.
- `training_midgame_start_ply` (default 8): Start adaptive sampling after opening.
- `training_midgame_top_k` (default 1): Sample from top-K after opening.
- `training_sealed_lane_top_k` (default 1): Tighten sampling when lane is sealed.
- `training_midgame_temp_scale` (default 0.0): Reduce temperature after opening.
- `training_score_delta_frac` (default 0.1): Keep moves within top-score band.
- `training_score_delta_abs` (default 300.0): Absolute score band for sampling.
- `training_score_delta_cap` (default 100.0): Hard cap on score spread for sampling.
- `training_guard_require_progress` (default 1): Require chain progress in sampling pool.
- `training_guard_allow_connector` (default 1): Allow connector targets as progress moves.
- `training_guard_start_ply` (default 6): Start guarded sampling after opening.
- `training_force_deterministic_open_lane` (default 1): Greedy once lane is open after opening.
- `training_guard_require_defense` (default 1): Require defensive value when opponent is urgent.
- `training_force_deterministic_urgent` (default 1): Greedy when opponent threat is urgent.
- `training_defense_span_threshold` (default 2): Opponent span to trigger defense guard.
- `training_defense_corridor_margin` (default 4): Corridor margin for intercept moves.
- `training_retreat_span_penalty` (default 200.0): Penalize span shrink after both edges.
- `training_retreat_gap_penalty` (default 150.0): Penalize gap regression after both edges.
- `training_retreat_requires_both` (default 1): Only penalize retreat after both edges.
- `training_temp_scale_d2` (default 0.35): Reduce depth-2 temperature to avoid random drift.
- `debug_sample_rate` (default 0.1): Fraction of debug games that record per-move scores.
- `debug_max_plies` (default 80): Cap per-move logging length in debug mode.
- `debug_trace` (default 0): When enabled, stores per-move trace in stats.

**Behavior notes:**
- Training mode ignores `deterministic_mode` so temperature sampling remains active.
- Debug mode keeps deterministic tie-breaks for correctness testing.

## Opening Book (Bridge Artifact)

The opening book is generated from Python replay games and consumed by the JS
engine for early plies. It is a lightweight bridge from Track A → Track B.

**Generator:**
- `scripts/GPU/replay/opening_book.py`

**Output file (default):**
- `assets/js/ai/opening-book.json`

**Perf check:**
- `node scripts/bench_opening_book.js`

**Key format:**
- `b:<size>|s:<start>|<p><row>,<col>|...`
- Example: `b:24|s:red|r11,12|b12,10`

**Lookup behavior:**
- JS checks the book before search in `assets/js/ai/search.js`.
- If an entry exists for the current position, the top move is used.

## Training Diagnostics (Self-Play)

Per-game diagnostics are recorded in replay metadata and aggregate logs:
- `stagnation_max`: max consecutive no-progress plies.
- `progress_events`: number of times edge progress increased.
- `opening_random_moves`: count of random opening moves.
- `avg_search_score`, `min_search_score`, `max_search_score` (when search used).
- Progress tracking uses `component_metrics` span + edge touches, matching JS
  deterministic game oracle semantics.

## evaluatePosition() Constants

| JS Line | JS Code | Value | Python Constant | Status |
|---------|---------|-------|-----------------|--------|
| 14 | `game.winner === player ? 10000 : -10000` | 10000 | hardcoded | OK |
| 21-22 | `evaluateConnectedPaths * 100` | 100 | `CONNECTED_PATHS_MULTIPLIER` | OK |
| 24-25 | `evaluatePotentialConnections * 20` | 20 | `POTENTIAL_CONNECTIONS_MULTIPLIER` | OK |
| 27-28 | `evaluateEdgeProgress * 30` | 30 | `EDGE_PROGRESS_MULTIPLIER` | OK |
| 32 | `(playerPegs - opponentPegs) * 2` | 2 | `PEG_DIFFERENCE_MULTIPLIER` | OK |
| 73/83 | `urgency = touchesTop/Bottom ? 2.5 : 1.0` | 2.5 | `URGENCY_MULTIPLIER` | OK |
| 75/85 | `200 * urgency * (1/(1+gap))` | 200 | `GAP_PULL_BASE` | OK |
| 76/86 | `40 * (gapTop + gapBottom)` | 40 | `GAP_PENALTY_BASE` | OK |
| 93 | `0.05 * game.moveCount` | 0.05 | `DRIFT_PENALTY` | OK |

## evaluateMove() Constants

| JS Line | JS Code | Value | Python Constant | Status |
|---------|---------|-------|-----------------|--------|
| 123 | `100 + distance * 5` | 100, 5 | `CONNECTION_BASE_BONUS`, `CONNECTION_DISTANCE_MULT` | OK |
| 133 | `if (spansBoard) score += 300` | 300 | `SPAN_BOARD_BONUS` | OK |
| 134 | `else if (wideSpan) score += 150` | 150 | `SPAN_WIDE_BONUS` | OK |
| 145 | `connectionCount * 75` | 75 | `MULTI_CONNECTION_MULT` | OK |
| 153/159 | `max(0, 12 - distanceToNearestGoal) * 8` | 12, 8 | `GOAL_DISTANCE_MAX`, `GOAL_DISTANCE_MULT` | OK |
| 179 | `opponentThreats * 25` | 25 | `OPPONENT_THREAT_MULT` | OK |
| 185 | `max(0, 24 - centerDistance) * 2` | 24, 2 | `CENTER_BIAS_MAX_DIST`, centerBias knob | OK |

## evaluateConnectedPaths() Constants

| JS Line | JS Code | Value | Python Constant | Status |
|---------|---------|-------|-----------------|--------|
| 209 | `avgComponentSize * 20` | 20 | `AVG_COMPONENT_SIZE_MULT` | ✓ FIXED |
| 211 | `(components.length - 3) * 30` | 30 | `COMPONENT_PENALTY_MULT` | ✓ FIXED |

## scoreComponent() Constants

| JS Line | JS Code | Value | Python Constant | Status |
|---------|---------|-------|-----------------|--------|
| 298 | `component.length * 10` | 10 | `COMPONENT_SIZE_MULT` | ✓ FIXED |
| 304/313 | `span * 20` | 20 | `SPAN_MULT` | ✓ FIXED |
| 307/316 | `+500` for full span | 500 | `FULL_SPAN_BONUS` | ✓ FIXED |

## evaluateWinningThreats() Constants

| JS Line | JS Code | Value | Python Constant | Status |
|---------|---------|-------|-----------------|--------|
| 227/234 | Full span (0↔23) | 800 | `THREAT_FULL_SPAN` | ✓ FIXED |
| 228/235 | Near span (≤1↔≥22) | 400 | `THREAT_NEAR_SPAN` | ✓ FIXED |
| 229/236 | Close span (≥22, ≤5) | 400 | `THREAT_NEAR_SPAN` | ✓ FIXED |
| 230/237 | Medium span (≤3↔≥20) | 200 | `THREAT_MEDIUM_SPAN` | ✓ FIXED |

## evaluatePotentialConnections() Constants

| JS Line | JS Code | Value | Python Constant | Status |
|---------|---------|-------|-----------------|--------|
| 333-337 | `score += 5` | 5 | `POTENTIAL_MOVE_BONUS` | ✓ FIXED |

## evaluateEdgeProgress() Constants

| JS Line | JS Code | Value | Python Location | Status |
|---------|---------|-------|-----------------|--------|
| 352-356 | `max(0, 12 - distanceToGoal)` | 12 | `EDGE_DISTANCE_MAX` | ✓ FIXED |

Python `evaluate_edge_progress()` now matches JS - simple loop adding `max(0, 12-dist)` per peg.

## Search-Phase Constants (search.js)

These are in the search phase, NOT in heuristics.js:

| Location | Description | Value | Python Location | Status |
|----------|-------------|-------|-----------------|--------|
| search.js | `nearFinishBonus` when near spanning | 2500 | `search.py` finish_bonus | OK |
| search.js | `immediateScore * 5` | 5 | `search.py` total_score | OK |
| search.js | `positionScore * 0.1` | 0.1 | `search.py` total_score | OK |

## Hardcoded Constants in movePriority (search.js) - NOT IN search.json

These constants are hardcoded in `movePriority()` in search.js and are **not configurable** via search.json:

| JS Line | Feature Name | Value | Description | Python Constant |
|---------|--------------|-------|-------------|-----------------|
| 1155 | threatReduction | `* 140` | Per-point threat reduction bonus | `THREAT_REDUCTION_MULT = 140` |
| 1160 | noThreatReduction | `-600 / -250` | Penalty when threat not reduced (urgent/normal) | `NO_THREAT_URGENT = 600`, `NO_THREAT_NORMAL = 250` |
| 1498 | opponentSpanReduction | `* 120` | Per-row/col opponent span reduction bonus | `SPAN_REDUCTION_MULT = 120` |
| 1506 | noSpanReductionPenalty | `-400` | Penalty when opponent urgent but span not reduced | `NO_SPAN_REDUCTION_PENALTY = 400` |
| 1517 | blackSpanUpgradePenalty | `-500` | Penalty if black newly spans both edges | `SPAN_UPGRADE_PENALTY = 500` |
| 1527 | redSpanUpgradePenalty | `-500` | Penalty if red newly spans both edges | `SPAN_UPGRADE_PENALTY = 500` |
| 1544/1564 | topBias/bottomBias | `* 12` | Defensive position bias near opponent edge | `DEFENSIVE_BIAS_MULT = 12` |
| 1551/1571 | aboveMinRowBonus/belowMaxRowBonus | `* 150` | Bonus for getting above/below opponent min/max | `DEFENSIVE_POSITION_BONUS = 150` |
| 1557/1577 | belowMinRowPenalty/aboveMaxRowPenalty | `* 90` | Penalty for staying below/above opponent min/max | `DEFENSIVE_POSITION_PENALTY = 90` |

**Note:** These should be added to Python `heuristics.py` as named constants for consistency.

## Tunable Knobs (from search.json)

These come from `search.json` and are tunable by the auto-tuner. They should NOT replace the hardcoded constants above:

### General Knobs (`rewards.general`)

| Key | Default Value | Python Key | Description |
|-----|---------------|------------|-------------|
| friendlyConnection | 12 | friendlyConnection | Bonus per knight-move neighbor (friendly peg) |
| opponentConnection | 35 | opponentConnection | Bonus per knight-move neighbor (opponent peg) |
| friendlyDistance | 3 | friendlyDistance | Bonus for proximity to friendly pegs |
| opponentDistance | 12 | opponentDistance | Bonus for proximity to opponent pegs |
| goalDistance | 1.2 | goalDistance | Bonus for proximity to goal edge |
| centerBias | 0.5 | centerBias | Center preference (early game) |
| isolated | 10 | isolated | Bonus when no nearby pegs |
| redGlobalMultiplier | 1.0 | redGlobalMultiplier | Red score multiplier |
| blackGlobalScale | 1.0 | blackGlobalScale | Black score scale |
| redBaseBonus | 0.0 | redBaseBonus | Flat bonus for Red |
| blackBasePenalty | 0.0 | blackBasePenalty | Flat penalty for Black |
| lateGameStart | 60 | lateGameStart | Move count when late game begins |
| lateGamePressure | 0.0 | lateGamePressure | Per-turn penalty after lateGameStart |

### Edge Offense Knobs (`rewards.edge.offense`)

| Key | Default Value | Python Key | Description |
|-----|---------------|------------|-------------|
| firstEdgeTouchRed | 420 | firstEdgeRed | Bonus for first connecting to goal edge (Red) |
| firstEdgeTouchBlack | 455 | firstEdgeBlack | Bonus for first connecting to goal edge (Black) |
| finishBonusBase | 3332 | finishBonusBase | Bonus for spanning both edges |
| finishBonusScale | 1.0 | finishBonusScale | Multiplier for finish bonus |
| nearFinishBonus | 2500 | nearFinishBonus | Bonus when close to finishing |
| finishGapSlope | 150 | finishGapSlope | Per-row/col bonus when finishing |
| finishThreshold | 4 | finishThreshold | Distance threshold for "near finish" |
| finishPenaltyBase | 1181 | finishPenaltyBase | Penalty base for stalling |
| spanGainBase | 180 | spanGainBase | Bonus for span increase |
| connectorBonus | 608 | connectorBonus | Bonus for connector moves |
| connectorBonusScale | 1.0 | connectorBonusScale | Multiplier for connector bonus |
| connectorTargetBonus | 500 | connectorTargetBonus | Bonus for targeting connector cells |
| doubleCoverageBase | 2400 | doubleCoverageBase | Base bonus for touching both goal edges |
| gapDecay | 23 | gapDecay | Gap reduction bonus |
| gapDecayScale | 1.0 | gapDecayScale | Multiplier for gap decay |
| redFinishPenaltyFactor | 0.55 | redFinishPenaltyFactor | Red-specific finish penalty multiplier |
| blackFinishScaleMultiplier | 1.0 | blackFinishScaleMultiplier | Black-specific finish scale |
| redSpanGainMultiplier | 1.0 | redSpanGainMultiplier | Red-specific span gain multiplier |
| blackSpanGainMultiplier | 1.0 | blackSpanGainMultiplier | Black-specific span gain multiplier |
| redDoubleCoverageBonus | 1000 | redDoubleCoverageBonus | Red double-coverage bonus |
| blackDoubleCoverageScale | 0.8 | blackDoubleCoverageScale | Black double-coverage scale |
| redFinishExtra | 0.0 | redFinishExtra | Extra finish bonus for Red |
| redGapDecayMultiplier | 1.0 | redGapDecayMultiplier | Red gap decay multiplier |

### Edge Defense Knobs (`rewards.edge.defense`)

| Key | Default Value | Python Key | Description |
|-----|---------------|------------|-------------|
| blockBonus | 900 | blockBonus | Bonus for blocking opponent connectors |
| missPenalty | 350 | missPenalty | Penalty for missing blocks |

### Edge Radius (`rewards.edge.radius`)

| Key | Default Value | Python Key | Description |
|-----|---------------|------------|-------------|
| radius | 3 | edgeRadius | Radius for connector target computation |

### Value Model

| Key | Default Value | Python Key | Description |
|-----|---------------|------------|-------------|
| valueModelScale | 600 | valueModelScale | Scale factor for value model adjustment |

## Key Distinction: Heuristics vs Search Phase

**CRITICAL**: There are TWO places where scoring happens:

1. **heuristics.js functions** - Use HARDCODED constants (100, 20, 30, etc.)
   - `evaluatePosition()`, `evaluateMove()`, `scoreComponent()`, etc.
   - These MUST use hardcoded values to match JS

2. **search.js phase** - Uses knobs from `search.json`
   - `finishBonus`, `nearFinishBonus`, `connectorBonus`, etc.
   - These are tunable and should come from knobs

Python was incorrectly using knob values (finishBonusBase=3332, connectorBonus=608) in heuristic functions where JS uses hardcoded values (500, 5).

## PORTED: `movePriority` Function (search.js lines 946-1450+)

**STATUS: PORTED** - The `move_priority()` function in Python `heuristics.py` now matches JS `movePriority`.

**Purpose:** This function orders moves for alpha-beta search. Good move ordering = better pruning = faster search. It uses TUNABLE KNOBS (not hardcoded constants) because these values are what the auto-tuner optimizes.

**Call chain:**
```
choose_move()
  → score_moves()
    → move_priority() [uses KNOBS]
  → minimax()
    → evaluate_position() [uses HARDCODED]
```

**Key difference from evaluate_move():**
- `evaluate_move()` = basic scoring with HARDCODED constants (from heuristics.js)
- `move_priority()` = advanced scoring with TUNABLE knobs (from search.js)

The function applies all REWARDS from search.json:
- `connectorTargetBonus` (500) - for moves targeting connectors
- `blockBonus` (900) - for blocking opponent connectors
- `friendlyConnection` (12) - per friendly connection
- `opponentConnection` (35) - per opponent connection
- `friendlyDistance` (3) - distance to friendly pegs
- `opponentDistance` (12) - distance to opponent pegs
- `goalDistance` (1.2) - distance to goal edges
- `centerBias` (0.5) - center preference
- `isolated` (10) - isolated move bonus
- Chain/frontier proximity bonuses
- First edge touch bonuses (425/455)
- Double coverage bonuses
- Span gain bonuses (180)
- Gap decay bonuses (23)
- Finish bonuses (3332)

**Results after porting:**
| Depth | Before | After |
|-------|--------|-------|
| d2 | 156 moves (384s) | 57 moves (19s) |
| d3 | 191 moves (3320s) | 37 moves (107s) |

### Complete `movePriority` Feature List (2024-12-28)

All features from JS search.js lines 946-1660 are now ported:

| Feature | JS Lines | Description |
|---------|----------|-------------|
| friendlyConnection | 1025-1030 | Bonus per knight-move neighbor (friendly peg) |
| opponentConnection | 1032-1037 | Bonus per knight-move neighbor (opponent peg) |
| friendlyDistance | 1039-1047 | Bonus for proximity to friendly pegs |
| opponentDistance | 1049-1057 | Bonus for proximity to opponent pegs |
| goalDistance | 1059-1068 | Bonus for proximity to goal edge |
| centerBias | 1070-1078 | Bonus for central positions |
| isolatedBonus | 1080-1084 | Bonus when no nearby pegs |
| chainProximity | 1086-1100 | Bonus for proximity to opponent's largest component |
| frontierProximity | 1102-1115 | Bonus for proximity to opponent frontier |
| frontierCapture | 1113-1115 | Big bonus for capturing opponent frontier cell |
| connectorProximity | 1117-1130 | Bonus for proximity to opponent connectors |
| connectorCapture | 1128-1130 | Big bonus for capturing opponent connector |
| **trailingPenalty** | 1132-1139 | Penalty for moves near opponent trailing cells |
| connectorTargetBonus | 1014-1017 | Bonus for targeting friendly connector targets |
| blockBonus | 1019-1023 | Bonus for blocking opponent connectors |
| threatReduction | 1147-1161 | Bonus for reducing opponent's connectivity |
| noThreatReduction | 1147-1161 | Penalty for not reducing threat |
| firstEdgeTouch | 1281-1311 | Bonus for first connecting to goal edge |
| **componentSpansBoth** | 1319-1350 | Double coverage requires largest component to span |
| doubleCoverage | 1313-1350 | Bonus when newly touching both goal edges |
| spanGain | 1352-1368 | Bonus for increasing span |
| gapReduction | 1370-1381 | Bonus for reducing gap to edges |
| **largestComponentSpanComplete** | 1383-1411 | Big bonus when largest component nearly spans both edges |
| edgeFinishAdvance | 1413-1457 | Bonus for advancing when near finish |
| edgeFinishStall | 1413-1457 | Penalty for stalling when near finish |
| missPenalty | 1463-1476 | Penalty for missing defense when opponent has connectors |
| **opponentSpanReduction** | 1483-1498 | Bonus for reducing opponent's span |
| **noSpanReductionPenalty** | 1500-1504 | Penalty when opponent is urgent but span not reduced |
| **spanUpgradePenalty** | 1506-1525 | Penalty if opponent newly spans both edges |
| **defensiveBiases** | 1534-1575 | Position-based defense when opponent touches one edge |
| **redBaseBonus** | 1592-1598 | Flat bonus for Red (currently 0) |
| **blackBasePenalty** | 1600-1606 | Flat penalty for Black (currently 0) |
| **redGlobalMultiplier** | 1608-1614 | Score multiplier for Red (currently 1.0) |
| **blackGlobalScale** | 1616-1622 | Score scale for Black (currently 1.0) |
| **lateGamePressure** | 1624-1639 | Per-turn penalty after lateGameStart (currently 0) |

**Bold** items were added in the 2024-12-28 update to complete JS parity.

## Additional Helper Functions (search.js)

These helper functions from search.js are now ported to Python:

### `hasReachableGoalEdge()` (Sealed Lane Detection)

**Purpose:** BFS to check if a player can still reach their goal edge. Returns `False` if the player's path to victory is completely blocked by opponent bridges/pegs.

**Usage in Python:** `has_reachable_goal_edge(state, player, metrics)`

**Integration:** Called before applying finish bonuses in `move_priority()`. If lane is sealed, finish bonuses are skipped.

### `computeConnectorTargets()`

**Purpose:** Compute target cells at the edge of the largest component that would extend toward the goal.

**Usage in Python:** `compute_connector_targets(state, player, metrics, radius=3)`

**Returns:** Set of "row:col" strings for cells that extend component toward goal.

**Integration:** Used in `move_priority()` for the connectorTargetBonus (500).

### `opponentUrgent` Calculation

**Purpose:** Determine if opponent is in an urgent (threatening) position.

**JS Logic:**
```javascript
const opponentUrgent = spanValue >= Math.max(6, Math.floor(boardSize / 4)) || largestLength >= 6;
```

**Usage in Python:** `compute_opponent_urgent(opponent, opponent_metrics, board_size)`

**Integration:** Used in `move_priority()` to adjust defense bonuses when opponent is threatening.

### `isGoalEdgeCoordinate()`

**Purpose:** Check if a coordinate is on the player's goal edge (not corners).

**Usage in Python:** `is_goal_edge_coordinate(player, row, col, board_size)`

## Value Model Integration

**File:** `scripts/GPU/ai/value_model.py`

The value model is a logistic regression that predicts P(win) from position features.

**Model Format:**
```json
{
  "type": "logistic_regression",
  "feature_keys": ["feature1", "feature2", ...],
  "weights": [bias, w1, w2, ...],
  "preproc": {
    "standardize": true,
    "mean": [...],
    "std": [...]
  }
}
```

**Integration:**
- `valueModelScale` knob (default: 600) controls adjustment magnitude
- Adjustment = `(probability - 0.5) * valueModelScale`
- Added to move scores in `score_moves()` if model is loaded

**Knob:**
| Key | Default Value | Description |
|-----|---------------|-------------|
| valueModelScale | 600 | Scale factor for value model adjustment |

## Sealed Lane Detection (`sealed_lane.py`)

**Purpose:** BFS-based reachability checking to determine if a player can still reach their goal edge. Used in `move_priority()` to skip finish bonuses when lane is sealed.

**Architecture:**

```
sealed_lane.py:
├── LaneKey (collision-safe cache key)
│   ├── player: 0=red, 1=black
│   ├── roi: (r0, r1, c0, c1) corridor bounds
│   ├── target_edges: bitmask (1=top/left, 2=bottom/right)
│   ├── comp_mask: exact bitmask of component pegs
│   ├── self_mask: exact bitmask of ALL friendly pegs in ROI
│   ├── opp_mask: exact bitmask of opponent pegs in ROI
│   └── bridges_sig: packed bridge endpoints (bbox-intersection filtered)
│
├── SealedLaneLRU (bounded LRU cache, 50k entries)
│   └── get(), put(), get_or_compute(), stats()
│
├── _compute_corridor_roi() - ROI strategy:
│   ├── Primary axis: extends to missing goal edges
│   ├── Orthogonal axis: bbox ± KNIGHT_MARGIN
│   └── ALWAYS applies KNIGHT_MARGIN even when touching
│
├── has_reachable_goal_edge_bounded() - BFS with invariants:
│   ├── Check ROI bounds BEFORE accessing any state
│   ├── Goal edge success requires is_goal_edge_coordinate()
│   └── Empty cells only expand if is_valid_placement()
│
├── check_sealed_lane() - single check with optional cache
└── sealed_lane_open_batch() - batch de-dup API
```

**Key Design Decisions:**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Key collision safety | Exact bitmasks, not hashes | MD5 is probabilistic; bitmasks are collision-free |
| ROI strategy | Corridor-to-goal | Balance cache hit rate vs correctness |
| Bridge filtering | Bbox-intersection | Catches bridges whose segment crosses ROI even if endpoints outside |
| Empty cell expansion | Requires is_valid_placement() | Matches JS semantics for reachability |
| Cache size | 50k entries per worker | Empirically good hit rate without memory explosion |

**Constants:**

| Constant | Value | Description |
|----------|-------|-------------|
| KNIGHT_MARGIN | 2 | BFS can reach 2 squares beyond component bbox |
| BRIDGE_MARGIN | 4 | Bridges can block paths up to 4 squares from endpoints |

**Integration in heuristics.py:**

```python
# score_moves() accepts optional cache
def score_moves(..., sealed_lane_cache: Optional[SealedLaneLRU] = None):
    ...

# move_priority() uses cache for finish bonus decisions
if sealed_lane_cache is not None:
    lane_open = check_sealed_lane(child_state, player, component, ...)
```

**Performance:** Cache provides ~70x speedup over uncached BFS.

## JS Oracle Alignment

**Purpose:** Verify Python AI implementations match JS exactly before training.

**Why it matters:**
```
Training (Python + GPU)     →     Deployment (JS + Browser)
      ↓                                    ↓
  Python semantics              JS semantics
      ↓                                    ↓
  Model learns X            Model expected to do X
                                   (but JS does Y if different)
```

If semantics differ, the model learns the wrong game dynamics.

**Test infrastructure:**

| File | Purpose |
|------|---------|
| `tests/js_oracle/sealed_lane_oracle.js` | Node.js oracle for hasReachableGoalEdge |
| `tests/js_oracle/test_sealed_lane_oracle.py` | Cross-validation tests |
| `tests/js_oracle/README.md` | Documentation |

**Running alignment tests:**
```bash
python3 tests/js_oracle/test_sealed_lane_oracle.py
```

**Current alignment status:**

| Function | Python | JS | Status |
|----------|--------|-----|--------|
| `hasReachableGoalEdge` | `sealed_lane.py` | `search.js:215` | ✅ 100% aligned |
| `isGoalEdgeCoordinate` | `sealed_lane.py` | `search.js:202` | ✅ 100% aligned |
| `isLegalPlacementForPlayer` | `board.py` | `search.js:183` | ✅ 100% aligned |
| `bridgesCross` | `bridge.py` | `twixtGame.js` | ✅ 100% aligned |
| `movePriority` | `heuristics.py` | `search.js:946-1660` | ✅ Aligned (tolerance <600) |
| `evaluatePosition` | `heuristics.py` | `heuristics.js` | ✅ <100 difference |
| `evaluateMove` | `heuristics.py` | `heuristics.js` | ✅ <50 difference |
| `connectivityScore` | `heuristics.py` | `heuristics.js` | ✅ <1 difference |
| `componentMetrics` | `heuristics.py` | `heuristics.js` | ✅ 100% aligned |
| `computeFrontier` | `heuristics.py` | `heuristics.js` | ✅ 100% aligned |

**Semantics verified:**
1. Goal edge definition (excludes corners)
2. Placement validity (edge restrictions per player)
3. Traversal rules (empty cells only expand if placeable)
4. Bridge crossing (both colors' bridges block)
5. Opponent blocking (opponent pegs impassable)

**When to run:**
- Before training a new model
- After changes to `sealed_lane.py`, `heuristics.py`, `bridge.py`
- As part of CI before release

**IMPORTANT: Serializing bridges for JS oracle:**

When converting Python `GameState` to JS format for oracle testing, **always use `state.bridges`** directly:

```python
# CORRECT: Use state.bridges (accounts for bridge crossings)
def state_bridges_to_js(state):
    bridges = []
    for (r1, c1), (r2, c2) in state.bridges:
        player = state.pegs.get((r1, c1))
        bridges.append({"r1": r1, "c1": c1, "r2": r2, "c2": c2, "player": player})
    return bridges

# WRONG: Naively computing knight-move adjacencies (ignores blocked bridges!)
def find_bridges_BUGGY(state):  # DO NOT USE
    for (r, c), player in state.pegs.items():
        for dr, dc in KNIGHT_OFFSETS:
            # This doesn't check if the bridge is blocked by a crossing!
            ...
```

The naive approach will include bridges that are actually blocked by opponent bridges crossing them, causing the JS oracle to compute different component sizes and incorrect scores.

## Update History

- 2024-12-23: Initial documentation created while debugging Python vs JS game length difference
- Key issue found: Python was using knob values (12, 3, 1.2) where JS uses hardcoded (100, 20, 30)
- Additional issues found: Python evaluate_edge_progress is too complex; score_component uses wrong values
- evaluateWinningThreats uses wrong threat level values (should be 800/400/400/200)
- evaluatePotentialConnections uses wrong value (should be 5, not connectorBonus/100)
- 2024-12-24: Ported movePriority from JS search.js to Python heuristics.py
  - Game length reduced from 156-191 moves to 37-57 moves
  - Game time reduced by 20-31x
- 2024-12-24: Added helper functions from search.js
  - `has_reachable_goal_edge()` - Sealed lane detection via BFS
  - `compute_connector_targets()` - Connector target computation
  - `compute_opponent_urgent()` - Opponent urgency calculation
  - `is_goal_edge_coordinate()` - Goal edge coordinate check
  - Integrated all into `move_priority()`
- 2024-12-24: Added value model integration
  - Updated `value_model.py` to match JS valueModel.js
  - Added valueModelScale knob (600)
  - Integrated into `score_moves()`
- 2024-12-24: Added collision-safe sealed lane caching (`sealed_lane.py`)
  - LaneKey with exact bitmasks (not MD5 hashes) for collision-free caching
  - Corridor ROI strategy: primary axis to missing goal edges, orthogonal bbox±margin
  - BFS invariants: ROI bounds check before state access, goal edge legality, placement validity
  - SealedLaneLRU with 50k entry limit, ~70x speedup over uncached
  - Integrated into `score_moves()` and `move_priority()` via optional cache parameter
  - Added sanity tests in `tests/test_sealed_lane.py`
- 2024-12-26: Added JS Oracle cross-validation for sealed lane detection
  - Created `tests/js_oracle/sealed_lane_oracle.js` - Node.js oracle script
  - Created `tests/js_oracle/test_sealed_lane_oracle.py` - Python cross-validation tests
  - Verified 100% alignment between Python and JS on 200+ test positions
  - Documented semantics: goal edges, placement validity, traversal rules, bridge crossing
  - Added `tests/js_oracle/README.md` with test infrastructure documentation
- 2024-12-28: Completed `movePriority` parity with JS
  - Added missing features: trailingPenalty, largestComponentSpanComplete, opponentSpanReduction,
    spanUpgradePenalty, defensiveBiases, componentSpansBoth check, global adjustments
  - Added new knobs: redGlobalMultiplier, blackGlobalScale, redBaseBonus, blackBasePenalty,
    redFinishExtra, redGapDecayMultiplier, lateGameStart, lateGamePressure
  - Created movePriority JS oracle and 6 new cross-validation tests
  - Total oracle tests: 42 (was 36)
  - Performance: d2=0.12s, d3=1.33s (59x/145x faster than baseline)
- 2024-12-28: Added CC caching and opponent CC invariance optimization
  - Revision-based cache invalidation for connected components
  - Opponent components passed through (invariant under our move)
  - 21 CC optimization tests including 3 invariance tests
- 2024-12-29: Documented hardcoded movePriority constants (not in search.json)
  - Fixed `edgeRadius` discrepancy: Python had 5, search.json has 3 - now aligned
  - Documented 9 hardcoded constants in search.js movePriority function:
    - threatReduction (*140), noThreatReduction (600/250)
    - opponentSpanReduction (*120), noSpanReductionPenalty (400)
    - spanUpgradePenalty (500), defensive biases (12/150/90)
  - Added named constants to Python heuristics.py for consistency
- 2024-12-29: Two-phase scoring optimization for `score_moves_batch`
  - Profiling revealed `extract_features` was 70% of time (86ms for 508 moves)
  - Root cause: `evaluate_potential_connections` called 1016x with 90k `is_valid_placement` checks
  - Solution: Two-phase approach:
    - Phase 1: Score ALL moves with `move_priority()` only (fast, ~9ms)
    - Phase 2: Extract features + value model for TOP-K candidates only
  - Added `value_model_top_k` parameter (default: 50)
  - Results: **3.9x speedup** (50ms → 13ms for 508 moves)
  - Increasing `top_n` to 50+ is now practical without performance penalty
- 2024-12-29: Ported CC caching optimizations to JS
  - **Opponent CC invariance** in `movePriority` (search.js:1485-1540)
    - Opponent's components don't change when we place our peg
    - Skip recomputing `opponentPost = componentMetrics()` - use passed-in `opponentMetrics`
    - Saves ~400 `componentMetrics()` calls per `getBestMove`
  - **CC caching** in `componentMetrics` (heuristics.js:415-484)
    - Added WeakMap cache keyed by game object + pegs.length + bridges.length + player
    - Cache hit returns immediately, avoiding `findConnectedComponents` traversal
    - Saves redundant computation when same state is queried multiple times
- 2024-12-29: JS/Python alignment debugging session
  - **Issue:** Python games 100% draws at 220 moves, JS games completed normally
  - **Root cause 1:** Defense miss penalty bug in Python `heuristics.py`
    - Was checking `opp_connectors` (frontier pegs, always empty in early game)
    - Should check `connector_targets` (knight-move positions from largest component)
    - Fixed at line 1421-1427: now uses `opp_connector_targets` from opponent_frontier
  - **Root cause 2:** Value model interference in oracle tests
    - JS oracle was applying value model adjustment (~300 points)
    - Added `clearValueModel()` to `valueModel.js` for pure heuristic testing
    - Oracle tests now disable value model for accurate comparison
  - **Root cause 3:** `top_n` mismatch between engines
    - Python comparison used `top_n=12`, but JS uses 20 at root (medium difficulty)
    - Updated `compare_engines.py` to use `top_n=20`
  - **Root cause 4:** JS minimax missing `opponentConnectorTargets` parameter
    - `minimax()` was passing parameters in wrong order to `orderMoves()`
    - Missing: `opponentConnectorTargets` computation
    - Caused: different move ordering at depth >0 vs root level
    - Fixed: Added `computeConnectorTargets()` call in minimax, passed correctly
  - **Root cause 5:** JS value model null probability bug
    - When value model is cleared, `evaluateValueModel()` returns `{probability: null}`
    - JS was computing `(null - 0.5) * 600 = -300` and adding to score
    - This caused a systematic -300 point offset in ALL JS scores
    - Fixed: Changed condition to `if (evaluation && evaluation.probability != null)`
    - File: `search.js:1619`
  - **Root cause 6:** Python firstEdgeRed knob mismatch
    - Python had `firstEdgeRed: 425`, JS has `firstEdgeTouchRed: 420`
    - Fixed: Changed Python DEFAULT_KNOBS to use 420
    - File: `heuristics.py:105`
  - **Root cause 7:** Python value model applied during comparison
    - Python `score_moves_batch` was using value model by default
    - Added `use_value_model` parameter to `choose_move()` and engine
    - Comparison now explicitly disables value model for fair comparison
    - Files: `search.py`, `selfplay/engine.py`, `compare_engines.py`
  - **Remaining issue:** ~250 point heuristic offset still present
    - Python scores ~250 points higher than JS on identical positions
    - Both engines now produce matching first-move scores (975.4)
    - But accumulated differences across game cause move selection divergence
  - **Key parameters for alignment:**
    | Parameter | JS Value | Python Value | Notes |
    |-----------|----------|--------------|-------|
    | Root level top_n | 20 (medium) | 20 | Matches |
    | Depth 1 top_n | 13 | 20 | JS uses depth-dependent formula |
    | missPenalty | 350 | 350 | Matches |
    | NO_THREAT_URGENT | 600 | 600 | Matches |
    | NO_THREAT_NORMAL | 250 | 250 | Matches |
    | blockBonus | 900 | 900 | Matches |
    | firstEdgeTouchRed | 420 | 420 | NOW MATCHES |
    | valueModelScale | 600 | 600 | Matches (but disabled in comparison) |
    | Depth-dependent move limit | Yes | Yes | NOW MATCHES |
- 2024-12-31: Fixed depth-dependent move limits in Python minimax
  - **Issue:** Python 100% draws (62 moves at d2) vs JS decisive games (45 moves)
  - **Root cause:** JS uses depth-dependent move limits in minimax:
    ```javascript
    limit = round((baseLimit * (depth + 1)) / (rootDepth + 1))
    // At root (depth=2): limit = 20
    // At depth 1: limit = round(20 * 2 / 3) = 13
    ```
  - Python used flat `top_n=20` at all depths
  - At Move 2, Red's killer response (2,12) was at heuristic rank #16
  - JS only looked at 13 moves at depth 1, never saw (2,12) → minimax = -6490
  - Python looked at 20 moves, found (2,12) → minimax = -8460 (1970 pts worse!)
  - This caused Python to prefer (22,2) over (20,0), diverging from JS
  - **Fix:** Added `root_depth` parameter to Python `minimax()` function
    - New formula matches JS: `limit = round((top_n * (depth+1)) / (root_depth+1))`
    - `choose_move()` now passes `root_depth=depth` when calling minimax
  - **Results after fix:**
    - Move 2: Python now picks (20,0) like JS (was (22,2))
    - First 13 moves now match exactly between engines
    - Python games now decisive: RED wins in 45 moves (was 100% draws at 62)
  - **Files changed:** `scripts/GPU/ai/search.py`
  - **Key alignment table updated:**
    | Parameter | JS Value | Python Value | Notes |
    |-----------|----------|--------------|-------|
    | Root level top_n | 20 | 20 | Matches |
    | Depth 1 limit (d2) | 13 | 13 | NOW MATCHES (was 20) |
    | Depth 0 limit (d2) | 7 | 7 | NOW MATCHES (was 20) |
- 2025-01-03: Fixed test script bridge serialization bug
  - **Issue:** Debug scripts showed Python/JS divergence at Turn 19 (black starts)
  - **Investigation:** Python chose (19,7), JS chose (17,12) with 1515 point score difference
  - **Root cause:** Test helper `find_bridges()` naively computed knight-move adjacencies
    - Did NOT check for bridge crossings (blocked bridges)
    - Passed invalid bridge `(17,11)↔(19,10)` to JS oracle
    - This bridge is actually blocked by red bridge `(18,9)↔(19,11)`
    - Caused JS oracle to compute 9 pegs in largest component (correct: 8)
  - **Fix:** Changed `find_bridges()` to `state_bridges_to_js()` using `state.bridges` directly
    - `state.bridges` correctly excludes blocked bridges
    - All debug scripts updated: `trace_game.py` and others
  - **Results:** Python and JS now produce identical moves for entire games (46+ turns)
  - **Key lesson:** Always use `state.bridges` for oracle serialization, never recompute naively
  - **Files changed:** `scripts/trace_game.py`, documentation added to this file
  - **Engines verified:** 100% move alignment for both red-starts and black-starts games
- 2025-01-15: Added training mode knob normalization and opening diversity
  - New knobs: `opening_random_plies`, `opening_random_top_k`, `enforce_symmetry`
  - Training uses stochastic tie-breaks; debug remains deterministic
- 2025-01-15: Updated training defaults for diversity/bias
  - `opening_random_plies` increased to 8
  - `opening_random_top_k` set to 0 (uniform random openings)
  - `enforce_symmetry` default set to 1
  - `opening_random_plies_d2` default set to 2
  - `opening_random_top_k_d2` default set to 6
- 2025-01-15: Added training throughput caps
  - `training_max_moves` default set to 50
  - `training_stall_limit` default set to 15
  - `training_max_moves_d2` default set to 80
  - `training_stall_limit_d2` default set to 30
- 2025-01-15: Added stall resolution by span margin
  - `training_span_win_margin` default set to 1
- 2025-01-15: Added training-only black span boost
  - `training_black_span_multiplier` default set to 1.1
- 2025-01-15: Added training-only goal/bridge pressure multipliers
  - `training_goal_distance_mult` default set to 1.4
  - `training_center_bias_mult` default set to 0.5
  - `training_span_gain_mult` default set to 1.3
  - `training_connector_bonus_scale` default set to 1.2
  - `training_finish_bonus_scale` default set to 1.2
- 2025-01-15: Added debug sampling controls
  - `debug_sample_rate` default set to 0.1
  - `debug_max_plies` default set to 80
  - `debug_trace` default set to 0
- 2025-01-15: Added opening book bridge and generator
  - Generator: `scripts/GPU/replay/opening_book.py`
  - JS loader: `assets/js/ai/openingBook.js`
- 2025-01-15: Added per-run outcome breakdown to tuning logs
  - JSONL rows now include `reasons` + `avg_moves` + `mode`
