# TwixT AI Implementation

This directory contains the JavaScript AI implementation for TwixT.

## Files

| File | Purpose |
|------|---------|
| `heuristics.js` | Position evaluation, component analysis, move scoring |
| `search.js` | Minimax search with alpha-beta pruning, move ordering |
| `search.json` | Tunable knobs/parameters for heuristics |
| `valueModel.js` | Neural network value model for position evaluation |

## Performance Optimizations

### 1. Connected Components (CC) Caching with Zobrist Hashing

**Files:** `heuristics.js`, `game/zobrist.js`, `game/twixtGame.js`

**Problem:** `componentMetrics()` was called repeatedly for the same game state, each time traversing all pegs and bridges to find connected components.

**Solution:** Zobrist hashing with LRU cache for O(1) key computation and bounded memory.

**How it works:**
1. **Zobrist table** (`zobrist.js`): Pre-computed random 64-bit values for each (row, col, player) combination
2. **Incremental hash** (`twixtGame.js`): `game.zKey` is XOR'd on each `placePeg()` and `undo()`
3. **LRU cache** (`heuristics.js`): Cache keyed by BigInt `zKey`, bounded to 20k entries

```javascript
// zobrist.js - O(1) hash updates via XOR
export function getZobristTable(size) { ... }

// twixtGame.js - maintains zKey incrementally
placePeg(row, col) {
  this.zKey ^= this.zTable[row][col][playerIndex(this.currentPlayer)];
}

// heuristics.js - LRU cache with BigInt keys
class CCCache {
  get(game, player) {
    return this.cache.get(game.zKey)?.[playerIndex(player)];
  }
}
```

**Performance:** 1.6x speedup at d2, 1.2x at d3

**Why Zobrist over string hashing:**
- String keys (`pegs.map(...).join('|')`) are O(n) with heavy GC churn
- Zobrist keys are O(1) updates, no allocations during search

### 2. Opponent CC Invariance (`search.js`)

**Location:** Lines 1485-1540

**Problem:** Inside `movePriority()`, we were computing `opponentPost = componentMetrics(this.game, opponent)` after placing our peg. But opponent's pegs and bridges don't change when WE move!

**Solution:** Use the pre-computed `opponentMetrics` instead of recomputing.

```javascript
// Before (wasteful):
const opponentPost = componentMetrics(this.game, opponent);

// After (optimized):
// Opponent's components are INVARIANT under our move
const opponentPost = opponentMetrics;
```

**Savings:** ~400 `componentMetrics()` calls per `getBestMove` invocation.

**Note:** This means `spanReduction` will always be 0 (opponent's span doesn't change from our move), and the "newly spans both edges" checks will never trigger. These code paths are kept for semantic correctness but are effectively no-ops.

### 3. Sealed Lane Caching (`search.js`)

**Location:** Lines 447, 1261-1281

**Problem:** `hasReachableGoalEdge()` BFS was called repeatedly for similar positions.

**Solution:** `this.sealedLaneCache` Map stores results keyed by position signature.

### 4. Bridge Crossing Cache (`search.js`)

**Location:** Lines 312-320

**Problem:** `bridgesCross()` geometric checks were repeated for same bridge pairs.

**Solution:** `bridgesCross.cache` Map stores results keyed by bridge pair.

## Tunable Knobs (`search.json`)

All tunable parameters are in `search.json`. These can be adjusted by the auto-tuner.

### General Rewards
| Key | Default | Description |
|-----|---------|-------------|
| `friendlyConnection` | 12 | Bonus per knight-move neighbor (friendly) |
| `opponentConnection` | 35 | Bonus per knight-move neighbor (opponent) |
| `friendlyDistance` | 3 | Proximity bonus to friendly pegs |
| `opponentDistance` | 12 | Proximity bonus to opponent pegs |
| `goalDistance` | 1.2 | Proximity bonus to goal edge |
| `centerBias` | 0.5 | Center position preference |
| `isolated` | 10 | Bonus for isolated moves |

### Edge/Offense Rewards
| Key | Default | Description |
|-----|---------|-------------|
| `edge.radius` | 3 | Radius for connector target computation |
| `connectorBonus` | 608 | Bonus for connector moves |
| `connectorTargetBonus` | 500 | Bonus for targeting connectors |
| `finishBonusBase` | 3332 | Bonus for spanning both edges |
| `nearFinishBonus` | 2500 | Bonus when close to finishing |
| `finishThreshold` | 4 | Gap threshold for "near finish" |
| `spanGainBase` | 180 | Bonus per span increase |
| `doubleCoverageBase` | 2400 | Bonus for touching both edges |
| `gapDecay` | 23 | Gap reduction bonus |
| `firstEdgeTouchRed` | 420 | First edge touch bonus (Red) |
| `firstEdgeTouchBlack` | 455 | First edge touch bonus (Black) |

### Defense Rewards
| Key | Default | Description |
|-----|---------|-------------|
| `blockBonus` | 900 | Bonus for blocking opponent connectors |
| `missPenalty` | 350 | Penalty for missing defensive moves |

### Red/Black Asymmetry
| Key | Default | Description |
|-----|---------|-------------|
| `redFinishPenaltyFactor` | 0.55 | Red-specific finish penalty multiplier |
| `blackFinishScaleMultiplier` | 1.0 | Black-specific finish scale |
| `redDoubleCoverageBonus` | 1000 | Red double-coverage bonus |
| `blackDoubleCoverageScale` | 0.8 | Black double-coverage scale |

## Hardcoded Constants (NOT in search.json)

These constants are hardcoded in `search.js` `movePriority()` and cannot be tuned via search.json:

| Location | Constant | Value | Description |
|----------|----------|-------|-------------|
| Line 1155 | `threatReduction` | `* 140` | Per-point threat reduction bonus |
| Line 1160 | `noThreatReduction` | `600 / 250` | Penalty (urgent/normal) |
| Line 1498 | `opponentSpanReduction` | `* 120` | Per-row span reduction bonus |
| Line 1506 | `noSpanReductionPenalty` | `400` | Penalty when opponent urgent |
| Line 1517/1527 | `spanUpgradePenalty` | `500` | Penalty if opponent spans both |
| Line 1544/1564 | `defensiveBias` | `* 12` | Position bias multiplier |
| Line 1551/1571 | `positionBonus` | `* 150` | Above/below opponent bonus |
| Line 1557/1577 | `positionPenalty` | `* 90` | Above/below opponent penalty |

## Architecture

```
getBestMove()
  ├── orderMoves()
  │   └── movePriority() [uses search.json knobs]
  │       ├── componentMetrics() [CACHED]
  │       ├── computeConnectorTargets()
  │       ├── hasReachableGoalEdge() [CACHED]
  │       └── evaluateValueModel()
  │
  └── minimax()
      ├── evaluatePosition() [uses heuristics.js hardcoded constants]
      └── orderMoves() [recursive]
```

## Python Parity Notes

To add Zobrist caching to the Python/GPU implementation:

1. **Create zobrist table:** Same structure as JS, use NumPy for random 64-bit ints
2. **Add zKey to GameState:** Maintain incrementally in `apply_move()` / `undo_move()`
3. **Cache by zKey:** Replace any position-based caching with BigInt lookup

**Key difference:** Python uses immutable `GameState` objects (new instance per move), so caching works differently. The JS approach mutates a single game object, requiring Zobrist for correctness.

## Memory Management for Long-Running Self-Play

When running multiple games at depth 3+, memory management is critical to avoid OOM crashes.

### Required Node.js Flags

```bash
# Recommended for depth 3 self-play
node --max-old-space-size=4096 --expose-gc scripts/legacy/selfPlay.js -g 10 -d 3

# For deterministic parity testing (no value model)
node --max-old-space-size=4096 --expose-gc scripts/legacy/selfPlay.js -g 10 -d 3 --no-value-model --deterministic
```

### Critical Performance Gotcha: `extractPositionalFeatures()`

**Location:** `heuristics.js` lines 595-668, `search.js` line 1764

**Problem:** `extractPositionalFeatures()` computes expensive metrics for value model evaluation:
- `componentMetrics()` for both players
- `computeFrontier()`
- `connectivityScore()` for both players
- `evaluatePotentialConnections()` for both players
- `evaluateEdgeProgress()` for both players

If called unconditionally in `movePriority()` for every candidate move, this causes **10x slowdown** (from ~4 min/game to ~43 min/game at depth 3).

**Solution:** The call is guarded by `isModelLoaded()`:
```javascript
// search.js - Only extract features when value model is active
if (isModelLoaded()) {
  this.lastPositionalFeatures = extractPositionalFeatures(this.game, player);
}
```

**Lesson:** Always check `isModelLoaded()` before calling `extractPositionalFeatures()`. When running with `--no-value-model`, this entire code path is skipped.

### Memory Strategies

| Strategy | Location | Description |
|----------|----------|-------------|
| **Streaming game writes** | `selfPlay.js` | Writes each game to JSONL immediately instead of accumulating in memory |
| **CCCache size limit** | `heuristics.js` | LRU cache limited to 5,000 entries (was 20,000) |
| **No board snapshots** | `selfPlay.js` | Removed per-move board cloning; peg counts come from featureContext |
| **Cache clearing** | `search.js` | `resetAllSearchCaches()` + `clearCaches()` called between games |
| **Aggressive GC** | `selfPlay.js` | Triple `forceGC()` calls between games (requires `--expose-gc`) |

### Self-Play Memory Flow

```
Game N starts
├── resetAllSearchCaches()  // Clear CCCache
├── forceGC() x3            // Reclaim previous game's memory
├── new TwixTGame()         // Fresh game instance
├── new TwixTAI() x2        // Fresh AI instances
│
├── [play game loop]
│   └── getBestMove()
│       └── movePriority()  // CCCache hit/miss, bounded LRU
│
├── Write game to JSONL     // Don't accumulate in memory
├── clearComponentCache()   // Additional cleanup
└── Game N ends
```

### What NOT to Do

1. **Don't store board snapshots** - Each 24x24 board clone retains ~576 cells. With ~100 moves/game, this adds up fast.

2. **Don't accumulate traces in memory** - Use JSONL streaming for multi-game runs.

3. **Don't call `extractPositionalFeatures()` unconditionally** - This is the #1 performance killer when value model is disabled.

4. **Don't skip cache clearing between games** - Memory will grow unboundedly.

## Update History

- 2025-01-19: Added memory management documentation for depth 3+ self-play
- 2025-01-19: Fixed `extractPositionalFeatures()` performance regression (10x speedup)
- 2025-01-19: Added streaming JSONL writes to prevent OOM in multi-game runs
- 2024-12-29: Implemented Zobrist hashing for CC caching (1.6x speedup at d2)
- 2024-12-29: Added CC caching to `componentMetrics()`
- 2024-12-29: Added opponent CC invariance optimization to `movePriority()`
- 2024-12-29: Documented hardcoded constants in `movePriority()`
- 2024-12-28: Added value model integration with positional features
- 2024-12-28: Fixed value model to use `extractPositionalFeatures()` instead of move features
