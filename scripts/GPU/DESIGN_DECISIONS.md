# TwixT GPU AutoTune - Design Decisions

This document locks down the game rules and design decisions for the Python/MLX implementation.
All values verified against the JavaScript implementation.

## 1. Pie Rule (Swap Rule)

**Decision: DISABLED**

The current JS engine does NOT implement a swap/pie rule. No swap option after first move.

## 2. Coordinate Conventions

| Property | Value |
|----------|-------|
| Board size | 24x24 (indices 0-23) |
| Origin | Top-left is (0, 0) |
| Red goal | Connects TOP (row 0) ↔ BOTTOM (row 23) |
| Black goal | Connects LEFT (col 0) ↔ RIGHT (col 23) |
| Starting player | Red moves first |

## 3. Edge Restrictions

**Corners are forbidden for all players:**
- (0, 0), (0, 23), (23, 0), (23, 23)

**Edge restrictions by player:**
- **Red** cannot place on cols 0 or 23 (left/right edges - black's goal edges)
- **Black** cannot place on rows 0 or 23 (top/bottom edges - red's goal edges)

## 4. Bridge Rules

**Knight-move offsets (8 directions):**
```
(-2, -1), (-2, +1), (-1, -2), (-1, +2),
(+1, -2), (+1, +2), (+2, -1), (+2, +1)
```

**Bridge creation:**
- Automatic when placing a peg knight-distance from same-player peg
- Bridge only created if it doesn't cross any existing bridge

## 5. Bridge Crossing Detection

**Endpoint touching is LEGAL (not a crossing):**
- If two bridges share an endpoint, they don't cross

**Use integer orientation tests:**
```python
def orient(ax, ay, bx, by, cx, cy):
    abx, aby = bx - ax, by - ay
    acx, acy = cx - ax, cy - ay
    v = abx * acy - aby * acx
    return 1 if v > 0 else (-1 if v < 0 else 0)
```

**Proper intersection (excluding endpoint touch):**
- Segments cross if they straddle each other's line
- Exclude cases where intersection is exactly at an endpoint

**Collinear overlaps:**
- Beyond shared endpoints count as crossing

## 6. Win Detection

**Red wins:**
- BFS from any peg in row 0 to any peg in row 23
- Traversal only via same-player bridges

**Black wins:**
- BFS from any peg in col 0 to any peg in col 23
- Traversal only via same-player bridges

**Note:** Goal edges are defined by peg positions (rows/cols 0 and 23),
not virtual edges. Pegs must physically exist on these rows/cols.

## 7. Draw/Termination Conditions

**Max moves limit:**
- Default: 220 moves total (configurable)
- If reached without winner → draw

**Stall detection:**
- Default: 40 consecutive moves without "progress" → draw
- "Progress" is defined as:
  - **Red:** span increases, OR touches top edge, OR touches bottom edge
  - **Black:** span increases, OR touches left edge, OR touches right edge

**Draw occurs if:**
- `stalled == True` (40+ moves without progress)
- OR `maxMoves` reached without `gameOver == True`

## 8. Versioning

| Version Field | Current Value |
|--------------|---------------|
| `rules_variant` | "standard" (no swap) |
| `engine_version` | "1.0.0" |
| `feature_spec_version` | "1.0.0" |
| `hash_spec_version` | "1.0.0" |

## 9. RNG Discipline

- One RNG stream per game: `seed = global_seed + game_index`
- Depth-specific: `seed = global_seed + depth * 100000 + game_index`
- Move ordering ties: stable sort by (score, row, col)
- Equal-best moves: first in sorted order (deterministic)

## 10. Storage Policy

| Game Type | Format |
|-----------|--------|
| Sweep games | Thin (move + score only) |
| Validation games | Full audit (candidates, features, decision reason) |
| Anomalies (>200 moves) | Full audit |
| On-demand | `--full-audit` flag |
