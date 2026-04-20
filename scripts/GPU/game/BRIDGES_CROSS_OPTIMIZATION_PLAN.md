# bridges_cross() Optimization Plan

## Overview
Three-phase optimization for bridge crossing detection in TwixT.

**Important**: This is *performance* work. The 5 correctness issues (sealed_lane.py) are separate and must be completed first.

---

## Correctness Issues (Separate from Performance)

These are concrete correctness / cache-safety issues to close out before performance work:

| Issue | Description | Status |
|-------|-------------|--------|
| 1A | `_pegs_to_mask` must be collision-free (MD5 → exact bitmask) | ✅ Fixed |
| 1B | `_bridge_signature` must be collision-free (hash → exact bytes, bbox-intersection filter) | ✅ Fixed |
| 2 | `SealedLaneLRU.put()` must update value when key exists | ✅ Fixed |
| 3 | ROI must include KNIGHT_MARGIN even when touching an edge | ✅ Fixed |
| 4 | BFS must enforce goal-edge legality + empty-cell placement discipline | ✅ Fixed |

**Notes on fixes:**
- Issue 1A: Bitmask is deterministic and collision-free. Keep bit order consistent (LSB-first within each byte).
- Issue 1B: Bbox intersection against expanded ROI (not "endpoint in ROI") is what makes it safe.
- Issue 4: Don't need `(row,col,is_peg)` in visited; `(row,col)` alone is fine because expandability is enforced when enqueueing empties.

---

## Phase 1: Optimized Geometry Version ✅ (Complete)

**Goal**: Immediate 2-4x speedup with bbox rejection + simplified knight-edge geometry.

### Key Insights
1. **Knight-edge simplification**: For TwixT knight-move segments (±1,±2 or ±2,±1):
   - Collinear overlaps cannot happen between distinct knight-edges
   - No interior lattice points exist on knight-edge segments
   - Only need pure "proper intersection" orientation test

2. **Bbox rejection**: Cheap filter that skips 60-80% of bridges before geometry

3. **Remove unnecessary collinear handling**: `_proper_intersect_knight()` replaces heavy `segments_intersect()`

### Implementation (Complete)
- `_orient()`: Orientation test (CCW/CW/collinear)
- `_proper_intersect_knight()`: Fast 4-orientation check for knight edges
- `bridges_cross()`: Bbox rejection + proper intersection only

### Completed Items
- [x] Add debug assertion for knight-edge invariant (costs nothing with `python -O`)
- [x] Run exhaustive equivalence test: compare `segments_intersect()` vs `_proper_intersect_knight()` for all knight-edge pairs on 24×24
  - **Result**: 2,024 edges, 2,034,344 pairs tested, 8,752 crossings found, **0 mismatches**

### Final Tightened Version
Bind locals once for Python speed win:

```python
def bridges_cross(state, r1: int, c1: int, r2: int, c2: int) -> bool:
    bridges = state.bridges
    if not bridges:
        return False  # Early exit

    proper = _proper_intersect_knight  # Bind local

    # x=col, y=row
    a1x, a1y = c1, r1
    a2x, a2y = c2, r2

    a_minx = a1x if a1x < a2x else a2x
    a_maxx = a2x if a1x < a2x else a1x
    a_miny = a1y if a1y < a2y else a2y
    a_maxy = a2y if a1y < a2y else a1y

    for (br1, bc1), (br2, bc2) in bridges:
        # shared endpoint is legal
        if ((r1 == br1 and c1 == bc1) or (r1 == br2 and c1 == bc2) or
            (r2 == br1 and c2 == bc1) or (r2 == br2 and c2 == bc2)):
            continue

        b_minx = bc1 if bc1 < bc2 else bc2
        b_maxx = bc2 if bc1 < bc2 else bc1
        if b_maxx < a_minx or b_minx > a_maxx:
            continue

        b_miny = br1 if br1 < br2 else br2
        b_maxy = br2 if br1 < br2 else br1
        if b_maxy < a_miny or b_miny > a_maxy:
            continue

        if proper(a1x, a1y, a2x, a2y, bc1, br1, bc2, br2):
            return True

    return False
```

---

## Phase 2: Bridge Mask Bitmask Version (Next)

**Goal**: O(1) crossing check via precomputed conflict bitmasks.

### Critical Invariants (Must Be True or Silent Bugs)

| # | Invariant | Status |
|---|-----------|--------|
| 1 | Canonical edge normalization consistent everywhere | ✅ `normalize_edge()` matches `edge_index` definition |
| 2 | `bridge_mask` stays in sync with `bridges` set | ✅ Both updated in `add_bridges_for_new_peg()` |
| 3 | Circular import avoidance | ✅ `bridge_geom.py` split, lazy import in `_compute_conflicts()` |
| 4 | Knight-edge only intersection assumption | ✅ Valid: no collinear overlaps, gcd(1,2)=1, endpoints handled |
| 5 | Fixed 24×24 board size | ✅ Hardcoded with assert in `add_bridges_for_new_peg()` |

**Invariant 2 Warning**: If any other code paths do `state.bridges.add()/remove()` directly, they must also update `bridge_mask`. Best practice: route all mutations through one helper.

**Additional Notes**:
- **Legacy data**: If loading older saved states with un-normalized bridges, `rebuild_bridge_mask()` should defensively normalize each edge before lookup.
- **Bridge removal/undo**: If engine ever removes bridges (undo, rollback), must clear the bit: `bridge_mask &= ~(1 << idx)`. If search uses copy-forward states and never mutates backward, this is not needed.
- **Board size**: 24×24 is now a hard constraint. The precomputed tables are specific to this size.

---

### Must-Fix Before Implementation

#### 1. Circular Import Risk
Problem: `edge_index.py` imports `_proper_intersect_knight` from `.bridge`, but `bridge.py` needs to import `EDGE_TO_IDX`/`CONFLICTS` from `.edge_index`.

**Fix**: Move `_orient` + `_proper_intersect_knight` into a tiny shared module:
```
scripts/GPU/game/
├── bridge_geom.py      # _orient, _proper_intersect_knight (no deps)
├── edge_index.py       # imports from bridge_geom
└── bridge.py           # imports from bridge_geom AND edge_index
```

#### 2. Lazy Loading of CONFLICTS
Don't precompute at module import (avoids ~0.5s penalty for quick scripts).

**Fix**: Use `@lru_cache(maxsize=1)` or compute on first call:
```python
_CONFLICTS: List[int] | None = None

def get_conflicts() -> List[int]:
    global _CONFLICTS
    if _CONFLICTS is None:
        _CONFLICTS = _compute_conflicts()
    return _CONFLICTS
```

#### 4. Symmetric Conflict Precompute (Cuts Work ~2x)
Loop `j in range(i+1, n)` and set both bits at once:
```python
for i, e1 in enumerate(all_edges):
    ...
    for j in range(i + 1, num_edges):
        e2 = all_edges[j]
        if _shares_endpoint(e1, e2):
            continue
        ...
        if _proper_intersect_knight(...):
            conflicts[i] |= (1 << j)
            conflicts[j] |= (1 << i)
```

#### 5. Hoist get_conflicts() Outside Hot Loop
In `add_bridges_for_new_peg()`, call `get_conflicts()` once at the start, not per-candidate:
```python
edge_to_idx = get_edge_to_idx()
conflicts = get_conflicts()  # Hoist outside loop

for dr, dc in KNIGHT_OFFSETS:
    ...
    idx = edge_to_idx.get(edge)
    if idx is None:
        continue

    # Inline the crossing check (avoid function call overhead)
    if state.bridge_mask & conflicts[idx]:
        continue
```

#### 6. `bridges_cross_fast()` Requires Canonical Edge
The function returns `False` for invalid/non-canonical edges (silent failure).

**Options**:
- **(A) Document requirement**: Caller must pass canonical edge (current hot path does)
- **(B) Normalize internally**: Safer but adds overhead

**Decision**: Use option (A) for hot path in `add_bridges_for_new_peg()` (already normalized).
Provide a safe wrapper for external callers:
```python
def bridges_cross_fast(bridge_mask: int, edge: Edge) -> bool:
    """O(1) crossing check. REQUIRES canonical edge."""
    idx = get_edge_to_idx().get(edge)
    if idx is None:
        return False  # Invalid/non-canonical edge
    return (bridge_mask & get_conflicts()[idx]) != 0

def check_crossing(bridge_mask: int, p1: Tuple[int,int], p2: Tuple[int,int]) -> bool:
    """Safe wrapper that normalizes before checking."""
    edge = normalize_edge(p1, p2)
    return bridges_cross_fast(bridge_mask, edge)
```

#### 3. Board Size Assertion
Hardcoding `BOARD_SIZE = 24` is fine if engine is fixed at 24, but add assertion:
```python
assert state.board_size == 24, "Edge index assumes 24x24 board"
```
Or parameterize and cache per board size.

### Concept
1. **Enumerate all canonical knight-edges** on 24×24 board
   - 2,024 edges total
   - Each edge gets a unique index (0 to 2023)
   - Store mapping: `edge_to_idx` and `idx_to_edge`

2. **Precompute conflict matrix**
   - For each edge pair (i, j), determine if they cross (using `_proper_intersect_knight`)
   - Store as bitmask: `conflicts[i]` is a bitmask of all edges that cross edge i
   - 8,752 crossings total (0.4% sparse)

3. **Add `bridge_mask` to GameState**
   - `bridge_mask: int = 0` (bit i = 1 if edge i exists)

4. **O(1) crossing check**
   ```python
   def bridges_cross_fast(bridge_mask: int, edge_idx: int) -> bool:
       return (bridge_mask & get_conflicts()[edge_idx]) != 0
   ```

5. **Bridge placement updates mask**
   ```python
   state.bridge_mask |= (1 << edge_idx)
   ```

### Implementation Steps
1. Create `bridge_geom.py` with `_orient`, `_proper_intersect_knight`
2. Create `edge_index.py` with lazy-loaded edge mappings and symmetric conflict precompute
3. Update `bridge.py` to import from both
4. Add `bridge_mask: int = 0` to `GameState`
5. Implement `bridges_cross_fast()` using bitmask AND (keep for readability/external use)
6. Update `add_bridges_for_new_peg()`:
   - Hoist `get_conflicts()` outside the loop
   - Inline the crossing check in hot loop (avoid function call overhead)
   - Set mask bits on bridge creation
7. Keep `bridges_cross()` as fallback/verification during transition
8. Add board size assertion

### File Structure After Phase 2
```
scripts/GPU/game/
├── bridge_geom.py      # _orient, _proper_intersect_knight (NEW)
├── edge_index.py       # Edge indexing + lazy conflicts (NEW)
├── bridge.py           # bridges_cross, bridges_cross_fast, add_bridges_for_new_peg
└── state.py            # GameState with bridge_mask field
```

### Expected Performance
- Crossing check: O(1) bitwise AND instead of O(n) iteration
- Memory: ~2024 edges × 2024 bits ≈ 500KB for conflict matrix
- State copy: O(1) int copy vs O(n) set copy

---

## Phase 3: Mask as Canonical Storage (Later)

**Goal**: Eliminate `state.bridges` set entirely; mask is the source of truth.

### Changes
1. Remove `state.bridges: Set[Tuple[Tuple[int,int], Tuple[int,int]]]`
2. `state.bridge_mask: int` becomes canonical
3. Reconstruct bridge list when needed: `[idx_to_edge[i] for i in range(N) if mask & (1 << i)]`
4. Update all code that reads/writes `state.bridges`

### Benefits
- Smaller state footprint
- Faster state copying (single int vs set copy)
- Cleaner crossing checks

### Risks
- Need to update serialization/deserialization
- Need to update any code that iterates `state.bridges`
- Ensure bridge display/rendering can reconstruct from mask

---

## File Locations
- Geometry primitives: `scripts/GPU/game/bridge_geom.py` (NEW)
- Edge indexing: `scripts/GPU/game/edge_index.py` (NEW)
- Bridge logic: `scripts/GPU/game/bridge.py`
- State: `scripts/GPU/game/state.py`
- Tests: `tests/test_knight_edge_equivalence.py`

---

## Progress Log

| Date | Phase | Status | Notes |
|------|-------|--------|-------|
| 2024-12-27 | Phase 1 | Core complete | bbox + _proper_intersect_knight implemented |
| 2024-12-27 | Phase 1 | Complete | Debug assertion added, equivalence test passed (2M+ pairs, 0 mismatches) |
| 2024-12-27 | Phase 1 | Complete | Tightened version applied (early exit, local binding) |
| 2024-12-27 | Phase 2 | Complete | All implementation done, tests passed |
| TBD | Phase 3 | Not started | Remove state.bridges, use mask as canonical |

## Phase 2 Test Results (2024-12-27)

```
Phase 2 Equivalence Tests
==================================================
1. Basic tests:
  [PASS] Edge count: 2024
  [PASS] Empty state: no crossings

2. Conflict matrix:
  [PASS] Conflict matrix is symmetric

3. Random configuration tests (50 configs x 200 edges):
  [PASS] 9,902 tests, 789 crossings, 0 mismatches

4. Mask synchronization:
  [PASS] Mask sync verified for 30 bridge additions
  [PASS] rebuild_bridge_mask handles non-canonical edges
  [PASS] Round-trip bridges -> mask -> bridges

5. API tests:
  [PASS] check_crossing() normalizes correctly

ALL PHASE 2 EQUIVALENCE TESTS PASSED!
```

## JavaScript Alignment (2024-12-27)

Updated `assets/js/game/twixtGame.js` to match Python implementation:
- Added `TwixTGame.orient()` - static orientation test
- Added `TwixTGame.properIntersectKnight()` - simplified knight-edge intersection
- Updated `bridgesCross()` with bbox rejection + early exit

### JS-Python Oracle Test Results

```
Bridge Crossing Oracle Tests (Python vs JavaScript)
============================================================
1. Basic Tests:
  [PASS] 50 empty bridge tests, 0 mismatches
  [PASS] 5 known crossing cases verified

2. Single Bridge Tests:
  [PASS] 200 tests, 0 mismatches
  [PASS] 100 shared endpoint tests, 0 mismatches

3. Multiple Bridge Tests:
  [PASS] 299 tests, 0 mismatches

4. Exhaustive Sample:
  [PASS] 9,900 pairs, 46 crossings, 0 mismatches

ALL BRIDGE CROSSING ORACLE TESTS PASSED!
Python and JavaScript implementations match.
```

### Oracle Test Files
- `tests/js_oracle/bridge_crossing_oracle.js` - JS oracle script
- `tests/js_oracle/test_bridge_crossing_oracle.py` - Cross-validation tests
