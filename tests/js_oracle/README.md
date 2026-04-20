# JS Oracle Tests

This directory contains cross-validation tests that verify Python AI implementations match the JavaScript implementations exactly.

## Why This Matters

```
Training (Python + GPU)          Deployment (JS + Browser)
        │                                   │
        ▼                                   ▼
┌─────────────────┐              ┌─────────────────┐
│  Python AI      │              │    JS AI        │
│  - sealed_lane  │   must       │  - search.js    │
│  - heuristics   │   match      │  - heuristics   │
│  - search       │   exactly    │  - valueModel   │
└─────────────────┘              └─────────────────┘
        │                                   │
        ▼                                   ▼
   Value Model                        Value Model
   Training                           Inference
```

If Python and JS have semantic differences:
- Model learns Python's view of the game during training
- Model runs under JS's view of the game during deployment
- Model may make suboptimal or incorrect decisions

## Files

| File | Purpose |
|------|---------|
| `sealed_lane_oracle.js` | Node.js script that computes JS sealed lane detection |
| `bridge_crossing_oracle.js` | Node.js script that computes JS bridge crossing detection |
| `heuristics_oracle.js` | Node.js script that computes JS heuristic scores |
| `deterministic_game_oracle.js` | Node.js script that plays full deterministic games for parity testing |
| `test_oracle.py` | Consolidated Python test file comparing Python vs JS results |

## Running Tests

```bash
# Run all oracle tests with pytest
pytest tests/js_oracle/ -v

# Run only oracle tests (skips if Node.js unavailable)
pytest tests/ -m oracle -v

# Run all tests except slow ones
pytest tests/ -m "not slow" -v

# Run full test suite
pytest tests/ -v
```

## Requirements

- **Node.js**: Must be installed and in PATH
- **Python 3.8+**: With project dependencies

## Test Coverage

### Sealed Lane Detection

Tests verify that `sealed_lane.py:check_sealed_lane()` matches `search.js:hasReachableGoalEdge()`:

| Test Case | Description |
|-----------|-------------|
| `empty_component` | Empty component returns False |
| `single_peg_center` | Single peg in center has open lane |
| `touching_top_edge` | Peg on goal edge can reach other edge |
| `touching_bottom_edge` | Peg on goal edge can reach other edge |
| `black_touching_left` | Black player edge detection |
| `blocked_by_opponent` | Opponent pegs block traversal |
| `with_bridges` | Bridge crossing detection matches |
| `random_positions` | Statistical test on 100+ random positions |

### Semantics Verified

1. **Goal edge definition**: `isGoalEdgeCoordinate()` - excludes corners
2. **Placement validity**: `isLegalPlacementForPlayer()` - edge restrictions
3. **Traversal rules**: Empty cells only expand if placeable
4. **Bridge crossing**: Both colors' bridges block crossings
5. **Opponent blocking**: Opponent pegs are impassable

### Bridge Crossing Detection

Tests verify that `bridge.py:bridges_cross()` matches `twixtGame.js:bridgesCross()`:

| Test Case | Description |
|-----------|-------------|
| `empty_bridges` | No bridges means no crossings |
| `known_crossings` | Manually verified crossing pairs |
| `single_bridge` | Random candidates against single bridge |
| `shared_endpoints` | Edges sharing endpoints don't cross |
| `multiple_bridges` | Random configurations with 5-15 bridges |
| `exhaustive_sample` | 100x100 edge pair sample |

### Bridge Crossing Semantics Verified

1. **Bbox rejection**: Cheap filter skips 60-80% of checks
2. **Knight-edge simplification**: No collinear overlaps possible (gcd(1,2)=1)
3. **Shared endpoints**: Legal, not a crossing
4. **Orientation test**: CCW/CW/collinear detection
5. **Proper intersection**: Interior crossing only (not endpoint touching)

### Deterministic Game Parity

Tests verify that Python and JS engines produce identical move sequences in deterministic mode:

| Test Case | Description |
|-----------|-------------|
| `test_single_game_seed_0` | Full game parity with seed 0 (starts black) |
| `test_single_game_seed_1` | Full game parity with seed 1 (starts red) |
| `test_multiple_seeds_depth_2` | 5 games at depth 2 |
| `test_extended_parity` | 10 games for statistical confidence (slow) |
| `test_starting_player_alternation` | Verify seed determines starting player |
| `test_move_count_parity` | Total move counts match between engines |

### Deterministic Mode Semantics Verified

1. **Lexicographic tie-break**: When scores are equal, (row, col) ordering determines winner
2. **No randomness**: randomFactor disabled, temperature = 0
3. **Starting player**: seed % 2 == 0 starts black, odd starts red
4. **Move-by-move parity**: Exact same sequence of moves in both engines

## Adding New Oracle Tests

To test other Python/JS functions:

1. Create `<function>_oracle.js` that reads JSON from stdin, outputs result
2. Create `test_<function>_oracle.py` that:
   - Generates test positions
   - Calls both Python and JS implementations
   - Compares results
   - Reports discrepancies

## Debugging Failures

If tests fail:

1. Check the specific position that failed
2. Print the full state (pegs, bridges, component)
3. Manually trace through both Python and JS logic
4. Common issues:
   - Coordinate conventions (row/col vs x/y)
   - Edge case handling (corners, boundaries)
   - Bridge crossing implementation differences

## Integration with CI

These tests should run:
- Before any release to production
- After changes to `sealed_lane.py`, `heuristics.py`, or `search.py`
- As part of the "JS Oracle Alignment" phase in the GPU training pipeline
