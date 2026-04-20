# GPU Oracle Tests

Cross-validation tests ensuring Python AI implementations match JavaScript exactly.

## Why This Matters

```
Training (Python + GPU)              Deployment (JS + Browser)
        │                                       │
        ▼                                       ▼
┌─────────────────────┐              ┌─────────────────────┐
│     Python AI       │              │       JS AI         │
│  - heuristics.py    │    must      │  - heuristics.js    │
│  - sealed_lane.py   │    match     │  - search.js        │
│  - search.py        │   exactly    │  - valueModel.js    │
└─────────────────────┘              └─────────────────────┘
        │                                       │
        ▼                                       ▼
   Value Model                            Value Model
   Training                               Inference
```

**If Python and JS have semantic differences:**
- Model learns Python's view of the game during training
- Model runs under JS's view of the game during deployment
- Model may make suboptimal or incorrect decisions

## Test Coverage

| Function | Python | JS | Oracle | Status |
|----------|--------|-----|--------|--------|
| `check_sealed_lane` | `sealed_lane.py` | `hasReachableGoalEdge` | `sealed_lane_oracle.js` | ✅ |
| `component_metrics` | `heuristics.py` | `componentMetrics` | `heuristics_oracle.js` | ✅ |
| `compute_frontier` | `heuristics.py` | `computeFrontier` | `heuristics_oracle.js` | ✅ |
| `connectivity_score` | `heuristics.py` | `connectivityScore` | `heuristics_oracle.js` | ✅ |
| `move_priority` | `heuristics.py` | `movePriority` | `heuristics_oracle.js` | ✅ |
| `evaluate_connected_paths` | `heuristics.py` | `evaluateConnectedPaths` | `heuristics_oracle.js` | ✅ |

## Directory Structure

```
scripts/GPU/oracle/
├── README.md                    # This file
├── __init__.py                  # Package init
├── base.py                      # Base oracle class with Node.js interface
├── sealed_lane_oracle.js        # JS oracle for sealed lane detection
├── heuristics_oracle.js         # JS oracle for all heuristics functions
├── test_sealed_lane.py          # Python tests for sealed lane
├── test_heuristics.py           # Python tests for heuristics
└── run_all.py                   # Run all oracle tests
```

## Running Tests

```bash
# Run all oracle tests
python -m scripts.GPU.oracle.run_all

# Run specific test suites
python -m scripts.GPU.oracle.test_sealed_lane
python -m scripts.GPU.oracle.test_heuristics

# Run with verbose output
python -m scripts.GPU.oracle.run_all --verbose
```

## Requirements

- **Node.js**: v16+ (must be in PATH)
- **Python 3.8+**: With project dependencies

## How Oracle Tests Work

1. **Generate test positions**: Python creates random or specific game states
2. **Call Python implementation**: Get Python's result for the function
3. **Call JS oracle**: Send state as JSON to Node.js subprocess, get JS result
4. **Compare results**: Verify exact match (or within epsilon for floats)
5. **Report discrepancies**: Show detailed diff for any failures

## Adding New Oracle Tests

### 1. Add JS function to `heuristics_oracle.js`

```javascript
// In the dispatcher object
'myNewFunction': (input) => {
    const parsed = parseGameState(input);
    return myNewFunction(parsed.game, parsed.player);
}
```

### 2. Add Python test in `test_heuristics.py`

```python
def test_my_new_function(self):
    """Test myNewFunction matches JS."""
    state = create_test_state()

    # Python result
    py_result = my_new_function(state, "red")

    # JS result
    js_result = self.oracle.call("myNewFunction", {
        "boardSize": state.board_size,
        "pegs": pegs_to_json(state),
        "bridges": bridges_to_json(state),
        "player": "red"
    })

    self.assertEqual(py_result, js_result)
```

## Debugging Failures

When tests fail:

1. **Check specific position**: Print full state (pegs, bridges, player)
2. **Trace both implementations**: Step through Python and JS logic
3. **Common issues**:
   - Coordinate conventions (row/col vs x/y)
   - Edge case handling (corners, boundaries)
   - Bridge crossing implementation differences
   - Floating point precision (use epsilon comparison)
   - Tie-breaking order differences

## Integration with CI

These tests should run:
- Before any production release
- After changes to `heuristics.py`, `sealed_lane.py`, or `search.py`
- After changes to `heuristics.js` or `search.js`
- As part of the GPU training pipeline validation phase

## Acceptance Criteria

**All oracle tests must pass with 100% agreement** before:
- Starting GPU training runs
- Deploying model updates to production
- Merging heuristic changes to main branch

Zero tolerance for semantic differences between Python and JS.
