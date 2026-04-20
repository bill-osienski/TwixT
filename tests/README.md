# TwixT Test Suite

Comprehensive test suite for the TwixT AI game engine. Tests cover game mechanics, AI heuristics, and Python-JavaScript alignment.

## Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run fast tests only (skip slow + oracle tests)
pytest tests/ -m "not slow and not oracle"

# Python-only tests (no Node.js required)
pytest tests/ -m "not oracle"
```

## Test Summary

| File | Tests | Description |
|------|-------|-------------|
| `test_batch_eval.py` | 14 | GPU batch evaluation: MLX/NumPy equivalence |
| `test_bridge_equivalence.py` | 9 | Bridge crossing: geometry + bitmask validation |
| `test_behavioral_regression.py` | 1 | JS move within Python top-K or eval delta (curated suite) |
| `test_cc_optimization.py` | 21 | Connected components: caching, immutability, opponent CC invariance |
| `test_sealed_lane.py` | 10 | Sealed lane detection and LRU cache |
| `js_oracle/test_oracle.py` | 42 | Python vs JS cross-validation (incl. movePriority) |

**Total: 96 tests**

## Markers

| Marker | Command | Description |
|--------|---------|-------------|
| `oracle` | `pytest -m oracle` | JS-Python cross-validation (requires Node.js) |
| `bridge` | `pytest -m bridge` | Bridge crossing tests |
| `slow` | `pytest -m slow` | Long-running tests (>5s) |

### Common Commands

```bash
# All bridge crossing tests
pytest tests/ -m bridge -v

# Fast tests only
pytest tests/ -m "not slow" -v

# Everything except oracle tests
pytest tests/ -m "not oracle" -v
```

### Behavioral Regression Logging

```bash
python3 scripts/record_behavioral_regression.py
```

Writes a time-series summary to `logs/behavioral-regression.json`.

## Test Descriptions

### test_bridge_equivalence.py

Validates bridge crossing optimizations:
- **Geometry equivalence**: `_proper_intersect_knight` matches `segments_intersect`
- **Bitmask equivalence**: O(1) bitmask lookup matches O(n) geometry check
- **Edge utilities**: normalization, round-trip encoding, conflict symmetry

### test_sealed_lane.py

Tests sealed lane detection heuristic and its LRU cache:
- Cache operations and LRU eviction
- Lane semantics (open/sealed detection)
- Cache correctness and performance
- Batch API validation

### js_oracle/test_oracle.py

Cross-validates Python AI against JavaScript implementations. Critical because training happens in Python (GPU) but deployment is in JavaScript (browser).

Covers:
- Bridge crossing detection
- Sealed lane detection
- Heuristics: `evaluatePosition`, `evaluateMove`, `connectivityScore`, `findConnectedComponents`, `componentMetrics`, `computeFrontier`
- Move ordering: `movePriority` (full move scoring with all features)
- Behavioral regression: JS move within Python top-K or eval delta (`test_behavioral_regression.py`)

**Known Difference:** `touches_*` fields differ by design (JS: all components, Python: largest only).

## File Structure

```
tests/
├── README.md
├── conftest.py              # Shared fixtures
├── test_batch_eval.py       # GPU batch evaluation
├── test_behavioral_regression.py
├── test_bridge_equivalence.py
├── test_cc_optimization.py  # Connected components caching
├── test_sealed_lane.py
└── js_oracle/
    ├── bridge_crossing_oracle.js
    ├── heuristics_oracle.js
    ├── sealed_lane_oracle.js
    └── test_oracle.py
```

## Requirements

- **Python 3.8+** with pytest
- **Node.js** (for oracle tests only)

## CI Commands

```bash
# Fast CI (no Node.js)
pytest tests/ -m "not oracle and not slow" -v --tb=short

# Full CI (requires Node.js)
pytest tests/ -v --tb=short
```
