# JS↔Python Heuristic AI Parity Design

**Date:** 2026-04-20
**Status:** Approved for implementation planning
**Spec owner:** bill-osienski

## 1. Problem

The `tests/js_oracle/test_oracle.py::TestDeterministicGameParity` tests assert that when the JS heuristic AI (`assets/js/ai/search.js` + `heuristics.js`) and the Python heuristic AI (`scripts/GPU/ai/search.py` + `heuristics.py`) run the same deterministic game (same seed, same depth, same `deterministic_mode=1` knob), they produce identical move sequences.

Currently 6 oracle parity tests + 1 behavioral regression test (`tests/test_behavioral_regression.py::test_js_move_within_python_topk`) fail because JS and Python pick different moves from the opening onward. Example divergence (seed=0, depth=2): Python picks `(12, 20)` for black's first move; JS picks `(11, 0)`. Divergence cascades — once the first move differs, the whole game differs.

**Root cause:** JS `search.js` contains heuristic features that Python has no equivalent for. A grep of JS `capture(...)` and `recordStat(...)` emission sites enumerates all heuristic features contributing to the score; the features without a Python counterpart include:

- `edgeFinishStall` (penalty for moves that touch a goal edge without advancing)
- `edgeFinishAdvance` (bonus for edge moves that improve span/gap)
- `finishLaneSealed` (guard condition blocking finish bonuses when lane is sealed)
- `doubleEdgeCoverage` (bonus for multi-edge threats)
- `redFinishExtra` / `blackFinishScaleMultiplier` (side-asymmetric finish scaling)
- `connectorCapture` / `connectorProximity` (connector-target scoring paths)

Python's `scripts/GPU/ai/heuristics.py::evaluate_move` and `search.py::choose_move` cover many of the same feature names (firstEdgeRed/Black, spanGainBase, goalDistance, etc.) but the scoring paths have drifted. The JS side has evolved past the Python port without the port following along.

## 2. Approach

**Line-by-line feature port.** For each JS heuristic feature without a Python equivalent, port the scoring formula verbatim — same magnitude, same side-asymmetric behavior, same lane-sealed guard. Reference JS `assets/js/ai/search.js` as the authoritative source; Python mirrors it.

**No algorithmic changes.** This is not an opportunity to rewrite or improve either side. It's straight translation. If a JS formula looks odd or suboptimal, preserve it exactly — align behavior first, refactor later in a separate spec.

**No new features.** Features that exist in Python but not JS stay. The goal is that JS and Python agree on move choice; either direction of drift is a parity bug.

## 3. Approach details

### 3.1 Feature enumeration

The JS heuristic AI spans **two files** — both must be audited:
- `assets/js/ai/search.js` — top-level search driver + edge/connector scoring
- `assets/js/ai/heuristics.js` — position metrics, component evaluation, move evaluation primitives

Emitting authoritative feature names from only `search.js` misses features implemented in `heuristics.js` (e.g. `componentMetrics`, `evaluatePosition` internals, frontier helpers). Task 1 **MUST** scan both JS files.

The enumeration sources (from each file):
- **Scoring sites:** every `capture(...)` and `recordStat(...)` call
- **Exported helpers:** every `export function ...` that returns a numeric score contribution
- **Knob lookups:** every `REWARDS.*` / `config.*` / `knobs.*` reference that drives a score
- **State mutation sites that alter future scoring decisions:** stateful caches and closures (`this.something = ...`, `globalThis.*`)

Example non-exhaustive list from the current JS (`grep -oh "capture('[A-Za-z]*'" assets/js/ai/search.js`):

```
aboveMaxRowPenalty, aboveMinRowBonus, belowMaxRowBonus, belowMinRowPenalty,
blackBasePenalty, blackGlobalScale, blackSpanUpgradePenalty, bottomBias,
centerBias, chainProximity, connectorCapture, connectorProximity,
doubleEdgeCoverage, edgeConnectorTarget, edgeDefenseBlock, edgeDefenseMiss,
edgeFinishAdvance, edgeFinishStall, edgeGapReduction, finishLaneSealed,
firstEdgeTouch, friendlyConnections, friendlyDistance, frontierCapture,
frontierProximity, goalDistance, immediateWin, isolatedBonus,
largestComponentSpanComplete, lateGamePressure, nearSpanFinish,
noSpanReductionPenalty, noThreatReduction, opponentConnections,
opponentDistance, opponentSpanReduction, redBaseBonus, redGlobalMultiplier,
redSpanUpgradePenalty, spanGain, threatReduction, topBias, trailingPenalty
```

Task 1 of the plan produces a **parity inventory report** that scans both JS files and all Python equivalents (`scripts/GPU/ai/heuristics.py`, `search.py`, `sealed_lane.py`, `move_ordering.py`), matches feature names (with snake_case ↔ camelCase normalization), and emits a CSV with columns:
`feature_name, in_js, in_py, js_files, js_sites, py_sites, divergence_kind` (in that order — the plan's inventory tool writes this header verbatim).

**Audit-completeness caveat.** The audit is a regex-based inventory — it can miss behavior encoded in ways that don't match our patterns (anonymous lambdas without a stat-name label, behavior driven by config-key access without a visible keyword, implicit behavior via shared mutable state). For this reason the audit is called an **inventory tool**, not an authoritative parity oracle. The authoritative parity oracle is Task 10's end-to-end game-level equality test (20 seeds × depth 2, exact move sequences must match). The inventory tool scopes the port but does not guarantee completeness; Phase C catches residual drift.

### 3.2 Deterministic mode contract

Both sides use `deterministic_mode=1` to:
- Disable temperature sampling → always pick argmax
- Use lexicographic tie-break on `(row, col)` when scores are equal
- Use the same starting player rule: seed even → black starts, seed odd → red

The tie-break rule is where subtle drift shows up — if JS iterates `for row in 0..N, for col in 0..N` but Python iterates differently, ties break differently. Task 2 of the plan pins down the iteration order in a unit test.

### 3.3 Scoring primitive alignment

The search flow in both implementations is:
1. Generate candidate moves
2. Score each candidate via `evaluateMove` / `evaluate_move`
3. Optionally: adjust score via position-level features (`friendlyConnections`, `opponentConnections`, etc.)
4. Optionally: adjust score via edge-specific features (`firstEdgeTouch`, `edgeFinishAdvance`, etc.)
5. Optionally: adjust score via connector/frontier features
6. Pick argmax (deterministic mode) or sample (training mode)

Task 3+ of the plan tackles each adjustment block independently. Each block has one task with the following mandatory structure:

1. **Read the JS source verbatim** — record the exact line range and formula
2. **Write a parity-delta unit test** using one of these test shapes (**no inequality-only or source-introspection tests**):
   - **Exact numeric delta (preferred):** given a fixed input position, call the JS oracle via `tests/js_oracle/heuristics_oracle.js` to get the JS-side score; call the Python implementation; assert `abs(py - js) < 0.01`.
   - **Exact feature firing condition:** given a fixed input position, assert the feature fires iff JS fires it, AND that the emitted delta matches within 0.01.
   - **Exact move equality:** for choose_move-level features, assert Python picks the same `(row, col)` as JS for a fixed `(seed, depth, ply)`.
3. **Implement the Python port** — match the JS formula exactly; if any magnitude or branching logic is ambiguous, STOP and escalate rather than approximating.
4. **Verify the parity-delta test passes** — if it doesn't pass to `abs(py - js) < 0.01`, the port is wrong; fix before moving on.

**Approximations are not acceptable.** If the JS logic (e.g. sealed-lane DSU traversal, opponent-frontier propagation) is too complex to port in one task, **STOP and escalate** to the controller for a sub-plan. Don't ship a partial port "hoping Phase C catches it" — Phase C is a final gate, not a correctness crutch.

**Escalation criteria** — stop and report BLOCKED when any of:
- JS formula references a scoring constant not in `search.json` and not obvious from code
- JS depends on stateful side effects across moves (caches, closures) that aren't trivially portable
- JS branching logic has ≥3 levels of nested conditionals and no comment explaining intent
- Python equivalent (e.g. `sealed_lane.py`) has a signature mismatch that prevents exact mirroring

An escalation causes the controller to expand scope (possibly by adding a sub-task) rather than letting the port silently approximate.

### 3.4 Out of scope

- Changes to `scripts/GPU/selfplay/engine.py` beyond what's required to wire updated heuristics
- Changes to MCTS or NN training paths — this is the pure-heuristic AI only
- Changes to `assets/js/ai/search.js` or `heuristics.js` — JS is the reference; Python mirrors
- Performance optimization of either side
- New heuristic features not already in JS

## 4. Validation gate

**All of the following must pass before the heuristic-parity work is considered complete:**

1. `.venv/bin/python -m pytest tests/js_oracle/test_oracle.py::TestDeterministicGameParity -v` → all 6 tests pass
2. `.venv/bin/python -m pytest tests/test_behavioral_regression.py::test_js_move_within_python_topk -v` → passes
3. `.venv/bin/python -m pytest tests/test_behavioral_regression.py` (full file) → no regressions vs pre-port baseline
4. For 20 random seeds × depth=2: Python and JS produce identical move sequences
5. Per-feature parity tests (Phase B) all use **exact numeric-delta** or **exact move-equality** assertions against the JS oracle. No inequality-only tests (`score_edge > score_mid`) or source-introspection tests (`'lane_open' in source`) remain in the heuristic-parity test suite — the CI grep gate `grep -En "assert.*getsource|assert .* > score_|assert .* > .*_score\b" tests/test_heuristic_parity.py` must return zero matches. (Plan Task 11 Step 1 runs this exact command.)

### 4.1 Parity-delta test shape

Every Phase B per-feature test MUST match this pattern:

```python
def test_feature_X_parity_for_fixture_Y():
    """JS and Python must agree on feature X's score contribution for fixture Y."""
    # 1. Load the known-good fixture (existing recorded state OR deterministic fresh-state)
    state = load_fixture("fixture_Y.json")  # see §5.4 fixtures

    # 2. Get JS score contribution for this feature at (row, col)
    js_score = call_js_heuristics_oracle(state, move=(row, col), player=player)

    # 3. Get Python score contribution
    py_score = evaluate_move(state, row, col, player)

    # 4. Assert exact parity (within float tolerance)
    assert abs(py_score - js_score) < 0.01, (
        f"Feature X parity broken: JS={js_score:.4f}, Python={py_score:.4f}, "
        f"diff={abs(py_score - js_score):.4f}"
    )
```

The JS oracle at `tests/js_oracle/heuristics_oracle.js` exposes `evaluateMove(state, move, player)` and returns the full score. Call it via subprocess as other oracle tests do (`tests/js_oracle/test_oracle.py::JSOracle.evaluate_move`).

## 5. File layout

### 5.1 Modified files

```
scripts/GPU/ai/
  heuristics.py          # add missing feature scoring
  search.py              # add missing adjustment blocks
```

### 5.2 New files

```
scripts/GPU/ai/
  heuristic_parity_audit.py      # Task 1 inventory tool (grep + normalize + CSV)
  capture_parity_fixture.py      # Task 2 fixture-capture CLI (seeded play + --moves-file fallback)
tests/
  test_heuristic_parity.py       # per-feature score-delta unit tests
  fixtures/heuristic_parity/
    README.md                    # how the fixtures were captured
    opening_empty_red.json       # fresh state, red-to-move
    opening_empty_black.json     # fresh state, black-to-move
    mid_game_seed0_ply10.json    # recorded from deterministic play (seed=0, ply=10)
    mid_game_seed1_ply15.json    # recorded from deterministic play (seed=1, ply=15)
    near_win_red.json            # red has chain top→nearly-bottom
    near_win_black.json          # symmetric
    sealed_lane.json             # opponent has sealed one finish lane
```

### 5.3 Untouched

JS files (`assets/js/ai/search.js`, `heuristics.js`) are the reference — do not modify.
Config file (`assets/js/ai/search.json`) is already consumed by both sides; do not modify.

### 5.4 Fixture capture

Fixtures under `tests/fixtures/heuristic_parity/` are committed JSON snapshots of known-good game states, captured via:

```bash
.venv/bin/python scripts/GPU/ai/capture_parity_fixture.py \
    --seed 0 --ply 10 --out tests/fixtures/heuristic_parity/mid_game_seed0_ply10.json
```

where `capture_parity_fixture.py` replays a deterministic game with `deterministic_mode=1` up to `--ply` and dumps the state.

**No inline inventing of fixtures inside test bodies.** Every Phase B test MUST load its fixture from `tests/fixtures/heuristic_parity/*.json`. If a needed fixture doesn't exist, add a task to capture it via the capture script.

**Fallback when seeded play doesn't yield the right topology.** The capture script also accepts `--moves-file <path>` to apply an explicit move list verbatim. The move list MUST come from a committed artifact (existing game log under `scripts/GPU/logs/games/`, oracle trace under `tests/js_oracle/`, or output of `scripts/trace_training_game.py`) — **never handwritten**. The captured fixture's `note` field must describe the move source for traceability.

**Fixture JSON schema (produced by capture_parity_fixture.py):**

```json
{
  "seed": 0,                              // omitted under --moves-file path
  "depth": 2,                             // omitted under --moves-file path
  "source": "moves-file:/tmp/moves.json", // present under --moves-file path
  "board_size": 24,
  "start_player": "black",
  "to_move": "red",
  "ply": 10,
  "move_history": [[r, c], ...],
  "expected_pegs": 10,                    // integrity field — checked on replay
  "note": "human-readable description, MUST include move source if --moves-file"
}
```

`test_all_fixtures_replay_to_valid_state` asserts that replaying `move_history` from a fresh GameState produces exactly `expected_pegs` pegs and matches `to_move`. If any fixture fails replay after a rule refactor, the test surfaces the drift rather than letting it silently invalidate downstream parity tests.

## 6. Rollout risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| JS uses a scoring constant that's not in `search.json` | High | Task 1 audit identifies these. Port as inline constant, annotate source line in JS. |
| JS has a hidden state (caches, side effects) that drives scoring | Medium | Task 1 audit scans for `this.cache`, `globalThis.*`, stateful closures. Surface for manual review before port. |
| Per-feature unit tests pass but game-level test still diverges | Medium | Gate 4 (20 seeds × depth 2) catches integration-level drift even if per-feature tests pass. |
| Port introduces Python-side regressions in existing paths | Low | Run full `test_training.py`, `test_self_play.py`, `test_heuristics.py` after each task. |
| JS scoring depends on floating-point determinism that Python can't match | Low | Tolerance is ≤0.01 absolute error per feature block; game-level comparisons use lexicographic tie-break which absorbs small FP drift. |

## 7. Success criteria

The port is successful iff all of:

1. Parity audit CSV (Task 1) shows no unresolved `in_js=True, in_py=False` rows
2. Validation gate Section 4 passes
3. No new failures in `test_training.py`, `test_self_play.py`, `test_heuristics.py`, `test_connectivity_channels.py`, `test_connectivity_masks.py`
4. Total Python suite is green (except for unrelated pre-existing flakes, which must be documented)

## 8. Next step

Invoke `superpowers:writing-plans` to produce the phased implementation plan from this spec.
