# TwixT GPU AutoTune - Implementation Plan

## Overview

The scaffold is already built in `/GPU_Plan/GPU/`. This plan focuses on filling in the stub modules to create a working system.

**Source scaffold**: `/Users/bill/Desktop/TwixT_Game/GPU_Plan/GPU/`
**Target location**: `/Users/bill/Desktop/TwixT_Game/scripts/GPU/`

---

## What's Already Complete

| Module | Status |
|--------|--------|
| `config/knobs.py` | ✅ All 12 knobs, HASH_FIELDS, CORE_BANDS |
| `tuning/hasher.py` | ✅ Stable config hashing |
| `tuning/state.py` | ✅ TuningState, load/save |
| `tuning/loop.py` | ✅ sweep_cycle, rank, validate |
| `replay/viewer.py` | ✅ Interactive ASCII replay |
| `utils/maybe_mlx.py` | ✅ MLX fallback |
| `cli.py` | ✅ All subcommands |
| `game/state.py` | ✅ GameState dataclass |

---

## Critical Design Decisions (Lock Before Coding)

### 1. Pie Rule (Swap Rule)
- [ ] **CONFIRM**: Does current JS engine include swap option after first move?
- [ ] Document decision: swap enabled or disabled
- [ ] Python must match JS behavior exactly

### 2. Determinism Invariants
Every game record MUST include:
- `seed` - RNG seed for reproducibility
- `starting_player` - who moves first
- `depth` - search depth used
- `config_hash` - which knobs were used
- `engine_version` - for debugging old replays
- `rules_variant` - "standard" / "swap" / etc.
- `feature_spec_version` - feature vector ordering/normalization
- `hash_spec_version` - which knobs are in hash

**RNG Discipline:**
- One RNG stream per game, seeded from `{global_seed, game_index}`
- No global RNG in shared code
- Move ordering ties: stable sort + seeded jitter only
- Identical minimax scores: deterministic tie-break (e.g., first in sorted order)
- Equal-best moves: seeded random choice from that game's RNG

### 3. Coordinate Conventions
- [ ] **VERIFY** vs JS: Are goal edges virtual or require pegs on boundary?
- [ ] **VERIFY** vs JS: Exact edge restriction rules
- [ ] **VERIFY** vs JS: Coordinate origin and display orientation
- [ ] Document: Red connects rows (top↔bottom), Black connects cols (left↔right)

### 4. Bridge Intersection Rules
`bridges_cross()` must handle:
- Proper intersection (middle of segments)
- Endpoint touching (legal - not a cross)
- Collinear overlap (shouldn't happen with knight moves)
- Use **integer/orientation tests**, not floats

### 5. Validation Gating + Success Definition ("Value-Ready")
**Exit criteria for declaring a config "value-ready":**
- Multiple **60/60** validation passes with **streak** (reset to 0 on any fail)
- Plus **macro 1200/1200 gate** before final declaration
- Streak threshold configurable (default: 2-4 consecutive passes)

### 6. Predicted-Bias Gate + Correlation/Data Hygiene
**Correlation model rules:**
- Rolling window with **decay** (older samples weighted less)
- **Up-weight probes** (they're designed to explore)
- **Down-weight/ignore low-info samples** (draw-heavy, near-zero bias)
- When R² is low, correlation is "untrustworthy" → relax gate

**Gate behavior:**
- If `predicted_bias > threshold` AND R² is good → reject candidate
- If gate becomes too strict → **fall back to core-clamped candidates** (don't stall)

### 7. Bucket Sampling Guarantees (Fixed Probes + Quotas)
**Explore bucket MUST include fixed probes each cycle:**
- Edge probes: `firstEdgeRed/Black ±5`
- Span probes: `blackSpanGainMultiplier ±0.05`
- Coverage probes: `redDoubleCoverageBonus`, `blackDoubleCoverageScale` at extremes

**Category weights MUST reserve slots for these probes** - they always appear.

### 8. Status Invariants (Durable State)
- **Never re-queue RETIRED or STABLE** hashes
- **Preserve streaks** across restarts (loaded from state file)
- **Reset streak to 0** on validation fail
- **Retire hashes that repeatedly fail** (e.g., 3+ validation failures)
- **Statuses are case-insensitive** (`"stable"` == `"STABLE"`)

### 9. Draw + Termination Definition (Must Match JS)
**VERIFY vs JS - what triggers a draw?**
- [ ] Board full / no legal moves?
- [ ] Move limit (ply cap)? If so, what number?
- [ ] Repetition detection?
- [ ] "No progress" / stall rule? (e.g., N consecutive non-extending moves)

**WHY this matters**: You use "drop low-info samples (draw-heavy...)" for correlation.
If Python's draw policy differs even slightly, you get different bias metrics.

### 10. Log + Replay Storage Policy (Prevent Disk Explosion)
Full auditable replay (top-N candidates + feature_summary every ply) can explode disk.

**Policy:**
- **Sweep games**: "thin" format (move + score only, no candidates)
- **Validation games**: full audit format (top-N candidates, features, decision reason)
- **Anomalies**: full audit format (games > 200 moves, unusual outcomes)
- **Selected hashes**: full audit on demand (`--full-audit` flag)

This balances debuggability with disk usage.

---

## Implementation Tasks (Fill in Stubs)

### Phase 0: JS Oracle Alignment (Before Everything Else)

**WHY**: Most port pain is one edge restriction, one corner rule, one coordinate flip.

**Deliverables:**
- [ ] Confirm Pie/Swap rule (on/off, tournament vs casual)
- [ ] Confirm coordinate convention (row/col, origin, display orientation)
- [ ] Confirm win detection semantics (virtual edges vs must-touch pegs)
- [ ] Confirm bridge crossing semantics (endpoint-touching allowed)

**File: `tests/js_oracle.py`** (new)
- [ ] Create JS-vs-Python comparison runner
- [ ] Check: legal moves set
- [ ] Check: apply move validity
- [ ] Check: bridges created + bridges blocked by crossings
- [ ] Check: winner detection

**Acceptance Criteria**: 0 mismatches vs JS across 10k random positions

### Phase 1: Game Rules (Port from twixtGame.js)

**File: `game/board.py`**
- [ ] Add TwixT edge restrictions:
  - Red cannot place on cols 0 or 23
  - Black cannot place on rows 0 or 23
  - No corners (0,0), (0,23), (23,0), (23,23)
- [ ] Fix `legal_moves()` to respect restrictions
- [ ] **VERIFY** these match JS exactly

**File: `game/bridge.py`**
- [ ] Implement `bridges_cross()` using integer orientation tests
- [ ] Handle endpoint-touching correctly (NOT a cross)
- [ ] Update `add_bridges_for_new_peg()` to check crossings

**File: `game/rules.py`**
- [ ] Implement `apply_move()` (place peg + create bridges)
- [ ] Implement `check_winner()` using BFS
  - Red wins: connected path row 0 → row 23
  - Black wins: connected path col 0 → col 23
- [ ] Add `is_game_over()` helper

**Acceptance Criteria**: 0 mismatches vs JS for legal moves + win detection on 10k positions

### Phase 1.5: Random/Greedy Self-Play (Bug Finder)

**WHY**: Find rule bugs FAST by running 1000 dumb games before investing in full search.

**File: `selfplay/random_policy.py`** (new)
- [ ] Implement `random_move()` - pick uniformly from legal moves
- [ ] Implement `greedy_move()` - simple heuristic (center bias, edge proximity)

**File: `selfplay/engine.py`**
- [ ] Add `play_random_game(seed)` - two random players
- [ ] Add `play_greedy_game(seed)` - two greedy players

**CLI: `python -m scripts.GPU.cli fuzz`**
- [ ] Run N random/greedy games
- [ ] Save all game records
- [ ] Report anomalies (games > 200 moves, weird winners, etc.)
- [ ] Replay suspicious games for debugging

**Acceptance Criteria**: 1000 random games, 0 illegal moves, no >200-move anomalies

### Phase 2: Heuristics (Port from heuristics.js)

**File: `ai/heuristics.py`**
- [ ] Port 28+ features:
  - Connection features (friendlyConnection, opponentConnection)
  - Distance features (friendlyDistance, opponentDistance, goalDistance)
  - Edge features (spanGain, edgeGapReduction, firstEdgeTouch)
  - Frontier/connector analysis
  - Component metrics
- [ ] Add `extract_features(state, move)` function
- [ ] Add `score_moves()` for batch scoring

**File: `ai/move_ordering.py`**
- [ ] Implement priority scoring from features
- [ ] Add top-N filtering

**Acceptance Criteria**: Feature values match JS within epsilon for 100 test positions

### Phase 3: Search + Value Model

**File: `ai/search.py`**
- [ ] Implement `minimax()` with alpha-beta pruning
- [ ] Implement `get_best_move()` with move ordering
- [ ] Support depth 2-4

**File: `ai/value_model.py`**
- [ ] Load `value-model.json` format
- [ ] Implement `evaluate()` with feature vector
- [ ] MLX batch inference path

**Acceptance Criteria**: AI plays reasonable games, value model output matches JS

### Phase 4: Self-Play Engine

**File: `selfplay/engine.py`**
- [ ] Implement `TwixtSimulator` class:
  - `play_game(knobs, depth, seed)` → GameRecord
  - Apply search with given knobs
  - Record moves + heuristics

**File: `selfplay/parallel.py`**
- [ ] Implement `run_games()`:
  - Run N games with given config
  - Write results to JSONL
  - Save game records to `games/<hash>/`
  - Return GameSummary

**Acceptance Criteria**: N games run, all replays load, summaries match replay outcomes

### Phase 5: Integration & Testing

- [ ] Copy scaffold from `GPU_Plan/GPU/` to `scripts/GPU/`
- [ ] Run `python -m scripts.GPU.cli init`
- [ ] Test with: `python -m scripts.GPU.cli sweep --games 2 --total 4`
- [ ] Verify replay: `python -m scripts.GPU.cli replay <game.json>`
- [ ] Run full tune cycle: `python -m scripts.GPU.cli tune --cycles 1`

---

## Key Files to Reference

| Purpose | File |
|---------|------|
| Game rules | `assets/js/game/twixtGame.js` |
| Bridge crossing | `assets/js/game/twixtGame.js:bridgesCross()` |
| 28+ features | `assets/js/ai/heuristics.js` |
| Minimax search | `assets/js/ai/search.js` |
| Value model | `assets/js/ai/valueModel.js` |
| Config format | `assets/js/ai/search.json` |

---

## Build Order

0. **JS Oracle Alignment** → 0 mismatches on 10k positions
1. **Game Rules** → legal moves, bridges, win detection match JS
1.5. **Random/Greedy Fuzz** → 1000 games, 0 anomalies
2. **Heuristics** → features match JS within epsilon
3. **Search + Value Model** → AI plays reasonably
4. **Self-Play Engine** → replays load, summaries match
5. **Integration** → full tune cycle works
6. **Optimization** → MLX performance tuning

Each phase has explicit acceptance criteria.

---

## Auditable Replay Format (Not Just Viewable)

For debugging "why did it do that?", each move record includes:

```python
{
    "turn": 5,
    "player": "red",
    "row": 11, "col": 8,
    "bridges_created": [[[11,8], [9,7]]],

    # Auditing fields (for debugging):
    "candidates": [                    # Top-N moves considered
        {"row": 11, "col": 8, "score": 1250.5},
        {"row": 10, "col": 6, "score": 1180.2},
        ...
    ],
    "feature_summary": {               # Key features for chosen move
        "spanGain": 180,
        "firstEdgeTouch": 420,
        ...
    },
    "search_score": 1250.5,
    "decision_reason": "best_minimax"  # or "fallback", "random_tiebreak"
}
```

This turns replay from "what did it do?" into "why did it do that?"

---

## Game Logic Verification

### 1. Interactive Play Mode (New CLI Command)

Add `python -m scripts.GPU.cli play` for human vs AI testing:
- ASCII board display after each move
- Human enters moves as `row,col` (e.g., `11,5`)
- Shows legal moves, bridges created, win detection
- Validates rules are working correctly

### 2. Comparison Test Suite

Add `python -m scripts.GPU.cli test-rules` that:
- Loads test positions from JSON
- Compares Python vs JS for:
  - Legal moves count and positions
  - Bridge crossing detection
  - Win detection
  - Move validation (edge restrictions)
- Reports any mismatches

### 3. Debug Replay Mode

Enhance replay viewer with `--debug` flag:
- Show legal moves at each turn
- Show bridge crossing checks
- Verify win condition at game end
- Flag any rule violations

### 4. Golden Test Positions

Create `tests/golden_positions.json` with:
- Known board states
- Expected legal moves
- Expected bridges
- Expected winner (if any)

Run with: `python -m scripts.GPU.cli test-rules --golden tests/golden_positions.json`

---

## CLI Commands (Already Wired + New)

```bash
# Existing commands
python -m scripts.GPU.cli init                    # Create logs folder
python -m scripts.GPU.cli sweep --games 10        # Run sweep
python -m scripts.GPU.cli rank --top 8            # Rank candidates
python -m scripts.GPU.cli validate --games 60     # Run validation
python -m scripts.GPU.cli tune --cycles 3         # Full loop
python -m scripts.GPU.cli replay game.json        # View replay
python -m scripts.GPU.cli status                  # Show state

# New verification commands
python -m scripts.GPU.cli play                    # Interactive human vs AI
python -m scripts.GPU.cli play --ai-vs-ai        # Watch AI vs AI game
python -m scripts.GPU.cli test-rules             # Run rule comparison tests
python -m scripts.GPU.cli test-rules --verbose   # Detailed test output
python -m scripts.GPU.cli fuzz --games 1000      # Run random games to find bugs
```

---

## GPU Acceleration Scope (Keep Realistic)

**What goes on GPU (MLX):**
- Feature evaluation (batch all candidate moves)
- Value model inference (matmul)
- Ridge regression training
- Move scoring

**What stays on CPU:**
- Move generation
- Bridge legality checks
- BFS win detection
- Game state management

**Performance Keys:**
1. **Batch size is king** - design to evaluate many moves/positions per MLX call
2. **Minimize device transfers** - once weights are on GPU, keep features/scoring there for the hot loop
3. **Don't bounce CPU↔GPU every move** - batch at the ply level

The "big win" comes from batching leaf evals and scoring many moves at once.
Keep MLX behind narrow interface (exactly like `maybe_mlx.py`).

---

## Optional: JS Viewer Bridge

Since you already have a JS board renderer for human play, consider:
- Export replay format that JS can read
- Tiny HTML page that loads game JSON and renders visually
- "Trust but verify" - ASCII for speed, JS for visual confirmation

Not required for MVP, but high value for debugging edge cases.
