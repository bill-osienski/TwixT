# AlphaZero Implementation Phases

## Overview

The full AlphaZero implementation is broken into 9 phases (Phase 0–8). Each phase has:
- **Goal**: What we're building
- **Deliverables**: Concrete files/tests
- **Gate**: What must pass before moving on
- **Estimated scope**: Rough size

Phases are sequential — each depends on the previous gate passing.

---

## Phase 0: Game Rules Parity (BLOCKER)

**Goal**: Python and Node.js game logic produce identical results for any position.

**Why first**: Everything else depends on rules being correct and matching. Training on wrong rules = useless model.

**Deliverables**:
1. `scripts/GPU/alphazero/game/twixt_state.py`
   - `TwixtState` class with full rules
   - `legal_moves()` — all valid placements
   - `apply_move()` — place peg, add links
   - `is_terminal()` — win or draw detection
   - `winner()` — path finding (BFS/union-find)
   - `_crosses_link()` — segment intersection
   - Constants: `BOARD_SIZE=24`, `MAX_PLIES=200`
   - **Draw semantics**: Draw = terminal with no winner, caused by:
     - (a) no legal moves (board fills), OR
     - (b) `ply >= MAX_PLIES` (forced draw, even if moves exist)
     - Python + Node must implement identical logic

2. `server/gameLogic.js`
   - Mirror implementation of above
   - Identical method signatures

3. `tests/test_game_rules_parity.py`
   - Generate 100+ random game sequences
   - Compare Python vs Node.js at each step:
     - Same legal moves (sorted)
     - Same terminal detection
     - Same winner
   - **Draw-specific tests**:
     - Test forced draw at `MAX_PLIES` (same termination)
     - Test no-legal-moves draw (if reachable)

**Gate**: Parity test passes 100% on random games. Draw semantics match.

**Scope**: ~400 lines Python, ~400 lines JS, ~100 lines tests

---

## Phase 1: Board Encoding Parity (BLOCKER)

**Goal**: 24-channel tensor encoding implemented identically in Python and Node.js.

**Why second**: Network input must match between training and inference.

**Deliverables**:
1. Add `to_tensor()` to Python `TwixtState`
   - Returns `(24, 24, 24)` numpy array
   - All 24 channels per spec

2. Add `toTensor()` to Node.js `TwixtState`
   - Returns nested array `[24][24][24]`
   - Identical channel layout

3. `tests/test_encoding_parity.py`
   - Generate 50+ positions (early/mid/late game)
   - Compare Python vs Node.js tensor output
   - Max diff < 1e-9

**Gate**: Encoding parity test passes on all test positions.

**Scope**: ~150 lines Python, ~150 lines JS, ~50 lines tests

---

## Phase 2: Network Architecture

**Goal**: MLX neural network that compiles and runs forward pass.

**Deliverables**:
1. `scripts/GPU/alphazero/network.py`
   - `BoardEncoder` (CNN + ResBlocks)
   - `PolicyHead` (gather-based, outputs N logits)
   - `ValueHead` (outputs scalar)
   - `AlphaZeroNetwork` (combined)
   - `create_network()` factory

2. `tests/test_network.py`
   - Forward pass on random board
   - Output shapes correct
   - Gradients flow (no NaN)

**Gate**: Network forward pass works, shapes correct, gradients flow.

**Scope**: ~200 lines network, ~50 lines tests

---

## Phase 3: MCTS Implementation

**Goal**: Working MCTS that uses network for evaluation.

**Deliverables**:
1. `scripts/GPU/alphazero/mcts.py`
   - `MCTSConfig` dataclass
   - `MCTSNode` with priors, visits, value
   - `MCTS` class:
     - `search()` — run N simulations
     - `_expand()` — NN eval, create children
     - `_select_child()` — PUCT formula
     - `_backup()` — propagate values
     - `_add_dirichlet_noise()` — root exploration
     - `select_move()` — temperature sampling

2. `tests/test_mcts.py`
   - MCTS on simple position
   - Visit counts increase with simulations
   - Best move changes with more search
   - Dirichlet noise affects root priors

3. **Critical convention tests** (from main plan):
   - **Leaf eval rule**: `_expand()` always calls NN, never returns cached `qValue`
   - **Single NN eval**: Assert NN called exactly once per node expansion (mock/count)
   - **Terminal value**: Test position where opponent just won → value = -1 for `to_move`
   - **Terminal value (draw)**: Test `isTerminal() && winner()==None` → value = 0
   - **Backup sign flip**: Value alternates sign up the path (check manually on 3-node path)

**Gate**: MCTS produces sensible visit distributions. All convention tests pass.

**Scope**: ~250 lines MCTS, ~80 lines tests

---

## Phase 4: Self-Play

**Goal**: Generate complete games with position records for training.

**Deliverables**:
1. `scripts/GPU/alphazero/self_play.py`
   - `PositionRecord` dataclass
   - `GameRecord` dataclass
   - `play_game()` — full game with MCTS at each move

2. `tests/test_self_play.py`
   - Generate 5 games
   - Games terminate (win or draw)
   - Position records have correct fields
   - Outcomes assigned correctly

3. `scripts/GPU/alphazero/generate_games.py`
   - CLI to generate N games
   - Save to JSON for inspection

4. **(Optional) Enhanced replay data** for viewer:
   - Per-move: root value estimate
   - Per-move: top-K alternative moves with visit counts
   - Compatible with existing replay viewer format
   - (Can defer to later phase if not needed for training)

**Gate**: Can generate 10 complete games, all terminate properly, records look correct.

**Scope**: ~150 lines self-play, ~50 lines tests, ~50 lines CLI (+50 if enhanced replay)

---

## Phase 5: Training Loop

**Goal**: End-to-end training that reduces loss.

**Deliverables**:
1. `scripts/GPU/alphazero/trainer.py`
   - `ReplayBuffer` class
   - `alphazero_loss()` — policy CE + value MSE + L2
   - `train_step()` — single gradient update
   - `train()` — full orchestrator

2. `tests/test_training.py`
   - Loss computes without error
   - Loss decreases over 10 steps on fixed batch
   - Checkpoints save/load correctly

3. `scripts/GPU/alphazero/train.py`
   - CLI entry point
   - Configurable hyperparameters

**Gate**: Loss decreases over 100 training steps. Checkpoint saves and resumes.

**Scope**: ~300 lines trainer, ~80 lines tests, ~50 lines CLI

---

## Phase 6: ONNX Export

**Goal**: Export trained model to ONNX with verified parity.

**Deliverables**:
1. `scripts/GPU/alphazero/export_onnx.py`
   - `OnnxAlphaZero` PyTorch model
   - `PARAM_MAP` weight mapping
   - `convert_weights()` MLX → PyTorch
   - `export_to_onnx()` main function

2. **ONNX interface contract** (critical for Node.js inference):
   - Input `board`: `(1, 24, 24, 24)` — 24 channels
   - Input `move_rows`: `(512,)` — padded to fixed size
   - Input `move_cols`: `(512,)` — padded to fixed size
   - Input `move_mask`: `(512,)` — 1.0 valid, 0.0 padding
   - Output `policy_logits`: `(512,)` — invalid positions masked to `-1e9` (not `-inf`)
   - Output `value`: scalar in `[-1, 1]`

3. `scripts/GPU/alphazero/verify_export.py`
   - Load MLX and ONNX models
   - Compare outputs on 10+ test boards (varying move counts)
   - Assert max diff < 1e-4
   - Verify masked logits are exactly `-1e9`

4. CLI: `python -m scripts.GPU.alphazero.export_onnx --checkpoint X --output model.onnx`

**Gate**: ONNX model produces identical outputs to MLX model (< 1e-4 diff). Interface contract verified.

**Scope**: ~250 lines export, ~100 lines verify

---

## Phase 7: Node.js Server

**Goal**: Express server that runs MCTS with ONNX model.

**Deliverables**:
1. `server/inference.js` — ONNX wrapper
2. `server/mcts.js` — Node MCTS (mirrors Python)
3. `server/cache.js` — LRU position cache (Uint8Array, FNV-1a, no spread operators)
   - Moves keyed as `"r,c"` strings (no array/object keys)
   - Cache key = board hash + order-independent move-set hash (sort moves before hashing)
4. `server/index.js` — Express API (`/api/move`, `/api/evaluate`, `/api/health`)

5. **Deterministic mode implementation** (required for parity testing):
   - Disable Dirichlet noise at root
   - Temperature = 0 move selection
   - Fixed RNG seed (same in Python + Node)
   - Stable tie-break: lexicographic `(row, col)` ordering
   - Flag: `deterministicMode: true` in request or config

6. `server/test_server.js`
   - `/api/health` returns `{ status: 'ok', modelLoaded: true }`
   - `/api/move` returns valid move with value
   - `/api/evaluate` returns value in [-1, 1]
   - Deterministic mode: same position → same move every time

7. Parity test (`tests/test_node_python_parity.js` or `.py`):
   - Run 10 positions through Python MCTS (deterministic)
   - Run same 10 positions through Node MCTS (deterministic)
   - Assert identical move selection

8. **Performance gate**:
   - `difficulty=hard` (200 sims) completes in <500ms on target machine
   - No `Math.max(...arr)` or spread operators in `inference.js` / `mcts.js` hot paths
   - Cache hit rate logged (should be >0 on repeated positions)

**Gate**: Server responds correctly. Deterministic parity with Python MCTS. Hard difficulty <500ms.

**Scope**: ~500 lines server code, ~150 lines tests

---

## Phase 8: Frontend Integration

**Goal**: Browser can play against AlphaZero with win bar.

**Deliverables**:
1. `assets/js/ai/alphaZeroClient.js`
   - `getMove()` with timeout fallback
   - `evaluate()` for win bar

2. `assets/js/ui/winBar.js`
   - Visual win prediction bar
   - CSS styling

3. Integration in game controller
   - Wire up AI move requests
   - Update win bar after each move
   - Graceful fallback to heuristics

**Gate**: Can play full game against AlphaZero in browser. Win bar updates.

**Scope**: ~200 lines JS, ~50 lines CSS

---

## Phase Summary

| Phase | Name | Blocker? | Est. Lines | Dependencies |
|-------|------|----------|------------|--------------|
| 0 | Game Rules Parity | YES | ~900 | None |
| 1 | Encoding Parity | YES | ~350 | Phase 0 |
| 2 | Network Architecture | No | ~250 | Phase 1 |
| 3 | MCTS | No | ~330 | Phase 2 |
| 4 | Self-Play | No | ~250 | Phase 3 |
| 5 | Training Loop | No | ~430 | Phase 4 |
| 6 | ONNX Export | No | ~350 | Phase 5 |
| 7 | Node.js Server | No | ~650 | Phase 6, Phase 0 |
| 8 | Frontend | No | ~250 | Phase 7 |

**Total**: ~3,800 lines of new code + tests

---

## Recommended Order of Work

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6 ──► Phase 7 ──► Phase 8
   │           │                                                           │
   │           └── BLOCKER: Cannot train until encoding matches ───────────┘
   │
   └── BLOCKER: Cannot do anything until rules match
```

**Start with Phase 0**. It's the foundation everything else builds on.

---

## Working Agreement

For each phase:
1. **Implement** the deliverables
2. **Test** with the specified tests
3. **Pass the gate** before moving on
4. **Commit** with clear message: `"Phase N: <description>"`

If a gate fails, fix it before proceeding. Don't accumulate tech debt across phases.
