# Connected-Component Adjacency Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the O(V·E) full-bridge-scan in `_get_connected_component` with an O(V+E) lazily-built adjacency cache, applied identically to the Python `TwixtState` and JS `gameLogic.js` engines, with no change to observable behavior.

**Architecture:** Each state gains a private, lazily-built `peg → [neighbor pegs]` map. `_get_connected_component` builds it once from `bridges` on first use and BFS-walks it thereafter. `winner()`, `_check_win()`, `connectivity_masks()` are unchanged — they keep calling `_get_connected_component`, now cheap. Invalidation is null-on-copy (the cache is never carried into a copied state), so the only mutation path that matters — `apply_move`, which copies then mutates — automatically gets a fresh cache; an explicit `_invalidate_adj()` covers in-place test/tool mutations.

**Tech Stack:** Python 3 (dataclasses, `collections.deque`, pytest, numpy); Node.js ES modules (`server/gameLogic.js`, `node:assert`). Cross-engine parity is guarded by existing pytest tests that shell out to `node`.

**Spec:** `docs/superpowers/specs/2026-05-29-connected-component-adjacency-design.md`

---

## File Structure

- **Modify** `scripts/GPU/alphazero/game/twixt_state.py` — add `_adj` field, `_build_adjacency()`, `_invalidate_adj()`; rewrite `_get_connected_component`. Everything else (winner/_check_win/connectivity_masks/copy/apply_move/to_dict/from_dict/__hash__/__eq__) is untouched.
- **Modify** `server/gameLogic.js` — identical mirror: `this._adj = null` in the constructor, `_buildAdjacency()`, `_invalidateAdj()`; rewrite `_getConnectedComponent`.
- **Create** `tests/test_twixt_state_cc_adjacency.py` — Python equivalence-vs-legacy tests (synthetic dense games + real replays + fixtures), cache-behavior tests, and a perf-smoke guard.
- **Create** `tests/cc_adjacency.test.mjs` — Node script asserting JS cache behavior + equivalence-vs-legacy over seeded random games.
- **Verify green, no change** `tests/test_game_rules_parity.py`, `tests/test_js_py_tensor_parity.py`, `tests/test_connectivity_masks.py`.

---

## Task 1: Python equivalence guard (passes against current code)

This test compares the engine's `_get_connected_component` to a reference copy of the **current** full-scan algorithm. It passes now (proving the harness + reference are correct) and must keep passing after the refactor (proving no behavior change).

**Files:**
- Create: `tests/test_twixt_state_cc_adjacency.py`

- [ ] **Step 1: Write the equivalence test + helpers**

```python
"""Equivalence + cache-behavior tests for the _get_connected_component
adjacency optimization (spec 2026-05-29). The "legacy" reference below is a
verbatim copy of the pre-optimization full-bridge-scan algorithm; the engine's
output must match it exactly for every position in the corpus."""
import glob
import json
import os
import random
from collections import deque

import numpy as np
import pytest

from scripts.GPU.alphazero.game.twixt_state import TwixtState

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- reference: pre-optimization O(V*E) full-bridge-scan BFS ---------------
def _legacy_component(pegs, bridges, start, player):
    visited, component = set(), set()
    queue = deque([start])
    while queue:
        pos = queue.popleft()
        if pos in visited:
            continue
        if pegs.get(pos) != player:
            continue
        visited.add(pos)
        component.add(pos)
        for p1, p2 in bridges:
            if p1 == pos:
                nb = p2
            elif p2 == pos:
                nb = p1
            else:
                continue
            if pegs.get(p1) != player:  # mirrors original's check on p1
                continue
            if nb not in visited:
                queue.append(nb)
    return component


def _components_legacy(pegs, bridges, player):
    seen, comps = set(), set()
    for peg, p in pegs.items():
        if p != player or peg in seen:
            continue
        comp = _legacy_component(pegs, bridges, peg, player)
        comps.add(frozenset(comp))
        seen |= comp
    return comps


def _components_optimized(state, player):
    seen, comps = set(), set()
    for peg, p in state.pegs.items():
        if p != player or peg in seen:
            continue
        comp = state._get_connected_component(peg, player)
        comps.add(frozenset(comp))
        seen |= comp
    return comps


def _legacy_winner(state):
    active = state.active_size
    pegs, bridges = state.pegs, state.bridges
    for col in range(active):
        if pegs.get((0, col)) == "red":
            if any(r == active - 1 for (r, c) in _legacy_component(pegs, bridges, (0, col), "red")):
                return "red"
    for row in range(active):
        if pegs.get((row, 0)) == "black":
            if any(c == active - 1 for (r, c) in _legacy_component(pegs, bridges, (row, 0), "black")):
                return "black"
    return None


def _legacy_masks(state, player):
    active = state.active_size
    m_g1 = np.zeros((active, active), dtype=np.float32)
    m_g2 = np.zeros((active, active), dtype=np.float32)
    m_both = np.zeros((active, active), dtype=np.float32)
    if player == "red":
        on_g1, on_g2 = (lambda r, c: r == 0), (lambda r, c: r == active - 1)
    else:
        on_g1, on_g2 = (lambda r, c: c == 0), (lambda r, c: c == active - 1)
    for comp in _components_legacy(state.pegs, state.bridges, player):
        t1 = any(on_g1(r, c) for (r, c) in comp)
        t2 = any(on_g2(r, c) for (r, c) in comp)
        for (r, c) in comp:
            if t1:
                m_g1[r, c] = 1.0
            if t2:
                m_g2[r, c] = 1.0
            if t1 and t2:
                m_both[r, c] = 1.0
    return m_g1, m_g2, m_both


def _assert_position_equivalent(state):
    for player in ("red", "black"):
        assert _components_optimized(state, player) == _components_legacy(
            state.pegs, state.bridges, player
        ), f"component mismatch for {player} at ply {state.ply}"
        for opt, leg in zip(state.connectivity_masks(player), _legacy_masks(state, player)):
            assert np.array_equal(opt, leg), f"mask mismatch for {player} at ply {state.ply}"
    assert state.winner() == _legacy_winner(state), f"winner mismatch at ply {state.ply}"


def _random_game(seed, active_size=24, max_ply=160):
    """Play random legal moves, yielding the state after each move."""
    rng = random.Random(seed)
    state = TwixtState(active_size=active_size)
    for _ in range(max_ply):
        moves = state.legal_moves()
        if not moves:
            break
        state = state.apply_move(rng.choice(moves))
        yield state


def test_equivalence_synthetic_dense():
    for seed in (1, 2, 3):
        plies = list(_random_game(seed, active_size=24, max_ply=160))
        assert len(plies) >= 100, "synthetic game should reach a dense regime"
        # Check a sample of plies plus the final (densest) position.
        for state in plies[::20] + [plies[-1]]:
            _assert_position_equivalent(state)


def test_equivalence_fixtures():
    # Empty board.
    _assert_position_equivalent(TwixtState(active_size=8))

    # Single red peg (singleton component). These fixtures construct-then-query,
    # so the cache builds lazily AFTER the mutations and needs no explicit
    # invalidation (that path is covered by test_invalidate_adj_picks_up_mutation).
    s = TwixtState(active_size=8)
    s.pegs[(3, 3)] = "red"
    _assert_position_equivalent(s)

    # Orphan bridge (endpoint without a peg) is ignored.
    s = TwixtState(active_size=8)
    s.pegs[(3, 3)] = "red"
    s.bridges.add(((3, 3), (5, 4)))  # (5,4) has no peg
    _assert_position_equivalent(s)
    assert s._get_connected_component((3, 3), "red") == {(3, 3)}

    # Cross-player bridge is ignored by both players.
    s = TwixtState(active_size=8)
    s.pegs[(3, 3)] = "red"
    s.pegs[(5, 4)] = "black"
    s.bridges.add(((3, 3), (5, 4)))
    _assert_position_equivalent(s)


def test_equivalence_real_replays():
    files = sorted(glob.glob(os.path.join(REPO_ROOT, "Replays", "**", "*.json"), recursive=True))
    if not files:
        pytest.skip("no Replays/ corpus present")
    for path in files[:40]:  # bound runtime; log the cap
        with open(path) as f:
            rec = json.load(f)
        moves = [(m["row"], m["col"]) for m in rec.get("moves", [])]
        if not moves:
            continue
        active = int(rec.get("meta", {}).get("board_size", 24))
        state = TwixtState(active_size=active)
        for mv in moves:
            state = state.apply_move(mv)
        _assert_position_equivalent(state)  # final (densest) position
    print(f"[cc-adjacency] checked {min(len(files), 40)}/{len(files)} replays (capped at 40)")
```

- [ ] **Step 2: Run it against the current (unmodified) engine**

Run: `python -m pytest tests/test_twixt_state_cc_adjacency.py -v`
Expected: PASS (3 tests). This proves the legacy reference matches the in-tree algorithm and establishes the regression baseline.

- [ ] **Step 3: Commit**

```bash
git add tests/test_twixt_state_cc_adjacency.py
git commit -m "test(perf): equivalence guard for _get_connected_component

Reference copy of the current full-scan BFS + corpus (synthetic dense,
real replays, fixtures). Passes against the current engine; will guard the
adjacency refactor.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Python cache-behavior tests (RED)

**Files:**
- Modify: `tests/test_twixt_state_cc_adjacency.py` (append)

- [ ] **Step 1: Append the cache-behavior tests**

```python
def test_adj_is_none_on_fresh_state():
    s = TwixtState(active_size=8)
    assert s._adj is None


def test_adj_built_lazily_on_query():
    s = TwixtState(active_size=8)
    s.pegs[(3, 3)] = "red"
    assert s._adj is None  # setting a peg does not build the cache
    s._get_connected_component((3, 3), "red")
    assert isinstance(s._adj, dict)


def test_adj_not_carried_into_copy():
    s = TwixtState(active_size=8)
    s.pegs[(3, 3)] = "red"
    s._get_connected_component((3, 3), "red")  # build cache
    assert s._adj is not None
    child = s.copy()
    assert child._adj is None


def test_apply_move_child_has_fresh_cache():
    s = TwixtState(active_size=8)
    s._get_connected_component((0, 0), "red")  # build cache on parent
    child = s.apply_move((3, 3))  # red plays (3,3)
    assert child._adj is None
    assert child._get_connected_component((3, 3), "red") == {(3, 3)}


def test_invalidate_adj_picks_up_mutation():
    s = TwixtState(active_size=8)
    s.pegs[(3, 3)] = "red"
    s._get_connected_component((3, 3), "red")  # build cache (no bridges yet)
    s.pegs[(5, 4)] = "red"
    s.bridges.add(((3, 3), (5, 4)))
    s._invalidate_adj()
    assert s._get_connected_component((3, 3), "red") == {(3, 3), (5, 4)}
```

- [ ] **Step 2: Run them to verify they FAIL**

Run: `python -m pytest tests/test_twixt_state_cc_adjacency.py -k "adj or invalidate" -v`
Expected: FAIL — `AttributeError: 'TwixtState' object has no attribute '_adj'` (and no `_invalidate_adj`). Do not commit yet.

---

## Task 3: Python implementation (GREEN)

**Files:**
- Modify: `scripts/GPU/alphazero/game/twixt_state.py` (field after `max_plies_limit` ~line 148; helpers + rewrite of `_get_connected_component` ~lines 391-430)

- [ ] **Step 1: Add the `_adj` field**

After the `max_plies_limit` field, add:

```python
    max_plies_limit: Optional[int] = None  # if set, state becomes terminal at this ply
    # Derived, lazily-built adjacency cache (peg -> neighbor pegs) backing
    # _get_connected_component. Not a constructor arg; never copied (null on
    # copy -> rebuilt lazily). Excluded from equality/repr (derived state).
    _adj: Optional[Dict[Pos, List[Pos]]] = field(
        default=None, init=False, compare=False, repr=False
    )
```

(`field`, `Dict`, `List`, `Optional` are already imported at the top of the file.)

- [ ] **Step 2: Add the build + invalidate helpers**

Insert immediately before `_get_connected_component`:

```python
    def _build_adjacency(self) -> Dict[Pos, List[Pos]]:
        """Build a peg -> [bridge-connected neighbor pegs] map from self.bridges.

        One map per state (not per player): bridges are same-player by
        construction, and per-player traversal correctness is enforced by the
        pop-time color check in _get_connected_component.
        """
        adj: Dict[Pos, List[Pos]] = {}
        for p1, p2 in self.bridges:
            adj.setdefault(p1, []).append(p2)
            adj.setdefault(p2, []).append(p1)
        return adj

    def _invalidate_adj(self) -> None:
        """Drop the cached adjacency map.

        Call after any in-place mutation of self.bridges / self.pegs on an
        EXISTING state. Production mutates only via apply_move (which copies
        first, so the child's cache is already empty); this is a safety hook
        for tests and tools that mutate state in place.
        """
        self._adj = None
```

- [ ] **Step 3: Rewrite `_get_connected_component`**

Replace the entire current method body with:

```python
    def _get_connected_component(self, start: Pos, player: str) -> Set[Pos]:
        """Get all positions connected to start via same-player bridges.

        BFS over a lazily-built adjacency map (O(V+E) amortized per call). The
        map is shared by winner(), _check_win(), and connectivity_masks() so
        feature-side and game-logic-side connectivity can never drift.
        """
        if self._adj is None:
            self._adj = self._build_adjacency()
        adj = self._adj

        visited: Set[Pos] = set()
        component: Set[Pos] = set()
        queue = deque([start])

        while queue:
            pos = queue.popleft()
            if pos in visited:
                continue
            if self.pegs.get(pos) != player:
                continue

            visited.add(pos)
            component.add(pos)

            for npos in adj.get(pos, ()):
                if npos not in visited:
                    queue.append(npos)

        return component
```

- [ ] **Step 4: Run the cache-behavior tests — expect GREEN**

Run: `python -m pytest tests/test_twixt_state_cc_adjacency.py -k "adj or invalidate" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the equivalence tests — expect still GREEN**

Run: `python -m pytest tests/test_twixt_state_cc_adjacency.py -v`
Expected: PASS (all).

- [ ] **Step 6: Run the existing connectivity/parity Python tests — no regression**

Run: `python -m pytest tests/test_connectivity_masks.py tests/test_connectivity_channels.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/game/twixt_state.py tests/test_twixt_state_cc_adjacency.py
git commit -m "perf(twixt_state): O(V+E) adjacency cache for _get_connected_component

Lazily build a per-state peg->neighbors map instead of scanning all bridges
per popped peg. winner()/_check_win()/connectivity_masks() unchanged. Cache
is null-on-copy; _invalidate_adj() covers in-place mutation. Output verified
identical to the legacy full-scan over synthetic-dense, replay, and fixture
corpora.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: JS cache + equivalence test (RED)

**Files:**
- Create: `tests/cc_adjacency.test.mjs`

- [ ] **Step 1: Write the Node test script**

```javascript
// Cache-behavior + equivalence checks for the JS _getConnectedComponent
// adjacency optimization (spec 2026-05-29). Run: node tests/cc_adjacency.test.mjs
import { TwixtState, BOARD_SIZE } from '../server/gameLogic.js';

// Reference: pre-optimization full-bridge-scan BFS (verbatim old algorithm).
function legacyComponent(state, start, player) {
  const visited = new Set();
  const component = new Set();
  const queue = [start];
  while (queue.length > 0) {
    const [row, col] = queue.shift();
    const key = `${row},${col}`;
    if (visited.has(key)) continue;
    if (state.getPeg(row, col) !== player) continue;
    visited.add(key);
    component.add(key);
    for (const bKey of state.bridges) {
      const [p1Str, p2Str] = bKey.split('-');
      const [r1, c1] = p1Str.split(',').map(Number);
      const [r2, c2] = p2Str.split(',').map(Number);
      let nr, nc;
      if (r1 === row && c1 === col) { nr = r2; nc = c2; }
      else if (r2 === row && c2 === col) { nr = r1; nc = c1; }
      else continue;
      if (state.getPeg(r1, c1) !== player) continue;
      const nKey = `${nr},${nc}`;
      if (!visited.has(nKey)) queue.push([nr, nc]);
    }
  }
  return component;
}

// Deterministic LCG (no deps) for reproducible random games.
function makeRng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

function playRandomGame(seed, activeSize, maxPly) {
  const rng = makeRng(seed);
  let state = new TwixtState({ activeSize });
  for (let i = 0; i < maxPly; i++) {
    const moves = state.legalMoves();
    if (moves.length === 0) break;
    state = state.applyMove(moves[Math.floor(rng() * moves.length)]);
  }
  return state;
}

function setsEqual(a, b) {
  if (a.size !== b.size) return false;
  for (const x of a) if (!b.has(x)) return false;
  return true;
}

let failures = 0;
function check(name, cond) {
  if (cond) console.log(`ok   - ${name}`);
  else { console.error(`FAIL - ${name}`); failures++; }
}

// 1. Lazy build
{
  const s = new TwixtState({ activeSize: 8 });
  check('fresh state _adj is null', s._adj === null);
  s.pegs.set('2,3', 'red');
  s._getConnectedComponent([2, 3], 'red');
  check('_adj is a Map after first query', s._adj instanceof Map);
}

// 2. Null on copy
{
  const s = new TwixtState({ activeSize: 8 });
  s.pegs.set('2,3', 'red');
  s._getConnectedComponent([2, 3], 'red');
  check('child _adj is null after copy', s.copy()._adj === null);
}

// 3. Invalidate picks up an in-place mutation (existence-guarded so the
//    script still runs every check pre-implementation, when _invalidateAdj
//    is absent; post-implementation the call always fires and is meaningful).
{
  const s = new TwixtState({ activeSize: 8 });
  s.pegs.set('2,3', 'red');
  s._getConnectedComponent([2, 3], 'red'); // build (no bridges yet)
  s.pegs.set('4,4', 'red');
  s.bridges.add('2,3-4,4');
  if (typeof s._invalidateAdj === 'function') s._invalidateAdj();
  check('rebuild after invalidate sees new bridge',
    s._getConnectedComponent([2, 3], 'red').has('4,4'));
}

// 4. Equivalence vs legacy over seeded random games
for (const seed of [1, 2, 3]) {
  const state = playRandomGame(seed, BOARD_SIZE, 120);
  let ok = true;
  for (const [key, p] of state.pegs) {
    const [r, c] = key.split(',').map(Number);
    const opt = state._getConnectedComponent([r, c], p);
    const leg = legacyComponent(state, [r, c], p);
    if (!setsEqual(opt, leg)) { ok = false; break; }
  }
  check(`equivalence vs legacy seed=${seed} (winner=${state.winner()})`, ok);
}

if (failures > 0) { console.error(`\n${failures} check(s) failed`); process.exit(1); }
console.log('\nAll cc-adjacency checks passed');
```

- [ ] **Step 2: Run it to verify it FAILS**

Run: `node tests/cc_adjacency.test.mjs`
Expected: FAIL (exit 1) — checks 1-2 fail because `_adj` is `undefined` (not `null`) on a fresh state and after `copy()`. Checks 3-4 pass against the current full-scan (it already matches legacy and needs no cache), but the two failures still force a non-zero exit. Do not commit yet.

---

## Task 5: JS implementation (GREEN) + cross-engine parity

**Files:**
- Modify: `server/gameLogic.js` (constructor ~line 163; helpers + rewrite of `_getConnectedComponent` ~lines 395-441)

- [ ] **Step 1: Initialize `_adj` in the constructor**

In the `constructor`, immediately after `this.ply = ply;`, add:

```javascript
    this.ply = ply;
    // Derived, lazily-built adjacency cache backing _getConnectedComponent.
    // Never copied (copy() runs the constructor -> null -> rebuilt lazily).
    // Mirrors Python TwixtState._adj.
    this._adj = null;
```

- [ ] **Step 2: Add the build + invalidate helpers**

Insert immediately before `_getConnectedComponent`:

```javascript
  /**
   * Build a "r,c" -> Array<[r,c]> adjacency map from this.bridges.
   * One map per state; per-player correctness is enforced by the pop-time
   * color check in _getConnectedComponent. Mirrors Python _build_adjacency.
   */
  _buildAdjacency() {
    const adj = new Map();
    for (const bKey of this.bridges) {
      const [p1Str, p2Str] = bKey.split('-');
      const [r1, c1] = p1Str.split(',').map(Number);
      const [r2, c2] = p2Str.split(',').map(Number);
      const k1 = `${r1},${c1}`;
      const k2 = `${r2},${c2}`;
      if (!adj.has(k1)) adj.set(k1, []);
      if (!adj.has(k2)) adj.set(k2, []);
      adj.get(k1).push([r2, c2]);
      adj.get(k2).push([r1, c1]);
    }
    return adj;
  }

  /**
   * Drop the cached adjacency map. Call after any in-place mutation of
   * this.bridges / this.pegs on an existing state. Mirrors Python
   * _invalidate_adj. Production mutates only via applyMove (fresh copy).
   */
  _invalidateAdj() {
    this._adj = null;
  }
```

- [ ] **Step 3: Rewrite `_getConnectedComponent`**

Replace the entire current method body with:

```javascript
  _getConnectedComponent(start, player) {
    if (this._adj === null) {
      this._adj = this._buildAdjacency();
    }
    const adj = this._adj;

    const visited = new Set();
    const component = new Set();
    const queue = [start];

    while (queue.length > 0) {
      const [row, col] = queue.shift();
      const key = `${row},${col}`;

      if (visited.has(key)) continue;
      if (this.getPeg(row, col) !== player) continue;

      visited.add(key);
      component.add(key);

      const neighbors = adj.get(key);
      if (neighbors === undefined) continue;
      for (const [nr, nc] of neighbors) {
        const nKey = `${nr},${nc}`;
        if (!visited.has(nKey)) {
          queue.push([nr, nc]);
        }
      }
    }

    return component;
  }
```

- [ ] **Step 4: Run the JS test — expect GREEN**

Run: `node tests/cc_adjacency.test.mjs`
Expected: "All cc-adjacency checks passed" (exit 0).

- [ ] **Step 5: Run the cross-engine parity tests — no regression**

Run: `python -m pytest tests/test_game_rules_parity.py tests/test_js_py_tensor_parity.py -v`
Expected: PASS — Python `winner()`/`is_terminal()` and the 30-channel tensor (incl. connectivity channels 24-29) still match the JS engine exactly.

- [ ] **Step 6: Lint the JS — no new errors**

Run: `npx eslint server/gameLogic.js`
Expected: no new errors beyond the two pre-existing `'c' is defined but never used` warnings at lines ~483-484 (unrelated to this change).

- [ ] **Step 7: Commit**

```bash
git add server/gameLogic.js tests/cc_adjacency.test.mjs
git commit -m "perf(gameLogic): mirror O(V+E) adjacency cache in JS engine

Same lazy peg->neighbors cache as Python TwixtState, applied to
_getConnectedComponent. Speeds up the Node-side MCTS (server/mcts.js) that
serves the live AI. Output verified identical to the legacy full-scan and to
the Python engine via the existing cross-engine parity tests.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Perf-smoke guard + full verification

**Files:**
- Modify: `tests/test_twixt_state_cc_adjacency.py` (append)

- [ ] **Step 1: Append a perf-smoke test**

```python
@pytest.mark.slow
def test_perf_smoke_dense_winner():
    """A dense position must resolve winner() quickly. With the old O(V*E)
    scan this loop would take many seconds; O(V+E) is well under the bound."""
    import time

    state = None
    for state in _random_game(7, active_size=24, max_ply=250):
        pass
    assert len(state.pegs) > 150, "expected a dense position"

    start = time.perf_counter()
    for _ in range(300):
        state._invalidate_adj()  # force rebuild + full traversal each call
        state.winner()
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0, f"dense winner() x300 took {elapsed:.2f}s (regression?)"
```

- [ ] **Step 2: Run the perf smoke**

Run: `python -m pytest tests/test_twixt_state_cc_adjacency.py::test_perf_smoke_dense_winner -v -m slow`
Expected: PASS (well under 3s on a normal machine).

- [ ] **Step 3: Run the full relevant suite**

Run: `python -m pytest tests/test_twixt_state_cc_adjacency.py tests/test_connectivity_masks.py tests/test_connectivity_channels.py tests/test_game_rules_parity.py tests/test_js_py_tensor_parity.py -v`
Expected: PASS (all). Then re-run the JS check: `node tests/cc_adjacency.test.mjs` → exit 0.

- [ ] **Step 4: Commit**

```bash
git add tests/test_twixt_state_cc_adjacency.py
git commit -m "test(perf): dense-position winner() perf-smoke guard

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Do not touch** `winner()`, `_check_win()`, `connectivity_masks()`, `copy()`, `apply_move()`, `to_dict`/`from_dict`, `__hash__`/`__eq__` in either engine. The whole point is that only neighbor *enumeration* changes.
- The `_adj` field is `init=False`, so no constructor call (including `copy()` and `from_dict`) needs to pass it; it defaults to `None` and `copy()` therefore yields a child with an empty cache automatically — that is the invalidation mechanism, not an oversight.
- `_invalidate_adj()` / `_invalidateAdj()` is only needed by code that mutates `bridges`/`pegs` on an already-queried state (tests, tools). Production self-play and MCTS go through `apply_move`/`applyMove`, which copy first.
- If `python -m pytest` cannot import `scripts...`, run from the repo root (there is a `conftest.py`/`pytest.ini` there); do not add `sys.path` hacks to the test.
- Real-replay coverage is capped at 40 files (logged) to bound test time; raise the cap locally if you want broader coverage. The lost worker-6 game was never saved, so the dense regime is covered by the synthetic ply-250 games.
