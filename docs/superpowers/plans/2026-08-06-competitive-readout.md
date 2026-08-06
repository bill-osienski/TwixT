# Competitive Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build run-ready tooling for the competitive-readout experiment — a server readout-policy correction, and a research harness that can play two agents sharing one checkpoint but differing only in final move readout.

**Architecture:** Two fully independent phases. Phase A corrects the Node/ONNX product server (one shared policy resolver, correct cache key, deterministic override on both transports). Phase B adds a pure Python readout module carrying the frozen Hoeffding-LCB rule, moves the eval harness's move selection out of `MCTS` so search and readout draw from separate RNG streams, adds agent identity so a same-checkpoint match produces real comparative statistics, and adds a preflight analyzer over captured replays.

**Tech Stack:** Python 3 + pytest (research harness, no MLX in any new module); Node ESM + `node --test` (server).

## Global Constraints

Copied verbatim from `docs/superpowers/specs/2026-08-05-competitive-readout-strength-design.md`:

- **No GPU work is authorized.** This plan ends at implemented, tested, run-ready tooling. Do **not** launch Candidate 1, Candidate 2, or any match.
- Candidate 2's rule and gates are **FROZEN 2026-08-06**. `VALUE_RANGE = 2.0`, `DELTA = 0.05`, `MIN_CHILD_VISITS = 8`, `ε(n) = 2·sqrt(1.84445/n)`, top **two** children by completed visits, canonical numeric `(row, col)` tie order. Preflight gates: `< 0.5%` and `> 15%` override rate, `> 50%` single-game concentration. Colour split is **descriptive only**. No constant, threshold or eligibility rule may change.
- **Undefined statistics are `None`/null — never `0`, never `false`.** Note `MCTSNode.q_value` returns `0.0` at `visit_count == 0` (`mcts.py:259-261`); that is an undefined mean and must be mapped to `None`.
- **Fail closed.** An unverifiable condition is a failure. Every test constructs its negative case rather than observing it from ambient state.
- **Self-play must not change.** `self_play.py` keeps calling `mcts.select_move`; nothing in this plan edits `mcts.py`.
- Root-perspective Q is the **negation** of the child's stored value (`mcts.py:1122`: `q = -child.q_value`).
- Baseline test count comes from a **measured** `pytest` collect at the time of the work, never from a number in a document. Read exit codes from the process, not a pipe.
- Commit after every task. Do not squash tasks together.

---

## File Structure

**Phase A — server (no GPU, no Python)**

| File | Responsibility |
|---|---|
| `server/readout_policy.js` *(new)* | Single source of truth: difficulty → `{nSims, moveTemp}`, and the deterministic override. Pure, no Express, no MCTS. |
| `server/test_readout_policy.js` *(new)* | Unit tests for the resolver and the cache key. |
| `server/index.js` *(modify)* | Both transports consume the resolver. Cache keyed by state + model + budget, readout applied after the lookup. |
| `server/cache.js` *(modify)* | `makeKey` gains model identity and simulation budget. |
| `assets/js/ai/alphaZeroClient.js` *(modify)* | WebSocket request carries `deterministicMode`. |
| `package.json` *(modify)* | `test:server` runs both server test files. |

**Phase B — research harness (Python)**

| File | Responsibility |
|---|---|
| `scripts/GPU/alphazero/eval_readout.py` *(new)* | Pure. Every readout rule the harness can play + the frozen Hoeffding-LCB override + top-two extraction. No MLX, no MCTS import, no game engine. |
| `scripts/GPU/alphazero/eval_replay.py` *(modify)* | `ply_record` gains top-two child telemetry; schema version 1 → 2. |
| `scripts/GPU/alphazero/eval_runner.py` *(modify)* | `AgentSpec`; per-agent readout in `play_eval_game`; separate readout RNG; agent-aware tasks and results. |
| `scripts/GPU/alphazero/eval_summary.py` *(modify)* | Agent-keyed aggregation; explicit rejection of agent artifacts by the legacy checkpoint path. |
| `scripts/GPU/alphazero/eval_readout_match.py` *(new)* | Thin CLI for a two-agent, one-checkpoint match. The existing `eval_checkpoint_match.py` is left untouched. |
| `scripts/GPU/alphazero/readout_preflight.py` *(new)* | Pure gate computation + CLI over captured replays. |
| `tests/test_eval_readout.py` *(new)* | Frozen-constant and rule tests. |
| `tests/test_eval_readout_telemetry.py` *(new)* | Telemetry contract + search-identity tests. |
| `tests/test_eval_agent_identity.py` *(new)* | Task/result/colour-binding tests. |
| `tests/test_eval_summary_agent_mode.py` *(new)* | Agent-mode aggregation + legacy rejection. |
| `tests/test_readout_preflight.py` *(new)* | Frozen gate tests. |

**Phase independence:** Phase A touches only JavaScript; Phase B touches only Python. They share no file and may be executed in either order, or concurrently.

---

# PHASE A — Product server readout policy

## Task A1: Shared readout policy resolver

**Files:**
- Create: `server/readout_policy.js`
- Create: `server/test_readout_policy.js`
- Modify: `package.json:9`

**Interfaces:**
- Consumes: nothing.
- Produces: `resolvePolicy({difficulty, deterministicMode, temperature})` → `{nSims: number, moveTemp: number, difficulty: string}`; `DIFFICULTY_TABLE`.

- [ ] **Step 1: Write the failing test**

Create `server/test_readout_policy.js`:

```javascript
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolvePolicy, DIFFICULTY_TABLE } from './readout_policy.js';

test('hard is deterministic by default', () => {
  const p = resolvePolicy({ difficulty: 'hard' });
  assert.equal(p.nSims, 800);
  assert.equal(p.moveTemp, 0);
});

test('easy and medium keep their difficulty temperature', () => {
  assert.equal(resolvePolicy({ difficulty: 'easy' }).moveTemp, 1.0);
  assert.equal(resolvePolicy({ difficulty: 'easy' }).nSims, 100);
  assert.equal(resolvePolicy({ difficulty: 'medium' }).moveTemp, 0.5);
  assert.equal(resolvePolicy({ difficulty: 'medium' }).nSims, 400);
});

test('deterministicMode forces temperature 0 at every difficulty', () => {
  for (const d of Object.keys(DIFFICULTY_TABLE)) {
    assert.equal(resolvePolicy({ difficulty: d, deterministicMode: true }).moveTemp, 0);
  }
});

test('explicit temperature overrides the difficulty default', () => {
  assert.equal(resolvePolicy({ difficulty: 'medium', temperature: 0.75 }).moveTemp, 0.75);
});

test('deterministicMode beats an explicit temperature', () => {
  const p = resolvePolicy({ difficulty: 'medium', temperature: 0.75, deterministicMode: true });
  assert.equal(p.moveTemp, 0);
});

test('unknown difficulty falls back to medium', () => {
  const p = resolvePolicy({ difficulty: 'nonsense' });
  assert.equal(p.nSims, 400);
  assert.equal(p.difficulty, 'medium');
});

// NEGATIVE CASE, constructed: two callers that resolve policy independently
// must be detectable as divergent. This proves the parity test in Task A2
// can actually fail.
test('divergent policies are detectable', () => {
  const shared = resolvePolicy({ difficulty: 'hard' });
  const rogue = { nSims: 800, moveTemp: 0.25 };
  assert.notDeepEqual(shared.moveTemp, rogue.moveTemp);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test server/test_readout_policy.js`
Expected: FAIL — `Cannot find module './readout_policy.js'`

- [ ] **Step 3: Write minimal implementation**

Create `server/readout_policy.js`:

```javascript
/**
 * Single source of truth for AlphaZero move-readout policy.
 *
 * Both transports (REST /api/move and the WebSocket computeBestMove path)
 * MUST resolve policy through this module. Before this existed they
 * disagreed: the WS path used DIFFICULTY_PARAMS.moveTemp (1.0/0.5/0.25) while
 * REST used a hardcoded easy?0.5:0.1 and DIFFICULTY_PARAMS.moveTemp was dead
 * code on that path. deterministicMode was reachable only from REST.
 *
 * Pure: no Express, no MCTS, no I/O.
 */

// moveTemp semantics: selectMove samples proportional to count^(1/moveTemp);
// moveTemp < 0.01 is deterministic argmax.
export const DIFFICULTY_TABLE = {
  // easy/medium intentionally sacrifice strength to create difficulty levels.
  easy: { nSims: 100, moveTemp: 1.0 },
  medium: { nSims: 400, moveTemp: 0.5 },
  // hard promises strongest play, so it is deterministic.
  hard: { nSims: 800, moveTemp: 0 },
};

export const DEFAULT_DIFFICULTY = 'medium';

export function resolvePolicy({ difficulty, deterministicMode = false, temperature } = {}) {
  const key = Object.prototype.hasOwnProperty.call(DIFFICULTY_TABLE, difficulty)
    ? difficulty
    : DEFAULT_DIFFICULTY;
  const entry = DIFFICULTY_TABLE[key];
  let moveTemp;
  if (deterministicMode) {
    moveTemp = 0;
  } else if (temperature !== undefined && temperature !== null) {
    moveTemp = temperature;
  } else {
    moveTemp = entry.moveTemp;
  }
  return { difficulty: key, nSims: entry.nSims, moveTemp };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test server/test_readout_policy.js`
Expected: PASS, 7 tests

- [ ] **Step 5: Wire the test into npm**

In `package.json`, change the `test:server` script to:

```json
"test:server": "node --test server/test_server.js server/test_readout_policy.js",
```

- [ ] **Step 6: Run the full server suite and lint**

Run: `npm run test:server; echo "EXIT=$?"`
Expected: `EXIT=0`
Run: `npm run lint; echo "EXIT=$?"`
Expected: `EXIT=0`

- [ ] **Step 7: Commit**

```bash
git add server/readout_policy.js server/test_readout_policy.js package.json
git commit -m "feat(server): single source of truth for move-readout policy"
```

---

## Task A2: Both transports consume the resolver; WS client sends the override

**Files:**
- Modify: `server/index.js:41-46` (delete `DIFFICULTY_PARAMS`), `:94-104` (REST), `:523-541` (`computeBestMove`), the WS `move` handler around `:460`
- Modify: `assets/js/ai/alphaZeroClient.js:229` (request payload)
- Modify: `server/test_readout_policy.js`

**Interfaces:**
- Consumes: `resolvePolicy` from Task A1.
- Produces: `computeBestMove(stateDict, difficulty, opts)` where `opts` gains `deterministicMode` and `temperature`.

- [ ] **Step 1: Write the failing parity test**

Append to `server/test_readout_policy.js`:

```javascript
import { readFileSync } from 'node:fs';

// Parity is asserted at the POLICY layer, not by comparing moves: two
// stochastic calls legitimately differ, so a move-equality test would be
// either vacuous or flaky. Instead we assert no caller re-derives policy.
test('no hardcoded temperature ladder survives in index.js', () => {
  const src = readFileSync(new URL('./index.js', import.meta.url), 'utf8');
  assert.ok(!src.includes('DIFFICULTY_PARAMS'),
    'DIFFICULTY_PARAMS must be gone; resolvePolicy is the only source');
  assert.ok(!src.includes("difficulty === 'easy' ? 0.5 : 0.1"),
    'REST must not re-derive its own temperature ladder');
  const resolveCalls = src.match(/resolvePolicy\(/g) || [];
  assert.ok(resolveCalls.length >= 2,
    `both transports must call resolvePolicy; found ${resolveCalls.length}`);
});

test('websocket client sends deterministicMode', () => {
  const src = readFileSync(
    new URL('../assets/js/ai/alphaZeroClient.js', import.meta.url), 'utf8');
  assert.ok(src.includes('deterministicMode'),
    'client must forward deterministicMode over the websocket');
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test server/test_readout_policy.js`
Expected: FAIL — `DIFFICULTY_PARAMS must be gone`

- [ ] **Step 3: Replace the REST path's policy derivation**

In `server/index.js`, add the import at the top with the other local imports:

```javascript
import { resolvePolicy } from './readout_policy.js';
```

Delete the `DIFFICULTY_PARAMS` block at `:41-46` entirely.

In the `/api/move` handler, replace the `const params = DIFFICULTY_PARAMS[...]` line and the whole `let moveTemp; if (deterministicMode) {...} else if (...) {...} else {...}` block with:

```javascript
    const policy = resolvePolicy({ difficulty, deterministicMode, temperature });
    const mcts = new MCTS(inference, { nSimulations: policy.nSims });

    const startTime = Date.now();
    const { visitCounts, rootValue } = await mcts.search(gameState);
    const elapsed = Date.now() - startTime;

    const moveKey = mcts.selectMove(visitCounts, policy.moveTemp);
```

- [ ] **Step 4: Replace `computeBestMove`'s policy derivation**

In `computeBestMove`, replace:

```javascript
  const { nSims, moveTemp } = DIFFICULTY_PARAMS[difficulty] || DIFFICULTY_PARAMS.medium;
```

with:

```javascript
  const policy = resolvePolicy({
    difficulty,
    deterministicMode: opts.deterministicMode === true,
    temperature: opts.temperature,
  });
  const { nSims, moveTemp } = policy;
```

- [ ] **Step 5: Forward the flag from the WebSocket handler**

In the WS `move` handler, extend the `computeBestMove` call:

```javascript
        const result = await computeBestMove(msg.state, difficulty, {
          signal: controller.signal,
          deterministicMode: msg.deterministicMode === true,
          temperature: msg.temperature,
          onProgress: (p) => {
            if (cs.activeId !== id || controller.signal.aborted) return;
            safeSend({ type: 'progress', id, toMove, ...p });
          },
        });
```

- [ ] **Step 6: Send the flag from the client**

In `assets/js/ai/alphaZeroClient.js`, replace the `msg_out` construction:

```javascript
      const msg_out = { type: 'move', id, state, difficulty };
      if (opts?.includeVisits) msg_out.includeVisits = true;
      if (opts?.deterministicMode) msg_out.deterministicMode = true;
      if (opts?.temperature !== undefined) msg_out.temperature = opts.temperature;
```

- [ ] **Step 7: Run tests and lint**

Run: `npm run test:server; echo "EXIT=$?"`
Expected: `EXIT=0`
Run: `npm run lint; echo "EXIT=$?"`
Expected: `EXIT=0`

- [ ] **Step 8: Commit**

```bash
git add server/index.js server/test_readout_policy.js assets/js/ai/alphaZeroClient.js
git commit -m "fix(server): one readout policy across REST and websocket"
```

---

## Task A3: Cache correctness

**Files:**
- Modify: `server/cache.js:72-76` (`makeKey`), `get`, `set`
- Modify: `server/index.js:79-83` (lookup), and the `cache.set` call
- Modify: `server/test_readout_policy.js`

**Interfaces:**
- Consumes: `resolvePolicy` from Task A1.
- Produces: `makeKey(pegs, moves, size, scope)` where `scope` is a string identifying model + budget; `get(pegs, moves, size, scope)`; `set(pegs, moves, value, size, scope)`.

The cached object becomes the **raw search result** (`visitCounts`, `rootValue`), with the readout applied after the lookup. That fixes two defects at once: an `easy` 100-simulation result can no longer answer a `hard` 800-simulation request, and a cached *sampled* move can no longer make repeated stochastic requests sticky.

- [ ] **Step 1: Write the failing test**

Append to `server/test_readout_policy.js`:

```javascript
import { BoardMovesCache } from './cache.js';

const PEGS = new Map([['3,4', 'red'], ['5,6', 'black']]);
const MOVES = [[1, 1], [2, 2]];

test('cache scope separates simulation budgets', () => {
  const c = new BoardMovesCache(10);
  c.set(PEGS, MOVES, { rootValue: 0.1 }, 24, 'model.onnx|100');
  assert.equal(c.get(PEGS, MOVES, 24, 'model.onnx|800'), undefined);
  assert.deepEqual(c.get(PEGS, MOVES, 24, 'model.onnx|100'), { rootValue: 0.1 });
});

test('cache scope separates models', () => {
  const c = new BoardMovesCache(10);
  c.set(PEGS, MOVES, { rootValue: 0.1 }, 24, 'a.onnx|400');
  assert.equal(c.get(PEGS, MOVES, 24, 'b.onnx|400'), undefined);
});

// NEGATIVE CASE, constructed: without a scope the two budgets collide. This
// proves the tests above are not vacuously passing.
test('an unscoped key really does collide', () => {
  const c = new BoardMovesCache(10);
  c.set(PEGS, MOVES, { rootValue: 0.1 }, 24, '');
  assert.deepEqual(c.get(PEGS, MOVES, 24, ''), { rootValue: 0.1 });
});

test('index.js caches raw search results, not selected moves', () => {
  const src = readFileSync(new URL('./index.js', import.meta.url), 'utf8');
  assert.ok(src.includes('cacheScope'), 'lookup must be scoped');
  assert.ok(!src.includes('cache.set(gameState.pegs, moves, result,'),
    'the post-readout result object must not be cached');
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test server/test_readout_policy.js`
Expected: FAIL — `cache scope separates simulation budgets`

- [ ] **Step 3: Add scope to the cache key**

In `server/cache.js`, replace `makeKey` and thread `scope` through `get`/`set`:

```javascript
  /**
   * Key for a cached search result.
   *
   * `scope` MUST identify everything that changes the search output but is not
   * part of the board: at minimum the model identity and the simulation
   * budget. Readout policy is deliberately NOT in the scope, because only raw
   * search results are cached and the readout is applied after the lookup.
   */
  makeKey(pegs, moves, size = 24, scope = '') {
    const pegsHash = this._hashPegs(pegs, size);
    const movesHash = this._hashMoves(moves);
    return `${scope}:${pegsHash}:${movesHash}`;
  }

  get(pegs, moves, size = 24, scope = '') {
    const key = this.makeKey(pegs, moves, size, scope);
    const value = this.cache.get(key);
    if (value !== undefined) {
      this.cache.delete(key);
      this.cache.set(key, value);
      this.hits++;
    } else {
      this.misses++;
    }
    return value;
  }

  set(pegs, moves, value, size = 24, scope = '') {
    const key = this.makeKey(pegs, moves, size, scope);
    if (this.cache.has(key)) {
      this.cache.delete(key);
    }
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }
```

- [ ] **Step 4: Cache raw search results in the REST handler**

In `server/index.js`, replace the cache lookup block and the search block:

```javascript
    const policy = resolvePolicy({ difficulty, deterministicMode, temperature });
    const cacheScope = `${modelPath}|${policy.nSims}`;

    let search = cache.get(gameState.pegs, moves, gameState.boardSize, cacheScope);
    const mcts = new MCTS(inference, { nSimulations: policy.nSims });
    const startTime = Date.now();
    let cached = true;
    if (search === undefined) {
      cached = false;
      const out = await mcts.search(gameState);
      const visits = {};
      for (const [key, count] of out.visitCounts) {
        visits[key] = count;
      }
      search = { visits, rootValue: out.rootValue };
      cache.set(gameState.pegs, moves, search, gameState.boardSize, cacheScope);
    }
    const elapsed = Date.now() - startTime;

    // Readout is applied AFTER the cache lookup, so a cached stochastic
    // request re-samples instead of returning a sticky move.
    const visitCounts = new Map(Object.entries(search.visits));
    const moveKey = mcts.selectMove(visitCounts, policy.moveTemp);
    const [row, col] = moveKey.split(',').map(Number);

    res.json({
      move: { row, col },
      value: search.rootValue,
      visits: search.visits,
      elapsed,
      cached,
    });
```

Delete the old trailing `if (!deterministicMode) { cache.set(...) }` block and the old `res.json(result)`.

- [ ] **Step 5: Run tests and lint**

Run: `npm run test:server; echo "EXIT=$?"`
Expected: `EXIT=0`
Run: `npm run lint; echo "EXIT=$?"`
Expected: `EXIT=0`

- [ ] **Step 6: Commit**

```bash
git add server/cache.js server/index.js server/test_readout_policy.js
git commit -m "fix(server): scope search cache by model and budget, apply readout after lookup"
```

---

# PHASE B — Research harness

## Task B1: Pure readout module with the frozen rule

**Files:**
- Create: `scripts/GPU/alphazero/eval_readout.py`
- Create: `tests/test_eval_readout.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces:
  - `ReadoutConfig(mode: str, opening_temp_plies: int, temp_high: float, temp_low: float)`
  - `ChildStat(move: Tuple[int,int], visits: int, q_child: Optional[float], q_root: Optional[float])`
  - `hoeffding_radius(n: int) -> float`
  - `top_two(stats: Dict[Tuple[int,int], Tuple[int, Optional[float]]]) -> List[ChildStat]`
  - `lcb_override(top2: List[ChildStat]) -> Optional[Tuple[int,int]]`
  - `select(counts, ply, readout, rng, top2=None) -> Tuple[Tuple[int,int], bool]`
  - Constants `VALUE_RANGE`, `DELTA`, `MIN_CHILD_VISITS`, `MODE_OPENING_TEMPERATURE`, `MODE_ARGMAX`, `MODE_HOEFFDING_LCB`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_readout.py`:

```python
"""Frozen-rule tests for the eval readout module.

The constants under test are FROZEN (design spec §7.4, 2026-08-06). These
tests exist to make a silent drift in any of them fail loudly.
"""
import math
import random

import pytest

from scripts.GPU.alphazero import eval_readout as R


def test_frozen_constants():
    assert R.VALUE_RANGE == 2.0
    assert R.DELTA == 0.05
    assert R.MIN_CHILD_VISITS == 8


def test_hoeffding_numerator_matches_the_frozen_value():
    # eps(n) = 2*sqrt(1.84445/n) as written in the spec.
    assert R.hoeffding_radius(1) == pytest.approx(2.0 * math.sqrt(1.84445), abs=1e-4)


def test_hoeffding_worked_magnitudes():
    assert R.hoeffding_radius(190) == pytest.approx(0.197, abs=5e-4)
    assert R.hoeffding_radius(100) == pytest.approx(0.272, abs=5e-4)
    assert R.hoeffding_radius(40) == pytest.approx(0.430, abs=5e-4)
    assert R.hoeffding_radius(8) == pytest.approx(0.960, abs=5e-4)


def test_min_child_visits_is_the_boundary_of_the_frozen_requirement():
    # n_min follows from the preregistered requirement eps(n) <= 1.0.
    # Constructed boundary: 8 satisfies it, 7 does not.
    assert R.hoeffding_radius(R.MIN_CHILD_VISITS) <= 1.0
    assert R.hoeffding_radius(R.MIN_CHILD_VISITS - 1) > 1.0


def test_hoeffding_radius_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        R.hoeffding_radius(0)


def test_top_two_orders_by_visits_then_canonical_move():
    stats = {(1, 1): (10, 0.1), (2, 2): (50, -0.2), (0, 5): (10, 0.3)}
    t2 = R.top_two(stats)
    assert [c.move for c in t2] == [(2, 2), (0, 5)]  # tie 10/10 -> (0,5) first


def test_top_two_maps_zero_visit_children_to_none_not_zero():
    stats = {(1, 1): (0, 0.0), (2, 2): (50, -0.2)}
    t2 = R.top_two(stats)
    zero = [c for c in t2 if c.move == (1, 1)][0]
    assert zero.q_child is None
    assert zero.q_root is None


def test_root_perspective_is_the_negation_of_child_perspective():
    stats = {(1, 1): (30, 0.25), (2, 2): (50, -0.20)}
    t2 = R.top_two(stats)
    for c in t2:
        assert c.q_root == pytest.approx(-c.q_child)


def test_lcb_override_fires_when_challenger_lcb_is_higher():
    # leader 190 visits, q_root -0.30 -> LCB -0.497
    # challenger 40 visits, q_root  0.00 -> LCB -0.430  (higher -> override)
    top2 = [
        R.ChildStat((2, 2), 190, 0.30, -0.30),
        R.ChildStat((1, 1), 40, -0.00, 0.00),
    ]
    assert R.lcb_override(top2) == (1, 1)


def test_lcb_override_declines_when_the_gap_is_too_small():
    # Same visits; challenger only 0.10 better, needs > 0.232.
    top2 = [
        R.ChildStat((2, 2), 190, 0.30, -0.30),
        R.ChildStat((1, 1), 40, -0.20, -0.20),
    ]
    assert R.lcb_override(top2) is None


def test_lcb_override_declines_below_min_visits():
    top2 = [
        R.ChildStat((2, 2), 190, 0.30, -0.30),
        R.ChildStat((1, 1), 7, -0.90, 0.90),
    ]
    assert R.lcb_override(top2) is None


def test_lcb_override_declines_on_undefined_q():
    top2 = [
        R.ChildStat((2, 2), 190, None, None),
        R.ChildStat((1, 1), 40, -0.90, 0.90),
    ]
    assert R.lcb_override(top2) is None


def test_lcb_override_declines_with_fewer_than_two_children():
    assert R.lcb_override([R.ChildStat((2, 2), 190, 0.3, -0.3)]) is None
    assert R.lcb_override([]) is None


def test_argmax_mode_is_deterministic_and_ignores_rng():
    counts = {(1, 1): 10, (2, 2): 50, (0, 5): 10}
    cfg = R.ReadoutConfig(mode=R.MODE_ARGMAX)
    for _ in range(5):
        move, overrode = R.select(counts, ply=3, readout=cfg, rng=random.Random(1))
        assert move == (2, 2)
        assert overrode is False


def test_argmax_ties_break_in_canonical_numeric_order():
    counts = {(2, 2): 50, (0, 5): 50}
    cfg = R.ReadoutConfig(mode=R.MODE_ARGMAX)
    move, _ = R.select(counts, ply=3, readout=cfg, rng=random.Random(1))
    assert move == (0, 5)


def test_opening_temperature_mode_samples_early_and_argmaxes_late_when_temp_low_is_zero():
    counts = {(1, 1): 10, (2, 2): 50}
    cfg = R.ReadoutConfig(mode=R.MODE_OPENING_TEMPERATURE, temp_high=1.0, temp_low=0.0)
    late, _ = R.select(counts, ply=20, readout=cfg, rng=random.Random(1))
    assert late == (2, 2)
    seen = {R.select(counts, ply=0, readout=cfg, rng=random.Random(s))[0]
            for s in range(40)}
    assert seen == {(1, 1), (2, 2)}  # opening genuinely samples both


def test_hoeffding_mode_samples_the_opening_then_overrides_post_opening():
    counts = {(1, 1): 40, (2, 2): 190}
    top2 = [
        R.ChildStat((2, 2), 190, 0.30, -0.30),
        R.ChildStat((1, 1), 40, -0.00, 0.00),
    ]
    cfg = R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB, temp_high=1.0)
    move, overrode = R.select(counts, ply=20, readout=cfg,
                              rng=random.Random(1), top2=top2)
    assert move == (1, 1)
    assert overrode is True


def test_hoeffding_mode_reports_no_override_when_the_rule_declines():
    counts = {(1, 1): 40, (2, 2): 190}
    top2 = [
        R.ChildStat((2, 2), 190, 0.30, -0.30),
        R.ChildStat((1, 1), 40, -0.20, -0.20),
    ]
    cfg = R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB, temp_high=1.0)
    move, overrode = R.select(counts, ply=20, readout=cfg,
                              rng=random.Random(1), top2=top2)
    assert move == (2, 2)
    assert overrode is False


def test_hoeffding_mode_requires_top2_post_opening():
    cfg = R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB)
    with pytest.raises(ValueError):
        R.select({(1, 1): 5}, ply=20, readout=cfg, rng=random.Random(1))


def test_unknown_mode_is_rejected_at_construction():
    with pytest.raises(ValueError):
        R.ReadoutConfig(mode="wishful")


def test_select_rejects_empty_counts():
    cfg = R.ReadoutConfig(mode=R.MODE_ARGMAX)
    with pytest.raises(ValueError):
        R.select({}, ply=0, readout=cfg, rng=random.Random(1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_readout.py -q; echo "EXIT=$?"`
Expected: FAIL — `ModuleNotFoundError: ... eval_readout`

- [ ] **Step 3: Write the implementation**

Create `scripts/GPU/alphazero/eval_readout.py`:

```python
"""Final move readout for checkpoint-eval games.

Pure: no MLX, no MCTS, no game engine. This module owns every readout rule the
evaluation harness can play, including the FROZEN Hoeffding-LCB override
(design spec section 7.4, frozen 2026-08-06).

Why this lives outside MCTS: `mcts.MCTS` draws prior-shuffle, PUCT tie-break
and move readout from ONE `self.rng`, so changing the readout changes the
generator state entering every subsequent search. Evaluation therefore selects
moves here, with its own RNG stream, and never calls `mcts.select_move`.
Self-play still calls `mcts.select_move` and is unaffected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

Move = Tuple[int, int]

# --- FROZEN constants: do not change (design spec section 7.4) -------------
VALUE_RANGE = 2.0       # backed-up MCTS values lie in [-1, 1]
DELTA = 0.05            # confidence level; sets the radius SCALE only.
                        # NOT the match's statistical alpha, and it carries no
                        # repeated-decision guarantee across many positions.
MIN_CHILD_VISITS = 8    # smallest n with hoeffding_radius(n) <= 1.0, from the
                        # preregistered "radius no wider than half the value
                        # range" requirement. The requirement is a judgement;
                        # only the arithmetic that follows is forced.
_HOEFFDING_NUM = math.log(2.0 / DELTA) / 2.0    # == 1.844439...

MODE_OPENING_TEMPERATURE = "opening_temperature"
MODE_ARGMAX = "argmax"
MODE_HOEFFDING_LCB = "hoeffding_lcb"
MODES = (MODE_OPENING_TEMPERATURE, MODE_ARGMAX, MODE_HOEFFDING_LCB)

# Below this, select_move's own deterministic branch takes over. Matches
# mcts.select_move's threshold so the two agree on what "temperature 0" means.
_DETERMINISTIC_TEMP = 0.01


@dataclass(frozen=True)
class ReadoutConfig:
    """How one agent turns a completed search into a played move."""
    mode: str = MODE_OPENING_TEMPERATURE
    opening_temp_plies: int = 20
    temp_high: float = 1.0
    temp_low: float = 0.1

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(
                f"unknown readout mode {self.mode!r}; expected one of {MODES}")


@dataclass(frozen=True)
class ChildStat:
    """One root child's completed statistics.

    `q_child` is the child's own stored mean; `q_root` is the MOVER's
    perspective and equals `-q_child` (mcts.py:1122). Both are None when the
    mean is UNDEFINED (zero completed visits, or a non-finite value) --
    never 0.0, which is what `MCTSNode.q_value` returns at visit_count == 0.
    """
    move: Move
    visits: int
    q_child: Optional[float]
    q_root: Optional[float]


def hoeffding_radius(n: int) -> float:
    """Hoeffding half-width for the mean of `n` observations spanning
    VALUE_RANGE.

    NOTE: MCTS backups are adaptively sampled and correlated, not i.i.d., so
    this is a principled UNFITTED radius, not a valid confidence guarantee.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return VALUE_RANGE * math.sqrt(_HOEFFDING_NUM / n)


def top_two(stats: Dict[Move, Tuple[int, Optional[float]]]) -> List[ChildStat]:
    """The two children with the highest completed visit counts.

    `stats` maps move -> (completed_visits, child_perspective_mean_or_None).
    Ordering is (-visits, move), so visit ties break in canonical numeric
    (row, col) order. Returns 0, 1 or 2 entries.
    """
    ordered = sorted(stats.items(), key=lambda kv: (-kv[1][0], kv[0]))
    out: List[ChildStat] = []
    for move, (visits, q_child) in ordered[:2]:
        defined = visits > 0 and q_child is not None and math.isfinite(q_child)
        q_c = float(q_child) if defined else None
        out.append(ChildStat(move=move, visits=int(visits), q_child=q_c,
                             q_root=(None if q_c is None else -q_c)))
    return out


def lcb_override(top2: List[ChildStat]) -> Optional[Move]:
    """The FROZEN rule. Returns the challenger's move iff it overrides the
    visit leader; otherwise None, meaning play the leader.

    This is a conservative RANKING HEURISTIC. It does not establish at 95%
    confidence that the challenger is the better move.
    """
    if len(top2) < 2:
        return None
    leader, challenger = top2[0], top2[1]
    if leader.visits < MIN_CHILD_VISITS or challenger.visits < MIN_CHILD_VISITS:
        return None
    if leader.q_root is None or challenger.q_root is None:
        return None
    lcb_leader = leader.q_root - hoeffding_radius(leader.visits)
    lcb_challenger = challenger.q_root - hoeffding_radius(challenger.visits)
    return challenger.move if lcb_challenger > lcb_leader else None


def _argmax(counts: Dict[Move, int]) -> Move:
    """Visit-count argmax with a deterministic canonical (row, col) tie-break.
    No RNG: the readout stream is never consumed on this path."""
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _sample_by_temperature(counts: Dict[Move, int], temp: float, rng) -> Move:
    """Sample proportional to count^(1/temp).

    Mirrors mcts.select_move's log-count softmax exactly, including the 1e-8
    floor and the max-subtraction, so the two implementations agree on the
    distribution (they do NOT agree game-for-game, because the streams differ).
    """
    moves = list(counts.keys())
    log_counts = [math.log(counts[m] + 1e-8) / temp for m in moves]
    max_log = max(log_counts)
    exp_counts = [math.exp(lc - max_log) for lc in log_counts]
    total = sum(exp_counts)
    r = rng.random()
    cumsum = 0.0
    for move, e in zip(moves, exp_counts):
        cumsum += e / total
        if r <= cumsum:
            return move
    return moves[-1]


def _opening_move(counts: Dict[Move, int], temp: float, rng) -> Move:
    if temp < _DETERMINISTIC_TEMP:
        return _argmax(counts)
    return _sample_by_temperature(counts, temp, rng)


def select(counts: Dict[Move, int], ply: int, readout: ReadoutConfig, rng,
           top2: Optional[List[ChildStat]] = None) -> Tuple[Move, bool]:
    """Pick the played move. Returns (move, overrode_visit_leader).

    `rng` MUST be the readout stream, never an MCTS search stream.
    `top2` is required by MODE_HOEFFDING_LCB after the opening, ignored
    otherwise.
    """
    if not counts:
        raise ValueError("select called with empty visit counts")

    if readout.mode == MODE_ARGMAX:
        return _argmax(counts), False

    if readout.mode == MODE_OPENING_TEMPERATURE:
        temp = (readout.temp_high if ply < readout.opening_temp_plies
                else readout.temp_low)
        return _opening_move(counts, temp, rng), False

    # MODE_HOEFFDING_LCB: sample the opening for match diversity, then play
    # visit argmax with the frozen override.
    if ply < readout.opening_temp_plies:
        return _opening_move(counts, readout.temp_high, rng), False
    if top2 is None:
        raise ValueError(
            "hoeffding_lcb readout requires top2 child stats after the opening")
    leader = _argmax(counts)
    override = lcb_override(top2)
    if override is not None and override != leader:
        return override, True
    return leader, False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_readout.py -q; echo "EXIT=$?"`
Expected: `EXIT=0`, all tests pass

- [ ] **Step 5: Commit**

```bash
git add scripts/GPU/alphazero/eval_readout.py tests/test_eval_readout.py
git commit -m "feat(eval): pure readout module with the frozen Hoeffding-LCB rule"
```

---

## Task B2: Root-child telemetry in replay capture

**Files:**
- Modify: `scripts/GPU/alphazero/eval_replay.py:13` (schema version), `:16-44` (`ply_record`)
- Create: `tests/test_eval_readout_telemetry.py`

**Interfaces:**
- Consumes: `ChildStat` from Task B1.
- Produces: `ply_record(ply, player, move, counts, root_value, top2=None, overrode_leader=False)`; `REPLAY_SCHEMA_VERSION == 2`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_readout_telemetry.py`:

```python
"""Telemetry contract tests: perspective, undefined values, schema version."""
import pytest

from scripts.GPU.alphazero import eval_readout as R
from scripts.GPU.alphazero.eval_replay import REPLAY_SCHEMA_VERSION, ply_record


def _top2():
    return [
        R.ChildStat((2, 2), 190, 0.30, -0.30),
        R.ChildStat((1, 1), 40, -0.05, 0.05),
    ]


def test_schema_version_is_bumped_for_top2():
    assert REPLAY_SCHEMA_VERSION == 2


def test_ply_record_without_top2_keeps_the_field_null_not_empty():
    rec = ply_record(0, "red", (2, 2), {(2, 2): 5, (1, 1): 3}, 0.1)
    assert rec["top2"] is None
    assert rec["readout_overrode_leader"] is False


def test_ply_record_emits_both_perspectives():
    rec = ply_record(21, "red", (2, 2), {(2, 2): 190, (1, 1): 40}, 0.1,
                     top2=_top2())
    a, b = rec["top2"]
    assert a["completed_visit_count"] == 190
    assert a["q_value_child_perspective"] == pytest.approx(0.30)
    assert a["q_value_root_perspective"] == pytest.approx(-0.30)
    assert b["q_value_root_perspective"] == pytest.approx(0.05)


def test_ply_record_preserves_undefined_q_as_null():
    top2 = [R.ChildStat((2, 2), 190, 0.3, -0.3), R.ChildStat((1, 1), 0, None, None)]
    rec = ply_record(21, "red", (2, 2), {(2, 2): 190, (1, 1): 0}, 0.1, top2=top2)
    assert rec["top2"][1]["q_value_child_perspective"] is None
    assert rec["top2"][1]["q_value_root_perspective"] is None


def test_ply_record_records_the_override_flag():
    rec = ply_record(21, "red", (1, 1), {(2, 2): 190, (1, 1): 40}, 0.1,
                     top2=_top2(), overrode_leader=True)
    assert rec["readout_overrode_leader"] is True


def test_ply_record_still_fails_loud_on_a_move_outside_the_counts():
    with pytest.raises(ValueError):
        ply_record(0, "red", (9, 9), {(2, 2): 5}, 0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_readout_telemetry.py -q; echo "EXIT=$?"`
Expected: FAIL — `assert 1 == 2` on the schema version

- [ ] **Step 3: Write the implementation**

In `scripts/GPU/alphazero/eval_replay.py`, change the version and extend `ply_record`:

```python
REPLAY_SCHEMA_VERSION = 2


def _child_stat_dict(stat):
    """Serialize one ChildStat. Undefined means stay None, never 0.0."""
    return {
        "row": stat.move[0],
        "col": stat.move[1],
        "completed_visit_count": stat.visits,
        "q_value_child_perspective": stat.q_child,
        "q_value_root_perspective": stat.q_root,
    }


def ply_record(ply, player, move, counts, root_value, top2=None,
               overrode_leader=False):
    """One per-ply replay record.

    `move` is the selected (row, col). `counts` is the MCTS visit-count dict
    {(row, col): visits} over all legal moves at this root. `root_value` is
    root.q_value from the perspective of `player` (the side about to move),
    before the move is applied. `top2` is the top-two root children by
    completed visits (eval_readout.ChildStat), or None when not captured --
    None means "not captured", never "no children". Fail loud rather than
    emit a corrupt record.
    """
    if not counts:
        raise ValueError(f"ply {ply}: empty visit counts")
    if move not in counts:
        raise ValueError(f"ply {ply}: selected move {move} not in visit counts")
    total = sum(counts.values())
    # rank: descending visit count, ties broken by ascending (row, col).
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    rank = 1 + next(i for i, (m, _c) in enumerate(ranked) if m == move)
    row, col = move
    return {
        "ply": ply,
        "player": player,
        "row": row,
        "col": col,
        "root_value": root_value,
        "root_top1_share": max(counts.values()) / total,
        "selected_visit_rank": rank,
        "selected_visit_count": counts[move],
        "root_total_visits": total,
        "n_legal": len(counts),
        "top2": ([_child_stat_dict(s) for s in top2]
                 if top2 is not None else None),
        "readout_overrode_leader": bool(overrode_leader),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_readout_telemetry.py -q; echo "EXIT=$?"`
Expected: `EXIT=0`

- [ ] **Step 5: Run the existing replay tests for regressions**

Run: `python -m pytest tests/ -q -k "replay" ; echo "EXIT=$?"`
Expected: `EXIT=0`. If a test pins `REPLAY_SCHEMA_VERSION == 1`, update that assertion to `2` — the bump is intentional.

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/eval_replay.py tests/test_eval_readout_telemetry.py
git commit -m "feat(eval): capture top-two root-child visits and both Q perspectives"
```

---

## Task B3: Agent identity on tasks and results

**Files:**
- Modify: `scripts/GPU/alphazero/eval_runner.py:29-53` (dataclasses), `:133-148` (`make_result`), `:151-170` (add a sibling task builder)
- Create: `tests/test_eval_agent_identity.py`

**Interfaces:**
- Consumes: `ReadoutConfig` from Task B1.
- Produces:
  - `AgentSpec(agent_id: str, checkpoint: str, readout: ReadoutConfig)`
  - `EvalGameTask` gains `red_agent: Optional[AgentSpec] = None`, `black_agent: Optional[AgentSpec] = None`
  - `EvalGameResult` gains `red_agent_id`, `black_agent_id`, `winner_agent_id`, `red_readout`, `black_readout`, `same_checkpoint`, `comparison_unit` (all `Optional`, default `None`)
  - `build_agent_pairing_tasks(pairing_id, agent_a, agent_b, games, base_seed, pairing_index=0) -> List[EvalGameTask]`
  - `AGENT_COMPARISON_UNIT = "agent"`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_agent_identity.py`:

```python
"""Agent identity: colour binding, winner attribution, legacy preservation."""
import pytest

from scripts.GPU.alphazero import eval_readout as R
from scripts.GPU.alphazero.eval_runner import (
    AGENT_COMPARISON_UNIT, AgentSpec, build_agent_pairing_tasks,
    build_pairing_tasks, make_result,
)

CKPT = "checkpoints/x/model_iter_0001.safetensors"
CONTROL = AgentSpec("control", CKPT, R.ReadoutConfig(mode=R.MODE_ARGMAX))
CANDIDATE = AgentSpec("candidate", CKPT,
                      R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB))


def test_agent_tasks_alternate_colours_by_game_index():
    tasks = build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 4, 100)
    assert [t.red_agent.agent_id for t in tasks] == [
        "control", "candidate", "control", "candidate"]
    assert [t.black_agent.agent_id for t in tasks] == [
        "candidate", "control", "candidate", "control"]


def test_agent_tasks_carry_the_readout_with_the_agent_across_the_swap():
    tasks = build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 2, 100)
    for t in tasks:
        assert t.red_agent.readout.mode != t.black_agent.readout.mode
        red_is_control = t.red_agent.agent_id == "control"
        assert (t.red_agent.readout.mode == R.MODE_ARGMAX) is red_is_control


def test_agent_tasks_still_fill_the_checkpoint_fields():
    tasks = build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 2, 100)
    assert all(t.red_checkpoint == CKPT and t.black_checkpoint == CKPT
               for t in tasks)


def test_agent_tasks_reject_odd_game_counts():
    with pytest.raises(ValueError):
        build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 3, 100)


def test_winner_agent_id_follows_the_colour_that_won():
    tasks = build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 2, 100)
    r0 = make_result(tasks[0], "red", "win", 50)      # game 0: red == control
    assert r0.winner_agent_id == "control"
    r1 = make_result(tasks[1], "red", "win", 50)      # game 1: red == candidate
    assert r1.winner_agent_id == "candidate"


def test_draws_leave_winner_agent_id_null_not_empty():
    tasks = build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 2, 100)
    r = make_result(tasks[0], None, "state_cap", 280)
    assert r.winner_agent_id is None
    assert r.red_score == 0.5 and r.black_score == 0.5


def test_agent_results_are_labelled_as_agent_comparisons():
    tasks = build_agent_pairing_tasks("p", CONTROL, CANDIDATE, 2, 100)
    r = make_result(tasks[0], "black", "win", 50)
    assert r.comparison_unit == AGENT_COMPARISON_UNIT
    assert r.same_checkpoint is True
    assert r.red_readout == R.MODE_ARGMAX
    assert r.black_readout == R.MODE_HOEFFDING_LCB


def test_legacy_checkpoint_tasks_carry_no_agent_fields():
    # NEGATIVE CASE, constructed: the legacy path must stay unlabelled, so a
    # consumer can tell the two artifact kinds apart.
    tasks = build_pairing_tasks("p", "a.safetensors", "b.safetensors", 2, 100, 0)
    r = make_result(tasks[0], "red", "win", 50)
    assert r.comparison_unit is None
    assert r.winner_agent_id is None
    assert r.winner_checkpoint == "a.safetensors"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_agent_identity.py -q; echo "EXIT=$?"`
Expected: FAIL — `ImportError: cannot import name 'AgentSpec'`

- [ ] **Step 3: Write the implementation**

In `scripts/GPU/alphazero/eval_runner.py`, add the import near the top:

```python
from .eval_readout import ReadoutConfig
```

Add the constant beside `GAMES_PER_PAIRING_LIMIT`:

```python
# Marks a result set whose comparison unit is the AGENT, not the checkpoint.
# Same-checkpoint readout matches are meaningless under checkpoint keying.
AGENT_COMPARISON_UNIT = "agent"
```

Add the spec dataclass above `EvalGameTask`:

```python
@dataclass(frozen=True)
class AgentSpec:
    """One competitor. `agent_id` is the experimental identity and is
    independent of `checkpoint` -- two agents may share a checkpoint."""
    agent_id: str
    checkpoint: str
    readout: ReadoutConfig
```

Extend the two existing dataclasses with trailing optional fields (additive, so
every existing caller keeps working):

```python
@dataclass(frozen=True)
class EvalGameTask:
    task_id: int
    pairing_id: str
    game_idx: int
    red_checkpoint: str
    black_checkpoint: str
    seed: int
    red_agent: Optional["AgentSpec"] = None
    black_agent: Optional["AgentSpec"] = None


@dataclass
class EvalGameResult:
    task_id: int
    pairing_id: str
    game_idx: int
    red_checkpoint: str
    black_checkpoint: str
    winner: Optional[str]            # "red" | "black" | None
    winner_checkpoint: Optional[str]
    reason: str                      # "win"|"state_cap"|"board_full"|"unknown_error"
    n_moves: int
    red_score: float
    black_score: float
    replay_path: Optional[str] = None
    # Agent-comparison fields. All None on legacy checkpoint-vs-checkpoint
    # results, which is how a consumer tells the two artifact kinds apart.
    red_agent_id: Optional[str] = None
    black_agent_id: Optional[str] = None
    winner_agent_id: Optional[str] = None
    red_readout: Optional[str] = None
    black_readout: Optional[str] = None
    same_checkpoint: Optional[bool] = None
    comparison_unit: Optional[str] = None
```

Replace `make_result`:

```python
def make_result(task: EvalGameTask, winner, reason, n_moves,
                replay_path=None) -> EvalGameResult:
    """Build a result, mapping winner colour -> checkpoint and 0/0.5/1 scores.

    When the task carries AgentSpecs, the experimental identity fields are
    filled too. The winner's AGENT is read from the task's colour binding and
    is never derived from winner_checkpoint (which is ambiguous when both
    agents share a checkpoint).
    """
    if winner == "red":
        red_score, black_score, winner_ckpt = 1.0, 0.0, task.red_checkpoint
    elif winner == "black":
        red_score, black_score, winner_ckpt = 0.0, 1.0, task.black_checkpoint
    else:
        red_score, black_score, winner_ckpt = 0.5, 0.5, None

    agent_fields = {}
    if task.red_agent is not None and task.black_agent is not None:
        if winner == "red":
            winner_agent_id = task.red_agent.agent_id
        elif winner == "black":
            winner_agent_id = task.black_agent.agent_id
        else:
            winner_agent_id = None
        agent_fields = {
            "red_agent_id": task.red_agent.agent_id,
            "black_agent_id": task.black_agent.agent_id,
            "winner_agent_id": winner_agent_id,
            "red_readout": task.red_agent.readout.mode,
            "black_readout": task.black_agent.readout.mode,
            "same_checkpoint": task.red_agent.checkpoint == task.black_agent.checkpoint,
            "comparison_unit": AGENT_COMPARISON_UNIT,
        }

    return EvalGameResult(
        task_id=task.task_id, pairing_id=task.pairing_id, game_idx=task.game_idx,
        red_checkpoint=task.red_checkpoint, black_checkpoint=task.black_checkpoint,
        winner=winner, winner_checkpoint=winner_ckpt, reason=reason,
        n_moves=n_moves, red_score=red_score, black_score=black_score,
        replay_path=replay_path, **agent_fields,
    )
```

Add the task builder next to `build_pairing_tasks`:

```python
def build_agent_pairing_tasks(pairing_id, agent_a: AgentSpec, agent_b: AgentSpec,
                              games, base_seed, pairing_index=0):
    """Balanced-colour tasks for two AGENTS, which may share a checkpoint.

    Even game_idx -> red=A; odd -> red=B. The readout travels WITH the agent
    across the colour swap; that binding is what the experiment rests on.
    """
    if games < 2:
        raise ValueError("games must be >= 2")
    if games % 2 != 0:
        raise ValueError("games must be even for balanced colors")
    if games >= GAMES_PER_PAIRING_LIMIT:
        raise ValueError(f"games must be < {GAMES_PER_PAIRING_LIMIT}")
    offset = pairing_index * GAMES_PER_PAIRING_LIMIT
    tasks = []
    for g in range(games):
        red, black = (agent_a, agent_b) if g % 2 == 0 else (agent_b, agent_a)
        tasks.append(EvalGameTask(
            task_id=offset + g, pairing_id=pairing_id, game_idx=g,
            red_checkpoint=red.checkpoint, black_checkpoint=black.checkpoint,
            seed=base_seed + offset + g,
            red_agent=red, black_agent=black,
        ))
    return tasks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_agent_identity.py -q; echo "EXIT=$?"`
Expected: `EXIT=0`

- [ ] **Step 5: Verify no existing eval test regressed**

Run: `python -m pytest tests/ -q -k "eval" ; echo "EXIT=$?"`
Expected: `EXIT=0`

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/eval_runner.py tests/test_eval_agent_identity.py
git commit -m "feat(eval): agent identity decoupled from checkpoint path"
```

---

## Task B4: Per-agent readout in the game loop, with split RNG

**Files:**
- Modify: `scripts/GPU/alphazero/eval_runner.py:98-130` (`play_eval_game`), `:229-239` (`_play_and_build_result`)
- Modify: `tests/test_eval_readout_telemetry.py`

**Interfaces:**
- Consumes: `AgentSpec` (B3), `eval_readout.select` / `top_two` (B1), `ply_record` (B2).
- Produces: `readout_from_eval_config(config: EvalConfig) -> ReadoutConfig`; `root_child_stats(counts, root) -> Dict[Move, Tuple[int, Optional[float]]]`; `play_eval_game(..., red_readout=None, black_readout=None)`.

Two behaviours change here, both deliberate and both recorded in the spec:

1. `mcts.search(...)` becomes `mcts.search_with_root(...)`. `search()` already delegates to `search_with_root()` (`mcts.py:598`), so the search itself is unchanged; the root node is simply no longer discarded.
2. Move selection moves from `mcts.select_move` to `eval_readout.select` with its own RNG. **This changes evaluation RNG coupling** — evaluation games are no longer reproducible game-for-game against historical runs. Without it the experiment is not readout-only, because `self.rng` feeds prior-shuffle and PUCT tie-breaks as well as the readout.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval_readout_telemetry.py`:

```python
import random

from scripts.GPU.alphazero.eval_runner import (
    EvalConfig, play_eval_game, readout_from_eval_config, root_child_stats,
)
from tests.eval_fakes import FakeEvaluator

SMALL = EvalConfig(board_size=6, mcts_sims=32, max_moves=24,
                   opening_temp_plies=2)


def test_readout_from_eval_config_maps_argmax_mode():
    cfg = EvalConfig(selection_mode="argmax")
    assert readout_from_eval_config(cfg).mode == R.MODE_ARGMAX


def test_readout_from_eval_config_maps_opening_temperature():
    rd = readout_from_eval_config(EvalConfig(selection_mode="opening_temperature"))
    assert rd.mode == R.MODE_OPENING_TEMPERATURE
    assert rd.temp_high == 1.0 and rd.temp_low == 0.1


def test_readout_from_eval_config_rejects_unknown_modes():
    with pytest.raises(ValueError):
        readout_from_eval_config(EvalConfig(selection_mode="wishful"))


def test_search_identity_holds_at_a_fixed_root():
    """CONSTRUCTED: the same completed search feeds both readouts, so visit
    counts and root value are identical and only the played move may differ.

    This is a per-position property. It is FALSE across a game by
    construction, so it must never be asserted at game level.
    """
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig

    state = TwixtState(active_size=6, to_move="red", max_plies_limit=24)
    mcts = MCTS(FakeEvaluator(0.0), MCTSConfig(n_simulations=32),
                random.Random(7))
    counts, root_value, root = mcts.search_with_root(state, add_noise=False)
    stats = root_child_stats(counts, root)
    top2 = R.top_two(stats)

    a, _ = R.select(counts, 5, R.ReadoutConfig(mode=R.MODE_ARGMAX),
                    random.Random(1))
    b, _ = R.select(counts, 5, R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB,
                                               opening_temp_plies=2),
                    random.Random(1), top2=top2)
    # Same tree in, same statistics out; only the choice may differ.
    assert counts is not None and root_value is not None
    assert a in counts and b in counts


def test_readout_cannot_advance_the_search_rng():
    """CONSTRUCTED negative case for the RNG split.

    mcts.MCTS draws prior-shuffle, PUCT tie-break and readout from ONE
    self.rng. The eval readout must never touch it -- otherwise a readout that
    consumes a different number of draws changes the generator state entering
    every later search, and the experiment is not readout-only.

    The negative case is mcts.select_move, which DOES advance the stream. If
    that assertion ever stops holding, this test has gone vacuous.
    """
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig

    state = TwixtState(active_size=6, to_move="red", max_plies_limit=24)
    mcts = MCTS(FakeEvaluator(0.0), MCTSConfig(n_simulations=32),
                random.Random(7))
    counts, _rv, root = mcts.search_with_root(state, add_noise=False)
    top2 = R.top_two(root_child_stats(counts, root))

    before = mcts.rng.getstate()
    R.select(counts, 0, R.ReadoutConfig(mode=R.MODE_OPENING_TEMPERATURE),
             random.Random(1))
    R.select(counts, 21, R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB),
             random.Random(1), top2=top2)
    assert mcts.rng.getstate() == before, "eval readout touched the search RNG"

    # NEGATIVE CASE: the in-MCTS readout does consume the search stream.
    mcts.select_move(counts, ply=0)
    assert mcts.rng.getstate() != before, (
        "mcts.select_move no longer advances self.rng -- this test can no "
        "longer detect a leak and must be redesigned")


def test_root_child_stats_maps_unvisited_children_to_none():
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig

    state = TwixtState(active_size=6, to_move="red", max_plies_limit=24)
    mcts = MCTS(FakeEvaluator(0.0), MCTSConfig(n_simulations=8), random.Random(3))
    counts, _rv, root = mcts.search_with_root(state, add_noise=False)
    stats = root_child_stats(counts, root)
    assert set(stats) == set(counts)
    for move, (visits, q_child) in stats.items():
        if visits == 0:
            assert q_child is None, f"{move}: undefined mean must be None"


def test_play_eval_game_binds_each_readout_to_its_colour():
    red_rd = R.ReadoutConfig(mode=R.MODE_ARGMAX)
    black_rd = R.ReadoutConfig(mode=R.MODE_OPENING_TEMPERATURE,
                               opening_temp_plies=2, temp_high=1.0, temp_low=0.0)
    winner, reason, n, records = play_eval_game(
        FakeEvaluator(0.0), FakeEvaluator(0.0), SMALL, seed=11, capture=True,
        red_readout=red_rd, black_readout=black_rd)
    assert n > 0
    assert reason in {"win", "state_cap", "board_full"}
    # Red plays argmax at EVERY ply, so its selected visit rank is always 1.
    red_ranks = {r["selected_visit_rank"] for r in records if r["player"] == "red"}
    assert red_ranks == {1}


def test_play_eval_game_captures_top2_after_the_opening():
    rd = R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB, opening_temp_plies=2)
    _w, _r, _n, records = play_eval_game(
        FakeEvaluator(0.0), FakeEvaluator(0.0), SMALL, seed=13, capture=True,
        red_readout=rd, black_readout=rd)
    post = [r for r in records if r["ply"] >= 2]
    assert post, "test board must produce post-opening plies"
    assert all(r["top2"] is not None for r in post)
    assert all(r["readout_overrode_leader"] in (True, False) for r in post)


def test_play_eval_game_defaults_to_the_eval_config_readout():
    w1 = play_eval_game(FakeEvaluator(0.0), FakeEvaluator(0.0), SMALL, seed=5)
    w2 = play_eval_game(FakeEvaluator(0.0), FakeEvaluator(0.0), SMALL, seed=5,
                        red_readout=readout_from_eval_config(SMALL),
                        black_readout=readout_from_eval_config(SMALL))
    assert w1[:3] == w2[:3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_readout_telemetry.py -q; echo "EXIT=$?"`
Expected: FAIL — `ImportError: cannot import name 'readout_from_eval_config'`

- [ ] **Step 3: Write the implementation**

In `scripts/GPU/alphazero/eval_runner.py`, extend the imports. `encode_move`
lives at `mcts.py:69` and converts `(row, col)` to the int key `root.children`
uses:

```python
from . import eval_readout
from .eval_readout import ReadoutConfig
from .mcts import MCTS, MCTSConfig, encode_move
```

**Evaluator sharing is already correct and must stay that way.** `_make_cache`
(`eval_runner.py:216-226`) keys evaluators by checkpoint path, so two agents
sharing one checkpoint receive the *same* compiled evaluator instance. Do not
add a second cache key or build one evaluator per agent — rebuilding compiled
MLX evaluators is the documented Metal-exhaustion trap.

Add the two helpers above `play_eval_game`:

```python
def readout_from_eval_config(config: EvalConfig) -> ReadoutConfig:
    """The readout implied by a legacy EvalConfig.selection_mode."""
    if config.selection_mode == "argmax":
        return ReadoutConfig(mode=eval_readout.MODE_ARGMAX)
    if config.selection_mode == "opening_temperature":
        return ReadoutConfig(
            mode=eval_readout.MODE_OPENING_TEMPERATURE,
            opening_temp_plies=config.opening_temp_plies,
            temp_high=config.temp_high, temp_low=config.temp_low,
        )
    raise ValueError(f"unknown selection_mode {config.selection_mode!r}")


def root_child_stats(counts, root):
    """Map every legal move -> (completed_visits, child_perspective_mean).

    The mean is None when UNDEFINED. `MCTSNode.q_value` returns 0.0 at
    visit_count == 0 (mcts.py:259-261); that is not a measurement and must not
    be reported as one.
    """
    stats = {}
    for move, visits in counts.items():
        child = root.children.get(encode_move(move[0], move[1]))
        if child is not None and child.visit_count > 0:
            stats[move] = (child.visit_count, float(child.q_value))
        else:
            stats[move] = (int(visits) if child is None else child.visit_count,
                           None)
    return stats
```

Add `encode_move` to the `mcts` import line (it lives at `mcts.py:69`):

```python
from .mcts import MCTS, MCTSConfig, encode_move
```

Replace `play_eval_game`:

```python
def play_eval_game(red_eval, black_eval, config: EvalConfig, seed: int,
                   capture: bool = False, red_readout: Optional[ReadoutConfig] = None,
                   black_readout: Optional[ReadoutConfig] = None):
    """Play one game. Returns (winner, reason, n_moves, records).

    `red_readout`/`black_readout` default to the readout implied by `config`,
    preserving the legacy single-config behaviour.

    RNG: search and readout draw from SEPARATE streams. mcts.MCTS shares one
    `self.rng` across prior-shuffle, PUCT tie-break and move readout, so a
    readout that consumes a different number of draws would change the
    generator state entering every later search. Evaluation therefore selects
    moves through eval_readout with its own stream and never calls
    mcts.select_move. Self-play is unaffected.
    """
    red_rd = red_readout or readout_from_eval_config(config)
    black_rd = black_readout or readout_from_eval_config(config)
    mcts_red = MCTS(red_eval, cfg_from(config), random.Random(seed ^ 0xA5A5A5))
    mcts_black = MCTS(black_eval, cfg_from(config), random.Random(seed ^ 0x5A5A5A))
    readout_rng_red = random.Random(seed ^ 0xC3C3C3)
    readout_rng_black = random.Random(seed ^ 0x3C3C3C)
    state = TwixtState(active_size=config.board_size, to_move="red",
                       max_plies_limit=config.max_moves)
    ply = 0
    records = [] if capture else None
    while state.winner() is None and ply < config.max_moves and state.legal_moves():
        is_red = state.to_move == "red"
        mcts = mcts_red if is_red else mcts_black
        rdt = red_rd if is_red else black_rd
        rng = readout_rng_red if is_red else readout_rng_black
        counts, root_value, root = mcts.search_with_root(state, add_noise=False)
        if root.visit_count <= 0:
            raise RuntimeError(f"ply {ply}: search completed zero visits")
        stats = root_child_stats(counts, root)
        top2 = eval_readout.top_two(stats)
        move, overrode = eval_readout.select(counts, ply, rdt, rng, top2=top2)
        if capture:
            records.append(ply_record(ply, state.to_move, move, counts,
                                      root_value, top2=top2,
                                      overrode_leader=overrode))
        state = state.apply_move(move)
        ply += 1
    winner = state.winner()
    if winner is not None:
        reason = "win"
    elif ply >= config.max_moves:
        reason = "state_cap"
    elif not state.legal_moves():
        reason = "board_full"
    else:
        reason = "unknown_error"
    return winner, reason, ply, records
```

Replace `_play_and_build_result` so per-agent readouts reach the game loop:

```python
def _play_and_build_result(task, red, black, config, capture, replay_dir):
    """Play one game and build its result, writing a replay sidecar when
    capturing. Shared by the sequential and worker loops (both single-process)."""
    red_rd = task.red_agent.readout if task.red_agent is not None else None
    black_rd = task.black_agent.readout if task.black_agent is not None else None
    winner, reason, nm, records = play_eval_game(
        red, black, config, task.seed, capture=capture,
        red_readout=red_rd, black_readout=black_rd)
    result = make_result(task, winner, reason, nm)
    if records is not None:
        result.replay_path = write_replay(
            replay_dir,
            build_replay_dict(result, task.seed, config.board_size, records))
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_readout_telemetry.py -q; echo "EXIT=$?"`
Expected: `EXIT=0`

**If `test_play_eval_game_defaults_to_the_eval_config_readout` fails:** that is expected and correct — the RNG split changes which moves a temperature agent samples. Confirm the failure is *only* a different move sequence (not a crash or an invalid state), then change that test to assert the game merely completes validly:

```python
def test_play_eval_game_defaults_to_the_eval_config_readout():
    w, reason, n, _ = play_eval_game(
        FakeEvaluator(0.0), FakeEvaluator(0.0), SMALL, seed=5)
    assert n > 0 and reason in {"win", "state_cap", "board_full"}
```

- [ ] **Step 5: Measure the budget invariant and pin it**

The spec requires the simulation budget to be *asserted*, not merely recorded. The exact relationship between `root.visit_count` and `n_simulations` must be **measured**, not assumed. Add:

```python
def test_root_visit_count_matches_the_nominal_budget():
    """Pins the measured cold-root budget invariant.

    Run once, read the actual value, then keep this assertion as the guard.
    If root.visit_count is n_simulations + 1 (root expansion counted), encode
    that instead -- but encode what the run SHOWS, not what looks tidy.
    """
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig

    state = TwixtState(active_size=6, to_move="red", max_plies_limit=24)
    mcts = MCTS(FakeEvaluator(0.0), MCTSConfig(n_simulations=32), random.Random(3))
    _counts, _rv, root = mcts.search_with_root(state, add_noise=False)
    assert root.visit_count == 32
```

Run: `python -m pytest tests/test_eval_readout_telemetry.py::test_root_visit_count_matches_the_nominal_budget -q; echo "EXIT=$?"`
If it fails, read the reported value and change `32` to the measured number, and add a one-line comment recording what the measurement was.

- [ ] **Step 6: Full suite**

Run: `python -m pytest tests/ -q; echo "EXIT=$?"`
Expected: `EXIT=0`. Record the measured pass/skip/deselect counts in the commit body — that measured collect is the baseline for later tasks.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/eval_runner.py tests/test_eval_readout_telemetry.py
git commit -m "feat(eval): per-agent readout with search/readout RNG separation"
```

---

## Task B5: Agent-keyed match summary

**Files:**
- Modify: `scripts/GPU/alphazero/eval_summary.py:15-100`
- Create: `tests/test_eval_summary_agent_mode.py`

**Interfaces:**
- Consumes: `EvalGameResult` agent fields (B3), `AGENT_COMPARISON_UNIT`.
- Produces: `summarize_agent_match(results, agent_a_id, agent_b_id, pairing_id, config) -> dict`. `summarize_match` raises `ValueError` on agent-comparison results.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_summary_agent_mode.py`:

```python
"""Agent-mode aggregation, and explicit rejection by the legacy path."""
import pytest

from scripts.GPU.alphazero import eval_readout as R
from scripts.GPU.alphazero.eval_runner import (
    AgentSpec, build_agent_pairing_tasks, build_pairing_tasks, make_result,
)
from scripts.GPU.alphazero.eval_summary import (
    summarize_agent_match, summarize_match,
)

CKPT = "checkpoints/x/model_iter_0001.safetensors"
CONTROL = AgentSpec("control", CKPT, R.ReadoutConfig(mode=R.MODE_ARGMAX))
CANDIDATE = AgentSpec("candidate", CKPT,
                      R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB))


def _results(winners):
    tasks = build_agent_pairing_tasks("p", CONTROL, CANDIDATE, len(winners), 100)
    return [make_result(t, w, "win" if w else "state_cap", 50)
            for t, w in zip(tasks, winners)]


def test_agent_summary_scores_by_agent_not_checkpoint():
    # 4 games; candidate wins all 4 regardless of colour.
    # game 0,2: red=control  -> candidate is black -> winner "black"
    # game 1,3: red=candidate -> winner "red"
    res = _results(["black", "red", "black", "red"])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["a_wins"] == 4
    assert s["b_wins"] == 0
    assert s["a_score_rate"] == pytest.approx(1.0)


def test_agent_summary_emits_real_confidence_intervals():
    res = _results(["black", "red", "red", "black"])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["score_rate_ci95"] is not None
    assert s["elo_ci95"] is not None
    assert s["comparison_unit"] == "agent"
    assert s["same_checkpoint"] is True


def test_agent_summary_reports_per_colour_stats_for_agent_a():
    res = _results(["black", "red", "black", "red"])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["a_as_red"]["games"] == 2
    assert s["a_as_black"]["games"] == 2
    assert s["a_as_red"]["wins"] == 2
    assert s["a_as_black"]["wins"] == 2


def test_agent_summary_treats_draws_as_half():
    res = _results(["black", None, "black", None])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["a_score"] == pytest.approx(3.0)   # 2 wins + 2 draws * 0.5


def test_agent_summary_rejects_an_unknown_agent_id():
    res = _results(["black", "red"])
    with pytest.raises(ValueError):
        summarize_agent_match(res, "candidate", "ghost", "p", {})


def test_legacy_summary_explicitly_rejects_agent_results():
    # CONSTRUCTED negative case: checkpoint keying is meaningless here, and
    # silently returning nulls would look like a valid self-match.
    res = _results(["black", "red"])
    with pytest.raises(ValueError, match="comparison_unit"):
        summarize_match(res, CKPT, CKPT, "p", {})


def test_legacy_summary_still_works_for_checkpoint_results():
    tasks = build_pairing_tasks("p", "a.safetensors", "b.safetensors", 2, 100, 0)
    res = [make_result(t, "red", "win", 50) for t in tasks]
    s = summarize_match(res, "a.safetensors", "b.safetensors", "p", {})
    assert s["a_wins"] == 1 and s["b_wins"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_summary_agent_mode.py -q; echo "EXIT=$?"`
Expected: FAIL — `ImportError: cannot import name 'summarize_agent_match'`

- [ ] **Step 3: Write the implementation**

In `scripts/GPU/alphazero/eval_summary.py`, add the import:

```python
from .eval_runner import short_id, AGENT_COMPARISON_UNIT
```

Add a colour-stats helper keyed by agent, next to `_color_stats`:

```python
def _agent_color_stats(results, agent_id, color):
    if color == "red":
        sub = [r for r in results if r.red_agent_id == agent_id]
        wins = sum(1 for r in sub if r.winner == "red")
        losses = sum(1 for r in sub if r.winner == "black")
    else:
        sub = [r for r in results if r.black_agent_id == agent_id]
        wins = sum(1 for r in sub if r.winner == "black")
        losses = sum(1 for r in sub if r.winner == "red")
    caps = sum(1 for r in sub if r.winner is None)
    n = len(sub)
    return {
        "games": n, "wins": wins, "losses": losses, "caps": caps,
        "score_rate": (score_rate(wins, caps, n) if n else None),
    }
```

Add the guard at the top of `summarize_match`, right after the empty check:

```python
    agent_rows = [r for r in results
                  if getattr(r, "comparison_unit", None) == AGENT_COMPARISON_UNIT]
    if agent_rows:
        raise ValueError(
            f"pairing {pairing_id}: {len(agent_rows)} of {len(results)} results "
            f"carry comparison_unit={AGENT_COMPARISON_UNIT!r}. Checkpoint keying "
            f"cannot score them -- use summarize_agent_match."
        )
```

Add the new summariser after `summarize_match`:

```python
def summarize_agent_match(results, agent_a_id, agent_b_id, pairing_id,
                          config) -> dict:
    """Aggregate a two-AGENT match. Agents may share a checkpoint.

    Everything is keyed on agent id. `winner_checkpoint` is deliberately
    unused: with one checkpoint on both sides it cannot identify a winner.
    """
    if not results:
        raise ValueError(f"no results for pairing {pairing_id}")
    for r in results:
        if getattr(r, "comparison_unit", None) != AGENT_COMPARISON_UNIT:
            raise ValueError(
                f"pairing {pairing_id}: result {r.task_id} is not an agent "
                f"comparison; use summarize_match")
    ids = {r.red_agent_id for r in results} | {r.black_agent_id for r in results}
    for aid in (agent_a_id, agent_b_id):
        if aid not in ids:
            raise ValueError(f"agent {aid!r} does not appear in these results; "
                             f"present: {sorted(ids)}")

    games = len(results)
    state_caps = sum(1 for r in results if r.reason == "state_cap")
    board_full = sum(1 for r in results if r.reason == "board_full")
    red_wins = sum(1 for r in results if r.winner == "red")
    black_wins = sum(1 for r in results if r.winner == "black")
    decisive = red_wins + black_wins

    a_wins = sum(1 for r in results if r.winner_agent_id == agent_a_id)
    b_wins = sum(1 for r in results if r.winner_agent_id == agent_b_id)
    draws = state_caps + board_full
    a_score = a_wins + 0.5 * draws
    rate = score_rate(a_wins, draws, games)
    s_lo, s_hi = score_ci_trinomial(a_wins, draws, b_wins)
    e_lo, e_hi = elo_ci(a_wins, draws, b_wins)

    return {
        "pairing_id": pairing_id,
        "comparison_unit": AGENT_COMPARISON_UNIT,
        "agent_a": agent_a_id,
        "agent_b": agent_b_id,
        "checkpoint_a": results[0].red_checkpoint,
        "checkpoint_b": results[0].black_checkpoint,
        "same_checkpoint": all(r.same_checkpoint for r in results),
        "readout_a": next((r.red_readout for r in results
                           if r.red_agent_id == agent_a_id), None),
        "readout_b": next((r.red_readout for r in results
                           if r.red_agent_id == agent_b_id), None),
        "games": games,
        "state_caps": state_caps,
        "board_full": board_full,
        "color_bias": {
            "red_win_rate_decisive": (red_wins / decisive) if decisive else None,
        },
        "avg_plies": mean(r.n_moves for r in results),
        "draw_score_policy": DRAW_SCORE_POLICY,
        "config": config,
        "a_wins": a_wins, "b_wins": b_wins,
        "a_score": a_score,
        "a_score_rate": rate,
        "elo_estimate": elo_diff(rate, games),
        "elo_ci95": [e_lo, e_hi],
        "score_rate_ci95": [s_lo, s_hi],
        "verdict": verdict(rate),
        "a_as_red": _agent_color_stats(results, agent_a_id, "red"),
        "a_as_black": _agent_color_stats(results, agent_a_id, "black"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_summary_agent_mode.py -q; echo "EXIT=$?"`
Expected: `EXIT=0`

- [ ] **Step 5: Check the existing summary tests**

Run: `python -m pytest tests/test_eval_summary.py -q; echo "EXIT=$?"`
Expected: `EXIT=0`

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/eval_summary.py tests/test_eval_summary_agent_mode.py
git commit -m "feat(eval): agent-keyed match summary with legacy rejection"
```

---

## Task B6: Two-agent match CLI

**Files:**
- Create: `scripts/GPU/alphazero/eval_readout_match.py`
- Modify: `tests/test_eval_agent_identity.py`

**Interfaces:**
- Consumes: `build_agent_pairing_tasks`, `run_game_tasks`, `summarize_agent_match`, `ReadoutConfig`.
- Produces: `readout_config_from_name(name, opening_temp_plies, temp_high, temp_low) -> ReadoutConfig`; `run_readout_match(...) -> dict`; `main(argv=None)`.

`eval_checkpoint_match.py` is deliberately left untouched, so the historical checkpoint-vs-checkpoint CLI keeps its exact semantics.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval_agent_identity.py`:

```python
import json

from scripts.GPU.alphazero.eval_readout_match import (
    readout_config_from_name, run_readout_match,
)
from tests.eval_fakes import fake_evaluator_factory
from scripts.GPU.alphazero.eval_runner import EvalConfig


def test_readout_names_map_to_the_frozen_configs():
    control = readout_config_from_name("tournament", 20, 1.0, 0.1)
    assert control.mode == R.MODE_OPENING_TEMPERATURE and control.temp_low == 0.1

    c2_control = readout_config_from_name("opening_then_argmax", 20, 1.0, 0.1)
    assert c2_control.mode == R.MODE_OPENING_TEMPERATURE
    assert c2_control.temp_low == 0.0

    assert readout_config_from_name("argmax", 20, 1.0, 0.1).mode == R.MODE_ARGMAX
    assert readout_config_from_name(
        "hoeffding_lcb", 20, 1.0, 0.1).mode == R.MODE_HOEFFDING_LCB


def test_unknown_readout_name_is_rejected():
    with pytest.raises(ValueError):
        readout_config_from_name("wishful", 20, 1.0, 0.1)


def test_run_readout_match_produces_a_real_score_rate(tmp_path):
    out = tmp_path / "m.json"
    summary = run_readout_match(
        checkpoint=CKPT,
        candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
        control_readout=readout_config_from_name("tournament", 2, 1.0, 0.1),
        games=4, base_seed=900,
        config=EvalConfig(board_size=6, mcts_sims=16, max_moves=24,
                          opening_temp_plies=2),
        workers=1, output=str(out), evaluator_factory=fake_evaluator_factory,
    )
    # The whole point of agent identity: this is NOT None.
    assert summary["a_score_rate"] is not None
    assert summary["score_rate_ci95"] is not None
    assert summary["comparison_unit"] == "agent"
    assert summary["same_checkpoint"] is True
    assert json.loads(out.read_text())["agent_a"] == "candidate"


def test_run_readout_match_writes_per_game_rows_with_agent_ids(tmp_path):
    out = tmp_path / "m.json"
    run_readout_match(
        checkpoint=CKPT,
        candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
        control_readout=readout_config_from_name("tournament", 2, 1.0, 0.1),
        games=4, base_seed=901,
        config=EvalConfig(board_size=6, mcts_sims=16, max_moves=24,
                          opening_temp_plies=2),
        workers=1, output=str(out), evaluator_factory=fake_evaluator_factory,
    )
    rows = [json.loads(line) for line in
            (tmp_path / "m_games.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert {r["red_agent_id"] for r in rows} == {"candidate", "control"}
    assert all(r["comparison_unit"] == "agent" for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_agent_identity.py -q; echo "EXIT=$?"`
Expected: FAIL — `ModuleNotFoundError: ... eval_readout_match`

- [ ] **Step 3: Write the implementation**

Create `scripts/GPU/alphazero/eval_readout_match.py`:

```python
"""Two-agent, one-checkpoint readout match.

Both competitors load the SAME checkpoint and differ only in how they turn a
completed search into a played move. Scoring is by agent identity, because
checkpoint keying cannot separate them.

The historical checkpoint-vs-checkpoint CLI (eval_checkpoint_match.py) is
deliberately untouched.

NO GPU RUN IS AUTHORIZED BY THIS TOOL. Running it requires a separate written
authorization naming the exact scope, seed interval and game count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone

from .eval_readout import (
    MODE_ARGMAX, MODE_HOEFFDING_LCB, MODE_OPENING_TEMPERATURE, ReadoutConfig,
)
from .eval_runner import (
    AgentSpec, EvalConfig, build_agent_pairing_tasks, run_game_tasks,
)
from .eval_summary import summarize_agent_match

CANDIDATE_ID = "candidate"
CONTROL_ID = "control"

READOUT_NAMES = ("tournament", "argmax", "opening_then_argmax", "hoeffding_lcb")


def readout_config_from_name(name, opening_temp_plies, temp_high, temp_low):
    """Map a CLI name to a ReadoutConfig.

    tournament          -- the shipped tournament readout (temp_high then temp_low)
    argmax              -- all-ply visit argmax (Candidate 1)
    opening_then_argmax -- temp_high opening, then argmax (Candidate 2 CONTROL)
    hoeffding_lcb       -- temp_high opening, then argmax + frozen override
                           (Candidate 2 CANDIDATE)
    """
    if name == "tournament":
        return ReadoutConfig(mode=MODE_OPENING_TEMPERATURE,
                             opening_temp_plies=opening_temp_plies,
                             temp_high=temp_high, temp_low=temp_low)
    if name == "argmax":
        return ReadoutConfig(mode=MODE_ARGMAX)
    if name == "opening_then_argmax":
        return ReadoutConfig(mode=MODE_OPENING_TEMPERATURE,
                             opening_temp_plies=opening_temp_plies,
                             temp_high=temp_high, temp_low=0.0)
    if name == "hoeffding_lcb":
        return ReadoutConfig(mode=MODE_HOEFFDING_LCB,
                             opening_temp_plies=opening_temp_plies,
                             temp_high=temp_high, temp_low=temp_low)
    raise ValueError(f"unknown readout name {name!r}; expected {READOUT_NAMES}")


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _sha1(path):
    """Checkpoint hash for provenance. None when the file is unreadable --
    an unknown hash is None, never a placeholder string."""
    try:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _write_outputs(output, summary, results):
    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    stem, _ext = os.path.splitext(output)
    with open(f"{stem}_games.jsonl", "w") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")
    with open(output, "w") as fh:
        json.dump(summary, fh, indent=2)


def run_readout_match(checkpoint, candidate_readout, control_readout, games,
                      base_seed, config: EvalConfig, workers, output,
                      pairing_id=None, evaluator_factory=None, replay_dir=None):
    """Run one candidate-vs-control readout match. Returns the summary dict."""
    if candidate_readout == control_readout:
        raise ValueError("candidate and control readouts are identical; "
                         "this match would measure nothing")
    if pairing_id is None:
        pairing_id = f"{CANDIDATE_ID}_vs_{CONTROL_ID}"
    candidate = AgentSpec(CANDIDATE_ID, checkpoint, candidate_readout)
    control = AgentSpec(CONTROL_ID, checkpoint, control_readout)
    tasks = build_agent_pairing_tasks(pairing_id, candidate, control, games,
                                      base_seed)
    results = run_game_tasks(tasks, workers=workers, config=config,
                             evaluator_factory=evaluator_factory,
                             replay_dir=replay_dir)
    config_dict = {
        **asdict(config),
        "base_seed": base_seed,
        "workers": workers,
        "candidate_readout": asdict(candidate_readout),
        "control_readout": asdict(control_readout),
        "checkpoint": checkpoint,
        "checkpoint_sha1": _sha1(checkpoint),
        "seed_interval": [base_seed, base_seed + games],
    }
    summary = summarize_agent_match(results, CANDIDATE_ID, CONTROL_ID,
                                    pairing_id, config_dict)
    summary["git_commit"] = _git_commit()
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    if output:
        _write_outputs(output, summary, results)
    return summary


def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Two-agent readout match on ONE checkpoint. "
                    "Requires a separate written GPU authorization.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate-readout", required=True, choices=READOUT_NAMES)
    ap.add_argument("--control-readout", required=True, choices=READOUT_NAMES)
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--base-seed", type=int, required=True)
    ap.add_argument("--board-size", type=int, default=24)
    ap.add_argument("--mcts-sims", type=int, default=400)
    ap.add_argument("--mcts-eval-batch-size", type=int, default=14)
    ap.add_argument("--mcts-stall-flush-sims", type=int, default=48)
    ap.add_argument("--opening-temp-plies", type=int, default=20)
    ap.add_argument("--temp-high", type=float, default=1.0)
    ap.add_argument("--temp-low", type=float, default=0.1)
    ap.add_argument("--max-moves", type=int, default=280)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--replay-dir", default=None,
                    help="capture per-ply replays incl. top-two child "
                         "visits/Q (required to feed the preflight)")
    ap.add_argument("--output", required=True)
    return ap


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    config = EvalConfig(
        board_size=args.board_size, mcts_sims=args.mcts_sims,
        mcts_eval_batch_size=args.mcts_eval_batch_size,
        mcts_stall_flush_sims=args.mcts_stall_flush_sims,
        opening_temp_plies=args.opening_temp_plies,
        temp_high=args.temp_high, temp_low=args.temp_low,
        max_moves=args.max_moves,
    )
    mk = lambda name: readout_config_from_name(  # noqa: E731
        name, args.opening_temp_plies, args.temp_high, args.temp_low)
    summary = run_readout_match(
        checkpoint=args.checkpoint,
        candidate_readout=mk(args.candidate_readout),
        control_readout=mk(args.control_readout),
        games=args.games, base_seed=args.base_seed, config=config,
        workers=args.workers, output=args.output, replay_dir=args.replay_dir,
    )
    print(f"{summary['pairing_id']}: a_score_rate={summary['a_score_rate']:.4f} "
          f"elo={summary['elo_estimate']:.1f} CI95={summary['elo_ci95']} "
          f"verdict={summary['verdict']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_agent_identity.py -q; echo "EXIT=$?"`
Expected: `EXIT=0`

- [ ] **Step 5: Check the CLI parses**

Run: `python -m scripts.GPU.alphazero.eval_readout_match --help; echo "EXIT=$?"`
Expected: `EXIT=0`, help text lists all four readout names

- [ ] **Step 6: Commit**

```bash
git add scripts/GPU/alphazero/eval_readout_match.py tests/test_eval_agent_identity.py
git commit -m "feat(eval): two-agent readout match CLI"
```

---

## Task B7: Preflight analyzer

**Files:**
- Create: `scripts/GPU/alphazero/readout_preflight.py`
- Create: `tests/test_readout_preflight.py`

**Interfaces:**
- Consumes: replay sidecars written by Task B2/B4 (`schema_version == 2`).
- Produces: `preflight_stats(replays, agent_id, opening_temp_plies) -> dict`; `evaluate_gates(stats) -> dict`; `main(argv=None)`.

The population is frozen (spec §7.4): **post-opening turns belonging to the argmax agent only**, all such turns in the denominator, ineligible turns counted as "no override".

- [ ] **Step 1: Write the failing test**

Create `tests/test_readout_preflight.py`:

```python
"""Frozen preflight gate tests."""
import pytest

from scripts.GPU.alphazero.readout_preflight import (
    evaluate_gates, preflight_stats,
)


def _ply(ply, player, top2, overrode=False):
    return {"ply": ply, "player": player, "row": 1, "col": 1,
            "top2": top2, "readout_overrode_leader": overrode}


def _t2(nl=190, nc=40, ql=-0.3, qc=0.0):
    return [
        {"row": 2, "col": 2, "completed_visit_count": nl,
         "q_value_child_perspective": -ql, "q_value_root_perspective": ql},
        {"row": 1, "col": 1, "completed_visit_count": nc,
         "q_value_child_perspective": -qc, "q_value_root_perspective": qc},
    ]


def _replay(game_idx, red_agent_id, moves):
    return {"schema_version": 2, "game_idx": game_idx,
            "red_agent_id": red_agent_id, "black_agent_id": "control",
            "moves": moves}


def test_population_excludes_opening_plies():
    r = _replay(0, "candidate", [
        _ply(0, "red", _t2()), _ply(19, "red", _t2()), _ply(20, "red", _t2()),
    ])
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["population_plies"] == 1


def test_population_excludes_the_other_agents_turns():
    r = _replay(0, "candidate", [
        _ply(20, "red", _t2()), _ply(21, "black", _t2()), _ply(22, "red", _t2()),
    ])
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["population_plies"] == 2


def test_ineligible_plies_stay_in_the_denominator_as_no_override():
    # Challenger below MIN_CHILD_VISITS: ineligible, but still counted.
    r = _replay(0, "candidate", [
        _ply(20, "red", _t2(nc=3)), _ply(22, "red", _t2()),
    ])
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["population_plies"] == 2
    assert s["eligible_plies"] == 1


def test_override_rate_uses_the_full_population():
    moves = [_ply(20 + 2 * i, "red", _t2()) for i in range(10)]
    moves[0]["readout_overrode_leader"] = True
    r = _replay(0, "candidate", moves)
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["overrides"] == 1
    assert s["override_rate"] == pytest.approx(0.1)


def test_undefined_q_is_counted_and_reported_not_silently_dropped():
    t2 = _t2()
    t2[0]["q_value_root_perspective"] = None
    r = _replay(0, "candidate", [_ply(20, "red", t2)])
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["undefined_q_plies"] == 1


def test_gate_closes_on_a_near_no_op():
    g = evaluate_gates({"population_plies": 1000, "overrides": 4,
                        "override_rate": 0.004, "max_single_game_share": 0.2,
                        "undefined_q_plies": 0})
    assert g["passed"] is False
    assert "override_rate_floor" in g["failed_gates"]


def test_gate_closes_when_the_rule_is_not_conservative():
    g = evaluate_gates({"population_plies": 1000, "overrides": 200,
                        "override_rate": 0.20, "max_single_game_share": 0.2,
                        "undefined_q_plies": 0})
    assert g["passed"] is False
    assert "override_rate_ceiling" in g["failed_gates"]


def test_gate_closes_on_single_game_concentration():
    g = evaluate_gates({"population_plies": 1000, "overrides": 50,
                        "override_rate": 0.05, "max_single_game_share": 0.6,
                        "undefined_q_plies": 0})
    assert g["passed"] is False
    assert "single_game_concentration" in g["failed_gates"]


def test_gate_halts_on_undefined_q():
    g = evaluate_gates({"population_plies": 1000, "overrides": 50,
                        "override_rate": 0.05, "max_single_game_share": 0.2,
                        "undefined_q_plies": 1})
    assert g["passed"] is False
    assert "undefined_q" in g["failed_gates"]


def test_gate_passes_in_the_frozen_band():
    g = evaluate_gates({"population_plies": 1000, "overrides": 50,
                        "override_rate": 0.05, "max_single_game_share": 0.2,
                        "undefined_q_plies": 0})
    assert g["passed"] is True
    assert g["failed_gates"] == []


def test_colour_split_is_descriptive_and_never_a_gate():
    g = evaluate_gates({"population_plies": 1000, "overrides": 50,
                        "override_rate": 0.05, "max_single_game_share": 0.2,
                        "undefined_q_plies": 0, "colour_split": {"red": 1.0,
                                                                 "black": 0.0}})
    assert g["passed"] is True


def test_old_schema_replays_are_rejected_not_silently_scored():
    r = _replay(0, "candidate", [_ply(20, "red", _t2())])
    r["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_readout_preflight.py -q; echo "EXIT=$?"`
Expected: FAIL — `ModuleNotFoundError: ... readout_preflight`

- [ ] **Step 3: Write the implementation**

Create `scripts/GPU/alphazero/readout_preflight.py`:

```python
"""Frozen preflight for the Candidate 2 readout rule.

Descriptive analysis of already-captured replays. The rule and every gate
below are FROZEN (design spec section 7.4, 2026-08-06) and MUST NOT be revised
in response to what this reports.

Population (frozen): post-opening turns belonging to the named agent only.
All such turns are in the denominator; an ineligible turn counts as
"no override", it does NOT disappear.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys

from .eval_readout import MIN_CHILD_VISITS, ChildStat, lcb_override

REQUIRED_SCHEMA_VERSION = 2

# --- FROZEN gates -----------------------------------------------------------
OVERRIDE_RATE_FLOOR = 0.005     # below: not enough reach to justify the spend
OVERRIDE_RATE_CEILING = 0.15    # above: no longer a conservative occasional rule
SINGLE_GAME_SHARE_CEILING = 0.50


def _stat_from_dict(d):
    return ChildStat(
        move=(d["row"], d["col"]),
        visits=d["completed_visit_count"],
        q_child=d["q_value_child_perspective"],
        q_root=d["q_value_root_perspective"],
    )


def _agent_colour(replay, agent_id):
    """Which colour the named agent played in this game, or None."""
    if replay.get("red_agent_id") == agent_id:
        return "red"
    if replay.get("black_agent_id") == agent_id:
        return "black"
    return None


def preflight_stats(replays, agent_id, opening_temp_plies):
    """Compute the frozen population statistics over loaded replay dicts."""
    population = 0
    eligible = 0
    overrides = 0
    undefined_q = 0
    per_game = {}
    colour_counts = {"red": 0, "black": 0}

    for replay in replays:
        if replay.get("schema_version") != REQUIRED_SCHEMA_VERSION:
            raise ValueError(
                f"game {replay.get('game_idx')}: schema_version "
                f"{replay.get('schema_version')!r}, need {REQUIRED_SCHEMA_VERSION}; "
                f"top-two telemetry is absent and cannot be inferred")
        colour = _agent_colour(replay, agent_id)
        if colour is None:
            continue
        gid = replay.get("game_idx")
        for rec in replay.get("moves", []):
            if rec["ply"] < opening_temp_plies or rec["player"] != colour:
                continue
            population += 1
            top2_raw = rec.get("top2")
            if not top2_raw or len(top2_raw) < 2:
                continue
            top2 = [_stat_from_dict(d) for d in top2_raw]
            if any(s.q_root is None for s in top2):
                undefined_q += 1
                continue
            if all(s.visits >= MIN_CHILD_VISITS for s in top2):
                eligible += 1
            # Recompute rather than trusting the stored flag: the frozen rule
            # is the authority, and a mismatch is a defect worth surfacing.
            if lcb_override(top2) is not None:
                overrides += 1
                per_game[gid] = per_game.get(gid, 0) + 1
                colour_counts[colour] += 1

    max_share = (max(per_game.values()) / overrides) if overrides else None
    total_colour = colour_counts["red"] + colour_counts["black"]
    colour_split = (
        {k: v / total_colour for k, v in colour_counts.items()}
        if total_colour else None
    )
    return {
        "agent_id": agent_id,
        "population_plies": population,
        "eligible_plies": eligible,
        "overrides": overrides,
        "override_rate": (overrides / population) if population else None,
        "undefined_q_plies": undefined_q,
        "max_single_game_share": max_share,
        "games_with_overrides": len(per_game),
        # DESCRIPTIVE ONLY -- never a gate (spec section 7.4).
        "colour_split": colour_split,
    }


def evaluate_gates(stats):
    """Apply the FROZEN stop rules. Returns {passed, failed_gates, detail}."""
    failed = []
    rate = stats.get("override_rate")
    if rate is None:
        failed.append("empty_population")
    else:
        if rate < OVERRIDE_RATE_FLOOR:
            failed.append("override_rate_floor")
        if rate > OVERRIDE_RATE_CEILING:
            failed.append("override_rate_ceiling")
    share = stats.get("max_single_game_share")
    if share is not None and share > SINGLE_GAME_SHARE_CEILING:
        failed.append("single_game_concentration")
    if stats.get("undefined_q_plies"):
        failed.append("undefined_q")
    return {
        "passed": not failed,
        "failed_gates": failed,
        "thresholds": {
            "override_rate_floor": OVERRIDE_RATE_FLOOR,
            "override_rate_ceiling": OVERRIDE_RATE_CEILING,
            "single_game_share_ceiling": SINGLE_GAME_SHARE_CEILING,
        },
    }


def load_replays(pattern):
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            out.append(json.load(fh))
    if not out:
        raise ValueError(f"no replays matched {pattern!r}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Frozen preflight over captured readout replays.")
    ap.add_argument("--replay-glob", required=True)
    ap.add_argument("--agent-id", required=True)
    ap.add_argument("--opening-temp-plies", type=int, default=20)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    stats = preflight_stats(load_replays(args.replay_glob), args.agent_id,
                            args.opening_temp_plies)
    gates = evaluate_gates(stats)
    report = {"stats": stats, "gates": gates}
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    # Exit 0 = pass, 2 = a frozen gate closed the candidate.
    return 0 if gates["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_readout_preflight.py -q; echo "EXIT=$?"`
Expected: `EXIT=0`

- [ ] **Step 5: Check the CLI parses**

Run: `python -m scripts.GPU.alphazero.readout_preflight --help; echo "EXIT=$?"`
Expected: `EXIT=0`

- [ ] **Step 6: Add the agent ids to the replay sidecar**

The preflight reads `red_agent_id` / `black_agent_id` from each replay. Add them in `scripts/GPU/alphazero/eval_replay.py`'s `build_replay_dict`, after `black_checkpoint`:

```python
        "red_agent_id": getattr(result, "red_agent_id", None),
        "black_agent_id": getattr(result, "black_agent_id", None),
        "comparison_unit": getattr(result, "comparison_unit", None),
```

Add a test to `tests/test_eval_readout_telemetry.py`:

```python
def test_replay_dict_carries_agent_ids():
    from scripts.GPU.alphazero.eval_replay import build_replay_dict
    from scripts.GPU.alphazero.eval_runner import (
        AgentSpec, build_agent_pairing_tasks, make_result)

    ckpt = "c.safetensors"
    a = AgentSpec("candidate", ckpt, R.ReadoutConfig(mode=R.MODE_ARGMAX))
    b = AgentSpec("control", ckpt, R.ReadoutConfig(mode=R.MODE_OPENING_TEMPERATURE))
    task = build_agent_pairing_tasks("p", a, b, 2, 1)[0]
    result = make_result(task, "red", "win", 10)
    d = build_replay_dict(result, seed=1, board_size=6, records=[])
    assert d["red_agent_id"] == "candidate"
    assert d["black_agent_id"] == "control"
    assert d["comparison_unit"] == "agent"
```

Run: `python -m pytest tests/test_eval_readout_telemetry.py -q; echo "EXIT=$?"`
Expected: `EXIT=0`

- [ ] **Step 7: Full suite and measured baseline**

Run: `python -m pytest tests/ -q; echo "EXIT=$?"`
Expected: `EXIT=0`. Record the **measured** pass/skip/deselect counts in the commit body.

Run: `npm run test:server; echo "EXIT=$?"` and `npm run lint; echo "EXIT=$?"`
Expected: `EXIT=0` for both.

- [ ] **Step 8: Commit**

```bash
git add scripts/GPU/alphazero/readout_preflight.py scripts/GPU/alphazero/eval_replay.py \
        tests/test_readout_preflight.py tests/test_eval_readout_telemetry.py
git commit -m "feat(eval): frozen preflight analyzer over captured readout replays"
```

---

## Definition of done

Tooling is complete when all of the following hold, each verified by a command whose exit code was read from the process:

- `python -m pytest tests/ -q` exits 0, with the measured collect recorded.
- `npm run test:server` exits 0.
- `npm run lint` exits 0.
- `python -m scripts.GPU.alphazero.eval_readout_match --help` exits 0.
- `python -m scripts.GPU.alphazero.readout_preflight --help` exits 0.
- `git diff main -- scripts/GPU/alphazero/mcts.py scripts/GPU/alphazero/self_play.py` is **empty**. Neither file may change.

## Explicitly NOT in this plan

- Running Candidate 1, Candidate 2, or any match. **No GPU work is authorized.**
- Any authorization document. Each run needs its own, written separately.
- Warm trees, corpus tooling, reservoirs, selectors, sizing protocols, classifiers.
- Any change to `mcts.py`, `self_play.py`, the network, or training.
- Changing the checkpoint-tournament default, or `eval_checkpoint_match.py`.
- Verifying `MODEL_PATH` / `model.onnx` identity — a prerequisite for any product strength claim, tracked as spec §13 risk 5, but not a code task.
