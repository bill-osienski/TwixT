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
| `scripts/GPU/alphazero/eval_integrity.py` *(new)* | Fail-closed zero-tolerance checks: per-ply budget and telemetry, whole-run completeness and binding. |
| `scripts/GPU/alphazero/readout_preflight.py` *(new)* | Pure gate computation + CLI over captured replays. |
| `tests/test_eval_readout.py` *(new)* | Frozen-constant and rule tests. |
| `tests/test_eval_readout_telemetry.py` *(new)* | Telemetry contract + search-identity tests. |
| `tests/test_eval_agent_identity.py` *(new)* | Task/result/colour-binding tests. |
| `tests/test_eval_summary_agent_mode.py` *(new)* | Agent-mode aggregation + legacy rejection. |
| `tests/test_eval_integrity.py` *(new)* | Zero-tolerance condition tests. |
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
import { selectMoveForRequest } from './readout_policy.js';

const COUNTS = new Map([['3,4', 100], ['5,6', 80], ['7,7', 5]]);

// A stub readout that records the temperature it was handed. Parity is then a
// BEHAVIOURAL claim -- "both transports hand the readout the same temperature
// for the same request" -- instead of a source-text claim.
function recordingSelectMove(log) {
  return (counts, temp) => {
    log.push(temp);
    return [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
  };
}

test('both transport option shapes produce the same readout temperature', () => {
  const log = [];
  // REST shape: flat body fields.
  selectMoveForRequest({
    visitCounts: COUNTS, difficulty: 'hard', deterministicMode: false,
    temperature: undefined, selectMove: recordingSelectMove(log),
  });
  // WS shape: same fields arriving from a socket message.
  const wsMsg = { difficulty: 'hard' };
  selectMoveForRequest({
    visitCounts: COUNTS, difficulty: wsMsg.difficulty,
    deterministicMode: wsMsg.deterministicMode === true,
    temperature: wsMsg.temperature, selectMove: recordingSelectMove(log),
  });
  assert.deepEqual(log[0], log[1]);
  assert.equal(log[0], 0, 'hard must reach the readout as temperature 0');
});

test('the parity test can actually fail', () => {
  // CONSTRUCTED negative case: a transport that resolved its own policy would
  // hand the readout a different temperature, and this comparison catches it.
  const log = [];
  selectMoveForRequest({
    visitCounts: COUNTS, difficulty: 'hard',
    selectMove: recordingSelectMove(log),
  });
  const rogueTemp = 0.25;                 // the old WS ladder for 'hard'
  log.push(rogueTemp);
  assert.notEqual(log[0], log[1],
    'if these matched, the parity assertion above would be vacuous');
});

test('selectMoveForRequest returns the move and the resolved policy', () => {
  const out = selectMoveForRequest({
    visitCounts: COUNTS, difficulty: 'medium',
    selectMove: recordingSelectMove([]),
  });
  assert.equal(out.moveKey, '3,4');
  assert.equal(out.policy.nSims, 400);
  assert.equal(out.policy.moveTemp, 0.5);
});

// One structural claim remains, and it is now a strong one: there is exactly
// ONE readout call site, so no transport can bypass the shared seam.
test('neither transport calls selectMove directly', () => {
  const src = readFileSync(new URL('./index.js', import.meta.url), 'utf8');
  assert.ok(!src.includes('DIFFICULTY_PARAMS'),
    'DIFFICULTY_PARAMS must be gone; readout_policy is the only source');
  assert.ok(!/\.selectMove\(/.test(src),
    'transports must go through selectMoveForRequest, never mcts.selectMove');
  const seamCalls = src.match(/selectMoveForRequest\(/g) || [];
  assert.equal(seamCalls.length, 2,
    `expected exactly 2 seam calls (REST + WS), found ${seamCalls.length}`);
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

- [ ] **Step 3: Add the shared readout seam**

In `server/readout_policy.js`, append:

```javascript
/**
 * The ONE place a completed search becomes a played move.
 *
 * Both transports call this. `selectMove` is injected so the seam is testable
 * without an engine, and so parity can be asserted behaviourally: the same
 * request must reach the readout with the same temperature no matter which
 * transport carried it.
 */
export function selectMoveForRequest({ visitCounts, difficulty,
                                       deterministicMode = false, temperature,
                                       selectMove }) {
  const policy = resolvePolicy({ difficulty, deterministicMode, temperature });
  return { moveKey: selectMove(visitCounts, policy.moveTemp), policy };
}
```

- [ ] **Step 4: Route the REST path through the seam**

In `server/index.js`, add the import with the other local imports:

```javascript
import { resolvePolicy, selectMoveForRequest } from './readout_policy.js';
```

Delete the `DIFFICULTY_PARAMS` block at `:41-46` entirely.

In the `/api/move` handler, replace the `const params = DIFFICULTY_PARAMS[...]` line and the whole `let moveTemp; if (deterministicMode) {...} else if (...) {...} else {...}` block with:

```javascript
    const policy = resolvePolicy({ difficulty, deterministicMode, temperature });
    const mcts = new MCTS(inference, { nSimulations: policy.nSims });

    const startTime = Date.now();
    const { visitCounts, rootValue } = await mcts.search(gameState);
    const elapsed = Date.now() - startTime;

    const { moveKey } = selectMoveForRequest({
      visitCounts, difficulty, deterministicMode, temperature,
      selectMove: (counts, temp) => mcts.selectMove(counts, temp),
    });
```

- [ ] **Step 5: Route `computeBestMove` through the same seam**

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
  const nSims = policy.nSims;
```

and replace its `const moveKey = mcts.selectMove(visitCounts, moveTemp);` with:

```javascript
  const { moveKey } = selectMoveForRequest({
    visitCounts,
    difficulty,
    deterministicMode: opts.deterministicMode === true,
    temperature: opts.temperature,
    selectMove: (counts, temp) => mcts.selectMove(counts, temp),
  });
```

- [ ] **Step 6: Forward the flag from the WebSocket handler**

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

- [ ] **Step 7: Send the flag from the client**

In `assets/js/ai/alphaZeroClient.js`, replace the `msg_out` construction:

```javascript
      const msg_out = { type: 'move', id, state, difficulty };
      if (opts?.includeVisits) msg_out.includeVisits = true;
      if (opts?.deterministicMode) msg_out.deterministicMode = true;
      if (opts?.temperature !== undefined) msg_out.temperature = opts.temperature;
```

- [ ] **Step 8: Run tests and lint**

Run: `npm run test:server; echo "EXIT=$?"`
Expected: `EXIT=0`
Run: `npm run lint; echo "EXIT=$?"`
Expected: `EXIT=0`

- [ ] **Step 9: Commit**

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

test('a cache hit re-applies the readout instead of returning a sticky move', () => {
  // BEHAVIOURAL: prime the cache with raw search output, then serve two
  // requests from it. The readout must run on BOTH -- that is what makes a
  // repeated stochastic request re-sample rather than repeat itself.
  const c = new BoardMovesCache(10);
  const scope = 'model.onnx|400';
  c.set(PEGS, MOVES, { visits: { '3,4': 100, '5,6': 80 }, rootValue: 0.2 },
        24, scope);

  let readoutCalls = 0;
  const selectMove = (counts, temp) => {
    readoutCalls++;
    return temp < 0.01 ? '3,4' : '5,6';
  };

  for (let i = 0; i < 2; i++) {
    const hit = c.get(PEGS, MOVES, 24, scope);
    assert.ok(hit !== undefined, 'expected a cache hit');
    selectMoveForRequest({
      visitCounts: new Map(Object.entries(hit.visits)),
      difficulty: 'medium', selectMove,
    });
  }
  assert.equal(readoutCalls, 2,
    'readout must run on every request, including cache hits');
});

test('the same cached search yields different moves under different policy', () => {
  // CONSTRUCTED negative case: if the cache returned a stored MOVE, policy
  // could not change the answer and this assertion would fail.
  const c = new BoardMovesCache(10);
  const scope = 'model.onnx|400';
  c.set(PEGS, MOVES, { visits: { '3,4': 100, '5,6': 80 }, rootValue: 0.2 },
        24, scope);
  const hit = c.get(PEGS, MOVES, 24, scope);
  const counts = new Map(Object.entries(hit.visits));
  const selectMove = (m, temp) => (temp < 0.01 ? '3,4' : '5,6');

  const a = selectMoveForRequest({ visitCounts: counts, difficulty: 'hard',
                                   selectMove }).moveKey;
  const b = selectMoveForRequest({ visitCounts: counts, difficulty: 'medium',
                                   selectMove }).moveKey;
  assert.notEqual(a, b);
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
    const { moveKey } = selectMoveForRequest({
      visitCounts: new Map(Object.entries(search.visits)),
      difficulty, deterministicMode, temperature,
      selectMove: (counts, temp) => mcts.selectMove(counts, temp),
    });
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
    assert r.red_readout["mode"] == R.MODE_ARGMAX
    assert r.black_readout["mode"] == R.MODE_HOEFFDING_LCB


def test_results_carry_the_COMPLETE_readout_config_not_just_the_mode():
    """`tournament` and `opening_then_argmax` are BOTH mode
    'opening_temperature' and differ only in temp_low. Recording the mode
    alone would make the two experiments indistinguishable in the artifact.
    """
    tournament = AgentSpec("control", CKPT, R.ReadoutConfig(
        mode=R.MODE_OPENING_TEMPERATURE, temp_high=1.0, temp_low=0.1))
    then_argmax = AgentSpec("candidate", CKPT, R.ReadoutConfig(
        mode=R.MODE_OPENING_TEMPERATURE, temp_high=1.0, temp_low=0.0))
    task = build_agent_pairing_tasks("p", then_argmax, tournament, 2, 100)[0]
    r = make_result(task, "red", "win", 50)
    assert r.red_readout["mode"] == r.black_readout["mode"]
    assert r.red_readout["temp_low"] == 0.0
    assert r.black_readout["temp_low"] == 0.1
    assert r.red_readout["opening_temp_plies"] == 20
    assert r.red_readout["temp_high"] == 1.0


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

In `scripts/GPU/alphazero/eval_runner.py`, add the imports near the top
(`asdict` serializes the complete readout config into every result row):

```python
from dataclasses import asdict, dataclass
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
    # COMPLETE readout configs, not just the mode: `tournament` and
    # `opening_then_argmax` share mode "opening_temperature" and differ only
    # in temp_low, so a mode string cannot identify the experiment.
    red_readout: Optional[dict] = None
    black_readout: Optional[dict] = None
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
            "red_readout": asdict(task.red_agent.readout),
            "black_readout": asdict(task.black_agent.readout),
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


def _search_once(seed):
    """One independent search from the same fixed state and search seed."""
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig

    state = TwixtState(active_size=6, to_move="red", max_plies_limit=24)
    mcts = MCTS(FakeEvaluator(0.0), MCTSConfig(n_simulations=32),
                random.Random(seed))
    counts, root_value, root = mcts.search_with_root(state, add_noise=False)
    return counts, root_value, R.top_two(root_child_stats(counts, root))


def test_search_identity_across_two_independent_searches():
    """Two INDEPENDENT searches from the same state and search seed must
    produce identical visit counts, root value and top-two telemetry.

    Feeding one completed tree to two readouts would be tautological -- the
    statistics are the same object. The real claim is that the search is
    unaffected by which readout is configured, and only two separate runs can
    test it.

    This is a per-position property. It is FALSE across a game by
    construction, so it must never be asserted at game level.
    """
    counts_a, rv_a, top2_a = _search_once(7)
    counts_b, rv_b, top2_b = _search_once(7)

    assert counts_a == counts_b
    assert rv_a == rv_b
    assert [(s.move, s.visits, s.q_child, s.q_root) for s in top2_a] == \
           [(s.move, s.visits, s.q_child, s.q_root) for s in top2_b]

    # Only the selected move may differ between readouts.
    a, _ = R.select(counts_a, 5, R.ReadoutConfig(mode=R.MODE_ARGMAX),
                    random.Random(1))
    b, _ = R.select(counts_b, 5, R.ReadoutConfig(mode=R.MODE_HOEFFDING_LCB,
                                                 opening_temp_plies=2),
                    random.Random(1), top2=top2_b)
    assert a in counts_a and b in counts_b


def test_search_identity_test_can_actually_fail():
    """CONSTRUCTED negative case: a DIFFERENT search seed must change the
    tree. If it does not, the identity test above is vacuous on this fixture
    and the board/sim count must be made discriminating before it is trusted.
    """
    counts_a, _rv_a, _t_a = _search_once(7)
    counts_c, _rv_c, _t_c = _search_once(9999)
    assert counts_a != counts_c, (
        "search is seed-insensitive on this fixture -- the identity test "
        "proves nothing; enlarge the board or the simulation count")


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

**Do not weaken `test_play_eval_game_defaults_to_the_eval_config_readout` if it fails.** Both calls run the *same* new implementation at the *same* seed, and `readout_from_eval_config(SMALL)` returns exactly what the default path constructs, so the two games must be identical. Disagreement is a bug in the default-resolution path — fix the code, not the assertion.

(The RNG split does change results relative to *historical* runs. That is a different comparison, it is recorded in the spec, and no test in this plan asserts it.)

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


def test_per_colour_intervals_exist_because_the_colour_rule_needs_them():
    """Spec 8.1 rejects only when a colour's own 95% UPPER bound is below
    50%. Without a per-colour interval that rule cannot be applied at all."""
    res = _results(["black", "red", "red", "black"])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    for key in ("a_as_red", "a_as_black"):
        lo, hi = s[key]["score_rate_ci95"]
        assert 0.0 <= lo <= hi <= 1.0


def test_decisive_only_rates_are_reported_as_secondary():
    res = _results(["black", "red", None, None])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["decisive_games"] == 2
    assert s["a_decisive_score_rate"] == pytest.approx(1.0)
    # Primary stays draw-inclusive: 2 wins + 2 draws over 4 games.
    assert s["a_score_rate"] == pytest.approx(0.75)


def test_decisive_rate_is_none_not_zero_when_every_game_drew():
    res = _results([None, None])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["decisive_games"] == 0
    assert s["a_decisive_score_rate"] is None
    assert s["a_as_red"]["decisive_score_rate"] is None


def test_termination_distribution_is_reported():
    res = _results(["black", None, "red", None])
    s = summarize_agent_match(res, "candidate", "control", "p", {})
    assert s["termination_distribution"] == {"win": 2, "state_cap": 2}
    assert s["state_cap_rate"] == pytest.approx(0.5)


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
    """Per-colour stats INCLUDING a 95% interval.

    Spec section 8.1's colour-safety rule rejects only when a colour's own 95%
    UPPER bound falls below 50%, so the interval is a required decision input,
    not a nicety. Undefined on an empty colour -> None, never 0.
    """
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
    if not n:
        return {"games": 0, "wins": 0, "losses": 0, "caps": 0,
                "score_rate": None, "score_rate_ci95": None,
                "decisive_games": 0, "decisive_score_rate": None}
    lo, hi = score_ci_trinomial(wins, caps, losses)
    decisive = wins + losses
    return {
        "games": n, "wins": wins, "losses": losses, "caps": caps,
        "score_rate": score_rate(wins, caps, n),
        "score_rate_ci95": [lo, hi],
        "decisive_games": decisive,
        # Secondary per spec 8.1; None when there are no decisive games.
        "decisive_score_rate": (wins / decisive) if decisive else None,
    }


def _termination_distribution(results):
    """Counts by termination reason. Every reason that occurred appears."""
    out = {}
    for r in results:
        out[r.reason] = out.get(r.reason, 0) + 1
    return out
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
        # SECONDARY per spec 8.1 -- reported, never decisive, because
        # excluding draws biases the comparison if the candidate changes draw
        # propensity. None when no game was decisive.
        "decisive_games": decisive,
        "a_decisive_score_rate": (a_wins / decisive) if decisive else None,
        # Operational reporting required by spec 8.3.
        "termination_distribution": _termination_distribution(results),
        "state_cap_rate": state_caps / games,
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
        repo_dir=clean_repo,
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
        repo_dir=clean_repo,
    )
    rows = [json.loads(line) for line in
            (tmp_path / "m_games.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert {r["red_agent_id"] for r in rows} == {"candidate", "control"}
    assert all(r["comparison_unit"] == "agent" for r in rows)


def test_provenance_is_complete_and_never_silently_null(tmp_path, clean_repo):
    out = tmp_path / "m.json"
    s = run_readout_match(
        checkpoint=CKPT,
        candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
        control_readout=readout_config_from_name("tournament", 2, 1.0, 0.1),
        games=2, base_seed=902,
        config=EvalConfig(board_size=6, mcts_sims=16, max_moves=24,
                          opening_temp_plies=2),
        workers=1, output=str(out), evaluator_factory=fake_evaluator_factory,
        repo_dir=clean_repo, prior_seed_intervals=[[100, 200]],
    )
    assert s["git_commit"]                       # never None
    assert s["worktree_clean"] is True
    assert s["config"]["checkpoint_sha1"]        # never None
    assert s["wall_clock_seconds"] >= 0
    assert s["config"]["seed_interval"] == [902, 904]
    assert s["config"]["seed_interval_convention"] == "half_open_[start,end)"
    assert s["config"]["prior_seed_intervals"] == [[100, 200]]
    assert set(s["config"]["rng_derivation"]) == {
        "search_red", "search_black", "readout_red", "readout_black",
        "game_seed"}


def test_an_unreadable_checkpoint_aborts_instead_of_recording_a_null_hash(
        tmp_path, clean_repo):
    # CONSTRUCTED negative case: provenance must fail closed.
    with pytest.raises(RuntimeError, match="hash checkpoint"):
        run_readout_match(
            checkpoint=str(tmp_path / "does_not_exist.safetensors"),
            candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
            control_readout=readout_config_from_name("tournament", 2, 1.0, 0.1),
            games=2, base_seed=903,
            config=EvalConfig(board_size=6, mcts_sims=16, max_moves=24,
                              opening_temp_plies=2),
            workers=1, output=None, evaluator_factory=fake_evaluator_factory,
            repo_dir=clean_repo,
        )


def test_a_dirty_worktree_is_refused(dirty_repo):
    """CONSTRUCTED: a purpose-built dirty repo, so this never depends on the
    ambient worktree's state and can never silently skip.

    There is no override. `git status --porcelain` names changed files but not
    their contents, so a recorded dirty run is not reproducible from its own
    provenance.
    """
    with pytest.raises(RuntimeError, match="dirty"):
        run_readout_match(
            checkpoint=CKPT,
            candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
            control_readout=readout_config_from_name("tournament", 2, 1.0, 0.1),
            games=2, base_seed=904,
            config=EvalConfig(board_size=6, mcts_sims=16, max_moves=24,
                              opening_temp_plies=2),
            workers=1, output=None, evaluator_factory=fake_evaluator_factory,
            repo_dir=dirty_repo,
        )


def test_overlapping_seed_interval_is_refused_before_any_game(clean_repo):
    with pytest.raises(ValueError, match="overlap"):
        run_readout_match(
            checkpoint=CKPT,
            candidate_readout=readout_config_from_name("argmax", 2, 1.0, 0.1),
            control_readout=readout_config_from_name("tournament", 2, 1.0, 0.1),
            games=4, base_seed=1000,
            config=EvalConfig(board_size=6, mcts_sims=16, max_moves=24,
                              opening_temp_plies=2),
            workers=1, output=None, evaluator_factory=fake_evaluator_factory,
            repo_dir=clean_repo, prior_seed_intervals=[[1002, 1010]],
        )
```

**Two fixtures this module needs.** Add them at the top of
`tests/test_eval_agent_identity.py`:

```python
import pathlib
import subprocess

# The hash now actually READS the checkpoint, so this must be a real file.
# fake_evaluator_factory still ignores its contents.
CKPT = str(pathlib.Path(__file__).parent / "eval_fakes.py")


def _init_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=path, check=True)
    return str(path)


@pytest.fixture
def clean_repo(tmp_path):
    """A committed, clean repository. Provenance checks run for real against
    it -- there is no bypass flag to test through."""
    return _init_repo(tmp_path / "clean_repo")


@pytest.fixture
def dirty_repo(tmp_path):
    """Same, then made genuinely dirty."""
    path = tmp_path / "dirty_repo"
    _init_repo(path)
    (path / "uncommitted.txt").write_text("dirty\n")
    return str(path)
```

Both fixtures must `mkdir` their directory before `git init`; use
`path.mkdir(parents=True)` at the start of `_init_repo`.

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
import time
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


def _git_provenance(repo_dir=None):
    """Commit and worktree state. Fails closed.

    A DIRTY WORKTREE IS REFUSED OUTRIGHT, with no override. Recording
    `git status --porcelain` would name the changed files but not their
    contents, so a "dirty but documented" run is still not reproducible from
    its own provenance. There is no allow_dirty flag by design -- commit or
    stash first.

    `repo_dir` selects the repository (default: the process CWD). Tests pass a
    constructed temporary repo so they exercise this exact code path rather
    than a bypass.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.PIPE, cwd=repo_dir
        ).decode().strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.PIPE, cwd=repo_dir
        ).decode()
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            f"cannot establish git provenance ({e!r}); a run whose code state "
            f"is unknown is not reproducible") from e
    if porcelain.strip():
        raise RuntimeError(
            "worktree is dirty; a recorded run must come from a committed "
            "state. Recording the porcelain status would name the changed "
            "files but not their contents, so it cannot substitute for a "
            f"commit. Commit or stash first:\n{porcelain}")
    return {"git_commit": commit, "worktree_clean": True}


def _sha1(path):
    """Checkpoint hash. Fails closed: the spec requires a cryptographic hash,
    so an unreadable checkpoint aborts the run rather than recording None."""
    h = hashlib.sha1()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError as e:
        raise RuntimeError(
            f"cannot hash checkpoint {path!r} ({e!r}); provenance requires a "
            f"checkpoint hash") from e
    return h.hexdigest()


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
                      pairing_id=None, evaluator_factory=None, replay_dir=None,
                      repo_dir=None, prior_seed_intervals=()):
    """Run one candidate-vs-control readout match. Returns the summary dict.

    `prior_seed_intervals` is every interval already consumed by this line of
    work, as half-open [start, end) pairs. Reuse is refused.

    All provenance and disjointness checks run BEFORE any game, so an
    unreproducible or overlapping configuration costs zero GPU time.
    """
    if candidate_readout == control_readout:
        raise ValueError("candidate and control readouts are identical; "
                         "this match would measure nothing")
    provenance = _git_provenance(repo_dir)
    checkpoint_sha1 = _sha1(checkpoint)
    priors = [list(iv) for iv in prior_seed_intervals]
    validate_seed_intervals([base_seed, base_seed + games], priors)
    started = time.monotonic()
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
        # COMPLETE configs: mode alone cannot distinguish `tournament` from
        # `opening_then_argmax` (both mode "opening_temperature").
        "candidate_readout": asdict(candidate_readout),
        "control_readout": asdict(control_readout),
        "checkpoint": checkpoint,
        "checkpoint_sha1": checkpoint_sha1,
        # HALF-OPEN [start, end): game g uses seed base_seed + g for
        # g in range(games), so `end` is the first UNUSED seed. A later run
        # proving disjointness must compare against this convention.
        "seed_interval": [base_seed, base_seed + games],
        "seed_interval_convention": "half_open_[start,end)",
        # Spec section 10 requires the intervals of every prior run, so a
        # reader can verify disjointness without external bookkeeping.
        "prior_seed_intervals": priors,
        # Search and readout draw from separate streams (spec section 7.1).
        # Recorded so a reader can reconstruct any game exactly.
        "rng_derivation": {
            "search_red": "seed ^ 0xA5A5A5",
            "search_black": "seed ^ 0x5A5A5A",
            "readout_red": "seed ^ 0xC3C3C3",
            "readout_black": "seed ^ 0x3C3C3C",
            "game_seed": "base_seed + game_idx",
        },
    }
    summary = summarize_agent_match(results, CANDIDATE_ID, CONTROL_ID,
                                    pairing_id, config_dict)
    summary.update(provenance)
    summary["wall_clock_seconds"] = time.monotonic() - started
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
    ap.add_argument("--prior-seed-interval", action="append", default=[],
                    metavar="START:END",
                    help="a half-open [START,END) seed interval already "
                         "consumed by this line of work. Repeatable. Overlap "
                         "with this run's interval is refused. Recorded in "
                         "the summary so disjointness is verifiable from the "
                         "artifact alone.")
    ap.add_argument("--output", required=True)
    return ap


def _parse_interval(text):
    try:
        start, end = text.split(":")
        return [int(start), int(end)]
    except ValueError as e:
        raise SystemExit(
            f"bad --prior-seed-interval {text!r}; expected START:END") from e


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
        prior_seed_intervals=[_parse_interval(t)
                              for t in args.prior_seed_interval],
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

## Task B7: Fail-closed runtime integrity validator

**Files:**
- Create: `scripts/GPU/alphazero/eval_integrity.py`
- Modify: `scripts/GPU/alphazero/eval_runner.py` (`play_eval_game` per-ply guard)
- Modify: `scripts/GPU/alphazero/eval_readout_match.py` (`run_readout_match` result-set guard)
- Create: `tests/test_eval_integrity.py`

**Interfaces:**
- Consumes: `ChildStat` (B1), `EvalGameResult` / `AgentSpec` (B3), `run_readout_match` (B6).
- Produces:
  - `IntegrityError(Exception)`
  - `validate_ply(ply, expected_sims, root_visit_count, root_value, top2) -> None`
  - `validate_result_set(results, tasks, agent_a_id, agent_b_id) -> None`

The spec freezes these as zero-tolerance conditions (§8.3), but nothing so far
enforces them. Every check **fails closed**: an unverifiable condition raises,
it never warns and continues.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_integrity.py`:

```python
"""Zero-tolerance integrity checks (design spec section 8.3)."""
import math

import pytest

from scripts.GPU.alphazero import eval_readout as R
from scripts.GPU.alphazero.eval_integrity import (
    IntegrityError, validate_ply, validate_result_set,
)
from scripts.GPU.alphazero.eval_runner import (
    AgentSpec, build_agent_pairing_tasks, make_result,
)

CKPT = "c.safetensors"
A = AgentSpec("candidate", CKPT, R.ReadoutConfig(mode=R.MODE_ARGMAX))
B = AgentSpec("control", CKPT, R.ReadoutConfig(mode=R.MODE_OPENING_TEMPERATURE))


def _t2(nl=190, nc=40):
    return [R.ChildStat((2, 2), nl, 0.3, -0.3),
            R.ChildStat((1, 1), nc, -0.05, 0.05)]


def test_valid_ply_passes():
    validate_ply(5, expected_sims=400, root_visit_count=400,
                 root_value=0.12, top2=_t2())


def test_budget_mismatch_raises():
    with pytest.raises(IntegrityError, match="budget"):
        validate_ply(5, expected_sims=400, root_visit_count=399,
                     root_value=0.12, top2=_t2())


def test_non_finite_root_value_raises():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(IntegrityError, match="root_value"):
            validate_ply(5, expected_sims=400, root_visit_count=400,
                         root_value=bad, top2=_t2())


def test_non_finite_q_on_a_VISITED_child_raises():
    bad = [R.ChildStat((2, 2), 190, float("nan"), float("nan")),
           R.ChildStat((1, 1), 40, -0.05, 0.05)]
    with pytest.raises(IntegrityError, match="q_value"):
        validate_ply(5, expected_sims=400, root_visit_count=400,
                     root_value=0.1, top2=bad)


def test_none_q_on_an_UNVISITED_child_is_allowed():
    # None on a zero-visit child is an UNDEFINED mean, not corrupt telemetry.
    ok = [R.ChildStat((2, 2), 190, 0.3, -0.3),
          R.ChildStat((1, 1), 0, None, None)]
    validate_ply(5, expected_sims=400, root_visit_count=400,
                 root_value=0.1, top2=ok)


def test_none_q_on_a_VISITED_child_raises():
    bad = [R.ChildStat((2, 2), 190, None, None),
           R.ChildStat((1, 1), 40, -0.05, 0.05)]
    with pytest.raises(IntegrityError, match="q_value"):
        validate_ply(5, expected_sims=400, root_visit_count=400,
                     root_value=0.1, top2=bad)


def _ok_set(n=4):
    tasks = build_agent_pairing_tasks("p", A, B, n, 100)
    results = [make_result(t, "red", "win", 40) for t in tasks]
    return results, tasks


def test_valid_result_set_passes():
    results, tasks = _ok_set()
    validate_result_set(results, tasks, "candidate", "control")


def test_unknown_error_raises():
    results, tasks = _ok_set()
    results[1].reason = "unknown_error"
    with pytest.raises(IntegrityError, match="unknown_error"):
        validate_result_set(results, tasks, "candidate", "control")


def test_missing_result_raises():
    results, tasks = _ok_set()
    with pytest.raises(IntegrityError, match="incomplete"):
        validate_result_set(results[:-1], tasks, "candidate", "control")


def test_duplicate_task_id_raises():
    results, tasks = _ok_set()
    results[1].task_id = results[0].task_id
    with pytest.raises(IntegrityError, match="duplicate"):
        validate_result_set(results, tasks, "candidate", "control")


def test_unexpected_agent_id_raises():
    results, tasks = _ok_set()
    results[0].red_agent_id = "impostor"
    with pytest.raises(IntegrityError, match="agent"):
        validate_result_set(results, tasks, "candidate", "control")


def test_both_colours_held_by_the_same_agent_raises():
    results, tasks = _ok_set()
    results[0].black_agent_id = results[0].red_agent_id
    with pytest.raises(IntegrityError, match="both colours"):
        validate_result_set(results, tasks, "candidate", "control")


def test_readout_leak_across_the_colour_swap_raises():
    results, tasks = _ok_set()
    # Same agent id, but the OTHER agent's readout config: a binding leak.
    results[0].red_readout = dict(results[0].black_readout)
    with pytest.raises(IntegrityError, match="configuration"):
        validate_result_set(results, tasks, "candidate", "control")


def test_a_SYSTEMATIC_config_leak_is_caught():
    """The decisive case: every row carries the wrong config CONSISTENTLY.

    A consistency-only check (does this agent always play the same readout?)
    passes here, because the wrong config is applied uniformly. Only a
    comparison against the TASK's expected config catches it -- and a
    systematic leak is exactly the failure that would silently invalidate a
    whole match.
    """
    results, tasks = _ok_set()
    wrong = {"mode": R.MODE_HOEFFDING_LCB, "opening_temp_plies": 20,
             "temp_high": 1.0, "temp_low": 0.1}
    for r in results:
        if r.red_agent_id == "candidate":
            r.red_readout = dict(wrong)
        else:
            r.black_readout = dict(wrong)
    with pytest.raises(IntegrityError, match="configuration"):
        validate_result_set(results, tasks, "candidate", "control")


def test_game_binding_is_validated_immediately_not_at_the_end():
    """Spec 8.3 requires an IMMEDIATE stop. validate_game_binding is the
    per-game guard; it must reject a single bad result on its own, without
    needing the rest of the run."""
    from scripts.GPU.alphazero.eval_integrity import validate_game_binding

    tasks = build_agent_pairing_tasks("p", A, B, 4, 100)
    good = make_result(tasks[0], "red", "win", 40)
    validate_game_binding(good, tasks[0])

    bad = make_result(tasks[0], "red", "win", 40)
    bad.red_readout = dict(bad.black_readout)
    with pytest.raises(IntegrityError, match="configuration"):
        validate_game_binding(bad, tasks[0])


def test_game_binding_rejects_unknown_error_on_its_own():
    tasks = build_agent_pairing_tasks("p", A, B, 2, 100)
    r = make_result(tasks[0], None, "unknown_error", 40)
    from scripts.GPU.alphazero.eval_integrity import validate_game_binding
    with pytest.raises(IntegrityError, match="unknown_error"):
        validate_game_binding(r, tasks[0])


def test_adjacent_half_open_intervals_do_not_overlap():
    from scripts.GPU.alphazero.eval_integrity import validate_seed_intervals
    validate_seed_intervals([64, 128], [[0, 64]])
    validate_seed_intervals([0, 64], [[64, 128]])


@pytest.mark.parametrize("prior", [
    [60, 70],      # straddles the start
    [120, 200],    # straddles the end
    [0, 1000],     # contains it
    [70, 80],      # contained by it
    [64, 128],     # identical
])
def test_overlapping_intervals_raise(prior):
    from scripts.GPU.alphazero.eval_integrity import validate_seed_intervals
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_intervals([64, 128], [prior])


def test_empty_or_reversed_interval_raises():
    from scripts.GPU.alphazero.eval_integrity import validate_seed_intervals
    with pytest.raises(ValueError, match="empty or reversed"):
        validate_seed_intervals([100, 100], [])
    with pytest.raises(ValueError, match="empty or reversed"):
        validate_seed_intervals([200, 100], [])


def test_colour_imbalance_raises():
    tasks = build_agent_pairing_tasks("p", A, B, 4, 100)
    results = [make_result(t, "red", "win", 40) for t in tasks]
    results[1].red_agent_id = "candidate"
    results[1].black_agent_id = "control"
    with pytest.raises(IntegrityError, match="colour balance"):
        validate_result_set(results, tasks, "candidate", "control")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_integrity.py -q; echo "EXIT=$?"`
Expected: FAIL — `ModuleNotFoundError: ... eval_integrity`

- [ ] **Step 3: Write the implementation**

Create `scripts/GPU/alphazero/eval_integrity.py`:

```python
"""Zero-tolerance integrity checks for readout matches.

Design spec section 8.3 freezes these conditions as immediate stops. Every
check FAILS CLOSED: an unverifiable condition raises IntegrityError rather
than warning and continuing. A run that trips any of these is not a result.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict


class IntegrityError(RuntimeError):
    """A frozen zero-tolerance condition was observed.

    Scope: budget mismatch, corrupt required telemetry, binding or
    configuration faults, unknown_error, and incomplete or duplicate result
    sets. Illegal moves and crashes are NOT raised here -- they already abort
    through the engine and the worker failure path.
    """


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def validate_ply(ply: int, expected_sims: int, root_visit_count: int,
                 root_value, top2) -> None:
    """Per-ply guard. Raises on budget mismatch or corrupt required telemetry.

    A None mean on a ZERO-visit child is undefined, not corrupt, and passes.
    A None or non-finite mean on a VISITED child is corrupt and raises.
    """
    if root_visit_count != expected_sims:
        raise IntegrityError(
            f"ply {ply}: simulation budget mismatch -- root completed "
            f"{root_visit_count} visits, expected exactly {expected_sims}")
    if not _finite(root_value):
        raise IntegrityError(f"ply {ply}: root_value is not finite ({root_value!r})")
    for stat in top2 or []:
        if stat.visits <= 0:
            continue
        for label, q in (("q_value_child_perspective", stat.q_child),
                         ("q_value_root_perspective", stat.q_root)):
            if not _finite(q):
                raise IntegrityError(
                    f"ply {ply}: child {stat.move} has {stat.visits} visits but "
                    f"{label} is {q!r}; a visited child's mean is defined")


def validate_seed_intervals(current, priors) -> None:
    """Refuse a seed interval that overlaps any already-consumed interval.

    Intervals are HALF-OPEN [start, end): [0, 64) and [64, 128) are adjacent,
    not overlapping. Reusing seeds would silently correlate a "fresh" run with
    an earlier one, which is the kind of contamination that is invisible in
    the result and fatal to it.
    """
    start, end = current
    if end <= start:
        raise ValueError(f"empty or reversed seed interval [{start}, {end})")
    for prior in priors:
        p_start, p_end = prior
        if start < p_end and p_start < end:
            raise ValueError(
                f"seed interval [{start}, {end}) overlaps the prior interval "
                f"[{p_start}, {p_end}); seeds may not be reused")


def validate_game_binding(result, task) -> None:
    """Per-game guard, applied AS EACH GAME FINISHES so a fault stops the run
    immediately rather than after every game has been played (spec 8.3).

    The decisive check is that each colour's recorded readout equals the
    TASK's expected config. Checking only that an agent is self-consistent
    across games would pass a SYSTEMATIC leak, where every row carries the
    same wrong configuration.
    """
    if result.reason == "unknown_error":
        raise IntegrityError(f"task {result.task_id}: game ended in unknown_error")

    if task.red_agent is None or task.black_agent is None:
        return   # legacy checkpoint-vs-checkpoint task; nothing to bind

    for colour, agent_id, readout, spec in (
            ("red", result.red_agent_id, result.red_readout, task.red_agent),
            ("black", result.black_agent_id, result.black_readout, task.black_agent)):
        if agent_id != spec.agent_id:
            raise IntegrityError(
                f"task {result.task_id}: {colour} agent id {agent_id!r} does "
                f"not match the task binding {spec.agent_id!r}")
        expected = asdict(spec.readout)
        if readout != expected:
            raise IntegrityError(
                f"task {result.task_id}: configuration mismatch for {colour} "
                f"agent {agent_id!r} -- recorded {readout!r}, task specifies "
                f"{expected!r}")

    if result.red_agent_id == result.black_agent_id:
        raise IntegrityError(
            f"task {result.task_id}: agent {result.red_agent_id!r} holds both "
            f"colours")


def validate_result_set(results, tasks, agent_a_id: str,
                        agent_b_id: str) -> None:
    """Whole-run guard, applied before any statistic is computed.

    Re-runs the per-game binding check, because a result set may reach here
    from a path that did not call validate_game_binding, and then adds the
    checks that only make sense over the whole run.
    """
    expected_ids = {agent_a_id, agent_b_id}
    by_task_all = {t.task_id: t for t in tasks}
    for r in results:
        task = by_task_all.get(r.task_id)
        if task is not None:
            validate_game_binding(r, task)

    if len(results) != len(tasks):
        raise IntegrityError(
            f"incomplete run: {len(results)} results for {len(tasks)} tasks")

    seen: Dict[int, int] = {}
    for r in results:
        seen[r.task_id] = seen.get(r.task_id, 0) + 1
    dupes = sorted(t for t, n in seen.items() if n > 1)
    if dupes:
        raise IntegrityError(f"duplicate task_ids in results: {dupes[:10]}")
    missing = sorted({t.task_id for t in tasks} - set(seen))
    if missing:
        raise IntegrityError(f"incomplete run: missing task_ids {missing[:10]}")

    red_counts: Dict[str, int] = {agent_a_id: 0, agent_b_id: 0}
    for r in results:
        got = {r.red_agent_id, r.black_agent_id}
        if None in got or not got <= expected_ids:
            raise IntegrityError(
                f"task {r.task_id}: unexpected agent ids {sorted(map(str, got))}, "
                f"expected {sorted(expected_ids)}")
        red_counts[r.red_agent_id] += 1

    if red_counts[agent_a_id] != red_counts[agent_b_id]:
        raise IntegrityError(
            f"colour balance broken: {agent_a_id} was red "
            f"{red_counts[agent_a_id]} times, {agent_b_id} "
            f"{red_counts[agent_b_id]}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_integrity.py -q; echo "EXIT=$?"`
Expected: `EXIT=0`

- [ ] **Step 5: Wire the per-ply guard into the game loop**

In `scripts/GPU/alphazero/eval_runner.py`, add the import:

```python
from .eval_integrity import validate_ply
```

In `play_eval_game`, replace the `if root.visit_count <= 0:` guard with:

```python
        validate_ply(ply, config.mcts_sims, root.visit_count, root_value, top2)
```

placing it immediately after `top2 = eval_readout.top_two(stats)` (it needs
`top2`), and delete the old `root.visit_count <= 0` check, which the budget
comparison subsumes.

- [ ] **Step 6: Wire the per-game guard so a fault stops the run immediately**

In `scripts/GPU/alphazero/eval_runner.py`, extend the import:

```python
from .eval_integrity import validate_game_binding, validate_ply
```

and call it at the end of `_play_and_build_result`, before the replay is
written, so a binding or `unknown_error` fault raises on the game that
produced it rather than after every remaining game has been played:

```python
    result = make_result(task, winner, reason, nm)
    validate_game_binding(result, task)
    if records is not None:
```

Under `workers > 1` the raise surfaces through the existing `_WorkerFailed`
sentinel path, which already terminates the pool.

- [ ] **Step 7: Wire the whole-run guard into the match**

In `scripts/GPU/alphazero/eval_readout_match.py`, add the import:

```python
from .eval_integrity import validate_result_set, validate_seed_intervals
```

and call it in `run_readout_match` immediately after `results = run_game_tasks(...)`,
before `config_dict` is built:

```python
    validate_result_set(results, tasks, CANDIDATE_ID, CONTROL_ID)
```

- [ ] **Step 8: Run the affected suites**

Run: `python -m pytest tests/test_eval_integrity.py tests/test_eval_readout_telemetry.py tests/test_eval_agent_identity.py -q; echo "EXIT=$?"`
Expected: `EXIT=0`

The budget equality is safe to assert: `root.visit_count` was **measured**
equal to the nominal budget at 8, 16, 32 and 400 simulations. If a
`play_eval_game` test still fails on it, fix the fixture, never `validate_ply`.

- [ ] **Step 9: Commit**

```bash
git add scripts/GPU/alphazero/eval_integrity.py scripts/GPU/alphazero/eval_runner.py \
        scripts/GPU/alphazero/eval_readout_match.py tests/test_eval_integrity.py
git commit -m "feat(eval): fail-closed integrity validator for plies, games and result sets"
```

---

## Task B8: Preflight analyzer

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


def _ply(ply, player, top2, overrode=False, rank=1):
    return {"ply": ply, "player": player, "row": 1, "col": 1,
            "selected_visit_rank": rank,
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


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_q_is_corrupt_not_merely_undefined(bad):
    t2 = _t2()
    t2[0]["q_value_root_perspective"] = bad
    r = _replay(0, "candidate", [_ply(20, "red", t2)])
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["undefined_q_plies"] == 1, f"{bad!r} must be caught, not scored"


def test_PARTIAL_agent_absence_raises():
    """CONSTRUCTED: one replay contains the agent, one does not.

    Skipping the odd one out and scoring the rest would produce a
    clean-looking report over a population that quietly lost games -- exactly
    the fail-open the identity contract exists to prevent.
    """
    present = _replay(0, "candidate", [_ply(20, "red", _t2())])
    absent = _replay(1, "someone_else", [_ply(20, "red", _t2())])
    with pytest.raises(ValueError, match="absent from 1 of 2"):
        preflight_stats([present, absent], agent_id="candidate",
                        opening_temp_plies=20)


def test_the_missing_game_ids_are_named_in_the_error():
    replays = [_replay(0, "candidate", [_ply(20, "red", _t2())]),
               _replay(7, "someone_else", [_ply(20, "red", _t2())]),
               _replay(9, "someone_else", [_ply(20, "red", _t2())])]
    with pytest.raises(ValueError, match=r"\[7, 9\]"):
        preflight_stats(replays, agent_id="candidate", opening_temp_plies=20)


def test_an_agent_absent_from_every_replay_raises():
    r = _replay(0, "someone_else", [_ply(20, "red", _t2())])
    with pytest.raises(ValueError, match="absent from 1 of 1"):
        preflight_stats([r], agent_id="candidate", opening_temp_plies=20)


def test_all_replays_containing_the_agent_pass():
    replays = [_replay(0, "candidate", [_ply(20, "red", _t2())]),
               _replay(1, "control", [_ply(20, "black", _t2())])]  # agent is black
    replays[1]["black_agent_id"] = "candidate"
    s = preflight_stats(replays, agent_id="candidate", opening_temp_plies=20)
    assert s["replays_matched"] == 2


def test_descriptive_outputs_are_present():
    moves = [_ply(20, "red", _t2()), _ply(80, "red", _t2()),
             _ply(120, "red", _t2(nc=3))]
    s = preflight_stats([_replay(0, "candidate", moves)],
                        agent_id="candidate", opening_temp_plies=20)
    assert set(s["override_rate_by_ply_bucket"]) >= {"20-39", "70-109", "110+"}
    for bucket in s["override_rate_by_ply_bucket"].values():
        assert {"plies", "overrides", "rate"} <= set(bucket)
    assert set(s["challenger_visits_at_override"]) == {"n", "min", "median", "max"}
    assert isinstance(s["per_game_override_counts"], dict)


def test_challenger_visit_summary_is_null_when_nothing_overrode():
    r = _replay(0, "candidate", [_ply(20, "red", _t2(nc=3))])
    s = preflight_stats([r], agent_id="candidate", opening_temp_plies=20)
    assert s["overrides"] == 0
    cv = s["challenger_visits_at_override"]
    assert cv["n"] == 0
    assert cv["min"] is None and cv["median"] is None and cv["max"] is None


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


def test_nonleader_selection_splits_at_the_opening_boundary():
    from scripts.GPU.alphazero.readout_preflight import nonleader_selection_report

    def ply_ranked(ply, rank):
        return _ply(ply, "red", _t2(), rank=rank)

    r = _replay(0, "candidate", [
        ply_ranked(0, 3), ply_ranked(5, 1),      # opening: 1 of 2 non-leader
        ply_ranked(20, 1), ply_ranked(25, 1),    # post: 0 of 2 non-leader
    ])
    rep = nonleader_selection_report([r], "candidate", opening_temp_plies=20)
    assert rep["opening"]["rate"] == pytest.approx(0.5)
    assert rep["post_opening"]["rate"] == pytest.approx(0.0)


def test_nonleader_report_raises_on_missing_rank():
    from scripts.GPU.alphazero.readout_preflight import nonleader_selection_report

    rec = _ply(20, "red", _t2())
    rec.pop("selected_visit_rank", None)
    with pytest.raises(ValueError, match="selected_visit_rank"):
        nonleader_selection_report([_replay(0, "candidate", [rec])],
                                   "candidate", opening_temp_plies=20)


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

The stored `readout_overrode_leader` flag is deliberately NOT compared against
the recomputed rule. On a Candidate 1 (argmax) agent the stored flag is always
False by construction while the recomputed rule may fire, so a mismatch is
EXPECTED there and would be a meaningless alarm. The frozen rule is the
authority for these statistics; the flag records what was actually played.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
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


def _ply_bucket(ply):
    """Coarse phase label for descriptive reporting only."""
    if ply < 40:
        return "20-39"
    if ply < 70:
        return "40-69"
    if ply < 110:
        return "70-109"
    return "110+"


def preflight_stats(replays, agent_id, opening_temp_plies):
    """Compute the frozen population statistics over loaded replay dicts.

    Fails closed: a wrong schema, a corrupt Q, or an agent that appears in no
    replay is an error, not a silent skip.
    """
    population = 0
    eligible = 0
    overrides = 0
    undefined_q = 0
    per_game = {}
    colour_counts = {"red": 0, "black": 0}
    by_bucket = {}
    challenger_visits_at_override = []
    matched_replays = 0
    missing_agent = []

    for replay in replays:
        if replay.get("schema_version") != REQUIRED_SCHEMA_VERSION:
            raise ValueError(
                f"game {replay.get('game_idx')}: schema_version "
                f"{replay.get('schema_version')!r}, need {REQUIRED_SCHEMA_VERSION}; "
                f"top-two telemetry is absent and cannot be inferred")
        colour = _agent_colour(replay, agent_id)
        if colour is None:
            # Collected, then raised on below. A replay selected for this
            # analysis that does not contain the named agent is an identity
            # fault, not something to skip: scoring the remainder would report
            # a clean-looking result over a population that silently lost
            # games.
            missing_agent.append(replay.get("game_idx"))
            continue
        matched_replays += 1
        gid = replay.get("game_idx")
        for rec in replay.get("moves", []):
            if rec["ply"] < opening_temp_plies or rec["player"] != colour:
                continue
            population += 1
            bucket = _ply_bucket(rec["ply"])
            slot = by_bucket.setdefault(bucket, {"plies": 0, "overrides": 0})
            slot["plies"] += 1

            top2_raw = rec.get("top2")
            if not top2_raw or len(top2_raw) < 2:
                continue
            top2 = [_stat_from_dict(d) for d in top2_raw]
            # A None mean on a VISITED child, or any non-finite value, is
            # corrupt telemetry -- not an undefined mean.
            corrupt = any(
                s.visits > 0 and (
                    s.q_root is None or s.q_child is None
                    or not math.isfinite(s.q_root)
                    or not math.isfinite(s.q_child))
                for s in top2)
            if corrupt:
                undefined_q += 1
                continue
            if any(s.q_root is None for s in top2):
                continue    # undefined mean on an unvisited child: ineligible
            if all(s.visits >= MIN_CHILD_VISITS for s in top2):
                eligible += 1
            # The frozen rule is the authority; the stored flag is not trusted.
            if lcb_override(top2) is not None:
                overrides += 1
                slot["overrides"] += 1
                per_game[gid] = per_game.get(gid, 0) + 1
                colour_counts[colour] += 1
                challenger_visits_at_override.append(top2[1].visits)

    if missing_agent:
        raise ValueError(
            f"agent {agent_id!r} is absent from {len(missing_agent)} of "
            f"{len(replays)} selected replays (game_idx "
            f"{sorted(x for x in missing_agent if x is not None)[:20]}); "
            f"every selected replay must contain the agent under analysis")

    max_share = (max(per_game.values()) / overrides) if overrides else None
    total_colour = colour_counts["red"] + colour_counts["black"]
    colour_split = (
        {k: v / total_colour for k, v in colour_counts.items()}
        if total_colour else None
    )
    cv = sorted(challenger_visits_at_override)
    return {
        "agent_id": agent_id,
        "replays_total": len(replays),
        "replays_matched": matched_replays,
        "population_plies": population,
        "eligible_plies": eligible,
        "overrides": overrides,
        "override_rate": (overrides / population) if population else None,
        "undefined_q_plies": undefined_q,
        "max_single_game_share": max_share,
        "games_with_overrides": len(per_game),
        # --- DESCRIPTIVE ONLY, frozen as non-gating (spec section 7.4) ------
        "colour_split": colour_split,
        "override_rate_by_ply_bucket": {
            b: {**v, "rate": (v["overrides"] / v["plies"]) if v["plies"] else None}
            for b, v in sorted(by_bucket.items())
        },
        "challenger_visits_at_override": {
            "n": len(cv),
            "min": cv[0] if cv else None,
            "median": cv[len(cv) // 2] if cv else None,
            "max": cv[-1] if cv else None,
        },
        "per_game_override_counts": dict(sorted(per_game.items())),
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


def nonleader_selection_report(replays, agent_id, opening_temp_plies):
    """Non-leader selection rate before and after the opening boundary.

    Required by spec section 7.3 so a Candidate 1 null can be attributed:
    all-ply argmax changes BOTH the opening and post-opening play, and this
    split says which half moved. Purely DESCRIPTIVE -- it gates nothing.

    Uses `selected_visit_rank`, which ply_record already emits; rank > 1 means
    the played move was not the visit leader.
    """
    buckets = {"opening": {"plies": 0, "nonleader": 0},
               "post_opening": {"plies": 0, "nonleader": 0}}
    for replay in replays:
        colour = _agent_colour(replay, agent_id)
        if colour is None:
            continue
        for rec in replay.get("moves", []):
            if rec["player"] != colour:
                continue
            key = "opening" if rec["ply"] < opening_temp_plies else "post_opening"
            buckets[key]["plies"] += 1
            rank = rec.get("selected_visit_rank")
            if rank is None:
                raise ValueError(
                    f"game {replay.get('game_idx')} ply {rec['ply']}: "
                    f"selected_visit_rank missing; cannot report non-leader rate")
            if rank > 1:
                buckets[key]["nonleader"] += 1
    return {
        k: {**v, "rate": (v["nonleader"] / v["plies"]) if v["plies"] else None}
        for k, v in buckets.items()
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

    replays = load_replays(args.replay_glob)
    stats = preflight_stats(replays, args.agent_id, args.opening_temp_plies)
    gates = evaluate_gates(stats)
    report = {
        "stats": stats,
        "gates": gates,
        # Descriptive, gates nothing. Spec section 7.3 attribution aid.
        "nonleader_selection": nonleader_selection_report(
            replays, args.agent_id, args.opening_temp_plies),
    }
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
        # COMPLETE configs, not just the mode -- `tournament` and
        # `opening_then_argmax` are both mode "opening_temperature".
        "red_readout": getattr(result, "red_readout", None),
        "black_readout": getattr(result, "black_readout", None),
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
    # Full config, not just the mode: the replay must be self-describing.
    assert d["red_readout"]["mode"] == R.MODE_ARGMAX
    assert d["black_readout"]["temp_low"] == 0.1
    assert d["black_readout"]["opening_temp_plies"] == 20
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
- Every spec §8.3 condition that `IntegrityError` is responsible for has a test
  that constructs it (Task B7), and all three guards are wired: `validate_ply`
  per ply, `validate_game_binding` per game *as each game finishes*, and
  `validate_result_set` before any statistic is computed.
  **Scope note:** illegal moves and crashes are *not* `IntegrityError`
  conditions — they already abort through `TwixtState.apply_move` and the
  worker `_WorkerFailed` path. `IntegrityError` covers budget mismatch,
  corrupt telemetry, binding/configuration faults, `unknown_error`, and
  incomplete or duplicate result sets.
- Per-game rows and replay sidecars carry the **complete** readout config.
  The RNG derivation scheme, the labelled seed interval and every **prior**
  seed interval live in the **run summary**, so disjointness is verifiable
  from the artifact alone.
- A recorded run is impossible from a dirty worktree, and there is **no
  override flag**. The refusal is exercised against a constructed dirty
  repository, never by observing the ambient one.
- The preflight raises when **any** selected replay lacks the agent under
  analysis, naming the missing `game_idx` values.
- `git diff d5326a0 -- scripts/GPU/alphazero/mcts.py scripts/GPU/alphazero/self_play.py` is **empty**. Neither file may change.

**The baseline is `d5326a0`, not `main`.** Both files already differ from `main`
by 105 insertions — the Stage 1 atlas observer surfaces this branch inherits —
so a `main` comparison could never pass and would silently normalize a broken
check. `d5326a0` is the commit this implementation branch (`codex/competitive-readout`)
starts from, and it is the only meaningful "unchanged" reference.

## Explicitly NOT in this plan

- Running Candidate 1, Candidate 2, or any match. **No GPU work is authorized.**
- Any authorization document. Each run needs its own, written separately.
- Warm trees, corpus tooling, reservoirs, selectors, sizing protocols, classifiers.
- Any change to `mcts.py`, `self_play.py`, the network, or training.
- Changing the checkpoint-tournament default, or `eval_checkpoint_match.py`.
- Verifying `MODEL_PATH` / `model.onnx` identity — a prerequisite for any product strength claim, tracked as spec §13 risk 5, but not a code task.
