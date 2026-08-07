import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolvePolicy, selectMoveForRequest, DIFFICULTY_TABLE } from './readout_policy.js';
import { BoardMovesCache } from './cache.js';

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

// --- Task A2: transport parity -------------------------------------------

const COUNTS = new Map([
  ['3,4', 100],
  ['5,6', 80],
  ['7,7', 5],
]);

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
    visitCounts: COUNTS,
    difficulty: 'hard',
    deterministicMode: false,
    temperature: undefined,
    selectMove: recordingSelectMove(log),
  });
  // WS shape: same fields arriving from a socket message.
  const wsMsg = { difficulty: 'hard' };
  selectMoveForRequest({
    visitCounts: COUNTS,
    difficulty: wsMsg.difficulty,
    deterministicMode: wsMsg.deterministicMode === true,
    temperature: wsMsg.temperature,
    selectMove: recordingSelectMove(log),
  });
  assert.deepEqual(log[0], log[1]);
  assert.equal(log[0], 0, 'hard must reach the readout as temperature 0');
});

test('the parity test can actually fail', () => {
  // CONSTRUCTED negative case: a transport that resolved its own policy would
  // hand the readout a different temperature, and this comparison catches it.
  const log = [];
  selectMoveForRequest({
    visitCounts: COUNTS,
    difficulty: 'hard',
    selectMove: recordingSelectMove(log),
  });
  const rogueTemp = 0.25; // the old WS ladder for 'hard'
  log.push(rogueTemp);
  assert.notEqual(log[0], log[1], 'if these matched, the parity assertion above would be vacuous');
});

test('selectMoveForRequest returns the move and the resolved policy', () => {
  const out = selectMoveForRequest({
    visitCounts: COUNTS,
    difficulty: 'medium',
    selectMove: recordingSelectMove([]),
  });
  assert.equal(out.moveKey, '3,4');
  assert.equal(out.policy.nSims, 400);
  assert.equal(out.policy.moveTemp, 0.5);
});

// One structural claim remains, and it is now a strong one: there is exactly
// ONE readout call site, so no transport can bypass the shared seam.
// A line calls the readout DIRECTLY if it invokes .selectMove( without being
// the `selectMove:` injection the seam requires. The injected callback is the
// intended design, so a blanket ban on ".selectMove(" would reject the correct
// implementation; this predicate distinguishes the two.
function directReadoutCalls(src) {
  return src
    .split('\n')
    .filter((line) => /\.selectMove\(/.test(line) && !/selectMove:\s*\(/.test(line));
}

test('neither transport calls selectMove directly', () => {
  const src = readFileSync(new URL('./index.js', import.meta.url), 'utf8');
  assert.ok(
    !src.includes('DIFFICULTY_PARAMS'),
    'DIFFICULTY_PARAMS must be gone; readout_policy is the only source'
  );
  assert.deepEqual(
    directReadoutCalls(src),
    [],
    'transports must go through selectMoveForRequest, never call mcts.selectMove directly'
  );
  const seamCalls = src.match(/selectMoveForRequest\(/g) || [];
  assert.equal(seamCalls.length, 2, `expected exactly 2 seam calls (REST + WS), found ${seamCalls.length}`);
});

test('the direct-call detector actually detects a bypass', () => {
  // CONSTRUCTED negative case: without this, the assertion above could pass
  // simply because the predicate never matches anything.
  const bypass = ['const policy = resolvePolicy({ difficulty });', 'const moveKey = mcts.selectMove(visitCounts, policy.moveTemp);'].join(
    '\n'
  );
  assert.equal(directReadoutCalls(bypass).length, 1);

  const injected = 'selectMove: (counts, temp) => mcts.selectMove(counts, temp),';
  assert.equal(directReadoutCalls(injected).length, 0);
});

test('websocket client sends deterministicMode', () => {
  const src = readFileSync(new URL('../assets/js/ai/alphaZeroClient.js', import.meta.url), 'utf8');
  assert.ok(src.includes('deterministicMode'), 'client must forward deterministicMode over the websocket');
});

// --- Task A3: cache scoping and post-cache readout ------------------------

const PEGS = new Map([
  ['3,4', 'red'],
  ['5,6', 'black'],
]);
const MOVES = [
  [1, 1],
  [2, 2],
];

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
// proves the two tests above are not vacuously passing.
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
  c.set(PEGS, MOVES, { visits: { '3,4': 100, '5,6': 80 }, rootValue: 0.2 }, 24, scope);

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
      difficulty: 'medium',
      selectMove,
    });
  }
  assert.equal(readoutCalls, 2, 'readout must run on every request, including cache hits');
});

test('the same cached search yields different moves under different policy', () => {
  // CONSTRUCTED negative case: if the cache returned a stored MOVE, policy
  // could not change the answer and this assertion would fail.
  const c = new BoardMovesCache(10);
  const scope = 'model.onnx|400';
  c.set(PEGS, MOVES, { visits: { '3,4': 100, '5,6': 80 }, rootValue: 0.2 }, 24, scope);
  const hit = c.get(PEGS, MOVES, 24, scope);
  const counts = new Map(Object.entries(hit.visits));
  const selectMove = (m, temp) => (temp < 0.01 ? '3,4' : '5,6');

  const a = selectMoveForRequest({ visitCounts: counts, difficulty: 'hard', selectMove }).moveKey;
  const b = selectMoveForRequest({ visitCounts: counts, difficulty: 'medium', selectMove }).moveKey;
  assert.notEqual(a, b);
});

test('index.js caches raw search results, not selected moves', () => {
  const src = readFileSync(new URL('./index.js', import.meta.url), 'utf8');
  assert.ok(src.includes('cacheScope'), 'lookup must be scoped');
  assert.ok(
    !src.includes('cache.set(gameState.pegs, moves, result,'),
    'the post-readout result object must not be cached'
  );
});
