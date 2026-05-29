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
  const state = playRandomGame(seed, BOARD_SIZE, 250);  // reach the dense regime the bug lived in
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
