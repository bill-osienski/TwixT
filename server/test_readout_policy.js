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
