/**
 * V4 (RNG): PyRandom reproduces CPython's random.Random streams exactly, for
 * the operations the protocol search uses. Fixture: generate_fixtures.py.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { PyRandom } from './py_random.js';
import { loadFixture, staleSources } from '../tests/fixtures/htb_integration/fake_evaluator.mjs';
import { pySum } from './protocol_mcts.js';

const FIX = loadFixture('rng.json');

test('the RNG fixture describes the research code as it is today', () => {
  assert.deepEqual(staleSources(FIX), []);
  assert.equal(FIX.provenance.python, '3.14.10');
});

for (const c of FIX.cases) {
  test(`seed ${c.seed}: random, getrandbits, choice, shuffle and 700 more draws match CPython`, () => {
    const r = new PyRandom(c.seed);
    assert.deepEqual([0, 1, 2, 3, 4].map(() => r.random()), c.random);
    assert.deepEqual([0, 1, 2, 5, 10, 31, 32].map((k) => r.getrandbits(k)), c.getrandbits);
    assert.deepEqual(
      [1, 2, 3, 7, 576, 577].map((n) => r.choice([...Array(n).keys()])),
      c.choice
    );
    const x = [...Array(100).keys()];
    r.shuffle(x);
    assert.deepEqual(x, c.shuffle);
    const tail = Array.from({ length: 700 }, () => r.random());
    assert.equal(tail[699], c.tail_last);
    // CPython's sum() over these 700 floats, compared by exact hex.
    const hex = (v) => {
      const b = Buffer.alloc(8);
      b.writeDoubleBE(v);
      return b.toString('hex');
    };
    const pyHex = (s) => {
      // float.hex() -> double, exactly.
      const m = /^(-?)0x([01])\.([0-9a-f]+)p([+-]\d+)$/.exec(s);
      const mant = BigInt(`0x${m[2]}${m[3].padEnd(13, '0')}`);
      return (m[1] ? -1 : 1) * Number(mant) * 2 ** (Number(m[4]) - 52);
    };
    assert.equal(hex(pySum(tail)), hex(pyHex(c.tail_sum_hex)));
  });
}

test('a different seed gives a different stream (the comparison above is not vacuous)', () => {
  assert.notEqual(new PyRandom(1).random(), new PyRandom(2).random());
});

test('seeds outside the supported range are refused', () => {
  assert.throws(() => new PyRandom(-1), TypeError);
  assert.throws(() => new PyRandom(1.5), TypeError);
  assert.throws(() => new PyRandom(2 ** 53), TypeError);
  assert.throws(() => new PyRandom(1).getrandbits(33), RangeError);
  assert.throws(() => new PyRandom(1).choice([]), RangeError);
});
