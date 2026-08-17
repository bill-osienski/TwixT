#!/usr/bin/env node
/**
 * Tests for the falsification harness's own logic.
 *
 * These all PASS and belong in the ordinary suite. The falsification ITSELF is
 * not run here: against the installed eager implementation it is supposed to
 * report a gate violation, so running it in the ordinary suite would make a
 * correct outcome look like a broken build.
 *
 * Nothing here loads a model.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

import { TwixtState } from '../../server/gameLogic.js';
import { BASELINE_MODEL_ID, C_PUCT, REPO_ROOT } from './cases.mjs';
import {
  EXIT_SATISFIED,
  EXIT_VIOLATED,
  FALSIFICATION,
  falsificationFixture,
  measureCopies,
} from './falsify.mjs';

test('the falsification parameters are exactly those frozen in §5', () => {
  assert.deepEqual({ ...FALSIFICATION }, {
    positionId: 'P11',
    sidecar: 'timing_02_opening_202.json',
    prefixPlies: 28,
    modelId: BASELINE_MODEL_ID,
    nSimulations: 8,
    cPuct: 1.5,
    gateMaxCopies: 8,
  });
  assert.equal(FALSIFICATION.cPuct, C_PUCT);
});

test('the gate is the bound the algorithm claims: S, not 2*(1+S)', () => {
  // 2*(1+S) = 18 would permit more than twice the copies the lazy design
  // predicts, and would still pass an implementation materialising a spare
  // child per simulation.
  assert.equal(FALSIFICATION.gateMaxCopies, FALSIFICATION.nSimulations);
  assert.notEqual(FALSIFICATION.gateMaxCopies, 2 * (1 + FALSIFICATION.nSimulations));
});

test('the parameters are frozen against mutation', () => {
  assert.throws(() => {
    FALSIFICATION.gateMaxCopies = 9999;
  }, TypeError);
  assert.equal(FALSIFICATION.gateMaxCopies, 8);
});

test('the frozen fixture resolves, is P11 at prefix 28, and has >= 400 legal moves', () => {
  const { position, describe } = falsificationFixture();
  assert.equal(position.id, 'P11');
  assert.equal(position.prefixPlies, 28);
  assert.equal(describe.prefix_plies, 28);
  assert.equal(describe.ply_after_prefix, 28);
  // §5 needs L >= 400 for the eager side to miss the gate by orders of magnitude.
  assert.ok(describe.n_legal >= 400, `n_legal=${describe.n_legal}`);
  assert.equal(describe.n_legal, 500, 'legal-move count at ply 28 on this board');
});

test('measureCopies counts exactly the copies made inside the window', async () => {
  const { state } = falsificationFixture();
  const legal = state.legalMoves();
  const n = await measureCopies(async () => {
    // Each applyMove performs exactly one copy().
    let s = state;
    for (let i = 0; i < 5; i++) s = s.applyMove(legal[i]);
  });
  assert.equal(n, 5);
});

test('copies made BEFORE the spy is installed are not counted', async () => {
  // This is why the prefix is replayed before installation rather than
  // subtracted afterwards: the replay of 28 plies performs 28 copies, and none
  // of them may enter the count.
  const { state } = falsificationFixture();
  const legal = state.legalMoves();
  const before = state.applyMove(legal[0]); // one copy, outside the window
  const n = await measureCopies(async () => {
    before.applyMove(before.legalMoves()[0]);
  });
  assert.equal(n, 1, 'a pre-window copy leaked into the count');
});

test('measureCopies counts zero when nothing is constructed', async () => {
  const n = await measureCopies(async () => {});
  assert.equal(n, 0);
});

test('the prototype is restored to the ORIGINAL function, even when the window throws', async () => {
  const original = TwixtState.prototype.copy;

  const n = await measureCopies(async () => {});
  assert.equal(n, 0);
  assert.equal(TwixtState.prototype.copy, original, 'not restored after a clean window');

  await assert.rejects(
    () =>
      measureCopies(async () => {
        throw new Error('window exploded');
      }),
    /window exploded/
  );
  assert.equal(TwixtState.prototype.copy, original, 'not restored after a throwing window');
});

test('a leaked patch would be visible: two windows do not accumulate', async () => {
  const { state } = falsificationFixture();
  const legal = state.legalMoves();
  const one = async () => {
    state.applyMove(legal[0]);
  };
  assert.equal(await measureCopies(one), 1);
  assert.equal(await measureCopies(one), 1, 'the previous window leaked its counter');
});

test('the gate arithmetic: 8 satisfies, 9 violates', () => {
  const satisfied = (c) => c <= FALSIFICATION.gateMaxCopies;
  assert.equal(satisfied(0), true);
  assert.equal(satisfied(8), true);
  assert.equal(satisfied(9), false);
  assert.equal(satisfied(4500), false, 'an eager-scale count must violate');
  assert.equal(EXIT_SATISFIED, 0);
  assert.equal(EXIT_VIOLATED, 1);
});

test('the falsification CLI is NOT part of the ordinary suite', () => {
  // While the eager implementation is installed the harness is SUPPOSED to
  // report a violation, so wiring the CLI into test:golden would make a correct
  // outcome look like a broken build.
  //
  // The distinction is the CLI versus its logic tests: `mcts_golden/falsify.mjs`
  // is the harness, `mcts_golden/test_falsify.mjs` is the tests for it and
  // BELONGS in the suite. A substring check for "falsify.mjs" cannot tell them
  // apart, so the directory-qualified path is used.
  const CLI = 'mcts_golden/falsify.mjs';
  const pkg = JSON.parse(readFileSync(join(REPO_ROOT, 'package.json'), 'utf8'));

  for (const [name, script] of Object.entries(pkg.scripts)) {
    if (name === 'falsify') continue;
    assert.equal(
      script.includes(CLI),
      false,
      `npm script "${name}" would run the falsification CLI`
    );
  }
  assert.ok(pkg.scripts.falsify?.includes(CLI), 'a dedicated falsify script must exist');

  // ...and the logic tests must actually be in the ordinary suite, or this file
  // never runs and every check in it is decorative.
  assert.ok(
    pkg.scripts['test:golden']?.includes('mcts_golden/test_falsify.mjs'),
    'test:golden must run the falsification LOGIC tests'
  );
});

test('falsify.mjs never calls process.exit', () => {
  const src = readFileSync(join(REPO_ROOT, 'tests', 'mcts_golden', 'falsify.mjs'), 'utf8')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1');
  assert.equal(/process\.exit\s*\(/.test(src), false);
  assert.match(src, /process\.exitCode\s*=/);
});
