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
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

import { TwixtState } from '../../server/gameLogic.js';
import { BASELINE_MODEL_ID, C_PUCT, REPO_ROOT } from './cases.mjs';
import {
  EXIT_ERROR,
  EXIT_REFUSED,
  EXIT_SATISFIED,
  EXIT_USAGE,
  EXIT_VIOLATED,
  FALSIFICATION,
  STAGES,
  assertStageSurface,
  codeOfThrown,
  describeThrown,
  exitCodeForError,
  falsificationFixture,
  mainWithCode,
  measureCopies,
  parseArgs,
  runFalsification,
} from './falsify.mjs';

const gitClean = () =>
  execFileSync('git', ['status', '--porcelain'], { cwd: REPO_ROOT }).toString().trim() === '';

const PINNED_EAGER_SURFACE =
  '228f57b55448f44136ffd41d6f092c9da904ca469a1e7bc4055656ffd8ef77bd';

/** A model whose session records that release() was called. */
function fakeModel({ modelId, log }) {
  return {
    modelId,
    inference: {
      evaluate: async (_b, moves) => ({
        priors: new Map(moves.map((m, i) => [`${m[0]},${m[1]}`, 1 / moves.length + i * 0])),
        value: 0.1,
      }),
      session: { release: async () => log.push('release') },
    },
  };
}

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

// --- stage binding -----------------------------------------------------------

test('the eager stage is BOUND to the 74dca6e execution surface', () => {
  assert.equal(STAGES.eager.expectedSurfaceSha256, PINNED_EAGER_SURFACE);
  assert.equal(STAGES.eager.requiredOutcome, 'violated');
  assert.throws(() => {
    STAGES.eager.expectedSurfaceSha256 = '0'.repeat(64);
  }, TypeError);
});

test('there is deliberately NO lazy stage yet', () => {
  // Its digest cannot be honestly preregistered before server/mcts.js changes;
  // a placeholder would be a gate that binds nothing.
  assert.equal(STAGES.lazy, undefined);
  assert.deepEqual(Object.keys(STAGES), ['eager']);
});

test('assertStageSurface refuses a surface that is not the stage&apos;s, and accepts the right one', (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  assert.throws(
    () => assertStageSurface('0'.repeat(64)),
    (err) => err.code === 'SURFACE_MISMATCH'
  );
  const ok = assertStageSurface(PINNED_EAGER_SURFACE);
  assert.match(ok.head, /^[0-9a-f]{40}$/);
  assert.equal(ok.digest, PINNED_EAGER_SURFACE);
});

test('an unknown stage is refused', async () => {
  await assert.rejects(
    () => runFalsification({ stage: 'lazy' }),
    (err) => err.code === 'UNKNOWN_STAGE'
  );
});

// --- model identity ----------------------------------------------------------

test('a loader supplying a DIFFERENT model is refused, and the session is released', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  const log = [];
  await assert.rejects(
    () =>
      runFalsification({
        stage: 'eager',
        loadFn: async () => fakeModel({ modelId: 'c34b7ff3297c785a', log }),
      }),
    (err) => err.code === 'MODEL_ROLE'
  );
  assert.deepEqual(log, ['release'], 'the session was not released on the role-refusal path');
});

test('MODEL_ROLE is refused BEFORE the search, so no count is produced', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  const log = [];
  let evaluated = 0;
  await assert.rejects(
    () =>
      runFalsification({
        stage: 'eager',
        loadFn: async () => ({
          modelId: 'not-the-baseline',
          inference: {
            evaluate: async () => {
              evaluated += 1;
              throw new Error('should not be reached');
            },
            session: { release: async () => log.push('release') },
          },
        }),
      }),
    (err) => err.code === 'MODEL_ROLE'
  );
  assert.equal(evaluated, 0, 'the search ran despite the wrong model');
  assert.deepEqual(log, ['release']);
});

// --- thrown-value classification ---------------------------------------------

test('describeThrown and codeOfThrown handle any value', () => {
  assert.equal(describeThrown(null), 'null');
  assert.equal(describeThrown(undefined), 'undefined');
  assert.equal(describeThrown(0), '0');
  assert.equal(describeThrown(''), '');
  assert.equal(describeThrown(false), 'false');
  assert.equal(describeThrown(new Error('boom')), 'boom');
  assert.equal(describeThrown({}), '[object Object]');
  assert.equal(codeOfThrown(null), null);
  assert.equal(codeOfThrown(undefined), null);
  assert.equal(codeOfThrown(7), null);
  assert.equal(codeOfThrown({ code: 'X' }), 'X');
});

test('EVERY harness fault maps to EXIT_ERROR, never to the gate-violation code', () => {
  for (const thrown of [null, undefined, 0, '', false, new Error('x'), { code: 'WEIRD' }]) {
    const code = exitCodeForError(thrown);
    assert.equal(code, EXIT_ERROR, `${String(thrown)} did not map to EXIT_ERROR`);
    assert.notEqual(code, EXIT_VIOLATED, 'a fault was indistinguishable from a gate violation');
  }
  // Named refusals are the one distinct class.
  assert.equal(exitCodeForError({ code: 'WORKTREE_DIRTY' }), EXIT_REFUSED);
  assert.equal(exitCodeForError({ code: 'MODEL_ROLE' }), EXIT_REFUSED);
  assert.equal(exitCodeForError({ code: 'SURFACE_MISMATCH' }), EXIT_REFUSED);
});

test('the CLI returns 4 for a thrown null, undefined or frozen Error — not 1', async () => {
  const frozen = Object.freeze(new Error('frozen fault'));
  for (const thrown of [null, undefined, 0, '', frozen]) {
    const code = await mainWithCode(['--stage', 'eager'], {
      runFn: async () => {
        throw thrown;
      },
    });
    assert.equal(code, EXIT_ERROR, `throwing ${String(thrown)} produced exit ${code}`);
    assert.notEqual(code, EXIT_VIOLATED);
  }
});

test('the CLI returns 3 for a named refusal and 1 only for a real gate violation', async () => {
  assert.equal(
    await mainWithCode(['--stage', 'eager'], {
      runFn: async () => {
        throw new (class extends Error {
          code = 'WORKTREE_DIRTY';
        })('dirty');
      },
    }),
    EXIT_REFUSED
  );
  assert.equal(
    await mainWithCode(['--stage', 'eager'], {
      runFn: async () => ({
        copy_count: 4500,
        gateMaxCopies: 8,
        n_legal: 500,
        stage: 'eager',
        satisfied: false,
        required_outcome: 'violated',
      }),
    }),
    EXIT_VIOLATED
  );
  assert.equal(
    await mainWithCode(['--stage', 'eager'], {
      runFn: async () => ({
        copy_count: 8,
        gateMaxCopies: 8,
        n_legal: 500,
        stage: 'eager',
        satisfied: true,
        required_outcome: 'violated',
      }),
    }),
    EXIT_SATISFIED
  );
});

// --- CLI arguments -----------------------------------------------------------

test('parseArgs requires a known stage and rejects everything else', () => {
  assert.deepEqual(parseArgs(['--stage', 'eager']), { stage: 'eager' });
  for (const argv of [[], ['--stage'], ['--stage', 'lazy'], ['--stage', 'eager', '--stage', 'eager'], ['oops']]) {
    assert.throws(() => parseArgs(argv), (err) => err.code === 'USAGE', JSON.stringify(argv));
  }
});

test('the CLI returns 2 on a usage error', async () => {
  assert.equal(await mainWithCode([]), EXIT_USAGE);
  assert.equal(await mainWithCode(['--stage', 'nope']), EXIT_USAGE);
});

test('the verdict wording claims a copy-count gate, not a scaling law', () => {
  const src = readFileSync(join(REPO_ROOT, 'tests', 'mcts_golden', 'falsify.mjs'), 'utf8');
  // One position at one simulation count cannot establish how allocation
  // scales; that argument belongs to design section 3.
  assert.match(src, /COPY-COUNT GATE SATISFIED/);
  assert.match(src, /COPY-COUNT GATE VIOLATED/);
  const verdicts = src.slice(src.indexOf('if (result.satisfied)'));
  assert.equal(/scales with/.test(verdicts), false, 'the verdict claims a scaling law');
});

test('falsify.mjs never calls process.exit', () => {
  const src = readFileSync(join(REPO_ROOT, 'tests', 'mcts_golden', 'falsify.mjs'), 'utf8')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1');
  assert.equal(/process\.exit\s*\(/.test(src), false);
  assert.match(src, /process\.exitCode\s*=/);
});
