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
import { executionSurfaceDigest } from '../product_match/p_decision.mjs';
import { join } from 'node:path';
import { test } from 'node:test';

import { TwixtState } from '../../server/gameLogic.js';
import { BASELINE_MODEL_ID, C_PUCT, REPO_ROOT, assertStageSurface } from './cases.mjs';
import { stageConfig } from './cases.mjs';
import {
  EXIT_ERROR,
  EXIT_REFUSED,
  EXIT_SATISFIED,
  EXIT_USAGE,
  EXIT_VIOLATED,
  FALSIFICATION,
  STAGES,
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

const EAGER_SURFACE = '228f57b55448f44136ffd41d6f092c9da904ca469a1e7bc4055656ffd8ef77bd';
const LAZY_SURFACE = 'd7fb6bc3fbc722e306940accadc2b8bdda6c92d125710b9b22c32d31dac4c769';
const CANDIDATE_DEFAULT_SURFACE =
  'ff80f895cecd4a491e27329ba6026862bf7507852c4672108dccb73a33528047';
/**
 * HEAD carries the CANDIDATE-DEFAULT surface. Test attribution only — the
 * committed falsification records still belong to `eager` and `lazy`, and
 * nothing here re-runs a falsification.
 */
const STAGE = 'candidate-default';

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

test('every stage is frozen, with its own surface and required outcome', () => {
  assert.equal(STAGES.eager.surfaceSha256, EAGER_SURFACE);
  assert.equal(STAGES.eager.falsificationOutcome, 'violated');
  assert.equal(STAGES.lazy.surfaceSha256, LAZY_SURFACE);
  assert.equal(STAGES.lazy.falsificationOutcome, 'satisfied');
  assert.equal(STAGES['candidate-default'].surfaceSha256, CANDIDATE_DEFAULT_SURFACE);
  assert.equal(STAGES['candidate-default'].falsificationOutcome, 'satisfied');
  // No two stages may share a surface: a duplicate would let a measurement be
  // attributed to either one.
  const surfaces = Object.values(STAGES).map((s) => s.surfaceSha256);
  assert.equal(new Set(surfaces).size, surfaces.length, 'two stages share a surface digest');
  assert.throws(() => {
    STAGES.eager.surfaceSha256 = '0'.repeat(64);
  }, TypeError);
});

test('the older stages keep their own surfaces, unaffected by later ones', () => {
  // The eager and lazy corpora and falsification records are immutable and
  // remain valid under the surfaces they recorded — not under HEAD's.
  assert.equal(STAGES.eager.surfaceSha256, EAGER_SURFACE);
  assert.equal(STAGES.eager.surfaceCommit, '74dca6e1535ee1e36d640dae3ba644c6c2ed2e5e');
  assert.equal(STAGES.lazy.surfaceSha256, LAZY_SURFACE);
  assert.equal(STAGES.lazy.surfaceCommit, '85894b93392e63ce8f6e008f368ff7e798f91853');
  assert.equal(STAGES.eager.artifactSchema, 'twixt-mcts-golden/1');
  assert.equal(STAGES.eager.carriesStageField, false);
});

test('the golden suite runs its files SEQUENTIALLY', () => {
  // `node --test` runs files in parallel by default. test_capture.mjs's N1-N3
  // deliberately dirty the worktree, and several tests here and there skip when
  // the tree is dirty — so in parallel a guard test skips NONDETERMINISTICALLY,
  // and a skipped guard proves nothing. Observed as "skipped 1" with no stable
  // cause until the files were run separately.
  const pkg = JSON.parse(readFileSync(join(REPO_ROOT, 'package.json'), 'utf8'));
  assert.match(
    pkg.scripts['test:golden'],
    /--test-concurrency=1/,
    'test:golden must serialise its files, or cleanliness guards skip at random'
  );
});

test('at the CURRENT HEAD only candidate-default is accepted; eager and lazy refuse', (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  // HEAD carries the candidate-default surface, so that is the only stage that
  // may make a measurement here. Both older stages must REFUSE rather than
  // silently attribute a measurement of this code to the surface their
  // committed evidence describes — including `lazy`, which was HEAD's stage
  // until the served pin moved.
  const ok = assertStageSurface('candidate-default', executionSurfaceDigest);
  assert.match(ok.head, /^[0-9a-f]{40}$/);
  assert.equal(ok.digest, CANDIDATE_DEFAULT_SURFACE);

  for (const stale of ['eager', 'lazy']) {
    assert.throws(
      () => assertStageSurface(stale, executionSurfaceDigest),
      (err) => err.code === 'EXECUTION_SURFACE_MOVED',
      `stage "${stale}" did not refuse at the moved surface`
    );
  }
});

test('a stage NAME is required, and only a frozen name is accepted', async () => {
  assert.throws(() => stageConfig(undefined), (err) => err.code === 'UNKNOWN_STAGE');
  assert.throws(() => stageConfig('made-up'), (err) => err.code === 'UNKNOWN_STAGE');
  // A caller cannot pass a digest where a stage name belongs.
  assert.throws(() => stageConfig(EAGER_SURFACE), (err) => err.code === 'UNKNOWN_STAGE');
  // ...and there is no default at the operational entry point.
  await assert.rejects(
    () => runFalsification({}),
    (err) => err.code === 'UNKNOWN_STAGE'
  );
});

test('INHERITED property names are not stages', async () => {
  // STAGES is an ordinary object, so indexing it with these returns something
  // truthy from Object.prototype. A truthiness check would have accepted them
  // as stages and then read `undefined` surfaces off them.
  for (const name of ['toString', 'constructor', '__proto__', 'valueOf', 'hasOwnProperty']) {
    assert.ok(STAGES[name] !== undefined, `premise: STAGES[${name}] is truthy via the prototype`);
    assert.throws(
      () => stageConfig(name),
      (err) => err.code === 'UNKNOWN_STAGE',
      `stageConfig accepted the inherited name ${name}`
    );
    assert.throws(
      () => parseArgs(['--stage', name]),
      (err) => err.code === 'USAGE',
      `the CLI accepted the inherited name ${name}`
    );
    await assert.rejects(
      () => runFalsification({ stage: name }),
      (err) => err.code === 'UNKNOWN_STAGE',
      `runFalsification accepted the inherited name ${name}`
    );
  }
  // Non-strings are refused too, including one that indexes to something real.
  for (const notAName of [undefined, null, 0, {}, ['eager']]) {
    assert.throws(() => stageConfig(notAName), (err) => err.code === 'UNKNOWN_STAGE');
  }
});

test('each stage’s commit actually PRODUCES its recorded surface digest', (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  // Every pair is correct today, but nothing forced them to stay paired: a
  // typo in either field would leave the digest gate working while every
  // artifact misattributed its provenance. Re-derive from git instead.
  for (const name of Object.keys(STAGES)) {
    const config = STAGES[name];
    assert.equal(
      executionSurfaceDigest(config.surfaceCommit, REPO_ROOT),
      config.surfaceSha256,
      `stage "${name}": ${config.surfaceCommit} does not produce ${config.surfaceSha256}`
    );
  }
});

// --- model identity ----------------------------------------------------------

test('a loader supplying a DIFFERENT model is refused, and the session is released', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  const log = [];
  await assert.rejects(
    () =>
      runFalsification({
        stage: STAGE,
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
        stage: STAGE,
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
  assert.equal(exitCodeForError({ code: 'EXECUTION_SURFACE_MOVED' }), EXIT_REFUSED);
});

test('the CLI returns 4 for a thrown null, undefined or frozen Error — not 1', async () => {
  const frozen = Object.freeze(new Error('frozen fault'));
  for (const thrown of [null, undefined, 0, '', frozen]) {
    const code = await mainWithCode(['--stage', STAGE], {
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
    await mainWithCode(['--stage', STAGE], {
      runFn: async () => {
        throw new (class extends Error {
          code = 'WORKTREE_DIRTY';
        })('dirty');
      },
    }),
    EXIT_REFUSED
  );
  assert.equal(
    await mainWithCode(['--stage', STAGE], {
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
    await mainWithCode(['--stage', STAGE], {
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
  assert.deepEqual(parseArgs(['--stage', 'lazy']), { stage: 'lazy' });
  for (const argv of [[], ['--stage'], ['--stage', 'nope'], ['--stage', 'eager', '--stage', 'eager'], ['oops']]) {
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
