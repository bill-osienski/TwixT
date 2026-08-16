#!/usr/bin/env node
/**
 * Tests for the golden-trace capture harness.
 *
 * Two groups:
 *   - STRUCTURAL: the 92-case matrix is exactly what §4.2 freezes, and the
 *     enumeration is pure and immutable.
 *   - CLEANLINESS (N1-N3): the guard actually refuses, and refuses BEFORE any
 *     fixture is read or any model is loaded.
 *
 * The N-tests deliberately dirty the real worktree and restore it in `finally`.
 * They must therefore not run concurrently with anything else that asserts
 * cleanliness — hence their own `npm run test:golden` script. Node runs tests
 * within one file sequentially, which is what makes the dirty/restore windows
 * safe here.
 *
 * Every N-test carries a POSITIVE CONTROL: without it, a `runCase` that threw
 * unconditionally would satisfy all three and the suite would prove nothing.
 */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { execFileSync } from 'node:child_process';
import { readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

import {
  BASELINE_MODEL_ID,
  CANDIDATE_MODEL_ID,
  EXPECTED_CASE_COUNT,
  FIXTURE_RELDIR,
  N_SIMULATIONS_LADDER,
  PINNED_EXECUTION_SURFACE_SHA256,
  REPO_ROOT,
  caseById,
  enumerateCases,
  enumeratePositions,
  prefixesFor,
} from './cases.mjs';
import { runCase } from './worker.mjs';

const OUT_DIR = join(REPO_ROOT, 'runs', 'mcts_golden_test');
const CAPTURE_MJS = join(REPO_ROOT, 'tests', 'mcts_golden', 'capture.mjs');
const WORKER_MJS = join(REPO_ROOT, 'tests', 'mcts_golden', 'worker.mjs');

const gitClean = () =>
  execFileSync('git', ['status', '--porcelain'], { cwd: REPO_ROOT })
    .toString()
    .trim() === '';

/** Count calls without doing the work, so ordering can be observed. */
function countingSeams(counters) {
  return {
    readFixture: () => {
      counters.readFixture += 1;
      throw new Error('readFixture should not have been reached');
    },
    captureTrace: () => {
      counters.captureTrace += 1;
      throw new Error('captureTrace should not have been reached');
    },
  };
}

// --- structural --------------------------------------------------------------

test('the corpus is exactly 92 cases', () => {
  assert.equal(enumerateCases().length, EXPECTED_CASE_COUNT);
  assert.equal(EXPECTED_CASE_COUNT, 92);
});

test('the 16 positions match the literal table frozen in the specification', () => {
  // Written out rather than derived, so a change to `prefixesFor` is caught
  // instead of silently redefining the corpus it is supposed to produce.
  const expected = [
    ['P01', 'timing_00_opening_200.json', 4],
    ['P02', 'timing_00_opening_200.json', 16],
    ['P03', 'timing_00_opening_200.json', 28],
    ['P04', 'timing_00_opening_200.json', 38],
    ['P05', 'timing_01_opening_201.json', 4],
    ['P06', 'timing_01_opening_201.json', 16],
    ['P07', 'timing_01_opening_201.json', 28],
    ['P08', 'timing_01_opening_201.json', 50],
    ['P09', 'timing_02_opening_202.json', 4],
    ['P10', 'timing_02_opening_202.json', 16],
    ['P11', 'timing_02_opening_202.json', 28],
    ['P12', 'timing_02_opening_202.json', 56],
    ['P13', 'timing_03_opening_203.json', 4],
    ['P14', 'timing_03_opening_203.json', 16],
    ['P15', 'timing_03_opening_203.json', 28],
    ['P16', 'timing_03_opening_203.json', 53],
  ];
  const actual = enumeratePositions().map((p) => [p.id, p.sidecar, p.prefixPlies]);
  assert.deepEqual(actual, expected);
});

test('the immediate-win positions are exactly P04, P08, P12, P16', () => {
  const winners = enumeratePositions()
    .filter((p) => p.immediateWin)
    .map((p) => p.id);
  assert.deepEqual(winners, ['P04', 'P08', 'P12', 'P16']);
});

test('the prefix rule is {4, 16, 28, plyCount - 1}', () => {
  assert.deepEqual(prefixesFor(57), [4, 16, 28, 56]);
  assert.deepEqual(prefixesFor(39), [4, 16, 28, 38]);
});

test('the matrix splits 80 baseline / 10 candidate / 2 abort', () => {
  const cases = enumerateCases();
  const golden = cases.filter((c) => c.kind === 'golden');
  const aborts = cases.filter((c) => c.kind === 'abort');
  assert.equal(golden.filter((c) => c.modelId === BASELINE_MODEL_ID).length, 80);
  assert.equal(golden.filter((c) => c.modelId === CANDIDATE_MODEL_ID).length, 10);
  assert.equal(aborts.length, 2);
});

test('the candidate runs on exactly P02 and P11, at every simulation count', () => {
  const positions = enumerateCases()
    .filter((c) => c.modelId === CANDIDATE_MODEL_ID)
    .map((c) => c.position.id);
  assert.deepEqual([...new Set(positions)].sort(), ['P02', 'P11']);
  for (const id of ['P02', 'P11']) {
    const sims = enumerateCases()
      .filter((c) => c.modelId === CANDIDATE_MODEL_ID && c.position.id === id)
      .map((c) => c.nSimulations);
    assert.deepEqual(sims, [...N_SIMULATIONS_LADDER]);
  }
});

test('every baseline position carries the full simulation ladder', () => {
  for (const position of enumeratePositions()) {
    const sims = enumerateCases()
      .filter(
        (c) =>
          c.kind === 'golden' &&
          c.modelId === BASELINE_MODEL_ID &&
          c.position.id === position.id
      )
      .map((c) => c.nSimulations);
    assert.deepEqual(sims, [...N_SIMULATIONS_LADDER], position.id);
  }
});

test('the abort cases are A1 and A2, on P07, at 64 simulations', () => {
  const aborts = enumerateCases().filter((c) => c.kind === 'abort');
  assert.deepEqual(
    aborts.map((c) => [c.caseId, c.position.id, c.nSimulations, c.trigger]),
    [
      ['A1', 'P07', 64, 'already_aborted'],
      ['A2', 'P07', 64, 'progress_done_5'],
    ]
  );
});

test('case ids are unique', () => {
  const ids = enumerateCases().map((c) => c.caseId);
  assert.equal(new Set(ids).size, ids.length);
});

test('the enumeration is frozen — mutation throws, and the result is unchanged', () => {
  const cases = enumerateCases();
  assert.throws(() => {
    cases[0].nSimulations = 999;
  }, TypeError);
  assert.throws(() => {
    cases.push({});
  }, TypeError);
  // Behavioural, not by inspection: a fresh enumeration is still correct.
  assert.equal(enumerateCases()[0].nSimulations, N_SIMULATIONS_LADDER[0]);
  assert.equal(enumerateCases().length, EXPECTED_CASE_COUNT);
});

test('`list` is pure: it runs and prints 92 cases', () => {
  const proc = spawnSync(process.execPath, [CAPTURE_MJS, 'list'], { encoding: 'utf8' });
  assert.equal(proc.status, 0, proc.stderr);
  assert.match(proc.stdout, /92 cases \(expected 92\)/);
});

// --- cleanliness: N1-N3 ------------------------------------------------------

test('POSITIVE CONTROL: on a clean worktree a dry run reaches the fixture', async (t) => {
  if (!gitClean()) {
    t.skip('worktree is dirty for unrelated reasons; control cannot run');
    return;
  }
  rmSync(OUT_DIR, { recursive: true, force: true });
  const testCase = caseById('G_P01_baseline_s1');
  let captureTraceCalls = 0;

  const artifact = await runCase(
    { testCase, outDir: OUT_DIR, dryRun: true },
    {
      captureTrace: () => {
        captureTraceCalls += 1;
        throw new Error('dry-run must not load a model');
      },
    }
  );

  assert.equal(artifact.status, 'dry-run');
  assert.equal(artifact.trace, null);
  assert.equal(captureTraceCalls, 0, 'dry-run must not reach captureTrace');
  assert.equal(artifact.case_id, 'G_P01_baseline_s1');
  assert.equal(artifact.execution_surface_sha256, PINNED_EXECUTION_SURFACE_SHA256);
  assert.equal(artifact.fixture.prefix_plies, 4);
  assert.equal(artifact.fixture.ply_after_prefix, 4);
  assert.ok(artifact.fixture.n_legal > 400, `n_legal=${artifact.fixture.n_legal}`);
  assert.equal(typeof artifact.pid, 'number');
});

/** Dirty one path, assert refusal without touching fixtures or models, restore. */
async function expectRefusalWhileDirty(t, dirty, clean) {
  if (!gitClean()) {
    t.skip('worktree is dirty for unrelated reasons; guard test cannot run');
    return;
  }
  const testCase = caseById('G_P01_baseline_s1');
  const counters = { readFixture: 0, captureTrace: 0 };
  try {
    dirty();
    assert.equal(gitClean(), false, 'the perturbation did not dirty the worktree');
    await assert.rejects(
      () => runCase({ testCase, outDir: OUT_DIR, dryRun: true }, countingSeams(counters)),
      (err) => err.code === 'WORKTREE_DIRTY'
    );
  } finally {
    clean();
  }
  assert.equal(gitClean(), true, 'the worktree was not restored');
  assert.equal(counters.readFixture, 0, 'refused only AFTER reading fixtures');
  assert.equal(counters.captureTrace, 0, 'refused only AFTER loading a model');
}

test('N1: a modified capture harness refuses, fixtures unread', async (t) => {
  const original = readFileSync(CAPTURE_MJS);
  await expectRefusalWhileDirty(
    t,
    () => writeFileSync(CAPTURE_MJS, Buffer.concat([original, Buffer.from('\n// N1\n')])),
    () => writeFileSync(CAPTURE_MJS, original)
  );
});

test('N2: a modified fixture sidecar refuses, fixtures unread', async (t) => {
  const sidecar = join(REPO_ROOT, FIXTURE_RELDIR, 'timing_00_opening_200.json');
  const original = readFileSync(sidecar);
  await expectRefusalWhileDirty(
    t,
    () => writeFileSync(sidecar, Buffer.concat([original, Buffer.from('\n')])),
    () => writeFileSync(sidecar, original)
  );
});

test('N3: a stray untracked file refuses, fixtures unread', async (t) => {
  const stray = join(REPO_ROOT, 'tests', 'mcts_golden', 'N3_stray.tmp');
  await expectRefusalWhileDirty(
    t,
    () => writeFileSync(stray, 'stray\n'),
    () => rmSync(stray, { force: true })
  );
});

test('N1 and N2 are independent: each dirties a different file', () => {
  // Guards against a future edit that made both tests perturb the same path,
  // which would let one satisfy the other and halve the coverage.
  assert.notEqual(CAPTURE_MJS, join(REPO_ROOT, FIXTURE_RELDIR, 'timing_00_opening_200.json'));
});

// --- per-case process isolation ---------------------------------------------

test('each case runs in its own process: three cases, three distinct pids', (t) => {
  if (!gitClean()) {
    t.skip('worktree is dirty for unrelated reasons');
    return;
  }
  rmSync(OUT_DIR, { recursive: true, force: true });
  const ids = ['G_P01_baseline_s1', 'G_P05_baseline_s1', 'G_P09_baseline_s1'];
  const pids = new Set();

  for (const id of ids) {
    const proc = spawnSync(
      process.execPath,
      [WORKER_MJS, id, OUT_DIR, '--dry-run'],
      { encoding: 'utf8' }
    );
    assert.equal(proc.status, 0, proc.stderr);
    const artifact = JSON.parse(readFileSync(join(OUT_DIR, `${id}.json`), 'utf8'));
    assert.equal(artifact.status, 'dry-run');
    pids.add(artifact.pid);
  }

  assert.equal(pids.size, ids.length, 'cases shared a process');
  assert.ok(!pids.has(process.pid), 'a case ran inside the test process');
});

test('the worker refuses an unknown case id', () => {
  const proc = spawnSync(process.execPath, [WORKER_MJS, 'G_NOPE_baseline_s1', OUT_DIR], {
    encoding: 'utf8',
  });
  assert.equal(proc.status, 2);
  assert.match(proc.stderr, /UNKNOWN_CASE/);
});
