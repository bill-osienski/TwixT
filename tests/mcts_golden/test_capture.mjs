#!/usr/bin/env node
/**
 * Tests for the golden-trace capture harness.
 *
 * Groups:
 *   - STRUCTURAL: the 92-case matrix is exactly what §4.2 freezes.
 *   - CLI: the mode is explicit and every other argument shape is rejected.
 *   - CLEANLINESS (N1-N3) and COMMIT: guards refuse BEFORE fixtures or models.
 *   - CLOBBER: existing evidence survives, including under concurrency.
 *   - VALIDATION: semantics, not container types, judged against independently
 *     derived targets.
 *   - INTEGRATION: the REAL MCTS, driven by a deterministic fake inference, so
 *     the producer/validator seam is exercised without ONNX.
 *
 * The N-tests deliberately dirty the real worktree and restore it in `finally`,
 * so they must not run concurrently with anything else asserting cleanliness —
 * hence their own `npm run test:golden`. Node runs tests within one file
 * sequentially, which is what makes those windows safe.
 *
 * Every guard test carries a POSITIVE CONTROL: without one, a `runCase` that
 * threw unconditionally would satisfy them all and prove nothing.
 */
import assert from 'node:assert/strict';
import { execFileSync, spawn, spawnSync } from 'node:child_process';
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

import {
  BASELINE_MODEL_ID,
  CANDIDATE_MODEL_ID,
  EXPECTED_CASE_COUNT,
  FIXTURE_RELDIR,
  N_SIMULATIONS_LADDER,
  STAGES,
  REPO_ROOT,
  SIDECARS,
  buildArtifact,
  caseById,
  enumerateCases,
  enumeratePositions,
  expectedSimulationsFor,
  prefixesFor,
  sha256,
  validateCorpus,
} from './cases.mjs';
import { runAll } from './capture.mjs';
import {
  captureTrace,
  deriveExpectedFixtures,
  mainWithCode,
  parseArgs,
  readFixture,
  runCase,
  searchAndTrace,
} from './worker.mjs';

const OUT_DIR = join(REPO_ROOT, 'runs', 'mcts_golden_test');
const CLOBBER_DIR = join(REPO_ROOT, 'runs', 'mcts_golden_clobber');
const CAPTURE_MJS = join(REPO_ROOT, 'tests', 'mcts_golden', 'capture.mjs');
const WORKER_MJS = join(REPO_ROOT, 'tests', 'mcts_golden', 'worker.mjs');

const git = (...a) => execFileSync('git', a, { cwd: REPO_ROOT }).toString().trim();
const gitClean = () => git('status', '--porcelain') === '';
const HEAD = git('rev-parse', 'HEAD');
/** HEAD carries the LAZY surface now, so operational calls name that stage. */
const STAGE = 'lazy';

/** Derived once from the pinned sidecars; the validator's independent target. */
const DERIVED_FIXTURES = deriveExpectedFixtures();

const legalKeyCache = new Map();
function realLegalKeys(positionId) {
  if (!legalKeyCache.has(positionId)) {
    const position = enumeratePositions().find((p) => p.id === positionId);
    const { state } = readFixture({ position });
    legalKeyCache.set(
      positionId,
      state.legalMoves().map((m) => `${m[0]},${m[1]}`)
    );
  }
  return legalKeyCache.get(positionId);
}

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
  assert.deepEqual(
    enumeratePositions().map((p) => [p.id, p.sidecar, p.prefixPlies]),
    expected
  );
});

test('the immediate-win positions are exactly P04, P08, P12, P16', () => {
  assert.deepEqual(
    enumeratePositions().filter((p) => p.immediateWin).map((p) => p.id),
    ['P04', 'P08', 'P12', 'P16']
  );
});

test('the prefix rule is {4, 16, 28, plyCount - 1}', () => {
  assert.deepEqual(prefixesFor(57), [4, 16, 28, 56]);
  assert.deepEqual(prefixesFor(39), [4, 16, 28, 38]);
});

test('the matrix splits 80 baseline / 10 candidate / 2 abort', () => {
  const cases = enumerateCases();
  const golden = cases.filter((c) => c.kind === 'golden');
  assert.equal(golden.filter((c) => c.modelId === BASELINE_MODEL_ID).length, 80);
  assert.equal(golden.filter((c) => c.modelId === CANDIDATE_MODEL_ID).length, 10);
  assert.equal(cases.filter((c) => c.kind === 'abort').length, 2);
});

test('the candidate runs on exactly P02 and P11, at every simulation count', () => {
  const positions = enumerateCases()
    .filter((c) => c.modelId === CANDIDATE_MODEL_ID)
    .map((c) => c.position.id);
  assert.deepEqual([...new Set(positions)].sort(), ['P02', 'P11']);
  for (const id of ['P02', 'P11']) {
    assert.deepEqual(
      enumerateCases()
        .filter((c) => c.modelId === CANDIDATE_MODEL_ID && c.position.id === id)
        .map((c) => c.nSimulations),
      [...N_SIMULATIONS_LADDER]
    );
  }
});

test('every baseline position carries the full simulation ladder', () => {
  for (const position of enumeratePositions()) {
    assert.deepEqual(
      enumerateCases()
        .filter(
          (c) =>
            c.kind === 'golden' &&
            c.modelId === BASELINE_MODEL_ID &&
            c.position.id === position.id
        )
        .map((c) => c.nSimulations),
      [...N_SIMULATIONS_LADDER],
      position.id
    );
  }
});

test('the abort cases are A1 and A2, on P07, at 64 simulations', () => {
  assert.deepEqual(
    enumerateCases()
      .filter((c) => c.kind === 'abort')
      .map((c) => [c.caseId, c.position.id, c.nSimulations, c.trigger]),
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
  assert.equal(enumerateCases()[0].nSimulations, N_SIMULATIONS_LADDER[0]);
  assert.equal(enumerateCases().length, EXPECTED_CASE_COUNT);
});

test('`list` is pure: it runs and prints 92 cases', () => {
  const proc = spawnSync(process.execPath, [CAPTURE_MJS, 'list'], { encoding: 'utf8' });
  assert.equal(proc.status, 0, proc.stderr);
  assert.match(proc.stdout, /92 cases \(expected 92\)/);
});

// --- CLI: the mode is explicit ----------------------------------------------

test('parseArgs accepts exactly the documented shape', () => {
  assert.deepEqual(parseArgs(['A1', '/out', '--expect-commit', HEAD, '--stage', STAGE, '--dry-run']), {
    caseId: 'A1',
    outDir: '/out',
    mode: 'dry-run',
    stage: STAGE,
    expectCommit: HEAD,
  });
  assert.equal(
    parseArgs(['A1', '/out', '--capture', '--expect-commit', HEAD, '--stage', STAGE]).mode,
    'capture'
  );
});

test('parseArgs rejects a missing mode, a typo, a doubled mode and a missing commit', () => {
  const bad = [
    ['A1', '/out', '--expect-commit', HEAD], // no mode: must NOT default to capture
    ['A1', '/out', '--expect-commit', HEAD, '--dryrun'], // typo, silently ignored before
    ['A1', '/out', '--expect-commit', HEAD, '--stage', STAGE, '--dry-run', '--capture'],
    ['A1', '/out', '--dry-run'], // no --expect-commit
    ['A1', '/out', '--expect-commit'], // flag without a value
    ['A1'], // no out dir
  ];
  for (const argv of bad) {
    assert.throws(
      () => parseArgs(argv),
      (err) => err.code === 'USAGE',
      JSON.stringify(argv)
    );
  }
});

test('the worker CLI exits 2 on a mode typo rather than capturing', () => {
  const proc = spawnSync(
    process.execPath,
    [WORKER_MJS, 'G_P01_baseline_s1', OUT_DIR, '--expect-commit', HEAD, '--dryrun'],
    { encoding: 'utf8' }
  );
  assert.equal(proc.status, 2, proc.stdout + proc.stderr);
  assert.match(proc.stderr, /unrecognised argument --dryrun/);
});

test('runCase refuses without an explicit stage, mode or valid expectCommit', async () => {
  const testCase = caseById('G_P01_baseline_s1');
  // The stage is checked first: without it the artifact could not name the
  // surface it describes.
  await assert.rejects(
    () => runCase({ testCase, outDir: OUT_DIR, mode: 'dry-run', expectCommit: HEAD }),
    (e) => e.code === 'UNKNOWN_STAGE'
  );
  await assert.rejects(
    () =>
      runCase({ testCase, outDir: OUT_DIR, mode: 'dry-run', stage: 'made-up', expectCommit: HEAD }),
    (e) => e.code === 'UNKNOWN_STAGE'
  );
  await assert.rejects(
    () => runCase({ testCase, outDir: OUT_DIR, stage: STAGE, expectCommit: HEAD }),
    (e) => e.code === 'MODE_REQUIRED'
  );
  await assert.rejects(
    () => runCase({ testCase, outDir: OUT_DIR, mode: 'dry-run', stage: STAGE }),
    (e) => e.code === 'EXPECT_COMMIT_REQUIRED'
  );
});

// --- cleanliness and commit binding ------------------------------------------

test('POSITIVE CONTROL: on a clean worktree a dry run reaches the fixture', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty for unrelated reasons');
  rmSync(OUT_DIR, { recursive: true, force: true });
  let captureTraceCalls = 0;

  const artifact = await runCase(
    {
      testCase: caseById('G_P01_baseline_s1'),
      outDir: OUT_DIR,
      mode: 'dry-run',
      stage: STAGE,
      expectCommit: HEAD,
    },
    {
      captureTrace: () => {
        captureTraceCalls += 1;
        throw new Error('dry-run must not load a model');
      },
    }
  );

  assert.equal(artifact.status, 'dry-run');
  assert.equal(artifact.trace, null);
  assert.equal(captureTraceCalls, 0);
  assert.equal(artifact.execution_surface_sha256, STAGES[STAGE].surfaceSha256);
  assert.equal(artifact.fixture.ply_after_prefix, 4);
  assert.ok(artifact.fixture.n_legal > 400);
});

async function expectRefusalWhileDirty(t, dirty, clean) {
  if (!gitClean()) return t.skip('worktree dirty for unrelated reasons');
  const testCase = caseById('G_P01_baseline_s1');
  const counters = { readFixture: 0, captureTrace: 0 };
  try {
    dirty();
    assert.equal(gitClean(), false, 'the perturbation did not dirty the worktree');
    await assert.rejects(
      () =>
        runCase(
          { testCase, outDir: OUT_DIR, mode: 'dry-run', stage: STAGE, expectCommit: HEAD },
          countingSeams(counters)
        ),
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
  assert.notEqual(
    CAPTURE_MJS,
    join(REPO_ROOT, FIXTURE_RELDIR, 'timing_00_opening_200.json')
  );
});

test('a commit made mid-run refuses at the NEXT case, fixtures unread', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty for unrelated reasons');
  // A clean commit leaves the surface digest unchanged, so without the commit
  // binding every later worker would succeed under a different capture_commit
  // and the corpus would only be rejected after all 92 finished.
  const counters = { readFixture: 0, captureTrace: 0 };
  await assert.rejects(
    () =>
      runCase(
        {
          testCase: caseById('G_P01_baseline_s1'),
          outDir: OUT_DIR,
          mode: 'dry-run',
          stage: STAGE,
          expectCommit: 'b'.repeat(40),
        },
        countingSeams(counters)
      ),
    (err) => err.code === 'CAPTURE_COMMIT_MOVED'
  );
  assert.equal(counters.readFixture, 0);
  assert.equal(counters.captureTrace, 0);
});

// --- process isolation -------------------------------------------------------

test('each case runs in its own process: three cases, three distinct pids', (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  rmSync(OUT_DIR, { recursive: true, force: true });
  const ids = ['G_P01_baseline_s1', 'G_P05_baseline_s1', 'G_P09_baseline_s1'];
  const pids = new Set();

  for (const id of ids) {
    const proc = spawnSync(
      process.execPath,
      [WORKER_MJS, id, OUT_DIR, '--expect-commit', HEAD, '--stage', STAGE, '--dry-run'],
      { encoding: 'utf8' }
    );
    assert.equal(proc.status, 0, proc.stderr);
    pids.add(JSON.parse(readFileSync(join(OUT_DIR, `${id}.json`), 'utf8')).pid);
  }
  assert.equal(pids.size, ids.length, 'cases shared a process');
  assert.ok(!pids.has(process.pid), 'a case ran inside the test process');
});

test('the worker refuses an unknown case id', () => {
  const proc = spawnSync(
    process.execPath,
    [WORKER_MJS, 'G_NOPE_baseline_s1', OUT_DIR, '--expect-commit', HEAD, '--stage', STAGE, '--dry-run'],
    { encoding: 'utf8' }
  );
  assert.equal(proc.status, 2);
  assert.match(proc.stderr, /UNKNOWN_CASE/);
});

// --- no clobber --------------------------------------------------------------

test('a stale FINAL artifact refuses, and is not deleted or modified', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  rmSync(CLOBBER_DIR, { recursive: true, force: true });
  mkdirSync(CLOBBER_DIR, { recursive: true });
  const stale = join(CLOBBER_DIR, 'G_P01_baseline_s1.json');
  const sentinel = '{"existing":"evidence"}\n';
  writeFileSync(stale, sentinel);

  await assert.rejects(
    () =>
      runCase({
        testCase: caseById('G_P01_baseline_s1'),
        outDir: CLOBBER_DIR,
        mode: 'dry-run',
        stage: STAGE,
        expectCommit: HEAD,
      }),
    (err) => err.code === 'ARTIFACT_EXISTS'
  );
  assert.equal(readFileSync(stale, 'utf8'), sentinel, 'existing evidence was altered');
});

test('a stale TEMP artifact refuses, and is not deleted or modified', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  rmSync(CLOBBER_DIR, { recursive: true, force: true });
  mkdirSync(CLOBBER_DIR, { recursive: true });
  const tmp = join(CLOBBER_DIR, 'G_P01_baseline_s1.json.tmp');
  const sentinel = '{"interrupted":"write"}\n';
  writeFileSync(tmp, sentinel);

  await assert.rejects(
    () =>
      runCase({
        testCase: caseById('G_P01_baseline_s1'),
        outDir: CLOBBER_DIR,
        mode: 'dry-run',
        stage: STAGE,
        expectCommit: HEAD,
      }),
    (err) => err.code === 'STALE_TEMP_ARTIFACT'
  );
  assert.equal(readFileSync(tmp, 'utf8'), sentinel, 'interrupted-write residue was altered');
});

test('POSITIVE CONTROL: the same case succeeds into an empty directory', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  rmSync(CLOBBER_DIR, { recursive: true, force: true });
  const artifact = await runCase(
    {
      testCase: caseById('G_P01_baseline_s1'),
      outDir: CLOBBER_DIR,
      mode: 'dry-run',
      stage: STAGE,
      expectCommit: HEAD,
    },
    { captureTrace: () => assert.fail('dry-run must not load a model') }
  );
  assert.equal(artifact.status, 'dry-run');
});

test('the orchestrator refuses a partially completed corpus, deleting nothing', (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  rmSync(CLOBBER_DIR, { recursive: true, force: true });
  mkdirSync(CLOBBER_DIR, { recursive: true });
  const partial = join(CLOBBER_DIR, 'G_P01_baseline_s1.json');
  writeFileSync(partial, '{"partial":"corpus"}\n');

  const proc = spawnSync(process.execPath, [CAPTURE_MJS, 'dry-run', CLOBBER_DIR, '--stage', STAGE], {
    encoding: 'utf8',
  });
  assert.equal(proc.status, 3, proc.stdout + proc.stderr);
  assert.match(proc.stderr, /OUTPUT_DIR_EXISTS/);
  assert.ok(existsSync(partial), 'the partial corpus was deleted');
  assert.equal(readdirSync(CLOBBER_DIR).length, 1, 'the partial corpus was added to');
});

test('the orchestrator refuses even an EMPTY existing directory', (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  rmSync(CLOBBER_DIR, { recursive: true, force: true });
  mkdirSync(CLOBBER_DIR, { recursive: true });
  const proc = spawnSync(process.execPath, [CAPTURE_MJS, 'dry-run', CLOBBER_DIR, '--stage', STAGE], {
    encoding: 'utf8',
  });
  assert.equal(proc.status, 3, proc.stdout + proc.stderr);
  assert.match(proc.stderr, /OUTPUT_DIR_EXISTS/);
});

test('two concurrent workers on one case: exactly one wins, its bytes intact', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  const dir = join(REPO_ROOT, 'runs', 'mcts_golden_race');
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });

  const args = [WORKER_MJS, 'G_P01_baseline_s1', dir, '--expect-commit', HEAD, '--stage', STAGE, '--dry-run'];
  const run = () =>
    new Promise((resolve) => {
      const p = spawn(process.execPath, args);
      let err = '';
      p.stderr.on('data', (d) => (err += d));
      p.on('close', (code) => resolve({ code, err }));
    });

  const results = await Promise.all([run(), run()]);
  const winners = results.filter((r) => r.code === 0);
  assert.equal(winners.length, 1, `expected exactly one winner, got ${winners.length}`);
  const loser = results.find((r) => r.code !== 0);
  assert.equal(loser.code, 3, loser.err);
  assert.match(loser.err, /ARTIFACT_EXISTS|STALE_TEMP_ARTIFACT/);

  const entries = readdirSync(dir);
  assert.deepEqual(entries, ['G_P01_baseline_s1.json'], `unexpected entries: ${entries}`);
  assert.equal(
    JSON.parse(readFileSync(join(dir, entries[0]), 'utf8')).case_id,
    'G_P01_baseline_s1'
  );
  rmSync(dir, { recursive: true, force: true });
});

// --- fixture identity --------------------------------------------------------

test('a fixture whose bytes do not match the pinned hash is refused', () => {
  const real = caseById('G_P01_baseline_s1');
  assert.throws(
    () => readFixture({ ...real, position: { ...real.position, sidecarSha256: 'f'.repeat(64) } }),
    (err) => err.code === 'FIXTURE_SHA256'
  );
});

test('every pinned sidecar hash matches the committed evidence', () => {
  for (const s of SIDECARS) {
    assert.equal(sha256(readFileSync(join(REPO_ROOT, FIXTURE_RELDIR, s.file))), s.sha256, s.file);
  }
});

// --- validation --------------------------------------------------------------

const FAKE_COMMIT = 'a'.repeat(40);

/** A synthetic corpus that is VALID for the requested mode. */
function fakeCorpus({ mode = 'dry-run', mutate } = {}) {
  const files = new Map();
  let pid = 1000;

  for (const c of enumerateCases()) {
    const sims = expectedSimulationsFor(c);
    let trace = null;

    if (mode === 'capture') {
      const keys = realLegalKeys(c.position.id);
      trace = {
        visit_counts: sims === 0 ? [] : keys.map((k, i) => [k, i === 0 ? sims : 0]),
        root_value: sims === 0 ? 0 : 0.25,
        selected_move: sims === 0 ? null : keys[0],
        progress: Array.from({ length: sims }, (_, i) => ({
          done: i + 1,
          total: c.nSimulations,
          valueEstimate: 0.25,
        })),
        progress_elapsed_ms: Array.from({ length: sims }, (_, i) => i * 10),
      };
    }

    files.set(`${c.caseId}.json`, {
      schema: STAGES[STAGE].artifactSchema,
      stage: STAGE,
      status: mode === 'capture' ? 'captured' : 'dry-run',
      case_id: c.caseId,
      kind: c.kind,
      trigger: c.trigger ?? null,
      position: {
        id: c.position.id,
        sidecar: c.position.sidecar,
        opening_id: c.position.openingId,
        prefix_plies: c.position.prefixPlies,
        immediate_win: c.position.immediateWin,
      },
      model_id: c.modelId,
      n_simulations: c.nSimulations,
      c_puct: 1.5,
      move_temp: 0,
      capture_commit: FAKE_COMMIT,
      pinned_surface_commit: STAGES[STAGE].surfaceCommit,
      execution_surface_sha256: STAGES[STAGE].surfaceSha256,
      fixture: { ...DERIVED_FIXTURES.get(c.position.id) },
      pid: pid++,
      trace,
    });
  }
  mutate?.(files);
  return {
    readdirSync: () => [...files.keys()],
    readFileSync: (p) => JSON.stringify(files.get(String(p).split('/').pop())),
  };
}

const validate = (fs, mode = 'dry-run') =>
  validateCorpus('/fake', {
    mode,
    stage: STAGE,
    expectedCaptureCommit: FAKE_COMMIT,
    expectedFixtures: DERIVED_FIXTURES,
    ...fs,
  });
const codesFor = (fs, mode) => validate(fs, mode).map((f) => f.code);

test('the COMMITTED eager corpus still validates, under the eager stage', () => {
  // The requirement that made stages necessary: existing evidence is immutable
  // and must keep validating under the surface it recorded, not under HEAD's.
  const dir = join(REPO_ROOT, 'tests', 'mcts_golden', 'golden', '841df60', 'artifacts');
  const failures = validateCorpus(dir, {
    mode: 'capture',
    stage: 'eager',
    expectedCaptureCommit: '841df6040a740a4b9f1753253e0e8bfc63e15366',
    expectedFixtures: DERIVED_FIXTURES,
    readdirSync,
    readFileSync,
  });
  assert.deepEqual(failures, [], 'the committed eager corpus no longer validates');
  assert.equal(readdirSync(dir).length, 92);
});

test('the eager corpus is REJECTED when judged under the lazy stage', () => {
  // Its schema, surface and pinned commit all belong to the eager stage, so
  // judging it as lazy must fail rather than quietly accept.
  const dir = join(REPO_ROOT, 'tests', 'mcts_golden', 'golden', '841df60', 'artifacts');
  const failures = validateCorpus(dir, {
    mode: 'capture',
    stage: 'lazy',
    expectedCaptureCommit: '841df6040a740a4b9f1753253e0e8bfc63e15366',
    expectedFixtures: DERIVED_FIXTURES,
    readdirSync,
    readFileSync,
  });
  assert.ok(failures.length > 0, 'an eager corpus was accepted as lazy');
  const fields = new Set(failures.map((f) => f.detail?.field));
  assert.ok(fields.has('schema'));
  assert.ok(fields.has('execution_surface_sha256'));
});

test('a NEW artifact records its stage, commit and surface', () => {
  const artifact = buildArtifact({
    testCase: caseById('G_P01_baseline_s1'),
    captureCommit: HEAD,
    fixture: DERIVED_FIXTURES.get('P01'),
    status: 'dry-run',
    stage: 'lazy',
  });
  assert.equal(artifact.stage, 'lazy');
  assert.equal(artifact.schema, STAGES.lazy.artifactSchema);
  assert.equal(artifact.capture_commit, HEAD);
  assert.equal(artifact.execution_surface_sha256, STAGES[STAGE].surfaceSha256);
  assert.equal(artifact.pinned_surface_commit, STAGES.lazy.surfaceCommit);

  // The legacy eager schema carries no stage field at all — not an empty one.
  const legacy = buildArtifact({
    testCase: caseById('G_P01_baseline_s1'),
    captureCommit: HEAD,
    fixture: DERIVED_FIXTURES.get('P01'),
    status: 'dry-run',
    stage: 'eager',
  });
  assert.equal('stage' in legacy, false);
  assert.equal(legacy.schema, STAGES.eager.artifactSchema);
});

test('validateCorpus THROWS if either mandatory binding is omitted', () => {
  const fs = fakeCorpus();
  assert.throws(
    () =>
      validateCorpus('/fake', {
        mode: 'dry-run',
        stage: STAGE,
        expectedFixtures: DERIVED_FIXTURES,
        ...fs,
      }),
    (e) => e.code === 'MISSING_EXPECTED_COMMIT'
  );
  assert.throws(
    () =>
      validateCorpus('/fake', {
        mode: 'dry-run',
        stage: STAGE,
        expectedCaptureCommit: FAKE_COMMIT,
        ...fs,
      }),
    (e) => e.code === 'MISSING_EXPECTED_FIXTURES'
  );
});

test('POSITIVE CONTROL: a well-formed dry-run corpus validates clean', () => {
  assert.deepEqual(validate(fakeCorpus()), []);
});

test('validation rejects 92 files carrying the wrong case ids', () => {
  const fs = fakeCorpus({
    mutate: (f) => (f.get('G_P01_baseline_s1.json').case_id = 'G_P16_baseline_s800'),
  });
  assert.ok(codesFor(fs).includes('ARTIFACT_FIELD'));
});

test('validation rejects a stale execution-surface digest', () => {
  const fs = fakeCorpus({
    mutate: (f) => (f.get('G_P03_baseline_s8.json').execution_surface_sha256 = '0'.repeat(64)),
  });
  assert.ok(codesFor(fs).includes('ARTIFACT_FIELD'));
});

test('validation rejects a capture_commit that is not the preflighted commit', () => {
  const failures = validateCorpus('/fake', {
    mode: 'dry-run',
    stage: STAGE,
    expectedCaptureCommit: 'd'.repeat(40),
    expectedFixtures: DERIVED_FIXTURES,
    ...fakeCorpus(),
  });
  assert.ok(failures.some((f) => f.detail?.field === 'capture_commit'));
});

test('a reused pid is NOT a validation failure (operating systems recycle pids)', () => {
  const fs = fakeCorpus({
    mutate: (f) =>
      (f.get('G_P02_baseline_s1.json').pid = f.get('G_P01_baseline_s1.json').pid),
  });
  assert.deepEqual(validate(fs), [], 'pid cardinality must be diagnostic, not a gate');
});

test('validation rejects a stray non-JSON file and a leftover .tmp', () => {
  const base = fakeCorpus();
  const withStray = {
    readdirSync: () => [...base.readdirSync(), 'notes.txt', 'A1.json.tmp'],
    readFileSync: base.readFileSync,
  };
  const set = validate(withStray).find((f) => f.code === 'ARTIFACT_FILENAME_SET');
  assert.ok(set, 'a stray non-JSON entry was ignored');
  assert.deepEqual(set.detail.stray.sort(), ['A1.json.tmp', 'notes.txt']);
});

test('validation rejects a missing file and a stray file', () => {
  const fs = fakeCorpus({
    mutate: (f) => {
      f.delete('A2.json');
      f.set('G_P99_baseline_s1.json', { schema: 'twixt-mcts-golden/1' });
    },
  });
  const set = validate(fs).find((f) => f.code === 'ARTIFACT_FILENAME_SET');
  assert.deepEqual(set.detail.missing, ['A2.json']);
  assert.deepEqual(set.detail.stray, ['G_P99_baseline_s1.json']);
});

test('validation rejects a dry-run artifact carrying a trace', () => {
  const fs = fakeCorpus({ mutate: (f) => (f.get('A1.json').trace = { visit_counts: [] }) });
  assert.ok(codesFor(fs).includes('ARTIFACT_FIELD'));
});

test('COLLUDING MUTATION: changing keys, hash and count together still fails', () => {
  // The whole point of re-deriving. A fabricator controls every field inside the
  // artifact, so a self-consistent fixture descriptor must not be enough.
  const fakeKeys = ['0,1', '0,2', '0,3'];
  const fs = fakeCorpus({
    mode: 'capture',
    mutate: (f) => {
      const a = f.get('G_P01_baseline_s8.json');
      a.fixture = {
        ...a.fixture,
        n_legal: fakeKeys.length,
        legal_moves_sha256: sha256(JSON.stringify(fakeKeys)),
      };
      a.trace.visit_counts = fakeKeys.map((k, i) => [k, i === 0 ? 8 : 0]);
      a.trace.selected_move = fakeKeys[0];
    },
  });
  const failures = validate(fs, 'capture');
  assert.ok(failures.length > 0, 'a self-consistent fabricated fixture was accepted');
  const fields = failures.map((f) => f.detail?.field);
  assert.ok(fields.includes('fixture.n_legal'));
  assert.ok(fields.includes('fixture.legal_moves_sha256'));
});

// --- trace semantics ---------------------------------------------------------

test('REGRESSION: a fabricated corpus — empty visit maps, no progress, root_value 999 — is REJECTED', () => {
  const fs = fakeCorpus({
    mode: 'capture',
    mutate: (f) => {
      for (const a of f.values()) {
        a.trace = {
          visit_counts: [],
          root_value: 999,
          selected_move: null,
          progress: [],
          progress_elapsed_ms: [],
        };
      }
    },
  });
  const failures = validate(fs, 'capture');
  assert.ok(failures.length > 0, 'the fabricated corpus was certified valid');
  const fields = new Set(failures.map((f) => f.detail?.field));
  assert.ok(fields.has('trace.root_value'), 'root_value 999 accepted');
  assert.ok(fields.has('trace.visit_counts'), 'empty visit map accepted for a real search');
  assert.ok(fields.has('trace.progress.length'), 'missing progress accepted');
});

test('POSITIVE CONTROL: a semantically well-formed capture corpus validates clean', () => {
  assert.deepEqual(validate(fakeCorpus({ mode: 'capture' }), 'capture'), []);
});

test('capture validation rejects an incomplete or reordered visit map', () => {
  const dropped = fakeCorpus({
    mode: 'capture',
    mutate: (f) => f.get('G_P01_baseline_s8.json').trace.visit_counts.pop(),
  });
  assert.ok(codesFor(dropped, 'capture').includes('ARTIFACT_FIELD'));

  const reordered = fakeCorpus({
    mode: 'capture',
    mutate: (f) => {
      const t = f.get('G_P01_baseline_s8.json').trace;
      t.visit_counts.reverse();
      t.selected_move = t.visit_counts.find((e) => e[1] > 0)[0];
    },
  });
  assert.ok(
    codesFor(reordered, 'capture').includes('ARTIFACT_FIELD'),
    'a reordered visit map passed the ordered-key hash'
  );
});

test('capture validation rejects duplicate move keys and negative counts', () => {
  const dup = fakeCorpus({
    mode: 'capture',
    mutate: (f) => {
      const vc = f.get('G_P01_baseline_s8.json').trace.visit_counts;
      vc[1][0] = vc[0][0];
    },
  });
  assert.ok(codesFor(dup, 'capture').includes('ARTIFACT_FIELD'));

  const neg = fakeCorpus({
    mode: 'capture',
    mutate: (f) => (f.get('G_P01_baseline_s8.json').trace.visit_counts[1][1] = -1),
  });
  assert.ok(codesFor(neg, 'capture').includes('ARTIFACT_FIELD'));
});

test('capture validation requires visit counts to sum to the simulation count', () => {
  const fs = fakeCorpus({
    mode: 'capture',
    mutate: (f) => (f.get('G_P01_baseline_s64.json').trace.visit_counts[0][1] = 63),
  });
  assert.ok(validate(fs, 'capture').some((f) => f.detail?.field === 'trace.visit_counts sum'));
});

test('capture validation enforces the abort contract for A1 and A2', () => {
  const a1 = fakeCorpus({
    mode: 'capture',
    mutate: (f) =>
      (f.get('A1.json').trace.visit_counts = [[realLegalKeys('P07')[0], 1]]),
  });
  assert.ok(codesFor(a1, 'capture').includes('ARTIFACT_FIELD'));

  const a2 = fakeCorpus({
    mode: 'capture',
    mutate: (f) => {
      const t = f.get('A2.json').trace;
      t.visit_counts[0][1] = 64;
      t.progress = Array.from({ length: 64 }, (_, i) => ({
        done: i + 1,
        total: 64,
        valueEstimate: 0.25,
      }));
      t.progress_elapsed_ms = Array.from({ length: 64 }, (_, i) => i);
    },
  });
  assert.ok(codesFor(a2, 'capture').includes('ARTIFACT_FIELD'));
});

test('capture validation rejects a broken done sequence and an out-of-range valueEstimate', () => {
  const seq = fakeCorpus({
    mode: 'capture',
    mutate: (f) => (f.get('G_P01_baseline_s8.json').trace.progress[3].done = 99),
  });
  assert.ok(codesFor(seq, 'capture').includes('ARTIFACT_FIELD'));

  const range = fakeCorpus({
    mode: 'capture',
    mutate: (f) => (f.get('G_P01_baseline_s8.json').trace.progress[3].valueEstimate = 7),
  });
  assert.ok(codesFor(range, 'capture').includes('ARTIFACT_FIELD'));
});

test('capture validation rejects a selected_move that does not follow from the counts', () => {
  const fs = fakeCorpus({
    mode: 'capture',
    mutate: (f) => {
      const t = f.get('G_P01_baseline_s8.json').trace;
      t.selected_move = t.visit_counts[3][0];
    },
  });
  assert.ok(validate(fs, 'capture').some((f) => f.detail?.field === 'trace.selected_move'));
});

test('capture validation rejects non-monotonic elapsed metadata', () => {
  const fs = fakeCorpus({
    mode: 'capture',
    mutate: (f) => (f.get('G_P01_baseline_s64.json').trace.progress_elapsed_ms[10] = 0),
  });
  assert.ok(codesFor(fs, 'capture').includes('ARTIFACT_FIELD'));
});

test('in capture mode, validation rejects `elapsed` smuggled into a compared entry', () => {
  const fs = fakeCorpus({
    mode: 'capture',
    mutate: (f) => (f.get('G_P01_baseline_s8.json').trace.progress[0].elapsed = 12),
  });
  assert.ok(
    validate(fs, 'capture').some((f) => String(f.detail?.found).includes('carries elapsed')),
    'elapsed was allowed inside a compared field'
  );
});

// --- INTEGRATION: the real MCTS, no ONNX -------------------------------------

/**
 * Deterministic stand-in for AlphaZeroInference. `server/mcts.js` has no imports
 * of its own, so the REAL search runs against this without any model.
 */
const fakeInference = {
  async evaluate(_boardTensor, moves) {
    const priors = new Map();
    const denom = (moves.length * (moves.length + 1)) / 2;
    moves.forEach((m, i) => priors.set(`${m[0]},${m[1]}`, (moves.length - i) / denom));
    return { priors, value: ((moves.length % 11) / 11) * 2 - 1 };
  },
};

// --- session lifecycle and exit discipline ----------------------------------
// The first real capture aborted AFTER publishing its artifact, with the ORT
// session never released. The old worker would then have called process.exit(),
// though the evidence does not establish that the call was reached. These tests
// pin the two halves the harness controls — release, and not forcing exit — and
// use FAKES only: no test at this gate loads a real model.

function fakeModel({ modelId, releaseThrows = false, log }) {
  return {
    modelId,
    inference: {
      ...fakeInference,
      session: {
        release: async () => {
          log.push('release');
          if (releaseThrows) throw new Error('release blew up');
        },
      },
    },
  };
}

test('LIFECYCLE: the session is released after a successful trace', async () => {
  const testCase = caseById('G_P01_baseline_s8');
  const { state } = readFixture(testCase);
  const log = [];
  const trace = await captureTrace(testCase, state, async () => {
    log.push('load');
    return fakeModel({ modelId: testCase.modelId, log });
  });
  log.push('returned');
  assert.deepEqual(log, ['load', 'release', 'returned'], 'release must precede the return');
  assert.equal(trace.progress.length, 8);
});

test('LIFECYCLE: the session is released when the search throws', async () => {
  const testCase = caseById('G_P01_baseline_s8');
  const { state } = readFixture(testCase);
  const log = [];
  const exploding = {
    modelId: testCase.modelId,
    inference: {
      evaluate: async () => {
        throw new Error('inference exploded');
      },
      session: {
        release: async () => log.push('release'),
      },
    },
  };
  await assert.rejects(
    () => captureTrace(testCase, state, async () => exploding),
    /inference exploded/
  );
  assert.deepEqual(log, ['release'], 'a failed search must still release the session');
});

test('LIFECYCLE: the session is released when the model role is wrong, and the role error survives', async () => {
  const testCase = caseById('G_P01_baseline_s8');
  const { state } = readFixture(testCase);
  const log = [];
  await assert.rejects(
    () => captureTrace(testCase, state, async () => fakeModel({ modelId: 'wrong', log })),
    (err) => err.code === 'MODEL_ROLE'
  );
  assert.deepEqual(log, ['release']);
});

test('LIFECYCLE: a failing release does not mask the original error, and is attached', async () => {
  const testCase = caseById('A1');
  const { state } = readFixture(testCase);
  const log = [];
  await assert.rejects(
    () =>
      captureTrace(testCase, state, async () =>
        fakeModel({ modelId: 'wrong', releaseThrows: true, log })
      ),
    (err) =>
      err.code === 'MODEL_ROLE' && err.secondary?.code === 'SESSION_RELEASE_FAILED'
  );
  assert.deepEqual(log, ['release']);
});

test('LIFECYCLE: a failing release FAILS an otherwise good trace', async () => {
  // A trace from a session that could not be released has not demonstrated the
  // clean teardown this remedy exists to establish. An earlier revision
  // returned the trace anyway, which would have certified a corpus while
  // proving nothing about teardown.
  const testCase = caseById('A1');
  const { state } = readFixture(testCase);
  const log = [];
  await assert.rejects(
    () =>
      captureTrace(testCase, state, async () =>
        fakeModel({ modelId: testCase.modelId, releaseThrows: true, log })
      ),
    (err) => err.code === 'SESSION_RELEASE_FAILED'
  );
  assert.deepEqual(log, ['release']);
});

/** Run captureTrace and report exactly what came out, falsy values included. */
async function captureOutcome(testCase, state, loadFn) {
  try {
    return { threw: false, value: await captureTrace(testCase, state, loadFn) };
  } catch (err) {
    return { threw: true, value: err };
  }
}

test('PRECEDENCE: a FROZEN primary error survives a release failure unchanged', async () => {
  const testCase = caseById('A1');
  const { state } = readFixture(testCase);
  const frozen = Object.freeze(new Error('frozen primary'));
  const log = [];

  const out = await captureOutcome(testCase, state, async () => ({
    modelId: testCase.modelId,
    inference: {
      evaluate: async () => {
        throw frozen;
      },
      session: {
        release: async () => {
          log.push('release');
          throw new Error('release blew up');
        },
      },
    },
  }));

  assert.equal(out.threw, true);
  // Identity, not just shape: attaching `secondary` to a frozen Error throws a
  // TypeError in strict mode, and that TypeError must not become the result.
  assert.equal(out.value, frozen, 'the frozen primary was replaced');
  assert.equal(out.value.message, 'frozen primary');
  assert.equal(out.value.secondary, undefined, 'a frozen error cannot carry the secondary');
  assert.deepEqual(log, ['release'], 'release was still attempted');
});

test('PRECEDENCE: a falsy thrown value is a failure, not a success', async () => {
  const testCase = caseById('A1');
  const { state } = readFixture(testCase);

  for (const thrown of [null, undefined, 0, '', false]) {
    const out = await captureOutcome(testCase, state, async () => ({
      modelId: testCase.modelId,
      inference: {
        evaluate: async () => {
          throw thrown;
        },
        session: { release: async () => {} },
      },
    }));
    assert.equal(out.threw, true, `throwing ${String(thrown)} was swallowed`);
    assert.equal(out.value, thrown, `throwing ${String(thrown)} changed the value`);
  }
});

test('PRECEDENCE: a falsy primary still takes precedence over a release failure', async () => {
  const testCase = caseById('A1');
  const { state } = readFixture(testCase);
  const log = [];

  const out = await captureOutcome(testCase, state, async () => ({
    modelId: testCase.modelId,
    inference: {
      evaluate: async () => {
        throw null;
      },
      session: {
        release: async () => {
          log.push('release');
          throw new Error('release blew up');
        },
      },
    },
  }));

  assert.equal(out.threw, true);
  assert.equal(out.value, null, 'the release failure replaced a falsy primary');
  assert.deepEqual(log, ['release']);
});

test('LIFECYCLE: a session with no callable release() fails rather than being skipped', async () => {
  const testCase = caseById('A1');
  const { state } = readFixture(testCase);
  await assert.rejects(
    () =>
      captureTrace(testCase, state, async () => ({
        modelId: testCase.modelId,
        inference: { ...fakeInference, session: {} },
      })),
    (err) => err.code === 'SESSION_RELEASE_UNAVAILABLE'
  );
  // ...and an absent session object is equally not a pass.
  await assert.rejects(
    () =>
      captureTrace(testCase, state, async () => ({
        modelId: testCase.modelId,
        inference: { ...fakeInference },
      })),
    (err) => err.code === 'SESSION_RELEASE_UNAVAILABLE'
  );
});

test('LIFECYCLE: a release failure publishes NO artifact', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  const dir = join(REPO_ROOT, 'runs', 'mcts_golden_release');
  rmSync(dir, { recursive: true, force: true });
  const testCase = caseById('A1');
  const log = [];

  await assert.rejects(
    () =>
      runCase(
        { testCase, outDir: dir, mode: 'capture', stage: STAGE, expectCommit: HEAD },
        {
          // The REAL captureTrace, driven by a fake model — so the publication
          // consequence of a release failure is exercised, not assumed.
          captureTrace: (tc, st) =>
            captureTrace(tc, st, async () =>
              fakeModel({ modelId: tc.modelId, releaseThrows: true, log })
            ),
        }
      ),
    (err) => err.code === 'SESSION_RELEASE_FAILED'
  );

  assert.deepEqual(log, ['release']);
  assert.equal(
    existsSync(join(dir, 'A1.json')),
    false,
    'an artifact was published despite teardown failing'
  );
  assert.equal(existsSync(join(dir, 'A1.json.tmp')), false, 'a temp file was left behind');
  rmSync(dir, { recursive: true, force: true });
});

/** Strip comments, so prose mentioning `process.exit()` is not read as a call. */
const stripComments = (src) =>
  src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1');

test('EXIT: no harness entry point calls process.exit — all return codes', () => {
  for (const file of ['worker.mjs', 'capture.mjs', 'falsify.mjs']) {
    const src = stripComments(
      readFileSync(join(REPO_ROOT, 'tests', 'mcts_golden', file), 'utf8')
    );
    assert.equal(
      /process\.exit\s*\(/.test(src),
      false,
      `${file}: process.exit tears the process down while native threads may be live`
    );
    assert.match(src, /process\.exitCode\s*=/, file);
  }
});

test('EXIT: mainWithCode returns the documented codes without terminating', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  rmSync(OUT_DIR, { recursive: true, force: true });

  assert.equal(await mainWithCode(['A1']), 2, 'usage error');
  assert.equal(
    await mainWithCode(['G_NOPE', OUT_DIR, '--expect-commit', HEAD, '--stage', STAGE, '--dry-run']),
    2,
    'unknown case'
  );
  assert.equal(
    await mainWithCode([
      'G_P01_baseline_s1',
      OUT_DIR,
      '--expect-commit',
      'b'.repeat(40),
      '--stage',
      STAGE,
      '--dry-run',
    ]),
    3,
    'refusal'
  );
  assert.equal(
    await mainWithCode(['G_P01_baseline_s1', OUT_DIR, '--expect-commit', HEAD, '--stage', STAGE, '--dry-run']),
    0,
    'success'
  );
  // Reaching here at all proves nothing terminated the test process.
  assert.ok(true);
});

test('the orchestrator preserves worker stdout, stderr, signal and status on failure', () => {
  const src = readFileSync(join(REPO_ROOT, 'tests', 'mcts_golden', 'capture.mjs'), 'utf8');
  for (const needle of ['proc.signal', 'proc.error', 'proc.stdout', 'proc.stderr', 'proc.status']) {
    assert.ok(src.includes(needle), `failure branch must surface ${needle}`);
  }
});

test('runAll returns the full failure record for a refusing worker', (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  const dir = join(REPO_ROOT, 'runs', 'mcts_golden_failrec');
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  // Pre-place case 1's artifact so the first worker refuses with ARTIFACT_EXISTS.
  writeFileSync(join(dir, 'G_P01_baseline_s1.json'), '{"stale":true}\n');

  const out = runAll({ outDir: dir, mode: 'dry-run', stage: STAGE, expectCommit: HEAD });
  assert.equal(out.ok, false);
  assert.equal(out.failedAt, 'G_P01_baseline_s1');
  assert.equal(out.failure.status, 3);
  assert.equal(out.failure.signal, null);
  assert.match(out.failure.stderr, /ARTIFACT_EXISTS/);
  assert.equal(typeof out.failure.stdout, 'string', 'stdout must be captured even when empty');
  rmSync(dir, { recursive: true, force: true });
});

test('INTEGRATION: real MCTS output satisfies the validator (normal case, A1, A2)', async () => {
  const ids = ['G_P01_baseline_s8', 'A1', 'A2'];
  const real = new Map();

  for (const id of ids) {
    const testCase = caseById(id);
    const { state, describe } = readFixture(testCase);
    const trace = await searchAndTrace(testCase, state, fakeInference);
    real.set(
      `${id}.json`,
      buildArtifact({
        testCase,
        captureCommit: FAKE_COMMIT,
        fixture: describe,
        status: 'captured',
        stage: STAGE,
        trace,
      })
    );
  }

  // Sanity: the abort contract actually held in the real search, so the
  // validator is being asked about genuine output rather than a stub.
  assert.equal(real.get('A1.json').trace.visit_counts.length, 0, 'A1 was not empty');
  assert.equal(real.get('A2.json').trace.progress.length, 5, 'A2 did not stop at 5');
  assert.equal(real.get('G_P01_baseline_s8.json').trace.progress.length, 8);

  // Splice the three real artifacts into an otherwise-synthetic valid corpus.
  const base = fakeCorpus({ mode: 'capture' });
  const spliced = {
    readdirSync: base.readdirSync,
    readFileSync: (p) => {
      const name = String(p).split('/').pop();
      return real.has(name) ? JSON.stringify(real.get(name)) : base.readFileSync(p);
    },
  };

  assert.deepEqual(
    validate(spliced, 'capture'),
    [],
    'the validator rejected genuine MCTS output'
  );
});
