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
  PINNED_EXECUTION_SURFACE_SHA256,
  REPO_ROOT,
  SIDECARS,
  caseById,
  enumerateCases,
  enumeratePositions,
  expectedSimulationsFor,
  prefixesFor,
  sha256,
  validateCorpus,
} from './cases.mjs';
import { readFixture, runCase } from './worker.mjs';

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

// --- evidence integrity: no clobber ------------------------------------------

const CLOBBER_DIR = join(REPO_ROOT, 'runs', 'mcts_golden_clobber');

test('a stale FINAL artifact refuses, and is not deleted or modified', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  rmSync(CLOBBER_DIR, { recursive: true, force: true });
  mkdirSync(CLOBBER_DIR, { recursive: true });
  const testCase = caseById('G_P01_baseline_s1');
  const stale = join(CLOBBER_DIR, 'G_P01_baseline_s1.json');
  const sentinel = '{"existing":"evidence"}\n';
  writeFileSync(stale, sentinel);

  await assert.rejects(
    () => runCase({ testCase, outDir: CLOBBER_DIR, dryRun: true }),
    (err) => err.code === 'ARTIFACT_EXISTS'
  );
  assert.equal(readFileSync(stale, 'utf8'), sentinel, 'existing evidence was altered');
});

test('a stale TEMP artifact refuses, and is not deleted or modified', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  rmSync(CLOBBER_DIR, { recursive: true, force: true });
  mkdirSync(CLOBBER_DIR, { recursive: true });
  const testCase = caseById('G_P01_baseline_s1');
  const tmp = join(CLOBBER_DIR, 'G_P01_baseline_s1.json.tmp');
  const sentinel = '{"interrupted":"write"}\n';
  writeFileSync(tmp, sentinel);

  await assert.rejects(
    () => runCase({ testCase, outDir: CLOBBER_DIR, dryRun: true }),
    (err) => err.code === 'STALE_TEMP_ARTIFACT'
  );
  assert.equal(readFileSync(tmp, 'utf8'), sentinel, 'interrupted-write residue was altered');
});

test('POSITIVE CONTROL: the same case succeeds into an empty directory', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  rmSync(CLOBBER_DIR, { recursive: true, force: true });
  const artifact = await runCase(
    { testCase: caseById('G_P01_baseline_s1'), outDir: CLOBBER_DIR, dryRun: true },
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

  const proc = spawnSync(process.execPath, [CAPTURE_MJS, 'dry-run', CLOBBER_DIR], {
    encoding: 'utf8',
  });
  assert.equal(proc.status, 3, proc.stdout + proc.stderr);
  assert.match(proc.stderr, /OUTPUT_DIR_EXISTS/);
  assert.ok(existsSync(partial), 'the partial corpus was deleted');
  assert.equal(readdirSync(CLOBBER_DIR).length, 1, 'the partial corpus was added to');
});

test('the orchestrator refuses even an EMPTY existing directory', (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  // "Empty right now" is a check-then-use race; "I created it" is not.
  rmSync(CLOBBER_DIR, { recursive: true, force: true });
  mkdirSync(CLOBBER_DIR, { recursive: true });
  const proc = spawnSync(process.execPath, [CAPTURE_MJS, 'dry-run', CLOBBER_DIR], {
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

  const args = [WORKER_MJS, 'G_P01_baseline_s1', dir, '--dry-run'];
  const run = () =>
    new Promise((resolve) => {
      const p = spawn(process.execPath, args);
      let err = '';
      p.stderr.on('data', (d) => (err += d));
      p.on('close', (code) => resolve({ code, err }));
    });

  const results = await Promise.all([run(), run()]);
  const winners = results.filter((r) => r.code === 0);
  const losers = results.filter((r) => r.code !== 0);

  assert.equal(winners.length, 1, `expected exactly one winner, got ${winners.length}`);
  assert.equal(losers.length, 1);
  assert.equal(losers[0].code, 3, losers[0].err);
  assert.match(losers[0].err, /ARTIFACT_EXISTS|STALE_TEMP_ARTIFACT/);

  // The winner's artifact is complete and no .tmp residue is left behind.
  const entries = readdirSync(dir);
  assert.deepEqual(entries, ['G_P01_baseline_s1.json'], `unexpected entries: ${entries}`);
  const artifact = JSON.parse(readFileSync(join(dir, entries[0]), 'utf8'));
  assert.equal(artifact.case_id, 'G_P01_baseline_s1');
  assert.equal(artifact.status, 'dry-run');
  rmSync(dir, { recursive: true, force: true });
});

// --- evidence integrity: fixture bytes ---------------------------------------

test('a fixture whose bytes do not match the pinned hash is refused', () => {
  const real = caseById('G_P01_baseline_s1');
  const tampered = {
    ...real,
    position: { ...real.position, sidecarSha256: 'f'.repeat(64) },
  };
  assert.throws(
    () => readFixture(tampered),
    (err) => err.code === 'FIXTURE_SHA256'
  );
});

test('POSITIVE CONTROL: the real pinned hash resolves the fixture', () => {
  const out = readFixture(caseById('G_P01_baseline_s1'));
  assert.equal(out.describe.sidecar_sha256, caseById('G_P01_baseline_s1').position.sidecarSha256);
  assert.ok(out.describe.n_legal > 400);
});

test('every pinned sidecar hash matches the committed evidence', () => {
  for (const s of SIDECARS) {
    const bytes = readFileSync(join(REPO_ROOT, FIXTURE_RELDIR, s.file));
    assert.equal(sha256(bytes), s.sha256, s.file);
  }
});

// --- evidence integrity: corpus validation ----------------------------------

const FAKE_COMMIT = 'a'.repeat(40);
const FAKE_LEGAL_KEYS = ['0,1', '0,2', '0,3', '0,4', '0,5'];
const FAKE_LEGAL_SHA = sha256(JSON.stringify(FAKE_LEGAL_KEYS));

/**
 * A synthetic corpus that is VALID for the requested mode, so perturbations can
 * be injected one at a time and attributed.
 *
 * In capture mode the traces are semantically well-formed, not merely
 * well-typed: visit counts cover the fixture's ordered legal-move keys and sum
 * to the expected simulation count, progress has the right length and done
 * sequence, and the readout follows from the counts.
 */
function fakeCorpus({ mode = 'dry-run', mutate } = {}) {
  const files = new Map();
  let pid = 1000;

  for (const c of enumerateCases()) {
    const sims = expectedSimulationsFor(c);
    let trace = null;

    if (mode === 'capture') {
      const visit_counts =
        sims === 0 ? [] : FAKE_LEGAL_KEYS.map((k, i) => [k, i === 0 ? sims : 0]);
      trace = {
        visit_counts,
        root_value: sims === 0 ? 0 : 0.25,
        selected_move: sims === 0 ? null : FAKE_LEGAL_KEYS[0],
        progress: Array.from({ length: sims }, (_, i) => ({
          done: i + 1,
          total: c.nSimulations,
          valueEstimate: 0.25,
        })),
        progress_elapsed_ms: Array.from({ length: sims }, (_, i) => i * 10),
      };
    }

    files.set(`${c.caseId}.json`, {
      schema: 'twixt-mcts-golden/1',
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
      pinned_surface_commit: '74dca6e1535ee1e36d640dae3ba644c6c2ed2e5e',
      execution_surface_sha256: PINNED_EXECUTION_SURFACE_SHA256,
      fixture: {
        sidecar: c.position.sidecar,
        sidecar_sha256: c.position.sidecarSha256,
        prefix_plies: c.position.prefixPlies,
        prefix_moves_sha256: 'b'.repeat(64),
        ply_after_prefix: c.position.prefixPlies,
        to_move: 'red',
        n_legal: FAKE_LEGAL_KEYS.length,
        legal_moves_sha256: FAKE_LEGAL_SHA,
      },
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

const codesFor = (opts) => validateCorpus('/fake', { mode: opts.mode ?? 'dry-run', ...opts.fs }).map((f) => f.code);

test('POSITIVE CONTROL: a well-formed corpus validates clean', () => {
  const failures = validateCorpus('/fake', { mode: 'dry-run', ...fakeCorpus() });
  assert.deepEqual(failures, []);
});

test('validation rejects 92 files carrying the wrong case ids', () => {
  const fs = fakeCorpus({
    mutate: (files) => {
      const a = files.get('G_P01_baseline_s1.json');
      a.case_id = 'G_P16_baseline_s800';
    },
  });
  const codes = validateCorpus('/fake', { mode: 'dry-run', ...fs }).map((f) => f.code);
  assert.ok(codes.includes('ARTIFACT_FIELD'));
});

test('validation rejects a stale execution-surface digest', () => {
  const fs = fakeCorpus({
    mutate: (files) => {
      files.get('G_P03_baseline_s8.json').execution_surface_sha256 = '0'.repeat(64);
    },
  });
  const codes = validateCorpus('/fake', { mode: 'dry-run', ...fs }).map((f) => f.code);
  assert.ok(codes.includes('ARTIFACT_FIELD'));
});

test('validation rejects a capture_commit that is not the preflighted commit', () => {
  const fs = fakeCorpus();
  // Uniform-but-unchecked is not enough: every artifact agrees on FAKE_COMMIT,
  // and it must still be rejected against the commit the orchestrator verified.
  const failures = validateCorpus('/fake', {
    mode: 'dry-run',
    expectedCaptureCommit: 'd'.repeat(40),
    ...fs,
  });
  assert.ok(failures.some((f) => f.detail?.field === 'capture_commit'));

  const clean = validateCorpus('/fake', {
    mode: 'dry-run',
    expectedCaptureCommit: FAKE_COMMIT,
    ...fakeCorpus(),
  });
  assert.deepEqual(clean, [], 'the matching commit should validate');
});

test('a reused pid is NOT a validation failure (operating systems recycle pids)', () => {
  const fs = fakeCorpus({
    mutate: (files) => {
      files.get('G_P02_baseline_s1.json').pid = files.get('G_P01_baseline_s1.json').pid;
    },
  });
  assert.deepEqual(codesFor({ fs }), [], 'pid cardinality must be diagnostic, not a gate');
});

test('validation rejects a stray non-JSON file and a leftover .tmp', () => {
  const base = fakeCorpus();
  const withStray = {
    readdirSync: () => [...base.readdirSync(), 'notes.txt', 'A1.json.tmp'],
    readFileSync: base.readFileSync,
  };
  const failures = validateCorpus('/fake', { mode: 'dry-run', ...withStray });
  const set = failures.find((f) => f.code === 'ARTIFACT_FILENAME_SET');
  assert.ok(set, 'a stray non-JSON entry was ignored');
  assert.deepEqual(set.detail.stray.sort(), ['A1.json.tmp', 'notes.txt']);
});

test('validation rejects a missing file and a stray file', () => {
  const fs = fakeCorpus({
    mutate: (files) => {
      files.delete('A2.json');
      files.set('G_P99_baseline_s1.json', { schema: 'twixt-mcts-golden/1' });
    },
  });
  const failures = validateCorpus('/fake', { mode: 'dry-run', ...fs });
  const set = failures.find((f) => f.code === 'ARTIFACT_FILENAME_SET');
  assert.ok(set);
  assert.deepEqual(set.detail.missing, ['A2.json']);
  assert.deepEqual(set.detail.stray, ['G_P99_baseline_s1.json']);
});

test('validation rejects a dry-run artifact carrying a trace', () => {
  const fs = fakeCorpus({
    mutate: (files) => {
      files.get('A1.json').trace = { visit_counts: [] };
    },
  });
  const codes = validateCorpus('/fake', { mode: 'dry-run', ...fs }).map((f) => f.code);
  assert.ok(codes.includes('ARTIFACT_FIELD'));
});

test('in capture mode, validation rejects a missing trace and an empty-object trace', () => {
  const missing = fakeCorpus({
    mutate: (files) => {
      for (const a of files.values()) a.status = 'captured';
    },
  });
  const codes = validateCorpus('/fake', { mode: 'capture', ...missing }).map((f) => f.code);
  assert.ok(codes.includes('ARTIFACT_FIELD'), 'null trace accepted in capture mode');

  const empty = fakeCorpus({
    mutate: (files) => {
      for (const a of files.values()) {
        a.status = 'captured';
        a.trace = {};
      }
    },
  });
  const emptyCodes = validateCorpus('/fake', { mode: 'capture', ...empty }).map((f) => f.code);
  assert.ok(emptyCodes.includes('ARTIFACT_FIELD'), 'an empty object passed as a trace');
});

// --- trace SEMANTICS ---------------------------------------------------------
// The review demonstrated a fabricated 92-case "capture" being certified. That
// exact corpus leads this group.

test('REGRESSION: a fabricated corpus — empty visit maps, no progress, root_value 999 — is REJECTED', () => {
  const fs = fakeCorpus({
    mode: 'capture',
    mutate: (files) => {
      for (const a of files.values()) {
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
  const failures = validateCorpus('/fake', { mode: 'capture', ...fs });
  assert.ok(failures.length > 0, 'the fabricated corpus was certified valid');
  const fields = new Set(failures.map((f) => f.detail?.field));
  assert.ok(fields.has('trace.root_value'), 'root_value 999 accepted');
  assert.ok(fields.has('trace.visit_counts'), 'empty visit map accepted for a real search');
  assert.ok(fields.has('trace.progress.length'), 'missing progress accepted');
});

test('POSITIVE CONTROL: a semantically well-formed capture corpus validates clean', () => {
  assert.deepEqual(validateCorpus('/fake', { mode: 'capture', ...fakeCorpus({ mode: 'capture' }) }), []);
});

test('capture validation rejects an incomplete or reordered visit map', () => {
  const dropped = fakeCorpus({
    mode: 'capture',
    mutate: (files) => files.get('G_P01_baseline_s8.json').trace.visit_counts.pop(),
  });
  assert.ok(codesFor({ mode: 'capture', fs: dropped }).includes('ARTIFACT_FIELD'));

  const reordered = fakeCorpus({
    mode: 'capture',
    mutate: (files) => {
      const t = files.get('G_P01_baseline_s8.json').trace;
      t.visit_counts.reverse();
      t.selected_move = FAKE_LEGAL_KEYS[0];
    },
  });
  assert.ok(
    codesFor({ mode: 'capture', fs: reordered }).includes('ARTIFACT_FIELD'),
    'a reordered visit map passed the ordered-key hash'
  );
});

test('capture validation rejects duplicate move keys and negative counts', () => {
  const dup = fakeCorpus({
    mode: 'capture',
    mutate: (files) => {
      files.get('G_P01_baseline_s8.json').trace.visit_counts[1][0] = FAKE_LEGAL_KEYS[0];
    },
  });
  assert.ok(codesFor({ mode: 'capture', fs: dup }).includes('ARTIFACT_FIELD'));

  const neg = fakeCorpus({
    mode: 'capture',
    mutate: (files) => {
      files.get('G_P01_baseline_s8.json').trace.visit_counts[1][1] = -1;
    },
  });
  assert.ok(codesFor({ mode: 'capture', fs: neg }).includes('ARTIFACT_FIELD'));
});

test('capture validation requires visit counts to sum to the simulation count', () => {
  const fs = fakeCorpus({
    mode: 'capture',
    mutate: (files) => {
      files.get('G_P01_baseline_s64.json').trace.visit_counts[0][1] = 63;
    },
  });
  const failures = validateCorpus('/fake', { mode: 'capture', ...fs });
  assert.ok(failures.some((f) => f.detail?.field === 'trace.visit_counts sum'));
});

test('capture validation enforces the abort contract for A1 and A2', () => {
  const a1 = fakeCorpus({
    mode: 'capture',
    mutate: (files) => {
      // A1 must return an EMPTY map: a populated one means it kept searching.
      files.get('A1.json').trace.visit_counts = [[FAKE_LEGAL_KEYS[0], 1]];
    },
  });
  assert.ok(codesFor({ mode: 'capture', fs: a1 }).includes('ARTIFACT_FIELD'));

  const a2 = fakeCorpus({
    mode: 'capture',
    mutate: (files) => {
      // A2 aborts at done === 5, so 64 completed simulations is impossible.
      const t = files.get('A2.json').trace;
      t.visit_counts[0][1] = 64;
      t.progress = Array.from({ length: 64 }, (_, i) => ({
        done: i + 1,
        total: 64,
        valueEstimate: 0.25,
      }));
      t.progress_elapsed_ms = Array.from({ length: 64 }, (_, i) => i);
    },
  });
  assert.ok(codesFor({ mode: 'capture', fs: a2 }).includes('ARTIFACT_FIELD'));
});

test('capture validation rejects a broken done sequence and an out-of-range valueEstimate', () => {
  const seq = fakeCorpus({
    mode: 'capture',
    mutate: (files) => {
      files.get('G_P01_baseline_s8.json').trace.progress[3].done = 99;
    },
  });
  assert.ok(codesFor({ mode: 'capture', fs: seq }).includes('ARTIFACT_FIELD'));

  const range = fakeCorpus({
    mode: 'capture',
    mutate: (files) => {
      files.get('G_P01_baseline_s8.json').trace.progress[3].valueEstimate = 7;
    },
  });
  assert.ok(codesFor({ mode: 'capture', fs: range }).includes('ARTIFACT_FIELD'));
});

test('capture validation rejects a selected_move that does not follow from the counts', () => {
  const fs = fakeCorpus({
    mode: 'capture',
    mutate: (files) => {
      files.get('G_P01_baseline_s8.json').trace.selected_move = FAKE_LEGAL_KEYS[3];
    },
  });
  const failures = validateCorpus('/fake', { mode: 'capture', ...fs });
  assert.ok(failures.some((f) => f.detail?.field === 'trace.selected_move'));
});

test('capture validation rejects non-monotonic elapsed metadata', () => {
  const fs = fakeCorpus({
    mode: 'capture',
    mutate: (files) => {
      files.get('G_P01_baseline_s64.json').trace.progress_elapsed_ms[10] = 0;
    },
  });
  assert.ok(codesFor({ mode: 'capture', fs }).includes('ARTIFACT_FIELD'));
});

test('in capture mode, validation rejects `elapsed` smuggled into a compared progress entry', () => {
  const fs = fakeCorpus({
    mode: 'capture',
    mutate: (files) => {
      files.get('G_P01_baseline_s8.json').trace.progress[0].elapsed = 12;
    },
  });
  const failures = validateCorpus('/fake', { mode: 'capture', ...fs });
  assert.ok(
    failures.some((f) => String(f.detail?.found).includes('carries elapsed')),
    'elapsed was allowed inside a compared field'
  );
});

test('the worker refuses an unknown case id', () => {
  const proc = spawnSync(process.execPath, [WORKER_MJS, 'G_NOPE_baseline_s1', OUT_DIR], {
    encoding: 'utf8',
  });
  assert.equal(proc.status, 2);
  assert.match(proc.stderr, /UNKNOWN_CASE/);
});
