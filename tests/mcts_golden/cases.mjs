#!/usr/bin/env node
/**
 * The frozen case matrix, the preflight guards, and the artifact schema for the
 * MCTS golden-trace capture.
 *
 * Specification: docs/superpowers/2026-08-16-mcts-memory-remediation-design.md
 * §4.2 (corpus), §4.3 (what is compared), §4.5 (capture protocol).
 *
 * Deliberately free of any ONNX import: this module is loaded by every worker,
 * including in dry-run, and dry-run must not be able to reach a model.
 */
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = join(HERE, '..', '..');
export const FIXTURE_RELDIR = 'tests/product_match/timing_failures/74dca6e';

export const SCHEMA = 'twixt-mcts-golden/1';

/**
 * The execution surface these traces describe.
 *
 * Equality against this digest IS the statement "every execution-surface file
 * is byte-identical to 74dca6e" (§4.5), because the digest is taken over those
 * ten blobs. It is not a proxy for it.
 */
export const PINNED_SURFACE_COMMIT = '74dca6e1535ee1e36d640dae3ba644c6c2ed2e5e';
export const PINNED_EXECUTION_SURFACE_SHA256 =
  '228f57b55448f44136ffd41d6f092c9da904ca469a1e7bc4055656ffd8ef77bd';

export const BASELINE_MODEL_ID = '1d64027db521a50f';
export const CANDIDATE_MODEL_ID = 'c34b7ff3297c785a';
export const C_PUCT = 1.5;
export const MOVE_TEMP = 0;

/** §4.2. Small values are deliberate: divergence shows at simulation 2-8. */
export const N_SIMULATIONS_LADDER = Object.freeze([1, 2, 8, 64, 800]);
/** §4.2: the candidate runs on exactly these two positions, and no others. */
export const CANDIDATE_POSITION_IDS = Object.freeze(['P02', 'P11']);

/**
 * The four committed failure sidecars, with the ply counts they record.
 *
 * `plyCount` is declared here rather than read at enumeration time so the
 * matrix is a pure function: `list` must work without touching the filesystem,
 * and a sidecar edited on disk must not be able to reshape the corpus. The
 * worker verifies the declared value against the file it actually reads.
 *
 * `sha256` PINS THE BYTES. Cleanliness and the execution-surface digest between
 * them do not protect these files: a *committed* edit to a sidecar leaves the
 * worktree clean and the ten-file surface digest unchanged, and would produce
 * traces from different input bytes than the preserved failure evidence. These
 * are the hashes recorded in
 * `tests/product_match/timing_failures/74dca6e/FAILURE.md`, so agreement is a
 * gate rather than an observation.
 */
export const SIDECARS = Object.freeze([
  Object.freeze({
    file: 'timing_00_opening_200.json',
    openingId: 200,
    plyCount: 39,
    sha256: '0a63df9b9fc5b5b8d28660b92277910d9b6299a2f40dbd6ce428d1d7665122f8',
  }),
  Object.freeze({
    file: 'timing_01_opening_201.json',
    openingId: 201,
    plyCount: 51,
    sha256: '900f1c5c67337e27ae970ce946d510247cea8026c205547689c25c66e920fc36',
  }),
  Object.freeze({
    file: 'timing_02_opening_202.json',
    openingId: 202,
    plyCount: 57,
    sha256: '960a72869a72ccf93fe44faf73b34b9e4c6f23347cb02a527f5f10802a0436ce',
  }),
  Object.freeze({
    file: 'timing_03_opening_203.json',
    openingId: 203,
    plyCount: 54,
    sha256: '4d51789f3ec226b9c084f8bd5ed14a2223264e5f350ad4932546d4e986df3cc9',
  }),
]);

/**
 * §4.2 prefix rule, applied uniformly: {4, 16, 28, ply_count - 1}.
 *
 * `ply_count - 1` is the immediate-win position: every sidecar records
 * `termination: "win"`, so the move at that ply wins on the spot.
 */
export const prefixesFor = (plyCount) => [4, 16, 28, plyCount - 1];

/** §4.2 abort fixtures: both on P07, baseline, 64 simulations. */
export const ABORT_SPECS = Object.freeze([
  Object.freeze({
    caseId: 'A1',
    trigger: 'already_aborted',
    note: 'signal is already aborted when search() is called',
  }),
  Object.freeze({
    caseId: 'A2',
    trigger: 'progress_done_5',
    note: 'abort fired from onProgress at progressEvery 1 when done === 5',
  }),
]);
export const ABORT_POSITION_ID = 'P07';
export const ABORT_N_SIMULATIONS = 64;

export const EXPECTED_CASE_COUNT = 92;

export class CaptureError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'CaptureError';
    this.code = code;
  }
}

export const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

/** The 16 positions of §4.2, in P01..P16 order. */
export function enumeratePositions() {
  const positions = [];
  for (const sidecar of SIDECARS) {
    for (const prefixPlies of prefixesFor(sidecar.plyCount)) {
      positions.push(
        Object.freeze({
          id: `P${String(positions.length + 1).padStart(2, '0')}`,
          sidecar: sidecar.file,
          sidecarSha256: sidecar.sha256,
          openingId: sidecar.openingId,
          plyCount: sidecar.plyCount,
          prefixPlies,
          immediateWin: prefixPlies === sidecar.plyCount - 1,
        })
      );
    }
  }
  return Object.freeze(positions);
}

/**
 * All 92 cases, in a fixed order: 80 baseline, 10 candidate, 2 abort.
 *
 * Pure — no filesystem, no git, no model. `list` and every structural test run
 * off this, so the corpus cannot be reshaped by anything on disk.
 */
export function enumerateCases() {
  const positions = enumeratePositions();
  const byId = new Map(positions.map((p) => [p.id, p]));
  const cases = [];

  for (const position of positions) {
    for (const nSimulations of N_SIMULATIONS_LADDER) {
      cases.push({
        caseId: `G_${position.id}_baseline_s${nSimulations}`,
        kind: 'golden',
        position,
        modelId: BASELINE_MODEL_ID,
        nSimulations,
      });
    }
  }

  for (const positionId of CANDIDATE_POSITION_IDS) {
    const position = byId.get(positionId);
    if (!position) {
      throw new CaptureError(
        'UNKNOWN_CANDIDATE_POSITION',
        `${positionId} is not one of the ${positions.length} enumerated positions`
      );
    }
    for (const nSimulations of N_SIMULATIONS_LADDER) {
      cases.push({
        caseId: `G_${positionId}_candidate_s${nSimulations}`,
        kind: 'golden',
        position,
        modelId: CANDIDATE_MODEL_ID,
        nSimulations,
      });
    }
  }

  const abortPosition = byId.get(ABORT_POSITION_ID);
  if (!abortPosition) {
    throw new CaptureError(
      'UNKNOWN_ABORT_POSITION',
      `${ABORT_POSITION_ID} is not one of the enumerated positions`
    );
  }
  for (const spec of ABORT_SPECS) {
    cases.push({
      caseId: spec.caseId,
      kind: 'abort',
      position: abortPosition,
      modelId: BASELINE_MODEL_ID,
      nSimulations: ABORT_N_SIMULATIONS,
      trigger: spec.trigger,
    });
  }

  if (cases.length !== EXPECTED_CASE_COUNT) {
    throw new CaptureError(
      'CASE_COUNT',
      `enumerated ${cases.length} cases, expected ${EXPECTED_CASE_COUNT}`
    );
  }
  return Object.freeze(cases.map((c) => Object.freeze(c)));
}

export function caseById(caseId) {
  const found = enumerateCases().find((c) => c.caseId === caseId);
  if (!found) {
    throw new CaptureError('UNKNOWN_CASE', `no such case: ${caseId}`);
  }
  return found;
}

// --- preflight ---------------------------------------------------------------
// Both guards run before ANY fixture read and before ANY model load. That
// ordering is the contract (§4.5) and is tested behaviourally, not by reading
// this comment.

const git = (...args) =>
  execFileSync('git', args, { cwd: REPO_ROOT, maxBuffer: 64 * 1024 * 1024 });

/**
 * Refuse unless the ENTIRE worktree is clean.
 *
 * The surface digest covers ten files, and neither this harness nor the fixture
 * sidecars is among them — so a digest check alone would let either be edited
 * in the working tree while still passing, and the artifact would name a commit
 * that supplied neither the code that ran nor the bytes that were read.
 *
 * Tracked modifications and untracked files get identical treatment. `-uno` is
 * deliberately NOT used: dropping the untracked half is the most likely way for
 * this to stop binding later. Ignored paths are already omitted by
 * `--porcelain`, so capture output lives under gitignored `runs/`.
 *
 * A local reimplementation rather than `harness.mjs::executionCommit`, for two
 * reasons: that module pulls ONNX Runtime into the import graph of every worker
 * including dry-run, and it exposes a `requireClean: false` switch that has no
 * business existing on this path.
 */
export function assertCleanWorktree() {
  let status;
  try {
    status = git('status', '--porcelain').toString().trim();
  } catch (err) {
    throw new CaptureError('WORKTREE_STATUS_UNAVAILABLE', err.message);
  }
  if (status !== '') {
    throw new CaptureError(
      'WORKTREE_DIRTY',
      `the worktree is not clean, so the capture commit would not describe the ` +
        `code and fixtures actually used:\n${status}`
    );
  }
}

/** Refuse unless every execution-surface file is byte-identical to 74dca6e. */
export function assertPinnedExecutionSurface(executionSurfaceDigest) {
  const head = git('rev-parse', 'HEAD').toString().trim();
  const actual = executionSurfaceDigest(head, REPO_ROOT);
  if (actual !== PINNED_EXECUTION_SURFACE_SHA256) {
    throw new CaptureError(
      'EXECUTION_SURFACE_MOVED',
      `execution surface at ${head} is ${actual}, expected ` +
        `${PINNED_EXECUTION_SURFACE_SHA256} (${PINNED_SURFACE_COMMIT})`
    );
  }
  return head;
}

/**
 * The full preflight, in the contractual order.
 *
 * Returns the capture commit. `executionSurfaceDigest` is passed in by the
 * caller so this module needs no import from the execution surface it is
 * checking.
 */
export function preflight(executionSurfaceDigest) {
  assertCleanWorktree();
  return assertPinnedExecutionSurface(executionSurfaceDigest);
}

// --- artifact ----------------------------------------------------------------

/**
 * Build a capture artifact.
 *
 * `progress` carries only the COMPARED fields (done, total, valueEstimate).
 * `progressElapsedMs` is a separate parallel array, because §4.3 records
 * `elapsed` as metadata and forbids comparing it — it comes from `Date.now()`
 * and no two runs reproduce it. Keeping it out of the compared structure makes
 * that impossible to get wrong by accident rather than merely discouraged.
 */
export function buildArtifact({
  testCase,
  captureCommit,
  fixture,
  status,
  trace = null,
}) {
  return {
    schema: SCHEMA,
    status,
    case_id: testCase.caseId,
    kind: testCase.kind,
    trigger: testCase.trigger ?? null,
    position: {
      id: testCase.position.id,
      sidecar: testCase.position.sidecar,
      opening_id: testCase.position.openingId,
      prefix_plies: testCase.position.prefixPlies,
      immediate_win: testCase.position.immediateWin,
    },
    model_id: testCase.modelId,
    n_simulations: testCase.nSimulations,
    c_puct: C_PUCT,
    move_temp: MOVE_TEMP,
    capture_commit: captureCommit,
    pinned_surface_commit: PINNED_SURFACE_COMMIT,
    execution_surface_sha256: PINNED_EXECUTION_SURFACE_SHA256,
    fixture,
    pid: process.pid,
    trace: trace === null ? null : {
      visit_counts: trace.visitCounts,
      root_value: trace.rootValue,
      selected_move: trace.selectedMove,
      progress: trace.progress,
      progress_elapsed_ms: trace.progressElapsedMs,
    },
  };
}

export const artifactName = (caseId) => `${caseId}.json`;

/** The exact filename set a complete corpus must contain — no more, no fewer. */
export const expectedArtifactNames = () =>
  enumerateCases().map((c) => artifactName(c.caseId));

const isHex64 = (v) => typeof v === 'string' && /^[0-9a-f]{64}$/.test(v);
const isPosInt = (v) => Number.isInteger(v) && v > 0;

/**
 * Validate a whole corpus directory against the enumerated matrix.
 *
 * A count of files and a count of distinct PIDs certify almost nothing: 92
 * files carrying the wrong case ids, the wrong commit, a stale surface digest,
 * a missing trace or an unexpected status would satisfy both. Success is
 * declared from THIS, never from cardinality.
 *
 * Returns a list of failures; empty means the corpus is valid.
 */
export function validateCorpus(outDir, { mode, readdirSync, readFileSync }) {
  const failures = [];
  const fail = (code, detail) => failures.push({ code, detail });

  const expectedStatus = mode === 'capture' ? 'captured' : 'dry-run';
  const expected = expectedArtifactNames();

  let present;
  try {
    present = readdirSync(outDir).filter((f) => f.endsWith('.json'));
  } catch (err) {
    return [{ code: 'OUTPUT_DIR_UNREADABLE', detail: err.message }];
  }

  // Exact set equality: a stray artifact is as disqualifying as a missing one.
  const sortedPresent = [...present].sort();
  const sortedExpected = [...expected].sort();
  if (JSON.stringify(sortedPresent) !== JSON.stringify(sortedExpected)) {
    const missing = sortedExpected.filter((f) => !present.includes(f));
    const stray = sortedPresent.filter((f) => !expected.includes(f));
    fail('ARTIFACT_FILENAME_SET', { missing, stray });
  }

  const pids = new Set();
  const commits = new Set();

  for (const testCase of enumerateCases()) {
    const name = artifactName(testCase.caseId);
    if (!present.includes(name)) continue; // already reported above

    let a;
    try {
      a = JSON.parse(readFileSync(join(outDir, name), 'utf8'));
    } catch (err) {
      fail('ARTIFACT_UNPARSEABLE', { file: name, message: err.message });
      continue;
    }

    const bad = (field, found, want) =>
      fail('ARTIFACT_FIELD', { file: name, field, found, expected: want });

    if (a.schema !== SCHEMA) bad('schema', a.schema, SCHEMA);
    if (a.status !== expectedStatus) bad('status', a.status, expectedStatus);
    if (a.case_id !== testCase.caseId) bad('case_id', a.case_id, testCase.caseId);
    if (a.kind !== testCase.kind) bad('kind', a.kind, testCase.kind);
    if ((a.trigger ?? null) !== (testCase.trigger ?? null))
      bad('trigger', a.trigger, testCase.trigger ?? null);
    if (a.model_id !== testCase.modelId) bad('model_id', a.model_id, testCase.modelId);
    if (a.n_simulations !== testCase.nSimulations)
      bad('n_simulations', a.n_simulations, testCase.nSimulations);
    if (a.c_puct !== C_PUCT) bad('c_puct', a.c_puct, C_PUCT);
    if (a.move_temp !== MOVE_TEMP) bad('move_temp', a.move_temp, MOVE_TEMP);

    const p = a.position ?? {};
    if (p.id !== testCase.position.id) bad('position.id', p.id, testCase.position.id);
    if (p.sidecar !== testCase.position.sidecar)
      bad('position.sidecar', p.sidecar, testCase.position.sidecar);
    if (p.opening_id !== testCase.position.openingId)
      bad('position.opening_id', p.opening_id, testCase.position.openingId);
    if (p.prefix_plies !== testCase.position.prefixPlies)
      bad('position.prefix_plies', p.prefix_plies, testCase.position.prefixPlies);
    if (p.immediate_win !== testCase.position.immediateWin)
      bad('position.immediate_win', p.immediate_win, testCase.position.immediateWin);

    if (a.pinned_surface_commit !== PINNED_SURFACE_COMMIT)
      bad('pinned_surface_commit', a.pinned_surface_commit, PINNED_SURFACE_COMMIT);
    if (a.execution_surface_sha256 !== PINNED_EXECUTION_SURFACE_SHA256)
      bad('execution_surface_sha256', a.execution_surface_sha256, PINNED_EXECUTION_SURFACE_SHA256);
    if (typeof a.capture_commit !== 'string' || a.capture_commit.length !== 40)
      bad('capture_commit', a.capture_commit, '40-char sha');
    else commits.add(a.capture_commit);

    const f = a.fixture ?? {};
    if (f.sidecar !== testCase.position.sidecar)
      bad('fixture.sidecar', f.sidecar, testCase.position.sidecar);
    if (f.sidecar_sha256 !== testCase.position.sidecarSha256)
      bad('fixture.sidecar_sha256', f.sidecar_sha256, testCase.position.sidecarSha256);
    if (f.prefix_plies !== testCase.position.prefixPlies)
      bad('fixture.prefix_plies', f.prefix_plies, testCase.position.prefixPlies);
    if (f.ply_after_prefix !== testCase.position.prefixPlies)
      bad('fixture.ply_after_prefix', f.ply_after_prefix, testCase.position.prefixPlies);
    if (!isHex64(f.prefix_moves_sha256))
      bad('fixture.prefix_moves_sha256', f.prefix_moves_sha256, 'sha256 hex');
    if (f.to_move !== 'red' && f.to_move !== 'black')
      bad('fixture.to_move', f.to_move, 'red|black');
    if (!isPosInt(f.n_legal)) bad('fixture.n_legal', f.n_legal, 'positive integer');

    if (!isPosInt(a.pid)) bad('pid', a.pid, 'positive integer');
    else pids.add(a.pid);

    // Trace nullability is mode-determined, and its shape is checked rather
    // than its presence: an empty object would otherwise pass as "a trace".
    if (mode !== 'capture') {
      if (a.trace !== null) bad('trace', typeof a.trace, 'null in dry-run');
    } else if (a.trace === null || typeof a.trace !== 'object') {
      bad('trace', a.trace, 'object in capture');
    } else {
      const t = a.trace;
      if (!Array.isArray(t.visit_counts))
        bad('trace.visit_counts', typeof t.visit_counts, 'array of [move, count]');
      else if (
        !t.visit_counts.every(
          (e) => Array.isArray(e) && e.length === 2 && typeof e[0] === 'string' && Number.isInteger(e[1])
        )
      )
        bad('trace.visit_counts', 'malformed entries', '[move, integer] pairs');
      if (typeof t.root_value !== 'number' || !Number.isFinite(t.root_value))
        bad('trace.root_value', t.root_value, 'finite number');
      if (t.selected_move !== null && typeof t.selected_move !== 'string')
        bad('trace.selected_move', t.selected_move, 'string or null');
      if (!Array.isArray(t.progress)) bad('trace.progress', typeof t.progress, 'array');
      else if (
        !t.progress.every(
          (e) =>
            e &&
            Number.isInteger(e.done) &&
            Number.isInteger(e.total) &&
            typeof e.valueEstimate === 'number' &&
            e.elapsed === undefined
        )
      )
        bad('trace.progress', 'malformed entries', '{done,total,valueEstimate} and NO elapsed');
      if (!Array.isArray(t.progress_elapsed_ms))
        bad('trace.progress_elapsed_ms', typeof t.progress_elapsed_ms, 'array');
      else if (Array.isArray(t.progress) && t.progress_elapsed_ms.length !== t.progress.length)
        bad('trace.progress_elapsed_ms', t.progress_elapsed_ms.length, t.progress?.length);
      else if (!t.progress_elapsed_ms.every((v) => Number.isFinite(v) && v >= 0))
        bad('trace.progress_elapsed_ms', 'non-finite or negative', 'finite, >= 0');
    }
  }

  // One process per case, and one commit for the whole corpus. A corpus whose
  // halves were produced at different commits is two partial runs, not one.
  if (present.length === expected.length && pids.size !== expected.length)
    fail('PID_NOT_UNIQUE_PER_CASE', { distinct: pids.size, cases: expected.length });
  if (commits.size > 1) fail('CAPTURE_COMMIT_NOT_UNIFORM', { commits: [...commits] });

  return failures;
}
