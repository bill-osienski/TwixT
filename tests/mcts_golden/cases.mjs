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
 */
export const SIDECARS = Object.freeze([
  Object.freeze({ file: 'timing_00_opening_200.json', openingId: 200, plyCount: 39 }),
  Object.freeze({ file: 'timing_01_opening_201.json', openingId: 201, plyCount: 51 }),
  Object.freeze({ file: 'timing_02_opening_202.json', openingId: 202, plyCount: 57 }),
  Object.freeze({ file: 'timing_03_opening_203.json', openingId: 203, plyCount: 54 }),
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
