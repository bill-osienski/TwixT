#!/usr/bin/env node
/**
 * Capture worker: runs exactly ONE case, in its own process, then exits.
 *
 *   node tests/mcts_golden/worker.mjs <case_id> <out_dir> [--dry-run]
 *
 * One process per case is the protocol (§4.5), not an implementation detail:
 * the eager implementation's per-search retention is the thing under study, so
 * sharing a process across cases would let one case's heap pressure contaminate
 * the next — reproducing, inside the capture harness, the very confound that
 * makes the original failure hard to attribute.
 *
 * ORDER IS THE CONTRACT: preflight runs before ANY fixture byte is read and
 * before ANY model is loaded. A refusal must happen with the fixtures unread
 * and no session created. This is tested behaviourally in test_capture.mjs, not
 * asserted by this comment.
 *
 * ONNX Runtime is reached only through a dynamic import inside `captureTrace`,
 * so a dry-run process structurally cannot load a model.
 *
 * Specification: docs/superpowers/2026-08-16-mcts-memory-remediation-design.md
 */
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, rename, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { TwixtState } from '../../server/gameLogic.js';
import { executionSurfaceDigest } from '../product_match/p_decision.mjs';
import {
  C_PUCT,
  CaptureError,
  FIXTURE_RELDIR,
  REPO_ROOT,
  artifactName,
  buildArtifact,
  caseById,
  preflight,
  sha256,
} from './cases.mjs';

export const EXIT_OK = 0;
export const EXIT_USAGE = 2;
export const EXIT_REFUSED = 3;
export const EXIT_FAILED = 4;

/** Refusals that mean "a guard said no", as opposed to "something broke". */
const REFUSAL_CODES = new Set([
  'WORKTREE_DIRTY',
  'EXECUTION_SURFACE_MOVED',
  'FIXTURE_SHA256',
  'ARTIFACT_EXISTS',
  'STALE_TEMP_ARTIFACT',
]);

/**
 * Resolve a case's position: read the sidecar, replay its prefix, describe it.
 *
 * Every move is asserted legal as it is applied. The sidecar's recorded
 * `ply_count` and `opening_id` are checked against the values the matrix
 * declares, so a sidecar swapped for a different game is caught rather than
 * silently reshaping the fixture.
 */
export function readFixture(testCase) {
  const { position } = testCase;
  const path = join(REPO_ROOT, FIXTURE_RELDIR, position.sidecar);
  const bytes = readFileSync(path);

  // Bytes first, before anything is parsed or replayed. Cleanliness and the
  // execution-surface digest do NOT protect these files: a committed edit to a
  // sidecar leaves the worktree clean and the ten-file digest unchanged, and
  // would silently produce traces from different input than the preserved
  // failure evidence.
  const actualSha256 = sha256(bytes);
  if (actualSha256 !== position.sidecarSha256) {
    throw new CaptureError(
      'FIXTURE_SHA256',
      `${position.sidecar} is ${actualSha256}, expected ${position.sidecarSha256} ` +
        `(the bytes preserved in ${FIXTURE_RELDIR}/FAILURE.md)`
    );
  }

  const sidecar = JSON.parse(bytes.toString('utf8'));

  if (sidecar.ply_count !== position.plyCount) {
    throw new CaptureError(
      'FIXTURE_PLY_COUNT',
      `${position.sidecar} records ply_count ${sidecar.ply_count}, matrix declares ${position.plyCount}`
    );
  }
  if (sidecar.opening_id !== position.openingId) {
    throw new CaptureError(
      'FIXTURE_OPENING_ID',
      `${position.sidecar} records opening_id ${sidecar.opening_id}, matrix declares ${position.openingId}`
    );
  }
  if (!Array.isArray(sidecar.moves) || sidecar.moves.length !== sidecar.ply_count) {
    throw new CaptureError(
      'FIXTURE_MOVES',
      `${position.sidecar} moves length ${sidecar.moves?.length} != ply_count ${sidecar.ply_count}`
    );
  }
  if (position.prefixPlies > sidecar.moves.length) {
    throw new CaptureError(
      'FIXTURE_PREFIX_TOO_LONG',
      `prefix ${position.prefixPlies} exceeds ${sidecar.moves.length} moves`
    );
  }

  const prefix = sidecar.moves.slice(0, position.prefixPlies);
  let state = new TwixtState({});
  for (let i = 0; i < prefix.length; i++) {
    const m = prefix[i];
    const legal = state.legalMoves();
    if (!legal.some((x) => x[0] === m[0] && x[1] === m[1])) {
      throw new CaptureError(
        'FIXTURE_ILLEGAL_MOVE',
        `${position.sidecar} move [${m}] is not legal at ply ${i}`
      );
    }
    state = state.applyMove(m);
  }
  if (state.isTerminal()) {
    throw new CaptureError(
      'FIXTURE_TERMINAL',
      `${position.id} is terminal after ${position.prefixPlies} plies; no search is possible`
    );
  }

  return {
    state,
    describe: {
      sidecar: position.sidecar,
      sidecar_sha256: actualSha256,
      prefix_plies: position.prefixPlies,
      prefix_moves_sha256: sha256(JSON.stringify(prefix)),
      ply_after_prefix: state.ply,
      to_move: state.toMove,
      n_legal: state.legalMoves().length,
    },
  };
}

/** Load the model and run the one search. The only path that touches ONNX. */
export async function captureTrace(testCase, state) {
  const { MCTS } = await import('../../server/mcts.js');
  const { loadModel } = await import('../product_match/harness.mjs');

  const model = await loadModel(join(REPO_ROOT, 'models', testCase.modelId));
  if (model.modelId !== testCase.modelId) {
    throw new CaptureError(
      'MODEL_ROLE',
      `models/${testCase.modelId} resolved to ${model.modelId}`
    );
  }

  const mcts = new MCTS(model.inference, {
    nSimulations: testCase.nSimulations,
    cPuct: C_PUCT,
  });

  const controller = new AbortController();
  if (testCase.trigger === 'already_aborted') controller.abort();

  const progress = [];
  const progressElapsedMs = [];
  const onProgress = ({ done, total, elapsed, valueEstimate }) => {
    // Compared fields and wall-clock metadata are kept in SEPARATE arrays, so
    // `elapsed` cannot drift into an equality comparison by accident (§4.3).
    progress.push({ done, total, valueEstimate });
    progressElapsedMs.push(elapsed);
    if (testCase.trigger === 'progress_done_5' && done === 5) controller.abort();
  };

  const { visitCounts, rootValue } = await mcts.search(state, {
    signal: controller.signal,
    onProgress,
    progressEvery: 1,
  });

  return {
    // Entries, not an object: iteration ORDER is part of what is compared (I1).
    visitCounts: [...visitCounts.entries()],
    rootValue,
    selectedMove:
      visitCounts.size === 0 ? null : mcts.selectMoveDeterministic(visitCounts),
    progress,
    progressElapsedMs,
  };
}

/**
 * Write atomically, and REFUSE rather than replace.
 *
 * `rename()` silently replaces an existing destination, so without these two
 * checks a re-run would overwrite completed evidence — including evidence from
 * a capture that had already failed, which is exactly what §9 says to preserve.
 * A stale temp file is refused too: it is the residue of an interrupted write
 * and is itself evidence. Neither guard deletes anything.
 */
async function writeAtomicNoClobber(path, value) {
  const tmp = `${path}.tmp`;
  if (existsSync(path)) {
    throw new CaptureError(
      'ARTIFACT_EXISTS',
      `${path} already exists; refusing to overwrite existing evidence`
    );
  }
  if (existsSync(tmp)) {
    throw new CaptureError(
      'STALE_TEMP_ARTIFACT',
      `${tmp} already exists, so a previous write was interrupted; refusing to overwrite it`
    );
  }
  await writeFile(tmp, `${JSON.stringify(value, null, 2)}\n`);
  await rename(tmp, path);
}

/**
 * Run one case end to end.
 *
 * `seams` exists only so tests can observe that a refusal happens with the
 * fixtures unread and no model loaded. **`preflight` is deliberately not among
 * them** — a caller must not be able to supply the standard it is judged by.
 * The CLI never passes `seams`.
 */
export async function runCase({ testCase, outDir, dryRun = false }, seams = {}) {
  const readFixtureFn = seams.readFixture ?? readFixture;
  const captureTraceFn = seams.captureTrace ?? captureTrace;

  // 1. Guards, before anything is read or loaded.
  const captureCommit = preflight(executionSurfaceDigest);

  // 2. Refuse to clobber BEFORE doing the work. Checked again at write time --
  //    this earlier check exists so a case that would be refused does not first
  //    spend minutes on an 800-simulation search.
  const finalPath = join(outDir, artifactName(testCase.caseId));
  if (existsSync(finalPath)) {
    throw new CaptureError(
      'ARTIFACT_EXISTS',
      `${finalPath} already exists; refusing to overwrite existing evidence`
    );
  }
  if (existsSync(`${finalPath}.tmp`)) {
    throw new CaptureError(
      'STALE_TEMP_ARTIFACT',
      `${finalPath}.tmp already exists, so a previous write was interrupted; refusing to overwrite it`
    );
  }

  // 3. Fixture.
  const { state, describe } = readFixtureFn(testCase);

  // 4. Model + search, unless this is a dry run.
  const trace = dryRun ? null : await captureTraceFn(testCase, state);

  const artifact = buildArtifact({
    testCase,
    captureCommit,
    fixture: describe,
    status: dryRun ? 'dry-run' : 'captured',
    trace,
  });

  await mkdir(outDir, { recursive: true });
  await writeAtomicNoClobber(finalPath, artifact);
  return artifact;
}

// --- CLI ---------------------------------------------------------------------

async function main() {
  const [caseId, outDir, ...rest] = process.argv.slice(2);
  if (!caseId || !outDir) {
    console.error('usage: worker.mjs <case_id> <out_dir> [--dry-run]');
    process.exit(EXIT_USAGE);
  }
  const dryRun = rest.includes('--dry-run');

  let testCase;
  try {
    testCase = caseById(caseId);
  } catch (err) {
    console.error(`${err.code}: ${err.message}`);
    process.exit(EXIT_USAGE);
  }

  try {
    const artifact = await runCase({ testCase, outDir, dryRun });
    console.log(`${artifact.status} ${artifact.case_id} pid=${artifact.pid}`);
    process.exit(EXIT_OK);
  } catch (err) {
    console.error(`${err.code ?? 'ERROR'}: ${err.message}`);
    process.exit(REFUSAL_CODES.has(err.code) ? EXIT_REFUSED : EXIT_FAILED);
  }
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) await main();
