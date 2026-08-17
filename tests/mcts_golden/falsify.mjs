#!/usr/bin/env node
/**
 * The preregistered falsification (design §5).
 *
 *   node tests/mcts_golden/falsify.mjs
 *
 * Counts `TwixtState` construction during ONE search and compares it to the
 * bound the lazy design actually claims. It must FAIL against the eager
 * implementation and pass only if allocation scales with simulations rather
 * than with simulations × legal moves. A test that passes against both would be
 * worthless.
 *
 * **RUNNING THIS IS SEPARATELY AUTHORIZED**, and it is deliberately NOT part of
 * `npm run test:golden`: while the eager implementation is installed this
 * harness is *supposed* to report a gate violation, so including it in the
 * ordinary suite would make a correct outcome look like a broken build.
 *
 * Exit codes:
 *   0  gate SATISFIED   (copyCount <= 8)
 *   1  gate VIOLATED    (copyCount > 8) — the REQUIRED outcome against eager code
 *   2  usage
 *   3  refused (dirty worktree, or the fixture is not the frozen one)
 *   4  error
 *
 * Specification: docs/superpowers/2026-08-16-mcts-memory-remediation-design.md §5
 */
import { join } from 'node:path';

import { TwixtState } from '../../server/gameLogic.js';
import { MCTS } from '../../server/mcts.js';
import { executionSurfaceDigest } from '../product_match/p_decision.mjs';
import {
  BASELINE_MODEL_ID,
  C_PUCT,
  CaptureError,
  REPO_ROOT,
  assertCleanWorktree,
  enumeratePositions,
} from './cases.mjs';
import { readFixture } from './worker.mjs';

export const EXIT_SATISFIED = 0;
export const EXIT_VIOLATED = 1;
export const EXIT_USAGE = 2;
export const EXIT_REFUSED = 3;
export const EXIT_ERROR = 4;

/**
 * Frozen by §5. Nothing here is a parameter: a falsification whose threshold or
 * position the caller supplies is not a falsification.
 */
export const FALSIFICATION = Object.freeze({
  positionId: 'P11',
  sidecar: 'timing_02_opening_202.json',
  prefixPlies: 28,
  modelId: BASELINE_MODEL_ID,
  nSimulations: 8,
  cPuct: C_PUCT,
  // The bound the algorithm claims: the root is not copied and each simulation
  // materialises at most one child, so S is the exact ceiling. NOT 2*(1+S),
  // which would permit more than twice the copies the design predicts.
  gateMaxCopies: 8,
});

/**
 * Count `TwixtState.prototype.copy` calls across one measured window.
 *
 * `applyMove` → `copy` is the sole construction path (design §1.1), so patching
 * `copy` counts every state the search materialises.
 *
 * The window is exactly the search: the caller replays the fixture prefix
 * BEFORE calling this, so those copies are excluded by construction rather than
 * by subtracting an estimate. The prototype is restored in `finally`, to the
 * original function object, even when the measured function throws — a leaked
 * patch would silently corrupt every later measurement in the process.
 */
export async function measureCopies(fn) {
  const original = TwixtState.prototype.copy;
  let count = 0;
  TwixtState.prototype.copy = function countingCopy(...args) {
    count += 1;
    return original.apply(this, args);
  };
  try {
    await fn();
  } finally {
    TwixtState.prototype.copy = original;
  }
  return count;
}

/** The frozen position, resolved from the pinned sidecar. */
export function falsificationFixture() {
  const position = enumeratePositions().find((p) => p.id === FALSIFICATION.positionId);
  if (!position) {
    throw new CaptureError('UNKNOWN_POSITION', `${FALSIFICATION.positionId} is not enumerated`);
  }
  if (
    position.sidecar !== FALSIFICATION.sidecar ||
    position.prefixPlies !== FALSIFICATION.prefixPlies
  ) {
    throw new CaptureError(
      'FIXTURE_DRIFT',
      `${position.id} is now ${position.sidecar}@${position.prefixPlies}, ` +
        `frozen as ${FALSIFICATION.sidecar}@${FALSIFICATION.prefixPlies}`
    );
  }
  return { position, ...readFixture({ position }) };
}

/**
 * Run the falsification.
 *
 * The execution-surface digest is RECORDED, not pinned. This harness runs twice
 * in the remediation — against the eager implementation, where it must report a
 * violation, and against the lazy one, where it must not — and `server/mcts.js`
 * is an execution-surface file, so the digest necessarily differs between those
 * runs. Pinning it would make the second run impossible. A clean worktree is
 * still required, so the measurement describes committed code.
 */
export async function runFalsification({ loadFn = null } = {}) {
  assertCleanWorktree();

  const { position, state, describe } = falsificationFixture();

  const load =
    loadFn ??
    (async (dir) => (await import('../product_match/harness.mjs')).loadModel(dir));
  const model = await load(join(REPO_ROOT, 'models', FALSIFICATION.modelId));

  let copyCount = null;
  let primaryThrew = false;
  let primary;
  try {
    const mcts = new MCTS(model.inference, {
      nSimulations: FALSIFICATION.nSimulations,
      cPuct: FALSIFICATION.cPuct,
    });
    // Spy installed here — after the prefix replay above, before search.
    copyCount = await measureCopies(() => mcts.search(state));
  } catch (err) {
    primaryThrew = true;
    primary = err;
  }

  // Same release discipline as the capture worker: a measurement taken from a
  // session that could not be released has not demonstrated a clean run.
  let releaseError = null;
  const release = model?.inference?.session?.release;
  if (typeof release !== 'function') {
    releaseError = new CaptureError(
      'SESSION_RELEASE_UNAVAILABLE',
      'the inference session exposes no callable release()'
    );
  } else {
    try {
      await model.inference.session.release();
    } catch (err) {
      releaseError = new CaptureError(
        'SESSION_RELEASE_FAILED',
        `session.release() rejected: ${err?.message ?? err}`
      );
    }
  }

  if (primaryThrew) {
    if (releaseError) {
      console.error(`SECONDARY ${releaseError.code}: ${releaseError.message}`);
      try {
        primary.secondary = releaseError;
      } catch {
        console.error('  (could not attach secondary to the primary error)');
      }
    }
    throw primary;
  }
  if (releaseError) throw releaseError;

  return {
    schema: 'twixt-mcts-falsification/1',
    ...FALSIFICATION,
    position: { id: position.id, sidecar: position.sidecar, prefix_plies: position.prefixPlies },
    n_legal: describe.n_legal,
    copy_count: copyCount,
    gate: `copyCount <= ${FALSIFICATION.gateMaxCopies}`,
    satisfied: copyCount <= FALSIFICATION.gateMaxCopies,
    execution_surface_sha256: executionSurfaceDigest('HEAD', REPO_ROOT),
  };
}

// --- CLI ---------------------------------------------------------------------

export async function mainWithCode(argv) {
  if (argv.length > 0) {
    console.error('usage: falsify.mjs   (no arguments; every parameter is frozen by §5)');
    return EXIT_USAGE;
  }
  let result;
  try {
    result = await runFalsification();
  } catch (err) {
    console.error(`${err.code ?? 'ERROR'}: ${err.message}`);
    return err.code === 'WORKTREE_DIRTY' || err.code === 'FIXTURE_DRIFT'
      ? EXIT_REFUSED
      : EXIT_ERROR;
  }

  console.log(JSON.stringify(result, null, 2));
  console.log('');
  console.log(
    `copyCount ${result.copy_count} vs gate ${result.gateMaxCopies} at ${result.n_legal} legal moves`
  );
  if (result.satisfied) {
    console.log('GATE SATISFIED — allocation scales with simulations');
    return EXIT_SATISFIED;
  }
  console.log('GATE VIOLATED — allocation scales with simulations x legal moves');
  console.log('(against the EAGER implementation this is the required outcome)');
  return EXIT_VIOLATED;
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) process.exitCode = await mainWithCode(process.argv.slice(2));
