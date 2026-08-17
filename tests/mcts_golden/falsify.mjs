#!/usr/bin/env node
/**
 * The preregistered falsification (design §5).
 *
 *   node tests/mcts_golden/falsify.mjs --stage eager
 *
 * Counts `TwixtState` construction during ONE search and compares it to the
 * bound the lazy design actually claims. It must FAIL against the eager
 * implementation and pass only if the copy count stays within that bound. A
 * test that passes against both would be worthless.
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
 *   3  refused (dirty worktree, wrong surface, wrong fixture, wrong model)
 *   4  error — EVERY harness fault, so a fault can never be mistaken for a
 *      gate violation
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
  STAGE_NAMES,
  STAGES,
  assertCleanWorktree,
  assertStageSurface,
  enumeratePositions,
  stageConfig,
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
 * Stages come from `cases.mjs`, so the capture harness and this one cannot
 * disagree about which surface a stage names. Each stage also fixes the outcome
 * this falsification is REQUIRED to produce there: `violated` for eager,
 * `satisfied` for lazy.
 */
export { STAGES, STAGE_NAMES };

/** Describe ANY thrown value safely — including null, undefined and primitives. */
export function describeThrown(err) {
  if (err === null) return 'null';
  if (err === undefined) return 'undefined';
  if (typeof err === 'object') {
    const msg = typeof err.message === 'string' ? err.message : '';
    return msg || Object.prototype.toString.call(err);
  }
  return String(err);
}

/** The `code` of a thrown value, if it safely has one. */
export function codeOfThrown(err) {
  if (err !== null && typeof err === 'object' && typeof err.code === 'string') return err.code;
  return null;
}

const REFUSAL_CODES = new Set([
  'WORKTREE_DIRTY',
  'FIXTURE_DRIFT',
  'EXECUTION_SURFACE_MOVED',
  'COMMIT_MOVED',
  'MODEL_ROLE',
  'UNKNOWN_POSITION',
]);

/**
 * Map any thrown value to an exit code.
 *
 * EVERY harness fault becomes `EXIT_ERROR`. This matters more than it looks:
 * `EXIT_VIOLATED` is 1, and an unhandled rejection also exits 1, so a thrown
 * `null` reaching the runtime would be indistinguishable from the very result
 * this harness exists to establish.
 */
export function exitCodeForError(err) {
  const code = codeOfThrown(err);
  return code !== null && REFUSAL_CODES.has(code) ? EXIT_REFUSED : EXIT_ERROR;
}

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
 * Run the falsification for one stage.
 *
 * Guards, in order and all before the search: clean worktree, the stage's
 * execution surface, the frozen fixture, and the model's identity. The surface,
 * commit and cleanliness are RE-CHECKED after the search, so a repository that
 * moves mid-run cannot make the code that actually ran look attributable to a
 * later commit.
 */
export async function runFalsification({ stage, loadFn = null } = {}) {
  // No default stage: a defaulted one would attribute the measurement to
  // whichever surface happened to be the default.
  const config = stageConfig(stage);

  assertCleanWorktree();
  const before = assertStageSurface(stage, executionSurfaceDigest);
  const { position, state, describe } = falsificationFixture();

  const load =
    loadFn ??
    (async (dir) => (await import('../product_match/harness.mjs')).loadModel(dir));
  const model = await load(join(REPO_ROOT, 'models', FALSIFICATION.modelId));

  let copyCount = null;
  let primaryThrew = false;
  let primary;
  try {
    // Identity BEFORE the search: the result reports the baseline model, so a
    // loader returning something else must not be able to produce a number
    // labelled as the baseline's.
    if (model?.modelId !== FALSIFICATION.modelId) {
      throw new CaptureError(
        'MODEL_ROLE',
        `models/${FALSIFICATION.modelId} resolved to ${model?.modelId}`
      );
    }
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

  // Same release discipline as the capture worker, on every path including the
  // model-role refusal: a measurement from a session that could not be released
  // has not demonstrated a clean run.
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
        `session.release() rejected: ${describeThrown(err)}`
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

  // The repository must not have moved under the measurement.
  assertCleanWorktree();
  const after = assertStageSurface(stage, executionSurfaceDigest);
  if (after.head !== before.head) {
    throw new CaptureError(
      'COMMIT_MOVED',
      `HEAD moved from ${before.head} to ${after.head} during the measurement`
    );
  }

  return {
    schema: 'twixt-mcts-falsification/1',
    stage,
    ...FALSIFICATION,
    loaded_model_id: model.modelId,
    position: { id: position.id, sidecar: position.sidecar, prefix_plies: position.prefixPlies },
    n_legal: describe.n_legal,
    copy_count: copyCount,
    gate: `copyCount <= ${FALSIFICATION.gateMaxCopies}`,
    satisfied: copyCount <= FALSIFICATION.gateMaxCopies,
    required_outcome: config.falsificationOutcome,
    execution_commit: after.head,
    execution_surface_sha256: after.digest,
  };
}

// --- CLI ---------------------------------------------------------------------

const USAGE = `usage: falsify.mjs --stage <${STAGE_NAMES.join('|')}>`;

export function parseArgs(argv) {
  let stage = null;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--stage') {
      if (stage !== null) throw new CaptureError('USAGE', '--stage given twice');
      stage = argv[++i];
      if (!stage) throw new CaptureError('USAGE', `--stage needs a value: ${USAGE}`);
    } else {
      throw new CaptureError('USAGE', `unrecognised argument ${argv[i]}: ${USAGE}`);
    }
  }
  if (stage === null) throw new CaptureError('USAGE', `--stage is required: ${USAGE}`);
  // Membership, not indexing: `STAGES['toString']` is truthy via the prototype.
  if (!STAGE_NAMES.includes(stage))
    throw new CaptureError('USAGE', `unknown stage ${stage}: ${USAGE}`);
  return { stage };
}

export async function mainWithCode(argv, { runFn = runFalsification } = {}) {
  let parsed;
  try {
    parsed = parseArgs(argv);
  } catch (err) {
    console.error(`${codeOfThrown(err) ?? 'ERROR'}: ${describeThrown(err)}`);
    return EXIT_USAGE;
  }

  let result;
  try {
    result = await runFn({ stage: parsed.stage });
  } catch (err) {
    // Any thrown value at all, including null/undefined/primitives.
    console.error(`${codeOfThrown(err) ?? 'ERROR'}: ${describeThrown(err)}`);
    return exitCodeForError(err);
  }

  console.log(JSON.stringify(result, null, 2));
  console.log('');
  console.log(
    `copyCount ${result.copy_count} vs gate ${result.gateMaxCopies} ` +
      `at ${result.n_legal} legal moves (stage: ${result.stage})`
  );
  // The verdict is about THIS measurement only: one position, one simulation
  // count. Whether allocation scales with simulations rather than with
  // simulations x legal moves is a structural claim argued in design §3, not
  // something a single copy count establishes.
  if (result.satisfied) {
    console.log('COPY-COUNT GATE SATISFIED');
    return EXIT_SATISFIED;
  }
  console.log('COPY-COUNT GATE VIOLATED');
  console.log(`(this stage requires: ${result.required_outcome})`);
  return EXIT_VIOLATED;
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
  // Even the top-level call is guarded: an escaping rejection would exit 1,
  // which is the code reserved for a gate violation.
  try {
    process.exitCode = await mainWithCode(process.argv.slice(2));
  } catch (err) {
    console.error(`ERROR: ${describeThrown(err)}`);
    process.exitCode = EXIT_ERROR;
  }
}
