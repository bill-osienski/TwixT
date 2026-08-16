#!/usr/bin/env node
/**
 * Capture worker: runs exactly ONE case, in its own process, then exits.
 *
 *   node tests/mcts_golden/worker.mjs <case_id> <out_dir> \
 *        --expect-commit <sha> (--dry-run | --capture)
 *
 * The mode is REQUIRED and explicit. There is no default: an optional
 * `--dry-run` flag means a typo or a forgotten argument silently performs a
 * real capture, which is the expensive, authorization-gated thing.
 *
 * One process per case is the protocol (§4.5), not an implementation detail:
 * the eager implementation's per-search retention is the thing under study, so
 * sharing a process across cases would let one case's heap pressure contaminate
 * the next — reproducing, inside the capture harness, the very confound that
 * makes the original failure hard to attribute.
 *
 * ORDER IS THE CONTRACT: preflight — cleanliness, pinned execution surface, and
 * the orchestrator's commit — runs before ANY fixture byte is read and before
 * ANY model is loaded. Tested behaviourally in test_capture.mjs.
 *
 * `server/mcts.js` has no imports of its own, so MCTS is loaded at the top
 * level. ONNX Runtime is reached only through the dynamic import inside
 * `captureTrace`, so a dry-run process structurally cannot load a model.
 *
 * Specification: docs/superpowers/2026-08-16-mcts-memory-remediation-design.md
 */
import { existsSync, readFileSync } from 'node:fs';
import { link, mkdir, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { TwixtState } from '../../server/gameLogic.js';
import { MCTS } from '../../server/mcts.js';
import { executionSurfaceDigest } from '../product_match/p_decision.mjs';
import {
  C_PUCT,
  CaptureError,
  FIXTURE_RELDIR,
  REPO_ROOT,
  artifactName,
  buildArtifact,
  caseById,
  enumeratePositions,
  preflight,
  sha256,
} from './cases.mjs';

export const EXIT_OK = 0;
export const EXIT_USAGE = 2;
export const EXIT_REFUSED = 3;
export const EXIT_FAILED = 4;

export const MODES = Object.freeze(['dry-run', 'capture']);

/** Refusals that mean "a guard said no", as opposed to "something broke". */
const REFUSAL_CODES = new Set([
  'WORKTREE_DIRTY',
  'EXECUTION_SURFACE_MOVED',
  'CAPTURE_COMMIT_MOVED',
  'FIXTURE_SHA256',
  'ARTIFACT_EXISTS',
  'STALE_TEMP_ARTIFACT',
]);

/**
 * Resolve a case's position: read the sidecar, replay its prefix, describe it.
 *
 * This function is also the INDEPENDENT source of truth used at validation
 * time (see `deriveExpectedFixtures`): the validator re-derives descriptors
 * from the pinned sidecars rather than believing the ones an artifact carries.
 */
export function readFixture(testCase) {
  const { position } = testCase;
  const path = join(REPO_ROOT, FIXTURE_RELDIR, position.sidecar);
  const bytes = readFileSync(path);

  // Bytes first, before anything is parsed or replayed. Cleanliness and the
  // execution-surface digest do NOT protect these files: a committed edit to a
  // sidecar leaves the worktree clean and the ten-file digest unchanged.
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

  const legalKeys = state.legalMoves().map((m) => `${m[0]},${m[1]}`);

  return {
    state,
    describe: {
      sidecar: position.sidecar,
      sidecar_sha256: actualSha256,
      prefix_plies: position.prefixPlies,
      prefix_moves_sha256: sha256(JSON.stringify(prefix)),
      ply_after_prefix: state.ply,
      to_move: state.toMove,
      n_legal: legalKeys.length,
      legal_moves_sha256: sha256(JSON.stringify(legalKeys)),
    },
  };
}

/**
 * Re-derive every position's descriptor from the pinned sidecars.
 *
 * The validator must not read the fixture descriptor out of the artifact it is
 * judging: a fabricated artifact can change the legal-move keys and their hash
 * together and agree with itself. These descriptors come from the committed,
 * hash-pinned sidecars instead.
 */
export function deriveExpectedFixtures() {
  const byPositionId = new Map();
  for (const position of enumeratePositions()) {
    byPositionId.set(position.id, readFixture({ position }).describe);
  }
  return byPositionId;
}

/**
 * Run the search and shape the trace. NO model loading — the inference object
 * is supplied, so this exact production code path can be exercised against a
 * deterministic fake without ONNX.
 */
export async function searchAndTrace(testCase, state, inference) {
  const mcts = new MCTS(inference, {
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
 * Load the model, trace, and REQUIRE a clean session release.
 *
 * The first real capture aborted with `mutex lock failed: Invalid argument`
 * after publishing its artifact. The session was never released, and the old
 * worker would have gone on to call `process.exit()` — though the evidence does
 * not establish that the call was reached (see
 * `capture_failures/0a76252/CORRECTIONS.md`).
 *
 * **Release is part of the result, not a courtesy.** A trace obtained from a
 * session that could not be released has not demonstrated the clean teardown
 * this whole remedy exists to establish, so it must NOT become a published
 * artifact. Optional chaining is deliberately absent: a missing `release`
 * silently skipped would certify exactly the thing being tested.
 *
 * When the trace itself failed, that error is PRIMARY and is what propagates; a
 * release failure is reported as secondary and attached, never swallowed and
 * never allowed to hide the real cause.
 *
 * `loadFn` is injectable so the lifecycle can be tested with a fake: no test
 * may load a real model at this gate.
 */
export async function captureTrace(testCase, state, loadFn = null) {
  const load =
    loadFn ??
    (async (dir) => (await import('../product_match/harness.mjs')).loadModel(dir));
  const model = await load(join(REPO_ROOT, 'models', testCase.modelId));

  // `threw` is tracked separately from the value: a thrown `null`, `0`, `''` or
  // `undefined` is a real failure, and testing the value's truthiness would
  // silently turn it into a success.
  let threw = false;
  let primary;
  let trace = null;
  try {
    if (model.modelId !== testCase.modelId) {
      throw new CaptureError(
        'MODEL_ROLE',
        `models/${testCase.modelId} resolved to ${model.modelId}`
      );
    }
    trace = await searchAndTrace(testCase, state, model.inference);
  } catch (err) {
    threw = true;
    primary = err;
  }

  let releaseError = null;
  const release = model?.inference?.session?.release;
  if (typeof release !== 'function') {
    releaseError = new CaptureError(
      'SESSION_RELEASE_UNAVAILABLE',
      'the inference session exposes no callable release(); teardown cannot be demonstrated'
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

  if (threw) {
    if (releaseError) {
      // The console line is the durable record. Attachment is BEST EFFORT:
      // a primitive, frozen or non-extensible primary would make the
      // assignment throw, and that TypeError would replace the very error this
      // branch exists to preserve.
      console.error(`SECONDARY ${releaseError.code}: ${releaseError.message}`);
      try {
        primary.secondary = releaseError;
      } catch {
        console.error('  (could not attach secondary to the primary error)');
      }
    }
    // The ORIGINAL value, unconditionally and unchanged — whatever it is.
    throw primary;
  }
  if (releaseError) throw releaseError;
  return trace;
}

/**
 * Write atomically and REFUSE rather than replace, with the kernel arbitrating.
 *
 * `wx` fails with EEXIST rather than truncating, and `link()` fails with EEXIST
 * rather than replacing — unlike `rename()`, which replaces silently and makes
 * any preceding existence check a race. Nothing is ever deleted except this
 * call's own temp file.
 */
async function writeAtomicNoClobber(path, value) {
  const tmp = `${path}.tmp`;
  try {
    await writeFile(tmp, `${JSON.stringify(value, null, 2)}\n`, { flag: 'wx' });
  } catch (err) {
    if (err.code === 'EEXIST') {
      throw new CaptureError(
        'STALE_TEMP_ARTIFACT',
        `${tmp} already exists — a previous write was interrupted, or another worker holds it; ` +
          `refusing to overwrite it`
      );
    }
    throw err;
  }

  try {
    await link(tmp, path);
  } catch (err) {
    await unlink(tmp).catch(() => {});
    if (err.code === 'EEXIST') {
      throw new CaptureError(
        'ARTIFACT_EXISTS',
        `${path} already exists; refusing to overwrite existing evidence`
      );
    }
    throw err;
  }
  await unlink(tmp);
}

/**
 * Run one case end to end.
 *
 * `mode` and `expectCommit` are REQUIRED — no defaults. A defaulted mode makes
 * the expensive path the accidental one, and a defaulted commit lets the
 * binding be disabled by omission.
 *
 * `seams` exists only so tests can observe that a refusal happens with the
 * fixtures unread and no model loaded. **`preflight` is deliberately not among
 * them** — a caller must not supply the standard it is judged by.
 */
export async function runCase({ testCase, outDir, mode, expectCommit }, seams = {}) {
  if (!MODES.includes(mode)) {
    throw new CaptureError('MODE_REQUIRED', `mode must be one of ${MODES.join('|')}, got ${mode}`);
  }
  if (typeof expectCommit !== 'string' || !/^[0-9a-f]{40}$/.test(expectCommit)) {
    throw new CaptureError(
      'EXPECT_COMMIT_REQUIRED',
      `expectCommit must be a 40-hex sha, got ${expectCommit}`
    );
  }
  const readFixtureFn = seams.readFixture ?? readFixture;
  const captureTraceFn = seams.captureTrace ?? captureTrace;
  const dryRun = mode === 'dry-run';

  // 1. Guards, before anything is read or loaded.
  const captureCommit = preflight(executionSurfaceDigest);
  if (captureCommit !== expectCommit) {
    // A clean commit made mid-run leaves the surface digest unchanged, so
    // without this every later worker would succeed under a different commit
    // and the whole corpus would only be rejected after the last case finished
    // — hours of capture thrown away.
    throw new CaptureError(
      'CAPTURE_COMMIT_MOVED',
      `HEAD is ${captureCommit} but this run was started at ${expectCommit}; ` +
        `the repository moved mid-capture`
    );
  }

  // 2. Refuse to clobber BEFORE doing the work, so a doomed case does not first
  //    spend minutes on an 800-simulation search. The binding guarantee is in
  //    writeAtomicNoClobber; these are fast-fail only.
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

const USAGE =
  'usage: worker.mjs <case_id> <out_dir> --expect-commit <sha> (--dry-run|--capture)';

/**
 * Parse argv strictly. Every unrecognised or duplicated argument is an error:
 * silently ignoring an unknown flag is how `--dryrun` becomes a real capture.
 */
export function parseArgs(argv) {
  const [caseId, outDir, ...rest] = argv;
  if (!caseId || !outDir) throw new CaptureError('USAGE', USAGE);

  let mode = null;
  let expectCommit = null;
  for (let i = 0; i < rest.length; i++) {
    const arg = rest[i];
    if (arg === '--dry-run' || arg === '--capture') {
      if (mode !== null) throw new CaptureError('USAGE', `mode given twice: ${USAGE}`);
      mode = arg.slice(2);
    } else if (arg === '--expect-commit') {
      if (expectCommit !== null) throw new CaptureError('USAGE', `--expect-commit given twice`);
      expectCommit = rest[++i];
      if (!expectCommit) throw new CaptureError('USAGE', `--expect-commit needs a value`);
    } else {
      throw new CaptureError('USAGE', `unrecognised argument ${arg}: ${USAGE}`);
    }
  }
  if (mode === null) throw new CaptureError('USAGE', `a mode is required: ${USAGE}`);
  if (expectCommit === null) throw new CaptureError('USAGE', `--expect-commit is required: ${USAGE}`);
  return { caseId, outDir, mode, expectCommit };
}

/**
 * Compute the exit code without terminating.
 *
 * Exported so the mapping can be tested directly, and so `main` never calls
 * `process.exit`: a forced exit tears the process down while native threads may
 * still be running, which is the half of the observed abort the harness
 * controls. Setting `process.exitCode` lets the event loop drain and the runtime
 * shut down in its own order.
 */
export async function mainWithCode(argv) {
  let parsed;
  let testCase;
  try {
    parsed = parseArgs(argv);
    testCase = caseById(parsed.caseId);
  } catch (err) {
    console.error(`${err.code ?? 'ERROR'}: ${err.message}`);
    return EXIT_USAGE;
  }

  try {
    const artifact = await runCase({
      testCase,
      outDir: parsed.outDir,
      mode: parsed.mode,
      expectCommit: parsed.expectCommit,
    });
    console.log(`${artifact.status} ${artifact.case_id} pid=${artifact.pid}`);
    return EXIT_OK;
  } catch (err) {
    console.error(`${err.code ?? 'ERROR'}: ${err.message}`);
    return REFUSAL_CODES.has(err.code) ? EXIT_REFUSED : EXIT_FAILED;
  }
}

async function main() {
  process.exitCode = await mainWithCode(process.argv.slice(2));
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) await main();
