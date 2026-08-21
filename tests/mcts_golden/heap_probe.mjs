#!/usr/bin/env node
/**
 * The §6 default-heap probe. ONE search, one measurement.
 *
 *   node tests/mcts_golden/heap_probe.mjs
 *
 * Design §6 fixes both the protocol and the criteria; this adds no schema and no
 * framework. It reuses the existing fixture loading (`readFixture`), model
 * loading (`loadModel`), stage preflight and session-release discipline.
 *
 * Criteria, preregistered and NOT revisable after a measurement:
 *   M1  the search COMPLETES without OOM at Node's default heap
 *   M2  the maximum observed `heapUsed` is <= 512 MiB (536,870,912 bytes)
 *
 * Sampling seams, exactly §6:
 *   H1  immediately before search()
 *   H2  immediately before AND after every evaluate()   (test-local proxy)
 *   H3  at every onProgress callback, progressEvery 1
 *   H4  immediately after search() returns
 *
 * A timer cannot observe the peak — the selection and expansion loops are
 * synchronous and block the event loop across the interval that matters — so
 * this reports the MAXIMUM OBSERVED heapUsed at those seams, and says so.
 *
 * RUNNING THIS IS SEPARATELY AUTHORIZED.
 *
 * Exit codes:
 *   0  both criteria met
 *   1  completed, but the ceiling was exceeded
 *   2  usage
 *   3  refused (dirty worktree, wrong surface, heap override, wrong model)
 *   4  error
 *
 * An OOM produces no exit code from this process: it aborts, which is itself
 * the M1 result and is captured by the operator.
 */
import { join } from 'node:path';
import { getHeapStatistics } from 'node:v8';

import { MCTS } from '../../server/mcts.js';
import { executionSurfaceDigest } from '../product_match/p_decision.mjs';
import {
  BASELINE_MODEL_ID,
  C_PUCT,
  CaptureError,
  REPO_ROOT,
  assertCleanWorktree,
  assertStageSurface,
  codeOfThrown,
  describeThrown,
  enumeratePositions,
} from './cases.mjs';
import { readFixture } from './worker.mjs';

export const EXIT_OK = 0;
export const EXIT_OVER_CEILING = 1;
export const EXIT_USAGE = 2;
export const EXIT_REFUSED = 3;
export const EXIT_ERROR = 4;

/** Frozen by §6. Nothing here is a parameter. */
export const HEAP_PROBE = Object.freeze({
  positionId: 'P11',
  sidecar: 'timing_02_opening_202.json',
  prefixPlies: 28,
  modelId: BASELINE_MODEL_ID,
  nSimulations: 800,
  cPuct: C_PUCT,
  stage: 'lazy',
  ceilingBytes: 512 * 1024 * 1024,
});

export const SEAMS = Object.freeze(['H1', 'H2.before', 'H2.after', 'H3', 'H4']);

const REFUSAL_CODES = new Set([
  'WORKTREE_DIRTY',
  'EXECUTION_SURFACE_MOVED',
  'HEAP_OVERRIDE',
  'MODEL_ROLE',
  'FIXTURE_DRIFT',
  'UNKNOWN_POSITION',
]);

export function exitCodeForError(err) {
  const code = codeOfThrown(err);
  return code !== null && REFUSAL_CODES.has(code) ? EXIT_REFUSED : EXIT_ERROR;
}

/**
 * V8 flags that change heap or GC sizing.
 *
 * Not just the `--max-*` family: `--initial-*`, `--min-*` and
 * `--preconfigured-*` reconfigure the heap too, and a run under any of them is
 * not at Node's default however it finishes. "Default heap" is the whole point
 * of M1 — a run that only completes because it was given different headroom
 * measures nothing.
 *
 * Both `-` and `_` spellings are accepted by V8, so both are matched.
 */
export const HEAP_SIZING_FLAG_STEMS = Object.freeze([
  'max-old-space-size',
  'min-old-space-size',
  'initial-old-space-size',
  'preconfigured-old-space-size',
  'max-semi-space-size',
  'min-semi-space-size',
  'max-young-generation-size',
  'max-heap-size',
  'initial-heap-size',
]);

const HEAP_SIZING_RE = new RegExp(
  `^--(?:${HEAP_SIZING_FLAG_STEMS.map((s) => s.replace(/-/g, '[-_]')).join('|')})\\b`,
  'i'
);

/** Pure, so it can be tested without spawning a process under a flag. */
export function heapOverrideFlags(flags) {
  return flags.filter((f) => HEAP_SIZING_RE.test(f));
}

export function assertDefaultHeap() {
  const flags = [
    ...process.execArgv,
    ...(process.env.NODE_OPTIONS ?? '').split(/\s+/),
  ].filter(Boolean);
  const offenders = heapOverrideFlags(flags);
  if (offenders.length) {
    throw new CaptureError(
      'HEAP_OVERRIDE',
      `the heap is not at its default: ${offenders.join(' ')}`
    );
  }
}

/** The frozen position, resolved through the existing fixture loader. */
export function heapProbeFixture() {
  const position = enumeratePositions().find((p) => p.id === HEAP_PROBE.positionId);
  if (!position) {
    throw new CaptureError('UNKNOWN_POSITION', `${HEAP_PROBE.positionId} is not enumerated`);
  }
  if (
    position.sidecar !== HEAP_PROBE.sidecar ||
    position.prefixPlies !== HEAP_PROBE.prefixPlies
  ) {
    throw new CaptureError(
      'FIXTURE_DRIFT',
      `${position.id} is now ${position.sidecar}@${position.prefixPlies}, ` +
        `frozen as ${HEAP_PROBE.sidecar}@${HEAP_PROBE.prefixPlies}`
    );
  }
  return { position, ...readFixture({ position }) };
}

/** Max and per-seam counts over the observations. Pure. */
export function summarize(observations) {
  const bySeam = Object.fromEntries(SEAMS.map((s) => [s, 0]));
  let max = -Infinity;
  for (const o of observations) {
    if (!Object.hasOwn(bySeam, o.seam)) {
      throw new CaptureError('UNKNOWN_SEAM', `unexpected seam ${String(o.seam)}`);
    }
    bySeam[o.seam] += 1;
    if (o.heapUsed > max) max = o.heapUsed;
  }
  return { maxHeapUsedBytes: observations.length ? max : null, seamCounts: bySeam };
}

/**
 * Refuse an INCOMPLETE observation set.
 *
 * M2 is a maximum over whatever was sampled, so a run missing H3 samples — or
 * one side of H2 — can report an artificially low maximum and pass. That is a
 * broken measurement, not a ceiling result, so it throws rather than returning
 * a verdict, and maps to EXIT_ERROR.
 *
 * H2 is bounded rather than fixed: one root expansion plus at most one leaf
 * expansion per simulation, and a simulation reaching a terminal leaf performs
 * none.
 */
export function assertObservationsComplete(seamCounts, nSimulations) {
  const h2Max = 1 + nSimulations;
  const problems = [];
  if (seamCounts.H1 !== 1) problems.push(`H1=${seamCounts.H1}, expected 1`);
  if (seamCounts.H4 !== 1) problems.push(`H4=${seamCounts.H4}, expected 1`);
  if (seamCounts.H3 !== nSimulations)
    problems.push(`H3=${seamCounts.H3}, expected ${nSimulations}`);
  if (seamCounts['H2.before'] !== seamCounts['H2.after'])
    problems.push(
      `H2.before=${seamCounts['H2.before']} != H2.after=${seamCounts['H2.after']}`
    );
  if (seamCounts['H2.before'] < 1 || seamCounts['H2.before'] > h2Max)
    problems.push(`H2=${seamCounts['H2.before']}, expected 1..${h2Max}`);
  if (problems.length) {
    throw new CaptureError(
      'INCOMPLETE_SAMPLING',
      `the observation set is incomplete, so its maximum is not the measurement: ${problems.join('; ')}`
    );
  }
}

/** Both criteria, applied to a completed run. Pure. */
export function evaluateCriteria({ completed, maxHeapUsedBytes }) {
  const m1 = completed === true;
  const m2 = m1 && Number.isFinite(maxHeapUsedBytes) && maxHeapUsedBytes <= HEAP_PROBE.ceilingBytes;
  return { m1, m2, passed: m1 && m2 };
}

/** Run the probe. `loadFn` is injectable so the lifecycle is testable with a fake. */
export async function runHeapProbe({ loadFn = null } = {}) {
  assertCleanWorktree();
  assertDefaultHeap();
  const { head, digest } = assertStageSurface(HEAP_PROBE.stage, executionSurfaceDigest);
  const { position, state, describe } = heapProbeFixture();

  const load =
    loadFn ??
    (async (dir) => (await import('../product_match/harness.mjs')).loadModel(dir));
  const model = await load(join(REPO_ROOT, 'models', HEAP_PROBE.modelId));

  const observations = [];
  const sample = (seam) => observations.push({ seam, heapUsed: process.memoryUsage().heapUsed });

  let completed = false;
  let threw = false;
  let primary;
  try {
    if (model?.modelId !== HEAP_PROBE.modelId) {
      throw new CaptureError(
        'MODEL_ROLE',
        `models/${HEAP_PROBE.modelId} resolved to ${model?.modelId}`
      );
    }
    // Test-local proxy: MCTS only calls evaluate(), so this is all the
    // instrumentation H2 needs and nothing reaches server/.
    const proxied = {
      evaluate: async (boardTensor, moves) => {
        sample('H2.before');
        const out = await model.inference.evaluate(boardTensor, moves);
        sample('H2.after');
        return out;
      },
    };
    const mcts = new MCTS(proxied, {
      nSimulations: HEAP_PROBE.nSimulations,
      cPuct: HEAP_PROBE.cPuct,
    });

    sample('H1');
    await mcts.search(state, { progressEvery: 1, onProgress: () => sample('H3') });
    sample('H4');
    completed = true;
  } catch (err) {
    threw = true;
    primary = err;
  }

  let releaseError = null;
  const release = model?.inference?.session?.release;
  if (typeof release !== 'function') {
    releaseError = new CaptureError('SESSION_RELEASE_UNAVAILABLE', 'no callable release()');
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

  if (threw) {
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

  const { maxHeapUsedBytes, seamCounts } = summarize(observations);
  // Completeness BEFORE any verdict: an incomplete set has no maximum worth
  // comparing to a ceiling.
  assertObservationsComplete(seamCounts, HEAP_PROBE.nSimulations);
  const criteria = evaluateCriteria({ completed, maxHeapUsedBytes });

  return {
    schema: 'twixt-mcts-heap-probe/1',
    ...HEAP_PROBE,
    loaded_model_id: model.modelId,
    position: { id: position.id, sidecar: position.sidecar, prefix_plies: position.prefixPlies },
    n_legal: describe.n_legal,
    completed,
    max_heap_used_bytes: maxHeapUsedBytes,
    max_heap_used_mib: Number((maxHeapUsedBytes / (1024 * 1024)).toFixed(2)),
    ceiling_bytes: HEAP_PROBE.ceilingBytes,
    observation_count: observations.length,
    seam_counts: seamCounts,
    heap_size_limit_bytes: getHeapStatistics().heap_size_limit,
    // The actual launch configuration, so the preserved evidence shows it
    // rather than resting on the guard having passed.
    exec_argv: [...process.execArgv],
    node_options: process.env.NODE_OPTIONS ?? null,
    m1_completed: criteria.m1,
    m2_within_ceiling: criteria.m2,
    passed: criteria.passed,
    execution_commit: head,
    execution_surface_sha256: digest,
    measurement_note:
      'Maximum OBSERVED heapUsed at the H1-H4 seams. A timer cannot observe the peak, because the selection and expansion loops block the event loop.',
  };
}

// --- CLI ---------------------------------------------------------------------

export async function mainWithCode(argv, { runFn = runHeapProbe } = {}) {
  if (argv.length > 0) {
    console.error('usage: heap_probe.mjs   (no arguments; every parameter is frozen by §6)');
    return EXIT_USAGE;
  }
  let result;
  try {
    result = await runFn({});
  } catch (err) {
    console.error(`${codeOfThrown(err) ?? 'ERROR'}: ${describeThrown(err)}`);
    return exitCodeForError(err);
  }

  console.log(JSON.stringify(result, null, 2));
  console.log('');
  console.log(
    `max observed heapUsed ${result.max_heap_used_mib} MiB vs ceiling 512 MiB ` +
      `over ${result.observation_count} observations`
  );
  console.log(`M1 completed: ${result.m1_completed}   M2 within ceiling: ${result.m2_within_ceiling}`);
  if (result.passed) {
    console.log('HEAP PROBE PASSED');
    return EXIT_OK;
  }
  console.log('HEAP PROBE FAILED — the ceiling is preregistered and may not be revised');
  return EXIT_OVER_CEILING;
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
  try {
    process.exitCode = await mainWithCode(process.argv.slice(2));
  } catch (err) {
    console.error(`ERROR: ${describeThrown(err)}`);
    process.exitCode = EXIT_ERROR;
  }
}
