#!/usr/bin/env node
/**
 * Focused tests for the §6 heap probe. The probe itself is NOT run here: it
 * loads a real model and performs the measurement, which is separately
 * authorized. Everything below uses fakes or pure functions.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

import { execFileSync } from 'node:child_process';

import { BASELINE_MODEL_ID, C_PUCT, REPO_ROOT, STAGES } from './cases.mjs';
import {
  EXIT_OK,
  EXIT_OVER_CEILING,
  EXIT_REFUSED,
  EXIT_ERROR,
  EXIT_USAGE,
  HEAP_PROBE,
  HEAP_SIZING_FLAG_STEMS,
  SEAMS,
  assertDefaultHeap,
  assertObservationsComplete,
  evaluateCriteria,
  exitCodeForError,
  heapOverrideFlags,
  heapProbeFixture,
  mainWithCode,
  runHeapProbe,
  summarize,
} from './heap_probe.mjs';

const gitClean = () =>
  execFileSync('git', ['status', '--porcelain'], { cwd: REPO_ROOT }).toString().trim() === '';

/** Deterministic stand-in for AlphaZeroInference. No model is loaded. */
const fakeModel = () => ({
  modelId: HEAP_PROBE.modelId,
  inference: {
    evaluate: async (_board, moves) => {
      const denom = (moves.length * (moves.length + 1)) / 2;
      return {
        priors: new Map(moves.map((m, i) => [`${m[0]},${m[1]}`, (moves.length - i) / denom])),
        value: 0.125,
      };
    },
    session: { release: async () => {} },
  },
});

test('every §6 parameter is frozen to its literal value', () => {
  assert.deepEqual({ ...HEAP_PROBE }, {
    positionId: 'P11',
    sidecar: 'timing_02_opening_202.json',
    prefixPlies: 28,
    modelId: BASELINE_MODEL_ID,
    nSimulations: 800,
    cPuct: C_PUCT,
    stage: 'lazy',
    ceilingBytes: 536870912,
  });
  assert.equal(HEAP_PROBE.ceilingBytes, 512 * 1024 * 1024);
  assert.equal(HEAP_PROBE.modelId, '1d64027db521a50f');
  assert.equal(HEAP_PROBE.cPuct, 1.5);
  assert.throws(() => {
    HEAP_PROBE.ceilingBytes = 1;
  }, TypeError);
});

test('the probe measures the LAZY surface', () => {
  assert.equal(HEAP_PROBE.stage, 'lazy');
  assert.equal(
    STAGES[HEAP_PROBE.stage].surfaceSha256,
    'd7fb6bc3fbc722e306940accadc2b8bdda6c92d125710b9b22c32d31dac4c769'
  );
});

test('the frozen fixture resolves to P11 @ 28 with 500 legal moves', () => {
  const { position, describe } = heapProbeFixture();
  assert.equal(position.id, 'P11');
  assert.equal(describe.prefix_plies, 28);
  assert.equal(describe.ply_after_prefix, 28);
  assert.equal(describe.n_legal, 500);
});

test('a heap-size override is detected and refused', () => {
  for (const flag of [
    '--max-old-space-size=8192',
    '--max_old_space_size=8192',
    '--max-semi-space-size=64',
    '--max-heap-size=4096',
    // Not just the --max-* family: these reconfigure the heap too, and a run
    // under any of them is not at Node's default however it finishes.
    '--initial-old-space-size=512',
    '--initial-heap-size=1024',
    '--min-semi-space-size=16',
    '--min-old-space-size=64',
    '--preconfigured-old-space-size=2048',
    '--max-young-generation-size=64',
  ]) {
    assert.deepEqual(heapOverrideFlags([flag]), [flag], flag);
  }
  // Every declared stem is actually matched, in both spellings.
  for (const stem of HEAP_SIZING_FLAG_STEMS) {
    assert.deepEqual(heapOverrideFlags([`--${stem}=1`]), [`--${stem}=1`], stem);
    const underscored = `--${stem.replace(/-/g, '_')}=1`;
    assert.deepEqual(heapOverrideFlags([underscored]), [underscored], underscored);
  }
  // ...and ordinary flags are not mistaken for one. --stack-size is the stack,
  // not the heap, and is deliberately not matched.
  assert.deepEqual(
    heapOverrideFlags(['--test', '--enable-source-maps', '-e', '--stack-size=2000']),
    []
  );
  // The live process must itself be at the default heap, or these tests are
  // running under exactly the condition the probe forbids.
  assert.doesNotThrow(() => assertDefaultHeap());
});

test('summarize reports the maximum and the per-seam counts', () => {
  const observations = [
    { seam: 'H1', heapUsed: 10 },
    { seam: 'H2.before', heapUsed: 30 },
    { seam: 'H2.after', heapUsed: 20 },
    { seam: 'H3', heapUsed: 25 },
    { seam: 'H4', heapUsed: 5 },
  ];
  const { maxHeapUsedBytes, seamCounts } = summarize(observations);
  assert.equal(maxHeapUsedBytes, 30);
  assert.deepEqual(seamCounts, { H1: 1, 'H2.before': 1, 'H2.after': 1, H3: 1, H4: 1 });
  assert.deepEqual(Object.keys(seamCounts), [...SEAMS]);
  assert.equal(summarize([]).maxHeapUsedBytes, null);
  assert.throws(
    () => summarize([{ seam: 'H9', heapUsed: 1 }]),
    (err) => err.code === 'UNKNOWN_SEAM'
  );
});

test('the criteria are exact at the boundary, with no tolerance', () => {
  const at = HEAP_PROBE.ceilingBytes;
  assert.deepEqual(evaluateCriteria({ completed: true, maxHeapUsedBytes: at }), {
    m1: true,
    m2: true,
    passed: true,
  });
  assert.equal(evaluateCriteria({ completed: true, maxHeapUsedBytes: at + 1 }).m2, false);
  assert.equal(evaluateCriteria({ completed: true, maxHeapUsedBytes: at + 1 }).passed, false);
  // M2 cannot pass without M1.
  assert.deepEqual(evaluateCriteria({ completed: false, maxHeapUsedBytes: 1 }), {
    m1: false,
    m2: false,
    passed: false,
  });
});

test('outcomes stay distinct, and no fault is reported as a ceiling failure', () => {
  assert.equal(new Set([EXIT_OK, EXIT_OVER_CEILING, EXIT_USAGE, EXIT_REFUSED, EXIT_ERROR]).size, 5);
  for (const code of ['WORKTREE_DIRTY', 'HEAP_OVERRIDE', 'EXECUTION_SURFACE_MOVED', 'MODEL_ROLE']) {
    assert.equal(exitCodeForError({ code }), EXIT_REFUSED, code);
  }
  for (const thrown of [null, undefined, 0, '', new Error('x'), { code: 'WEIRD' }]) {
    const c = exitCodeForError(thrown);
    assert.equal(c, EXIT_ERROR, String(thrown));
    assert.notEqual(c, EXIT_OVER_CEILING, 'a fault was reported as a ceiling failure');
  }
});

test('the CLI takes no arguments and maps outcomes to exit codes', async () => {
  assert.equal(await mainWithCode(['anything']), EXIT_USAGE);
  const base = {
    max_heap_used_mib: 1,
    observation_count: 3,
    m1_completed: true,
    m2_within_ceiling: true,
  };
  assert.equal(await mainWithCode([], { runFn: async () => ({ ...base, passed: true }) }), EXIT_OK);
  assert.equal(
    await mainWithCode([], {
      runFn: async () => ({ ...base, m2_within_ceiling: false, passed: false }),
    }),
    EXIT_OVER_CEILING
  );
  for (const thrown of [null, Object.freeze(new Error('frozen'))]) {
    assert.equal(
      await mainWithCode([], {
        runFn: async () => {
          throw thrown;
        },
      }),
      EXIT_ERROR,
      String(thrown)
    );
  }
});

test('the probe is not wired into any suite, and never calls process.exit', () => {
  const pkg = JSON.parse(readFileSync(join(REPO_ROOT, 'package.json'), 'utf8'));
  const CLI = 'mcts_golden/heap_probe.mjs';
  for (const [name, script] of Object.entries(pkg.scripts)) {
    if (name === 'heap-probe') continue;
    assert.equal(script.includes(CLI), false, `npm script "${name}" would run the probe`);
  }
  const src = readFileSync(join(REPO_ROOT, 'tests', 'mcts_golden', 'heap_probe.mjs'), 'utf8')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1');
  assert.equal(/process\.exit\s*\(/.test(src), false);
  assert.match(src, /process\.exitCode\s*=/);
});

// --- completeness of the observation set -------------------------------------

const COMPLETE = Object.freeze({
  H1: 1,
  'H2.before': 401,
  'H2.after': 401,
  H3: 800,
  H4: 1,
});

test('a complete observation set is accepted', () => {
  assert.doesNotThrow(() => assertObservationsComplete({ ...COMPLETE }, 800));
  // H2 at both ends of its permitted range.
  assert.doesNotThrow(() =>
    assertObservationsComplete({ ...COMPLETE, 'H2.before': 1, 'H2.after': 1 }, 800)
  );
  assert.doesNotThrow(() =>
    assertObservationsComplete({ ...COMPLETE, 'H2.before': 801, 'H2.after': 801 }, 800)
  );
});

test('an INCOMPLETE observation set is refused, never scored', () => {
  // Each of these would otherwise yield a maximum over a partial sample and
  // could pass the ceiling on that basis.
  const broken = [
    ['missing H1', { ...COMPLETE, H1: 0 }],
    ['duplicate H1', { ...COMPLETE, H1: 2 }],
    ['missing H4', { ...COMPLETE, H4: 0 }],
    ['short H3', { ...COMPLETE, H3: 799 }],
    ['no H3 at all', { ...COMPLETE, H3: 0 }],
    ['H2 halves disagree', { ...COMPLETE, 'H2.after': 400 }],
    ['H2 zero', { ...COMPLETE, 'H2.before': 0, 'H2.after': 0 }],
    ['H2 above the bound', { ...COMPLETE, 'H2.before': 802, 'H2.after': 802 }],
  ];
  for (const [label, counts] of broken) {
    assert.throws(
      () => assertObservationsComplete(counts, 800),
      (err) => err.code === 'INCOMPLETE_SAMPLING',
      label
    );
  }
});

test('an incomplete measurement is an ERROR, not a ceiling failure', () => {
  const code = exitCodeForError({ code: 'INCOMPLETE_SAMPLING' });
  assert.equal(code, EXIT_ERROR);
  assert.notEqual(code, EXIT_OVER_CEILING, 'a broken sample was reported as over the ceiling');
  assert.notEqual(code, EXIT_OK);
});

// --- the real H1-H4 wiring, with a fake inference -----------------------------

test('the real seam wiring produces a COMPLETE observation set (fake inference, no model)', async (t) => {
  if (!gitClean()) return t.skip('worktree dirty');
  // Exercises the actual sampling path end to end without loading a model, so
  // the completeness rule is checked against real seam counts rather than only
  // synthetic ones. The heapUsed number this produces is NOT the section 6
  // measurement and is not evidence of anything.
  const result = await runHeapProbe({ loadFn: async () => fakeModel() });

  assert.equal(result.completed, true);
  assert.equal(result.seam_counts.H1, 1);
  assert.equal(result.seam_counts.H4, 1);
  assert.equal(result.seam_counts.H3, HEAP_PROBE.nSimulations);
  assert.equal(result.seam_counts['H2.before'], result.seam_counts['H2.after']);
  assert.ok(result.seam_counts['H2.before'] >= 1);
  assert.ok(result.seam_counts['H2.before'] <= 1 + HEAP_PROBE.nSimulations);
  assert.equal(
    result.observation_count,
    1 + 1 + HEAP_PROBE.nSimulations + 2 * result.seam_counts['H2.before']
  );

  assert.equal(result.n_legal, 500);
  assert.equal(result.loaded_model_id, HEAP_PROBE.modelId);
  assert.equal(result.execution_surface_sha256, STAGES.lazy.surfaceSha256);
  assert.ok(Number.isFinite(result.max_heap_used_bytes));

  // The launch configuration is recorded, not merely guarded.
  assert.ok(Array.isArray(result.exec_argv));
  assert.equal(heapOverrideFlags(result.exec_argv).length, 0);
  assert.ok(result.node_options === null || typeof result.node_options === 'string');
});
