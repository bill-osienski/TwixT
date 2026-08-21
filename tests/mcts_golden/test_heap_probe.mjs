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

import { BASELINE_MODEL_ID, C_PUCT, REPO_ROOT, STAGES } from './cases.mjs';
import {
  EXIT_OK,
  EXIT_OVER_CEILING,
  EXIT_REFUSED,
  EXIT_ERROR,
  EXIT_USAGE,
  HEAP_PROBE,
  SEAMS,
  assertDefaultHeap,
  evaluateCriteria,
  exitCodeForError,
  heapOverrideFlags,
  heapProbeFixture,
  mainWithCode,
  summarize,
} from './heap_probe.mjs';

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
  ]) {
    assert.deepEqual(heapOverrideFlags([flag]), [flag], flag);
  }
  // ...and ordinary flags are not mistaken for one.
  assert.deepEqual(heapOverrideFlags(['--test', '--enable-source-maps', '-e']), []);
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
