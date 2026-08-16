#!/usr/bin/env node
/**
 * Golden-trace capture orchestrator.
 *
 *   node tests/mcts_golden/capture.mjs list
 *   node tests/mcts_golden/capture.mjs dry-run <out_dir>
 *   node tests/mcts_golden/capture.mjs capture <out_dir>
 *
 * `list` is pure — no git, no filesystem, no model — so the corpus can be
 * inspected without touching anything.
 *
 * `dry-run` performs everything except loading a model and searching: preflight,
 * fixture resolution, artifact write. It is how the harness is exercised before
 * any capture is authorized.
 *
 * RUNNING `capture` IS SEPARATELY AUTHORIZED. The capability exists; invoking it
 * is a distinct gate, exactly as `timing.mjs` documents for the timing smoke.
 * The construction of this harness authorizes no capture.
 *
 * Each case runs in a FRESH child process (§4.5). This is sequential on purpose:
 * concurrency would let cases contend for memory, and per-search retention is
 * the thing being measured.
 *
 * Specification: docs/superpowers/2026-08-16-mcts-memory-remediation-design.md
 */
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { executionSurfaceDigest } from '../product_match/p_decision.mjs';
import {
  CaptureError,
  EXPECTED_CASE_COUNT,
  enumerateCases,
  preflight,
  validateCorpus,
} from './cases.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const WORKER = join(HERE, 'worker.mjs');

export function formatCase(testCase, index) {
  return [
    String(index).padStart(2, '0'),
    testCase.caseId.padEnd(28),
    testCase.kind.padEnd(6),
    testCase.position.id,
    `${testCase.position.sidecar.replace('timing_', '').replace('.json', '')}`.padEnd(14),
    `prefix=${String(testCase.position.prefixPlies).padStart(2)}`,
    `sims=${String(testCase.nSimulations).padStart(3)}`,
    testCase.modelId,
    testCase.trigger ?? '',
  ].join('  ');
}

/**
 * Spawn one worker per case, sequentially, and stop at the first failure.
 *
 * A failing case is not skipped: a corpus missing a case is not the corpus
 * (§4.2 fixes the count at 92), so continuing past a failure would produce a
 * set that looks complete in a directory listing and is not.
 */
export function runAll({ outDir, dryRun }) {
  const cases = enumerateCases();
  const results = [];

  for (let i = 0; i < cases.length; i++) {
    const testCase = cases[i];
    const args = [WORKER, testCase.caseId, outDir];
    if (dryRun) args.push('--dry-run');

    const proc = spawnSync(process.execPath, args, { encoding: 'utf8' });
    const label = `[${String(i + 1).padStart(2)}/${cases.length}] ${testCase.caseId}`;

    if (proc.status !== 0) {
      console.error(`${label}  FAILED exit=${proc.status}`);
      if (proc.stderr) console.error(proc.stderr.trim());
      return { ok: false, results, failedAt: testCase.caseId };
    }
    console.log(`${label}  ${proc.stdout.trim()}`);
    results.push(testCase.caseId);
  }
  return { ok: true, results, failedAt: null };
}

/**
 * Refuse to start into a directory that already holds anything.
 *
 * Counting files after the fact is too late: the workers refuse individually,
 * but a run that begins against a half-finished corpus has already blurred the
 * line between two runs. Nothing is deleted — an existing directory is somebody
 * else's evidence until they say otherwise.
 */
export function createOutputDirExclusively(outDir) {
  // The LEAF is created non-recursively, so mkdir itself is the exclusion:
  // it fails with EEXIST if anything is already there. An existing directory
  // is not accepted even when empty — "empty right now" is a check-then-use
  // race, whereas "I created it" is decided by the kernel. Parents are created
  // first, since only the leaf carries the claim.
  mkdirSync(dirname(outDir), { recursive: true });
  try {
    mkdirSync(outDir);
  } catch (err) {
    if (err.code === 'EEXIST') {
      throw new CaptureError(
        'OUTPUT_DIR_EXISTS',
        `${outDir} already exists; refusing to run into a directory this run did not create. ` +
          `Move it aside or choose another — this command will not delete anything.`
      );
    }
    throw err;
  }
}

/**
 * Read back what was written, for the human-readable line only.
 *
 * NOT a verdict. Counting files and distinct PIDs certifies almost nothing —
 * 92 files with the wrong case ids, a stale surface digest or missing traces
 * would satisfy both. `validateCorpus` decides; this only prints.
 */
export function summarize(outDir) {
  const files = readdirSync(outDir).filter((f) => f.endsWith('.json'));
  const pids = new Set();
  const statuses = new Map();
  for (const f of files) {
    const a = JSON.parse(readFileSync(join(outDir, f), 'utf8'));
    pids.add(a.pid);
    statuses.set(a.status, (statuses.get(a.status) ?? 0) + 1);
  }
  return { artifacts: files.length, distinctPids: pids.size, statuses };
}

function main() {
  const [mode, outDir] = process.argv.slice(2);

  if (mode === 'list') {
    const cases = enumerateCases();
    cases.forEach((c, i) => console.log(formatCase(c, i + 1)));
    console.log(`\n${cases.length} cases (expected ${EXPECTED_CASE_COUNT})`);
    return;
  }

  if (mode !== 'dry-run' && mode !== 'capture') {
    console.error('usage: capture.mjs list | dry-run <out_dir> | capture <out_dir>');
    process.exit(2);
  }
  if (!outDir) {
    console.error(`usage: capture.mjs ${mode} <out_dir>`);
    process.exit(2);
  }

  // Fail fast before spawning 92 processes. This does NOT replace the guard —
  // every worker re-runs preflight for itself, which is what actually binds.
  let commit;
  try {
    commit = preflight(executionSurfaceDigest);
    createOutputDirExclusively(outDir);
  } catch (err) {
    console.error(`${err.code ?? 'ERROR'}: ${err.message}`);
    process.exit(3);
  }
  console.log(`capture commit ${commit}`);
  console.log(`mode           ${mode}\n`);

  const { ok, failedAt } = runAll({ outDir, dryRun: mode === 'dry-run' });
  if (!ok) {
    console.error(`\nstopped at ${failedAt}; the corpus is incomplete and is not a corpus`);
    process.exit(1);
  }

  const s = summarize(outDir);
  console.log(
    `\n${s.artifacts} artifacts, ${s.distinctPids} distinct pids, ` +
      `${[...s.statuses].map(([k, v]) => `${k}=${v}`).join(' ')}`
  );

  // The verdict comes from full validation, never from those counts.
  const failures = validateCorpus(outDir, {
    mode,
    expectedCaptureCommit: commit,
    readdirSync,
    readFileSync,
  });
  if (failures.length) {
    console.error(`\nREFUSED: ${failures.length} corpus validation failure(s)`);
    for (const f of failures.slice(0, 20)) {
      console.error(`  ${f.code} ${JSON.stringify(f.detail)}`);
    }
    if (failures.length > 20) console.error(`  ... and ${failures.length - 20} more`);
    process.exit(1);
  }
  console.log(`corpus VALID: ${EXPECTED_CASE_COUNT} cases verified against the matrix`);
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) main();
