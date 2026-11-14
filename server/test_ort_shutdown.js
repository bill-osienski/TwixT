/**
 * ONNX Runtime shutdown check (ledger entry 64 and its dated note; do-not-repeat #194).
 *
 * A driver that loads onnxruntime-node must end by setting `process.exitCode` and letting Node exit, never by calling
 * `process.exit()`: on macOS, onnxruntime-node 1.23.2 aborts during a forced shutdown (microsoft/onnxruntime#26445),
 * which turned Diagnostic O's COMPLETED run into exit 134. This runs the synthetic probe
 * (tests/fixtures/onnx_shutdown/probe.mjs, a 4-element Add model) three times in the natural-exit mode and requires a
 * clean exit each time. The forced modes are not run here: they are expected to abort and leave a crash report.
 */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const PROBE = join(dirname(fileURLToPath(import.meta.url)), '..', 'tests', 'fixtures', 'onnx_shutdown', 'probe.mjs');

test('a probe that sets process.exitCode exits cleanly after inference, 3 of 3', () => {
  for (let i = 0; i < 3; i++) {
    const r = spawnSync(process.execPath, [PROBE, 'natural'], { encoding: 'utf8', timeout: 60_000 });
    assert.equal(r.signal, null, `run ${i}: signal ${r.signal}; stderr: ${r.stderr}`);
    assert.equal(r.status, 0, `run ${i}: exit ${r.status}; stderr: ${r.stderr}`);
    assert.match(r.stdout, /inference ok/);
    assert.doesNotMatch(r.stderr, /mutex lock failed|terminating due to uncaught exception/);
  }
});
