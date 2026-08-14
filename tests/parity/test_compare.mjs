/**
 * Negative tests for the parity comparator's evidence validation.
 *
 *   node --test tests/parity/test_compare.mjs
 *
 * These do not re-measure anything. They take the COMMITTED evidence, corrupt
 * exactly one property of it, and assert the comparator refuses to return PASS.
 *
 * The comparator is driven as a subprocess, the same way it is really invoked,
 * so the test covers the actual entry point and its exit code rather than an
 * internal function that production might not call the same way.
 *
 * Motivation: an earlier version of the guard rejected only a total absence of
 * measurements, so a payload with 1 of the required 10 entries, or with an
 * empty permuted-logit array, still produced PARITY PASS. Presence was being
 * checked where completeness was claimed.
 */
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert';
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(HERE, '..', '..');
const RESULTS = join(HERE, 'results');
const COMPARE = join(HERE, 'compare.mjs');

const REQUIRED_POSITIONS = 10;

let dir;
let basePy;
let baseNode;

const loadGz = async (name) =>
  JSON.parse(gunzipSync(await readFile(join(RESULTS, name))).toString('utf8'));

/** Run the comparator as a subprocess; resolve with its exit code and output. */
function runCompare(pyPath, nodePath, outPath) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(
      process.execPath,
      [COMPARE, pyPath, nodePath, outPath],
      {
        cwd: REPO_ROOT,
      }
    );
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => (stdout += d));
    child.stderr.on('data', (d) => (stderr += d));
    child.on('error', reject);
    child.on('exit', (code) => resolvePromise({ code, stdout, stderr }));
  });
}

/**
 * Write a mutated copy of the committed evidence and run the comparator on it.
 * `mutate` receives deep clones, so each case starts from pristine evidence.
 */
async function runMutated(name, mutate) {
  const py = structuredClone(basePy);
  const nd = structuredClone(baseNode);
  mutate(py, nd);
  const pyPath = join(dir, `${name}_py.json`);
  const ndPath = join(dir, `${name}_node.json`);
  await writeFile(pyPath, JSON.stringify(py));
  await writeFile(ndPath, JSON.stringify(nd));
  return runCompare(pyPath, ndPath, join(dir, `${name}_out.json`));
}

/** Assert the run failed, and that `metric` is among the recorded failures. */
async function assertFailsWith(result, metric, outPath) {
  assert.strictEqual(
    result.code,
    1,
    `expected FAIL, got exit ${result.code}\n${result.stdout}`
  );
  assert.match(result.stdout, /PARITY FAIL/);
  const report = JSON.parse(await readFile(outPath, 'utf8'));
  assert.strictEqual(report.verdict, 'FAIL');
  const metrics = report.failures.map((f) => f.metric);
  assert.ok(
    metrics.includes(metric),
    `expected failure "${metric}", got: ${[...new Set(metrics)].join(', ')}`
  );
}

describe('comparator evidence validation', () => {
  before(async () => {
    dir = await mkdtemp(join(tmpdir(), 'twixt-compare-'));
    basePy = await loadGz('python_side.json.gz');
    baseNode = await loadGz('node_side.json.gz');
  });
  after(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('control: the committed evidence passes and reproduces the tracked verdict', async () => {
    const out = join(dir, 'control_out.json');
    const r = await runCompare(
      join(RESULTS, 'python_side.json.gz'),
      join(RESULTS, 'node_side.json.gz'),
      out
    );
    assert.strictEqual(r.code, 0, r.stdout);
    assert.match(r.stdout, /PARITY PASS/);
    const produced = await readFile(out, 'utf8');
    const tracked = await readFile(join(RESULTS, 'parity_result.json'), 'utf8');
    assert.strictEqual(
      produced,
      tracked,
      'verdict must reproduce byte for byte'
    );
  });

  it('the control really does carry the full evidence it claims', async () => {
    // Guards the guard: if the committed evidence had fewer than the required
    // measurements, every negative test below would pass for the wrong reason.
    assert.strictEqual(basePy.equivariance.length, REQUIRED_POSITIONS);
    assert.strictEqual(baseNode.equivariance.length, REQUIRED_POSITIONS);
    for (const e of basePy.equivariance) {
      const base = basePy.positions.find((p) => p.id === e.id);
      assert.strictEqual(e.permutation.length, base.n_legal);
      assert.strictEqual(e.mlx_permuted_logits.length, base.n_legal);
      assert.strictEqual(e.permuted_logits.length, base.n_legal);
    }
  });

  it('rejects 1 of the required 10 MLX measurements', async () => {
    const name = 'one_of_ten';
    const r = await runMutated(name, (py) => {
      for (const e of py.equivariance.slice(1)) delete e.mlx_permuted_logits;
    });
    await assertFailsWith(
      r,
      'move_order_equivariance_position_count',
      join(dir, `${name}_out.json`)
    );
  });

  it('rejects an empty MLX permuted-logit array', async () => {
    const name = 'empty_payload';
    const r = await runMutated(name, (py) => {
      py.equivariance[0].mlx_permuted_logits = [];
    });
    await assertFailsWith(
      r,
      'move_order_equivariance_payload_missing_or_empty',
      join(dir, `${name}_out.json`)
    );
  });

  it('rejects a truncated MLX payload', async () => {
    const name = 'truncated_payload';
    const r = await runMutated(name, (py) => {
      py.equivariance[0].mlx_permuted_logits =
        py.equivariance[0].mlx_permuted_logits.slice(0, 5);
    });
    await assertFailsWith(
      r,
      'move_order_equivariance_payload_length',
      join(dir, `${name}_out.json`)
    );
  });

  it('rejects a truncated permutation', async () => {
    const name = 'truncated_permutation';
    const r = await runMutated(name, (py) => {
      py.equivariance[0].permutation = py.equivariance[0].permutation.slice(
        0,
        5
      );
    });
    await assertFailsWith(
      r,
      'move_order_equivariance_permutation_length',
      join(dir, `${name}_out.json`)
    );
  });

  it('rejects a permutation that repeats an index', async () => {
    const name = 'duplicate_index';
    const r = await runMutated(name, (py) => {
      py.equivariance[0].permutation[1] = py.equivariance[0].permutation[0];
    });
    await assertFailsWith(
      r,
      'move_order_equivariance_invalid_permutation',
      join(dir, `${name}_out.json`)
    );
  });

  it('rejects a permutation index outside the legal-move range', async () => {
    const name = 'out_of_range';
    const r = await runMutated(name, (py) => {
      py.equivariance[0].permutation[0] = 99999;
    });
    await assertFailsWith(
      r,
      'move_order_equivariance_invalid_permutation',
      join(dir, `${name}_out.json`)
    );
  });

  it('rejects duplicated positions padding the count', async () => {
    const name = 'duplicate_position';
    const r = await runMutated(name, (py) => {
      // Nine real entries plus a repeat of the first: ten by length, nine by
      // coverage. Counting entries rather than distinct positions would pass.
      py.equivariance = [
        ...py.equivariance.slice(0, 9),
        structuredClone(py.equivariance[0]),
      ];
    });
    await assertFailsWith(
      r,
      'move_order_equivariance_duplicate_position',
      join(dir, `${name}_out.json`)
    );
  });

  it('rejects a missing equivariance array entirely', async () => {
    const name = 'no_array';
    const r = await runMutated(name, (py) => {
      delete py.equivariance;
    });
    await assertFailsWith(
      r,
      'move_order_equivariance_not_measured',
      join(dir, `${name}_out.json`)
    );
  });

  it('rejects surfaces measured on different position sets', async () => {
    const name = 'set_mismatch';
    const r = await runMutated(name, (py, nd) => {
      // Node measures nine of the ten the Python side used.
      nd.equivariance = nd.equivariance.slice(0, 9);
    });
    await assertFailsWith(
      r,
      'move_order_equivariance_position_count',
      join(dir, `${name}_out.json`)
    );
  });

  it('rejects surfaces measured under different permutations', async () => {
    const name = 'perm_mismatch';
    const r = await runMutated(name, (py, nd) => {
      const p = nd.equivariance[0].permutation;
      [p[0], p[1]] = [p[1], p[0]];
    });
    await assertFailsWith(
      r,
      'move_order_equivariance_permutation_mismatch',
      join(dir, `${name}_out.json`)
    );
  });

  it('still catches a genuine equivariance violation', async () => {
    // The original purpose of the gate, not just its structural preconditions.
    const name = 'not_exact';
    const r = await runMutated(name, (py) => {
      py.equivariance[0].mlx_permuted_logits[0] += 1e-9;
    });
    await assertFailsWith(
      r,
      'move_order_equivariance_not_exact',
      join(dir, `${name}_out.json`)
    );
  });
});
