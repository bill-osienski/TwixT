/**
 * ONNX Runtime correctness canary for the product's runtime (ledger entries
 * 55-56, do-not-repeat #186).
 *
 * Python onnxruntime 1.24.4 computes same-shape Conv nodes whose filter 0 is
 * all zero with ONE parameter set, within a session and across sessions in a
 * process. The validated H-TB student has four such 128x128 convolutions, so a
 * runtime with that defect would silently corrupt its play. This test fails if
 * the onnxruntime-node the server loads has it.
 *
 * Self-contained: the ONNX graphs are encoded here from synthetic weights (an
 * integer generator shared with tests/parity/ort_conv_canary.py). No model, no
 * fixture files. Every output is compared with ITS OWN float64 reference, not
 * with the other outputs.
 */
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { test } from 'node:test';
import * as ort from 'onnxruntime-node';

const C = 128;
const H = 24;
const K = 3;
const TOL = 1e-3; // a correct runtime is ~2e-6 off; a conflated output ~5-20
const DISTINCT = 1.0;

/** Values in [-0.1, 0.1]; the Python twin is ort_conv_canary.py:gen. */
function gen(seed, n) {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const u = Math.imul(i ^ seed, 2654435761 | 0) >>> 0;
    out[i] = ((u % 20001) - 10000) / 100000;
  }
  return out;
}

export function convParams(seed, zeroFilter0) {
  const W = Float32Array.from(gen(seed, C * C * K * K));
  if (zeroFilter0) W.fill(0, 0, C * K * K);
  return { W, B: Float32Array.from(gen(seed + 1000, C)) };
}

// gen(11, 3) and gen(7, 2) from tests/parity/ort_conv_canary.py:gen (Python 3.14.10, NumPy 2.4.1).
const GEN_11_3 = [0.08155, 0.05109, -0.05379];
const GEN_7_2 = [0.01088, -0.094];

const X = Float32Array.from(gen(7, C * H * H), (v) => v * 10);

/** float64 reference: zero padding 1, stride 1. */
function reference({ W, B }) {
  const out = new Float64Array(C * H * H);
  for (let o = 0; o < C; o++) {
    const base = o * H * H;
    for (let p = 0; p < H * H; p++) out[base + p] = B[o];
    for (let c = 0; c < C; c++) {
      for (let ki = 0; ki < K; ki++) {
        for (let kj = 0; kj < K; kj++) {
          const w = W[((o * C + c) * K + ki) * K + kj];
          if (w === 0) continue;
          for (let r = 0; r < H; r++) {
            const rr = r + ki - 1;
            if (rr < 0 || rr >= H) continue;
            for (let q = 0; q < H; q++) {
              const qq = q + kj - 1;
              if (qq < 0 || qq >= H) continue;
              out[base + r * H + q] += w * X[(c * H + rr) * H + qq];
            }
          }
        }
      }
    }
  }
  return out;
}

// --- a minimal ONNX protobuf encoder (ModelProto with Conv nodes) ------------

const varint = (n) => {
  const out = [];
  let v = BigInt(n);
  do {
    let b = Number(v & 0x7fn);
    v >>= 7n;
    if (v > 0n) b |= 0x80;
    out.push(b);
  } while (v > 0n);
  return Buffer.from(out);
};
const tag = (field, wire) => varint(field * 8 + wire);
const vfield = (field, n) => Buffer.concat([tag(field, 0), varint(n)]);
const lfield = (field, payload) => Buffer.concat([tag(field, 2), varint(payload.length), payload]);
const sfield = (field, s) => lfield(field, Buffer.from(s, 'utf8'));

const tensor = (name, dims, f32) =>
  Buffer.concat([
    ...dims.map((d) => vfield(1, d)),
    vfield(2, 1), // FLOAT
    sfield(8, name),
    lfield(9, Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength)),
  ]);
const valueInfo = (name, dims) =>
  Buffer.concat([
    sfield(1, name),
    lfield(2, lfield(1, Buffer.concat([vfield(1, 1), lfield(2, Buffer.concat(dims.map((d) => lfield(1, vfield(1, d)))))]))),
  ]);
const convNode = (k) =>
  Buffer.concat([
    sfield(1, 'x'),
    sfield(1, `W${k}`),
    sfield(1, `B${k}`),
    sfield(2, `y${k}`),
    sfield(4, 'Conv'),
    lfield(5, Buffer.concat([sfield(1, 'pads'), ...[1, 1, 1, 1].map((v) => vfield(8, v)), vfield(20, 7)])),
  ]);

export function model(params) {
  const graph = Buffer.concat([
    ...params.map((_, k) => lfield(1, convNode(k))),
    sfield(2, 'canary'),
    ...params.flatMap((p, k) => [lfield(5, tensor(`W${k}`, [C, C, K, K], p.W)), lfield(5, tensor(`B${k}`, [C], p.B))]),
    lfield(11, valueInfo('x', [1, C, H, H])),
    ...params.map((_, k) => lfield(12, valueInfo(`y${k}`, [1, C, H, H]))),
  ]);
  return Buffer.concat([vfield(1, 10), lfield(8, Buffer.concat([sfield(1, ''), vfield(2, 18)])), lfield(7, graph)]);
}

async function session(params) {
  const s = await ort.InferenceSession.create(model(params));
  return async () => {
    const out = await s.run({ x: new ort.Tensor('float32', X, [1, C, H, H]) });
    return params.map((_, k) => out[`y${k}`].data);
  };
}

function maxAbsErr(y, ref) {
  let d = 0;
  for (let i = 0; i < ref.length; i++) d = Math.max(d, Math.abs(y[i] - ref[i]));
  return d;
}

function assertDistinct(refs) {
  for (let a = 0; a < refs.length; a++) {
    for (let b = a + 1; b < refs.length; b++) {
      assert.ok(maxAbsErr(refs[a], refs[b]) > DISTINCT, `references ${a} and ${b} are not distinct`);
    }
  }
}

test('the runtime under test is the pinned onnxruntime-node 1.23.2', () => {
  const require = createRequire(import.meta.url);
  assert.equal(require('onnxruntime-node/package.json').version, '1.23.2');
});

test('the generator matches its Python twin on fixed values', () => {
  // Pinned so that the two canaries cannot drift apart: gen(11, 3) and gen(7, 2) as computed by
  // tests/parity/ort_conv_canary.py:gen.
  assert.deepEqual([...gen(11, 3)], GEN_11_3);
  assert.deepEqual([...gen(7, 2)], GEN_7_2);
});

test('control: two dense Convs in one session are each correct', async () => {
  const params = [convParams(11, false), convParams(12, false)];
  const refs = params.map(reference);
  assertDistinct(refs);
  const ys = await (await session(params))();
  ys.forEach((y, k) => assert.ok(maxAbsErr(y, refs[k]) <= TOL, `conv${k}: ${maxAbsErr(y, refs[k])}`));
});

test('within one session: three Convs with an all-zero filter 0 are each correct', async () => {
  const params = [convParams(21, true), convParams(22, true), convParams(23, true)];
  const refs = params.map(reference);
  assertDistinct(refs);
  const ys = await (await session(params))();
  ys.forEach((y, k) => assert.ok(maxAbsErr(y, refs[k]) <= TOL, `conv${k}: ${maxAbsErr(y, refs[k])}`));
});

test('across sessions in one process: each zero-filter Conv stays its own', async () => {
  const a = convParams(31, true);
  const b = convParams(32, true);
  const [refA, refB] = [reference(a), reference(b)];
  assertDistinct([refA, refB]);
  const runA = await session([a]);
  const firstA = (await runA())[0];
  const runB = await session([b]);
  const yB = (await runB())[0];
  const againA = (await runA())[0];
  assert.ok(maxAbsErr(firstA, refA) <= TOL, `first session, before the second: ${maxAbsErr(firstA, refA)}`);
  assert.ok(maxAbsErr(yB, refB) <= TOL, `second session: ${maxAbsErr(yB, refB)}`);
  assert.ok(maxAbsErr(againA, refA) <= TOL, `first session, after the second: ${maxAbsErr(againA, refA)}`);
});

test('the check can fail: a conflated output is far outside the tolerance', () => {
  // CONSTRUCTED negative case: had the runtime used conv A's parameters for conv B, B's output would be A's
  // reference. That must fail B's check by a wide margin, or the tests above would be vacuous.
  const a = reference(convParams(21, true));
  const b = reference(convParams(22, true));
  assert.ok(maxAbsErr(a, b) > 100 * TOL);
});

