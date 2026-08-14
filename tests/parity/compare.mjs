#!/usr/bin/env node
/**
 * Apply the preregistered parity gates and reach the verdict.
 *
 *   node tests/parity/compare.mjs <python_side.json> <node_side.json> <out.json>
 *
 * The gates live here, in one place, separate from both measurement halves, so
 * that neither half can decide its own result. Every threshold is transcribed
 * from the specification and none is computed from the data.
 *
 * Specification: docs/superpowers/2026-08-13-phase2-parity-specification.md
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';

/**
 * Read a measurement half, gzipped or not.
 *
 * The committed evidence is gzipped (4.73 MB -> 1.43 MB), so an auditor can
 * recompute the verdict straight from the repository with no decompression
 * step of their own.
 */
async function readSide(path) {
  const raw = await readFile(path);
  const text = path.endsWith('.gz')
    ? gunzipSync(raw).toString('utf8')
    : raw.toString('utf8');
  return JSON.parse(text);
}

// --- preregistered thresholds, transcribed from the specification -----------

const TOL = {
  S2: { maxLogit: 1e-4, meanLogit: 1e-5, maxValue: 1e-4 },
  S3: { maxLogit: 1e-5, meanLogit: 1e-6, maxValue: 1e-5 },
  S4: { maxLogit: 1.1e-4, meanLogit: 1.1e-5, maxValue: 1.1e-4 },
};
const NEAR_TIE_EXEMPTION_CAP = 6; // per surface pair, over the primary 120
// Specification §6.1: "tested on 10 primary positions with a fixed permutation
// seed". Hardcoded here rather than read from the measurement files, so a
// payload cannot satisfy the count by declaring a smaller one.
const EQUIVARIANCE_REQUIRED_POSITIONS = 10;
const TOPK_SET_AGREEMENT_MIN = 0.95;
const KENDALL_TAU_MEDIAN_MIN = 0.99;
const SIGN_CHECK_DEADBAND = 1e-3;
const PERSPECTIVE_TOL = 1e-6;

const REFERENCE = { S2: 'mlx', S3: 'ort_py', S4: 'mlx' };

// --- helpers ---------------------------------------------------------------

const maxAbs = (a, b) =>
  a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
const meanAbs = (a, b) =>
  a.length === 0
    ? 0
    : a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0) / a.length;

const argmax = (xs) => {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[best]) best = i;
  return best;
};

/** Gap between the best and second-best score; Infinity when there is no second. */
const topGap = (xs) => {
  if (xs.length < 2) return Infinity;
  let a = -Infinity;
  let b = -Infinity;
  for (const x of xs) {
    if (x > a) {
      b = a;
      a = x;
    } else if (x > b) b = x;
  }
  return a - b;
};

const topKSet = (xs, k) =>
  new Set(
    xs
      .map((v, i) => [v, i])
      .sort((p, q) => q[0] - p[0])
      .slice(0, k)
      .map(([, i]) => i)
  );

const setsEqual = (a, b) => a.size === b.size && [...a].every((x) => b.has(x));

/** Kendall tau-a. Defined as 1 for fewer than two items, per the specification. */
function kendallTau(a, b) {
  const n = a.length;
  if (n < 2) return 1;
  let concordant = 0;
  let discordant = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const da = a[i] - a[j];
      const db = b[i] - b[j];
      const s = Math.sign(da) * Math.sign(db);
      if (s > 0) concordant++;
      else if (s < 0) discordant++;
    }
  }
  return (concordant - discordant) / ((n * (n - 1)) / 2);
}

const median = (xs) => {
  const s = [...xs].sort((p, q) => p - q);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

// --- main ------------------------------------------------------------------

const [pyArg, nodeArg, outArg] = process.argv.slice(2);
if (!pyArg || !nodeArg || !outArg) {
  console.error(
    'usage: compare.mjs <python_side.json> <node_side.json> <out.json>'
  );
  process.exit(2);
}

const py = await readSide(resolve(pyArg));
const nd = await readSide(resolve(nodeArg));

const failures = [];
const fail = (surface, metric, detail) =>
  failures.push({ surface, metric, detail });

// Both halves must have measured the same corpus and the same artifact.
if (py.corpus_sha256 !== nd.corpus_sha256)
  fail('setup', 'corpus_mismatch', {
    python: py.corpus_sha256,
    node: nd.corpus_sha256,
  });
if (py.model_id !== nd.model_id)
  fail('setup', 'model_mismatch', { python: py.model_id, node: nd.model_id });
if (py.graph_sha256 !== nd.graph_sha256)
  fail('setup', 'graph_mismatch', {
    python: py.graph_sha256,
    node: nd.graph_sha256,
  });
if (py.positions.length !== nd.positions.length)
  fail('setup', 'position_count_mismatch', {
    python: py.positions.length,
    node: nd.positions.length,
  });

const byId = new Map(nd.positions.map((p) => [p.id, p]));

// --- S1: encoding parity, exact --------------------------------------------

const s1 = { positions: 0, encoding_mismatches: [], legal_move_mismatches: [] };
for (const p of py.positions) {
  const n = byId.get(p.id);
  if (!n) {
    fail('S1', 'missing_position', { id: p.id });
    continue;
  }
  s1.positions++;
  if (p.encoding_sha256 !== n.encoding_sha256) {
    s1.encoding_mismatches.push({
      id: p.id,
      python: p.encoding_sha256,
      node: n.encoding_sha256,
    });
  }
  const same =
    p.n_legal === n.n_legal &&
    p.legal_moves.every(
      (m, i) => m[0] === n.legal_moves[i][0] && m[1] === n.legal_moves[i][1]
    );
  if (!same)
    s1.legal_move_mismatches.push({
      id: p.id,
      python: p.n_legal,
      node: n.n_legal,
    });
}
if (s1.encoding_mismatches.length)
  fail('S1', 'board_tensor_not_identical', s1.encoding_mismatches);
if (s1.legal_move_mismatches.length)
  fail('S1', 'legal_move_list_not_identical', s1.legal_move_mismatches);

// --- masks, exact ----------------------------------------------------------

const maskFailures = [];
for (const p of py.positions) {
  const n = byId.get(p.id);
  if (!p.ort_py.mask_tail_all_neg1e9)
    maskFailures.push({ id: p.id, surface: 'ort_py' });
  if (n && !n.ort_node.mask_tail_all_neg1e9)
    maskFailures.push({ id: p.id, surface: 'ort_node' });
}
if (maskFailures.length)
  fail('masks', 'masked_entries_not_exactly_-1e9', maskFailures);

// --- move-order equivariance, exact ----------------------------------------

// All three surfaces are checked, native MLX included. MLX is the reference
// endpoint for S2 and S4, so omitting it would leave the reference side of two
// surfaces unmeasured while still reporting those surfaces as gated.
const EQUIVARIANCE_SIDES = ['mlx', 'ort_py', 'ort_node'];
const equivariance = Object.fromEntries(
  EQUIVARIANCE_SIDES.map((s) => [s, { checked: 0, mismatches: [] }])
);
const pyBase = new Map(py.positions.map((p) => [p.id, p]));

/**
 * Check one surface's move-order equivariance.
 *
 * Counting a measurement is not enough: an entry that is present but truncated,
 * empty, or carrying a malformed permutation would otherwise be counted and
 * then compared over zero elements, which passes trivially. Every entry is
 * therefore validated STRUCTURALLY before it is counted, and the surface must
 * end with exactly the preregistered number of distinct positions.
 */
const checkEquivariance = (side, entries, permutedOf, baseOf) => {
  const seenIds = new Set();

  if (!Array.isArray(entries)) {
    fail(side, 'move_order_equivariance_not_measured', {
      note: 'no equivariance array present for this surface',
    });
    return seenIds;
  }

  for (const e of entries) {
    const permuted = permutedOf(e);
    if (!Array.isArray(permuted) || permuted.length === 0) {
      fail(side, 'move_order_equivariance_payload_missing_or_empty', {
        id: e.id,
        length: Array.isArray(permuted) ? 0 : null,
      });
      continue;
    }

    const basePosition = baseOf(e.id);
    if (!basePosition) {
      fail(side, 'move_order_equivariance_unknown_position', { id: e.id });
      continue;
    }
    const n = basePosition.logits.length;

    if (!Array.isArray(e.permutation) || e.permutation.length !== n) {
      fail(side, 'move_order_equivariance_permutation_length', {
        id: e.id,
        permutation_length: Array.isArray(e.permutation)
          ? e.permutation.length
          : null,
        expected: n,
      });
      continue;
    }
    if (permuted.length !== n) {
      fail(side, 'move_order_equivariance_payload_length', {
        id: e.id,
        payload_length: permuted.length,
        expected: n,
      });
      continue;
    }

    // A permutation must contain every index exactly once; anything else means
    // the comparison below would not actually cover the whole output.
    const covered = new Uint8Array(n);
    let malformed = null;
    for (const idx of e.permutation) {
      if (!Number.isInteger(idx) || idx < 0 || idx >= n) {
        malformed = { reason: 'index_out_of_range', index: idx };
        break;
      }
      if (covered[idx]) {
        malformed = { reason: 'duplicate_index', index: idx };
        break;
      }
      covered[idx] = 1;
    }
    if (malformed) {
      fail(side, 'move_order_equivariance_invalid_permutation', {
        id: e.id,
        ...malformed,
      });
      continue;
    }

    if (seenIds.has(e.id)) {
      fail(side, 'move_order_equivariance_duplicate_position', { id: e.id });
      continue;
    }
    seenIds.add(e.id);
    equivariance[side].checked++;

    let worst = 0;
    for (let k = 0; k < n; k++) {
      const d = Math.abs(permuted[k] - basePosition.logits[e.permutation[k]]);
      if (d > worst) worst = d;
    }
    if (worst !== 0)
      equivariance[side].mismatches.push({ id: e.id, max_abs_diff: worst });
  }

  if (seenIds.size !== EQUIVARIANCE_REQUIRED_POSITIONS) {
    fail(side, 'move_order_equivariance_position_count', {
      valid_measurements: seenIds.size,
      required: EQUIVARIANCE_REQUIRED_POSITIONS,
    });
  }
  return seenIds;
};

const equivarianceIds = {
  mlx: checkEquivariance(
    'mlx',
    py.equivariance,
    (e) => e.mlx_permuted_logits,
    (id) => pyBase.get(id)?.mlx
  ),
  ort_py: checkEquivariance(
    'ort_py',
    py.equivariance,
    (e) => e.permuted_logits,
    (id) => pyBase.get(id)?.ort_py
  ),
  ort_node: checkEquivariance(
    'ort_node',
    nd.equivariance,
    (e) => e.permuted_logits,
    (id) => byId.get(id)?.ort_node
  ),
};

// The surfaces must have been measured on the SAME positions under the SAME
// permutation, or the three results are not comparable evidence about one gate.
const sortedIds = (s) => [...equivarianceIds[s]].sort().join(',');
for (const side of ['ort_py', 'ort_node']) {
  if (sortedIds(side) !== sortedIds('mlx'))
    fail(side, 'move_order_equivariance_position_set_mismatch', {
      mlx: sortedIds('mlx'),
      [side]: sortedIds(side),
    });
}
{
  const pyPerm = new Map(
    (py.equivariance ?? []).map((e) => [e.id, e.permutation])
  );
  for (const e of nd.equivariance ?? []) {
    const other = pyPerm.get(e.id);
    if (!other) continue;
    const same =
      Array.isArray(e.permutation) &&
      other.length === e.permutation.length &&
      other.every((v, i) => v === e.permutation[i]);
    if (!same)
      fail('ort_node', 'move_order_equivariance_permutation_mismatch', {
        id: e.id,
      });
  }
}

for (const side of EQUIVARIANCE_SIDES) {
  if (equivariance[side].mismatches.length)
    fail(
      side,
      'move_order_equivariance_not_exact',
      equivariance[side].mismatches
    );
}

// --- S2/S3/S4: numerical + ordering ----------------------------------------

const pick = (p, n, which) =>
  which === 'mlx' ? p.mlx : which === 'ort_py' ? p.ort_py : n.ort_node;

const surfaces = [
  { id: 'S2', a: 'mlx', b: 'ort_py' },
  { id: 'S3', a: 'ort_py', b: 'ort_node' },
  { id: 'S4', a: 'mlx', b: 'ort_node' },
];

const surfaceResults = {};
for (const { id, a, b } of surfaces) {
  const tol = TOL[id];
  const refName = REFERENCE[id];
  const band = 2 * tol.maxLogit;

  let maxLogit = 0;
  let meanLogitAcc = 0;
  let maxValue = 0;
  let maxLogitAt = null;
  let maxValueAt = null;
  const top1Disagreements = [];
  const exempted = [];
  const topkAgree = [];
  const taus = [];
  const signMismatches = [];
  const edgeOrdering = [];
  const edgeTop1Failures = [];

  for (const p of py.positions) {
    const n = byId.get(p.id);
    if (!n) continue;
    const A = pick(p, n, a);
    const B = pick(p, n, b);
    const isPrimary = p.stratum !== 'edge';

    const ml = maxAbs(A.logits, B.logits);
    if (ml > maxLogit) {
      maxLogit = ml;
      maxLogitAt = p.id;
    }
    meanLogitAcc += meanAbs(A.logits, B.logits);

    const mv = Math.abs(A.value - B.value);
    if (mv > maxValue) {
      maxValue = mv;
      maxValueAt = p.id;
    }

    if (
      Math.sign(A.value) !== Math.sign(B.value) &&
      Math.abs(A.value) > SIGN_CHECK_DEADBAND &&
      Math.abs(B.value) > SIGN_CHECK_DEADBAND
    ) {
      signMismatches.push({ id: p.id, a: A.value, b: B.value });
    }

    // Ordering is computed for EVERY position. Only the aggregate percentage
    // and median gates are restricted to the primary 120, so the six
    // deliberately extreme edge cases cannot dilute a rate; their ordering is
    // reported individually instead, as the specification requires.
    const ref = refName === a ? A : B;
    const topA = argmax(A.logits);
    const topB = argmax(B.logits);
    const gap = topGap(ref.logits);
    const k = Math.min(5, p.n_legal);
    const topkOk = setsEqual(topKSet(A.logits, k), topKSet(B.logits, k));
    const tau = kendallTau(A.logits, B.logits);

    if (isPrimary) {
      if (topA !== topB) {
        const entry = { id: p.id, ref_gap: gap, a_top: topA, b_top: topB };
        if (gap <= band) exempted.push(entry);
        else top1Disagreements.push(entry);
      }
      topkAgree.push(topkOk ? 1 : 0);
      taus.push(tau);
    } else {
      edgeOrdering.push({
        id: p.id,
        n_legal: p.n_legal,
        k,
        top1_agree: topA === topB,
        top1_a: topA,
        top1_b: topB,
        ref_gap: gap === Infinity ? null : gap,
        within_near_tie_band: gap <= band,
        topk_set_agree: topkOk,
        kendall_tau: tau,
      });
      // A per-position rule, unlike the percentage and median gates: an edge
      // top-1 flip outside the ambiguous band is a failure wherever it occurs.
      if (topA !== topB && gap > band) {
        edgeTop1Failures.push({
          id: p.id,
          ref_gap: gap,
          a_top: topA,
          b_top: topB,
        });
      }
    }
  }

  const nPos = py.positions.length;
  const res = {
    reference: refName,
    near_tie_band: band,
    max_abs_logit_diff: maxLogit,
    max_abs_logit_at: maxLogitAt,
    mean_abs_logit_diff: meanLogitAcc / nPos,
    max_abs_value_diff: maxValue,
    max_abs_value_at: maxValueAt,
    top1_disagreements_outside_band: top1Disagreements,
    near_tie_exemptions: exempted.length,
    near_tie_exemption_detail: exempted,
    topk_set_agreement: topkAgree.reduce((s, v) => s + v, 0) / topkAgree.length,
    kendall_tau_median: median(taus),
    kendall_tau_min: Math.min(...taus),
    aggregate_gates_scope: `primary ${taus.length} positions only`,
    edge_ordering: edgeOrdering,
    edge_ordering_note:
      'Reported individually, per the specification. These entries are deliberately extreme, so they are excluded from the percentage and median gates above; a top-1 flip outside the near-tie band is still a failure here, since that is a per-position rule rather than a rate.',
    edge_top1_failures: edgeTop1Failures,
    sign_mismatches: signMismatches,
    tolerances: tol,
    margins: {
      max_abs_logit: tol.maxLogit - maxLogit,
      mean_abs_logit: tol.meanLogit - meanLogitAcc / nPos,
      max_abs_value: tol.maxValue - maxValue,
    },
  };
  surfaceResults[id] = res;

  if (res.max_abs_logit_diff > tol.maxLogit)
    fail(id, 'max_abs_logit_diff', {
      measured: res.max_abs_logit_diff,
      limit: tol.maxLogit,
    });
  if (res.mean_abs_logit_diff > tol.meanLogit)
    fail(id, 'mean_abs_logit_diff', {
      measured: res.mean_abs_logit_diff,
      limit: tol.meanLogit,
    });
  if (res.max_abs_value_diff > tol.maxValue)
    fail(id, 'max_abs_value_diff', {
      measured: res.max_abs_value_diff,
      limit: tol.maxValue,
    });
  if (top1Disagreements.length > 0)
    fail(id, 'top1_disagreement_outside_near_tie_band', top1Disagreements);
  if (edgeTop1Failures.length > 0)
    fail(id, 'edge_top1_disagreement_outside_near_tie_band', edgeTop1Failures);
  if (exempted.length > NEAR_TIE_EXEMPTION_CAP)
    fail(id, 'near_tie_exemptions_exceed_cap', {
      exemptions: exempted.length,
      cap: NEAR_TIE_EXEMPTION_CAP,
    });
  if (res.topk_set_agreement < TOPK_SET_AGREEMENT_MIN)
    fail(id, 'topk_set_agreement', {
      measured: res.topk_set_agreement,
      minimum: TOPK_SET_AGREEMENT_MIN,
    });
  if (res.kendall_tau_median < KENDALL_TAU_MEDIAN_MIN)
    fail(id, 'kendall_tau_median', {
      measured: res.kendall_tau_median,
      minimum: KENDALL_TAU_MEDIAN_MIN,
    });
  if (signMismatches.length)
    fail(id, 'value_sign_disagreement', signMismatches);
}

// --- value range + red-perspective conversion ------------------------------

const rangeViolations = [];
for (const p of py.positions) {
  const n = byId.get(p.id);
  for (const [surface, v] of [
    ['mlx', p.mlx.value],
    ['ort_py', p.ort_py.value],
    ['ort_node', n ? n.ort_node.value : 0],
  ]) {
    if (!(v >= -1 && v <= 1))
      rangeViolations.push({ id: p.id, surface, value: v });
  }
}
if (rangeViolations.length)
  fail('value', 'raw_value_out_of_range', rangeViolations);

let perspectiveMax = 0;
const perspectiveMismatches = [];
for (const p of py.positions) {
  const n = byId.get(p.id);
  if (!n) continue;
  for (let i = 0; i < p.red_perspective_probes.length; i++) {
    const d = Math.abs(
      p.red_perspective_probes[i] - n.red_perspective_probes[i]
    );
    if (d > perspectiveMax) perspectiveMax = d;
    if (d > PERSPECTIVE_TOL)
      perspectiveMismatches.push({
        id: p.id,
        probe: i,
        python: p.red_perspective_probes[i],
        node: n.red_perspective_probes[i],
      });
  }
}
if (perspectiveMismatches.length)
  fail('value', 'red_perspective_conversion', perspectiveMismatches);

// --- verdict ---------------------------------------------------------------

const verdict = failures.length === 0 ? 'PASS' : 'FAIL';

const report = {
  schema: 'twixt-parity-result/1',
  specification: 'docs/superpowers/2026-08-13-phase2-parity-specification.md',
  verdict,
  positions_compared: py.positions.length,
  primary_compared: py.positions.filter((p) => p.stratum !== 'edge').length,
  edge_compared: py.positions.filter((p) => p.stratum === 'edge').length,
  audit: {
    recompute:
      'node tests/parity/compare.mjs tests/parity/results/python_side.json.gz tests/parity/results/node_side.json.gz /tmp/verdict.json',
    note: 'Reproduces this file byte for byte from the committed gzipped measurement halves; compare.mjs reads either form. This makes the VERDICT auditable from the repository alone. It does NOT make the measurement reproducible from scratch: the source checkpoint is untracked, so regenerating the halves requires calib020_0001 (SHA-1 209cf2d4fd24a48553d259dd71b4954867b9473e) from outside the repository.',
  },
  corpus_sha256: py.corpus_sha256,
  model_id: py.model_id,
  graph_sha256: py.graph_sha256,
  source_checkpoint_sha1: py.source_checkpoint_sha1,
  scratch_export_path: py.scratch_export_path,
  node_loaded_through_manifest: nd.loaded_through_manifest === true,
  environment: { python_side: py.environment, node_side: nd.environment },
  gates: {
    s1_encoding: {
      positions: s1.positions,
      encoding_mismatches: s1.encoding_mismatches.length,
      legal_move_mismatches: s1.legal_move_mismatches.length,
      rule: 'exact equality, tolerance zero',
      mechanism:
        'SHA-256 over canonical little-endian float32 NCHW bytes on both sides',
    },
    masks: {
      violations: maskFailures.length,
      rule: 'every index >= n_legal is exactly -1e9',
      note: 'Checked on both ONNX Runtime surfaces. Native MLX returns only n_legal logits and exposes no padded tail, so the check is not applicable there rather than skipped.',
    },
    equivariance,
    surfaces: surfaceResults,
    value: {
      raw_range_violations: rangeViolations.length,
      red_perspective_max_abs_diff: perspectiveMax,
      red_perspective_tolerance: PERSPECTIVE_TOL,
      note: 'Both conversions are applied to the same fixed synthetic side-to-move values, so this isolates the conversion logic instead of re-measuring the S3 model difference.',
    },
  },
  failures,
};

const outPath = resolve(outArg);
await mkdir(dirname(outPath), { recursive: true });
await writeFile(outPath, JSON.stringify(report, null, 2));

// --- human summary ---------------------------------------------------------

const f = (x) => (x === 0 ? '0' : x.toExponential(3));
console.log(`\nPARITY ${verdict}  (${report.positions_compared} positions)\n`);
console.log(
  'S1 encoding   exact:',
  s1.encoding_mismatches.length === 0 ? 'PASS' : 'FAIL'
);
console.log(
  '   legal moves:     ',
  s1.legal_move_mismatches.length === 0 ? 'PASS' : 'FAIL'
);
console.log(
  'masks         exact:',
  maskFailures.length === 0 ? 'PASS' : 'FAIL'
);
for (const side of EQUIVARIANCE_SIDES) {
  const e = equivariance[side];
  const complete = e.checked === EQUIVARIANCE_REQUIRED_POSITIONS;
  console.log(
    `equivariance ${side.padEnd(9)}`,
    e.mismatches.length > 0
      ? 'FAIL'
      : complete
        ? `PASS (${e.checked}/${EQUIVARIANCE_REQUIRED_POSITIONS})`
        : `INCOMPLETE (${e.checked}/${EQUIVARIANCE_REQUIRED_POSITIONS})`
  );
}
console.log('');
for (const { id } of surfaces) {
  const r = surfaceResults[id];
  console.log(
    `${id}  maxlogit ${f(r.max_abs_logit_diff)} / ${f(r.tolerances.maxLogit)}` +
      `   meanlogit ${f(r.mean_abs_logit_diff)} / ${f(r.tolerances.meanLogit)}` +
      `   maxvalue ${f(r.max_abs_value_diff)} / ${f(r.tolerances.maxValue)}`
  );
  console.log(
    `     top1 disagreements ${r.top1_disagreements_outside_band.length}` +
      `   near-tie exemptions ${r.near_tie_exemptions}/${NEAR_TIE_EXEMPTION_CAP}` +
      `   top-k ${(r.topk_set_agreement * 100).toFixed(1)}%` +
      `   tau median ${r.kendall_tau_median.toFixed(6)}   [${r.aggregate_gates_scope}]`
  );
  const edgeAgree = r.edge_ordering.filter((e) => e.top1_agree).length;
  const edgeTopk = r.edge_ordering.filter((e) => e.topk_set_agree).length;
  console.log(
    `     edge (reported separately): top1 ${edgeAgree}/${r.edge_ordering.length}` +
      `   top-k ${edgeTopk}/${r.edge_ordering.length}` +
      `   tau min ${Math.min(...r.edge_ordering.map((e) => e.kendall_tau)).toFixed(6)}` +
      `   failures ${r.edge_top1_failures.length}`
  );
}
console.log(
  `\nred-perspective max diff: ${f(perspectiveMax)} / ${f(PERSPECTIVE_TOL)}`
);
console.log(`value range violations:   ${rangeViolations.length}`);
if (failures.length) {
  console.log(`\n${failures.length} FAILURE(S):`);
  for (const x of failures) console.log(`  ${x.surface}  ${x.metric}`);
}
console.log(`\nwrote ${outPath}`);
process.exit(verdict === 'PASS' ? 0 : 1);
