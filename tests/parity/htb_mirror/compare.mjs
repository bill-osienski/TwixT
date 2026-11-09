#!/usr/bin/env node
/**
 * V3 verdict: averaged-output parity between the research MirrorAveragingEvaluator(student) and the product's
 * MirrorAveragingInference(pinned hard model).
 *
 *   node tests/parity/htb_mirror/compare.mjs <python_side.json> <node_side.json> <out.json>
 *
 * The rule was declared in the integration proposal (§7, V3): "the S4 bounds and V2's ordering rules applied to
 * the averaged priors and value". Transcribed here from August's specification and tests/parity/compare.mjs,
 * before any V3 output exists:
 *   - legal-move lists identical (exact), every position;
 *   - max abs prior difference <= 1.1e-4, mean abs prior difference <= 1.1e-5, max abs value difference <= 1.1e-4,
 *     every position;
 *   - ordering by prior, reference = the research side: top-1 must agree except where the reference top-1/top-2
 *     gap <= 2 x 1.1e-4, at most 6 such exemptions over the primary 120; an edge-case top-1 disagreement outside
 *     the band is a failure; top-k sets (k = min(5, n_legal)) agree on >= 95% of the primary 120; Kendall tau-a
 *     median over the primary 120 >= 0.99;
 *   - values in [-1, 1]; value signs agree except where |value| <= 1e-3.
 * PASS only if every rule holds. Exit 0 PASS, 1 FAIL.
 */
import { readFile, writeFile } from 'node:fs/promises';

const TOL = { maxPrior: 1.1e-4, meanPrior: 1.1e-5, maxValue: 1.1e-4 };
const BAND = 2 * TOL.maxPrior;
const CAP = 6;
const TOPK_MIN = 0.95;
const TAU_MIN = 0.99;
const DEADBAND = 1e-3;

const argmax = (xs) => xs.reduce((b, v, i) => (v > xs[b] ? i : b), 0);
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
const topKSet = (xs, k) => new Set(xs.map((v, i) => [v, i]).sort((p, q) => q[0] - p[0]).slice(0, k).map(([, i]) => i));
const setsEqual = (a, b) => a.size === b.size && [...a].every((x) => b.has(x));
function kendallTau(a, b) {
  const n = a.length;
  if (n < 2) return 1;
  let c = 0;
  let d = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const s = Math.sign(a[i] - a[j]) * Math.sign(b[i] - b[j]);
      if (s > 0) c++;
      else if (s < 0) d++;
    }
  }
  return (c - d) / ((n * (n - 1)) / 2);
}
const median = (xs) => {
  const s = [...xs].sort((p, q) => p - q);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

const [pyPath, ndPath, outPath] = process.argv.slice(2);
const py = JSON.parse(await readFile(pyPath, 'utf8'));
const nd = JSON.parse(await readFile(ndPath, 'utf8'));
const byId = new Map(nd.positions.map((p) => [p.id, p]));
const failures = [];
let maxPrior = 0;
let maxPriorAt = null;
let meanAcc = 0;
let maxValue = 0;
let maxValueAt = null;
const exempt = [];
const top1Out = [];
const edge = [];
const topk = [];
const taus = [];
const signs = [];
const range = [];
if (py.positions.length !== 126 || nd.positions.length !== 126) failures.push('position count');
for (const p of py.positions) {
  const n = byId.get(p.id);
  if (!n) {
    failures.push(`missing ${p.id}`);
    continue;
  }
  if (JSON.stringify(p.legal_moves) !== JSON.stringify(n.legal_moves)) failures.push(`legal moves ${p.id}`);
  const A = p.priors;
  const B = n.priors;
  let ml = 0;
  let sum = 0;
  for (let i = 0; i < A.length; i++) {
    const d = Math.abs(A[i] - B[i]);
    ml = Math.max(ml, d);
    sum += d;
  }
  if (ml > maxPrior) [maxPrior, maxPriorAt] = [ml, p.id];
  meanAcc += sum / A.length;
  const mv = Math.abs(p.value - n.value);
  if (mv > maxValue) [maxValue, maxValueAt] = [mv, p.id];
  for (const v of [p.value, n.value]) if (!(v >= -1 && v <= 1)) range.push({ id: p.id, v });
  if (Math.sign(p.value) !== Math.sign(n.value) && Math.abs(p.value) > DEADBAND && Math.abs(n.value) > DEADBAND) {
    signs.push({ id: p.id, a: p.value, b: n.value });
  }
  const ta = argmax(A);
  const tb = argmax(B);
  const gap = topGap(A);
  const k = Math.min(5, A.length);
  const kOk = setsEqual(topKSet(A, k), topKSet(B, k));
  if (p.stratum !== 'edge') {
    if (ta !== tb) (gap <= BAND ? exempt : top1Out).push({ id: p.id, gap });
    topk.push(kOk ? 1 : 0);
    taus.push(kendallTau(A, B));
  } else {
    edge.push({ id: p.id, top1_agree: ta === tb, gap: gap === Infinity ? null : gap, topk_set_agree: kOk, tau: kendallTau(A, B) });
    if (ta !== tb && gap > BAND) failures.push(`edge top-1 outside band ${p.id}`);
  }
}
const meanPrior = meanAcc / py.positions.length;
const topkRate = topk.reduce((a, b) => a + b, 0) / topk.length;
const tauMedian = median(taus);
if (maxPrior > TOL.maxPrior) failures.push('max_abs_prior_diff');
if (meanPrior > TOL.meanPrior) failures.push('mean_abs_prior_diff');
if (maxValue > TOL.maxValue) failures.push('max_abs_value_diff');
if (top1Out.length) failures.push('top1_disagreement_outside_near_tie_band');
if (exempt.length > CAP) failures.push('near_tie_exemptions_over_cap');
if (topkRate < TOPK_MIN) failures.push('topk_set_agreement');
if (tauMedian < TAU_MIN) failures.push('kendall_tau_median');
if (signs.length) failures.push('value_sign_disagreement');
if (range.length) failures.push('value_range');
const verdict = failures.length ? 'FAIL' : 'PASS';
const report = {
  schema: 'htb-mirror-parity/1', verdict, failures, tolerances: { ...TOL, band: BAND, cap: CAP, topk_min: TOPK_MIN, tau_min: TAU_MIN },
  max_abs_prior_diff: maxPrior, max_abs_prior_at: maxPriorAt, mean_abs_prior_diff: meanPrior, max_abs_value_diff: maxValue,
  max_abs_value_at: maxValueAt, top1_outside_band: top1Out, near_tie_exemptions: exempt, topk_set_agreement: topkRate,
  kendall_tau_median: tauMedian, kendall_tau_min: Math.min(...taus), edge, sign_disagreements: signs, range_violations: range,
  python_side: { side: py.side, environment: py.environment, state_digest: py.state_digest }, node_side: { side: nd.side, model_id: nd.model_id, node: nd.node },
};
await writeFile(outPath, JSON.stringify(report, null, 1));
const e = (x) => (x === 0 ? '0' : x.toExponential(3));
console.log(`V3 MIRROR-AVERAGED PARITY ${verdict}`);
console.log(`  max abs prior ${e(maxPrior)} / ${e(TOL.maxPrior)}   mean ${e(meanPrior)} / ${e(TOL.meanPrior)}   max abs value ${e(maxValue)} / ${e(TOL.maxValue)}`);
console.log(`  top-1 outside band ${top1Out.length}   exemptions ${exempt.length}/${CAP}   top-k ${(100 * topkRate).toFixed(1)}%   tau median ${tauMedian.toFixed(6)}   edge top-1 ${edge.filter((x) => x.top1_agree).length}/${edge.length}`);
if (failures.length) console.log('  FAILURES:', failures.join(', '));
process.exit(verdict === 'PASS' ? 0 : 1);
