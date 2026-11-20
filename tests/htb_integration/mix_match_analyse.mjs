/**
 * H-MX match: the frozen analyser. Card: docs/superpowers/2026-11-18-htb-mix-distillation-card.md (revision 1), §9, §10,
 * §11, §13. Its sha256 is pinned in both match wrappers; they refuse another.
 *
 *   node tests/htb_integration/mix_match_analyse.mjs timing    -> the smoke's timing.json (stdout); reads no outcome
 *   node tests/htb_integration/mix_match_analyse.mjs analyse   -> the match analysis (stdout)
 *
 * Admission (both), as H-OP's frozen analyser (onpolicy_match_analyse.mjs) with H-MX's constants: the stage's
 * outer.json says COMPLETED with exit 0, the gate restored, no survivors, the record written by the script and the
 * stage's ceiling (TERM = KILL - 10 s); outer and inner records name the same launch commit; pgid.txt is the driver's
 * pid; record.json is COMPLETED with the stage's kill_s, onnxruntime-node 1.23.2, the current hard 78c16a31c3dba37f, the
 * challenger the committed H-MX export record names, the 200-simulation protocol policy and the pinned openings file;
 * the export record is EXPORT ATTEMPT 2's (docs/superpowers/2026-11-20-htb-mix-export-attempt-2-amendment.md §4:
 * htb-mix-export-2/, `attempt: 2`, that amendment's sha256; attempt 1, VOID, is never read) and its wrapper's evidence
 * is a clean completion bound to it; every scheduled game file exists, and
 * nothing else; every game replays legally from the empty board under the product's game-end rule; its opening,
 * colours, result and score equal what is re-derived here; it has exactly one per-move row per played ply from ply 4,
 * in order, each carrying the re-derived seeds, a 200-simulation budget and 200 root visits spent, on BOTH sides.
 * Anything else: INVALID (exit 3), no number.
 *
 * The smoke's four games are admitted the same way (so they are replay-validated), but their outcomes are excluded from
 * scoring and from every report: `timing` reads wall times only.
 *
 * Before any outcome is read, the guard: the challenger's total search seconds <= 1.10 x the current hard's; otherwise
 * INVALID and nothing is scored.
 *
 * The verdict (card §9): over the 400 pairs, W pairs with score > 0.5, L pairs with score < 0.5, m = W + L decisive
 * pairs. The exact one-sided sign test rejects iff m >= 1 and P(Binomial(m, 1/2) >= W) <= 0.05, computed EXACTLY in
 * integers. PASS iff the test rejects AND the mean pair score exceeds 0.50 (an additional guard, not part of the 5%
 * bound: it can only remove a PASS). Anything else that completes and passes the guards is NON_PASS.
 *
 * Openings and seeds are re-derived HERE (the opening rule per listed index, through H-OP's deriveOpening with this
 * namespace; the seed formula), independently of mix_match.mjs; a test checks the whole lists against the Python build.
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { TwixtState } from '../../server/gameLogic.js';
import { WRAPPER_FILES, deriveOpening as hopDeriveOpening, exportWrapperFailures } from './onpolicy_match_analyse.mjs';

export { WRAPPER_FILES, exportWrapperFailures };

export const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const NS = 'htb-mix/20261118';
const OPENINGS_REL = 'docs/superpowers/evidence/htb-mix-openings/openings.json';
const OPENINGS_SHA256 = 'f4ae110ff9e7547e413b15fff29ca0ff94d34c5f21d5d8b7da76af6577ee3563';
/** Export attempt 2's record (export amendment 1, §4); attempt 1's (htb-mix-export/, VOID, entry 75) is never read. */
export const EXPORT_REL = 'docs/superpowers/evidence/htb-mix-export-2/record.json';
export const EXPORT_KIND = 'htb-mix-export-record/1';
export const EXPORT_ATTEMPT = 2;
export const EXPORT_AMENDMENT_SHA256 = 'f8ce7792afc73fc3262ff0fb6aff58925d4b060d6b9a97b38fff94ea7c2ac032';
/** The export stage's wrapper evidence (card §8, §11; attempt 2); its ceiling, 3,600 s, is H-OP's, checked by exportWrapperFailures. */
export const EXPORT_OUTER_REL = 'docs/superpowers/evidence/htb-mix-export-2-run';
export const PAIRS = { smoke: 2, match: 400 };
export const MATCH_CAP_S = 64800;
export const KILL_S = { smoke: 1800, match: 64800 };
export const GRACE_S = 10;
export const SIMS = 200;
export const PINS = { current_model_id: '78c16a31c3dba37f', onnxruntime_node: '1.23.2' };
export const LAUNCH_FACTOR = 1.5;
export const SEARCH_SECONDS_GUARD = 1.1;
export const ALPHA_ONE_SIDED = 0.05;
export const DIRS = {
  smoke: { run: 'docs/superpowers/evidence/htb-mix-match-smoke', outer: 'docs/superpowers/evidence/htb-mix-match-smoke-run' },
  match: { run: 'docs/superpowers/evidence/htb-mix-match', outer: 'docs/superpowers/evidence/htb-mix-match-run' },
};
const SEED_STAGE = { smoke: 'match smoke', match: 'match' };
const OPENING_LIST = { smoke: 'match_smoke', match: 'match' };
const OPENING_KIND = { smoke: 'match smoke opening', match: 'match opening' };
const GAME_SCHEMA = 'htb-mix-match-game/1';
const RECORD_SCHEMA = 'htb-mix-match-record/1';

export class Invalid extends Error {}

const sha = (s) => createHash('sha256').update(s).digest('hex');

// ------------------------------------------------------------------------------------- independent re-derivations

/** Opening `index` of `kind` under the benchmark's rule, in the H-MX namespace (H-OP's derivation, this namespace). */
export const deriveOpening = (kind, index) => hopDeriveOpening(kind, index, NS);

export function deriveSeeds(stage, pair, game, ply) {
  const h = sha(`${NS}/${SEED_STAGE[stage]}/pair ${pair}/game ${game}/ply ${ply}`);
  return { search: Number(BigInt(`0x${h.slice(0, 12)}`)), readout: Number(BigInt(`0x${h.slice(12, 24)}`)) };
}

/** The committed openings for a stage: the pinned file, each listed opening the rule's at its index, distinct
 * directly and reflected, indices increasing. */
export function stageOpenings(stage, root = REPO_ROOT) {
  const p = join(root, OPENINGS_REL);
  if (!existsSync(p)) throw new Invalid(`${OPENINGS_REL} missing`);
  const raw = readFileSync(p);
  if (sha(raw) !== OPENINGS_SHA256) throw new Invalid(`${OPENINGS_REL} is not the pinned file`);
  const list = JSON.parse(raw.toString('utf8'))[OPENING_LIST[stage]];
  if (!Array.isArray(list) || list.length !== PAIRS[stage]) throw new Invalid(`the ${OPENING_LIST[stage]} list does not hold ${PAIRS[stage]} openings`);
  const seen = new Set();
  let last = -1;
  for (const o of list) {
    if (!(Number.isInteger(o.index) && o.index > last)) throw new Invalid(`opening indices are not increasing at ${o.index}`);
    last = o.index;
    if (JSON.stringify(o.moves) !== JSON.stringify(deriveOpening(OPENING_KIND[stage], o.index))) throw new Invalid(`opening ${o.index} is not the rule's`);
    const a = JSON.stringify(o.moves);
    const b = JSON.stringify(o.moves.map(([r, c]) => [r, 23 - c]));
    if (seen.has(a) || seen.has(b)) throw new Invalid(`opening ${o.index} repeats an earlier one`);
    seen.add(a);
  }
  return list;
}

// ------------------------------------------------------------------------------------------ the exact sign test

/** C(m, k) exactly, as a BigInt. */
function choose(m, k) {
  if (k < 0 || k > m) return 0n;
  let r = 1n;
  const kk = Math.min(k, m - k);
  for (let i = 1; i <= kk; i++) r = (r * BigInt(m - kk + i)) / BigInt(i);
  return r;
}

/** P(Binomial(m, 1/2) >= w), exactly: { num, den } with den = 2^m, and its value as a Number. */
export function binomialTail(m, w) {
  let num = 0n;
  for (let k = Math.max(w, 0); k <= m; k++) num += choose(m, k);
  const den = 1n << BigInt(m);
  return { num, den, p: Number(num) / Number(den) };
}

/** The test's rejection, decided in integers: P(Bin(m, 1/2) >= w) <= 1/20 iff 20 * num <= 2^m. */
export function rejects(m, w) {
  if (!(m >= 1)) return false;
  const { num, den } = binomialTail(m, w);
  return 20n * num <= den;
}

/** The smallest W that rejects for this m, or null when none can (m <= 4). */
export function criticalW(m) {
  for (let w = 0; w <= m; w++) if (rejects(m, w)) return w;
  return null;
}

/** The exact one-sided sign test over the decisive pairs (card §9). */
export function signTest(w, l) {
  const m = w + l;
  const t = m >= 1 ? binomialTail(m, w) : { p: 1 };
  return { W: w, L: l, m, p_one_sided: t.p, critical_W: criticalW(m), alpha: ALPHA_ONE_SIDED, rejects: rejects(m, w) };
}

// log-space binomial, for the descriptive Clopper-Pearson interval only (never the verdict)
const LOGF = [0];
const logFact = (n) => {
  while (LOGF.length <= n) LOGF.push(LOGF[LOGF.length - 1] + Math.log(LOGF.length));
  return LOGF[n];
};
function binomCdf(k, m, p) { // P(X <= k)
  if (k < 0) return 0;
  if (k >= m) return 1;
  if (p <= 0) return 1;
  if (p >= 1) return 0;
  let s = 0;
  for (let i = 0; i <= k; i++) s += Math.exp(logFact(m) - logFact(i) - logFact(m - i) + i * Math.log(p) + (m - i) * Math.log1p(-p));
  return Math.min(1, s);
}
function bisect(f, lo, hi) { // f increasing in p; returns p with f(p) ~ 0
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (f(mid) < 0) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/** The two-sided 95% Clopper-Pearson interval for w / m (descriptive). */
export function clopperPearson(w, m, level = 0.95) {
  if (m < 1) return [0, 1];
  const a = (1 - level) / 2;
  const lower = w === 0 ? 0 : bisect((p) => (1 - binomCdf(w - 1, m, p)) - a, 0, 1); // P(X >= w; p) = a
  const upper = w === m ? 1 : bisect((p) => a - binomCdf(w, m, p), 0, 1); // P(X <= w; p) = a
  return [lower, upper];
}

/** Two-sided 95% Hoeffding over n pairs, descriptive only: sqrt(ln 40 / (2n)). */
export const twoSidedHalfWidth = (nPairs) => Math.sqrt(Math.log(40) / (2 * nPairs));

// ---------------------------------------------------------------------------------------------------- admission

function readJson(p, what) {
  if (!existsSync(p)) throw new Invalid(`${what} missing: ${p}`);
  try {
    return JSON.parse(readFileSync(p, 'utf8'));
  } catch {
    throw new Invalid(`${what} unparseable: ${p}`);
  }
}

/** Validates one game against the schedule; returns it. Its outcome is read only by the match analysis. */
export function validateGame(g, { stage, pair, game, opening }) {
  const where = `pair ${pair} game ${game}`;
  if (g.schema !== GAME_SCHEMA || g.stage !== stage || g.pair !== pair || g.game !== game) throw new Invalid(`${where}: identity fields`);
  const colour = game === 0 ? 'red' : 'black';
  if (g.challenger_colour !== colour) throw new Invalid(`${where}: challenger colour`);
  if (g.opening_index !== opening.index || JSON.stringify(g.opening) !== JSON.stringify(opening.moves)) throw new Invalid(`${where}: opening`);
  if (!Array.isArray(g.moves) || JSON.stringify(g.moves.slice(0, 4)) !== JSON.stringify(opening.moves)) throw new Invalid(`${where}: moves do not start with the opening`);
  let s = new TwixtState();
  for (let k = 0; k < g.moves.length; k++) {
    if (s.isTerminal()) throw new Invalid(`${where}: play continued after the game ended at ply ${k}`);
    const mv = g.moves[k];
    if (!s.legalMoves().some((m) => m[0] === mv[0] && m[1] === mv[1])) throw new Invalid(`${where}: illegal move at ply ${k}`);
    s = s.applyMove(mv);
  }
  if (!s.isTerminal()) throw new Invalid(`${where}: the game did not end`);
  const winner = s.winner();
  if (g.result !== (winner ?? 'draw') || g.plies !== g.moves.length) throw new Invalid(`${where}: result or length`);
  if (g.termination !== (winner === null ? 'no_legal_moves' : 'connection')) throw new Invalid(`${where}: termination`);
  const score = winner === null ? 0.5 : winner === colour ? 1 : 0;
  if (g.challenger_score !== score) throw new Invalid(`${where}: score`);
  // Exactly one row per played ply from ply 4, in order: no duplicate, no gap, no extra.
  if (!Array.isArray(g.per_move) || g.per_move.length !== g.moves.length - 4) throw new Invalid(`${where}: ${g.per_move?.length} per-move rows for ${g.moves.length - 4} played plies`);
  g.per_move.forEach((pm, i) => {
    if (pm.ply !== 4 + i) throw new Invalid(`${where}: per-move row ${i} is ply ${pm.ply}, expected ${4 + i}`);
    const mover = (pm.ply % 2 === 0 ? 'red' : 'black') === colour ? 'challenger' : 'current';
    if (pm.mover !== mover || JSON.stringify(pm.move) !== JSON.stringify(g.moves[pm.ply])) throw new Invalid(`${where}: per-move ply ${pm.ply}`);
    if (JSON.stringify(pm.seeds) !== JSON.stringify(deriveSeeds(stage, pair, game, pm.ply))) throw new Invalid(`${where}: seeds at ply ${pm.ply}`);
    if (pm.nSimulations !== SIMS || pm.visits !== SIMS) throw new Invalid(`${where}: ${mover} budget ${pm.nSimulations}, visits ${pm.visits} at ply ${pm.ply}; the card's is ${SIMS}`);
    if (!(Number.isFinite(pm.ms) && pm.ms >= 0)) throw new Invalid(`${where}: move time at ply ${pm.ply}`);
  });
  if (!(g.wall_ms > 0)) throw new Invalid(`${where}: wall time`);
  return g;
}

/** H-OP's outer/inner binding, for this stage's ceiling, this match's players and the H-MX export record. */
export function bindingFailures({ outer: o, record: rec, pgid, stage, exportRecord }) {
  const f = [];
  const want = { inner_exit: 0, outer_exit: 0, outcome: 'COMPLETED', gate_restored: true, survivors: '', record_by: 'script', term_s: KILL_S[stage] - GRACE_S, grace_s: GRACE_S };
  for (const [k, v] of Object.entries(want)) if (o?.[k] !== v) f.push(`outer ${k} ${JSON.stringify(o?.[k])}, expected ${JSON.stringify(v)}`);
  if (rec?.schema !== RECORD_SCHEMA || rec?.status !== 'COMPLETED' || rec?.written_by !== 'script' || rec?.stage !== stage) f.push(`record.json is not a COMPLETED ${stage} record`);
  if (rec?.kill_s !== KILL_S[stage]) f.push(`record kill_s ${rec?.kill_s}, the card's ${stage} ceiling is ${KILL_S[stage]}`);
  if (!/^[0-9a-f]{40}$/.test(rec?.head ?? '') || o?.head !== rec.head) f.push('the outer and inner records do not name the same launch commit');
  if (!Number.isInteger(rec?.pid) || pgid !== `${rec.pid}\n`) f.push('pgid.txt is not the driver pid');
  if (rec?.runtime?.onnxruntime_node !== PINS.onnxruntime_node) f.push(`onnxruntime-node ${rec?.runtime?.onnxruntime_node}`);
  if (rec?.current?.model_id !== PINS.current_model_id) f.push(`current model ${rec?.current?.model_id}`);
  if (rec?.policy?.nSims !== SIMS || rec?.policy?.search !== 'protocol' || rec?.policy?.moveTemp !== 1) f.push('the policy is not the 200-simulation protocol');
  if (rec?.openings_sha256 !== OPENINGS_SHA256) f.push('the record does not name the pinned openings');
  const c = exportRecord?.challenger;
  if (exportRecord?.kind !== EXPORT_KIND || exportRecord?.status !== 'COMPLETED' || exportRecord?.verdict !== 'PARITY_PASS') f.push('the export record is not a COMPLETED PARITY_PASS H-MX export record');
  if (exportRecord?.attempt !== EXPORT_ATTEMPT || exportRecord?.export_amendment_sha256 !== EXPORT_AMENDMENT_SHA256) f.push('the export record is not export attempt 2 under its amendment');
  for (const k of ['model_id', 'dir', 'graph_sha256', 'external_data_sha256', 'source_sha256']) {
    if (!c?.[k] || rec?.challenger?.[k] !== c[k]) f.push(`challenger ${k} is not the export record's`);
  }
  if (rec?.challenger?.onnx_sha256 !== c?.graph_sha256) f.push('the loaded challenger graph is not the export record\'s');
  if (rec?.challenger?.model_id === PINS.current_model_id) f.push('the challenger is the current hard');
  return f;
}

export function admit(stage, root = REPO_ROOT) {
  const { run, outer } = DIRS[stage];
  const o = readJson(join(root, outer, 'outer.json'), 'outer.json');
  const rec = readJson(join(root, run, 'record.json'), 'record.json');
  const exportRecord = readJson(join(root, EXPORT_REL), 'export record');
  const pgidPath = join(root, outer, 'pgid.txt');
  const pgid = existsSync(pgidPath) ? readFileSync(pgidPath, 'utf8') : null;
  const bf = bindingFailures({ outer: o, record: rec, pgid, stage, exportRecord });
  const ePgid = join(root, EXPORT_OUTER_REL, 'pgid.txt');
  bf.push(...exportWrapperFailures({ outer: readJson(join(root, EXPORT_OUTER_REL, 'outer.json'), 'export outer.json'),
    pgid: existsSync(ePgid) ? readFileSync(ePgid, 'utf8') : null, exportRecord }));
  if (bf.length) throw new Invalid(`the ${stage} evidence is not bound to its run: ${bf.join('; ')}`);
  const openings = stageOpenings(stage, root);
  const expected = [];
  for (let p = 0; p < PAIRS[stage]; p++) for (const g of [0, 1]) expected.push({ pair: p, game: g });
  const names = expected.map((t) => `pair_${String(t.pair).padStart(4, '0')}_game_${t.game}.json`);
  const gdir = join(root, run, 'games');
  if (!existsSync(gdir)) throw new Invalid('the games directory is missing');
  const present = readdirSync(gdir).sort();
  if (JSON.stringify(present) !== JSON.stringify([...names].sort())) throw new Invalid(`the games directory does not hold exactly the ${names.length} scheduled games`);
  if (rec.games !== names.length) throw new Invalid('record.json game count');
  const games = expected.map((t, i) => validateGame(readJson(join(gdir, names[i]), names[i]), { stage, ...t, opening: openings[t.pair] }));
  return { outer: o, record: rec, games };
}

// ---------------------------------------------------------------------------------------------------------- timing

/** The launch estimate from the smoke: wall times only, never an outcome. */
export function timing(adm) {
  const startup = adm.record.startup_seconds;
  if (!(startup > 0)) throw new Invalid('record.json startup_seconds');
  const perGame = adm.games.reduce((a, g) => a + g.wall_ms / 1000, 0) / adm.games.length;
  const projection = startup + perGame * 2 * PAIRS.match;
  const estimate = LAUNCH_FACTOR * projection;
  const r = (x) => Math.round(x * 1000) / 1000;
  return {
    schema: 'htb-mix-match-timing/1',
    startup_s: r(startup),
    game_s_mean: r(perGame),
    smoke_plies: adm.games.map((g) => g.plies),
    match_games: 2 * PAIRS.match,
    projection_s: r(projection),
    launch_factor: LAUNCH_FACTOR,
    launch_estimate_s: r(estimate),
    match_cap_s: MATCH_CAP_S,
    verdict: estimate <= MATCH_CAP_S ? 'LAUNCH_OK' : 'NO_LAUNCH',
    note: 'an estimate from 4 smoke games; long draws can exceed it; the 18 h match cap is fixed',
    smoke_outcomes: 'replay-validated on admission; excluded from scoring and from every report',
  };
}

// ---------------------------------------------------------------------------------------------------------- analysis

function summarise(xs) {
  const s = [...xs].sort((a, b) => a - b);
  return { n: s.length, mean: s.reduce((a, b) => a + b, 0) / s.length, median: s[s.length >> 1], min: s[0], max: s[s.length - 1] };
}

/** The search-seconds guard, from per-move times only (no outcome is read). */
export function searchSecondsGuard(games) {
  const t = { challenger: 0, current: 0 };
  for (const g of games) for (const pm of g.per_move) t[pm.mover] += pm.ms / 1000;
  const ratio = t.current > 0 ? t.challenger / t.current : Infinity;
  return { challenger_s: t.challenger, current_s: t.current, ratio, limit: SEARCH_SECONDS_GUARD, within: ratio <= SEARCH_SECONDS_GUARD };
}

export const READING = {
  PASS: 'PASS: the exact one-sided sign test rejects at alpha 0.05 and the mean pair score exceeds 0.50. Over the planned opening '
    + 'sample (fresh 4-ply openings by the frozen rule, each played twice with colours reversed, under this 200-simulation '
    + 'protocol) the challenger wins more decisive pairs than it loses against the current hard. This holds for the whole '
    + 'update (the data mix, the dose and the continued training together) and estimates no size of advantage. The 5% bound '
    + 'assumes (a) independent pairs, (b) exchangeable, fair win/loss signs under the null, (c) a fixed design with no '
    + 'sequential look, top-up or discarded pair, and (d) its scope is the planned opening distribution only. The mean '
    + 'condition is an additional guard that can only remove a PASS. It authorizes proposing an integration decision, nothing more.',
  NON_PASS: 'NON_PASS: superiority in decisive pairs is not shown. This is not evidence of no gain: the planned power is about '
    + '80% at a 0.691 decisive-pair win probability and a 0.114 decisive-pair rate, from specified inputs, conditional on '
    + 'completion. The current hard stays; no top-up or re-score.',
};

export function analyse(adm, nPairs = PAIRS.match) {
  const guard = searchSecondsGuard(adm.games);
  if (!guard.within) {
    return {
      schema: 'htb-mix-match-analysis/1',
      verdict: 'INVALID',
      reason: `the challenger's search seconds are ${guard.ratio} x the current hard's, above ${SEARCH_SECONDS_GUARD}; nothing is scored`,
      search_seconds_guard: guard,
    };
  }
  const pairs = [];
  for (let p = 0; p < nPairs; p++) {
    const two = adm.games.filter((g) => g.pair === p);
    if (two.length !== 2 || two[0].game === two[1].game) throw new Invalid(`pair ${p} has ${two.length} games`);
    pairs.push((two[0].challenger_score + two[1].challenger_score) / 2);
  }
  const w = pairs.filter((x) => x > 0.5).length;
  const l = pairs.filter((x) => x < 0.5).length;
  const test = signTest(w, l);
  const sum = pairs.reduce((a, b) => a + b, 0);
  const mean = sum / nPairs;
  const meanGuard = mean > 0.5;
  const h2 = twoSidedHalfWidth(nPairs);
  const wdl = (gs) => ({ W: gs.filter((g) => g.challenger_score === 1).length, D: gs.filter((g) => g.challenger_score === 0.5).length, L: gs.filter((g) => g.challenger_score === 0).length });
  const colour = (c) => {
    const gs = adm.games.filter((g) => g.challenger_colour === c);
    return { ...wdl(gs), score: gs.reduce((a, g) => a + g.challenger_score, 0) / gs.length };
  };
  const verdict = test.rejects && meanGuard ? 'PASS' : 'NON_PASS';
  return {
    schema: 'htb-mix-match-analysis/1',
    rule: 'PASS iff the exact one-sided sign test over the m decisive pairs rejects (m >= 1 and P(Binomial(m, 1/2) >= W) <= 0.05) '
      + 'and the mean pair score exceeds 0.50 (an additional guard), over the n = 400 pairs',
    pairs: nPairs,
    games: adm.games.length,
    challenger: wdl(adm.games),
    sign_test: test,
    decisive_pairs: { W: w, L: l, m: w + l, split: nPairs - w - l, rate: (w + l) / nPairs,
      win_share: w + l ? w / (w + l) : null, clopper_pearson_95: clopperPearson(w, w + l) },
    mean_pair_score: mean,
    pair_score_sum: sum,
    mean_guard: { value: mean, above_half: meanGuard },
    verdict,
    search_seconds_guard: guard,
    descriptive: {
      two_sided_95_hoeffding_half_width: h2,
      two_sided_95_hoeffding_interval: [mean - h2, mean + h2],
      by_colour: { challenger_red: colour('red'), challenger_black: colour('black') },
      draws: adm.games.filter((g) => g.result === 'draw').length,
      plies: summarise(adm.games.map((g) => g.plies)),
      distinct_games: new Set(adm.games.map((g) => JSON.stringify(g.moves))).size,
    },
    reading: READING[verdict],
  };
}

function cli() {
  const [cmd] = process.argv.slice(2);
  try {
    if (cmd === 'timing') {
      process.stdout.write(`${JSON.stringify(timing(admit('smoke')), null, 1)}\n`);
      return 0;
    }
    if (cmd === 'analyse') {
      const a = analyse(admit('match'));
      process.stdout.write(`${JSON.stringify(a, null, 1)}\n`);
      return a.verdict === 'INVALID' ? 3 : 0;
    }
    console.error('usage: mix_match_analyse.mjs <timing|analyse>');
    return 2;
  } catch (e) {
    if (e instanceof Invalid) {
      console.error(`INVALID: ${e.message}`);
      return 3;
    }
    throw e;
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) process.exitCode = cli();
