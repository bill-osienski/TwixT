/**
 * H-OP match: the frozen analyser. Card: docs/superpowers/2026-11-14-htb-onpolicy-distillation-card.md (revision 3),
 * §10, §11, §12, §14. Its sha256 is pinned in both match wrappers; they refuse another.
 *
 *   node tests/htb_integration/onpolicy_match_analyse.mjs timing    -> the smoke's timing.json (stdout); reads no outcome
 *   node tests/htb_integration/onpolicy_match_analyse.mjs analyse   -> the match analysis (stdout)
 *
 * Admission (both): the stage's outer.json says COMPLETED with exit 0, the gate restored, no survivors, the record
 * written by the script and the stage's ceiling (TERM = KILL - 10 s); outer and inner records name the same launch
 * commit; pgid.txt is the driver's pid; record.json is COMPLETED with the stage's kill_s, onnxruntime-node 1.23.2, the
 * current hard 78c16a31c3dba37f, the challenger the committed export record names, the 200-simulation protocol policy
 * and the pinned openings file; every scheduled game file exists, and nothing else; every game replays legally from the
 * empty board under the product's game-end rule; its opening, colours, result and score equal what is re-derived here;
 * it has exactly one per-move row per played ply from ply 4, in order, each carrying the re-derived seeds, a 200-simulation
 * budget and 200 root visits spent, on BOTH sides. Anything else: INVALID (exit 3), no number.
 *
 * The smoke's four games are admitted the same way (so they are replay-validated), but their outcomes are excluded from
 * scoring and from every report: `timing` reads wall times only.
 *
 * The verdict (card §10): PASS iff mean pair score - sqrt(ln 20 / 544) > 0.5 over the 272 pairs (one-sided Hoeffding,
 * alpha 0.05): 625/1088 passes and 624/1088 does not. Before any outcome is read, the guard: the challenger's total
 * search seconds <= 1.10 x the current hard's; otherwise INVALID and nothing is scored. A PASS establishes superiority
 * over 0.50 under this protocol for the whole update (the new data source and the continued training together); it does
 * not establish a gain of 0.60, which is only the planning target.
 *
 * Openings and seeds are re-derived HERE (the opening rule per listed index; the seed formula), independently of
 * onpolicy_match.mjs; a test checks the whole lists against the Python build.
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { TwixtState } from '../../server/gameLogic.js';

export const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const NS = 'htb-onpolicy/20261114';
const OPENINGS_REL = 'docs/superpowers/evidence/htb-onpolicy-openings/openings.json';
const OPENINGS_SHA256 = 'ca4e3bba05bfc6abddf782b1a1f68fc474bc3600410c7cbd4308073d306d3915';
const EXPORT_REL = 'docs/superpowers/evidence/htb-onpolicy-export/record.json';
/** The export stage's wrapper evidence and ceiling (card §9, §12; the handoff repair of 2026-11-14). */
export const EXPORT_OUTER_REL = 'docs/superpowers/evidence/htb-onpolicy-export-run';
export const EXPORT_KILL_S = 3600;
export const WRAPPER_FILES = ['outer.json', 'pgid.txt', 'idle_check.txt', 'bg_check.txt'];
export const PAIRS = { smoke: 2, match: 272 };
export const MATCH_CAP_S = 64800;
export const KILL_S = { smoke: 1800, match: 64800 };
export const GRACE_S = 10;
export const SIMS = 200;
export const PINS = { current_model_id: '78c16a31c3dba37f', onnxruntime_node: '1.23.2' };
export const LAUNCH_FACTOR = 1.5;
export const SEARCH_SECONDS_GUARD = 1.1;
export const ALPHA_ONE_SIDED = 0.05;
export const DIRS = {
  smoke: { run: 'docs/superpowers/evidence/htb-onpolicy-match-smoke', outer: 'docs/superpowers/evidence/htb-onpolicy-match-smoke-run' },
  match: { run: 'docs/superpowers/evidence/htb-onpolicy-match', outer: 'docs/superpowers/evidence/htb-onpolicy-match-run' },
};
const SEED_STAGE = { smoke: 'match smoke', match: 'match' };
const OPENING_LIST = { smoke: 'match_smoke', match: 'match' };
const OPENING_KIND = { smoke: 'match smoke opening', match: 'match opening' };

export class Invalid extends Error {}

const sha = (s) => createHash('sha256').update(s).digest('hex');

// ------------------------------------------------------------------------------------- independent re-derivations

/** Opening `index` of `kind` under the benchmark's rule, in this namespace (rows/cols 4-19, sorted, sha256[:12] mod n). */
export function deriveOpening(kind, index, ns = NS) {
  let s = new TwixtState();
  const moves = [];
  for (let k = 0; k < 4; k++) {
    const c = [];
    for (const [r, col] of s.legalMoves()) if (r >= 4 && r <= 19 && col >= 4 && col <= 19) c.push([r, col]);
    c.sort((a, b) => (a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1]));
    const pick = c[Number(BigInt(`0x${sha(`${ns}/${kind} ${index}/ply ${k}`).slice(0, 12)}`) % BigInt(c.length))];
    moves.push(pick);
    s = s.applyMove(pick);
  }
  return moves;
}

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

/** One-sided Hoeffding at alpha 0.05 over n pairs: sqrt(ln 20 / (2n)). */
export const passHalfWidth = (nPairs) => Math.sqrt(Math.log(1 / ALPHA_ONE_SIDED) / (2 * nPairs));
/** Two-sided 95% Hoeffding, descriptive only: sqrt(ln 40 / (2n)). */
export const twoSidedHalfWidth = (nPairs) => Math.sqrt(Math.log(40) / (2 * nPairs));
/** The pass rule (card §10). */
export const passes = (mean, nPairs = PAIRS.match) => mean - passHalfWidth(nPairs) > 0.5;

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
  if (g.schema !== 'htb-onpolicy-match-game/1' || g.stage !== stage || g.pair !== pair || g.game !== game) throw new Invalid(`${where}: identity fields`);
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

/**
 * The export handoff (the repair of 2026-11-14): the export wrapper's outer.json is a clean completion under the export
 * ceiling, bound to the export record by launch commit and pid. An inner success followed by an outer timeout fails.
 * The same checks as `bindingFailures`, for the export stage. An empty list admits.
 */
export function exportWrapperFailures({ outer: o, pgid, exportRecord: r }) {
  const f = [];
  const want = { inner_exit: 0, outer_exit: 0, outcome: 'COMPLETED', gate_restored: true, survivors: '', record_by: 'script', term_s: EXPORT_KILL_S - GRACE_S, grace_s: GRACE_S };
  for (const [k, v] of Object.entries(want)) if (o?.[k] !== v) f.push(`export outer ${k} ${JSON.stringify(o?.[k])}, expected ${JSON.stringify(v)}`);
  if (r?.kill_s !== EXPORT_KILL_S) f.push(`export record kill_s ${r?.kill_s}, the card's export ceiling is ${EXPORT_KILL_S}`);
  if (!/^[0-9a-f]{40}$/.test(r?.head ?? '') || o?.head !== r.head) f.push('the export record and its outer.json do not name the same launch commit');
  if (!Number.isInteger(r?.pid) || pgid !== `${r.pid}\n`) f.push('the export pgid.txt is not the export record\'s pid');
  return f;
}

/** G3's outer/inner binding, for this stage's ceiling and this match's players. An empty list admits. */
export function bindingFailures({ outer: o, record: rec, pgid, stage, exportRecord }) {
  const f = [];
  const want = { inner_exit: 0, outer_exit: 0, outcome: 'COMPLETED', gate_restored: true, survivors: '', record_by: 'script', term_s: KILL_S[stage] - GRACE_S, grace_s: GRACE_S };
  for (const [k, v] of Object.entries(want)) if (o?.[k] !== v) f.push(`outer ${k} ${JSON.stringify(o?.[k])}, expected ${JSON.stringify(v)}`);
  if (rec?.schema !== 'htb-onpolicy-match-record/1' || rec?.status !== 'COMPLETED' || rec?.written_by !== 'script' || rec?.stage !== stage) f.push(`record.json is not a COMPLETED ${stage} record`);
  if (rec?.kill_s !== KILL_S[stage]) f.push(`record kill_s ${rec?.kill_s}, the card's ${stage} ceiling is ${KILL_S[stage]}`);
  if (!/^[0-9a-f]{40}$/.test(rec?.head ?? '') || o?.head !== rec.head) f.push('the outer and inner records do not name the same launch commit');
  if (!Number.isInteger(rec?.pid) || pgid !== `${rec.pid}\n`) f.push('pgid.txt is not the driver pid');
  if (rec?.runtime?.onnxruntime_node !== PINS.onnxruntime_node) f.push(`onnxruntime-node ${rec?.runtime?.onnxruntime_node}`);
  if (rec?.current?.model_id !== PINS.current_model_id) f.push(`current model ${rec?.current?.model_id}`);
  if (rec?.policy?.nSims !== SIMS || rec?.policy?.search !== 'protocol' || rec?.policy?.moveTemp !== 1) f.push('the policy is not the 200-simulation protocol');
  if (rec?.openings_sha256 !== OPENINGS_SHA256) f.push('the record does not name the pinned openings');
  const c = exportRecord?.challenger;
  if (exportRecord?.status !== 'COMPLETED' || exportRecord?.verdict !== 'PARITY_PASS') f.push('the export record is not a COMPLETED PARITY_PASS');
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
    schema: 'htb-onpolicy-match-timing/1',
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

export function analyse(adm, nPairs = PAIRS.match) {
  const guard = searchSecondsGuard(adm.games);
  if (!guard.within) {
    return {
      schema: 'htb-onpolicy-match-analysis/1',
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
  const sum = pairs.reduce((a, b) => a + b, 0);
  const mean = sum / nPairs;
  const h = passHalfWidth(nPairs);
  const h2 = twoSidedHalfWidth(nPairs);
  const wdl = (gs) => ({ W: gs.filter((g) => g.challenger_score === 1).length, D: gs.filter((g) => g.challenger_score === 0.5).length, L: gs.filter((g) => g.challenger_score === 0).length });
  const colour = (c) => {
    const gs = adm.games.filter((g) => g.challenger_colour === c);
    return { ...wdl(gs), score: gs.reduce((a, g) => a + g.challenger_score, 0) / gs.length };
  };
  const verdict = passes(mean, nPairs) ? 'PASS' : 'NON_PASS';
  return {
    schema: 'htb-onpolicy-match-analysis/1',
    rule: 'PASS iff mean pair score - sqrt(ln 20 / (2n)) > 0.5 over the n = 272 pairs (one-sided Hoeffding, alpha 0.05)',
    pairs: nPairs,
    games: adm.games.length,
    challenger: wdl(adm.games),
    mean_pair_score: mean,
    pair_score_sum: sum,
    one_sided_half_width: h,
    lower_bound: mean - h,
    verdict,
    search_seconds_guard: guard,
    descriptive: {
      two_sided_95_half_width: h2,
      two_sided_95_interval: [mean - h2, mean + h2],
      by_colour: { challenger_red: colour('red'), challenger_black: colour('black') },
      draws: adm.games.filter((g) => g.result === 'draw').length,
      plies: summarise(adm.games.map((g) => g.plies)),
      distinct_games: new Set(adm.games.map((g) => JSON.stringify(g.moves))).size,
    },
    reading: verdict === 'PASS'
      ? 'PASS: the challenger\'s true mean pair score against the current hard is above 0.50 under this protocol (one-sided, alpha 0.05). This covers the whole update -- the new data source and the continued training together -- and does not attribute the gain to either. It does not establish a gain of 0.60, which is only the planning target. It authorizes proposing an integration decision, nothing more.'
      : 'NON_PASS: superiority over 0.50 is not shown. This is not evidence of no gain: the planning power (about 80% at a true 0.60) is model-based and assumes completion. The current hard stays; no top-up or re-score.',
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
    console.error('usage: onpolicy_match_analyse.mjs <timing|analyse>');
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
