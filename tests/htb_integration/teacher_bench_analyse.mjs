/**
 * Student-teacher benchmark: the frozen analyser. Card: docs/superpowers/2026-11-14-htb-teacher-benchmark-card.md
 * (§6-§9, revision 1). Its sha256 is written into the card before any model loads; the wrappers refuse another.
 *
 *   node tests/htb_integration/teacher_bench_analyse.mjs timing    -> the smoke's timing.json (stdout); reads no outcome
 *   node tests/htb_integration/teacher_bench_analyse.mjs analyse   -> the match analysis (stdout)
 *
 * Admission (both): the stage's outer.json says COMPLETED with exit 0, the gate restored, no survivors, the record
 * written by the script and the stage's ceiling (TERM = KILL - 10 s); outer and inner records name the same launch
 * commit; pgid.txt is the driver's pid; record.json is COMPLETED with the stage's kill_s, the pinned runtimes and
 * the student's model and 200-simulation policy; every scheduled game file exists, and nothing else; every game replays
 * legally from the empty board under the product's game-end rule; its opening, colours, result and score equal what
 * is re-derived here; it has exactly one per-move row per played ply after the opening, in order; every arm P student
 * row carries the re-derived seeds, a 200-simulation budget and 200 root visits spent; arm N and teacher rows carry none. Anything else:
 * exit 3, no number. twixtbot's early-draw flag (both lanes) is observational: only its type is checked.
 *
 * The smoke's four games are admitted the same way (so they are replay-validated), but their outcomes are excluded
 * from scoring and from every report: `timing` reads wall times only.
 *
 * Openings and seeds are re-derived HERE, independently of teacher_bench.mjs (a test checks they agree).
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { TwixtState } from '../../server/gameLogic.js';

export const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const NS = 'htb-teacher-bench/20261114';
export const PAIRS = { smoke: { N: 1, P: 1 }, match: { N: 200, P: 110 } };
export const MATCH_CAP_S = 21600;
export const KILL_S = { smoke: 1800, match: 21600 };
export const GRACE_S = 10;
export const P_SIMS = 200;
export const PINS = { model_id: '78c16a31c3dba37f', onnxruntime_node: '1.23.2', teacher_env: { python: '3.13.15', tensorflow: '2.21.0', numpy: '2.5.2' } };
export const LAUNCH_FACTOR = 1.5;
export const DIRS = {
  smoke: { run: 'docs/superpowers/evidence/htb-teacher-bench-smoke', outer: 'docs/superpowers/evidence/htb-teacher-bench-smoke-run' },
  match: { run: 'docs/superpowers/evidence/htb-teacher-bench-match', outer: 'docs/superpowers/evidence/htb-teacher-bench-match-run' },
};

export class Invalid extends Error {}

const sha = (s) => createHash('sha256').update(s).digest('hex');

// ------------------------------------------------------------------------------------- independent re-derivations

export function deriveOpening(kind, index) {
  let s = new TwixtState();
  const moves = [];
  for (let k = 0; k < 4; k++) {
    const c = [];
    for (const [r, col] of s.legalMoves()) if (r >= 4 && r <= 19 && col >= 4 && col <= 19) c.push([r, col]);
    c.sort((a, b) => (a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1]));
    const pick = c[Number(BigInt(`0x${sha(`${NS}/${kind} ${index}/ply ${k}`).slice(0, 12)}`) % BigInt(c.length))];
    moves.push(pick);
    s = s.applyMove(pick);
  }
  return moves;
}

export function deriveOpenings(kind, n) {
  const seen = new Set();
  const out = [];
  let i = 0;
  while (out.length < n) {
    const m = deriveOpening(kind, i);
    const a = JSON.stringify(m);
    const b = JSON.stringify(m.map(([r, c]) => [r, 23 - c]));
    if (!seen.has(a) && !seen.has(b)) {
      seen.add(a);
      out.push({ index: i, moves: m });
    }
    i += 1;
  }
  return out;
}

export function deriveSeeds(stage, pair, game, ply) {
  const h = sha(`${NS}/${stage}/pair ${pair}/game ${game}/ply ${ply}`);
  return { search: Number(BigInt(`0x${h.slice(0, 12)}`)), readout: Number(BigInt(`0x${h.slice(12, 24)}`)) };
}

export const hoeffdingHalfWidth = (nPairs) => Math.sqrt(Math.log(40) / (2 * nPairs));

export function classify(mean, nPairs) {
  const h = hoeffdingHalfWidth(nPairs);
  if (mean + h < 0.5) return 'TRAILS';
  if (mean - h > 0.5) return 'AHEAD';
  return 'UNRESOLVED';
}

// ---------------------------------------------------------------------------------------------------- admission

function readJson(p, what) {
  if (!existsSync(p)) throw new Invalid(`${what} missing: ${p}`);
  try {
    return JSON.parse(readFileSync(p, 'utf8'));
  } catch {
    throw new Invalid(`${what} unparseable: ${p}`);
  }
}

/** Validates one game against the schedule; returns it. Reads its outcome only for the match analysis. */
export function validateGame(g, { stage, arm, pair, game, opening }) {
  const where = `${arm} pair ${pair} game ${game}`;
  if (g.schema !== 'htb-teacher-bench-game/1' || g.stage !== stage || g.arm !== arm || g.pair !== pair || g.game !== game) throw new Invalid(`${where}: identity fields`);
  const colour = game === 0 ? 'red' : 'black';
  if (g.student_colour !== colour) throw new Invalid(`${where}: student colour`);
  if (g.opening_index !== opening.index || JSON.stringify(g.opening) !== JSON.stringify(opening.moves)) throw new Invalid(`${where}: opening`);
  if (JSON.stringify(g.moves.slice(0, 4)) !== JSON.stringify(opening.moves)) throw new Invalid(`${where}: moves do not start with the opening`);
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
  const score = winner === null ? 0.5 : winner === colour ? 1 : 0;
  if (g.student_score !== score) throw new Invalid(`${where}: score`);
  // Exactly one row per played ply after the opening, in order: no duplicate, no gap, no extra.
  if (!Array.isArray(g.per_move) || g.per_move.length !== g.moves.length - 4) throw new Invalid(`${where}: ${g.per_move?.length} per-move rows for ${g.moves.length - 4} played plies`);
  g.per_move.forEach((pm, i) => {
    if (pm.ply !== 4 + i) throw new Invalid(`${where}: per-move row ${i} is ply ${pm.ply}, expected ${4 + i}`);
    const mover = (pm.ply % 2 === 0 ? 'red' : 'black') === colour ? 'student' : 'teacher';
    if (pm.mover !== mover || JSON.stringify(pm.move) !== JSON.stringify(g.moves[pm.ply])) throw new Invalid(`${where}: per-move ply ${pm.ply}`);
    if (mover === 'student' && arm === 'P') {
      if (JSON.stringify(pm.seeds) !== JSON.stringify(deriveSeeds(stage, pair, game, pm.ply))) throw new Invalid(`${where}: seeds at ply ${pm.ply}`);
      if (pm.nSimulations !== P_SIMS || pm.visits !== P_SIMS) throw new Invalid(`${where}: budget ${pm.nSimulations}, visits ${pm.visits} at ply ${pm.ply}; the card's is ${P_SIMS}`);
    } else if (pm.seeds !== undefined || pm.nSimulations !== undefined || pm.visits !== undefined) {
      throw new Invalid(`${where}: a ${mover} row in arm ${arm} carries a search budget or seeds at ply ${pm.ply}`);
    }
  });
  for (const f of ['tb_draw_flag_ply', 'tb_draw_flag_ply_reflected']) {
    if (g[f] !== null && !(Number.isInteger(g[f]) && g[f] >= 0 && g[f] <= g.plies)) throw new Invalid(`${where}: ${f}`);
  }
  if (!(g.wall_ms > 0)) throw new Invalid(`${where}: wall time`);
  return g;
}

/** G3's outer/inner binding (g3_analyse.outerFailures), for this stage's ceiling. An empty list admits. */
export function bindingFailures({ outer: o, record: rec, pgid, stage }) {
  const f = [];
  const want = { inner_exit: 0, outer_exit: 0, outcome: 'COMPLETED', gate_restored: true, survivors: '', record_by: 'script', term_s: KILL_S[stage] - GRACE_S, grace_s: GRACE_S };
  for (const [k, v] of Object.entries(want)) if (o?.[k] !== v) f.push(`outer ${k} ${JSON.stringify(o?.[k])}, expected ${JSON.stringify(v)}`);
  if (rec?.status !== 'COMPLETED' || rec?.written_by !== 'script' || rec?.stage !== stage) f.push(`record.json is not a COMPLETED ${stage} record`);
  if (rec?.kill_s !== KILL_S[stage]) f.push(`record kill_s ${rec?.kill_s}, the card's ${stage} ceiling is ${KILL_S[stage]}`);
  if (!/^[0-9a-f]{40}$/.test(rec?.head ?? '') || o?.head !== rec.head) f.push('the outer and inner records do not name the same launch commit');
  if (!Number.isInteger(rec?.pid) || pgid !== `${rec.pid}\n`) f.push('pgid.txt is not the driver pid');
  if (rec?.runtime?.onnxruntime_node !== PINS.onnxruntime_node) f.push(`onnxruntime-node ${rec?.runtime?.onnxruntime_node}`);
  if (JSON.stringify(rec?.runtime?.teacher_env) !== JSON.stringify(PINS.teacher_env)) f.push(`teacher runtime ${JSON.stringify(rec?.runtime?.teacher_env)}`);
  if (rec?.student?.model_id !== PINS.model_id || rec?.student?.policy?.nSims !== P_SIMS) f.push('student model or budget');
  return f;
}

export function admit(stage, root = REPO_ROOT) {
  const { run, outer } = DIRS[stage];
  const o = readJson(join(root, outer, 'outer.json'), 'outer.json');
  const rec = readJson(join(root, run, 'record.json'), 'record.json');
  const pgidPath = join(root, outer, 'pgid.txt');
  const pgid = existsSync(pgidPath) ? readFileSync(pgidPath, 'utf8') : null;
  const bf = bindingFailures({ outer: o, record: rec, pgid, stage });
  if (bf.length) throw new Invalid(`the ${stage} evidence is not bound to its run: ${bf.join('; ')}`);
  const kind = stage === 'smoke' ? 'smoke opening' : 'opening';
  const openings = deriveOpenings(kind, Math.max(PAIRS[stage].N, PAIRS[stage].P));
  const expected = [];
  for (const arm of ['N', 'P']) for (let p = 0; p < PAIRS[stage][arm]; p++) for (const g of [0, 1]) expected.push({ arm, pair: p, game: g });
  const names = expected.map((t) => `${t.arm}_pair_${String(t.pair).padStart(4, '0')}_game_${t.game}.json`);
  const present = readdirSync(join(root, run, 'games')).sort();
  if (JSON.stringify(present) !== JSON.stringify([...names].sort())) throw new Invalid(`the games directory does not hold exactly the ${names.length} scheduled games`);
  if (rec.games !== names.length) throw new Invalid('record.json game count');
  const games = expected.map((t, i) => validateGame(readJson(join(root, run, 'games', names[i]), names[i]), { stage, ...t, opening: openings[t.pair] }));
  return { outer: o, record: rec, games };
}

// ---------------------------------------------------------------------------------------------------------- timing

/** The launch estimate from the smoke: wall times only, never an outcome. */
export function timing(adm) {
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  const per = (arm) => mean(adm.games.filter((g) => g.arm === arm).map((g) => g.wall_ms / 1000));
  const startup = adm.record.startup_seconds;
  if (!(startup > 0)) throw new Invalid('record.json startup_seconds');
  const nGame = per('N');
  const pGame = per('P');
  const projection = startup + nGame * 2 * PAIRS.match.N + pGame * 2 * PAIRS.match.P;
  const estimate = LAUNCH_FACTOR * projection;
  const r = (x) => Math.round(x * 1000) / 1000;
  return {
    schema: 'htb-teacher-bench-timing/1',
    startup_s: r(startup),
    n_game_s_mean: r(nGame),
    p_game_s_mean: r(pGame),
    smoke_plies: adm.games.map((g) => g.plies),
    projection_s: r(projection),
    launch_factor: LAUNCH_FACTOR,
    launch_estimate_s: r(estimate),
    match_cap_s: MATCH_CAP_S,
    verdict: estimate <= MATCH_CAP_S ? 'LAUNCH_OK' : 'NO_LAUNCH',
    note: 'an estimate from 2 games per arm; long draws can exceed it; the 6.0 h match cap is fixed',
    smoke_outcomes: 'replay-validated on admission; excluded from scoring and from every report',
  };
}

// ---------------------------------------------------------------------------------------------------------- analysis

function summarise(xs) {
  const s = [...xs].sort((a, b) => a - b);
  return { n: s.length, mean: s.reduce((a, b) => a + b, 0) / s.length, median: s[s.length >> 1], min: s[0], max: s[s.length - 1] };
}

function armResult(games, nPairs) {
  const pairs = [];
  for (let p = 0; p < nPairs; p++) {
    const two = games.filter((g) => g.pair === p);
    if (two.length !== 2) throw new Invalid(`pair ${p} has ${two.length} games`);
    pairs.push((two[0].student_score + two[1].student_score) / 2);
  }
  const mean = pairs.reduce((a, b) => a + b, 0) / nPairs;
  const h = hoeffdingHalfWidth(nPairs);
  const wdl = (gs) => ({ W: gs.filter((g) => g.student_score === 1).length, D: gs.filter((g) => g.student_score === 0.5).length, L: gs.filter((g) => g.student_score === 0).length });
  const colour = (c) => {
    const gs = games.filter((g) => g.student_colour === c);
    return { ...wdl(gs), score: gs.reduce((a, g) => a + g.student_score, 0) / gs.length };
  };
  const flagged = games.filter((g) => g.tb_draw_flag_ply !== null);
  return {
    pairs: nPairs,
    games: games.length,
    student: wdl(games),
    mean_pair_score: mean,
    half_width: h,
    interval: [mean - h, mean + h],
    classification: classify(mean, nPairs),
    by_colour: { student_red: colour('red'), student_black: colour('black') },
    plies: summarise(games.map((g) => g.plies)),
    tb_draw_flag: {
      games: flagged.length,
      in_draws: flagged.filter((g) => g.result === 'draw').length,
      in_decisive: flagged.filter((g) => g.result !== 'draw').length,
      draws_without_flag: games.filter((g) => g.result === 'draw' && g.tb_draw_flag_ply === null).length,
      flag_ply: flagged.length ? summarise(flagged.map((g) => g.tb_draw_flag_ply)) : null,
    },
    pair_scores: pairs,
  };
}

const READING = {
  TRAILS: 'The student\'s raw policy still trails the policy it learned from, under our rules. This would justify proposing a fresh distillation experiment (student\'s own play, labelled by this teacher, fitted from the current student, positions kept separate from evaluation, tested against the current student on fresh games); it needs its own approval.',
  UNRESOLVED: 'The question stays open; this card supports no follow-up.',
  AHEAD: 'The student\'s raw policy no longer trails its teacher under our rules; this card supports nothing toward more distillation from this teacher.',
};

export function analyse(adm, pairs = PAIRS.match) {
  const N = armResult(adm.games.filter((g) => g.arm === 'N'), pairs.N);
  const P = armResult(adm.games.filter((g) => g.arm === 'P'), pairs.P);
  const common = P.pair_scores.map((s, p) => s - N.pair_scores[p]);
  return {
    schema: 'htb-teacher-bench-analysis/1',
    method: 'two-sided Hoeffding 95% over pairs, half-width sqrt(ln 40 / (2n)); TRAILS if mean + h < 0.5, AHEAD if mean - h > 0.5',
    arm_N: N,
    arm_P: P,
    arm_P_minus_N_common_openings: { pairs: common.length, mean: common.reduce((a, b) => a + b, 0) / common.length },
    reading: { decided_by: 'arm N only (card §8)', arm_N: N.classification, text: READING[N.classification], arm_P_note: 'Arm P is product context only: it mixes the network, our search and the T 1.0 opening sampling, and cannot say which accounts for its result.' },
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
      process.stdout.write(`${JSON.stringify(analyse(admit('match')), null, 1)}\n`);
      return 0;
    }
    console.error('usage: teacher_bench_analyse.mjs <timing|analyse>');
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
