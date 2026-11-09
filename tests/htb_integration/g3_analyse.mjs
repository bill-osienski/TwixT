#!/usr/bin/env node
/**
 * G3 analyser: validates a stage's evidence, then derives the match ceiling (smoke) or the verdict (match).
 *
 *   node tests/htb_integration/g3_analyse.mjs timing  docs/superpowers/evidence/htb-g3-smoke [--write]
 *   node tests/htb_integration/g3_analyse.mjs verdict docs/superpowers/evidence/htb-g3-match [--write]
 *
 * Each stage is judged with its wrapper's record in `<stage_dir>-run/` (outer.json, pgid.txt): the script's COMPLETED
 * record counts only if the wrapper also reports a clean, completed, bound run. `verdict` also re-validates the
 * committed smoke and re-derives its timing, which must equal the committed timing.json byte for byte.
 *
 * Card: docs/superpowers/2026-11-09-htb-g3-match-card.md §5 (timing), §6 (verdict), §7 (validation).
 *
 * It imports nothing from g3_match.mjs, the driver that produced the evidence: the seeds, colours, model identities,
 * simulation counts and rules below are transcribed from the card, and every game is replayed from the empty board
 * with the product's own rules (TwixtState, its only import beyond the standard library). Nothing is computed from the
 * data except the measurements the card names. A run that fails any check is INVALID and gets no number.
 *
 * --write puts timing.json / verdict.json in the stage directory, create-only. Output carries no timestamps, so
 * rerunning it on the same evidence gives the same bytes (the match wrapper relies on that).
 *
 * Exit: timing 0 WITHIN_24H, 1 EXCEEDS_24H; verdict 0 PASS, 1 FAIL; both 3 INVALID, 2 USAGE.
 */
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { TwixtState } from '../../server/gameLogic.js';

const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const BINDING_RELPATH = 'tests/htb_integration/g3_binding.json';
const SMOKE_RELPATH = 'docs/superpowers/evidence/htb-g3-smoke';
const SMOKE_TIMING_RELPATH = `${SMOKE_RELPATH}/timing.json`;

const deepFreeze = (o) => {
  for (const v of Object.values(o)) if (v && typeof v === 'object') deepFreeze(v);
  return Object.freeze(o);
};

/** Transcribed from the card. Tests may pass a replacement; the CLI never does. */
export const SPEC = deepFreeze({
  schema: 'htb-g3-game/1',
  recordSchema: 'htb-g3-record/1',
  matchSeed: 20261109,
  pairs: { smoke: 1, match: 110 },
  newModelId: '78c16a31c3dba37f',
  legacyModelId: 'c34b7ff3297c785a',
  sims: { new_hard: 200, legacy_hard: 800 },
  search: { new_hard: 'protocol', legacy_hard: 'plain' },
  halvingFromPly: 20,
  // §5.1 and the wrappers
  smokeTermS: 2390,
  graceS: 10,
  // §5 timing rule
  projectedGames: 220,
  minProjectedPlies: 64,
  allowance: 1.5,
  limitS: 86400,
  termLeadS: 10,
  cleanupS: 10,
  // §6 verdict rule
  verdictGames: 220,
  passAbove: 0.5,
});

const sha256 = (s) => createHash('sha256').update(s).digest('hex');
const sidecarName = (pair, game) => `pair_${String(pair).padStart(4, '0')}_game_${game}.json`;

/** The new hard's streams for one move (card §4), re-derived independently of the driver. */
export function expectedSeeds(spec, stage, pair, game, ply) {
  const h = sha256(`htb-g3/${spec.matchSeed}/${stage}/pair ${pair}/game ${game}/ply ${ply}`);
  return { search: Number.parseInt(h.slice(0, 12), 16), readout: Number.parseInt(h.slice(12, 24), 16) };
}

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const mean = (xs) => xs.reduce((a, x) => a + x, 0) / xs.length;
const quantile = (xs, q) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.ceil(q * s.length) - 1))];
};

/** Replay one game with the product's rules; failures as strings. */
function replay(sc) {
  const out = [];
  let state = new TwixtState({});
  for (let i = 0; i < sc.moves.length; i++) {
    if (state.isTerminal()) {
      out.push(`terminal before move ${i}`);
      return { out };
    }
    const [r, c] = sc.moves[i];
    if (!state.legalMoves().some((m) => m[0] === r && m[1] === c)) {
      out.push(`illegal move ${i} [${r},${c}]`);
      return { out };
    }
    state = state.applyMove([r, c]);
  }
  if (!state.isTerminal()) out.push('the last position is not terminal');
  const termination = state.winner() !== null ? 'win' : state.legalMoves().length === 0 ? 'no_legal_moves' : 'max_plies';
  return { out, state, termination };
}

/** The wrapper's outer record with its inner record (card §7); failures as strings. Independent of the driver's. */
export function outerFailures({ outer, pgid, record, termS, spec = SPEC }) {
  if (!outer) return ['wrapper: no readable outer.json'];
  const f = [];
  const want = { inner_exit: 0, outer_exit: 0, outcome: 'COMPLETED', gate_restored: true, survivors: '', record_by: 'script', term_s: termS, grace_s: spec.graceS };
  for (const [k, v] of Object.entries(want)) if (outer[k] !== v) f.push(`wrapper: ${k} ${JSON.stringify(outer[k])}, expected ${JSON.stringify(v)}`);
  if (!Number.isInteger(outer.wall_seconds) || outer.wall_seconds < 0) f.push('wrapper: wall_seconds');
  if (outer.head !== record?.head) f.push('wrapper: head is not the record head');
  if (!Number.isInteger(record?.pid) || pgid !== `${record.pid}\n`) f.push('wrapper: pgid.txt is not the record pid');
  return f;
}

/**
 * Validate a stage directory against the card. Returns {failures, record, games, progress}. `timing` (match only) is
 * the committed smoke timing whose kill_s the record must carry.
 */
export function validateRun({ dir, runDir, stage, binding, spec = SPEC, timing = null }) {
  const failures = [];
  const f = (m) => failures.push(m);
  const nPairs = spec.pairs[stage];
  const readJson = (p) => JSON.parse(readFileSync(p, 'utf8'));
  let record;
  try {
    record = readJson(join(dir, 'record.json'));
  } catch (e) {
    return { failures: [`record.json unreadable: ${e.message}`] };
  }
  if (record.schema !== spec.recordSchema) f('record schema');
  if (record.stage !== stage) f('record stage');
  if (record.status !== 'COMPLETED') f(`record status ${record.status}`);
  if (record.games_completed !== 2 * nPairs) f(`games_completed ${record.games_completed}`);
  if (record.binding_digest !== binding.digest) f('record binding digest');
  if (!(Number.isFinite(record.startup_ms) && record.startup_ms > 0)) f('record startup_ms');
  if (stage === 'match') {
    if (!timing || timing.verdict !== 'WITHIN_24H') f('no WITHIN_24H smoke timing');
    else if (record.kill_s !== timing.kill_s) f(`record kill_s ${record.kill_s} is not the timing's ${timing.kill_s}`);
  }
  let outer = null;
  let pgid = null;
  try {
    outer = readJson(join(runDir, 'outer.json'));
    pgid = readFileSync(join(runDir, 'pgid.txt'), 'utf8');
  } catch {
    /* judged below */
  }
  const termS = stage === 'smoke' ? spec.smokeTermS : timing?.term_s;
  for (const m of outerFailures({ outer, pgid, record, termS, spec })) f(m);

  const expectedNames = [];
  for (let p = 0; p < nPairs; p++) for (const g of [0, 1]) expectedNames.push(sidecarName(p, g));
  let present = [];
  try {
    present = readdirSync(join(dir, 'games')).sort();
  } catch {
    f('no games directory');
  }
  if (!same(present, [...expectedNames].sort())) f(`games/ holds ${present.length} files, expected exactly ${expectedNames.length}`);

  let progress = [];
  try {
    progress = readFileSync(join(dir, 'progress.jsonl'), 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
  } catch {
    f('progress.jsonl unreadable');
  }
  if (progress.length !== expectedNames.length) f(`progress has ${progress.length} lines`);

  const games = [];
  let prevEnded = record.startup_ms;
  for (let k = 0; k < expectedNames.length; k++) {
    const pair = Math.floor(k / 2);
    const game = k % 2;
    const tag = `pair ${pair} game ${game}`;
    let sc;
    try {
      sc = readJson(join(dir, 'games', expectedNames[k]));
    } catch {
      f(`${tag}: unreadable`);
      continue;
    }
    const newIsRed = game === 0;
    const expect = {
      schema: spec.schema,
      stage,
      pair_index: pair,
      game_in_pair: game,
      new_colour: newIsRed ? 'red' : 'black',
      red_player: newIsRed ? 'new_hard' : 'legacy_hard',
      black_player: newIsRed ? 'legacy_hard' : 'new_hard',
      new_model_id: spec.newModelId,
      legacy_model_id: spec.legacyModelId,
      match_seed: spec.matchSeed,
      binding_digest: binding.digest,
      head: record.head,
      node_version: binding.runtime.node,
    };
    for (const [key, v] of Object.entries(expect)) if (sc[key] !== v) f(`${tag}: ${key} ${JSON.stringify(sc[key])}`);
    if (binding.models.new_hard.model_id !== spec.newModelId || binding.models.legacy_hard.model_id !== spec.legacyModelId) {
      f('binding model ids differ from the card');
    }
    const { out, state, termination } = replay(sc);
    for (const m of out) f(`${tag}: ${m}`);
    if (state) {
      const result = state.gameResult();
      if (sc.result !== result) f(`${tag}: result ${sc.result}, replay ${result}`);
      if (sc.termination !== termination) f(`${tag}: termination ${sc.termination}, replay ${termination}`);
      if (sc.ply_count !== state.ply) f(`${tag}: ply_count ${sc.ply_count}, replay ${state.ply}`);
      const newColour = newIsRed ? 'red' : 'black';
      const score = result === 'draw' ? 0.5 : result === newColour ? 1 : 0;
      if (sc.new_score !== score) f(`${tag}: new_score ${sc.new_score}, replay ${score}`);
    }
    if (!Array.isArray(sc.per_move) || sc.per_move.length !== sc.moves.length) {
      f(`${tag}: per_move length`);
      continue;
    }
    let msSum = 0;
    for (let i = 0; i < sc.per_move.length; i++) {
      const pm = sc.per_move[i];
      const redToMove = i % 2 === 0;
      const label = redToMove === newIsRed ? 'new_hard' : 'legacy_hard';
      const want = {
        ply: i,
        player: label,
        n_simulations: spec.sims[label],
        search: spec.search[label],
        value_kind: label === 'new_hard' ? (i < spec.halvingFromPly ? 'root_mean' : 'played_move_mean') : null,
        seeds: label === 'new_hard' ? expectedSeeds(spec, stage, pair, game, i) : null,
      };
      for (const [key, v] of Object.entries(want)) if (!same(pm[key], v)) f(`${tag} ply ${i}: ${key} ${JSON.stringify(pm[key])}`);
      if (!(Number.isFinite(pm.ms) && pm.ms >= 0)) f(`${tag} ply ${i}: ms`);
      msSum += pm.ms;
    }
    if (sc.move_ms_total !== msSum) f(`${tag}: move_ms_total is not the sum of its moves`);
    const pl = progress[k];
    if (!pl || pl.pair_index !== pair || pl.game_in_pair !== game) f(`${tag}: progress line out of order`);
    else {
      if (pl.move_ms_total !== sc.move_ms_total || pl.ply_count !== sc.ply_count) f(`${tag}: progress disagrees with the sidecar`);
      if (!(pl.ended_ms > prevEnded && pl.ended_ms - prevEnded >= sc.move_ms_total)) f(`${tag}: progress time not increasing`);
    }
    games.push({ sc, segmentMs: pl ? pl.ended_ms - prevEnded : NaN });
    if (pl) prevEnded = pl.ended_ms;
  }
  return { failures, record, games, progress };
}

const movesOf = (games, label) => games.flatMap((g) => g.sc.per_move.filter((m) => m.player === label));

function latency(games, label, spec) {
  const all = movesOf(games, label);
  const band = (pred) => {
    const xs = all.filter(pred).map((m) => m.ms / 1000);
    return xs.length ? { moves: xs.length, mean_s: mean(xs), p50_s: quantile(xs, 0.5), p95_s: quantile(xs, 0.95), max_s: Math.max(...xs) } : { moves: 0 };
  };
  return {
    simulations_per_move: spec.sims[label],
    all: band(() => true),
    ply_below_20: band((m) => m.ply < spec.halvingFromPly),
    ply_20_on: band((m) => m.ply >= spec.halvingFromPly),
  };
}

/** Card §5: the projection and the hard ceiling, from a validated smoke. */
export function timingFrom(run, spec = SPEC) {
  const games = run.games;
  const meanPlies = mean(games.map((g) => g.sc.ply_count));
  const L = Math.max(spec.minProjectedPlies, meanPlies);
  const perMoveS = (label) => {
    const ms = movesOf(games, label).map((m) => m.ms);
    return ms.reduce((a, x) => a + x, 0) / ms.length / 1000;
  };
  const sNew = perMoveS('new_hard');
  const sLegacy = perMoveS('legacy_hard');
  const overheadS = mean(games.map((g) => (g.segmentMs - g.sc.move_ms_total) / 1000));
  const startupS = run.record.startup_ms / 1000;
  const perGameS = (L / 2) * (sNew + sLegacy) + overheadS;
  const projectionS = startupS + spec.projectedGames * perGameS;
  const killS = Math.ceil((spec.allowance * projectionS) / 60) * 60;
  return {
    kind: 'htb-g3-timing/1',
    inputs: {
      smoke_plies: games.map((g) => g.sc.ply_count),
      mean_plies: meanPlies,
      projected_plies: L,
      new_hard_s_per_move: sNew,
      legacy_hard_s_per_move: sLegacy,
      overhead_s_per_game: overheadS,
      startup_s: startupS,
    },
    formula: 'projection = startup + 220 x ((L / 2) x (s_new + s_legacy) + overhead); kill = ceil(1.5 x projection / 60) x 60; term = kill - 10; cleanup observed through kill + 10',
    per_game_s: perGameS,
    projection_s: projectionS,
    kill_s: killS,
    term_s: killS - spec.termLeadS,
    cleanup_through_s: killS + spec.cleanupS,
    limit_s: spec.limitS,
    verdict: killS <= spec.limitS ? 'WITHIN_24H' : 'EXCEEDS_24H',
    latency: { new_hard: latency(games, 'new_hard', spec), legacy_hard: latency(games, 'legacy_hard', spec) },
  };
}

/** Card §6: the rule, on the 220 game scores. */
export function verdictFromScores(scores, spec = SPEC) {
  if (scores.length !== spec.verdictGames) throw new Error(`${scores.length} scores, expected ${spec.verdictGames}`);
  const total = scores.reduce((a, x) => a + x, 0);
  const m = total / scores.length;
  const bound = Math.sqrt(Math.log(20) / spec.verdictGames);
  return { total_score: total, mean_pair_score: m, hoeffding_bound: bound, lower: m - bound, verdict: m - bound > spec.passAbove ? 'PASS' : 'FAIL' };
}

export function verdictFrom(run, timing, spec = SPEC) {
  const games = run.games;
  const scores = games.map((g) => g.sc.new_score);
  const v = verdictFromScores(scores, spec);
  const pairScores = [];
  for (let k = 0; k < games.length; k += 2) pairScores.push((scores[k] + scores[k + 1]) / 2);
  const count = (pred) => games.filter(pred).length;
  const plies = games.map((g) => g.sc.ply_count);
  const lat = { new_hard: latency(games, 'new_hard', spec), legacy_hard: latency(games, 'legacy_hard', spec) };
  return {
    kind: 'htb-g3-verdict/1',
    ...v,
    games: games.length,
    pairs: pairScores.length,
    pair_score_counts: Object.fromEntries([0, 0.25, 0.5, 0.75, 1].map((s) => [s, pairScores.filter((x) => x === s).length])),
    new_wins: count((g) => g.sc.new_score === 1),
    legacy_wins: count((g) => g.sc.new_score === 0),
    draws: count((g) => g.sc.new_score === 0.5),
    new_wins_as_red: count((g) => g.sc.new_colour === 'red' && g.sc.new_score === 1),
    new_wins_as_black: count((g) => g.sc.new_colour === 'black' && g.sc.new_score === 1),
    terminations: Object.fromEntries(['win', 'no_legal_moves', 'max_plies'].map((t) => [t, count((g) => g.sc.termination === t)])),
    plies: { mean: mean(plies), max: Math.max(...plies), min: Math.min(...plies) },
    distinct_games: new Set(games.map((g) => JSON.stringify(g.sc.moves))).size,
    costs: {
      latency: lat,
      search_seconds: {
        new_hard: movesOf(games, 'new_hard').reduce((a, m) => a + m.ms, 0) / 1000,
        legacy_hard: movesOf(games, 'legacy_hard').reduce((a, m) => a + m.ms, 0) / 1000,
      },
      s_per_move_ratio_new_over_legacy: lat.new_hard.all.mean_s / lat.legacy_hard.all.mean_s,
      run_seconds_to_last_game: run.progress.at(-1).ended_ms / 1000,
      projection_s: timing.projection_s,
      kill_s: timing.kill_s,
    },
  };
}

/** The analyser's own bytes must be the bound ones. */
export function selfCheck(binding, root = REPO_ROOT) {
  const bad = [];
  for (const [rel, h] of Object.entries(binding.analyser_files)) {
    if (sha256(readFileSync(join(root, rel))) !== h) bad.push(rel);
  }
  return bad;
}

const isMain = process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
  const [cmd, dir, ...rest] = process.argv.slice(2);
  const write = rest.length === 1 && rest[0] === '--write';
  if (!['timing', 'verdict'].includes(cmd) || !dir || (rest.length && !write)) {
    console.error('usage: g3_analyse.mjs timing|verdict <stage_dir> [--write]');
    process.exitCode = 2;
  } else {
    const binding = JSON.parse(readFileSync(join(REPO_ROOT, BINDING_RELPATH), 'utf8'));
    const stale = selfCheck(binding);
    const stage = cmd === 'timing' ? 'smoke' : 'match';
    const failures = stale.map((r) => `analyser file differs from the binding: ${r}`);
    let timing = null;
    if (stage === 'match') {
      // The smoke must be committed, successful, and still derive the committed timing byte for byte.
      const smokeDir = join(REPO_ROOT, SMOKE_RELPATH);
      const pending = execFileSync('git', ['status', '--porcelain', '--untracked-files=all', '--', SMOKE_RELPATH, `${SMOKE_RELPATH}-run`], { cwd: REPO_ROOT }).toString();
      if (pending) failures.push(`smoke evidence not committed: ${pending.split('\n')[0]}`);
      const smoke = validateRun({ dir: smokeDir, runDir: `${smokeDir}-run`, stage: 'smoke', binding });
      failures.push(...smoke.failures.map((m) => `smoke: ${m}`));
      try {
        const text = readFileSync(join(REPO_ROOT, SMOKE_TIMING_RELPATH), 'utf8');
        timing = JSON.parse(text);
        if (!smoke.failures.length && `${JSON.stringify(timingFrom(smoke), null, 1)}\n` !== text) failures.push('smoke: timing.json is not what the smoke derives');
      } catch {
        failures.push('smoke: timing.json unreadable');
      }
    }
    const run = validateRun({ dir, runDir: `${dir.replace(/\/+$/, '')}-run`, stage, binding, timing });
    failures.push(...run.failures);
    let out;
    let code;
    if (failures.length) {
      out = { kind: `htb-g3-${cmd}/1`, verdict: 'INVALID', failures: failures.slice(0, 50), failure_count: failures.length };
      code = 3;
    } else if (stage === 'smoke') {
      out = timingFrom(run);
      code = out.verdict === 'WITHIN_24H' ? 0 : 1;
    } else {
      out = verdictFrom(run, timing);
      code = out.verdict === 'PASS' ? 0 : 1;
    }
    const text = `${JSON.stringify(out, null, 1)}\n`;
    process.stdout.write(text);
    if (write) {
      const p = join(dir, `${cmd}.json`);
      if (existsSync(p)) {
        console.error(`${p} exists (create-only)`);
        code = 2;
      } else writeFileSync(p, text, { flag: 'wx' });
    }
    process.exitCode = code;
  }
}
