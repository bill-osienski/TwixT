/**
 * Diagnostic O (opening attribution), the user's revised approval of 2026-11-12. Plan:
 * docs/superpowers/2026-11-12-htb-diagnostic-o-plan.md. It plays no game and trains nothing.
 *
 * On each of the 1,100 saved positions where the student (new `hard`, model 78c16a31c3dba37f) moved as Black in the
 * opening of a G3 match game (plies 1, 3, ..., 19 of the 110 `pair_*_game_1.json` games):
 *
 *   A. the student, through the product's own inference stack (AlphaZeroInference wrapped in
 *      MirrorAveragingInference, exactly as server/index.js and the G3 driver build it):
 *      - the 200-simulation search, replicated from protocolMove's plies-0-19 branch with the recorded search seed,
 *        then the T 1.0 readout with the recorded readout seed. The sampled move MUST equal the recorded move; any
 *        mismatch stops the run (exit 3) before anything is analysed;
 *      - the 800-simulation search: the same class, network, position and search seed (and so the same tie rule);
 *      - the raw network on the position and on the position after each candidate move (played, 200 leader,
 *        800 leader);
 *   B. the pinned twixtbot raw network (scripts/GPU/alphazero/htb_diag_o_teacher.py, in the teacher venv), mirror-
 *      averaged as in the H-TB label path, on the same positions;
 *   C. the frozen analysis (tests/htb_integration/diag_o_analyse.mjs), whose sha256 the plan records.
 *
 * Visit leader: most visits; ties by higher raw (root) prior, then (row, col) ascending.
 * All values are reported from Black's side; the search values separately from the raw values.
 *
 * Usage: node tests/htb_integration/diag_o.mjs run --teacher-python <py> --clone-root <dir> --kill-s <s>
 * Exit: 0 COMPLETED, 2 USAGE, 3 VOID (a stop), 4 REFUSED (gate closed or a precondition).
 */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

import { TwixtState } from '../../server/gameLogic.js';
import { PyRandom } from '../../server/py_random.js';
import { sampleByTemperature } from '../../server/protocol_mcts.js';
import { RootSequentialHalvingMCTS } from '../../server/root_halving.js';

export const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
export const GAMES_REL = 'docs/superpowers/evidence/htb-g3-match/games';
export const GROUPS_REL = 'docs/superpowers/evidence/htb-diag-o-plan/race_groups.json';
export const RUN_REL = 'docs/superpowers/evidence/htb-diag-o';
export const ANALYSER_REL = 'tests/htb_integration/diag_o_analyse.mjs';
export const STUDENT_MODEL_ID = '78c16a31c3dba37f';
export const PLIES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19];
export const SIMS_DEEP = 800;
export const C_PUCT = 1.5; // protocolMove's

export class DiagStop extends Error {
  constructor(step, message) {
    super(message);
    this.name = 'DiagStop';
    this.step = step;
  }
}
export class DiagRefusal extends Error {
  constructor(message) {
    super(message);
    this.name = 'DiagRefusal';
  }
}

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');
const keyOf = (m) => `${m[0]},${m[1]}`;
const decode = (k) => k.split(',').map(Number);

/** Most visits; ties by higher prior, then (row, col) ascending. */
export function visitLeader(visitCounts, priors) {
  let best = null;
  for (const [k, n] of visitCounts) {
    if (best === null) {
      best = k;
      continue;
    }
    const nb = visitCounts.get(best);
    if (n > nb) best = k;
    else if (n === nb) {
      const pk = priors.get(k) ?? 0;
      const pb = priors.get(best) ?? 0;
      if (pk > pb) best = k;
      else if (pk === pb) {
        const [rk, ck] = decode(k);
        const [rb, cb] = decode(best);
        if (rk < rb || (rk === rb && ck < cb)) best = k;
      }
    }
  }
  return best;
}

/** 1 + the number of moves with strictly more visits. */
export function visitRank(visitCounts, key) {
  const n = visitCounts.get(key) ?? 0;
  let r = 1;
  for (const v of visitCounts.values()) if (v > n) r += 1;
  return r;
}

function searchSummary(mcts, out, priors, played) {
  const counts = out.visitCounts;
  let total = 0;
  for (const v of counts.values()) total += v;
  const leader = visitLeader(counts, priors);
  const moveValueBlack = (k) => {
    const child = mcts.lastRoot?.children.get(k);
    return child && child.visitCount > 0 ? -child.qValue : null; // the mover (Black) side, as protocolMove
  };
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
  return {
    total_visits: total,
    leader,
    leader_visits: counts.get(leader) ?? 0,
    played_visits: counts.get(played) ?? 0,
    played_share: total > 0 ? (counts.get(played) ?? 0) / total : null,
    played_rank: visitRank(counts, played),
    root_value_black: out.rootValue, // the root mean, side to move = Black
    move_value_black: { played: moveValueBlack(played), leader: moveValueBlack(leader) },
    top10: top.map(([k, v]) => [k, v]),
  };
}

/**
 * Stage A on one game: yields one record per opening position. `inference` is the student's product stack.
 * Stops (DiagStop) on any identity or reproduction failure.
 */
export async function* studentStage(game, inference, { policy, simsDeep = SIMS_DEEP, onTiming } = {}) {
  if (game.stage !== 'match' || game.new_colour !== 'black' || game.game_in_pair !== 1) {
    throw new DiagStop('input', `pair ${game.pair_index}: not a match game with the student as Black`);
  }
  if (game.new_model_id !== STUDENT_MODEL_ID) throw new DiagStop('input', `pair ${game.pair_index}: model ${game.new_model_id}`);
  let state = new TwixtState();
  for (let k = 0; k <= PLIES[PLIES.length - 1]; k++) {
    const recorded = game.moves[k];
    if (PLIES.includes(k)) {
      const pm = game.per_move[k];
      if (!pm || pm.ply !== k || pm.player !== 'new_hard' || !pm.seeds) {
        throw new DiagStop('input', `pair ${game.pair_index} ply ${k}: no recorded new_hard seeds`);
      }
      if (state.toMove !== 'black' || state.ply !== k) throw new DiagStop('input', `pair ${game.pair_index} ply ${k}: state mismatch`);
      const t0 = performance.now();
      const legal = state.legalMoves();
      const raw = await inference.evaluate(state.toTensorHWC(), legal);
      const priors = raw.priors;

      // 200 simulations: protocolMove's plies-0-19 branch, replicated so the tree can be read.
      const m200 = new RootSequentialHalvingMCTS(inference, { nSimulations: policy.nSims, cPuct: C_PUCT, rng: new PyRandom(pm.seeds.search) });
      const o200 = await m200.search(TwixtState.fromDict(state.toDict()));
      if (o200.choice !== null || o200.aborted) throw new DiagStop('search', `pair ${game.pair_index} ply ${k}: unexpected choice/abort`);
      const played = sampleByTemperature(o200.visitCounts, policy.moveTemp, new PyRandom(pm.seeds.readout));
      if (played !== keyOf(recorded)) {
        throw new DiagStop('reproduction', `pair ${game.pair_index} ply ${k}: reproduced ${played}, recorded ${keyOf(recorded)}`);
      }
      const t1 = performance.now();
      const s200 = searchSummary(m200, o200, priors, played);

      // 800 simulations: same class, network, position and search seed.
      const m800 = new RootSequentialHalvingMCTS(inference, { nSimulations: simsDeep, cPuct: C_PUCT, rng: new PyRandom(pm.seeds.search) });
      const o800 = await m800.search(TwixtState.fromDict(state.toDict()));
      if (o800.choice !== null || o800.aborted) throw new DiagStop('search', `pair ${game.pair_index} ply ${k}: unexpected choice/abort (800)`);
      const s800 = searchSummary(m800, o800, priors, played);
      const t2 = performance.now();

      // Raw student after each candidate (Red to move there; reported from Black's side).
      const cands = [...new Set([played, s200.leader, s800.leader])];
      const after = {};
      for (const c of cands) {
        const s2 = state.applyMove(decode(c));
        const r2 = await inference.evaluate(s2.toTensorHWC(), s2.legalMoves());
        after[c] = -r2.value;
      }
      const sortedPriors = [...priors.entries()].sort((a, b) => b[1] - a[1]);
      const priorRank = (c) => 1 + sortedPriors.filter(([, p]) => p > (priors.get(c) ?? 0)).length;
      const rec = {
        pair: game.pair_index,
        ply: k,
        prefix: game.moves.slice(0, k),
        played,
        leader200: s200.leader,
        leader800: s800.leader,
        candidates: cands,
        student_raw: {
          value_black: raw.value, // side to move = Black
          prior: Object.fromEntries(cands.map((c) => [c, priors.get(c) ?? 0])),
          prior_rank: Object.fromEntries(cands.map((c) => [c, priorRank(c)])),
          top_prior_move: sortedPriors[0][0],
          after_value_black: after,
        },
        search200: s200,
        search800: s800,
        ms: { s200: t1 - t0, s800: t2 - t1 },
      };
      if (onTiming) onTiming(rec.ms);
      yield rec;
    }
    state = state.applyMove(recorded);
  }
}

export function loadGames(root = REPO_ROOT) {
  const dir = join(root, GAMES_REL);
  const files = readdirSync(dir).filter((f) => f.endsWith('_game_1.json')).sort();
  return files.map((f) => ({ file: f, game: JSON.parse(readFileSync(join(dir, f), 'utf8')) }));
}

function parseArgs(argv) {
  const [cmd, ...rest] = argv;
  const opts = {};
  for (let i = 0; i < rest.length; i += 2) {
    if (!rest[i].startsWith('--') || rest[i + 1] === undefined) return null;
    opts[rest[i].slice(2)] = rest[i + 1];
  }
  return { cmd, opts };
}

async function run(opts) {
  const GATE = await import('./diag_o_authorization.mjs');
  if (GATE.DIAG_O_AUTHORIZED !== true) throw new DiagRefusal('DIAG_O_AUTHORIZED is closed');
  for (const k of ['teacher-python', 'clone-root', 'kill-s']) if (!opts[k]) throw new DiagRefusal(`missing --${k}`);
  const out = join(REPO_ROOT, RUN_REL);
  if (existsSync(out)) throw new DiagRefusal(`${RUN_REL} exists (one-shot)`);
  const groups = JSON.parse(readFileSync(join(REPO_ROOT, GROUPS_REL), 'utf8'));
  const games = loadGames();
  if (games.length !== 110 || groups.n_games !== 110) throw new DiagStop('input', `expected 110 Black games, got ${games.length}`);

  // Imported here so the module (and its tests) never touch ONNX Runtime on import.
  const { resolveHardModel } = await import('../../server/model_manifest.js');
  const { resolvePolicy } = await import('../../server/readout_policy.js');
  const { AlphaZeroInference } = await import('../../server/inference.js');
  const { MirrorAveragingInference } = await import('../../server/mirror_inference.js');
  const hard = await resolveHardModel({}, REPO_ROOT);
  const modelId = hard.manifest.model_id ?? hard.manifest.modelId ?? hard.manifest.id;
  if (modelId !== STUDENT_MODEL_ID) throw new DiagRefusal(`hard model is ${modelId}, expected ${STUDENT_MODEL_ID}`);
  const policy = resolvePolicy({ difficulty: 'hard', hardMode: 'protocol' });
  if (policy.nSims !== 200 || policy.moveTemp !== 1.0 || policy.search !== 'protocol') throw new DiagRefusal(`unexpected policy ${JSON.stringify(policy)}`);

  mkdirSync(out);
  const started = new Date().toISOString();
  const tStart = performance.now();
  const progress = join(out, 'progress.jsonl');
  const studentFile = join(out, 'student.jsonl');
  const base = new AlphaZeroInference(hard.graphPath);
  await base.load();
  const inference = new MirrorAveragingInference(base);
  const onnxSha = sha256(readFileSync(hard.graphPath));

  let n = 0;
  const timing = { s200: 0, s800: 0 };
  for (const { file, game } of games) {
    for await (const rec of studentStage(game, inference, { policy, onTiming: (ms) => { timing.s200 += ms.s200; timing.s800 += ms.s800; } })) {
      appendFileSync(studentFile, JSON.stringify({ file, ...rec }) + '\n');
      n += 1;
    }
    appendFileSync(progress, JSON.stringify({ t: new Date().toISOString(), file, positions: n }) + '\n');
  }
  if (n !== 1100) throw new DiagStop('student', `${n} positions, expected 1,100`);
  const tStudent = performance.now();

  // Stage B: the teacher, in its own venv, on the same positions and the positions after each candidate.
  const teacherIn = join(out, 'teacher_in.json');
  const lines = readFileSync(studentFile, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
  writeFileSync(teacherIn, JSON.stringify(lines.map((r) => ({ pair: r.pair, ply: r.ply, prefix: r.prefix, candidates: r.candidates }))) + '\n', { flag: 'wx' });
  const tb = spawnSync(opts['teacher-python'], ['-B', '-m', 'scripts.GPU.alphazero.htb_diag_o_teacher', 'run', '--clone-root', opts['clone-root'], '--in', teacherIn, '--out', join(out, 'teacher.json')], {
    cwd: REPO_ROOT,
    env: { ...process.env, PYTHONPATH: '.' },
    stdio: ['ignore', 'inherit', 'inherit'],
  });
  if (tb.status !== 0) throw new DiagStop('teacher', `teacher stage exit ${tb.status} signal ${tb.signal}`);
  const tTeacher = performance.now();

  // Stage C: the frozen analysis.
  const an = spawnSync(process.execPath, [join(REPO_ROOT, ANALYSER_REL), 'analyse', out], { cwd: REPO_ROOT, stdio: ['ignore', 'pipe', 'inherit'] });
  if (an.status !== 0) throw new DiagStop('analysis', `analyser exit ${an.status}`);
  writeFileSync(join(out, 'analysis.json'), an.stdout, { flag: 'wx' });
  const tEnd = performance.now();

  const record = {
    schema: 'htb-diag-o-record/1',
    written_by: 'script',
    status: 'COMPLETED',
    started,
    ended: new Date().toISOString(),
    positions: n,
    reproduced: n,
    student: { model_id: STUDENT_MODEL_ID, onnx_sha256: onnxSha, inference: 'MirrorAveragingInference(AlphaZeroInference)', policy },
    sims_deep: SIMS_DEEP,
    seconds: { student: (tStudent - tStart) / 1000, teacher: (tTeacher - tStudent) / 1000, analysis: (tEnd - tTeacher) / 1000, total: (tEnd - tStart) / 1000 },
    search_seconds: { s200: timing.s200 / 1000, s800: timing.s800 / 1000 },
    analyser_sha256: sha256(readFileSync(join(REPO_ROOT, ANALYSER_REL))),
    groups_sha256: sha256(readFileSync(join(REPO_ROOT, GROUPS_REL))),
    kill_s: Number(opts['kill-s']),
    head: spawnSync('git', ['rev-parse', 'HEAD'], { cwd: REPO_ROOT, encoding: 'utf8' }).stdout.trim(),
    node_version: process.version,
  };
  writeFileSync(join(out, 'record.json'), JSON.stringify(record, null, 1) + '\n', { flag: 'wx' });
}

async function main() {
  const a = parseArgs(process.argv.slice(2));
  if (!a || a.cmd !== 'run') {
    console.error('usage: node tests/htb_integration/diag_o.mjs run --teacher-python <py> --clone-root <dir> --kill-s <s>');
    return 2;
  }
  try {
    await run(a.opts);
    return 0;
  } catch (err) {
    if (err instanceof DiagRefusal) {
      console.error(`REFUSED: ${err.message}`);
      return 4;
    }
    if (err instanceof DiagStop) {
      console.error(`STOP at ${err.step}: ${err.message}`);
      const out = join(REPO_ROOT, RUN_REL);
      if (existsSync(out) && !existsSync(join(out, 'record.json'))) {
        writeFileSync(join(out, 'record.json'), JSON.stringify({ schema: 'htb-diag-o-record/1', written_by: 'script', status: 'VOID', void: { step: err.step, detail: err.message } }, null, 1) + '\n', { flag: 'wx' });
      }
      return 3;
    }
    console.error(err?.stack ?? String(err));
    return 1;
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().then((code) => process.exit(code));
}
