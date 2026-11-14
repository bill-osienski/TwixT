/**
 * Student-teacher benchmark driver. Card: docs/superpowers/2026-11-14-htb-teacher-benchmark-card.md (revision 1).
 *
 * The current H-TB student (product hard model 78c16a31c3dba37f, MirrorAveragingInference(AlphaZeroInference), as
 * server/index.js builds it) against its pinned twixtbot teacher (scripts/GPU/alphazero/htb_bench_teacher.py, in the
 * teacher venv: the label path's mirror-averaged raw policy, argmax, no search), under our rules.
 *
 *   arm N: the student network's mirror-averaged policy argmax over legal moves, ties to the lower (row, col).
 *   arm P: the student as shipped in hard: protocolMove (200 sims, T 1.0 readout through ply 19), seeded per move.
 *
 * Every game starts from a supplied 4-ply opening; pair p plays opening p with the student as Red (game 0) and as
 * Black (game 1). The PRODUCT's game-end rule is authoritative (Node TwixtState.isTerminal(), no cap). twixtbot's
 * early-draw flag is recorded per game and never ends one. After the opening and after every ply, the canonical state
 * (ply, side to move, pegs and bridges by colour, legal set, winner, rules-level terminal) is hashed here and in the
 * teacher process, and any difference stops the run.
 *
 * Usage: node tests/htb_integration/teacher_bench.mjs run <smoke|match> --teacher-python <py> --clone-root <dir> --kill-s <s>
 * Exit (set through process.exitCode; never process.exit, ledger entry 64): 0 COMPLETED, 2 USAGE, 3 VOID, 4 REFUSED.
 */
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

import { TwixtState } from '../../server/gameLogic.js';
import { protocolMove } from '../../server/protocol_move.js';
import { resolvePolicy } from '../../server/readout_policy.js';

export const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
export const NS = 'htb-teacher-bench/20261114';
export const STUDENT_MODEL_ID = '78c16a31c3dba37f';
export const CENTRAL_MIN = 4;
export const CENTRAL_MAX = 19;
export const OPENING_PLIES = 4;
export const LAST = 23;
export const PAIRS = { smoke: { N: 1, P: 1 }, match: { N: 200, P: 110 } };
export const KILL_S = { smoke: 1800, match: 21600 };
export const RUN_REL = {
  smoke: 'docs/superpowers/evidence/htb-teacher-bench-smoke',
  match: 'docs/superpowers/evidence/htb-teacher-bench-match',
};
export const TIMING_REL = `${RUN_REL.smoke}/timing.json`;

export class BenchStop extends Error {
  constructor(step, message) {
    super(message);
    this.name = 'BenchStop';
    this.step = step;
  }
}
export class BenchRefusal extends Error {
  constructor(message) {
    super(message);
    this.name = 'BenchRefusal';
  }
}

const sha256 = (s) => createHash('sha256').update(s).digest('hex');
const key = (m) => `${m[0]},${m[1]}`;
const cmpRC = (a, b) => a[0] - b[0] || a[1] - b[1];

// ---------------------------------------------------------------------------------------------------------- openings

/** Opening `index` of the namespace `kind` ('opening' or 'smoke opening'): 4 plies from the central 16x16. */
export function openingAt(kind, index) {
  let s = new TwixtState();
  const moves = [];
  for (let k = 0; k < OPENING_PLIES; k++) {
    const cands = s.legalMoves()
      .filter(([r, c]) => r >= CENTRAL_MIN && r <= CENTRAL_MAX && c >= CENTRAL_MIN && c <= CENTRAL_MAX)
      .map(([r, c]) => [r, c])
      .sort(cmpRC);
    const h = sha256(`${NS}/${kind} ${index}/ply ${k}`);
    const mv = cands[parseInt(h.slice(0, 12), 16) % cands.length];
    moves.push(mv);
    s = s.applyMove(mv);
  }
  return moves;
}

/** The first `n` openings of `kind` that are distinct, also up to the left-right reflection (column c -> 23 - c). */
export function openingList(kind, n) {
  const seen = new Set();
  const out = [];
  for (let i = 0; out.length < n; i++) {
    const moves = openingAt(kind, i);
    const k1 = JSON.stringify(moves);
    const k2 = JSON.stringify(moves.map(([r, c]) => [r, LAST - c]));
    if (seen.has(k1) || seen.has(k2)) continue;
    seen.add(k1);
    out.push({ index: i, moves });
  }
  return out;
}

/** Arm P's student seeds for one move: two 48-bit integers, as in G3. */
export function moveSeeds(stage, pair, game, ply) {
  const h = sha256(`${NS}/${stage}/pair ${pair}/game ${game}/ply ${ply}`);
  return { search: parseInt(h.slice(0, 12), 16), readout: parseInt(h.slice(12, 24), 16) };
}

/** Every scheduled game of a stage, in play order: arm N's pairs, then arm P's (on arm N's first openings). */
export function schedule(stage) {
  const kind = stage === 'smoke' ? 'smoke opening' : 'opening';
  const openings = openingList(kind, Math.max(PAIRS[stage].N, PAIRS[stage].P));
  const out = [];
  for (const arm of ['N', 'P']) {
    for (let p = 0; p < PAIRS[stage][arm]; p++) {
      for (const g of [0, 1]) {
        out.push({ arm, pair: p, game: g, studentColour: g === 0 ? 'red' : 'black', openingIndex: openings[p].index, opening: openings[p].moves });
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------- canonical state

/** The canonical state; htb_bench_teacher.canonical builds the same object, byte for byte, from our Python TwixtState. */
export function canonical(state) {
  const pegs = { red: [], black: [] };
  for (const [k, colour] of state.pegs) pegs[colour].push(k.split(',').map(Number));
  const bridges = { red: [], black: [] };
  for (const b of state.bridges) {
    const [a, c] = b.split('-').map((p) => p.split(',').map(Number)).sort(cmpRC);
    bridges[state.pegs.get(key(a))].push([a, c]);
  }
  const cmpB = (x, y) => cmpRC(x[0], y[0]) || cmpRC(x[1], y[1]);
  for (const c of ['red', 'black']) {
    pegs[c].sort(cmpRC);
    bridges[c].sort(cmpB);
  }
  const legal = state.legalMoves().map(([r, c]) => [r, c]).sort(cmpRC);
  const winner = state.winner();
  return {
    ply: state.ply,
    to_move: state.toMove,
    pegs,
    bridges,
    legal,
    winner,
    terminal: winner !== null || legal.length === 0,
  };
}

export const canonDigest = (state) => sha256(JSON.stringify(canonical(state)));

// ---------------------------------------------------------------------------------------------------------- students

/** Arm N: the mirror-averaged policy argmax over legal moves, ties to the lower (row, col). No search. */
export async function rawPolicyMove(inference, state) {
  const moves = state.legalMoves();
  const { priors } = await inference.evaluate(state.toTensorHWC(), moves);
  let best = null;
  let bestP = -Infinity;
  for (const m of [...moves].sort(cmpRC)) {
    const p = priors.get(key(m));
    if (!Number.isFinite(p)) throw new BenchStop('student', `non-finite prior for ${key(m)} at ply ${state.ply}`);
    if (p > bestP) {
      best = m;
      bestP = p;
    }
  }
  return { moveKey: key(best), prob: bestP };
}

export function makeStudent(arm, inference, policy) {
  if (arm === 'N') {
    return { arm, async move(state) {
      const { moveKey, prob } = await rawPolicyMove(inference, state);
      return { moveKey, extra: { prob } };
    } };
  }
  if (arm === 'P') {
    return { arm, async move(state, { stage, pair, game }) {
      const seeds = moveSeeds(stage, pair, game, state.ply);
      const gameState = TwixtState.fromDict(state.toDict());
      const out = await protocolMove({ gameState, policy, inference, seeds });
      return { moveKey: out.moveKey, extra: { seeds, nSimulations: out.nSimulations } };
    } };
  }
  throw new Error(`unknown arm ${arm}`);
}

// ---------------------------------------------------------------------------------------------------------- teacher

/** The teacher process: JSON lines, one reply per request. */
export class TeacherClient {
  constructor(child) {
    this.child = child;
    this.pending = [];
    this.exited = null;
    const rl = createInterface({ input: child.stdout });
    rl.on('line', (line) => {
      const waiter = this.pending.shift();
      if (!waiter) return;
      try {
        waiter.resolve(JSON.parse(line));
      } catch {
        waiter.reject(new BenchStop('teacher', `unparseable teacher reply: ${line.slice(0, 200)}`));
      }
    });
    child.on('exit', (code, signal) => {
      this.exited = { code, signal };
      for (const w of this.pending.splice(0)) w.reject(new BenchStop('teacher', `teacher process exited (code ${code}, signal ${signal})`));
    });
  }

  static spawn(python, args, { cwd, env }) {
    const child = spawn(python, args, { cwd, env, stdio: ['pipe', 'pipe', 'inherit'] });
    return new TeacherClient(child);
  }

  next() {
    if (this.exited) return Promise.reject(new BenchStop('teacher', `teacher process exited (code ${this.exited.code})`));
    return new Promise((resolve, reject) => this.pending.push({ resolve, reject }));
  }

  request(obj) {
    const p = this.next();
    this.child.stdin.write(`${JSON.stringify(obj)}\n`);
    return p;
  }

  async close() {
    const rep = await this.request({ cmd: 'close' });
    if (!rep.ok) throw new BenchStop('teacher', 'close refused');
    if (!this.exited) await new Promise((resolve) => this.child.once('exit', resolve));
    if (this.exited.code !== 0) throw new BenchStop('teacher', `teacher process exit ${this.exited.code} after close`);
  }
}

function checkReply(rep, state, what) {
  if (!rep || rep.ok !== true) throw new BenchStop(rep?.stop ?? 'teacher', `${what}: ${rep?.detail ?? 'no reply'}`);
  if (rep.ply !== state.ply) throw new BenchStop('canonical', `${what}: teacher at ply ${rep.ply}, driver at ${state.ply}`);
  if (rep.canon !== canonDigest(state)) throw new BenchStop('canonical', `${what}: canonical state differs at ply ${state.ply}`);
}

function applyLegal(state, mv, who) {
  if (!state.legalMoves().some((m) => m[0] === mv[0] && m[1] === mv[1])) {
    throw new BenchStop('illegal', `${who} chose ${key(mv)}, illegal at ply ${state.ply}`);
  }
  return state.applyMove(mv);
}

// ---------------------------------------------------------------------------------------------------------- one game

/** One game under the product's game-end rule. `teacher` has request(obj) (the client, or a test double). */
export async function playGame({ stage, task, student, teacher }) {
  const t0 = performance.now();
  let state = new TwixtState();
  for (const mv of task.opening) state = applyLegal(state, mv, 'opening');
  checkReply(await teacher.request({ cmd: 'start', opening: task.opening }), state, 'start');
  const moves = task.opening.map((m) => [...m]);
  const perMove = [];
  let tbDrawPly = null;
  while (!state.isTerminal()) {
    const ply = state.ply;
    const mover = state.toMove === task.studentColour ? 'student' : 'teacher';
    const m0 = performance.now();
    let mv;
    let extra;
    if (mover === 'student') {
      const out = await student.move(state, { stage, pair: task.pair, game: task.game });
      mv = out.moveKey.split(',').map(Number);
      extra = out.extra;
    } else {
      const rep = await teacher.request({ cmd: 'move' });
      if (!rep?.ok) throw new BenchStop(rep?.stop ?? 'teacher', `move at ply ${ply}: ${rep?.detail ?? 'no reply'}`);
      mv = rep.move;
      extra = { prob: rep.prob, value: rep.value };
    }
    const ms = performance.now() - m0;
    state = applyLegal(state, mv, mover);
    const rep = await teacher.request({ cmd: 'apply', move: mv });
    checkReply(rep, state, `apply at ply ${ply}`);
    if (rep.tb_draw && tbDrawPly === null) tbDrawPly = rep.tb_draw_ply;
    moves.push(mv);
    perMove.push({ ply, mover, move: mv, ms, ...extra });
  }
  const winner = state.winner();
  const result = winner ?? 'draw';
  const studentScore = winner === null ? 0.5 : winner === task.studentColour ? 1 : 0;
  return {
    schema: 'htb-teacher-bench-game/1',
    stage,
    arm: task.arm,
    pair: task.pair,
    game: task.game,
    student_colour: task.studentColour,
    opening_index: task.openingIndex,
    opening: task.opening,
    moves,
    plies: moves.length,
    result,
    student_score: studentScore,
    termination: winner === null ? 'no_legal_moves' : 'connection',
    tb_draw_flag_ply: tbDrawPly,
    wall_ms: performance.now() - t0,
    per_move: perMove,
  };
}

// ---------------------------------------------------------------------------------------------------------- the run

function parseArgs(argv) {
  const [cmd, stage, ...rest] = argv;
  const opts = {};
  for (let i = 0; i < rest.length; i += 2) {
    if (!rest[i]?.startsWith('--') || rest[i + 1] === undefined) return null;
    opts[rest[i].slice(2)] = rest[i + 1];
  }
  return { cmd, stage, opts };
}

let activeTeacher = null; // closed on any stop, so the process can end naturally

async function run(stage, opts) {
  const GATE = await import('./teacher_bench_authorization.mjs');
  const gateName = stage === 'smoke' ? 'BENCH_SMOKE_AUTHORIZED' : 'BENCH_MATCH_AUTHORIZED';
  if (GATE[gateName] !== true) throw new BenchRefusal(`${gateName} is closed`);
  for (const k of ['teacher-python', 'clone-root', 'kill-s']) if (!opts[k]) throw new BenchRefusal(`missing --${k}`);
  if (Number(opts['kill-s']) !== KILL_S[stage]) throw new BenchRefusal(`--kill-s ${opts['kill-s']} is not the card's ${KILL_S[stage]}`);
  if (stage === 'match') {
    const t = existsSync(join(REPO_ROOT, TIMING_REL)) ? JSON.parse(readFileSync(join(REPO_ROOT, TIMING_REL), 'utf8')) : null;
    if (t?.verdict !== 'LAUNCH_OK') throw new BenchRefusal(`${TIMING_REL} is missing or not LAUNCH_OK`);
  }
  const out = join(REPO_ROOT, RUN_REL[stage]);
  if (existsSync(out)) throw new BenchRefusal(`${RUN_REL[stage]} exists (one-shot)`);
  const tasks = schedule(stage);

  const tStart = performance.now();
  const { resolveHardModel } = await import('../../server/model_manifest.js');
  const { AlphaZeroInference } = await import('../../server/inference.js');
  const { MirrorAveragingInference } = await import('../../server/mirror_inference.js');
  const hard = await resolveHardModel({}, REPO_ROOT);
  const modelId = hard.manifest.model_id ?? hard.manifest.modelId ?? hard.manifest.id;
  if (modelId !== STUDENT_MODEL_ID) throw new BenchRefusal(`hard model is ${modelId}, expected ${STUDENT_MODEL_ID}`);
  const policy = resolvePolicy({ difficulty: 'hard', hardMode: 'protocol' });
  if (policy.nSims !== 200 || policy.moveTemp !== 1.0 || policy.search !== 'protocol') throw new BenchRefusal(`unexpected policy ${JSON.stringify(policy)}`);

  mkdirSync(join(out, 'games'), { recursive: true });
  const started = new Date().toISOString();
  const base = new AlphaZeroInference(hard.graphPath);
  await base.load();
  const inference = new MirrorAveragingInference(base);
  const onnxSha = sha256(readFileSync(hard.graphPath));

  const teacher = TeacherClient.spawn(opts['teacher-python'], ['-B', '-m', 'scripts.GPU.alphazero.htb_bench_teacher', 'serve', '--stage', stage, '--clone-root', opts['clone-root']], {
    cwd: REPO_ROOT,
    env: { ...process.env, PYTHONPATH: '.' },
  });
  activeTeacher = teacher;
  const ready = await teacher.next();
  if (!ready?.ok || !ready.ready) throw new BenchStop('teacher', `teacher did not start: ${JSON.stringify(ready).slice(0, 300)}`);
  const startupSeconds = (performance.now() - tStart) / 1000;

  const students = { N: makeStudent('N', inference, policy), P: makeStudent('P', inference, policy) };
  const armSeconds = { N: 0, P: 0 };
  const progress = join(out, 'progress.jsonl');
  let n = 0;
  for (const task of tasks) {
    const rec = await playGame({ stage, task, student: students[task.arm], teacher });
    const name = `${task.arm}_pair_${String(task.pair).padStart(4, '0')}_game_${task.game}.json`;
    writeFileSync(join(out, 'games', name), `${JSON.stringify(rec)}\n`, { flag: 'wx' });
    armSeconds[task.arm] += rec.wall_ms / 1000;
    n += 1;
    // Progress carries no outcome: games done and plies only.
    appendFileSync(progress, `${JSON.stringify({ t: new Date().toISOString(), arm: task.arm, pair: task.pair, game: task.game, plies: rec.plies, done: n, of: tasks.length })}\n`);
  }
  await teacher.close();
  activeTeacher = null;
  const record = {
    schema: 'htb-teacher-bench-record/1',
    written_by: 'script',
    status: 'COMPLETED',
    stage,
    started,
    ended: new Date().toISOString(),
    games: n,
    pairs: PAIRS[stage],
    startup_seconds: startupSeconds,
    arm_seconds: armSeconds,
    total_seconds: (performance.now() - tStart) / 1000,
    student: { model_id: STUDENT_MODEL_ID, onnx_sha256: onnxSha, inference: 'MirrorAveragingInference(AlphaZeroInference)', policy },
    teacher: { identity: ready.identity, pins: ready.teacher, load_seconds: ready.load_seconds },
    kill_s: KILL_S[stage],
    node_version: process.version,
  };
  writeFileSync(join(out, 'record.json'), `${JSON.stringify(record, null, 1)}\n`, { flag: 'wx' });
}

async function main() {
  const a = parseArgs(process.argv.slice(2));
  if (!a || a.cmd !== 'run' || !['smoke', 'match'].includes(a.stage)) {
    console.error('usage: node tests/htb_integration/teacher_bench.mjs run <smoke|match> --teacher-python <py> --clone-root <dir> --kill-s <s>');
    return 2;
  }
  try {
    await run(a.stage, a.opts);
    return 0;
  } catch (err) {
    if (activeTeacher && !activeTeacher.exited) activeTeacher.child.stdin.end(); // its loop then exits 3
    if (err instanceof BenchRefusal) {
      console.error(`REFUSED: ${err.message}`);
      return 4;
    }
    const out = join(REPO_ROOT, RUN_REL[a.stage]);
    const step = err instanceof BenchStop ? err.step : 'crashed';
    console.error(err instanceof BenchStop ? `STOP at ${err.step}: ${err.message}` : (err?.stack ?? String(err)));
    if (existsSync(out) && !existsSync(join(out, 'record.json'))) {
      writeFileSync(join(out, 'record.json'), `${JSON.stringify({ schema: 'htb-teacher-bench-record/1', written_by: 'script', status: 'VOID', stage: a.stage, void: { step, detail: err?.message ?? String(err) } }, null, 1)}\n`, { flag: 'wx' });
    }
    return err instanceof BenchStop ? 3 : 1;
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  // ledger entry 64: never process.exit() with ONNX Runtime loaded; the teacher child is closed or has exited.
  main().then((code) => {
    process.exitCode = code;
  });
}
