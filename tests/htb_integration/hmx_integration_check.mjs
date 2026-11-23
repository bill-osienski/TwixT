#!/usr/bin/env node
/**
 * H-MX integration check driver (proposal docs/superpowers/2026-11-22-htb-mix-integration-proposal.md, revision 1,
 * §5). One gated run against the staged tree (this repository, HEAD = the one-line gate opening on top of the build),
 * the reference tree (today's product, REF_COMMIT) and the revert tree (the build with the swap reverted). It loads
 * models and starts servers; it plays no new game.
 *
 *   G0  the model-loading suites, in the staged tree;
 *   S1  the staged server's banner and /api/model-info name 627c8d774df838a8 (unseeded server);
 *   S2  G2's live REST and WS checks L3-L9, with the hard id as a parameter (unseeded server);
 *   A0  the 845 recorded challenger rows replayed in-process (resolveHardModel, built as server/index.js builds hard);
 *   A1  the same rows over WS, A2 over REST, each against a fresh staged server under the test-only seed preload;
 *   S4  easy and medium search outputs equal to the reference server's on 142 positions; hard is served by 627c;
 *   R1  rollback by restart: a copy of the staged tree without models/627c8d774df838a8, HARD_MODEL_MANIFEST naming
 *       78c16a31c3dba37f: (a) the seeded 142-position capture equals the reference exactly, (b) the 844 recorded
 *       current-hard rows replay exactly over WS;
 *   R2  rollback by revert: the revert tree's server, (a) and (b) as R1, and every B6 and G0 suite there;
 *   T   per-move timing against the match's (informational).
 * Every comparison is exact; one mismatch fails its check. The browser smoke V runs after this driver, by hand.
 *
 * Usage: node tests/htb_integration/hmx_integration_check.mjs run check --ref-root <dir> --revert-root <dir> --kill-s 14400
 * Exit (process.exitCode only; never process.exit with ONNX Runtime loaded, ledger entry 64): 0 COMPLETED (the
 * record holds the verdict), 2 USAGE, 3 VOID, 4 REFUSED.
 */
import { spawn, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import net from 'node:net';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import WebSocket from 'ws';

import { TwixtState } from '../../server/gameLogic.js';
import { protocolMove } from '../../server/protocol_move.js';
import { POLICY_CONFLICT_MESSAGE, resolvePolicy } from '../../server/readout_policy.js';
import { enumerateCases } from '../mcts_golden/cases.mjs';
import { readFixture } from '../mcts_golden/worker.mjs';

export const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
export const NS = 'hmx-integration/20261122';
export const NEW_ID = '627c8d774df838a8';
export const CURRENT_ID = '78c16a31c3dba37f';
export const DEFAULT_ID = 'c34b7ff3297c785a';
export const REF_COMMIT = 'abaf20beddf70e7857b41667c34c4c5a6809f47d';
export const NEW_MANIFEST_SHA256 = 'abf3931d920f788de146cf8dafbd213581f682a754b65df1520d42eae954071d';
export const NEW_GRAPH_SHA256 = '3beaeed15a97f28c69e639b6c5ab2466f961a1aa75f232755e581d592fcb55af';
export const CURRENT_MANIFEST_SHA256 = '81bbf545ec332d5a56541fcb79c95a0ab1b94bb6176118e21fcd73733ccfd0d7';
export const MODEL_MANIFEST_JS_SHA256 = {
  old: 'ecde544b03b4e50f673edaf89f73434c878bfbda0a147341adb9197d788fc511',
  new: '054f9434071d1502482550fb91ed34b57462ba5a29a02ab9cbcc559dab06f3c2',
};
export const MATCH_REL = 'docs/superpowers/evidence/htb-mix-match';
export const MATCH_RECORD_SHA256 = '6bfa8bb6082da5e30b6816728d0a6ba1e6b04ea801fa9e26c85fee9dc6017371';
export const REPLAY_EXTRA_GAME = 'pair_0153_game_1.json';
export const EXPECTED = Object.freeze({ games: 41, challenger: 845, current: 844, positions: 142 });
export const SIMS = 200;
export const KILL_S = 14400;
export const ORT_VERSION = '1.23.2';
export const RUN_REL = 'docs/superpowers/evidence/hmx-integration-check';
export const OUTER_REL = 'docs/superpowers/evidence/hmx-integration-check-run';
export const GATE = 'HMX_INTEGRATION_CHECK_AUTHORIZED';
export const GATE_REL = 'tests/htb_integration/hmx_integration_authorization.mjs';
export const PRELOAD_REL = 'tests/htb_integration/hmx_seed_preload.mjs';
export const RECORD_SCHEMA = 'hmx-integration-check-record/1';
/** The paths the swap commits change; the revert tree differs from the staged tree in exactly these and the gate. */
export const SWAP_FILES = Object.freeze([
  'server/model_manifest.js',
  'server/test_hard_model.js',
  'tests/htb_integration/test_g3.mjs',
  'tests/mcts_golden/cases.mjs',
  'tests/mcts_golden/test_capture.mjs',
  'tests/mcts_golden/test_falsify.mjs',
]);
const MODEL_ENV = ['HARD_MODE', 'HARD_MODEL_MANIFEST', 'MODEL_MANIFEST', 'HMX_SEED_QUEUE', 'NODE_OPTIONS'];

/** The build's model-free commands (proposal §4, B6) and the model-loading suites (§5.3, G0), from a tree's root. */
export function b6Commands(mainPython) {
  return [
    ['server', ['node', '--test', 'server/test_server.js', 'server/test_readout_policy.js', 'server/test_mcts.js',
      'server/test_alpha_zero_client.js', 'server/test_py_random.js', 'server/test_mirror_inference.js',
      'server/test_protocol_mcts.js', 'server/test_protocol_move.js', 'server/test_hard_model.js']],
    ['parity', ['npm', 'run', 'test:parity']],
    ['match', ['node', '--test', 'tests/product_match/test_openings.mjs', 'tests/product_match/test_timing.mjs']],
    ['golden', ['npm', 'run', 'test:golden']],
    ['integration', ['node', '--test', 'tests/htb_integration/test_g3.mjs', 'tests/htb_integration/test_mix_match.mjs',
      'tests/htb_integration/test_onpolicy_match.mjs', 'tests/htb_integration/test_teacher_bench.mjs',
      'tests/htb_integration/test_diag_o.mjs', 'tests/htb_integration/test_hmx_integration.mjs']],
    ['smoke', ['npm', 'test']],
    ['lint', ['npm', 'run', 'lint']],
    ['wrapper', [mainPython, '-m', 'pytest', '-q', 'tests/test_hmx_integration_wrapper.py']],
  ];
}
export const G0_COMMANDS = Object.freeze([
  ['g0-server', ['node', '--test', 'server/test_model_manifest.js', 'server/test_ort_conv_canary.js', 'server/test_ort_shutdown.js']],
  ['g0-match', ['node', '--test', 'tests/product_match/test_harness.mjs', 'tests/product_match/test_analyse.mjs']],
]);
/**
 * Tests that fail in any LINKED worktree, because they spawn `<tree>/.venv/bin/python` and a linked worktree has no
 * .venv (build record, B6 baseline at REF_COMMIT). Accepted in the revert tree's integration suite only as exactly
 * this set; any other failure, or a missing one, fails the suite check.
 */
export const LINKED_WORKTREE_FAILURES = Object.freeze([
  'the wrappers: their derivation, ceilings, gate names, driver and the pinned analyser sha',
  'canonical state: Node and our Python TwixtState give the same digest at every ply (G3 draw and a decisive game)',
  "end to end through the real serve loop: TeacherClient, Python lockstep on twixtbot's Game, a fake network",
]);

export class CheckRefusal extends Error {}
export class CheckStop extends Error {
  constructor(step, message) {
    super(message);
    this.step = step;
  }
}

const sha256 = (b) => createHash('sha256').update(b).digest('hex');
const fileSha = (p) => sha256(readFileSync(p));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const gitIn = (root, ...args) => spawnSync('git', args, { cwd: root, encoding: 'utf8' });

// ------------------------------------------------------------------------------------------------- the samples

/** The replayed match games: both games of every pair p = 0 (mod 20), plus the longest game. Fixed before any run. */
export function selectGames(root = REPO_ROOT) {
  const dir = join(root, MATCH_REL, 'games');
  return readdirSync(dir)
    .filter((n) => {
      const m = /^pair_(\d{4})_game_[01]\.json$/.exec(n);
      return m && (Number(m[1]) % 20 === 0 || n === REPLAY_EXTRA_GAME);
    })
    .sort();
}

/** One row per recorded move by `mover`, in game then ply order, each with the position it was played from. */
export function replayRows(mover, root = REPO_ROOT) {
  const rows = [];
  for (const name of selectGames(root)) {
    const g = JSON.parse(readFileSync(join(root, MATCH_REL, 'games', name), 'utf8'));
    for (const r of g.per_move) {
      if (r.mover !== mover) continue;
      const moves = g.moves.slice(0, r.ply);
      const state = TwixtState.fromMoves(moves);
      const colour = mover === 'challenger' ? g.challenger_colour : g.challenger_colour === 'red' ? 'black' : 'red';
      if (state.ply !== r.ply || state.toMove !== colour || !same(g.moves[r.ply], r.move)) {
        throw new CheckRefusal(`${name} ply ${r.ply}: the recorded row does not reconstruct`);
      }
      rows.push({ game: name, ply: r.ply, moves, seeds: r.seeds, move: r.move, nSimulations: r.nSimulations, visits: r.visits, ms: r.ms });
    }
  }
  return rows;
}

/** The capture positions: G2's 126 non-terminal parity-corpus positions and the 16 golden positions. */
export function capturePositions(root = REPO_ROOT) {
  const corpus = JSON.parse(readFileSync(join(root, 'tests/parity/corpus.json'), 'utf8'));
  const out = [...corpus.primary, ...corpus.edge]
    .map((p) => ({ id: p.id, state: TwixtState.fromMoves(p.moves) }))
    .filter((p) => !p.state.isTerminal());
  const seen = new Set();
  for (const c of enumerateCases()) {
    if (!c.position || seen.has(c.position.id)) continue;
    seen.add(c.position.id);
    out.push({ id: `golden_${c.position.id}`, state: readFixture(c).state });
  }
  return out;
}

/** The hard seeds for capture position `id` (proposal §5.5): the first 12 hex digits of each namespaced sha256. */
export function positionSeeds(id) {
  const h = (k) => parseInt(sha256(`${NS}/${id}/${k}`).slice(0, 12), 16);
  return { search: h('search'), readout: h('readout') };
}

/** The flat queue a seeded server consumes: search then readout, per request, in order. */
export const seedQueue = (seedPairs) => seedPairs.flatMap((s) => [s.search, s.readout]);

/** Every way the preload's stderr log differs from consuming exactly `queue`, in order. */
export function seedLogProblems(output, queue) {
  const used = [];
  for (const line of output.split('\n')) {
    const m = /^HMX_SEED (\d+) (\d+)$/.exec(line.trim());
    if (m) used.push([Number(m[1]), Number(m[2])]);
  }
  const p = [];
  if (used.length !== queue.length) p.push(`consumed ${used.length} seeds, expected ${queue.length}`);
  for (let i = 0; i < Math.min(used.length, queue.length); i++) {
    if (used[i][0] !== i || used[i][1] !== queue[i]) {
      p.push(`seed ${i}: consumed #${used[i][0]} = ${used[i][1]}, expected ${queue[i]}`);
      break;
    }
  }
  return p;
}

// ------------------------------------------------------------------------------------------------- comparators

const HARD_FIELDS = ['type', 'keys', 'move', 'visits', 'rootValue', 'moveValue', 'moveVisits', 'valueRed', 'valueKind', 'nSimulations', 'search', 'modelId'];
const PLAIN_FIELDS = ['type', 'keys', 'visits', 'rootValue', 'valueRed', 'nSimulations'];

/**
 * Compare a capture with the reference capture. Easy and medium: every PLAIN_FIELD equal. Hard: under 'exact' every
 * HARD_FIELD equal; under 'served' the subject's hard is the protocol at 200 simulations with `hardId`, visits 200.
 */
export function compareCaptures(ref, sub, rule, hardId) {
  const failures = [];
  if (!ref || !sub || ref.refused || sub.refused) return { verdict: 'FAIL', failures: ['a capture did not serve'], hardOk: 0, plainOk: 0 };
  const refPos = ref.positions ?? [];
  const subById = new Map((sub.positions ?? []).map((p) => [p.id, p]));
  if (refPos.length !== EXPECTED.positions || (sub.positions ?? []).length !== EXPECTED.positions) failures.push('position count');
  let hardOk = 0;
  let plainOk = 0;
  for (const r of refPos) {
    const s = subById.get(r.id);
    if (!s) {
      failures.push(`missing ${r.id}`);
      continue;
    }
    for (const d of ['easy', 'medium']) {
      const bad = PLAIN_FIELDS.filter((k) => !same(r[d]?.[k], s[d]?.[k]));
      if (bad.length || r[d]?.type !== 'bestmove') failures.push(`${r.id} ${d}: ${bad.join(',') || 'not a bestmove'}`);
      else plainOk += 1;
    }
    if (rule === 'exact') {
      const bad = HARD_FIELDS.filter((k) => !same(r.hard?.[k], s.hard?.[k]));
      if (bad.length || r.hard?.type !== 'bestmove') failures.push(`${r.id} hard: ${bad.join(',') || 'not a bestmove'}`);
      else hardOk += 1;
    } else if (rule === 'served') {
      const h = s.hard ?? {};
      const total = h.visits ? Object.values(h.visits).reduce((a, b) => a + b, 0) : null;
      if (h.type === 'bestmove' && h.search === 'protocol' && h.modelId === hardId && h.nSimulations === SIMS && total === SIMS) hardOk += 1;
      else failures.push(`${r.id} hard is not the protocol with ${hardId}`);
    } else failures.push(`unknown rule ${rule}`);
  }
  return { verdict: failures.length ? 'FAIL' : 'PASS', failures: failures.slice(0, 50), failureCount: failures.length, hardOk, plainOk };
}

/**
 * Compare replay answers with the recorded rows, exactly: the move, the model, the protocol, and the budget (the
 * root visits total; nSimulations too where the transport reports it, which REST does not).
 */
export function compareReplay(rows, answers, hardId) {
  const failures = [];
  if (answers.length !== rows.length) failures.push(`${answers.length} answers for ${rows.length} rows`);
  let equal = 0;
  for (let i = 0; i < Math.min(rows.length, answers.length); i++) {
    const r = rows[i];
    const a = answers[i];
    const bad = [];
    if (!same(a.move, r.move)) bad.push('move');
    if (a.visitsTotal !== SIMS || r.visits !== SIMS) bad.push('visits');
    if (a.nSimulations !== undefined && a.nSimulations !== SIMS) bad.push('nSimulations');
    if (r.nSimulations !== SIMS) bad.push('recorded nSimulations');
    if (a.search !== 'protocol') bad.push('search');
    if (a.modelId !== hardId) bad.push('modelId');
    if (bad.length) failures.push(`${r.game} ply ${r.ply}: ${bad.join(',')}${a.error ? ` (${a.error})` : ''}`);
    else equal += 1;
  }
  return { verdict: failures.length ? 'FAIL' : 'PASS', rows: rows.length, equal, failures: failures.slice(0, 50), failureCount: failures.length };
}

// ------------------------------------------------------------------------------------------------- suites

/** Counts and failing test names from node --test's default (spec) reporter output. */
export function parseNodeTest(text) {
  const num = (k) => {
    const m = new RegExp(`^ℹ ${k} (\\d+)$`, 'm').exec(text);
    return m ? Number(m[1]) : null;
  };
  const failing = [];
  const i = text.indexOf('✖ failing tests:');
  if (i >= 0) {
    for (const line of text.slice(i).split('\n').slice(1)) {
      const m = /^✖ (.*) \([\d.]+m?s\)$/.exec(line);
      if (m) failing.push(m[1]);
    }
  }
  return { tests: num('tests'), pass: num('pass'), fail: num('fail'), skipped: num('skipped'), cancelled: num('cancelled'), failing };
}

/** Counts from pytest -q's summary line. */
export function parsePytest(text) {
  const line = text.trim().split('\n').pop() ?? '';
  const n = (k) => Number((new RegExp(`(\\d+) ${k}`).exec(line) ?? [0, 0])[1]);
  return { pass: n('passed'), fail: n('failed') + n('error'), skipped: n('skipped'), summary: line };
}

/**
 * One suite's verdict. PASS when it exits 0 with no failure; or, where `allowedFailures` is given (the revert tree's
 * integration suite), when exactly that set of tests fails and nothing else.
 */
export function suiteVerdict({ label, code, text, allowedFailures = null }) {
  const counts = label === 'wrapper' ? parsePytest(text) : label === 'lint' ? {} : parseNodeTest(text);
  let ok = code === 0 && (counts.fail ?? 0) === 0;
  let note = null;
  if (!ok && allowedFailures) {
    const got = [...new Set(counts.failing ?? [])].sort();
    ok = counts.fail === allowedFailures.length && same(got, [...allowedFailures].sort()) && counts.cancelled === 0;
    note = ok ? 'only the linked-worktree .venv failures' : `failing: ${JSON.stringify(got)}`;
  }
  return { label, code, verdict: ok ? 'PASS' : 'FAIL', counts, note };
}

function runSuites(root, commands, outDir, prefix, allowed = {}) {
  const results = [];
  const env = { ...process.env };
  for (const k of MODEL_ENV) delete env[k];
  for (const [label, argv] of commands) {
    const r = spawnSync(argv[0], argv.slice(1), { cwd: root, env, encoding: 'utf8', maxBuffer: 256 * 1024 * 1024 });
    const text = `${r.stdout ?? ''}${r.stderr ?? ''}`;
    writeFileSync(join(outDir, `${prefix}${label}.log`), text, { flag: 'wx' });
    results.push(suiteVerdict({ label, code: r.status, text, allowedFailures: allowed[label] ?? null }));
  }
  return { verdict: results.every((x) => x.verdict === 'PASS') ? 'PASS' : 'FAIL', suites: results };
}

// ------------------------------------------------------------------------------------------------- servers

const freePort = () =>
  new Promise((res) => {
    const s = net.createServer().listen(0, '127.0.0.1', () => {
      const { port } = s.address();
      s.close(() => res(port));
    });
  });

/**
 * Start `<root>/server/index.js` on a free port: model environment cleared, then `env` applied; with the seed
 * preload when `queueFile` is given. Its output goes to `logPath` when it stops.
 */
export async function startServer({ root, env = {}, queueFile = null, logPath }) {
  const port = await freePort();
  const e = { ...process.env, PORT: String(port) };
  for (const k of MODEL_ENV) delete e[k];
  Object.assign(e, env);
  const args = [];
  if (queueFile) {
    e.HMX_SEED_QUEUE = queueFile;
    args.push('--import', join(REPO_ROOT, PRELOAD_REL));
  }
  args.push(join(root, 'server/index.js'));
  const proc = spawn(process.execPath, args, { env: e, cwd: root });
  let output = '';
  let exited = null;
  proc.stdout.on('data', (d) => (output += d));
  proc.stderr.on('data', (d) => (output += d));
  const done = new Promise((r) => proc.on('exit', (code, sig) => r((exited = code ?? sig))));
  const base = `http://127.0.0.1:${port}`;
  let up = false;
  for (let i = 0; i < 600 && exited === null; i++) {
    try {
      if ((await fetch(`${base}/api/health`)).ok) {
        up = true;
        break;
      }
    } catch {
      /* not yet */
    }
    await sleep(100);
  }
  const stop = async () => {
    if (exited === null) {
      proc.kill('SIGTERM');
      const t = setTimeout(() => proc.kill('SIGKILL'), 10000);
      await done;
      clearTimeout(t);
    }
    if (logPath) writeFileSync(logPath, output, { flag: 'wx' });
    return output;
  };
  return { port, base, up, args: [process.execPath, ...args], output: () => output, stop };
}

const bannerLines = (text) => text.split('\n').filter((l) => /Model id|Hard|Manifest|HMX_SEED_PRELOAD/.test(l));

/** A WS session that sends one request at a time and returns its final (non-progress) answer. */
async function wsSession(port) {
  const ws = new WebSocket(`ws://127.0.0.1:${port}/ws`);
  const finals = new Map();
  const progress = new Map();
  ws.on('message', (raw) => {
    const m = JSON.parse(raw.toString());
    if (m.type === 'progress') progress.set(m.id, (progress.get(m.id) ?? 0) + 1);
    else finals.set(m.id, m);
  });
  await new Promise((r, j) => {
    ws.on('open', r);
    ws.on('error', j);
  });
  return {
    async request(msg, ms = 180000) {
      ws.send(JSON.stringify(msg));
      const t0 = Date.now();
      while (!finals.has(msg.id)) {
        if (Date.now() - t0 > ms) return { type: 'timeout', id: msg.id };
        await sleep(2);
      }
      return { ...finals.get(msg.id), progressMessages: progress.get(msg.id) ?? 0 };
    },
    close: () => ws.close(),
  };
}

const visitsTotal = (v) => (v ? Object.values(v).reduce((a, b) => a + b, 0) : null);
const asMove = (m) => (m ? [m.row, m.col] : null);

/** The seeded capture (proposal §5.5): per position, hard then easy then medium, over WS with includeVisits. */
async function capture(server, positions) {
  const s = await wsSession(server.port);
  const out = [];
  for (const p of positions) {
    const row = { id: p.id, ply: p.state.ply, toMove: p.state.toMove };
    for (const d of ['hard', 'easy', 'medium']) {
      const m = await s.request({ type: 'move', id: `${p.id}:${d}`, state: p.state.toDict(), difficulty: d, includeVisits: true });
      const keys = Object.keys(m).filter((k) => k !== 'progressMessages').sort();
      row[d] = { type: m.type, message: m.message, keys, move: m.move, visits: m.visits, rootValue: m.rootValue, moveValue: m.moveValue,
        moveVisits: m.moveVisits, valueRed: m.valueRed, valueKind: m.valueKind, nSimulations: m.nSimulations, search: m.search, modelId: m.modelId };
    }
    out.push(row);
  }
  s.close();
  return out;
}

async function replayWs(server, rows) {
  const s = await wsSession(server.port);
  const out = [];
  for (const [i, r] of rows.entries()) {
    const m = await s.request({ type: 'move', id: `r${i}`, state: TwixtState.fromMoves(r.moves).toDict(), difficulty: 'hard', includeVisits: true });
    out.push({ move: asMove(m.move), visitsTotal: visitsTotal(m.visits), nSimulations: m.nSimulations, search: m.search, modelId: m.modelId,
      elapsed: m.elapsed, error: m.type === 'bestmove' ? undefined : `${m.type}: ${m.message}` });
  }
  s.close();
  return out;
}

async function replayRest(server, rows) {
  const out = [];
  for (const r of rows) {
    const res = await fetch(`${server.base}/api/move`, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ state: TwixtState.fromMoves(r.moves).toDict(), difficulty: 'hard' }) });
    const b = await res.json();
    out.push({ move: asMove(b.move), visitsTotal: visitsTotal(b.visits), search: b.search, modelId: b.modelId, cached: b.cached,
      elapsed: b.elapsed, error: res.status === 200 ? undefined : `${res.status}: ${b.error}` });
  }
  return out;
}

async function replayInProcess(inference, rows) {
  const policy = resolvePolicy({ difficulty: 'hard', hardMode: 'protocol' });
  const out = [];
  for (const r of rows) {
    // As the transports do: the position arrives as a dict (server/index.js TwixtState.fromDict), as in the match.
    const gameState = TwixtState.fromDict(TwixtState.fromMoves(r.moves).toDict());
    const o = await protocolMove({ gameState, policy, inference, seeds: r.seeds });
    let total = 0;
    for (const v of o.visitCounts.values()) total += v;
    out.push({ move: o.moveKey.split(',').map(Number), visitsTotal: total, nSimulations: o.nSimulations, search: 'protocol', modelId: NEW_ID });
  }
  return out;
}

// ------------------------------------------------------------------------------------------------- S1, S2 (G2's L1-L9)

/** G2's live check (tests/htb_integration/live_check.mjs), with the expected hard id as a parameter. Unseeded. */
export async function liveCheck(server, hardId) {
  const checks = [];
  const check = (id, ok, detail) => checks.push({ id, ok: Boolean(ok), detail });
  const banner = server.output();
  const corpus = JSON.parse(readFileSync(join(REPO_ROOT, 'tests/parity/corpus.json'), 'utf8'));
  const all = [...corpus.primary, ...corpus.edge];
  const pick = (pred) => all.find(pred);
  const POS = [
    { label: 'empty board', moves: [] },
    pick((p) => p.ply >= 2 && p.ply < 20 && p.to_move === 'red'),
    pick((p) => p.ply >= 2 && p.ply < 20 && p.to_move === 'black'),
    pick((p) => p.ply >= 20 && p.ply < 50 && p.to_move === 'red'),
    pick((p) => p.ply >= 20 && p.ply < 50 && p.to_move === 'black'),
    pick((p) => p.ply >= 100 && p.to_move === 'red'),
    pick((p) => p.ply >= 100 && p.to_move === 'black'),
  ].map((p) => ({ label: p.label ?? p.id, state: TwixtState.fromMoves(p.moves) }));
  const { base } = server;
  const post = async (path, body) => {
    const r = await fetch(`${base}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
    return { status: r.status, body: await r.json() };
  };
  const legal = (state, move) => state.legalMoves().some(([r, c]) => r === move?.row && c === move?.col);
  const clamp = (v) => Math.max(-1, Math.min(1, v));
  const ws = new WebSocket(`ws://127.0.0.1:${server.port}/ws`);
  const msgs = [];
  ws.on('message', (raw) => msgs.push(JSON.parse(raw.toString())));
  await new Promise((r) => ws.on('open', r));
  const send = (o) => ws.send(JSON.stringify(o));
  const waitFor = async (pred, ms = 60000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      const m = msgs.find(pred);
      if (m) return m;
      await sleep(10);
    }
    return null;
  };
  let info = null;
  try {
    check('L1 banner', /Hard mode:\s+protocol/.test(banner) && new RegExp(`Hard model id:\\s+${hardId}`).test(banner) && new RegExp(`Model id:\\s+${DEFAULT_ID}`).test(banner), bannerLines(banner));
    info = await (await fetch(`${base}/api/model-info`)).json();
    check('L2 model-info', info.hard_mode === 'protocol' && info.hard_model_id === hardId && info.model_id === DEFAULT_ID, { hard_mode: info.hard_mode, hard_model_id: info.hard_model_id, model_id: info.model_id });
    const timings = [];
    for (const [i, p] of POS.entries()) {
      const id = `h${i}`;
      send({ type: 'move', id, state: p.state.toDict(), difficulty: 'hard', includeVisits: true });
      const m = await waitFor((x) => x.id === id && (x.type === 'bestmove' || x.type === 'error'), 120000);
      const prog = msgs.filter((x) => x.id === id && x.type === 'progress');
      const ply = p.state.ply;
      const total = visitsTotal(m?.visits);
      const display = m?.valueKind === 'played_move_mean' ? m.moveValue : m?.rootValue;
      const expectedRed = m ? clamp(p.state.toMove === 'red' ? display : -display) : null;
      const ok = m?.type === 'bestmove' && m.search === 'protocol' && m.modelId === hardId && m.nSimulations === SIMS &&
        legal(p.state, m.move) && total === SIMS && m.moveVisits === m.visits[`${m.move.row},${m.move.col}`] &&
        m.valueKind === (ply < 20 ? 'root_mean' : 'played_move_mean') && m.evalToMove === p.state.toMove &&
        m.toMove === p.state.toMove && m.valueRed === expectedRed && Number.isFinite(m.rootValue) && Math.abs(m.rootValue) <= 1 &&
        (m.valueKind === 'root_mean' || (m.moveVisits > 0 && Number.isFinite(m.moveValue))) &&
        prog.length > 0 && prog.every((x) => x.total === SIMS && x.toMove === p.state.toMove);
      check(`L3 WS hard ${p.label} (ply ${ply}, ${p.state.toMove})`, ok, { move: m?.move, valueKind: m?.valueKind, visitsTotal: total, progressMessages: prog.length, elapsed: m?.elapsed, type: m?.type, error: m?.message });
      timings.push({ ply, elapsedMs: m?.elapsed });
    }
    const s0 = (await (await fetch(`${base}/api/stats`)).json()).cache;
    const p4 = POS[4];
    const r4 = await post('/api/move', { state: p4.state.toDict(), difficulty: 'hard' });
    const s1 = (await (await fetch(`${base}/api/stats`)).json()).cache;
    const restOk = r4.status === 200 && r4.body.cached === false && r4.body.search === 'protocol' && r4.body.modelId === hardId &&
      legal(p4.state, r4.body.move) && r4.body.valueKind === 'played_move_mean' && Number.isFinite(r4.body.value) && visitsTotal(r4.body.visits) === SIMS;
    check('L4 REST hard, cache untouched', restOk && same(s0, s1), { status: r4.status, cached: r4.body.cached, statsBefore: s0, statsAfter: s1 });
    const m1 = await post('/api/move', { state: POS[1].state.toDict(), difficulty: 'medium' });
    const m2 = await post('/api/move', { state: POS[1].state.toDict(), difficulty: 'medium' });
    const s2 = (await (await fetch(`${base}/api/stats`)).json()).cache;
    check('L4 control: REST medium is cached', m1.body.cached === false && m2.body.cached === true && s2.hits === s1.hits + 1 && !('search' in m2.body), { first: m1.body.cached, second: m2.body.cached });
    const sd = POS[3].state.toDict();
    const o1 = await post('/api/move', { state: sd, difficulty: 'hard', temperature: 0.5 });
    const o2 = await post('/api/move', { state: sd, difficulty: 'hard', deterministicMode: true });
    const o3 = await post('/api/move', { state: sd, difficulty: 'hard', deterministicMode: false });
    send({ type: 'move', id: 'o4', state: sd, difficulty: 'hard', temperature: 0.5 });
    const o4 = await waitFor((x) => x.id === 'o4' && x.type !== 'progress', 5000);
    send({ type: 'move', id: 'o5', state: sd, difficulty: 'hard', deterministicMode: true });
    const o5 = await waitFor((x) => x.id === 'o5' && x.type !== 'progress', 5000);
    check('L5 overrides refused (REST 400, WS error, one message); deterministicMode:false accepted',
      o1.status === 400 && o1.body.error === POLICY_CONFLICT_MESSAGE && o1.body.code === 'POLICY_CONFLICT' &&
      o2.status === 400 && o2.body.code === 'POLICY_CONFLICT' && o3.status === 200 && o3.body.search === 'protocol' && o3.body.modelId === hardId &&
      o4?.type === 'error' && o4.message === POLICY_CONFLICT_MESSAGE && o5?.type === 'error' && o5.message === POLICY_CONFLICT_MESSAGE,
      { rest: [o1.status, o2.status, o3.status], ws: [o4?.type, o5?.type] });
    const chain = [];
    const red = Array.from({ length: 12 }, (_, k) => [2 * k, (k % 2) ? 6 : 5]);
    const black = [...Array.from({ length: 11 }, (_, k) => [2 * k + 1, 19]), [4, 15]];
    for (let k = 0; k < 12; k++) chain.push(red[k], black[k]);
    const won = TwixtState.fromMoves(chain).applyMove([23, 4]);
    const small = new TwixtState({ boardSize: 12 });
    const f1 = await post('/api/move', { state: won.toDict(), difficulty: 'hard' });
    const f2 = await post('/api/move', { state: small.toDict(), difficulty: 'hard' });
    send({ type: 'move', id: 'f3', state: won.toDict(), difficulty: 'hard' });
    const f3 = await waitFor((x) => x.id === 'f3' && x.type !== 'progress', 5000);
    send({ type: 'move', id: 'f4', state: small.toDict(), difficulty: 'hard' });
    const f4 = await waitFor((x) => x.id === 'f4' && x.type !== 'progress', 5000);
    check('L6 won position and 12x12 board refused on both transports', won.winner() === 'red' &&
      f1.status === 400 && f1.body.code === 'TERMINAL_ROOT' && f2.status === 400 && f2.body.code === 'NOT_24' &&
      f3?.type === 'error' && f3.message === f1.body.error && f4?.type === 'error' && f4.message === f2.body.error,
      { rest: [f1.status, f1.body.code, f2.status, f2.body.code] });
    const ply30 = POS[3].state.toDict();
    send({ type: 'move', id: 'k1', state: ply30, difficulty: 'hard' });
    await sleep(50);
    send({ type: 'cancel', id: 'k1' });
    await sleep(Math.max(3000, 3 * (timings[3]?.elapsedMs ?? 1000)));
    const k1 = msgs.filter((x) => x.id === 'k1' && x.type !== 'progress');
    send({ type: 'move', id: 'k2', state: ply30, difficulty: 'hard' });
    await sleep(30);
    send({ type: 'move', id: 'k3', state: ply30, difficulty: 'hard' });
    const k3 = await waitFor((x) => x.id === 'k3' && x.type === 'bestmove', 60000);
    await sleep(Math.max(2000, 2 * (timings[3]?.elapsedMs ?? 1000)));
    const k2 = msgs.filter((x) => x.id === 'k2' && x.type !== 'progress');
    check('L7 cancel answers nothing; a newer move aborts the older; the server keeps serving',
      k1.length === 0 && k2.length === 0 && k3?.search === 'protocol', { cancelledAnswers: k1.length, supersededAnswers: k2.length, latest: k3?.type });
    const KEYS = ['elapsed', 'evalToMove', 'id', 'move', 'nSimulations', 'rootValue', 'toMove', 'type', 'valueRed'];
    for (const [d, n] of [['easy', 100], ['medium', 400]]) {
      send({ type: 'move', id: `p_${d}`, state: POS[3].state.toDict(), difficulty: d });
      const m = await waitFor((x) => x.id === `p_${d}` && x.type === 'bestmove', 60000);
      check(`L8 ${d} WS response keeps today's key set`, m && same(Object.keys(m).sort(), KEYS) && m.nSimulations === n, { keys: m && Object.keys(m).sort() });
    }
    check('L9 timing recorded (informational)', true, { hardMoves: timings });
  } catch (err) {
    check('harness error', false, String(err?.stack ?? err));
  } finally {
    ws.close();
  }
  const s1 = checks.filter((c) => c.id.startsWith('L1') || c.id.startsWith('L2'));
  const s2 = checks.filter((c) => !(c.id.startsWith('L1') || c.id.startsWith('L2')));
  return {
    S1: { verdict: s1.length === 2 && s1.every((c) => c.ok) ? 'PASS' : 'FAIL', checks: s1, banner: bannerLines(banner), modelInfo: info },
    S2: { verdict: s2.length > 0 && s2.every((c) => c.ok) ? 'PASS' : 'FAIL', checks: s2 },
  };
}

// ------------------------------------------------------------------------------------------------- trees

const ortVersionAt = (root) => createRequire(join(root, 'server', 'inference.js'))('onnxruntime-node/package.json').version;

function treeProblems(root, label) {
  const p = [];
  if (!existsSync(join(root, 'server/index.js'))) return [`${label} ${root} has no server/index.js`];
  try {
    const st = lstatSync(join(root, 'node_modules'));
    if (!st.isDirectory() || st.isSymbolicLink()) p.push(`${label}: node_modules is not a real directory`);
  } catch {
    p.push(`${label}: node_modules is missing`);
  }
  try {
    if (ortVersionAt(root) !== ORT_VERSION) p.push(`${label}: onnxruntime-node ${ortVersionAt(root)}, not ${ORT_VERSION}`);
    const req = createRequire(join(root, 'server', 'inference.js'));
    if (!req.resolve('onnxruntime-node/package.json').startsWith(join(resolve(root), 'node_modules'))) p.push(`${label}: onnxruntime-node does not resolve from its own node_modules`);
  } catch (e) {
    p.push(`${label}: cannot resolve onnxruntime-node (${e.message})`);
  }
  const st = gitIn(root, 'status', '--porcelain');
  if (st.status !== 0 || st.stdout.trim() !== '') p.push(`${label}: the tree is not clean`);
  return p;
}

/** The reference tree is today's product at REF_COMMIT, clean, with its own node_modules. */
export function refTreeProblems(refRoot) {
  const p = treeProblems(refRoot, 'reference');
  if (gitIn(refRoot, 'rev-parse', 'HEAD').stdout.trim() !== REF_COMMIT) p.push(`reference HEAD is not ${REF_COMMIT}`);
  if (existsSync(join(refRoot, 'server/model_manifest.js')) && fileSha(join(refRoot, 'server/model_manifest.js')) !== MODEL_MANIFEST_JS_SHA256.old) p.push('reference server/model_manifest.js is not today\'s');
  return p;
}

/** The revert tree differs from the staged HEAD in exactly the swap files and the gate, with today's HARD_MODEL_ID. */
export function revertTreeProblems(revertRoot, stagedRoot = REPO_ROOT) {
  const p = treeProblems(revertRoot, 'revert');
  const rh = gitIn(revertRoot, 'rev-parse', 'HEAD').stdout.trim();
  const sh = gitIn(stagedRoot, 'rev-parse', 'HEAD').stdout.trim();
  const d = gitIn(stagedRoot, 'diff', '--name-only', rh, sh);
  const changed = d.status === 0 ? d.stdout.trim().split('\n').filter(Boolean).sort() : null;
  const want = [...SWAP_FILES, GATE_REL].sort();
  if (!same(changed, want)) p.push(`revert HEAD differs from the staged HEAD in ${JSON.stringify(changed)}, expected ${JSON.stringify(want)}`);
  if (existsSync(join(revertRoot, 'server/model_manifest.js')) && fileSha(join(revertRoot, 'server/model_manifest.js')) !== MODEL_MANIFEST_JS_SHA256.old) p.push('revert server/model_manifest.js is not today\'s');
  return p;
}

/** The staged tree serves the new pin from the pinned manifest. */
export function stagedTreeProblems(root = REPO_ROOT) {
  const p = [];
  if (fileSha(join(root, 'server/model_manifest.js')) !== MODEL_MANIFEST_JS_SHA256.new) p.push('staged server/model_manifest.js is not the swap');
  if (fileSha(join(root, 'models', NEW_ID, 'manifest.json')) !== NEW_MANIFEST_SHA256) p.push('staged new manifest is not the pinned bytes');
  if (fileSha(join(root, 'models', CURRENT_ID, 'manifest.json')) !== CURRENT_MANIFEST_SHA256) p.push('staged rollback manifest is not the pinned bytes');
  if (fileSha(join(root, MATCH_REL, 'record.json')) !== MATCH_RECORD_SHA256) p.push('the match record is not the committed one');
  return p;
}

// ------------------------------------------------------------------------------------------------- the record

export const CHECK_IDS = Object.freeze(['G0', 'S1', 'S2', 'A0', 'A1', 'A2', 'S4', 'R1', 'R2']);

/** Every way a record fails to be a complete, self-consistent COMPLETED record; empty when it is. */
export function recordProblems(rec) {
  const p = [];
  if (rec?.schema !== RECORD_SCHEMA) p.push('schema');
  if (rec?.status !== 'COMPLETED') p.push('status');
  for (const id of CHECK_IDS) if (!['PASS', 'FAIL'].includes(rec?.checks?.[id]?.verdict)) p.push(`check ${id} has no verdict`);
  const all = CHECK_IDS.every((id) => rec?.checks?.[id]?.verdict === 'PASS');
  if (rec?.verdict !== (all ? 'PASS' : 'FAIL')) p.push('verdict does not follow the checks');
  if (rec?.staged?.hard_model_id !== NEW_ID) p.push('staged hard model');
  if (rec?.kill_s !== KILL_S) p.push('kill_s');
  return p;
}

// ------------------------------------------------------------------------------------------------- the run

function parseArgs(argv) {
  const [cmd, stage, ...rest] = argv;
  const opts = {};
  for (let i = 0; i < rest.length; i += 2) {
    if (!rest[i]?.startsWith('--') || rest[i + 1] === undefined) return null;
    opts[rest[i].slice(2)] = rest[i + 1];
  }
  return { cmd, stage, opts };
}

function mainPython() {
  const common = gitIn(REPO_ROOT, 'rev-parse', '--path-format=absolute', '--git-common-dir').stdout.trim();
  return join(dirname(common), '.venv', 'bin', 'python');
}

async function run(opts) {
  const GATES = await import('./hmx_integration_authorization.mjs');
  if (GATES[GATE] !== true) throw new CheckRefusal(`${GATE} is closed`);
  for (const k of ['ref-root', 'revert-root', 'kill-s']) if (!opts[k]) throw new CheckRefusal(`missing --${k}`);
  if (Number(opts['kill-s']) !== KILL_S) throw new CheckRefusal(`--kill-s ${opts['kill-s']} is not the proposal's ${KILL_S}`);
  for (const k of MODEL_ENV) if (process.env[k] !== undefined) throw new CheckRefusal(`${k} is set in the driver's environment`);
  const out = join(REPO_ROOT, RUN_REL);
  if (existsSync(out)) throw new CheckRefusal(`${RUN_REL} exists (one-shot)`);
  if (ortVersionAt(REPO_ROOT) !== ORT_VERSION) throw new CheckRefusal(`onnxruntime-node ${ortVersionAt(REPO_ROOT)}, the proposal pins ${ORT_VERSION}`);
  const refRoot = resolve(opts['ref-root']);
  const revertRoot = resolve(opts['revert-root']);
  const problems = [...stagedTreeProblems(), ...refTreeProblems(refRoot), ...revertTreeProblems(revertRoot)];
  if (problems.length) throw new CheckRefusal(problems.join('; '));
  const py = mainPython();
  if (!existsSync(py)) throw new CheckRefusal(`${py} is missing`);
  const challengerRows = replayRows('challenger');
  const currentRows = replayRows('current');
  const positions = capturePositions();
  if (selectGames().length !== EXPECTED.games || challengerRows.length !== EXPECTED.challenger || currentRows.length !== EXPECTED.current || positions.length !== EXPECTED.positions) {
    throw new CheckRefusal('the fixed samples do not have their planned sizes');
  }
  const head = gitIn(REPO_ROOT, 'rev-parse', 'HEAD').stdout.trim();

  const tStart = performance.now();
  const started = new Date().toISOString();
  mkdirSync(join(out, 'servers'), { recursive: true });
  mkdirSync(join(out, 'suites'), { recursive: true });
  const put = (name, obj) => writeFileSync(join(out, name), `${JSON.stringify(obj, null, 1)}\n`, { flag: 'wx' });
  const progress = (phase) => appendFileSync(join(out, 'progress.jsonl'), `${JSON.stringify({ t: new Date().toISOString(), phase, s: Math.round((performance.now() - tStart) / 1000) })}\n`);
  const queueFile = (name, pairs) => {
    const f = join(out, 'servers', `${name}.seeds.json`);
    writeFileSync(f, JSON.stringify(seedQueue(pairs)), { flag: 'wx' });
    return f;
  };
  const posSeeds = positions.map((p) => positionSeeds(p.id));
  const checks = {};
  const servers = [];
  const serve = async (name, cfg) => {
    const s = await startServer({ ...cfg, logPath: join(out, 'servers', `${name}.log`) });
    servers.push(s);
    if (!s.up) {
      await s.stop();
      throw new CheckStop(name, `the ${name} server did not serve`);
    }
    return s;
  };
  const seededCapture = async (name, root, env) => {
    const q = queueFile(name, posSeeds);
    const s = await serve(name, { root, env, queueFile: q });
    const info = await (await fetch(`${s.base}/api/model-info`)).json();
    const positionsOut = await capture(s, positions);
    const text = await s.stop();
    const cap = { root, env, argv: s.args, modelInfo: info, banner: bannerLines(text), seedProblems: seedLogProblems(text, seedQueue(posSeeds)), positions: positionsOut };
    put(`${name}.json`, cap);
    return cap;
  };
  const seededReplay = async (name, root, env, rows, hardId, how) => {
    const s = await serve(name, { root, env, queueFile: queueFile(name, rows.map((r) => r.seeds)) });
    const answers = how === 'rest' ? await replayRest(s, rows) : await replayWs(s, rows);
    const text = await s.stop();
    const cmp = compareReplay(rows, answers, hardId);
    const seedProblems = seedLogProblems(text, seedQueue(rows.map((r) => r.seeds)));
    const res = { ...cmp, verdict: cmp.verdict === 'PASS' && seedProblems.length === 0 ? 'PASS' : 'FAIL', seedProblems, banner: bannerLines(text) };
    put(`${name}.json`, { ...res, answers });
    return { res, answers };
  };

  try {
    // G0: the model-loading suites, in the staged tree.
    checks.G0 = runSuites(REPO_ROOT, G0_COMMANDS, join(out, 'suites'), 'staged_');
    progress('G0');

    // S1, S2: the unseeded staged server.
    {
      const s = await serve('staged_live', { root: REPO_ROOT });
      const live = await liveCheck(s, NEW_ID);
      await s.stop();
      checks.S1 = live.S1;
      checks.S2 = live.S2;
      put('s1_s2.json', live);
    }
    progress('S1 S2');

    // A0: in-process, the hard model as server/index.js builds it.
    {
      const { resolveHardModel } = await import('../../server/model_manifest.js');
      const { AlphaZeroInference } = await import('../../server/inference.js');
      const { MirrorAveragingInference } = await import('../../server/mirror_inference.js');
      const hard = await resolveHardModel({}, REPO_ROOT);
      if (hard.manifest.model_id !== NEW_ID || fileSha(hard.graphPath) !== NEW_GRAPH_SHA256) throw new CheckStop('A0', `in-process hard is ${hard.manifest.model_id}`);
      const base = new AlphaZeroInference(hard.graphPath);
      await base.load();
      const answers = await replayInProcess(new MirrorAveragingInference(base), challengerRows);
      checks.A0 = compareReplay(challengerRows, answers, NEW_ID);
      put('a0.json', { ...checks.A0, answers });
    }
    progress('A0');

    // A1 (WS) and A2 (REST): fresh seeded staged servers.
    const a1 = await seededReplay('a1_ws', REPO_ROOT, {}, challengerRows, NEW_ID, 'ws');
    checks.A1 = a1.res;
    progress('A1');
    checks.A2 = (await seededReplay('a2_rest', REPO_ROOT, {}, challengerRows, NEW_ID, 'rest')).res;
    progress('A2');

    // S4: reference and staged seeded captures.
    const ref = await seededCapture('ref_capture', refRoot, {});
    progress('reference capture');
    const staged = await seededCapture('staged_capture', REPO_ROOT, {});
    const s4 = compareCaptures(ref, staged, 'served', NEW_ID);
    const refOk = ref.modelInfo?.hard_model_id === CURRENT_ID && ref.seedProblems.length === 0;
    checks.S4 = { ...s4, verdict: s4.verdict === 'PASS' && refOk && staged.seedProblems.length === 0 ? 'PASS' : 'FAIL', referenceHard: ref.modelInfo?.hard_model_id, seedProblems: [...ref.seedProblems, ...staged.seedProblems] };
    progress('S4');

    // R1: rollback by restart, in a copy of the staged tree without the new artifact. A rollback server that does
    // not serve is that check's FAIL, not a VOID.
    const copy = join(tmpdir(), `hmx-r1-${process.pid}`);
    try {
      const cp = spawnSync('cp', ['-c', '-R', REPO_ROOT, copy], { encoding: 'utf8' });
      if (cp.status !== 0) throw new CheckStop('R1', `cannot copy the staged tree: ${cp.stderr}`);
      rmSync(join(copy, '.git'), { force: true, recursive: true });
      rmSync(join(copy, 'models', NEW_ID), { recursive: true });
      // The documented rollback command (proposal §6, step 1): relative to the tree the server starts in.
      const env = { HARD_MODEL_MANIFEST: `models/${CURRENT_ID}/manifest.json` };
      const cap = await seededCapture('r1_capture', copy, env);
      const a = compareCaptures(ref, cap, 'exact', CURRENT_ID);
      const bannerOk = cap.banner.some((l) => new RegExp(`Hard model id:\\s+${CURRENT_ID}`).test(l)) && cap.modelInfo?.hard_model_id === CURRENT_ID && cap.modelInfo?.hard_mode === 'protocol';
      const b = (await seededReplay('r1_replay', copy, env, currentRows, CURRENT_ID, 'ws')).res;
      checks.R1 = { verdict: a.verdict === 'PASS' && bannerOk && cap.seedProblems.length === 0 && b.verdict === 'PASS' && !existsSync(join(copy, 'models', NEW_ID)) ? 'PASS' : 'FAIL',
        a: { ...a, bannerOk, seedProblems: cap.seedProblems }, b: { verdict: b.verdict, rows: b.rows, equal: b.equal, failureCount: b.failureCount } };
    } catch (err) {
      if (!(err instanceof CheckStop) || !String(err.step).startsWith('r1_')) throw err;
      checks.R1 = { verdict: 'FAIL', error: err.message };
    } finally {
      rmSync(copy, { recursive: true, force: true });
    }
    progress('R1');

    // R2: rollback by revert.
    try {
      const cap = await seededCapture('r2_capture', revertRoot, {});
      const a = compareCaptures(ref, cap, 'exact', CURRENT_ID);
      const bannerOk = cap.banner.some((l) => new RegExp(`Hard model id:\\s+${CURRENT_ID}`).test(l)) && cap.modelInfo?.hard_model_id === CURRENT_ID && cap.modelInfo?.hard_mode === 'protocol';
      const b = (await seededReplay('r2_replay', revertRoot, {}, currentRows, CURRENT_ID, 'ws')).res;
      progress('R2 servers');
      const suites = runSuites(revertRoot, [...b6Commands(py), ...G0_COMMANDS], join(out, 'suites'), 'revert_', { integration: LINKED_WORKTREE_FAILURES });
      checks.R2 = { verdict: a.verdict === 'PASS' && bannerOk && cap.seedProblems.length === 0 && b.verdict === 'PASS' && suites.verdict === 'PASS' ? 'PASS' : 'FAIL',
        a: { ...a, bannerOk, seedProblems: cap.seedProblems }, b: { verdict: b.verdict, rows: b.rows, equal: b.equal, failureCount: b.failureCount }, suites };
    } catch (err) {
      if (!(err instanceof CheckStop) || !String(err.step).startsWith('r2_')) throw err;
      checks.R2 = { verdict: 'FAIL', error: err.message };
    }
    progress('R2');

    // T: per-move timing from A1 against the match's own (informational).
    const med = (xs) => {
      const s = [...xs].sort((x, y) => x - y);
      return s.length ? s[Math.floor(s.length / 2)] : null;
    };
    const t = {};
    for (const [k, f] of [['ply_lt_20', (r) => r.ply < 20], ['ply_ge_20', (r) => r.ply >= 20]]) {
      const idx = challengerRows.map((r, i) => (f(r) ? i : -1)).filter((i) => i >= 0);
      t[k] = { n: idx.length, served_median_ms: med(idx.map((i) => a1.answers[i].elapsed)), match_median_ms: med(idx.map((i) => challengerRows[i].ms)) };
    }
    put('timing.json', t);
  } finally {
    for (const s of servers) await s.stop().catch(() => {});
  }

  const verdict = CHECK_IDS.every((id) => checks[id]?.verdict === 'PASS') ? 'PASS' : 'FAIL';
  const summary = Object.fromEntries(Object.entries(checks).map(([k, v]) => [k, v.verdict]));
  const record = {
    schema: RECORD_SCHEMA,
    written_by: 'script',
    status: 'COMPLETED',
    verdict,
    summary,
    started,
    ended: new Date().toISOString(),
    total_seconds: (performance.now() - tStart) / 1000,
    head,
    staged: { root: REPO_ROOT, hard_model_id: NEW_ID, model_manifest_js_sha256: MODEL_MANIFEST_JS_SHA256.new, new_manifest_sha256: NEW_MANIFEST_SHA256 },
    reference: { root: refRoot, commit: REF_COMMIT, hard_model_id: CURRENT_ID },
    revert: { root: revertRoot, head: gitIn(revertRoot, 'rev-parse', 'HEAD').stdout.trim() },
    samples: { games: EXPECTED.games, challenger_rows: challengerRows.length, current_rows: currentRows.length, positions: positions.length, match_record_sha256: MATCH_RECORD_SHA256 },
    checks: Object.fromEntries(Object.entries(checks).map(([k, v]) => [k, { ...v, failures: v.failures?.slice(0, 20) }])),
    kill_s: KILL_S,
    pid: process.pid,
    runtime: { node: process.version, onnxruntime_node: ORT_VERSION },
  };
  const rp = recordProblems(record);
  if (rp.length) throw new CheckStop('record', `the record is not admissible: ${rp.join('; ')}`);
  put('record.json', record);
}

async function main() {
  const a = parseArgs(process.argv.slice(2));
  if (!a || a.cmd !== 'run' || a.stage !== 'check') {
    console.error('usage: node tests/htb_integration/hmx_integration_check.mjs run check --ref-root <dir> --revert-root <dir> --kill-s 14400');
    return 2;
  }
  try {
    await run(a.opts);
    return 0;
  } catch (err) {
    if (err instanceof CheckRefusal) {
      console.error(`REFUSED: ${err.message}`);
      return 4;
    }
    const out = join(REPO_ROOT, RUN_REL);
    const step = err instanceof CheckStop ? err.step : 'crashed';
    console.error(err instanceof CheckStop ? `STOP at ${err.step}: ${err.message}` : (err?.stack ?? String(err)));
    if (existsSync(out) && !existsSync(join(out, 'record.json'))) {
      writeFileSync(join(out, 'record.json'), `${JSON.stringify({ schema: RECORD_SCHEMA, written_by: 'script', status: 'VOID', void: { step, detail: err?.message ?? String(err) } }, null, 1)}\n`, { flag: 'wx' });
    }
    return 3;
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().then((code) => {
    process.exitCode = code;
  });
}
