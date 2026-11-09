#!/usr/bin/env node
/**
 * G2 live check: real REST and WebSocket requests against a real server (server/index.js) with the pinned models.
 *
 *   node tests/htb_integration/live_check.mjs <out.json>
 *
 * Starts the server on a free port with the default environment (HARD_MODE unset, i.e. protocol) and exercises:
 *   L1 startup banner: the hard mode and both model ids;
 *   L2 /api/model-info;
 *   L3 WS hard moves before and after ply 20, both colours: protocol fields, legality, visit totals, the played
 *      move's actual visits, valueKind by ply, value perspective (valueRed from the displayed value, by side to
 *      move), progress messages;
 *   L4 REST hard: the same fields, cached:false, and the cache untouched (/api/stats); control: a repeated REST
 *      medium request IS served from the cache;
 *   L5 conflicting overrides refused on both transports with the one message; deterministicMode:false accepted;
 *   L6 a won position and a 12x12 board refused on both transports;
 *   L7 cancellation (WS cancel -> no answer for that id) and latest-wins (a newer move aborts the older one);
 *   L8 easy and medium WS responses keep today's exact key set and simulation counts;
 *   L9 per-move elapsed time for hard (informational, for G3's timing formula).
 * Each check records its observations. PASS only if every check passes. Exit 0 PASS, 1 FAIL, 4 error.
 */
import { spawn } from 'node:child_process';
import { writeFileSync, readFileSync } from 'node:fs';
import net from 'node:net';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import WebSocket from 'ws';

import { TwixtState } from '../../server/gameLogic.js';
import { POLICY_CONFLICT_MESSAGE } from '../../server/readout_policy.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const out = process.argv[2];
const checks = [];
const check = (id, ok, detail) => checks.push({ id, ok: Boolean(ok), detail });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const freePort = () =>
  new Promise((resolve) => {
    const s = net.createServer().listen(0, () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
  });

const corpus = JSON.parse(readFileSync(join(ROOT, 'tests/parity/corpus.json'), 'utf8'));
const all = [...corpus.primary, ...corpus.edge];
const pick = (pred) => all.find(pred);
const POSITIONS = [
  { label: 'empty board', moves: [] },
  pick((p) => p.ply >= 2 && p.ply < 20 && p.to_move === 'red'),
  pick((p) => p.ply >= 2 && p.ply < 20 && p.to_move === 'black'),
  pick((p) => p.ply >= 20 && p.ply < 50 && p.to_move === 'red'),
  pick((p) => p.ply >= 20 && p.ply < 50 && p.to_move === 'black'),
  pick((p) => p.ply >= 100 && p.to_move === 'red'),
  pick((p) => p.ply >= 100 && p.to_move === 'black'),
].map((p) => ({ label: p.label ?? p.id, state: TwixtState.fromMoves(p.moves) }));

const port = await freePort();
const base = `http://127.0.0.1:${port}`;
const env = { ...process.env, PORT: String(port) };
delete env.HARD_MODE;
delete env.HARD_MODEL_MANIFEST;
delete env.MODEL_MANIFEST;
const server = spawn(process.execPath, [join(ROOT, 'server/index.js')], { env, cwd: ROOT });
let banner = '';
server.stdout.on('data', (d) => (banner += d));
server.stderr.on('data', (d) => (banner += d));
const stop = () => server.kill('SIGTERM');

async function waitUp() {
  for (let i = 0; i < 200; i++) {
    try {
      const r = await fetch(`${base}/api/health`);
      if (r.ok) return;
    } catch {
      /* not yet */
    }
    await sleep(100);
  }
  throw new Error(`server did not start:\n${banner}`);
}

const post = async (path, body) => {
  const r = await fetch(`${base}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  return { status: r.status, body: await r.json() };
};

function wsClient() {
  const ws = new WebSocket(`ws://127.0.0.1:${port}/ws`);
  const msgs = [];
  ws.on('message', (raw) => msgs.push(JSON.parse(raw.toString())));
  const open = new Promise((r) => ws.on('open', r));
  const waitFor = async (pred, ms = 60000) => {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) {
      const m = msgs.find(pred);
      if (m) return m;
      await sleep(10);
    }
    return null;
  };
  return { ws, msgs, open, waitFor, send: (o) => ws.send(JSON.stringify(o)) };
}

const legal = (state, move) => state.legalMoves().some(([r, c]) => r === move.row && c === move.col);
const clamp = (v) => Math.max(-1, Math.min(1, v));

try {
  await waitUp();
  // L1, L2
  check('L1 banner', /Hard mode:\s+protocol/.test(banner) && /Hard model id:\s+78c16a31c3dba37f/.test(banner) && /Model id:\s+c34b7ff3297c785a/.test(banner), banner.split('\n').filter((l) => /Model id|Hard/.test(l)));
  const info = await (await fetch(`${base}/api/model-info`)).json();
  check('L2 model-info', info.hard_mode === 'protocol' && info.hard_model_id === '78c16a31c3dba37f' && info.model_id === 'c34b7ff3297c785a', { hard_mode: info.hard_mode, hard_model_id: info.hard_model_id, model_id: info.model_id });

  // L3 WS hard
  const c = wsClient();
  await c.open;
  const timings = [];
  for (const [i, p] of POSITIONS.entries()) {
    const id = `h${i}`;
    c.send({ type: 'move', id, state: p.state.toDict(), difficulty: 'hard', includeVisits: true });
    const m = await c.waitFor((x) => x.id === id && (x.type === 'bestmove' || x.type === 'error'), 120000);
    const prog = c.msgs.filter((x) => x.id === id && x.type === 'progress');
    const ply = p.state.ply;
    const total = m?.visits ? Object.values(m.visits).reduce((a, b) => a + b, 0) : null;
    const display = m?.valueKind === 'played_move_mean' ? m.moveValue : m?.rootValue;
    const expectedRed = m ? clamp(p.state.toMove === 'red' ? display : -display) : null;
    const ok =
      m?.type === 'bestmove' && m.search === 'protocol' && m.modelId === '78c16a31c3dba37f' && m.nSimulations === 200 &&
      legal(p.state, m.move) && total === 200 && m.moveVisits === m.visits[`${m.move.row},${m.move.col}`] &&
      m.valueKind === (ply < 20 ? 'root_mean' : 'played_move_mean') && m.evalToMove === p.state.toMove &&
      m.toMove === p.state.toMove && m.valueRed === expectedRed && Number.isFinite(m.rootValue) && Math.abs(m.rootValue) <= 1 &&
      (m.valueKind === 'root_mean' || (m.moveVisits > 0 && Number.isFinite(m.moveValue))) &&
      prog.length > 0 && prog.every((x) => x.total === 200 && x.toMove === p.state.toMove);
    check(`L3 WS hard ${p.label} (ply ${ply}, ${p.state.toMove})`, ok, {
      move: m?.move, valueKind: m?.valueKind, rootValue: m?.rootValue, moveValue: m?.moveValue, moveVisits: m?.moveVisits,
      valueRed: m?.valueRed, expectedRed, visitsTotal: total, progressMessages: prog.length, elapsed: m?.elapsed, type: m?.type, error: m?.message,
    });
    timings.push({ ply, elapsedMs: m?.elapsed });
  }

  // L4 REST hard + cache
  const s0 = (await (await fetch(`${base}/api/stats`)).json()).cache;
  const p4 = POSITIONS[4];
  const r4 = await post('/api/move', { state: p4.state.toDict(), difficulty: 'hard' });
  const s1 = (await (await fetch(`${base}/api/stats`)).json()).cache;
  const restOk = r4.status === 200 && r4.body.cached === false && r4.body.search === 'protocol' && r4.body.modelId === '78c16a31c3dba37f' &&
    legal(p4.state, r4.body.move) && r4.body.valueKind === 'played_move_mean' && Number.isFinite(r4.body.value) &&
    Object.values(r4.body.visits).reduce((a, b) => a + b, 0) === 200;
  check('L4 REST hard, cache untouched', restOk && JSON.stringify(s0) === JSON.stringify(s1), { status: r4.status, cached: r4.body.cached, statsBefore: s0, statsAfter: s1 });
  const m1 = await post('/api/move', { state: POSITIONS[1].state.toDict(), difficulty: 'medium' });
  const m2 = await post('/api/move', { state: POSITIONS[1].state.toDict(), difficulty: 'medium' });
  const s2 = (await (await fetch(`${base}/api/stats`)).json()).cache;
  check('L4 control: REST medium is cached', m1.body.cached === false && m2.body.cached === true && s2.hits === s1.hits + 1 && !('search' in m2.body), { first: m1.body.cached, second: m2.body.cached, stats: s2 });

  // L5 overrides
  const sd = POSITIONS[3].state.toDict();
  const o1 = await post('/api/move', { state: sd, difficulty: 'hard', temperature: 0.5 });
  const o2 = await post('/api/move', { state: sd, difficulty: 'hard', deterministicMode: true });
  const o3 = await post('/api/move', { state: sd, difficulty: 'hard', deterministicMode: false });
  // One request at a time: on one connection a newer move aborts the older one (latest-wins), which would
  // suppress the older refusal. Run 1 sent the L6 pair back-to-back and lost an answer that way.
  c.send({ type: 'move', id: 'o4', state: sd, difficulty: 'hard', temperature: 0.5 });
  const o4 = await c.waitFor((x) => x.id === 'o4' && x.type !== 'progress', 5000);
  c.send({ type: 'move', id: 'o5', state: sd, difficulty: 'hard', deterministicMode: true });
  const o5 = await c.waitFor((x) => x.id === 'o5' && x.type !== 'progress', 5000);
  check('L5 overrides refused (REST 400, WS error, one message); deterministicMode:false accepted',
    o1.status === 400 && o1.body.error === POLICY_CONFLICT_MESSAGE && o1.body.code === 'POLICY_CONFLICT' &&
    o2.status === 400 && o2.body.code === 'POLICY_CONFLICT' && o3.status === 200 && o3.body.search === 'protocol' &&
    o4?.type === 'error' && o4.message === POLICY_CONFLICT_MESSAGE && o5?.type === 'error' && o5.message === POLICY_CONFLICT_MESSAGE,
    { rest: [o1.status, o2.status, o3.status], ws: [o4?.type, o4?.message, o5?.type] });

  // L6 refusals
  const chain = [];
  const red = Array.from({ length: 12 }, (_, k) => [2 * k, (k % 2) ? 6 : 5]);
  const black = [...Array.from({ length: 11 }, (_, k) => [2 * k + 1, 19]), [4, 15]];
  for (let k = 0; k < 12; k++) chain.push(red[k], black[k]);
  const won = TwixtState.fromMoves(chain).applyMove([23, 4]);
  const small = new TwixtState({ boardSize: 12 });
  const f1 = await post('/api/move', { state: won.toDict(), difficulty: 'hard' });
  const f2 = await post('/api/move', { state: small.toDict(), difficulty: 'hard' });
  c.send({ type: 'move', id: 'f3', state: won.toDict(), difficulty: 'hard' });
  const f3 = await c.waitFor((x) => x.id === 'f3' && x.type !== 'progress', 5000);
  c.send({ type: 'move', id: 'f4', state: small.toDict(), difficulty: 'hard' });
  const f4 = await c.waitFor((x) => x.id === 'f4' && x.type !== 'progress', 5000);
  check('L6 won position and 12x12 board refused on both transports', won.winner() === 'red' &&
    f1.status === 400 && f1.body.code === 'TERMINAL_ROOT' && f2.status === 400 && f2.body.code === 'NOT_24' &&
    f3?.type === 'error' && f3.message === f1.body.error && f4?.type === 'error' && f4.message === f2.body.error,
    { rest: [f1.status, f1.body.code, f2.status, f2.body.code], ws: [f3?.message, f4?.message] });

  // L7 cancellation and latest-wins
  const ply30 = POSITIONS[3].state.toDict();
  c.send({ type: 'move', id: 'k1', state: ply30, difficulty: 'hard' });
  await sleep(50);
  c.send({ type: 'cancel', id: 'k1' });
  await sleep(Math.max(3000, 3 * (timings[3]?.elapsedMs ?? 1000)));
  const k1 = c.msgs.filter((x) => x.id === 'k1' && x.type !== 'progress');
  c.send({ type: 'move', id: 'k2', state: ply30, difficulty: 'hard' });
  await sleep(30);
  c.send({ type: 'move', id: 'k3', state: ply30, difficulty: 'hard' });
  const k3 = await c.waitFor((x) => x.id === 'k3' && x.type === 'bestmove', 60000);
  await sleep(Math.max(2000, 2 * (timings[3]?.elapsedMs ?? 1000)));
  const k2 = c.msgs.filter((x) => x.id === 'k2' && x.type !== 'progress');
  check('L7 cancel answers nothing; a newer move aborts the older; the server keeps serving',
    k1.length === 0 && k2.length === 0 && k3?.search === 'protocol', { cancelledAnswers: k1.length, supersededAnswers: k2.length, latest: k3?.type });

  // L8 easy / medium shape
  const KEYS = ['elapsed', 'evalToMove', 'id', 'move', 'nSimulations', 'rootValue', 'toMove', 'type', 'valueRed'];
  for (const [d, n] of [['easy', 100], ['medium', 400]]) {
    c.send({ type: 'move', id: `p_${d}`, state: POSITIONS[3].state.toDict(), difficulty: d });
    const m = await c.waitFor((x) => x.id === `p_${d}` && x.type === 'bestmove', 60000);
    check(`L8 ${d} WS response keeps today's key set`, m && JSON.stringify(Object.keys(m).sort()) === JSON.stringify(KEYS) && m.nSimulations === n, { keys: m && Object.keys(m).sort() });
  }
  check('L9 timing recorded (informational)', true, { hardMoves: timings });
  c.ws.close();
} catch (err) {
  check('harness error', false, String(err?.stack ?? err));
} finally {
  stop();
}
const verdict = checks.every((x) => x.ok) ? 'PASS' : 'FAIL';
writeFileSync(out, JSON.stringify({ verdict, port, node: process.version, checks }, null, 1));
for (const x of checks) console.log(`${x.ok ? 'ok  ' : 'FAIL'} ${x.id}`);
console.log(`LIVE CHECK ${verdict}`);
process.exitCode = verdict === 'PASS' ? 0 : 1;
