#!/usr/bin/env node
/**
 * G2 rollback checks R1-R3 (integration proposal §7, §8; decision D7: HARD_MODE=legacy).
 *
 *   node tests/htb_integration/rollback_check.mjs capture <out.json> --root <repo_root> [--env KEY=VAL ...]
 *   node tests/htb_integration/rollback_check.mjs compare <reference.json> <subject.json> --hard exact|protocol
 *
 * capture: starts <repo_root>/server/index.js on a free port with the given environment (HARD_MODE,
 * HARD_MODEL_MANIFEST and MODEL_MANIFEST are cleared first, then --env applied). It records the banner and
 * /api/model-info, then for each position (the 126 non-terminal Phase 2 parity-corpus positions and the 16 golden
 * positions) sends one WS request per difficulty, in turn, with includeVisits:
 *   - hard: move, visits, rootValue, valueRed, nSimulations, and the response's key set;
 *   - easy, medium: visits, rootValue, valueRed, nSimulations and the key set. Their moves are sampled with
 *     Math.random, so moves are recorded but not compared.
 * If the server exits before it serves, the capture records `refused` with its output.
 *
 * compare: reference = today's hard, recorded from a server at 80acd87e (before the integration).
 *   - --hard exact (R1: the integrated server under HARD_MODE=legacy with the hard artifact unavailable; R3: the
 *     code revert): hard's move, visits, rootValue, valueRed, nSimulations and key set are equal for every position.
 *   - --hard protocol (R2: the integrated server in its default mode): hard is served by the protocol (search
 *     'protocol', modelId 78c16a31c3dba37f, 200 simulations) for every position.
 *   - Always: easy and medium visits, rootValue, valueRed, nSimulations and key set are equal for every position.
 * Exit 0 PASS, 1 FAIL, 2 usage.
 */
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import net from 'node:net';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import WebSocket from 'ws';

import { TwixtState } from '../../server/gameLogic.js';
import { enumerateCases } from '../mcts_golden/cases.mjs';
import { readFixture } from '../mcts_golden/worker.mjs';

const HERE_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const DIFFICULTIES = ['hard', 'easy', 'medium'];

function positions() {
  const corpus = JSON.parse(readFileSync(join(HERE_ROOT, 'tests/parity/corpus.json'), 'utf8'));
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

const freePort = () =>
  new Promise((resolve) => {
    const s = net.createServer().listen(0, () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
  });

async function capture(outPath, root, envPairs) {
  const port = await freePort();
  const env = { ...process.env, PORT: String(port) };
  for (const k of ['HARD_MODE', 'HARD_MODEL_MANIFEST', 'MODEL_MANIFEST']) delete env[k];
  for (const kv of envPairs) {
    const i = kv.indexOf('=');
    env[kv.slice(0, i)] = kv.slice(i + 1);
  }
  const server = spawn(process.execPath, [join(root, 'server/index.js')], { env, cwd: root });
  let output = '';
  let exited = null;
  server.stdout.on('data', (d) => (output += d));
  server.stderr.on('data', (d) => (output += d));
  server.on('exit', (code) => (exited = code));
  const base = `http://127.0.0.1:${port}`;
  let up = false;
  for (let i = 0; i < 300 && exited === null; i++) {
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
  const record = { root, env: envPairs, startedAt: new Date().toISOString(), node: process.version };
  if (!up) {
    server.kill('SIGTERM');
    writeFileSync(outPath, JSON.stringify({ ...record, refused: true, exitCode: exited, output }, null, 1));
    console.log(`server did not serve (exit ${exited})`);
    return;
  }
  record.modelInfo = await (await fetch(`${base}/api/model-info`)).json();
  const ws = new WebSocket(`ws://127.0.0.1:${port}/ws`);
  const msgs = new Map();
  ws.on('message', (raw) => {
    const m = JSON.parse(raw.toString());
    if (m.type !== 'progress') msgs.set(m.id, m);
  });
  await new Promise((r) => ws.on('open', r));
  const results = [];
  for (const p of positions()) {
    const row = { id: p.id, ply: p.state.ply, toMove: p.state.toMove };
    for (const d of DIFFICULTIES) {
      const id = `${p.id}:${d}`;
      ws.send(JSON.stringify({ type: 'move', id, state: p.state.toDict(), difficulty: d, includeVisits: true }));
      while (!msgs.has(id)) await sleep(5);
      const m = msgs.get(id);
      row[d] = {
        type: m.type,
        message: m.message,
        keys: Object.keys(m).sort(),
        move: m.move,
        visits: m.visits,
        rootValue: m.rootValue,
        valueRed: m.valueRed,
        nSimulations: m.nSimulations,
        search: m.search,
        modelId: m.modelId,
      };
    }
    results.push(row);
  }
  ws.close();
  server.kill('SIGTERM');
  record.finishedAt = new Date().toISOString();
  record.banner = output.split('\n').filter((l) => /Model id|Hard|Manifest/.test(l));
  writeFileSync(outPath, JSON.stringify({ ...record, refused: false, positions: results }));
  console.log(`captured ${results.length} positions x ${DIFFICULTIES.length} difficulties -> ${outPath}`);
}

function compare(refPath, subPath, hardRule) {
  const ref = JSON.parse(readFileSync(refPath, 'utf8'));
  const sub = JSON.parse(readFileSync(subPath, 'utf8'));
  const failures = [];
  if (ref.refused || sub.refused) failures.push('a capture did not serve');
  const subById = new Map((sub.positions ?? []).map((p) => [p.id, p]));
  if ((ref.positions ?? []).length !== (sub.positions ?? []).length) failures.push('position count');
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  let hardEqual = 0;
  let plainEqual = 0;
  for (const r of ref.positions ?? []) {
    const s = subById.get(r.id);
    if (!s) {
      failures.push(`missing ${r.id}`);
      continue;
    }
    for (const d of ['easy', 'medium']) {
      const f = ['type', 'keys', 'visits', 'rootValue', 'valueRed', 'nSimulations'];
      const bad = f.filter((k) => !same(r[d][k], s[d][k]));
      if (bad.length) failures.push(`${r.id} ${d}: ${bad.join(',')}`);
      else plainEqual += 1;
    }
    if (hardRule === 'exact') {
      const f = ['type', 'keys', 'move', 'visits', 'rootValue', 'valueRed', 'nSimulations'];
      const bad = f.filter((k) => !same(r.hard[k], s.hard[k]));
      if (bad.length) failures.push(`${r.id} hard: ${bad.join(',')}`);
      else hardEqual += 1;
    } else if (!(s.hard.type === 'bestmove' && s.hard.search === 'protocol' && s.hard.modelId === '78c16a31c3dba37f' && s.hard.nSimulations === 200)) {
      failures.push(`${r.id} hard is not the protocol`);
    } else hardEqual += 1;
  }
  const verdict = failures.length ? 'FAIL' : 'PASS';
  console.log(JSON.stringify({ verdict, hardRule, positions: (ref.positions ?? []).length, hardOk: hardEqual, plainOk: plainEqual,
    refModelInfo: { model_id: ref.modelInfo?.model_id }, subModelInfo: { model_id: sub.modelInfo?.model_id, hard_mode: sub.modelInfo?.hard_mode, hard_model_id: sub.modelInfo?.hard_model_id },
    subBanner: sub.banner, failures: failures.slice(0, 30), failureCount: failures.length }, null, 1));
  return verdict === 'PASS' ? 0 : 1;
}

const [cmd, ...rest] = process.argv.slice(2);
if (cmd === 'capture') {
  const out = rest[0];
  const rootIdx = rest.indexOf('--root');
  const envs = [];
  rest.forEach((a, i) => a === '--env' && envs.push(rest[i + 1]));
  await capture(out, rest[rootIdx + 1], envs);
} else if (cmd === 'compare') {
  process.exitCode = compare(rest[0], rest[1], rest[rest.indexOf('--hard') + 1]);
} else {
  console.error('usage: see the header');
  process.exitCode = 2;
}
