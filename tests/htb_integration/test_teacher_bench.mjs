/**
 * Student-teacher benchmark: model-free tests (fake evaluator, an in-process fake teacher). No ONNX model, no
 * TensorFlow, no gate opened. The cross-language canonical-state test runs our Python TwixtState (repo .venv) only.
 * Run: node --test tests/htb_integration/test_teacher_bench.mjs
 */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

import { TwixtState } from '../../server/gameLogic.js';
import { MirrorAveragingInference } from '../../server/mirror_inference.js';
import { protocolMove } from '../../server/protocol_move.js';
import { resolvePolicy } from '../../server/readout_policy.js';
import { fakeInference } from '../fixtures/htb_integration/fake_evaluator.mjs';
import * as AN from './teacher_bench_analyse.mjs';
import * as GATE from './teacher_bench_authorization.mjs';
import * as D from './teacher_bench.mjs';

const PY = join(D.REPO_ROOT, '.venv/bin/python');
const G3 = (f) => JSON.parse(readFileSync(join(D.REPO_ROOT, 'docs/superpowers/evidence/htb-g3-match/games', f), 'utf8'));
const SMALL = { ...resolvePolicy({ difficulty: 'hard', hardMode: 'protocol' }), nSims: 64 };

/** In-process teacher double: Node state, the same canonical digest, first-legal move (or a scripted fault). */
function fakeTeacher({ corruptAtPly = null, drawFlagAtPly = null, illegalAtPly = null } = {}) {
  let s = null;
  let flag = null;
  const rep = () => ({ ok: true, ply: s.ply, canon: s.ply === corruptAtPly ? 'x'.repeat(64) : D.canonDigest(s), tb_draw: flag !== null, tb_draw_ply: flag });
  const bump = () => {
    if (flag === null && drawFlagAtPly !== null && s.ply >= drawFlagAtPly) flag = s.ply;
  };
  return {
    async request(r) {
      if (r.cmd === 'start') {
        s = new TwixtState();
        flag = null;
        for (const m of r.opening) s = s.applyMove(m);
        return rep();
      }
      if (r.cmd === 'apply') {
        s = s.applyMove(r.move);
        bump();
        return rep();
      }
      if (r.cmd === 'move') {
        if (s.ply === illegalAtPly) return { ok: true, move: [0, 0], prob: 1, value: 0 };
        const m = [...s.legalMoves()].sort((a, b) => a[0] - b[0] || a[1] - b[1])[0];
        return { ok: true, move: m, prob: 0.5, value: 0 };
      }
      throw new Error(`fake teacher: ${r.cmd}`);
    },
  };
}

test('the gates are closed in the committed state', () => {
  assert.equal(GATE.BENCH_SMOKE_AUTHORIZED, false);
  assert.equal(GATE.BENCH_MATCH_AUTHORIZED, false);
});

test('openings: the analyser re-derives the driver\'s; distinct up to reflection; central; legal', () => {
  for (const [kind, n] of [['opening', 200], ['smoke opening', 1]]) {
    const a = D.openingList(kind, n);
    const b = AN.deriveOpenings(kind, n);
    assert.deepEqual(a, b);
    const keys = new Set();
    for (const { moves } of a) {
      assert.equal(moves.length, 4);
      let s = new TwixtState();
      for (const [r, c] of moves) {
        assert.ok(r >= 4 && r <= 19 && c >= 4 && c <= 19);
        assert.ok(s.legalMoves().some((m) => m[0] === r && m[1] === c));
        s = s.applyMove([r, c]);
      }
      const k1 = JSON.stringify(moves);
      const k2 = JSON.stringify(moves.map(([r, c]) => [r, 23 - c]));
      assert.ok(!keys.has(k1) && !keys.has(k2));
      keys.add(k1);
    }
  }
  assert.deepEqual(D.openingList('opening', 110), D.openingList('opening', 200).slice(0, 110)); // arm P = arm N's first 110
});

test('schedule: 620 match games and 4 smoke games, colour-reversed pairs, arm P on arm N\'s openings', () => {
  const m = D.schedule('match');
  assert.equal(m.length, 620);
  assert.equal(m.filter((t) => t.arm === 'N').length, 400);
  assert.equal(m.filter((t) => t.arm === 'P').length, 220);
  for (const t of m) assert.equal(t.studentColour, t.game === 0 ? 'red' : 'black');
  const nOpen = new Map(m.filter((t) => t.arm === 'N').map((t) => [t.pair, t.opening]));
  for (const t of m.filter((x) => x.arm === 'P')) assert.deepEqual(t.opening, nOpen.get(t.pair));
  assert.equal(D.schedule('smoke').length, 4);
});

test('seeds: the analyser re-derives them; every reachable arm-P seed is distinct and below 2^48', () => {
  const seen = new Set();
  for (const [stage, pairs] of [['smoke', 1], ['match', 110]]) {
    for (let p = 0; p < pairs; p++) {
      for (const g of [0, 1]) {
        for (let ply = 0; ply < 576; ply++) {
          const s = D.moveSeeds(stage, p, g, ply);
          if (ply % 97 === 0) assert.deepEqual(s, AN.deriveSeeds(stage, p, g, ply));
          for (const v of [s.search, s.readout]) {
            assert.ok(Number.isSafeInteger(v) && v < 2 ** 48);
            assert.ok(!seen.has(v));
            seen.add(v);
          }
        }
      }
    }
  }
});

test('canonical state: Node and our Python TwixtState give the same digest at every ply (G3 draw and a decisive game)', () => {
  for (const f of ['pair_0057_game_1.json', 'pair_0001_game_0.json']) {
    const moves = G3(f).moves;
    const dir = mkdtempSync(join(tmpdir(), 'bench-canon-'));
    writeFileSync(join(dir, 'm.json'), JSON.stringify(moves));
    const r = spawnSync(PY, ['-B', '-m', 'scripts.GPU.alphazero.htb_bench_teacher', 'canon', join(dir, 'm.json')], { cwd: D.REPO_ROOT, env: { ...process.env, PYTHONPATH: '.' }, encoding: 'utf8', maxBuffer: 1 << 26 });
    assert.equal(r.status, 0, r.stderr);
    const py = JSON.parse(r.stdout);
    let s = new TwixtState();
    const node = [D.canonDigest(s)];
    for (const m of moves) {
      s = s.applyMove(m);
      node.push(D.canonDigest(s));
    }
    assert.equal(py.length, moves.length + 1);
    assert.deepEqual(node, py, f);
  }
});

test('canonical state: a bridge, winner or legal-set difference changes the digest', () => {
  let s = new TwixtState();
  for (const m of [[10, 10], [5, 5], [12, 11], [5, 8]]) s = s.applyMove(m);
  const c = D.canonical(s);
  assert.deepEqual(c.bridges, { red: [[[10, 10], [12, 11]]], black: [] });
  const d0 = D.canonDigest(s);
  const t = TwixtState.fromDict(s.toDict());
  t.bridges = new Set();
  assert.notEqual(D.canonDigest(t), d0);
});

test('arm N: the mirror-averaged policy argmax, ties to the lower (row, col)', async () => {
  const s = new TwixtState().applyMove([10, 10]);
  const legal = s.legalMoves();
  const flat = { async evaluate(_h, moves) { return { priors: new Map(moves.map((m) => [`${m[0]},${m[1]}`, 0.25])), value: 0 }; } };
  const sorted = [...legal].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  assert.equal((await D.rawPolicyMove(flat, s)).moveKey, `${sorted[0][0]},${sorted[0][1]}`);
  const peaked = { async evaluate(_h, moves) { return { priors: new Map(moves.map((m) => [`${m[0]},${m[1]}`, m[0] === 7 && m[1] === 9 ? 0.9 : 0.01])), value: 0 }; } };
  assert.equal((await D.rawPolicyMove(peaked, s)).moveKey, '7,9');
});

test('arm P: the student is protocolMove with the derived seeds, move for move', async () => {
  const inference = new MirrorAveragingInference(fakeInference());
  const st = D.makeStudent('P', inference, SMALL);
  let s = new TwixtState();
  for (const m of [[10, 10], [5, 5], [12, 11], [5, 8]]) s = s.applyMove(m);
  for (let k = 0; k < 4; k++) {
    const ours = await st.move(s, { stage: 'match', pair: 3, game: 1 });
    const ref = await protocolMove({ gameState: TwixtState.fromDict(s.toDict()), policy: SMALL, inference, seeds: D.moveSeeds('match', 3, 1, s.ply) });
    assert.equal(ours.moveKey, ref.moveKey);
    assert.deepEqual(ours.extra.seeds, D.moveSeeds('match', 3, 1, s.ply));
    s = s.applyMove(ours.moveKey.split(',').map(Number));
  }
});

test('one game: product end rule, the tb draw flag recorded but never ending play, and the analyser admits it', async () => {
  const inference = new MirrorAveragingInference(fakeInference());
  const task = D.schedule('smoke')[0];
  const rec = await D.playGame({ stage: 'smoke', task, student: D.makeStudent('N', inference, SMALL), teacher: fakeTeacher({ drawFlagAtPly: 9 }) });
  assert.equal(rec.tb_draw_flag_ply, 9);
  assert.ok(rec.plies > 9);
  let s = new TwixtState();
  for (const m of rec.moves) s = s.applyMove(m);
  assert.ok(s.isTerminal());
  assert.equal(rec.result, s.winner() ?? 'draw');
  const openings = AN.deriveOpenings('smoke opening', 1);
  assert.equal(AN.validateGame(rec, { stage: 'smoke', arm: 'N', pair: 0, game: 0, opening: openings[0] }), rec);
  assert.throws(() => AN.validateGame({ ...rec, student_score: 1 - rec.student_score === 0.5 ? 0 : 1 - rec.student_score }, { stage: 'smoke', arm: 'N', pair: 0, game: 0, opening: openings[0] }), AN.Invalid);
});

test('a canonical mismatch or an illegal teacher move stops the game', async () => {
  const inference = new MirrorAveragingInference(fakeInference());
  const task = D.schedule('smoke')[0];
  const student = D.makeStudent('N', inference, SMALL);
  await assert.rejects(D.playGame({ stage: 'smoke', task, student, teacher: fakeTeacher({ corruptAtPly: 7 }) }), (e) => e instanceof D.BenchStop && e.step === 'canonical');
  await assert.rejects(D.playGame({ stage: 'smoke', task, student, teacher: fakeTeacher({ illegalAtPly: 5 }) }), (e) => e instanceof D.BenchStop && e.step === 'illegal');
});

test('classification: Hoeffding half-widths, the boundaries and sweeps', () => {
  assert.equal(AN.hoeffdingHalfWidth(200).toFixed(4), '0.0960');
  assert.equal(AN.hoeffdingHalfWidth(110).toFixed(4), '0.1295');
  const h = AN.hoeffdingHalfWidth(200);
  assert.equal(AN.classify(0.5 - h - 1e-9, 200), 'TRAILS');
  assert.equal(AN.classify(0.5 - h + 1e-9, 200), 'UNRESOLVED');
  assert.equal(AN.classify(0.5 + h + 1e-9, 200), 'AHEAD');
  assert.equal(AN.classify(0.5, 110), 'UNRESOLVED');
  assert.equal(AN.classify(0, 200), 'TRAILS');
  assert.equal(AN.classify(1, 110), 'AHEAD');
});

test('analyse: per-arm scores, draws and colours; the reading follows arm N only', () => {
  const games = [];
  const mk = (arm, pair, game, score) => ({ arm, pair, game, student_colour: game ? 'black' : 'red', student_score: score, result: score === 0.5 ? 'draw' : 'x', plies: 50 + pair, tb_draw_flag_ply: score === 0.5 ? 80 : null });
  for (let p = 0; p < 200; p++) for (const g of [0, 1]) games.push(mk('N', p, g, p < 30 ? 0.5 : 0));
  for (let p = 0; p < 110; p++) for (const g of [0, 1]) games.push(mk('P', p, g, g === 0 ? 1 : 0.5));
  const a = AN.analyse({ games });
  assert.equal(a.arm_N.classification, 'TRAILS');
  assert.equal(a.arm_N.student.D, 60);
  assert.equal(a.arm_N.tb_draw_flag.in_draws, 60);
  assert.equal(a.arm_P.mean_pair_score, 0.75);
  assert.equal(a.arm_P.classification, 'AHEAD');
  assert.equal(a.arm_P.by_colour.student_red.score, 1);
  assert.equal(a.reading.arm_N, 'TRAILS');
  assert.match(a.reading.text, /fresh distillation experiment/);
  assert.match(a.reading.arm_P_note, /cannot say which/);
});

test('smoke admission and timing: outer.json required; the estimate reads wall times only', async () => {
  const root = mkdtempSync(join(tmpdir(), 'bench-adm-'));
  const run = join(root, AN.DIRS.smoke.run);
  const outer = join(root, AN.DIRS.smoke.outer);
  mkdirSync(join(run, 'games'), { recursive: true });
  mkdirSync(outer, { recursive: true });
  const inference = new MirrorAveragingInference(fakeInference());
  for (const task of D.schedule('smoke')) {
    const rec = await D.playGame({ stage: 'smoke', task, student: D.makeStudent(task.arm, inference, SMALL), teacher: fakeTeacher() });
    writeFileSync(join(run, 'games', `${task.arm}_pair_0000_game_${task.game}.json`), JSON.stringify(rec));
  }
  writeFileSync(join(run, 'record.json'), JSON.stringify({ status: 'COMPLETED', written_by: 'script', stage: 'smoke', games: 4, startup_seconds: 12.5 }));
  assert.throws(() => AN.admit('smoke', root), AN.Invalid); // no outer.json yet
  const good = { inner_exit: 0, outcome: 'COMPLETED', outer_exit: 0, gate_restored: true, survivors: '', record_by: 'script' };
  writeFileSync(join(outer, 'outer.json'), JSON.stringify({ ...good, outcome: 'CRASHED', inner_exit: 134, outer_exit: 134 }));
  assert.throws(() => AN.admit('smoke', root), AN.Invalid);
  writeFileSync(join(outer, 'outer.json'), JSON.stringify(good));
  const t = AN.timing(AN.admit('smoke', root));
  assert.ok(t.projection_s > 12.5 && t.launch_estimate_s === Math.round(1.5 * t.projection_s * 1000) / 1000);
  assert.equal(t.match_cap_s, 21600);
  assert.equal(t.verdict, t.launch_estimate_s <= 21600 ? 'LAUNCH_OK' : 'NO_LAUNCH');
  assert.ok(!JSON.stringify(t).includes('score') && !JSON.stringify(t).includes('result'));
});

test('run refuses while the gate is closed, and creates nothing', () => {
  for (const stage of ['smoke', 'match']) {
    const r = spawnSync(process.execPath, [join(D.REPO_ROOT, 'tests/htb_integration/teacher_bench.mjs'), 'run', stage, '--teacher-python', 'x', '--clone-root', 'y', '--kill-s', String(D.KILL_S[stage])], { encoding: 'utf8' });
    assert.equal(r.status, 4);
    assert.match(r.stderr, /AUTHORIZED is closed/);
    assert.equal(existsSync(join(D.REPO_ROOT, D.RUN_REL[stage])), false);
  }
});

const CLONE = join(process.env.HOME ?? '', 'twixt-cache/htb-teacher/twixtbot-ui');
test('end to end through the real serve loop: TeacherClient, Python lockstep on twixtbot\'s Game, a fake network', { skip: !existsSync(join(CLONE, 'src/backend/twixt.py')) && 'pinned clone absent' }, async () => {
  const boot = [
    'import sys; sys.path.insert(0, "tests")',
    'from test_htb_bench_teacher import FakeTeacher, _twixt',
    'from scripts.GPU.alphazero import htb_bench_teacher as T',
    'sys.stdout.write(\'{"ok": true, "ready": true}\\n\'); sys.stdout.flush()',
    'sys.exit(T.serve(T.BenchTeacher(FakeTeacher(), _twixt()[0]), sys.stdin, sys.stdout))',
  ].join('\n');
  const teacher = D.TeacherClient.spawn(PY, ['-B', '-c', boot], { cwd: D.REPO_ROOT, env: { ...process.env, PYTHONPATH: '.' } });
  assert.equal((await teacher.next()).ready, true);
  const inference = new MirrorAveragingInference(fakeInference());
  const recs = [];
  for (const task of D.schedule('smoke')) {
    recs.push(await D.playGame({ stage: 'smoke', task, student: D.makeStudent(task.arm, inference, SMALL), teacher }));
  }
  await teacher.close();
  assert.equal(teacher.exited.code, 0);
  const openings = AN.deriveOpenings('smoke opening', 1);
  for (const r of recs) AN.validateGame(r, { stage: 'smoke', arm: r.arm, pair: 0, game: r.game, opening: openings[0] });
  // a stop in the Python lockstep reaches the driver as a BenchStop and the process exits 3
  const t2 = D.TeacherClient.spawn(PY, ['-B', '-c', boot], { cwd: D.REPO_ROOT, env: { ...process.env, PYTHONPATH: '.' } });
  await t2.next();
  await t2.request({ cmd: 'start', opening: [[10, 10], [5, 5], [12, 11], [5, 8]] });
  const bad = await t2.request({ cmd: 'apply', move: [10, 10] });
  assert.equal(bad.ok, false);
  assert.equal(bad.stop, 'illegal');
  await new Promise((resolve) => (t2.exited ? resolve() : t2.child.once('exit', resolve)));
  assert.equal(t2.exited.code, 3);
});
