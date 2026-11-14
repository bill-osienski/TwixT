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

const REAL = resolvePolicy({ difficulty: 'hard', hardMode: 'protocol' }); // 200 simulations, as shipped
const HEAD = 'a'.repeat(40);

/** A smoke evidence tree as the wrapper and driver would leave it, games played at the card's 200-sim budget. */
async function smokeTree() {
  const root = mkdtempSync(join(tmpdir(), 'bench-adm-'));
  const run = join(root, AN.DIRS.smoke.run);
  const outer = join(root, AN.DIRS.smoke.outer);
  mkdirSync(join(run, 'games'), { recursive: true });
  mkdirSync(outer, { recursive: true });
  const inference = new MirrorAveragingInference(fakeInference());
  const games = {};
  for (const task of D.schedule('smoke')) {
    const rec = await D.playGame({ stage: 'smoke', task, student: D.makeStudent(task.arm, inference, REAL), teacher: fakeTeacher() });
    games[`${task.arm}_pair_0000_game_${task.game}.json`] = rec;
  }
  const record = { schema: 'htb-teacher-bench-record/1', written_by: 'script', status: 'COMPLETED', stage: 'smoke', games: 4, startup_seconds: 12.5, kill_s: 1800, head: HEAD, pid: 4242,
    student: { model_id: '78c16a31c3dba37f', policy: REAL }, runtime: { node: process.version, onnxruntime_node: '1.23.2', teacher_env: { python: '3.13.15', tensorflow: '2.21.0', numpy: '2.5.2' } } };
  const outerRec = { inner_exit: 0, outcome: 'COMPLETED', outer_exit: 0, gate_restored: true, survivors: '', record_by: 'script', wall_seconds: 300, term_s: 1790, grace_s: 10, head: HEAD };
  const write = ({ g = games, r = record, o = outerRec, pgid = '4242\n' } = {}) => {
    for (const [name, rec] of Object.entries(g)) writeFileSync(join(run, 'games', name), JSON.stringify(rec));
    writeFileSync(join(run, 'record.json'), JSON.stringify(r));
    writeFileSync(join(outer, 'outer.json'), JSON.stringify(o));
    writeFileSync(join(outer, 'pgid.txt'), pgid);
  };
  return { root, games, record, outerRec, write };
}

test('smoke admission binds the evidence to its run; the timing estimate reads wall times only', async () => {
  const t = await smokeTree();
  t.write();
  const est = AN.timing(AN.admit('smoke', t.root));
  assert.ok(est.projection_s > 12.5 && Math.abs(est.launch_estimate_s - 1.5 * est.projection_s) < 0.002); // both rounded to ms
  assert.equal(est.match_cap_s, 21600);
  assert.equal(est.verdict, est.launch_estimate_s <= 21600 ? 'LAUNCH_OK' : 'NO_LAUNCH');
  assert.match(est.smoke_outcomes, /excluded from scoring and from every report/);
  assert.ok(!/score|result|"W"|"L"/.test(JSON.stringify(est)));
  // each binding failure, against the clean control above
  const bad = [
    ['missing outer', () => t.write({ o: null })],
    ['crashed', () => t.write({ o: { ...t.outerRec, outcome: 'CRASHED', inner_exit: 134, outer_exit: 134 } })],
    ['another launch commit', () => t.write({ o: { ...t.outerRec, head: 'b'.repeat(40) } })],
    ['pgid is not the driver', () => t.write({ pgid: '4243\n' })],
    ['the match ceiling on the smoke', () => t.write({ o: { ...t.outerRec, term_s: 21590 } })],
    ['record ceiling', () => t.write({ r: { ...t.record, kill_s: 21600 } })],
    ['onnxruntime-node', () => t.write({ r: { ...t.record, runtime: { ...t.record.runtime, onnxruntime_node: '1.24.4' } } })],
    ['teacher runtime', () => t.write({ r: { ...t.record, runtime: { ...t.record.runtime, teacher_env: { python: '3.13.15', tensorflow: '2.20.0', numpy: '2.5.2' } } } })],
    ['student budget', () => t.write({ r: { ...t.record, student: { ...t.record.student, policy: { ...REAL, nSims: 64 } } } })],
  ];
  for (const [what, corrupt] of bad) {
    corrupt();
    assert.throws(() => AN.admit('smoke', t.root), AN.Invalid, what);
    t.write();
    AN.admit('smoke', t.root); // the control admits again
  }
});

test('per-move evidence: one row per played ply, and exactly 200 simulations spent on every arm P student move', async () => {
  const t = await smokeTree();
  const name = 'P_pair_0000_game_0.json';
  const g = t.games[name];
  const sIdx = g.per_move.findIndex((m) => m.mover === 'student');
  const variants = {
    'a 64-simulation budget': { ...g, per_move: g.per_move.map((m, i) => (i === sIdx ? { ...m, nSimulations: 64 } : m)) },
    'fewer visits than the budget': { ...g, per_move: g.per_move.map((m, i) => (i === sIdx ? { ...m, visits: 64 } : m)) },
    'a duplicate row': { ...g, per_move: [...g.per_move.slice(0, 2), g.per_move[1], ...g.per_move.slice(3)] },
    'a missing row': { ...g, per_move: g.per_move.slice(1) },
    'an extra row': { ...g, per_move: [...g.per_move, g.per_move.at(-1)] },
    'a teacher row with a budget': { ...g, per_move: g.per_move.map((m) => (m.mover === 'teacher' ? { ...m, nSimulations: 200 } : m)) },
  };
  for (const [what, bad] of Object.entries(variants)) {
    t.write({ g: { ...t.games, [name]: bad } });
    assert.throws(() => AN.admit('smoke', t.root), AN.Invalid, what);
  }
  t.write();
  AN.admit('smoke', t.root);
  // the old 64-simulation fixture: the whole smoke played at 64 sims is refused, so it can no longer yield an estimate
  const inference = new MirrorAveragingInference(fakeInference());
  const task = D.schedule('smoke').find((x) => x.arm === 'P');
  const at64 = await D.playGame({ stage: 'smoke', task, student: D.makeStudent('P', inference, SMALL), teacher: fakeTeacher() });
  assert.ok(at64.per_move.some((m) => m.mover === 'student' && m.nSimulations === 64 && m.visits === 64));
  assert.throws(() => AN.validateGame(at64, { stage: 'smoke', arm: 'P', pair: 0, game: task.game, opening: AN.deriveOpenings('smoke opening', 1)[0] }), AN.Invalid);
});

test('runtime pins and committed smoke evidence: checked before any model loads', () => {
  assert.equal(D.ortVersion(), '1.23.2');
  assert.deepEqual(D.teacherEnvProblems({ python: '3.13.15', tensorflow: '2.21.0', numpy: '2.5.2' }), []);
  assert.equal(D.teacherEnvProblems({ python: '3.13.15', tensorflow: '2.20.0', numpy: '2.5.2' }).length, 1);
  assert.equal(D.teacherEnvProblems(undefined).length, 3);
  const ok = (cmd) => (cmd[0] === 'ls-files' && cmd[1] === '--') ? { status: 0, stdout: 'a\nb\nc\nd\n' } : { status: 0, stdout: '' };
  const fake = (over) => (...args) => over(args) ?? ok(args);
  assert.deepEqual(D.smokeEvidenceProblems(fake(() => null)), []);
  assert.ok(D.smokeEvidenceProblems(fake((a) => (a[0] === 'ls-files' && a[1] === '--error-unmatch' && a[3].endsWith('pgid.txt') ? { status: 1 } : null))).some((p) => /pgid.txt is not committed/.test(p)));
  assert.ok(D.smokeEvidenceProblems(fake((a) => (a[0] === 'diff' ? { status: 1 } : null))).some((p) => /differs from HEAD/.test(p)));
  assert.ok(D.smokeEvidenceProblems(fake((a) => (a[1] === '--others' ? { status: 0, stdout: 'x.json\n' } : null))).some((p) => /untracked/.test(p)));
  assert.ok(D.smokeEvidenceProblems(fake((a) => (a[0] === 'ls-files' && a[1] === '--' ? { status: 0, stdout: 'a\n' } : null))).some((p) => /1 committed smoke games/.test(p)));
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
    recs.push(await D.playGame({ stage: 'smoke', task, student: D.makeStudent(task.arm, inference, REAL), teacher }));
  }
  await teacher.close();
  assert.equal(teacher.exited.code, 0);
  const openings = AN.deriveOpenings('smoke opening', 1);
  for (const r of recs) {
    AN.validateGame(r, { stage: 'smoke', arm: r.arm, pair: 0, game: r.game, opening: openings[0] });
    assert.ok('tb_draw_flag_ply_reflected' in r);
  }
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

test('non-finite student output: finite priors with a NaN value stop both arms before a move is returned', async () => {
  let calls = 0;
  const nanValue = {
    async evaluate(_h, moves) {
      calls += 1;
      return { priors: new Map(moves.map((m) => [`${m[0]},${m[1]}`, 1 / moves.length])), value: Number.NaN };
    },
  };
  let s = new TwixtState();
  for (const m of [[10, 10], [5, 5], [12, 11], [5, 8]]) s = s.applyMove(m);
  for (const arm of ['N', 'P']) {
    calls = 0;
    const student = D.makeStudent(arm, nanValue, REAL);
    let returned = null;
    await assert.rejects(
      (async () => {
        returned = await student.move(s, { stage: 'match', pair: 0, game: 0 });
      })(),
      (e) => e instanceof D.BenchStop && e.step === 'student' && /non-finite student value/.test(e.message),
      `arm ${arm}`,
    );
    assert.equal(returned, null, `arm ${arm} returned a move`);
    assert.equal(calls, 1, `arm ${arm} evaluated past the first non-finite output`);
  }
  // and inside a game the stop reaches playGame, with no move applied
  const task = D.schedule('smoke').find((t) => t.arm === 'P' && t.studentColour === 'red');
  await assert.rejects(D.playGame({ stage: 'smoke', task, student: D.makeStudent('P', nanValue, REAL), teacher: fakeTeacher() }), (e) => e instanceof D.BenchStop && e.step === 'student');
});

test('non-finite student output: a NaN or missing prior also stops, and a finite evaluation passes through unchanged', async () => {
  const s = new TwixtState().applyMove([10, 10]);
  const moves = s.legalMoves();
  const mk = (f) => ({ async evaluate(_h, mv) { return f(mv); } });
  const g1 = D.finiteGuard(mk((mv) => ({ priors: new Map(mv.map((m, i) => [`${m[0]},${m[1]}`, i === 3 ? Number.NaN : 0.01])), value: 0.1 })));
  await assert.rejects(g1.evaluate(null, moves), (e) => e instanceof D.BenchStop && /prior/.test(e.message));
  const g2 = D.finiteGuard(mk((mv) => ({ priors: new Map(mv.slice(1).map((m) => [`${m[0]},${m[1]}`, 0.01])), value: 0.1 })));
  await assert.rejects(g2.evaluate(null, moves), (e) => e instanceof D.BenchStop && /missing/.test(e.message));
  const good = { priors: new Map(moves.map((m) => [`${m[0]},${m[1]}`, 0.01])), value: -0.2 };
  assert.equal(await D.finiteGuard(mk(() => good)).evaluate(null, moves), good);
});
