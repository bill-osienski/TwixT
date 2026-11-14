/**
 * Diagnostic O: model-free tests (fake evaluator only). No ONNX model, no teacher, no gate opened.
 * Run: node --test tests/htb_integration/test_diag_o.mjs
 */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

import { TwixtState } from '../../server/gameLogic.js';
import { MirrorAveragingInference } from '../../server/mirror_inference.js';
import { protocolMove } from '../../server/protocol_move.js';
import { resolvePolicy } from '../../server/readout_policy.js';
import { fakeInference } from '../fixtures/htb_integration/fake_evaluator.mjs';
import * as AN from './diag_o_analyse.mjs';
import * as GATE from './diag_o_authorization.mjs';
import * as D from './diag_o.mjs';

const SMALL = { nSims: 24, moveTemp: 1.0, search: 'protocol' };

/** A synthetic "saved game": the student's opening moves made by protocolMove itself (the product function). */
async function syntheticGame(inference, policy = SMALL, pair = 7) {
  const real = resolvePolicy({ difficulty: 'hard', hardMode: 'protocol' });
  let state = new TwixtState();
  const moves = [];
  const per_move = [];
  for (let k = 0; k < 20; k++) {
    let mv;
    if (k % 2 === 1) {
      const seeds = { search: 1000 + 17 * k + pair, readout: 5000 + 31 * k + pair };
      // protocolMove refuses any temperature but 1.0 and takes nSims from the policy; use a small budget for speed.
      const out = await protocolMove({ gameState: TwixtState.fromDict(state.toDict()), policy: { ...real, nSims: policy.nSims }, inference, seeds });
      mv = out.moveKey.split(',').map(Number);
      per_move.push({ ply: k, player: 'new_hard', seeds });
    } else {
      const legal = state.legalMoves();
      mv = legal[(37 * k + 11 * pair) % legal.length];
      per_move.push({ ply: k, player: 'legacy_hard', seeds: null });
    }
    moves.push(mv);
    state = state.applyMove(mv);
  }
  return { stage: 'match', new_colour: 'black', game_in_pair: 1, pair_index: pair, new_model_id: D.STUDENT_MODEL_ID, moves, per_move };
}

test('the gate is closed in the committed state', () => {
  assert.equal(GATE.DIAG_O_AUTHORIZED, false);
});

test('visitLeader: most visits, then higher prior, then (row, col)', () => {
  const counts = new Map([['5,5', 10], ['3,4', 12], ['2,9', 12], ['1,1', 3]]);
  assert.equal(D.visitLeader(counts, new Map([['3,4', 0.2], ['2,9', 0.1]])), '3,4');
  assert.equal(D.visitLeader(counts, new Map([['3,4', 0.1], ['2,9', 0.1]])), '2,9');
  assert.equal(D.visitRank(counts, '5,5'), 3);
  assert.equal(D.visitRank(counts, '2,9'), 1);
  assert.equal(D.visitRank(counts, '9,9'), 5);
});

test('studentStage reproduces protocolMove seed for seed and yields 10 Black positions', async () => {
  const inference = new MirrorAveragingInference(fakeInference());
  const game = await syntheticGame(inference);
  const recs = [];
  for await (const r of D.studentStage(game, inference, { policy: SMALL, simsDeep: 48 })) recs.push(r);
  assert.deepEqual(recs.map((r) => r.ply), D.PLIES);
  for (const r of recs) {
    assert.equal(r.played, `${game.moves[r.ply][0]},${game.moves[r.ply][1]}`);
    assert.equal(r.search200.total_visits, SMALL.nSims);
    assert.equal(r.search800.total_visits, 48);
    assert.ok(r.candidates.includes(r.played) && r.candidates.includes(r.leader200) && r.candidates.includes(r.leader800));
    for (const c of r.candidates) assert.ok(Number.isFinite(r.student_raw.after_value_black[c]));
    assert.ok(r.search200.played_share > 0 && r.search200.played_share <= 1);
    assert.deepEqual(r.prefix, game.moves.slice(0, r.ply));
  }
});

test('the 800 search uses the same seed: at the same budget it equals the 200 search exactly', async () => {
  const inference = new MirrorAveragingInference(fakeInference());
  const game = await syntheticGame(inference, SMALL, 3);
  for await (const r of D.studentStage(game, inference, { policy: SMALL, simsDeep: SMALL.nSims })) {
    assert.deepEqual(r.search800.top10, r.search200.top10);
    assert.equal(r.leader800, r.leader200);
    assert.equal(r.search800.root_value_black, r.search200.root_value_black);
  }
});

test('a reproduction mismatch stops the stage', async () => {
  const inference = new MirrorAveragingInference(fakeInference());
  const game = await syntheticGame(inference);
  const legal = (() => {
    let s = new TwixtState();
    for (let k = 0; k < 5; k++) s = s.applyMove(game.moves[k]);
    return s.legalMoves();
  })();
  const other = legal.find((m) => `${m}` !== `${game.moves[5]}`);
  game.moves[5] = other;
  await assert.rejects(async () => {
    for await (const _ of D.studentStage(game, inference, { policy: SMALL, simsDeep: 30 })) void _;
  }, (e) => e instanceof D.DiagStop && e.step === 'reproduction' && /ply 5/.test(e.message));
});

test('a game that is not the student as Black is refused', async () => {
  const inference = new MirrorAveragingInference(fakeInference());
  const game = await syntheticGame(inference);
  await assert.rejects(async () => {
    for await (const _ of D.studentStage({ ...game, new_colour: 'red' }, inference, { policy: SMALL })) void _;
  }, (e) => e instanceof D.DiagStop && e.step === 'input');
});

test('run refuses while the gate is closed, and creates nothing', () => {
  // Since the run (2026-11-14, a1d22c0d) its evidence directory exists; "nothing" now means nothing new.
  const dir = join(D.REPO_ROOT, D.RUN_REL);
  const before = existsSync(dir) ? readdirSync(dir).sort() : null;
  const r = spawnSync(process.execPath, [join(D.REPO_ROOT, 'tests/htb_integration/diag_o.mjs'), 'run', '--teacher-python', 'x', '--clone-root', 'y', '--kill-s', '1'], { encoding: 'utf8' });
  assert.equal(r.status, 4);
  assert.match(r.stderr, /DIAG_O_AUTHORIZED is closed/);
  assert.deepEqual(existsSync(dir) ? readdirSync(dir).sort() : null, before);
});

test('the frozen groups: 110 Black games, 96 W / 12 L / 2 D, the replay table', () => {
  const g = JSON.parse(readFileSync(join(D.REPO_ROOT, D.GROUPS_REL), 'utf8'));
  assert.equal(g.n_games, 110);
  assert.deepEqual([g.W, g.L, g.D], [96, 12, 2]);
  const b2 = g.games.filter((x) => x.race19 === 'behind_2plus');
  assert.deepEqual([b2.filter((x) => x.outcome === 'W').length, b2.filter((x) => x.outcome === 'L').length, b2.filter((x) => x.outcome === 'D').length], [31, 12, 2]);
});

// ---------------------------------------------------------------- analysis

test('aucLowerIsLoss, wilson and the bootstrap are correct and reproducible', () => {
  assert.equal(AN.aucLowerIsLoss([0, 1], [2, 3]), 1);
  assert.equal(AN.aucLowerIsLoss([2, 3], [0, 1]), 0);
  assert.equal(AN.aucLowerIsLoss([1], [1]), 0.5);
  const w = AN.wilson(8, 12);
  assert.ok(w.lo > 0.39 && w.lo < 0.40 && w.hi > 0.86 && w.hi < 0.87);
  const L = [1, 2, 3, 4];
  const W = [0, 0, 1, 0, 1];
  const a = AN.bootstrap(L, W, (x, y) => AN.mean(x) - AN.mean(y), { B: 500 });
  const b = AN.bootstrap(L, W, (x, y) => AN.mean(x) - AN.mean(y), { B: 500 });
  assert.deepEqual(a, b);
  assert.ok(a.lo > 0);
});

function cmpWith({ lowL = 0, lowW = 0, rlow = [0, 0, { lo: -1, hi: 1 }], bdiff = [0, 0, { lo: -1, hi: 1 }], anyB = 0, aucT = 0.5, aucS = 0.5, dlo = -0.1 }) {
  return {
    any_R_low: { L: [lowL, 12], W: [lowW, 96] },
    any_B_diff: { L: [anyB, 12], W: [0, 96] },
    R_low: { mean_L: rlow[0], mean_W: rlow[1], ci95: rlow[2] },
    B_diff: { mean_L: bdiff[0], mean_W: bdiff[1], ci95: bdiff[2] },
    auc: { v19_teacher_raw: { auc: aucT }, v19_student_raw: { auc: aucS }, v19_teacher_minus_student_raw: { ci95: { lo: dlo } } },
  };
}

test('signals: each fires only with all its conditions; priority readout > teacher > budget', () => {
  assert.equal(AN.signals(cmpWith({}), 12).recommended, null);
  const r = AN.signals(cmpWith({ lowL: 8, rlow: [1.5, 0.5, { lo: 0.2, hi: 2 }] }), 12);
  assert.equal(r.readout, true);
  assert.equal(AN.signals(cmpWith({ lowL: 7, rlow: [1.5, 0.5, { lo: 0.2, hi: 2 }] }), 12).readout, false);
  assert.equal(AN.signals(cmpWith({ lowL: 8, rlow: [1.5, 0.8, { lo: 0.2, hi: 2 }] }), 12).readout, false); // ratio < 2
  assert.equal(AN.signals(cmpWith({ lowL: 8, rlow: [1.5, 0.5, { lo: -0.1, hi: 2 }] }), 12).readout, false); // CI
  const t = AN.signals(cmpWith({ aucT: 0.8, aucS: 0.55, dlo: 0.05 }), 12);
  assert.equal(t.teacher, true);
  assert.equal(t.recommended, 'teacher');
  assert.equal(AN.signals(cmpWith({ aucT: 0.8, aucS: 0.65, dlo: 0.05 }), 12).teacher, false);
  assert.equal(AN.signals(cmpWith({ aucT: 0.8, aucS: 0.55, dlo: -0.01 }), 12).teacher, false);
  const both = AN.signals(cmpWith({ lowL: 9, rlow: [2, 0.5, { lo: 0.5, hi: 3 }], aucT: 0.9, aucS: 0.5, dlo: 0.1, anyB: 10, bdiff: [3, 1, { lo: 1, hi: 4 }] }), 12);
  assert.deepEqual(both.fired, ['readout', 'teacher', 'budget']);
  assert.equal(both.recommended, 'readout');
});

test('analyse runs end to end on synthetic stage outputs, with draws kept separate', async () => {
  const inference = new MirrorAveragingInference(fakeInference());
  const students = [];
  const games = [];
  const outcomes = ['W', 'W', 'L', 'D', 'W', 'L'];
  for (let i = 0; i < outcomes.length; i++) {
    const game = await syntheticGame(inference, SMALL, 20 + i);
    for await (const r of D.studentStage(game, inference, { policy: SMALL, simsDeep: 32 })) students.push(r);
    games.push({ pair: 20 + i, outcome: outcomes[i], race19: i % 2 ? 'behind_2plus' : 'ahead' });
  }
  const teacher = {
    records: students.map((s, j) => ({
      pair: s.pair, ply: s.ply, value_black: Math.sin(j), top_move: s.leader200, top_prob: 0.3,
      prob: Object.fromEntries(s.candidates.map((c) => [c, 0.1])), rank: Object.fromEntries(s.candidates.map((c) => [c, 2])),
      after_value_black: Object.fromEntries(s.candidates.map((c, k) => [c, Math.cos(j + k)])),
    })),
  };
  const res = AN.analyse(students, teacher, { n_games: games.length, games }, { B: 200 });
  assert.deepEqual(res.counts, { L: 2, W: 3, D: 1, B2_L: 1, B2_W: 1, B2_D: 1 }); // behind: i = 1 (W), 3 (D), 5 (L)
  assert.equal(res.all_black_games.R_low.n_L, 2);
  assert.equal(res.all_black_games.v19_student_raw.D_values.length, 1);
  assert.ok(['readout', 'teacher', 'budget', null].includes(res.signals.recommended));
  assert.throws(() => AN.analyse(students.slice(1), teacher, { n_games: games.length, games }, { B: 50 }), /positions/);
});
