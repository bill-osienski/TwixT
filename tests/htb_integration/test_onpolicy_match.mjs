/**
 * H-OP match: model-free tests (fake evaluators at the real 200-simulation budget). No ONNX model loads, no gate opens.
 * Run: node --test tests/htb_integration/test_onpolicy_match.mjs
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

import { TwixtState } from '../../server/gameLogic.js';
import { MirrorAveragingInference } from '../../server/mirror_inference.js';
import { protocolMove } from '../../server/protocol_move.js';
import { resolvePolicy } from '../../server/readout_policy.js';
import { fakeInference } from '../fixtures/htb_integration/fake_evaluator.mjs';
import * as AN from './onpolicy_match_analyse.mjs';
import * as GATE from './onpolicy_authorization.mjs';
import * as D from './onpolicy_match.mjs';
import * as TB from './teacher_bench.mjs';

const REAL = resolvePolicy({ difficulty: 'hard', hardMode: 'protocol' }); // 200 simulations, as shipped
const SMALL = { ...REAL, nSims: 64 };
const HEAD = 'a'.repeat(40);
const sha = (b) => createHash('sha256').update(b).digest('hex');
const OPENINGS = D.loadOpenings();

/** A second fake network: the same shape as fakeInference with different numbers, so the two sides differ. */
const otherFake = () => {
  const base = fakeInference();
  return {
    async evaluate(hwc, moves) {
      const out = await base.evaluate(hwc, moves);
      const priors = new Map();
      let s = 0;
      for (const [k, v] of out.priors) s += v * v;
      for (const [k, v] of out.priors) priors.set(k, (v * v) / s);
      return { priors, value: -out.value / 2 };
    },
  };
};
const players = (policy = REAL) => ({
  challenger: D.makePlayer('challenger', new MirrorAveragingInference(fakeInference()), policy),
  current: D.makePlayer('current', new MirrorAveragingInference(otherFake()), policy),
});

/**
 * A FAST stub with a search player's evidence shape, for the admission tests only: it plays a hash-chosen legal move
 * and records the derived seeds and a 200 / 200 budget, as the analyser requires. The analyser validates records,
 * not searches; one real 200-simulation game below covers the search itself (fake games run to ~570 plies at ~130 ms
 * per move, too slow to build whole evidence trees from).
 */
const stubPlayer = (salt) => ({
  async move(state, { stage, pair, game }) {
    const legal = [...state.legalMoves()].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const h = createHash('sha256').update(`${salt}/${pair}/${game}/${state.ply}`).digest();
    const mv = legal[h.readUInt32BE(0) % legal.length];
    return { moveKey: `${mv[0]},${mv[1]}`, extra: { seeds: D.moveSeeds(stage, pair, game, state.ply), nSimulations: 200, visits: 200 } };
  },
});
const stubs = () => ({ challenger: stubPlayer('c'), current: stubPlayer('k') });

test('the gates are closed in the committed state', () => {
  assert.equal(GATE.ONPOLICY_MATCH_SMOKE_AUTHORIZED, false);
  assert.equal(GATE.ONPOLICY_MATCH_AUTHORIZED, false);
});

test('openings: the pinned file, its sizes, and the analyser re-derives every listed opening', () => {
  const raw = readFileSync(join(D.REPO_ROOT, D.OPENINGS_REL));
  assert.equal(sha(raw), D.OPENINGS_SHA256);
  assert.equal(OPENINGS.match.length, 272);
  assert.equal(OPENINGS.match_smoke.length, 2);
  assert.deepEqual(AN.stageOpenings('match'), OPENINGS.match);
  assert.deepEqual(AN.stageOpenings('smoke'), OPENINGS.match_smoke);
  // a tampered file is refused by both sides
  const root = mkdtempSync(join(tmpdir(), 'op-open-'));
  mkdirSync(join(root, 'docs/superpowers/evidence/htb-onpolicy-openings'), { recursive: true });
  writeFileSync(join(root, D.OPENINGS_REL), raw.toString('utf8').replace('"match"', '"match" '));
  assert.throws(() => D.loadOpenings(root), D.MatchRefusal);
  assert.throws(() => AN.stageOpenings('match', root), AN.Invalid);
});

test('openings: Node re-derives the whole Python-built file, exclusions included (benchmark, G3, generation, smokes)', () => {
  const seq = (m) => JSON.stringify(m);
  const refl = (m) => JSON.stringify(m.map(([r, c]) => [r, 23 - c]));
  const excl = new Set();
  for (const [kind, n] of [['opening', 200], ['smoke opening', 1]]) for (const o of TB.openingList(kind, n)) excl.add(seq(o.moves));
  for (const d of ['docs/superpowers/evidence/htb-g3-match/games', 'docs/superpowers/evidence/htb-g3-smoke/games']) {
    for (const f of readdirSync(join(D.REPO_ROOT, d)).filter((x) => x.endsWith('.json'))) {
      excl.add(seq(JSON.parse(readFileSync(join(D.REPO_ROOT, d, f), 'utf8')).moves.slice(0, 4)));
    }
  }
  const list = (kind, n) => {
    const out = [];
    for (let i = 0; out.length < n; i++) {
      const m = AN.deriveOpening(kind, i);
      if (excl.has(seq(m)) || excl.has(refl(m))) continue;
      excl.add(seq(m));
      out.push({ index: i, moves: m });
    }
    return out;
  };
  for (const [k, kind, n] of [['gen', 'gen opening', 4000], ['gen_smoke', 'gen smoke opening', 40], ['match', 'match opening', 272], ['match_smoke', 'match smoke opening', 2]]) {
    assert.deepEqual(list(kind, n), OPENINGS[k], k);
  }
});

test('openings (amendment 1): Node re-derives the generation extension -- the gen sequence continued past the frozen 4,000, distinct from every list', () => {
  const EXT_REL = 'docs/superpowers/evidence/htb-onpolicy-openings/gen-extension.json';
  const raw = readFileSync(join(D.REPO_ROOT, EXT_REL));
  assert.equal(sha(raw), '578d34f371b238a9919a6eeb9f71c3b30eb286b7b08e95d1722e1fc17f270664');
  const ext = JSON.parse(raw.toString('utf8'));
  assert.equal(ext.base_openings_sha256, D.OPENINGS_SHA256);
  assert.equal(ext.first_position, 4000);
  const seq = (m) => JSON.stringify(m);
  const refl = (m) => JSON.stringify(m.map(([r, c]) => [r, 23 - c]));
  const excl = new Set();
  for (const [kind, n] of [['opening', 200], ['smoke opening', 1]]) for (const o of TB.openingList(kind, n)) excl.add(seq(o.moves));
  for (const d of ['docs/superpowers/evidence/htb-g3-match/games', 'docs/superpowers/evidence/htb-g3-smoke/games']) {
    for (const f of readdirSync(join(D.REPO_ROOT, d)).filter((x) => x.endsWith('.json'))) {
      excl.add(seq(JSON.parse(readFileSync(join(D.REPO_ROOT, d, f), 'utf8')).moves.slice(0, 4)));
    }
  }
  for (const k of ['gen', 'gen_smoke', 'match', 'match_smoke']) for (const o of OPENINGS[k]) excl.add(seq(o.moves));
  const out = [];
  for (let i = OPENINGS.gen.at(-1).index + 1; out.length < 2500; i++) {
    const m = AN.deriveOpening('gen opening', i);
    if (excl.has(seq(m)) || excl.has(refl(m))) continue;
    excl.add(seq(m));
    out.push({ index: i, moves: m });
  }
  assert.deepEqual(out, ext.gen_extension);
});

test('schedule: 544 match games and 4 smoke games; pair p plays opening p, challenger Red in game 0', () => {
  const m = D.schedule('match', OPENINGS);
  assert.equal(m.length, 544);
  for (const t of m) {
    assert.equal(t.challengerColour, t.game === 0 ? 'red' : 'black');
    assert.deepEqual(t.opening, OPENINGS.match[t.pair].moves);
    assert.equal(t.openingIndex, OPENINGS.match[t.pair].index);
  }
  assert.equal(D.schedule('smoke', OPENINGS).length, 4);
  assert.deepEqual(D.schedule('smoke', OPENINGS)[2].opening, OPENINGS.match_smoke[1].moves);
});

test('seeds: the formula, the analyser re-derives them, all distinct and below 2^48', () => {
  const h = sha('htb-onpolicy/20261114/match/pair 3/game 1/ply 17');
  assert.deepEqual(D.moveSeeds('match', 3, 1, 17), { search: parseInt(h.slice(0, 12), 16), readout: parseInt(h.slice(12, 24), 16) });
  const hs = sha('htb-onpolicy/20261114/match smoke/pair 0/game 0/ply 4');
  assert.equal(D.moveSeeds('smoke', 0, 0, 4).search, parseInt(hs.slice(0, 12), 16));
  const seen = new Set();
  for (const [stage, pairs] of [['smoke', 2], ['match', 272]]) {
    for (let p = 0; p < pairs; p++) {
      for (const g of [0, 1]) {
        for (let ply = 4; ply < 576; ply++) {
          const s = D.moveSeeds(stage, p, g, ply);
          if (ply % 113 === 0) assert.deepEqual(s, AN.deriveSeeds(stage, p, g, ply));
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

test('a player is protocolMove with the derived seeds, move for move, and records 200 budgeted and 200 spent', async () => {
  const inference = new MirrorAveragingInference(fakeInference());
  const pl = D.makePlayer('challenger', inference, REAL);
  let s = new TwixtState();
  for (const m of OPENINGS.match[0].moves) s = s.applyMove(m);
  for (let k = 0; k < 3; k++) {
    const ours = await pl.move(s, { stage: 'match', pair: 0, game: 0 });
    const ref = await protocolMove({ gameState: TwixtState.fromDict(s.toDict()), policy: REAL, inference, seeds: D.moveSeeds('match', 0, 0, s.ply) });
    assert.equal(ours.moveKey, ref.moveKey);
    assert.equal(ours.extra.nSimulations, 200);
    assert.equal(ours.extra.visits, 200);
    s = s.applyMove(ours.moveKey.split(',').map(Number));
  }
});

test('the non-finite guard stops either side before a move is returned', async () => {
  const nan = { async evaluate(_h, moves) { return { priors: new Map(moves.map((m) => [`${m[0]},${m[1]}`, 1 / moves.length])), value: Number.NaN }; } };
  let s = new TwixtState();
  for (const m of OPENINGS.match[0].moves) s = s.applyMove(m);
  for (const who of ['challenger', 'current']) {
    await assert.rejects(D.makePlayer(who, nan, REAL).move(s, { stage: 'match', pair: 0, game: 0 }), (e) => e instanceof D.MatchStop && e.step === who);
  }
  const missing = { async evaluate(_h, moves) { return { priors: new Map(moves.slice(1).map((m) => [`${m[0]},${m[1]}`, 0.01])), value: 0 }; } };
  await assert.rejects(D.finiteGuard(missing, 'current').evaluate(null, s.legalMoves()), /missing/);
});

/** A smoke or match evidence tree as the wrapper and driver would leave it; games played at 200 sims with fakes. */
async function evidenceTree(stage, nPairs) {
  const root = mkdtempSync(join(tmpdir(), 'op-adm-'));
  const run = join(root, AN.DIRS[stage].run);
  const outer = join(root, AN.DIRS[stage].outer);
  mkdirSync(join(run, 'games'), { recursive: true });
  mkdirSync(outer, { recursive: true });
  mkdirSync(join(root, 'docs/superpowers/evidence/htb-onpolicy-openings'), { recursive: true });
  writeFileSync(join(root, D.OPENINGS_REL), readFileSync(join(D.REPO_ROOT, D.OPENINGS_REL)));
  mkdirSync(join(root, 'docs/superpowers/evidence/htb-onpolicy-export'), { recursive: true });
  const challenger = { model_id: '0123456789abcdef', dir: '/tmp/x/0123456789abcdef', graph_sha256: 'g'.repeat(64), external_data_sha256: 'd'.repeat(64), source_sha256: 's'.repeat(64) };
  const exportRecord = { kind: 'htb-onpolicy-export-record/1', status: 'COMPLETED', verdict: 'PARITY_PASS', challenger, head: 'c'.repeat(40), pid: 5151, kill_s: 3600 };
  writeFileSync(join(root, D.EXPORT_REL), JSON.stringify(exportRecord));
  putExportOuter(root, exportRecord);
  const games = {};
  const pl = stubs();
  for (const task of D.schedule(stage, OPENINGS).filter((t) => t.pair < nPairs)) {
    games[`pair_${String(task.pair).padStart(4, '0')}_game_${task.game}.json`] = await D.playGame({ stage, task, players: pl });
  }
  const record = { schema: 'htb-onpolicy-match-record/1', written_by: 'script', status: 'COMPLETED', stage, games: Object.keys(games).length, startup_seconds: 20.5,
    kill_s: AN.KILL_S[stage], head: HEAD, pid: 4242, challenger: { ...challenger, onnx_sha256: challenger.graph_sha256 }, current: { model_id: '78c16a31c3dba37f' },
    policy: REAL, openings_sha256: D.OPENINGS_SHA256, runtime: { node: process.version, onnxruntime_node: '1.23.2' } };
  const outerRec = { inner_exit: 0, outcome: 'COMPLETED', outer_exit: 0, gate_restored: true, survivors: '', record_by: 'script', wall_seconds: 300, term_s: AN.KILL_S[stage] - 10, grace_s: 10, head: HEAD };
  const write = ({ g = games, r = record, o = outerRec, pgid = '4242\n' } = {}) => {
    for (const [name, rec] of Object.entries(g)) writeFileSync(join(run, 'games', name), JSON.stringify(rec));
    writeFileSync(join(run, 'record.json'), JSON.stringify(r));
    writeFileSync(join(outer, 'outer.json'), JSON.stringify(o));
    writeFileSync(join(outer, 'pgid.txt'), pgid);
  };
  return { root, games, record, outerRec, write, exportRecord };
}

test('end to end: one real game at 200 simulations, both sides fake networks through protocolMove, validates', async () => {
  const task = D.schedule('smoke', OPENINGS)[1]; // the challenger as Black
  const g = await D.playGame({ stage: 'smoke', task, players: players() });
  let s = new TwixtState();
  for (const m of g.moves) s = s.applyMove(m);
  assert.ok(s.isTerminal());
  assert.ok(g.per_move.every((m) => m.nSimulations === 200 && m.visits === 200));
  assert.ok(g.per_move.some((m) => m.mover === 'challenger') && g.per_move.some((m) => m.mover === 'current'));
  assert.equal(AN.validateGame(g, { stage: 'smoke', pair: 0, game: 1, opening: OPENINGS.match_smoke[0] }), g);
  // the same game re-scored as if played with the challenger Red is refused
  assert.throws(() => AN.validateGame({ ...g, game: 0 }, { stage: 'smoke', pair: 0, game: 0, opening: OPENINGS.match_smoke[0] }), AN.Invalid);
});

test('smoke admission binds the evidence to its run; timing reads wall times only; each binding failure refuses', async () => {
  const t = await evidenceTree('smoke', 2);
  t.write();
  const adm = AN.admit('smoke', t.root);
  assert.equal(adm.games.length, 4);
  const est = AN.timing(adm);
  assert.equal(est.match_cap_s, 64800);
  assert.equal(est.match_games, 544);
  assert.ok(Math.abs(est.launch_estimate_s - 1.5 * est.projection_s) < 0.002);
  assert.equal(est.verdict, 'LAUNCH_OK');
  assert.ok(!/score|result|"W"|"L"/.test(JSON.stringify(est)));
  // EXCEEDS: the same admission with a long startup gives NO_LAUNCH
  assert.equal(AN.timing({ ...adm, record: { ...adm.record, startup_seconds: 50000 } }).verdict, 'NO_LAUNCH');
  const bad = [
    ['missing outer', () => t.write({ o: null })],
    ['crashed', () => t.write({ o: { ...t.outerRec, outcome: 'CRASHED', inner_exit: 134, outer_exit: 134 } })],
    ['another launch commit', () => t.write({ o: { ...t.outerRec, head: 'b'.repeat(40) } })],
    ['pgid is not the driver', () => t.write({ pgid: '4243\n' })],
    ['the match ceiling on the smoke', () => t.write({ o: { ...t.outerRec, term_s: 64790 } })],
    ['record ceiling', () => t.write({ r: { ...t.record, kill_s: 64800 } })],
    ['onnxruntime-node', () => t.write({ r: { ...t.record, runtime: { ...t.record.runtime, onnxruntime_node: '1.24.4' } } })],
    ['budget', () => t.write({ r: { ...t.record, policy: { ...REAL, nSims: 64 } } })],
    ['another challenger', () => t.write({ r: { ...t.record, challenger: { ...t.record.challenger, model_id: 'ffffffffffffffff' } } })],
    ['current as challenger', () => t.write({ r: { ...t.record, current: { model_id: 'ffffffffffffffff' } } })],
  ];
  for (const [what, corrupt] of bad) {
    corrupt();
    assert.throws(() => AN.admit('smoke', t.root), AN.Invalid, what);
    t.write();
    AN.admit('smoke', t.root);
  }
  // the handoff repair: an export that completed inside but timed out outside fails admission at analysis too
  putExportOuter(t.root, t.exportRecord, OUTER_TIMEOUT);
  assert.throws(() => AN.admit('smoke', t.root), /export outer outcome "TIMEOUT"/);
  putExportOuter(t.root, t.exportRecord);
  AN.admit('smoke', t.root);
});

test('per-move evidence: a 64-sim or 199-visit row, a missing, duplicate or extra row, or wrong seeds are INVALID', async () => {
  const t = await evidenceTree('smoke', 1);
  const name = 'pair_0000_game_0.json';
  const g = t.games[name];
  const iC = g.per_move.findIndex((m) => m.mover === 'current');
  const variants = {
    'a 64-simulation budget': { ...g, per_move: g.per_move.map((m, i) => (i === 0 ? { ...m, nSimulations: 64 } : m)) },
    '199 visits spent': { ...g, per_move: g.per_move.map((m, i) => (i === iC ? { ...m, visits: 199 } : m)) },
    'a duplicate row': { ...g, per_move: [...g.per_move.slice(0, 2), g.per_move[1], ...g.per_move.slice(3)] },
    'a missing row': { ...g, per_move: g.per_move.slice(1) },
    'an extra row': { ...g, per_move: [...g.per_move, g.per_move.at(-1)] },
    'wrong seeds': { ...g, per_move: g.per_move.map((m, i) => (i === 1 ? { ...m, seeds: D.moveSeeds('match', 0, 0, m.ply) } : m)) },
    'a wrong score': { ...g, challenger_score: g.challenger_score === 1 ? 0 : 1 },
  };
  const ctx = { stage: 'smoke', pair: 0, game: 0, opening: OPENINGS.match_smoke[0] };
  assert.equal(AN.validateGame(g, ctx), g);
  for (const [what, v] of Object.entries(variants)) assert.throws(() => AN.validateGame(v, ctx), AN.Invalid, what);
  // a real protocolMove at 64 sims records 64 / 64, and such a row is refused
  let s = new TwixtState();
  for (const m of OPENINGS.match_smoke[0].moves) s = s.applyMove(m);
  const p64 = await players(SMALL).challenger.move(s, { stage: 'smoke', pair: 0, game: 0 });
  assert.equal(p64.extra.nSimulations, 64);
  assert.equal(p64.extra.visits, 64);
  const row64 = { ...g, per_move: g.per_move.map((m, i) => (i === 0 ? { ...m, ...p64.extra } : m)) };
  assert.throws(() => AN.validateGame(row64, ctx), AN.Invalid);
});

test('the pass rule: 625/1088 passes and 624/1088 does not; analyse scores pairs and reads the verdict precisely', () => {
  assert.equal(AN.passHalfWidth(272).toFixed(6), Math.sqrt(Math.log(20) / 544).toFixed(6));
  assert.equal(AN.twoSidedHalfWidth(272).toFixed(4), '0.0823');
  assert.equal(AN.passes(625 / 1088), true);
  assert.equal(AN.passes(624 / 1088), false);
  const mkGames = (quarters) => {
    // quarters = total pair-score in units of 1/4 (sum of 2 x game scores in halves); spread over 272 pairs
    const games = [];
    let left = quarters;
    for (let p = 0; p < 272; p++) {
      const q = Math.min(4, left); // pair score q/4: games scored (a, b) with a + b = q/2
      left -= q;
      const a = q >= 2 ? 1 : q / 2;
      const b = q / 2 - a;
      for (const [game, sc] of [[0, a], [1, b]]) {
        games.push({ pair: p, game, challenger_colour: game ? 'black' : 'red', challenger_score: sc, result: sc === 0.5 ? 'draw' : 'x', plies: 40, moves: [[p, game]], per_move: [{ mover: 'challenger', ms: 100 }, { mover: 'current', ms: 100 }] });
      }
    }
    assert.equal(left, 0);
    return games;
  };
  const pass = AN.analyse({ games: mkGames(625) });
  assert.equal(pass.mean_pair_score, 625 / 1088);
  assert.equal(pass.verdict, 'PASS');
  assert.match(pass.reading, /above 0\.50 under this protocol/);
  assert.match(pass.reading, /whole update/);
  assert.match(pass.reading, /does not establish a gain of 0\.60/);
  const non = AN.analyse({ games: mkGames(624) });
  assert.equal(non.verdict, 'NON_PASS');
  assert.match(non.reading, /not evidence of no gain/);
  assert.equal(pass.descriptive.two_sided_95_half_width.toFixed(4), '0.0823');
  assert.equal(pass.descriptive.distinct_games, 544);
});

test('the search-seconds guard: above 1.10x the current hard is INVALID and nothing is scored', () => {
  const games = [];
  for (let p = 0; p < 272; p++) for (const game of [0, 1]) games.push({ pair: p, game, challenger_colour: game ? 'black' : 'red', challenger_score: 1, result: 'x', plies: 40, moves: [[p, game]], per_move: [{ mover: 'challenger', ms: 111 }, { mover: 'current', ms: 100 }] });
  const a = AN.analyse({ games });
  assert.equal(a.verdict, 'INVALID');
  assert.ok(!('mean_pair_score' in a) && !('challenger' in a));
  assert.ok(a.search_seconds_guard.ratio > 1.1);
  for (const g of games) g.per_move[0].ms = 110;
  assert.equal(AN.analyse({ games }).verdict, 'PASS');
});

/** The export wrapper's evidence as it writes it on a clean completion; `over` changes outer.json. */
function putExportOuter(root, rec, over = {}) {
  const dir = join(root, AN.EXPORT_OUTER_REL);
  mkdirSync(dir, { recursive: true });
  const outer = { inner_exit: 0, outer_exit: 0, outcome: 'COMPLETED', gate_restored: true, survivors: '', record_by: 'script', term_s: 3590, grace_s: 10, head: rec.head, ...over };
  writeFileSync(join(dir, 'outer.json'), JSON.stringify(outer));
  writeFileSync(join(dir, 'pgid.txt'), `${rec.pid}\n`);
  for (const f of ['idle_check.txt', 'bg_check.txt']) writeFileSync(join(dir, f), 'ok\n');
}
const OUTER_TIMEOUT = { inner_exit: 124, outcome: 'TIMEOUT', outer_exit: 124 };

test('export-record admission: missing, uncommitted, wrong sha, wrong dir, same model id as current are refused', () => {
  const root = mkdtempSync(join(tmpdir(), 'op-exp-'));
  const dir = join(root, 'pair');
  mkdirSync(dir, { recursive: true });
  mkdirSync(join(root, 'docs/superpowers/evidence/htb-onpolicy-export'), { recursive: true });
  writeFileSync(join(dir, 'model.onnx'), 'graph');
  writeFileSync(join(dir, 'model.onnx.data'), 'data');
  const c = { model_id: '0123456789abcdef', dir, graph_sha256: sha('graph'), external_data_sha256: sha('data'), source_sha256: 's'.repeat(64) };
  const man = { model_id: c.model_id, graph: { sha256: c.graph_sha256 }, external_data: { sha256: c.external_data_sha256 } };
  writeFileSync(join(dir, 'manifest.json'), JSON.stringify(man));
  const rec = { kind: 'htb-onpolicy-export-record/1', status: 'COMPLETED', verdict: 'PARITY_PASS', challenger: c, head: 'f'.repeat(40), pid: 4242, kill_s: 3600 };
  const okGit = () => ({ status: 0, stdout: '' });
  const put = (r) => writeFileSync(join(root, D.EXPORT_REL), JSON.stringify(r));
  putExportOuter(root, rec);
  const probs = (over = {}) => D.exportRecordProblems(over.dir ?? dir, { root, g: over.g ?? okGit });
  assert.ok(D.exportRecordProblems(dir, { root, g: okGit }).some((p) => /missing/.test(p)));
  put(rec);
  assert.deepEqual(probs(), []);
  assert.ok(probs({ g: (...a) => ({ status: a[0] === 'ls-files' ? 1 : 0, stdout: '' }) }).some((p) => /not committed/.test(p)));
  assert.ok(probs({ g: (...a) => ({ status: a[0] === 'diff' ? 1 : 0, stdout: '' }) }).some((p) => /differs from HEAD/.test(p)));
  assert.ok(probs({ dir: join(root, 'elsewhere') }).some((p) => /is not the export record's/.test(p)));
  put({ ...rec, verdict: 'PARITY_FAIL' });
  assert.ok(probs().some((p) => /verdict/.test(p)));
  put({ ...rec, challenger: { ...c, graph_sha256: 'f'.repeat(64) } });
  assert.ok(probs().some((p) => /model.onnx sha256/.test(p)));
  put({ ...rec, challenger: { ...c, model_id: '78c16a31c3dba37f' } });
  assert.ok(probs().some((p) => /is the current hard/.test(p)));
  put(rec);
  writeFileSync(join(dir, 'manifest.json'), JSON.stringify({ ...man, model_id: 'ffffffffffffffff' }));
  assert.ok(probs().some((p) => /manifest's model_id/.test(p)));
});

test('committed smoke evidence is checked before the match, through git', () => {
  const ok = (cmd) => (cmd[0] === 'ls-files' && cmd[1] === '--') ? { status: 0, stdout: 'a\nb\nc\nd\n' } : { status: 0, stdout: '' };
  const fake = (over) => (...args) => over(args) ?? ok(args);
  assert.deepEqual(D.smokeEvidenceProblems(fake(() => null)), []);
  assert.ok(D.smokeEvidenceProblems(fake((a) => (a[0] === 'ls-files' && a[1] === '--error-unmatch' && a[3].endsWith('timing.json') ? { status: 1 } : null))).some((p) => /timing.json is not committed/.test(p)));
  assert.ok(D.smokeEvidenceProblems(fake((a) => (a[0] === 'diff' ? { status: 1 } : null))).some((p) => /differs from HEAD/.test(p)));
  assert.ok(D.smokeEvidenceProblems(fake((a) => (a[0] === 'ls-files' && a[1] === '--' ? { status: 0, stdout: 'a\n' } : null))).some((p) => /1 committed smoke games/.test(p)));
});

test('run refuses while the gate is closed, and creates nothing', () => {
  // Since the smoke (2026-11-17, 5fb4a2d5) its evidence directory exists; "nothing" now means nothing new.
  for (const stage of ['smoke', 'match']) {
    const dir = join(D.REPO_ROOT, D.RUN_REL[stage]);
    const before = existsSync(dir) ? readdirSync(dir).sort() : null;
    const r = spawnSync(process.execPath, [join(D.REPO_ROOT, 'tests/htb_integration/onpolicy_match.mjs'), 'run', stage, '--challenger-dir', '/nonexistent', '--kill-s', String(D.KILL_S[stage])], { encoding: 'utf8' });
    assert.equal(r.status, 4);
    assert.match(r.stderr, /AUTHORIZED is closed/);
    assert.deepEqual(existsSync(dir) ? readdirSync(dir).sort() : null, before);
  }
  const u = spawnSync(process.execPath, [join(D.REPO_ROOT, 'tests/htb_integration/onpolicy_match.mjs'), 'run', 'other'], { encoding: 'utf8' });
  assert.equal(u.status, 2);
});

test('exit codes are set through process.exitCode, never process.exit()', () => {
  for (const f of ['onpolicy_match.mjs', 'onpolicy_match_analyse.mjs']) {
    const src = readFileSync(join(D.REPO_ROOT, 'tests/htb_integration', f), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
    assert.ok(!/process\.exit\(/.test(src), f);
    assert.ok(/process\.exitCode = /.test(src), f);
  }
});

test('the wrappers: ceilings, gate names, driver and the pinned analyser sha', () => {
  const analyserSha = sha(readFileSync(join(D.REPO_ROOT, 'tests/htb_integration/onpolicy_match_analyse.mjs')));
  for (const [stage, term, kill, gate] of [['smoke', 1790, 1800, 'ONPOLICY_MATCH_SMOKE_AUTHORIZED'], ['match', 64790, 64800, 'ONPOLICY_MATCH_AUTHORIZED']]) {
    const w = readFileSync(join(D.REPO_ROOT, D.OUTER_REL[stage], '01_wrapper.sh'), 'utf8');
    assert.match(w, new RegExp(`^TERM_S=${term}$`, 'm'));
    assert.match(w, new RegExp(`^KILL_S=${kill}$`, 'm'));
    assert.match(w, new RegExp(`^GATE=${gate}$`, 'm'));
    assert.match(w, new RegExp(`^STAGE=${stage}$`, 'm'));
    assert.match(w, /^DRIVER=tests\/htb_integration\/onpolicy_match\.mjs$/m);
    assert.match(w, new RegExp(`^ANALYSER_SHA256=${analyserSha}$`, 'm'));
    assert.match(w, /test_ort_shutdown\.js/);
    assert.match(w, /idle_gate wait/);
    assert.match(w, /background_compute check/);
    const syn = spawnSync('bash', ['-n', join(D.REPO_ROOT, D.OUTER_REL[stage], '01_wrapper.sh')], { encoding: 'utf8' });
    assert.equal(syn.status, 0, syn.stderr);
  }
  const m = readFileSync(join(D.REPO_ROOT, D.OUTER_REL.match, '01_wrapper.sh'), 'utf8');
  assert.match(m, /"\$NODE" "\$ANALYSER" timing/);
  assert.match(m, /t\.verdict === "LAUNCH_OK" && t\.match_cap_s === 64800/);
});


test('REGRESSION (handoff repair): an export that completed inside but timed out outside is refused before any model loads', () => {
  const root = mkdtempSync(join(tmpdir(), 'op-exo-'));
  const dir = join(root, 'pair');
  mkdirSync(dir, { recursive: true });
  mkdirSync(join(root, 'docs/superpowers/evidence/htb-onpolicy-export'), { recursive: true });
  writeFileSync(join(dir, 'model.onnx'), 'graph');
  writeFileSync(join(dir, 'model.onnx.data'), 'data');
  const c = { model_id: '0123456789abcdef', dir, graph_sha256: sha('graph'), external_data_sha256: sha('data'), source_sha256: 's'.repeat(64) };
  writeFileSync(join(dir, 'manifest.json'), JSON.stringify({ model_id: c.model_id, graph: { sha256: c.graph_sha256 }, external_data: { sha256: c.external_data_sha256 } }));
  const rec = { kind: 'htb-onpolicy-export-record/1', status: 'COMPLETED', verdict: 'PARITY_PASS', challenger: c, head: 'f'.repeat(40), pid: 4242, kill_s: 3600 };
  writeFileSync(join(root, D.EXPORT_REL), JSON.stringify(rec));
  const okGit = () => ({ status: 0, stdout: '' });
  const probs = (g = okGit) => D.exportRecordProblems(dir, { root, g });
  putExportOuter(root, rec);
  assert.deepEqual(probs(), []);
  putExportOuter(root, rec, OUTER_TIMEOUT);                    // the record says COMPLETED, the wrapper timed out
  assert.ok(probs().some((p) => /export outer outcome "TIMEOUT"/.test(p)));
  assert.ok(probs().some((p) => /export outer inner_exit 124/.test(p)));
  // the analyser refuses the same evidence at analysis time
  const outer = JSON.parse(readFileSync(join(root, AN.EXPORT_OUTER_REL, 'outer.json'), 'utf8'));
  assert.ok(AN.exportWrapperFailures({ outer, pgid: '4242\n', exportRecord: rec }).some((p) => /TIMEOUT/.test(p)));
  // launch commit, pid, cleanup, ceiling, and every wrapper file committed and clean
  const cases = [
    [{ term_s: 1790 }, /export outer term_s/], [{ grace_s: 5 }, /export outer grace_s/], [{ head: 'e'.repeat(40) }, /launch commit/],
    [{ survivors: 'unconfirmed cleanup: process group 4242 4243' }, /export outer survivors/], [{ gate_restored: false }, /gate_restored/],
    [{ record_by: 'wrapper' }, /record_by/],
  ];
  for (const [over, re] of cases) {
    putExportOuter(root, rec, over);
    assert.ok(probs().some((p) => re.test(p)), JSON.stringify(over));
  }
  putExportOuter(root, rec);
  writeFileSync(join(root, AN.EXPORT_OUTER_REL, 'pgid.txt'), '1\n');
  assert.ok(probs().some((p) => /pgid.txt is not the export record's pid/.test(p)));
  putExportOuter(root, { ...rec });
  writeFileSync(join(root, D.EXPORT_REL), JSON.stringify({ ...rec, kill_s: 1800 }));
  assert.ok(probs().some((p) => /export ceiling/.test(p)));
  writeFileSync(join(root, D.EXPORT_REL), JSON.stringify(rec));
  assert.ok(probs((...a) => ({ status: a[0] === 'ls-files' && a[1] === '--error-unmatch' && String(a[3]).endsWith('bg_check.txt') ? 1 : 0, stdout: '' })).some((p) => /bg_check.txt is not committed/.test(p)));
  assert.ok(probs((...a) => ({ status: 0, stdout: a[1] === '--others' ? 'stray.json\n' : '' })).some((p) => /untracked export evidence/.test(p)));
  assert.ok(probs((...a) => ({ status: a[0] === 'diff' ? 1 : 0, stdout: '' })).some((p) => /differs from HEAD/.test(p)));
});

test('both stages check the export wrapper before any model loads (source order in run)', () => {
  const src = readFileSync(join(D.REPO_ROOT, 'tests/htb_integration/onpolicy_match.mjs'), 'utf8');
  const body = src.slice(src.indexOf('async function run(stage, opts)'));
  const check = body.indexOf('exportRecordProblems(');
  assert.ok(check > 0);
  for (const load of ["import('../../server/model_manifest.js')", "import('../../server/inference.js')", 'base.load()']) {
    assert.ok(body.indexOf(load) > check, `${load} comes after the export check`);
  }
  assert.ok(check < body.indexOf("if (stage === 'match')"), 'the check applies to the smoke and the match alike');
  const ana = readFileSync(join(D.REPO_ROOT, 'tests/htb_integration/onpolicy_match_analyse.mjs'), 'utf8');
  assert.ok(ana.slice(ana.indexOf('export function admit(')).includes('exportWrapperFailures('));
});
