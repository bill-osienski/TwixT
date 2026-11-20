/**
 * H-MX match: model-free tests (fake evaluators at the real 200-simulation budget). No ONNX model loads, no gate opens.
 * Card docs/superpowers/2026-11-18-htb-mix-distillation-card.md §9-§11. Mirrors test_onpolicy_match.mjs.
 * Run: node --test tests/htb_integration/test_mix_match.mjs
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { test } from 'node:test';

import { TwixtState } from '../../server/gameLogic.js';
import { MirrorAveragingInference } from '../../server/mirror_inference.js';
import { protocolMove } from '../../server/protocol_move.js';
import { resolvePolicy } from '../../server/readout_policy.js';
import { fakeInference } from '../fixtures/htb_integration/fake_evaluator.mjs';
import * as AN from './mix_match_analyse.mjs';
import * as GATE from './mix_authorization.mjs';
import * as D from './mix_match.mjs';
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
      for (const [, v] of out.priors) s += v * v;
      for (const [k, v] of out.priors) priors.set(k, (v * v) / s);
      return { priors, value: -out.value / 2 };
    },
  };
};
const players = (policy = REAL) => ({
  challenger: D.makePlayer('challenger', new MirrorAveragingInference(fakeInference()), policy),
  current: D.makePlayer('current', new MirrorAveragingInference(otherFake()), policy),
});

/** A FAST stub with a search player's evidence shape, for the admission tests only (as in H-OP's tests). */
const stubPlayer = (salt) => ({
  async move(state, { stage, pair, game }) {
    const legal = [...state.legalMoves()].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const h = createHash('sha256').update(`${salt}/${pair}/${game}/${state.ply}`).digest();
    const mv = legal[h.readUInt32BE(0) % legal.length];
    return { moveKey: `${mv[0]},${mv[1]}`, extra: { seeds: D.moveSeeds(stage, pair, game, state.ply), nSimulations: 200, visits: 200 } };
  },
});
const stubs = () => ({ challenger: stubPlayer('c'), current: stubPlayer('k') });

test('the gates are closed in the committed state, and the file holds only the two declarations', () => {
  assert.equal(GATE.MIX_MATCH_SMOKE_AUTHORIZED, false);
  assert.equal(GATE.MIX_MATCH_AUTHORIZED, false);
  const src = readFileSync(join(D.REPO_ROOT, 'tests/htb_integration/mix_authorization.mjs'), 'utf8');
  assert.deepEqual(src.split('\n').filter((l) => l.startsWith('export ')), ['export const MIX_MATCH_SMOKE_AUTHORIZED = false;', 'export const MIX_MATCH_AUTHORIZED = false;']);
});

test('openings: the pinned file, its sizes, and the analyser re-derives every listed opening', () => {
  const raw = readFileSync(join(D.REPO_ROOT, D.OPENINGS_REL));
  assert.equal(sha(raw), D.OPENINGS_SHA256);
  assert.equal(OPENINGS.match.length, 400);
  assert.equal(OPENINGS.match_smoke.length, 2);
  assert.equal(OPENINGS.namespace, 'htb-mix/20261118');
  assert.deepEqual(AN.stageOpenings('match'), OPENINGS.match);
  assert.deepEqual(AN.stageOpenings('smoke'), OPENINGS.match_smoke);
  const root = mkdtempSync(join(tmpdir(), 'mx-open-'));
  mkdirSync(join(root, 'docs/superpowers/evidence/htb-mix-openings'), { recursive: true });
  writeFileSync(join(root, D.OPENINGS_REL), raw.toString('utf8').replace('"match"', '"match" '));
  assert.throws(() => D.loadOpenings(root), D.MatchRefusal);
  assert.throws(() => AN.stageOpenings('match', root), AN.Invalid);
});

test('openings: Node re-derives all 402 Python-built openings, excluding the benchmark, G3 and every H-OP opening', () => {
  const seq = (m) => JSON.stringify(m);
  const refl = (m) => JSON.stringify(m.map(([r, c]) => [r, 23 - c]));
  const excl = new Set();
  for (const [kind, n] of [['opening', 200], ['smoke opening', 1]]) for (const o of TB.openingList(kind, n)) excl.add(seq(o.moves));
  for (const d of ['docs/superpowers/evidence/htb-g3-match/games', 'docs/superpowers/evidence/htb-g3-smoke/games']) {
    for (const f of readdirSync(join(D.REPO_ROOT, d)).filter((x) => x.endsWith('.json'))) {
      excl.add(seq(JSON.parse(readFileSync(join(D.REPO_ROOT, d, f), 'utf8')).moves.slice(0, 4)));
    }
  }
  const hop = [['docs/superpowers/evidence/htb-onpolicy-openings/openings.json', 'ca4e3bba05bfc6abddf782b1a1f68fc474bc3600410c7cbd4308073d306d3915'],
    ['docs/superpowers/evidence/htb-onpolicy-openings/gen-extension.json', '578d34f371b238a9919a6eeb9f71c3b30eb286b7b08e95d1722e1fc17f270664']];
  let nHop = 0;
  for (const [rel, want] of hop) {
    const raw = readFileSync(join(D.REPO_ROOT, rel));
    assert.equal(sha(raw), want, rel);
    const o = JSON.parse(raw.toString('utf8'));
    for (const k of ['gen', 'gen_smoke', 'match', 'match_smoke', 'gen_extension']) for (const e of o[k] ?? []) { excl.add(seq(e.moves)); nHop += 1; }
  }
  assert.equal(nHop, 6814);
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
  assert.deepEqual(list('match opening', 400), OPENINGS.match);
  assert.deepEqual(list('match smoke opening', 2), OPENINGS.match_smoke);
});

test('schedule: 800 match games and 4 smoke games; pair p plays opening p, challenger Red in game 0', () => {
  const m = D.schedule('match', OPENINGS);
  assert.equal(m.length, 800);
  for (const t of m) {
    assert.equal(t.challengerColour, t.game === 0 ? 'red' : 'black');
    assert.deepEqual(t.opening, OPENINGS.match[t.pair].moves);
    assert.equal(t.openingIndex, OPENINGS.match[t.pair].index);
  }
  assert.equal(D.schedule('smoke', OPENINGS).length, 4);
  assert.deepEqual(D.schedule('smoke', OPENINGS)[2].opening, OPENINGS.match_smoke[1].moves);
});

test('seeds: the formula, the analyser re-derives them, all distinct and below 2^48', () => {
  const h = sha('htb-mix/20261118/match/pair 3/game 1/ply 17');
  assert.deepEqual(D.moveSeeds('match', 3, 1, 17), { search: parseInt(h.slice(0, 12), 16), readout: parseInt(h.slice(12, 24), 16) });
  const hs = sha('htb-mix/20261118/match smoke/pair 0/game 0/ply 4');
  assert.equal(D.moveSeeds('smoke', 0, 0, 4).search, parseInt(hs.slice(0, 12), 16));
  const seen = new Set();
  for (const [stage, pairs] of [['smoke', 2], ['match', 400]]) {
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

test('the non-finite guard (H-OP\'s, imported) stops either side before a move is returned', async () => {
  const nan = { async evaluate(_h, moves) { return { priors: new Map(moves.map((m) => [`${m[0]},${m[1]}`, 1 / moves.length])), value: Number.NaN }; } };
  let s = new TwixtState();
  for (const m of OPENINGS.match[0].moves) s = s.applyMove(m);
  for (const who of ['challenger', 'current']) {
    await assert.rejects(D.makePlayer(who, nan, REAL).move(s, { stage: 'match', pair: 0, game: 0 }), (e) => e instanceof D.MatchStop && e.step === who);
  }
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
/** What export attempt 2's record carries (export amendment 1, §2, §4). */
const ATTEMPT_2 = { attempt: 2, export_amendment_sha256: AN.EXPORT_AMENDMENT_SHA256 };
/** Export attempt 1's committed evidence (ledger entry 75, VOID), never read by the match stages. */
const A1_REC = 'docs/superpowers/evidence/htb-mix-export/record.json';
const A1_OUTER = 'docs/superpowers/evidence/htb-mix-export-run';
const copyA1 = (root, recRel, outerRel) => {
  mkdirSync(join(root, dirname(recRel)), { recursive: true });
  mkdirSync(join(root, outerRel), { recursive: true });
  writeFileSync(join(root, recRel), readFileSync(join(D.REPO_ROOT, A1_REC)));
  for (const f of readdirSync(join(D.REPO_ROOT, A1_OUTER))) writeFileSync(join(root, outerRel, f), readFileSync(join(D.REPO_ROOT, A1_OUTER, f)));
};

test('export attempt 1 (the committed VOID record, entry 75) can never admit a match: only attempt 2 is read', async () => {
  // the paths: the driver, the analyser and both wrappers read attempt 2's evidence only
  assert.equal(D.EXPORT_REL, 'docs/superpowers/evidence/htb-mix-export-2/record.json');
  assert.equal(AN.EXPORT_REL, D.EXPORT_REL);
  assert.equal(AN.EXPORT_OUTER_REL, 'docs/superpowers/evidence/htb-mix-export-2-run');
  assert.equal(AN.EXPORT_ATTEMPT, 2);
  assert.equal(AN.EXPORT_AMENDMENT_SHA256, sha(readFileSync(join(D.REPO_ROOT, 'docs/superpowers/2026-11-20-htb-mix-export-attempt-2-amendment.md'))));
  for (const f of ['mix_match.mjs', 'mix_match_analyse.mjs']) {
    const src = readFileSync(join(D.REPO_ROOT, 'tests/htb_integration', f), 'utf8');
    assert.ok(!src.includes('evidence/htb-mix-export/') && !src.includes("evidence/htb-mix-export-run'"), f);
  }
  for (const stage of ['smoke', 'match']) {
    const w = readFileSync(join(D.REPO_ROOT, D.OUTER_REL[stage], '01_wrapper.sh'), 'utf8');
    assert.ok(!w.includes('evidence/htb-mix-export/') && !w.includes('evidence/htb-mix-export-run'), stage);
  }
  const voidRec = JSON.parse(readFileSync(join(D.REPO_ROOT, A1_REC), 'utf8'));
  assert.equal(voidRec.status, 'VOID');
  const okGit = () => ({ status: 0, stdout: '' });
  // 1. attempt 1's real committed evidence at its own paths, nothing for attempt 2: the driver finds no export record
  const root = mkdtempSync(join(tmpdir(), 'mx-a1-'));
  copyA1(root, A1_REC, A1_OUTER);
  assert.deepEqual(D.exportRecordProblems('/nonexistent', { root, g: okGit }), [`${D.EXPORT_REL} is missing`]);
  // 2. the same VOID record and wrapper files copied into attempt 2's paths: refused on every count
  copyA1(root, D.EXPORT_REL, AN.EXPORT_OUTER_REL);
  const probs = D.exportRecordProblems('/nonexistent', { root, g: okGit });
  for (const re of [/status "VOID"/, /verdict undefined/, /attempt undefined/, /export_amendment_sha256 undefined/, /challenger.model_id missing/]) {
    assert.ok(probs.some((p) => re.test(p)), `${re} in ${probs.join('; ')}`);
  }
  // 3. the analyser's binding check, given the VOID record as the export record
  const bf = AN.bindingFailures({ outer: {}, record: {}, pgid: null, stage: 'smoke', exportRecord: voidRec });
  assert.ok(bf.includes('the export record is not a COMPLETED PARITY_PASS H-MX export record'));
  assert.ok(bf.includes('the export record is not export attempt 2 under its amendment'));
  // 4. an otherwise admissible smoke whose only export evidence is attempt 1's: admit() refuses
  const t = await evidenceTree('smoke', 2);
  t.write();
  AN.admit('smoke', t.root);
  rmSync(join(t.root, dirname(D.EXPORT_REL)), { recursive: true });
  rmSync(join(t.root, AN.EXPORT_OUTER_REL), { recursive: true });
  copyA1(t.root, A1_REC, A1_OUTER);
  assert.throws(() => AN.admit('smoke', t.root), /export record missing/);
  copyA1(t.root, D.EXPORT_REL, AN.EXPORT_OUTER_REL);
  assert.throws(() => AN.admit('smoke', t.root), /not export attempt 2 under its amendment/);
});

/** A smoke or match evidence tree as the wrapper and driver would leave it; stub players. */
async function evidenceTree(stage, nPairs) {
  const root = mkdtempSync(join(tmpdir(), 'mx-adm-'));
  const run = join(root, AN.DIRS[stage].run);
  const outer = join(root, AN.DIRS[stage].outer);
  mkdirSync(join(run, 'games'), { recursive: true });
  mkdirSync(outer, { recursive: true });
  mkdirSync(join(root, 'docs/superpowers/evidence/htb-mix-openings'), { recursive: true });
  writeFileSync(join(root, D.OPENINGS_REL), readFileSync(join(D.REPO_ROOT, D.OPENINGS_REL)));
  mkdirSync(join(root, dirname(D.EXPORT_REL)), { recursive: true });
  const challenger = { model_id: '0123456789abcdef', dir: '/tmp/x/0123456789abcdef', graph_sha256: 'g'.repeat(64), external_data_sha256: 'd'.repeat(64), source_sha256: 's'.repeat(64) };
  const exportRecord = { kind: D.EXPORT_KIND, status: 'COMPLETED', verdict: 'PARITY_PASS', challenger, head: 'c'.repeat(40), pid: 5151, kill_s: 3600, ...ATTEMPT_2 };
  writeFileSync(join(root, D.EXPORT_REL), JSON.stringify(exportRecord));
  putExportOuter(root, exportRecord);
  const games = {};
  const pl = stubs();
  for (const task of D.schedule(stage, OPENINGS).filter((t) => t.pair < nPairs)) {
    games[`pair_${String(task.pair).padStart(4, '0')}_game_${task.game}.json`] = await D.playGame({ stage, task, players: pl });
  }
  const record = { schema: D.RECORD_SCHEMA, written_by: 'script', status: 'COMPLETED', stage, games: Object.keys(games).length, startup_seconds: 20.5,
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
  assert.equal(g.schema, 'htb-mix-match-game/1');
  assert.ok(g.per_move.every((m) => m.nSimulations === 200 && m.visits === 200));
  assert.ok(g.per_move.some((m) => m.mover === 'challenger') && g.per_move.some((m) => m.mover === 'current'));
  assert.equal(AN.validateGame(g, { stage: 'smoke', pair: 0, game: 1, opening: OPENINGS.match_smoke[0] }), g);
  assert.throws(() => AN.validateGame({ ...g, game: 0 }, { stage: 'smoke', pair: 0, game: 0, opening: OPENINGS.match_smoke[0] }), AN.Invalid);
});

test('smoke admission binds the evidence to its run; timing reads wall times only over 800 games; each binding failure refuses', async () => {
  const t = await evidenceTree('smoke', 2);
  t.write();
  const adm = AN.admit('smoke', t.root);
  assert.equal(adm.games.length, 4);
  const est = AN.timing(adm);
  assert.equal(est.schema, 'htb-mix-match-timing/1');
  assert.equal(est.match_cap_s, 64800);
  assert.equal(est.match_games, 800);
  const perGame = adm.games.reduce((a, g) => a + g.wall_ms / 1000, 0) / 4;
  assert.ok(Math.abs(est.projection_s - (20.5 + perGame * 800)) < 0.002);
  assert.ok(Math.abs(est.launch_estimate_s - 1.5 * est.projection_s) < 0.002);
  assert.equal(est.verdict, 'LAUNCH_OK');
  assert.ok(!/score|result|"W"|"L"/.test(JSON.stringify(est)));
  // the boundary: 1.5 x (startup + mean game x 800) <= 64,800 s
  const at = (gameS) => AN.timing({ record: { startup_seconds: 1 }, games: [{ wall_ms: gameS * 1000, plies: 30 }] }).verdict;
  assert.equal(at(53.99), 'LAUNCH_OK');   // 1.5 x (1 + 53.99 x 800) = 64,789.5 <= 64,800
  assert.equal(at(54.01), 'NO_LAUNCH');   // 1.5 x (1 + 54.01 x 800) = 64,813.5
  assert.equal(AN.timing({ ...adm, record: { ...adm.record, startup_seconds: 50000 } }).verdict, 'NO_LAUNCH');
  const bad = [
    ['missing outer', () => t.write({ o: null })],
    ['crashed', () => t.write({ o: { ...t.outerRec, outcome: 'CRASHED', inner_exit: 134, outer_exit: 134 } })],
    ['another launch commit', () => t.write({ o: { ...t.outerRec, head: 'b'.repeat(40) } })],
    ['pgid is not the driver', () => t.write({ pgid: '4243\n' })],
    ['the match ceiling on the smoke', () => t.write({ o: { ...t.outerRec, term_s: 64790 } })],
    ['record ceiling', () => t.write({ r: { ...t.record, kill_s: 64800 } })],
    ['the H-OP record schema', () => t.write({ r: { ...t.record, schema: 'htb-onpolicy-match-record/1' } })],
    ['onnxruntime-node', () => t.write({ r: { ...t.record, runtime: { ...t.record.runtime, onnxruntime_node: '1.24.4' } } })],
    ['budget', () => t.write({ r: { ...t.record, policy: { ...REAL, nSims: 64 } } })],
    ['another challenger', () => t.write({ r: { ...t.record, challenger: { ...t.record.challenger, model_id: 'ffffffffffffffff' } } })],
    ['current as challenger', () => t.write({ r: { ...t.record, current: { model_id: 'ffffffffffffffff' } } })],
    ['H-OP\'s openings', () => t.write({ r: { ...t.record, openings_sha256: 'ca4e3bba05bfc6abddf782b1a1f68fc474bc3600410c7cbd4308073d306d3915' } })],
  ];
  for (const [what, corrupt] of bad) {
    corrupt();
    assert.throws(() => AN.admit('smoke', t.root), AN.Invalid, what);
    t.write();
    AN.admit('smoke', t.root);
  }
  // an H-OP export record in the H-MX slot is refused
  writeFileSync(join(t.root, D.EXPORT_REL), JSON.stringify({ ...t.exportRecord, kind: 'htb-onpolicy-export-record/1' }));
  assert.throws(() => AN.admit('smoke', t.root), /H-MX export record/);
  writeFileSync(join(t.root, D.EXPORT_REL), JSON.stringify(t.exportRecord));
  // export attempt 2 only (export amendment 1 §4): another attempt number or no amendment pin is refused
  for (const over of [{ attempt: 1 }, { attempt: undefined }, { export_amendment_sha256: 'f'.repeat(64) }]) {
    writeFileSync(join(t.root, D.EXPORT_REL), JSON.stringify({ ...t.exportRecord, ...over }));
    assert.throws(() => AN.admit('smoke', t.root), /not export attempt 2 under its amendment/, JSON.stringify(over));
  }
  writeFileSync(join(t.root, D.EXPORT_REL), JSON.stringify(t.exportRecord));
  // the handoff repair: an export that completed inside but timed out outside fails admission at analysis too
  putExportOuter(t.root, t.exportRecord, OUTER_TIMEOUT);
  assert.throws(() => AN.admit('smoke', t.root), /export outer outcome "TIMEOUT"/);
  putExportOuter(t.root, t.exportRecord);
  AN.admit('smoke', t.root);
  // a missing or an extra game file
  const gdir = join(t.root, AN.DIRS.smoke.run, 'games');
  writeFileSync(join(gdir, 'pair_0002_game_0.json'), '{}');
  assert.throws(() => AN.admit('smoke', t.root), /exactly the 4 scheduled games/);
});

test('per-move evidence: a 64-sim or 199-visit row, a missing, duplicate or extra row, or wrong seeds are INVALID', async () => {
  const t = await evidenceTree('smoke', 1);
  const g = t.games['pair_0000_game_0.json'];
  const iC = g.per_move.findIndex((m) => m.mover === 'current');
  const variants = {
    'a 64-simulation budget': { ...g, per_move: g.per_move.map((m, i) => (i === 0 ? { ...m, nSimulations: 64 } : m)) },
    '199 visits spent': { ...g, per_move: g.per_move.map((m, i) => (i === iC ? { ...m, visits: 199 } : m)) },
    'a duplicate row': { ...g, per_move: [...g.per_move.slice(0, 2), g.per_move[1], ...g.per_move.slice(3)] },
    'a missing row': { ...g, per_move: g.per_move.slice(1) },
    'an extra row': { ...g, per_move: [...g.per_move, g.per_move.at(-1)] },
    'H-OP seeds': { ...g, per_move: g.per_move.map((m, i) => (i === 1 ? { ...m, seeds: { search: 1, readout: 2 } } : m)) },
    'wrong seeds': { ...g, per_move: g.per_move.map((m, i) => (i === 1 ? { ...m, seeds: D.moveSeeds('match', 0, 0, m.ply) } : m)) },
    'a wrong score': { ...g, challenger_score: g.challenger_score === 1 ? 0 : 1 },
    'the H-OP game schema': { ...g, schema: 'htb-onpolicy-match-game/1' },
  };
  const ctx = { stage: 'smoke', pair: 0, game: 0, opening: OPENINGS.match_smoke[0] };
  assert.equal(AN.validateGame(g, ctx), g);
  for (const [what, v] of Object.entries(variants)) assert.throws(() => AN.validateGame(v, ctx), AN.Invalid, what);
  let s = new TwixtState();
  for (const m of OPENINGS.match_smoke[0].moves) s = s.applyMove(m);
  const p64 = await players(SMALL).challenger.move(s, { stage: 'smoke', pair: 0, game: 0 });
  assert.equal(p64.extra.nSimulations, 64);
  assert.equal(p64.extra.visits, 64);
  const row64 = { ...g, per_move: g.per_move.map((m, i) => (i === 0 ? { ...m, ...p64.extra } : m)) };
  assert.throws(() => AN.validateGame(row64, ctx), AN.Invalid);
});

test('the exact sign test: critical values, exact tails, and the 5% boundary decided in integers', () => {
  for (const [m, w] of [[30, 20], [40, 26], [46, 30], [50, 32], [60, 37], [10, 9], [5, 5]]) assert.equal(AN.criticalW(m), w, `m=${m}`);
  for (const m of [0, 1, 2, 3, 4]) assert.equal(AN.criticalW(m), null);
  const t = AN.binomialTail(10, 9);
  assert.equal(t.num, 11n);
  assert.equal(t.den, 1024n);
  assert.equal(t.p, 11 / 1024);
  assert.equal(AN.binomialTail(40, 26).p <= 0.05, true);
  assert.equal(AN.binomialTail(40, 25).p > 0.05, true);
  // exactness: the critical W is the boundary in exact integer arithmetic (every m to 100, then spot values to 400)
  for (const m of [...Array.from({ length: 96 }, (_, i) => i + 5), 150, 200, 250, 300, 350, 399, 400]) {
    const c = AN.criticalW(m);
    assert.ok(AN.rejects(m, c) && !AN.rejects(m, c - 1), `m=${m}`);
  }
  assert.equal(AN.rejects(0, 0), false);
  const st = AN.signTest(26, 14);
  assert.equal(st.m, 40);
  assert.equal(st.critical_W, 26);
  assert.equal(st.rejects, true);
  assert.equal(AN.signTest(25, 15).rejects, false);
  // Clopper-Pearson (descriptive): reference values for 26/40, 0/10 and 10/10
  const [lo, hi] = AN.clopperPearson(26, 40);
  assert.ok(Math.abs(lo - 0.48316) < 1e-4 && Math.abs(hi - 0.79372) < 1e-4);
  assert.ok(Math.abs(AN.clopperPearson(0, 10)[1] - (1 - 0.025 ** (1 / 10))) < 1e-9);
  assert.ok(Math.abs(AN.clopperPearson(10, 10)[0] - 0.025 ** (1 / 10)) < 1e-9);
  assert.equal(AN.twoSidedHalfWidth(400).toFixed(4), '0.0679');
});

/** 400 pairs with given counts of pair scores: { 1: n1, 0.75: n75, 0.5: n5, 0.25: n25, 0: n0 }. */
function mkPairs(counts, ms = [100, 100]) {
  const games = [];
  let p = 0;
  const split = { 1: [1, 1], 0.75: [1, 0.5], 0.5: [1, 0], 0.25: [0.5, 0], 0: [0, 0] };
  for (const [score, n] of Object.entries(counts)) {
    for (let i = 0; i < n; i++, p++) {
      for (const [game, sc] of [[0, split[score][0]], [1, split[score][1]]]) {
        games.push({ pair: p, game, challenger_colour: game ? 'black' : 'red', challenger_score: sc, result: sc === 0.5 ? 'draw' : 'x', plies: 40, moves: [[p, game]], per_move: [{ mover: 'challenger', ms: ms[0] }, { mover: 'current', ms: ms[1] }] });
      }
    }
  }
  assert.equal(p, 400);
  return games;
}

test('the verdict: PASS at the critical W, NON_PASS one below, and the reading says what each establishes', () => {
  // 46 decisive pairs: 30 wins / 16 losses rejects; 29 / 17 does not
  const pass = AN.analyse({ games: mkPairs({ 1: 30, 0: 16, 0.5: 354 }) });
  assert.equal(pass.verdict, 'PASS');
  assert.equal(pass.schema, 'htb-mix-match-analysis/1');
  assert.deepEqual([pass.sign_test.W, pass.sign_test.L, pass.sign_test.m, pass.sign_test.critical_W], [30, 16, 46, 30]);
  assert.ok(pass.sign_test.p_one_sided <= 0.05);
  assert.equal(pass.mean_pair_score, (30 + 0.5 * 354) / 400);
  assert.equal(pass.mean_guard.above_half, true);
  assert.equal(pass.decisive_pairs.split, 354);
  assert.match(pass.reading, /wins more decisive pairs than it loses/);
  assert.match(pass.reading, /planned opening/);
  assert.match(pass.reading, /whole update/);
  assert.match(pass.reading, /\(a\) independent pairs, \(b\) exchangeable, fair win\/loss signs under the null, \(c\) a fixed design/);
  assert.match(pass.reading, /\(d\) its scope is the planned opening distribution/);
  assert.match(pass.reading, /additional guard that can only remove a PASS/);
  assert.equal(pass.descriptive.two_sided_95_hoeffding_half_width.toFixed(4), '0.0679');
  assert.equal(pass.descriptive.distinct_games, 800);
  const non = AN.analyse({ games: mkPairs({ 1: 29, 0: 17, 0.5: 354 }) });
  assert.equal(non.verdict, 'NON_PASS');
  assert.match(non.reading, /not evidence of no gain/);
  assert.match(non.reading, /0\.691/);
  assert.match(non.reading, /no top-up or re-score/);
  // no decisive pair at all: NON_PASS, never PASS
  assert.equal(AN.analyse({ games: mkPairs({ 0.5: 400 }) }).verdict, 'NON_PASS');
});

test('the mean guard turns a sign-test rejection into NON_PASS (0.75 pairs against 0 pairs)', () => {
  // 30 pairs at 0.75 (W) against 16 at 0 (L): the test rejects (m 46, W 30), but the mean is
  // (30 x 0.75 + 354 x 0.5) / 400 = 0.49875 <= 0.5
  const a = AN.analyse({ games: mkPairs({ 0.75: 30, 0: 16, 0.5: 354 }) });
  assert.equal(a.sign_test.rejects, true);
  assert.ok(a.mean_pair_score <= 0.5);
  assert.equal(a.mean_guard.above_half, false);
  assert.equal(a.verdict, 'NON_PASS');
});

test('the search-seconds guard: above 1.10x the current hard is INVALID and nothing is scored', () => {
  const games = mkPairs({ 1: 400 }, [111, 100]);
  const a = AN.analyse({ games });
  assert.equal(a.verdict, 'INVALID');
  assert.ok(!('mean_pair_score' in a) && !('sign_test' in a) && !('challenger' in a));
  assert.ok(a.search_seconds_guard.ratio > 1.1);
  for (const g of games) g.per_move[0].ms = 110;
  assert.equal(AN.analyse({ games }).verdict, 'PASS');
});

test('export-record admission: missing, uncommitted, wrong kind or verdict, wrong sha, wrong dir, current id are refused', () => {
  const root = mkdtempSync(join(tmpdir(), 'mx-exp-'));
  const dir = join(root, 'pair');
  mkdirSync(dir, { recursive: true });
  mkdirSync(join(root, dirname(D.EXPORT_REL)), { recursive: true });
  writeFileSync(join(dir, 'model.onnx'), 'graph');
  writeFileSync(join(dir, 'model.onnx.data'), 'data');
  const c = { model_id: '0123456789abcdef', dir, graph_sha256: sha('graph'), external_data_sha256: sha('data'), source_sha256: 's'.repeat(64) };
  const man = { model_id: c.model_id, graph: { sha256: c.graph_sha256 }, external_data: { sha256: c.external_data_sha256 } };
  writeFileSync(join(dir, 'manifest.json'), JSON.stringify(man));
  const rec = { kind: 'htb-mix-export-record/1', status: 'COMPLETED', verdict: 'PARITY_PASS', challenger: c, head: 'f'.repeat(40), pid: 4242, kill_s: 3600, ...ATTEMPT_2 };
  const okGit = () => ({ status: 0, stdout: '' });
  const put = (r) => writeFileSync(join(root, D.EXPORT_REL), JSON.stringify(r));
  putExportOuter(root, rec);
  const probs = (over = {}) => D.exportRecordProblems(over.dir ?? dir, { root, g: over.g ?? okGit, fileSha: over.fileSha });
  assert.ok(D.exportRecordProblems(dir, { root, g: okGit }).some((p) => /missing/.test(p)));
  put(rec);
  assert.deepEqual(probs(), []);
  assert.ok(probs({ g: (...a) => ({ status: a[0] === 'ls-files' ? 1 : 0, stdout: '' }) }).some((p) => /not committed/.test(p)));
  assert.ok(probs({ g: (...a) => ({ status: a[0] === 'diff' ? 1 : 0, stdout: '' }) }).some((p) => /differs from HEAD/.test(p)));
  assert.ok(probs({ g: (...a) => ({ status: 0, stdout: a[1] === '--others' ? 'stray.json\n' : '' }) }).some((p) => /untracked export evidence/.test(p)));
  assert.ok(probs({ dir: join(root, 'elsewhere') }).some((p) => /is not the export record's/.test(p)));
  assert.ok(probs({ fileSha: () => 'f'.repeat(64) }).some((p) => /model.onnx sha256/.test(p)));
  put({ ...rec, kind: 'htb-onpolicy-export-record/1' });
  assert.ok(probs().some((p) => /export record kind/.test(p)));
  put({ ...rec, verdict: 'PARITY_FAIL' });
  assert.ok(probs().some((p) => /verdict/.test(p)));
  put({ ...rec, attempt: 1 });
  assert.ok(probs().some((p) => /export record attempt 1, expected 2/.test(p)));
  put({ ...rec, export_amendment_sha256: undefined });
  assert.ok(probs().some((p) => /export record export_amendment_sha256 undefined/.test(p)));
  put({ ...rec, challenger: { ...c, graph_sha256: 'f'.repeat(64) } });
  assert.ok(probs().some((p) => /model.onnx sha256/.test(p)));
  put({ ...rec, challenger: { ...c, model_id: '78c16a31c3dba37f' } });
  assert.ok(probs().some((p) => /is the current hard/.test(p)));
  put(rec);
  writeFileSync(join(dir, 'manifest.json'), JSON.stringify({ ...man, model_id: 'ffffffffffffffff' }));
  assert.ok(probs().some((p) => /manifest's model_id/.test(p)));
  writeFileSync(join(dir, 'manifest.json'), JSON.stringify(man));
  // the handoff repair: a completed-inside, timed-out-outside export, and every wrapper fault
  putExportOuter(root, rec, OUTER_TIMEOUT);
  assert.ok(probs().some((p) => /export outer outcome "TIMEOUT"/.test(p)));
  for (const [over, re] of [[{ term_s: 1790 }, /export outer term_s/], [{ head: 'e'.repeat(40) }, /launch commit/], [{ survivors: 'x' }, /survivors/], [{ gate_restored: false }, /gate_restored/]]) {
    putExportOuter(root, rec, over);
    assert.ok(probs().some((p) => re.test(p)), JSON.stringify(over));
  }
  putExportOuter(root, rec);
  writeFileSync(join(root, AN.EXPORT_OUTER_REL, 'pgid.txt'), '1\n');
  assert.ok(probs().some((p) => /pgid.txt is not the export record's pid/.test(p)));
});

test('committed smoke evidence is checked before the match, through git', () => {
  const ok = (cmd) => (cmd[0] === 'ls-files' && cmd[1] === '--') ? { status: 0, stdout: 'a\nb\nc\nd\n' } : { status: 0, stdout: '' };
  const fake = (over) => (...args) => over(args) ?? ok(args);
  assert.deepEqual(D.smokeEvidenceProblems(fake(() => null)), []);
  assert.ok(D.smokeEvidenceProblems(fake((a) => (a[0] === 'ls-files' && a[1] === '--error-unmatch' && a[3].endsWith('timing.json') ? { status: 1 } : null))).some((p) => /timing.json is not committed/.test(p)));
  assert.ok(D.smokeEvidenceProblems(fake((a) => (a[0] === 'diff' ? { status: 1 } : null))).some((p) => /differs from HEAD/.test(p)));
  assert.ok(D.smokeEvidenceProblems(fake((a) => (a[0] === 'ls-files' && a[1] === '--' ? { status: 0, stdout: 'a\n' } : null))).some((p) => /1 committed smoke games/.test(p)));
});

test('run refuses while each gate is closed, and creates nothing new', () => {
  for (const stage of ['smoke', 'match']) {
    const dir = join(D.REPO_ROOT, D.RUN_REL[stage]);
    const before = existsSync(dir) ? readdirSync(dir).sort() : null;
    const r = spawnSync(process.execPath, [join(D.REPO_ROOT, 'tests/htb_integration/mix_match.mjs'), 'run', stage, '--challenger-dir', '/nonexistent', '--kill-s', String(D.KILL_S[stage])], { encoding: 'utf8' });
    assert.equal(r.status, 4);
    assert.match(r.stderr, new RegExp(`${D.GATES[stage]} is closed`));
    assert.deepEqual(existsSync(dir) ? readdirSync(dir).sort() : null, before);
  }
  const u = spawnSync(process.execPath, [join(D.REPO_ROOT, 'tests/htb_integration/mix_match.mjs'), 'run', 'other'], { encoding: 'utf8' });
  assert.equal(u.status, 2);
});

test('exit codes are set through process.exitCode, never process.exit()', () => {
  for (const f of ['mix_match.mjs', 'mix_match_analyse.mjs']) {
    const src = readFileSync(join(D.REPO_ROOT, 'tests/htb_integration', f), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
    assert.ok(!/process\.exit\(/.test(src), f);
    assert.ok(/process\.exitCode = /.test(src), f);
  }
});

test('both stages check the export wrapper before any model loads (source order in run)', () => {
  const src = readFileSync(join(D.REPO_ROOT, 'tests/htb_integration/mix_match.mjs'), 'utf8');
  const body = src.slice(src.indexOf('async function run(stage, opts)'));
  const check = body.indexOf('exportRecordProblems(');
  assert.ok(check > 0);
  for (const load of ["import('../../server/model_manifest.js')", "import('../../server/inference.js')", 'base.load()']) {
    assert.ok(body.indexOf(load) > check, `${load} comes after the export check`);
  }
  assert.ok(check < body.indexOf("if (stage === 'match')"));
  const ana = readFileSync(join(D.REPO_ROOT, 'tests/htb_integration/mix_match_analyse.mjs'), 'utf8');
  assert.ok(ana.slice(ana.indexOf('export function admit(')).includes('exportWrapperFailures('));
});

test('the wrappers: their derivation, ceilings, gate names, driver and the pinned analyser sha', () => {
  const analyserSha = sha(readFileSync(join(D.REPO_ROOT, 'tests/htb_integration/mix_match_analyse.mjs')));
  const spec = spawnSync(join(D.REPO_ROOT, '.venv/bin/python'), ['-c', 'from tests import htb_mix_match_wrapper_spec as W\nfor s in ("match-smoke", "match"):\n    assert W.wrapper_path(s).read_text() == W.derive(s), s\nprint("ok")'], { cwd: D.REPO_ROOT, encoding: 'utf8' });
  assert.equal(spec.status, 0, spec.stderr);
  for (const [stage, term, kill, gate] of [['smoke', 1790, 1800, 'MIX_MATCH_SMOKE_AUTHORIZED'], ['match', 64790, 64800, 'MIX_MATCH_AUTHORIZED']]) {
    const w = readFileSync(join(D.REPO_ROOT, D.OUTER_REL[stage], '01_wrapper.sh'), 'utf8');
    assert.match(w, new RegExp(`^TERM_S=${term}$`, 'm'));
    assert.match(w, new RegExp(`^KILL_S=${kill}$`, 'm'));
    assert.match(w, new RegExp(`^GATE=${gate}$`, 'm'));
    assert.match(w, new RegExp(`^STAGE=${stage}$`, 'm'));
    assert.match(w, /^GATE_REL=tests\/htb_integration\/mix_authorization\.mjs$/m);
    assert.match(w, /^DRIVER=tests\/htb_integration\/mix_match\.mjs$/m);
    assert.match(w, /^ANALYSER=tests\/htb_integration\/mix_match_analyse\.mjs$/m);
    assert.match(w, new RegExp(`^ANALYSER_SHA256=${analyserSha}$`, 'm'));
    assert.match(w, /^EXPORT_REL=docs\/superpowers\/evidence\/htb-mix-export-2\/record\.json$/m);
    assert.match(w, /^PLAN_REL=docs\/superpowers\/2026-11-18-htb-mix-distillation-card\.md$/m);
    assert.match(w, /^RECORD_KIND=htb-mix-match-record\/1$/m);
    assert.ok(!/onpolicy/.test(w.slice(w.indexOf('set -u'))), `${stage}: no H-OP path below the header`);
    assert.match(w, /test_ort_shutdown\.js/);
    assert.match(w, /idle_gate wait/);
    assert.match(w, /background_compute check/);
    const syn = spawnSync('bash', ['-n', join(D.REPO_ROOT, D.OUTER_REL[stage], '01_wrapper.sh')], { encoding: 'utf8' });
    assert.equal(syn.status, 0, syn.stderr);
  }
  const m = readFileSync(join(D.REPO_ROOT, D.OUTER_REL.match, '01_wrapper.sh'), 'utf8');
  assert.match(m, /"\$NODE" "\$ANALYSER" timing/);
  assert.match(m, /t\.verdict === "LAUNCH_OK" && t\.match_cap_s === 64800/);
  assert.match(m, /^TIMING_REL=docs\/superpowers\/evidence\/htb-mix-match-smoke\/timing\.json$/m);
});
