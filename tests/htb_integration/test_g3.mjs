/**
 * G3 driver and analyser, model-free (card docs/superpowers/2026-11-09-htb-g3-match-card.md §9).
 *
 * Fake evaluators and scripted players only: no model is loaded and no product game is played. The two player
 * adapters are checked against the product functions they wrap, with the G1 fake evaluator; the game loop, the
 * evidence, the timing formula, the verdict rule and the analyser's refusals are checked on scripted games.
 */
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, unlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

import { TwixtState } from '../../server/gameLogic.js';
import { MCTS } from '../../server/mcts.js';
import { MirrorAveragingInference } from '../../server/mirror_inference.js';
import { PyRandom } from '../../server/py_random.js';
import { protocolMove } from '../../server/protocol_move.js';
import { resolvePolicy, selectMoveForRequest } from '../../server/readout_policy.js';
import { fakeInference } from '../fixtures/htb_integration/fake_evaluator.mjs';
import * as A from './g3_analyse.mjs';
import * as GATES from './g3_authorization.mjs';
import * as D from './g3_match.mjs';

// ---------------------------------------------------------------------------------------------------------------
// Scripted games: a red chain that wins at ply 25, a black chain that wins at ply 26, and far-away filler moves.
// ---------------------------------------------------------------------------------------------------------------
const RED_CHAIN = [[0, 5], [2, 6], [4, 5], [6, 6], [8, 5], [10, 6], [12, 5], [14, 6], [16, 5], [18, 6], [20, 5], [22, 6], [23, 4]];
const BLACK_CHAIN = [[5, 0], [6, 2], [5, 4], [6, 6], [5, 8], [6, 10], [5, 12], [6, 14], [5, 16], [6, 18], [5, 20], [6, 22], [4, 23]];
const FAR_RED = [[1, 20], [3, 21], [1, 22], [3, 19], [7, 21], [9, 20], [11, 21], [13, 20], [15, 21], [17, 20], [19, 21], [21, 20], [13, 22], [15, 19]];
const FAR_BLACK = [[20, 1], [21, 3], [19, 3], [18, 1], [22, 2], [17, 3], [16, 1], [22, 4], [20, 3], [18, 3], [16, 3], [14, 1], [12, 2], [10, 1]];

const pick = (state, list) => {
  const m = list[Math.floor(state.ply / 2)];
  const legal = state.legalMoves();
  return m && legal.some((x) => x[0] === m[0] && x[1] === m[1]) ? m : legal[0];
};

/**
 * Two scripted players sharing a game counter. outcomeFor(gameIndex) says whether the new player wins that game; the
 * winner plays its colour's chain and the loser plays filler. The counter advances at ply 0 of every game.
 */
function scriptedPlayers(outcomeFor = () => true) {
  const ctx = { gameIndex: -1 };
  const play = (label) => ({
    label,
    modelId: label === 'new_hard' ? D.NEW_MODEL_ID : D.LEGACY_MODEL_ID,
    async move(state) {
      if (state.ply === 0) ctx.gameIndex += 1;
      const newWins = outcomeFor(ctx.gameIndex);
      const iWin = (label === 'new_hard') === newWins;
      const red = state.toMove === 'red';
      const list = iWin ? (red ? RED_CHAIN : BLACK_CHAIN) : red ? FAR_RED : FAR_BLACK;
      const m = pick(state, list);
      return label === 'new_hard'
        ? { moveKey: `${m[0]},${m[1]}`, nSimulations: 200, search: 'protocol', valueKind: state.ply < 20 ? 'root_mean' : 'played_move_mean' }
        : { moveKey: `${m[0]},${m[1]}`, nSimulations: 800, search: 'plain', valueKind: null };
    },
  });
  return { newHard: play('new_hard'), legacyHard: play('legacy_hard') };
}

/** A monotonic fake clock: +100 ms per call. */
const fakeClock = (start = 1000) => {
  let t = start;
  return () => (t += 100);
};

const TEST_BINDING = Object.freeze({
  digest: 'test-digest',
  runtime: { node: process.version },
  models: { new_hard: { model_id: D.NEW_MODEL_ID }, legacy_hard: { model_id: D.LEGACY_MODEL_ID } },
});
const HEADER = { binding_digest: TEST_BINDING.digest, head: 'f'.repeat(40), node_version: process.version, ort_version: '1.23.2' };

const PID = 4242;
/** outer.json as the wrapper writes it after a clean, completed run. */
const cleanOuter = (termS) => ({
  inner_exit: 0, outcome: 'COMPLETED', outer_exit: 0, gate_restored: true, survivors: '', record_by: 'script',
  wall_seconds: 300, term_s: termS, grace_s: 10, tree: '/tmp/w', head: HEADER.head, pre_open: 'e'.repeat(40), python: '/py', node: '/node',
});

/** runStage plus the record runCli writes and the wrapper's outer record, into a fresh temp dir. */
async function syntheticRun(stage, { nPairs, outcomeFor, killS = null, bgCheck = () => ({ ok: true, output: 'CLEAR' }) } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'g3-'));
  const outDir = join(root, stage);
  mkdirSync(outDir);
  const r = await D.runStage({
    stage,
    outDir,
    players: scriptedPlayers(outcomeFor),
    nPairs,
    bgCheck,
    header: HEADER,
    now: fakeClock(1000),
    startupMs: 1000,
  });
  writeFileSync(
    join(outDir, 'record.json'),
    JSON.stringify({ schema: D.RECORD_SCHEMA, stage, pairs: nPairs, head: HEADER.head, binding_digest: TEST_BINDING.digest, kill_s: killS, pid: PID, status: 'COMPLETED', ...r })
  );
  // The wrapper's side, as a clean run leaves it.
  const runDir = `${outDir}-run`;
  mkdirSync(runDir);
  writeFileSync(join(runDir, 'pgid.txt'), `${PID}\n`);
  writeFileSync(join(runDir, 'outer.json'), JSON.stringify(cleanOuter(stage === 'smoke' ? 2390 : killS - 10)));
  return { root, outDir, runDir };
}
const smokeSpec = A.SPEC;
const tinyMatchSpec = (pairs) => ({ ...A.SPEC, pairs: { ...A.SPEC.pairs, match: pairs }, verdictGames: 2 * pairs });

// ---------------------------------------------------------------------------------------------------------------

test('both launch gates are closed in the committed state', () => {
  assert.equal(GATES.G3_SMOKE_AUTHORIZED, false);
  assert.equal(GATES.G3_MATCH_AUTHORIZED, false);
  const src = readFileSync(new URL('./g3_authorization.mjs', import.meta.url), 'utf8');
  assert.match(src, /^export const G3_SMOKE_AUTHORIZED = false;$/m);
  assert.match(src, /^export const G3_MATCH_AUTHORIZED = false;$/m);
});

test('preflight refuses a closed gate before reading anything else (the real state)', () => {
  for (const stage of ['smoke', 'match']) {
    assert.throws(() => D.preflight(stage, { python: '/bin/sh', killS: 60 }, {}), (e) => e.code === 'GATE_CLOSED');
  }
});

test('driver and analyser transcribe the same card constants', () => {
  assert.equal(A.SPEC.schema, D.SCHEMA);
  assert.equal(A.SPEC.recordSchema, D.RECORD_SCHEMA);
  assert.equal(A.SPEC.matchSeed, D.MATCH_SEED);
  assert.deepEqual(A.SPEC.pairs, { ...D.PAIRS });
  assert.equal(A.SPEC.newModelId, D.NEW_MODEL_ID);
  assert.equal(A.SPEC.legacyModelId, D.LEGACY_MODEL_ID);
  assert.equal(A.SPEC.sims.new_hard, resolvePolicy({ difficulty: 'hard', hardMode: 'protocol' }).nSims);
  assert.equal(A.SPEC.sims.legacy_hard, resolvePolicy({ difficulty: 'hard', hardMode: 'legacy' }).nSims);
  assert.ok(Object.isFrozen(A.SPEC) && Object.isFrozen(A.SPEC.pairs));
});

test('seeds: deterministic, 48-bit, re-derived identically by the analyser, and all distinct across both stages', () => {
  assert.deepEqual(D.moveSeeds('match', 3, 1, 41), D.moveSeeds('match', 3, 1, 41));
  const seen = new Set();
  for (const stage of ['smoke', 'match']) {
    for (let p = 0; p < D.PAIRS[stage]; p++) {
      for (const g of [0, 1]) {
        for (let ply = g; ply < 577; ply += 2) {
          const s = D.moveSeeds(stage, p, g, ply);
          assert.ok(Number.isSafeInteger(s.search) && s.search < 2 ** 48 && s.readout < 2 ** 48);
          if (ply % 50 === g) assert.deepEqual(A.expectedSeeds(A.SPEC, stage, p, g, ply), s);
          for (const x of [s.search, s.readout]) {
            assert.ok(!seen.has(x), `seed collision at ${stage} ${p} ${g} ${ply}`);
            seen.add(x);
          }
        }
      }
    }
  }
});

/** Random legal play to `plies` without a win. */
function randomPosition(plies, seed) {
  const rng = new PyRandom(seed);
  let s = new TwixtState({});
  while (s.ply < plies) {
    const legal = s.legalMoves();
    rng.shuffle(legal);
    s = s.applyMove(legal.find((m) => s.applyMove(m).winner() === null));
  }
  return s;
}

test('the new-hard adapter is protocolMove with the given seeds (plies 0 and 23), reproducibly', async () => {
  const inference = new MirrorAveragingInference(fakeInference());
  const player = D.makeNewHardPlayer({ inference, modelId: D.NEW_MODEL_ID });
  const policy = resolvePolicy({ difficulty: 'hard', hardMode: 'protocol' });
  for (const state of [new TwixtState({}), randomPosition(23, 7)]) {
    const seeds = D.moveSeeds('match', 0, 0, state.ply);
    const a = await player.move(state, { seeds });
    const b = await player.move(state, { seeds });
    const direct = await protocolMove({ gameState: state, policy, inference, seeds });
    assert.equal(a.moveKey, direct.moveKey);
    assert.equal(b.moveKey, a.moveKey);
    assert.equal(a.valueKind, state.ply < 20 ? 'root_mean' : 'played_move_mean');
    assert.equal(a.nSimulations, 200);
  }
});

test('the legacy adapter is computeBestMove\'s plain branch under HARD_MODE=legacy, and deterministic', async () => {
  const inference = fakeInference();
  const player = D.makeLegacyHardPlayer({ inference, modelId: D.LEGACY_MODEL_ID });
  const state = randomPosition(12, 3);
  const a = await player.move(state, {});
  const b = await player.move(state, {});
  const mcts = new MCTS(inference, { nSimulations: 800 });
  const { visitCounts } = await mcts.search(state);
  const { moveKey } = selectMoveForRequest({ visitCounts, difficulty: 'hard', hardMode: 'legacy', selectMove: (c, t) => mcts.selectMove(c, t) });
  assert.equal(a.moveKey, moveKey);
  assert.equal(b.moveKey, moveKey);
  assert.deepEqual([a.nSimulations, a.search, a.valueKind], [800, 'plain', null]);
});

test('playGame: colours, seeds for the new side only, the product end rule, and the score', async () => {
  for (const newIsRed of [true, false]) {
    const g = await D.playGame({ ...(({ newHard, legacyHard }) => ({ newPlayer: newHard, legacyPlayer: legacyHard }))(scriptedPlayers()), newIsRed, stage: 'smoke', pair: 0, game: newIsRed ? 0 : 1, now: fakeClock() });
    assert.equal(g.result, newIsRed ? 'red' : 'black');
    assert.equal(g.newScore, 1);
    assert.equal(g.termination, 'win');
    assert.equal(g.plyCount, newIsRed ? 25 : 26);
    for (const pm of g.perMove) {
      const isNew = (pm.ply % 2 === 0) === newIsRed;
      assert.equal(pm.player, isNew ? 'new_hard' : 'legacy_hard');
      assert.deepEqual(pm.seeds, isNew ? D.moveSeeds('smoke', 0, newIsRed ? 0 : 1, pm.ply) : null);
      assert.equal(pm.ms, 100);
    }
  }
  const lose = await D.playGame({ newPlayer: scriptedPlayers(() => false).newHard, legacyPlayer: scriptedPlayers(() => false).legacyHard, newIsRed: true, stage: 'smoke', pair: 0, game: 0, now: fakeClock() });
  assert.equal(lose.newScore, 0);
});

test('playGame stops on the abort signal with no partial result', async () => {
  const c = new AbortController();
  c.abort();
  const p = scriptedPlayers();
  await assert.rejects(D.playGame({ newPlayer: p.newHard, legacyPlayer: p.legacyHard, newIsRed: true, stage: 'match', pair: 0, game: 0, signal: c.signal }), (e) => e instanceof D.G3Void && e.step === 'terminated');
});

test('runStage stops VOID on a failed background-compute check, after that game\'s sidecar', async () => {
  await assert.rejects(syntheticRun('smoke', { nPairs: 1, bgCheck: () => ({ ok: false, output: 'BOINC DETECTED' }) }), (e) => e instanceof D.G3Void && e.step === 'background_compute');
});

test('timing: a valid smoke validates, and the card §5 formula gives the hand-computed ceiling', async () => {
  const { root, outDir, runDir } = await syntheticRun('smoke', { nPairs: 1 });
  try {
    const run = A.validateRun({ dir: outDir, runDir, stage: 'smoke', binding: TEST_BINDING, spec: smokeSpec });
    assert.deepEqual(run.failures, []);
    const t = A.timingFrom(run, smokeSpec);
    // game 0: 25 plies, game 1: 26; each move 100 ms; each game's segment adds its 2 clock calls per move + 1.
    assert.deepEqual(t.inputs.smoke_plies, [25, 26]);
    assert.equal(t.inputs.projected_plies, 64); // max(64, 25.5)
    assert.equal(t.inputs.new_hard_s_per_move, 0.1);
    assert.equal(t.inputs.legacy_hard_s_per_move, 0.1);
    assert.ok(Math.abs(t.inputs.overhead_s_per_game - 2.65) < 1e-12); // (5100 - 2500 + 5300 - 2600) / 2 ms
    assert.equal(t.inputs.startup_s, 1);
    assert.ok(Math.abs(t.projection_s - (1 + 220 * (32 * 0.2 + 2.65))) < 1e-9); // 1,992 s
    assert.equal(t.kill_s, 3000); // ceil(2,988 / 60) = 50 minutes
    assert.equal(t.term_s, 2990);
    assert.equal(t.cleanup_through_s, 3010);
    assert.equal(t.verdict, 'WITHIN_24H');
    assert.equal(A.timingFrom(run, { ...smokeSpec, limitS: 2999 }).verdict, 'EXCEEDS_24H');
    // A mean length above 64 is used as measured.
    const long = { ...run, games: run.games.map((g) => ({ ...g, sc: { ...g.sc, ply_count: 100 } })) };
    assert.equal(A.timingFrom(long, smokeSpec).inputs.projected_plies, 100);
  } finally {
    rmSync(root, { recursive: true });
  }
});

test('the analyser refuses tampered evidence (each against an untampered control)', async () => {
  const { root, outDir, runDir } = await syntheticRun('smoke', { nPairs: 1 });
  const g0 = join(outDir, 'games', 'pair_0000_game_0.json');
  const original = readFileSync(g0, 'utf8');
  const check = () => A.validateRun({ dir: outDir, runDir, stage: 'smoke', binding: TEST_BINDING, spec: smokeSpec }).failures;
  const tampers = {
    illegal: (s) => ((s.moves[3] = s.moves[1]), s),
    seed: (s) => ((s.per_move[0].seeds.search += 1), s),
    score: (s) => ((s.new_score = 0), s),
    colour: (s) => ((s.red_player = 'legacy_hard'), s),
    model: (s) => ((s.new_model_id = D.LEGACY_MODEL_ID), s),
    sims: (s) => ((s.per_move[1].n_simulations = 400), s),
    valueKind: (s) => ((s.per_move[22].value_kind = 'root_mean'), s),
    binding: (s) => ((s.binding_digest = 'other'), s),
    truncated: (s) => (s.moves.pop(), s.per_move.pop(), (s.ply_count -= 1), s),
    timing: (s) => ((s.per_move[2].ms += 5), s),
  };
  try {
    assert.deepEqual(check(), []);
    for (const [name, fn] of Object.entries(tampers)) {
      writeFileSync(g0, JSON.stringify(fn(JSON.parse(original))));
      assert.ok(check().length > 0, `tamper ${name} was not caught`);
      writeFileSync(g0, original);
      assert.deepEqual(check(), [], `control after ${name}`);
    }
    const rec = join(outDir, 'record.json');
    const recText = readFileSync(rec, 'utf8');
    writeFileSync(rec, JSON.stringify({ ...JSON.parse(recText), status: 'VOID' }));
    assert.ok(check().some((m) => m.includes('status')));
    writeFileSync(rec, recText);
    writeFileSync(join(outDir, 'games', 'extra.json'), '{}');
    assert.ok(check().length > 0);
    unlinkSync(join(outDir, 'games', 'extra.json'));
    unlinkSync(join(outDir, 'games', 'pair_0000_game_1.json'));
    assert.ok(check().length > 0);
  } finally {
    rmSync(root, { recursive: true });
  }
});

test('the verdict rule: mean - sqrt(ln 20 / 220) > 0.5; 135.5 / 220 fails, 136 / 220 passes', () => {
  const scores = (total) => {
    const s = Array(220).fill(0);
    let left = total;
    for (let i = 0; i < 220 && left > 0; i++) {
      s[i] = left >= 1 ? 1 : 0.5;
      left -= s[i];
    }
    return s;
  };
  assert.equal(A.verdictFromScores(scores(135.5)).verdict, 'FAIL');
  assert.equal(A.verdictFromScores(scores(136)).verdict, 'PASS');
  assert.ok(Math.abs(A.verdictFromScores(scores(136)).hoeffding_bound - Math.sqrt(Math.log(20) / 220)) < 1e-15);
  assert.throws(() => A.verdictFromScores(scores(10).slice(1)));
});

test('a synthetic match validates under its timing and is scored (tiny spec, both outcomes)', async () => {
  const timing = { verdict: 'WITHIN_24H', kill_s: 3000, term_s: 2990, projection_s: 1992 };
  const pairs = 4;
  const { root, outDir, runDir } = await syntheticRun('match', { nPairs: pairs, outcomeFor: (k) => k !== 3, killS: 3000 });
  try {
    const spec = tinyMatchSpec(pairs);
    const run = A.validateRun({ dir: outDir, runDir, stage: 'match', binding: TEST_BINDING, spec, timing });
    assert.deepEqual(run.failures, []);
    const v = A.verdictFrom(run, timing, spec);
    assert.equal(v.total_score, 7);
    assert.equal(v.new_wins, 7);
    assert.equal(v.legacy_wins, 1);
    assert.equal(v.pair_score_counts['0.5'], 1);
    assert.equal(v.pair_score_counts['1'], 3);
    assert.equal(v.distinct_games, 2); // scripted games repeat; real ones are seeded
    assert.ok(A.validateRun({ dir: outDir, runDir, stage: 'match', binding: TEST_BINDING, spec, timing: { ...timing, kill_s: 3060, term_s: 3050 } }).failures.length > 0);
    assert.ok(A.validateRun({ dir: outDir, runDir, stage: 'match', binding: TEST_BINDING, spec, timing: null }).failures.length > 0);
  } finally {
    rmSync(root, { recursive: true });
  }
});

test('the binding covers the driver\'s and analyser\'s whole import closure, minus the gate, deterministically', () => {
  const driver = D.importClosure(D.DRIVER_RELPATH);
  for (const f of ['server/protocol_move.js', 'server/root_halving.js', 'server/protocol_mcts.js', 'server/py_random.js', 'server/mcts.js', 'server/gameLogic.js', 'server/inference.js', 'server/mirror_inference.js', 'server/model_manifest.js', 'server/readout_policy.js', D.GATE_RELPATH]) {
    assert.ok(driver.includes(f), f);
  }
  const analyser = D.importClosure(D.ANALYSER_RELPATH);
  assert.ok(!analyser.includes(D.DRIVER_RELPATH), 'the analyser must not import the driver');
  assert.deepEqual(analyser, [D.ANALYSER_RELPATH, 'server/gameLogic.js'].sort());
});

test('the committed binding equals a fresh computation (no file in it has moved)', () => {
  const committed = JSON.parse(readFileSync(new URL('./g3_binding.json', import.meta.url), 'utf8'));
  const fresh = D.computeBinding();
  assert.deepEqual(D.bindingDiff(committed, fresh), []);
  assert.ok(!(D.GATE_RELPATH in fresh.driver_files));
  assert.deepEqual(A.selfCheck(committed), []);
  assert.deepEqual(D.bindingDiff(fresh, { ...fresh, match_seed: 1 }), ['match_seed']);
});

// ---------------------------------------------------------------------------------------------------------------
// The wrapper's outer record (the user's launch-clearance review, 2026-11-09)
// ---------------------------------------------------------------------------------------------------------------

/** Failure modes of the outer evidence; each is checked against the untampered control. */
const OUTER_TAMPERS = {
  'missing outer.json': null,
  'TIMEOUT over an inner COMPLETED': (o) => ({ ...o, inner_exit: 124, outer_exit: 124, outcome: 'TIMEOUT' }),
  'gate not restored': (o) => ({ ...o, gate_restored: false, outer_exit: 5 }),
  'survivors': (o) => ({ ...o, survivors: '4243 ' }),
  'unconfirmed cleanup': (o) => ({ ...o, inner_exit: 125, outer_exit: 125, outcome: 'TIMEOUT', survivors: 'unconfirmed cleanup: process group 4242 4243 ' }),
  'record written by the wrapper': (o) => ({ ...o, record_by: 'wrapper' }),
  'string exit': (o) => ({ ...o, inner_exit: '0' }),
  'another run (head)': (o) => ({ ...o, head: 'd'.repeat(40) }),
  'another ceiling': (o) => ({ ...o, term_s: 2990 }),
};

test('the analyser judges the wrapper\'s outer record with the inner one (each failure against a clean control)', async () => {
  const { root, outDir, runDir } = await syntheticRun('smoke', { nPairs: 1 });
  const outerPath = join(runDir, 'outer.json');
  const pgidPath = join(runDir, 'pgid.txt');
  const original = readFileSync(outerPath, 'utf8');
  const check = () => A.validateRun({ dir: outDir, runDir, stage: 'smoke', binding: TEST_BINDING, spec: smokeSpec }).failures;
  try {
    assert.deepEqual(check(), []);
    for (const [name, fn] of Object.entries(OUTER_TAMPERS)) {
      if (fn === null) unlinkSync(outerPath);
      else writeFileSync(outerPath, JSON.stringify(fn(JSON.parse(original))));
      assert.ok(check().some((m) => m.startsWith('wrapper:')), `${name} was not caught`);
      writeFileSync(outerPath, original);
      assert.deepEqual(check(), [], `control after ${name}`);
    }
    writeFileSync(pgidPath, '9999\n'); // another process group: the outer record is not this run's
    assert.ok(check().some((m) => m.includes('pgid')));
    writeFileSync(pgidPath, `${PID}\n`);
    assert.deepEqual(check(), []);
    // With a TIMEOUT outer record the run gets no timing at all from the CLI path: validateRun fails first.
    writeFileSync(outerPath, JSON.stringify(OUTER_TAMPERS['TIMEOUT over an inner COMPLETED'](JSON.parse(original))));
    assert.ok(A.validateRun({ dir: outDir, runDir, stage: 'smoke', binding: TEST_BINDING, spec: smokeSpec }).failures.length > 0);
  } finally {
    rmSync(root, { recursive: true });
  }
});

test('the driver\'s outer check agrees with the analyser\'s on every case', () => {
  const record = { status: 'COMPLETED', head: HEADER.head, pid: PID };
  const pgid = `${PID}\n`;
  const outer = cleanOuter(2390);
  assert.deepEqual(D.outerProblems({ outer, record, pgid, termS: 2390 }), []);
  assert.deepEqual(A.outerFailures({ outer, record, pgid, termS: 2390 }), []);
  for (const [name, fn] of Object.entries(OUTER_TAMPERS)) {
    const o = fn === null ? null : fn(outer);
    assert.ok(D.outerProblems({ outer: o, record, pgid, termS: 2390 }).length > 0, `driver: ${name}`);
    assert.ok(A.outerFailures({ outer: o, record, pgid, termS: 2390 }).length > 0, `analyser: ${name}`);
  }
  assert.ok(D.outerProblems({ outer, record: { ...record, status: 'VOID' }, pgid, termS: 2390 }).length > 0);
  assert.ok(D.outerProblems({ outer, record, pgid: '1\n', termS: 2390 }).length > 0);
});

/** Committed smoke evidence, as smokeAdmission reads it, from an in-memory map. */
function admissionFixture(edit = (f) => f) {
  const files = edit({
    'docs/superpowers/evidence/htb-g3-smoke/record.json': JSON.stringify({ stage: 'smoke', status: 'COMPLETED', head: HEADER.head, pid: PID, binding_digest: 'B' }),
    'docs/superpowers/evidence/htb-g3-smoke/timing.json': JSON.stringify({ verdict: 'WITHIN_24H', kill_s: 46080, term_s: 46070 }),
    'docs/superpowers/evidence/htb-g3-smoke-run/outer.json': JSON.stringify(cleanOuter(2390)),
    'docs/superpowers/evidence/htb-g3-smoke-run/pgid.txt': `${PID}\n`,
  });
  return {
    bound: { digest: 'B' },
    readCommitted: (rel) => {
      if (!(rel in files)) throw new Error(`${rel} is not committed`);
      return files[rel];
    },
    untrackedOrChanged: () => [],
  };
}

test('match admission requires committed, successful smoke evidence (each failure against a clean control)', () => {
  const ok = admissionFixture();
  assert.equal(D.smokeAdmission({ ...ok, killS: 46080 }), 46080);
  const refused = (fx, killS = 46080, code = 'SMOKE_EVIDENCE') =>
    assert.throws(() => D.smokeAdmission({ ...fx, killS }), (e) => e.code === code);
  const outerPath = 'docs/superpowers/evidence/htb-g3-smoke-run/outer.json';
  refused(admissionFixture((f) => (delete f[outerPath], f)));
  refused(admissionFixture((f) => (delete f['docs/superpowers/evidence/htb-g3-smoke-run/pgid.txt'], f)));
  for (const [name, fn] of Object.entries(OUTER_TAMPERS)) {
    if (fn === null) continue;
    refused(admissionFixture((f) => ({ ...f, [outerPath]: JSON.stringify(fn(cleanOuter(2390))) })), 46080, 'SMOKE_EVIDENCE');
    assert.equal(D.smokeAdmission({ ...admissionFixture(), killS: 46080 }), 46080, `control after ${name}`);
  }
  refused({ ...ok, untrackedOrChanged: () => ['?? docs/superpowers/evidence/htb-g3-smoke/games/x.json'] });
  refused({ ...ok, bound: { digest: 'other' } });
  refused(admissionFixture((f) => ({ ...f, 'docs/superpowers/evidence/htb-g3-smoke/timing.json': JSON.stringify({ verdict: 'EXCEEDS_24H', kill_s: 90000 }) })), 90000, 'TIMING');
  refused(ok, 46020, 'TIMING');
});

test('match admission refuses in the real state: no smoke evidence is committed', () => {
  assert.throws(
    () => D.smokeAdmission({ bound: { digest: 'x' }, killS: 60, readCommitted: D.gitReadCommitted, untrackedOrChanged: D.gitUntrackedOrChanged }),
    (e) => e.code === 'SMOKE_EVIDENCE'
  );
});

test('the binding hashes the Node executable and every file of the two runtime packages', () => {
  const b = D.computeBinding();
  assert.equal(b.runtime.node_executable.path, process.execPath);
  assert.match(b.runtime.node_executable.sha256, /^[0-9a-f]{64}$/);
  const ort = b.runtime.packages['node_modules/onnxruntime-node'];
  for (const f of ['package.json', 'dist/index.js', 'dist/binding.js', 'bin/napi-v6/darwin/arm64/onnxruntime_binding.node', 'bin/napi-v6/darwin/arm64/libonnxruntime.1.23.2.dylib']) {
    assert.ok(`node_modules/onnxruntime-node/${f}` in ort, f);
  }
  assert.ok('node_modules/onnxruntime-common/package.json' in b.runtime.packages['node_modules/onnxruntime-common']);
});
