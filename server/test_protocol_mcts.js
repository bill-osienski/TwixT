/**
 * V4: the research tie rule, opening readout and root sequential halving,
 * exactly equal to the Python search seed for seed, including ties. Fixture:
 * tests/fixtures/htb_integration/search.json (generate_fixtures.py), produced by
 * the research classes driven by the same fake evaluator. No model.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { TwixtState } from './gameLogic.js';
import { MCTS, MCTSNode } from './mcts.js';
import { MirrorAveragingInference } from './mirror_inference.js';
import { PyRandom } from './py_random.js';
import { ProtocolMCTS, TIE_EPS, sampleByTemperature } from './protocol_mcts.js';
import { HALVING, RootSequentialHalvingMCTS, score } from './root_halving.js';
import { fakeInference, loadFixture, staleSources } from '../tests/fixtures/htb_integration/fake_evaluator.mjs';

const FIX = loadFixture('search.json');
const evaluatorFor = (name) =>
  name === 'mirror_fake' ? new MirrorAveragingInference(fakeInference()) : fakeInference();
const nonZero = (counts) => Object.fromEntries([...counts].filter(([, v]) => v > 0));

test('the search fixture describes the research code as it is today', () => {
  assert.deepEqual(staleSources(FIX), []);
  assert.equal(FIX.tie_eps, TIE_EPS);
});

test('the fixture exercises the tie rule: exact, within-1e-8 and order-dependent ties all occur', () => {
  const sum = (k) => FIX.cases.reduce((a, c) => a + c.result.tie_stats[k], 0);
  assert.ok(sum('exact_ties') > 1000, `exact ties ${sum('exact_ties')}`);
  assert.ok(sum('near_ties') > 500, `near ties ${sum('near_ties')}`);
  assert.ok(sum('order_dependent') > 200, `order-dependent sets ${sum('order_dependent')}`);
});

for (const c of FIX.cases.filter((x) => x.kind === 'plain')) {
  test(`plain ${c.id} (ply ${c.ply}, ${c.evaluator}, N ${c.n_sims}): visits, root value and readout match`, async () => {
    const state = TwixtState.fromMoves(c.moves);
    const mcts = new ProtocolMCTS(evaluatorFor(c.evaluator), {
      nSimulations: c.n_sims,
      cPuct: 1.5,
      rng: new PyRandom(c.search_seed),
    });
    const { visitCounts, rootValue } = await mcts.search(state);
    assert.equal(visitCounts.size, c.result.n_legal);
    assert.deepEqual(nonZero(visitCounts), c.result.counts);
    assert.equal(rootValue, c.result.root_value);
    const move = sampleByTemperature(visitCounts, 1.0, new PyRandom(c.readout_seed));
    assert.equal(move, c.result.readout_move);
  });
}

for (const c of FIX.cases.filter((x) => x.kind === 'halving')) {
  test(`halving ${c.id} (ply ${c.ply}, m ${Math.min(16, c.result.n_legal)}, ${c.evaluator}): rounds, visits, choice and values match`, async () => {
    const state = TwixtState.fromMoves(c.moves);
    const mcts = new RootSequentialHalvingMCTS(evaluatorFor(c.evaluator), {
      nSimulations: c.n_sims,
      cPuct: 1.5,
      rng: new PyRandom(c.search_seed),
    });
    const out = await mcts.search(state);
    assert.deepEqual(mcts.lastRounds, c.result.rounds);
    assert.equal(out.visitCounts.size, c.result.n_legal);
    assert.deepEqual(nonZero(out.visitCounts), c.result.counts);
    assert.equal(out.rootValue, c.result.root_value);
    assert.equal(out.choice, c.result.choice);
    const child = mcts.lastRoot.children.get(out.choice);
    assert.equal(child.visitCount, c.result.choice_visits);
    assert.equal(-child.qValue, c.result.choice_value);
  });
}

test('a wrong search seed changes the result (the comparisons above are not vacuous)', async () => {
  const c = FIX.cases.find((x) => x.id === 'plain_empty_0');
  const mcts = new ProtocolMCTS(fakeInference(), { nSimulations: c.n_sims, rng: new PyRandom(c.search_seed + 1) });
  const { visitCounts } = await mcts.search(TwixtState.fromMoves(c.moves));
  assert.notDeepEqual(nonZero(visitCounts), c.result.counts);
});

test('the product rule would not reproduce the research search (the port is needed)', async () => {
  const c = FIX.cases.find((x) => x.id === 'plain_empty_0');
  const { visitCounts } = await new MCTS(fakeInference(), { nSimulations: c.n_sims }).search(
    TwixtState.fromMoves(c.moves)
  );
  assert.notDeepEqual(nonZero(visitCounts), c.result.counts);
});

test('at N = 200 with 16 candidates the allotment is 3 / 6 / 12 / 28 (cumulative 3 / 9 / 21 / 49)', () => {
  for (const c of FIX.cases.filter((x) => x.kind === 'halving' && x.result.n_legal >= 16)) {
    const [r1, r2, r3, r4] = c.result.rounds;
    const visits = (k) => c.result.counts[k] ?? 0;
    for (const k of r1.filter((x) => !r2.includes(x))) assert.equal(visits(k), 3, `${c.id} ${k}`);
    for (const k of r2.filter((x) => !r3.includes(x))) assert.equal(visits(k), 9, `${c.id} ${k}`);
    for (const k of r3.filter((x) => !r4.includes(x))) assert.equal(visits(k), 21, `${c.id} ${k}`);
    for (const k of r4) assert.equal(visits(k), 49, `${c.id} ${k}`);
  }
});

test('fewer candidates change the allotment: the played move reports its ACTUAL visits', () => {
  const m9 = FIX.cases.find((x) => x.id === 'halving_m9');
  assert.deepEqual(m9.result.rounds.map((r) => r.length), [9, 5, 3, 2]);
  // 200 // (4 * 9) = 5, then 10, then 16, then the remaining 57 split 29 / 28.
  assert.ok([59, 60].includes(m9.result.choice_visits), String(m9.result.choice_visits));
  const m1 = FIX.cases.find((x) => x.id === 'halving_m1');
  assert.equal(m1.result.choice_visits, 200);
});

test('the winning-child fixture plays a winning move', () => {
  const c = FIX.cases.find((x) => x.id === 'halving_winning_child');
  assert.ok(FIX.chain_wins.includes(c.result.choice), c.result.choice);
  assert.ok(c.result.rounds[0].some((k) => FIX.chain_wins.includes(k)));
});

test('the halving constants are the frozen variant', () => {
  assert.deepEqual(
    { ...HALVING },
    { FROM_PLY: 20, CANDIDATES: 16, C_VISIT: 50, C_SCALE: 1, PRIOR_FLOOR: 1e-12, BOARD: 24 }
  );
  assert.ok(Object.isFrozen(HALVING));
  assert.equal(score(0.5, 0, 0), Math.log(0.5) + 25);
  assert.equal(score(0, 1, 10), Math.log(1e-12) + 60);
});

// --- the tie rule, by construction -------------------------------------------

/** A node whose children all have the given scores at a fresh parent (q = 0, n = 0). */
function nodeWithScores(scores) {
  const node = new MCTSNode(new TwixtState({}));
  node.visitCount = 0; // sqrt(N + 1) = 1, so score = 1.5 * prior
  node.moves = scores.map((_, i) => [0, i + 1]);
  node.order = node.moves.slice();
  node.priors = new Map(node.moves.map((m, i) => [`${m[0]},${m[1]}`, scores[i] / 1.5]));
  return node;
}

/** Records every choice() set without consuming a real stream. */
const recordingRng = (log) => ({
  shuffle() {},
  choice(xs) {
    log.push(xs.map((m) => `${m[0]},${m[1]}`));
    return xs[0];
  },
});

test('ties: a running best within 1e-8 collects every member, and only those', () => {
  const log = [];
  const mcts = new ProtocolMCTS(fakeInference(), { rng: recordingRng(log) });
  mcts._selectChild(nodeWithScores([0.3, 0.5, 0.5 + 0.5e-8, 0.4, 0.5 - 0.9e-8]));
  assert.deepEqual(log[0], ['0,2', '0,3', '0,5']);
});

test('ties: the set depends on order, as the research rule does', () => {
  // a, a + 0.6e-8, a + 1.2e-8: scanned upward the third starts a new set only
  // if it is more than 1e-8 above the RUNNING best (a), which it is.
  const a = 0.5;
  const log = [];
  const mcts = new ProtocolMCTS(fakeInference(), { rng: recordingRng(log) });
  mcts._selectChild(nodeWithScores([a, a + 0.6e-8, a + 1.2e-8]));
  mcts._selectChild(nodeWithScores([a + 1.2e-8, a + 0.6e-8, a]));
  assert.deepEqual(log[0], ['0,3']);
  assert.deepEqual(log[1], ['0,1', '0,2']);
});

test('ties: choice() consumes the stream even for a single best move', () => {
  const r = new PyRandom(5);
  const twin = new PyRandom(5);
  const mcts = new ProtocolMCTS(fakeInference(), { rng: r });
  mcts._selectChild(nodeWithScores([0.1, 0.9]));
  twin.choice([0]);
  assert.equal(r.random(), twin.random());
});

test('the plain product search still breaks an exact tie lexicographically', async () => {
  const flat = {
    async evaluate(_b, moves) {
      return { priors: new Map(moves.map((m) => [`${m[0]},${m[1]}`, 1 / moves.length])), value: 0 };
    },
  };
  const { visitCounts } = await new MCTS(flat, { nSimulations: 1 }).search(new TwixtState({}));
  const visited = [...visitCounts].filter(([, v]) => v > 0).map(([k]) => k);
  assert.deepEqual(visited, ['0,1']); // the smallest "r,c" string among the legal moves
});

test('ProtocolMCTS refuses to run without its search stream', () => {
  assert.throws(() => new ProtocolMCTS(fakeInference(), {}), /search stream/);
});
