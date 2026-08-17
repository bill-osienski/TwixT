#!/usr/bin/env node
/**
 * Focused tests for MCTS lazy child-state materialization.
 *
 * The change defers constructing a child `TwixtState` until PUCT first descends
 * into that move. These tests pin the properties that must NOT change, and the
 * one that must: retention bounded by simulations rather than by
 * simulations × legal moves.
 *
 * A deterministic fake inference is used throughout — `server/mcts.js` has no
 * imports of its own, so the real search runs with no model.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { TwixtState } from './gameLogic.js';
import { MCTS, MCTSNode } from './mcts.js';

/** Priors strictly decreasing in move order, so the argmax is unambiguous. */
const fakeInference = (valueFor = (n) => ((n % 11) / 11) * 2 - 1) => ({
  calls: 0,
  async evaluate(_boardTensor, moves) {
    this.calls += 1;
    const denom = (moves.length * (moves.length + 1)) / 2;
    const priors = new Map(
      moves.map((m, i) => [`${m[0]},${m[1]}`, (moves.length - i) / denom])
    );
    return { priors, value: valueFor(moves.length) };
  },
});

/** Flat priors, so every candidate ties and the tie-break decides alone. */
const flatInference = () => ({
  async evaluate(_boardTensor, moves) {
    const p = 1 / moves.length;
    return { priors: new Map(moves.map((m) => [`${m[0]},${m[1]}`, p])), value: 0 };
  },
});

/** A small board keeps these tests fast; the mechanism is size-independent. */
const smallState = () => new TwixtState({ boardSize: 8 });

/** Count TwixtState constructions across a window (applyMove -> copy). */
async function countCopies(fn) {
  const original = TwixtState.prototype.copy;
  let n = 0;
  TwixtState.prototype.copy = function counting(...args) {
    n += 1;
    return original.apply(this, args);
  };
  try {
    await fn();
  } finally {
    TwixtState.prototype.copy = original;
  }
  return n;
}

test('expansion records candidate moves but materializes NO children', async () => {
  const mcts = new MCTS(fakeInference(), { nSimulations: 1, cPuct: 1.5 });
  const root = new MCTSNode(smallState());
  const copies = await countCopies(() => mcts._expand(root));

  assert.ok(root.moves.length > 0);
  assert.equal(root.moves.length, root.state.legalMoves().length);
  assert.equal(root.priors.size, root.moves.length, 'every legal move needs a prior');
  assert.equal(root.children.size, 0, 'expansion materialized children');
  assert.equal(copies, 0, 'expansion constructed states');
});

test('descending materializes exactly ONE child, and only the selected one', async () => {
  const mcts = new MCTS(fakeInference(), { nSimulations: 1, cPuct: 1.5 });
  const root = new MCTSNode(smallState());
  await mcts._expand(root);

  const copies = await countCopies(async () => {
    mcts._selectChild(root);
  });
  assert.equal(copies, 1, 'selection constructed more than the winner');
  assert.equal(root.children.size, 1);
});

test('re-selecting the same move does not construct a second state', async () => {
  const mcts = new MCTS(fakeInference(), { nSimulations: 1, cPuct: 1.5 });
  const root = new MCTSNode(smallState());
  await mcts._expand(root);

  const [firstKey, firstChild] = mcts._selectChild(root);
  const copies = await countCopies(async () => {
    const [againKey, againChild] = mcts._selectChild(root);
    assert.equal(againKey, firstKey);
    assert.equal(againChild, firstChild, 'a second object was created for one move');
  });
  assert.equal(copies, 0);
});

test('an unmaterialized move scores identically to a materialized-but-unvisited one', async () => {
  // The load-bearing claim: deferring construction cannot change selection,
  // because an unvisited child contributes q = 0 and N_child = 0, both
  // derivable from priors alone, and its state is never read while scoring.
  const mcts = new MCTS(fakeInference(), { nSimulations: 1, cPuct: 1.5 });

  const lazyRoot = new MCTSNode(smallState());
  await mcts._expand(lazyRoot);
  const [lazyPick] = mcts._selectChild(lazyRoot);

  // Same position, but every child materialized up front and left unvisited.
  const eagerRoot = new MCTSNode(smallState());
  await mcts._expand(eagerRoot);
  for (const move of eagerRoot.moves) {
    const key = `${move[0]},${move[1]}`;
    eagerRoot.children.set(key, new MCTSNode(eagerRoot.state.applyMove(move), eagerRoot, move));
  }
  const [eagerPick] = mcts._selectChild(eagerRoot);

  assert.equal(lazyPick, eagerPick, 'pre-materializing the children changed the choice');
});

test('with all candidates tied, the lexicographically smallest key wins', async () => {
  const mcts = new MCTS(flatInference(), { nSimulations: 1, cPuct: 1.5 });
  const root = new MCTSNode(new TwixtState({ boardSize: 24 }));
  await mcts._expand(root);

  const [pick] = mcts._selectChild(root);
  assert.equal(pick, [...root.priors.keys()].sort()[0]);
});

test('the tie-break is STRING lexicographic, not numeric', async () => {
  // The two orderings only disagree on multi-digit rows: "10,5" < "9,3" as
  // strings, while 10 > 9 numerically. A real board's minimum is "0,1" under
  // both, so it cannot tell them apart — this pins the semantics directly.
  assert.equal('10,5' < '9,3', true, 'premise: string order differs from numeric here');

  const mcts = new MCTS(flatInference(), { nSimulations: 1, cPuct: 1.5 });
  const root = new MCTSNode(new TwixtState({ boardSize: 24 }));
  // Two candidates only, both tied, chosen so the orderings disagree.
  root.moves = [
    [9, 3],
    [10, 5],
  ];
  root.priors = new Map([
    ['9,3', 0.5],
    ['10,5', 0.5],
  ]);

  const [pick] = mcts._selectChild(root);
  assert.equal(pick, '10,5', 'the tie-break became numeric — this changes played moves');
});

test('a strictly better score wins, and an equal one only wins if lexicographically smaller', async () => {
  const mcts = new MCTS(flatInference(), { nSimulations: 1, cPuct: 1.5 });
  const root = new MCTSNode(smallState());
  await mcts._expand(root);
  const keys = [...root.priors.keys()];
  const sorted = [...keys].sort();

  // All tied -> lexicographically smallest wins.
  assert.equal(mcts._selectChild(root)[0], sorted[0]);

  // Give a LATER-sorting move a strictly better score by visiting a different
  // one badly: a visited child with a negative q lowers its own score.
  const worstKey = sorted[0];
  const worst = root.children.get(worstKey);
  worst.visitCount = 5;
  worst.valueSum = 5; // qValue 1 -> q = -1 for the parent, a bad option
  const [pickAfter] = mcts._selectChild(root);
  assert.notEqual(pickAfter, worstKey, 'a strictly worse option was still chosen');
});

test('visitCounts covers EVERY legal root move, in order, including zeros', async () => {
  const mcts = new MCTS(fakeInference(), { nSimulations: 3, cPuct: 1.5 });
  const state = smallState();
  const { visitCounts } = await mcts.search(state);

  const legalKeys = state.legalMoves().map((m) => `${m[0]},${m[1]}`);
  assert.deepEqual([...visitCounts.keys()], legalKeys, 'key set or order changed');
  assert.equal(visitCounts.size, legalKeys.length);

  const zeros = [...visitCounts.values()].filter((v) => v === 0).length;
  assert.ok(zeros > 0, 'no zero-count entries survived — index.js publishes these');
  assert.equal(
    [...visitCounts.values()].reduce((s, v) => s + v, 0),
    3,
    'every simulation backs up through exactly one root child'
  );
});

test('retention is bounded by simulations, not simulations x legal moves', async () => {
  const S = 8;
  const state = smallState();
  const L = state.legalMoves().length;
  const mcts = new MCTS(fakeInference(), { nSimulations: S, cPuct: 1.5 });

  const copies = await countCopies(() => mcts.search(state));

  assert.ok(L >= 20, `sanity: only ${L} legal moves`);
  assert.ok(copies <= S, `copies=${copies} exceeded the ${S} the design claims`);
  assert.ok(
    copies < L,
    `copies=${copies} is not below the legal-move count ${L}; retention still scales with L`
  );
});

test('an aborted-before-first-simulation search still returns the empty map', async () => {
  const mcts = new MCTS(fakeInference(), { nSimulations: 8, cPuct: 1.5 });
  const controller = new AbortController();
  controller.abort();
  const out = await mcts.search(smallState(), { signal: controller.signal });
  assert.equal(out.visitCounts.size, 0);
  assert.equal(out.rootValue, 0);
});

test('progress reports one entry per simulation with the running root value', async () => {
  const mcts = new MCTS(fakeInference(), { nSimulations: 4, cPuct: 1.5 });
  const seen = [];
  await mcts.search(smallState(), {
    progressEvery: 1,
    onProgress: ({ done, total, valueEstimate }) => seen.push({ done, total, valueEstimate }),
  });
  assert.deepEqual(seen.map((p) => p.done), [1, 2, 3, 4]);
  assert.ok(seen.every((p) => p.total === 4));
  assert.ok(seen.every((p) => p.valueEstimate >= -1 && p.valueEstimate <= 1));
});

test('one network evaluation per expansion, and none for a re-descended child', async () => {
  const inference = fakeInference();
  const mcts = new MCTS(inference, { nSimulations: 5, cPuct: 1.5 });
  await mcts.search(smallState());
  // 1 root expansion + at most one leaf expansion per simulation.
  assert.ok(inference.calls <= 6, `evaluate called ${inference.calls} times`);
  assert.ok(inference.calls >= 2);
});
