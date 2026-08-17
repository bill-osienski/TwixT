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

test('each expansion performs exactly one evaluation', async () => {
  // Counted against the expansions actually performed, not against a range: a
  // bound of "between 2 and 6" neither shows an expansion happened nor detects
  // a duplicate evaluation inside it.
  const inference = fakeInference();
  const mcts = new MCTS(inference, { nSimulations: 1, cPuct: 1.5 });

  const root = new MCTSNode(smallState());
  assert.equal(inference.calls, 0);
  await mcts._expand(root);
  assert.equal(inference.calls, 1, 'root expansion');

  const [, child] = mcts._selectChild(root);
  assert.equal(inference.calls, 1, 'selection evaluated something');
  await mcts._expand(child);
  assert.equal(inference.calls, 2, 'child expansion');
});

test('re-selecting a materialized move reuses the cached child object', async () => {
  // Narrow by design: this is about `children` caching, NOT about evaluation.
  // `_selectChild` cannot evaluate anything, so asserting "no new evaluation"
  // here would be vacuous — that claim is made by the search-level test below.
  const inference = fakeInference();
  const mcts = new MCTS(inference, { nSimulations: 1, cPuct: 1.5 });

  const root = new MCTSNode(smallState());
  await mcts._expand(root);
  const [firstKey, firstChild] = mcts._selectChild(root);

  // qValue -1 for the child means q = +1 from the parent's perspective, beating
  // every unvisited sibling's q = 0, so the next selection must return it.
  firstChild.visitCount = 1;
  firstChild.valueSum = -1;

  const copies = await countCopies(async () => {
    const [againKey, againChild] = mcts._selectChild(root);
    assert.equal(againKey, firstKey, 'the setup failed to force a re-selection');
    assert.equal(againChild, firstChild, 'a second node was created for the same move');
  });
  assert.equal(copies, 0, 're-selecting reconstructed the child state');
  assert.equal(root.children.size, 1, 'a duplicate child entry was added');
});

test('across a REAL search, no node is expanded twice and traversal repeats', async () => {
  // The property that matters, asserted at the level where it can actually be
  // violated: a full search re-descends through already-expanded nodes many
  // times, and must never expand the same node object again.
  const inference = fakeInference();
  const state = smallState();
  // The simulation count must comfortably exceed the branching factor, or every
  // simulation just expands another fresh root child and no node is ever
  // re-traversed. Measured on this board (48 legal moves): 53 simulations gives
  // ZERO re-traversed children, 100 gives 12. The first draft used 12 and
  // proved nothing — which this test's own final assertion caught.
  const nSimulations = 3 * state.legalMoves().length;
  const mcts = new MCTS(inference, { nSimulations, cPuct: 1.5 });

  const expandedNodes = [];
  const realExpand = mcts._expand.bind(mcts);
  mcts._expand = (node) => {
    expandedNodes.push(node);
    return realExpand(node);
  };

  const traversedFrom = [];
  const realSelect = mcts._selectChild.bind(mcts);
  mcts._selectChild = (node) => {
    traversedFrom.push(node);
    return realSelect(node);
  };

  await mcts.search(state);

  assert.equal(
    new Set(expandedNodes).size,
    expandedNodes.length,
    'the same node object was expanded more than once'
  );
  assert.equal(
    inference.calls,
    expandedNodes.length,
    'evaluations did not match expansions one-for-one'
  );

  // Prove re-traversal actually happened, rather than assuming it: some node
  // was selected from more than once, and at least one of those was a CHILD
  // (the root is trivially re-traversed every simulation).
  const timesSelectedFrom = new Map();
  for (const node of traversedFrom) {
    timesSelectedFrom.set(node, (timesSelectedFrom.get(node) ?? 0) + 1);
  }
  const repeats = [...timesSelectedFrom.values()].filter((n) => n > 1);
  assert.ok(repeats.length >= 1, 'no node was traversed twice; the test proves nothing');

  const rootNode = traversedFrom[0];
  const childRepeats = [...timesSelectedFrom.entries()].filter(
    ([node, n]) => node !== rootNode && n > 1
  );
  assert.ok(
    childRepeats.length >= 1,
    'no already-expanded CHILD was traversed again; only the root repeated'
  );
});
