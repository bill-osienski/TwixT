/**
 * V5 and V6 for the protocol move (server/protocol_move.js) and its wiring in
 * server/index.js: values and valueKind, refusals, aborts, legal moves and the
 * game-end boundary. Fake evaluators only; no model.
 *
 * server/index.js starts a server on import, so its transport wiring is checked
 * here at the source level; the live transports are exercised by the startup
 * and rollback tests (R1, R2), which load models and belong to G2.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { BOARD_SIZE, MAX_PLIES, TwixtState } from './gameLogic.js';
import { MirrorAveragingInference } from './mirror_inference.js';
import { PyRandom } from './py_random.js';
import { ABORTED_MESSAGE, ProtocolRefusal, displayValue, protocolMove } from './protocol_move.js';
import { PROTOCOL_HARD, resolvePolicy } from './readout_policy.js';
import { fakeInference, loadFixture } from '../tests/fixtures/htb_integration/fake_evaluator.mjs';

const FIX = loadFixture('search.json');
const POLICY = resolvePolicy({ difficulty: 'hard', hardMode: 'protocol' });
const caseById = (id) => FIX.cases.find((c) => c.id === id);
const mirrorFake = () => new MirrorAveragingInference(fakeInference());
const legalKeys = (s) => new Set(s.legalMoves().map((m) => `${m[0]},${m[1]}`));

/** Random legal play with a PyRandom stream, avoiding wins, to `plies`. Null if it cannot get there. */
function randomPosition(plies, seed) {
  const rng = new PyRandom(seed);
  let s = new TwixtState({});
  while (s.ply < plies) {
    const legal = s.legalMoves();
    if (legal.length === 0) return null;
    rng.shuffle(legal);
    let next = null;
    for (const m of legal) {
      const t = s.applyMove(m);
      if (t.winner() === null) {
        next = t;
        break;
      }
    }
    if (!next) return null;
    s = next;
  }
  return s;
}

test('the resolved protocol policy is the frozen table entry', () => {
  assert.deepEqual(POLICY, { difficulty: 'hard', nSims: 200, moveTemp: 1.0, search: 'protocol' });
  assert.equal(POLICY.nSims, PROTOCOL_HARD.nSims);
});

test('ply < 20: the protocol search and readout, root_mean; values truthful', async () => {
  const c = caseById('plain_opening_25');
  const out = await protocolMove({
    gameState: TwixtState.fromMoves(c.moves),
    policy: POLICY,
    inference: fakeInference(),
    seeds: { search: c.search_seed, readout: c.readout_seed },
  });
  assert.equal(out.moveKey, c.result.readout_move);
  assert.equal(out.rootValue, c.result.root_value);
  assert.equal(out.valueKind, 'root_mean');
  assert.equal(displayValue(out), out.rootValue);
  assert.equal(out.moveVisits, c.result.counts[out.moveKey] ?? 0);
  assert.equal(out.rounds, null);
});

test('ply >= 20: the halving choice is played, played_move_mean from its actual visits', async () => {
  for (const id of ['halving_midgame_10', 'halving_m9', 'halving_m1']) {
    const c = caseById(id);
    const out = await protocolMove({
      gameState: TwixtState.fromMoves(c.moves),
      policy: POLICY,
      inference: fakeInference(),
      seeds: { search: c.search_seed, readout: 1 },
    });
    assert.equal(out.moveKey, c.result.choice, id);
    assert.equal(out.valueKind, 'played_move_mean', id);
    assert.equal(out.moveVisits, c.result.choice_visits, id);
    assert.equal(out.moveValue, c.result.choice_value, id);
    // rootValue stays the actual root mean; the display value is the move's.
    assert.equal(out.rootValue, c.result.root_value, id);
    assert.equal(displayValue(out), out.moveValue, id);
  }
});

test('the readout stream is separate: a different readout seed can change only the opening move', async () => {
  const c = caseById('plain_empty_0');
  const moves = new Set();
  for (let r = 0; r < 8; r++) {
    const out = await protocolMove({
      gameState: new TwixtState({}),
      policy: POLICY,
      inference: fakeInference(),
      seeds: { search: c.search_seed, readout: r },
    });
    assert.equal(out.rootValue, c.result.root_value);
    moves.add(out.moveKey);
  }
  assert.ok(moves.size > 1, 'T 1.0 sampling must vary with the readout stream');
});

test('without seeds each request draws fresh streams (no request can choose one)', async () => {
  const s = randomPosition(4, 3);
  const keys = new Set();
  for (let i = 0; i < 6; i++) {
    const out = await protocolMove({ gameState: s, policy: POLICY, inference: fakeInference() });
    keys.add(JSON.stringify([...out.visitCounts]));
  }
  assert.ok(keys.size > 1, 'fresh search streams should vary the search');
});

test('refusals: a board other than 24x24, a won root and a root with no legal move', async () => {
  const small = new TwixtState({ activeSize: 12 });
  await assert.rejects(
    protocolMove({ gameState: small, policy: POLICY, inference: fakeInference() }),
    (e) => e instanceof ProtocolRefusal && e.code === 'NOT_24'
  );
  const chain = caseById('halving_winning_child');
  const won = TwixtState.fromMoves(chain.moves).applyMove(FIX.chain_wins[0].split(',').map(Number));
  assert.equal(won.winner(), 'red');
  await assert.rejects(
    protocolMove({ gameState: won, policy: POLICY, inference: fakeInference() }),
    (e) => e instanceof ProtocolRefusal && e.code === 'TERMINAL_ROOT'
  );
  const pegs = new Map();
  for (let r = 0; r < BOARD_SIZE; r++) for (let c = 0; c < BOARD_SIZE; c++) pegs.set(`${r},${c}`, (r + c) % 2 ? 'red' : 'black');
  const full = new TwixtState({ pegs, ply: pegs.size });
  assert.equal(full.legalMoves().length, 0);
  await assert.rejects(
    protocolMove({ gameState: full, policy: POLICY, inference: fakeInference() }),
    (e) => e instanceof ProtocolRefusal && e.code === 'TERMINAL_ROOT'
  );
});

test('an aborted search plays nothing, before or after ply 20', async () => {
  for (const plies of [5, 30]) {
    const ctrl = new AbortController();
    let n = 0;
    const inference = {
      async evaluate(b, moves) {
        n += 1;
        if (n === 40) ctrl.abort();
        return fakeInference().evaluate(b, moves);
      },
    };
    await assert.rejects(
      protocolMove({ gameState: randomPosition(plies, 9), policy: POLICY, inference, signal: ctrl.signal }),
      (e) => e.message === ABORTED_MESSAGE
    );
  }
});

test('protocolMove refuses a plain policy or a changed opening temperature', async () => {
  const s = new TwixtState({});
  await assert.rejects(
    protocolMove({ gameState: s, policy: resolvePolicy({ difficulty: 'medium' }), inference: fakeInference() }),
    /plain policy/
  );
  await assert.rejects(
    protocolMove({ gameState: s, policy: { ...POLICY, moveTemp: 0.5 }, inference: fakeInference() }),
    /T 1.0/
  );
});

// --- V6: legal moves and the game-end boundary -------------------------------

test('V6: every played move is legal and the candidates are legal, from the opening to a near-full board', async () => {
  let checked = 0;
  for (const [plies, seed] of [[0, 1], [7, 2], [19, 3], [20, 4], [45, 5], [120, 6], [280, 7], [400, 8], [520, 9]]) {
    const s = randomPosition(plies, seed);
    if (!s) continue;
    const out = await protocolMove({ gameState: s, policy: POLICY, inference: mirrorFake(), seeds: { search: seed, readout: seed } });
    const legal = legalKeys(s);
    assert.ok(legal.has(out.moveKey), `ply ${plies}: ${out.moveKey} is not legal`);
    if (out.rounds) {
      assert.equal(out.rounds[0].length, Math.min(16, legal.size));
      for (const k of out.rounds[0]) assert.ok(legal.has(k), `ply ${plies}: candidate ${k}`);
    }
    checked += 1;
  }
  assert.ok(checked >= 8, `only ${checked} positions reached`);
});

test('V6: the ply-20 switch, and plies 279 / 280 / 281 search normally under the product rule', async () => {
  const base = randomPosition(281, 17);
  assert.ok(base, 'random play did not reach ply 281');
  const at = (ply) => randomPosition(ply, 17);
  const kinds = {};
  for (const ply of [19, 20, 279, 280, 281]) {
    const s = at(ply);
    assert.equal(s.ply, ply);
    assert.equal(s.isTerminal(), false, `ply ${ply} is not terminal under the product rule`);
    const out = await protocolMove({ gameState: s, policy: POLICY, inference: mirrorFake(), seeds: { search: ply, readout: ply } });
    assert.ok(legalKeys(s).has(out.moveKey));
    kinds[ply] = out.valueKind;
  }
  assert.deepEqual(kinds, {
    19: 'root_mean',
    20: 'played_move_mean',
    279: 'played_move_mean',
    280: 'played_move_mean',
    281: 'played_move_mean',
  });
});

test('V6: ply 600 is unreachable on 24x24, and the move-number feature stays below 1', () => {
  assert.ok(BOARD_SIZE * BOARD_SIZE < MAX_PLIES, 'each ply fills one of 576 cells');
  const deep = caseById('halving_m1');
  const s = TwixtState.fromMoves(deep.moves);
  const hwc = s.toTensorHWC();
  assert.ok(hwc[0][0][23] < 1, `channel 23 is ${hwc[0][0][23]} at ply ${s.ply}`);
  assert.equal(hwc[0][0][23], Math.fround(s.ply / MAX_PLIES));
});

// --- the transports' wiring in server/index.js (source level; see header) ----

const INDEX = readFileSync(new URL('./index.js', import.meta.url), 'utf8');

/** The text of the block that opens at `opener`, up to its matching brace. */
function blockAt(src, opener, from = 0) {
  const start = src.indexOf(opener, from);
  assert.ok(start >= 0, `missing: ${opener}`);
  let depth = 0;
  for (let i = src.indexOf('{', start); i < src.length; i++) {
    if (src[i] === '{') depth += 1;
    if (src[i] === '}') depth -= 1;
    if (depth === 0) return { text: src.slice(start, i + 1), start };
  }
  throw new Error('unbalanced');
}

test('index.js: both transports route protocol policies to protocolMove, neither passes seeds', () => {
  const calls = INDEX.match(/protocolMove\(\{[^}]*\}\)/g) || [];
  assert.equal(calls.length, 2, `expected 2 protocolMove calls, found ${calls.length}`);
  for (const call of calls) {
    assert.ok(!/seeds/.test(call), `a transport passes seeds: ${call}`);
    assert.ok(/inference: hardInference/.test(call), `a transport passes another model: ${call}`);
  }
  assert.equal((INDEX.match(/if \(policy\.search === 'protocol'\)/g) || []).length, 2);
});

test('index.js REST: the protocol branch never touches the cache and returns before the lookup', () => {
  const rest = blockAt(INDEX, "app.post('/api/move'");
  const branch = blockAt(rest.text, "if (policy.search === 'protocol')");
  assert.ok(!/cache\./.test(branch.text), 'the protocol branch must not read or write the cache');
  assert.ok(/cached: false/.test(branch.text));
  assert.ok(/return res\.json\(/.test(branch.text));
  assert.ok(branch.start < rest.text.indexOf('cache.get('), 'the branch must come before the cache lookup');
});

test('index.js: conflicts and refusals answer 400 on REST, and the plain response is unchanged', () => {
  assert.ok(/err instanceof PolicyConflict \|\| err instanceof ProtocolRefusal/.test(INDEX));
  assert.ok(/res\.status\(400\)\.json\(\{ error: err\.message, code: err\.code \}\)/.test(INDEX));
  // Protocol-only fields are added to the WS response only for protocol moves.
  const ws = blockAt(INDEX, "if (result.search === 'protocol')");
  for (const f of ['search', 'valueKind', 'moveValue', 'moveVisits', 'modelId']) {
    assert.ok(new RegExp(`response\\.${f} = result\\.${f}`).test(ws.text), f);
  }
});

test('index.js WS: valueRed comes from the displayed value, rootValue stays the root mean', () => {
  const branch = blockAt(INDEX, "if (policy.search === 'protocol')", INDEX.indexOf('async function computeBestMove'));
  assert.ok(/rootValue: out\.rootValue/.test(branch.text));
  assert.ok(/valueRed: toValueRed\(gameState, displayValue\(out\)\)/.test(branch.text));
});

test('index.js startup: HARD_MODE is parsed first, and legacy never resolves the hard model', () => {
  const main = blockAt(INDEX, 'async function main()');
  const parse = main.text.indexOf('hardMode = parseHardMode()');
  assert.ok(parse >= 0 && parse < main.text.indexOf('resolveModel()'));
  assert.ok(/hardMode === 'protocol' \? await resolveHardModel\(\) : null/.test(main.text));
  assert.equal((INDEX.match(/resolveHardModel\(/g) || []).length, 1);
});
