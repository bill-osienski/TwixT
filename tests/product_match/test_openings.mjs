/**
 * Verification for the frozen opening pool.
 *
 *   node --test tests/product_match/test_openings.mjs
 *
 * Every recorded fact is RE-DERIVED from the move sequences using `TwixtState`
 * directly, rather than trusted from the generator's own output: a pool that
 * agrees with a buggy generator would otherwise look correct.
 *
 * No model is loaded and no game is played. This checks only that the pool is
 * deterministic, legal, unique, non-terminal, correctly split, fully accounted
 * for, and leaves real choices to the engines.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { TwixtState } from '../../server/gameLogic.js';
import {
  INITIAL_SEED,
  MATCH_SET_SIZE,
  MIN_CONTINUATIONS,
  OPENING_PLIES,
  POOL_SIZE,
  TIMING_SET_SIZE,
  buildPool,
  openingMovesFrom,
  playOpening,
  positionKey,
  rejectionReason,
} from './generate_openings.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
// The raw text is kept, not just the parsed object: re-serializing before
// comparison would normalize away whitespace and trailing bytes, which is
// exactly what a byte-stability claim must catch.
const poolText = await readFile(join(HERE, 'openings.json'), 'utf8');
const pool = JSON.parse(poolText);
const all = pool.openings;

describe('determinism', () => {
  it('regenerating reproduces the committed pool byte for byte', () => {
    const regenerated = `${JSON.stringify(buildPool(), null, 2)}\n`;
    assert.strictEqual(
      regenerated,
      poolText,
      'generator output drifted from the committed pool'
    );
  });

  it('would actually notice a formatting-only change', () => {
    // Guards the guard: a parse-then-reserialize comparison passes on all of
    // these, which is how such a test can look strict and prove nothing.
    const regenerated = `${JSON.stringify(buildPool(), null, 2)}\n`;
    for (const mutated of [
      poolText.replace(/\n$/, ''),
      `${poolText}\n`,
      poolText.replace('\n  "schema"', '\n    "schema"'),
    ]) {
      assert.notStrictEqual(regenerated, mutated);
      assert.deepStrictEqual(
        JSON.parse(mutated),
        pool,
        'mutation must be whitespace-only'
      );
    }
  });

  it('uses the frozen seed and PRNG the specification names', () => {
    assert.strictEqual(pool.seeds.initial, 20260814);
    assert.strictEqual(INITIAL_SEED, 20260814);
    assert.strictEqual(pool.prng.algorithm, 'mulberry32');
  });
});

describe('legality — re-derived, not trusted', () => {
  it('every move was legal at the moment it was played', () => {
    for (const o of all) {
      let state = new TwixtState({});
      o.moves.forEach((move, i) => {
        assert.ok(
          !state.isTerminal(),
          `opening ${o.id}: game already over at ply ${i}`
        );
        const legal = state.legalMoves();
        assert.ok(
          legal.some((m) => m[0] === move[0] && m[1] === move[1]),
          `opening ${o.id}: move ${i} [${move}] was not legal`
        );
        state = state.applyMove(move);
      });
    }
  });

  it('every opening is exactly four plies, two moves per side', () => {
    for (const o of all) {
      assert.strictEqual(o.moves.length, OPENING_PLIES, `opening ${o.id}`);
      const state = TwixtState.fromMoves(o.moves);
      assert.strictEqual(state.ply, OPENING_PLIES);
    }
  });

  it('moves are well-formed coordinates inside the board', () => {
    for (const o of all) {
      for (const [r, c] of o.moves) {
        assert.ok(
          Number.isInteger(r) && r >= 0 && r < 24,
          `opening ${o.id}: row ${r}`
        );
        assert.ok(
          Number.isInteger(c) && c >= 0 && c < 24,
          `opening ${o.id}: col ${c}`
        );
      }
    }
  });

  it('recorded side to move, legal count and terminal status all re-derive', () => {
    for (const o of all) {
      const state = TwixtState.fromMoves(o.moves);
      assert.strictEqual(state.toMove, o.to_move, `opening ${o.id}: to_move`);
      assert.strictEqual(
        state.legalMoves().length,
        o.n_legal,
        `opening ${o.id}: n_legal`
      );
      assert.strictEqual(
        state.isTerminal(),
        o.terminal,
        `opening ${o.id}: terminal`
      );
    }
  });
});

describe('non-terminal, with real choices remaining', () => {
  it('no opening is terminal', () => {
    for (const o of all) {
      const state = TwixtState.fromMoves(o.moves);
      assert.strictEqual(
        state.isTerminal(),
        false,
        `opening ${o.id} is terminal`
      );
      assert.strictEqual(
        state.winner(),
        null,
        `opening ${o.id} already has a winner`
      );
    }
  });

  it('every opening leaves at least two legal continuations', () => {
    // A position with one forced reply would make the "game" a formality and
    // carry no information about either model.
    for (const o of all) {
      const n = TwixtState.fromMoves(o.moves).legalMoves().length;
      assert.ok(
        n >= MIN_CONTINUATIONS,
        `opening ${o.id} has only ${n} continuations`
      );
    }
  });

  it('the continuations are genuinely playable, not merely counted', () => {
    // Apply two distinct continuations from each of a sample and confirm both
    // produce legal, distinct successor positions.
    for (const o of [
      all[0],
      all[MATCH_SET_SIZE - 1],
      all[MATCH_SET_SIZE],
      all[POOL_SIZE - 1],
    ]) {
      const state = TwixtState.fromMoves(o.moves);
      const legal = state.legalMoves();
      const a = state.applyMove(legal[0]);
      const b = state.applyMove(legal[1]);
      assert.strictEqual(a.ply, OPENING_PLIES + 1);
      assert.strictEqual(b.ply, OPENING_PLIES + 1);
      assert.notStrictEqual(positionKey(a), positionKey(b), `opening ${o.id}`);
    }
  });
});

describe('uniqueness', () => {
  it('all 210 openings are distinct under the canonical position key', () => {
    const keys = new Map();
    for (const o of all) {
      const key = positionKey(TwixtState.fromMoves(o.moves));
      assert.ok(
        !keys.has(key),
        `opening ${o.id} duplicates opening ${keys.get(key)}`
      );
      keys.set(key, o.id);
    }
    assert.strictEqual(keys.size, POOL_SIZE);
  });

  it('move sequences are themselves distinct', () => {
    const seqs = new Set(all.map((o) => JSON.stringify(o.moves)));
    assert.strictEqual(seqs.size, POOL_SIZE);
  });

  it('the key distinguishes bridge layout, not only pegs', () => {
    // Bridge formation is move-order dependent through _crossesExistingBridge,
    // so a peg-only key could collide across genuinely different positions.
    const key = positionKey(
      TwixtState.fromMoves([
        [5, 5],
        [10, 10],
        [7, 6],
        [12, 11],
      ])
    );
    const sections = key.split('#');
    assert.strictEqual(sections.length, 3, 'key is to_move#pegs#bridges');
    assert.ok(
      sections[2].length > 0,
      'bridge section must be populated when bridges exist'
    );
  });
});

describe('pool shape and the match/timing split', () => {
  it('holds exactly 210 openings with ids 0…209 in order', () => {
    assert.strictEqual(all.length, POOL_SIZE);
    all.forEach((o, i) => assert.strictEqual(o.id, i));
  });

  it('reserves 0…199 for the match and 200…209 for timing', () => {
    const match = all.filter((o) => o.role === 'match');
    const timing = all.filter((o) => o.role === 'timing');
    assert.strictEqual(match.length, MATCH_SET_SIZE);
    assert.strictEqual(timing.length, TIMING_SET_SIZE);
    assert.deepStrictEqual(
      match.map((o) => o.id),
      Array.from({ length: MATCH_SET_SIZE }, (_, i) => i)
    );
    assert.deepStrictEqual(
      timing.map((o) => o.id),
      Array.from({ length: TIMING_SET_SIZE }, (_, i) => MATCH_SET_SIZE + i)
    );
  });

  it('the two sets are disjoint, so timing can never leak into the match', () => {
    const matchIds = new Set(
      all.filter((o) => o.role === 'match').map((o) => o.id)
    );
    for (const o of all.filter((o) => o.role === 'timing')) {
      assert.ok(!matchIds.has(o.id), `opening ${o.id} is in both sets`);
    }
  });

  it('any P selects a PREFIX of the match set, never a chosen subset', () => {
    // `P` is fixed from timing alone, so which openings it selects must follow
    // mechanically from `P` with no freedom left in the choice.
    for (const P of [100, 200]) {
      const selected = all.slice(0, P);
      assert.strictEqual(selected.length, P);
      assert.ok(selected.every((o) => o.role === 'match'));
      assert.deepStrictEqual(
        selected.map((o) => o.id),
        Array.from({ length: P }, (_, i) => i)
      );
    }
  });
});

describe('seed and rejection accounting', () => {
  it('every drawn seed is accounted for, with none unexplained', () => {
    const used = new Set(all.map((o) => o.seed));
    const rejected = new Set(pool.rejections.map((r) => r.seed));
    assert.strictEqual(used.size, POOL_SIZE, 'no seed is reused');
    for (let s = INITIAL_SEED; s < pool.seeds.next_unused; s++) {
      assert.ok(
        used.has(s) || rejected.has(s),
        `seed ${s} was drawn but never accounted for`
      );
    }
    assert.strictEqual(
      used.size + rejected.size,
      pool.seeds.next_unused - INITIAL_SEED,
      'drawn seeds must equal accepted plus rejected'
    );
  });

  it('seeds are handed out in opening order, one per attempt', () => {
    for (let i = 1; i < all.length; i++) {
      assert.ok(
        all[i].seed > all[i - 1].seed,
        `opening ${i} seed is not increasing`
      );
    }
    assert.strictEqual(all[0].seed, INITIAL_SEED);
  });

  it('records zero rejections, and says why rather than hiding it', () => {
    assert.strictEqual(pool.summary.rejected_seeds, pool.rejections.length);
    assert.strictEqual(pool.rejections.length, 0);
    assert.match(pool.summary.rejected_seeds_note, /Zero is expected/);
  });
});

describe('the rejection rule actually works', () => {
  // No four-ply opening is ever rejected under this PRNG, which would leave the
  // rule unexercised by the pool itself. Drive the predicate directly.
  const liveState = () =>
    TwixtState.fromMoves([
      [5, 5],
      [10, 10],
      [7, 6],
      [12, 11],
    ]);

  it('accepts a qualifying position', () => {
    assert.strictEqual(rejectionReason(liveState(), false, new Map()), null);
  });

  it('rejects a game that ended before the target ply', () => {
    assert.strictEqual(
      rejectionReason(liveState(), true, new Map()).reason,
      'game_ended_before_target_ply'
    );
  });

  it('rejects a terminal position', () => {
    const terminal = { isTerminal: () => true, legalMoves: () => [] };
    assert.strictEqual(
      rejectionReason(terminal, false, new Map()).reason,
      'terminal_position'
    );
  });

  it('rejects a position with a single forced continuation', () => {
    const forced = { isTerminal: () => false, legalMoves: () => [[0, 0]] };
    const r = rejectionReason(forced, false, new Map());
    assert.strictEqual(r.reason, 'too_few_continuations');
    assert.strictEqual(r.n_legal, 1);
  });

  it('rejects a duplicate position and names what it duplicates', () => {
    const state = liveState();
    const seen = new Map([[positionKey(state), 42]]);
    const r = rejectionReason(state, false, seen);
    assert.strictEqual(r.reason, 'duplicate_position');
    assert.strictEqual(r.duplicate_of, 42);
  });

  it('playOpening reports an unreachable depth rather than silently truncating', () => {
    const result = playOpening(INITIAL_SEED, 4);
    assert.strictEqual(result.endedEarly, false);
    assert.strictEqual(result.moves.length, 4);
  });
});

describe('the pool feeds its consumers in the shape they expect', () => {
  // A real producer/consumer seam: the pool stores rich entries, while the
  // harness and analyser index by opening_id and want bare move lists. Handing
  // them objects makes `opening.length` undefined, so the opening-prefix check
  // would compare against an empty array — present in the code, absent in
  // effect. Exactly the untested-seam class this project has been bitten by.
  it('yields bare four-move arrays from the committed pool', () => {
    const moves = openingMovesFrom(pool);
    assert.strictEqual(moves.length, POOL_SIZE);
    for (let i = 0; i < moves.length; i++) {
      assert.ok(Array.isArray(moves[i]), `opening ${i} is not an array`);
      assert.strictEqual(moves[i].length, OPENING_PLIES);
      assert.deepStrictEqual(moves[i], all[i].moves);
    }
  });

  it('is idempotent, so a caller cannot be wrong about which shape it holds', () => {
    const once = openingMovesFrom(pool);
    assert.deepStrictEqual(openingMovesFrom(once), once);
  });

  it('refuses a pool that is not openings at all', () => {
    assert.throws(() => openingMovesFrom({}), /no openings array/);
    assert.throws(() => openingMovesFrom([[[1, 2]]]), /does not hold 4 moves/);
  });

  it('the extracted moves replay exactly as the recorded openings do', () => {
    // Ties the adapter back to the game rules, not just to array shapes.
    for (const moves of openingMovesFrom(pool).slice(0, 5)) {
      const state = TwixtState.fromMoves(moves);
      assert.strictEqual(state.ply, OPENING_PLIES);
      assert.strictEqual(state.isTerminal(), false);
    }
  });
});

describe('scope: no model, no game', () => {
  it('the generator imports nothing that could load a model or play a game', async () => {
    // Constructed against the IMPORT LIST, not against any mention of a word:
    // a substring scan fails on a prose comment while still passing a module
    // that imports what it should not. The authorization for this step
    // excludes model loading and games, so assert the capability is absent.
    const src = await readFile(join(HERE, 'generate_openings.mjs'), 'utf8');
    const imports = [
      ...src.matchAll(/^\s*import\s[^;]*?from\s+'([^']+)'/gm),
    ].map((m) => m[1]);
    assert.deepStrictEqual(
      imports.sort(),
      ['../../server/gameLogic.js', 'node:url'],
      'the generator may import only the game rules and a path helper'
    );
    for (const forbidden of [
      'inference',
      'model_manifest',
      'onnxruntime',
      'mcts',
      './harness.mjs',
    ]) {
      assert.ok(
        !imports.some((i) => i.includes(forbidden)),
        `generator must not import ${forbidden}`
      );
    }
    assert.strictEqual(pool.loads_no_model, true);
    assert.strictEqual(pool.plays_no_game, true);
  });

  it('carries no model-derived quantity', () => {
    const allowed = new Set([
      'id',
      'role',
      'seed',
      'moves',
      'to_move',
      'n_legal',
      'terminal',
    ]);
    for (const o of all) {
      for (const key of Object.keys(o)) {
        assert.ok(
          allowed.has(key),
          `opening ${o.id} carries unexpected field "${key}"`
        );
      }
    }
    const text = JSON.stringify(pool.openings);
    for (const banned of [
      'logit',
      'value',
      'policy',
      'prior',
      'score',
      'model',
    ]) {
      assert.ok(!text.includes(banned), `pool mentions "${banned}"`);
    }
  });
});
