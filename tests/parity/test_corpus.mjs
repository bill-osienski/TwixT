/**
 * Verification for the Phase 2 parity corpus.
 *
 *   node --test tests/parity/test_corpus.mjs
 *
 * Every recorded fact is RE-DERIVED here from the move sequences using
 * TwixtState directly, rather than trusted from the generator's own output. A
 * corpus that agrees with a buggy generator would otherwise look correct.
 *
 * No model is loaded and no tensor is built: this checks the corpus is legal,
 * deterministic, unique, correctly stratified, balanced, and free of
 * model-derived quantities — nothing about parity itself.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { TwixtState } from '../../server/gameLogic.js';
import {
  INITIAL_SEED,
  STRATA,
  PER_STRATUM,
  buildCorpus,
  playTo,
  positionKey,
  spread,
} from './generate_corpus.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const corpus = JSON.parse(await readFile(join(HERE, 'corpus.json'), 'utf8'));
const all = [...corpus.primary, ...corpus.edge];

describe('determinism', () => {
  it('regenerating reproduces the committed corpus byte for byte', () => {
    const regenerated = `${JSON.stringify(buildCorpus(), null, 2)}\n`;
    const committed = `${JSON.stringify(corpus, null, 2)}\n`;
    assert.strictEqual(
      regenerated,
      committed,
      'generator output drifted from the committed corpus'
    );
  });

  it('the seed allocation is fully accounted for', () => {
    // Every seed between the initial and the next unused one was either
    // accepted by a position or recorded as a rejection. No silent draws.
    const used = new Set(all.map((p) => p.seed).filter((s) => s !== null));
    const rejected = new Set(corpus.rejections.map((r) => r.seed));
    for (let s = INITIAL_SEED; s < corpus.seeds.next_unused; s++) {
      assert.ok(
        used.has(s) || rejected.has(s),
        `seed ${s} was drawn but never accounted for`
      );
    }
  });

  it('spread is total and evenly distributed', () => {
    assert.deepStrictEqual(spread([1, 2, 3], 1), [1]);
    assert.deepStrictEqual(spread([1, 2, 3], 3), [1, 2, 3]);
    // n > length repeats rather than failing
    assert.strictEqual(spread([1, 2, 3], 5).length, 5);
    assert.strictEqual(spread([1, 2, 3], 5)[0], 1);
    assert.strictEqual(spread([1, 2, 3], 5)[4], 3);
  });
});

describe('legality — re-derived, not trusted', () => {
  it('every move was legal at the moment it was played', () => {
    for (const p of all) {
      let state = new TwixtState({});
      p.moves.forEach((move, i) => {
        const legal = state.legalMoves();
        const found = legal.some((m) => m[0] === move[0] && m[1] === move[1]);
        assert.ok(
          found,
          `${p.id}: move ${i} [${move}] was not legal at ply ${i}`
        );
        assert.ok(
          !state.isTerminal(),
          `${p.id}: game was already over at ply ${i}`
        );
        state = state.applyMove(move);
      });
    }
  });

  it('recorded ply, side to move, legal count and terminal status all re-derive', () => {
    for (const p of all) {
      const state = TwixtState.fromMoves(p.moves);
      assert.strictEqual(
        p.ply,
        p.moves.length,
        `${p.id}: ply disagrees with move count`
      );
      assert.strictEqual(state.toMove, p.to_move, `${p.id}: to_move disagrees`);
      assert.strictEqual(
        state.legalMoves().length,
        p.n_legal,
        `${p.id}: n_legal disagrees`
      );
      assert.strictEqual(
        state.isTerminal(),
        p.terminal,
        `${p.id}: terminal disagrees`
      );
    }
  });

  it('no position is terminal and every position has a legal move', () => {
    for (const p of all) {
      assert.strictEqual(p.terminal, false, `${p.id} is terminal`);
      assert.ok(p.n_legal >= 1, `${p.id} has no legal moves`);
    }
  });

  it('side to move follows ply parity, so the colour split is structural', () => {
    for (const p of all) {
      assert.strictEqual(
        p.to_move,
        p.ply % 2 === 0 ? 'red' : 'black',
        `${p.id}: side to move does not follow ply parity`
      );
    }
  });
});

describe('uniqueness', () => {
  it('all 126 positions are distinct under the canonical key', () => {
    const keys = new Map();
    for (const p of all) {
      const key = positionKey(TwixtState.fromMoves(p.moves));
      assert.ok(!keys.has(key), `${p.id} duplicates ${keys.get(key)}`);
      keys.set(key, p.id);
    }
    assert.strictEqual(keys.size, 126);
  });

  it('the key distinguishes bridge layout, not just pegs', () => {
    // Bridge formation is move-order dependent through _crossesExistingBridge,
    // so a peg-only key could collide. Confirm the key includes bridges.
    const a = TwixtState.fromMoves([
      [5, 5],
      [10, 10],
      [7, 6],
    ]);
    assert.ok(
      positionKey(a).includes('#'),
      'key must have peg and bridge sections'
    );
    const sections = positionKey(a).split('#');
    assert.strictEqual(sections.length, 3, 'key is to_move#pegs#bridges');
    assert.ok(
      sections[2].length > 0,
      'bridge section must be populated for a bridged position'
    );
  });

  it('move sequences are themselves distinct', () => {
    const seqs = new Set(all.map((p) => JSON.stringify(p.moves)));
    assert.strictEqual(seqs.size, all.length);
  });
});

describe('strata and balance', () => {
  it('120 primary positions, exactly 30 per stratum', () => {
    assert.strictEqual(corpus.primary.length, 120);
    for (const s of STRATA) {
      const n = corpus.primary.filter((p) => p.stratum === s.name).length;
      assert.strictEqual(n, PER_STRATUM, `stratum ${s.name}`);
    }
  });

  it('every primary position sits inside its stratum and hit its target ply', () => {
    for (const p of corpus.primary) {
      const s = STRATA.find((x) => x.name === p.stratum);
      assert.ok(s, `${p.id}: unknown stratum`);
      assert.ok(
        p.ply >= s.lo && p.ply <= s.hi,
        `${p.id}: ply ${p.ply} outside ${s.name}`
      );
      assert.strictEqual(p.ply, p.target_ply, `${p.id}: missed its target ply`);
    }
  });

  it('each side to move is at least 40% of the primary 120', () => {
    const red = corpus.primary.filter((p) => p.to_move === 'red').length;
    const black = corpus.primary.length - red;
    assert.ok(red / 120 >= 0.4, `red ${red}/120`);
    assert.ok(black / 120 >= 0.4, `black ${black}/120`);
    assert.strictEqual(
      red,
      60,
      'construction should give an exact 50/50 split'
    );
    assert.strictEqual(black, 60);
  });

  it('the aggregate gates are scoped to the primary set only', () => {
    // The specification computes percentage and median gates over the primary
    // 120. Edge positions are deliberately extreme and must not dilute them.
    assert.strictEqual(corpus.primary.length, 120);
    assert.strictEqual(corpus.edge.length, 6);
    for (const e of corpus.edge) assert.strictEqual(e.stratum, 'edge');
    for (const p of corpus.primary) assert.notStrictEqual(p.stratum, 'edge');
  });
});

describe('edge coverage', () => {
  const byId = (id) => corpus.edge.find((e) => e.id === id);

  it('has exactly the six required, separately labelled cases', () => {
    const ids = corpus.edge.map((e) => e.id).sort();
    assert.deepStrictEqual(ids, [
      'edge_black_to_move',
      'edge_empty_board',
      'edge_fewest_legal',
      'edge_one_move',
      'edge_over_512_legal',
      'edge_red_to_move',
    ]);
    for (const e of corpus.edge)
      assert.ok(e.label && e.label.length > 0, `${e.id} lacks a label`);
  });

  it('empty board is ply 0 with the maximum legal-move count', () => {
    const e = byId('edge_empty_board');
    assert.strictEqual(e.ply, 0);
    assert.strictEqual(e.moves.length, 0);
    assert.strictEqual(e.n_legal, Math.max(...all.map((p) => p.n_legal)));
  });

  it('one-move case is ply 1', () => {
    assert.strictEqual(byId('edge_one_move').ply, 1);
  });

  it('fewest-legal case has the minimum legal-move count in the corpus', () => {
    const e = byId('edge_fewest_legal');
    assert.strictEqual(e.n_legal, Math.min(...all.map((p) => p.n_legal)));
    assert.ok(e.ply > 220, 'must be deeper than any primary position');
  });

  it('the >512 case exceeds 512 and is NOT the empty board', () => {
    const e = byId('edge_over_512_legal');
    assert.ok(e.n_legal > 512, `n_legal ${e.n_legal}`);
    assert.ok(e.ply > 0, 'must not duplicate the empty board');
    assert.notStrictEqual(
      JSON.stringify(e.moves),
      JSON.stringify(byId('edge_empty_board').moves)
    );
  });

  it('the two side-to-move cases are what they claim', () => {
    assert.strictEqual(byId('edge_black_to_move').to_move, 'black');
    assert.strictEqual(byId('edge_red_to_move').to_move, 'red');
  });
});

describe('the retry rule actually works', () => {
  // Zero seeds were rejected in the real run, because random play under this
  // PRNG never ends before the deepest target of ply 220. That leaves the
  // rejection path unexercised by the corpus itself, so drive it directly.
  it('detects a target ply the game cannot reach', () => {
    const result = playTo(INITIAL_SEED, 700);
    assert.strictEqual(
      result.endedEarly,
      true,
      'ply 700 is unreachable and must be detected'
    );
    assert.ok(result.moves.length < 700);
  });

  it('does not report ending early when the target is reached', () => {
    const result = playTo(INITIAL_SEED, 50);
    assert.strictEqual(result.endedEarly, false);
    assert.strictEqual(result.moves.length, 50);
  });

  it('records zero rejections, and says why rather than hiding it', () => {
    assert.strictEqual(corpus.summary.rejected_seeds, corpus.rejections.length);
    assert.strictEqual(corpus.rejections.length, 0);
    assert.match(corpus.summary.rejected_seeds_note, /Zero is expected/);
  });
});

describe('no model-derived quantities', () => {
  it('the corpus declares and contains only game-derived facts', () => {
    assert.strictEqual(corpus.contains_model_derived_quantities, false);
    const allowed = new Set([
      'id',
      'label',
      'stratum',
      'target_ply',
      'seed',
      'ply',
      'to_move',
      'n_legal',
      'terminal',
      'moves',
    ]);
    for (const p of all) {
      for (const key of Object.keys(p)) {
        assert.ok(
          allowed.has(key),
          `${p.id} carries unexpected field "${key}"`
        );
      }
    }
  });

  it('no serialized tensor, logit, value or policy appears anywhere', () => {
    const text = JSON.stringify(corpus);
    for (const banned of [
      'logit',
      'policy',
      'value',
      'tensor',
      'prior',
      'checkpoint',
      'onnx',
    ]) {
      assert.ok(!text.includes(banned), `corpus mentions "${banned}"`);
    }
  });
});
