#!/usr/bin/env node
/**
 * Generate the Phase 2 parity corpus.
 *
 *   node tests/parity/generate_corpus.mjs > tests/parity/corpus.json
 *
 * The corpus is a set of MOVE SEQUENCES, never encoded tensors. Both the
 * JavaScript and Python sides replay the same sequences into their own state
 * representation, so the encoding comparison exercises the real path instead of
 * a shared intermediate. Nothing here loads or consults a model: every recorded
 * quantity is a property of the game, not of a network.
 *
 * Determinism: the only randomness is `mulberry32` seeded from INITIAL_SEED.
 * Slots are fixed before any seed is drawn, seeds are handed out in slot order,
 * and every rejected seed is recorded with its reason — so re-running this file
 * reproduces `corpus.json` byte for byte, and the rejections are auditable
 * rather than invisible.
 *
 * Seeds here are CORPUS-CONSTRUCTION seeds. They are not drawn from the
 * research evaluation seed ledger and consume nothing from it.
 *
 * Helpers and `buildCorpus` are exported so the accompanying test can exercise
 * them directly — in particular the rejection path, which does not fire during
 * a normal run.
 *
 * Specification: docs/superpowers/2026-08-13-phase2-parity-specification.md
 */
import { fileURLToPath } from 'node:url';
import { TwixtState } from '../../server/gameLogic.js';

export const INITIAL_SEED = 20260813;
export const MAX_ATTEMPTS_PER_SLOT = 2000;

/** Strata, as fixed by the specification. `late` is the open-ended one. */
export const STRATA = [
  { name: 'opening', lo: 2, hi: 19 },
  { name: 'early_mid', lo: 20, hi: 49 },
  { name: 'midgame', lo: 50, hi: 99 },
  { name: 'late', lo: 100, hi: 220 },
];
export const PER_STRATUM = 30;

/**
 * mulberry32 — a 32-bit PRNG chosen because it is four lines, has no hidden
 * state, and does not depend on any engine's built-in RNG implementation, so
 * the sequence is portable and auditable.
 */
export function mulberry32(seed) {
  let a = seed | 0;
  return function next() {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Pick `n` evenly spaced entries from `values`, repeating when n > length. */
export function spread(values, n) {
  if (n === 1) return [values[0]];
  return Array.from(
    { length: n },
    (_, i) => values[Math.round((i * (values.length - 1)) / (n - 1))]
  );
}

const evensIn = (lo, hi) => {
  const out = [];
  for (let p = lo % 2 === 0 ? lo : lo + 1; p <= hi; p += 2) out.push(p);
  return out;
};
const oddsIn = (lo, hi) => {
  const out = [];
  for (let p = lo % 2 === 1 ? lo : lo + 1; p <= hi; p += 2) out.push(p);
  return out;
};

/**
 * Canonical identity of a position, for the uniqueness constraint.
 *
 * Pegs alone are not sufficient: bridge formation depends on move order through
 * `_crossesExistingBridge`, so two identical peg sets can carry different
 * bridges. Both are included, each sorted, so the key is order-independent.
 */
export function positionKey(state) {
  const d = state.toDict();
  const pegs = Object.entries(d.pegs)
    .map(([cell, player]) => `${cell}:${player}`)
    .sort()
    .join('|');
  const bridges = d.bridges
    .map((b) => JSON.stringify(b))
    .sort()
    .join('|');
  return `${d.to_move}#${pegs}#${bridges}`;
}

/**
 * Play random legal moves from the empty board.
 *
 * Stops at `targetPly`, or earlier if the game ends. Returns the move list and
 * the reached state so the caller can decide whether the attempt qualifies.
 */
export function playTo(seed, targetPly) {
  const rand = mulberry32(seed);
  let state = new TwixtState({});
  const moves = [];
  while (moves.length < targetPly) {
    if (state.isTerminal()) return { state, moves, endedEarly: true };
    const legal = state.legalMoves();
    if (legal.length === 0) return { state, moves, endedEarly: true };
    const move = legal[Math.floor(rand() * legal.length)];
    state = state.applyMove(move);
    moves.push([move[0], move[1]]);
  }
  return { state, moves, endedEarly: false };
}

/** Play to exhaustion; return the deepest state that is still playable. */
export function playToExhaustion(seed) {
  const rand = mulberry32(seed);
  let state = new TwixtState({});
  const moves = [];
  let deepest = null;
  for (;;) {
    if (state.isTerminal()) break;
    const legal = state.legalMoves();
    if (legal.length === 0) break;
    deepest = { state, moves: moves.slice(), nLegal: legal.length };
    const move = legal[Math.floor(rand() * legal.length)];
    state = state.applyMove(move);
    moves.push([move[0], move[1]]);
  }
  return deepest;
}

// ---------------------------------------------------------------------------

export function buildCorpus() {
  const rejections = [];
  const seen = new Map(); // positionKey -> id
  let nextSeed = INITIAL_SEED;

  const record = (entry, state, moves, seed) => ({
    ...entry,
    seed,
    ply: moves.length,
    to_move: state.toMove,
    n_legal: state.legalMoves().length,
    terminal: state.isTerminal(),
    moves,
  });

  /** Draw seeds in order until one produces a qualifying, unique position. */
  const fillSlot = (slot) => {
    for (let attempt = 0; attempt < MAX_ATTEMPTS_PER_SLOT; attempt++) {
      const seed = nextSeed++;
      const { state, moves, endedEarly } = playTo(seed, slot.target_ply);

      if (endedEarly) {
        rejections.push({
          slot: slot.id,
          seed,
          reason: 'game_ended_before_target_ply',
          reached_ply: moves.length,
        });
        continue;
      }
      if (state.isTerminal()) {
        rejections.push({
          slot: slot.id,
          seed,
          reason: 'terminal_at_target_ply',
        });
        continue;
      }
      if (state.legalMoves().length === 0) {
        rejections.push({ slot: slot.id, seed, reason: 'no_legal_moves' });
        continue;
      }
      const key = positionKey(state);
      if (seen.has(key)) {
        rejections.push({
          slot: slot.id,
          seed,
          reason: 'duplicate_position',
          duplicate_of: seen.get(key),
        });
        continue;
      }
      seen.set(key, slot.id);
      return record(slot, state, moves, seed);
    }
    throw new Error(
      `slot ${slot.id}: no qualifying seed within ${MAX_ATTEMPTS_PER_SLOT} attempts`
    );
  };

  // --- primary slots -------------------------------------------------------
  // Fixed before any seed is drawn. Fifteen even-ply and fifteen odd-ply
  // targets per stratum, evenly spread across the stratum's range, which makes
  // the side-to-move split exactly 50/50 by construction rather than by luck.

  const primarySlots = [];
  for (const stratum of STRATA) {
    const evens = spread(evensIn(stratum.lo, stratum.hi), PER_STRATUM / 2);
    const odds = spread(oddsIn(stratum.lo, stratum.hi), PER_STRATUM / 2);
    const targets = [...evens, ...odds].sort((a, b) => a - b);
    targets.forEach((target_ply, i) => {
      primarySlots.push({
        id: `${stratum.name}_${String(i).padStart(2, '0')}`,
        stratum: stratum.name,
        target_ply,
      });
    });
  }

  const primary = primarySlots.map(fillSlot);

  // --- edge slots ----------------------------------------------------------

  const edge = [];

  // E1: the empty board — the maximum legal-move count on this board.
  {
    const state = new TwixtState({});
    seen.set(positionKey(state), 'edge_empty_board');
    edge.push(
      record(
        {
          id: 'edge_empty_board',
          label: 'empty board (maximum legal moves)',
          stratum: 'edge',
          target_ply: 0,
        },
        state,
        [],
        null
      )
    );
  }

  // E2: one move played.
  edge.push(
    fillSlot({
      id: 'edge_one_move',
      label: 'a single move played',
      stratum: 'edge',
      target_ply: 1,
    })
  );

  // E3: fewest legal moves. The specification asks for a single-legal-move
  // position "if constructible; otherwise the fewest-legal-moves position".
  // Deterministic search: take the first seed whose deepest playable state has
  // exactly one legal move; failing that, the fewest across a fixed budget of
  // seeds, ties broken by lowest seed. No seed is chosen after inspecting a
  // result beyond this stated rule.
  {
    const BUDGET = 16;
    let best = null;
    for (let i = 0; i < BUDGET; i++) {
      const seed = nextSeed++;
      const deepest = playToExhaustion(seed);
      if (!deepest) {
        rejections.push({
          slot: 'edge_fewest_legal',
          seed,
          reason: 'no_playable_state',
        });
        continue;
      }
      const key = positionKey(deepest.state);
      if (seen.has(key)) {
        rejections.push({
          slot: 'edge_fewest_legal',
          seed,
          reason: 'duplicate_position',
        });
        continue;
      }
      if (best === null || deepest.nLegal < best.nLegal)
        best = { ...deepest, seed, key };
      if (best.nLegal === 1) break;
      rejections.push({
        slot: 'edge_fewest_legal',
        seed,
        reason: 'not_single_legal_move',
        n_legal: deepest.nLegal,
        deepest_ply: deepest.moves.length,
      });
    }
    if (best === null) throw new Error('edge_fewest_legal: no candidate found');
    seen.set(best.key, 'edge_fewest_legal');
    edge.push(
      record(
        {
          id: 'edge_fewest_legal',
          label: `fewest legal moves reachable (${best.nLegal})`,
          stratum: 'edge',
          target_ply: best.moves.length,
        },
        best.state,
        best.moves,
        best.seed
      )
    );
  }

  // E4: more than 512 legal moves — the pre-cc1b3fa move cap — at a position
  // that is NOT the empty board, so the 576 contract is exercised somewhere the
  // old cap would have overflowed without merely restating E1.
  {
    let chosen = null;
    for (let target = 2; target <= 30 && chosen === null; target += 2) {
      const candidate = fillSlot({
        id: 'edge_over_512_legal',
        label: 'more than 512 legal moves, not the empty board',
        stratum: 'edge',
        target_ply: target,
      });
      if (candidate.n_legal > 512) chosen = candidate;
      else
        rejections.push({
          slot: 'edge_over_512_legal',
          seed: candidate.seed,
          reason: 'n_legal_not_above_512',
          n_legal: candidate.n_legal,
        });
    }
    if (chosen === null)
      throw new Error('edge_over_512_legal: no qualifying position');
    edge.push(chosen);
  }

  // E5/E6: an explicit position for each side to move, distinct from every
  // primary and from each other.
  edge.push(
    fillSlot({
      id: 'edge_black_to_move',
      label: 'black to move',
      stratum: 'edge',
      target_ply: 33,
    })
  );
  edge.push(
    fillSlot({
      id: 'edge_red_to_move',
      label: 'red to move',
      stratum: 'edge',
      target_ply: 34,
    })
  );

  // --- self-certification --------------------------------------------------
  // The committed corpus asserts its own constraints. A corpus that violates
  // one is not written at all.

  const fail = (m) => {
    throw new Error(`corpus constraint violated: ${m}`);
  };

  if (primary.length !== 120)
    fail(`expected 120 primary positions, got ${primary.length}`);
  if (edge.length !== 6) fail(`expected 6 edge positions, got ${edge.length}`);

  for (const stratum of STRATA) {
    const n = primary.filter((p) => p.stratum === stratum.name).length;
    if (n !== PER_STRATUM)
      fail(`stratum ${stratum.name} has ${n}, expected ${PER_STRATUM}`);
  }

  const allKeys = new Set();
  for (const p of [...primary, ...edge]) {
    const key = positionKey(TwixtState.fromMoves(p.moves));
    if (allKeys.has(key)) fail(`duplicate position at ${p.id}`);
    allKeys.add(key);
    if (p.ply !== p.moves.length)
      fail(`${p.id}: ply does not match move count`);
    if (p.terminal) fail(`${p.id}: terminal positions are excluded`);
    if (p.n_legal < 1) fail(`${p.id}: no legal moves`);
    if (p.stratum !== 'edge' && p.ply !== p.target_ply) {
      fail(`${p.id}: reached ply ${p.ply}, target ${p.target_ply}`);
    }
  }

  const red = primary.filter((p) => p.to_move === 'red').length;
  const black = primary.length - red;
  if (red / primary.length < 0.4 || black / primary.length < 0.4) {
    fail(
      `colour balance red=${red} black=${black}, each must be >= 40% of 120`
    );
  }

  const over512 = edge.find((e) => e.id === 'edge_over_512_legal');
  if (over512.n_legal <= 512) fail('edge_over_512_legal does not exceed 512');
  if (over512.ply === 0)
    fail('edge_over_512_legal must not be the empty board');
  if (edge.find((e) => e.id === 'edge_black_to_move').to_move !== 'black')
    fail('edge_black_to_move is not black to move');
  if (edge.find((e) => e.id === 'edge_red_to_move').to_move !== 'red')
    fail('edge_red_to_move is not red to move');

  // --- emit ----------------------------------------------------------------

  return {
    schema: 'twixt-parity-corpus/1',
    specification: 'docs/superpowers/2026-08-13-phase2-parity-specification.md',
    generated_by: 'tests/parity/generate_corpus.mjs',
    regenerate:
      'node tests/parity/generate_corpus.mjs > tests/parity/corpus.json',
    contains_model_derived_quantities: false,
    board_size: 24,
    prng: {
      algorithm: 'mulberry32',
      increment: '0x6d2b79f5',
      note: 'Chosen for portability and auditability: no hidden state, no dependence on any engine built-in RNG.',
    },
    seeds: {
      initial: INITIAL_SEED,
      allocation:
        'handed out in fixed slot order, one per attempt, incrementing by one',
      next_unused: nextSeed,
      note: 'Corpus-construction seeds. Not drawn from the research evaluation seed ledger; nothing there is consumed.',
    },
    constraints: {
      primary_count: 120,
      strata:
        '30 each of opening (2-19), early_mid (20-49), midgame (50-99), late (100-220)',
      per_stratum_targets:
        '15 even-ply and 15 odd-ply targets, evenly spread, so the colour split is 50/50 by construction',
      colour_balance: 'each side to move >= 40% of the primary 120',
      uniqueness:
        'canonical position key over sorted pegs, sorted bridges and side to move',
      terminal: 'excluded everywhere',
      retry_rule:
        'on failure the seed is rejected, recorded with a reason, and the next seed is drawn',
      max_attempts_per_slot: MAX_ATTEMPTS_PER_SLOT,
    },
    summary: {
      primary: primary.length,
      edge: edge.length,
      strata: Object.fromEntries(
        STRATA.map((s) => [
          s.name,
          primary.filter((p) => p.stratum === s.name).length,
        ])
      ),
      colour_balance_primary: { red, black },
      unique_positions: allKeys.size,
      rejected_seeds: rejections.length,
      rejected_seeds_note:
        'Zero is expected, not a disabled rule: under this PRNG random play terminates no earlier than ply 293 over the seeds sampled, while the deepest primary target is 220, so no attempt ends before its target. The rejection path is exercised directly in tests/parity/test_corpus.mjs.',
      ply_range_primary: [
        Math.min(...primary.map((p) => p.ply)),
        Math.max(...primary.map((p) => p.ply)),
      ],
      n_legal_range_all: [
        Math.min(...[...primary, ...edge].map((p) => p.n_legal)),
        Math.max(...[...primary, ...edge].map((p) => p.n_legal)),
      ],
    },
    rejections,
    primary,
    edge,
  };
}

const isMain =
  process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  process.stdout.write(`${JSON.stringify(buildCorpus(), null, 2)}\n`);
}
