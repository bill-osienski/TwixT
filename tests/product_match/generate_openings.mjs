#!/usr/bin/env node
/**
 * Generate the frozen opening pool for the product-stack comparison.
 *
 *   node tests/product_match/generate_openings.mjs > tests/product_match/openings.json
 *
 * Openings exist because hard play is fully deterministic: `moveTemp: 0`, no
 * Dirichlet noise, argmax with a lexicographic tie-break. A given position and
 * colour assignment therefore yields exactly ONE game, forever — playing 400
 * games from the empty board would play one game 400 times. Diversity here is
 * structural, not a refinement.
 *
 * This file loads NO model and plays no game. Every recorded quantity is a
 * property of the rules; nothing here consults a network, selects `P`, or
 * touches the timing decision.
 *
 * Determinism: the only randomness is `mulberry32` seeded from INITIAL_SEED.
 * Slots are fixed before any seed is drawn, seeds are handed out in slot order,
 * and every rejected seed is recorded with its reason — so re-running this file
 * reproduces `openings.json` byte for byte and the rejections are auditable
 * rather than invisible.
 *
 * Specification: docs/superpowers/2026-08-14-product-stack-comparison-specification.md §4
 */
import { fileURLToPath } from 'node:url';
import { TwixtState } from '../../server/gameLogic.js';

/** Every constant below is frozen by specification §4. */
export const INITIAL_SEED = 20260814;
export const POOL_SIZE = 210;
export const MATCH_SET_SIZE = 200; // openings 0…199
export const TIMING_SET_SIZE = 10; // openings 200…209
export const OPENING_PLIES = 4; // two moves per side
export const MIN_CONTINUATIONS = 2;
export const MAX_ATTEMPTS_PER_SLOT = 2000;

/**
 * mulberry32 — four lines, no hidden state, no dependence on any engine's
 * built-in RNG, so the sequence is portable and auditable. The same generator
 * the parity corpus and the analyser's bootstrap use.
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

/**
 * Canonical identity of a position, per §4's uniqueness rule.
 *
 * Pegs alone are not sufficient: bridge formation depends on move order through
 * `_crossesExistingBridge`, so two identical peg sets can carry different
 * bridges and are different positions. Both are included, each sorted, so the
 * key is independent of the order they were produced in.
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
 * Play `plies` random legal moves from the empty board.
 *
 * Returns the reached state and whether the game ended early. A four-ply
 * opening cannot end early on a 24×24 board — a win needs a full side-to-side
 * connection — but the check is kept because it is the rule, not an assumption
 * about the board size.
 */
export function playOpening(seed, plies = OPENING_PLIES) {
  const rand = mulberry32(seed);
  let state = new TwixtState({});
  const moves = [];
  while (moves.length < plies) {
    if (state.isTerminal()) return { state, moves, endedEarly: true };
    const legal = state.legalMoves();
    if (legal.length === 0) return { state, moves, endedEarly: true };
    const move = legal[Math.floor(rand() * legal.length)];
    state = state.applyMove(move);
    moves.push([move[0], move[1]]);
  }
  return { state, moves, endedEarly: false };
}

/**
 * Does this candidate qualify? Returns `null` if it does, or a rejection reason.
 *
 * Separated from the draw loop so the rejection path can be exercised directly:
 * under this PRNG no four-ply opening is ever rejected, which would otherwise
 * leave the rule untested.
 */
export function rejectionReason(state, endedEarly, seen) {
  if (endedEarly) return { reason: 'game_ended_before_target_ply' };
  if (state.isTerminal()) return { reason: 'terminal_position' };
  const nLegal = state.legalMoves().length;
  if (nLegal < MIN_CONTINUATIONS)
    return { reason: 'too_few_continuations', n_legal: nLegal };
  const key = positionKey(state);
  if (seen.has(key))
    return { reason: 'duplicate_position', duplicate_of: seen.get(key) };
  return null;
}

/**
 * Read a committed pool file into the bare `moves` arrays every consumer wants.
 *
 * The pool records rich entries (id, role, seed, counts); the harness and the
 * analyser both index by `opening_id` and need only the move list. Without this
 * they would receive objects, `opening.length` would be `undefined`, and the
 * opening-prefix check would silently compare against an empty array — a check
 * present in the code and absent in effect.
 *
 * Lives here because this module owns the pool format. Accepts an already-bare
 * array too, so a caller cannot be wrong about which shape it holds.
 */
export function openingMovesFrom(poolFile) {
  const entries = Array.isArray(poolFile) ? poolFile : poolFile.openings;
  if (!Array.isArray(entries))
    throw new Error('opening pool has no openings array');
  return entries.map((e, i) => {
    const moves = Array.isArray(e) ? e : e.moves;
    if (!Array.isArray(moves) || moves.length !== OPENING_PLIES) {
      throw new Error(`opening ${i} does not hold ${OPENING_PLIES} moves`);
    }
    return moves;
  });
}

export function buildPool() {
  const rejections = [];
  const seen = new Map(); // positionKey -> opening id
  let nextSeed = INITIAL_SEED;
  const openings = [];

  for (let id = 0; id < POOL_SIZE; id++) {
    let placed = false;
    for (
      let attempt = 0;
      attempt < MAX_ATTEMPTS_PER_SLOT && !placed;
      attempt++
    ) {
      const seed = nextSeed++;
      const { state, moves, endedEarly } = playOpening(seed);
      const rejected = rejectionReason(state, endedEarly, seen);
      if (rejected) {
        rejections.push({ opening_id: id, seed, ...rejected });
        continue;
      }
      seen.set(positionKey(state), id);
      openings.push({
        id,
        role: id < MATCH_SET_SIZE ? 'match' : 'timing',
        seed,
        moves,
        to_move: state.toMove,
        n_legal: state.legalMoves().length,
        terminal: state.isTerminal(),
      });
      placed = true;
    }
    if (!placed) {
      throw new Error(
        `opening ${id}: no qualifying seed within ${MAX_ATTEMPTS_PER_SLOT} attempts`
      );
    }
  }

  // --- self-certification --------------------------------------------------
  // A pool that violates its own constraints is not written at all.
  const fail = (m) => {
    throw new Error(`opening pool constraint violated: ${m}`);
  };

  if (openings.length !== POOL_SIZE)
    fail(`expected ${POOL_SIZE} openings, got ${openings.length}`);
  const matchSet = openings.filter((o) => o.role === 'match');
  const timingSet = openings.filter((o) => o.role === 'timing');
  if (matchSet.length !== MATCH_SET_SIZE)
    fail(`match set has ${matchSet.length}`);
  if (timingSet.length !== TIMING_SET_SIZE)
    fail(`timing set has ${timingSet.length}`);
  if (matchSet.some((o) => o.id >= MATCH_SET_SIZE))
    fail('match set is not the prefix 0…199');
  if (timingSet.some((o) => o.id < MATCH_SET_SIZE))
    fail('timing set is not the suffix 200…209');

  const keys = new Set();
  for (const o of openings) {
    if (o.moves.length !== OPENING_PLIES)
      fail(`opening ${o.id} is not ${OPENING_PLIES} plies`);
    if (o.terminal) fail(`opening ${o.id} is terminal`);
    if (o.n_legal < MIN_CONTINUATIONS)
      fail(`opening ${o.id} has ${o.n_legal} continuations`);
    if (o.id !== openings.indexOf(o))
      fail(`opening ids are not 0…${POOL_SIZE - 1} in order`);
    const key = positionKey(TwixtState.fromMoves(o.moves));
    if (keys.has(key)) fail(`opening ${o.id} duplicates an earlier position`);
    keys.add(key);
  }
  if (keys.size !== POOL_SIZE) fail(`only ${keys.size} distinct positions`);

  return {
    schema: 'twixt-opening-pool/1',
    specification:
      'docs/superpowers/2026-08-14-product-stack-comparison-specification.md',
    generated_by: 'tests/product_match/generate_openings.mjs',
    regenerate:
      'node tests/product_match/generate_openings.mjs > tests/product_match/openings.json',
    loads_no_model: true,
    plays_no_game: true,
    board_size: 24,
    prng: {
      algorithm: 'mulberry32',
      increment: '0x6d2b79f5',
      note: 'Chosen for portability and auditability: no hidden state, no dependence on any engine built-in RNG.',
    },
    seeds: {
      initial: INITIAL_SEED,
      allocation:
        'handed out in fixed opening-id order, one per attempt, incrementing by one',
      next_unused: nextSeed,
      note: 'Opening-construction seeds. Not drawn from the research evaluation seed ledger; nothing there is consumed.',
    },
    constraints: {
      pool_size: POOL_SIZE,
      match_set: `openings 0…${MATCH_SET_SIZE - 1}`,
      timing_set: `openings ${MATCH_SET_SIZE}…${POOL_SIZE - 1}, never used in the match`,
      opening_plies: OPENING_PLIES,
      min_continuations: MIN_CONTINUATIONS,
      uniqueness:
        'canonical position key over sorted pegs, sorted bridges and side to move',
      exclusions: 'terminal, or fewer than two legal continuations',
      retry_rule:
        'on rejection the seed is recorded with a reason and the next seed is drawn',
      max_attempts_per_slot: MAX_ATTEMPTS_PER_SLOT,
    },
    summary: {
      openings: openings.length,
      match_set: matchSet.length,
      timing_set: timingSet.length,
      distinct_positions: keys.size,
      rejected_seeds: rejections.length,
      rejected_seeds_note:
        'Zero is expected, not a disabled rule: a four-ply opening cannot end a game on a 24x24 board, always leaves hundreds of continuations, and collisions among 210 four-move sequences drawn from ~528 options are vanishingly unlikely. The rejection predicate is exercised directly in tests/product_match/test_openings.mjs.',
      side_to_move: [...new Set(openings.map((o) => o.to_move))],
      side_to_move_note:
        'Uniformly red: four plies is an even number, so red is always to move. This introduces no bias, because every opening is played twice with the colours swapped (specification §4 pairing).',
      n_legal_range: [
        Math.min(...openings.map((o) => o.n_legal)),
        Math.max(...openings.map((o) => o.n_legal)),
      ],
    },
    rejections,
    openings,
  };
}

const isMain =
  process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  process.stdout.write(`${JSON.stringify(buildPool(), null, 2)}\n`);
}
