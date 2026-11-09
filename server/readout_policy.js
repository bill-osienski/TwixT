/**
 * Single source of truth for AlphaZero move-readout policy.
 *
 * Both transports (REST /api/move and the WebSocket computeBestMove path)
 * MUST resolve policy through this module. Before this existed they
 * disagreed: the WS path used DIFFICULTY_PARAMS.moveTemp (1.0/0.5/0.25) while
 * REST used a hardcoded easy?0.5:0.1 and DIFFICULTY_PARAMS.moveTemp was dead
 * code on that path. deterministicMode was reachable only from REST.
 *
 * `hard` has two modes, chosen once at server startup by HARD_MODE
 * (integration proposal §4.5, §8):
 *   - 'protocol' (the default): the validated H-TB protocol, served by
 *     server/protocol_move.js, never by selectMoveForRequest;
 *   - 'legacy': today's hard, kept by name as LEGACY_HARD: plain MCTS, 800
 *     simulations, argmax. This is the rollback.
 * Resolving 'hard' requires the mode explicitly, so no caller can get either
 * one by default.
 *
 * Pure: no Express, no MCTS, no I/O.
 */

// moveTemp semantics: selectMove samples proportional to count^(1/moveTemp);
// moveTemp < 0.01 is deterministic argmax.
//
// The validated H-TB protocol (ledger entry 54): 200 simulations; opening
// readout at T 1.0 for plies 0-19; root sequential halving's choice from ply 20.
export const PROTOCOL_HARD = Object.freeze({ nSims: 200, search: 'protocol', moveTemp: 1.0 });

// Today's hard, kept by name for HARD_MODE=legacy. hard promised strongest
// play, so it was deterministic.
export const LEGACY_HARD = Object.freeze({ nSims: 800, search: 'plain', moveTemp: 0 });

export const DIFFICULTY_TABLE = {
  // easy/medium intentionally sacrifice strength to create difficulty levels.
  easy: { nSims: 100, search: 'plain', moveTemp: 1.0 },
  medium: { nSims: 400, search: 'plain', moveTemp: 0.5 },
  hard: PROTOCOL_HARD,
};

export const DEFAULT_DIFFICULTY = 'medium';

export const HARD_MODES = Object.freeze(['protocol', 'legacy']);
export const DEFAULT_HARD_MODE = 'protocol';

/** HARD_MODE from the environment: unset or empty means the default; anything else must be a known mode. */
export function parseHardMode(env = process.env) {
  const raw = env.HARD_MODE;
  if (raw === undefined || raw === '') return DEFAULT_HARD_MODE;
  if (!HARD_MODES.includes(raw)) {
    throw new Error(`HARD_MODE must be one of ${HARD_MODES.join(', ')}; got ${JSON.stringify(raw)}`);
  }
  return raw;
}

export const POLICY_CONFLICT_MESSAGE =
  'hard plays the validated protocol; temperature and deterministicMode overrides are not accepted';

/** A request whose overrides conflict with the protocol (decision D3: refused, not ignored). */
export class PolicyConflict extends Error {
  constructor(message = POLICY_CONFLICT_MESSAGE) {
    super(message);
    this.name = 'PolicyConflict';
    this.code = 'POLICY_CONFLICT';
  }
}

export function resolvePolicy({ difficulty, deterministicMode = false, temperature, hardMode } = {}) {
  const key = Object.prototype.hasOwnProperty.call(DIFFICULTY_TABLE, difficulty)
    ? difficulty
    : DEFAULT_DIFFICULTY;
  let entry = DIFFICULTY_TABLE[key];
  if (key === 'hard') {
    if (!HARD_MODES.includes(hardMode)) {
      throw new Error(`resolving 'hard' needs hardMode (${HARD_MODES.join(' or ')}), got ${String(hardMode)}`);
    }
    if (hardMode === 'legacy') entry = LEGACY_HARD;
  }
  if (entry.search === 'protocol') {
    // deterministicMode: false (or absent) asks for nothing different; any
    // truthy deterministicMode, or any temperature at all, conflicts.
    if (deterministicMode || (temperature !== undefined && temperature !== null)) {
      throw new PolicyConflict();
    }
    return { difficulty: key, nSims: entry.nSims, moveTemp: entry.moveTemp, search: 'protocol' };
  }
  let moveTemp;
  if (deterministicMode) {
    moveTemp = 0;
  } else if (temperature !== undefined && temperature !== null) {
    moveTemp = temperature;
  } else {
    moveTemp = entry.moveTemp;
  }
  return { difficulty: key, nSims: entry.nSims, moveTemp, search: 'plain' };
}

/**
 * The ONE place a completed PLAIN search becomes a played move.
 *
 * Both transports call this. `selectMove` is injected so the seam is testable
 * without an engine, and so parity can be asserted behaviourally: the same
 * request must reach the readout with the same temperature no matter which
 * transport carried it. A protocol policy is refused here: its move comes from
 * server/protocol_move.js.
 *
 * @param {object} args
 * @param {Map<string, number>} args.visitCounts - "row,col" -> visit count
 * @param {string} args.difficulty
 * @param {boolean} [args.deterministicMode]
 * @param {number} [args.temperature]
 * @param {string} [args.hardMode] - required when difficulty is 'hard'
 * @param {(counts: Map<string, number>, temp: number) => string} args.selectMove
 * @returns {{moveKey: string, policy: {difficulty: string, nSims: number, moveTemp: number, search: string}}}
 */
export function selectMoveForRequest({
  visitCounts,
  difficulty,
  deterministicMode = false,
  temperature,
  hardMode,
  selectMove,
}) {
  const policy = resolvePolicy({ difficulty, deterministicMode, temperature, hardMode });
  if (policy.search !== 'plain') {
    throw new Error(`selectMoveForRequest serves plain searches only, got ${policy.search}`);
  }
  return { moveKey: selectMove(visitCounts, policy.moveTemp), policy };
}
