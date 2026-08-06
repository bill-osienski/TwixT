/**
 * Single source of truth for AlphaZero move-readout policy.
 *
 * Both transports (REST /api/move and the WebSocket computeBestMove path)
 * MUST resolve policy through this module. Before this existed they
 * disagreed: the WS path used DIFFICULTY_PARAMS.moveTemp (1.0/0.5/0.25) while
 * REST used a hardcoded easy?0.5:0.1 and DIFFICULTY_PARAMS.moveTemp was dead
 * code on that path. deterministicMode was reachable only from REST.
 *
 * Pure: no Express, no MCTS, no I/O.
 */

// moveTemp semantics: selectMove samples proportional to count^(1/moveTemp);
// moveTemp < 0.01 is deterministic argmax.
export const DIFFICULTY_TABLE = {
  // easy/medium intentionally sacrifice strength to create difficulty levels.
  easy: { nSims: 100, moveTemp: 1.0 },
  medium: { nSims: 400, moveTemp: 0.5 },
  // hard promises strongest play, so it is deterministic.
  hard: { nSims: 800, moveTemp: 0 },
};

export const DEFAULT_DIFFICULTY = 'medium';

export function resolvePolicy({ difficulty, deterministicMode = false, temperature } = {}) {
  const key = Object.prototype.hasOwnProperty.call(DIFFICULTY_TABLE, difficulty)
    ? difficulty
    : DEFAULT_DIFFICULTY;
  const entry = DIFFICULTY_TABLE[key];
  let moveTemp;
  if (deterministicMode) {
    moveTemp = 0;
  } else if (temperature !== undefined && temperature !== null) {
    moveTemp = temperature;
  } else {
    moveTemp = entry.moveTemp;
  }
  return { difficulty: key, nSims: entry.nSims, moveTemp };
}
