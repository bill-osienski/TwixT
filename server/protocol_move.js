/**
 * One `hard` move under the H-TB protocol (integration proposal §4.6, §5).
 *
 * The single place a protocol search becomes a played move, used by both
 * transports in server/index.js. Plain searches (easy, medium, legacy hard)
 * never come here; they keep readout_policy.selectMoveForRequest.
 *
 * - Refuses a board other than 24x24 and a finished position, explicitly.
 * - Seeds two fresh streams per request (search and readout). `seeds` is a
 *   test-only argument: server/index.js never passes it, so no request can
 *   choose a stream.
 * - Plies 0-19: the 200-simulation ProtocolMCTS search, then the opening
 *   readout at T 1.0 on the readout stream. Ply >= 20: root sequential
 *   halving, whose choice is played.
 * - Values, kept truthful (the user's review, 2026-11-09):
 *     rootValue  the root mean, always (side to move);
 *     moveValue  the played move's mean from the mover's side, over its actual
 *                visit count `moveVisits` (null when it has no visits);
 *     valueKind  which of the two the display value comes from: 'root_mean'
 *                at plies 0-19, 'played_move_mean' from ply 20 (D6).
 *   `displayValue(result)` is that value; the transports derive `valueRed`
 *   from it. It is a search estimate, not a calibrated win probability.
 */
import { randomInt } from 'node:crypto';

import { PyRandom } from './py_random.js';
import { sampleByTemperature } from './protocol_mcts.js';
import { RootSequentialHalvingMCTS, HALVING, ProtocolSearchError } from './root_halving.js';

/** A request the protocol will not serve; the transports answer 400 / WS error. */
export class ProtocolRefusal extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'ProtocolRefusal';
    this.code = code;
  }
}

/** The text both transports send for an aborted search (as the plain WS path). */
export const ABORTED_MESSAGE = 'Search aborted';

const freshSeed = () => randomInt(0, 2 ** 48 - 1);

export async function protocolMove({ gameState, policy, inference, signal, onProgress, seeds }) {
  if (policy?.search !== 'protocol') {
    throw new Error(`protocolMove called with a ${policy?.search} policy`);
  }
  if (policy.moveTemp !== 1.0) {
    throw new Error(`the protocol's opening readout is T 1.0, got ${policy.moveTemp}`);
  }
  if (gameState.boardSize !== HALVING.BOARD || gameState.activeSize !== HALVING.BOARD) {
    throw new ProtocolRefusal('NOT_24', 'hard plays the validated protocol on the 24x24 board only');
  }
  if (gameState.isTerminal()) {
    throw new ProtocolRefusal('TERMINAL_ROOT', 'hard does not search a finished position');
  }

  const s = seeds ?? { search: freshSeed(), readout: freshSeed() };
  const mcts = new RootSequentialHalvingMCTS(inference, {
    nSimulations: policy.nSims,
    cPuct: 1.5,
    rng: new PyRandom(s.search),
  });

  let out;
  try {
    out = await mcts.search(gameState, { signal, onProgress, progressEvery: 50, progressMinMs: 100 });
  } catch (err) {
    if (err instanceof ProtocolSearchError) throw new ProtocolRefusal(err.code, err.message);
    throw err;
  }
  // Never a partial result: an aborted search plays nothing.
  if (out.aborted || (signal != null && signal.aborted)) throw new Error(ABORTED_MESSAGE);

  let moveKey;
  let valueKind;
  if (out.choice !== null) {
    moveKey = out.choice;
    valueKind = 'played_move_mean';
  } else {
    moveKey = sampleByTemperature(out.visitCounts, policy.moveTemp, new PyRandom(s.readout));
    valueKind = 'root_mean';
  }
  const child = mcts.lastRoot.children.get(moveKey);
  const moveVisits = child?.visitCount ?? 0;
  const moveValue = moveVisits > 0 ? -child.qValue : null;

  return {
    moveKey,
    visitCounts: out.visitCounts,
    rootValue: out.rootValue,
    moveValue,
    moveVisits,
    valueKind,
    nSimulations: policy.nSims,
    rounds: mcts.lastRounds,
  };
}

/** The value the MCTS bar shows: the played move's mean from ply 20, else the root mean. */
export function displayValue(result) {
  if (result.valueKind === 'played_move_mean') return result.moveValue;
  return result.rootValue;
}
