/**
 * The research PUCT tie rule and opening readout, for the H-TB protocol only
 * (integration proposal §4.3). `easy` and `medium` never use this module: they
 * keep server/mcts.js's lexicographic exact-equality tie-break.
 *
 * Transcribed from scripts/GPU/alphazero/mcts.py and eval_readout.py:
 *
 *   - At each expansion the moves are shuffled with the SEARCH stream
 *     (`_expand_batch`: `self.rng.shuffle(pairs)`), and that order is the
 *     node's iteration order for selection.
 *   - `_select_child` keeps a RUNNING best: a score more than 1e-8 above the
 *     current best starts a new set; a score within 1e-8 of the current best
 *     joins it. It then plays `rng.choice` over the set, which consumes the
 *     stream even for a one-member set.
 *   - The opening readout samples proportional to count^(1/T) with the 1e-8
 *     floor and max-subtraction (`_sample_by_temperature`), on a SEPARATE
 *     readout stream.
 *
 * server/mcts.js is not edited: MCTS.search already dispatches through
 * `this._expand` and `this._selectChild`.
 */
import { MCTS } from './mcts.js';

export const TIE_EPS = 1e-8;

export class ProtocolMCTS extends MCTS {
  /**
   * @param {object} inference - `evaluate(boardTensorHWC, moves)`
   * @param {object} config - { nSimulations, cPuct, rng } where `rng` is the
   *   search stream (a PyRandom); required, never defaulted.
   */
  constructor(inference, config = {}) {
    super(inference, config);
    if (!config.rng) throw new Error('ProtocolMCTS needs its search stream (config.rng)');
    this.rng = config.rng;
    this.lastRoot = null;
  }

  async _expand(node) {
    const value = await super._expand(node);
    // The research shuffles after the evaluator call, once per expanded node.
    // `node.moves` keeps legal-move order, so the visit-count output order (and
    // the readout's iteration order) is unchanged.
    const order = node.moves.slice();
    this.rng.shuffle(order);
    node.order = order;
    if (node.parent === null) this.lastRoot = node;
    return value;
  }

  _selectChild(node) {
    const c = this.cPuct;
    const sqrtParent = Math.sqrt(node.visitCount + 1);
    let bestScore = -Infinity;
    let bestMoves = [];
    for (const move of node.order) {
      const key = `${move[0]},${move[1]}`;
      const prior = node.priors.get(key);
      const child = node.children.get(key);
      let q;
      let n;
      if (child !== undefined && child.visitCount > 0) {
        q = -child.qValue;
        n = child.visitCount;
      } else {
        q = 0.0; // FPU 0, as the protocol's MCTSConfig
        n = 0;
      }
      // The research's operation order: c * prior * sqrt_parent / (1 + n).
      const u = (c * prior * sqrtParent) / (1 + n);
      const score = q + u;
      if (score > bestScore + TIE_EPS) {
        bestScore = score;
        bestMoves = [move];
      } else if (Math.abs(score - bestScore) <= TIE_EPS) {
        bestMoves.push(move);
      }
    }
    if (bestMoves.length === 0) throw new Error('no selectable moves in _selectChild');
    const move = this.rng.choice(bestMoves);
    const key = `${move[0]},${move[1]}`;
    return [key, this._childFor(node, key, move)];
  }
}

/**
 * CPython's built-in sum() over floats (3.12+): Neumaier compensated summation,
 * starting from the integer 0, with the compensation added only when it is
 * non-zero and finite.
 */
export function pySum(xs) {
  let total = 0.0;
  let comp = 0.0;
  for (const x of xs) {
    const t = total + x;
    if (Math.abs(total) >= Math.abs(x)) comp += total - t + x;
    else comp += x - t + total;
    total = t;
  }
  if (comp !== 0 && Number.isFinite(comp)) total += comp;
  return total;
}

/**
 * eval_readout._sample_by_temperature: sample a move key from `counts` (a Map
 * in legal-move order) proportional to count^(1/temp), with the readout stream.
 */
export function sampleByTemperature(counts, temp, rng) {
  const moves = [...counts.keys()];
  if (moves.length === 0) throw new Error('sampleByTemperature called with empty visit counts');
  const logCounts = moves.map((m) => Math.log(counts.get(m) + 1e-8) / temp);
  let maxLog = logCounts[0];
  for (const lc of logCounts) if (lc > maxLog) maxLog = lc;
  const exps = logCounts.map((lc) => Math.exp(lc - maxLog));
  const total = pySum(exps);
  const r = rng.random();
  let cumsum = 0.0;
  for (let i = 0; i < moves.length; i++) {
    cumsum += exps[i] / total;
    if (r <= cumsum) return moves[i];
  }
  return moves[moves.length - 1];
}
