/**
 * Root sequential halving for the H-TB protocol (integration proposal §4.4).
 *
 * A transcription of scripts/GPU/alphazero/root_sequential_halving.py
 * (RootSequentialHalvingMCTS), with the same frozen constants:
 *
 *   - plies 0-19: the ordinary search (here ProtocolMCTS, with the research tie
 *     rule); no choice is made and the caller's readout applies;
 *   - from ply 20, 24x24 only, no noise: the CANDIDATES highest-prior legal
 *     moves (fewer if fewer are legal), ties by (row, col);
 *   - R = ceil(log2 m) rounds. Each round but the last gives every survivor
 *     N // (R * survivors) visits, round-robin in rank order, then keeps the top
 *     ceil(survivors / 2) by score. The last round splits the rest evenly, any
 *     remainder one visit each in rank order;
 *   - score = log(max(prior, 1e-12)) + (C_VISIT + max visits) * C_SCALE *
 *     (q + 1) / 2, q the candidate's mean from the root player's side; rank
 *     ties: higher prior, then (row, col);
 *   - the top-ranked final survivor is the move to play.
 *
 * Descent below the root is ProtocolMCTS's PUCT through a root-override seam,
 * as `_run_single_simulation(root, root_move_override=m)`.
 *
 * The visit allocation depends on the number of candidates m; nothing here
 * assumes m = 16. The played move's statistics are reported from its actual
 * visit count.
 */
import { MCTSNode } from './mcts.js';
import { ProtocolMCTS } from './protocol_mcts.js';

export const HALVING = Object.freeze({
  FROM_PLY: 20,
  CANDIDATES: 16,
  C_VISIT: 50.0,
  C_SCALE: 1.0,
  PRIOR_FLOOR: 1e-12,
  BOARD: 24,
});

export class ProtocolSearchError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'ProtocolSearchError';
    this.code = code;
  }
}

export function score(prior, qRoot, maxVisits) {
  const { C_VISIT, C_SCALE, PRIOR_FLOOR } = HALVING;
  return Math.log(Math.max(prior, PRIOR_FLOOR)) + ((C_VISIT + maxVisits) * C_SCALE * (qRoot + 1.0)) / 2.0;
}

/** Python's int.bit_length for a non-negative integer. */
function bitLength(n) {
  let k = 0;
  for (let v = n; v > 0; v = Math.floor(v / 2)) k += 1;
  return k;
}

const decode = (key) => key.split(',').map(Number);
const byRowCol = (a, b) => {
  const [ra, ca] = decode(a);
  const [rb, cb] = decode(b);
  return ra - rb || ca - cb;
};

function terminalValue(state) {
  const winner = state.winner();
  if (winner === null) return 0;
  return winner === state.toMove ? 1 : -1;
}

const isAborted = (signal) => signal != null && signal.aborted === true;

export class RootSequentialHalvingMCTS extends ProtocolMCTS {
  constructor(inference, config = {}) {
    super(inference, config);
    this.lastRounds = null; // survivors (move keys) entering each visiting round
  }

  /**
   * @returns {Promise<{visitCounts: Map<string, number>, rootValue: number,
   *   choice: string|null, aborted: boolean}>}
   *   `choice` is null below ply 20 (the caller's readout applies).
   */
  async search(rootState, opts = {}) {
    if (rootState.ply < HALVING.FROM_PLY) {
      this.lastRounds = null;
      const out = await super.search(rootState, opts);
      return { ...out, choice: null, aborted: isAborted(opts.signal) };
    }
    if (rootState.activeSize !== HALVING.BOARD || rootState.boardSize !== HALVING.BOARD) {
      throw new ProtocolSearchError('NOT_24', `24x24 only, got active size ${rootState.activeSize}`);
    }
    if (rootState.isTerminal()) {
      throw new ProtocolSearchError('TERMINAL_ROOT', 'no search from a terminal root');
    }
    const { signal, onProgress, progressEvery = 0, progressMinMs = 0 } = opts;
    const n = this.nSimulations;
    const root = new MCTSNode(rootState);
    await this._expand(root);
    const abortedResult = () => ({ visitCounts: new Map(), rootValue: 0, choice: null, aborted: true });
    if (isAborted(signal)) return abortedResult();
    const prior = root.priors;

    const t0 = Date.now();
    let lastProg = t0;
    let done = 0;
    const progress = () => {
      if (!onProgress || !(progressEvery > 0 || progressMinMs > 0)) return;
      const now = Date.now();
      const everyN = progressEvery > 0 && done % progressEvery === 0;
      const everyMs = progressMinMs > 0 && now - lastProg >= progressMinMs;
      if (everyN || everyMs) {
        lastProg = now;
        const valueEstimate = root.visitCount > 0 ? Math.max(-1, Math.min(1, root.valueSum / root.visitCount)) : 0;
        onProgress({ done, total: n, elapsed: now - t0, valueEstimate });
      }
    };

    const rank = (keys) => {
      let top = 0;
      for (const child of root.children.values()) if (child.visitCount > top) top = child.visitCount;
      const s = new Map(keys.map((k) => [k, score(prior.get(k), -root.children.get(k).qValue, top)]));
      return keys.slice().sort((a, b) => {
        const sa = s.get(a);
        const sb = s.get(b);
        if (sa !== sb) return sa > sb ? -1 : 1;
        const pa = prior.get(a);
        const pb = prior.get(b);
        if (pa !== pb) return pa > pb ? -1 : 1;
        return byRowCol(a, b);
      });
    };

    // Round-robin root-override simulations, in the allotment's (rank) order.
    const visit = async (allotment) => {
      let most = 0;
      for (const [, k] of allotment) if (k > most) most = k;
      for (let t = 0; t < most; t++) {
        for (const [key, k] of allotment) {
          if (t < k) {
            if (isAborted(signal)) return false;
            await this._simulate(root, key);
            done += 1;
            progress();
          }
        }
      }
      return true;
    };

    const keys = root.moves.map((m) => `${m[0]},${m[1]}`);
    let survivors = keys
      .slice()
      .sort((a, b) => {
        const pa = prior.get(a);
        const pb = prior.get(b);
        if (pa !== pb) return pa > pb ? -1 : 1;
        return byRowCol(a, b);
      })
      .slice(0, HALVING.CANDIDATES);
    const rounds = bitLength(survivors.length - 1);
    this.lastRounds = [];
    let used = 0;
    for (let i = 0; i < rounds - 1; i++) {
      const each = Math.floor(n / (rounds * survivors.length));
      if (each < 1) {
        throw new ProtocolSearchError('BUDGET', `${n} simulations cannot visit ${survivors.length} survivors`);
      }
      this.lastRounds.push(survivors.slice());
      if (!(await visit(survivors.map((k) => [k, each])))) return abortedResult();
      used += each * survivors.length;
      survivors = rank(survivors).slice(0, Math.ceil(survivors.length / 2));
    }
    const remaining = n - used;
    const each = Math.floor(remaining / survivors.length);
    const extra = remaining % survivors.length;
    if (each < 1) {
      throw new ProtocolSearchError('BUDGET', `${n} simulations cannot visit ${survivors.length} survivors`);
    }
    this.lastRounds.push(survivors.slice());
    if (!(await visit(survivors.map((k, i) => [k, each + (i < extra ? 1 : 0)])))) return abortedResult();
    const choice = rank(survivors)[0];

    const visitCounts = new Map();
    for (const key of keys) visitCounts.set(key, root.children.get(key)?.visitCount ?? 0);
    return { visitCounts, rootValue: root.qValue, choice, aborted: false };
  }

  /** One synchronous simulation whose first hop is forced to `overrideKey`. */
  async _simulate(root, overrideKey) {
    let node = root;
    const path = [node];
    let override = overrideKey;
    while (node.isExpanded && !node.state.isTerminal()) {
      let child;
      if (override !== null) {
        child = this._childFor(node, override, decode(override));
        override = null;
      } else {
        [, child] = this._selectChild(node);
      }
      path.push(child);
      node = child;
    }
    const value = node.state.isTerminal() ? terminalValue(node.state) : await this._expand(node);
    this._backup(path, value);
  }
}
