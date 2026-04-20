/**
 * MCTS for Node.js server inference.
 *
 * Key design decisions:
 * - Uses ONNX inference instead of MLX
 * - No Dirichlet noise (inference only, not training)
 * - Fixed simulation count based on difficulty
 * - Single NN eval per expansion (stores nnValue to avoid re-evaluation)
 *
 * PUCT formula: Q + c * P * sqrt(N+1) / (1 + N_child)
 * - Uses sqrt(N+1) convention (not sqrt(N))
 * - Q is negated child value (opponent's loss = our gain)
 */

/**
 * Check if AbortController signal is aborted.
 */
function isAborted(signal) {
  return !!signal && signal.aborted;
}

/**
 * Clamp value to [-1, 1] to prevent UI glitches from aggregate drift.
 */
function clamp(x) {
  return Math.max(-1, Math.min(1, x));
}

export class MCTSNode {
  constructor(state, parent = null, move = null) {
    this.state = state;
    this.parent = parent;
    this.move = move; // The move that led to this state, as [row, col]

    this.visitCount = 0;
    this.valueSum = 0;

    this.priors = null; // Map<"row,col", prior>
    this.nnValue = null; // Stored NN value (single eval per expansion)
    this.children = new Map(); // Map<"row,col", MCTSNode>
  }

  get qValue() {
    return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount;
  }

  get isExpanded() {
    return this.priors !== null;
  }
}

export class MCTS {
  constructor(inference, config = {}) {
    this.inference = inference;
    this.cPuct = config.cPuct ?? 1.5;
    this.nSimulations = config.nSimulations ?? 200;
  }

  async search(rootState, opts = {}) {
    /**
     * Run MCTS from given state.
     *
     * Options:
     *   - signal: AbortController signal for cancellation
     *   - onProgress: callback({ done, total, elapsed, valueEstimate }) for progress updates
     *   - progressEvery: emit progress every N simulations (0 = disabled)
     *   - progressMinMs: emit progress at least every N ms (0 = disabled)
     *
     * Returns:
     *   { visitCounts: Map<"row,col", count>, rootValue: number }
     */
    const { signal, onProgress, progressEvery = 0, progressMinMs = 0 } = opts;
    const t0 = Date.now();
    let lastProg = t0;

    const root = new MCTSNode(rootState);

    // Expand root
    await this._expand(root);

    // Check abort after initial expand (which does NN eval)
    if (isAborted(signal)) {
      return { visitCounts: new Map(), rootValue: 0 };
    }

    const total = this.nSimulations;

    // Run simulations
    for (let i = 0; i < total; i++) {
      // Check abort before starting simulation
      if (isAborted(signal)) break;

      let node = root;
      const searchPath = [node];

      // SELECT: traverse using PUCT until we hit unexpanded or terminal
      while (node.isExpanded && !node.state.isTerminal()) {
        const [, child] = this._selectChild(node);
        node = child;
        searchPath.push(node);
      }

      // EXPAND & EVALUATE
      let value;
      if (node.state.isTerminal()) {
        // Terminal: explicit value from perspective of node's to_move
        const winner = node.state.winner();
        if (winner === null) {
          value = 0; // Draw
        } else if (winner === node.state.toMove) {
          value = 1; // Current player won
        } else {
          value = -1; // Current player lost
        }
      } else {
        // Non-terminal leaf: expand and get NN value
        value = await this._expand(node);

        // Check abort after NN evaluation
        if (isAborted(signal)) break;
      }

      // BACKUP
      this._backup(searchPath, value);

      // Check abort after simulation completes
      if (isAborted(signal)) break;

      // Emit progress with valueEstimate for live eval bar
      if (onProgress && (progressEvery > 0 || progressMinMs > 0)) {
        const now = Date.now();
        const everyN = progressEvery > 0 && (i + 1) % progressEvery === 0;
        const everyMs = progressMinMs > 0 && now - lastProg >= progressMinMs;
        if (everyN || everyMs) {
          lastProg = now;

          // Current value estimate from root (for live eval bar)
          // Clamp to [-1, 1] to prevent UI glitches from aggregate drift
          const valueEstimate =
            root.visitCount > 0 ? clamp(root.valueSum / root.visitCount) : 0;

          onProgress({ done: i + 1, total, elapsed: now - t0, valueEstimate });
        }
      }
    }

    // Collect visit counts
    const visitCounts = new Map();
    for (const [moveKey, child] of root.children) {
      visitCounts.set(moveKey, child.visitCount);
    }

    return { visitCounts, rootValue: root.qValue };
  }

  async _expand(node) {
    /**
     * Expand node: single NN eval, store priors and value.
     * Returns: NN value for backup
     */
    const moves = node.state.legalMoves();

    if (moves.length === 0) {
      // No legal moves (should be terminal, but handle gracefully)
      node.priors = new Map();
      node.nnValue = 0;
      return 0;
    }

    const boardTensor = node.state.toTensorHWC();

    // Single NN call
    const { priors, value } = await this.inference.evaluate(boardTensor, moves);

    // Store on node (avoids second NN call in backup)
    node.priors = priors;
    node.nnValue = value;

    // Create children (unexpanded)
    for (const move of moves) {
      const key = `${move[0]},${move[1]}`;
      const childState = node.state.applyMove(move);
      node.children.set(key, new MCTSNode(childState, node, move));
    }

    return value;
  }

  _selectChild(node) {
    /**
     * Select child using PUCT: Q + c * P * sqrt(N+1) / (1 + N_child)
     *
     * Note: Uses sqrt(N+1) convention to handle unvisited root.
     * Q is negated because child's value is from opponent's perspective.
     */
    const sqrtParent = Math.sqrt(node.visitCount + 1);

    let bestScore = -Infinity;
    let bestMove = null;
    let bestChild = null;

    for (const [moveKey, child] of node.children) {
      const prior = node.priors.get(moveKey) || 0;

      // Q from child perspective (negate because opponent's loss = our gain)
      const q = child.visitCount > 0 ? -child.qValue : 0;

      // PUCT bonus
      const u = (this.cPuct * prior * sqrtParent) / (1 + child.visitCount);

      const score = q + u;

      if (score > bestScore) {
        bestScore = score;
        bestMove = moveKey;
        bestChild = child;
      } else if (score === bestScore) {
        // Lexicographic tie-break for determinism
        if (moveKey < bestMove) {
          bestMove = moveKey;
          bestChild = child;
        }
      }
    }

    return [bestMove, bestChild];
  }

  _backup(searchPath, leafValue) {
    /**
     * Propagate value up, alternating sign.
     *
     * leafValue is from perspective of the leaf node's to_move.
     * As we go up, we negate (opponent's gain = our loss).
     */
    let value = leafValue;
    for (let i = searchPath.length - 1; i >= 0; i--) {
      const node = searchPath[i];
      node.visitCount += 1;
      node.valueSum += value;
      value = -value;
    }
  }

  selectMove(visitCounts, temperature = 0.1) {
    /**
     * Select move from visit counts.
     *
     * temperature=0: deterministic (highest count, lexicographic tie-break)
     * temperature>0: sample proportional to count^(1/temp)
     */
    const moves = Array.from(visitCounts.keys());
    const counts = moves.map((m) => visitCounts.get(m));

    if (temperature < 0.01) {
      // Deterministic with lexicographic tie-break
      let maxCount = -1;
      let bestMove = null;

      for (let i = 0; i < moves.length; i++) {
        if (counts[i] > maxCount) {
          maxCount = counts[i];
          bestMove = moves[i];
        } else if (counts[i] === maxCount) {
          // Lexicographic tie-break: "r,c" string comparison
          if (moves[i] < bestMove) {
            bestMove = moves[i];
          }
        }
      }
      return bestMove;
    }

    // Temperature sampling - loop max, no spread operator
    const logCounts = counts.map((c) => Math.log(c + 1e-8) / temperature);
    let maxLog = -Infinity;
    for (let i = 0; i < logCounts.length; i++) {
      if (logCounts[i] > maxLog) maxLog = logCounts[i];
    }

    let sumExp = 0;
    const exps = [];
    for (let i = 0; i < logCounts.length; i++) {
      const exp = Math.exp(logCounts[i] - maxLog);
      exps.push(exp);
      sumExp += exp;
    }

    let r = Math.random();
    for (let i = 0; i < moves.length; i++) {
      r -= exps[i] / sumExp;
      if (r <= 0) return moves[i];
    }

    return moves[moves.length - 1];
  }

  /**
   * Select move deterministically (temperature=0).
   * Used for parity testing with Python.
   */
  selectMoveDeterministic(visitCounts) {
    return this.selectMove(visitCounts, 0);
  }
}
