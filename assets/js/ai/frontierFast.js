/**
 * Fast frontier computation using DSU bounds and stamp tables.
 *
 * Avoids:
 * - componentMetrics() calls (expensive BFS)
 * - Set<string> allocations
 * - Object creation per neighbor
 *
 * Uses:
 * - DSU best root + bounds (O(1))
 * - Band scan near leading edges
 * - Stamp table for O(1) membership (no Set/Map)
 * - Reusable buffers (no per-call allocations)
 */

// Knight move offsets for TwixT bridges
const KNIGHT = [
  [-2, -1],
  [-2, +1],
  [-1, -2],
  [-1, +2],
  [+1, -2],
  [+1, +2],
  [+2, -1],
  [+2, +1],
];

/**
 * Reusable buffers for frontier computation.
 * Create once per AI instance, reuse across all minimax nodes.
 */
export class FrontierBuffers {
  constructor(boardSize) {
    this.S = boardSize;
    this.N = boardSize * boardSize;

    // Stamp table for O(1) membership checks (avoids Set allocations)
    this.stamp = new Uint32Array(this.N);
    this.cur = 1;

    // Output arrays (cell indices, not objects)
    this.frontier = []; // All frontier cells
    this.connectors = []; // Cells in connector band
    this.trailing = []; // Other frontier cells
    this.targets = []; // Connector target cells from bounds
  }

  /**
   * Advance stamp counter (avoids clearing array).
   */
  nextStamp() {
    this.cur += 2; // Need 2 values per iteration (targets + seen)
    if (this.cur >= 0xfffffffe) {
      this.cur = 1;
      this.stamp.fill(0);
    }
  }

  /**
   * Clear output arrays for reuse.
   */
  clear() {
    this.frontier.length = 0;
    this.connectors.length = 0;
    this.trailing.length = 0;
    this.targets.length = 0;
  }
}

// Must match REWARDS.edge.radius from search.json
const EDGE_RADIUS = 3;

/**
 * Compute connector targets from DSU bounds only (no member iteration).
 * Adds target cell indices to outArr.
 * MUST match computeConnectorTargets() from heuristics.js for parity.
 *
 * @param {Object} game - TwixTGame instance
 * @param {string} player - "red" or "black"
 * @param {Object} m - DSU metrics (minR, maxR, minC, maxC)
 * @param {Array} outArr - Output array (will be cleared and filled)
 * @returns {Array} outArr with target indices
 */
export function computeConnectorTargetsFromBounds(game, player, m, outArr) {
  outArr.length = 0;
  if (!m || m.size === 0) return outArr;

  const S = game.boardSize;

  if (player === 'red') {
    // Red extends vertically - targets are rows above/below bounds
    // Expand column range by radius (matches old computeConnectorTargets)
    const r1 = m.minR - 1;
    const r2 = m.maxR + 1;
    for (let c = m.minC - EDGE_RADIUS; c <= m.maxC + EDGE_RADIUS; c++) {
      if (c < 0 || c >= S) continue;
      if (r1 >= 0 && game.isEmpty(r1, c)) {
        // Check placement legality: red can't place on left/right edges
        if (c !== 0 && c !== S - 1) {
          outArr.push(r1 * S + c);
        }
      }
      if (r2 < S && game.isEmpty(r2, c)) {
        if (c !== 0 && c !== S - 1) {
          outArr.push(r2 * S + c);
        }
      }
    }
  } else {
    // Black extends horizontally - targets are cols left/right of bounds
    // Expand row range by radius (matches old computeConnectorTargets)
    const c1 = m.minC - 1;
    const c2 = m.maxC + 1;
    for (let r = m.minR - EDGE_RADIUS; r <= m.maxR + EDGE_RADIUS; r++) {
      if (r < 0 || r >= S) continue;
      if (c1 >= 0 && game.isEmpty(r, c1)) {
        // Check placement legality: black can't place on top/bottom edges
        if (r !== 0 && r !== S - 1) {
          outArr.push(r * S + c1);
        }
      }
      if (c2 < S && game.isEmpty(r, c2)) {
        if (r !== 0 && r !== S - 1) {
          outArr.push(r * S + c2);
        }
      }
    }
  }

  return outArr;
}

/**
 * Fast frontier computation using DSU bounds and band scanning.
 *
 * Instead of iterating all pegs in largest component (which requires
 * maintaining component membership), we:
 * 1. Get best component bounds from DSU (O(1))
 * 2. Scan player pegs within K rows/cols of bounds
 * 3. Add knight-neighbor empty cells to frontier
 *
 * @param {Object} game - TwixTGame instance
 * @param {string} player - "red" or "black"
 * @param {FrontierBuffers} buffers - Reusable buffers
 * @returns {FrontierBuffers} Same buffers object with filled arrays
 */
export function computeFrontierFast(game, player, buffers) {
  const S = buffers.S;
  buffers.clear();
  buffers.nextStamp();

  // Get best component root and metrics
  const root = game.getBestComponentRoot(player);
  if (root < 0) return buffers;

  const m = game.getDSUMetrics(player, root);
  if (!m || m.size === 0) return buffers;

  // 1) Compute connector targets from bounds
  computeConnectorTargetsFromBounds(game, player, m, buffers.targets);

  // Mark connector targets in stamp table (value = cur)
  const targetMark = buffers.cur;
  for (let i = 0; i < buffers.targets.length; i++) {
    buffers.stamp[buffers.targets[i]] = targetMark;
  }

  // Stamp value for "seen in frontier"
  const seenMark = buffers.cur + 1;

  // 2) Scan pegs within K rows/cols of bounds (thin band)
  const K = 3;
  const rLo = Math.max(0, m.minR - K);
  const rHi = Math.min(S - 1, m.maxR + K);
  const cLo = Math.max(0, m.minC - K);
  const cHi = Math.min(S - 1, m.maxC + K);

  // Get DSU to check component membership
  const dsu = player === 'red' ? game.redDSU : game.blackDSU;

  // Iterate player pegs (not full board scan)
  // CRITICAL: Only include pegs in the BEST component (matching old computeFrontier behavior)
  const pegs = game.getPlayerPegs(player);
  for (let p = 0; p < pegs.length; p++) {
    const row = pegs[p].row;
    const col = pegs[p].col;

    // Skip pegs not in the best component
    const pegIdx = row * S + col;
    if (dsu && dsu.find(pegIdx) !== root) continue;

    // Skip pegs outside the band
    if (row < rLo || row > rHi || col < cLo || col > cHi) continue;

    // Check knight neighbors for empty cells
    for (let k = 0; k < 8; k++) {
      const rr = row + KNIGHT[k][0];
      const cc = col + KNIGHT[k][1];

      // Bounds check
      if (rr < 0 || rr >= S || cc < 0 || cc >= S) continue;

      // Must be empty
      if (!game.isEmpty(rr, cc)) continue;

      // Placement legality for player
      const atTopOrBottom = rr === 0 || rr === S - 1;
      const atLeftOrRight = cc === 0 || cc === S - 1;
      if (atTopOrBottom && atLeftOrRight) continue; // Corner
      if (player === 'red' && atLeftOrRight) continue;
      if (player === 'black' && atTopOrBottom) continue;

      const idx = rr * S + cc;

      // Skip if already seen (use stamp table, not Set)
      if (buffers.stamp[idx] === seenMark) continue;
      buffers.stamp[idx] = seenMark;

      // Add to frontier
      buffers.frontier.push(idx);

      // Classify: connector if in target band, else trailing
      if (buffers.stamp[idx] === targetMark) {
        // Note: this check is now false since we just set it to seenMark
        // Need to check before setting seenMark
      }
    }
  }

  // Re-classify frontier into connectors vs trailing
  // MUST match old computeFrontier logic: connector = near GOAL EDGE (not component bounds)
  // Use translated property names from getDSUMetrics
  const wantTop = player === 'red' ? !m.touchesTop : false;
  const wantBottom = player === 'red' ? !m.touchesBottom : false;
  const wantLeft = player === 'black' ? !m.touchesLeft : false;
  const wantRight = player === 'black' ? !m.touchesRight : false;

  for (let i = 0; i < buffers.frontier.length; i++) {
    const idx = buffers.frontier[i];
    const row = (idx / S) | 0;
    const col = idx % S;

    let isConnector = false;
    if (player === 'red') {
      const topThreshold = wantTop ? 5 : 3;
      const bottomThreshold = wantBottom ? 5 : 3;
      if (wantTop && row <= topThreshold) isConnector = true;
      if (wantBottom && row >= S - 1 - bottomThreshold) isConnector = true;
      if (!wantTop && !wantBottom && (row <= topThreshold || row >= S - 1 - bottomThreshold)) {
        isConnector = true;
      }
    } else {
      const leftThreshold = wantLeft ? 5 : 3;
      const rightThreshold = wantRight ? 5 : 3;
      if (wantLeft && col <= leftThreshold) isConnector = true;
      if (wantRight && col >= S - 1 - rightThreshold) isConnector = true;
      if (!wantLeft && !wantRight && (col <= leftThreshold || col >= S - 1 - rightThreshold)) {
        isConnector = true;
      }
    }

    if (isConnector) {
      buffers.connectors.push(idx);
    } else {
      buffers.trailing.push(idx);
    }
  }

  return buffers;
}

/**
 * Convert cell index to row.
 * @param {number} idx - Cell index
 * @param {number} S - Board size
 * @returns {number} Row
 */
export function idxToRow(idx, S) {
  return (idx / S) | 0;
}

/**
 * Convert cell index to col.
 * @param {number} idx - Cell index
 * @param {number} S - Board size
 * @returns {number} Col
 */
export function idxToCol(idx, S) {
  return idx % S;
}

/**
 * Per-depth buffer pool for minimax.
 * Solves buffer aliasing: each recursion depth gets its own FrontierBuffers,
 * preventing deeper calls from overwriting data still in use by upper levels.
 *
 * Usage:
 *   const pool = new FrontierBufferPool(boardSize, maxDepth);
 *   // In minimax at depth d:
 *   const buf = pool.get(d);
 *   computeFrontierFast(game, player, buf);
 *   // buf.frontier, buf.connectors, etc. are valid until next get(d) call
 */
export class FrontierBufferPool {
  constructor(boardSize, maxDepth = 8) {
    this.buffers = [];
    for (let d = 0; d <= maxDepth; d++) {
      this.buffers.push(new FrontierBuffers(boardSize));
    }
  }

  /**
   * Get buffer for given depth level.
   * @param {number} depth - Current minimax depth (0 = root)
   * @returns {FrontierBuffers}
   */
  get(depth) {
    // Clamp to pool size (in case depth exceeds maxDepth)
    const idx = Math.min(depth, this.buffers.length - 1);
    return this.buffers[idx];
  }
}
