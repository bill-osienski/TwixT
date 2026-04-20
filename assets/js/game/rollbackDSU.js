/**
 * Rollback DSU (Disjoint Set Union) for TwixT connected components.
 *
 * Maintains connected components incrementally with O(α(n)) union operations.
 * Supports rollback to any snapshot for minimax search undo.
 *
 * Key design:
 * - NO path compression (breaks rollback) - uses union-by-size instead
 * - Tracks per-component metrics: size, span, edge touches
 * - Tracks bestRoot (largest/most important component) with rollback
 * - Rollback restores previous state exactly
 */
export class RollbackDSU {
  /**
   * @param {number} n - Total cells (boardSize * boardSize)
   * @param {string} player - "red" or "black"
   * @param {number} boardSize - Board dimension (e.g., 24)
   */
  constructor(n, player, boardSize) {
    this.n = n;
    this.player = player;
    this.S = boardSize;

    // DSU arrays - using typed arrays for memory efficiency
    this.parent = new Int16Array(n);
    this.size = new Int16Array(n);
    this.minR = new Int16Array(n);
    this.maxR = new Int16Array(n);
    this.minC = new Int16Array(n);
    this.maxC = new Int16Array(n);
    this.touchA = new Uint8Array(n); // Goal edge A (top for red, left for black)
    this.touchB = new Uint8Array(n); // Goal edge B (bottom for red, right for black)
    this.active = new Uint8Array(n); // Whether cell has a peg

    // Initialize each cell as its own parent
    for (let i = 0; i < n; i++) {
      this.parent[i] = i;
    }

    // Best component tracking (the "largest" component)
    this.bestRoot = -1;
    this.bestScore = -1;

    // Change stack for rollback
    this.stack = [];
  }

  /**
   * Score a root for "best component" comparison.
   * Higher is better: primary=size, secondary=span, tertiary=edge touches
   * @param {number} root - Root index
   * @returns {number} Score value
   */
  scoreRoot(root) {
    if (root < 0 || !this.active[root]) return -1;
    const size = this.size[root];
    const span =
      this.player === 'red'
        ? this.maxR[root] - this.minR[root]
        : this.maxC[root] - this.minC[root];
    const touches = this.touchA[root] + this.touchB[root];
    return size * 1000000 + span * 1000 + touches;
  }

  /**
   * Update bestRoot if the given root has a higher score.
   * Pushes old state to stack for rollback.
   * @param {number} root - Root to check
   */
  maybeUpdateBest(root) {
    const newScore = this.scoreRoot(root);
    if (newScore > this.bestScore) {
      // Push old best for rollback
      this.stack.push({
        type: 'best',
        oldRoot: this.bestRoot,
        oldScore: this.bestScore,
      });
      this.bestRoot = root;
      this.bestScore = newScore;
    }
  }

  /**
   * Get current stack depth as a snapshot marker.
   * @returns {number}
   */
  snapshot() {
    return this.stack.length;
  }

  /**
   * Rollback all changes since the given snapshot.
   * @param {number} snap - Snapshot marker from snapshot()
   */
  rollback(snap) {
    while (this.stack.length > snap) {
      const rec = this.stack.pop();
      if (rec.type === 'activate') {
        const i = rec.i;
        this.active[i] = 0;
        this.parent[i] = i;
        this.size[i] = 0;
      } else if (rec.type === 'union') {
        const { a, b, sizeA, minRA, maxRA, minCA, maxCA, touchAA, touchBA } =
          rec;
        // Restore b as separate root
        this.parent[b] = b;
        // Restore a's old aggregates
        this.size[a] = sizeA;
        this.minR[a] = minRA;
        this.maxR[a] = maxRA;
        this.minC[a] = minCA;
        this.maxC[a] = maxCA;
        this.touchA[a] = touchAA;
        this.touchB[a] = touchBA;
      } else if (rec.type === 'best') {
        // Restore old best
        this.bestRoot = rec.oldRoot;
        this.bestScore = rec.oldScore;
      }
    }
  }

  /**
   * Find root of component containing x (no path compression).
   * @param {number} x - Cell index
   * @returns {number} Root index
   */
  find(x) {
    while (this.parent[x] !== x) {
      x = this.parent[x];
    }
    return x;
  }

  /**
   * Activate a cell (place a peg).
   * @param {number} i - Cell index
   * @param {number} row - Row coordinate
   * @param {number} col - Column coordinate
   */
  activate(i, row, col) {
    // Record for rollback
    this.stack.push({ type: 'activate', i });

    this.active[i] = 1;
    this.parent[i] = i;
    this.size[i] = 1;
    this.minR[i] = this.maxR[i] = row;
    this.minC[i] = this.maxC[i] = col;

    // Goal edges depend on player
    if (this.player === 'red') {
      // Red connects top (row 0) to bottom (row S-1)
      this.touchA[i] = row === 0 ? 1 : 0;
      this.touchB[i] = row === this.S - 1 ? 1 : 0;
    } else {
      // Black connects left (col 0) to right (col S-1)
      this.touchA[i] = col === 0 ? 1 : 0;
      this.touchB[i] = col === this.S - 1 ? 1 : 0;
    }

    // Check if this new singleton is now the best
    this.maybeUpdateBest(i);
  }

  /**
   * Union two components (when a bridge connects them).
   * Uses union-by-size for efficiency without path compression.
   * @param {number} x - Cell index 1
   * @param {number} y - Cell index 2
   * @returns {number} New root
   */
  union(x, y) {
    let rx = this.find(x);
    let ry = this.find(y);
    if (rx === ry) return rx;

    // Union by size - merge smaller into larger
    if (this.size[rx] < this.size[ry]) {
      const t = rx;
      rx = ry;
      ry = t;
    }

    // Save rx state for rollback
    this.stack.push({
      type: 'union',
      a: rx,
      b: ry,
      sizeA: this.size[rx],
      minRA: this.minR[rx],
      maxRA: this.maxR[rx],
      minCA: this.minC[rx],
      maxCA: this.maxC[rx],
      touchAA: this.touchA[rx],
      touchBA: this.touchB[rx],
    });

    // Merge ry into rx
    this.parent[ry] = rx;
    this.size[rx] += this.size[ry];
    this.minR[rx] = Math.min(this.minR[rx], this.minR[ry]);
    this.maxR[rx] = Math.max(this.maxR[rx], this.maxR[ry]);
    this.minC[rx] = Math.min(this.minC[rx], this.minC[ry]);
    this.maxC[rx] = Math.max(this.maxC[rx], this.maxC[ry]);
    this.touchA[rx] |= this.touchA[ry];
    this.touchB[rx] |= this.touchB[ry];

    // Check if merged component is now the best
    this.maybeUpdateBest(rx);

    return rx;
  }

  /**
   * Get the best (largest/most important) component root.
   * @returns {number} Root index, or -1 if no components
   */
  getBestRoot() {
    return this.bestRoot;
  }

  /**
   * Get metrics for a component given its root.
   * @param {number} root - Root index from find()
   * @returns {Object} Component metrics
   */
  rootMetrics(root) {
    if (root < 0 || !this.active[root]) {
      return {
        size: 0,
        minR: 0,
        maxR: 0,
        minC: 0,
        maxC: 0,
        touchA: false,
        touchB: false,
        spanR: 0,
        spanC: 0,
        finished: 0,
      };
    }
    return {
      size: this.size[root],
      minR: this.minR[root],
      maxR: this.maxR[root],
      minC: this.minC[root],
      maxC: this.maxC[root],
      touchA: !!this.touchA[root],
      touchB: !!this.touchB[root],
      spanR: this.maxR[root] - this.minR[root],
      spanC: this.maxC[root] - this.minC[root],
      finished: this.touchA[root] && this.touchB[root] ? 1 : 0,
    };
  }

  /**
   * Check if a cell is active (has a peg).
   * @param {number} i - Cell index
   * @returns {boolean}
   */
  isActive(i) {
    return this.active[i] === 1;
  }

  /**
   * Reset DSU to initial state (all cells inactive).
   */
  reset() {
    for (let i = 0; i < this.n; i++) {
      this.parent[i] = i;
      this.size[i] = 0;
      this.active[i] = 0;
    }
    this.bestRoot = -1;
    this.bestScore = -1;
    this.stack = [];
  }
}

/**
 * Helper to convert (row, col) to flat index.
 * @param {number} row
 * @param {number} col
 * @param {number} boardSize
 * @returns {number}
 */
export function cellIndex(row, col, boardSize) {
  return row * boardSize + col;
}
