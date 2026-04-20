import { getZobristTable, playerIndex } from './zobrist.js';
import { RollbackDSU, cellIndex } from './rollbackDSU.js';

export default class TwixTGame {
  constructor() {
    this.boardSize = 24;
    this.board = Array(this.boardSize)
      .fill(null)
      .map(() => Array(this.boardSize).fill(null));
    this.currentPlayer = 'red';
    this.pegs = [];
    this.bridges = [];
    this.moveHistory = [];
    this.gameOver = false;
    this.winner = null;
    this.moveCount = 0;
    this.startingPlayer = 'red';

    // Zobrist hash for O(1) incremental board state hashing
    this.zTable = getZobristTable(this.boardSize);
    this.zKey = 0n;

    // Rollback DSU for incremental connected component tracking
    const n = this.boardSize * this.boardSize;
    this.redDSU = new RollbackDSU(n, 'red', this.boardSize);
    this.blackDSU = new RollbackDSU(n, 'black', this.boardSize);

    // AI settings
    this.isAIGame = false;
    this.aiPlayer = 'black'; // AI always plays black
    this.aiDifficulty = 'medium';
    this.aiDepth = { easy: 2, medium: 3, hard: 4 };
  }

  isValidPegPlacement(row, col) {
    // Occupied?
    if (this.board[row][col] !== null) return false;

    // Corners forbidden
    if (
      (row === 0 || row === this.boardSize - 1) &&
      (col === 0 || col === this.boardSize - 1)
    ) {
      return false;
    }

    // Edge legality per player (you can place on *your* goal edges, not the opponent's)
    if (this.currentPlayer === 'red') {
      // Red connects top↔bottom; cannot place on left/right edges
      if (col === 0 || col === this.boardSize - 1) return false;
    } else {
      // Black connects left↔right; cannot place on top/bottom edges
      if (row === 0 || row === this.boardSize - 1) return false;
    }

    return true;
  }

  placePeg(row, col) {
    if (!this.isValidPegPlacement(row, col)) return false;
    return this._placePegInternal(row, col);
  }

  /**
   * Force-place a peg without validation.
   * Used by replay viewer to show exactly what was recorded (for debugging).
   */
  forcePlacePeg(row, col) {
    return this._placePegInternal(row, col);
  }

  _placePegInternal(row, col) {
    const player = this.currentPlayer;
    this.board[row][col] = player;
    const peg = { row, col, player };
    this.pegs.push(peg);
    this.moveCount++;

    // Update Zobrist hash (O(1) XOR)
    this.zKey ^= this.zTable[row][col][playerIndex(player)];

    // Update DSU - take snapshot before changes for rollback
    const dsu = player === 'red' ? this.redDSU : this.blackDSU;
    const dsuSnap = dsu.snapshot();
    const i = cellIndex(row, col, this.boardSize);
    dsu.activate(i, row, col);

    this.moveHistory.push({
      type: 'peg',
      peg,
      bridges: [],
      dsuSnap, // Store snapshot for undo
    });

    const newBridges = this.createBridges(row, col);
    this.moveHistory[this.moveHistory.length - 1].bridges = newBridges;

    // Union DSU components for each new bridge
    for (const bridge of newBridges) {
      const j = cellIndex(bridge.to.row, bridge.to.col, this.boardSize);
      dsu.union(i, j);
    }

    if (this.checkWin(player)) {
      this.gameOver = true;
      this.winner = player;
      return true;
    }

    this.currentPlayer = player === 'red' ? 'black' : 'red';
    return true;
  }

  /** Build bridges from the newly placed peg at (row,col). */
  createBridges(row, col) {
    const newBridges = [];
    const player = this.currentPlayer;

    // Knight offsets (TwixT bridge geometry)
    const KNIGHT_MOVES = [
      [-2, -1],
      [-2, 1],
      [-1, -2],
      [-1, 2],
      [1, -2],
      [1, 2],
      [2, -1],
      [2, 1],
    ];

    for (const [dr, dc] of KNIGHT_MOVES) {
      const r2 = row + dr;
      const c2 = col + dc;

      // In-bounds and same player's peg at the other end?
      if (r2 < 0 || r2 >= this.boardSize || c2 < 0 || c2 >= this.boardSize)
        continue;
      if (this.board[r2][c2] !== player) continue;

      // Already have this exact bridge (either direction)?
      const exists = this.bridges.some(
        (b) =>
          (b.from.row === row &&
            b.from.col === col &&
            b.to.row === r2 &&
            b.to.col === c2) ||
          (b.from.row === r2 &&
            b.from.col === c2 &&
            b.to.row === row &&
            b.to.col === col)
      );
      if (exists) continue;

      // Forbid crossings with ANY existing bridge (own or opponent)
      if (this.bridgesCross(row, col, r2, c2)) continue;

      // Create and record the new bridge
      const bridge = { from: { row, col }, to: { row: r2, col: c2 }, player };
      this.bridges.push(bridge);
      newBridges.push(bridge);
    }

    return newBridges;
  }

  /**
   * Orientation test for three points.
   * Returns: 1 if CCW, -1 if CW, 0 if collinear
   */
  static orient(ax, ay, bx, by, cx, cy) {
    const v = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    return v > 0 ? 1 : v < 0 ? -1 : 0;
  }

  /**
   * Fast proper intersection test for TwixT knight-edges.
   *
   * For knight-move segments (delta ±1,±2 or ±2,±1):
   * - Collinear overlaps cannot happen between distinct knight-edges
   * - No interior lattice points exist (gcd(1,2)=1)
   * - Only need pure orientation test
   *
   * Returns true if segments properly cross.
   */
  static properIntersectKnight(x1, y1, x2, y2, x3, y3, x4, y4) {
    const o1 = TwixTGame.orient(x1, y1, x2, y2, x3, y3);
    const o2 = TwixTGame.orient(x1, y1, x2, y2, x4, y4);
    if (o1 === 0 || o2 === 0 || o1 === o2) return false;

    const o3 = TwixTGame.orient(x3, y3, x4, y4, x1, y1);
    const o4 = TwixTGame.orient(x3, y3, x4, y4, x2, y2);
    if (o3 === 0 || o4 === 0 || o3 === o4) return false;

    return true;
  }

  /**
   * Return true if candidate (r1,c1)-(r2,c2) would cross any existing bridge.
   *
   * Optimized for TwixT:
   * - Bbox rejection skips most bridges without geometry (60-80% skip rate)
   * - Simplified intersection test (no collinear cases for knight edges)
   * - Shared endpoints are legal (not a crossing)
   *
   * Uses x=col, y=row convention.
   */
  bridgesCross(r1, c1, r2, c2) {
    const bridges = this.bridges;
    if (bridges.length === 0) return false; // Early exit

    // Candidate endpoints (x=col, y=row)
    const a1x = c1,
      a1y = r1;
    const a2x = c2,
      a2y = r2;

    // Candidate bbox
    const a_minx = a1x < a2x ? a1x : a2x;
    const a_maxx = a1x < a2x ? a2x : a1x;
    const a_miny = a1y < a2y ? a1y : a2y;
    const a_maxy = a1y < a2y ? a2y : a1y;

    for (const br of bridges) {
      const bc1 = br.from.col,
        br1 = br.from.row;
      const bc2 = br.to.col,
        br2 = br.to.row;

      // Sharing an endpoint is legal, not a crossing
      if (
        (r1 === br1 && c1 === bc1) ||
        (r1 === br2 && c1 === bc2) ||
        (r2 === br1 && c2 === bc1) ||
        (r2 === br2 && c2 === bc2)
      ) {
        continue;
      }

      // Bbox rejection (cheap - skips most bridges)
      const b_minx = bc1 < bc2 ? bc1 : bc2;
      const b_maxx = bc1 < bc2 ? bc2 : bc1;
      if (b_maxx < a_minx || b_minx > a_maxx) continue;

      const b_miny = br1 < br2 ? br1 : br2;
      const b_maxy = br1 < br2 ? br2 : br1;
      if (b_maxy < a_miny || b_miny > a_maxy) continue;

      // Proper intersection only (fast for knight edges)
      if (TwixTGame.properIntersectKnight(a1x, a1y, a2x, a2y, bc1, br1, bc2, br2)) {
        return true;
      }
    }
    return false;
  }

  /** Robust segment–segment intersection; endpoint touching is NOT a crossing. */
  lineSegmentsIntersect(x1, y1, x2, y2, x3, y3, x4, y4) {
    function orient(ax, ay, bx, by, cx, cy) {
      const abx = bx - ax,
        aby = by - ay;
      const acx = cx - ax,
        acy = cy - ay;
      const v = abx * acy - aby * acx;
      return v > 0 ? 1 : v < 0 ? -1 : 0;
    }
    function onSegment(ax, ay, bx, by, cx, cy) {
      return (
        Math.min(ax, bx) <= cx &&
        cx <= Math.max(ax, bx) &&
        Math.min(ay, by) <= cy &&
        cy <= Math.max(ay, by)
      );
    }

    const o1 = orient(x1, y1, x2, y2, x3, y3);
    const o2 = orient(x1, y1, x2, y2, x4, y4);
    const o3 = orient(x3, y3, x4, y4, x1, y1);
    const o4 = orient(x3, y3, x4, y4, x2, y2);

    // Proper intersection (exclude endpoint-only touching)
    if (o1 !== o2 && o3 !== o4) {
      const endpointTouch =
        (o1 === 0 && onSegment(x1, y1, x2, y2, x3, y3)) ||
        (o2 === 0 && onSegment(x1, y1, x2, y2, x4, y4)) ||
        (o3 === 0 && onSegment(x3, y3, x4, y4, x1, y1)) ||
        (o4 === 0 && onSegment(x3, y3, x4, y4, x2, y2));
      return !endpointTouch;
    }

    // Collinear overlaps beyond shared endpoints count as crossing
    if (o1 === 0 && onSegment(x1, y1, x2, y2, x3, y3)) {
      const shares = (x3 === x1 && y3 === y1) || (x3 === x2 && y3 === y2);
      return !shares;
    }
    if (o2 === 0 && onSegment(x1, y1, x2, y2, x4, y4)) {
      const shares = (x4 === x1 && y4 === y1) || (x4 === x2 && y4 === y2);
      return !shares;
    }
    if (o3 === 0 && onSegment(x3, y3, x4, y4, x1, y1)) {
      const shares = (x1 === x3 && y1 === y3) || (x1 === x4 && y1 === y4);
      return !shares;
    }
    if (o4 === 0 && onSegment(x3, y3, x4, y4, x2, y2)) {
      const shares = (x2 === x3 && y2 === y3) || (x2 === x4 && y2 === y4);
      return !shares;
    }

    return false;
  }

  checkWin(player) {
    if (player === 'red') {
      // Red wins if connected path from row 0 to row 23 via red bridges
      for (let startCol = 0; startCol < this.boardSize; startCol++) {
        if (this.board[0][startCol] === 'red') {
          const component = this.getConnectedComponent(0, startCol, player);
          for (const key of component) {
            const [row] = key.split(',').map(Number);
            if (row === this.boardSize - 1) return true;
          }
        }
      }
    } else {
      // Black wins if connected path from col 0 to col 23 via black bridges
      for (let startRow = 0; startRow < this.boardSize; startRow++) {
        if (this.board[startRow][0] === 'black') {
          const component = this.getConnectedComponent(startRow, 0, player);
          for (const key of component) {
            const [, col] = key.split(',').map(Number);
            if (col === this.boardSize - 1) return true;
          }
        }
      }
    }
    return false;
  }

  getConnectedComponent(startRow, startCol, player) {
    const visited = new Set();
    const queue = [[startRow, startCol]];
    const component = new Set();

    while (queue.length > 0) {
      const [row, col] = queue.shift();
      const key = `${row},${col}`;
      if (visited.has(key)) continue;
      if (this.board[row][col] !== player) continue;

      visited.add(key);
      component.add(key);

      // Explore neighbors through same-player bridges
      for (const bridge of this.bridges) {
        if (bridge.player !== player) continue;

        let nr, nc;
        if (bridge.from.row === row && bridge.from.col === col) {
          nr = bridge.to.row;
          nc = bridge.to.col;
        } else if (bridge.to.row === row && bridge.to.col === col) {
          nr = bridge.from.row;
          nc = bridge.from.col;
        } else {
          continue;
        }

        const nkey = `${nr},${nc}`;
        if (!visited.has(nkey)) queue.push([nr, nc]);
      }
    }

    return component;
  }

  undo() {
    if (this.moveHistory.length === 0) return false;

    const lastMove = this.moveHistory.pop();
    const { row, col, player } = lastMove.peg;

    this.board[row][col] = null;
    this.pegs.pop();
    this.moveCount--;

    // Reverse Zobrist hash (XOR same value undoes it)
    this.zKey ^= this.zTable[row][col][playerIndex(player)];

    // Rollback DSU to snapshot
    const dsu = player === 'red' ? this.redDSU : this.blackDSU;
    if (lastMove.dsuSnap !== undefined) {
      dsu.rollback(lastMove.dsuSnap);
    }

    for (const bridge of lastMove.bridges) {
      const idx = this.bridges.findIndex(
        (b) =>
          b.from.row === bridge.from.row &&
          b.from.col === bridge.from.col &&
          b.to.row === bridge.to.row &&
          b.to.col === bridge.to.col
      );
      if (idx !== -1) this.bridges.splice(idx, 1);
    }

    this.gameOver = false;
    this.winner = null;
    this.currentPlayer = player;
    return true;
  }

  reset() {
    this.board = Array(this.boardSize)
      .fill(null)
      .map(() => Array(this.boardSize).fill(null));
    this.currentPlayer = 'red';
    this.pegs = [];
    this.bridges = [];
    this.moveHistory = [];
    this.gameOver = false;
    this.winner = null;
    this.moveCount = 0;
    this.startingPlayer = 'red';
    this.zKey = 0n;

    // Reset DSUs
    this.redDSU.reset();
    this.blackDSU.reset();
  }

  // AI Configuration
  setGameMode(isAI, difficulty = 'medium') {
    this.isAIGame = isAI;
    this.aiDifficulty = difficulty;
    // AI will always play the opposite colour from the human
    this.aiPlayer = isAI ? 'black' : null;
  }

  // Get all valid moves for current player
  getValidMoves() {
    const moves = [];
    for (let row = 0; row < this.boardSize; row++) {
      for (let col = 0; col < this.boardSize; col++) {
        if (this.isValidPegPlacement(row, col)) {
          moves.push({ row, col });
        }
      }
    }
    return moves;
  }

  /**
   * Check if a cell is empty.
   * @param {number} row
   * @param {number} col
   * @returns {boolean}
   */
  isEmpty(row, col) {
    return this.board[row][col] === null;
  }

  /**
   * Get the best (largest) component root for a player. O(1).
   * @param {string} player - "red" or "black"
   * @returns {number} Root index, or -1 if no components
   */
  getBestComponentRoot(player) {
    const dsu = player === 'red' ? this.redDSU : this.blackDSU;
    return dsu.getBestRoot();
  }

  /**
   * Get all pegs for a player (array reference, don't modify).
   * @param {string} player - "red" or "black"
   * @returns {Array} Array of {row, col, player} objects
   */
  getPlayerPegs(player) {
    // Return filtered view - could be optimized with separate arrays if needed
    return this.pegs.filter((p) => p.player === player);
  }

  /**
   * Get DSU-based component metrics for a player's best component. O(1).
   * @param {string} player - "red" or "black"
   * @param {number} [root] - Optional specific root (defaults to bestRoot)
   * @returns {Object} Metrics for the component
   */
  getDSUMetrics(player, root) {
    const dsu = player === 'red' ? this.redDSU : this.blackDSU;
    const r = root !== undefined ? root : dsu.getBestRoot();

    if (r < 0) {
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
        maxRowSpan: 0,
        maxColSpan: 0,
        touchesTop: false,
        touchesBottom: false,
        touchesLeft: false,
        touchesRight: false,
      };
    }

    const m = dsu.rootMetrics(r);

    // Map to componentMetrics-compatible field names
    return {
      ...m,
      maxRowSpan: m.spanR,
      maxColSpan: m.spanC,
      touchesTop: player === 'red' ? m.touchA : false,
      touchesBottom: player === 'red' ? m.touchB : false,
      touchesLeft: player === 'black' ? m.touchA : false,
      touchesRight: player === 'black' ? m.touchB : false,
    };
  }

  /**
   * Get DSU metrics for a specific cell (after placing a peg there).
   * Use this in movePriority after placePeg() to get component state.
   * @param {number} row
   * @param {number} col
   * @param {string} player
   * @returns {Object} Metrics for the component containing (row, col)
   */
  getDSUMetricsForCell(row, col, player) {
    const dsu = player === 'red' ? this.redDSU : this.blackDSU;
    const i = cellIndex(row, col, this.boardSize);

    if (!dsu.isActive(i)) {
      return null;
    }

    const root = dsu.find(i);
    const m = dsu.rootMetrics(root);

    return {
      ...m,
      maxRowSpan: m.spanR,
      maxColSpan: m.spanC,
      touchesTop: player === 'red' ? m.touchA : false,
      touchesBottom: player === 'red' ? m.touchB : false,
      touchesLeft: player === 'black' ? m.touchA : false,
      touchesRight: player === 'black' ? m.touchB : false,
    };
  }
}
