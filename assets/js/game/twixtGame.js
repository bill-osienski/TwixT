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

    this.board[row][col] = this.currentPlayer;
    const peg = { row, col, player: this.currentPlayer };
    this.pegs.push(peg);
    this.moveCount++;

    this.moveHistory.push({
      type: 'peg',
      peg,
      bridges: []
    });

    const newBridges = this.createBridges(row, col);
    this.moveHistory[this.moveHistory.length - 1].bridges = newBridges;

    if (this.checkWin(this.currentPlayer)) {
      this.gameOver = true;
      this.winner = this.currentPlayer;
      return true;
    }

    this.currentPlayer = this.currentPlayer === 'red' ? 'black' : 'red';
    return true;
  }

    /** Build bridges from the newly placed peg at (row,col). */
    createBridges(row, col) {
      const newBridges = [];
      const player = this.currentPlayer;

      // Knight offsets (TwixT bridge geometry)
      const KNIGHT_MOVES = [
        [-2, -1], [-2, 1], [-1, -2], [-1, 2],
        [ 1, -2], [ 1, 2], [ 2, -1], [ 2, 1]
      ];

      for (const [dr, dc] of KNIGHT_MOVES) {
        const r2 = row + dr;
        const c2 = col + dc;

        // In-bounds and same player's peg at the other end?
        if (r2 < 0 || r2 >= this.boardSize || c2 < 0 || c2 >= this.boardSize) continue;
        if (this.board[r2][c2] !== player) continue;

        // Already have this exact bridge (either direction)?
        const exists = this.bridges.some(b =>
          (b.from.row === row && b.from.col === col && b.to.row === r2 && b.to.col === c2) ||
          (b.from.row === r2  && b.from.col === c2  && b.to.row === row && b.to.col === col)
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

    /** Return true if candidate (r1,c1)-(r2,c2) would cross any existing bridge. */
    bridgesCross(r1, c1, r2, c2) {
      const a1x = c1, a1y = r1;
      const a2x = c2, a2y = r2;

      for (const br of this.bridges) {
        // Sharing an endpoint is legal, not a crossing
        const sharesEndpoint =
          (a1x === br.from.col && a1y === br.from.row) ||
          (a1x === br.to.col   && a1y === br.to.row)   ||
          (a2x === br.from.col && a2y === br.from.row) ||
          (a2x === br.to.col   && a2y === br.to.row);
        if (sharesEndpoint) continue;

        if (this.lineSegmentsIntersect(
          a1x, a1y, a2x, a2y,
          br.from.col, br.from.row, br.to.col, br.to.row
        )) {
          return true;
        }
      }
      return false;
    }

    /** Robust segment–segment intersection; endpoint touching is NOT a crossing. */
    lineSegmentsIntersect(x1, y1, x2, y2, x3, y3, x4, y4) {
      function orient(ax, ay, bx, by, cx, cy) {
        const abx = bx - ax, aby = by - ay;
        const acx = cx - ax, acy = cy - ay;
        const v = abx * acy - aby * acx;
        return v > 0 ? 1 : v < 0 ? -1 : 0;
      }
      function onSegment(ax, ay, bx, by, cx, cy) {
        return Math.min(ax, bx) <= cx && cx <= Math.max(ax, bx) &&
               Math.min(ay, by) <= cy && cy <= Math.max(ay, by);
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

    this.board[lastMove.peg.row][lastMove.peg.col] = null;
    this.pegs.pop();
    this.moveCount--;

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
    this.currentPlayer = lastMove.peg.player;
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
  }

  // AI Configuration
  setGameMode(isAI, difficulty = 'medium') {
    this.isAIGame = isAI;
    this.aiDifficulty = difficulty;
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
}