/**
 * TwixT game state for AlphaZero inference.
 *
 * This implementation MUST exactly match the Python version in
 * scripts/GPU/alphazero/game/twixt_state.py for parity between
 * training (Python) and inference (Node.js).
 *
 * Constants:
 *   BOARD_SIZE: 24x24 physical tensor dimension
 *   MAX_PLIES: 600 (forced draw after this many moves)
 *
 * Curriculum:
 *   activeSize (<= boardSize) defines the playable region [0, activeSize) x
 *   [0, activeSize). Regions outside activeSize are zero-padded in the tensor.
 *   Mirrors Python's `TwixtState.active_size`.
 *
 * Draw semantics:
 *   - (a) No legal moves (board fills), OR
 *   - (b) ply >= MAX_PLIES (forced draw, even if moves exist)
 */

// Game constants (MUST match Python)
export const BOARD_SIZE = 24;
export const MAX_PLIES = 600;

// Knight-move offsets for TwixT bridges
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

// Direction channel mapping for tensor encoding
// Maps "dr,dc" string to channel offset (0-7)
// Channels 2-9 for red links, 10-17 for black links
// Direction names: NNE, ENE, ESE, SSE, SSW, WSW, WNW, NNW
const DIRECTION_TO_CHANNEL = {
  '2,1': 0, // NNE: +2 row, +1 col
  '1,2': 1, // ENE: +1 row, +2 col
  '-1,2': 2, // ESE: -1 row, +2 col
  '-2,1': 3, // SSE: -2 row, +1 col
  '-2,-1': 4, // SSW: -2 row, -1 col
  '-1,-2': 5, // WSW: -1 row, -2 col
  '1,-2': 6, // WNW: +1 row, -2 col
  '2,-1': 7, // NNW: +2 row, -1 col
};

// Channel indices for tensor encoding (MUST match Python)
const CHANNEL_RED_PEGS = 0;
const CHANNEL_BLACK_PEGS = 1;
const CHANNEL_RED_LINKS_START = 2; // 2-9 (8 directions)
const CHANNEL_BLACK_LINKS_START = 10; // 10-17 (8 directions)
const CHANNEL_CURRENT_PLAYER = 18;
const CHANNEL_RED_TOP_DIST = 19;
const CHANNEL_RED_BOTTOM_DIST = 20;
const CHANNEL_BLACK_LEFT_DIST = 21;
const CHANNEL_BLACK_RIGHT_DIST = 22;
const CHANNEL_MOVE_NUMBER = 23;
// Phase 2 connectivity channels (see spec 2026-04-19)
const CHANNEL_RED_CONN_TOP = 24;
const CHANNEL_RED_CONN_BOTTOM = 25;
const CHANNEL_RED_CONN_BOTH = 26;
const CHANNEL_BLACK_CONN_LEFT = 27;
const CHANNEL_BLACK_CONN_RIGHT = 28;
const CHANNEL_BLACK_CONN_BOTH = 29;
export const NUM_CHANNELS = 30;

/**
 * Return bridge endpoints in canonical order (smaller pos first).
 * Position comparison: (r1,c1) < (r2,c2) iff r1 < r2 || (r1 === r2 && c1 < c2)
 */
function canonicalBridge(p1, p2) {
  const [r1, c1] = p1;
  const [r2, c2] = p2;
  if (r1 < r2 || (r1 === r2 && c1 < c2)) {
    return [p1, p2];
  }
  return [p2, p1];
}

/**
 * Bridge key for Set/Map operations.
 */
function bridgeKey(bridge) {
  const [[r1, c1], [r2, c2]] = bridge;
  return `${r1},${c1}-${r2},${c2}`;
}

/**
 * Orientation test for three points.
 * Returns: 1 if CCW, -1 if CW, 0 if collinear
 */
function orient(ax, ay, bx, by, cx, cy) {
  const v = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  if (v > 0) return 1;
  if (v < 0) return -1;
  return 0;
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
function properIntersectKnight(x1, y1, x2, y2, x3, y3, x4, y4) {
  const o1 = orient(x1, y1, x2, y2, x3, y3);
  const o2 = orient(x1, y1, x2, y2, x4, y4);
  if (o1 === 0 || o2 === 0 || o1 === o2) return false;

  const o3 = orient(x3, y3, x4, y4, x1, y1);
  const o4 = orient(x3, y3, x4, y4, x2, y2);
  if (o3 === 0 || o4 === 0 || o3 === o4) return false;

  return true;
}

/**
 * Immutable-style TwixT game state.
 *
 * Methods return new state objects rather than mutating in place.
 * This is important for MCTS tree search.
 */
export class TwixtState {
  /**
   * @param {Object} opts
   * @param {number} [opts.boardSize=24] - Physical tensor dimension (always 24)
   * @param {number} [opts.activeSize] - Curriculum playable region (<= boardSize).
   *                                     Defaults to boardSize.
   * @param {string} [opts.toMove="red"]
   * @param {Map<string, string>} [opts.pegs] - Map of "r,c" -> player
   * @param {Set<string>} [opts.bridges] - Set of bridge keys
   * @param {number} [opts.ply=0]
   */
  constructor({
    boardSize = BOARD_SIZE,
    activeSize,
    toMove = 'red',
    pegs = new Map(),
    bridges = new Set(),
    ply = 0,
  } = {}) {
    this.boardSize = boardSize;
    this.activeSize = activeSize ?? boardSize;
    // Validate activeSize (mirrors Python's __post_init__)
    if (this.activeSize < 1 || this.activeSize > this.boardSize) {
      throw new Error(
        `activeSize must be in [1, ${this.boardSize}], got ${this.activeSize}`
      );
    }
    this.toMove = toMove;
    this.pegs = pegs;
    this.bridges = bridges;
    this.ply = ply;
    // Derived, lazily-built adjacency cache backing _getConnectedComponent.
    // Never copied (copy() runs the constructor -> null -> rebuilt lazily).
    // Mirrors Python TwixtState._adj.
    this._adj = null;
  }

  /**
   * Create a deep copy of the state.
   */
  copy() {
    return new TwixtState({
      boardSize: this.boardSize,
      activeSize: this.activeSize,
      toMove: this.toMove,
      pegs: new Map(this.pegs),
      bridges: new Set(this.bridges),
      ply: this.ply,
    });
  }

  /**
   * Get peg at position, or null if empty.
   * @param {number} row
   * @param {number} col
   * @returns {string|null}
   */
  getPeg(row, col) {
    return this.pegs.get(`${row},${col}`) || null;
  }

  /**
   * Check if a peg placement is valid for current player.
   *
   * Rules (with curriculum activeSize):
   * 1. Cell must be within [0, activeSize) x [0, activeSize)
   * 2. Cell must be empty
   * 3. Corners of ACTIVE region are forbidden
   * 4. Red cannot place on left/right edges of ACTIVE region
   * 5. Black cannot place on top/bottom edges of ACTIVE region
   */
  isValidPlacement(row, col) {
    const active = this.activeSize;

    // Out of active bounds
    if (row < 0 || row >= active || col < 0 || col >= active) {
      return false;
    }

    // Occupied
    if (this.pegs.has(`${row},${col}`)) {
      return false;
    }

    // Corners of ACTIVE region forbidden
    if (
      (row === 0 || row === active - 1) &&
      (col === 0 || col === active - 1)
    ) {
      return false;
    }

    // Edge restrictions by player (using activeSize)
    if (this.toMove === 'red') {
      // Red connects top<->bottom; cannot place on left/right edges
      if (col === 0 || col === active - 1) {
        return false;
      }
    } else {
      // Black connects left<->right; cannot place on top/bottom edges
      if (row === 0 || row === active - 1) {
        return false;
      }
    }

    return true;
  }

  /**
   * Return all valid move positions for current player.
   * Only considers positions within [0, activeSize) x [0, activeSize).
   * @returns {Array<[number, number]>} List of [row, col] tuples, sorted.
   */
  legalMoves() {
    const moves = [];
    const active = this.activeSize;
    for (let row = 0; row < active; row++) {
      for (let col = 0; col < active; col++) {
        if (this.isValidPlacement(row, col)) {
          moves.push([row, col]);
        }
      }
    }
    return moves;
  }

  /**
   * Check if a candidate bridge crosses any existing bridge.
   * Uses bbox rejection for efficiency, then proper intersection test.
   * Shared endpoints are legal (not a crossing).
   * Uses x=col, y=row convention.
   */
  _crossesExistingBridge(r1, c1, r2, c2) {
    if (this.bridges.size === 0) {
      return false;
    }

    // Candidate endpoints (x=col, y=row)
    const a1x = c1,
      a1y = r1;
    const a2x = c2,
      a2y = r2;

    // Candidate bbox
    const aMinX = Math.min(a1x, a2x);
    const aMaxX = Math.max(a1x, a2x);
    const aMinY = Math.min(a1y, a2y);
    const aMaxY = Math.max(a1y, a2y);

    for (const bKey of this.bridges) {
      // Parse bridge key: "r1,c1-r2,c2"
      const [p1Str, p2Str] = bKey.split('-');
      const [br1, bc1] = p1Str.split(',').map(Number);
      const [br2, bc2] = p2Str.split(',').map(Number);

      // Shared endpoint check (legal, not a crossing)
      if (
        (r1 === br1 && c1 === bc1) ||
        (r1 === br2 && c1 === bc2) ||
        (r2 === br1 && c2 === bc1) ||
        (r2 === br2 && c2 === bc2)
      ) {
        continue;
      }

      // Bridge endpoints in x,y convention
      const b1x = bc1,
        b1y = br1;
      const b2x = bc2,
        b2y = br2;

      // Bbox rejection
      const bMinX = Math.min(b1x, b2x);
      const bMaxX = Math.max(b1x, b2x);
      if (bMaxX < aMinX || bMinX > aMaxX) continue;

      const bMinY = Math.min(b1y, b2y);
      const bMaxY = Math.max(b1y, b2y);
      if (bMaxY < aMinY || bMinY > aMaxY) continue;

      // Proper intersection test
      if (properIntersectKnight(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Find all new bridges created by placing a peg at (row, col).
   * @param {TwixtState} newState - State with the new peg already added
   * @param {number} row
   * @param {number} col
   * @param {string} player
   * @returns {Array<string>} Bridge keys
   */
  _findNewBridges(newState, row, col, player) {
    const newBridges = [];

    const active = this.activeSize;
    for (const [dr, dc] of KNIGHT_MOVES) {
      const r2 = row + dr;
      const c2 = col + dc;

      // In ACTIVE bounds? (not boardSize)
      if (r2 < 0 || r2 >= active || c2 < 0 || c2 >= active) {
        continue;
      }

      // Same player's peg at other end?
      if (newState.getPeg(r2, c2) !== player) {
        continue;
      }

      // Bridge already exists?
      const bridge = canonicalBridge([row, col], [r2, c2]);
      const bKey = bridgeKey(bridge);
      if (newState.bridges.has(bKey)) {
        continue;
      }

      // Would cross an existing bridge? (check against original state)
      if (this._crossesExistingBridge(row, col, r2, c2)) {
        continue;
      }

      newBridges.push(bKey);
    }

    return newBridges;
  }

  /**
   * Apply a move and return a new state.
   * Does NOT validate the move - caller must ensure move is legal.
   * @param {[number, number]} move - [row, col]
   * @returns {TwixtState}
   */
  applyMove(move) {
    const [row, col] = move;
    const player = this.toMove;

    // Create new state
    const newState = this.copy();
    newState.pegs.set(`${row},${col}`, player);
    newState.ply += 1;

    // Find and add new bridges
    const newBridges = this._findNewBridges(newState, row, col, player);
    for (const bKey of newBridges) {
      newState.bridges.add(bKey);
    }

    // Switch player
    newState.toMove = player === 'red' ? 'black' : 'red';

    return newState;
  }

  /**
   * Build a "r,c" -> Array<[r,c]> adjacency map from this.bridges.
   * One map per state; per-player correctness is enforced by the pop-time
   * color check in _getConnectedComponent. Mirrors Python _build_adjacency.
   */
  _buildAdjacency() {
    const adj = new Map();
    for (const bKey of this.bridges) {
      const [p1Str, p2Str] = bKey.split('-');
      const [r1, c1] = p1Str.split(',').map(Number);
      const [r2, c2] = p2Str.split(',').map(Number);
      const k1 = `${r1},${c1}`;
      const k2 = `${r2},${c2}`;
      if (!adj.has(k1)) adj.set(k1, []);
      if (!adj.has(k2)) adj.set(k2, []);
      adj.get(k1).push([r2, c2]);
      adj.get(k2).push([r1, c1]);
    }
    return adj;
  }

  /**
   * Drop the cached adjacency map. Call after any in-place mutation of
   * this.bridges / this.pegs on an existing state. Mirrors Python
   * _invalidate_adj. Production mutates only via applyMove (fresh copy).
   */
  _invalidateAdj() {
    this._adj = null;
  }

  /**
   * Get all positions connected to start via same-player bridges (BFS).
   * @param {[number, number]} start
   * @param {string} player
   * @returns {Set<string>} Set of "r,c" keys
   */
  _getConnectedComponent(start, player) {
    if (this._adj === null) {
      this._adj = this._buildAdjacency();
    }
    const adj = this._adj;

    const visited = new Set();
    const component = new Set();
    const queue = [start];

    while (queue.length > 0) {
      const [row, col] = queue.shift();
      const key = `${row},${col}`;

      if (visited.has(key)) continue;
      if (this.getPeg(row, col) !== player) continue;

      visited.add(key);
      component.add(key);

      const neighbors = adj.get(key);
      if (neighbors === undefined) continue;
      for (const [nr, nc] of neighbors) {
        const nKey = `${nr},${nc}`;
        if (!visited.has(nKey)) {
          queue.push([nr, nc]);
        }
      }
    }

    return component;
  }

  /**
   * Return connectivity masks (touchesG1, touchesG2, touchesBoth) for `player`.
   *
   * Each mask is a Float32Array of length activeSize*activeSize with 1.0 on
   * cells where `player` has a peg whose bridge-connected component touches
   * the named goal edge.
   *
   * Uses the EXACT same `_getConnectedComponent` BFS that `_checkWin` uses,
   * mirroring Python's `TwixtState.connectivity_masks`, so feature-side and
   * game-logic-side connectivity can never drift.
   *
   * For red: goal1 = row 0 (top), goal2 = row activeSize-1 (bottom).
   * For black: goal1 = col 0 (left), goal2 = col activeSize-1 (right).
   *
   * @param {string} player - "red" or "black"
   * @returns {[Float32Array, Float32Array, Float32Array]}
   *          [touchesG1, touchesG2, touchesBoth], each flat [row*active + col]
   */
  _connectivityMasks(player) {
    const active = this.activeSize;
    const n = active * active;
    const mG1 = new Float32Array(n);
    const mG2 = new Float32Array(n);
    const mBoth = new Float32Array(n);

    // Collect player's pegs
    const playerPegs = [];
    for (const [key, col] of this.pegs) {
      if (col === player) {
        const [r, c] = key.split(',').map(Number);
        playerPegs.push([r, c, key]);
      }
    }
    if (playerPegs.length === 0) {
      return [mG1, mG2, mBoth];
    }

    // Goal-edge predicates per player
    let onG1, onG2;
    if (player === 'red') {
      onG1 = (r, _c) => r === 0;
      onG2 = (r, _c) => r === active - 1;
    } else {
      // black
      onG1 = (r, c) => c === 0;
      onG2 = (r, c) => c === active - 1;
    }

    // Bucket pegs into components via existing BFS. Pegs already seen by a
    // prior BFS are tagged so we don't recompute.
    const seen = new Set();
    const components = [];
    for (const [r, c, key] of playerPegs) {
      if (seen.has(key)) continue;
      const comp = this._getConnectedComponent([r, c], player);
      components.push(comp);
      for (const k of comp) {
        seen.add(k);
      }
    }

    // Per component: does it touch goal1? goal2? Then mark all its pegs.
    for (const comp of components) {
      let touchesG1 = false;
      let touchesG2 = false;
      for (const key of comp) {
        const [r, c] = key.split(',').map(Number);
        if (onG1(r, c)) touchesG1 = true;
        if (onG2(r, c)) touchesG2 = true;
      }
      for (const key of comp) {
        const [r, c] = key.split(',').map(Number);
        const idx = r * active + c;
        if (touchesG1) mG1[idx] = 1.0;
        if (touchesG2) mG2[idx] = 1.0;
        if (touchesG1 && touchesG2) mBoth[idx] = 1.0;
      }
    }

    return [mG1, mG2, mBoth];
  }

  /**
   * Check if player has won (connected their two edges).
   *
   * With curriculum:
   * - Red wins: path from row 0 to row (activeSize - 1)
   * - Black wins: path from col 0 to col (activeSize - 1)
   */
  _checkWin(player) {
    const active = this.activeSize;

    if (player === 'red') {
      // Check each peg on top edge (row 0) within active region
      for (let col = 0; col < active; col++) {
        if (this.getPeg(0, col) === 'red') {
          const component = this._getConnectedComponent([0, col], 'red');
          // Check if any peg in component is on bottom edge
          for (const key of component) {
            const [r] = key.split(',').map(Number);
            if (r === active - 1) {
              return true;
            }
          }
        }
      }
    } else {
      // Check each peg on left edge (col 0) within active region
      for (let row = 0; row < active; row++) {
        if (this.getPeg(row, 0) === 'black') {
          const component = this._getConnectedComponent([row, 0], 'black');
          // Check if any peg in component is on right edge
          for (const key of component) {
            const [, c] = key.split(',').map(Number);
            if (c === active - 1) {
              return true;
            }
          }
        }
      }
    }

    return false;
  }

  /**
   * Return the winner if any, else null.
   * @returns {string|null}
   */
  winner() {
    if (this._checkWin('red')) return 'red';
    if (this._checkWin('black')) return 'black';
    return null;
  }

  /**
   * Check if game is over (win or draw).
   * Terminal conditions:
   * 1. A player has won
   * 2. No legal moves remain (board full in playable area)
   * 3. ply >= MAX_PLIES (forced draw)
   */
  isTerminal() {
    // Check for winner
    if (this.winner() !== null) {
      return true;
    }

    // Forced draw after MAX_PLIES
    if (this.ply >= MAX_PLIES) {
      return true;
    }

    // No legal moves = draw
    if (this.legalMoves().length === 0) {
      return true;
    }

    return false;
  }

  /**
   * Return game result: "red", "black", or "draw".
   * Returns null if game is not terminal.
   * @returns {string|null}
   */
  gameResult() {
    if (!this.isTerminal()) {
      return null;
    }

    const w = this.winner();
    if (w !== null) {
      return w;
    }

    return 'draw';
  }

  /**
   * Serialize state to plain object (for JSON).
   */
  toDict() {
    const pegs = {};
    for (const [key, player] of this.pegs) {
      pegs[key] = player;
    }

    const bridges = [];
    const sortedBridges = Array.from(this.bridges).sort();
    for (const bKey of sortedBridges) {
      const [p1Str, p2Str] = bKey.split('-');
      const [r1, c1] = p1Str.split(',').map(Number);
      const [r2, c2] = p2Str.split(',').map(Number);
      bridges.push([
        [r1, c1],
        [r2, c2],
      ]);
    }

    return {
      board_size: this.boardSize,
      active_size: this.activeSize,
      to_move: this.toMove,
      pegs,
      bridges,
      ply: this.ply,
    };
  }

  /**
   * Deserialize state from plain object.
   * @param {Object} d
   * @returns {TwixtState}
   */
  static fromDict(d) {
    const pegs = new Map();
    for (const [key, player] of Object.entries(d.pegs)) {
      pegs.set(key, player);
    }

    const bridges = new Set();
    for (const [p1, p2] of d.bridges) {
      const bridge = canonicalBridge(p1, p2);
      bridges.add(bridgeKey(bridge));
    }

    const boardSize = d.board_size || BOARD_SIZE;
    return new TwixtState({
      boardSize,
      activeSize: d.active_size ?? boardSize,
      toMove: d.to_move,
      pegs,
      bridges,
      ply: d.ply || pegs.size,
    });
  }

  /**
   * Create state from sequence of moves.
   * @param {Array<[number, number]>} moves
   * @param {number} [activeSize] - Curriculum playable region (default = BOARD_SIZE)
   * @returns {TwixtState}
   */
  static fromMoves(moves, activeSize = BOARD_SIZE) {
    let state = new TwixtState({ activeSize });
    for (const move of moves) {
      state = state.applyMove(move);
    }
    return state;
  }

  /**
   * Convert state to 30-channel tensor for neural network input.
   *
   * @returns {Float32Array[]} Array of 30 Float32Arrays, each of size boardSize*boardSize
   *          Laid out as [channel][row * boardSize + col]
   *
   * CURRICULUM NOTE:
   *   - Playable region is [0, activeSize) x [0, activeSize)
   *   - Padded region (outside activeSize) is zeroed
   *   - Edge distance channels use activeSize as the boundary, not boardSize
   *
   * Channel layout (MUST match Python exactly):
   *   0: Red pegs (1 where red peg exists)
   *   1: Black pegs (1 where black peg exists)
   *   2-9: Red link directions (8 knight-move directions)
   *   10-17: Black link directions (8 knight-move directions)
   *   18: Current player indicator (1 if red to move, 0 if black)
   *   19: Red top edge distance (normalized 0-1, closer to row 0 = higher)
   *   20: Red bottom edge distance (normalized 0-1, closer to row activeSize-1 = higher)
   *   21: Black left edge distance (normalized 0-1, closer to col 0 = higher)
   *   22: Black right edge distance (normalized 0-1, closer to col activeSize-1 = higher)
   *   23: Move number / game phase (ply / MAX_PLIES, normalized 0-1)
   *   24: Red connected to top edge (1 on pegs whose component touches row 0)
   *   25: Red connected to bottom edge (1 on pegs whose component touches row activeSize-1)
   *   26: Red connected to both edges (1 on pegs whose component touches top AND bottom)
   *   27: Black connected to left edge (1 on pegs whose component touches col 0)
   *   28: Black connected to right edge (1 on pegs whose component touches col activeSize-1)
   *   29: Black connected to both edges (1 on pegs whose component touches left AND right)
   *
   * Link encoding: For each link, mark 1 at BOTH endpoints in the
   * appropriate direction channel. This makes links visible from either end.
   */
  toTensor() {
    const size = this.boardSize; // Physical tensor size (always 24)
    const active = this.activeSize; // Curriculum playable region
    const numCells = size * size;

    // Initialize all channels to zeros
    const tensor = [];
    for (let c = 0; c < NUM_CHANNELS; c++) {
      tensor.push(new Float32Array(numCells));
    }

    // Channel 0-1: Peg positions (only within active region can have pegs)
    for (const [key, player] of this.pegs) {
      const [r, c] = key.split(',').map(Number);
      const idx = r * size + c;
      if (player === 'red') {
        tensor[CHANNEL_RED_PEGS][idx] = 1.0;
      } else {
        tensor[CHANNEL_BLACK_PEGS][idx] = 1.0;
      }
    }

    // Channels 2-17: Link directions
    for (const bKey of this.bridges) {
      // Parse bridge key: "r1,c1-r2,c2"
      const [p1Str, p2Str] = bKey.split('-');
      const [r1, c1] = p1Str.split(',').map(Number);
      const [r2, c2] = p2Str.split(',').map(Number);

      // Determine player from peg color at first endpoint
      const player = this.getPeg(r1, c1);
      if (player === null) continue; // Should not happen with valid bridges

      // Calculate direction from endpoint 1 to endpoint 2
      const dr = r2 - r1;
      const dc = c2 - c1;
      const dirKey = `${dr},${dc}`;
      const dirOffset = DIRECTION_TO_CHANNEL[dirKey];
      if (dirOffset === undefined) continue; // Should not happen

      // Calculate reverse direction (endpoint 2 to endpoint 1)
      const revDirKey = `${-dr},${-dc}`;
      const revDirOffset = DIRECTION_TO_CHANNEL[revDirKey];

      // Determine base channel for this player's links
      const baseChannel =
        player === 'red' ? CHANNEL_RED_LINKS_START : CHANNEL_BLACK_LINKS_START;

      // Mark 1 at BOTH endpoints in appropriate direction channels
      const idx1 = r1 * size + c1;
      const idx2 = r2 * size + c2;
      tensor[baseChannel + dirOffset][idx1] = 1.0;
      if (revDirOffset !== undefined) {
        tensor[baseChannel + revDirOffset][idx2] = 1.0;
      }
    }

    // Channel 18: Current player indicator (fill only active region)
    if (this.toMove === 'red') {
      for (let r = 0; r < active; r++) {
        for (let c = 0; c < active; c++) {
          tensor[CHANNEL_CURRENT_PLAYER][r * size + c] = 1.0;
        }
      }
    }
    // else: already 0.0

    // Channels 19-22: Edge distances USING activeSize (curriculum semantics)
    // Goal edges are at 0 and activeSize-1, not 0 and boardSize-1
    const maxIdx = Math.max(1, active - 1); // Avoid div-by-zero for activeSize=1
    for (let r = 0; r < active; r++) {
      for (let c = 0; c < active; c++) {
        const idx = r * size + c;
        // Red top edge distance: closer to row 0 = higher value
        tensor[CHANNEL_RED_TOP_DIST][idx] = 1.0 - r / maxIdx;
        // Red bottom edge distance: closer to row (active-1) = higher value
        tensor[CHANNEL_RED_BOTTOM_DIST][idx] = r / maxIdx;
        // Black left edge distance: closer to col 0 = higher value
        tensor[CHANNEL_BLACK_LEFT_DIST][idx] = 1.0 - c / maxIdx;
        // Black right edge distance: closer to col (active-1) = higher value
        tensor[CHANNEL_BLACK_RIGHT_DIST][idx] = c / maxIdx;
      }
    }

    // Channel 23: Move number / game phase (fill only active region)
    const movePhase = this.ply / MAX_PLIES;
    for (let r = 0; r < active; r++) {
      for (let c = 0; c < active; c++) {
        tensor[CHANNEL_MOVE_NUMBER][r * size + c] = movePhase;
      }
    }

    // Channels 24-29: Connectivity masks (Phase 2 — matches Python
    // twixt_state.connectivity_masks; uses the same _getConnectedComponent
    // BFS as _checkWin for feature/game-logic parity).
    const [redTop, redBot, redBoth] = this._connectivityMasks('red');
    const [blackLeft, blackRight, blackBoth] = this._connectivityMasks('black');
    for (let r = 0; r < active; r++) {
      for (let c = 0; c < active; c++) {
        const tensorIdx = r * size + c;
        const maskIdx = r * active + c;
        tensor[CHANNEL_RED_CONN_TOP][tensorIdx] = redTop[maskIdx];
        tensor[CHANNEL_RED_CONN_BOTTOM][tensorIdx] = redBot[maskIdx];
        tensor[CHANNEL_RED_CONN_BOTH][tensorIdx] = redBoth[maskIdx];
        tensor[CHANNEL_BLACK_CONN_LEFT][tensorIdx] = blackLeft[maskIdx];
        tensor[CHANNEL_BLACK_CONN_RIGHT][tensorIdx] = blackRight[maskIdx];
        tensor[CHANNEL_BLACK_CONN_BOTH][tensorIdx] = blackBoth[maskIdx];
      }
    }

    // IMPORTANT: Regions outside activeSize are already zeros (from Float32Array).
    // This is intentional for consistent curriculum training.

    return tensor;
  }

  /**
   * Convert tensor to nested array format [channel][row][col] for JSON serialization.
   * @returns {number[][][]} 3D array of shape [24][24][24]
   */
  toTensorNested() {
    const flat = this.toTensor();
    const size = this.boardSize;
    const nested = [];

    for (let c = 0; c < NUM_CHANNELS; c++) {
      const channel = [];
      for (let r = 0; r < size; r++) {
        const row = [];
        for (let col = 0; col < size; col++) {
          row.push(flat[c][r * size + col]);
        }
        channel.push(row);
      }
      nested.push(channel);
    }

    return nested;
  }

  /**
   * Convert tensor to HWC format [row][col][channel] for ONNX inference.
   * This is the NHWC layout expected by inference.js.
   * @returns {number[][][]} 3D array of shape [24][24][24] in HWC order
   */
  toTensorHWC() {
    const flat = this.toTensor();
    const size = this.boardSize;
    const hwc = [];

    for (let r = 0; r < size; r++) {
      const row = [];
      for (let col = 0; col < size; col++) {
        const channels = [];
        for (let c = 0; c < NUM_CHANNELS; c++) {
          channels.push(flat[c][r * size + col]);
        }
        row.push(channels);
      }
      hwc.push(row);
    }

    return hwc;
  }
}

/**
 * Build a state tensor from a move sequence at a given curriculum activeSize.
 *
 * Mirrors Python's `TwixtState(active_size=N).apply_move(...) -> to_tensor()`
 * construction pattern. Intended for JS/Python tensor parity tests.
 *
 * @param {number} activeSize - Curriculum playable region (1..BOARD_SIZE)
 * @param {Array<[number, number]>} moves - Ordered [row, col] placements
 * @returns {Float32Array[]} Array of NUM_CHANNELS Float32Arrays, each of size
 *          boardSize*boardSize, laid out as [channel][row * boardSize + col]
 */
export function buildStateTensor(activeSize, moves) {
  let state = new TwixtState({ activeSize });
  for (const [r, c] of moves) {
    state = state.applyMove([r, c]);
  }
  return state.toTensor();
}
