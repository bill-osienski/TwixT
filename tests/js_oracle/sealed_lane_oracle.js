#!/usr/bin/env node
/**
 * JS Oracle for Sealed Lane Detection
 *
 * This script provides a command-line interface to the JS hasReachableGoalEdge
 * function, allowing Python tests to verify that Python's sealed lane detection
 * matches JS semantics exactly.
 *
 * Usage:
 *   node sealed_lane_oracle.js < input.json
 *   echo '{"boardSize":24,"pegs":[...],"bridges":[...],"player":"red","component":[...]}' | node sealed_lane_oracle.js
 *
 * Input JSON format:
 * {
 *   "boardSize": 24,
 *   "pegs": [{"row": 12, "col": 12, "player": "red"}, ...],
 *   "bridges": [{"r1": 12, "c1": 12, "r2": 10, "c2": 11}, ...],
 *   "player": "red",
 *   "component": [{"row": 12, "col": 12}, ...],
 *   "touchesTop": false,
 *   "touchesBottom": false,
 *   "touchesLeft": false,
 *   "touchesRight": false
 * }
 *
 * Output JSON format:
 * {
 *   "reachable": true|false,
 *   "error": null|"error message"
 * }
 *
 * This oracle is used by tests/test_js_oracle.py to verify Python/JS alignment.
 */

'use strict';

// Knight move offsets (same as in search.js)
const KNIGHT_OFFSETS = [
  [-2, -1], [-2, 1], [-1, -2], [-1, 2],
  [1, -2], [1, 2], [2, -1], [2, 1]
];

/**
 * Check if a coordinate is on the player's goal edge (not corners).
 * Matches search.js isGoalEdgeCoordinate exactly.
 */
function isGoalEdgeCoordinate(player, row, col, boardSize) {
  if (player === 'red') {
    if (row !== 0 && row !== boardSize - 1) {
      return false;
    }
    return col > 0 && col < boardSize - 1;
  }
  if (col !== 0 && col !== boardSize - 1) {
    return false;
  }
  return row > 0 && row < boardSize - 1;
}

/**
 * Check if a placement is legal for a player.
 * Matches search.js isLegalPlacementForPlayer exactly.
 */
function isLegalPlacementForPlayer(board, boardSize, player, row, col) {
  if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) {
    return false;
  }
  if (board[row][col] !== null) {
    return false;
  }
  const onTopOrBottom = row === 0 || row === boardSize - 1;
  const onLeftOrRight = col === 0 || col === boardSize - 1;
  if (onTopOrBottom && onLeftOrRight) {
    return false; // corners
  }
  if (player === 'red') {
    return !(col === 0 || col === boardSize - 1);
  }
  return !(row === 0 || row === boardSize - 1);
}

/**
 * Check if two knight-move segments cross.
 * Uses orientation test for proper intersection.
 */
function segmentsCross(r1, c1, r2, c2, r3, c3, r4, c4) {
  // Orientation of triplet (p, q, r)
  // 0 = collinear, 1 = clockwise, 2 = counter-clockwise
  function orientation(pr, pc, qr, qc, rr, rc) {
    const val = (qc - pc) * (rr - qr) - (qr - pr) * (rc - qc);
    if (val === 0) return 0;
    return val > 0 ? 1 : 2;
  }

  // Check if point (qr, qc) lies on segment (pr, pc)-(rr, rc)
  function onSegment(pr, pc, qr, qc, rr, rc) {
    return qr <= Math.max(pr, rr) && qr >= Math.min(pr, rr) &&
           qc <= Math.max(pc, rc) && qc >= Math.min(pc, rc);
  }

  const o1 = orientation(r1, c1, r2, c2, r3, c3);
  const o2 = orientation(r1, c1, r2, c2, r4, c4);
  const o3 = orientation(r3, c3, r4, c4, r1, c1);
  const o4 = orientation(r3, c3, r4, c4, r2, c2);

  // General case: proper intersection
  if (o1 !== o2 && o3 !== o4) {
    // Check that intersection is not at endpoints
    // Endpoint touching is allowed in TwixT
    const sharesEndpoint =
      (r1 === r3 && c1 === c3) || (r1 === r4 && c1 === c4) ||
      (r2 === r3 && c2 === c3) || (r2 === r4 && c2 === c4);
    return !sharesEndpoint;
  }

  // Collinear cases - check if segments overlap (not just touch)
  if (o1 === 0 && onSegment(r1, c1, r3, c3, r2, c2)) {
    // Check it's not just endpoint touching
    if (!((r3 === r1 && c3 === c1) || (r3 === r2 && c3 === c2))) {
      return true;
    }
  }
  if (o2 === 0 && onSegment(r1, c1, r4, c4, r2, c2)) {
    if (!((r4 === r1 && c4 === c1) || (r4 === r2 && c4 === c2))) {
      return true;
    }
  }
  if (o3 === 0 && onSegment(r3, c3, r1, c1, r4, c4)) {
    if (!((r1 === r3 && c1 === c3) || (r1 === r4 && c1 === c4))) {
      return true;
    }
  }
  if (o4 === 0 && onSegment(r3, c3, r2, c2, r4, c4)) {
    if (!((r2 === r3 && c2 === c3) || (r2 === r4 && c2 === c4))) {
      return true;
    }
  }

  return false;
}

/**
 * Check if a potential knight-move bridge crosses any existing bridge.
 */
function bridgesCross(bridges, r1, c1, r2, c2) {
  for (const bridge of bridges) {
    if (segmentsCross(r1, c1, r2, c2, bridge.r1, bridge.c1, bridge.r2, bridge.c2)) {
      return true;
    }
  }
  return false;
}

/**
 * BFS to check if player can reach their goal edge.
 * This is the core function that must match Python exactly.
 *
 * Matches search.js hasReachableGoalEdge semantics:
 * - Opponent pegs block
 * - ALL bridges (both colors) block crossings
 * - Goal edge requires isGoalEdgeCoordinate check
 * - Empty cells only expand if isLegalPlacementForPlayer
 */
function hasReachableGoalEdge(input) {
  const { boardSize, board, bridges, player, component,
          touchesTop, touchesBottom, touchesLeft, touchesRight } = input;

  if (!component || component.length === 0) {
    return false;
  }

  // Determine target edges
  const targetSet = new Set();
  if (player === 'red') {
    if (!touchesTop) targetSet.add(0);
    if (!touchesBottom) targetSet.add(boardSize - 1);
  } else {
    if (!touchesLeft) targetSet.add(0);
    if (!touchesRight) targetSet.add(boardSize - 1);
  }

  // Already touching all needed edges
  if (targetSet.size === 0) {
    return true;
  }

  // BFS
  const visited = new Set();
  const queue = [];
  let head = 0;

  const enqueue = (row, col, type) => {
    const key = `${type}:${row}:${col}`;
    if (visited.has(key)) return;
    visited.add(key);
    queue.push({ row, col, type });
  };

  // Start from component pegs
  for (const peg of component) {
    enqueue(peg.row, peg.col, 'peg');
  }

  while (head < queue.length) {
    const { row, col, type } = queue[head++];

    // Check if we've reached a goal edge
    if (player === 'red') {
      if (targetSet.has(row) && isGoalEdgeCoordinate(player, row, col, boardSize)) {
        if (type === 'peg' || isLegalPlacementForPlayer(board, boardSize, player, row, col)) {
          return true;
        }
      }
    } else {
      if (targetSet.has(col) && isGoalEdgeCoordinate(player, row, col, boardSize)) {
        if (type === 'peg' || isLegalPlacementForPlayer(board, boardSize, player, row, col)) {
          return true;
        }
      }
    }

    // Skip invalid placements for empty cells
    if (type === 'empty' && !isLegalPlacementForPlayer(board, boardSize, player, row, col)) {
      continue;
    }

    // Explore knight moves
    for (const [dr, dc] of KNIGHT_OFFSETS) {
      const nr = row + dr;
      const nc = col + dc;

      if (nr < 0 || nr >= boardSize || nc < 0 || nc >= boardSize) {
        continue;
      }

      const occupant = board[nr][nc];
      let nextType = null;
      if (occupant === null) {
        nextType = 'empty';
      } else if (occupant === player) {
        nextType = 'peg';
      } else {
        continue; // opponent peg blocks
      }

      // Check bridge crossing
      if (bridgesCross(bridges, row, col, nr, nc)) {
        continue;
      }

      enqueue(nr, nc, nextType);
    }
  }

  return false;
}

/**
 * Convert input JSON to internal format with board array.
 */
function parseInput(input) {
  const boardSize = input.boardSize || 24;

  // Create empty board
  const board = Array(boardSize).fill(null).map(() => Array(boardSize).fill(null));

  // Place pegs
  for (const peg of (input.pegs || [])) {
    board[peg.row][peg.col] = peg.player;
  }

  // Parse bridges
  const bridges = (input.bridges || []).map(b => ({
    r1: b.r1, c1: b.c1, r2: b.r2, c2: b.c2
  }));

  // Parse component
  const component = (input.component || []).map(p => ({
    row: p.row, col: p.col
  }));

  return {
    boardSize,
    board,
    bridges,
    player: input.player || 'red',
    component,
    touchesTop: input.touchesTop || false,
    touchesBottom: input.touchesBottom || false,
    touchesLeft: input.touchesLeft || false,
    touchesRight: input.touchesRight || false
  };
}

/**
 * Main entry point - read JSON from stdin, output result to stdout.
 */
function main() {
  let inputData = '';

  process.stdin.setEncoding('utf8');

  process.stdin.on('data', (chunk) => {
    inputData += chunk;
  });

  process.stdin.on('end', () => {
    try {
      const input = JSON.parse(inputData);
      const parsed = parseInput(input);
      const reachable = hasReachableGoalEdge(parsed);

      console.log(JSON.stringify({
        reachable,
        error: null
      }));
    } catch (err) {
      console.log(JSON.stringify({
        reachable: null,
        error: err.message
      }));
      process.exit(1);
    }
  });
}

main();
