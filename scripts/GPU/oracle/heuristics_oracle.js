#!/usr/bin/env node
/**
 * Heuristics Oracle - JS implementations for cross-validation with Python
 *
 * This script provides command-line access to all key heuristics functions,
 * allowing Python tests to verify exact semantic match.
 *
 * Usage:
 *   echo '{"function":"componentMetrics","boardSize":24,"pegs":[...],"player":"red"}' | node heuristics_oracle.js
 *
 * Supported functions:
 *   - componentMetrics: Get component analysis (touches, spans, largest)
 *   - computeFrontier: Get frontier cells and connectors
 *   - connectivityScore: Get connectivity score for player
 *   - findConnectedComponents: Get all connected components
 *   - evaluateConnectedPaths: Path evaluation score
 *   - movePriority: Full move priority calculation
 *   - hasReachableGoalEdge: Sealed lane detection (from search.js)
 *
 * Output: { "result": <value>, "error": null }
 */

'use strict';

// Knight move offsets (standard TwixT moves)
const KNIGHT_OFFSETS = [
  [-2, -1], [-2, 1], [-1, -2], [-1, 2],
  [1, -2], [1, 2], [2, -1], [2, 1]
];

// ============================================================================
// Game State Parsing
// ============================================================================

/**
 * Parse input JSON into a game-like object.
 * Creates a board array and provides game interface methods.
 */
function parseGameState(input) {
  const boardSize = input.boardSize || 24;

  // Create empty board
  const board = Array(boardSize).fill(null).map(() => Array(boardSize).fill(null));

  // Place pegs
  const pegs = [];
  for (const peg of (input.pegs || [])) {
    board[peg.row][peg.col] = peg.player;
    pegs.push(peg);
  }

  // Parse bridges as array of edge objects
  const bridges = (input.bridges || []).map(b => ({
    r1: b.r1, c1: b.c1, r2: b.r2, c2: b.c2
  }));

  // Create bridge set for quick lookup
  const bridgeSet = new Set();
  for (const b of bridges) {
    const key1 = `${b.r1},${b.c1}-${b.r2},${b.c2}`;
    const key2 = `${b.r2},${b.c2}-${b.r1},${b.c1}`;
    bridgeSet.add(key1);
    bridgeSet.add(key2);
  }

  return {
    boardSize,
    board,
    pegs,
    bridges,
    bridgeSet,
    moveCount: input.moveCount || pegs.length,
    currentPlayer: input.toMove || input.player || 'red',
    gameOver: false,
    winner: null,

    // Method to check if two pegs are connected by a bridge
    hasBridge(r1, c1, r2, c2) {
      const key = `${r1},${c1}-${r2},${c2}`;
      return bridgeSet.has(key);
    }
  };
}

// ============================================================================
// Connected Components
// ============================================================================

/**
 * Find all connected components for a player.
 * Two pegs are connected if they share a bridge.
 */
function findConnectedComponents(game, player) {
  const playerPegs = game.pegs.filter(p => p.player === player);
  if (playerPegs.length === 0) return [];

  const visited = new Set();
  const components = [];

  for (const startPeg of playerPegs) {
    const key = `${startPeg.row},${startPeg.col}`;
    if (visited.has(key)) continue;

    const component = [];
    const queue = [startPeg];

    while (queue.length > 0) {
      const peg = queue.shift();
      const pegKey = `${peg.row},${peg.col}`;
      if (visited.has(pegKey)) continue;
      visited.add(pegKey);
      component.push(peg);

      // Find connected pegs via bridges
      for (const [dr, dc] of KNIGHT_OFFSETS) {
        const nr = peg.row + dr;
        const nc = peg.col + dc;
        if (nr < 0 || nr >= game.boardSize || nc < 0 || nc >= game.boardSize) continue;
        if (game.board[nr][nc] !== player) continue;

        // Check if bridge exists
        if (game.hasBridge(peg.row, peg.col, nr, nc)) {
          const nkey = `${nr},${nc}`;
          if (!visited.has(nkey)) {
            queue.push({ row: nr, col: nc, player });
          }
        }
      }
    }

    if (component.length > 0) {
      components.push(component);
    }
  }

  return components;
}

// ============================================================================
// Component Metrics
// ============================================================================

/**
 * Compute metrics for player's connected components.
 * Matches heuristics.js componentMetrics exactly.
 */
function componentMetrics(game, player) {
  const components = findConnectedComponents(game, player);
  const boardSize = game.boardSize;

  let maxRowSpan = 0;
  let maxColSpan = 0;
  let touchesTop = false;
  let touchesBottom = false;
  let touchesLeft = false;
  let touchesRight = false;
  let largestComponent = [];
  let minRowOverall = boardSize;
  let maxRowOverall = -1;
  let minColOverall = boardSize;
  let maxColOverall = -1;

  for (const component of components) {
    const rows = component.map(p => p.row);
    const cols = component.map(p => p.col);

    const minRow = Math.min(...rows);
    const maxRow = Math.max(...rows);
    const minCol = Math.min(...cols);
    const maxCol = Math.max(...cols);

    maxRowSpan = Math.max(maxRowSpan, maxRow - minRow);
    maxColSpan = Math.max(maxColSpan, maxCol - minCol);

    if (component.length > largestComponent.length) {
      largestComponent = component;
    }

    minRowOverall = Math.min(minRowOverall, minRow);
    maxRowOverall = Math.max(maxRowOverall, maxRow);
    minColOverall = Math.min(minColOverall, minCol);
    maxColOverall = Math.max(maxColOverall, maxCol);

    // True edge touches
    if (minRow === 0) touchesTop = true;
    if (maxRow === boardSize - 1) touchesBottom = true;
    if (minCol === 0) touchesLeft = true;
    if (maxCol === boardSize - 1) touchesRight = true;
  }

  return {
    components,
    maxRowSpan,
    maxColSpan,
    touchesTop,
    touchesBottom,
    touchesLeft,
    touchesRight,
    largestComponent,
    minRow: minRowOverall === boardSize ? null : minRowOverall,
    maxRow: maxRowOverall === -1 ? null : maxRowOverall,
    minCol: minColOverall === boardSize ? null : minColOverall,
    maxCol: maxColOverall === -1 ? null : maxColOverall,
  };
}

// ============================================================================
// Compute Frontier
// ============================================================================

/**
 * Compute frontier cells and connectors for a player.
 * Matches heuristics.js computeFrontier exactly.
 */
function computeFrontier(game, player) {
  const boardSize = game.boardSize;
  const frontier = [];
  const connectors = [];
  const trailing = [];
  const seen = new Set();
  const metrics = componentMetrics(game, player);
  const component = metrics.largestComponent || [];

  if (component.length === 0) {
    return { frontier, metrics, connectors, trailing };
  }

  const wantTop = player === 'red' ? !metrics.touchesTop : false;
  const wantBottom = player === 'red' ? !metrics.touchesBottom : false;
  const wantLeft = player === 'black' ? !metrics.touchesLeft : false;
  const wantRight = player === 'black' ? !metrics.touchesRight : false;

  for (const peg of component) {
    for (const [dr, dc] of KNIGHT_OFFSETS) {
      const row = peg.row + dr;
      const col = peg.col + dc;

      if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) continue;
      if (game.board[row][col] !== null) continue;

      const key = `${row},${col}`;
      if (seen.has(key)) continue;

      // Avoid illegal edge placements
      const atTopOrBottom = row === 0 || row === boardSize - 1;
      const atLeftOrRight = col === 0 || col === boardSize - 1;
      if (atTopOrBottom && atLeftOrRight) continue;
      if (player === 'red' && atLeftOrRight) continue;
      if (player === 'black' && atTopOrBottom) continue;

      frontier.push({ row, col });
      let isConnector = false;

      if (player === 'red') {
        const topThreshold = wantTop ? 5 : 3;
        const bottomThreshold = wantBottom ? boardSize - 6 : boardSize - 4;

        if (wantTop && row <= topThreshold) {
          isConnector = true;
        }
        if (wantBottom && row >= bottomThreshold) {
          isConnector = true;
        }

        if (!wantTop && !wantBottom) {
          if (row > topThreshold && row < bottomThreshold) {
            trailing.push({ row, col });
          }
        }
      } else {
        const leftThreshold = wantLeft ? 5 : 3;
        const rightThreshold = wantRight ? boardSize - 6 : boardSize - 4;

        if (wantLeft && col <= leftThreshold) {
          isConnector = true;
        }
        if (wantRight && col >= rightThreshold) {
          isConnector = true;
        }

        if (!wantLeft && !wantRight) {
          if (col > leftThreshold && col < rightThreshold) {
            trailing.push({ row, col });
          }
        }
      }

      if (isConnector) {
        connectors.push({ row, col });
      }

      seen.add(key);
    }
  }

  return { frontier, metrics, connectors, trailing };
}

// ============================================================================
// Connectivity Score
// ============================================================================

/**
 * Calculate connectivity score for a player.
 * Matches heuristics.js connectivityScore exactly.
 */
function connectivityScore(game, player) {
  const metrics = componentMetrics(game, player);
  const boardSize = game.boardSize;

  if (!metrics.largestComponent || metrics.largestComponent.length === 0) {
    return 0;
  }

  let score = 0;

  // Base score from largest component size
  score += metrics.largestComponent.length * 10;

  // Span bonuses
  if (player === 'red') {
    score += metrics.maxRowSpan * 8;
    // Edge touch bonuses
    if (metrics.touchesTop) score += 50;
    if (metrics.touchesBottom) score += 50;
    // Both edges = big bonus
    if (metrics.touchesTop && metrics.touchesBottom) score += 200;
  } else {
    score += metrics.maxColSpan * 8;
    if (metrics.touchesLeft) score += 50;
    if (metrics.touchesRight) score += 50;
    if (metrics.touchesLeft && metrics.touchesRight) score += 200;
  }

  // Gap penalty (distance to edges)
  const gap = player === 'red'
    ? (metrics.minRow || 0) + (boardSize - 1 - (metrics.maxRow || boardSize - 1))
    : (metrics.minCol || 0) + (boardSize - 1 - (metrics.maxCol || boardSize - 1));
  score -= gap * 2;

  return score;
}

// ============================================================================
// Evaluate Connected Paths
// ============================================================================

/**
 * Score a single component - matches JS heuristics.js scoreComponent exactly.
 */
function scoreComponent(component, player, boardSize) {
  // JS: component.length * 10
  let score = component.length * 10;

  if (player === 'red') {
    const minRow = Math.min(...component.map(p => p.row));
    const maxRow = Math.max(...component.map(p => p.row));
    const span = maxRow - minRow;
    // JS: span * 20
    score += span * 20;
    // true top-bottom span
    if (minRow === 0 && maxRow === boardSize - 1) {
      score += 500;
    }
  } else {
    const minCol = Math.min(...component.map(p => p.col));
    const maxCol = Math.max(...component.map(p => p.col));
    const span = maxCol - minCol;
    // JS: span * 20
    score += span * 20;
    // true left-right span
    if (minCol === 0 && maxCol === boardSize - 1) {
      score += 500;
    }
  }

  return score;
}

/**
 * Evaluate winning threats for all components.
 * Matches JS heuristics.js evaluateWinningThreats exactly.
 */
function evaluateWinningThreats(player, components, boardSize) {
  let score = 0;

  for (const component of components) {
    if (player === 'red') {
      const minRow = Math.min(...component.map(p => p.row));
      const maxRow = Math.max(...component.map(p => p.row));
      // JS: strongest threat = true-edge span (0 ↔ 23 on 24×24)
      if (minRow === 0 && maxRow === boardSize - 1) score += 800;
      else if (minRow <= 1 && maxRow >= boardSize - 2) score += 400;
      else if (maxRow >= boardSize - 2 && minRow <= 5) score += 400;
      else if (minRow <= 3 && maxRow >= boardSize - 4) score += 200;
    } else {
      const minCol = Math.min(...component.map(p => p.col));
      const maxCol = Math.max(...component.map(p => p.col));
      if (minCol === 0 && maxCol === boardSize - 1) score += 800;
      else if (minCol <= 1 && maxCol >= boardSize - 2) score += 400;
      else if (maxCol >= boardSize - 2 && minCol <= 5) score += 400;
      else if (minCol <= 3 && maxCol >= boardSize - 4) score += 200;
    }
  }

  return score;
}

/**
 * Evaluate connected path score for a player.
 * Matches heuristics.js evaluateConnectedPaths exactly.
 */
function evaluateConnectedPaths(game, player) {
  let score = 0;
  const playerPegs = game.pegs.filter(p => p.player === player);
  if (playerPegs.length === 0) return -100;

  const components = findConnectedComponents(game, player);
  if (components.length === 0) return -50;

  for (const component of components) {
    score += scoreComponent(component, player, game.boardSize);
  }

  if (components.length > 0) {
    const avgComponentSize = playerPegs.length / components.length;
    score += avgComponentSize * 20;
    if (components.length > 3) {
      score -= (components.length - 3) * 30;
    }
  }

  score += evaluateWinningThreats(player, components, game.boardSize);
  return score;
}

// ============================================================================
// Sealed Lane Detection (from search.js)
// ============================================================================

/**
 * Check if a coordinate is on the player's goal edge (not corners).
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
 * Check if placement is legal for player.
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
 * Check if two segments cross (for bridge blocking).
 */
function segmentsCross(r1, c1, r2, c2, r3, c3, r4, c4) {
  function orientation(pr, pc, qr, qc, rr, rc) {
    const val = (qc - pc) * (rr - qr) - (qr - pr) * (rc - qc);
    if (val === 0) return 0;
    return val > 0 ? 1 : 2;
  }

  function onSegment(pr, pc, qr, qc, rr, rc) {
    return qr <= Math.max(pr, rr) && qr >= Math.min(pr, rr) &&
           qc <= Math.max(pc, rc) && qc >= Math.min(pc, rc);
  }

  const o1 = orientation(r1, c1, r2, c2, r3, c3);
  const o2 = orientation(r1, c1, r2, c2, r4, c4);
  const o3 = orientation(r3, c3, r4, c4, r1, c1);
  const o4 = orientation(r3, c3, r4, c4, r2, c2);

  if (o1 !== o2 && o3 !== o4) {
    const sharesEndpoint =
      (r1 === r3 && c1 === c3) || (r1 === r4 && c1 === c4) ||
      (r2 === r3 && c2 === c3) || (r2 === r4 && c2 === c4);
    return !sharesEndpoint;
  }

  if (o1 === 0 && onSegment(r1, c1, r3, c3, r2, c2)) {
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
 * Check if a potential bridge crosses any existing bridge.
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
 */
function hasReachableGoalEdge(game, player, metrics) {
  const { boardSize, board, bridges } = game;
  const component = metrics.largestComponent || [];

  if (component.length === 0) {
    return false;
  }

  // Determine target edges
  const targetSet = new Set();
  if (player === 'red') {
    if (!metrics.touchesTop) targetSet.add(0);
    if (!metrics.touchesBottom) targetSet.add(boardSize - 1);
  } else {
    if (!metrics.touchesLeft) targetSet.add(0);
    if (!metrics.touchesRight) targetSet.add(boardSize - 1);
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

// ============================================================================
// Move Priority (simplified version for oracle testing)
// ============================================================================

/**
 * Calculate move priority score.
 * Simplified version for oracle testing - focuses on key features.
 */
function movePriority(game, move, player, params) {
  const opponent = player === 'red' ? 'black' : 'red';
  const boardSize = game.boardSize;
  const board = game.board;

  let score = 0;
  const features = {};

  // Connection counts
  let friendlyConnections = 0;
  let opponentConnections = 0;
  for (const [dr, dc] of KNIGHT_OFFSETS) {
    const r = move.row + dr;
    const c = move.col + dc;
    if (r < 0 || r >= boardSize || c < 0 || c >= boardSize) continue;
    if (board[r][c] === player) {
      friendlyConnections++;
    } else if (board[r][c] === opponent) {
      opponentConnections++;
    }
  }

  features.friendlyConnections = friendlyConnections * 60;
  features.opponentConnections = opponentConnections * 30;
  score += features.friendlyConnections + features.opponentConnections;

  // Goal distance
  const goalDistance = player === 'red'
    ? Math.min(move.row, boardSize - 1 - move.row)
    : Math.min(move.col, boardSize - 1 - move.col);
  features.goalDistance = Math.max(0, 12 - goalDistance) * 15;
  score += features.goalDistance;

  // Center bias
  const center = (boardSize - 1) / 2;
  const centerDist = Math.abs(move.row - center) + Math.abs(move.col - center);
  features.centerBias = Math.max(0, 16 - centerDist) * 3;
  score += features.centerBias;

  return { score, features };
}

// ============================================================================
// Main Dispatcher
// ============================================================================

const functions = {
  'componentMetrics': (input) => {
    const game = parseGameState(input);
    return componentMetrics(game, input.player);
  },

  'computeFrontier': (input) => {
    const game = parseGameState(input);
    return computeFrontier(game, input.player);
  },

  'connectivityScore': (input) => {
    const game = parseGameState(input);
    return connectivityScore(game, input.player);
  },

  'findConnectedComponents': (input) => {
    const game = parseGameState(input);
    return findConnectedComponents(game, input.player);
  },

  'evaluateConnectedPaths': (input) => {
    const game = parseGameState(input);
    return evaluateConnectedPaths(game, input.player);
  },

  'hasReachableGoalEdge': (input) => {
    const game = parseGameState(input);
    const metrics = {
      largestComponent: input.component || [],
      touchesTop: input.touchesTop || false,
      touchesBottom: input.touchesBottom || false,
      touchesLeft: input.touchesLeft || false,
      touchesRight: input.touchesRight || false,
    };
    return hasReachableGoalEdge(game, input.player, metrics);
  },

  'movePriority': (input) => {
    const game = parseGameState(input);
    const move = { row: input.moveRow, col: input.moveCol };
    return movePriority(game, move, input.player, input.params || {});
  },

  'isGoalEdgeCoordinate': (input) => {
    return isGoalEdgeCoordinate(input.player, input.row, input.col, input.boardSize);
  },

  'isLegalPlacementForPlayer': (input) => {
    const game = parseGameState(input);
    return isLegalPlacementForPlayer(game.board, game.boardSize, input.player, input.row, input.col);
  }
};

// ============================================================================
// Main Entry Point
// ============================================================================

function main() {
  let inputData = '';

  process.stdin.setEncoding('utf8');

  process.stdin.on('data', (chunk) => {
    inputData += chunk;
  });

  process.stdin.on('end', () => {
    try {
      const input = JSON.parse(inputData);
      const funcName = input.function;

      if (!funcName || !functions[funcName]) {
        console.log(JSON.stringify({
          result: null,
          error: `Unknown function: ${funcName}. Available: ${Object.keys(functions).join(', ')}`
        }));
        process.exit(1);
      }

      const result = functions[funcName](input);

      console.log(JSON.stringify({
        result,
        error: null
      }));
    } catch (err) {
      console.log(JSON.stringify({
        result: null,
        error: err.message
      }));
      process.exit(1);
    }
  });
}

main();
