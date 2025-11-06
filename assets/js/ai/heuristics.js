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

export function evaluatePosition(game, player) {
  if (game.gameOver) {
    return game.winner === player ? 10000 : -10000;
  }

  const opponent = player === 'red' ? 'black' : 'red';
  let score = 0;

  // --- your existing coarse terms (kept) ---
  score += evaluateConnectedPaths(game, player) * 100;
  score -= evaluateConnectedPaths(game, opponent) * 100;

  score += evaluatePotentialConnections(game, player) * 20;
  score -= evaluatePotentialConnections(game, opponent) * 20;

  score += evaluateEdgeProgress(game, player) * 30;
  score -= evaluateEdgeProgress(game, opponent) * 30;

  const playerPegs = game.pegs.filter((p) => p.player === player).length;
  const opponentPegs = game.pegs.filter((p) => p.player === opponent).length;
  score += (playerPegs - opponentPegs) * 2;

  // --- NEW: shortest-path-to-goal pull on the largest component ---
  // Approximates “fewest additional rows/cols” needed to touch both goal edges.
  // Uses thresholds (≤1 / ≥N-2) to respect knight reach to the edge.
  try {
    const { largestComponent } = componentMetrics(game, player);
    if (largestComponent && largestComponent.length) {
      let minR = Infinity,
        maxR = -Infinity,
        minC = Infinity,
        maxC = -Infinity;
      for (const p of largestComponent) {
        if (p.row < minR) minR = p.row;
        if (p.row > maxR) maxR = p.row;
        if (p.col < minC) minC = p.col;
        if (p.col > maxC) maxC = p.col;
      }
      const N = game.boardSize;
      const topThr = 1,
        botThr = N - 2,
        leftThr = 1,
        rightThr = N - 2;

      const touchesTop = minR <= topThr;
      const touchesBottom = maxR >= botThr;
      const touchesLeft = minC <= leftThr;
      const touchesRight = maxC >= rightThr;

      const gapTop = Math.max(0, minR - topThr);
      const gapBottom = Math.max(0, botThr - maxR);
      const gapLeft = Math.max(0, minC - leftThr);
      const gapRight = Math.max(0, rightThr - maxC);

      if (player === 'red') {
        // If one side is already touched, strongly pull to complete the other.
        const gap = touchesTop
          ? gapBottom
          : touchesBottom
            ? gapTop
            : Math.min(gapTop, gapBottom);
        const urgency = touchesTop || touchesBottom ? 2.5 : 1.0;
        // Closer = bigger bonus; also penalize being far from either edge a little.
        score += 200 * urgency * (1 / (1 + gap));
        score -= 40 * (gapTop + gapBottom);
      } else {
        const gap = touchesLeft
          ? gapRight
          : touchesRight
            ? gapLeft
            : Math.min(gapLeft, gapRight);
        const urgency = touchesLeft || touchesRight ? 2.5 : 1.0;
        score += 200 * urgency * (1 / (1 + gap));
        score -= 40 * (gapLeft + gapRight);
      }
    }
  } catch {
    // If componentMetrics isn't available for some reason, just skip the term.
  }

  // --- NEW: small “drift” penalty to prefer finishing sooner ---
  score -= 0.05 * (game.moveCount || 0);

  return score;
}

export function evaluateMove(game, move, player) {
  let score = 0;
  const opponent = player === 'red' ? 'black' : 'red';

  let connectionCount = 0;
  for (const [dr, dc] of KNIGHT_MOVES) {
    const checkRow = move.row + dr;
    const checkCol = move.col + dc;

    if (
      checkRow >= 0 &&
      checkRow < game.boardSize &&
      checkCol >= 0 &&
      checkCol < game.boardSize &&
      game.board[checkRow][checkCol] === player
    ) {
      const originalCurrent = game.currentPlayer;
      game.currentPlayer = player;
      const crosses = game.bridgesCross(move.row, move.col, checkRow, checkCol);
      game.currentPlayer = originalCurrent;

      if (!crosses) {
        connectionCount++;
        const distance =
          Math.abs(move.row - checkRow) + Math.abs(move.col - checkCol);
        score += 100 + distance * 5;

        if (player === 'black') {
          const spansBoard =
            (move.col <= 3 && checkCol >= 20) ||
            (move.col >= 20 && checkCol <= 3);
          const wideSpan = Math.abs(move.col - checkCol) > 10;
          if (spansBoard) score += 300;
          else if (wideSpan) score += 150;
        } else {
          const spansBoard =
            (move.row <= 3 && checkRow >= 20) ||
            (move.row >= 20 && checkRow <= 3);
          const wideSpan = Math.abs(move.row - checkRow) > 10;
          if (spansBoard) score += 300;
          else if (wideSpan) score += 150;
        }
      }
    }
  }

  if (connectionCount >= 2) {
    score += connectionCount * 75;
  }

  if (player === 'black') {
    const distanceToNearestGoal = Math.min(
      move.col,
      game.boardSize - 1 - move.col
    );
    score += Math.max(0, 12 - distanceToNearestGoal) * 8;
  } else {
    const distanceToNearestGoal = Math.min(
      move.row,
      game.boardSize - 1 - move.row
    );
    score += Math.max(0, 12 - distanceToNearestGoal) * 8;
  }

  let opponentThreats = 0;
  for (const [dr, dc] of KNIGHT_MOVES) {
    const checkRow = move.row + dr;
    const checkCol = move.col + dc;

    if (
      checkRow >= 0 &&
      checkRow < game.boardSize &&
      checkCol >= 0 &&
      checkCol < game.boardSize &&
      game.board[checkRow][checkCol] === opponent
    ) {
      opponentThreats++;
    }
  }

  if (opponentThreats > 0) {
    score += opponentThreats * 25;
  }

  if (game.moveCount < 10) {
    const centerDistance =
      Math.abs(move.row - 11.5) + Math.abs(move.col - 11.5);
    score += Math.max(0, 24 - centerDistance) * 2;
  }

  return score;
}

export function connectivityScore(game, player) {
  return evaluateConnectedPaths(game, player);
}

function evaluateConnectedPaths(game, player) {
  let score = 0;
  const playerPegs = game.pegs.filter((p) => p.player === player);
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

  score += evaluateWinningThreats(player, components);
  return score;
}

function evaluateWinningThreats(player, components) {
  let score = 0;

  for (const component of components) {
    if (player === 'red') {
      const minRow = Math.min(...component.map((p) => p.row));
      const maxRow = Math.max(...component.map((p) => p.row));
      // strongest threat = true-edge span (0 ↔ 23 on 24×24)
      if (minRow === 0 && maxRow === 23) score += 800;
      else if (minRow <= 1 && maxRow >= 22) score += 400;
      else if (maxRow >= 22 && minRow <= 5) score += 400;
      else if (minRow <= 3 && maxRow >= 20) score += 200;
    } else {
      const minCol = Math.min(...component.map((p) => p.col));
      const maxCol = Math.max(...component.map((p) => p.col));
      if (minCol === 0 && maxCol === 23) score += 800;
      else if (minCol <= 1 && maxCol >= 22) score += 400;
      else if (maxCol >= 22 && minCol <= 5) score += 400;
      else if (minCol <= 3 && maxCol >= 20) score += 200;
    }
  }

  return score;
}

export function findConnectedComponents(game, player) {
  const playerPegs = game.pegs.filter((p) => p.player === player);
  const visited = new Set();
  const components = [];

  for (const peg of playerPegs) {
    const pegKey = `${peg.row},${peg.col}`;
    if (visited.has(pegKey)) continue;

    const component = [];
    const stack = [peg];

    while (stack.length > 0) {
      const currentPeg = stack.pop();
      const currentKey = `${currentPeg.row},${currentPeg.col}`;
      if (visited.has(currentKey)) continue;

      visited.add(currentKey);
      component.push(currentPeg);

      for (const bridge of game.bridges) {
        if (bridge.player !== player) continue;

        let connectedPeg = null;
        if (
          bridge.from.row === currentPeg.row &&
          bridge.from.col === currentPeg.col
        ) {
          connectedPeg = { row: bridge.to.row, col: bridge.to.col, player };
        } else if (
          bridge.to.row === currentPeg.row &&
          bridge.to.col === currentPeg.col
        ) {
          connectedPeg = { row: bridge.from.row, col: bridge.from.col, player };
        }

        if (connectedPeg) {
          const connectedKey = `${connectedPeg.row},${connectedPeg.col}`;
          if (!visited.has(connectedKey)) {
            stack.push(connectedPeg);
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

function scoreComponent(component, player, boardSize) {
  let score = component.length * 10;

  if (player === 'red') {
    const minRow = Math.min(...component.map((p) => p.row));
    const maxRow = Math.max(...component.map((p) => p.row));
    const span = maxRow - minRow;
    score += span * 20;
    // true top↔bottom span
    if (minRow === 0 && maxRow === boardSize - 1) {
      score += 500;
    }
  } else {
    const minCol = Math.min(...component.map((p) => p.col));
    const maxCol = Math.max(...component.map((p) => p.col));
    const span = maxCol - minCol;
    score += span * 20;
    // true left↔right span
    if (minCol === 0 && maxCol === boardSize - 1) {
      score += 500;
    }
  }

  return score;
}

function evaluatePotentialConnections(game, player) {
  let score = 0;
  const playerPegs = game.pegs.filter((p) => p.player === player);

  for (const peg of playerPegs) {
    for (const [dr, dc] of KNIGHT_MOVES) {
      const newRow = peg.row + dr;
      const newCol = peg.col + dc;
      if (isValidPlacementForPlayer(game, player, newRow, newCol)) {
        if (player === 'red') {
          if (peg.row < 12 && newRow > peg.row) score += 5;
          if (peg.row > 12 && newRow < peg.row) score += 5;
        } else {
          if (peg.col < 12 && newCol > peg.col) score += 5;
          if (peg.col > 12 && newCol < peg.col) score += 5;
        }
      }
    }
  }

  return score;
}

function evaluateEdgeProgress(game, player) {
  let score = 0;
  const playerPegs = game.pegs.filter((p) => p.player === player);

  for (const peg of playerPegs) {
    if (player === 'red') {
      const distanceToGoal = Math.min(peg.row, game.boardSize - 1 - peg.row);
      score += Math.max(0, 12 - distanceToGoal);
    } else {
      const distanceToGoal = Math.min(peg.col, game.boardSize - 1 - peg.col);
      score += Math.max(0, 12 - distanceToGoal);
    }
  }

  return score;
}

function isValidPlacementForPlayer(game, player, row, col) {
  if (row < 0 || row >= game.boardSize || col < 0 || col >= game.boardSize)
    return false;
  if (game.board[row][col] !== null) return false;

  const atTopOrBottom = row === 0 || row === game.boardSize - 1;
  const atLeftOrRight = col === 0 || col === game.boardSize - 1;
  if (atTopOrBottom && atLeftOrRight) return false;

  if (player === 'red') {
    if (atLeftOrRight) return false;
  } else {
    if (atTopOrBottom) return false;
  }

  return true;
}

export function componentMetrics(game, player) {
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
    const rows = component.map((p) => p.row);
    const cols = component.map((p) => p.col);

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

    // === changed: true-edge touches ===
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

export function computeFrontier(game, player) {
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
    for (const [dr, dc] of KNIGHT_MOVES) {
      const row = peg.row + dr;
      const col = peg.col + dc;

      if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) continue;
      if (game.board[row][col] !== null) continue;

      const key = `${row},${col}`;
      if (seen.has(key)) continue;

      // Avoid illegal edge placements for each player
      const atTopOrBottom = row === 0 || row === boardSize - 1;
      const atLeftOrRight = col === 0 || col === boardSize - 1;
      if (atTopOrBottom && atLeftOrRight) continue;
      if (player === 'red' && atLeftOrRight) continue;
      if (player === 'black' && atTopOrBottom) continue;

      frontier.push({ row, col });
      let isConnector = false;
      if (player === 'red') {
        const topThreshold = wantTop ? 5 : 3;
        const bottomThreshold = wantBottom ? 5 : 3;
        if (wantTop && row <= topThreshold) isConnector = true;
        if (wantBottom && row >= boardSize - 1 - bottomThreshold)
          isConnector = true;
        if (
          !wantTop &&
          !wantBottom &&
          (row <= topThreshold || row >= boardSize - 1 - bottomThreshold)
        ) {
          isConnector = true;
        }
      } else {
        const leftThreshold = wantLeft ? 5 : 3;
        const rightThreshold = wantRight ? 5 : 3;
        if (wantLeft && col <= leftThreshold) isConnector = true;
        if (wantRight && col >= boardSize - 1 - rightThreshold)
          isConnector = true;
        if (
          !wantLeft &&
          !wantRight &&
          (col <= leftThreshold || col >= boardSize - 1 - rightThreshold)
        ) {
          isConnector = true;
        }
      }
      if (isConnector) {
        connectors.push({ row, col });
      } else {
        trailing.push({ row, col });
      }
      seen.add(key);
    }
  }

  return { frontier, metrics, connectors, trailing };
}
