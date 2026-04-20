/**
 * Heuristics Oracle - JS implementation for cross-validation with Python.
 *
 * Reads game state from stdin, calls heuristic functions, returns results as JSON.
 *
 * Usage: echo '{"boardSize":24,"pegs":[...],"bridges":[...],"player":"red"}' | node heuristics_oracle.js
 */

import * as readline from 'readline';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Import the actual JS heuristics and game modules
const heuristicsPath = join(__dirname, '../../assets/js/ai/heuristics.js');
const gamePath = join(__dirname, '../../assets/js/game/twixtGame.js');
const searchPath = join(__dirname, '../../assets/js/ai/search.js');
const valueModelPath = join(__dirname, '../../assets/js/ai/valueModel.js');

const heuristics = await import(heuristicsPath);
const { default: TwixTGame } = await import(gamePath);
const { default: TwixTAI, computeConnectorTargets } = await import(searchPath);
const { clearValueModel } = await import(valueModelPath);

// Disable value model for pure heuristic comparison
clearValueModel();

/**
 * Create a TwixTGame instance from serialized state.
 */
function createGameFromState(state) {
  const game = new TwixTGame(state.boardSize || 24);

  // Clear board
  for (let r = 0; r < game.boardSize; r++) {
    for (let c = 0; c < game.boardSize; c++) {
      game.board[r][c] = null;
    }
  }
  game.pegs = [];
  game.bridges = [];

  // Add pegs
  for (const peg of state.pegs || []) {
    game.board[peg.row][peg.col] = peg.player;
    game.pegs.push({ row: peg.row, col: peg.col, player: peg.player });
  }

  // Add bridges
  for (const bridge of state.bridges || []) {
    game.bridges.push({
      from: { row: bridge.r1, col: bridge.c1 },
      to: { row: bridge.r2, col: bridge.c2 },
      player: bridge.player,
    });
  }

  game.currentPlayer = state.currentPlayer || 'red';
  game.moveCount = state.moveCount || state.pegs?.length || 0;
  game.gameOver = state.gameOver || false;
  game.winner = state.winner || null;

  return game;
}

/**
 * Run all heuristic functions and return results.
 */
function runHeuristics(state, player) {
  const game = createGameFromState(state);
  const opponent = player === 'red' ? 'black' : 'red';
  const results = {};

  // evaluatePosition
  try {
    results.evaluatePosition = heuristics.evaluatePosition(game, player);
  } catch (e) {
    results.evaluatePosition = { error: e.message };
  }

  // evaluateConnectedPaths (via connectivityScore which is exported)
  try {
    results.evaluateConnectedPaths = heuristics.connectivityScore(game, player);
    results.evaluateConnectedPathsOpponent = heuristics.connectivityScore(game, opponent);
  } catch (e) {
    results.evaluateConnectedPaths = { error: e.message };
  }

  // findConnectedComponents
  try {
    const components = heuristics.findConnectedComponents(game, player);
    results.findConnectedComponents = components.map(comp =>
      comp.map(p => ({ row: p.row, col: p.col }))
    );
  } catch (e) {
    results.findConnectedComponents = { error: e.message };
  }

  // componentMetrics
  try {
    const metrics = heuristics.componentMetrics(game, player);
    results.componentMetrics = {
      maxRowSpan: metrics.maxRowSpan,
      maxColSpan: metrics.maxColSpan,
      touchesTop: metrics.touchesTop,
      touchesBottom: metrics.touchesBottom,
      touchesLeft: metrics.touchesLeft,
      touchesRight: metrics.touchesRight,
      largestComponentSize: metrics.largestComponent?.length || 0,
      minRow: metrics.minRow,
      maxRow: metrics.maxRow,
      minCol: metrics.minCol,
      maxCol: metrics.maxCol,
      numComponents: metrics.components?.length || 0,
    };
  } catch (e) {
    results.componentMetrics = { error: e.message };
  }

  // computeFrontier
  try {
    const frontier = heuristics.computeFrontier(game, player);
    results.computeFrontier = {
      frontierSize: frontier.frontier?.length || 0,
      connectorsSize: frontier.connectors?.length || 0,
      trailingSize: frontier.trailing?.length || 0,
      frontier: (frontier.frontier || []).map(p => ({ row: p.row, col: p.col })),
      connectors: (frontier.connectors || []).map(p => ({ row: p.row, col: p.col })),
    };
  } catch (e) {
    results.computeFrontier = { error: e.message };
  }

  return results;
}

/**
 * Evaluate a specific move.
 */
function runEvaluateMove(state, move, player) {
  const game = createGameFromState(state);

  try {
    const score = heuristics.evaluateMove(game, move, player);
    return { evaluateMove: score };
  } catch (e) {
    return { evaluateMove: { error: e.message } };
  }
}

/**
 * Run movePriority scoring for a move.
 * This tests the full move ordering heuristic from TwixTAI.
 */
function runMovePriority(state, move, player) {
  const game = createGameFromState(state);
  const opponent = player === 'red' ? 'black' : 'red';

  try {
    const ai = new TwixTAI(game, player);
    ai.rootDepth = 2; // Set a default depth
    ai.logHeuristics = false; // Suppress debug output

    // Pre-compute metrics like the real orderMoves does
    const friendlyPegs = game.pegs.filter(p => p.player === player);
    const opponentPegs = game.pegs.filter(p => p.player === opponent);
    const friendlyMetrics = heuristics.componentMetrics(game, player);
    const opponentMetrics = heuristics.componentMetrics(game, opponent);
    const opponentThreat = heuristics.connectivityScore(game, opponent);
    const {
      frontier: opponentFrontier,
      connectors: opponentConnectors,
      trailing: opponentTrailing,
    } = heuristics.computeFrontier(game, opponent);

    // Compute connector targets using TwixTAI's internal method (same as getBestMove)
    // This is done by computing the targets like getBestMove does
    const friendlyConnectorTargets = computeConnectorTargets(game, player, friendlyMetrics);
    const opponentConnectorTargets = computeConnectorTargets(game, opponent, opponentMetrics);

    // Determine opponent urgency like getBestMove does
    const spanValue = opponentMetrics
      ? opponent === 'red'
        ? opponentMetrics.maxRowSpan
        : opponentMetrics.maxColSpan
      : 0;
    const largestLength = opponentMetrics?.largestComponent?.length || 0;
    const opponentUrgent =
      spanValue >= Math.max(6, Math.floor(game.boardSize / 4)) ||
      largestLength >= 6;

    // Create a capture function to record score components
    const breakdown = {};
    const capture = (name, value) => {
      breakdown[name] = (breakdown[name] || 0) + value;
    };

    // Call movePriority with all the precomputed data + capture function
    const score = ai.movePriority(
      move,
      player,
      friendlyPegs,
      opponentPegs,
      opponent,
      opponentThreat,
      friendlyMetrics,
      friendlyConnectorTargets,
      opponentConnectorTargets,
      opponentMetrics,
      opponentFrontier,
      opponentConnectors,
      opponentTrailing,
      opponentUrgent,
      capture  // Pass capture function for breakdown
    );

    breakdown.total = score;
    return { movePriority: score, breakdown };
  } catch (e) {
    return { movePriority: { error: e.message, stack: e.stack } };
  }
}

/**
 * Main entry point - read from stdin and process.
 */
async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false,
  });

  let inputData = '';

  for await (const line of rl) {
    inputData += line;
  }

  try {
    const request = JSON.parse(inputData);

    if (request.command === 'evaluateMove') {
      // Single move evaluation
      const result = runEvaluateMove(request.state, request.move, request.player);
      console.log(JSON.stringify(result));
    } else if (request.command === 'movePriority') {
      // Move priority scoring (full heuristic)
      const result = runMovePriority(request.state, request.move, request.player);
      console.log(JSON.stringify(result));
    } else if (request.command === 'batch') {
      // Batch evaluation of multiple states
      const results = [];
      for (const item of request.items) {
        if (item.command === 'evaluateMove') {
          results.push(runEvaluateMove(item.state, item.move, item.player));
        } else if (item.command === 'movePriority') {
          results.push(runMovePriority(item.state, item.move, item.player));
        } else {
          results.push(runHeuristics(item.state, item.player));
        }
      }
      console.log(JSON.stringify({ results }));
    } else {
      // Default: run all heuristics for a state
      const result = runHeuristics(request.state || request, request.player);
      console.log(JSON.stringify(result));
    }
  } catch (e) {
    console.log(JSON.stringify({ error: e.message, stack: e.stack }));
    process.exit(1);
  }
}

main();
