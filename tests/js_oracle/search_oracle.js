/**
 * Search Oracle - JS implementation for comparing getBestMove with Python.
 *
 * Reads game state from stdin, calls getBestMove, returns results as JSON.
 *
 * Usage: echo '{"boardSize":24,"pegs":[...],"bridges":[],"currentPlayer":"red","depth":2}' | node search_oracle.js
 */

import * as readline from 'readline';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Import the actual JS modules
const gamePath = join(__dirname, '../../assets/js/game/twixtGame.js');
const searchPath = join(__dirname, '../../assets/js/ai/search.js');
const valueModelPath = join(__dirname, '../../assets/js/ai/valueModel.js');

const { default: TwixTGame } = await import(gamePath);
const { default: TwixTAI } = await import(searchPath);
const { clearValueModel } = await import(valueModelPath);

// Disable value model for pure heuristic comparison
clearValueModel();

/**
 * Create a TwixTGame instance from serialized state.
 */
function createGameFromState(state) {
  const game = new TwixTGame(state.boardSize || 24);
  game.deterministicMode = true;
  game.disableOpeningBook = true;

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
    const from = bridge.from || { row: bridge.r1, col: bridge.c1 };
    const to = bridge.to || { row: bridge.r2, col: bridge.c2 };
    game.bridges.push({
      from: from,
      to: to,
      player: bridge.player,
    });
  }

  game.currentPlayer = state.currentPlayer || state.toMove || 'red';
  game.moveCount = state.moveCount || state.pegs?.length || 0;
  game.gameOver = state.gameOver || false;
  game.winner = state.winner || null;

  return game;
}

/**
 * Run getBestMove and return results.
 * We need to intercept the internal scoredMoves to get scores since getBestMove only returns the move.
 */
function runGetBestMove(state, depth) {
  const game = createGameFromState(state);

  try {
    const ai = new TwixTAI(game, game.currentPlayer);

    // Override rootDepth to force specific depth
    ai.rootDepth = depth;

    // Enable debug to capture move details
    ai.debugEnabled = true;

    // Disable randomness for deterministic results
    const originalRandom = Math.random;
    Math.random = () => 1; // Always return 1 to skip randomization

    const moveResult = ai.getBestMove();

    // Restore random
    Math.random = originalRandom;

    if (!moveResult) {
      return { row: null, col: null, score: 0, candidates: [] };
    }

    // Access debug info if available
    const lastTrace = ai.moveTrace && ai.moveTrace.length > 0 ? ai.moveTrace[ai.moveTrace.length - 1] : null;
    const moves = lastTrace ? lastTrace.moves : [];

    // Sort by totalScore descending
    const sortedMoves = [...moves].sort((a, b) => (b.totalScore || 0) - (a.totalScore || 0));
    const bestScore = sortedMoves.length > 0 ? sortedMoves[0].totalScore : 0;

    return {
      row: moveResult.row,
      col: moveResult.col,
      score: bestScore,
      candidates: sortedMoves.slice(0, 10).map(m => ({
        row: m.move.row,
        col: m.move.col,
        score: m.totalScore,
        minimax: m.minimaxScore,
        immediate: m.immediateScore,
        position: m.positionScore,
        heuristic: m.heuristicScore,
      })),
    };
  } catch (e) {
    return { error: e.message, stack: e.stack };
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
    const depth = request.depth || 2;
    const result = runGetBestMove(request, depth);
    console.log(JSON.stringify(result));
  } catch (e) {
    console.log(JSON.stringify({ error: e.message, stack: e.stack }));
    process.exit(1);
  }
}

main();
