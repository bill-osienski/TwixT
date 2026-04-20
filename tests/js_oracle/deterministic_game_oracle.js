/**
 * Deterministic Game Oracle - Plays a deterministic TwixT game for parity testing.
 *
 * Reads configuration from stdin, plays a game with deterministic mode enabled,
 * and outputs the full move trace for comparison with Python.
 *
 * Usage:
 *   echo '{"seed": 0, "depth": 2, "maxMoves": 220}' | node deterministic_game_oracle.js
 *
 * Input JSON:
 *   - seed: Game seed (determines starting player: even=black, odd=red)
 *   - depth: Search depth for AI
 *   - maxMoves: Maximum moves before declaring draw (default 220)
 *   - stallLimit: Stall detection limit (default 40)
 *
 * Output JSON:
 *   - winner: "red" | "black" | "draw"
 *   - moves: Array of {turn, player, row, col}
 *   - totalMoves: Number of moves played
 *   - reason: "win" | "stall" | "max_moves" | "no_moves"
 *   - startingPlayer: "red" | "black"
 */

import * as readline from 'readline';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Import game and AI modules
const gamePath = join(__dirname, '../../assets/js/game/twixtGame.js');
const searchPath = join(__dirname, '../../assets/js/ai/search.js');
const heuristicsPath = join(__dirname, '../../assets/js/ai/heuristics.js');
const valueModelPath = join(__dirname, '../../assets/js/ai/valueModel.js');

const { default: TwixTGame } = await import(gamePath);
const { default: TwixTAI } = await import(searchPath);
const { componentMetrics, clearComponentCache } = await import(heuristicsPath);
const { clearValueModel } = await import(valueModelPath);

// Disable value model for pure heuristic comparison
clearValueModel();

/**
 * Play a deterministic game and return the trace.
 */
function playDeterministicGame(config) {
  const { seed = 0, depth = 2, maxMoves = 220, stallLimit = 40 } = config;

  // Alternating start: even seeds start black, odd start red (matches Python)
  const startingPlayer = seed % 2 === 0 ? 'black' : 'red';

  const game = new TwixTGame(24);
  game.deterministicMode = true; // Enable deterministic mode
  game.startingPlayer = startingPlayer;
  game.currentPlayer = startingPlayer;
  game.isAIGame = true;

  const moves = [];
  let stagnationCounter = 0;
  let reason = 'max_moves';

  const progressState = {
    red: { span: 0, touchesTop: false, touchesBottom: false },
    black: { span: 0, touchesLeft: false, touchesRight: false },
  };

  // Create AIs for both players
  const redAI = new TwixTAI(game, 'red');
  const blackAI = new TwixTAI(game, 'black');
  redAI.debugEnabled = false;
  blackAI.debugEnabled = false;
  redAI.rootDepth = depth;
  blackAI.rootDepth = depth;

  for (let turn = 0; turn < maxMoves; turn++) {
    const player = game.currentPlayer;
    const ai = player === 'red' ? redAI : blackAI;

    const move = ai.getBestMove();
    if (!move) {
      reason = 'no_moves';
      break;
    }

    const success = game.placePeg(move.row, move.col);
    if (!success) {
      reason = 'no_moves';
      break;
    }

    moves.push({
      turn,
      player,
      row: move.row,
      col: move.col,
    });

    if (game.gameOver) {
      reason = 'win';
      break;
    }

    // Progress tracking (same as selfPlay.js)
    let advanced = false;
    try {
      const metrics = componentMetrics(game, player) || {};
      if (player === 'red') {
        const span = Number(metrics.maxRowSpan) || 0;
        if (span > progressState.red.span) {
          progressState.red.span = span;
          advanced = true;
        }
        if (!progressState.red.touchesTop && metrics.touchesTop) {
          progressState.red.touchesTop = true;
          advanced = true;
        }
        if (!progressState.red.touchesBottom && metrics.touchesBottom) {
          progressState.red.touchesBottom = true;
          advanced = true;
        }
      } else {
        const span = Number(metrics.maxColSpan) || 0;
        if (span > progressState.black.span) {
          progressState.black.span = span;
          advanced = true;
        }
        if (!progressState.black.touchesLeft && metrics.touchesLeft) {
          progressState.black.touchesLeft = true;
          advanced = true;
        }
        if (!progressState.black.touchesRight && metrics.touchesRight) {
          progressState.black.touchesRight = true;
          advanced = true;
        }
      }
    } catch {
      // If component metrics fail, continue without stall detection
    }

    if (advanced) {
      stagnationCounter = 0;
    } else {
      stagnationCounter++;
      if (stagnationCounter >= stallLimit) {
        reason = 'stall';
        break;
      }
    }
  }

  // Clear cache between games
  clearComponentCache();

  // Determine winner
  let winner;
  if (reason === 'win') {
    winner = game.winner;
  } else {
    winner = 'draw';
  }

  return {
    winner,
    moves,
    totalMoves: moves.length,
    reason,
    startingPlayer,
    seed,
    depth,
  };
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
    const config = JSON.parse(inputData);
    const result = playDeterministicGame(config);
    console.log(JSON.stringify(result));
  } catch (e) {
    console.log(JSON.stringify({ error: e.message, stack: e.stack }));
    process.exit(1);
  }
}

main();
