import TwixTGame from '../assets/js/game/twixtGame.js';
import TwixTAI from '../assets/js/ai/search.js';
import { componentMetrics } from '../assets/js/ai/heuristics.js';
import { ensureValueModelLoaded } from '../assets/js/ai/valueModel.js';
import { program } from 'commander';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';

// ---------------- CLI ----------------
program
  .option('-g, --games <number>', 'number of self-play games', '10')
  .option('-d, --depth <number>', 'search depth per side', '3')
  .option('-o, --output <file>', 'output JSON file', 'selfplay-trace.json')
  .option('--verbose', 'print progress to stdout', false)
  .option('--core-id <number>', 'process identifier for temp files')
  .option('--run-id <string>', 'unique run identifier');

program.parse(process.argv);
const opts = program.opts();

const GAME_COUNT_RAW = parseInt(opts.games, 10) || 10;
const GAME_COUNT = GAME_COUNT_RAW;
const START_PLAN = process.env.START_PLAN
  ? process.env.START_PLAN.split(',').filter(Boolean)
  : null;
const usePlan = Array.isArray(START_PLAN) && START_PLAN.length === GAME_COUNT;
if (START_PLAN && !usePlan) {
  console.warn('[selfPlay] START_PLAN provided but length mismatch; falling back to alternating starts.');
}
const SEARCH_DEPTH = parseInt(opts.depth, 10) || 3;
const OUTPUT_FILE  = opts.output;
const VERBOSE      = !!opts.verbose;
const CORE_ID      = opts.coreId;
const RUN_ID       = opts.runId;

const traces = [];

// ---------------- Signal-aware shutdown ----------------
let stopRequested = false;  // break safely between turns/games
let aborted = false;        // for final logging/exit code

function requestShutdown(signal = 'SIGINT') {
  if (aborted) return;
  aborted = true;
  stopRequested = true;
  if (VERBOSE) console.log(`[selfPlay] received ${signal}, will stop after current move/game…`);
  // 130 = Ctrl+C (SIGINT), 143 = SIGTERM
  process.exitCode = (signal === 'SIGINT') ? 130 : 143;
}

process.once('SIGINT',  () => requestShutdown('SIGINT'));
process.once('SIGTERM', () => requestShutdown('SIGTERM'));
process.once('SIGQUIT', () => requestShutdown('SIGQUIT'));

// ---------------- macOS caffeinate ----------------
let caffeinateProc = null;

if (process.platform === 'darwin') {
  try {
    caffeinateProc = spawn('caffeinate', ['-im']);
    if (VERBOSE) console.log('[selfPlay] caffeinate started (-im)');
  } catch (err) {
    console.warn('[selfPlay] Failed to start caffeinate:', err);
  }
}

const cleanupCaffeinate = () => {
  if (caffeinateProc) {
    try { caffeinateProc.kill('SIGTERM'); } catch {}
    caffeinateProc = null;
    if (VERBOSE) console.log('[selfPlay] caffeinate stopped');
  }
};

process.on('exit', cleanupCaffeinate);

// ---------------- Helpers ----------------
function createAI(game, player) {
  const ai = new TwixTAI(game, player);
  if (typeof ai.setPlayer === 'function') {
    ai.setPlayer(player);
  }
  ai.debugEnabled = true;
  ai.moveTrace = [];
  ai.rootDepth = SEARCH_DEPTH;
  return ai;
}

function cloneBoard(board) {
  return board.map(row => [...row]);
}

function summarizeGame(game, draw = false) {
  return {
    boardSize: game.boardSize,
    totalMoves: game.moveCount,
    winner: draw ? null : game.winner,
    gameOver: draw ? false : game.gameOver,
    draw
  };
}

// Produce a lean representation of moves for storage (drops board/lastMoveTrace)
function toLeanMoves(moves) {
  if (!Array.isArray(moves)) return [];
  return moves.map((m) => {
    const player = m.player;
    const opp = player === 'red' ? 'black' : 'red';
    const ctx = m.featureContext ? { ...m.featureContext } : {};

    // Ensure peg counts exist; if missing, compute from board snapshot
    if ((ctx.playerPegCount == null || ctx.opponentPegCount == null) && Array.isArray(m.board)) {
      let playerCount = 0;
      let oppCount = 0;
      for (const row of m.board) {
        for (const cell of row) {
          if (cell === player) playerCount++;
          else if (cell === opp) oppCount++;
        }
      }
      if (ctx.playerPegCount == null) ctx.playerPegCount = playerCount;
      if (ctx.opponentPegCount == null) ctx.opponentPegCount = oppCount;
    }

    return {
      turn: m.turn,
      player: m.player,
      move: m.move,
      heuristics: m.heuristics ?? m.features ?? null,
      featureContext: ctx,
      valueModel: m.valueModel ?? null,
      heuristicScore: m.heuristicScore ?? null,
    };
  });
}

function playGame(gameNumber) {
  const game = new TwixTGame();
  const redAI = createAI(game, 'red');
  const blackAI = createAI(game, 'black');

  const gameTrace = {
    gameNumber,
    moves: [],
    summary: null,
    stalled: false
  };

  // Randomize starting color (tracks for analysis)
  const startsBlack = usePlan
    ? START_PLAN[gameNumber - 1] === 'black'
    : (gameNumber % 2 === 0);
  game.startingPlayer = startsBlack ? 'black' : 'red';
  game.currentPlayer = game.startingPlayer;
  game.isAIGame = true;

  // Record for trace consumers
  const startingPlayer = game.startingPlayer;

  // Allow extra room for AI endgames; human play usually ends earlier
  const maxMovesEnv = Number.parseInt(process.env.MAX_MOVES ?? '', 10);
  const maxMoves = Number.isFinite(maxMovesEnv) && maxMovesEnv > 0 ? maxMovesEnv : 220;
  const stallLimitEnv = Number.parseInt(process.env.STALL_LIMIT ?? '', 10);
  const stallLimit = Number.isFinite(stallLimitEnv) && stallLimitEnv > 0 ? stallLimitEnv : 40;
  let stagnationCounter = 0;
  let stalled = false;
  const progressState = {
    red: { span: 0, touchesTop: false, touchesBottom: false },
    black: { span: 0, touchesLeft: false, touchesRight: false }
  };

  for (let turn = 0; turn < maxMoves && !stopRequested; turn++) {
    const player = game.currentPlayer;
    const ai = player === 'red' ? redAI : blackAI;

    const move = ai.getBestMove();
    if (!move) break;

    const success = game.placePeg(move.row, move.col);
    if (!success) break; // avoid infinite loop on illegal move

    gameTrace.moves.push({
      turn,
      player,
      move,
      board: cloneBoard(game.board), // kept in-memory for counts; pruned on write
      heuristics: ai.lastChosenFeatures || {},
      featureContext: {
        ...(ai.lastChosenContext || {}),
        startingPlayer
      },
      valueModel: ai.lastChosenValueModel || null,
      heuristicScore: ai.lastChosenHeuristicScore ?? null,
      // lastMoveTrace intentionally omitted from persisted data
    });

    if (game.gameOver) break;

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
    } catch (_) {
      // If component metrics fail, ignore and continue without breaking.
    }

    if (advanced) {
      stagnationCounter = 0;
    } else {
      stagnationCounter++;
      if (stagnationCounter >= stallLimit) {
        stalled = true;
        break;
      }
    }
  }

  // Detect draw if we hit maxMoves without a winner
  const draw = stalled || !game.gameOver;
  const summary = summarizeGame(game, draw);
  summary.startingPlayer = startingPlayer;
  if (stalled) summary.stalled = true;
  gameTrace.summary = summary;
  gameTrace.stalled = stalled;
  return gameTrace;
}

// ---------------- Main ----------------
(async () => {
  try {
    try {
      await ensureValueModelLoaded();
    } catch (err) {
      console.warn('[selfPlay] Value model unavailable, continuing without it:', err?.message || err);
    }

    let completedGames = 0;

    for (let i = 1; i <= GAME_COUNT && !stopRequested; i++) {
      if (VERBOSE) {
        const gameLabel = CORE_ID ? `Core ${CORE_ID}: Game ${i}` : `Playing self-game ${i}`;
        console.log(`${gameLabel} / ${GAME_COUNT}`);
      }

      const trace = playGame(i);
      const isDraw = !!trace?.summary?.draw;
      const stalled = !!trace?.summary?.stalled;

      // In parallel mode, write each finished game immediately with provenance.
      if (CORE_ID && RUN_ID) {
        const gameData = {
          moves: toLeanMoves(trace.moves), // lean moves (no board snapshots)
          summary: trace.summary,
          meta: {
            runId: RUN_ID,
            coreId: parseInt(CORE_ID, 10),
            seq: i,
            depth: SEARCH_DEPTH,
            createdAt: new Date().toISOString(),
            aborted: false,
            draw: isDraw,
            stalled,
            startingPlayer: trace.summary?.startingPlayer ?? 'red'
          }
        };

        const tempPath = path.join('temp', `run-${RUN_ID}`, `temp-core-${CORE_ID}.jsonl`);
        const line = JSON.stringify(gameData) + '\n';

        await fs.promises.mkdir(path.dirname(tempPath), { recursive: true });
        const fh = await fs.promises.open(tempPath, 'a');
        await fh.write(line);
        await fh.sync();
        await fh.close();
      } else {
        // Standalone mode: keep in memory, write once at end (lean moves to keep file small)
        traces.push({
          ...trace,
          moves: toLeanMoves(trace.moves),
        });
      }

      completedGames++;

      if (VERBOSE) {
        const gameLabel = CORE_ID ? `Core ${CORE_ID}: Game ${i}` : `Game ${i}`;
        console.log(`${gameLabel} completed:`, trace.summary);
      }
    }

    // Only write output file in standalone mode
    if (!CORE_ID || !RUN_ID) {
      const outputPath = path.resolve(process.cwd(), OUTPUT_FILE);
      const payload = {
        generatedAt: new Date().toISOString(),
        gameRequested: GAME_COUNT,
        gameCompleted: traces.length,
        searchDepth: SEARCH_DEPTH,
        aborted,
        games: traces
      };
      await fs.promises.writeFile(outputPath, JSON.stringify(payload, null, 2));
      if (VERBOSE) {
        console.log(`Saved trace to ${outputPath}`);
      }
    } else if (VERBOSE) {
      console.log(
        `Core ${CORE_ID}: Finished ${completedGames} of ${GAME_COUNT} games (stopRequested=${stopRequested}, aborted=${aborted})`
      );
    }

    if (process.exitCode == null) {
      process.exitCode = aborted ? 130 : 0;
    }

    if (VERBOSE) {
      if (aborted) console.log('[selfPlay] shutdown complete (aborted by signal).');
      else console.log('[selfPlay] all requested games completed.');
    }

    cleanupCaffeinate();
    process.exit(process.exitCode ?? 0);

  } catch (err) {
    console.error('[selfPlay] fatal error:', err?.stack || err);
    cleanupCaffeinate();
    if (process.exitCode == null) process.exitCode = 1;
    process.exit(process.exitCode);
  }
})();
