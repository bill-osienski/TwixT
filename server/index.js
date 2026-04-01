/**
 * Express server for AlphaZero inference API.
 *
 * Endpoints:
 * - POST /api/move - Get best move with MCTS (HTTP fallback)
 * - POST /api/evaluate - Quick position evaluation (no MCTS)
 * - POST /api/analyze-position - Deterministic MCTS analysis for game recording
 * - GET /api/health - Health check
 * - WS /ws - WebSocket for real-time MCTS with progress streaming
 *
 * Difficulty levels map to MCTS simulation counts:
 * - easy: 100 simulations
 * - medium: 400 simulations
 * - hard: 800 simulations
 */
/* global AbortController */
import http from 'http';
import express from 'express';
import cors from 'cors';
import WebSocket, { WebSocketServer } from 'ws';
import { execSync } from 'child_process';
import { AlphaZeroInference } from './inference.js';
import { MCTS } from './mcts.js';
import { TwixtState } from './gameLogic.js';
import { BoardMovesCache } from './cache.js';

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// Global instances
let inference = null;
let cache = null;
let modelPath = null;

// Difficulty -> simulations and temperature mapping
const DIFFICULTY_PARAMS = {
  easy: { nSims: 100, moveTemp: 1.0 },
  medium: { nSims: 400, moveTemp: 0.5 },
  hard: { nSims: 800, moveTemp: 0.25 },
};

/**
 * Get best move for a position.
 *
 * Request body:
 *   - state: Serialized game state from TwixtState.toDict()
 *   - difficulty: "easy" | "medium" | "hard" (default: "medium")
 *   - temperature: Optional temperature for move selection (default by difficulty)
 *   - deterministicMode: If true, use temperature=0 (default: false)
 *
 * Response:
 *   - move: { row, col }
 *   - value: Root Q-value after MCTS
 *   - visits: Object mapping "row,col" -> visit count
 */
app.post('/api/move', async (req, res) => {
  try {
    const { state, difficulty = 'medium', temperature, deterministicMode = false } = req.body;

    if (!state) {
      return res.status(400).json({ error: 'Missing state in request body' });
    }

    // Reconstruct state from serialized dict
    const gameState = TwixtState.fromDict(state);
    const moves = gameState.legalMoves();

    if (moves.length === 0) {
      return res.json({ error: 'no_legal_moves' });
    }

    // Check cache
    const cached = cache.get(gameState.pegs, moves, gameState.boardSize);
    if (cached && !deterministicMode) {
      // Don't use cache in deterministic mode to ensure consistency
      return res.json({ ...cached, cached: true });
    }

    // Run MCTS
    const params = DIFFICULTY_PARAMS[difficulty] || DIFFICULTY_PARAMS.medium;
    const mcts = new MCTS(inference, { nSimulations: params.nSims });

    const startTime = Date.now();
    const { visitCounts, rootValue } = await mcts.search(gameState);
    const elapsed = Date.now() - startTime;

    // Select move with appropriate temperature
    let moveTemp;
    if (deterministicMode) {
      moveTemp = 0;
    } else if (temperature !== undefined) {
      moveTemp = temperature;
    } else {
      // Default temperature by difficulty
      moveTemp = difficulty === 'easy' ? 0.5 : 0.1;
    }

    const moveKey = mcts.selectMove(visitCounts, moveTemp);
    const [row, col] = moveKey.split(',').map(Number);

    // Convert visit counts to plain object
    const visits = {};
    for (const [key, count] of visitCounts) {
      visits[key] = count;
    }

    const result = {
      move: { row, col },
      value: rootValue,
      visits,
      elapsed,
    };

    // Cache result (only if not deterministic mode)
    if (!deterministicMode) {
      cache.set(gameState.pegs, moves, result, gameState.boardSize);
    }

    res.json(result);
  } catch (err) {
    console.error('Error in /api/move:', err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * Evaluate position without full MCTS (quick evaluation).
 * Used for win prediction bar updates.
 *
 * Request body:
 *   - state: Serialized game state from TwixtState.toDict()
 *
 * Response:
 *   - value: NN value estimate in [-1, 1]
 *   - terminal: true if game is over
 */
app.post('/api/evaluate', async (req, res) => {
  try {
    const { state } = req.body;

    if (!state) {
      return res.status(400).json({ error: 'Missing state in request body' });
    }

    const gameState = TwixtState.fromDict(state);
    const moves = gameState.legalMoves();

    if (gameState.isTerminal()) {
      // Terminal position - return exact value
      const winner = gameState.winner();
      let value;
      if (winner === null) {
        value = 0; // Draw
      } else if (winner === gameState.toMove) {
        value = 1;
      } else {
        value = -1;
      }
      return res.json({ value, terminal: true });
    }

    // Single NN evaluation (no MCTS for speed)
    const boardTensor = gameState.toTensorHWC();
    const { value } = await inference.evaluate(boardTensor, moves);

    // Convert to Red's perspective on server
    const evalToMove = gameState.toMove;
    const valueRed = evalToMove === 'red' ? value : -value;
    const valueRedClamped = Math.max(-1, Math.min(1, valueRed));

    res.json({
      value, // Original STM perspective
      evalToMove,
      valueRed: valueRedClamped, // Red perspective (for bar)
      terminal: false,
    });
  } catch (err) {
    console.error('Evaluate error:', err);
    res.status(500).json({ error: err.message });
  }
});

/**
 * Health check endpoint.
 */
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    modelLoaded: inference !== null && inference.session !== null,
    cacheSize: cache ? cache.size : 0,
    cacheHitRate: cache ? cache.hitRate.toFixed(3) : 0,
  });
});

/**
 * Model and engine metadata endpoint.
 */
app.get('/api/model-info', (req, res) => {
  let gitSha = 'unknown';
  try {
    gitSha = execSync('git rev-parse --short HEAD', { encoding: 'utf-8' }).trim();
  } catch { /* ignore */ }

  res.json({
    checkpoint_path: modelPath,
    model_iter: null,
    network: { hidden: 128, blocks: 6 },
    mcts_defaults: {
      eval_batch_size: 14,
      stall_flush_sims: 48,
      virtual_visits: 8,
    },
    engine: {
      server_git_sha: gitSha,
      server_build_utc: new Date().toISOString(),
      rules_version: 'twixt_v1',
      evaluator_device: 'cpu_onnx',
      dtype: 'float32',
    },
  });
});

/**
 * Analyze a position with deterministic MCTS (no noise, temp=0).
 * Returns the root value and top-K candidate moves.
 * Used by the GameRecorder to grade human moves.
 *
 * Request body:
 *   - state: Serialized game state {board_size, to_move, pegs, bridges, ply}
 *   - simulations: Number of MCTS simulations (10-1000, default 200)
 *   - top_k: Number of top candidates to return (1-50, default 10)
 *   - state_hash_client: Optional client-side state hash for determinism
 *
 * Response:
 *   - root_value: MCTS root Q-value (side-to-move perspective)
 *   - total_visits: Total visit count across all moves
 *   - top1_share: Visit share of the best move
 *   - topk_shares: Visit shares of top 5 moves
 *   - candidates: Array of {move, visit_count, visit_share, prior, child_q}
 *   - sims_used, seed_used, state_hash_server, elapsed_ms
 */
app.post('/api/analyze-position', async (req, res) => {
  const { state, state_hash_client = null } = req.body;
  if (!state) return res.status(400).json({ error: 'state required' });

  // Clamp inputs to safe ranges
  const simulations = Math.max(10, Math.min(Number(req.body.simulations) || 200, 1000));
  const top_k = Math.max(1, Math.min(Number(req.body.top_k) || 10, 50));

  // Validate state shape
  if (!state.board_size || !state.to_move || typeof state.pegs !== 'object') {
    return res.status(400).json({ error: 'invalid state shape: need board_size, to_move, pegs' });
  }

  const t0 = Date.now();

  // DETERMINISM: seed derived from state_hash + packet_source + sims_used
  // SERVER-AUTHORITATIVE: same position + same packet type + same sims = same seed
  const seedInput = `${state_hash_client || 'none'}_precompute_${simulations}`;
  let seedUsed = 0;
  for (let i = 0; i < seedInput.length; i++) seedUsed = ((seedUsed << 5) - seedUsed + seedInput.charCodeAt(i)) | 0;
  seedUsed = Math.abs(seedUsed);

  try {
    const gameState = TwixtState.fromDict(state);
    const mcts = new MCTS(inference, { nSimulations: simulations });
    const { visitCounts, rootValue } = await mcts.search(gameState);

    // Build sorted candidates from visit counts
    const totalVisits = Array.from(visitCounts.values()).reduce((a, b) => a + b, 0);
    const entries = Array.from(visitCounts.entries())
      .map(([moveKey, count]) => {
        const [r, c] = moveKey.split(',').map(Number);
        return {
          move: { row: r, col: c },
          visit_count: count,
          visit_share: totalVisits > 0 ? count / totalVisits : 0,
          prior: null,   // TODO v2: extract from root node
          child_q: null, // TODO v2: extract from root node
        };
      })
      .sort((a, b) => b.visit_count - a.visit_count);

    const candidates = entries.slice(0, top_k);
    const topkShares = candidates.map(c => c.visit_share);
    const top1Share = topkShares.length > 0 ? topkShares[0] : 0;

    // Compute server-side state hash if available
    const stateHashServer = gameState.zobristKey
      ? `z:${gameState.zobristKey.toString(16).padStart(16, '0')}`
      : null;

    res.json({
      root_value: rootValue,
      total_visits: totalVisits,
      top1_share: top1Share,
      topk_shares: topkShares.slice(0, 5),
      candidates,
      sims_used: simulations,
      seed_used: seedUsed,
      state_hash_server: stateHashServer,
      elapsed_ms: Date.now() - t0,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * Cache statistics endpoint.
 */
app.get('/api/stats', (req, res) => {
  res.json({
    cache: {
      size: cache.size,
      maxSize: cache.maxSize,
      hits: cache.hits,
      misses: cache.misses,
      hitRate: cache.hitRate.toFixed(3),
    },
  });
});

/**
 * Clear cache endpoint.
 */
app.post('/api/cache/clear', (req, res) => {
  cache.clear();
  res.json({ status: 'ok', message: 'Cache cleared' });
});

// ============================================================================
// WebSocket Server for real-time MCTS with progress streaming
// ============================================================================

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });

// Per-connection state: { activeId, controllers: Map<id, AbortController> }
const conns = new Map();

wss.on('connection', (ws) => {
  conns.set(ws, { activeId: null, controllers: new Map() });

  // Safe send: socket can close mid-search (catch rare send errors too)
  const safeSend = (obj) => {
    if (ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify(obj));
    } catch {
      // Ignore send errors on closed socket
    }
  };

  ws.on('close', () => {
    // IMPORTANT: abort all pending controllers to prevent leaks
    const cs = conns.get(ws);
    if (cs) {
      for (const ctrl of cs.controllers.values()) ctrl.abort();
      cs.controllers.clear();
    }
    conns.delete(ws);
  });

  ws.on('message', async (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw.toString());
    } catch {
      safeSend({ type: 'error', id: null, message: 'Invalid JSON' });
      return;
    }

    const cs = conns.get(ws);
    if (!cs) return;

    if (msg.type === 'cancel') {
      const ctrl = cs.controllers.get(msg.id);
      if (ctrl) ctrl.abort();
      cs.controllers.delete(msg.id);
      if (cs.activeId === msg.id) cs.activeId = null;
      return;
    }

    if (msg.type === 'move') {
      // Validate state before processing
      if (!msg.state || typeof msg.state !== 'object') {
        safeSend({ type: 'error', id: msg.id ?? null, message: 'Missing state' });
        return;
      }

      // Validate id and difficulty
      const id = typeof msg.id === 'string' ? msg.id : `anon_${Date.now()}`;
      const difficulty = ['easy', 'medium', 'hard'].includes(msg.difficulty)
        ? msg.difficulty
        : 'medium';

      // Latest-wins: abort previous active request
      if (cs.activeId) {
        const prev = cs.controllers.get(cs.activeId);
        if (prev) prev.abort();
        cs.controllers.delete(cs.activeId);
      }

      cs.activeId = id;
      const controller = new AbortController();
      cs.controllers.set(id, controller);

      // Echo toMove back so client doesn't have to track state
      const toMove = msg.state.to_move;

      try {
        const result = await computeBestMove(msg.state, difficulty, {
          signal: controller.signal,
          onProgress: (p) => {
            if (cs.activeId !== id || controller.signal.aborted) return;
            safeSend({ type: 'progress', id, toMove, ...p });
          },
        });

        // Double-check abort before sending result
        if (controller.signal.aborted || cs.activeId !== id) return;

        safeSend({ type: 'bestmove', id, toMove, ...result });
        if (cs.activeId === id) cs.activeId = null;
      } catch (e) {
        if (!controller.signal.aborted) {
          safeSend({ type: 'error', id, message: e.message });
        }
        if (cs.activeId === id) cs.activeId = null;
      } finally {
        cs.controllers.delete(id);
      }
    }
  });
});

/**
 * Compute best move with MCTS, supporting cancellation and progress.
 *
 * @param {object} stateDict - Serialized game state
 * @param {string} difficulty - "easy" | "medium" | "hard"
 * @param {object} opts - { signal, onProgress }
 * @returns {Promise<{move, value, elapsed, nSimulations}>}
 */
async function computeBestMove(stateDict, difficulty = 'medium', opts = {}) {
  const t0 = Date.now();
  const gameState = TwixtState.fromDict(stateDict);
  const { nSims, moveTemp } = DIFFICULTY_PARAMS[difficulty] || DIFFICULTY_PARAMS.medium;

  const mcts = new MCTS(inference, { nSimulations: nSims });
  const { visitCounts, rootValue } = await mcts.search(gameState, {
    signal: opts.signal,
    onProgress: opts.onProgress,
    progressEvery: 50,
    progressMinMs: 100,
  });

  // Handle aborted search (no visits completed)
  if (!visitCounts || visitCounts.size === 0) {
    throw new Error('Search aborted');
  }

  const moveKey = mcts.selectMove(visitCounts, moveTemp);
  const [row, col] = moveKey.split(',').map(Number);

  // Convert to Red's perspective on server
  const evalToMove = gameState.toMove;
  const valueRed = evalToMove === 'red' ? rootValue : -rootValue;
  const valueRedClamped = Math.max(-1, Math.min(1, valueRed));

  return {
    move: { row, col },
    evalToMove, // Which side was evaluated
    rootValue, // Original STM perspective (debug)
    valueRed: valueRedClamped, // Red perspective (for bar)
    elapsed: Date.now() - t0,
    nSimulations: nSims,
  };
}

// ============================================================================
// Server Startup
// ============================================================================

async function main() {
  modelPath = process.env.MODEL_PATH || './model.onnx';
  const port = process.env.PORT || 3001;
  const cacheSize = parseInt(process.env.CACHE_SIZE || '10000', 10);

  console.log('AlphaZero Inference Server');
  console.log('==========================');
  console.log(`Model path: ${modelPath}`);
  console.log(`Port: ${port}`);
  console.log(`Cache size: ${cacheSize}`);
  console.log();

  console.log('Loading ONNX model...');
  inference = new AlphaZeroInference(modelPath);
  await inference.load();

  cache = new BoardMovesCache(cacheSize);

  server.listen(port, () => {
    console.log();
    console.log(`Server running on http://localhost:${port}`);
    console.log(`WebSocket available at ws://localhost:${port}/ws`);
    console.log('Endpoints:');
    console.log('  POST /api/move             - Get best move (HTTP fallback)');
    console.log('  POST /api/evaluate         - Quick position evaluation');
    console.log('  POST /api/analyze-position - Deterministic MCTS analysis');
    console.log('  GET  /api/health           - Health check');
    console.log('  GET  /api/stats            - Cache statistics');
    console.log('  WS   /ws                   - WebSocket for real-time MCTS');
  });
}

main().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
