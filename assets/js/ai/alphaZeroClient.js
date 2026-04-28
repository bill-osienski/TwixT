/**
 * AlphaZero client with WebSocket for real-time progress and server fallback to heuristics.
 *
 * This client communicates with the AlphaZero inference server via WebSocket for
 * real-time progress updates, with HTTP fallback for availability checks.
 * Falls back to the heuristic-based TwixTAI when the server is unavailable.
 *
 * Usage:
 *   import { alphaZero } from './alphaZeroClient.js';
 *
 *   // Set progress callback for live eval bar
 *   alphaZero.onProgress = (msg) => winBar.update(msg.valueEstimate, msg.toMove);
 *
 *   // Get a move
 *   const result = await alphaZero.getMove(game, 'medium');
 *   // result = { move: {row, col}, value: number|null, toMove: string, source: 'alphazero'|'heuristics' }
 *
 *   // Cancel in-flight request (e.g., on undo)
 *   alphaZero.cancelLast();
 */

/* global WebSocket, AbortController */

import TwixTAI from './search.js';

export class AlphaZeroClient {
  constructor(serverUrl = 'http://localhost:3001') {
    this.serverUrl = serverUrl;
    this.timeout = 30000; // 30 second timeout for WebSocket moves
    // Hard mode runs 800 MCTS sims and contends the single ONNX session.
    // 500ms was too tight; the eval would AbortError every turn and the NN
    // win-bar would freeze on its last successful value. 3000ms gives
    // headroom; successful evals still resolve the moment the server replies.
    this.evalTimeout = 3000;
    this.available = null; // null = unknown, true/false = checked
    this.lastCheckTime = 0;
    this.checkInterval = 30000; // Re-check availability every 30 seconds

    // WebSocket state
    this.ws = null;
    this.reqCounter = 0;
    this.pending = new Map(); // id -> { resolve, reject }
    this.lastId = null;

    // Callback for progress updates (set by controller)
    this.onProgress = null;
  }

  /**
   * Convert HTTP URL to WebSocket URL.
   */
  _wsUrl() {
    return (
      this.serverUrl.replace('http://', 'ws://').replace('https://', 'wss://') +
      '/ws'
    );
  }

  /**
   * Ensure WebSocket is connected. Reuses existing connection if open.
   */
  _ensureWS() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this._wsUrl());

      this.ws.onopen = () => {
        this.available = true;
        resolve();
      };

      this.ws.onerror = () => {
        this.available = false;
        reject(new Error('WebSocket connection failed'));
      };

      // Clean up pending promises on close to prevent hung AI moves
      this.ws.onclose = () => {
        this.available = false;
        for (const [, p] of this.pending.entries()) {
          p.reject(new Error('WebSocket closed'));
        }
        this.pending.clear();
        this.lastId = null;
      };

      this.ws.onmessage = (ev) => {
        let msg;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }

        if (msg.type === 'progress' && this.onProgress) {
          // msg includes: done, total, elapsed, valueEstimate, toMove
          this.onProgress(msg);
          return;
        }

        if (msg.type === 'bestmove' || msg.type === 'error') {
          const p = this.pending.get(msg.id);
          if (!p) return;
          this.pending.delete(msg.id);
          if (msg.type === 'error') {
            p.reject(new Error(msg.message));
          } else {
            p.resolve(msg);
          }
        }
      };
    });
  }

  /**
   * Cancel the last in-flight request.
   * Called when user undoes a move or makes a new request.
   */
  cancelLast() {
    if (!this.lastId || !this.ws || this.ws.readyState !== WebSocket.OPEN)
      return;
    try {
      this.ws.send(JSON.stringify({ type: 'cancel', id: this.lastId }));
    } catch {
      // Ignore send errors
    }
    this.lastId = null;
  }

  /**
   * Convert TwixTGame to the state format expected by the server.
   * The server expects TwixtState.toDict() format.
   */
  _gameToServerState(game) {
    // Build pegs map: "row,col" -> player
    const pegs = {};
    for (const peg of game.pegs) {
      pegs[`${peg.row},${peg.col}`] = peg.player;
    }

    // Build bridges list: [[from, to], ...]
    const bridges = game.bridges.map((b) => [
      [b.from.row, b.from.col],
      [b.to.row, b.to.col],
    ]);

    return {
      board_size: game.boardSize,
      to_move: game.currentPlayer,
      pegs,
      bridges,
      ply: game.moveCount,
    };
  }

  /**
   * Check if the AlphaZero server is available.
   * Caches result for checkInterval ms to avoid spamming.
   */
  async checkAvailability() {
    const now = Date.now();

    // Use cached result if recent
    if (this.available !== null && now - this.lastCheckTime < this.checkInterval) {
      return this.available;
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1000);

      const response = await fetch(`${this.serverUrl}/api/health`, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      const data = await response.json();
      this.available = data.status === 'ok' && data.modelLoaded;
      this.lastCheckTime = now;

      if (this.available) {
        console.log('AlphaZero server available');
      }

      return this.available;
    } catch {
      this.available = false;
      this.lastCheckTime = now;
      return false;
    }
  }

  /**
   * Get best move from AlphaZero server via WebSocket, with fallback to heuristics.
   *
   * @param {TwixTGame} game - Current game state
   * @param {string} difficulty - 'easy' | 'medium' | 'hard'
   * @returns {Promise<{move: {row: number, col: number}, value: number|null, toMove: string, source: string}>}
   */
  async getMove(game, difficulty = 'medium', opts = {}) {
    // Cancel any previous in-flight request (latest-wins on client side too)
    this.cancelLast();

    const state = this._gameToServerState(game);
    const id = `req_${++this.reqCounter}`;
    this.lastId = id;

    try {
      await this._ensureWS();

      const promise = new Promise((resolve, reject) => {
        this.pending.set(id, { resolve, reject });

        // Timeout for the request
        setTimeout(() => {
          if (this.pending.has(id)) {
            this.pending.delete(id);
            reject(new Error('Request timeout'));
          }
        }, this.timeout);
      });

      const msg_out = { type: 'move', id, state, difficulty };
      if (opts?.includeVisits) msg_out.includeVisits = true;
      this.ws.send(JSON.stringify(msg_out));
      const msg = await promise;

      console.log(
        `AlphaZero move: (${msg.move.row}, ${msg.move.col}), ` +
          `value: ${msg.rootValue?.toFixed(3) ?? 'N/A'}, ` +
          `valueRed: ${msg.valueRed?.toFixed(3) ?? 'N/A'}, ` +
          `elapsed: ${msg.elapsed}ms, ` +
          `sims: ${msg.nSimulations}`
      );

      // msg includes: move, rootValue, valueRed, evalToMove, toMove, elapsed, nSimulations
      return {
        move: msg.move,
        value: msg.rootValue, // keep for debugging
        valueRed: msg.valueRed, // use for bar
        evalToMove: msg.evalToMove,
        toMove: msg.toMove,
        source: 'alphazero',
        visits: msg.visits || null,
        n_legal_moves: msg.n_legal_moves || null,
        topk_shares: msg.topk_shares || null,
      };
    } catch (err) {
      console.warn(
        'AlphaZero WebSocket unavailable, falling back to heuristics:',
        err.message
      );
      this.available = false; // Disable for future calls (will re-check after interval)
      return this._fallbackToHeuristics(game, difficulty);
    }
  }

  /**
   * Fallback to heuristic-based AI when server is unavailable.
   */
  _fallbackToHeuristics(game, difficulty) {
    const ai = new TwixTAI(game);
    const move = ai.getBestMove(difficulty);

    return {
      move,
      value: null, // Heuristics don't provide value
      toMove: game.currentPlayer,
      source: 'heuristics',
    };
  }

  /**
   * Evaluate position for win bar (no move selection).
   * Returns value in [-1, 1] where +1 = current player winning.
   *
   * @param {TwixTGame} game - Current game state
   * @returns {Promise<number|null>} Value or null if unavailable
   */
  async evaluate(game) {
    if (this.available === null) {
      await this.checkAvailability();
    }

    if (!this.available) {
      return null; // Can't evaluate without server
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.evalTimeout);

      const state = this._gameToServerState(game);

      const response = await fetch(`${this.serverUrl}/api/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        console.warn(`NN eval HTTP ${response.status} ${response.statusText}`);
        return null;
      }

      const data = await response.json();
      return data.valueRed; // Red-perspective value directly
    } catch (err) {
      // Distinguish timeout/abort from network/parse failures so the next
      // eval freeze is diagnosable from the browser console.
      if (err?.name === 'AbortError') {
        console.warn(`NN eval timed out after ${this.evalTimeout}ms`);
      } else {
        console.warn('NN eval request failed:', err?.message || err);
      }
      return null;
    }
  }

  /**
   * Force re-check of server availability.
   */
  async recheckAvailability() {
    this.lastCheckTime = 0;
    return this.checkAvailability();
  }

  /**
   * Analyze a position: run MCTS and return top-K move candidates with visit stats.
   *
   * @param {TwixTGame} game - Current game state
   * @param {number} simulations - Number of MCTS simulations
   * @param {number} topK - Number of top moves to return
   * @param {string|null} stateHashClient - Optional client-side state hash for caching
   * @param {AbortSignal|null} signal - Optional abort signal
   * @returns {Promise<Object>} Analysis result from server
   */
  async analyzePosition(game, simulations = 200, topK = 10, stateHashClient = null, signal = null) {
    const state = this._gameToServerState(game);
    const response = await fetch(`${this.serverUrl}/api/analyze-position`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state, simulations, top_k: topK, state_hash_client: stateHashClient }),
      signal,
    });
    if (!response.ok) throw new Error(`analyze-position failed: ${response.status}`);
    return response.json();
  }

  /**
   * Save a game record to the server.
   *
   * @param {Object} gameJson - Serialized game data
   * @returns {Promise<Object>} Server response
   */
  async saveGame(gameJson) {
    const response = await fetch(`${this.serverUrl}/api/save-game`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(gameJson),
    });
    if (!response.ok) throw new Error(`save-game failed: ${response.status}`);
    return response.json();
  }

  /**
   * Get model info from the server with a 2-second timeout.
   *
   * @returns {Promise<Object>} Model info from server
   */
  async getModelInfo() {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 2000);
    try {
      const response = await fetch(`${this.serverUrl}/api/model-info`, { signal: ac.signal });
      if (!response.ok) throw new Error(`model-info failed: ${response.status}`);
      return response.json();
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Check if AlphaZero is currently available.
   */
  isAvailable() {
    return this.available === true;
  }
}

// Singleton instance for easy import
export const alphaZero = new AlphaZeroClient();

export default AlphaZeroClient;
