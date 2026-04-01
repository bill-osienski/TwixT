/**
 * GameRecorder - Records Human vs AI games with per-move AI analysis.
 *
 * State machine: IDLE -> RECORDING -> SAVING -> IDLE
 * Analysis cache: LRU by state_hash, max 64 entries.
 */

const CACHE_MAX = 64;
const SIMS_BG = 200;
const SIMS_BG_OPENING = 100;  // When legal_moves > 400
const TOP_K = 10;
const UPGRADE_SUPPRESSED_PLIES = 2;

export class GameRecorder {
  constructor() {
    this.enabled = localStorage.getItem('twixt_recording') === 'true';
    this.state = 'IDLE';  // IDLE | RECORDING | SAVING
    this.moves = [];
    this.modelInfo = null;
    this.gameMetadata = null;
    this.analysisCache = new Map();  // state_hash -> analysis packet
    this.abortController = null;
    this.analysisStatus = 'idle';  // idle | analyzing | ready | cancelled
    this.onAnalysisStatusChange = null;  // callback for UI indicator
    this._packetCounter = 0;
    this._lastTurnStart = null;
    this._alphaZeroRef = null;
  }

  // --- Toggle ---
  get isEnabled() { return this.enabled; }

  toggle() {
    if (this.state === 'RECORDING') return this.enabled;  // locked during game
    this.enabled = !this.enabled;
    localStorage.setItem('twixt_recording', String(this.enabled));
    return this.enabled;
  }

  // --- Game lifecycle ---
  async startRecording(game, difficulty, alphaZeroClient) {
    if (!this.enabled || this.state !== 'IDLE') return;
    this.state = 'RECORDING';
    this.moves = [];
    this.analysisCache.clear();
    this._packetCounter = 0;
    this._alphaZeroRef = alphaZeroClient;

    // Fetch model info
    try {
      this.modelInfo = await alphaZeroClient.getModelInfo();
    } catch {
      this.modelInfo = null;
    }

    this.gameMetadata = {
      game_id: `manual_${new Date().toISOString().replace(/[:.]/g, '-')}`,
      created_utc: new Date().toISOString(),
      difficulty,
      human_player: game.currentPlayer === game.aiPlayer
        ? (game.currentPlayer === 'red' ? 'black' : 'red')
        : game.currentPlayer,
      ai_player: game.aiPlayer || 'black',
    };
  }

  // --- Precompute (human turn) ---
  async precomputeAnalysis(game, alphaZeroClient) {
    if (this.state !== 'RECORDING') return;

    // Single in-flight request: cancel any previous
    this._cancelPrecompute();

    const stateHash = this._getStateHash(game);
    if (!stateHash) return;

    // Check cache (keyed by state_hash, not ply)
    if (this.analysisCache.has(stateHash)) {
      this._setAnalysisStatus('ready');
      return;
    }

    // Determine sims (lower for openings with many legal moves)
    const legalMoves = game.getLegalMoves ? game.getLegalMoves() : [];
    const legalCount = legalMoves.length || 500;
    const simsBg = legalCount > 400 ? SIMS_BG_OPENING : SIMS_BG;

    this._setAnalysisStatus('analyzing');
    this.abortController = new AbortController();

    const startedUtc = new Date().toISOString();
    const requestId = `req_p${game.moveCount}_pre`;

    try {
      const result = await alphaZeroClient.analyzePosition(
        game, simsBg, TOP_K, stateHash, this.abortController.signal
      );

      const packet = {
        packet_id: `p${game.moveCount}_precompute`,
        source: 'precompute',
        intent: 'analysis_only',
        // Store both hashes: client (Zobrist) and server (canonical)
        state_hash: result.state_hash_server || stateHash,
        state_hash_client: stateHash,
        state_hash_server: result.state_hash_server || null,
        sims_used: result.sims_used || simsBg,
        seed_used: result.seed_used,
        status: 'completed',
        determinism: { add_noise: false, temperature: 0.0 },
        root: {
          root_value: result.root_value,
          abs_root_value: Math.abs(result.root_value),
          total_visits: result.total_visits,
          top1_share: result.top1_share,
          topk_shares: result.topk_shares,
        },
        candidates_topN: result.candidates || [],
        timing: {
          request_id: requestId,
          started_utc: startedUtc,
          ended_utc: new Date().toISOString(),
          elapsed_ms: result.elapsed_ms,
        },
      };

      // LRU eviction
      if (this.analysisCache.size >= CACHE_MAX) {
        const firstKey = this.analysisCache.keys().next().value;
        this.analysisCache.delete(firstKey);
      }
      this.analysisCache.set(stateHash, packet);
      this._setAnalysisStatus('ready');
    } catch (err) {
      if (err.name === 'AbortError') {
        // Store cancelled packet per spec (omit candidates/chosen_move_eval)
        const cancelledPacket = {
          packet_id: `p${game.moveCount}_precompute`,
          source: 'precompute',
          intent: 'analysis_only',
          state_hash: stateHash,
          status: 'cancelled',
          cancel_reason: 'human_moved',
          determinism: { add_noise: false, temperature: 0.0 },
          timing: {
            request_id: requestId,
            started_utc: startedUtc,
            ended_utc: new Date().toISOString(),
            elapsed_ms: Date.now() - new Date(startedUtc).getTime(),
          },
        };
        this.analysisCache.set(stateHash, cancelledPacket);
        this._setAnalysisStatus('cancelled');
      } else {
        // Store error packet
        const errorPacket = {
          packet_id: `p${game.moveCount}_precompute`,
          source: 'precompute',
          intent: 'analysis_only',
          state_hash: stateHash,
          status: 'error',
          error: { code: 'request_failed', message: err.message },
          timing: {
            request_id: requestId,
            started_utc: startedUtc,
            ended_utc: new Date().toISOString(),
            elapsed_ms: Date.now() - new Date(startedUtc).getTime(),
          },
        };
        this.analysisCache.set(stateHash, errorPacket);
        console.warn('GameRecorder: precompute failed', err);
        this._setAnalysisStatus('idle');
      }
    }
  }

  // --- Record a move ---
  recordMove(game, source, player, row, col, aiAnalysis = null) {
    if (this.state !== 'RECORDING') return;

    const stateHash = this._getStateHash(game);
    const turnStartTime = this._lastTurnStart || Date.now();
    const thinkMs = Date.now() - turnStartTime;

    const moveEntry = {
      ply: this.moves.length,
      to_move: player,
      move_played: { row, col },
      move_source: source === 'alphazero' ? 'ai' : 'human',
      t_utc: new Date().toISOString(),
      think_ms: thinkMs,
      state: {
        ply_in_state: game.moveCount - 1,
        legal_moves_count: game.getLegalMoves ? game.getLegalMoves().length : null,
        max_plies_limit: 420,
        state_hash: stateHash,
      },
      analysis: [],
    };

    // Attach precompute analysis for human moves
    if (source === 'human' && stateHash) {
      const cached = this.analysisCache.get(stateHash);
      if (cached && cached.status === 'completed') {
        // Compute chosen_move_eval
        const chosenEval = this._computeChosenMoveEval(cached, row, col);
        cached.chosen_move_eval = chosenEval;
        moveEntry.analysis.push({ ...cached });  // shallow copy to avoid mutation

        // Upgrade trigger: if human's move not in top-5 and past opening
        if (this.moves.length >= UPGRADE_SUPPRESSED_PLIES
            && chosenEval.rank_by_visits > 5) {
          this._fireUpgradeEval(game, stateHash, row, col, moveEntry);
        }
      } else if (cached) {
        // Cancelled or error -- still attach for completeness
        moveEntry.analysis.push({ ...cached });
      }
    }

    // Attach AI analysis
    if (source === 'alphazero' && aiAnalysis) {
      moveEntry.analysis.push(aiAnalysis);
    }

    this.moves.push(moveEntry);
    this._lastTurnStart = Date.now();
  }

  // --- Build chosen_move_eval ---
  _computeChosenMoveEval(packet, row, col) {
    const candidates = packet.candidates_topN || [];
    const totalVisits = packet.root?.total_visits || 0;

    // Find the chosen move in candidates
    let chosenCandidate = candidates.find(
      c => c.move.row === row && c.move.col === col
    );
    let inTopN = true;

    if (!chosenCandidate) {
      inTopN = false;
      chosenCandidate = {
        move: { row, col },
        visit_count: 0,
        visit_share: 0,
        in_topN: false,
      };
    }

    const bestShare = candidates.length > 0 ? candidates[0].visit_share : 0;

    return {
      move_played: { row, col },
      was_top1_by_visits: candidates.length > 0
        && candidates[0].move.row === row
        && candidates[0].move.col === col,
      rank_by_visits: inTopN
        ? candidates.findIndex(c => c.move.row === row && c.move.col === col) + 1
        : candidates.length + 1,
      chosen_visit_count: chosenCandidate.visit_count,
      chosen_visit_share: chosenCandidate.visit_share,
      delta_vs_best_share: bestShare - (chosenCandidate.visit_share || 0),
      in_topN: inTopN,
    };
  }

  // --- Build AI move analysis packet from WS response ---
  buildAIMovePacket(game, wsResponse) {
    const stateHash = this._getStateHash(game);
    const visits = wsResponse.visits || {};
    const totalVisits = Object.values(visits).reduce((a, b) => a + b, 0);

    // Build candidates from visits
    const candidates = Object.entries(visits)
      .map(([key, count]) => {
        const [r, c] = key.split(',').map(Number);
        return {
          move: { row: r, col: c },
          visit_count: count,
          visit_share: totalVisits > 0 ? count / totalVisits : 0,
          prior: null,
          child_q: null,
        };
      })
      .sort((a, b) => b.visit_count - a.visit_count)
      .slice(0, TOP_K);

    const topkShares = candidates.map(c => c.visit_share);

    // Build chosen_move_eval for the AI's own move
    const aiMove = wsResponse.move;
    const chosenEval = this._computeChosenMoveEval(
      { candidates_topN: candidates, root: { total_visits: totalVisits } },
      aiMove.row, aiMove.col
    );

    // Determine root_value from correct perspective
    const rootValue = wsResponse.evalToMove === wsResponse.toMove
      ? wsResponse.valueRed
      : -wsResponse.valueRed;

    return {
      packet_id: `p${game.moveCount}_ai_move`,
      source: 'ai_move',
      intent: 'played_move',
      state_hash: stateHash,
      sims_used: wsResponse.nSimulations,
      seed_used: null,
      status: 'completed',
      determinism: {
        add_noise: true,
        temperature: wsResponse.temperature || 0.25,
      },
      root: {
        root_value: rootValue,
        abs_root_value: Math.abs(rootValue),
        total_visits: totalVisits,
        top1_share: topkShares[0] || 0,
        topk_shares: topkShares.slice(0, 5),
      },
      chosen_move_eval: chosenEval,
      candidates_topN: candidates,
      timing: {
        request_id: `req_p${game.moveCount}_ai`,
        started_utc: null,
        ended_utc: new Date().toISOString(),
        elapsed_ms: wsResponse.elapsed,
      },
    };
  }

  // --- Upgrade eval (full sims, fires when human not in top-5) ---
  async _fireUpgradeEval(game, stateHash, row, col, moveEntry) {
    const diffParams = { easy: 100, medium: 400, hard: 800 };
    const fullSims = diffParams[this.gameMetadata?.difficulty] || 400;
    const requestId = `req_p${moveEntry.ply}_upg`;
    const startedUtc = new Date().toISOString();

    try {
      const result = await this._alphaZeroRef?.analyzePosition(
        game, fullSims, TOP_K, stateHash
      );
      if (!result) return;

      const upgradePacket = {
        packet_id: `p${moveEntry.ply}_upgrade`,
        upgrades_packet_id: `p${moveEntry.ply}_precompute`,
        source: 'upgrade',
        intent: 'analysis_only',
        state_hash: result.state_hash_server || stateHash,
        state_hash_client: stateHash,
        state_hash_server: result.state_hash_server || null,
        trigger: 'human_not_in_top5',
        sims_used: result.sims_used || fullSims,
        seed_used: result.seed_used,
        status: 'completed',
        determinism: { add_noise: false, temperature: 0.0 },
        root: {
          root_value: result.root_value,
          abs_root_value: Math.abs(result.root_value),
          total_visits: result.total_visits,
          top1_share: result.top1_share,
          topk_shares: result.topk_shares,
        },
        chosen_move_eval: this._computeChosenMoveEval(
          { candidates_topN: result.candidates, root: { total_visits: result.total_visits } },
          row, col
        ),
        candidates_topN: result.candidates || [],
        timing: {
          request_id: requestId,
          started_utc: startedUtc,
          ended_utc: new Date().toISOString(),
          elapsed_ms: result.elapsed_ms,
        },
      };
      moveEntry.analysis.push(upgradePacket);
    } catch (err) {
      console.warn('GameRecorder: upgrade eval failed', err);
    }
  }

  // --- Undo support ---
  truncateToMove(plyCount) {
    if (this.state !== 'RECORDING') return;
    // Truncate moves by ply count, but do NOT evict analysis cache --
    // entries are keyed by state_hash and reusable if the same position
    // reappears after undo + different move.
    this.moves = this.moves.slice(0, plyCount);
  }

  // --- Finalize and save ---
  async finalize(game, alphaZeroClient) {
    if (this.state !== 'RECORDING') return null;
    this.state = 'SAVING';
    this._cancelPrecompute();

    const gameJson = this._buildGameJson(game);

    try {
      const result = await alphaZeroClient.saveGame(gameJson);
      this.state = 'IDLE';
      return result;
    } catch (err) {
      console.error('GameRecorder: save failed', err);
      this.state = 'IDLE';
      return { ok: false, error: err.message };
    }
  }

  _buildGameJson(game) {
    const difficulty = this.gameMetadata?.difficulty || 'medium';
    const diffParams = { easy: 100, medium: 400, hard: 800 };

    return {
      format_version: 'twixt_manual_game_v1',
      game: {
        game_id: this.gameMetadata?.game_id,
        created_utc: this.gameMetadata?.created_utc,
        rules: {
          game: 'twixt',
          active_size: game.boardSize || 24,
          board_size: 24,
          encoding: 'row_col',
          max_plies_limit: 420,
        },
        model: this.modelInfo || {},
        mcts: {
          simulations: diffParams[difficulty] || 400,
          selection: { mode: 'argmax_visits', temperature: 0.0 },
        },
        difficulty,
        human_player: this.gameMetadata?.human_player || 'red',
        ai_player: this.gameMetadata?.ai_player || 'black',
        recording: {
          enabled: true,
          sims_bg: SIMS_BG,
          upgrade_trigger: 'human_not_in_top5',
          upgrade_suppressed_opening_plies: UPGRADE_SUPPRESSED_PLIES,
          candidates_topN_size: TOP_K,
          compress: false,
        },
        termination: {
          winner: game.winner || null,
          reason: game.winner ? 'win' : (game.gameOver ? 'draw' : null),
          final_ply: this.moves.length,
        },
      },
      engine: this.modelInfo?.engine || {},
      moves: this.moves,
    };
  }

  // --- Internal helpers ---
  _cancelPrecompute() {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  _setAnalysisStatus(status) {
    this.analysisStatus = status;
    if (this.onAnalysisStatusChange) {
      this.onAnalysisStatusChange(status);
    }
  }

  _getStateHash(game) {
    if (game.zKey) {
      return `z:${game.zKey.toString(16).padStart(16, '0')}`;
    }
    return null;
  }

  markTurnStart() {
    this._lastTurnStart = Date.now();
  }
}

export const gameRecorder = new GameRecorder();
