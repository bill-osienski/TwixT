# Human Game Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record 1-Player (Human vs AI) TwixT games from the browser with per-move AI analysis, saving rich JSON to `logs/human-games/`.

**Architecture:** A browser-side `GameRecorder` module manages recording state and analysis caching. On the human's turn it fires a background `POST /api/analyze-position` for grading; on the AI's turn it captures the existing MCTS data from the extended WebSocket response. On game end, the full JSON is sent to `POST /api/save-game`. The server writes atomically to `logs/human-games/`.

**Tech Stack:** JavaScript (ES modules, browser + Node.js/Express), ONNX MCTS inference server, WebSocket, Zobrist hashing

**Spec:** `docs/superpowers/specs/2026-04-01-human-game-recording-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `server/index.js` | Modify | Add `POST /api/analyze-position`, `POST /api/save-game`, `GET /api/model-info` endpoints; extend WS `bestmove` with `visits`/`n_legal_moves`/`topk_shares` when `includeVisits` flag set |
| `assets/js/ui/gameRecorder.js` | Create | Recording state machine, analysis cache (LRU by state_hash), JSON assembly, AbortController management, save trigger |
| `assets/js/game/gameController.js` | Modify | Hook GameRecorder into move flow: trigger precompute on human turn, capture AI analysis, handle undo truncation, finalize on game end |
| `assets/js/ai/alphaZeroClient.js` | Modify | Add `analyzePosition(game, sims, topK)` method; extend `getMove()` to pass `includeVisits` flag and return extended response |
| `TwixT.html` | Modify | Add REC toggle button + analysis status indicator in header bar |
| `logs/human-games/.gitkeep` | Create | Empty directory for saved games |

---

### Task 1: Server endpoint -- `GET /api/model-info`

**Files:**
- Modify: `server/index.js` (add route after existing `/api/health`)

- [ ] **Step 1: Add the endpoint**

After the `/api/health` route in `server/index.js`, add:

```javascript
app.get('/api/model-info', (req, res) => {
  const cp = require('child_process');
  let gitSha = 'unknown';
  try {
    gitSha = cp.execSync('git rev-parse --short HEAD', { encoding: 'utf-8' }).trim();
  } catch { /* ignore */ }

  res.json({
    checkpoint_path: modelPath,  // already in scope from server init
    model_iter: null,            // TODO: parse from checkpoint filename if available
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
```

- [ ] **Step 2: Verify**

Run: `npm start` then `curl http://localhost:3001/api/model-info | jq .`
Expected: JSON with `checkpoint_path`, `network`, `engine` fields.

- [ ] **Step 3: Commit**

```bash
git add server/index.js
git commit -m "feat(recording): add GET /api/model-info endpoint"
```

---

### Task 2: Server endpoint -- `POST /api/analyze-position`

**Files:**
- Modify: `server/index.js` (add route after `/api/model-info`)

This endpoint runs deterministic MCTS (no noise, temp=0) and returns root value + top-K candidates. It reuses the existing `MCTS` class and `inference` object.

- [ ] **Step 1: Add the endpoint**

```javascript
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
  // This is SERVER-AUTHORITATIVE: even if client sends state_hash_client,
  // the server uses it only as input to the hash. Same position + same
  // packet type + same sims = same seed = reproducible results.
  // Different packet types (precompute vs upgrade) get different seeds,
  // avoiding identical tie-breaks.
  const seedInput = `${state_hash_client || 'none'}_precompute_${simulations}`;
  let seedUsed = 0;
  for (let i = 0; i < seedInput.length; i++) seedUsed = ((seedUsed << 5) - seedUsed + seedInput.charCodeAt(i)) | 0;
  seedUsed = Math.abs(seedUsed);

  try {
    const gameState = TwixtState.fromDict(state);
    const mcts = new MCTS(inference, { nSimulations: simulations });
    const { visitCounts, rootValue } = await mcts.search(gameState);

    // Compute state_hash on server side
    const stateHashServer = gameState.zobristKey
      ? `z:${gameState.zobristKey.toString(16).padStart(16, '0')}`
      : null;

    // Build sorted candidates
    const totalVisits = Array.from(visitCounts.values()).reduce((a, b) => a + b, 0);
    const entries = Array.from(visitCounts.entries())
      .map(([moveKey, count]) => {
        const [r, c] = moveKey.split(',').map(Number);
        const child = null; // access via root if available
        return {
          move: { row: r, col: c },
          visit_count: count,
          visit_share: totalVisits > 0 ? count / totalVisits : 0,
          prior: null,  // TODO: extract from root.priors if accessible
          child_q: null, // TODO: extract from root.children if accessible
        };
      })
      .sort((a, b) => b.visit_count - a.visit_count);

    const candidates = entries.slice(0, top_k);
    const topkShares = candidates.map(c => c.visit_share);
    const top1Share = topkShares.length > 0 ? topkShares[0] : 0;

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
```

**Note:** The `MCTS` class and `TwixtState` are already imported at the top of `server/index.js`. The `inference` object is available in module scope. The search runs with default PUCT (no noise, no Dirichlet -- the JS MCTS doesn't add noise by default). `temperature=0` is handled by not calling `selectMove`.

To extract `prior` and `child_q` from the MCTS root, we need access to the root node. The current `MCTS.search()` returns `{visitCounts, rootValue}` but not the root node. We have two options:
1. Modify `MCTS.search()` to also return the root node
2. Skip `prior`/`child_q` for v1 and add them later

**For v1: skip `prior`/`child_q`** (set to `null`). Add a TODO comment. This keeps the endpoint working without modifying the MCTS internals.

- [ ] **Step 2: Verify**

```bash
curl -X POST http://localhost:3001/api/analyze-position \
  -H 'Content-Type: application/json' \
  -d '{"state":{"board_size":24,"to_move":"red","pegs":{},"bridges":[],"ply":0},"simulations":50,"top_k":5}' | jq .
```

Expected: JSON with `root_value`, `total_visits` ~50, `candidates` array of 5 moves.

- [ ] **Step 3: Commit**

```bash
git add server/index.js
git commit -m "feat(recording): add POST /api/analyze-position endpoint"
```

---

### Task 3: Server endpoint -- `POST /api/save-game`

**Files:**
- Modify: `server/index.js`
- Create: `logs/human-games/.gitkeep`

- [ ] **Step 1: Create the output directory**

```bash
mkdir -p logs/human-games
touch logs/human-games/.gitkeep
```

- [ ] **Step 2: Add the endpoint**

```javascript
import { writeFile, rename, readdir, mkdir } from 'fs/promises';

// Body size limit for save-game (10MB should cover any game)
// Add this near the top of server/index.js where express.json() is configured:
// app.use(express.json({ limit: '10mb' }));

app.post('/api/save-game', async (req, res) => {
  const game = req.body;
  if (!game || !game.game || !game.game.game_id) {
    return res.status(400).json({ error: 'invalid game JSON' });
  }

  // Hardcoded output dir -- never accept client-provided path
  const gamesDir = join(__dirname, '..', 'logs', 'human-games');
  await mkdir(gamesDir, { recursive: true });

  // Build filename: manual_YYYY-MM-DD_NNN_GAMEID.json
  const now = new Date();
  const dateStr = now.toISOString().slice(0, 10);
  const gameIdShort = game.game.game_id.slice(-6);

  // Find next NNN for today
  const existing = await readdir(gamesDir).catch(() => []);
  const todayFiles = existing.filter(f => f.startsWith(`manual_${dateStr}_`));
  const nnn = String(todayFiles.length + 1).padStart(3, '0');

  const filename = `manual_${dateStr}_${nnn}_${gameIdShort}.json`;
  const tmpPath = join(gamesDir, `${filename}.tmp`);
  const finalPath = join(gamesDir, filename);

  // Atomic write: tmp then rename
  await writeFile(tmpPath, JSON.stringify(game, null, 2));
  await rename(tmpPath, finalPath);

  res.json({ ok: true, path: `logs/human-games/${filename}` });
});
```

**Note:** `__dirname` and `join` are already available in `server/index.js`. The `writeFile`, `rename`, `readdir`, `mkdir` imports from `fs/promises` need to be added at the top.

- [ ] **Step 3: Verify**

```bash
curl -X POST http://localhost:3001/api/save-game \
  -H 'Content-Type: application/json' \
  -d '{"game":{"game_id":"manual_2026-04-01_test"}}' | jq .
ls logs/human-games/
```

Expected: `{"ok":true,"path":"logs/human-games/manual_2026-04-01_001_01_test.json"}` and file exists.

- [ ] **Step 4: Commit**

```bash
git add server/index.js logs/human-games/.gitkeep
git commit -m "feat(recording): add POST /api/save-game endpoint with atomic writes"
```

---

### Task 4: Extend WebSocket `bestmove` response with visit data

**Files:**
- Modify: `server/index.js` (WebSocket handler)

- [ ] **Step 1: Extend the WS handler**

In the WebSocket `'move'` message handler, the client can now send `includeVisits: true`. After `computeBestMove()` returns, if the flag is set, add visit data to the `bestmove` response.

Find the section that sends the `bestmove` message (around the WS handler). Modify:

```javascript
// In the ws.on('message') handler, when processing type 'move':
const includeVisits = data.includeVisits === true;

// ... existing computeBestMove call ...

// Extend the bestmove response
const response = {
  type: 'bestmove',
  id: data.id,
  toMove: result.evalToMove,
  move: result.move,
  evalToMove: result.evalToMove,
  valueRed: result.valueRed,
  elapsed: result.elapsed,
  nSimulations: result.nSimulations,
};

if (includeVisits && result.visits) {
  response.visits = result.visits;  // { "row,col": count }
  response.n_legal_moves = Object.keys(result.visits).length;
  // Compute topk_shares
  const total = Object.values(result.visits).reduce((a, b) => a + b, 0);
  const shares = Object.values(result.visits)
    .map(v => total > 0 ? v / total : 0)
    .sort((a, b) => b - a);
  response.topk_shares = shares.slice(0, 5);
}

ws.send(JSON.stringify(response));
```

**Problem:** The current `computeBestMove()` does not return `visits`. It calls `mcts.search()` which returns `{visitCounts, rootValue}`, but `computeBestMove` only returns `{move, evalToMove, rootValue, valueRed, elapsed, nSimulations}`.

**Fix:** Modify `computeBestMove()` to also return `visits` when available. After the `mcts.search()` call, convert `visitCounts` Map to a plain object:

```javascript
// In computeBestMove(), after const { visitCounts, rootValue } = await mcts.search(...)
const visits = {};
for (const [key, count] of visitCounts) {
  visits[key] = count;
}
// Add to return value:
return { move, evalToMove, rootValue, valueRed, elapsed, nSimulations, visits };
```

- [ ] **Step 2: Verify**

Start server, open browser console, test with recording flag:
```javascript
// In browser console after connecting WS:
ws.send(JSON.stringify({type:'move', id:'test1', state: gameController.alphaZero._gameToServerState(gameController.game), difficulty:'easy', includeVisits:true}));
```

Expected: `bestmove` response includes `visits`, `n_legal_moves`, `topk_shares`.

- [ ] **Step 3: Commit**

```bash
git add server/index.js
git commit -m "feat(recording): extend WS bestmove with visits/n_legal_moves/topk_shares"
```

---

### Task 5: Create `GameRecorder` module

**Files:**
- Create: `assets/js/ui/gameRecorder.js`

This is the core browser-side module. It manages recording state, analysis caching, and JSON assembly.

- [ ] **Step 1: Create the module**

```javascript
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
  }

  // --- Toggle ---
  get isEnabled() { return this.enabled; }

  toggle() {
    if (this.state === 'RECORDING') return;  // locked during game
    this.enabled = !this.enabled;
    localStorage.setItem('twixt_recording', this.enabled);
    return this.enabled;
  }

  // --- Game lifecycle ---
  async startRecording(game, difficulty, alphaZeroClient) {
    if (!this.enabled || this.state !== 'IDLE') return;
    this.state = 'RECORDING';
    this.moves = [];
    this.analysisCache.clear();
    this._packetCounter = 0;

    this._alphaZeroRef = alphaZeroClient;  // Keep ref for upgrade evals

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

    // Cancel any in-flight request
    this._cancelPrecompute();

    const stateHash = game.zKey ? `z:${game.zKey.toString(16).padStart(16, '0')}` : null;
    if (!stateHash) return;

    // Check cache
    if (this.analysisCache.has(stateHash)) {
      this._setAnalysisStatus('ready');
      return;
    }

    // Determine sims
    const legalCount = game.getLegalMoves ? game.getLegalMoves().length : 500;
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
        // Store both hashes: client (Zobrist from browser) and server (canonical)
        // Prefer server hash when present; mismatch indicates a state reconstruction bug
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
          sims_used: simsBg,
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

    const stateHash = game.zKey ? `z:${game.zKey.toString(16).padStart(16, '0')}` : null;
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
        moveEntry.analysis.push(cached);

        // Upgrade trigger: if human's move not in top-5 and past opening
        if (this.moves.length >= UPGRADE_SUPPRESSED_PLIES
            && chosenEval.rank_by_visits > 5) {
          this._fireUpgradeEval(game, stateHash, row, col, moveEntry);
        }
      } else if (cached) {
        // Cancelled or error -- still attach for completeness
        moveEntry.analysis.push(cached);
      }
    }

    // Attach AI analysis
    if (source === 'alphazero' && aiAnalysis) {
      moveEntry.analysis.push(aiAnalysis);
    }

    this.moves.push(moveEntry);
    this._lastTurnStart = Date.now();
  }

  // --- Upgrade eval (full sims, fires when human not in top-5) ---
  async _fireUpgradeEval(game, stateHash, row, col, moveEntry) {
    // Use full difficulty sims
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
        state_hash: stateHash,
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
      // Move not in top-N; create a stub
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
      delta_vs_best_share: bestShare - chosenCandidate.visit_share,
      in_topN: inTopN,
    };
  }

  // --- Build AI move analysis packet from WS response ---
  buildAIMovePacket(game, wsResponse) {
    const stateHash = game.zKey ? `z:${game.zKey.toString(16).padStart(16, '0')}` : null;
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

    return {
      packet_id: `p${game.moveCount}_ai_move`,
      source: 'ai_move',
      intent: 'played_move',
      state_hash: stateHash,
      sims_used: wsResponse.nSimulations,
      seed_used: null,  // AI move uses server's own seed
      status: 'completed',
      determinism: {
        add_noise: true,
        temperature: wsResponse.temperature || 0.25,
      },
      root: {
        root_value: wsResponse.evalToMove === wsResponse.toMove
          ? wsResponse.valueRed  // Already from correct perspective
          : -wsResponse.valueRed,
        abs_root_value: Math.abs(wsResponse.valueRed),
        total_visits: totalVisits,
        top1_share: topkShares[0] || 0,
        topk_shares: topkShares.slice(0, 5),
      },
      chosen_move_eval: chosenEval,
      candidates_topN: candidates,
      timing: {
        request_id: `req_p${game.moveCount}_ai`,
        started_utc: null,  // Not tracked for WS
        ended_utc: new Date().toISOString(),
        elapsed_ms: wsResponse.elapsed,
      },
    };
  }

  // --- Undo support ---
  truncateToMove(plyCount) {
    if (this.state !== 'RECORDING') return;
    // Truncate moves array by ply count, but do NOT evict analysis cache
    // entries -- they're keyed by state_hash and reusable if the same
    // position reappears after undo + different move.
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

  markTurnStart() {
    this._lastTurnStart = Date.now();
  }
}

export const gameRecorder = new GameRecorder();
```

- [ ] **Step 2: Verify syntax**

```bash
node -e "import('./assets/js/ui/gameRecorder.js').then(() => console.log('OK')).catch(e => console.error(e.message))"
```

- [ ] **Step 3: Commit**

```bash
git add assets/js/ui/gameRecorder.js
git commit -m "feat(recording): create GameRecorder module with analysis caching and JSON assembly"
```

---

### Task 6: Add `analyzePosition()` and `saveGame()` to alphaZeroClient

**Files:**
- Modify: `assets/js/ai/alphaZeroClient.js`

- [ ] **Step 1: Add `analyzePosition()` method**

Add to the `AlphaZeroClient` class:

```javascript
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
```

- [ ] **Step 2: Add `saveGame()` method**

```javascript
async saveGame(gameJson) {
  const response = await fetch(`${this.serverUrl}/api/save-game`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(gameJson),
  });
  if (!response.ok) throw new Error(`save-game failed: ${response.status}`);
  return response.json();
}
```

- [ ] **Step 3: Add `getModelInfo()` method**

```javascript
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
```

- [ ] **Step 4: Extend `getMove()` to pass `includeVisits` and return extended data**

In the `getMove()` method (around line 199), when sending the WS message, add `includeVisits`:

```javascript
// Find the ws.send call in getMove/getBestMove
// Pass includeVisits flag -- caller sets this based on recording state
const msg = {
  type: 'move',
  id,
  state: this._gameToServerState(game),
  difficulty,
};
if (opts?.includeVisits) msg.includeVisits = true;
```

And in the promise resolution, pass through the extra fields:

```javascript
// In the onmessage handler for 'bestmove':
resolve({
  move: data.move,
  value: data.evalToMove,
  valueRed: data.valueRed,
  evalToMove: data.evalToMove,
  toMove: data.toMove,
  source: 'alphazero',
  elapsed: data.elapsed,
  nSimulations: data.nSimulations,
  // Extended fields for recording
  visits: data.visits || null,
  n_legal_moves: data.n_legal_moves || null,
  topk_shares: data.topk_shares || null,
});
```

- [ ] **Step 5: Commit**

```bash
git add assets/js/ai/alphaZeroClient.js
git commit -m "feat(recording): add analyzePosition, saveGame, getModelInfo to alphaZeroClient"
```

---

### Task 7: Hook GameRecorder into GameController

**Files:**
- Modify: `assets/js/game/gameController.js`

- [ ] **Step 1: Import and wire GameRecorder**

At the top of `gameController.js`, add:

```javascript
import { gameRecorder } from '../ui/gameRecorder.js';
```

- [ ] **Step 2: Hook into startGame()**

In the `startGame()` method, after `game.reset()` and game mode setup:

```javascript
// Start recording if enabled and 1-Player mode
if (isAI && gameRecorder.isEnabled) {
  await gameRecorder.startRecording(this.game, difficulty, this.alphaZero);
}
```

- [ ] **Step 3: Hook into human turn start**

After the AI makes its move (end of `makeAIMove()`), when control returns to the human, trigger precompute:

```javascript
// After AI move completes and it's human's turn
if (gameRecorder.state === 'RECORDING' && !this.game.gameOver) {
  gameRecorder.markTurnStart();
  gameRecorder.precomputeAnalysis(this.game, this.alphaZero);
}
```

Also trigger on game start if human goes first (in `startGame()`):

```javascript
if (isAI && gameRecorder.state === 'RECORDING' && this.game.currentPlayer !== this.game.aiPlayer) {
  gameRecorder.markTurnStart();
  gameRecorder.precomputeAnalysis(this.game, this.alphaZero);
}
```

- [ ] **Step 4: Hook into human move recording**

In `onPlayerMove()` or wherever the human's move is registered, before calling `recordMove()`:

```javascript
// Record to GameRecorder (human move)
if (gameRecorder.state === 'RECORDING') {
  gameRecorder.recordMove(this.game, 'human', player, row, col);
}
```

- [ ] **Step 5: Hook into AI move recording**

In `makeAIMove()`, pass `includeVisits` only when recording is active. Modify the `alphaZero.getMove()` call:

```javascript
const isRecording = gameRecorder.state === 'RECORDING';
const result = await alphaZero.getMove(game, difficulty, { includeVisits: isRecording });
```

Then, after the AI move result is received, build and attach the AI analysis packet:

```javascript
if (isRecording && result.visits) {
  const aiPacket = gameRecorder.buildAIMovePacket(this.game, result);
  gameRecorder.recordMove(this.game, 'alphazero', this.game.aiPlayer, result.move.row, result.move.col, aiPacket);
}
```

- [ ] **Step 6: Hook into undo**

In the `undo()` method, truncate the recorder:

```javascript
if (gameRecorder.state === 'RECORDING') {
  gameRecorder.truncateToMove(this.game.moveCount);
}
```

- [ ] **Step 7: Hook into game end**

In the win detection code (where `showWinner()` is called):

```javascript
if (gameRecorder.state === 'RECORDING') {
  const saveResult = await gameRecorder.finalize(this.game, this.alphaZero);
  // saveResult.ok and saveResult.path available for UI feedback
}
```

- [ ] **Step 8: Commit**

```bash
git add assets/js/game/gameController.js
git commit -m "feat(recording): hook GameRecorder into game controller move flow"
```

---

### Task 8: UI -- REC toggle + analysis status indicator

**Files:**
- Modify: `TwixT.html`
- Modify: `assets/js/game/gameController.js` (for event wiring)

- [ ] **Step 1: Add REC button and indicator to HTML**

In `TwixT.html`, after the existing control buttons (after the `<a id="replay-link">` element, around line 354):

```html
<button id="rec-toggle" aria-label="Toggle game recording" title="Record game">
  <span id="rec-dot" class="rec-dot"></span> REC
</button>
<span id="analysis-status" class="analysis-status"></span>
```

Add CSS (in the `<style>` block):

```css
.rec-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #666;
  margin-right: 4px;
  vertical-align: middle;
}

.rec-dot.active {
  background: #ff3333;
  animation: pulse-rec 1.5s ease-in-out infinite;
}

@keyframes pulse-rec {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.analysis-status {
  font-size: 0.75rem;
  color: #999;
  margin-left: 8px;
  vertical-align: middle;
}

.analysis-status.analyzing {
  color: #ffaa00;
}

.analysis-status.ready {
  color: #44cc44;
}
```

- [ ] **Step 2: Wire the toggle in gameController.js**

In the constructor or init section:

```javascript
// REC toggle
const recBtn = document.getElementById('rec-toggle');
const recDot = document.getElementById('rec-dot');
const analysisEl = document.getElementById('analysis-status');

if (recBtn) {
  // Init state from recorder
  if (gameRecorder.isEnabled) recDot.classList.add('active');

  recBtn.addEventListener('click', () => {
    const nowEnabled = gameRecorder.toggle();
    recDot.classList.toggle('active', nowEnabled);
  });

  // Analysis status callback
  gameRecorder.onAnalysisStatusChange = (status) => {
    analysisEl.textContent = status === 'analyzing' ? 'Analyzing...'
      : status === 'ready' ? 'Ready'
      : '';
    analysisEl.className = `analysis-status ${status}`;
  };
}
```

- [ ] **Step 3: Add recording note to difficulty modal**

In the difficulty modal section of `TwixT.html`, add a conditional note element:

```html
<p id="recording-note" class="recording-note" style="display:none;">
  Recording enabled - game will be saved when complete
</p>
```

In `gameController.js`, when the difficulty modal opens, show/hide the note:

```javascript
const recordingNote = document.getElementById('recording-note');
if (recordingNote) {
  recordingNote.style.display = gameRecorder.isEnabled ? 'block' : 'none';
}
```

- [ ] **Step 4: Commit**

```bash
git add TwixT.html assets/js/game/gameController.js
git commit -m "feat(recording): add REC toggle button and analysis status indicator to UI"
```

---

### Task 9: End-to-end smoke test

**Files:** None (verification only)

- [ ] **Step 1: Start the server**

```bash
npm start
```

- [ ] **Step 2: Open the game in browser**

Navigate to `http://localhost:5500/TwixT.html`

- [ ] **Step 3: Enable recording**

Click the REC button -- red dot should appear and pulse.

- [ ] **Step 4: Start a 1-Player game (Easy)**

Select 1 Player -> Easy. Should see "Recording enabled" note in difficulty modal.

- [ ] **Step 5: Play a few moves**

- Make a human move. Watch for "Analyzing..." -> "Ready" status near REC button.
- AI should respond with its move.
- Make another human move.
- Repeat 3-4 times.

- [ ] **Step 6: Check the saved game**

After the game ends (win/lose):
- Check `logs/human-games/` for a new JSON file.
- Verify the JSON structure matches the spec: `format_version`, `game`, `engine`, `moves` array with nested `analysis` arrays.
- Verify human moves have `analysis[].source = "precompute"`.
- Verify AI moves have `analysis[].source = "ai_move"` with `visits` data.

- [ ] **Step 7: Test undo**

Start a new recorded game, make 3 moves, undo, make a different move. End the game. Verify the saved JSON has correct ply sequence (no duplicates from the undone move).

- [ ] **Step 8: Test recording off**

Turn off REC. Play a game. Verify no file is saved to `logs/human-games/`.
