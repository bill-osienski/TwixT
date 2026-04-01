# Human Game Recording Design Spec

## Goal

Record 1-Player (Human vs AI) games from the browser UI with per-move AI analysis, so human play can be studied and compared against the model's evaluation.

## Scope

- Human vs AI games only (1-Player mode)
- Player opts in via a UI toggle before the game starts
- Every move is recorded with the AI's evaluation of that position
- Output saved as JSON to `logs/human-games/`

---

## Architecture

Two main components:

1. **Browser-side `GameRecorder`** -- manages recording state, collects move data + analysis packets, sends the completed game to the server on game end.

2. **Server-side endpoints** -- all on the AI server (port 3001, Express with `express.json()` already available): `POST /api/analyze-position` for deterministic MCTS eval, `POST /api/save-game` for writing JSON to disk, `GET /api/model-info` for model metadata.

### Data Flow

```
Human's turn starts (recording on)
  -> Browser fires POST /api/analyze-position (200 sims, deterministic, no noise)
  -> Response cached in GameRecorder keyed by `state_hash` (not ply -- ply breaks on undo/redo)
  -> Human clicks a move
  -> GameRecorder attaches cached analysis as the move's analysis_packet
  -> Computes chosen_move_eval (rank, visit share, delta vs best)
  -> If human's move not in top-5 by visits -> fire upgrade eval (full sims)

AI's turn
  -> Browser calls existing POST /api/move (full sims per difficulty)
  -> Response already has value + visit distribution
  -> GameRecorder captures as the AI move's analysis_packet

Game ends
  -> GameRecorder finalizes JSON (termination, winner, final_ply)
  -> POST /api/save-game with the full JSON
  -> Server writes to logs/human-games/manual_YYYY-MM-DD_NNN.json
```

---

## UI

### Header Toggle

A "REC" button in the game header bar (next to New Game / Undo / Help / Replay). When active, displays a red dot indicator. State persists in localStorage across page reloads.

### Difficulty Modal

When recording is on, the difficulty modal shows a subtle note: "Recording enabled -- game will be saved when complete."

### During Gameplay

A small red dot in the game status bar as a persistent reminder that recording is active.

**Analysis status indicator**: When recording is on and it's the human's turn, a small indicator near the REC dot shows background analysis state:
- Spinner + "Analyzing..." while precompute is in-flight
- Checkmark + "Ready" when precompute completes

This is purely informational -- it never blocks the human from moving. If they move while "Analyzing..." is shown, the request is cancelled and the game proceeds normally.

### Game End

If recording is on, the winner modal includes a note confirming the game was saved (or an error message if save failed).

### Lock Behavior

Recording can only be toggled before a game starts. Once a game is in progress with recording on, it stays on until the game ends. This prevents partial recordings.

### Undo Behavior

When a player undoes during a recorded game:
- The GameRecorder truncates its `moves` array back to the undo point.
- Any analysis packets for the undone ply(s) are discarded.
- If undo goes past an AI move (undoing both the AI response and the human move that triggered it), both moves and their analysis packets are removed.
- The next move recorded starts from the reverted state.
- A cancelled precompute for the new (re-entered) human turn is re-triggered.

---

## Analysis Pipeline

### Human Turn (Background Precompute)

- Triggered when it becomes the human's turn and recording is on
- `POST /api/analyze-position` with current state, `simulations: 200` (server uses `add_noise: false`, `temperature: 0`, deterministic seed derived from position -- see Seed Strategy below)
- Server returns: `root_value`, `total_visits`, `top1_share`, `topk_shares`, `candidates_topN` (top 10 moves with visit_count, visit_share, prior, child_q)
- Result cached in GameRecorder keyed by `state_hash` (not ply -- ply breaks on undo/redo)
- If the human moves before analysis finishes, the in-flight request is cancelled via AbortController and the packet is stored with `"status": "cancelled"` and `"cancel_reason"` (`"human_moved"`, `"network"`, `"server_timeout"`). Schema rules for non-completed packets: `root.elapsed_ms` and `root.root_value` MAY be present (with `"partial": true` on root); `candidates_topN` and `chosen_move_eval` MUST be omitted. This keeps downstream parsers sane -- they can trust that `candidates_topN` only exists on completed packets.

### On Human Move Played

- Look up cached analysis for this ply
- Compute `chosen_move_eval`: where the human's move ranked in the AI's candidate list
  - `rank_by_visits`: position in visit-sorted list
  - `chosen_visit_share`: fraction of visits on human's move
  - `delta_vs_best_share`: gap between human's pick and AI's top move
  - `was_top1_by_visits`: boolean
- **Upgrade trigger**: if the human's move is not in top-5 by visits AND the precompute completed, fire a full-sims eval for that ply. Stored as a separate upgrade analysis packet. **Exception**: upgrade is suppressed for ply 0-1 (opening moves) since 200 sims across 500+ legal moves produces extremely diffuse visits where "not in top-5" is nearly guaranteed and uninformative.

### AI Turn

- The browser uses the WebSocket path (`/ws`) for AI moves, which currently returns `{ move, evalToMove, rootValue, valueRed, elapsed, nSimulations }` but does **not** include the visit distribution.
- **Modification needed**: The WebSocket `bestmove` response in `server/index.js` must be extended when the browser passes `"includeVisits": true` in the WS move request. The extended response adds these fields to the existing `bestmove` message:
  ```json
  {
    "visits": { "10,12": 73, "12,10": 56, "11,11": 41 },
    "n_legal_moves": 575,
    "topk_shares": [0.182, 0.141, 0.103, 0.061, 0.047]
  }
  ```
  `visits` is a dict mapping `"row,col"` string keys to integer visit counts (same format as the existing HTTP `/api/move` response). `n_legal_moves` is the count of legal moves at that position. `topk_shares` is the top-5 visit shares for quick access.
- GameRecorder restructures the response (including visits) into the same analysis_packet format, including `chosen_move_eval` for the AI's own move (rank=1, visit share, etc.) so AI and human packets are structurally identical for downstream comparison.
- `analysis_source: "ai_move"`, `intent: "played_move"`
- **Intentionally noisy**: AI move packets record `add_noise: true` and the difficulty's temperature (e.g., 0.25) because they capture what the AI actually did under game conditions, not a deterministic eval. This is by design -- comparing "AI played under noise" vs "human graded deterministically" is a valid and useful analysis axis. For future apples-to-apples comparison, a `source: "ai_move_deterministic"` packet can be added (see Future Work).

### Simulation Counts

- Background precompute: `sims_bg = min(200, sims_ai)`
- Upgrade (when triggered): full `sims_ai` from difficulty setting
- AI moves: full `sims_ai` from difficulty setting
- Every analysis packet stores `sims_used`

---

## Grading Human Moves

Human moves are graded by the AI's raw evaluation of the position -- the same way the AI would score any move. No special labeling system.

The grade is the `chosen_move_eval` block:
- `rank_by_visits`: 1 = human picked the AI's top choice, higher = further from AI's preference
- `chosen_visit_share`: how much of the search focused on that move
- `delta_vs_best_share`: gap between human's pick and the best move's share
- `was_top1_by_visits`: did the human match the AI's top pick?

This lets downstream analysis derive whatever labels it wants (blunder, inaccuracy, etc.) from the raw data.

---

## JSON Output Format

File: `logs/human-games/manual_YYYY-MM-DD_NNN.json`

```json
{
  "format_version": "twixt_manual_game_v1",
  "game": {
    "game_id": "manual_2026-04-01_001",
    "created_utc": "2026-04-01T12:34:56Z",
    "rules": {
      "game": "twixt",
      "active_size": 24,
      "board_size": 24,
      "encoding": "row_col",
      "max_plies_limit": 420
    },
    "model": {
      "checkpoint_path": "checkpoints/alphazero-fresh/model_iter_0799.safetensors",
      "model_iter": 799,
      "network": { "hidden": 128, "blocks": 6 }
    },
    "mcts": {
      "simulations": 400,
      "eval_batch_size": 14,
      "stall_flush_sims": 48,
      "virtual_visits": 8,
      "selection": { "mode": "argmax_visits", "temperature": 0.0 }
    },
    "difficulty": "medium",
    "human_player": "red",
    "ai_player": "black",
    "recording": {
      "enabled": true,
      "sims_bg": 200,
      "upgrade_trigger": "human_not_in_top5",
      "upgrade_suppressed_opening_plies": 2,
      "candidates_topN_size": 10,
      "compress": false
    },
    "termination": {
      "winner": "red",
      "reason": "win",
      "final_ply": 87
    }
  },

  "engine": {
    "server_git_sha": "abc123def",
    "server_build_utc": "2026-03-28T10:00:00Z",
    "rules_version": "twixt_v1",
    "mcts_config_hash": "sha256:7f3a...",
    "evaluator_device": "metal",
    "dtype": "float32"
  },

  "moves": [
    {
      "ply": 0,
      "to_move": "red",
      "move_played": { "row": 12, "col": 12 },
      "move_source": "human",
      "t_utc": "2026-04-01T12:35:02.341Z",
      "think_ms": 6341,
      "state": {
        "ply_in_state": 0,
        "legal_moves_count": 576,
        "max_plies_limit": 420,
        "state_hash": "z:0000000000000000"
      },
      "analysis": [
        {
          "packet_id": "p0_precompute",
          "source": "precompute",
          "intent": "analysis_only",
          "state_hash": "z:0000000000000000",
          "sims_used": 200,
          "seed_used": 738291,
          "status": "completed",
          "determinism": { "add_noise": false, "temperature": 0.0 },
          "root": {
            "root_value": -0.237,
            "abs_root_value": 0.237,
            "total_visits": 200,
            "top1_share": 0.182,
            "topk_shares": [0.182, 0.141, 0.103, 0.061, 0.047]
          },
          "chosen_move_eval": {
            "move_played": { "row": 12, "col": 12 },
            "was_top1_by_visits": false,
            "rank_by_visits": 3,
            "chosen_visit_count": 41,
            "chosen_visit_share": 0.103,
            "delta_vs_best_share": 0.079
          },
          "candidates_topN": [
            { "rank": 1, "move": { "row": 10, "col": 12 }, "visit_count": 73, "visit_share": 0.182, "prior": 0.0412, "child_q": -0.198 },
            { "rank": 2, "move": { "row": 12, "col": 10 }, "visit_count": 56, "visit_share": 0.141, "prior": 0.0389, "child_q": -0.211 }
          ],
          "timing": { "request_id": "req_p0_pre", "started_utc": "2026-04-01T12:34:56.000Z", "ended_utc": "2026-04-01T12:34:56.142Z", "elapsed_ms": 142 }
        },
        {
          "packet_id": "p0_upgrade",
          "upgrades_packet_id": "p0_precompute",
          "source": "upgrade",
          "intent": "analysis_only",
          "state_hash": "z:0000000000000000",
          "trigger": "human_not_in_top5",
          "sims_used": 400,
          "seed_used": 912847,
          "status": "completed",
          "determinism": { "add_noise": false, "temperature": 0.0 },
          "root": {
            "root_value": -0.198,
            "abs_root_value": 0.198,
            "total_visits": 400,
            "top1_share": 0.210,
            "topk_shares": [0.210, 0.162, 0.104, 0.058, 0.041]
          },
          "chosen_move_eval": {
            "move_played": { "row": 12, "col": 12 },
            "was_top1_by_visits": false,
            "rank_by_visits": 6,
            "chosen_visit_count": 16,
            "chosen_visit_share": 0.040,
            "delta_vs_best_share": 0.170
          },
          "candidates_topN": [
            { "rank": 1, "move": { "row": 10, "col": 12 }, "visit_count": 84, "visit_share": 0.210, "prior": 0.0412, "child_q": -0.166 }
          ],
          "timing": { "request_id": "req_p0_upg", "started_utc": "2026-04-01T12:35:02.400Z", "ended_utc": "2026-04-01T12:35:02.701Z", "elapsed_ms": 301 }
        }
      ]
    },
    {
      "ply": 1,
      "to_move": "black",
      "move_played": { "row": 11, "col": 11 },
      "move_source": "ai",
      "t_utc": "2026-04-01T12:35:03.112Z",
      "think_ms": 612,
      "state": {
        "ply_in_state": 1,
        "legal_moves_count": 575,
        "max_plies_limit": 420,
        "state_hash": "z:a3f8e2b1c9d04567"
      },
      "analysis": [
        {
          "packet_id": "p1_ai_move",
          "source": "ai_move",
          "intent": "played_move",
          "state_hash": "z:a3f8e2b1c9d04567",
          "sims_used": 400,
          "seed_used": 445102,
          "status": "completed",
          "determinism": { "add_noise": true, "temperature": 0.25 },
          "root": {
            "root_value": 0.112,
            "abs_root_value": 0.112,
            "total_visits": 400,
            "top1_share": 0.182,
            "topk_shares": [0.182, 0.141, 0.103, 0.061, 0.047]
          },
          "chosen_move_eval": {
            "move_played": { "row": 11, "col": 11 },
            "was_top1_by_visits": true,
            "rank_by_visits": 1,
            "chosen_visit_count": 73,
            "chosen_visit_share": 0.182,
            "delta_vs_best_share": 0.0
          },
          "candidates_topN": [
            { "rank": 1, "move": { "row": 11, "col": 11 }, "visit_count": 73, "visit_share": 0.182, "prior": 0.0398, "child_q": 0.105 }
          ],
          "timing": { "request_id": "req_p1_ai", "started_utc": "2026-04-01T12:35:02.500Z", "ended_utc": "2026-04-01T12:35:03.112Z", "elapsed_ms": 612 }
        }
      ]
    }
  ]
}
```

Key design decisions:
- **Analysis nested inside moves**: `moves[].analysis[]` array keeps packets co-located with their move, eliminating indexing bugs on undo/truncation
- Each packet has `source`: `"precompute"` / `"upgrade"` / `"ai_move"` and a unique `packet_id`
- Each packet has `intent`: `"analysis_only"` (precompute/upgrade) or `"played_move"` (ai_move) -- prevents accidentally comparing noisy AI-play packets to deterministic grading packets
- Upgrade packets reference their precompute via `upgrades_packet_id`
- `sims_used`, `seed_used`, and `determinism` (add_noise, temperature) on every packet for reproducibility
- `seed_used` is deterministic: `hash(state_hash, packet_source, sims_used)` -- reproducible but avoids identical tie-breaks across packet types
- `status` on every packet: `"completed"` / `"cancelled"` / `"error"` (with optional `error: {code, message}`)
- `timing` block on every packet: `request_id`, `started_utc`, `ended_utc`, `elapsed_ms`
- `t_utc` and `think_ms` on every move entry (correlates bad moves with time pressure, validates precompute finished before click)
- `state_hash` (Zobrist) per move for replay verification, AND mandatory inside each analysis packet (so packets remain self-contained even if moves structure is refactored later)
- `engine` section at top level (git SHA, rules version, MCTS config hash) for long-term debuggability
- `game.recording` block captures recording config (sims_bg, upgrade trigger, suppressed opening plies)
- Upgrade packets are optional (only when human picks outside top-5, suppressed for ply 0-1)
- `candidates_topN` stores top 10 moves. If the chosen move is not in top 10, its entry is appended with `"in_topN": false` so `chosen_move_eval` can always be recomputed from candidates
- `root.total_visits` and `root.top1_share` always present even if candidates are truncated
- Termination info lives only in `game.termination` (no duplicate block)
- Cancelled packets include `cancel_reason` (`"human_moved"`, `"network"`, `"server_timeout"`) and may include `"partial": true` root stats if any data arrived before cancellation
- Error packets include `"error": {"code": "...", "message": "..."}` for diagnostics (e.g., server timeout, model not loaded)

---

## Traceability Fields

These fields make datasets debuggable months later:

### `analysis_packet_id` (string)

Every analysis packet gets a unique ID (e.g., `"p0_precompute"`, `"p0_upgrade"`, `"p1_ai_move"`). Upgrade packets reference the precompute they're upgrading via `"upgrades_packet_id": "p0_precompute"`.

### `engine` section (top-level in JSON)

```json
"engine": {
  "server_git_sha": "abc123def",
  "server_build_utc": "2026-03-28T10:00:00Z",
  "rules_version": "twixt_v1",
  "mcts_config_hash": "sha256:7f3a...",
  "evaluator_device": "metal",
  "dtype": "float32"
}
```

Populated by the server at save time. Makes it possible to reproduce results or identify when a model/engine change caused a shift. `evaluator_device` and `dtype` catch the case where "same checkpoint" behaves differently across machines.

### `state_hash` per move

Each entry in the `moves` array includes a `state_hash` field so replays and analyzers can verify they're in sync with the recorded game state.

**Algorithm**: Zobrist hashing (XOR of per-peg random values), implemented in `assets/js/game/zobrist.js` (browser) and `scripts/GPU/alphazero/game/twixt_state.py` (server). Both implementations use the same seed table, producing identical hashes for the same board state. Format: `"z:"` prefix + 16 hex chars (64-bit hash). The hash covers: all placed pegs (position + color) + `to_move`. It does NOT include ply count or max_plies_limit (those are stored separately).

```json
{
  "ply": 0,
  "to_move": "red",
  "move_played": { "row": 12, "col": 12 },
  "move_source": "human",
  "state": {
    "ply_in_state": 0,
    "legal_moves_count": 576,
    "max_plies_limit": 420,
    "state_hash": "z:8a3f..."
  }
}
```

---

## Server Endpoints

### `POST /api/analyze-position` (AI server, port 3001)

Runs deterministic MCTS on a position without choosing a move.

The server always uses deterministic settings (`add_noise: false`, `temperature: 0`).

**Seed strategy**: `seed_used = hash(state_hash, packet_source, sims_used)`. This is deterministic (same position + same packet type = same seed = reproducible), but avoids identical tie-breaks across different packet types (precompute vs upgrade) for the same position. The `seed_used` is returned in the response and stored in every analysis packet.

**Request:**
```json
{
  "state": {
    "moves": [[12,12], [11,11]],
    "nextPlayer": "red",
    "activeSize": 24,
    "maxPliesLimit": 420
  },
  "state_hash_client": "z:a3f8e2b1c9d04567",
  "simulations": 200,
  "top_k": 10
}
```

The `state` field uses the same serialization as the existing `/api/move` endpoint: `moves` is the list of `[row, col]` pairs played so far, `nextPlayer` is whose turn it is, `activeSize` is the board dimension, and `maxPliesLimit` is the ply cap. The server reconstructs the full `TwixtState` from this. This matches the existing `TwixtState.toDict()` / `fromDict()` contract used by `alphaZeroClient.js`.

`state_hash_client` is optional: the browser's Zobrist hash of the position. The server computes its own hash after reconstruction and returns both in the response -- a mismatch indicates a state reconstruction bug. The response includes `"state_hash_server": "z:..."` for comparison.

**Response:**
```json
{
  "root_value": -0.237,
  "total_visits": 200,
  "top1_share": 0.182,
  "topk_shares": [0.182, 0.141, 0.103, 0.061, 0.047],
  "candidates": [
    { "move": {"row":10,"col":12}, "visit_count": 73, "visit_share": 0.182, "prior": 0.0412, "child_q": -0.198 },
    "..."
  ],
  "sims_used": 200,
  "seed_used": 738291,
  "state_hash_server": "z:a3f8e2b1c9d04567",
  "elapsed_ms": 142
}
```

Same computation as `/api/move` but with deterministic settings and no move selection. `state_hash_server` can be compared against `state_hash_client` from the request to verify state reconstruction integrity. `sims_used` is server-authoritative (not just echoing the request) so dynamic sims scaling, abs_floor clamping, or future throttling is accurately reflected.

### `POST /api/save-game` (AI server, port 3001)

Writes game JSON to disk. Lives on the AI server (Express) since it already has JSON body parsing and middleware -- the static server (port 5500) is a raw `http.createServer` with no POST handling.

**Request:** Full game JSON (body)

**Response:**
```json
{ "ok": true, "path": "logs/human-games/manual_2026-04-01_001_a1b2c3.json" }
```

**Filename**: `manual_YYYY-MM-DD_NNN_GAMEID.json` where NNN auto-increments per day and GAMEID is a short hash from `game.game_id` to avoid collisions if two sessions save simultaneously.

**Atomic writes**: The server writes to `*.tmp` first, then renames to the final filename. This prevents partial writes from appearing as valid game files.

### `GET /api/model-info` (AI server, port 3001)

Returns model metadata for the `game.model` section. The server reads this from its startup config: checkpoint path is known at launch (from `startServer.js` auto-detection or CLI arg), network architecture is fixed (`hidden: 128, blocks: 6`), and MCTS defaults come from the server's config constants.

**Response:**
```json
{
  "checkpoint_path": "checkpoints/alphazero-fresh/model_iter_0799.safetensors",
  "model_iter": 799,
  "network": { "hidden": 128, "blocks": 6 },
  "mcts_defaults": { "eval_batch_size": 14, "stall_flush_sims": 48, "virtual_visits": 8 }
}
```

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `assets/js/ui/gameRecorder.js` | Create | Recording state machine, analysis caching, JSON assembly, save trigger |
| `assets/js/game/gameController.js` | Modify | Hook GameRecorder into move flow, trigger analysis on human turn |
| `TwixT.html` | Modify | Add REC toggle button + red dot indicator |
| `server/index.js` | Modify | Add `POST /api/analyze-position`, `POST /api/save-game`, and `GET /api/model-info` |
| `logs/human-games/` | Create | Output directory for saved games |

---

## Guardrails

### Background Eval Guardrails

1. **Cancellation**: If human moves before precompute finishes, cancel via AbortController. Store `"status": "cancelled"` with `cancel_reason`.
2. **Upgrade is conditional**: Only fires when human's move is not in top-5 by visits AND precompute completed.
3. **Sim budget**: Background uses `min(200, sims_ai)`. Upgrade uses full `sims_ai`. **Opening cap**: if `legal_moves_count > 400`, use `min(100, sims_ai)` instead of 200 since visits get extremely diffuse and upgrade triggers become noisy.
4. **Single in-flight request**: Only 1 analyze request in-flight per client at a time. New request (e.g., from undo + re-enter human turn) cancels previous. Prevents overlapping requests from spam-clicks or rapid undo.
5. **Analysis cache**: LRU cache capped at 64 entries keyed by `state_hash`. Prevents memory bloat from repeated undo/redo.

### Recording Guardrails

1. **Lock during game**: Toggle only changeable before game starts. Prevents partial recordings.
2. **1-Player only**: Recording toggle only appears/functions in 1-Player (vs AI) mode.
3. **localStorage persistence**: Toggle state survives page reload.

### Size / Retention

1. **`candidates_topN_size`**: Configurable in recording block (default 10). Can be lowered to reduce file size.
2. **Compression**: `recording.compress: true` causes server to write `.json.gz` instead of `.json`. Default off for development, recommended on for long runs.
3. **No max ply cap** by default -- full games are recorded. Can add `recording.max_plies_saved` later if needed.

---

## Privacy

No user-identifying information is stored. Game files contain only board state, moves, and AI analysis. If accounts are added later, an optional `player_id` field can be added to `game` -- it MUST be omittable and MUST NOT be required by any downstream parser.

---

## Source of Truth

The **server is authoritative** for:
- `legal_moves_count` (computed from reconstructed state)
- `state_hash_server` (computed after reconstruction)
- `sims_used` (actual sims run, may differ from requested due to abs_floor/throttling)
- Rules interpretation (legal moves, terminal conditions)

Client-side fields (`state_hash_client`, requested `simulations`) are hints for debugging and integrity checks, not authoritative.

---

## Future Work (TODO)

- `source: "ai_move_deterministic"`: Optional second deterministic eval for AI move positions, enabling "AI played under noise" vs "AI would pick deterministically" comparison.
- **WS fallback**: If WebSocket `includeVisits` extension can't be deployed immediately, the recorder can fall back to calling `POST /api/analyze-position` for the AI move state with `sims_used = sims_ai` and treat it as the AI packet (deterministic). Slower, but keeps the recorder working without WS changes.
