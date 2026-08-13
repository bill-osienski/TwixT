# TwixT Game AI

A TwixT board game implementation with AlphaZero-style neural network AI featuring real-time evaluation and WebSocket communication.

## Quick Start

### Requirements

- **Node.js** v18+ (for game server and AI inference)
- **Python 3.10+** (for training and ONNX export)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies (for training/export)
pip install -r requirements.txt
```

### Running the Game

```bash
npm start
```

This starts:
- **Game server** on http://localhost:5500 (opens browser automatically)
- **AI server** on http://localhost:3001 (if ONNX model exists)

### The AI Model

The served model is **pinned**, committed, and validated at startup. No setup step
is required, and `npm start` never exports or selects a model.

The artifact lives under `models/<content-address>/` as three files: the ONNX
graph, its `model.onnx.data` weight sidecar, and a `manifest.json` recording both
SHA-256 hashes plus the tensor contract. `server/model_manifest.js` is the only
loading path, shared by `npm start` and `npm run server`. If either file is
missing or its hash does not match, startup fails loudly — it does not fall back
to another artifact and does not re-export one.

Startup checks more than the hashes. It parses the graph's external-data
references and requires every one to name the declared sidecar (a filename merely
appearing somewhere in the bytes proves nothing); it checks tensor names, types
and shapes against the runtime's own metadata; and it checks all of that against
the interface this server can actually consume, derived from `NUM_CHANNELS` and
`BOARD_SIZE` in `gameLogic.js`. That last step is separate on purpose — a model
and a manifest can agree perfectly with each other and still be unusable here.

The served board is the official 24×24. The engine supports other sizes for
curriculum training, but a model built for one is rejected rather than served.

To try a different model, stage it in its own `models/<content-address>/`
directory with its own manifest and point `MODEL_MANIFEST` at it:

```bash
MODEL_MANIFEST=models/<content-address>/manifest.json npm run server
```

The `model_id` is not chosen — it is `sha256(graph_sha256 + ":" +
external_data_sha256)` truncated to 16 hex, so it addresses the **pair**. Startup
rejects a manifest whose id does not follow from its own hashes. Derive it with
`computeModelId` from `server/model_manifest.js`.

Changing the default is a tracked edit to `DEFAULT_MODEL_ID` in
`server/model_manifest.js`, so the served model changes only when someone decides
it should.

**The current baseline's provenance is unknown** and is recorded as `unknown`
rather than guessed. Its identity is verified; no parity or playing-strength
claim is attached to it. See
`docs/superpowers/2026-08-13-product-model-alignment-decision-memo.md`.

## Architecture

### Frontend
- **Three.js r128** - 3D board rendering
- **Vanilla JavaScript** (ES modules)
- **WebSocket client** - Real-time AI communication with live evaluation bar

### Backend
- **Express.js** - HTTP API endpoints
- **WebSocket server** - Real-time MCTS progress streaming
- **ONNX Runtime** - Neural network inference

### AI System
- **AlphaZero-style MCTS** with neural network policy/value heads
- **Live evaluation bar** - Shows win probability during AI thinking
- **Request cancellation** - Undo during AI move cancels computation
- **Fallback to heuristics** - Works without AI server (weaker play)

## AI Training (AlphaZero)

The AI uses Monte Carlo Tree Search guided by a neural network trained via self-play.

### Training Requirements

```bash
pip install -r requirements.txt
# Requires: mlx, safetensors, numpy, torch, onnx
```

### Training Commands

```bash
# Start/resume AlphaZero training
python3 -m scripts.GPU.alphazero.train --iterations 200 --games 50

# Export a trained model to ONNX, into its own staging directory.
# Never export over a pinned model — see "The AI Model" above.
python3 -m scripts.GPU.alphazero.export_onnx \
  --weights checkpoints/alphazero-fresh/model_iter_XXXX.safetensors \
  --output models/staging-<name>/model.onnx
```

### Training Features

- **Curriculum learning** - Starts on 8x8, progresses to 24x24
- **Parallel self-play** - Multi-process game generation
- **GPU inference server** - Batched neural network evaluation
- **Automatic checkpointing** - Saves model every iteration

See `docs/alphazero-twixt.md` for detailed training documentation.

## Development

### Scripts

```bash
npm start          # Start game + AI servers
npm run server     # Start AI server only
npm test           # Run smoke tests
npm run lint       # Check code style
npm run lint:fix   # Auto-fix style issues
```

### Project Structure

```
assets/js/
├── ai/              # AI clients and heuristics
│   ├── alphaZeroClient.js  # WebSocket client for AI server
│   └── search.js           # Fallback heuristic AI
├── game/            # Game logic and rendering
└── ui/              # UI components (win bar, etc.)

server/
├── index.js         # Express + WebSocket server
├── mcts.js          # Monte Carlo Tree Search
├── inference.js     # ONNX model loading
└── model.onnx       # Exported neural network

scripts/GPU/alphazero/
├── train.py         # Main training loop
├── self_play.py     # Game generation
├── network.py       # Neural network (MLX)
└── export_onnx.py   # ONNX export
```

### Configuration

- **AI Parameters:** `assets/js/ai/search.json`
- **Difficulty levels:** easy (100 sims), medium (400 sims), hard (800 sims)

## Replay Viewer

Open `Replay.html` via the local server to load and step through saved games:

```bash
npm start
# Navigate to http://localhost:5500/Replay.html
```

## Troubleshooting

### AI server not starting
- Read the startup error: model validation names the exact failure
  (`MANIFEST_MISSING`, `GRAPH_MISSING`, `DATA_MISSING`, `GRAPH_HASH_MISMATCH`,
  `DATA_HASH_MISMATCH`, `EXTERNAL_REF_MISMATCH`, `CONTRACT_MISMATCH`, ...)
- Both files of the pair must be present and hash-match:
  `shasum -a 256 models/<content-address>/model.onnx*` against `manifest.json`
- Startup never re-exports a model to repair this; fix the artifact or the manifest
- Check that port 3001 is not in use

### "AlphaZero server not available" in browser
- The game falls back to heuristic AI (weaker but functional)
- Start the AI server: `npm run server`

### Python import errors
- Activate your virtual environment: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
