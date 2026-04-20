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

### First-Time Setup: Export the AI Model

The AI server requires an ONNX model exported from a trained checkpoint:

```bash
python3 -m scripts.GPU.alphazero.export_onnx --weights checkpoints/alphazero-fresh/model_iter_0168.safetensors --output server/model.onnx
```

After exporting, `npm start` will automatically use the model.

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

# Export trained model to ONNX for the game server
python3 -m scripts.GPU.alphazero.export_onnx --weights checkpoints/alphazero-fresh/model_iter_XXXX.safetensors --output server/model.onnx
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
- Ensure `server/model.onnx` exists (run the export command above)
- Check that port 3001 is not in use

### "AlphaZero server not available" in browser
- The game falls back to heuristic AI (weaker but functional)
- Start the AI server: `npm run server`

### Python import errors
- Activate your virtual environment: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
