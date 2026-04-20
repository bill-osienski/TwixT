#!/bin/bash
# Overnight AlphaZero Training Script
#
# This script runs AlphaZero training with settings appropriate for
# an overnight run on Apple Silicon (MLX).
#
# Usage: ./scripts/train_overnight.sh
#
# Expected duration: 10-14 hours depending on hardware
# Expected output: ~50 checkpoints in checkpoints/alphazero/

set -e

# Configuration
ITERATIONS=50           # Number of training iterations
GAMES_PER_ITER=20       # Self-play games per iteration (1000 total games)
TRAIN_STEPS=100         # Training steps per iteration
BATCH_SIZE=64           # Batch size for training
MCTS_SIMS=100           # MCTS simulations per move (balanced for overnight)
MAX_MOVES=150           # Maximum moves per game
HIDDEN=128              # Network hidden channels (standard size)
BLOCKS=6                # Residual blocks (standard depth)
LR=0.001                # Learning rate
CHECKPOINT_DIR="checkpoints/alphazero"
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

# Create directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p logs

echo "=============================================="
echo "AlphaZero Overnight Training"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Iterations:      $ITERATIONS"
echo "  Games/iter:      $GAMES_PER_ITER"
echo "  Train steps:     $TRAIN_STEPS"
echo "  MCTS sims:       $MCTS_SIMS"
echo "  Network:         ${HIDDEN}ch x ${BLOCKS}blocks"
echo "  Checkpoint dir:  $CHECKPOINT_DIR"
echo "  Log file:        $LOG_FILE"
echo ""
echo "Expected: ~$(($ITERATIONS * $GAMES_PER_ITER)) games, ~$(($ITERATIONS * $GAMES_PER_ITER * 50)) positions"
echo ""
echo "Starting at $(date)"
echo "=============================================="
echo ""

# Change to project root and activate virtual environment
cd "$(dirname "$0")/.."
source .venv313/bin/activate

# Run training
python -m scripts.GPU.alphazero.train \
    --iterations $ITERATIONS \
    --games-per-iter $GAMES_PER_ITER \
    --train-steps $TRAIN_STEPS \
    --batch-size $BATCH_SIZE \
    --simulations $MCTS_SIMS \
    --max-moves $MAX_MOVES \
    --hidden $HIDDEN \
    --blocks $BLOCKS \
    --lr $LR \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Check the final checkpoint in $CHECKPOINT_DIR"
echo "2. Export to ONNX:"
echo "   python -m scripts.GPU.alphazero.export_onnx \\"
echo "       --weights $CHECKPOINT_DIR/model_iter_0050.safetensors \\"
echo "       --output model.onnx --hidden $HIDDEN --blocks $BLOCKS"
echo ""
echo "3. Start the server:"
echo "   MODEL_PATH=./model.onnx npm run server"
