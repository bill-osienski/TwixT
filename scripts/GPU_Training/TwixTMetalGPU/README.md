# TwixT GPU Training Infrastructure

GPU-accelerated self-play training for TwixT AI using Apple Metal on M3 MacBook Pro.

## Directory Structure

```
TwixTMetalGPU/
├── .build/                    # Swift build artifacts (generated)
├── Sources/
│   └── TwixTMetalGPU/
│       ├── GameState.swift           # GPU-optimized game logic
│       ├── MetalAI.swift            # Metal AI engine
│       ├── SelfPlayEngine.swift     # Self-play orchestration
│       ├── main.swift               # CLI worker
│       └── Shaders/
│           └── MoveEvaluation.metal # GPU compute kernels
├── Package.swift              # Swift package config
├── Makefile                  # Build automation
└── README.md                 # This file

scripts/
├── selfPlayGPU.js            # GPU orchestrator (new)
├── selfPlayParallel.js       # CPU orchestrator (old)
├── selfPlay.js               # Single-core self-play
├── consolidator.js           # Merges parallel worker output
├── combineSelfPlay.js        # Combines multiple traces
├── selfplay_checker.js       # Validates output
└── train_value.py           # Trains value model from traces

Root files:
├── build-gpu.sh             # Automated build script
├── test-gpu.sh              # Test suite
├── GPU_QUICKSTART.md        # Getting started
├── GPU_IMPLEMENTATION_SUMMARY.md  # Technical details
└── value-model.json         # Trained model weights
```

## Quick Start

### 1. Build GPU Worker

```bash
cd /Users/bill/Desktop/TwixT_Game\ Claude
./build-gpu.sh
```

### 2. Test Setup

```bash
./test-gpu.sh
```

### 3. Run GPU Training

```bash
# GPU-accelerated (recommended)
node scripts/selfPlayGPU.js --games 100 --workers 6 --depth 3

# CPU fallback (if needed)
node scripts/selfPlayParallel.js -g 100 -d 3
```

### 4. Train Model

```bash
python3 scripts/train_value.py \
  --input selfplay.json \
  --output value-model.json \
  --standardize \
  --gamma 0.998 \
  --epochs 2000
```

## Performance

- **CPU (12 cores)**: 15-20 minutes for 60 games
- **GPU (18 cores)**: 3-5 minutes for 60 games
- **Speedup**: 3-5x faster

## Usage Examples

### Basic Training Run

```bash
node scripts/selfPlayGPU.js --games 60 --workers 6 --depth 3
```

### Overnight Training

```bash
caffeinate -i node scripts/selfPlayGPU.js --games 5000 --workers 8 --depth 3
```

### Background Execution

```bash
nohup node scripts/selfPlayGPU.js --games 1000 --workers 6 --depth 3 > training.log 2>&1 &
tail -f training.log
```

### Benchmark GPU vs CPU

```bash
cd TwixTMetalGPU
make benchmark
```

## Monitoring

### GPU Usage

```bash
# Terminal 1: Run training
node scripts/selfPlayGPU.js --games 100 --workers 6 --verbose

# Terminal 2: Monitor GPU
sudo powermetrics --samplers gpu_power -i 1000
```

### Activity Monitor

1. Open Activity Monitor
2. Window → GPU History
3. Should see 70-90% utilization during training

## Building

### Quick Build

```bash
cd TwixTMetalGPU
make build
```

### Manual Build

```bash
cd TwixTMetalGPU
swift build -c release
```

### Clean Build

```bash
cd TwixTMetalGPU
make clean
make build
```

## Troubleshooting

### "Swift not found"

```bash
xcode-select --install
```

### "Binary not found"

```bash
cd TwixTMetalGPU
swift build -c release
```

### Low Performance

- Check GPU utilization (should be 70-90%)
- Reduce worker count if too high
- Verify release build (not debug)

## Files Generated

- `selfplay.json` - Training data
- `value-model.json` - Trained weights
- `temp/run-{id}/` - Temporary worker files
- `.build/` - Swift build artifacts

## Configuration

### Worker Count (GPU)

- **M3 (18 cores)**: 6-8 workers
- **M3 Pro (18 cores)**: 8-12 workers
- **M3 Max (40 cores)**: 12-18 workers

### Search Depth

- **2**: Fast, lower quality (2-3 min / 100 games)
- **3**: Balanced (recommended) (4-6 min / 100 games)
- **4**: High quality, slow (10-15 min / 100 games)

## Documentation

- `GPU_QUICKSTART.md` - Getting started guide
- `GPU_IMPLEMENTATION_SUMMARY.md` - Complete technical overview
- This file - Reference documentation

## License

Same as parent project.
