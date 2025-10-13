# TwixT GPU Training System

GPU-accelerated self-play training for TwixT AI using Metal 4+ on Apple Silicon (M3).

## Quick Start

### 1. Build the GPU Worker (First Time Only)

```bash
cd /Users/bill/Desktop/Twixt_Game/scripts/GPU_Training
chmod +x build-gpu.sh
./build-gpu.sh
```

This compiles the Swift Metal worker binary. You only need to do this once, or when you update the Swift code.

### 2. Run Training

From the project root:

```bash
cd /Users/bill/Desktop/Twixt_Game
node scripts/GPU_Training/selfPlayGPU.js --games 480 --workers 6 --depth 3
```

## Command Line Arguments

### selfPlayGPU.js Options

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--games <number>` | `-g` | `60` | Total number of self-play games to generate (single depth mode) |
| `--depth <number>` | `-d` | `3` | Search depth per side (single depth mode) |
| `--depth-config <config>` | - | - | Multiple depths as `"depth:games,depth:games"` (e.g., `"2:240,3:240"`) |
| `--workers <number>` | `-w` | `6` | Number of parallel GPU workers per depth |
| `--verbose` | - | `false` | Print detailed progress to stdout |
| `--build` | - | `false` | Build Swift binary automatically if not found |

**Note:** Use either `--games`/`--depth` (single depth) OR `--depth-config` (multiple depths), not both.

### Examples

**Single Depth Mode:**

Quick test (2 games, depth 2):
```bash
node scripts/GPU_Training/selfPlayGPU.js --games 2 --workers 2 --depth 2 --verbose
```

Standard training batch (480 games, depth 3):
```bash
node scripts/GPU_Training/selfPlayGPU.js --games 480 --workers 6 --depth 3
```

Large batch (1000 games, depth 4):
```bash
node scripts/GPU_Training/selfPlayGPU.js --games 1000 --workers 8 --depth 4 --build
```

**Multiple Depth Mode:**

Balanced mix of depth 2 and 3 (240 games each):
```bash
node scripts/GPU_Training/selfPlayGPU.js --depth-config "2:240,3:240" --workers 6
```

Quick multi-depth test:
```bash
node scripts/GPU_Training/selfPlayGPU.js --depth-config "2:4,3:4" --workers 2 --verbose
```

Three depths for varied training (300 games at d2, 400 at d3, 300 at d4):
```bash
node scripts/GPU_Training/selfPlayGPU.js --depth-config "2:300,3:400,4:300" --workers 6
```

**Why use multiple depths?**
- Training diversity: Mix of fast/weak (d2) and slow/strong (d3-d4) games
- Better generalization for the AI
- Recommended ratio: 40-50% depth 2, 50-60% depth 3+

## How It Works

### Pipeline Overview

```
┌─────────────────┐
│  selfPlayGPU.js │  (Orchestrator)
└────────┬────────┘
         │ spawns
         ├──────────────┬──────────────┬──────────────┐
         ▼              ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ Worker 1│   │ Worker 2│   │ Worker 3│   │ Worker 4│
    │  (GPU)  │   │  (GPU)  │   │  (GPU)  │   │  (GPU)  │
    └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
         │             │             │             │
         ▼             ▼             ▼             ▼
    temp-core-1   temp-core-2   temp-core-3   temp-core-4
         │             │             │             │
         └─────────────┴─────────────┴─────────────┘
                       │
                       ▼
                ┌──────────────┐
                │ Consolidator │  (Merges temp files)
                └──────┬───────┘
                       │
                       ▼
                  selfplay.json  (Final output)
```

### Output Files

- **Main output**: `/Users/bill/Desktop/Twixt_Game/selfplay.json`
  - Automatically created if missing
  - Appends new games if already exists
  - This is the file used for training

- **Temp files**: `/Users/bill/Desktop/Twixt_Game/temp/run-{timestamp}/`
  - Individual worker outputs (JSONL format)
  - Kept for debugging
  - Safe to delete after successful run

### Configuration Files

The system uses these configuration files (automatically loaded):

- **`/Users/bill/Desktop/Twixt_Game/assets/js/ai/search.json`**
  - AI heuristics (connection weights, goal distance, etc.)
  - Edit this file to tune AI behavior between runs
  - No recompilation needed after changes

- **`/Users/bill/Desktop/Twixt_Game/value-mode.json`**
  - Trained neural network for position evaluation
  - Optional - system works without it
  - Uses only search.json heuristics if missing

## Performance Guidelines

### Recommended Settings

**Single Depth:**

| Use Case | Games | Workers | Depth | Time (approx) |
|----------|-------|---------|-------|---------------|
| Quick test | 10 | 2 | 2 | ~2 minutes |
| Small batch | 60 | 4 | 3 | ~15 minutes |
| Standard batch | 480 | 6 | 3 | ~2 hours |
| Large batch | 1000 | 8 | 4 | ~8 hours |

**Multiple Depths (Recommended for Training):**

| Use Case | Depth Config | Workers | Total Games | Time (approx) |
|----------|--------------|---------|-------------|---------------|
| Quick test | `"2:4,3:4"` | 2 | 8 | ~3 minutes |
| Small batch | `"2:30,3:30"` | 4 | 60 | ~20 minutes |
| Standard batch | `"2:240,3:240"` | 6 | 480 | ~3 hours |
| Large balanced | `"2:400,3:600"` | 6 | 1000 | ~6 hours |
| Three depths | `"2:200,3:400,4:200"` | 6 | 800 | ~8 hours |

### Worker Count Guidelines

- **M3 Pro (18 GPU cores)**: Use 6-8 workers
- **More workers = faster completion** (up to a point)
- Each worker uses GPU resources
- Monitor Activity Monitor to check GPU utilization

### Depth Guidelines

| Depth | Strength | Speed | Recommended For |
|-------|----------|-------|-----------------|
| 2 | Basic | Fast | Testing, debugging |
| 3 | Good | Moderate | Standard training |
| 4 | Strong | Slow | High-quality training |
| 5+ | Very strong | Very slow | Advanced training |

## Troubleshooting

### Swift Binary Not Found

**Error:**
```
✗ Swift binary not found at: /Users/bill/Desktop/Twixt_Game/scripts/GPU_Training/TwixTMetalGPU/.build/release/twixt-metal-worker
```

**Solution:**
```bash
cd /Users/bill/Desktop/Twixt_Game/scripts/GPU_Training
./build-gpu.sh
```

Or use `--build` flag:
```bash
node scripts/GPU_Training/selfPlayGPU.js --games 60 --workers 6 --depth 3 --build
```

### value-mode.json Not Found

**Warning:**
```
⚠️  value-mode.json not found at: /Users/bill/Desktop/Twixt_Game/value-mode.json
   GPU workers will use heuristics from search.json only
```

**This is OK!** The system will still run using only the heuristics from `search.json`. If you want to use the value model, make sure `value-mode.json` exists in the project root.

### Metal Initialization Failed

**Error:**
```
[MetalAI] Metal initialization failed: ...
[TwixTMetalWorker] Continuing with CPU-based evaluation
```

**Solution:** The system automatically falls back to CPU evaluation. This is slower but still works. Check that:
- You're running on Apple Silicon (M1/M2/M3)
- macOS is up to date
- Metal is supported (`system_profiler SPDisplaysDataType | grep Metal`)

### Games Hitting 220 Move Limit (Draws)

If many games end in draws at 220 moves, your heuristics may be too defensive. See "Tuning Heuristics" below.

## Tuning Heuristics

Edit `/Users/bill/Desktop/Twixt_Game/assets/js/ai/search.json` to adjust AI behavior.

**Key parameters:**

```json
{
  "valueModelScale": 600,
  "rewards": {
    "general": {
      "friendlyConnection": 12,      // Higher = more clustering
      "opponentConnection": 35,      // Higher = more defensive
      "goalDistance": 1.2,           // Higher = rush to edges
      "centerBias": 0.5,             // Higher = favor center
      "redGlobalMultiplier": 1.18,   // Adjust red strength
      "blackGlobalScale": 0.82       // Adjust black strength
    }
  }
}
```

**No recompilation needed!** Just edit the file and run training again.

For detailed tuning guide, see: `/Users/bill/Desktop/TwixT_Game Claude/HEURISTICS_TUNING_GUIDE.md`

## Monitoring Progress

### During Run

With `--verbose` flag, you'll see:
```
[Core 1] Playing game 5/10
[Game 5] Turn 30, red to move
[Game 5] Turn 40, black to move
...
[Core 1] Completed game 5: 87 moves
```

### After Completion

Check game count:
```bash
cd /Users/bill/Desktop/Twixt_Game
grep "gameCount" selfplay.json
```

Analyze results:
```bash
# Win rates
grep -o '"winner":"[^"]*"' selfplay.json | sort | uniq -c

# Average game length
grep -o '"totalMoves":[0-9]*' selfplay.json | \
  awk -F: '{sum+=$2; count++} END {print sum/count}'

# Draw count
grep '"draw":true' selfplay.json | wc -l
```

## System Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3) with Metal support
- **OS**: macOS 15.0+ (for Swift 6.0)
- **Software**:
  - Xcode Command Line Tools (`xcode-select --install`)
  - Node.js (for orchestrator script)
  - Swift 6.0+ (included with Xcode CLT)

## Advanced Usage

### Rebuild After Code Changes

If you modify Swift source files:

```bash
cd /Users/bill/Desktop/Twixt_Game/scripts/GPU_Training/TwixTMetalGPU
swift build -c release
```

### Clean Build

```bash
cd /Users/bill/Desktop/Twixt_Game/scripts/GPU_Training/TwixTMetalGPU
swift package clean
swift build -c release
```

### Standalone Worker Test

Test the Swift worker directly:

```bash
cd /Users/bill/Desktop/Twixt_Game/scripts/GPU_Training/TwixTMetalGPU
.build/release/twixt-metal-worker --games 2 --depth 2 --verbose
```

### Custom Paths

The orchestrator expects this directory structure:
```
/Users/bill/Desktop/Twixt_Game/
├── assets/js/ai/search.json          (heuristics)
├── value-mode.json                    (optional neural net)
├── selfplay.json                      (output)
├── temp/                              (worker temp files)
└── scripts/
    ├── consolidator.js                (merger)
    └── GPU_Training/
        ├── selfPlayGPU.js             (orchestrator)
        ├── build-gpu.sh               (build script)
        └── TwixTMetalGPU/             (Swift project)
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Run with `--verbose` to see detailed output
3. Check temp files in `temp/run-{timestamp}/` for worker errors
4. Verify Metal support: `system_profiler SPDisplaysDataType | grep Metal`

## Performance Notes

- **GPU utilization**: Monitor with Activity Monitor → GPU History
- **Memory usage**: ~29 GB GPU working set per worker
- **CPU usage**: Minimal (mainly orchestration)
- **Disk I/O**: Temp files written incrementally

The M3 Pro with 18 GPU cores can efficiently run 6-8 parallel workers, giving significant speedup over CPU-only training.
