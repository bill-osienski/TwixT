#!/bin/bash

# Build script for TwixT Metal GPU worker
# Usage: ./build-gpu.sh

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   TwixT GPU Worker Build Script (Metal on M3)           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Error: This script requires macOS with Metal support"
    exit 1
fi

# Check for Swift
if ! command -v swift &> /dev/null; then
    echo "❌ Error: Swift is not installed"
    echo "   Install Xcode Command Line Tools: xcode-select --install"
    exit 1
fi

echo "✓ Swift found: $(swift --version | head -n1)"

# Check for Metal support
echo ""
echo "Checking Metal support..."
if system_profiler SPDisplaysDataType | grep -q "Metal"; then
    echo "✓ Metal supported"
    GPU_INFO=$(system_profiler SPDisplaysDataType | grep -A 2 "Metal" | head -n3)
    echo "$GPU_INFO"
else
    echo "⚠️  Warning: Metal support not detected"
    echo "   GPU acceleration may not be available"
fi

# Navigate to Swift project directory
cd "$(dirname "$0")/TwixTMetalGPU"

echo ""
echo "Building Swift package..."
echo "─────────────────────────────────────────────────────────"

# Clean previous builds
echo "Cleaning previous builds..."
swift package clean

# Build release binary
echo "Building release binary..."
swift build -c release

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "Binary location: TwixTMetalGPU/.build/release/twixt-metal-worker"
    echo ""
    echo "Next steps:"
    echo "  1. Test the binary:"
    echo "     cd scripts/GPU_Training/TwixTMetalGPU && .build/release/twixt-metal-worker --games 5 --verbose"
    echo ""
    echo "  2. Run GPU-accelerated self-play (from project root):"
    echo "     node scripts/GPU_Training/selfPlayGPU.js --games 60 --workers 6 --depth 3"
    echo ""
    echo "  3. Benchmark GPU vs CPU:"
    echo "     cd scripts/GPU_Training/TwixTMetalGPU && make benchmark"
    echo ""
else
    echo ""
    echo "❌ Build failed"
    echo "Check the error messages above"
    exit 1
fi
