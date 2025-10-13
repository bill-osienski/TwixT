.PHONY: build clean test run benchmark install

# Build release binary
build:
	@echo "Building TwixT Metal GPU worker..."
	swift build -c release
	@echo "✓ Binary: .build/release/twixt-metal-worker"

# Clean build artifacts
clean:
	swift package clean
	rm -rf .build

# Run tests (when implemented)
test:
	swift test

# Run standalone self-play
run: build
	.build/release/twixt-metal-worker --games 10 --depth 3 --verbose

# Benchmark GPU vs CPU
benchmark: build
	@echo "=== CPU Baseline (12 cores) ==="
	@time node ../scripts/selfPlayParallel.js -g 30 -d 3
	@echo ""
	@echo "=== GPU Accelerated (18 GPU cores) ==="
	@time node ../scripts/selfPlayGPU.js --games 30 --workers 6 --depth 3
	@echo ""
	@echo "✓ Check selfplay.json for results"

# Install binary to PATH
install: build
	@echo "Installing twixt-metal-worker to /usr/local/bin..."
	@sudo cp .build/release/twixt-metal-worker /usr/local/bin/
	@echo "✓ Installed. Run: twixt-metal-worker --help"

# Check Metal support
check-metal:
	@echo "Checking Metal support..."
	@system_profiler SPDisplaysDataType | grep -A 3 "Metal"
	@echo ""
	@swift --version

# Profile GPU usage
profile: build
	@echo "Starting GPU profiling..."
	@echo "Run: node ../scripts/selfPlayGPU.js --games 20 --workers 4 --verbose"
	@echo "Monitoring GPU..."
	@sudo powermetrics --samplers gpu_power -i 1000
