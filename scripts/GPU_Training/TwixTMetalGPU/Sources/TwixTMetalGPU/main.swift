import Foundation
import ArgumentParser

struct TwixTMetalWorker: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "twixt-metal-worker",
        abstract: "GPU-accelerated TwixT self-play worker using Metal",
        version: "1.0.0"
    )

    @Option(name: .shortAndLong, help: "Number of self-play games to run")
    var games: Int = 10

    @Option(name: .shortAndLong, help: "Search depth per side")
    var depth: Int = 3

    @Option(name: .shortAndLong, help: "Output JSON file")
    var output: String = "selfplay-trace.json"

    @Flag(name: .long, help: "Print progress to stdout")
    var verbose: Bool = false

    @Option(name: .long, help: "Worker ID for parallel execution")
    var coreId: Int?

    @Option(name: .long, help: "Run ID for parallel execution")
    var runId: String?

    @Option(name: .long, help: "Path to value-model.json")
    var valueModel: String = "value-model.json"

    @Option(name: .long, help: "Path to search.json")
    var heuristicsConfig: String = "search.json"

    func run() throws {
        if verbose {
            print("[TwixTMetalWorker] Starting GPU-accelerated self-play")
            print("[TwixTMetalWorker] Games: \(games), Depth: \(depth)")
        }

        // Load heuristics configuration from search.json
        var heuristics = HeuristicsConfig.default
        if FileManager.default.fileExists(atPath: heuristicsConfig) {
            do {
                heuristics = try HeuristicsConfig.load(from: heuristicsConfig)
                if verbose {
                    print("[TwixTMetalWorker] Loaded heuristics from \(heuristicsConfig)")
                }
            } catch {
                if verbose {
                    print("[TwixTMetalWorker] Failed to load heuristics: \(error)")
                    print("[TwixTMetalWorker] Using default heuristics")
                }
            }
        } else if verbose {
            print("[TwixTMetalWorker] Search config not found at \(heuristicsConfig), using defaults")
        }

        // Load value model for CPU parity & optional GPU usage
        var valueModelEvaluator: ValueModelEvaluator? = nil
        if FileManager.default.fileExists(atPath: valueModel) {
            if let evaluator = ValueModelEvaluator.load(from: valueModel) {
                valueModelEvaluator = evaluator
                if verbose {
                    print("[TwixTMetalWorker] Loaded value model from \(valueModel)")
                }
            } else if verbose {
                print("[TwixTMetalWorker] Failed to parse value model at \(valueModel); continuing without it")
            }
        } else if verbose {
            print("[TwixTMetalWorker] Value model not found at \(valueModel), continuing without it")
        }

        // Initialize Metal AI (optional - fallback to CPU if not available)
        var ai: MetalAI? = nil
        do {
            ai = try MetalAI()

            // Load value model if available
            if FileManager.default.fileExists(atPath: valueModel) {
                try ai?.loadValueModel(from: valueModel)
            }
        } catch {
            if verbose {
                print("[TwixTMetalWorker] Metal initialization failed: \(error)")
                print("[TwixTMetalWorker] Continuing with CPU-based evaluation")
            }
        }

        // Create self-play engine
        let engine = SelfPlayEngine(
            ai: ai,
            searchDepth: depth,
            verbose: verbose,
            heuristics: heuristics,
            valueModel: valueModelEvaluator
        )

        // Check if running in parallel mode
        let isParallelMode = coreId != nil && runId != nil

        if isParallelMode {
            try runParallelMode(engine: engine)
        } else {
            try runStandaloneMode(engine: engine)
        }
    }

    private func runStandaloneMode(engine: SelfPlayEngine) throws {
        if verbose {
            print("[TwixTMetalWorker] Running in standalone mode")
        }

        // Run self-play games
        let traces = try engine.runSelfPlay(gameCount: games, startingGameNumber: 1)

        // Create output
        let output = SelfPlayOutput(
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            gameRequested: games,
            gameCompleted: traces.count,
            searchDepth: depth,
            aborted: false,
            games: traces
        )

        // Write to file
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(output)

        let outputPath = URL(fileURLWithPath: self.output)
        try data.write(to: outputPath)

        if verbose {
            print("[TwixTMetalWorker] Saved trace to \(outputPath.path)")
            print("[TwixTMetalWorker] Completed \(traces.count) games")
        }
    }

    private func runParallelMode(engine: SelfPlayEngine) throws {
        guard let coreId = coreId, let runId = runId else {
            throw ValidationError("Both --core-id and --run-id are required for parallel mode")
        }

        if verbose {
            print("[TwixTMetalWorker] Running in parallel mode (Core \(coreId), Run \(runId))")
        }

        // Create temp directory
        let tempDir = URL(fileURLWithPath: "temp/run-\(runId)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        // Output file for this worker
        let outputFile = tempDir.appendingPathComponent("temp-core-\(coreId).jsonl")

        // Open file for appending
        if !FileManager.default.fileExists(atPath: outputFile.path) {
            FileManager.default.createFile(atPath: outputFile.path, contents: nil)
        }

        guard let fileHandle = try? FileHandle(forWritingTo: outputFile) else {
            throw CocoaError(.fileWriteNoPermission)
        }

        defer {
            try? fileHandle.close()
        }

        // Run games and write to JSONL incrementally
        for seq in 1...games {
            if verbose {
                print("[Core \(coreId)] Playing game \(seq)/\(games)")
            }

            let trace = try engine.playGame(gameNumber: seq)

            // Create parallel worker output format
            let workerOutput = ParallelWorkerOutput(
                moves: trace.moves,
                summary: trace.summary,
                meta: GameMeta(
                    runId: runId,
                    coreId: coreId,
                    seq: seq,
                    depth: depth,
                    createdAt: ISO8601DateFormatter().string(from: Date()),
                    aborted: false,
                    draw: trace.summary.draw,
                    stalled: trace.stalled
                )
            )

            // Encode as single line JSON
            let encoder = JSONEncoder()
            let data = try encoder.encode(workerOutput)

            // Write line
            try fileHandle.seekToEnd()
            try fileHandle.write(contentsOf: data)
            try fileHandle.write(contentsOf: Data("\n".utf8))
            try fileHandle.synchronize()

            if verbose {
                print("[Core \(coreId)] Completed game \(seq): \(trace.summary.totalMoves) moves")
            }
        }

        if verbose {
            print("[Core \(coreId)] Finished \(games) games")
        }
    }
}

// Entry point
TwixTMetalWorker.main()
