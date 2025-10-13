import Foundation
import Metal

/// GPU-accelerated TwixT AI using Metal 4 compute shaders
/// Optimized for Apple Silicon M3 Pro (18 GPU cores)
class MetalAI {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary

    // Compute pipelines
    let evaluateMovePipeline: MTLComputePipelineState
    let batchEvaluatePipeline: MTLComputePipelineState
    let valueModelPipeline: MTLComputePipelineState

    // Value model weights (loaded from value-model.json)
    var valueModelWeights: [Float]?
    var featureCount: Int = 0

    init() throws {
        // Get default Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw AIError.noMetalDevice
        }
        self.device = device

        print("[MetalAI] Using GPU: \(device.name)")

        guard let queue = device.makeCommandQueue() else {
            throw AIError.failedToCreateCommandQueue
        }
        self.commandQueue = queue

        // Load Metal shader library
        // Try default library first (for precompiled shaders)
        var library: MTLLibrary? = device.makeDefaultLibrary()

        // If default library fails, compile from source (Swift Package Manager workaround)
        if library == nil {
            print("[MetalAI] Default library not found, compiling shaders from source...")

            // Find the Metal source file in the bundle
            guard let bundleURL = Bundle.module.url(forResource: "MoveEvaluation", withExtension: "metal") else {
                throw AIError.failedToLoadShaderLibrary
            }

            guard let metalSource = try? String(contentsOf: bundleURL, encoding: .utf8) else {
                throw AIError.failedToLoadShaderLibrary
            }

            // Compile Metal source at runtime
            do {
                library = try device.makeLibrary(source: metalSource, options: nil)
                print("[MetalAI] Successfully compiled shaders from source")
            } catch {
                print("[MetalAI] Shader compilation failed: \(error)")
                throw AIError.failedToLoadShaderLibrary
            }
        }

        guard let library = library else {
            throw AIError.failedToLoadShaderLibrary
        }
        self.library = library

        // Create compute pipelines
        guard let evaluateMoveFunc = library.makeFunction(name: "evaluateMoves"),
              let evaluateMovePipeline = try? device.makeComputePipelineState(function: evaluateMoveFunc) else {
            throw AIError.failedToCreatePipeline("evaluateMoves")
        }
        self.evaluateMovePipeline = evaluateMovePipeline

        guard let batchEvalFunc = library.makeFunction(name: "batchEvaluatePositions"),
              let batchEvalPipeline = try? device.makeComputePipelineState(function: batchEvalFunc) else {
            throw AIError.failedToCreatePipeline("batchEvaluatePositions")
        }
        self.batchEvaluatePipeline = batchEvalPipeline

        guard let valueModelFunc = library.makeFunction(name: "evaluateValueModel"),
              let valueModelPipeline = try? device.makeComputePipelineState(function: valueModelFunc) else {
            throw AIError.failedToCreatePipeline("evaluateValueModel")
        }
        self.valueModelPipeline = valueModelPipeline

        print("[MetalAI] Initialized successfully with \(device.recommendedMaxWorkingSetSize / 1024 / 1024) MB working set")
    }

    enum AIError: Error {
        case noMetalDevice
        case failedToCreateCommandQueue
        case failedToLoadShaderLibrary
        case failedToCreatePipeline(String)
        case failedToCreateBuffer
        case gpuExecutionFailed
    }

    /// Load value model weights from JSON
    func loadValueModel(from path: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let model = try JSONDecoder().decode(ValueModel.self, from: data)

        self.valueModelWeights = model.weights.map { Float($0) }
        self.featureCount = model.feature_keys.count

        print("[MetalAI] Loaded value model: \(model.type), \(featureCount) features, \(model.metrics.test_accuracy * 100)% accuracy")
    }

    struct ValueModel: Codable {
        let type: String
        let feature_keys: [String]
        let weights: [Double]
        let metrics: Metrics

        struct Metrics: Codable {
            let test_accuracy: Double
            let train_accuracy: Double
        }
    }

    /// Evaluate all valid moves on GPU and return scored moves
    func evaluateMoves(game: GameState) throws -> [(move: GameState.Move, score: Float)] {
        let validMoves = game.getValidMoves()
        guard !validMoves.isEmpty else {
            return []
        }

        // Prepare input buffers (Metal 4: Use .storageModeShared for Apple Silicon unified memory)
        let boardBuffer = device.makeBuffer(bytes: game.board,
                                            length: game.board.count * MemoryLayout<UInt8>.stride,
                                            options: [.storageModeShared, .cpuCacheModeWriteCombined])

        // Flatten moves to [row0, col0, row1, col1, ...]
        var flatMoves: [Int32] = []
        for move in validMoves {
            flatMoves.append(Int32(move.row))
            flatMoves.append(Int32(move.col))
        }

        let movesBuffer = device.makeBuffer(bytes: flatMoves,
                                            length: flatMoves.count * MemoryLayout<Int32>.stride,
                                            options: .storageModeShared)

        var moveCount = Int32(validMoves.count)
        let moveCountBuffer = device.makeBuffer(bytes: &moveCount,
                                                length: MemoryLayout<Int32>.stride,
                                                options: .storageModeShared)

        // Flatten pegs to [row, col, player, row, col, player, ...]
        var flatPegs: [UInt8] = []
        for peg in game.pegs {
            flatPegs.append(peg.row)
            flatPegs.append(peg.col)
            flatPegs.append(peg.player)
        }

        let pegsBuffer = device.makeBuffer(bytes: flatPegs,
                                           length: max(1, flatPegs.count) * MemoryLayout<UInt8>.stride,
                                           options: .storageModeShared)

        var pegCount = Int32(game.pegCount)
        let pegCountBuffer = device.makeBuffer(bytes: &pegCount,
                                               length: MemoryLayout<Int32>.stride,
                                               options: .storageModeShared)

        var currentPlayer = game.currentPlayer.rawValue
        let playerBuffer = device.makeBuffer(bytes: &currentPlayer,
                                             length: MemoryLayout<UInt8>.stride,
                                             options: .storageModeShared)

        // Output buffer - matches Metal MoveScore struct
        // struct MoveScore { int row; int col; float heuristicScore; HeuristicFeatures features[28 floats]; }
        // Size = 4 + 4 + 4 + (28 * 4) = 124 bytes per move
        let bytesPerMove = 4 + 4 + 4 + (28 * 4)
        let scoresBuffer = device.makeBuffer(length: validMoves.count * bytesPerMove,
                                             options: .storageModeShared)

        guard let boardBuffer = boardBuffer,
              let movesBuffer = movesBuffer,
              let moveCountBuffer = moveCountBuffer,
              let pegsBuffer = pegsBuffer,
              let pegCountBuffer = pegCountBuffer,
              let playerBuffer = playerBuffer,
              let scoresBuffer = scoresBuffer else {
            throw AIError.failedToCreateBuffer
        }

        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AIError.gpuExecutionFailed
        }

        computeEncoder.setComputePipelineState(evaluateMovePipeline)
        computeEncoder.setBuffer(boardBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(movesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(moveCountBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(pegsBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(pegCountBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(playerBuffer, offset: 0, index: 5)
        computeEncoder.setBuffer(scoresBuffer, offset: 0, index: 6)

        // Dispatch threads (Metal 4 + Apple Silicon: Optimal threadgroup size for M3)
        // M3 Pro has 18 GPU cores, use 32-thread SIMD width for best performance
        let optimalThreadsPerGroup = min(256, evaluateMovePipeline.maxTotalThreadsPerThreadgroup)
        let threadGroupSize = MTLSize(width: optimalThreadsPerGroup, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (validMoves.count + optimalThreadsPerGroup - 1) / optimalThreadsPerGroup,
                                   height: 1, depth: 1)

        // Metal 4: Use dispatchThreads API for better occupancy on Apple Silicon
        if device.supportsFamily(.apple9) {
            // M3 family - use newer dispatch method
            let gridSize = MTLSize(width: validMoves.count, height: 1, depth: 1)
            computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        } else {
            computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        }
        computeEncoder.endEncoding()

        // Execute and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read results - need to match Metal shader struct layout
        // Metal struct: { Int32 row; Int32 col; Float score; HeuristicFeatures features[28 floats]; }
        let basePointer = scoresBuffer.contents()
        var results: [(move: GameState.Move, score: Float)] = []

        // Calculate stride in bytes: Int32 + Int32 + Float + 28*Float = 4 + 4 + 4 + 112 = 124 bytes
        let strideBytes = MemoryLayout<Int32>.stride * 2 + MemoryLayout<Float>.stride * (1 + 28)

        for i in 0..<validMoves.count {
            let offsetBytes = i * strideBytes
            let resultPtr = basePointer.advanced(by: offsetBytes)

            // Read as Int32s for row/col
            let row = resultPtr.load(as: Int32.self)
            let col = resultPtr.advanced(by: 4).load(as: Int32.self)
            let score = resultPtr.advanced(by: 8).load(as: Float.self)

            results.append((move: GameState.Move(row: Int(row), col: Int(col)), score: score))
        }

        return results
    }

    /// Batch evaluate positions for multiple games (used in minimax)
    func batchEvaluatePositions(boards: [[UInt8]], players: [UInt8]) throws -> [Float] {
        let batchSize = boards.count
        guard batchSize > 0 else {
            return []
        }

        // Flatten boards
        var flatBoards: [UInt8] = []
        for board in boards {
            flatBoards.append(contentsOf: board)
        }

        let boardsBuffer = device.makeBuffer(bytes: flatBoards,
                                             length: flatBoards.count * MemoryLayout<UInt8>.stride,
                                             options: .storageModeShared)

        var batchSizeInt = Int32(batchSize)
        let batchSizeBuffer = device.makeBuffer(bytes: &batchSizeInt,
                                                length: MemoryLayout<Int32>.stride,
                                                options: .storageModeShared)

        let playersBuffer = device.makeBuffer(bytes: players,
                                              length: players.count * MemoryLayout<UInt8>.stride,
                                              options: .storageModeShared)

        let scoresBuffer = device.makeBuffer(length: batchSize * MemoryLayout<Float>.stride,
                                             options: .storageModeShared)

        guard let boardsBuffer = boardsBuffer,
              let batchSizeBuffer = batchSizeBuffer,
              let playersBuffer = playersBuffer,
              let scoresBuffer = scoresBuffer else {
            throw AIError.failedToCreateBuffer
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw AIError.gpuExecutionFailed
        }

        computeEncoder.setComputePipelineState(batchEvaluatePipeline)
        computeEncoder.setBuffer(boardsBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(batchSizeBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(playersBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(scoresBuffer, offset: 0, index: 3)

        let threadGroupSize = MTLSize(width: min(batchEvaluatePipeline.maxTotalThreadsPerThreadgroup, batchSize),
                                      height: 1, depth: 1)
        let threadGroups = MTLSize(width: (batchSize + threadGroupSize.width - 1) / threadGroupSize.width,
                                   height: 1, depth: 1)

        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let scoresPointer = scoresBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: scoresPointer, count: batchSize))
    }
}
