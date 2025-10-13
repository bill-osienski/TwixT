import Foundation

/// Self-play engine that runs games using GPU-accelerated AI
class SelfPlayEngine {
    let ai: MetalAI?
    let searchDepth: Int
    let verbose: Bool
    let heuristics: HeuristicsConfig
    let valueModel: ValueModelEvaluator?
    private let searchAI: TwixTSwiftAI

    init(ai: MetalAI? = nil,
         searchDepth: Int = 3,
         verbose: Bool = false,
         heuristics: HeuristicsConfig? = nil,
         valueModel: ValueModelEvaluator? = nil) {
        self.ai = ai
        self.searchDepth = searchDepth
        self.verbose = verbose
        self.heuristics = heuristics ?? .default
        self.valueModel = valueModel
        self.searchAI = TwixTSwiftAI(config: self.heuristics, valueModel: valueModel)
    }

    /// Run a single self-play game
    func playGame(gameNumber: Int) throws -> GameTrace {
        var game = GameState()
        var moves: [MoveTrace] = []

        let maxMoves = 220
        var turnCount = 0

        if verbose {
            print("[Game \(gameNumber)] Starting self-play...")
        }

        while turnCount < maxMoves && !game.gameOver {
            let player = game.currentPlayer

            // Choose move using Swift search (parity with JS)
            guard let bestMove = searchAI.chooseMove(game: game, depth: searchDepth) else {
                if verbose {
                    print("[Game \(gameNumber)] No valid moves available")
                }
                break
            }

            let moveDetail = searchAI.lastDetail
            let featureContext = moveDetail?.featureContext ?? FeatureContext(
                turn: turnCount,
                player: player.name,
                playerPegCount: game.pegCount + 1,
                opponentPegCount: game.pegCount
            )
            let heuristicsMap = moveDetail?.heuristics ?? [:]

            // Place the peg
            let success = game.placePeg(row: bestMove.row, col: bestMove.col)
            if !success {
                if verbose {
                    print("[Game \(gameNumber)] Failed to place peg at (\(bestMove.row), \(bestMove.col))")
                }
                break
            }

            moves.append(
                MoveTrace(
                    turn: turnCount,
                    player: player.name,
                    move: MoveData(row: bestMove.row, col: bestMove.col),
                    heuristics: heuristicsMap,
                    featureContext: featureContext,
                    valueModel: moveDetail?.valueModel,
                    heuristicScore: moveDetail?.heuristicScore
                )
            )

            turnCount += 1

            if verbose && turnCount % 10 == 0 {
                print("[Game \(gameNumber)] Turn \(turnCount), \(game.currentPlayer.name) to move")
            }

            // Check for win
            if game.gameOver {
                if verbose {
                    print("[Game \(gameNumber)] Game over! Winner: \(game.winner?.name ?? "none")")
                }
                break
            }
        }

        let summary = GameSummary(
            boardSize: GameState.boardSize,
            totalMoves: turnCount,
            winner: game.winner?.name,
            gameOver: game.gameOver,
            draw: !game.gameOver && turnCount >= maxMoves
        )

        return GameTrace(
            gameNumber: gameNumber,
            moves: moves,
            summary: summary,
            stalled: false
        )
    }

    /// Run multiple games in sequence (can be parallelized by running multiple workers)
    func runSelfPlay(gameCount: Int, startingGameNumber: Int = 1) throws -> [GameTrace] {
        var traces: [GameTrace] = []

        for i in 0..<gameCount {
            let gameNum = startingGameNumber + i
            let trace = try playGame(gameNumber: gameNum)
            traces.append(trace)

            if verbose {
                print("[SelfPlay] Completed game \(gameNum)/\(startingGameNumber + gameCount - 1): \(trace.summary.totalMoves) moves, winner: \(trace.summary.winner ?? "draw")")
            }
        }

        return traces
    }
}

// MARK: - Data structures matching JavaScript format

struct GameTrace: Codable {
    let gameNumber: Int
    let moves: [MoveTrace]
    let summary: GameSummary
    let stalled: Bool
}

struct MoveTrace: Codable {
    let turn: Int
    let player: String
    let move: MoveData
    let heuristics: [String: Double]
    let featureContext: FeatureContext
    var valueModel: ValueModelResult?
    var heuristicScore: Double?
}

struct MoveData: Codable {
    let row: Int
    let col: Int
}

struct FeatureContext: Codable {
    let turn: Int
    let player: String
    let playerPegCount: Int
    let opponentPegCount: Int
}

struct ValueModelResult: Codable {
    let probability: Double?
    let adjustment: Double?
    let logit: Double?
}

struct GameSummary: Codable {
    let boardSize: Int
    let totalMoves: Int
    let winner: String?
    let gameOver: Bool
    let draw: Bool
}

// MARK: - Output format for consolidator compatibility

struct SelfPlayOutput: Codable {
    let generatedAt: String
    let gameRequested: Int
    let gameCompleted: Int
    let searchDepth: Int
    let aborted: Bool
    let games: [GameTrace]
}

struct ParallelWorkerOutput: Codable {
    let moves: [MoveTrace]
    let summary: GameSummary
    let meta: GameMeta
}

struct GameMeta: Codable {
    let runId: String
    let coreId: Int
    let seq: Int
    let depth: Int
    let createdAt: String
    let aborted: Bool
    let draw: Bool
    let stalled: Bool
}
