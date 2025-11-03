import Foundation

/// Swift reimplementation of the JavaScript TwixTAI search logic.
/// Produces the same heuristic feature breakdown, value-model annotations, and depth-aware search.
final class TwixTSwiftAI {
    struct MoveDetail {
        let move: GameState.Move
        let featureContext: FeatureContext
        let heuristics: [String: Double]
        let valueModel: ValueModelResult?
        let heuristicScore: Double
        let minimaxScore: Double
        let immediateScore: Double
        let positionScore: Double
        let totalScore: Double
    }

    private let config: HeuristicsConfig
    private let valueModel: ValueModelEvaluator?

    private(set) var lastDetail: MoveDetail?

    init(config: HeuristicsConfig, valueModel: ValueModelEvaluator?) {
        self.config = config
        self.valueModel = valueModel
    }

    func chooseMove(game: GameState, depth: Int, preferredMoves: [GameState.Move] = []) -> GameState.Move? {
        guard let detail = bestMoveDetail(game: game, depth: depth, preferredMoves: preferredMoves) else {
            lastDetail = nil
            return nil
        }
        lastDetail = detail
        return detail.move
    }

    // MARK: - Core search

    private func bestMoveDetail(game: GameState, depth: Int, preferredMoves: [GameState.Move]) -> MoveDetail? {
        let moves = game.getValidMoves()
        guard !moves.isEmpty else { return nil }

        let player = game.currentPlayer
        let opponent = player.opponent
        let opponentThreat = connectivityScore(game: game, player: opponent)
        let friendlyMetrics = componentMetrics(game: game, player: player)
        let friendlyTargets = computeConnectorTargets(game: game, player: player, metrics: friendlyMetrics)
        let opponentFrontier = computeFrontier(game: game, player: opponent)
        let opponentTargets = computeConnectorTargets(game: game, player: opponent, metrics: opponentFrontier.metrics)

        let ordered = orderMoves(
            game: game,
            moves: moves,
            player: player,
            depth: depth,
            opponent: opponent,
            opponentThreatBefore: opponentThreat,
            friendlyMetrics: friendlyMetrics,
            friendlyConnectorTargets: friendlyTargets,
            opponentConnectorTargets: opponentTargets,
            opponentMetrics: opponentFrontier.metrics,
            opponentFrontier: opponentFrontier.frontier,
            opponentConnectors: opponentFrontier.connectors,
            opponentTrailing: opponentFrontier.trailing
        )

        let validMoveSet = Set(moves)
        var seen = Set<GameState.Move>()
        var candidates: [GameState.Move] = []

        for move in preferredMoves where validMoveSet.contains(move) {
            if seen.insert(move).inserted {
                candidates.append(move)
            }
        }

        let fallback = ordered.isEmpty ? moves : ordered
        for move in fallback {
            if seen.insert(move).inserted {
                candidates.append(move)
            }
        }

        if candidates.isEmpty {
            candidates = moves
        }

        var best: MoveDetail?
        for move in candidates {
            if let detail = evaluateMoveDetail(
                game: game,
                move: move,
                depth: depth,
                opponentThreatBefore: opponentThreat,
                friendlyMetrics: friendlyMetrics,
                friendlyConnectorTargets: friendlyTargets,
                opponentConnectorTargets: opponentTargets,
                opponentMetrics: opponentFrontier.metrics,
                opponentFrontier: opponentFrontier.frontier,
                opponentConnectors: opponentFrontier.connectors,
                opponentTrailing: opponentFrontier.trailing
            ) {
                // Optional debug logging can be added here if needed.
                if let current = best {
                    if detail.totalScore > current.totalScore {
                        best = detail
                    }
                } else {
                    best = detail
                }
            }
        }

        return best
    }

    private func evaluateMoveDetail(
        game: GameState,
        move: GameState.Move,
        depth: Int,
        opponentThreatBefore: Double,
        friendlyMetrics: ComponentMetrics,
        friendlyConnectorTargets: ConnectorTargets?,
        opponentConnectorTargets: ConnectorTargets?,
        opponentMetrics: ComponentMetrics,
        opponentFrontier: [GameState.Coord],
        opponentConnectors: [GameState.Coord],
        opponentTrailing: [GameState.Coord]
    ) -> MoveDetail? {
        let player = game.currentPlayer
        let opponent = player.opponent
        let friendlyPegs = game.pegs.filter { $0.player == player.rawValue }
        let opponentPegs = game.pegs.filter { $0.player == opponent.rawValue }

        let priority = movePriority(
            game: game,
            move: move,
            player: player,
            friendlyPegs: friendlyPegs,
            opponentPegs: opponentPegs,
            opponent: opponent,
            opponentThreatBefore: opponentThreatBefore,
            friendlyMetrics: friendlyMetrics,
            friendlyConnectorTargets: friendlyConnectorTargets,
            opponentConnectorTargets: opponentConnectorTargets,
            opponentMetrics: opponentMetrics,
            opponentFrontier: opponentFrontier,
            opponentConnectors: opponentConnectors,
            opponentTrailing: opponentTrailing,
            captureDetails: true
        )

        var simulated = game
        guard simulated.placePeg(row: move.row, col: move.col) else {
            return nil
        }

        var features = priority.features
        var heuristicScore = priority.score
        if simulated.gameOver && simulated.winner == player {
            heuristicScore += 10000.0
            features["immediateWin", default: 0.0] += 10000.0
        }

        let immediateScore = evaluateMoveHeuristic(game: simulated, move: move, player: player)
        let positionScore = evaluatePosition(game: simulated, player: player)
        let minimaxScore: Double
        if depth <= 1 || simulated.gameOver {
            minimaxScore = positionScore
        } else {
            minimaxScore = minimax(
                game: simulated,
                depth: depth - 1,
                isMaximizing: false,
                alpha: -Double.infinity,
                beta: Double.infinity,
                rootPlayer: player
            )
        }

        let totalScore = minimaxScore + immediateScore * 5.0 + positionScore * 0.1

        return MoveDetail(
            move: move,
            featureContext: priority.featureContext,
            heuristics: features,
            valueModel: priority.valueModel.map {
                ValueModelResult(probability: $0.probability, adjustment: $0.adjustment, logit: $0.logit)
            },
            heuristicScore: heuristicScore,
            minimaxScore: minimaxScore,
            immediateScore: immediateScore,
            positionScore: positionScore,
            totalScore: totalScore
        )
    }

    private func orderMoves(
        game: GameState,
        moves: [GameState.Move],
        player: GameState.Player,
        depth: Int,
        opponent: GameState.Player,
        opponentThreatBefore: Double,
        friendlyMetrics: ComponentMetrics,
        friendlyConnectorTargets: ConnectorTargets?,
        opponentConnectorTargets: ConnectorTargets?,
        opponentMetrics: ComponentMetrics,
        opponentFrontier: [GameState.Coord],
        opponentConnectors: [GameState.Coord],
        opponentTrailing: [GameState.Coord]
    ) -> [GameState.Move] {
        guard !moves.isEmpty else { return [] }

        let friendlyPegs = game.pegs.filter { $0.player == player.rawValue }
        let opponentPegs = game.pegs.filter { $0.player == opponent.rawValue }

        let boardSize = GameState.boardSize
        let spanValue = opponent == .red ? opponentMetrics.maxRowSpan : opponentMetrics.maxColSpan
        let opponentUrgent = spanValue >= max(6, boardSize / 4) || opponentMetrics.largestComponent.count >= 6

        var scored: [(GameState.Move, Double)] = []
        scored.reserveCapacity(moves.count)

        for move in moves {
            let result = movePriority(
                game: game,
                move: move,
                player: player,
                friendlyPegs: friendlyPegs,
                opponentPegs: opponentPegs,
                opponent: opponent,
                opponentThreatBefore: opponentThreatBefore,
                friendlyMetrics: friendlyMetrics,
                friendlyConnectorTargets: friendlyConnectorTargets,
                opponentConnectorTargets: opponentConnectorTargets,
                opponentMetrics: opponentMetrics,
                opponentFrontier: opponentFrontier,
                opponentConnectors: opponentConnectors,
                opponentTrailing: opponentTrailing,
                captureDetails: false,
                opponentUrgentOverride: opponentUrgent
            )
            scored.append((move, result.score))
        }

        scored.sort { $0.1 > $1.1 }

        let baseLimit = 20.0
        let effectiveDepth = max(1, depth)
        let depthFactor = Double(depth + 1)
        let limit = max(6, min(scored.count, Int(round(baseLimit * depthFactor / Double(effectiveDepth + 1)))))

        return scored.prefix(limit).map { $0.0 }
    }

    // MARK: - Move priority (ported from JS)

    private func movePriority(
        game: GameState,
        move: GameState.Move,
        player: GameState.Player,
        friendlyPegs: [GameState.PegData],
        opponentPegs: [GameState.PegData],
        opponent: GameState.Player,
        opponentThreatBefore: Double,
        friendlyMetrics: ComponentMetrics,
        friendlyConnectorTargets: ConnectorTargets?,
        opponentConnectorTargets: ConnectorTargets?,
        opponentMetrics: ComponentMetrics,
        opponentFrontier: [GameState.Coord],
        opponentConnectors: [GameState.Coord],
        opponentTrailing: [GameState.Coord],
        captureDetails: Bool,
        opponentUrgentOverride: Bool? = nil
    ) -> MovePriorityResult {
        var accumulator = FeatureAccumulator()
        var score = 0.0
        let general = config.general
        let offense = config.edge.offense
        let defense = config.edge.defense

        let boardSize = GameState.boardSize
        let boardLimit = GameState.boardSize - 1
        let moveCoord = GameState.Coord(row: move.row, col: move.col)
        let friendlyTargetSet = friendlyConnectorTargets?.positions ?? []
        let opponentTargetSet = opponentConnectorTargets?.positions ?? []

        let friendlyConnections = countConnections(game: game, move: move, color: player)
        let opponentConnections = countConnections(game: game, move: move, color: opponent)

        func addFeature(_ key: String, _ value: Double) {
            guard value != 0 else { return }
            accumulator.capture(key, value)
            score += value
        }

        addFeature("friendlyConnections", Double(friendlyConnections) * Double(general.friendlyConnection))
        addFeature("opponentConnections", Double(opponentConnections) * Double(general.opponentConnection))

        let friendlyDist = minDistance(move: move, pegs: friendlyPegs)
        if let friendlyDist = friendlyDist {
            addFeature("friendlyDistance", max(0.0, 10.0 - friendlyDist) * Double(general.friendlyDistance))
        }

        let opponentDist = minDistance(move: move, pegs: opponentPegs)
        if let opponentDist = opponentDist {
            addFeature("opponentDistance", max(0.0, 10.0 - opponentDist) * Double(general.opponentDistance))
        }

        let goalDistance = player == .red
            ? Double(min(move.row, boardSize - 1 - move.row))
            : Double(min(move.col, boardSize - 1 - move.col))
        addFeature("goalDistance", max(0.0, 12.0 - goalDistance) * Double(general.goalDistance))

        let center = Double(boardSize - 1) / 2.0
        let centerDist = abs(Double(move.row) - center) + abs(Double(move.col) - center)
        addFeature("centerBias", max(0.0, 16.0 - centerDist) * Double(general.centerBias))

        if friendlyDist == nil && opponentDist == nil {
            addFeature("isolatedBonus", Double(general.isolated))
        }

        if friendlyTargetSet.contains(moveCoord) {
            addFeature("edgeConnectorTarget", Double(offense.connectorTargetBonus))
        }

        var blockedOpponentConnector = false
        if opponentTargetSet.contains(moveCoord) {
            addFeature("edgeDefenseBlock", Double(defense.blockBonus))
            blockedOpponentConnector = true
        }

        var opponentUrgent = opponentUrgentOverride ?? {
            let spanValue = opponent == .red ? opponentMetrics.maxRowSpan : opponentMetrics.maxColSpan
            return spanValue >= max(6, boardSize / 4) || opponentMetrics.largestComponent.count >= 6
        }()
        if opponentMetrics.largestComponent.isEmpty {
            opponentUrgent = false
        }

        if !opponentMetrics.largestComponent.isEmpty {
            let distToChain = distance(move: moveCoord, component: opponentMetrics.largestComponent)
            addFeature("chainProximity", max(0.0, 12.0 - Double(distToChain)) * (opponentUrgent ? 30.0 : 15.0))
        }

        if !opponentFrontier.isEmpty {
            let dist = distance(move: moveCoord, cells: opponentFrontier)
            if dist != Int.max {
                addFeature("frontierProximity", max(0.0, 10.0 - Double(dist)) * (opponentUrgent ? 35.0 : 16.0))
                if dist == 0 {
                    addFeature("frontierCapture", opponentUrgent ? 550.0 : 220.0)
                }
            }
        }

        if !opponentConnectors.isEmpty {
            let dist = distance(move: moveCoord, cells: opponentConnectors)
            if dist != Int.max {
                addFeature("connectorProximity", max(0.0, 8.0 - Double(dist)) * (opponentUrgent ? 55.0 : 30.0))
                if dist == 0 {
                    addFeature("connectorCapture", opponentUrgent ? 700.0 : 320.0)
                }
            }
        }

        if !opponentTrailing.isEmpty {
            let dist = distance(move: moveCoord, cells: opponentTrailing)
            if dist != Int.max {
                let penalty = max(0.0, 6.0 - Double(dist)) * 6.0
                addFeature("trailingPenalty", -penalty)
            }
        }

        var simulated = game
        _ = simulated.placePeg(row: move.row, col: move.col)

        let opponentThreatAfter = connectivityScore(game: simulated, player: opponent)
        let threatReduction = opponentThreatBefore - opponentThreatAfter
        if !opponentMetrics.largestComponent.isEmpty && game.moveCount > 1 {
            if threatReduction > 0 {
                addFeature("threatReduction", threatReduction * 140.0)
            } else {
                addFeature("noThreatReduction", -(opponentUrgent ? 600.0 : 250.0))
            }
        }

        let postMetrics = componentMetrics(game: simulated, player: player)

        let postLargest = postMetrics.largestComponent
        var postMinR = Int.max
        var postMaxR = Int.min
        var postMinC = Int.max
        var postMaxC = Int.min
        for coord in postLargest {
            postMinR = min(postMinR, coord.row)
            postMaxR = max(postMaxR, coord.row)
            postMinC = min(postMinC, coord.col)
            postMaxC = max(postMaxC, coord.col)
        }

        let friendlyLargest = friendlyMetrics.largestComponent
        var friendlyMinR = Int.max
        var friendlyMaxR = Int.min
        var friendlyMinC = Int.max
        var friendlyMaxC = Int.min
        for coord in friendlyLargest {
            friendlyMinR = min(friendlyMinR, coord.row)
            friendlyMaxR = max(friendlyMaxR, coord.row)
            friendlyMinC = min(friendlyMinC, coord.col)
            friendlyMaxC = max(friendlyMaxC, coord.col)
        }

        if player == .black {
            let newLeft = postMetrics.touchesLeft && !friendlyMetrics.touchesLeft
            let newRight = postMetrics.touchesRight && !friendlyMetrics.touchesRight
            if newLeft || newRight {
                addFeature("firstEdgeTouch", Double(offense.firstEdgeTouchBlack))
            }
        } else {
            let newTop = postMetrics.touchesTop && !friendlyMetrics.touchesTop
            let newBottom = postMetrics.touchesBottom && !friendlyMetrics.touchesBottom
            if newTop || newBottom {
                addFeature("firstEdgeTouch", Double(offense.firstEdgeTouchRed))
            }
        }

        if player == .black {
            let hadBoth = friendlyMetrics.touchesLeft && friendlyMetrics.touchesRight
            let hasBoth = postMetrics.touchesLeft && postMetrics.touchesRight
            let componentSpansBoth = !postLargest.isEmpty && postMinC <= 0 && postMaxC >= boardLimit
            if hasBoth && !hadBoth && componentSpansBoth {
                addFeature("doubleEdgeCoverage", 2400.0 * Double(offense.blackDoubleCoverageScale))
            }
        } else {
            let hadBoth = friendlyMetrics.touchesTop && friendlyMetrics.touchesBottom
            let hasBoth = postMetrics.touchesTop && postMetrics.touchesBottom
            let componentSpansBoth = !postLargest.isEmpty && postMinR <= 0 && postMaxR >= boardLimit
            if hasBoth && !hadBoth && componentSpansBoth {
                addFeature("doubleEdgeCoverage", 2400.0 + Double(offense.redDoubleCoverageBonus))
            }
        }

        let spanBefore = player == .red ? friendlyMetrics.maxRowSpan : friendlyMetrics.maxColSpan
        let spanAfter = player == .red ? postMetrics.maxRowSpan : postMetrics.maxColSpan
        let spanGain = spanAfter - spanBefore
        if spanGain > 0 {
            var multiplier = 180.0 * (player == .black ? Double(offense.blackSpanGainMultiplier) : 1.0)
            if player == .red && (postMetrics.touchesTop || postMetrics.touchesBottom) {
                multiplier *= Double(offense.redSpanGainMultiplier)
            }
            addFeature("spanGain", Double(spanGain) * multiplier)
        }

        let prevMinAxis: Int
        let prevMaxAxis: Int
        let postMinAxis: Int
        let postMaxAxis: Int
        if player == .red {
            prevMinAxis = friendlyLargest.isEmpty ? (friendlyMetrics.minRow ?? boardLimit) : friendlyMinR
            prevMaxAxis = friendlyLargest.isEmpty ? (friendlyMetrics.maxRow ?? 0) : friendlyMaxR
            postMinAxis = postLargest.isEmpty ? (postMetrics.minRow ?? boardLimit) : postMinR
            postMaxAxis = postLargest.isEmpty ? (postMetrics.maxRow ?? 0) : postMaxR
        } else {
            prevMinAxis = friendlyLargest.isEmpty ? (friendlyMetrics.minCol ?? boardLimit) : friendlyMinC
            prevMaxAxis = friendlyLargest.isEmpty ? (friendlyMetrics.maxCol ?? 0) : friendlyMaxC
            postMinAxis = postLargest.isEmpty ? (postMetrics.minCol ?? boardLimit) : postMinC
            postMaxAxis = postLargest.isEmpty ? (postMetrics.maxCol ?? 0) : postMaxC
        }

        let gapBefore = max(0, prevMinAxis) + max(0, boardLimit - prevMaxAxis)
        let gapAfter = max(0, postMinAxis) + max(0, boardLimit - postMaxAxis)
        let gapImprovement = gapBefore - gapAfter
        if gapImprovement > 0 {
            let multiplier = Double(offense.gapDecay) * (player == .red ? Double(offense.redGapDecayMultiplier) : 1.0)
            addFeature("edgeGapReduction", Double(gapImprovement) * multiplier)
        }

        if !postLargest.isEmpty {
            let lcTouchesTop = postMinR <= 1
            let lcTouchesBottom = postMaxR >= boardLimit - 1
            let lcTouchesLeft = postMinC <= 1
            let lcTouchesRight = postMaxC >= boardLimit - 1
            let redSpans = lcTouchesTop && lcTouchesBottom
            let blackSpans = lcTouchesLeft && lcTouchesRight
            if (player == .red && redSpans) || (player == .black && blackSpans) {
                let base = Double(offense.finishBonusBase) * 2.0 * (player == .black ? Double(offense.blackFinishScaleMultiplier) : 1.0)
                addFeature("largestComponentSpanComplete", base)
            }
        }

        let touchesBothPost = player == .red
            ? (postMetrics.touchesTop && postMetrics.touchesBottom)
            : (postMetrics.touchesLeft && postMetrics.touchesRight)
        let nearFinish = gapAfter <= Int(offense.finishThreshold)

        if touchesBothPost || nearFinish {
            let progressMade = spanGain > 0 || gapImprovement > 0
            let finishScaleBase = max(0.0, Double(offense.finishBonusBase) - Double(gapAfter) * 150.0)
            if progressMade {
                var bonusBase = Double(offense.connectorBonus) + finishScaleBase
                if player == .black {
                    bonusBase *= Double(offense.blackFinishScaleMultiplier)
                }
                if player == .red {
                    bonusBase += Double(offense.redFinishExtra)
                }
                addFeature("edgeFinishAdvance", bonusBase)
            } else {
                let penaltyBase = Double(offense.finishPenaltyBase) + Double(gapAfter) * 150.0
                let penalty = penaltyBase * (player == .red ? Double(offense.redFinishPenaltyFactor) : 1.0)
                addFeature("edgeFinishStall", -penalty)
            }
        }

        if let opponentTargets = opponentConnectorTargets,
           !opponentTargets.positions.isEmpty,
           !blockedOpponentConnector,
           !touchesBothPost,
           game.moveCount > 1 {
            let defensePenalty = Double(defense.missPenalty) * (opponentUrgent ? 1.5 : 1.0)
            addFeature("edgeDefenseMiss", -defensePenalty)
        }

        let opponentPostMetrics = componentMetrics(game: simulated, player: opponent)
        let opponentSpanBefore = opponent == .red ? opponentMetrics.maxRowSpan : opponentMetrics.maxColSpan
        let opponentSpanAfter = opponent == .red ? opponentPostMetrics.maxRowSpan : opponentPostMetrics.maxColSpan
        let spanReduction = opponentSpanBefore - opponentSpanAfter
        if spanReduction > 0 {
            addFeature("opponentSpanReduction", Double(spanReduction) * 120.0)
        } else if opponentUrgent {
            addFeature("noSpanReductionPenalty", -400.0)
        }

        if opponent == .black &&
            opponentPostMetrics.touchesLeft && opponentPostMetrics.touchesRight &&
            !(opponentMetrics.touchesLeft && opponentMetrics.touchesRight) {
            addFeature("blackSpanUpgradePenalty", -500.0)
        }

        if opponent == .red &&
            opponentPostMetrics.touchesTop && opponentPostMetrics.touchesBottom &&
            !(opponentMetrics.touchesTop && opponentMetrics.touchesBottom) {
            addFeature("redSpanUpgradePenalty", -500.0)
        }

        if opponent == .red &&
            opponentMetrics.touchesBottom && !opponentMetrics.touchesTop {
            let topBias = max(0.0, Double(boardSize - move.row)) * 12.0
            addFeature("topBias", topBias)
            if let minRow = opponentMetrics.minRow {
                addFeature("aboveMinRowBonus", max(0.0, Double(minRow - move.row)) * 150.0)
                addFeature("belowMinRowPenalty", -max(0.0, Double(move.row - minRow)) * 90.0)
            }
        } else if opponent == .red &&
                    opponentMetrics.touchesTop && !opponentMetrics.touchesBottom {
            let bottomBias = max(0.0, Double(move.row)) * 12.0
            addFeature("bottomBias", bottomBias)
            if let maxRow = opponentMetrics.maxRow {
                addFeature("belowMaxRowBonus", max(0.0, Double(move.row - maxRow)) * 150.0)
                addFeature("aboveMaxRowPenalty", -max(0.0, Double(maxRow - move.row)) * 90.0)
            }
        }

        let featureContext = FeatureContext(
            turn: game.moveCount,
            player: player.name,
            playerPegCount: friendlyPegs.count + 1,
            opponentPegCount: opponentPegs.count
        )

        var valueResult: ValueModelEvaluator.Result? = nil
        if let evaluator = valueModel {
            valueResult = evaluator.evaluate(
                heuristics: accumulator.snapshot,
                featureContext: featureContext,
                scale: Double(config.valueModelScale)
            )
            if let adjustment = valueResult?.adjustment {
                score += adjustment
            }
        }

        if player == .red {
            addFeature("redBaseBonus", Double(general.redBaseBonus))
            if general.redGlobalMultiplier != 1.0 {
                let delta = score * (Double(general.redGlobalMultiplier) - 1.0)
                addFeature("redGlobalMultiplier", delta)
            }
        } else {
            addFeature("blackBasePenalty", -Double(general.blackBasePenalty))
            if general.blackGlobalScale != 1.0 {
                let delta = score * (Double(general.blackGlobalScale) - 1.0)
                addFeature("blackGlobalScale", delta)
            }
        }

        let lateStart = Int(general.lateGameStart)
        let latePressure = Double(general.lateGamePressure)
        if latePressure > 0 {
            let lateTurns = (game.moveCount + 1) - lateStart
            if lateTurns > 0 {
                addFeature("lateGamePressure", -Double(lateTurns) * latePressure)
            }
        }

        return MovePriorityResult(
            score: score,
            features: captureDetails ? accumulator.snapshot : [:],
            featureContext: featureContext,
            valueModel: valueResult
        )
    }

    // MARK: - Minimax / evaluation helpers

    private func minimax(
        game: GameState,
        depth: Int,
        isMaximizing: Bool,
        alpha: Double,
        beta: Double,
        rootPlayer: GameState.Player
    ) -> Double {
        if depth == 0 || game.gameOver {
            return evaluatePosition(game: game, player: rootPlayer)
        }

        var alphaVar = alpha
        var betaVar = beta
        var bestScore = isMaximizing ? -Double.infinity : Double.infinity

        let allMoves = game.getValidMoves()
        if allMoves.isEmpty {
            return evaluatePosition(game: game, player: rootPlayer)
        }

        let currentPlayer = game.currentPlayer
        let opponent = currentPlayer.opponent
        let opponentThreat = connectivityScore(game: game, player: opponent)
        let friendlyMetrics = componentMetrics(game: game, player: currentPlayer)
        let friendlyConnectorTargets = computeConnectorTargets(game: game, player: currentPlayer, metrics: friendlyMetrics)
        let opponentFrontierData = computeFrontier(game: game, player: opponent)
        let opponentConnectorTargets = computeConnectorTargets(
            game: game,
            player: opponent,
            metrics: opponentFrontierData.metrics
        )

        let orderedMoves = orderMoves(
            game: game,
            moves: allMoves,
            player: currentPlayer,
            depth: depth,
            opponent: opponent,
            opponentThreatBefore: opponentThreat,
            friendlyMetrics: friendlyMetrics,
            friendlyConnectorTargets: friendlyConnectorTargets,
            opponentConnectorTargets: opponentConnectorTargets,
            opponentMetrics: opponentFrontierData.metrics,
            opponentFrontier: opponentFrontierData.frontier,
            opponentConnectors: opponentFrontierData.connectors,
            opponentTrailing: opponentFrontierData.trailing
        )

        let candidates = orderedMoves.isEmpty ? allMoves : orderedMoves

        for move in candidates {
            var simulated = game
            guard simulated.placePeg(row: move.row, col: move.col) else { continue }

            let score = minimax(
                game: simulated,
                depth: depth - 1,
                isMaximizing: !isMaximizing,
                alpha: alphaVar,
                beta: betaVar,
                rootPlayer: rootPlayer
            )

            if isMaximizing {
                bestScore = max(bestScore, score)
                alphaVar = max(alphaVar, score)
            } else {
                bestScore = min(bestScore, score)
                betaVar = min(betaVar, score)
            }

            if betaVar <= alphaVar {
                break
            }
        }

        return bestScore
    }

    private func evaluateMoveHeuristic(game: GameState, move: GameState.Move, player: GameState.Player) -> Double {
        var score = 0.0
        let opponent = player.opponent

        var connectionCount = 0
        for offset in knightOffsets {
            let r = move.row + offset.row
            let c = move.col + offset.col
            if r < 0 || r >= GameState.boardSize || c < 0 || c >= GameState.boardSize { continue }
            let idx = game.boardIndex(row: r, col: c)
            if game.board[idx] == player.rawValue {
                let fromCoord = GameState.Coord(row: move.row, col: move.col)
                let toCoord = GameState.Coord(row: r, col: c)
                if bridgeWouldCross(game: game, from: fromCoord, to: toCoord) {
                    continue
                }

                let distance = abs(move.row - r) + abs(move.col - c)
                score += 100.0 + Double(distance) * 5.0
                if player == .black {
                    let spansBoard = (min(move.col, c) <= 3 && max(move.col, c) >= GameState.boardSize - 4)
                    let wideSpan = abs(move.col - c) > 10
                    if spansBoard {
                        score += 300.0
                    } else if wideSpan {
                        score += 150.0
                    }
                } else {
                    let spansBoard = (min(move.row, r) <= 3 && max(move.row, r) >= GameState.boardSize - 4)
                    let wideSpan = abs(move.row - r) > 10
                    if spansBoard {
                        score += 300.0
                    } else if wideSpan {
                        score += 150.0
                    }
                }

                connectionCount += 1
            }
        }

        if connectionCount >= 2 {
            score += Double(connectionCount) * 75.0
        }

        let goalDistance = player == .red
            ? Double(min(move.row, GameState.boardSize - 1 - move.row))
            : Double(min(move.col, GameState.boardSize - 1 - move.col))
        score += max(0.0, 12.0 - goalDistance) * 8.0

        var opponentThreats = 0
        for offset in knightOffsets {
            let r = move.row + offset.row
            let c = move.col + offset.col
            if r < 0 || r >= GameState.boardSize || c < 0 || c >= GameState.boardSize { continue }
            let idx = game.boardIndex(row: r, col: c)
            if game.board[idx] == opponent.rawValue {
                opponentThreats += 1
            }
        }

        if opponentThreats > 0 {
            score += Double(opponentThreats) * 25.0
        }

        if game.moveCount < 10 {
            let center = Double(GameState.boardSize - 1) / 2.0
            let centerDist = abs(Double(move.row) - center) + abs(Double(move.col) - center)
            score += max(0.0, 24.0 - centerDist) * 2.0
        }

        return score
    }

    private func bridgeWouldCross(game: GameState, from: GameState.Coord, to: GameState.Coord) -> Bool {
        for bridge in game.bridges {
            let existingFrom = GameState.Coord(row: Int(bridge.fromRow), col: Int(bridge.fromCol))
            let existingTo = GameState.Coord(row: Int(bridge.toRow), col: Int(bridge.toCol))

            if from == existingFrom || from == existingTo || to == existingFrom || to == existingTo {
                continue
            }

            if segmentsIntersect(
                x1: from.col, y1: from.row,
                x2: to.col, y2: to.row,
                x3: existingFrom.col, y3: existingFrom.row,
                x4: existingTo.col, y4: existingTo.row
            ) {
                return true
            }
        }
        return false
    }

    private func segmentsIntersect(
        x1: Int, y1: Int,
        x2: Int, y2: Int,
        x3: Int, y3: Int,
        x4: Int, y4: Int
    ) -> Bool {
        func orientation(_ ax: Int, _ ay: Int, _ bx: Int, _ by: Int, _ cx: Int, _ cy: Int) -> Int {
            let value = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if value > 0 { return 1 }
            if value < 0 { return -1 }
            return 0
        }

        func onSegment(_ ax: Int, _ ay: Int, _ bx: Int, _ by: Int, _ cx: Int, _ cy: Int) -> Bool {
            return min(ax, bx) <= cx && cx <= max(ax, bx) &&
                min(ay, by) <= cy && cy <= max(ay, by)
        }

        let o1 = orientation(x1, y1, x2, y2, x3, y3)
        let o2 = orientation(x1, y1, x2, y2, x4, y4)
        let o3 = orientation(x3, y3, x4, y4, x1, y1)
        let o4 = orientation(x3, y3, x4, y4, x2, y2)

        if o1 != o2 && o3 != o4 {
            let touchesEndpoint =
                (o1 == 0 && onSegment(x1, y1, x2, y2, x3, y3)) ||
                (o2 == 0 && onSegment(x1, y1, x2, y2, x4, y4)) ||
                (o3 == 0 && onSegment(x3, y3, x4, y4, x1, y1)) ||
                (o4 == 0 && onSegment(x3, y3, x4, y4, x2, y2))
            return !touchesEndpoint
        }

        if o1 == 0 && onSegment(x1, y1, x2, y2, x3, y3) {
            return !(x3 == x1 && y3 == y1) && !(x3 == x2 && y3 == y2)
        }
        if o2 == 0 && onSegment(x1, y1, x2, y2, x4, y4) {
            return !(x4 == x1 && y4 == y1) && !(x4 == x2 && y4 == y2)
        }

        return false
    }

    private func evaluatePosition(game: GameState, player: GameState.Player) -> Double {
        if game.gameOver {
            return game.winner == player ? 10000.0 : -10000.0
        }

        let opponent = player.opponent
        var score = 0.0

        score += connectivityScore(game: game, player: player) * 100.0
        score -= connectivityScore(game: game, player: opponent) * 100.0

        score += evaluatePotentialConnections(game: game, player: player) * 20.0
        score -= evaluatePotentialConnections(game: game, player: opponent) * 20.0

        score += evaluateEdgeProgress(game: game, player: player) * 30.0
        score -= evaluateEdgeProgress(game: game, player: opponent) * 30.0

        let playerPegCount = game.pegs.filter { $0.player == player.rawValue }.count
        let opponentPegCount = game.pegs.filter { $0.player == opponent.rawValue }.count
        score += Double(playerPegCount - opponentPegCount) * 2.0

        let metrics = componentMetrics(game: game, player: player)
        if !metrics.largestComponent.isEmpty {
            var minR = Int.max, maxR = Int.min, minC = Int.max, maxC = Int.min
            for p in metrics.largestComponent {
                minR = min(minR, p.row)
                maxR = max(maxR, p.row)
                minC = min(minC, p.col)
                maxC = max(maxC, p.col)
            }

            let N = GameState.boardSize
            let touchesTop = minR <= 1
            let touchesBottom = maxR >= N - 2
            let touchesLeft = minC <= 1
            let touchesRight = maxC >= N - 2

            if player == .red {
                let gapTop = max(0, minR - 1)
                let gapBottom = max(0, (N - 2) - maxR)
                let gap = touchesTop ? gapBottom : touchesBottom ? gapTop : min(gapTop, gapBottom)
                let urgency = (touchesTop || touchesBottom) ? 2.5 : 1.0
                score += 200.0 * urgency * (1.0 / (1.0 + Double(gap)))
                score -= 40.0 * Double(gapTop + gapBottom)
            } else {
                let gapLeft = max(0, minC - 1)
                let gapRight = max(0, (N - 2) - maxC)
                let gap = touchesLeft ? gapRight : touchesRight ? gapLeft : min(gapLeft, gapRight)
                let urgency = (touchesLeft || touchesRight) ? 2.5 : 1.0
                score += 200.0 * urgency * (1.0 / (1.0 + Double(gap)))
                score -= 40.0 * Double(gapLeft + gapRight)
            }
        }

        score -= 0.05 * Double(game.moveCount)
        return score
    }

    private func connectivityScore(game: GameState, player: GameState.Player) -> Double {
        let components = findConnectedComponents(game: game, player: player)
        guard !components.isEmpty else { return -100.0 }

        var score = 0.0
        let totalPegs = components.reduce(0) { $0 + $1.count }

        for component in components {
            score += scoreComponent(component: component, player: player)
        }

        let avgComponentSize = Double(totalPegs) / Double(components.count)
        score += avgComponentSize * 20.0
        if components.count > 3 {
            score -= Double(components.count - 3) * 30.0
        }

        score += evaluateWinningThreats(player: player, components: components)
        return score
    }

    // MARK: - Structural analysis helpers

    private func findConnectedComponents(game: GameState, player: GameState.Player) -> [[GameState.Coord]] {
        let playerPegs = game.pegs.filter { $0.player == player.rawValue }
        var visited = Set<GameState.Coord>()
        var components: [[GameState.Coord]] = []

        for peg in playerPegs {
            let start = GameState.Coord(row: Int(peg.row), col: Int(peg.col))
            if visited.contains(start) { continue }

            var stack: [GameState.Coord] = [start]
            var component: [GameState.Coord] = []

            while let current = stack.popLast() {
                if visited.contains(current) { continue }
                let idx = game.boardIndex(row: current.row, col: current.col)
                guard game.board[idx] == player.rawValue else { continue }

                visited.insert(current)
                component.append(current)

                for bridge in game.bridges where bridge.player == player.rawValue {
                    let from = GameState.Coord(row: Int(bridge.fromRow), col: Int(bridge.fromCol))
                    let to = GameState.Coord(row: Int(bridge.toRow), col: Int(bridge.toCol))
                    if from == current && !visited.contains(to) {
                        stack.append(to)
                    } else if to == current && !visited.contains(from) {
                        stack.append(from)
                    }
                }
            }

            if !component.isEmpty {
                components.append(component)
            }
        }

        return components
    }

    private func scoreComponent(component: [GameState.Coord], player: GameState.Player) -> Double {
        var score = Double(component.count) * 10.0
        let rows = component.map(\.row)
        let cols = component.map(\.col)
        guard let minRow = rows.min(),
              let maxRow = rows.max(),
              let minCol = cols.min(),
              let maxCol = cols.max()
        else { return score }

        if player == .red {
            score += Double(maxRow - minRow) * 20.0
            if minRow == 0 && maxRow == GameState.boardSize - 1 {
                score += 500.0
            }
        } else {
            score += Double(maxCol - minCol) * 20.0
            if minCol == 0 && maxCol == GameState.boardSize - 1 {
                score += 500.0
            }
        }

        return score
    }

    private func evaluateWinningThreats(player: GameState.Player, components: [[GameState.Coord]]) -> Double {
        var score = 0.0
        for component in components {
            let rows = component.map(\.row)
            let cols = component.map(\.col)
            guard let minRow = rows.min(),
                  let maxRow = rows.max(),
                  let minCol = cols.min(),
                  let maxCol = cols.max()
            else { continue }

            if player == .red {
                if minRow == 0 && maxRow == GameState.boardSize - 1 {
                    score += 800.0
                } else if minRow <= 1 && maxRow >= GameState.boardSize - 2 {
                    score += 400.0
                } else if maxRow >= GameState.boardSize - 2 && minRow <= 5 {
                    score += 400.0
                } else if minRow <= 3 && maxRow >= GameState.boardSize - 4 {
                    score += 200.0
                }
            } else {
                if minCol == 0 && maxCol == GameState.boardSize - 1 {
                    score += 800.0
                } else if minCol <= 1 && maxCol >= GameState.boardSize - 2 {
                    score += 400.0
                } else if maxCol >= GameState.boardSize - 2 && minCol <= 5 {
                    score += 400.0
                } else if minCol <= 3 && maxCol >= GameState.boardSize - 4 {
                    score += 200.0
                }
            }
        }
        return score
    }

    private func evaluatePotentialConnections(game: GameState, player: GameState.Player) -> Double {
        var score = 0.0
        let pegs = game.pegs.filter { $0.player == player.rawValue }
        for peg in pegs {
            for offset in knightOffsets {
                let newRow = Int(peg.row) + offset.row
                let newCol = Int(peg.col) + offset.col
                if isValidPlacementForPlayer(game: game, player: player, row: newRow, col: newCol) {
                    if player == .red {
                        if Int(peg.row) < 12 && newRow > Int(peg.row) { score += 5.0 }
                        if Int(peg.row) > 12 && newRow < Int(peg.row) { score += 5.0 }
                    } else {
                        if Int(peg.col) < 12 && newCol > Int(peg.col) { score += 5.0 }
                        if Int(peg.col) > 12 && newCol < Int(peg.col) { score += 5.0 }
                    }
                }
            }
        }
        return score
    }

    private func evaluateEdgeProgress(game: GameState, player: GameState.Player) -> Double {
        var score = 0.0
        let pegs = game.pegs.filter { $0.player == player.rawValue }
        for peg in pegs {
            if player == .red {
                let distance = min(Int(peg.row), GameState.boardSize - 1 - Int(peg.row))
                score += Double(max(0, 12 - distance))
            } else {
                let distance = min(Int(peg.col), GameState.boardSize - 1 - Int(peg.col))
                score += Double(max(0, 12 - distance))
            }
        }
        return score
    }

    private func isValidPlacementForPlayer(game: GameState, player: GameState.Player, row: Int, col: Int) -> Bool {
        guard row >= 0, row < GameState.boardSize, col >= 0, col < GameState.boardSize else { return false }
        if game.board[game.boardIndex(row: row, col: col)] != 0 { return false }

        let atTopOrBottom = row == 0 || row == GameState.boardSize - 1
        let atLeftOrRight = col == 0 || col == GameState.boardSize - 1
        if atTopOrBottom && atLeftOrRight { return false }

        if player == .red {
            if atLeftOrRight { return false }
        } else {
            if atTopOrBottom { return false }
        }
        return true
    }

    private func componentMetrics(game: GameState, player: GameState.Player) -> ComponentMetrics {
        let components = findConnectedComponents(game: game, player: player)
        let boardSize = GameState.boardSize

        var maxRowSpan = 0
        var maxColSpan = 0
        var touchesTop = false
        var touchesBottom = false
        var touchesLeft = false
        var touchesRight = false
        var largest: [GameState.Coord] = []
        var minRowOverall = boardSize
        var maxRowOverall = -1
        var minColOverall = boardSize
        var maxColOverall = -1

        for component in components {
            let rows = component.map(\.row)
            let cols = component.map(\.col)
            guard let minRow = rows.min(),
                  let maxRow = rows.max(),
                  let minCol = cols.min(),
                  let maxCol = cols.max()
            else { continue }

            maxRowSpan = max(maxRowSpan, maxRow - minRow)
            maxColSpan = max(maxColSpan, maxCol - minCol)

            if component.count > largest.count {
                largest = component
            }

            minRowOverall = min(minRowOverall, minRow)
            maxRowOverall = max(maxRowOverall, maxRow)
            minColOverall = min(minColOverall, minCol)
            maxColOverall = max(maxColOverall, maxCol)

            if minRow == 0 { touchesTop = true }
            if maxRow == boardSize - 1 { touchesBottom = true }
            if minCol == 0 { touchesLeft = true }
            if maxCol == boardSize - 1 { touchesRight = true }
        }

        return ComponentMetrics(
            components: components,
            maxRowSpan: maxRowSpan,
            maxColSpan: maxColSpan,
            touchesTop: touchesTop,
            touchesBottom: touchesBottom,
            touchesLeft: touchesLeft,
            touchesRight: touchesRight,
            largestComponent: largest,
            minRow: minRowOverall == boardSize ? nil : minRowOverall,
            maxRow: maxRowOverall == -1 ? nil : maxRowOverall,
            minCol: minColOverall == boardSize ? nil : minColOverall,
            maxCol: maxColOverall == -1 ? nil : maxColOverall
        )
    }

    private func computeFrontier(game: GameState, player: GameState.Player) -> FrontierData {
        let boardSize = GameState.boardSize
        var frontier: [GameState.Coord] = []
        var connectors: [GameState.Coord] = []
        var trailing: [GameState.Coord] = []
        var seen = Set<GameState.Coord>()
        let metrics = componentMetrics(game: game, player: player)
        let component = metrics.largestComponent

        guard !component.isEmpty else {
            return FrontierData(frontier: frontier, connectors: connectors, trailing: trailing, metrics: metrics)
        }

        let wantTop = player == .red ? !metrics.touchesTop : false
        let wantBottom = player == .red ? !metrics.touchesBottom : false
        let wantLeft = player == .black ? !metrics.touchesLeft : false
        let wantRight = player == .black ? !metrics.touchesRight : false

        for peg in component {
            for offset in knightOffsets {
                let row = peg.row + offset.row
                let col = peg.col + offset.col
                let coord = GameState.Coord(row: row, col: col)

                if row < 0 || row >= boardSize || col < 0 || col >= boardSize { continue }
                if game.board[game.boardIndex(row: row, col: col)] != 0 { continue }
                if seen.contains(coord) { continue }

                let atTopOrBottom = row == 0 || row == boardSize - 1
                let atLeftOrRight = col == 0 || col == boardSize - 1
                if atTopOrBottom && atLeftOrRight { continue }
                if player == .red && atLeftOrRight { continue }
                if player == .black && atTopOrBottom { continue }

                frontier.append(coord)
                var isConnector = false
                if player == .red {
                    let topThreshold = wantTop ? 5 : 3
                    let bottomThreshold = wantBottom ? 5 : 3
                    if wantTop && row <= topThreshold { isConnector = true }
                    if wantBottom && row >= boardSize - 1 - bottomThreshold { isConnector = true }
                    if !wantTop && !wantBottom && (row <= topThreshold || row >= boardSize - 1 - bottomThreshold) {
                        isConnector = true
                    }
                } else {
                    let leftThreshold = wantLeft ? 5 : 3
                    let rightThreshold = wantRight ? 5 : 3
                    if wantLeft && col <= leftThreshold { isConnector = true }
                    if wantRight && col >= boardSize - 1 - rightThreshold { isConnector = true }
                    if !wantLeft && !wantRight && (col <= leftThreshold || col >= boardSize - 1 - rightThreshold) {
                        isConnector = true
                    }
                }

                if isConnector {
                    connectors.append(coord)
                } else {
                    trailing.append(coord)
                }

                seen.insert(coord)
            }
        }

        return FrontierData(frontier: frontier, connectors: connectors, trailing: trailing, metrics: metrics)
    }

    private func computeConnectorTargets(game: GameState, player: GameState.Player, metrics: ComponentMetrics) -> ConnectorTargets? {
        guard !metrics.largestComponent.isEmpty else { return nil }

        let component = metrics.largestComponent
        let boardSize = GameState.boardSize
        let radius = config.edge.radius

        var minR = boardSize
        var maxR = -1
        var minC = boardSize
        var maxC = -1

        for point in component {
            minR = min(minR, point.row)
            maxR = max(maxR, point.row)
            minC = min(minC, point.col)
            maxC = max(maxC, point.col)
        }

        var targets = Set<GameState.Coord>()

        func addTarget(row: Int, col: Int) {
            guard row >= 0, row < boardSize, col >= 0, col < boardSize else { return }
            if game.board[game.boardIndex(row: row, col: col)] != 0 { return }
            if player == .red && (col == 0 || col == boardSize - 1) { return }
            if player == .black && (row == 0 || row == boardSize - 1) { return }
            targets.insert(GameState.Coord(row: row, col: col))
        }

        if player == .red {
            for c in (minC - radius)...(maxC + radius) {
                addTarget(row: minR - 1, col: c)
                addTarget(row: maxR + 1, col: c)
            }
        } else {
            for r in (minR - radius)...(maxR + radius) {
                addTarget(row: r, col: minC - 1)
                addTarget(row: r, col: maxC + 1)
            }
        }

        return targets.isEmpty ? nil : ConnectorTargets(positions: targets)
    }

    // MARK: - Small utilities

    private func countConnections(game: GameState, move: GameState.Move, color: GameState.Player) -> Int {
        var count = 0
        for offset in knightOffsets {
            let r = move.row + offset.row
            let c = move.col + offset.col
            if r < 0 || r >= GameState.boardSize || c < 0 || c >= GameState.boardSize { continue }
            let idx = game.boardIndex(row: r, col: c)
            if game.board[idx] == color.rawValue {
                count += 1
            }
        }
        return count
    }

    private func minDistance(move: GameState.Move, pegs: [GameState.PegData]) -> Double? {
        guard !pegs.isEmpty else { return nil }
        var best = Double.greatestFiniteMagnitude
        for peg in pegs {
            let dist = abs(Double(move.row) - Double(peg.row)) + abs(Double(move.col) - Double(peg.col))
            if dist < best {
                best = dist
            }
        }
        return best.isFinite ? best : nil
    }

    private func distance(move: GameState.Coord, component: [GameState.Coord]) -> Int {
        var best = Int.max
        for coord in component {
            let dist = abs(move.row - coord.row) + abs(move.col - coord.col)
            if dist < best {
                best = dist
            }
        }
        return best
    }

    private func distance(move: GameState.Coord, cells: [GameState.Coord]) -> Int {
        var best = Int.max
        for coord in cells {
            let dist = abs(move.row - coord.row) + abs(move.col - coord.col)
            if dist < best {
                best = dist
            }
        }
        return best
    }

    private var knightOffsets: [GameState.Coord] {
        [
            .init(row: -2, col: -1), .init(row: -2, col: 1),
            .init(row: -1, col: -2), .init(row: -1, col: 2),
            .init(row: 1, col: -2),  .init(row: 1, col: 2),
            .init(row: 2, col: -1),  .init(row: 2, col: 1)
        ]
    }
}

// MARK: - Supporting types

private struct FeatureAccumulator {
    private(set) var snapshot: [String: Double] = [:]

    mutating func capture(_ key: String, _ value: Double) {
        guard value.isFinite else { return }
        snapshot[key, default: 0.0] += value
    }
}

private struct ComponentMetrics {
    let components: [[GameState.Coord]]
    let maxRowSpan: Int
    let maxColSpan: Int
    let touchesTop: Bool
    let touchesBottom: Bool
    let touchesLeft: Bool
    let touchesRight: Bool
    let largestComponent: [GameState.Coord]
    let minRow: Int?
    let maxRow: Int?
    let minCol: Int?
    let maxCol: Int?
}

private struct FrontierData {
    let frontier: [GameState.Coord]
    let connectors: [GameState.Coord]
    let trailing: [GameState.Coord]
    let metrics: ComponentMetrics
}

private struct ConnectorTargets {
    let positions: Set<GameState.Coord>

    func contains(_ coord: GameState.Coord) -> Bool {
        positions.contains(coord)
    }
}

private struct MovePriorityResult {
    let score: Double
    let features: [String: Double]
    let featureContext: FeatureContext
    let valueModel: ValueModelEvaluator.Result?
}
