import Foundation

/// Heuristics configuration loaded from JSON file
/// Allows tuning AI behavior without recompiling
struct HeuristicsConfig: Codable {
    let general: GeneralHeuristics
    let edge: EdgeHeuristics
    let valueModelScale: Float

    /// Wrapper for JavaScript format compatibility
    private struct RewardsWrapper: Codable {
        let rewards: RewardsContent
        let valueModelScale: Float

        struct RewardsContent: Codable {
            let general: GeneralHeuristics
            let edge: EdgeHeuristics
        }
    }

    struct GeneralHeuristics: Codable {
        let friendlyConnection: Float
        let opponentConnection: Float
        let friendlyDistance: Float
        let opponentDistance: Float
        let goalDistance: Float
        let centerBias: Float
        let isolated: Float
        let redGlobalMultiplier: Float
        let blackGlobalScale: Float
        let redBaseBonus: Float
        let blackBasePenalty: Float
        let lateGameStart: Float
        let lateGamePressure: Float
    }

    struct EdgeHeuristics: Codable {
        let radius: Int
        let offense: OffenseHeuristics
        let defense: DefenseHeuristics

        struct OffenseHeuristics: Codable {
            let gapDecay: Float
            let connectorBonus: Float
            let finishThreshold: Float
            let finishBonusBase: Float
            let finishPenaltyBase: Float
            let connectorTargetBonus: Float
            let redFinishExtra: Float
            let redSpanGainMultiplier: Float
            let redGapDecayMultiplier: Float
            let redFinishPenaltyFactor: Float
            let blackFinishScaleMultiplier: Float
            let blackSpanGainMultiplier: Float
            let blackDoubleCoverageScale: Float
            let firstEdgeTouchBlack: Float
            let firstEdgeTouchRed: Float
            let redDoubleCoverageBonus: Float
        }

        struct DefenseHeuristics: Codable {
            let blockBonus: Float
            let missPenalty: Float
        }
    }

    /// Load configuration from JSON file
    /// Supports both JavaScript format (with "rewards" wrapper) and direct format
    static func load(from path: String) throws -> HeuristicsConfig {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let decoder = JSONDecoder()

        // Try JavaScript format first (with rewards wrapper)
        if let wrapper = try? decoder.decode(RewardsWrapper.self, from: data) {
            return HeuristicsConfig(
                general: wrapper.rewards.general,
                edge: wrapper.rewards.edge,
                valueModelScale: wrapper.valueModelScale
            )
        }

        // Fall back to direct format
        return try decoder.decode(HeuristicsConfig.self, from: data)
    }

    /// Default configuration (fallback if file not found)
    static let `default` = HeuristicsConfig(
        general: GeneralHeuristics(
            friendlyConnection: 12,
            opponentConnection: 35,
            friendlyDistance: 3,
            opponentDistance: 12,
            goalDistance: 1.2,
            centerBias: 0.5,
            isolated: 10,
            redGlobalMultiplier: 1.18,
            blackGlobalScale: 0.82,
            redBaseBonus: 150,
            blackBasePenalty: 360,
            lateGameStart: 60,
            lateGamePressure: 0
        ),
        edge: EdgeHeuristics(
            radius: 3,
            offense: EdgeHeuristics.OffenseHeuristics(
                gapDecay: 23,
                connectorBonus: 620,
                finishThreshold: 4,
                finishBonusBase: 3400,
                finishPenaltyBase: 2050,
                connectorTargetBonus: 500,
                redFinishExtra: 2000,
                redSpanGainMultiplier: 2.2,
                redGapDecayMultiplier: 2.0,
                redFinishPenaltyFactor: 0.78,
                blackFinishScaleMultiplier: 0.99,
                blackSpanGainMultiplier: 1.05,
                blackDoubleCoverageScale: 0.8,
                firstEdgeTouchBlack: 500,
                firstEdgeTouchRed: 680,
                redDoubleCoverageBonus: 280
            ),
            defense: EdgeHeuristics.DefenseHeuristics(
                blockBonus: 900,
                missPenalty: 350
            )
        ),
        valueModelScale: 600
    )
}
