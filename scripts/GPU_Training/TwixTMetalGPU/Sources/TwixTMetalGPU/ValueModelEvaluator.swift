import Foundation

/// Lightweight evaluator for the logistic value model used by the JavaScript pipeline.
struct ValueModelEvaluator: Codable {
    struct Result {
        let probability: Double?
        let adjustment: Double?
        let logit: Double?
    }

    private struct Preproc: Codable {
        let standardize: Bool
        let mean: [Double]?
        let std: [Double]?
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case featureKeys = "feature_keys"
        case weights
        case preproc
    }

    let type: String
    let featureKeys: [String]
    let weights: [Double]
    let preproc: Preproc?

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.type = try container.decode(String.self, forKey: .type)
        self.featureKeys = try container.decode([String].self, forKey: .featureKeys)
        self.weights = try container.decode([Double].self, forKey: .weights)
        self.preproc = try container.decodeIfPresent(Preproc.self, forKey: .preproc)
    }

    static func load(from path: String) -> ValueModelEvaluator? {
        guard FileManager.default.fileExists(atPath: path) else { return nil }
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let decoder = JSONDecoder()
            return try decoder.decode(ValueModelEvaluator.self, from: data)
        } catch {
            return nil
        }
    }

    func evaluate(heuristics: [String: Double], featureContext: FeatureContext, scale: Double) -> Result {
        guard weights.count == featureKeys.count + 1 else {
            return Result(probability: nil, adjustment: nil, logit: nil)
        }

        var vector = [Double](repeating: 0.0, count: featureKeys.count)
        for (index, key) in featureKeys.enumerated() {
            if let value = heuristics[key] {
                vector[index] = value
            } else {
                switch key {
                case "turn":
                    vector[index] = Double(featureContext.turn)
                case "playerPegCount":
                    vector[index] = Double(featureContext.playerPegCount)
                case "opponentPegCount":
                    vector[index] = Double(featureContext.opponentPegCount)
                default:
                    vector[index] = 0.0
                }
            }
        }

        let processed: [Double]
        if let preproc = preproc, preproc.standardize,
           let mean = preproc.mean, let std = preproc.std,
           mean.count == vector.count, std.count == vector.count {
            processed = (0..<vector.count).map { index in
                let denom = std[index] == 0 ? 1.0 : std[index]
                return (vector[index] - mean[index]) / denom
            }
        } else {
            processed = vector
        }

        var logit = weights[0]
        for (value, weight) in zip(processed, weights.dropFirst()) {
            logit += weight * value
        }

        let probability = sigmoid(logit)
        let adjustment = (probability - 0.5) * scale
        return Result(probability: probability, adjustment: adjustment, logit: logit)
    }
}

private func sigmoid(_ z: Double) -> Double {
    if z < -35 { return 1e-15 }
    if z > 35 { return 1 - 1e-15 }
    return 1.0 / (1.0 + exp(-z))
}
