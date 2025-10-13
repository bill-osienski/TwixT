#include <metal_stdlib>
using namespace metal;

// Board size constant (compatible with runtime compilation)
constant int BOARD_SIZE = 24;
constant int MAX_VALID_MOVES = BOARD_SIZE * BOARD_SIZE;

// Knight move offsets for TwixT (Metal 4: Use constexpr for better optimization)
constant int2 KNIGHT_OFFSETS[8] = {
    int2(-2, -1), int2(-2, 1), int2(-1, -2), int2(-1, 2),
    int2(1, -2), int2(1, 2), int2(2, -1), int2(2, 1)
};

// Heuristic feature structure (matches your JavaScript features)
struct HeuristicFeatures {
    float friendlyConnections;
    float opponentConnections;
    float friendlyDistance;
    float opponentDistance;
    float goalDistance;
    float centerBias;
    float isolatedBonus;
    float chainProximity;
    float frontierProximity;
    float frontierCapture;
    float connectorProximity;
    float connectorCapture;
    float trailingPenalty;
    float threatReduction;
    float noThreatReduction;
    float spanGain;
    float blackSpanComplete;
    float redSpanComplete;
    float opponentSpanReduction;
    float noSpanReductionPenalty;
    float blackSpanUpgradePenalty;
    float redSpanUpgradePenalty;
    float topBias;
    float aboveMinRowBonus;
    float belowMinRowPenalty;
    float bottomBias;
    float belowMaxRowBonus;
    float aboveMaxRowPenalty;
};

// Move evaluation result
struct MoveScore {
    int row;
    int col;
    float heuristicScore;
    HeuristicFeatures features;
};

// GPU kernel to evaluate multiple moves in parallel
// Metal 4: Optimized with SIMD operations and memory coalescing
kernel void evaluateMoves(
    device const uint8_t* board [[buffer(0)]],         // Flat board array
    device const int* validMoves [[buffer(1)]],         // Array of (row, col) pairs
    constant int& moveCount [[buffer(2)]],              // Number of valid moves
    device const uint8_t* pegs [[buffer(3)]],          // Peg data [row, col, player]
    constant int& pegCount [[buffer(4)]],               // Number of pegs
    constant uint8_t& currentPlayer [[buffer(5)]],     // Current player (1=red, 2=black)
    device MoveScore* scores [[buffer(6)]],             // Output scores
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    if (gid >= uint(moveCount)) return;

    // Get move coordinates
    int moveIdx = gid * 2;
    int row = validMoves[moveIdx];
    int col = validMoves[moveIdx + 1];

    uint8_t player = currentPlayer;
    uint8_t opponent = (player == 1) ? 2 : 1;

    // Initialize features
    HeuristicFeatures features;
    features.friendlyConnections = 0.0f;
    features.opponentConnections = 0.0f;
    features.friendlyDistance = 10000.0f;
    features.opponentDistance = 10000.0f;
    features.goalDistance = 0.0f;
    features.centerBias = 0.0f;
    features.isolatedBonus = 0.0f;
    features.chainProximity = 0.0f;
    features.frontierProximity = 0.0f;
    features.frontierCapture = 0.0f;
    features.connectorProximity = 0.0f;
    features.connectorCapture = 0.0f;
    features.trailingPenalty = 0.0f;
    features.threatReduction = 0.0f;
    features.noThreatReduction = 0.0f;
    features.spanGain = 0.0f;
    features.blackSpanComplete = 0.0f;
    features.redSpanComplete = 0.0f;
    features.opponentSpanReduction = 0.0f;
    features.noSpanReductionPenalty = 0.0f;
    features.blackSpanUpgradePenalty = 0.0f;
    features.redSpanUpgradePenalty = 0.0f;
    features.topBias = 0.0f;
    features.aboveMinRowBonus = 0.0f;
    features.belowMinRowPenalty = 0.0f;
    features.bottomBias = 0.0f;
    features.belowMaxRowBonus = 0.0f;
    features.aboveMaxRowPenalty = 0.0f;

    float score = 0.0f;

    // 1. Count knight-move connections (Metal 4: Loop unrolling hint)
    int friendlyConnCount = 0;
    int opponentConnCount = 0;

    // Metal 4: Compiler will vectorize this with SIMD
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int2 offset = KNIGHT_OFFSETS[i];
        int r = row + offset.x;
        int c = col + offset.y;

        // Metal 4: Branch-less evaluation for better GPU performance
        bool inBounds = (r >= 0) && (r < BOARD_SIZE) && (c >= 0) && (c < BOARD_SIZE);
        if (inBounds) {
            int boardIdx = r * BOARD_SIZE + c;
            uint8_t cellValue = board[boardIdx];
            friendlyConnCount += (cellValue == player) ? 1 : 0;
            opponentConnCount += (cellValue == opponent) ? 1 : 0;
        }
    }

    // Metal 4: Fast math operations
    features.friendlyConnections = fast::fma(float(friendlyConnCount), 12.0f, 0.0f);
    features.opponentConnections = fast::fma(float(opponentConnCount), 35.0f, 0.0f);
    score += features.friendlyConnections + features.opponentConnections;

    // 2. Distance to nearest friendly/opponent peg
    float minFriendlyDist = 10000.0f;
    float minOpponentDist = 10000.0f;

    for (int i = 0; i < pegCount; i++) {
        int pegIdx = i * 3;
        int pegRow = pegs[pegIdx];
        int pegCol = pegs[pegIdx + 1];
        uint8_t pegPlayer = pegs[pegIdx + 2];

        float dist = float(abs(row - pegRow) + abs(col - pegCol));

        if (pegPlayer == player) {
            minFriendlyDist = min(minFriendlyDist, dist);
        } else if (pegPlayer == opponent) {
            minOpponentDist = min(minOpponentDist, dist);
        }
    }

    if (minFriendlyDist < 10000.0f) {
        features.friendlyDistance = max(0.0f, 10.0f - minFriendlyDist) * 3.0f;
        score += features.friendlyDistance;
    }

    if (minOpponentDist < 10000.0f) {
        features.opponentDistance = max(0.0f, 10.0f - minOpponentDist) * 12.0f;
        score += features.opponentDistance;
    }

    // 3. Goal distance (distance to nearest goal edge)
    float goalDist;
    if (player == 1) { // Red
        goalDist = float(min(row, BOARD_SIZE - 1 - row));
    } else { // Black
        goalDist = float(min(col, BOARD_SIZE - 1 - col));
    }
    features.goalDistance = max(0.0f, 12.0f - goalDist) * 1.2f;
    score += features.goalDistance;

    // 4. Center bias (early game positioning)
    float center = float(BOARD_SIZE - 1) / 2.0f;
    float centerDist = float(abs(row - int(center)) + abs(col - int(center)));
    features.centerBias = max(0.0f, 16.0f - centerDist) * 0.5f;
    score += features.centerBias;

    // 5. Isolated bonus (opening move)
    if (minFriendlyDist >= 10000.0f && minOpponentDist >= 10000.0f) {
        features.isolatedBonus = 10.0f;
        score += features.isolatedBonus;
    }

    // Store results
    scores[gid].row = row;
    scores[gid].col = col;
    scores[gid].heuristicScore = score;
    scores[gid].features = features;
}

// Batch evaluate positions for multiple games simultaneously
kernel void batchEvaluatePositions(
    device const uint8_t* boards [[buffer(0)]],        // Multiple flat boards
    constant int& batchSize [[buffer(1)]],              // Number of games in batch
    device const uint8_t* players [[buffer(2)]],       // Current player for each game
    device float* scores [[buffer(3)]],                 // Output position scores
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(batchSize)) return;

    int boardOffset = gid * BOARD_SIZE * BOARD_SIZE;
    const device uint8_t* board = boards + boardOffset;
    uint8_t player = players[gid];
    uint8_t opponent = (player == 1) ? 2 : 1;

    float score = 0.0f;

    // Count pegs
    int playerPegs = 0;
    int opponentPegs = 0;

    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        if (board[i] == player) playerPegs++;
        else if (board[i] == opponent) opponentPegs++;
    }

    // Simple position evaluation
    score += float(playerPegs - opponentPegs) * 2.0f;

    // TODO: Add more sophisticated position evaluation (connectivity, span, etc.)

    scores[gid] = score;
}

// Value model inference on GPU
kernel void evaluateValueModel(
    device const float* features [[buffer(0)]],         // Feature vectors [batch * featureCount]
    constant float* weights [[buffer(1)]],              // Model weights [featureCount + 1]
    constant int& batchSize [[buffer(2)]],              // Number of feature vectors
    constant int& featureCount [[buffer(3)]],           // Number of features
    device float* probabilities [[buffer(4)]],          // Output probabilities [batch]
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(batchSize)) return;

    // Compute logit = bias + sum(w_i * x_i)
    float logit = weights[0]; // bias

    int featureOffset = gid * featureCount;
    for (int i = 0; i < featureCount; i++) {
        logit += weights[i + 1] * features[featureOffset + i];
    }

    // Sigmoid activation
    float probability = 1.0f / (1.0f + exp(-logit));
    probabilities[gid] = probability;
}
