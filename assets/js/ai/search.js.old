import { evaluateMove, evaluatePosition, connectivityScore, componentMetrics, computeFrontier } from './heuristics.js';
import { evaluateValueModel, maybeLoadValueModel } from './valueModel.js';

let config;
if (typeof process !== 'undefined' && process?.versions?.node) {
    const { createRequire } = await import('module');
    const require = createRequire(import.meta.url);
    config = require('./search.json');
} else {
    try {
        const jsonModule = await import('./search.json', { assert: { type: 'json' } });
        config = jsonModule?.default ?? jsonModule;
    } catch {
        const jsonModule = await import('./search.json', { with: { type: 'json' } });
        config = jsonModule?.default ?? jsonModule;
    }
}

if (!config?.rewards?.general || !config?.rewards?.edge?.offense || !config?.rewards?.edge?.defense) {
    throw new Error('Invalid search.json configuration: expected rewards.general, rewards.edge.offense, and rewards.edge.defense.');
}

const KNIGHT_OFFSETS = [
    [-2, -1], [-2, 1], [-1, -2], [-1, 2],
    [1, -2], [1, 2], [2, -1], [2, 1]
];

const VALUE_MODEL_SCALE = typeof config.valueModelScale === 'number'
    ? config.valueModelScale
    : 600;

const REWARDS = Object.freeze({
    general: Object.freeze({ ...config.rewards.general }),
    edge: Object.freeze({
        radius: config.rewards.edge.radius,
        offense: Object.freeze({ ...config.rewards.edge.offense }),
        defense: Object.freeze({ ...config.rewards.edge.defense })
    })
});

const sealedLaneLogEveryRaw = config?.debug?.performance?.sealedLaneLogEvery;
const DEBUG_OPTIONS = Object.freeze({
    performance: Object.freeze({
        sealedLane: !!(config?.debug?.performance?.sealedLane),
        sealedLaneLogEvery: Number.isFinite(sealedLaneLogEveryRaw) && sealedLaneLogEveryRaw > 0
            ? Math.floor(sealedLaneLogEveryRaw)
            : 0
    })
});

const SEALED_LANE_DEBUG = {
    enabled: DEBUG_OPTIONS.performance.sealedLane,
    logEvery: DEBUG_OPTIONS.performance.sealedLaneLogEvery,
    stats: (() => {
        const stats = {
            enabled: DEBUG_OPTIONS.performance.sealedLane,
            calls: 0,
            openPaths: 0,
            sealed: 0,
            totalMs: 0,
            maxMs: 0,
            bridgeChecks: 0,
            bridgeCacheMisses: 0,
            nodesVisited: 0,
            enqueued: 0,
            reasons: Object.create(null),
            reset() {
                this.calls = 0;
                this.openPaths = 0;
                this.sealed = 0;
                this.totalMs = 0;
                this.maxMs = 0;
                this.bridgeChecks = 0;
                this.bridgeCacheMisses = 0;
                this.nodesVisited = 0;
                this.enqueued = 0;
                for (const key of Object.keys(this.reasons)) {
                    delete this.reasons[key];
                }
            }
        };
        return stats;
    })()
};

if (SEALED_LANE_DEBUG.enabled && typeof globalThis !== 'undefined') {
    globalThis.__TwixTSealedLaneStats = SEALED_LANE_DEBUG.stats;
}

const HEURISTIC_STATS = (() => {
    if (typeof globalThis === 'undefined') return null;
    if (!globalThis.__TwixTAIStats) {
        globalThis.__TwixTAIStats = { perDepth: Object.create(null) };
        if (typeof process !== 'undefined' && process?.on && !process.__TwixTAIStatsHooked) {
            process.__TwixTAIStatsHooked = true;
            process.once('exit', () => {
                try {
                    const stats = globalThis.__TwixTAIStats;
                    if (!stats) return;
                    const summary = {};
                    for (const [depth, entries] of Object.entries(stats.perDepth || {})) {
                        summary[depth] = {};
                        for (const [key, value] of Object.entries(entries)) {
                            summary[depth][key] = {
                                red: { count: value.red.count, sum: value.red.sum },
                                black: { count: value.black.count, sum: value.black.sum }
                            };
                        }
                    }
                    if (Object.keys(summary).length) {
                        console.info('[TwixTAI] heuristic stats', JSON.stringify(summary));
                    }
                } catch {
                    // ignore logging issues during shutdown
                }
            });
        }
    }
    return globalThis.__TwixTAIStats;
})();

const now = typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? () => performance.now()
    : () => Date.now();

if (SEALED_LANE_DEBUG.enabled &&
    typeof process !== 'undefined' &&
    process?.on &&
    !process.__twixtSealedLaneExitHooked) {

    process.__twixtSealedLaneExitHooked = true;
    process.once('exit', () => {
        const stats = SEALED_LANE_DEBUG.stats;
        const avgMs = stats.calls ? stats.totalMs / stats.calls : 0;
        const summary = {
            calls: stats.calls,
            openPaths: stats.openPaths,
            sealed: stats.sealed,
            avgMs,
            maxMs: stats.maxMs,
            bridgeChecks: stats.bridgeChecks,
            bridgeCacheMisses: stats.bridgeCacheMisses,
            nodesVisited: stats.nodesVisited,
            enqueued: stats.enqueued,
            reasons: { ...stats.reasons }
        };
        const logger = (typeof console !== 'undefined' && console.info) ? console.info : null;
        if (logger) {
            logger('[TwixTAI] sealed lane summary', summary);
        }
    });
}

function isLegalPlacementForPlayer(game, player, row, col) {
    const boardSize = game.boardSize;
    if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) {
        return false;
    }
    if (game.board[row][col] !== null) {
        return false;
    }
    const onTopOrBottom = row === 0 || row === boardSize - 1;
    const onLeftOrRight = col === 0 || col === boardSize - 1;
    if (onTopOrBottom && onLeftOrRight) {
        return false;
    }
    if (player === 'red') {
        return !(col === 0 || col === boardSize - 1);
    }
    return !(row === 0 || row === boardSize - 1);
}

function isGoalEdgeCoordinate(player, row, col, boardSize) {
    if (player === 'red') {
        if (row !== 0 && row !== boardSize - 1) {
            return false;
        }
        return col > 0 && col < boardSize - 1;
    }
    if (col !== 0 && col !== boardSize - 1) {
        return false;
    }
    return row > 0 && row < boardSize - 1;
}

function hasReachableGoalEdge(game, player, metrics) {
    const component = metrics?.largestComponent;
    if (!component || component.length === 0) {
        if (SEALED_LANE_DEBUG.enabled) {
            const stats = SEALED_LANE_DEBUG.stats;
            stats.calls++;
            stats.sealed++;
            stats.reasons.emptyComponent = (stats.reasons.emptyComponent || 0) + 1;
        }
        return false;
    }

    const boardSize = game.boardSize;
    const board = game.board;
    const canCheckCross = typeof game?.bridgesCross === 'function';
    const targetSet = new Set();

    if (player === 'red') {
        if (!metrics.touchesTop) targetSet.add(0);
        if (!metrics.touchesBottom) targetSet.add(boardSize - 1);
    } else {
        if (!metrics.touchesLeft) targetSet.add(0);
        if (!metrics.touchesRight) targetSet.add(boardSize - 1);
    }

    const track = SEALED_LANE_DEBUG.enabled;
    const startTime = track ? now() : 0;
    let localBridgeChecks = 0;
    let localCacheMisses = 0;
    let localVisited = 0;
    let localEnqueued = 0;

    const finish = (value, reason) => {
        if (track) {
            const elapsed = now() - startTime;
            const stats = SEALED_LANE_DEBUG.stats;
            stats.calls++;
            if (value) {
                stats.openPaths++;
            } else {
                stats.sealed++;
            }
            stats.totalMs += elapsed;
            if (elapsed > stats.maxMs) {
                stats.maxMs = elapsed;
            }
            stats.bridgeChecks += localBridgeChecks;
            stats.bridgeCacheMisses += localCacheMisses;
            stats.nodesVisited += localVisited;
            stats.enqueued += localEnqueued;
            stats.reasons[reason] = (stats.reasons[reason] || 0) + 1;

            if (SEALED_LANE_DEBUG.logEvery &&
                stats.calls % SEALED_LANE_DEBUG.logEvery === 0 &&
                typeof console !== 'undefined' &&
                typeof console.info === 'function') {

                const avgMs = stats.calls ? stats.totalMs / stats.calls : 0;
                console.info('[TwixTAI] sealed lane stats', {
                    calls: stats.calls,
                    openPaths: stats.openPaths,
                    sealed: stats.sealed,
                    avgMs,
                    maxMs: stats.maxMs,
                    bridgeChecks: stats.bridgeChecks,
                    bridgeCacheMisses: stats.bridgeCacheMisses
                });
            }
        }
        return value;
    };

    if (targetSet.size === 0) {
        return finish(true, 'alreadyTouching');
    }

    const visited = new Set();
    const queue = [];
    let head = 0;

    const enqueue = (row, col, type) => {
        const key = `${type}:${row}:${col}`;
        if (visited.has(key)) return;
        visited.add(key);
        queue.push({ row, col, type });
        if (track) localEnqueued++;
    };

    const bridgesCross = (r1, c1, r2, c2) => {
        if (!canCheckCross) {
            return false;
        }
        if (track) localBridgeChecks++;
        const lowRow = r1 < r2 || (r1 === r2 && c1 <= c2);
        const key = lowRow
            ? `${r1}:${c1}|${r2}:${c2}`
            : `${r2}:${c2}|${r1}:${c1}`;
        if (bridgesCross.cache.has(key)) {
            return bridgesCross.cache.get(key);
        }
        if (track) localCacheMisses++;
        const crosses = game.bridgesCross(r1, c1, r2, c2);
        bridgesCross.cache.set(key, crosses);
        return crosses;
    };
    bridgesCross.cache = new Map();

    for (const peg of component) {
        enqueue(peg.row, peg.col, 'peg');
    }

    while (head < queue.length) {
        const current = queue[head++];
        const { row, col, type } = current;
        if (track) localVisited++;

        if (player === 'red') {
            if (targetSet.has(row) && isGoalEdgeCoordinate(player, row, col, boardSize)) {
                if (type === 'peg' || (type === 'empty' && isLegalPlacementForPlayer(game, player, row, col))) {
                    return finish(true, 'goalReachable');
                }
            }
        } else if (targetSet.has(col) && isGoalEdgeCoordinate(player, row, col, boardSize)) {
            if (type === 'peg' || (type === 'empty' && isLegalPlacementForPlayer(game, player, row, col))) {
                return finish(true, 'goalReachable');
            }
        }

        if (type === 'empty' && !isLegalPlacementForPlayer(game, player, row, col)) {
            continue;
        }

        for (const [dr, dc] of KNIGHT_OFFSETS) {
            const nr = row + dr;
            const nc = col + dc;

            if (nr < 0 || nr >= boardSize || nc < 0 || nc >= boardSize) {
                continue;
            }

            const occupant = board[nr][nc];
            let nextType = null;
            if (occupant === null) {
                nextType = 'empty';
            } else if (occupant === player) {
                nextType = 'peg';
            } else {
                continue;
            }

            if (bridgesCross(row, col, nr, nc)) {
                continue;
            }

            enqueue(nr, nc, nextType);
        }
    }

    return finish(false, 'sealed');
}

function computeConnectorTargets(game, player, metrics) {
    if (!metrics || !metrics.largestComponent || metrics.largestComponent.length === 0) {
        return null;
    }

    const component = metrics.largestComponent;
    const boardSize = game.boardSize;
    const radius = REWARDS.edge.radius;

    let minR = boardSize, maxR = -1, minC = boardSize, maxC = -1;
    for (const p of component) {
        if (p.row < minR) minR = p.row;
        if (p.row > maxR) maxR = p.row;
        if (p.col < minC) minC = p.col;
        if (p.col > maxC) maxC = p.col;
    }

    const targets = new Set();
    const addTarget = (row, col) => {
        if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) return;
        if (game.board[row][col] !== null) return;
        if (player === 'red' && (col === 0 || col === boardSize - 1)) return;
        if (player === 'black' && (row === 0 || row === boardSize - 1)) return;
        targets.add(`${row}:${col}`);
    };

    if (player === 'red') {
        for (let c = minC - radius; c <= maxC + radius; c++) {
            addTarget(minR - 1, c);
            addTarget(maxR + 1, c);
        }
    } else {
        for (let r = minR - radius; r <= maxR + radius; r++) {
            addTarget(r, minC - 1);
            addTarget(r, maxC + 1);
        }
    }

    return targets.size ? targets : null;
}

export default class TwixTAI {
    constructor(game, player = null) {
        this.game = game;
        this.player = player ?? (game && typeof game.aiPlayer === 'string' ? game.aiPlayer : null);
        this.rootDepth = 0;
        this.sealedLaneCache = new Map();
        this.recordStat = (key, playerSide, amount = 1) => {
            if (!HEURISTIC_STATS || !this.rootDepth) {
                return;
            }
            const depthKey = String(this.rootDepth);
            const perDepth = HEURISTIC_STATS.perDepth[depthKey] ||= Object.create(null);
            const entry = perDepth[key] ||= {
                red: { count: 0, sum: 0 },
                black: { count: 0, sum: 0 }
            };
            const bucket = playerSide === 'red' ? entry.red : entry.black;
            bucket.count += 1;
            bucket.sum += amount;
        };
        if (typeof window !== 'undefined') {
            if (window.TwixTAI_DEBUG === undefined) {
                window.TwixTAI_DEBUG = false;
            }
            if (!window.enableTwixTAIDebug) {
                window.enableTwixTAIDebug = (flag = true) => {
                    window.TwixTAI_DEBUG = !!flag;

                };
            }
            window.__latestTwixTAI = this;
            if (!window.downloadTwixTAILog && typeof document !== 'undefined') {
                window.downloadTwixTAILog = (overrideData) => {
                    const instance = window.__latestTwixTAI;
                    if (!instance) {

                        return;
                    }

                    const payload = overrideData || instance.moveTrace;
                    if (!payload || payload.length === 0) {

                        return;
                    }

                    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = `twixt-ai-log-${Date.now()}.json`;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    URL.revokeObjectURL(url);
                };
            }
        }
        this.debugEnabled = typeof window !== 'undefined' ? !!window.TwixTAI_DEBUG : false;
        this.lastHeuristicBreakdown = null;
        this.moveTrace = [];
        this.currentHeuristic = null;
        this.lastHeuristicFeatures = null;
        this.lastFeatureContext = null;
        this.lastValueModelProbability = null;
        this.lastValueModelLogit = null;
        this.lastValueModelAdjustment = null;
        this.lastChosenFeatures = null;
        this.lastChosenContext = null;
        this.lastChosenValueModel = null;
        this.lastChosenHeuristicScore = null;

        maybeLoadValueModel();
    }

    setPlayer(player) {
        if (player) {
            this.player = player;
        }
    }

    getPlayer() {
        if (this.player) {
            return this.player;
        }
        if (this.game && typeof this.game.aiPlayer === 'string') {
            return this.game.aiPlayer;
        }
        return 'black';
    }

    getBestMove() {
        const depthMap = this.game.aiDepth || { easy: 2, medium: 3, hard: 4 };
        const difficulty = this.game.aiDifficulty || 'medium';

        // Prefer an explicit override (e.g., from selfPlay.js)
        let depth = (Number.isFinite(this.rootDepth) && this.rootDepth > 0)
          ? this.rootDepth
          : (depthMap[difficulty] || 2);

        this.rootDepth = depth;
        this.debugEnabled = typeof window !== 'undefined' ? !!window.TwixTAI_DEBUG : this.debugEnabled;

        const allMoves = this.game.getValidMoves();
        const aiPlayer = this.getPlayer();
        const opponent = aiPlayer === 'red' ? 'black' : 'red';
        const opponentThreat = connectivityScore(this.game, opponent);
        const friendlyMetrics = componentMetrics(this.game, aiPlayer);
        const friendlyConnectorTargets = computeConnectorTargets(this.game, aiPlayer, friendlyMetrics);
        const { frontier: opponentFrontier, connectors: opponentConnectors, trailing: opponentTrailing, metrics: opponentMetrics } = computeFrontier(this.game, opponent);
        const opponentConnectorTargets = computeConnectorTargets(this.game, opponent, opponentMetrics);
        const candidateMoves = this.orderMoves(
            allMoves,
            aiPlayer,
            depth,
            opponent,
            opponentThreat,
            friendlyMetrics,
            friendlyConnectorTargets,
            opponentConnectorTargets,
            opponentMetrics,
            opponentFrontier,
            opponentConnectors,
            opponentTrailing
        );
        const moves = candidateMoves.length > 0 ? candidateMoves : allMoves;

        if (moves.length === 0) return null;
        if (moves.length === 1) return moves[0];

        const spanValue = opponentMetrics
            ? (opponent === 'red' ? opponentMetrics.maxRowSpan : opponentMetrics.maxColSpan)
            : 0;
        const largestLength = opponentMetrics?.largestComponent?.length || 0;
        const opponentUrgent = spanValue >= Math.max(6, Math.floor(this.game.boardSize / 4)) || largestLength >= 6;

        const scoredMoves = [];
        const moveDetails = [];

        const originalPlayer = this.game.currentPlayer;

        // Hoist peg lists once per ply (don’t recompute per candidate)
        const friendlyPegsForEval = this.game.pegs.filter(p => p.player === aiPlayer);
        const opponentPegsForEval = this.game.pegs.filter(p => p.player === opponent);

        for (const move of moves) {
          const heuristicScore = this.movePriority(
            move,
            aiPlayer,
            friendlyPegsForEval,
            opponentPegsForEval,
            opponent,
            opponentThreat,
            friendlyMetrics,
            friendlyConnectorTargets,
            opponentConnectorTargets,
            opponentMetrics,
            opponentFrontier,
            opponentConnectors,
            opponentTrailing,
            opponentUrgent
          );

          const featureSnapshot = this.lastHeuristicFeatures ? { ...this.lastHeuristicFeatures } : null;
          const heuristicBreakdown = this.debugEnabled && this.lastHeuristicBreakdown
            ? { ...this.lastHeuristicBreakdown }
            : null;
          const valueModelSnapshot = (this.lastValueModelProbability !== null || this.lastValueModelAdjustment !== null)
            ? {
                probability: this.lastValueModelProbability,
                adjustment:  this.lastValueModelAdjustment,
                logit:       this.lastValueModelLogit
              }
            : null;
          const featureContext = this.lastFeatureContext ? { ...this.lastFeatureContext } : null;

            // Simulate the move for immediate/position/minimax
            this.game.currentPlayer = aiPlayer;
            const success = this.game.placePeg(move.row, move.col);
            if (!success) continue;

            // If this move already wins, snap it up immediately.
            if (typeof this.game.checkWin === 'function' && this.game.checkWin(aiPlayer)) {
              // optional trace (safe even if `capture` isn't defined)
              if (typeof capture === 'function') capture('immediateWin', 10000);

              this.game.undo();                         // revert the simulation
              this.game.currentPlayer = originalPlayer; // restore player
              return move;                              // pick the winning move now
            }

            // --- Near-finish bonus (one move from spanning both edges) ---
            let finishBonus = 0;
            if (typeof componentMetrics === 'function') {
              const postMetrics = componentMetrics(this.game, aiPlayer);

              // Largest component bounding box AFTER this move
              const lc = postMetrics.largestComponent || [];
              if (lc.length > 0) {
                let minR = Infinity, maxR = -Infinity, minC = Infinity, maxC = -Infinity;
                for (const p of lc) {
                  if (p.row < minR) minR = p.row;
                  if (p.row > maxR) maxR = p.row;
                  if (p.col < minC) minC = p.col;
                  if (p.col > maxC) maxC = p.col;
                }

                const N = this.game.boardSize;
                const touchesTop    = !!postMetrics.touchesTop;
                const touchesBottom = !!postMetrics.touchesBottom;
                const touchesLeft   = !!postMetrics.touchesLeft;
                const touchesRight  = !!postMetrics.touchesRight;

                // "Band next to the edge" is index 1 or N-2 on a 0..N-1 board
                const redNearFinish =
                  aiPlayer === 'red' &&
                  ((touchesTop && maxR >= N - 2) || (touchesBottom && minR <= 1));

                const blackNearFinish =
                  aiPlayer === 'black' &&
                  ((touchesLeft && maxC >= N - 2) || (touchesRight && minC <= 1));

                if (redNearFinish || blackNearFinish) {
                  const nearFinishBonus = REWARDS.edge.offense.nearFinishBonus ?? 0;
                  if (nearFinishBonus !== 0) {
                    if (typeof capture === 'function') capture('nearSpanFinish', nearFinishBonus);
                    finishBonus = nearFinishBonus;
                  }
                }
              }
            }

            // Otherwise, keep scoring normally
            const immediateScore = evaluateMove(this.game, move, aiPlayer);
            const positionScore  = evaluatePosition(this.game, aiPlayer);
            const minimaxScore   = (depth <= 1)
              ? positionScore
              : this.minimax(depth - 1, false, -Infinity, Infinity, aiPlayer);

            this.game.undo();

            const totalScore =
              minimaxScore +
              immediateScore * 5 +
              positionScore  * 0.1 +
              finishBonus;

            const detail = {
              move,
              totalScore,
              minimaxScore,
              immediateScore,
              positionScore,
              heuristicScore,
              heuristics: heuristicBreakdown,
              features: featureSnapshot,
              featureContext,
              valueModel: valueModelSnapshot
            };

            scoredMoves.push({ move, score: totalScore, detail });
            moveDetails.push(detail);
        }

        this.game.currentPlayer = originalPlayer;

        if (scoredMoves.length === 0) {
            return moves[0];
        }

        scoredMoves.sort((a, b) => b.score - a.score);

        if (this.debugEnabled) {
            const snapshot = {
                timestamp: Date.now(),
                player: aiPlayer,
                depth,
                board: this.debugSnapshot(),
                moves: moveDetails
            };

            this.moveTrace.push(snapshot);
        }

        const bestDetail = scoredMoves[0] ? scoredMoves[0].detail : null;
        this.lastChosenFeatures = bestDetail && bestDetail.features ? { ...bestDetail.features } : null;
        this.lastChosenContext = bestDetail && bestDetail.featureContext ? { ...bestDetail.featureContext } : null;
        this.lastChosenValueModel = bestDetail && bestDetail.valueModel ? { ...bestDetail.valueModel } : null;
        this.lastChosenHeuristicScore = bestDetail ? bestDetail.heuristicScore : null;

        const randomFactor = difficulty === 'easy' ? 0.3 : difficulty === 'medium' ? 0.1 : 0.02;

        if (Math.random() < randomFactor) {
            const topChoices = difficulty === 'easy' ? 5 : difficulty === 'medium' ? 3 : 2;
            const topMoves = scoredMoves.slice(0, Math.min(topChoices, scoredMoves.length));
            return topMoves[Math.floor(Math.random() * topMoves.length)].move;
        }

        return scoredMoves[0].move;
    }

    minimax(depth, isMaximizing, alpha = -Infinity, beta = Infinity, rootPlayer = null) {
        const aiPlayer = rootPlayer || this.getPlayer();
        if (depth === 0 || this.game.gameOver) {
            return evaluatePosition(this.game, aiPlayer);
        }

        const allMoves = this.game.getValidMoves();
        const opponent = this.game.currentPlayer === 'red' ? 'black' : 'red';
        const opponentThreat = connectivityScore(this.game, opponent);
        const friendlyMetrics = componentMetrics(this.game, this.game.currentPlayer);
        const friendlyConnectorTargets = computeConnectorTargets(this.game, this.game.currentPlayer, friendlyMetrics);
        const { frontier: opponentFrontier, connectors: opponentConnectors, trailing: opponentTrailing, metrics: opponentMetrics } = computeFrontier(this.game, opponent);
        let moves = this.orderMoves(
            allMoves,
            this.game.currentPlayer,
            depth,
            opponent,
            opponentThreat,
            friendlyMetrics,
            friendlyConnectorTargets,
            opponentMetrics,
            opponentFrontier,
            opponentConnectors,
            opponentTrailing
        );
        if (moves.length === 0) {
            moves = allMoves;
        }

        if (moves.length === 0) {
            return evaluatePosition(this.game, aiPlayer);
        }

        if (isMaximizing) {
            let maxScore = -Infinity;
            for (const move of moves) {
                const success = this.game.placePeg(move.row, move.col);
                if (!success) {
                    continue;
                }

                const score = this.minimax(depth - 1, false, alpha, beta, aiPlayer);
                maxScore = Math.max(maxScore, score);

                this.game.undo();

                alpha = Math.max(alpha, score);
                if (beta <= alpha) break;
            }
            return maxScore;
        }

        let minScore = Infinity;
        for (const move of moves) {
            const success = this.game.placePeg(move.row, move.col);
            if (!success) {
                continue;
            }

            const score = this.minimax(depth - 1, true, alpha, beta, aiPlayer);
            minScore = Math.min(minScore, score);

            this.game.undo();

            beta = Math.min(beta, score);
            if (beta <= alpha) break;
        }
        return minScore;
    }

    orderMoves(moves, player, depth, opponent, opponentThreatBefore = 0, friendlyMetrics = null, friendlyConnectorTargets = null, opponentConnectorTargets = null, opponentMetrics = null, opponentFrontier = null, opponentConnectors = null, opponentTrailing = null) {
        if (!moves || moves.length === 0) return [];

        const difficulty = this.game.aiDifficulty || 'medium';
        const friendlyPegs = this.game.pegs.filter(p => p.player === player);
        const opponentPegs = this.game.pegs.filter(p => p.player === opponent);

        const boardSize = this.game.boardSize;
        const spanValue = opponentMetrics
            ? (opponent === 'red' ? opponentMetrics.maxRowSpan : opponentMetrics.maxColSpan)
            : 0;
        const largestLength = opponentMetrics?.largestComponent?.length || 0;
        const opponentUrgent = spanValue >= Math.max(6, Math.floor(boardSize / 4)) || largestLength >= 6;

        const scored = moves.map(move => ({
            move,
            score: this.movePriority(
                move,
                player,
                friendlyPegs,
                opponentPegs,
                opponent,
                opponentThreatBefore,
                friendlyMetrics,
                friendlyConnectorTargets,
                opponentConnectorTargets,
                opponentMetrics,
                opponentFrontier,
                opponentConnectors,
                opponentTrailing,
                opponentUrgent
            )
        }));

        scored.sort((a, b) => b.score - a.score);

        const baseLimit = difficulty === 'easy' ? 14 : difficulty === 'medium' ? 20 : 26;
        const effectiveDepth = Math.max(1, this.rootDepth || depth || 1);
        const depthFactor = Math.max(1, depth + 1);
        const limit = Math.max(6, Math.min(moves.length, Math.round(baseLimit * depthFactor / (effectiveDepth + 1))));

        return scored.slice(0, limit).map(entry => entry.move);
    }

    movePriority(move, player, friendlyPegs, opponentPegs, opponent, opponentThreatBefore, friendlyMetrics, friendlyConnectorTargets, opponentConnectorTargets, opponentMetrics, opponentFrontier, opponentConnectors, opponentTrailing, opponentUrgent) {
        if (this.debugEnabled) {
            this.currentHeuristic = {};
        } else {
            this.currentHeuristic = null;
        }

        maybeLoadValueModel();

        const featureTotals = Object.create(null);
        const capture = (key, value) => {
            if (!Number.isFinite(value)) {
                return;
            }
            featureTotals[key] = (featureTotals[key] || 0) + value;
            if (this.debugEnabled) {
                if (!this.currentHeuristic) {
                    this.currentHeuristic = {};
                }
                this.currentHeuristic[key] = (this.currentHeuristic[key] || 0) + value;
            }
        };

        const board = this.game.board;
        const boardSize = this.game.boardSize;
        const friendlyConnectorSet = (friendlyConnectorTargets && typeof friendlyConnectorTargets.has === 'function') ? friendlyConnectorTargets : null;
        const opponentConnectorSet = (opponentConnectorTargets && typeof opponentConnectorTargets.has === 'function') ? opponentConnectorTargets : null;
        const moveKey = `${move.row}:${move.col}`;
        let blockedOpponentConnector = false;

        let friendlyConnections = 0;
        let opponentConnections = 0;
        for (const [dr, dc] of KNIGHT_OFFSETS) {
            const r = move.row + dr;
            const c = move.col + dc;
            if (r < 0 || r >= boardSize || c < 0 || c >= boardSize) continue;
            if (board[r][c] === player) {
                friendlyConnections++;
            } else if (board[r][c] === opponent) {
                opponentConnections++;
            }
        }

        let score = 0;

        if (friendlyConnectorSet && friendlyConnectorSet.has(moveKey)) {
            capture('edgeConnectorTarget', REWARDS.edge.offense.connectorTargetBonus);
            score += REWARDS.edge.offense.connectorTargetBonus;
        }

        if (opponentConnectorSet && opponentConnectorSet.has(moveKey)) {
            capture('edgeDefenseBlock', REWARDS.edge.defense.blockBonus);
            score += REWARDS.edge.defense.blockBonus;
            blockedOpponentConnector = true;
        }

        const friendlyConnectionScore = friendlyConnections * REWARDS.general.friendlyConnection;
        if (friendlyConnectionScore !== 0) {
            capture('friendlyConnections', friendlyConnectionScore);
            score += friendlyConnectionScore;
        }

        const opponentConnectionScore = opponentConnections * REWARDS.general.opponentConnection;
        if (opponentConnectionScore !== 0) {
            capture('opponentConnections', opponentConnectionScore);
            score += opponentConnectionScore;
        }

        const friendlyDist = this.minDistanceToPeg(move, friendlyPegs);
        if (Number.isFinite(friendlyDist)) {
            const friendlyDistanceBonus = Math.max(0, 10 - friendlyDist) * REWARDS.general.friendlyDistance;
            if (friendlyDistanceBonus !== 0) {
                capture('friendlyDistance', friendlyDistanceBonus);
                score += friendlyDistanceBonus;
            }
        }

        const opponentDist = this.minDistanceToPeg(move, opponentPegs);
        if (Number.isFinite(opponentDist)) {
            const opponentDistanceBonus = Math.max(0, 10 - opponentDist) * REWARDS.general.opponentDistance;
            if (opponentDistanceBonus !== 0) {
                capture('opponentDistance', opponentDistanceBonus);
                score += opponentDistanceBonus;
            }
        }

        const goalDistance = player === 'red'
            ? Math.min(move.row, boardSize - 1 - move.row)
            : Math.min(move.col, boardSize - 1 - move.col);
        const goalBonus = Math.max(0, 12 - goalDistance) * REWARDS.general.goalDistance;
        if (goalBonus !== 0) {
            capture('goalDistance', goalBonus);
            score += goalBonus;
        }

        const center = (boardSize - 1) / 2;
        const centerDist = Math.abs(move.row - center) + Math.abs(move.col - center);
        const centerBias = Math.max(0, 16 - centerDist) * REWARDS.general.centerBias;
        if (centerBias !== 0) {
            capture('centerBias', centerBias);
            score += centerBias;
        }

        if (!Number.isFinite(friendlyDist) && !Number.isFinite(opponentDist)) {
            const isolatedBonus = REWARDS.general.isolated;
            capture('isolatedBonus', isolatedBonus);
            score += isolatedBonus;
        }

        if (opponentMetrics && Array.isArray(opponentMetrics.largestComponent) && opponentMetrics.largestComponent.length > 0) {
            const distToChain = this.distanceToComponent(move, opponentMetrics.largestComponent);
            const bonus = Math.max(0, 12 - distToChain) * (opponentUrgent ? 30 : 15);
            if (bonus !== 0) {
                capture('chainProximity', bonus);
                score += bonus;
            }
        }

        if (opponentFrontier && opponentFrontier.length > 0) {
            const distToFrontier = this.distanceToSet(move, opponentFrontier);
            const proximityBonus = Math.max(0, 10 - distToFrontier) * (opponentUrgent ? 35 : 16);
            if (proximityBonus !== 0) {
                capture('frontierProximity', proximityBonus);
                score += proximityBonus;
            }
            if (distToFrontier === 0) {
                const overlapBonus = opponentUrgent ? 550 : 220;
                capture('frontierCapture', overlapBonus);
                score += overlapBonus;
            }
        }

        if (opponentConnectors && opponentConnectors.length > 0) {
            const distToConnector = this.distanceToSet(move, opponentConnectors);
            const proximityBonus = Math.max(0, 8 - distToConnector) * (opponentUrgent ? 55 : 30);
            if (proximityBonus !== 0) {
                capture('connectorProximity', proximityBonus);
                score += proximityBonus;
            }
            if (distToConnector === 0) {
                const connectorBonus = opponentUrgent ? 700 : 320;
                capture('connectorCapture', connectorBonus);
                score += connectorBonus;
            }
        }

        if (opponentTrailing && opponentTrailing.length > 0) {
            const distTrailing = this.distanceToSet(move, opponentTrailing);
            const penalty = Math.max(0, 6 - distTrailing) * 6;
            if (penalty !== 0) {
                capture('trailingPenalty', -penalty);
                score -= penalty;
            }
        }

        const originalPlayer = this.game.currentPlayer;
        if (opponentThreatBefore > 0 || friendlyMetrics || opponentMetrics) {
          this.game.currentPlayer = player;
          const success = this.game.placePeg(move.row, move.col);
          if (success) {
            // Threat reduction (existing)
          if (opponentThreatBefore > 0 && this.game.moveCount > 1) {
            const threatAfter = connectivityScore(this.game, opponent);
            const threatReduction = opponentThreatBefore - threatAfter;
            if (threatReduction > 0) {
              const bonus = threatReduction * 140;
              if (typeof capture === 'function') capture('threatReduction', bonus);
                score += bonus;
              } else {
                const penalty = opponentUrgent ? 600 : 250;
                if (typeof capture === 'function') capture('noThreatReduction', -penalty);
                score -= penalty;
              }
            }

            if (friendlyMetrics) {
              // Metrics AFTER placing the peg
              const postMetrics = componentMetrics(this.game, player);
              const boardLimit = this.game.boardSize - 1;
              const postLargest = postMetrics.largestComponent || [];
              let postMinR = Infinity, postMaxR = -Infinity, postMinC = Infinity, postMaxC = -Infinity;
              if (postLargest.length > 0) {
                for (const p of postLargest) {
                  if (p.row < postMinR) postMinR = p.row;
                  if (p.row > postMaxR) postMaxR = p.row;
                  if (p.col < postMinC) postMinC = p.col;
                  if (p.col > postMaxC) postMaxC = p.col;
                }
              }
              const friendlyLargest = (friendlyMetrics.largestComponent || []);
              let friendlyMinR = Infinity, friendlyMaxR = -Infinity, friendlyMinC = Infinity, friendlyMaxC = -Infinity;
              if (friendlyLargest.length > 0) {
                for (const p of friendlyLargest) {
                  if (p.row < friendlyMinR) friendlyMinR = p.row;
                  if (p.row > friendlyMaxR) friendlyMaxR = p.row;
                  if (p.col < friendlyMinC) friendlyMinC = p.col;
                  if (p.col > friendlyMaxC) friendlyMaxC = p.col;
                }
              }

              const goalSpanBefore = player === 'red' ? friendlyMetrics.maxRowSpan : friendlyMetrics.maxColSpan;
              const goalSpanAfter  = player === 'red' ? postMetrics.maxRowSpan    : postMetrics.maxColSpan;
              const spanGain = goalSpanAfter - goalSpanBefore;

              const prevMinAxis = player === 'red'
                ? (Number.isFinite(friendlyMinR) ? friendlyMinR : (friendlyMetrics.minRow ?? boardLimit))
                : (Number.isFinite(friendlyMinC) ? friendlyMinC : (friendlyMetrics.minCol ?? boardLimit));
              const prevMaxAxis = player === 'red'
                ? (Number.isFinite(friendlyMaxR) ? friendlyMaxR : (friendlyMetrics.maxRow ?? 0))
                : (Number.isFinite(friendlyMaxC) ? friendlyMaxC : (friendlyMetrics.maxCol ?? 0));
              const postMinAxis = player === 'red'
                ? (Number.isFinite(postMinR) ? postMinR : (postMetrics.minRow ?? boardLimit))
                : (Number.isFinite(postMinC) ? postMinC : (postMetrics.minCol ?? boardLimit));
              const postMaxAxis = player === 'red'
                ? (Number.isFinite(postMaxR) ? postMaxR : (postMetrics.maxRow ?? 0))
                : (Number.isFinite(postMaxC) ? postMaxC : (postMetrics.maxCol ?? 0));

              const prevGapFront = Math.max(0, prevMinAxis);
              const prevGapBack  = Math.max(0, boardLimit - prevMaxAxis);
              const postGapFront = Math.max(0, postMinAxis);
              const postGapBack  = Math.max(0, boardLimit - postMaxAxis);
              const gapBefore = prevGapFront + prevGapBack;
              const gapAfter  = postGapFront + postGapBack;
              const gapImprovement = gapBefore - gapAfter;

              const touchesBothPost = player === 'red'
                ? (postMetrics.touchesTop && postMetrics.touchesBottom)
                : (postMetrics.touchesLeft && postMetrics.touchesRight);
              const nearFinish = gapAfter <= REWARDS.edge.offense.finishThreshold;
              let finishLaneOpen = true;

              if (nearFinish && !touchesBothPost) {
                const bounds = {
                  minR: Number.isFinite(postMinR) ? postMinR : null,
                  maxR: Number.isFinite(postMaxR) ? postMaxR : null,
                  minC: Number.isFinite(postMinC) ? postMinC : null,
                  maxC: Number.isFinite(postMaxC) ? postMaxC : null
                };
                const cacheKey = this.getSealedLaneCacheKey(player, postMetrics, bounds);
                if (cacheKey && this.sealedLaneCache && this.sealedLaneCache.has(cacheKey)) {
                  finishLaneOpen = this.sealedLaneCache.get(cacheKey);
                } else {
                  finishLaneOpen = hasReachableGoalEdge(this.game, player, postMetrics);
                  if (cacheKey && this.sealedLaneCache) {
                    this.sealedLaneCache.set(cacheKey, finishLaneOpen);
                  }
                }
              }

              if (finishLaneOpen) {
                // ---- A) First-time edge touch bonus (newly touching your own goal edge) ----
                if (player === 'black') {
                  const newLeft  = postMetrics.touchesLeft  && !friendlyMetrics.touchesLeft;
                  const newRight = postMetrics.touchesRight && !friendlyMetrics.touchesRight;
                  if (newLeft || newRight) {
                    const touchBonus = REWARDS.edge.offense.firstEdgeTouchBlack;
                    if (touchBonus !== 0) {
                      if (typeof capture === 'function') capture('firstEdgeTouch', touchBonus);
                      score += touchBonus;
                      this.recordStat('firstEdgeTouch', player, touchBonus);
                    }
                  }
                } else { // red
                  const newTop    = postMetrics.touchesTop    && !friendlyMetrics.touchesTop;
                  const newBottom = postMetrics.touchesBottom && !friendlyMetrics.touchesBottom;
                  if (newTop || newBottom) {
                    const touchBonus = REWARDS.edge.offense.firstEdgeTouchRed;
                    if (touchBonus !== 0) {
                      if (typeof capture === 'function') capture('firstEdgeTouch', touchBonus);
                      score += touchBonus;
                      this.recordStat('firstEdgeTouch', player, touchBonus);
                    }
                  }
                }

                // ---- B) Double-edge coverage upgrade (touch both goal edges for the first time) ----
                if (player === 'black') {
                  const hadBoth = friendlyMetrics.touchesLeft && friendlyMetrics.touchesRight;
                  const hasBoth = postMetrics.touchesLeft && postMetrics.touchesRight;
                  const componentSpansBoth =
                    postLargest.length > 0 &&
                    postMinC <= 0 &&
                    postMaxC >= boardLimit;
                  if (hasBoth && !hadBoth && componentSpansBoth) {
                    const coverageBonus = REWARDS.edge.offense.doubleCoverageBase * REWARDS.edge.offense.blackDoubleCoverageScale;
                    if (typeof capture === 'function') capture('doubleEdgeCoverage', coverageBonus);
                    score += coverageBonus;
                    this.recordStat('doubleEdgeCoverage', player, coverageBonus);
                  }
                } else {
                  const hadBoth = friendlyMetrics.touchesTop && friendlyMetrics.touchesBottom;
                  const hasBoth = postMetrics.touchesTop && postMetrics.touchesBottom;
                  const componentSpansBoth =
                    postLargest.length > 0 &&
                    postMinR <= 0 &&
                    postMaxR >= boardLimit;
                  if (hasBoth && !hadBoth && componentSpansBoth) {
                    const coverageBonus = REWARDS.edge.offense.doubleCoverageBase + REWARDS.edge.offense.redDoubleCoverageBonus;
                    if (typeof capture === 'function') capture('doubleEdgeCoverage', coverageBonus);
                    score += coverageBonus;
                    this.recordStat('doubleEdgeCoverage', player, coverageBonus);
                  }
                }

                if (spanGain > 0) {
                  let multiplier = REWARDS.edge.offense.spanGainBase * (player === 'black' ? REWARDS.edge.offense.blackSpanGainMultiplier : 1);
                  if (player === 'red' && (postMetrics.touchesTop || postMetrics.touchesBottom)) {
                    multiplier *= REWARDS.edge.offense.redSpanGainMultiplier;
                  }
                  const bonus = spanGain * multiplier;
                  if (typeof capture === 'function') capture('spanGain', bonus);
                  score += bonus;
                  this.recordStat('spanGain', player, bonus);
                }

                if (gapImprovement > 0) {
                  const gapMultiplier = REWARDS.edge.offense.gapDecay * (player === 'red' ? REWARDS.edge.offense.redGapDecayMultiplier : 1);
                  const gapBonus = gapImprovement * gapMultiplier;
                  if (typeof capture === 'function') capture('edgeGapReduction', gapBonus);
                  score += gapBonus;
                  this.recordStat('edgeGapReduction', player, gapBonus);
                }

                if (postLargest.length > 0) {
                  const lcTouchesTop    = (postMinR <= 1);
                  const lcTouchesBottom = (postMaxR >= boardLimit - 1);
                  const lcTouchesLeft   = (postMinC <= 1);
                  const lcTouchesRight  = (postMaxC >= boardLimit - 1);

                  const redSpans   = lcTouchesTop && lcTouchesBottom;
                  const blackSpans = lcTouchesLeft && lcTouchesRight;

                  if ((player === 'red' && redSpans) || (player === 'black' && blackSpans)) {
                    const spanCompleteBonus = REWARDS.edge.offense.finishBonusBase * 2 * (player === 'black' ? REWARDS.edge.offense.blackFinishScaleMultiplier : 1);
                    if (typeof capture === 'function') capture('largestComponentSpanComplete', spanCompleteBonus);
                    score += spanCompleteBonus;
                    this.recordStat('largestComponentSpanComplete', player, spanCompleteBonus);
                  }
                }

                if (touchesBothPost || nearFinish) {
                  const progressMade = spanGain > 0 || gapImprovement > 0;
                  const finishScaleBase = Math.max(0, REWARDS.edge.offense.finishBonusBase - gapAfter * REWARDS.edge.offense.finishGapSlope);
                  if (progressMade) {
                    let bonusBase = REWARDS.edge.offense.connectorBonus + finishScaleBase;
                    if (this.rootDepth >= 3 && player === 'red') {
                      bonusBase += REWARDS.general.redDepth3Bonus ?? 0;
                      if (REWARDS.general.redDepth3Bonus) {
                        this.recordStat('redDepth3BonusApplied', player, REWARDS.general.redDepth3Bonus);
                      }
                    }
                    if (player === 'black') {
                      bonusBase *= REWARDS.edge.offense.blackFinishScaleMultiplier;
                    }
                    if (player === 'red') {
                      bonusBase += REWARDS.edge.offense.redFinishExtra;
                    }
                    if (typeof capture === 'function') capture('edgeFinishAdvance', bonusBase);
                    score += bonusBase;
                    this.recordStat('edgeFinishAdvance', player, bonusBase);
                  } else {
                    const penaltyBase = REWARDS.edge.offense.finishPenaltyBase + gapAfter * REWARDS.edge.offense.finishGapSlope;
                    const penalty = penaltyBase * (player === 'red' ? REWARDS.edge.offense.redFinishPenaltyFactor : 1);
                    if (typeof capture === 'function') capture('edgeFinishStall', -penalty);
                    score -= penalty;
                    this.recordStat('edgeFinishStall', player, -penalty);
                  }
                }
              } else if (typeof capture === 'function') {
                capture('finishLaneSealed', 0);
                this.recordStat('finishLaneSealed', player, 1);
              }

              if (opponentConnectorSet && opponentConnectorSet.size > 0 && !blockedOpponentConnector && !touchesBothPost) {
                const defensePenalty = REWARDS.edge.defense.missPenalty * (opponentUrgent ? 1.5 : 1);
                if (this.game.moveCount > 1) {
                  if (typeof capture === 'function') capture('edgeDefenseMiss', -defensePenalty);
                  score -= defensePenalty;
                }
              }

              // (Removed old +400 span-complete blocks)
            }

            // Opponent-side effects (existing)
            if (opponentMetrics) {
              const opponentPost = componentMetrics(this.game, opponent);
              const opponentSpanBefore = opponent === 'red' ? opponentMetrics.maxRowSpan : opponentMetrics.maxColSpan;
              const opponentSpanAfter  = opponent === 'red' ? opponentPost.maxRowSpan    : opponentPost.maxColSpan;
              const spanReduction = opponentSpanBefore - opponentSpanAfter;
              if (spanReduction > 0) {
                const bonus = spanReduction * 120;
                if (typeof capture === 'function') capture('opponentSpanReduction', bonus);
                score += bonus;
              }

              if (opponentUrgent && spanReduction <= 0) {
                if (typeof capture === 'function') capture('noSpanReductionPenalty', -400);
                score -= 400;
              }

              if (opponent === 'black' && opponentPost.touchesLeft && opponentPost.touchesRight &&
                  !(opponentMetrics.touchesLeft && opponentMetrics.touchesRight)) {
                if (typeof capture === 'function') capture('blackSpanUpgradePenalty', -500);
                score -= 500;
              }
              if (opponent === 'red' && opponentPost.touchesTop && opponentPost.touchesBottom &&
                  !(opponentMetrics.touchesTop && opponentMetrics.touchesBottom)) {
                if (typeof capture === 'function') capture('redSpanUpgradePenalty', -500);
                score -= 500;
              }
            }

            this.game.undo();
          }
        }

        this.game.currentPlayer = originalPlayer;

        if (opponentMetrics && opponent === 'red') {
            if (opponentMetrics.touchesBottom && !opponentMetrics.touchesTop) {
                const topBias = Math.max(0, boardSize - move.row) * 12;
                if (topBias !== 0) {
                    capture('topBias', topBias);
                    score += topBias;
                }
                if (opponentMetrics.minRow !== null) {
                    const bonusExtra = Math.max(0, opponentMetrics.minRow - move.row) * 150;
                    if (bonusExtra !== 0) {
                        capture('aboveMinRowBonus', bonusExtra);
                        score += bonusExtra;
                    }
                    const penaltyValue = Math.max(0, move.row - opponentMetrics.minRow) * 90;
                    if (penaltyValue !== 0) {
                        capture('belowMinRowPenalty', -penaltyValue);
                        score -= penaltyValue;
                    }
                }
            } else if (opponentMetrics.touchesTop && !opponentMetrics.touchesBottom) {
                const bottomBias = Math.max(0, move.row) * 12;
                if (bottomBias !== 0) {
                    capture('bottomBias', bottomBias);
                    score += bottomBias;
                }
                if (opponentMetrics.maxRow !== null) {
                    const bonusExtra = Math.max(0, move.row - opponentMetrics.maxRow) * 150;
                    if (bonusExtra !== 0) {
                        capture('belowMaxRowBonus', bonusExtra);
                        score += bonusExtra;
                    }
                    const penaltyValue = Math.max(0, opponentMetrics.maxRow - move.row) * 90;
                    if (penaltyValue !== 0) {
                        capture('aboveMaxRowPenalty', -penaltyValue);
                        score -= penaltyValue;
                    }
                }
            }
        }

        const featureContext = {
            turn: this.game.moveCount,
            player,
            playerPegCount: friendlyPegs.length + 1,
            opponentPegCount: opponentPegs.length
        };

        const evaluation = evaluateValueModel(featureTotals, featureContext);
        let valueAdjustment = null;
        if (evaluation) {
            valueAdjustment = (evaluation.probability - 0.5) * VALUE_MODEL_SCALE;
            score += valueAdjustment;
        }

        if (player === 'red' && REWARDS.general.redBaseBonus) {
            const bonus = REWARDS.general.redBaseBonus;
            if (typeof bonus === 'number' && bonus !== 0) {
                capture('redBaseBonus', bonus);
                score += bonus;
            }
        }

        if (player === 'black' && REWARDS.general.blackBasePenalty) {
            const penalty = REWARDS.general.blackBasePenalty;
            if (typeof penalty === 'number' && penalty !== 0) {
                capture('blackBasePenalty', -penalty);
                score -= penalty;
            }
        }

        if (player === 'red' && REWARDS.general.redGlobalMultiplier !== 1) {
            const delta = score * (REWARDS.general.redGlobalMultiplier - 1);
            if (delta !== 0) {
                capture('redGlobalMultiplier', delta);
                score += delta;
            }
        }

        if (player === 'black' && REWARDS.general.blackGlobalScale !== 1) {
            const delta = score * (REWARDS.general.blackGlobalScale - 1);
            if (delta !== 0) {
                capture('blackGlobalScale', delta);
                score += delta;
            }
        }

        const lateStart = REWARDS.general.lateGameStart;
        const latePressure = REWARDS.general.lateGamePressure;
        if (Number.isFinite(lateStart) && Number.isFinite(latePressure) && latePressure > 0) {
            const lateTurns = (this.game.moveCount + 1) - lateStart;
            if (lateTurns > 0) {
                const penalty = lateTurns * latePressure;
                if (penalty !== 0) {
                    capture('lateGamePressure', -penalty);
                    score -= penalty;
                }
            }
        }

        const featureSnapshot = { ...featureTotals };
        this.lastHeuristicFeatures = featureSnapshot;
        this.lastFeatureContext = featureContext;
        this.lastValueModelProbability = evaluation ? evaluation.probability : null;
        this.lastValueModelLogit = evaluation ? evaluation.logit : null;
        this.lastValueModelAdjustment = evaluation ? valueAdjustment : null;

        if (this.debugEnabled) {
            const breakdown = { ...featureSnapshot };
            breakdown.valueModelProbability = this.lastValueModelProbability;
            breakdown.valueModelAdjustment = this.lastValueModelAdjustment;
            breakdown.valueModelLogit = this.lastValueModelLogit;
            this.lastHeuristicBreakdown = breakdown;
        } else {
            this.lastHeuristicBreakdown = null;
        }

        return score;
    }

    minDistanceToPeg(move, pegs) {
        if (!pegs || pegs.length === 0) return Infinity;
        let best = Infinity;
        for (const peg of pegs) {
            const dist = Math.abs(peg.row - move.row) + Math.abs(peg.col - move.col);
            if (dist < best) {
                best = dist;
            }
        }
        return best;
    }

    distanceToComponent(move, component) {
        let best = Infinity;
        for (const peg of component) {
            const dist = Math.abs(peg.row - move.row) + Math.abs(peg.col - move.col);
            if (dist < best) {
                best = dist;
            }
        }
        return best;
    }

    distanceToSet(move, cells) {
        let best = Infinity;
        for (const cell of cells) {
            const dist = Math.abs(cell.row - move.row) + Math.abs(cell.col - move.col);
            if (dist < best) {
                best = dist;
            }
        }
        return best;
    }

    moveKey(move) {
        return `${move.row}:${move.col}`;
    }

    getSealedLaneCacheKey(player, postMetrics, bounds) {
        if (!postMetrics || !bounds) return null;
        const {
            touchesTop = false,
            touchesBottom = false,
            touchesLeft = false,
            touchesRight = false,
            largestComponent = []
        } = postMetrics;
        const { minR, maxR, minC, maxC } = bounds;
        return [
            player,
            minR ?? 'n',
            maxR ?? 'n',
            minC ?? 'n',
            maxC ?? 'n',
            touchesTop ? 1 : 0,
            touchesBottom ? 1 : 0,
            touchesLeft ? 1 : 0,
            touchesRight ? 1 : 0,
            largestComponent.length || 0
        ].join('|');
    }

    debugSnapshot() {
        return this.game.board.map(row => [...row]);
    }

    downloadTrace() {
        if (!this.moveTrace.length) {

            return;
        }

        const blob = new Blob([JSON.stringify(this.moveTrace, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `twixt-ai-log-${Date.now()}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}
