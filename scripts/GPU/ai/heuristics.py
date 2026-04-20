"""TwixT heuristic evaluation functions with configurable knobs.

Ported from assets/js/ai/heuristics.js with tunable parameters.

Key functions:
- find_connected_components: Find all bridge-connected components for a player
- component_metrics: Compute span/touch metrics for components
- evaluate_position: Score a board position for a player
- evaluate_move: Score a candidate move
- extract_features: Get feature vector for ML model
- score_moves: Batch score all candidate moves
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple, Union, overload

from ..game.state import GameState, Pos, Component, Components
from ..game.board import is_valid_placement
from ..game.bridge import KNIGHT_OFFSETS, bridges_cross

# Import sealed lane detection (lazy import to avoid circular deps)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .sealed_lane import SealedLaneLRU


# =============================================================================
# HARDCODED CONSTANTS - These match JS heuristics.js exactly
# These are NOT tunable knobs - they are fixed multipliers used in evaluation
# =============================================================================

# evaluatePosition() multipliers (JS heuristics.js lines ~5-15)
CONNECTED_PATHS_MULTIPLIER = 100      # JS: evaluateConnectedPaths * 100
POTENTIAL_CONNECTIONS_MULTIPLIER = 20  # JS: evaluatePotentialConnections * 20
EDGE_PROGRESS_MULTIPLIER = 30          # JS: evaluateEdgeProgress * 30
PEG_DIFFERENCE_MULTIPLIER = 2          # JS: (playerPegs - opponentPegs) * 2

# Gap scoring in evaluatePosition() (JS heuristics.js lines ~55-65)
GAP_PULL_BASE = 200       # JS: 200 * urgency * (1/(1+gap))
GAP_PENALTY_BASE = 40     # JS: 40 * (gapTop + gapBottom)
URGENCY_MULTIPLIER = 2.5  # JS: urgency = 2.5 when touching one edge
DRIFT_PENALTY = 0.05      # JS: 0.05 * moveCount

# evaluateMove() values (JS heuristics.js lines ~70-120)
CONNECTION_BASE_BONUS = 100   # JS: 100 + distance * 5
CONNECTION_DISTANCE_MULT = 5  # JS: distance * 5
SPAN_BOARD_BONUS = 300        # JS: if spansBoard, score += 300
SPAN_WIDE_BONUS = 150         # JS: else if wideSpan, score += 150
MULTI_CONNECTION_MULT = 75    # JS: connectionCount * 75
GOAL_DISTANCE_MAX = 12        # JS: max(0, 12 - distanceToNearestGoal)
GOAL_DISTANCE_MULT = 8        # JS: * 8
OPPONENT_THREAT_MULT = 25     # JS: opponentThreats * 25
CENTER_BIAS_MAX_DIST = 24     # JS: max(0, 24 - centerDistance)
CENTER_BIAS_MULT = 2          # JS: * 2 (NOT * centerBias knob!)

# evaluateEdgeProgress() - JS is SIMPLE: just max(0, 12-dist) per peg
EDGE_DISTANCE_MAX = 12        # JS: Math.max(0, 12 - distanceToGoal)

# scoreComponent() values (JS heuristics.js lines 297-321)
COMPONENT_SIZE_MULT = 10      # JS: component.length * 10
SPAN_MULT = 20                # JS: span * 20
FULL_SPAN_BONUS = 500         # JS: +500 for full span (0↔boardSize-1)

# evaluatePotentialConnections() values (JS heuristics.js lines 323-344)
POTENTIAL_MOVE_BONUS = 5      # JS: score += 5 for each valid goal-direction move

# evaluateWinningThreats() values (JS heuristics.js lines 219-242)
THREAT_FULL_SPAN = 800        # JS: minRow === 0 && maxRow === 23
THREAT_NEAR_SPAN = 400        # JS: minRow <= 1 && maxRow >= 22
THREAT_MEDIUM_SPAN = 200      # JS: minRow <= 3 && maxRow >= 20

# evaluateConnectedPaths() values (JS heuristics.js lines 195-217)
AVG_COMPONENT_SIZE_MULT = 20  # JS: avgComponentSize * 20
COMPONENT_PENALTY_MULT = 30   # JS: (components.length - 3) * 30

# =============================================================================
# movePriority() HARDCODED CONSTANTS - These are in search.js, NOT search.json
# These cannot be tuned via search.json - they are fixed in the code
# =============================================================================

# Threat reduction scoring (JS search.js lines 1151-1164)
THREAT_REDUCTION_MULT = 140       # JS: threatReduction * 140
NO_THREAT_URGENT = 600            # JS: penalty when urgent and no threat reduction
NO_THREAT_NORMAL = 250            # JS: penalty when not urgent and no threat reduction

# Opponent span reduction scoring (JS search.js lines 1496-1508)
SPAN_REDUCTION_MULT = 120         # JS: spanReduction * 120
NO_SPAN_REDUCTION_PENALTY = 400   # JS: penalty when opponent urgent but span not reduced

# Span upgrade penalty (JS search.js lines 1510-1529)
SPAN_UPGRADE_PENALTY = 500        # JS: penalty if opponent newly spans both edges

# Defensive position biases (JS search.js lines 1542-1583)
DEFENSIVE_BIAS_MULT = 12          # JS: topBias/bottomBias * 12
DEFENSIVE_POSITION_BONUS = 150    # JS: aboveMinRowBonus/belowMaxRowBonus * 150
DEFENSIVE_POSITION_PENALTY = 90   # JS: belowMinRowPenalty/aboveMaxRowPenalty * 90


# =============================================================================
# TUNABLE KNOBS - These come from search.json and can be adjusted by auto-tuner
# =============================================================================

DEFAULT_KNOBS: Dict[str, float] = {
    # Edge touch bonuses (first peg on goal edge) - from search.json
    "firstEdgeRed": 420.0,        # search.json: firstEdgeTouchRed
    "firstEdgeBlack": 455.0,      # search.json: firstEdgeTouchBlack

    # Finish bonuses - from search.json edge.offense
    "finishBonusBase": 3332.0,
    "finishBonusScale": 1.0,
    "nearFinishBonus": 2500.0,
    "finishGapSlope": 150.0,      # search.json value (hardcoded uses GAP_PULL_BASE=200)
    "finishThreshold": 4.0,
    "finishPenaltyBase": 1181.0,

    # Red/Black asymmetry multipliers - from search.json
    "redFinishPenaltyFactor": 0.55,  # search.json: 0.55
    "blackFinishScaleMultiplier": 1.0,
    "redSpanGainMultiplier": 1.0,    # search.json: 1
    "blackSpanGainMultiplier": 1.0,
    "redDoubleCoverageBonus": 1000.0,  # search.json: 1000
    "blackDoubleCoverageScale": 0.8,   # search.json: 0.8

    # Span and coverage - from search.json
    "spanGainBase": 180.0,
    "doubleCoverageBase": 2400.0,

    # Connector and gap - from search.json
    "connectorBonus": 608.0,
    "connectorBonusScale": 1.0,
    "connectorTargetBonus": 500.0,  # search.json: connectorTargetBonus
    "gapDecay": 23.0,
    "gapDecayScale": 1.0,

    # Connection scoring - from search.json general (used differently in JS)
    "friendlyConnection": 12.0,
    "opponentConnection": 35.0,
    "friendlyDistance": 3.0,
    "opponentDistance": 12.0,
    "goalDistance": 1.2,

    # Defense - from search.json
    "blockBonus": 900.0,
    "missPenalty": 350.0,
    "edgeRadius": 3,  # search.json: rewards.edge.radius (used for connector targets)

    # Other - from search.json
    "centerBias": 0.5,
    "isolated": 10.0,
    "lateGameStart": 60.0,
    "lateGamePressure": 0.0,

    # Red/Black global adjustments - from search.json general
    "redGlobalMultiplier": 1.0,
    "blackGlobalScale": 1.0,
    "redBaseBonus": 0.0,
    "blackBasePenalty": 0.0,

    # Additional finish knobs - from search.json edge.offense
    "redFinishExtra": 0.0,
    "redGapDecayMultiplier": 1.0,

    # Value model
    "valueModelScale": 600.0,

    # Temperature schedule for stochastic play (used by engine.py / search.py)
    "temp_early": 0.9,              # Temperature for early game (high exploration)
    "temp_late": 0.12,              # Temperature for late game (exploitation)
    "temp_transition_ply": 10,      # Ply at which decay begins
    "temp_tau": 10.0,               # Decay time constant (larger = slower decay)
    "deterministic_mode": 0,        # 1 = force T=0 everywhere

    # Training-only knobs (Python self-play)
    "opening_random_plies": 8,      # Randomize early plies for opening diversity
    "opening_random_top_k": 0,      # 0 = uniform random over legal moves
    "opening_random_plies_d2": 2,   # More opening diversity at depth >= 2
    "opening_random_top_k_d2": 6,   # Sample from top-K heuristic moves
    "opening_uniform_plies_d2": 0,  # Fully random opening plies at depth >= 2
    "opening_uniform_margin_d2": 3, # Avoid edge rows/cols for uniform openings
    "opening_forbid_edge_plies_d2": 2,  # Forbid edge zones for early plies
    "opening_forbid_edge_margin_d2": 3, # Edge margin to forbid in opening
    "opening_quadrant_plies_d2": 0,     # Bias opening toward a random quadrant
    "opening_quadrant_margin_d2": 2,    # Margin for quadrant bias
    "enforce_symmetry": 1,          # 1 = neutralize color-specific biases
    "training_max_moves": 50,       # Cap game length to boost throughput
    "training_stall_limit": 15,     # Earlier stall to reduce long draws
    "training_max_moves_d2": 80,    # Depth-2 cap (reduce stalls vs d1)
    "training_stall_limit_d2": 30,  # Depth-2 stall limit
    "training_span_win_margin": 1, # Span margin to decide winner on stall/max moves
    "training_black_span_multiplier": 1.1,  # Boost black span gain in training
    "training_goal_distance_mult": 1.1,     # Push moves toward goal edges
    "training_center_bias_mult": 0.8,       # Keep some center preference
    "training_span_gain_mult": 1.3,         # Reward span growth
    "training_connector_bonus_scale": 1.2,  # Reward bridge/connectivity
    "training_finish_bonus_scale": 1.2,     # Increase finish pressure
    "training_adjacent_penalty": 60.0,      # Penalize adjacent (non-bridge) clumping
    "training_edge_push_weight": 25.0,      # Push toward missing goal edges
    "training_edge_push_max": 12.0,         # Max distance considered for edge push
    "training_missing_edge_half_penalty": 80.0,   # Penalize staying on wrong half
    "training_second_edge_push_scale": 1.2,       # Extra push after first edge is touched
    "training_global_span_gain_weight": 300.0,   # Reward overall span growth (all pegs)
    "training_first_edge_bonus_scale": 1.4,      # Amplify first edge touch bonus
    "training_defense_scale": 1.3,               # Boost opponent blocking/pressure
    "training_edge_push_decay_after_both": 1,    # Disable edge push after both edges touched
    "training_opening_center_plies": 8,          # Opening phase to favor center
    "training_opening_center_bias_mult": 2.0,    # Extra center pull during opening
    "training_opening_goal_distance_scale": 0.5, # Reduce goal-edge pull during opening
    "training_opening_edge_push_scale": 0.2,     # Suppress edge-push early
    "training_opening_missing_half_scale": 0.2,  # Suppress wrong-half penalty early
    "training_edge_push_ramp_plies": 6,          # Ramp edge-push after opening
    "training_second_edge_bonus": 400.0,         # Bonus for touching second edge
    "training_edge_touch_requires_bridge": 1,    # Require bridge/connection for edge bonus
    "training_goal_isolation_scale": 0.4,        # Reduce goal bonus on isolated moves
    "training_edge_touch_min_component": 3,      # Min component size for edge bonuses
    "training_edge_progress_weight": 140.0,      # Reward progress toward missing edges
    "training_edge_progress_min_component": 3,   # Use overall bounds until component reaches this size
    "training_bridge_bonus": 80.0,               # Reward creating bridges
    "training_component_growth_bonus": 50.0,     # Reward growing largest component
    "training_isolated_penalty": 120.0,          # Penalize isolated placements
    "training_new_component_penalty": 150.0,     # Penalize creating new components
    "training_bridge_bonus_after_both_scale": 0.3,   # Downscale bridge bonus after both edges
    "training_growth_bonus_after_both_scale": 0.3,   # Downscale growth bonus after both edges
    "training_redundant_bridge_penalty": 120.0,      # Penalize bridges without progress
    "training_ladder_bonus": 180.0,              # Reward straight chain extension
    "training_ladder_max_dev": 2.0,              # Max lateral deviation for ladder bonus
    "training_ladder_requires_progress": 1,      # Require span/gap progress for ladder bonus
    "training_sealed_lane_penalty": 1600.0,      # Penalize chasing a sealed lane
    "training_midgame_start_ply": 8,             # Start adaptive sampling after opening
    "training_midgame_top_k": 1,                 # Sample from top-K after opening
    "training_sealed_lane_top_k": 1,             # Tighten sampling when lane is sealed
    "training_midgame_temp_scale": 0.0,          # Reduce temperature after opening
    "training_score_delta_frac": 0.1,            # Keep moves within top-score band
    "training_score_delta_abs": 300.0,           # Absolute score band for sampling
    "training_score_delta_cap": 100.0,           # Hard cap on score spread
    "training_guard_require_progress": 1,        # Require chain progress in sampling pool
    "training_guard_allow_connector": 1,         # Allow connector targets as progress moves
    "training_guard_start_ply": 6,               # Start guarded sampling after opening
    "training_force_deterministic_open_lane": 1, # Greedy when lane is open after opening
    "training_guard_require_defense": 1,         # Require defensive value when opponent has any pressure
    "training_force_deterministic_urgent": 1,    # Greedy when opponent threat is urgent
    "training_defense_span_threshold": 2,        # Opponent span to trigger defense guard
    "training_defense_corridor_margin": 4,       # Corridor margin for intercept moves
    "training_retreat_span_penalty": 200.0,      # Penalize span shrink after both edges
    "training_retreat_gap_penalty": 150.0,       # Penalize gap regression after both edges
    "training_retreat_requires_both": 1,         # Only penalize retreat after both edges
    "training_temp_scale_d2": 0.35,             # Reduce depth-2 sampling temperature
    "adjacentPenalty": 0.0,                 # Training-only override via normalize_knobs_for_mode
    "edgePushWeight": 0.0,                  # Training-only override via normalize_knobs_for_mode
    "edgePushMax": 0.0,                     # Training-only override via normalize_knobs_for_mode
    "missingEdgeHalfPenalty": 0.0,          # Training-only override via normalize_knobs_for_mode
    "globalSpanGainWeight": 0.0,            # Training-only override via normalize_knobs_for_mode
    "debug_sample_rate": 0.1,       # Fraction of games to record per-move scores
    "debug_max_plies": 80,          # Cap per-move logging
    "debug_trace": 0,               # 1 = capture per-move trace in stats
}


def get_knobs(knobs: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Merge provided knobs with defaults."""
    if knobs is None:
        return DEFAULT_KNOBS.copy()
    result = DEFAULT_KNOBS.copy()
    result.update(knobs)
    return result


# =============================================================================
# Valid Cells Precomputation (for optimized feature extraction)
# =============================================================================

def precompute_valid_cells(state: GameState, player: str) -> Set[Pos]:
    """Precompute all valid placement cells for a player.

    This enables O(1) validity lookups instead of calling is_valid_placement
    repeatedly in hot loops. For a 24x24 board with ~30 pegs, this is ~540
    valid cells that can be checked with a single set lookup.

    Args:
        state: Current game state
        player: Player to check validity for

    Returns:
        Set of (row, col) positions where player can legally place
    """
    board_size = state.board_size
    last = board_size - 1
    valid: Set[Pos] = set()

    for r in range(board_size):
        for c in range(board_size):
            # Skip occupied
            if (r, c) in state.pegs:
                continue

            # Skip corners
            if (r == 0 or r == last) and (c == 0 or c == last):
                continue

            # Player-specific edge restrictions
            if player == "red":
                # Red cannot place on left/right edges (col 0 or last)
                if c == 0 or c == last:
                    continue
            else:
                # Black cannot place on top/bottom edges (row 0 or last)
                if r == 0 or r == last:
                    continue

            valid.add((r, c))

    return valid


def _symmetrize_knobs(knobs: Dict[str, float]) -> Dict[str, float]:
    """Neutralize color-specific biases for training balance."""
    k = dict(knobs)

    def avg(key_a: str, key_b: str) -> None:
        va = float(k.get(key_a, 0.0))
        vb = float(k.get(key_b, 0.0))
        v = 0.5 * (va + vb)
        k[key_a] = v
        k[key_b] = v

    avg("firstEdgeRed", "firstEdgeBlack")
    avg("redSpanGainMultiplier", "blackSpanGainMultiplier")

    k["redGlobalMultiplier"] = 1.0
    k["blackGlobalScale"] = 1.0
    k["redBaseBonus"] = 0.0
    k["blackBasePenalty"] = 0.0
    k["redFinishExtra"] = 0.0
    k["redGapDecayMultiplier"] = 1.0
    k["redFinishPenaltyFactor"] = 1.0
    k["blackFinishScaleMultiplier"] = 1.0
    k["redDoubleCoverageBonus"] = 0.0
    k["blackDoubleCoverageScale"] = 1.0

    return k


def normalize_knobs_for_mode(
    knobs: Optional[Dict[str, float]],
    *,
    mode: str,
) -> Dict[str, float]:
    """Return knobs normalized for the given mode."""
    merged = get_knobs(knobs)
    if mode != "training":
        return merged

    merged["deterministic_mode"] = 0
    if merged.get("enforce_symmetry", 1):
        merged = _symmetrize_knobs(merged)

    black_span_mult = float(merged.get("training_black_span_multiplier", 1.0))
    if black_span_mult != 1.0:
        merged["blackSpanGainMultiplier"] = merged.get("blackSpanGainMultiplier", 1.0) * black_span_mult

    merged["goalDistance"] = merged.get("goalDistance", 1.0) * float(
        merged.get("training_goal_distance_mult", 1.0)
    )
    merged["centerBias"] = merged.get("centerBias", 1.0) * float(
        merged.get("training_center_bias_mult", 1.0)
    )
    merged["spanGainBase"] = merged.get("spanGainBase", 1.0) * float(
        merged.get("training_span_gain_mult", 1.0)
    )
    merged["connectorBonusScale"] = merged.get("connectorBonusScale", 1.0) * float(
        merged.get("training_connector_bonus_scale", 1.0)
    )
    merged["finishBonusScale"] = merged.get("finishBonusScale", 1.0) * float(
        merged.get("training_finish_bonus_scale", 1.0)
    )
    merged["adjacentPenalty"] = float(merged.get("training_adjacent_penalty", 0.0))
    merged["edgePushWeight"] = float(merged.get("training_edge_push_weight", 0.0))
    merged["edgePushMax"] = float(merged.get("training_edge_push_max", 0.0))
    merged["missingEdgeHalfPenalty"] = float(merged.get("training_missing_edge_half_penalty", 0.0))
    merged["globalSpanGainWeight"] = float(merged.get("training_global_span_gain_weight", 0.0))
    first_edge_scale = float(merged.get("training_first_edge_bonus_scale", 1.0))
    if first_edge_scale != 1.0:
        merged["firstEdgeRed"] = merged.get("firstEdgeRed", 0.0) * first_edge_scale
        merged["firstEdgeBlack"] = merged.get("firstEdgeBlack", 0.0) * first_edge_scale
    defense_scale = float(merged.get("training_defense_scale", 1.0))
    if defense_scale != 1.0:
        merged["blockBonus"] = merged.get("blockBonus", 0.0) * defense_scale
        merged["opponentConnection"] = merged.get("opponentConnection", 0.0) * defense_scale
        merged["opponentDistance"] = merged.get("opponentDistance", 0.0) * defense_scale
    return merged


# =============================================================================
# Component analysis functions
# =============================================================================

def _get_player_adjacency(state: GameState, player: str) -> Dict[Pos, List[Pos]]:
    """Build adjacency list from bridges for BOTH players and cache it.

    Cost (first build per state): O(B)
    Subsequent calls: O(1)
    """
    assert player in ("red", "black")

    cached = state._adj_cache
    if cached is not None:
        rev, adj_by_player = cached
        if rev == state.cc_revision:
            return adj_by_player[player]

    # Bind locals for hot loop
    pegs = state.pegs
    bridges = state.bridges

    adj_by_player: Dict[str, Dict[Pos, List[Pos]]] = {"red": {}, "black": {}}
    adj_red = adj_by_player["red"]
    adj_black = adj_by_player["black"]

    for a, b in bridges:
        pa = pegs.get(a)
        if pa is None:
            continue
        if pa != pegs.get(b):
            continue

        adj = adj_red if pa == "red" else adj_black
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    state._adj_cache = (state.cc_revision, adj_by_player)
    return adj_by_player[player]


def _compute_connected_components(
    pegs: Dict[Pos, str], adj: Dict[Pos, List[Pos]], player: str
) -> Components:
    """DFS over adjacency list to find connected components.

    Returns IMMUTABLE tuple-of-tuples.
    Optimized: scans pegs dict once instead of building intermediate list.
    """
    visited: Set[Pos] = set()
    components: List[Component] = []

    for pos, owner in pegs.items():
        if owner != player or pos in visited:
            continue

        if pos not in adj:
            # Isolated peg = singleton component
            visited.add(pos)
            components.append((pos,))
            continue

        # DFS
        stack = [pos]
        visited.add(pos)
        component: List[Pos] = []

        while stack:
            curr = stack.pop()
            component.append(curr)
            for neighbor in adj.get(curr, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(tuple(component))

    return tuple(components)


def find_connected_components(state: GameState, player: str) -> Components:
    """Find all bridge-connected components for a player.

    Returns IMMUTABLE tuple-of-tuples. Cached per (state revision, player).
    """
    assert player in ("red", "black")

    # Check cache
    cached = state._cc_cache.get(player)
    if cached is not None:
        rev, components = cached
        if rev == state.cc_revision:
            return components

    # No early exit - let _compute_connected_components return () naturally
    # This avoids double-scanning state.pegs.values()
    adj = _get_player_adjacency(state, player)
    result = _compute_connected_components(state.pegs, adj, player)
    state._cc_cache[player] = (state.cc_revision, result)
    return result


def component_metrics(
    state: GameState,
    player: str,
    components: Optional[Components] = None,
) -> Dict:
    """Compute metrics about all components for a player.

    IMPORTANT: touches_top/bottom/left/right are for the LARGEST component only!
    This prevents rewarding having disconnected pegs on opposite edges.

    Args:
        state: Current game state
        player: "red" or "black"
        components: Pre-computed components (optional, avoids redundant lookup)

    Returns dict with:
    - components: list of all components
    - max_row_span: max row span of largest component
    - max_col_span: max col span of largest component
    - touches_top/bottom/left/right: edge touching flags (ANY component, matching JS)
    - largest_component: the biggest component
    - min_row/max_row/min_col/max_col: bounds across ALL components (matching JS)
    """
    if components is None:
        components = find_connected_components(state, player)
    board_size = state.board_size

    # Track overall bounds across ALL components (matching JS componentMetrics)
    min_row_overall = board_size
    max_row_overall = -1
    min_col_overall = board_size
    max_col_overall = -1

    # Track max spans and largest component
    max_row_span = 0
    max_col_span = 0
    largest_component: Component = ()

    # Track edge touches across ANY component (matching JS)
    touches_top = False
    touches_bottom = False
    touches_left = False
    touches_right = False

    for component in components:
        rows = [r for r, c in component]
        cols = [c for r, c in component]

        comp_min_row = min(rows)
        comp_max_row = max(rows)
        comp_min_col = min(cols)
        comp_max_col = max(cols)

        # Update overall bounds (across ALL components)
        min_row_overall = min(min_row_overall, comp_min_row)
        max_row_overall = max(max_row_overall, comp_max_row)
        min_col_overall = min(min_col_overall, comp_min_col)
        max_col_overall = max(max_col_overall, comp_max_col)

        # Update spans (track max across all components)
        max_row_span = max(max_row_span, comp_max_row - comp_min_row)
        max_col_span = max(max_col_span, comp_max_col - comp_min_col)

        # Update largest component
        if len(component) > len(largest_component):
            largest_component = component

        # Update edge touches (ANY component touching = True, matching JS)
        if comp_min_row == 0:
            touches_top = True
        if comp_max_row == board_size - 1:
            touches_bottom = True
        if comp_min_col == 0:
            touches_left = True
        if comp_max_col == board_size - 1:
            touches_right = True

    # If no components, return empty metrics
    if not largest_component:
        return {
            "components": components,
            "max_row_span": 0,
            "max_col_span": 0,
            "touches_top": False,
            "touches_bottom": False,
            "touches_left": False,
            "touches_right": False,
            "largest_component": (),
            "min_row": None,
            "max_row": None,
            "min_col": None,
            "max_col": None,
        }

    return {
        "components": components,
        "max_row_span": max_row_span,
        "max_col_span": max_col_span,
        "touches_top": touches_top,
        "touches_bottom": touches_bottom,
        "touches_left": touches_left,
        "touches_right": touches_right,
        "largest_component": largest_component,
        "min_row": min_row_overall if min_row_overall < board_size else None,
        "max_row": max_row_overall if max_row_overall >= 0 else None,
        "min_col": min_col_overall if min_col_overall < board_size else None,
        "max_col": max_col_overall if max_col_overall >= 0 else None,
    }


def compute_frontier(
    state: GameState,
    player: str,
    metrics: Optional[Dict] = None,
) -> Dict:
    """Compute frontier moves (knight-move reachable empty cells from largest component).

    Args:
        state: Current game state
        player: "red" or "black"
        metrics: Pre-computed component_metrics (optional, avoids redundant call)

    Returns dict with:
    - frontier: all reachable empty cells
    - connectors: cells near goal edges
    - trailing: cells not near goal edges
    - metrics: component metrics
    """
    if metrics is None:
        metrics = component_metrics(state, player)
    component = metrics["largest_component"]
    board_size = state.board_size

    frontier: List[Pos] = []
    connectors: List[Pos] = []
    trailing: List[Pos] = []
    seen: Set[Pos] = set()

    if not component:
        return {"frontier": frontier, "metrics": metrics, "connectors": connectors, "trailing": trailing}

    want_top = player == "red" and not metrics["touches_top"]
    want_bottom = player == "red" and not metrics["touches_bottom"]
    want_left = player == "black" and not metrics["touches_left"]
    want_right = player == "black" and not metrics["touches_right"]

    for peg_row, peg_col in component:
        for dr, dc in KNIGHT_OFFSETS:
            row = peg_row + dr
            col = peg_col + dc

            if row < 0 or row >= board_size or col < 0 or col >= board_size:
                continue
            if (row, col) in state.pegs:
                continue

            pos = (row, col)
            if pos in seen:
                continue

            # Check valid placement for player
            if not is_valid_placement(state, player, row, col):
                continue

            frontier.append(pos)

            # Determine if this is a connector (near goal edge)
            is_connector = False
            if player == "red":
                top_threshold = 5 if want_top else 3
                bottom_threshold = 5 if want_bottom else 3
                if want_top and row <= top_threshold:
                    is_connector = True
                if want_bottom and row >= board_size - 1 - bottom_threshold:
                    is_connector = True
                if not want_top and not want_bottom:
                    if row <= top_threshold or row >= board_size - 1 - bottom_threshold:
                        is_connector = True
            else:
                left_threshold = 5 if want_left else 3
                right_threshold = 5 if want_right else 3
                if want_left and col <= left_threshold:
                    is_connector = True
                if want_right and col >= board_size - 1 - right_threshold:
                    is_connector = True
                if not want_left and not want_right:
                    if col <= left_threshold or col >= board_size - 1 - right_threshold:
                        is_connector = True

            if is_connector:
                connectors.append(pos)
            else:
                trailing.append(pos)

            seen.add(pos)

    return {"frontier": frontier, "metrics": metrics, "connectors": connectors, "trailing": trailing}


def compute_frontier_fast(
    state: GameState,
    player: str,
    metrics: Dict,
    valid_cells: Set[Pos],
) -> Dict:
    """Optimized version using precomputed valid cells.

    Same logic as compute_frontier but uses O(1) set lookup instead of
    calling is_valid_placement for each knight move.
    """
    component = metrics["largest_component"]
    board_size = state.board_size

    frontier: List[Pos] = []
    connectors: List[Pos] = []
    trailing: List[Pos] = []
    seen: Set[Pos] = set()

    if not component:
        return {"frontier": frontier, "metrics": metrics, "connectors": connectors, "trailing": trailing}

    want_top = player == "red" and not metrics["touches_top"]
    want_bottom = player == "red" and not metrics["touches_bottom"]
    want_left = player == "black" and not metrics["touches_left"]
    want_right = player == "black" and not metrics["touches_right"]

    for peg_row, peg_col in component:
        for dr, dc in KNIGHT_OFFSETS:
            row = peg_row + dr
            col = peg_col + dc
            pos = (row, col)

            if pos in seen:
                continue

            # Use precomputed valid cells (O(1) lookup)
            if pos not in valid_cells:
                continue

            frontier.append(pos)

            # Determine if this is a connector (near goal edge)
            is_connector = False
            if player == "red":
                top_threshold = 5 if want_top else 3
                bottom_threshold = 5 if want_bottom else 3
                if want_top and row <= top_threshold:
                    is_connector = True
                if want_bottom and row >= board_size - 1 - bottom_threshold:
                    is_connector = True
                if not want_top and not want_bottom:
                    if row <= top_threshold or row >= board_size - 1 - bottom_threshold:
                        is_connector = True
            else:
                left_threshold = 5 if want_left else 3
                right_threshold = 5 if want_right else 3
                if want_left and col <= left_threshold:
                    is_connector = True
                if want_right and col >= board_size - 1 - right_threshold:
                    is_connector = True
                if not want_left and not want_right:
                    if col <= left_threshold or col >= board_size - 1 - right_threshold:
                        is_connector = True

            if is_connector:
                connectors.append(pos)
            else:
                trailing.append(pos)

            seen.add(pos)

    return {"frontier": frontier, "metrics": metrics, "connectors": connectors, "trailing": trailing}


# =============================================================================
# Scoring functions (with knobs)
# =============================================================================

def score_component(component: Component, player: str, board_size: int, k: Dict[str, float]) -> float:
    """Score a single connected component using HARDCODED values matching JS.

    JS scoreComponent (lines 297-321):
    - score = component.length * 10
    - score += span * 20
    - if full span (0↔boardSize-1): score += 500
    """
    # JS: component.length * 10
    score = len(component) * COMPONENT_SIZE_MULT

    if player == "red":
        rows = [r for r, c in component]
        min_row = min(rows)
        max_row = max(rows)
        span = max_row - min_row
        # JS: span * 20
        score += span * SPAN_MULT

        # JS: +500 for true top-to-bottom span
        if min_row == 0 and max_row == board_size - 1:
            score += FULL_SPAN_BONUS
    else:
        cols = [c for r, c in component]
        min_col = min(cols)
        max_col = max(cols)
        span = max_col - min_col
        # JS: span * 20
        score += span * SPAN_MULT

        # JS: +500 for true left-to-right span
        if min_col == 0 and max_col == board_size - 1:
            score += FULL_SPAN_BONUS

    return score


def evaluate_winning_threats(player: str, components: List[Component], board_size: int, k: Dict[str, float]) -> float:
    """Evaluate winning threat potential from near-complete spans.

    JS evaluateWinningThreats (lines 219-242):
    - Full span (0↔23): +800
    - Near span (≤1↔≥22): +400
    - Close span (≥22, ≤5): +400
    - Medium span (≤3↔≥20): +200

    Uses HARDCODED values to match JS exactly.
    """
    score = 0.0

    for component in components:
        if player == "red":
            rows = [r for r, c in component]
            min_row = min(rows)
            max_row = max(rows)

            # JS: if (minRow === 0 && maxRow === 23) score += 800
            if min_row == 0 and max_row == board_size - 1:
                score += THREAT_FULL_SPAN
            # JS: else if (minRow <= 1 && maxRow >= 22) score += 400
            elif min_row <= 1 and max_row >= board_size - 2:
                score += THREAT_NEAR_SPAN
            # JS: else if (maxRow >= 22 && minRow <= 5) score += 400
            elif max_row >= board_size - 2 and min_row <= 5:
                score += THREAT_NEAR_SPAN
            # JS: else if (minRow <= 3 && maxRow >= 20) score += 200
            elif min_row <= 3 and max_row >= board_size - 4:
                score += THREAT_MEDIUM_SPAN
        else:
            cols = [c for r, c in component]
            min_col = min(cols)
            max_col = max(cols)

            # JS: if (minCol === 0 && maxCol === 23) score += 800
            if min_col == 0 and max_col == board_size - 1:
                score += THREAT_FULL_SPAN
            # JS: else if (minCol <= 1 && maxCol >= 22) score += 400
            elif min_col <= 1 and max_col >= board_size - 2:
                score += THREAT_NEAR_SPAN
            # JS: else if (maxCol >= 22 && minCol <= 5) score += 400
            elif max_col >= board_size - 2 and min_col <= 5:
                score += THREAT_NEAR_SPAN
            # JS: else if (minCol <= 3 && maxCol >= 20) score += 200
            elif min_col <= 3 and max_col >= board_size - 4:
                score += THREAT_MEDIUM_SPAN

    return score


def evaluate_connected_paths(
    state: GameState,
    player: str,
    k: Dict[str, float],
    components: Optional[Components] = None,
) -> float:
    """Evaluate connectivity quality for a player.

    JS evaluateConnectedPaths (lines 195-217):
    - Sum scoreComponent() for each component
    - avgComponentSize * 20
    - (components.length - 3) * 30 penalty if > 3 components
    - + evaluateWinningThreats()

    Args:
        state: Current game state
        player: "red" or "black"
        k: Knobs dict
        components: Pre-computed components (optional, avoids redundant lookup)

    Uses HARDCODED values to match JS exactly.
    """
    if components is None:
        components = find_connected_components(state, player)

    # Derive peg count from components (avoids O(P) scan of state.pegs)
    peg_count = sum(len(c) for c in components)
    if peg_count == 0:
        return -100.0
    if not components:
        return -50.0

    score = 0.0

    for component in components:
        score += score_component(component, player, state.board_size, k)

    # JS: avgComponentSize * 20 and (components.length - 3) * 30 penalty
    if components:
        avg_component_size = peg_count / len(components)
        score += avg_component_size * AVG_COMPONENT_SIZE_MULT
        if len(components) > 3:
            score -= (len(components) - 3) * COMPONENT_PENALTY_MULT

    score += evaluate_winning_threats(player, components, state.board_size, k)
    return score


def evaluate_potential_connections(state: GameState, player: str, k: Dict[str, float]) -> float:
    """Evaluate potential knight-move expansions.

    JS evaluatePotentialConnections (lines 323-344):
    - For each peg, check knight moves that progress toward goal
    - score += 5 for each valid goal-direction move

    Uses HARDCODED value (5) to match JS exactly.
    """
    score = 0.0
    player_pegs = [(r, c) for (r, c), p in state.pegs.items() if p == player]

    for row, col in player_pegs:
        for dr, dc in KNIGHT_OFFSETS:
            new_row = row + dr
            new_col = col + dc
            if is_valid_placement(state, player, new_row, new_col):
                if player == "red":
                    # JS: if (peg.row < 12 && newRow > peg.row) score += 5
                    if row < 12 and new_row > row:
                        score += POTENTIAL_MOVE_BONUS
                    # JS: if (peg.row > 12 && newRow < peg.row) score += 5
                    if row > 12 and new_row < row:
                        score += POTENTIAL_MOVE_BONUS
                else:
                    # JS: if (peg.col < 12 && newCol > peg.col) score += 5
                    if col < 12 and new_col > col:
                        score += POTENTIAL_MOVE_BONUS
                    # JS: if (peg.col > 12 && newCol < peg.col) score += 5
                    if col > 12 and new_col < col:
                        score += POTENTIAL_MOVE_BONUS

    return score


def evaluate_potential_connections_fast(
    state: GameState,
    player: str,
    valid_cells: Set[Pos],
) -> float:
    """Optimized version using precomputed valid cells.

    Same logic as evaluate_potential_connections but uses O(1) set lookup
    instead of calling is_valid_placement for each knight move.
    """
    score = 0.0
    player_pegs = [(r, c) for (r, c), p in state.pegs.items() if p == player]

    for row, col in player_pegs:
        for dr, dc in KNIGHT_OFFSETS:
            new_row = row + dr
            new_col = col + dc
            if (new_row, new_col) in valid_cells:
                if player == "red":
                    if row < 12 and new_row > row:
                        score += POTENTIAL_MOVE_BONUS
                    if row > 12 and new_row < row:
                        score += POTENTIAL_MOVE_BONUS
                else:
                    if col < 12 and new_col > col:
                        score += POTENTIAL_MOVE_BONUS
                    if col > 12 and new_col < col:
                        score += POTENTIAL_MOVE_BONUS

    return score


def evaluate_edge_progress(state: GameState, player: str, k: Dict[str, float]) -> float:
    """Evaluate progress toward goal edges.

    JS evaluateEdgeProgress (lines 346-361) is SIMPLE:
    - For each peg: score += max(0, 12 - distanceToGoal)
    - distanceToGoal = min(row, boardSize - 1 - row) for red
    - distanceToGoal = min(col, boardSize - 1 - col) for black

    Uses HARDCODED value (12) to match JS exactly.
    Note: The complex span/gap/finish logic was WRONG - JS doesn't have it.
    """
    score = 0.0
    player_pegs = [(r, c) for (r, c), p in state.pegs.items() if p == player]
    board_size = state.board_size

    for row, col in player_pegs:
        if player == "red":
            # JS: Math.min(peg.row, game.boardSize - 1 - peg.row)
            distance_to_goal = min(row, board_size - 1 - row)
        else:
            # JS: Math.min(peg.col, game.boardSize - 1 - peg.col)
            distance_to_goal = min(col, board_size - 1 - col)

        # JS: Math.max(0, 12 - distanceToGoal)
        score += max(0, EDGE_DISTANCE_MAX - distance_to_goal)

    return score


def evaluate_position(
    state: GameState,
    player: str,
    knobs: Optional[Dict[str, float]] = None,
    *,
    player_components: Optional[Components] = None,
    opponent_components: Optional[Components] = None,
    player_metrics: Optional[Dict] = None,
) -> float:
    """Evaluate a position for a player.

    Main evaluation function that combines all heuristics using knobs.
    Returns score where positive = good for player.

    Args:
        state: Current game state
        player: Player to evaluate for
        knobs: Optional tuning parameters
        player_components: Pre-computed player components (avoids redundant computation)
        opponent_components: Pre-computed opponent components
        player_metrics: Pre-computed player metrics
    """
    k = get_knobs(knobs)

    # Check for game over
    from ..game.rules import check_winner
    winner = check_winner(state)
    if winner is not None:
        return 10000.0 if winner == player else -10000.0

    opponent = "red" if player == "black" else "black"
    score = 0.0

    # Compute components if not provided
    if player_components is None:
        player_components = find_connected_components(state, player)
    if opponent_components is None:
        opponent_components = find_connected_components(state, opponent)

    # Connected paths evaluation - uses HARDCODED constants (not knobs)
    score += evaluate_connected_paths(state, player, k, player_components) * CONNECTED_PATHS_MULTIPLIER
    score -= evaluate_connected_paths(state, opponent, k, opponent_components) * CONNECTED_PATHS_MULTIPLIER

    # Potential connections - uses HARDCODED constant
    score += evaluate_potential_connections(state, player, k) * POTENTIAL_CONNECTIONS_MULTIPLIER
    score -= evaluate_potential_connections(state, opponent, k) * POTENTIAL_CONNECTIONS_MULTIPLIER

    # Edge progress - uses HARDCODED constant
    score += evaluate_edge_progress(state, player, k) * EDGE_PROGRESS_MULTIPLIER
    score -= evaluate_edge_progress(state, opponent, k) * EDGE_PROGRESS_MULTIPLIER

    # Peg count advantage - uses HARDCODED constant
    player_pegs = sum(1 for p in state.pegs.values() if p == player)
    opponent_pegs = sum(1 for p in state.pegs.values() if p == opponent)
    score += (player_pegs - opponent_pegs) * PEG_DIFFERENCE_MULTIPLIER

    # Shortest-path-to-goal pull on largest component (key for finishing!)
    # Use pre-computed metrics if available
    if player_metrics is None:
        metrics = component_metrics(state, player, player_components)
    else:
        metrics = player_metrics
    largest = metrics["largest_component"]
    if largest:
        min_r = min(r for r, c in largest)
        max_r = max(r for r, c in largest)
        min_c = min(c for r, c in largest)
        max_c = max(c for r, c in largest)

        n = state.board_size
        top_thr = 1
        bot_thr = n - 2
        left_thr = 1
        right_thr = n - 2

        touches_top = min_r <= top_thr
        touches_bottom = max_r >= bot_thr
        touches_left = min_c <= left_thr
        touches_right = max_c >= right_thr

        gap_top = max(0, min_r - top_thr)
        gap_bottom = max(0, bot_thr - max_r)
        gap_left = max(0, min_c - left_thr)
        gap_right = max(0, right_thr - max_c)

        # Use HARDCODED constants matching JS heuristics.js
        if player == "red":
            gap = gap_bottom if touches_top else (gap_top if touches_bottom else min(gap_top, gap_bottom))
            mult = URGENCY_MULTIPLIER if (touches_top or touches_bottom) else 1.0
            # Strong pull toward finishing - JS: 200 * urgency * (1/(1+gap))
            score += GAP_PULL_BASE * mult * (1 / (1 + gap))
            # Gap penalty - JS: 40 * (gapTop + gapBottom)
            score -= GAP_PENALTY_BASE * (gap_top + gap_bottom)
        else:
            gap = gap_right if touches_left else (gap_left if touches_right else min(gap_left, gap_right))
            mult = URGENCY_MULTIPLIER if (touches_left or touches_right) else 1.0
            score += GAP_PULL_BASE * mult * (1 / (1 + gap))
            score -= GAP_PENALTY_BASE * (gap_left + gap_right)

    # Move drift penalty (prefer finishing sooner) - JS: 0.05 * moveCount
    move_count = len(state.move_history)
    score -= DRIFT_PENALTY * move_count

    return score


def evaluate_move(
    state: GameState,
    row: int,
    col: int,
    player: str,
    knobs: Optional[Dict[str, float]] = None
) -> float:
    """Evaluate a candidate move for a player using knobs.

    Scores the move based on:
    - Bridge connections created
    - Span bonuses (wide connections)
    - Goal distance
    - Blocking opponent threats
    - Center bias (early game)
    """
    k = get_knobs(knobs)
    score = 0.0
    opponent = "red" if player == "black" else "black"
    board_size = state.board_size

    # Count bridge connections - uses HARDCODED constants matching JS
    connection_count = 0
    for dr, dc in KNIGHT_OFFSETS:
        check_row = row + dr
        check_col = col + dc

        if check_row < 0 or check_row >= board_size:
            continue
        if check_col < 0 or check_col >= board_size:
            continue

        if (check_row, check_col) in state.pegs and state.pegs[(check_row, check_col)] == player:
            # Check if bridge would cross existing bridges
            if not bridges_cross(state, row, col, check_row, check_col):
                connection_count += 1
                distance = abs(row - check_row) + abs(col - check_col)
                # JS: score += 100 + distance * 5
                score += CONNECTION_BASE_BONUS + distance * CONNECTION_DISTANCE_MULT

                # Span bonuses - JS: spans_board +300, wide_span +150
                if player == "black":
                    spans_board = (col <= 3 and check_col >= 20) or (col >= 20 and check_col <= 3)
                    wide_span = abs(col - check_col) > 10
                    if spans_board:
                        score += SPAN_BOARD_BONUS
                    elif wide_span:
                        score += SPAN_WIDE_BONUS
                else:
                    spans_board = (row <= 3 and check_row >= 20) or (row >= 20 and check_row <= 3)
                    wide_span = abs(row - check_row) > 10
                    if spans_board:
                        score += SPAN_BOARD_BONUS
                    elif wide_span:
                        score += SPAN_WIDE_BONUS

    # Multi-connection bonus - JS: connectionCount * 75
    if connection_count >= 2:
        score += connection_count * MULTI_CONNECTION_MULT

    # Goal distance - JS: max(0, 12 - distanceToNearestGoal) * 8
    if player == "red":
        distance_to_goal = min(row, board_size - 1 - row)
    else:
        distance_to_goal = min(col, board_size - 1 - col)
    score += max(0, GOAL_DISTANCE_MAX - distance_to_goal) * GOAL_DISTANCE_MULT

    # NOTE: firstEdgeTouch bonuses are handled in search.js, NOT here.
    # JS evaluateMove() does NOT have firstEdge logic - removed to match JS.

    # Opponent threat blocking - JS: opponentThreats * 25
    opponent_threats = 0
    for dr, dc in KNIGHT_OFFSETS:
        check_row = row + dr
        check_col = col + dc

        if check_row < 0 or check_row >= board_size:
            continue
        if check_col < 0 or check_col >= board_size:
            continue

        if (check_row, check_col) in state.pegs and state.pegs[(check_row, check_col)] == opponent:
            opponent_threats += 1

    if opponent_threats > 0:
        score += opponent_threats * OPPONENT_THREAT_MULT

    # Center bias (early game only) - JS: max(0, 24 - centerDistance) * 2
    # Note: JS uses hardcoded * 2, NOT * centerBias knob
    move_count = len(state.move_history)
    if move_count < 10:
        center = (board_size - 1) / 2  # 11.5 for 24x24
        center_distance = abs(row - center) + abs(col - center)
        score += max(0, CENTER_BIAS_MAX_DIST - center_distance) * CENTER_BIAS_MULT

    return score


# =============================================================================
# Feature extraction
# =============================================================================

def extract_features(state: GameState, player: Optional[str] = None) -> Dict[str, float]:
    """Compute feature vector for a position.

    If player is not specified, uses state.to_move.
    Returns dict of feature_name -> value.
    """
    if player is None:
        player = state.to_move
    opponent = "red" if player == "black" else "black"
    k = DEFAULT_KNOBS

    features: Dict[str, float] = {}

    # Connected paths
    features["friendly_connected_paths"] = evaluate_connected_paths(state, player, k)
    features["opponent_connected_paths"] = evaluate_connected_paths(state, opponent, k)

    # Potential connections
    features["friendly_potential"] = evaluate_potential_connections(state, player, k)
    features["opponent_potential"] = evaluate_potential_connections(state, opponent, k)

    # Edge progress
    features["friendly_edge_progress"] = evaluate_edge_progress(state, player, k)
    features["opponent_edge_progress"] = evaluate_edge_progress(state, opponent, k)

    # Peg counts
    features["friendly_pegs"] = sum(1 for p in state.pegs.values() if p == player)
    features["opponent_pegs"] = sum(1 for p in state.pegs.values() if p == opponent)

    # Component metrics
    friendly_metrics = component_metrics(state, player)
    opponent_metrics = component_metrics(state, opponent)

    features["friendly_max_row_span"] = friendly_metrics["max_row_span"]
    features["friendly_max_col_span"] = friendly_metrics["max_col_span"]
    features["opponent_max_row_span"] = opponent_metrics["max_row_span"]
    features["opponent_max_col_span"] = opponent_metrics["max_col_span"]

    features["friendly_component_count"] = len(friendly_metrics["components"])
    features["opponent_component_count"] = len(opponent_metrics["components"])

    features["friendly_largest_size"] = len(friendly_metrics["largest_component"])
    features["opponent_largest_size"] = len(opponent_metrics["largest_component"])

    # Edge touches
    features["friendly_touches_top"] = 1.0 if friendly_metrics["touches_top"] else 0.0
    features["friendly_touches_bottom"] = 1.0 if friendly_metrics["touches_bottom"] else 0.0
    features["friendly_touches_left"] = 1.0 if friendly_metrics["touches_left"] else 0.0
    features["friendly_touches_right"] = 1.0 if friendly_metrics["touches_right"] else 0.0

    features["opponent_touches_top"] = 1.0 if opponent_metrics["touches_top"] else 0.0
    features["opponent_touches_bottom"] = 1.0 if opponent_metrics["touches_bottom"] else 0.0
    features["opponent_touches_left"] = 1.0 if opponent_metrics["touches_left"] else 0.0
    features["opponent_touches_right"] = 1.0 if opponent_metrics["touches_right"] else 0.0

    # Move count
    features["move_count"] = len(state.move_history)

    # Bridge count
    features["total_bridges"] = len(state.bridges)

    # Frontier analysis
    frontier = compute_frontier(state, player)
    features["frontier_size"] = len(frontier["frontier"])
    features["connector_count"] = len(frontier["connectors"])
    features["trailing_count"] = len(frontier["trailing"])

    return features


# =============================================================================
# Sealed Lane Detection and Connector Targets (ported from search.js)
# =============================================================================

def is_goal_edge_coordinate(player: str, row: int, col: int, board_size: int) -> bool:
    """Check if a coordinate is on the player's goal edge.

    JS: isGoalEdgeCoordinate (search.js lines 202-213)
    """
    if player == "red":
        # Red's goal edges are row 0 and row boardSize-1
        if row != 0 and row != board_size - 1:
            return False
        # Must not be in corners (col 0 or col boardSize-1)
        return 0 < col < board_size - 1
    else:
        # Black's goal edges are col 0 and col boardSize-1
        if col != 0 and col != board_size - 1:
            return False
        # Must not be in corners (row 0 or row boardSize-1)
        return 0 < row < board_size - 1


def has_reachable_goal_edge(
    state: GameState,
    player: str,
    metrics: Dict,
) -> bool:
    """Check if the player can still reach their goal edge via BFS.

    JS: hasReachableGoalEdge (search.js lines 215-389)

    This is "sealed lane" detection - returns False if the player's path
    to victory is completely blocked by opponent bridges/pegs.
    """
    component = metrics.get("largest_component", [])
    if not component:
        return False

    board_size = state.board_size

    # Determine which goal edges we need to reach
    target_set: Set[int] = set()
    if player == "red":
        if not metrics.get("touches_top"):
            target_set.add(0)
        if not metrics.get("touches_bottom"):
            target_set.add(board_size - 1)
    else:
        if not metrics.get("touches_left"):
            target_set.add(0)
        if not metrics.get("touches_right"):
            target_set.add(board_size - 1)

    # If already touching all needed edges, lane is open
    if not target_set:
        return True

    # BFS to find if we can reach a goal edge
    visited: Set[str] = set()
    queue: List[Tuple[int, int, str]] = []  # (row, col, type)

    def enqueue(row: int, col: int, cell_type: str):
        key = f"{cell_type}:{row}:{col}"
        if key in visited:
            return
        visited.add(key)
        queue.append((row, col, cell_type))

    # Start from all pegs in the component
    for r, c in component:
        enqueue(r, c, "peg")

    head = 0
    while head < len(queue):
        row, col, cell_type = queue[head]
        head += 1

        # Check if we've reached a goal edge
        if player == "red":
            if row in target_set and is_goal_edge_coordinate(player, row, col, board_size):
                if cell_type == "peg" or is_valid_placement(state, player, row, col):
                    return True
        else:
            if col in target_set and is_goal_edge_coordinate(player, row, col, board_size):
                if cell_type == "peg" or is_valid_placement(state, player, row, col):
                    return True

        # Skip invalid placements for empty cells
        if cell_type == "empty" and not is_valid_placement(state, player, row, col):
            continue

        # Explore knight moves
        for dr, dc in KNIGHT_OFFSETS:
            nr, nc = row + dr, col + dc

            if nr < 0 or nr >= board_size or nc < 0 or nc >= board_size:
                continue

            # Determine next cell type
            if (nr, nc) not in state.pegs:
                next_type = "empty"
            elif state.pegs[(nr, nc)] == player:
                next_type = "peg"
            else:
                continue  # Opponent peg, can't pass

            # Check if bridge would cross existing bridges
            if bridges_cross(state, row, col, nr, nc):
                continue

            enqueue(nr, nc, next_type)

    return False


def compute_connector_targets(
    state: GameState,
    player: str,
    metrics: Dict,
    radius: int = 3,
) -> Optional[Set[str]]:
    """Compute target cells at the edge of the largest component.

    JS: computeConnectorTargets (search.js lines 391-437)

    Returns a Set of "row:col" strings for cells that would extend
    the component toward the goal.
    """
    component = metrics.get("largest_component", [])
    if not component:
        return None

    board_size = state.board_size

    # Find bounding box of component
    min_r = min(r for r, c in component)
    max_r = max(r for r, c in component)
    min_c = min(c for r, c in component)
    max_c = max(c for r, c in component)

    targets: Set[str] = set()

    def add_target(row: int, col: int):
        if row < 0 or row >= board_size or col < 0 or col >= board_size:
            return
        if (row, col) in state.pegs:
            return
        # Check player-specific edge restrictions
        if player == "red" and (col == 0 or col == board_size - 1):
            return
        if player == "black" and (row == 0 or row == board_size - 1):
            return
        targets.add(f"{row}:{col}")

    if player == "red":
        # Red extends vertically - add targets above and below
        for c in range(min_c - radius, max_c + radius + 1):
            add_target(min_r - 1, c)
            add_target(max_r + 1, c)
    else:
        # Black extends horizontally - add targets left and right
        for r in range(min_r - radius, max_r + radius + 1):
            add_target(r, min_c - 1)
            add_target(r, max_c + 1)

    return targets if targets else None


def compute_opponent_urgent(
    opponent: str,
    opponent_metrics: Dict,
    board_size: int,
) -> bool:
    """Determine if opponent is in an urgent (threatening) position.

    JS version (search.js lines 900-907):
    const opponentUrgent = spanValue >= Math.max(6, Math.floor(boardSize / 4)) || largestLength >= 6;
    """
    if not opponent_metrics:
        return False

    largest_component = opponent_metrics.get("largest_component", ())
    largest_length = len(largest_component)

    if opponent == "red":
        span_value = opponent_metrics.get("max_row_span", 0)
    else:
        span_value = opponent_metrics.get("max_col_span", 0)

    threshold = max(6, board_size // 4)
    return span_value >= threshold or largest_length >= 6


def min_distance_to_pegs(row: int, col: int, pegs: List[Pos]) -> float:
    """Compute minimum Manhattan distance to any peg in the list."""
    if not pegs:
        return float('inf')
    return min(abs(row - r) + abs(col - c) for r, c in pegs)


def distance_to_component(row: int, col: int, component: List[Pos]) -> float:
    """Compute minimum Manhattan distance to any peg in a component."""
    if not component:
        return float('inf')
    return min(abs(row - r) + abs(col - c) for r, c in component)


def move_priority(
    state: GameState,
    row: int,
    col: int,
    player: str,
    knobs: Dict[str, float],
    friendly_pegs: List[Pos],
    opponent_pegs: List[Pos],
    friendly_metrics: Dict,
    opponent_metrics: Dict,
    friendly_frontier: Dict,
    opponent_frontier: Dict,
    sealed_lane_cache: Optional["SealedLaneLRU"] = None,
    opponent_threat_before: float = 0.0,
    child_state: Optional[GameState] = None,
    opponent_components: Optional[Components] = None,
    post_components: Optional[Components] = None,
    post_metrics: Optional[Dict] = None,
) -> float:
    """Score a move for ordering using REWARDS from search.json.

    This is the Python port of JS search.js movePriority() function.
    Uses tunable knobs for all scoring to match JS behavior.

    Args:
        ...
        opponent_threat_before: Opponent's connectivity score before this move.
            Used to calculate threat reduction bonus/penalty.
        child_state: Pre-computed child state (optional, avoids redundant apply_move).
        opponent_components: Pre-computed opponent components (invariant under our move).
        post_components: Pre-computed player components after this move (optional).
        post_metrics: Pre-computed player metrics after this move (optional).
    """
    opponent = "red" if player == "black" else "black"
    board_size = state.board_size
    score = 0.0

    # Count connections to friendly and opponent pegs
    friendly_connections = 0
    opponent_connections = 0
    for dr, dc in KNIGHT_OFFSETS:
        r, c = row + dr, col + dc
        if r < 0 or r >= board_size or c < 0 or c >= board_size:
            continue
        if (r, c) in state.pegs:
            if state.pegs[(r, c)] == player:
                friendly_connections += 1
            else:
                opponent_connections += 1

    # Connection scoring - uses knobs
    score += friendly_connections * knobs["friendlyConnection"]
    score += opponent_connections * knobs["opponentConnection"]

    # Distance to friendly pegs - uses knobs
    friendly_dist = min_distance_to_pegs(row, col, friendly_pegs)
    if friendly_dist < float('inf'):
        score += max(0, 10 - friendly_dist) * knobs["friendlyDistance"]

    # Distance to opponent pegs - uses knobs
    opponent_dist = min_distance_to_pegs(row, col, opponent_pegs)
    if opponent_dist < float('inf'):
        score += max(0, 10 - opponent_dist) * knobs["opponentDistance"]

    # Goal distance - uses knobs
    if player == "red":
        goal_distance = min(row, board_size - 1 - row)
    else:
        goal_distance = min(col, board_size - 1 - col)
    goal_bonus = max(0, 12 - goal_distance) * knobs["goalDistance"]
    if knobs.get("training_goal_isolation_scale", 1.0) != 1.0:
        if friendly_connections == 0 and friendly_dist == float('inf'):
            goal_bonus *= float(knobs.get("training_goal_isolation_scale", 1.0))
    score += goal_bonus

    # Center bias - uses knobs
    center = (board_size - 1) / 2
    center_dist = abs(row - center) + abs(col - center)
    score += max(0, 16 - center_dist) * knobs["centerBias"]

    # Isolated move bonus
    if friendly_dist == float('inf') and opponent_dist == float('inf'):
        score += knobs["isolated"]
        isolated_penalty = knobs.get("training_isolated_penalty", 0.0)
        if isolated_penalty:
            score -= isolated_penalty

    # Training-only: penalize orthogonal clumping (adjacent non-bridge pegs).
    adjacent_penalty = knobs.get("adjacentPenalty", 0.0)
    if adjacent_penalty:
        adjacent_count = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r, c = row + dr, col + dc
            if r < 0 or r >= board_size or c < 0 or c >= board_size:
                continue
            if state.pegs.get((r, c)) == player:
                adjacent_count += 1
        if adjacent_count:
            score -= adjacent_penalty * adjacent_count

    # Training-only: push toward missing goal edges (reduce one-side clustering).
    edge_push_weight = knobs.get("edgePushWeight", 0.0)
    edge_push_max = knobs.get("edgePushMax", 0.0)
    decay_after_both = knobs.get("training_edge_push_decay_after_both", 1)
    second_edge_scale = knobs.get("training_second_edge_push_scale", 1.0)
    opening_plies = int(knobs.get("training_opening_center_plies", 0))
    opening_center_mult = float(knobs.get("training_opening_center_bias_mult", 1.0))
    opening_goal_scale = float(knobs.get("training_opening_goal_distance_scale", 1.0))
    opening_edge_scale = float(knobs.get("training_opening_edge_push_scale", 1.0))
    opening_missing_scale = float(knobs.get("training_opening_missing_half_scale", 1.0))
    edge_push_ramp_plies = int(knobs.get("training_edge_push_ramp_plies", 0))
    base_missing_edge_penalty = knobs.get("missingEdgeHalfPenalty", 0.0)
    if opening_plies > 0:
        turn = len(state.move_history) + 1
        if turn <= opening_plies:
            if opening_center_mult != 1.0:
                score += max(0, 16 - center_dist) * knobs["centerBias"] * (opening_center_mult - 1.0)
            if opening_goal_scale != 1.0:
                score += goal_bonus * (opening_goal_scale - 1.0)
            edge_push_weight *= opening_edge_scale
            missing_edge_penalty = base_missing_edge_penalty * opening_missing_scale
        else:
            missing_edge_penalty = base_missing_edge_penalty
            if edge_push_ramp_plies > 0:
                ramp = min(1.0, (turn - opening_plies) / edge_push_ramp_plies)
                edge_push_weight *= (opening_edge_scale + (1.0 - opening_edge_scale) * ramp)
                missing_edge_penalty *= (opening_missing_scale + (1.0 - opening_missing_scale) * ramp)
    else:
        missing_edge_penalty = base_missing_edge_penalty
    if friendly_metrics:
        if player == "red":
            touches_a = friendly_metrics.get("touches_top")
            touches_b = friendly_metrics.get("touches_bottom")
        else:
            touches_a = friendly_metrics.get("touches_left")
            touches_b = friendly_metrics.get("touches_right")

        if decay_after_both and touches_a and touches_b:
            edge_push_weight = 0.0
        elif second_edge_scale != 1.0 and (touches_a ^ touches_b):
            edge_push_weight *= second_edge_scale
    if edge_push_weight and edge_push_max and friendly_metrics:
        if player == "red":
            if not friendly_metrics.get("touches_top"):
                score += max(0.0, edge_push_max - row) * edge_push_weight
            if not friendly_metrics.get("touches_bottom"):
                score += max(0.0, edge_push_max - (board_size - 1 - row)) * edge_push_weight
        else:
            if not friendly_metrics.get("touches_left"):
                score += max(0.0, edge_push_max - col) * edge_push_weight
            if not friendly_metrics.get("touches_right"):
                score += max(0.0, edge_push_max - (board_size - 1 - col)) * edge_push_weight

    # Training-only: penalize staying on the wrong half when a goal edge is missing.
    # missing_edge_penalty is set above to allow opening-phase scaling
    if friendly_metrics:
        if decay_after_both and touches_a and touches_b:
            missing_edge_penalty = 0.0
        elif second_edge_scale != 1.0 and (touches_a ^ touches_b):
            missing_edge_penalty *= second_edge_scale
    if missing_edge_penalty and friendly_metrics:
        center = (board_size - 1) / 2
        if player == "red":
            if not friendly_metrics.get("touches_top") and row > center:
                score -= missing_edge_penalty * ((row - center) / center)
            if not friendly_metrics.get("touches_bottom") and row < center:
                score -= missing_edge_penalty * ((center - row) / center)
        else:
            if not friendly_metrics.get("touches_left") and col > center:
                score -= missing_edge_penalty * ((col - center) / center)
            if not friendly_metrics.get("touches_right") and col < center:
                score -= missing_edge_penalty * ((center - col) / center)

    # Opponent urgency check - use compute_opponent_urgent to match JS
    opponent_urgent = compute_opponent_urgent(opponent, opponent_metrics, board_size)

    # Chain proximity bonus (opponent's largest component)
    if opponent_metrics and opponent_metrics["largest_component"]:
        dist_to_chain = distance_to_component(row, col, opponent_metrics["largest_component"])
        bonus = max(0, 12 - dist_to_chain) * (30 if opponent_urgent else 15)
        score += bonus

    # Frontier/connector proximity
    if opponent_frontier:
        opp_frontier_list = opponent_frontier.get("frontier", [])
        if opp_frontier_list:
            dist_to_frontier = min_distance_to_pegs(row, col, opp_frontier_list)
            score += max(0, 10 - dist_to_frontier) * (35 if opponent_urgent else 16)
            if dist_to_frontier == 0:
                score += 550 if opponent_urgent else 220

        opp_connectors = opponent_frontier.get("connectors", [])
        if opp_connectors:
            dist_to_connector = min_distance_to_pegs(row, col, opp_connectors)
            score += max(0, 8 - dist_to_connector) * (55 if opponent_urgent else 30)
            if dist_to_connector == 0:
                score += 700 if opponent_urgent else 320

        # Trailing penalty (JS search.js lines 1132-1139)
        opp_trailing = opponent_frontier.get("trailing", [])
        if opp_trailing:
            dist_trailing = min_distance_to_pegs(row, col, opp_trailing)
            penalty = max(0, 6 - dist_trailing) * 6
            if penalty > 0:
                score -= penalty

    # Friendly connector targeting - use precomputed connector targets from score_moves
    # (connector_targets passed via friendly_frontier["connector_targets"] to avoid recomputing)
    connector_targets = friendly_frontier.get("connector_targets")
    if connector_targets and f"{row}:{col}" in connector_targets:
        score += knobs.get("connectorTargetBonus", 500)

    # Opponent connector blocking (JS: opponentConnectorSet from computeConnectorTargets)
    # NOTE: Use connector_targets (bounding box adjacent cells), NOT connectors (knight-move frontier)!
    # This matches JS which uses opponentConnectorTargets from computeConnectorTargets().
    blocked_opponent_connector = False
    if opponent_frontier:
        opp_connector_targets = opponent_frontier.get("connector_targets")
        if opp_connector_targets and f"{row}:{col}" in opp_connector_targets:
            score += knobs["blockBonus"]
            blocked_opponent_connector = True

    # === Apply move and evaluate post-move metrics ===
    from ..game.rules import apply_move as apply_move_fn

    if child_state is None:
        child_state = apply_move_fn(state, row, col)
    bridge_delta = len(child_state.bridges) - len(state.bridges)

    # Use pre-computed components/metrics if provided, else compute them
    if post_components is None:
        post_components = find_connected_components(child_state, player)
    if post_metrics is None:
        post_metrics = component_metrics(child_state, player, post_components)
    new_component_penalty = knobs.get("training_new_component_penalty", 0.0)
    if new_component_penalty:
        pre_count = len(friendly_metrics.get("components", ())) if friendly_metrics else 0
        post_count = len(post_components)
        if post_count > pre_count:
            score -= (post_count - pre_count) * new_component_penalty

    # Training-only: reward creating bridges and growing largest component
    bridge_bonus = knobs.get("training_bridge_bonus", 0.0)
    growth_bonus = knobs.get("training_component_growth_bonus", 0.0)
    touches_both_now = (player == "red" and post_metrics["touches_top"] and post_metrics["touches_bottom"]) or \
        (player == "black" and post_metrics["touches_left"] and post_metrics["touches_right"])
    if touches_both_now:
        bridge_bonus *= knobs.get("training_bridge_bonus_after_both_scale", 1.0)
        growth_bonus *= knobs.get("training_growth_bonus_after_both_scale", 1.0)
    if bridge_bonus and bridge_delta > 0:
        score += bridge_delta * bridge_bonus
    if growth_bonus:
        pre_size = len(friendly_metrics.get("largest_component", ())) if friendly_metrics else 0
        post_size = len(post_metrics.get("largest_component", ())) if post_metrics else 0
        if post_size > pre_size:
            score += (post_size - pre_size) * growth_bonus
        elif touches_both_now and bridge_delta > 0:
            redundant_penalty = knobs.get("training_redundant_bridge_penalty", 0.0)
            if redundant_penalty:
                score -= redundant_penalty

    # Threat reduction calculation (JS search.js lines 1147-1161)
    # Measures how much this move reduces opponent's connectivity
    # NOTE: Opponent components are invariant under our move (opponent pegs+bridges unchanged).
    # We reuse the passed-in opponent_components instead of recomputing.
    move_count = len(state.move_history)
    if opponent_threat_before > 0 and move_count > 1:
        if opponent_components is None:
            opponent_components = find_connected_components(state, opponent)  # fallback
        threat_after = evaluate_connected_paths(child_state, opponent, knobs, opponent_components)
        threat_reduction = opponent_threat_before - threat_after
        if threat_reduction > 0:
            # Bonus for reducing opponent's connectivity (JS: threatReduction * 140)
            score += threat_reduction * THREAT_REDUCTION_MULT
        else:
            # Penalty for not reducing threat (JS: opponentUrgent ? 600 : 250)
            penalty = NO_THREAT_URGENT if opponent_urgent else NO_THREAT_NORMAL
            score -= penalty

    # Span gain calculation
    if friendly_metrics and post_metrics:
        # Training-only: reward overall span growth across all pegs (not just largest component).
        global_span_weight = knobs.get("globalSpanGainWeight", 0.0)
        if global_span_weight:
            if player == "red":
                before_min = friendly_metrics.get("min_row")
                before_max = friendly_metrics.get("max_row")
                after_min = post_metrics.get("min_row")
                after_max = post_metrics.get("max_row")
            else:
                before_min = friendly_metrics.get("min_col")
                before_max = friendly_metrics.get("max_col")
                after_min = post_metrics.get("min_col")
                after_max = post_metrics.get("max_col")
            if before_min is not None and before_max is not None and after_min is not None and after_max is not None:
                before_span = before_max - before_min
                after_span = after_max - after_min
                if after_span > before_span:
                    score += (after_span - before_span) * global_span_weight

        if player == "red":
            span_before = friendly_metrics["max_row_span"]
            span_after = post_metrics["max_row_span"]
        else:
            span_before = friendly_metrics["max_col_span"]
            span_after = post_metrics["max_col_span"]
        span_gain = span_after - span_before

        # Gap calculation - use LARGEST component bounds, matching JS behavior
        # JS computes friendlyMinR/friendlyMaxR from friendlyMetrics.largestComponent,
        # then falls back to componentMetrics.minRow/maxRow if largest is empty.
        # This prevents isolated pegs from affecting gap calculations incorrectly.
        board_limit = board_size - 1

        # Compute pre-move bounds from LARGEST component (like JS lines 1191-1201)
        friendly_largest = friendly_metrics.get("largest_component", ())
        if friendly_largest:
            friendly_rows = [r for r, c in friendly_largest]
            friendly_cols = [c for r, c in friendly_largest]
            friendly_min_r = min(friendly_rows)
            friendly_max_r = max(friendly_rows)
            friendly_min_c = min(friendly_cols)
            friendly_max_c = max(friendly_cols)
        else:
            friendly_min_r = friendly_max_r = None
            friendly_min_c = friendly_max_c = None

        # Compute post-move bounds from LARGEST component (like JS lines 1176-1189)
        post_largest_for_gap = post_metrics.get("largest_component", ())
        if post_largest_for_gap:
            post_rows = [r for r, c in post_largest_for_gap]
            post_cols = [c for r, c in post_largest_for_gap]
            post_min_r_gap = min(post_rows)
            post_max_r_gap = max(post_rows)
            post_min_c_gap = min(post_cols)
            post_max_c_gap = max(post_cols)
        else:
            post_min_r_gap = post_max_r_gap = None
            post_min_c_gap = post_max_c_gap = None

        # Compute gap using largest-component bounds with fallback to all-component bounds
        if player == "red":
            prev_min = friendly_min_r if friendly_min_r is not None else (friendly_metrics.get("min_row") or board_limit)
            prev_max = friendly_max_r if friendly_max_r is not None else (friendly_metrics.get("max_row") or 0)
            post_min = post_min_r_gap if post_min_r_gap is not None else (post_metrics.get("min_row") or board_limit)
            post_max = post_max_r_gap if post_max_r_gap is not None else (post_metrics.get("max_row") or 0)
        else:
            prev_min = friendly_min_c if friendly_min_c is not None else (friendly_metrics.get("min_col") or board_limit)
            prev_max = friendly_max_c if friendly_max_c is not None else (friendly_metrics.get("max_col") or 0)
            post_min = post_min_c_gap if post_min_c_gap is not None else (post_metrics.get("min_col") or board_limit)
            post_max = post_max_c_gap if post_max_c_gap is not None else (post_metrics.get("max_col") or 0)

        gap_before = prev_min + (board_limit - prev_max)
        gap_after = post_min + (board_limit - post_max)
        gap_improvement = gap_before - gap_after

        # Training-only: reward largest-component progress toward missing edges
        edge_progress_weight = knobs.get("training_edge_progress_weight", 0.0)
        if edge_progress_weight:
            min_comp_for_largest = int(knobs.get("training_edge_progress_min_component", 0))
            use_overall = min_comp_for_largest and len(friendly_largest) < min_comp_for_largest
            if player == "red":
                prev_min = friendly_metrics.get("min_row") if use_overall else friendly_min_r
                prev_max = friendly_metrics.get("max_row") if use_overall else friendly_max_r
                post_min = post_metrics.get("min_row") if use_overall else post_min_r_gap
                post_max = post_metrics.get("max_row") if use_overall else post_max_r_gap
                if not friendly_metrics.get("touches_top") and prev_min is not None and post_min is not None:
                    score += max(0, prev_min - post_min) * edge_progress_weight
                if not friendly_metrics.get("touches_bottom") and prev_max is not None and post_max is not None:
                    score += max(0, post_max - prev_max) * edge_progress_weight
            else:
                prev_min = friendly_metrics.get("min_col") if use_overall else friendly_min_c
                prev_max = friendly_metrics.get("max_col") if use_overall else friendly_max_c
                post_min = post_metrics.get("min_col") if use_overall else post_min_c_gap
                post_max = post_metrics.get("max_col") if use_overall else post_max_c_gap
                if not friendly_metrics.get("touches_left") and prev_min is not None and post_min is not None:
                    score += max(0, prev_min - post_min) * edge_progress_weight
                if not friendly_metrics.get("touches_right") and prev_max is not None and post_max is not None:
                    score += max(0, post_max - prev_max) * edge_progress_weight

        touches_both = (player == "red" and post_metrics["touches_top"] and post_metrics["touches_bottom"]) or \
                       (player == "black" and post_metrics["touches_left"] and post_metrics["touches_right"])
        near_finish = gap_after <= knobs["finishThreshold"]

        # First edge touch bonus
        edge_requires_bridge = knobs.get("training_edge_touch_requires_bridge", 0)
        edge_ok = True
        if edge_requires_bridge:
            edge_ok = bridge_delta > 0 or friendly_connections > 0
        min_comp = int(knobs.get("training_edge_touch_min_component", 0))
        if min_comp > 0:
            edge_ok = edge_ok and len(post_metrics.get("largest_component", ())) >= min_comp

        if player == "black":
            new_left = post_metrics["touches_left"] and not friendly_metrics["touches_left"]
            new_right = post_metrics["touches_right"] and not friendly_metrics["touches_right"]
            if (new_left or new_right) and edge_ok:
                score += knobs["firstEdgeBlack"]
            had_both = friendly_metrics["touches_left"] and friendly_metrics["touches_right"]
            has_both = post_metrics["touches_left"] and post_metrics["touches_right"]
        else:
            new_top = post_metrics["touches_top"] and not friendly_metrics["touches_top"]
            new_bottom = post_metrics["touches_bottom"] and not friendly_metrics["touches_bottom"]
            if (new_top or new_bottom) and edge_ok:
                score += knobs["firstEdgeRed"]
            had_both = friendly_metrics["touches_top"] and friendly_metrics["touches_bottom"]
            has_both = post_metrics["touches_top"] and post_metrics["touches_bottom"]

        if has_both and not had_both and edge_ok:
            score += knobs.get("training_second_edge_bonus", 0.0)

        # Double coverage bonus (JS search.js lines 1313-1350)
        # Requires largest component to actually span both edges
        post_largest = post_metrics.get("largest_component", ())
        if post_largest:
            # Get bounding box of largest component
            post_min_r = min(r for r, c in post_largest)
            post_max_r = max(r for r, c in post_largest)
            post_min_c = min(c for r, c in post_largest)
            post_max_c = max(c for r, c in post_largest)
        else:
            post_min_r = post_max_r = post_min_c = post_max_c = 0

        if player == "black":
            had_both = friendly_metrics["touches_left"] and friendly_metrics["touches_right"]
            has_both = post_metrics["touches_left"] and post_metrics["touches_right"]
            component_spans_both = post_largest and post_min_c <= 0 and post_max_c >= board_limit
            if has_both and not had_both and component_spans_both:
                score += knobs["doubleCoverageBase"] * knobs["blackDoubleCoverageScale"]
        else:
            had_both = friendly_metrics["touches_top"] and friendly_metrics["touches_bottom"]
            has_both = post_metrics["touches_top"] and post_metrics["touches_bottom"]
            component_spans_both = post_largest and post_min_r <= 0 and post_max_r >= board_limit
            if has_both and not had_both and component_spans_both:
                score += knobs["doubleCoverageBase"] + knobs["redDoubleCoverageBonus"]

        # Span gain bonus
        if span_gain > 0:
            multiplier = knobs["spanGainBase"]
            if player == "black":
                multiplier *= knobs["blackSpanGainMultiplier"]
            else:
                multiplier *= knobs["redSpanGainMultiplier"]
            score += span_gain * multiplier

        # Gap reduction bonus
        if gap_improvement > 0:
            gap_mult = knobs["gapDecay"]
            if player == "red":
                gap_mult *= knobs.get("redGapDecayMultiplier", 1.0)
            score += gap_improvement * gap_mult
        else:
            # Training-only: penalize gap regression after both edges
            if knobs.get("training_retreat_gap_penalty", 0.0):
                requires_both = knobs.get("training_retreat_requires_both", 1)
                if not requires_both or touches_both:
                    score -= (-gap_improvement) * knobs["training_retreat_gap_penalty"]

        # Training-only: ladder alignment reward (favor straight chain extension)
        ladder_bonus = knobs.get("training_ladder_bonus", 0.0)
        if ladder_bonus:
            requires_progress = knobs.get("training_ladder_requires_progress", 1)
            if not requires_progress or span_gain > 0 or gap_improvement > 0:
                if bridge_delta > 0 or friendly_connections > 0:
                    post_largest = post_metrics.get("largest_component", ())
                    if post_largest:
                        if player == "red":
                            cols = [c for r, c in post_largest]
                            center_line = sum(cols) / len(cols)
                            deviation = abs(col - center_line)
                        else:
                            rows = [r for r, c in post_largest]
                            center_line = sum(rows) / len(rows)
                            deviation = abs(row - center_line)
                        max_dev = float(knobs.get("training_ladder_max_dev", 0.0))
                        if max_dev > 0.0 and deviation <= max_dev:
                            score += ladder_bonus * (1.0 - (deviation / max_dev))

        # Training-only: penalize overall span shrink after both edges
        span_penalty = knobs.get("training_retreat_span_penalty", 0.0)
        if span_penalty:
            requires_both = knobs.get("training_retreat_requires_both", 1)
            if not requires_both or touches_both:
                if player == "red":
                    before_min = friendly_metrics.get("min_row")
                    before_max = friendly_metrics.get("max_row")
                    after_min = post_metrics.get("min_row")
                    after_max = post_metrics.get("max_row")
                else:
                    before_min = friendly_metrics.get("min_col")
                    before_max = friendly_metrics.get("max_col")
                    after_min = post_metrics.get("min_col")
                    after_max = post_metrics.get("max_col")
                if None not in (before_min, before_max, after_min, after_max):
                    before_span = before_max - before_min
                    after_span = after_max - after_min
                    if after_span < before_span:
                        score -= (before_span - after_span) * span_penalty

        # Largest component span complete bonus (JS search.js lines 1383-1411)
        # Big bonus when largest component nearly spans both edges
        if post_largest:
            lc_touches_top = post_min_r <= 1
            lc_touches_bottom = post_max_r >= board_limit - 1
            lc_touches_left = post_min_c <= 1
            lc_touches_right = post_max_c >= board_limit - 1

            red_spans = lc_touches_top and lc_touches_bottom
            black_spans = lc_touches_left and lc_touches_right

            if (player == "red" and red_spans) or (player == "black" and black_spans):
                span_complete_bonus = knobs["finishBonusBase"] * 2
                if player == "black":
                    span_complete_bonus *= knobs["blackFinishScaleMultiplier"]
                score += span_complete_bonus

        progress_made = span_gain > 0 or gap_improvement > 0

        # Sealed lane detection - use cache for performance
        if sealed_lane_cache is not None:
            from .sealed_lane import check_sealed_lane
            # Get touches for post-move state
            if player == "red":
                touches_tl = post_metrics["touches_top"]
                touches_br = post_metrics["touches_bottom"]
            else:
                touches_tl = post_metrics["touches_left"]
                touches_br = post_metrics["touches_right"]
            lane_open = check_sealed_lane(
                child_state, player, post_metrics["largest_component"],
                touches_tl, touches_br, sealed_lane_cache
            )
        else:
            # Fallback: assume lane is open (optimistic but fast)
            lane_open = True

        if not lane_open:
            sealed_penalty = knobs.get("training_sealed_lane_penalty", 0.0)
            if sealed_penalty and (touches_both or near_finish or not progress_made):
                score -= sealed_penalty
        if (touches_both or near_finish) and lane_open:
            finish_scale = max(0, knobs["finishBonusBase"] - gap_after * knobs["finishGapSlope"])

            if progress_made:
                bonus = knobs["connectorBonus"] + finish_scale
                if player == "black":
                    bonus *= knobs["blackFinishScaleMultiplier"]
                if player == "red":
                    bonus += knobs.get("redFinishExtra", 0.0)
                score += bonus
            else:
                penalty = knobs["finishPenaltyBase"] + gap_after * knobs["finishGapSlope"]
                if player == "red":
                    penalty *= knobs["redFinishPenaltyFactor"]
                score -= penalty

        # Defense miss penalty (JS search.js lines 1467-1479)
        # JS checks opponentConnectorSet (from opponentConnectorTargets), NOT opponentConnectors
        # BUG FIX: Was checking "connectors" (frontier pegs), should check "connector_targets"
        if opponent_frontier and not blocked_opponent_connector and not touches_both:
            opp_connector_targets = opponent_frontier.get("connector_targets")
            if opp_connector_targets and move_count > 1:
                score -= knobs["missPenalty"] * (1.5 if opponent_urgent else 1.0)

        # Opponent span reduction (JS search.js lines 1482-1505)
        # NOTE: Opponent's connected components are INVARIANT under our move - opponent's
        # pegs and bridges don't change when we place our peg. So opponent_post_metrics
        # equals opponent_metrics, and span_reduction is always 0 unless opponent moved.
        # We keep the logic structure but avoid redundant find_connected_components call.
        if opponent_metrics:
            # Reuse opponent_metrics since it's invariant (opponent didn't move)
            opponent_post_metrics = opponent_metrics

            if opponent == "red":
                opp_span_before = opponent_metrics.get("max_row_span", 0)
                opp_span_after = opponent_post_metrics.get("max_row_span", 0)
            else:
                opp_span_before = opponent_metrics.get("max_col_span", 0)
                opp_span_after = opponent_post_metrics.get("max_col_span", 0)

            span_reduction = opp_span_before - opp_span_after
            if span_reduction > 0:
                # JS: spanReduction * 120
                score += span_reduction * SPAN_REDUCTION_MULT

            # Penalty for not reducing urgent opponent's span (JS: -400)
            if opponent_urgent and span_reduction <= 0:
                score -= NO_SPAN_REDUCTION_PENALTY

            # Span upgrade penalty - if opponent newly spans both edges (JS: -500)
            if opponent == "black":
                opp_had_both = opponent_metrics.get("touches_left", False) and opponent_metrics.get("touches_right", False)
                opp_has_both = opponent_post_metrics.get("touches_left", False) and opponent_post_metrics.get("touches_right", False)
                if opp_has_both and not opp_had_both:
                    score -= SPAN_UPGRADE_PENALTY
            else:
                opp_had_both = opponent_metrics.get("touches_top", False) and opponent_metrics.get("touches_bottom", False)
                opp_has_both = opponent_post_metrics.get("touches_top", False) and opponent_post_metrics.get("touches_bottom", False)
                if opp_has_both and not opp_had_both:
                    score -= SPAN_UPGRADE_PENALTY

    # Opponent-specific defensive biases (JS search.js lines 1534-1575)
    # When opponent is Red touching one edge, bias moves toward blocking the other edge
    if opponent_metrics and opponent == "red":
        if opponent_metrics.get("touches_bottom") and not opponent_metrics.get("touches_top"):
            # Red touching bottom but not top - bias toward top (lower row numbers)
            # JS: topBias = max(0, boardSize - row) * 12
            top_bias = max(0, board_size - row) * DEFENSIVE_BIAS_MULT
            score += top_bias
            if opponent_metrics.get("min_row") is not None:
                opp_min_row = opponent_metrics["min_row"]
                # Bonus for being above opponent's min row (JS: * 150)
                above_bonus = max(0, opp_min_row - row) * DEFENSIVE_POSITION_BONUS
                score += above_bonus
                # Penalty for being below opponent's min row (JS: * 90)
                below_penalty = max(0, row - opp_min_row) * DEFENSIVE_POSITION_PENALTY
                score -= below_penalty
        elif opponent_metrics.get("touches_top") and not opponent_metrics.get("touches_bottom"):
            # Red touching top but not bottom - bias toward bottom (higher row numbers)
            # JS: bottomBias = max(0, row) * 12
            bottom_bias = max(0, row) * DEFENSIVE_BIAS_MULT
            score += bottom_bias
            if opponent_metrics.get("max_row") is not None:
                opp_max_row = opponent_metrics["max_row"]
                # Bonus for being below opponent's max row (JS: * 150)
                below_bonus = max(0, row - opp_max_row) * DEFENSIVE_POSITION_BONUS
                score += below_bonus
                # Penalty for being above opponent's max row (JS: * 90)
                above_penalty = max(0, opp_max_row - row) * DEFENSIVE_POSITION_PENALTY
                score -= above_penalty

    # Global adjustments (JS search.js lines 1592-1639)
    # Red base bonus
    if player == "red" and knobs.get("redBaseBonus", 0) != 0:
        score += knobs["redBaseBonus"]

    # Black base penalty
    if player == "black" and knobs.get("blackBasePenalty", 0) != 0:
        score -= knobs["blackBasePenalty"]

    # Red global multiplier
    if player == "red" and knobs.get("redGlobalMultiplier", 1.0) != 1.0:
        delta = score * (knobs["redGlobalMultiplier"] - 1)
        score += delta

    # Black global scale
    if player == "black" and knobs.get("blackGlobalScale", 1.0) != 1.0:
        delta = score * (knobs["blackGlobalScale"] - 1)
        score += delta

    # Late game pressure penalty
    late_start = knobs.get("lateGameStart", 60.0)
    late_pressure = knobs.get("lateGamePressure", 0.0)
    if late_pressure > 0:
        move_count = len(state.move_history)
        late_turns = move_count + 1 - late_start
        if late_turns > 0:
            score -= late_turns * late_pressure

    return score


def score_moves(
    state: GameState,
    moves: Iterable[Tuple[int, int]],
    player: Optional[str] = None,
    knobs: Optional[Dict[str, float]] = None,
    value_model: Optional["ValueModel"] = None,
    sealed_lane_cache: Optional["SealedLaneLRU"] = None,
) -> List[Tuple[Tuple[int, int], float]]:
    """Return (move, score) pairs for all candidate moves using movePriority.

    Uses the full JS search.js movePriority scoring with REWARDS from knobs.
    Optionally applies value model adjustment if provided.

    Args:
        state: Current game state
        moves: Candidate moves to score
        player: Player to score for (default: state.to_move)
        knobs: Tunable parameters (default: DEFAULT_KNOBS)
        value_model: Optional value model for ML-based scoring adjustment
        sealed_lane_cache: Optional LRU cache for sealed lane detection

    Returns:
        List of (move, score) pairs, sorted by score descending
    """
    if player is None:
        player = state.to_move

    k = get_knobs(knobs)
    opponent = "red" if player == "black" else "black"

    # Precompute components once, then pass through to avoid redundant lookups
    friendly_pegs = [(r, c) for (r, c), p in state.pegs.items() if p == player]
    opponent_pegs = [(r, c) for (r, c), p in state.pegs.items() if p == opponent]

    friendly_components = find_connected_components(state, player)
    opponent_components = find_connected_components(state, opponent)

    friendly_metrics = component_metrics(state, player, friendly_components)
    opponent_metrics = component_metrics(state, opponent, opponent_components)

    friendly_frontier = compute_frontier(state, player, friendly_metrics)
    opponent_frontier = compute_frontier(state, opponent, opponent_metrics)

    # Precompute connector targets once (JS does this in getBestMove, not per-move)
    friendly_connector_targets = compute_connector_targets(state, player, friendly_metrics)
    friendly_frontier["connector_targets"] = friendly_connector_targets

    # Also compute opponent connector targets for blockBonus check (JS: opponentConnectorTargets)
    opponent_connector_targets = compute_connector_targets(state, opponent, opponent_metrics)
    opponent_frontier["connector_targets"] = opponent_connector_targets

    # Precompute opponent threat for threat reduction calculation (JS search.js line 551)
    opponent_threat_before = evaluate_connected_paths(state, opponent, k, opponent_components)

    # Try to get value model if not provided
    if value_model is None:
        try:
            from .value_model import get_cached_model
            value_model = get_cached_model()
        except ImportError:
            pass

    from ..game.rules import apply_move as apply_move_fn

    scored = []
    for row, col in moves:
        # Compute child state once if value_model needs it
        child_state = None
        if value_model is not None:
            child_state = apply_move_fn(state, row, col)

        score = move_priority(
            state, row, col, player, k,
            friendly_pegs, opponent_pegs,
            friendly_metrics, opponent_metrics,
            friendly_frontier, opponent_frontier,
            sealed_lane_cache,
            opponent_threat_before,
            child_state=child_state,  # Reuse if computed
            opponent_components=opponent_components,  # Invariant under our move
        )

        # Apply value model adjustment if available
        if value_model is not None and child_state is not None:
            features = extract_features(child_state, player)
            # Add context features that JS uses
            features["turn"] = len(state.move_history) + 1
            features["player"] = 1.0 if player == "red" else 0.0
            features["playerPegCount"] = len(friendly_pegs) + 1
            features["opponentPegCount"] = len(opponent_pegs)

            evaluation = value_model.evaluate(features)
            adjustment = evaluation.get("adjustment")
            if adjustment is not None:
                score += adjustment

        scored.append(((row, col), score))

    # Sort by score descending
    scored.sort(key=lambda x: -x[1])
    return scored


Move = Tuple[int, int]


# Default: run value model on top 50 candidates (not all ~400)
# This reduces feature extraction from O(N) to O(50), giving ~10x speedup
VALUE_MODEL_TOP_K = 50


@overload
def score_moves_batch(
    state: GameState,
    moves: Iterable[Move],
    player: Optional[str] = ...,
    knobs: Optional[Dict[str, float]] = ...,
    sealed_lane_cache: Optional["SealedLaneLRU"] = ...,
    *,
    return_children: Literal[False] = False,
    value_model_top_k: int = ...,
) -> List[Tuple[Move, float]]: ...


@overload
def score_moves_batch(
    state: GameState,
    moves: Iterable[Move],
    player: Optional[str] = ...,
    knobs: Optional[Dict[str, float]] = ...,
    sealed_lane_cache: Optional["SealedLaneLRU"] = ...,
    *,
    return_children: Literal[True],
    value_model_top_k: int = ...,
) -> List[Tuple[Move, float, GameState]]: ...


def score_moves_batch(
    state: GameState,
    moves: Iterable[Move],
    player: Optional[str] = None,
    knobs: Optional[Dict[str, float]] = None,
    sealed_lane_cache: Optional["SealedLaneLRU"] = None,
    *,
    return_children: bool = False,
    value_model_top_k: int = VALUE_MODEL_TOP_K,
) -> Union[List[Tuple[Move, float]], List[Tuple[Move, float, GameState]]]:
    """GPU-accelerated batch move scoring with two-phase optimization.

    Phase 1: Score ALL moves with move_priority() (fast, ~24ms for 500 moves)
    Phase 2: Extract features + value model for TOP-K candidates only (expensive)

    This two-phase approach reduces feature extraction from O(N) to O(K),
    giving ~10x speedup when N=500 and K=50.

    Args:
        state: Current game state
        moves: Candidate moves to score
        player: Player to score for (default: state.to_move)
        knobs: Tunable parameters (default: DEFAULT_KNOBS)
        sealed_lane_cache: Optional LRU cache for sealed lane detection
        return_children: If True, return (move, score, child_state) tuples
        value_model_top_k: Number of top candidates to run through value model
                          (default: 50). Set to 0 to disable value model.

    Returns:
        List of (move, score) pairs sorted by score descending, or
        List of (move, score, child_state) triples if return_children=True
    """
    from ..game.rules import apply_move as apply_move_fn
    from .batch_eval import batch_extract_features_cached, get_batch_value_model

    if player is None:
        player = state.to_move

    k = get_knobs(knobs)
    opponent = "red" if player == "black" else "black"

    # Precompute components once, then pass through
    friendly_pegs = [(r, c) for (r, c), p in state.pegs.items() if p == player]
    opponent_pegs = [(r, c) for (r, c), p in state.pegs.items() if p == opponent]

    friendly_components = find_connected_components(state, player)
    opponent_components = find_connected_components(state, opponent)

    friendly_metrics = component_metrics(state, player, friendly_components)
    opponent_metrics = component_metrics(state, opponent, opponent_components)

    friendly_frontier = compute_frontier(state, player, friendly_metrics)
    opponent_frontier = compute_frontier(state, opponent, opponent_metrics)

    # NOTE: JS uses REWARDS.edge.radius (5) for connector target radius
    edge_radius = int(k.get("edgeRadius", 5))

    friendly_connector_targets = compute_connector_targets(state, player, friendly_metrics, radius=edge_radius)
    friendly_frontier["connector_targets"] = friendly_connector_targets

    # Also compute opponent connector targets for blockBonus check (JS: opponentConnectorTargets)
    opponent_connector_targets = compute_connector_targets(state, opponent, opponent_metrics, radius=edge_radius)
    opponent_frontier["connector_targets"] = opponent_connector_targets

    opponent_threat_before = evaluate_connected_paths(state, opponent, k, opponent_components)

    # Convert moves to list for indexing
    moves_list = list(moves)
    if not moves_list:
        return []

    # Phase 1: Compute base scores (move_priority) for all moves
    # Pre-compute child states and their components in batch to avoid redundant computation
    base_scores: List[float] = []
    child_states: List[GameState] = []
    post_components_list: List[Components] = []
    post_metrics_list: List[Dict] = []

    # First pass: create all child states and compute their components
    for row, col in moves_list:
        child_state = apply_move_fn(state, row, col)
        child_states.append(child_state)
        # Compute player components for child state (needed for span/growth calculations)
        post_comps = find_connected_components(child_state, player)
        post_components_list.append(post_comps)
        post_metrics_list.append(component_metrics(child_state, player, post_comps))

    # Second pass: score moves with pre-computed data
    for i, (row, col) in enumerate(moves_list):
        score = move_priority(
            state, row, col, player, k,
            friendly_pegs, opponent_pegs,
            friendly_metrics, opponent_metrics,
            friendly_frontier, opponent_frontier,
            sealed_lane_cache,
            opponent_threat_before,
            child_state=child_states[i],
            opponent_components=opponent_components,
            post_components=post_components_list[i],
            post_metrics=post_metrics_list[i],
        )
        base_scores.append(score)

    # Phase 2: Value model inference for TOP-K candidates only
    # This is the key optimization - instead of extracting features for all ~500 moves,
    # we only do it for the top candidates based on move_priority scores.
    batch_model = get_batch_value_model() if value_model_top_k > 0 else None

    if batch_model is not None and value_model_top_k > 0:
        # Sort by base score to find top-k candidates
        indexed_scores = sorted(
            enumerate(base_scores),
            key=lambda x: -x[1]
        )

        # Take top-k indices
        top_k = min(value_model_top_k, len(indexed_scores))
        top_indices = [idx for idx, _ in indexed_scores[:top_k]]

        # Extract features ONLY for top-k child states
        # Use cached extraction: opponent features computed once from parent
        top_child_states = [child_states[i] for i in top_indices]
        base_turn = len(state.move_history)
        feature_dicts = batch_extract_features_cached(
            state,  # parent state for opponent feature caching
            top_child_states,
            player,
            base_turn,
            len(friendly_pegs),
            len(opponent_pegs),
        )

        # Batch evaluate top-k
        evaluations = batch_model.batch_evaluate(feature_dicts)

        # Add adjustments to corresponding base scores
        for j, evaluation in enumerate(evaluations):
            original_idx = top_indices[j]
            adjustment = evaluation.get("adjustment")
            if adjustment is not None:
                base_scores[original_idx] += adjustment

    # Build results with correct pairing, then sort by score (x[1])
    if return_children:
        scored3: List[Tuple[Move, float, GameState]] = [
            (moves_list[i], base_scores[i], child_states[i])
            for i in range(len(moves_list))
        ]
        scored3.sort(key=lambda x: -x[1])
        return scored3
    else:
        scored2: List[Tuple[Move, float]] = [
            (moves_list[i], base_scores[i])
            for i in range(len(moves_list))
        ]
        scored2.sort(key=lambda x: -x[1])
        return scored2
