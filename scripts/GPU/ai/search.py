"""TwixT minimax search with alpha-beta pruning.

Ported from assets/js/ai/search.js with simplifications for the tuning pipeline.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..game.rules import apply_move, check_winner, generate_moves
from ..game.state import GameState
from .heuristics import evaluate_position, evaluate_move, score_moves_batch

# Temperature floor to avoid division issues
TEMP_FLOOR = 1e-6


@dataclass
class SearchResult:
    """Result of a search operation."""
    row: int
    col: int
    score: float
    depth: int = 0
    nodes_searched: int = 0
    candidates: List[Dict] = field(default_factory=list)


def deterministic_best(candidates: List[Dict]) -> Dict:
    """Select best move with stable tie-break (lexicographic by row, col).

    Args:
        candidates: List of dicts with 'row', 'col', 'score' keys

    Returns:
        The best candidate dict
    """
    best = candidates[0]
    for c in candidates[1:]:
        if c["score"] > best["score"]:
            best = c
        elif c["score"] == best["score"]:
            # Tie-break: lexicographic by (row, col)
            if (c["row"], c["col"]) < (best["row"], best["col"]):
                best = c
    return best


def fallback_best(
    candidates: List[Dict],
    rng: Optional[random.Random],
    *,
    mode: str,
) -> Dict:
    """Pick a fallback best move when sampling is disabled or degenerate."""
    if mode == "training" and rng is not None:
        best_score = candidates[0]["score"]
        best = [c for c in candidates if c["score"] == best_score]
        return rng.choice(best)
    return deterministic_best(candidates)


def pick_move_from_scores(
    candidates: List[Dict],
    temperature: float,
    rng: random.Random,
    *,
    mode: str,
) -> Dict:
    """Select move using softmax sampling with normalized temperature.

    Uses score normalization so temperature values in [0.1, 1.0] are meaningful:
        p_i ∝ exp((s_i - s_max) / (T * S))
    where S = max_score - min_score (score spread).

    This makes temperature independent of the absolute score scale.

    Args:
        candidates: List of dicts with 'row', 'col', 'score' keys
        temperature: Sampling temperature (0 = deterministic, 1 = high exploration)
        rng: Random number generator for reproducibility

    Returns:
        The selected candidate dict
    """
    SCALE_FLOOR = 1e-6

    # Deterministic mode: use stable tie-break (or random best in training)
    if temperature <= TEMP_FLOOR:
        return fallback_best(candidates, rng, mode=mode)

    # Compute score spread for normalization
    scores = [c["score"] for c in candidates]
    max_score = max(scores)
    min_score = min(scores)
    S = max(max_score - min_score, SCALE_FLOOR)

    # Normalized softmax: exp((s - max) / (T * S))
    T = max(temperature, TEMP_FLOOR)
    exps = [math.exp((s - max_score) / (T * S)) for s in scores]
    Z = sum(exps)

    # Guard against degenerate Z
    if not math.isfinite(Z) or Z <= 0:
        return fallback_best(candidates, rng, mode=mode)

    # Sample from distribution
    r = rng.random()
    cumulative = 0.0
    for i, exp_val in enumerate(exps):
        cumulative += exp_val / Z
        if r <= cumulative:
            return candidates[i]

    # Fallback to last candidate (numerical edge case)
    return candidates[-1]


def temperature_for_ply(ply: int, knobs: Optional[Dict[str, float]]) -> float:
    """Get temperature using smooth exponential decay schedule.

    Formula: T(p) = T_late + (T_early - T_late) * exp(-max(0, p - p0) / tau)

    Where:
    - p = ply number (0-based)
    - p0 = temp_transition_ply (decay starts here)
    - tau = temp_tau (time constant; larger = slower decay)

    Args:
        ply: Current ply number (0-based)
        knobs: Configuration knobs with temperature parameters

    Returns:
        Temperature value for the current ply
    """
    if knobs is None:
        return 0.0  # Deterministic by default

    if knobs.get("deterministic_mode", 0):
        return 0.0

    early = knobs.get("temp_early", 0.9)
    late = knobs.get("temp_late", 0.12)
    p0 = int(knobs.get("temp_transition_ply", 10))
    tau = max(TEMP_FLOOR, knobs.get("temp_tau", 10))

    d = max(0, ply - p0)
    alpha = math.exp(-d / tau)
    return late + (early - late) * alpha


def minimax(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    root_player: str,
    top_n: int = 20,  # Base limit for move ordering (JS: baseLimit)
    knobs: Optional[Dict[str, float]] = None,
    value_model_k: int = 0,  # Disable value model by default in minimax
    root_depth: int = 0,  # Total search depth from root (for depth-dependent limits)
    *,
    player_components: Optional[tuple] = None,
    opponent_components: Optional[tuple] = None,
    player_metrics: Optional[dict] = None,
) -> float:
    """Minimax with alpha-beta pruning.

    Args:
        state: Current game state
        depth: Remaining search depth
        alpha: Alpha value for pruning
        beta: Beta value for pruning
        maximizing: True if maximizing player's turn
        root_player: The player we're evaluating for
        top_n: Base number of top moves to consider (adjusted by depth)
        knobs: Configuration knobs for heuristic evaluation
        value_model_k: Number of top candidates for value model (0 = disabled)
        root_depth: Total search depth from root (for JS-compatible depth-dependent limits)
        player_components: Pre-computed player components (optional)
        opponent_components: Pre-computed opponent components (optional)
        player_metrics: Pre-computed player metrics (optional)

    Returns:
        Evaluation score from root_player's perspective
    """
    # Terminal conditions
    winner = check_winner(state)
    if winner is not None:
        return 10000.0 if winner == root_player else -10000.0

    if depth == 0:
        return evaluate_position(
            state, root_player, knobs,
            player_components=player_components,
            opponent_components=opponent_components,
            player_metrics=player_metrics,
        )

    # Generate and order moves
    moves = generate_moves(state)
    if not moves:
        return evaluate_position(
            state, root_player, knobs,
            player_components=player_components,
            opponent_components=opponent_components,
            player_metrics=player_metrics,
        )

    # Score and order moves for better pruning (batch + return children)
    scored = score_moves_batch(state, moves, knobs=knobs, return_children=True, value_model_top_k=value_model_k)

    # JS-compatible depth-dependent move limit:
    # limit = round((baseLimit * (depth + 1)) / (rootDepth + 1))
    # At root (depth==rootDepth): limit = baseLimit
    # At depth 1 (if rootDepth=2): limit = round(20 * 2 / 3) = 13
    # At depth 0: no moves needed (terminal)
    effective_root = max(1, root_depth) if root_depth > 0 else max(1, depth)
    depth_factor = max(1, depth + 1)
    limit = max(6, min(len(moves), round((top_n * depth_factor) / (effective_root + 1))))

    ordered = scored[:limit]
    del scored  # Free unused children ASAP

    # Pre-compute components for efficient recursive calls
    # Key insight: the non-moving player's components are INVARIANT under the moving player's moves
    from .heuristics import find_connected_components, component_metrics as comp_metrics_fn

    current_player = state.to_move  # Who moves now
    other_player = "red" if current_player == "black" else "black"

    # Get invariant components (player who didn't move)
    # Reuse from parent if available, otherwise compute once
    if current_player == root_player:
        # root_player is moving, opponent components are invariant
        invariant_comps = opponent_components if opponent_components is not None else find_connected_components(state, other_player)
    else:
        # opponent is moving, root_player's components are invariant
        invariant_comps = player_components if player_components is not None else find_connected_components(state, other_player)

    if maximizing:
        max_eval = float("-inf")
        for (row, col), _, child in ordered:
            # Compute the moving player's new components
            child_mover_comps = find_connected_components(child, current_player)

            # Map to root_player perspective for recursive call
            if current_player == root_player:
                next_player_comps = child_mover_comps
                next_opponent_comps = invariant_comps
                next_metrics = comp_metrics_fn(child, root_player, child_mover_comps)
            else:
                next_player_comps = invariant_comps
                next_opponent_comps = child_mover_comps
                next_metrics = comp_metrics_fn(child, root_player, invariant_comps)

            eval_score = minimax(
                child, depth - 1, alpha, beta, False, root_player, top_n, knobs, value_model_k, root_depth,
                player_components=next_player_comps,
                opponent_components=next_opponent_comps,
                player_metrics=next_metrics,
            )
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = float("inf")
        for (row, col), _, child in ordered:
            # Compute the moving player's new components
            child_mover_comps = find_connected_components(child, current_player)

            # Map to root_player perspective for recursive call
            if current_player == root_player:
                next_player_comps = child_mover_comps
                next_opponent_comps = invariant_comps
                next_metrics = comp_metrics_fn(child, root_player, child_mover_comps)
            else:
                next_player_comps = invariant_comps
                next_opponent_comps = child_mover_comps
                next_metrics = comp_metrics_fn(child, root_player, invariant_comps)

            eval_score = minimax(
                child, depth - 1, alpha, beta, True, root_player, top_n, knobs, value_model_k, root_depth,
                player_components=next_player_comps,
                opponent_components=next_opponent_comps,
                player_metrics=next_metrics,
            )
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval


def choose_move(
    state: GameState,
    knobs: Optional[Dict[str, float]] = None,
    *,
    depth: int = 2,
    top_n: int = 20,  # Matches JS orderMoves limit at medium difficulty
    use_value_model: bool = True,  # Set False for pure heuristic comparison
    temperature: float = 0.0,  # 0.0 = deterministic (greedy), >0 = softmax sampling
    rng: Optional[random.Random] = None,  # For reproducible sampling
    mode: str = "debug",  # "training" or "debug"
) -> SearchResult:
    """Choose the best move using minimax search.

    Args:
        state: Current game state
        knobs: Optional configuration knobs (for future use with tuned heuristics)
        depth: Search depth
        top_n: Number of top moves to consider at each level
        use_value_model: Whether to use value model for move scoring
        temperature: Sampling temperature (0 = deterministic, >0 = softmax)
        rng: Random number generator for reproducible sampling

    Returns:
        SearchResult with best move and score
    """
    player = state.to_move
    moves = generate_moves(state)
    value_model_k = 50 if use_value_model else 0

    if not moves:
        return SearchResult(row=0, col=0, score=0.0, depth=depth)

    # Single move - no need to search
    if len(moves) == 1:
        row, col = moves[0]
        return SearchResult(row=row, col=col, score=0.0, depth=depth)

    # Score and order moves (batch + return children)
    scored = score_moves_batch(state, moves, knobs=knobs, return_children=True, value_model_top_k=value_model_k)
    ordered_moves = scored[:top_n]
    del scored  # Free unused children ASAP

    nodes_searched = 0
    candidates = []
    pre_metrics = None
    connector_targets = None
    opponent_metrics = None
    opponent_frontier = None
    opponent_connector_targets = None
    opponent_urgent = False
    opponent_corridor = None
    opponent_pressure = False
    if mode == "training":
        from .heuristics import component_metrics, compute_connector_targets, compute_frontier, compute_opponent_urgent
        pre_metrics = component_metrics(state, player)
        connector_targets = compute_connector_targets(state, player, pre_metrics)
        opponent = "red" if player == "black" else "black"
        opponent_metrics = component_metrics(state, opponent)
        opponent_frontier = compute_frontier(state, opponent, opponent_metrics)
        opponent_connector_targets = compute_connector_targets(state, opponent, opponent_metrics)
        opponent_urgent = compute_opponent_urgent(opponent, opponent_metrics, state.board_size)
        span_threshold = int((knobs or {}).get("training_defense_span_threshold", 0))
        corridor_margin = int((knobs or {}).get("training_defense_corridor_margin", 0))
        if opponent == "red":
            span = opponent_metrics.get("max_row_span", 0)
            min_c = opponent_metrics.get("min_col")
            max_c = opponent_metrics.get("max_col")
            if min_c is not None and max_c is not None:
                c0 = max(0, min_c - corridor_margin)
                c1 = min(state.board_size - 1, max_c + corridor_margin)
                opponent_corridor = (0, state.board_size - 1, c0, c1)
        else:
            span = opponent_metrics.get("max_col_span", 0)
            min_r = opponent_metrics.get("min_row")
            max_r = opponent_metrics.get("max_row")
            if min_r is not None and max_r is not None:
                r0 = max(0, min_r - corridor_margin)
                r1 = min(state.board_size - 1, max_r + corridor_margin)
                opponent_corridor = (r0, r1, 0, state.board_size - 1)
        opponent_pressure = span_threshold > 0 and span >= span_threshold
        if opponent_frontier:
            opp_frontier = opponent_frontier.get("frontier", [])
            opp_connectors = opponent_frontier.get("connectors", [])
            if opp_frontier or opp_connectors:
                opponent_pressure = True
    sealed_lane_cache = None
    if mode == "training":
        try:
            from .sealed_lane import SealedLaneLRU, check_sealed_lane
            sealed_lane_cache = SealedLaneLRU(max_entries=20_000)
        except ImportError:
            sealed_lane_cache = None
    best_move = ordered_moves[0][0]
    best_score = float("-inf")

    # Pre-compute components for ordered candidates to avoid redundant find_connected_components calls
    # This is the key optimization: compute once, reuse for evaluate_position, component_metrics, etc.
    from .heuristics import find_connected_components, component_metrics as comp_metrics_fn
    opponent = "red" if player == "black" else "black"

    # Opponent components are INVARIANT under player's moves (opponent's pegs/bridges don't change)
    opponent_comps = find_connected_components(state, opponent)

    # Pre-compute player components and metrics for each candidate child state
    ordered_post_components = []
    ordered_post_metrics = []
    for (_, _, child) in ordered_moves:
        post_comps = find_connected_components(child, player)
        ordered_post_components.append(post_comps)
        ordered_post_metrics.append(comp_metrics_fn(child, player, post_comps))

    for i, ((row, col), heuristic_score, child) in enumerate(ordered_moves):
        # Use pre-computed components for this candidate
        post_components = ordered_post_components[i]
        post_metrics = ordered_post_metrics[i]
        # child already computed by score_moves_batch
        nodes_searched += 1

        # Check for immediate win
        winner = check_winner(child)
        if winner == player:
            return SearchResult(
                row=row,
                col=col,
                score=10000.0,
                depth=depth,
                nodes_searched=nodes_searched,
                candidates=[{"row": row, "col": col, "score": 10000.0}],
            )

        # Minimax search from opponent's perspective
        if depth > 1:
            eval_score = minimax(
                child,
                depth - 1,
                float("-inf"),
                float("inf"),
                False,
                player,
                top_n,
                knobs,
                value_model_k,
                root_depth=depth,  # Pass root depth for JS-compatible move limits
                player_components=post_components,
                opponent_components=opponent_comps,
                player_metrics=post_metrics,
            )
        else:
            # Use pre-computed components for terminal evaluation
            eval_score = evaluate_position(
                child, player, knobs,
                player_components=post_components,
                opponent_components=opponent_comps,
                player_metrics=post_metrics,
            )

        # Combine minimax score with immediate move evaluation
        # NOTE: JS calls evaluateMove AFTER placePeg, so use child state (post-move)
        immediate_score = evaluate_move(child, row, col, player, knobs)
        # Use pre-computed components for position evaluation
        position_score = evaluate_position(
            child, player, knobs,
            player_components=post_components,
            opponent_components=opponent_comps,
            player_metrics=post_metrics,
        )

        # Near-finish bonus (like JS) - huge bonus when one move from winning
        # IMPORTANT: Use largestComponent bounds, not overall bounds (matches JS)
        # Use pre-computed post_metrics instead of recomputing
        finish_bonus = 0.0
        lc = post_metrics["largest_component"]
        if lc:
            # Calculate bounds from largestComponent only (matching JS getBestMove)
            lc_rows = [r for r, c in lc]
            lc_cols = [c for r, c in lc]
            min_r = min(lc_rows)
            max_r = max(lc_rows)
            min_c = min(lc_cols)
            max_c = max(lc_cols)
            board_size = state.board_size

            touches_top = post_metrics.get("touches_top", False)
            touches_bottom = post_metrics.get("touches_bottom", False)
            touches_left = post_metrics.get("touches_left", False)
            touches_right = post_metrics.get("touches_right", False)

            if player == "red":
                # JS logic: (touchesTop && maxR >= N-2) || (touchesBottom && minR <= 1)
                near_finish = (touches_top and max_r >= board_size - 2) or \
                              (touches_bottom and min_r <= 1)
                if near_finish:
                    finish_bonus = 2500.0  # nearFinishBonus from search.json
            else:
                # JS logic: (touchesLeft && maxC >= N-2) || (touchesRight && minC <= 1)
                near_finish = (touches_left and max_c >= board_size - 2) or \
                              (touches_right and min_c <= 1)
                if near_finish:
                    finish_bonus = 2500.0
        lane_open = True
        if sealed_lane_cache is not None:
            touches_tl = touches_top if player == "red" else touches_left
            touches_br = touches_bottom if player == "red" else touches_right
            player_id = 0 if player == "red" else 1
            lane_open = check_sealed_lane(
                child, player_id, post_metrics["largest_component"],
                touches_tl, touches_br, sealed_lane_cache
            )
        if not lane_open:
            finish_bonus = 0.0

        progress_flag = False
        if pre_metrics:
            if player == "red":
                span_gain = post_metrics.get("max_row_span", 0) - pre_metrics.get("max_row_span", 0)
                gap_before = pre_metrics.get("min_row", 0) + (state.board_size - 1 - pre_metrics.get("max_row", 0))
                gap_after = post_metrics.get("min_row", 0) + (state.board_size - 1 - post_metrics.get("max_row", 0))
            else:
                span_gain = post_metrics.get("max_col_span", 0) - pre_metrics.get("max_col_span", 0)
                gap_before = pre_metrics.get("min_col", 0) + (state.board_size - 1 - pre_metrics.get("max_col", 0))
                gap_after = post_metrics.get("min_col", 0) + (state.board_size - 1 - post_metrics.get("max_col", 0))
            progress_flag = span_gain > 0 or gap_after < gap_before

        is_connector = False
        if connector_targets is not None:
            is_connector = f"{row}:{col}" in connector_targets

        def_block = False
        if opponent_connector_targets is not None and f"{row}:{col}" in opponent_connector_targets:
            def_block = True
        if opponent_frontier:
            opp_frontier = opponent_frontier.get("frontier", [])
            opp_connectors = opponent_frontier.get("connectors", [])
            if not def_block and (row, col) in opp_frontier:
                def_block = True
            if not def_block and (row, col) in opp_connectors:
                def_block = True
        intercept = False
        if opponent_corridor is not None:
            r0, r1, c0, c1 = opponent_corridor
            if r0 <= row <= r1 and c0 <= col <= c1:
                intercept = True

        total_score = eval_score + immediate_score * 5 + position_score * 0.1 + finish_bonus

        candidates.append({
            "row": row,
            "col": col,
            "score": total_score,
            "minimax": eval_score,
            "immediate": immediate_score,
            "position": position_score,
            "heuristic": heuristic_score,
            "lane_open": lane_open,
            "progress": progress_flag,
            "connector": is_connector,
            "defense": def_block,
            "intercept": intercept,
        })

        if total_score > best_score:
            best_score = total_score
            best_move = (row, col)

    # Sort candidates by score (descending)
    candidates.sort(key=lambda x: -x["score"])

    # Select move using guarded sampling or deterministic best
    if temperature > TEMP_FLOOR and rng is not None:
        sample_pool = candidates
        if mode == "training":
            ply = len(state.move_history) + 1
            start_ply = int((knobs or {}).get("training_guard_start_ply", 0))
            if start_ply > 0 and ply >= start_ply:
                frac = float((knobs or {}).get("training_score_delta_frac", 0.0))
                abs_delta = float((knobs or {}).get("training_score_delta_abs", 0.0))
                cap_delta = float((knobs or {}).get("training_score_delta_cap", 0.0))
                max_score = sample_pool[0]["score"]
                if frac > 0.0 or abs_delta > 0.0 or cap_delta > 0.0:
                    band = max(abs(max_score) * frac, abs_delta)
                    if cap_delta > 0.0:
                        band = min(band, cap_delta)
                    cutoff = max_score - band
                    gated = [c for c in sample_pool if c["score"] >= cutoff]
                    if gated:
                        sample_pool = gated
                # Guarded progress filter
                require_progress = bool((knobs or {}).get("training_guard_require_progress", 0))
                allow_connector = bool((knobs or {}).get("training_guard_allow_connector", 0))
                if require_progress:
                    guarded = [
                        c for c in sample_pool
                        if c.get("progress") or (allow_connector and c.get("connector"))
                    ]
                    if guarded:
                        sample_pool = guarded
                # Defensive filter when opponent urgent
                if (opponent_urgent or opponent_pressure) and (knobs or {}).get("training_guard_require_defense", 0):
                    defenders = [c for c in candidates if c.get("defense") or c.get("intercept")]
                    if defenders:
                        sample_pool = defenders
                    else:
                        sample_pool = candidates[:1]
                if (opponent_urgent or opponent_pressure) and (knobs or {}).get("training_force_deterministic_urgent", 0):
                    temperature = 0.0
                # Lane gating
                if any(c.get("lane_open") for c in sample_pool):
                    sample_pool = [c for c in sample_pool if c.get("lane_open")]
                    if (knobs or {}).get("training_force_deterministic_open_lane", 0):
                        temperature = 0.0
                else:
                    sealed_top_k = int((knobs or {}).get("training_sealed_lane_top_k", 0))
                    if sealed_top_k > 0:
                        sample_pool = sample_pool[:sealed_top_k]
                    if sample_pool:
                        sample_pool = sample_pool[:1]
                temp_scale = float((knobs or {}).get("training_midgame_temp_scale", 1.0))
                temperature *= temp_scale
        selected = pick_move_from_scores(sample_pool, temperature, rng, mode=mode)
    else:
        # Deterministic mode: use stable tie-break (or random best in training)
        selected = fallback_best(candidates, rng, mode=mode)

    return SearchResult(
        row=selected["row"],
        col=selected["col"],
        score=selected["score"],
        depth=depth,
        nodes_searched=nodes_searched,
        candidates=candidates,
    )


def get_best_move(
    state: GameState,
    depth: int = 2,
    top_n: int = 12,
    *,
    mode: str = "debug",
) -> Tuple[int, int]:
    """Simple interface to get best move coordinates.

    Args:
        state: Current game state
        depth: Search depth
        top_n: Number of moves to consider

    Returns:
        (row, col) tuple of best move
    """
    result = choose_move(state, depth=depth, top_n=top_n, mode=mode)
    return (result.row, result.col)


def choose_move_hybrid(
    state: GameState,
    knobs: Optional[Dict[str, float]] = None,
    *,
    depth: int = 2,
    top_n: int = 20,
    gpu_model: Optional["MoveRanker"] = None,
    use_heuristic_fallback: bool = True,
    temperature: float = 0.0,
    rng: Optional[random.Random] = None,
    mode: str = "debug",
) -> SearchResult:
    """Hybrid search: GPU model for ordering, heuristics for evaluation.

    Flow:
    1. GPU model scores all candidate moves in ~2ms (if available)
    2. Take top-N for deeper minimax search
    3. Minimax uses heuristic evaluation at leaves (grounded)

    This gives fast initial ordering (GPU) with accurate deep search
    (tried-and-tested heuristics).

    Args:
        state: Current game state
        knobs: Configuration knobs for heuristic evaluation
        depth: Search depth
        top_n: Number of top moves to search deeper
        gpu_model: Trained MoveRanker model (None = use heuristics only)
        use_heuristic_fallback: Fall back to heuristics if GPU fails
        temperature: Sampling temperature (0 = deterministic)
        rng: Random number generator for sampling
        mode: "training" or "debug"

    Returns:
        SearchResult with best move and score
    """
    from ..utils.maybe_mlx import try_import_mlx

    player = state.to_move
    moves = generate_moves(state)

    if not moves:
        return SearchResult(row=0, col=0, score=0.0, depth=depth)

    # Single move - no need to search
    if len(moves) == 1:
        row, col = moves[0]
        return SearchResult(row=row, col=col, score=0.0, depth=depth)

    # Phase 1: Move ordering (GPU or heuristic)
    ordered_moves = None

    if gpu_model is not None:
        _mlx_env = try_import_mlx()
        if _mlx_env.available:
            try:
                from .tensor_repr import state_to_tensor

                mx = _mlx_env.mx
                board_tensor = state_to_tensor(state)
                gpu_logits = gpu_model.score_all_moves(board_tensor, moves)
                mx.eval(gpu_logits)  # Force evaluation

                # Sort by GPU score (ordering only)
                order = mx.argsort(-gpu_logits).tolist()
                ordered_moves = [moves[i] for i in order[:top_n]]

            except Exception as e:
                if not use_heuristic_fallback:
                    raise
                # Fall back to heuristics
                ordered_moves = None

    if ordered_moves is None:
        # Fallback: CPU heuristic ordering
        scored = score_moves_batch(
            state, moves, knobs=knobs, return_children=False
        )
        ordered_moves = [(r, c) for (r, c), _ in scored[:top_n]]

    # Phase 2: Minimax on top candidates (heuristic evaluation at leaves)
    # Apply child states for ordered moves
    ordered_with_children = []
    for (row, col) in ordered_moves:
        child = apply_move(state, row, col)
        ordered_with_children.append((row, col, child))

    nodes_searched = 0
    candidates = []
    best_move = ordered_moves[0]
    best_score = float("-inf")

    for (row, col, child) in ordered_with_children:
        nodes_searched += 1

        # Check for immediate win
        winner = check_winner(child)
        if winner == player:
            return SearchResult(
                row=row, col=col, score=10000.0,
                depth=depth, nodes_searched=nodes_searched,
                candidates=[{"row": row, "col": col, "score": 10000.0}],
            )

        # Minimax search
        if depth > 1:
            eval_score = minimax(
                child, depth - 1, float("-inf"), float("inf"),
                False, player, top_n, knobs, 0, root_depth=depth,
            )
        else:
            eval_score = evaluate_position(child, player, knobs)

        candidates.append({
            "row": row,
            "col": col,
            "score": eval_score,
        })

        if eval_score > best_score:
            best_score = eval_score
            best_move = (row, col)

    # Sort candidates
    candidates.sort(key=lambda x: -x["score"])

    # Apply temperature sampling if requested
    if temperature > TEMP_FLOOR and rng is not None and len(candidates) > 1:
        selected = pick_move_from_scores(candidates, temperature, rng, mode=mode)
    else:
        selected = deterministic_best(candidates)

    return SearchResult(
        row=selected["row"],
        col=selected["col"],
        score=selected["score"],
        depth=depth,
        nodes_searched=nodes_searched,
        candidates=candidates,
    )


# Type hint for optional import
try:
    from .move_model import MoveRanker
except ImportError:
    MoveRanker = None  # type: ignore
