"""TwixT self-play engine using real game rules and AI search."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..game.state import GameState
from ..game.rules import apply_move, check_winner, generate_moves
from ..ai.search import choose_move, temperature_for_ply
from ..ai.heuristics import component_metrics, normalize_knobs_for_mode, score_moves
from ..replay.format import Move


@dataclass
class SimOutcome:
    """Result of a simulated game."""
    winner: str  # "red"|"black"|"draw"
    moves: List[Move]
    total_moves: int
    reason: str  # "win"|"stall"|"max_moves"|"no_moves"
    starting_player: str  # "red"|"black" - needed for color_advantage calculation
    stats: Dict[str, Any] = field(default_factory=dict)


class TwixtSimulator:
    """Real TwixT game simulator using game rules and AI search.

    Features:
    - Uses actual game rules (edge restrictions, bridge crossing, win detection)
    - Uses minimax search with alpha-beta pruning
    - Records detailed game telemetry for replay
    - Supports training (stochastic) and debug (deterministic) modes
    """

    def __init__(self, board_size: int = 24, max_moves: int = 220, stall_limit: int = 40):
        """Initialize simulator.

        Args:
            board_size: Board size (default 24x24)
            max_moves: Maximum moves before declaring draw
            stall_limit: Consecutive moves without progress before stall
        """
        self.board_size = board_size
        self.max_moves = max_moves
        self.stall_limit = stall_limit

    def play_one(
        self,
        knobs: Dict[str, float],
        *,
        seed: int,
        depth: int,
        top_n: int = 12,
        use_value_model: bool = True,
        starting_player: Optional[str] = None,
        rng: Optional[random.Random] = None,
        mode: str = "training",
    ) -> SimOutcome:
        """Play a single game with given configuration.

        Args:
            knobs: Configuration knobs (for heuristic tuning and temperature schedule)
            seed: Random seed (for reproducibility)
            depth: Search depth
            top_n: Number of moves to consider at each search level
            use_value_model: Whether to use value model for move scoring
            starting_player: "red" or "black" (default: alternates based on seed)
            rng: Random number generator for temperature sampling (optional)
            mode: "training" (stochastic) or "debug" (deterministic)

        Returns:
            SimOutcome with winner, moves, and game metadata
        """
        knobs_for_play = normalize_knobs_for_mode(knobs, mode=mode)

        # Alternate starting player for balance: even seeds start black, odd start red
        if starting_player is None:
            starting_player = "black" if seed % 2 == 0 else "red"

        # Create RNG from seed if not provided
        if rng is None:
            rng = random.Random(seed)

        state = GameState(board_size=self.board_size, to_move=starting_player)
        moves: List[Move] = []
        stats: Dict[str, Any] = {
            "stagnation_max": 0,
            "progress_events": 0,
            "opening_random_moves": 0,
        }
        score_values: List[float] = []
        stagnation = 0
        progress_state = {
            "red": {"span": 0, "touch_top": False, "touch_bottom": False, "size": 0},
            "black": {"span": 0, "touch_left": False, "touch_right": False, "size": 0},
        }
        record_scores = False
        max_logged_plies = None
        trace_enabled = bool(knobs_for_play.get("debug_trace", 0))
        if trace_enabled:
            stats["trace"] = []
        if mode == "debug":
            sample_rate = float(knobs_for_play.get("debug_sample_rate", 1.0))
            max_logged_plies = int(knobs_for_play.get("debug_max_plies", 120))
            record_scores = rng.random() < max(0.0, min(1.0, sample_rate))
            if trace_enabled:
                record_scores = True

        if mode == "training":
            if depth >= 2:
                cap_moves = int(knobs_for_play.get("training_max_moves_d2", self.max_moves))
                stall_limit = int(knobs_for_play.get("training_stall_limit_d2", self.stall_limit))
            else:
                cap_moves = int(knobs_for_play.get("training_max_moves", self.max_moves))
                stall_limit = int(knobs_for_play.get("training_stall_limit", self.stall_limit))
            cap_moves = min(cap_moves, self.max_moves)
        else:
            cap_moves = self.max_moves
            stall_limit = self.stall_limit

        opening_quadrant = None
        if mode == "training":
            opening_quadrant = rng.choice((0, 1, 2, 3))

        for turn in range(1, cap_moves + 1):
            player = state.to_move
            legal_moves = generate_moves(state)

            if not legal_moves:
                return SimOutcome(
                    winner="draw",
                    moves=moves,
                    total_moves=len(moves),
                    reason="no_moves",
                    starting_player=starting_player,
                    stats=_finalize_stats(stats, score_values),
                )

            if mode == "debug":
                temperature = 0.0
            else:
                temperature = temperature_for_ply(turn - 1, knobs_for_play)
                if mode == "training" and depth >= 2:
                    temperature *= float(knobs_for_play.get("training_temp_scale_d2", 1.0))

            if depth >= 2:
                opening_plies = int(knobs_for_play.get("opening_random_plies_d2", 0))
                opening_top_k = int(knobs_for_play.get("opening_random_top_k_d2", 0))
                opening_uniform_plies = int(knobs_for_play.get("opening_uniform_plies_d2", 0))
                opening_uniform_margin = int(knobs_for_play.get("opening_uniform_margin_d2", 0))
                opening_forbid_plies = int(knobs_for_play.get("opening_forbid_edge_plies_d2", 0))
                opening_forbid_margin = int(knobs_for_play.get("opening_forbid_edge_margin_d2", 0))
                opening_quadrant_plies = int(knobs_for_play.get("opening_quadrant_plies_d2", 0))
                opening_quadrant_margin = int(knobs_for_play.get("opening_quadrant_margin_d2", 0))
            else:
                opening_plies = int(knobs_for_play.get("opening_random_plies", 0))
                opening_top_k = int(knobs_for_play.get("opening_random_top_k", 0))
                opening_uniform_plies = 0
                opening_uniform_margin = 0
                opening_forbid_plies = 0
                opening_forbid_margin = 0
                opening_quadrant_plies = 0
                opening_quadrant_margin = 0

            if mode == "training" and opening_forbid_plies > 0 and turn <= opening_forbid_plies:
                limit = state.board_size - 1 - opening_forbid_margin
                if opening_forbid_margin > 0:
                    center_moves = [
                        (r, c) for (r, c) in legal_moves
                        if opening_forbid_margin <= r <= limit and opening_forbid_margin <= c <= limit
                    ]
                    if center_moves:
                        legal_moves = center_moves

            if mode == "training" and opening_quadrant_plies > 0 and turn <= opening_quadrant_plies and opening_quadrant is not None:
                mid = (state.board_size - 1) // 2
                r_min = opening_quadrant_margin
                r_max = state.board_size - 1 - opening_quadrant_margin
                c_min = opening_quadrant_margin
                c_max = state.board_size - 1 - opening_quadrant_margin
                if opening_quadrant in (0, 1):  # top
                    r_max = max(r_min, mid)
                else:  # bottom
                    r_min = min(r_max, mid + 1)
                if opening_quadrant in (0, 2):  # left
                    c_max = max(c_min, mid)
                else:  # right
                    c_min = min(c_max, mid + 1)
                quad_moves = [
                    (r, c) for (r, c) in legal_moves
                    if r_min <= r <= r_max and c_min <= c <= c_max
                ]
                if quad_moves:
                    legal_moves = quad_moves

            if mode == "training" and opening_uniform_plies > 0 and turn <= opening_uniform_plies:
                if opening_uniform_margin > 0:
                    limit = state.board_size - 1 - opening_uniform_margin
                    center_moves = [
                        (r, c) for (r, c) in legal_moves
                        if opening_uniform_margin <= r <= limit and opening_uniform_margin <= c <= limit
                    ]
                else:
                    center_moves = None
                row, col = rng.choice(center_moves or legal_moves)
                search_score = 0.0
                stats["opening_random_moves"] += 1
            elif mode == "training" and opening_plies > 0 and turn <= opening_plies:
                if opening_top_k <= 0:
                    row, col = rng.choice(legal_moves)
                    search_score = 0.0
                    stats["opening_random_moves"] += 1
                else:
                    scored = score_moves(state, legal_moves, knobs=knobs_for_play)
                    limit = max(1, min(opening_top_k, len(scored)))
                    top = scored[:limit]
                    move, score = rng.choice(top)
                    row, col = move
                    search_score = score
                    stats["opening_random_moves"] += 1
            else:
                # Use search to find best move with temperature sampling
                result = choose_move(
                    state, knobs_for_play,
                    depth=depth, top_n=top_n,
                    use_value_model=use_value_model,
                    temperature=temperature, rng=rng, mode=mode
                )
                row, col = result.row, result.col
                search_score = result.score
                score_values.append(float(result.score))

            # Record detailed move info (score only for sampled debug games)
            move_record = Move(
                turn=turn,
                player=player,
                row=row,
                col=col,
                search_score=search_score if record_scores and (max_logged_plies is None or turn <= max_logged_plies) else None,
            )

            # Store bridges that will be created
            from ..game.bridge import KNIGHT_OFFSETS, bridges_cross, normalize_edge
            bridges_created = []
            for dr, dc in KNIGHT_OFFSETS:
                r2, c2 = row + dr, col + dc
                if 0 <= r2 < self.board_size and 0 <= c2 < self.board_size:
                    if (r2, c2) in state.pegs and state.pegs[(r2, c2)] == player:
                        if not bridges_cross(state, row, col, r2, c2):
                            edge = normalize_edge((row, col), (r2, c2))
                            if edge not in state.bridges:
                                bridges_created.append({
                                    "from": {"row": edge[0][0], "col": edge[0][1]},
                                    "to": {"row": edge[1][0], "col": edge[1][1]},
                                })
            move_record.bridges_created = bridges_created

            moves.append(move_record)

            # Apply the move
            state = apply_move(state, row, col)

            # Check for winner
            winner = check_winner(state)
            if winner:
                return SimOutcome(
                    winner=winner,
                    moves=moves,
                    total_moves=len(moves),
                    reason="win",
                    starting_player=starting_player,
                    stats=_finalize_stats(stats, score_values),
                )

            # Progress check (span + edge touches)
            metrics = component_metrics(state, player)
            if player == "red":
                span = metrics.get("max_row_span", 0)
                touch_top = bool(metrics.get("touches_top"))
                touch_bottom = bool(metrics.get("touches_bottom"))
                size = len(metrics.get("largest_component") or ())
                prev = progress_state["red"]
                advanced = (
                    span > prev["span"]
                    or size > prev["size"]
                    or (touch_top and not prev["touch_top"])
                    or (touch_bottom and not prev["touch_bottom"])
                )
                progress_state["red"] = {
                    "span": span,
                    "touch_top": touch_top,
                    "touch_bottom": touch_bottom,
                    "size": size,
                }
            else:
                span = metrics.get("max_col_span", 0)
                touch_left = bool(metrics.get("touches_left"))
                touch_right = bool(metrics.get("touches_right"))
                size = len(metrics.get("largest_component") or ())
                prev = progress_state["black"]
                advanced = (
                    span > prev["span"]
                    or size > prev["size"]
                    or (touch_left and not prev["touch_left"])
                    or (touch_right and not prev["touch_right"])
                )
                progress_state["black"] = {
                    "span": span,
                    "touch_left": touch_left,
                    "touch_right": touch_right,
                    "size": size,
                }

            if advanced:
                stagnation = 0
                stats["progress_events"] += 1
            else:
                stagnation += 1
                if stagnation > stats["stagnation_max"]:
                    stats["stagnation_max"] = stagnation

            if trace_enabled and (max_logged_plies is None or turn <= max_logged_plies):
                stats["trace"].append(
                    {
                        "turn": turn,
                        "player": player,
                        "row": row,
                        "col": col,
                        "search_score": search_score,
                        "stagnation": stagnation,
                        "progress_red": progress_state["red"],
                        "progress_black": progress_state["black"],
                        "largest_size": size,
                    }
                )

            if stagnation >= stall_limit:
                winner_override = _winner_from_span(state, margin=int(knobs_for_play.get("training_span_win_margin", 1)))
                return SimOutcome(
                    winner=winner_override or "draw",
                    moves=moves,
                    total_moves=len(moves),
                    reason="stall",
                    starting_player=starting_player,
                    stats=_finalize_stats({**stats, "stall_turn": turn}, score_values),
                )

        winner_override = _winner_from_span(state, margin=int(knobs_for_play.get("training_span_win_margin", 1)))
        return SimOutcome(
            winner=winner_override or "draw",
            moves=moves,
            total_moves=len(moves),
            reason="max_moves",
            starting_player=starting_player,
            stats=_finalize_stats({**stats, "max_moves": cap_moves}, score_values),
        )

    def play_batch(
        self,
        knobs: Dict[str, float],
        *,
        seeds: List[int],
        depth: int,
        top_n: int = 12,
        mode: str = "training",
    ) -> List[SimOutcome]:
        """Play multiple games with given configuration.

        Args:
            knobs: Configuration knobs
            seeds: List of random seeds
            depth: Search depth
            top_n: Number of moves to consider

        Returns:
            List of SimOutcome for each game
        """
        return [self.play_one(knobs, seed=s, depth=depth, top_n=top_n, mode=mode) for s in seeds]

    def play_paired(
        self,
        knobs: Dict[str, float],
        *,
        seed: int,
        depth: int,
        top_n: int = 12,
        use_value_model: bool = True,
        mode: str = "training",
    ) -> Tuple[SimOutcome, SimOutcome]:
        """Play two games with same seed but swapped starting players.

        Returns (outcome1, outcome2) where colors are swapped between games.
        This allows measuring bias with much lower variance.

        Args:
            knobs: Configuration knobs
            seed: Random seed for both games
            depth: Search depth
            top_n: Number of moves to consider
            use_value_model: Whether to use value model

        Returns:
            Tuple of (game1_outcome, game2_outcome)
        """
        rng1 = random.Random(seed)
        rng2 = random.Random(seed)  # Same seed for reproducibility

        # Game 1: seed determines starting player (even=black, odd=red)
        outcome1 = self.play_one(
            knobs, seed=seed, depth=depth, top_n=top_n,
            use_value_model=use_value_model, rng=rng1, mode=mode
        )

        # Game 2: opposite starting player
        first_player = "black" if seed % 2 == 0 else "red"
        opposite = "red" if first_player == "black" else "black"
        outcome2 = self.play_one(
            knobs, seed=seed, depth=depth, top_n=top_n,
            use_value_model=use_value_model, starting_player=opposite, rng=rng2, mode=mode
        )

        return (outcome1, outcome2)

    def play_fast(
        self,
        knobs: Dict[str, float],
        *,
        seed: int,
        mode: str = "training",
    ) -> SimOutcome:
        """Play a fast game using greedy heuristics only (no search).

        Much faster than play_one but lower quality play.
        Useful for quick bias estimation.

        Args:
            knobs: Configuration knobs (unused for now)
            seed: Random seed for tiebreaking

        Returns:
            SimOutcome
        """
        knobs_for_play = normalize_knobs_for_mode(knobs, mode=mode)
        rng = random.Random(seed)
        # Alternate starting player like play_one
        starting_player = "black" if seed % 2 == 0 else "red"
        state = GameState(board_size=self.board_size, to_move=starting_player)
        moves: List[Move] = []
        stats: Dict[str, Any] = {
            "stagnation_max": 0,
            "progress_events": 0,
            "opening_random_moves": 0,
        }
        stagnation = 0
        progress_state = {
            "red": {"span": 0, "touch_top": False, "touch_bottom": False, "size": 0},
            "black": {"span": 0, "touch_left": False, "touch_right": False, "size": 0},
        }

        for turn in range(1, self.max_moves + 1):
            player = state.to_move
            legal_moves = generate_moves(state)

            if not legal_moves:
                return SimOutcome(
                    winner="draw",
                    moves=moves,
                    total_moves=len(moves),
                    reason="no_moves",
                    starting_player=starting_player,
                    stats=stats,
                )

            # Score moves with heuristics (using knobs)
            scored = score_moves(state, legal_moves, knobs=knobs_for_play)
            best_score = scored[0][1]
            best_moves = [m for m, s in scored if s == best_score]
            row, col = rng.choice(best_moves)

            moves.append(Move(turn=turn, player=player, row=row, col=col))
            state = apply_move(state, row, col)

            winner = check_winner(state)
            if winner:
                return SimOutcome(
                    winner=winner,
                    moves=moves,
                    total_moves=len(moves),
                    reason="win",
                    starting_player=starting_player,
                    stats=stats,
                )

            # Progress check (span + edge touches)
            metrics = component_metrics(state, player)
            if player == "red":
                span = metrics.get("max_row_span", 0)
                touch_top = bool(metrics.get("touches_top"))
                touch_bottom = bool(metrics.get("touches_bottom"))
                size = len(metrics.get("largest_component") or ())
                prev = progress_state["red"]
                advanced = (
                    span > prev["span"]
                    or size > prev["size"]
                    or (touch_top and not prev["touch_top"])
                    or (touch_bottom and not prev["touch_bottom"])
                )
                progress_state["red"] = {
                    "span": span,
                    "touch_top": touch_top,
                    "touch_bottom": touch_bottom,
                    "size": size,
                }
            else:
                span = metrics.get("max_col_span", 0)
                touch_left = bool(metrics.get("touches_left"))
                touch_right = bool(metrics.get("touches_right"))
                size = len(metrics.get("largest_component") or ())
                prev = progress_state["black"]
                advanced = (
                    span > prev["span"]
                    or size > prev["size"]
                    or (touch_left and not prev["touch_left"])
                    or (touch_right and not prev["touch_right"])
                )
                progress_state["black"] = {
                    "span": span,
                    "touch_left": touch_left,
                    "touch_right": touch_right,
                    "size": size,
                }

            if advanced:
                stagnation = 0
                stats["progress_events"] += 1
            else:
                stagnation += 1
                if stagnation > stats["stagnation_max"]:
                    stats["stagnation_max"] = stagnation

            if stagnation >= self.stall_limit:
                return SimOutcome(
                    winner="draw",
                    moves=moves,
                    total_moves=len(moves),
                    reason="stall",
                    starting_player=starting_player,
                    stats={**stats, "stall_turn": turn},
                )

        return SimOutcome(
            winner="draw",
            moves=moves,
            total_moves=len(moves),
            reason="max_moves",
            starting_player=starting_player,
            stats={**stats, "max_moves": self.max_moves},
        )


def compute_bias(outcomes: List[SimOutcome]) -> float:
    """Compute red-win bias from a list of game outcomes.

    Returns value between -1 (all black wins) and +1 (all red wins).
    Draws count as 0.
    """
    if not outcomes:
        return 0.0

    red_wins = sum(1 for o in outcomes if o.winner == "red")
    black_wins = sum(1 for o in outcomes if o.winner == "black")
    total_decisive = red_wins + black_wins

    if total_decisive == 0:
        return 0.0

    return (red_wins - black_wins) / total_decisive


def _finalize_stats(stats: Dict[str, Any], scores: List[float]) -> Dict[str, Any]:
    if scores:
        stats["avg_search_score"] = sum(scores) / len(scores)
        stats["min_search_score"] = min(scores)
        stats["max_search_score"] = max(scores)
    else:
        stats["avg_search_score"] = None
        stats["min_search_score"] = None
        stats["max_search_score"] = None
    return stats


def _winner_from_span(state: GameState, *, margin: int) -> Optional[str]:
    red = component_metrics(state, "red")
    black = component_metrics(state, "black")
    red_span = int(red.get("max_row_span", 0))
    black_span = int(black.get("max_col_span", 0))
    if red_span - black_span >= margin:
        return "red"
    if black_span - red_span >= margin:
        return "black"
    return None


def paired_games_report(outcomes: List[Tuple[SimOutcome, SimOutcome]]) -> Dict[str, Any]:
    """Analyze paired game outcomes for bias measurement.

    Args:
        outcomes: List of (game_A, game_B) pairs where:
            - game_A: First player is determined by seed (even=black, odd=red)
            - game_B: First player is opposite of game_A

    Returns:
        Dict with aggregate stats and color-advantage metrics.
    """
    red_wins = 0
    black_wins = 0
    draws = 0
    paired_delta = 0  # How often winner differs between paired games
    color_advantage_score = 0  # Net first-move advantage

    for o1, o2 in outcomes:
        # Aggregate stats
        for o in (o1, o2):
            if o.winner == "red":
                red_wins += 1
            elif o.winner == "black":
                black_wins += 1
            else:
                draws += 1

        # Paired delta: did swapping colors change the winner?
        if o1.winner != o2.winner:
            paired_delta += 1

        # Color-advantage score:
        # o1 starting player determined by seed, o2 has opposite
        # Check if first player won both games
        first_player_o1 = o1.starting_player  # who moved first in game 1
        first_player_o2 = o2.starting_player  # who moved first in game 2 (opposite)

        first_won_o1 = (o1.winner == first_player_o1)
        first_won_o2 = (o2.winner == first_player_o2)

        if first_won_o1 and first_won_o2:
            color_advantage_score += 1  # First-move advantage
        elif not first_won_o1 and not first_won_o2 and o1.winner != "draw" and o2.winner != "draw":
            color_advantage_score -= 1  # Second-move advantage (rare)
        # else: 0 (split or draws)

    total_games = len(outcomes) * 2
    n_pairs = max(1, len(outcomes))

    return {
        "total_games": total_games,
        "red_wins": red_wins,
        "black_wins": black_wins,
        "draws": draws,
        "red_win_rate": red_wins / max(1, red_wins + black_wins),
        "paired_delta": paired_delta,
        "paired_delta_rate": paired_delta / n_pairs,
        "color_advantage_score": color_advantage_score,
        "color_advantage_normalized": color_advantage_score / n_pairs,
    }
