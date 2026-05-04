"""Save AlphaZero games in replay-compatible format.

Saves games to scripts/GPU/logs/games/ with naming:
  iter_{iteration:04d}_game_{game_idx:03d}.json

Format is compatible with Replay.html viewer.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


def save_game_replay(
    games_dir: Path,
    iteration: int,
    game_idx: int,
    winner: Optional[str],
    move_history: Tuple[Tuple[int, int], ...],
    n_moves: int,
    active_size: int = 24,
    simulations: int = 0,
    draw_reason: Optional[str] = None,
    start_player: str = "red",
    resigned_by: Optional[str] = None,
    opening_diagnostics: Optional[list] = None,
    opening_diagnostics_meta: Optional[dict] = None,
    # Per-game stats persistence (spec 2026-04-29)
    worker_id: Optional[int] = None,
    wall_time_s: Optional[float] = None,
    adjudication_block_reason: Optional[str] = None,
    final_root_value: Optional[float] = None,
    final_top1_share: Optional[float] = None,
    leaf_evals: int = 0,
    backups: int = 0,
    nn_batches: int = 0,
    # Per-move stats (spec 2026-05-03 §5).
    # Lists are 1:1 with move_history; entries default to None when the
    # caller does not supply per-move data (e.g., legacy callers).
    move_root_values: Optional[list] = None,
    move_top1_shares: Optional[list] = None,
    # Inline closeout diagnostics (spec 2026-05-03 §8.5)
    goal_completion_diagnostics: Optional[list] = None,
    goal_completion_diagnostics_meta: Optional[dict] = None,
) -> Path:
    """Save a single game in replay-compatible format.

    Args:
        games_dir: Directory to save games (e.g., scripts/GPU/logs/games)
        iteration: Training iteration number
        game_idx: Game index within the iteration
        winner: "red", "black", or None for draw
        move_history: Tuple of (row, col) moves
        n_moves: Total number of moves
        active_size: Board size used
        simulations: MCTS simulations per move
        draw_reason: Reason for draw if applicable
        start_player: Starting player ("red" or "black")

    Returns:
        Path to saved file
    """
    games_dir.mkdir(parents=True, exist_ok=True)

    # Build moves array with player alternation from actual starting player
    moves = []
    players = [start_player, "black" if start_player == "red" else "red"]
    n_history = len(move_history)
    if move_root_values is not None and len(move_root_values) != n_history:
        import sys as _sys
        _sys.stderr.write(
            f"[game_saver] move_root_values length {len(move_root_values)} "
            f"!= move_history length {n_history}; tail entries default to null.\n"
        )
    if move_top1_shares is not None and len(move_top1_shares) != n_history:
        import sys as _sys
        _sys.stderr.write(
            f"[game_saver] move_top1_shares length {len(move_top1_shares)} "
            f"!= move_history length {n_history}; tail entries default to null.\n"
        )
    for i, (row, col) in enumerate(move_history):
        player = players[i % 2]
        rv = None
        if move_root_values is not None and i < len(move_root_values):
            v = move_root_values[i]
            rv = float(v) if v is not None else None
        ts = None
        if move_top1_shares is not None and i < len(move_top1_shares):
            v = move_top1_shares[i]
            ts = float(v) if v is not None else None
        moves.append({
            "turn": i + 1,
            "player": player,
            "row": int(row),
            "col": int(col),
            "bridges_created": [],
            "heuristics": {},
            "search_score": rv,
            "root_top1_share": ts,
        })

    # Determine winner string for format
    winner_str = winner if winner else "draw"

    # Determine reason
    if winner:
        # Winner exists - could be normal win, resignation, or adjudication
        if draw_reason == "resign":
            reason = "resign"
        elif draw_reason == "adjudicated":
            reason = "adjudicated"
        else:
            reason = "win"
    elif draw_reason:
        reason = draw_reason
    else:
        reason = "draw"

    # Build meta dict
    meta = {
        "board_size": active_size,
        "mode": "alphazero",
        "reason": reason,
        "iteration": iteration,
        "game_idx": game_idx,
        "simulations": simulations,
        "n_moves": n_moves,
        "starting_player": start_player,
    }
    # Add resigned_by only for resign games
    if reason == "resign" and resigned_by:
        meta["resigned_by"] = resigned_by

    # Per-game stats persistence (spec 2026-04-29).
    # Nullable flat fields use explicit None checks so 0.0 is preserved.
    # Compute counters always present; None upstream → 0 (counters are non-negative).
    meta["worker_id"] = int(worker_id) if worker_id is not None else None
    meta["wall_time_s"] = float(wall_time_s) if wall_time_s is not None else None
    meta["adjudication_block_reason"] = adjudication_block_reason
    meta["final_root_value"] = float(final_root_value) if final_root_value is not None else None
    meta["final_top1_share"] = float(final_top1_share) if final_top1_share is not None else None
    meta["compute"] = {
        "leaf_evals": int(leaf_evals or 0),
        "backups": int(backups or 0),
        "nn_batches": int(nn_batches or 0),
    }

    record = {
        "id": f"iter_{iteration:04d}_game_{game_idx:03d}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": "alphazero",
        "depth": simulations,
        "seed": game_idx,
        "winner": winner_str,
        "starting_player": start_player,
        "moves": moves,
        "meta": meta,
    }

    # Inline closeout diagnostics: top-level JSON keys (spec §8.5).
    # Both keys absent when meta is None — clean schema on disabled runs.
    if goal_completion_diagnostics_meta is not None:
        record["goal_completion_diagnostics"] = list(goal_completion_diagnostics or [])
        record["goal_completion_diagnostics_meta"] = goal_completion_diagnostics_meta

    if opening_diagnostics:
        record["opening_diagnostics"] = opening_diagnostics
        record["opening_diagnostics_meta"] = opening_diagnostics_meta

    # Save with iteration/game naming
    filename = f"iter_{iteration:04d}_game_{game_idx:03d}.json"
    filepath = games_dir / filename

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)

    return filepath


class GameSaver:
    """Manages saving sample games during training.

    Saves up to `max_games_per_iter` games per iteration.
    """

    def __init__(
        self,
        games_dir: Path,
        max_games_per_iter: int = 5,
        simulations: int = 0,
        active_size: int = 24,
    ):
        self.games_dir = Path(games_dir)
        self.max_games_per_iter = max_games_per_iter
        self.simulations = simulations
        self.active_size = active_size

        # Per-iteration state
        self._current_iter = -1
        self._games_saved_this_iter = 0

    def set_iteration(self, iteration: int, simulations: int = None, active_size: int = None):
        """Reset for a new iteration."""
        self._current_iter = iteration
        self._games_saved_this_iter = 0
        if simulations is not None:
            self.simulations = simulations
        if active_size is not None:
            self.active_size = active_size

    def maybe_save_game(
        self,
        winner: Optional[str],
        move_history: Optional[Tuple[Tuple[int, int], ...]],
        n_moves: int,
        draw_reason: Optional[str] = None,
        start_player: str = "red",
        resigned_by: Optional[str] = None,
        opening_diagnostics: Optional[list] = None,
        opening_diagnostics_meta: Optional[dict] = None,
        # Per-game stats persistence (spec 2026-04-29)
        worker_id: Optional[int] = None,
        wall_time_s: Optional[float] = None,
        adjudication_block_reason: Optional[str] = None,
        final_root_value: Optional[float] = None,
        final_top1_share: Optional[float] = None,
        leaf_evals: int = 0,
        backups: int = 0,
        nn_batches: int = 0,
        # Per-move stats (spec 2026-05-03 §5).
        move_root_values: Optional[list] = None,
        move_top1_shares: Optional[list] = None,
        # Inline closeout diagnostics (spec 2026-05-03 §8.5)
        goal_completion_diagnostics: Optional[list] = None,
        goal_completion_diagnostics_meta: Optional[dict] = None,
    ) -> Optional[Path]:
        """Save game if we haven't reached the limit for this iteration.

        Returns:
            Path to saved file, or None if skipped
        """
        if self.max_games_per_iter <= 0:
            return None

        if move_history is None or len(move_history) == 0:
            return None

        if self._games_saved_this_iter >= self.max_games_per_iter:
            return None

        filepath = save_game_replay(
            games_dir=self.games_dir,
            iteration=self._current_iter,
            game_idx=self._games_saved_this_iter,
            winner=winner,
            move_history=move_history,
            n_moves=n_moves,
            active_size=self.active_size,
            simulations=self.simulations,
            draw_reason=draw_reason,
            start_player=start_player,
            resigned_by=resigned_by,
            opening_diagnostics=opening_diagnostics,
            opening_diagnostics_meta=opening_diagnostics_meta,
            # Per-game stats persistence (spec 2026-04-29)
            worker_id=worker_id,
            wall_time_s=wall_time_s,
            adjudication_block_reason=adjudication_block_reason,
            final_root_value=final_root_value,
            final_top1_share=final_top1_share,
            leaf_evals=leaf_evals,
            backups=backups,
            nn_batches=nn_batches,
            move_root_values=move_root_values,
            move_top1_shares=move_top1_shares,
            goal_completion_diagnostics=goal_completion_diagnostics,
            goal_completion_diagnostics_meta=goal_completion_diagnostics_meta,
        )

        self._games_saved_this_iter += 1
        return filepath

    @property
    def games_saved_this_iter(self) -> int:
        return self._games_saved_this_iter
