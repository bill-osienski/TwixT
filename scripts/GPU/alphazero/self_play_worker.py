"""Self-play worker process for multi-process training.

Runs in a separate CPU-only process. Generates games using MCTS with
RemoteEvaluator, streaming positions in chunks to the main process.

Key features:
- Chunked position streaming (backpressure via bounded queue)
- Periodic stats reporting
- Explicit WorkerDone signal (not Queue.empty())
"""
from __future__ import annotations
import random
import signal
import sys
import time
from typing import Any, List, Optional

from .ipc_messages import WorkerStats, WorkerDone, GameComplete
from .remote_evaluator import RemoteEvaluator
from .self_play import (
    play_game, PositionRecord,
    DRAW_TIMEOUT, DRAW_BOARD_FULL, DRAW_STATE_CAP, DRAW_UNKNOWN, RESIGN,
    ADJUDICATED,
)
from .mcts import MCTSConfig

# Mapping from string draw reasons to int codes
_DRAW_REASON_TO_INT = {
    None: 0,
    DRAW_TIMEOUT: 1,
    DRAW_BOARD_FULL: 2,
    DRAW_STATE_CAP: 3,
    DRAW_UNKNOWN: 4,
    RESIGN: 5,  # Resign (has winner but also has draw_reason for metadata)
    ADJUDICATED: 6,  # Adjudicated at timeout (has winner, decisive)
}


def self_play_worker_main(
    worker_id: int,
    request_queue: Any,
    response_queue: Any,
    position_queue: Any,
    stats_queue: Optional[Any],
    mcts_config: MCTSConfig,
    # Dynamic scheduling inputs
    games_total: int,
    next_game_id: Any,  # ctx.Value with internal lock via .get_lock()
    seed: int,
    chunk_size: int = 32,
    max_moves: int = 200,
    add_noise: bool = True,
    active_size: int = 24,
    # Resign parameters
    resign_enabled: bool = False,
    resign_min_ply: int = 80,
    resign_threshold: float = -0.97,
    resign_window: int = 12,
    resign_k: int = 8,
    resign_min_visits: int = 200,
    resign_min_top1_share: float = 0.0,
    # Adjudication parameters
    adjudicate_enabled: bool = False,
    adjudicate_min_ply: int = 120,
    adjudicate_threshold: float = 0.90,
    adjudicate_min_visits: int = 200,
    adjudicate_min_top1_share: float = 0.0,
    adjudicate_debug: bool = False,
    # Phase 4: per-game replay cap (None/0 disables)
    max_positions_per_game: Optional[int] = None,
    endgame_keep_positions: int = 16,
) -> None:
    """Worker process entry point.

    Args:
        worker_id: Unique ID for this worker
        request_queue: Shared queue for inference requests
        response_queue: Per-worker queue for inference responses
        position_queue: Shared queue for streaming positions to trainer
        stats_queue: Optional queue for stats/telemetry
        mcts_config: MCTS configuration
        games_total: Total games to generate across all workers
        next_game_id: Shared atomic counter for dynamic game assignment
        seed: Random seed for reproducibility
        chunk_size: Number of positions per queue put (default 32)
        max_moves: Maximum moves per game before timeout
        add_noise: Whether to add Dirichlet noise at root
        active_size: Curriculum board size
    """
    # Ignore SIGINT in workers - let parent handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        _worker_loop(
            worker_id, request_queue, response_queue, position_queue,
            stats_queue, mcts_config, games_total, next_game_id, seed,
            chunk_size, max_moves, add_noise, active_size,
            resign_enabled, resign_min_ply, resign_threshold,
            resign_window, resign_k, resign_min_visits, resign_min_top1_share,
            adjudicate_enabled, adjudicate_min_ply, adjudicate_threshold,
            adjudicate_min_visits, adjudicate_min_top1_share,
            adjudicate_debug,
            max_positions_per_game, endgame_keep_positions,
        )
    except (KeyboardInterrupt, BrokenPipeError, EOFError, RuntimeError):
        # Graceful exit on interrupt, queue closure, or evaluator timeout
        # RuntimeError from RemoteEvaluator is expected during shutdown
        sys.exit(0)


def _worker_loop(
    worker_id: int,
    request_queue: Any,
    response_queue: Any,
    position_queue: Any,
    stats_queue: Optional[Any],
    mcts_config: MCTSConfig,
    games_total: int,
    next_game_id: Any,  # ctx.Value with internal lock via .get_lock()
    seed: int,
    chunk_size: int,
    max_moves: int,
    add_noise: bool,
    active_size: int,
    # Resign parameters
    resign_enabled: bool,
    resign_min_ply: int,
    resign_threshold: float,
    resign_window: int,
    resign_k: int,
    resign_min_visits: int,
    resign_min_top1_share: float,
    # Adjudication parameters
    adjudicate_enabled: bool,
    adjudicate_min_ply: int,
    adjudicate_threshold: float,
    adjudicate_min_visits: int,
    adjudicate_min_top1_share: float,
    adjudicate_debug: bool,
    # Phase 4: per-game replay cap
    max_positions_per_game: Optional[int],
    endgame_keep_positions: int,
) -> None:
    """Inner worker loop (extracted for clean exception handling)."""
    import time

    evaluator = RemoteEvaluator(worker_id, request_queue, response_queue)

    t0 = time.time()
    games_played = 0
    positions_sent = 0

    while True:
        # Atomically claim a game index (use built-in lock from ctx.Value)
        with next_game_id.get_lock():
            gid = next_game_id.value
            if gid >= games_total:
                break
            next_game_id.value += 1

        # Per-game RNG for reproducibility regardless of scheduling
        game_seed = (seed ^ (gid * 0x9E3779B1)) & 0x7FFFFFFF
        game_rng = random.Random(game_seed)

        # Generate one game (timed for parallel-mode percentile stats)
        game_t0 = time.perf_counter()
        game = play_game(
            evaluator=evaluator,
            mcts_config=mcts_config,
            rng=game_rng,
            max_moves=max_moves,
            add_noise=add_noise,
            active_size=active_size,
            game_id=gid,
            resign_enabled=resign_enabled,
            resign_min_ply=resign_min_ply,
            resign_threshold=resign_threshold,
            resign_window=resign_window,
            resign_k=resign_k,
            resign_min_visits=resign_min_visits,
            resign_min_top1_share=resign_min_top1_share,
            adjudicate_enabled=adjudicate_enabled,
            adjudicate_min_ply=adjudicate_min_ply,
            adjudicate_threshold=adjudicate_threshold,
            adjudicate_min_visits=adjudicate_min_visits,
            adjudicate_min_top1_share=adjudicate_min_top1_share,
            adjudicate_debug=adjudicate_debug,
            max_positions_per_game=max_positions_per_game,
            endgame_keep_positions=endgame_keep_positions,
        )
        games_played += 1

        # Stream positions in chunks (backpressure if queue full)
        buf: List[PositionRecord] = []
        for p in game.positions:
            buf.append(p)
            if len(buf) >= chunk_size:
                position_queue.put(buf)
                positions_sent += len(buf)
                buf = []

        # Send remaining positions
        if buf:
            position_queue.put(buf)
            positions_sent += len(buf)

        # Send game completion signal with MCTS stats
        winner = game.winner if game.winner is not None else "draw"

        # Compute draw_reason_int from game.draw_reason (handles all cases)
        draw_reason_int = _DRAW_REASON_TO_INT.get(game.draw_reason, 0)
        # Defensive: if draw_reason is set but not in dict, force unknown
        if game.draw_reason is not None and draw_reason_int == 0:
            draw_reason_int = _DRAW_REASON_TO_INT[DRAW_UNKNOWN]

        if stats_queue is not None:
            # Convert move_history to tuple for frozen dataclass
            move_history_tuple = tuple(tuple(m) for m in game.move_history) if game.move_history else None
            stats_queue.put(GameComplete(
                worker_id=worker_id,
                winner=winner,
                draw_reason=draw_reason_int,
                n_moves=game.n_moves,
                n_positions=len(game.positions),
                wall_time_s=time.perf_counter() - game_t0,
                nn_calls=game.nn_calls,
                expand_calls=game.expand_calls,
                nn_batches=game.nn_batches,
                total_backups=game.total_backups,
                total_waiters=game.total_waiters,
                unique_leaves=game.unique_leaves,
                max_waiters=game.max_waiters,
                flush_full=game.flush_full,
                flush_stall=game.flush_stall,
                flush_tail=game.flush_tail,
                move_history=move_history_tuple,
                start_player=game.start_player,
                rg_checks_red=game.rg_checks_red,
                rg_checks_black=game.rg_checks_black,
                rg_value_hits_red=game.rg_value_hits_red,
                rg_value_hits_black=game.rg_value_hits_black,
                rg_eligible_red=game.rg_eligible_red,
                rg_eligible_black=game.rg_eligible_black,
                rg_top1_samples=game.rg_top1_samples,
                adj_attempted=game.adj_attempted,
                adj_blocked_by=game.adj_blocked_by,
                adj_abs_rv=game.adj_abs_rv,
                adj_top1=game.adj_top1,
                adj_total_visits=game.adj_total_visits,
                opening_diagnostics=tuple(game.opening_diagnostics),
                opening_diagnostics_meta=game.opening_diagnostics_meta,
                n_positions_original=game.n_positions_original,
                n_positions_kept=game.n_positions_kept,
                # Per-game stats persistence (spec 2026-04-29)
                final_root_value=game.final_root_value,
                final_top1_share=game.final_top1_share,
                move_root_values=tuple(game.move_root_values),
                move_top1_shares=tuple(game.move_top1_shares),
            ))

        # Periodic stats
        if stats_queue is not None and (games_played % 5) == 0:
            stats_queue.put(WorkerStats(worker_id, games_played, positions_sent))

    # Final stats + explicit done signal (don't rely on Queue.empty())
    wall_time_s = time.time() - t0
    if stats_queue is not None:
        stats_queue.put(WorkerStats(worker_id, games_played, positions_sent))
    position_queue.put(WorkerDone(worker_id, games_played, positions_sent, wall_time_s))
