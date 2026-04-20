"""IPC message types for multi-process self-play.

These dataclasses define the communication protocol between:
- Worker processes (CPU-only MCTS)
- Main process (GPU inference server + training)

All types are frozen (immutable) and pickle-safe for multiprocessing.Queue.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class InferenceRequest:
    """Request from worker to inference server."""
    worker_id: int
    request_id: int
    boards: np.ndarray      # (B, H, W, C) float32
    move_rows: np.ndarray   # (B, M) int32
    move_cols: np.ndarray   # (B, M) int32
    move_mask: np.ndarray   # (B, M) float32
    active_size: int


@dataclass(frozen=True)
class InferenceResponse:
    """Response from inference server to worker."""
    request_id: int
    priors: np.ndarray      # (B, M) float32
    values: np.ndarray      # (B,) float32


@dataclass(frozen=True)
class WorkerStats:
    """Periodic stats from worker for monitoring."""
    worker_id: int
    games_played: int
    positions_sent: int


@dataclass(frozen=True)
class StopSignal:
    """Signal to stop the inference server."""
    reason: str = "stop"


@dataclass(frozen=True)
class WorkerDone:
    """Signal that a worker has finished all its games."""
    worker_id: int
    games_played: int
    positions_sent: int
    wall_time_s: float


@dataclass(frozen=True)
class GameComplete:
    """Signal that a worker has finished one game (for curriculum + MCTS stats)."""
    worker_id: int
    winner: str  # "red", "black", or "draw"
    draw_reason: int  # 0=none, 1=timeout, 2=board_full, 3=state_cap, 4=unknown, 5=resign, 6=adjudicated
    n_moves: int
    n_positions: int

    # MCTS stats (per game)
    nn_calls: int
    expand_calls: int
    nn_batches: int
    total_backups: int
    total_waiters: int
    unique_leaves: int
    max_waiters: int
    flush_full: int
    flush_stall: int
    flush_tail: int

    # Optional move history for replay saving (tuple for frozen dataclass)
    move_history: Optional[Tuple[Tuple[int, int], ...]] = None
    # Starting player for correct replay attribution ("red" or "black")
    start_player: str = "red"
    # Resign gate stats
    rg_checks_red: int = 0
    rg_checks_black: int = 0
    rg_value_hits_red: int = 0
    rg_value_hits_black: int = 0
    rg_eligible_red: int = 0
    rg_eligible_black: int = 0
    rg_top1_samples: Tuple[float, ...] = ()
    # Adjudication diagnostics
    adj_attempted: bool = False
    adj_blocked_by: Optional[str] = None  # "ply", "threshold", "visits", "top1", or None
    adj_abs_rv: Optional[float] = None
    adj_top1: Optional[float] = None
    adj_total_visits: Optional[int] = None
    # Opening penalty diagnostics (per-root records for diagnostic window)
    opening_diagnostics: Tuple[dict, ...] = ()
    opening_diagnostics_meta: Optional[dict] = None
    # Phase 4: per-game replay cap diagnostics
    n_positions_original: int = 0
    n_positions_kept: int = 0
