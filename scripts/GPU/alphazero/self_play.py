"""Self-play game generation for AlphaZero training.

This module generates training data by playing games using MCTS with
neural network guidance. Each game produces position records that can
be used to train the network.

Key conventions:
- to_move is stored explicitly in each position (not inferred from ply)
- Outcomes are from the perspective of to_move at each position
- Visit counts are raw (not normalized) - normalization happens in training
"""
from __future__ import annotations

import gc
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mcts import MCTS, MCTSConfig, MCTSNode, encode_move, decode_move
from .opening_diagnostics import (
    build_root_diagnostic,
    build_root_child_details,
    compute_diagnostic_end_ply,
)
from .evaluator import Evaluator
from .game import (
    TwixtState, DIRECTION_TO_CHANNEL,
    CHANNEL_RED_LINKS_START, CHANNEL_BLACK_LINKS_START,
    CHANNEL_BLACK_LEFT_DIST, CHANNEL_BLACK_RIGHT_DIST,
)

# --- Temporary opening diagnostics (enable via TWIXT_OPENING_DEBUG env var) ---
_OPENING_DEBUG = os.environ.get("TWIXT_OPENING_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
_OPENING_DEBUG_GAMES = 16   # Only log for game_id < this
_OPENING_DEBUG_PLIES = 2    # Only log for ply < this (0 and 1)
_OPENING_DEBUG_TOPK = 12    # Show top-K moves

# --- Phase 1: Root-child diagnostics (deep per-child inspection at early plies) ---
# For ply < CHILD_DETAIL_PLIES we attach `root_summary` + `top_children` to the
# per-root diagnostic record. These fields reveal exactly why the search chose
# what it did (q vs u vs score vs ties). Unconditional on penalties; cheap.
CHILD_DETAIL_PLIES = 2      # Attach child details for ply in [0, 1]
CHILD_DETAIL_TOPK = 10      # Top-K children (by visits) to include

# Draw reason constants (used in GameRecord, curriculum, trainer)
DRAW_TIMEOUT = "timeout_selfplay"
DRAW_BOARD_FULL = "terminal_board_full"
DRAW_STATE_CAP = "terminal_state_cap"
DRAW_UNKNOWN = "terminal_unknown"

# Resign constant (game ended by resignation - has winner, not a draw)
RESIGN = "resign"

# Adjudication constant (game hit max_moves; winner assigned by final MCTS eval)
ADJUDICATED = "adjudicated"


def opponent(player: str) -> str:
    """Return opponent color."""
    return "black" if player == "red" else "red"


# --- Phase 4: Per-game replay contribution cap ---

def apply_game_position_cap(
    positions: List["PositionRecord"],
    max_positions_per_game: Optional[int],
    endgame_keep_positions: int,
    rng: random.Random,
) -> Tuple[List["PositionRecord"], int, int]:
    """Cap the positions that a single game contributes to replay.

    Long games flood the buffer (every position gets the same outcome label),
    which lets a few drifting games dominate training signal. This helper
    enforces a per-game cap:
      - if `max_positions_per_game` is None or <= 0: no-op (pass through)
      - if len(positions) <= cap: no-op
      - else: keep the last `endgame_keep_positions` unconditionally (protects
              conversion/endgame supervision), then uniformly sample the
              remainder from the earlier positions to fill the quota.

    Positions are assumed to be in ply order (mirror augmentations, when
    enabled, are interleaved adjacent to their original — preserved by the
    index-based logic below).

    Args:
        positions: list of PositionRecord (in play order).
        max_positions_per_game: cap (None/<=0 disables).
        endgame_keep_positions: positions at the tail to keep unconditionally.
        rng: random.Random used for uniform sampling (reproducible per seed).

    Returns:
        (kept_positions, n_original, n_kept) where n_kept == len(kept).
    """
    n_orig = len(positions)
    if max_positions_per_game is None or max_positions_per_game <= 0:
        return list(positions), n_orig, n_orig
    if n_orig <= max_positions_per_game:
        return list(positions), n_orig, n_orig

    ek = max(0, min(endgame_keep_positions, max_positions_per_game, n_orig))
    remainder_quota = max_positions_per_game - ek
    endgame_start = n_orig - ek
    earlier = positions[:endgame_start]
    endgame = positions[endgame_start:]

    if remainder_quota >= len(earlier):
        # Quota covers all earlier positions; keep everything up to cap
        kept = list(earlier) + list(endgame)
    elif remainder_quota <= 0:
        kept = list(endgame)
    else:
        idxs = rng.sample(range(len(earlier)), remainder_quota)
        idxs.sort()  # preserve play order
        kept = [earlier[i] for i in idxs] + list(endgame)

    return kept, n_orig, len(kept)

# --- Horizontal mirror augmentation ---
try:
    _MIRROR_PROB = float(os.environ.get("TWIXT_MIRROR_PROB", "0.5"))
except ValueError:
    _MIRROR_PROB = 0.5
if _MIRROR_PROB < 0.0:
    _MIRROR_PROB = 0.0
elif _MIRROR_PROB > 1.0:
    _MIRROR_PROB = 1.0


def _build_mirror_dir_perm():
    """Build channel permutation for horizontal mirror (dc -> -dc).
    Returns list of length 8: perm[i] = j means dir channel i maps to
    dir channel j after left-right flip.
    """
    idx_to_dir = {v: k for k, v in DIRECTION_TO_CHANNEL.items()}
    if len(idx_to_dir) != 8:
        raise ValueError(f"Expected 8 link dirs, got {len(idx_to_dir)}")
    keys = sorted(idx_to_dir.keys())
    if keys != list(range(8)):
        raise ValueError(f"Expected dir channel ids 0..7, got {keys}")
    perm = [0] * 8
    for i in keys:
        dr, dc = idx_to_dir[i]
        mirrored = (dr, -dc)
        if mirrored not in DIRECTION_TO_CHANNEL:
            raise ValueError(f"Missing mirrored dir for {(dr, dc)} -> {mirrored}")
        perm[i] = DIRECTION_TO_CHANNEL[mirrored]
    return perm


_MIRROR_DIR_PERM = _build_mirror_dir_perm()


def _mirror_position_lr(board_hwc, legal_moves, visit_counts, active_size):
    """Horizontal (left<->right) mirror within the active square.

    Args:
        board_hwc: (H, W, C) numpy array -- full 24x24x24
        legal_moves: list of (row, col) tuples
        visit_counts: list of ints (parallel to legal_moves)
        active_size: curriculum board size

    Returns:
        (mirrored_board, mirrored_moves, visit_counts)
        visit_counts unchanged (same parallel order).
    """
    S = int(active_size)
    H, W = board_hwc.shape[:2]
    if S < 1 or S > H or S > W:
        raise ValueError(f"Bad active_size={S} for board shape {board_hwc.shape}")
    out = board_hwc.copy()

    # 1) Spatial flip -- only the active square [0:S, 0:S)
    out[:S, :S, :] = out[:S, :S, :][:, ::-1, :]

    # 2) Permute red link direction channels within active square
    #    perm[src] = dst: original dir src maps to mirrored dir dst
    RED0 = CHANNEL_RED_LINKS_START
    red = out[:S, :S, RED0:RED0+8].copy()
    for src_i, dst_j in enumerate(_MIRROR_DIR_PERM):
        out[:S, :S, RED0 + dst_j] = red[:, :, src_i]

    # 3) Permute black link direction channels within active square
    BLK0 = CHANNEL_BLACK_LINKS_START
    blk = out[:S, :S, BLK0:BLK0+8].copy()
    for src_i, dst_j in enumerate(_MIRROR_DIR_PERM):
        out[:S, :S, BLK0 + dst_j] = blk[:, :, src_i]

    # 4) Swap BLACK_LEFT_DIST <-> BLACK_RIGHT_DIST within active square
    LEFT = CHANNEL_BLACK_LEFT_DIST
    RIGHT = CHANNEL_BLACK_RIGHT_DIST
    tmp = out[:S, :S, LEFT].copy()
    out[:S, :S, LEFT] = out[:S, :S, RIGHT]
    out[:S, :S, RIGHT] = tmp

    # Unchanged: 0,1 (pegs -- flip handles), 18 (player -- uniform),
    # 19,20 (red dists -- row-based), 23 (move num -- uniform)

    # 5) Mirror legal_moves
    mirrored_moves = [(r, S - 1 - c) for r, c in legal_moves]

    # Dev-only checks (stripped by python -O)
    if __debug__:
        if not all(0 <= r < S and 0 <= c < S for r, c in mirrored_moves):
            raise ValueError(f"Mirrored move out of bounds (active_size={S})")
        # Peg channel sums must be identical pre/post mirror
        for ch in (0, 1):
            orig_sum = float(board_hwc[:S, :S, ch].sum())
            mirror_sum = float(out[:S, :S, ch].sum())
            if abs(orig_sum - mirror_sum) > 1e-6:
                raise ValueError(f"Mirror changed ch{ch} sum: {orig_sum} -> {mirror_sum}")
        # Mirrored moves must stay unique if input was unique
        if (
            len(set(legal_moves)) == len(legal_moves)
            and len(set(mirrored_moves)) != len(mirrored_moves)
        ):
            raise ValueError("Duplicate mirrored_moves output")

    return out, mirrored_moves, visit_counts


def _log_opening_debug(
    game_id, ply, to_move, n_sims, root_value,
    visit_counts, priors_raw, priors_used, chosen_move,
):
    """Print opening debug info for one ply. Temporary diagnostic."""
    # Detect prior key type (int move_id vs (row,col) tuple)
    # Fall back to priors_used if priors_raw is empty/None
    sample_src = priors_raw or priors_used
    if sample_src:
        sample_key = next(iter(sample_src))
        priors_keyed_by_tuple = isinstance(sample_key, tuple)
    else:
        priors_keyed_by_tuple = False

    # Build (row, col, visits, p_raw, p_used) sorted by visits desc
    entries = []
    for (r, c), v in visit_counts.items():
        if priors_keyed_by_tuple:
            p_raw = priors_raw.get((r, c), 0.0) if priors_raw else 0.0
            p_used = priors_used.get((r, c), 0.0) if priors_used else 0.0
        else:
            mid = encode_move(r, c)
            p_raw = priors_raw.get(mid, 0.0) if priors_raw else 0.0
            p_used = priors_used.get(mid, 0.0) if priors_used else 0.0
        entries.append((r, c, v, p_raw, p_used))
    entries.sort(key=lambda e: (-e[2], -e[4]))  # visits desc, p_used desc (display only; tie counts unaffected)

    if not entries:
        print(f"[OPENDBG] gid={game_id} ply={ply} NO VISIT COUNTS")
        return

    top_k = entries[:_OPENING_DEBUG_TOPK]

    # Tie diagnostics
    max_v = entries[0][2]
    second_v = entries[1][2] if len(entries) > 1 else 0
    gap = max_v - second_v
    top_ties = sum(1 for e in entries if e[2] == max_v)
    near_thresh = max_v * 0.98 if max_v > 0 else 0
    near_ties = sum(1 for e in entries if e[2] >= near_thresh)

    # Chosen move rank (1-indexed)
    chosen_rank = next(
        (i + 1 for i, e in enumerate(entries) if (e[0], e[1]) == chosen_move),
        -1,
    )

    # Header
    print(
        f"[OPENDBG] gid={game_id} ply={ply} to_move={to_move} "
        f"sims={n_sims} rootV={root_value:+.3f}"
    )
    # Top-K with both raw and used priors
    parts = [f"({r},{c}) v={v} p_raw={p_raw:.3f} p_used={p_used:.3f}" for r, c, v, p_raw, p_used in top_k]
    print(f"  top: {' | '.join(parts)}")
    # Summary
    print(
        f"  top1={max_v} top2={second_v} gap={gap} "
        f"top_ties={top_ties} near_ties={near_ties} "
        f"chosen=({chosen_move[0]},{chosen_move[1]}) rank={chosen_rank}"
    )


@dataclass
class PositionRecord:
    """Single training position from self-play.

    IMPORTANT: to_move is stored explicitly, NOT inferred from move index.
    This ensures correct value targets even with non-standard starting positions.

    Attributes:
        board_tensor: Board state as numpy array (H, W, C) NHWC format
                      Stored in MLX-native layout to avoid transpose during training
        to_move: Current player ("red" or "black") - explicit, not inferred
        legal_moves: List of (row, col) legal moves
        visit_counts: Raw visit counts (same order as legal_moves)
        outcome: +1 if to_move won, -1 if lost, 0 draw (set after game ends)
        active_size: Curriculum board size (for training with masked pooling)
        ply: Ply at which this position occurred (0 = first move)
        game_n_moves: Total plies played in the source game (set in outcome loop)
        conversion: Spec 2: closeout aux-loss metadata (optional dict)
    """

    board_tensor: np.ndarray  # (H, W, C) numpy array - NHWC format
    to_move: str  # "red" or "black"
    legal_moves: List[Tuple[int, int]]
    visit_counts: List[int]
    outcome: Optional[float] = None
    active_size: int = 24  # Curriculum board size
    ply: int = 0                        # ply at which this position occurred
    game_n_moves: Optional[int] = None  # total plies in the source game (set in outcome loop)
    conversion: Optional[dict] = None   # Spec 2: closeout aux-loss metadata

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "board_tensor": self.board_tensor.tolist(),
            "to_move": self.to_move,
            "legal_moves": self.legal_moves,
            "visit_counts": self.visit_counts,
            "outcome": self.outcome,
            "active_size": self.active_size,
            "ply": self.ply,
            "game_n_moves": self.game_n_moves,
            "conversion": self.conversion,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PositionRecord:
        """Create from dict."""
        return cls(
            board_tensor=np.array(d["board_tensor"], dtype=np.float32),
            to_move=d["to_move"],
            legal_moves=[tuple(m) for m in d["legal_moves"]],
            visit_counts=d["visit_counts"],
            outcome=d["outcome"],
            active_size=d.get("active_size", 24),
            ply=d.get("ply", 0),
            game_n_moves=d.get("game_n_moves"),
            conversion=d.get("conversion"),     # defaults to None for pre-Spec-2 dicts
        )


@dataclass
class GameRecord:
    """Complete self-play game.

    Attributes:
        positions: List of position records from the game
        winner: "red", "black", or None for draw
        n_moves: Total number of moves played
        move_history: List of (row, col) moves played (for replay/debugging)
        start_player: Starting player ("red" or "black") for replay attribution
        resigned_by: Player who resigned ("red" or "black"), or None if no resignation
        nn_calls: Number of NN evaluations during this game (logical leaf evals)
        expand_calls: Number of node expansions during this game (diagnostic)
        nn_batches: Number of actual NN batch invocations (physical)
        total_backups: Total backups performed (must equal plies * simulations)
        total_waiters: Total waiters backed up (for avg_waiters calculation)
        unique_leaves: Unique leaves expanded (for avg_waiters calculation)
        max_waiters: Max waiters on any single leaf (dogpile detector)
        flush_full: Batch-full flushes (healthy)
        flush_stall: Stall flushes (tree narrowed)
        flush_tail: Tail flushes (end of sims)
    """

    positions: List[PositionRecord]
    winner: Optional[str]
    n_moves: int
    move_history: List[Tuple[int, int]] = field(default_factory=list)
    start_player: str = "red"  # Starting player for replay attribution
    draw_reason: Optional[str] = None  # DRAW_TIMEOUT, DRAW_BOARD_FULL, DRAW_STATE_CAP, DRAW_UNKNOWN, or RESIGN
    resigned_by: Optional[str] = None  # Who resigned (or None)
    nn_calls: int = 0
    expand_calls: int = 0
    nn_batches: int = 0
    total_backups: int = 0
    total_waiters: int = 0
    unique_leaves: int = 0
    max_waiters: int = 0
    flush_full: int = 0
    flush_stall: int = 0
    flush_tail: int = 0
    # Adjudication diagnostics (per-game, aggregated by trainer)
    adj_attempted: bool = False           # timeout + adjudicate_enabled
    adj_blocked_by: Optional[str] = None  # "ply", "threshold", "visits", "top1", or None if eligible
    adj_abs_rv: Optional[float] = None
    adj_top1: Optional[float] = None
    adj_total_visits: Optional[int] = None
    # Resign gate stats (per-game, aggregated by trainer)
    rg_checks_red: int = 0
    rg_checks_black: int = 0
    rg_value_hits_red: int = 0
    rg_value_hits_black: int = 0
    rg_eligible_red: int = 0
    rg_eligible_black: int = 0
    rg_top1_samples: Tuple[float, ...] = ()
    # Opening penalty diagnostics (per-root records for diagnostic window plies)
    opening_diagnostics: List[dict] = field(default_factory=list)
    opening_diagnostics_meta: Optional[dict] = None
    # Phase 4: per-game replay cap diagnostics
    # n_positions_original = positions produced before cap (includes mirrors)
    # n_positions_kept     = positions retained after cap (what trainer sees)
    n_positions_original: int = 0
    n_positions_kept: int = 0
    # Per-game stats persistence (spec 2026-04-29):
    # wall_time_s: per-game wall-clock duration; trainer/IPC paths both populate.
    # final_root_value / final_top1_share: snapshot from the last completed
    # MCTS root search before the game ended (mcts._final_root_value /
    # mcts._final_top1_share). None only in degenerate cases (no search ran,
    # or root had no children with visits).
    wall_time_s: Optional[float] = None
    final_root_value: Optional[float] = None
    final_top1_share: Optional[float] = None
    # Per-move stats (spec 2026-05-03 §5). Both lists are 1:1 with move_history;
    # individual entries are float when populated by MCTS, None for degenerate
    # plies (no visits / no value). Length-equal to move_history on every
    # code path including the resign branch.
    move_root_values: List[Optional[float]] = field(default_factory=list)
    move_top1_shares: List[Optional[float]] = field(default_factory=list)
    # Inline closeout diagnostics (spec 2026-05-03 §8.5). meta is None
    # when emit_enabled was False (clean schema on disabled runs).
    goal_completion_diagnostics: List[dict] = field(default_factory=list)
    goal_completion_diagnostics_meta: Optional[dict] = None
    # Compact per-game goal-completion summary (spec 2026-05-05). None when
    # goal_completion_record_enabled=False on the upstream play_game.
    goal_completion_record: Optional[dict] = None
    # Spec 3 Fix 1: per-game closeout_td1 visit-forcing telemetry snapshot.
    # Captured at game end via mcts.get_closeout_td1_telemetry(). None means
    # the field was never populated (older code path / not captured).
    closeout_td1_telemetry: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "positions": [p.to_dict() for p in self.positions],
            "winner": self.winner,
            "n_moves": self.n_moves,
            "move_history": self.move_history,
            "start_player": self.start_player,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GameRecord:
        """Create from dict."""
        return cls(
            positions=[PositionRecord.from_dict(p) for p in d["positions"]],
            winner=d["winner"],
            n_moves=d["n_moves"],
            move_history=[tuple(m) for m in d.get("move_history", [])],
            start_player=d.get("start_player", "red"),
        )


def _merge_closeout_td1_telemetry(per_worker_telemetry: list) -> dict:
    """Sum closeout_td1_visit_forcing counters across workers/games and
    recompute weighted rates.

    Per-worker telemetry blocks come from MCTS.get_closeout_td1_telemetry().
    Config fields (enabled, min_visits, etc.) are taken from the first
    non-empty block; counter fields are summed; rates are recomputed
    against the summed positions_triggered using the per-block triggered
    weight (so top1/top5 stay correct as weighted averages).
    """
    if not per_worker_telemetry:
        return {}
    first = next((t for t in per_worker_telemetry if t), {})
    out = {k: first.get(k) for k in
           ("enabled", "min_visits", "max_forced_moves",
            "require_high_value", "high_value_threshold")}
    sums = {
        "positions_triggered": 0,
        "positions_skipped_no_candidates": 0,
        "positions_skipped_high_value_gate": 0,
        "forced_sims_total": 0,
        "selected_forced_move_count": 0,
        "_top1_hits": 0,
        "_top5_hits": 0,
    }
    for t in per_worker_telemetry:
        if not t:
            continue
        for k in sums:
            if k.startswith("_"):
                continue
            sums[k] += int(t.get(k, 0) or 0)
        triggered = int(t.get("positions_triggered", 0) or 0)
        sums["_top1_hits"] += int(round(
            (t.get("post_force_endpoint_visit_top1_rate", 0) or 0) * triggered
        ))
        sums["_top5_hits"] += int(round(
            (t.get("post_force_endpoint_visit_top5_rate", 0) or 0) * triggered
        ))
    triggered_total = sums["positions_triggered"]
    out.update({
        "positions_triggered": triggered_total,
        "positions_skipped_no_candidates": sums["positions_skipped_no_candidates"],
        "positions_skipped_high_value_gate": sums["positions_skipped_high_value_gate"],
        "forced_sims_total": sums["forced_sims_total"],
        "selected_forced_move_count": sums["selected_forced_move_count"],
        "selected_forced_move_rate": (
            (sums["selected_forced_move_count"] / triggered_total)
            if triggered_total > 0 else 0.0
        ),
        "post_force_endpoint_visit_top1_rate": (
            (sums["_top1_hits"] / triggered_total) if triggered_total > 0 else 0.0
        ),
        "post_force_endpoint_visit_top5_rate": (
            (sums["_top5_hits"] / triggered_total) if triggered_total > 0 else 0.0
        ),
    })
    return out


def play_game(
    evaluator: Evaluator,
    mcts_config: Optional[MCTSConfig] = None,
    rng: Optional[random.Random] = None,
    max_moves: int = 200,
    add_noise: bool = True,
    active_size: int = 24,
    start_player: Optional[str] = None,
    game_id: int = 0,
    # Resign parameters (conservative defaults = disabled)
    resign_enabled: bool = False,
    resign_min_ply: int = 80,
    resign_threshold: float = -0.97,
    resign_window: int = 12,
    resign_k: int = 8,
    resign_min_visits: int = 200,
    resign_min_top1_share: float = 0.0,  # Optional: require top move support
    # Adjudication-at-timeout parameters (disabled by default)
    adjudicate_enabled: bool = False,
    adjudicate_min_ply: int = 120,
    adjudicate_threshold: float = 0.90,
    adjudicate_min_visits: int = 200,
    adjudicate_min_top1_share: float = 0.0,
    adjudicate_debug: bool = False,
    # Phase 4: per-game replay cap (0/None disables; cap applied before returning)
    max_positions_per_game: Optional[int] = None,
    endgame_keep_positions: int = 16,
    # Phase 3 closeout diagnostics (spec 2026-05-03 §8). Defaults make
    # the diagnostic active by default; callers/tests can disable via the
    # emit_enabled flag. Six knobs configure threshold, depth, sizing,
    # safety cap, and a perf escape hatch for distance-reducing computation.
    goal_completion_emit_enabled: bool = True,
    goal_completion_emit_threshold: int = 3,
    goal_completion_emit_min_component: int = 8,
    goal_completion_max_depth: int = 3,
    goal_completion_skip_distance_reducing: bool = False,
    goal_completion_max_records_per_game: int = 64,
    # Compact per-game goal-completion record (spec 2026-05-05).
    # Independent of emit_enabled — emits even when Phase 3 detailed
    # diagnostics are disabled.
    goal_completion_record_enabled: bool = True,
    goal_completion_detection_threshold: int = 2,
    goal_completion_high_value_threshold: float = 0.9,
    goal_completion_high_value_delay_threshold_plies: int = 6,
    goal_completion_min_component_size: int = 8,
    # Spec 2: conversion auxiliary loss attach point (§5.5).
    # When enabled, attaches conversion metadata to PositionRecords whose
    # pre-move gc_state_full is conversion-eligible. Reuses gc_state_full
    # from the BFS-reuse contract (Spec 1.5); forces full BFS only when needed.
    conversion_policy_loss_enabled: bool = False,
    conversion_max_total_goal_distance: int = 2,
    # min_component_size reuses goal_completion_emit_min_component —
    # single source of truth, no duplicate flag.
) -> GameRecord:
    """Play one self-play game.

    Args:
        evaluator: Evaluator for MCTS leaf evaluation (implements Evaluator protocol)
        mcts_config: MCTS configuration (uses defaults if None)
        rng: Random number generator (creates new one if None)
        max_moves: Maximum moves before declaring draw
        add_noise: Whether to add Dirichlet noise at root (for training)
        active_size: Curriculum board size (default 24 = full board)
        start_player: "red" or "black" (default: random)
        resign_enabled: Enable resign logic (default: False)
        resign_min_ply: Don't resign before this ply (default: 80)
        resign_threshold: Resign when root_value <= this (default: -0.97)
        resign_window: Sliding window size for resign check (default: 12)
        resign_k: Resign if K of last W checks meet condition (default: 8)
        resign_min_visits: Require root.visit_count >= this (default: 200)
        resign_min_top1_share: Require top move's visit share >= this (default: 0.0 = disabled)
        adjudicate_enabled: Enable timeout adjudication (default: False)
        adjudicate_min_ply: Don't adjudicate before this ply (default: 120)
        adjudicate_threshold: Absolute root_value threshold for adjudication (default: 0.90)
        adjudicate_min_visits: Require root.visit_count >= this (default: 200)
        adjudicate_min_top1_share: Require top move's visit share >= this (default: 0.0 = disabled)
        goal_completion_record_enabled: Emit compact per-game goal_completion_record (default: True). Independent of emit_enabled — record emits even when Phase 3 detailed diagnostics are off (spec 2026-05-05 §6).
        goal_completion_detection_threshold: total_goal_distance threshold for tracker detection (default: 2). Must be <= goal_completion_emit_threshold (validated at startup; ValueError on violation).
        goal_completion_high_value_threshold: search_score threshold for "high-value" classification (default: 0.9).
        goal_completion_high_value_delay_threshold_plies: Conversion-delay threshold (in plies) for the root_value_high_but_delayed bad-case flag (default: 6).
        goal_completion_min_component_size: Minimum dominant-component size for tracker classification (default: 8).

    Returns:
        GameRecord with all positions and outcomes assigned
    """
    game_t0 = time.perf_counter()
    mcts_config = mcts_config or MCTSConfig()
    rng = rng or random.Random()

    mcts = MCTS(evaluator, mcts_config, rng)

    # Spec §12.1 invariant: detection_threshold must be <= emit_threshold
    # so post-detection plies are guaranteed to have full state available
    # when Phase 3 emit is enabled.
    if goal_completion_detection_threshold > goal_completion_emit_threshold:
        raise ValueError(
            "detection_threshold must be <= emit_threshold "
            f"(got {goal_completion_detection_threshold} > {goal_completion_emit_threshold})"
        )

    # Spec §6: per-game tracker. enabled=False short-circuits to no-op.
    from .goal_completion_tracker import GoalCompletionGameTracker
    gc_tracker = GoalCompletionGameTracker(
        enabled=goal_completion_record_enabled,
        detection_threshold=goal_completion_detection_threshold,
        high_value_threshold=goal_completion_high_value_threshold,
        high_value_delay_threshold_plies=goal_completion_high_value_delay_threshold_plies,
        max_depth=goal_completion_max_depth,
        min_component_size=goal_completion_min_component_size,
    )

    # Compute opening diagnostics window
    cfg = mcts.config
    _diag_end_ply, _diag_used_floor = compute_diagnostic_end_ply(
        cfg.root_edge_band_penalty_ply, cfg.root_near_corner_penalty_ply,
    )
    _opening_diags: List[dict] = []

    # Determine starting player (random if not specified)
    if start_player is None:
        start_player = "red" if rng.random() < 0.5 else "black"

    # Initialize state with active_size for curriculum
    state = TwixtState(
        active_size=active_size,
        to_move=start_player,
        max_plies_limit=max_moves,  # Unify cap with self-play loop
    )

    # Invariant: caps must match (catch divergence bugs early)
    assert state.max_plies_limit == max_moves, "State cap must match self-play cap"

    # Initialize root for tree reuse
    root = MCTSNode(state=state)

    # Resign tracking (K of last W sliding window)
    resign_window_hits: deque = deque(maxlen=resign_window)
    resigned_by: Optional[str] = None
    winner: Optional[str] = None
    draw_reason: Optional[str] = None
    adj_root_value = None     # Final MCTS eval at cap (for diagnostics)
    adj_top1_share = None     # Top-1 visit share at cap (for diagnostics)
    adj_attempted = False     # Whether adjudication was attempted (timeout + enabled)
    adj_blocked_by = None     # "ply", "threshold", "visits", "top1", or None if eligible
    adj_total_visits = None   # Total search visits at cap
    # Resign gate breakdown (by to_move color)
    rg_checks_red = 0;    rg_checks_black = 0
    rg_value_hits_red = 0; rg_value_hits_black = 0   # visits>=min AND value<=threshold
    rg_eligible_red = 0;   rg_eligible_black = 0     # value+visits+share all passed
    rg_top1_samples = []   # top1_share values on value_hits only (for percentiles)

    positions = []
    move_history = []
    # Per-move stats accumulators (spec 2026-05-03 §5). Appended at the same
    # point as move_history.append(move) so the resign branch (which breaks
    # without playing a move) does not add phantom entries.
    move_root_values: list = []
    move_top1_shares: list = []

    # Closeout diagnostics (spec 2026-05-03 §8). Best-effort, never raises
    # into the training path. Meta echoes config and tracks counters.
    goal_completion_diagnostics: list = []
    goal_completion_diagnostics_meta: Optional[dict] = None
    if goal_completion_emit_enabled:
        goal_completion_diagnostics_meta = {
            "enabled": True,
            "max_depth": goal_completion_max_depth,
            "emit_threshold": goal_completion_emit_threshold,
            "emit_min_component_size": goal_completion_emit_min_component,
            "max_records_per_game": goal_completion_max_records_per_game,
            "skip_distance_reducing": goal_completion_skip_distance_reducing,
            "diagnostic_version": 1,
            "computed_inline": True,
            "selection_perspective": "side_to_move",
            "storage": "in_game_json",
            "error_count": 0,
            "resign_dropped_partial_count": 0,
            "skipped_missing_priors_count": 0,
            "records_dropped_by_cap": 0,
        }

    ply = 0
    while not state.is_terminal() and ply < max_moves:
        # --- Compute gc_state once per ply, shared by Phase 3 + tracker.
        # This must happen BEFORE search_from_root so that Spec 3 Fix 1
        # (td=1 endpoint-completion visit forcing) can consume the current
        # board's gc_state_full inside the MCTS root search.
        gc_state_for_diag = None     # cheap: total_goal_distance only
        gc_state_full = None         # full: includes endpoint_completion_moves
        partial_diag = None
        total_now = None
        need_cheap = (
            goal_completion_emit_enabled
            or gc_tracker.enabled
            or conversion_policy_loss_enabled  # Spec 2: may need BFS for conversion
            or mcts.config.closeout_td1_visit_forcing_enabled  # Spec 3 Fix 1
        )
        if need_cheap:
            try:
                from .connectivity_diagnostics import compute_goal_completion_state
                gc_state_for_diag = compute_goal_completion_state(
                    state, state.to_move,
                    max_depth=goal_completion_max_depth,
                    min_component_size=goal_completion_emit_min_component,
                    enumerate_moves=False,
                )
            except Exception as _e:
                if goal_completion_diagnostics_meta is not None:
                    goal_completion_diagnostics_meta["error_count"] += 1
                import sys as _sys
                _sys.stderr.write(f"[gc-cheap] ply={ply} error: {_e!r}\n")

            # Decide whether to upgrade to gc_state_full (spec §8.2).
            total_now = (
                gc_state_for_diag.get("total_goal_distance")
                if gc_state_for_diag is not None else None
            )
            needs_phase3_full = (
                goal_completion_emit_enabled
                and gc_state_for_diag is not None
                and total_now is not None
                and total_now <= goal_completion_emit_threshold
            )
            needs_tracker_full = (
                gc_tracker.enabled
                and gc_state_for_diag is not None
                and total_now is not None
                and (
                    gc_tracker.is_detected(state.to_move)
                    or total_now <= gc_tracker.detection_threshold
                )
            )
            # Spec 2 §5.5: conversion forces full BFS when ply is eligible.
            needs_conversion_full = (
                conversion_policy_loss_enabled
                and gc_state_for_diag is not None
                and total_now is not None
                and total_now <= conversion_max_total_goal_distance
            )
            # Spec 3 Fix 1: td=1 root visit forcing needs enumerated
            # endpoint_completion_moves, available only on the full state.
            needs_closeout_td1_full = (
                mcts.config.closeout_td1_visit_forcing_enabled
                and gc_state_for_diag is not None
                and total_now is not None
                and total_now == 1
            )
            if (needs_phase3_full or needs_tracker_full
                    or needs_conversion_full or needs_closeout_td1_full):
                try:
                    gc_state_full = compute_goal_completion_state(
                        state, state.to_move,
                        max_depth=goal_completion_max_depth,
                        min_component_size=goal_completion_emit_min_component,
                        enumerate_moves=True,
                    )
                except Exception as _e:
                    if goal_completion_diagnostics_meta is not None:
                        goal_completion_diagnostics_meta["error_count"] += 1
                    import sys as _sys
                    _sys.stderr.write(f"[gc-full] ply={ply} error: {_e!r}\n")

        # Run MCTS search from current root (reuses subtree)
        visit_counts, root_value, root = mcts.search_from_root(
            root, add_noise=add_noise, ply=ply, gc_state_full=gc_state_full,
        )

        # Build opening diagnostic record if within window
        if ply < _diag_end_ply and root.priors_raw is not None:
            _rec = build_root_diagnostic(
                ply=ply,
                side_to_move=state.to_move,
                visit_counts=visit_counts,
                priors_raw=root.priors_raw,
                priors_adjusted=root.priors,
                board_size=active_size,
                band_width=cfg.root_edge_band_width,
                corner_radius=cfg.root_near_corner_radius,
                edge_penalty=cfg.root_edge_band_penalty,
                corner_penalty=cfg.root_near_corner_penalty,
                edge_penalty_ply=cfg.root_edge_band_penalty_ply,
                corner_penalty_ply=cfg.root_near_corner_penalty_ply,
                corner_penalty_early=cfg.root_near_corner_penalty_early,
                corner_penalty_early_plies=cfg.root_near_corner_penalty_early_plies,
                decode_fn=decode_move,
            )
            # Attach deep per-child details for the earliest plies only
            if ply < CHILD_DETAIL_PLIES:
                _child = build_root_child_details(
                    root=root,
                    c_puct=cfg.c_puct,
                    board_size=active_size,
                    band_width=cfg.root_edge_band_width,
                    corner_radius=cfg.root_near_corner_radius,
                    top_k=CHILD_DETAIL_TOPK,
                    decode_fn=decode_move,
                )
                _rec["root_summary"] = _child["root_summary"]
                _rec["top_children"] = _child["top_children"]
            _opening_diags.append(_rec)

        # Phase 3 partial build uses gc_state_full (computed pre-search above)
        # plus post-search visit_counts / root priors.
        if (goal_completion_emit_enabled
                and gc_state_full is not None
                and total_now is not None
                and total_now <= goal_completion_emit_threshold):
            if (goal_completion_diagnostics_meta is not None
                    and len(goal_completion_diagnostics) >= goal_completion_max_records_per_game):
                goal_completion_diagnostics_meta["records_dropped_by_cap"] += 1
            elif root.priors_raw is None:
                if goal_completion_diagnostics_meta is not None:
                    goal_completion_diagnostics_meta["skipped_missing_priors_count"] += 1
            else:
                try:
                    from .closeout_diagnostics import build_closeout_diagnostic_partial
                    _decode_fn = lambda mid, _a=active_size: (mid // _a, mid % _a)
                    partial_diag = build_closeout_diagnostic_partial(
                        ply=ply,
                        side_to_move=state.to_move,
                        visit_counts=visit_counts,
                        priors_raw=root.priors_raw,
                        priors_adjusted=getattr(root, "priors", None),
                        root=root,
                        goal_completion_state=gc_state_full,
                        board_size=active_size,
                        skip_distance_reducing=goal_completion_skip_distance_reducing,
                        decode_fn=_decode_fn,
                    )
                except Exception as _e:
                    if goal_completion_diagnostics_meta is not None:
                        goal_completion_diagnostics_meta["error_count"] += 1
                    import sys as _sys
                    _sys.stderr.write(f"[closeout-diag] ply={ply} partial: {_e!r}\n")

        # --- RESIGN CHECK (after search, before move selection) ---
        # root_value is from state.to_move perspective:
        #   +1 = to_move winning, -1 = to_move losing
        if resign_enabled and ply >= resign_min_ply:
            is_red = (state.to_move == "red")
            if is_red:
                rg_checks_red += 1
            else:
                rg_checks_black += 1

            # Always compute top1_share on value hits (for distribution tracking)
            value_visits_ok = (root.visit_count >= resign_min_visits
                               and root_value <= resign_threshold)

            if value_visits_ok:
                total_visits = sum(visit_counts.values())
                top1_visits = max(visit_counts.values()) if visit_counts else 0
                top1_share = top1_visits / total_visits if total_visits > 0 else 0
                rg_top1_samples.append(top1_share)

                if is_red:
                    rg_value_hits_red += 1
                else:
                    rg_value_hits_black += 1

                # Share gate
                if resign_min_top1_share > 0:
                    share_ok = top1_share >= resign_min_top1_share
                else:
                    share_ok = True

                condition_met = share_ok
            else:
                condition_met = False

            if condition_met:
                if is_red:
                    rg_eligible_red += 1
                else:
                    rg_eligible_black += 1

            resign_window_hits.append(1 if condition_met else 0)
            window_sum = sum(resign_window_hits)

            if window_sum >= resign_k:
                # Phase 3: track partials built this ply but dropped because
                # the resign branch breaks before finalize. The partial would
                # otherwise be discarded silently — count it for transparency.
                if (partial_diag is not None
                        and goal_completion_diagnostics_meta is not None):
                    goal_completion_diagnostics_meta["resign_dropped_partial_count"] += 1
                resigned_by = state.to_move
                winner = opponent(resigned_by)
                draw_reason = RESIGN
                break

        # Record position with explicit to_move
        moves = list(visit_counts.keys())
        counts = [visit_counts[m] for m in moves]

        # Convert board tensor from (C, H, W) to (H, W, C) for NHWC storage
        # This avoids transpose overhead during training
        board_chw = state.to_tensor()  # (C, H, W)
        board_hwc = np.transpose(board_chw, (1, 2, 0))  # (H, W, C)

        # Spec 2 §5.5: build conversion metadata using the SAME gc_state_full above.
        # Must be done BEFORE move selection (pre-move semantics).
        conversion_meta = None
        if conversion_policy_loss_enabled and gc_state_full is not None:
            from .conversion_loss import is_conversion_eligible
            if is_conversion_eligible(
                gc_state_full,
                max_total_goal_distance=conversion_max_total_goal_distance,
                min_component_size=goal_completion_emit_min_component,
            ):
                conversion_meta = {
                    "version": 1,
                    "total_goal_distance":       gc_state_full.get("total_goal_distance"),
                    "largest_component_size":    gc_state_full.get("largest_component_size"),
                    "endpoint_completion_moves": [list(m) for m in (gc_state_full.get("endpoint_completion_moves") or [])],
                    "distance_reducing_moves":   [list(m) for m in (gc_state_full.get("distance_reducing_moves") or [])],
                    "conversion_category":       gc_state_full.get("category"),
                    "selected_primary_class":    None,    # telemetry-only; deferred
                }

        positions.append(
            PositionRecord(
                board_tensor=board_hwc,  # (H, W, C) NHWC format
                to_move=state.to_move,  # Explicit, not inferred from ply
                legal_moves=moves,
                visit_counts=counts,  # Raw counts, not normalized
                active_size=active_size,  # Store for training with masked pooling
                ply=ply,  # Ply at which this position occurred
                conversion=conversion_meta,  # Spec 2: None unless conversion-eligible
            )
        )

        # Probabilistic mirror augmentation
        if _MIRROR_PROB > 0 and rng.random() < _MIRROR_PROB:
            m_board, m_moves, m_counts = _mirror_position_lr(
                board_hwc, moves, counts, active_size
            )
            # Spec 2: mirrored positions intentionally drop conversion metadata
            # (defaults to None). Completion/reducing move coordinates would need
            # column-flip (col -> active_size - 1 - col) to remain valid; deferred.
            positions.append(
                PositionRecord(
                    board_tensor=m_board,
                    to_move=state.to_move,
                    legal_moves=m_moves,
                    visit_counts=m_counts,
                    active_size=active_size,
                    ply=ply,  # Mirror shares the same ply as the primary
                )
            )

        # Select move
        move = mcts.select_move(visit_counts, ply)

        if (
            _OPENING_DEBUG
            and game_id < _OPENING_DEBUG_GAMES
            and ply < _OPENING_DEBUG_PLIES
        ):
            n_sims = getattr(mcts.config, "n_simulations", None)
            _log_opening_debug(
                game_id, ply, state.to_move,
                n_sims, root_value,
                visit_counts, root.priors_raw, root.priors, move,
            )

        # Tracker observes pre-move state. Independent of Phase 3 emit.
        if gc_tracker.enabled:
            # search_score (root_value) is from state.to_move's perspective.
            _ss = float(root_value) if root_value is not None else None
            try:
                gc_tracker.observe_pre_move(
                    state=state,
                    ply=ply + 1,                     # tracker uses 1-indexed ply
                    side_to_move=state.to_move,
                    selected_move=move,
                    search_score=_ss,
                    gc_state_cheap=gc_state_for_diag,
                    gc_state_full=gc_state_full,
                )
            except Exception as _e:
                import sys as _sys
                _sys.stderr.write(
                    f"[gc-tracker] ply={ply} observe error: {_e!r}\n"
                )

        # --- Phase 3: finalize closeout diagnostic if partial was built ---
        # Must run BEFORE state is advanced to root.state so state.to_move
        # still reflects the side that selected `move`.
        if partial_diag is not None and gc_state_full is not None:
            try:
                from .closeout_diagnostics import finalize_closeout_diagnostic
                full_diag = finalize_closeout_diagnostic(
                    partial_diag,
                    state_before=state,
                    player=state.to_move,
                    selected_move=move,
                    goal_state_before=gc_state_full,
                )
                goal_completion_diagnostics.append(full_diag)
            except Exception as _e:
                if goal_completion_diagnostics_meta is not None:
                    goal_completion_diagnostics_meta["error_count"] += 1
                import sys as _sys
                _sys.stderr.write(
                    f"[closeout-diag] ply={ply} finalize error: {_e!r}\n"
                )

        # TREE REUSE: advance root to chosen child
        root = mcts.advance_root(root, move)

        # SYNC: state comes from root (don't apply_move twice!)
        state = root.state
        # Capture per-move root value and top1 share before move-history append.
        # IMPORTANT: search_score (root_value) is ALWAYS from state.to_move
        # perspective at search time -- i.e., +1 means "side about to play this
        # move thinks it is winning." Phase 2's analyzer aggregation only
        # looks at winner-to-move plies in the watch window, so no sign flip
        # is needed there. If a future enhancement ever adds loser-side
        # analysis, that path MUST flip the sign for non-winner plies.
        move_root_values.append(float(root_value) if root_value is not None else None)
        if visit_counts:
            _total = sum(visit_counts.values())
            _top1  = max(visit_counts.values())
            move_top1_shares.append(float(_top1 / _total) if _total > 0 else None)
        else:
            move_top1_shares.append(None)
        move_history.append(move)
        ply += 1

    # Compute terminal status (only if not resigned)
    is_timeout = (ply >= max_moves)
    is_terminal = state.is_terminal()

    # Resign already set winner/draw_reason; handle normal endings
    if resigned_by is None:
        winner = state.winner() if is_terminal else None

        if winner is None:
            # No winner - determine draw reason
            # Check ply first (authoritative for timeout)
            if is_timeout:
                # INVARIANT: adjudicate only when winner is None
                # (guaranteed here by outer `if winner is None:` guard)
                draw_reason = None  # Reset before adjudication attempt
                # --- ADJUDICATE TIMEOUT (optional) ---
                if adjudicate_enabled:
                    # Fresh root for adjudication (no tree reuse -- cumulative
                    # q_value/visit_count from the game loop would be stale).
                    # Must remove max_plies_limit because at timeout the state
                    # has ply >= max_plies_limit, making is_terminal() True and
                    # preventing any search (all sims short-circuit immediately).
                    adj_state = state.copy()
                    adj_state.max_plies_limit = None
                    adj_root0 = MCTSNode(state=adj_state)
                    # Spec 3 Fix 1: adjudication rerun does not pre-compute a
                    # fresh gc_state_full for adj_state, so pass None to skip
                    # td=1 visit forcing here. This path is non-training and
                    # the spec scopes Fix 1 to self-play only.
                    adj_visit_counts, adj_root_value, adj_root = mcts.search_from_root(
                        adj_root0, add_noise=False, ply=ply, gc_state_full=None,
                    )

                    # Compute from fresh search results (not cumulative root)
                    total_visits = sum(adj_visit_counts.values()) if adj_visit_counts else 0
                    top1_visits = max(adj_visit_counts.values()) if adj_visit_counts else 0
                    adj_top1_share = (top1_visits / total_visits) if total_visits > 0 else 0.0

                    # Compute ALL gates explicitly (even if disabled)
                    ply_ok = (ply >= adjudicate_min_ply)
                    thr_ok = (abs(adj_root_value) >= adjudicate_threshold)
                    visits_ok = (total_visits >= adjudicate_min_visits)
                    top1_ok = (adj_top1_share >= adjudicate_min_top1_share) if adjudicate_min_top1_share > 0 else True
                    eligible = ply_ok and thr_ok and visits_ok and top1_ok

                    # Deterministic "first failure" label
                    if not ply_ok:
                        blocked_by = "ply"
                    elif not thr_ok:
                        blocked_by = "threshold"
                    elif not visits_ok:
                        blocked_by = "visits"
                    elif not top1_ok:
                        blocked_by = "top1"
                    else:
                        blocked_by = None

                    # Winner mapping (for debug, even if not eligible)
                    if adj_root_value >= adjudicate_threshold:
                        winner_if = state.to_move
                    elif adj_root_value <= -adjudicate_threshold:
                        winner_if = opponent(state.to_move)
                    else:
                        winner_if = None

                    # ADJ_DEBUG (only when --adjudicate-debug)
                    if adjudicate_debug:
                        status = "ELIGIBLE" if eligible else f"BLOCKED({blocked_by})"
                        wif = f" winner_if={winner_if}" if winner_if else ""
                        print(
                            f"  ADJ_DEBUG: ply={ply} to_move={state.to_move} "
                            f"rv={adj_root_value:.3f} abs={abs(adj_root_value):.3f} "
                            f"total={total_visits} top1={adj_top1_share:.3f} "
                            f"| req: minply={adjudicate_min_ply} thr={adjudicate_threshold} "
                            f"minv={adjudicate_min_visits} mintop1={adjudicate_min_top1_share} "
                            f"| ok: ply={'Y' if ply_ok else 'N'} thr={'Y' if thr_ok else 'N'} "
                            f"v={'Y' if visits_ok else 'N'} t1={'Y' if top1_ok else 'N'} "
                            f"=> {status}{wif}"
                        )

                    # Store diagnostics for GameRecord / trainer aggregation
                    adj_attempted = True
                    adj_blocked_by = blocked_by
                    adj_total_visits = total_visits

                    # Decide winner if all gates pass
                    if eligible:
                        if adj_root_value >= adjudicate_threshold:
                            winner = state.to_move
                            draw_reason = ADJUDICATED
                        elif adj_root_value <= -adjudicate_threshold:
                            winner = opponent(state.to_move)
                            draw_reason = ADJUDICATED

                # Fall back to state_cap if not adjudicated (or adjudication disabled/skipped).
                # Reaching ply >= max_moves means we hit the per-game ply cap; that is
                # semantically "state cap exhausted," not a runtime timeout. DRAW_TIMEOUT
                # is reserved for genuine runtime/session timeouts (currently unused in
                # play_game; kept as a distinct category for future use).
                if draw_reason is None:
                    draw_reason = DRAW_STATE_CAP
            elif is_terminal:
                # State is terminal but no winner - why?
                if not state.legal_moves():
                    draw_reason = DRAW_BOARD_FULL
                elif state.max_plies_limit is not None and state.ply >= state.max_plies_limit:
                    draw_reason = DRAW_STATE_CAP
                else:
                    draw_reason = DRAW_UNKNOWN

    # Assign outcomes to positions (from perspective of to_move at each position)
    for pos in positions:
        if winner is None:
            pos.outcome = 0.0  # Draw
        elif winner == pos.to_move:
            pos.outcome = 1.0  # Player at this position won
        else:
            pos.outcome = -1.0  # Player at this position lost
        pos.game_n_moves = ply  # ply is the total plies played in this game

    # Phase 4: apply per-game replay cap (no-op if disabled / game short enough)
    # MUST run AFTER outcome assignment so kept positions already carry outcomes.
    n_positions_original = len(positions)
    positions, _n_orig, n_positions_kept = apply_game_position_cap(
        positions,
        max_positions_per_game=max_positions_per_game,
        endgame_keep_positions=endgame_keep_positions,
        rng=rng,
    )

    # Diagnostic: print timeout trace
    if draw_reason == DRAW_TIMEOUT:
        last_moves = move_history[-10:] if len(move_history) >= 10 else move_history
        print(f"  TIMEOUT: plies={ply}, last10={last_moves}")

    if draw_reason == ADJUDICATED and adj_root_value is not None:
        top1_str = f", top1={adj_top1_share:.2f}" if adj_top1_share is not None else ""
        print(f"  ADJUDICATED: plies={ply}, to_move={state.to_move}, winner={winner}, root_value={adj_root_value:.3f}{top1_str}")

    # Invariant: per-move accumulators must be 1:1 with move_history on
    # every code path, including resign. The saver tolerates mismatch
    # defensively (with a stderr warning) -- in-process production code
    # should not produce a mismatch.
    assert len(move_root_values) == len(move_history), (
        f"move_root_values length {len(move_root_values)} != "
        f"move_history length {len(move_history)}"
    )
    assert len(move_top1_shares) == len(move_history), (
        f"move_top1_shares length {len(move_top1_shares)} != "
        f"move_history length {len(move_history)}"
    )

    # Compact goal-completion record (spec §6).
    _gc_reason_for_record = "win"
    if winner is None:
        if draw_reason in ("terminal_state_cap",):
            _gc_reason_for_record = "state_cap"
        elif draw_reason in ("timeout_selfplay",):
            _gc_reason_for_record = "timeout"
        elif draw_reason in ("terminal_board_full",):
            _gc_reason_for_record = "board_full"
        else:
            _gc_reason_for_record = "unknown"
    elif resigned_by is not None:
        _gc_reason_for_record = "resign"
    gc_record = gc_tracker.finalize_game(
        winner=winner,
        reason=_gc_reason_for_record,
        n_moves=len(move_history),
        starting_player=start_player,
        iteration=0,                        # populated downstream by trainer/saver path
        game_idx=game_id,
        game_id=f"game_{game_id:03d}",
    )

    return GameRecord(
        positions=positions,
        winner=winner,
        n_moves=ply,  # ply is authoritative (not state.ply)
        move_history=move_history,
        start_player=start_player,  # Needed for correct replay attribution
        draw_reason=draw_reason,
        resigned_by=resigned_by,  # Who resigned (or None)
        nn_calls=mcts._nn_call_count,
        expand_calls=mcts._expand_calls,
        nn_batches=mcts._nn_batches,
        total_backups=mcts._total_backups,
        total_waiters=mcts._total_waiters_backed_up,
        unique_leaves=mcts._unique_leaves_expanded,
        max_waiters=mcts._max_waiters_on_any_leaf,
        flush_full=mcts._flush_full,
        flush_stall=mcts._flush_stall,
        flush_tail=mcts._flush_tail,
        adj_attempted=adj_attempted,
        adj_blocked_by=adj_blocked_by,
        adj_abs_rv=abs(adj_root_value) if adj_root_value is not None else None,
        adj_top1=adj_top1_share,
        adj_total_visits=adj_total_visits,
        rg_checks_red=rg_checks_red,
        rg_checks_black=rg_checks_black,
        rg_value_hits_red=rg_value_hits_red,
        rg_value_hits_black=rg_value_hits_black,
        rg_eligible_red=rg_eligible_red,
        rg_eligible_black=rg_eligible_black,
        rg_top1_samples=tuple(rg_top1_samples),
        opening_diagnostics=_opening_diags,
        opening_diagnostics_meta={
            "version": 3,
            "diagnostic_end_ply": _diag_end_ply,
            "extra_plies_after_penalty": 2,
            "floor_min_ply": 4,
            "used_floor": _diag_used_floor,
            "child_detail_plies": CHILD_DETAIL_PLIES,
            "child_detail_topk": CHILD_DETAIL_TOPK,
            # Phase 2: run-level near-corner config echo (required #1).
            # Individual per-root records also carry their own `config` block;
            # this top-level copy exists so consumers can read the run setup
            # without scanning the per-ply records.
            "near_corner_penalty": cfg.root_near_corner_penalty,
            "near_corner_penalty_ply": cfg.root_near_corner_penalty_ply,
            "near_corner_penalty_early": cfg.root_near_corner_penalty_early,
            "near_corner_penalty_early_plies": cfg.root_near_corner_penalty_early_plies,
            "near_corner_radius": cfg.root_near_corner_radius,
        },
        n_positions_original=n_positions_original,
        n_positions_kept=n_positions_kept,
        # Per-game stats persistence (spec 2026-04-29)
        wall_time_s=time.perf_counter() - game_t0,
        final_root_value=mcts._final_root_value,
        final_top1_share=mcts._final_top1_share,
        move_root_values=move_root_values,
        move_top1_shares=move_top1_shares,
        goal_completion_diagnostics=goal_completion_diagnostics,
        goal_completion_diagnostics_meta=goal_completion_diagnostics_meta,
        goal_completion_record=gc_record,
        # Spec 3 Fix 1: snapshot per-game closeout_td1 visit-forcing telemetry.
        # Safe to call even when the feature is off (returns config echo + zeros).
        closeout_td1_telemetry=mcts.get_closeout_td1_telemetry(),
    )


def play_games(
    evaluator: Evaluator,
    n_games: int,
    mcts_config: Optional[MCTSConfig] = None,
    seed: Optional[int] = None,
    max_moves: int = 200,
    add_noise: bool = True,
    progress_callback=None,
    active_size: int = 24,
    max_positions_per_game: Optional[int] = None,
    endgame_keep_positions: int = 16,
) -> List[GameRecord]:
    """Play multiple self-play games.

    Args:
        evaluator: Evaluator for MCTS leaf evaluation (implements Evaluator protocol)
        n_games: Number of games to play
        mcts_config: MCTS configuration
        seed: Random seed for reproducibility
        max_moves: Maximum moves per game
        add_noise: Whether to add Dirichlet noise
        progress_callback: Optional callback(game_idx, game_record) for progress
        active_size: Curriculum board size (default 24 = full board)

    Returns:
        List of GameRecord objects
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    games = []

    for i in range(n_games):
        # Create new RNG for each game (seeded from main RNG for reproducibility)
        game_rng = random.Random(rng.randint(0, 2**31))

        # Randomize starting player for each game
        start_player = "red" if game_rng.random() < 0.5 else "black"

        game = play_game(
            evaluator,
            mcts_config=mcts_config,
            rng=game_rng,
            max_moves=max_moves,
            add_noise=add_noise,
            active_size=active_size,
            start_player=start_player,
            game_id=i,
            max_positions_per_game=max_positions_per_game,
            endgame_keep_positions=endgame_keep_positions,
        )
        games.append(game)

        # Clear Python refs after each game
        gc.collect()

        if progress_callback:
            progress_callback(i, game)

    return games
