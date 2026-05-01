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
    """

    board_tensor: np.ndarray  # (H, W, C) numpy array - NHWC format
    to_move: str  # "red" or "black"
    legal_moves: List[Tuple[int, int]]
    visit_counts: List[int]
    outcome: Optional[float] = None
    active_size: int = 24  # Curriculum board size
    ply: int = 0                        # ply at which this position occurred
    game_n_moves: Optional[int] = None  # total plies in the source game (set in outcome loop)

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

    Returns:
        GameRecord with all positions and outcomes assigned
    """
    game_t0 = time.perf_counter()
    mcts_config = mcts_config or MCTSConfig()
    rng = rng or random.Random()

    mcts = MCTS(evaluator, mcts_config, rng)

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

    ply = 0
    while not state.is_terminal() and ply < max_moves:
        # Run MCTS search from current root (reuses subtree)
        visit_counts, root_value, root = mcts.search_from_root(
            root, add_noise=add_noise, ply=ply
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

        positions.append(
            PositionRecord(
                board_tensor=board_hwc,  # (H, W, C) NHWC format
                to_move=state.to_move,  # Explicit, not inferred from ply
                legal_moves=moves,
                visit_counts=counts,  # Raw counts, not normalized
                active_size=active_size,  # Store for training with masked pooling
                ply=ply,  # Ply at which this position occurred
            )
        )

        # Probabilistic mirror augmentation
        if _MIRROR_PROB > 0 and rng.random() < _MIRROR_PROB:
            m_board, m_moves, m_counts = _mirror_position_lr(
                board_hwc, moves, counts, active_size
            )
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

        # TREE REUSE: advance root to chosen child
        root = mcts.advance_root(root, move)

        # SYNC: state comes from root (don't apply_move twice!)
        state = root.state
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
                    adj_visit_counts, adj_root_value, adj_root = mcts.search_from_root(
                        adj_root0, add_noise=False, ply=ply
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
