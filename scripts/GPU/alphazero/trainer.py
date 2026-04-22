"""Training loop for AlphaZero.

This module provides:
- alphazero_loss(): Combined policy + value + L2 loss
- train_step(): Single gradient update
- ReplayBuffer: Fixed-size buffer with uniform sampling
- train(): Full orchestrator with checkpointing

Key conventions:
- Visit counts are normalized to policy targets (sum to 1)
- Outcomes are from perspective of to_move at each position
- Board tensors are transposed from (C,H,W) to (H,W,C) for MLX
"""
from __future__ import annotations

import csv
import gc
import json
import multiprocessing as mp
import os
import platform
import queue
import random
import subprocess
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .network import AlphaZeroNetwork, create_network
from .curriculum import CurriculumManager
from .game_saver import GameSaver
from .opening_diagnostics import (
    aggregate_opening_diagnostics,
    aggregate_root_child_details,
    build_early_override_summary,
    compute_diagnostic_end_ply,
)


class MainModule(nn.Module):
    """Holds references to the live encoder + policy_head modules.

    Used for two-optimizer training: opt_main updates this module (encoder + policy),
    while opt_value updates network.value_head separately.

    Why a wrapper instead of two update() calls on separate modules?
    If MLX increments Adam's internal step counter per update() call, doing
    opt_main.update(encoder, ...) then opt_main.update(policy_head, ...) would
    effectively "double-step" the optimizer each training step. This wrapper
    avoids that entire class of bug.
    """
    def __init__(self, encoder: nn.Module, policy_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.policy_head = policy_head

# Per-size max moves table (tuned to give Black time to convert)
# NOTE: These are PLIES (half-moves), not full moves. TwixT alternates turns,
# so 200 plies = 100 moves per player at size 16.
MAX_MOVES_TABLE = {
    8: 90,
    10: 110,
    12: 160,
    16: 200,
    20: 250,
    24: 380,
}

# Per-size simulation counts (balances quality vs throughput)
# Larger boards get fewer sims to prevent timeout explosion
SIMS_TABLE = {
    8: 400,
    10: 400,
    12: 300,
    16: 200,
    20: 150,
    24: 400,
}

# Per-size training steps (larger boards need more gradient updates)
TRAIN_STEPS_TABLE = {
    8: 100,
    10: 100,
    12: 120,
    16: 140,
    20: 160,
    24: 160,
}

# Metrics schema version for forward compatibility
METRICS_SCHEMA_VERSION = 1

# Value head health thresholds (pre-tanh magnitude)
VALUE_P99_CAUTION = 4.0
VALUE_P99_WARN = 4.6
VALUE_P99_CRIT = 5.2
VALUE_SAT_CAUTION = 0.02
VALUE_SAT_WARN = 0.05
VALUE_SAT_CRIT = 0.15

# Value head rolling window for instability detection
VALUE_WINDOW = 20
VALUE_WARN_FRACTION = 0.50  # warn if >= 50% of eligible iters are warn-level
VALUE_MIN_ELIGIBLE = 10     # don't evaluate rolling logic until this many eligible iters
VALUE_MIN_SAMPLES = 128     # minimum sanity sample count to be eligible


def clip_grad_norm(grads: Any, max_norm: float = 1.0) -> Tuple[Any, mx.array]:
    """Clip gradients by global norm (memory-efficient, no concat).

    Args:
        grads: Nested dict/list/tuple of gradients from nn.value_and_grad()
        max_norm: Maximum allowed global norm

    Returns:
        Tuple of (clipped_grads, global_norm) - global_norm useful for diagnostics
    """
    sumsq = mx.array(0.0, dtype=mx.float32)

    def accum(g):
        nonlocal sumsq
        if g is None:
            return  # Skip None gradients
        if isinstance(g, mx.array):
            gg = g.astype(mx.float32)
            sumsq = sumsq + mx.sum(gg * gg)
        elif isinstance(g, dict):
            for v in g.values():
                accum(v)
        elif isinstance(g, (list, tuple)):
            for v in g:
                accum(v)

    accum(grads)
    global_norm = mx.sqrt(sumsq)
    scale = mx.minimum(
        mx.array(1.0, dtype=mx.float32),
        mx.array(max_norm, dtype=mx.float32) / (global_norm + 1e-8)
    )

    def apply_scale(g):
        if g is None:
            return None  # Preserve None gradients
        if isinstance(g, mx.array):
            return g * scale
        elif isinstance(g, dict):
            return {k: apply_scale(v) for k, v in g.items()}
        elif isinstance(g, (list, tuple)):
            scaled = [apply_scale(v) for v in g]
            return type(g)(scaled)  # preserve list/tuple type
        return g

    return apply_scale(grads), global_norm


def first_array_leaf(tree):
    """Return first mx.array found in nested dict/list/tuple tree.

    Used for one-time verification that optimizer updates land in the live model.
    """
    if tree is None:
        return None
    if isinstance(tree, mx.array):
        return tree
    if isinstance(tree, dict):
        for v in tree.values():
            out = first_array_leaf(v)
            if out is not None:
                return out
    if isinstance(tree, (list, tuple)):
        for v in tree:
            out = first_array_leaf(v)
            if out is not None:
                return out
    return None


def generate_run_id() -> str:
    """Generate unique run ID: ISO timestamp + short random suffix."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{ts}_{suffix}"


def get_scaled_max_moves(active_size: int, fallback_mult: float = 10.0) -> int:
    """Get max moves for curriculum size."""
    return MAX_MOVES_TABLE.get(active_size, int(fallback_mult * active_size))


def get_scaled_simulations(active_size: int, cli_sims: Optional[int] = None) -> int:
    """Get simulations for curriculum size.

    Precedence: CLI overrides table when specified; otherwise use table defaults.
    - If cli_sims is None: use SIMS_TABLE[size] (default 400 if size not in table)
    - If cli_sims is set: use that value directly
    """
    if cli_sims is not None:
        return cli_sims
    return SIMS_TABLE.get(active_size, 400)


def get_scaled_train_steps(active_size: int, base_steps: Optional[int] = None) -> int:
    """Get train steps for curriculum size.

    Args:
        active_size: Current curriculum board size
        base_steps: CLI override. None=use table, >0=use CLI value, 0=skip training

    Returns:
        Number of training steps (0 means skip)
    """
    if base_steps == 0:
        return 0  # Explicit skip via --train-steps 0
    if base_steps is not None:
        return base_steps  # User specified a value, use it
    return TRAIN_STEPS_TABLE.get(active_size, 160)  # Default from table


def get_value_health_level(
    p99: Optional[float], frac_sat: Optional[float], n_samples: int
) -> Tuple[bool, str, str]:
    """
    Determine value head health level.

    Returns:
        (eligible, level, trigger) where:
        - level is one of: "ok", "caution", "warn", "crit", "skip"
        - trigger is one of: "p99", "sat", "both", "" (empty if ok/skip)
    """
    # Check eligibility - "skip" is never treated as warn/crit
    if n_samples < VALUE_MIN_SAMPLES or p99 is None or frac_sat is None:
        return False, "skip", ""

    # Determine level and trigger (check crit first, then warn, then caution)
    p99_crit = p99 >= VALUE_P99_CRIT
    sat_crit = frac_sat >= VALUE_SAT_CRIT
    if p99_crit or sat_crit:
        trigger = "both" if (p99_crit and sat_crit) else ("p99" if p99_crit else "sat")
        return True, "crit", trigger

    p99_warn = p99 >= VALUE_P99_WARN
    sat_warn = frac_sat >= VALUE_SAT_WARN
    if p99_warn or sat_warn:
        trigger = "both" if (p99_warn and sat_warn) else ("p99" if p99_warn else "sat")
        return True, "warn", trigger

    p99_caution = p99 >= VALUE_P99_CAUTION
    sat_caution = frac_sat >= VALUE_SAT_CAUTION
    if p99_caution or sat_caution:
        trigger = "both" if (p99_caution and sat_caution) else ("p99" if p99_caution else "sat")
        return True, "caution", trigger

    return True, "ok", ""


if TYPE_CHECKING:
    from .self_play import GameRecord, PositionRecord

# Import draw reason constants for tracking
from .self_play import (
    DRAW_TIMEOUT, DRAW_BOARD_FULL, DRAW_STATE_CAP, DRAW_UNKNOWN, RESIGN, ADJUDICATED,
    CHILD_DETAIL_PLIES,
)


def flatten_params(params, prefix=""):
    """Flatten nested parameter dict to list of (name, array) pairs."""
    result = []
    if isinstance(params, dict):
        for k, v in params.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            result.extend(flatten_params(v, new_prefix))
    elif isinstance(params, list):
        for i, v in enumerate(params):
            new_prefix = f"{prefix}[{i}]"
            result.extend(flatten_params(v, new_prefix))
    elif isinstance(params, mx.array):
        result.append((prefix, params))
    return result


# Absolute sims floor (minimum simulations regardless of freeze factor)
ABS_SIMS_FLOOR = 100

# CSV field order for metrics persistence (authoritative list)
CSV_FIELDNAMES = [
    # Identity
    "schema_version", "run_id", "row_id", "timestamp", "iteration", "active_size", "max_moves",
    # Config
    "games_per_iter", "simulations", "train_steps_per_iteration", "batch_size", "buffer_size_limit",
    "mcts_eval_batch_size", "mcts_pending_virtual_visits", "mcts_stall_flush_sims",
    "network_hidden", "network_blocks",
    # Sims transparency
    "requested_sims_cli", "base_sims_table", "effective_sims_used",
    "sims_clamped_to_floor", "effective_reason",
    # Self-play
    "games_generated", "positions_added", "buffer_size_end", "avg_plies",
    "p95_plies", "max_plies_observed", "avg_game_seconds", "p95_game_seconds",
    # Results
    "red_wins", "black_wins", "draws",
    "timeout_draws", "board_full_draws", "state_cap_draws", "unknown_draws",
    # MCTS
    "total_backups", "leaf_evals", "nn_batches", "avg_batch",
    "avg_waiters", "max_waiters",
    "flush_full", "flush_stall", "flush_tail",
    # Training
    "avg_total_loss", "avg_policy_loss", "avg_value_loss", "avg_l2_loss",
    # Curriculum
    "draw_rate_true", "timeout_rate", "draw_rate_timeout", "promoted_this_iter",
    "curriculum_frozen", "sims_used", "sims_next", "sims_reduction_factor",
    # Promotion/demotion
    "promotion_allowed", "promotion_reason", "demotion_triggered",
    "consecutive_promotable_iters", "consecutive_demotable_iters",
    "full_iteration", "iter_timeout_rate", "iter_plies_ratio",
    # Timing
    "self_play_wall_s", "train_wall_s", "iter_wall_s", "positions_per_sec",
    # Derived
    "stall_flush_rate", "backups_per_game", "leaf_evals_per_game",
    # Sanity stats (z distribution)
    "z_mean", "z_std", "z_min", "z_max",
    "z_count_pos", "z_count_zero", "z_count_neg", "z_n",
    # z split by to_move (who is about to play)
    "z_mean_to_move_red", "z_n_to_move_red",
    "z_count_pos_to_move_red", "z_count_zero_to_move_red", "z_count_neg_to_move_red",
    "z_mean_to_move_black", "z_n_to_move_black",
    "z_count_pos_to_move_black", "z_count_zero_to_move_black", "z_count_neg_to_move_black",
    "z_bad_value_count",
    # Sanity stats (policy)
    "pi_len_mismatch_frac", "pi_all_zero_frac", "pi_negative_frac",
    "pi_empty_legal_frac", "pi_empty_visits_frac",
    # Sanity stats (value)
    "v_sample_n", "v_pred_mean", "v_pred_std", "v_pred_min", "v_pred_max",
    "v_mse_vs_z", "v_sign_agree", "v_z_batch_mismatch",
    # Value head saturation diagnostics
    "v_frac_sat", "v_pre_min", "v_pre_max", "v_pre_mean", "v_pre_std",
    "v_pre_abs_p99", "v_zv_corr", "v_non_draw_n",
    # Value head classification diagnostics (balanced accuracy, MCC)
    "v_label_pos", "v_label_neg", "v_maj_baseline", "v_bal_acc", "v_mcc",
    "v_cm_tp", "v_cm_tn", "v_cm_fp", "v_cm_fn",
    # Phase 4: per-game replay contribution cap
    "replay_cap_enabled", "replay_cap_max", "replay_cap_endgame_keep",
    "replay_cap_games_capped", "replay_cap_capped_rate",
    "replay_cap_total_orig", "replay_cap_total_kept",
    "replay_cap_mean_orig", "replay_cap_mean_kept", "replay_cap_kept_fraction",
    # Phase 2: connectivity-bucketed sanity (flattened from sanity_by_connectivity dict;
    # nested form lives in iteration_<N>.json under sanity_by_connectivity)
    "sbc_winning_n", "sbc_winning_sign_agree", "sbc_winning_median_abs_v",
    "sbc_no_winning_n", "sbc_no_winning_sign_agree", "sbc_no_winning_median_abs_v",
    # Phase 2: inline forced-probe summary (flattened; nested form in iteration_<N>.json)
    "fps_n", "fps_sign_correct", "fps_sign_correct_pct", "fps_median_abs_v",
    "fps_delta_sign_correct_pct", "fps_delta_median_abs_v",
    "fps_rolling5_sign_correct_pct", "fps_rolling5_median_abs_v",
]


def append_metrics_csv(metrics_path: str, metrics: dict, fieldnames: List[str]):
    """Append one row to metrics CSV. Create with header if doesn't exist."""
    file_exists = os.path.exists(metrics_path)

    try:
        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
    except Exception as e:
        print(f"WARNING: failed to write metrics CSV: {e}")


# =============================================================================
# Sanity stat helpers (computed once per iteration, no hot-loop impact)
# =============================================================================

import math


def _basic_stats(values: List[float]) -> dict:
    """Compute basic statistics for a list of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "count_pos": 0, "count_zero": 0, "count_neg": 0, "n": 0}
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(var)
    return {
        "mean": mean, "std": std, "min": min(values), "max": max(values),
        "count_pos": sum(1 for x in values if x > 0),
        "count_zero": sum(1 for x in values if x == 0),
        "count_neg": sum(1 for x in values if x < 0),
        "n": n
    }


def summarize_z(positions: List["PositionRecord"]) -> dict:
    """Compute z (outcome) distribution stats.

    Note: z is outcome from to_move's POV (+1 if to_move won, -1 if lost, 0 draw).
    The to_move splits show how outcomes distribute when it's red's turn vs black's turn.
    """
    zs_all, zs_red, zs_black = [], [], []
    bad_z = 0
    for rec in positions:
        z = float(rec.outcome)
        if z not in (-1.0, 0.0, 1.0):
            bad_z += 1
        zs_all.append(z)
        (zs_red if rec.to_move == "red" else zs_black).append(z)

    s_all, s_r, s_b = _basic_stats(zs_all), _basic_stats(zs_red), _basic_stats(zs_black)
    return {
        # Overall z distribution
        "z_mean": s_all["mean"], "z_std": s_all["std"],
        "z_min": s_all["min"], "z_max": s_all["max"],
        "z_count_pos": s_all["count_pos"], "z_count_zero": s_all["count_zero"],
        "z_count_neg": s_all["count_neg"], "z_n": s_all["n"],
        # Split by to_move (who is about to play, NOT winner)
        "z_mean_to_move_red": s_r["mean"],
        "z_n_to_move_red": s_r["n"],
        "z_count_pos_to_move_red": s_r["count_pos"],
        "z_count_zero_to_move_red": s_r["count_zero"],
        "z_count_neg_to_move_red": s_r["count_neg"],
        "z_mean_to_move_black": s_b["mean"],
        "z_n_to_move_black": s_b["n"],
        "z_count_pos_to_move_black": s_b["count_pos"],
        "z_count_zero_to_move_black": s_b["count_zero"],
        "z_count_neg_to_move_black": s_b["count_neg"],
        "z_bad_value_count": bad_z,  # Should be 0; >0 means z not in {-1,0,+1}
    }


def _classify_position_from_tensor(
    board_tensor: np.ndarray, winning_size_threshold: int = 8
) -> str:
    """Bucket a position into 'winning_structure' or 'no_winning_structure'.

    Uses channels 24-29 (Phase 2 connectivity masks) directly — no state
    reconstruction needed. Mirrors the intent of
    value_calibration.classify_position but operates on the raw NHWC tensor
    so it can be applied during training without rebuilding TwixtState.

    Channel layout (from twixt_state.py):
        24: red_connected_to_top
        25: red_connected_to_bottom
        26: red_connected_to_both    (terminal-only when nonzero)
        27: black_connected_to_left
        28: black_connected_to_right
        29: black_connected_to_both  (terminal-only when nonzero)

    Bucket rule (matches spec §9.2 intent):
        winning_structure  = either color has a goal-touching component AND
                             (≥winning_size_threshold pegs in that component
                              OR pegs touching both goal edges).

    For pre-Phase-2 24-channel tensors (no channels 24-29) returns
    "unknown" — caller should drop these from per-bucket aggregates.
    """
    # NHWC tensor: (H, W, C). Channel access via [..., ch]
    if board_tensor.ndim != 3 or board_tensor.shape[-1] < 30:
        return "unknown"
    # Sums over H,W of each connectivity channel
    s_red_top = float(np.sum(board_tensor[..., 24]))
    s_red_bot = float(np.sum(board_tensor[..., 25]))
    s_blk_left = float(np.sum(board_tensor[..., 27]))
    s_blk_right = float(np.sum(board_tensor[..., 28]))
    # "winning structure" approximation:
    #   - >= threshold pegs in any single goal-touching mask, OR
    #   - both goal-edge masks for one color are non-empty (two
    #     goal-touching components, or one component touching both)
    red_winning = (
        s_red_top >= winning_size_threshold
        or s_red_bot >= winning_size_threshold
        or (s_red_top > 0 and s_red_bot > 0)
    )
    black_winning = (
        s_blk_left >= winning_size_threshold
        or s_blk_right >= winning_size_threshold
        or (s_blk_left > 0 and s_blk_right > 0)
    )
    if red_winning or black_winning:
        return "winning_structure"
    return "no_winning_structure"


def summarize_policy_sanity(positions: List["PositionRecord"]) -> dict:
    """Check structural validity of policy targets."""
    mismatched, all_zero, negative, empty_legal, empty_visits = 0, 0, 0, 0, 0
    for rec in positions:
        if not rec.legal_moves:
            empty_legal += 1
        if not rec.visit_counts:
            empty_visits += 1
        if len(rec.visit_counts) != len(rec.legal_moves):
            mismatched += 1
        if any(v < 0 for v in rec.visit_counts):
            negative += 1
        if rec.visit_counts and sum(rec.visit_counts) == 0:
            all_zero += 1
    n = max(1, len(positions))
    return {
        "pi_len_mismatch_frac": mismatched / n,
        "pi_all_zero_frac": all_zero / n,
        "pi_negative_frac": negative / n,
        "pi_empty_legal_frac": empty_legal / n,
        "pi_empty_visits_frac": empty_visits / n,
    }


def summarize_value_sanity(
    network: "AlphaZeroNetwork",
    positions: List["PositionRecord"],
    active_size: int,
    sample_n: int = 256,
    seed: int = 0,
    max_moves_cap: int = 512,
) -> dict:
    """Compute value head sanity stats via forward pass (canonicalization-safe).

    Uses a fixed seed for reproducible sampling across runs/resumes.
    Does NOT call network.eval() to avoid changing model state.

    Args:
        network: AlphaZero network
        positions: List of positions to sample from
        active_size: Curriculum board size
        sample_n: Max positions to sample
        seed: Random seed for reproducible sampling
        max_moves_cap: Must match training's max_moves_cap to avoid padding differences
    """
    if not positions:
        return {"v_sample_n": 0}

    # Use fixed seed for reproducible sampling (doesn't affect training RNG)
    sanity_rng = random.Random(seed)
    sample = positions if len(positions) <= sample_n else sanity_rng.sample(positions, sample_n)

    # Match training's max_moves_cap to avoid padding differences
    boards, move_rows, move_cols, move_mask, _, outcomes = make_padded_batch(
        sample, max_moves_cap=max_moves_cap
    )

    # Belt-and-suspenders: verify make_padded_batch outcomes match rec.outcome
    z_from_recs = np.array([float(rec.outcome) for rec in sample], dtype=np.float32)
    z_from_batch = np.array(outcomes.tolist(), dtype=np.float32).reshape(-1)
    batch_mismatch = int(np.sum(np.abs(z_from_recs - z_from_batch) > 1e-6))

    # Forward pass WITH CANONICALIZATION and pretanh (don't bypass it!)
    _, values, pretanh = network.forward_padded(
        boards, move_rows, move_cols, move_mask,
        active_size=active_size,
        return_value_pretanh=True
    )
    mx.eval(values, pretanh)

    # Convert to numpy, ensure 1D
    v_np = np.array(values.tolist(), dtype=np.float32).reshape(-1)
    pre_np = np.array(pretanh.tolist(), dtype=np.float32).reshape(-1)
    z_np = np.array(outcomes.tolist(), dtype=np.float32).reshape(-1)

    n = len(v_np)
    v_mean, v_std = float(np.mean(v_np)), float(np.std(v_np))
    v_min, v_max = float(np.min(v_np)), float(np.max(v_np))
    mse = float(np.mean((v_np - z_np) ** 2))

    # Sign agreement on non-draws (epsilon-stable)
    eps = 1e-6
    non_draw_mask = z_np != 0
    non_draw_n = int(non_draw_mask.sum())
    z_nd = z_np[non_draw_mask]
    v_nd = v_np[non_draw_mask]

    # Epsilon-stable sign: avoid flapping around 0
    pred_sign = np.where(v_nd > eps, 1.0, np.where(v_nd < -eps, -1.0, 0.0))
    true_sign = np.sign(z_nd)  # ±1
    sign_agree = float(np.mean(pred_sign == true_sign)) if z_nd.size > 0 else 0.0

    # --- Extra diagnostics to make sign_agree comparable across runs ---
    # Label distribution
    pos = int(np.sum(z_nd > 0))
    neg = int(np.sum(z_nd < 0))
    maj_acc = (max(pos, neg) / non_draw_n) if non_draw_n > 0 else 0.0

    # Pred sign with epsilon
    pred_pos = v_nd > eps
    pred_neg = v_nd < -eps

    # Confusion matrix (only where prediction is non-zero)
    use = pred_pos | pred_neg
    use_n = int(np.sum(use))

    if use_n > 0:
        z_use = z_nd[use]
        p_use = pred_pos[use]

        tp = int(np.sum(p_use & (z_use > 0)))
        tn = int(np.sum((~p_use) & (z_use < 0)))
        fp = int(np.sum(p_use & (z_use < 0)))
        fn = int(np.sum((~p_use) & (z_use > 0)))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = 0.5 * (tpr + tnr)

        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt(denom) if denom > 0 else 0.0
    else:
        tp = tn = fp = fn = 0
        bal_acc = 0.0
        mcc = 0.0
    # --- end extra diagnostics ---

    # Saturation stats
    frac_sat = float(np.mean(np.abs(v_np) >= 0.999))

    # Pre-tanh stats (critical for diagnosing value head explosion)
    pre_min, pre_max = float(np.min(pre_np)), float(np.max(pre_np))
    pre_mean, pre_std = float(np.mean(pre_np)), float(np.std(pre_np))
    pre_abs_p99 = float(np.percentile(np.abs(pre_np), 99))  # Best saturation signal

    # Correlation-like signal: mean(z * v) for non-draws (direction agreement)
    zv_corr = float(np.mean(z_np[non_draw_mask] * v_np[non_draw_mask])) if non_draw_n > 0 else 0.0

    # === Connectivity-bucketed sanity (Phase 2 connectivity feedback) ===
    # Bucket each sampled position by board_tensor connectivity channels and
    # report sign_agree + median |v| per bucket. Skips silently for 24-channel
    # tensors (pre-Phase-2 checkpoints) — those return "unknown".
    bucket_idx = {"winning_structure": [], "no_winning_structure": []}
    for i, rec in enumerate(sample):
        bucket = _classify_position_from_tensor(rec.board_tensor)
        if bucket in bucket_idx:
            bucket_idx[bucket].append(i)

    def _bucket_stats(idxs):
        if not idxs:
            return {"n": 0, "sign_agree": None, "median_abs_v": None}
        z_b = z_np[idxs]
        v_b = v_np[idxs]
        non_draw_b = z_b != 0
        if int(non_draw_b.sum()) == 0:
            sign_agree_b = None
        else:
            pred_sign_b = np.where(v_b[non_draw_b] > eps, 1.0,
                                   np.where(v_b[non_draw_b] < -eps, -1.0, 0.0))
            true_sign_b = np.sign(z_b[non_draw_b])
            sign_agree_b = float(np.mean(pred_sign_b == true_sign_b))
        abs_v = np.abs(v_b)
        median_abs = float(np.median(abs_v)) if abs_v.size > 0 else None
        return {
            "n": len(idxs),
            "sign_agree": sign_agree_b,
            "median_abs_v": median_abs,
        }

    sanity_by_connectivity = {
        "winning_structure": _bucket_stats(bucket_idx["winning_structure"]),
        "no_winning_structure": _bucket_stats(bucket_idx["no_winning_structure"]),
        "winning_size_threshold": 8,
    }

    return {
        "v_sample_n": n,
        "v_pred_mean": v_mean,
        "v_pred_std": v_std,
        "v_pred_min": v_min,
        "v_pred_max": v_max,
        "v_mse_vs_z": mse,
        "v_sign_agree": sign_agree,
        "v_z_batch_mismatch": batch_mismatch,
        # New fields for saturation diagnostics
        "v_frac_sat": frac_sat,
        "v_pre_min": pre_min,
        "v_pre_max": pre_max,
        "v_pre_mean": pre_mean,
        "v_pre_std": pre_std,
        "v_pre_abs_p99": pre_abs_p99,  # If 20-80, tanh is pinned
        "v_zv_corr": zv_corr,
        "v_non_draw_n": non_draw_n,
        # Balanced accuracy diagnostics (imbalance-robust)
        "v_label_pos": pos,
        "v_label_neg": neg,
        "v_maj_baseline": maj_acc,
        "v_bal_acc": bal_acc,
        "v_mcc": mcc,
        "v_cm_tp": tp,
        "v_cm_tn": tn,
        "v_cm_fp": fp,
        "v_cm_fn": fn,
        # Phase 2 connectivity-bucketed sanity (channels 24-29)
        "sanity_by_connectivity": sanity_by_connectivity,
    }


def verify_canonicalization(
    positions: List["PositionRecord"],
    active_size: int,
    sample_n: int = 32,
) -> dict:
    """Verify canonicalization invariants on a sample of positions.

    Checks:
    1. CH_TO_MOVE (channel 18) is 1 for all samples after canonicalization
    2. Move coords are in bounds [0, active_size) for valid slots
    3. Move mask is unchanged
    4. For black samples: channel swap is correct (ch0 = rotated old ch1)
    5. For red samples: coords are unchanged
    """
    from .network import (
        canonicalize_batch, CH_TO_MOVE, CH_RED_PEG, CH_BLACK_PEG
    )

    if not positions:
        return {"canon_errors": [], "canon_checked": 0}

    sample = positions if len(positions) <= sample_n else random.sample(positions, sample_n)

    boards, move_rows, move_cols, move_mask, _, _ = make_padded_batch(sample)

    boards_c, rows_c, cols_c, mask_c = canonicalize_batch(
        boards, move_rows, move_cols, move_mask, active_size
    )

    # Sync for comparison
    mx.eval(boards_c, rows_c, cols_c, mask_c)

    errors = []

    # 1. CH_TO_MOVE should be 1 for all after canonicalization
    ch18 = boards_c[:, 0, 0, CH_TO_MOVE]
    if not bool(mx.all(ch18 > 0.5).item()):
        errors.append("CH_TO_MOVE not all 1 after canonicalization")

    # 2. Move coords in bounds for valid slots
    valid = mask_c > 0.5
    rows_valid = (rows_c >= 0) | ~valid
    rows_upper = (rows_c < active_size) | ~valid
    cols_valid = (cols_c >= 0) | ~valid
    cols_upper = (cols_c < active_size) | ~valid
    if not bool(mx.all(rows_valid & rows_upper).item()):
        errors.append("move rows out of bounds")
    if not bool(mx.all(cols_valid & cols_upper).item()):
        errors.append("move cols out of bounds")

    # 3. Mask unchanged
    if not bool(mx.all(move_mask == mask_c).item()):
        errors.append("mask changed")

    # 4. For black-to-move samples: verify channel swap is correct
    is_black = boards[:, 0, 0, CH_TO_MOVE] < 0.5
    black_mask = is_black[:, None, None, None]

    # Get rotated boards for comparison (same rotation as canonicalize_batch)
    boards_rot = mx.transpose(boards, (0, 2, 1, 3))[:, :, ::-1, :]

    # Expected: for black samples, canonical ch0 should be rotated old black pegs
    expected_cur_pegs = boards_rot[:, :, :, CH_BLACK_PEG:CH_BLACK_PEG + 1]
    actual_cur_pegs = boards_c[:, :, :, CH_RED_PEG:CH_RED_PEG + 1]

    # Check only black samples
    diff = mx.abs(actual_cur_pegs - expected_cur_pegs)
    bad = mx.any((diff > 1e-6) & black_mask)
    if bad.item():
        errors.append("channel swap incorrect for current pegs")

    # 5. For red-to-move samples: coords should be unchanged
    is_red = ~is_black
    red_mask_2d = is_red[:, None] & (mask_c > 0.5)
    rows_diff = mx.any((rows_c != move_rows) & red_mask_2d)
    cols_diff = mx.any((cols_c != move_cols) & red_mask_2d)
    if rows_diff.item() or cols_diff.item():
        errors.append("red-to-move coords were incorrectly rotated")

    return {"canon_errors": errors, "canon_checked": len(sample)}


def make_padded_batch(
    positions: List["PositionRecord"],
    max_moves_cap: int = 512,
):
    """Prepare batched tensors with padded moves for training.

    Args:
        positions: List of PositionRecord
        max_moves_cap: Maximum moves to consider (truncates if exceeded)

    Returns:
        Tuple of (boards_mx, move_rows, move_cols, move_mask, target_pi, outcomes)
    """
    B = len(positions)
    assert B > 0

    # ASSERT: all positions should have same active_size (no mixed batches)
    active_sizes = set(p.active_size for p in positions)
    if len(active_sizes) > 1:
        raise ValueError(f"Mixed active_sizes in batch: {active_sizes}")

    # ASSERT: no move coords exceed active_size
    active_size = positions[0].active_size
    for i, p in enumerate(positions):
        for r, c in p.legal_moves:
            assert r < active_size and c < active_size, (
                f"Move ({r},{c}) exceeds active_size={active_size} at position {i}"
            )

    # Stack boards: (B, H, W, C)
    boards_np = np.stack(
        [p.board_tensor for p in positions], axis=0
    ).astype(np.float32)
    boards_mx = mx.array(boards_np)

    # Determine M (max moves in this batch, capped)
    max_len = max(len(p.legal_moves) for p in positions)
    M = min(max_len, max_moves_cap)

    # Initialize padded arrays
    move_rows_np = np.zeros((B, M), dtype=np.int32)
    move_cols_np = np.zeros((B, M), dtype=np.int32)
    move_mask_np = np.zeros((B, M), dtype=np.float32)
    target_pi_np = np.zeros((B, M), dtype=np.float32)
    outcomes_np = np.array(
        [float(p.outcome) for p in positions], dtype=np.float32
    )

    for i, p in enumerate(positions):
        moves = p.legal_moves[:M]
        counts = p.visit_counts[:M]
        n = len(moves)

        if n == 0:
            continue

        for j, ((r, c), cnt) in enumerate(zip(moves, counts)):
            move_rows_np[i, j] = r
            move_cols_np[i, j] = c
            move_mask_np[i, j] = 1.0
            target_pi_np[i, j] = float(cnt)

        # Normalize counts only over valid moves
        s = target_pi_np[i, :n].sum()
        if s > 0:
            target_pi_np[i, :n] /= s

    return (
        boards_mx,
        mx.array(move_rows_np),
        mx.array(move_cols_np),
        mx.array(move_mask_np),
        mx.array(target_pi_np),
        mx.array(outcomes_np),
    )


def _compute_progress_weighted_value_loss(
    values: mx.array,
    outcomes: mx.array,
    plies: np.ndarray,        # (B,) int32
    game_n_moves: np.ndarray, # (B,) int32
    floor: float = 0.25,
) -> mx.array:
    """Progress-weighted value loss with normalized weighted mean.

    weight_i = floor + (1 - floor) * progress_i
    progress_i = clip(ply_i / max(game_n_moves_i - 1, 1), 0, 1)
    loss = sum(w * err^2) / sum(w)   (normalized weighted mean)

    Edge case: game_n_moves <= 1 -> denominator clamp yields progress = 1.0.
    Edge case: sum(w) == 0 -> fallback to unweighted mean (shouldn't happen in
    practice since floor > 0 is the typical case).
    """
    denom = np.maximum(game_n_moves - 1, 1).astype(np.float32)
    progress = np.clip(plies.astype(np.float32) / denom, 0.0, 1.0)
    weights_np = floor + (1.0 - floor) * progress
    weights = mx.array(weights_np)
    err_sq = (values - outcomes) ** 2
    total_w = mx.sum(weights)
    if float(total_w) == 0.0:
        return mx.mean(err_sq)  # fallback; shouldn't happen with floor>=0
    return mx.sum(weights * err_sq) / total_w


def alphazero_loss_batch(
    network: AlphaZeroNetwork,
    positions: List["PositionRecord"],
    l2_weight: float = 1e-4,
    value_weight: float = 0.5,
    max_moves_cap: int = 512,
    active_size: int = 24,
    progress_weighted: bool = True,
    progress_weight_floor: float = 0.25,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Batched policy + value + L2 loss (vectorized, no loops).

    This replaces the old per-position loop with a single batched forward pass,
    dramatically reducing Metal allocations and avoiding resource limit crashes.

    Args:
        network: AlphaZero network
        positions: List of training positions
        l2_weight: L2 regularization weight
        value_weight: Weight for value loss (default 0.5; Phase 2 bump from 0.25)
        max_moves_cap: Maximum moves per position (512 for TwixT)
        active_size: Curriculum board size for masked pooling
        progress_weighted: If True, use progress-weighted value loss (default True; Phase 2)
        progress_weight_floor: Floor weight for earliest plies when progress_weighted=True

    Returns:
        Tuple of (total_loss, policy_loss, value_loss, l2_loss)
        NOTE: value_loss is RAW (unweighted) for meaningful logging

    IMPORTANT: total_loss MUST be first element because nn.value_and_grad()
    only differentiates the first returned value.
    """
    boards, move_rows, move_cols, move_mask, target_pi, outcomes = make_padded_batch(
        positions, max_moves_cap=max_moves_cap
    )

    # Extract ply / game_n_moves from positions for progress-weighted mode.
    # Old records default ply=0, game_n_moves=None; `or 1` keeps np int happy.
    plies_np = np.array(
        [getattr(p, "ply", 0) for p in positions], dtype=np.int32
    )
    game_n_moves_np = np.array(
        [getattr(p, "game_n_moves", None) or 1 for p in positions], dtype=np.int32
    )

    # Single batched forward pass with curriculum active_size
    logits, values, _ = network.forward_padded(
        boards, move_rows, move_cols, move_mask, active_size=active_size
    )  # (B, M), (B,), None

    # Policy loss: cross entropy with masked logits
    # logits already have -1e9 where mask==0, so logsumexp ignores those
    log_probs = logits - mx.logsumexp(logits, axis=1, keepdims=True)  # (B, M)
    policy_loss = -mx.sum(target_pi * log_probs, axis=1)  # (B,)
    policy_loss = mx.mean(policy_loss)

    # Value loss: progress-weighted or plain MSE (RAW, unweighted for diagnostics)
    if progress_weighted:
        value_loss = _compute_progress_weighted_value_loss(
            values, outcomes, plies_np, game_n_moves_np, floor=progress_weight_floor
        )
    else:
        value_loss = mx.mean((values - outcomes) ** 2)

    # L2 regularization (still loops but over params, not positions)
    l2_loss = mx.array(0.0)
    for _, param in flatten_params(network.parameters()):
        l2_loss = l2_loss + mx.sum(param ** 2)
    l2_loss = l2_weight * l2_loss

    # Combine losses (value weighted lower to prevent dominance)
    total_loss = policy_loss + value_weight * value_loss + l2_loss

    # CRITICAL: total_loss must be first for nn.value_and_grad()
    return total_loss, policy_loss, value_loss, l2_loss


def train_step(
    network: AlphaZeroNetwork,
    main_module: MainModule,
    opt_main: optim.Optimizer,
    opt_value: optim.Optimizer,
    batch: List["PositionRecord"],
    l2_weight: float = 1e-4,
    value_weight: float = 0.5,
    max_moves_cap: int = 512,
    active_size: int = 24,
    value_grad_max_norm: float = 0.5,
    progress_weighted: bool = True,
    progress_weight_floor: float = 0.25,
) -> Tuple[float, float, float, float]:
    """Single training step with two optimizers and separate gradient clipping.

    Uses separate optimizers for encoder+policy (opt_main) and value head (opt_value)
    to allow different learning rates. Updates are done via real nn.Module references
    to guarantee mutations land in the live network.

    Args:
        network: AlphaZero network
        main_module: MainModule wrapper holding encoder + policy_head references
        opt_main: Optimizer for encoder + policy head
        opt_value: Optimizer for value head (typically lower LR)
        batch: List of training positions
        l2_weight: L2 regularization weight
        value_weight: Weight for value loss (default 0.5; Phase 2 bump from 0.25)
        max_moves_cap: Maximum moves per position
        active_size: Curriculum board size for masked pooling
        value_grad_max_norm: Max grad norm for value head (default 0.5)
        progress_weighted: If True, use progress-weighted value loss (default True; Phase 2)
        progress_weight_floor: Floor weight for earliest plies (default 0.25)

    Returns:
        Tuple of (total_loss, policy_loss, value_loss, l2_loss) as floats
    """
    def loss_fn(model):
        # Returns (total, policy, value, l2) - total is first for grad
        return alphazero_loss_batch(
            model, batch,
            l2_weight=l2_weight,
            value_weight=value_weight,
            max_moves_cap=max_moves_cap,
            active_size=active_size,
            progress_weighted=progress_weighted,
            progress_weight_floor=progress_weight_floor,
        )

    # value_and_grad differentiates first element (total_loss)
    loss_tuple, grads = nn.value_and_grad(network, loss_fn)(network)

    # Unpack losses
    total_loss, policy_loss, value_loss, l2_loss = loss_tuple

    # Slice GRADS only (not params) into module-shaped trees
    # NOTE: Assumes network.parameters() / grads use top-level keys: encoder, policy_head, value_head
    main_grads = {
        "encoder": grads["encoder"],
        "policy_head": grads["policy_head"],
    }
    value_grads = grads["value_head"]

    # Clip main grads (encoder + policy) at 1.0
    main_grads, main_gnorm = clip_grad_norm(main_grads, max_norm=1.0)

    # Clip value head grads more aggressively
    value_grads, value_gnorm = clip_grad_norm(value_grads, max_norm=value_grad_max_norm)

    # Force evaluation before updates
    mx.eval(main_grads, value_grads, main_gnorm, value_gnorm)

    # Update REAL modules (guaranteed to mutate network)
    opt_main.update(main_module, main_grads)
    opt_value.update(network.value_head, value_grads)

    # Evaluate all arrays before extracting Python floats
    mx.eval(network.parameters(), opt_main.state, opt_value.state, loss_tuple)

    return (
        float(total_loss.item()),
        float(policy_loss.item()),
        float(value_loss.item()),
        float(l2_loss.item()),
    )


class ReplayBuffer:
    """Fixed-size buffer of training positions with uniform sampling.

    Uses ring buffer semantics - oldest positions are overwritten when
    buffer is full.
    """

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer: List["PositionRecord"] = []
        self.index = 0

    def add_game(self, game: "GameRecord"):
        """Add all positions from a game to the buffer."""
        for pos in game.positions:
            if len(self.buffer) < self.max_size:
                self.buffer.append(pos)
            else:
                # Overwrite oldest (ring buffer)
                self.buffer[self.index] = pos
            self.index = (self.index + 1) % self.max_size

    def add_positions(self, positions: List["PositionRecord"]):
        """Add positions directly to the buffer."""
        for pos in positions:
            if len(self.buffer) < self.max_size:
                self.buffer.append(pos)
            else:
                self.buffer[self.index] = pos
            self.index = (self.index + 1) % self.max_size

    def sample(
        self,
        batch_size: int,
        rng: Optional[random.Random] = None,
        active_size: Optional[int] = None,
    ) -> List["PositionRecord"]:
        """Sample random batch from buffer.

        Args:
            batch_size: Number of positions to sample
            rng: Optional RNG for reproducibility
            active_size: If provided, only sample positions with this active_size
                        (for curriculum learning - avoids mixed board sizes in batch)

        Returns:
            List of sampled positions
        """
        if rng is None:
            rng = random.Random()

        # Filter by active_size if specified (curriculum learning)
        if active_size is not None:
            eligible = [p for p in self.buffer if p.active_size == active_size]
            if not eligible:
                return []
            return rng.sample(eligible, min(batch_size, len(eligible)))

        return rng.sample(self.buffer, min(batch_size, len(self.buffer)))

    def count_by_active_size(self, active_size: int) -> int:
        """Count positions with given active_size."""
        return sum(1 for p in self.buffer if p.active_size == active_size)

    def __len__(self):
        return len(self.buffer)


def run_parallel_selfplay(
    evaluator,  # LocalGPUEvaluator
    mcts_config: "MCTSConfig",
    games_to_play: int,
    n_workers: int,
    master_rng: random.Random,
    max_moves: int,
    active_size: int,
    curriculum: "CurriculumManager",
    buffer: ReplayBuffer,
    game_saver: Optional[GameSaver] = None,
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
    # Phase 4: per-game replay contribution cap
    max_positions_per_game: Optional[int] = None,
    endgame_keep_positions: int = 16,
) -> Tuple[
    List["GameRecord"],  # All game records (for stats)
    List["PositionRecord"],  # New positions (for sanity stats)
    Dict[str, Any],  # Aggregated stats
]:
    """Run parallel self-play with multiple CPU workers and single GPU inference server.

    Architecture:
    - Main process: GPU inference server (thread) + position consumer
    - Worker processes (CPU-only): MCTS with RemoteEvaluator

    Args:
        evaluator: LocalGPUEvaluator for GPU inference
        mcts_config: MCTS configuration
        games_to_play: Total games to generate across all workers
        n_workers: Number of parallel workers
        master_rng: RNG for generating worker seeds
        max_moves: Max moves per game
        active_size: Curriculum board size
        curriculum: CurriculumManager for recording game results
        buffer: ReplayBuffer to add positions to

    Returns:
        Tuple of (games, new_positions, stats_dict)
    """
    import signal
    from .ipc_messages import StopSignal, WorkerDone, WorkerStats, GameComplete
    from .inference_server import InferenceServer
    from .self_play_worker import self_play_worker_main
    from .self_play import DRAW_TIMEOUT, DRAW_BOARD_FULL, DRAW_STATE_CAP, DRAW_UNKNOWN, RESIGN, ADJUDICATED

    # Signal handling for clean Ctrl+C shutdown
    # (queue.get() doesn't reliably raise KeyboardInterrupt on macOS with spawn)
    interrupted = False
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def sigint_handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, sigint_handler)

    # macOS requires spawn context
    ctx = mp.get_context("spawn")

    # Queue sizing (bounded for backpressure)
    request_q_max = 256
    response_q_max = 64
    position_q_max = 128
    stats_q_max = 128

    # Create queues
    request_queue = ctx.Queue(maxsize=request_q_max)
    response_queues: Dict[int, Any] = {
        wid: ctx.Queue(maxsize=response_q_max) for wid in range(n_workers)
    }
    position_queue = ctx.Queue(maxsize=position_q_max)
    stats_queue = ctx.Queue(maxsize=stats_q_max)

    # Dynamic scheduling: shared atomic counter for game assignment
    # (ctx.Value has internal lock via .get_lock(), no separate Lock needed)
    next_game_id = ctx.Value("i", 0)  # int

    # Start inference server in thread (same process as GPU)
    server = InferenceServer(
        evaluator=evaluator,
        request_queue=request_queue,
        response_queues=response_queues,
        max_batch_rows=mcts_config.eval_batch_size,
        flush_ms=2,
        stats_queue=stats_queue,
    )
    server_thread = threading.Thread(target=server.run_forever, daemon=True)
    server_thread.start()

    # Don't spawn more workers than games (avoids idle processes in small runs)
    n_spawn = min(n_workers, games_to_play)

    # Start worker processes
    workers = []
    for wid in range(n_spawn):
        worker_seed = master_rng.randint(0, 2**31)

        p = ctx.Process(
            target=self_play_worker_main,
            kwargs={
                "worker_id": wid,
                "request_queue": request_queue,
                "response_queue": response_queues[wid],
                "position_queue": position_queue,
                "stats_queue": stats_queue,
                "mcts_config": mcts_config,
                # Dynamic scheduling inputs
                "games_total": games_to_play,
                "next_game_id": next_game_id,
                "seed": worker_seed,
                "chunk_size": 32,
                "max_moves": max_moves,
                "add_noise": True,
                "active_size": active_size,
                # Resign parameters
                "resign_enabled": resign_enabled,
                "resign_min_ply": resign_min_ply,
                "resign_threshold": resign_threshold,
                "resign_window": resign_window,
                "resign_k": resign_k,
                "resign_min_visits": resign_min_visits,
                "resign_min_top1_share": resign_min_top1_share,
                # Adjudication parameters
                "adjudicate_enabled": adjudicate_enabled,
                "adjudicate_min_ply": adjudicate_min_ply,
                "adjudicate_threshold": adjudicate_threshold,
                "adjudicate_min_visits": adjudicate_min_visits,
                "adjudicate_min_top1_share": adjudicate_min_top1_share,
                "adjudicate_debug": adjudicate_debug,
                # Phase 4: per-game replay cap
                "max_positions_per_game": max_positions_per_game,
                "endgame_keep_positions": endgame_keep_positions,
            },
        )
        p.start()
        workers.append(p)

    print(f"  Started {n_spawn} workers, dynamic game assignment for {games_to_play} games")

    # Track stats
    games_records = []
    new_positions: List["PositionRecord"] = []
    workers_done = 0
    worker_done_stats: Dict[int, WorkerDone] = {}
    total_positions = 0
    games_completed = 0
    total_plies = 0
    # Per-game accumulators for percentile/timing stats (parallel-mode parity
    # with the sequential path which populates these inline at line 2299-2300).
    game_plies_acc: list = []
    game_durations_acc: list = []

    # Result tracking
    red_wins = 0
    black_wins = 0
    draws = 0
    timeout_draws = 0
    board_full_draws = 0
    state_cap_draws = 0
    unknown_draws = 0
    # Resign tracking (decisive, not draw)
    resign_games = 0
    resigned_by_red = 0
    resigned_by_black = 0
    # Adjudication tracking (decisive, not draw)
    adjudicated_games = 0
    adjudicated_red_wins = 0
    adjudicated_black_wins = 0
    adj_blocked_ply = 0
    adj_blocked_threshold = 0
    adj_blocked_visits = 0
    adj_blocked_top1 = 0
    adj_attempts = 0
    adj_abs_rv_samples = []   # for percentiles
    adj_top1_samples = []     # for percentiles

    # MCTS stats accumulators
    total_backups = 0
    total_nn_calls = 0
    total_expand_calls = 0
    total_nn_batches = 0
    total_waiters = 0
    total_unique_leaves = 0
    max_waiters = 0
    total_flush_full = 0
    total_flush_stall = 0
    total_flush_tail = 0

    # Resign gate aggregation
    rg_checks_red = 0;    rg_checks_black = 0
    rg_value_hits_red = 0; rg_value_hits_black = 0
    rg_eligible_red = 0;   rg_eligible_black = 0
    rg_top1_all = []       # collect all top1_range samples across games
    all_opening_diagnostics = []

    # Phase 4: per-game replay cap aggregation (IPC path)
    total_positions_original = 0
    total_positions_kept = 0
    games_capped = 0
    # Per-length-bucket accounting (edges in plies): [0,40), [40,80), [80,120),
    # [120,160), [160,200), [200, inf). Each bucket stores [games, sum_original,
    # sum_kept].
    _PLY_BUCKET_EDGES = (40, 80, 120, 160, 200)
    _n_buckets = len(_PLY_BUCKET_EDGES) + 1
    ply_bucket_games = [0] * _n_buckets
    ply_bucket_positions_original = [0] * _n_buckets
    ply_bucket_positions_kept = [0] * _n_buckets
    # Termination + length breakdown (parity with sequential path at
    # trainer.py :2340-2354). Previously only the sequential branch
    # populated these, leaving parallel-mode sidecars with all-zeros
    # under replay_cap.positions_by_termination / _short / _long.
    positions_by_termination = {"win": 0, "resign": 0, "adjudicated": 0, "timeout": 0}
    positions_in_short_games = 0   # games with n_moves <= 80
    positions_in_long_games = 0    # games with n_moves > 200

    def _ply_bucket_index(n_moves: int) -> int:
        for i, edge in enumerate(_PLY_BUCKET_EDGES):
            if n_moves < edge:
                return i
        return _n_buckets - 1

    # Helper to process stats queue messages
    def process_stats_message(msg):
        nonlocal games_completed, total_plies
        nonlocal red_wins, black_wins, draws
        nonlocal timeout_draws, board_full_draws, state_cap_draws, unknown_draws
        nonlocal resign_games, resigned_by_red, resigned_by_black
        nonlocal adjudicated_games, adjudicated_red_wins, adjudicated_black_wins
        nonlocal adj_blocked_ply, adj_blocked_threshold, adj_blocked_visits, adj_blocked_top1
        nonlocal adj_attempts, adj_abs_rv_samples, adj_top1_samples
        nonlocal total_backups, total_nn_calls, total_expand_calls, total_nn_batches
        nonlocal total_waiters, total_unique_leaves, max_waiters
        nonlocal total_flush_full, total_flush_stall, total_flush_tail
        nonlocal rg_checks_red, rg_checks_black
        nonlocal rg_value_hits_red, rg_value_hits_black
        nonlocal rg_eligible_red, rg_eligible_black
        nonlocal rg_top1_all
        nonlocal all_opening_diagnostics
        nonlocal total_positions_original, total_positions_kept, games_capped
        nonlocal positions_by_termination, positions_in_short_games, positions_in_long_games

        if isinstance(msg, dict) and msg.get("type") == "server_error":
            raise RuntimeError(f"InferenceServer crashed: {msg.get('error')}")
        elif isinstance(msg, GameComplete):
            games_completed += 1
            total_plies += msg.n_moves
            # Per-game stats for parallel-mode percentile parity with sequential path
            game_plies_acc.append(int(msg.n_moves))
            game_durations_acc.append(float(msg.wall_time_s))

            # Phase 4: replay cap accounting
            # Older messages may lack these fields (default to 0) → treat as uncapped.
            n_orig = getattr(msg, "n_positions_original", 0) or 0
            n_kept = getattr(msg, "n_positions_kept", 0) or 0
            # If both are 0 (message produced before cap was threaded), fall
            # back to n_positions so the totals stay meaningful.
            if n_orig == 0 and n_kept == 0:
                n_orig = msg.n_positions
                n_kept = msg.n_positions
            total_positions_original += n_orig
            total_positions_kept += n_kept
            if n_kept < n_orig:
                games_capped += 1
            _bi = _ply_bucket_index(msg.n_moves)
            ply_bucket_games[_bi] += 1
            ply_bucket_positions_original[_bi] += n_orig
            ply_bucket_positions_kept[_bi] += n_kept
            # Termination-type bucket + short/long length buckets (parity
            # with sequential path at trainer.py :2342-2354). draw_reason
            # int encoding: 0=None, 1=timeout, 2=board_full, 3=state_cap,
            # 4=unknown, 5=resign, 6=adjudicated.
            if msg.winner and msg.draw_reason == 5:
                _term = "resign"
            elif msg.winner and msg.draw_reason == 6:
                _term = "adjudicated"
            elif msg.winner:
                _term = "win"
            else:
                _term = "timeout"
            positions_by_termination[_term] += n_kept
            if msg.n_moves <= 80:
                positions_in_short_games += n_kept
            elif msg.n_moves > 200:
                positions_in_long_games += n_kept

            # Track results
            if msg.winner == "red":
                red_wins += 1
            elif msg.winner == "black":
                black_wins += 1
            else:
                draws += 1
                # Track draw breakdown (draw_reason: 1=timeout, 2=board_full, 3=state_cap, 4=unknown)
                if msg.draw_reason == 1:  # timeout
                    timeout_draws += 1
                elif msg.draw_reason == 2:  # board_full
                    board_full_draws += 1
                elif msg.draw_reason == 3:  # state_cap
                    state_cap_draws += 1
                else:
                    unknown_draws += 1

            # Track resign (separate from draw breakdown)
            # draw_reason int: 0=None, 1=timeout, 2=board_full, 3=state_cap, 4=unknown, 5=resign
            if msg.draw_reason == 5:
                resign_games += 1
                # Derive who resigned: loser = opponent of winner
                if msg.winner == "red":
                    resigned_by_black += 1
                elif msg.winner == "black":
                    resigned_by_red += 1

            # Track adjudication (separate from draw breakdown, code 6)
            if msg.draw_reason == 6:
                adjudicated_games += 1
                if msg.winner == "red":
                    adjudicated_red_wins += 1
                elif msg.winner == "black":
                    adjudicated_black_wins += 1

            # Adjudication diagnostics aggregation
            if msg.adj_attempted:
                adj_attempts += 1
                if msg.adj_abs_rv is not None:
                    adj_abs_rv_samples.append(msg.adj_abs_rv)
                if msg.adj_top1 is not None:
                    adj_top1_samples.append(msg.adj_top1)
                if msg.adj_blocked_by == "ply":
                    adj_blocked_ply += 1
                elif msg.adj_blocked_by == "threshold":
                    adj_blocked_threshold += 1
                elif msg.adj_blocked_by == "visits":
                    adj_blocked_visits += 1
                elif msg.adj_blocked_by == "top1":
                    adj_blocked_top1 += 1

            # MCTS stats aggregation
            total_backups += msg.total_backups
            total_nn_calls += msg.nn_calls
            total_expand_calls += msg.expand_calls
            total_nn_batches += msg.nn_batches
            total_waiters += msg.total_waiters
            total_unique_leaves += msg.unique_leaves
            max_waiters = max(max_waiters, msg.max_waiters)  # max, not sum!
            total_flush_full += msg.flush_full
            total_flush_stall += msg.flush_stall
            total_flush_tail += msg.flush_tail

            # Resign gate aggregation
            rg_checks_red += msg.rg_checks_red
            rg_checks_black += msg.rg_checks_black
            rg_value_hits_red += msg.rg_value_hits_red
            rg_value_hits_black += msg.rg_value_hits_black
            rg_eligible_red += msg.rg_eligible_red
            rg_eligible_black += msg.rg_eligible_black
            rg_top1_all.extend(msg.rg_top1_samples)

            # Record to curriculum (draw_reason only relevant when winner is None)
            draw_reason = msg.draw_reason if msg.winner is None else None
            curriculum.record_game(msg.winner, draw_reason)

            # Collect opening diagnostics for sidecar aggregation
            if msg.opening_diagnostics:
                all_opening_diagnostics.append(list(msg.opening_diagnostics))

            # Save game replay if enabled
            if game_saver is not None and msg.move_history is not None:
                # Map draw_reason int back to string (0=None, 1-4=draw reasons, 5=resign)
                # Note: resign has winner but also has draw_reason=5 for metadata
                draw_reason_str = {
                    0: None, 1: "timeout", 2: "board_full", 3: "state_cap", 4: "unknown", 5: "resign", 6: "adjudicated"
                }.get(msg.draw_reason)

                # Derive resigned_by from msg (resign means loser resigned)
                resigned_by = None
                if draw_reason_str == "resign" and msg.winner and msg.winner != "draw":
                    resigned_by = "black" if msg.winner == "red" else "red"

                game_saver.maybe_save_game(
                    winner=msg.winner if msg.winner != "draw" else None,
                    move_history=msg.move_history,
                    n_moves=msg.n_moves,
                    draw_reason=draw_reason_str,
                    start_player=msg.start_player,
                    resigned_by=resigned_by,
                    opening_diagnostics=list(msg.opening_diagnostics) if msg.opening_diagnostics else None,
                    opening_diagnostics_meta=msg.opening_diagnostics_meta,
                )

            if games_completed % 5 == 0:
                print(f"  Games: {games_completed}/{games_to_play}")

    # Consume positions until all workers done (or interrupted)
    # Wrap in try/finally to ensure worker cleanup on interrupt
    try:
        while workers_done < len(workers) and not interrupted:
            # Check for server errors and game completions
            while True:
                try:
                    msg = stats_queue.get_nowait()
                    process_stats_message(msg)
                except queue.Empty:
                    break

            # Get positions from workers
            try:
                item = position_queue.get(timeout=0.5)
            except queue.Empty:
                continue  # Also re-checks interrupted flag

            if isinstance(item, WorkerDone):
                workers_done += 1
                worker_done_stats[item.worker_id] = item
                print(f"  Worker {item.worker_id} done ({workers_done}/{len(workers)})")
                continue

            # item is a list of PositionRecord
            if isinstance(item, list):
                buffer.add_positions(item)
                new_positions.extend(item)
                total_positions += len(item)

        # Final stats drain
        while True:
            try:
                msg = stats_queue.get_nowait()
                process_stats_message(msg)
            except queue.Empty:
                break

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)

        # Always clean up workers, even on KeyboardInterrupt
        # Terminate workers first (don't wait for them to finish)
        for p in workers:
            if p.is_alive():
                p.terminate()
        # Give them a moment to die, then join
        for p in workers:
            p.join(timeout=1.0)

        # Stop inference server
        try:
            request_queue.put(StopSignal(), timeout=0.5)
        except queue.Full:
            pass
        server.stop()
        server_thread.join(timeout=1.0)

        # Clean up queues to prevent blocking on exit
        # cancel_join_thread() prevents Queue from blocking in atexit
        for q in [request_queue, position_queue, stats_queue]:
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass
        for q in response_queues.values():
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass

    # Re-raise KeyboardInterrupt after cleanup so caller can handle it
    if interrupted:
        raise KeyboardInterrupt()

    # ---- Two-line diagnostics: worker imbalance ----
    if worker_done_stats:
        # Line 1: games + positions
        parts1 = []
        for wid in sorted(worker_done_stats.keys()):
            s = worker_done_stats[wid]
            parts1.append(f"w{wid}={s.games_played}/{s.positions_sent}")
        print("  Worker summary (games,pos): " + " ".join(parts1))

        # Line 2: time + imbalance
        times = [(wid, worker_done_stats[wid].wall_time_s) for wid in worker_done_stats]
        times.sort(key=lambda x: x[1])
        min_t = times[0][1]
        max_w, max_t = times[-1]
        imbalance = (max_t / max(1e-9, min_t)) if min_t > 0 else float("inf")

        parts2 = [f"w{wid}={t:.1f}" for wid, t in sorted(times, key=lambda x: x[0])]
        print(
            "  Worker summary (time_s):    "
            + " ".join(parts2)
            + f" | imbalance={imbalance:.2f}x | straggler=w{max_w}"
        )

    # Build stats dict (simplified - parallel mode doesn't have per-game timing)
    stats = {
        "games_generated": games_completed,
        "positions_added": total_positions,
        "red_wins": red_wins,
        "black_wins": black_wins,
        "draws": draws,
        "timeout_draws": timeout_draws,
        "board_full_draws": board_full_draws,
        "state_cap_draws": state_cap_draws,
        "unknown_draws": unknown_draws,
        "resign_games": resign_games,
        "resigned_by_red": resigned_by_red,
        "resigned_by_black": resigned_by_black,
        "adjudicated_games": adjudicated_games,
        "adjudicated_red_wins": adjudicated_red_wins,
        "adjudicated_black_wins": adjudicated_black_wins,
        "adj_attempts": adj_attempts,
        "adj_blocked_ply": adj_blocked_ply,
        "adj_blocked_threshold": adj_blocked_threshold,
        "adj_blocked_visits": adj_blocked_visits,
        "adj_blocked_top1": adj_blocked_top1,
        "adj_abs_rv_samples": adj_abs_rv_samples,
        "adj_top1_samples": adj_top1_samples,
        "total_plies": total_plies,
        # Per-game accumulators (parallel-mode parity with sequential path —
        # outer loop assigns these to game_plies_list / game_durations so the
        # existing percentile code at the per-iter sanity block sees real data)
        "game_plies_list": game_plies_acc,
        "game_durations": game_durations_acc,
        # MCTS stats aggregated from workers
        "total_backups": total_backups,
        "total_nn_calls": total_nn_calls,
        "total_nn_batches": total_nn_batches,
        "total_waiters": total_waiters,
        "total_unique_leaves": total_unique_leaves,
        "max_waiters": max_waiters,
        "total_flush_full": total_flush_full,
        "total_flush_stall": total_flush_stall,
        "total_flush_tail": total_flush_tail,
        # Resign gate stats
        "rg_checks_red": rg_checks_red,
        "rg_checks_black": rg_checks_black,
        "rg_value_hits_red": rg_value_hits_red,
        "rg_value_hits_black": rg_value_hits_black,
        "rg_eligible_red": rg_eligible_red,
        "rg_eligible_black": rg_eligible_black,
        "rg_top1_all": rg_top1_all,
        # Phase 4: replay cap stats
        "total_positions_original": total_positions_original,
        "total_positions_kept": total_positions_kept,
        "games_capped": games_capped,
        "ply_bucket_edges": list(_PLY_BUCKET_EDGES),
        "ply_bucket_games": list(ply_bucket_games),
        "ply_bucket_positions_original": list(ply_bucket_positions_original),
        "ply_bucket_positions_kept": list(ply_bucket_positions_kept),
        # Termination-type + length breakdown (parity with sequential path).
        "positions_by_termination": dict(positions_by_termination),
        "positions_in_short_games": positions_in_short_games,
        "positions_in_long_games": positions_in_long_games,
        # Pre-existing bug fix: `all_opening_diagnostics` accumulated from
        # GameComplete IPC messages was previously dropped at function exit,
        # so parallel-worker runs never wrote opening_penalty_diagnostics /
        # root_child_diagnostics into the sidecar. Return the list so the
        # outer train() loop can feed it into the sidecar aggregation.
        "all_opening_diagnostics": list(all_opening_diagnostics),
    }

    return games_records, new_positions, stats


def train(
    n_iterations: int = 100,
    games_per_iteration: int = 25,
    train_steps_per_iteration: Optional[int] = None,
    batch_size: int = 64,
    buffer_size: int = 100000,
    checkpoint_dir: str = "checkpoints/alphazero",
    mcts_simulations: Optional[int] = None,
    learning_rate: float = 1e-3,
    value_lr_scale: float = 0.1,
    value_grad_max_norm: float = 0.5,
    l2_weight: float = 1e-4,
    value_weight: float = 0.5,
    progress_weighted: bool = True,
    progress_weight_floor: float = 0.25,
    hidden: int = 128,
    n_blocks: int = 6,
    max_moves: int = 200,
    resume_from: Optional[str] = None,
    load_weights_from: Optional[str] = None,
    seed: Optional[int] = None,
    progress_callback=None,
    metal_cache_limit: int = 2 * 1024 * 1024 * 1024,  # 2GB default cache limit
    # MCTS batching parameters
    mcts_eval_batch_size: int = 14,
    mcts_pending_virtual_visits: int = 8,
    mcts_stall_flush_sims: int = 16,
    # Curriculum learning parameters
    curriculum_sizes: tuple = (8, 10, 12, 16, 20, 24),
    curriculum_window: int = 200,
    curriculum_draw_threshold: float = 0.3,
    curriculum_min_wins: int = 5,
    # Multi-process self-play
    n_workers: int = 1,
    # Game replay saving
    save_games: bool = True,  # True = save all games to logs/games/, False = disabled
    games_dir_override: Optional[str] = None,  # Override games output directory
    # MCTS exploration tuning (None = use MCTSConfig defaults)
    dirichlet_alpha: Optional[float] = None,
    dirichlet_eps: Optional[float] = None,
    temp_high: Optional[float] = None,
    temp_low: Optional[float] = None,
    temp_threshold_ply: Optional[int] = None,
    # Opening exploration boost (None = use MCTSConfig defaults)
    opening_noise_ply: Optional[int] = None,
    opening_dirichlet_alpha: Optional[float] = None,
    opening_dirichlet_eps: Optional[float] = None,
    # Edge-band prior penalty (None = use MCTSConfig defaults)
    root_edge_band_penalty: Optional[float] = None,
    root_edge_band_penalty_ply: Optional[int] = None,
    root_edge_band_width: Optional[int] = None,
    # Near-corner prior penalty (None = use MCTSConfig defaults)
    root_near_corner_penalty: Optional[float] = None,
    root_near_corner_penalty_ply: Optional[int] = None,
    root_near_corner_radius: Optional[int] = None,
    # Phase 2: early-only near-corner penalty override (ply 0..early_plies-1)
    root_near_corner_penalty_early: Optional[float] = None,
    root_near_corner_penalty_early_plies: Optional[int] = None,
    # Resign parameters (conservative defaults = disabled)
    resign_enabled: bool = False,
    resign_min_ply: int = 80,
    resign_threshold: float = -0.97,
    resign_window: int = 12,
    resign_k: int = 8,
    resign_min_visits: int = 200,
    resign_min_top1_share: float = 0.0,
    # Adjudication parameters (disabled by default)
    adjudicate_enabled: bool = False,
    adjudicate_min_ply: int = 120,
    adjudicate_threshold: float = 0.90,
    adjudicate_min_visits: int = 200,
    adjudicate_min_top1_share: float = 0.0,
    adjudicate_debug: bool = False,
    # Phase 4: per-game replay contribution cap (0/None disables)
    max_positions_per_game: Optional[int] = None,
    endgame_keep_positions: int = 16,
    # Phase 2: inline forced-probe per-iter eval (additive observability)
    probes_path: str = "tests/probes/twixt_probes.json",
    probes_inline_disable: bool = False,
) -> AlphaZeroNetwork:
    """Full AlphaZero training loop with curriculum learning.

    Each iteration:
    1. Self-play: generate games with current network at curriculum active_size
    2. Add positions to replay buffer
    3. Train on random batches from buffer
    4. Check for curriculum promotion
    5. Checkpoint

    Args:
        n_iterations: Total training iterations
        games_per_iteration: Self-play games per iteration
        train_steps_per_iteration: Gradient updates per iteration
        batch_size: Positions per training step
        buffer_size: Max replay buffer capacity
        checkpoint_dir: Where to save checkpoints
        mcts_simulations: MCTS simulations per move (None=use SIMS_TABLE per size)
        learning_rate: Optimizer learning rate
        l2_weight: L2 regularization weight
        hidden: Network hidden channels
        n_blocks: Network residual blocks
        max_moves: Base max moves (scaled by curriculum)
        resume_from: Path to checkpoint to resume from
        seed: Random seed for reproducibility
        progress_callback: Optional callback(iteration, metrics)
        metal_cache_limit: Metal GPU cache limit in bytes (default 2GB)
        curriculum_sizes: Tuple of board sizes to progress through
        curriculum_window: Games window for curriculum metrics
        curriculum_draw_threshold: Max draw rate for promotion
        curriculum_min_wins: Min wins per color for promotion

    Returns:
        Trained network
    """
    from .mcts import MCTSConfig
    from .self_play import play_game
    from .local_evaluator import LocalGPUEvaluator

    # Set Metal cache limit to prevent memory overflow
    if mx.metal.is_available():
        mx.set_cache_limit(metal_cache_limit)
        limit_gb = metal_cache_limit / (1024 * 1024 * 1024)
        print(f"Metal cache limit set to {limit_gb:.1f}GB")

    # Setup directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Create network
    network = create_network(hidden=hidden, n_blocks=n_blocks)

    # Create evaluator (wraps network for MCTS)
    evaluator = LocalGPUEvaluator(network)

    # Create wrapper module that references encoder + policy_head
    # This ensures opt_main.update() mutates the live network params
    main_module = MainModule(network.encoder, network.policy_head)

    # Create TWO optimizers: main (encoder+policy) and value head
    opt_main = optim.Adam(learning_rate=learning_rate)
    value_lr = learning_rate * value_lr_scale
    opt_value = optim.Adam(learning_rate=value_lr)

    buffer = ReplayBuffer(max_size=buffer_size)

    # Build exploration overrides dict (only non-None values)
    mcts_exploration_overrides = {}
    if dirichlet_alpha is not None:
        mcts_exploration_overrides["dirichlet_alpha"] = dirichlet_alpha
    if dirichlet_eps is not None:
        mcts_exploration_overrides["dirichlet_eps"] = dirichlet_eps
    if temp_high is not None:
        mcts_exploration_overrides["temp_high"] = temp_high
    if temp_low is not None:
        mcts_exploration_overrides["temp_low"] = temp_low
    if temp_threshold_ply is not None:
        mcts_exploration_overrides["temp_threshold_ply"] = temp_threshold_ply
    if opening_noise_ply is not None:
        mcts_exploration_overrides["opening_noise_ply"] = opening_noise_ply
    if opening_dirichlet_alpha is not None:
        mcts_exploration_overrides["opening_dirichlet_alpha"] = opening_dirichlet_alpha
    if opening_dirichlet_eps is not None:
        mcts_exploration_overrides["opening_dirichlet_eps"] = opening_dirichlet_eps
    if root_edge_band_penalty is not None:
        mcts_exploration_overrides["root_edge_band_penalty"] = root_edge_band_penalty
    if root_edge_band_penalty_ply is not None:
        mcts_exploration_overrides["root_edge_band_penalty_ply"] = root_edge_band_penalty_ply
    if root_edge_band_width is not None:
        mcts_exploration_overrides["root_edge_band_width"] = root_edge_band_width
    if root_near_corner_penalty is not None:
        mcts_exploration_overrides["root_near_corner_penalty"] = root_near_corner_penalty
    if root_near_corner_penalty_ply is not None:
        mcts_exploration_overrides["root_near_corner_penalty_ply"] = root_near_corner_penalty_ply
    if root_near_corner_radius is not None:
        mcts_exploration_overrides["root_near_corner_radius"] = root_near_corner_radius
    # Phase 2: early-only near-corner override
    if root_near_corner_penalty_early is not None:
        mcts_exploration_overrides["root_near_corner_penalty_early"] = root_near_corner_penalty_early
    if root_near_corner_penalty_early_plies is not None:
        mcts_exploration_overrides["root_near_corner_penalty_early_plies"] = root_near_corner_penalty_early_plies

    # Default MCTS config (overridden per iteration with curriculum-scaled values)
    default_sims = mcts_simulations if mcts_simulations is not None else 400
    mcts_config = MCTSConfig(
        n_simulations=default_sims,
        eval_batch_size=mcts_eval_batch_size,
        pending_virtual_visits=mcts_pending_virtual_visits,
        stall_flush_sims=mcts_stall_flush_sims,
        **mcts_exploration_overrides,
    )

    # Create curriculum manager
    curriculum = CurriculumManager(
        sizes=curriculum_sizes,
        window=curriculum_window,
        draw_threshold=curriculum_draw_threshold,
        min_wins_each=curriculum_min_wins,
    )

    # Freeze state for timeout mitigation
    # Freeze blocks promotion AND reduces sims; preserved across size changes
    consecutive_high_timeout_iters = 0
    consecutive_good_timeout_iters = 0
    curriculum_frozen = False
    sims_reduction_factor = 1.0
    last_active_size = None

    # Promotion/demotion state (streak counters reset on size change)
    consecutive_promotable_iters = 0
    consecutive_demotable_iters = 0

    # Value head tripwire: warn if saturation detected 2+ consecutive iters
    consecutive_saturation_iters = 0

    # Balance tripwire: detect symmetry breaks (e.g., canonicalization bugs)
    consecutive_black_low_iters = 0  # black win rate < 25%
    recent_red_dominant = []  # rolling window of "red won more" flags (last 20 iters)
    BALANCE_WINDOW = 20
    BALANCE_BLACK_LOW_THRESHOLD = 0.25
    BALANCE_RED_DOMINANT_THRESHOLD = 0.80
    BALANCE_MIN_DECISIVE = 20  # minimum decisive games to trust balance stats
    BALANCE_MIN_DECISIVE_FRAC = 0.30  # minimum fraction of games that must be decisive
    BALANCE_CONSECUTIVE_BLACK_LOW = 3  # consecutive iters before warning

    # Value head instability tracking
    value_history: deque = deque(maxlen=VALUE_WINDOW)  # {eligible, level, trigger, p99, sat}
    value_warn_streak = 0  # consecutive warn/crit eligible iters
    value_instability_active = False  # noise control: only print rolling warning once per streak

    # Phase 2: load forced-tier probes once at startup; skipped silently if file
    # missing (Phase 0 of spec — probe suite curation — may not be done yet).
    forced_probes: List[dict] = []
    probes_load_status: str = "disabled"
    if not probes_inline_disable:
        if os.path.exists(probes_path):
            try:
                with open(probes_path) as _pf:
                    _probes_data = json.load(_pf)
                _all_probes = _probes_data.get("probes") or _probes_data.get("candidates") or []
                forced_probes = [p for p in _all_probes if p.get("confidence") == "forced"]
                probes_load_status = f"loaded ({len(forced_probes)} forced probes from {probes_path})"
            except Exception as _e:
                probes_load_status = f"failed to parse {probes_path}: {_e}"
        else:
            probes_load_status = (
                f"probes file not found at {probes_path} "
                "(Phase 0 not yet committed; per-iter Probe block will be skipped)"
            )
    print(f"  Inline forced-probe eval: {probes_load_status}")

    # Rolling window of recent forced-probe stats for delta + rolling-5 output
    forced_probe_history: deque = deque(maxlen=5)  # last 5 iters of {sign_correct_pct, median_abs_v}

    start_iteration = 0
    master_rng = random.Random(seed)

    # Load weights-only if specified (no state restore)
    if load_weights_from:
        network.load_weights(load_weights_from)
        print(f"Loaded weights-only from {load_weights_from} (no state restored)")

    # Resume from checkpoint if specified (full state restore)
    elif resume_from:
        network.load_weights(resume_from)
        state_path = Path(resume_from).with_suffix(".json")
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
                start_iteration = state.get("iteration", 0)
                # Restore curriculum state
                if "curriculum" in state:
                    curriculum = CurriculumManager.from_dict(state["curriculum"])
                # Restore freeze state (critical for resume consistency)
                if "freeze_state" in state:
                    fs = state["freeze_state"]
                    consecutive_high_timeout_iters = fs.get("consecutive_high_timeout_iters", 0)
                    consecutive_good_timeout_iters = fs.get("consecutive_good_timeout_iters", 0)
                    curriculum_frozen = fs.get("curriculum_frozen", False)
                    sims_reduction_factor = fs.get("sims_reduction_factor", 1.0)
                    consecutive_saturation_iters = fs.get("consecutive_saturation_iters", 0)
                    last_active_size = curriculum.active_size  # Avoid spurious reset
                # Restore curriculum state (promotion/demotion tracking)
                if "curriculum_state" in state:
                    cs = state["curriculum_state"]
                    consecutive_promotable_iters = cs.get("consecutive_promotable_iters", 0)
                    consecutive_demotable_iters = cs.get("consecutive_demotable_iters", 0)
        print(f"Resumed from {resume_from}, iteration {start_iteration}")
        print(f"  Curriculum: active_size={curriculum.active_size}")
        print(f"  Freeze: frozen={curriculum_frozen}, factor={sims_reduction_factor:.2f}")
        print(f"  Curriculum: promotable={consecutive_promotable_iters}, demotable={consecutive_demotable_iters}")

    # Generate unique run ID for this process (used for metrics tracking)
    run_id = generate_run_id()

    print(f"Starting training: {n_iterations} iterations")
    print(f"  Run ID: {run_id}")
    print(f"  Games/iter: {games_per_iteration}")
    print(f"  Train steps/iter: {train_steps_per_iteration}")
    print(f"  Batch size: {batch_size}")
    print(f"  Buffer size: {buffer_size}")
    print(f"  MCTS simulations: {mcts_simulations or 'SIMS_TABLE (per size)'}")
    print(f"  Learning rate: {learning_rate} (value head: {value_lr})")
    print(f"  Value grad max norm: {value_grad_max_norm}")
    print(f"  Value weight: {value_weight} (target, warmup applies)")
    print(f"  Value weight: {value_weight} (progress_weighted={progress_weighted}, floor={progress_weight_floor})")  # Phase 2 banner
    print(f"  Curriculum sizes: {curriculum_sizes}")
    print(f"  Starting active_size: {curriculum.active_size}")
    print(f"  Sims policy: {'CLI override' if mcts_simulations else 'SIMS_TABLE'}")
    print(f"  Workers: {n_workers}" + (" (parallel)" if n_workers > 1 else " (sequential)"))
    print(f"  Save games: {save_games}")

    # Games directory (used for both game replays and per-iteration stats sidecars)
    if games_dir_override:
        games_dir = Path(games_dir_override)
    else:
        games_dir = Path(__file__).parent.parent / "logs" / "games"

    # Initialize game saver for replay files
    game_saver = None
    if save_games:
        game_saver = GameSaver(
            games_dir=games_dir,
            max_games_per_iter=999999,  # Effectively unlimited
        )
        print(f"  Games dir: {games_dir}")

    # Spawn caffeinate on macOS to prevent system sleep during training
    caffeinate_proc = None
    if platform.system() == "Darwin":
        try:
            caffeinate_proc = subprocess.Popen(
                ["caffeinate", "-i"],  # -i = prevent idle sleep
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"  Caffeinate: enabled (PID {caffeinate_proc.pid})")
        except Exception as e:
            print(f"  Caffeinate: failed to start ({e})")

    def _stop_caffeinate():
        """Kill caffeinate process if running."""
        nonlocal caffeinate_proc
        if caffeinate_proc is not None:
            try:
                caffeinate_proc.terminate()
                caffeinate_proc.wait(timeout=2)
                print("  Caffeinate: stopped")
            except Exception:
                pass
            caffeinate_proc = None

    for iteration in range(start_iteration, n_iterations):
        iter_start = time.perf_counter()

        # Get curriculum parameters for this iteration
        active_size = curriculum.active_size
        scaled_max_moves = get_scaled_max_moves(active_size)

        # Reset game saver for this iteration (after we know active_size)
        if game_saver is not None:
            game_saver.set_iteration(iteration, active_size=active_size)

        # Reset ONLY streak counters on size change (NOT freeze state!)
        # This prevents streak counters from leaking across size boundaries
        # but keeps freeze protection active through size changes
        if active_size != last_active_size and last_active_size is not None:
            consecutive_high_timeout_iters = 0
            consecutive_good_timeout_iters = 0
            consecutive_saturation_iters = 0
            consecutive_promotable_iters = 0
            consecutive_demotable_iters = 0
            print(f"  Size changed: {last_active_size} -> {active_size}, streaks reset")
            # NOTE: curriculum_frozen and sims_reduction_factor are PRESERVED
            #       to maintain freeze protection through size changes
        last_active_size = active_size

        # Compute sims with full transparency (once at iteration start)
        # CLI overrides table when specified; otherwise use table defaults
        base_sims_from_table = SIMS_TABLE.get(active_size, 400)  # fallback 400 if size not in table
        if mcts_simulations is not None:
            # CLI specified: use CLI value as override
            base_sims_effective = mcts_simulations
        else:
            # CLI not specified: use table value
            base_sims_effective = base_sims_from_table

        # Apply freeze factor
        sims_after_freeze = int(base_sims_effective * sims_reduction_factor)

        # Apply absolute floor (100 sims minimum)
        effective_sims = max(ABS_SIMS_FLOOR, sims_after_freeze)
        sims_clamped_to_floor = (sims_after_freeze < ABS_SIMS_FLOOR)

        # Determine why we got this value (primary reason)
        if sims_clamped_to_floor:
            effective_reason = "abs_floor"
        elif sims_reduction_factor < 1.0:
            effective_reason = "freeze"
        elif mcts_simulations is not None:
            effective_reason = "cli"
        else:
            effective_reason = "table"

        sims_used = effective_sims

        # Update game saver with simulations count
        if game_saver is not None:
            game_saver.simulations = sims_used

        # Compute scaled train steps for this size
        scaled_train_steps = get_scaled_train_steps(active_size, train_steps_per_iteration)

        # Create MCTS config for this iteration with scaled sims
        iter_mcts_config = MCTSConfig(
            n_simulations=sims_used,
            eval_batch_size=mcts_eval_batch_size,
            pending_virtual_visits=mcts_pending_virtual_visits,
            stall_flush_sims=mcts_stall_flush_sims,
            **mcts_exploration_overrides,
        )

        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{n_iterations}")
        print(f"  Curriculum: active_size={active_size}, max_moves={scaled_max_moves}")
        cli_str = str(mcts_simulations) if mcts_simulations is not None else "-"
        print(f"  Sims: cli={cli_str}, table={base_sims_from_table}, "
              f"factor={sims_reduction_factor:.2f}, effective={sims_used} ({effective_reason})")
        print(f"{'='*60}")

        # 1. Self-play (inference mode - freeze BN behavior)
        selfplay_start = time.perf_counter()
        network.eval()
        print(f"\nSelf-play: generating {games_per_iteration} games" +
              (f" (parallel, {n_workers} workers)..." if n_workers > 1 else "..."))

        # Initialize tracking variables
        games_generated = 0
        positions_added = 0
        red_wins = 0
        black_wins = 0
        draws = 0
        timeout_draws = 0
        board_full_draws = 0
        state_cap_draws = 0
        unknown_draws = 0
        # Resign tracking (decisive, not draw)
        resign_games = 0
        resigned_by_red = 0
        resigned_by_black = 0
        # Adjudication tracking (decisive, not draw)
        adjudicated_games = 0
        adjudicated_red_wins = 0
        adjudicated_black_wins = 0
        adj_blocked_ply = 0
        adj_blocked_threshold = 0
        adj_blocked_visits = 0
        adj_blocked_top1 = 0
        adj_attempts = 0
        adj_abs_rv_samples = []
        adj_top1_samples = []
        # Resign gate aggregation
        rg_checks_red = 0;    rg_checks_black = 0
        rg_value_hits_red = 0; rg_value_hits_black = 0
        rg_eligible_red = 0;   rg_eligible_black = 0
        rg_top1_all = []
        all_opening_diagnostics = []  # Collect per-game diagnostic lists for sidecar aggregation
        total_nn_calls = 0
        total_expand_calls = 0
        total_nn_batches = 0
        total_plies = 0
        total_backups = 0
        total_waiters = 0
        total_unique_leaves = 0
        max_waiters = 0
        total_flush_full = 0
        total_flush_stall = 0
        total_flush_tail = 0
        selfplay_progress = []
        new_positions = []  # Collect this iteration's positions for sanity stats
        game_plies_list = []  # Per-game ply counts for percentiles
        game_durations = []  # Per-game wall times
        interrupted = False

        # Phase 4: per-game replay cap accounting (iteration-level).
        # The parallel path populates these via parallel_stats; the sequential
        # path accumulates them inline from game records. Either way these
        # feed the sidecar + CSV.
        total_positions_original_iter = 0
        total_positions_kept_iter = 0
        games_capped_iter = 0
        # Phase 1 (2026-04-19): replay-cap termination-type breakdown
        positions_by_termination_iter = {"win": 0, "resign": 0, "adjudicated": 0, "timeout": 0}
        positions_in_short_games_iter = 0   # games with n_moves <= 80
        positions_in_long_games_iter = 0    # games with n_moves > 200
        _PLY_BUCKET_EDGES_ITER = (40, 80, 120, 160, 200)
        _n_buckets_iter = len(_PLY_BUCKET_EDGES_ITER) + 1
        ply_bucket_games_iter = [0] * _n_buckets_iter
        ply_bucket_positions_original_iter = [0] * _n_buckets_iter
        ply_bucket_positions_kept_iter = [0] * _n_buckets_iter

        def _ply_bucket_index_iter(n_moves: int) -> int:
            for i, edge in enumerate(_PLY_BUCKET_EDGES_ITER):
                if n_moves < edge:
                    return i
            return _n_buckets_iter - 1

        if n_workers > 1:
            # === PARALLEL SELF-PLAY ===
            try:
                _, new_positions, parallel_stats = run_parallel_selfplay(
                    evaluator=evaluator,
                    mcts_config=iter_mcts_config,
                    games_to_play=games_per_iteration,
                    n_workers=n_workers,
                    master_rng=master_rng,
                    max_moves=scaled_max_moves,
                    active_size=active_size,
                    curriculum=curriculum,
                    buffer=buffer,
                    game_saver=game_saver,
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

                # Unpack stats
                games_generated = parallel_stats["games_generated"]
                positions_added = parallel_stats["positions_added"]
                red_wins = parallel_stats["red_wins"]
                black_wins = parallel_stats["black_wins"]
                draws = parallel_stats["draws"]
                timeout_draws = parallel_stats["timeout_draws"]
                board_full_draws = parallel_stats["board_full_draws"]
                state_cap_draws = parallel_stats["state_cap_draws"]
                unknown_draws = parallel_stats["unknown_draws"]
                resign_games = parallel_stats.get("resign_games", 0)
                resigned_by_red = parallel_stats.get("resigned_by_red", 0)
                resigned_by_black = parallel_stats.get("resigned_by_black", 0)
                adjudicated_games = parallel_stats.get("adjudicated_games", 0)
                adjudicated_red_wins = parallel_stats.get("adjudicated_red_wins", 0)
                adjudicated_black_wins = parallel_stats.get("adjudicated_black_wins", 0)
                adj_attempts = parallel_stats.get("adj_attempts", 0)
                adj_blocked_ply = parallel_stats.get("adj_blocked_ply", 0)
                adj_blocked_threshold = parallel_stats.get("adj_blocked_threshold", 0)
                adj_blocked_visits = parallel_stats.get("adj_blocked_visits", 0)
                adj_blocked_top1 = parallel_stats.get("adj_blocked_top1", 0)
                adj_abs_rv_samples = parallel_stats.get("adj_abs_rv_samples", [])
                adj_top1_samples = parallel_stats.get("adj_top1_samples", [])
                total_plies = parallel_stats["total_plies"]
                # Per-game ply/timing lists from parallel workers (drives
                # p95_plies / max_plies_observed / avg_game_seconds / p95_game_seconds)
                game_plies_list = parallel_stats.get("game_plies_list", [])
                game_durations = parallel_stats.get("game_durations", [])
                # MCTS stats not available in parallel mode
                total_backups = parallel_stats.get("total_backups", 0)
                total_nn_calls = parallel_stats.get("total_nn_calls", 0)
                total_nn_batches = parallel_stats.get("total_nn_batches", 0)
                total_waiters = parallel_stats.get("total_waiters", 0)
                total_unique_leaves = parallel_stats.get("total_unique_leaves", 0)
                max_waiters = parallel_stats.get("max_waiters", 0)
                total_flush_full = parallel_stats.get("total_flush_full", 0)
                total_flush_stall = parallel_stats.get("total_flush_stall", 0)
                total_flush_tail = parallel_stats.get("total_flush_tail", 0)
                # Resign gate stats
                rg_checks_red = parallel_stats.get("rg_checks_red", 0)
                rg_checks_black = parallel_stats.get("rg_checks_black", 0)
                rg_value_hits_red = parallel_stats.get("rg_value_hits_red", 0)
                rg_value_hits_black = parallel_stats.get("rg_value_hits_black", 0)
                rg_eligible_red = parallel_stats.get("rg_eligible_red", 0)
                rg_eligible_black = parallel_stats.get("rg_eligible_black", 0)
                rg_top1_all = parallel_stats.get("rg_top1_all", [])
                # Phase 4: pull replay-cap stats from parallel stats
                total_positions_original_iter = parallel_stats.get("total_positions_original", 0)
                total_positions_kept_iter = parallel_stats.get("total_positions_kept", 0)
                games_capped_iter = parallel_stats.get("games_capped", 0)
                ply_bucket_games_iter = parallel_stats.get("ply_bucket_games", ply_bucket_games_iter)
                ply_bucket_positions_original_iter = parallel_stats.get(
                    "ply_bucket_positions_original", ply_bucket_positions_original_iter
                )
                ply_bucket_positions_kept_iter = parallel_stats.get(
                    "ply_bucket_positions_kept", ply_bucket_positions_kept_iter
                )
                # Termination-type + length breakdown — previously dropped in
                # parallel mode, leaving sidecar.replay_cap.positions_by_termination
                # / _short / _long at all-zero. See trainer.py :2342-2354 for the
                # sequential-path logic this mirrors.
                _par_pbt = parallel_stats.get("positions_by_termination") or {}
                for _term in positions_by_termination_iter:
                    positions_by_termination_iter[_term] = int(
                        _par_pbt.get(_term, 0) or 0
                    )
                positions_in_short_games_iter = int(
                    parallel_stats.get("positions_in_short_games", 0) or 0
                )
                positions_in_long_games_iter = int(
                    parallel_stats.get("positions_in_long_games", 0) or 0
                )
                # Pull opening-diagnostics accumulated inside the parallel loop
                # (previously dropped, causing empty sidecar opening/root_child blocks).
                _par_od = parallel_stats.get("all_opening_diagnostics", [])
                if _par_od:
                    all_opening_diagnostics.extend(_par_od)

            except KeyboardInterrupt:
                print(f"\n\nInterrupted during parallel self-play!")
                print("Saving partial checkpoint and exiting...")
                interrupted = True
            except Exception as e:
                print(f"\n\nError during parallel self-play: {type(e).__name__}: {e}")
                print("Saving partial checkpoint and exiting...")
                interrupted = True

        else:
            # === SEQUENTIAL SELF-PLAY ===
            try:
                for g in range(games_per_iteration):
                    game_rng = random.Random(master_rng.randint(0, 2**31))

                    # Randomize starting player
                    start_player = "red" if game_rng.random() < 0.5 else "black"

                    game_t0 = time.perf_counter()
                    game = play_game(
                        evaluator,
                        mcts_config=iter_mcts_config,  # Use scaled sims for this iteration
                        rng=game_rng,
                        max_moves=scaled_max_moves,  # Use curriculum-scaled max_moves
                        add_noise=True,
                        active_size=active_size,  # Curriculum board size
                        start_player=start_player,
                        game_id=g,
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
                    game_dur = time.perf_counter() - game_t0
                    game_plies_list.append(game.n_moves)
                    game_durations.append(game_dur)
                    buffer.add_game(game)
                    new_positions.extend(game.positions)  # Collect for sanity stats
                    games_generated += 1
                    positions_added += len(game.positions)

                    # Phase 4: per-game replay cap accounting.
                    # n_positions_original may be 0 on older records; fall back
                    # to live positions count in that case.
                    _n_orig = game.n_positions_original or len(game.positions)
                    _n_kept = game.n_positions_kept or len(game.positions)
                    total_positions_original_iter += _n_orig
                    total_positions_kept_iter += _n_kept
                    if _n_kept < _n_orig:
                        games_capped_iter += 1
                    _bi = _ply_bucket_index_iter(game.n_moves)
                    ply_bucket_games_iter[_bi] += 1
                    ply_bucket_positions_original_iter[_bi] += _n_orig
                    ply_bucket_positions_kept_iter[_bi] += _n_kept
                    # Phase 1 (2026-04-19): termination-type classification.
                    # Map game outcome to a bucket so we can see which kinds of
                    # games dominate the training set.
                    # - winner + RESIGN draw_reason → "resign"
                    # - winner + ADJUDICATED draw_reason → "adjudicated"
                    # - winner with no draw_reason → "win" (natural in-game terminal)
                    # - no winner (DRAW_TIMEOUT / DRAW_BOARD_FULL / DRAW_STATE_CAP / DRAW_UNKNOWN) → "timeout"
                    if game.winner and game.draw_reason == RESIGN:
                        _term = "resign"
                    elif game.winner and game.draw_reason == ADJUDICATED:
                        _term = "adjudicated"
                    elif game.winner:
                        _term = "win"
                    else:
                        _term = "timeout"
                    positions_by_termination_iter[_term] += _n_kept
                    if game.n_moves <= 80:
                        positions_in_short_games_iter += _n_kept
                    elif game.n_moves > 200:
                        positions_in_long_games_iter += _n_kept
                    total_nn_calls += game.nn_calls
                    total_expand_calls += game.expand_calls
                    total_nn_batches += game.nn_batches
                    total_plies += game.n_moves
                    total_backups += game.total_backups
                    total_waiters += game.total_waiters
                    total_unique_leaves += game.unique_leaves
                    max_waiters = max(max_waiters, game.max_waiters)
                    total_flush_full += game.flush_full
                    total_flush_stall += game.flush_stall
                    total_flush_tail += game.flush_tail

                    # Track results for display
                    if game.winner == "red":
                        red_wins += 1
                    elif game.winner == "black":
                        black_wins += 1
                    else:
                        draws += 1
                        # Track draw breakdown for display (using constants)
                        if game.draw_reason == DRAW_TIMEOUT:
                            timeout_draws += 1
                        elif game.draw_reason == DRAW_BOARD_FULL:
                            board_full_draws += 1
                        elif game.draw_reason == DRAW_STATE_CAP:
                            state_cap_draws += 1
                        else:
                            unknown_draws += 1

                    # Track resign (separate from draw breakdown since resign is decisive)
                    if game.draw_reason == RESIGN:
                        resign_games += 1
                        # Defensive: guard against missing resigned_by
                        if getattr(game, "resigned_by", None) == "red":
                            resigned_by_red += 1
                        elif getattr(game, "resigned_by", None) == "black":
                            resigned_by_black += 1

                    # Track adjudication (separate from draw, decisive)
                    if game.draw_reason == ADJUDICATED:
                        adjudicated_games += 1
                        if game.winner == "red":
                            adjudicated_red_wins += 1
                        elif game.winner == "black":
                            adjudicated_black_wins += 1

                    # Adjudication diagnostics aggregation
                    if game.adj_attempted:
                        adj_attempts += 1
                        if game.adj_abs_rv is not None:
                            adj_abs_rv_samples.append(game.adj_abs_rv)
                        if game.adj_top1 is not None:
                            adj_top1_samples.append(game.adj_top1)
                        if game.adj_blocked_by == "ply":
                            adj_blocked_ply += 1
                        elif game.adj_blocked_by == "threshold":
                            adj_blocked_threshold += 1
                        elif game.adj_blocked_by == "visits":
                            adj_blocked_visits += 1
                        elif game.adj_blocked_by == "top1":
                            adj_blocked_top1 += 1

                    # Resign gate aggregation
                    rg_checks_red += game.rg_checks_red
                    rg_checks_black += game.rg_checks_black
                    rg_value_hits_red += game.rg_value_hits_red
                    rg_value_hits_black += game.rg_value_hits_black
                    rg_eligible_red += game.rg_eligible_red
                    rg_eligible_black += game.rg_eligible_black
                    rg_top1_all.extend(game.rg_top1_samples)

                    # Always record to curriculum (draw_reason is None for wins)
                    curriculum.record_game(game.winner, game.draw_reason)

                    # Collect opening diagnostics for sidecar aggregation
                    if game.opening_diagnostics:
                        all_opening_diagnostics.append(game.opening_diagnostics)

                    # Save game replay if enabled
                    if game_saver is not None and game.move_history:
                        move_history_tuple = tuple(tuple(m) for m in game.move_history)
                        game_saver.maybe_save_game(
                            winner=game.winner,
                            move_history=move_history_tuple,
                            n_moves=game.n_moves,
                            draw_reason=game.draw_reason,
                            start_player=game.start_player,
                            resigned_by=game.resigned_by,
                            opening_diagnostics=game.opening_diagnostics if game.opening_diagnostics else None,
                            opening_diagnostics_meta=game.opening_diagnostics_meta,
                        )

                    # Flush MLX graph and clear caches after each game
                    # mx.eval() forces pending lazy ops to materialize, freeing intermediates
                    mx.eval()
                    gc.collect()
                    mx.clear_cache()

                    if (g + 1) % 5 == 0 or g == games_per_iteration - 1:
                        # Memory telemetry
                        active_mb = mx.get_active_memory() / (1024 * 1024)
                        cache_mb = mx.get_cache_memory() / (1024 * 1024)
                        print(
                            f"  Games: {g+1}/{games_per_iteration}, "
                            f"Buffer: {len(buffer)} positions, "
                            f"GPU: {active_mb:.0f}MB active, {cache_mb:.0f}MB cache"
                        )
                        # Capture progress snapshot (same cadence as print)
                        selfplay_progress.append({
                            "games_done": g + 1,
                            "buffer_size": len(buffer),
                            "elapsed_s": time.perf_counter() - selfplay_start,
                        })
            except KeyboardInterrupt:
                print(f"\n\nInterrupted during self-play! Completed {games_generated}/{games_per_iteration} games.")
                print("Saving partial checkpoint and exiting...")
                interrupted = True

        # Print self-play summary (even if interrupted with partial data)
        if games_generated > 0:
            print(f"  Generated {games_generated} games, {positions_added} positions")
            print(f"  Results: Red={red_wins}, Black={black_wins}, Draws={draws}")
            print(f"    Draw breakdown: timeout={timeout_draws}, board_full={board_full_draws}, "
                  f"state_cap={state_cap_draws}, unknown={unknown_draws}")
            if resign_games > 0:
                print(f"    Resign: {resign_games} (by_red={resigned_by_red}, by_black={resigned_by_black})")
            if adjudicated_games > 0 or (adjudicate_enabled and timeout_draws > 0):
                print(f"    Adjudicated: {adjudicated_games} (red_wins={adjudicated_red_wins}, black_wins={adjudicated_black_wins}, remaining_timeouts={timeout_draws})")
                if adj_attempts > 0:
                    print(f"    Adjudication blocks: ply={adj_blocked_ply} thr={adj_blocked_threshold} visits={adj_blocked_visits} top1={adj_blocked_top1} (attempts={adj_attempts})")
                    if adj_abs_rv_samples:
                        rv_arr = np.array(adj_abs_rv_samples)
                        t1_arr = np.array(adj_top1_samples) if adj_top1_samples else np.array([0.0])
                        print(f"    Adj stats: abs_rv p50={np.median(rv_arr):.3f} p90={np.percentile(rv_arr, 90):.3f} top1 p50={np.median(t1_arr):.3f} p10={np.percentile(t1_arr, 10):.3f}")
            rg_checks = rg_checks_red + rg_checks_black
            if rg_checks > 0:
                rg_vhits = rg_value_hits_red + rg_value_hits_black
                rg_elig = rg_eligible_red + rg_eligible_black
                rg_blocked = rg_vhits - rg_elig
                print(f"    Resign gate (plies >= {resign_min_ply}):")
                print(f"      checks={rg_checks} (red={rg_checks_red}, black={rg_checks_black})")
                print(f"      value_hits={rg_vhits} (red={rg_value_hits_red}, black={rg_value_hits_black})")
                if resign_min_top1_share > 0:
                    print(f"      blocked_by_top1={rg_blocked} (red={rg_value_hits_red - rg_eligible_red}, black={rg_value_hits_black - rg_eligible_black})  [min_top1={resign_min_top1_share}]")
                print(f"      eligible_hits={rg_elig} (red={rg_eligible_red}, black={rg_eligible_black})")
                if rg_top1_all:
                    arr = sorted(rg_top1_all)
                    n = len(arr)
                    p50 = arr[n // 2]
                    p90 = arr[int(n * 0.9)]
                    p99 = arr[int(n * 0.99)]
                    print(f"      top1_share_on_value_hits: p50={p50:.2f} p90={p90:.2f} p99={p99:.2f}")
            print(f"  Buffer size: {len(buffer)}")
            avg_plies = total_plies / games_generated

            # === Balance tripwire: detect symmetry breaks ===
            total_decisive = red_wins + black_wins
            total_games = red_wins + black_wins + draws
            draw_rate = draws / total_games if total_games > 0 else 0.0
            decisive_frac = total_decisive / total_games if total_games > 0 else 0.0

            if total_decisive >= BALANCE_MIN_DECISIVE and decisive_frac >= BALANCE_MIN_DECISIVE_FRAC:
                iter_black_rate = black_wins / total_decisive
                iter_red_rate = red_wins / total_decisive
                iter_red_dominant = red_wins > black_wins

                print(f"  Balance: black={iter_black_rate:.1%}, red={iter_red_rate:.1%}, draw={draw_rate:.1%} "
                      f"(decisive={total_decisive}/{total_games}, window={len(recent_red_dominant)}/{BALANCE_WINDOW})")

                # Track consecutive low black win rate
                if iter_black_rate < BALANCE_BLACK_LOW_THRESHOLD:
                    consecutive_black_low_iters += 1
                    if consecutive_black_low_iters >= BALANCE_CONSECUTIVE_BLACK_LOW:
                        print(f"  ⚠️  BALANCE WARNING: black decisive win rate {iter_black_rate:.1%} "
                              f"(<{BALANCE_BLACK_LOW_THRESHOLD:.0%}) for {consecutive_black_low_iters} consecutive iters | "
                              f"decisive={total_decisive}/{total_games} ({decisive_frac:.0%}), draw={draw_rate:.0%}")
                else:
                    consecutive_black_low_iters = 0

                # Track rolling red-dominant rate
                recent_red_dominant.append(iter_red_dominant)
                if len(recent_red_dominant) > BALANCE_WINDOW:
                    recent_red_dominant.pop(0)

                if len(recent_red_dominant) >= BALANCE_WINDOW:
                    red_dominant_rate = sum(recent_red_dominant) / len(recent_red_dominant)
                    if red_dominant_rate > BALANCE_RED_DOMINANT_THRESHOLD:
                        print(f"  ⚠️  BALANCE WARNING: Red dominated {red_dominant_rate:.0%} of "
                              f"last {BALANCE_WINDOW} eligible iters (>{BALANCE_RED_DOMINANT_THRESHOLD:.0%}) | "
                              f"decisive={total_decisive}/{total_games} ({decisive_frac:.0%}), draw={draw_rate:.0%}")
            else:
                # Not enough signal this iter; avoid building false consecutive streaks
                consecutive_black_low_iters = 0
                print(f"  Balance: skipped (decisive={total_decisive}/{total_games}, draw={draw_rate:.1%})")

            # Compute per-game timing percentiles
            if game_plies_list:
                plies_arr = np.array(game_plies_list)
                p95_plies = float(np.percentile(plies_arr, 95))
                max_plies_observed = int(np.max(plies_arr))
            else:
                p95_plies = 0.0
                max_plies_observed = 0

            if game_durations:
                durations_arr = np.array(game_durations)
                avg_game_seconds = float(np.mean(durations_arr))
                p95_game_seconds = float(np.percentile(durations_arr, 95))
            else:
                avg_game_seconds = 0.0
                p95_game_seconds = 0.0
            avg_batch = total_nn_calls / total_nn_batches if total_nn_batches > 0 else 0
            avg_waiters = total_waiters / total_unique_leaves if total_unique_leaves > 0 else 0
            print(f"  Backups: {total_backups}, Leaf evals: {total_nn_calls}, NN batches: {total_nn_batches}")
            print(f"  Avg batch: {avg_batch:.1f}, Avg waiters: {avg_waiters:.1f}, Max waiters: {max_waiters}")
            print(f"  Flushes: full={total_flush_full}, stall={total_flush_stall}, tail={total_flush_tail}")
            print(f"  Avg plies: {avg_plies:.1f}")

            # Compute sanity stats from this iteration's positions
            z_stats = summarize_z(new_positions)
            pi_stats = summarize_policy_sanity(new_positions)
            v_stats = summarize_value_sanity(
                network, new_positions, active_size,
                sample_n=256, seed=iteration
            )

            # Print sanity summary
            print(f"\n  Sanity ({len(new_positions)} positions):")
            print(f"    z: mean={z_stats['z_mean']:.3f}, std={z_stats['z_std']:.3f}, "
                  f"[+/0/-]={z_stats['z_count_pos']}/{z_stats['z_count_zero']}/{z_stats['z_count_neg']}")
            if z_stats.get('z_bad_value_count', 0) > 0:
                print(f"    z WARNING: {z_stats['z_bad_value_count']} values not in {{-1,0,+1}}!")
            print(f"    z by to_move: red={z_stats['z_mean_to_move_red']:.3f} "
                  f"(+/0/-={z_stats['z_count_pos_to_move_red']}/{z_stats['z_count_zero_to_move_red']}/{z_stats['z_count_neg_to_move_red']}), "
                  f"black={z_stats['z_mean_to_move_black']:.3f} "
                  f"(+/0/-={z_stats['z_count_pos_to_move_black']}/{z_stats['z_count_zero_to_move_black']}/{z_stats['z_count_neg_to_move_black']})")
            print(f"    v: mean={v_stats.get('v_pred_mean',0):.3f}, std={v_stats.get('v_pred_std',0):.3f}, "
                  f"range=[{v_stats.get('v_pred_min',0):.2f},{v_stats.get('v_pred_max',0):.2f}], "
                  f"mse={v_stats.get('v_mse_vs_z',0):.4f}, sign_agree={v_stats.get('v_sign_agree',0):.1%}")
            print(f"    v cls: labels(+/-)={v_stats.get('v_label_pos',0)}/{v_stats.get('v_label_neg',0)}, "
                  f"maj_base={v_stats.get('v_maj_baseline',0):.1%}, "
                  f"bal_acc={v_stats.get('v_bal_acc',0):.1%}, mcc={v_stats.get('v_mcc',0):.3f}")
            print(f"    v pretanh: range=[{v_stats.get('v_pre_min',0):.2f},{v_stats.get('v_pre_max',0):.2f}], "
                  f"p99={v_stats.get('v_pre_abs_p99',0):.2f}, frac_sat={v_stats.get('v_frac_sat',0):.3f}, "
                  f"zv_corr={v_stats.get('v_zv_corr',0):.3f} (n={v_stats.get('v_non_draw_n',0)})")
            # Phase 2 connectivity-bucketed sanity (channels 24-29)
            sbc = v_stats.get("sanity_by_connectivity", {})
            if sbc:
                ws = sbc.get("winning_structure", {})
                ns = sbc.get("no_winning_structure", {})
                ws_n = ws.get("n", 0)
                ns_n = ns.get("n", 0)
                if ws_n + ns_n > 0:
                    print(f"    Sanity by connectivity (threshold: largest_component>={sbc.get('winning_size_threshold', 8)} "
                          f"OR n_goal_touching>=2):")
                    def _fmt(b, n):
                        if n == 0:
                            return "(n=0)"
                        sa = b.get("sign_agree")
                        mv = b.get("median_abs_v")
                        sa_s = f"{sa:.1%}" if sa is not None else "n/a"
                        mv_s = f"{mv:.3f}" if mv is not None else "n/a"
                        return f"(n={n}): sign_agree={sa_s}, median |v|={mv_s}"
                    print(f"      winning_structure    {_fmt(ws, ws_n)}")
                    print(f"      no_winning_structure {_fmt(ns, ns_n)}")
                else:
                    # 24-channel checkpoint or empty sample — silent skip
                    pass
            if v_stats.get('v_z_batch_mismatch', 0) > 0:
                print(f"    v WARNING: z_batch_mismatch={v_stats['v_z_batch_mismatch']} "
                      "(make_padded_batch outcomes differ from rec.outcome!)")

            # Phase 2: inline forced-probe NN-only eval (additive observability)
            forced_probe_summary: Optional[dict] = None
            if probes_inline_disable:
                pass  # explicitly disabled
            elif not forced_probes:
                # File missing or empty — print a one-line stub so the row exists
                # for log-grep continuity, but only on the first iter to avoid spam.
                if iteration == start_iteration:
                    print(f"  Probe (forced, NN-only): (skipped — {probes_load_status})")
            else:
                from .probe_eval import run_forced_probes_inline
                _probe_res = run_forced_probes_inline(
                    network, forced_probes, active_size=active_size
                )
                if _probe_res["n"] == 0:
                    print(f"  Probe (forced, NN-only): (skipped: no probes for "
                          f"active_size={active_size}, {_probe_res['n_skipped_size']} probes "
                          f"available at other sizes)")
                else:
                    sc = _probe_res["sign_correct"]
                    n = _probe_res["n"]
                    sc_pct = _probe_res["sign_correct_pct"] or 0.0
                    mv = _probe_res["median_abs_v"]
                    # Rolling-5 (excludes current iter; window updated below)
                    rolling = list(forced_probe_history)
                    if rolling:
                        r_pcts = [r["sign_correct_pct"] for r in rolling
                                 if r.get("sign_correct_pct") is not None]
                        r_mvs = [r["median_abs_v"] for r in rolling
                                if r.get("median_abs_v") is not None]
                        roll_pct = sum(r_pcts) / len(r_pcts) if r_pcts else None
                        roll_mv = sum(r_mvs) / len(r_mvs) if r_mvs else None
                    else:
                        roll_pct = None
                        roll_mv = None
                    # Delta vs immediately-prior iter
                    if rolling:
                        prev = rolling[-1]
                        prev_pct = prev.get("sign_correct_pct")
                        prev_mv = prev.get("median_abs_v")
                        delta_pct = (sc_pct - prev_pct) if prev_pct is not None else None
                        delta_mv = (mv - prev_mv) if (mv is not None and prev_mv is not None) else None
                    else:
                        delta_pct = None
                        delta_mv = None

                    print(f"  Probe (forced, NN-only, n={n}):")
                    mv_s = f"{mv:.3f}" if mv is not None else "n/a"
                    print(f"    sign_correct={sc}/{n} ({sc_pct:.1%}), median |v|={mv_s}")
                    if delta_pct is not None:
                        d_pct_s = f"{delta_pct*100:+.1f}pp"
                        d_mv_s = f"{delta_mv:+.3f}" if delta_mv is not None else "n/a"
                        print(f"    delta vs prev: sign {d_pct_s}, |v| {d_mv_s}")
                    if roll_pct is not None:
                        roll_mv_s = f"{roll_mv:.3f}" if roll_mv is not None else "n/a"
                        print(f"    rolling(5 prior): sign={roll_pct:.1%}, median |v|={roll_mv_s}")

                    # Build summary dict for sidecar (after print — uses pre-update rolling)
                    forced_probe_summary = {
                        "n": n,
                        "n_skipped_size": _probe_res["n_skipped_size"],
                        "sign_correct": sc,
                        "sign_correct_pct": sc_pct,
                        "median_abs_v": mv,
                        "delta_sign_correct_pct": delta_pct,
                        "delta_median_abs_v": delta_mv,
                        "rolling5_sign_correct_pct": roll_pct,
                        "rolling5_median_abs_v": roll_mv,
                    }
                    # Update rolling window AFTER reading prev
                    forced_probe_history.append(
                        {"sign_correct_pct": sc_pct, "median_abs_v": mv}
                    )
            # Warn on ANY structural policy issue
            pi_has_issues = any(pi_stats.get(k, 0) > 0 for k in [
                'pi_len_mismatch_frac', 'pi_all_zero_frac', 'pi_negative_frac',
                'pi_empty_legal_frac', 'pi_empty_visits_frac'])
            if pi_has_issues:
                print(f"    pi WARNING: mismatch={pi_stats['pi_len_mismatch_frac']:.3f}, "
                      f"all_zero={pi_stats['pi_all_zero_frac']:.3f}, "
                      f"negative={pi_stats['pi_negative_frac']:.3f}, "
                      f"empty_legal={pi_stats['pi_empty_legal_frac']:.3f}, "
                      f"empty_visits={pi_stats['pi_empty_visits_frac']:.3f}")

            # === Value head health monitoring ===
            v_pre_p99 = v_stats.get("v_pre_abs_p99", None)
            v_sat = v_stats.get("v_frac_sat", None)
            v_n_samples = v_stats.get("v_non_draw_n", 0)

            # Get health level
            v_eligible, v_level, v_trigger = get_value_health_level(v_pre_p99, v_sat, v_n_samples)

            # Print status (quiet when healthy or skipped - only print problems)
            if v_level == "caution":
                print(f"    ⚠️  Value head: p99={v_pre_p99:.2f}, sat={v_sat:.3f} (caution; trigger={v_trigger})")
            elif v_level == "warn":
                print(f"    🚨 Value head: p99={v_pre_p99:.2f}, sat={v_sat:.3f} (warning; trigger={v_trigger})")
            elif v_level == "crit":
                print(f"    💥 Value head: p99={v_pre_p99:.2f}, sat={v_sat:.3f} (critical; trigger={v_trigger})")

            # Record to history (normalize: level/trigger only meaningful when eligible)
            value_history.append({
                "eligible": v_eligible,
                "level": v_level if v_eligible else None,
                "trigger": v_trigger if v_eligible else "",
                "p99": v_pre_p99 if v_eligible else None,
                "sat": v_sat if v_eligible else None,
            })

            # Update streak (only if eligible)
            if v_eligible:
                if v_level in ("warn", "crit"):
                    value_warn_streak += 1
                    # Action hint on first warn/crit of streak
                    if value_warn_streak == 1:
                        print(f"    Suggested: lower --value-lr-scale, reduce value_grad_max_norm, or increase sims")
                    # Extra hint on critical
                    if v_level == "crit":
                        print(f"    Suggested: immediately lower value LR and/or clamp value grads")
                else:
                    value_warn_streak = 0

            # Compute rolling stats from eligible entries
            eligible_entries = [x for x in value_history if x.get("eligible")]
            eligible_count = len(eligible_entries)

            if eligible_count >= VALUE_MIN_ELIGIBLE:
                warn_only = sum(1 for x in eligible_entries if x.get("level") == "warn")
                crit_only = sum(1 for x in eligible_entries if x.get("level") == "crit")
                warn_crit_count = warn_only + crit_only
                warn_rate = warn_crit_count / eligible_count

                if warn_rate >= VALUE_WARN_FRACTION:
                    # Only print once per streak (noise control)
                    if not value_instability_active:
                        value_instability_active = True
                        print(f"    🚨 VALUE INSTABILITY: warn/crit in {warn_crit_count}/{eligible_count} eligible iters "
                              f"(warn={warn_only} crit={crit_only}, window={VALUE_WINDOW}, threshold={VALUE_WARN_FRACTION:.0%}) | "
                              f"consec_warn_streak={value_warn_streak}")
                else:
                    value_instability_active = False  # reset when condition clears
            # else: don't reset value_instability_active - avoid forgetting state during eligibility dips

            # Legacy: keep consecutive_saturation_iters for checkpoint compatibility
            if v_eligible and v_level in ("warn", "crit"):
                consecutive_saturation_iters += 1
            else:
                consecutive_saturation_iters = 0

            # --- Write per-iteration stats sidecar (atomic) ---
            _total_games = red_wins + black_wins + draws
            _total_decisive = red_wins + black_wins
            _red_natural = red_wins - resigned_by_black - adjudicated_red_wins
            _black_natural = black_wins - resigned_by_red - adjudicated_black_wins

            _bal_red = round(red_wins / _total_decisive * 100, 1) if _total_decisive > 0 else 0.0
            _bal_black = round(black_wins / _total_decisive * 100, 1) if _total_decisive > 0 else 0.0
            _bal_draw = round(draws / _total_games * 100, 1) if _total_games > 0 else 0.0

            _rg_checks = rg_checks_red + rg_checks_black
            _rg_vhits = rg_value_hits_red + rg_value_hits_black
            _rg_elig = rg_eligible_red + rg_eligible_black

            _adj_stats = {}
            if adj_abs_rv_samples:
                _rv = np.array(adj_abs_rv_samples)
                _t1 = np.array(adj_top1_samples) if adj_top1_samples else np.array([0.0])
                _adj_stats = {
                    "abs_root_value": {"p50": round(float(np.percentile(_rv, 50)), 3), "p90": round(float(np.percentile(_rv, 90)), 3)},
                    "top1_share": {"p50": round(float(np.percentile(_t1, 50)), 3), "p10": round(float(np.percentile(_t1, 10)), 3)},
                }

            _rg_top1 = {}
            if rg_top1_all:
                _arr = np.array(rg_top1_all)
                _rg_top1 = {
                    "p50": round(float(np.percentile(_arr, 50)), 2),
                    "p90": round(float(np.percentile(_arr, 90)), 2),
                    "p99": round(float(np.percentile(_arr, 99)), 2),
                }

            _sidecar = {
                "iteration": iteration,
                "games_per_iter": games_generated,
                "results": {"red_wins": red_wins, "black_wins": black_wins, "draws": draws},
                "draw_breakdown": {"timeout": timeout_draws, "board_full": board_full_draws, "state_cap": state_cap_draws, "unknown": unknown_draws},
                "termination": {"win": _red_natural + _black_natural, "resign": resign_games, "adjudicated": adjudicated_games, "timeout": timeout_draws},
                "termination_by_winner": {
                    "red": {"win": _red_natural, "resign": resigned_by_black, "adjudicated": adjudicated_red_wins},
                    "black": {"win": _black_natural, "resign": resigned_by_red, "adjudicated": adjudicated_black_wins},
                    "draw": {"timeout": timeout_draws},
                },
                "avg_plies": round(avg_plies, 1),
                "balance": {"red_pct": _bal_red, "black_pct": _bal_black, "draw_pct": _bal_draw, "decisive_games": _total_decisive, "window": f"{len(recent_red_dominant)}/{BALANCE_WINDOW}"},
                "targets": {"z_pos": z_stats.get("z_count_pos", 0), "z_zero": z_stats.get("z_count_zero", 0), "z_neg": z_stats.get("z_count_neg", 0)},
                "adjudication": {
                    "attempts": adj_attempts, "adjudicated": adjudicated_games, "red_wins": adjudicated_red_wins, "black_wins": adjudicated_black_wins, "remaining_timeouts": timeout_draws,
                    "blocks": {"ply": adj_blocked_ply, "threshold": adj_blocked_threshold, "visits": adj_blocked_visits, "top1": adj_blocked_top1},
                    "stats": _adj_stats,
                },
                "resign": {"total": resign_games, "by_red": resigned_by_red, "by_black": resigned_by_black},
                "resign_gate": {
                    "checks": _rg_checks, "red_checks": rg_checks_red, "black_checks": rg_checks_black,
                    "value_hits": _rg_vhits, "red_value_hits": rg_value_hits_red, "black_value_hits": rg_value_hits_black,
                    "blocked_by_top1": _rg_vhits - _rg_elig, "red_blocked_by_top1": rg_value_hits_red - rg_eligible_red, "black_blocked_by_top1": rg_value_hits_black - rg_eligible_black,
                    "eligible_hits": _rg_elig, "red_eligible_hits": rg_eligible_red, "black_eligible_hits": rg_eligible_black,
                    "top1_share_on_value_hits": _rg_top1, "min_top1_share": resign_min_top1_share,
                },
                "compute": {"buffer_size": len(buffer), "backups": total_backups, "leaf_evals": total_nn_calls, "nn_batches": total_nn_batches},
                # Phase 2 connectivity-bucketed sanity (None/empty for 24-channel checkpoints)
                "sanity_by_connectivity": v_stats.get("sanity_by_connectivity"),
                # Phase 2 inline forced-probe summary (None when probes file missing/disabled)
                "forced_probe_summary": forced_probe_summary,
                "replay_cap": {
                    "enabled": bool(max_positions_per_game and max_positions_per_game > 0),
                    "max_positions_per_game": int(max_positions_per_game) if max_positions_per_game else 0,
                    "endgame_keep_positions": int(endgame_keep_positions),
                    "games_total": games_generated,
                    "games_capped": games_capped_iter,
                    "capped_rate": (games_capped_iter / games_generated) if games_generated else 0.0,
                    "total_positions_original": total_positions_original_iter,
                    "total_positions_kept": total_positions_kept_iter,
                    "mean_positions_original": (total_positions_original_iter / games_generated) if games_generated else 0.0,
                    "mean_positions_kept": (total_positions_kept_iter / games_generated) if games_generated else 0.0,
                    "kept_fraction": (total_positions_kept_iter / total_positions_original_iter) if total_positions_original_iter else 1.0,
                    "positions_by_termination": dict(positions_by_termination_iter),
                    "positions_in_short_games": positions_in_short_games_iter,
                    "positions_in_long_games": positions_in_long_games_iter,
                    "by_length_bucket": {
                        "edges_ply": list(_PLY_BUCKET_EDGES_ITER),
                        "games": list(ply_bucket_games_iter),
                        "positions_original": list(ply_bucket_positions_original_iter),
                        "positions_kept": list(ply_bucket_positions_kept_iter),
                    },
                },
            }

            # Aggregate opening diagnostics into sidecar
            if all_opening_diagnostics:
                _diag_end, _diag_floor_used = compute_diagnostic_end_ply(
                    iter_mcts_config.root_edge_band_penalty_ply,
                    iter_mcts_config.root_near_corner_penalty_ply,
                )
                _sidecar["opening_penalty_diagnostics"] = aggregate_opening_diagnostics(
                    all_game_diagnostics=all_opening_diagnostics,
                    diagnostic_end_ply=_diag_end,
                    extra_plies=2,
                    floor_min_ply=4,
                    used_floor=_diag_floor_used,
                    games_total_iter=games_generated,
                )
                # Phase 1: root-child diagnostics rollup (ply 0 .. CHILD_DETAIL_PLIES-1)
                _sidecar["root_child_diagnostics"] = aggregate_root_child_details(
                    all_game_diagnostics=all_opening_diagnostics,
                    child_detail_max_ply=CHILD_DETAIL_PLIES,
                )
                # Phase 2 required #3: compact early-override summary block
                # Combines mass (from opening diagnostics) + best-by-*
                # (from root-child diagnostics) for the critical ply 0-1 window.
                _sidecar["early_override_summary"] = build_early_override_summary(
                    opd_aggregate=_sidecar["opening_penalty_diagnostics"],
                    rcd_aggregate=_sidecar["root_child_diagnostics"],
                    early_plies=CHILD_DETAIL_PLIES,
                )

            games_dir.mkdir(parents=True, exist_ok=True)
            _tmp = games_dir / f"iter_{iteration:04d}_stats.json.tmp"
            _final = games_dir / f"iter_{iteration:04d}_stats.json"
            try:
                with open(_tmp, "w", encoding="utf-8") as _sf:
                    json.dump(_sidecar, _sf, indent=2)
                os.replace(_tmp, _final)
            except Exception as e:
                print(f"  WARNING: failed to write stats sidecar: {e}")
                try:
                    if _tmp.exists():
                        _tmp.unlink()
                except Exception:
                    pass

        else:
            avg_plies = 0
            avg_batch = 0
            avg_waiters = 0
            p95_plies = 0.0
            max_plies_observed = 0
            avg_game_seconds = 0.0
            p95_game_seconds = 0.0
            z_stats = {}
            pi_stats = {}
            v_stats = {}
        selfplay_end = time.perf_counter()

        # Print games saved summary
        if game_saver is not None:
            print(f"  Games saved: {game_saver.games_saved_this_iter}/{games_generated}")

        # Get curriculum metrics AFTER all record_game() calls
        curriculum_metrics = curriculum.get_metrics()

        # 2. Training (train mode - BN updates enabled) - skip if interrupted
        train_start = time.perf_counter()
        network.train()
        avg_total_loss = None
        avg_policy_loss = None
        avg_value_loss = None
        avg_l2_loss = None
        steps_done = 0  # Track actual steps completed for full_iteration detection
        train_completed = False  # Set True after training finishes or if no training needed

        # Value weight warmup schedule: 0.05 -> 0.10 -> 0.25
        # iteration is 0-based: 0, 1, 2, 3, ...
        if iteration < 2:
            curr_value_weight = 0.05
        elif iteration < 4:
            curr_value_weight = 0.10
        else:
            curr_value_weight = value_weight  # Full target value

        if interrupted:
            print(f"\nSkipping training (interrupted)")
            train_completed = False  # explicit
        else:
            # Check positions available for CURRENT active_size (curriculum filtering)
            positions_available = buffer.count_by_active_size(active_size)
            if positions_available >= batch_size and scaled_train_steps > 0:
                print(f"\nTraining: {scaled_train_steps} steps... (value_weight={curr_value_weight})")

                # Four accumulators for loss split
                sum_total = 0.0
                sum_policy = 0.0
                sum_value = 0.0
                sum_l2 = 0.0

                train_rng = random.Random(master_rng.randint(0, 2**31))

                try:
                    for step in range(scaled_train_steps):
                        batch = buffer.sample(batch_size, rng=train_rng, active_size=active_size)
                        # Pass active_size to training (use current curriculum stage)
                        loss_total, loss_policy, loss_value, loss_l2 = train_step(
                            network=network,
                            main_module=main_module,
                            opt_main=opt_main,
                            opt_value=opt_value,
                            batch=batch,
                            l2_weight=l2_weight,
                            value_weight=curr_value_weight,
                            active_size=active_size,
                            value_grad_max_norm=value_grad_max_norm,
                            progress_weighted=progress_weighted,
                            progress_weight_floor=progress_weight_floor,
                        )

                        # Sums first, then steps_done (ensures denominator matches included samples)
                        sum_total += loss_total
                        sum_policy += loss_policy
                        sum_value += loss_value
                        sum_l2 += loss_l2
                        steps_done += 1

                        if (step + 1) % 20 == 0:
                            avg = sum_total / (step + 1)
                            print(
                                f"  Step {step+1}/{scaled_train_steps}, "
                                f"Loss: {avg:.4f}"
                            )

                    # Only mark completed if loop finishes normally
                    train_completed = True

                    # Compute averages using steps_done (guards against partial)
                    if steps_done > 0:
                        avg_total_loss = sum_total / steps_done
                        avg_policy_loss = sum_policy / steps_done
                        avg_value_loss = sum_value / steps_done
                        avg_l2_loss = sum_l2 / steps_done

                        print(f"  Average loss: {avg_total_loss:.4f} "
                              f"(policy={avg_policy_loss:.4f}, value={avg_value_loss:.4f}, l2={avg_l2_loss:.4f})")

                    # Clear cache once after training (not during)
                    gc.collect()
                    mx.clear_cache()

                except KeyboardInterrupt:
                    print(f"\nInterrupted during training at step {steps_done}/{scaled_train_steps}!")
                    interrupted = True
                    train_completed = False  # explicit

                except Exception as e:
                    print(f"\nTraining error at step {steps_done}/{scaled_train_steps}: {type(e).__name__}: {e}")
                    interrupted = True
                    train_completed = False  # explicit
                    # NOTE: do NOT re-raise. Let execution continue to checkpoint save.

            else:
                if scaled_train_steps == 0:
                    print(f"\nSkipping training (train_steps=0)")
                    train_completed = True  # No training needed = completed
                else:
                    print(f"\nSkipping training (buffer too small): positions={positions_available} batch={batch_size}")
                    train_completed = False  # explicit: not a full iteration
        train_end = time.perf_counter()

        # D) Determine "full iteration"
        selfplay_completed = (games_generated == games_per_iteration)
        full_iteration = selfplay_completed and train_completed and not interrupted

        # E) Compute iteration metrics (used by freeze/promo/demo/CSV)
        iter_timeout_rate = timeout_draws / games_generated if games_generated > 0 else 0.0
        iter_plies_ratio = avg_plies / scaled_max_moves if scaled_max_moves > 0 else 1.0

        # F) Freeze update (Phase 6) - only on full iterations
        if full_iteration:
            timeout_rate_for_freeze = iter_timeout_rate
            if timeout_rate_for_freeze >= 0.25:
                consecutive_high_timeout_iters += 1
                consecutive_good_timeout_iters = 0
                if consecutive_high_timeout_iters >= 2:
                    curriculum_frozen = True
                    # Only decay if not already at effective floor
                    # NOTE: sims_after_freeze computed at iteration start, before floor clamp
                    if sims_after_freeze > ABS_SIMS_FLOOR:
                        sims_reduction_factor = max(0.5, sims_reduction_factor * 0.8)
                        print(f"  FREEZE: timeout={timeout_rate_for_freeze:.1%}, factor={sims_reduction_factor:.2f}")
                    else:
                        # At floor - stop decaying, keep factor as-is
                        print(f"  FREEZE: at sims floor (sims=100), factor={sims_reduction_factor:.2f}")
            else:
                consecutive_high_timeout_iters = 0
                consecutive_good_timeout_iters += 1
                if curriculum_frozen and consecutive_good_timeout_iters >= 3:
                    # Gradual ramp-up instead of instant reset
                    new_factor = min(1.0, sims_reduction_factor * 1.25)  # +25% per good iter
                    if new_factor >= 1.0:
                        curriculum_frozen = False
                        sims_reduction_factor = 1.0
                        print(f"  UNFREEZE: fully recovered, factor=1.0")
                    else:
                        sims_reduction_factor = new_factor
                        print(f"  RECOVER: ramping up, factor={sims_reduction_factor:.2f}")

        # Compute sims_next (always, even on partial iterations)
        sims_next = int(base_sims_effective * sims_reduction_factor)
        sims_next = max(ABS_SIMS_FLOOR, sims_next)

        # G) Demotion gating (Phase 5) - only on full iterations
        demotion_triggered = False
        if full_iteration:
            if iter_timeout_rate >= 0.40 or iter_plies_ratio >= 0.95:
                consecutive_demotable_iters += 1
            else:
                consecutive_demotable_iters = 0

            if consecutive_demotable_iters >= 2 and curriculum.idx > 0:
                old_size = curriculum.active_size
                demotion_triggered = curriculum.demote()
                if demotion_triggered:
                    new_size = curriculum.active_size
                    new_max_moves = get_scaled_max_moves(new_size)
                    new_base_sims = mcts_simulations if mcts_simulations is not None else SIMS_TABLE.get(new_size, 400)
                    print(f"\n*** CURRICULUM DEMOTED: {old_size} -> {new_size} ***")
                    print(f"    Reason: timeout={iter_timeout_rate:.1%}, plies_ratio={iter_plies_ratio:.2f}")
                    print(f"    Next iter: max_moves={new_max_moves}, base_sims={new_base_sims}")
                    # Safety belt: reset counters immediately on size change
                    consecutive_promotable_iters = 0
                    consecutive_demotable_iters = 0

        # H) Promotion gating (Phase 4) - only on full iterations
        promoted = False
        promotion_allowed = False
        promotion_reason = "partial_iter"
        if full_iteration:
            # Promotion gate: need 3 consecutive good iterations
            if iter_timeout_rate <= 0.10 and iter_plies_ratio <= 0.85:
                consecutive_promotable_iters += 1
            else:
                consecutive_promotable_iters = 0

            promotion_allowed = (consecutive_promotable_iters >= 3) and not curriculum_frozen

            # Determine reason
            if curriculum_frozen:
                promotion_reason = "frozen"
            elif iter_timeout_rate > 0.10:
                promotion_reason = f"timeout={iter_timeout_rate:.1%}"
            elif iter_plies_ratio > 0.85:
                promotion_reason = f"plies_ratio={iter_plies_ratio:.2f}"
            elif consecutive_promotable_iters < 3:
                promotion_reason = f"need_{3 - consecutive_promotable_iters}_more_good_iters"
            else:
                promotion_reason = "allowed"

            if promotion_allowed:
                promoted = curriculum.maybe_promote()
                if promoted:
                    new_size = curriculum.active_size
                    new_max_moves = get_scaled_max_moves(new_size)
                    new_base_sims = mcts_simulations if mcts_simulations is not None else SIMS_TABLE.get(new_size, 400)
                    print(f"\n*** CURRICULUM PROMOTED: active_size={new_size} ***")
                    print(f"    Metrics: timeout={iter_timeout_rate:.1%}, plies_ratio={iter_plies_ratio:.2f}")
                    print(f"    Next iter: max_moves={new_max_moves}, base_sims={new_base_sims}")
                    # Safety belt: reset counters immediately on size change
                    consecutive_promotable_iters = 0
                    consecutive_demotable_iters = 0
            else:
                # Only print when blocked AND reason is meaningful (avoid log spam)
                # Skip the "need_X_more_good_iters" message for quiet logs
                if promotion_reason != "allowed" and not promotion_reason.startswith("need_"):
                    print(f"  Promotion blocked: {promotion_reason}")

        # 3. Checkpoint
        iter_end = time.perf_counter()

        # Compute timing metrics
        self_play_wall_s = selfplay_end - selfplay_start
        train_wall_s = train_end - train_start if (train_steps_per_iteration or 0) > 0 else 0.0
        iter_wall_s = iter_end - iter_start
        positions_per_sec = positions_added / self_play_wall_s if self_play_wall_s > 0 else 0.0

        # Compute derived metrics for regression detection
        total_flushes = total_flush_full + total_flush_stall + total_flush_tail
        stall_flush_rate = total_flush_stall / total_flushes if total_flushes > 0 else 0.0
        backups_per_game = total_backups / games_generated if games_generated > 0 else 0.0
        leaf_evals_per_game = total_nn_calls / games_generated if games_generated > 0 else 0.0

        # I) State summary for eyeballing regressions
        train_str = f"{steps_done}/{scaled_train_steps}" if scaled_train_steps > 0 else "n/a"
        print(f"  State: size={active_size} sims={sims_used} factor={sims_reduction_factor:.2f} "
              f"frozen={curriculum_frozen} promo_streak={consecutive_promotable_iters} "
              f"demote_streak={consecutive_demotable_iters} full={full_iteration} "
              f"train={train_str}")

        # Partial checkpoint naming (don't overwrite canonical checkpoints on interrupt)
        if interrupted:
            ckpt_base = f"model_iter_{iteration+1:04d}_partial"
        else:
            ckpt_base = f"model_iter_{iteration+1:04d}"

        ckpt_path = os.path.join(checkpoint_dir, f"{ckpt_base}.safetensors")
        network.save_weights(ckpt_path)

        # Note: curriculum_metrics already computed after record_game calls

        # Build iteration_metrics dict for CSV and JSON
        iteration_metrics = {
            # Identity
            "schema_version": METRICS_SCHEMA_VERSION,
            "run_id": run_id,
            "row_id": f"{run_id}:{iteration + 1}",
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration + 1,
            "active_size": active_size,
            "max_moves": scaled_max_moves,

            # Config snapshot
            "games_per_iter": games_per_iteration,
            "simulations": mcts_simulations,
            "train_steps_per_iteration": train_steps_per_iteration,
            "batch_size": batch_size,
            "buffer_size_limit": buffer.max_size,
            "mcts_eval_batch_size": mcts_eval_batch_size,
            "mcts_pending_virtual_visits": mcts_pending_virtual_visits,
            "mcts_stall_flush_sims": mcts_stall_flush_sims,
            "network_hidden": hidden,
            "network_blocks": n_blocks,

            # Sims transparency
            "requested_sims_cli": mcts_simulations,
            "base_sims_table": base_sims_from_table,
            "effective_sims_used": sims_used,
            "sims_clamped_to_floor": sims_clamped_to_floor,
            "effective_reason": effective_reason,

            # Self-play outputs
            "games_generated": games_generated,
            "positions_added": positions_added,
            "buffer_size_end": len(buffer),
            "avg_plies": avg_plies,
            "p95_plies": p95_plies,
            "max_plies_observed": max_plies_observed,
            "avg_game_seconds": avg_game_seconds,
            "p95_game_seconds": p95_game_seconds,

            # Results + draw breakdown
            "red_wins": red_wins,
            "black_wins": black_wins,
            "draws": draws,
            "timeout_draws": timeout_draws,
            "board_full_draws": board_full_draws,
            "state_cap_draws": state_cap_draws,
            "unknown_draws": unknown_draws,

            # MCTS rollup
            "total_backups": total_backups,
            "leaf_evals": total_nn_calls,
            "nn_batches": total_nn_batches,
            "avg_batch": avg_batch,
            "avg_waiters": avg_waiters,
            "max_waiters": max_waiters,
            "flush_full": total_flush_full,
            "flush_stall": total_flush_stall,
            "flush_tail": total_flush_tail,

            # Training
            "avg_total_loss": avg_total_loss,
            "avg_policy_loss": avg_policy_loss,
            "avg_value_loss": avg_value_loss,
            "avg_l2_loss": avg_l2_loss,

            # Curriculum
            "draw_rate_true": curriculum_metrics["draw_rate_true"],
            "timeout_rate": curriculum_metrics["timeout_rate"],
            "draw_rate_timeout": curriculum_metrics["draw_rate_timeout"],  # Compat alias
            "promoted_this_iter": promoted,
            "curriculum_frozen": curriculum_frozen,
            "sims_used": sims_used,
            "sims_next": sims_next,
            "sims_reduction_factor": sims_reduction_factor,

            # Promotion/demotion
            "promotion_allowed": promotion_allowed,
            "promotion_reason": promotion_reason,
            "demotion_triggered": demotion_triggered,
            "consecutive_promotable_iters": consecutive_promotable_iters,
            "consecutive_demotable_iters": consecutive_demotable_iters,
            "full_iteration": full_iteration,
            "iter_timeout_rate": iter_timeout_rate,
            "iter_plies_ratio": iter_plies_ratio,

            # Timing
            "self_play_wall_s": self_play_wall_s,
            "train_wall_s": train_wall_s,
            "iter_wall_s": iter_wall_s,
            "positions_per_sec": positions_per_sec,

            # Derived regression detectors
            "stall_flush_rate": stall_flush_rate,
            "backups_per_game": backups_per_game,
            "leaf_evals_per_game": leaf_evals_per_game,

            # Sanity stats
            **z_stats,
            **pi_stats,
            # v_stats spread carefully: nested 'sanity_by_connectivity' is flattened
            # into sbc_* keys below (CSV can't hold nested dicts). The full nested
            # version lives in the iteration JSON via _sidecar.
            **{k: v for k, v in v_stats.items() if k != "sanity_by_connectivity"},
            # Phase 2: connectivity-bucketed sanity (flattened scalars)
            "sbc_winning_n": (v_stats.get("sanity_by_connectivity") or {}).get("winning_structure", {}).get("n"),
            "sbc_winning_sign_agree": (v_stats.get("sanity_by_connectivity") or {}).get("winning_structure", {}).get("sign_agree"),
            "sbc_winning_median_abs_v": (v_stats.get("sanity_by_connectivity") or {}).get("winning_structure", {}).get("median_abs_v"),
            "sbc_no_winning_n": (v_stats.get("sanity_by_connectivity") or {}).get("no_winning_structure", {}).get("n"),
            "sbc_no_winning_sign_agree": (v_stats.get("sanity_by_connectivity") or {}).get("no_winning_structure", {}).get("sign_agree"),
            "sbc_no_winning_median_abs_v": (v_stats.get("sanity_by_connectivity") or {}).get("no_winning_structure", {}).get("median_abs_v"),
            # Phase 2: inline forced-probe (flattened scalars; None when probe file missing/disabled)
            "fps_n": (forced_probe_summary or {}).get("n"),
            "fps_sign_correct": (forced_probe_summary or {}).get("sign_correct"),
            "fps_sign_correct_pct": (forced_probe_summary or {}).get("sign_correct_pct"),
            "fps_median_abs_v": (forced_probe_summary or {}).get("median_abs_v"),
            "fps_delta_sign_correct_pct": (forced_probe_summary or {}).get("delta_sign_correct_pct"),
            "fps_delta_median_abs_v": (forced_probe_summary or {}).get("delta_median_abs_v"),
            "fps_rolling5_sign_correct_pct": (forced_probe_summary or {}).get("rolling5_sign_correct_pct"),
            "fps_rolling5_median_abs_v": (forced_probe_summary or {}).get("rolling5_median_abs_v"),

            # Phase 4: per-game replay contribution cap
            "replay_cap_enabled": int(bool(max_positions_per_game and max_positions_per_game > 0)),
            "replay_cap_max": int(max_positions_per_game) if max_positions_per_game else 0,
            "replay_cap_endgame_keep": int(endgame_keep_positions),
            "replay_cap_games_capped": int(games_capped_iter),
            "replay_cap_capped_rate": round(
                (games_capped_iter / games_generated) if games_generated else 0.0, 4
            ),
            "replay_cap_total_orig": int(total_positions_original_iter),
            "replay_cap_total_kept": int(total_positions_kept_iter),
            "replay_cap_mean_orig": round(
                (total_positions_original_iter / games_generated) if games_generated else 0.0, 2
            ),
            "replay_cap_mean_kept": round(
                (total_positions_kept_iter / games_generated) if games_generated else 0.0, 2
            ),
            "replay_cap_kept_fraction": round(
                (total_positions_kept_iter / total_positions_original_iter)
                if total_positions_original_iter else 1.0, 4
            ),
        }

        # Write metrics to CSV (append-only)
        metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
        append_metrics_csv(metrics_path, iteration_metrics, CSV_FIELDNAMES)

        # Build expanded state dict for JSON checkpoint
        state = {
            **iteration_metrics,
            # Curriculum state for resume (existing)
            "curriculum": curriculum.to_dict(),
            # Freeze state for resume (must persist across runs)
            "freeze_state": {
                "consecutive_high_timeout_iters": consecutive_high_timeout_iters,
                "consecutive_good_timeout_iters": consecutive_good_timeout_iters,
                "curriculum_frozen": curriculum_frozen,
                "sims_reduction_factor": sims_reduction_factor,
                "consecutive_saturation_iters": consecutive_saturation_iters,
            },
            # Curriculum state for resume (promotion/demotion tracking)
            "curriculum_state": {
                "consecutive_promotable_iters": consecutive_promotable_iters,
                "consecutive_demotable_iters": consecutive_demotable_iters,
            },
            # Self-play progress snapshots (nested, JSON only)
            "selfplay_progress": selfplay_progress,
        }

        state_path = ckpt_path.replace(".safetensors", ".json")
        try:
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"WARNING: failed to write checkpoint JSON: {e}")

        print(f"\nCheckpoint saved: {ckpt_path}")
        print(f"  Curriculum: active_size={curriculum.active_size}, "
              f"draw_rate_true={curriculum_metrics['draw_rate_true']:.1%}, "
              f"timeout_rate_rolling={curriculum_metrics['timeout_rate']:.1%}")
        if not full_iteration:
            print(f"  Iter metrics (partial): iter_timeout_rate={iter_timeout_rate:.1%} "
                  f"({timeout_draws}/{games_generated} games)")
        print(f"  Status: sims_used={sims_used}, sims_next={sims_next}, "
              f"factor={sims_reduction_factor:.2f}, frozen={curriculum_frozen}")
        print(f"  Timing: iter={iter_wall_s:.1f}s, selfplay={self_play_wall_s:.1f}s, train={train_wall_s:.1f}s")

        # Progress callback
        if progress_callback:
            progress_callback(iteration + 1, state)

        # Exit early if interrupted
        if interrupted:
            print(f"\n{'='*60}")
            print("Training interrupted. Resume with --resume flag.")
            print(f"{'='*60}")
            _stop_caffeinate()
            return network

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    _stop_caffeinate()
    return network
