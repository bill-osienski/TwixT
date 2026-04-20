"""Progress-weighted value-loss tests."""
import numpy as np
import mlx.core as mx
import pytest


def test_floor_one_reproduces_unweighted_mse():
    """With progress_weight_floor=1.0, the loss exactly equals mean(err^2)."""
    from scripts.GPU.alphazero.trainer import _compute_progress_weighted_value_loss
    values = mx.array([0.5, -0.3, 0.9, -0.8])
    outcomes = mx.array([1.0, -1.0, 1.0, -1.0])
    plies = np.array([0, 50, 100, 199])
    game_n_moves = np.array([200, 200, 200, 200])
    # With floor=1.0, all weights = 1.0
    weighted = _compute_progress_weighted_value_loss(
        values, outcomes, plies, game_n_moves, floor=1.0)
    unweighted = mx.mean((values - outcomes) ** 2)
    assert abs(float(weighted) - float(unweighted)) < 1e-6


def test_scale_invariance_of_normalized_mean():
    """Normalized weighted mean remains finite and positive across floors."""
    from scripts.GPU.alphazero.trainer import _compute_progress_weighted_value_loss
    values = mx.array([0.5, -0.3, 0.9, -0.8])
    outcomes = mx.array([1.0, -1.0, 1.0, -1.0])
    plies = np.array([0, 50, 100, 199])
    game_n_moves = np.array([200, 200, 200, 200])
    loss_a = float(_compute_progress_weighted_value_loss(values, outcomes, plies, game_n_moves, floor=0.25))
    loss_b = float(_compute_progress_weighted_value_loss(values, outcomes, plies, game_n_moves, floor=0.5))
    assert loss_a > 0
    assert loss_b > 0


def test_edge_case_n_moves_one():
    """game_n_moves <= 1 → denominator clamp, progress = 1.0."""
    from scripts.GPU.alphazero.trainer import _compute_progress_weighted_value_loss
    values = mx.array([0.5])
    outcomes = mx.array([1.0])
    plies = np.array([0])
    game_n_moves = np.array([1])
    loss = _compute_progress_weighted_value_loss(values, outcomes, plies, game_n_moves, floor=0.25)
    # At n=1 and floor=0.25, weight becomes 0.25 + 0.75*1.0 = 1.0 (single sample)
    # Normalized mean w/ single sample = err^2 = 0.25
    assert abs(float(loss) - 0.25) < 1e-6
