import numpy as np
import mlx.core as mx
import mlx.optimizers as optim

from scripts.GPU.alphazero.trainer import (
    alphazero_loss_batch, train_step, MainModule, flatten_params,
)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord


def _main_pos():
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="red", legal_moves=[(0, 0), (1, 1), (2, 2)],
        visit_counts=[10, 5, 3], outcome=1.0, active_size=24,
        ply=0, game_n_moves=10,
    )


def _calib_pos(target=-0.5):
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="black", legal_moves=[(0, 0), (1, 1)],
        visit_counts=[0, 0], outcome=target, active_size=24,
        ply=20, game_n_moves=None,
    )


def test_disabled_returns_seven_tuple():
    net = create_network(hidden=64, n_blocks=2)
    out = alphazero_loss_batch(net, [_main_pos() for _ in range(3)])
    assert len(out) == 7


def test_zero_weight_is_inert_seven_tuple():
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    out = alphazero_loss_batch(
        net, pos,
        calibration_positions=[_calib_pos()],
        calibration_loss_weight=0.0,
    )
    # Zero weight means calibration is inactive: no extended tuple, no leaked term.
    assert len(out) == 7


def test_enabled_returns_ten_tuple_and_adds_weighted_mse():
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    calib = [_calib_pos(-0.5), _calib_pos(-0.5)]
    value_weight = 0.5
    conversion_weight = 0.0
    calibration_weight = 0.02
    out = alphazero_loss_batch(
        net,
        pos,
        value_weight=value_weight,
        conversion_loss_weight=conversion_weight,
        calibration_positions=calib,
        calibration_loss_weight=calibration_weight,
    )
    assert len(out) == 10
    (
        total,
        policy_loss,
        value_loss,
        l2_loss,
        aux_loss,
        _aux_coverage,
        _aux_n_eligible,
        calib_loss,
        _calib_value_mean,
        calib_n,
    ) = out
    assert calib_n == 2
    expected_total = (
        policy_loss
        + value_weight * value_loss
        + l2_loss
        + conversion_weight * aux_loss
        + calibration_weight * calib_loss
    )
    np.testing.assert_allclose(
        float(total.item()),
        float(expected_total.item()),
        atol=1e-6,
    )


def _vh_gnorm(grads):
    return sum(float(mx.sum(mx.abs(p)).item())
               for _, p in flatten_params(grads["value_head"]))


def test_calibration_gradient_reaches_value_head():
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    calib = [_calib_pos(-0.5)]

    def off(m):
        return alphazero_loss_batch(m, pos, value_weight=0.0, l2_weight=0.0)

    def on(m):
        return alphazero_loss_batch(m, pos, value_weight=0.0, l2_weight=0.0,
                                    calibration_positions=calib,
                                    calibration_loss_weight=0.02)

    _, g_off = nn_value_and_grad(net, off)
    _, g_on = nn_value_and_grad(net, on)
    # Essential claim: calibration drives a value-head gradient the disabled
    # path does not. Assert the margin (robust to MLX init/path noise) rather
    # than relying solely on exact-zero for the disabled case.
    assert _vh_gnorm(g_on) > 1e-6
    assert _vh_gnorm(g_on) > _vh_gnorm(g_off)


def nn_value_and_grad(net, fn):
    import mlx.nn as nn
    return nn.value_and_grad(net, fn)(net)


def test_weighted_calibration_loss_matches_formula():
    """calib_loss == Σ(wᵢ·mseᵢ)/Σ(wᵢ). Recover per-sample mse with one-hot weights,
    then verify the [1,3] weighting (network is deterministic across forward-only calls)."""
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    calib = [_calib_pos(-0.5), _calib_pos(0.25)]  # distinct targets → distinct mse

    def calib_loss(weights):
        out = alphazero_loss_batch(
            net, pos, calibration_positions=calib,
            calibration_weights=np.array(weights, dtype=np.float32),
            calibration_loss_weight=0.02)
        return float(out[7].item())  # index 7 = calib_loss

    mse0 = calib_loss([1.0, 0.0])
    mse1 = calib_loss([0.0, 1.0])
    expected = (1.0 * mse0 + 3.0 * mse1) / (1.0 + 3.0)
    np.testing.assert_allclose(calib_loss([1.0, 3.0]), expected, rtol=1e-5)


def test_equal_weights_equal_unweighted_mean():
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    calib = [_calib_pos(-0.5), _calib_pos(0.25)]

    def calib_loss(weights):
        out = alphazero_loss_batch(
            net, pos, calibration_positions=calib,
            calibration_weights=(None if weights is None
                                 else np.array(weights, dtype=np.float32)),
            calibration_loss_weight=0.02)
        return float(out[7].item())

    np.testing.assert_allclose(calib_loss([2.0, 2.0]), calib_loss(None), rtol=1e-6)


def test_all_zero_calibration_weights_are_finite_zero_loss():
    """Σw==0 → mx.maximum(Σw, 1e-8) keeps it finite (0.0), never NaN.
    Explicit zero-weight rows are now allowed, so pin this behavior."""
    net = create_network(hidden=64, n_blocks=2)
    pos = [_main_pos() for _ in range(3)]
    calib = [_calib_pos(-0.5), _calib_pos(0.25)]
    out = alphazero_loss_batch(
        net, pos, calibration_positions=calib,
        calibration_weights=np.array([0.0, 0.0], dtype=np.float32),
        calibration_loss_weight=0.02)
    calib_loss = float(out[7].item())
    assert np.isfinite(calib_loss)
    assert calib_loss == 0.0


def test_train_step_arity_disabled_and_enabled():
    net = create_network(hidden=64, n_blocks=2)
    mm = MainModule(net.encoder, net.policy_head)
    opt_main = optim.Adam(learning_rate=1e-3)
    opt_value = optim.Adam(learning_rate=1e-4)
    pos = [_main_pos() for _ in range(3)]

    off = train_step(network=net, main_module=mm, opt_main=opt_main,
                     opt_value=opt_value, batch=pos)
    assert len(off) == 7

    on = train_step(network=net, main_module=mm, opt_main=opt_main,
                    opt_value=opt_value, batch=pos,
                    calibration_positions=[_calib_pos()],
                    calibration_loss_weight=0.02)
    assert len(on) == 10
    assert all(np.isfinite(x) for x in on[:9])
    assert on[9] == 1


def test_train_step_accepts_weights_returns_ten_tuple():
    net = create_network(hidden=64, n_blocks=2)
    mm = MainModule(net.encoder, net.policy_head)
    opt_main = optim.Adam(learning_rate=1e-3)
    opt_value = optim.Adam(learning_rate=1e-4)
    pos = [_main_pos() for _ in range(3)]
    out = train_step(network=net, main_module=mm, opt_main=opt_main, opt_value=opt_value,
                     batch=pos, calibration_positions=[_calib_pos(-0.5), _calib_pos(0.25)],
                     calibration_weights=np.array([1.0, 3.0], dtype=np.float32),
                     calibration_loss_weight=0.02)
    assert len(out) == 10
    assert all(np.isfinite(x) for x in out[:9])
    assert out[9] == 2
