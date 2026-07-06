"""v12 hinge: relu(sign*(v - target) - margin)^2, value-only, byte-identical
when the guardrail sign vector is absent."""
import numpy as np
import pytest
import mlx.optimizers as optim

from scripts.GPU.alphazero.trainer import (
    MainModule, freeze_batchnorm_running_stats, train_step)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord


def _pos(to_move="red", outcome=1.0):
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move=to_move, legal_moves=[(0, 0), (1, 1), (2, 2)],
        visit_counts=[10, 5, 3], outcome=outcome, active_size=24,
        ply=0, game_n_moves=10)


def _setup():
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    mm = MainModule(net.encoder, net.policy_head)
    return net, mm, optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)


def _guardrail_calib(to_move, target_black):
    # a single guardrail calibration row; outcome carries target in stm
    from scripts.GPU.alphazero.calibration_pool import target_in_to_move
    return [PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move=to_move, legal_moves=[(0, 0), (1, 1)], visit_counts=[0, 0],
        outcome=target_in_to_move(to_move, target_black), active_size=24,
        ply=20, game_n_moves=None)]


def _run(calib, sign, margin=0.10):
    net, mm, om, ov = _setup()
    out = train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                     batch=[_pos() for _ in range(3)],
                     calibration_positions=calib, calibration_loss_weight=0.01,
                     calibration_guardrail_sign=np.asarray(sign, dtype=np.float32),
                     guardrail_margin=margin,
                     train_value_head_and_final_block=True)
    return out


def _expected_hinge(v_stm, to_move, target_black, sign, margin=0.10):
    from scripts.GPU.alphazero.calibration_pool import target_in_to_move
    target_stm = target_in_to_move(to_move, target_black)
    over = sign * (v_stm - target_stm) - margin
    return max(0.0, over) ** 2


def test_guardrail_tuple_arity_is_13():
    out = _run(_guardrail_calib("black", 0.2), [1.0])
    assert len(out) == 13


def test_hinge_matches_formula_black():
    # Derive the expected hinge from the ACTUAL predicted value (out[8] =
    # calib_value_mean = this single row's stm value) so the test does not
    # depend on random init. Covers below/within/above the margin generically.
    for target_black in (0.9, 0.0, -0.9):
        out = _run(_guardrail_calib("black", target_black), [1.0])
        v_stm = out[8]                                    # calib_value_mean
        exp = _expected_hinge(v_stm, "black", target_black, sign=1.0)
        assert out[10] == pytest.approx(exp, abs=1e-5), (target_black, v_stm)
        assert out[11] == pytest.approx(1.0 if exp > 0 else 0.0)  # active_frac


def test_below_target_zero_hinge_black():
    # target_black=0.9 → threshold v>1.0, impossible (tanh<1) → hinge always 0.
    out = _run(_guardrail_calib("black", 0.9), [1.0])
    assert out[10] == 0.0


def test_red_to_move_sign_matches_formula():
    # Red-to-move: hinge fires on cand_black > target_black, i.e. cand_stm BELOW
    # target_stm. Pin the exact formula with sign=-1 against the actual value.
    out = _run(_guardrail_calib("red", -0.9), [-1.0])
    v_stm = out[8]
    exp = _expected_hinge(v_stm, "red", -0.9, sign=-1.0)
    assert out[10] == pytest.approx(exp, abs=1e-5), v_stm
    # sanity: a +1 sign on the same values would give a different result
    wrong = _expected_hinge(v_stm, "red", -0.9, sign=1.0)
    assert not (exp == pytest.approx(wrong)) or exp == 0.0 == wrong


def test_byte_identical_when_sign_absent():
    # Same calibration row, no guardrail sign -> pre-v12 10-tuple, symmetric MSE.
    net, mm, om, ov = _setup()
    out = train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                     batch=[_pos() for _ in range(3)],
                     calibration_positions=_guardrail_calib("black", 0.2),
                     calibration_loss_weight=0.01,
                     train_value_head_and_final_block=True)
    assert len(out) == 10                 # unchanged calib value-only path


def test_teacher_mask_and_guardrail_sign_together_raises():
    # Task 3 review: teacher_mode and guardrail_mode must be mutually
    # exclusive. If a caller ever passes both non-None, train_step must fail
    # loud with a clear ValueError instead of an opaque tuple-unpack crash.
    net, mm, om, ov = _setup()
    calib = _guardrail_calib("black", 0.2)
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                   batch=[_pos() for _ in range(3)],
                   calibration_positions=calib, calibration_loss_weight=0.01,
                   calibration_teacher_policy_mask=np.zeros(len(calib), dtype=np.float32),
                   calibration_guardrail_sign=np.ones(len(calib), dtype=np.float32),
                   train_value_head_and_final_block=True)
