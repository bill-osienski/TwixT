"""v13 train_step wiring: projection off is byte-identical (13-tuple); on with a
mixed A+guardrail batch produces a 14-tuple with projection telemetry; the
value-head-only surface is rejected."""
import numpy as np
import pytest
import mlx.optimizers as optim

from scripts.GPU.alphazero.trainer import (
    MainModule, freeze_batchnorm_running_stats, train_step)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord
from scripts.GPU.alphazero.calibration_pool import target_in_to_move


def _pos(to_move="red", outcome=1.0):
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move=to_move, legal_moves=[(0, 0), (1, 1), (2, 2)],
        visit_counts=[10, 5, 3], outcome=outcome, active_size=24,
        ply=0, game_n_moves=10)


def _row(to_move, target_black):
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move=to_move, legal_moves=[(0, 0), (1, 1)], visit_counts=[0, 0],
        outcome=target_in_to_move(to_move, target_black), active_size=24,
        ply=20, game_n_moves=None)


def _setup():
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    mm = MainModule(net.encoder, net.policy_head)
    return net, mm, optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)


# A row (sign 0) + guardrail row (sign +1)
_CALIB = [_row("black", -0.35), _row("red", -0.9)]
_SIGN = np.array([0.0, 1.0], dtype=np.float32)


def _run(projection, **kw):
    net, mm, om, ov = _setup()
    return train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                      batch=[_pos() for _ in range(3)],
                      calibration_positions=_CALIB, calibration_loss_weight=0.01,
                      calibration_guardrail_sign=_SIGN, guardrail_margin=0.10,
                      train_value_head_and_final_block=True,
                      post_opening_calibration_gradient_projection=projection, **kw)


def test_projection_off_is_13_tuple():
    out = _run(projection=False)
    assert len(out) == 13                     # byte-identical guardrail path


def test_projection_on_appends_telemetry_dict():
    out = _run(projection=True)
    assert len(out) == 14
    telem = out[13]
    assert isinstance(telem, dict)
    assert set(telem) >= {"evaluated", "conflict", "skip_reason", "dot", "cos",
                          "c", "removed_norm", "norm_G", "norm_A"}


def test_projection_requires_final_block_surface():
    net, mm, om, ov = _setup()
    with pytest.raises(ValueError, match="train-value-head-and-final-block"):
        train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                   batch=[_pos() for _ in range(3)],
                   calibration_positions=_CALIB, calibration_loss_weight=0.01,
                   calibration_guardrail_sign=_SIGN, guardrail_margin=0.10,
                   train_value_head_only=True,
                   post_opening_calibration_gradient_projection=True)


def test_projection_no_guardrail_rows_skips():
    # all-A batch (sign all 0) -> no guardrail rows -> skip_reason no_guardrail
    net, mm, om, ov = _setup()
    out = train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                     batch=[_pos() for _ in range(3)],
                     calibration_positions=[_row("black", -0.35)],
                     calibration_loss_weight=0.01,
                     calibration_guardrail_sign=np.array([0.0], np.float32),
                     guardrail_margin=0.10, train_value_head_and_final_block=True,
                     post_opening_calibration_gradient_projection=True)
    assert len(out) == 14
    assert out[13]["evaluated"] is False and out[13]["skip_reason"] == "no_guardrail"


def test_cli_projection_flag_and_plumb():
    from scripts.GPU.alphazero import train as train_mod
    from scripts.GPU.alphazero import trainer as trainer_mod
    tsrc = open(train_mod.__file__).read()
    assert '"--post-opening-calibration-gradient-projection"' in tsrc
    assert ("post_opening_calibration_gradient_projection="
            "args.post_opening_calibration_gradient_projection,") in tsrc
    rsrc = open(trainer_mod.__file__).read()
    # forwarded to train_step at the calibration call site(s)
    assert ("post_opening_calibration_gradient_projection="
            "post_opening_calibration_gradient_projection,") in rsrc


def test_projection_telemetry_accumulation_and_json():
    from scripts.GPU.alphazero import trainer as trainer_mod
    from scripts.GPU.alphazero import calibration_pool as cp_mod
    rsrc = open(trainer_mod.__file__).read()
    # accumulators + the 14-tuple telemetry read
    assert "sum_proj_dot" in rsrc
    assert "proj_conflict_steps" in rsrc
    assert "len(_ret) == 14" in rsrc
    assert 'proj["skip_reason"]' in rsrc
    csrc = open(cp_mod.__file__).read()
    for k in ('"calib_projection_enabled"', '"calib_projection_conflict_steps"',
              '"calib_projection_conflict_rate"', '"calib_projection_dot_avg"',
              '"calib_projection_cos_avg"', '"calib_projection_c_avg"',
              '"calib_projection_removed_norm_avg"',
              '"calib_projection_guardrail_grad_norm_avg"',
              '"calib_projection_a_grad_norm_avg"',
              '"calib_projection_no_a_steps"', '"calib_projection_no_guardrail_steps"',
              '"calib_projection_tiny_guardrail_steps"',
              '"calib_projection_no_conflict_steps"', '"calib_projection_scope"'):
        assert k in csrc, k


# The sidecar loss block (calibration_pool.py, asserted above) is NOT the row the
# operator inspects — model_iter_*.json is a FLATTENED copy built by the
# `_teacher_calib_scalars` mirror in trainer.py's train() loop. That mirror must
# also list every calib_projection_* key, or the projection telemetry is silently
# dropped from the per-iteration row (the v13-run bug this guards against).
_PROJECTION_FLAT_KEYS = (
    '"calib_projection_enabled"', '"calib_projection_scope"',
    '"calib_projection_conflict_steps"', '"calib_projection_conflict_rate"',
    '"calib_projection_dot_avg"', '"calib_projection_cos_avg"',
    '"calib_projection_c_avg"', '"calib_projection_removed_norm_avg"',
    '"calib_projection_guardrail_grad_norm_avg"', '"calib_projection_a_grad_norm_avg"',
    '"calib_projection_no_a_steps"', '"calib_projection_no_guardrail_steps"',
    '"calib_projection_tiny_guardrail_steps"', '"calib_projection_no_conflict_steps"',
)


def test_projection_telemetry_flattened_into_model_iter_row():
    from scripts.GPU.alphazero import trainer as trainer_mod
    rsrc = open(trainer_mod.__file__).read()
    # trainer.py never defines the sidecar keys (those live in calibration_pool);
    # the only place these quoted strings appear is the flattening mirror, so this
    # pins that the mirror surfaces the projection telemetry into the flat row.
    for k in _PROJECTION_FLAT_KEYS:
        assert k in rsrc, k
