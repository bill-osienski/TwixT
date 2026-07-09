"""v14b: gradient projection over the value-only adapter surface
{value_head, value_adapter}. Combines v13's A-yields-to-guardrail projection with
v14's isolated adapter surface. No new flag; slot [13] is self-describing."""
import numpy as np
import pytest
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord
from scripts.GPU.alphazero.calibration_pool import target_in_to_move
from scripts.GPU.alphazero.trainer import (
    MainModule, ValueModule, train_step, freeze_batchnorm_running_stats)


def _adapter_net():
    net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    freeze_batchnorm_running_stats(net)
    return net


def _pos():
    return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                          to_move="red", legal_moves=[(0, 0), (1, 1), (2, 2)],
                          visit_counts=[10, 5, 3], outcome=1.0, active_size=24,
                          ply=0, game_n_moves=10)


def _row(target_black):
    # Shared zero board so an A row and a guardrail row have collinear-up-to-sign
    # value gradients over ANY surface (the v13c forced-conflict construction).
    return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                          to_move="black", legal_moves=[(0, 0), (1, 1)],
                          visit_counts=[0, 0], outcome=target_in_to_move("black", target_black),
                          active_size=24, ply=20, game_n_moves=None)


def _v14b_call(net, projection=True, strength=1.0):
    mm = MainModule(net.encoder, net.policy_head)
    vm = ValueModule(net.value_head, net.value_adapter)
    om, ov = optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)
    # Forced A-vs-guardrail conflict: A row (sign 0) target +0.9 pulls value UP;
    # guardrail row (sign +1) target -0.9 -> hinge relu(v+0.8) active for v>-0.8
    # pushes DOWN -> dot < 0 on the shared zero board.
    calib = [_row(0.9), _row(-0.9)]
    sign = np.array([0.0, 1.0], dtype=np.float32)   # 0 = A row, 1 = guardrail row
    return train_step(
        network=net, main_module=mm, opt_main=om, opt_value=ov,
        batch=[_pos() for _ in range(3)], calibration_positions=calib,
        calibration_loss_weight=0.01, calibration_guardrail_sign=sign,
        guardrail_margin=0.10, train_value_head_and_value_adapter=True,
        value_module=vm,
        post_opening_calibration_gradient_projection=projection,
        post_opening_calibration_projection_strength=strength)


def test_projection_accepted_on_adapter_surface():
    # v14: this combo raised ValueError; v14b: it must run and return a 14-tuple.
    ret = _v14b_call(_adapter_net(), projection=True)
    assert isinstance(ret, tuple) and len(ret) == 14


def test_projection_still_rejected_on_value_head_only():
    net = _adapter_net()
    mm = MainModule(net.encoder, net.policy_head)
    with pytest.raises(ValueError, match="multi-component trainable surface"):
        train_step(network=net, main_module=mm,
                   opt_main=optim.Adam(learning_rate=1e-3),
                   opt_value=optim.Adam(learning_rate=1e-3),
                   batch=[_pos() for _ in range(3)],
                   calibration_positions=[_row(-0.9)],
                   calibration_loss_weight=0.01,
                   calibration_guardrail_sign=np.array([1.0], dtype=np.float32),
                   guardrail_margin=0.10,
                   train_value_head_only=True,
                   post_opening_calibration_gradient_projection=True)


def test_slot13_is_projection_dict_with_folded_grad_norm():
    mx.random.seed(0)
    extra = _v14b_call(_adapter_net(), projection=True)[13]
    assert isinstance(extra, dict)                              # self-describing: dict
    assert "conflict" in extra and "removed_norm" in extra     # projection telemetry
    assert "value_adapter_grad_norm" in extra                  # folded (post-projection)
    assert isinstance(extra["value_adapter_grad_norm"], float)
    assert extra["value_adapter_grad_norm"] >= 0.0


def test_forced_conflict_projects_on_adapter_surface():
    mx.random.seed(0)
    proj = _v14b_call(_adapter_net(), projection=True)[13]
    assert proj["conflict"] is True
    assert proj["c"] != 0.0
    assert proj["removed_norm"] > 0.0


def test_v14_float_slot_unchanged_when_projection_off():
    mx.random.seed(0)
    assert isinstance(_v14b_call(_adapter_net(), projection=False)[13], float)


def test_projection_changes_applied_value_side_update():
    # On the forced conflict, projection removes part of the A push -> the applied
    # value-side update differs from projection-off (same seed, same batch).
    mx.random.seed(0); net_on = _adapter_net()
    before = {k: np.array(v) for k, v in tree_flatten(net_on.parameters())}
    _v14b_call(net_on, projection=True)
    after_on = {k: np.array(v) for k, v in tree_flatten(net_on.parameters())}
    mx.random.seed(0); net_off = _adapter_net()
    _v14b_call(net_off, projection=False)
    after_off = {k: np.array(v) for k, v in tree_flatten(net_off.parameters())}
    vs = [k for k in after_on if k.startswith("value_head.") or k.startswith("value_adapter.")]
    assert any(not np.array_equal(before[k], after_on[k]) for k in vs)      # value side trained
    assert any(not np.array_equal(after_on[k], after_off[k]) for k in vs)   # projection changed it


def test_accumulator_branches_by_type_not_flag():
    # Pin the train() slot-[13] disambiguation: type-based (isinstance dict) with a
    # graceful .get so v13 (dict, no key) and v14b (dict + key) both work.
    from scripts.GPU.alphazero import trainer as trainer_mod
    src = open(trainer_mod.__file__).read()
    assert "isinstance(_extra, dict)" in src
    assert 'get("value_adapter_grad_norm", 0.0)' in src


def test_calib_projection_scope_reflects_surface():
    # Fast-follow: calib_projection_scope was a hardcoded v13 string, so v14b runs
    # mislabeled their surface. It now reflects proj_scope from the accumulator.
    from scripts.GPU.alphazero.calibration_pool import build_post_opening_calibration_block
    b_adapter = build_post_opening_calibration_block(
        config={}, enabled=True,
        loss_accumulator={"steps_done": 1, "proj_enabled": True,
                          "proj_scope": "value_head_and_value_adapter"})
    assert b_adapter["loss"]["calib_projection_scope"] == "value_head_and_value_adapter"
    # v13 / no proj_scope key -> the final-block default (byte-identical to before)
    b_default = build_post_opening_calibration_block(
        config={}, enabled=True, loss_accumulator={"steps_done": 1, "proj_enabled": True})
    assert b_default["loss"]["calib_projection_scope"] == "value_head_and_final_block"
    # trainer.py derives proj_scope from the single-source-of-truth helper
    from scripts.GPU.alphazero import trainer as trainer_mod
    assert '"proj_scope": training_surface_label(' in open(trainer_mod.__file__).read()
