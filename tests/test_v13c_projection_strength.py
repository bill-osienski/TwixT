"""v13c: projection_strength folds into the effective projection weight
(strength * calibration_loss_weight), scaling the conflict correction without
touching the geometry (c/cos/dot) or any guardrail row."""
import numpy as np
import pytest
import mlx.core as mx
import mlx.optimizers as optim

from scripts.GPU.alphazero.trainer import (
    project_conflicting_gradient, MainModule, freeze_batchnorm_running_stats,
    train_step)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord
from scripts.GPU.alphazero.calibration_pool import target_in_to_move


def _surf(vh, blk):
    return {"value_head": {"w": mx.array(vh, dtype=mx.float32)},
            "block": {"w": mx.array(blk, dtype=mx.float32)}}


def test_weight_scaling_is_linear_same_geometry():
    # folding strength into weight: 2x weight -> 2x correction + removed_norm,
    # identical c/cos/dot (geometry is weight-independent).
    st, a, g = _surf([1.0, 0.0], [0.0]), _surf([1.0, 0.0], [0.0]), _surf([-1.0, 0.0], [0.0])
    out1, t1 = project_conflicting_gradient(st, a, g, weight=0.01)          # strength 1.0
    out2, t2 = project_conflicting_gradient(st, a, g, weight=0.02)          # strength 2.0
    assert t1["conflict"] is True and t2["conflict"] is True
    assert t1["c"] == t2["c"] and t1["dot"] == t2["dot"] and t1["cos"] == t2["cos"]
    dev1 = 1.0 - float(out1["value_head"]["w"][0].item())
    dev2 = 1.0 - float(out2["value_head"]["w"][0].item())
    assert dev2 == pytest.approx(2.0 * dev1, abs=1e-6)
    assert t2["removed_norm"] == pytest.approx(2.0 * t1["removed_norm"], abs=1e-6)


def _pos():
    return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                          to_move="red", legal_moves=[(0, 0), (1, 1), (2, 2)],
                          visit_counts=[10, 5, 3], outcome=1.0, active_size=24,
                          ply=0, game_n_moves=10)


def _row(to_move, target_black):
    return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                          to_move=to_move, legal_moves=[(0, 0), (1, 1)],
                          visit_counts=[0, 0], outcome=target_in_to_move(to_move, target_black),
                          active_size=24, ply=20, game_n_moves=None)


# Forced A-vs-guardrail conflict on the applied surface. Row 0 is an A-correction
# row (sign 0) pulling the value UP toward +0.9; row 1 is a guardrail row with the
# realistic black-to-move sign +1, whose hinge relu(1*(v-(-0.9))-0.10)=relu(v+0.8)
# is active for any fresh-net value v>-0.8 and pushes the value DOWN. Both share the
# same zero board, so g_A and g_G are collinear up to sign => dot<0 (a genuine
# conflict) for any v in (-0.8, 0.9) — i.e. essentially every fresh-net init. Seeded
# for reproducible geometry. (The pre-fix rows gave a stm guardrail target of +0.9
# with sign +1, needing v>1.0 to activate the tanh-bounded hinge — never satisfied,
# so norm_G=0 and the train_step tests asserted a vacuous 0==0.)
_SEED = 0
_CALIB = [_row("black", 0.9), _row("black", -0.9)]
_SIGN = np.array([0.0, 1.0], dtype=np.float32)


def _run(strength):
    mx.random.seed(_SEED)
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    mm = MainModule(net.encoder, net.policy_head)
    om, ov = optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)
    return train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                      batch=[_pos() for _ in range(3)], calibration_positions=_CALIB,
                      calibration_loss_weight=0.01, calibration_guardrail_sign=_SIGN,
                      guardrail_margin=0.10, train_value_head_and_final_block=True,
                      post_opening_calibration_gradient_projection=True,
                      post_opening_calibration_projection_strength=strength)


def test_train_step_folds_strength_into_weight():
    # On a forced conflict step (asserted, so this can never silently pass on a
    # vacuous c=0/removed_norm=0 no-conflict step), removed_norm through the full
    # train_step path must equal strength * calib_weight * |c| * norm_G — a missing
    # multiply is caught here.
    projs = {}
    for strength in (1.0, 2.0):
        proj = _run(strength)[13]
        assert proj["conflict"] is True and proj["c"] != 0.0 and proj["norm_G"] > 0.0, (strength, proj)
        expected = strength * 0.01 * abs(proj["c"]) * proj["norm_G"]
        assert proj["removed_norm"] == pytest.approx(expected, rel=1e-5, abs=1e-9), (strength, proj)
        projs[strength] = proj
    # geometry (c/dot/norm_G) is weight-independent; only the applied correction
    # magnitude scales: 2x strength -> exactly 2x removed_norm, same conflict.
    assert projs[2.0]["c"] == projs[1.0]["c"]
    assert projs[2.0]["dot"] == projs[1.0]["dot"]
    assert projs[2.0]["norm_G"] == projs[1.0]["norm_G"]
    assert projs[2.0]["removed_norm"] == pytest.approx(
        2.0 * projs[1.0]["removed_norm"], rel=1e-5, abs=1e-9)


def test_projection_strength_default_is_one():
    # strength omitted -> effective weight == calibration_loss_weight (numerically
    # identical projection update to v13); removed_norm uses 1.0. Conflict asserted
    # so the default path is exercised on a real correction, not a vacuous 0==0.
    mx.random.seed(_SEED)
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    mm = MainModule(net.encoder, net.policy_head)
    om, ov = optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)
    proj = train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                      batch=[_pos() for _ in range(3)], calibration_positions=_CALIB,
                      calibration_loss_weight=0.01, calibration_guardrail_sign=_SIGN,
                      guardrail_margin=0.10, train_value_head_and_final_block=True,
                      post_opening_calibration_gradient_projection=True)[13]
    assert proj["conflict"] is True and proj["c"] != 0.0
    assert proj["removed_norm"] == pytest.approx(1.0 * 0.01 * abs(proj["c"]) * proj["norm_G"],
                                                 rel=1e-5, abs=1e-9)


def test_strength_does_not_resurrect_no_op():
    # no conflict (dot>=0): the correction is 0 regardless of weight/strength.
    st, a, g = _surf([1.0, 2.0], [3.0]), _surf([1.0, 0.0], [0.0]), _surf([1.0, 0.0], [0.0])
    out2, t2 = project_conflicting_gradient(st, a, g, weight=0.02)          # strength 2.0
    assert t2["conflict"] is False and t2["c"] == 0.0 and t2["removed_norm"] == 0.0
    assert out2 is st                                                       # unchanged


def test_cli_and_telemetry_wiring():
    from scripts.GPU.alphazero import train as train_mod
    from scripts.GPU.alphazero import trainer as trainer_mod
    from scripts.GPU.alphazero import calibration_pool as cp_mod
    tsrc = open(train_mod.__file__).read()
    assert '"--post-opening-calibration-projection-strength"' in tsrc
    assert ("post_opening_calibration_projection_strength="
            "args.post_opening_calibration_projection_strength,") in tsrc
    rsrc = open(trainer_mod.__file__).read()
    # the fold-into-weight multiply + the plumb + both accumulator/mirror keys
    assert ("post_opening_calibration_projection_strength * calibration_loss_weight") in rsrc
    assert ("post_opening_calibration_projection_strength="
            "post_opening_calibration_projection_strength,") in rsrc
    assert '"proj_strength"' in rsrc
    assert '"calib_projection_strength"' in rsrc          # flattening mirror tuple
    csrc = open(cp_mod.__file__).read()
    assert '"calib_projection_strength"' in csrc          # sidecar loss block
