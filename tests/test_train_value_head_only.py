"""--train-value-head-only: opt_main.update is skipped so ONLY value_head.*
tensors can change. Behavior tests run real train_step on a tiny net (with BN
running stats frozen, as v8 runs use BOTH flags); wiring that lives inside the
4000-line train loop is pinned source-level (precedent:
tests/test_trainer_teacher_mode_gate.py)."""
import re

import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from scripts.GPU.alphazero import train as train_mod
from scripts.GPU.alphazero import trainer as trainer_mod
from scripts.GPU.alphazero.trainer import (
    MainModule, freeze_batchnorm_running_stats, train_step)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord


def _pos():
    return PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="red", legal_moves=[(0, 0), (1, 1), (2, 2)],
        visit_counts=[10, 5, 3], outcome=1.0, active_size=24,
        ply=0, game_n_moves=10,
    )


def _setup():
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)     # v8 pairs the flags; keeps
    #  encoder.*.running_mean/var from moving via forward-pass tracking
    mm = MainModule(net.encoder, net.policy_head)
    opt_main = optim.Adam(learning_rate=1e-3)
    opt_value = optim.Adam(learning_rate=1e-3)
    return net, mm, opt_main, opt_value


def _params(net):
    return dict(tree_flatten(net.parameters()))


def _changed_keys(before, after):
    return sorted(k for k in before
                  if not bool(mx.array_equal(before[k], after[k]).item()))


def test_flag_on_only_value_head_changes():
    net, mm, opt_main, opt_value = _setup()
    before = _params(net)
    for _ in range(2):
        out = train_step(network=net, main_module=mm, opt_main=opt_main,
                         opt_value=opt_value, batch=[_pos() for _ in range(3)],
                         train_value_head_only=True)
    assert len(out) == 7                       # arity unchanged
    changed = _changed_keys(before, _params(net))
    assert changed, "value head must still train"
    assert all(k.startswith("value_head.") for k in changed), changed


def test_flag_off_default_trains_encoder_and_policy_too():
    net, mm, opt_main, opt_value = _setup()
    before = _params(net)
    for _ in range(2):
        train_step(network=net, main_module=mm, opt_main=opt_main,
                   opt_value=opt_value, batch=[_pos() for _ in range(3)])
    changed = _changed_keys(before, _params(net))
    assert any(k.startswith("encoder.") for k in changed)
    assert any(k.startswith("policy_head.") for k in changed)
    assert any(k.startswith("value_head.") for k in changed)


def test_flag_on_with_calibration_batch_keeps_14_tuple():
    """v8 trains WITH the v7 calibration manifest: the masked teacher-mode
    path must still return its 14-tuple under the flag."""
    net, mm, opt_main, opt_value = _setup()
    calib = [PositionRecord(
        board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
        to_move="black", legal_moves=[(0, 0), (1, 1)],
        visit_counts=[0, 0], outcome=-0.35, active_size=24,
        ply=20, game_n_moves=None)]
    out = train_step(network=net, main_module=mm, opt_main=opt_main,
                     opt_value=opt_value, batch=[_pos() for _ in range(3)],
                     calibration_positions=calib,
                     calibration_loss_weight=0.01,
                     calibration_teacher_policy_mask=np.zeros((1,), dtype=np.float32),
                     teacher_value_weight=1.0, teacher_policy_kl_weight=0.25,
                     train_value_head_only=True)
    assert len(out) == 14


def test_train_loop_wiring_source_level():
    src = open(trainer_mod.__file__).read()
    # both train_step call sites in the train loop forward the flag
    assert len(re.findall(r"train_value_head_only=train_value_head_only,", src)) == 2
    # checkpoint JSON records the run config
    assert '"train_value_head_only": train_value_head_only,' in src


def test_cli_flag_exists_and_plumbs():
    src = open(train_mod.__file__).read()
    assert '"--train-value-head-only"' in src
    assert "train_value_head_only=args.train_value_head_only," in src
