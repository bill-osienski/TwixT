"""Tiny end-to-end v14 smoke: adapter net + guardrail calibration -> gate opens,
surface isolated. Runs on synthetic data (no gitignored fixtures)."""
import sys
import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord
from scripts.GPU.alphazero.calibration_pool import target_in_to_move
from scripts.GPU.alphazero.trainer import (
    MainModule, ValueModule, train_step, freeze_batchnorm_running_stats)


def _pos():
    return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                          to_move="red", legal_moves=[(0, 0), (1, 1), (2, 2)],
                          visit_counts=[10, 5, 3], outcome=1.0, active_size=24,
                          ply=0, game_n_moves=10)


def _guard_row(target_black):
    return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                          to_move="black", legal_moves=[(0, 0), (1, 1)],
                          visit_counts=[0, 0], outcome=target_in_to_move("black", target_black),
                          active_size=24, ply=20, game_n_moves=None)


def main() -> int:
    mx.random.seed(0)
    net = create_network(hidden=64, n_blocks=2, value_adapter=True)
    freeze_batchnorm_running_stats(net)
    mm = MainModule(net.encoder, net.policy_head)
    vm = ValueModule(net.value_head, net.value_adapter)
    om, ov = optim.Adam(learning_rate=1e-2), optim.Adam(learning_rate=1e-2)
    calib = [_guard_row(0.9), _guard_row(-0.9)]
    sign = np.array([1.0, 1.0], dtype=np.float32)
    before = {k: np.array(v) for k, v in tree_flatten(net.parameters())}
    for _ in range(10):
        train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                   batch=[_pos() for _ in range(3)], calibration_positions=calib,
                   calibration_loss_weight=0.01, calibration_guardrail_sign=sign,
                   guardrail_margin=0.10, train_value_head_and_value_adapter=True,
                   value_module=vm)
    after = {k: np.array(v) for k, v in tree_flatten(net.parameters())}
    gate = float(net.value_adapter.gate[0])
    if gate == 0.0:
        print(f"SMOKE FAIL: gate never left 0 (gate={gate})")
        return 1
    for k in after:
        if k.startswith("value_head.") or k.startswith("value_adapter."):
            continue
        if not np.array_equal(before[k], after[k]):
            print(f"SMOKE FAIL: frozen tensor changed: {k}")
            return 1
    print(f"SMOKE OK: gate={gate:.4f}; surface isolated (value_head + value_adapter only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
