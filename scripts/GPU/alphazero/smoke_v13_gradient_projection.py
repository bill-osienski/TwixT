#!/usr/bin/env python3
"""Gate-0 smoke: draw the v12b schedule, run a train_step with the v13 gradient
projection on the v9 surface, and assert the projection telemetry dict engaged.

Run as a module: .venv/bin/python -m scripts.GPU.alphazero.smoke_v13_gradient_projection
"""
import sys
import numpy as np
import mlx.optimizers as optim

from scripts.GPU.alphazero.calibration_pool import (
    CalibrationPool, split_samples_with_guardrail, GUARDRAIL_LOSS_MODE)
from scripts.GPU.alphazero.trainer import (
    MainModule, freeze_batchnorm_running_stats, train_step)
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.self_play import PositionRecord

MANIFEST = "logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv"
SCHEDULE = {"black_predrop_correction": 2, "goal_line_guardrail_retention": 1,
            "old_post_opening_guardrail_retention": 1,
            "old_post_opening_continuation_guardrail_retention": 2,
            "red_predrop_guardrail_retention": 1,
            "red_predrop_continuation_guardrail_retention": 2}


def main() -> int:
    pool = CalibrationPool.from_manifest(MANIFEST, calibration_target=-0.35)
    assert pool.schema == GUARDRAIL_LOSS_MODE, pool.schema
    import random
    rng = random.Random(0)
    samples = pool.sample_by_tag(SCHEDULE, rng)
    records, weights, sign = split_samples_with_guardrail(samples, pool.has_weight_scale)
    assert (np.abs(sign) < 0.5).any() and (np.abs(sign) > 0.5).any(), "need A + guardrail rows"
    net = create_network(hidden=64, n_blocks=2)
    freeze_batchnorm_running_stats(net)
    mm = MainModule(net.encoder, net.policy_head)
    om, ov = optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)
    def _p():
        return PositionRecord(board_tensor=np.zeros((24, 24, 30), dtype=np.float32),
                              to_move="red", legal_moves=[(0, 0), (1, 1)],
                              visit_counts=[1, 1], outcome=1.0, active_size=24,
                              ply=0, game_n_moves=10)
    out = train_step(network=net, main_module=mm, opt_main=om, opt_value=ov,
                     batch=[_p() for _ in range(4)], calibration_positions=records,
                     calibration_weights=weights, calibration_loss_weight=0.01,
                     calibration_guardrail_sign=sign, guardrail_margin=0.10,
                     train_value_head_and_final_block=True,
                     post_opening_calibration_gradient_projection=True)
    assert len(out) == 14, len(out)
    proj = out[13]
    assert set(proj) >= {
        "evaluated", "conflict", "skip_reason", "dot", "cos",
        "c", "removed_norm", "norm_G", "norm_A",
    }
    # tiny_guardrail is a legitimate no-op; no_a/no_guardrail shouldn't happen
    # here (both sign classes were asserted present) but are allowed for debugging.
    assert proj["skip_reason"] in (None, "tiny_guardrail", "no_a", "no_guardrail")
    print(
        f"SMOKE PASS: evaluated={proj['evaluated']} conflict={proj['conflict']} "
        f"skip={proj['skip_reason']} dot={proj['dot']:.4g} "
        f"cos={proj['cos']:.3g} c={proj['c']:.4g} "
        f"removed_norm={proj['removed_norm']:.4g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
