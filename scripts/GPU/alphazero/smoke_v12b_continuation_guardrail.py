#!/usr/bin/env python3
"""Gate-0 smoke: load the v12b continuation-guardrail manifest, draw the v12b
schedule, run a train_step with the guardrail sign, and assert the guardrail
telemetry engaged and at least one continuation-guardrail row was drawn.

Run as a module (not by file path) so the scripts.* imports resolve:
  .venv/bin/python -m scripts.GPU.alphazero.smoke_v12b_continuation_guardrail
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
CONTINUATION_TAGS = {"old_post_opening_continuation_guardrail_retention",
                     "red_predrop_continuation_guardrail_retention"}


def main() -> int:
    pool = CalibrationPool.from_manifest(MANIFEST, calibration_target=-0.35)
    assert pool.schema == GUARDRAIL_LOSS_MODE, pool.schema
    import random
    rng = random.Random(0)
    samples = pool.sample_by_tag(SCHEDULE, rng)
    assert any(s.tag in CONTINUATION_TAGS for s in samples), "no continuation guardrail row drawn"
    records, weights, sign = split_samples_with_guardrail(samples, pool.has_weight_scale)
    assert set(np.unique(sign)) <= {-1.0, 0.0, 1.0}
    assert (np.abs(sign) > 0).sum() > 0, "no guardrail rows drawn"
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
                     train_value_head_and_final_block=True)
    assert len(out) == 13, len(out)
    print(f"SMOKE PASS: guardrail_hinge_loss={out[10]:.4g} active_frac={out[11]:.3g} "
          f"guardrail_n={out[12]} (schema={pool.schema})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
