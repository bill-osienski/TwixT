"""v6 gate-0 mechanics smoke (pattern: smoke_mcts_root_retention_v5).

Asserts, at the BASE checkpoint, BEFORE training:
  1. manifest schema == searched_continuation_retention (loading already
     re-applies extra_moves_json and verifies continuation side/sha1 per row);
  2. continuation value anchors reproduce under the eval-mode calibration
     forward: value MSE ~ 0;
  3. per-row policy mask: policy CE == 0.0 exactly on a value-only manifest,
     finite when --emit-continuation-policy rows exist; no NaN anywhere;
  4. HARD schedule assertion: one sample_by_tag round with the locked v6
     schedule draws exactly the scheduled counts per tag (acceptance
     criterion 1 reads calib_n_drawn_by_tag, NOT n_teacher_retention_drawn,
     which is policy-mask-derived and stays 0 on a value-only run).
"""
from __future__ import annotations

import argparse
import math
import random
import sys

from .calibration_pool import (
    CONTINUATION_LOSS_MODE, CalibrationPool, split_samples_with_modes)
from .trainer import (
    alphazero_loss_batch, CALIB_VALUE_TERM_IDX, CALIB_POLICY_CE_IDX,
    CALIB_POLICY_KL_EST_IDX, CALIB_N_RETENTION_IDX)

V6_TAG_SCHEDULE = {
    "black_predrop_correction": 2,
    "goal_line_continuation_retention": 1,
    "old_post_opening_continuation_retention": 2,
    "red_predrop_continuation_retention": 2,
}


def assert_continuation_retention_mechanics(network, manifest_path: str,
                                            value_tol: float = 1e-4) -> dict:
    pool = CalibrationPool.from_manifest(manifest_path, calibration_target=-0.35)
    if pool.schema != CONTINUATION_LOSS_MODE:
        raise AssertionError(
            f"manifest schema is {pool.schema!r}, expected {CONTINUATION_LOSS_MODE}")
    continuation = [s for s in pool._samples
                    if s.loss_mode == CONTINUATION_LOSS_MODE]
    if not continuation:
        raise AssertionError("no searched_continuation_retention rows in manifest")
    records, weights, mask = split_samples_with_modes(
        continuation, pool.has_weight_scale)
    n_policy_rows = int(sum(1 for s in continuation if s.has_policy_target))
    prev_training = network.training
    network.eval()
    try:
        out = alphazero_loss_batch(
            network, records,
            calibration_positions=records,
            calibration_weights=weights,
            calibration_loss_weight=1.0,
            calibration_teacher_policy_mask=mask,
            teacher_value_weight=1.0, teacher_policy_kl_weight=1.0,
        )
        value_mse = float(out[CALIB_VALUE_TERM_IDX])
        policy_ce = float(out[CALIB_POLICY_CE_IDX])
        kl_est = float(out[CALIB_POLICY_KL_EST_IDX])
        n_retention = int(out[CALIB_N_RETENTION_IDX])
    finally:
        network.train(prev_training)
    if not (math.isfinite(value_mse) and math.isfinite(policy_ce)
            and math.isfinite(kl_est)):
        raise AssertionError(
            f"non-finite terms: value_mse={value_mse}, ce={policy_ce}, kl={kl_est}")
    if abs(value_mse) > value_tol:
        raise AssertionError(
            f"continuation value anchor FAILED to reproduce: "
            f"value_mse={value_mse:.3e} (tol={value_tol}). Check eval-mode "
            f"forward / checkpoint / perspective / extra-move reconstruction.")
    if n_policy_rows == 0 and policy_ce != 0.0:
        raise AssertionError(
            f"value-only manifest but policy_ce={policy_ce} != 0 (mask leak?)")
    if n_retention != n_policy_rows:
        raise AssertionError(
            f"mask count {n_retention} != policy-carrying rows {n_policy_rows}")
    # HARD schedule assertion (acceptance criterion 1 source of truth).
    draws = pool.sample_by_tag(V6_TAG_SCHEDULE, random.Random(0))
    draws_by_tag: dict = {}
    for s in draws:
        draws_by_tag[s.tag] = draws_by_tag.get(s.tag, 0) + 1
    if draws_by_tag != V6_TAG_SCHEDULE:
        raise AssertionError(
            f"schedule draw mismatch: {draws_by_tag} != {V6_TAG_SCHEDULE}")
    return {"n_continuation": len(continuation), "n_policy_rows": n_policy_rows,
            "value_mse": value_mse, "policy_ce": policy_ce, "kl_est": kl_est,
            "draws_by_tag": draws_by_tag,
            "tag_counts": pool.tag_counts()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="v6 continuation-retention gate-0 smoke.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--value-tol", type=float, default=1e-4)
    args = ap.parse_args(argv)
    from .probe_eval import load_network_for_scoring
    network, *_ = load_network_for_scoring(args.base_checkpoint)
    report = assert_continuation_retention_mechanics(
        network, args.manifest, value_tol=args.value_tol)
    print(f"PASS v6 continuation retention mechanics: "
          f"n_continuation={report['n_continuation']}, "
          f"value_mse={report['value_mse']:.3e}, "
          f"policy_ce={report['policy_ce']:.4f} "
          f"({report['n_policy_rows']} policy rows), "
          f"draws_by_tag={report['draws_by_tag']}, "
          f"pool_tags={report['tag_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
