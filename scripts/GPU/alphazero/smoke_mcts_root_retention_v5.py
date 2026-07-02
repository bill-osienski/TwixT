"""v5 gate-0 mechanics smoke. UNLIKE the v4 self-distillation smoke, policy
CE ~ 0 is NOT expected here: root visits are search-improved, so at gate-0
(candidate == BASE) kl_est = CE - H(target) > 0 is the healthy state. The
smoke asserts only what must hold: value term ~ 0 (raw eval-mode anchor
reproduces under the eval-mode calibration forward), policy CE finite and
positive, mask aligned, no NaN. Run after building the v5 manifest, before
training. Pair with the builder's --gate-cases-csv cross-check (root values
vs gate CSVs), which validates the search targets themselves.
"""
from __future__ import annotations

import argparse
import math
import sys

from .calibration_pool import (
    CalibrationPool, split_samples_with_modes, RETENTION_POLICY_LOSS_MODES)
from .trainer import (
    alphazero_loss_batch, CALIB_VALUE_TERM_IDX, CALIB_POLICY_CE_IDX,
    CALIB_POLICY_KL_EST_IDX, CALIB_N_RETENTION_IDX)


def assert_root_retention_mechanics(network, manifest_path: str,
                                    value_tol: float = 1e-4) -> dict:
    pool = CalibrationPool.from_manifest(manifest_path, calibration_target=-0.35)
    if pool.schema != "mcts_root_retention":
        raise AssertionError(
            f"manifest schema is {pool.schema!r}, expected mcts_root_retention")
    retention = [s for s in pool._samples
                 if s.loss_mode in RETENTION_POLICY_LOSS_MODES]
    if not retention:
        raise AssertionError("no mcts_root_retention rows in manifest")
    records, weights, mask = split_samples_with_modes(retention, pool.has_weight_scale)
    if not all(m == 1.0 for m in mask.tolist()):
        raise AssertionError("retention rows produced mask != 1.0 (v3-rerun hazard)")
    prev_training = network.training
    network.eval()      # batch-independent forward; loss path re-wraps in eval anyway
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
            f"raw value anchor FAILED to reproduce: value_mse={value_mse:.3e} "
            f"(tol={value_tol}). Check eval-mode caching / checkpoint / perspective.")
    if policy_ce <= 0.0:
        raise AssertionError(f"policy CE not positive: {policy_ce!r}")
    if kl_est < -1e-6:
        raise AssertionError(f"kl_est negative beyond numerical floor: {kl_est!r}")
    return {"value_mse": value_mse, "policy_ce": policy_ce,
            "kl_est": kl_est, "n_retention": n_retention}


def main(argv=None):
    ap = argparse.ArgumentParser(description="v5 gate-0 root-retention mechanics smoke")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--value-tol", type=float, default=1e-4)
    args = ap.parse_args(argv)
    from .probe_eval import load_network_for_scoring
    network, *_ = load_network_for_scoring(args.base_checkpoint)
    stats = assert_root_retention_mechanics(network, args.manifest,
                                            value_tol=args.value_tol)
    print(f"PASS v5 gate-0 mechanics: value_mse={stats['value_mse']:.3e}, "
          f"policy_ce={stats['policy_ce']:.4f}, kl_est={stats['kl_est']:.4f} "
          f"(EXPECTED > 0 — root visits are search-improved), "
          f"n_retention={stats['n_retention']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
