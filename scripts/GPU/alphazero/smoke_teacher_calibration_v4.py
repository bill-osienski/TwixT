"""Gate-0 pre-flight self-distillation check (spec §5.1). With base == teacher,
the v4 calibration forward must reproduce the teacher's stored value and policy
on retention rows: value_mse ≈ 0 and kl_est ≈ 0 (CE ≈ teacher entropy). Run after
building the manifest, before any training run.
"""
from __future__ import annotations

import argparse
import sys

from .calibration_pool import CalibrationPool, split_samples_with_modes
from .trainer import (
    alphazero_loss_batch, CALIB_VALUE_TERM_IDX, CALIB_POLICY_KL_EST_IDX)


def assert_self_distillation(network, manifest_path: str, tol: float = 1e-4) -> dict:
    pool = CalibrationPool.from_manifest(manifest_path, calibration_target=-0.35)
    if pool.schema != "teacher_retention":
        raise AssertionError(f"manifest schema is {pool.schema!r}, expected teacher_retention")
    retention = [s for s in pool._samples if s.loss_mode == "teacher_retention"]
    if not retention:
        raise AssertionError("no teacher_retention rows in manifest")
    records, weights, mask = split_samples_with_modes(retention, pool.has_weight_scale)
    # The main-batch losses are IGNORED here — we only read the calibration outputs.
    # Reusing `records` as the main batch avoids constructing unrelated dummy positions.
    out = alphazero_loss_batch(
        network, records,
        calibration_positions=records,
        calibration_weights=weights,
        calibration_loss_weight=1.0,
        calibration_teacher_policy_mask=mask,
        teacher_value_weight=1.0, teacher_policy_kl_weight=1.0,
    )
    value_mse = float(out[CALIB_VALUE_TERM_IDX])
    kl_est = float(out[CALIB_POLICY_KL_EST_IDX])
    if abs(value_mse) > tol or abs(kl_est) > tol:
        raise AssertionError(
            f"self-distillation FAILED: value_mse={value_mse:.3e}, kl_est={kl_est:.3e} "
            f"(tol={tol}). Check checkpoint / canonicalization / perspective / policy "
            f"alignment / accidental MCTS targets.")
    return {"value_mse": value_mse, "kl_est": kl_est}


def main(argv=None):
    ap = argparse.ArgumentParser(description="v4 gate-0 self-distillation smoke")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--teacher-checkpoint", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args(argv)
    from .probe_eval import load_network_for_scoring
    network = load_network_for_scoring(args.teacher_checkpoint)
    stats = assert_self_distillation(network, args.manifest, tol=args.tol)
    print(f"PASS gate-0 self-distillation: value_mse={stats['value_mse']:.3e}, "
          f"kl_est={stats['kl_est']:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
