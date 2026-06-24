"""Smoke: real v2 manifest → pool → split → weighted calibration loss.

Run: .venv/bin/python -m scripts.GPU.alphazero.smoke_targeted_calibration_v2 \
        logs/eval/targeted_calibration_v2_from_calib020_0001.csv
Exits non-zero on any failure. Integration check (needs the manifest + replays).
"""
from __future__ import annotations

import random
import sys

import numpy as np

from .calibration_pool import CalibrationPool, split_samples
from .network import create_network
from .trainer import alphazero_loss_batch


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    manifest = argv[0] if argv else "logs/eval/targeted_calibration_v2_from_calib020_0001.csv"

    pool = CalibrationPool.from_manifest(manifest, calibration_target=-0.35)
    assert pool.schema == "per_row_target", pool.schema
    assert pool.has_weight_scale is True
    print("pool:", len(pool), "schema", pool.schema, "tags", pool.tag_counts())

    samples = pool.sample(6, random.Random(0))
    records, weights = split_samples(samples, pool.has_weight_scale)
    assert weights is not None and len(weights) == len(records) == 6

    net = create_network(in_channels=30, hidden=64, n_blocks=2)
    out = alphazero_loss_batch(
        net, records,                      # use calib records as a stand-in main batch
        calibration_positions=records,
        calibration_weights=weights,
        calibration_loss_weight=0.01)
    assert len(out) == 10, len(out)
    calib_loss = float(out[7].item())
    calib_mean = float(out[8].item())
    assert np.isfinite(calib_loss) and np.isfinite(calib_mean)
    print(f"OK calib_loss={calib_loss:.4f} calib_value_mean={calib_mean:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
