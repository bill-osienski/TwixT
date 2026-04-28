"""Verify the trainer's per-iter sidecar carries probe_summary keyed by
tier alongside the legacy forced_probe_summary field.

Doesn't run training; uses a synthetic iter dict and the trainer's
sidecar serialization helper.
"""
from __future__ import annotations


def test_sidecar_contains_tiered_probe_summary():
    """The helper must produce {'forced': <payload>, 'strong_advantage': <payload-or-None>}."""
    from scripts.GPU.alphazero.trainer import build_probe_summary_block

    forced_payload = {
        "n": 28, "sign_correct": 25, "sign_correct_pct": 0.893,
        "median_abs_v": 0.61, "delta_sign_correct_pct": 0.02,
        "delta_median_abs_v": 0.01, "rolling5_sign_correct_pct": 0.86,
        "rolling5_median_abs_v": 0.59, "n_skipped_size": 0,
    }

    block = build_probe_summary_block(
        forced_summary=forced_payload,
        strong_advantage_summary=None,
    )
    assert block == {"forced": forced_payload, "strong_advantage": None}

    sa_payload = dict(forced_payload, n=20, sign_correct=14)
    block2 = build_probe_summary_block(
        forced_summary=forced_payload,
        strong_advantage_summary=sa_payload,
    )
    assert block2 == {"forced": forced_payload, "strong_advantage": sa_payload}
