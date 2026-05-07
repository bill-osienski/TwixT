"""Sidecar telemetry builders for Spec 2 conversion correction + recovery
bucket. Pure dict math — no I/O, no MLX."""
from __future__ import annotations
from typing import Optional


def build_conversion_training_block(
    config: dict,
    *,
    enabled: bool,
    buffer_stats: dict,
    loss_accumulator: dict,
    sample_accumulator: Optional[dict],
) -> dict:
    """Build the per-iter conversion_training sidecar block.

    Schema is stable across enabled/disabled — every field is present,
    only values differ. When enabled=False, loss/sample fields emit zeros
    regardless of accumulator state (defensive).

    consistency.available is False when sample_accumulator is None (Phase 2:
    loss wired but sampler stats not yet wired). True when sample_accumulator
    is provided (Phase 3+).
    """
    steps = max(loss_accumulator.get("steps_done", 0), 0)
    batch_size = loss_accumulator.get("batch_size", 1)

    if enabled:
        avg_aux = (loss_accumulator["sum_aux"] / steps) if steps > 0 else 0.0
        avg_cov = (loss_accumulator["sum_aux_coverage"] / steps) if steps > 0 else 0.0
        seen = loss_accumulator.get("sum_aux_n_eligible", 0)
        seen_frac = (seen / (steps * batch_size)) if steps > 0 else 0.0
    else:
        avg_aux = 0.0
        avg_cov = 0.0
        seen = 0
        seen_frac = 0.0

    # Sample accumulator may be None during Phase 2 (before sampler stats wired).
    # Also, consistency check is only meaningful when enabled=True (otherwise
    # loss accumulator forces seen=0 while sampler-side drawn can be >0 from
    # incidental eligible positions in uniform batches, creating false negatives).
    # In both cases, emit consistency.available=False.
    sample_stats_meaningful = enabled and sample_accumulator is not None
    if not sample_stats_meaningful:
        sample_block = {
            "eligible_drawn_total": 0,
            "eligible_drawn_fraction": 0.0,
            "cap_was_binding_steps": 0,
            "boost_inactive_steps": 0,
        }
        consistency_block = {
            "drawn_vs_seen_match": None,
            "drawn_minus_seen": None,
            "available": False,
        }
    else:
        drawn_total = sample_accumulator.get("eligible_drawn_total", 0)
        sample_block = {
            "eligible_drawn_total": drawn_total,
            "eligible_drawn_fraction": (
                drawn_total / (steps * batch_size) if steps > 0 else 0.0
            ),
            "cap_was_binding_steps": sample_accumulator.get("cap_was_binding_steps", 0),
            "boost_inactive_steps": sample_accumulator.get("boost_inactive_steps", 0),
        }
        drawn_minus_seen = drawn_total - seen
        consistency_block = {
            "drawn_vs_seen_match": (drawn_minus_seen == 0),
            "drawn_minus_seen": int(drawn_minus_seen),
            "available": True,
        }

    return {
        "version": 1,
        "enabled": bool(enabled),
        "config": dict(config),
        "buffer": dict(buffer_stats),
        "loss": {
            "aux_loss_avg_iter": float(avg_aux),
            "aux_target_coverage_rate": float(avg_cov),
            "aux_positions_seen_in_training": int(seen),
            "aux_positions_fraction_in_batches": float(seen_frac),
        },
        "sample_stats": sample_block,
        "consistency": consistency_block,
    }
