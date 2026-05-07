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
    # Phase 2 forward-compat seam: in Phase 3+ production wiring, sample_accumulator
    # is always a dict. None is reachable from unit tests and the disabled path.
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


def is_recovery_or_extreme_closeout_drift(
    record: dict,
    *,
    du_threshold: int,
    delay_threshold: int,
) -> bool:
    """Predicate for the recovery / extreme-closeout-drift bucket.

    Three OR-ed clauses (Spec 2 §8.4):
      - dominant_unavailable_moves >= du_threshold
      - conversion_delay_plies >= delay_threshold
      - outcome_class == 2 AND detected (state_cap_after_detection)
    """
    if not record.get("detected"):
        return False

    # du_moves field varies by outcome_class (Spec 2 §8.4 lock)
    outcome_class = record.get("outcome_class")
    if outcome_class == 1:
        du_moves = record.get("winner_moves_with_dominant_unavailable")
    else:
        du_moves = record.get("dominant_unavailable_moves")
    if du_moves is not None and du_moves >= du_threshold:
        return True

    delay = record.get("conversion_delay_plies")
    if delay is not None and delay >= delay_threshold:
        return True

    if outcome_class == 2 and record.get("reason") == "state_cap":
        return True

    return False


def _trigger_breakdown(record, *, du_threshold, delay_threshold):
    """Return the set of clauses that fired for this record."""
    triggers = set()
    outcome_class = record.get("outcome_class")
    if outcome_class == 1:
        du = record.get("winner_moves_with_dominant_unavailable")
    else:
        du = record.get("dominant_unavailable_moves")
    if du is not None and du >= du_threshold:
        triggers.add("dominant_unavailable")
    delay = record.get("conversion_delay_plies")
    if delay is not None and delay >= delay_threshold:
        triggers.add("delay_ge_threshold")
    if outcome_class == 2 and record.get("detected") and record.get("reason") == "state_cap":
        triggers.add("state_cap_after_detection")
    return triggers


def _percentile(values, p):
    if not values:
        return 0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def build_recovery_block(
    records: list,
    *,
    du_threshold: int,
    delay_threshold: int,
    enabled: bool = True,
) -> dict:
    """Build the per-iter recovery_or_extreme_closeout_drift sidecar block.

    Telemetry-only — Spec 2 §8.3-§8.5. Reads existing goal_completion_record
    fields. No new computation.
    """
    if not enabled:
        return {
            "version": 1,
            "enabled": False,
            "config": {
                "dominant_unavailable_moves_threshold": du_threshold,
                "delay_threshold": delay_threshold,
            },
            "games_total": len(records),
            "detected_games": 0,
            "count": 0,
            "rate": 0.0,
            "rate_among_detected": 0.0,
            "dominant_unavailable_moves": {"p50": 0, "p90": 0, "p95": 0, "max": 0, "mean": 0.0},
            "trigger_breakdown": {
                "dominant_unavailable_only": 0,
                "delay_ge_threshold_only": 0,
                "state_cap_after_detection_only": 0,
                "multiple_triggers": 0,
            },
        }

    games_total = len(records)
    detected_games = sum(1 for r in records if r.get("detected"))

    # DU values across detected games
    du_values = []
    for r in records:
        if not r.get("detected"):
            continue
        oc = r.get("outcome_class")
        v = (r.get("winner_moves_with_dominant_unavailable")
             if oc == 1
             else r.get("dominant_unavailable_moves"))
        if v is not None:
            du_values.append(v)

    qualifying = [
        r for r in records
        if is_recovery_or_extreme_closeout_drift(
            r, du_threshold=du_threshold, delay_threshold=delay_threshold
        )
    ]

    # Trigger breakdown — mutually exclusive partition
    breakdown = {
        "dominant_unavailable_only": 0,
        "delay_ge_threshold_only": 0,
        "state_cap_after_detection_only": 0,
        "multiple_triggers": 0,
    }
    for r in qualifying:
        triggers = _trigger_breakdown(r, du_threshold=du_threshold,
                                      delay_threshold=delay_threshold)
        if len(triggers) >= 2:
            breakdown["multiple_triggers"] += 1
        elif "dominant_unavailable" in triggers:
            breakdown["dominant_unavailable_only"] += 1
        elif "delay_ge_threshold" in triggers:
            breakdown["delay_ge_threshold_only"] += 1
        elif "state_cap_after_detection" in triggers:
            breakdown["state_cap_after_detection_only"] += 1

    return {
        "version": 1,
        "enabled": True,
        "config": {
            "dominant_unavailable_moves_threshold": du_threshold,
            "delay_threshold": delay_threshold,
        },
        "games_total": games_total,
        "detected_games": detected_games,
        "count": len(qualifying),
        "rate": (len(qualifying) / games_total) if games_total > 0 else 0.0,
        "rate_among_detected": (
            len(qualifying) / detected_games if detected_games > 0 else 0.0
        ),
        "dominant_unavailable_moves": {
            "p50": _percentile(du_values, 50),
            "p90": _percentile(du_values, 90),
            "p95": _percentile(du_values, 95),
            "max": max(du_values) if du_values else 0,
            "mean": (sum(du_values) / len(du_values)) if du_values else 0.0,
        },
        "trigger_breakdown": breakdown,
    }
