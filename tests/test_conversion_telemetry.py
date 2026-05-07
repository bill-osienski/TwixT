# tests/test_conversion_telemetry.py
"""Sidecar telemetry tests (Spec 2 §8). Recovery-bucket tests added in Task 15."""
from scripts.GPU.alphazero.conversion_telemetry import build_conversion_training_block


def _config(loss_weight=0.05, effective=0.05):
    return {
        "configured_loss_weight": loss_weight,
        "effective_loss_weight": effective,
        "completion_weight": 1.0,
        "reducer_weight": 0.35,
        "max_total_goal_distance": 2,
        "min_component_size": 8,
        "sample_boost": 1.0,
        "max_batch_fraction": 0.15,
    }


def _zero_buffer_stats():
    return {
        "eligible_positions_in_buffer": 0,
        "eligible_position_rate": 0.0,
        "eligible_positions_at_active_size": 0,
        "eligible_rate_at_active_size": 0.0,
    }


def _zero_loss_acc():
    return {
        "sum_aux": 0.0,
        "sum_aux_coverage": 0.0,
        "sum_aux_n_eligible": 0,
        "steps_done": 0,
        "batch_size": 256,
    }


def test_conversion_training_block_schema_when_disabled():
    block = build_conversion_training_block(
        config=_config(effective=0.0),
        enabled=False,
        buffer_stats=_zero_buffer_stats(),
        loss_accumulator=_zero_loss_acc(),
        sample_accumulator=None,
    )
    assert block["version"] == 1
    assert block["enabled"] is False
    assert block["config"]["configured_loss_weight"] == 0.05
    assert block["config"]["effective_loss_weight"] == 0.0
    assert block["loss"]["aux_loss_avg_iter"] == 0.0
    assert block["loss"]["aux_target_coverage_rate"] == 0.0
    assert block["loss"]["aux_positions_seen_in_training"] == 0
    assert block["loss"]["aux_positions_fraction_in_batches"] == 0.0
    # Stable schema
    assert "consistency" in block
    # When sample_accumulator is None, consistency reports unavailable
    assert block["consistency"]["available"] is False
    assert block["consistency"]["drawn_vs_seen_match"] is None
    assert block["consistency"]["drawn_minus_seen"] is None


def test_conversion_training_consistency_unavailable_when_phase2_only():
    """Phase 2 wires loss before Phase 3 sampler stats. With sum_aux_n_eligible>0
    but sample_accumulator=None, consistency must report available=False, NOT
    drawn_vs_seen_match=False (which would be a false positive)."""
    block = build_conversion_training_block(
        config=_config(),
        enabled=True,
        buffer_stats={"eligible_positions_in_buffer": 100,
                      "eligible_position_rate": 0.1,
                      "eligible_positions_at_active_size": 100,
                      "eligible_rate_at_active_size": 0.1},
        loss_accumulator={"sum_aux": 100.0, "sum_aux_coverage": 5.0,
                          "sum_aux_n_eligible": 1280, "steps_done": 50,
                          "batch_size": 256},
        sample_accumulator=None,
    )
    assert block["enabled"] is True
    assert block["loss"]["aux_positions_seen_in_training"] == 1280
    assert block["consistency"]["available"] is False
    assert block["consistency"]["drawn_vs_seen_match"] is None
    assert block["consistency"]["drawn_minus_seen"] is None


def test_conversion_training_block_schema_when_enabled():
    block = build_conversion_training_block(
        config=_config(),
        enabled=True,
        buffer_stats={"eligible_positions_in_buffer": 1234,
                      "eligible_position_rate": 0.0247,
                      "eligible_positions_at_active_size": 980,
                      "eligible_rate_at_active_size": 0.0312},
        loss_accumulator={"sum_aux": 100.0,    # 100 / 50 = 2.0 avg
                          "sum_aux_coverage": 5.0,    # 5 / 50 = 0.1
                          "sum_aux_n_eligible": 1280,
                          "steps_done": 50,
                          "batch_size": 256},
        sample_accumulator=None,
    )
    assert block["enabled"] is True
    assert block["config"]["effective_loss_weight"] == 0.05
    assert block["loss"]["aux_loss_avg_iter"] == 2.0
    assert block["loss"]["aux_target_coverage_rate"] == 0.1
    assert block["loss"]["aux_positions_seen_in_training"] == 1280
    assert block["loss"]["aux_positions_fraction_in_batches"] == 1280 / (50 * 256)


def test_conversion_training_block_disabled_emits_zero_telemetry():
    """Even with non-zero accumulator, if enabled=False the loss block reports zeros."""
    block = build_conversion_training_block(
        config=_config(effective=0.0),
        enabled=False,
        buffer_stats=_zero_buffer_stats(),
        loss_accumulator={"sum_aux": 999.0, "sum_aux_coverage": 0.5,
                          "sum_aux_n_eligible": 9999, "steps_done": 50,
                          "batch_size": 256},
        sample_accumulator=None,
    )
    assert block["loss"]["aux_loss_avg_iter"] == 0.0
    assert block["loss"]["aux_target_coverage_rate"] == 0.0
    assert block["loss"]["aux_positions_seen_in_training"] == 0


def test_conversion_training_consistency_unavailable_when_disabled_even_with_sample_accumulator():
    """When enabled=False, consistency must report available=False even
    if a sample_accumulator dict is provided. Disabled conversion is a
    superset of 'consistency check is N/A'."""
    block = build_conversion_training_block(
        config=_config(effective=0.0),
        enabled=False,
        buffer_stats=_zero_buffer_stats(),
        loss_accumulator=_zero_loss_acc(),
        sample_accumulator={"eligible_drawn_total": 100,
                            "cap_was_binding_steps": 0,
                            "boost_inactive_steps": 0},
    )
    assert block["consistency"]["available"] is False
    assert block["consistency"]["drawn_vs_seen_match"] is None
    # Sample stats also zeroed (disabled path)
    assert block["sample_stats"]["eligible_drawn_total"] == 0


def test_drawn_vs_seen_match_flags_divergence():
    """ANCHOR (Spec 2 §11.3): when sampler-drawn != loss-seen, flag false
    and report exact delta."""
    block = build_conversion_training_block(
        config=_config(),
        enabled=True,
        buffer_stats=_zero_buffer_stats(),
        loss_accumulator={"sum_aux": 100.0, "sum_aux_coverage": 5.0,
                          "sum_aux_n_eligible": 1280, "steps_done": 50,
                          "batch_size": 256},
        sample_accumulator={"eligible_drawn_total": 1300,    # mismatch by +20
                            "cap_was_binding_steps": 0,
                            "boost_inactive_steps": 0},
    )
    assert block["consistency"]["available"] is True
    assert block["consistency"]["drawn_vs_seen_match"] is False
    assert block["consistency"]["drawn_minus_seen"] == 20


def test_drawn_vs_seen_match_naming_correctness():
    """Spec 2 §8.2 lock: drawn = sampler count, seen = loss count.
    NOT reversed."""
    block = build_conversion_training_block(
        config=_config(),
        enabled=True,
        buffer_stats=_zero_buffer_stats(),
        loss_accumulator={"sum_aux": 0.0, "sum_aux_coverage": 0.0,
                          "sum_aux_n_eligible": 90, "steps_done": 1,
                          "batch_size": 100},
        sample_accumulator={"eligible_drawn_total": 100,
                            "cap_was_binding_steps": 0,
                            "boost_inactive_steps": 0},
    )
    # drawn (100) - seen (90) = 10 (positive)
    assert block["consistency"]["drawn_minus_seen"] == 10


from scripts.GPU.alphazero.conversion_telemetry import (
    build_recovery_block,
    is_recovery_or_extreme_closeout_drift,
)


def _record(
    detected=True, outcome_class=1, du_moves=0, delay=0,
    state_cap=False,
):
    rec = {
        "detected": detected,
        "outcome_class": outcome_class,
        "winner_moves_with_dominant_unavailable": du_moves if outcome_class == 1 else None,
        "dominant_unavailable_moves": du_moves if outcome_class == 2 else None,
        "conversion_delay_plies": delay,
        "reason": "state_cap" if state_cap else "win",
    }
    return rec


def test_recovery_predicate_three_triggers():
    # DU clause
    rec_du = _record(du_moves=15)
    assert is_recovery_or_extreme_closeout_drift(rec_du, du_threshold=10, delay_threshold=20)
    # Delay clause
    rec_delay = _record(delay=25)
    assert is_recovery_or_extreme_closeout_drift(rec_delay, du_threshold=10, delay_threshold=20)
    # State-cap clause
    rec_cap = _record(outcome_class=2, state_cap=True)
    assert is_recovery_or_extreme_closeout_drift(rec_cap, du_threshold=10, delay_threshold=20)


def test_recovery_predicate_state_cap_after_detection_required_for_class2():
    """Class 2 with detected=False → not counted."""
    rec = _record(detected=False, outcome_class=2, state_cap=True)
    assert not is_recovery_or_extreme_closeout_drift(rec, du_threshold=10, delay_threshold=20)


def test_recovery_block_class2_dominant_unavailable_handling():
    """Spec 2 §8.4 lock: Class 2 du_moves explicitly defined; no silent zero."""
    # Class 2 with du_moves field present, NOT state_cap
    rec_class2 = _record(outcome_class=2, du_moves=15, state_cap=False)
    assert is_recovery_or_extreme_closeout_drift(rec_class2, du_threshold=10, delay_threshold=20)


def test_recovery_block_excludes_undetected_games():
    rec = _record(detected=False, du_moves=15, delay=25)
    assert not is_recovery_or_extreme_closeout_drift(rec, du_threshold=10, delay_threshold=20)


def test_recovery_block_percentiles_handcrafted():
    records = [_record(du_moves=v) for v in [0, 1, 2, 3, 4, 5, 10, 15, 20, 22]]
    block = build_recovery_block(records, du_threshold=10, delay_threshold=20)
    p = block["dominant_unavailable_moves"]
    assert 4 <= p["p50"] <= 5    # midpoint (linear interpolation gives 4.5)
    assert p["p90"] >= 15
    assert p["max"] == 22


def test_recovery_rate_denominators():
    """Spec 2 §8.3: rate = count/games_total; rate_among_detected = count/detected_games."""
    records = [
        _record(detected=True, du_moves=15),    # triggers
        _record(detected=True, du_moves=0),     # no trigger
        _record(detected=False),                # not detected
    ]
    block = build_recovery_block(records, du_threshold=10, delay_threshold=20)
    assert block["games_total"] == 3
    assert block["detected_games"] == 2
    assert block["count"] == 1
    assert block["rate"] == 1 / 3
    assert block["rate_among_detected"] == 0.5


def test_recovery_block_renamed_to_extreme_closeout_drift():
    """Spec 2 §5 lock: function names use the renamed form."""
    block = build_recovery_block([], du_threshold=10, delay_threshold=20)
    assert "version" in block
    assert "config" in block
    assert "trigger_breakdown" in block
