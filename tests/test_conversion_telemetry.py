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
