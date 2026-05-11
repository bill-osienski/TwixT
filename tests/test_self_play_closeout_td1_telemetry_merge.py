"""Tests for the worker-telemetry merge helper in self_play.py.

Spec 3 Fix 1 (§4.5): merge per-game/per-worker
closeout_td1_visit_forcing telemetry into a single sidecar block.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.self_play import _merge_closeout_td1_telemetry


def test_merge_sums_counters_and_recomputes_rates():
    workers = [
        {"enabled": True, "min_visits": 8, "max_forced_moves": 4,
         "require_high_value": False, "high_value_threshold": 0.95,
         "positions_triggered": 10, "positions_skipped_no_candidates": 0,
         "positions_skipped_high_value_gate": 0, "forced_sims_total": 80,
         "selected_forced_move_count": 7,
         "selected_forced_move_rate": 0.7,
         "post_force_endpoint_visit_top1_rate": 0.6,
         "post_force_endpoint_visit_top5_rate": 0.8},
        {"enabled": True, "min_visits": 8, "max_forced_moves": 4,
         "require_high_value": False, "high_value_threshold": 0.95,
         "positions_triggered": 20, "positions_skipped_no_candidates": 1,
         "positions_skipped_high_value_gate": 0, "forced_sims_total": 160,
         "selected_forced_move_count": 16,
         "selected_forced_move_rate": 0.8,
         "post_force_endpoint_visit_top1_rate": 0.75,
         "post_force_endpoint_visit_top5_rate": 0.9},
    ]
    out = _merge_closeout_td1_telemetry(workers)
    assert out["enabled"] is True
    assert out["min_visits"] == 8
    assert out["positions_triggered"] == 30
    assert out["forced_sims_total"] == 240
    assert out["selected_forced_move_count"] == 23
    assert abs(out["selected_forced_move_rate"] - (23 / 30)) < 1e-6
    # Weighted top5 = (0.8 * 10 + 0.9 * 20) / 30 = 26/30
    assert abs(out["post_force_endpoint_visit_top5_rate"] - (26 / 30)) < 1e-6


def test_merge_handles_empty_input():
    assert _merge_closeout_td1_telemetry([]) == {}


def test_merge_handles_zero_triggered():
    workers = [{"enabled": True, "positions_triggered": 0, "forced_sims_total": 0,
                "selected_forced_move_count": 0,
                "post_force_endpoint_visit_top1_rate": 0.0,
                "post_force_endpoint_visit_top5_rate": 0.0}]
    out = _merge_closeout_td1_telemetry(workers)
    assert out["positions_triggered"] == 0
    assert out["selected_forced_move_rate"] == 0.0


def test_merge_handles_mixed_shape_with_none_and_zero_workers():
    """Multiple workers with mixed activity (some zero, some non-zero) plus
    a None entry. Verifies counters sum correctly and rates are weighted
    only by workers that actually triggered."""
    workers = [
        {"enabled": True, "min_visits": 8, "max_forced_moves": 4,
         "require_high_value": False, "high_value_threshold": 0.95,
         "positions_triggered": 0, "positions_skipped_no_candidates": 2,
         "positions_skipped_high_value_gate": 0, "forced_sims_total": 0,
         "selected_forced_move_count": 0,
         "selected_forced_move_rate": 0.0,
         "post_force_endpoint_visit_top1_rate": 0.0,
         "post_force_endpoint_visit_top5_rate": 0.0},
        {"enabled": True, "min_visits": 8, "max_forced_moves": 4,
         "require_high_value": False, "high_value_threshold": 0.95,
         "positions_triggered": 5, "positions_skipped_no_candidates": 1,
         "positions_skipped_high_value_gate": 0, "forced_sims_total": 40,
         "selected_forced_move_count": 3,
         "selected_forced_move_rate": 0.6,
         "post_force_endpoint_visit_top1_rate": 0.4,
         "post_force_endpoint_visit_top5_rate": 0.8},
        None,
    ]
    out = _merge_closeout_td1_telemetry(workers)
    # Config echoes survive None / zero-only entries
    assert out["enabled"] is True
    assert out["min_visits"] == 8
    # Sums include zero-triggered worker's skip counter
    assert out["positions_triggered"] == 5
    assert out["positions_skipped_no_candidates"] == 3
    assert out["forced_sims_total"] == 40
    assert out["selected_forced_move_count"] == 3
    # Weighted rates: only the non-zero worker contributes weight
    assert abs(out["selected_forced_move_rate"] - 0.6) < 1e-6
    assert abs(out["post_force_endpoint_visit_top1_rate"] - 0.4) < 1e-6
    assert abs(out["post_force_endpoint_visit_top5_rate"] - 0.8) < 1e-6
