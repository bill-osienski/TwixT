"""Tests for Fix 1 telemetry aggregation in the analyzer (spec §1.2)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import (
    aggregate_closeout_td1_visit_forcing,
    format_closeout_td1_visit_forcing_report,
)


def test_aggregator_sums_across_iterations():
    sidecars = {
        140: {"closeout_td1_visit_forcing": {
            "enabled": True, "min_visits": 8, "max_forced_moves": 4,
            "positions_triggered": 100, "forced_sims_total": 500,
            "selected_forced_move_count": 70,
            "post_force_endpoint_visit_top5_rate": 0.85,
            "post_force_endpoint_visit_top1_rate": 0.72,
        }},
        141: {"closeout_td1_visit_forcing": {
            "enabled": True, "min_visits": 8, "max_forced_moves": 4,
            "positions_triggered": 80, "forced_sims_total": 400,
            "selected_forced_move_count": 60,
            "post_force_endpoint_visit_top5_rate": 0.9,
            "post_force_endpoint_visit_top1_rate": 0.8,
        }},
    }
    s = aggregate_closeout_td1_visit_forcing(sidecars)
    assert s["enabled"] is True
    assert s["positions_triggered_total"] == 180
    assert s["forced_sims_total"] == 900
    assert abs(s["selected_forced_move_rate"] - (130 / 180)) < 1e-6
    # weighted rates
    expected_top5 = (0.85 * 100 + 0.9 * 80) / 180
    assert abs(s["post_force_endpoint_visit_top5_rate"] - expected_top5) < 1e-6


def test_report_formatter_emits_section():
    summary = {
        "enabled": True, "min_visits": 8, "max_forced_moves": 4,
        "iters_covered": [140, 141, 142],
        "positions_triggered_total": 180, "forced_sims_total": 900,
        "selected_forced_move_count": 130, "selected_forced_move_rate": 0.722,
        "post_force_endpoint_visit_top1_rate": 0.75,
        "post_force_endpoint_visit_top5_rate": 0.875,
    }
    lines = format_closeout_td1_visit_forcing_report(summary)
    body = "\n".join(lines)
    assert "Closeout td=1 visit forcing" in body
    assert "Iters covered" in body
    assert "180" in body
