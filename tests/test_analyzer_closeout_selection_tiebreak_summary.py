"""Tests for Fix 2 telemetry aggregation."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import (
    aggregate_closeout_selection_tiebreak,
    format_closeout_selection_tiebreak_report,
)


def test_aggregator_sums_overrides():
    sidecars = {
        150: {"closeout_selection_tiebreak": {
            "enabled": True, "eligible_positions": 100,
            "overrides": 20, "override_to_endpoint": 15, "override_to_reducer": 5,
            "would_have_selected_redundant": 12, "would_have_selected_off_chain": 6,
            "would_have_selected_other": 2,
        }},
    }
    s = aggregate_closeout_selection_tiebreak(sidecars)
    assert s["eligible_positions"] == 100
    assert s["overrides"] == 20
    assert s["override_rate"] == 0.2


def test_format_emits_section():
    summary = {"iters_covered": [150], "enabled": True,
               "eligible_positions": 50, "overrides": 10, "override_rate": 0.2,
               "override_to_endpoint": 8, "override_to_reducer": 2,
               "would_have_selected_redundant": 6, "would_have_selected_off_chain": 4,
               "would_have_selected_other": 0}
    body = "\n".join(format_closeout_selection_tiebreak_report(summary))
    assert "Closeout selection tie-break" in body
    assert "Overrides: 10" in body
