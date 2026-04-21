"""Tests for the analyzer's Phase 2 sidecar field propagation.

Verifies:
- aggregate_sidecars collects per-iter sanity_by_connectivity + forced_probe_summary
- The per-iter rows and latest snapshot are shaped as expected
- Absent sidecar blocks degrade gracefully
- format_sanity_by_connectivity_report renders without error (with and without data)
- format_forced_probe_report renders without error (with and without data)
"""
from __future__ import annotations

from scripts.twixt_replay_analyzer import (
    aggregate_sidecars,
    format_sanity_by_connectivity_report,
    format_forced_probe_report,
)


def _make_sidecar(iter_num, sbc=None, fps=None):
    """Minimal sidecar dict matching the trainer's _sidecar shape."""
    return {
        "iteration": iter_num,
        "games_per_iter": 100,
        "results": {"red_wins": 40, "black_wins": 50, "draws": 10},
        "draw_breakdown": {"timeout": 0, "board_full": 10, "state_cap": 0, "unknown": 0},
        "termination": {"win": 90, "resign": 0, "adjudicated": 0, "timeout": 0},
        "termination_by_winner": {
            "red": {"win": 40, "resign": 0, "adjudicated": 0},
            "black": {"win": 50, "resign": 0, "adjudicated": 0},
            "draw": {"timeout": 0},
        },
        "targets": {"z_pos": 50, "z_zero": 10, "z_neg": 40},
        "avg_plies": 20.0 + iter_num,  # vary by iter
        "balance": {},
        "resign": {"total": 0, "by_red": 0, "by_black": 0},
        "adjudication": {},
        "resign_gate": {},
        "compute": {},
        "sanity_by_connectivity": sbc,
        "forced_probe_summary": fps,
    }


def _make_sbc(winning_n, winning_sa, winning_mv, no_winning_n, no_winning_sa, no_winning_mv):
    return {
        "winning_structure": {
            "n": winning_n, "sign_agree": winning_sa, "median_abs_v": winning_mv,
        },
        "no_winning_structure": {
            "n": no_winning_n, "sign_agree": no_winning_sa, "median_abs_v": no_winning_mv,
        },
        "winning_size_threshold": 8,
    }


def _make_fps(n, sc, sc_pct, mv, delta_pct=None, r5_pct=None, r5_mv=None):
    return {
        "n": n, "n_skipped_size": 0,
        "sign_correct": sc, "sign_correct_pct": sc_pct,
        "median_abs_v": mv,
        "delta_sign_correct_pct": delta_pct, "delta_median_abs_v": None,
        "rolling5_sign_correct_pct": r5_pct, "rolling5_median_abs_v": r5_mv,
    }


# ---------- aggregate_sidecars with Phase 2 fields ----------

def test_aggregate_collects_sanity_by_connectivity_per_iter():
    """Every iter with a sanity_by_connectivity block gets a row in by_iter."""
    sidecars = {
        5: _make_sidecar(5, sbc=_make_sbc(50, 0.80, 0.60, 200, 0.70, 0.30)),
        6: _make_sidecar(6, sbc=_make_sbc(60, 0.88, 0.74, 195, 0.78, 0.45)),
        7: _make_sidecar(7, sbc=_make_sbc(70, 0.91, 0.78, 190, 0.82, 0.52)),
    }
    agg = aggregate_sidecars(sidecars)
    rows = agg["sanity_by_connectivity_by_iter"]
    assert len(rows) == 3
    assert [r["iteration"] for r in rows] == [5, 6, 7]
    assert rows[-1]["winning_sign_agree"] == 0.91
    assert rows[-1]["winning_median_abs_v"] == 0.78
    assert rows[-1]["no_winning_n"] == 190


def test_aggregate_sanity_by_connectivity_latest_snapshot():
    """Latest iter's dict is captured verbatim in sanity_by_connectivity_latest."""
    sidecars = {
        5: _make_sidecar(5, sbc=_make_sbc(50, 0.80, 0.60, 200, 0.70, 0.30)),
        6: _make_sidecar(6, sbc=_make_sbc(60, 0.88, 0.74, 195, 0.78, 0.45)),
    }
    agg = aggregate_sidecars(sidecars)
    latest = agg["sanity_by_connectivity_latest"]
    assert latest["winning_structure"]["sign_agree"] == 0.88
    assert latest["winning_size_threshold"] == 8


def test_aggregate_collects_forced_probe_per_iter():
    """Every iter with a forced_probe_summary block gets a row in by_iter."""
    sidecars = {
        5: _make_sidecar(5, fps=_make_fps(24, 18, 0.75, 0.60)),
        6: _make_sidecar(6, fps=_make_fps(24, 20, 0.83, 0.68, delta_pct=0.08, r5_pct=0.77, r5_mv=0.63)),
    }
    agg = aggregate_sidecars(sidecars)
    rows = agg["forced_probe_by_iter"]
    assert len(rows) == 2
    assert rows[-1]["sign_correct"] == 20
    assert rows[-1]["sign_correct_pct"] == 0.83
    assert rows[-1]["delta_sign_correct_pct"] == 0.08
    assert rows[-1]["rolling5_sign_correct_pct"] == 0.77


def test_aggregate_skips_sidecars_without_phase2_blocks():
    """Absent sanity_by_connectivity / forced_probe_summary → empty lists."""
    sidecars = {
        5: _make_sidecar(5, sbc=None, fps=None),
        6: _make_sidecar(6, sbc=None, fps=None),
    }
    agg = aggregate_sidecars(sidecars)
    assert agg["sanity_by_connectivity_by_iter"] == []
    assert agg["forced_probe_by_iter"] == []
    assert agg["sanity_by_connectivity_latest"] == {}
    assert agg["forced_probe_latest"] == {}


def test_aggregate_mixed_phase1_and_phase2_sidecars():
    """Some iters with blocks, some without — only the present ones contribute rows."""
    sidecars = {
        5: _make_sidecar(5, sbc=None, fps=None),
        6: _make_sidecar(6, sbc=_make_sbc(60, 0.88, 0.74, 195, 0.78, 0.45), fps=None),
        7: _make_sidecar(7, sbc=_make_sbc(70, 0.91, 0.78, 190, 0.82, 0.52),
                         fps=_make_fps(24, 19, 0.79, 0.65)),
    }
    agg = aggregate_sidecars(sidecars)
    sbc_rows = agg["sanity_by_connectivity_by_iter"]
    fps_rows = agg["forced_probe_by_iter"]
    assert len(sbc_rows) == 2
    assert [r["iteration"] for r in sbc_rows] == [6, 7]
    assert len(fps_rows) == 1
    assert fps_rows[0]["iteration"] == 7


# ---------- Report formatters degrade gracefully ----------

def test_format_sanity_by_connectivity_with_data():
    """Renders a section header + trend table when data is present."""
    by_iter = [
        {"iteration": 5, "winning_n": 50, "winning_sign_agree": 0.80,
         "winning_median_abs_v": 0.60, "no_winning_n": 200,
         "no_winning_sign_agree": 0.70, "no_winning_median_abs_v": 0.30,
         "winning_size_threshold": 8},
    ]
    latest = _make_sbc(50, 0.80, 0.60, 200, 0.70, 0.30)
    lines = format_sanity_by_connectivity_report(by_iter, latest)
    text = "\n".join(lines)
    assert "Value Head Sanity by Connectivity Bucket" in text
    assert "winning_structure" in text
    assert "sanity_by_connectivity_by_iter.csv" in text


def test_format_sanity_by_connectivity_without_data():
    """Stubs gracefully when no data."""
    lines = format_sanity_by_connectivity_report([], {})
    text = "\n".join(lines)
    assert "Value Head Sanity by Connectivity Bucket" in text
    assert "not available" in text


def test_format_forced_probe_with_data():
    """Renders section with trend when data is present."""
    by_iter = [
        {"iteration": 5, "n": 24, "sign_correct": 18, "sign_correct_pct": 0.75,
         "median_abs_v": 0.60, "delta_sign_correct_pct": None,
         "rolling5_sign_correct_pct": None, "rolling5_median_abs_v": None},
    ]
    latest = _make_fps(24, 18, 0.75, 0.60)
    lines = format_forced_probe_report(by_iter, latest)
    text = "\n".join(lines)
    assert "Forced-Tier Probe Sign-Agree" in text
    assert "forced_probe_by_iter.csv" in text


def test_format_forced_probe_without_data():
    """Stubs gracefully when no data (probes file absent or inline disabled)."""
    lines = format_forced_probe_report([], {})
    text = "\n".join(lines)
    assert "Forced-Tier Probe Sign-Agree" in text
    assert "not available" in text
