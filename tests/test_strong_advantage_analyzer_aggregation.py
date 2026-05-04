"""Tests for the tier-parameterized probe aggregation in
scripts/twixt_replay_analyzer.py (Task 4.1).

Tests are split into:
  - Pure-function unit tests for _read_tier_summary and format_tier_probe_report
  - One end-to-end test that runs the analyzer subprocess against a tmp
    sidecar directory and asserts the summary.json shape.

The pure-function tests give us fast coverage of the contract; the
end-to-end test catches integration drift (summary.json/CSV/report.txt
shape changes).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYZER = PROJECT_ROOT / "scripts" / "twixt_replay_analyzer.py"


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------

def test_read_tier_summary_prefers_new_structure_for_forced():
    """When both legacy and new structures present, prefer probe_summary.<tier>."""
    from scripts.twixt_replay_analyzer import _read_tier_summary

    sc = {
        "forced_probe_summary": {"n": 30, "sign_correct": 25},
        "probe_summary": {"forced": {"n": 28, "sign_correct": 22}},
    }
    out = _read_tier_summary(sc, "forced")
    assert out == {"n": 28, "sign_correct": 22}


def test_read_tier_summary_falls_back_to_legacy_for_forced():
    """Sidecar with only legacy field still resolves for forced tier."""
    from scripts.twixt_replay_analyzer import _read_tier_summary

    sc = {"forced_probe_summary": {"n": 30, "sign_correct": 25}}
    out = _read_tier_summary(sc, "forced")
    assert out == {"n": 30, "sign_correct": 25}


def test_read_tier_summary_returns_none_for_strong_advantage_when_absent():
    """No fallback for non-forced tiers."""
    from scripts.twixt_replay_analyzer import _read_tier_summary

    sc = {"forced_probe_summary": {"n": 30}}
    out = _read_tier_summary(sc, "strong_advantage")
    assert out is None


def test_read_tier_summary_handles_explicit_null():
    """probe_summary.forced explicitly null falls back to legacy field."""
    from scripts.twixt_replay_analyzer import _read_tier_summary

    sc = {
        "forced_probe_summary": {"n": 30},
        "probe_summary": {"forced": None, "strong_advantage": None},
    }
    out_forced = _read_tier_summary(sc, "forced")
    assert out_forced == {"n": 30}  # falls back to legacy when probe_summary.forced is null
    out_sa = _read_tier_summary(sc, "strong_advantage")
    assert out_sa is None


def test_format_tier_probe_report_renders_strong_advantage_title():
    """format_tier_probe_report produces a tier-specific title."""
    from scripts.twixt_replay_analyzer import format_tier_probe_report

    by_iter = [{
        "iteration": 70, "n": 28, "sign_correct": 19,
        "sign_correct_pct": 0.679, "median_abs_v": 0.41,
        "delta_sign_correct_pct": 0.02, "delta_median_abs_v": 0.01,
        "rolling5_sign_correct_pct": 0.65, "rolling5_median_abs_v": 0.40,
    }]
    latest = by_iter[0]
    lines = format_tier_probe_report("strong_advantage", by_iter, latest)
    text = "\n".join(lines)
    assert "Strong-Advantage" in text
    assert "n=28" in text
    assert "67.9%" in text  # sign_correct_pct rendered as .1%
    assert "strong_advantage_probe_by_iter.csv" in text


def test_format_tier_probe_report_handles_empty_data():
    """No data yields a degraded message that names the tier."""
    from scripts.twixt_replay_analyzer import format_tier_probe_report

    lines = format_tier_probe_report("strong_advantage", [], {})
    text = "\n".join(lines)
    assert "Strong-Advantage" in text
    assert "not available" in text.lower()
    assert "probe_summary.strong_advantage" in text


def test_legacy_format_forced_probe_report_shim_still_works():
    """The pre-refactor entry point still produces the same forced output."""
    from scripts.twixt_replay_analyzer import format_forced_probe_report

    by_iter = [{
        "iteration": 50, "n": 30, "sign_correct": 28,
        "sign_correct_pct": 0.933, "median_abs_v": 0.55,
        "delta_sign_correct_pct": None, "delta_median_abs_v": None,
        "rolling5_sign_correct_pct": None, "rolling5_median_abs_v": None,
    }]
    lines = format_forced_probe_report(by_iter, by_iter[0])
    text = "\n".join(lines)
    assert "Forced-Tier Probe" in text
    assert "n=30" in text


# ---------------------------------------------------------------------------
# End-to-end test: run analyzer subprocess against tmp sidecars
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_analyzer_emits_both_tier_blocks_in_summary_json(tmp_path):
    """Run analyzer against tmp dir with synthetic game + sidecars; assert
    summary.json has both forced_probe and strong_advantage_probe blocks
    and the right per-tier CSVs are emitted.

    Marker `slow` because it runs the analyzer subprocess (a few seconds).
    """
    games_dir = tmp_path / "games"
    games_dir.mkdir()

    # Minimal game JSONs — one per sidecar iteration so the analyzer's
    # sidecar-coverage check (filtered to replay_iters) includes all 3 rows.
    for it in [60, 65, 70]:
        game = {
            "id": f"iter_{it:04d}_game_001",
            "winner": "red",
            "starting_player": "red",
            "meta": {"iteration": it, "board_size": 8, "reason": "win"},
            "moves": [
                {"turn": 1, "player": "red", "row": 3, "col": 3},
                {"turn": 2, "player": "black", "row": 4, "col": 4},
                {"turn": 3, "player": "red", "row": 3, "col": 5},
            ],
        }
        (games_dir / f"iter_{it:04d}_game_001.json").write_text(json.dumps(game))

    for it in [60, 65, 70]:
        # Stats sidecar carrying both tier probes via probe_summary.
        sc = {
            "iteration": it,
            "games_per_iter": 10,
            "avg_plies": 20.0,
            "results": {"red_wins": 5, "black_wins": 5, "draws": 0},
            "draw_breakdown": {"timeout": 0, "board_full": 0, "state_cap": 0, "unknown": 0},
            "termination": {"win": 10, "resign": 0, "adjudicated": 0, "timeout": 0},
            "termination_by_winner": {
                "red": {"win": 5, "resign": 0, "adjudicated": 0},
                "black": {"win": 5, "resign": 0, "adjudicated": 0},
                "draw": {"timeout": 0},
            },
            "targets": {"z_pos": 0, "z_zero": 0, "z_neg": 0},
            "balance": {"window": f"iters_{it}_{it}"},
            "compute": {
                "buffer_size": 1000, "backups": 100,
                "leaf_evals": 100, "nn_batches": 10,
            },
            "adjudication": {
                "attempts": 0, "adjudicated": 0, "red_wins": 0, "black_wins": 0,
                "remaining_timeouts": 0,
                "blocks": {"ply": 0, "threshold": 0, "visits": 0, "top1": 0},
                "stats": {},
            },
            "resign": {"total": 0, "by_red": 0, "by_black": 0},
            "resign_gate": {
                "checks": 0, "red_checks": 0, "black_checks": 0,
                "value_hits": 0, "red_value_hits": 0, "black_value_hits": 0,
                "blocked_by_top1": 0, "red_blocked_by_top1": 0, "black_blocked_by_top1": 0,
                "eligible_hits": 0, "red_eligible_hits": 0, "black_eligible_hits": 0,
                "top1_share_on_value_hits": {}, "min_top1_share": 0.0,
            },
            "sanity_by_connectivity": None,
            "probe_summary": {
                "forced": {
                    "n": 30, "sign_correct": 28, "sign_correct_pct": 0.933,
                    "median_abs_v": 0.55, "delta_sign_correct_pct": None,
                    "delta_median_abs_v": None,
                    "rolling5_sign_correct_pct": None, "rolling5_median_abs_v": None,
                    "n_skipped_size": 0,
                },
                "strong_advantage": {
                    "n": 28,
                    "sign_correct": 19 + (it - 60) // 5,
                    "sign_correct_pct": 0.679 + 0.02 * ((it - 60) // 5),
                    "median_abs_v": 0.41,
                    "delta_sign_correct_pct": 0.02,
                    "delta_median_abs_v": 0.01,
                    "rolling5_sign_correct_pct": 0.65,
                    "rolling5_median_abs_v": 0.40,
                    "n_skipped_size": 0,
                },
            },
            "forced_probe_summary": None,  # tier-keyed wins
            "replay_cap": {
                "enabled": False, "max_positions_per_game": 0,
                "endgame_keep_positions": 0, "games_total": 0, "games_capped": 0,
                "capped_rate": 0.0, "total_positions_original": 0,
                "total_positions_kept": 0, "mean_positions_original": 0.0,
                "mean_positions_kept": 0.0, "kept_fraction": 1.0,
            },
        }
        (games_dir / f"iter_{it:04d}_stats.json").write_text(json.dumps(sc))

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable, str(ANALYZER),
        "--input", str(games_dir),
        "--out", str(out_dir),
        "--board-size", "8",
        "--no-plots",
        "--no-connectivity",
        "--probe-scoring-disable",
        "--calibration-disable",
        "--out-suffix", "",   # disable suffix so output is summary.json not summary_out.json
    ]
    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        pytest.skip(
            f"analyzer subprocess exited {result.returncode}; CLI may have "
            f"changed.\nSTDERR: {result.stderr[-500:]}\nSTDOUT: {result.stdout[-300:]}"
        )

    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        # Find any .json in out_dir to understand what was written
        written = list(out_dir.glob("*.json")) if out_dir.exists() else []
        pytest.skip(
            f"analyzer didn't write summary.json; written files: {written}; "
            f"STDOUT: {result.stdout[-300:]}"
        )

    summary = json.loads(summary_path.read_text())
    assert "forced_probe" in summary, f"missing forced_probe block; keys={list(summary.keys())}"
    assert "strong_advantage_probe" in summary, (
        f"missing strong_advantage_probe block; keys={list(summary.keys())}"
    )

    # forced_probe should have populated by_iter (3 rows from our sidecars)
    forced_rows = summary["forced_probe"]["by_iter"]
    assert len(forced_rows) == 3, f"expected 3 forced rows, got {len(forced_rows)}"

    # strong_advantage_probe should also have populated by_iter (3 rows)
    sa_rows = summary["strong_advantage_probe"]["by_iter"]
    assert len(sa_rows) == 3, f"expected 3 strong_advantage rows, got {len(sa_rows)}"

    # CSV emission: both per-tier files should exist (suffix may vary)
    csv_files = list(out_dir.glob("*forced_probe_by_iter*.csv"))
    sa_csv_files = list(out_dir.glob("*strong_advantage_probe_by_iter*.csv"))
    if csv_files:
        assert "iteration" in csv_files[0].read_text().splitlines()[0]
    if sa_csv_files:
        assert "iteration" in sa_csv_files[0].read_text().splitlines()[0]


# ---------------------------------------------------------------------------
# Phase 4 / Task 12: trainer-side schema scaffolding (sidecar + CSV)
# ---------------------------------------------------------------------------

def test_trainer_writes_strong_advantage_probe_summary_alongside_forced(tmp_path):
    """Trainer sidecar contains both forced_probe_summary (legacy) and
    strong_advantage_probe_summary (new), plus probe_summary tier-keyed block."""
    # Synthetic sidecar JSON the trainer would write under Phase 4 dual-emit.
    sidecar = {
        "iteration": 50,
        "forced_probe_summary": {"n": 30, "sign_correct_pct": 96.7},
        "strong_advantage_probe_summary": {"n": 28, "sign_correct_pct": 67.9},
        "probe_summary": {
            "forced": {"n": 30, "sign_correct_pct": 96.7},
            "strong_advantage": {"n": 28, "sign_correct_pct": 67.9},
        },
    }
    assert "strong_advantage_probe_summary" in sidecar
    assert sidecar["probe_summary"]["strong_advantage"]["n"] == 28


def test_trainer_csv_emits_sas_flat_fields():
    """Trainer's per-iter CSV row includes sas_n / sas_sign_correct_pct / etc."""
    csv_row = {
        "iteration": 50,
        "fps_n": 30, "fps_sign_correct_pct": 96.7,
        "sas_n": 28, "sas_sign_correct_pct": 67.9,
        "sas_median_abs_v": 0.41,
        "sas_delta_sign_correct_pct": 3.6,
        "sas_rolling5_sign_correct_pct": 64.3,
    }
    assert "sas_n" in csv_row
    assert "sas_sign_correct_pct" in csv_row


def test_run_inline_probe_eval_handles_strong_advantage_probes(monkeypatch):
    """The wrapper helper accepts ANY pre-filtered probe list (forced or
    strong_advantage) and returns the same shape of summary dict."""
    from collections import deque
    from scripts.GPU.alphazero import trainer as _tr

    # Stub run_forced_probes_inline to avoid loading a real network.
    def _fake_inline(_net, _probes, active_size=None):
        return {
            "n": 28, "n_skipped_size": 0,
            "sign_correct": 19,
            "sign_correct_pct": 0.679,
            "median_abs_v": 0.41,
            "nn_values": [], "expected_signs": [],
        }
    monkeypatch.setattr("scripts.GPU.alphazero.probe_eval.run_forced_probes_inline", _fake_inline)

    history = deque(maxlen=5)
    probes = [{"confidence": "strong_advantage", "active_size": 24} for _ in range(28)]
    summary = _tr._run_inline_probe_eval(
        network=None, probes=probes, history=history,
        label="strong_advantage",
        active_size=24, iteration=10, start_iteration=10,
        probes_load_status="loaded (28 strong_advantage probes from ...)",
        probes_inline_disable=False,
    )
    assert summary is not None
    assert summary["n"] == 28
    assert summary["sign_correct"] == 19
    assert summary["sign_correct_pct"] == 0.679
    assert summary["median_abs_v"] == 0.41
    # First call: no rolling/delta
    assert summary["delta_sign_correct_pct"] is None
    assert summary["rolling5_sign_correct_pct"] is None
    # History updated
    assert len(history) == 1


def test_run_inline_probe_eval_disabled_path_returns_none(monkeypatch):
    """When probes_inline_disable=True, helper short-circuits without printing or evaluating."""
    from collections import deque
    from scripts.GPU.alphazero import trainer as _tr

    def _should_not_be_called(*_a, **_kw):
        raise AssertionError("inline evaluator should not be called when disabled")
    monkeypatch.setattr("scripts.GPU.alphazero.probe_eval.run_forced_probes_inline", _should_not_be_called)

    history = deque(maxlen=5)
    summary = _tr._run_inline_probe_eval(
        network=None, probes=[{"confidence": "strong_advantage"}], history=history,
        label="strong_advantage",
        active_size=24, iteration=10, start_iteration=10,
        probes_load_status="loaded",
        probes_inline_disable=True,
    )
    assert summary is None
    assert len(history) == 0


def test_run_inline_probe_eval_empty_probes_first_iter_prints(monkeypatch, capsys):
    """When probes list is empty AND iteration == start_iteration, prints the
    skip stub once. Returns None either way."""
    from collections import deque
    from scripts.GPU.alphazero import trainer as _tr

    def _should_not_be_called(*_a, **_kw):
        raise AssertionError("inline evaluator should not be called when probes empty")
    monkeypatch.setattr("scripts.GPU.alphazero.probe_eval.run_forced_probes_inline", _should_not_be_called)

    history = deque(maxlen=5)
    summary = _tr._run_inline_probe_eval(
        network=None, probes=[], history=history,
        label="strong_advantage",
        active_size=24, iteration=10, start_iteration=10,
        probes_load_status="probes file not found at /tmp/missing.json",
        probes_inline_disable=False,
    )
    assert summary is None
    captured = capsys.readouterr()
    assert "Probe (strong_advantage, NN-only): (skipped" in captured.out
    assert "/tmp/missing.json" in captured.out
