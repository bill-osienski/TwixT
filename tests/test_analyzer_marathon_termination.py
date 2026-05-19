"""Analyzer-side integration tests for marathon-termination diagnostics."""
import sys, csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import (
    write_marathon_termination_csv,
    format_marathon_termination_report,
)
from scripts.GPU.alphazero.marathon_termination_diagnostics import (
    aggregate_marathon_termination,
)


def _per_game(iteration, *, reason="win", n_moves=80, winner="red",
              adj_block=None, diagnostics=None):
    record = {
        "iteration": iteration, "game_idx": 0,
        "winner": winner, "reason": reason, "n_moves": n_moves,
        "first_total_goal_distance": 2,
        "winner_moves_with_dominant_unavailable": 0,
        "conversion_delay_plies": 0,
    }
    meta = {"reason": reason, "n_moves": n_moves}
    if adj_block is not None:
        meta["adjudication_block_reason"] = adj_block
    return record, meta, (diagnostics or [])


def _cfg():
    return dict(resign_threshold=-0.945, resign_min_ply=80,
                resign_min_visits=200, resign_min_top1_share=0.102)


def test_analyzer_writes_marathon_termination_csv(tmp_path):
    """Spec §7 test. CSV emits per-iter rows + a range-total row at iteration=-1."""
    games = [
        _per_game(220, reason="state_cap", n_moves=280, winner=None, adj_block="top1"),
        _per_game(221, reason="state_cap", n_moves=280, winner=None, adj_block="threshold"),
    ]
    agg = aggregate_marathon_termination(games, **_cfg())
    out = tmp_path / "marathon.csv"
    write_marathon_termination_csv(str(out), agg)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 3  # 220, 221, -1
    iters = sorted(int(r["iteration"]) for r in rows)
    assert iters == [-1, 220, 221]
    # Range-total row's adjudication_gate_min_top1_share == 1, value_below_threshold == 1.
    rt = next(r for r in rows if int(r["iteration"]) == -1)
    assert int(rt["adjudication_gate_min_top1_share"]) == 1
    assert int(rt["adjudication_gate_value_below_threshold"]) == 1


def test_analyzer_report_includes_marathon_section_with_decision_suggestion():
    """Spec §7 test. Report section header + decision-rule line are rendered."""
    games = [
        _per_game(220, reason="state_cap", n_moves=280, winner=None, adj_block="top1")
        for _ in range(10)
    ]
    agg = aggregate_marathon_termination(games, **_cfg())
    lines = format_marathon_termination_report(agg, range_label="220-229")
    text = "\n".join(lines)
    assert "Marathon termination diagnostics (220-229)" in text
    assert "state_cap 280-ply games: 10" in text
    assert "adjudication gate blocked by:" in text
    assert "Suggested termination action:" in text
    # 10/10 blocked by min_top1_share → suggestion mentions adjudicate-min-top1-share.
    assert "adjudicate-min-top1-share" in text


def test_format_marathon_termination_report_neutral_when_no_signal_dominates():
    """When no remedy clearly dominates, suggestion line says 'no dominant remedy'."""
    # Single non-state-cap game with no resign-gate-block signal.
    games = [_per_game(220, reason="win", n_moves=80, winner="red")]
    agg = aggregate_marathon_termination(games, **_cfg())
    lines = format_marathon_termination_report(agg, range_label="220-229")
    text = "\n".join(lines)
    assert "no dominant remedy" in text


def test_report_emits_observability_warning_when_diagnostics_sparse(tmp_path):
    """Spec §3.1 observability follow-up. If less than 50% of games have
    >=15 own-entries on either side, the report surfaces a WARNING line
    so a zero no-progress rate isn't confused with a no-data state."""
    # Diagnostics are short (5 entries / game) — won't reach 15 own-entries.
    short_diag = [
        {"ply": p, "side_to_move": "red" if i % 2 == 0 else "black",
         "selected_move_classification": {"primary_class": "redundant_reinforcement"}}
        for i, p in enumerate(range(50, 55))
    ]
    games = [_per_game(220, reason="win", n_moves=80, winner="red", diagnostics=short_diag)
             for _ in range(10)]
    agg = aggregate_marathon_termination(games, **_cfg())
    lines = format_marathon_termination_report(agg, range_label="220-229")
    text = "\n".join(lines)
    assert "WARNING" in text
    assert "observability" in text


def test_csv_includes_observability_columns(tmp_path):
    """Observability counters present per row + range-total row."""
    games = [_per_game(220, reason="win", n_moves=80, winner="red")]
    agg = aggregate_marathon_termination(games, **_cfg())
    out = tmp_path / "m.csv"
    write_marathon_termination_csv(str(out), agg)
    rows = list(csv.DictReader(open(out)))
    for r in rows:
        assert "diagnostics_entries_red" in r
        assert "diagnostics_entries_black" in r
        assert "no_progress_observable_games_red" in r
        assert "no_progress_observable_games_black" in r
