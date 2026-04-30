"""Tests for aggregate_per_game_stats and format_per_game_stats_report.

Spec: docs/superpowers/specs/2026-04-29-analyzer-per-game-stats-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest  # for pytest.approx in tests with non-trivial float arithmetic

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Sentinel distinguishes "field is absent from meta" (old schema) from
# "field is present and explicitly null" (e.g., worker_id=None for in-process).
_OMIT = object()


def _make_replay(
    *,
    n_moves=100,
    reason="win",
    worker_id=_OMIT,
    wall_time_s=_OMIT,
    final_root_value=_OMIT,
    final_top1_share=_OMIT,
    leaf_evals=_OMIT,
    backups=_OMIT,
    nn_batches=_OMIT,
    include_compute=True,
    omit_n_moves=False,
    omit_reason=False,
    omit_meta=False,
):
    """Construct a minimal replay record for aggregate_per_game_stats tests.

    Sentinel _OMIT means the meta key is absent (old-schema). Passing
    `None` means the key is present and explicitly null. Passing a value
    means the key is present with that value.
    """
    if omit_meta:
        return {}
    meta = {}
    if not omit_n_moves:
        meta["n_moves"] = n_moves
    if not omit_reason:
        meta["reason"] = reason
    if worker_id is not _OMIT:
        meta["worker_id"] = worker_id
    if wall_time_s is not _OMIT:
        meta["wall_time_s"] = wall_time_s
    if final_root_value is not _OMIT:
        meta["final_root_value"] = final_root_value
    if final_top1_share is not _OMIT:
        meta["final_top1_share"] = final_top1_share
    if include_compute:
        compute = {}
        if leaf_evals is not _OMIT:
            compute["leaf_evals"] = leaf_evals
        if backups is not _OMIT:
            compute["backups"] = backups
        if nn_batches is not _OMIT:
            compute["nn_batches"] = nn_batches
        if compute:
            meta["compute"] = compute
    return {"meta": meta}


# -------------------------------------------------------------------------
# Test 1: empty replays
# -------------------------------------------------------------------------

def test_aggregate_returns_zero_coverage_for_empty_replays():
    """aggregate_per_game_stats([]) returns the documented zero-coverage shape."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    out = aggregate_per_game_stats([])

    assert out["n_games_total"] == 0
    assert out["n_games_with_any_stats"] == 0
    # Coverage map present with all zeros, including pre-existing fields
    cov = out["coverage"]
    for key in ("wall_time_s", "worker_id", "final_root_value", "final_top1_share",
                "compute.leaf_evals", "compute.backups", "compute.nn_batches",
                "n_moves", "reason"):
        assert cov[key] == 0, f"coverage[{key!r}] should be 0"
    # Distribution blocks all null
    assert out["game_length"] is None
    assert out["wall_time_s"] is None
    assert out["final_root_value"] is None
    assert out["final_top1_share"] is None
    assert out["compute_per_game"] is None
    # Outcomes always present, all zero
    assert out["outcomes"] == {"decisive": 0, "resign": 0, "adjudicated": 0,
                                "timeout": 0, "draw_other": 0}
    # Worker balance shape
    wb = out["worker_balance"]
    assert wb["by_worker"] == {}
    assert wb["in_process_count"] == 0
    assert wb["max_min_wall_time_ratio"] is None
    assert wb["max_min_games_ratio"] is None
    assert wb["wall_time_cv"] is None


# -------------------------------------------------------------------------
# Test 2: old-schema only (no persistence-era fields)
# -------------------------------------------------------------------------

def test_aggregate_returns_zero_coverage_for_old_schema_only():
    """Replays without any persistence fields → all persistence blocks null,
    but game_length and outcomes still populated from pre-existing fields."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        _make_replay(n_moves=100, reason="win"),
        _make_replay(n_moves=120, reason="resign"),
        _make_replay(n_moves=80,  reason="timeout_selfplay"),
    ]
    out = aggregate_per_game_stats(replays)

    assert out["n_games_total"] == 3
    assert out["n_games_with_any_stats"] == 0
    # Persistence-era coverage all zero
    for key in ("wall_time_s", "worker_id", "final_root_value", "final_top1_share",
                "compute.leaf_evals", "compute.backups", "compute.nn_batches"):
        assert out["coverage"][key] == 0
    # Pre-existing fields fully covered
    assert out["coverage"]["n_moves"] == 3
    assert out["coverage"]["reason"] == 3
    # Persistence blocks all null
    assert out["wall_time_s"] is None
    assert out["final_root_value"] is None
    assert out["final_top1_share"] is None
    assert out["compute_per_game"] is None
    # game_length and outcomes populated
    assert out["game_length"] is not None
    assert out["game_length"]["max"] == 120
    assert out["game_length"]["min"] == 80
    assert out["outcomes"]["decisive"] == 1
    assert out["outcomes"]["resign"] == 1
    assert out["outcomes"]["timeout"] == 1


# -------------------------------------------------------------------------
# Test 9: outcomes categorize meta.reason correctly
# -------------------------------------------------------------------------

def test_aggregate_outcomes_categorizes_meta_reason():
    """meta.reason values map to the five outcome categories per spec §4.

    timeout and timeout_selfplay both → outcomes.timeout
    Unrecognized reasons → outcomes.draw_other
    """
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        _make_replay(reason="win"),                # decisive
        _make_replay(reason="win"),                # decisive
        _make_replay(reason="resign"),             # resign
        _make_replay(reason="adjudicated"),        # adjudicated
        _make_replay(reason="timeout"),            # timeout
        _make_replay(reason="timeout_selfplay"),   # timeout
        _make_replay(reason="board_full"),         # draw_other
        _make_replay(reason="state_cap"),          # draw_other
        _make_replay(reason="unknown"),            # draw_other
        _make_replay(reason="something_weird"),    # draw_other (unrecognized)
    ]
    out = aggregate_per_game_stats(replays)

    assert out["outcomes"]["decisive"]    == 2
    assert out["outcomes"]["resign"]      == 1
    assert out["outcomes"]["adjudicated"] == 1
    assert out["outcomes"]["timeout"]     == 2
    assert out["outcomes"]["draw_other"]  == 4
    # Counts sum to n_games_total (mutually exclusive categories invariant)
    assert sum(out["outcomes"].values()) == out["n_games_total"] == 10


# -------------------------------------------------------------------------
# Test 10: game_length uses meta.n_moves and computes percentiles
# -------------------------------------------------------------------------

def test_aggregate_game_length_uses_meta_n_moves():
    """game_length stats computed from meta.n_moves; percentiles correct."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    # Use 10 evenly-spaced values so percentiles are easy to check.
    n_moves_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    replays = [_make_replay(n_moves=n) for n in n_moves_values]
    out = aggregate_per_game_stats(replays)

    gl = out["game_length"]
    assert gl["min"] == 10
    assert gl["max"] == 100
    assert gl["mean"] == 55.0
    # numpy.percentile linear interpolation on this set:
    # p50 = 55, p90 = 91, p95 = 95.5, p99 = 99.1
    assert gl["p50"] == 55.0
    assert gl["p90"] == 91.0
    assert abs(gl["p95"] - 95.5) < 1e-9
    assert abs(gl["p99"] - 99.1) < 1e-9
    # coverage reflects all 10 replays carried n_moves
    assert out["coverage"]["n_moves"] == 10


# -------------------------------------------------------------------------
# Test 3: full coverage populates all blocks
# -------------------------------------------------------------------------

def test_aggregate_full_coverage_populates_all_blocks():
    """5 replays, every persistence-era field populated → all blocks non-null."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = []
    for i in range(5):
        replays.append(_make_replay(
            n_moves=100 + i,
            worker_id=i % 2,
            wall_time_s=10.0 + i,
            final_root_value=0.1 * i,
            final_top1_share=0.2 + 0.1 * i,
            leaf_evals=1000 + i * 100,
            backups=2000 + i * 100,
            nn_batches=50 + i * 5,
        ))
    out = aggregate_per_game_stats(replays)

    assert out["n_games_total"] == 5
    assert out["n_games_with_any_stats"] == 5
    # Every coverage entry == 5 (except worker_id — that's covered by Task 3's test 7)
    for key in ("wall_time_s", "final_root_value", "final_top1_share",
                "compute.leaf_evals", "compute.backups", "compute.nn_batches",
                "n_moves", "reason"):
        assert out["coverage"][key] == 5, f"coverage[{key!r}] should be 5"
    # Distribution blocks non-null
    assert out["wall_time_s"] is not None
    assert out["wall_time_s"]["mean"] == 12.0           # mean of [10,11,12,13,14] — exact int math
    assert out["wall_time_s"]["min"] == 10.0
    assert out["wall_time_s"]["max"] == 14.0
    assert out["wall_time_s"]["total"] == 60.0          # sum
    # Decimal arithmetic on [0, 0.1, 0.2, 0.3, 0.4] is not byte-exact in IEEE 754.
    assert out["final_root_value"]["mean"]     == pytest.approx(0.2)
    assert out["final_root_value"]["abs_mean"] == pytest.approx(0.2)  # all values >= 0 here
    assert out["final_top1_share"]["mean"]     == pytest.approx(0.4)  # mean of [0.2, 0.3, 0.4, 0.5, 0.6]
    assert out["final_top1_share"]["min"]      == pytest.approx(0.2)
    assert out["compute_per_game"] is not None
    assert out["compute_per_game"]["leaf_evals"]["mean"] == 1200.0  # mean of [1000,1100,1200,1300,1400] — exact int math
    assert out["compute_per_game"]["backups"]["mean"] == 2200.0
    assert out["compute_per_game"]["nn_batches"]["mean"] == 60.0


# -------------------------------------------------------------------------
# Test 4: per-field coverage counts independently
# -------------------------------------------------------------------------

def test_aggregate_per_field_coverage_counts_independently():
    """Mixed coverage: 8 have wall_time_s, 5 have final_top1_share, 7 have nn_batches."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = []
    for i in range(10):
        kw = {"n_moves": 50 + i, "reason": "win"}
        if i < 8:
            kw["wall_time_s"] = 1.0 * (i + 1)
        if i < 5:
            kw["final_top1_share"] = 0.5
        if i < 7:
            kw["leaf_evals"] = 100
            kw["backups"]   = 200
            kw["nn_batches"] = 10
        # else: include_compute=True but no compute keys means meta.compute absent
        replays.append(_make_replay(**kw))

    out = aggregate_per_game_stats(replays)

    assert out["n_games_total"] == 10
    assert out["coverage"]["wall_time_s"] == 8
    assert out["coverage"]["final_top1_share"] == 5
    assert out["coverage"]["compute.nn_batches"] == 7
    assert out["coverage"]["compute.leaf_evals"] == 7
    assert out["coverage"]["compute.backups"] == 7
    assert out["coverage"]["final_root_value"] == 0
    assert out["coverage"]["worker_id"] == 0
    # Distribution blocks computed only over their covering games
    assert out["wall_time_s"] is not None
    assert out["wall_time_s"]["total"] == 36.0          # 1+2+...+8
    assert out["final_top1_share"] is not None
    assert out["final_top1_share"]["mean"] == 0.5       # all five are 0.5
    assert out["compute_per_game"]["nn_batches"]["mean"] == 10.0
    # final_root_value has zero coverage → null
    assert out["final_root_value"] is None


# -------------------------------------------------------------------------
# Test 8: missing compute subkey is excluded, not zero
# -------------------------------------------------------------------------

def test_aggregate_compute_subkey_missing_is_excluded_not_zero():
    """meta.compute = {leaf_evals: 100, backups: 200} (no nn_batches) →
    coverage.compute.nn_batches == 0; nn_batches block is null;
    leaf_evals/backups stats reflect actual values, not depressed by phantom zeros.
    """
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        _make_replay(leaf_evals=100, backups=200),  # no nn_batches
        _make_replay(leaf_evals=300, backups=400),  # no nn_batches
    ]
    out = aggregate_per_game_stats(replays)

    assert out["coverage"]["compute.leaf_evals"] == 2
    assert out["coverage"]["compute.backups"] == 2
    assert out["coverage"]["compute.nn_batches"] == 0
    assert out["compute_per_game"] is not None
    assert out["compute_per_game"]["leaf_evals"]["mean"] == 200.0  # (100+300)/2
    assert out["compute_per_game"]["backups"]["mean"] == 300.0     # (200+400)/2
    assert out["compute_per_game"]["nn_batches"] is None


# -------------------------------------------------------------------------
# Test 8b: empty meta.compute does not count as carrying any persistence stats
# -------------------------------------------------------------------------

def test_empty_compute_object_does_not_count_as_any_stats():
    """A replay with meta.compute = {} (key present but no subkeys) must NOT
    increment n_games_with_any_stats — we count actual populated fields, not
    just key presence."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    # _make_replay with no compute kwargs and include_compute=True → meta.compute
    # is OMITTED entirely. To get an explicit empty {} we construct directly.
    replay = {"meta": {"n_moves": 100, "reason": "win", "compute": {}}}
    out = aggregate_per_game_stats([replay])

    assert out["n_games_total"] == 1
    assert out["n_games_with_any_stats"] == 0  # empty compute does not count
    assert out["coverage"]["compute.leaf_evals"] == 0
    assert out["coverage"]["compute.backups"] == 0
    assert out["coverage"]["compute.nn_batches"] == 0
    assert out["compute_per_game"] is None


# -------------------------------------------------------------------------
# Test 5: worker_balance groups by worker_id and computes ratios + CV
# -------------------------------------------------------------------------

def test_aggregate_worker_balance_groups_by_worker_id():
    """4 replays from 2 workers with different wall_time_s → all metrics correct."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        # Worker 0: 2 games, total wall_time 30s
        _make_replay(worker_id=0, wall_time_s=10.0, n_moves=100),
        _make_replay(worker_id=0, wall_time_s=20.0, n_moves=120),
        # Worker 1: 2 games, total wall_time 60s
        _make_replay(worker_id=1, wall_time_s=25.0, n_moves=110),
        _make_replay(worker_id=1, wall_time_s=35.0, n_moves=130),
    ]
    out = aggregate_per_game_stats(replays)

    wb = out["worker_balance"]
    assert wb["by_worker"]["0"]["games"] == 2
    assert wb["by_worker"]["0"]["wall_time_total_s"] == 30.0
    assert wb["by_worker"]["0"]["wall_time_mean_s"] == 15.0
    assert wb["by_worker"]["1"]["games"] == 2
    assert wb["by_worker"]["1"]["wall_time_total_s"] == 60.0
    assert wb["by_worker"]["1"]["wall_time_mean_s"] == 30.0
    assert wb["in_process_count"] == 0
    # max/min ratios
    assert wb["max_min_wall_time_ratio"] == 2.0   # 60 / 30
    assert wb["max_min_games_ratio"] == 1.0       # 2 / 2
    # CV: per-worker totals = [30, 60], mean=45, stddev (ddof=0) = sqrt(((30-45)^2 + (60-45)^2)/2) = 15
    # CV = 15/45 = 0.333...
    assert abs(wb["wall_time_cv"] - (15.0 / 45.0)) < 1e-9


# -------------------------------------------------------------------------
# Test 6: per-worker n_moves_total
# -------------------------------------------------------------------------

def test_aggregate_worker_balance_includes_n_moves_per_worker():
    """by_worker[w]["n_moves_total"] is sum of meta.n_moves across that worker's games."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        _make_replay(worker_id=0, n_moves=100, wall_time_s=1.0),
        _make_replay(worker_id=0, n_moves=200, wall_time_s=1.0),
        _make_replay(worker_id=1, n_moves=150, wall_time_s=1.0),
    ]
    out = aggregate_per_game_stats(replays)

    assert out["worker_balance"]["by_worker"]["0"]["n_moves_total"] == 300
    assert out["worker_balance"]["by_worker"]["1"]["n_moves_total"] == 150


# -------------------------------------------------------------------------
# Test 7: in-process games counted separately from worker games
# -------------------------------------------------------------------------

def test_aggregate_in_process_games_counted_separately():
    """worker_id=None (in-process) increments in_process_count, not by_worker.
    worker_id=0 is a legitimate worker key, not conflated with null.
    """
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        _make_replay(worker_id=0,    wall_time_s=1.0),  # worker 0
        _make_replay(worker_id=1,    wall_time_s=1.0),  # worker 1
        _make_replay(worker_id=None, wall_time_s=1.0),  # in-process
        _make_replay(worker_id=None, wall_time_s=1.0),  # in-process
    ]
    out = aggregate_per_game_stats(replays)

    wb = out["worker_balance"]
    assert set(wb["by_worker"].keys()) == {"0", "1"}
    assert "None" not in wb["by_worker"]
    assert wb["in_process_count"] == 2
    # coverage["worker_id"] counts ALL games where the field is present (incl. explicit null)
    assert out["coverage"]["worker_id"] == 4
    # All four games carry persistence-era fields (worker_id present is sufficient,
    # even when explicitly null — that's a meaningful "in-process new schema" signal).
    assert out["n_games_with_any_stats"] == 4


# -------------------------------------------------------------------------
# Test 11: single worker yields null ratios and null CV
# -------------------------------------------------------------------------

def test_aggregate_single_worker_yields_null_ratios():
    """One distinct worker → all three imbalance metrics are None."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        _make_replay(worker_id=0, wall_time_s=10.0),
        _make_replay(worker_id=0, wall_time_s=20.0),
    ]
    out = aggregate_per_game_stats(replays)

    wb = out["worker_balance"]
    assert wb["max_min_wall_time_ratio"] is None
    assert wb["max_min_games_ratio"] is None
    assert wb["wall_time_cv"] is None
    # by_worker still populated
    assert wb["by_worker"]["0"]["games"] == 2


# -------------------------------------------------------------------------
# Test 12: worker with zero wall_time_total excluded from ratio
# -------------------------------------------------------------------------

def test_aggregate_workers_with_zero_wall_time_excluded_from_ratio():
    """Per spec §7: worker with wall_time_total_s == 0 is excluded from
    max_min_wall_time_ratio computation; if fewer than 2 workers remain, ratio is None.
    """
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    # Two workers, but one has wall_time_total_s == 0 (no wall_time_s on its replays)
    replays = [
        _make_replay(worker_id=0, wall_time_s=10.0),
        _make_replay(worker_id=0, wall_time_s=20.0),
        _make_replay(worker_id=1),  # no wall_time_s
    ]
    out = aggregate_per_game_stats(replays)

    wb = out["worker_balance"]
    # by_worker still has both, but worker 1's wall_time_total_s == 0
    assert wb["by_worker"]["0"]["wall_time_total_s"] == 30.0
    assert wb["by_worker"]["1"]["wall_time_total_s"] == 0.0
    # Only 1 worker remains after excluding zero-time worker → ratio is None
    assert wb["max_min_wall_time_ratio"] is None
    assert wb["wall_time_cv"] is None
    # max_min_games_ratio is well-defined (1 game vs 2 games)
    assert wb["max_min_games_ratio"] == 2.0


# -------------------------------------------------------------------------
# Test 13: uniform per-worker wall_time yields ratio=1.0, cv=0.0
# -------------------------------------------------------------------------

def test_aggregate_uniform_per_worker_wall_time_yields_unity_ratio_zero_cv():
    """All per-worker wall_time_total equal → max_min_ratio=1.0, cv=0.0."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        _make_replay(worker_id=0, wall_time_s=10.0),
        _make_replay(worker_id=1, wall_time_s=10.0),
        _make_replay(worker_id=2, wall_time_s=10.0),
    ]
    out = aggregate_per_game_stats(replays)

    wb = out["worker_balance"]
    assert wb["max_min_wall_time_ratio"] == 1.0
    assert wb["wall_time_cv"] == 0.0
    assert wb["max_min_games_ratio"] == 1.0


# -------------------------------------------------------------------------
# Test 14: zero-coverage short message
# -------------------------------------------------------------------------

def test_format_renders_zero_coverage_short_message():
    """n_games_with_any_stats == 0 → game_length and outcomes lines render,
    then the short 'no games carry new persistence fields' message.
    """
    from scripts.twixt_replay_analyzer import (
        aggregate_per_game_stats, format_per_game_stats_report,
    )

    # Old-schema-only replays
    replays = [_make_replay(n_moves=100, reason="win") for _ in range(3)]
    pgs = aggregate_per_game_stats(replays)
    lines = format_per_game_stats_report(pgs)
    text = "\n".join(lines)

    # game_length and outcomes are still rendered (they use pre-existing fields)
    assert "Game length:" in text
    assert "Outcomes:" in text
    # The short fallback message
    assert "no games carry new persistence fields" in text
    # No persistence-era stat lines
    assert "Wall time:" not in text
    assert "Workers:" not in text
    assert "Final root:" not in text
    assert "Final top1:" not in text
    assert "Compute/game:" not in text


# -------------------------------------------------------------------------
# Test 15: full block rendering with uniform coverage suppresses Coverage line
# -------------------------------------------------------------------------

def test_format_renders_full_block():
    """Fully-populated per_game_stats → expected lines in expected order.
    Uniform coverage → no separate 'Coverage:' line printed.
    """
    from scripts.twixt_replay_analyzer import (
        aggregate_per_game_stats, format_per_game_stats_report,
    )

    replays = [
        _make_replay(worker_id=0, wall_time_s=10.0, final_root_value=0.5,
                     final_top1_share=0.4, leaf_evals=1000, backups=2000,
                     nn_batches=50, n_moves=100),
        _make_replay(worker_id=1, wall_time_s=20.0, final_root_value=-0.3,
                     final_top1_share=0.6, leaf_evals=1500, backups=2500,
                     nn_batches=80, n_moves=120),
    ]
    pgs = aggregate_per_game_stats(replays)
    lines = format_per_game_stats_report(pgs)
    text = "\n".join(lines)

    # Headers and stat lines all present
    assert "Per-game stats" in text
    assert "Game length:" in text
    assert "Outcomes:" in text
    assert "Wall time:" in text
    assert "Workers:" in text
    assert "Final root:" in text
    assert "Final top1:" in text
    assert "Compute/game:" in text
    # Uniform coverage → no Coverage: line
    assert "Coverage:" not in text


# -------------------------------------------------------------------------
# Test 16: partial coverage prints the Coverage: line
# -------------------------------------------------------------------------

def test_format_renders_coverage_line_on_partial_coverage():
    """Non-uniform per-field coverage → Coverage: line is printed."""
    from scripts.twixt_replay_analyzer import (
        aggregate_per_game_stats, format_per_game_stats_report,
    )

    # 3 replays: all have wall_time_s, only 2 have final_top1_share
    replays = [
        _make_replay(worker_id=0, wall_time_s=10.0, final_top1_share=0.5),
        _make_replay(worker_id=0, wall_time_s=15.0, final_top1_share=0.6),
        _make_replay(worker_id=0, wall_time_s=20.0),  # no final_top1_share
    ]
    pgs = aggregate_per_game_stats(replays)
    lines = format_per_game_stats_report(pgs)
    text = "\n".join(lines)

    assert "Coverage:" in text


# -------------------------------------------------------------------------
# Test 17: zero-coverage field is omitted entirely (no "n/a" line)
# -------------------------------------------------------------------------

def test_format_omits_lines_for_zero_coverage_fields():
    """When a field has zero coverage, omit its line entirely."""
    from scripts.twixt_replay_analyzer import (
        aggregate_per_game_stats, format_per_game_stats_report,
    )

    # Replays carry wall_time_s but not final_top1_share
    replays = [
        _make_replay(worker_id=0, wall_time_s=10.0),
        _make_replay(worker_id=0, wall_time_s=20.0),
    ]
    pgs = aggregate_per_game_stats(replays)
    lines = format_per_game_stats_report(pgs)
    text = "\n".join(lines)

    assert "Wall time:" in text
    assert "Final top1:" not in text
    # Specifically: we never render "Final top1: n/a" (we omit the whole line instead).
    # Generic "n/a" might appear in unrelated future text, so be precise.
    assert "Final top1: n/a" not in text
    assert "Final root: n/a" not in text


# -------------------------------------------------------------------------
# Test 18: single worker line shape
# -------------------------------------------------------------------------

def test_format_handles_single_worker():
    """One distinct worker → 'Workers: 1 active; games=N; wall-time mean=Xs (in-process: M)'."""
    from scripts.twixt_replay_analyzer import (
        aggregate_per_game_stats, format_per_game_stats_report,
    )

    replays = [
        _make_replay(worker_id=0, wall_time_s=10.0),
        _make_replay(worker_id=0, wall_time_s=20.0),
    ]
    pgs = aggregate_per_game_stats(replays)
    lines = format_per_game_stats_report(pgs)
    text = "\n".join(lines)

    assert "1 active" in text
    assert "ratio" not in text   # no ratios printed when only 1 worker
    assert "cv=" not in text     # no CV either


# -------------------------------------------------------------------------
# Test 19: in-process only (worker_id all null)
# -------------------------------------------------------------------------

def test_format_handles_in_process_only():
    """All worker_id == null → 'Workers: 0 active; in-process: N'."""
    from scripts.twixt_replay_analyzer import (
        aggregate_per_game_stats, format_per_game_stats_report,
    )

    replays = [
        _make_replay(worker_id=None, wall_time_s=10.0),
        _make_replay(worker_id=None, wall_time_s=20.0),
        _make_replay(worker_id=None, wall_time_s=30.0),
    ]
    pgs = aggregate_per_game_stats(replays)
    lines = format_per_game_stats_report(pgs)
    text = "\n".join(lines)

    assert "0 active" in text
    assert "in-process: 3" in text


# -------------------------------------------------------------------------
# Test 20: human-readable duration formatting
# -------------------------------------------------------------------------

def test_format_human_readable_duration():
    """_format_duration_human handles the three cases per spec §5.2."""
    from scripts.twixt_replay_analyzer import _format_duration_human

    # < 60s → 'X.Xs'
    assert _format_duration_human(0.0) == "0.0s"
    assert _format_duration_human(30.0) == "30.0s"
    assert _format_duration_human(59.9) == "59.9s"
    # < 1h → 'Xm Ys'
    assert _format_duration_human(60.0)  == "1m 0s"
    assert _format_duration_human(145.0) == "2m 25s"
    assert _format_duration_human(3599.0) == "59m 59s"
    # >= 1h → 'XhYm'
    assert _format_duration_human(3600.0)   == "1h0m"
    assert _format_duration_human(17852.4)  == "4h57m"
