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
