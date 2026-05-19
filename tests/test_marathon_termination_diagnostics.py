"""Tests for the marathon-termination diagnostics module.

Spec: docs/superpowers/specs/2026-05-19-marathon-termination-tuning-design.md
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.marathon_termination_diagnostics import (
    detect_no_progress_windows,
    classify_adjudication_coverage,
    compute_resign_gate_breakdown,
    value_uncertain_guard,
    game_length_bucket,
    ADJUDICATION_GATE_BUCKETS,
    GAME_LENGTH_BUCKETS,
    NO_PROGRESS_WINDOW_SIZE,
)


def _ply_entry(*, primary_class: str,
               own_td_before: int = 5, own_td_after: int = 5,
               opp_td_before: int = 5, opp_td_after: int = 5):
    """A goal_completion_diagnostics entry, minimally shaped for the detector."""
    return {
        "ply": 50, "side_to_move": "red",
        "goal_completion": {
            "total_goal_distance_before": own_td_before,
            "category": "one_endpoint_distance_2",
            "_own_td_after": own_td_after,
            "_opp_td_before": opp_td_before,
            "_opp_td_after": opp_td_after,
        },
        "selected_move": [10, 10],
        "selected_move_classification": {
            "primary_class": primary_class,
            "total_goal_distance_before": own_td_before,
            "total_goal_distance_after": own_td_after,
        },
    }


def test_no_progress_window_detects_pure_structural_run():
    """Spec §7 test 1. 15 consecutive redundant_reinforcement moves
    with no goal-distance progress → 1 window detected."""
    entries = [
        _ply_entry(primary_class="redundant_reinforcement",
                   own_td_before=5, own_td_after=5,
                   opp_td_before=5, opp_td_after=5)
        for _ in range(15)
    ]
    assert detect_no_progress_windows(entries, side="red") == 1


def test_no_progress_window_breaks_on_distance_reduction():
    """Spec §7 test 2. 14 redundant + 1 reduces_total_goal_distance → 0 windows."""
    entries = [
        _ply_entry(primary_class="redundant_reinforcement",
                   own_td_before=5, own_td_after=5)
        for _ in range(14)
    ]
    entries.append(_ply_entry(
        primary_class="reduces_total_goal_distance",
        own_td_before=5, own_td_after=4,
    ))
    assert detect_no_progress_windows(entries, side="red") == 0


def test_no_progress_window_breaks_on_endpoint_completion():
    """Spec §7 test 3. 14 redundant + 1 completes_endpoint → 0 windows."""
    entries = [
        _ply_entry(primary_class="redundant_reinforcement")
        for _ in range(14)
    ]
    entries.append(_ply_entry(primary_class="completes_endpoint"))
    assert detect_no_progress_windows(entries, side="red") == 0


def test_no_progress_window_breaks_on_opponent_block():
    """Spec §7 test 4. 14 redundant + 1 blocks_opponent_closeout → 0 windows."""
    entries = [
        _ply_entry(primary_class="redundant_reinforcement")
        for _ in range(14)
    ]
    entries.append(_ply_entry(primary_class="blocks_opponent_closeout"))
    assert detect_no_progress_windows(entries, side="red") == 0


def test_no_progress_window_window_size_15():
    """Spec §7 test 5. Exactly 14 redundant → 0 windows; 15 → 1."""
    e14 = [_ply_entry(primary_class="redundant_reinforcement") for _ in range(14)]
    e15 = [_ply_entry(primary_class="redundant_reinforcement") for _ in range(15)]
    assert detect_no_progress_windows(e14, side="red") == 0
    assert detect_no_progress_windows(e15, side="red") == 1
    # Sanity-check the exported constant.
    assert NO_PROGRESS_WINDOW_SIZE == 15


def test_no_progress_window_opponent_block_uses_primary_class_only():
    """Spec §7 test 6. The opponent-block test uses the
    primary_class == 'blocks_opponent_closeout' marker (Spec 4 vocabulary)
    — confirms we are NOT applying a stricter local recomputation. If
    Spec 4's defense classifier flagged the move, we trust it."""
    # A move classified as blocks_opponent_closeout (per Spec 4) — even if
    # the inline distance fields look ambiguous — must count as a block.
    entries = [
        _ply_entry(primary_class="redundant_reinforcement")
        for _ in range(14)
    ]
    entries.append({
        "ply": 50, "side_to_move": "red",
        "goal_completion": {"total_goal_distance_before": 5, "category": "x"},
        "selected_move": [10, 10],
        "selected_move_classification": {
            "primary_class": "blocks_opponent_closeout",
        },
    })
    # Run of 14 followed by a block → 0 no-progress windows.
    assert detect_no_progress_windows(entries, side="red") == 0


def _gc_record_state_cap(**meta_overrides):
    """A per-game (record, meta, diagnostics) triple for a 280-ply state_cap game."""
    record = {
        "iteration": 220, "game_idx": 0,
        "winner": None, "reason": "state_cap", "n_moves": 280,
        "first_total_goal_distance": 2,
        "winner_moves_with_dominant_unavailable": 0,
        "conversion_delay_plies": 0,
    }
    meta = {"reason": "state_cap", "n_moves": 280}
    meta.update(meta_overrides)
    return record, meta, []


def test_adjudication_coverage_blocked_by_min_top1():
    """Spec §7 test 7. self-play 'top1' → bucket 'min_top1_share'."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason="top1")
    assert classify_adjudication_coverage(rec, meta, diag) == "min_top1_share"


def test_adjudication_coverage_blocked_by_value_below_threshold():
    """Spec §7 test 8. self-play 'threshold' → bucket 'value_below_threshold'."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason="threshold")
    assert classify_adjudication_coverage(rec, meta, diag) == "value_below_threshold"


def test_adjudication_coverage_blocked_by_min_visits():
    """Spec §7 test 9. self-play 'visits' → bucket 'min_visits'."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason="visits")
    assert classify_adjudication_coverage(rec, meta, diag) == "min_visits"


def test_adjudication_coverage_not_attempted_when_ply_blocked():
    """Spec §7 test 10. self-play 'ply' → 'not_attempted' (adjudication
    couldn't run because the ply gate fired)."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason="ply")
    assert classify_adjudication_coverage(rec, meta, diag) == "not_attempted"


def test_adjudication_coverage_would_have_passed_when_none_on_state_cap():
    """Spec §7 test 11. state_cap game with adjudication_block_reason
    PRESENT as None (key exists, value is None) → 'would_have_passed'
    (bug indicator: game state-capped, key was set but no blocking gate
    recorded → an attempt should have happened). MUST NOT silently
    collapse to 'not_attempted'."""
    rec, meta, diag = _gc_record_state_cap(adjudication_block_reason=None)
    # Explicit assertion: the key IS in meta, just None-valued.
    assert "adjudication_block_reason" in meta
    assert meta["adjudication_block_reason"] is None
    assert classify_adjudication_coverage(rec, meta, diag) == "would_have_passed"


def test_adjudication_coverage_missing_signal_when_key_absent():
    """Spec §7 test 12. Old-format per-game JSON where the key isn't
    present at all → 'missing_signal' (observability gap, not a bug)."""
    rec, meta, diag = _gc_record_state_cap()
    assert "adjudication_block_reason" not in meta
    assert classify_adjudication_coverage(rec, meta, diag) == "missing_signal"


def test_adjudication_coverage_skipped_for_non_state_cap_games():
    """Spec §7 test 14. Game ending in win → returns None
    (excluded from §3.2 scope)."""
    record = {
        "iteration": 220, "game_idx": 0,
        "winner": "red", "reason": "win", "n_moves": 80,
    }
    meta = {"reason": "win", "adjudication_block_reason": None}
    assert classify_adjudication_coverage(record, meta, []) is None


def test_adjudication_gate_buckets_export_matches_spec_taxonomy():
    """Spec §3.2 enumeration. The exported tuple lists exactly the six
    bucket names so the analyzer and tests share one source of truth."""
    assert set(ADJUDICATION_GATE_BUCKETS) == {
        "not_attempted",
        "value_below_threshold",
        "min_top1_share",
        "min_visits",
        "missing_signal",
        "would_have_passed",
    }


def _losing_side_ply(*, search_score: float, top1_share: float,
                     visit_count: int = 250, ply: int = 200,
                     side_to_move: str = "black"):
    """A goal_completion_diagnostics entry suitable for resign-gate scoring.
    'side_to_move' here is the LOSING side's own-move; the helper uses
    the convention that root_summary.q_value is the score from the
    side-to-move's perspective."""
    return {
        "ply": ply, "side_to_move": side_to_move,
        "root_summary": {
            "q_value": search_score,
            "visit_count": visit_count,
        },
        "root_top1_share": top1_share,
    }


def _resign_thresholds():
    """Production defaults from the 220-229 launch command (memory entry)."""
    return dict(
        resign_threshold=-0.945,
        resign_min_ply=80,
        resign_min_visits=200,
        resign_min_top1_share=0.102,
    )


def test_resign_gate_breakdown_separates_value_hits_from_eligible_hits():
    """Spec §7 test 15 + 16. Game where value crosses threshold often
    but visits/top1 sometimes fail: value_hits >= eligible_hits >= blocked_by_top1."""
    diag = [
        # Hit 1: value crosses, all gates pass except top1 (eligible, blocked by top1).
        _losing_side_ply(search_score=-0.95, top1_share=0.05, visit_count=300, ply=200),
        # Hit 2: value crosses, all gates pass (eligible, NOT blocked).
        _losing_side_ply(search_score=-0.97, top1_share=0.20, visit_count=300, ply=210),
        # Hit 3: value crosses but visits fail (value_hit yes, eligible no).
        _losing_side_ply(search_score=-0.96, top1_share=0.20, visit_count=100, ply=220),
        # Non-hit: value below threshold doesn't qualify (no value_hit).
        _losing_side_ply(search_score=-0.50, top1_share=0.20, visit_count=300, ply=230),
    ]
    record = {"winner": "red", "n_moves": 250}  # losing side: black
    out = compute_resign_gate_breakdown(record, diag, losing_side="black", **_resign_thresholds())
    assert out["value_hits"] == 3
    assert out["eligible_hits"] == 2          # hits 1 and 2 (hit 3 fails visits)
    assert out["blocked_by_top1"] == 1        # hit 1 only
    # over_value_hits = 1/3, over_eligible_hits = 1/2.
    assert abs(out["top1_block_rate_over_value_hits"] - 1/3) < 1e-9
    assert abs(out["top1_block_rate_over_eligible_hits"] - 1/2) < 1e-9


def test_resign_gate_breakdown_empty_when_no_value_hits():
    """Game where the loser never crossed resign_threshold → all counts 0,
    rates 0.0 (not NaN)."""
    diag = [
        _losing_side_ply(search_score=-0.50, top1_share=0.20, ply=200),
    ]
    record = {"winner": "red", "n_moves": 250}
    out = compute_resign_gate_breakdown(record, diag, losing_side="black", **_resign_thresholds())
    assert out["value_hits"] == 0
    assert out["eligible_hits"] == 0
    assert out["blocked_by_top1"] == 0
    assert out["top1_block_rate_over_value_hits"] == 0.0
    assert out["top1_block_rate_over_eligible_hits"] == 0.0


def test_game_length_bucket_partitions():
    """Spec §3.3 game-length partition: short / mid / long."""
    assert game_length_bucket(50) == "short"
    assert game_length_bucket(100) == "short"
    assert game_length_bucket(101) == "mid"
    assert game_length_bucket(200) == "mid"
    assert game_length_bucket(201) == "long"
    assert game_length_bucket(280) == "long"
    assert set(GAME_LENGTH_BUCKETS) == {"short", "mid", "long"}


def test_resign_separates_no_value_signal_from_blocked_by_top1():
    """Spec §7 test 16. A game with no value hits is distinguishable from
    a game with high blocked_by_top1 rate."""
    record = {"winner": "red", "n_moves": 250}
    diag_no_value = [_losing_side_ply(search_score=-0.50, top1_share=0.20)]
    diag_high_block = [
        _losing_side_ply(search_score=-0.97, top1_share=0.05, visit_count=300, ply=200),
        _losing_side_ply(search_score=-0.97, top1_share=0.05, visit_count=300, ply=210),
    ]
    out_no_value = compute_resign_gate_breakdown(record, diag_no_value, losing_side="black", **_resign_thresholds())
    out_high_block = compute_resign_gate_breakdown(record, diag_high_block, losing_side="black", **_resign_thresholds())
    # Distinguishable: no-value has zero value_hits; high-block has positive value_hits + positive blocked_by_top1.
    assert out_no_value["value_hits"] == 0
    assert out_no_value["blocked_by_top1"] == 0
    assert out_high_block["value_hits"] == 2
    assert out_high_block["blocked_by_top1"] == 2


def test_value_uncertain_guard_blocks_termination_when_neutral():
    """Spec §7 test 17. Last 10 own-plies for both sides with |score|<0.30
    → guard returns True (do not terminate)."""
    diagnostics = []
    for ply in range(220, 240):
        side = "red" if ply % 2 == 0 else "black"
        diagnostics.append({
            "ply": ply, "side_to_move": side,
            "root_summary": {"q_value": 0.05},  # near neutral
        })
    assert value_uncertain_guard(diagnostics) is True


def test_value_uncertain_guard_blocks_termination_when_oscillatory():
    """Spec §7 test 18. EITHER side's last 10 own-plies show >=3 sign-flips
    in its OWN q_value sequence → guard True. Per-side oscillation, NOT the
    natural interleaved turn-alternation (which would conflate every stable
    game with an oscillatory one — see test 19)."""
    diagnostics = []
    # Red's own q sequence oscillates: +0.5, -0.5, +0.5, ...
    # Black's own q sequence oscillates: -0.5, +0.5, -0.5, ...
    # 20 entries total = 10 own-plies per side.
    for i in range(20):
        if i % 2 == 0:  # red
            score = 0.5 if (i // 2) % 2 == 0 else -0.5
            side = "red"
        else:  # black
            score = -0.5 if (i // 2) % 2 == 0 else 0.5
            side = "black"
        diagnostics.append({
            "ply": 220 + i, "side_to_move": side,
            "root_summary": {"q_value": score},
        })
    assert value_uncertain_guard(diagnostics) is True


def test_value_uncertain_guard_allows_termination_when_stable_losing():
    """Spec §7 test 19. Last 10 own-plies stably below -0.30 for the loser
    AND above 0.30 for the winner → guard returns False (terminate is safe)."""
    diagnostics = []
    for i in range(20):
        side = "red" if i % 2 == 0 else "black"
        score = 0.85 if side == "red" else -0.85
        diagnostics.append({
            "ply": 220 + i, "side_to_move": side,
            "root_summary": {"q_value": score},
        })
    assert value_uncertain_guard(diagnostics) is False


from scripts.GPU.alphazero.marathon_termination_diagnostics import (
    aggregate_marathon_termination,
)


def _per_game(iteration, game_idx, *, reason="win", n_moves=80, winner="red",
              adj_block=None, diagnostics=None):
    record = {
        "iteration": iteration, "game_idx": game_idx,
        "winner": winner, "reason": reason, "n_moves": n_moves,
        "first_total_goal_distance": 2,
        "winner_moves_with_dominant_unavailable": 0,
        "conversion_delay_plies": 0,
    }
    meta = {"reason": reason, "n_moves": n_moves}
    if adj_block is not None:
        meta["adjudication_block_reason"] = adj_block
    return record, meta, (diagnostics or [])


def test_aggregate_marathon_termination_per_iter_and_range_totals():
    """Spec §7 test 20. 3 games across 2 iters → correct per-iter rows + range-total row."""
    games = [
        _per_game(220, 0, reason="state_cap", n_moves=280, winner=None, adj_block="top1"),
        _per_game(220, 1, reason="state_cap", n_moves=280, winner=None, adj_block="threshold"),
        _per_game(221, 0, reason="state_cap", n_moves=280, winner=None, adj_block="top1"),
    ]
    resign_cfg = dict(
        resign_threshold=-0.945, resign_min_ply=80,
        resign_min_visits=200, resign_min_top1_share=0.102,
    )
    out = aggregate_marathon_termination(games, **resign_cfg)
    assert out["per_iter"][220]["state_cap_280_games"] == 2
    assert out["per_iter"][221]["state_cap_280_games"] == 1
    assert out["per_iter"][220]["adjudication_gate_counts"]["min_top1_share"] == 1
    assert out["per_iter"][220]["adjudication_gate_counts"]["value_below_threshold"] == 1
    assert out["per_iter"][221]["adjudication_gate_counts"]["min_top1_share"] == 1
    assert out["range_total"]["state_cap_280_games"] == 3
    assert out["range_total"]["adjudication_gate_counts"]["min_top1_share"] == 2
    assert out["range_total"]["adjudication_gate_counts"]["value_below_threshold"] == 1


def test_aggregate_marathon_termination_no_progress_window_mean():
    """Spec §3.4. Per-iter mean of detected no-progress windows across games."""
    games = [
        _per_game(220, 0, reason="win", n_moves=80, winner="red",
                  diagnostics=[]),
        _per_game(220, 1, reason="win", n_moves=80, winner="red",
                  diagnostics=[
                      {"ply": p, "side_to_move": "red" if i % 2 == 0 else "black",
                       "selected_move_classification": {"primary_class": "redundant_reinforcement"}}
                      for i, p in enumerate(range(50, 80))
                  ]),
    ]
    resign_cfg = dict(
        resign_threshold=-0.945, resign_min_ply=80,
        resign_min_visits=200, resign_min_top1_share=0.102,
    )
    out = aggregate_marathon_termination(games, **resign_cfg)
    # Iter 220 has two games; one has 0 windows, the other has at least one
    # 15-window run on one side. Mean across games > 0.
    assert out["per_iter"][220]["mean_no_progress_windows_per_game"] > 0.0
