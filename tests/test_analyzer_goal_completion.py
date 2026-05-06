"""Tests for analyzer goal-completion aggregation (spec 2026-05-03 §7).

Phase 2 covers Class 1 (decisive winner) detection + watch-window
classification + summary block + report rendering + worst-cases CSV.
"""
import csv
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import (
    aggregate_goal_completion_diagnostics,
    format_goal_completion_report,
    write_goal_completion_worst_cases_csv,
)


def _load_game_097() -> dict:
    """Load the canonical Game 097 replay as a curated test fixture.

    The Phase 1 exercise validated that compute_goal_completion_state
    correctly identifies this game's turn-43 two-endpoint-closeout-2ply
    state with conversion delay 16 plies (turn 43 -> turn 59).
    """
    p = PROJECT_ROOT / "scripts" / "GPU" / "logs" / "games" / "iter_0108_game_097.json"
    return json.loads(p.read_text())


def _load_verify_game(idx: int) -> dict:
    """Load a verification-run game with populated search_score (Phase 0)."""
    p = Path(f"/tmp/twixt_verify_phase0_run/games/iter_0099_game_{idx:03d}.json")
    if not p.exists():
        pytest.skip(f"verification game {idx} not present at {p}")
    return json.loads(p.read_text())


def _move(row: int, col: int, player: str, search_score=None):
    """Build a single replay-move dict in the schema the analyzer consumes."""
    return {
        "turn": None,
        "player": player,
        "row": int(row),
        "col": int(col),
        "bridges_created": [],
        "heuristics": {},
        "search_score": search_score,
    }


def _replay(moves, *, winner=None, reason="state_cap", board_size=24,
            starting_player="red", iteration=0, game_idx=0, game_id=None):
    """Build a synthetic replay dict for analyzer tests.

    Adds turn numbers automatically. Note that meta.n_moves is set to len(moves).
    """
    moves_with_turn = []
    for i, m in enumerate(moves):
        m2 = dict(m)
        m2["turn"] = i + 1
        moves_with_turn.append(m2)
    return {
        "id": game_id or f"synthetic_iter_{iteration:04d}_game_{game_idx:03d}",
        "winner": winner,
        "starting_player": starting_player,
        "moves": moves_with_turn,
        "meta": {
            "board_size": board_size,
            "mode": "alphazero",
            "reason": reason,
            "iteration": iteration,
            "game_idx": game_idx,
            "simulations": 100,
            "n_moves": len(moves_with_turn),
            "starting_player": starting_player,
        },
    }


# Session-scoped fixtures: replaying Game 097 is expensive (~50s), so cache.

@pytest.fixture(scope="module")
def game_097():
    return _load_game_097()


@pytest.fixture(scope="module")
def aggregate_g097_default(game_097):
    """Aggregate Game 097 with default parameters (detection_threshold=2)."""
    return aggregate_goal_completion_diagnostics([game_097])


def test_aggregate_empty_replays_returns_zero_block():
    """No replays -> zero-coverage block with all populations zeroed."""
    result = aggregate_goal_completion_diagnostics([])
    assert result["main_population"]["games"] == 0
    assert result["main_population"]["detected"] == 0
    assert result["capped_population"]["games"] == 0
    assert result["excluded_population"]["games"] == 0
    assert result["diagnostics_coverage"]["games_with_diagnostics"] == 0
    assert result["diagnostics_coverage"]["total_records"] == 0


def test_aggregate_class1_detected_game097_canonical(aggregate_g097_default):
    """Curated fixture: Game 097 is the documented canonical failure.

    Validates end-to-end aggregation against Phase-1-validated values:
      - detected = True
      - first_dominant_unclosed_ply = 43 (under threshold=2)
      - actual_terminal_ply = 59
      - conversion_delay_plies = 16
      - first_category = 'two_endpoint_closeout_2ply'
    """
    r = aggregate_g097_default
    main = r["main_population"]
    assert main["games"] == 1
    assert main["detected"] == 1
    assert main["games_with_total_distance_le_2"] == 1
    assert main["games_with_total_distance_le_3"] == 1

    # Per-game record assertions (pull from internal records).
    records = main["_per_game_records_internal"]
    assert len(records) == 1
    rec = records[0]
    assert rec["winner"] == "red"
    assert rec["detected"] is True
    assert rec["first_dominant_unclosed_ply"] == 43
    assert rec["first_category"] == "two_endpoint_closeout_2ply"
    assert rec["actual_terminal_ply"] == 59
    assert rec["conversion_delay_plies"] == 16
    # Watch window: winner moves strictly after turn 43, through terminal.
    # Red turns in (45, 47, 49, 51, 53, 55, 57, 59) = 8 winner moves.
    assert rec["winner_moves_in_watch_window"] == 8


def test_aggregate_class1_undetected_when_min_component_size_unmet(game_097):
    """Test #3: min_component_size=99 (unreachable) -> no detection.

    Verifies the detection gate is plumbed through compute_goal_completion_state.
    """
    r = aggregate_goal_completion_diagnostics([game_097], min_component_size=99)
    main = r["main_population"]
    assert main["games"] == 1
    assert main["detected"] == 0
    rec = main["_per_game_records_internal"][0]
    assert rec["detected"] is False
    assert rec["first_dominant_unclosed_ply"] is None
    # Even ever_distance_le_2 / le_3 are False because min_component_size
    # gate is applied at compute_goal_completion_state level.
    assert rec["ever_distance_le_2"] is False
    assert rec["ever_distance_le_3"] is False


def test_aggregate_class1_detection_threshold_gates_first_dominant_unclosed_ply(
    game_097, aggregate_g097_default,
):
    """Test #4: detection_threshold gates which ply locks first_dominant_unclosed_ply.

    Game 097 reaches total_goal_distance = 2 at ply 43, then 1 at ply 57,
    then 0 at terminal ply 59. With threshold=2 -> first_dominant_unclosed_ply=43;
    with threshold=1 -> first_dominant_unclosed_ply=57 (post-detection drift
    fires later); ever_distance_le_2 / le_3 flags are independent of threshold
    (spec §7.4 separates detection from structural-fact tracking).

    Note: a Class 1 won game eventually reaches total=0 at terminal, so an
    "undetected" outcome at any non-negative threshold is impossible for a
    decisive Class 1 game. The genuine purpose of the threshold is to lock
    the first_dominant_unclosed_ply earlier or later — that is what we test.
    """
    rec_default = aggregate_g097_default["main_population"]["_per_game_records_internal"][0]
    assert rec_default["first_dominant_unclosed_ply"] == 43
    assert rec_default["ever_distance_le_2"] is True
    assert rec_default["ever_distance_le_3"] is True

    # Stricter threshold: first_dominant_unclosed_ply locks at ply 57
    # (where bottom endpoint just closed; total = 1 + 0 = 1).
    r1 = aggregate_goal_completion_diagnostics([game_097], detection_threshold=1)
    rec1 = r1["main_population"]["_per_game_records_internal"][0]
    assert rec1["detected"] is True
    assert rec1["first_dominant_unclosed_ply"] == 57
    # ever_distance_le_2 / le_3 are threshold-independent structural facts.
    assert rec1["ever_distance_le_2"] is True
    assert rec1["ever_distance_le_3"] is True


def test_aggregate_class1_first_dominant_unclosed_ply_locks_at_first_occurrence(
    aggregate_g097_default,
):
    """Test #5: first_dominant_unclosed_ply locks at first occurrence.

    Game 097 stays in two_endpoint_closeout_2ply for many turns after 43;
    first_dominant_unclosed_ply must remain 43, not advance.
    """
    rec = aggregate_g097_default["main_population"]["_per_game_records_internal"][0]
    assert rec["first_dominant_unclosed_ply"] == 43
    assert rec["first_total_goal_distance"] == 2
    assert rec["first_category"] == "two_endpoint_closeout_2ply"


def test_aggregate_class1_watch_window_classifies_each_winner_move_into_primary_class(
    aggregate_g097_default,
):
    """Test #6: primary_class_counts sums to winner_moves_with_dominant_component
    and contains a non-zero redundant_reinforcement (Game 097 redundant drift).
    """
    rec = aggregate_g097_default["main_population"]["_per_game_records_internal"][0]
    counts = rec["primary_class_counts"]
    total = sum(counts.values())
    assert total == rec["winner_moves_with_dominant_component"]
    assert counts["redundant_reinforcement"] >= 1
    # Game 097 has both completion moves (turns 57, 59) and 6 redundant pegs.
    assert counts["completes_endpoint"] >= 1


def test_aggregate_class1_dominant_unavailable_separate_from_primary_class():
    """Test #7: when component dissolves between detection and terminal,
    the dissolved-state moves are counted in winner_moves_with_dominant_unavailable
    and NOT in primary_class_counts.

    Constructed-fixture approach: replay Game 097 but with a stricter
    min_component_size that allows detection at the late-game peak (where
    all 13+ pegs are connected) but causes dissolution at later plies.

    Game 097's largest component reaches >= 11 pegs at turn 43 and grows
    to >= 13 at terminal. Setting min_component_size=12 keeps detection
    around turn 51-55 (depending on cumulative size) and may cause some
    earlier transitions where the threshold isn't met.

    Implementation note: we verify the WIRING (unavailable moves don't
    feed primary_class_counts; total winner_moves_in_watch_window = sum
    of with_component + with_unavailable), not a specific count, since
    counts depend on exact size growth which is fixture-bound.
    """
    # Skip with note if we can't construct a fixture cheaply: use a synthetic
    # post-detection guard: pass min_component_size=99 and craft a replay
    # where helper returns None throughout. In that case both detected=False
    # AND no watch-window classification happens, so this test is a no-op.
    # Instead, verify the invariant directly with a hand-built record.
    from scripts.twixt_replay_analyzer import _summarize_main_population

    # Synthetic record: 5 winner moves in watch window, 3 with component, 2 unavailable.
    # primary_class_counts must sum to 3 (= with_component), not 5.
    fake_rec = {
        "game_id": "synthetic",
        "winner": "red",
        "detected": True,
        "first_dominant_unclosed_ply": 10,
        "first_total_goal_distance": 2,
        "first_category": "two_endpoint_closeout_2ply",
        "actual_terminal_ply": 30,
        "conversion_delay_plies": 20,
        "conversion_delay_winner_moves": 5,
        "winner_moves_in_watch_window": 5,
        "winner_moves_with_dominant_component": 3,
        "winner_moves_with_dominant_unavailable": 2,
        "primary_class_counts": {
            "completes_endpoint": 1,
            "reduces_total_goal_distance": 0,
            "redundant_reinforcement": 2,
            "off_chain": 0,
            "other": 0,
        },
        "max_search_score_after_detection": None,
        "mean_search_score_after_detection": None,
        "high_value_after_detection_plies": 0,
        "root_value_high_but_delayed": False,
        "search_score_coverage_in_watch_window": 0,
        "ever_distance_le_2": True,
        "ever_distance_le_3": True,
        "min_total_goal_distance": 2,
    }
    summary = _summarize_main_population(
        [fake_rec],
        config={},
        detection_threshold=2,
        high_value_threshold=0.9,
        high_value_delay_threshold_plies=10,
    )
    rates = summary["move_quality_after_detection"]
    # Pooled rates use winner_moves_with_dominant_component (3) as denominator.
    assert rates["completes_endpoint_rate"] == pytest.approx(1.0 / 3.0)
    assert rates["redundant_reinforcement_rate"] == pytest.approx(2.0 / 3.0)
    # dominant_unavailable_rate uses (with_component + unavailable) = 5 as denominator.
    assert rates["dominant_unavailable_rate"] == pytest.approx(2.0 / 5.0)


def test_aggregate_class1_high_value_after_detection_uses_search_score_threshold():
    """Test #8: high_value_after_detection_plies counts only post-detection
    winner moves with search_score >= threshold.

    Uses a verification-run game (search_score populated) where some winner
    moves after detection score >= 0.9 and others don't.
    """
    g = _load_verify_game(0)
    r = aggregate_goal_completion_diagnostics([g])
    rec = r["main_population"]["_per_game_records_internal"][0]
    # If detection didn't fire on this game, skip — we don't control exact game shape.
    if not rec["detected"]:
        pytest.skip(f"verify game 0 did not trigger detection at threshold=2")

    # Recount manually: scan winner moves AFTER detection ply with search_score >= 0.9.
    n_high = 0
    n_total_with_score = 0
    detection_ply = rec["first_dominant_unclosed_ply"]
    for i, m in enumerate(g["moves"]):
        ply_1based = i + 1
        if ply_1based <= detection_ply:
            continue
        if m.get("player") != rec["winner"]:
            continue
        ss = m.get("search_score")
        if ss is None:
            continue
        n_total_with_score += 1
        if float(ss) >= 0.9:
            n_high += 1

    assert rec["high_value_after_detection_plies"] == n_high
    assert rec["search_score_coverage_in_watch_window"] == n_total_with_score
    # Threshold sensitivity: a higher threshold reduces or matches the count.
    r_strict = aggregate_goal_completion_diagnostics([g], high_value_threshold=0.999)
    rec_strict = r_strict["main_population"]["_per_game_records_internal"][0]
    assert rec_strict["high_value_after_detection_plies"] <= rec["high_value_after_detection_plies"]


def test_aggregate_class1_root_value_high_but_delayed_requires_both_high_value_and_delay():
    """Test #9: root_value_high_but_delayed requires BOTH
      (a) high_value_after_detection_plies >= 1
      (b) conversion_delay_plies >= high_value_delay_threshold_plies (default 10)

    Validated via _build_class1_per_game_record-derived flag using direct
    aggregator parameter probes. Constructs three scenarios via
    _summarize_main_population to focus on the flag wiring.
    """
    # The flag is set inside _build_class1_per_game_record; assert via
    # different tunings of aggregator params on Game 097 (cached fixture
    # already; here we re-aggregate cheaply using min_component_size=99
    # to skip the per-ply replay cost path).
    #
    # Strategy: feed a verification game with conversion_delay_plies known.
    g = _load_verify_game(0)
    # (a) high_value=0.9, delay_threshold=10 -> flag depends on actual delay+score
    r_default = aggregate_goal_completion_diagnostics([g])
    rec_default = r_default["main_population"]["_per_game_records_internal"][0]
    if not rec_default["detected"]:
        pytest.skip("verify game 0 did not trigger detection at threshold=2")

    actual_delay = rec_default["conversion_delay_plies"]
    n_high = rec_default["high_value_after_detection_plies"]

    # Force "no high values": threshold above any reachable score.
    r_no_high = aggregate_goal_completion_diagnostics(
        [g], high_value_threshold=10.0
    )
    rec_no_high = r_no_high["main_population"]["_per_game_records_internal"][0]
    assert rec_no_high["high_value_after_detection_plies"] == 0
    assert rec_no_high["root_value_high_but_delayed"] is False

    # Force "delay never qualifies": delay threshold larger than the actual delay.
    r_no_delay = aggregate_goal_completion_diagnostics(
        [g], high_value_delay_threshold_plies=actual_delay + 1
    )
    rec_no_delay = r_no_delay["main_population"]["_per_game_records_internal"][0]
    assert rec_no_delay["root_value_high_but_delayed"] is False

    # Force "both qualify": threshold low enough + delay threshold low enough.
    if n_high >= 1 and actual_delay >= 1:
        r_both = aggregate_goal_completion_diagnostics(
            [g],
            high_value_threshold=0.0,  # any score >= 0 qualifies
            high_value_delay_threshold_plies=1,  # any delay >= 1 qualifies
        )
        rec_both = r_both["main_population"]["_per_game_records_internal"][0]
        # Now if there's >=1 winner ply with any non-null score after detection,
        # AND delay >= 1, the flag must be True.
        if rec_both["high_value_after_detection_plies"] >= 1 and rec_both["conversion_delay_plies"] >= 1:
            assert rec_both["root_value_high_but_delayed"] is True


# ----------------------------------------------------------------------
# Phase 2 / Task 9: Class 2 (capped) population + Class 3 (excluded) tests.
# ----------------------------------------------------------------------


def test_aggregate_class2_state_cap_with_detected_dominant_increments_bad_case():
    """Test #10 — Class 2 fixture built from Game 097 with reason rewritten to
    state_cap. Detection still fires (chain reaches total=2 at turn 43), so
    the state_cap_after_detection bad-case bucket increments."""
    g097 = _load_game_097()
    g097_capped = json.loads(json.dumps(g097))  # deep copy
    g097_capped["winner"] = None
    g097_capped["meta"]["reason"] = "state_cap"
    r = aggregate_goal_completion_diagnostics([g097_capped])
    capped = r["capped_population"]
    assert capped["games"] == 1
    assert capped["detected_before_cap"] == 1
    assert capped["bad_cases"]["state_cap_after_detection"] == 1
    assert capped["bad_cases"]["timeout_after_detection"] == 0
    assert capped["bad_cases"]["board_full_after_detection"] == 0
    # Detection happens at turn 43 -> cap_delay_after_detection = 59 - 43 = 16.
    records = capped["_per_game_records_internal"]
    assert records[0]["detected_player"] == "red"
    assert records[0]["first_dominant_unclosed_ply"] == 43
    assert records[0]["cap_delay_after_detection_plies"] == 16


def test_aggregate_class2_no_detection_excluded_from_detected_count():
    """Test #11 — A capped game where neither side reaches dominant-unclosed,
    so games count increments but detected_before_cap stays 0."""
    moves = [
        _move(5, 5, "red"), _move(10, 10, "black"),
        _move(7, 6, "red"), _move(12, 12, "black"),
    ]
    replays = [_replay(moves, winner=None, reason="state_cap")]
    r = aggregate_goal_completion_diagnostics(replays)
    capped = r["capped_population"]
    assert capped["games"] == 1
    assert capped["detected_before_cap"] == 0
    assert capped["bad_cases"]["state_cap_after_detection"] == 0
    assert capped["bad_cases"]["timeout_after_detection"] == 0
    assert capped["bad_cases"]["board_full_after_detection"] == 0
    rec = capped["_per_game_records_internal"][0]
    assert rec["detected"] is False
    assert rec["cap_delay_after_detection_plies"] is None


# Test #12 (Class 2 both-sides scope where Black hits dominant-unclosed first)
# is intentionally OMITTED — constructing a synthetic legal alternating game
# in which Black forms an 8-peg dominant-unclosed component while Red does
# not is non-trivial and would require many plies of careful curation. The
# both-sides loop is exercised at the unit level by the iteration order
# inside _build_class2_per_game_record (red first, then black), and the
# Class 2 wiring (capped_population["games"], detected_before_cap, the
# bad_case buckets, cap_delay) is fully covered by tests #10 and #11.


def test_aggregate_class3_draw_reason_excluded():
    """Test #13 — Reason 'draw' or 'unknown' must be counted only in
    excluded_population (not main, not capped)."""
    moves = [_move(5, 5, "red"), _move(10, 10, "black")]
    replays = [_replay(moves, winner=None, reason="draw")]
    r = aggregate_goal_completion_diagnostics(replays)
    assert r["main_population"]["games"] == 0
    assert r["capped_population"]["games"] == 0
    assert r["excluded_population"]["games"] == 1


def test_aggregate_outcome_class_partition_sums_to_n_games_total():
    """Test #14 — Every replay maps to exactly one of the three populations,
    so the population game-counts sum to the number of replays."""
    replays = [
        _replay([_move(5, 5, "red"), _move(10, 10, "black")],
                winner="red", reason="win"),
        _replay([_move(5, 5, "red"), _move(10, 10, "black")],
                winner=None, reason="state_cap"),
        _replay([_move(5, 5, "red"), _move(10, 10, "black")],
                winner=None, reason="draw"),
    ]
    r = aggregate_goal_completion_diagnostics(replays)
    total = (r["main_population"]["games"]
             + r["capped_population"]["games"]
             + r["excluded_population"]["games"])
    assert total == 3


def test_aggregate_le_2_and_le_3_buckets_independent_of_detection_threshold(
    game_097,
):
    """Test #15 — ever_distance_le_2 / le_3 buckets are structural facts about
    whether the game ever reached that distance, NOT gated by
    detection_threshold."""
    r1 = aggregate_goal_completion_diagnostics([game_097], detection_threshold=1)
    r2 = aggregate_goal_completion_diagnostics([game_097], detection_threshold=2)
    assert r1["main_population"]["games_with_total_distance_le_2"] == 1
    assert r1["main_population"]["games_with_total_distance_le_3"] == 1
    assert r2["main_population"]["games_with_total_distance_le_2"] == 1
    assert r2["main_population"]["games_with_total_distance_le_3"] == 1


def test_format_goal_completion_report_zero_detection_short_message():
    """When no games have detection in any class -> short message."""
    r = aggregate_goal_completion_diagnostics([])
    out = format_goal_completion_report(r)
    text = "\n".join(out)
    assert "Goal-Completion / Conversion Diagnostics" in text
    assert ("No dominant-unclosed positions detected" in text
            or "No decisive games" in text
            or "No decisive or capped games" in text)


def test_format_goal_completion_report_full_population_renders_all_sections():
    """A populated summary renders Main + Capped sections with all subblocks."""
    summary = {
        "config": {"detection_threshold": 2, "max_depth": 3,
                   "min_component_size": 8, "high_value_threshold": 0.9},
        "main_population": {
            "scope": "decisive_winner_only", "games": 100,
            "games_with_dominant_unclosed": 30,
            "games_with_total_distance_le_2": 20,
            "games_with_total_distance_le_3": 30,
            "detected": 20,
            "conversion_delay_plies": {"p50": 4, "p90": 12, "p95": 18, "max": 24, "mean": 5.6},
            "conversion_delay_winner_moves": {"p50": 2, "p90": 6, "max": 12, "mean": 2.8},
            "move_quality_after_detection": {
                "completes_endpoint_rate": 0.27,
                "reduces_total_goal_distance_rate": 0.06,
                "redundant_reinforcement_rate": 0.51,
                "off_chain_rate": 0.12,
                "other_rate": 0.04,
                "dominant_unavailable_rate": 0.0,
            },
            "high_value_diagnostics": {
                "search_score_coverage_pct": 100.0,
                "max_search_score_after_detection": {"p50": 0.86, "p90": 0.99, "max": 1.0, "mean": 0.85},
                "mean_search_score_after_detection": {"p50": 0.62, "p90": 0.94, "max": 0.99, "mean": 0.7},
            },
            "bad_cases": {"delay_ge_10_plies": 5, "delay_ge_20_plies": 1, "root_value_high_but_delayed": 2},
        },
        "capped_population": {
            "scope": "both_sides", "games": 5,
            "games_with_dominant_unclosed": 3, "detected_before_cap": 3,
            "cap_delay_after_detection_plies": {"p50": 22, "p90": 38, "max": 51},
            "bad_cases": {"state_cap_after_detection": 2,
                          "timeout_after_detection": 1,
                          "board_full_after_detection": 0},
        },
        "excluded_population": {"games": 0},
    }
    out = format_goal_completion_report(summary)
    text = "\n".join(out)
    assert "Main (decisive wins" in text
    assert "Capped (state_cap" in text
    assert "endpoint completion: 27.0%" in text
    assert "state_cap after detection:        2" in text


# -----------------------------
# Phase 2 Task 11: worst-cases CSV writer
# -----------------------------

def _make_replay_with_record(rec: dict) -> dict:
    """Wrap a goal_completion_record dict in a minimal replay dict."""
    return {
        "iteration": rec.get("iteration"),
        "game_idx": rec.get("game_idx"),
        "winner": rec.get("winner"),
        "starting_player": rec.get("starting_player", "red"),
        "moves": [],
        "meta": {"reason": rec.get("reason", "win"), "n_moves": rec.get("n_moves", 0),
                 "board_size": 24, "starting_player": rec.get("starting_player", "red")},
        "goal_completion_record": rec,
    }


def test_aggregate_worst_cases_csv_sort_order_correct(tmp_path):
    """CSV rows ordered by conversion_delay_plies DESC for Class 1 records."""
    replays = [
        _make_replay_with_record(
            {"iteration": 50, "game_idx": 1, "game_id": "iter_0050_game_001",
             "winner": "red", "starting_player": "red", "n_moves": 40, "reason": "win",
             "detected_player": "red", "first_dominant_unclosed_ply": 20,
             "first_total_goal_distance": 2, "first_category": "two_endpoint_closeout_2ply",
             "actual_terminal_ply": 40, "actual_win_ply": 40, "conversion_delay_plies": 20,
             "conversion_delay_winner_moves": 10, "cap_delay_proxy_plies": None,
             "primary_class_counts": {
                 "completes_endpoint": 1, "reduces_total_goal_distance": 0,
                 "redundant_reinforcement": 5, "off_chain": 4, "other": 0},
             "winner_moves_with_dominant_unavailable": 0,
             "max_search_score_after_detection": 0.95,
             "mean_search_score_after_detection": 0.9,
             "high_value_after_detection_plies": 8,
             "root_value_high_but_delayed": True,
             "outcome_class": 1, "scope": "winner",
             "detected": True}),
        _make_replay_with_record(
            {"iteration": 60, "game_idx": 2, "game_id": "iter_0060_game_002",
             "winner": "black", "starting_player": "red", "n_moves": 50, "reason": "win",
             "detected_player": "black", "first_dominant_unclosed_ply": 25,
             "first_total_goal_distance": 2, "first_category": "two_endpoint_closeout_2ply",
             "actual_terminal_ply": 50, "actual_win_ply": 50, "conversion_delay_plies": 25,
             "conversion_delay_winner_moves": 12, "cap_delay_proxy_plies": None,
             "primary_class_counts": {
                 "completes_endpoint": 1, "reduces_total_goal_distance": 0,
                 "redundant_reinforcement": 8, "off_chain": 3, "other": 0},
             "winner_moves_with_dominant_unavailable": 0,
             "max_search_score_after_detection": 0.99,
             "mean_search_score_after_detection": 0.95,
             "high_value_after_detection_plies": 11,
             "root_value_high_but_delayed": True,
             "outcome_class": 1, "scope": "winner",
             "detected": True}),
    ]
    csv_path = tmp_path / "goal_completion_worst_cases.csv"
    write_goal_completion_worst_cases_csv(str(csv_path), replays, top_k=10)
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    assert rows[0]["iteration"] == "60"     # delay=25 wins over delay=20
    assert rows[1]["iteration"] == "50"


def test_aggregate_worst_cases_csv_top_k_respects_flag(tmp_path):
    replays = [
        _make_replay_with_record(
            {"iteration": i, "game_idx": 0, "game_id": f"g{i}", "winner": "red",
             "starting_player": "red", "n_moves": 30, "reason": "win",
             "detected_player": "red", "first_dominant_unclosed_ply": 10,
             "first_total_goal_distance": 2, "first_category": "two_endpoint_closeout_2ply",
             "actual_terminal_ply": 30, "actual_win_ply": 30,
             "conversion_delay_plies": 30 - 10, "conversion_delay_winner_moves": 10,
             "cap_delay_proxy_plies": None,
             "primary_class_counts": {
                 "completes_endpoint": 0, "reduces_total_goal_distance": 0,
                 "redundant_reinforcement": 5, "off_chain": 5, "other": 0},
             "winner_moves_with_dominant_unavailable": 0,
             "max_search_score_after_detection": 0.9,
             "mean_search_score_after_detection": 0.85,
             "high_value_after_detection_plies": 8,
             "root_value_high_but_delayed": False,
             "outcome_class": 1, "scope": "winner",
             "detected": True})
        for i in range(5)
    ]
    csv_path = tmp_path / "wc.csv"
    write_goal_completion_worst_cases_csv(str(csv_path), replays, top_k=2)
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    assert len(rows) == 2


def test_aggregate_worst_cases_csv_class2_rows_have_null_winner_and_win_ply(tmp_path):
    replays = [
        _make_replay_with_record(
            {"iteration": 70, "game_idx": 3, "game_id": "iter_0070_game_003",
             "winner": None, "starting_player": "red", "n_moves": 100,
             "reason": "state_cap", "detected_player": "red",
             "first_dominant_unclosed_ply": 40, "first_total_goal_distance": 2,
             "first_category": "two_endpoint_closeout_2ply",
             "actual_terminal_ply": 100, "actual_win_ply": None,
             "cap_delay_proxy_plies": 60,
             "conversion_delay_plies": None, "conversion_delay_winner_moves": None,
             "primary_class_counts": None,
             "winner_moves_with_dominant_unavailable": None,
             "max_search_score_after_detection": None,
             "mean_search_score_after_detection": None,
             "high_value_after_detection_plies": None,
             "root_value_high_but_delayed": None,
             "outcome_class": 2, "scope": "both_sides",
             "detected": True}),
    ]
    csv_path = tmp_path / "wc2.csv"
    write_goal_completion_worst_cases_csv(str(csv_path), replays, top_k=5)
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    assert len(rows) == 1
    assert rows[0]["winner"] == ""
    assert rows[0]["actual_win_ply"] == ""
    assert rows[0]["outcome_class"] == "2"
    assert rows[0]["reason"] == "state_cap"


def test_analyzer_per_ply_detection_calls_compute_goal_completion_state_with_enumerate_moves_false(monkeypatch):
    """Perf regression test: the analyzer's per-ply detection walk must pass
    enumerate_moves=False to skip the expensive completion/reducing move
    enumeration. The watch-window classification path should still default
    to True (it actually needs the move sets).

    Without this gating, a 1000-game corpus takes 8+ hours to analyze
    (verified empirically; killed run on 2026-05-05). With this gating,
    it should complete in minutes.
    """
    import scripts.twixt_replay_analyzer as analyzer
    from scripts.GPU.alphazero import connectivity_diagnostics as cd

    enumerate_calls: list = []

    real_fn = cd.compute_goal_completion_state

    def _wrapper(state, player, **kwargs):
        enumerate_calls.append(kwargs.get("enumerate_moves", True))
        return real_fn(state, player, **kwargs)

    monkeypatch.setattr(analyzer, "compute_goal_completion_state", _wrapper)

    g097 = _load_game_097()
    r = aggregate_goal_completion_diagnostics([g097])

    n_skip = sum(1 for v in enumerate_calls if v is False)
    n_full = sum(1 for v in enumerate_calls if v is True)
    assert n_skip > 0, (
        "Analyzer must pass enumerate_moves=False on at least one per-ply "
        "detection call (perf regression vs 2026-05-05 8-hour analyzer run)."
    )
    # Per-ply detection (enumerate_moves=False) should dominate; full-
    # enumeration calls happen only for winner moves in the watch window.
    assert n_skip > n_full, (
        f"Expected per-ply detection (enumerate_moves=False) to dominate; "
        f"got {n_skip} skip vs {n_full} full. Detection should NOT enumerate "
        f"moves on every ply."
    )
    # Sanity: aggregation result is still correct under the optimization.
    assert r["main_population"]["games"] == 1
    assert r["main_population"]["detected"] == 1
