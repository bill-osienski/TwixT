"""Shared aggregator: per-game goal_completion_record list ->
goal_completion_summary block.

Pure functions, no I/O, no BFS. Used by both the trainer (per-iteration
sidecar) and the analyzer (cross-iteration roll-up).
"""
from __future__ import annotations

from typing import List, Optional


def _zero_class_counts() -> dict:
    return {
        "completes_endpoint": 0,
        "reduces_total_goal_distance": 0,
        "redundant_reinforcement": 0,
        "off_chain": 0,
        "other": 0,
    }


def _normalize_record(r: dict) -> dict:
    """Forward/backward-tolerant normalization at function boundary."""
    pcc = r.get("primary_class_counts")
    if pcc is None:
        pcc = _zero_class_counts()
    return {
        "version": int(r.get("version", 1)),
        "outcome_class": int(r.get("outcome_class", 3)),
        "reason": r.get("reason") or "unknown",
        "winner": r.get("winner"),
        "detected_player": r.get("detected_player"),
        "detected": bool(r.get("detected", False)),
        "ever_distance_le_2": bool(r.get("ever_distance_le_2", False)),
        "ever_distance_le_3": bool(r.get("ever_distance_le_3", False)),
        "min_total_goal_distance": r.get("min_total_goal_distance"),
        "first_dominant_unclosed_ply": r.get("first_dominant_unclosed_ply"),
        "first_total_goal_distance": r.get("first_total_goal_distance"),
        "first_category": r.get("first_category"),
        "actual_terminal_ply": r.get("actual_terminal_ply"),
        "actual_win_ply": r.get("actual_win_ply"),
        "conversion_delay_plies": r.get("conversion_delay_plies"),
        "conversion_delay_winner_moves": r.get("conversion_delay_winner_moves"),
        "cap_delay_proxy_plies": r.get("cap_delay_proxy_plies"),
        "primary_class_counts": pcc,
        "max_search_score_after_detection": r.get("max_search_score_after_detection"),
        "mean_search_score_after_detection": r.get("mean_search_score_after_detection"),
        "high_value_after_detection_plies": r.get("high_value_after_detection_plies"),
        "root_value_high_but_delayed": r.get("root_value_high_but_delayed"),
        "winner_moves_in_watch_window": r.get("winner_moves_in_watch_window"),
        "winner_moves_with_dominant_component": r.get("winner_moves_with_dominant_component"),
        "winner_moves_with_dominant_unavailable": r.get("winner_moves_with_dominant_unavailable"),
        "search_score_coverage_in_watch_window": r.get("search_score_coverage_in_watch_window"),
    }


def aggregate_goal_completion_records(
    records: List[Optional[dict]],
    config: dict,
    games_total: Optional[int] = None,
) -> dict:
    games_total = games_total if games_total is not None else len(records)
    valid = [_normalize_record(r) for r in records if r is not None]

    main = [r for r in valid if r["outcome_class"] == 1]
    capped = [r for r in valid if r["outcome_class"] == 2]
    excluded = [r for r in valid if r["outcome_class"] == 3]

    return {
        "version": 1,
        "config": dict(config),
        "diagnostics_coverage": {
            "games_total": games_total,
            "games_with_record": len(valid),
            "coverage_rate": (len(valid) / games_total) if games_total else 0.0,
            "games_class_1": len(main),
            "games_class_2": len(capped),
            "games_class_3": len(excluded),
        },
        # Population summaries land in Task 5; emit empty skeletons here.
        "main_population": {"n": len(main)},
        "capped_population": {"n": len(capped)},
        "excluded_population": {"n": len(excluded)},
    }
