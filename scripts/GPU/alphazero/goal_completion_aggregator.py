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
        "main_population": _summarize_main_population(main, config),
        "capped_population": _summarize_capped_population(capped),
        "excluded_population": {"n": len(excluded), "games": len(excluded)},
    }


def _percentile(sorted_values: list, p: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _stats_block(values: list) -> dict:
    if not values:
        return {"p10": 0, "p50": 0, "p90": 0, "p95": 0, "max": 0, "mean": 0.0, "min": 0}
    s = sorted(values)
    return {
        "p10": _percentile(s, 10),
        "p50": _percentile(s, 50),
        "p90": _percentile(s, 90),
        "p95": _percentile(s, 95),
        "max": float(s[-1]),
        "min": float(s[0]),
        "mean": float(sum(s) / len(s)),
    }


def _summarize_main_population(main: List[dict], config: dict) -> dict:
    if not main:
        return {"n": 0}

    detected = [r for r in main if r["detected"]]
    n = len(main)

    delays = [r["conversion_delay_plies"] for r in detected
              if r["conversion_delay_plies"] is not None]
    delays_winner_moves = [r["conversion_delay_winner_moves"] for r in detected
                           if r["conversion_delay_winner_moves"] is not None]

    # Pooled primary_class counts -> rates (new style: single total denominator).
    pooled = _zero_class_counts()
    pooled["dominant_unavailable"] = 0
    total_classified = 0
    for r in detected:
        pcc = r.get("primary_class_counts") or {}
        for k in ("completes_endpoint", "reduces_total_goal_distance",
                  "redundant_reinforcement", "off_chain", "other"):
            pooled[k] += int(pcc.get(k, 0))
            total_classified += int(pcc.get(k, 0))
        # dominant_unavailable comes from winner_moves_with_dominant_unavailable
        pooled["dominant_unavailable"] += int(r.get("winner_moves_with_dominant_unavailable") or 0)
        total_classified += int(r.get("winner_moves_with_dominant_unavailable") or 0)
    if total_classified > 0:
        primary_class_rates = {k: v / total_classified for k, v in pooled.items()}
    else:
        primary_class_rates = {k: 0.0 for k in pooled}

    # Legacy denominator semantics for move_quality_after_detection:
    # 5 base classes use pooled_with_component; dominant_unavailable uses
    # pooled_with_component + pooled_unavailable.
    pooled_with_component = sum(
        pooled[k] for k in ("completes_endpoint", "reduces_total_goal_distance",
                             "redundant_reinforcement", "off_chain", "other")
    )
    pooled_unavailable = pooled["dominant_unavailable"]

    if pooled_with_component > 0:
        move_quality_after_detection = {
            "completes_endpoint_rate": pooled["completes_endpoint"] / pooled_with_component,
            "reduces_total_goal_distance_rate": pooled["reduces_total_goal_distance"] / pooled_with_component,
            "redundant_reinforcement_rate": pooled["redundant_reinforcement"] / pooled_with_component,
            "off_chain_rate": pooled["off_chain"] / pooled_with_component,
            "other_rate": pooled["other"] / pooled_with_component,
            "dominant_unavailable_rate": (
                pooled_unavailable / (pooled_with_component + pooled_unavailable)
                if (pooled_with_component + pooled_unavailable) > 0 else 0.0
            ),
        }
    else:
        move_quality_after_detection = None

    # search_score_after_detection (nested: max distribution + mean distribution).
    max_scores = [r["max_search_score_after_detection"] for r in detected
                  if r.get("max_search_score_after_detection") is not None]
    mean_scores = [r["mean_search_score_after_detection"] for r in detected
                   if r.get("mean_search_score_after_detection") is not None]

    # Legacy high_value_diagnostics block (format_goal_completion_report reads this).
    coverage_records = [
        r for r in detected
        if int(r.get("search_score_coverage_in_watch_window") or 0) > 0
    ]
    if detected and coverage_records:
        max_scores_only = [r["max_search_score_after_detection"] for r in coverage_records
                           if r.get("max_search_score_after_detection") is not None]
        mean_scores_only = [r["mean_search_score_after_detection"] for r in coverage_records
                            if r.get("mean_search_score_after_detection") is not None]
        coverage_pct = 100.0 * len(coverage_records) / len(detected)
        high_value_diagnostics = {
            "search_score_coverage_pct": coverage_pct,
            "max_search_score_after_detection": {
                "p50": _percentile(sorted(max_scores_only), 50),
                "p90": _percentile(sorted(max_scores_only), 90),
                "max": float(max(max_scores_only)) if max_scores_only else 0.0,
            },
            "mean_search_score_after_detection": {
                "p50": _percentile(sorted(mean_scores_only), 50),
                "p90": _percentile(sorted(mean_scores_only), 90),
                "max": float(max(mean_scores_only)) if mean_scores_only else 0.0,
            },
        }
    elif detected:
        high_value_diagnostics = {
            "search_score_coverage_pct": 0.0,
            "max_search_score_after_detection": None,
            "mean_search_score_after_detection": None,
        }
    else:
        high_value_diagnostics = None

    bad = {
        "delay_ge_10_plies": sum(1 for d in delays if d >= 10),
        "delay_ge_20_plies": sum(1 for d in delays if d >= 20),
        "high_value_after_detection_plies_total": sum(
            int(r.get("high_value_after_detection_plies") or 0) for r in detected
        ),
        "root_value_high_but_delayed": sum(
            1 for r in detected if r.get("root_value_high_but_delayed") is True
        ),
    }

    return {
        "scope": "decisive_winner_only",
        "n": n,
        "games": n,                                   # legacy alias
        "games_with_dominant_unclosed": sum(1 for r in main if r["detected"]),
        "games_with_total_distance_le_2": sum(1 for r in main if r.get("ever_distance_le_2")),
        "games_with_total_distance_le_3": sum(1 for r in main if r.get("ever_distance_le_3")),
        "detected": len(detected),
        "detection_rate": (len(detected) / n) if n else 0.0,
        "min_total_goal_distance": _stats_block(
            [r["min_total_goal_distance"] for r in main if r.get("min_total_goal_distance") is not None]
        ),
        "conversion_delay_plies": _stats_block(delays),
        "conversion_delay_winner_moves": _stats_block(delays_winner_moves),
        "primary_class_rates": primary_class_rates,           # new
        "move_quality_after_detection": move_quality_after_detection,  # legacy
        "search_score_after_detection": {                     # new
            "max": _stats_block(max_scores),
            "mean": _stats_block(mean_scores),
        },
        "high_value_diagnostics": high_value_diagnostics,    # legacy
        "bad_cases": bad,
    }


def _summarize_capped_population(capped: List[dict]) -> dict:
    if not capped:
        return {"n": 0}
    detected = [r for r in capped if r["detected"]]
    proxies = [r["cap_delay_proxy_plies"] for r in detected
               if r.get("cap_delay_proxy_plies") is not None]
    side_counts = {"red": 0, "black": 0}
    for r in detected:
        s = r.get("detected_player")
        if s in side_counts:
            side_counts[s] += 1
    bad_cases = {
        "state_cap_after_detection": sum(1 for r in detected if r.get("reason") == "state_cap"),
        "timeout_after_detection": sum(
            1 for r in detected if r.get("reason") in ("timeout", "timeout_selfplay")
        ),
        "board_full_after_detection": sum(1 for r in detected if r.get("reason") == "board_full"),
    }
    return {
        "n": len(capped),
        "games": len(capped),                              # legacy alias
        "detected": len(detected),
        "detected_before_cap": len(detected),              # legacy alias
        "cap_delay_proxy_plies": _stats_block(proxies),   # new
        "cap_delay_after_detection_plies": _stats_block(proxies),  # legacy alias (same block)
        "first_detector_side": side_counts,
        "bad_cases": bad_cases,
    }
