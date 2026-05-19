"""Long-tail bucket classifier.

Assigns each long-tail goal-completion game (delay>=20 / state_cap /
dom_unavail>=20 / n_moves==280) to exactly one of five mutually exclusive
failure buckets by priority order, so the next-action decision can be
driven by bucket counts instead of manual triage.

Spec: docs/superpowers/specs/2026-05-19-long-tail-bucket-classifier-design.md
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Tuple


LONG_TAIL_BUCKETS = (
    "marathon_or_state_cap",
    "dominant_unavailable_contested",
    "td3_drift",
    "td2_alt_in_top5",
    "td2_reducer_buried",
    "unclassified",
)

NOT_LONG_TAIL = "not_long_tail"


# Filter thresholds (spec §3).
_DELAY_THRESHOLD = 20
_DOM_UNAVAIL_FILTER_THRESHOLD = 20
_DOM_UNAVAIL_CONTESTED_THRESHOLD = 10  # priority 2 bucket (spec §2.2)
_N_MOVES_CAP = 280
_TD3_THRESHOLD = 3
_REDUND_PICKS_TOP5_ALT_MIN_SHARE = 0.50


def matches_long_tail_filter(record: dict) -> bool:
    """Whether a game qualifies for long-tail classification."""
    if int(record.get("conversion_delay_plies") or 0) >= _DELAY_THRESHOLD:
        return True
    if record.get("reason") == "state_cap":
        return True
    if int(record.get("winner_moves_with_dominant_unavailable") or 0) >= _DOM_UNAVAIL_FILTER_THRESHOLD:
        return True
    if int(record.get("n_moves") or 0) == _N_MOVES_CAP:
        return True
    return False


def _redundant_pick_has_top5_alt(entry: dict) -> bool:
    """A redundant_reinforcement ply has a top-5 alternative if either
    the endpoint_completion or distance_reducing ranking is in visit top-5."""
    ec = entry.get("endpoint_completion_ranking") or {}
    dr = entry.get("distance_reducing_ranking") or {}
    return bool(ec.get("any_in_visit_top5")) or bool(dr.get("any_in_visit_top5"))


def _td2_subclass(diagnostics: list) -> str:
    """Subclassify a td=2 game by inspecting its redundant_reinforcement plies.

    Returns 'td2_alt_in_top5' when >= 50% of redundant-pick plies had an
    endpoint or distance-reducing alternative in visit top-5; otherwise
    'td2_reducer_buried'. If no redundant-pick plies exist, returns
    'td2_reducer_buried' (defensive — no evidence of usable alternatives).
    """
    redund = [
        e for e in (diagnostics or [])
        if (e.get("selected_move_classification") or {}).get("primary_class") == "redundant_reinforcement"
    ]
    if not redund:
        return "td2_reducer_buried"
    with_alt = sum(1 for e in redund if _redundant_pick_has_top5_alt(e))
    share = with_alt / len(redund)
    return "td2_alt_in_top5" if share >= _REDUND_PICKS_TOP5_ALT_MIN_SHARE else "td2_reducer_buried"


def classify_long_tail_bucket(record: dict, diagnostics: list) -> str:
    """Return the long-tail bucket for a single game. See spec §2 for the
    priority order. Returns NOT_LONG_TAIL when the game does not match the
    long-tail filter."""
    if not matches_long_tail_filter(record):
        return NOT_LONG_TAIL

    # Priority 1: marathon / state-cap (covers state_cap + 280-ply wins).
    if record.get("reason") == "state_cap" or int(record.get("n_moves") or 0) == _N_MOVES_CAP:
        return "marathon_or_state_cap"

    # Priority 2: contested chain.
    if int(record.get("winner_moves_with_dominant_unavailable") or 0) >= _DOM_UNAVAIL_CONTESTED_THRESHOLD:
        return "dominant_unavailable_contested"

    # Priority 3: td>=3 drift.
    if int(record.get("first_total_goal_distance") or 0) >= _TD3_THRESHOLD:
        return "td3_drift"

    # Priorities 4 + 5: td==2 subclassification by visit-top-5 alternative share.
    if int(record.get("first_total_goal_distance") or 0) == 2:
        return _td2_subclass(diagnostics)

    # Fallback (e.g., td==1 with long delay — defensive).
    return "unclassified"


def aggregate_long_tail_buckets(
    records_with_diagnostics: Iterable[Tuple[dict, list]],
) -> dict:
    """Aggregate per-game classifications into a per-iter + range table.

    records_with_diagnostics: iterable of (record, diagnostics) tuples.

    Returns a dict with keys per spec §5: per_iter, range_total,
    per_iter_shares, range_total_shares, total_long_tail_games_per_iter,
    total_long_tail_games_range.
    """
    per_iter_counts: dict = defaultdict(lambda: {b: 0 for b in LONG_TAIL_BUCKETS})
    per_iter_totals: dict = defaultdict(int)

    for record, diagnostics in records_with_diagnostics:
        bucket = classify_long_tail_bucket(record, diagnostics)
        if bucket == NOT_LONG_TAIL:
            continue
        it = record.get("iteration")
        per_iter_counts[it][bucket] += 1
        per_iter_totals[it] += 1

    # Range-level totals.
    range_total = {b: 0 for b in LONG_TAIL_BUCKETS}
    for counts in per_iter_counts.values():
        for b, c in counts.items():
            range_total[b] += c
    total_range = sum(range_total.values())

    # Shares.
    per_iter_shares: dict = {}
    for it, counts in per_iter_counts.items():
        denom = per_iter_totals[it] or 1
        per_iter_shares[it] = {b: round(counts[b] / denom, 3) for b in LONG_TAIL_BUCKETS}
    range_total_shares = {
        b: round(range_total[b] / (total_range or 1), 3) for b in LONG_TAIL_BUCKETS
    }

    return {
        "per_iter": {it: dict(counts) for it, counts in per_iter_counts.items()},
        "range_total": range_total,
        "per_iter_shares": per_iter_shares,
        "range_total_shares": range_total_shares,
        "total_long_tail_games_per_iter": dict(per_iter_totals),
        "total_long_tail_games_range": total_range,
    }
