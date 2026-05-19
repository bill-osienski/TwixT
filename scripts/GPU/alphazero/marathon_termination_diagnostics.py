"""Marathon-termination diagnostics. Pure-function analyzer surface
that computes no-progress windows, adjudication-coverage gate-block
distributions, resign-gate breakdown by game-length partition, and the
value-uncertain guard predicate used by termination knobs.

Spec: docs/superpowers/specs/2026-05-19-marathon-termination-tuning-design.md

All inputs are per-game data already on disk:
  - per-game `goal_completion_record` (dict)
  - per-game `goal_completion_diagnostics` (list of per-ply entries)
  - per-game `meta` (dict; carries adjudication_block_reason from
    self_play.py:1248-1249 via game_saver.py:146)

No self-play change required (Task 0 outcome A).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Optional, Tuple


NO_PROGRESS_WINDOW_SIZE = 15

# §3.1 — local-structural primary_class vocabulary for the no-progress
# window detector. Matches Spec 4's primary_class names + the
# goal_completion classifier vocabulary (completes_endpoint /
# reduces_total_goal_distance / redundant_reinforcement / off_chain
# from self_play._classify_argmax_against_gc, and the Spec 4 broader
# vocabulary including blocks_opponent_closeout).
_LOCAL_STRUCTURAL_CLASSES = frozenset({
    "redundant_reinforcement",
    "off_chain",
    "connects_to_existing_component",
    "improves_own_largest_component",
})


def _is_local_structural(entry: dict) -> bool:
    cls = (entry.get("selected_move_classification") or {}).get("primary_class")
    return cls in _LOCAL_STRUCTURAL_CLASSES


def _own_distance_reduced(entry: dict) -> bool:
    cls_info = entry.get("selected_move_classification") or {}
    before = cls_info.get("total_goal_distance_before")
    after = cls_info.get("total_goal_distance_after")
    if before is None or after is None:
        return False
    return int(after) < int(before)


def _completed_endpoint(entry: dict) -> bool:
    cls = (entry.get("selected_move_classification") or {}).get("primary_class")
    return cls == "completes_endpoint"


def _blocked_opponent(entry: dict) -> bool:
    """Opponent block (Spec §3.1, definition pinned).

    Primary signal: `selected_move_classification.primary_class ==
    "blocks_opponent_closeout"` (Spec 4 defense-classifier output).

    This is the SHARED definition with the recovery-retargeting
    diagnostic — both consult the same classifier output via
    `primary_class`, so the two diagnostics cannot diverge on what
    counts as a block.
    """
    cls = (entry.get("selected_move_classification") or {}).get("primary_class")
    return cls == "blocks_opponent_closeout"


def detect_no_progress_windows(diagnostics: list, *, side: str) -> int:
    """Count distinct sliding-window no-progress runs for `side`.

    A no-progress window of length NO_PROGRESS_WINDOW_SIZE (=15) ends at
    own-move ply t iff the trailing 15 own-moves for this side all satisfy:
      - moves are local-structural (primary_class in the structural set)
      - no own-move reduced own total_goal_distance
      - no own-move completed an endpoint
      - no own-move was an opponent block

    Overlapping windows anchored at distinct end-plies count separately
    only if there's at least 1 own-move gap between their end-plies;
    otherwise they collapse into the longest contiguous run.

    Concretely: count maximal-length runs of consecutive own-moves
    satisfying the four conditions, then sum floor(run_length / 15) for
    each run.
    """
    own_entries = [e for e in (diagnostics or []) if e.get("side_to_move") == side]
    if len(own_entries) < NO_PROGRESS_WINDOW_SIZE:
        return 0

    def is_no_progress(e: dict) -> bool:
        return (
            _is_local_structural(e)
            and not _own_distance_reduced(e)
            and not _completed_endpoint(e)
            and not _blocked_opponent(e)
        )

    windows = 0
    run_len = 0
    for e in own_entries:
        if is_no_progress(e):
            run_len += 1
        else:
            windows += run_len // NO_PROGRESS_WINDOW_SIZE
            run_len = 0
    windows += run_len // NO_PROGRESS_WINDOW_SIZE
    return windows


ADJUDICATION_GATE_BUCKETS = (
    "not_attempted",
    "value_below_threshold",
    "min_top1_share",
    "min_visits",
    "missing_signal",
    "would_have_passed",
)


# Mapping: self_play.py's adj_blocked_by value -> spec §3.2 bucket name.
# Source: self_play.py:1213-1222 (deterministic first-failure label).
_BLOCKED_BY_TO_BUCKET = {
    "ply":       "not_attempted",
    "threshold": "value_below_threshold",
    "top1":      "min_top1_share",
    "visits":    "min_visits",
}


def classify_adjudication_coverage(
    record: dict, meta: dict, diagnostics: list,
) -> Optional[str]:
    """Classify which gate blocked adjudication for a state_cap game.

    Returns one of ADJUDICATION_GATE_BUCKETS, or None if the game does
    not qualify for §3.2 (not a state_cap game).

    Inputs (all already on disk; Task 0 outcome A):
      record: per-game goal_completion_record
      meta:   per-game `meta` dict (contains adjudication_block_reason
              from game_saver.py:146, populated by self_play.py:1248-1249)
      diagnostics: per-game goal_completion_diagnostics

    None handling matches spec §3.2 strictly:
      - key absent from meta      -> 'missing_signal' (observability gap)
      - key present, value is None -> 'would_have_passed' (bug indicator)
      - known value                -> direct mapping via _BLOCKED_BY_TO_BUCKET
    """
    meta = meta or {}
    if meta.get("reason") != "state_cap" and record.get("reason") != "state_cap":
        return None  # not in scope

    if "adjudication_block_reason" not in meta:
        # Old-format JSON written before the field was persisted.
        return "missing_signal"

    reason = meta["adjudication_block_reason"]
    if reason in _BLOCKED_BY_TO_BUCKET:
        return _BLOCKED_BY_TO_BUCKET[reason]
    if reason is None:
        # Key present but null: an attempt should have happened (adjudication
        # was on for our 220-229 launches). Treat as a bug indicator. If
        # adjudication was actually disabled in a different launch config,
        # the report MUST surface this as a caveat alongside the count.
        return "would_have_passed"
    # Unknown string value — defensive fallback.
    return "missing_signal"


GAME_LENGTH_BUCKETS = ("short", "mid", "long")


def game_length_bucket(n_moves: int) -> str:
    """§3.3 partition: short (<=100), mid (101-200), long (>200)."""
    n = int(n_moves or 0)
    if n <= 100:
        return "short"
    if n <= 200:
        return "mid"
    return "long"


def compute_resign_gate_breakdown(
    record: dict,
    diagnostics: list,
    *,
    losing_side: str,
    resign_threshold: float,
    resign_min_ply: int,
    resign_min_visits: int,
    resign_min_top1_share: float,
) -> dict:
    """Spec §3.3 resign-gate breakdown for the losing side, looking at
    its last 40 plies (plies where side_to_move == losing_side).

    Returns:
      value_hits           : count of own-plies where q_value < resign_threshold
      eligible_hits        : count of value_hits also satisfying
                             ply >= resign_min_ply AND visits >= resign_min_visits
      blocked_by_top1      : count of eligible_hits where top1_share <
                             resign_min_top1_share
      final_eval_below_thr : at the last own-ply for the loser,
                             q_value < resign_threshold
      top1_block_rate_over_value_hits     : blocked_by_top1 / max(value_hits, 1)
                                            (returns 0.0 when value_hits == 0)
      top1_block_rate_over_eligible_hits  : blocked_by_top1 / max(eligible_hits, 1)
                                            (returns 0.0 when eligible_hits == 0)
    """
    own = [e for e in (diagnostics or []) if e.get("side_to_move") == losing_side]
    own = own[-40:]  # last 40 own-plies

    value_hits = 0
    eligible_hits = 0
    blocked_by_top1 = 0
    for e in own:
        rs = e.get("root_summary") or {}
        q = rs.get("q_value")
        visits = int(rs.get("visit_count") or 0)
        top1 = e.get("root_top1_share")
        ply = int(e.get("ply") or 0)
        if q is None or q >= resign_threshold:
            continue
        value_hits += 1
        if ply >= resign_min_ply and visits >= resign_min_visits:
            eligible_hits += 1
            if top1 is not None and float(top1) < resign_min_top1_share:
                blocked_by_top1 += 1

    final_eval_below_thr = bool(
        own and (own[-1].get("root_summary") or {}).get("q_value") is not None
        and (own[-1]["root_summary"]["q_value"] < resign_threshold)
    )

    def _rate(num, denom):
        return float(num) / float(denom) if denom > 0 else 0.0

    return {
        "value_hits": value_hits,
        "eligible_hits": eligible_hits,
        "blocked_by_top1": blocked_by_top1,
        "final_eval_below_thr": final_eval_below_thr,
        "top1_block_rate_over_value_hits": _rate(blocked_by_top1, value_hits),
        "top1_block_rate_over_eligible_hits": _rate(blocked_by_top1, eligible_hits),
    }


def value_uncertain_guard(
    diagnostics: list,
    *,
    window_per_side: int = 10,
    neutral_abs_threshold: float = 0.30,
    sign_flip_min: int = 3,
) -> bool:
    """Spec §5.1 value-uncertain guard.

    Returns True (DO NOT terminate) when EITHER:
      - both sides' last `window_per_side` own-plies have
        |q_value| < neutral_abs_threshold for ALL plies in the window
      - EITHER side's last `window_per_side` own-plies show >= sign_flip_min
        sign-flips in q_value (per-side oscillation, NOT the interleaved
        natural turn-alternation)

    Returns False (termination is safe per this guard) when both sides
    show stable, non-neutral assessments.

    Per-side sign-flip semantics: a stable game where red consistently
    sees +0.85 and black consistently sees -0.85 has 0 per-side flips —
    decisive, terminate is safe. A side whose own q-value alternates
    sign across its own moves IS uncertain — do not terminate.

    Implementation note: callers in self-play / training enforce this at
    the termination call-site, not the diagnostic call-site. This
    predicate is pure and side-effect-free.
    """
    own_red = [
        (e["root_summary"] or {}).get("q_value")
        for e in (diagnostics or [])
        if e.get("side_to_move") == "red" and (e.get("root_summary") or {}).get("q_value") is not None
    ]
    own_black = [
        (e["root_summary"] or {}).get("q_value")
        for e in (diagnostics or [])
        if e.get("side_to_move") == "black" and (e.get("root_summary") or {}).get("q_value") is not None
    ]
    last_red = own_red[-window_per_side:]
    last_black = own_black[-window_per_side:]

    # Neutral both-sides condition.
    both_neutral = (
        len(last_red) >= 1
        and len(last_black) >= 1
        and all(abs(q) < neutral_abs_threshold for q in last_red)
        and all(abs(q) < neutral_abs_threshold for q in last_black)
    )
    if both_neutral:
        return True

    # Oscillatory: per-side sign-flip count over each side's own last-window.
    def _sign_flips(values: list) -> int:
        return sum(
            1 for a, b in zip(values, values[1:])
            if (a > 0 and b < 0) or (a < 0 and b > 0)
        )

    return (
        _sign_flips(last_red) >= sign_flip_min
        or _sign_flips(last_black) >= sign_flip_min
    )


def _losing_side(record: dict) -> Optional[str]:
    winner = record.get("winner")
    if winner == "red":
        return "black"
    if winner == "black":
        return "red"
    return None


def aggregate_marathon_termination(
    games: Iterable[Tuple[dict, dict, list]],
    *,
    resign_threshold: float,
    resign_min_ply: int,
    resign_min_visits: int,
    resign_min_top1_share: float,
) -> dict:
    """Aggregate per-game diagnostics into per-iter and range tables.

    games: iterable of (record, meta, diagnostics) triples.

    Returns the structure consumed by write_marathon_termination_csv +
    format_marathon_termination_report (see spec §4.2/§4.3).
    """
    def _empty_iter_row() -> dict:
        return {
            "games_total": 0,
            "state_cap_280_games": 0,
            "no_progress_window_counts": [],  # per-game ints
            "adjudication_gate_counts": {b: 0 for b in ADJUDICATION_GATE_BUCKETS},
            "resign_top1_block_over_value_hits": {b: [] for b in GAME_LENGTH_BUCKETS},
            "resign_top1_block_over_eligible_hits": {b: [] for b in GAME_LENGTH_BUCKETS},
            # Observability counters (spec §3.1 follow-up — surface coverage
            # gaps so a zero no-progress rate is not confused with a
            # no-data situation).
            "diagnostics_entries_red": 0,
            "diagnostics_entries_black": 0,
            "no_progress_observable_games_red": 0,    # games with >=15 red own-entries
            "no_progress_observable_games_black": 0,  # games with >=15 black own-entries
        }

    per_iter: dict = defaultdict(_empty_iter_row)

    for record, meta, diagnostics in games:
        it = record.get("iteration")
        row = per_iter[it]
        row["games_total"] += 1

        # Observability: count diagnostics entries by side per game.
        red_entries = sum(1 for e in (diagnostics or []) if e.get("side_to_move") == "red")
        black_entries = sum(1 for e in (diagnostics or []) if e.get("side_to_move") == "black")
        row["diagnostics_entries_red"] += red_entries
        row["diagnostics_entries_black"] += black_entries
        if red_entries >= NO_PROGRESS_WINDOW_SIZE:
            row["no_progress_observable_games_red"] += 1
        if black_entries >= NO_PROGRESS_WINDOW_SIZE:
            row["no_progress_observable_games_black"] += 1

        # §3.1 — no-progress window count summed across both sides.
        npw = (
            detect_no_progress_windows(diagnostics, side="red")
            + detect_no_progress_windows(diagnostics, side="black")
        )
        row["no_progress_window_counts"].append(npw)

        # §3.2 — only for state_cap 280-ply games.
        n_moves = int(record.get("n_moves") or 0)
        if record.get("reason") == "state_cap" and n_moves == 280:
            row["state_cap_280_games"] += 1
            bucket = classify_adjudication_coverage(record, meta, diagnostics)
            if bucket:
                row["adjudication_gate_counts"][bucket] += 1

        # §3.3 — resign-gate breakdown (only when winner is known).
        loser = _losing_side(record)
        if loser is not None:
            br = compute_resign_gate_breakdown(
                record, diagnostics,
                losing_side=loser,
                resign_threshold=resign_threshold,
                resign_min_ply=resign_min_ply,
                resign_min_visits=resign_min_visits,
                resign_min_top1_share=resign_min_top1_share,
            )
            len_bucket = game_length_bucket(record.get("n_moves") or 0)
            row["resign_top1_block_over_value_hits"][len_bucket].append(
                br["top1_block_rate_over_value_hits"]
            )
            row["resign_top1_block_over_eligible_hits"][len_bucket].append(
                br["top1_block_rate_over_eligible_hits"]
            )

    # Compute derived per-iter values.
    def _finalize(row: dict) -> dict:
        npw_list = row["no_progress_window_counts"]
        mean_npw = sum(npw_list) / len(npw_list) if npw_list else 0.0
        finalized = {
            "games_total": row["games_total"],
            "state_cap_280_games": row["state_cap_280_games"],
            "mean_no_progress_windows_per_game": round(mean_npw, 3),
            "adjudication_gate_counts": dict(row["adjudication_gate_counts"]),
            "mean_resign_top1_block_rate_over_value_hits": {
                b: round(sum(vs) / len(vs), 3) if vs else 0.0
                for b, vs in row["resign_top1_block_over_value_hits"].items()
            },
            "mean_resign_top1_block_rate_over_eligible_hits": {
                b: round(sum(vs) / len(vs), 3) if vs else 0.0
                for b, vs in row["resign_top1_block_over_eligible_hits"].items()
            },
            "observability": {
                "diagnostics_entries_red":   row["diagnostics_entries_red"],
                "diagnostics_entries_black": row["diagnostics_entries_black"],
                "no_progress_observable_games_red":   row["no_progress_observable_games_red"],
                "no_progress_observable_games_black": row["no_progress_observable_games_black"],
            },
        }
        return finalized

    per_iter_final = {it: _finalize(row) for it, row in per_iter.items()}

    # Range totals.
    range_row = _empty_iter_row()
    for it, row in per_iter.items():
        range_row["games_total"] += row["games_total"]
        range_row["state_cap_280_games"] += row["state_cap_280_games"]
        range_row["no_progress_window_counts"].extend(row["no_progress_window_counts"])
        for b, c in row["adjudication_gate_counts"].items():
            range_row["adjudication_gate_counts"][b] += c
        for b in GAME_LENGTH_BUCKETS:
            range_row["resign_top1_block_over_value_hits"][b].extend(
                row["resign_top1_block_over_value_hits"][b]
            )
            range_row["resign_top1_block_over_eligible_hits"][b].extend(
                row["resign_top1_block_over_eligible_hits"][b]
            )
        range_row["diagnostics_entries_red"]   += row["diagnostics_entries_red"]
        range_row["diagnostics_entries_black"] += row["diagnostics_entries_black"]
        range_row["no_progress_observable_games_red"]   += row["no_progress_observable_games_red"]
        range_row["no_progress_observable_games_black"] += row["no_progress_observable_games_black"]
    range_final = _finalize(range_row)

    return {
        "per_iter": per_iter_final,
        "range_total": range_final,
    }
