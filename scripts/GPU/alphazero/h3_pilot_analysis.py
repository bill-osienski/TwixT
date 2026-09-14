"""H3 PILOT — pair-preserving handling, timing, and complete/partial reporting.

Implements the card's §4.4, §5, §5.1 and §6 and nothing else.

🔴 THIS MODULE CANNOT PRODUCE A STRENGTH VERDICT. There is deliberately no rate,
no interval and no comparison anywhere in it, and a test asserts those words do
not appear among the report's keys.

THE RULE THAT DECIDES EVERY PARTIAL CASE: **a monotone COUNT may fire; a RATIO or
a QUANTILE may not.** An event already observed cannot be un-observed by playing
more games, so a count past its threshold is past it for the full schedule. A
quantile is not monotone under adding observations, so it is not evaluated unless
the schedule completed.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import h3_pilot_rules as R

FIRED = "FIRED"
CLEAR = "CLEAR"
UNDETERMINED = "NOT FIRED -- UNDETERMINED"
UNRESOLVED = "UNRESOLVED -- SCHEDULE INCOMPLETE"


class H3AnalysisError(RuntimeError):
    """A refusal from the pilot's analysis. Never a verdict."""


def transcript_digest(moves: Sequence[Tuple[str, int, int]]) -> str:
    """sha256 over the ordered `(mover, row, col)` sequence -- AND NOTHING ELSE.

    🔑 H2's §2.2 lesson, inherited: hashing the whole record makes `seed`,
    `task_id` and `rep` give a distinct digest for a cell that played ONE game
    many times, which is exactly the degeneracy the screen exists to find.

    MOVERS ARE IN THE KEY. That is what makes §5.1's derivation hold: a game our
    incumbent played as red can never share a digest with one it played as black,
    so a red slot can only collide with a red slot.
    """
    payload = [[str(m), int(r), int(c)] for (m, r, c) in moves]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile. `statistics.quantiles` needs n >= 2 and
    reports cut points, which is a different thing."""
    if not values:
        raise H3AnalysisError("no values")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def _check_timing(games: Sequence[Dict[str, Any]]) -> None:
    """FAIL CLOSED on a missing or impossible duration.

    Attempt 3's records carried NO timing at all, which is why the card's §4.1 was
    unanswerable after the fact. A monotonic clock cannot run backwards, so a
    negative elapsed is not a slow game -- it is a broken record.
    """
    for g in games:
        if "elapsed_s" not in g:
            raise H3AnalysisError(
                f"{g.get('task_id')}: no elapsed_s. Per-game timing is REQUIRED; "
                f"without it the pilot cannot answer its own runtime question.")
        try:
            v = float(g["elapsed_s"])
        except (TypeError, ValueError):
            raise H3AnalysisError(f"{g.get('task_id')}: elapsed_s is not a number")
        if v < 0:
            raise H3AnalysisError(
                f"{g.get('task_id')}: elapsed_s {v} is negative; a monotonic clock "
                f"does not run backwards")


def summarise(games: Sequence[Dict[str, Any]], *,
              total_elapsed_s: float) -> Dict[str, Any]:
    """The whole report. `games` are per-game records as the runner writes them."""
    games = [dict(g) for g in games]
    _check_timing(games)

    # ── nominal → drop incomplete → drop within-pair-identical → collapse dups
    by_pair: Dict[Any, List[Dict[str, Any]]] = {}
    for g in games:
        by_pair.setdefault(g["pair_id"], []).append(g)

    incomplete = [p for p, gs in by_pair.items()
                  if len(gs) != 2
                  or {x["incumbent_colour"] for x in gs} != {"red", "black"}]
    scored = {p: gs for p, gs in by_pair.items() if p not in incomplete}

    def _slot(gs, colour):
        return next(x for x in gs if x["incumbent_colour"] == colour)

    pairs = []
    for p, gs in sorted(scored.items()):
        red, black = _slot(gs, "red"), _slot(gs, "black")
        pairs.append({"pair_id": p, "n_games": 2,
                      "red_digest": red["transcript_digest"],
                      "black_digest": black["transcript_digest"],
                      "capped": sum(1 for x in gs if x.get("terminal_reason") == "cap"),
                      "elapsed_s": float(red["elapsed_s"]) + float(black["elapsed_s"])})

    identical = [p for p in pairs if p["red_digest"] == p["black_digest"]]
    informative = [p for p in pairs if p not in identical]

    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    duplicates = 0
    distinct: List[Dict[str, Any]] = []
    for p in informative:
        key = (p["red_digest"], p["black_digest"])
        if key in seen:
            duplicates += 1
            continue
        seen[key] = p
        distinct.append(p)

    # ── §5.1: overlap among pairs_distinct, AFTER duplicates collapse
    relations = 0
    involved = set()
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            a, b = distinct[i], distinct[j]
            if {a["red_digest"], a["black_digest"]} & {b["red_digest"], b["black_digest"]}:
                relations += 1
                involved.add(a["pair_id"])
                involved.add(b["pair_id"])

    elapsed = [float(g["elapsed_s"]) for g in games]
    timing = {"n": len(elapsed),
              "min": min(elapsed) if elapsed else None,
              "median": _percentile(elapsed, 0.5) if elapsed else None,
              "p90": _percentile(elapsed, 0.9) if elapsed else None,
              "max": max(elapsed) if elapsed else None,
              "total_elapsed_s": float(total_elapsed_s)}

    capped = sum(1 for g in games if g.get("terminal_reason") == "cap")
    complete = (len(games) == R.N_GAMES and not incomplete
                and len(scored) == R.N_OPENINGS)

    def _count_rule(name, observed, ceiling, what):
        """A MONOTONE COUNT. Fires the moment it is exceeded, complete or not;
        below the ceiling it is CLEAR only if the schedule finished."""
        if observed > ceiling:
            return {"status": FIRED, "observed": observed, "ceiling": ceiling,
                    "what": what}
        return {"status": CLEAR if complete else UNDETERMINED,
                "observed": observed, "ceiling": ceiling, "what": what}

    rules = {
        "S1": _count_rule("S1", duplicates, R.MAX_DUPLICATE_PAIRS,
                          "duplicate pairs"),
        "S2": _count_rule("S2", capped, R.MAX_CAPPED_GAMES, "capped games"),
        "S3": _count_rule("S3", len(identical), R.MAX_WITHIN_PAIR_IDENTICAL,
                          "within-pair-identical pairs"),
        "S4a": _count_rule("S4a", float(total_elapsed_s), float(R.MAX_TOTAL_ELAPSED_S),
                           "total elapsed seconds"),
    }
    # ── S4b is a RATIO OF TWO QUANTILES and is NOT monotone: the games that were
    #    never played could pull p90 down and the median up, reversing it. It is
    #    not evaluated at all unless the schedule completed.
    if not complete:
        rules["S4b"] = {"status": UNRESOLVED, "observed": None,
                        "ceiling": R.MAX_P90_OVER_MEDIAN,
                        "what": "p90 / median, not evaluable on a partial run"}
    else:
        ratio = (timing["p90"] / timing["median"]) if timing["median"] else None
        rules["S4b"] = {
            "status": FIRED if (ratio is not None and ratio > R.MAX_P90_OVER_MEDIAN)
                      else CLEAR,
            "observed": ratio, "ceiling": R.MAX_P90_OVER_MEDIAN,
            "what": "p90 / median"}

    withheld = len(pairs) < R.REPORT_FLOOR_PAIRS
    outcomes = None
    if not withheld:
        tally: Dict[str, int] = {}
        for g in games:
            tally[str(g.get("terminal_reason"))] = \
                tally.get(str(g.get("terminal_reason")), 0) + 1
        outcomes = {"by_terminal_reason": tally,
                    "note": "REPORTED, AND IT DECIDES NOTHING. Not a stop rule "
                            "(card §6) and not evidence of strength."}

    return {
        "is_strength_verdict": False,
        "complete": complete,
        "games_completed": len(games),
        "games_nominal": R.N_GAMES,
        "pairs_nominal": R.N_OPENINGS,
        "pairs_scored": len(pairs),
        "incomplete_pairs": len(incomplete),
        "within_pair_identical": len(identical),
        "pairs_informative": len(informative),
        "duplicate_pairs": duplicates,
        "pairs_distinct": len(distinct),
        "partial_overlap_pairs": len(involved),
        "partial_overlap_relations": relations,
        "capped_games": capped,
        "timing": timing,
        "stop_rules": rules,
        "any_fired": any(r["status"] == FIRED for r in rules.values()),
        "interpretation_withheld": withheld,
        "outcome_distribution": outcomes,
        "pairs": distinct,
        "partial_note": None if complete else (
            "PARTIAL. Every count and timing describes the COMPLETED games only. "
            "No stop rule is CLEAR, S4b is not evaluated, and no cap rate or "
            "runtime tail is stated for the full schedule."),
    }
