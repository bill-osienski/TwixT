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

from . import h2_match_rules as H2R
from . import h3_pilot_rules as R

FIRED = "FIRED"
CLEAR = "CLEAR"
UNDETERMINED = "NOT FIRED -- UNDETERMINED"
UNRESOLVED = "UNRESOLVED -- SCHEDULE INCOMPLETE"


class H3AnalysisError(RuntimeError):
    """A refusal from the pilot's analysis. Never a verdict."""


# 🔴 H2'S TRANSCRIPT CONTRACT, INHERITED WHOLE -- not reimplemented.
#
# My first version hashed the ordered `(mover, row, col)` moves and NOTHING ELSE,
# coercing with `str()` and `int()`. Three defects in one:
#   * it OMITTED the terminal reason and winner, so two games with identical moves
#     and opposite results collided into one transcript;
#   * it COERCED, so a malformed record normalised into a valid-looking transcript
#     instead of being refused -- `"10"` became 10 and `True` became 1;
#   * it validated no ply sequence and no mover parity at all.
# `h2_match_rules.transcript` already does all of it, type-strictly, with both
# ends of the ply sequence anchored to records the run itself wrote. Binding to it
# also means a later correction there cannot silently pass this module by.
transcript = H2R.transcript
transcript_digest = H2R.transcript_digest

#: A game record must carry a digest of this shape; anything else is malformed.
_HEX = set("0123456789abcdef")


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


def _duration(value: Any, what: str) -> float:
    """A FINITE, NON-NEGATIVE duration from a monotonic clock, or a refusal.

    🔴 `float()` alone accepted `nan`, `inf` and `True`. A NaN duration poisons
    every quantile silently -- a median of NaN is not a slow run, it is a broken
    record -- and `True` is not one second. A monotonic clock also cannot run
    backwards, so a negative is not a datum either.
    """
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise H3AnalysisError(
            f"{what} is {value!r} ({type(value).__name__}); a real number is required")
    v = float(value)
    if not math.isfinite(v):
        raise H3AnalysisError(f"{what} is {v}; a finite duration is required")
    if v < 0:
        raise H3AnalysisError(
            f"{what} is {v}; negative -- a monotonic clock does not run backwards")
    return v


def _check_timing(games: Sequence[Dict[str, Any]]) -> None:
    """FAIL CLOSED on a missing or impossible duration.

    Attempt 3's records carried NO timing at all, which is why the card's §4.1 was
    unanswerable after the fact.
    """
    for g in games:
        if "elapsed_s" not in g:
            raise H3AnalysisError(
                f"{g.get('task_id')}: no elapsed_s. Per-game timing is REQUIRED; "
                f"without it the pilot cannot answer its own runtime question.")
        _duration(g["elapsed_s"], f"{g.get('task_id')}: elapsed_s")


#: A game whose `terminal_reason` is one of these did not produce a result. Its
#: PAIR is excluded whole -- never half -- and counted.
NOT_SCOREABLE_REASONS = ("void", "VOID", "interrupted", "error")


def _validate_records(games: Sequence[Dict[str, Any]]) -> None:
    """REFUSE a corrupt input. Distinct from EXCLUSION, which is for legitimate
    run outcomes: a duplicated task or an unknown pair is a HARNESS fault, and
    reporting over it would describe something that never happened.
    """
    seen = set()
    for g in games:
        tid = g.get("task_id")
        if not isinstance(tid, str) or not tid:
            raise H3AnalysisError(f"a record has task_id {tid!r}; a string is required")
        if tid in seen:
            raise H3AnalysisError(
                f"duplicate task_id {tid!r}: the same game appears twice, which "
                f"would count it twice. This is a harness fault, not a result.")
        seen.add(tid)
        pid = g.get("pair_id")
        if type(pid) is not int or isinstance(pid, bool):
            raise H3AnalysisError(
                f"{tid}: pair_id is {pid!r} ({type(pid).__name__}); an int is required")
        if not 0 <= pid < R.N_OPENINGS:
            raise H3AnalysisError(
                f"{tid}: pair_id {pid} is outside the planned 0..{R.N_OPENINGS - 1}; "
                f"an unexpected pair means the record does not belong to this run")
        if g.get("incumbent_colour") not in ("red", "black"):
            raise H3AnalysisError(
                f"{tid}: incumbent_colour {g.get('incumbent_colour')!r}")
        d = g.get("transcript_digest")
        if not isinstance(d, str) or len(d) != 64 or not set(d) <= _HEX:
            raise H3AnalysisError(
                f"{tid}: transcript_digest {d!r} is not a sha256 hex digest")
        reason = g.get("terminal_reason")
        if reason not in tuple(H2R.TERMINAL_REASONS) + NOT_SCOREABLE_REASONS:
            raise H3AnalysisError(
                f"{tid}: terminal_reason {reason!r} is not one this protocol has")
        winner = g.get("winner")
        if winner is not None and winner not in H2R.WINNERS:
            raise H3AnalysisError(f"{tid}: winner {winner!r}")
        if type(g.get("plies")) is not int or isinstance(g.get("plies"), bool):
            raise H3AnalysisError(
                f"{tid}: plies is {g.get('plies')!r}; an int is required")


def _scoreable(g: Dict[str, Any]) -> bool:
    """Did this game produce a result at all?

    A VOID or an interrupted game did not. Nor did a `win` with no winner: that
    names no outcome and must not be guessed at.
    """
    if g.get("terminal_reason") in NOT_SCOREABLE_REASONS:
        return False
    if g.get("terminal_reason") == "win" and g.get("winner") not in ("red", "black"):
        return False
    return True


def _pair_outcome(g: Dict[str, Any]) -> str:
    """This game's CATEGORICAL outcome from the incumbent's side. Never a score."""
    if g.get("terminal_reason") == "cap":
        return "cap"
    return ("incumbent_win" if g.get("winner") == g.get("incumbent_colour")
            else "incumbent_loss")


def summarise(games: Sequence[Dict[str, Any]], *,
              total_elapsed_s: float) -> Dict[str, Any]:
    """The whole report. `games` are per-game records as the runner writes them."""
    games = [dict(g) for g in games]
    _validate_records(games)          # REFUSE a corrupt input before anything else
    _check_timing(games)
    total = _duration(total_elapsed_s, "total_elapsed_s")

    # ── nominal → drop incomplete → drop within-pair-identical → collapse dups
    by_pair: Dict[Any, List[Dict[str, Any]]] = {}
    for g in games:
        by_pair.setdefault(g["pair_id"], []).append(g)

    # EXCLUDED WHOLE, never half, with the reason kept separate: a pair missing a
    # game is a different fact from one whose game VOIDed.
    incomplete, voided, unscoreable = [], [], []
    for pid, gs in by_pair.items():
        if len(gs) != 2 or {x["incumbent_colour"] for x in gs} != {"red", "black"}:
            incomplete.append(pid)
        elif any(x.get("terminal_reason") in NOT_SCOREABLE_REASONS for x in gs):
            voided.append(pid)
        elif not all(_scoreable(x) for x in gs):
            unscoreable.append(pid)
    excluded = set(incomplete) | set(voided) | set(unscoreable)
    scored = {p: gs for p, gs in by_pair.items() if p not in excluded}

    def _slot(gs, colour):
        return next(x for x in gs if x["incumbent_colour"] == colour)

    pairs = []
    for p, gs in sorted(scored.items()):
        red, black = _slot(gs, "red"), _slot(gs, "black")
        pairs.append({"pair_id": p, "n_games": 2,
                      "red_digest": red["transcript_digest"],
                      "black_digest": black["transcript_digest"],
                      # the ORDERED categorical outcome: incumbent as red, then as
                      # black. A category, never a score.
                      "ordered_outcome": f"{_pair_outcome(red)}|{_pair_outcome(black)}",
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
              "total_elapsed_s": total}

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
        "S4a": _count_rule("S4a", total, float(R.MAX_TOTAL_ELAPSED_S),
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
        # 🔴 OVER PAIRS, not over games. The pair is the unit (H3 §5), so a
        # per-game terminal-reason tally answers a question the design does not
        # ask -- and my first version reported exactly that.
        tally: Dict[str, int] = {}
        for pr in distinct:
            tally[pr["ordered_outcome"]] = tally.get(pr["ordered_outcome"], 0) + 1
        outcomes = {
            "unit": "pair",
            "by_ordered_pair_outcome": tally,
            "categories": "incumbent_win | incumbent_loss | cap, ordered as "
                          "(incumbent as red)|(incumbent as black)",
            "note": "REPORTED, AND IT DECIDES NOTHING. Categorical counts over "
                    "pairs: no rate, no interval, no comparison. Not a stop rule "
                    "(card §6) and not evidence of strength."}

    return {
        "is_strength_verdict": False,
        "complete": complete,
        "games_completed": len(games),
        "games_nominal": R.N_GAMES,
        "pairs_nominal": R.N_OPENINGS,
        "pairs_scored": len(pairs),
        "incomplete_pairs": len(incomplete),
        "pairs_excluded_void": len(voided),
        "pairs_excluded_unscoreable": len(unscoreable),
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
