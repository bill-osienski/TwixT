"""H3 FULL STUDY — the paired score, the Hoeffding interval, the sensitivities.

Implements card §2, §3 and §4 as amended twice, and nothing else.

🔑 THIS MODULE *CAN* PRODUCE A STRENGTH VERDICT — unlike the pilot's analysis,
which deliberately could not. That is the whole difference between the two
experiments, and it is why every suppression rule below is explicit and
fail-closed: the only safe default for a module that may say "stronger" is to say
nothing.

THE RULE THAT DECIDES EVERY PARTIAL CASE, inherited from the pilot: **a monotone
COUNT may fire; a RATIO or a QUANTILE may not.**
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import h2_match_rules as H2R
from . import h3_study_rules as R

FIRED = "FIRED"
CLEAR = "CLEAR"

#: The null. From the QUESTION, never from an observation.
PARITY = 0.5

INCUMBENT_STRONGER = "INCUMBENT STRONGER"
T1J_STRONGER = "T1J STRONGER"
INCONCLUSIVE = "INCONCLUSIVE"
NO_VERDICT = "NO VERDICT"

#: 🔴 The wording that must accompany every interval. Amendment 1: distinct seeds
#: and distinct openings do not PROVE independence.
INTERVAL_NOTE = (
    "NOMINAL UNDER THE DECLARED PAIR-INDEPENDENCE MODEL. Rejection sampling "
    "conditions the population jointly, the strata are fixed rather than drawn, "
    "the same two engines play every game, and the software is deterministic. "
    "What the pairing buys is narrower and real: the two games of a pair are not "
    "counted as independent observations.")

#: H2's transcript contract, INHERITED WHOLE -- never reimplemented.
transcript = H2R.transcript
transcript_digest = H2R.transcript_digest

_HEX = set("0123456789abcdef")
NOT_SCOREABLE_REASONS = ("void", "VOID", "interrupted", "error")


class H3StudyAnalysisError(RuntimeError):
    """A refusal from the study's analysis. Never a verdict."""


# ═══════════════════════ identity (card §4.1) ══════════════════════════════
def game_identity(g: Dict[str, Any]) -> Tuple[str, str, str]:
    """`(opening_digest, incumbent_colour, transcript_digest)`.

    🔴 THE OPENING IS IN IT. H2's transcript EXCLUDES the opening because H2
    compared repetitions within ONE FIXED opening, where it was a constant. In
    this study the opening is the VARIABLE: two different openings can produce the
    same post-opening continuation and therefore the same transcript digest while
    being entirely different games, because the moves land on different boards.
    """
    return (g["opening_digest"], g["incumbent_colour"], g["transcript_digest"])


def pair_identity(games: Sequence[Dict[str, Any]]) -> Tuple[str, str, str]:
    """`(opening_digest, red_digest, black_digest)`."""
    by = {g["incumbent_colour"]: g for g in games}
    return (games[0]["opening_digest"],
            by["red"]["transcript_digest"], by["black"]["transcript_digest"])


def pair_score(incumbent_points: float) -> float:
    """The pair's score for the incumbent: its points over two."""
    if not isinstance(incumbent_points, (int, float)) or isinstance(
            incumbent_points, bool):
        raise H3StudyAnalysisError(f"points {incumbent_points!r}")
    if not 0.0 <= float(incumbent_points) <= 2.0:
        raise H3StudyAnalysisError(
            f"points {incumbent_points!r} is outside 0..2 for a two-game pair")
    return float(incumbent_points) / 2.0


# ═══════════════════════ validation ════════════════════════════════════════
def _duration(value: Any, what: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise H3StudyAnalysisError(f"{what}: elapsed_s is {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise H3StudyAnalysisError(f"{what}: elapsed_s is {v!r}, not finite")
    if v < 0:
        raise H3StudyAnalysisError(f"{what}: elapsed_s is negative ({v})")
    return v


def _validate(games: Sequence[Dict[str, Any]]) -> None:
    """REFUSE a corrupt input. Distinct from EXCLUSION, which is for legitimate
    run outcomes."""
    seen = set()
    for g in games:
        tid = g.get("task_id")
        if not isinstance(tid, str) or not tid:
            raise H3StudyAnalysisError(f"a record has task_id {tid!r}")
        if tid in seen:
            raise H3StudyAnalysisError(
                f"duplicate task_id {tid!r}: the same game twice is a harness "
                f"fault, not a result")
        seen.add(tid)
        pid = g.get("pair_id")
        if type(pid) is not int or isinstance(pid, bool) or not 0 <= pid < R.N_PAIRS:
            raise H3StudyAnalysisError(f"{tid}: pair_id {pid!r}")
        # 🔴 §6.2's three additions, REQUIRED -- the pilot's records dropped the
        # seed and its exposure had to be DERIVED.
        s = g.get("seed")
        if type(s) is not int or isinstance(s, bool):
            raise H3StudyAnalysisError(
                f"{tid}: seed is {s!r}; every record must carry its own seed so "
                f"accounting is READ and not derived from a plan")
        if g.get("stratum") not in R.STRATA:
            raise H3StudyAnalysisError(f"{tid}: stratum {g.get('stratum')!r}")
        od = g.get("opening_digest")
        if not isinstance(od, str) or len(od) != 64:
            raise H3StudyAnalysisError(
                f"{tid}: opening_digest {od!r} is not a digest; without it the "
                f"pair identity cannot be computed from the record")
        if g.get("incumbent_colour") not in ("red", "black"):
            raise H3StudyAnalysisError(f"{tid}: incumbent_colour")
        d = g.get("transcript_digest")
        if not isinstance(d, str) or len(d) != 64 or not set(d) <= _HEX:
            raise H3StudyAnalysisError(f"{tid}: transcript_digest {d!r}")
        if g.get("terminal_reason") not in (tuple(H2R.TERMINAL_REASONS)
                                            + NOT_SCOREABLE_REASONS):
            raise H3StudyAnalysisError(
                f"{tid}: terminal_reason {g.get('terminal_reason')!r}")
        w = g.get("winner")
        if w is not None and w not in H2R.WINNERS:
            raise H3StudyAnalysisError(f"{tid}: winner {w!r}")
        if type(g.get("plies")) is not int or isinstance(g.get("plies"), bool):
            raise H3StudyAnalysisError(f"{tid}: plies {g.get('plies')!r}")
        _duration(g.get("elapsed_s"), tid)


def _scoreable(g: Dict[str, Any]) -> bool:
    if g.get("terminal_reason") in NOT_SCOREABLE_REASONS:
        return False
    if g.get("terminal_reason") == "win" and g.get("winner") not in ("red", "black"):
        return False
    return True


def _points(g: Dict[str, Any]) -> float:
    """The incumbent's points for ONE game: 1 win, 0 loss, 0.5 cap (§4.5)."""
    if g.get("terminal_reason") == "cap":
        return 0.5
    return 1.0 if g.get("winner") == g.get("incumbent_colour") else 0.0


# ═══════════════════════ the estimate ══════════════════════════════════════
def _estimate(scores: Sequence[float]) -> Dict[str, Any]:
    """Mean, Hoeffding interval, and whether it is decisive against parity."""
    n = len(scores)
    if n == 0:
        return {"n": 0, "computable": False, "mean": None, "half_width": None,
                "interval": None, "decisive": False, "favours": None,
                "interval_note": INTERVAL_NOTE}
    mean = sum(scores) / n
    h = R.half_width(n)
    lo, hi = mean - h, mean + h
    decisive = lo > PARITY or hi < PARITY
    return {"n": n, "computable": True, "mean": mean, "half_width": h,
            "interval": [lo, hi], "decisive": decisive,
            "favours": None if not decisive else (
                "incumbent" if lo > PARITY else "t1j"),
            "interval_note": INTERVAL_NOTE}


# ═══════════════════════ the verdict rule (card §4.7) ══════════════════════
def evaluate_verdict(*, primary: Dict[str, Any],
                     sensitivities: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The JOINT rule. FAIL-CLOSED: every path that is not an unambiguous
    agreement returns NO VERDICT.

    A strength verdict issues only if ALL FOUR hold:
      1. the primary is computable and excludes parity;
      2. EVERY sensitivity is COMPUTABLE — an absent check is not a passed check;
      3. every sensitivity's mean lies STRICTLY on the primary's side of parity —
         `sign(mean - 0.5)` identical and NON-ZERO. A mean of exactly 0.5 does not
         "cross", so a rule phrased as crossing would have let it through, and an
         estimate sitting on the null supports neither direction;
      4. no sensitivity is decisive in the OPPOSITE direction.

    A sensitivity that merely WIDENS to straddle parity while keeping the
    primary's sign is reduced precision from the dropped pairs, not disagreement;
    suppressing there would fail the study for arithmetic rather than evidence.
    """
    if not primary.get("computable") or not primary.get("decisive"):
        return {"verdict": NO_VERDICT if not primary.get("computable")
                else INCONCLUSIVE,
                "why": ("the primary estimate could not be computed"
                        if not primary.get("computable")
                        else "the primary interval straddles parity")}
    side = primary.get("favours")
    for name, s in sorted(sensitivities.items()):
        if not s.get("computable"):
            return {"verdict": NO_VERDICT,
                    "why": (f"the {name} sensitivity is NOT COMPUTABLE, and an "
                            f"absent check is not a passed check")}
        mean = s.get("mean")
        if mean is None:
            return {"verdict": NO_VERDICT,
                    "why": f"the {name} sensitivity has no mean"}
        if mean == PARITY:
            return {"verdict": NO_VERDICT,
                    "why": (f"the {name} sensitivity sits exactly on parity "
                            f"({PARITY}); it supports neither direction")}
        # 🔑 THE DECISIVE CASE IS CHECKED FIRST, and the order is the point. A
        # sensitivity decisive the other way ALSO has its mean on the other side,
        # so testing the sign first would make this branch unreachable and the
        # worse diagnosis would never be printed -- each interval looking
        # conclusive on its own is the most dangerous disagreement there is.
        if s.get("decisive") and s.get("favours") not in (None, side):
            return {"verdict": NO_VERDICT,
                    "why": (f"the {name} sensitivity is DECISIVE IN THE OPPOSITE "
                            f"DIRECTION to the primary: each looks conclusive "
                            f"alone and they disagree")}
        s_side = "incumbent" if mean > PARITY else "t1j"
        if s_side != side:
            return {"verdict": NO_VERDICT,
                    "why": (f"the {name} sensitivity's estimate lies on the other "
                            f"side of parity from the primary")}
    return {"verdict": INCUMBENT_STRONGER if side == "incumbent" else T1J_STRONGER,
            "why": "the primary is decisive and every sensitivity agrees in sign"}


# ═══════════════════════ the report ════════════════════════════════════════
def summarise(games: Sequence[Dict[str, Any]], *,
              total_elapsed_s: float) -> Dict[str, Any]:
    """The whole report."""
    games = [dict(g) for g in games]
    _validate(games)
    _duration(total_elapsed_s, "total_elapsed_s")

    by_pair: Dict[int, List[Dict[str, Any]]] = {}
    for g in games:
        by_pair.setdefault(g["pair_id"], []).append(g)

    excluded_incomplete = excluded_void = 0
    scored: List[Dict[str, Any]] = []
    duplicates = 0
    seen_identity: Dict[Tuple[str, str, str], int] = {}

    for pid in sorted(by_pair):
        rows = by_pair[pid]
        # a pair is scored only if it is COMPLETE: exactly one game per colour
        # 🔴 MORE THAN TWO ROWS FOR ONE pair_id IS A HARNESS FAULT, counted as a
        # duplicate before anything else. The first version checked the colour
        # multiset FIRST, so four rows read as "incomplete" -- the wrong diagnosis
        # for the opposite problem.
        if len(rows) > 2:
            duplicates += 1
            first = {}
            for g in rows:
                first.setdefault(g["incumbent_colour"], g)
            rows = [first[c] for c in sorted(first)]
        if sorted(g["incumbent_colour"] for g in rows) != ["black", "red"]:
            excluded_incomplete += 1
            continue
        if not all(_scoreable(g) for g in rows):
            excluded_void += 1
            continue
        ident = pair_identity(rows)
        if ident in seen_identity:
            duplicates += 1
            continue
        seen_identity[ident] = pid
        pts = sum(_points(g) for g in rows)
        scored.append({
            "pair_id": pid,
            "segment": rows[0].get("segment"),
            "stratum": rows[0]["stratum"],
            "opening_digest": rows[0]["opening_digest"],
            "score": pair_score(pts),
            "capped": sum(1 for g in rows if g.get("terminal_reason") == "cap"),
            "identical": len({g["transcript_digest"] for g in rows}) == 1,
            "elapsed_s": sum(_duration(g["elapsed_s"], g["task_id"]) for g in rows),
        })

    # ── shared continuations: DESCRIPTIVE, and they gate NOTHING (§4.3)
    cont: Dict[str, set] = {}
    for g in games:
        cont.setdefault(g["transcript_digest"], set()).add(g["opening_digest"])
    shared = {d: ops for d, ops in cont.items() if len(ops) > 1}
    shared_pairs = {p["pair_id"] for p in scored
                    for g in by_pair[p["pair_id"]]
                    if g["transcript_digest"] in shared}

    primary = _estimate([p["score"] for p in scored])
    sens = {
        "cap_free": _estimate([p["score"] for p in scored if p["capped"] == 0]),
        "identity_free": _estimate([p["score"] for p in scored
                                    if not p["identical"]]),
        "both": _estimate([p["score"] for p in scored
                           if p["capped"] == 0 and not p["identical"]]),
    }

    capped_games = sum(1 for g in games if g.get("terminal_reason") == "cap")
    within_identical = sum(1 for p in scored if p["identical"])
    gates = [
        {"rule": "duplicate_pairs", "ceiling": R.MAX_DUPLICATE_PAIRS,
         "observed": duplicates,
         "what": "duplicate pairs -- a HARNESS FAULT: the openings are distinct "
                 "by construction, so two pairs cannot share an identity"},
        {"rule": "within_pair_identical", "ceiling": R.MAX_WITHIN_PAIR_IDENTICAL,
         "observed": within_identical,
         "what": "pairs whose two games are identical, contributing no "
                 "discrimination"},
        {"rule": "capped_games", "ceiling": R.MAX_CAPPED_GAMES,
         "observed": capped_games,
         "what": "capped games -- the configuration is not producing decisive "
                 "games"},
    ]
    for g in gates:
        g["status"] = FIRED if g["observed"] > g["ceiling"] else CLEAR
    below_floor = len(scored) < R.REPORT_FLOOR_PAIRS
    withheld = below_floor or any(g["status"] == FIRED for g in gates)

    if withheld:
        decision = {"verdict": NO_VERDICT,
                    "why": ("fewer than the floor of completed pairs"
                            if below_floor else
                            "a degeneracy gate fired; interpretation is withheld "
                            "and the counts stand")}
    else:
        decision = evaluate_verdict(primary=primary, sensitivities=sens)

    by_stratum = {}
    for s in R.STRATA:
        xs = [p["score"] for p in scored if p["stratum"] == s]
        by_stratum[s] = {"n": len(xs),
                         "mean": (sum(xs) / len(xs)) if xs else None,
                         "note": "DESCRIPTIVE ONLY -- no interval, no verdict"}

    return {
        "design": "H3_FULL_STUDY",
        "games_seen": len(games),
        "games_scored": len(scored) * 2,
        "pairs_scored": len(scored),
        "pairs_excluded_incomplete": excluded_incomplete,
        "pairs_excluded_void": excluded_void,
        "duplicate_pairs": duplicates,
        "within_pair_identical": within_identical,
        "capped_games": capped_games,
        "pairs_with_a_cap": sum(1 for p in scored if p["capped"]),
        "shared_continuation_pairs": len(shared_pairs),
        "shared_continuation_relations": len(shared),
        "shared_continuation_note":
            "DESCRIPTIVE ONLY, AND GATES NOTHING. Two games from different "
            "openings are distinct observations even when their move sequences "
            "match: the opening changes the board those moves are played on.",
        "primary": primary,
        "sensitivities": sens,
        "by_stratum": by_stratum,
        "gates": gates,
        "below_report_floor": below_floor,
        "interpretation_withheld": withheld,
        "verdict": decision["verdict"],
        "verdict_note": decision["why"],
        "is_strength_verdict": decision["verdict"] in (INCUMBENT_STRONGER,
                                                       T1J_STRONGER),
        "total_elapsed_s": float(total_elapsed_s),
        "pairs": scored,
    }
