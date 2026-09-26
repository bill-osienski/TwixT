"""H4 CONFIRMATORY ANALYSIS. GATED; NOT AUTHORIZED.

Frozen by `docs/superpowers/2026-09-23-t1j-h4-analysis-card.md` (amended
6490055), §3 and §4.1. 🔴 THE CARD IS THE AUTHORITY.

🔴 ORDER IS THE CONTRACT (card §3.1). The gate is read FIRST -- before any file
is opened, stat'ed or hashed -- then durability of every file the decision rests
on, then a PROCEED artifact whose two hashes recompute, then exactly the four
H4_STUDY segment files the study manifest names, each bound to its own slot.
Anything else refuses with no partial estimate: an incomplete study is not a
smaller study.

🔴 MULTIPLICITY (card §3.3). Every valid pair is counted as many times as it
occurs. Identical trajectories are observations; there is no duplicate rule of
any kind. A malformed pair is an integrity fault and REFUSES -- it is never
dropped. H3 is not used in any form.
"""
from __future__ import annotations

import collections
import math
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import h4_pilot_feasibility as F
from . import h4_runner as R

#: THE H4 STUDY AGGREGATION IS NOT AUTHORIZED. Changing this is a reviewed
#: one-line code change. Read FIRST by `_run`, the only path to an estimate, so
#: both public entries -- `run_confirmatory` and `main` -- pass through it before
#: touching any file. No override exists: not argv, not the environment.
H4_STUDY_AGGREGATION_AUTHORIZED = False

ALPHA = 0.05
STUDY_PAIRS = F.SEGMENTS[F.STUDY] * F.PAIRS_PER_SEGMENT[F.STUDY]
INDEPENDENCE = ("nominal two-sided 95% Hoeffding under a DECLARED independence model: "
                "independent seed bundles AND isolated T1j process realizations. That "
                "independence is an assumption, not a finding (replacement card §6).")
READINGS = {
    "above": "incumbent stronger in empty-board direct play under this joint "
             "execution protocol",
    "below": "T1j stronger under this protocol; H4 alone cannot localize the cause",
    "contains": "unresolved or near parity; says nothing about why H3 differed"}
CAP_FREE_LABEL = ("DESCRIPTIVE and CONDITIONAL: which pairs it keeps is decided by the "
                  "OBSERVED caps, so it is not a second confirmatory claim")
NOT_ESTIMABLE = "not estimable"

EXIT_OK, EXIT_REFUSED, EXIT_UNAUTHORIZED = 0, 4, 5


class H4AggregationUnauthorized(Exception):
    """The gate is closed. Nothing was opened, read or written."""


# ─────────────────────────── pairs, §3.2 ───────────────────────────

def incumbent_points(result: Mapping[str, Any], incumbent_colour: str) -> float:
    """The frozen winner encoding. Anything outside it is an integrity fault."""
    reason, winner = result.get("terminal_reason"), result.get("winner")
    if reason == "win" and winner in ("red", "black"):
        return 1.0 if winner == incumbent_colour else 0.0
    if reason == "cap" and winner is None:
        return 0.5
    F.refuse(f"{result.get('task_id')}: terminal_reason {reason!r} with winner "
             f"{winner!r} is an integrity fault")


def form_pairs(segments: Sequence[Sequence[Mapping[str, Any]]]) -> List[Tuple[Any, Any]]:
    """Exactly two games per `pair_id`; any malformed pair REFUSES, never drops."""
    games = [(i, g) for i, seg in enumerate(segments) for g in seg]
    for key in ("task_id", "seed"):
        counts = collections.Counter(g["start"][key] for _i, g in games)
        repeated = [v for v, n in counts.items() if n > 1]
        if repeated:
            F.refuse(f"{key} {repeated[0]!r} appears {counts[repeated[0]]} times: an "
                     f"integrity fault, never a replicate")
    by_pair: Dict[Any, List[Tuple[int, Any]]] = collections.defaultdict(list)
    for i, g in games:
        by_pair[g["start"]["pair_id"]].append((i, g))
    pairs = []
    for pid, members in by_pair.items():
        if len(members) != 2:
            F.refuse(f"pair {pid}: {len(members)} games, not exactly two")
        (sa, a), (sb, b) = sorted(members, key=lambda m: m[1]["start"]["arm"])
        if sa != sb:
            F.refuse(f"pair {pid} crosses segments {sa} and {sb}")
        if (a["start"]["arm"], b["start"]["arm"]) != ("A", "B"):
            F.refuse(f"pair {pid}: arms {a['start']['arm']}, {b['start']['arm']}, "
                     f"not one A and one B")
        for g, colours in ((a, R.ARMS["A"]), (b, R.ARMS["B"])):
            s = g["start"]
            if (s["incumbent_colour"], s["t1j_colour"]) != colours:
                F.refuse(f"{s['task_id']}: colours {s['incumbent_colour']}/"
                         f"{s['t1j_colour']} are not Arm {s['arm']}'s")
            if any((p["actor"] == "incumbent") != (p["mover"] == s["incumbent_colour"])
                   for p in g["plies"]):
                F.refuse(f"{s['task_id']}: a ply record's actor disagrees with the colours")
        ia, ib = a["start"]["game_index"], b["start"]["game_index"]
        if ia % 2 or ib != ia + 1:
            F.refuse(f"pair {pid}: game_index {ia}, {ib} are not adjacent 2k, 2k+1")
        for g in (a, b):
            res = g["result"]
            if res is None or res.get("terminal_reason") not in ("win", "cap"):
                F.refuse(f"pair {pid}: a game is not complete")
            pts = incumbent_points(res, g["start"]["incumbent_colour"])
            if (res.get("incumbent_points"), res.get("t1j_points")) != (pts, 1.0 - pts):
                F.refuse(f"{res['task_id']}: recorded points disagree with the "
                         f"recomputed ones")
        pairs.append((a, b))
    return pairs


# ─────────────────────── scoring and the interval, §3.3 ───────────────────────

def half_width(n: int) -> float:
    return math.sqrt(math.log(2 / ALPHA) / (2 * n))


def interval(scores: Sequence[float]) -> Dict[str, Any]:
    """RAW: never intersected with [0, 1]."""
    n = len(scores)
    mean = sum(scores) / n
    h = half_width(n)
    return {"n": n, "mean": mean, "half_width": h, "lower": mean - h, "upper": mean + h}


def reading(iv: Mapping[str, float]) -> str:
    """STRICT, on the raw bounds: a lower bound of exactly 0.5 is not above."""
    if iv["lower"] > 0.5:
        return "above"
    if iv["upper"] < 0.5:
        return "below"
    return "contains"


def _outcome(g) -> str:
    return g["result"]["winner"] or "cap"


def estimate(pairs: Sequence[Tuple[Any, Any]]) -> Dict[str, Any]:
    scores = [(incumbent_points(a["result"], a["start"]["incumbent_colour"])
               + incumbent_points(b["result"], b["start"]["incumbent_colour"])) / 2
              for a, b in pairs]
    primary = interval(scores)
    capped = [any(g["result"]["terminal_reason"] == "cap" for g in pair) for pair in pairs]
    kept = [s for s, c in zip(scores, capped) if not c]
    cap_free = ({**interval(kept), "label": CAP_FREE_LABEL} if kept else
                {"n": 0, "status": NOT_ESTIMABLE, "label": CAP_FREE_LABEL})
    view = F.blinded_view([g for pair in pairs for g in pair])
    patterns = collections.Counter(f"{_outcome(a)}|{_outcome(b)}" for a, b in pairs)
    return {"primary": {**primary, "reading": reading(primary),
                        "wording": READINGS[reading(primary)],
                        "independence": INDEPENDENCE},
            "cap_free_sensitivity": cap_free,
            "caps": {"cap_affected_pairs": sum(capped),
                     "cap_games": sum(1 for v in view if v["terminal_reason"] == "cap")},
            "concentration": {**F.concentration(view),
                              "outcome_patterns": dict(patterns),
                              "note": "substantial mass on few games does not reduce n "
                                      "to the unique count"}}


# ─────────────────────────────── the run, §3.1 ───────────────────────────────

def _run(*, pilot_manifest: str, study_manifest: str, artifact: str, pilot_results: str,
         feasibility_report: str, segments: Sequence[str], out: str,
         _fixture: bool) -> Dict[str, Any]:
    if not H4_STUDY_AGGREGATION_AUTHORIZED:                          # 1. FIRST
        raise H4AggregationUnauthorized(
            "the H4 study aggregation is UNAUTHORIZED. No file was opened, read or "
            "written.")
    F.check_free(out)
    if len(segments) != F.SEGMENTS[F.STUDY]:
        F.refuse(f"{len(segments)} segment files: the study is exactly "
                 f"{F.SEGMENTS[F.STUDY]}; an incomplete study is not a smaller study")
    F.check_durable([pilot_manifest, study_manifest, pilot_results, feasibility_report,
                     artifact, *segments], _fixture=_fixture)        # 2.
    art = F._load_json(artifact, "decision artifact")               # 3.
    if not isinstance(art, dict) or art.get("decision") != "PROCEED":
        F.refuse(f"the decision artifact says "
                 f"{art.get('decision') if isinstance(art, dict) else art!r}, not PROCEED")
    for path, key in ((pilot_results, "pilot_results_sha256"),
                      (feasibility_report, "feasibility_report_sha256")):
        if F.sha256_file(path) != art.get(key):
            F.refuse(f"the artifact's {key} does not recompute against {path}")
    manifest = F.load_manifest(study_manifest, F.STUDY, _fixture=_fixture)   # 4.
    loaded = []
    for path, entry in zip(segments, manifest["segments"]):
        F.bind_header(F.read_header(path), F.STUDY, entry, _fixture=_fixture)
        try:
            games = R.load_games(path, allow_fixture=_fixture)
            F.segment_end(path)
        except (R.H4RunError, ValueError, KeyError, TypeError, IndexError) as e:
            F.refuse(f"segment {entry['segment']}: {e}")
        F.bind_games(games, entry["schedule"])
        loaded.append(games)
    pairs = form_pairs(loaded)
    if len(pairs) != STUDY_PAIRS:
        F.refuse(f"{len(pairs)} pairs, not {STUDY_PAIRS}")
    result = {"analysis": "H4_CONFIRMATORY", "fixture": _fixture,
              "inputs": {p: F.sha256_file(p) for p in (
                  pilot_manifest, study_manifest, artifact, pilot_results,
                  feasibility_report, *segments)},
              **estimate(pairs), "written_at": F.now_iso()}
    F.write_create_only(out, result)
    return result


def run_confirmatory(*, pilot_manifest: str, study_manifest: str, artifact: str,
                     pilot_results: str, feasibility_report: str, segments: Sequence[str],
                     out: str) -> Dict[str, Any]:
    """PUBLIC ENTRY. Gated; fixture data is refused here: no argument relaxes it."""
    return _run(pilot_manifest=pilot_manifest, study_manifest=study_manifest,
                artifact=artifact, pilot_results=pilot_results,
                feasibility_report=feasibility_report, segments=segments, out=out,
                _fixture=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY."""
    import argparse
    ap = argparse.ArgumentParser(prog="h4_confirmatory_analysis",
                                 description="H4 confirmatory analysis. NOT AUTHORIZED.")
    for flag in ("--pilot-manifest", "--study-manifest", "--artifact", "--pilot-results",
                 "--feasibility-report", "--out"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--segment", action="append", required=True)
    a = ap.parse_args(argv)
    try:
        run_confirmatory(pilot_manifest=a.pilot_manifest, study_manifest=a.study_manifest,
                         artifact=a.artifact, pilot_results=a.pilot_results,
                         feasibility_report=a.feasibility_report, segments=a.segment,
                         out=a.out)
    except H4AggregationUnauthorized as e:
        print(str(e), file=sys.stderr)
        return EXIT_UNAUTHORIZED
    except F.H4AnalysisRefused as e:
        print(f"REFUSED, nothing written: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"estimate written: {a.out}")
    return EXIT_OK


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
