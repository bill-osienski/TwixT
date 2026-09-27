"""H5 CONFIRMATORY ANALYSIS. GATED; NOT AUTHORIZED.

Frozen by `docs/superpowers/2026-09-27-t1j-h5-implementation-card.md` §6, under
the H5 replication card §3-§6 and §12. 🔴 THE CARDS ARE THE AUTHORITY.

🔴 ORDER IS THE CONTRACT (card §6.1). The gate is read FIRST -- before any file
is opened, stat'ed or hashed -- then the destination, durability of every input,
the PINNED H4 estimate's hash, an `H5_STUDY` manifest, exactly four segment files
each bound to its own entry, and 296 pairs. Anything else refuses with no partial
estimate. There is NO pilot and no proceed/stop artifact, and no parameter for one.

🔴 H5's OWN STAGE AND HEADER BINDING are what refuse fixture data on the public
path. `h4_runner.load_games` recognizes only H4's fixture label, so an
`H5_SYNTHETIC_FIXTURE` file would pass it; `bind_header` below refuses it first.

🔴 MULTIPLICITY: every valid pair counted as often as it occurs. Pair formation,
the winner encoding and the raw interval are H4's own functions, called unchanged.
H4's verdict WORDING is not: H5 asks a different question. No pooled estimate, no
test or interval on the H4-H5 difference, no per-segment interval, no H3.
"""
from __future__ import annotations

import collections
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

from . import h4_confirmatory_analysis as C4
from . import h4_pilot_feasibility as F
from . import h4_runner as R4
from . import h5_aggregation_authorization as AUTH
from . import h5_runner as H5

STAGE = H5.STUDY_DESIGN
FIXTURE_DESIGN = H5.FIXTURE_DESIGN
SEGMENT_COUNT = len(H5.SEGMENTS)
STUDY_PAIRS = SEGMENT_COUNT * H5.PAIRS_PER_SEGMENT
BOUND = F.BOUND

#: 🔴 Card §4.1: the PINNED H4 estimate, checked BEFORE it is parsed.
H4_ESTIMATE = "docs/superpowers/evidence/t1j-h4-confirmatory/estimate.json"
H4_ESTIMATE_SHA256 = "619f515a1535d365160bbcdd470db2b1c7844a5221c8857171305c0556c1f55a"
#: Card §6.1 step 2: the one destination.
OUT = "docs/superpowers/evidence/t1j-h5-confirmatory/estimate.json"

#: Replication card §3.1: on the RAW, COMPUTED bounds -- never a rounded threshold.
REPLICATION, OPPOSITE, INCONCLUSIVE = ("DIRECTIONAL_REPLICATION", "OPPOSITE_DIRECTION",
                                       "INCONCLUSIVE")
#: Replication card §6.
WORDING = {
    REPLICATION: "in a second, independent (fresh, non-overlapping) set of 296 pairs, the "
                 "incumbent's expected score is again above 0.5 in empty-board direct "
                 "play under this joint execution protocol; this says nothing about the "
                 "size of the difference beyond H5's own interval",
    OPPOSITE: "H5's interval lies below 0.5 under this protocol, conflicting with H4; both "
              "results stand as reported and the conflict is not resolved by pooling",
    INCONCLUSIVE: "H5 does not replicate H4's direction at its preregistered precision; "
                  "this is not evidence of parity and not a refutation of H4"}
#: Replication card §6 and §12 C1.
INDEPENDENCE = ("nominal two-sided 95% Hoeffding under a DECLARED independence model. "
                "'Independent set' means fresh, non-overlapping pairs; independence of "
                "their outcomes -- across H5's pairs, and between H5 and H4 -- is an "
                "assumption, not an observed property. Replication does not verify it.")
H4_STANDS = "H4's published result stands unchanged whatever H5 finds"
CAP_FREE_LABEL = C4.CAP_FREE_LABEL
DESCRIPTIVE = "DESCRIPTIVE -- no interval, no test, no pooled estimate"
PER_SEGMENT_LABEL = "DESCRIPTIVE, in run order -- no interval, no test, no trend statistic"

EXIT_OK, EXIT_REFUSED, EXIT_UNAUTHORIZED = 0, 4, 5


class H5AggregationUnauthorized(Exception):
    """The gate is closed. Nothing was opened, read or written."""


# ─────────────────────────── binding: H5's stage and header ───────────────────────────

def load_manifest(path: str, *, _fixture: bool) -> Dict[str, Any]:
    """A committed H5 manifest, validated -- its digests RECOMPUTED."""
    m = F._load_json(path, "manifest")
    if not isinstance(m, dict) or m.get("stage") != STAGE:
        F.refuse(f"{path}: an {STAGE} manifest is required, not "
                 f"{m.get('stage') if isinstance(m, dict) else m!r}")
    entries = m.get("segments")
    if not isinstance(entries, list) or len(entries) != SEGMENT_COUNT:
        F.refuse(f"{path}: an {STAGE} manifest names exactly {SEGMENT_COUNT} segments")
    runtime = {"depth": R4.DEPTH, "query_timeout_s": R4.QUERY_TIMEOUT_S,
               "replay_timeout_s": R4.REPLAY_TIMEOUT_S, "ply_cap": R4.PLY_CAP,
               "h4_acceptance": True}
    for i, e in enumerate(entries):
        where = f"{path}: segment entry {i}"
        missing = [k for k in ("segment", "schedule", *BOUND) if k not in e]
        if missing:
            F.refuse(f"{where} lacks {missing}")
        if e["segment"] != i:
            F.refuse(f"{where} is labelled segment {e['segment']!r}")
        try:
            R4.check_schedule(e["schedule"], mode="fixture" if _fixture else "study")
        except R4.H4RunError as err:
            F.refuse(f"{where}: {err}")
        if len(e["schedule"]) != 2 * H5.PAIRS_PER_SEGMENT:
            F.refuse(f"{where}: {len(e['schedule'])} tasks, not {2 * H5.PAIRS_PER_SEGMENT}")
        if R4.schedule_digest(e["schedule"]) != e["schedule_digest"]:
            F.refuse(f"{where}: schedule_digest does not recompute from the schedule")
        if e["seeds"] != [t["seed"] for t in e["schedule"]]:
            F.refuse(f"{where}: seeds are not the schedule's seeds")
        if sorted(e["cards"]) != sorted(H5.CARDS):
            F.refuse(f"{where}: cards must be exactly H5's {len(H5.CARDS)}")
        if sorted(e["code"]) != sorted(H5.CODE):
            F.refuse(f"{where}: code must be exactly H5's {len(H5.CODE)} files")
        rt = e["t1j_runtime"]
        wrong = [k for k, v in runtime.items() if not isinstance(rt, dict) or rt.get(k) != v]
        if wrong:
            F.refuse(f"{where}: t1j_runtime {wrong} differ from the frozen runtime")
    return m


def bind_header(header: Mapping[str, Any], entry: Mapping[str, Any], *,
                _fixture: bool) -> None:
    """The results header equals its manifest entry, FIELD BY FIELD -- and its
    design is H5's STUDY label on the public path, never a fixture label."""
    want = FIXTURE_DESIGN if _fixture else STAGE
    if header.get("design") != want:
        F.refuse(f"design {header.get('design')!r}, not {want!r}: study and fixture data "
                 f"are never interchangeable")
    if header.get("evidence") is not (not _fixture):
        F.refuse(f"evidence flag {header.get('evidence')!r} does not fit {want}")
    if header.get("segment") != entry["segment"]:
        F.refuse(f"segment {header.get('segment')!r} is not manifest slot {entry['segment']}")
    for f in BOUND:
        if f not in header:
            F.refuse(f"the header has no `{f}` field")
        if header.get(f) != entry[f]:
            F.refuse(f"header `{f}` differs from the manifest")


def load_h4_estimate(path: str) -> Dict[str, Any]:
    """The PINNED H4 estimate: hashed BEFORE it is parsed; never recomputed."""
    if F.sha256_file(path) != H4_ESTIMATE_SHA256:
        F.refuse(f"{path} does not hash to the pinned H4 estimate "
                 f"{H4_ESTIMATE_SHA256[:8]}...")
    est = F._load_json(path, "H4 estimate")
    if not isinstance(est, dict) or est.get("analysis") != "H4_CONFIRMATORY" \
            or est.get("fixture") is not False:
        F.refuse(f"{path} is not the published H4 confirmatory estimate")
    return est


# ─────────────────────────────── the estimate, §6.2 ───────────────────────────────

def verdict(iv: Mapping[str, float]) -> str:
    """Replication card §3.1: STRICT, on the raw computed bounds."""
    if iv["lower"] > 0.5:
        return REPLICATION
    if iv["upper"] < 0.5:
        return OPPOSITE
    return INCONCLUSIVE


def pair_score(a, b) -> float:
    return (C4.incumbent_points(a["result"], a["start"]["incumbent_colour"])
            + C4.incumbent_points(b["result"], b["start"]["incumbent_colour"])) / 2


def _outcome(g) -> str:
    return g["result"]["winner"] or "cap"


def estimate(segments: Sequence[Sequence[Mapping[str, Any]]],
             ends: Sequence[Mapping[str, Any]], h4: Mapping[str, Any]) -> Dict[str, Any]:
    pairs = C4.form_pairs(segments)
    if len(pairs) != STUDY_PAIRS:
        F.refuse(f"{len(pairs)} pairs, not {STUDY_PAIRS}")
    seg_of = {g["start"]["pair_id"]: k for k, seg in enumerate(segments) for g in seg}
    scores = [pair_score(a, b) for a, b in pairs]
    capped = [any(g["result"]["terminal_reason"] == "cap" for g in pair) for pair in pairs]
    primary = C4.interval(scores)
    v = verdict(primary)
    kept = [s for s, c in zip(scores, capped) if not c]
    cap_free = ({**C4.interval(kept), "label": CAP_FREE_LABEL} if kept else
                {"n": 0, "status": C4.NOT_ESTIMABLE, "label": CAP_FREE_LABEL})
    per_segment = []
    for k in range(SEGMENT_COUNT):
        idx = [i for i, (a, _b) in enumerate(pairs) if seg_of[a["start"]["pair_id"]] == k]
        per_segment.append({
            "segment": k, "pairs": len(idx), "mean": sum(scores[i] for i in idx) / len(idx),
            "cap_affected_pairs": sum(capped[i] for i in idx),
            "cap_games": sum(g["result"]["terminal_reason"] == "cap"
                             for i in idx for g in pairs[i]),
            "total_s": ends[k]["total_s"]})
    view = F.blinded_view([g for pair in pairs for g in pair])
    patterns = collections.Counter(f"{_outcome(a)}|{_outcome(b)}" for a, b in pairs)
    caps = {"cap_affected_pairs": sum(capped),
            "cap_games": sum(1 for x in view if x["terminal_reason"] == "cap")}
    h4p = h4["primary"]
    return {
        "primary": {**primary, "verdict": v, "wording": WORDING[v],
                    "independence": INDEPENDENCE, "h4": H4_STANDS},
        "cap_free_sensitivity": cap_free,
        "caps": caps,
        "per_segment": {"label": PER_SEGMENT_LABEL, "segments": per_segment},
        "concentration": {**F.concentration(view), "outcome_patterns": dict(patterns),
                          "note": "substantial mass on few games does not reduce n to "
                                  "the unique count"},
        "h4_beside_h5": {
            "label": DESCRIPTIVE,
            "h4": {"n": h4p["n"], "mean": h4p["mean"], "lower": h4p["lower"],
                   "upper": h4p["upper"], "cap_free_sensitivity": h4["cap_free_sensitivity"],
                   "caps": h4["caps"], "estimate_sha256": H4_ESTIMATE_SHA256},
            "h5": {"n": primary["n"], "mean": primary["mean"], "lower": primary["lower"],
                   "upper": primary["upper"], "cap_free_sensitivity": cap_free,
                   "caps": caps},
            "difference_of_means": primary["mean"] - h4p["mean"]}}


# ─────────────────────────────── the run, §6.1 ───────────────────────────────

def _run(*, study_manifest: str, segments: Sequence[str], h4_estimate: str, out: str,
         _fixture: bool) -> Dict[str, Any]:
    if not AUTH.H5_STUDY_AGGREGATION_AUTHORIZED:                     # 1. FIRST
        raise H5AggregationUnauthorized(
            "the H5 study aggregation is UNAUTHORIZED. No file was opened, read or "
            "written.")
    F.check_free(out)                                                # 2.
    if len(segments) != SEGMENT_COUNT:
        F.refuse(f"{len(segments)} segment files: the study is exactly {SEGMENT_COUNT}; "
                 f"an incomplete study is not a smaller study")
    F.check_durable([study_manifest, *segments, h4_estimate], _fixture=_fixture)   # 3.
    h4 = load_h4_estimate(h4_estimate)                               # 4.
    manifest = load_manifest(study_manifest, _fixture=_fixture)      # 5.
    loaded, ends = [], []
    for path, entry in zip(segments, manifest["segments"]):          # 6.
        bind_header(F.read_header(path), entry, _fixture=_fixture)
        try:
            games = R4.load_games(path, allow_fixture=_fixture)
            ends.append(F.segment_end(path))
        except (R4.H4RunError, ValueError, KeyError, TypeError, IndexError) as e:
            F.refuse(f"segment {entry['segment']}: {e}")
        F.bind_games(games, entry["schedule"])
        loaded.append(games)
    result = {"analysis": "H5_CONFIRMATORY", "fixture": _fixture,          # 7.
              **estimate(loaded, ends, h4),
              "inputs": {p: F.sha256_file(p) for p in (study_manifest, *segments,
                                                       h4_estimate)},
              "written_at": F.now_iso()}
    parent = os.path.dirname(os.path.abspath(out))
    if not os.path.isdir(parent):
        os.mkdir(parent)                          # only now: a refusal leaves nothing
    F.write_create_only(out, result)
    return result


def run_confirmatory(*, study_manifest: str, segments: Sequence[str]) -> Dict[str, Any]:
    """PUBLIC ENTRY. Gated; the destination and the H4 estimate are the fixed,
    pinned paths in THIS repository; fixture data is refused: no argument relaxes it."""
    return _run(study_manifest=study_manifest, segments=segments,
                h4_estimate=os.path.join(R4.REPO_ROOT, H4_ESTIMATE),
                out=os.path.join(R4.REPO_ROOT, OUT), _fixture=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY."""
    import argparse
    ap = argparse.ArgumentParser(prog="h5_confirmatory_analysis",
                                 description="H5 confirmatory analysis. NOT AUTHORIZED.")
    ap.add_argument("--study-manifest", required=True)
    ap.add_argument("--segment", action="append", required=True)
    a = ap.parse_args(argv)
    try:
        run_confirmatory(study_manifest=a.study_manifest, segments=a.segment)
    except H5AggregationUnauthorized as e:
        print(str(e), file=sys.stderr)
        return EXIT_UNAUTHORIZED
    except F.H4AnalysisRefused as e:
        print(f"REFUSED, nothing written: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"estimate written: {OUT}")
    return EXIT_OK


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
