"""H4 PILOT FEASIBILITY -- outcome-blinded. Ungated: it reads no outcome.

Frozen by `docs/superpowers/2026-09-23-t1j-h4-analysis-card.md` (amended
6490055), §0, §1, §2.1 and §4.1. 🔴 THE CARD IS THE AUTHORITY: where this module
and the card disagree, the card wins and the disagreement is an amendment.

Reports, for exactly one pilot results file bound to the committed pilot
manifest: operational failures, the runtime projection, cap-affected pairs,
replay integrity, process behaviour and TRAJECTORY-ONLY concentration, then the
§1.4 rules. Nothing else.

🔴 THE BLIND IS PROCEDURAL (card §1.2). Moves determine outcomes, so a move list
could be replayed into a winner; what this module guarantees is that it neither
computes nor outputs one. It works on a BLINDED VIEW built from a whitelist
immediately after the reader returns, it never reads an outcome field of its
own accord and never imports the decision or confirmatory module, and its report
is refused -- not written -- if it carries an outcome field.

It also holds what the three H4 analysis entries SHARE -- the manifest binding
(§0.1), durability (§2.1) and the create-only atomic output (§4.1) -- because it
is the one module the other two may import. It imports neither of them.
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import h4_runner as R

CARD = "docs/superpowers/2026-09-23-t1j-h4-analysis-card.md"
RUNNER_CARD, FOURB_CARD, REPLACEMENT_CARD = R.CARDS

#: Card §0.1: what a manifest carries, and the header fields bound to it.
MANIFEST_CARDS = (CARD, RUNNER_CARD, FOURB_CARD, REPLACEMENT_CARD)
MANIFEST_CODE = tuple(f"scripts/GPU/alphazero/{m}" for m in (
    "h4_runner.py", "e4_screen_integration.py", "t1j_adapter.py", "e4_screen_runner.py",
    "h2_match_rules.py"))
BOUND = ("schedule_digest", "seeds", "incumbent_identity", "t1j_runtime", "cards", "code")

PILOT, STUDY = R.DESIGNS["pilot"], R.DESIGNS["study"]
SEGMENTS = {PILOT: 1, STUDY: 4}
PAIRS_PER_SEGMENT = {PILOT: 16, STUDY: 74}
PILOT_GAMES = 2 * PAIRS_PER_SEGMENT[PILOT]
SEGMENT_GAMES = 2 * PAIRS_PER_SEGMENT[STUDY]

#: Card §1.4, transcribed from replacement §5.4.
RUNTIME_LIMIT_S = 10_800
CAP_STOP_PAIRS = 4
PRECEDENCE = ("VOID", "STOP_RUNTIME", "STOP_CAP", "STOP_COLLAPSE")
COLLAPSE_STATEMENT = "empirically complete concentration in the pilot"
#: ⚠ NOT FROZEN BY THE CARD: §1.3 says "first-k-ply prefix frequencies" without k.
#: Plies 1-6 cover T1j's native-initial replies. Descriptive only; stops nothing.
PREFIX_KS = (1, 2, 3, 4, 5, 6)

#: Card §1.2: the blinded view is a WHITELIST. Outcome fields are never copied.
VIEW_START = ("task_id", "pair_id", "arm", "incumbent_colour", "t1j_colour", "seed",
              "game_index")
VIEW_RESULT = ("plies", "terminal_reason", "elapsed_s", "queries", "replays",
               "distinct_pids")
#: Card §4: a report carrying any of these keys is refused before it is written.
REPORT_FORBIDDEN = R.BLIND_FIELDS + ("interval", "mean", "lower", "upper",
                                     "outcome_patterns")

EXIT_OK, EXIT_REFUSED = 0, 4


class H4AnalysisRefused(Exception):
    """Refused: nothing was written. Never a statement about either engine."""


def refuse(msg: str):
    raise H4AnalysisRefused(msg)


# ───────────────────────────── shared: §4.1, §2.1 ─────────────────────────────

def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check_free(out: str) -> None:
    """Card §4.1: an occupied destination is refused BEFORE any computation."""
    if os.path.lexists(out):
        refuse(f"{out} is occupied; outputs are create-only")


def write_create_only(out: str, obj: Mapping[str, Any]) -> None:
    """Card §4.1: temp file IN THE DESTINATION DIRECTORY, fsynced, `os.link`ed to
    its name (which fails if the name exists), temp removed, directory fsynced.
    A refusal leaves neither the output nor the temporary file."""
    d = os.path.dirname(os.path.abspath(out))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{os.path.basename(out)}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, out)
        except FileExistsError:
            refuse(f"{out} is occupied; outputs are create-only")
    finally:
        os.unlink(tmp)
    dfd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def check_durable(paths: Sequence[str]) -> None:
    """Card §2.1: tracked by git at HEAD AND unmodified -- each file, separately,
    in the repository that contains it."""
    for p in paths:
        d, name = os.path.split(os.path.abspath(p))
        for args, what in ((["ls-files", "--error-unmatch", "--", name], "not tracked"),
                           (["diff", "--quiet", "HEAD", "--", name],
                            "modified against HEAD")):
            r = subprocess.run(["git", "-C", d, *args], capture_output=True, text=True)
            if r.returncode != 0:
                refuse(f"{p} is not durable: {what} (card §2.1)")


# ───────────────────────────── shared: binding, §0.1 ─────────────────────────────

def _load_json(path: str, what: str) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as e:
        refuse(f"{what} {path} is unreadable: {e}")


def load_manifest(path: str, stage: str, *, _fixture: bool) -> Dict[str, Any]:
    """A committed preregistration manifest, validated -- its digests RECOMPUTED."""
    m = _load_json(path, "manifest")
    if not isinstance(m, dict) or m.get("stage") != stage:
        refuse(f"{path}: a {stage} manifest is required, not "
               f"{m.get('stage') if isinstance(m, dict) else m!r}")
    entries = m.get("segments")
    if not isinstance(entries, list) or len(entries) != SEGMENTS[stage]:
        refuse(f"{path}: a {stage} manifest names exactly {SEGMENTS[stage]} segment(s)")
    runtime = {"depth": R.DEPTH, "query_timeout_s": R.QUERY_TIMEOUT_S,
               "replay_timeout_s": R.REPLAY_TIMEOUT_S, "ply_cap": R.PLY_CAP,
               "h4_acceptance": True}
    for i, e in enumerate(entries):
        where = f"{path}: segment entry {i}"
        missing = [k for k in ("segment", "schedule", *BOUND) if k not in e]
        if missing:
            refuse(f"{where} lacks {missing}")
        if e["segment"] != i:
            refuse(f"{where} is labelled segment {e['segment']!r}")
        try:
            R.check_schedule(e["schedule"], mode="fixture" if _fixture else
                             ("pilot" if stage == PILOT else "study"))
        except R.H4RunError as err:
            refuse(f"{where}: {err}")
        if len(e["schedule"]) != 2 * PAIRS_PER_SEGMENT[stage]:
            refuse(f"{where}: {len(e['schedule'])} tasks, not "
                   f"{2 * PAIRS_PER_SEGMENT[stage]}")
        if R.schedule_digest(e["schedule"]) != e["schedule_digest"]:
            refuse(f"{where}: schedule_digest does not recompute from the schedule")
        if e["seeds"] != [t["seed"] for t in e["schedule"]]:
            refuse(f"{where}: seeds are not the schedule's seeds")
        if sorted(e["cards"]) != sorted(MANIFEST_CARDS):
            refuse(f"{where}: cards must be exactly {list(MANIFEST_CARDS)}")
        if sorted(e["code"]) != sorted(MANIFEST_CODE):
            refuse(f"{where}: code must be exactly {list(MANIFEST_CODE)}")
        rt = e["t1j_runtime"]
        wrong = [k for k, v in runtime.items() if not isinstance(rt, dict) or rt.get(k) != v]
        if wrong:
            refuse(f"{where}: t1j_runtime {wrong} differ from the frozen runtime")
    return m


def read_header(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            header = json.loads(fh.readline())
    except (OSError, ValueError) as e:
        refuse(f"{path}: no readable header: {e}")
    if not isinstance(header, dict) or header.get("record_type") != "header":
        refuse(f"{path}: the first record is not a header")
    return header


def bind_header(header: Mapping[str, Any], stage: str, entry: Mapping[str, Any], *,
                _fixture: bool) -> None:
    """Card §0.1: the results header equals its manifest entry, FIELD BY FIELD."""
    want = R.FIXTURE_DESIGN if _fixture else stage
    if header.get("design") != want:
        refuse(f"design {header.get('design')!r}, not {want!r}: pilot, study and "
               f"fixture data are never interchangeable")
    if header.get("evidence") is not (not _fixture):
        refuse(f"evidence flag {header.get('evidence')!r} does not fit {want}")
    if header.get("segment") != entry["segment"]:
        refuse(f"segment {header.get('segment')!r} is not manifest slot {entry['segment']}")
    if "code" not in header:
        refuse("the header has no `code` field: the Python code that played is not "
               "bound, which needs the runner amendment of card §0.1")
    for f in BOUND:
        if header.get(f) != entry[f]:
            refuse(f"header `{f}` differs from the manifest")


def bind_games(games: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]]):
    """The games ARE the manifest's schedule, in its order -- ids, arms, colours, seeds."""
    keys = ("task_id", "pair_id", "arm", "incumbent_colour", "t1j_colour", "seed")
    got = [tuple(g["start"].get(k) for k in keys) + (g["start"].get("game_index"),)
           for g in games]
    want = [tuple(t[k] for k in keys) + (i,) for i, t in enumerate(schedule)]
    if got != want:
        refuse(f"the file's {len(got)} games are not the manifest's schedule of "
               f"{len(want)} tasks")


def segment_end(path: str) -> Dict[str, Any]:
    end = R.read_records(path)[-1]
    if end.get("record_type") != "segment_end" or end.get("complete") is not True:
        raise R.H4RunError(f"{path}: not a complete segment_end")
    return end


# ───────────────────────────── the blinded view, §1.2 ─────────────────────────────

def check_view(view: Sequence[Mapping[str, Any]]) -> None:
    for v in view:
        leaked = [k for k in v if k in R.BLIND_FIELDS]
        if leaked:
            refuse(f"the blinded view carries {leaked}")
        if v.get("terminal_reason") not in ("win", "cap"):
            refuse(f"terminal_reason {v.get('terminal_reason')!r} is not a category")


def blinded_view(games: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Built IMMEDIATELY after the reader returns, from a whitelist."""
    view = []
    for g in games:
        v = {k: g["start"][k] for k in VIEW_START}
        v.update({k: g["result"][k] for k in VIEW_RESULT})
        v["moves"] = [list(p["move"]) for p in g["plies"]]
        v["routines"] = [[p["ply"], p["mover"], p["source"]] for p in g["plies"]
                         if p["actor"] == "t1j"]
        v["all_accepted"] = all(o["outcome"] == "accepted" for o in g["processes"])
        v["position_digests_rederived"] = len(g["processes"])
        view.append(v)
    check_view(view)
    return view


def _pairs(view):
    return [(view[i], view[i + 1]) for i in range(0, len(view), 2)]


def _key(moves) -> str:
    return " ".join(f"{r},{c}" for r, c in moves)


def concentration(view: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Card §1.3: TRAJECTORY ONLY. Descriptive; it never drops or weights anything."""
    out: Dict[str, Any] = {"by_arm": {}}
    for arm in ("A", "B"):
        games = [v for v in view if v["arm"] == arm]
        seqs = collections.Counter(_key(v["moves"]) for v in games)
        out["by_arm"][arm] = {
            "games": len(games), "unique_games": len(seqs),
            "max_game_frequency": max(seqs.values()),
            "prefix_frequencies": {str(k): dict(collections.Counter(
                _key(v["moves"][:k]) for v in games)) for k in PREFIX_KS}}
    tuples = collections.Counter((_key(a["moves"]), _key(b["moves"]))
                                 for a, b in _pairs(view))
    out["pair_tuples"] = {"pairs": sum(tuples.values()), "distinct": len(tuples),
                          "max_frequency": max(tuples.values())}
    routines: Dict[str, Dict[str, Dict[str, int]]] = {}
    for v in view:
        for ply, colour, source in v["routines"]:
            d = routines.setdefault(str(ply), {}).setdefault(colour, {})
            d[source] = d.get(source, 0) + 1
    out["routines_by_ply_and_colour"] = routines
    return out


# ───────────────────────────── the rules, §1.4 ─────────────────────────────

def rules(view: Sequence[Mapping[str, Any]], end: Mapping[str, Any]) -> List[Dict[str, Any]]:
    setup_s, total_s = end["setup_s"], end["total_s"]
    mean_game_s = (total_s - setup_s) / PILOT_GAMES
    projected = setup_s + SEGMENT_GAMES * mean_game_s
    pairs = _pairs(view)
    cap_pairs = sum(1 for a, b in pairs if "cap" in (a["terminal_reason"],
                                                      b["terminal_reason"]))
    identical = concentration(view)["pair_tuples"]["max_frequency"]
    fired = []
    if projected > RUNTIME_LIMIT_S:
        fired.append({"rule": "STOP_RUNTIME", "projected_segment_s": projected,
                      "limit_s": RUNTIME_LIMIT_S,
                      "statement": "four-segment H4 is operationally infeasible; "
                                   "redesign separately; no re-segmenting"})
    if cap_pairs >= CAP_STOP_PAIRS:
        fired.append({"rule": "STOP_CAP", "cap_affected_pairs": cap_pairs,
                      "stop_at": CAP_STOP_PAIRS})
    if identical == PAIRS_PER_SEGMENT[PILOT]:
        fired.append({"rule": "STOP_COLLAPSE", "identical_pair_tuples": identical,
                      "of": len(pairs), "statement": COLLAPSE_STATEMENT})
    return fired


def verdict(fired: Sequence[Mapping[str, Any]]) -> str:
    names = {f["rule"] for f in fired}
    return next((p for p in PRECEDENCE if p in names), "PROCEED")


# ───────────────────────────── the report ─────────────────────────────

def _void_reason(e: BaseException) -> str:
    """The reader's message, unless it names an outcome field -- a VOID pilot is
    still a blinded one."""
    msg = f"{type(e).__name__}: {e}"
    if any(b in msg.lower() for b in R.BLIND_FIELDS):
        return f"{type(e).__name__}: message withheld -- it names an outcome field"
    return msg


def _run_void_summary(path: str) -> Optional[Dict[str, Any]]:
    try:
        last = R.read_records(path)[-1]
    except (OSError, ValueError, IndexError):
        return None
    if last.get("record_type") != "run_void":
        return None
    return {k: last.get(k) for k in ("stage", "classification", "exception")}


def build_report(manifest: str, results: str, *, _fixture: bool) -> Dict[str, Any]:
    """The whole report except `written_at`, JSON-normalised so it compares equal
    to itself read back. Refuses on any binding fault; VOIDs on any record fault."""
    m = load_manifest(manifest, PILOT, _fixture=_fixture)
    entry = m["segments"][0]
    header = read_header(results)
    bind_header(header, PILOT, entry, _fixture=_fixture)
    report: Dict[str, Any] = {
        "report": "H4_PILOT_FEASIBILITY", "design": header["design"],
        "pilot_results_sha256": sha256_file(results),
        "pilot_manifest_sha256": sha256_file(manifest),
        "analysis_card_sha256": sha256_file(os.path.join(R.REPO_ROOT, CARD))}
    try:
        games = R.load_games(results, allow_fixture=_fixture)
        end = segment_end(results)
    except (R.H4RunError, ValueError, KeyError, TypeError, IndexError) as e:
        report.update(verdict="VOID", rules_fired=[{
            "rule": "VOID", "reader": _void_reason(e),
            "run_void": _run_void_summary(results)}])
        return json.loads(json.dumps(report))
    bind_games(games, entry["schedule"])
    view = blinded_view(games)
    fired = rules(view, end)
    mean_game_s = (end["total_s"] - end["setup_s"]) / PILOT_GAMES
    report.update(
        verdict=verdict(fired), rules_fired=fired,
        operational={"segment_end": True, "reader": "passed", "games": len(view)},
        runtime={"setup_s": end["setup_s"], "pilot_wall_s": end["total_s"],
                 "mean_game_s": mean_game_s,
                 "projected_segment_s": end["setup_s"] + SEGMENT_GAMES * mean_game_s,
                 "limit_s": RUNTIME_LIMIT_S},
        caps={"cap_affected_pairs": sum(1 for a, b in _pairs(view) if "cap" in (
            a["terminal_reason"], b["terminal_reason"])),
              "cap_games": sum(1 for v in view if v["terminal_reason"] == "cap"),
              "stop_at": CAP_STOP_PAIRS},
        games=[{k: v[k] for k in ("task_id", "pair_id", "arm", "plies", "terminal_reason",
                                  "elapsed_s", "queries", "replays", "distinct_pids",
                                  "all_accepted", "position_digests_rederived")}
               for v in view],
        concentration=concentration(view))
    return json.loads(json.dumps(report))


def check_report_blind(report: Mapping[str, Any]) -> None:
    """Card §1.2, §4: refused, not written, if it carries an outcome field."""
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)
    leaked = sorted({k for k in keys(report) if k in REPORT_FORBIDDEN})
    leaked += R.blinding_violations(json.dumps(report, sort_keys=True))
    if leaked:
        refuse(f"the feasibility report carries outcome fields {sorted(set(leaked))}")


def _write_report(manifest: str, results: str, out: str, *, _fixture: bool) -> Dict[str, Any]:
    check_free(out)
    check_durable([manifest, results])
    report = build_report(manifest, results, _fixture=_fixture)
    report["written_at"] = now_iso()
    check_report_blind(report)
    write_create_only(out, report)
    return report


def write_feasibility_report(manifest: str, results: str, out: str) -> Dict[str, Any]:
    """PUBLIC ENTRY. Fixture data is refused here: no argument relaxes it."""
    return _write_report(manifest, results, out, _fixture=False)


# ─────────────────────────────────── the CLI ───────────────────────────────────

def _main(argv: Optional[Sequence[str]], *, _fixture: bool) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="h4_pilot_feasibility",
                                 description="H4 pilot feasibility (outcome-blinded).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        report = _write_report(a.manifest, a.results, a.out, _fixture=_fixture)
    except H4AnalysisRefused as e:
        print(f"REFUSED, nothing written: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"feasibility report written: {a.out} (rules verdict {report['verdict']})")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY."""
    return _main(argv, _fixture=False)


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
