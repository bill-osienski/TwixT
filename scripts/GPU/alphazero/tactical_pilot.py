"""THE PILOT TOOLING: post-segment verification, the outcome-blind pilot report, the sizing.

`docs/superpowers/2026-10-05-tactical-candidate-evaluation-card.md` §19 A9 (with A1.2, A1.4,
A2.1-A2.5, A8.5 item 5, A8.7).

VERIFY (A9.3) re-checks one completed segment outcome-blind -- everything R1's predecessor check
does except the verification record it produces -- and times itself in two parts, the per-game
loop and everything else, from this module's first statement.

REPORT (A9.4) reads, of the pilot, only what A1.2 allows: per game the pair, the arm, the
termination TYPE, the move list (for equality), each candidate decision's baseline and final move
and its scan and search times, the game's wall time; and the timing intervals. Never a winner, a
score, a per-colour result, a terminal ply, a game length, a replay summary or a variance.

SIZE (A9.5) re-derives the report's measurements from the evidence, then applies the frozen rule:
the Clopper-Pearson bound, sigma_plan, both n formulas, the method rule, whole-pair packing with
the workload-scaled verification estimate, and the projection against 80 h. n is never shrunk.

Every consumer enforces A8.5 item 5 through R1's own readers. Refusals write nothing; after them
every outcome -- a failed check, an exception, an interruption -- is recorded, create-only and
never visible incomplete. Gate: `tactical_pilot_authorization`, created closed.
"""
from __future__ import annotations

from time import monotonic as _monotonic      # A9.3: the verification's clock starts here
_T_START = _monotonic()

import hashlib
import json
import math
import os
import pathlib
import sys
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from . import candidate_match_runner as CM
from . import h4_pilot_feasibility as F
from . import h4_runner as R4
from . import tactical_pilot_authorization as AUTH

MODULE = "scripts.GPU.alphazero.tactical_pilot"
CARD = CM.CARD
PILOT_PAIRS = 16                                 # D5
DELTA, ALPHA = 0.05, 0.05                        # A2.2, A1.4
Z = NormalDist().inv_cdf(0.80)                   # A1.1: ~80 % planning power
HOEFFDING_CONST = math.sqrt(math.log(40) / 2)    # A1.1
EB_LOG = math.log(80)                            # A1.5: two-sided 95 %
CP_ALPHA = 0.05                                  # A1.2: one-sided 95 % Clopper-Pearson upper bound
SCAN_SHARE_MAX, CAPPED_PAIRS_STOP, PILOT_LIMIT_S = 0.25, 4, 14_400.0   # A2.4
CEILING_S, MARGIN = 288_000.0, 1.25              # A2.2
REPORT_DIR, SIZING_DIR = "tactical-candidate-pilot-report", "tactical-candidate-sizing"
ASSUMPTION = {"hoeffding": "independent pairs (Hoeffding 1963, general form)",
              "empirical_bernstein": "i.i.d. pairs, ASSUMED (A2.5) -- beyond the independence assumption"}
EXIT_OK, EXIT_REFUSED, EXIT_UNAUTHORIZED = 0, 4, 5


class PilotRefused(Exception):
    """A refusal: nothing was written."""


class PilotUnauthorized(Exception):
    """The gate is closed. Nothing was opened, read, computed or written."""


class PilotStop(Exception):
    """A check failed: recorded as a non-pass verdict."""


def _finite_nonneg(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v) and v >= 0


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _failure(step: str, e: BaseException) -> Dict[str, Any]:
    return {"step": step, "exception_type": type(e).__name__, "reason": str(e),
            "status": "INTERRUPTED" if isinstance(e, (KeyboardInterrupt, SystemExit)) else "FAILED"}


def _published_path(d: pathlib.Path, stem: str) -> pathlib.Path:
    return d / f"{stem}.published.json"


def _failed_path(d: pathlib.Path, stem: str) -> pathlib.Path:
    return d / f"{stem}.publication_failed.json"


def _publish(d: pathlib.Path, name: str, obj: Mapping[str, Any], owned: Optional[List[bool]] = None) -> str:
    """A9.2: the complete bytes, fsynced in a temporary file, published by a create-only link;
    then the directory is synced. A record is never visible incomplete. The temporary file is
    created exclusively under a held SIGINT and `owned` is marked before an interruption can
    surface: that creation is how `verify` claims its record."""
    raw = (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()
    tmp = d / f".{name}.partial"
    with CM._sigint_held():
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        if owned is not None:
            owned[0] = True
    try:
        CM._write_all(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.link(tmp, d / name)                       # fails if the record exists
    os.unlink(tmp)
    CM._fsync_dir(d)
    return _sha(raw)


def _publish_record(d: pathlib.Path, stem: str, record: Mapping[str, Any], at: List[str],
                    owned: Optional[List[bool]] = None) -> None:
    """A9.2: the record, then -- the LAST act -- its marker, bound to the record's sha256."""
    at[0] = "record"
    sha = _publish(d, f"{stem}.json", record, owned)
    at[0] = "marker"
    _publish(d, _published_path(d, stem).name, {"status": "PUBLISHED", "record": f"{stem}.json",
                                                "record_sha256": sha})


def _record_publication_failure(d: pathlib.Path, stem: str, e: BaseException, step: str) -> None:
    """A9.2: a failure or interruption once the destination is ours, recorded in a separate,
    create-only, fsynced file; an existing record is never rewritten."""
    target = _failed_path(d, stem)
    if target.exists():
        return
    body = {"card": CARD, "record": f"{stem}.json", "step": step,
            "status": "INTERRUPTED" if isinstance(e, (KeyboardInterrupt, SystemExit)) else "PUBLICATION_FAILED",
            "failure": {"exception_type": type(e).__name__, "reason": str(e)},
            "record_present": (d / f"{stem}.json").exists(), "published_present": _published_path(d, stem).exists()}
    try:
        with open(target, "x", encoding="utf-8") as fh:
            json.dump(body, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:                            # noqa: BLE001 -- nothing more can be recorded
        return
    try:
        CM._fsync_dir(d)
    except Exception:                            # noqa: BLE001 -- the record itself is written
        pass


def read_published(d: pathlib.Path, stem: str, bad: Callable[[str], Exception]) -> Dict[str, Any]:
    """A9.2: a CONSUMER's check of a tool record -- no recorded publication failure; the record and
    its marker committed; the marker PUBLISHED and bound to THIS record. Even when the record
    exists and parses. Returns the record."""
    if _failed_path(d, stem).exists():
        raise bad(f"{stem}.json: its publication failed ({_failed_path(d, stem).name})")
    path, marker = d / f"{stem}.json", _published_path(d, stem)
    try:
        F.check_durable([str(path), str(marker)], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise bad(f"{stem}.json was never published, or is not committed ({e})") from None
    m = json.loads(marker.read_text(encoding="utf-8"))
    if not isinstance(m, dict) or (m.get("status"), m.get("record")) != ("PUBLISHED", f"{stem}.json"):
        raise bad(f"{stem}.json is not PUBLISHED")
    if m.get("record_sha256") != CM._file_sha(str(path)):
        raise bad(f"{stem}.json is not the one published")
    return json.loads(path.read_text(encoding="utf-8"))


def _gate() -> None:
    if not AUTH.TACTICAL_PILOT_ANALYSIS_AUTHORIZED:
        raise PilotUnauthorized("the pilot tooling is UNAUTHORIZED. Nothing was opened, read, "
                                "computed or written.")


# ─────────────────────────────── A9.3: the post-segment verification ───────────────────────────

def _check_segment(role: str, k: int, d: pathlib.Path, at: List[str], hashes: Dict[str, Any],
                   loop_s: List[float]) -> int:
    """Every check of A9.3, in order; `at` names the current step. Raises PilotStop (or R1's
    MatchRefused) on the first failure. Returns the number of game files verified."""
    at[0] = "plan"
    plan, plan_sha = CM._committed_plan(role, k)
    at[0] = "timing"
    _secs, hashes["timing"] = CM._read_timing(d, PilotStop)
    hashes["watchdog"] = CM._file_sha(str(d / CM.WATCHDOG_FILE))
    at[0] = "finalization"
    CM._read_finalization(d, PilotStop, hashes["watchdog"], hashes["timing"])
    hashes["finalized"] = CM._file_sha(str(CM._finalized_path(d)))
    at[0] = "durability"
    try:
        F.check_durable([str(d / n) for n in (CM.CLAIM_FILE, "log.jsonl", "record.json", CM.WATCHDOG_FILE)],
                        _fixture=False)
    except F.H4AnalysisRefused as e:
        raise PilotStop(f"a core file is not committed ({e})") from None
    at[0] = "claim"
    claim = json.loads((d / CM.CLAIM_FILE).read_text(encoding="utf-8"))
    if (claim.get("plan_sha256"), claim.get("role"), claim.get("segment")) != (plan_sha, role, k):
        raise PilotStop("the claim is for another plan, role or segment")
    at[0] = "watchdog"
    w = json.loads((d / CM.WATCHDOG_FILE).read_text(encoding="utf-8"))
    raw = (d / "record.json").read_bytes()
    hashes["record"] = _sha(raw)
    if (w.get("status"), w.get("record_status")) != ("COMPLETE", "COMPLETE"):
        raise PilotStop(f"the watchdog says {w.get('status')}/{w.get('record_status')}")
    if (w.get("journal") or {}).get("sha256") != CM._file_sha(str(d / "log.jsonl")):
        raise PilotStop("the journal is not the one the watchdog hashed")
    if w.get("record_sha256") != hashes["record"]:
        raise PilotStop("the record is not the one the watchdog hashed")
    if not _finite_nonneg(w.get("elapsed_s")):
        raise PilotStop(f"the watchdog's elapsed time {w.get('elapsed_s')!r} is not a finite, nonnegative number")
    at[0] = "record"
    rec = json.loads(raw)
    if (rec.get("card"), rec.get("mode"), rec.get("status"), rec.get("plan_sha256")) != \
            (CARD, role, "COMPLETE", CM._sha(plan)) or rec.get("plan") != plan:
        raise PilotStop("the record is not a COMPLETE record of the committed plan")
    want = [(p["pair_id"], arm) for p in plan["pairs"] for arm in CM.ARMS]
    games = rec.get("games")
    if not isinstance(games, list) or [(g.get("pair_id"), g.get("arm")) for g in games] != want:
        raise PilotStop("its games are not the plan's pairs and arms, in order")
    if [x.get("pair_id") for x in rec.get("pairs") or []] != [p["pair_id"] for p in plan["pairs"]]:
        raise PilotStop("its replay checks do not name every planned pair")
    at[0] = "games"
    t0 = _monotonic()
    try:
        for g in games:                          # A9.3: the per-game loop, timed as t_games
            if g.get("file") != f"{g['pair_id']}-{g['arm']}.json":
                raise PilotStop(f"game {g['pair_id']}-{g['arm']} names the file {g.get('file')!r}")
            path = CM._game_file(d / CM.GAMES_DIR, g["pair_id"], g["arm"], ".json")
            try:
                F.check_durable([str(path)], _fixture=False)
            except F.H4AnalysisRefused as e:
                raise PilotStop(f"{path.name} is not committed ({e})") from None
            if CM._file_sha(str(path)) != g.get("sha256"):
                raise PilotStop(f"{path.name} is not the file the record hashed")
    finally:
        loop_s[0] = _monotonic() - t0
    return len(games)


def verify(role: str, segment: int) -> Dict[str, Any]:
    """PUBLIC ENTRY (A9.3). Refusals write nothing; then the verdict -- VERIFIED or NOT_VERIFIED
    -- is always recorded, an interruption re-raised after it."""
    _gate()
    try:
        CM._role(role, segment)
    except CM.MatchRefused as e:
        raise PilotRefused(str(e)) from None
    path = CM.verification_path(role, segment)
    stem, vdir = path.stem, path.parent
    leftovers = (path, vdir / f".{path.name}.partial", _published_path(vdir, stem),
                 vdir / f".{_published_path(vdir, stem).name}.partial", _failed_path(vdir, stem))
    if any(p.exists() for p in leftovers):
        raise PilotRefused(f"{path} is occupied: each segment is verified once")
    d = CM.out_dir(role, segment)
    if not d.is_dir():
        raise PilotRefused(f"{d}: no such segment to verify")
    path.parent.mkdir(parents=True, exist_ok=True)
    at, hashes, loop_s = ["start"], {"watchdog": None, "record": None, "timing": None, "finalized": None}, [0.0]
    failure, interrupt, game_files = None, None, None
    try:
        game_files = _check_segment(role, segment, d, at, hashes, loop_s)
    except (KeyboardInterrupt, SystemExit) as e:
        failure, interrupt = _failure(at[0], e), e
    except Exception as e:                       # noqa: BLE001 -- a recorded NOT_VERIFIED
        failure = _failure(at[0], e)
    t_games = loop_s[0]
    t_fixed = (_monotonic() - _T_START) - t_games    # A9.3: just before the record is written
    record = {"card": CARD, "role": role, "segment": segment,
              "verdict": "VERIFIED" if failure is None else "NOT_VERIFIED", "failure": failure,
              "watchdog_sha256": hashes["watchdog"], "record_sha256": hashes["record"],
              "timing_sha256": hashes["timing"], "finalized_sha256": hashes["finalized"],
              "game_files": game_files, "t_fixed_s": t_fixed, "t_games_s": t_games,
              "clock": "time.monotonic, from tactical_pilot's first statement",
              "note": "outcome-blind; its times feed a PLANNING ESTIMATE (A8.7), not a bound; the record's "
                      "own write is not timed"}
    owned, step = [False], ["record"]
    try:                                         # A9.2: from the claim on, every exit is recorded
        _publish_record(vdir, stem, record, step, owned)
    except BaseException as e:
        if owned[0]:
            _record_publication_failure(vdir, stem, e, step[0])
        raise
    if interrupt is not None:
        raise interrupt
    return record


# ─────────────────────────────── A9.4: the outcome-blind measurements ───────────────────────────

def blind_game(game: Mapping[str, Any]) -> Dict[str, Any]:
    """The ONLY fields of a game file the report reads (A1.2, A9.4): no winner, no terminal ply,
    no game length, no score."""
    return {"pair_id": game["pair_id"], "arm": game["arm"], "reason": game["terminal"]["reason"],
            "moves": game["moves"], "seconds": game["seconds"],
            "decisions": [{"baseline": list(w["baseline"][:2]), "final": list(w["final"][:2]),
                           "scan_s": w["scan_s"], "search_s": w["search_s"]} for w in game["witnesses"]]}


def measure(at: Optional[List[str]] = None) -> Dict[str, Any]:
    """The pilot's outcome-blind measurements, after every consumer check (A9.4); `at` names the
    current step for the record. Raises PilotStop."""
    at = at if at is not None else [""]
    d = CM.out_dir("pilot", 0)
    at[0] = "plan"
    try:
        plan, plan_sha = CM._committed_plan("pilot", 0)
    except CM.MatchRefused as e:
        raise PilotStop(f"the pilot plan: {e}") from None
    if len(plan["pairs"]) != PILOT_PAIRS:
        raise PilotStop(f"the pilot plan has {len(plan['pairs'])} pairs, not {PILOT_PAIRS} (D5)")
    at[0] = "durability"
    try:
        F.check_durable([str(d / n) for n in (CM.CLAIM_FILE, "record.json", CM.WATCHDOG_FILE)], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise PilotStop(f"a core file is not committed ({e})") from None
    raw = (d / "record.json").read_bytes()
    rec = json.loads(raw)
    at[0] = "timing"
    _secs, timing_sha = CM._read_timing(d, PilotStop)
    watchdog_sha = CM._file_sha(str(d / CM.WATCHDOG_FILE))
    at[0] = "finalization"
    CM._read_finalization(d, PilotStop, watchdog_sha, timing_sha)
    at[0] = "verification"
    CM._read_verification("pilot", 0, {"watchdog": watchdog_sha, "record": _sha(raw), "timing": timing_sha},
                          len(rec["games"]), PilotStop)
    at[0] = "status"
    w = json.loads((d / CM.WATCHDOG_FILE).read_text(encoding="utf-8"))
    if (w.get("status"), w.get("record_status"), rec.get("status")) != ("COMPLETE", "COMPLETE", "COMPLETE"):
        raise PilotStop("the pilot is not COMPLETE")
    at[0] = "games"
    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for entry in rec["games"]:
        path = CM._game_file(d / CM.GAMES_DIR, entry["pair_id"], entry["arm"], ".json")
        gr = path.read_bytes()
        if _sha(gr) != entry["sha256"]:
            raise PilotStop(f"{path.name} is not the file the record hashed")
        g = blind_game(json.loads(gr))
        by_pair.setdefault(g["pair_id"], {})[g["arm"]] = g
    if sorted(by_pair) != sorted(p["pair_id"] for p in plan["pairs"]) or \
            any(sorted(v) != sorted(CM.ARMS) for v in by_pair.values()):
        raise PilotStop("the games are not the plan's pairs, both arms")
    games = [g for v in by_pair.values() for g in v.values()]
    for g in games:
        if not _finite_nonneg(g["seconds"]) or \
                not all(_finite_nonneg(x["scan_s"]) and _finite_nonneg(x["search_s"]) for x in g["decisions"]):
            raise PilotStop(f"game {g['pair_id']}-{g['arm']}: a time that is not finite and nonnegative")
    at[0] = "measurements"
    t = json.loads(CM._timing_path(d).read_text(encoding="utf-8"))
    parts = {k: float(t[k]) for k in ("startup_s", "pre_launch_s", "child_s", "settlement_s")}
    fired = sorted(pid for pid, v in by_pair.items()
                   if any(x["baseline"] != x["final"] for g in v.values() for x in g["decisions"]))
    quiet_breaks = sorted(pid for pid, v in by_pair.items() if pid not in fired and
                          (v["A"]["moves"] != v["B"]["moves"] or v["A"]["reason"] != v["B"]["reason"]))
    capped = sorted(pid for pid, v in by_pair.items() if any(g["reason"] == "cap" for g in v.values()))
    scan = math.fsum(x["scan_s"] for g in games for x in g["decisions"])
    search = math.fsum(x["search_s"] for g in games for x in g["decisions"])
    G = math.fsum(g["seconds"] for g in games)
    return {"pairs": PILOT_PAIRS, "fired_pairs": len(fired), "firing_pair_ids": fired,
            "replay_ok": not quiet_breaks, "replay_breaks": quiet_breaks,
            "capped_pairs": len(capped), "capped_pair_ids": capped,
            "scan_share": scan / (scan + search) if scan + search > 0 else None,
            "G_s": G, "t_bar_s": G / PILOT_PAIRS, "timing": parts,
            "O_s": parts["startup_s"] + parts["pre_launch_s"] + (parts["child_s"] - G) + parts["settlement_s"],
            "pilot_s": math.fsum(parts.values()),
            "inputs": {"plan_sha256": plan_sha, "watchdog_sha256": watchdog_sha, "record_sha256": _sha(raw),
                       "timing_sha256": timing_sha,
                       "finalized_sha256": CM._file_sha(str(CM._finalized_path(d))),
                       "verification_sha256": CM._file_sha(str(CM.verification_path("pilot", 0)))}}


def stops(m: Mapping[str, Any]) -> List[str]:
    """A1.2 and A2.4, every one that applies."""
    out = []
    if m["fired_pairs"] == 0:
        out.append("f = 0: the candidate never acted (A1.2)")
    if not m["replay_ok"]:
        out.append(f"non-firing pairs not move-identical: {m['replay_breaks']} (A1.2)")
    if m["scan_share"] is None or m["scan_share"] > SCAN_SHARE_MAX:
        out.append(f"scan share {m['scan_share']} above {SCAN_SHARE_MAX} (A2.4)")
    if m["capped_pairs"] >= CAPPED_PAIRS_STOP:
        out.append(f"{m['capped_pairs']} of {PILOT_PAIRS} pairs capped (A2.4: stop at {CAPPED_PAIRS_STOP})")
    if m["pilot_s"] > PILOT_LIMIT_S:
        out.append(f"the pilot took {m['pilot_s']:.0f} s, above {PILOT_LIMIT_S:.0f} s (A2.4)")
    if m["G_s"] <= 0 or m["timing"]["child_s"] < m["G_s"]:
        out.append("invalid timing: no game time, or the child's interval is shorter than its games (A8.7)")
    return out


def _dest(name: str) -> pathlib.Path:
    return pathlib.Path(R4.EVIDENCE_ROOT) / name


def _recorded(out: pathlib.Path, stem: str, compute: Callable[[List[str]], Dict[str, Any]],
              stop_key: str) -> Dict[str, Any]:
    """A9.2: create the one-shot destination (ours under a held SIGINT); compute; ALWAYS record
    the outcome with its step -- a failed check IN the record, a failure from the creation through
    the publication in {stem}.publication_failed.json."""
    ours, step, interrupt = False, ["create"], None
    try:
        with CM._sigint_held():
            try:
                os.mkdir(out)
            except FileExistsError:
                raise PilotRefused(f"{out} is occupied: {stem} is one-shot") from None
            ours = True
        at = ["start"]
        try:
            record = compute(at)
        except (KeyboardInterrupt, SystemExit) as e:
            record, interrupt = {"verdict": "STOP", "failure": _failure(at[0], e)}, e
        except Exception as e:                   # noqa: BLE001 -- a recorded STOP
            record = {"verdict": "STOP", "failure": _failure(at[0], e)}
        record = {"card": CARD, **record}
        record.setdefault(stop_key, [])
        _publish_record(out, stem, record, step)
    except BaseException as e:
        if ours:
            _record_publication_failure(out, stem, e, step[0])
        raise
    if interrupt is not None:
        raise interrupt
    return record


def report() -> Dict[str, Any]:
    """PUBLIC ENTRY (A9.4). PROCEED_TO_SIZING or STOP with every stop that applies."""
    _gate()
    out = _dest(REPORT_DIR)
    if os.path.lexists(out):
        raise PilotRefused(f"{out} is occupied: the pilot report is one-shot")
    try:
        F.check_durable([str(CM.verification_path("pilot", 0))], _fixture=False)
    except F.H4AnalysisRefused as e:
        raise PilotRefused(f"the pilot's verification record is absent or uncommitted: verify first ({e})") from None

    def compute(at):
        m = measure(at)
        at[0] = "stops"
        s = stops(m)
        return {"verdict": "STOP" if s else "PROCEED_TO_SIZING", "stops": s, "measurements": m,
                "note": "outcome-blind (A1.2): no winner, score, per-colour result, terminal ply, game length, "
                        "replay summary or variance was read"}
    return _recorded(out, "report", compute, "stops")


# ─────────────────────────────── A9.5: the frozen sizing rule ───────────────────────────────

def _binom_cdf(f: int, n: int, p: float) -> float:
    return math.fsum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(f + 1))


def cp_upper(f: int, n: int = PILOT_PAIRS, alpha: float = CP_ALPHA) -> float:
    """The one-sided (1 - alpha) Clopper-Pearson upper bound for f of n; 1 at f = n."""
    if type(f) is not int or not 0 <= f <= n:
        raise PilotStop(f"f = {f!r} is not a count of {n} pairs")
    if f == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _binom_cdf(f, n, mid) > alpha else (lo, mid)
    return (lo + hi) / 2


def sigma_plan(p_u: float) -> float:
    return min(0.5, math.sqrt(p_u) / 2)


def n_hoeffding(sigma: float, delta: float = DELTA) -> int:
    return math.ceil(((HOEFFDING_CONST + Z * sigma) / delta) ** 2)


def n_eb(sigma: float, delta: float = DELTA) -> int:
    n = 2
    while delta < math.sqrt(2 * sigma ** 2 * EB_LOG / n) + 7 * EB_LOG / (3 * (n - 1)) + Z * sigma / math.sqrt(n):
        n += 1
    return n


def choose(nh: int, ne: int) -> str:
    """A2.1: the smaller n; a tie chooses Hoeffding."""
    return "hoeffding" if nh <= ne else "empirical_bernstein"


def pack(n: int, t_bar: float, O: float, V: Callable[[int], float]) -> List[int]:
    """A8.7: whole pairs, segments filled in order; c_j^max shrinks by the predecessors' validation."""
    segs, used = [], 0.0
    while sum(segs) < n:
        cap = math.floor((CM.SEGMENT_DEADLINE_S - CM.MARGIN_S - O - used) / t_bar)
        if cap < 1:
            raise PilotStop(f"segment {len(segs)} could hold {cap} pairs: one pair cannot fit a segment (A8.7)")
        c = min(cap, n - sum(segs))
        segs.append(c)
        used += V(c)
    return segs


def plan_study(f: int, t_bar: float, O: float, t_fixed: float, t_games: float, game_files: int) -> Dict[str, Any]:
    """The frozen rule (A1.4, A2.1-A2.2, A8.7), pure. Raises PilotStop on invalid inputs or capacity."""
    for name, v in (("t_bar", t_bar), ("O", O), ("t_fixed", t_fixed), ("t_games", t_games)):
        if not _finite_nonneg(v):
            raise PilotStop(f"{name} = {v!r} is not a finite, nonnegative number (A8.7)")
    if t_bar <= 0 or type(game_files) is not int or game_files <= 0:
        raise PilotStop(f"t_bar = {t_bar!r} and game_files = {game_files!r} must be positive (A8.7)")
    if not 1 <= f <= PILOT_PAIRS:
        raise PilotStop(f"f = {f!r}: sizing needs f >= 1 (A1.2)")
    p_u = cp_upper(f)
    s = sigma_plan(p_u)
    nh, ne = n_hoeffding(s), n_eb(s)
    method = choose(nh, ne)
    n = nh if method == "hoeffding" else ne
    b = t_games / game_files

    def V(c: int) -> float:
        return t_fixed + b * 2 * c
    segs = pack(n, t_bar, O, V)
    k = len(segs)
    verification = math.fsum((k + 1 - i) * V(c) for i, c in enumerate(segs))
    final = math.fsum(V(c) for c in segs)
    game_s, setup_s = n * t_bar, k * O
    projected = MARGIN * (game_s + setup_s + verification)
    return {"verdict": "PROCEED" if projected <= CEILING_S else "STOP",
            "stops": [] if projected <= CEILING_S else
            [f"projected {projected:.0f} s exceeds the {CEILING_S:.0f} s ceiling: stop before starting; "
             "n is never shrunk (A2.2)"],
            "f": f, "p_U": p_u, "sigma_plan": s, "delta": DELTA, "alpha": ALPHA, "z": Z,
            "n_hoeffding": nh, "n_eb": ne, "method": method, "method_assumption": ASSUMPTION[method], "n": n,
            "t_bar_s": t_bar, "O_s": O, "t_fixed_s": t_fixed, "b_s": b,
            "segments": [{"segment": j, "pairs": c,
                          "verification_reserve_s": V(c) + (final if j == k - 1 else 0.0)}
                         for j, c in enumerate(segs)],
            "k": k, "game_s": game_s, "setup_s": setup_s, "verification_s": verification,
            "projected_s": projected, "ceiling_s": CEILING_S, "margin": MARGIN,
            "notes": ["n is a planning approximation, not a distribution-free power guarantee (A2.3)",
                      "V(c) = t_fixed + b x 2c is a workload-scaled PLANNING ESTIMATE, not a bound (A8.7)"]}


def size() -> Dict[str, Any]:
    """PUBLIC ENTRY (A9.5). Refuses without a committed PROCEED_TO_SIZING report."""
    _gate()
    out = _dest(SIZING_DIR)
    if os.path.lexists(out):
        raise PilotRefused(f"{out} is occupied: the sizing is one-shot")
    rpath = _dest(REPORT_DIR) / "report.json"
    rep = read_published(_dest(REPORT_DIR), "report",
                         lambda why: PilotRefused(f"the pilot report is absent, uncommitted or unpublished: {why}"))
    if rep.get("verdict") != "PROCEED_TO_SIZING":
        raise PilotRefused(f"the pilot report says {rep.get('verdict')!r}: no sizing (A1.2, A2.4)")

    def compute(at):
        m = measure(at)
        at[0] = "report"
        if m != rep.get("measurements"):
            raise PilotStop("the report's measurements are not those of the evidence now")
        at[0] = "sizing"
        v = json.loads(CM.verification_path("pilot", 0).read_text(encoding="utf-8"))
        result = plan_study(m["fired_pairs"], m["t_bar_s"], m["O_s"], v["t_fixed_s"], v["t_games_s"],
                            v["game_files"])
        return {**result, "inputs": {**m["inputs"], "report_sha256": CM._file_sha(str(rpath))}}
    return _recorded(out, "sizing", compute, "stops")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY. The gate is read before the arguments are acted on."""
    import argparse
    ap = argparse.ArgumentParser(prog="tactical_pilot", description="Pilot tooling. NOT AUTHORIZED.")
    ap.add_argument("step", choices=("verify", "report", "size"))
    ap.add_argument("--role", choices=("pilot", "study"))
    ap.add_argument("--segment", type=int, default=0)
    a = ap.parse_args(argv)
    if not AUTH.TACTICAL_PILOT_ANALYSIS_AUTHORIZED:
        print("the pilot tooling is UNAUTHORIZED. Nothing was opened, read, computed or written.",
              file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        if a.step == "verify":
            if a.role is None:
                raise PilotRefused("verify needs --role")
            res = verify(a.role, a.segment)
        else:
            res = report() if a.step == "report" else size()
        print(json.dumps({"verdict": res["verdict"]}))
    except (PilotRefused, PilotUnauthorized) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
