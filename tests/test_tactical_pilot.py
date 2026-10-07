"""Pilot tooling (card §19 A9) -- SYNTHETIC ONLY.

No model and no game: every segment here is BUILT in R1's own formats -- R1's `_save_game`,
`_settle` and `_finalize` -- by `build()`, with test-only seeds (the registry stubbed) and
'committed' meaning present. The frozen sizing rule is pinned against the card's own tables.
"""
import contextlib
import json
import math
import os
import pathlib
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import candidate_match_runner as CM
from scripts.GPU.alphazero import h4_pilot_feasibility as F
from scripts.GPU.alphazero import h4_runner as R4
from scripts.GPU.alphazero import tactical_pilot as TP
from scripts.GPU.alphazero import tactical_pilot_authorization as AUTH

ROOT = pathlib.Path(__file__).resolve().parents[1]
MOVES = [[r, c] for r in range(1, 9) for c in (3, 7, 11, 15, 19)]          # 40 plies, any will do


@pytest.fixture
def ev(tmp_path, monkeypatch):
    """The gate open, the evidence root in tmp, the registry stubbed, committed == present."""
    monkeypatch.setattr(AUTH, "TACTICAL_PILOT_ANALYSIS_AUTHORIZED", True)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setattr(R4, "seed_registry_faults", lambda tasks: [])

    def durable(paths, _fixture):
        for x in paths:
            if not os.path.exists(x):
                raise F.H4AnalysisRefused(f"{x} is not committed")
    monkeypatch.setattr(F, "check_durable", durable)
    return tmp_path


def _witness(ply, fired, scan, search):
    base = [1, ply % 20 + 1, 5, 0.1]
    return {"ply": ply, "rule": "override" if fired else "inactive", "class": None, "reason": None,
            "override": fired, "unsearched": False, "baseline": base,
            "final": [2, ply % 20 + 1, 0, 0.0] if fired else list(base), "scan_s": scan, "search_s": search}


def _game(pid, arm, seeds, spec):
    cand = CM.ARMS[arm]
    moves = [list(m) for m in MOVES]
    if spec.get("moves_differ") and arm == "B":
        moves[-1] = [23, 23]
    own = [p for p in range(len(moves)) if p % 2 == (0 if cand == "red" else 1)]
    fire = spec.get("fired") and arm == "B"
    terminal = {"reason": spec.get(f"reason_{arm}", "win"), "ply": spec.get("ply", len(moves))}
    return {"pair_id": pid, "arm": arm, "candidate": cand,
            "roles": {c: "candidate" if c == cand else "incumbent" for c in CM.COLOURS}, "seeds": dict(seeds),
            "moves": moves, "terminal": terminal,
            "witnesses": [_witness(p, fire and p == own[0], spec.get("scan", 0.01), spec.get("search", 1.0))
                          for p in own],
            "digest": CM.game_digest(moves, terminal), "seconds": spec.get("seconds", 120.0)}


def build(role="pilot", k=0, specs=None, *, startup=2.0, pre=3.0, extra=50.0, settle=4.0, mutate=None,
          agree=None):
    """A completed, finalized segment in R1's formats. Default pilot: 16 pairs, pair 0 fires; every
    game 120 s (G = 3,840 s); timing 2 / 3 / G + 50 / 4, so O = 59 s and the pilot 3,899 s."""
    specs = [{"fired": i == 0} for i in range(16)] if specs is None else specs
    pairs = [{"pair_id": f"{role}-{i:02d}", "seeds": {"red": 1_000 + 2 * i, "black": 1_001 + 2 * i}}
             for i in range(len(specs))]
    plan = {"mode": role, "pairs": pairs}
    CM.plan_path(role, k).parent.mkdir(parents=True, exist_ok=True)
    CM.plan_path(role, k).write_text(json.dumps(plan))
    d = CM.out_dir(role, k)
    CM._create(d)
    CM._claim(d, {"plan_sha256": CM._file_sha(str(CM.plan_path(role, k))), "role": role, "segment": k})
    gdir = d / CM.GAMES_DIR
    os.mkdir(gdir)
    entries, records, G = [], [], 0.0
    for p, spec in zip(pairs, specs):
        for arm in CM.ARMS:
            g = _game(p["pair_id"], arm, p["seeds"], spec)
            if mutate is not None:
                mutate(g)
            entries.append(CM._save_game(gdir, g))
            G += g["seconds"]
        records.append({"pair_id": p["pair_id"], "fired": bool(spec.get("fired")),
                        "first_override": {"A": None, "B": 1 if spec.get("fired") else None},
                        "agree_through": len(MOVES) if agree is None else agree})
    (d / "log.jsonl").write_text(json.dumps({"step": "model_load", "observation": {}}) + "\n")
    rec = {"card": CM.CARD, "mode": role, "status": "COMPLETE", "failure": None, "plan_sha256": CM._sha(plan),
           "plan": plan, "games": entries, "pairs": records, "cleanup_failures": []}
    (d / "record.json").write_text(json.dumps(rec))
    CM._settle(d, {"terminated": False, "cause": None, "limit_s": 14_400.0, "elapsed_s": G + extra - 1.0,
                   "child_pid": 1, "rc": 0, "signal": None, "survivors_killed": False, "game_limit_s": 900.0,
                   "game_breach": None})
    m = [0.0, startup, startup + pre]
    m += [m[-1] + G + extra, m[-1] + G + extra + settle]
    CM._finalize(d, m)
    return d


@pytest.fixture
def scripted(monkeypatch):
    """The verification's clock: the loop from 10 s to 42 s (t_games 32), the record at 50 s
    (t_fixed 18) -- so b = 1 s per game file and V(c) = 18 + 2c."""
    ticks = iter([10.0, 42.0, 50.0])
    monkeypatch.setattr(TP, "_T_START", 0.0)
    monkeypatch.setattr(TP, "_monotonic", lambda: next(ticks))


def vrec(role="pilot", k=0):
    return json.loads(CM.verification_path(role, k).read_text())


# ─────────────────────────────── gate, CLI, names ───────────────────────────────

def test_the_gate_is_created_CLOSED_and_its_module_holds_only_it():
    import ast
    assert AUTH.TACTICAL_PILOT_ANALYSIS_AUTHORIZED is False
    body = ast.parse(pathlib.Path(AUTH.__file__).read_text()).body
    assert len(body) == 2 and isinstance(body[0], ast.Expr) and body[1].targets[0].id == "TACTICAL_PILOT_ANALYSIS_AUTHORIZED"


@pytest.mark.parametrize("entry,args", [("verify", ("pilot", 0)), ("report", ()), ("size", ())])
def test_every_public_entry_refuses_with_the_gate_shut_BEFORE_anything(entry, args, monkeypatch):
    for owner, name in ((CM, "_committed_plan"), (CM, "_read_timing"), (CM, "_read_finalization"),
                        (CM, "_read_verification"), (F, "check_durable"), (TP, "_publish"), (os, "mkdir")):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail("touched before the gate"))
    with pytest.raises(TP.PilotUnauthorized):
        getattr(TP, entry)(*args)


@pytest.mark.parametrize("argv", [["verify", "--role", "pilot"], ["report"], ["size"]])
def test_the_CLI_exits_5_and_takes_no_output_directory(argv, tmp_path):
    r = subprocess.run([sys.executable, "-m", TP.MODULE, *argv], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == TP.EXIT_UNAUTHORIZED == 5, r.stderr
    r = subprocess.run([sys.executable, "-m", TP.MODULE, *argv, "--out-dir", str(tmp_path)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 2 and "--out-dir" in r.stderr


def test_the_DESTINATIONS_are_the_contracts(ev):
    root = pathlib.Path(R4.EVIDENCE_ROOT)
    assert CM.verification_path("pilot", 0) == root / "tactical-candidate-verification" / "pilot-s0.json"
    assert CM.verification_path("study", 3) == root / "tactical-candidate-verification" / "study-s3.json"
    assert TP._dest(TP.REPORT_DIR) == root / "tactical-candidate-pilot-report"
    assert TP._dest(TP.SIZING_DIR) == root / "tactical-candidate-sizing"


# ─────────────────────────────── A9.3: the post-segment verification ───────────────────────────

def test_VERIFY_a_genuine_segment_and_R1_ACCEPTS_its_record(ev, scripted):
    """The record is exactly what R1's predecessor check reads: R1 validates the segment with it."""
    d = build("study", 0)
    rec = TP.verify("study", 0)
    assert (rec["verdict"], rec["failure"], rec["game_files"]) == ("VERIFIED", None, 32)
    assert (rec["t_games_s"], rec["t_fixed_s"]) == (32.0, 18.0), "the loop 10-42 s; the record at 50 s"
    assert rec == vrec("study", 0)
    assert (rec["watchdog_sha256"], rec["timing_sha256"], rec["finalized_sha256"]) == \
        tuple(CM._file_sha(str(d / n)) for n in (CM.WATCHDOG_FILE, "timing.json", "finalized.json"))
    timing = 2.0 + 3.0 + (3_840.0 + 50.0) + 4.0
    assert CM._validate_segment("study", 0) == timing + 18.0 + 32.0
    assert [x.name for x in CM.verification_path("study", 0).parent.iterdir()] == ["study-s0.json"], "no temporary"


def test_PUBLICATION_is_CREATE_ONLY(tmp_path):
    TP._publish(tmp_path, "r.json", {"a": 1})
    with pytest.raises(FileExistsError):
        TP._publish(tmp_path, "r.json", {"a": 2})
    assert json.loads((tmp_path / "r.json").read_text()) == {"a": 1}


def _unlink(name):
    return lambda d: (d / name).unlink()


def _write(name, text):
    return lambda d: (d / name).write_text(text)


def _edit(name, f):
    def edit(d):
        obj = json.loads((d / name).read_text())
        f(obj)
        (d / name).write_text(json.dumps(obj))
    return edit


def _refinalize(d):
    """Re-bind finalized.json to the watchdog as it is now."""
    _edit("finalized.json", lambda f: f.update(watchdog_sha256=CM._file_sha(str(d / CM.WATCHDOG_FILE))))(d)


def _rehash(d):
    """Re-bind the watchdog to the record as it is now, then the finalization to the watchdog."""
    _edit(CM.WATCHDOG_FILE, lambda w: w.update(record_sha256=CM._file_sha(str(d / "record.json"))))(d)
    _refinalize(d)


def _record_edit(f):
    return lambda d: (_edit("record.json", f)(d), _rehash(d))


#: (breakage, the step that fails, the reason) -- markers missing or in conflict, and the rest
VERIFY_BREAKS = {
    "finalized.json missing": (_unlink("finalized.json"), "finalization", "never finalized"),
    "a late failure recorded, never finalized": (
        lambda d: ((d / "finalized.json").unlink(), (d / "finalization_failed.json").write_text("{}")),
        "finalization", "finalization failed"),
    "CONFLICTING markers: finalized AND a late failure": (_write("finalization_failed.json", "{}"),
                                                          "finalization", "finalization failed"),
    "finalized.json bound another watchdog": (_edit("finalized.json", lambda f: f.update(watchdog_sha256="0" * 64)),
                                              "finalization", "bound other files"),
    "finalized.json not FINALIZED": (_edit("finalized.json", lambda f: f.update(status="PENDING")),
                                     "finalization", "not FINALIZED"),
    "timing NaN": (_edit("timing.json", lambda t: t.update(child_s=float("nan"))), "timing",
                   "not a finite, nonnegative"),
    "timing missing": (_unlink("timing.json"), "timing", "timing.json is not committed"),
    "the claim is for segment 3": (_edit(CM.CLAIM_FILE, lambda c: c.update(segment=3)), "claim",
                                   "another plan, role or segment"),
    "the watchdog says STOP": (_edit(CM.WATCHDOG_FILE, lambda w: w.update(status="STOP")), "finalization",
                               "bound other files"),
    "the journal altered": (_write("log.jsonl", "{}\n"), "watchdog", "journal is not the one"),
    "a game file altered": (lambda d: (d / CM.GAMES_DIR / "pilot-03-B.json").write_text("{}"), "games",
                            "pilot-03-B.json is not the file the record hashed"),
    "a game file missing": (lambda d: (d / CM.GAMES_DIR / "pilot-03-A.json").unlink(), "games", "not committed"),
    "the plan uncommitted": (lambda d: CM.plan_path("pilot", 0).unlink(), "plan", "must be committed"),
    "claim.json missing": (_unlink(CM.CLAIM_FILE), "durability", "not committed"),
    "the record altered": (_edit("record.json", lambda r: r.update(extra=1)), "watchdog",
                           "record is not the one the watchdog hashed"),
    "the watchdog's elapsed NaN, re-finalized": (
        lambda d: (_edit(CM.WATCHDOG_FILE, lambda w: w.update(elapsed_s=float("nan")))(d), _refinalize(d)),
        "watchdog", "elapsed time"),
    "the record STOP, rehashed": (_record_edit(lambda r: r.update(status="STOP")), "record",
                                  "not a COMPLETE record of the committed plan"),
    "the games reordered, rehashed": (_record_edit(lambda r: r["games"].reverse()), "record",
                                      "not the plan's pairs and arms"),
    "the replay checks missing, rehashed": (_record_edit(lambda r: r.update(pairs=[])), "record",
                                            "do not name every planned pair"),
    "a game names another file, rehashed": (_record_edit(lambda r: r["games"][4].update(file="x.json")), "games",
                                            "names the file"),
}


@pytest.mark.parametrize("name", VERIFY_BREAKS)
def test_VERIFY_records_NOT_VERIFIED_with_its_step_and_R1_REFUSES_it(ev, name):
    breakage, step, why = VERIFY_BREAKS[name]
    d = build()
    breakage(d)
    rec = TP.verify("pilot", 0)
    assert (rec["verdict"], rec["failure"]["step"], rec["failure"]["status"]) == ("NOT_VERIFIED", step, "FAILED")
    assert why in rec["failure"]["reason"], rec["failure"]
    with pytest.raises(CM.MatchRefused, match="NOT_VERIFIED|verified other files|for another segment|not committed"):
        CM._read_verification("pilot", 0, {"watchdog": rec["watchdog_sha256"] or "x",
                                           "record": rec["record_sha256"] or "x",
                                           "timing": rec["timing_sha256"] or "x"}, 32, lambda w: CM.MatchRefused(w))


def test_VERIFY_a_watchdog_that_says_STOP_while_its_finalization_still_binds(ev):
    """The watchdog edited AND re-finalized: the watchdog check itself must refuse."""
    d = build()
    _edit(CM.WATCHDOG_FILE, lambda w: w.update(status="STOP"))(d)
    _edit("finalized.json", lambda f: f.update(watchdog_sha256=CM._file_sha(str(d / CM.WATCHDOG_FILE))))(d)
    rec = TP.verify("pilot", 0)
    assert (rec["verdict"], rec["failure"]["step"]) == ("NOT_VERIFIED", "watchdog") and "STOP" in rec["failure"]["reason"]


def test_VERIFY_an_INTERRUPTION_is_recorded_then_RE_RAISED(ev, monkeypatch):
    build()
    real = CM._file_sha

    def sha(path):
        if path.endswith("pilot-05-A.json"):
            raise KeyboardInterrupt
        return real(path)
    monkeypatch.setattr(CM, "_file_sha", sha)
    with pytest.raises(KeyboardInterrupt):
        TP.verify("pilot", 0)
    rec = vrec()
    assert (rec["verdict"], rec["failure"]["step"], rec["failure"]["status"]) == ("NOT_VERIFIED", "games", "INTERRUPTED")
    assert rec["t_games_s"] >= 0 and math.isfinite(rec["t_fixed_s"])


def test_VERIFY_an_UNEXPECTED_exception_is_recorded(ev, monkeypatch):
    build()
    monkeypatch.setattr(CM, "_game_file", lambda *a: 1 / 0)
    rec = TP.verify("pilot", 0)
    assert (rec["verdict"], rec["failure"]["exception_type"], rec["failure"]["step"]) == \
        ("NOT_VERIFIED", "ZeroDivisionError", "games")


@pytest.mark.parametrize("case", ["occupied", "no segment", "unknown role", "pilot segment 1"])
def test_VERIFY_refusals_write_NOTHING(ev, case):
    if case == "occupied":
        build()
        path = CM.verification_path("pilot", 0)
        path.parent.mkdir(parents=True)
        path.write_text("theirs")
    args = {"unknown role": ("practice", 0), "pilot segment 1": ("pilot", 1)}.get(case, ("pilot", 0))
    with pytest.raises(TP.PilotRefused):
        TP.verify(*args)
    vdir = pathlib.Path(R4.EVIDENCE_ROOT) / "tactical-candidate-verification"
    if case == "occupied":
        assert sorted(os.listdir(vdir)) == ["pilot-s0.json"]
        assert CM.verification_path("pilot", 0).read_text() == "theirs"
    else:
        assert not vdir.exists()


def test_a_RECORD_is_never_visible_incomplete(ev, monkeypatch):
    build()
    monkeypatch.setattr(CM, "_write", lambda fd, data: 0)       # the record's bytes never land
    with pytest.raises(CM.MatchStop, match="no progress"):
        TP.verify("pilot", 0)
    assert not CM.verification_path("pilot", 0).exists()
    assert (CM.verification_path("pilot", 0).parent / ".pilot-s0.json.partial").exists()
    monkeypatch.setattr(CM, "_write", os.write)
    with pytest.raises(TP.PilotRefused, match="occupied"):
        TP.verify("pilot", 0)


# ─────────────────────────────── A9.4: the outcome-blind report ───────────────────────────────

class Strict(dict):
    """A mapping that FAILS the test on any read of a key it does not allow, and on enumeration."""

    def __init__(self, data, allowed):
        super().__init__(data)
        self.allowed = set(allowed)

    def __getitem__(self, k):
        assert k in self.allowed, f"read the forbidden field {k!r}"
        return super().__getitem__(k)

    def get(self, k, default=None):
        assert k in self.allowed, f"read the forbidden field {k!r}"
        return super().get(k, default)

    def __iter__(self):
        raise AssertionError("enumerated a game's fields")

    keys = values = items = __iter__


def test_the_REPORT_reads_ONLY_the_allowed_fields_of_a_game():
    g = _game("p", "A", {"red": 1, "black": 2}, {"fired": False})
    strict = Strict({**g, "winner": "red", "score": 1.0,
                     "terminal": Strict({**g["terminal"], "winner": "red"}, {"reason"}),
                     "witnesses": [Strict(w, {"baseline", "final", "scan_s", "search_s"}) for w in g["witnesses"]]},
                    {"pair_id", "arm", "terminal", "moves", "seconds", "witnesses"})
    out = TP.blind_game(strict)
    assert set(out) == {"pair_id", "arm", "reason", "moves", "seconds", "decisions"}
    assert set(out["decisions"][0]) == {"baseline", "final", "scan_s", "search_s"}


def _pipeline(scripted_clock=True):
    TP.verify("pilot", 0)
    return TP.report()


def test_REPORT_a_genuine_pilot_PROCEEDS_with_its_blind_measurements(ev, scripted):
    build()
    rep = _pipeline()
    m = rep["measurements"]
    assert (rep["verdict"], rep["stops"]) == ("PROCEED_TO_SIZING", [])
    assert (m["fired_pairs"], m["firing_pair_ids"], m["capped_pairs"], m["replay_ok"]) == (1, ["pilot-00"], 0, True)
    assert m["scan_share"] == pytest.approx(0.01 / 1.01)
    assert (m["G_s"], m["t_bar_s"], m["O_s"], m["pilot_s"]) == (3_840.0, 240.0, 59.0, 3_899.0)
    out = pathlib.Path(R4.EVIDENCE_ROOT) / TP.REPORT_DIR
    assert sorted(os.listdir(out)) == ["report.json"] and json.loads((out / "report.json").read_text()) == rep


def test_REPORT_LEAKS_no_outcome_and_is_INVARIANT_to_every_outcome_field(ev, scripted, tmp_path, monkeypatch):
    """Two pilots identical in what A1.2 allows, different in winners, scores, terminal plies and
    the record's replay summaries: the same measurements; and no blind field in the text."""
    def red(g):
        g.update(winner="red", score=1.0)
        g["terminal"]["ply"] = 39
    build(mutate=red, agree=17)
    a = _pipeline()
    text = (pathlib.Path(R4.EVIDENCE_ROOT) / TP.REPORT_DIR / "report.json").read_text()
    assert not R4.blinding_violations(text)
    for key in ('"winner"', '"score"', '"moves"', '"ply"', '"terminal"', '"agree_through"', '"seeds"'):
        assert key not in text, key
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(other))
    ticks = iter([10.0, 42.0, 50.0])
    monkeypatch.setattr(TP, "_monotonic", lambda: next(ticks))

    def black(g):
        g.update(winner="black", score=0.0)
        g["terminal"]["ply"] = 40
    build(mutate=black, agree=40)
    b = _pipeline()
    strip = lambda r: {k: v for k, v in r["measurements"].items() if k != "inputs"}   # noqa: E731
    assert strip(a) == strip(b) and a["verdict"] == b["verdict"]
    assert a["measurements"]["inputs"]["record_sha256"] != b["measurements"]["inputs"]["record_sha256"]


#: (specs or build kwargs, the stop's words) -- each at, and just past, its boundary
def _specs(n=16, **over):
    out = [{"fired": i == 0} for i in range(n)]
    for i, kw in over.items():
        out[int(i[1:])].update(kw)
    return out


STOPS = {
    "f = 0": ({"specs": [{} for _ in range(16)]}, "f = 0"),
    "a quiet pair whose moves differ": ({"specs": _specs(p3={"moves_differ": True})}, "not move-identical"),
    "a quiet pair whose termination differs": ({"specs": _specs(p3={"reason_B": "cap"})}, "not move-identical"),
    "scan share 1/3": ({"specs": [{"fired": i == 0, "scan": 1.0, "search": 2.0} for i in range(16)]}, "scan share"),
    "4 capped pairs": ({"specs": _specs(**{f"p{i}": {"reason_A": "cap", "reason_B": "cap"} for i in (2, 3, 4, 5)})},
                       "4 of 16 pairs capped"),
    "a pilot of 14,401 s": ({"extra": 10_552.0}, "above 14400 s"),
    "the child shorter than its games": ({"extra": -10.0}, "invalid timing"),
}
NO_STOP = {
    "scan share exactly 0.25": {"specs": [{"fired": i == 0, "scan": 1.0, "search": 3.0} for i in range(16)]},
    "3 capped pairs": {"specs": _specs(**{f"p{i}": {"reason_A": "cap", "reason_B": "cap"} for i in (2, 3, 4)})},
    "a pilot of exactly 14,400 s": {"extra": 10_551.0},
    "a FIRED pair whose games differ": {"specs": _specs(p0={"fired": True, "moves_differ": True})},
}


@pytest.mark.parametrize("name", STOPS)
def test_REPORT_STOPS_on_each_frozen_condition(ev, scripted, name):
    kw, why = STOPS[name]
    build(**kw)
    rep = _pipeline()
    assert rep["verdict"] == "STOP" and any(why in s for s in rep["stops"]), rep["stops"]


@pytest.mark.parametrize("name", NO_STOP)
def test_REPORT_does_NOT_stop_at_a_boundary(ev, scripted, name):
    build(**NO_STOP[name])
    rep = _pipeline()
    assert (rep["verdict"], rep["stops"]) == ("PROCEED_TO_SIZING", []), rep["stops"]


def test_REPORT_lists_EVERY_stop_that_applies(ev, scripted):
    build(specs=[{"reason_A": "cap", "reason_B": "cap"} for _ in range(16)])
    rep = _pipeline()
    assert len(rep["stops"]) == 2 and any("f = 0" in s for s in rep["stops"]) and \
        any("16 of 16 pairs capped" in s for s in rep["stops"])


def test_REPORT_STOPS_on_a_plan_that_is_not_16_pairs(ev, scripted):
    build(specs=[{"fired": True}] * 15)
    rep = _pipeline()
    assert (rep["verdict"], rep["failure"]["step"]) == ("STOP", "plan") and "not 16" in rep["failure"]["reason"]


def test_REPORT_STOPS_on_a_NEGATIVE_game_time(ev, scripted):
    build(mutate=lambda g: g.update(seconds=-1.0) if g["pair_id"] == "pilot-07" and g["arm"] == "A" else None)
    rep = _pipeline()
    assert (rep["verdict"], rep["failure"]["step"]) == ("STOP", "games")


#: consumer checks after the verification: markers, binding, files
CONSUMER = {
    "finalized.json removed after verification": (_unlink("finalized.json"), "finalization"),
    "a late failure recorded after verification": (_write("finalization_failed.json", "{}"), "finalization"),
    "the verification bound to another record": (
        lambda d: _edit("pilot-s0.json", lambda v: v.update(record_sha256="0" * 64))(
            CM.verification_path("pilot", 0).parent), "verification"),
    "the verification NOT_VERIFIED": (
        lambda d: _edit("pilot-s0.json", lambda v: v.update(verdict="NOT_VERIFIED"))(
            CM.verification_path("pilot", 0).parent), "verification"),
    "a game file altered after verification": (       # still a valid game: only the hash can tell
        lambda d: _edit("pilot-09-A.json", lambda g: g.update(seconds=1.0))(d / CM.GAMES_DIR), "games"),
}


@pytest.mark.parametrize("name", CONSUMER)
def test_REPORT_as_a_CONSUMER_STOPS_on_failed_or_incomplete_evidence(ev, scripted, name):
    breakage, step = CONSUMER[name]
    d = build()
    TP.verify("pilot", 0)
    breakage(d)
    rep = TP.report()
    assert (rep["verdict"], rep["failure"]["step"]) == ("STOP", step), rep["failure"]


def test_REPORT_refuses_WITHOUT_a_verification_record_and_writes_nothing(ev):
    build()
    with pytest.raises(TP.PilotRefused, match="verify first"):
        TP.report()
    assert not (pathlib.Path(R4.EVIDENCE_ROOT) / TP.REPORT_DIR).exists()


def test_REPORT_an_INTERRUPTION_is_recorded_with_its_step_then_RE_RAISED(ev, scripted, monkeypatch):
    build()
    TP.verify("pilot", 0)

    def boom(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(CM, "_read_finalization", boom)
    with pytest.raises(KeyboardInterrupt):
        TP.report()
    rep = json.loads((pathlib.Path(R4.EVIDENCE_ROOT) / TP.REPORT_DIR / "report.json").read_text())
    assert (rep["verdict"], rep["failure"]["step"], rep["failure"]["status"]) == ("STOP", "finalization", "INTERRUPTED")


def test_REPORT_is_ONE_SHOT(ev, scripted):
    build()
    _pipeline()
    with pytest.raises(TP.PilotRefused, match="one-shot"):
        TP.report()


# ─────────────────────────────── A9.5: the frozen sizing rule ───────────────────────────────

#: A2.2's table, extended to every f; computed once, here PINNED
N_TABLE = {1: (992, 734), 2: (1031, 858), 3: (1063, 969), 4: (1091, 1071), 5: (1116, 1167), 6: (1138, 1257),
           7: (1159, 1342), 8: (1178, 1423), 9: (1195, 1500), 10: (1211, 1572), 11: (1226, 1639),
           12: (1239, 1700), 13: (1250, 1755), 14: (1259, 1799), 15: (1265, 1828), 16: (1266, 1832)}
P_U = {0: 0.171, 1: 0.264, 2: 0.344, 3: 0.417, 4: 0.484, 5: 0.548, 8: 0.721, 12: 0.910, 16: 1.000}
SIGMA = {1: 0.257, 2: 0.293, 3: 0.323, 4: 0.348, 5: 0.370, 8: 0.425, 12: 0.477, 16: 0.500}


@pytest.mark.parametrize("f", sorted(N_TABLE))
def test_the_SIZING_reproduces_the_frozen_table_for_every_f(f):
    s = TP.sigma_plan(TP.cp_upper(f))
    nh, ne = TP.n_hoeffding(s), TP.n_eb(s)
    assert (nh, ne) == N_TABLE[f]
    assert TP.choose(nh, ne) == ("empirical_bernstein" if f <= 4 else "hoeffding")


def test_the_CLOPPER_PEARSON_bound_and_SIGMA_PLAN_are_A2_2s():
    assert {f: round(TP.cp_upper(f), 3) for f in P_U} == P_U
    assert {f: round(TP.sigma_plan(TP.cp_upper(f)), 3) for f in SIGMA} == SIGMA
    assert TP._binom_cdf(1, 16, TP.cp_upper(1)) == pytest.approx(0.05, abs=1e-12), "one-sided 95 %"


@pytest.mark.parametrize("delta,nh,ne", [(0.079, 508, 818), (0.05, 1266, 1832), (0.04, 1978, 2747),
                                         (0.03, 3517, 4673), (0.02, 7912, 10032)])
def test_the_n_FORMULAS_reproduce_A1_1s_worst_case_table(delta, nh, ne):
    assert (TP.n_hoeffding(0.5, delta), TP.n_eb(0.5, delta)) == (nh, ne)


def test_the_PRINTED_constants_give_the_same_n_for_every_f(monkeypatch):
    exact = {f: (TP.n_hoeffding(TP.sigma_plan(TP.cp_upper(f))), TP.n_eb(TP.sigma_plan(TP.cp_upper(f))))
             for f in N_TABLE}
    monkeypatch.setattr(TP, "Z", 0.8416)
    monkeypatch.setattr(TP, "HOEFFDING_CONST", 1.3581)
    assert {f: (TP.n_hoeffding(TP.sigma_plan(TP.cp_upper(f))), TP.n_eb(TP.sigma_plan(TP.cp_upper(f))))
            for f in N_TABLE} == exact


def test_the_CONSTANTS_are_the_cards():
    assert (TP.DELTA, TP.ALPHA, TP.CP_ALPHA, TP.PILOT_PAIRS) == (0.05, 0.05, 0.05, 16)
    assert (TP.SCAN_SHARE_MAX, TP.CAPPED_PAIRS_STOP, TP.PILOT_LIMIT_S) == (0.25, 4, 14_400.0)
    assert (TP.CEILING_S, TP.MARGIN) == (80 * 3600.0, 1.25)
    assert TP.Z == pytest.approx(0.8416, abs=5e-5) and TP.HOEFFDING_CONST == pytest.approx(1.3581, abs=5e-5)
    assert TP.EB_LOG == math.log(80)


@pytest.mark.parametrize("nh,ne,method", [(1000, 1000, "hoeffding"), (1000, 999, "empirical_bernstein"),
                                          (999, 1000, "hoeffding")])
def test_the_METHOD_rule_takes_the_smaller_n_and_a_tie_goes_to_HOEFFDING(nh, ne, method):
    assert TP.choose(nh, ne) == method


def test_PACKING_is_by_WHOLE_pairs():
    """A8.7's example: 14 pairs fit a segment, so 100 pairs need 8 segments, not 7."""
    segs = TP.pack(100, 1_000.0, 340.0, lambda c: 0.0)
    assert segs == [14] * 7 + [2] and len(segs) == 8 == math.ceil(100 / 14)


def test_PACKING_charges_each_segment_its_predecessors_VALIDATION():
    segs = TP.pack(40, 1_000.0, 400.0, lambda c: 100.0 + 2 * c)
    assert segs == [13, 13, 13, 1]           # 13,940 s, then -126, -252, -378: floor 13, 13, 13
    with pytest.raises(TP.PilotStop, match="one pair cannot fit a segment"):
        TP.pack(40, 1_000.0, 400.0, lambda c: 6_000.0)    # 13,940 -> 7,940 -> 1,940 -> < 1 pair


def test_PACKING_refuses_a_capacity_below_ONE_pair():
    with pytest.raises(TP.PilotStop, match="segment 0 could hold 0 pairs"):
        TP.pack(10, 20_000.0, 0.0, lambda c: 0.0)


def test_the_PROJECTION_assembled_by_hand(monkeypatch):
    """n = 30 (stubbed); t-bar 1,000; O 400; V(c) = 100 + 2c. Segments 13, 13, 4; verification
    4 x 126 + 3 x 126 + 2 x 108 = 1,098; projected = 1.25 x (30,000 + 1,200 + 1,098) = 40,372.5."""
    monkeypatch.setattr(TP, "n_hoeffding", lambda s, delta=TP.DELTA: 30)
    monkeypatch.setattr(TP, "n_eb", lambda s, delta=TP.DELTA: 40)
    r = TP.plan_study(8, 1_000.0, 400.0, 100.0, 32.0, 32)
    assert (r["method"], r["n"], r["k"], r["b_s"]) == ("hoeffding", 30, 3, 1.0)
    assert [s["pairs"] for s in r["segments"]] == [13, 13, 4]
    assert (r["game_s"], r["setup_s"], r["verification_s"], r["projected_s"]) == (30_000.0, 1_200.0, 1_098.0, 40_372.5)
    assert [s["verification_reserve_s"] for s in r["segments"]] == [126.0, 126.0, 108.0 + 360.0]
    assert (r["verdict"], r["stops"]) == ("PROCEED", [])


@pytest.mark.parametrize("n,verdict", [(240, "PROCEED"), (241, "STOP")])
def test_the_PROJECTION_at_EXACTLY_80_h_proceeds(monkeypatch, n, verdict):
    """t-bar 960, no overhead, no verification: 1.25 x 240 x 960 = 288,000 s exactly."""
    monkeypatch.setattr(TP, "n_hoeffding", lambda s, delta=TP.DELTA: n)
    monkeypatch.setattr(TP, "n_eb", lambda s, delta=TP.DELTA: n + 1)
    r = TP.plan_study(8, 960.0, 0.0, 0.0, 0.0, 32)
    assert (r["verdict"], r["projected_s"] == TP.CEILING_S) == (verdict, n == 240)


def test_the_PROJECTION_STOPS_above_80_h_and_never_shrinks_n(monkeypatch):
    monkeypatch.setattr(TP, "n_hoeffding", lambda s, delta=TP.DELTA: 300)
    monkeypatch.setattr(TP, "n_eb", lambda s, delta=TP.DELTA: 400)
    r = TP.plan_study(8, 1_000.0, 400.0, 100.0, 32.0, 32)
    assert (r["verdict"], r["n"]) == ("STOP", 300) and r["projected_s"] > TP.CEILING_S
    assert "never shrunk" in r["stops"][0]


@pytest.mark.parametrize("args,why", [
    ((1, float("nan"), 59.0, 18.0, 32.0, 32), "t_bar"), ((1, 240.0, -1.0, 18.0, 32.0, 32), "O = -1.0"),
    ((1, 240.0, 59.0, float("inf"), 32.0, 32), "t_fixed"), ((1, 240.0, 59.0, 18.0, -2.0, 32), "t_games"),
    ((1, 0.0, 59.0, 18.0, 32.0, 32), "must be positive"), ((1, 240.0, 59.0, 18.0, 32.0, 0), "must be positive"),
    ((1, 240.0, 59.0, 18.0, 32.0, True), "must be positive"), ((0, 240.0, 59.0, 18.0, 32.0, 32), "f >= 1"),
], ids=["t_bar NaN", "O negative", "t_fixed infinite", "t_games negative", "t_bar zero", "no game files",
        "game files a boolean", "f = 0"])
def test_the_SIZING_refuses_INVALID_timings(args, why):
    with pytest.raises(TP.PilotStop, match=why):
        TP.plan_study(*args)


def test_SIZE_end_to_end_on_a_genuine_pilot(ev, scripted):
    """f = 1 -> EB, n = 734; t-bar 240 s, O 59 s; V(c) = 18 + 2c from the verification's record."""
    build()
    _pipeline()
    s = TP.size()
    assert (s["verdict"], s["f"], s["method"], s["n"]) == ("PROCEED", 1, "empirical_bernstein", 734)
    assert "i.i.d. pairs, ASSUMED" in s["method_assumption"]
    assert (s["t_bar_s"], s["O_s"], s["t_fixed_s"], s["b_s"]) == (240.0, 59.0, 18.0, 1.0)
    pairs = [x["pairs"] for x in s["segments"]]
    assert sum(pairs) == 734 and len(pairs) == s["k"]
    V = [18.0 + 2.0 * c for c in pairs]
    assert s["verification_s"] == pytest.approx(math.fsum((s["k"] + 1 - i) * v for i, v in enumerate(V)))
    assert s["projected_s"] == pytest.approx(1.25 * (734 * 240.0 + s["k"] * 59.0 + s["verification_s"]))
    assert [x["verification_reserve_s"] for x in s["segments"]][-1] == pytest.approx(V[-1] + math.fsum(V))
    assert any("planning approximation" in n for n in s["notes"]) and any("PLANNING ESTIMATE" in n for n in s["notes"])
    assert s["inputs"]["report_sha256"] == CM._file_sha(str(pathlib.Path(R4.EVIDENCE_ROOT) / TP.REPORT_DIR / "report.json"))


def test_SIZE_a_slow_pilot_STOPS_before_starting(ev, scripted):
    build(specs=[{"fired": i == 0, "seconds": 300.0} for i in range(16)], extra=50.0)
    _pipeline()
    s = TP.size()
    assert (s["verdict"], s["n"]) == ("STOP", 734) and s["projected_s"] > TP.CEILING_S


def test_SIZE_refuses_a_STOP_report_and_an_absent_one(ev, scripted):
    with pytest.raises(TP.PilotRefused, match="absent or uncommitted"):
        TP.size()
    build(specs=[{} for _ in range(16)])
    assert _pipeline()["verdict"] == "STOP"
    with pytest.raises(TP.PilotRefused, match="no sizing"):
        TP.size()
    assert not (pathlib.Path(R4.EVIDENCE_ROOT) / TP.SIZING_DIR).exists()


@pytest.mark.parametrize("name,breakage,step", [
    ("the report altered", lambda d: _edit("report.json", lambda r: r["measurements"].update(fired_pairs=2))(
        pathlib.Path(R4.EVIDENCE_ROOT) / TP.REPORT_DIR), "report"),
    ("a late failure recorded after the report", _write("finalization_failed.json", "{}"), "finalization"),
    ("finalized.json removed after the report", _unlink("finalized.json"), "finalization"),
], ids=["report altered", "late failure", "finalized removed"])
def test_SIZE_as_a_CONSUMER_STOPS_on_a_changed_report_or_failed_finalization(ev, scripted, name, breakage, step):
    d = build()
    _pipeline()
    breakage(d)
    s = TP.size()
    assert (s["verdict"], s["failure"]["step"]) == ("STOP", step), s.get("failure")
