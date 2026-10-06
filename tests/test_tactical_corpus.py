"""Tactical candidate corpus -- the generator and the parent's publication (card §14 A4, §15 A5).

SYNTHETIC ONLY: the 8,000-game generation is NOT run. Runs use a SMALL specification (seeds 0-7
of each frozen stream, otherwise the frozen settings), small synthetic sources, and -- where the
point is not the checker -- a fast stand-in for the cross-check. No model, no decision.
"""
import contextlib
import copy
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import pytest

from scripts.GPU.alphazero import candidate_match_runner as CM
from scripts.GPU.alphazero import h4_pilot_feasibility as F
from scripts.GPU.alphazero import h4_runner as R4
from scripts.GPU.alphazero import h5_cap_completion as CC
from scripts.GPU.alphazero import h5_incumbent_tactics as T
from scripts.GPU.alphazero import tactical_candidate as TC
from scripts.GPU.alphazero import tactical_candidate_authorization as AUTH
from scripts.GPU.alphazero import tactical_corpus as TQ
from tests import test_candidate_match_runner as TR
from tests import test_h5_cap_completion as TCC
from tests import test_h5_incumbent_tactics as TT

ROOT = pathlib.Path(__file__).resolve().parents[1]
SMALL = {**TQ.FROZEN_SPEC, "seeds": [0, 8]}   # 16 games: one win, tq24-pegs:3 (end 193)
EMPTY = {"fixtures": {}, "h5_corpus": {"selected": [], "fixture_equivalent": []},
         "h5_record": {"rows": []}, "prior": {"pinned": {}, "standin": {}, "bases": []}}


def moves_sha(moves):
    return hashlib.sha256(json.dumps(moves).encode()).hexdigest()[:16]


def claimed(path):
    path.mkdir(parents=True, exist_ok=True)
    d = path / TQ.OUT_NAME
    TQ._claim(d, {})
    return d


def fast_check(state):
    return {"agree": True, "replies_examined": 0, "classify": None, "exhaustive": None,
            "new_opponent_wins": []}


# ─────────────────────────────── the frozen specification and the pinned sources ─────────────

def test_the_FROZEN_SPECIFICATION_is_the_cards():
    assert TQ.FROZEN_SPEC == {
        "streams": ["tq24-uniform", "tq24-pegs"], "seeds": [0, 4000], "p_near": 0.7,
        "knight": [[1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]],
        "window": 20, "min_ply": 10, "max_ply": 270, "quota": 20, "select": "tq24-select",
        "cap": 280, "board": 24, "python": "3.14.7"}
    assert (TQ.HARD_LIMIT_S, TQ.MARGIN_S, TQ.OUT_NAME) == (10_800, 60, "tactical-candidate-corpus")
    assert (TQ.PROVISIONAL, TQ.CORPUS) == ("corpus.provisional.json", "corpus.json")
    assert TQ.out_dir() == pathlib.Path(R4.EVIDENCE_ROOT) / "tactical-candidate-corpus"


def test_the_SOURCES_are_bound_to_their_pinned_hashes(monkeypatch):
    s = TQ._sources()
    assert len(s["fixtures"]) == 10 and len(s["h5_corpus"]["selected"]) == 63
    assert len(s["h5_corpus"]["fixture_equivalent"]) == 23 and len(s["h5_record"]["rows"]) == 60
    for name in ("H5_CORPUS_SHA256", "H5_RECORD_SHA256", "PRIOR_SHA256"):
        with monkeypatch.context() as m:
            m.setattr(TQ, name, "0" * 64)
            with pytest.raises(TQ.CorpusRefused, match="not the pinned file"):
                TQ._sources()


def test_the_PRIOR_FILE_still_matches_the_suite_and_the_STAND_IN():
    prior = json.loads(TQ.PRIOR_FILE.read_text())
    for name in ("DRAW24", "OWN_WIN_279", "THREAT_279", "UNDEFENDABLE_256", "OWN_WIN_AND_THREAT_264",
                 "SINGLE_260", "MULTI_269"):
        assert prior["pinned"][name] == [list(m) for m in getattr(TT, name)], name
    assert (prior["pinned"]["ONE"], prior["pinned"]["TWO"]) == ([list(m) for m in TCC.ONE],
                                                                [list(m) for m in TCC.TWO])
    for pair in (TR.FIRED, TR.QUIET):
        for arm in "AB":
            assert prior["standin"][f"{pair['pair_id']}-{arm}"] == \
                CM.play_game(TR.fake_make(), pair, arm, limit_s=600)["moves"], (pair, arm)
    assert {(b["pinned"], b["plies"]) for b in prior["bases"]} == \
        {(n, len(prior["pinned"][n])) for n in prior["pinned"]} | {("ONE", 23), ("DRAW24", 40), ("DRAW24", 569)}


# ─────────────────────────────── generation: policies, reproducibility, the window ────────────

class ScriptedRng:
    """Records the draws; random() returns `u`; choice() returns the first element."""

    def __init__(self, u):
        self.u, self.calls = u, []

    def random(self):
        self.calls.append("random")
        return self.u

    def choice(self, seq):
        self.calls.append(("choice", len(seq)))
        return seq[0]


def test_the_MOVE_POLICIES_and_their_EXACT_draw_order():
    empty = T.replay([])
    r = ScriptedRng(0.0)
    TQ._move("tq24-uniform", r, empty, SMALL)
    assert r.calls == [("choice", len(empty.legal_moves()))]
    r = ScriptedRng(0.0)
    TQ._move("tq24-pegs", r, empty, SMALL)
    assert r.calls == [("choice", len(empty.legal_moves()))], "no own peg: NO u is drawn"
    s = T.replay([(10, 10), (5, 5)])                  # red to move, one red peg
    near = [m for m in sorted(s.legal_moves()) if any((m[0] - 10, m[1] - 10) == tuple(k)
                                                       for k in SMALL["knight"])]
    for u, n in ((0.69, len(near)), (0.7, len(s.legal_moves()))):
        r = ScriptedRng(u)
        TQ._move("tq24-pegs", r, s, SMALL)
        assert r.calls == ["random", ("choice", n)], u
    with pytest.raises(TQ.CorpusRefused, match="no move policy"):
        TQ._move("tq24-other", ScriptedRng(0.0), empty, SMALL)


def test_GENERATION_is_REPRODUCIBLE_and_pinned():
    for stream, seed, sha, reason, end in (("tq24-uniform", 0, "8e0e41050fc6e5e6", "cap", 280),
                                           ("tq24-pegs", 0, "734c2aa016c0013e", "cap", 280),
                                           ("tq24-pegs", 3, "44061b61c22e37ee", "win", 193)):
        a, b = TQ.play(stream, seed, SMALL), TQ.play(stream, seed, SMALL)
        assert a == b and (moves_sha(a["moves"]), a["reason"], a["end"]) == (sha, reason, end)


def test_a_game_ends_at_the_FIRST_WINNER_or_the_CAP():
    g = TQ.play("tq24-pegs", 3, SMALL)
    assert T.replay(g["moves"]).winner() is not None and T.replay(g["moves"][:-1]).winner() is None
    c = TQ.play("tq24-uniform", 0, SMALL)
    assert len(c["moves"]) == 280 and T.replay(c["moves"]).winner() is None


def test_the_WINDOW_is_the_last_20_plies_of_a_WON_game_inside_10_270():
    g = TQ.play("tq24-pegs", 3, SMALL)
    w = TQ.window(g, SMALL)
    assert [p["ply"] for p in w] == list(range(173, 193))
    for p in w:
        s = T.replay(g["moves"][:p["ply"]])
        assert (p["class"], p["digest"]) == (T.classify(s, semantics="protocol")["class"], T.state_digest(s))
    assert TQ.window(TQ.play("tq24-uniform", 0, SMALL), SMALL) == [], "a capped game contributes nothing"
    late = {"reason": "win", "end": 285, "moves": [list(m) for m in TT.DRAW24[:285]]}
    assert [p["ply"] for p in TQ.window(late, SMALL)] == list(range(265, 271)), "never past ply 270"
    early = {"reason": "win", "end": 15, "moves": [list(m) for m in TT.DRAW24[:15]]}
    assert [p["ply"] for p in TQ.window(early, SMALL)] == list(range(10, 15)), "never before ply 10"


# ─────────────────────────────── the exclusion inventory (§14 A4.4, §15 A5.1) ─────────────────

def test_the_NEIGHBOURHOOD_is_one_and_two_ply_inside_the_eligible_window():
    assert TQ.neighbourhood(TT.DRAW24, SMALL) == set(), "a full board has no extension"
    assert TQ.neighbourhood(TT.OWN_WIN_279, SMALL) == set(), "past ply 270: never eligible"
    m = TQ.neighbourhood(TT.MULTI_269, SMALL)
    assert len(m) == len(T.replay(TT.MULTI_269).legal_moves()), "ply 270 only: no second ply"


@pytest.fixture(scope="module")
def production_inventory():
    return TQ.build_inventory(TQ._sources(), TQ.FROZEN_SPEC)


def test_the_INVENTORY_holds_every_identifiable_prior_and_test_derived_position(production_inventory):
    inv, summary = production_inventory
    assert summary["counts"] == {"a": 10, "b": 86, "c": 60, "d": 2234, "e": 880}
    assert summary["neighbourhood"]["count"] == 1_238_361 and summary["total_distinct"] == 1_241_115
    d = lambda moves: bytes.fromhex(T.state_digest(T.replay(moves)))   # noqa: E731
    s = T.replay(TT.SINGLE_260)
    bad = TT.other(s, T.classify(s, semantics="protocol")["D"])
    explicit = [TT.SINGLE_260 + [bad], TT.SINGLE_260 + [bad, (8, 10)], TT.SINGLE_260 + [(9, 10)],
                TT.OWN_WIN_AND_THREAT_264 + [(4, 7)], TT.QUIET_40 + [TT.other(T.replay(TT.QUIET_40))],
                TT.ONE[:23] + [(12, 4)]]
    u = T.replay(TT.UNDEFENDABLE_256)
    first = sorted(u.legal_moves())[0]
    win = next(t for t in CC.threats_against(u) if u.apply_move(first).is_valid_placement(*t))
    explicit += [TT.UNDEFENDABLE_256 + [first], TT.UNDEFENDABLE_256 + [first, win]]
    for moves in explicit:
        assert d(moves) in inv, moves[-2:]
    for game in json.loads(TQ.PRIOR_FILE.read_text())["standin"].values():
        assert all(bytes.fromhex(x) in inv for x in TQ._prefixes(game))
    assert all(bytes.fromhex(x) in inv for x in TQ._prefixes(TT.DRAW24))


def test_a_SMALL_inventory_lists_each_source_digest_by_digest():
    pos = TT.QUIET_40
    src = {"fixtures": {"F1": T.state_digest(T.replay(pos))},
           "h5_corpus": {"selected": [{"key": "k", "state_digest": T.state_digest(T.replay(pos[:12]))}],
                         "fixture_equivalent": []},
           "h5_record": {"rows": [{"id": "r", "moves": pos[:11]}]},
           "prior": {"pinned": {"P": pos[:3]}, "standin": {}, "bases": []}}
    inv, summary = TQ.build_inventory(src, SMALL)
    assert summary["counts"] == {"a": 1, "b": 1, "c": 1, "d": 4, "e": 0}
    assert summary["listed"][T.state_digest(T.replay(pos))] == ["a:F1"]
    assert summary["listed"][T.state_digest(T.replay(pos[:2]))] == ["d:P[:2]"]
    assert len(inv) == summary["total_distinct"] == 7


# ─────────────────────────────── selection (§14 A4.5) ───────────────────────────────

def hexd(i):
    return f"{i:064x}"


def games_spec(spec):
    """spec: {(stream, seed): [(ply, class, digest-int), ...]}."""
    return [{"stream": k[0], "seed": k[1], "positions": [{"ply": p, "class": c, "digest": hexd(x)}
                                                         for p, c, x in v]}
            for k, v in spec.items()]


def test_SELECTION_is_one_per_class_and_game_seeded_sorted_and_reproducible():
    games = games_spec({("tq24-pegs", s): [(100 + i, "QUIET", 1000 * s + i) for i in range(5)]
                        for s in range(30)})
    a = TQ.select(games, set(), SMALL)
    assert a == TQ.select(copy.deepcopy(games), set(), SMALL)
    q = [e for e in a["selected"] if e["class"] == "QUIET"]
    assert len(q) == 20 and len({e["seed"] for e in q}) == 20
    assert [(e["stream"], e["seed"]) for e in q] == sorted((e["stream"], e["seed"]) for e in q)
    import random
    keys = sorted(("tq24-pegs", s) for s in range(30))
    assert sorted(e["seed"] for e in q) == sorted(k[1] for k in random.Random("tq24-select:QUIET").sample(keys, 20))
    for e in q:
        assert e["ply"] == random.Random(f"tq24-select:QUIET:tq24-pegs:{e['seed']}").choice(list(range(100, 105)))


def test_SELECTION_records_SHORTAGES_repeats_and_PRIOR_exclusions():
    games = games_spec({("tq24-pegs", 1): [(150, "OWN_WIN", 7), (149, "QUIET", 8)],
                        ("tq24-pegs", 2): [(150, "OWN_WIN", 7), (140, "DEFEND_MULTI", 9)],
                        ("tq24-uniform", 5): [(120, "QUIET", 10)]})
    r = TQ.select(games, {bytes.fromhex(hexd(10))}, SMALL)
    assert r["excluded_prior"] == {"OWN_WIN": 0, "DEFEND_SINGLE": 0, "DEFEND_MULTI": 0,
                                   "UNDEFENDABLE": 0, "QUIET": 1}
    assert r["quota_shortfall"] == {"OWN_WIN": 18, "DEFEND_SINGLE": 20, "DEFEND_MULTI": 19,
                                    "UNDEFENDABLE": 20, "QUIET": 19}
    assert r["universe"]["OWN_WIN"] == {"positions": 2, "games": 2}
    first = [e for e in r["selected"] if e["state_digest"] == hexd(7)]
    assert [e["repeat_of"] for e in first] == [None, "tq24-pegs:1@150"]
    assert r["repeats"] == {hexd(7): ["tq24-pegs:1@150", "tq24-pegs:2@150"]}
    assert all(e["state_digest"] != hexd(10) for e in r["selected"])


# ─────────────────────────────── the child's run ───────────────────────────────

@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """ONE small freeze with the REAL exhaustive cross-check."""
    d = claimed(tmp_path_factory.mktemp("small"))
    return d, TQ._run(SMALL, EMPTY, d)


def test_RUN_COMPLETE_writes_a_PROVISIONAL_payload_and_NEVER_corpus_json(small_run):
    d, rec = small_run
    assert (rec["status"], rec["games_processed"], rec["games_planned"]) == ("COMPLETE", 16, 16)
    assert not (d / TQ.CORPUS).exists(), "only the parent publishes"
    raw = (d / TQ.PROVISIONAL).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == rec["provisional_sha256"]
    c = json.loads(raw)
    assert (c["spec"], c["games_generated"], c["games_won"]) == (SMALL, 16, 1)
    assert c["selected"] and all(e["stream"] == "tq24-pegs" and e["seed"] == 3 for e in c["selected"])
    assert all(e["cross_check"]["agree"] for e in c["selected"])
    for e in c["selected"]:                       # no reply is examined after a winning move
        assert (e["cross_check"]["replies_examined"] == 0) == (e["class"] == "OWN_WIN"), e["class"]
    assert all(v >= 19 for v in c["quota_shortfall"].values()), "shortages recorded, not filled"
    steps = [json.loads(x)["step"] for x in (d / "log.jsonl").read_text().splitlines()]
    assert steps == ["inventory"] + ["game"] * 16 + ["cross_check", "provisional"]


def test_RUN_every_completed_game_is_DURABLE_in_order(small_run):
    d, _ = small_run
    rows = [json.loads(x) for x in (d / TQ.CANDIDATES).read_text().splitlines()]
    assert [(r["stream"], r["seed"]) for r in rows] == \
        [(s, k) for s in SMALL["streams"] for k in range(0, 8)]
    assert [r for r in rows if r["positions"]] == [rows[11]] and rows[11]["end"] == 193


def test_the_whole_RUN_is_REPRODUCIBLE_byte_for_byte(tmp_path, monkeypatch):
    monkeypatch.setattr(TC, "cross_check", lambda s: {**fast_check(s), "replies_examined": 1})
    a, b = claimed(tmp_path / "a"), claimed(tmp_path / "b")
    TQ._run(SMALL, EMPTY, a)
    TQ._run(SMALL, EMPTY, b)
    for f in (TQ.PROVISIONAL, TQ.INVENTORY, TQ.CANDIDATES):
        assert (a / f).read_bytes() == (b / f).read_bytes(), f


def test_RUN_a_PRIOR_position_is_EXCLUDED_and_counted(small_run, tmp_path, monkeypatch):
    monkeypatch.setattr(TC, "cross_check", fast_check)
    d0, _ = small_run
    rows = [json.loads(x) for x in (d0 / TQ.CANDIDATES).read_text().splitlines()]
    p = rows[11]["positions"][0]
    src = copy.deepcopy(EMPTY)
    src["h5_corpus"]["selected"] = [{"key": "prior", "state_digest": p["digest"]}]
    d = claimed(tmp_path)
    TQ._run(SMALL, src, d)
    c = json.loads((d / TQ.PROVISIONAL).read_text())
    assert c["excluded_prior"][p["class"]] == 1
    assert p["digest"] not in {e["state_digest"] for e in c["selected"]}


def test_RUN_a_CHECKER_DISAGREEMENT_STOPS_with_no_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(TC, "cross_check", lambda s: {**fast_check(s), "agree": False})
    d = claimed(tmp_path)
    rec = TQ._run(SMALL, EMPTY, d)
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == ("STOP", "cross_check", "CorpusStop")
    assert "DISAGREE" in f["reason"] and not (d / TQ.PROVISIONAL).exists()


def test_RUN_an_INTERRUPTION_keeps_every_completed_game_and_no_payload(tmp_path, monkeypatch):
    real = TQ.play

    def play(stream, seed, spec):
        if (stream, seed) == ("tq24-pegs", 3):
            raise KeyboardInterrupt
        return real(stream, seed, spec)
    monkeypatch.setattr(TQ, "play", play)
    d = claimed(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        TQ._run(SMALL, EMPTY, d)
    rec = json.loads((d / "record.json").read_text())
    assert (rec["status"], rec["games_processed"], rec["failure"]["failing_step"]) == \
        ("INTERRUPTED", 11, "generate:tq24-pegs:3")
    assert len((d / TQ.CANDIDATES).read_text().splitlines()) == 11 and not (d / TQ.PROVISIONAL).exists()


def test_RUN_a_NON_REPRODUCIBLE_generator_STOPS(tmp_path, monkeypatch):
    monkeypatch.setattr(TC, "cross_check", fast_check)
    real, calls = TQ.play, collections_counter()

    def play(stream, seed, spec):
        calls[(stream, seed)] += 1
        if (stream, seed) == ("tq24-pegs", 3) and calls[(stream, seed)] > 1:
            return {**real(stream, seed, spec), "moves": real("tq24-uniform", 3, spec)["moves"]}
        return real(stream, seed, spec)        # the REPLAY above is a legal game -- but another one
    monkeypatch.setattr(TQ, "play", play)
    rec = TQ._run(SMALL, EMPTY, claimed(tmp_path))
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "replay")
    assert "not reproducible" in rec["failure"]["reason"]


def collections_counter():
    import collections
    return collections.Counter()


def test_RUN_the_DEADLINE_and_the_INTERPRETER_are_enforced(tmp_path, monkeypatch):
    t = [0.0]

    def clock():
        t[0] += 1000.0
        return t[0]
    rec = TQ._run(SMALL, EMPTY, claimed(tmp_path / "t"), clock=clock, wall_s=5000.0)
    assert (rec["status"], rec["failure"]["exception_type"]) == ("STOP", "CorpusStop")
    assert "deadline passed" in rec["failure"]["reason"]
    d = claimed(tmp_path / "p")
    with pytest.raises(TQ.CorpusRefused, match="the interpreter is"):
        TQ._run({**SMALL, "python": "2.7.18"}, EMPTY, d)
    assert sorted(os.listdir(d)) == [TQ.CLAIM_FILE]


# ─────────────────────────────── the parent: validation and publication (§15 A5.2) ────────────

WATCH = {"terminated": False, "cause": None, "limit_s": 10_800.0, "elapsed_s": 100.0, "child_pid": 1,
         "rc": 0, "signal": None, "survivors_killed": False}


@pytest.fixture
def copy_of(small_run, tmp_path):
    d0, _ = small_run
    d = tmp_path / TQ.OUT_NAME
    shutil.copytree(d0, d)
    return d


def test_the_PARENT_publishes_the_provisional_BYTES_after_a_COMPLETE_exit(copy_of):
    w = TQ._settle(copy_of, WATCH, spec=SMALL)
    raw = (copy_of / TQ.CORPUS).read_bytes()
    assert raw == (copy_of / TQ.PROVISIONAL).read_bytes() and w["status"] == "COMPLETE"
    assert w["published"] == {"file": "corpus.json", "corpus_sha256": hashlib.sha256(raw).hexdigest()}
    assert json.loads((copy_of / TQ.WATCHDOG_FILE).read_text()) == w


def _rewrite(d, f):
    c = json.loads((d / TQ.PROVISIONAL).read_text())
    f(c)
    os.chmod(d / TQ.PROVISIONAL, 0o644)
    (d / TQ.PROVISIONAL).write_bytes(TQ._canonical(c))
    rec = json.loads((d / "record.json").read_text())
    rec["provisional_sha256"] = hashlib.sha256((d / TQ.PROVISIONAL).read_bytes()).hexdigest()
    (d / "record.json").write_text(json.dumps(rec))


REFUSALS = {
    "the payload altered after the record": (lambda d: (d / TQ.PROVISIONAL).write_text("{}"), WATCH,
                                             "not the one the record hashed"),
    "the record is STOP": (lambda d: (d / "record.json").write_text(json.dumps({"status": "STOP"})), WATCH,
                           "the child's run is STOP"),
    "a nonzero exit": (lambda d: None, {**WATCH, "rc": 1}, "the child's run is EXITED_ABNORMALLY"),
    "terminated": (lambda d: None, {**WATCH, "terminated": True, "rc": -9}, "the child's run is TERMINATED"),
    "another specification": (lambda d: _rewrite(d, lambda c: c.update(spec={**SMALL, "quota": 5})), WATCH,
                              "frozen specification"),
    "inventory.json altered": (lambda d: (d / TQ.INVENTORY).write_text("{}"), WATCH,
                               "inventory.json is not the inventory"),
    "a position's moves altered": (lambda d: _rewrite(d, lambda c: c["selected"][0]["moves"].pop()), WATCH,
                                   "does not replay to its digest"),
    "a class altered": (lambda d: _rewrite(d, lambda c: c["selected"][0]["class_dict"].update(T_void=[[1, 1]])),
                        WATCH, "not of its recorded class"),
    "a cross-check not agreeing": (lambda d: _rewrite(d, lambda c: c["selected"][0]["cross_check"].update(agree=False)),
                                   WATCH, "no agreeing cross-check"),
    "a quota exceeded": (lambda d: _rewrite(d, lambda c: c["selected"].extend([c["selected"][0]] * 20)), WATCH,
                         "quotas are exceeded"),
}


@pytest.mark.parametrize("name", REFUSALS)
def test_the_PARENT_publishes_NOTHING_unless_everything_holds(copy_of, name):
    breakage, watch, why = REFUSALS[name]
    breakage(copy_of)
    w = TQ._settle(copy_of, watch, spec=SMALL)
    assert w["published"] is None and why in w["not_published"], w["not_published"]
    assert w["status"] != "COMPLETE" and not (copy_of / TQ.CORPUS).exists()


COMPLETED_THEN_HUNG = r'''
import json, os, signal, sys, time, shutil
src, d = sys.argv[1], sys.argv[2]
for f in ("inventory.json", "candidates.jsonl", "log.jsonl", "corpus.provisional.json", "record.json"):
    shutil.copy(os.path.join(src, f), os.path.join(d, f))
signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())
time.sleep(60)
'''


@pytest.fixture
def open_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(AUTH, "TACTICAL_CANDIDATE_QUALIFICATION_AUTHORIZED", True)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setattr(TQ, "_sources", lambda: EMPTY)
    for k, v in SMALL.items():
        monkeypatch.setitem(TQ.FROZEN_SPEC, k, v)
    return tmp_path


def test_a_COMPLETED_child_payload_followed_by_TERMINATION_is_NEVER_published(open_gate, small_run, monkeypatch):
    d0, _ = small_run
    monkeypatch.setattr(TQ, "_child_argv", lambda: [sys.executable, "-c", COMPLETED_THEN_HUNG, str(d0),
                                                    str(TQ.out_dir())])
    monkeypatch.setattr(TQ, "HARD_LIMIT_S", 3.0)
    t0 = time.monotonic()
    w = TQ.freeze()
    out = TQ.out_dir()
    assert time.monotonic() - t0 < 20
    assert (w["status"], w["record_status"], w["published"]) == ("TERMINATED", "COMPLETE", None)
    assert (out / TQ.PROVISIONAL).exists() and not (out / TQ.CORPUS).exists()
    assert w["provisional_sha256"] == json.loads((out / "record.json").read_text())["provisional_sha256"]


def test_PUBLIC_ENTRY_runs_in_the_real_state_and_the_PARENT_publishes(open_gate, monkeypatch):
    monkeypatch.setattr(TC, "cross_check", fast_check)

    def watch(argv, *, limit_s, env, cwd, stdout_path, stderr_path):
        assert list(argv[1:]) == ["-m", TQ.MODULE, "freeze-claimed"] and env["PYTHONUNBUFFERED"] == "1"
        assert limit_s <= TQ.HARD_LIMIT_S
        monkeypatch.setenv(TQ.CLAIM_ENV, env[TQ.CLAIM_ENV])
        with open(stdout_path, "x") as so, open(stderr_path, "x") as se, \
                contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            rc = TQ.main(["freeze-claimed"])
        return {**WATCH, "limit_s": limit_s, "rc": rc}
    monkeypatch.setattr(CC, "_watch", watch)
    w = TQ.freeze()
    out = TQ.out_dir()
    assert (w["status"], w["record_status"]) == ("COMPLETE", "COMPLETE")
    assert (out / TQ.CORPUS).read_bytes() == (out / TQ.PROVISIONAL).read_bytes()
    monkeypatch.setattr(F, "check_durable", lambda paths, _fixture: None)
    corpus, sha = TQ.committed_corpus()
    assert sha == w["published"]["corpus_sha256"] and corpus["spec"] == SMALL


@pytest.mark.parametrize("breakage,why", [
    (lambda d: (d / TQ.WATCHDOG_FILE).write_text(json.dumps(
        {**json.loads((d / TQ.WATCHDOG_FILE).read_text()), "status": "STOP"})), "did not publish THIS"),
    (lambda d: (d / TQ.CORPUS).write_text("{}"), "did not publish THIS"),
    (lambda d: (d / "record.json").write_text(json.dumps({"status": "COMPLETE", "provisional_sha256": "x"})),
     "does not hash to the published corpus"),
], ids=["watchdog not COMPLETE", "corpus altered", "record hash mismatch"])
def test_QUALIFICATION_requires_the_parents_COMPLETE_publication(copy_of, monkeypatch, breakage, why):
    TQ._settle(copy_of, WATCH, spec=SMALL)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(copy_of.parent))
    monkeypatch.setattr(F, "check_durable", lambda paths, _fixture: None)
    assert TQ.committed_corpus()[1] == json.loads((copy_of / TQ.WATCHDOG_FILE).read_text())["published"]["corpus_sha256"]
    breakage(copy_of)
    with pytest.raises(TQ.CorpusRefused, match=why):
        TQ.committed_corpus()


def test_QUALIFICATION_requires_the_corpus_COMMITTED(copy_of, monkeypatch):
    TQ._settle(copy_of, WATCH, spec=SMALL)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(copy_of.parent))
    with pytest.raises(TQ.CorpusRefused, match="not committed"):
        TQ.committed_corpus()


# ─────────────────────────────── gates and the CLI ───────────────────────────────

@pytest.mark.parametrize("entry", ["freeze", "freeze_claimed"])
def test_every_public_entry_refuses_with_the_gate_shut_BEFORE_anything(entry, monkeypatch):
    for owner, name in ((TQ, "_sources"), (TQ, "_claim"), (TQ, "_run"), (CC, "_watch"), (os, "mkdir")):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail("touched before the gate"))
    with pytest.raises(TQ.CorpusUnauthorized):
        getattr(TQ, entry)()


def test_the_CLI_exits_5_and_takes_no_output_directory(tmp_path):
    r = subprocess.run([sys.executable, "-m", TQ.MODULE, "freeze"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == TQ.EXIT_UNAUTHORIZED == 5, r.stderr
    r = subprocess.run([sys.executable, "-m", TQ.MODULE, "freeze", "--out-dir", str(tmp_path)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 2 and "--out-dir" in r.stderr


def test_PUBLIC_ENTRY_refuses_another_INTERPRETER_before_any_claim(open_gate, monkeypatch):
    monkeypatch.setattr(TQ.platform, "python_version", lambda: "3.13.0")
    monkeypatch.setattr(CC, "_watch", lambda *a, **k: pytest.fail("a child was started"))
    with pytest.raises(TQ.CorpusRefused, match="the interpreter is 3.13.0"):
        TQ.freeze()
    assert not TQ.out_dir().exists()


def test_the_CHILD_takes_only_an_UNCONSUMED_claim_for_THE_frozen_spec(open_gate, monkeypatch):
    d = TQ.out_dir()
    _c, nonce = TQ._claim(d, {"spec_sha256": "x"})
    monkeypatch.setenv(TQ.CLAIM_ENV, "0" * 32)
    with pytest.raises(TQ.CorpusRefused, match="nonce does not match"):
        TQ.freeze_claimed()
    monkeypatch.setenv(TQ.CLAIM_ENV, nonce)
    with pytest.raises(TQ.CorpusRefused, match="another specification"):
        TQ.freeze_claimed()
