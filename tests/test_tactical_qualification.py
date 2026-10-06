"""Tactical candidate qualification run (card §17 A7).

SYNTHETIC ONLY: no model is loaded and the real incumbent makes no decision. The search is the
REAL frozen MCTS path built by R1's production builder, driven by the steered stub evaluator. The
bindings are the REAL ones (the committed corpus, the pinned code, the prior file), read-only.
The stub cannot reproduce the real fingerprints, so reproduction runs against STUB pins computed
here -- and is shown to STOP on the real pins. Runs use a small call list through `plan_of`.
"""
import contextlib
import dataclasses
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

from scripts.GPU.alphazero import candidate_match_runner as CM
from scripts.GPU.alphazero import h4_runner as R4
from scripts.GPU.alphazero import h5_cap_completion as CC
from scripts.GPU.alphazero import h5_incumbent_defence as D
from scripts.GPU.alphazero import h5_incumbent_defence_factors as X
from scripts.GPU.alphazero import h5_incumbent_tactics as T
from scripts.GPU.alphazero import tactical_candidate as TC
from scripts.GPU.alphazero import tactical_candidate_authorization as AUTH
from scripts.GPU.alphazero import tactical_corpus as TQ
from scripts.GPU.alphazero import tactical_qualification as Q
from scripts.GPU.alphazero import twixtbot_g3_reference as G3
from tests.test_h5_incumbent_defence import CONFIG, FIX, IDENT, Steered, build

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS_DIR = TQ.out_dir()                         # the REAL published corpus, read-only
REAL_PINS = dict(X.PINNED_FINGERPRINTS)
#: one call of each kind and class: an override (020) and every boundary
SMALL = ("repro:E1", "corpus:000", "corpus:020", "corpus:060", "boundary:ply-280", "boundary:board-8")
WATCH = {"terminated": False, "cause": None, "limit_s": 3600.0, "elapsed_s": 100.0, "child_pid": 1,
         "rc": 0, "signal": None, "survivors_killed": False}


@pytest.fixture(scope="module")
def bound():
    return Q.bindings()


@pytest.fixture(scope="module")
def stub_pins():
    out = {}
    for p in FIX:
        r = D.decide(build(), CC.state_of(p["moves"]), capture=False)
        out[p["id"]] = D.fingerprint(r["root"], r["counts"], r["root_value"])
    return out


@pytest.fixture
def pinned(monkeypatch, stub_pins):
    monkeypatch.setattr(X, "PINNED_FINGERPRINTS", dict(stub_pins))


def make():
    return CM.production_builder(CONFIG, Steered(), IDENT)


def call(bound, cid):
    return next(c for c in Q.plan_of(bound["corpus"]) if c["id"] == cid)


def small_plan(corpus, ids=SMALL):
    return [c for c in REAL_PLAN_OF(corpus) if c["id"] in ids]


REAL_PLAN_OF = Q.plan_of


def claimed(tmp_path):
    d = tmp_path / Q.OUT_NAME
    Q._claim(d, {})
    return d


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


# ─────────────────────────────── bindings and the call list (§17 A7.1, A7.2) ──────────────────

def test_the_BINDINGS_are_the_committed_corpus_its_code_and_the_pinned_files(bound):
    assert bound["corpus_sha256"] == "1f040a9208fb3049de4fbf6e73884ad0fafa067fed3d9c275a4abbedf149179c"
    assert sorted(bound["corpus"]["code"]) == ["h5_incumbent_tactics.py", "tactical_candidate.py",
                                               "tactical_corpus.py"]
    assert sorted(Q.CODE_PINS) == ["candidate_match_runner.py", "h5_incumbent_defence.py",
                                   "h5_incumbent_defence_factors.py"]
    for name, sha in Q.CODE_PINS.items():
        assert CM._file_sha(str(pathlib.Path(TQ.__file__).with_name(name))) == sha, name
    assert bound["prior"] == json.loads(TQ.PRIOR_FILE.read_text())


def _code(monkeypatch, bound):
    c = json.loads(json.dumps(bound["corpus"]))
    c["code"]["tactical_candidate.py"] = "0" * 64
    monkeypatch.setattr(TQ, "committed_corpus", lambda: (c, bound["corpus_sha256"]))


def _refuse(monkeypatch, bound):
    def no():
        raise TQ.CorpusRefused("probe")
    monkeypatch.setattr(TQ, "committed_corpus", no)


@pytest.mark.parametrize("breakage,why", [
    (_refuse, "the corpus is not accepted: probe"),
    (_code, "tactical_candidate.py is not the code the corpus recorded"),
    (lambda mp, b: mp.setitem(Q.CODE_PINS, "candidate_match_runner.py", "0" * 64),
     "candidate_match_runner.py is not the pinned code"),
    (lambda mp, b: mp.setitem(Q.CODE_PINS, "h5_incumbent_defence_factors.py", "0" * 64),
     "h5_incumbent_defence_factors.py is not the pinned code"),
    (lambda mp, b: mp.setattr(TQ, "PRIOR_SHA256", "0" * 64), "prior-position file is not the pinned one"),
], ids=["corpus refused", "wrapper changed", "runner changed", "fixture reader changed", "prior file"])
def test_the_BINDINGS_refuse_anything_else(monkeypatch, bound, breakage, why):
    breakage(monkeypatch, bound)
    with pytest.raises(Q.QualRefused, match=why):
        Q.bindings()


def test_the_CALL_LIST_is_the_cards_109_calls_in_order(bound):
    plan = Q.plan_of(bound["corpus"])
    assert len(plan) == 109 and len(bound["corpus"]["selected"]) == 100
    assert [c["id"] for c in plan[:5]] == [f"repro:{f}" for f in D.POSITION_IDS] == \
        [f"repro:{f}" for f in Q.FIXTURES]
    assert [c["index"] for c in plan[5:105]] == list(range(100))
    assert [c["id"] for c in plan[105:]] == ["boundary:own-win-279", "boundary:threat-279",
                                             "boundary:ply-280", "boundary:board-8"]
    assert len({c["id"] for c in plan}) == 109


def test_the_RUN_CONSTANTS_are_the_cards():
    assert (Q.HARD_LIMIT_S, Q.MARGIN_S, Q.CALL_S, Q.SEED) == (3600.0, 60.0, 120.0, -1) == \
        (3600.0, 60.0, 120.0, D.SEED)
    assert Q.out_dir() == pathlib.Path(R4.EVIDENCE_ROOT) / "tactical-candidate-qualification"
    assert Q.AUTH is AUTH and Q.CM is CM and Q.TC is TC


# ─────────────────────────────── one call, on the real search ───────────────────────────────

def test_REPRODUCTION_passes_on_the_pins_it_was_given(pinned, bound):
    row = Q.observe(call(bound, "repro:E1"), make(), CONFIG, bound)
    assert row["move"] == list(FIX[0]["recorded"]) and row["fingerprint"] == X.PINNED_FINGERPRINTS["E1"]
    assert row["checks"] == ["configuration", "the recorded move", "the pinned fingerprint"]


def test_REPRODUCTION_STOPS_on_the_REAL_pins_under_the_stub(bound):
    assert X.PINNED_FINGERPRINTS == REAL_PINS
    with pytest.raises(Q.QualStop, match=r"repro:E1: check 'the pinned fingerprint' failed"):
        Q.observe(call(bound, "repro:E1"), make(), CONFIG, bound)


def test_REPRODUCTION_STOPS_on_another_recorded_move(pinned, bound, monkeypatch):
    real = D.positions
    monkeypatch.setattr(D, "positions", lambda f: [{**p, "recorded": (0, 1)} for p in real(f)])
    with pytest.raises(Q.QualStop, match="check 'the recorded move' failed"):
        Q.observe(call(bound, "repro:E2"), make(), CONFIG, bound)


EQUIVALENCE = ["configuration (incumbent)", "configuration (candidate)", "counts", "tie set",
               "tree fingerprint", "root value", "search stream", "readout stream", "one witness",
               "the baseline is the incumbent's move"]


@pytest.mark.parametrize("index", [0, 20, 40, 60, 80])
def test_a_CORPUS_call_checks_equivalence_and_the_M1_readout(bound, index):
    e = bound["corpus"]["selected"][index]
    row = Q.observe(call(bound, f"corpus:{index:03d}"), make(), CONFIG, bound)
    assert row["checks"][:2] == ["the corpus position", *EQUIVALENCE][:2]
    assert row["checks"][1:11] == EQUIVALENCE
    assert "the witness class is the recorded class" in row["checks"]
    assert row["checks"][-2:] == ["the candidate's move follows M1", "the witness records the move"]
    assert row["cls"] == e["class"] and row["stream"] == e["stream"] and row["ply"] == e["ply"]
    if e["class"] in TC.INACTIVE:
        assert row["rule"] == "inactive" and row["final"] == row["baseline"]
    else:
        assert "an active rule ends inside the admissible set" in row["checks"]
        c = e["class_dict"]
        assert row["final"] in (c["W"] if c["class"] == "OWN_WIN" else c["D"])


def test_the_stub_makes_the_readout_OVERRIDE_somewhere(bound):
    row = Q.observe(call(bound, "corpus:020"), make(), CONFIG, bound)
    assert (row["cls"], row["rule"], row["override"]) == ("DEFEND_SINGLE", "override", True)
    assert row["final"] != row["baseline"]


def test_the_readout_takes_the_MOST_VISITED_admissible_move_not_the_first(bound):
    """At 020 the override lands on D[0] anyway; at 021 the most-visited saving move is another one."""
    row = Q.observe(call(bound, "corpus:021"), make(), CONFIG, bound)
    D_ = bound["corpus"]["selected"][21]["class_dict"]["D"]
    assert (row["rule"], row["unsearched"]) == ("override", False)
    assert row["final"] in D_ and row["final"] != D_[0]


def test_the_BOUNDARIES_a_winning_280th_move_a_void_threat_and_two_refusals(bound):
    m = make()
    win = Q.observe(call(bound, "boundary:own-win-279"), m, CONFIG, bound)
    s = T.replay(bound["prior"]["pinned"]["OWN_WIN_279"])
    assert s.ply == 279 and s.apply_move(tuple(win["final"])).winner() == s.to_move
    assert win["checks"][-1] == "the 280th move wins"
    void = Q.observe(call(bound, "boundary:threat-279"), m, CONFIG, bound)
    assert void["rule"] == "inactive" and void["checks"][-1] == "the void threat leaves the rule inactive"
    for b in ("ply-280", "board-8"):
        row = Q.observe(call(bound, f"boundary:{b}"), m, CONFIG, bound)
        assert [x for x in row["checks"] if x.startswith(("refused", "no search"))] == [
            "refused (enabled)", "no search (enabled)", "refused (disabled)", "no search (disabled)"]


# ─────────────────────────────── negative controls: every check can fail ───────────────────────

def _after_search(agent, draw):
    real = agent.mcts.search_with_root

    def search_with_root(*a, **k):
        out = real(*a, **k)
        draw(out)
        return out
    agent.mcts.search_with_root = search_with_root


def drifted(change):
    """R1's builder, with `change` applied to the CANDIDATE's agent only."""
    real = make()

    def m(colour, seed, role):
        a = real(colour, seed, role)
        if role == "candidate":
            change(a.agent)
        return a
    return m


@pytest.mark.parametrize("change,why", [
    (lambda a: setattr(a.mcts, "config", dataclasses.replace(a.mcts.config, fpu_value=-0.5)),
     "configuration \\(candidate\\)"),
    (lambda a: a.mcts.rng.random(), "counts"),
    (lambda a: _after_search(a, lambda out: a.mcts.rng.random()), "search stream"),
    (lambda a: _after_search(a, lambda out: setattr(out[2], "nn_value", float(out[2].nn_value or 0) + 1e-3)),
     "tree fingerprint"),
    (lambda a: a.readout_rng.random(), "readout stream"),
], ids=["configuration", "search stream (before: the tree moves)", "search stream (after the search)",
        "tree (counts equal)", "readout stream"])
def test_a_DRIFTED_candidate_STOPS(bound, change, why):
    with pytest.raises(Q.QualStop, match=f"corpus:020: check '{why}' failed"):
        Q.observe(call(bound, "corpus:020"), drifted(change), CONFIG, bound)


def test_a_candidate_with_ANOTHER_SEED_STOPS(bound):
    real = make()
    with pytest.raises(Q.QualStop, match="the candidate was built with seed -3"):
        Q.observe(call(bound, "corpus:020"),
                  lambda c, s, r: real(c, -3 if r == "candidate" else s, r), CONFIG, bound)


def _witness(edit):
    real = TC.witness

    def w(*a, **k):
        out = real(*a, **k)
        edit(out)
        return out
    return w


def _keep_baseline(w):
    w.update(final=w["baseline"], override=False, rule="preserved")


def _another_admissible(w):                       # inside A, but not the most-visited admissible move
    other = next(m for m in w["class"]["D"] if m != list(w["final"]["move"]))
    w.update(final={**w["final"], "move": other})


@pytest.mark.parametrize("edit,why", [
    (_keep_baseline, "an active rule ends inside the admissible set"),
    (_another_admissible, "the candidate's move follows M1"),
    (lambda w: w.update(**{"class": {**w["class"], "class": "QUIET"}}), "the witness class is the recorded class"),
    (lambda w: w.update(override=False), "the witness records the move"),
    (lambda w: w.update(baseline={**w["baseline"], "move": [0, 1]}), "the baseline is the incumbent's move"),
], ids=["baseline kept", "another admissible move", "wrong class", "witness misrecords", "wrong baseline"])
def test_a_READOUT_that_deviates_STOPS(bound, monkeypatch, edit, why):
    monkeypatch.setattr(TC, "witness", _witness(edit))
    with pytest.raises(Q.QualStop, match=f"check '{why}' failed"):
        Q.observe(call(bound, "corpus:020"), make(), CONFIG, bound)


@pytest.mark.parametrize("edit,why", [
    (lambda w: w.update(**{"class": {**w["class"], "T_void": None}}), "the witness class"),
    (lambda w: w.update(rule="preserved"), "the void threat leaves the rule inactive"),
], ids=["wrong class", "rule not inactive"])
def test_a_WITNESS_at_the_void_threat_that_deviates_STOPS(bound, monkeypatch, edit, why):
    monkeypatch.setattr(TC, "witness", _witness(edit))
    with pytest.raises(Q.QualStop, match=f"boundary:threat-279: check '{why}' failed"):
        Q.observe(call(bound, "boundary:threat-279"), make(), CONFIG, bound)


def test_a_candidate_WITHOUT_a_witness_STOPS(bound):
    real = make()

    def m(colour, seed, role):
        a = real(colour, seed, role)
        return TC.TacticalCandidate(a.agent, enabled=False) if role == "candidate" else a
    with pytest.raises(Q.QualStop, match="corpus:000: check 'one witness' failed"):
        Q.observe(call(bound, "corpus:000"), m, CONFIG, bound)


def test_a_decision_that_searches_TWICE_STOPS(pinned, bound, monkeypatch):
    real = G3.SeededReferenceAgent.__call__

    def twice(self, state):
        self.mcts.search_with_root(state, add_noise=False)
        return real(self, state)
    monkeypatch.setattr(G3.SeededReferenceAgent, "__call__", twice)
    with pytest.raises(Q.QualStop, match="2 searches for one incumbent decision, expected 1"):
        Q.observe(call(bound, "repro:E1"), make(), CONFIG, bound)


def _fpu(a):
    a.mcts.config = dataclasses.replace(a.mcts.config, fpu_value=-0.5)


def test_a_DRIFTED_incumbent_STOPS(pinned, bound):
    real = make()

    def m(colour, seed, role):
        a = real(colour, seed, role)
        if role == "incumbent":
            _fpu(a)
        return a
    with pytest.raises(Q.QualStop, match="repro:E1: check 'configuration' failed"):
        Q.observe(call(bound, "repro:E1"), m, CONFIG, bound)
    with pytest.raises(Q.QualStop, match="corpus:000: check 'configuration \\(incumbent\\)' failed"):
        Q.observe(call(bound, "corpus:000"), m, CONFIG, bound)


def test_a_NON_WINNING_280th_move_STOPS_even_when_M1_agrees(bound, monkeypatch):
    """A detector that called a non-winning move a win at ply 279: the wrapper and the readout
    check both follow it; only the boundary's own check sees the move does not win."""
    s = T.replay(bound["prior"]["pinned"]["OWN_WIN_279"])
    nonwin = next(m for m in sorted(s.legal_moves()) if s.apply_move(m).winner() is None)
    real = CC.immediate_wins
    monkeypatch.setattr(CC, "immediate_wins", lambda st: [nonwin] if st.ply == 279 else real(st))
    with pytest.raises(Q.QualStop, match="boundary:own-win-279: check 'the 280th move wins' failed"):
        Q.observe(call(bound, "boundary:own-win-279"), make(), CONFIG, bound)


def _searched(self, state):
    self.agent.mcts.search_with_root(state, add_noise=False)


def _searched_traceless(self, state):             # a search that leaves both streams as they were
    st = (self.agent.mcts.rng.getstate(), self.agent.readout_rng.getstate())
    self.agent.mcts.search_with_root(state, add_noise=False)
    self.agent.mcts.rng.setstate(st[0])
    self.agent.readout_rng.setstate(st[1])


def _drew(self, state):                           # no search, but a stream drawn
    self.agent.readout_rng.random()


@pytest.mark.parametrize("before", [_searched, _searched_traceless, _drew],
                         ids=["searched", "searched, streams restored", "a stream drawn"])
@pytest.mark.parametrize("b", ["ply-280", "board-8"])
def test_a_refusal_AFTER_any_search_or_draw_STOPS(bound, monkeypatch, b, before):
    real = TC.TacticalCandidate.__call__

    def late(self, state):                        # does something, then refuses as before
        before(self, state)
        return real(self, state)
    monkeypatch.setattr(TC.TacticalCandidate, "__call__", late)
    with pytest.raises(Q.QualStop, match=f"boundary:{b}: check 'no search \\(enabled\\)' failed"):
        Q.observe(call(bound, f"boundary:{b}"), make(), CONFIG, bound)


def test_a_DISABLED_candidate_that_skips_the_refusal_STOPS(bound, monkeypatch):
    real = TC.TacticalCandidate.__call__
    monkeypatch.setattr(TC.TacticalCandidate, "__call__",
                        lambda self, state: real(self, state) if self.enabled else self.agent(state))
    for b in ("ply-280", "board-8"):
        with pytest.raises(Q.QualStop, match=f"boundary:{b}: check 'refused \\(disabled\\)' failed"):
            Q.observe(call(bound, f"boundary:{b}"), make(), CONFIG, bound)


def test_ANOTHER_exception_is_not_a_refusal(bound, monkeypatch):
    def broken(self, state):
        raise RuntimeError("not a refusal")
    monkeypatch.setattr(TC.TacticalCandidate, "__call__", broken)
    for b in ("ply-280", "board-8"):
        with pytest.raises(RuntimeError, match="not a refusal"):
            Q.observe(call(bound, f"boundary:{b}"), make(), CONFIG, bound)


def test_a_SKIPPED_cap_refusal_STOPS(bound, monkeypatch):
    monkeypatch.setattr(TC, "PLY_CAP", 281)
    with pytest.raises(Q.QualStop, match="boundary:ply-280: check 'refused \\(enabled\\)' failed"):
        Q.observe(call(bound, "boundary:ply-280"), make(), CONFIG, bound)


def test_an_ALTERED_corpus_position_STOPS(bound):
    b = json.loads(json.dumps({k: bound[k] for k in ("corpus", "corpus_sha256", "prior")}))
    b["corpus"]["selected"][3]["moves"] = b["corpus"]["selected"][3]["moves"][:-1]
    with pytest.raises(Q.QualStop, match="corpus:003: check 'the corpus position' failed"):
        Q.observe({"id": "corpus:003", "kind": "corpus", "index": 3}, make(), CONFIG, b)


# ─────────────────────────────── the child's run ───────────────────────────────

def run_small(tmp_path, bound, ids=SMALL, loader=Steered, **kw):
    d = claimed(tmp_path)
    return d, Q._run(small_plan(bound["corpus"], ids), CONFIG, IDENT, loader, d, bound, **kw)


def test_RUN_COMPLETE_is_QUALIFIED_with_every_call_durable_in_order(pinned, bound, tmp_path):
    d, rec = run_small(tmp_path, bound)
    assert (rec["status"], rec["verdict"], rec["failure"]) == ("COMPLETE", "QUALIFIED", None)
    assert (rec["calls"], rec["observed"], rec["seed"]) == (len(SMALL), len(SMALL), -1)
    assert rec["corpus_sha256"] == bound["corpus_sha256"] and rec["incumbent_identity"] == IDENT
    assert rec["plan_sha256"] == Q._sha(small_plan(bound["corpus"]))
    assert [r["id"] for r in lines(d / Q.OBSERVATIONS)] == list(SMALL)
    assert rec["observations_sha256"] == CM._file_sha(str(d / Q.OBSERVATIONS))
    log = lines(d / "log.jsonl")
    assert [x["step"] for x in log] == ["model_load"] + ["call_start", "call"] * len(SMALL)
    assert json.loads((d / "record.json").read_text()) == rec
    pc = rec["summary"]["per_class"]
    assert sum(v["positions"] for v in pc.values()) == 3 and pc["DEFEND_SINGLE"]["overrides"] == 1
    assert pc["UNDEFENDABLE"]["rules"] == {"inactive": 1} and pc["QUIET"]["positions"] == 0


def test_RUN_a_failed_check_STOPS_NOT_QUALIFIED_and_keeps_earlier_calls(pinned, bound, tmp_path, monkeypatch):
    monkeypatch.setattr(TC, "witness", _witness(_keep_baseline))
    d, rec = run_small(tmp_path, bound)
    assert (rec["status"], rec["verdict"]) == ("STOP", "NOT_QUALIFIED")
    assert rec["failure"]["failing_step"] == "corpus:020" and "admissible set" in rec["failure"]["reason"]
    assert [r["id"] for r in lines(d / Q.OBSERVATIONS)] == ["repro:E1", "corpus:000"]
    assert rec["summary"] is None and rec["observed"] == 2


def test_RUN_a_MODEL_LOAD_failure_is_a_recorded_STOP(pinned, bound, tmp_path):
    def loader():
        raise RuntimeError("no model")
    d, rec = run_small(tmp_path, bound, loader=loader)
    assert (rec["status"], rec["verdict"], rec["failure"]["failing_step"]) == ("STOP", "NOT_QUALIFIED", "model_load")
    assert not (d / Q.OBSERVATIONS).exists() and rec["observations_sha256"] is None


def test_RUN_an_INTERRUPTION_is_recorded_and_RE_RAISED(pinned, bound, tmp_path):
    d = claimed(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        Q._run(small_plan(bound["corpus"]), CONFIG, IDENT, lambda: Steered(fail_on_call=1500), d, bound)
    rec = json.loads((d / "record.json").read_text())
    assert (rec["status"], rec["verdict"]) == ("INTERRUPTED", "NOT_QUALIFIED")
    assert [r["id"] for r in lines(d / Q.OBSERVATIONS)] == list(SMALL[:rec["observed"]])
    assert 1 <= rec["observed"] < len(SMALL) and rec["failure"]["failing_step"] == SMALL[rec["observed"]]


def test_RUN_the_PER_CALL_alarm_is_a_recorded_STOP(pinned, bound, tmp_path):
    d, rec = run_small(tmp_path, bound, call_s=0.05)
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "repro:E1")
    assert rec["failure"]["exception_type"] == "MatchTimeout" and "call repro:E1" in rec["failure"]["reason"]


def test_RUN_the_DEADLINE_is_a_recorded_STOP(pinned, bound, tmp_path):
    t = [0.0]

    def clock():
        t[0] += 10.0
        return t[0]
    d, rec = run_small(tmp_path, bound, clock=clock, wall_s=55.0)   # five reads: load, 2 x (check, alarm)
    assert rec["status"] == "STOP" and rec["failure"]["reason"] == "the 55 s deadline passed before corpus:020"
    assert rec["observed"] == 2


def test_RUN_the_COMPLETION_CHECK_is_inside_the_record(pinned, bound, tmp_path, monkeypatch):
    real = Q.observe
    monkeypatch.setattr(Q, "observe", lambda c, *a: {**real(c, *a), "id": "corpus:999"}
                        if c["id"] == "corpus:000" else real(c, *a))
    d, rec = run_small(tmp_path, bound)
    assert (rec["status"], rec["verdict"]) == ("INCOMPLETE", "NOT_QUALIFIED")
    assert rec["failure"]["missing"] == ["corpus:000"] and rec["summary"] is None


def test_RUN_the_INTERPRETATION_is_inside_the_record(pinned, bound, tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "summarize", lambda rows: 1 / 0)
    d, rec = run_small(tmp_path, bound, ids=SMALL[:2])
    assert (rec["status"], rec["verdict"], rec["failure"]["failing_step"]) == \
        ("STOP", "NOT_QUALIFIED", "interpretation")


def test_RUN_refuses_without_an_UNCONSUMED_claim(bound, tmp_path):
    d = claimed(tmp_path)
    (d / "stranger").write_text("")
    with pytest.raises(Q.QualRefused, match="unconsumed claim"):
        Q._run(small_plan(bound["corpus"]), CONFIG, IDENT, lambda: pytest.fail("loaded"), d, bound)


# ─────────────────────────────── the parent and the public entries ───────────────────────────

@pytest.fixture
def open_gate(monkeypatch, tmp_path, stub_pins):
    monkeypatch.setattr(AUTH, "TACTICAL_CANDIDATE_QUALIFICATION_AUTHORIZED", True)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setattr(TQ, "out_dir", lambda: CORPUS_DIR)
    monkeypatch.setattr(R4, "load_production_evaluator", lambda identity: Steered())
    monkeypatch.setattr(Q, "plan_of", small_plan)
    monkeypatch.setattr(X, "PINNED_FINGERPRINTS", dict(stub_pins))
    return tmp_path


def in_process_watch(monkeypatch):
    def watch(argv, *, limit_s, env, cwd, stdout_path, stderr_path):
        assert list(argv[1:]) == ["-m", Q.MODULE, "run-claimed"] and env["PYTHONUNBUFFERED"] == "1"
        assert limit_s < Q.HARD_LIMIT_S and cwd == str(R4.REPO_ROOT)     # the claim's age is spent
        monkeypatch.setenv(Q.CLAIM_ENV, env[Q.CLAIM_ENV])
        with open(stdout_path, "x") as so, open(stderr_path, "x") as se, \
                contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            rc = Q.main(["run-claimed"])
        return {**WATCH, "limit_s": limit_s, "rc": rc}
    monkeypatch.setattr(CC, "_watch", watch)


def test_PUBLIC_ENTRY_runs_in_the_real_state(open_gate, monkeypatch):
    in_process_watch(monkeypatch)
    w = Q.run()
    out = Q.out_dir()
    rec = json.loads((out / "record.json").read_text())
    assert (w["status"], w["record_status"], w["verdict"], w["child_rc"]) == \
        ("COMPLETE", "COMPLETE", "QUALIFIED", 0), rec["failure"]
    assert w == json.loads((out / Q.WATCHDOG_FILE).read_text())
    assert w["record_sha256"] == CM._file_sha(str(out / "record.json"))
    assert w["observations_sha256"] == rec["observations_sha256"] == CM._file_sha(str(out / Q.OBSERVATIONS))
    claim = json.loads((out / Q.CLAIM_FILE).read_text())
    assert (claim["corpus_sha256"], claim["plan_sha256"]) == (rec["corpus_sha256"], rec["plan_sha256"])
    assert [r["id"] for r in lines(out / Q.OBSERVATIONS)] == list(SMALL)
    assert Q.HARD_LIMIT_S - Q.MARGIN_S - 60 < rec["deadline_s"] < Q.HARD_LIMIT_S - Q.MARGIN_S
    assert json.loads((out / Q.CHILD_STREAMS["stdout"]).read_text()) == {"status": "COMPLETE"}


def test_PUBLIC_ENTRY_a_STOPPED_child_is_NOT_QUALIFIED(open_gate, monkeypatch):
    in_process_watch(monkeypatch)
    monkeypatch.setattr(X, "PINNED_FINGERPRINTS", REAL_PINS)
    w = Q.run()
    assert (w["status"], w["verdict"], w["record_status"]) == ("STOP", "NOT_QUALIFIED", "STOP")


def test_PUBLIC_ENTRY_refuses_an_occupied_destination_before_anything(open_gate, monkeypatch):
    Q.out_dir().mkdir()
    monkeypatch.setattr(Q, "bindings", lambda: pytest.fail("bound"))
    with pytest.raises(Q.QualRefused, match="occupied"):
        Q.run()
    assert list(Q.out_dir().iterdir()) == []


def test_PUBLIC_ENTRY_refuses_BROKEN_BINDINGS_before_the_claim(open_gate, monkeypatch):
    monkeypatch.setitem(Q.CODE_PINS, "h5_incumbent_defence.py", "0" * 64)
    monkeypatch.setattr(CC, "_watch", lambda *a, **k: pytest.fail("a child was started"))
    with pytest.raises(Q.QualRefused, match="h5_incumbent_defence.py is not the pinned code"):
        Q.run()
    assert not Q.out_dir().exists()


def test_the_CHILD_checks_its_bindings_BEFORE_the_load(open_gate, monkeypatch):
    in_process_watch(monkeypatch)
    real = Q.bindings
    calls = []

    def once():                                   # the parent's check passes; the child's fails
        calls.append(1)
        if len(calls) > 1:
            raise Q.QualRefused("the child's binding probe")
        return real()
    monkeypatch.setattr(Q, "bindings", once)
    monkeypatch.setattr(R4, "load_production_evaluator", lambda identity: pytest.fail("loaded"))
    w = Q.run()
    assert (w["status"], w["verdict"], w["child_rc"]) == ("EXITED_WITHOUT_RECORD", "NOT_QUALIFIED", 4)
    assert "the child's binding probe" in (Q.out_dir() / Q.CHILD_STREAMS["stderr"]).read_text()


def test_the_CHILD_takes_only_an_UNCONSUMED_claim_for_THIS_corpus_and_call_list(open_gate, monkeypatch):
    _c, nonce = Q._claim(Q.out_dir(), {"corpus_sha256": "x", "plan_sha256": "y"})
    monkeypatch.setenv(Q.CLAIM_ENV, "0" * 32)
    with pytest.raises(Q.QualRefused, match="nonce does not match"):
        Q.run_claimed()
    monkeypatch.setenv(Q.CLAIM_ENV, nonce)
    with pytest.raises(Q.QualRefused, match="another corpus or call list"):
        Q.run_claimed()
    assert sorted(p.name for p in Q.out_dir().iterdir()) == [Q.CLAIM_FILE]


def test_PUBLIC_ENTRY_the_REAL_child_invocation_reads_the_CLOSED_on_disk_gate(open_gate):
    w = Q.run()
    out = Q.out_dir()
    assert (w["status"], w["verdict"], w["child_exit"]) == ("EXITED_WITHOUT_RECORD", "NOT_QUALIFIED",
                                                            "exit status 5")
    assert "UNAUTHORIZED" in (out / Q.CHILD_STREAMS["stderr"]).read_text()
    assert sorted(p.name for p in out.iterdir()) == sorted([Q.CLAIM_FILE, Q.WATCHDOG_FILE,
                                                            *Q.CHILD_STREAMS.values()])


HANG = r'''
import json, os, signal, sys, time
d = sys.argv[1]
open(os.path.join(d, "log.jsonl"), "x").write('{"step": "model_load"}\n')
json.dump({"status": "COMPLETE", "verdict": "QUALIFIED"}, open(os.path.join(d, "record.json"), "x"))
sys.stdout.write("OUT before\n")
signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())
time.sleep(60)
'''


def test_PUBLIC_ENTRY_a_hung_child_is_TERMINATED_never_QUALIFIED(open_gate, monkeypatch):
    monkeypatch.setattr(Q, "_child_argv", lambda: [sys.executable, "-c", HANG, str(Q.out_dir())])
    monkeypatch.setattr(Q, "HARD_LIMIT_S", 3.0)
    t0 = time.monotonic()
    w = Q.run()
    assert time.monotonic() - t0 < 20, "the parent did not enforce its limit"
    assert (w["status"], w["record_status"], w["verdict"], w["child_exit"]) == \
        ("TERMINATED", "COMPLETE", "NOT_QUALIFIED", "killed by signal SIGKILL")
    assert (Q.out_dir() / Q.CHILD_STREAMS["stdout"]).read_text() == "OUT before\n"
    assert w["journal"]["lines"] == 1


@pytest.mark.parametrize("exc,status", [(RuntimeError, "SETTLE_FAILED"), (KeyboardInterrupt, "INTERRUPTED")])
def test_a_SETTLEMENT_failure_or_interruption_still_leaves_watchdog_json(open_gate, monkeypatch, exc, status):
    in_process_watch(monkeypatch)

    def boom(d):
        raise exc("settlement probe")
    monkeypatch.setattr(CC, "_journal", boom)
    with pytest.raises(exc):
        Q.run()
    w = json.loads((Q.out_dir() / Q.WATCHDOG_FILE).read_text())
    assert (w["status"], w["verdict"], w["failure"]["exception_type"], w["record_present"]) == \
        (status, "NOT_QUALIFIED", exc.__name__, True)


def test_SETTLE_an_unreadable_or_abnormal_record_is_never_QUALIFIED(tmp_path):
    for rc, body, status in ((0, "{", "EXITED_WITHOUT_RECORD"),
                             (1, json.dumps({"status": "COMPLETE", "verdict": "QUALIFIED"}), "EXITED_ABNORMALLY"),
                             (0, json.dumps({"status": "COMPLETE", "verdict": "NOT_QUALIFIED"}), "COMPLETE")):
        d = tmp_path / f"{rc}{status}"
        d.mkdir()
        (d / "record.json").write_text(body)
        w = Q._settle(d, {**WATCH, "rc": rc})
        assert (w["status"], w["verdict"]) == (status, "NOT_QUALIFIED")


@pytest.mark.parametrize("entry", ["run", "run_claimed"])
def test_every_public_entry_refuses_with_the_gate_shut_BEFORE_anything(entry, monkeypatch):
    for owner, name in ((Q, "bindings"), (Q, "_claim"), (Q, "_run"), (CC, "_watch"), (os, "mkdir"),
                        (R4, "load_production_evaluator")):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail("touched before the gate"))
    with pytest.raises(Q.QualUnauthorized):
        getattr(Q, entry)()


def test_the_CLI_exits_5_and_takes_no_output_directory(tmp_path):
    r = subprocess.run([sys.executable, "-m", Q.MODULE, "run"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == Q.EXIT_UNAUTHORIZED == 5, r.stderr
    r = subprocess.run([sys.executable, "-m", Q.MODULE, "run", "--out-dir", str(tmp_path)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 2 and "--out-dir" in r.stderr
