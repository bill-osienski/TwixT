"""Candidate match runner (R1) -- SYNTHETIC ONLY.

No model is loaded, no research seed is used, no real game is played. Games are played by a fast
deterministic stand-in (`Fake`) whose stream derives exactly as the real agent's -- seed ^
SEARCH_MASK[colour] -- behind the REAL candidate wrapper; the production construction path and
the first plies of exact replay are also checked on the REAL frozen search with the stub
evaluator. Seeds are negative (synthetic).
"""
import contextlib
import copy
import hashlib
import json
import os
import pathlib
import random
import shutil
import signal
import subprocess
import sys
import time

import pytest

from scripts.GPU.alphazero import candidate_match_authorization as AUTH
from scripts.GPU.alphazero import candidate_match_runner as CM
from scripts.GPU.alphazero import eval_readout as RO
from scripts.GPU.alphazero import h4_pilot_feasibility as F
from scripts.GPU.alphazero import h4_runner as R4
from scripts.GPU.alphazero import h5_cap_completion as CC
from scripts.GPU.alphazero import h5_incumbent_tactics as T
from scripts.GPU.alphazero import tactical_candidate as TC
from scripts.GPU.alphazero import tactical_corpus as TQ
from scripts.GPU.alphazero import tactical_qualification as Q
from scripts.GPU.alphazero import twixtbot_g3_reference as G3
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move as enc
from tests.test_h5_incumbent_defence import CONFIG, IDENT, Steered

ROOT = pathlib.Path(__file__).resolve().parents[1]
K = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
#: probed and pinned (each test re-checks what it needs): FIRED -- black's first override at ply
#: 129, both games end in a win; QUIET -- nothing fires, both games reach the cap
FIRED = {"pair_id": "p-fired", "seeds": {"red": -3, "black": -4}}
QUIET = {"pair_id": "p-quiet", "seeds": {"red": -13, "black": -14}}
PLAN = {"mode": "fixture", "pairs": [FIRED, QUIET]}
#: the REAL committed qualification and corpus evidence, read-only (A8.4), wherever a test moves
#: the evidence root
QUAL_DIR, CORPUS_DIR = Q.out_dir(), TQ.out_dir()


def real_evidence(monkeypatch):
    monkeypatch.setattr(Q, "out_dir", lambda: QUAL_DIR)
    monkeypatch.setattr(TQ, "out_dir", lambda: CORPUS_DIR)


@pytest.fixture(scope="module")
def binding():
    """The guard on the REAL committed qualification evidence -- no model; reads only."""
    return CM.qualification_binding()


class Fake:
    """A fast, deterministic stand-in for the seeded reference agent: one stream from
    seed ^ SEARCH_MASK[colour]; five legal moves 'searched' (preferring a knight's step from
    its own pegs); a REAL capture -- counts, and a root holding THE state."""

    def __init__(self, colour, seed, capture, log=None):
        self.colour, self.seed, self.capture, self.last_capture = colour, seed, capture, None
        self.rng = random.Random(seed ^ G3.SeededReferenceAgent.SEARCH_MASK[colour])
        self.log = log

    def __call__(self, state):
        if self.log is not None:
            self.log.append((self.colour, state.ply))
        legal = sorted(state.legal_moves())
        own = {(r + a, c + b) for (r, c), col in state.pegs.items() if col == self.colour
               for a, b in K}
        near = [m for m in legal if m in own]
        pool = near if near and self.rng.random() < 0.7 else legal
        picks = self.rng.sample(pool, min(5, len(pool)))
        counts = {m: 0 for m in legal}
        root = MCTSNode(state=state, priors={enc(*m): 1.0 for m in legal})
        for i, m in enumerate(picks):
            counts[m] = 5 - i
            root.children[enc(*m)] = MCTSNode(state=state.apply_move(m), parent=root,
                                              move=enc(*m), visit_count=5 - i)
        move = RO._argmax(counts)
        if self.capture:
            self.last_capture = {"ply": state.ply, "move": move, "counts": counts, "root": root}
        return move


def fake_make(builds=None, asked=None, agent=Fake):
    def make(colour, seed, role):
        a = agent(colour, seed, role == "candidate", asked)
        if builds is not None:
            builds.append((colour, seed, role, a))
        return TC.TacticalCandidate(a, enabled=True) if role == "candidate" else a
    return make


def fake_builder(**kw):
    return lambda evaluator: fake_make(**kw)


@pytest.fixture(scope="module")
def played():
    """Both pinned pairs, played once: games, the builds and every (colour, ply) asked."""
    out = {}
    for p in (FIRED, QUIET):
        builds, asked = [], []
        a = CM.play_game(fake_make(builds, asked), p, "A", limit_s=600)
        b = CM.play_game(fake_make(builds, asked), p, "B", limit_s=600)
        out[p["pair_id"]] = {"A": a, "B": b, "builds": builds, "asked": asked}
    return out


# ─────────────────────────────── the gate, the constants, the plan ───────────────────────────

def test_the_gate_is_created_CLOSED():
    assert AUTH.CANDIDATE_MATCH_AUTHORIZED is False


def test_the_CONSTANTS_are_the_cards():
    assert (CM.BOARD, CM.PLY_CAP) == (T.BOARD, T.PLY_CAP) == (TC.BOARD, TC.PLY_CAP) == (24, 280)
    assert (CM.GAME_DEADLINE_S, CM.SEGMENT_DEADLINE_S, CM.STUDY_CEILING_S) == (900, 14_400, 288_000)
    assert (CM.CHILD_GRACE_S, CM.POLL_S) == (30, 0.5), "the child's alarm fires before the parent's kill"
    import inspect
    assert inspect.signature(CM._run).parameters["game_s"].default == CM.GAME_DEADLINE_S - CM.CHILD_GRACE_S
    assert CM.ARMS == {"A": "red", "B": "black"}


@pytest.mark.parametrize("plan,mode,why", [
    ({"mode": "pilot", "pairs": [FIRED]}, "fixture", "not a fixture plan"),
    ({"mode": "fixture", "pairs": []}, "fixture", "none given"),
    ({"mode": "fixture", "pairs": [{"pair_id": "x", "seeds": {"red": -1}}]}, "fixture", "one per colour"),
    ({"mode": "fixture", "pairs": [FIRED, {**QUIET, "pair_id": "p-fired"}]}, "fixture", "unique"),
    ({"mode": "fixture", "pairs": [FIRED, {**QUIET, "seeds": {"red": -3, "black": -9}}]}, "fixture",
     "a seed repeats"),
    ({"mode": "fixture", "pairs": [{"pair_id": "x", "seeds": {"red": -1, "black": -1}}]}, "fixture",
     "a seed repeats"),
    ({"mode": "fixture", "pairs": [{"pair_id": "x", "seeds": {"red": -1, "black": 2.0}}]}, "fixture",
     "must be an int"),
    ({"mode": "fixture", "pairs": [{"pair_id": "x", "seeds": {"red": -1, "black": 5}}]}, "fixture",
     "NEGATIVE seeds only"),
    ({"mode": "pilot", "pairs": [{"pair_id": "x", "seeds": {"red": -1, "black": 5}}]}, "pilot",
     "negative"),
    ({"mode": "pilot", "pairs": [{"pair_id": "x", "seeds": {"red": 987_654_301, "black": 987_654_303}}]},
     "pilot", "not registered"),
])
def test_the_PLAN_is_refused(plan, mode, why):
    with pytest.raises(CM.MatchRefused, match=why):
        CM.check_plan(plan, mode=mode)


# ─────────────────────────────── seed coupling and exact replay (card §7.1) ───────────────────

@pytest.mark.parametrize("pid", ["p-fired", "p-quiet"])
def test_SEEDS_FOLLOW_COLOUR_never_role_and_every_game_builds_FRESH_agents(played, pid):
    b = played[pid]["builds"]
    seeds = {"p-fired": FIRED, "p-quiet": QUIET}[pid]["seeds"]
    assert [(c, s, r) for c, s, r, _ in b] == [
        ("red", seeds["red"], "candidate"), ("black", seeds["black"], "incumbent"),
        ("red", seeds["red"], "incumbent"), ("black", seeds["black"], "candidate")]
    assert len({id(a) for *_, a in b}) == 4, "one FRESH agent per colour per game"


def test_a_pair_where_NOTHING_FIRED_is_MOVE_IDENTICAL(played):
    a, b = played["p-quiet"]["A"], played["p-quiet"]["B"]
    assert CM.first_override(a) is None and CM.first_override(b) is None
    assert a["moves"] == b["moves"] and a["digest"] == b["digest"]
    assert CM.check_pair(a, b) == {"pair_id": "p-quiet", "fired": False,
                                   "first_override": {"A": None, "B": None}, "agree_through": 280}


def test_a_FIRED_pair_agrees_up_to_the_first_override_where_the_BASELINE_is_the_other_move(played):
    a, b = played["p-fired"]["A"], played["p-fired"]["B"]
    pc = CM.check_pair(a, b)
    assert pc["fired"] and pc["first_override"] == {"A": None, "B": 129} and pc["agree_through"] == 129
    assert a["moves"][:129] == b["moves"][:129] and a["moves"][129] != b["moves"][129]
    w = next(x for x in b["witnesses"] if x["ply"] == 129)
    assert (w["rule"], w["override"], w["baseline"][:2]) == ("override", True, a["moves"][129])


@pytest.mark.parametrize("breakage,why", [
    ("quiet: one move of B changed", "nothing fired, yet the two games differ"),
    ("fired: a move before the override changed", "differ before the first override"),
    ("fired: the overriding baseline changed", "is not the other game's move"),
    ("arms swapped", "not one A-then-B pair"),
])
def test_an_EXACT_REPLAY_breach_STOPS(played, breakage, why):
    pid = "p-quiet" if breakage.startswith("quiet") else "p-fired"
    a, b = copy.deepcopy(played[pid]["A"]), copy.deepcopy(played[pid]["B"])
    if breakage.startswith("quiet"):
        b["moves"][100] = [0, 1]
    elif breakage == "fired: a move before the override changed":
        b["moves"][50] = [0, 1]
    elif breakage == "fired: the overriding baseline changed":
        next(x for x in b["witnesses"] if x["ply"] == 129)["baseline"][:2] = [0, 1]
    else:
        a, b = b, a
    with pytest.raises(CM.MatchStop, match=why):
        CM.check_pair(a, b)


# ─────────────────────────────── one game: board, cap, winner first, witnesses ────────────────

def test_the_WINNER_is_checked_first_and_nothing_is_asked_after_a_win(played):
    for arm in "AB":
        g = played["p-fired"][arm]
        s = T.replay(g["moves"])
        assert g["terminal"] == {"reason": "win", "ply": len(g["moves"])} and s.ply < 280
        assert s.winner() is not None and T.replay(g["moves"][:-1]).winner() is None


def test_the_CAP_ends_the_game_at_280_and_no_move_is_asked_at_280(played):
    g = played["p-quiet"]["A"]
    assert g["terminal"] == {"reason": "cap", "ply": 280} and len(g["moves"]) == 280
    assert T.replay(g["moves"]).winner() is None
    assert max(p for _c, p in played["p-quiet"]["asked"]) == 279


def test_every_game_is_on_the_OFFICIAL_board_and_digested(played):
    for p in played.values():
        for arm in "AB":
            g = p[arm]
            assert T.replay(g["moves"]).board_size == T.replay(g["moves"]).active_size == 24
            assert g["digest"] == CM.game_digest(g["moves"], g["terminal"])


def test_the_WITNESSES_are_exactly_the_candidates_decisions(played):
    for p in played.values():
        for arm, parity in (("A", 0), ("B", 1)):
            g = p[arm]
            assert [w["ply"] for w in g["witnesses"]] == list(range(parity, len(g["moves"]), 2))
            assert all(w["final"][:2] == g["moves"][w["ply"]] for w in g["witnesses"])


def test_a_SKIPPED_witness_and_an_ILLEGAL_move_STOP():
    class Skipping(TC.TacticalCandidate):
        def __call__(self, state):
            m = super().__call__(state)
            if state.ply == 10:
                self.witnesses.pop()
            return m

    def make(colour, seed, role):
        a = Fake(colour, seed, role == "candidate")
        return Skipping(a, enabled=True) if role == "candidate" else a
    with pytest.raises(CM.MatchStop, match="witnesses are not its decisions"):
        CM.play_game(make, FIRED, "A", limit_s=600)

    class Cheat(Fake):
        def __call__(self, state):
            m = super().__call__(state)
            return next(iter(state.pegs)) if state.ply == 7 else m
    with pytest.raises(CM.MatchStop, match="illegal move"):
        CM.play_game(fake_make(agent=Cheat), FIRED, "A", limit_s=600)


def test_the_GAME_LIMIT_is_HARD_and_also_checked_between_moves():
    class Slow(Fake):
        def __call__(self, state):
            if state.ply == 3:
                time.sleep(3)
            return super().__call__(state)
    t0 = time.monotonic()
    with pytest.raises(CM.MatchTimeout, match="hard limit of 0.5 s was exceeded"):
        CM.play_game(fake_make(agent=Slow), FIRED, "A", limit_s=0.5)
    assert time.monotonic() - t0 < 2.5, "interrupted mid-call, not after it"
    t = [0.0]

    def clock():
        t[0] += 1.0
        return t[0]
    with pytest.raises(CM.MatchTimeout, match="passed at ply"):
        CM.play_game(fake_make(), FIRED, "A", limit_s=20, clock=clock)


# ─────────────────────────────── the production construction path (real search, stub) ─────────

def test_the_PRODUCTION_BUILDER_binds_seed_to_colour_and_wraps_only_the_candidate():
    make = CM.production_builder(CONFIG, Steered(), IDENT)
    cand, inc = make("red", -21, "candidate"), make("black", -22, "incumbent")
    assert isinstance(cand, TC.TacticalCandidate) and cand.enabled and cand.agent.capture
    assert not isinstance(inc, TC.TacticalCandidate) and inc.capture is False
    for agent, colour, seed in ((cand.agent, "red", -21), (inc, "black", -22)):
        assert (agent.colour, agent.seed) == (colour, seed)
        fresh = G3.SeededReferenceAgent.SEARCH_MASK[colour] ^ seed
        assert agent.mcts.rng.getstate() == random.Random(fresh).getstate()


def test_EXACT_REPLAY_on_the_REAL_search_over_the_first_plies():
    """Both games of one pair, stepped together on the frozen search: identical until an override."""
    make = CM.production_builder(CONFIG, Steered(), IDENT)
    seeds = {"red": -31, "black": -32}
    A = {c: make(c, seeds[c], "candidate" if c == "red" else "incumbent") for c in CM.COLOURS}
    B = {c: make(c, seeds[c], "candidate" if c == "black" else "incumbent") for c in CM.COLOURS}
    sa, sb = T.replay([]), T.replay([])
    for _ in range(4):
        ma, mb = A[sa.to_move](sa), B[sb.to_move](sb)
        assert ma == mb
        sa, sb = sa.apply_move(ma), sb.apply_move(mb)
    inner = lambda a: a.agent if isinstance(a, TC.TacticalCandidate) else a   # noqa: E731
    for c in CM.COLOURS:
        assert inner(A[c]).mcts.rng.getstate() == inner(B[c]).mcts.rng.getstate()
    assert [w["override"] for w in A["red"].witnesses + B["black"].witnesses] == [False] * 4


# ─────────────────────────────── the child's run (record, journal, blinding) ──────────────────

def claimed(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    d = tmp_path / "seg"
    CM._create(d)
    CM._claim(d, {})
    return d


def journal(d):
    return [json.loads(line) for line in (d / "log.jsonl").read_text().splitlines()]


JOURNAL_KEYS = {"model_load": {"incumbent_identity"}, "game_start": {"pair_id", "arm"},
                "game": {"pair_id", "arm", "digest", "overrides", "seconds"},
                "pair": {"pair_id", "fired", "first_override"}}


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    d = claimed(tmp_path_factory.mktemp("complete"))
    return d, CM._run(PLAN, CONFIG, IDENT, object, d, mode="fixture", builder=fake_builder())


def test_RUN_COMPLETE_records_every_game_and_every_replay_check(complete):
    d, rec = complete
    assert (rec["status"], rec["failure"]) == ("COMPLETE", None)
    assert [(g["pair_id"], g["arm"]) for g in rec["games"]] == \
        [("p-fired", "A"), ("p-fired", "B"), ("p-quiet", "A"), ("p-quiet", "B")]
    assert [p["fired"] for p in rec["pairs"]] == [True, False]
    assert [e["step"] for e in journal(d)] == ["model_load"] + \
        ["game_start", "game", "game_start", "game", "pair"] * 2
    assert rec["plan_sha256"] == CM._sha(PLAN) and rec["wrapper_sha256"] == CM._file_sha(TC.__file__)
    assert json.loads((d / "record.json").read_text())["status"] == "COMPLETE"


def test_the_JOURNAL_carries_NO_OUTCOME(complete):
    d, _ = complete
    for e in journal(d):
        assert set(e) == {"step", "observation"} and set(e["observation"]) == JOURNAL_KEYS[e["step"]], e
    text = (d / "log.jsonl").read_text()
    assert not R4.blinding_violations(text)
    for key in ("winner", "terminal", "reason", "moves", "agree_through", "score", "ply"):
        assert f'"{key}"' not in text, key


def test_RUN_an_EXACT_REPLAY_failure_is_a_recorded_STOP(tmp_path):
    shared = random.Random(7)

    class Drifting(Fake):                         # draws from a SHARED stream: not replayable
        def __call__(self, state):
            if shared.random() < 0.5:
                self.rng.random()
            return super().__call__(state)
    rec = CM._run({"mode": "fixture", "pairs": [QUIET]}, CONFIG, IDENT, object, claimed(tmp_path),
                  mode="fixture", builder=fake_builder(agent=Drifting))
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == ("STOP", "replay:p-quiet", "MatchStop")
    assert len(rec["games"]) == 2


def test_RUN_MODEL_LOAD_INTERRUPTED_GAME_LIMIT_and_SEGMENT_LIMIT(tmp_path):
    def load():
        raise FileNotFoundError("checkpoint not found")
    rec = CM._run(PLAN, CONFIG, IDENT, load, claimed(tmp_path / "m"), mode="fixture",
                  builder=fake_builder())
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "model_load")

    class Interrupting(Fake):
        def __call__(self, state):
            if state.ply == 5:
                raise KeyboardInterrupt
            return super().__call__(state)
    d = claimed(tmp_path / "i")
    with pytest.raises(KeyboardInterrupt):
        CM._run(PLAN, CONFIG, IDENT, object, d, mode="fixture", builder=fake_builder(agent=Interrupting))
    assert json.loads((d / "record.json").read_text())["status"] == "INTERRUPTED"

    class Slow(Fake):
        def __call__(self, state):
            if state.ply == 3:
                time.sleep(3)
            return super().__call__(state)
    rec = CM._run(PLAN, CONFIG, IDENT, object, claimed(tmp_path / "g"), mode="fixture",
                  builder=fake_builder(agent=Slow), game_s=0.5)
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == ("STOP", "game:p-fired-A", "MatchTimeout")
    t = [0.0]

    def slow_load():
        t[0] += 20_000.0
        return object()
    rec = CM._run(PLAN, CONFIG, IDENT, slow_load, claimed(tmp_path / "s"), mode="fixture",
                  builder=fake_builder(), clock=lambda: t[0])
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "game:p-fired-A")
    assert "segment" in rec["failure"]["reason"]


def test_RUN_an_INEXACT_set_is_INCOMPLETE_and_a_bad_plan_writes_NOTHING(tmp_path, monkeypatch):
    real = CM.check_pair
    monkeypatch.setattr(CM, "check_pair", lambda a, b: {**real(a, b), "pair_id": "renamed"})
    rec = CM._run({"mode": "fixture", "pairs": [QUIET]}, CONFIG, IDENT, object, claimed(tmp_path / "x"),
                  mode="fixture", builder=fake_builder())
    assert (rec["status"], rec["failure"]["failing_step"], rec["failure"]["exception_type"]) == \
        ("INCOMPLETE", "completion_check", None)
    d = claimed(tmp_path / "bad")
    with pytest.raises(CM.MatchRefused):
        CM._run({"mode": "fixture", "pairs": []}, CONFIG, IDENT, object, d, mode="fixture")
    assert sorted(os.listdir(d)) == [CM.CLAIM_FILE]


# ─────────────────────────────── the claim, the parent, the public entries ────────────────────

def test_the_CHILD_takes_only_an_UNCONSUMED_claim_with_ITS_nonce(tmp_path, monkeypatch):
    d = tmp_path / "c"
    CM._create(d)
    _claim, nonce = CM._claim(d, {"plan_sha256": "x"})
    monkeypatch.setenv(CM.CLAIM_ENV, "0" * 32)
    with pytest.raises(CM.MatchRefused, match="nonce does not match"):
        CM._accept_claim(d)
    monkeypatch.setenv(CM.CLAIM_ENV, nonce)
    assert CM._accept_claim(d)["plan_sha256"] == "x"
    (d / "log.jsonl").write_text("")
    with pytest.raises(CM.MatchRefused, match="unconsumed claim"):
        CM._accept_claim(d)


@pytest.mark.parametrize("entry", ["run", "run_claimed"])
def test_every_public_entry_refuses_with_the_gate_shut_BEFORE_anything(entry, monkeypatch):
    for owner, name in ((CM, "_committed_plan"), (CM, "qualification_binding"), (CM, "_segment_limit"),
                        (CM, "_create"), (CM, "_claim"), (CM, "_run"), (CM, "_watch_games"), (os, "mkdir"),
                        (R4, "load_production_evaluator")):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail("touched before the gate"))
    with pytest.raises(CM.MatchUnauthorized):
        getattr(CM, entry)("pilot", 0)


def test_the_CLI_exits_5_and_takes_no_output_directory(tmp_path):
    r = subprocess.run([sys.executable, "-m", CM.MODULE, "run", "--role", "pilot"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == CM.EXIT_UNAUTHORIZED == 5, r.stderr
    r = subprocess.run([sys.executable, "-m", CM.MODULE, "run", "--role", "pilot", "--out-dir",
                        str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "--out-dir" in r.stderr


@pytest.mark.parametrize("role,segment", [("pilot", 1), ("study", -1), ("practice", 0)])
def test_only_the_pilot_and_study_segments_exist(role, segment):
    with pytest.raises(CM.MatchRefused, match="no .* segment"):
        CM._role(role, segment)


PILOT_PLAN = {"mode": "pilot", "pairs": [{"pair_id": "p1", "seeds": {"red": 3, "black": 4}}]}


def commit_plan(monkeypatch, plan):
    raw = json.dumps(plan).encode()
    monkeypatch.setattr(CM, "_committed_plan", lambda role, segment: (
        CM.check_plan(plan, mode=role) or plan, hashlib.sha256(raw).hexdigest()))


@pytest.fixture
def open_gate(monkeypatch, tmp_path, binding):
    """The gate open, the evidence root in tmp, the plan 'committed' and bound to the REAL
    qualification (A8.3-A8.4, checked for real against the committed evidence), the seeds
    'registered', the stand-in agents for speed. Everything else -- guard, claim, child entry,
    record, timing -- is real."""
    monkeypatch.setattr(AUTH, "CANDIDATE_MATCH_AUTHORIZED", True)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setattr(R4, "seed_registry_faults", lambda tasks: [])
    monkeypatch.setattr(R4, "load_production_evaluator", lambda identity: object())
    monkeypatch.setattr(CM, "production_builder", lambda config, ev, ident: fake_make())
    real_evidence(monkeypatch)
    commit_plan(monkeypatch, {**PILOT_PLAN, "qualification": binding})
    return tmp_path


def test_PUBLIC_ENTRY_runs_in_the_real_state_and_reports_STATUSES_only(open_gate, monkeypatch):
    def watch(argv, *, limit_s, game_s, d, env, cwd, stdout_path, stderr_path):
        assert list(argv[1:]) == ["-m", CM.MODULE, "run-claimed", "--role", "pilot", "--segment", "0"]
        assert env["PYTHONUNBUFFERED"] == "1" and limit_s <= CM.SEGMENT_DEADLINE_S
        assert (game_s, d) == (CM.GAME_DEADLINE_S, CM.out_dir("pilot", 0)), "the parent enforces 15 min"
        monkeypatch.setenv(CM.CLAIM_ENV, env[CM.CLAIM_ENV])
        with open(stdout_path, "x") as so, open(stderr_path, "x") as se, \
                contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            rc = CM.main(list(argv[3:]))
        return {"child_pid": os.getpid(), "limit_s": limit_s, "terminated": False, "cause": None,
                "signal": None, "rc": rc, "elapsed_s": 1.0, "survivors_killed": False,
                "game_limit_s": game_s, "game_breach": None}
    monkeypatch.setattr(CM, "_watch_games", watch)
    res = CM.run("pilot", 0)
    out = CM.out_dir("pilot", 0)
    rec = json.loads((out / "record.json").read_text())
    assert (res["status"], res["record_status"], rec["status"]) == ("COMPLETE", "COMPLETE", "COMPLETE")
    assert (out / CM.CHILD_STREAMS["stdout"]).read_text().strip() == '{"status": "COMPLETE"}'
    for text in ((out / CM.WATCHDOG_FILE).read_text(), (out / "log.jsonl").read_text()):
        assert not R4.blinding_violations(text) and "terminal" not in text and "winner" not in text
        assert '"moves"' not in text and '"agree_through"' not in text, "no game length, no move"
    claim = json.loads((out / CM.CLAIM_FILE).read_text())
    assert (claim["role"], claim["segment"]) == ("pilot", 0) and len(claim["plan_sha256"]) == 64


def test_PUBLIC_ENTRY_the_REAL_child_reads_the_CLOSED_on_disk_gate(open_gate):
    res = CM.run("pilot", 0)
    assert (res["status"], res["child_exit"]) == ("EXITED_WITHOUT_RECORD", "exit status 5")


def test_the_CHILD_refuses_a_claim_for_another_plan_role_or_segment(open_gate, monkeypatch):
    CM._create(CM.out_dir("pilot", 0))
    _c, nonce = CM._claim(CM.out_dir("pilot", 0), {"plan_sha256": "t" * 64, "role": "pilot", "segment": 0})
    monkeypatch.setenv(CM.CLAIM_ENV, nonce)
    with pytest.raises(CM.MatchRefused, match="another plan, role or segment"):
        CM.run_claimed("pilot", 0)


def test_PUBLIC_ENTRY_refuses_an_UNCOMMITTED_plan_before_any_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(AUTH, "CANDIDATE_MATCH_AUTHORIZED", True)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setattr(CM, "_watch_games", lambda *a, **k: pytest.fail("a child was started"))
    with pytest.raises(CM.MatchRefused, match="must be committed before any model load"):
        CM.run("pilot", 0)
    assert not CM.out_dir("pilot", 0).exists()


HANG = r'''
import json, os, signal, sys, time
d = sys.argv[1]
open(os.path.join(d, "log.jsonl"), "x").write('{"step": "model_load"}\n')
json.dump({"status": "COMPLETE"}, open(os.path.join(d, "record.json"), "x"))
signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())
time.sleep(60)
'''


def test_PUBLIC_ENTRY_a_hung_child_is_TERMINATED_never_COMPLETE(open_gate, monkeypatch):
    monkeypatch.setattr(CM, "_child_argv", lambda role, segment: [sys.executable, "-c", HANG,
                                                                  str(CM.out_dir(role, segment))])
    monkeypatch.setattr(CM, "SEGMENT_DEADLINE_S", 3.0)
    t0 = time.monotonic()
    res = CM.run("pilot", 0)
    assert time.monotonic() - t0 < 20
    assert (res["status"], res["record_status"]) == ("TERMINATED", "COMPLETE")


# ─────────────────────────────── §13 A3: a hard game limit, durable evidence, validated predecessors

def test_RUN_every_completed_game_and_decision_is_DURABLE(complete):
    d, rec = complete
    for g in rec["games"]:
        raw = (d / CM.GAMES_DIR / g["file"]).read_bytes()
        game = json.loads(raw)
        assert hashlib.sha256(raw).hexdigest() == g["sha256"] and game["digest"] == g["digest"]
        lines = (d / CM.GAMES_DIR / g["file"].replace(".json", ".partial.jsonl")).read_text().splitlines()
        dec = [json.loads(x) for x in lines]
        assert [x["move"] for x in dec] == game["moves"] and [x["ply"] for x in dec] == list(range(len(dec)))
        assert [x["witness"] for x in dec if x["witness"]] == game["witnesses"]


UNRESPONSIVE = r'''
import json, os, signal, sys, time
d = sys.argv[1]
with open(os.path.join(d, "log.jsonl"), "x") as fh:
    fh.write(json.dumps({"step": "model_load", "observation": {}}) + "\n")
    fh.write(json.dumps({"step": "game_start", "observation": {"pair_id": "p1", "arm": "A"}}) + "\n")
signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())   # no handler can ever run
time.sleep(120)
'''


def test_an_UNRESPONSIVE_child_is_KILLED_by_the_PARENT_at_the_GAME_limit(open_gate, monkeypatch):
    monkeypatch.setattr(CM, "_child_argv", lambda role, segment: [sys.executable, "-c", UNRESPONSIVE,
                                                                  str(CM.out_dir(role, segment))])
    monkeypatch.setattr(CM, "GAME_DEADLINE_S", 2.0)          # the segment keeps its 4 h
    t0 = time.monotonic()
    res = CM.run("pilot", 0)
    assert time.monotonic() - t0 < 10, "killed at the GAME limit, not the segment's"
    assert (res["status"], res["game_breach"], res["signal"]) == ("TERMINATED", "p1-A", "SIGKILL")
    assert "game p1-A open longer than 2.0 s" in res["cause"] and res["child_exit"] == "killed by signal SIGKILL"


KILLER = r'''
import os, pathlib, signal, sys
sys.path.insert(0, sys.argv[2])
from tests.test_candidate_match_runner import CONFIG, FIRED, IDENT, Fake, fake_builder
from scripts.GPU.alphazero import candidate_match_runner as CM
calls = [0]
class Killer(Fake):                               # SIGKILL mid-way through game B
    def __call__(self, state):
        calls[0] += 1
        if calls[0] == int(sys.argv[3]) + 31:
            os.kill(os.getpid(), signal.SIGKILL)
        return super().__call__(state)
CM._run({"mode": "fixture", "pairs": [FIRED]}, CONFIG, IDENT, object, pathlib.Path(sys.argv[1]),
        mode="fixture", builder=fake_builder(agent=Killer))
'''


def test_a_KILL_after_completed_observations_loses_NONE_of_them(played, tmp_path):
    d = claimed(tmp_path)
    plies_a = len(played["p-fired"]["A"]["moves"])
    r = subprocess.run([sys.executable, "-c", KILLER, str(d), str(ROOT), str(plies_a)], cwd=ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == -signal.SIGKILL, r.stderr
    assert not (d / "record.json").exists()
    assert [e["step"] for e in journal(d)] == ["model_load", "game_start", "game", "game_start"]
    a = json.loads((d / CM.GAMES_DIR / "p-fired-A.json").read_text())
    assert a["moves"] == played["p-fired"]["A"]["moves"], "the completed game, whole"
    b = [json.loads(x) for x in (d / CM.GAMES_DIR / "p-fired-B.partial.jsonl").read_text().splitlines()]
    assert [x["ply"] for x in b] == list(range(30)), "every completed decision of the killed game"
    assert [x["move"] for x in b] == played["p-fired"]["B"]["moves"][:30]
    summary = CM._settle(d, {"terminated": True, "cause": "test kill", "limit_s": 1.0, "elapsed_s": 1.0,
                             "child_pid": 1, "rc": -9, "signal": "SIGKILL", "survivors_killed": False,
                             "game_limit_s": 900.0, "game_breach": "p-fired-B"})
    assert [g["file"] for g in summary["games"]] == ["p-fired-A.json", "p-fired-A.partial.jsonl",
                                                     "p-fired-B.partial.jsonl"]
    assert all(g["sha256"] == hashlib.sha256((d / CM.GAMES_DIR / g["file"]).read_bytes()).hexdigest()
               for g in summary["games"])
    assert summary["record_sha256"] is None and summary["status"] == "TERMINATED"


STUDY_PLAN = {"mode": "study", "pairs": [{"pair_id": "s1", "seeds": {"red": 13, "black": 14}}]}
#: segment 0's measured time, every component distinct and nonzero: timing.json's four intervals
#: (m4 - m0 = 270,015 s) and its verification run (100 s). The watchdog's child time is 270,000 s.
MARKS = (0.0, 3.0, 10.0, 270_010.0, 270_015.0)
VERIFY = {"t_fixed_s": 40.0, "t_games_s": 60.0}
FULL_S0 = 270_115.0


def write_verification(role, k, d, **over):
    path = CM.verification_path(role, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = {"role": role, "segment": k, "verdict": "VERIFIED", "game_files": 2 * len(STUDY_PLAN["pairs"]),
         "watchdog_sha256": CM._file_sha(str(d / CM.WATCHDOG_FILE)),
         "record_sha256": CM._file_sha(str(d / "record.json")),
         "timing_sha256": CM._file_sha(str(d / "timing.json")), **VERIFY, **over}
    path.write_text(json.dumps(v))
    publish_marker(path)


def publish_marker(path):
    """A9.2: the verification record's PUBLISHED marker, bound to the record as it is now."""
    (path.parent / f"{path.stem}.published.json").write_text(json.dumps(
        {"status": "PUBLISHED", "record": path.name, "record_sha256": CM._file_sha(str(path))}))


@pytest.fixture(scope="module")
def pristine(tmp_path_factory):
    """A genuinely completed synthetic study segment 0 -- plan, claim, games, record, watchdog."""
    root = tmp_path_factory.mktemp("pristine")
    mp = pytest.MonkeyPatch()
    mp.setattr(R4, "EVIDENCE_ROOT", str(root))
    mp.setattr(R4, "seed_registry_faults", lambda tasks: [])
    try:
        CM.plan_path("study", 0).parent.mkdir(parents=True)
        CM.plan_path("study", 0).write_text(json.dumps(STUDY_PLAN))
        d = CM.out_dir("study", 0)
        CM._create(d)
        CM._claim(d, {"plan_sha256": CM._file_sha(str(CM.plan_path("study", 0))), "role": "study",
                      "segment": 0})
        assert CM._run(STUDY_PLAN, CONFIG, IDENT, object, d, mode="study",
                       builder=fake_builder())["status"] == "COMPLETE"
        CM._settle(d, {"terminated": False, "cause": None, "limit_s": 14_400.0, "elapsed_s": 270_000.0,
                       "child_pid": 1, "rc": 0, "signal": None, "survivors_killed": False,
                       "game_limit_s": 900.0, "game_breach": None})
        CM._finalize(d, MARKS)
        write_verification("study", 0, d)
    finally:
        mp.undo()
    return root


@pytest.fixture
def segment0(pristine, tmp_path, monkeypatch):
    root = tmp_path / "ev"
    shutil.copytree(pristine, root)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(root))
    monkeypatch.setattr(R4, "seed_registry_faults", lambda tasks: [])

    def durable(paths, _fixture):                 # committed == present, for these controls
        for x in paths:
            if not os.path.exists(x):
                raise F.H4AnalysisRefused(f"{x} is not committed")
    monkeypatch.setattr(F, "check_durable", durable)
    monkeypatch.setattr(CM, "_T_START", 1_000.0)  # this process's time so far: exactly 85 s
    monkeypatch.setattr(CM, "_monotonic", lambda: 1_085.0)
    return CM.out_dir("study", 0)


def reserve(r):
    return {"verification_reserve_s": r}


def test_a_GENUINE_predecessor_VALIDATES_with_its_FULL_measured_time(segment0):
    """A8.5 item 4: timing.json's m4 - m0 plus its verification run -- not the child time alone."""
    assert CM._validate_segment("study", 0) == FULL_S0 == sum(b - a for a, b in zip(MARKS, MARKS[1:])) + 100.0
    w = json.loads((segment0 / CM.WATCHDOG_FILE).read_text())
    assert w["elapsed_s"] == 270_000.0 != FULL_S0
    assert w["record_sha256"] == CM._file_sha(str(segment0 / "record.json")) and len(w["games"]) == 4


def _rehash_record(d):
    w = json.loads((d / CM.WATCHDOG_FILE).read_text())
    w["record_sha256"] = CM._file_sha(str(d / "record.json"))
    (d / CM.WATCHDOG_FILE).write_text(json.dumps(w))


def _edit_record(d, f):
    rec = json.loads((d / "record.json").read_text())
    f(rec)
    (d / "record.json").write_text(json.dumps(rec))
    _rehash_record(d)


def _edit_json(path, f):
    obj = json.loads(path.read_text())
    f(obj)
    path.write_text(json.dumps(obj))


BREAKS = {
    "the record is {}": (lambda d: (d / "record.json").write_text("{}"), "not the one its watchdog hashed"),
    "the record is {}, rehashed": (lambda d: ((d / "record.json").write_text("{}"), _rehash_record(d)),
                                   "not a COMPLETE record of the committed plan"),
    "a game file altered": (lambda d: (d / CM.GAMES_DIR / "s1-B.json").write_text("{}"),
                            "s1-B.json is not the file its record hashed"),
    "a game file missing": (lambda d: (d / CM.GAMES_DIR / "s1-A.json").unlink(), "a game file is not committed"),
    "a game names another file": (lambda d: _edit_record(d, lambda r: r["games"][0].update(file="x.json")),
                                  "names the file"),
    "the games are not the plan's": (lambda d: _edit_record(d, lambda r: r["games"].reverse()),
                                     "not the plan's pairs and arms"),
    "the claim is for segment 3": (lambda d: _edit_json(d / CM.CLAIM_FILE, lambda c: c.update(segment=3)),
                                   "claim is for another plan, role or segment"),
    "the watchdog says STOP": (lambda d: _edit_json(d / CM.WATCHDOG_FILE, lambda w: w.update(status="STOP")),
                               "its watchdog says STOP"),
    "the journal altered": (lambda d: (d / "log.jsonl").write_text("{}\\n"), "journal is not the one"),
    "the plan changed after the run": (lambda d: CM.plan_path("study", 0).write_text(json.dumps(
        {"mode": "study", "pairs": [{"pair_id": "s2", "seeds": {"red": 13, "black": 14}}]})),
        "claim is for another plan"),
    "the watchdog is not JSON": (lambda d: (d / CM.WATCHDOG_FILE).write_text("not json"), "malformed"),
    "the record is missing": (lambda d: (d / "record.json").unlink(), "not committed"),
    "the replay checks are missing": (lambda d: _edit_record(d, lambda r: r.update(pairs=[])),
                                      "do not name every planned pair"),
    **{f"elapsed {name}": ((lambda v: lambda d: _edit_json(d / CM.WATCHDOG_FILE, lambda w: w.update(elapsed_s=v)))(v),
                           "not a finite, nonnegative number")
       for name, v in (("NaN", float("nan")), ("negative", -3600.0), ("infinite", float("inf")),
                       ("a string", "100"), ("a boolean", True))},
    "elapsed missing": (lambda d: _edit_json(d / CM.WATCHDOG_FILE, lambda w: w.pop("elapsed_s")),
                        "not a finite, nonnegative number"),
}


@pytest.mark.parametrize("name", BREAKS)
def test_a_MALFORMED_or_MISMATCHED_predecessor_is_REFUSED(segment0, name):
    breakage, why = BREAKS[name]
    breakage(segment0)
    with pytest.raises(CM.MatchRefused, match=why):
        CM._segment_limit("study", 1, reserve(0.0))


def test_the_PARENT_reads_only_the_OPEN_games_pair_and_arm(tmp_path):
    d = tmp_path
    assert CM._open_game(d) is None
    lines = [{"step": "model_load", "observation": {}},
             {"step": "game_start", "observation": {"pair_id": "p1", "arm": "A"}}]
    (d / "log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines) + '{"step": "ga')
    assert CM._open_game(d) == "p1-A", "a line still being written is skipped"
    lines.append({"step": "game", "observation": {"pair_id": "p1", "arm": "A"}})
    (d / "log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert CM._open_game(d) is None


def test_the_80_h_BACKSTOP_admits_an_EXACT_fit_and_refuses_ONE_second_more(segment0):
    """A8.5 item 4: used 270,115 + this process 85 + 4 h 14,400 + reserve = 288,000 exactly. Every
    component is >= 3 s: omitting any one admits the over-run; counting any twice refuses the fit."""
    assert CM._segment_limit("study", 1, reserve(3_400.0)) == CM.SEGMENT_DEADLINE_S
    with pytest.raises(CM.MatchRefused, match="could pass the 80 h ceiling"):
        CM._segment_limit("study", 1, reserve(3_401.0))


def test_a_pair_where_BOTH_fired_is_checked_up_to_the_EARLIER_override(played):
    a, b = copy.deepcopy(played["p-fired"]["A"]), played["p-fired"]["B"]
    w = next(x for x in a["witnesses"] if x["ply"] == 140)
    w["override"] = True                          # A "fires" later, at 140; B fired at 129
    pc = CM.check_pair(a, b)
    assert pc["first_override"] == {"A": 140, "B": 129} and pc["agree_through"] == 129


def test_the_ALARM_is_disarmed_after_every_game_however_it_ends():
    import signal
    t = [0.0]

    def clock():
        t[0] += 1.0
        return t[0]
    before = signal.getsignal(signal.SIGALRM)
    with pytest.raises(CM.MatchTimeout):
        CM.play_game(fake_make(), FIRED, "A", limit_s=20, clock=clock)
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0.0 and signal.getsignal(signal.SIGALRM) is before


def test_the_NAMES_are_the_frozen_ones():
    root = pathlib.Path(R4.EVIDENCE_ROOT)
    assert CM.out_dir("pilot", 0) == root / "tactical-candidate-pilot"
    assert CM.out_dir("study", 3) == root / "tactical-candidate-study-s3"
    assert CM.plan_path("study", 3) == root / "tactical-candidate-plans" / "study-s3.json"
    assert CM.MODULE == "scripts.GPU.alphazero.candidate_match_runner"


# ─────────────────────────────── review: complete writes, safe identifiers ─────────────────────

GAME = {"pair_id": "w1", "arm": "A", "digest": "d", "moves": [[1, 2]] * 40,
        "terminal": {"reason": "cap", "ply": 280}}


def test_a_SHORT_write_is_continued_to_the_last_byte(tmp_path, monkeypatch):
    calls = []

    def seven(fd, data):
        calls.append(len(data))
        return os.write(fd, bytes(data[:7]))
    monkeypatch.setattr(CM, "_write", seven)
    summary = CM._save_game(tmp_path, GAME)
    raw = (tmp_path / "w1-A.json").read_bytes()
    assert raw == (json.dumps(GAME, indent=1, sort_keys=True) + "\n").encode() and len(calls) > 1
    assert summary["sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("first", [0, 7], ids=["at once", "after 7 bytes"])
def test_ZERO_progress_is_a_recorded_failure_never_a_loop(tmp_path, monkeypatch, first):
    n = [0]

    def stalls(fd, data):
        n[0] += 1
        if n[0] > 50:
            pytest.fail("looped on a write that made no progress")
        return os.write(fd, bytes(data[:first])) if n[0] == 1 and first else 0
    monkeypatch.setattr(CM, "_write", stalls)
    with pytest.raises(CM.MatchStop, match=f"no progress at byte {first} of"):
        CM._save_game(tmp_path, GAME)


def test_a_writer_that_OVERSTATES_its_progress_is_caught_by_the_size(tmp_path, monkeypatch):
    monkeypatch.setattr(CM, "_write", lambda fd, data: (os.write(fd, bytes(data[:7])), len(data))[1])
    with pytest.raises(CM.MatchStop, match="holds 7 bytes"):
        CM._save_game(tmp_path, GAME)


def test_a_failed_write_inside_a_RUN_is_a_recorded_STOP(tmp_path, monkeypatch):
    d = claimed(tmp_path)                         # the claim is written by the same path (A8.5)
    monkeypatch.setattr(CM, "_write", lambda fd, data: 0)
    rec = CM._run({"mode": "fixture", "pairs": [FIRED]}, CONFIG, IDENT, object, d,
                  mode="fixture", builder=fake_builder())
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == ("STOP", "game:p-fired-A", "MatchStop")
    assert "no progress" in f["reason"] and rec["games"] == []


UNSAFE = ["../outside", "a/b", ".hidden", "-lead", "x" * 65, "has space", "", "ok.dot"]


@pytest.mark.parametrize("pid", UNSAFE)
def test_an_UNSAFE_pair_id_is_refused_by_the_plan(pid):
    with pytest.raises(CM.MatchRefused, match="unique and SAFE"):
        CM.check_plan({"mode": "fixture", "pairs": [{"pair_id": pid, "seeds": {"red": -1, "black": -2}}]},
                      mode="fixture")


def test_a_game_file_stays_DIRECTLY_inside_games_whatever_the_plan_said(tmp_path):
    gdir = tmp_path / "games"
    gdir.mkdir()
    for pid in ("../outside", "a/b", ".."):
        with pytest.raises(CM.MatchStop, match="would not lie directly inside"):
            CM._save_game(gdir, {**GAME, "pair_id": pid})
    with pytest.raises(CM.MatchStop, match="would not lie directly inside"):
        CM._game_file(gdir, "ok", "C", ".json")
    assert sorted(os.listdir(tmp_path)) == ["games"] and os.listdir(gdir) == []
    assert CM._game_file(gdir, "p-1", "B", ".partial.jsonl") == gdir / "p-1-B.partial.jsonl"


def test_a_RUN_with_an_unsafe_id_refuses_BEFORE_loading_anything(tmp_path):
    d = claimed(tmp_path)
    plan = {"mode": "fixture", "pairs": [{"pair_id": "../outside", "seeds": {"red": -1, "black": -2}}]}
    with pytest.raises(CM.MatchRefused, match="SAFE"):
        CM._run(plan, CONFIG, IDENT, lambda: pytest.fail("loaded"), d, mode="fixture", builder=fake_builder())
    assert sorted(os.listdir(d)) == [CM.CLAIM_FILE] and not (tmp_path / "outside-A.json").exists()


# ─────────────────────────────── A8.4: the qualification guard ───────────────────────────────

def test_the_GUARD_binds_the_REAL_committed_qualification(binding):
    """Reads the committed evidence for real: seven pinned files, the corpus and call list now."""
    assert binding == {"commit": "d815461eb8daa7f462dc8c492316717a0b3045b4", "files": CM.QUALIFICATION_FILES,
                       "corpus_sha256": "1f040a9208fb3049de4fbf6e73884ad0fafa067fed3d9c275a4abbedf149179c",
                       "call_list_sha256": "dae1e0f7b6723e64e99ae9154cc474fd3ff989f380396ba08b13f8e1c1d3fe7b"}
    assert sorted(CM.QUALIFICATION_FILES) == sorted(os.listdir(QUAL_DIR)) and CM.QUALIFICATION_CALLS == 109
    assert CM._file_sha(Q.__file__) == CM.QUALIFICATION_MODULE_SHA256


@pytest.fixture
def qual_copy(tmp_path, monkeypatch):
    """A COPY of the qualification evidence, 'committed' as present (durability is tested on its own)."""
    d = tmp_path / Q.OUT_NAME
    shutil.copytree(QUAL_DIR, d)
    monkeypatch.setattr(Q, "out_dir", lambda: d)
    monkeypatch.setattr(TQ, "out_dir", lambda: CORPUS_DIR)
    monkeypatch.setattr(F, "check_durable", lambda paths, _fixture: None)
    return d


def _qedit(name, f):
    """Edit one evidence file; re-hash it into the watchdog where the watchdog hashes it."""
    def edit(d):
        p = d / name
        if name.endswith(".jsonl"):
            p.write_text(f(p.read_text()))
        else:
            _edit_json(p, f)
        keys = {"record.json": ("record_sha256",), "observations.jsonl": ("observations_sha256",)}.get(name, ())
        if keys or name == "log.jsonl":
            w = json.loads((d / CM.WATCHDOG_FILE).read_text())
            for k in keys:
                w[k] = CM._file_sha(str(p))
            if name == "log.jsonl":
                w["journal"]["sha256"] = CM._file_sha(str(p))
            (d / CM.WATCHDOG_FILE).write_text(json.dumps(w))
    return edit


def _lines(f):
    return lambda text: "".join(x + "\n" for x in f(text.splitlines()))


#: content breakages -- each file then RE-PINNED, so the content check itself must refuse
GUARD_CONTENT = {
    "watchdog STOP": (_qedit("watchdog.json", lambda w: w.update(status="STOP")), "watchdog is not COMPLETE"),
    "watchdog NOT_QUALIFIED": (_qedit("watchdog.json", lambda w: w.update(verdict="NOT_QUALIFIED")),
                               "watchdog is not COMPLETE"),
    "watchdog child rc 1": (_qedit("watchdog.json", lambda w: w.update(child_rc=1)), "watchdog is not COMPLETE"),
    "watchdog killed": (_qedit("watchdog.json", lambda w: w.update(signal="SIGKILL")), "watchdog is not COMPLETE"),
    "watchdog hashed another record": (_qedit("watchdog.json", lambda w: w.update(record_sha256="0" * 64)),
                                       "watchdog hashed other files"),
    "watchdog hashed other observations": (_qedit("watchdog.json", lambda w: w.update(observations_sha256="0" * 64)),
                                           "watchdog hashed other files"),
    "watchdog hashed another journal": (_qedit("watchdog.json", lambda w: w["journal"].update(sha256="0" * 64)),
                                        "watchdog hashed other files"),
    "record STOP": (_qedit("record.json", lambda r: r.update(status="STOP")), "record is not COMPLETE"),
    "record NOT_QUALIFIED": (_qedit("record.json", lambda r: r.update(verdict="NOT_QUALIFIED")),
                             "record is not COMPLETE"),
    "record with a failure": (_qedit("record.json", lambda r: r.update(failure={"reason": "x"})),
                              "record is not COMPLETE"),
    "record with a cleanup failure": (_qedit("record.json", lambda r: r.update(cleanup_failures=[{}])),
                                      "record is not COMPLETE"),
    "record 108 calls": (_qedit("record.json", lambda r: r.update(calls=108)), "all 109 calls"),
    "record 108 observed": (_qedit("record.json", lambda r: r.update(observed=108)), "all 109 calls"),
    "an observation dropped": (_qedit("observations.jsonl", _lines(lambda ls: ls[:-1])), "all 109 calls"),
    "two observations swapped": (_qedit("observations.jsonl", _lines(lambda ls: [ls[1], ls[0], *ls[2:]])),
                                 "not the call list, in order"),
    "claim on another corpus": (_qedit("claim.json", lambda c: c.update(corpus_sha256="0" * 64)), "another corpus"),
    "record on another corpus": (_qedit("record.json", lambda r: r.update(corpus_sha256="0" * 64)), "another corpus"),
    "claim with another call list": (_qedit("claim.json", lambda c: c.update(plan_sha256="0" * 64)),
                                     "another call list"),
    "record with another call list": (_qedit("record.json", lambda r: r.update(plan_sha256="0" * 64)),
                                      "another call list"),
    "the watchdog is not JSON": (lambda d: (d / CM.WATCHDOG_FILE).write_text("not json"), "malformed"),
}


@pytest.mark.parametrize("name", GUARD_CONTENT)
def test_the_GUARD_refuses_evidence_whose_CONTENT_does_not_qualify(qual_copy, monkeypatch, name):
    assert CM.qualification_binding()["corpus_sha256"].startswith("1f040a92")    # the copy binds unaltered
    edit, why = GUARD_CONTENT[name]
    edit(qual_copy)
    for n in CM.QUALIFICATION_FILES:                                               # re-pinned: content alone
        monkeypatch.setitem(CM.QUALIFICATION_FILES, n, CM._file_sha(str(qual_copy / n)))
    with pytest.raises(CM.MatchRefused, match=why):
        CM.qualification_binding()


@pytest.mark.parametrize("name", sorted(CM.QUALIFICATION_FILES))
def test_the_GUARD_refuses_an_evidence_file_that_is_not_the_PINNED_one(qual_copy, name):
    with open(qual_copy / name, "ab") as fh:
        fh.write(b" ")
    with pytest.raises(CM.MatchRefused, match=f"{name} is not the pinned file"):
        CM.qualification_binding()


@pytest.mark.parametrize("edit", [lambda d: (d / "extra.txt").write_text(""),
                                  lambda d: (d / "child.stderr.txt").unlink()], ids=["one more", "one fewer"])
def test_the_GUARD_wants_EXACTLY_the_seven_files(qual_copy, edit):
    edit(qual_copy)
    with pytest.raises(CM.MatchRefused, match="exactly the seven evidence files"):
        CM.qualification_binding()


def test_the_GUARD_wants_the_evidence_COMMITTED_and_UNMODIFIED(tmp_path, monkeypatch):
    d = tmp_path / Q.OUT_NAME                     # a byte-identical copy that git does not hold
    shutil.copytree(QUAL_DIR, d)
    monkeypatch.setattr(Q, "out_dir", lambda: d)
    with pytest.raises(CM.MatchRefused, match="not committed and unmodified"):
        CM.qualification_binding()


def test_the_GUARD_wants_the_QUALIFIED_call_list_definition_and_an_ACCEPTED_corpus(qual_copy, monkeypatch):
    monkeypatch.setattr(CM, "QUALIFICATION_MODULE_SHA256", "0" * 64)
    with pytest.raises(CM.MatchRefused, match="not the qualified call-list definition"):
        CM.qualification_binding()
    monkeypatch.undo()
    real_evidence(monkeypatch)

    def refused():
        raise TQ.CorpusRefused("probe")
    monkeypatch.setattr(TQ, "committed_corpus", refused)
    with pytest.raises(CM.MatchRefused, match="corpus is not accepted now"):
        CM.qualification_binding()


def test_the_PLAN_must_carry_EXACTLY_the_guards_binding(binding, monkeypatch):
    real_evidence(monkeypatch)
    assert CM._bound_plan({"qualification": binding}) == binding
    for plan in ({}, {"qualification": {**binding, "call_list_sha256": "0" * 64}},
                 {"qualification": {**binding, "files": {**binding["files"], "log.jsonl": "0" * 64}}}):
        with pytest.raises(CM.MatchRefused, match="not bound to the accepted qualification"):
            CM._bound_plan(plan)


def test_PUBLIC_ENTRY_a_GUARD_failure_refuses_BEFORE_the_claim(open_gate, monkeypatch):
    monkeypatch.setattr(CM, "_watch_games", lambda *a, **k: pytest.fail("a child was started"))
    monkeypatch.setattr(CM, "QUALIFICATION_MODULE_SHA256", "0" * 64)
    with pytest.raises(CM.MatchRefused, match="call-list definition"):
        CM.run("pilot", 0)
    assert not CM.out_dir("pilot", 0).exists()


def test_PUBLIC_ENTRY_an_UNBOUND_plan_refuses_BEFORE_the_claim(open_gate, monkeypatch):
    commit_plan(monkeypatch, PILOT_PLAN)
    monkeypatch.setattr(CM, "_watch_games", lambda *a, **k: pytest.fail("a child was started"))
    with pytest.raises(CM.MatchRefused, match="not bound to the accepted qualification"):
        CM.run("pilot", 0)
    assert not CM.out_dir("pilot", 0).exists()


def _child_claim(binding, monkeypatch, **over):
    out = CM.out_dir("pilot", 0)
    plan = {**PILOT_PLAN, "qualification": binding}
    CM._create(out)
    _c, nonce = CM._claim(out, {"plan_sha256": hashlib.sha256(json.dumps(plan).encode()).hexdigest(),
                                "role": "pilot", "segment": 0, "qualification": binding, **over})
    monkeypatch.setenv(CM.CLAIM_ENV, nonce)


def test_the_CHILD_checks_the_GUARD_before_the_LOAD(open_gate, binding, monkeypatch):
    _child_claim(binding, monkeypatch)
    monkeypatch.setattr(R4, "load_production_evaluator", lambda identity: pytest.fail("loaded"))
    monkeypatch.setattr(CM, "QUALIFICATION_MODULE_SHA256", "0" * 64)
    with pytest.raises(CM.MatchRefused, match="call-list definition"):
        CM.run_claimed("pilot", 0)
    assert sorted(os.listdir(CM.out_dir("pilot", 0))) == [CM.CLAIM_FILE]


def test_the_CHILD_refuses_a_claim_bound_to_ANOTHER_qualification(open_gate, binding, monkeypatch):
    _child_claim(binding, monkeypatch, qualification={**binding, "commit": "0" * 40})
    monkeypatch.setattr(R4, "load_production_evaluator", lambda identity: pytest.fail("loaded"))
    with pytest.raises(CM.MatchRefused, match="bound to another qualification"):
        CM.run_claimed("pilot", 0)


# ─────────────────────────────── A8.5: parent recording and timing ───────────────────────────

def in_process(monkeypatch):
    """The child IN THIS PROCESS through its real CLI, with the parent's nonce (as the real watch)."""
    def watch(argv, *, limit_s, game_s, d, env, cwd, stdout_path, stderr_path):
        monkeypatch.setenv(CM.CLAIM_ENV, env[CM.CLAIM_ENV])
        with open(stdout_path, "x") as so, open(stderr_path, "x") as se, \
                contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            rc = CM.main(list(argv[3:]))
        return {"child_pid": os.getpid(), "limit_s": limit_s, "terminated": False, "cause": None,
                "signal": None, "rc": rc, "elapsed_s": 1.0, "survivors_killed": False,
                "game_limit_s": game_s, "game_breach": None}
    monkeypatch.setattr(CM, "_watch_games", watch)


def test_TIMING_tiles_the_parents_run_from_m0_to_m4_without_overlap(open_gate, monkeypatch):
    """A8.5 item 3: m1 at entry; m2 after the claim, before launch; m3 after the child, before the
    watchdog; m4 after watchdog.json; then timing.json."""
    in_process(monkeypatch)
    out, seen = CM.out_dir("pilot", 0), []

    def clock():
        seen.append(sorted(os.listdir(out)) if out.exists() else None)
        return 10.0 * len(seen)
    monkeypatch.setattr(CM, "_T_START", 5.0)
    monkeypatch.setattr(CM, "_monotonic", clock)
    res = CM.run("pilot", 0)
    assert res["status"] == "COMPLETE" and len(seen) == 4
    m1, m2, m3, m4 = seen
    assert m1 is None, "m1: before the destination exists"
    assert m2 == [CM.CLAIM_FILE], "m2: after the claim, before the child's streams"
    assert "record.json" in m3 and CM.WATCHDOG_FILE not in m3, "m3: after the child, before settlement"
    assert CM.WATCHDOG_FILE in m4 and "timing.json" not in m4, "m4: after watchdog.json, before timing"
    t = json.loads((out / "timing.json").read_text())
    assert {k: t[k] for k in ("startup_s", "pre_launch_s", "child_s", "settlement_s")} == \
        {"startup_s": 5.0, "pre_launch_s": 10.0, "child_s": 10.0, "settlement_s": 10.0}


def test_TIMING_is_written_even_when_the_child_leaves_no_record(open_gate):
    res = CM.run("pilot", 0)                      # the real child reads the CLOSED on-disk gate
    out = CM.out_dir("pilot", 0)
    assert res["status"] == "EXITED_WITHOUT_RECORD"
    t = json.loads((out / "timing.json").read_text())
    assert all(CM._nonneg(t[k]) for k in ("startup_s", "pre_launch_s", "child_s", "settlement_s"))


@pytest.mark.parametrize("exc,status", [(RuntimeError, "CLAIM_FAILED"), (KeyboardInterrupt, "INTERRUPTED")])
def test_a_CLAIM_failure_or_interruption_still_leaves_watchdog_json(open_gate, monkeypatch, exc, status):
    def boom(fd, data):
        raise exc("claim probe")
    monkeypatch.setattr(CM, "_write", boom)
    monkeypatch.setattr(CM, "_watch_games", lambda *a, **k: pytest.fail("a child was started"))
    with pytest.raises(exc, match="claim probe"):
        CM.run("pilot", 0)
    out = CM.out_dir("pilot", 0)
    w = json.loads((out / CM.WATCHDOG_FILE).read_text())
    assert (w["status"], w["stage"], w["failure"]["exception_type"], w["record_present"]) == \
        (status, "claim", exc.__name__, False)
    assert sorted(os.listdir(out)) == [CM.CLAIM_FILE, CM.WATCHDOG_FILE]


@pytest.fixture
def sigint_default():
    """Python's own SIGINT handler: a background job (`cmd &`) inherits SIGINT IGNORED."""
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous)


def test_an_INTERRUPT_as_the_destination_is_created_is_recorded_IN_IT(open_gate, monkeypatch, sigint_default):
    import threading
    real = os.mkdir

    def mkdir_then_sigint(path, *a, **k):
        real(path, *a, **k)
        if pathlib.Path(path) == CM.out_dir("pilot", 0):
            signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
    monkeypatch.setattr(os, "mkdir", mkdir_then_sigint)
    monkeypatch.setattr(CM, "_watch_games", lambda *a, **k: pytest.fail("a child was started"))
    with pytest.raises(KeyboardInterrupt):
        CM.run("pilot", 0)
    w = json.loads((CM.out_dir("pilot", 0) / CM.WATCHDOG_FILE).read_text())
    assert (w["status"], w["stage"], w["claim_present"]) == ("INTERRUPTED", "claim", False)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_a_destination_created_by_ANOTHER_invocation_is_never_written(open_gate, monkeypatch):
    out, real = CM.out_dir("pilot", 0), CM._bound_plan

    def racing(plan):                             # another invocation creates it after the first check
        b = real(plan)
        out.mkdir()
        (out / "theirs").write_text("x")
        return b
    monkeypatch.setattr(CM, "_bound_plan", racing)
    with pytest.raises(CM.MatchRefused, match="occupied"):
        CM.run("pilot", 0)
    assert sorted(os.listdir(out)) == ["theirs"]


@pytest.mark.parametrize("where", ["watch", "settle"])
@pytest.mark.parametrize("exc,status", [(RuntimeError, "SETTLE_FAILED"), (KeyboardInterrupt, "INTERRUPTED")])
def test_a_WATCH_or_SETTLEMENT_failure_still_leaves_watchdog_json(open_gate, monkeypatch, where, exc, status):
    def boom(*a, **k):
        raise exc(f"{where} probe")
    if where == "watch":
        monkeypatch.setattr(CM, "_watch_games", boom)
    else:
        in_process(monkeypatch)
        monkeypatch.setattr(CC, "_journal", boom)
    with pytest.raises(exc, match=f"{where} probe"):
        CM.run("pilot", 0)
    w = json.loads((CM.out_dir("pilot", 0) / CM.WATCHDOG_FILE).read_text())
    assert (w["status"], w["stage"], w["failure"]["exception_type"], w["claim_present"]) == \
        (status, "settle", exc.__name__, True)


# ─────────────────────────────── A8.5: the child's cleanup ───────────────────────────────

def _failing_close(monkeypatch, cls, exc):
    real = cls.close

    def close(self):
        real(self)
        raise exc("close probe")
    monkeypatch.setattr(cls, "close", close)


class Breaks(Fake):                               # a STOP mid-game, while its partial file is open
    def __call__(self, state):
        if state.ply == 5:
            raise RuntimeError("mid-game probe")
        return super().__call__(state)


def _run_raising(exc, d, **kw):
    with pytest.raises(exc) if exc is KeyboardInterrupt else contextlib.nullcontext():
        CM._run({"mode": "fixture", "pairs": [FIRED]}, CONFIG, IDENT, object, d, mode="fixture", **kw)
    return json.loads((d / "record.json").read_text())


@pytest.mark.parametrize("exc", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("handle,cls", [("partial", CM._Partial), ("journal", CC._Log)], ids=["partial", "journal"])
def test_a_CLEANUP_failure_after_a_STOP_is_recorded_and_the_STOP_kept(tmp_path, monkeypatch, handle, cls, exc):
    d = claimed(tmp_path)
    _failing_close(monkeypatch, cls, exc)
    rec = _run_raising(exc, d, builder=fake_builder(agent=Breaks))
    assert rec["status"] == ("INTERRUPTED" if exc is KeyboardInterrupt else "STOP")
    assert (rec["failure"]["failing_step"], rec["failure"]["reason"]) == ("game:p-fired-A", "mid-game probe")
    assert rec["cleanup_failures"] == [{"handle": handle, "exception_type": exc.__name__, "reason": "close probe"}]
    dec = (d / CM.GAMES_DIR / "p-fired-A.partial.jsonl").read_text().splitlines()
    assert [json.loads(x)["ply"] for x in dec] == list(range(5)), "the completed decisions are kept"


@pytest.mark.parametrize("exc", [RuntimeError, KeyboardInterrupt])
def test_a_JOURNAL_close_failure_after_EVERY_game_is_recorded_never_COMPLETE(tmp_path, monkeypatch, exc):
    d = claimed(tmp_path)
    _failing_close(monkeypatch, CC._Log, exc)
    rec = _run_raising(exc, d, builder=fake_builder())
    assert rec["status"] == ("INTERRUPTED" if exc is KeyboardInterrupt else "STOP")
    assert (rec["failure"]["failing_step"], rec["failure"]["exception_type"]) == ("close:journal", exc.__name__)
    assert len(rec["games"]) == 2 and rec["cleanup_failures"] == []


def test_a_PARTIAL_close_failure_in_a_game_is_a_recorded_STOP(tmp_path, monkeypatch):
    d = claimed(tmp_path)
    _failing_close(monkeypatch, CM._Partial, RuntimeError)
    rec = _run_raising(RuntimeError, d, builder=fake_builder())
    assert (rec["status"], rec["failure"]["failing_step"], rec["cleanup_failures"]) == ("STOP", "game:p-fired-A", [])


def test_a_COMPLETE_run_records_no_cleanup_failure(complete):
    assert complete[1]["cleanup_failures"] == []


# ─────────────────────────────── A8.5 item 4: predecessor timing and verification ─────────────

def _ver(f):
    """Edit the verification record AND re-bind its marker, so the edit reaches its own check."""
    return lambda d: (_edit_json(CM.verification_path("study", 0), f), publish_marker(CM.verification_path("study", 0)))


def _vmarker(f):
    return lambda d: _edit_json(CM.verification_path("study", 0).parent / "study-s0.published.json", f)


def _timing(f):
    return lambda d: _edit_json(d / "timing.json", f)


BREAKS_A8 = {
    "timing.json missing": (lambda d: (d / "timing.json").unlink(), "timing.json is not committed"),
    "timing.json not JSON": (lambda d: (d / "timing.json").write_text("not json"), "malformed"),
    **{f"timing {k} NaN": (_timing(lambda t, k=k: t.update({k: float("nan")})), "not a finite, nonnegative")
       for k in ("startup_s", "pre_launch_s", "child_s", "settlement_s")},
    **{f"timing settlement {name}": (_timing(lambda t, v=v: t.update(settlement_s=v)), "not a finite, nonnegative")
       for name, v in (("negative", -1.0), ("infinite", float("inf")), ("a string", "5"), ("a boolean", True))},
    "timing startup missing": (_timing(lambda t: t.pop("startup_s")), "not a finite, nonnegative"),
    "verification missing": (lambda d: CM.verification_path("study", 0).unlink(),
                             "verification record is not committed"),
    "verification not JSON": (lambda d: (CM.verification_path("study", 0).write_text("x"),
                                         publish_marker(CM.verification_path("study", 0))), "malformed"),
    "verification never published": (
        lambda d: (CM.verification_path("study", 0).parent / "study-s0.published.json").unlink(), "never published"),
    "verification publication failed": (
        lambda d: (CM.verification_path("study", 0).parent / "study-s0.publication_failed.json").write_text("{}"),
        "publication failed"),
    "verification marker not PUBLISHED": (_vmarker(lambda m: m.update(status="X")), "not PUBLISHED"),
    "verification marker bound to another record": (_vmarker(lambda m: m.update(record_sha256="0" * 64)),
                                                     "not the one published"),
    "verification record edited after publication": (
        lambda d: _edit_json(CM.verification_path("study", 0), lambda v: v.update(extra=1)), "not the one published"),
    "verification for segment 3": (_ver(lambda v: v.update(segment=3)), "for another segment"),
    "verification for the pilot": (_ver(lambda v: v.update(role="pilot")), "for another segment"),
    "verification of another watchdog": (_ver(lambda v: v.update(watchdog_sha256="0" * 64)), "verified other files"),
    "verification of another record": (_ver(lambda v: v.update(record_sha256="0" * 64)), "verified other files"),
    "verification of another timing": (_ver(lambda v: v.update(timing_sha256="0" * 64)), "verified other files"),
    "timing.json changed after its finalization": (
        lambda d: (d / "timing.json").write_text(json.dumps(json.loads((d / "timing.json").read_text()))),
        "finalization bound other files"),
    "timing.json replaced and RE-FINALIZED after its verification": (
        lambda d: ((d / "timing.json").write_text(json.dumps(json.loads((d / "timing.json").read_text()))),
                   _edit_json(d / "finalized.json", lambda f: f.update(
                       timing_sha256=CM._file_sha(str(d / "timing.json"))))),
        "verified other files"),
    "finalized.json missing": (lambda d: (d / "finalized.json").unlink(), "never finalized"),
    "a late failure recorded": (lambda d: (d / "finalization_failed.json").write_text("{}"),
                                "finalization failed"),
    "a late failure recorded, finalized.json also present": (
        lambda d: (d / "finalization_failed.json").write_text(json.dumps({"stage": "finalize"})),
        "finalization failed"),
    "finalized.json not FINALIZED": (lambda d: _edit_json(d / "finalized.json", lambda f: f.update(status="X")),
                                     "not FINALIZED"),
    "finalized.json bound another watchdog": (
        lambda d: _edit_json(d / "finalized.json", lambda f: f.update(watchdog_sha256="0" * 64)),
        "finalization bound other files"),
    "finalized.json not JSON": (lambda d: (d / "finalized.json").write_text("{"), "malformed"),
    "verification NOT_VERIFIED": (_ver(lambda v: v.update(verdict="NOT_VERIFIED")), "record says 'NOT_VERIFIED'"),
    "verification of 3 game files": (_ver(lambda v: v.update(game_files=3)), "counts 3 game files, not 2"),
    "verification game files a string": (_ver(lambda v: v.update(game_files="2")), "game files, not 2"),
    "verification t_fixed NaN": (_ver(lambda v: v.update(t_fixed_s=float("nan"))), "not a finite, nonnegative"),
    "verification t_games negative": (_ver(lambda v: v.update(t_games_s=-1.0)), "not a finite, nonnegative"),
    "verification t_games missing": (_ver(lambda v: v.pop("t_games_s")), "not a finite, nonnegative"),
}


@pytest.mark.parametrize("name", BREAKS_A8)
def test_a_predecessor_with_BAD_timing_or_verification_is_REFUSED(segment0, name):
    breakage, why = BREAKS_A8[name]
    breakage(segment0)
    with pytest.raises(CM.MatchRefused, match=why):
        CM._segment_limit("study", 1, reserve(0.0))


@pytest.mark.parametrize("bad", [None, float("nan"), -1.0, float("inf"), "0", True], ids=str)
def test_a_study_plan_needs_a_VALID_verification_reserve(segment0, bad):
    plan = {} if bad is None else reserve(bad)
    with pytest.raises(CM.MatchRefused, match="verification_reserve_s"):
        CM._segment_limit("study", 1, plan)


def test_the_PILOT_has_no_predecessor_and_no_reserve():
    assert CM._segment_limit("pilot", 0, {}) == CM.SEGMENT_DEADLINE_S


def test_the_BACKSTOP_is_not_described_as_a_hard_guarantee():
    doc = CM._segment_limit.__doc__
    assert "NOT a hard end-to-end 80-hour guarantee" in doc and "planning estimate" in doc


# ─────────────────────────────── A8.6: the amendment stays inside its allowlist ───────────────

QUALIFIED_R1 = "a53e7ac857f045a281af85d7dd268445e2419e1c71fd2a0458092a5c2c7e21b6"
A8_CHANGED = {"_claim", "run", "run_claimed", "_run", "main", "_validate_segment", "_segment_limit"}
A8_ADDED = {"_T_START", "QUALIFICATION_COMMIT", "QUALIFICATION_FILES", "QUALIFICATION_MODULE_SHA256",
            "QUALIFICATION_CALLS", "qualification_binding", "_bound_plan", "_create", "_sigint_held",
            "_record_parent_failure", "_timing_path", "_write_timing", "_read_timing", "verification_path",
            "_read_verification", "_nonneg", "_finalize", "_write_finalized", "_read_finalization",
            "_finalized_path", "_finalization_failed_path"}


def _top_nodes(src):
    import ast
    out = {}
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            key = n.name
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            t = n.targets if isinstance(n, ast.Assign) else [n.target]
            names = sorted(e.id for x in t for e in (x.elts if isinstance(x, ast.Tuple) else [x]))
            key = names[0] if len(names) == 1 else tuple(names)
        elif isinstance(n, ast.Expr):
            key = "<docstring>"
        else:
            key = "<" + ast.dump(n) + ">"
        out[key] = n
    return out


def test_the_R1_AMENDMENT_stays_inside_the_A8_6_allowlist():
    """A preview of the binding record's first check -- it supports, never replaces, the review."""
    import ast
    old = subprocess.run(["git", "show", f"{CM.QUALIFICATION_COMMIT}:scripts/GPU/alphazero/candidate_match_runner.py"],
                         cwd=ROOT, capture_output=True, check=True).stdout
    assert hashlib.sha256(old).hexdigest() == QUALIFIED_R1
    a, b = _top_nodes(old), _top_nodes(pathlib.Path(CM.__file__).read_bytes())
    assert a.keys() <= b.keys(), "nothing qualified was removed"
    for key in a:
        if key not in A8_CHANGED:
            assert ast.dump(a[key]) == ast.dump(b[key]), f"{key} changed outside the allowlist"
    for key in b.keys() - a.keys():
        assert key in A8_ADDED or (isinstance(key, str) and key.startswith("<ImportFrom(module='time'")), key

    def calls(fn, name):
        return [ast.dump(c) for c in ast.walk(fn) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == name]
    for name in ("production_builder", "play_game"):
        assert calls(a["_run"], name) == calls(b["_run"], name) != [], name


# ─────────────────────────────── A8.5 item 5: finalization, and late failures recorded ──────────

class Trip:
    """One-shot fault injection into the parent's LATE steps, chosen by what is being written:
    'timing' (startup_s) or 'finalized' (FINALIZED). `when` is before writing, after the complete
    bytes, the file's sync, the link, or the directory's sync."""

    def __init__(self, monkeypatch, out, kind, when, exc):
        self.out, self.kind, self.when, self.exc, self.armed, self.fired = out, kind, when, exc, False, False
        real_all, real_fsync, real_dir, real_link = CM._write_all, os.fsync, CM._fsync_dir, os.link
        marker = b'"startup_s"' if kind == "timing" else b'"FINALIZED"'

        def write_all(fd, raw):
            mine = marker in bytes(raw)
            if mine and when == "before writing":
                self.fire()
            real_all(fd, raw)
            if mine and when == "after complete bytes":
                self.fire()
            self.armed = mine and when == "file sync"

        def fsync(fd):
            if self.armed:
                self.armed = False
                self.fire()
            real_fsync(fd)

        def fsync_dir(d):
            timing, final = CM._timing_path(out).exists(), (out / "finalized.json").exists()
            if when == "directory sync" and ((kind == "timing" and timing and not final) or
                                             (kind == "finalized" and final)):
                self.fire()
            real_dir(d)

        def link(src, dst, *a, **k):
            if when == "link":
                self.fire()
            return real_link(src, dst, *a, **k)
        monkeypatch.setattr(CM, "_write_all", write_all)
        monkeypatch.setattr(os, "fsync", fsync)
        monkeypatch.setattr(CM, "_fsync_dir", fsync_dir)
        monkeypatch.setattr(os, "link", link)

    def fire(self):
        if not self.fired:
            self.fired = True
            raise self.exc(f"late probe: {self.kind} {self.when}")


LATE = [("timing", w) for w in ("before writing", "after complete bytes", "file sync", "directory sync")] + \
       [("finalized", w) for w in ("before writing", "after complete bytes", "file sync", "link", "directory sync")]


@pytest.mark.parametrize("exc,status", [(RuntimeError, "FINALIZE_FAILED"), (KeyboardInterrupt, "INTERRUPTED")])
@pytest.mark.parametrize("kind,when", LATE, ids=[f"{k}-{w}" for k, w in LATE])
def test_a_LATE_failure_is_RECORDED_the_evidence_KEPT_and_every_consumer_REJECTS(open_gate, monkeypatch,
                                                                                  kind, when, exc, status):
    in_process(monkeypatch)
    out, kept = CM.out_dir("pilot", 0), {}
    real_settle = CM._settle

    def settle(d, watch):                         # the evidence as it stood when the watchdog was written
        res = real_settle(d, watch)
        kept.update({p.name: p.read_bytes() for p in d.rglob("*") if p.is_file()})
        return res
    monkeypatch.setattr(CM, "_settle", settle)
    trip = Trip(monkeypatch, out, kind, when, exc)
    with pytest.raises(exc, match=f"late probe: {kind} {when}"):
        CM.run("pilot", 0)
    assert trip.fired
    f = json.loads((out / "finalization_failed.json").read_text())
    assert (f["stage"], f["status"], f["failure"]["exception_type"]) == ("finalize", status, exc.__name__)
    for name, raw in kept.items():                # watchdog, record, journal, games: byte for byte
        assert (out.rglob(name).__next__()).read_bytes() == raw, name
    w = json.loads((out / CM.WATCHDOG_FILE).read_text())
    assert w["status"] == "COMPLETE", "the watchdog alone still says COMPLETE"
    monkeypatch.undo()                            # the consumer, on the evidence as left
    monkeypatch.setattr(F, "check_durable", lambda paths, _fixture: None)
    timing = CM._timing_path(out)
    with pytest.raises(CM.MatchRefused, match="finalization failed"):
        CM._read_finalization(out, lambda why: CM.MatchRefused(why), CM._file_sha(str(out / CM.WATCHDOG_FILE)),
                              CM._file_sha(str(timing)) if timing.exists() else None)


def test_the_REPRODUCED_case_valid_timing_COMPLETE_watchdog_failed_directory_sync(open_gate, monkeypatch):
    """The review's reproduction: timing.json holds valid JSON, the watchdog says COMPLETE -- and the
    segment is still rejected, because its finalization failed and is recorded."""
    in_process(monkeypatch)
    out = CM.out_dir("pilot", 0)
    Trip(monkeypatch, out, "timing", "directory sync", OSError)
    with pytest.raises(OSError):
        CM.run("pilot", 0)
    t = json.loads(CM._timing_path(out).read_text())
    assert all(CM._nonneg(t[k]) for k in ("startup_s", "pre_launch_s", "child_s", "settlement_s"))
    assert json.loads((out / CM.WATCHDOG_FILE).read_text())["status"] == "COMPLETE"
    assert not (out / "finalized.json").exists() and (out / "finalization_failed.json").exists()


def test_a_FINALIZED_segment_carries_its_bound_marker_and_no_failure_record(open_gate, monkeypatch):
    in_process(monkeypatch)
    CM.run("pilot", 0)
    out = CM.out_dir("pilot", 0)
    f = json.loads((out / "finalized.json").read_text())
    assert f == {"status": "FINALIZED", "watchdog_sha256": CM._file_sha(str(out / CM.WATCHDOG_FILE)),
                 "timing_sha256": CM._file_sha(str(CM._timing_path(out)))}
    assert not (out / "finalization_failed.json").exists() and not (out / ".finalized.json.partial").exists()


def test_a_failing_WATCHDOG_write_is_recorded_LATE_never_silently(open_gate, monkeypatch):
    """_settle's own write fails after creating watchdog.json: recorded in the failure file."""
    in_process(monkeypatch)
    real_dump = json.dump

    def dump(obj, fh, *a, **k):
        if getattr(fh, "name", "").endswith(CM.WATCHDOG_FILE) and "child_streams" in obj:
            fh.write("{")
            raise OSError("watchdog write probe")
        return real_dump(obj, fh, *a, **k)
    monkeypatch.setattr(json, "dump", dump)
    with pytest.raises(OSError, match="watchdog write probe"):
        CM.run("pilot", 0)
    out = CM.out_dir("pilot", 0)
    assert (out / CM.WATCHDOG_FILE).read_text() == "{"
    f = json.loads((out / "finalization_failed.json").read_text())
    assert (f["stage"], f["status"]) == ("settle", "SETTLE_FAILED")
