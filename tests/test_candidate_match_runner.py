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
    for owner, name in ((CM, "_committed_plan"), (CM, "_segment_limit"), (CM, "_claim"),
                        (CM, "_run"), (CM, "_watch_games"), (os, "mkdir"), (R4, "load_production_evaluator")):
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


@pytest.fixture
def open_gate(monkeypatch, tmp_path):
    """The gate open, the evidence root in tmp, the plan 'committed', the seeds 'registered',
    the stand-in agents for speed. Everything else -- claim, child entry, record -- is real."""
    monkeypatch.setattr(AUTH, "CANDIDATE_MATCH_AUTHORIZED", True)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setattr(R4, "seed_registry_faults", lambda tasks: [])
    monkeypatch.setattr(R4, "load_production_evaluator", lambda identity: object())
    monkeypatch.setattr(CM, "production_builder", lambda config, ev, ident: fake_make())
    raw = json.dumps(PILOT_PLAN).encode()
    monkeypatch.setattr(CM, "_committed_plan", lambda role, segment: (
        CM.check_plan(PILOT_PLAN, mode=role) or PILOT_PLAN,
        __import__("hashlib").sha256(raw).hexdigest()))
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
        CM._claim(d, {"plan_sha256": CM._file_sha(str(CM.plan_path("study", 0))), "role": "study",
                      "segment": 0})
        assert CM._run(STUDY_PLAN, CONFIG, IDENT, object, d, mode="study",
                       builder=fake_builder())["status"] == "COMPLETE"
        CM._settle(d, {"terminated": False, "cause": None, "limit_s": 14_400.0, "elapsed_s": 75 * 3600.0,
                       "child_pid": 1, "rc": 0, "signal": None, "survivors_killed": False,
                       "game_limit_s": 900.0, "game_breach": None})
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
    return CM.out_dir("study", 0)


def test_a_GENUINE_predecessor_VALIDATES_and_counts_toward_the_80_h_ceiling(segment0):
    assert CM._validate_segment("study", 0) == 75 * 3600.0
    assert CM._segment_limit("study", 1) == CM.SEGMENT_DEADLINE_S, "75 h + 4 h <= 80 h"
    w = json.loads((segment0 / CM.WATCHDOG_FILE).read_text())
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
}


@pytest.mark.parametrize("name", BREAKS)
def test_a_MALFORMED_or_MISMATCHED_predecessor_is_REFUSED(segment0, name):
    breakage, why = BREAKS[name]
    breakage(segment0)
    with pytest.raises(CM.MatchRefused, match=why):
        CM._segment_limit("study", 1)


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


def test_the_80_h_CEILING_counts_the_validated_segments(segment0):
    _edit_json(segment0 / CM.WATCHDOG_FILE, lambda w: w.update(elapsed_s=77 * 3600.0))
    with pytest.raises(CM.MatchRefused, match="ceiling"):
        CM._segment_limit("study", 1)


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
