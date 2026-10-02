"""H5 incumbent defence mechanism study (EXPLORATORY) -- card §1-§8 with §9 A1.

SYNTHETIC ONLY: no model is loaded and the real incumbent makes no decision. The search
is the REAL frozen MCTS path (400 simulations, argmax readout) driven by a stub
evaluator; the classifier is also tested on hand-built trees whose answers are known.
The five fixtures are read from the committed frozen set, read-only. Messages fixed.
"""
import contextlib
import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pytest

from scripts.GPU.alphazero import eval_readout as RO
from scripts.GPU.alphazero import h4_runner as R4
from scripts.GPU.alphazero import h5_cap_completion as CC
from scripts.GPU.alphazero import h5_incumbent_defence as M
from scripts.GPU.alphazero import h5_incumbent_defence_authorization as AUTH
from scripts.GPU.alphazero import twixtbot_g3_reference as G3
from scripts.GPU.alphazero.mcts import MCTS, MCTSNode, encode_move as enc

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = R4.frozen_argmax_config()
IDENT = M._identity(CONFIG)
FIX = M.positions(M._frozen())
BY_ID = {p["id"]: p for p in FIX}
#: root ply -> the recorded move, and one ply deeper T1j's recorded reply at E1-E3
STEER = {37: (1, 2), 38: (16, 4), 55: (1, 4), 56: (11, 7), 79: (11, 10), 80: (14, 8),
         41: (16, 9), 127: (12, 5)}


class Steered:
    """Real MCTS, no model. At a steered ply 0.9 of the prior goes to that ply's move;
    elsewhere the prior is uniform over the legal moves. Values are small and fixed."""

    def __init__(self, steer=STEER, fail_on_call=None, fail_with=KeyboardInterrupt):
        self._g3_reference, self._g3_sha1 = IDENT["reference"], IDENT["reference_sha1"]
        self.steer, self.calls, self._state = steer, 0, None
        self.fail_on_call, self.fail_with = fail_on_call, fail_with

    def build_input_tensor(self, state):
        self._state = state
        return state.to_tensor()

    def infer(self, boards, rows, cols, mask, active_size):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise self.fail_with()
        s = self._state
        priors = mask.astype(np.float32).copy()
        priors /= priors.sum(axis=1, keepdims=True)
        target = self.steer.get(s.ply)
        if target is not None:
            hit = (rows[0] == target[0]) & (cols[0] == target[1]) & (mask[0] > 0)
            if hit.any():
                priors[0] *= 0.1
                priors[0][hit] += 0.9
        values = np.full(mask.shape[0], ((s.ply * 7919 + len(s.pegs)) % 13 - 6) / 100.0,
                         dtype=np.float32)
        return priors, values


def build(ev=None):
    return M.study_builder(CONFIG, ev or Steered(), M.tasks_for(IDENT))


# ─────────────────────────────── the gate ───────────────────────────────

def test_the_gate_is_false_as_published():
    assert AUTH.H5_INCUMBENT_DEFENCE_AUTHORIZED is False


@pytest.mark.parametrize("entry", ["run", "run_claimed"])
def test_every_public_entry_refuses_with_the_gate_shut_BEFORE_anything(entry, monkeypatch,
                                                                        tmp_path):
    for owner, name in ((M, "_frozen"), (M, "_identity"), (M, "_claim"), (CC, "_watch"),
                        (R4, "load_production_evaluator"), (os, "mkdir")):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail("touched before the gate"))
    with pytest.raises(M.DefenceUnauthorized):
        getattr(M, entry)(str(tmp_path / "x-EXPLORATORY"))


def test_the_CLI_exits_5_as_a_fresh_subprocess_and_creates_nothing(tmp_path):
    out = tmp_path / "x-EXPLORATORY"
    r = subprocess.run([sys.executable, "-m", M.MODULE, "run", "--out-dir", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == M.EXIT_UNAUTHORIZED == 5, r.stderr
    assert not out.exists(), "the closed CLI created the evidence directory"


# ─────────────────────────────── the fixtures (card §6, §9 A1.6) ───────────────────────────────

def test_the_five_FIXTURES_come_from_the_committed_frozen_set_with_their_pins():
    assert [(p["id"], p["expect"], p["recorded"]) for p in FIX] == [
        ("E1", "miss", (1, 2)), ("E2", "miss", (1, 4)), ("E3", "miss", (11, 10)),
        ("CD1", "block", (16, 9)), ("CD2", "block", (12, 5))]
    assert {p["id"]: p["saving"] for p in FIX} == {
        "E1": [(16, 9), (17, 9)], "E2": [(12, 5)], "E3": [(16, 6), (17, 7)],
        "CD1": [(16, 9), (17, 9)], "CD2": [(12, 4), (12, 5), (13, 5)]}


def test_a_wrong_CONTROL_PIN_is_refused(monkeypatch):
    monkeypatch.setitem(M.CONTROL_SAVING, "CD2", ((12, 5), (13, 5)))
    with pytest.raises(M.DefenceRefused, match="CD2: the saving moves are"):
        M.positions(M._frozen())


# ─────────────────────────────── the classifier, on known trees (§9 A1.5) ───────────────────

def tree(priors, children, expanded=()):
    """A root with `priors` {(r,c): p} and children {(r,c): (visits, value_sum)}."""
    root = MCTSNode(state=None, priors={enc(*m): p for m, p in priors.items()})
    root.priors_raw, root.nn_value = root.priors, 0.25
    for m, (v, vs) in children.items():
        root.children[enc(*m)] = MCTSNode(state=None, parent=root, move=enc(*m),
                                          visit_count=v, value_sum=vs)
    for m in expanded:
        root.children[enc(*m)].priors = {}
    return root


KNOWN = tree({(1, 1): 0.5, (2, 2): 0.2, (3, 3): 0.1, (4, 4): 0.0, (5, 5): 0.2},
             {(1, 1): (10, 4.0), (2, 2): (3, -1.5), (5, 5): (0, 0.0)})


@pytest.mark.parametrize("move,want", [((9, 9), "A"), ((3, 3), "B"), ((5, 5), "B"),
                                       ((4, 4), "B0"), ((2, 2), "C"), ((1, 1), "D")])
def test_the_CLASSIFIER_on_a_known_tree(move, want):
    assert M.category(KNOWN, enc(*move), enc(1, 1)) == want


def test_the_REFUTATION_classes_R0_R1_R2():
    m, w = (1, 1), (7, 7)
    assert M.refutation(tree({m: 1.0}, {m: (1, 0.0)}), enc(*m), enc(*w))["category"] == "R0"
    t = tree({m: 1.0}, {m: (5, 0.0)}, expanded=[m])
    t.children[enc(*m)].priors = {enc(*w): 0.01}
    assert M.refutation(t, enc(*m), enc(*w))["category"] == "R1", "W unexpanded must be R1"
    t.children[enc(*m)].children[enc(*w)] = MCTSNode(state=None, move=enc(*w), visit_count=0)
    assert M.refutation(t, enc(*m), enc(*w))["category"] == "R1", "W with 0 visits must be R1"
    t.children[enc(*m)].children[enc(*w)].visit_count = 2
    t.children[enc(*m)].children[enc(*w)].value_sum = -2.0
    r = M.refutation(t, enc(*m), enc(*w))
    assert r["category"] == "R2"
    assert (r["W"]["q_node"], r["W"]["q_root_mover"]) == (-1.0, -1.0), \
        "a grandchild's mean is the root mover's own perspective"


def test_PERSPECTIVES_are_labelled_by_depth_and_an_unvisited_mean_is_undefined():
    s = M.node_stat(KNOWN, enc(1, 1), 1)
    assert (s["q_node"], s["q_root_mover"]) == (0.4, -0.4), "a root child flips sign"
    u = M.node_stat(KNOWN, enc(3, 3), 1)
    assert (u["q_node"], u["q_root_mover"], u["q_note"]) == \
        (None, None, "undefined: never visited")


def test_the_CORRECTED_R2_ARITHMETIC_against_the_real_backup():
    """§9 A1.2: a terminal W under m adds +1 to m (T1j's perspective) and -1 at the root."""
    root, m, w = MCTSNode(state=None), MCTSNode(state=None), MCTSNode(state=None)
    MCTS(evaluator=object())._backup([root, m, w], -1.0)   # -1 to the incumbent to move at W
    assert (w.value_sum, m.value_sum, root.value_sum) == (-1.0, 1.0, -1.0)


def test_the_TIE_SET_and_the_readout_resolve_ties_by_row_col():
    counts = {(3, 4): 7, (2, 9): 7, (1, 1): 2}
    assert M.tie_set(counts) == [[2, 9], [3, 4]]
    assert RO._argmax(counts) == (2, 9), "the readout's tie-break is not (row, col)"


# ─────────────────────────────── the pass-through (§9 A1.1) ───────────────────────────────

class Fake:
    def __init__(self, result=None, boom=None):
        self.result, self.boom, self.seen = result, boom, None

    def search_with_root(self, *a, **k):
        self.seen = (a, k)
        if self.boom is not None:
            raise self.boom
        return self.result


def test_the_PASS_THROUGH_forwards_arguments_returns_THE_object_and_re_raises():
    sentinel = (object(), 0.5, object())
    f = Fake(result=sentinel)
    box = M.retain_search(f)
    assert f.search_with_root("state", add_noise=False) is sentinel
    assert f.seen == (("state",), {"add_noise": False}) and box["results"] == [sentinel]
    boom = RuntimeError("search failed")
    g = Fake(boom=boom)
    gbox = M.retain_search(g)
    with pytest.raises(RuntimeError) as e:
        g.search_with_root("state")
    assert e.value is boom and gbox["results"] == [], "the exception was changed or swallowed"


def test_the_PASS_THROUGH_keeps_THE_objects_the_readout_used(monkeypatch):
    seen, real = {}, RO.select
    monkeypatch.setattr(RO, "select", lambda counts, *a, **k: (
        seen.setdefault("counts", counts), real(counts, *a, **k))[1])
    d = M.decide(build(), M.CC.state_of(BY_ID["E1"]["moves"]), capture=False)
    assert d["counts"] is seen["counts"], "the retained counts are not the readout's"


def test_the_PASS_THROUGH_changes_nothing_at_400_simulations():
    state = M.CC.state_of(BY_ID["E1"]["moves"])
    plain = build()(state.to_move, False)
    move = plain(state)
    kept = M.decide(build(), state, capture=False)
    on = M.decide(build(), state, capture=True)
    assert move == kept["move"] == on["move"]
    assert (plain.mcts.rng.getstate(), plain.readout_rng.getstate()) == \
        (kept["search_rng"], kept["readout_rng"]), "the pass-through moved a stream"
    assert kept["root"].visit_count == CONFIG.mcts_sims == 400
    assert M.fingerprint(kept["root"], kept["counts"], kept["root_value"]) == \
        M.fingerprint(on["root"], on["counts"], on["root_value"])


# ─────────────────────────────── the fingerprint (§9 A1.3) ───────────────────────────────

def _mutations():
    def prior(t):
        t.priors[enc(2, 2)] += 1e-9

    def raw_prior(t):
        t.priors_raw[enc(2, 2)] += 1e-9

    def aliasing(t):
        t.priors_raw = t.priors

    def order(t):
        t.priors = dict(reversed(list(t.priors.items())))

    def value_sum(t):
        t.children[enc(2, 2)].value_sum += 1e-9

    def nn_value(t):
        t.nn_value += 1e-9

    def visits(t):
        t.children[enc(2, 2)].visit_count += 1
    return [prior, raw_prior, aliasing, order, value_sum, nn_value, visits]


def split_tree():
    """KNOWN with priors and raw priors as SEPARATE equal dicts, so each field must be
    seen on its own (in a real search they are one object)."""
    t = copy.deepcopy(KNOWN)
    t.priors_raw = dict(t.priors)
    return t


@pytest.mark.parametrize("mutate", _mutations(), ids=lambda f: f.__name__)
def test_the_FINGERPRINT_sees_every_inspected_field(mutate):
    counts = {(1, 1): 10, (2, 2): 3}
    t = split_tree()
    before = M.fingerprint(t, counts, 0.1)
    assert M.fingerprint(split_tree(), counts, 0.1) == before, "not deterministic"
    mutate(t)
    assert M.fingerprint(t, counts, 0.1) != before, f"{mutate.__name__} went unseen"


def test_the_FINGERPRINT_sees_the_counts_and_the_root_value():
    counts = {(1, 1): 10, (2, 2): 3}
    base = M.fingerprint(KNOWN, counts, 0.1)
    assert M.fingerprint(KNOWN, {**counts, (2, 2): 4}, 0.1) != base
    assert M.fingerprint(KNOWN, counts, 0.1 + 1e-12) != base


# ─────────────────────────────── one fixture, observed (§5, §9 A1) ───────────────────────────────

def test_OBSERVE_baseline_every_check_passes_at_400_simulations():
    row = M.observe(build(), CONFIG, BY_ID["E1"])
    assert row["move"] == [1, 2] and row["expect"] == "miss"
    assert {"move", "search stream", "readout stream", "counts", "tie set",
            "tree fingerprint off == on", "readings agree"} <= set(row["checks"])
    rd = row["reading"]
    assert [s["move"] for s in rd["saving"]] == [[16, 9], [17, 9]]
    assert all(s["category"] in ("A", "B", "B0", "C") for s in rd["saving"])
    assert rd["refutation"][0]["category"] in ("R0", "R1", "R2")
    assert rd["candidates"] == len(rd["root_priors"]) > 0


def _bad_readers():
    def prior(root, *a):
        k = next(iter(root.priors))
        root.priors[k] += 1e-9

    def value_sum(root, *a):
        next(c for c in root.children.values() if c.visit_count).value_sum += 1e-9

    def nn_value(root, *a):
        root.nn_value += 1e-9

    def visits(root, *a):
        next(iter(root.children.values())).visit_count += 1

    def search_draw(root, *a):
        LAST[-1].mcts.rng.random()

    def readout_draw(root, *a):
        LAST[-1].readout_rng.random()
    return [(prior, "reader left the tree unchanged"), (value_sum, "reader left the tree unchanged"),
            (nn_value, "reader left the tree unchanged"), (visits, "reader left the tree unchanged"),
            (search_draw, "reader drew from no stream"), (readout_draw, "reader drew from no stream")]


LAST = []


@pytest.mark.parametrize("bad,check", _bad_readers(), ids=lambda x: getattr(x, "__name__", ""))
def test_OBSERVE_STOPS_on_a_reader_that_changes_or_draws_anything(monkeypatch, bad, check):
    real_read, real_decide = M.read, M.decide

    def keep_agent(*a, **k):
        d = real_decide(*a, **k)
        LAST.append(d["agent"])
        return d
    monkeypatch.setattr(M, "decide", keep_agent)
    monkeypatch.setattr(M, "read", lambda root, *a: (bad(root, *a), real_read(root, *a))[1])
    with pytest.raises(M.DefenceStop, match=f"E1: preservation check '{check}"):
        M.observe(build(), CONFIG, BY_ID["E1"])


def test_OBSERVE_STOPS_when_the_move_is_not_the_recorded_one():
    with pytest.raises(M.DefenceStop, match="E1: preservation check 'move'"):
        M.observe(build(), CONFIG, {**BY_ID["E1"], "recorded": (16, 9)})


def test_OBSERVE_STOPS_when_capture_draws_from_a_stream(monkeypatch):
    real = G3.SeededReferenceAgent.__call__

    def drawing(self, state):
        move = real(self, state)
        if self.capture:
            self.readout_rng.random()
        return move
    monkeypatch.setattr(G3.SeededReferenceAgent, "__call__", drawing)
    with pytest.raises(M.DefenceStop, match="'readout stream'"):
        M.observe(build(), CONFIG, BY_ID["E1"])


def test_OBSERVE_STOPS_when_capture_changes_the_tree(monkeypatch):
    real = G3.SeededReferenceAgent.__call__

    def touching(self, state):
        move = real(self, state)
        if self.capture:
            self.last_capture["root"].nn_value += 1e-9
        return move
    monkeypatch.setattr(G3.SeededReferenceAgent, "__call__", touching)
    with pytest.raises(M.DefenceStop, match="'tree fingerprint off == on'"):
        M.observe(build(), CONFIG, BY_ID["E1"])


def test_DECIDE_STOPS_when_capture_reports_other_objects_than_the_search_returned(monkeypatch):
    real = G3.SeededReferenceAgent.__call__

    def copying(self, state):
        move = real(self, state)
        if self.capture:
            self.last_capture["counts"] = dict(self.last_capture["counts"])
        return move
    monkeypatch.setattr(G3.SeededReferenceAgent, "__call__", copying)
    with pytest.raises(M.DefenceStop, match="capture reports objects other"):
        M.decide(build(), M.CC.state_of(BY_ID["E1"]["moves"]), capture=True)


def test_DECIDE_STOPS_when_one_decision_runs_two_searches(monkeypatch):
    real = G3.SeededReferenceAgent.__call__

    def twice(self, state):
        self.mcts.search_with_root(state, add_noise=False)
        return real(self, state)
    monkeypatch.setattr(G3.SeededReferenceAgent, "__call__", twice)
    with pytest.raises(M.DefenceStop, match="2 searches for one decision"):
        M.decide(build(), M.CC.state_of(BY_ID["E1"]["moves"]), capture=False)


def test_OBSERVE_STOPS_on_a_configuration_that_is_not_THE_object():
    twin = copy.deepcopy(CONFIG)                 # equal, but not THE object
    with pytest.raises(M.DefenceStop, match="'configuration \\(off\\)'"):
        M.observe(M.study_builder(twin, Steered(), M.tasks_for(IDENT)), CONFIG, BY_ID["E1"])


def test_the_STUDY_BUILDER_is_the_production_construction_plus_capture():
    ev, tasks = Steered(), M.tasks_for(IDENT)
    prod = R4.production_incumbent_build(CONFIG)(tasks["black"], ev)
    study = M.study_builder(CONFIG, ev, tasks)("black", False)
    assert (type(prod), prod.seed, prod.colour, prod.capture) == \
        (type(study), study.seed, study.colour, study.capture) == \
        (G3.SeededReferenceAgent, -1, "black", False)
    assert prod.config is study.config is CONFIG and prod.mcts.evaluator is study.mcts.evaluator
    state = M.CC.state_of(BY_ID["E1"]["moves"])
    assert prod(state) == study(state)
    assert (prod.mcts.rng.getstate(), prod.readout_rng.getstate()) == \
        (study.mcts.rng.getstate(), study.readout_rng.getstate())


# ─────────────────────────────── the child's run (card §8, §9 A1.7) ───────────────────────────────

def claimed(tmp_path):
    d = tmp_path / "x-EXPLORATORY"
    M._claim(d)
    return d


def log_steps(d):
    return [json.loads(line)["step"] for line in (d / "log.jsonl").read_text().splitlines()]


def test_RUN_all_five_fixtures_COMPLETE_each_persisted_before_the_next(tmp_path, monkeypatch):
    d, seen, real = claimed(tmp_path), [], M.observe
    monkeypatch.setattr(M, "observe", lambda b, c, p: (
        seen.append(log_steps(d).count("position")), real(b, c, p))[1])
    rec = M._run(FIX, CONFIG, IDENT, Steered, d)
    assert (rec["status"], rec["failure"], rec["interpretable"]) == ("COMPLETE", None, True)
    assert [r["position"] for r in rec["rows"]] == list(M.POSITION_IDS)
    assert seen == [0, 1, 2, 3, 4], "a row was not on disk before the next fixture"
    assert log_steps(d) == ["model_load"] + ["position"] * 5
    for r in rec["rows"]:
        cats = [s["category"] for s in r["reading"]["saving"]]
        if r["expect"] == "block":
            assert "D" in cats and r["reading"]["refutation"] is None
        else:
            assert "D" not in cats and r["reading"]["refutation"]


def test_RUN_a_MODEL_LOAD_failure_is_a_STOP_naming_it(tmp_path):
    def loader():
        raise FileNotFoundError("checkpoint not found")
    rec = M._run(FIX, CONFIG, IDENT, loader, claimed(tmp_path))
    assert (rec["status"], rec["failure"]["failing_step"], rec["failure"]["exception_type"]) \
        == ("STOP", "model_load", "FileNotFoundError")
    assert rec["rows"] == [] and not rec["interpretable"]


def test_RUN_a_PRESERVATION_failure_STOPs_naming_the_position_and_the_check(tmp_path):
    fixtures = [FIX[0], {**FIX[1], "recorded": (12, 5)}] + FIX[2:]
    rec = M._run(fixtures, CONFIG, IDENT, Steered, claimed(tmp_path))
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"]) == \
        ("STOP", "observe:E2", "DefenceStop")
    assert "E2: preservation check 'move'" in f["reason"] and len(rec["rows"]) == 1


def test_RUN_an_INTERRUPTION_is_recorded_and_re_raised(tmp_path):
    d = claimed(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        M._run(FIX, CONFIG, IDENT, lambda: Steered(fail_on_call=2000), d)
    rec = json.loads((d / "record.json").read_text())
    assert rec["status"] == "INTERRUPTED" and rec["failure"]["failing_step"].startswith("observe:")
    assert log_steps(d)[: len(rec["rows"]) + 1] == ["model_load"] + ["position"] * len(rec["rows"])


def test_RUN_an_INCOMPLETE_fixture_set_names_what_is_missing_and_invents_no_exception(tmp_path):
    rec = M._run(FIX[:4], CONFIG, IDENT, Steered, claimed(tmp_path))
    f = rec["failure"]
    assert (rec["status"], f["failing_step"], f["exception_type"], f["missing"]) == \
        ("INCOMPLETE", "completion_check", None, ["CD2"])
    assert not rec["interpretable"]


def test_RUN_the_DEADLINE_stops_before_the_first_fixture(tmp_path):
    t = [0.0]

    def loader():
        t[0] += 10_000.0
        return Steered()
    rec = M._run(FIX, CONFIG, IDENT, loader, claimed(tmp_path), clock=lambda: t[0])
    assert (rec["status"], rec["failure"]["failing_step"]) == ("STOP", "observe:E1")
    assert "deadline passed before observe:E1" in rec["failure"]["reason"]


def test_RUN_refuses_an_unclaimed_or_consumed_directory(tmp_path):
    with pytest.raises(M.DefenceRefused, match="unconsumed claim"):
        M._run(FIX, CONFIG, IDENT, lambda: pytest.fail("loaded"), tmp_path / "nothing")
    d = claimed(tmp_path)
    (d / "log.jsonl").write_text("")
    with pytest.raises(M.DefenceRefused, match="unconsumed claim"):
        M._run(FIX, CONFIG, IDENT, lambda: pytest.fail("loaded"), d)


def test_the_CHILD_takes_only_an_UNCONSUMED_claim_with_ITS_nonce(tmp_path, monkeypatch):
    d = tmp_path / "x-EXPLORATORY"
    claim, nonce = M._claim(d)
    assert nonce not in (d / M.CLAIM_FILE).read_text(), "the nonce itself was written down"
    monkeypatch.setenv(M.CLAIM_ENV, "0" * 32)
    with pytest.raises(M.DefenceRefused, match="nonce does not match"):
        M._accept_claim(d)
    monkeypatch.setenv(M.CLAIM_ENV, nonce)
    for name in M.CHILD_STREAMS.values():
        (d / name).write_text("")
    assert M._accept_claim(d) == claim
    (d / "stranger.txt").write_text("")
    with pytest.raises(M.DefenceRefused, match="unconsumed claim"):
        M._accept_claim(d)


# ─────────────────────────────── the public entry, real state ───────────────────────────────

@pytest.fixture
def open_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(AUTH, "H5_INCUMBENT_DEFENCE_AUTHORIZED", True)
    monkeypatch.setattr(R4, "EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setattr(R4, "load_production_evaluator", lambda identity: Steered())
    return tmp_path


def inproc(monkeypatch):
    """The child run IN THIS PROCESS through its real CLI, its streams to the parent's
    files, with the nonce the parent hands it."""
    def watch(argv, *, limit_s, env, cwd, stdout_path, stderr_path):
        assert list(argv[1:4]) == ["-m", M.MODULE, "run-claimed"], argv
        monkeypatch.setenv(M.CLAIM_ENV, env[M.CLAIM_ENV])
        t0 = time.monotonic()
        with open(stdout_path, "x") as so, open(stderr_path, "x") as se, \
                contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            rc = M.main(list(argv[3:]))
        return {"child_pid": os.getpid(), "limit_s": limit_s, "terminated": False,
                "cause": None, "signal": None, "rc": rc, "elapsed_s": time.monotonic() - t0,
                "survivors_killed": False}
    monkeypatch.setattr(CC, "_watch", watch)


def test_PUBLIC_ENTRY_runs_the_study_in_the_real_state(open_gate, monkeypatch):
    inproc(monkeypatch)
    out = open_gate / "x-EXPLORATORY"
    res = M.run(str(out))
    rec = json.loads((out / "record.json").read_text())
    assert (res["status"], res["record_status"], res["child_rc"]) == \
        ("COMPLETE", "COMPLETE", 0), rec["failure"]
    assert res == json.loads((out / M.WATCHDOG_FILE).read_text())
    assert [r["position"] for r in rec["rows"]] == list(M.POSITION_IDS)
    assert rec["incumbent_identity"] == IDENT and rec["seed"] == -1
    assert res["journal"]["lines"] == 6
    assert json.loads((out / M.CHILD_STREAMS["stdout"]).read_text()) == {"status": "COMPLETE"}
    assert M.HARD_LIMIT_S - M.MARGIN_S - 60 < rec["deadline_s"] <= M.HARD_LIMIT_S - M.MARGIN_S


def test_PUBLIC_ENTRY_refuses_an_occupied_destination_before_anything(open_gate, monkeypatch):
    out = open_gate / "x-EXPLORATORY"
    out.mkdir()
    monkeypatch.setattr(CC, "_watch", lambda *a, **k: pytest.fail("a child was started"))
    with pytest.raises(M.DefenceRefused, match="occupied"):
        M.run(str(out))
    assert list(out.iterdir()) == []


def test_PUBLIC_ENTRY_the_REAL_child_invocation_reads_the_CLOSED_on_disk_gate(open_gate):
    out = open_gate / "x-EXPLORATORY"
    res = M.run(str(out))
    assert (res["status"], res["child_exit"]) == ("EXITED_WITHOUT_RECORD", "exit status 5")
    assert "UNAUTHORIZED" in (out / M.CHILD_STREAMS["stderr"]).read_text()
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [M.CLAIM_FILE, M.WATCHDOG_FILE, *M.CHILD_STREAMS.values()])


HANG = r'''
import json, os, signal, sys, time
d = sys.argv[1]
open(os.path.join(d, "log.jsonl"), "x").write('{"step": "model_load"}\n')
json.dump({"status": "COMPLETE"}, open(os.path.join(d, "record.json"), "x"))
sys.stdout.write("OUT before\n")
signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())
time.sleep(60)
'''


def test_PUBLIC_ENTRY_a_hung_child_is_TERMINATED_never_COMPLETE(open_gate, monkeypatch):
    out = open_gate / "x-EXPLORATORY"
    monkeypatch.setattr(M, "_child_argv", lambda o: [sys.executable, "-c", HANG, str(o)])
    monkeypatch.setattr(M, "HARD_LIMIT_S", 3.0)
    t0 = time.monotonic()
    res = M.run(str(out))
    assert time.monotonic() - t0 < 20, "the parent did not enforce its limit"
    assert (res["status"], res["record_status"], res["child_exit"]) == \
        ("TERMINATED", "COMPLETE", "killed by signal SIGKILL")
    assert json.loads((out / "record.json").read_text()) == {"status": "COMPLETE"}
    assert (out / M.CHILD_STREAMS["stdout"]).read_text() == "OUT before\n"
    raw = (out / "log.jsonl").read_bytes()
    assert res["journal"] == {"file": "log.jsonl", "lines": 1,
                              "sha256": hashlib.sha256(raw).hexdigest()}
