"""The H-TB SCREEN (plan `docs/superpowers/2026-11-06-htb-screen-plan.md`; card §9.1, §9.3), built only on the user's
review (2026-11-06): `htb_screen run screen`, its wrapper, its proposed ceiling and its seed copies.

CPU only. No test loads the student or calib020_0001, touches the GPU, opens a real gate, runs a seed mode on the real
repository, writes the registry or plays a real game:
* admission runs on the REAL committed fit records (read-only) and on synthetic ones;
* the screen runs through the real record boundary with GPU-free fakes and a stubbed game;
* the whole-network round trip (save as the fit saved, load as the match loads) runs on a RANDOM network on the CPU;
* the wrapper and seed copies are checked as exact derivations of the cycle-2 screen's reviewed files.

EVERY TEST IS ISOLATED, WHATEVER THE GATE SAYS (#74): the autouse fixture makes the real seams and the real record
boundary unreachable."""
import hashlib
import importlib.util
import inspect
import json
import math
import os
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest

from scripts.GPU.alphazero import combination_policy_pilot as CPP
from scripts.GPU.alphazero import eval_runner as ER
from scripts.GPU.alphazero import htb_pilot as H
from scripts.GPU.alphazero import htb_screen as HS
from scripts.GPU.alphazero import htb_screen_authorization as GATE
from scripts.GPU.alphazero import mirror_halving_combination as K
from scripts.GPU.alphazero import root_sequential_halving as RSH
from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator
from tests import _combination_fakes as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import htb_screen_copies_spec as SPEC  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = 777000                                   # a fake plan's base: not a registry seed
PLAN_OK = {"kind": HS.PLAN_KIND, "base": BASE, "n_slots": 64}
CLEAR = {"ok": True, "n_procs": 800, "hits": [], "n_hits": 0, "exempt": [], "error": None}
BOINC = {"ok": False, "n_procs": 801, "hits": ["5635 boinc_project hsgamma_FGRP5"], "n_hits": 1, "exempt": [],
         "error": None}
_REAL_CONCLUDE = HS.conclude
_REAL_SCREEN_SEAMS = HS._real_screen_seams


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def unreachable(*_a, **_k):
        raise AssertionError("a test reached a real seam or the real record boundary")
    for name in ("_real_screen_seams", "conclude"):
        monkeypatch.setattr(HS, name, unreachable)
    monkeypatch.setenv(HS.SETTING, HS.SETTING_VALUE)
    import mlx.core as mx
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_THE_ISOLATION_IS_IN_FORCE():
    with pytest.raises(AssertionError, match="real seam"):
        HS._real_screen_seams(pathlib.Path("/x"))
    with pytest.raises(AssertionError, match="real seam"):
        HS.conclude(None, None, {})


# ─────────────────────────────── gate, pins, entry ───────────────────────────────

def test_THE_GATE_IS_A_CLOSED_SINGLE_DECLARATION_and_the_closed_gate_refuses_after_the_pins_hold(monkeypatch, capsys):
    import ast
    assert HS.GATE is GATE and HS.GATE_NAME == "HTB_SCREEN_AUTHORIZED" and GATE.HTB_SCREEN_AUTHORIZED is False
    tree = ast.parse(pathlib.Path(GATE.__file__).read_text())
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert len(body) == 1 and body[0].targets[0].id == HS.GATE_NAME
    monkeypatch.setattr(CPP, "_mlx_loaded", lambda: False)
    assert HS.main(["run", "screen", "--checkpoint", "/x"]) == HS.EXIT_REFUSED
    assert "HTB_SCREEN_AUTHORIZED is closed" in capsys.readouterr().err


def test_THE_PINS_ARE_THE_FILES_and_the_card_is_the_pilot_s_unchanged():
    assert HS.CARD == H.CARD and HS.CARD_SHA256 == H.CARD_SHA256
    for rel, sha in ((HS.PLAN_DOC, HS.PLAN_DOC_SHA256), (HS.CARD, HS.CARD_SHA256), *HS.PINNED.items()):
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == sha, rel
    assert {k: v for k, v in HS.PINNED.items() if k in CPP.PINNED} == CPP.PINNED


@pytest.mark.parametrize("env, loaded, why", [("1", False, "MLX_ENABLE_TF32 is '1'"), (None, False, "is None"),
                                              ("0", True, "MLX was imported")], ids=["tf32-1", "unset", "mlx-loaded"])
def test_THE_PREFLIGHT_REFUSES_A_WRONG_SETTING_OR_A_LOADED_MLX(monkeypatch, capsys, env, loaded, why):
    if env is None:
        monkeypatch.delenv(HS.SETTING, raising=False)
    else:
        monkeypatch.setenv(HS.SETTING, env)
    monkeypatch.setattr(CPP, "_mlx_loaded", lambda: loaded)
    monkeypatch.setattr(GATE, "HTB_SCREEN_AUTHORIZED", True)
    assert HS.main(["run", "screen", "--checkpoint", "/x"]) == HS.EXIT_REFUSED
    assert why in capsys.readouterr().err


def test_A_CHANGED_PIN_REFUSES(monkeypatch, capsys):
    monkeypatch.setattr(CPP, "_mlx_loaded", lambda: False)
    monkeypatch.setattr(HS, "PLAN_DOC_SHA256", "0" * 64)
    assert HS.main(["run", "screen", "--checkpoint", "/x"]) == HS.EXIT_REFUSED
    assert "is not the pinned bytes" in capsys.readouterr().err


def test_RUN_STAYS_ONE_SHOT_even_with_its_gate_open(monkeypatch, capsys):
    monkeypatch.setattr(CPP, "_mlx_loaded", lambda: False)
    monkeypatch.setattr(GATE, "HTB_SCREEN_AUTHORIZED", True)
    monkeypatch.setattr(HS, "EVIDENCE", H.EVIDENCE_FIT)                   # an existing destination
    assert HS.main(["run", "screen", "--checkpoint", str(ROOT / HS.CARD)]) == HS.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "exists: the run is one-shot" in err or f"{HS.STUDENT['path']} is not a file" in err


def test_USAGE_IS_EXACTLY_RUN_SCREEN_WITH_A_CHECKPOINT(capsys):
    for argv in ([], ["run", "screen"], ["run", "strength", "--checkpoint", "/x"], ["time"]):
        assert HS.main(argv) == HS.EXIT_USAGE
    assert "run screen --checkpoint" in capsys.readouterr().err


def test_IMPORTING_THE_MODULE_LOADS_NO_MLX():
    out = subprocess.run([sys.executable, "-c", "import sys; import scripts.GPU.alphazero.htb_screen; "
                          "print(any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules))"],
                         cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert out.returncode == 0 and out.stdout.strip() == "False", out.stderr


# ─────────────────────────────── admission (plan §2) ───────────────────────────────

def test_REAL_STATE_THE_COMMITTED_FIT_ADMITS_THE_SCREEN_AND_NAMES_THE_PINNED_STUDENT():
    adm = HS.fit_admission(CPP._read_record, CPP._read_outer, CPP._read_committed_bytes)
    assert adm == {f"{H.EVIDENCE_FIT}/record.json": "ELIGIBLE", "docs/superpowers/evidence/htb-fit-run/outer.json":
                   "COMPLETED"}
    fit = json.loads((ROOT / H.EVIDENCE_FIT / "record.json").read_text())
    assert fit["student"] == {"path": str(HS.STUDENT["path"]), "sha256": HS.STUDENT["sha256"],
                              "state_digest": HS.STUDENT["state_digest"]}
    assert fit["fit"]["reference_digest"] == HS.REFERENCE_DIGEST != HS.STUDENT["state_digest"]
    # RE-ANCHORED 2026-11-06 (the ONE run, gate 138ede34 -> evidence 25f4863a): the screen ran once, on this admission
    rec = json.loads((ROOT / HS.EVIDENCE / "record.json").read_text())
    assert rec["status"] == "COMPLETED" and rec["admission"] == adm and rec["student"] == fit["student"]


def _good_fit():
    fit = json.loads((ROOT / H.EVIDENCE_FIT / "record.json").read_text())
    outer = json.loads((ROOT / HS.FIT_RUN / "outer.json").read_text())
    label = (ROOT / H.EVIDENCE_LABEL / "record.json").read_bytes()
    return fit, outer, label


def _readers(fit, outer, label):
    def rec(rel):
        if fit is None:
            raise FileNotFoundError(f"{rel}/record.json")
        return fit
    return rec, (lambda rel: outer), (lambda rel: label)


@pytest.mark.parametrize("edit, why", [
    (lambda f, o: f.update(status="STOP"), "status is 'STOP'"),
    (lambda f, o: f.update(card_sha256="0" * 64), "card_sha256"),
    (lambda f, o: f.update(kind="htb-label-record/1"), "kind is"),
    (lambda f, o: f["verdict"].update(verdict="STOP"), "not ELIGIBLE"),
    (lambda f, o: f["verdict"].update(F2=False), "not ELIGIBLE"),
    (lambda f, o: f["student"].update(sha256="0" * 64), "student is not the pinned one"),
    (lambda f, o: f["student"].update(state_digest="0" * 64), "student is not the pinned one"),
    (lambda f, o: f["student"].update(path="/tmp/other.safetensors"), "student is not the pinned one"),
    (lambda f, o: f["fit"].update(reference_digest="0" * 64), "reference digest"),
    (lambda f, o: f.update(label_record_sha256="0" * 64), "label record"),
    (lambda f, o: o.update(outcome="TIMEOUT", inner_exit=124, outer_exit=124), "outer.json inner_exit"),
    (lambda f, o: o.update(survivors="123 "), "outer.json survivors"),
    (lambda f, o: o.update(gate_restored=False), "outer.json gate_restored"),
    (lambda f, o: o.update(record_by="wrapper"), "outer.json record_by"),
], ids=["status", "card", "kind", "verdict", "F2", "sha256", "digest", "path", "reference", "label", "timeout",
        "survivors", "gate", "record_by"])
def test_ADMISSION_REFUSES_ANYTHING_BUT_THE_ONE_ELIGIBLE_FIT_WITH_CONFIRMED_CLEANUP(edit, why):
    fit, outer, label = _good_fit()
    HS.fit_admission(*_readers(fit, outer, label))                        # the control: the real records admit
    edit(fit, outer)
    with pytest.raises(HS.Refused, match="does not admit") as e:
        HS.fit_admission(*_readers(fit, outer, label))
    assert why in str(e.value)


def test_ADMISSION_REFUSES_A_MISSING_FIT_RECORD():
    _fit, outer, label = _good_fit()
    with pytest.raises(HS.Refused, match="does not exist"):
        HS.fit_admission(*_readers(None, outer, label))


# ─────────────────────────────── the rule (plan §5) ───────────────────────────────

def _g(ev=1.0, sec=1.0):
    return {"evaluations_ratio": ev, "search_seconds_ratio": sec, "evaluations_within": ev <= 1.05,
            "search_seconds_within": sec <= 1.10, "guards": dict(K.GUARDS)}


@pytest.mark.parametrize("pairs, g, verdict, n_reasons", [
    ([0.5] * 32, _g(), "CONTINUE", 0),                                   # exactly 0.50 is not below it
    ([1.0] * 32, _g(), "CONTINUE", 0),
    ([0.5] * 31 + [0.0], _g(), "STOP", 1),                                # 0.484375: the nearest below
    ([0.5] * 31 + [1.0], _g(), "CONTINUE", 0),                           # 0.515625
    ([1.0] * 32, _g(ev=1.06), "STOP", 1),
    ([1.0] * 32, _g(sec=1.11), "STOP", 1),
    ([1.0] * 32, _g(ev=1.05, sec=1.10), "CONTINUE", 0),                   # the guards are inclusive
    ([0.0] * 32, _g(ev=2.0, sec=2.0), "STOP", 3),
], ids=["exactly-half", "all-wins", "just-below", "just-above", "evaluations", "seconds", "guards-at-limit", "all"])
def test_THE_RULE_IS_STOP_BELOW_ONE_HALF_OR_ON_EITHER_GUARD(pairs, g, verdict, n_reasons):
    v = HS.screen_verdict(pairs, g)
    assert v["verdict"] == verdict and len(v["stop_reasons"]) == n_reasons, v
    assert v["mean_pair_score"] == pytest.approx(sum(pairs) / 32) and v["stop_below"] == 0.50
    assert v["guards"] == {"evaluations": 1.05, "search_seconds": 1.10} and "not a strength result" in v["note"]


def test_THE_RULE_REFUSES_A_WRONG_COHORT_OR_GUARD_SET():
    with pytest.raises(ValueError, match="31 pairs"):
        HS.screen_verdict([0.5] * 31, _g())
    with pytest.raises(ValueError, match="guards"):
        HS.screen_verdict([0.5] * 32, {**_g(), "guards": {"evaluations": 2.0, "search_seconds": 1.10}})
    assert HS.SCREEN == {"pairs": 32, "stop_below": 0.50} and HS.GUARDS == {"evaluations": 1.05, "search_seconds": 1.10}


# ─────────────────────────────── the screen, through the real record boundary, on fakes ───────────────────────────────

class SideFake(F.FakeEvaluator):
    """A side with its own network object, named and carrying a state digest; different seeds give different priors."""

    def __init__(self, name, digest, seed):
        super().__init__(seed=seed)
        self.network = types.SimpleNamespace(name=name, digest=digest)


def stub_play(outcome, searches=(2, 2), secs=(0.25, 0.25), extra_candidate_calls=0, calls=None, fail_at=None):
    """A scripted game that checks the wiring it is handed, makes one averaged call per side (two underlying calls
    each) plus any extra candidate calls, reports the scripted searches at the scripted seconds (candidate, reference)
    and the winner outcome(game_idx, candidate_is_red) names."""
    def play(task, red, black, config, capture, replay_dir, classes, on_search):
        if task.game_idx == fail_at:
            raise RuntimeError(f"injected failure in game {fail_at}")
        cand_red = task.red_agent.agent_id == HS.CANDIDATE
        assert (task.game_idx % 2 == 0) == cand_red and task.seed == BASE + task.game_idx
        assert type(red) is MirrorAveragingEvaluator and type(black) is MirrorAveragingEvaluator
        cand, ref = (red, black) if cand_red else (black, red)
        assert cand.base.network.name == "student" and ref.base.network.name == "reference"
        assert classes == {HS.CANDIDATE: RSH.RootSequentialHalvingMCTS, HS.REFERENCE: RSH.RootSequentialHalvingMCTS}
        assert capture is True and (config.board_size, config.max_moves) == (24, 280)
        assert ER.agent_sims(task.red_agent, config) == ER.agent_sims(task.black_agent, config) == 200
        args = (np.array([[0, 1]], np.int32), np.array([[1, 2]], np.int32), np.ones((1, 2), np.float32), 24)
        for ev in (red, black) + (cand,) * extra_candidate_calls:
            ev.infer(np.zeros((1, 24, 24, 30), np.float32), *args)
        for is_red, n in ((True, searches[0]), (False, searches[1])):
            for _ in range(n):
                on_search(is_red, secs[0] if is_red == cand_red else secs[1], True)
        if calls is not None:
            calls.append(task.game_idx)
        who = outcome(task.game_idx, cand_red)
        winner = None if who is None else ("red" if (who == HS.CANDIDATE) == cand_red else "black")
        return ER.make_result(task, winner, "win" if winner else "state_cap", 4, config=config)
    return play


def screen_seams(play, loaded=None, **over):
    loaded = [] if loaded is None else loaded

    def side(name, digest, seed):
        def load():
            ev = SideFake(name, digest, seed)
            loaded.append(ev)
            return ev
        return load

    s = dict(versions=lambda: dict(HS.VERSIONS), read_plan=lambda rel: dict(PLAN_OK), seed_ok=lambda s: True,
             checkpoint_sha1=lambda: HS.M.CHECKPOINT_SHA1, student_sha256=lambda: HS.STUDENT["sha256"],
             load_candidate=side("student", HS.STUDENT["state_digest"], 7),
             load_reference=side("reference", HS.REFERENCE_DIGEST, 8),
             digest=lambda net: net.digest,
             require_eval=lambda ev, when: {"modules_in_training_mode": 0, "when": when}, loaded_modules=lambda: {},
             background=lambda: dict(CLEAR), play_and_build=play, between_games=lambda: None)
    s.update(over)
    return HS.ScreenSeams(**s)


def _screen(tmp_path, seams):
    return _REAL_CONCLUDE(lambda o, r, lg: HS._screen(seams, o, r, lg), tmp_path / "ev", {"fit": "ELIGIBLE"})


def _events(tmp_path):
    return [json.loads(x) for x in (tmp_path / "ev" / "progress.jsonl").read_text().splitlines()]


def _wins_first(n_points):
    whole = int(n_points)
    return lambda g, _cr: HS.CANDIDATE if g < whole else (None if g == whole and n_points > whole else HS.REFERENCE)


@pytest.mark.parametrize("outcome, mean, verdict", [
    (lambda g, cr: HS.CANDIDATE, 1.0, "CONTINUE"),
    (lambda g, cr: HS.REFERENCE, 0.0, "STOP"),
    (lambda g, cr: None, 0.5, "CONTINUE"),
    (_wins_first(32), 0.5, "CONTINUE"),
    (_wins_first(31.5), 31.5 / 64, "STOP"),                     # 0.4921875: the nearest reachable mean below 0.50
], ids=["all-candidate", "all-reference", "all-drawn", "exactly-half", "just-below"])
def test_THE_SCREEN_PLAYS_ALL_64_GAMES_WITH_THE_WHOLE_STUDENT_AND_APPLIES_THE_RULE(tmp_path, outcome, mean, verdict):
    calls, loaded = [], []
    code, rec = _screen(tmp_path, screen_seams(stub_play(outcome, calls=calls), loaded=loaded))
    assert code == HS.EXIT_COMPLETED and rec["status"] == "COMPLETED", rec.get("void")
    assert rec["kind"] == "htb-screen-record/1" and rec["card_sha256"] == H.CARD_SHA256
    assert rec["plan_doc_sha256"] == HS.PLAN_DOC_SHA256 and rec["admission"] == {"fit": "ELIGIBLE"}
    assert calls == list(range(64)) and [ev.network.name for ev in loaded] == ["student", "reference"]
    assert rec["screen"]["pairs"] == 32 and rec["screen"]["mean_pair_score"] == pytest.approx(mean)
    assert rec["verdict"] == rec["screen"]["verdict"] == verdict
    assert rec["seeds"] == {"base": BASE, "n_slots": 64}
    assert rec["student"] == {"path": str(HS.STUDENT["path"]), "sha256": HS.STUDENT["sha256"],
                              "state_digest": HS.STUDENT["state_digest"]}
    want = {HS.CANDIDATE: HS.STUDENT["state_digest"], HS.REFERENCE: HS.REFERENCE_DIGEST}
    assert rec["digests"] == rec["digests_at_end"] == want
    assert rec["opening_probe"]["max_prior_abs_diff"] > 0
    assert rec["protocol"]["classes"] == {k: "scripts.GPU.alphazero.root_sequential_halving.RootSequentialHalvingMCTS"
                                          for k in (HS.CANDIDATE, HS.REFERENCE)}
    assert rec["protocol"]["evaluators"] == {k: "MirrorAveragingEvaluator" for k in (HS.CANDIDATE, HS.REFERENCE)}
    agents = rec["protocol"]["agents"]
    assert [a["mcts_sims"] for a in agents] == [200, 200]
    assert agents[0]["checkpoint"] == str(HS.STUDENT["path"]) and agents[1]["checkpoint"] == HS.M.CHECKPOINT
    games = [json.loads(x) for x in (tmp_path / "ev" / "games.jsonl").read_text().splitlines()]
    assert [g["slot"] for g in games] == list(range(64)) and games[0]["pairing_id"].startswith("htb-screen")
    for side in (HS.CANDIDATE, HS.REFERENCE):
        assert rec["cost_totals"][side]["network_infer_calls"] == 128 and rec["cost_totals"][side]["searches"] == 128
    assert rec["guards"]["evaluations_ratio"] == 1.0 and rec["guards"]["search_seconds_ratio"] == 1.0
    assert rec["background_checks"] == 64 + 2 and rec["setting"] == {HS.SETTING: HS.SETTING_VALUE}


@pytest.mark.parametrize("kw, reason", [
    ({"secs": (0.30, 0.25)}, "search-seconds ratio"),                   # 1.2 > 1.10
    ({"extra_candidate_calls": 1}, "evaluations ratio"),                # 2.0 > 1.05
], ids=["seconds", "evaluations"])
def test_A_FAILED_COMPUTE_GUARD_STOPS_A_WINNING_SCREEN(tmp_path, kw, reason):
    code, rec = _screen(tmp_path, screen_seams(stub_play(lambda g, cr: HS.CANDIDATE, **kw)))
    assert code == HS.EXIT_COMPLETED and rec["screen"]["mean_pair_score"] == 1.0
    assert rec["verdict"] == "STOP" and len(rec["screen"]["stop_reasons"]) == 1
    assert reason in rec["screen"]["stop_reasons"][0]


def test_A_GUARD_INSIDE_ITS_LIMIT_DOES_NOT_STOP(tmp_path):
    code, rec = _screen(tmp_path, screen_seams(stub_play(lambda g, cr: HS.CANDIDATE, secs=(0.27, 0.25))))
    assert rec["guards"]["search_seconds_ratio"] == pytest.approx(1.08) and rec["verdict"] == "CONTINUE"


def test_EVERY_GAME_IS_LOGGED_AS_STARTED_BEFORE_IT_IS_PLAYED_and_BOINC_is_read_around_every_game(tmp_path):
    _screen(tmp_path, screen_seams(stub_play(lambda g, cr: HS.CANDIDATE)))
    ev = _events(tmp_path)
    names = [e["event"] for e in ev]
    assert names[0] == "start" and names[1] == "bg_check" and ev[1]["when"] == "before load"
    starts = [e for e in ev if e["event"] == "game_start"]
    assert [e["game"] for e in starts] == [e["slot"] for e in starts] == list(range(64))
    assert [e["red"] for e in starts[:2]] == [HS.CANDIDATE, HS.REFERENCE]
    for g in range(64):
        s = next(i for i, e in enumerate(ev) if e["event"] == "game_start" and e["game"] == g)
        t = next(i for i, e in enumerate(ev) if e["event"] == "game_end" and e["game"] == g)
        assert s < t and ev[t + 1]["event"] == "bg_check" and ev[t + 1]["when"] == f"after game {g}"
    assert [e["when"] for e in ev if e["event"] == "bg_check"][-1] == "before the verdict" and names[-1] == "end"


def _same_priors():
    return SideFake("student", HS.STUDENT["state_digest"], 8)


@pytest.mark.parametrize("over, step, why", [
    ({"versions": lambda: {**HS.VERSIONS, "python": "3.14.9"}}, "versions", "are not the plan's pinned"),
    ({"read_plan": lambda rel: {**PLAN_OK, "n_slots": 65}}, "seeds", "not exactly 64 slots"),
    ({"read_plan": lambda rel: {**PLAN_OK, "base": str(BASE)}}, "seeds", "not exactly 64 slots"),
    ({"read_plan": lambda rel: {**PLAN_OK, "kind": "combination-policy-cycle2-screen-plan/1"}}, "seeds",
     "not exactly 64 slots"),
    ({"seed_ok": lambda s: s != BASE + 63}, "seeds", "1 seeds not ACCOUNTED"),
    ({"checkpoint_sha1": lambda: "0" * 40}, "checkpoint", "is not calib020_0001's"),
    ({"student_sha256": lambda: "0" * 64}, "candidate", "sha256"),
    ({"load_candidate": lambda: SideFake("student", "0" * 64, 7)}, "candidate", "is not the student's"),
    ({"load_candidate": lambda: SideFake("student", HS.REFERENCE_DIGEST, 7)}, "candidate", "is not the student's"),
    ({"load_reference": lambda: SideFake("reference", HS.STUDENT["state_digest"], 8)}, "reference",
     "is not calib020_0001's"),
    ({"load_candidate": _same_priors}, "candidate", "opening priors equal"),
    ({"background": lambda: dict(BOINC)}, "background_compute", "before load: BOINC detected"),
], ids=["versions", "plan-size", "plan-base", "plan-kind", "seed", "checkpoint", "student-file", "student-digest",
        "student-is-reference", "reference-digest", "probe-equal", "boinc-before-load"])
def test_EACH_PRECONDITION_IS_A_VOID_BEFORE_ANY_GAME(tmp_path, over, step, why):
    calls = []
    code, rec = _screen(tmp_path, screen_seams(stub_play(lambda g, cr: HS.CANDIDATE, calls=calls), **over))
    assert code == HS.EXIT_VOID and rec["status"] == "VOID" and rec["void"]["step"] == step, rec.get("void")
    assert why in rec["void"]["detail"] and calls == [] and rec["verdict"] is None
    assert not (tmp_path / "ev" / "games.jsonl").exists() or not (tmp_path / "ev" / "games.jsonl").read_text()


def test_BOTH_SIDES_SHARING_A_NETWORK_IS_A_VOID(tmp_path):
    one = SideFake("student", HS.STUDENT["state_digest"], 7)
    code, rec = _screen(tmp_path, screen_seams(stub_play(lambda g, cr: HS.CANDIDATE), load_candidate=lambda: one,
                                               load_reference=lambda: one))
    assert code == HS.EXIT_VOID and rec["void"]["step"] == "load"


@pytest.mark.parametrize("side, step", [("student", "candidate"), ("reference", "reference")])
def test_A_SIDE_WHOSE_STATE_CHANGES_DURING_THE_RUN_IS_A_VOID_WITH_NO_SCORE(tmp_path, side, step):
    """The candidate stays frozen (and the reference unchanged): both digests are read again at the end."""
    loaded = []

    def play(task, red, black, *a):
        if task.game_idx == 63:
            next(ev for ev in loaded if ev.network.name == side).network.digest = "changed"
        return stub_play(lambda g, cr: HS.CANDIDATE)(task, red, black, *a)
    code, rec = _screen(tmp_path, screen_seams(play, loaded=loaded))
    assert code == HS.EXIT_VOID and rec["void"]["step"] == step and "at the end" in rec["void"]["detail"]
    assert rec["verdict"] is None and "screen" not in rec and "pair_scores" not in rec


@pytest.mark.parametrize("kw, step", [({"fail_at": 9}, "exception"), ({"searches": (2, 1)}, "attribution"),
                                      ({"searches": (0, 4)}, "attribution")], ids=["game-fails", "count", "one-side"])
def test_A_GAME_THAT_FAILS_OR_MISCOUNTS_ITS_SEARCHES_LEAVES_NO_SCORE(tmp_path, kw, step):
    code, rec = _screen(tmp_path, screen_seams(stub_play(lambda g, cr: HS.CANDIDATE, **kw)))
    assert code == HS.EXIT_VOID and rec["void"]["step"] == step, rec.get("void")
    assert rec["verdict"] is None and "screen" not in rec and "pair_scores" not in rec


def test_BOINC_MID_RUN_IS_A_VOID_WITH_NO_SCORE(tmp_path):
    seen = iter([CLEAR] * 6 + [BOINC])
    code, rec = _screen(tmp_path, screen_seams(stub_play(lambda g, cr: HS.CANDIDATE), background=lambda: next(seen)))
    assert code == HS.EXIT_VOID and rec["void"]["step"] == "background_compute" and rec["verdict"] is None


# ─────────────────────────────── reuse: the loop, the seams, the whole-network round trip ───────────────────────────────

def _loop(src):
    start = src.index("    results = []\n")
    end_mark = '    _watch(seams, rec, log, "before the verdict")\n'
    return src[start:src.index(end_mark) + len(end_mark)]


def test_THE_LOOP_IS_THE_CYCLE_2_SCREEN_S_TEXT_EXACTLY():
    ours, reviewed = _loop(inspect.getsource(HS._screen)), _loop(inspect.getsource(CPP._screen))
    assert ours == reviewed and ours.count("\n") == 30
    assert HS._watch is CPP._watch and HS.opening_probe is CPP.opening_probe


def test_THE_REAL_SEAMS_ARE_THE_MATCH_S_LOADER_LOOP_AND_CHECKS(monkeypatch):
    """Built, never called with a real model: both sides through the match's loader -- the student file whole, the
    checkpoint -- and the match's loop, between-games step, the registry's seed rule and the background check."""
    from scripts.GPU.alphazero import background_compute as BG
    from scripts.GPU.alphazero import mirror_halving_match as MHM
    real = _REAL_SCREEN_SEAMS(pathlib.Path("/nonexistent/model.safetensors"))
    assert real.play_and_build is K.play_and_build and real.between_games is MHM._between_games
    assert real.background is BG.observe and real.digest is H._digest and real.versions is CPP._versions
    seen = []
    monkeypatch.setattr(ER, "_default_evaluator_factory", lambda path: seen.append(path) or path)
    assert real.load_candidate() == str(HS.STUDENT["path"]) and real.load_reference() == "/nonexistent/model.safetensors"
    assert seen == [str(HS.STUDENT["path"]), "/nonexistent/model.safetensors"]


def test_THE_WHOLE_NETWORK_SURVIVES_THE_FIT_S_SAVE_AND_THE_MATCH_S_LOAD_with_its_BatchNorm_statistics(tmp_path):
    """A RANDOM 128x6 network (no checkpoint) with moved BatchNorm statistics, saved as `run fit` saved the student and
    loaded as the screen loads it: the state digest is unchanged, the network is in evaluation mode, and the loaded
    priors differ from a different network's -- so the plan's digest check is the right identity check."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from scripts.GPU.alphazero import search_distillation_d2 as D
    from scripts.GPU.alphazero import search_distillation_d2_measure as M
    from scripts.GPU.alphazero.network import create_network

    net = create_network(in_channels=30)
    stats = [(k, mx.array(np.random.default_rng(3).uniform(0.5, 1.5, v.shape).astype(np.float32)))
             for k, v in tree_flatten(net.parameters()) if k.endswith(("running_mean", "running_var"))]
    assert stats
    net.update(tree_unflatten(stats))
    net.eval()
    saved = D.save_student(net, str(tmp_path / "student.safetensors"))
    assert saved["state_digest"] == H._digest(net)
    ev = ER._default_evaluator_factory(saved["path"])
    assert H._digest(ev.network) == saved["state_digest"]
    assert M._require_eval(ev.network, "after load")["modules_in_training_mode"] == 0
    # a deterministically DIFFERENT network: freshly initialized networks give near-uniform opening priors (about 1e-5
    # apart, sometimes exactly equal), which made this check flaky (fixed 2026-11-06); a seeded shift of every weight
    # makes the two networks' priors differ by construction
    rng = np.random.default_rng(5)
    other = create_network(in_channels=30)
    other.update(tree_unflatten([(k, v + mx.array(rng.normal(0, 0.2, v.shape).astype(np.float32)))
                                 for k, v in tree_flatten(other.parameters())
                                 if not k.endswith(("running_mean", "running_var"))]))
    other.eval()
    D.save_student(other, str(tmp_path / "other.safetensors"))
    probe = HS.opening_probe(ev, ER._default_evaluator_factory(str(tmp_path / "other.safetensors")))
    assert probe["max_prior_abs_diff"] > 1e-4 and probe["legal"] > 500


# ─────────────────────────────── the wrapper and the seed copies ───────────────────────────────

@pytest.mark.parametrize("kind", ["wrapper", "survey", "closing"])
def test_EACH_COPY_IS_THE_DECLARED_DERIVATION_OF_ITS_REVIEWED_SOURCE(kind):
    src, sha = SPEC.SOURCES[kind]
    raw = (ROOT / src).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sha
    target = ROOT / SPEC.TARGETS[kind]
    assert target.read_text() == SPEC.derive(kind, raw.decode())


def test_THE_WRAPPER_RUNS_THIS_MODULE_BEHIND_ITS_OWN_CLOSED_GATE_and_ran_once():
    """RE-ANCHORED 2026-11-06 (the ONE run: gate 138ede34, evidence 25f4863a): its create-only outputs exist, the
    run COMPLETED with cleanup confirmed, and the gate is closed again."""
    p = ROOT / SPEC.TARGETS["wrapper"]
    text = p.read_text()
    assert os.access(p, os.X_OK) and subprocess.run(["bash", "-n", str(p)]).returncode == 0
    assert f"\nTERM_S={SPEC.TERM_S}\n" in text and "\nGRACE_S=10\n" in text and SPEC.TERM_S == 3410
    assert "\nGATE=HTB_SCREEN_AUTHORIZED\n" in text and "\nGATE_REL=scripts/GPU/alphazero/htb_screen_authorization.py\n" in text
    assert f"\nRUN_REL={HS.EVIDENCE}\n" in text and f"\nOUT_REL={HS.RUN_DIR}\n" in text
    assert "\nMODULE=scripts.GPU.alphazero.htb_screen\n" in text and '-m "$MODULE" run screen --checkpoint "$CKPT"' in text
    assert f"\nRECORD_KIND={HS.RECORD_KIND}\n" in text and "MLX_ENABLE_TF32=0" in text
    assert "  3) outcome=VOID ;;\n  4) outcome=REFUSED ;;\n  2) outcome=USAGE ;;\n" in text
    assert sorted(x.name for x in p.parent.iterdir()) == [
        "00_prelaunch.txt", "01_wrapper.sh", "bg_check.txt", "idle_check.txt", "outer.json", "pgid.txt", "plan.json",
        "run.stderr.txt", "run.stdout.txt"]
    outer = json.loads((p.parent / "outer.json").read_text())
    assert {k: outer[k] for k in ("inner_exit", "outer_exit", "outcome", "gate_restored", "survivors", "record_by",
                                  "term_s", "grace_s")} == {
        "inner_exit": 0, "outer_exit": 0, "outcome": "COMPLETED", "gate_restored": True, "survivors": "",
        "record_by": "script", "term_s": SPEC.TERM_S, "grace_s": 10}
    assert outer["wall_seconds"] < SPEC.TERM_S and (ROOT / HS.EVIDENCE / "record.json").is_file()
    assert GATE.HTB_SCREEN_AUTHORIZED is False


def _module(kind):
    spec = importlib.util.spec_from_file_location(f"_test_htb_screen_{kind}", ROOT / SPEC.TARGETS[kind])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_THE_SURVEY_COPY_SERVES_THIS_SCREEN_S_PLAN_and_loads_the_pinned_method():
    S = _module("survey")
    assert S.N_SLOTS == 64 and S.HERE == "docs/superpowers/evidence/htb-screen-seed-survey" and S.PLAN == HS.PLAN
    assert S.build_plan(BASE, "ab" * 32) == {"kind": HS.PLAN_KIND, "base": BASE, "n_slots": 64,
                                            "survey_record_sha256": "ab" * 32}
    assert S.slot_lines((BASE, BASE + 64))[63] == f"    SLOT 63 seed {BASE + 63}"
    survey, _proof, SEL = S.load_method()
    assert (survey.N, survey.PLAN, survey.HERE) == (64, HS.PLAN, S.HERE) and SEL.BAND[0] == 2027 * 10 ** 5
    assert all((ROOT / rel).is_file() for rel in S.OUT.values())                    # each mode ran once (below)


def test_THE_SCREEN_S_SEEDS_ARE_SURVEYED_PROVEN_REGISTERED_AND_CLOSED_and_every_mode_refuses_to_rerun():
    """Seed preparation (the user's review, 2026-11-06): survey 2ed2f3d4, pre-proof 5b326407, registration 7f2ab4d7,
    post-proof bd876999. The block is READ from the committed plan and survey record -- no seed literal here.
    RE-ANCHORED 2026-11-06 (the ONE run 25f4863a; K1 e819742b RECONCILES 15/15; K2 74025ee2): every game started, so
    the whole block is EXPOSED and RETIRED, each its list's last entry, and no seed of it is available again."""
    from scripts.GPU.alphazero import e4_screen_reference as REG
    from scripts.GPU.alphazero import seed_band_2027 as SB
    S = _module("survey")
    method, _proof, sel = S.load_method()
    plan = json.loads((ROOT / HS.PLAN).read_text())
    raw = (ROOT / S.OUT["survey"]).read_bytes()
    rec = S.read_record(raw.decode(), method, sel)                       # CLEAN, every control passed, a clean tree
    block = rec["block"]
    assert plan == S.build_plan(rec["base"], hashlib.sha256(raw).hexdigest()) and block == (plan["base"], plan["base"] + 64)
    assert SB.BAND[0] <= block[0] and block[1] <= SB.BAND[1] and block[0] % SB.ALIGN == 0
    for mode in ("pre", "post"):
        text = (ROOT / S.OUT[mode]).read_text()
        assert f"EXPECTED STATE ({mode}): HOLDS" in text and "controls passed: 55/55" in text
        assert f"PROOF VERDICT ({mode}): THE H-TB SCREEN BLOCK IS CLEAN AND THE STATE IS AS EXPECTED\n" in text
        assert text.splitlines()[2].endswith("; working tree clean")
        assert f"sha256 {hashlib.sha256(raw).hexdigest()}" in text
    assert REG.ACCOUNTED_SEED_INTERVALS.count(block) == 1 and REG.ACCOUNTED_SEED_INTERVALS[-1] == block
    assert all(hi <= block[0] for lo, hi in REG.ACCOUNTED_SEED_INTERVALS[:-1] if hi > SB.BAND[0])
    for intervals in (REG.EXPOSED_SEED_INTERVALS, REG.RETIRED_SEED_INTERVALS):
        assert intervals.count(block) == 1 and intervals[-1] == block
    assert not any(lo < block[1] and block[0] < hi for lo, hi in REG.TEST_ONLY_SEED_INTERVALS)
    assert all(REG.seed_is_accounted(s) and REG.seed_is_exposed(s) and REG.seed_is_retired(s)
               and REG.seed_is_unavailable(s) and not REG.seed_is_test_only(s) for s in range(*block))
    for mode in ("survey", "pre", "post"):
        with pytest.raises(S.Refused, match="each mode runs once"):
            S.main(["x", "--mode", mode])
    C = _module("closing")
    k1 = (ROOT / C.OUT).read_text()
    assert f"VERDICT: RECONCILES -- EXPOSE {list(block)}; RETIRE {list(block)}\n" in k1
    assert "controls passed: 15/15" in k1 and k1.splitlines()[2].endswith("; working tree clean")
    assert all(ok for _l, ok, _d in C._controls(block[0]))
    assert GATE.HTB_SCREEN_AUTHORIZED is False
    rec = json.loads((ROOT / HS.EVIDENCE / "record.json").read_text())
    assert rec["status"] == "COMPLETED" and rec["seeds"] == {"base": block[0], "n_slots": 64}


def test_THE_CLOSING_COPY_READS_THIS_SCREEN_S_RUN_and_its_controls_all_pass():
    C = _module("closing")
    assert (C.RUN, C.RUN_DIR, C.N_SLOTS) == (HS.EVIDENCE, HS.RUN_DIR, 64)
    controls = C._controls(BASE)
    assert len(controls) == 15 and all(ok for _label, ok, _d in controls), controls
    # RE-ANCHORED 2026-11-06 (K1 run once, e819742b): its record exists, and a second run refuses before writing
    before = (ROOT / C.OUT).read_bytes()
    assert C.main() == 4 and (ROOT / C.OUT).read_bytes() == before


# ─────────────────────────────── the proposed ceiling (plan §8) ───────────────────────────────

def test_THE_PROPOSED_CEILING_RECOMPUTES_FROM_THE_COMMITTED_CYCLE_2_STRENGTH_RUN():
    ev = ROOT / "docs/superpowers/evidence"
    rec = json.loads((ev / "combination-policy-cycle2-strength/record.json").read_text())
    outer = json.loads((ev / "combination-policy-cycle2-strength-run/outer.json").read_text())
    games = [json.loads(x) for x in (ev / "combination-policy-cycle2-strength/games.jsonl").read_text().splitlines()]
    assert rec["versions"] == HS.VERSIONS and len(games) == 1018
    c = rec["cost_totals"]["candidate"]
    sps, ppg = c["search_seconds"] / c["searches"], sum(g["n_moves"] for g in games) / len(games)
    proj = sps * ppg * 64
    ceiling = math.ceil(proj * 1.5 / 60) * 60
    assert (round(sps, 4), round(ppg, 3), round(proj, 1), round(proj * 1.5, 1)) == (0.5559, 63.725, 2267.0, 3400.5)
    assert ceiling == 3420 == SPEC.TERM_S + 10
    assert round(outer["wall_seconds"] / len(games) * 64 * 1.5, 1) == 3392.8
    capped = [g for g in games if g["reason"] == "state_cap"]
    slow = max(g["seconds"] / g["n_moves"] for g in capped)
    assert int((SPEC.TERM_S - proj) / (280 * slow - sps * ppg)) == 8 and round(len(capped) / len(games) * 64, 1) == 2.8
    doc = (ROOT / HS.PLAN_DOC).read_text()
    assert "**3,420 s — TERM at 3,410 s, KILL 10 s later, cleanup observation through 3,430 s.**" in doc
