"""The combination policy pilot's SCREEN (card `docs/superpowers/2026-10-28-combination-policy-pilot-card.md` §5 step
5), built only on the user's review (ledger, 2026-10-31): `combination_policy_pilot run screen`, its wrapper
(`docs/superpowers/evidence/combination-policy-screen-run/01_wrapper.sh`), its proposed ceiling and its seed copies.

CPU only. No test loads calib020_0001 or the trained head for evaluation, touches the GPU, opens a real gate, scans the
real repository for a survey, writes the registry or plays a real game against the proven agent:
* admission and the head binding run on the REAL committed training records (read-only) and on synthetic ones;
* the screen itself runs through the real record boundary with GPU-free fakes and a stubbed game;
* "search reaches the residual head through forward_padded" is checked on TINY RANDOM networks on the CPU, through the
  real LocalGPUEvaluator (compiled, as the match loads it), the averaging wrapper and the halving search;
* the wrapper runs in a throwaway repository with stubs; the seed copies' modes run on a synthetic repository.

🔴 EVERY TEST IS ISOLATED, WHATEVER THE GATE SAYS (#74): the autouse fixture makes every real seam and the real
record boundary unreachable."""
import difflib
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import types

import numpy as np
import pytest

from scripts.GPU.alphazero import combination_policy_pilot as P
from scripts.GPU.alphazero import eval_runner as ER
from scripts.GPU.alphazero import mirror_halving_combination as K
from scripts.GPU.alphazero import root_sequential_halving as RSH
from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator
from tests import _combination_fakes as F
from tests.test_combination_policy_launch import STUB_BG, STUB_IDLE, STUB_PILOT, _alive, _git

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = ROOT / "docs/superpowers/evidence"
WRAPPER_TRAIN = EV / "combination-policy-train-run/01_wrapper.sh"
WRAPPER = EV / "combination-policy-screen-run/01_wrapper.sh"
GATE_NAME = "COMBINATION_POLICY_SCREEN_AUTHORIZED"
SURVEY_GEN = "combination-policy-generate-seed-survey/01_seed_survey.py"
SURVEY = "combination-policy-screen-seed-survey/01_seed_survey.py"
CLOSING_S2 = "mirror-halving-strength2-seed-closing/01_derive_usage.py"
CLOSING = "combination-policy-screen-seed-closing/01_derive_usage.py"
TRAIN = P.KINDS["train"]["evidence"]
TRAIN_RUN = "docs/superpowers/evidence/combination-policy-train-run"
BASE = 777000                                   # a fake plan's base: not a registry seed
HEAD_D = dict(P.TRAINED_HEAD["digests"])
PARENT_D = {**HEAD_D, "policy_head.": "the parent's head"}
CLEAR = {"ok": True, "n_procs": 800, "hits": [], "n_hits": 0, "exempt": [], "error": None}
BOINC = {"ok": False, "n_procs": 801, "hits": ["5635 boinc_project hsgamma_FGRP5"], "n_hits": 1, "exempt": [],
         "error": None}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """EVERY test: the pilot's real seams and record boundary are unreachable; MLX runs on the CPU."""
    def unreachable(*_a, **_k):
        raise AssertionError("a test reached a real seam or the real record boundary")
    for name in ("_real_gen_seams", "_real_train_seams", "_real_screen_seams", "conclude"):
        monkeypatch.setattr(P, name, unreachable)
    monkeypatch.setattr(P, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv(P.SETTING, P.SETTING_VALUE)
    import mlx.core as mx
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


_REAL_CONCLUDE = P.conclude            # bound at import, before the isolation replaces the module's name
_REAL_SCREEN_SEAMS = P._real_screen_seams


def test_THE_ISOLATION_IS_IN_FORCE():
    with pytest.raises(AssertionError, match="real seam"):
        P._real_screen_seams(pathlib.Path("/nonexistent"), {})
    with pytest.raises(AssertionError, match="real seam"):
        P.conclude("screen", None, pathlib.Path("/nonexistent"), {})


# ─────────────────────────────── the gate, the kind and admission ───────────────────────────────

def test_THE_SCREEN_GATE_IS_A_CLOSED_SINGLE_DECLARATION_and_the_closed_gate_refuses(monkeypatch, capsys):
    import ast
    from scripts.GPU.alphazero import combination_policy_screen_authorization as G
    gate, name = P.KINDS["screen"]["gate"]
    assert gate is G and name == GATE_NAME and getattr(G, GATE_NAME) is False
    tree = ast.parse(pathlib.Path(G.__file__).read_text())
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert len(body) == 1 and body[0].targets[0].id == GATE_NAME
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    assert P.main(["run", "screen", "--checkpoint", "/x"]) == P.EXIT_REFUSED
    assert "is closed" in capsys.readouterr().err


def test_THE_SCREEN_IS_32_PAIRS_ON_ITS_OWN_PATHS_and_the_card_s_rule():
    s = P.KINDS["screen"]
    assert s["pairs"] == K.SCREEN["pairs"] == 32 and K.SCREEN["stop_below"] == 0.40
    assert s["evidence"] == "docs/superpowers/evidence/combination-policy-screen"
    assert s["plan"] == "docs/superpowers/evidence/combination-policy-screen-run/plan.json"
    others = {P.KINDS[k]["evidence"] for k in ("throughput", "throughput2", "generate", "train")}
    others |= {str(pathlib.Path(P.KINDS[k]["plan"]).parent) for k in ("throughput", "throughput2", "generate")}
    assert {s["evidence"], str(pathlib.Path(s["plan"]).parent)} & others == set()
    assert "amendment" not in s and "slots" not in s and not (ROOT / s["evidence"]).exists()
    assert P.ADMISSION["screen"] == [(TRAIN, "verdict", "ELIGIBLE")]
    assert P.OUTER_ADMISSION["screen"] == [(TRAIN_RUN, TRAIN)]
    assert (K.CANDIDATE, P.REFERENCE) == ("candidate", "reference") and K.CANDIDATE_SIMS == 200


def test_REAL_STATE_THE_COMMITTED_TRAINING_RECORDS_ADMIT_THE_SCREEN_AND_BIND_THE_SAVED_HEAD():
    """Read-only, through the real committed readers: the training record (COMPLETED, ELIGIBLE) and its outer.json
    admit the screen, and the binding is exactly head.npz's committed bytes, sha256 and digests."""
    assert P.admitted("screen", P._read_record, P._read_outer) == {TRAIN: "ELIGIBLE",
                                                                   f"{TRAIN_RUN}/outer.json": "COMPLETED"}
    h = P.trained_head(P._read_record, P._read_committed_bytes)
    raw = (ROOT / P.TRAINED_HEAD["file"]).read_bytes()
    assert h["bytes"] == raw and hashlib.sha256(raw).hexdigest() == h["sha256"] == P.TRAINED_HEAD["sha256"]
    rec = json.loads((ROOT / TRAIN / "record.json").read_text(encoding="utf-8"))
    assert h["digests"] == rec["digests_after"] and rec["head_sha256"] == h["sha256"]
    assert {p: rec["digests_before"][p] for p in ("encoder.", "value_head.")} == \
        {p: h["digests"][p] for p in ("encoder.", "value_head.")}
    assert P.digest(P._npz_arrays(raw), "policy_head.") == h["digests"]["policy_head."]   # the bytes ARE that head


GOOD_TRAIN = {"run": "train", "status": "COMPLETED", "verdict": "ELIGIBLE", "card_sha256": P.CARD_SHA256,
              "head": "a" * 40, "head_file": "head.npz", "head_sha256": P.TRAINED_HEAD["sha256"],
              "digests_before": {**HEAD_D, "policy_head.": "before"}, "digests_after": dict(HEAD_D)}
GOOD_OUTER = {"inner_exit": 0, "outcome": "COMPLETED", "outer_exit": 0, "gate_restored": True, "survivors": "",
              "record_by": "script", "head": "a" * 40}


def _readers(rec=None, outer=GOOD_OUTER, data=None):
    rec = GOOD_TRAIN if rec is None else rec

    def read_record(rel):
        assert rel == TRAIN, rel
        return rec

    def read_outer(rel):
        assert rel == TRAIN_RUN, rel
        if outer is None:
            raise FileNotFoundError(rel)
        return outer

    def read_bytes(rel):
        assert rel == P.TRAINED_HEAD["file"], rel
        if data is False:
            raise FileNotFoundError(rel)
        return (ROOT / rel).read_bytes() if data is None else data
    return read_record, read_outer, read_bytes


def test_ADMISSION_NEEDS_THE_ELIGIBLE_TRAINING_RECORD_AND_ITS_SUCCESSFUL_OUTER_JSON():
    rr, ro, _rb = _readers()
    assert P.admitted("screen", rr, ro) == {TRAIN: "ELIGIBLE", f"{TRAIN_RUN}/outer.json": "COMPLETED"}
    for change in ({"verdict": "STOP"}, {"status": "VOID"}, {"card_sha256": "0" * 64}):
        with pytest.raises(P.Refused, match="not COMPLETED with verdict ELIGIBLE"):
            P.admitted("screen", _readers({**GOOD_TRAIN, **change})[0], ro)
    with pytest.raises(P.Refused, match="outer.json does not exist"):
        P.admitted("screen", *_readers(outer=None)[:2])
    for change in ({"outcome": "TIMEOUT", "outer_exit": 124, "inner_exit": 124}, {"survivors": "123 "},
                   {"gate_restored": False}, {"record_by": "wrapper"}, {"head": "b" * 40}):
        with pytest.raises(P.Refused, match="does not show a successful run"):
            P.admitted("screen", *_readers(outer={**GOOD_OUTER, **change})[:2])


@pytest.mark.parametrize("rec, data, why", [
    ({"verdict": "STOP"}, None, "not an ELIGIBLE training run"),
    ({"run": "generate"}, None, "not an ELIGIBLE training run"),
    ({"head_sha256": "0" * 64}, None, "not the pinned head"),
    ({"head_file": "other.npz"}, None, "not the pinned head"),
    ({"digests_after": {**HEAD_D, "policy_head.": "x"}}, None, "digests after training are not the pinned"),
    ({"digests_before": {**HEAD_D, "encoder.": "x"}}, None, "before training is not the pinned one"),
    ({"digests_before": {**HEAD_D, "value_head.": "x"}}, None, "before training is not the pinned one"),
    ({}, b"altered bytes", "head.npz is not the pinned bytes"),
    ({}, False, "does not exist: the screen has no trained head"),
], ids=["stop", "not-train", "sha", "file", "digest-after", "encoder-before", "value-before", "bytes", "missing"])
def test_THE_HEAD_BINDING_REFUSES_ANYTHING_BUT_THE_SAVED_HEAD(rec, data, why):
    rr, _ro, rb = _readers({**GOOD_TRAIN, **rec}, data=data)
    with pytest.raises(P.Refused, match=re.escape(why)):
        P.trained_head(rr, rb)


def test_RUN_SCREEN_WITH_ITS_GATE_OPEN_PASSES_EVERY_REAL_CHECK_TO_THE_SEAMS(monkeypatch, capsys, tmp_path):
    """The gate opened in this process only. Through the REAL pins, admission and head binding the entry reaches its
    checkpoint refusal; with a checkpoint file present it reaches the real seams -- unreachable here -- and nothing
    else: no evidence directory is made."""
    gate, name = P.KINDS["screen"]["gate"]
    monkeypatch.setattr(gate, name, True)
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    assert P.main(["run", "screen", "--checkpoint", "/no/such/file"]) == P.EXIT_REFUSED
    assert "--checkpoint must name an existing file" in capsys.readouterr().err
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"not a model")
    with pytest.raises(AssertionError, match="real seam"):
        P.main(["run", "screen", "--checkpoint", str(ckpt)])
    assert not (ROOT / P.KINDS["screen"]["evidence"]).exists()


# ─────────────────────────────── the screen on fakes, through the real record boundary ───────────────────────────────

class ScreenFake(F.FakeEvaluator):
    """FakeEvaluator with its own network object; once `install_head` marks that network trained, its priors move as a
    new head's would, and its values do not."""

    def __init__(self, log=None):
        super().__init__()
        self.network = types.SimpleNamespace(trained=False)
        self._biases = (self.bias, self.bias + np.random.default_rng(99).normal(0, 0.5, (24, 24)))
        self.log = log

    def infer(self, boards, rows, cols, mask, active_size):
        if self.log is not None:
            self.log.append(("infer", self.network.trained))
        self.bias = self._biases[self.network.trained]
        return super().infer(boards, rows, cols, mask, active_size)


def stub_play(outcome, searches=(2, 2), calls=None, fail_at=None):
    """A scripted game that checks the wiring it is handed, makes one averaged call per side (two underlying calls
    each), reports the scripted searches and the winner outcome(game_idx, candidate_is_red) names."""
    def play(task, red, black, config, capture, replay_dir, classes, on_search):
        if task.game_idx == fail_at:
            raise RuntimeError(f"injected failure in game {fail_at}")
        cand_red = task.red_agent.agent_id == K.CANDIDATE
        assert (task.game_idx % 2 == 0) == cand_red and task.seed == BASE + task.game_idx
        assert type(red) is MirrorAveragingEvaluator and type(black) is MirrorAveragingEvaluator
        assert red.base.network.trained is cand_red and black.base.network.trained is (not cand_red)
        assert classes == {K.CANDIDATE: RSH.RootSequentialHalvingMCTS, P.REFERENCE: RSH.RootSequentialHalvingMCTS}
        assert capture is True and (config.board_size, config.max_moves) == (24, 280)
        assert ER.agent_sims(task.red_agent, config) == ER.agent_sims(task.black_agent, config) == 200
        args = (np.array([[0, 1]], np.int32), np.array([[1, 2]], np.int32), np.ones((1, 2), np.float32), 24)
        for ev in (red, black):
            ev.infer(np.zeros((1, 24, 24, 30), np.float32), *args)
        for is_red, n in ((True, searches[0]), (False, searches[1])):
            for _ in range(n):
                on_search(is_red, 0.25, True)
        if calls is not None:
            calls.append(task.game_idx)
        who = outcome(task.game_idx, cand_red)
        winner = None if who is None else ("red" if (who == K.CANDIDATE) == cand_red else "black")
        return ER.make_result(task, winner, "win" if winner else "state_cap", 4, config=config)
    return play


def screen_seams(play, loaded=None, log=None, **over):
    loaded = [] if loaded is None else loaded

    def load():
        ev = ScreenFake(log)
        loaded.append(ev)
        return ev

    def install(net):
        net.trained = True
        if log is not None:
            log.append(("install", True))
        return dict(HEAD_D)

    s = dict(versions=lambda: dict(P.VERSIONS), read_plan=lambda rel: {"base": BASE, "n_slots": 64},
             seed_ok=lambda s: True, checkpoint_sha1=lambda: P.M.CHECKPOINT_SHA1, load_evaluator=load,
             install_head=install, digests=lambda net: dict(PARENT_D),
             require_eval=lambda ev, when: {"modules_in_training_mode": 0, "when": when}, loaded_modules=lambda: {},
             background=lambda: dict(CLEAR), play_and_build=play, between_games=lambda: None)
    s.update(over)
    return P.ScreenSeams(**s)


HEAD = {"file": P.TRAINED_HEAD["file"], "sha256": P.TRAINED_HEAD["sha256"], "digests": dict(HEAD_D), "bytes": b""}


def _screen(tmp_path, seams):
    return _REAL_CONCLUDE("screen", lambda o, r, lg: P._screen(seams, HEAD, o, r, lg), tmp_path / "ev", {})


def _events(tmp_path):
    return [json.loads(x) for x in (tmp_path / "ev" / "progress.jsonl").read_text().splitlines()]


def _wins_first(n_points):
    """The candidate wins games 0..floor(n)-1, draws the next when n has a half, and loses the rest."""
    whole = int(n_points)
    return lambda g, _cr: K.CANDIDATE if g < whole else (None if g == whole and n_points > whole else P.REFERENCE)


@pytest.mark.parametrize("outcome, mean, verdict", [
    (lambda g, cr: K.CANDIDATE, 1.0, "CONTINUE"),
    (lambda g, cr: P.REFERENCE, 0.0, "STOP"),
    (lambda g, cr: None, 0.5, "CONTINUE"),
    (_wins_first(26), 26 / 64, "CONTINUE"),          # 0.40625: the nearest reachable mean above 0.40
    (_wins_first(25.5), 25.5 / 64, "STOP"),          # 0.3984: the nearest below
], ids=["all-candidate", "all-reference", "all-drawn", "just-above", "just-below"])
def test_THE_SCREEN_PLAYS_ALL_64_GAMES_AND_APPLIES_THE_FROZEN_RULE_ONLY(tmp_path, outcome, mean, verdict):
    calls, loaded = [], []
    code, rec = _screen(tmp_path, screen_seams(stub_play(outcome, calls=calls), loaded=loaded))
    assert code == P.EXIT_COMPLETED and rec["status"] == "COMPLETED", rec.get("void")
    assert calls == list(range(64)) and len(loaded) == 2
    assert rec["screen"]["pairs"] == 32 and rec["screen"]["mean_pair_score"] == pytest.approx(mean)
    assert rec["verdict"] == rec["screen"]["verdict"] == verdict and "not a strength result" in rec["screen"]["note"]
    assert rec["seeds"] == {"base": BASE, "n_slots": 64} and rec["trained_head"] == {k: HEAD[k] for k in
                                                                                    ("file", "sha256", "digests")}
    assert rec["digests"] == {K.CANDIDATE: HEAD_D, P.REFERENCE: PARENT_D}
    assert rec["head_probe"]["max_prior_abs_diff"] > 0
    assert rec["head_probe"]["value_abs_diff"] == 0.0 and rec["head_probe"]["legal"] == _opening_inputs()[1].shape[1]
    assert rec["protocol"]["classes"] == {k: "scripts.GPU.alphazero.root_sequential_halving.RootSequentialHalvingMCTS"
                                          for k in (K.CANDIDATE, P.REFERENCE)}
    assert rec["protocol"]["evaluators"] == {K.CANDIDATE: "MirrorAveragingEvaluator",
                                             P.REFERENCE: "MirrorAveragingEvaluator"}
    assert [a["mcts_sims"] for a in rec["protocol"]["agents"]] == [200, 200]
    assert rec["protocol"]["agents"][0]["checkpoint"].endswith("+" + P.TRAINED_HEAD["file"])
    games = [json.loads(x) for x in (tmp_path / "ev" / "games.jsonl").read_text().splitlines()]
    assert [g["slot"] for g in games] == list(range(64)) and all(g["cost"][K.CANDIDATE]["searches"] == 2 for g in games)
    for side in (K.CANDIDATE, P.REFERENCE):                # one averaged call per game: two underlying calls
        assert rec["cost_totals"][side]["network_infer_calls"] == 128 and rec["cost_totals"][side]["searches"] == 128
    assert rec["cost_ratios"]["evaluations_ratio"] == 1.0 and "descriptive" in rec["cost_ratios"]["note"]
    assert rec["background_checks"] == 64 + 2 and rec["setting"] == {P.SETTING: P.SETTING_VALUE}


def test_EVERY_GAME_IS_LOGGED_AS_STARTED_BEFORE_IT_IS_PLAYED_and_BOINC_is_read_around_every_game(tmp_path):
    _screen(tmp_path, screen_seams(stub_play(lambda g, cr: K.CANDIDATE)))
    ev = _events(tmp_path)
    names = [e["event"] for e in ev]
    assert names[0] == "start" and names[1] == "bg_check" and ev[1]["when"] == "before load"
    starts = [e for e in ev if e["event"] == "game_start"]
    assert [e["game"] for e in starts] == [e["slot"] for e in starts] == list(range(64))
    assert [e["red"] for e in starts[:2]] == [K.CANDIDATE, P.REFERENCE]
    for g in range(64):
        s = next(i for i, e in enumerate(ev) if e["event"] == "game_start" and e["game"] == g)
        t = next(i for i, e in enumerate(ev) if e["event"] == "game_end" and e["game"] == g)
        assert s < t and ev[t + 1] == {**ev[t + 1], "event": "bg_check", "when": f"after game {g}"}
    assert [e["when"] for e in ev if e["event"] == "bg_check"][-1] == "before the verdict" and names[-1] == "end"


def test_THE_HEAD_IS_INSTALLED_BEFORE_ANY_INFERENCE_and_the_probe_precedes_the_cost_count(tmp_path):
    """The real loader compiles its forward pass on the first call, so the head must be in place first: the first
    inference is the probe's, after the install; the probe's two calls are not counted as game cost."""
    log = []
    code, rec = _screen(tmp_path, screen_seams(stub_play(lambda g, cr: K.CANDIDATE), log=log))
    assert code == P.EXIT_COMPLETED, rec.get("void")
    assert log[:3] == [("install", True), ("infer", True), ("infer", False)]
    assert sum(1 for e in log if e[0] == "infer") == 2 + 64 * 2 * 2
    assert rec["cost_totals"][K.CANDIDATE]["network_infer_calls"] == 128


@pytest.mark.parametrize("over, step, why", [
    ({"install_head": lambda net: {**HEAD_D, "policy_head.": "another head"}}, "head", "not the trained head's"),
    ({"install_head": lambda net: {**HEAD_D, "encoder.": "another encoder"}}, "head", "not the trained head's"),
    ({"install_head": lambda net: dict(HEAD_D)}, "head", "search does not reach the head"),
    ({"digests": lambda net: {**PARENT_D, "value_head.": "x"}}, "reference", "not calib020_0001's parent"),
    ({"digests": lambda net: dict(HEAD_D)}, "reference", "not calib020_0001's parent"),
    ({"read_plan": lambda rel: {"base": BASE, "n_slots": 65}}, "seeds", "not exactly 64 slots"),
    ({"read_plan": lambda rel: {"base": str(BASE), "n_slots": 64}}, "seeds", "not exactly 64 slots"),
    ({"seed_ok": lambda s: s != BASE + 63}, "seeds", "1 seeds not ACCOUNTED"),
    ({"checkpoint_sha1": lambda: "0" * 40}, "checkpoint", "is not calib020_0001's"),
    ({"versions": lambda: {**P.VERSIONS, "mlx": "0.0"}}, "versions", "are not the card's"),
    ({"background": lambda: dict(BOINC)}, "background_compute", "before load: BOINC detected"),
], ids=["head-arrays", "head-encoder", "probe-equal", "reference-value", "reference-trained", "plan-size",
        "plan-base", "seed", "checkpoint", "versions", "boinc-before-load"])
def test_EACH_PRECONDITION_IS_A_VOID_BEFORE_ANY_GAME(tmp_path, over, step, why):
    calls = []
    code, rec = _screen(tmp_path, screen_seams(stub_play(lambda g, cr: K.CANDIDATE, calls=calls), **over))
    assert code == P.EXIT_VOID and rec["status"] == "VOID" and rec["void"]["step"] == step, rec.get("void")
    assert why in rec["void"]["detail"] and calls == [] and rec["verdict"] is None
    assert not (tmp_path / "ev" / "games.jsonl").exists() or not (tmp_path / "ev" / "games.jsonl").read_text()


def test_BOTH_SIDES_SHARING_A_NETWORK_IS_A_VOID(tmp_path):
    one = ScreenFake()
    code, rec = _screen(tmp_path, screen_seams(stub_play(lambda g, cr: K.CANDIDATE), load_evaluator=lambda: one))
    assert code == P.EXIT_VOID and rec["void"]["step"] == "load"


@pytest.mark.parametrize("kw, step", [({"fail_at": 9}, "exception"), ({"searches": (2, 1)}, "attribution"),
                                      ({"searches": (0, 4)}, "attribution")], ids=["game-fails", "count", "one-side"])
def test_A_GAME_THAT_FAILS_OR_MISCOUNTS_ITS_SEARCHES_LEAVES_NO_SCORE(tmp_path, kw, step):
    code, rec = _screen(tmp_path, screen_seams(stub_play(lambda g, cr: K.CANDIDATE, **kw)))
    assert code == P.EXIT_VOID and rec["void"]["step"] == step, rec.get("void")
    assert rec["verdict"] is None and "screen" not in rec and "pair_scores" not in rec


def test_BOINC_MID_RUN_IS_A_VOID_WITH_NO_SCORE(tmp_path):
    seen = iter([CLEAR] * 6 + [BOINC])
    code, rec = _screen(tmp_path, screen_seams(stub_play(lambda g, cr: K.CANDIDATE), background=lambda: next(seen)))
    assert code == P.EXIT_VOID and rec["void"]["step"] == "background_compute"
    assert "after game 5" in rec["void"]["detail"] and rec["verdict"] is None and "screen" not in rec


def test_THE_REAL_SEAMS_ARE_THE_MATCH_S_LOADER_LOOP_AND_CHECKS():
    """Built, never called with a real checkpoint: the seams name the strength match's loader, loop and between-games
    step, the registry's seed rule, the background check and this module's head install."""
    import inspect
    from scripts.GPU.alphazero import background_compute as BG
    from scripts.GPU.alphazero import mirror_halving_match as MHM
    real = _REAL_SCREEN_SEAMS(pathlib.Path("/nonexistent/model.safetensors"), HEAD)
    assert real.play_and_build is K.play_and_build and real.between_games is MHM._between_games
    assert real.background is BG.observe and real.digests is P.network_digests
    assert "ER._default_evaluator_factory(str(checkpoint))" in inspect.getsource(_REAL_SCREEN_SEAMS)
    assert "install_trained_head(net, _npz_arrays(head[\"bytes\"]))" in inspect.getsource(_REAL_SCREEN_SEAMS)


# ─────────────────────────────── search reaches the residual head through forward_padded (real MLX, CPU) ───────────────

def _tiny(seed=1):
    import mlx.core as mx
    from scripts.GPU.alphazero.network import create_network
    mx.random.seed(seed)
    net = create_network(hidden=128, n_blocks=1, in_channels=30)
    net.eval()
    return net


def _trained_arrays():
    """A stand-in 'trained head' for the tiny parent: arm R installed, every trainable head array moved (extra_out off
    zero), the BatchNorm running statistics kept."""
    from scripts.GPU.alphazero import policy_head_capacity as PHC
    net = _tiny()
    PHC.install(net, "R", init_seed=P.RECIPE["init_seed"])
    rng = np.random.default_rng(5)
    return {k: v if k.endswith(("running_mean", "running_var")) else v + rng.normal(0, 0.05, v.shape).astype(v.dtype)
            for k, v in P.snapshot(net).items() if k.startswith("policy_head.")}


def test_THE_INSTALL_LOADS_EVERY_SAVED_ARRAY_and_leaves_the_encoder_and_value_head_the_parent_s():
    from scripts.GPU.alphazero import policy_head_capacity as PHC
    arrays = _trained_arrays()
    parent = P.network_digests(_tiny())
    net = _tiny()
    got = P.install_trained_head(net, arrays)
    assert type(net.policy_head) is PHC.ResidualPolicyHead and not any(m.training for _, m in net.named_modules())
    assert got == {"encoder.": parent["encoder."], "value_head.": parent["value_head."],
                   "policy_head.": P.digest(arrays, "policy_head.")} and got["policy_head."] != parent["policy_head."]
    snap = P.snapshot(net)
    assert all(np.array_equal(snap[k], v) for k, v in arrays.items())
    with pytest.raises(ValueError, match="installed twice"):
        P.install_trained_head(net, arrays)


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "dtype"])
def test_THE_INSTALL_REFUSES_ARRAYS_THAT_ARE_NOT_EXACTLY_ARM_R_S_HEAD(change):
    arrays = _trained_arrays()
    k = "policy_head.extra_out.weight"
    if change == "missing":
        del arrays[k]
    elif change == "extra":
        arrays["policy_head.spare"] = np.zeros(1, np.float32)
    elif change == "shape":
        arrays[k] = np.zeros((2, 64), np.float32)
    else:
        arrays[k] = arrays[k].astype(np.float64)
    with pytest.raises(P.Void, match="not exactly arm R's head"):
        P.install_trained_head(_tiny(), arrays)


def _opening_inputs():
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    s = TwixtState(active_size=24, to_move="red", max_plies_limit=280)
    legal = s.legal_moves()
    board = np.transpose(s.to_tensor(), (1, 2, 0))[None].astype(np.float32)
    rows = np.array([[m[0] for m in legal]], np.int32)
    cols = np.array([[m[1] for m in legal]], np.int32)
    return board, rows, cols, np.ones(rows.shape, np.float32)


def test_THE_COMPILED_EVALUATOR_REACHES_THE_TRAINED_HEAD_THROUGH_forward_padded():
    """As the match loads it (LocalGPUEvaluator, compile=True): the head installed before the first call, the
    evaluator's priors are the masked softmax of forward_padded's logits WITH the residual path, different from the
    parent's. The probe the run applies sees the same."""
    import mlx.core as mx
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    net = _tiny()
    P.install_trained_head(net, _trained_arrays())
    ev, parent = LocalGPUEvaluator(net, compile=True), LocalGPUEvaluator(_tiny(), compile=True)
    board, rows, cols, mask = _opening_inputs()
    p, _v = ev.infer(board, rows, cols, mask, 24)
    q, _w = parent.infer(board, rows, cols, mask, 24)
    logits = np.array(net.forward_padded(mx.array(board), mx.array(rows), mx.array(cols), mx.array(mask), 24)[0],
                      np.float64)[0]
    direct = np.exp(logits - logits.max())
    assert np.allclose(p[0], direct / direct.sum(), atol=1e-6) and np.max(np.abs(p[0] - q[0])) > 1e-6
    probe = P.opening_probe(ev, parent)
    assert probe["legal"] == rows.shape[1] and probe["max_prior_abs_diff"] > 1e-6


def test_THE_SCREEN_S_AVERAGED_PRIOR_IS_THE_OBJECTIVE_TRAINING_FITTED():
    """The candidate as the screen builds it -- the saved arrays installed, the compiled evaluator, the averaging
    wrapper -- gives exp(log p_bar), training's own prediction (card §4), on positions from several plies."""
    import random
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    net = _tiny()
    P.install_trained_head(net, _trained_arrays())
    avg = MirrorAveragingEvaluator(LocalGPUEvaluator(net, compile=True))
    rng, s, states = random.Random(11), TwixtState(active_size=24, to_move="red", max_plies_limit=280), []
    for ply in range(31):
        if ply in (0, 7, 20, 30):
            states.append(s)
        s = s.apply_move(rng.choice(s.legal_moves()))
    width = max(len(x.legal_moves()) for x in states)
    data = {"boards": np.stack([np.transpose(x.to_tensor(), (1, 2, 0)) for x in states]).astype(np.float32),
            "rows": np.zeros((len(states), width), np.int32), "cols": np.zeros((len(states), width), np.int32),
            "mask": np.zeros((len(states), width), np.float32)}
    for i, x in enumerate(states):
        legal = x.legal_moves()
        data["rows"][i, :len(legal)] = [m[0] for m in legal]
        data["cols"][i, :len(legal)] = [m[1] for m in legal]
        data["mask"][i, :len(legal)] = 1.0
    lp = np.asarray(P.log_averaged(net.policy_head, *P.batch_inputs(net, data, np.arange(len(states)))))
    for i, x in enumerate(states):
        n = len(x.legal_moves())
        p, _ = avg.infer(data["boards"][i:i + 1], data["rows"][i:i + 1, :n], data["cols"][i:i + 1, :n],
                         data["mask"][i:i + 1, :n], 24)
        assert np.allclose(np.exp(lp[i, :n]), p[0], rtol=0, atol=2e-6), i


def test_THE_VALUES_ARE_THE_PARENT_S_BITWISE_UNCOMPILED_and_within_float_rounding_compiled():
    """MEASURED HERE (2026-10-31, tiny CPU networks): uncompiled, the candidate's value equals the parent's bitwise.
    Compiled -- as the match loads both sides -- it may differ in the last bits: the compiled graph is fused
    differently once the residual branch exists (the parent's own compiled and uncompiled values differ by as much).
    That is rounding in the compiled graph, not a changed value path; the run records it (head_probe.value_abs_diff)
    and does not gate on it."""
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    inputs = (*_opening_inputs(), 24)
    v = {}
    for compiled in (False, True):
        net = _tiny()
        P.install_trained_head(net, _trained_arrays())
        v[compiled] = (LocalGPUEvaluator(net, compile=compiled).infer(*inputs)[1],
                       LocalGPUEvaluator(_tiny(), compile=compiled).infer(*inputs)[1])
    assert np.array_equal(*v[False])
    assert np.max(np.abs(v[True][0] - v[True][1])) < 1e-5
    assert np.max(np.abs(v[True][1] - v[False][1])) < 1e-5                # the parent's own compiled rounding


def test_THE_HALVING_SEARCH_OVER_THE_AVERAGING_WRAPPER_CALLS_ONLY_forward_padded_ON_THE_RESIDUAL_HEAD(monkeypatch):
    """Every underlying evaluation of a real halving search at ply 20 (200 simulations, mirror-averaged) goes through
    ResidualPolicyHead.forward_padded, never its __call__ (which refuses); the search returns a halving choice."""
    from scripts.GPU.alphazero import policy_head_capacity as PHC
    import random
    from scripts.GPU.alphazero.external_learning_stage3 import PROTOCOL
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
    seen = {"forward_padded": 0, "call": 0}
    real_fp = PHC.ResidualPolicyHead.forward_padded

    def counted(self, *a):
        seen["forward_padded"] += 1
        return real_fp(self, *a)
    monkeypatch.setattr(PHC.ResidualPolicyHead, "forward_padded", counted)
    net = _tiny()
    P.install_trained_head(net, _trained_arrays())
    base = LocalGPUEvaluator(net)                   # uncompiled, so every call traces through Python
    cost = K.new_cost()
    K.instrument(base, cost)
    rng, s = random.Random(3), TwixtState(active_size=24, to_move="red", max_plies_limit=280)
    for _ in range(20):
        s = s.apply_move(rng.choice(s.legal_moves()))
    search = K.candidate_search(base, ER.EvalConfig(**PROTOCOL), random.Random(4))
    assert type(search) is RSH.RootSequentialHalvingMCTS and type(search.evaluator) is MirrorAveragingEvaluator
    counts, _value, root = search.search_with_root(s, add_noise=False)
    assert search.last_choice is not None and sum(counts.values()) > 0
    assert seen["forward_padded"] == cost["network_infer_calls"] > 0 and cost["network_infer_calls"] % 2 == 0
    with pytest.raises(TypeError, match="not for search"):
        net.policy_head(None)


# ─────────────────────────────── the wrapper's bytes and the proposed ceiling ───────────────────────────────

def _code(path):
    return path.read_text().split("set -u\n", 1)[1]


def _screen_ceiling():
    """The proposed ceiling, from completed match records -- per-side costs and move counts only, never a winner,
    reason or score: strength2's per combination search time, generation's plies per game (this agent against
    itself), 64 games, x 1.5, rounded up to the minute; and the capped games TERM leaves room for, at the slowest
    per-search rate of any 280-ply game in the two completed combination matches."""
    def rows(d):
        return [json.loads(x) for x in (EV / d / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    s2 = [(g["n_moves"], g["cost"]["candidate"]) for g in rows("mirror-halving-strength2")]
    old = [(g["n_moves"], g["cost"]["candidate"]) for g in rows("mirror-halving-screen")]
    gen = [g["n_moves"] for g in rows("combination-policy-generate")]
    searches = sum(c["searches"] for _n, c in s2)
    per = math.fsum(c["search_seconds"] for _n, c in s2) / searches
    plies = sum(gen) / len(gen)
    projected = 64 * plies * per
    ceiling = int(math.ceil(1.5 * projected / 60.0) * 60)
    capped_rate = max(c["search_seconds"] / c["searches"] for n, c in s2 + old if n == 280)
    uncapped = [n for n in gen if n != 280]
    plies_uncapped = sum(uncapped) / len(uncapped)
    room = (ceiling - 10 - 64 * plies_uncapped * per) / (280 * capped_rate - plies_uncapped * per)
    return {"searches": searches, "per": per, "plies": plies, "projected": projected, "ceiling": ceiling,
            "capped_rate": capped_rate, "plies_uncapped": plies_uncapped, "room": math.floor(room),
            "cap_rate": (len(gen) - len(uncapped)) / len(gen)}


def test_THE_PROPOSED_CEILING_IS_1_5_X_THE_COMPLETED_MATCH_PROJECTION_3480_S_with_room_for_10_capped_games():
    c = _screen_ceiling()
    assert c["searches"] == 29453 and round(c["per"], 4) == 0.5508 and round(c["plies"], 2) == 64.75
    assert round(c["projected"], 1) == 2282.3 and c["ceiling"] == 3480
    assert round(c["capped_rate"], 4) == 0.6285 and round(c["plies_uncapped"], 1) == 55.1 and c["room"] == 10
    assert round(c["cap_rate"] * 64, 2) == 2.75                     # capped games expected at generation's rate
    w = _code(WRAPPER)
    term, grace = (int(re.search(rf"^{k}=(\d+)$", w, re.M).group(1)) for k in ("TERM_S", "GRACE_S"))
    assert (term, grace) == (3470, 10) and term + grace == c["ceiling"] and term + 2 * grace == 3490
    head = WRAPPER.read_text().split("set -u\n", 1)[0]
    for said in ("0.5508 s", "29,453", "64.75 plies", "2,282.3 s", "3,480 s", "3490 s", "0.6285 s", "55.1 plies",
                 "10 capped games", "2.75 expected", "4.3%"):
        assert said in head, said


def test_THE_WRAPPER_IS_TRAINING_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """The paths, the gate, `run screen` and TERM_S, line for line; everything else byte-identical. Its header is the
    screen's paragraph followed by training's header unchanged."""
    train, screen = _code(WRAPPER_TRAIN).splitlines(), _code(WRAPPER).splitlines()
    assert len(train) == len(screen)
    changed = [(a, b) for a, b in zip(train, screen) if a != b]
    assert [b for _a, b in changed] == [
        "OUT_REL=docs/superpowers/evidence/combination-policy-screen-run",
        "RUN_REL=docs/superpowers/evidence/combination-policy-screen",
        "GATE_REL=scripts/GPU/alphazero/combination_policy_screen_authorization.py",
        f"GATE={GATE_NAME}", "TERM_S=3470",
        '/usr/bin/perl -e "$SUPERVISOR" "$OUT_REL/pgid.txt" "$TERM_S" "$GRACE_S" env MLX_ENABLE_TF32=0 PYTHONPATH=. '
        '"$PY" -B -m "$MODULE" run screen --checkpoint "$CKPT" \\'], changed
    head_train = WRAPPER_TRAIN.read_text().split("set -u\n", 1)[0]
    head = WRAPPER.read_text().split("set -u\n", 1)[0]
    assert head.startswith("#!/bin/bash\n# THE COMBINATION POLICY PILOT: THE SCREEN, 32 COLOUR-REVERSED PAIRS")
    assert head.endswith(head_train[len("#!/bin/bash\n"):])
    ours = head[:-len(head_train) + len("#!/bin/bash\n")]
    assert "CEILING PROPOSED FOR" in ours and "BUILT ONLY" in ours and "no automatic retry" in ours
    assert os.access(WRAPPER, os.R_OK) and subprocess.run(["bash", "-n", str(WRAPPER)]).returncode == 0


def test_THE_WRAPPER_SERVES_THE_PILOT_S_SCREEN_PATHS():
    w = _code(WRAPPER)
    out, run = (re.search(rf"^{k}=(\S+)$", w, re.M).group(1) for k in ("OUT_REL", "RUN_REL"))
    kind = P.KINDS["screen"]
    assert run == kind["evidence"] and out == str(WRAPPER.parent.relative_to(ROOT)) == \
        str(pathlib.Path(kind["plan"]).parent) == f"{kind['evidence']}-run"
    assert re.search(r"^GATE_REL=(\S+)$", w, re.M).group(1) == \
        str(pathlib.Path(kind["gate"][0].__file__).relative_to(ROOT))
    assert re.search(r"^MODULE=(\S+)$", w, re.M).group(1) == P.__name__ and f"RECORD_KIND={P.RECORD_KIND}\n" in w
    assert "CKPT=\"$MAIN/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors\"" in w
    assert sorted(x.name for x in WRAPPER.parent.iterdir()) == ["01_wrapper.sh"]      # no plan: no seeds yet


# ─────────────────────────────── fake launches of the REAL screen wrapper ───────────────────────────────

STUB_PILOT_SCREEN = STUB_PILOT.replace("combination-policy-throughput", "combination-policy-screen")
assert STUB_PILOT_SCREEN.count("combination-policy-screen") == 2


def _fake_repo(tmp, term_s=None):
    """A throwaway repository: the real screen wrapper (TERM_S changed only when a test says so), its gate opened by a
    one-line commit, and three stubs."""
    repo = tmp / "repo"
    pkg = repo / "scripts/GPU/alphazero"
    pkg.mkdir(parents=True)
    for d in (repo / "scripts", repo / "scripts/GPU", pkg):
        (d / "__init__.py").write_text("")
    gate = pkg / "combination_policy_screen_authorization.py"
    gate.write_text(f'"""stub gate"""\n{GATE_NAME} = False\n')
    (pkg / "idle_gate.py").write_text(STUB_IDLE)
    (pkg / "background_compute.py").write_text(STUB_BG)
    (pkg / "combination_policy_pilot.py").write_text(STUB_PILOT_SCREEN)
    wdir = repo / "docs/superpowers/evidence/combination-policy-screen-run"
    wdir.mkdir(parents=True)
    raw = WRAPPER.read_bytes()
    if term_s is not None:
        assert raw.count(b"\nTERM_S=3470\n") == 1
        raw = raw.replace(b"\nTERM_S=3470\n", f"\nTERM_S={term_s}\n".encode())
    (wdir / "01_wrapper.sh").write_bytes(raw)
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/python").symlink_to(sys.executable)
    ckpt = repo / "checkpoints/alphazero-v2-calib020-from0409"
    ckpt.mkdir(parents=True)
    (ckpt / "model_iter_0001.safetensors").write_bytes(b"not a model")
    (repo / ".gitignore").write_text(".venv/\ncheckpoints/\n__pycache__/\nscratch/\n")
    _git(tmp, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    gate.write_text(gate.read_text().replace(f"{GATE_NAME} = False", f"{GATE_NAME} = True"))
    _git(repo, "commit", "-qam", "open the gate")
    (repo / "scratch").mkdir()
    return repo, wdir, gate


def _launch(repo, wdir, idle="idle", bg="clear", run="ok", timeout=60):
    env = {k: v for k, v in os.environ.items() if k not in ("MLX_ENABLE_TF32", "PYTHONPATH")}
    env.update(FAKE_IDLE=idle, FAKE_BG=bg, FAKE_RUN=run, FAKE_OUT=str(repo / "scratch"), PYTHONDONTWRITEBYTECODE="1",
               FAKE_CARD_SHA=P.CARD_SHA256)
    return subprocess.run(["bash", str(wdir / "01_wrapper.sh")], cwd=str(repo), env=env, capture_output=True,
                          text=True, timeout=timeout)


@pytest.mark.parametrize("idle, bg", [("busy", "clear"), ("idle", "boinc"), ("idle", "failed")],
                         ids=["busy", "boinc", "check-fails"])
def test_FAKE_LAUNCH_A_BUSY_MACHINE_BOINC_OR_A_FAILED_CHECK_PREVENTS_THE_LAUNCH(tmp_path, idle, bg):
    repo, wdir, gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir, idle, bg)
    assert p.returncode == 1 and ": nothing launched" in p.stderr, p.stderr
    assert "FAKE_LAUNCHED" not in p.stdout + p.stderr
    assert sorted(x.name for x in wdir.iterdir()) == ["01_wrapper.sh"]
    assert not (repo / "docs/superpowers/evidence/combination-policy-screen").exists()
    assert f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_A_GATE_OPENED_WITH_ANY_OTHER_CHANGE_IS_REFUSED(tmp_path):
    repo, wdir, gate = _fake_repo(tmp_path)
    _git(repo, "reset", "-q", "--soft", "HEAD~1")
    (repo / "scripts/GPU/alphazero/idle_gate.py").write_text(STUB_IDLE + "\n")
    _git(repo, "commit", "-qam", "open the gate and change something else")
    p = _launch(repo, wdir)
    assert p.returncode == 1 and "does not change only" in p.stderr, p.stderr
    assert "FAKE_LAUNCHED" not in p.stdout + p.stderr and f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_AN_IDLE_MACHINE_REACHES_RUN_SCREEN_with_its_exact_arguments_and_ceiling(tmp_path):
    repo, wdir, gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir, run="record")
    assert p.returncode == 0, p.stderr
    launched = json.loads((wdir / "run.stdout.txt").read_text().split("FAKE_LAUNCHED", 1)[1])
    ckpt = str(repo.resolve() / "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors")
    assert launched["argv"] == ["run", "screen", "--checkpoint", ckpt] and launched["tf32"] == "0"
    assert (wdir / "pgid.txt").read_text() == f"{launched['pgid']}\n"
    outer = json.loads((wdir / "outer.json").read_text())
    assert outer["outcome"] == "COMPLETED" and outer["survivors"] == "" and outer["gate_restored"] is True
    assert outer["term_s"] == 3470 and outer["grace_s"] == 10 and outer["record_by"] == "script"
    rec = json.loads((repo / "docs/superpowers/evidence/combination-policy-screen/record.json").read_text())
    assert not P.outer_faults(outer, rec)
    assert f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_THE_CEILING_TERMINATES_THE_WHOLE_GROUP_at_the_screen_s_paths(tmp_path):
    """TERM_S changed to 2 in this copy only: the run and its child outlive it; the group TERM ends both, and the
    wrapper's VOID record lands in the screen's evidence directory."""
    repo, wdir, gate = _fake_repo(tmp_path, term_s=2)
    t = time.monotonic()
    p = _launch(repo, wdir, run="outlive")
    child = int((repo / "scratch/child.pid").read_text())
    assert time.monotonic() - t < 30 and p.returncode == 124, p.stderr
    for _ in range(50):
        if not _alive(child):
            break
        time.sleep(0.1)
    assert not _alive(child)
    outer = json.loads((wdir / "outer.json").read_text())
    assert outer["outcome"] == "TIMEOUT" and outer["inner_exit"] == 124 and outer["survivors"] == ""
    rec = json.loads((repo / "docs/superpowers/evidence/combination-policy-screen/record.json").read_text())
    assert outer["record_by"] == "wrapper" and rec["status"] == "VOID" and rec["void"]["step"] == "timeout"
    assert f"{GATE_NAME} = False\n" in gate.read_text()


# ─────────────────────────────── the seed survey copy (built, never run on the repository) ───────────────────────────────

def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, EV / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # module level only: no mode runs, nothing is written
    return mod


def _split(rel):
    text = (EV / rel).read_text()
    end = text.index('"""', 3)
    return text[3:end], text[end + 3:]


def test_THE_SURVEY_COPY_IS_GENERATION_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """Below the docstring: the block size, paths, plan kind, module name, titles and verdict lines, the slot
    listing's heading, and the one control's expected value -- nothing else. The docstring is the screen's paragraph
    followed by generation's description unchanged."""
    doc_gen, code_gen = _split(SURVEY_GEN)
    doc, code = _split(SURVEY)
    assert doc.endswith(doc_gen) and "§5 step 5" in doc[:-len(doc_gen)] and "ONE CONTROL'S EXPECTED VALUE" in doc
    a, b = code_gen.splitlines(), code.splitlines()
    removed, added = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            removed += [x.strip() for x in a[i1:i2]]
            added += [x.strip() for x in b[j1:j2]]
    assert removed == [
        "N_SLOTS = 1000", 'HERE = "docs/superpowers/evidence/combination-policy-generate-seed-survey"',
        'PLAN = "docs/superpowers/evidence/combination-policy-generate-run/plan.json"',
        'VERDICT_OK = "THE COMBINATION-POLICY GENERATION BLOCK IS CLEAN"',
        'spec = importlib.util.spec_from_file_location(f"_combination_policy_generate_{key}", p)',
        'return {"kind": "combination-policy-generate-plan/1", "base": base, "n_slots": N_SLOTS,',
        'faults.append("the generation plan is absent or not committed")',
        'faults.append("the generation plan is not build_plan of the surveyed block")',
        'SEL.candidate([], [], N_SLOTS) == (None, e + N_SLOTS, (e + N_SLOTS, e + 2 * N_SLOTS)), ""))',
        "say(f\"COMBINATION POLICY PILOT -- GENERATION GAME-SEED {'SURVEY' if mode == 'survey' else "
        "'COLLISION PROOF, mode ' + mode.upper()}\")",
        'say("SLOTS (combination_selfplay.generate: slot g plays seed BASE + g, whichever worker claims it):")',
        "say(f\"PROOF VERDICT ({mode}): {'THE COMBINATION-POLICY GENERATION BLOCK IS CLEAN AND THE STATE IS AS "
        "EXPECTED' if ok else 'NOT PROVEN -- STOP'}\""], removed
    assert added == [
        "N_SLOTS = 64", 'HERE = "docs/superpowers/evidence/combination-policy-screen-seed-survey"',
        'PLAN = "docs/superpowers/evidence/combination-policy-screen-run/plan.json"',
        'VERDICT_OK = "THE COMBINATION-POLICY SCREEN BLOCK IS CLEAN"',
        'spec = importlib.util.spec_from_file_location(f"_combination_policy_screen_{key}", p)',
        'return {"kind": "combination-policy-screen-plan/1", "base": base, "n_slots": N_SLOTS,',
        'faults.append("the screen plan is absent or not committed")',
        'faults.append("the screen plan is not build_plan of the surveyed block")',
        "b0 = next(b for b in range(e + N_SLOTS, e + N_SLOTS + SEL.ALIGN) if b % SEL.ALIGN == 0)",
        'SEL.candidate([], [], N_SLOTS) == (None, b0, (b0, b0 + N_SLOTS)), ""))',
        "say(f\"COMBINATION POLICY PILOT -- SCREEN GAME-SEED {'SURVEY' if mode == 'survey' else "
        "'COLLISION PROOF, mode ' + mode.upper()}\")",
        'say("SLOTS (combination_policy_pilot run screen: game g plays seed BASE + g, in order, in one process):")',
        "say(f\"PROOF VERDICT ({mode}): {'THE COMBINATION-POLICY SCREEN BLOCK IS CLEAN AND THE STATE IS AS "
        "EXPECTED' if ok else 'NOT PROVEN -- STOP'}\""], added


def test_THE_SURVEY_COPY_SERVES_THE_SCREEN_IN_THE_2027_BAND_and_holds_no_seed_literal():
    from scripts.GPU.alphazero import seed_band_2027 as SB
    survey = _load(SURVEY, "_screen_survey")
    assert survey.N_SLOTS == 2 * P.KINDS["screen"]["pairs"] == 64 and survey.PLAN == P.KINDS["screen"]["plan"]
    S, _Pf, SEL = survey.load_method()
    assert S.N == 64 and S.scan is SEL.scan and SEL.__name__.endswith("seed_band_2027")
    assert survey.PINNED == _load(SURVEY_GEN, "_gen_survey_pins").PINNED
    base = SEL.candidate([], [], 64)[1]
    assert survey.build_plan(base, "0" * 64) == {"kind": "combination-policy-screen-plan/1", "base": base,
                                                 "n_slots": 64, "survey_record_sha256": "0" * 64}
    assert set(survey.OUT.values()) & set(_load(SURVEY_GEN, "_gen_survey_out").OUT.values()) == set()
    assert not any((ROOT / p).exists() for p in survey.OUT.values())            # never run
    code = _split(SURVEY)[1]
    assert not SB.scan(code) and not re.search(r"\b20[0-9]{7}\b", code)


def test_THE_SURVEY_COPY_S_CONTROLS_ALL_PASS_on_a_synthetic_2027_block():
    """The copied controls (state 12, plants 8, exemptions 6, enumeration 6) and the two-band controls (10), with the
    real prior terms and a 64-seed block the rule places in an empty band; the repository is not scanned here."""
    survey = _load(SURVEY, "_screen_survey_controls")
    S, Pf, SEL = survey.load_method()
    registries, named, gen, masks, consumed = S.prior_terms()
    block = SEL.candidate([], [], 64)[2]
    prior = (registries, named, gen, consumed, {})
    rec = {"head": "0" * 40, "registry_sha256": "0" * 64, "base": block[0], "block": block}
    groups = {"state": survey._state_controls(S, Pf, rec, "a" * 64),
              "plants": S._plant_controls(block, prior, masks, {}, True),
              "exemptions": S._exemption_controls(block), "enumeration": Pf._enumeration_controls(S),
              "two-band": survey._two_band_controls(S, SEL, block, prior, masks)}
    assert {k: len(v) for k, v in groups.items()} == {"state": 12, "plants": 8, "exemptions": 6, "enumeration": 6,
                                                      "two-band": 10}
    for k, v in groups.items():
        assert all(ok for _l, ok, _d in v), (k, [r for r in v if not r[1]])


def test_THE_COPIED_EMPTY_BAND_CONTROL_WOULD_FAIL_AT_64_SEEDS_so_the_corrected_one_is_needed():
    """Necessity of the one changed control: generation's expected value (the edge + N) is not the rule's answer for a
    block size that is not a multiple of 100; the corrected expectation is."""
    from scripts.GPU.alphazero import seed_band_2027 as SB
    e = SB.BAND[0]
    got = SB.candidate([], [], 64)
    assert got != (None, e + 64, (e + 64, e + 128)) and got[1] % SB.ALIGN == 0 and e + 64 <= got[1] < e + 64 + SB.ALIGN
    assert SB.candidate([], [], 1000) == (None, e + 1000, (e + 1000, e + 2000))      # where the copied form held


PINNED_FILES = ("docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py",
                "docs/superpowers/evidence/tactical-candidate-seed-registration/01_collision_proof.py",
                "scripts/GPU/alphazero/injected_defect_controls.py", "scripts/GPU/alphazero/seed_band_2027.py")


def _mode(repo, mode, monkeypatch, capsys):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location(f"_screen_survey_{mode}", repo / "docs/superpowers/evidence" / SURVEY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    code = mod.main(["x", "--mode", mode])
    capsys.readouterr()
    return mod, code, (repo / mod.OUT[mode]).read_text(encoding="utf-8")


def test_THE_THREE_MODES_REHEARSED_ON_A_SYNTHETIC_REPOSITORY(tmp_path, monkeypatch, capsys):
    """survey -> pre -> registration -> post, on a throwaway git repository holding only the copy, the pinned method,
    a synthetic registry and two planted literals (one per band): the rule places the 64-seed block above the 2027
    literal; every mode's controls pass and each verdict holds. Nothing of the real repository is scanned or written."""
    from scripts.GPU.alphazero import e4_screen_reference as REG
    from scripts.GPU.alphazero import seed_band_2027 as SB
    repo = tmp_path / "synthetic"

    def git(*a):
        return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo,
                                       text=True).strip()
    for rel in (*PINNED_FILES, f"docs/superpowers/evidence/{SURVEY}"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes((ROOT / rel).read_bytes())
    top = max([SB.BAND[0]] + [hi for lo, hi in REG.ACCOUNTED_SEED_INTERVALS if hi > SB.BAND[0]])
    old, new = SB.BAND_2026[0] + 61_000, top + 23_456
    reg = repo / "scripts/GPU/alphazero/e4_screen_reference.py"
    reg.write_text(f"ACCOUNTED_SEED_INTERVALS = (\n    ({old}, {old + 10}),\n)\nEXPOSED_SEED_INTERVALS = ()\n"
                   "RETIRED_SEED_INTERVALS = ()\nTEST_ONLY_SEED_INTERVALS = ()\n")
    (repo / "tests").mkdir()
    (repo / "tests/test_planted.py").write_text(f"A = {old + 500}\nB = {new:_}\n")
    (repo / "docs/superpowers/evidence/combination-policy-screen-run").mkdir(parents=True)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "synthetic")

    mod, code, survey = _mode(repo, "survey", monkeypatch, capsys)
    assert code == 0, survey[-2500:]
    anchor, base, block = SB.candidate(REG.ACCOUNTED_SEED_INTERVALS, [new], 64)
    assert anchor == new and block[1] - block[0] == 64
    assert f"in the 2027 band = {new}; BASE = {base}; block = [{block[0]}, {block[1]}) -- 64 seeds" in survey
    assert survey.endswith("SURVEY VERDICT   : THE COMBINATION-POLICY SCREEN BLOCK IS CLEAN\n")
    passed, total = re.search(r"^controls passed: (\d+)/(\d+)$", survey, re.M).groups()
    assert passed == total and int(total) == 55, total
    S, _Pf, SEL = mod.load_method()
    assert mod.read_record(survey, S, SEL)["block"] == block
    git("add", "-A")
    git("commit", "-qm", "survey record")

    _m, code, pre = _mode(repo, "pre", monkeypatch, capsys)
    assert code == 0 and "EXPECTED STATE (pre): HOLDS" in pre, pre[-2500:]
    git("add", "-A")
    git("commit", "-qm", "pre record")

    sha = hashlib.sha256((repo / mod.OUT["survey"]).read_bytes()).hexdigest()
    reg.write_text(reg.read_text().replace(f"    ({old}, {old + 10}),\n)",
                                           f"    ({old}, {old + 10}),\n    ({block[0]}, {block[1]}),\n)"))
    (repo / mod.PLAN).write_text(json.dumps(mod.build_plan(block[0], sha), indent=1, sort_keys=True) + "\n")
    git("add", "-A")
    git("commit", "-qm", "registration")
    _m, code, post = _mode(repo, "post", monkeypatch, capsys)
    assert code == 0 and "EXPECTED STATE (post): HOLDS" in post, post[-2500:]
    assert post.endswith("PROOF VERDICT (post): THE COMBINATION-POLICY SCREEN BLOCK IS CLEAN AND THE STATE IS AS "
                         "EXPECTED\n")


# ─────────────────────────────── the seed closing copy (built, never run here) ───────────────────────────────

def test_THE_CLOSING_COPY_IS_STRENGTH2_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """Below the docstring: the paths, the block size and the title, line for line -- the sequential match's rule
    (started games a prefix 0..k-1 in order) is untouched. The docstring is the screen's paragraph followed by
    strength2's description unchanged."""
    doc2, code2 = _split(CLOSING_S2)
    doc, code = _split(CLOSING)
    assert doc.endswith(doc2) and "§5 step 5" in doc[:-len(doc2)] and "IN ORDER IN ONE PROCESS" in doc
    a, b = code2.splitlines(), code.splitlines()
    assert len(a) == len(b)
    assert [y.strip() for x, y in zip(a, b) if x != y] == [
        'HERE = "docs/superpowers/evidence/combination-policy-screen-seed-closing"',
        'RUN = "docs/superpowers/evidence/combination-policy-screen"',
        'RUN_DIR = "docs/superpowers/evidence/combination-policy-screen-run"', "N_SLOTS = 64",
        'lines = ["COMBINATION POLICY PILOT -- SCREEN SEED-USAGE DERIVATION (closing)", "=" * 96,']


def test_THE_CLOSING_COPY_SERVES_THE_SCREEN_and_holds_no_seed_literal():
    from scripts.GPU.alphazero import seed_band_2027 as SB
    closing = _load(CLOSING, "_screen_closing")
    kind = P.KINDS["screen"]
    assert closing.N_SLOTS == 2 * kind["pairs"] == 64 and closing.INPUTS[0] == kind["plan"]
    assert closing.RUN == kind["evidence"] and closing.RUN_DIR == str(pathlib.Path(kind["plan"]).parent)
    assert closing.OUT == "docs/superpowers/evidence/combination-policy-screen-seed-closing/02_derive_usage_run.txt"
    assert not (ROOT / closing.OUT).exists()
    code = _split(CLOSING)[1]
    assert not SB.scan(code) and not re.search(r"\b20[0-9]{7}\b", code)


def test_THE_CLOSING_COPY_S_CONTROLS_ALL_PASS_at_a_synthetic_base():
    results = _load(CLOSING, "_screen_closing_c")._controls(BASE)
    assert len(results) == 15 and all(ok for _l, ok, _d in results), [r for r in results if not r[1]]


def test_THE_CLOSING_RECONCILES_THE_SCREEN_S_OWN_EVENTS_completed_and_VOID_mid_run(tmp_path):
    """The screen's progress.jsonl, from the real record boundary on fakes, read by the closing's rule: a completed
    run exposes and retires the whole block; a run VOID in game 9 exposes games 0-9 only and retires the whole block."""
    closing = _load(CLOSING, "_screen_closing_r")
    plan = {"base": BASE, "n_slots": 64}
    regs = {"ACCOUNTED": [(BASE, BASE + 64)], "EXPOSED": [], "RETIRED": []}
    _screen(tmp_path / "a", screen_seams(stub_play(lambda g, cr: K.CANDIDATE)))
    faults, expose, retire = closing.reconcile(plan, _events(tmp_path / "a"), {"outer_exit": 0, "survivors": ""},
                                               "COMPLETED", regs)
    assert (faults, expose, retire) == ([], (BASE, BASE + 64), (BASE, BASE + 64))
    _screen(tmp_path / "b", screen_seams(stub_play(lambda g, cr: K.CANDIDATE, fail_at=9)))
    faults, expose, retire = closing.reconcile(plan, _events(tmp_path / "b"), {"outer_exit": 3, "survivors": ""},
                                               "VOID", regs)
    assert (faults, expose, retire) == ([], (BASE, BASE + 10), (BASE, BASE + 64))
