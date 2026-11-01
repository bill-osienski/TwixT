"""The combination policy pilot's STRENGTH COMPARISON (card
`docs/superpowers/2026-10-28-combination-policy-pilot-card.md` §5 step 6), built only on the user's review (ledger,
2026-11-01): `combination_policy_pilot run strength` -- the screen's reviewed path on 509 pairs with the frozen
strength rule -- its wrapper (`docs/superpowers/evidence/combination-policy-strength-run/01_wrapper.sh`), its ceiling
(51,780 s, the user's) and its seed copies.

CPU only. No test loads calib020_0001 or the trained head for evaluation, touches the GPU, opens a real gate, scans the
real repository for a survey, writes the registry or plays a real game:
* admission and the head binding run on the REAL committed training and screen records (read-only) and on synthetic
  ones;
* the comparison runs through the real record boundary with the screen tests' GPU-free fakes and stubbed games;
* the wrapper runs in a throwaway repository with stubs; the seed copies' modes run on a synthetic repository.
The screen's own checks of the head install, forward_padded and the compiled prior
(tests/test_combination_policy_screen.py) cover the path both kinds share; they are not repeated here.

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

import numpy as np
import pytest

from scripts.GPU.alphazero import combination_policy_pilot as P
from scripts.GPU.alphazero import mirror_halving_combination as K
from tests.test_combination_policy_launch import STUB_BG, STUB_IDLE, STUB_PILOT, _alive, _git
from tests.test_combination_policy_screen import (BASE, BOINC, CLEAR, HEAD, HEAD_D, PARENT_D, ScreenFake, _wins_first,
                                                  screen_seams, stub_play)

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = ROOT / "docs/superpowers/evidence"
WRAPPER_SCREEN = EV / "combination-policy-screen-run/01_wrapper.sh"
WRAPPER = EV / "combination-policy-strength-run/01_wrapper.sh"
GATE_NAME = "COMBINATION_POLICY_STRENGTH_AUTHORIZED"
SURVEY_SCREEN = "combination-policy-screen-seed-survey/01_seed_survey.py"
SURVEY = "combination-policy-strength-seed-survey/01_seed_survey.py"
CLOSING_SCREEN = "combination-policy-screen-seed-closing/01_derive_usage.py"
CLOSING = "combination-policy-strength-seed-closing/01_derive_usage.py"
TRAIN, SCREEN = P.KINDS["train"]["evidence"], P.KINDS["screen"]["evidence"]
TRAIN_RUN = "docs/superpowers/evidence/combination-policy-train-run"
SCREEN_RUN = "docs/superpowers/evidence/combination-policy-screen-run"
GAMES = 1018
SCREEN_PLAN = {"base": BASE - 1000, "n_slots": 64}    # a fake screen block below the fake strength block


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


def test_THE_ISOLATION_IS_IN_FORCE():
    with pytest.raises(AssertionError, match="real seam"):
        P._real_screen_seams(pathlib.Path("/nonexistent"), {})
    with pytest.raises(AssertionError, match="real seam"):
        P.conclude("strength", None, pathlib.Path("/nonexistent"), {})


def test_THE_COMPARISON_S_BLOCK_IS_SURVEYED_PROVEN_AND_REGISTERED_NOTHING_HAS_RUN():
    """RE-ANCHORED 2026-11-01 (survey 11e3e485, pre-proof fa9cd930, registration ea2376d7, post-proof 10e44eb9): the
    survey is CLEAN and both proofs HOLD, each 55/55 on a clean tree, the proofs bound to the survey record's sha256;
    the plan is the copy's build_plan; the block (read from the plan and the records -- no seed literal here, since
    the proofs scan tests/) lies in the 2027 band above the screen's, with the gap at least its own size, ACCOUNTED
    exactly once and last, in no other registry, every seed schedulable, disjoint from the screen's block; the
    registry the post-proof proved is the current one. Nothing has run: the gate is closed, the run directory holds
    the wrapper and the plan only, and no run evidence or closing record exists. Read-only."""
    from scripts.GPU.alphazero import combination_policy_strength_authorization as G
    from scripts.GPU.alphazero import e4_screen_reference as REG
    from scripts.GPU.alphazero import seed_band_2027 as SB
    kind = P.KINDS["strength"]
    assert getattr(G, GATE_NAME) is False
    assert sorted(x.name for x in WRAPPER.parent.iterdir()) == ["01_wrapper.sh", "plan.json"]
    assert not (ROOT / kind["evidence"]).exists()
    here = (EV / SURVEY).parent
    assert sorted(x.name for x in here.iterdir() if x.name != "__pycache__") == \
        ["01_seed_survey.py", "02_seed_survey_run.txt", "03_collision_proof_pre_run.txt",
         "04_collision_proof_post_run.txt"]
    assert sorted(x.name for x in (EV / CLOSING).parent.iterdir() if x.name != "__pycache__") == ["01_derive_usage.py"]
    raw = (here / "02_seed_survey_run.txt").read_bytes()
    survey, sha = raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()
    assert survey.splitlines()[2].endswith("; working tree clean") and "controls passed: 55/55" in survey
    assert survey.endswith("SURVEY VERDICT   : THE COMBINATION-POLICY STRENGTH BLOCK IS CLEAN\n")
    for rel, mode in (("03_collision_proof_pre_run.txt", "pre"), ("04_collision_proof_post_run.txt", "post")):
        text = (here / rel).read_text(encoding="utf-8")
        assert text.splitlines()[2].endswith("; working tree clean")
        assert f"02_seed_survey_run.txt sha256 {sha} (read at " in text
        assert f"EXPECTED STATE ({mode}): HOLDS" in text and "controls passed: 55/55" in text
        assert text.endswith(f"PROOF VERDICT ({mode}): THE COMBINATION-POLICY STRENGTH BLOCK IS CLEAN AND THE STATE "
                             "IS AS EXPECTED\n")
    post = (here / "04_collision_proof_post_run.txt").read_text(encoding="utf-8")
    reg_rel = "scripts/GPU/alphazero/e4_screen_reference.py"
    assert f"\nregistry {reg_rel} sha256 {hashlib.sha256((ROOT / reg_rel).read_bytes()).hexdigest()}\n" in post
    plan = json.loads((ROOT / kind["plan"]).read_text(encoding="utf-8"))
    assert type(plan["base"]) is int and plan["n_slots"] == GAMES
    assert plan == _load(SURVEY, "_strength_survey_registered").build_plan(plan["base"], sha)
    block = (plan["base"], plan["base"] + GAMES)
    assert SB.BAND[0] <= block[0] < block[1] <= SB.BAND[1] and block[0] % SB.ALIGN == 0
    assert f"block = [{block[0]}, {block[1]}) -- {GAMES} seeds" in survey
    assert f"THE BLOCK, read from the record: [{block[0]}, {block[1]}) -- BASE {block[0]}" in post
    gap = int(re.search(rf"^THE BLOCK \({block[0]}, {block[1]}\) .* gap=(\d+) \(floor {GAMES}\)", post, re.M).group(1))
    assert gap >= GAMES
    assert REG.ACCOUNTED_SEED_INTERVALS.count(block) == 1 and REG.ACCOUNTED_SEED_INTERVALS[-1] == block
    assert not any(lo < block[1] and block[0] < hi
                   for lo, hi in REG.EXPOSED_SEED_INTERVALS + REG.RETIRED_SEED_INTERVALS + REG.TEST_ONLY_SEED_INTERVALS)
    screen = json.loads((ROOT / P.KINDS["screen"]["plan"]).read_text(encoding="utf-8"))
    screen_block = (screen["base"], screen["base"] + screen["n_slots"])
    gen = json.loads((ROOT / P.KINDS["generate"]["plan"]).read_text(encoding="utf-8"))
    gen_block = (gen["base"], gen["base"] + gen["n_slots"])
    assert screen_block[1] <= block[0]
    assert [iv for iv in REG.ACCOUNTED_SEED_INTERVALS if iv[1] > SB.BAND[0]] == [gen_block, screen_block, block]
    assert all(REG.seed_is_accounted(s) and not REG.seed_is_unavailable(s) and not REG.seed_is_test_only(s)
               for s in range(*block))


# ─────────────────────────────── the gate, the kind, the rule and admission ───────────────────────────────

def test_THE_STRENGTH_GATE_IS_A_CLOSED_SINGLE_DECLARATION_and_the_closed_gate_refuses(monkeypatch, capsys):
    import ast
    from scripts.GPU.alphazero import combination_policy_strength_authorization as G
    gate, name = P.KINDS["strength"]["gate"]
    assert gate is G and name == GATE_NAME and getattr(G, GATE_NAME) is False
    tree = ast.parse(pathlib.Path(G.__file__).read_text())
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert len(body) == 1 and body[0].targets[0].id == GATE_NAME
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    assert P.main(["run", "strength", "--checkpoint", "/x"]) == P.EXIT_REFUSED
    assert "is closed" in capsys.readouterr().err


def test_THE_COMPARISON_IS_509_PAIRS_ON_ITS_OWN_PATHS():
    s = P.KINDS["strength"]
    assert s["pairs"] == K.STRENGTH["pairs"] == 509 and 2 * s["pairs"] == GAMES
    assert s["evidence"] == "docs/superpowers/evidence/combination-policy-strength"
    assert s["plan"] == "docs/superpowers/evidence/combination-policy-strength-run/plan.json"
    others = {P.KINDS[k]["evidence"] for k in P.KINDS if k != "strength"}
    others |= {str(pathlib.Path(v["plan"]).parent) for k, v in P.KINDS.items() if k != "strength" and "plan" in v}
    assert {s["evidence"], str(pathlib.Path(s["plan"]).parent)} & others == set()
    assert "amendment" not in s and "slots" not in s and not (ROOT / s["evidence"]).exists()
    assert sorted(x.name for x in WRAPPER.parent.iterdir()) == ["01_wrapper.sh", "plan.json"]  # the seeds' plan only


def test_THE_RULE_IS_THE_CARD_S_EXACTLY():
    """PASS iff mean - sqrt(ln 40 / 1018) > 0.5 AND evaluations <= 1.05 AND search seconds <= 1.10 (card §5 step 6;
    the user's review, 2026-11-01): the frozen combination module's own constants and verdict, unchanged."""
    assert K.STRENGTH == {"pairs": 509, "alpha": 0.05, "null": 0.5} and K.GUARDS == {"evaluations": 1.05,
                                                                                     "search_seconds": 1.10}
    assert math.log(2 / K.STRENGTH["alpha"]) == math.log(40) and 2 * K.STRENGTH["pairs"] == 1018
    pairs = [1.0] * 300 + [0.0] * 209
    assert K.hoeffding_lower(pairs, 0.05) == math.fsum(pairs) / 509 - math.sqrt(math.log(40) / 1018)
    assert P.M._sha((ROOT / "scripts/GPU/alphazero/mirror_halving_combination.py").read_bytes()) == \
        P.PINNED["scripts/GPU/alphazero/mirror_halving_combination.py"]


def test_ADMISSION_IS_TRAINING_S_AND_THE_SCREEN_S_RECORDS_AND_OUTER_JSONS():
    assert P.ADMISSION["strength"] == [(TRAIN, "verdict", "ELIGIBLE"), (SCREEN, "verdict", "CONTINUE")]
    assert P.OUTER_ADMISSION["strength"] == [(TRAIN_RUN, TRAIN), (SCREEN_RUN, SCREEN)]
    assert P.ADMISSION["screen"] == [(TRAIN, "verdict", "ELIGIBLE")]                     # the screen's unchanged
    assert P.OUTER_ADMISSION["screen"] == [(TRAIN_RUN, TRAIN)]


def test_REAL_STATE_THE_COMMITTED_RECORDS_ADMIT_THE_COMPARISON_AND_BIND_THE_SCREEN_S_HEAD():
    """Read-only, through the real committed readers: training's record (ELIGIBLE) and the screen's (CONTINUE), with
    both outer.json files, admit the comparison; the binding is head.npz's committed bytes, and the screen's record
    played exactly that head on calib020_0001."""
    assert P.admitted("strength", P._read_record, P._read_outer) == {
        TRAIN: "ELIGIBLE", SCREEN: "CONTINUE", f"{TRAIN_RUN}/outer.json": "COMPLETED",
        f"{SCREEN_RUN}/outer.json": "COMPLETED"}
    h = P.trained_head(P._read_record, P._read_committed_bytes, "strength")
    raw = (ROOT / P.TRAINED_HEAD["file"]).read_bytes()
    assert h["bytes"] == raw and hashlib.sha256(raw).hexdigest() == h["sha256"] == P.TRAINED_HEAD["sha256"]
    screen = json.loads((ROOT / SCREEN / "record.json").read_text(encoding="utf-8"))
    assert screen["trained_head"] == {k: h[k] for k in ("file", "sha256", "digests")}
    assert screen["digests"][K.CANDIDATE] == h["digests"] and screen["checkpoint_sha1"] == P.M.CHECKPOINT_SHA1


GOOD_TRAIN = {"run": "train", "status": "COMPLETED", "verdict": "ELIGIBLE", "card_sha256": P.CARD_SHA256,
              "head": "a" * 40, "head_file": "head.npz", "head_sha256": P.TRAINED_HEAD["sha256"],
              "digests_before": {**HEAD_D, "policy_head.": "before"}, "digests_after": dict(HEAD_D)}
GOOD_SCREEN = {"run": "screen", "status": "COMPLETED", "verdict": "CONTINUE", "card_sha256": P.CARD_SHA256,
               "head": "c" * 40, "checkpoint_sha1": P.M.CHECKPOINT_SHA1,
               "trained_head": {k: P.TRAINED_HEAD[k] for k in ("file", "sha256", "digests")},
               "digests": {K.CANDIDATE: dict(HEAD_D), P.REFERENCE: dict(PARENT_D)}}
GOOD_OUTER = {"inner_exit": 0, "outcome": "COMPLETED", "outer_exit": 0, "gate_restored": True, "survivors": "",
              "record_by": "script"}


def _readers(train=None, screen=None, train_outer=None, screen_outer=None, data=None):
    recs = {TRAIN: GOOD_TRAIN if train is None else train, SCREEN: GOOD_SCREEN if screen is None else screen}
    outers = {TRAIN_RUN: {**GOOD_OUTER, "head": "a" * 40} if train_outer is None else train_outer,
              SCREEN_RUN: {**GOOD_OUTER, "head": "c" * 40} if screen_outer is None else screen_outer}

    def read_record(rel):
        if recs[rel] is False:
            raise FileNotFoundError(rel)
        return recs[rel]

    def read_outer(rel):
        if outers[rel] is False:
            raise FileNotFoundError(rel)
        return outers[rel]

    def read_bytes(rel):
        assert rel == P.TRAINED_HEAD["file"], rel
        return (ROOT / rel).read_bytes() if data is None else data
    return read_record, read_outer, read_bytes


def test_ADMISSION_NEEDS_THE_SCREEN_S_CONTINUE_AND_ITS_SUCCESSFUL_OUTER_JSON_as_well_as_training_s():
    rr, ro, _rb = _readers()
    assert P.admitted("strength", rr, ro) == {TRAIN: "ELIGIBLE", SCREEN: "CONTINUE",
                                              f"{TRAIN_RUN}/outer.json": "COMPLETED",
                                              f"{SCREEN_RUN}/outer.json": "COMPLETED"}
    with pytest.raises(P.Refused, match="not COMPLETED with verdict ELIGIBLE"):
        P.admitted("strength", *_readers(train={**GOOD_TRAIN, "verdict": "STOP"})[:2])
    for change in ({"verdict": "STOP"}, {"status": "VOID"}, {"card_sha256": "0" * 64}):
        with pytest.raises(P.Refused, match="combination-policy-screen/record.json is not COMPLETED with verdict "
                                            "CONTINUE"):
            P.admitted("strength", *_readers(screen={**GOOD_SCREEN, **change})[:2])
    with pytest.raises(P.Refused, match="combination-policy-screen/record.json does not exist"):
        P.admitted("strength", *_readers(screen=False)[:2])
    with pytest.raises(P.Refused, match="combination-policy-screen-run/outer.json does not exist"):
        P.admitted("strength", *_readers(screen_outer=False)[:2])
    for change in ({"outcome": "TIMEOUT", "outer_exit": 124, "inner_exit": 124}, {"survivors": "123 "},
                   {"gate_restored": False}, {"record_by": "wrapper"}, {"head": "b" * 40}):
        with pytest.raises(P.Refused, match="combination-policy-screen-run/outer.json does not show a successful run"):
            P.admitted("strength", *_readers(screen_outer={**GOOD_OUTER, "head": "c" * 40, **change})[:2])
    with pytest.raises(P.Refused, match="combination-policy-train-run/outer.json does not show a successful run"):
        P.admitted("strength", *_readers(train_outer={**GOOD_OUTER, "head": "a" * 40, "survivors": "9 "})[:2])


@pytest.mark.parametrize("screen, why", [
    ({"trained_head": {**GOOD_SCREEN["trained_head"], "sha256": "0" * 64}}, "did not play the pinned head"),
    ({"trained_head": None}, "did not play the pinned head"),
    ({"digests": {K.CANDIDATE: dict(PARENT_D), P.REFERENCE: dict(PARENT_D)}}, "did not play the pinned head"),
    ({"checkpoint_sha1": "0" * 40}, "did not play the pinned head"),
], ids=["sha", "absent", "candidate-digests", "checkpoint"])
def test_THE_HEAD_BINDING_ALSO_NEEDS_THE_SCREEN_TO_HAVE_PLAYED_THIS_HEAD(screen, why):
    rr, _ro, rb = _readers(screen={**GOOD_SCREEN, **screen})
    with pytest.raises(P.Refused, match=re.escape(why) + ".*the strength is not admitted"):
        P.trained_head(rr, rb, "strength")
    assert P.trained_head(rr, rb, "screen")["sha256"] == P.TRAINED_HEAD["sha256"]   # the screen never read it
    rr, _ro, rb = _readers()
    assert P.trained_head(rr, rb, "strength")["digests"] == HEAD_D


def test_THE_HEAD_BINDING_KEEPS_EVERY_TRAINING_CHECK_FOR_THE_COMPARISON():
    for train, data, why in (({**GOOD_TRAIN, "head_sha256": "0" * 64}, None, "not the pinned head"),
                             (None, b"altered bytes", "head.npz is not the pinned bytes")):
        rr, _ro, rb = _readers(train=train, data=data)
        with pytest.raises(P.Refused, match=re.escape(why)):
            P.trained_head(rr, rb, "strength")
    rr, _ro, rb = _readers(screen=False)
    with pytest.raises(P.Refused, match="does not exist: the strength has no trained head"):
        P.trained_head(rr, rb, "strength")


def test_RUN_STRENGTH_WITH_ITS_GATE_OPEN_PASSES_EVERY_REAL_CHECK_TO_THE_SCREEN_S_SEAMS(monkeypatch, capsys, tmp_path):
    """The gate opened in this process only. Through the REAL pins, admission and head binding the entry reaches its
    checkpoint refusal; with a checkpoint file present it reaches the screen's real seams with the bound head -- here a
    recorder that stops -- and nothing else: no evidence directory is made."""
    gate, name = P.KINDS["strength"]["gate"]
    monkeypatch.setattr(gate, name, True)
    monkeypatch.setattr(P, "_mlx_loaded", lambda: False)
    assert P.main(["run", "strength", "--checkpoint", "/no/such/file"]) == P.EXIT_REFUSED
    assert "--checkpoint must name an existing file" in capsys.readouterr().err
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"not a model")
    seen = []

    def recorder(path, head):
        seen.append((path, head["sha256"], head["bytes"] == (ROOT / P.TRAINED_HEAD["file"]).read_bytes()))
        raise AssertionError("stopped at the screen's real seams")
    monkeypatch.setattr(P, "_real_screen_seams", recorder)
    with pytest.raises(AssertionError, match="the screen's real seams"):
        P.main(["run", "strength", "--checkpoint", str(ckpt)])
    assert seen == [(ckpt, P.TRAINED_HEAD["sha256"], True)]
    assert not (ROOT / P.KINDS["strength"]["evidence"]).exists()


# ─────────────────────────────── the comparison on fakes, through the real record boundary ────────────────────────────

def _plans(rel):
    return {P.KINDS["strength"]["plan"]: {"base": BASE, "n_slots": GAMES}, P.KINDS["screen"]["plan"]: SCREEN_PLAN}[rel]


def strength_seams(play, **over):
    return screen_seams(play, **{"read_plan": _plans, **over})


def _strength(tmp_path, seams):
    return _REAL_CONCLUDE("strength", lambda o, r, lg: P._screen(seams, HEAD, o, r, lg, "strength"), tmp_path / "ev",
                          {})


def _events(tmp_path):
    return [json.loads(x) for x in (tmp_path / "ev" / "progress.jsonl").read_text().splitlines()]


def _skewed(outcome, seconds=1.0, extra_candidate_calls=0):
    """The screen's stub game with the candidate's search seconds scaled, or extra underlying evaluations made by the
    candidate's side: what moves the two cost guards."""
    inner = stub_play(outcome)

    def play(task, red, black, config, capture, replay_dir, classes, on_search):
        cand_red = task.red_agent.agent_id == K.CANDIDATE
        args = (np.array([[0, 1]], np.int32), np.array([[1, 2]], np.int32), np.ones((1, 2), np.float32), 24)
        for _ in range(extra_candidate_calls):
            (red if cand_red else black).infer(np.zeros((1, 24, 24, 30), np.float32), *args)

        def scaled(is_red, secs, halving):
            on_search(is_red, secs * (seconds if is_red == cand_red else 1.0), halving)
        return inner(task, red, black, config, capture, replay_dir, classes, scaled)
    return play


THRESHOLD = 0.5 + math.sqrt(math.log(40) / 1018)          # the least mean pair score that passes, exclusive


@pytest.mark.parametrize("outcome, points, verdict", [
    (lambda g, cr: K.CANDIDATE, GAMES, "PASS"),
    (lambda g, cr: P.REFERENCE, 0, "NON_PASS"),
    (lambda g, cr: None, GAMES / 2, "NON_PASS"),
    (_wins_first(570.5), 570.5, "PASS"),                   # 0.56041: the nearest reachable mean above the bound
    (_wins_first(570), 570, "NON_PASS"),                   # 0.55992: the nearest below
], ids=["all-candidate", "all-reference", "all-drawn", "just-above", "just-below"])
def test_THE_COMPARISON_PLAYS_ALL_1018_GAMES_AND_APPLIES_THE_FROZEN_RULE_ONLY(tmp_path, outcome, points, verdict):
    calls, loaded = [], []
    play = stub_play(outcome, calls=calls)
    code, rec = _strength(tmp_path, screen_seams(play, loaded=loaded, read_plan=_plans))
    assert code == P.EXIT_COMPLETED and rec["status"] == "COMPLETED", rec.get("void")
    assert calls == list(range(GAMES)) and len(loaded) == 2 and rec["run"] == "strength"
    s = rec["strength"]
    assert s["pairs"] == 509 and s["mean_pair_score"] == pytest.approx(points / GAMES)
    assert (s["mean_pair_score"] > THRESHOLD) is (verdict == "PASS")
    assert s["hoeffding_lower"] == pytest.approx(points / GAMES - math.sqrt(math.log(40) / 1018))
    assert rec["verdict"] == s["verdict"] == verdict and s["compute_matched"] is True
    assert rec["guards"]["evaluations_ratio"] == 1.0 and rec["guards"]["search_seconds_ratio"] == 1.0
    assert rec["guards"]["guards"] == K.GUARDS and "screen" not in rec and "cost_ratios" not in rec
    assert rec["seeds"] == {"base": BASE, "n_slots": GAMES}
    assert rec["screen_block_excluded"] == [BASE - 1000, BASE - 936]
    assert rec["trained_head"] == {k: HEAD[k] for k in ("file", "sha256", "digests")}
    assert rec["digests"] == {K.CANDIDATE: HEAD_D, P.REFERENCE: PARENT_D}
    assert rec["head_probe"]["max_prior_abs_diff"] > 0
    assert rec["protocol"]["classes"] == {k: "scripts.GPU.alphazero.root_sequential_halving.RootSequentialHalvingMCTS"
                                          for k in (K.CANDIDATE, P.REFERENCE)}
    assert rec["protocol"]["evaluators"] == {K.CANDIDATE: "MirrorAveragingEvaluator",
                                             P.REFERENCE: "MirrorAveragingEvaluator"}
    assert [a["mcts_sims"] for a in rec["protocol"]["agents"]] == [200, 200]
    games = [json.loads(x) for x in (tmp_path / "ev" / "games.jsonl").read_text().splitlines()]
    assert [g["slot"] for g in games] == list(range(GAMES)) and {g["pairing_id"] for g in games} == \
        {"combination-policy-strength"}
    assert rec["background_checks"] == GAMES + 2 and rec["setting"] == {P.SETTING: P.SETTING_VALUE}


@pytest.mark.parametrize("kw, ev_ratio, sec_ratio, verdict", [
    ({"seconds": 1.09}, 1.0, 1.09, "PASS"),
    ({"seconds": 1.11}, 1.0, 1.11, "NOT_COMPUTE_MATCHED"),
    ({"extra_candidate_calls": 1}, 2.0, 1.0, "NOT_COMPUTE_MATCHED"),
], ids=["seconds-within", "seconds-over", "evaluations-over"])
def test_A_BOUND_ABOVE_0_5_PASSES_ONLY_WITH_BOTH_COST_GUARDS_over_the_whole_run(tmp_path, kw, ev_ratio, sec_ratio,
                                                                                  verdict):
    code, rec = _strength(tmp_path, strength_seams(_skewed(lambda g, cr: K.CANDIDATE, **kw)))
    assert code == P.EXIT_COMPLETED, rec.get("void")
    g = rec["guards"]
    assert g["evaluations_ratio"] == pytest.approx(ev_ratio) and g["search_seconds_ratio"] == pytest.approx(sec_ratio)
    assert rec["strength"]["bound_above_null"] is True and rec["verdict"] == verdict
    assert g == K.guards(rec["cost_totals"][K.CANDIDATE], rec["cost_totals"][P.REFERENCE])


def test_EVERY_GAME_IS_LOGGED_AS_STARTED_BEFORE_IT_IS_PLAYED_and_BOINC_is_read_around_every_game(tmp_path):
    _strength(tmp_path, strength_seams(stub_play(lambda g, cr: K.CANDIDATE)))
    ev = _events(tmp_path)
    names = [e["event"] for e in ev]
    assert names[0] == "start" and names[1] == "bg_check" and ev[1]["when"] == "before load"
    starts = [i for i, e in enumerate(ev) if e["event"] == "game_start"]
    ends = [i for i, e in enumerate(ev) if e["event"] == "game_end"]
    assert [ev[i]["game"] for i in starts] == [ev[i]["slot"] for i in starts] == [ev[i]["game"] for i in ends] == \
        list(range(GAMES))
    assert [ev[i]["red"] for i in starts[:2]] == [K.CANDIDATE, P.REFERENCE]
    for g, (s, t) in enumerate(zip(starts, ends)):
        assert s < t and ev[t + 1]["event"] == "bg_check" and ev[t + 1]["when"] == f"after game {g}"
    assert [e["when"] for e in ev if e["event"] == "bg_check"][-1] == "before the verdict" and names[-1] == "end"


@pytest.mark.parametrize("over, step, why", [
    ({"read_plan": lambda rel: {"base": BASE, "n_slots": 64}}, "seeds", "not exactly 1018 slots"),
    ({"read_plan": lambda rel: {"base": BASE, "n_slots": GAMES}}, "seeds", "overlaps the screen's"),
    ({"read_plan": lambda rel: {P.KINDS["screen"]["plan"]: {"base": BASE + GAMES - 1, "n_slots": 64}}.get(
        rel, {"base": BASE, "n_slots": GAMES})}, "seeds", "overlaps the screen's"),
    ({"read_plan": lambda rel: {P.KINDS["screen"]["plan"]: {}}.get(rel, {"base": BASE, "n_slots": GAMES})}, "seeds",
     "the screen's plan is unreadable"),
    ({"seed_ok": lambda s: s != BASE + GAMES - 1}, "seeds", "1 seeds not ACCOUNTED"),
    ({"install_head": lambda net: {**HEAD_D, "policy_head.": "another head"}}, "head", "not the trained head's"),
    ({"install_head": lambda net: dict(HEAD_D)}, "head", "search does not reach the head"),
    ({"digests": lambda net: dict(HEAD_D)}, "reference", "not calib020_0001's parent"),
    ({"checkpoint_sha1": lambda: "0" * 40}, "checkpoint", "is not calib020_0001's"),
    ({"background": lambda: dict(BOINC)}, "background_compute", "before load: BOINC detected"),
], ids=["screen-sized-plan", "same-block", "overlap-at-the-top", "screen-plan-unreadable", "seed", "head-arrays",
        "probe-equal", "reference-trained", "checkpoint", "boinc-before-load"])
def test_EACH_PRECONDITION_IS_A_VOID_BEFORE_ANY_GAME(tmp_path, over, step, why):
    calls = []
    code, rec = _strength(tmp_path, strength_seams(stub_play(lambda g, cr: K.CANDIDATE, calls=calls), **over))
    assert code == P.EXIT_VOID and rec["status"] == "VOID" and rec["void"]["step"] == step, rec.get("void")
    assert why in rec["void"]["detail"] and calls == [] and rec["verdict"] is None
    assert not (tmp_path / "ev" / "games.jsonl").exists() or not (tmp_path / "ev" / "games.jsonl").read_text()


def test_BOTH_SIDES_SHARING_A_NETWORK_IS_A_VOID(tmp_path):
    one = ScreenFake()
    code, rec = _strength(tmp_path, strength_seams(stub_play(lambda g, cr: K.CANDIDATE), load_evaluator=lambda: one))
    assert code == P.EXIT_VOID and rec["void"]["step"] == "load"


@pytest.mark.parametrize("kw, step", [({"fail_at": 900}, "exception"), ({"searches": (2, 1)}, "attribution")],
                         ids=["game-fails", "count"])
def test_A_GAME_THAT_FAILS_OR_MISCOUNTS_ITS_SEARCHES_LEAVES_NO_SCORE(tmp_path, kw, step):
    code, rec = _strength(tmp_path, strength_seams(stub_play(lambda g, cr: K.CANDIDATE, **kw)))
    assert code == P.EXIT_VOID and rec["void"]["step"] == step, rec.get("void")
    assert rec["verdict"] is None and "strength" not in rec and "pair_scores" not in rec and "guards" not in rec


def test_BOINC_MID_RUN_IS_A_VOID_WITH_NO_SCORE(tmp_path):
    seen = iter([CLEAR] * 6 + [BOINC])
    code, rec = _strength(tmp_path, strength_seams(stub_play(lambda g, cr: K.CANDIDATE), background=lambda: next(seen)))
    assert code == P.EXIT_VOID and rec["void"]["step"] == "background_compute"
    assert "after game 5" in rec["void"]["detail"] and rec["verdict"] is None and "strength" not in rec


def test_THE_SCREEN_S_PATH_IS_UNCHANGED_FOR_THE_SCREEN(tmp_path):
    """The kind argument's default is the screen: its 64 games, its rule, its descriptive ratios, its pairing id."""
    code, rec = _REAL_CONCLUDE("screen", lambda o, r, lg: P._screen(
        screen_seams(stub_play(lambda g, cr: K.CANDIDATE)), HEAD, o, r, lg), tmp_path / "ev", {})
    assert code == P.EXIT_COMPLETED and rec["verdict"] == rec["screen"]["verdict"] == "CONTINUE"
    assert "strength" not in rec and "guards" not in rec and "screen_block_excluded" not in rec
    assert "descriptive" in rec["cost_ratios"]["note"]
    games = [json.loads(x) for x in (tmp_path / "ev" / "games.jsonl").read_text().splitlines()]
    assert len(games) == 64 and {g["pairing_id"] for g in games} == {"combination-policy-screen"}


# ─────────────────────────────── the wrapper's bytes and the ceiling ───────────────────────────────

def _code(path):
    return path.read_text().split("set -u\n", 1)[1]


def test_THE_CEILING_IS_THE_USER_S_51780_S_FROM_THE_SCREEN_S_COMPLETED_WALL_TIME():
    """2,169 s (the screen's committed outer.json) x 1,018 / 64 x 1.5, rounded up to the minute (the user's review,
    2026-11-01): 51,780 s; TERM 51,770 s, KILL 10 s later, cleanup observation through 51,790 s."""
    outer = json.loads((ROOT / SCREEN_RUN / "outer.json").read_text())
    assert outer["wall_seconds"] == 2169 and outer["outcome"] == "COMPLETED"
    raw = outer["wall_seconds"] * GAMES / 64 * 1.5
    ceiling = int(math.ceil(raw / 60.0) * 60)
    assert round(raw, 1) == 51751.0 and ceiling == 51780 and divmod(ceiling, 60) == (863, 0)
    assert divmod(ceiling, 3600)[0] == 14 and divmod(ceiling % 3600, 60)[0] == 23
    w = _code(WRAPPER)
    term, grace = (int(re.search(rf"^{k}=(\d+)$", w, re.M).group(1)) for k in ("TERM_S", "GRACE_S"))
    assert (term, grace) == (51770, 10) and term + grace == ceiling and term + 2 * grace == 51790
    head = WRAPPER.read_text().split("set -u\n", 1)[0]
    for said in ("2,169 s", "1,018 / 64 x 1.5", "51,751.0 s", "51,780 s (14 h 23 min)", "TERM at 51770 s",
                 "51780 s in all", "through 51790 s", "not a completion guarantee", "no automatic retry"):
        assert said in head, said


def test_THE_WRAPPER_IS_THE_SCREEN_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """The paths, the gate, `run strength` and TERM_S, line for line; everything else byte-identical. Its header is the
    comparison's paragraph followed by the screen's header unchanged."""
    screen, strength = _code(WRAPPER_SCREEN).splitlines(), _code(WRAPPER).splitlines()
    assert len(screen) == len(strength)
    changed = [(a, b) for a, b in zip(screen, strength) if a != b]
    assert [b for _a, b in changed] == [
        "OUT_REL=docs/superpowers/evidence/combination-policy-strength-run",
        "RUN_REL=docs/superpowers/evidence/combination-policy-strength",
        "GATE_REL=scripts/GPU/alphazero/combination_policy_strength_authorization.py",
        f"GATE={GATE_NAME}", "TERM_S=51770",
        '/usr/bin/perl -e "$SUPERVISOR" "$OUT_REL/pgid.txt" "$TERM_S" "$GRACE_S" env MLX_ENABLE_TF32=0 PYTHONPATH=. '
        '"$PY" -B -m "$MODULE" run strength --checkpoint "$CKPT" \\'], changed
    head_screen = WRAPPER_SCREEN.read_text().split("set -u\n", 1)[0]
    head = WRAPPER.read_text().split("set -u\n", 1)[0]
    assert head.startswith("#!/bin/bash\n# THE COMBINATION POLICY PILOT: THE STRENGTH COMPARISON, 509 COLOUR-REVERSED")
    assert head.endswith(head_screen[len("#!/bin/bash\n"):])
    ours = head[:-len(head_screen) + len("#!/bin/bash\n")]
    assert "CEILING SET BY THE" in ours and "BUILT ONLY" in ours and "sqrt(ln 40 / 1018)" in ours
    assert os.access(WRAPPER, os.R_OK) and subprocess.run(["bash", "-n", str(WRAPPER)]).returncode == 0


def test_THE_WRAPPER_SERVES_THE_PILOT_S_STRENGTH_PATHS():
    w = _code(WRAPPER)
    out, run = (re.search(rf"^{k}=(\S+)$", w, re.M).group(1) for k in ("OUT_REL", "RUN_REL"))
    kind = P.KINDS["strength"]
    assert run == kind["evidence"] and out == str(WRAPPER.parent.relative_to(ROOT)) == \
        str(pathlib.Path(kind["plan"]).parent) == f"{kind['evidence']}-run"
    assert re.search(r"^GATE_REL=(\S+)$", w, re.M).group(1) == \
        str(pathlib.Path(kind["gate"][0].__file__).relative_to(ROOT))
    assert re.search(r"^MODULE=(\S+)$", w, re.M).group(1) == P.__name__ and f"RECORD_KIND={P.RECORD_KIND}\n" in w
    assert "CKPT=\"$MAIN/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors\"" in w


# ─────────────────────────────── fake launches of the REAL strength wrapper ───────────────────────────────

STUB_PILOT_STRENGTH = STUB_PILOT.replace("combination-policy-throughput", "combination-policy-strength")
assert STUB_PILOT_STRENGTH.count("combination-policy-strength") == 2


def _fake_repo(tmp, term_s=None):
    """A throwaway repository: the real strength wrapper (TERM_S changed only when a test says so), its gate opened by
    a one-line commit, and three stubs."""
    repo = tmp / "repo"
    pkg = repo / "scripts/GPU/alphazero"
    pkg.mkdir(parents=True)
    for d in (repo / "scripts", repo / "scripts/GPU", pkg):
        (d / "__init__.py").write_text("")
    gate = pkg / "combination_policy_strength_authorization.py"
    gate.write_text(f'"""stub gate"""\n{GATE_NAME} = False\n')
    (pkg / "idle_gate.py").write_text(STUB_IDLE)
    (pkg / "background_compute.py").write_text(STUB_BG)
    (pkg / "combination_policy_pilot.py").write_text(STUB_PILOT_STRENGTH)
    wdir = repo / "docs/superpowers/evidence/combination-policy-strength-run"
    wdir.mkdir(parents=True)
    raw = WRAPPER.read_bytes()
    if term_s is not None:
        assert raw.count(b"\nTERM_S=51770\n") == 1
        raw = raw.replace(b"\nTERM_S=51770\n", f"\nTERM_S={term_s}\n".encode())
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
    assert not (repo / "docs/superpowers/evidence/combination-policy-strength").exists()
    assert f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_A_GATE_OPENED_WITH_ANY_OTHER_CHANGE_IS_REFUSED(tmp_path):
    repo, wdir, gate = _fake_repo(tmp_path)
    _git(repo, "reset", "-q", "--soft", "HEAD~1")
    (repo / "scripts/GPU/alphazero/idle_gate.py").write_text(STUB_IDLE + "\n")
    _git(repo, "commit", "-qam", "open the gate and change something else")
    p = _launch(repo, wdir)
    assert p.returncode == 1 and "does not change only" in p.stderr, p.stderr
    assert "FAKE_LAUNCHED" not in p.stdout + p.stderr and f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_AN_IDLE_MACHINE_REACHES_RUN_STRENGTH_with_its_exact_arguments_and_ceiling(tmp_path):
    repo, wdir, gate = _fake_repo(tmp_path)
    p = _launch(repo, wdir, run="record")
    assert p.returncode == 0, p.stderr
    launched = json.loads((wdir / "run.stdout.txt").read_text().split("FAKE_LAUNCHED", 1)[1])
    ckpt = str(repo.resolve() / "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors")
    assert launched["argv"] == ["run", "strength", "--checkpoint", ckpt] and launched["tf32"] == "0"
    assert (wdir / "pgid.txt").read_text() == f"{launched['pgid']}\n"
    outer = json.loads((wdir / "outer.json").read_text())
    assert outer["outcome"] == "COMPLETED" and outer["survivors"] == "" and outer["gate_restored"] is True
    assert outer["term_s"] == 51770 and outer["grace_s"] == 10 and outer["record_by"] == "script"
    rec = json.loads((repo / "docs/superpowers/evidence/combination-policy-strength/record.json").read_text())
    assert not P.outer_faults(outer, rec)
    assert f"{GATE_NAME} = False\n" in gate.read_text()


def test_FAKE_LAUNCH_THE_CEILING_TERMINATES_THE_WHOLE_GROUP_at_the_comparison_s_paths(tmp_path):
    """TERM_S changed to 2 in this copy only: the run and its child outlive it; the group TERM ends both, and the
    wrapper's VOID record lands in the comparison's evidence directory."""
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
    rec = json.loads((repo / "docs/superpowers/evidence/combination-policy-strength/record.json").read_text())
    assert outer["record_by"] == "wrapper" and rec["status"] == "VOID" and rec["void"]["step"] == "timeout"
    assert f"{GATE_NAME} = False\n" in gate.read_text()


# ─────────────────────────────── the seed survey copy (built, never run on the repository) ────────────────────────────

def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, EV / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # module level only: no mode runs, nothing is written
    return mod


def _split(rel):
    text = (EV / rel).read_text()
    end = text.index('"""', 3)
    return text[3:end], text[end + 3:]


def _changes(old, new):
    a, b = old.splitlines(), new.splitlines()
    removed, added = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            removed += [x.strip() for x in a[i1:i2]]
            added += [x.strip() for x in b[j1:j2]]
    return removed, added


def test_THE_SURVEY_COPY_IS_THE_SCREEN_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """Below the docstring: the block size, paths, plan kind, module name, titles and verdict lines and the slot
    listing's heading -- nothing else; the screen's corrected empty-band control is kept. The docstring is the
    comparison's paragraph followed by the screen's description unchanged."""
    doc_s, code_s = _split(SURVEY_SCREEN)
    doc, code = _split(SURVEY)
    assert doc.endswith(doc_s) and "§5 step 6" in doc[:-len(doc_s)] and "(1018)" in doc[:-len(doc_s)]
    removed, added = _changes(code_s, code)
    assert removed == [
        "N_SLOTS = 64", 'HERE = "docs/superpowers/evidence/combination-policy-screen-seed-survey"',
        'PLAN = "docs/superpowers/evidence/combination-policy-screen-run/plan.json"',
        'VERDICT_OK = "THE COMBINATION-POLICY SCREEN BLOCK IS CLEAN"',
        'spec = importlib.util.spec_from_file_location(f"_combination_policy_screen_{key}", p)',
        'return {"kind": "combination-policy-screen-plan/1", "base": base, "n_slots": N_SLOTS,',
        'faults.append("the screen plan is absent or not committed")',
        'faults.append("the screen plan is not build_plan of the surveyed block")',
        "say(f\"COMBINATION POLICY PILOT -- SCREEN GAME-SEED {'SURVEY' if mode == 'survey' else "
        "'COLLISION PROOF, mode ' + mode.upper()}\")",
        'say("SLOTS (combination_policy_pilot run screen: game g plays seed BASE + g, in order, in one process):")',
        "say(f\"PROOF VERDICT ({mode}): {'THE COMBINATION-POLICY SCREEN BLOCK IS CLEAN AND THE STATE IS AS "
        "EXPECTED' if ok else 'NOT PROVEN -- STOP'}\""], removed
    assert added == [x.replace("screen", "strength").replace("SCREEN", "STRENGTH").replace("N_SLOTS = 64",
                                                                                          "N_SLOTS = 1018")
                     for x in removed], added


def test_THE_SURVEY_COPY_SERVES_THE_COMPARISON_IN_THE_2027_BAND_and_holds_no_seed_literal():
    from scripts.GPU.alphazero import seed_band_2027 as SB
    survey = _load(SURVEY, "_strength_survey")
    assert survey.N_SLOTS == 2 * P.KINDS["strength"]["pairs"] == GAMES and survey.PLAN == P.KINDS["strength"]["plan"]
    S, _Pf, SEL = survey.load_method()
    assert S.N == GAMES and S.scan is SEL.scan and SEL.__name__.endswith("seed_band_2027")
    assert survey.PINNED == _load(SURVEY_SCREEN, "_screen_survey_pins").PINNED
    base = SEL.candidate([], [], GAMES)[1]
    assert survey.build_plan(base, "0" * 64) == {"kind": "combination-policy-strength-plan/1", "base": base,
                                                 "n_slots": GAMES, "survey_record_sha256": "0" * 64}
    assert set(survey.OUT.values()) & set(_load(SURVEY_SCREEN, "_screen_survey_out").OUT.values()) == set()
    assert all((ROOT / p).is_file() for p in survey.OUT.values())   # each mode ran once (the lifecycle test above)
    code = _split(SURVEY)[1]
    assert not SB.scan(code) and not re.search(r"\b20[0-9]{7}\b", code)


def test_THE_SURVEY_COPY_S_CONTROLS_ALL_PASS_on_a_synthetic_2027_block():
    """The copied controls (state 12, plants 8, exemptions 6, enumeration 6) and the two-band controls (10), with the
    real prior terms and a 1,018-seed block the rule places in an empty band; the repository is not scanned here."""
    survey = _load(SURVEY, "_strength_survey_controls")
    S, Pf, SEL = survey.load_method()
    registries, named, gen, masks, consumed = S.prior_terms()
    block = SEL.candidate([], [], GAMES)[2]
    assert block[1] - block[0] == GAMES and block[0] % SEL.ALIGN == 0
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


PINNED_FILES = ("docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py",
                "docs/superpowers/evidence/tactical-candidate-seed-registration/01_collision_proof.py",
                "scripts/GPU/alphazero/injected_defect_controls.py", "scripts/GPU/alphazero/seed_band_2027.py")


def _mode(repo, mode, monkeypatch, capsys):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location(f"_strength_survey_{mode}",
                                                  repo / "docs/superpowers/evidence" / SURVEY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    code = mod.main(["x", "--mode", mode])
    capsys.readouterr()
    return mod, code, (repo / mod.OUT[mode]).read_text(encoding="utf-8")


def test_THE_THREE_MODES_REHEARSED_ON_A_SYNTHETIC_REPOSITORY(tmp_path, monkeypatch, capsys):
    """survey -> pre -> registration -> post, on a throwaway git repository holding only the copy, the pinned method,
    a synthetic registry and two planted literals (one per band): the rule places the 1,018-seed block above the 2027
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
    (repo / "docs/superpowers/evidence/combination-policy-strength-run").mkdir(parents=True)
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "synthetic")

    mod, code, survey = _mode(repo, "survey", monkeypatch, capsys)
    assert code == 0, survey[-2500:]
    anchor, base, block = SB.candidate(REG.ACCOUNTED_SEED_INTERVALS, [new], GAMES)
    assert anchor == new and block[1] - block[0] == GAMES
    assert f"in the 2027 band = {new}; BASE = {base}; block = [{block[0]}, {block[1]}) -- 1018 seeds" in survey
    assert survey.endswith("SURVEY VERDICT   : THE COMBINATION-POLICY STRENGTH BLOCK IS CLEAN\n")
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
    assert post.endswith("PROOF VERDICT (post): THE COMBINATION-POLICY STRENGTH BLOCK IS CLEAN AND THE STATE IS AS "
                         "EXPECTED\n")


# ─────────────────────────────── the seed closing copy (built, never run here) ───────────────────────────────

def test_THE_CLOSING_COPY_IS_THE_SCREEN_S_WITH_ONLY_THE_STATED_DIFFERENCES():
    """Below the docstring: the paths, the block size and the title, line for line -- the sequential rule (started
    games a prefix 0..k-1 in order) is untouched. The docstring is the comparison's paragraph followed by the screen's
    description unchanged."""
    doc_s, code_s = _split(CLOSING_SCREEN)
    doc, code = _split(CLOSING)
    assert doc.endswith(doc_s) and "§5 step 6" in doc[:-len(doc_s)] and "IN ORDER IN ONE" in doc[:-len(doc_s)]
    a, b = code_s.splitlines(), code.splitlines()
    assert len(a) == len(b)
    assert [y.strip() for x, y in zip(a, b) if x != y] == [
        'HERE = "docs/superpowers/evidence/combination-policy-strength-seed-closing"',
        'RUN = "docs/superpowers/evidence/combination-policy-strength"',
        'RUN_DIR = "docs/superpowers/evidence/combination-policy-strength-run"', "N_SLOTS = 1018",
        'lines = ["COMBINATION POLICY PILOT -- STRENGTH SEED-USAGE DERIVATION (closing)", "=" * 96,']


def test_THE_CLOSING_COPY_SERVES_THE_COMPARISON_and_holds_no_seed_literal():
    from scripts.GPU.alphazero import seed_band_2027 as SB
    closing = _load(CLOSING, "_strength_closing")
    kind = P.KINDS["strength"]
    assert closing.N_SLOTS == 2 * kind["pairs"] == GAMES and closing.INPUTS[0] == kind["plan"]
    assert closing.RUN == kind["evidence"] and closing.RUN_DIR == str(pathlib.Path(kind["plan"]).parent)
    assert closing.OUT == "docs/superpowers/evidence/combination-policy-strength-seed-closing/02_derive_usage_run.txt"
    assert not (ROOT / closing.OUT).exists()
    code = _split(CLOSING)[1]
    assert not SB.scan(code) and not re.search(r"\b20[0-9]{7}\b", code)


def test_THE_CLOSING_COPY_S_CONTROLS_ALL_PASS_at_a_synthetic_base():
    results = _load(CLOSING, "_strength_closing_c")._controls(BASE)
    assert len(results) == 15 and all(ok for _l, ok, _d in results), [r for r in results if not r[1]]


def test_THE_CLOSING_COPY_S_CONTROLS_ALL_PASS_at_the_registered_base():
    """The same 15 controls at the plan's BASE, read from the committed plan."""
    base = json.loads((ROOT / P.KINDS["strength"]["plan"]).read_text(encoding="utf-8"))["base"]
    results = _load(CLOSING, "_strength_closing_rb")._controls(base)
    assert len(results) == 15 and all(ok for _l, ok, _d in results), [r for r in results if not r[1]]


def test_THE_CLOSING_RECONCILES_THE_COMPARISON_S_OWN_EVENTS_completed_and_VOID_mid_run(tmp_path):
    """The comparison's progress.jsonl, from the real record boundary on fakes, read by the closing's rule: a completed
    run exposes and retires the whole block; a run VOID in game 900 exposes games 0-900 only and retires the whole
    block."""
    closing = _load(CLOSING, "_strength_closing_r")
    plan = {"base": BASE, "n_slots": GAMES}
    regs = {"ACCOUNTED": [(BASE, BASE + GAMES)], "EXPOSED": [], "RETIRED": []}
    _strength(tmp_path / "a", strength_seams(stub_play(lambda g, cr: K.CANDIDATE)))
    faults, expose, retire = closing.reconcile(plan, _events(tmp_path / "a"), {"outer_exit": 0, "survivors": ""},
                                               "COMPLETED", regs)
    assert (faults, expose, retire) == ([], (BASE, BASE + GAMES), (BASE, BASE + GAMES))
    _strength(tmp_path / "b", strength_seams(stub_play(lambda g, cr: K.CANDIDATE, fail_at=900)))
    faults, expose, retire = closing.reconcile(plan, _events(tmp_path / "b"), {"outer_exit": 3, "survivors": ""},
                                               "VOID", regs)
    assert (faults, expose, retire) == ([], (BASE, BASE + 901), (BASE, BASE + GAMES))
