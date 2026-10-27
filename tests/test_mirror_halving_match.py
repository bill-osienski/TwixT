"""The screen and strength comparison's gated entry (`scripts/GPU/alphazero/mirror_halving_match.py`).

The public entry is tested in the REAL state where it can be: the gates open (monkeypatched, never edited), the pinned
card and code, the admission rule, the agents, classes and evaluators the run builds, the scoring and both verdicts at
the card's sizes (32 and 509 pairs). Only the model and the games are replaced: every evaluator is a GPU-free fake and
`play_and_build` is a stub that checks the wiring it receives and reports a scripted result. The real copied loop with
the real composition is exercised by the step-B loop check (docs/superpowers/evidence/mirror-halving-loop-check/).

🔴 EVERY TEST IS ISOLATED, WHATEVER THE GATE SAYS (ledger entry 28, #74): the autouse fixture redirects both evidence
directories, REPLACES `_real_seams` with fake seams and `_read_record` with an in-memory one, so nothing reached through
the public entry can load a checkpoint, play a game or read a real record. The fresh-process checks stub `conclude`."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.GPU.alphazero import eval_runner as ER
from scripts.GPU.alphazero import mirror_halving_c1 as C1
from scripts.GPU.alphazero import mirror_halving_combination as K
from scripts.GPU.alphazero import mirror_halving_match as MATCH
from scripts.GPU.alphazero import mirror_halving_screen_authorization as SCREEN_GATE
from scripts.GPU.alphazero import background_compute as BG
from scripts.GPU.alphazero import mirror_halving_strength2_authorization as STRENGTH2_GATE
from scripts.GPU.alphazero import mirror_halving_strength_authorization as STRENGTH_GATE
from scripts.GPU.alphazero import root_sequential_halving as RSH
from scripts.GPU.alphazero import search_distillation_d2_measure as M
from scripts.GPU.alphazero.mcts import MCTS
from scripts.GPU.alphazero.mirror_averaging import MirrorAveragingEvaluator
from tests.test_mirror_halving_c1 import FakeEv

REAL_SEAMS, REAL_READ_RECORD = MATCH._real_seams, MATCH._read_record
C1_PASS = {"status": "COMPLETED", "verdict": "PASS", "card_sha256": C1.CARD_SHA256}
SCREEN_CONTINUE = {"status": "COMPLETED", "screen_verdict": "CONTINUE", "card_sha256": C1.CARD_SHA256}
RECORDS = {}                                   # the in-memory committed records, reset for every test
#: the committed screen record admission reads: the REAL path, never the redirected output directory
SCREEN_REL = MATCH.ADMISSION["strength"][1][0]


def stub_play(outcome, secs=(1.0, 1.0), calls=None):
    """A scripted game: checks the wiring it is handed, spends one network call per side (two positions each: the
    averaging wrapper's two underlying calls, the incumbent's batch of two), reports one search per side and the
    winner outcome(game_idx, candidate_is_red) returns ("candidate" | "incumbent" | None)."""
    def play(task, red, black, config, capture, replay_dir, classes, on_search):
        cand_red = task.red_agent.agent_id == K.CANDIDATE
        cand_ev, inc_ev = (red, black) if cand_red else (black, red)
        assert type(cand_ev) is MirrorAveragingEvaluator and type(inc_ev) is FakeEv
        assert classes == {K.CANDIDATE: RSH.RootSequentialHalvingMCTS, K.INCUMBENT: MCTS} and capture is True
        assert ER.agent_sims(task.red_agent, config) == (200 if cand_red else 400)
        assert (config.board_size, config.max_moves, config.mcts_sims) == (24, 280, 400)
        args = (np.array([[0]], np.int32), np.array([[1]], np.int32), np.ones((1, 1), np.float32), 24)
        cand_ev.infer(np.zeros((1, 24, 24, 30), np.float32), *args)
        inc_ev.infer(np.zeros((2, 24, 24, 30), np.float32), *(np.repeat(a, 2, axis=0) for a in args[:3]), 24)
        on_search(cand_red, secs[0], True)
        on_search(not cand_red, secs[1], False)
        if calls is not None:
            calls.append(task.game_idx)
        who = outcome(task.game_idx, cand_red)
        winner = None if who is None else ("red" if (who == K.CANDIDATE) == cand_red else "black")
        return ER.make_result(task, winner, "win" if winner else "state_cap", 30, config=config)
    return play


def seams(play, **over):
    s = dict(versions=lambda: dict(C1.VERSIONS), checkpoint_sha1=lambda: M.CHECKPOINT_SHA1, load_evaluator=FakeEv,
             require_eval=lambda ev, when: {"modules_in_training_mode": 0, "when": when}, loaded_modules=lambda: {},
             read_plan=lambda rel: {"base": 777000, "n_slots": 64 if "screen" in rel else 1018},
             seed_ok=lambda s: True, play_and_build=play, between_games=lambda: None)
    s.update(over)
    return MATCH.Seams(**s)


ALL_CANDIDATE = stub_play(lambda g, cand_red: K.CANDIDATE)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    for kind in MATCH.KINDS:
        monkeypatch.setitem(MATCH.KINDS[kind], "evidence", str(tmp_path / kind))
    monkeypatch.setattr(MATCH, "_real_seams", lambda ckpt: seams(ALL_CANDIDATE))
    RECORDS.clear()
    RECORDS[C1.EVIDENCE_DIR] = dict(C1_PASS)

    def read_record(rel):
        if rel not in RECORDS:
            raise FileNotFoundError(rel)
        return RECORDS[rel]

    monkeypatch.setattr(MATCH, "_read_record", read_record)
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(MATCH, "_mlx_loaded", lambda: False)
    (tmp_path / "ckpt.safetensors").write_bytes(b"not a model")
    return tmp_path


@pytest.fixture
def opened(monkeypatch, isolated):
    monkeypatch.setattr(SCREEN_GATE, "MIRROR_HALVING_SCREEN_AUTHORIZED", True)
    monkeypatch.setattr(STRENGTH_GATE, "MIRROR_HALVING_STRENGTH_AUTHORIZED", True)
    return isolated


def _run(tmp, kind):
    return MATCH.main(["run", kind, "--checkpoint", str(tmp / "ckpt.safetensors")])


def _record(tmp, kind):
    return json.loads((tmp / kind / "record.json").read_text())


# ─────────────────────────────── refusals ───────────────────────────────

@pytest.mark.parametrize("kind", ["screen", "strength"])
def test_BOTH_GATES_ARE_CLOSED_IN_THE_SOURCE_and_a_closed_gate_refuses(isolated, capsys, kind):
    assert SCREEN_GATE.MIRROR_HALVING_SCREEN_AUTHORIZED is False
    assert STRENGTH_GATE.MIRROR_HALVING_STRENGTH_AUTHORIZED is False
    RECORDS[SCREEN_REL] = dict(SCREEN_CONTINUE)
    assert _run(isolated, kind) == 4 and "is closed" in capsys.readouterr().err
    assert not (isolated / kind).exists()


def test_NO_TEST_HERE_CAN_REACH_THE_REAL_MODEL_GAMES_RECORDS_OR_PATHS(isolated):
    assert MATCH._real_seams is not REAL_SEAMS and MATCH._read_record is not REAL_READ_RECORD
    assert MATCH._real_seams(None).play_and_build is ALL_CANDIDATE
    assert {k: v["evidence"] for k, v in MATCH.KINDS.items()} == {k: str(isolated / k) for k in MATCH.KINDS}


def test_ANYTHING_BUT_run_KIND_WITH_A_CHECKPOINT_IS_USAGE():
    for argv in ([], ["run", "screen"], ["run", "match", "--checkpoint", "/x"], ["run", "screen", "--ckpt", "/x"]):
        assert MATCH.main(argv) == 2


def test_THE_PINS_ARE_THE_FILES_ON_DISK():
    assert MATCH.CARD_SHA256 == C1.CARD_SHA256 == M._sha((MATCH.ROOT / MATCH.CARD).read_bytes())
    assert set(MATCH.PINNED) == {"scripts/GPU/alphazero/mirror_halving_combination.py",
                                 "scripts/GPU/alphazero/mirror_averaging.py",
                                 "scripts/GPU/alphazero/root_sequential_halving.py",
                                 "scripts/GPU/alphazero/mirror_halving_c1.py"}
    for rel, sha in MATCH.PINNED.items():
        assert M._sha((MATCH.ROOT / rel).read_bytes()) == sha, rel


@pytest.mark.parametrize("value", [None, "1", ""])
def test_ANY_SETTING_BUT_EXACTLY_0_REFUSES_even_with_the_gates_open(opened, monkeypatch, value, capsys):
    if value is None:
        monkeypatch.delenv("MLX_ENABLE_TF32")
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", value)
    assert _run(opened, "screen") == 4 and "MLX_ENABLE_TF32" in capsys.readouterr().err


def test_MLX_ALREADY_LOADED_OR_A_WRONG_PIN_REFUSES(opened, monkeypatch, capsys):
    monkeypatch.setattr(MATCH, "_mlx_loaded", lambda: True)
    assert _run(opened, "screen") == 4 and "imported" in capsys.readouterr().err
    monkeypatch.setattr(MATCH, "_mlx_loaded", lambda: False)
    monkeypatch.setitem(MATCH.PINNED, "scripts/GPU/alphazero/mirror_halving_c1.py", "0" * 64)
    assert _run(opened, "screen") == 4 and "pinned" in capsys.readouterr().err


@pytest.mark.parametrize("kind, records, why", [
    ("screen", {}, "does not exist"),
    ("screen", {C1.EVIDENCE_DIR: {**C1_PASS, "verdict": "STOP"}}, "not COMPLETED with verdict PASS"),
    ("screen", {C1.EVIDENCE_DIR: {**C1_PASS, "status": "VOID"}}, "not COMPLETED"),
    ("screen", {C1.EVIDENCE_DIR: {**C1_PASS, "card_sha256": "0" * 64}}, "under this card"),
    ("strength", {C1.EVIDENCE_DIR: C1_PASS}, "does not exist"),
    ("strength", {C1.EVIDENCE_DIR: C1_PASS, "SCREEN": {**SCREEN_CONTINUE, "screen_verdict": "STOP"}},
     "screen_verdict CONTINUE"),
    ("strength", {"SCREEN": SCREEN_CONTINUE}, "does not exist"),
])
def test_ADMISSION_NEEDS_C1_PASS_AND_FOR_STRENGTH_A_SCREEN_CONTINUE(opened, capsys, kind, records, why):
    RECORDS.clear()
    RECORDS.update({(SCREEN_REL if k == "SCREEN" else k): v
                           for k, v in records.items()})
    assert _run(opened, kind) == 4
    assert why in capsys.readouterr().err and not (opened / kind).exists()


def test_THE_ADMISSION_RECORDS_ARE_THE_COMMITTED_ONES():
    """The real reader: a missing record is FileNotFoundError, and it consults git for an uncommitted one."""
    assert MATCH.ADMISSION == {"screen": [("docs/superpowers/evidence/mirror-halving-c1", "verdict", "PASS")],
                               "strength": [("docs/superpowers/evidence/mirror-halving-c1", "verdict", "PASS"),
                                            ("docs/superpowers/evidence/mirror-halving-screen", "screen_verdict",
                                             "CONTINUE")],
                               "strength2": [("docs/superpowers/evidence/mirror-halving-c1", "verdict", "PASS"),
                                             ("docs/superpowers/evidence/mirror-halving-screen", "screen_verdict",
                                              "CONTINUE")]}       # attempt 2: admitted exactly as attempt 1
    with pytest.raises(FileNotFoundError):
        REAL_READ_RECORD("docs/superpowers/evidence/no-such-run")


def test_AN_EXISTING_EVIDENCE_DIRECTORY_REFUSES(opened, capsys):
    (opened / "screen").mkdir()
    assert _run(opened, "screen") == 4 and "one-shot" in capsys.readouterr().err
    assert list((opened / "screen").iterdir()) == []


# ─────────────────────────────── fresh processes ───────────────────────────────

PRELUDE = """
import pathlib, sys, tempfile
from scripts.GPU.alphazero import mirror_halving_c1 as C1
from scripts.GPU.alphazero import mirror_halving_match as MATCH
from scripts.GPU.alphazero import mirror_halving_screen_authorization as SCREEN_GATE
print("MLX_AFTER_IMPORT", MATCH._mlx_loaded(), flush=True)
t = pathlib.Path(tempfile.mkdtemp()); (t / "ckpt.safetensors").write_bytes(b"x")
for k in MATCH.KINDS:
    MATCH.KINDS[k]["evidence"] = str(t / k)
MATCH._real_seams = lambda ckpt: None
MATCH._read_record = lambda rel: {"status": "COMPLETED", "verdict": "PASS", "card_sha256": C1.CARD_SHA256}
def stub(kind, seams, out, admission, head=None):   # never a real run, whatever a mutant does above it
    print("CONCLUDE_REACHED", flush=True)
    return 0, {"status": "STUB"}
MATCH.conclude = stub
"""


def _fresh(body, setting):
    env = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"}
    env.update(PYTHONPATH=str(MATCH.ROOT), PYTHONDONTWRITEBYTECODE="1")
    if setting is not None:
        env["MLX_ENABLE_TF32"] = setting
    code = PRELUDE + body + "\nsys.exit(MATCH.main(['run', 'screen', '--checkpoint', str(t / 'ckpt.safetensors')]))\n"
    return subprocess.run([sys.executable, "-c", code], cwd=MATCH.ROOT, env=env, capture_output=True, text=True,
                          timeout=120)


def test_FRESH_importing_the_entry_loads_no_MLX_and_the_real_checks_pass_to_the_gate():
    p = _fresh("", "0")
    assert "MLX_AFTER_IMPORT False" in p.stdout, p.stdout + p.stderr
    assert p.returncode == 4 and "is closed" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


@pytest.mark.parametrize("setting", [None, "1"])
def test_FRESH_without_the_setting_the_real_check_refuses(setting):
    p = _fresh("SCREEN_GATE.MIRROR_HALVING_SCREEN_AUTHORIZED = True", setting)
    assert p.returncode == 4 and "MLX_ENABLE_TF32" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_MLX_imported_before_the_check_refuses():
    p = _fresh("SCREEN_GATE.MIRROR_HALVING_SCREEN_AUTHORIZED = True\nimport mlx.core", "0")
    assert p.returncode == 4 and "imported" in p.stderr and "CONCLUDE_REACHED" not in p.stdout


def test_FRESH_with_the_setting_the_gate_open_and_admission_every_real_check_passes():
    p = _fresh("SCREEN_GATE.MIRROR_HALVING_SCREEN_AUTHORIZED = True", "0")
    assert p.returncode == 0 and "CONCLUDE_REACHED" in p.stdout, p.stderr


# ─────────────────────────────── the public entry end to end ───────────────────────────────

@pytest.mark.parametrize("outcome, verdict", [(lambda g, r: K.CANDIDATE, "CONTINUE"),
                                              (lambda g, r: K.INCUMBENT, "STOP")])
def test_THE_SCREEN_PLAYS_ALL_64_GAMES_AND_JUDGES_GROSS_REGRESSION_ONLY(opened, monkeypatch, outcome, verdict):
    calls = []
    monkeypatch.setattr(MATCH, "_real_seams", lambda ckpt: seams(stub_play(outcome, calls=calls)))
    assert _run(opened, "screen") == 0
    rec = _record(opened, "screen")
    assert rec["status"] == "COMPLETED" and rec["screen_verdict"] == verdict and rec["strength_verdict"] is None
    assert calls == list(range(64)) and len(rec["pair_scores"]) == 32
    assert rec["admission"] == {C1.EVIDENCE_DIR: "PASS"} and rec["seeds"] == {"base": 777000, "n_slots": 64}
    assert rec["protocol"]["evaluators"] == {K.CANDIDATE: "MirrorAveragingEvaluator", K.INCUMBENT: "FakeEv"}
    rows = [json.loads(x) for x in (opened / "screen" / "games.jsonl").read_text().splitlines()]
    assert [r["game_idx"] for r in rows] == list(range(64))
    assert rec["cost_totals"][K.CANDIDATE]["network_positions"] == rec["cost_totals"][K.INCUMBENT][
        "network_positions"] == 128


@pytest.mark.parametrize("outcome, secs, verdict", [
    (lambda g, r: K.CANDIDATE, (1.0, 1.0), "PASS"),
    (lambda g, r: K.CANDIDATE, (1.2, 1.0), "NOT_COMPUTE_MATCHED"),
    (lambda g, r: K.CANDIDATE if g % 2 else K.INCUMBENT, (1.0, 1.0), "NON_PASS"),
    (lambda g, r: None, (1.0, 1.0), "NON_PASS"),
])
def test_THE_STRENGTH_COMPARISON_PLAYS_1018_GAMES_AND_APPLIES_BOTH_CONDITIONS(opened, monkeypatch, outcome, secs,
                                                                              verdict):
    RECORDS[SCREEN_REL] = dict(SCREEN_CONTINUE)
    monkeypatch.setattr(MATCH, "_real_seams", lambda ckpt: seams(stub_play(outcome, secs=secs)))
    assert _run(opened, "strength") == 0
    rec = _record(opened, "strength")
    assert rec["status"] == "COMPLETED" and rec["strength_verdict"] == verdict and rec["screen_verdict"] is None
    assert len(rec["pair_scores"]) == 509 and rec["strength"]["hoeffding_lower"] == K.hoeffding_lower(
        rec["pair_scores"], 0.05)
    assert rec["guards"]["search_seconds_ratio"] == pytest.approx(secs[0] / secs[1])


def test_A_HALVING_CHOICE_FROM_THE_INCUMBENT_IS_A_VOID(opened, monkeypatch):
    def play(task, red, black, config, capture, replay_dir, classes, on_search):
        on_search(True, 1.0, True)
        on_search(False, 1.0, True)
        return ER.make_result(task, "red", "win", 30, config=config)

    monkeypatch.setattr(MATCH, "_real_seams", lambda ckpt: seams(play))
    assert _run(opened, "screen") == 3
    rec = _record(opened, "screen")
    assert rec["status"] == "VOID" and rec["void"]["step"] == "attribution" and rec["screen_verdict"] is None


@pytest.mark.parametrize("over, step", [({"seed_ok": lambda s: s != 777005}, "seeds"),
                                        ({"read_plan": lambda rel: {"base": 777000, "n_slots": 63}}, "seeds"),
                                        ({"checkpoint_sha1": lambda: "0" * 40}, "checkpoint"),
                                        ({"versions": lambda: {"mlx": "0.31.0", "numpy": "2.4.1",
                                                               "python": "3.14.9"}}, "versions")])
def test_SEEDS_PLAN_CHECKPOINT_AND_VERSIONS_ARE_CHECKED_BEFORE_ANY_GAME(opened, monkeypatch, over, step):
    calls = []
    monkeypatch.setattr(MATCH, "_real_seams", lambda ckpt: seams(stub_play(lambda g, r: K.CANDIDATE, calls=calls),
                                                                 **over))
    assert _run(opened, "screen") == 3
    assert _record(opened, "screen")["void"]["step"] == step and calls == []


def test_A_GAME_THAT_FAILS_MID_RUN_LEAVES_NO_SCORE(opened, monkeypatch):
    inner = stub_play(lambda g, r: K.CANDIDATE)

    def play(task, *a):
        if task.game_idx == 10:
            raise RuntimeError("boom")
        return inner(task, *a)

    monkeypatch.setattr(MATCH, "_real_seams", lambda ckpt: seams(play))
    assert _run(opened, "screen") == 3
    rec = _record(opened, "screen")
    assert rec["status"] == "VOID" and "pair_scores" not in rec and rec["screen_verdict"] is None
    assert len((opened / "screen" / "games.jsonl").read_text().splitlines()) == 10


def test_EVERY_GAME_IS_LOGGED_AS_STARTED_BEFORE_IT_IS_PLAYED(opened, monkeypatch):
    """The closing derivation exposes the started prefix from `game_start` (game = slot), logged BEFORE each game."""
    def play(task, *a):
        events = [json.loads(x) for x in (opened / "screen" / "progress.jsonl").read_text().splitlines()]
        assert events[-1]["event"] == "game_start" and events[-1]["game"] == events[-1]["slot"] == task.game_idx
        if task.game_idx == 3:
            raise RuntimeError("interrupted mid-game")
        return ALL_CANDIDATE(task, *a)

    monkeypatch.setattr(MATCH, "_real_seams", lambda ckpt: seams(play))
    assert _run(opened, "screen") == 3
    events = [json.loads(x) for x in (opened / "screen" / "progress.jsonl").read_text().splitlines()]
    assert [e["game"] for e in events if e["event"] == "game_start"] == [0, 1, 2, 3]
    assert [e["game"] for e in events if e["event"] == "game_end"] == [0, 1, 2]


# ─────────────────────────────── attempt 2: the one fresh attempt, BOINC kept out ───────────────────────────────

CLEAR = {"ok": True, "n_procs": 800, "hits": [], "n_hits": 0, "exempt": ["97693 boinc_project /opt/podman/bin/krunkit"],
         "error": None}
BOINC = {"ok": False, "n_procs": 801, "hits": ["5635 boinc_project hsgamma_FGRP5"], "n_hits": 1, "exempt": [],
         "error": None}
FAILED = {"ok": False, "n_procs": 0, "hits": [], "n_hits": 0, "exempt": [], "error": "OSError: ps failed"}


def scripted(*answers, then=CLEAR):
    """A background check answering in turn, then `then` for ever; the calls are counted."""
    calls = []

    def check():
        calls.append(1)
        return dict(answers[len(calls) - 1] if len(calls) <= len(answers) else then)
    return check, calls


@pytest.fixture
def opened2(monkeypatch, isolated):
    monkeypatch.setattr(STRENGTH2_GATE, "MIRROR_HALVING_STRENGTH2_AUTHORIZED", True)
    RECORDS[SCREEN_REL] = dict(SCREEN_CONTINUE)
    return isolated


def _events(tmp, kind):
    return [json.loads(x) for x in (tmp / kind / "progress.jsonl").read_text().splitlines()]


def test_ATTEMPT_2_S_GATE_IS_CLOSED_IN_THE_SOURCE_and_a_closed_gate_refuses(isolated, capsys):
    assert STRENGTH2_GATE.MIRROR_HALVING_STRENGTH2_AUTHORIZED is False
    RECORDS[SCREEN_REL] = dict(SCREEN_CONTINUE)
    assert _run(isolated, "strength2") == 4 and "MIRROR_HALVING_STRENGTH2_AUTHORIZED is closed" in capsys.readouterr().err
    assert not (isolated / "strength2").exists()


def test_ATTEMPT_2_IS_THE_STRENGTH_COMPARISON_ON_ITS_OWN_PATHS():
    one, two = MATCH.KINDS["strength"], MATCH.KINDS["strength2"]
    assert two["pairs"] == one["pairs"] == 509 and MATCH.ADMISSION["strength2"] == MATCH.ADMISSION["strength"]
    assert two["plan"] == "docs/superpowers/evidence/mirror-halving-strength2-run/plan.json" != one["plan"]
    assert two["gate"] == (STRENGTH2_GATE, "MIRROR_HALVING_STRENGTH2_AUTHORIZED") and two["watch_background"] is True
    assert "watch_background" not in one and "amendment" not in one and "amendment" not in MATCH.KINDS["screen"]
    rel, sha = two["amendment"]
    assert M._sha((MATCH.ROOT / rel).read_bytes()) == sha


def test_ATTEMPT_2_PLAYS_1018_GAMES_CHECKING_FOR_BOINC_BEFORE_LOAD_AFTER_EVERY_GAME_AND_BEFORE_THE_VERDICT(
        opened2, monkeypatch):
    check, calls = scripted()
    monkeypatch.setattr(MATCH, "_real_seams", lambda ckpt: seams(ALL_CANDIDATE, background=check))
    assert _run(opened2, "strength2") == 0
    rec = _record(opened2, "strength2")
    assert rec["status"] == "COMPLETED" and rec["run"] == "strength2" and rec["strength_verdict"] == "PASS"
    assert len(rec["pair_scores"]) == 509 and rec["screen_verdict"] is None
    assert (rec["amendment"], rec["amendment_sha256"]) == MATCH.KINDS["strength2"]["amendment"]
    checks = [e for e in _events(opened2, "strength2") if e["event"] == "bg_check"]
    assert len(calls) == len(checks) == rec["background_checks"] == 1 + 1018 + 1
    assert [c["when"] for c in checks[:2]] == ["before load", "after game 0"] and checks[-1]["when"] == "before the verdict"
    assert all(c["ok"] is True and c["exempt"] == CLEAR["exempt"] for c in checks)


@pytest.mark.parametrize("answer, said", [(BOINC, "BOINC detected: ['5635 boinc_project hsgamma_FGRP5']"),
                                          (FAILED, "the check failed: OSError: ps failed")], ids=["boinc", "failed"])
def test_ATTEMPT_2_BOINC_OR_A_FAILED_CHECK_MID_RUN_IS_A_VOID_WITH_NO_SCORE(opened2, monkeypatch, answer, said):
    check, _calls = scripted(*([CLEAR] * 6), answer)            # before load, then after games 0-4; then game 5
    played = []
    monkeypatch.setattr(MATCH, "_real_seams",
                        lambda ckpt: seams(stub_play(lambda g, r: K.CANDIDATE, calls=played), background=check))
    assert _run(opened2, "strength2") == 3
    rec = _record(opened2, "strength2")
    assert rec["status"] == "VOID" and rec["void"] == {"step": "background_compute", "detail": f"after game 5: {said}"}
    assert "pair_scores" not in rec and "guards" not in rec and rec["strength_verdict"] is None
    assert played == list(range(6)) and len((opened2 / "strength2" / "games.jsonl").read_text().splitlines()) == 6
    last = [e for e in _events(opened2, "strength2") if e["event"] == "bg_check"][-1]
    assert last["when"] == "after game 5" and last["ok"] is False and _events(opened2, "strength2")[-1]["event"] == "end"


@pytest.mark.parametrize("answer", [BOINC, FAILED], ids=["boinc", "failed"])
def test_ATTEMPT_2_BOINC_BEFORE_LOAD_IS_A_VOID_BEFORE_ANY_LOAD_OR_GAME(opened2, monkeypatch, answer):
    check, _calls = scripted(answer)
    loads, played = [], []

    def load():
        loads.append(1)
        return FakeEv()
    monkeypatch.setattr(MATCH, "_real_seams", lambda ckpt: seams(stub_play(lambda g, r: K.CANDIDATE, calls=played),
                                                                 background=check, load_evaluator=load))
    assert _run(opened2, "strength2") == 3
    rec = _record(opened2, "strength2")
    assert rec["void"]["step"] == "background_compute" and rec["void"]["detail"].startswith("before load: ")
    assert loads == [] and played == [] and not (opened2 / "strength2" / "games.jsonl").exists()


def test_ATTEMPT_2_WITHOUT_A_WIRED_CHECK_IS_A_VOID(opened2, monkeypatch):
    monkeypatch.setattr(MATCH, "_real_seams", lambda ckpt: seams(ALL_CANDIDATE))      # background left None
    assert _run(opened2, "strength2") == 3
    assert _record(opened2, "strength2")["void"] == {"step": "background_compute",
                                                     "detail": "before load: no background-compute check is wired"}


@pytest.mark.parametrize("kind", ["screen", "strength"])
def test_ATTEMPT_1_AND_THE_SCREEN_NEVER_CALL_THE_CHECK(opened, monkeypatch, kind):
    RECORDS[SCREEN_REL] = dict(SCREEN_CONTINUE)

    def never():
        raise AssertionError("checked")
    monkeypatch.setattr(MATCH, "_real_seams", lambda ckpt: seams(ALL_CANDIDATE, background=never))
    assert _run(opened, kind) == 0
    rec = _record(opened, kind)
    assert rec["status"] == "COMPLETED" and "background_checks" not in rec and "amendment" not in rec
    assert not any(e["event"] == "bg_check" for e in _events(opened, kind))


def test_ATTEMPT_2_REFUSES_AN_AMENDMENT_THAT_IS_NOT_THE_PINNED_ONE(opened2, monkeypatch, capsys):
    rel, _sha = MATCH.KINDS["strength2"]["amendment"]
    monkeypatch.setitem(MATCH.KINDS["strength2"], "amendment", (rel, "0" * 64))
    assert _run(opened2, "strength2") == 4 and "is not the pinned amendment" in capsys.readouterr().err
    assert not (opened2 / "strength2").exists()


def test_THE_REAL_SEAMS_WIRE_THE_REAL_CHECK():
    """Construction only: the real seams are lazy, so nothing is loaded or read."""
    assert REAL_SEAMS(MATCH.ROOT / "no-such-checkpoint").background is BG.observe
