"""The H-TB CONFIRMATION TEST (plan `docs/superpowers/2026-11-06-htb-strength-plan.md`; card §9.2, §9.3), built only on
the user's review (2026-11-06): `htb_strength run strength`, its wrapper, its ceiling (adopted on the build review) and its seed copies.

CPU only. No test loads the student or calib020_0001, touches the GPU, opens a real gate, runs a seed mode on the real
repository, writes the registry or plays a real game:
* admission runs on the REAL committed fit and screen records (read-only) and on synthetic ones;
* the test runs through the real record boundary with GPU-free fakes and a stubbed game;
* the wrapper and seed copies are checked as exact derivations of the H-TB screen's reviewed files.

EVERY TEST IS ISOLATED, WHATEVER THE GATE SAYS (#74): the autouse fixture makes the real seams (this module's and the
screen's) and both real record boundaries unreachable."""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
import pathlib
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import combination_policy_pilot as CPP
from scripts.GPU.alphazero import eval_runner as ER
from scripts.GPU.alphazero import htb_pilot as H
from scripts.GPU.alphazero import htb_screen as HS
from scripts.GPU.alphazero import htb_strength as HT
from scripts.GPU.alphazero import htb_strength_authorization as GATE
from scripts.GPU.alphazero import mirror_halving_combination as K

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import htb_strength_copies_spec as SPEC  # noqa: E402
import test_htb_screen as TS  # noqa: E402  (the screen's fakes: SideFake, stub_play; reused, not copied)

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = TS.BASE                                  # a fake plan's base: not a registry seed
SCREEN_BASE = BASE - 1000                       # a fake screen block, disjoint from [BASE, BASE + 220)
PLAN_OK = {"kind": HT.PLAN_KIND, "base": BASE, "n_slots": 220}
SCREEN_PLAN = {"kind": HS.PLAN_KIND, "base": SCREEN_BASE, "n_slots": 64}
CLEAR, BOINC = TS.CLEAR, TS.BOINC
_REAL_CONCLUDE = HT.conclude
_REAL_STRENGTH_SEAMS = HT._real_strength_seams
_REAL_SCREEN_SEAMS = HS._real_screen_seams


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def unreachable(*_a, **_k):
        raise AssertionError("a test reached a real seam or the real record boundary")
    for mod, name in ((HT, "_real_strength_seams"), (HT, "conclude"), (HS, "_real_screen_seams"), (HS, "conclude")):
        monkeypatch.setattr(mod, name, unreachable)
    monkeypatch.setenv(HT.SETTING, HT.SETTING_VALUE)
    import mlx.core as mx
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_THE_ISOLATION_IS_IN_FORCE():
    for fn in (lambda: HT._real_strength_seams(pathlib.Path("/x")), lambda: HT.conclude(None, None, {}),
               lambda: HS._real_screen_seams(pathlib.Path("/x"))):
        with pytest.raises(AssertionError, match="real seam"):
            fn()


# ─────────────────────────────── gate, pins, entry ───────────────────────────────

def test_THE_GATE_IS_A_CLOSED_SINGLE_DECLARATION_and_the_closed_gate_refuses_after_the_pins_hold(monkeypatch, capsys):
    assert HT.GATE is GATE and HT.GATE_NAME == "HTB_STRENGTH_AUTHORIZED" and GATE.HTB_STRENGTH_AUTHORIZED is False
    tree = ast.parse(pathlib.Path(GATE.__file__).read_text())
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert len(body) == 1 and body[0].targets[0].id == HT.GATE_NAME
    monkeypatch.setattr(CPP, "_mlx_loaded", lambda: False)
    assert HT.main(["run", "strength", "--checkpoint", "/x"]) == HT.EXIT_REFUSED
    assert "HTB_STRENGTH_AUTHORIZED is closed" in capsys.readouterr().err


def test_THE_SCREEN_S_GATE_DOES_NOT_OPEN_THIS_ONE(monkeypatch, capsys):
    monkeypatch.setattr(CPP, "_mlx_loaded", lambda: False)
    monkeypatch.setattr(HS.GATE, "HTB_SCREEN_AUTHORIZED", True)
    assert HT.main(["run", "strength", "--checkpoint", "/x"]) == HT.EXIT_REFUSED
    assert "HTB_STRENGTH_AUTHORIZED is closed" in capsys.readouterr().err


def test_THE_PINS_ARE_THE_FILES_and_include_the_screen_s_module_and_plan():
    assert HT.CARD == H.CARD and HT.CARD_SHA256 == H.CARD_SHA256
    for rel, sha in ((HT.PLAN_DOC, HT.PLAN_DOC_SHA256), (HT.CARD, HT.CARD_SHA256), *HT.PINNED.items()):
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == sha, rel
    assert {k: v for k, v in HT.PINNED.items() if k in HS.PINNED} == HS.PINNED
    assert HT.PINNED["scripts/GPU/alphazero/htb_screen.py"] == hashlib.sha256(
        pathlib.Path(HS.__file__).read_bytes()).hexdigest()
    assert HT.PINNED[HS.PLAN_DOC] == HS.PLAN_DOC_SHA256
    assert HT.VERSIONS == HS.VERSIONS and HT.STUDENT is HS.STUDENT and HT.REFERENCE_DIGEST == HS.REFERENCE_DIGEST


@pytest.mark.parametrize("env, loaded, why", [("1", False, "MLX_ENABLE_TF32 is '1'"), (None, False, "is None"),
                                              ("0", True, "MLX was imported")], ids=["tf32-1", "unset", "mlx-loaded"])
def test_THE_PREFLIGHT_REFUSES_A_WRONG_SETTING_OR_A_LOADED_MLX(monkeypatch, capsys, env, loaded, why):
    if env is None:
        monkeypatch.delenv(HT.SETTING, raising=False)
    else:
        monkeypatch.setenv(HT.SETTING, env)
    monkeypatch.setattr(CPP, "_mlx_loaded", lambda: loaded)
    monkeypatch.setattr(GATE, "HTB_STRENGTH_AUTHORIZED", True)
    assert HT.main(["run", "strength", "--checkpoint", "/x"]) == HT.EXIT_REFUSED
    assert why in capsys.readouterr().err


@pytest.mark.parametrize("name", ["PLAN_DOC_SHA256", "CARD_SHA256"])
def test_A_CHANGED_PIN_REFUSES(monkeypatch, capsys, name):
    monkeypatch.setattr(CPP, "_mlx_loaded", lambda: False)
    monkeypatch.setattr(HT, name, "0" * 64)
    assert HT.main(["run", "strength", "--checkpoint", "/x"]) == HT.EXIT_REFUSED
    assert "is not the pinned bytes" in capsys.readouterr().err


def test_A_CHANGED_SCREEN_MODULE_PIN_REFUSES(monkeypatch, capsys):
    monkeypatch.setattr(CPP, "_mlx_loaded", lambda: False)
    monkeypatch.setattr(HT, "PINNED", {**HT.PINNED, "scripts/GPU/alphazero/htb_screen.py": "0" * 64})
    assert HT.main(["run", "strength", "--checkpoint", "/x"]) == HT.EXIT_REFUSED
    assert "htb_screen.py is not the pinned bytes" in capsys.readouterr().err


def test_RUN_STAYS_ONE_SHOT_even_with_its_gate_open(monkeypatch, capsys):
    monkeypatch.setattr(CPP, "_mlx_loaded", lambda: False)
    monkeypatch.setattr(GATE, "HTB_STRENGTH_AUTHORIZED", True)
    monkeypatch.setattr(HT, "EVIDENCE", HS.EVIDENCE)                      # an existing destination
    assert HT.main(["run", "strength", "--checkpoint", str(ROOT / HT.CARD)]) == HT.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "exists: the run is one-shot" in err or f"{HT.STUDENT['path']} is not a file" in err


def test_USAGE_IS_EXACTLY_RUN_STRENGTH_WITH_A_CHECKPOINT(capsys):
    for argv in ([], ["run", "strength"], ["run", "screen", "--checkpoint", "/x"], ["time"]):
        assert HT.main(argv) == HT.EXIT_USAGE
    assert "run strength --checkpoint" in capsys.readouterr().err


def test_IMPORTING_THE_MODULE_LOADS_NO_MLX():
    out = subprocess.run([sys.executable, "-c", "import sys; import scripts.GPU.alphazero.htb_strength; "
                          "print(any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules))"],
                         cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert out.returncode == 0 and out.stdout.strip() == "False", out.stderr


# ─────────────────────────────── admission (plan §4) ───────────────────────────────

def test_REAL_STATE_THE_COMMITTED_FIT_AND_SCREEN_ADMIT_THE_CONFIRMATION_TEST():
    adm = HT.admission(CPP._read_record, CPP._read_outer, CPP._read_committed_bytes)
    assert adm == {f"{H.EVIDENCE_FIT}/record.json": "ELIGIBLE", f"{HS.FIT_RUN}/outer.json": "COMPLETED",
                   f"{HS.EVIDENCE}/record.json": "CONTINUE", f"{HS.RUN_DIR}/outer.json": "COMPLETED"}


def _good_screen():
    rec = json.loads((ROOT / HS.EVIDENCE / "record.json").read_text())
    outer = json.loads((ROOT / HS.RUN_DIR / "outer.json").read_text())
    return rec, outer


def _readers(rec, outer):
    def read(rel):
        if rec is None:
            raise FileNotFoundError(f"{rel}/record.json")
        assert rel == HS.EVIDENCE
        return rec
    return read, (lambda rel: outer)


@pytest.mark.parametrize("edit, why", [
    (lambda r, o: r.update(kind="htb-fit-record/1"), "kind is"),
    (lambda r, o: r.update(status="VOID"), "status is 'VOID'"),
    (lambda r, o: r.update(verdict="STOP"), "verdict is 'STOP'"),
    (lambda r, o: r["screen"].update(verdict="STOP"), "own verdict is not CONTINUE"),
    (lambda r, o: r.update(card_sha256="0" * 64), "card_sha256"),
    (lambda r, o: r.update(plan_doc_sha256="0" * 64), "plan_doc_sha256"),
    (lambda r, o: r.update(written_by="wrapper"), "written_by"),
    (lambda r, o: r["student"].update(sha256="0" * 64), "student is not the pinned one"),
    (lambda r, o: r["digests"].update(candidate="0" * 64), "digests are not"),
    (lambda r, o: r["digests_at_end"].update(reference="0" * 64), "digests_at_end are not"),
    (lambda r, o: o.update(outcome="TIMEOUT", inner_exit=124, outer_exit=124), "outer.json inner_exit"),
    (lambda r, o: o.update(survivors="123 "), "outer.json survivors"),
    (lambda r, o: o.update(gate_restored=False), "outer.json gate_restored"),
    (lambda r, o: o.update(record_by="wrapper"), "outer.json record_by"),
], ids=["kind", "status", "verdict", "screen-verdict", "card", "plan", "written-by", "student", "digests",
        "digests-at-end", "timeout", "survivors", "gate", "record-by"])
def test_ADMISSION_REFUSES_ANYTHING_BUT_THE_ONE_CONTINUE_SCREEN_WITH_CONFIRMED_CLEANUP(edit, why):
    rec, outer = _good_screen()
    HT.screen_admission(*_readers(rec, outer))                            # the control: the real records admit
    rec, outer = copy.deepcopy(rec), copy.deepcopy(outer)
    edit(rec, outer)
    with pytest.raises(HT.Refused, match="does not admit") as e:
        HT.screen_admission(*_readers(rec, outer))
    assert why in str(e.value)


def test_ADMISSION_REFUSES_A_MISSING_SCREEN_RECORD():
    _rec, outer = _good_screen()
    with pytest.raises(HT.Refused, match="does not exist"):
        HT.screen_admission(*_readers(None, outer))


def test_A_REFUSED_FIT_IS_REFUSED_BY_THE_PUBLIC_ENTRY(monkeypatch, capsys):
    monkeypatch.setattr(CPP, "_mlx_loaded", lambda: False)
    monkeypatch.setattr(GATE, "HTB_STRENGTH_AUTHORIZED", True)
    monkeypatch.setattr(HS, "REFERENCE_DIGEST", "0" * 64)                 # the fit no longer names this reference
    assert HT.main(["run", "strength", "--checkpoint", "/x"]) == HT.EXIT_REFUSED
    assert "the fit does not admit the screen" in capsys.readouterr().err


# ─────────────────────────────── the rule (plan §6) ───────────────────────────────

def _g(ev=1.0, sec=1.0):
    return {"evaluations_ratio": ev, "search_seconds_ratio": sec, "evaluations_within": ev <= 1.05,
            "search_seconds_within": sec <= 1.10, "guards": dict(K.GUARDS)}


def _mean_over_220(k):
    """110 pair scores whose mean is exactly k / 220, built from pair scores 1, 0.5 and 0 only (no drawn game). A drawn
    game scores half a point, so a pair can also score 0.75 or 0.25 and reachable means are multiples of 1 / 440."""
    ones, half = divmod(k, 2)
    return [1.0] * ones + [0.5] * half + [0.0] * (110 - ones - half)


def _with_a_draw(k_halves):
    """110 pair scores whose mean is exactly (k_halves / 2) / 220 with k_halves odd: one pair is a win and a draw (0.75)."""
    assert k_halves % 2 == 1
    k = (k_halves - 1) // 2                                               # 135.5 = 135 + 0.5
    ones, half = divmod(k, 2)
    assert half == 1                                                      # 67 wins, one 0.5 pair -> make it 0.75
    return [1.0] * ones + [0.75] + [0.0] * (110 - ones - 1)


def test_THE_BOUND_IS_THE_CARD_S_ONE_SIDED_FORMULA_and_not_the_two_sided_one():
    assert HT.HALF_WIDTH == math.sqrt(math.log(20) / 220) and round(HT.HALF_WIDTH, 6) == 0.116692
    assert round(0.5 + HT.HALF_WIDTH, 5) == 0.61669 and HT.STRENGTH == {"pairs": 110, "alpha": 0.05, "null": 0.5}
    two_sided = math.sqrt(math.log(2 / 0.05) / 220)
    assert round(two_sided, 6) == 0.129490 and HT.HALF_WIDTH < two_sided
    pairs = _mean_over_220(137)                                           # 0.6227: PASS here, not by ln(2 / alpha)
    assert HT.strength_verdict(pairs, _g())["verdict"] == "PASS" and K.hoeffding_lower(pairs, 0.05) < 0.5
    # reachable means are multiples of 1 / 440 (a drawn game scores half a point): 272/440 = 136/220 is the least
    # that passes and 271/440 = 135.5/220 the greatest that does not
    passing = [k for k in range(441) if k / 440 - HT.HALF_WIDTH > 0.5]
    assert passing[0] == 272 and sum(_with_a_draw(271)) == 67.75


@pytest.mark.parametrize("pairs, g, verdict, n_reasons", [
    ([1.0] * 110, _g(), "PASS", 0),
    (_mean_over_220(136), _g(), "PASS", 0),                               # 0.6182: the least reachable mean that passes
    (_with_a_draw(271), _g(), "NON_PASS", 1),                             # 135.5/220 = 0.6159, a draw: the greatest that does not
    (_mean_over_220(135), _g(), "NON_PASS", 1),                           # 0.6136: the greatest with no drawn game
    ([0.5] * 110, _g(), "NON_PASS", 1),
    ([1.0] * 110, _g(ev=1.06), "NON_PASS", 1),
    ([1.0] * 110, _g(sec=1.11), "NON_PASS", 1),
    ([1.0] * 110, _g(ev=1.05, sec=1.10), "PASS", 0),                      # the guards are inclusive
    ([0.0] * 110, _g(ev=2.0, sec=2.0), "NON_PASS", 3),
], ids=["all-wins", "least-pass", "greatest-non-pass-a-draw", "greatest-non-pass-no-draw", "half", "evaluations",
        "seconds", "guards-at-limit", "all"])
def test_THE_RULE_IS_PASS_ONLY_ABOVE_THE_BOUND_WITH_BOTH_GUARDS(pairs, g, verdict, n_reasons):
    v = HT.strength_verdict(pairs, g)
    assert v["verdict"] == verdict and len(v["reasons"]) == n_reasons, v
    assert v["mean_pair_score"] == pytest.approx(sum(pairs) / 110)
    assert v["lower_bound"] == v["mean_pair_score"] - HT.HALF_WIDTH and v["one_sided"] is True
    assert v["guards"] == {"evaluations": 1.05, "search_seconds": 1.10} and "never pooled with the screen" in v["note"]


def test_THE_RULE_REFUSES_A_WRONG_COHORT_OR_GUARD_SET():
    with pytest.raises(ValueError, match="109 pairs"):
        HT.strength_verdict([1.0] * 109, _g())
    with pytest.raises(ValueError, match="32 pairs"):
        HT.strength_verdict([1.0] * 32, _g())                             # the screen's cohort is not this one
    with pytest.raises(ValueError, match="guards"):
        HT.strength_verdict([1.0] * 110, {**_g(), "guards": {"evaluations": 1.06, "search_seconds": 1.10}})
    assert HT.GUARDS == {"evaluations": 1.05, "search_seconds": 1.10}


# ─────────────────────────────── the test, through the real record boundary, on fakes ───────────────────────────────

def plans(strength=None, screen=None):
    table = {HT.PLAN: dict(PLAN_OK if strength is None else strength),
             HS.PLAN: dict(SCREEN_PLAN if screen is None else screen)}
    return lambda rel: table[rel]


def seams(play, loaded=None, **over):
    """The screen's fake seams (test_htb_screen.screen_seams), with this test's two plans."""
    return TS.screen_seams(play, loaded=loaded, **{"read_plan": plans(), **over})


def _run(tmp_path, s):
    return _REAL_CONCLUDE(lambda o, r, lg: HT._strength(s, o, r, lg), tmp_path / "ev", {"fit": "ELIGIBLE",
                                                                                         "screen": "CONTINUE"})


def _events(tmp_path):
    return [json.loads(x) for x in (tmp_path / "ev" / "progress.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("outcome, mean, verdict", [
    (lambda g, cr: HT.CANDIDATE, 1.0, "PASS"),
    (lambda g, cr: HT.REFERENCE, 0.0, "NON_PASS"),
    (lambda g, cr: None, 0.5, "NON_PASS"),
    (TS._wins_first(136), 136 / 220, "PASS"),
    (TS._wins_first(135.5), 135.5 / 220, "NON_PASS"),                     # game 135 drawn: pair 67 scores 0.75
    (TS._wins_first(135), 135 / 220, "NON_PASS"),
], ids=["all-candidate", "all-reference", "all-drawn", "least-pass", "greatest-non-pass-a-draw",
        "greatest-non-pass-no-draw"])
def test_THE_TEST_PLAYS_ALL_220_GAMES_WITH_THE_WHOLE_STUDENT_AND_APPLIES_THE_RULE(tmp_path, outcome, mean, verdict):
    calls, loaded = [], []
    code, rec = _run(tmp_path, seams(TS.stub_play(outcome, calls=calls), loaded=loaded))
    assert code == HT.EXIT_COMPLETED and rec["status"] == "COMPLETED", rec.get("void")
    assert rec["kind"] == "htb-strength-record/1" and rec["run"] == "strength" and rec["card_sha256"] == H.CARD_SHA256
    assert rec["plan_doc_sha256"] == HT.PLAN_DOC_SHA256 and rec["admission"] == {"fit": "ELIGIBLE", "screen": "CONTINUE"}
    assert calls == list(range(220)) and [ev.network.name for ev in loaded] == ["student", "reference"]
    assert rec["strength"]["pairs"] == 110 and rec["strength"]["mean_pair_score"] == pytest.approx(mean)
    assert rec["verdict"] == rec["strength"]["verdict"] == verdict
    assert rec["seeds"] == {"base": BASE, "n_slots": 220} and rec["screen_block_excluded"] == [SCREEN_BASE, SCREEN_BASE + 64]
    want = {HT.CANDIDATE: HT.STUDENT["state_digest"], HT.REFERENCE: HT.REFERENCE_DIGEST}
    assert rec["digests"] == rec["digests_at_end"] == want and rec["opening_probe"]["max_prior_abs_diff"] > 0
    assert rec["protocol"]["evaluators"] == {k: "MirrorAveragingEvaluator" for k in (HT.CANDIDATE, HT.REFERENCE)}
    agents = rec["protocol"]["agents"]
    assert [a["mcts_sims"] for a in agents] == [200, 200]
    assert agents[0]["checkpoint"] == str(HT.STUDENT["path"]) and agents[1]["checkpoint"] == HT.M.CHECKPOINT
    games = [json.loads(x) for x in (tmp_path / "ev" / "games.jsonl").read_text().splitlines()]
    assert [g["slot"] for g in games] == list(range(220)) and games[0]["pairing_id"].startswith("htb-strength")
    for side in (HT.CANDIDATE, HT.REFERENCE):
        assert rec["cost_totals"][side]["network_infer_calls"] == 440 and rec["cost_totals"][side]["searches"] == 440
    assert rec["guards"]["evaluations_ratio"] == 1.0 and rec["background_checks"] == 220 + 2
    assert "screen" not in rec


@pytest.mark.parametrize("kw, reason", [
    ({"secs": (0.30, 0.25)}, "search-seconds ratio"),                   # 1.2 > 1.10
    ({"extra_candidate_calls": 1}, "evaluations ratio"),                # 2.0 > 1.05
], ids=["seconds", "evaluations"])
def test_A_FAILED_COMPUTE_GUARD_MAKES_A_WINNING_TEST_NON_PASS(tmp_path, kw, reason):
    code, rec = _run(tmp_path, seams(TS.stub_play(lambda g, cr: HT.CANDIDATE, **kw)))
    assert code == HT.EXIT_COMPLETED and rec["strength"]["mean_pair_score"] == 1.0
    assert rec["verdict"] == "NON_PASS" and len(rec["strength"]["reasons"]) == 1
    assert reason in rec["strength"]["reasons"][0]


def test_EVERY_GAME_IS_LOGGED_AS_STARTED_BEFORE_IT_IS_PLAYED_and_BOINC_is_read_around_every_game(tmp_path):
    _run(tmp_path, seams(TS.stub_play(lambda g, cr: HT.CANDIDATE)))
    ev = _events(tmp_path)
    names = [e["event"] for e in ev]
    assert names[0] == "start" and names[1] == "bg_check" and ev[1]["when"] == "before load"
    starts = [e for e in ev if e["event"] == "game_start"]
    assert [e["game"] for e in starts] == [e["slot"] for e in starts] == list(range(220))
    for g in range(220):
        s = next(i for i, e in enumerate(ev) if e["event"] == "game_start" and e["game"] == g)
        t = next(i for i, e in enumerate(ev) if e["event"] == "game_end" and e["game"] == g)
        assert s < t and ev[t + 1]["event"] == "bg_check" and ev[t + 1]["when"] == f"after game {g}"
    assert [e["when"] for e in ev if e["event"] == "bg_check"][-1] == "before the verdict" and names[-1] == "end"


@pytest.mark.parametrize("over, step, why", [
    ({"versions": lambda: {**HT.VERSIONS, "python": "3.14.9"}}, "versions", "are not the plan's pinned"),
    ({"read_plan": plans(strength={**PLAN_OK, "n_slots": 221})}, "seeds", "not exactly 220 slots"),
    ({"read_plan": plans(strength={**PLAN_OK, "n_slots": 64})}, "seeds", "not exactly 220 slots"),
    ({"read_plan": plans(strength={**PLAN_OK, "base": str(BASE)})}, "seeds", "not exactly 220 slots"),
    ({"read_plan": plans(strength={**PLAN_OK, "kind": HS.PLAN_KIND})}, "seeds", "not exactly 220 slots"),
    ({"seed_ok": lambda s: s != BASE + 219}, "seeds", "1 seeds not ACCOUNTED"),
    ({"read_plan": plans(screen={**SCREEN_PLAN, "base": BASE + 219})}, "seeds", "overlaps the screen's"),
    ({"read_plan": plans(screen={**SCREEN_PLAN, "base": BASE - 63})}, "seeds", "overlaps the screen's"),
    ({"read_plan": plans(screen={"kind": HS.PLAN_KIND})}, "seeds", "unreadable"),
    ({"read_plan": plans(screen={**SCREEN_PLAN, "kind": HT.PLAN_KIND})}, "seeds", "unreadable"),
    ({"checkpoint_sha1": lambda: "0" * 40}, "checkpoint", "is not calib020_0001's"),
    ({"student_sha256": lambda: "0" * 64}, "candidate", "sha256"),
    ({"load_candidate": lambda: TS.SideFake("student", "0" * 64, 7)}, "candidate", "is not the student's"),
    ({"load_reference": lambda: TS.SideFake("reference", HT.STUDENT["state_digest"], 8)}, "reference",
     "is not calib020_0001's"),
    ({"load_candidate": TS._same_priors}, "candidate", "opening priors equal"),
    ({"background": lambda: dict(BOINC)}, "background_compute", "before load: BOINC detected"),
], ids=["versions", "plan-221", "plan-screen-size", "plan-base", "plan-kind", "seed", "overlap-top", "overlap-bottom",
        "screen-plan-base", "screen-plan-kind", "checkpoint", "student-file", "student-digest", "reference-digest",
        "probe-equal", "boinc-before-load"])
def test_EACH_PRECONDITION_IS_A_VOID_BEFORE_ANY_GAME(tmp_path, over, step, why):
    calls = []
    code, rec = _run(tmp_path, seams(TS.stub_play(lambda g, cr: HT.CANDIDATE, calls=calls), **over))
    assert code == HT.EXIT_VOID and rec["status"] == "VOID" and rec["void"]["step"] == step, rec.get("void")
    assert why in rec["void"]["detail"] and calls == [] and rec["verdict"] is None


def test_ADJACENT_BLOCKS_DO_NOT_OVERLAP(tmp_path):
    for screen_base in (BASE - 64, BASE + 220):
        code, rec = _run(tmp_path / str(screen_base), seams(TS.stub_play(lambda g, cr: HT.CANDIDATE),
                                                            read_plan=plans(screen={**SCREEN_PLAN, "base": screen_base})))
        assert code == HT.EXIT_COMPLETED, rec.get("void")


def test_BOTH_SIDES_SHARING_A_NETWORK_IS_A_VOID(tmp_path):
    one = TS.SideFake("student", HT.STUDENT["state_digest"], 7)
    code, rec = _run(tmp_path, seams(TS.stub_play(lambda g, cr: HT.CANDIDATE), load_candidate=lambda: one,
                                     load_reference=lambda: one))
    assert code == HT.EXIT_VOID and rec["void"]["step"] == "load"


@pytest.mark.parametrize("side, step", [("student", "candidate"), ("reference", "reference")])
def test_A_SIDE_WHOSE_STATE_CHANGES_DURING_THE_RUN_IS_A_VOID_WITH_NO_SCORE(tmp_path, side, step):
    loaded = []

    def play(task, red, black, *a):
        if task.game_idx == 219:
            next(ev for ev in loaded if ev.network.name == side).network.digest = "changed"
        return TS.stub_play(lambda g, cr: HT.CANDIDATE)(task, red, black, *a)
    code, rec = _run(tmp_path, seams(play, loaded=loaded))
    assert code == HT.EXIT_VOID and rec["void"]["step"] == step and "at the end" in rec["void"]["detail"]
    assert rec["verdict"] is None and "strength" not in rec and "pair_scores" not in rec


@pytest.mark.parametrize("kw, step", [({"fail_at": 150}, "exception"), ({"searches": (2, 1)}, "attribution")],
                         ids=["game-fails", "count"])
def test_A_GAME_THAT_FAILS_OR_MISCOUNTS_ITS_SEARCHES_LEAVES_NO_SCORE(tmp_path, kw, step):
    code, rec = _run(tmp_path, seams(TS.stub_play(lambda g, cr: HT.CANDIDATE, **kw)))
    assert code == HT.EXIT_VOID and rec["void"]["step"] == step, rec.get("void")
    assert rec["verdict"] is None and "strength" not in rec and "pair_scores" not in rec


def test_BOINC_MID_RUN_IS_A_VOID_WITH_NO_SCORE(tmp_path):
    seen = iter([CLEAR] * 100 + [BOINC])
    code, rec = _run(tmp_path, seams(TS.stub_play(lambda g, cr: HT.CANDIDATE), background=lambda: next(seen)))
    assert code == HT.EXIT_VOID and rec["void"]["step"] == "background_compute" and rec["verdict"] is None


# ─────────────────────────────── reuse: the loop, the boundary, the seams ───────────────────────────────

def test_THE_LOOP_IS_THE_SCREEN_S_TEXT_EXACTLY_which_is_the_cycle_2_screen_s():
    ours = TS._loop(inspect.getsource(HT._strength))
    assert ours == TS._loop(inspect.getsource(HS._screen)) == TS._loop(inspect.getsource(CPP._screen))
    assert ours.count("\n") == 30
    assert HT._watch is HS._watch is CPP._watch and HT.opening_probe is HS.opening_probe is CPP.opening_probe
    assert HT._identities is HS._identities and HT.ScreenSeams is HS.ScreenSeams


def _body_without_docstring(fn):
    node = ast.parse(inspect.getsource(fn).lstrip()).body[0]
    node.body = node.body[1:] if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
        else node.body
    return ast.unparse(node)


def test_THE_RECORD_BOUNDARY_IS_THE_SCREEN_S_with_only_the_run_name_changed():
    ours, screen = _body_without_docstring(_REAL_CONCLUDE), _body_without_docstring(TS._REAL_CONCLUDE)
    assert screen.count("'run': 'screen'") == 1 and ours == screen.replace("'run': 'screen'", "'run': 'strength'")
    assert _body_without_docstring(HT._preflight) == _body_without_docstring(HS._preflight)


def test_THE_REAL_SEAMS_ARE_THE_SCREEN_S(monkeypatch):
    """Built, never called with a real model: the screen's real seams, unchanged -- both sides through the match's
    loader, its loop, between-games step, the registry's seed rule and the background check."""
    from scripts.GPU.alphazero import background_compute as BG
    from scripts.GPU.alphazero import mirror_halving_match as MHM
    monkeypatch.setattr(HS, "_real_screen_seams", _REAL_SCREEN_SEAMS)
    real = _REAL_STRENGTH_SEAMS(pathlib.Path("/nonexistent/model.safetensors"))
    assert type(real) is HS.ScreenSeams and real.play_and_build is K.play_and_build
    assert real.between_games is MHM._between_games and real.background is BG.observe and real.digest is H._digest
    seen = []
    monkeypatch.setattr(ER, "_default_evaluator_factory", lambda path: seen.append(path) or path)
    assert real.load_candidate() == str(HT.STUDENT["path"]) and real.load_reference() == "/nonexistent/model.safetensors"
    assert seen == [str(HT.STUDENT["path"]), "/nonexistent/model.safetensors"]
    assert real.read_plan(HS.PLAN) == json.loads((ROOT / HS.PLAN).read_text())


# ─────────────────────────────── the wrapper and the seed copies ───────────────────────────────

@pytest.mark.parametrize("kind", ["wrapper", "survey", "closing"])
def test_EACH_COPY_IS_THE_DECLARED_DERIVATION_OF_THE_SCREEN_S_REVIEWED_FILE(kind):
    src, sha = SPEC.SOURCES[kind]
    raw = (ROOT / src).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sha
    assert (ROOT / SPEC.TARGETS[kind]).read_text() == SPEC.derive(kind, raw.decode())
    from tests import htb_screen_copies_spec as SCREEN_SPEC
    assert src == SCREEN_SPEC.TARGETS[kind]                                # the screen's copies, the ones that ran


def test_THE_WRAPPER_RUNS_THIS_MODULE_BEHIND_ITS_OWN_CLOSED_GATE_and_was_never_run():
    p = ROOT / SPEC.TARGETS["wrapper"]
    text = p.read_text()
    assert os.access(p, os.X_OK) and subprocess.run(["bash", "-n", str(p)]).returncode == 0
    assert f"\nTERM_S={SPEC.TERM_S}\n" in text and "\nGRACE_S=10\n" in text and SPEC.TERM_S == 11690
    assert "\nGATE=HTB_STRENGTH_AUTHORIZED\n" in text and \
        "\nGATE_REL=scripts/GPU/alphazero/htb_strength_authorization.py\n" in text
    assert f"\nRUN_REL={HT.EVIDENCE}\n" in text and f"\nOUT_REL={HT.RUN_DIR}\n" in text
    assert "\nMODULE=scripts.GPU.alphazero.htb_strength\n" in text and \
        '-m "$MODULE" run strength --checkpoint "$CKPT"' in text
    assert f"\nRECORD_KIND={HT.RECORD_KIND}\n" in text and "MLX_ENABLE_TF32=0" in text
    assert "  3) outcome=VOID ;;\n  4) outcome=REFUSED ;;\n  2) outcome=USAGE ;;\n" in text
    assert "htb_screen" not in text.split("\nset -u\n", 1)[1]                # the body names only this run
    assert sorted(x.name for x in p.parent.iterdir()) == ["01_wrapper.sh"]   # no plan, no run output
    assert not (ROOT / HT.EVIDENCE).exists()


def _module(kind):
    spec = importlib.util.spec_from_file_location(f"_test_htb_strength_{kind}", ROOT / SPEC.TARGETS[kind])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_THE_SURVEY_COPY_SERVES_THIS_TEST_S_PLAN_and_loads_the_pinned_method():
    S = _module("survey")
    assert S.N_SLOTS == 220 and S.HERE == "docs/superpowers/evidence/htb-strength-seed-survey" and S.PLAN == HT.PLAN
    assert S.build_plan(BASE, "ab" * 32) == {"kind": HT.PLAN_KIND, "base": BASE, "n_slots": 220,
                                            "survey_record_sha256": "ab" * 32}
    assert S.slot_lines((BASE, BASE + 220))[219] == f"    SLOT 219 seed {BASE + 219}"
    survey, _proof, SEL = S.load_method()
    assert (survey.N, survey.PLAN, survey.HERE) == (220, HT.PLAN, S.HERE) and SEL.BAND[0] == 2027 * 10 ** 5
    assert not any((ROOT / rel).exists() for rel in S.OUT.values())          # no mode has run


def test_REAL_STATE_NO_SEED_STEP_HAS_RUN_and_the_gate_is_closed():
    from scripts.GPU.alphazero import e4_screen_reference as REG
    assert not (ROOT / HT.PLAN).exists() and not (ROOT / HT.EVIDENCE).exists()
    assert not (ROOT / "docs/superpowers/evidence/htb-strength-seed-closing/02_derive_usage_run.txt").exists()
    screen = json.loads((ROOT / HS.PLAN).read_text())
    block = (screen["base"], screen["base"] + screen["n_slots"])
    assert REG.ACCOUNTED_SEED_INTERVALS[-1] == REG.EXPOSED_SEED_INTERVALS[-1] == REG.RETIRED_SEED_INTERVALS[-1] == block
    assert all(REG.seed_is_unavailable(s) for s in range(*block))         # the screen's seeds can never be reused
    assert GATE.HTB_STRENGTH_AUTHORIZED is False


def test_THE_CLOSING_COPY_READS_THIS_TEST_S_RUN_and_its_controls_all_pass():
    C = _module("closing")
    assert (C.RUN, C.RUN_DIR, C.N_SLOTS) == (HT.EVIDENCE, HT.RUN_DIR, 220)
    controls = C._controls(BASE)
    assert len(controls) == 15 and all(ok for _label, ok, _d in controls), controls
    assert not (ROOT / C.OUT).exists()


# ─────────────────────────────── the adopted ceiling (plan §9) ───────────────────────────────

def test_THE_ADOPTED_CEILING_RECOMPUTES_FROM_THE_COMMITTED_CYCLE_2_STRENGTH_COMPARISON():
    """The user's review (2026-11-06): TERM 11,690 s, KILL 11,700 s, cleanup through 11,710 s -- the historical
    longer-game pace with the same 1.5 allowance; the proposed 6,600 s (from the screen's short games) not adopted."""
    ev = ROOT / "docs/superpowers/evidence"
    c2_outer = json.loads((ev / "combination-policy-cycle2-strength-run/outer.json").read_text())
    c2 = [json.loads(x) for x in (ev / "combination-policy-cycle2-strength/games.jsonl").read_text().splitlines()]
    assert len(c2) == 1018 and c2_outer["wall_seconds"] == 35978
    assert round(c2_outer["wall_seconds"] / 1018, 1) == 35.3 and round(sum(g["n_moves"] for g in c2) / 1018, 1) == 63.7
    proj = c2_outer["wall_seconds"] * 220 / 1018
    ceiling = math.ceil(proj * 1.5 / 60) * 60
    assert (round(proj, 1), round(proj * 1.5, 1), ceiling) == (7775.2, 11662.8, 11700)
    assert (SPEC.TERM_S, SPEC.TERM_S + 10, SPEC.TERM_S + 20) == (11690, ceiling, 11710)
    c2_slow = max(g["seconds"] / g["n_moves"] for g in c2 if g["reason"] == "state_cap")
    assert round(c2_slow, 4) == 0.6272 and int((SPEC.TERM_S - proj) / (280 * c2_slow - proj / 220)) == 27
    # the expected runtime may still follow the screen's faster pace
    outer = json.loads((ev / "htb-screen-run/outer.json").read_text())
    rec = json.loads((ev / "htb-screen/record.json").read_text())
    games = [json.loads(x) for x in (ev / "htb-screen/games.jsonl").read_text().splitlines()]
    assert rec["versions"] == HT.VERSIONS and len(games) == 64 and outer["wall_seconds"] == 1279
    s_proj = outer["wall_seconds"] * 220 / 64
    slow = max(g["seconds"] / g["n_moves"] for g in games)
    assert round(s_proj, 1) == 4396.6 and round(slow, 4) == 0.5492
    assert int((SPEC.TERM_S - s_proj) / (280 * slow - outer["wall_seconds"] / 64)) == 54
    per_ply = sum(g["seconds"] for g in games) / sum(g["n_moves"] for g in games)
    assert round(SPEC.TERM_S / 220, 1) == 53.1 and round(SPEC.TERM_S / 220 / per_ply) == 99 and round(per_ply, 3) == 0.536
    doc = (ROOT / HT.PLAN_DOC).read_text()
    assert "**11,700 s — TERM at 11,690 s, KILL 10 s later, cleanup\n  observation through 11,710 s.**" in doc
    assert "## 9. The ceiling (ADOPTED by the user's review, 2026-11-06)" in doc and "PROPOSED" not in doc
    assert "multiples of 1/440" in doc and "**135.5/220 = 0.6159**" in doc
