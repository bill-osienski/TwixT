"""The A8.8 step-5 collision proof (card §18 A8.1-A8.2) -- its controls, SYNTHETIC ONLY.

The proof script is imported by path; it is never RUN here, and nothing is registered. Its pins, the
survey record's parse, the fail-closed enumeration, the expected registry state of each mode and the
run-time controls are exercised on temporary trees and synthetic registries. No band value is spelled.
"""
import importlib.util
import json
import os
import pathlib
import shutil

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/superpowers/evidence/tactical-candidate-seed-registration/01_collision_proof.py"


@pytest.fixture(scope="module")
def P():
    spec = importlib.util.spec_from_file_location("_pilot_collision_proof", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def S(P):
    return P.load_survey()


@pytest.fixture(scope="module")
def record(P, S):
    return P.read_record((ROOT / P.SURVEY_RECORD).read_text(), S)


# ─────────────────────────────── the script and its pins ───────────────────────────────

def test_the_script_and_this_file_SPELL_no_band_literal(S):
    for path in (SCRIPT, pathlib.Path(__file__)):
        assert S.scan(path.read_text()) == {}, path


def test_the_RECORDS_are_the_paths_A8_2_exempts(P, S):
    assert P.OUT["pre"] == S.OWN_RECORDS[3] and P.OUT["post"] == S.OWN_RECORDS[4]
    assert f"{P.HERE}/01_collision_proof.py" == S.OWN_RECORDS[2]
    assert pathlib.Path(P.__file__).resolve() == SCRIPT


def test_the_PINS_are_the_committed_survey_script_and_record(P, S):
    assert P._sha((ROOT / P.SURVEY_SCRIPT).read_bytes()) == P.SURVEY_SCRIPT_SHA256
    assert P._sha((ROOT / P.SURVEY_RECORD).read_bytes()) == P.SURVEY_RECORD_SHA256
    assert S.scan is not None and S.repo_literals and S.check and S.prior_terms, "the survey's own functions"


@pytest.mark.parametrize("which", ["script", "record"])
def test_an_ALTERED_pin_is_REFUSED(P, tmp_path, which):
    for rel in (P.SURVEY_SCRIPT, P.SURVEY_RECORD):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    target = tmp_path / (P.SURVEY_SCRIPT if which == "script" else P.SURVEY_RECORD)
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(P.ProofRefused, match="is not the pinned file"):
        P.load_survey(tmp_path)


def test_the_BLOCK_is_READ_from_the_record_and_agrees_with_the_frozen_rule(record, S):
    lo, hi = record["block"]
    assert hi - lo == 32 and lo == record["base"] and lo % 10_000 == 0 and S.BAND[0] <= lo < hi <= S.BAND[1]
    assert record["pairs"] == S.pair_seeds(record["block"]) and len(record["pairs"]) == 16
    assert len(record["head"]) == 40 and len(record["registry_sha256"]) == 64


@pytest.mark.parametrize("edit,why", [
    (lambda t: t.replace("THE PILOT BLOCK IS CLEAN", "NOT CLEAN -- STOP (no adjustment)"), "not CLEAN"),
    (lambda t: t.replace("controls passed: 27/27", "controls passed: 26/27"), "failed control"),
    (lambda t: t.replace("; working tree clean", "; working tree DIRTY"), "dirty tree"),
    (lambda t: t.replace("CLEAN=True", "CLEAN=False"), "block check is not CLEAN"),
    (lambda t: t.replace("  pilot-03  red ", "  pilot-03  red 1", 1), "not the frozen rule's"),
    (lambda t: t + "SURVEY VERDICT   : THE PILOT BLOCK IS CLEAN\n", "exactly once"),
], ids=["verdict", "controls", "dirty", "block check", "a pair seed", "verdict twice"])
def test_a_record_that_is_not_a_CLEAN_survey_is_REFUSED(P, S, edit, why):
    with pytest.raises(P.ProofRefused, match=why):
        P.read_record(edit((ROOT / P.SURVEY_RECORD).read_text()), S)


# ─────────────────────────────── the enumeration, fail-closed ───────────────────────────────

def roots(tmp_path, S):
    for base in S.ROOTS:
        (tmp_path / base).mkdir(parents=True)
        (tmp_path / base / "ok.txt").write_text("x")
    return tmp_path


def test_an_UNREADABLE_directory_STOPS_where_the_survey_SKIPPED_it(P, S, tmp_path):
    t = roots(tmp_path, S)
    locked = t / "docs/superpowers/evidence/locked"
    locked.mkdir(parents=True)
    (locked / "hidden.txt").write_text("x")
    locked.chmod(0)
    try:
        files, _ = S.repo_files(t)                  # the historical survey: silently omitted
        assert not any("hidden" in str(p) for p in files)
        with pytest.raises(S.SurveyStop, match="a directory could not be read"):
            P.strict_files(S, t)
    finally:
        locked.chmod(0o700)


@pytest.mark.parametrize("case", ["missing root", "root is a file", "root is a symlink", "fifo"])
def test_the_ENUMERATION_STOPS_on_anything_it_cannot_classify(P, S, tmp_path, case):
    t = roots(tmp_path, S)
    if case == "missing root":
        shutil.rmtree(t / "tests")
    elif case == "root is a file":
        shutil.rmtree(t / "scripts")
        (t / "scripts").write_text("x")
    elif case == "root is a symlink":
        shutil.move(str(t / "tests"), str(t / "real_tests"))
        os.symlink(t / "real_tests", t / "tests")
    else:
        os.mkfifo(t / "scripts" / "pipe")
    with pytest.raises(S.SurveyStop):
        P.strict_files(S, t)


def test_a_SOUND_tree_is_enumerated_like_the_survey_minus_nothing(P, S, tmp_path):
    t = roots(tmp_path, S)
    (t / "docs/superpowers/a/b").mkdir(parents=True)
    (t / "docs/superpowers/a/b/run.log").write_text("x")
    os.symlink(t / "tests/ok.txt", t / "tests/link.txt")
    os.symlink(t / "docs/superpowers/a", t / "scripts/linkdir")
    assert P.strict_files(S, t) == S.repo_files(t)


# ─────────────────────────────── the expected state, by mode ───────────────────────────────

def reg(accounted, exposed="()", other="X = 1"):
    return (f"ACCOUNTED_SEED_INTERVALS = (\n{accounted})\nEXPOSED_SEED_INTERVALS = {exposed}\n"
            f"RETIRED_SEED_INTERVALS = ()\nTEST_ONLY_SEED_INTERVALS = ()\n{other}\n")


@pytest.fixture
def world(P, record):
    lo, hi = record["block"]
    old = f"    ({lo - 9000}, {lo - 8000}),\n"
    pre = reg(old)
    rec = {**record, "registry_sha256": P._sha(pre.encode())}
    plan = {"mode": "pilot", "pairs": [{"pair_id": p["pair_id"], "seeds": {"red": p["red"], "black": p["black"]}}
                                       for p in record["pairs"]]}
    return {"lo": lo, "hi": hi, "old": old, "pre": pre, "rec": rec, "plan": plan,
            "good": reg(old + f"    ({lo}, {hi}),   # the pilot\n")}


def test_PRE_holds_only_for_the_surveys_registry_and_no_plan(P, S, world):
    w = world
    assert P.state_faults("pre", w["rec"], w["pre"], None, None, False, S) == []
    assert P.state_faults("pre", w["rec"], reg(w["old"], other="X = 2"), None, None, False, S)
    assert any("already holds" in f for f in P.state_faults("pre", w["rec"], w["good"], None, None, False, S))
    assert any("plan already exists" in f for f in P.state_faults("pre", w["rec"], w["pre"], None, w["plan"], True, S))


def test_POST_holds_only_for_exactly_the_block_appended_last_and_the_surveys_plan(P, S, world):
    w = world
    lo, hi = w["lo"], w["hi"]
    assert P.state_faults("post", w["rec"], w["good"], w["pre"], w["plan"], True, S) == []
    bad_plan = json.loads(json.dumps(w["plan"]))
    bad_plan["pairs"][5]["seeds"]["red"] += 2
    wrong = {
        "appended first": (reg(f"    ({lo}, {hi}),\n" + w["old"]), w["plan"], True),
        "two appended": (reg(w["old"] + f"    ({lo}, {hi}),\n    ({hi + 50}, {hi + 60}),\n"), w["plan"], True),
        "another tuple": (reg(w["old"] + f"    ({lo}, {hi - 1}),\n"), w["plan"], True),
        "EXPOSED changed": (reg(w["old"] + f"    ({lo}, {hi}),\n", exposed=f"(({lo - 9000}, {lo - 8999}),)"),
                            w["plan"], True),
        "another statement": (reg(w["old"] + f"    ({lo}, {hi}),\n", other="X = 3"), w["plan"], True),
        "no plan": (w["good"], None, False),
        "plan untracked": (w["good"], w["plan"], False),
        "a plan seed": (w["good"], bad_plan, True),
        "a plan of 15": (w["good"], {**w["plan"], "pairs": w["plan"]["pairs"][:15]}, True),
        "a plan for study": (w["good"], {**w["plan"], "mode": "study"}, True),
    }
    for name, (cur, plan, tracked) in wrong.items():
        assert P.state_faults("post", w["rec"], cur, w["pre"], plan, tracked, S), name
    rec2 = {**w["rec"], "registry_sha256": "0" * 64}
    assert any("pre-registration registry" in f
               for f in P.state_faults("post", rec2, w["good"], w["pre"], w["plan"], True, S))


def test_a_registry_assigned_TWICE_is_REFUSED(P, S, world):
    with pytest.raises(P.ProofRefused, match="more than once"):
        P.state_faults("pre", world["rec"], world["pre"] + "EXPOSED_SEED_INTERVALS = ()\n", None, None, False, S)


def test_only_the_blocks_OWN_last_interval_leaves_the_POST_prior_terms(P, world):
    lo, hi = world["lo"], world["hi"]
    regs = {"ACCOUNTED": [(lo - 9000, lo - 8000), (lo, hi)], "EXPOSED": [], "RETIRED": [], "TEST_ONLY": []}
    assert P.prior_registries("post", regs, (lo, hi))["ACCOUNTED"] == [(lo - 9000, lo - 8000)]
    assert P.prior_registries("pre", regs, (lo, hi)) == regs
    first = {**regs, "ACCOUNTED": [(lo, hi), (lo - 9000, lo - 8000)]}
    assert P.prior_registries("post", first, (lo, hi)) == first, "not last: it stays -- and the state check fails"


# ─────────────────────────────── the run-time controls and the entry ───────────────────────────────

def test_the_RUN_controls_all_PASS(P, S, record):
    enum = P._enumeration_controls(S)
    assert len(enum) == 6 and all(ok for _l, ok, _d in enum), enum
    state = P._state_controls(S, record)
    assert len(state) == 14 and all(ok for _l, ok, _d in state), [x for x in state if not x[1]]


@pytest.mark.parametrize("argv,why", [(["x"], "--mode must be"), (["x", "--mode", "both"], "--mode must be")])
def test_the_ENTRY_refuses_a_bad_mode_before_anything(P, argv, why):
    with pytest.raises(P.ProofRefused, match=why):
        P.main(argv)


def test_the_ENTRY_is_CREATE_ONLY_per_mode(P, monkeypatch, tmp_path):
    monkeypatch.setattr(P, "ROOT", tmp_path)
    (tmp_path / P.OUT["pre"]).parent.mkdir(parents=True)
    (tmp_path / P.OUT["pre"]).write_text("already")
    monkeypatch.setattr(P, "load_survey", lambda root=None: pytest.fail("scanned before the create-only check"))
    with pytest.raises(P.ProofRefused, match="runs once"):
        P.main(["x", "--mode", "pre"])
    assert (tmp_path / P.OUT["pre"]).read_text() == "already"


@pytest.mark.parametrize("clean,faults,passed,total,dirty,ok,words", [
    (True, [], 40, 40, False, True, "IS AS EXPECTED"),
    (False, [], 40, 40, False, False, "NOT CLEAN"),
    (True, ["x"], 40, 40, False, False, "STATE"),
    (True, [], 39, 40, False, False, "A CONTROL FAILED"),
    (True, [], 40, 40, True, False, "DIRTY TREE"),
], ids=["proven", "not clean", "state", "a control", "dirty"])
def test_the_VERDICT_needs_every_condition(P, clean, faults, passed, total, dirty, ok, words):
    got, text = P.verdict(clean, faults, passed, total, dirty)
    assert got is ok and words in text
