"""H5 production qualification, EARLY and BINDING (implementation card §3, §4). NOT RUN.

No JVM, no checkpoint load, no research seed, no game. The incumbent identity,
the pinned H4 manifest, the registry `git show` at the pinned commit and the H5
study CLI's refusal run for REAL; the checkpoint loader, the builder, the compile
and the toolchain paths are stubbed at their module attributes. The stub compile
returns H4's BOUND toolchain content -- read from the pinned manifest -- so the
equivalence check is exercised against the real H4 record, not against a fake.
"""
import ast
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import e4_screen_command as SCREEN_CMD
from scripts.GPU.alphazero import e4_screen_reference as REF
from scripts.GPU.alphazero import h4_runner as R4
from scripts.GPU.alphazero import h5_production_qualification as Q
from scripts.GPU.alphazero import h5_production_qualification_authorization as QAUTH
from scripts.GPU.alphazero import h5_runner as H5
from scripts.GPU.alphazero import twixtbot_g3_reference as G3
from tests.test_h4_production_path import StubEvaluator
from tests.test_h4_runner import FAKE_TOOLCHAIN
from tests.test_h5_runner import survey_blocks

ROOT = pathlib.Path(__file__).resolve().parents[1]
PLACEHOLDERS = {k: (10_000 + 1_000 * k, 10_148 + 1_000 * k) for k in range(4)}
H4_ENTRY = json.loads((ROOT / Q.H4_STUDY_MANIFEST).read_text())["segments"][0]
H4_CONTENT = H4_ENTRY["t1j_runtime"]["toolchain"]
LOCAL = {k: FAKE_TOOLCHAIN[k] for k in R4.TOOLCHAIN_LOCAL}


def _identity():
    cfg = R4.frozen_argmax_config()
    return cfg, R4.frozen_incumbent_identity(cfg, design="H5_STUDY")


def real_bound():
    """The bound identity H5 would record with H4's toolchain content."""
    _cfg, ident = _identity()
    return json.loads(json.dumps(H5.bound_identity(ident, H4_CONTENT)))


# ─────────────────────────────── 1. the gate ───────────────────────────────

def test_the_QUALIFICATION_refuses_while_its_gate_is_shut(tmp_path):
    with pytest.raises(Q.QualificationUnauthorized):
        Q.qualify(str(tmp_path / "rec"), str(tmp_path / "cls"))
    assert os.listdir(tmp_path) == []


def test_the_QUALIFICATION_CLI_exits_5_as_a_fresh_subprocess(tmp_path):
    r = subprocess.run([sys.executable, "-m",
                        "scripts.GPU.alphazero.h5_production_qualification",
                        "--out-dir", str(tmp_path / "rec"),
                        "--classes-root", str(tmp_path / "cls")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == Q.EXIT_UNAUTHORIZED == 5, f"exit {r.returncode}, {r.stderr!r}"
    assert os.listdir(tmp_path) == []


def test_the_gate_is_read_at_CALL_TIME_at_both_entries_and_nothing_chooses_the_role():
    src = pathlib.Path(Q.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for e in ("qualify", "main"):
        assert any(isinstance(n, ast.Attribute)
                   and n.attr == "H5_PRODUCTION_QUALIFICATION_AUTHORIZED"
                   and getattr(n.value, "id", None) == "QAUTH" for n in ast.walk(fns[e])), e
    assert {a.arg for f in ("qualify", "_qualify", "main") for a in
            fns[f].args.args + fns[f].args.kwonlyargs} <= {"out_dir", "classes_root", "argv"}, \
        "no parameter may choose the role"
    assert not {"environ", "getenv"} & {n.attr for n in ast.walk(tree)
                                        if isinstance(n, ast.Attribute)}


# ─────────────────────────────── 2. the role, derived ───────────────────────────────

@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(H5, "SEED_BLOCKS", dict(PLACEHOLDERS))
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        tuple(REF.ACCOUNTED_SEED_INTERVALS) + tuple(PLACEHOLDERS.values()))


def test_EMPTY_blocks_derive_EARLY(monkeypatch):
    monkeypatch.setattr(H5, "SEED_BLOCKS", {})
    assert Q.derive_role()[0] == Q.EARLY


def test_the_REAL_closed_state_derives_STOP():
    """After the closing edit the four survey blocks are EXPOSED and RETIRED: the REAL
    state derives STOP, so no qualification can bind spent seeds again. Fixed
    messages -- a failure must not print a block."""
    role, seen, reason = Q.derive_role()
    assert role == Q.STOP, "the closed real state does not derive STOP"
    assert reason.startswith("seeds fail the shared registry") and "exposed, retired" in reason, \
        "STOP, but not on the shared registry's exposed-and-retired refusal"
    assert seen["seed_blocks"] == {str(k): list(v) for k, v in survey_blocks().items()}, \
        "the observation is not the step-5 survey's blocks"


def test_FOUR_registered_blocks_derive_BINDING(registered):
    role, seen, reason = Q.derive_role()
    assert (role, reason) == (Q.BINDING, None)
    assert seen["seed_blocks"] == {str(k): list(v) for k, v in PLACEHOLDERS.items()}


@pytest.mark.parametrize("blocks,match", [
    ({k: PLACEHOLDERS[k] for k in range(3)}, "not \\[0, 1, 2, 3\\]"),
    ({**PLACEHOLDERS, 3: (13_000, 13_146)}, "not 148 each"),
    ({**PLACEHOLDERS, 1: (10_100, 10_248)}, "overlap"),
    ({**PLACEHOLDERS, 2: (50_000, 50_148)}, "not accounted"),
    ({**PLACEHOLDERS, 0: (202_634_000, 202_634_148)}, "exposed, retired")],
    ids=["segments", "size", "overlap", "unaccounted", "h4_block"])
def test_ANY_OTHER_block_state_derives_STOP(registered, monkeypatch, blocks, match):
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        tuple(REF.ACCOUNTED_SEED_INTERVALS) + ((10_100, 10_248),))
    monkeypatch.setattr(H5, "SEED_BLOCKS", blocks)
    role, _seen, reason = Q.derive_role()
    assert role == Q.STOP
    import re
    assert re.search(match, reason), reason


# ───────────────────── 3. h4_equivalence against the PINNED record (§4.2) ─────────────────────

def test_the_REAL_code_state_is_EQUIVALENT_to_H4s_bound_record():
    """REAL STATE: the pins recompute, 36 H4 files hash as bound, the registry
    differs only in its interval registries, the cards are H4's, and the identity
    differs only in `design`."""
    obs = Q.h4_equivalence(real_bound())
    assert obs["h4_code_files_equal"] == 36 and obs["h4_cards_equal"] == 5
    assert obs["identity_differs_only_in"] == ["design"]


def _stops(fn, match):
    with pytest.raises(Q._Stop, match=match):
        fn()


def _manifest_with(monkeypatch, edit):
    m = json.loads((ROOT / Q.H4_STUDY_MANIFEST).read_text())
    for e in m["segments"]:
        edit(e)
    monkeypatch.setattr(Q, "pinned_h4_manifest", lambda: m)


@pytest.mark.parametrize("edit,match", [
    (lambda e: e["code"].update({R4.CODE[0]: "0" * 64}), "H4 play-path files changed"),
    (lambda e: e["cards"].update({R4.CARDS[0]: "0" * 64}), "H4 cards changed")])
def test_a_MOVED_H4_file_or_card_STOPS(monkeypatch, edit, match):
    _manifest_with(monkeypatch, edit)
    _stops(lambda: Q.h4_equivalence(real_bound()), match)


def test_H4_entries_that_DISAGREE_with_each_other_STOP(monkeypatch):
    m = json.loads((ROOT / Q.H4_STUDY_MANIFEST).read_text())
    m["segments"][3]["t1j_runtime"]["ply_cap"] = 274
    monkeypatch.setattr(Q, "pinned_h4_manifest", lambda: m)
    _stops(lambda: Q.h4_equivalence(real_bound()), "do not share")


def test_a_REGISTRY_changed_OUTSIDE_its_interval_registries_STOPS(monkeypatch):
    real = Q.h4_registry_source()
    monkeypatch.setattr(Q, "h4_registry_source", lambda: real + "\nCONSUMED_SEEDS = ()\n")
    _stops(lambda: Q.h4_equivalence(real_bound()), "changed outside its four interval")


def test_a_registry_changed_ONLY_in_an_interval_registry_is_equal():
    """CLEAN BASELINE for the registry check: only the four intervals are ignored."""
    real = Q.h4_registry_source()
    assert Q._registry_body(real + "\nRETIRED_SEED_INTERVALS = ((1, 2),)\n") \
        == Q._registry_body(real), "an interval registry edit must be ignored"
    assert Q._registry_body(real + "\nCONSUMED_SEEDS = ()\n") != Q._registry_body(real), \
        "a CONSUMED_SEEDS edit must NOT be ignored"


def test_a_DIFFERENT_T1J_RUNTIME_STOPS():
    fake = json.loads(json.dumps(real_bound()))
    fake["t1j_runtime"]["toolchain"] = {k: FAKE_TOOLCHAIN[k] for k in R4.TOOLCHAIN_CONTENT}
    _stops(lambda: Q.h4_equivalence(fake), "runtime differs from H4's in \\['toolchain'\\]")


@pytest.mark.parametrize("edit", [
    lambda i: i["eval_config"].update(mcts_sims=401),
    lambda i: i.update(design="H4_STUDY"),
    lambda i: i.update(reference_sha1="0" * 40)])
def test_an_IDENTITY_differing_in_more_than_design_STOPS(edit):
    b = real_bound()
    edit(b["incumbent_identity"])
    _stops(lambda: Q.h4_equivalence(b), "identity differs from H4's")


@pytest.mark.parametrize("pin", ["H4_STUDY_MANIFEST_SHA256", "H4_REGISTRY_BLOB_SHA256"])
def test_a_PIN_that_does_not_recompute_STOPS_before_its_content_is_used(monkeypatch, pin):
    monkeypatch.setattr(Q, pin, "0" * 64)
    _stops(lambda: Q.h4_equivalence(real_bound()), "does not hash to its pin")


def test_an_UNTRACKED_pinned_manifest_STOPS(monkeypatch, tmp_path):
    """Durability: the pinned file must be committed in THIS repository."""
    d = tmp_path / "repo"
    d.mkdir()
    subprocess.run(["git", "-C", str(d), "init", "-q"], check=True)
    monkeypatch.setattr(R4, "REPO_ROOT", d.resolve())
    target = d / Q.H4_STUDY_MANIFEST
    target.parent.mkdir(parents=True)
    shutil.copy2(ROOT / Q.H4_STUDY_MANIFEST, target)
    _stops(Q.pinned_h4_manifest, "not durable: not tracked")


# ─────────────────────────── 4. a whole qualification, stubbed ───────────────────────────

@pytest.fixture
def qual_open(monkeypatch, tmp_path):
    """Gate open, the evidence root a temporary directory, every external act
    stubbed except the H5 CLI's refusal and the H4 record reads. EARLY by default:
    step 6 filled the real `SEED_BLOCKS`, so the empty state is set here; a test
    wanting BINDING requests `registered` AFTER this fixture."""
    monkeypatch.setattr(QAUTH, "H5_PRODUCTION_QUALIFICATION_AUTHORIZED", True)
    monkeypatch.setattr(H5, "SEED_BLOCKS", {})
    monkeypatch.setattr(Q, "EVIDENCE_ROOT", tmp_path / "evidence")
    (tmp_path / "evidence").mkdir()
    _cfg, ident = _identity()
    box = {"sha1": ident["reference_sha1"], "move": (0, 4), "classes": [None, None], "n": 0}
    monkeypatch.setattr(SCREEN_CMD, "_default_load_evaluator",
                        lambda root: StubEvaluator(ident["reference"], box["sha1"]))
    monkeypatch.setattr(G3, "build_reference_agent", lambda **kw: (lambda s: box["move"]))

    def comp(deadline, *, paths):
        box["n"] += 1
        classes = box["classes"][box["n"] - 1] or H4_CONTENT["classes"]
        return {**H4_CONTENT, **LOCAL, "classes_dir": paths.classes, "classes": classes}
    monkeypatch.setattr(Q.D1, "_default_compile", comp)
    monkeypatch.setattr(R4, "_verified_t1j_paths", lambda classes: R4.T1jPaths(
        java="/j", jar="/x", classes=classes, ply_cap=280))
    return box


def stage_log(rec_dir):
    return [json.loads(l) for l in (rec_dir / "stages.jsonl").read_text().splitlines()]


def early(tmp_path):
    return Q.qualify(str(tmp_path / "evidence" / "early"), str(tmp_path / "cls"))


def test_an_EARLY_qualification_runs_CLEAN_and_says_it_binds_NOTHING(tmp_path, qual_open):
    rec = early(tmp_path)
    assert (rec["role"], rec["result"], rec["failing_stage"]) == ("EARLY", "CLEAN", None), \
        rec["reason"]
    assert "binds NOTHING" in rec["evidence_note"] and "NOT evidence of strength" in \
        rec["evidence_note"]
    checks = rec["checks"]
    assert list(checks) == list(Q.STAGES)
    assert checks["role"]["role"] == "EARLY"
    assert checks["study_cli"] == {"exit": 5}, "the REAL H5 CLI, as a fresh subprocess"
    assert checks["h4_equivalence"]["t1j_runtime_equal"] is True
    assert checks["bound_identity"] == real_bound()
    assert [e["stage"] for e in stage_log(tmp_path / "evidence" / "early")] == list(Q.STAGES)
    assert json.loads((tmp_path / "evidence" / "early" / "record.json").read_text()) == \
        json.loads(json.dumps(rec))


def test_a_BINDING_qualification_writes_ONLY_to_the_binding_dir(tmp_path, qual_open,
                                                                registered, monkeypatch):
    monkeypatch.setattr(Q, "BINDING_DIR", str(tmp_path / "binding"))
    with pytest.raises(Q.QualificationRefused, match="BINDING qualification writes ONLY"):
        early(tmp_path)
    assert not (tmp_path / "evidence" / "early").exists() and qual_open["n"] == 0
    rec = Q.qualify(str(tmp_path / "binding"), str(tmp_path / "cls"))
    assert (rec["role"], rec["result"]) == ("BINDING", "CLEAN"), rec["reason"]
    assert "binds NOTHING" not in rec["evidence_note"]
    with pytest.raises(Q.QualificationRefused, match="occupied"):
        Q.qualify(str(tmp_path / "binding"), str(tmp_path / "cls2"))


def test_an_EARLY_qualification_may_NEVER_write_to_the_binding_dir(tmp_path, qual_open,
                                                                   monkeypatch):
    monkeypatch.setattr(Q, "BINDING_DIR", str(tmp_path / "evidence" / "binding"))
    with pytest.raises(Q.QualificationRefused, match="may never write to the binding"):
        Q.qualify(str(tmp_path / "evidence" / "binding"), str(tmp_path / "cls"))
    assert not (tmp_path / "evidence" / "binding").exists() and qual_open["n"] == 0


def test_an_EARLY_record_goes_UNDER_the_evidence_tree(tmp_path, qual_open):
    with pytest.raises(Q.QualificationRefused, match="under the evidence tree"):
        Q.qualify(str(tmp_path / "loose"), str(tmp_path / "cls"))
    assert not (tmp_path / "loose").exists()


def test_a_STOP_role_is_a_RECORDED_STOP_at_the_role_stage(tmp_path, qual_open, monkeypatch):
    monkeypatch.setattr(H5, "SEED_BLOCKS", {0: PLACEHOLDERS[0]})
    rec = early(tmp_path)
    assert (rec["role"], rec["result"], rec["failing_stage"]) == ("STOP", "STOP", "role")
    assert rec["checks"] == {} and qual_open["n"] == 0


@pytest.mark.parametrize("fault,stage,match", [
    ({"classes": [None, {"E3bDump.class": "e" * 64}]}, "reproducibility",
     "does not compile reproducibly"),
    ({"move": (0, 0)}, "incumbent_move", "not a legal empty-board move"),
    ({"sha1": "0" * 40}, "checkpoint", "checkpoint")])
def test_a_failed_check_is_a_recorded_STOP_that_KEEPS_every_earlier_stage(
        tmp_path, qual_open, fault, stage, match):
    qual_open.update(fault)
    rec = early(tmp_path)
    assert (rec["result"], rec["failing_stage"]) == ("STOP", stage)
    assert match in rec["reason"]
    before = list(Q.STAGES[:Q.STAGES.index(stage)])
    assert list(rec["checks"]) == before
    assert [e["stage"] for e in stage_log(tmp_path / "evidence" / "early")] == before


def test_a_NON_H4_TOOLCHAIN_is_a_recorded_STOP_at_h4_equivalence(tmp_path, qual_open,
                                                                 monkeypatch):
    fake = {"E3bDump.class": "f" * 64}
    qual_open["classes"] = [fake, fake]
    rec = early(tmp_path)
    assert (rec["result"], rec["failing_stage"]) == ("STOP", "h4_equivalence")
    assert "runtime differs from H4's in ['toolchain']" in rec["reason"]
    assert "study_cli" in rec["checks"] and "bound_identity" not in rec["checks"]


def test_a_STUDY_CLI_that_does_not_refuse_is_a_STOP(tmp_path, qual_open, monkeypatch):
    monkeypatch.setattr(Q, "_study_cli_refuses", lambda tmp: 0)
    rec = early(tmp_path)
    assert (rec["result"], rec["failing_stage"]) == ("STOP", "study_cli")


@pytest.mark.parametrize("where", ["occupied_record", "classes_in_repo", "occupied_classes"])
def test_the_qualification_REFUSES_a_bad_destination_and_WRITES_NOTHING(tmp_path, qual_open,
                                                                       where):
    rec, cls = tmp_path / "evidence" / "early", tmp_path / "cls"
    if where == "occupied_record":
        rec.mkdir()
    elif where == "occupied_classes":
        cls.mkdir()
    else:
        cls = ROOT / "never-created-h5-classes"
    try:
        with pytest.raises(Q.QualificationRefused):
            Q.qualify(str(rec), str(cls))
        assert qual_open["n"] == 0 and not (ROOT / "never-created-h5-classes").exists()
    finally:
        shutil.rmtree(ROOT / "never-created-h5-classes", ignore_errors=True)
    if where == "occupied_record":
        assert os.listdir(rec) == []
    else:
        assert not rec.exists(), "the record directory is claimed only after every check"


@pytest.mark.parametrize("what", ["classes_root", "stage_log"])
def test_a_SETUP_failure_after_the_claim_is_a_RECORDED_STOP_not_a_bare_directory(
        tmp_path, qual_open, monkeypatch, what):
    """Review 2026-09-27: after the claim, NOTHING may leave an occupied record
    directory without `record.json` -- not the class root, not the stage log."""
    cls = tmp_path / "cls"
    if what == "classes_root":
        real = os.makedirs

        def refuse(path, *a, **k):
            if os.fspath(path) == str(cls):
                raise OSError("class root refused")
            return real(path, *a, **k)
        monkeypatch.setattr(os, "makedirs", refuse)
    else:
        def broken(path):
            raise OSError("stage log refused")
        monkeypatch.setattr(Q.Q4, "_StageLog", broken)
    rec = early(tmp_path)
    assert (rec["result"], rec["failing_stage"]) == ("STOP", "setup")
    assert rec["reason"].startswith("OSError:") and rec["checks"] == {}
    on_disk = json.loads((tmp_path / "evidence" / "early" / "record.json").read_text())
    assert (on_disk["result"], on_disk["failing_stage"]) == ("STOP", "setup")
    assert qual_open["n"] == 0 and not cls.exists()
