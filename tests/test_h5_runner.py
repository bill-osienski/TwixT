"""H5 runner (implementation card §0-§2, §8). SYNTHETIC FIXTURES ONLY.

No JVM, no research seed, no research game. Fixture games run through the REAL
harness, the REAL §4B adapter and H4's REAL persistence -- H5 calls H4's objects
-- stubbed only at the process boundary (`subprocess.run`) with a STUB incumbent,
exactly as `tests/test_h4_runner.py` does. Study-mode tests use PLACEHOLDER
positive blocks outside every registered interval, patched into `SEED_BLOCKS`
and the ACCOUNTED registry for the test only, a temporary git repository for
durability, and stop BEFORE any incumbent is built.
"""
import ast
import builtins
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import e4_screen_command as SCREEN_CMD
from scripts.GPU.alphazero import e4_screen_reference as REF
from scripts.GPU.alphazero import h4_confirmatory_analysis as C4
from scripts.GPU.alphazero import h4_pilot_authorization as AUTH4
from scripts.GPU.alphazero import h4_production_qualification as Q4
from scripts.GPU.alphazero import h4_production_qualification_authorization as QAUTH4
from scripts.GPU.alphazero import h4_runner as R4
from scripts.GPU.alphazero import h5_aggregation_authorization as CAUTH5
from scripts.GPU.alphazero import h5_confirmatory_analysis as C5
from scripts.GPU.alphazero import h5_production_qualification as Q5
from scripts.GPU.alphazero import h5_production_qualification_authorization as QAUTH5
from scripts.GPU.alphazero import h5_runner as H5
from scripts.GPU.alphazero import h5_study_authorization as AUTH
from scripts.GPU.alphazero import twixtbot_g3_reference as G3
from tests.test_h4_pilot_feasibility import commit, edit_records, new_repo
from tests.test_h5_confirmatory_analysis import compose, templates  # noqa: F401
from tests.test_h4_production_path import StubEvaluator, gate_module_faults, play_path_closure
from tests.test_h4_runner import FAKE_TOOLCHAIN, SEQ, stub_incumbent, wire  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROOTS = ("scripts/GPU/alphazero/h5_runner.py",
         "scripts/GPU/alphazero/h5_production_qualification.py")
GATES = {"scripts/GPU/alphazero/h5_study_authorization.py": "H5_STUDY_EXECUTION_AUTHORIZED",
         "scripts/GPU/alphazero/h5_production_qualification_authorization.py":
             "H5_PRODUCTION_QUALIFICATION_AUTHORIZED",
         "scripts/GPU/alphazero/h5_aggregation_authorization.py":
             "H5_STUDY_AGGREGATION_AUTHORIZED"}


@pytest.fixture
def gate_open(monkeypatch):
    """Tests must run in the REAL state: the public entry, gate OPEN."""
    monkeypatch.setattr(AUTH, "H5_STUDY_EXECUTION_AUTHORIZED", True)


def _paths(tmp_path):
    return H5.T1jPaths(java="/j", jar="/x.jar", classes=str(tmp_path / "classes"),
                       ply_cap=280)


def run(tmp_path, pairs=(("p0", -1, -2),), name="run", **kw):
    sched = H5.make_schedule(pairs, mode="fixture")
    kw.setdefault("incumbent_build", stub_incumbent)
    kw.setdefault("incumbent_identity", {"stub": True})
    kw.setdefault("_compile", lambda d: dict(FAKE_TOOLCHAIN))
    return H5.run_study(mode="fixture", schedule=sched, out_dir=str(tmp_path / name),
                        paths=_paths(tmp_path), deadline_s=600, **kw)


def results_of(tmp_path, name="run"):
    return str(tmp_path / name / "results.jsonl")


# ─────────────────────────────── 1. the gates (§1) ───────────────────────────────

@pytest.mark.parametrize("path,gate", list(GATES.items()))
def test_each_H5_GATE_MODULE_is_only_its_declaration_and_is_CLOSED(path, gate):
    src = (ROOT / path).read_text(encoding="utf-8")
    assert gate_module_faults(src, gate) == []
    assert "= False\n" in src


def test_the_H5_gates_are_published_CLOSED():
    assert AUTH.H5_STUDY_EXECUTION_AUTHORIZED is False
    assert QAUTH5.H5_PRODUCTION_QUALIFICATION_AUTHORIZED is False
    assert CAUTH5.H5_STUDY_AGGREGATION_AUTHORIZED is False


def test_the_public_entry_refuses_while_the_gate_is_shut(tmp_path, wire):
    with pytest.raises(H5.H4RunError, match="H5 study is UNAUTHORIZED"):
        run(tmp_path)
    assert wire["calls"] == [] and not (tmp_path / "run").exists()


def test_the_CLI_refuses_in_a_fresh_subprocess(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.h5_runner", "--manifest", "m.json",
         "--segment", "0", "--out-dir", str(tmp_path / "o"),
         "--classes", str(tmp_path / "c")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == H5.EXIT_UNAUTHORIZED == 5, r.stderr
    assert not (tmp_path / "o").exists() and not (tmp_path / "c").exists()


def gate_reads(source):
    """Per public entry: does it read `AUTH.H5_STUDY_EXECUTION_AUTHORIZED` AS AN
    ATTRIBUTE? Plus any other `_AUTHORIZED` name the module binds."""
    tree = ast.parse(source)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    reads = {e: any(isinstance(n, ast.Attribute) and n.attr == "H5_STUDY_EXECUTION_AUTHORIZED"
                    and isinstance(n.value, ast.Name) and n.value.id == "AUTH"
                    for n in ast.walk(fns[e])) for e in ("run_study", "main")}
    bound = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
             and n.id.endswith("_AUTHORIZED")}
    bound |= {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
              for a in n.names if a.name.endswith("_AUTHORIZED")}
    return reads, bound


def test_the_gate_is_read_at_both_entries_at_CALL_TIME_and_has_no_override():
    src = pathlib.Path(H5.__file__).read_text(encoding="utf-8")
    assert gate_reads(src) == ({"run_study": True, "main": True}, set())
    attrs = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    assert not {"environ", "getenv"} & attrs


@pytest.mark.parametrize("plant", [
    ("    if not AUTH.H5_STUDY_EXECUTION_AUTHORIZED:\n        raise H4RunError(\"the H5 study",
     "    if False:\n        raise H4RunError(\"the H5 study"),
    ("from . import h5_study_authorization as AUTH\n",
     "from . import h5_study_authorization as AUTH\n"
     "from .h5_study_authorization import H5_STUDY_EXECUTION_AUTHORIZED\n")])
def test_the_gate_walker_is_NOT_VACUOUS(plant):
    src = pathlib.Path(H5.__file__).read_text(encoding="utf-8")
    assert src.count(plant[0]) == 1
    assert gate_reads(src.replace(plant[0], plant[1])) != (
        {"run_study": True, "main": True}, set())


# ──────────────── 2. the COPY: H4's text, H5's namespace (§0, §2.3) ────────────────

#: Card §2.3: the ONLY free names of each copied function that resolve to H5's own
#: objects. Every other free name must resolve to H4's VERY object.
SUBSTITUTED = {"header_candidate": {"DESIGNS", "bound_identity"},
               "_run_unguarded": {"DESIGNS", "header_candidate"}}


def _fn(source, name):
    return next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def free_names(fn):
    local = {a.arg for a in ast.walk(fn) if isinstance(a, ast.arg)}
    local |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)
              and isinstance(n.ctx, ast.Store)}
    local |= {h.name for h in ast.walk(fn) if isinstance(h, ast.ExceptHandler) and h.name}
    loads = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)
             and isinstance(n.ctx, ast.Load)}
    return sorted(loads - local - set(dir(builtins)))


def copy_faults(h5_source, h5_module):
    """Why the H5 copy is NOT H4's text in H5's namespace, per card §2.3."""
    h4_source = pathlib.Path(R4.__file__).read_text(encoding="utf-8")
    bad = []
    for name, subs in SUBSTITUTED.items():
        ours, theirs = _fn(h5_source, name), _fn(h4_source, name)
        if ast.dump(ours) != ast.dump(theirs):
            bad.append(f"{name}: the AST differs from H4's")
        for free in free_names(ours):
            same = getattr(h5_module, free, None) is getattr(R4, free, object())
            if free in subs and same:
                bad.append(f"{name}: {free} is H4's, not H5's substitute")
            if free not in subs and not same:
                bad.append(f"{name}: {free} is not H4's very object")
    return bad


def test_the_COPIED_functions_are_H4s_TEXT_in_H5s_NAMESPACE():
    src = pathlib.Path(H5.__file__).read_text(encoding="utf-8")
    assert copy_faults(src, H5) == []
    assert H5.DESIGNS["study"] == "H5_STUDY" and R4.DESIGNS["study"] == "H4_STUDY"


@pytest.mark.parametrize("old,new", [
    ("    claim_directory(out_dir)                      # THE claim",
     "    os.makedirs(out_dir, exist_ok=True)           # THE claim"),
    ('"complete": True, "setup_s": setup_s,', '"complete": False, "setup_s": setup_s,')],
    ids=["claim", "complete"])
def test_the_copy_check_sees_an_EDITED_copy(old, new):
    """CLEAN-BASELINE CONTROL: an edit to the copied text is seen."""
    src = pathlib.Path(H5.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1
    assert copy_faults(src.replace(old, new), H5) != []


def test_the_copy_check_sees_a_REBOUND_name(monkeypatch):
    """CLEAN-BASELINE CONTROL: a copied loop calling something other than H4's
    `play_game`, or H4's own `header_candidate`, is seen."""
    src = pathlib.Path(H5.__file__).read_text(encoding="utf-8")
    monkeypatch.setattr(H5, "play_game", lambda **kw: None)
    assert "_run_unguarded: play_game is not H4's very object" in copy_faults(src, H5)
    monkeypatch.undo()
    monkeypatch.setattr(H5, "header_candidate", R4.header_candidate)
    assert "_run_unguarded: header_candidate is H4's, not H5's substitute" in \
        copy_faults(src, H5)


def test_H5_edits_NO_H4_file():
    """Card §0: every H4 play-path file except the registry hashes to H4's bound
    study manifest (the registry is checked by the qualification, §4.2)."""
    m = json.loads((ROOT / Q5.H4_STUDY_MANIFEST).read_text())
    code = m["segments"][0]["code"]
    moved = [p for p, h in code.items()
             if p != Q5.REGISTRY and R4._sha256_or_none(ROOT / p) != h]
    assert moved == []


# ───────────────────── 3. `CODE` is the whole play path (§2.4) ─────────────────────

def test_CODE_is_EXACTLY_the_play_path_closure_minus_the_H5_gate_modules():
    files, dynamic = play_path_closure(ROOT, roots=ROOTS)
    assert dynamic == []
    assert sorted(set(files) - set(H5.CODE)) == sorted(H5.GATE_MODULES)
    assert sorted(set(files) - set(H5.GATE_MODULES)) == sorted(H5.CODE)
    assert len(H5.CODE) == len(set(H5.CODE))


def test_CODE_holds_H4s_whole_path_AND_H4s_gate_modules_but_no_H5_gate():
    assert set(R4.CODE) <= set(H5.CODE)
    assert set(R4.GATE_MODULES) <= set(H5.CODE), \
        "opening an H4 gate during H5 must change H5's bound code"
    assert not set(GATES) & set(H5.CODE)
    for outside in ("h5_manifest_writer.py", "h5_confirmatory_analysis.py"):
        assert f"scripts/GPU/alphazero/{outside}" not in H5.CODE


def test_a_NEW_dependency_cannot_enter_unlisted(tmp_path):
    """CLEAN-BASELINE CONTROL: a function-level import of a new module planted in
    one listed file grows the closure."""
    for rel in list(H5.CODE) + list(GATES):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, tmp_path / rel)
    (tmp_path / "scripts/GPU/alphazero/brand_new_dependency.py").write_text("X = 1\n")
    victim = tmp_path / "scripts/GPU/alphazero/h5_runner.py"
    victim.write_text(victim.read_text() + "\n\ndef _later():\n    from . import "
                      "brand_new_dependency\n")
    files, _dynamic = play_path_closure(tmp_path, roots=ROOTS)
    assert "scripts/GPU/alphazero/brand_new_dependency.py" in files


# ───────────────────── 4. constants, spelled out from the cards (§2.1) ─────────────────────

def test_the_constants_are_the_cards():
    assert H5.DESIGNS == {"fixture": "H5_SYNTHETIC_FIXTURE", "study": "H5_STUDY"}, \
        "no pilot mode"
    assert list(H5.CARDS) == [
        "docs/superpowers/2026-09-25-t1j-h5-replication-card.md",
        "docs/superpowers/2026-09-27-t1j-h5-implementation-card.md",
        "docs/superpowers/2026-09-22-t1j-h4-runner-persistence-card.md",
        "docs/superpowers/2026-09-22-t1j-h4-4b-acceptance-qualification-card.md",
        "docs/superpowers/2026-09-21-t1j-h4-replacement-card.md",
        "docs/superpowers/2026-09-23-t1j-h4-analysis-card.md",
        "docs/superpowers/2026-09-24-t1j-h4-step4-seed-card.md"]
    assert H5.EVIDENCE_DIRS == {k: f"docs/superpowers/evidence/t1j-h5-study-segment{k}"
                                for k in range(4)}
    assert H5.PAIRS_PER_SEGMENT == 74 and H5.PAIR_TAG == "h5s"


def test_SEED_BLOCKS_is_EMPTY_until_step_6():
    assert H5.SEED_BLOCKS == {}


# ──────────────── 5. a fixture pair end to end, through H4's persistence ────────────────

def test_a_FIXTURE_PAIR_runs_end_to_end_and_every_record_re_derives(tmp_path, wire,
                                                                    gate_open):
    out = run(tmp_path)
    assert out["complete"] and out["games_completed"] == 2
    recs = H5.R4.read_records(results_of(tmp_path))
    head = recs[0]
    assert (head["design"], head["evidence"]) == ("H5_SYNTHETIC_FIXTURE", False)
    assert head["t1j_runtime"]["h4_acceptance"] is True
    assert recs[-1]["record_type"] == "segment_end"
    games = R4.load_games(results_of(tmp_path), allow_fixture=True)
    assert [g["task_id"] for g in games] == ["p0-A", "p0-B"]
    assert all(g["result"]["plies"] == len(SEQ) for g in games)


def test_the_HEADER_hashes_the_SEVEN_cards_and_EVERY_CODE_file(tmp_path, wire, gate_open):
    import hashlib
    run(tmp_path)
    head = R4.read_records(results_of(tmp_path))[0]
    for field, paths in (("cards", H5.CARDS), ("code", H5.CODE)):
        assert sorted(head[field]) == sorted(paths), field
        for p in paths:
            assert head[field][p] == hashlib.sha256((ROOT / p).read_bytes()).hexdigest(), p
    assert len(head["cards"]) == 7


def test_the_T1J_RUNTIME_is_built_by_H4s_own_function(tmp_path, wire, gate_open):
    run(tmp_path)
    head = R4.read_records(results_of(tmp_path))[0]
    content = {k: FAKE_TOOLCHAIN[k] for k in R4.TOOLCHAIN_CONTENT}
    assert head["t1j_runtime"] == R4.bound_identity({"stub": True}, content)["t1j_runtime"]


def test_H4s_READER_does_NOT_recognize_the_H5_fixture_label(tmp_path, wire, gate_open):
    """WHY the H5 analysis binds the stage and header itself: `load_games` refuses
    only H4's fixture label, so an H5 fixture file passes it. The H5 analysis's
    refusal is proven in tests/test_h5_confirmatory_analysis.py."""
    run(tmp_path)
    assert len(R4.load_games(results_of(tmp_path))) == 2


def test_a_TIMEOUT_voids_through_the_COPIED_loop(tmp_path, wire, gate_open):
    wire["timeout_at"] = 3
    with pytest.raises(H5.H4RunVoid) as ei:
        run(tmp_path)
    assert ei.value.classification == "timeout"
    last = R4.read_records(results_of(tmp_path))[-1]
    assert (last["record_type"], last["classification"]) == ("run_void", "timeout")


def test_the_TRACE_and_CONSOLE_never_carry_an_outcome(tmp_path, wire, gate_open, capsys):
    run(tmp_path)
    cap = capsys.readouterr()
    trace = (tmp_path / "run" / "trace.jsonl").read_text()
    assert R4.blinding_violations(trace + cap.out + cap.err) == []
    assert "game_done" in trace


def test_MULTIPLICITY_identical_games_are_all_persisted(tmp_path, wire, gate_open):
    run(tmp_path, pairs=(("p0", -1, -2), ("p1", -3, -4)))
    games = R4.load_games(results_of(tmp_path), allow_fixture=True)
    arm_a = [g["result"]["transcript_digest"] for g in games if g["result"]["arm"] == "A"]
    assert len(games) == 4 and arm_a[0] == arm_a[1]


def test_FIXTURE_mode_refuses_a_destination_under_the_EVIDENCE_tree(tmp_path, gate_open):
    sched = H5.make_schedule([("p", -1, -2)], mode="fixture")
    with pytest.raises(H5.H4RunError, match="evidence tree"):
        H5.run_study(mode="fixture", schedule=sched, out_dir=str(R4.EVIDENCE_ROOT / "x"),
                     paths=_paths(tmp_path), deadline_s=60, incumbent_build=stub_incumbent,
                     incumbent_identity={"stub": True}, _compile=lambda d: {})


def test_an_OCCUPIED_directory_is_refused_before_any_subprocess(tmp_path, wire, gate_open):
    (tmp_path / "run").mkdir()
    with pytest.raises(H5.H4RunError, match="occupied"):
        run(tmp_path)
    assert wire["calls"] == []


# ───────────────────────────── 6. STUDY mode (§2.3) ─────────────────────────────

PLACEHOLDERS = {k: (10_000 + 1_000 * k, 10_148 + 1_000 * k) for k in range(4)}


def _identity():
    cfg = R4.frozen_argmax_config()
    return cfg, R4.frozen_incumbent_identity(cfg, design="H5_STUDY")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A temporary git repository standing in for THIS one, for durability only;
    code and cards are still hashed from the real files."""
    r = new_repo(tmp_path / "repo")
    monkeypatch.setattr(R4, "REPO_ROOT", r.resolve())
    return r


@pytest.fixture
def placeholder(monkeypatch, repo):
    monkeypatch.setattr(H5, "SEED_BLOCKS", dict(PLACEHOLDERS))
    monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS",
                        tuple(REF.ACCOUNTED_SEED_INTERVALS) + tuple(PLACEHOLDERS.values()))
    monkeypatch.setattr(H5, "EVIDENCE_DIRS", {k: str(repo / f"seg{k}") for k in range(4)})


@pytest.fixture
def study_open(monkeypatch, placeholder):
    monkeypatch.setattr(AUTH, "H5_STUDY_EXECUTION_AUTHORIZED", True)


@pytest.fixture
def spies(monkeypatch):
    """Every load, build, compile, game and NON-GIT process: tests prove NONE ran.
    Git is let through -- it is how durability is checked."""
    calls = {"load": 0, "build": 0, "compile": 0, "game": 0, "subprocess": 0}
    _cfg, ident = _identity()
    real_run = subprocess.run

    def load(root):
        calls["load"] += 1
        return StubEvaluator(ident["reference"], ident["reference_sha1"])

    def build(**kw):
        calls["build"] += 1
        raise AssertionError("an incumbent was built")

    def comp(deadline, *, paths):
        calls["compile"] += 1
        return dict(calls.get("toolchain") or FAKE_TOOLCHAIN)

    def game(**kw):
        calls["game"] += 1
        raise RuntimeError("SENTINEL: reached the first game")

    def proc(args, *a, **k):
        if list(args)[:1] == ["git"]:
            return real_run(args, *a, **k)
        calls["subprocess"] += 1
        raise AssertionError("a process was started")

    def verified(classes):
        calls["toolchain_resolved"] = calls.get("toolchain_resolved", 0) + 1
        return H5.T1jPaths(java="/j", jar="/x.jar", classes=classes, ply_cap=280)
    monkeypatch.setattr(SCREEN_CMD, "_default_load_evaluator", load)
    monkeypatch.setattr(G3, "build_reference_agent", build)
    monkeypatch.setattr(H5, "_compile_helper_verified", comp)
    monkeypatch.setattr(H5, "play_game", game)
    monkeypatch.setattr(subprocess, "run", proc)
    monkeypatch.setattr(H5, "_verified_t1j_paths", verified)
    return calls


def manifest_entries(content=None):
    _cfg, ident = _identity()
    content = content or {k: FAKE_TOOLCHAIN[k] for k in R4.TOOLCHAIN_CONTENT}
    out = []
    for k in range(4):
        sched = H5.canonical_schedule(k, ident)
        head = H5.header_candidate(mode="study", segment=k, schedule=sched,
                                   incumbent_identity=ident, toolchain_content=content)
        out.append({"segment": k, "schedule": sched,
                    **{f: head[f] for f in R4.MANIFEST_BOUND},
                    "evidence_dir": H5.EVIDENCE_DIRS[k]})
    return out


def write_manifest(repo, edit=None, *, stage="H5_STUDY", committed=True, name="manifest"):
    entries = manifest_entries()
    if edit:
        edit(entries[0])
    p = repo / f"{name}.json"
    p.write_text(json.dumps({"stage": stage, "segments": entries}))
    if committed:
        commit(repo, p)
    return str(p)


def study(repo, tmp_path, *, segment=0, out_dir=None, **kw):
    if "manifest" not in kw:
        kw["manifest"] = write_manifest(repo)
    return H5.run_study(mode="study", segment=segment,
                        out_dir=out_dir or H5.EVIDENCE_DIRS[segment],
                        classes=str(tmp_path / "cls"), deadline_s=60, **kw)


def nothing_ran(calls, out=None):
    if out is not None:
        assert not pathlib.Path(out).exists()
    assert (calls["load"], calls["build"], calls["compile"], calls["game"],
            calls["subprocess"], calls.get("toolchain_resolved", 0)) == (0, 0, 0, 0, 0, 0)


def test_STUDY_mode_refuses_while_the_gate_is_shut(repo, tmp_path, spies, placeholder):
    with pytest.raises(H5.H4RunError, match="UNAUTHORIZED"):
        study(repo, tmp_path)
    nothing_ran(spies, H5.EVIDENCE_DIRS[0])


def test_STUDY_mode_refuses_while_SEED_BLOCKS_is_EMPTY(repo, tmp_path, spies, study_open,
                                                      monkeypatch):
    m = write_manifest(repo)
    monkeypatch.setattr(H5, "SEED_BLOCKS", {})
    with pytest.raises(H5.H4RunError, match="SEED_BLOCKS is empty"):
        study(repo, tmp_path, manifest=m)
    nothing_ran(spies, H5.EVIDENCE_DIRS[0])


@pytest.mark.parametrize("seam", ["paths", "schedule", "incumbent_build", "evaluator",
                                  "incumbent_identity", "_compile"])
def test_STUDY_mode_accepts_NO_injected_seam(repo, tmp_path, spies, study_open, seam):
    value = {"paths": H5.T1jPaths(java="/j", jar="/x", classes="/c", ply_cap=280),
             "schedule": [{}], "incumbent_build": lambda t, e: None, "evaluator": object(),
             "incumbent_identity": {"stub": True}, "_compile": lambda d: {}}[seam]
    with pytest.raises(H5.H4RunError, match="accepts no injected seam"):
        study(repo, tmp_path, **{seam: value})
    nothing_ran(spies, H5.EVIDENCE_DIRS[0])


@pytest.mark.parametrize("how", ["untracked", "modified"])
def test_an_UNCOMMITTED_or_MODIFIED_manifest_is_refused(repo, tmp_path, spies, study_open,
                                                       how):
    if how == "untracked":
        m = write_manifest(repo, committed=False)
    else:
        m = write_manifest(repo)
        pathlib.Path(m).write_text(pathlib.Path(m).read_text() + "\n")
    with pytest.raises(H5.H4RunError, match="the manifest is not durable"):
        study(repo, tmp_path, manifest=m)
    nothing_ran(spies, H5.EVIDENCE_DIRS[0])


@pytest.mark.parametrize("stage", ["H4_STUDY", "H4_PILOT", "H5_SYNTHETIC_FIXTURE"])
def test_a_manifest_of_ANOTHER_STAGE_is_refused(repo, tmp_path, spies, study_open, stage):
    with pytest.raises(H5.H4RunError, match="is not a H5_STUDY manifest"):
        study(repo, tmp_path, manifest=write_manifest(repo, stage=stage))
    nothing_ran(spies, H5.EVIDENCE_DIRS[0])


def test_H4s_OWN_committed_study_manifest_is_refused(repo, tmp_path, spies, study_open):
    p = repo / "h4_study_manifest.json"
    shutil.copy2(ROOT / Q5.H4_STUDY_MANIFEST, p)
    commit(repo, p)
    with pytest.raises(H5.H4RunError, match="is not a H5_STUDY manifest"):
        study(repo, tmp_path, manifest=str(p))
    nothing_ran(spies, H5.EVIDENCE_DIRS[0])


def _segment_file(repo, k, last):
    d = pathlib.Path(H5.EVIDENCE_DIRS[k])
    d.mkdir()
    (d / "results.jsonl").write_text(json.dumps({"record_type": "header"}) + "\n"
                                     + json.dumps(last) + "\n")
    return d / "results.jsonl"


VOIDED = {"record_type": "run_void", "classification": "timeout"}
#: Card §10 A1: the positive predecessor is SYNTHETIC -- composed from fixture games
#: played by the real H5 runner, its envelope matching entry 0 -- and lives in a
#: temporary repository OUTSIDE the evidence tree.
SYNTHETIC_NOTE = "SYNTHETIC FIXTURE: composed for a test, NOT evidence"


def predecessor(repo, templates, edit=None, *, committed=True):
    e = manifest_entries()[0]
    d = pathlib.Path(H5.EVIDENCE_DIRS[0])
    assert not R4._inside(d, R4.EVIDENCE_ROOT), "a test predecessor is never evidence"
    d.mkdir()
    f = d / "results.jsonl"
    compose(f, templates, e, [("red", "black")] * 74, design="H5_STUDY")
    edit_records(f, lambda i, r: dict(r, synthetic_note=SYNTHETIC_NOTE) if i == 0 else r)
    if edit:
        edit_records(f, edit)
    if committed:
        commit(repo, f)
    return f


def _where(f, pred):
    return next(i for i, r in enumerate(R4.read_records(str(f))) if pred(r))


def _move(pred, to):
    """Edit that moves the first record matching `pred` to index `to`."""
    def edit_all(path):
        recs = R4.read_records(str(path))
        i = next(i for i, r in enumerate(recs) if pred(r))
        recs.insert(to, recs.pop(i))
        pathlib.Path(path).write_text("".join(json.dumps(r) + "\n" for r in recs))
    return edit_all


def _is(kind, task=None):
    return lambda r: r["record_type"] == kind and (task is None or r.get("task_id") == task)


def _header(**kw):
    return lambda i, r: dict(r, **kw) if i == 0 else r


def _only_header_and_end(i, r):
    return r if r["record_type"] in ("header", "segment_end") else None


def _drop_task(task):
    return lambda i, r: None if r.get("task_id") == task else r


def _dup(kind, task):
    """Edit_all: duplicate the first `kind` record of `task` in place."""
    def edit_all(path):
        recs = R4.read_records(str(path))
        i = next(i for i, r in enumerate(recs) if _is(kind, task)(r))
        recs.insert(i, dict(recs[i]))
        pathlib.Path(path).write_text("".join(json.dumps(r) + "\n" for r in recs))
    return edit_all


def _before_end(record):
    def edit_all(path):
        recs = R4.read_records(str(path))
        recs.insert(len(recs) - 1, record)
        pathlib.Path(path).write_text("".join(json.dumps(r) + "\n" for r in recs))
    return edit_all


def _append(record):
    def edit_all(path):
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    return edit_all


T0, T1 = "h5s0-00-A", "h5s0-00-B"
STRUCTURAL = {
    "bare_header_and_end": (_only_header_and_end, "no game_start where one is due"),
    "missing_last_game": (_drop_task("h5s0-73-B"), "no game_start where one is due"),
    "wrong_seed": (lambda i, r: dict(r, seed=r["seed"] + 1)
                   if r["record_type"] == "game_start" and r["game_index"] == 5 else r,
                   "is not the schedule's"),
    "games_out_of_order": (lambda i, r: dict(r, game_index=1 - r["game_index"])
                           if r["record_type"] == "game_start" and r["game_index"] < 2 else r,
                           "is not the schedule's"),
    "header_other_segment": (_header(segment=2), "the header is"),
    "header_fixture_design": (_header(design="H5_SYNTHETIC_FIXTURE", evidence=False),
                              "the header is"),
    "header_other_entry": (_header(schedule_digest="0" * 64),
                           "differ from manifest entry 0"),
    "duplicate_result": ("dup_result", "no game_start where one is due"),
    "stray_result": (lambda i, r: dict(r, task_id=T1)
                     if r["record_type"] == "game_result" and r["task_id"] == T0 else r,
                     "game_result of 'h5s0-00-B' where its own game_result is due"),
    "result_before_start": ("result_first", "no game_start where one is due"),
    "foreign_ply": (lambda i, r: dict(r, task_id=T1)
                    if r["record_type"] == "ply" and r["task_id"] == T0 and r["ply"] == 3
                    else r, "a ply record of 'h5s0-00-B' inside 'h5s0-00-A'"),
    "unknown_record": ("unknown", "mystery"),
    "run_void_before_end": ("void", "\\['run_void', 'segment_end'\\]"),
    "record_after_end": ("after_end", "\\['segment_end', 'ply'\\]"),
    "short_count": (lambda i, r: dict(r, games_completed=147)
                    if r["record_type"] == "segment_end" else r, "games_completed=147"),
    "incomplete_end": (lambda i, r: dict(r, complete=False)
                       if r["record_type"] == "segment_end" else r, "complete=False"),
}
EDIT_ALL = {"dup_result": _dup("game_result", T0),
            "result_first": _move(_is("game_result", T0), 1),
            "unknown": _before_end({"record_type": "mystery", "task_id": "h5s0-73-B"}),
            "void": _before_end(VOIDED),
            "after_end": _append({"record_type": "ply", "task_id": "h5s0-73-B"})}


@pytest.mark.parametrize("state,match", [
    ("absent", "needs segment 0 committed first"),
    ("uncommitted", "needs segment 0 committed first"),
    ("modified", "needs segment 0 committed first")], ids=["absent", "uncommitted", "modified"])
def test_segment_k_REFUSES_unless_segment_k_minus_1_is_COMMITTED(repo, tmp_path, spies,
                                                                study_open, templates,
                                                                state, match):
    m = write_manifest(repo)
    if state != "absent":
        f = predecessor(repo, templates, committed=state != "uncommitted")
        if state == "modified":
            f.write_text(f.read_text() + "\n")
    with pytest.raises(H5.H4RunError, match=match):
        study(repo, tmp_path, segment=1, manifest=m)
    nothing_ran(spies, H5.EVIDENCE_DIRS[1])


@pytest.mark.parametrize("case", list(STRUCTURAL))
def test_segment_k_REFUSES_a_predecessor_that_is_not_STRUCTURALLY_COMPLETE(
        repo, tmp_path, spies, study_open, templates, case):
    """Card §10 A1: each clause, alone. The file stays COMMITTED -- only its structure
    is wrong."""
    edit, match = STRUCTURAL[case]
    m = write_manifest(repo)
    f = predecessor(repo, templates, committed=False)
    if isinstance(edit, str):
        EDIT_ALL[edit](f)
    else:
        edit_records(f, edit)
    commit(repo, f)
    with pytest.raises(H5.H4RunError, match="needs segment 0 STRUCTURALLY COMPLETE: .*"
                       + match):
        study(repo, tmp_path, segment=1, manifest=m)
    nothing_ran(spies, H5.EVIDENCE_DIRS[1])


def test_segment_k_PROCEEDS_after_a_STRUCTURALLY_COMPLETE_synthetic_predecessor(
        repo, tmp_path, spies, study_open, templates):
    """CLEAN BASELINE for every clause above: the same synthetic 148-game file,
    unedited, gets segment 1 to its first game and no further."""
    m = write_manifest(repo)
    f = predecessor(repo, templates)
    views = H5.predecessor_views(str(f))
    assert sum(v["record_type"] == "game_start" for v in views) == 148
    assert R4.read_records(str(f))[0]["synthetic_note"] == SYNTHETIC_NOTE
    with pytest.raises(H5.H4RunVoid, match="SENTINEL"):
        study(repo, tmp_path, segment=1, manifest=m)
    header = R4.read_records(os.path.join(H5.EVIDENCE_DIRS[1], "results.jsonl"))[0]
    assert (header["design"], header["segment"], header["evidence"]) == ("H5_STUDY", 1, True)


#: Card §10 A1: the blind is PROCEDURAL -- the check keeps no outcome field and
#: names none. `terminal_reason` is held out too: the check has no use for it.
OUTCOME_FIELDS = set(R4.BLIND_FIELDS) | {"terminal_reason"}
PREDECESSOR_FNS = ("predecessor_views", "structural_faults", "check_predecessor")


def outcome_literals(source):
    tree = ast.parse(source)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in PREDECESSOR_FNS]
    assert len(fns) == len(PREDECESSOR_FNS)
    return sorted({n.value for f in fns for n in ast.walk(f)
                   if isinstance(n, ast.Constant) and n.value in OUTCOME_FIELDS})


def test_the_predecessor_check_is_BLIND_procedurally(templates, tmp_path):
    assert not set(sum(H5.PREDECESSOR_VIEW.values(), ())) & OUTCOME_FIELDS, \
        "the predecessor whitelist keeps an outcome field"
    assert outcome_literals(pathlib.Path(H5.__file__).read_text(encoding="utf-8")) == []
    src = tmp_path / "p.jsonl"
    compose(src, templates, _fixture_entry(), [("red", "black")] * 74, design="H5_STUDY")
    views = H5.predecessor_views(str(src))
    assert not {k for v in views for k in v} & OUTCOME_FIELDS, "a view carries an outcome field"
    assert "load_games" not in {n.attr for n in ast.walk(ast.parse(
        pathlib.Path(H5.__file__).read_text(encoding="utf-8"))) if isinstance(n, ast.Attribute)}


@pytest.mark.parametrize("plant", ['            r.get("winner")\n', '            x = "t1j_points"\n'],
                         ids=["winner_get", "points_literal"])
def test_the_blind_check_is_NOT_VACUOUS(plant):
    src = pathlib.Path(H5.__file__).read_text(encoding="utf-8")
    anchor = '            r = json.loads(line)\n'
    assert src.count(anchor) == 1
    assert outcome_literals(src.replace(anchor, anchor + plant)) != []


def _fixture_entry():
    from tests.test_h5_confirmatory_analysis import entry
    return entry(0)


@pytest.mark.parametrize("edit,field", [
    (lambda e: e.update(schedule_digest="0" * 64), "schedule_digest"),
    (lambda e: e.update(seeds=e["seeds"][::-1]), "seeds"),
    (lambda e: e["incumbent_identity"].update(design="H4_STUDY"), "incumbent_identity"),
    (lambda e: e["cards"].update({H5.CARDS[1]: "0" * 64}), "cards"),
    (lambda e: e["code"].update({H5.CODE[-1]: "0" * 64}), "code"),
    (lambda e: e["t1j_runtime"].update(ply_cap=274), "t1j_runtime"),
    (lambda e: e["t1j_runtime"].update(h4_acceptance=False), "t1j_runtime")])
def test_a_manifest_that_would_not_bind_is_refused_BEFORE_ANYTHING_is_claimed(
        repo, tmp_path, spies, study_open, edit, field):
    with pytest.raises(H5.H4RunError, match=f"would not bind to its manifest: .*'{field}'"):
        study(repo, tmp_path, manifest=write_manifest(repo, edit=edit))
    nothing_ran(spies, H5.EVIDENCE_DIRS[0])


def _swap_arm_seeds(e):
    a, b = e["schedule"][0], e["schedule"][1]
    a["seed"], b["seed"] = b["seed"], a["seed"]


def _h4_schedule(e):
    _cfg, ident = _identity()
    e["schedule"] = R4.canonical_schedule("study", 0, ident)


@pytest.mark.parametrize("edit", [_swap_arm_seeds, _h4_schedule,
                                  lambda e: e["schedule"][0].update(pair_id="x"),
                                  lambda e: e.update(schedule=e["schedule"][:-2])])
def test_a_NON_CANONICAL_schedule_H4s_included_is_refused(repo, tmp_path, spies,
                                                          study_open, edit):
    with pytest.raises(H5.H4RunError, match="not the CANONICAL|whole pairs|adjacent"):
        study(repo, tmp_path, manifest=write_manifest(repo, edit=edit))
    nothing_ran(spies, H5.EVIDENCE_DIRS[0])


@pytest.mark.parametrize("registry,match", [
    ("ACCOUNTED", "not accounted"), ("EXPOSED", "exposed"), ("RETIRED", "retired"),
    ("TEST_ONLY", "test_only"), ("CONSUMED", "consumed")])
def test_a_seed_the_SHARED_REGISTRY_forbids_is_refused(repo, tmp_path, spies, study_open,
                                                      monkeypatch, registry, match):
    m = write_manifest(repo)
    lo = PLACEHOLDERS[0][0]
    if registry == "ACCOUNTED":
        monkeypatch.setattr(REF, "ACCOUNTED_SEED_INTERVALS", tuple(
            iv for iv in REF.ACCOUNTED_SEED_INTERVALS if iv != PLACEHOLDERS[0]))
    elif registry == "CONSUMED":
        monkeypatch.setattr(REF, "CONSUMED_SEEDS", tuple(REF.CONSUMED_SEEDS) + (lo,))
    else:
        name = f"{registry}_SEED_INTERVALS"
        monkeypatch.setattr(REF, name, tuple(getattr(REF, name)) + ((lo, lo + 1),))
    with pytest.raises(H5.H4RunError, match=f"fail the shared registry.*{match}"):
        study(repo, tmp_path, manifest=m)
    nothing_ran(spies, H5.EVIDENCE_DIRS[0])


def test_an_H4_SEED_BLOCK_is_refused_by_the_registry(repo, tmp_path, spies, study_open,
                                                    monkeypatch):
    """Every H4 block is EXPOSED and RETIRED: an H5 block equal to one never runs."""
    monkeypatch.setattr(H5, "SEED_BLOCKS", {**PLACEHOLDERS, 0: R4.SEED_BLOCKS[("study", 0)]})
    with pytest.raises(H5.H4RunError, match="fail the shared registry.*exposed, retired"):
        study(repo, tmp_path)
    nothing_ran(spies, H5.EVIDENCE_DIRS[0])


def test_the_run_writes_ONLY_to_its_manifests_evidence_dir(repo, tmp_path, spies,
                                                          study_open):
    with pytest.raises(H5.H4RunError, match="writes ONLY to its manifest's evidence_dir"):
        study(repo, tmp_path, out_dir=str(tmp_path / "elsewhere"))
    assert not (tmp_path / "elsewhere").exists()
    h4_dir = R4.EVIDENCE_DIRS[("study", 0)]
    with pytest.raises(H5.H4RunError, match="writes ONLY to its manifest's evidence_dir"):
        study(repo, tmp_path, manifest=write_manifest(
            repo, edit=lambda e: e.update(evidence_dir=h4_dir), name="m2"))
    nothing_ran(spies, H5.EVIDENCE_DIRS[0])


def test_an_evidence_dir_LEFT_BY_A_VOID_refuses_and_nothing_runs(repo, tmp_path, spies,
                                                                   study_open):
    m = write_manifest(repo)
    _segment_file(repo, 0, VOIDED)
    with pytest.raises(H5.H4RunError, match="occupied"):
        study(repo, tmp_path, manifest=m)
    assert (spies["load"], spies["compile"], spies["game"], spies["build"],
            spies["subprocess"]) == (0, 0, 0, 0, 0)
    assert os.listdir(H5.EVIDENCE_DIRS[0]) == ["results.jsonl"], "the VOID's record kept"


def test_a_task_naming_ANOTHER_incumbent_is_refused(repo, tmp_path, spies, study_open):
    def edit(e):
        e["schedule"][0]["reference_sha1"] = "0" * 40
    with pytest.raises(H5.H4RunError, match="another incumbent reference"):
        study(repo, tmp_path, manifest=write_manifest(repo, edit=edit))
    nothing_ran(spies, H5.EVIDENCE_DIRS[0])


def test_a_TOOLCHAIN_that_compiles_differently_VOIDS_before_the_first_game(
        repo, tmp_path, spies, study_open):
    spies["toolchain"] = dict(FAKE_TOOLCHAIN, classes={"E3bDump.class": "e" * 64})
    with pytest.raises(H5.H4RunVoid, match="does not bind to its manifest") as ei:
        study(repo, tmp_path)
    assert ei.value.classification == "unexpected"
    recs = R4.read_records(os.path.join(H5.EVIDENCE_DIRS[0], "results.jsonl"))
    assert [r["record_type"] for r in recs] == ["run_void"], "no header, no game"
    assert (spies["load"], spies["compile"], spies["game"], spies["build"]) == (1, 1, 0, 0)


def test_a_manifest_that_BINDS_reaches_the_first_game_and_no_further(repo, tmp_path, spies,
                                                                     study_open):
    """CLEAN BASELINE for every study-mode refusal above."""
    with pytest.raises(H5.H4RunVoid, match="SENTINEL"):
        study(repo, tmp_path)
    header, void = R4.read_records(os.path.join(H5.EVIDENCE_DIRS[0], "results.jsonl"))
    assert (header["design"], header["evidence"]) == ("H5_STUDY", True)
    assert header["t1j_local"] == {k: FAKE_TOOLCHAIN[k] for k in R4.TOOLCHAIN_LOCAL}
    assert void["stage"] == "game" and spies["game"] == 1 and spies["build"] == 0


def test_the_CLI_refuses_a_manifest_that_would_not_bind_with_exit_4(repo, tmp_path, spies,
                                                                   study_open):
    m = write_manifest(repo, edit=lambda e: e.update(seeds=e["seeds"][::-1]))
    assert H5.main(["--manifest", m, "--segment", "0", "--out-dir", H5.EVIDENCE_DIRS[0],
                    "--classes", str(tmp_path / "c")]) == H5.EXIT_REFUSED
    nothing_ran(spies, H5.EVIDENCE_DIRS[0])


# ───────────────── 7. gate SEPARATION and H4 / H5 material (§1, §8) ─────────────────

def test_with_EVERY_H4_gate_open_every_H5_entry_still_refuses(tmp_path, wire, monkeypatch):
    monkeypatch.setattr(AUTH4, "H4_PILOT_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(QAUTH4, "H4_PRODUCTION_QUALIFICATION_AUTHORIZED", True)
    monkeypatch.setattr(C4, "H4_STUDY_AGGREGATION_AUTHORIZED", True)
    with pytest.raises(H5.H4RunError, match="H5 study is UNAUTHORIZED"):
        run(tmp_path)
    with pytest.raises(Q5.QualificationUnauthorized):
        Q5.qualify(str(tmp_path / "rec"), str(tmp_path / "cls"))
    with pytest.raises(C5.H5AggregationUnauthorized):
        C5.run_confirmatory(study_manifest="m", segments=[])
    assert wire["calls"] == [] and os.listdir(tmp_path) == []


def test_with_EVERY_H5_gate_open_every_H4_entry_still_refuses(tmp_path, wire, monkeypatch):
    monkeypatch.setattr(AUTH, "H5_STUDY_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(QAUTH5, "H5_PRODUCTION_QUALIFICATION_AUTHORIZED", True)
    monkeypatch.setattr(CAUTH5, "H5_STUDY_AGGREGATION_AUTHORIZED", True)
    sched = R4.make_schedule([("p0", -1, -2)], mode="fixture")
    with pytest.raises(R4.H4RunError, match="H4 pilot is UNAUTHORIZED"):
        R4.run_games(mode="fixture", schedule=sched, out_dir=str(tmp_path / "run"),
                     paths=_paths(tmp_path), deadline_s=60, incumbent_build=stub_incumbent,
                     incumbent_identity={"stub": True}, _compile=lambda d: {})
    with pytest.raises(Q4.QualificationUnauthorized):
        Q4.qualify(str(tmp_path / "rec"), str(tmp_path / "cls"))
    with pytest.raises(C4.H4AggregationUnauthorized):
        C4.run_confirmatory(pilot_manifest="m", study_manifest="s", artifact="a",
                            pilot_results="r", feasibility_report="f", segments=[],
                            out=str(tmp_path / "e.json"))
    assert wire["calls"] == [] and os.listdir(tmp_path) == []


def test_H4s_RUNNER_refuses_an_H5_manifest(repo, tmp_path, spies, placeholder, monkeypatch):
    monkeypatch.setattr(AUTH4, "H4_PILOT_EXECUTION_AUTHORIZED", True)
    with pytest.raises(R4.H4RunError, match="is not a H4_STUDY manifest"):
        R4.run_games(mode="study", manifest=write_manifest(repo), segment=0,
                     out_dir=str(tmp_path / "out"), classes=str(tmp_path / "c"),
                     deadline_s=60)
    nothing_ran(spies, tmp_path / "out")
