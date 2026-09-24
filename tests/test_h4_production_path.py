"""H4 step 3P-a: the production path (runner card §12, §12.6). GATES CLOSED.

No JVM, no checkpoint load, no research seed, no game. The production incumbent's
identity is computed for REAL (it reads the frozen L0 plan and the qualified
settings, loading no model); the checkpoint loader, the reference-agent builder
and the helper compile are stubbed at their module attributes, because the real
ones refuse to run inside a test process or need the external toolchain. Pilot-
mode tests use placeholder positive seeds 1..n, outside every registered block,
and every one of them stops BEFORE any incumbent is built.
"""
import ast
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

from scripts.GPU.alphazero import e4_screen_command as SCREEN_CMD
from scripts.GPU.alphazero import h4_pilot_authorization as AUTH
from scripts.GPU.alphazero import h4_production_qualification as Q
from scripts.GPU.alphazero import h4_production_qualification_authorization as QAUTH
from scripts.GPU.alphazero import h4_runner as R
from scripts.GPU.alphazero import twixtbot_g3_reference as G3
from tests.test_h4_runner import FAKE_TOOLCHAIN, card_code_list

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "scripts"
ROOTS = ("scripts/GPU/alphazero/h4_runner.py",
         "scripts/GPU/alphazero/h4_production_qualification.py")
GATES = {"scripts/GPU/alphazero/h4_pilot_authorization.py": "H4_PILOT_EXECUTION_AUTHORIZED",
         "scripts/GPU/alphazero/h4_production_qualification_authorization.py":
             "H4_PRODUCTION_QUALIFICATION_AUTHORIZED"}


# ───────────────── 1. the gate modules hold ONLY their declaration (§12.6.2) ─────────────────

def gate_module_faults(source, gate):
    body = ast.parse(source).body
    bad = []
    if not (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        bad.append("no docstring first")
    rest = body[1:]
    if len(rest) != 1:
        bad.append(f"{len(rest)} statements after the docstring, not 1")
    for n in rest:
        ok = (isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == gate
              and isinstance(n.value, ast.Constant) and isinstance(n.value.value, bool))
        if not ok:
            bad.append(f"line {n.lineno}: {ast.unparse(n)[:60]}")
    return bad


@pytest.mark.parametrize("path,gate", list(GATES.items()))
def test_each_GATE_MODULE_is_only_its_declaration_and_is_CLOSED(path, gate):
    src = (ROOT / path).read_text(encoding="utf-8")
    assert gate_module_faults(src, gate) == []
    assert "= False\n" in src


@pytest.mark.parametrize("plant", ["import os\n", "def play(state):\n    return (0, 0)\n",
                                   "OTHER_SWITCH = True\n",
                                   "H4_PILOT_EXECUTION_AUTHORIZED = bool(1)\n"])
def test_the_gate_module_walker_is_NOT_VACUOUS(plant):
    """CLEAN-BASELINE CONTROL: play logic planted in a gate module is seen."""
    path, gate = next(iter(GATES.items()))
    src = (ROOT / path).read_text(encoding="utf-8") + plant
    assert gate_module_faults(src, gate) != []


def test_the_gates_are_published_CLOSED():
    assert AUTH.H4_PILOT_EXECUTION_AUTHORIZED is False
    assert QAUTH.H4_PRODUCTION_QUALIFICATION_AUTHORIZED is False


# ───────────────── 2. `code` is the whole first-party play path (§12.6.1) ─────────────────

def _resolve(repo, f, node):
    """First-party files one import node names."""
    out, here = [], f.parent
    if isinstance(node, ast.ImportFrom):
        if node.level:
            base = here
            for _ in range(node.level - 1):
                base = base.parent
            target = base.joinpath(*(node.module or "").split(".")) if node.module else base
        elif (node.module or "").startswith("scripts."):
            target = repo.joinpath(*node.module.split("."))
        else:
            return out
        names = [target] + [target / a.name for a in node.names]
    else:
        names = [repo.joinpath(*a.name.split(".")) for a in node.names
                 if a.name.startswith("scripts.")]
    for t in names:
        for c in (t.with_suffix(".py"), t / "__init__.py"):
            if c.is_file():
                out.append(c)
    return out


def play_path_closure(repo, roots=ROOTS):
    """(files, dynamic imports): every first-party file reachable by import AT ANY
    DEPTH from the roots, plus every package __init__ on the way."""
    seen, todo, dynamic = set(), [repo / r for r in roots], []
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for pkg in f.parents:
            if pkg == repo:
                break
            if (pkg / "__init__.py").is_file() and pkg / "__init__.py" not in seen:
                todo.append(pkg / "__init__.py")
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                todo.extend(_resolve(repo, f, n))
            elif isinstance(n, ast.Call) and (
                    getattr(n.func, "attr", None) == "import_module"
                    or getattr(n.func, "id", None) == "__import__"):
                dynamic.append(f"{f.relative_to(repo)}: {ast.unparse(n)[:60]}")
    return sorted(str(p.relative_to(repo)) for p in seen), dynamic


def test_CODE_is_EXACTLY_the_play_path_closure_minus_the_gate_modules():
    files, dynamic = play_path_closure(ROOT)
    assert dynamic == [], "a dynamic import cannot be followed; it is refused"
    assert sorted(set(files) - set(R.CODE)) == sorted(GATES) == sorted(R.GATE_MODULES)
    assert sorted(set(files) - set(GATES)) == sorted(R.CODE)


def test_CODE_is_the_list_frozen_in_the_CARD():
    assert list(R.CODE) == card_code_list()
    assert len(R.CODE) == len(set(R.CODE)) == 37


@pytest.mark.parametrize("plant,want", [
    ("\n\ndef _later():\n    from . import brand_new_dependency\n", "new"),
    ("\n\ndef _later():\n    import importlib\n    importlib.import_module('x')\n",
     "dynamic")])
def test_a_NEW_dependency_cannot_enter_unlisted(tmp_path, plant, want):
    """CLEAN-BASELINE CONTROL: a copy of the path, with a FUNCTION-LEVEL import of
    a new module planted in one listed file -- the closure grows, or the dynamic
    import is caught."""
    for rel in list(R.CODE) + list(GATES):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, tmp_path / rel)
    (tmp_path / "scripts/GPU/alphazero/brand_new_dependency.py").write_text("X = 1\n")
    victim = tmp_path / "scripts/GPU/alphazero/twixtbot_g3_schedule.py"
    victim.write_text(victim.read_text() + plant)
    files, dynamic = play_path_closure(tmp_path)
    if want == "new":
        assert "scripts/GPU/alphazero/brand_new_dependency.py" in files
        assert sorted(set(files) - set(GATES)) != sorted(R.CODE)
    else:
        assert dynamic


# ───────────────────── 3. the production incumbent (§12.1 items 1-3) ─────────────────────

def _identity():
    cfg = R.frozen_argmax_config()
    return cfg, R.frozen_incumbent_identity(cfg, design="H4_PILOT")


def test_the_FROZEN_identity_passes_its_own_check():
    """CLEAN BASELINE: the check is not refusing everything."""
    cfg, ident = _identity()
    R.check_incumbent_identity(ident, cfg, design="H4_PILOT")
    assert ident["eval_config"]["selection_mode"] == R.H2R.SELECTION_MODE


@pytest.mark.parametrize("tamper", [
    lambda i: i.update(reference_sha1="0" * 40),
    lambda i: i.update(design="H4_STUDY"),
    lambda i: i["eval_config"].update(selection_mode="opening_temperature"),
    lambda i: i["argmax_config"].update(
        {k: float(v) for k, v in i["argmax_config"].items()
         if type(v) is int and not isinstance(v, bool)})])
def test_a_TAMPERED_identity_is_refused_type_strictly(tamper):
    cfg, ident = _identity()
    ident = json.loads(json.dumps(ident))
    tamper(ident)
    with pytest.raises(R.H4RunError):
        R.check_incumbent_identity(ident, cfg, design="H4_PILOT")


def test_a_CONFIG_that_would_play_differently_is_refused():
    cfg, ident = _identity()
    other = cfg.__class__(**{**cfg.__dict__, "selection_mode": "opening_temperature"})
    with pytest.raises(R.H4RunError):
        R.check_incumbent_identity(ident, other, design="H4_PILOT")


class StubEvaluator:
    def __init__(self, reference, sha1):
        self._g3_reference, self._g3_sha1 = reference, sha1


def test_the_evaluator_must_carry_THE_identitys_checkpoint(monkeypatch):
    _cfg, ident = _identity()
    monkeypatch.setattr(SCREEN_CMD, "_default_load_evaluator",
                        lambda root: StubEvaluator(ident["reference"], ident["reference_sha1"]))
    assert R.load_production_evaluator(ident)._g3_sha1 == ident["reference_sha1"]
    monkeypatch.setattr(SCREEN_CMD, "_default_load_evaluator",
                        lambda root: StubEvaluator(ident["reference"], "0" * 40))
    with pytest.raises(R.H4RunError, match="loaded evaluator"):
        R.load_production_evaluator(ident)


def test_the_QUALIFIED_LOADER_recomputes_the_checkpoint_sha1(tmp_path):
    """Card §12.1 item 3: the existing loader already binds the checkpoint -- shown
    on a same-named file with other bytes. It refuses BEFORE any model is built."""
    meta = G3.REFERENCE_CHECKPOINTS["calib020_0001"]
    fake = tmp_path / meta["path"]
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"not the checkpoint")
    with pytest.raises(G3.ReferenceError, match="!= pinned"):
        G3.load_reference_evaluator("calib020_0001", str(tmp_path))


def test_the_builder_is_handed_THE_config_object(monkeypatch):
    cfg, _ident = _identity()
    seen = {}
    monkeypatch.setattr(G3, "build_reference_agent", lambda **kw: seen.update(kw) or "agent")
    build = R.production_incumbent_build(cfg)
    task = R.make_schedule([("p", 1, 2)], mode="pilot")[0]
    assert build(task, "ev") == "agent"
    assert seen["config"] is cfg and build.config is cfg
    assert seen["colour"] == task["incumbent_colour"] == "red"


# ───────────────────── 4. pilot and study mode (§12.1 items 4-6, §12.6.3) ─────────────────────

@pytest.fixture
def pilot_open(monkeypatch):
    monkeypatch.setattr(AUTH, "H4_PILOT_EXECUTION_AUTHORIZED", True)


@pytest.fixture
def spies(monkeypatch):
    """Records every load, build, compile, game and process: tests prove NONE ran."""
    calls = {"load": 0, "build": 0, "compile": 0, "game": 0, "subprocess": 0}
    _cfg, ident = _identity()

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

    def proc(*a, **k):
        calls["subprocess"] += 1
        raise AssertionError("a process was started")

    def verified(classes):
        calls["toolchain_resolved"] = calls.get("toolchain_resolved", 0) + 1
        return R.T1jPaths(java="/j", jar="/x.jar", classes=classes, ply_cap=280)
    monkeypatch.setattr(SCREEN_CMD, "_default_load_evaluator", load)
    monkeypatch.setattr(G3, "build_reference_agent", build)
    monkeypatch.setattr(R, "_compile_helper_verified", comp)
    monkeypatch.setattr(R, "play_game", game)
    monkeypatch.setattr(subprocess, "run", proc)
    monkeypatch.setattr(R, "_verified_t1j_paths", verified)
    return calls


def write_manifest(tmp_path, mode="pilot", edit=None, content=None):
    _cfg, ident = _identity()
    sched = R.make_schedule([("p0", 1, 2), ("p1", 3, 4)], mode=mode, reference=ident)
    content = content or {k: FAKE_TOOLCHAIN[k] for k in R.TOOLCHAIN_CONTENT}
    head = R.header_candidate(mode=mode, segment=0, schedule=sched,
                              incumbent_identity=ident, toolchain_content=content)
    entry = {"segment": 0, "schedule": sched, **{k: head[k] for k in R.MANIFEST_BOUND}}
    if edit:
        edit(entry)
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"stage": R.DESIGNS[mode], "segments": [entry]}))
    return str(p)


def pilot(tmp_path, **kw):
    if "manifest" not in kw:                      # NOT setdefault: it would write eagerly
        kw["manifest"] = write_manifest(tmp_path)  # over an edited manifest
    return R.run_games(mode="pilot", out_dir=str(tmp_path / "out"),
                       classes=str(tmp_path / "cls"), deadline_s=60, **kw)


def nothing_ran(tmp_path, calls):
    assert not (tmp_path / "out").exists()
    assert (calls["load"], calls["build"], calls["compile"], calls["game"],
            calls["subprocess"], calls.get("toolchain_resolved", 0)) == (0, 0, 0, 0, 0, 0)


def test_PILOT_mode_refuses_while_the_gate_is_shut(tmp_path, spies):
    with pytest.raises(R.H4RunError, match="UNAUTHORIZED"):
        pilot(tmp_path)
    nothing_ran(tmp_path, spies)


@pytest.mark.parametrize("seam", ["paths", "schedule", "incumbent_build", "evaluator",
                                  "incumbent_identity", "_compile"])
def test_PILOT_mode_accepts_NO_injected_seam(tmp_path, spies, pilot_open, seam):
    value = {"paths": R.T1jPaths(java="/j", jar="/x", classes="/c", ply_cap=280),
             "schedule": [{}], "incumbent_build": lambda t, e: None, "evaluator": object(),
             "incumbent_identity": {"stub": True}, "_compile": lambda d: {}}[seam]
    with pytest.raises(R.H4RunError, match="accepts no injected seam"):
        pilot(tmp_path, **{seam: value})
    nothing_ran(tmp_path, spies)


@pytest.mark.parametrize("manifest,match", [
    (None, "plays only a committed manifest"),
    (lambda t: write_manifest(t, mode="study"), "not a H4_PILOT manifest"),
    (lambda t: str(t / "missing.json"), "unreadable")])
def test_PILOT_mode_refuses_a_missing_or_WRONG_STAGE_manifest(tmp_path, spies, pilot_open,
                                                             manifest, match):
    with pytest.raises(R.H4RunError, match=match):
        R.run_games(mode="pilot", manifest=manifest(tmp_path) if manifest else None,
                    out_dir=str(tmp_path / "out"), deadline_s=60,
                    classes=str(tmp_path / "c"))
    nothing_ran(tmp_path, spies)


@pytest.mark.parametrize("edit,field", [
    (lambda e: e.update(schedule_digest="0" * 64), "schedule_digest"),
    (lambda e: e.update(seeds=e["seeds"][::-1]), "seeds"),
    (lambda e: e["incumbent_identity"].update(design="H4_STUDY"), "incumbent_identity"),
    (lambda e: e["cards"].update({R.CARDS[3]: "0" * 64}), "cards"),
    (lambda e: e["code"].update({R.CODE[-1]: "0" * 64}), "code"),
    (lambda e: e["t1j_runtime"].update(ply_cap=274), "t1j_runtime"),
    (lambda e: e["t1j_runtime"].update(depth=6.0), "t1j_runtime")])
def test_a_manifest_that_would_not_bind_is_refused_BEFORE_ANYTHING_is_claimed(
        tmp_path, spies, pilot_open, edit, field):
    with pytest.raises(R.H4RunError, match=f"would not bind to its manifest: .*'{field}'"):
        pilot(tmp_path, manifest=write_manifest(tmp_path, edit=edit))
    nothing_ran(tmp_path, spies)


def test_a_task_naming_ANOTHER_incumbent_is_refused(tmp_path, spies, pilot_open):
    def edit(e):
        e["schedule"][0]["reference_sha1"] = "0" * 40
    with pytest.raises(R.H4RunError, match="another incumbent reference"):
        pilot(tmp_path, manifest=write_manifest(tmp_path, edit=edit))
    nothing_ran(tmp_path, spies)


def test_a_TOOLCHAIN_that_compiles_differently_VOIDS_before_the_first_game(
        tmp_path, spies, pilot_open):
    spies["toolchain"] = dict(FAKE_TOOLCHAIN, classes={"E3bDump.class": "e" * 64})
    with pytest.raises(R.H4RunVoid, match="does not bind to its manifest") as ei:
        pilot(tmp_path)
    assert ei.value.classification == "unexpected"
    recs = R.read_records(str(tmp_path / "out" / "results.jsonl"))
    assert [r["record_type"] for r in recs] == ["run_void"], "no header, no game"
    assert (spies["load"], spies["compile"], spies["game"], spies["build"]) == (1, 1, 0, 0)
    assert spies["toolchain_resolved"] == 1, "the VERIFIED toolchain, resolved by the runner"


def test_a_manifest_that_BINDS_reaches_the_first_game_and_no_further(tmp_path, spies,
                                                                     pilot_open):
    """CLEAN BASELINE for the binding: a matching manifest gets through BOTH checks.
    The first game is a sentinel, so no incumbent is ever built."""
    with pytest.raises(R.H4RunVoid) as ei:
        pilot(tmp_path)
    assert "SENTINEL" in str(ei.value)
    header, void = R.read_records(str(tmp_path / "out" / "results.jsonl"))
    assert (header["design"], header["evidence"]) == ("H4_PILOT", True)
    assert header["t1j_local"] == {k: FAKE_TOOLCHAIN[k] for k in R.TOOLCHAIN_LOCAL}
    assert void["stage"] == "game" and spies["game"] == 1 and spies["build"] == 0


def test_the_CLI_refuses_a_manifest_that_would_not_bind_with_exit_4(tmp_path, spies,
                                                                   pilot_open):
    m = write_manifest(tmp_path, edit=lambda e: e.update(seeds=e["seeds"][::-1]))
    assert R.main(["--mode", "pilot", "--manifest", m, "--segment", "0", "--out-dir",
                   str(tmp_path / "out"), "--classes", str(tmp_path / "c")]) == R.EXIT_REFUSED
    nothing_ran(tmp_path, spies)


# ───────────────────── 5. the 3P-b qualification, NOT run (§12.2) ─────────────────────

def test_the_QUALIFICATION_refuses_while_its_gate_is_shut(tmp_path, spies):
    with pytest.raises(Q.QualificationUnauthorized):
        Q.qualify(str(tmp_path / "rec"), str(tmp_path / "cls"))
    assert not (tmp_path / "rec").exists() and not (tmp_path / "cls").exists()
    assert spies["load"] == spies["compile"] == 0


def test_the_QUALIFICATION_CLI_exits_5_as_a_fresh_subprocess(tmp_path):
    r = subprocess.run([sys.executable, "-m",
                        "scripts.GPU.alphazero.h4_production_qualification",
                        "--out-dir", str(tmp_path / "rec"),
                        "--classes-root", str(tmp_path / "cls")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == Q.EXIT_UNAUTHORIZED, f"exit {r.returncode}, {r.stderr!r}"
    assert not (tmp_path / "rec").exists() and not (tmp_path / "cls").exists()


def test_the_qualification_gate_is_read_at_CALL_TIME_at_both_entries():
    tree = ast.parse(pathlib.Path(Q.__file__).read_text(encoding="utf-8"))
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for e in ("qualify", "main"):
        assert any(isinstance(n, ast.Attribute)
                   and n.attr == "H4_PRODUCTION_QUALIFICATION_AUTHORIZED"
                   and getattr(n.value, "id", None) == "QAUTH" for n in ast.walk(fns[e])), e


@pytest.fixture
def qual_open(monkeypatch):
    """Gate open, every external act stubbed: the checkpoint loader, the builder
    (an agent playing one legal move), the compile, and the toolchain paths."""
    monkeypatch.setattr(QAUTH, "H4_PRODUCTION_QUALIFICATION_AUTHORIZED", True)
    _cfg, ident = _identity()
    box = {"sha1": ident["reference_sha1"], "move": (0, 4), "classes": [{}, {}], "n": 0}
    monkeypatch.setattr(SCREEN_CMD, "_default_load_evaluator",
                        lambda root: StubEvaluator(ident["reference"], box["sha1"]))
    monkeypatch.setattr(G3, "build_reference_agent", lambda **kw: (lambda s: box["move"]))

    def comp(deadline, *, paths):
        box["n"] += 1
        return dict(FAKE_TOOLCHAIN, classes_dir=paths.classes,
                    classes=box["classes"][box["n"] - 1] or FAKE_TOOLCHAIN["classes"])
    monkeypatch.setattr(Q.D1, "_default_compile", comp)
    monkeypatch.setattr(R, "_verified_t1j_paths", lambda classes: R.T1jPaths(
        java="/j", jar="/x", classes=classes, ply_cap=280))
    return box


def test_a_CLEAN_qualification_records_the_bound_identity_from_the_PILOTS_function(
        tmp_path, qual_open):
    rec = Q.qualify(str(tmp_path / "rec"), str(tmp_path / "cls"))
    assert rec["result"] == "CLEAN", rec["reason"]
    assert json.loads((tmp_path / "rec" / "record.json").read_text()) == \
        json.loads(json.dumps(rec))
    cfg, ident = _identity()
    content = {k: FAKE_TOOLCHAIN[k] for k in R.TOOLCHAIN_CONTENT}
    assert rec["checks"]["bound"] == R.bound_identity(ident, content), \
        "the qualification's bound identity is not the pilot header function's"
    assert rec["checks"]["pilot_cli_exit"] == R.EXIT_UNAUTHORIZED
    assert rec["checks"]["incumbent_move"] == [0, 4] and qual_open["n"] == 2
    assert "NOT evidence of strength" in rec["evidence_note"]


@pytest.mark.parametrize("fault,match", [
    ({"classes": [{}, {"E3bDump.class": "e" * 64}]}, "does not compile reproducibly"),
    ({"move": (0, 0)}, "not a legal empty-board move"),
    ({"sha1": "0" * 40}, "checkpoint")])
def test_a_failed_check_is_a_recorded_STOP(tmp_path, qual_open, fault, match):
    qual_open.update(fault)
    rec = Q.qualify(str(tmp_path / "rec"), str(tmp_path / "cls"))
    assert rec["result"] == "STOP" and match in rec["reason"]
    assert rec["checks"] is None
    assert (tmp_path / "rec" / "record.json").exists()


@pytest.mark.parametrize("where", ["occupied_record", "classes_in_repo"])
def test_the_qualification_REFUSES_a_bad_destination_before_anything(tmp_path, qual_open,
                                                                    where):
    rec, cls = tmp_path / "rec", tmp_path / "cls"
    if where == "occupied_record":
        rec.mkdir()
    else:
        cls = ROOT / "never-created-h4-classes"
    with pytest.raises(Q.QualificationRefused):
        Q.qualify(str(rec), str(cls))
    assert qual_open["n"] == 0 and not (ROOT / "never-created-h4-classes").exists()


def test_a_pilot_CLI_that_does_not_refuse_is_a_recorded_STOP(tmp_path, qual_open,
                                                             monkeypatch):
    monkeypatch.setattr(Q, "_pilot_cli_refuses", lambda tmp: 0)
    rec = Q.qualify(str(tmp_path / "rec"), str(tmp_path / "cls"))
    assert rec["result"] == "STOP" and "the pilot CLI exited 0" in rec["reason"]
