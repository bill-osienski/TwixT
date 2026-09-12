"""H2 CONTAINMENT -- the protections added after the 2026-09-12 incident.

🔴 WHAT HAPPENED. An injected-defect control deleted `check_gate()` from `run_h2`.
The gate was the only thing between that control and a real match, and deleting
safeguards is exactly what the harness does -- so the control resolved the
toolchain, compiled the helper, loaded the model and PLAYED 383 REAL GAMES,
spending a registered seed block.

These tests are deliberately NARROW and deliberately SEPARATE from the
real-builder compatibility tests: they assert that the production boundary cannot
be reached from a test process, whatever authorization checks have been removed.
Nothing here starts a JVM, loads a model, draws a seed or plays a game -- and the
point is that nothing here CAN.
"""
import ast
import inspect
import os
import pathlib
import sys

import pytest

from scripts.GPU.alphazero import h2_match_plan as PLAN
from scripts.GPU.alphazero import h2_match_runner as RUN


def _task():
    return PLAN.build_tasks(PLAN.load_source_plan())[0]


# ───────────────── 1. the seam checks authorization ITSELF ──────────────────

def _call_lines(src: str, inner: str = None) -> dict:
    """EARLIEST source line at which each name is called, inside `src`'s function
    (or its nested `inner`).

    🔑 `ast.walk` is BREADTH-FIRST, so it reaches a shallow later statement before a
    deeper earlier one: the first version of this helper read the seam's SECOND
    `check_gate()` (a direct child of `play`) instead of its first (nested inside
    `if state is None:`) and reported the guard as coming AFTER the toolchain. The
    fix is `min` over every occurrence -- which is also the right question: the
    EARLIEST guard must precede the EARLIEST effect.
    """
    import textwrap
    fn = ast.parse(textwrap.dedent(src)).body[0]
    if inner is not None:
        fn = next(n for n in ast.walk(fn)
                  if isinstance(n, ast.FunctionDef) and n.name == inner)
    lines = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if name:
                lines[name] = min(lines.get(name, node.lineno), node.lineno)
    return lines


def _seam_lines(path: str) -> dict:
    """`_call_lines` for the production seam of an arbitrary module: the INNERMOST
    function that resolves the toolchain. Found by structure, not by name -- H1 calls
    it `setup` and H2 calls it `play`."""
    mod = ast.parse(pathlib.Path(path).read_text())
    best = None
    for node in ast.walk(mod):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(getattr(getattr(n, "func", None), "attr", "") == "verified_paths"
                   for n in ast.walk(node) if isinstance(n, ast.Call)):
            continue
        if best is None or node.lineno > best.lineno:   # innermost = latest def
            best = node
    assert best is not None, f"{path} has no production seam"
    lines = {}
    for node in ast.walk(best):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if name:
                lines[name] = min(lines.get(name, node.lineno), node.lineno)
    return lines


def test_THE_SEAM_CHECKS_THE_GATE_ITSELF_not_only_the_entry():
    """The second check. `run_h2` checks the gate; so does the seam, so deleting
    one leaves the other."""
    play = RUN._production_play("/tmp/h2-never-written")
    with pytest.raises(RUN.H2Error, match="UNAUTHORIZED"):
        play(task=_task(), identity={}, timeout_s=1)


def test_the_seam_checks_BEFORE_anything_effectful():
    """Order, by AST: the gate and the boundary come before the toolchain, the
    compile, the evaluator and the seed -- so a refusal costs nothing."""
    lines = _call_lines(inspect.getsource(RUN._production_play), "play")
    for guard in ("check_gate", "_assert_the_production_acts_are_inert"):
        assert guard in lines, f"{guard} is not called in the seam at all"
    for effect in ("verified_paths", "_default_compile", "_default_load_evaluator"):
        assert effect in lines, effect
        assert lines["check_gate"] < lines[effect], f"gate after {effect}"
        assert lines["_assert_the_production_acts_are_inert"] < lines[effect], (
            f"boundary after {effect}")


def test_the_gate_is_checked_for_EVERY_GAME_not_only_the_first():
    """A seam already constructed must not keep playing if the gate closes."""
    src = inspect.getsource(RUN._production_play)
    assert src.count("check_gate()") >= 2, src.count("check_gate()")


# ─────────── 2. the production boundary is INERT under a test runner ─────────

def test_THE_BOUNDARY_REFUSES_INSIDE_A_TEST_PROCESS_even_with_the_gate_OPEN(monkeypatch):
    """🔑 THE CONTAINMENT ITSELF, and the reason a second gate check is not enough:
    a control can delete two checks as easily as one.

    The gate is forced OPEN here -- the exact state the incident created -- and the
    boundary must STILL refuse, because `pytest` is loaded in this interpreter.
    """
    monkeypatch.setattr(RUN, "H2_EXECUTION_AUTHORIZED", True)
    RUN.check_gate()                                    # the gate no longer refuses
    play = RUN._production_play("/tmp/h2-never-written")
    with pytest.raises(RUN.H2ContainmentError, match="loaded"):
        play(task=_task(), identity={}, timeout_s=1)


def test_the_boundary_refuses_even_with_EVERY_authorization_check_removed(monkeypatch):
    """The incident's exact shape: no gate anywhere. The boundary is what stops it."""
    monkeypatch.setattr(RUN, "H2_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(RUN, "check_gate", lambda: None)     # deleted, as a control would
    play = RUN._production_play("/tmp/h2-never-written")
    with pytest.raises(RUN.H2ContainmentError):
        play(task=_task(), identity={}, timeout_s=1)


def test_the_containment_error_is_a_FAILURE_not_a_verdict():
    """It must be an H2Error subclass -- the wrapper maps those to a refusal exit
    code -- and must never be confused with a VOID, which describes a real run."""
    assert issubclass(RUN.H2ContainmentError, RUN.H2Error)
    assert not issubclass(RUN.H2ContainmentError, RUN.H2VoidError)


def _real_acts():
    from scripts.GPU.alphazero import d1_probe as D1
    from scripts.GPU.alphazero import e4_screen_command as CMD
    from scripts.GPU.alphazero import e4_screen_runner as HARNESS
    from scripts.GPU.alphazero import t1j_toolchain as TC
    return ((TC, "verified_paths"), (D1, "_default_compile"),
            (CMD, "_default_load_evaluator"), (HARNESS, "play_task"))


@pytest.mark.parametrize("module", ["pytest", "_pytest", "unittest"])
def test_every_test_framework_trips_the_boundary(module, monkeypatch):
    from scripts.GPU.alphazero import e4_screen_command as CMD
    fake = {k: v for k, v in sys.modules.items()
            if k not in ("pytest", "_pytest", "unittest")}
    fake[module] = sys.modules.get(module) or object()
    monkeypatch.setattr(sys, "modules", fake)
    with pytest.raises(CMD.ContainmentError, match="verified_paths"):
        CMD.assert_production_acts_are_inert("a test", _real_acts())


def test_the_boundary_does_NOT_refuse_when_no_test_framework_is_loaded(monkeypatch):
    """The negative half: a real run is a fresh subprocess with no test framework, and
    the boundary must let it through -- otherwise H2 could never run at all. Note the
    acts here are all REAL, which is the only interesting case."""
    from scripts.GPU.alphazero import e4_screen_command as CMD
    clean = {k: v for k, v in sys.modules.items()
             if k not in ("pytest", "_pytest", "unittest")}
    monkeypatch.setattr(sys, "modules", clean)
    CMD.assert_production_acts_are_inert("a real run", _real_acts())   # raises nothing


def test_the_boundary_PERMITS_a_test_whose_acts_are_ALL_replaced():
    """🔑 THE OTHER HALF, and the one my first repair got wrong: an honest seam test
    replaces every production act and must be allowed through. Refusing it cost
    eleven passing tests, including the real-builder compatibility tests."""
    from scripts.GPU.alphazero import e4_screen_command as CMD

    class Fake:
        __name__ = "fake_module"
    f = Fake()
    f.verified_paths = lambda: None
    CMD.assert_production_acts_are_inert("a mocked seam", ((f, "verified_paths"),))


def test_the_boundary_NAMES_EVERY_act_that_is_still_real():
    """A boundary that reports only the first real act lets the next one look mocked."""
    from scripts.GPU.alphazero import e4_screen_command as CMD
    with pytest.raises(CMD.ContainmentError) as e:
        CMD.assert_production_acts_are_inert("a test", _real_acts())
    for name in ("verified_paths", "_default_compile", "_default_load_evaluator",
                 "play_task"):
        assert name in str(e.value), name


# ────────────── 3. the gate-removal CONTROL is gone from the harness ─────────

HARNESS_DEFAULT = ("docs/superpowers/evidence/2026-09-12-t1j-harness-containment"
                   "/01_controls.py.txt")
# The override exists ONLY so a negative control can point this read-only inspection
# at a copy carrying a rogue control; the default is the real harness, and
# `test_the_harness_path_DEFAULTS_to_the_evidence_copy` pins it.
HARNESS = pathlib.Path(os.environ.get("H2_CONTAINMENT_HARNESS", HARNESS_DEFAULT))


def test_the_harness_path_DEFAULTS_to_the_evidence_copy():
    assert HARNESS_DEFAULT == ("docs/superpowers/evidence/"
                               "2026-09-12-t1j-harness-containment/01_controls.py.txt")
    assert pathlib.Path(HARNESS_DEFAULT).is_file()


def _defects():
    """The control list, read WITHOUT running the harness -- whose whole job is to
    edit this repository's source files. Only top-level assignments are evaluated,
    and only those containing no call at all: the module's constants (`PROBE`, the
    file paths) resolve, its `main()` and its file writes cannot run.
    """
    ns = {}
    for node in ast.parse(HARNESS.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(n, (ast.Call, ast.Await)) for n in ast.walk(node)):
            continue                       # nothing that could RUN anything
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         str(HARNESS), "exec"), ns)
        except NameError:
            continue                       # depends on a value we refused to compute
        if "DEFECTS" in ns:
            return ns["DEFECTS"]
    raise AssertionError("no literal DEFECTS list in the harness")


def test_NO_CONTROL_DELETES_AN_AUTHORIZATION_CHECK():
    """🔴 The control that caused the incident is REMOVED, not re-aimed. A control
    whose content is "remove the only safeguard" must not exist in a harness that
    can reach production."""
    offenders = {}
    for label, f, old, new, _t in _defects():
        for guard in ("check_gate", "_assert_the_production_acts_are_inert",
                      "assert_production_acts_are_inert"):
            if old.count(guard) > new.count(guard):
                offenders.setdefault(f, []).append(label)

    # H2's own gate-removal control -- the one that caused the incident -- is GONE,
    # not re-aimed: nothing may remove an authorization check from H2's runner.
    assert "scripts/GPU/alphazero/h2_match_runner.py" not in offenders, offenders

    # 🔑 THREE INHERITED H1 CONTROLS legitimately move or delete H1's barriers; that
    # IS the defect they inject. They are admissible ONLY because H1's production
    # seam is INERT to a test process whatever the gate says. Any other file with
    # such a control must earn the same, or the control must go.
    assert set(offenders) <= {"scripts/GPU/alphazero/h1_viability_runner.py"}, offenders
    for f in offenders:
        lines = _seam_lines(f)
        assert "assert_production_acts_are_inert" in lines, (
            f"{f} has gate-removal controls {offenders[f]} but its production seam "
            f"has NO containment boundary -- that is the incident, again")
        for effect in ("verified_paths", "_default_compile",
                       "_default_load_evaluator"):
            assert lines["assert_production_acts_are_inert"] < lines[effect], (
                f"{f}: boundary after {effect}")


def test_the_harness_REFUSES_TO_RUN_ON_IMPORT_OR_EXEC():
    """🔴 INCIDENT 2. The harness's driver is top-level code, so `exec`ing the file to
    READ its control list STARTED it: it injected defects into H1's source for seven
    minutes while other tests ran against the tree it was mutating. The guard must
    reject every namespace shape whose `__name__` is not exactly `__main__` --
    including a namespace with no `__name__` at all."""
    src = HARNESS.read_text()
    assert 'if __name__ != "__main__":' in src
    # and the guard must come BEFORE the driver: no statement that edits a file may
    # precede it.
    mod = ast.parse(src)
    guard_line = min(n.lineno for n in mod.body
                     if isinstance(n, ast.If) and "__main__" in ast.dump(n.test))
    for node in mod.body:
        if node.lineno >= guard_line:
            break
        assert isinstance(node, (ast.Expr, ast.Import, ast.ImportFrom, ast.Assign,
                                 ast.AnnAssign)), (
            f"line {node.lineno}: {type(node).__name__} runs BEFORE the import guard")

    for ns in ({}, {"__name__": "ctl"}, {"__name__": "__main__x"}):
        with pytest.raises(ImportError, match="ONLY as __main__"):
            exec(compile(src, str(HARNESS), "exec"), dict(ns))


def test_the_harness_RESTORES_ON_A_SIGNAL_and_VERIFIES_the_restore():
    """A SIGKILL cannot be caught, but SIGTERM/SIGINT/SIGHUP can -- and the harness
    now verifies every restore and stops rather than running the next control against
    an unverified tree."""
    src = HARNESS.read_text()
    assert "SIGTERM" in src and "SIGINT" in src and "SIGHUP" in src
    assert "RESTORATION FAILED" in src
    assert "raise SystemExit(2)" in src


# ─────────── 4. the entry's own gate check, asserted without executing ───────

def test_run_h2_CALLS_check_gate_FIRST_by_AST():
    """The behavioural test asserts the refusal; this asserts the CALL IS THERE and
    is first -- the property the deleted control used to cover, held now by a test
    that cannot execute anything."""
    fn = ast.parse(inspect.getsource(RUN.run_h2).lstrip()).body[0]
    stmts = [n for n in fn.body if not isinstance(n, ast.Expr)
             or not isinstance(n.value, ast.Constant)]
    first = stmts[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert getattr(first.value.func, "id", "") == "check_gate", ast.dump(first)
