"""THE INJECTED-DEFECT HARNESS'S OWN CONTAINMENT -- the four gaps found by
reviewing `144a141`'s packaged harness on 2026-09-12.

The harness edits this repository's source files and runs its test suite against
them. Four of its own behaviours made its verdict unsound:

  1. a FAILED CLEAN BASELINE did not stop it, and it exited 0 anyway -- run 1 on
     2026-09-12 printed `clean baseline: FAIL` and `538/538 defects rejected`
     and returned 0 (preserved as `08_harness_run_FAILED_BASELINE.txt`);
  2. ANY nonzero pytest exit scored as REJECTED, so collection errors, crashes,
     interruptions and "no tests ran" were counted as successful controls;
  3. it MUTATED THE SHARED WORKING TREE and restored snapshots taken at start-up
     UNCONDITIONALLY, so a concurrent edit was silently overwritten -- the
     mechanism of incident 2, which wiped a containment guard added mid-run;
  4. its import guard tested `__name__ != "__main__"`, which does NOT stop an
     `exec` in a namespace where `__name__` is already `"__main__"` -- the way
     incident 2 actually started it.

Every test here runs the driver as a FRESH SUBPROCESS against a THROWAWAY GIT
REPOSITORY built in a temp dir, because an exit status can only be qualified
from outside the process that produces it, and because a harness that is asked
whether it damages a working tree must be given one it is allowed to damage.

Nothing here touches the T1j programme: no gate, no seed, no model, no JVM.
"""
import hashlib
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DRIVER = ROOT / "scripts/GPU/alphazero/run_injected_defect_controls.py"
DEFS = ROOT / "scripts/GPU/alphazero/injected_defect_controls.py"

# the driver's exit statuses, which are part of its contract
OK, REFUSED, BASELINE_FAILED, PROBLEMS, TREE_VIOLATED = 0, 2, 3, 4, 5

TARGET_TEST = '''\
import src


def test_value():
    assert src.VALUE != "", "THE VALUE IS EMPTY"
    assert src.VALUE == "good", "THE VALUE IS WRONG"
'''

MEDDLE_TEST = '''\
import pathlib


def test_meddles_with_the_source_under_test():
    """Stands in for a concurrent edit arriving while a control HOLDS the file.

    Only while the defect is injected -- during the clean baseline this test must
    do nothing, or the anchor would be gone before the control ever runs.
    """
    p = pathlib.Path("src.py")
    if 'VALUE = "bad"' in p.read_text():
        p.write_text("VALUE = 'edited by someone else'\\n")
'''


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


@pytest.fixture
def sandbox(tmp_path):
    """A throwaway git repository with one source file and one test over it."""
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "src.py").write_text('VALUE = "good"\n')
    (repo / "tests" / "test_target.py").write_text(TARGET_TEST)
    (repo / "tests" / "test_meddle.py").write_text(MEDDLE_TEST)
    (repo / "conftest.py").write_text(
        "import pathlib, sys\n"
        "sys.path.insert(0, str(pathlib.Path(__file__).parent))\n")
    (repo / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    for cmd in (["init", "-q", "-b", "main"], ["add", "-A"],
                ["-c", "user.email=t@t", "-c", "user.name=t",
                 "commit", "-qm", "sandbox"]):
        subprocess.run(["git", *cmd], cwd=repo, check=True,
                       capture_output=True, text=True)
    return repo


def write_defects(repo, controls, reasons):
    """A defects module for the driver to load BY PATH, outside the sandbox repo
    so that adding it cannot itself dirty the tree under test."""
    p = repo.parent / "defects_under_test.py"
    p.write_text("DEFECTS = %r\nEXPECTED_REASONS = %r\n" % (controls, reasons))
    return p


def drive(repo, defects, *extra, run_flag=True):
    cmd = [sys.executable, str(DRIVER)]
    if run_flag:
        cmd.append("--run")
    cmd += ["--defects", str(defects), *extra]
    return subprocess.run(cmd, cwd=repo, capture_output=True, text=True,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


GOOD = ("the value is wrong", "src.py", 'VALUE = "good"', 'VALUE = "bad"',
        "tests/test_target.py::test_value")
GOOD_REASON = {"the value is wrong": "THE VALUE IS WRONG"}


# ═══════════════ GAP 1: a failed baseline must ABORT, and problems must EXIT NONZERO

def test_a_FAILED_BASELINE_ABORTS_BEFORE_THE_FIRST_INJECTION(sandbox):
    """🔴 THE WORKED EXAMPLE, 2026-09-12 run 1: two controls named tests that had
    been renamed, pytest exits nonzero for a node id it cannot find, and those
    controls scored REJECTED FOR FREE while the driver printed `clean baseline:
    FAIL` and returned 0. Here the same shape -- a target node that does not
    exist -- must stop the driver before it writes anything."""
    ctl = [("names a test that does not exist", "src.py",
            'VALUE = "good"', 'VALUE = "bad"',
            "tests/test_target.py::test_no_such_test")]
    before = (_sha(sandbox / "src.py"), (sandbox / "src.py").stat().st_mtime_ns)

    r = drive(sandbox, write_defects(sandbox, ctl, {"names a test that does not exist": "x"}))

    assert r.returncode == BASELINE_FAILED, (r.returncode, r.stdout, r.stderr)
    assert "baseline" in r.stdout.lower() and "FAIL" in r.stdout
    assert "REJECTED" not in r.stdout and "NOT CAUGHT" not in r.stdout, (
        "a control ran after the baseline failed: " + r.stdout)
    assert (_sha(sandbox / "src.py"),
            (sandbox / "src.py").stat().st_mtime_ns) == before, (
        "the driver injected a defect after its baseline failed")


def test_A_NOT_CAUGHT_DEFECT_EXITS_NONZERO(sandbox):
    """A defect nothing rejects is the harness's whole reason to exist, and it
    must be visible to a shell -- not only to a reader of the summary."""
    ctl = [("nothing checks the comment", "src.py",
            'VALUE = "good"', 'VALUE = "good"  # cosmetic',
            "tests/test_target.py::test_value")]
    r = drive(sandbox, write_defects(sandbox, ctl, {"nothing checks the comment": "x"}))
    assert "NOT CAUGHT" in r.stdout, r.stdout
    assert r.returncode == PROBLEMS, (r.returncode, r.stdout)


def test_A_STALE_CONTROL_EXITS_NONZERO(sandbox):
    """A control whose anchor no longer matches HAS NOT RUN. Printing that and
    returning 0 is the same lie as counting it."""
    ctl = [("anchor no longer present", "src.py",
            'VALUE = "vanished"', 'VALUE = "bad"',
            "tests/test_target.py::test_value")]
    r = drive(sandbox, write_defects(sandbox, ctl, {"anchor no longer present": "x"}))
    assert "STALE" in r.stdout, r.stdout
    assert r.returncode == PROBLEMS, (r.returncode, r.stdout)


def test_A_DUPLICATE_INJECTION_EXITS_NONZERO(sandbox):
    """Two labels over one injection report one experiment as two."""
    ctl = [GOOD, ("the same injection under another name", *GOOD[1:])]
    r = drive(sandbox, write_defects(
        sandbox, ctl, {**GOOD_REASON,
                       "the same injection under another name": "THE VALUE IS WRONG"}))
    assert "DUPLICATE" in r.stdout, r.stdout
    assert r.returncode == PROBLEMS, (r.returncode, r.stdout)


def test_A_RUN_WITH_NO_PROBLEMS_EXITS_ZERO(sandbox):
    """The other half. A driver that always exits nonzero has no verdict either."""
    r = drive(sandbox, write_defects(sandbox, [GOOD], GOOD_REASON))
    assert r.returncode == OK, (r.returncode, r.stdout, r.stderr)
    assert "REJECTED" in r.stdout


# ═══════════ GAP 2: only the INTENDED failure, for its stated REASON, is a pass

def test_A_COLLECTION_ERROR_IS_INDETERMINATE_NOT_REJECTED(sandbox):
    """🔴 `rejected = r.returncode != 0`. A defect that makes the target test file
    unimportable makes pytest ERROR -- the test never ran, so nothing was shown to
    reject anything, and scoring it as a successful control is a false pass."""
    ctl = [("breaks the module outright", "src.py",
            'VALUE = "good"', 'VALUE = (', "tests/test_target.py::test_value")]
    r = drive(sandbox, write_defects(sandbox, ctl, {"breaks the module outright": "THE VALUE IS WRONG"}))
    assert "INDETERMINATE" in r.stdout, r.stdout
    assert "REJECTED" not in r.stdout, r.stdout
    assert r.returncode == PROBLEMS, (r.returncode, r.stdout)


def test_A_FAILURE_FOR_THE_WRONG_REASON_IS_INDETERMINATE(sandbox):
    """The test fails, but at a different assertion than the control claims. The
    control has not been shown to bind what it says it binds."""
    ctl = [("empties the value", "src.py", 'VALUE = "good"', 'VALUE = ""',
            "tests/test_target.py::test_value")]
    r = drive(sandbox, write_defects(sandbox, ctl, {"empties the value": "THE VALUE IS WRONG"}))
    assert "INDETERMINATE" in r.stdout, r.stdout
    assert "THE VALUE IS EMPTY" in r.stdout, "the observed reason is not reported"
    assert r.returncode == PROBLEMS, (r.returncode, r.stdout)


def test_A_CONTROL_WITH_NO_EXPECTED_REASON_IS_INDETERMINATE(sandbox):
    """FAIL CLOSED on absence. An expectation that defaults to "anything" is a
    switch-off wearing the name of a check."""
    r = drive(sandbox, write_defects(sandbox, [GOOD], {}))
    assert "INDETERMINATE" in r.stdout, r.stdout
    assert r.returncode == PROBLEMS, (r.returncode, r.stdout)


def test_EVERY_REAL_CONTROL_DECLARES_AN_EXPECTED_REASON():
    """...and the real control list is held to it, not only the fixtures."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_idc_probe", DEFS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    missing = [lab for lab, *_ in mod.DEFECTS if lab not in mod.EXPECTED_REASONS]
    assert not missing, missing
    assert len(mod.DEFECTS) == len({lab for lab, *_ in mod.DEFECTS}), "duplicate labels"


# ═══════════ GAP 3: the mutations happen in a DISPOSABLE CHECKOUT, not the tree

def test_THE_SHARED_WORKING_TREE_IS_NEVER_WRITTEN(sandbox):
    """🔑 mtime, not content. The old harness also ended with the right bytes on
    disk -- it wrote the defect and wrote the original back. An unchanged mtime is
    the only evidence that the shared tree was never the thing being mutated."""
    src = sandbox / "src.py"
    before = (src.read_bytes(), src.stat().st_mtime_ns)
    r = drive(sandbox, write_defects(sandbox, [GOOD], GOOD_REASON))
    assert r.returncode == OK, (r.returncode, r.stdout, r.stderr)
    assert (src.read_bytes(), src.stat().st_mtime_ns) == before, (
        "the shared working tree was written during the run")


def test_A_CONCURRENT_EDIT_IS_NOT_OVERWRITTEN_and_the_run_STOPS(sandbox):
    """🔴 INCIDENT 2's mechanism, reproduced: something else edits the file while a
    control holds it. Restoring a snapshot taken at start-up over that edit is how
    a containment guard added mid-run was wiped. The driver must notice the file no
    longer holds what IT injected, stop, and leave the edit where it can be read."""
    ctl = [("a control whose test edits the source", "src.py",
            'VALUE = "good"', 'VALUE = "bad"',
            "tests/test_meddle.py::test_meddles_with_the_source_under_test")]
    r = drive(sandbox, write_defects(
        sandbox, ctl, {"a control whose test edits the source": "irrelevant"}))

    assert r.returncode == TREE_VIOLATED, (r.returncode, r.stdout, r.stderr)
    assert "does not hold" in r.stdout, r.stdout
    line = next((l for l in r.stdout.splitlines()
                 if l.startswith("checkout PRESERVED: ")), None)
    assert line, "the checkout holding the edit was not named: " + r.stdout
    kept = pathlib.Path(line.split(": ", 1)[1].strip()) / "src.py"
    assert kept.read_text() == "VALUE = 'edited by someone else'\n", (
        "the concurrent edit was overwritten instead of preserved: " + kept.read_text())


def test_THE_DISPOSABLE_CHECKOUT_IS_NAMED_AND_IS_NOT_THE_REPO(sandbox):
    r = drive(sandbox, write_defects(sandbox, [GOOD], GOOD_REASON))
    line = next((l for l in r.stdout.splitlines() if l.startswith("checkout: ")), None)
    assert line, r.stdout
    where = pathlib.Path(line.split("checkout: ", 1)[1].strip()).resolve()
    assert where != sandbox.resolve() and sandbox.resolve() not in where.parents


def test_A_CLEAN_RUN_REMOVES_ITS_CHECKOUT(sandbox):
    r = drive(sandbox, write_defects(sandbox, [GOOD], GOOD_REASON))
    line = next(l for l in r.stdout.splitlines() if l.startswith("checkout: "))
    assert not pathlib.Path(line.split("checkout: ", 1)[1].strip()).exists()
    wt = subprocess.run(["git", "worktree", "list"], cwd=sandbox,
                        capture_output=True, text=True).stdout
    assert wt.strip().count("\n") == 0, wt


def test_A_DIRTY_TREE_IS_REFUSED(sandbox):
    """ponytail: the checkout is `git worktree add HEAD`, so uncommitted work
    would not be under test. Refuse rather than silently test something else."""
    (sandbox / "src.py").write_text('VALUE = "good"  # uncommitted\n')
    r = drive(sandbox, write_defects(sandbox, [GOOD], GOOD_REASON))
    assert r.returncode == REFUSED, (r.returncode, r.stdout, r.stderr)
    assert "clean" in (r.stdout + r.stderr).lower()


# ═══════════ GAP 4: reading the control list cannot start a run

def test_EXECING_THE_CONTROL_LIST_AS___main___RUNS_NOTHING(sandbox):
    """🔴 INCIDENT 2 EXACTLY. `if __name__ != "__main__": raise ImportError` passes
    when `__name__` is ALREADY `"__main__"` -- which is the namespace you get from
    `exec(path.read_text())` in a script. The definitions must contain no driver at
    all, so there is nothing for any namespace shape to start."""
    src = sandbox / "src.py"
    before = (src.read_bytes(), src.stat().st_mtime_ns)
    probe = sandbox.parent / "exec_probe.py"
    probe.write_text(
        "import pathlib\n"
        f"ns = {{'__name__': '__main__', '__file__': {str(DEFS)!r}}}\n"
        f"exec(compile(pathlib.Path({str(DEFS)!r}).read_text(), {str(DEFS)!r}, 'exec'), ns)\n"
        "print('DEFECTS', len(ns['DEFECTS']))\n")
    r = subprocess.run([sys.executable, str(probe)], cwd=sandbox,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("DEFECTS "), r.stdout
    assert int(r.stdout.split()[1]) > 0, "the control list did not come back readable"
    assert (src.read_bytes(), src.stat().st_mtime_ns) == before


def test_THE_CONTROL_LIST_HAS_NO_DRIVER_IN_IT():
    """By structure, not by hope: a module whose body is only definitions cannot
    run anything, whatever namespace it is executed in."""
    import ast
    mod = ast.parse(DEFS.read_text())
    for node in mod.body:
        assert isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign,
                                 ast.AnnAssign, ast.FunctionDef, ast.ClassDef)) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)), (
            f"line {node.lineno}: a top-level {type(node).__name__} in the control list")
    assert "DEFECTS" in DEFS.read_text()


def test_EXECING_THE_DRIVER_AS___main___WITHOUT_THE_RUN_FLAG_REFUSES(sandbox):
    """The driver is not the control list, but it is the other file a reader
    reaches for. Executing it without an explicit request to run must refuse."""
    src = sandbox / "src.py"
    before = (src.read_bytes(), src.stat().st_mtime_ns)
    r = drive(sandbox, write_defects(sandbox, [GOOD], GOOD_REASON), run_flag=False)
    assert r.returncode == REFUSED, (r.returncode, r.stdout, r.stderr)
    assert "--run" in (r.stdout + r.stderr)
    assert (src.read_bytes(), src.stat().st_mtime_ns) == before
