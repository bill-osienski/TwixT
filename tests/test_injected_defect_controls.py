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
DRIVER_DEFAULT = "scripts/GPU/alphazero/run_injected_defect_controls.py"
DEFS_DEFAULT = "scripts/GPU/alphazero/injected_defect_controls.py"
# The overrides exist ONLY so a negative control can re-introduce a gap in a COPY
# and require the matching test below to fail. The defaults are the live files and
# `test_THE_PATHS_DEFAULT_TO_THE_LIVE_HARNESS` pins them.
DRIVER = pathlib.Path(os.environ.get("IDC_DRIVER", str(ROOT / DRIVER_DEFAULT)))
DEFS = pathlib.Path(os.environ.get("IDC_DEFECTS", str(ROOT / DEFS_DEFAULT)))

# the driver's exit statuses, which are part of its contract
OK, REFUSED, BASELINE_FAILED, PROBLEMS, TREE_VIOLATED = 0, 2, 3, 4, 5

TARGET_TEST = '''\
import src


def test_value():
    assert src.VALUE != "", "THE VALUE IS EMPTY"
    assert src.VALUE == "good", "THE VALUE IS WRONG"
'''

SLOW_TEST = '''\
import os
import pathlib
import time


def test_slow():
    if 'VALUE = "bad"' in pathlib.Path("src.py").read_text():
        pathlib.Path("SLEEPING").write_text(str(os.getpid()))
        time.sleep(30)
'''

LONG_TEST = '''\
import src

LONG = "THE VALUE IS WRONG AND THIS MESSAGE IS DELIBERATELY LONGER THAN EIGHTY COLUMNS SO THAT A TRUNCATED SUMMARY LINE CANNOT CARRY IT"


def test_a_deliberately_long_name_that_pushes_the_summary_line_past_eighty_columns():
    assert src.VALUE == "good", LONG
'''

FIXTURE_TEST = '''\
import pytest

import src


@pytest.fixture
def loaded():
    if src.VALUE != "good":
        raise RuntimeError("THE FIXTURE COULD NOT BUILD ITS STATE")
    return src.VALUE


def test_with_a_fixture(loaded):
    assert loaded == "good", "THE VALUE IS WRONG"
'''

VOLATILE_TEST = '''\
import src


def test_volatile(tmp_path):
    """Its failure message carries a per-run tmp path AND an object address --
    the two shapes that made frozen reasons unmatchable on a second run."""
    assert src.VALUE == "good", f"WRONG under {tmp_path} for {object()!r}"
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
    (repo / "tests" / "test_fixture.py").write_text(FIXTURE_TEST)
    (repo / "tests" / "test_volatile.py").write_text(VOLATILE_TEST)
    (repo / "tests" / "test_long.py").write_text(LONG_TEST)
    (repo / "conftest.py").write_text(
        "import pathlib, sys\n"
        "sys.path.insert(0, str(pathlib.Path(__file__).parent))\n")
    # Slow ONLY while the defect is held, so the clean baseline stays instant --
    # and it publishes ITS OWN PID, which is the only honest way to ask later
    # whether the driver really stopped it or merely orphaned it.
    (repo / "tests" / "test_slow.py").write_text(SLOW_TEST)
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

def test_THE_PATHS_DEFAULT_TO_THE_LIVE_HARNESS():
    """The override is for negative controls only; the default is what will run."""
    assert DRIVER_DEFAULT == "scripts/GPU/alphazero/run_injected_defect_controls.py"
    assert DEFS_DEFAULT == "scripts/GPU/alphazero/injected_defect_controls.py"
    assert (ROOT / DRIVER_DEFAULT).is_file() and (ROOT / DEFS_DEFAULT).is_file()


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

    assert "baseline" in r.stdout.lower() and "FAIL" in r.stdout
    # 🔑 EVERY outcome word, not just the two that mean "pass"/"fail". With the
    # checkout in place a driver that ignores its baseline no longer touches the
    # shared tree, so mtime cannot see it -- what it still does is REPORT on
    # controls whose targets were never shown to work, in any of the four shapes.
    for marker in ("REJECTED", "NOT CAUGHT", "INDETERMINATE", "STALE CONTROL"):
        assert marker not in r.stdout, (
            f"A CONTROL RAN AFTER THE BASELINE FAILED ({marker})")
    assert "ABORTED before the first injection." in r.stdout, (
        "THE DRIVER DID NOT SAY IT ABORTED")
    assert (_sha(sandbox / "src.py"),
            (sandbox / "src.py").stat().st_mtime_ns) == before, (
        "A DEFECT WAS INJECTED AFTER THE BASELINE FAILED")
    assert r.returncode == BASELINE_FAILED, (
        f"A FAILED BASELINE EXITED {r.returncode}, NOT {BASELINE_FAILED}")


def test_A_NOT_CAUGHT_DEFECT_EXITS_NONZERO(sandbox):
    """A defect nothing rejects is the harness's whole reason to exist, and it
    must be visible to a shell -- not only to a reader of the summary."""
    ctl = [("nothing checks the comment", "src.py",
            'VALUE = "good"', 'VALUE = "good"  # cosmetic',
            "tests/test_target.py::test_value")]
    r = drive(sandbox, write_defects(sandbox, ctl, {"nothing checks the comment": "x"}))
    assert "NOT CAUGHT" in r.stdout, r.stdout
    assert r.returncode == PROBLEMS, (
        f"A NOT CAUGHT DEFECT EXITED {r.returncode}, NOT {PROBLEMS}")


def test_A_STALE_CONTROL_EXITS_NONZERO(sandbox):
    """A control whose anchor no longer matches HAS NOT RUN. Printing that and
    returning 0 is the same lie as counting it."""
    ctl = [("anchor no longer present", "src.py",
            'VALUE = "vanished"', 'VALUE = "bad"',
            "tests/test_target.py::test_value")]
    r = drive(sandbox, write_defects(sandbox, ctl, {"anchor no longer present": "x"}))
    assert "STALE" in r.stdout, r.stdout
    assert r.returncode == PROBLEMS, (
        f"A STALE CONTROL EXITED {r.returncode}, NOT {PROBLEMS}")


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
    assert r.returncode == OK, (
        f"A RUN WITH NOTHING WRONG DID NOT EXIT ZERO (exit {r.returncode})\n"
        + r.stdout)
    assert "REJECTED" in r.stdout


# ═══════════ GAP 2: only the INTENDED failure, for its stated REASON, is a pass

def test_A_COLLECTION_ERROR_IS_INDETERMINATE_NOT_REJECTED(sandbox):
    """🔴 `rejected = r.returncode != 0`. A defect that makes the target test file
    unimportable makes pytest ERROR -- the test never ran, so nothing was shown to
    reject anything, and scoring it as a successful control is a false pass."""
    ctl = [("breaks the module outright", "src.py",
            'VALUE = "good"', 'VALUE = (', "tests/test_target.py::test_value")]
    r = drive(sandbox, write_defects(sandbox, ctl, {"breaks the module outright": "THE VALUE IS WRONG"}))
    assert "INDETERMINATE" in r.stdout, (
        "A COLLECTION ERROR WAS NOT REPORTED AS INDETERMINATE:\n" + r.stdout)
    assert "REJECTED" not in r.stdout, (
        "A COLLECTION ERROR SCORED AS A REJECTION:\n" + r.stdout)
    assert r.returncode == PROBLEMS, (r.returncode, r.stdout)


def test_A_FAILURE_FOR_THE_WRONG_REASON_IS_INDETERMINATE(sandbox):
    """The test fails, but at a different assertion than the control claims. The
    control has not been shown to bind what it says it binds."""
    ctl = [("empties the value", "src.py", 'VALUE = "good"', 'VALUE = ""',
            "tests/test_target.py::test_value")]
    r = drive(sandbox, write_defects(sandbox, ctl, {"empties the value": "THE VALUE IS WRONG"}))
    assert "INDETERMINATE" in r.stdout, (
        "A FAILURE AT A DIFFERENT ASSERTION SCORED AS A REJECTION:\n" + r.stdout)
    assert "THE VALUE IS EMPTY" in r.stdout, "the observed reason is not reported"
    assert r.returncode == PROBLEMS, (r.returncode, r.stdout)


def test_A_CONTROL_WITH_NO_EXPECTED_REASON_IS_INDETERMINATE(sandbox):
    """FAIL CLOSED on absence. An expectation that defaults to "anything" is a
    switch-off wearing the name of a check."""
    r = drive(sandbox, write_defects(sandbox, [GOOD], {}))
    assert "INDETERMINATE" in r.stdout, (
        "A CONTROL WITH NO DECLARED REASON WAS NOT INDETERMINATE:\n" + r.stdout)
    assert r.returncode == PROBLEMS, (r.returncode, r.stdout)


def test_AN_ORPHAN_EXPECTED_REASON_EXITS_NONZERO(sandbox):
    """The other direction from a missing reason: one declared for a control that
    no longer exists. It binds nothing and it HIDES a rename -- found by hand on
    2026-09-12, when two controls were re-anchored for the third seed block and
    their old reasons stayed behind, still looking like coverage."""
    r = drive(sandbox, write_defects(
        sandbox, [GOOD], {**GOOD_REASON, "a control that was renamed away": "x"}))
    assert "ORPHAN REASON" in r.stdout, r.stdout
    assert r.returncode == PROBLEMS, (
        f"AN ORPHAN REASON EXITED {r.returncode}, NOT {PROBLEMS}")


def test_EVERY_REAL_CONTROL_DECLARES_AN_EXPECTED_REASON():
    """...and the real control list is held to it, not only the fixtures."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_idc_probe", DEFS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    missing = [lab for lab, *_ in mod.DEFECTS if lab not in mod.EXPECTED_REASONS]
    assert not missing, missing
    orphans = [k for k in mod.EXPECTED_REASONS
               if k not in {lab for lab, *_ in mod.DEFECTS}]
    assert not orphans, orphans
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
        "THE SHARED WORKING TREE WAS WRITTEN DURING THE RUN")


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

    assert r.returncode == TREE_VIOLATED, (
        f"A CONCURRENT EDIT DID NOT STOP THE RUN (exit {r.returncode})\n" + r.stdout)
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
    assert (src.read_bytes(), src.stat().st_mtime_ns) == before, (
        "EXECING THE CONTROL LIST AS __main__ WROTE TO A SOURCE FILE")


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
    assert r.returncode == REFUSED, (
        f"THE DRIVER RAN WITHOUT --run (exit {r.returncode})\n{r.stdout}{r.stderr}")
    assert "--run" in (r.stdout + r.stderr)
    assert (src.read_bytes(), src.stat().st_mtime_ns) == before


def test_A_SIGNAL_STOPS_THE_TEST_SUBPROCESS_AND_KEEPS_THE_CHECKOUT(sandbox):
    """"Manage child-process shutdown explicitly."

    🔑 THE OBSERVABLE IS THE CHILD'S PID, NOT THE PARENT'S EXIT TIME. My first
    version signalled as soon as the injection appeared and asserted the driver
    exited quickly -- which it does either way, because a parent that unwinds
    simply ORPHANS its child. It passed in 0.007 s having proved nothing. The
    driver spends the run blocked on a pytest subprocess holding an injected tree;
    what must be true is that THAT PROCESS IS GONE.
    """
    import signal as _sig
    import time as _t

    ctl = [("a control with a slow test", "src.py", 'VALUE = "good"', 'VALUE = "bad"',
            "tests/test_slow.py::test_slow")]
    defects = write_defects(sandbox, ctl, {"a control with a slow test": "irrelevant"})
    proc = subprocess.Popen(
        [sys.executable, str(DRIVER), "--run", "--defects", str(defects)],
        cwd=sandbox, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})

    marker, deadline = None, _t.time() + 60
    while _t.time() < deadline:               # wait until the child is IN the sleep
        listing = subprocess.run(["git", "worktree", "list"], cwd=sandbox,
                                 capture_output=True, text=True).stdout
        if listing.strip().count("\n") >= 1:
            cand = pathlib.Path(listing.splitlines()[1].split()[0]) / "SLEEPING"
            if cand.exists() and cand.read_text().strip():
                marker = cand
                break
        _t.sleep(0.05)
    if marker is None:
        proc.kill()
        pytest.fail("the child never reached the sleep, so nothing was signalled")
    child_pid = int(marker.read_text().strip())

    proc.send_signal(_sig.SIGTERM)
    out, err = proc.communicate(timeout=25)

    assert proc.returncode == 130, (proc.returncode, out, err)
    assert "SIGNAL" in out, out
    gone, until = False, _t.time() + 15
    while _t.time() < until:
        try:
            os.kill(child_pid, 0)
        except (ProcessLookupError, PermissionError):
            gone = True
            break
        _t.sleep(0.1)
    assert gone, (f"pytest {child_pid} is STILL RUNNING against the injected "
                  f"checkout -- the driver unwound and orphaned it")
    line = next((l for l in out.splitlines() if l.startswith("checkout PRESERVED: ")), None)
    assert line, out
    kept = pathlib.Path(line.split(": ", 1)[1].strip()) / "src.py"
    assert kept.read_text().strip() == 'VALUE = "bad"', (
        "the checkout was discarded after being announced as preserved")


LONG_NODE = ("tests/test_long.py::"
             "test_a_deliberately_long_name_that_pushes_the_summary_line_past_eighty_columns")
LONG_REASON = ("THE VALUE IS WRONG AND THIS MESSAGE IS DELIBERATELY LONGER THAN EIGHTY "
               "COLUMNS SO THAT A TRUNCATED SUMMARY LINE CANNOT CARRY IT")


def test_THE_REASON_IS_READ_FROM_UNTRUNCATED_EVIDENCE(sandbox):
    """🔴 pytest fits its `FAILED <node> - <reason>` line to the terminal -- 80
    columns when stdout is a pipe -- and when the node id alone is already too long
    it appends NO reason at all. That is most of this programme's test names, which
    is why the first recording run on 2026-09-12 harvested 31 reasons out of 539,
    all of them stubs. The reason must come from the traceback, which is not fitted
    to anything, so a message longer than the line still arrives whole."""
    ctl = [("a long name and a long message", "src.py",
            'VALUE = "good"', 'VALUE = "bad"', LONG_NODE)]
    r = drive(sandbox, write_defects(sandbox, ctl,
                                     {"a long name and a long message": LONG_REASON}))
    assert "REJECTED" in r.stdout, (
        "A LONG NODE ID OR MESSAGE WAS LOST TO TRUNCATION:\n" + r.stdout)
    assert r.returncode == OK, (r.returncode, r.stdout)


def test_THE_RECORDED_REASON_IS_THE_WHOLE_ASSERTION_TEXT(sandbox, tmp_path):
    """--record is how EXPECTED_REASONS gets populated, so a truncated recording
    would freeze a stub as the thing every later run compares against."""
    ctl = [("a long name and a long message", "src.py",
            'VALUE = "good"', 'VALUE = "bad"', LONG_NODE)]
    out = tmp_path / "observed.json"
    drive(sandbox, write_defects(sandbox, ctl, {"a long name and a long message": LONG_REASON}),
          "--record", str(out))
    import json
    recorded = json.loads(out.read_text())
    # the exception TYPE is part of the reason -- "AssertionError: ..." pins more
    # than the message alone, and the whole message must survive
    assert recorded == {
        "a long name and a long message": "AssertionError: " + LONG_REASON}, recorded


def test_A_FIXTURE_SETUP_ERROR_IS_INDETERMINATE_NOT_REJECTED(sandbox):
    """🔴 THE SHAPE THAT WAS ACTUALLY THERE. Of the 539 real controls, six make the
    target test ERROR rather than fail -- five at fixture setup or import, one by
    leaving a SyntaxError in the module under test. pytest still exits nonzero, so
    `rejected = r.returncode != 0` counted all six as successful controls. The named
    test never ran, so nothing was shown to reject anything.

    Note this errors with rc == 1 and an ERROR summary rather than a FAILED one --
    an exit code alone cannot tell it from a real failure.
    """
    ctl = [("breaks the fixture, not the assertion", "src.py",
            'VALUE = "good"', 'VALUE = "bad"',
            "tests/test_fixture.py::test_with_a_fixture")]
    # 🔑 the control DECLARES THE ERROR'S OWN TEXT, so the reason check cannot be
    # what rejects it. What must reject it is that the named test never ran.
    r = drive(sandbox, write_defects(
        sandbox, ctl,
        {"breaks the fixture, not the assertion": "THE FIXTURE COULD NOT BUILD ITS STATE"}))
    assert "INDETERMINATE" in r.stdout, (
        "A FIXTURE SETUP ERROR WAS NOT REPORTED AS INDETERMINATE:\n" + r.stdout)
    assert "REJECTED" not in r.stdout, (
        "A FIXTURE SETUP ERROR SCORED AS A REJECTION:\n" + r.stdout)
    assert r.returncode == PROBLEMS, (r.returncode, r.stdout)


def test_A_REASON_RECORDED_ON_ONE_RUN_MATCHES_ON_THE_NEXT(sandbox, tmp_path):
    """🔴 A reason that cannot match twice is not a check. The frozen reasons are
    compared as text, and pytest's diagnostics carry content that changes every run
    -- the numbered tmp session, object repr addresses, and this driver's own
    randomly named checkout. Ten of the 539 real controls drifted on exactly those
    when the frozen reasons were first measured, reported INDETERMINATE, and none
    of the ten had anything wrong with it.

    Records on one run and requires the recorded value to hold on the NEXT one,
    which is the only way to ask this question honestly.
    """
    import json
    ctl = [("a message with per-run content", "src.py",
            'VALUE = "good"', 'VALUE = "bad"', "tests/test_volatile.py::test_volatile")]
    out = tmp_path / "observed.json"
    drive(sandbox, write_defects(sandbox, ctl, {}), "--record", str(out))
    recorded = json.loads(out.read_text())
    assert recorded, "nothing was recorded to re-check"

    r = drive(sandbox, write_defects(sandbox, ctl, recorded))
    assert "REJECTED" in r.stdout, (
        "A REASON RECORDED ON ONE RUN DID NOT MATCH ON THE NEXT:\n"
        + f"recorded: {recorded}\n" + r.stdout)
    assert r.returncode == OK, (r.returncode, r.stdout)


def test_EVERY_CONTROLS_ANCHOR_MATCHES_ITS_SOURCE_EXACTLY_ONCE():
    """🔴 TWO CONTROLS WENT STALE AND THE SUITE COULD NOT SEE IT.

    Editing the wrapper changed two lines that existing controls anchor on, and
    only the HARNESS noticed -- twenty minutes later. My pre-check looked at the
    controls needing REASONS, which are the new ones, and a control that already
    has a reason is exactly the one whose anchor has had time to rot.

    A zero-match anchor means the control DID NOT RUN. A multi-match anchor means
    it injected somewhere ambiguous. Both are checked here, for every control, on
    every suite run.
    """
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("_idc_anchor", DEFS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    offenders = []
    for label, rel, old, _new, _node in mod.DEFECTS:
        path = pathlib.Path(rel)
        if not path.exists():
            offenders.append((label, rel, "FILE MISSING"))
            continue
        n = path.read_text().count(old)
        if n != 1:
            offenders.append((label, rel, f"{n} matches"))
    assert offenders == [], offenders


def test_EVERY_CONTROLS_TARGET_TEST_EXISTS():
    """A control naming a test that no longer exists scores REJECTED FOR FREE:
    pytest exits non-zero for a node id it cannot find. The clean-baseline check
    catches it at run time; this catches it at edit time."""
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("_idc_target", DEFS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    missing = []
    for label, _rel, _old, _new, node in mod.DEFECTS:
        f, _, t = node.partition("::")
        t = t.split("[")[0]
        if f.startswith("tests/") and pathlib.Path(f).exists():
            if f"def {t}(" not in pathlib.Path(f).read_text():
                missing.append((label, node))
    assert missing == [], missing
