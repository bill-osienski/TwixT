#!/usr/bin/env python3
"""NEGATIVE CONTROLS for the injected-defect harness's own containment tests.

Each control RE-INTRODUCES exactly one of the four gaps found in the harness
packaged at `144a141`, in a COPY, and requires the matching narrow test to FAIL
FOR ITS STATED REASON -- not merely to fail. A test that fails for some other
reason has not been shown to bind the gap it names.

Nothing here edits a live file: the copies live in a temp dir and the tests are
aimed at them through `IDC_DRIVER` / `IDC_DEFECTS`, whose defaults are pinned by
`test_THE_PATHS_DEFAULT_TO_THE_LIVE_HARNESS`. Both live files are hashed before
and after and the run fails if either moved.

Run:  .venv/bin/python scripts/GPU/alphazero/harness_negative_controls.py
"""
import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
PY_BIN = str(ROOT / ".venv/bin/python")
TEST = "tests/test_injected_defect_controls.py"
DRIVER = ROOT / "scripts/GPU/alphazero/run_injected_defect_controls.py"
DEFS = ROOT / "scripts/GPU/alphazero/injected_defect_controls.py"
TMP = pathlib.Path(tempfile.mkdtemp(prefix="harness-negative-controls-"))
results = []


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


LIVE = {DRIVER: sha(DRIVER), DEFS: sha(DEFS)}


def broken(source, old, new, name):
    """A copy of `source` with one gap re-introduced. The anchor must be unique,
    or the control has not run."""
    src = source.read_text()
    if src.count(old) != 1:
        raise SystemExit(f"🔴 {name}: STALE ANCHOR, matched {src.count(old)} times")
    out = TMP / name
    out.write_text(src.replace(old, new))
    return out


def control(label, node, expect, *, driver=None, defects=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("IDC_DRIVER", None)
    env.pop("IDC_DEFECTS", None)
    if driver:
        env["IDC_DRIVER"] = str(driver)
    if defects:
        env["IDC_DEFECTS"] = str(defects)
    r = subprocess.run([PY_BIN, "-m", "pytest", f"{TEST}::{node}", "-q",
                        "-p", "no:cacheprovider", "--no-header"],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    failed = r.returncode != 0
    right_reason = expect in out
    caught = failed and right_reason
    if failed and not right_reason:
        label += f"  [FAILED FOR THE WRONG REASON: {expect!r} absent]"
    detail = next((l.strip() for l in out.splitlines() if expect in l),
                  (out.strip().splitlines() or [""])[-1])
    results.append((label, caught, detail[:200]))
    print(f"{'CAUGHT    ' if caught else 'NOT CAUGHT'} {label}", flush=True)


# ═════════ GAP 1: a failed baseline must ABORT, and problems must EXIT NONZERO

control("NC1 the baseline failure prints but does not abort (the 2026-09-12 run 1 shape)",
        "test_a_FAILED_BASELINE_ABORTS_BEFORE_THE_FIRST_INJECTION",
        "A CONTROL RAN AFTER THE BASELINE FAILED",
        driver=broken(DRIVER,
                      '        print(f"clean baseline over {len(nodes)} distinct target tests: FAIL")\n'
                      '        print("ABORTED before the first injection.")\n'
                      '        discard()\n'
                      '        return BASELINE_FAILED\n',
                      '        print(f"clean baseline over {len(nodes)} distinct target tests: FAIL")\n',
                      "nc1_baseline_does_not_abort.py"))

control("NC2 a reported problem leaves the exit status at 0",
        "test_A_NOT_CAUGHT_DEFECT_EXITS_NONZERO",
        "A NOT CAUGHT DEFECT EXITED 0",
        driver=broken(DRIVER,
                      "    return OK if problems == 0 else PROBLEMS\n",
                      "    return OK\n",
                      "nc2_problems_exit_zero.py"))

# ═════════ GAP 2: only the INTENDED failure, for its stated REASON, is a pass

control("NC3 any nonzero pytest exit counts as a rejection",
        "test_A_COLLECTION_ERROR_IS_INDETERMINATE_NOT_REJECTED",
        "A COLLECTION ERROR WAS NOT REPORTED AS INDETERMINATE",
        driver=broken(DRIVER,
                      '    if rc != 1:\n'
                      '        return "INDETERMINATE", f"pytest exited {rc}, which is not a test failure"\n',
                      '    if rc != 0:\n'
                      '        return "REJECTED", "nonzero exit"\n',
                      "nc3_any_nonzero_is_rejection.py"))

control("NC4 a control with no declared reason is allowed to pass anyway",
        "test_A_CONTROL_WITH_NO_EXPECTED_REASON_IS_INDETERMINATE",
        "A CONTROL WITH NO DECLARED REASON WAS NOT INDETERMINATE",
        driver=broken(DRIVER,
                      '    if expected is None:\n'
                      '        return "INDETERMINATE", "the control declares no expected reason"\n',
                      '    if expected is None:\n'
                      '        expected = ""\n',
                      "nc4_missing_reason_defaults_open.py"))

control("NC5 the declared reason is never actually looked for",
        "test_A_FAILURE_FOR_THE_WRONG_REASON_IS_INDETERMINATE",
        "A FAILURE AT A DIFFERENT ASSERTION SCORED AS A REJECTION",
        driver=broken(DRIVER,
                      '    if expected not in "\\n".join(evidence):\n',
                      "    if False:\n",
                      "nc5_reason_never_checked.py"))

# ═════════ GAP 3: the mutations happen in a DISPOSABLE CHECKOUT, not the tree

control("NC6 the injections go back into the shared working tree",
        "test_THE_SHARED_WORKING_TREE_IS_NEVER_WRITTEN",
        "THE SHARED WORKING TREE WAS WRITTEN DURING THE RUN",
        driver=broken(DRIVER,
                      '    checkout = _CHECKOUT = pathlib.Path(holder) / "checkout"\n'
                      '    git(repo, "worktree", "add", "--detach", str(checkout), "HEAD")\n',
                      '    checkout = _CHECKOUT = repo\n',
                      "nc6_mutates_the_shared_tree.py"))

control("NC7 the start-up snapshot is written back over a concurrent edit",
        "test_A_CONCURRENT_EDIT_IS_NOT_OVERWRITTEN_and_the_run_STOPS",
        "A CONCURRENT EDIT DID NOT STOP THE RUN",
        driver=broken(DRIVER,
                      '            if f.read_text() != injected:\n'
                      '                print(f"🔴 {path} does not hold the content this driver injected -- "\n'
                      '                      f"something else wrote it. STOPPING rather than overwrite it.")\n'
                      '                print(f"checkout PRESERVED: {checkout}")\n'
                      '                _PRESERVE = True\n'
                      '                return TREE_VIOLATED\n',
                      '',
                      "nc7_restores_unconditionally.py"))

# ⚠ leaves one orphaned pytest sleeping up to 30 s -- which is the defect.
control("NC8 a signal unwinds the driver and ORPHANS the pytest subprocess",
        "test_A_SIGNAL_STOPS_THE_TEST_SUBPROCESS_AND_KEEPS_THE_CHECKOUT",
        "is STILL RUNNING against the injected",
        driver=broken(DRIVER,
                      "    if _CHILD is not None:\n"
                      "        _CHILD.terminate()\n"
                      "        try:\n"
                      "            _CHILD.wait(timeout=10)\n"
                      "        except subprocess.TimeoutExpired:\n"
                      "            _CHILD.kill()\n"
                      "            _CHILD.wait()\n",
                      "",
                      "nc8_orphans_the_child.py"))

# ═════════ GAP 4: reading the control list cannot start a run

# 🔑 The rogue copy carries INCIDENT 2's EXACT guard and a single top-level write --
# not the real driver. Re-running the real harness by `exec` is the incident, and is
# not reproduced; its shape is.
_ROGUE = (DEFS.read_text()
          + '\n\nif __name__ != "__main__":\n'
            '    raise ImportError("runs ONLY as __main__")\n'
            'import pathlib as _p\n'
            '_p.Path("src.py").write_text(\'VALUE = "started by exec"\\n\')\n')
_rogue_path = TMP / "nc9_control_list_with_a_driver.py"
_rogue_path.write_text(_ROGUE)

control("NC9 a driver put back into the control list, behind incident 2's own guard",
        "test_EXECING_THE_CONTROL_LIST_AS___main___RUNS_NOTHING",
        "EXECING THE CONTROL LIST AS __main__ WROTE TO A SOURCE FILE",
        defects=_rogue_path)

control("NC9b the same driver, caught structurally instead of behaviourally",
        "test_THE_CONTROL_LIST_HAS_NO_DRIVER_IN_IT",
        "a top-level If in the control list",
        defects=_rogue_path)

control("NC10 the driver runs without being asked to",
        "test_EXECING_THE_DRIVER_AS___main___WITHOUT_THE_RUN_FLAG_REFUSES",
        "THE DRIVER RAN WITHOUT --run",
        driver=broken(DRIVER,
                      "    if not a.run:\n",
                      "    if False:\n",
                      "nc10_runs_without_the_flag.py"))

control("NC14 a fixture-setup ERROR counts as a rejection",
        "test_A_FIXTURE_SETUP_ERROR_IS_INDETERMINATE_NOT_REJECTED",
        "A FIXTURE SETUP ERROR WAS NOT REPORTED AS INDETERMINATE",
        driver=broken(DRIVER,
                      '    failed = failed_lines(node, out)\n'
                      '    if not failed:\n'
                      '        return "INDETERMINATE", f"{node} is not in the FAILED summary"\n'
                      '    if any(l.startswith("ERROR") for l in out.splitlines()):\n'
                      '        return "INDETERMINATE", "pytest reported an ERROR as well as the failure"\n',
                      '    failed = failed_lines(node, out) or ["FAILED (assumed)"]\n',
                      "nc14_error_counts_as_rejection.py"))

control("NC13 the reason is taken from the FITTED summary line, not the traceback",
        "test_THE_RECORDED_REASON_IS_THE_WHOLE_ASSERTION_TEXT",
        "assert {} == {",        # a long node id gets NO reason appended at all
        driver=broken(DRIVER,
                      '                ev = evidence_lines(out)\n',
                      '                ev = [l.split(" - ", 1)[1] for l in failed_lines(node, out)\n'
                      '                      if " - " in l]\n',
                      "nc13_reason_from_the_summary.py"))

# ═════════ THE OTHER DIRECTION: a check nothing can satisfy is an outage, not
# containment. The 2026-09-12 lesson from the H2 boundary, applied here.

control("NC11 nothing can ever be REJECTED, so a clean run never exits 0",
        "test_A_RUN_WITH_NO_PROBLEMS_EXITS_ZERO",
        "A RUN WITH NOTHING WRONG DID NOT EXIT ZERO",
        driver=broken(DRIVER,
                      '    return "REJECTED", observed[:200]\n',
                      '    return "INDETERMINATE", observed[:200]\n',
                      "nc11_nothing_can_pass.py"))


moved = [str(p) for p, h in LIVE.items() if sha(p) != h]
print(f"\n=== negative controls: {sum(1 for _, c, _ in results if c)}/{len(results)} caught ===")
for label, c, detail in results:
    print(f"  [{'x' if c else ' '}] {label}\n      {detail}")
print("live files unchanged:", not moved, moved or "")
raise SystemExit(0 if all(c for _, c, _ in results) and not moved else 1)
