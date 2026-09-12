#!/usr/bin/env python3
"""THE INJECTED-DEFECT DRIVER -- explicitly invoked, and it mutates only a
DISPOSABLE CHECKOUT.

    python scripts/GPU/alphazero/run_injected_defect_controls.py --run

Four behaviours of the previous harness made its verdict unsound; each is
addressed here and each has a narrow test in `tests/test_injected_defect_controls.py`
plus a negative control that re-introduces the gap.

 1. A FAILED CLEAN BASELINE now ABORTS before the first injection, and every
    reported problem sets a nonzero exit status. Run 1 on 2026-09-12 printed
    `clean baseline: FAIL`, `538/538 defects rejected`, and RETURNED 0.
 2. A control is REJECTED only when the named node FAILS -- not errors, not
    "no tests ran", not interrupted -- and its output carries the reason the
    control declares. Anything else is INDETERMINATE: a third outcome, reported
    as such, never folded into the pass count.
 3. The injections happen in a `git worktree` made for the run and removed after
    it. The shared working tree is never written. Where a file IS restored in
    place -- between controls, inside the checkout -- the driver first verifies
    it still holds exactly what IT injected, and STOPS rather than overwrite
    somebody else's edit. That overwrite is how incident 2 wiped a containment
    guard that had been added mid-run.
 4. There is no driver in the control list. `injected_defect_controls.py` is
    data; reading or importing it cannot start anything. This file refuses to do
    anything without `--run`, so `exec`ing it does not start a run either.

EXIT STATUS
    0  every control ran, rejected its defect, and for the stated reason
    2  refused before doing anything (no --run, dirty tree, bad arguments)
    3  the clean baseline failed -- NOTHING was injected
    4  the run finished and reported problems (not caught / stale / duplicate /
       indeterminate)
    5  the checkout was written by something other than this driver; the run
       STOPPED and the checkout is preserved for inspection
"""
import argparse
import importlib.util
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile

OK, REFUSED, BASELINE_FAILED, PROBLEMS, TREE_VIOLATED = 0, 2, 3, 4, 5

_CHILD = None          # the pytest subprocess, so a signal can stop it explicitly
_CHECKOUT = None       # the disposable checkout, so a signal can name it
_PRESERVE = False      # a checkout we announced as kept must not then be discarded


def load_defects(path):
    """Import the control list BY PATH. Safe by construction: that module is data."""
    spec = importlib.util.spec_from_file_location("_injected_defect_controls", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.DEFECTS), dict(getattr(mod, "EXPECTED_REASONS", {}))


def git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args], check=check,
                          capture_output=True, text=True)


def purge(root):
    for d in pathlib.Path(root).rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)


def run_pytest(cwd, nodes):
    """A fresh subprocess, held in a global so `_on_signal` can shut it down."""
    global _CHILD
    _CHILD = subprocess.Popen(
        [sys.executable, "-m", "pytest", *nodes, "-q", "-rf", "--tb=short",
         "-p", "no:cacheprovider"],
        cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        # `--tb=short` explicitly: the evidence this driver reads is the traceback,
        # and it must not depend on what an ini file happens to say about it.
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    out, err = _CHILD.communicate()
    rc, _CHILD = _CHILD.returncode, None
    return rc, out + "\n" + err


def failed_lines(node, out):
    """The short-summary lines for THIS node. `startswith` alone would also match a
    LONGER test name that begins with this one (`::test_value` vs `::test_value_2`);
    the next character must end the id or open a parametrisation."""
    want = f"FAILED {node}"
    return [l for l in out.splitlines()
            if l.startswith(want) and (len(l) == len(want) or l[len(want)] in " [")]


# 🔴 A REASON THAT CANNOT MATCH TWICE IS NOT A CHECK. Frozen reasons are compared
# as text, and pytest's own diagnostics carry content that differs every run: the
# numbered tmp session (`pytest-16280` vs `pytest-16522`), object repr addresses,
# and this driver's own randomly named checkout. Ten of 539 controls drifted on
# exactly these three shapes the first time the frozen reasons were measured.
_VOLATILE = (
    (re.compile(r"pytest-\d+"), "pytest-<n>"),
    (re.compile(r"0x[0-9a-f]{4,}"), "0x<addr>"),
    (re.compile(r"injected-defect-controls-\w+"), "injected-defect-controls-<x>"),
)


def stable(text):
    """`text` with the parts that differ run to run replaced by placeholders.

    Applied to BOTH sides of every comparison and at recording time, so a reason
    frozen before this existed still matches.
    """
    for pattern, placeholder in _VOLATILE:
        text = pattern.sub(placeholder, text)
    return text


def evidence_lines(out):
    """The `E   ` lines of the traceback -- the assertion text as pytest printed it.

    🔴 NOT the `FAILED <node> - <reason>` summary line. pytest fits that line to the
    terminal, which is 80 columns when stdout is a pipe, and it does so in two ways:
    a short node id keeps its reason but TRUNCATED ("Failed..."), and a node id that
    is already too long gets NO reason appended at all. This programme's test names
    are mostly in the second class, so the first recording run on 2026-09-12
    harvested 31 reasons out of 539 -- and those 31 were stubs. The traceback is not
    fitted to anything.
    """
    return [l[4:].rstrip() for l in out.splitlines() if l.startswith("E   ")]


def classify(node, expected, rc, out):
    """(outcome, reason). The whole point: `rc != 0` is not a rejection.

    pytest exits 1 for test failures, 2 interrupted, 3 internal error, 4 usage
    error, 5 no tests ran -- and a collection error takes the test out of the run
    entirely. Only a real FAILURE of the NAMED node, at the assertion the control
    declares, shows that the control binds what it claims.
    """
    if expected is None:
        return "INDETERMINATE", "the control declares no expected reason"
    if rc == 0:
        return "NOT CAUGHT", (out.strip().splitlines() or [""])[-1][:160]
    if rc != 1:
        return "INDETERMINATE", f"pytest exited {rc}, which is not a test failure"
    failed = failed_lines(node, out)
    if not failed:
        return "INDETERMINATE", f"{node} is not in the FAILED summary"
    if any(l.startswith("ERROR") for l in out.splitlines()):
        return "INDETERMINATE", "pytest reported an ERROR as well as the failure"
    evidence = evidence_lines(out)
    observed = evidence[0] if evidence else failed[0]
    # against the ASSERTION TEXT, not the whole output: a string that happens to
    # appear in a source-context line is not the reason the test failed.
    if stable(expected) not in stable("\n".join(evidence)):
        return "INDETERMINATE", f"failed for another reason: {observed[:200]}"
    return "REJECTED", observed[:200]


def _on_signal(sig, _frame):
    global _PRESERVE
    print(f"\n🔴 SIGNAL {sig}: stopping the test subprocess and keeping the checkout",
          flush=True)
    if _CHILD is not None:
        _CHILD.terminate()
        try:
            _CHILD.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _CHILD.kill()
            _CHILD.wait()
    if _CHECKOUT is not None:
        _PRESERVE = True
        print(f"checkout PRESERVED: {_CHECKOUT}", flush=True)
    raise SystemExit(130)


def main(argv):
    global _CHECKOUT, _PRESERVE
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--run", action="store_true",
                    help="REQUIRED. Without it this driver does nothing.")
    ap.add_argument("--defects", default=str(
        pathlib.Path(__file__).with_name("injected_defect_controls.py")))
    ap.add_argument("--record", default=None,
                    help="also write the OBSERVED reasons, as JSON, to this path")
    a = ap.parse_args(argv)

    if not a.run:
        print("REFUSED: this driver EDITS source files and runs the suite against "
              "them. Pass --run to start it.")
        return REFUSED

    top = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True)
    if top.returncode != 0:
        print("REFUSED: not inside a git repository -- the disposable checkout is a "
              "git worktree.")
        return REFUSED
    repo = pathlib.Path(top.stdout.strip())

    # ponytail: the checkout is `git worktree add HEAD`, so uncommitted work would
    # not be the thing under test. Refuse rather than quietly test something else.
    # Upgrade path if this bites: carry `git diff HEAD` into the checkout.
    if git(repo, "status", "--porcelain").stdout.strip():
        print("REFUSED: the working tree is not clean. The checkout is made from "
              "HEAD, so uncommitted changes would NOT be under test.")
        return REFUSED

    defects, reasons = load_defects(a.defects)

    # ── two ways a tally inflates at GENERATION rather than at RECORDING
    seen, dupes = {}, []
    for lab, f, o, n, t in defects:
        key = (f, o, n, t)
        if key in seen:
            dupes.append((lab, seen[key]))
        else:
            seen[key] = lab
    for lab, other in dupes:
        print(f"🔴 DUPLICATE INJECTION -- counted twice: {lab!r} duplicates {other!r}")
    labels = [lab for lab, *_ in defects]
    dupe_labels = sorted({l for l in labels if labels.count(l) > 1})
    for lab in dupe_labels:
        print(f"🔴 DUPLICATE LABEL -- one expected reason for two controls: {lab!r}")
    # 🔴 AND THE OTHER DIRECTION. A declared reason whose control has been renamed
    # or removed binds nothing and hides the rename: found by hand on 2026-09-12
    # after two controls were re-anchored and their old reasons stayed behind.
    orphans = sorted(set(reasons) - set(labels))
    for lab in orphans:
        print(f"🔴 ORPHAN REASON -- declared for a control that does not exist: {lab!r}")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass

    holder = tempfile.mkdtemp(prefix="injected-defect-controls-")
    checkout = _CHECKOUT = pathlib.Path(holder) / "checkout"
    git(repo, "worktree", "add", "--detach", str(checkout), "HEAD")
    print(f"checkout: {checkout}", flush=True)

    def discard():
        git(repo, "worktree", "remove", "--force", str(checkout), check=False)
        shutil.rmtree(holder, ignore_errors=True)

    # ═══════════════════ THE CLEAN BASELINE, AND IT IS A STOP ══════════════════
    # A target test that is already broken FAILS under the defect too, so its
    # control reports REJECTED for work it never did. Run 1 on 2026-09-12 printed
    # this FAIL and kept going for 538 controls.
    nodes = sorted({t for *_r, t in defects})
    purge(checkout)
    rc, out = run_pytest(checkout, nodes)
    if rc != 0:
        print("🔴 BASELINE FAILURES -- these tests do not pass on CLEAN source, so "
              "every control naming them is meaningless:")
        for line in out.splitlines():
            if line.startswith(("FAILED", "ERROR", "  (no match")):
                print("   ", line[:150])
        print(f"clean baseline over {len(nodes)} distinct target tests: FAIL")
        print("ABORTED before the first injection.")
        discard()
        return BASELINE_FAILED
    print(f"clean baseline over {len(nodes)} distinct target tests: PASS", flush=True)

    counts = {"REJECTED": 0, "NOT CAUGHT": 0, "INDETERMINATE": 0}
    stale = 0
    observed_reasons = {}
    try:
        for label, path, old, new, node in defects:
            f = checkout / path
            src = f.read_text()
            if src.count(old) != 1:
                print(f"  STALE CONTROL -- DID NOT RUN  {label}: anchor matched "
                      f"{src.count(old)} times in {path}")
                stale += 1
                continue
            injected = src.replace(old, new)
            f.write_text(injected)
            purge(checkout)
            rc, out = run_pytest(checkout, [node])

            # 🔴 VERIFY BEFORE RESTORING. Writing a start-up snapshot back over a file
            # that something else has changed is how incident 2 wiped a guard added
            # mid-run. If the file is not what we injected, we do not know what it is.
            if f.read_text() != injected:
                print(f"🔴 {path} does not hold the content this driver injected -- "
                      f"something else wrote it. STOPPING rather than overwrite it.")
                print(f"checkout PRESERVED: {checkout}")
                _PRESERVE = True
                return TREE_VIOLATED
            f.write_text(src)

            outcome, reason = classify(node, reasons.get(label), rc, out)
            counts[outcome] += 1
            print(f"  {outcome:13s}  {label}")
            if outcome != "REJECTED":
                print(f"      {reason}")
            # Harvest evidence from ANY nonzero run, including one whose named test
            # never ran: a control that errors still has to DECLARE what it produces,
            # and it will still be INDETERMINATE when it is measured.
            if rc != 0:
                ev = evidence_lines(out)
                if ev:
                    observed_reasons[label] = stable(ev[0])
    finally:
        if not _PRESERVE:        # a violated or signalled checkout is EVIDENCE
            discard()

    if a.record:
        pathlib.Path(a.record).write_text(json.dumps(observed_reasons, indent=1,
                                                     ensure_ascii=False))
        print(f"recorded {len(observed_reasons)} observed reasons -> {a.record}")

    problems = (counts["NOT CAUGHT"] + counts["INDETERMINATE"] + stale
                + len(dupes) + len(dupe_labels) + len(orphans))
    print(f"\n{counts['REJECTED']}/{len(defects)} defects rejected; "
          f"{counts['NOT CAUGHT']} not caught; {counts['INDETERMINATE']} "
          f"indeterminate; {stale} stale")
    print("distinct injections:", len(seen), "| duplicate labels:", len(dupe_labels),
          "| orphan reasons:", len(orphans))
    print("clean baseline: PASS")   # the only path that reaches here
    print("PROBLEMS:", problems)
    return OK if problems == 0 else PROBLEMS


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
