"""The H1 match EXECUTION WRAPPER. NOTHING EXECUTES.

The 2026-09-05 match was launched by an ad-hoc script that exited 0
unconditionally, so its VOID was invisible to anything reading the status, and
the gate was restored by hand. This wrapper is the tested replacement: it reads
the runner's gate at BOTH entries, refuses spent or existing output paths before
spawning, supervises the match as a worker in its own process group (the
requalification's supervisor, reused), maps every outcome to its own exit code,
and RESTORES THE GATE in the runner source after every exit -- a restoration
that fails supersedes every other code.

No model is loaded, no JVM started, no game played, no seed drawn: the runner is
replaced at its public entry, and the supervisor is exercised with throwaway
Python children.
"""
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from scripts.GPU.alphazero import h1_match_command as CMD
from scripts.GPU.alphazero import h1_viability_runner as RUN
from scripts.GPU.alphazero import runtime_requalification as RQ


# ──────────────────────────────── the gate ──────────────────────────────────

def test_the_wrapper_reads_the_RUNNERS_gate_and_it_is_False():
    assert RUN.H1_EXECUTION_AUTHORIZED is False
    assert CMD.gate_is_open() is False


def test_main_refuses_without_spawning_while_the_gate_is_shut(monkeypatch, tmp_path):
    monkeypatch.setattr(CMD, "supervise",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    monkeypatch.setattr(CMD, "restore_gate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("restored")))
    rc = CMD.main(["--results", str(tmp_path / "r.jsonl"), "--trace", str(tmp_path / "t.jsonl")])
    assert rc == CMD.EXIT_UNAUTHORIZED == 5
    assert not (tmp_path / "r.jsonl").exists() and not (tmp_path / "t.jsonl").exists()


def test_the_worker_refuses_without_running_while_the_gate_is_shut(monkeypatch, tmp_path):
    monkeypatch.setattr(RUN, "run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    rc = CMD.worker_main(["--results", str(tmp_path / "r.jsonl"),
                          "--trace", str(tmp_path / "t.jsonl")])
    assert rc == CMD.EXIT_UNAUTHORIZED == 5


@pytest.mark.parametrize("extra", [[], ["--worker"]])
def test_both_entries_refuse_as_a_FRESH_SUBPROCESS(tmp_path, extra):
    r = subprocess.run(
        [sys.executable, "-m", "scripts.GPU.alphazero.h1_match_command", *extra,
         "--results", str(tmp_path / "r.jsonl"), "--trace", str(tmp_path / "t.jsonl")],
        capture_output=True, text=True, cwd=".", timeout=120)
    assert r.returncode == CMD.EXIT_UNAUTHORIZED == 5, (r.returncode, r.stderr)
    assert "NOT AUTHORIZED" in r.stderr or "UNAUTHORIZED" in r.stderr
    assert not (tmp_path / "r.jsonl").exists() and not (tmp_path / "t.jsonl").exists()


def test_the_wrapper_names_only_the_H1_gate():
    import pathlib
    src = pathlib.Path(CMD.__file__).read_text(encoding="utf-8")
    for other in ("D1_EXECUTION_AUTHORIZED", "L0_EXECUTION_AUTHORIZED", "SCREEN_AUTHORIZED",
                  "LOWPLY_QUALIFICATION_AUTHORIZED", "RUNTIME_REQUAL_AUTHORIZED"):
        assert other not in src, other
    assert "H1_EXECUTION_AUTHORIZED" in src


# ─────────────────────────── fresh output locations ─────────────────────────

def test_the_default_output_paths_are_the_ATTEMPT2_evidence_dir_and_are_now_SPENT():
    """⚠ INVERTED by the match: attempt 2 ran once on 2026-09-07 and COMPLETED,
    so the default paths EXIST and the runner's precheck REFUSES them -- the
    wrapper cannot re-run the spent attempt by accident, even with the gate open."""
    assert CMD.OUT_DIR == "docs/superpowers/evidence/2026-09-07-t1j-h1-match-attempt2"
    assert CMD.DEFAULT_RESULTS == f"{CMD.OUT_DIR}/01_h1_results.jsonl"
    assert CMD.DEFAULT_TRACE == f"{CMD.OUT_DIR}/02_h1_trace.jsonl"
    assert os.path.exists(CMD.DEFAULT_RESULTS) and os.path.exists(CMD.DEFAULT_TRACE)
    with pytest.raises(RUN.H1Error, match="already exists"):
        RUN.check_output_paths(CMD.DEFAULT_RESULTS, CMD.DEFAULT_TRACE)
    # attempt 1's paths are NOT the defaults, and still exist
    assert os.path.exists("docs/superpowers/evidence/2026-09-05-t1j-h1-match/01_h1_results.jsonl")


def test_main_refuses_an_EXISTING_output_path_before_spawning(monkeypatch, tmp_path):
    monkeypatch.setattr(RUN, "H1_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(CMD, "supervise",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    (tmp_path / "r.jsonl").write_text("")
    copy = _runner_copy(tmp_path, "True")
    rc = CMD.main(["--results", str(tmp_path / "r.jsonl"), "--trace", str(tmp_path / "t.jsonl")],
                  _runner_source=str(copy))
    assert rc == CMD.EXIT_REFUSED == 7
    # 🔴 REVIEW REPRO: the refusal used to return BEFORE the restoration boundary,
    # leaving the real gate open. The refusal must still CLOSE the target.
    assert "H1_EXECUTION_AUTHORIZED = False\n" in copy.read_text(), \
        "output refusal skipped gate restoration"


def test_output_refusal_with_a_FAILED_restoration_is_exit_10_not_7(monkeypatch, tmp_path):
    monkeypatch.setattr(RUN, "H1_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(CMD, "supervise",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    monkeypatch.setattr(CMD, "restore_gate", lambda path: False)
    (tmp_path / "r.jsonl").write_text("")
    rc = CMD.main(["--results", str(tmp_path / "r.jsonl"), "--trace", str(tmp_path / "t.jsonl")],
                  _runner_source=str(_runner_copy(tmp_path, "True")))
    assert rc == CMD.EXIT_GATE_NOT_RESTORED == 10


def test_the_CLI_has_NO_runner_source_override_and_a_decoy_is_rejected(monkeypatch, tmp_path):
    """🔴 REVIEW REPRO: `--runner-source decoy` made the wrapper "restore" the
    decoy and report success while the REAL runner's gate stayed open. The
    override is gone from the CLI; the target is bound to the imported runner's
    source, and substitution exists only as a PRIVATE keyword for tests."""
    monkeypatch.setattr(RUN, "H1_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(CMD, "supervise",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))
    restored = []
    monkeypatch.setattr(CMD, "restore_gate", lambda path: restored.append(path) or True)
    decoy = _runner_copy(tmp_path, "True")
    with pytest.raises(SystemExit) as exc:
        CMD.main(["--results", str(tmp_path / "r.jsonl"), "--trace", str(tmp_path / "t.jsonl"),
                  "--runner-source", str(decoy)])
    assert exc.value.code == 2, "argparse must reject the unknown option"
    assert restored == [], "nothing was restored on a rejected command line"
    assert "H1_EXECUTION_AUTHORIZED = True" in decoy.read_text()
    assert "--runner-source" not in CMD._parser().format_help()


def test_restoration_targets_the_IMPORTED_runners_source_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(RUN, "H1_EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(CMD, "supervise", lambda cmd, **k: {
        "exit_code": 0, "timed_out": False, "interrupted": False, "group_cleared": True})
    restored = []
    monkeypatch.setattr(CMD, "restore_gate", lambda path: restored.append(path) or True)
    rc = CMD.main(["--results", str(tmp_path / "r.jsonl"), "--trace", str(tmp_path / "t.jsonl")])
    assert rc == 0
    assert [os.path.realpath(p) for p in restored] == [os.path.realpath(RUN.__file__)]


# ─────────────────────────────── exit codes ─────────────────────────────────

def test_exit_codes_are_the_requalifications_plus_two_and_distinct():
    assert (CMD.EXIT_COMPLETED, CMD.EXIT_VOID, CMD.EXIT_UNEXPECTED, CMD.EXIT_UNAUTHORIZED,
            CMD.EXIT_TIMEOUT, CMD.EXIT_REFUSED, CMD.EXIT_CLEANUP_FAILED,
            CMD.EXIT_INTERRUPTED, CMD.EXIT_GATE_NOT_RESTORED) == (0, 3, 4, 5, 6, 7, 8, 9, 10)
    assert CMD.EXIT_VOID == RQ.EXIT_VOID and CMD.EXIT_TIMEOUT == RQ.EXIT_TIMEOUT
    assert CMD.EXIT_CLEANUP_FAILED == RQ.EXIT_CLEANUP_FAILED
    assert CMD.EXIT_INTERRUPTED == RQ.EXIT_INTERRUPTED == 9


@pytest.mark.parametrize("outcome,code", [
    ("completed", 0), ("void", 3), ("interrupt", 9), ("refused", 7), ("boom", 4)])
def test_the_worker_maps_each_outcome_to_its_own_exit_code(monkeypatch, tmp_path,
                                                          outcome, code):
    """🔴 THE 2026-09-05 WRAPPER EXITED 0 FOR EVERYTHING. A VOID is 3, an operator
    interrupt is 9 (not a VOID, not a success), a refusal 7, the unnamed 4."""
    monkeypatch.setattr(RUN, "H1_EXECUTION_AUTHORIZED", True)
    seen = {}

    def fake_run(results_path, *, mode, trace_path):
        seen.update(results=results_path, mode=mode, trace=trace_path)
        if outcome == "void":
            raise RUN.H1VoidError("instrument")
        if outcome == "interrupt":
            raise KeyboardInterrupt()
        if outcome == "refused":
            raise RUN.H1Error("precondition")
        if outcome == "boom":
            raise RuntimeError("boom")
        return 0

    monkeypatch.setattr(RUN, "run", fake_run)
    rc = CMD.worker_main(["--results", str(tmp_path / "r.jsonl"),
                          "--trace", str(tmp_path / "t.jsonl")])
    assert rc == code
    assert seen["mode"] == RUN.MATCH_MODE
    assert seen["results"] == str(tmp_path / "r.jsonl") and seen["trace"] == str(tmp_path / "t.jsonl")


def test_the_worker_calls_the_PUBLIC_entry_with_paths_only(monkeypatch, tmp_path):
    """No task, callable, evaluator or plan path can be injected through the
    wrapper: it calls `run(results, mode=, trace_path=)` and nothing else."""
    import inspect
    src = inspect.getsource(CMD.worker_main)
    assert "RUN.run(" in src and "_run(" not in src.replace("RUN.run(", "")
    for forbidden in ("_tasks", "_plan_path", "_agent_factory", "_setup", "_binder"):
        assert forbidden not in src, forbidden


# ───────────────────────────── gate restoration ─────────────────────────────

def _runner_copy(tmp_path, value):
    src = open(RUN.__file__, encoding="utf-8").read()
    line = f"H1_EXECUTION_AUTHORIZED = {value}\n"
    assert src.count("H1_EXECUTION_AUTHORIZED = False\n") == 1
    p = tmp_path / "runner_copy.py"
    p.write_text(src.replace("H1_EXECUTION_AUTHORIZED = False\n", line), encoding="utf-8")
    return p


def test_restore_gate_rewrites_True_to_False_and_verifies(tmp_path):
    p = _runner_copy(tmp_path, "True")
    assert CMD.restore_gate(str(p)) is True
    assert "H1_EXECUTION_AUTHORIZED = False\n" in p.read_text()
    assert "H1_EXECUTION_AUTHORIZED = True" not in p.read_text()


def test_restore_gate_is_idempotent_on_an_already_closed_gate(tmp_path):
    p = _runner_copy(tmp_path, "False")
    assert CMD.restore_gate(str(p)) is True


def test_restore_gate_reports_FAILURE_when_the_line_is_missing_or_unwritable(tmp_path):
    p = tmp_path / "no_gate.py"
    p.write_text("x = 1\n")
    assert CMD.restore_gate(str(p)) is False
    assert CMD.restore_gate(str(tmp_path / "absent.py")) is False


def test_restore_gate_reports_FAILURE_when_the_rewrite_did_not_take(tmp_path, monkeypatch):
    """The rewrite is VERIFIED by re-reading. With the replace neutralised the
    file still says True, and success here would be an open gate reported closed."""
    p = _runner_copy(tmp_path, "True")
    monkeypatch.setattr(CMD.os, "replace", lambda src, dst: os.unlink(src))
    assert CMD.restore_gate(str(p)) is False
    assert "H1_EXECUTION_AUTHORIZED = True" in p.read_text()


def test_the_REAL_runner_source_is_the_file_restore_gate_targets_by_default():
    assert os.path.realpath(CMD.RUNNER_SOURCE) == os.path.realpath(RUN.__file__)


def _open_main(monkeypatch, tmp_path, supervise_result, *, restore_ok=True):
    monkeypatch.setattr(RUN, "H1_EXECUTION_AUTHORIZED", True)
    calls = {"supervise": [], "restore": []}

    def fake_supervise(cmd, *, timeout_s, kill_grace_s, **kw):
        calls["supervise"].append({"cmd": cmd, "timeout_s": timeout_s, "kill_grace_s": kill_grace_s})
        if isinstance(supervise_result, BaseException):
            raise supervise_result
        return dict(supervise_result)

    def fake_restore(path):
        calls["restore"].append(path)
        return restore_ok

    monkeypatch.setattr(CMD, "supervise", fake_supervise)
    monkeypatch.setattr(CMD, "restore_gate", fake_restore)
    runner_copy = _runner_copy(tmp_path, "True")
    rc = CMD.main(["--results", str(tmp_path / "r.jsonl"), "--trace", str(tmp_path / "t.jsonl")],
                  _runner_source=str(runner_copy))
    return rc, calls, runner_copy


def test_main_supervises_a_WORKER_under_the_H1_deadline_plus_grace_and_passes_0_through(
        monkeypatch, tmp_path):
    rc, calls, runner_copy = _open_main(
        monkeypatch, tmp_path,
        {"exit_code": 0, "timed_out": False, "interrupted": False, "group_cleared": True})
    assert rc == CMD.EXIT_COMPLETED == 0
    (sup,) = calls["supervise"]
    assert sup["cmd"][:3] == [sys.executable, "-m", "scripts.GPU.alphazero.h1_match_command"]
    assert "--worker" in sup["cmd"]
    assert str(tmp_path / "r.jsonl") in sup["cmd"] and str(tmp_path / "t.jsonl") in sup["cmd"]
    assert sup["timeout_s"] == RUN.RUN_DEADLINE_S + CMD.SUPERVISOR_GRACE_S == 10800 + 60
    assert calls["restore"] == [str(runner_copy)], "the gate is restored exactly once, after the run"


@pytest.mark.parametrize("worker_code", [0, 3, 9])
def test_main_restores_the_gate_after_EVERY_outcome_and_a_failed_restore_supersedes(
        monkeypatch, tmp_path, worker_code):
    rc, calls, _ = _open_main(
        monkeypatch, tmp_path,
        {"exit_code": worker_code, "timed_out": False, "interrupted": False, "group_cleared": True},
        restore_ok=False)
    assert calls["restore"], "the gate was not restored"
    assert rc == CMD.EXIT_GATE_NOT_RESTORED == 10


def test_main_restores_the_gate_even_when_the_supervisor_RAISES(monkeypatch, tmp_path):
    rc, calls, _ = _open_main(monkeypatch, tmp_path, RuntimeError("supervisor crashed"))
    assert calls["restore"], "an exception in the supervisor skipped the restore"
    assert rc == CMD.EXIT_UNEXPECTED == 4


def test_main_NEVER_reports_success_when_cleanup_failed(monkeypatch, tmp_path):
    rc, calls, _ = _open_main(
        monkeypatch, tmp_path,
        {"exit_code": 0, "timed_out": False, "interrupted": False, "group_cleared": False})
    assert rc == CMD.EXIT_CLEANUP_FAILED == 8 and calls["restore"]


def test_main_reports_TIMEOUT_when_the_supervisor_killed_the_tree(monkeypatch, tmp_path):
    rc, calls, _ = _open_main(
        monkeypatch, tmp_path,
        {"exit_code": RQ.EXIT_TIMEOUT, "timed_out": True, "interrupted": False, "group_cleared": True})
    assert rc == CMD.EXIT_TIMEOUT == 6 and calls["restore"]


def test_main_reports_INTERRUPTED_when_the_operator_stopped_it(monkeypatch, tmp_path):
    rc, calls, _ = _open_main(
        monkeypatch, tmp_path,
        {"exit_code": 9, "timed_out": False, "interrupted": True, "group_cleared": True})
    assert rc == CMD.EXIT_INTERRUPTED == 9 and calls["restore"]


# ─────────────── the shared supervisor forwards an operator interrupt ───────
#
# The worker runs in its OWN session, so a terminal Ctrl-C reaches only the
# wrapper. Unforwarded, the match would keep playing with nobody watching and
# the gate would be restored under it. The supervisor forwards SIGINT to the
# group, waits for the worker to write its INTERRUPTED record, then cleans up.

INT_WORKER = textwrap.dedent("""
    import signal, sys, time
    def stop(*_):
        open(sys.argv[1] + ".got_sigint", "w").write("1")   # PROOF the forwarded SIGINT arrived
        sys.exit(9)                                     # what worker_main does on KeyboardInterrupt
    signal.signal(signal.SIGINT, stop)
    open(sys.argv[1], "w").write("ready")
    time.sleep(300)
""")


def test_the_supervisor_FORWARDS_an_operator_interrupt_to_the_worker_group(tmp_path, monkeypatch):
    """The operator's KeyboardInterrupt is raised INSIDE the supervisor's wait --
    delivered deterministically by a Popen whose first wait raises once the
    worker has installed its handler. (A first version sent SIGINT to the pytest
    process itself; under the full suite another module's signal disposition
    swallowed it and the wait ran to its timeout: a test that depends on the
    process's signal state is not a test of the supervisor.)"""
    ready = tmp_path / "ready"

    class InterruptingPopen(subprocess.Popen):
        def wait(self, timeout=None):
            if not getattr(self, "_raised", False):
                self._raised = True
                end = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < end:
                    time.sleep(0.02)
                assert ready.exists(), "the worker never became ready"
                time.sleep(0.1)
                raise KeyboardInterrupt()
            return super().wait(timeout=timeout)

    monkeypatch.setattr(RQ.subprocess, "Popen", InterruptingPopen)
    t0 = time.monotonic()
    r = RQ.supervise([sys.executable, "-c", INT_WORKER, str(ready)],
                     timeout_s=60, kill_grace_s=2, interrupt_grace_s=5)
    # 🔴 BOTH of these, because without forwarding the grace-timeout KILL path
    # ALSO reports 9 and clears the group -- a control removing the forward was
    # NOT CAUGHT until the worker's own handler had to leave its mark, fast.
    assert time.monotonic() - t0 < 4, "the worker did not exit on the forwarded SIGINT"
    assert (ready.parent / "ready.got_sigint").exists(), "the worker's SIGINT handler never ran"
    assert r["interrupted"] is True and r["timed_out"] is False
    assert r["exit_code"] == 9, "the worker's own INTERRUPTED code passes through"
    assert r["group_cleared"] is True


def test_the_supervisor_reports_interrupted_as_a_field_on_every_result(tmp_path):
    r = RQ.supervise([sys.executable, "-c", "import sys; sys.exit(0)"],
                     timeout_s=30, kill_grace_s=1)
    assert r == {"exit_code": 0, "timed_out": False, "interrupted": False, "group_cleared": True}
