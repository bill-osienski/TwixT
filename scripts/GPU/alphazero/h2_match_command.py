"""H2 — the execution WRAPPER. Supervises one run and restores the gate.

Modelled on `h1_match_command`, whose lessons are inherited rather than
rediscovered: the gate line is restored in the runner's OWN source resolved from
the imported module, there is NO `--runner-source` override (a production
override let a decoy path be restored while the real gate stayed open), and
restoration runs after EVERY exit including a refusal.

🔴 NOTHING HERE HAS RUN. The runner's gate is False; this wrapper refuses with it.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Mapping, Optional, Sequence

from . import h2_match_runner as RUN
from . import runtime_requalification as RQ
from .runtime_requalification import supervise                     # shared, tested

OUT_DIR = "docs/superpowers/evidence/2026-09-09-t1j-h2-deterministic-readout"
DEFAULT_RESULTS = f"{OUT_DIR}/03_h2_results.jsonl"
DEFAULT_TRACE = f"{OUT_DIR}/04_h2_trace.jsonl"
#: The frozen report location from the card's §7. Create-only, like every output.
DEFAULT_REPORT = f"{OUT_DIR}/09_report.json"

#: 🔴 AN INHERITED ANONYMOUS PIPE. THE TWO VERSIONS BEFORE THIS WERE BOTH WRONG.
#:
#: `H2_SUPERVISED_WORKER=1` was caller-settable and contradicted this module's own
#: claim that no environment variable reaches the path.
#:
#: Then a FILE holding a random token, whose path came in on argv. That was worse
#: in one specific way: the worker accepted ANY file of the right length -- the
#: token was never compared with anything -- and it UNLINKED whatever path it was
#: given, valid or not. `--worker --capability <ordinary-file>` DELETED that file
#: with the gate still closed. A bypass guard that destroys data is not a guard.
#:
#: Now: the supervisor makes an anonymous pipe, writes a random token into it and
#: passes the READ END to the child through `pass_fds`. Everything above fd 2 is
#: closed on exec by default, so the descriptor exists in the child ONLY because
#: the parent chose to pass it. THERE IS NO PATH, so there is nothing to delete,
#: and no file of any length can stand in for it.
#:
#: ⚠ WHAT THIS IS AND IS NOT, precisely. It removes the accidental bypass -- typing
#: `--worker` cannot run an unsupervised match -- and it cannot damage anything. It
#: is NOT an authentication boundary: a caller who deliberately constructs a pipe,
#: writes 64 hex characters into it and passes the descriptor number can still
#: reach the worker. No local mechanism proves parentage without a secret shared
#: out of band, and nothing here pretends otherwise. THE GATE is what authorizes a
#: run; this only stops the path being reached by accident or by typing.
CAPABILITY_BYTES = 32

#: The file whose gate line is restored: the runner's own source, resolved from
#: the imported module and never retyped as a path.
RUNNER_SOURCE = RUN.__file__
_GATE_OPEN = re.compile(r"^H2_EXECUTION_AUTHORIZED = True$", re.M)
_GATE_CLOSED = "H2_EXECUTION_AUTHORIZED = False"

#: This module, so the supervised worker is THIS file re-entered with --worker.
MODULE = "scripts.GPU.alphazero.h2_match_command"

#: The outer cap exceeds the runner's own deadline by this much, so the runner
#: reports a deadline VOID as a VOID before the supervisor turns it into a kill.
SUPERVISOR_GRACE_S = 60
#: How long a SIGINT-forwarded worker gets to write its record.
INTERRUPT_GRACE_S = 120

EXIT_COMPLETED = 0
EXIT_VOID = RQ.EXIT_VOID                       # 3
EXIT_UNEXPECTED = RQ.EXIT_UNEXPECTED           # 4
EXIT_UNAUTHORIZED = RQ.EXIT_UNAUTHORIZED       # 5
EXIT_TIMEOUT = RQ.EXIT_TIMEOUT                 # 6
EXIT_REFUSED = RQ.EXIT_REFUSED                 # 7
EXIT_CLEANUP_FAILED = RQ.EXIT_CLEANUP_FAILED   # 8
EXIT_INTERRUPTED = RQ.EXIT_INTERRUPTED         # 9
EXIT_GATE_NOT_RESTORED = 10
#: Outcomes that are RESULTS, not failures, and are not "COMPLETED" either.
EXIT_DEGENERATE = 11
EXIT_NO_RATE = 12


def gate_is_open() -> bool:
    """The RUNNER's gate, read live. This module has none of its own."""
    return RUN.H2_EXECUTION_AUTHORIZED is True


def restore_gate(_runner_source: str = RUNNER_SOURCE) -> bool:
    """Rewrite the open gate line to the closed one in the runner SOURCE, then
    VERIFY by reading the file back.

    True when the file holds exactly one closed gate line and no open one --
    including when it already did. False on any failure, which the caller must
    treat as its own outcome and never as success.

    🔑 The source path is a PRIVATE keyword with no CLI flag. H1 learned this:
    an overridable path lets a decoy file be restored while the real gate stays
    open, and the exit code would then report success.
    """
    try:
        src = open(_runner_source, encoding="utf-8").read()
    except OSError:
        return False
    if src.count(_GATE_CLOSED + "\n") == 1 and not _GATE_OPEN.search(src):
        return True
    if len(_GATE_OPEN.findall(src)) != 1:
        return False
    new = _GATE_OPEN.sub(_GATE_CLOSED, src)
    try:
        tmp = _runner_source + ".restoring"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(new)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, _runner_source)
        back = open(_runner_source, encoding="utf-8").read()
    except OSError:
        return False
    return back.count(_GATE_CLOSED + "\n") == 1 and not _GATE_OPEN.search(back)


def _parser():
    import argparse
    ap = argparse.ArgumentParser(
        prog="h2_match_command",
        description="THE H2 DETERMINISTIC-READOUT MATCH. NOT AUTHORIZED.")
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--trace", default=DEFAULT_TRACE)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--worker", action="store_true",
                    help="internal: run the match in this process. Refused without the "
                         "supervisor's single-use capability.")
    ap.add_argument("--capability-fd", type=int, default=None, dest="capability_fd",
                    help="internal: the inherited read end of the supervisor's "
                         "capability pipe. NOT a path: nothing here opens or deletes "
                         "a file the caller names.")
    # 🔴 NO --runner-source, and no flag that reaches the gate. Opening H2 is a
    # reviewed one-line edit plus a separate authorization, and nothing here
    # accepts an environment variable or a config file either.
    return ap


def _make_capability() -> int:
    """An anonymous pipe carrying one random token. Returns the READ end.

    The write end is closed here, so the child reads the token and then EOF; the
    channel cannot be reused and there is no file anywhere.
    """
    import secrets
    r_fd, w_fd = os.pipe()
    with os.fdopen(w_fd, "w") as fh:
        fh.write(secrets.token_hex(CAPABILITY_BYTES))
    return r_fd


def _consume_capability(fd: Optional[int]) -> bool:
    """Read the token from an INHERITED descriptor. DELETES NOTHING, EVER.

    🔴 The previous version unlinked whatever path it was handed, including one
    that failed validation, so a mistyped `--capability` destroyed an ordinary
    file while the gate was shut. There is no path here to destroy, and this
    function's only effect is to close the descriptor it was given.
    """
    if fd is None:
        return False
    try:
        with os.fdopen(int(fd), "r", closefd=True) as fh:
            token = fh.read().strip()
    except (OSError, ValueError, TypeError):
        return False
    return len(token) == CAPABILITY_BYTES * 2 and all(
        c in "0123456789abcdef" for c in token)


def worker_main(argv: Sequence[str]) -> int:
    """THE SUPERVISED CHILD. Calls the public entry, which resolves its own
    schedule, identity, deadline and play seam -- this passes only the outputs.

    🔴 IT REFUSES UNLESS THE SUPERVISOR SPAWNED IT. `--worker` was a public bypass
    around supervision AND gate restoration: typing it ran the match in-process,
    unbounded, with nothing to restore the gate afterwards.
    """
    a = _parser().parse_args(list(argv))
    if not _consume_capability(a.capability_fd):
        print("refused: --worker runs the match UNSUPERVISED and outside the gate "
              "restoration boundary. It is spawned by the supervisor, which hands it "
              "a single-use capability; it is not a way to run H2 by hand.",
              file=sys.stderr)
        return EXIT_REFUSED
    if not gate_is_open():
        print("the H2 match is NOT AUTHORIZED inside the worker.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        report = RUN.run_h2(results_path=a.results, trace_path=a.trace,
                            report_path=a.report)
    except KeyboardInterrupt:
        print("INTERRUPTED by the operator; the trace records INTERRUPTED.",
              file=sys.stderr)
        return EXIT_INTERRUPTED
    except RUN.H2VoidError as e:
        print(f"VOID: {e}", file=sys.stderr)
        return EXIT_VOID
    except RUN.H2Error as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:                                    # noqa: BLE001
        print(f"UNEXPECTED {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    return _persist_and_classify(report, a.report)


def _persist_and_classify(report: Mapping[str, Any], report_path: str) -> int:
    """SAY WHAT THE REPORT WAS. The RUNNER writes it.

    🔴 Two corrections live here. The report was once RETURNED AND DISCARDED, with
    the worker printing that the verdict was "in" the results file and the frozen
    report never written. It was then written HERE -- after the runner had already
    committed `run_end/OK`, so a write failure exited VOID while the durable trace
    said OK. The runner now persists and fsyncs it BEFORE the terminal record, and
    this function only classifies. It verifies the file exists rather than assuming
    it: a classification that cannot see its own artifact is a claim, not a check.
    """
    if not os.path.lexists(report_path):
        print(f"VOID: the runner returned a report but {report_path} does not exist; "
              f"the verdict has no durable record.", file=sys.stderr)
        return EXIT_VOID

    outcome = report.get("outcome")
    if not report.get("reported"):
        if outcome == "INCONCLUSIVE — DEGENERATE DESIGN":
            print(f"DEGENERATE DESIGN: {report.get('reason')}", file=sys.stderr)
            return EXIT_DEGENERATE
        if outcome == "CAP_SATURATED_NO_RATE":
            print(f"NO RATE: {report.get('reason')}", file=sys.stderr)
            return EXIT_NO_RATE
        print(f"REFUSED: {report.get('reason')}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"COMPLETED: outcome {outcome!r}; the report is in {report_path}")
    return EXIT_COMPLETED


def main(argv: Optional[Sequence[str]] = None, *,
         _runner_source: str = RUNNER_SOURCE) -> int:
    """CLI. Gate, then INSIDE the restoration boundary: output precheck and a
    SUPERVISED worker; then gate restoration, whatever happened.

    🔴 THE WORKER RUNS IN ITS OWN PROCESS GROUP AND UNDER AN OUTER CAP. The
    runner's 480-minute deadline is polled BETWEEN games, so a single blocked game
    -- a hung JVM, a stalled query -- could overrun it indefinitely. `supervise`
    makes the worker a session leader, so a timeout kills the WHOLE GROUP including
    a java child that outlives it, and an operator interrupt is forwarded rather
    than leaving an orphan running with nobody watching.

    `_runner_source` is a PRIVATE test seam, keyword-only and never on argv.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    a = _parser().parse_args(argv)
    if a.worker:
        return worker_main(argv)
    if not gate_is_open():
        print("the H2 match is NOT AUTHORIZED (H2_EXECUTION_AUTHORIZED is False). No "
              "worker was spawned, no JVM started, no file written.", file=sys.stderr)
        # Restore anyway: a wrapper that only tidies after the paths it expects is
        # not a wrapper, and the gate should already be closed.
        if not restore_gate(_runner_source):
            print("🔴 THE GATE COULD NOT BE VERIFIED CLOSED. Restore it BY HAND.",
                  file=sys.stderr)
            return EXIT_GATE_NOT_RESTORED
        return EXIT_UNAUTHORIZED

    code = EXIT_UNEXPECTED
    try:
        try:
            RUN.check_output_paths(a.results, a.trace, a.report)
            refused = False
        except RUN.H2Error as e:
            print(f"refused before spawning: {e}", file=sys.stderr)
            code, refused = EXIT_REFUSED, True   # no return: the finally must run
        if not refused:
            cap_fd = _make_capability()          # the child's only way in
            try:
                r = supervise([sys.executable, "-m", MODULE, "--worker",
                               "--capability-fd", str(cap_fd), *argv],
                              timeout_s=RUN.RUN_DEADLINE_S + SUPERVISOR_GRACE_S,
                              kill_grace_s=5.0, interrupt_grace_s=INTERRUPT_GRACE_S,
                              pass_fds=[cap_fd])
            finally:
                try:
                    os.close(cap_fd)         # nothing on disk, nothing to unlink
                except OSError:
                    pass
            if r["timed_out"]:
                print(f"TIMEOUT: the worker exceeded "
                      f"{RUN.RUN_DEADLINE_S + SUPERVISOR_GRACE_S}s; its process group was "
                      f"killed (cleared={r['group_cleared']}).", file=sys.stderr)
            if r["interrupted"]:
                print(f"INTERRUPTED by the operator; forwarded to the worker, which "
                      f"exited {r['exit_code']}.", file=sys.stderr)
            if not r["group_cleared"]:
                print(f"CLEANUP FAILED: a descendant of the worker survived; the worker "
                      f"itself exited {r['exit_code']}. Nothing here is a success.",
                      file=sys.stderr)
                code = EXIT_CLEANUP_FAILED
            elif r["timed_out"]:
                code = EXIT_TIMEOUT
            elif r["interrupted"]:
                code = EXIT_INTERRUPTED
            else:
                code = r["exit_code"]
    except Exception as e:                                    # noqa: BLE001
        print(f"UNEXPECTED in the supervisor: {type(e).__name__}: {e}", file=sys.stderr)
        code = EXIT_UNEXPECTED
    finally:
        # 🔴 RESTORED WHATEVER HAPPENED -- refusal, timeout, interrupt, crash or
        # completion -- and a failed restoration SUPERSEDES every other code,
        # including a refusal: an open gate is the larger fact.
        if not restore_gate(_runner_source):
            print(f"GATE NOT RESTORED: {_runner_source} could not be rewritten to "
                  f"H2_EXECUTION_AUTHORIZED = False. Restore it BY HAND before "
                  f"anything else.", file=sys.stderr)
            code = EXIT_GATE_NOT_RESTORED
    return code


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
