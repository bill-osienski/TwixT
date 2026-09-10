"""H2 — the execution WRAPPER. Supervises one run and restores the gate.

Modelled on `h1_match_command`, whose lessons are inherited rather than
rediscovered: the gate line is restored in the runner's OWN source resolved from
the imported module, there is NO `--runner-source` override (a production
override let a decoy path be restored while the real gate stayed open), and
restoration runs after EVERY exit including a refusal.

🔴 NOTHING HERE HAS RUN. The runner's gate is False; this wrapper refuses with it.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Optional, Sequence

from . import h2_match_runner as RUN
from . import runtime_requalification as RQ
from .runtime_requalification import supervise                     # shared, tested

OUT_DIR = "docs/superpowers/evidence/2026-09-09-t1j-h2-deterministic-readout"
DEFAULT_RESULTS = f"{OUT_DIR}/03_h2_results.jsonl"
DEFAULT_TRACE = f"{OUT_DIR}/04_h2_trace.jsonl"

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
    ap.add_argument("--worker", action="store_true",
                    help="internal: run the match in this process (spawned by the "
                         "supervisor, never by hand)")
    # 🔴 NO --runner-source, and no flag that reaches the gate. Opening H2 is a
    # reviewed one-line edit plus a separate authorization, and nothing here
    # accepts an environment variable or a config file either.
    return ap


def worker_main(argv: Sequence[str]) -> int:
    """THE SUPERVISED CHILD. Calls the public entry, which resolves its own
    schedule, identity, deadline and play seam -- this passes only the outputs.
    """
    a = _parser().parse_args(list(argv))
    if not gate_is_open():
        print("the H2 match is NOT AUTHORIZED inside the worker.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        report = RUN.run_h2(results_path=a.results, trace_path=a.trace)
    except RUN.H2VoidError as e:
        print(f"VOID: {e}", file=sys.stderr)
        return EXIT_VOID
    except RUN.H2Error as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:                                    # noqa: BLE001
        print(f"UNEXPECTED {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    print(f"COMPLETED: outcome {report.get('outcome')!r}; the verdict is in {a.results}")
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
            RUN.check_output_paths(a.results, a.trace)
            refused = False
        except RUN.H2Error as e:
            print(f"refused before spawning: {e}", file=sys.stderr)
            code, refused = EXIT_REFUSED, True   # no return: the finally must run
        if not refused:
            r = supervise([sys.executable, "-m", MODULE, "--worker", *argv],
                          timeout_s=RUN.RUN_DEADLINE_S + SUPERVISOR_GRACE_S,
                          kill_grace_s=5.0, interrupt_grace_s=INTERRUPT_GRACE_S)
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
