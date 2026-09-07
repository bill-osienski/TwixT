"""THE H1 MATCH EXECUTION WRAPPER -- attempt 2. IT IS NOT AUTHORIZED.

The 2026-09-05 match was launched by an ad-hoc script that exited 0
unconditionally: its VOID was invisible to anything reading the status, and the
gate was restored by hand afterwards. This is the tested replacement.

WHAT IT DOES, IN ORDER
  1. Reads the RUNNER's gate, `h1_viability_runner.H1_EXECUTION_AUTHORIZED`, at
     BOTH entries. It has no gate of its own: a second switch for the same run
     would be a second thing to leave open.
  2. Refuses existing or same-file output paths BEFORE spawning anything
     (`check_output_paths`, the runner's own precheck), so a knowable path
     condition never becomes a VOID that retires a block.
  3. Runs the match as a WORKER in its own process group under the
     requalification's supervisor: outer cap = the runner's 180-minute deadline
     + grace, SIGINT forwarded, the group terminated/killed/probed after every
     exit. The worker calls the PUBLIC entry `run(results, mode=match,
     trace_path=)` with paths only -- nothing can be injected through here.
  4. RESTORES THE GATE in the runner's SOURCE after every exit -- worker done,
     timeout, interrupt, or a supervisor crash -- and verifies the rewrite. A
     restoration that fails is exit 10 and supersedes every other code.

EXIT CODES, one meaning each (the requalification's, plus two):
  0 COMPLETED  the match played all 224 games and the report was written --
               VIABLE, NOT_VIABLE, INCONCLUSIVE or CAP_SATURATED_NO_RATE; the
               verdict is in the results file, not in the status
  3 VOID       instrument failure; no result may be published; the block retires
  4 UNEXPECTED an unnamed exception, reported rather than escaping
  5 UNAUTHORIZED  the gate, at whichever entry was reached; nothing spawned
  6 TIMEOUT    the outer supervisor killed the process group
  7 REFUSED    a precondition (existing output, unregistered block, ...)
  8 CLEANUP_FAILED  a descendant survived; whatever the worker said, not success
  9 INTERRUPTED  the operator stopped the run; not a VOID, not a success, and
               drawn seeds stay EXPOSED (see INTERRUPT_ACCOUNTING_RULE)
 10 GATE_NOT_RESTORED  the source could not be rewritten to False; supersedes all

No model is loaded, no JVM started, no seed drawn and no game played by this
module: all of that is the runner's, behind its gate.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Optional, Sequence

from . import h1_viability_runner as RUN
from . import runtime_requalification as RQ
from .runtime_requalification import supervise                     # shared, tested

MODULE = "scripts.GPU.alphazero.h1_match_command"

#: FRESH output locations for attempt 2. Attempt 1's live under
#: evidence/2026-09-05-t1j-h1-match/ and are preserved, never reused.
OUT_DIR = "docs/superpowers/evidence/2026-09-07-t1j-h1-match-attempt2"
DEFAULT_RESULTS = f"{OUT_DIR}/01_h1_results.jsonl"
DEFAULT_TRACE = f"{OUT_DIR}/02_h1_trace.jsonl"

#: The file whose gate line is restored. The runner's own source, resolved from
#: the imported module, never retyped as a path.
RUNNER_SOURCE = RUN.__file__
_GATE_OPEN = re.compile(r"^H1_EXECUTION_AUTHORIZED = True$", re.M)
_GATE_CLOSED = "H1_EXECUTION_AUTHORIZED = False"

#: The outer cap exceeds the runner's own deadline by this much, so the runner
#: reports a deadline VOID as a VOID before the supervisor turns it into a kill.
SUPERVISOR_GRACE_S = 60
#: How long a SIGINT-forwarded worker gets to write its INTERRUPTED record.
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
    return RUN.H1_EXECUTION_AUTHORIZED is True


def restore_gate(path: str = RUNNER_SOURCE) -> bool:
    """Rewrite `H1_EXECUTION_AUTHORIZED = True` to False in the runner SOURCE and
    verify. True when the file now holds exactly one closed gate line and no open
    one -- including when it already did. False on any failure: the caller must
    treat that as its own outcome, never as success."""
    try:
        src = open(path, encoding="utf-8").read()
    except OSError:
        return False
    if src.count(_GATE_CLOSED + "\n") == 1 and not _GATE_OPEN.search(src):
        return True
    if len(_GATE_OPEN.findall(src)) != 1:
        return False
    new = _GATE_OPEN.sub(_GATE_CLOSED, src)
    try:
        tmp = path + ".restoring"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(new)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        back = open(path, encoding="utf-8").read()
    except OSError:
        return False
    return back.count(_GATE_CLOSED + "\n") == 1 and not _GATE_OPEN.search(back)


def _parser():
    import argparse
    ap = argparse.ArgumentParser(prog="h1_match_command",
                                 description="THE H1 MATCH (attempt 2). NOT AUTHORIZED.")
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--trace", default=DEFAULT_TRACE)
    # 🔴 NO `--runner-source`. A production override let a decoy path be
    # "restored" while the real runner's gate stayed open (review, 2026-09-07).
    # The restoration target is bound to the IMPORTED runner's source; tests
    # substitute through the PRIVATE keyword of `main`, unreachable from argv.
    ap.add_argument("--worker", action="store_true",
                    help="internal: run the match in this process (spawned by main)")
    return ap


def worker_main(argv: Optional[Sequence[str]] = None) -> int:
    """The match, in THIS process, through the runner's PUBLIC entry with paths
    only. Reads the gate itself: it does not trust the parent."""
    a = _parser().parse_args(argv)
    if not gate_is_open():
        print("the H1 match is NOT AUTHORIZED (H1_EXECUTION_AUTHORIZED is False). No "
              "JVM was started, no model loaded, no game played, no file written.",
              file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        rc = RUN.run(a.results, mode=RUN.MATCH_MODE, trace_path=a.trace)
    except RUN.H1VoidError as e:
        print(f"VOID: {e}", file=sys.stderr)
        return EXIT_VOID
    except RUN.H1Error as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except KeyboardInterrupt:
        print("INTERRUPTED: the operator stopped the match. Not a VOID and not a "
              f"result. {RUN.INTERRUPT_ACCOUNTING_RULE}", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as e:                                    # noqa: BLE001
        print(f"UNEXPECTED {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    print(f"COMPLETED: the runner returned {rc}; the verdict is in {a.results}")
    return EXIT_COMPLETED if rc == 0 else EXIT_UNEXPECTED


def main(argv: Optional[Sequence[str]] = None, *,
         _runner_source: Optional[str] = None) -> int:
    """CLI. Gate, then INSIDE the restoration boundary: output precheck,
    supervised worker; then gate restoration, whatever happened.

    `_runner_source` is a PRIVATE test seam (keyword-only, never on argv): the
    file whose gate line is restored. Production always restores the imported
    runner's own source.

    🔴 EVERY POST-AUTHORIZATION STEP IS INSIDE THE BOUNDARY. The output precheck
    used to return exit 7 BEFORE the `finally`, so an existing results file
    left the gate open with zero restorations (review, 2026-09-07). It still
    refuses before spawning; it no longer skips the restore, and a failed
    restore supersedes the refusal.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    a = _parser().parse_args(argv)
    if a.worker:
        return worker_main(argv)
    if not gate_is_open():
        print("the H1 match is NOT AUTHORIZED (H1_EXECUTION_AUTHORIZED is False). No "
              "worker was spawned, no JVM started, no file written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    target = RUNNER_SOURCE if _runner_source is None else _runner_source

    code = EXIT_UNEXPECTED
    try:
        try:
            RUN.check_output_paths(a.results, a.trace)
            refused = False
        except RUN.H1Error as e:
            print(f"refused before spawning: {e}", file=sys.stderr)
            code, refused = EXIT_REFUSED, True   # no return: the finally must run
        if not refused:
            r = supervise([sys.executable, "-m", MODULE, "--worker", *argv],
                          timeout_s=RUN.RUN_DEADLINE_S + SUPERVISOR_GRACE_S,
                          kill_grace_s=5.0, interrupt_grace_s=INTERRUPT_GRACE_S)
            if r["timed_out"]:
                print(f"TIMEOUT: the worker exceeded {RUN.RUN_DEADLINE_S + SUPERVISOR_GRACE_S}s; "
                      f"its process group was killed (cleared={r['group_cleared']}).",
                      file=sys.stderr)
            if r["interrupted"]:
                print(f"INTERRUPTED by the operator; forwarded to the worker, which exited "
                      f"{r['exit_code']}.", file=sys.stderr)
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
        # 🔴 THE GATE IS RESTORED WHATEVER HAPPENED -- refusal, timeout, interrupt,
        # crash or completion -- and a restoration that fails is its own outcome:
        # an open gate beside any other code is the larger fact, so it
        # supersedes, INCLUDING a refusal.
        if not restore_gate(target):
            print(f"GATE NOT RESTORED: {target} could not be rewritten to "
                  f"H1_EXECUTION_AUTHORIZED = False. Restore it BY HAND before "
                  f"anything else.", file=sys.stderr)
            code = EXIT_GATE_NOT_RESTORED
    return code


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
