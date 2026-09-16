"""H3 FULL STUDY — the supervised WRAPPER for the OPENING-GENERATION RUN.

TRANSPOSED FROM `h3_pilot_command`, NOT REWRITTEN. That launch path ran end to
end on 2026-09-15 -- 40/40, exit 0, the gate restored by its own `finally` -- and
a second implementation of a proven path is a second thing to get wrong.

🔴 GENERATION IS A RUN, AND IT HAD NO LAUNCH PATH. The public entry drove the
production collaborators directly: no outer deadline, no process group, nothing
to stop a surviving JVM, and no unconditional gate restoration. Opening the gate
and calling it would have been exactly the shape H2 and the pilot each had to fix.

WHAT THIS GUARANTEES, on every exit:
  * the worker runs in its OWN PROCESS GROUP under an outer cap, so a timeout
    kills the whole group including a java child that outlives it;
  * a surviving descendant is a CLEANUP FAILURE, never a success;
  * `H3_GENERATION_AUTHORIZED = False` is restored in a `finally` and VERIFIED by
    reading the file back;
  * gate-restoration failure and cleanup failure SUPERSEDE the worker's result --
    an open gate or a live JVM is the larger fact.

🔴 ANY ATTEMPTED GENERATION RETIRES THE WHOLE RANGE. Attempts consume generation
seeds whether or not an opening was accepted, so there is no partial retirement
to argue about.

🔴 NOTHING HERE HAS RUN. The generator's gate is False and the destination is
absent; this wrapper refuses with them.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Mapping, Optional, Sequence

from . import h3_study_generator as RUN
from . import runtime_requalification as RQ
from .runtime_requalification import supervise                 # shared, tested

#: 🔴 PER ATTEMPT, and never a directory holding a spent run's records. H2's
#: default pointed at a spent attempt's directory until the pre-run verification
#: found it, and the launch would have been refused after the gate was opened.
#: Resolved FROM THE GENERATOR, so the two cannot disagree about the destination.
def default_paths():
    return (RUN.DEFAULT_OUT, RUN.DEFAULT_TRACE)

#: Directories holding a SPENT run's records. Nothing new may write into one.
SPENT_OUT_DIRS = (
    "docs/superpowers/evidence/2026-09-09-t1j-h2-deterministic-readout",
    "docs/superpowers/evidence/2026-09-12-t1j-h2-match-attempt3",
    "docs/superpowers/evidence/2026-09-12-t1j-INCIDENT-control-harness-ran-a-match",
    "docs/superpowers/evidence/2026-09-14-t1j-h3-pilot",
)
#: 🔴 THE OPENING-GENERATION DESTINATION IS NOT LISTED HERE, and that is
#: deliberate. `docs/superpowers/evidence/2026-09-15-t1j-h3-study-openings` is
#: ABSENT and UNSPENT: a directory is marked spent only once an attempted run has
#: CONSUMED it. Listing it in advance would refuse the very run it is for, which
#: is the mirror of H2's defect -- there the default pointed INTO a spent
#: directory and the launch would have been refused after the gate was opened.

CAPABILITY_BYTES = 32
RUNNER_SOURCE = RUN.__file__
_GATE_OPEN = re.compile(r"^H3_GENERATION_AUTHORIZED = True$", re.M)
_GATE_CLOSED = "H3_GENERATION_AUTHORIZED = False"
MODULE = "scripts.GPU.alphazero.h3_generation_command"

SUPERVISOR_GRACE_S = 60
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
#: OUTCOMES THAT ARE RESULTS, not failures. The pilot's job is to answer three
#: questions; a fired stop rule IS the answer, and a partial run is informative.
EXIT_PARTIAL = 13
EXIT_STOP_RULE_FIRED = 14


def gate_is_open() -> bool:
    """The RUNNER's gate, read live. This module has none of its own."""
    return RUN.H3_GENERATION_AUTHORIZED is True


def restore_gate(_runner_source: str = RUNNER_SOURCE) -> bool:
    """Rewrite the open gate line closed in the runner SOURCE, then VERIFY by
    reading the file back. True when the file holds exactly one closed line and no
    open one -- including when it already did. False on any failure, which the
    caller must treat as its own outcome and never as success.

    🔑 The source path is a PRIVATE keyword with no CLI flag. H1 learned this: an
    overridable path lets a decoy file be restored while the real gate stays open.
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
        prog="h3_generation_command",
        description="THE H3 FULL STUDY, one segment. NOT AUTHORIZED, it has no seed block, and its opening set has not been generated.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--worker", action="store_true",
                    help="internal: run in this process. Refused without the "
                         "supervisor's single-use capability.")
    ap.add_argument("--capability-fd", type=int, default=None, dest="capability_fd",
                    help="internal: the inherited read end of the capability pipe. "
                         "NOT a path: nothing here opens or deletes a named file.")
    # 🔴 NO --runner-source, no flag that reaches the gate or the seeds.
    return ap


def _make_capability() -> int:
    """An anonymous pipe carrying one random token; returns the READ end.

    🔴 NOT A FILE. The version before H2's used a path, accepted ANY file of the
    right length, and UNLINKED whatever it was given -- a guard that destroyed
    data. There is no path here, so there is nothing to destroy.
    """
    import secrets
    r_fd, w_fd = os.pipe()
    with os.fdopen(w_fd, "w") as fh:
        fh.write(secrets.token_hex(CAPABILITY_BYTES))
    return r_fd


def _consume_capability(fd: Optional[int]) -> bool:
    """Read the token from an INHERITED descriptor. DELETES NOTHING, EVER."""
    if fd is None:
        return False
    try:
        with os.fdopen(int(fd), "r", closefd=True) as fh:
            token = fh.read().strip()
    except (OSError, ValueError, TypeError):
        return False
    return len(token) == CAPABILITY_BYTES * 2 and all(
        c in "0123456789abcdef" for c in token)


def _resolve_paths(a):
    out, trace = default_paths()
    return (a.out or out, a.trace or trace)


def _check_destination(out: str, trace: str) -> None:
    """CREATE-ONLY, and never inside a SPENT directory."""
    if os.path.abspath(out) == os.path.abspath(trace):
        raise RUN.H3GenerationError("the two outputs must be two files")
    for p in (out, trace):
        if os.path.lexists(p):
            raise RUN.H3GenerationError(
                f"the output path already exists: {p}. The opening set is "
                f"create-only: a second generation would silently replace the "
                f"population the study is defined over.")
        for spent in SPENT_OUT_DIRS:
            if p.startswith(spent.rstrip("/") + "/"):
                raise RUN.H3GenerationError(
                    f"{p} is inside a SPENT run's directory {spent}")


def _classify(report: Mapping[str, Any]) -> int:
    """SAY WHAT THE REPORT WAS. A fired rule and a partial run are RESULTS.

    A fired stop rule takes precedence over PARTIAL: a monotone count that has
    crossed its threshold is CONCLUSIVE, and a partial run is not (card §4.4).
    """
    if report.get("n") != RUN.RULES.PAIRS_PER_STRATUM:
        return EXIT_VOID
    return EXIT_COMPLETED


def worker_main(argv: Sequence[str]) -> int:
    """THE SUPERVISED CHILD. Refuses unless the supervisor spawned it."""
    a = _parser().parse_args(list(argv))
    if not _consume_capability(a.capability_fd):
        print("refused: --worker runs the pilot UNSUPERVISED and outside the gate "
              "restoration boundary. The supervisor spawns it with a single-use "
              "capability; it is not a way to run the pilot by hand.",
              file=sys.stderr)
        return EXIT_REFUSED
    if not gate_is_open():
        print("the H3 opening generation is NOT AUTHORIZED inside the worker.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        out_path, trace_path = _resolve_paths(a)
        report = RUN.generate_co_produced(out_path=out_path,
                                          trace_path=trace_path)
    except KeyboardInterrupt:
        print("INTERRUPTED by the operator; the trace records INTERRUPTED.",
              file=sys.stderr)
        return EXIT_INTERRUPTED
    except RUN.H3GenerationCleanupError as e:
        # 🔑 ITS OWN CODE, OUTRANKING the body's outcome: a run that produced a
        # population and left a JVM alive has not succeeded.
        print(f"CLEANUP FAILED: {e}", file=sys.stderr)
        return EXIT_CLEANUP_FAILED
    except RUN.H3GenerationError as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as e:                                    # noqa: BLE001
        print(f"UNEXPECTED {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED
    if not os.path.lexists(out_path):
        print(f"VOID: the generator returned a doc but {out_path} does not exist.",
              file=sys.stderr)
        return EXIT_VOID
    code = _classify(report)
    print(f"{'COMPLETED' if code == EXIT_COMPLETED else 'RESULT'}: "
          f"openings={report.get('n')} "
          f"digest={report.get('opening_set_digest')} out={out_path}")
    return code


def main(argv: Optional[Sequence[str]] = None, *,
         _runner_source: str = RUNNER_SOURCE) -> int:
    """CLI. Gate, then INSIDE the restoration boundary: output precheck and a
    SUPERVISED worker; then gate restoration, whatever happened.

    The worker runs in its OWN PROCESS GROUP under an outer cap, so a timeout
    kills the WHOLE GROUP including a java child that outlives it, and an operator
    interrupt is forwarded rather than leaving an orphan running.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    a = _parser().parse_args(argv)
    if a.worker:
        return worker_main(argv)
    if not gate_is_open():
        print("the H3 opening generation is NOT AUTHORIZED (H3_GENERATION_AUTHORIZED is "
              "False). No worker was spawned, no JVM started, no file written.",
              file=sys.stderr)
        if not restore_gate(_runner_source):
            print("🔴 THE GATE COULD NOT BE VERIFIED CLOSED. Restore it BY HAND.",
                  file=sys.stderr)
            return EXIT_GATE_NOT_RESTORED
        return EXIT_UNAUTHORIZED

    code = EXIT_UNEXPECTED
    try:
        try:
            _check_destination(*_resolve_paths(a))
            refused = False
        except RUN.H3GenerationError as e:
            print(f"refused before spawning: {e}", file=sys.stderr)
            code, refused = EXIT_REFUSED, True   # no return: the finally must run
        if not refused:
            cap_fd = _make_capability()
            try:
                r = supervise([sys.executable, "-m", MODULE, "--worker",
                               "--capability-fd", str(cap_fd), *argv],
                              timeout_s=RUN.RULES.SEGMENT_DEADLINE_S + SUPERVISOR_GRACE_S,
                              kill_grace_s=5.0, interrupt_grace_s=INTERRUPT_GRACE_S,
                              pass_fds=[cap_fd])
            finally:
                try:
                    os.close(cap_fd)
                except OSError:
                    pass
            if r["timed_out"]:
                print(f"TIMEOUT: the worker exceeded the outer cap; its process "
                      f"group was killed (cleared={r['group_cleared']}).",
                      file=sys.stderr)
            if r["interrupted"]:
                print(f"INTERRUPTED by the operator; forwarded to the worker, "
                      f"which exited {r['exit_code']}.", file=sys.stderr)
            if not r["group_cleared"]:
                print(f"CLEANUP FAILED: a descendant of the worker survived; the "
                      f"worker itself exited {r['exit_code']}. Nothing here is a "
                      f"success.", file=sys.stderr)
                code = EXIT_CLEANUP_FAILED
            elif r["timed_out"]:
                code = EXIT_TIMEOUT
            elif r["interrupted"]:
                code = EXIT_INTERRUPTED
            else:
                code = r["exit_code"]
    except Exception as e:                                    # noqa: BLE001
        print(f"UNEXPECTED in the supervisor: {type(e).__name__}: {e}",
              file=sys.stderr)
        code = EXIT_UNEXPECTED
    finally:
        # 🔴 RESTORED WHATEVER HAPPENED -- refusal, timeout, interrupt, crash or
        # completion -- and a failed restoration SUPERSEDES every other code,
        # including a refusal: an open gate is the larger fact.
        if not restore_gate(_runner_source):
            print(f"GATE NOT RESTORED: {_runner_source} could not be rewritten to "
                  f"{_GATE_CLOSED}. Restore it BY HAND before anything else.",
                  file=sys.stderr)
            code = EXIT_GATE_NOT_RESTORED
    return code


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
