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

OUT_DIR = "docs/superpowers/evidence/2026-09-09-t1j-h2-deterministic-readout"
DEFAULT_RESULTS = f"{OUT_DIR}/03_h2_results.jsonl"
DEFAULT_TRACE = f"{OUT_DIR}/04_h2_trace.jsonl"

#: The file whose gate line is restored: the runner's own source, resolved from
#: the imported module and never retyped as a path.
RUNNER_SOURCE = RUN.__file__
_GATE_OPEN = re.compile(r"^H2_EXECUTION_AUTHORIZED = True$", re.M)
_GATE_CLOSED = "H2_EXECUTION_AUTHORIZED = False"

EXIT_COMPLETED = 0
EXIT_VOID = RQ.EXIT_VOID                       # 3
EXIT_UNEXPECTED = RQ.EXIT_UNEXPECTED           # 4
EXIT_UNAUTHORIZED = RQ.EXIT_UNAUTHORIZED       # 5
EXIT_REFUSED = RQ.EXIT_REFUSED                 # 7
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
    # 🔴 NO --runner-source, and no flag that reaches the gate. Opening H2 is a
    # reviewed one-line edit plus a separate authorization, and nothing here
    # accepts an environment variable or a config file either.
    return ap


def main(argv: Optional[Sequence[str]] = None, *,
         _runner_source: str = RUNNER_SOURCE) -> int:
    """One run, supervised, with the gate restored after EVERY exit.

    🔑 RESTORATION IS UNCONDITIONAL AND LAST. H1's second blocker was an output
    refusal that returned before the `finally`, leaving the gate open on a path
    nobody thought of as a run. Every exit below goes through the same restore.
    """
    args = _parser().parse_args(argv)
    if not gate_is_open():
        print("H2 execution is UNAUTHORIZED. No model was loaded, no JVM started, no "
              "seed drawn, no game played and no file written.", file=sys.stderr)
        # Restore anyway: the gate should already be closed, and a wrapper that
        # only tidies up after the paths it expects is not a wrapper.
        if not restore_gate(_runner_source):
            print("🔴 THE GATE COULD NOT BE VERIFIED CLOSED. Restore it BY HAND.",
                  file=sys.stderr)
            return EXIT_GATE_NOT_RESTORED
        return EXIT_UNAUTHORIZED

    code = EXIT_UNEXPECTED
    try:                                                      # pragma: no cover
        raise RUN.H2Error(
            "no play seam is wired into this wrapper: H2 is implemented "
            "code-and-test only, and the gameplay seam is supplied by the "
            "execution authorization, not by this module.")
    except RUN.H2VoidError as e:                              # pragma: no cover
        print(f"VOID: {e}", file=sys.stderr)
        code = EXIT_VOID
    except RUN.H2Error as e:                                  # pragma: no cover
        print(f"refused: {e}", file=sys.stderr)
        code = EXIT_REFUSED
    except Exception as e:                                    # noqa: BLE001
        print(f"UNEXPECTED {type(e).__name__}: {e}", file=sys.stderr)
        code = EXIT_UNEXPECTED
    finally:                                                  # pragma: no cover
        if not restore_gate(_runner_source):
            print("🔴 THE GATE COULD NOT BE VERIFIED CLOSED. Restore it BY HAND.",
                  file=sys.stderr)
            code = EXIT_GATE_NOT_RESTORED
    return code


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
