"""THE POPULATION-FREEZE COMMAND — one shot, and the barrier closes behind it.

🔴 WHY THIS EXISTS. `freeze_population()` checks the barrier and writes the
official artifact; it does NOT close the barrier afterwards, and it must not —
a body that reopens and re-closes its own authorization is a body that can be
re-entered. So the restoration lives out here, in a `finally`, in the one place
that is not the thing being authorized.

Without it the barrier was not one-shot: once `H3_POPULATION_FREEZE_AUTHORIZED`
was edited to `True`, it stayed True, and repeated calls could produce any number
of candidate "official" populations. Uniform generation is deterministic, so each
would hold the SAME openings — which makes it worse, not better: several byte-
identical artifacts in different places, none of them distinguishable as the one
the study is defined over.

WHAT THIS COMMAND GUARANTEES

  * the entry it calls takes NO paths. The destination is the frozen default and
    nothing can redirect it;
  * the barrier is restored in a `finally` that runs after success, refusal,
    timeout and interrupt alike;
  * restoration is VERIFIED by reading the source file back, never by trusting
    the rewrite's own return value;
  * **a failed restoration supersedes every other outcome.** A frozen population
    with the barrier left open is not a success: the next invocation would freeze
    again.

WHAT IT DOES NOT DO. It never OPENS the barrier. Opening it is a source edit made
under its own authorization, exactly as every gate in this programme is opened;
a command that could open its own barrier would make the barrier a formality.

Run:  .venv/bin/python -m scripts.GPU.alphazero.h3_freeze_command --run
"""
from __future__ import annotations

import os
import re
import sys
from typing import Optional

from . import h3_study_generator as GEN
from . import h3_study_rules as RULES

GENERATOR_SOURCE = GEN.__file__

_BARRIER_OPEN = re.compile(r"^H3_POPULATION_FREEZE_AUTHORIZED = True$", re.M)
_BARRIER_CLOSED = "H3_POPULATION_FREEZE_AUTHORIZED = False"

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_NOT_AUTHORIZED = 5
EXIT_TIMEOUT = 6
EXIT_INTERRUPTED = 7
EXIT_FAILED = 8
#: 🔴 ITS OWN CODE, AND IT SUPERSEDES EVERY OTHER. A population written while the
#: barrier stayed open is not a completed freeze.
EXIT_BARRIER_NOT_RESTORED = 9


def barrier_is_open() -> bool:
    """The generator's barrier, read live."""
    return GEN.H3_POPULATION_FREEZE_AUTHORIZED is True


def restore_barrier(_generator_source: str = GENERATOR_SOURCE) -> bool:
    """Rewrite the open barrier line closed in the generator SOURCE, then VERIFY
    by reading the file back.

    True when the file holds exactly one closed line and no open one — including
    when it already did. False on any failure, which the caller must treat as its
    own outcome and never as success.

    🔑 The source path is a PRIVATE keyword with no CLI flag. H1 learned this: an
    overridable path lets a decoy file be restored while the real one stays open.
    """
    try:
        with open(_generator_source, encoding="utf-8") as fh:
            src = fh.read()
    except OSError:
        return False
    if src.count(_BARRIER_CLOSED + "\n") == 1 and not _BARRIER_OPEN.search(src):
        return True
    if len(_BARRIER_OPEN.findall(src)) != 1:
        return False
    new = _BARRIER_OPEN.sub(_BARRIER_CLOSED, src)
    try:
        tmp = _generator_source + ".restoring"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(new)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, _generator_source)
        with open(_generator_source, encoding="utf-8") as fh:
            back = fh.read()
    except OSError:
        return False
    return back.count(_BARRIER_CLOSED + "\n") == 1 and not _BARRIER_OPEN.search(back)


def barrier_readback(_generator_source: str = GENERATOR_SOURCE) -> Optional[str]:
    """What the SOURCE FILE says the barrier is, as text.

    🔑 READ FROM THE FILE, NOT FROM MEMORY. The module object in this process was
    imported before the rewrite and still holds the old value; reporting that
    would be reporting what we hoped rather than what is on disk.
    """
    try:
        with open(_generator_source, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    if _BARRIER_OPEN.search(text):
        return "True"
    if text.count(_BARRIER_CLOSED + "\n") == 1:
        return "False"
    return None


def _parser():
    import argparse
    ap = argparse.ArgumentParser(add_help=True, description=__doc__)
    ap.add_argument("--run", action="store_true",
                    help="REQUIRED. Without it this command does nothing.")
    return ap


def main(argv=None) -> int:
    a = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if not a.run:
        print("REFUSED: this command writes the OFFICIAL population artifact and "
              "fixes the study's population. Pass --run to start it.")
        return EXIT_REFUSED

    if not barrier_is_open():
        print("REFUSED: H3_POPULATION_FREEZE_AUTHORIZED is False. Freezing the "
              "population is a separate reviewed step; this command does not "
              "open its own barrier.")
        return EXIT_NOT_AUTHORIZED

    outcome, code, doc = "UNEXPECTED", EXIT_FAILED, None
    try:
        doc = GEN.freeze_population()           # NO PATHS. Nothing to redirect.
        outcome, code = "COMPLETED", EXIT_OK
    except GEN.H3GenerationDeadline as e:
        outcome, code = "TIMEOUT", EXIT_TIMEOUT
        print(f"TIMEOUT: {e}")
    except KeyboardInterrupt:
        outcome, code = "INTERRUPTED", EXIT_INTERRUPTED
        print("INTERRUPTED")
    except GEN.H3GenerationError as e:
        outcome, code = "REFUSED", EXIT_FAILED
        print(f"REFUSED: {e}")
    finally:
        # 🔴 UNCONDITIONAL, AND IT SUPERSEDES. Whatever happened above, the
        # barrier closes here and the close is VERIFIED from the file.
        # 🔑 THE MODULE GLOBAL IS READ AT CALL TIME, not bound as a default at
        # import. `def restore_barrier(_src=GENERATOR_SOURCE)` freezes the path
        # when the module loads, so nothing -- including a test -- can observe
        # the restoration against a different file. Still no CLI flag: the path
        # is not reachable from argv, only from this line.
        restored = restore_barrier(GENERATOR_SOURCE)
        readback = barrier_readback(GENERATOR_SOURCE)
        if not restored or readback != "False":
            outcome, code = "BARRIER_NOT_RESTORED", EXIT_BARRIER_NOT_RESTORED
            print(f"🔴 BARRIER NOT RESTORED (restore_barrier={restored}, "
                  f"file says {readback!r}). A population written while the "
                  f"barrier stays open is NOT a completed freeze: the next "
                  f"invocation would freeze again. Close it by hand and record "
                  f"that this happened.")

    print(f"outcome            : {outcome}")
    print(f"barrier readback   : {barrier_readback(GENERATOR_SOURCE)!r}  "
          f"(from the FILE)")
    print(f"artifact           : {GEN.DEFAULT_OUT}")
    print(f"  exists           : {os.path.lexists(GEN.DEFAULT_OUT)}")
    print(f"trace              : {GEN.DEFAULT_TRACE}")
    print(f"  exists           : {os.path.lexists(GEN.DEFAULT_TRACE)}")
    if doc is not None:
        print(f"n                  : {doc['n']}")
        print(f"opening_set_digest : {doc['opening_set_digest']}")
        print(f"\n🔑 OPENING_SET_DIGEST IS *NOT* PINNED BY THIS COMMAND. Recording "
              f"it in `h3_study_rules` is a separate reviewed edit, made after "
              f"the artifact above has been inspected. Freezing produces the "
              f"population; pinning it is what makes the study play it.")
    print(f"exit               : {code}")
    return code


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
