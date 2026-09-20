"""H3 — THE SUPERVISED COMBINATION COMMAND. ONE SHOT, AND IT CLOSES ITS GATE.

🔴 WHY THE GATE NEEDED A WRAPPER AT ALL.

Opening `H3_COMBINATION_AUTHORIZED` and calling `combine()` by hand leaves the
gate OPEN in the source afterwards -- and most certainly leaves it open if the
combination raises, which is exactly when nobody is thinking about it. An
authorization that has to be withdrawn by remembering is not a one-shot
authorization; it is a standing one with a note attached.

So the gate is restored in an UNCONDITIONAL `finally`, the SOURCE FILE is read
back to confirm it, and the receipt is written AFTER restoration so that it can
record what the readback actually said.

🔑 IT ACCEPTS NOTHING. No path, no input, no destination, no collaborator. The
only argument is `--run`, which is not a parameter of the work: it exists so
that importing this module, or running it by accident, does nothing at all.
Every seam that could aim the combination elsewhere lives on
`h3_combine.combine_unguarded`, which this never calls.

🔴 A FAILED RESTORATION SUPERSEDES EVERY OTHER OUTCOME. If the combination
succeeded but the gate could not be closed, the exit status is
`EXIT_GATE_NOT_RESTORED` and not `EXIT_OK` -- the tree is left in a state where
a second combination could be attempted, and that is the more urgent fact about
the run. H1 learned this the hard way; the ordering is copied from it
deliberately.
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from typing import Any, Dict, Optional

from . import h3_combine as COMBINE
from . import h3_study_command as CMD
from . import h3_study_runner as RUN

COMBINE_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "h3_combine.py")

_GATE_OPEN = re.compile(r"^H3_COMBINATION_AUTHORIZED\s*=\s*True\s*$", re.M)
_GATE_CLOSED = "H3_COMBINATION_AUTHORIZED = False"

#: the parent-owned receipt. Create-only, beside the report it describes.
RECEIPT = f"{COMBINE.COMBINED_OUT_DIR}/00_combination_receipt.json"

EXIT_OK = 0
EXIT_REFUSED = 2                 # --run absent, or the module refused outright
EXIT_NOT_AUTHORIZED = 5          # the gate was shut; nothing was attempted
EXIT_FAILED = 8                  # the combination raised
#: 🔴 SUPERSEDING. Reported instead of any of the above when the gate could not
#: be closed, because a tree left open is the more urgent outcome.
EXIT_GATE_NOT_RESTORED = 9


def gate_is_open(_combine_source: str = COMBINE_SOURCE) -> bool:
    try:
        with open(_combine_source, encoding="utf-8") as fh:
            return bool(_GATE_OPEN.search(fh.read()))
    except OSError:
        return False


def restore_gate(_combine_source: str = COMBINE_SOURCE) -> bool:
    """Rewrite the open gate line closed in the SOURCE, then verify by reading
    the file back. True when the file holds exactly one closed line and no open
    one -- including when it already did.

    🔑 The source path is a PRIVATE keyword with no CLI flag. An overridable
    path lets a decoy file be restored while the real one stays open.
    """
    try:
        with open(_combine_source, encoding="utf-8") as fh:
            src = fh.read()
    except OSError:
        return False
    if src.count(_GATE_CLOSED + "\n") == 1 and not _GATE_OPEN.search(src):
        return True
    if len(_GATE_OPEN.findall(src)) != 1:
        return False
    new = _GATE_OPEN.sub(_GATE_CLOSED, src)
    try:
        tmp = _combine_source + ".restoring"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(new)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, _combine_source)
    except OSError:
        return False
    return gate_readback(_combine_source) == "False"


def gate_readback(_combine_source: str = COMBINE_SOURCE) -> Optional[str]:
    """What the SOURCE FILE says the gate is: "True", "False", or None when it
    cannot be read.

    🔑 READ FROM THE FILE, not from this process's imported module, which still
    holds the value it had at import and would happily report a closed gate for
    a file that is wide open.
    """
    try:
        with open(_combine_source, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    if _GATE_OPEN.search(text):
        return "True"
    if text.count(_GATE_CLOSED + "\n") == 1:
        return "False"
    return None


def write_receipt(payload: Dict[str, Any], _receipt: str = RECEIPT) -> bool:
    """Create-only, and it never raises into the caller's outcome: a receipt
    that could not be written is recorded as such by the exit status, and must
    not turn a completed combination into a crash."""
    try:
        if os.path.lexists(_receipt):
            return False
        os.makedirs(os.path.dirname(_receipt), exist_ok=True)
        fd = os.open(_receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        return True
    except OSError:
        return False


def _parser():
    import argparse
    ap = argparse.ArgumentParser(
        prog="h3_combine_command",
        description="THE H3 COMBINATION, once. Runs only with the combination "
                    "gate OPEN, restores it afterwards whatever happens, and "
                    "writes its report to a fixed create-only destination.")
    #: 🔴 THE ONLY ARGUMENT, AND IT IS NOT A PARAMETER OF THE WORK. There is no
    #: --out, no --inputs, no --segments and no --force: anything that could aim
    #: this somewhere else would make opening the gate authorize something other
    #: than what was reviewed.
    ap.add_argument("--run", action="store_true",
                    help="REQUIRED. Without it this command does nothing.")
    return ap


def main(argv=None, *, _combine_source: str = COMBINE_SOURCE,
         _receipt: str = RECEIPT) -> int:
    a = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if not a.run:
        print("REFUSED: --run is required. Nothing was read, pooled or written.")
        return EXIT_REFUSED

    #: 🔴 THE UNAUTHORIZED PATH WRITES NO RECEIPT, DELIBERATELY. It returns
    #: before anything is attempted, and a receipt here would CREATE THE
    #: DESTINATION -- so the real combination would then be refused for an
    #: output directory occupied by a refusal. H2's defect, by a new road.
    if not gate_is_open(_combine_source):
        print("the H3 COMBINATION is NOT AUTHORIZED "
              "(H3_COMBINATION_AUTHORIZED is False in the source). Nothing was "
              "read, nothing was pooled and nothing was written.")
        return EXIT_NOT_AUTHORIZED

    outcome, detail, report = "UNEXPECTED", None, None
    status = EXIT_FAILED
    try:
        report = COMBINE.combine()
        outcome, status = "COMPLETED", EXIT_OK
    except COMBINE.H3CombineError as exc:
        outcome, detail, status = "REFUSED", str(exc), EXIT_REFUSED
        print(f"REFUSED: {exc}")
    except BaseException as exc:                                 # noqa: BLE001
        outcome, detail, status = "FAILED", f"{type(exc).__name__}: {exc}", EXIT_FAILED
        traceback.print_exc()
    finally:
        #: unconditional, and BEFORE the receipt, so the receipt can record what
        #: the readback said rather than what we hoped it would say.
        restored = restore_gate(_combine_source)
        readback = gate_readback(_combine_source)
        receipt = {
            "design": "H3_FULL_STUDY_COMBINATION",
            "outcome": outcome,
            "detail": detail,
            "exit_code": status if restored else EXIT_GATE_NOT_RESTORED,
            "gate_restored": restored,
            "gate_readback": readback,
            "report_path": COMBINE.COMBINED_REPORT,
            "report_exists": os.path.lexists(COMBINE.COMBINED_REPORT),
            "inputs": [
                {"segment": k,
                 "results_path": CMD.default_paths(k)[0],
                 "trace_path": CMD.default_paths(k)[1],
                 "report_path": CMD.default_paths(k)[2],
                 "receipt_path": CMD.receipt_path(k),
                 "seed_block": list(RUN.SEGMENT_SEED_BLOCKS[k])}
                for k in range(len(RUN.SEGMENT_SEED_BLOCKS))],
            "receipt_path": _receipt,
            "n_games": (report or {}).get("n_games"),
            "n_pairs": (report or {}).get("n_pairs"),
            "verdict": (report or {}).get("verdict"),
            "interpretation_withheld": (report or {}).get("interpretation_withheld"),
            "is_strength_verdict": (report or {}).get("is_strength_verdict"),
        }
        wrote = write_receipt(receipt, _receipt)
        print(f"outcome={outcome} gate_restored={restored} "
              f"gate_readback={readback} receipt_written={wrote}")

    if not restored or readback != "False":
        print("🔴 THE COMBINATION GATE WAS NOT RESTORED. This supersedes the "
              "run's own outcome: the source is left able to combine again.")
        return EXIT_GATE_NOT_RESTORED
    return status


if __name__ == "__main__":
    sys.exit(main())
