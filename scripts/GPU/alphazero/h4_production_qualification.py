"""H4 PRODUCTION-PATH QUALIFICATION (step 3P-b). GATED; NOT RUN.

Frozen by `docs/superpowers/2026-09-22-t1j-h4-runner-persistence-card.md` §12.2
and §12.6. 🔴 THE CARD IS THE AUTHORITY.

ONE run, on the real toolchain and checkpoint, with NO game, NO T1j query or
replay JVM, NO research seed and NO outcome. In order, stopping at the first
failure:

  1. the incumbent's identity, built by the production path and read off the
     object that would play, equals the frozen one type-strictly;
  2. the checkpoint loads through the production path (sha1 recomputed by the
     qualified loader, then its tags checked against that identity);
  3. the incumbent makes ONE legal decision from the empty board, under a named
     synthetic seed -- no T1j call, no second ply;
  4. the helper compiles TWICE, into two fresh directories, with IDENTICAL class
     hashes -- or class hashes could not be bound in advance: STOP;
  5. the pilot CLI, as a fresh subprocess, refuses while the pilot gate is shut;
  6. the bound identity (code, cards, T1j runtime content, incumbent identity) is
     built by the SAME function the pilot's header uses.

The result is CLEAN or STOP, written create-only and marked NOT evidence of
strength. A STOP is a result: nothing is repaired and rerun here.
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence

from . import d1_probe as D1
from . import h4_production_qualification_authorization as QAUTH
from . import h4_runner as R
from . import t1j_adapter as A

DESIGN = R.DESIGNS["pilot"]
#: A named SYNTHETIC seed (runner card §10.1): it reaches one incumbent decision
#: and nothing else, and can never collide with a research block.
SEED = -1
COMPILE_DEADLINE_S = 1_800
EXIT_CLEAN, EXIT_STOP, EXIT_REFUSED, EXIT_UNAUTHORIZED = 0, 2, 4, 5


class QualificationRefused(Exception):
    """Refused before anything ran: nothing was loaded, compiled or written."""


class QualificationUnauthorized(QualificationRefused):
    """The gate is closed."""


def _stop(reason: str):
    raise _Stop(reason)


class _Stop(Exception):
    pass


def _pilot_cli_refuses(tmp: str) -> int:
    r = subprocess.run([sys.executable, "-m", "scripts.GPU.alphazero.h4_runner",
                        "--mode", "pilot", "--manifest", os.path.join(tmp, "none.json"),
                        "--segment", "0", "--out-dir", os.path.join(tmp, "out"),
                        "--classes", os.path.join(tmp, "cls")],
                       cwd=str(R.REPO_ROOT), capture_output=True, text=True)
    return r.returncode


def _checks(classes_root: str) -> Dict[str, Any]:
    from .game.twixt_state import TwixtState
    got: Dict[str, Any] = {}
    config = R.frozen_argmax_config()
    identity = R.frozen_incumbent_identity(config, design=DESIGN)
    try:
        R.check_incumbent_identity(identity, config, design=DESIGN)
    except R.H4RunError as e:
        _stop(f"incumbent identity: {e}")
    got["incumbent_identity"] = identity
    try:
        evaluator = R.load_production_evaluator(identity)
    except R.H4RunError as e:
        _stop(f"checkpoint: {e}")
    got["checkpoint"] = {"reference": identity["reference"],
                         "reference_sha1": identity["reference_sha1"]}
    task = R.make_schedule([("h4-production-qualification", SEED, SEED - 1)],
                           mode="fixture", reference=identity)[0]
    state = TwixtState(active_size=A.BOARD_N, to_move="red")
    move = R.production_incumbent_build(config)(task, evaluator)(state)
    move = tuple(move) if move is not None else None
    if move is None or move not in set(state.legal_moves()):
        _stop(f"the incumbent returned {move}, not a legal empty-board move")
    got["incumbent_move"] = list(move)
    identities = []
    for leaf in ("compile-a", "compile-b"):
        paths = R._verified_t1j_paths(os.path.join(classes_root, leaf))
        R.check_destinations(os.path.join(classes_root, "unused-out"), paths.classes,
                             mode="pilot")
        deadline = D1.Deadline(limit_s=COMPILE_DEADLINE_S)
        deadline.start()
        with D1._supervisor(deadline):
            identities.append(R.split_toolchain(D1._default_compile(deadline, paths=paths)))
    (a, local_a), (b, local_b) = identities
    if a != b:
        diff = sorted(k for k in a if a[k] != b[k])
        _stop(f"the helper does not compile reproducibly: {diff} differ between two "
              f"compiles, so class hashes cannot be bound in advance")
    got["t1j_local"] = [local_a, local_b]
    rc = _pilot_cli_refuses(classes_root)
    if rc != R.EXIT_UNAUTHORIZED:
        _stop(f"the pilot CLI exited {rc}, not {R.EXIT_UNAUTHORIZED}, with its gate shut")
    got["pilot_cli_exit"] = rc
    got["bound"] = R.bound_identity(identity, a)
    return got


def _write_create_only(path: str, obj: Dict[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def _qualify(out_dir: str, classes_root: str) -> Dict[str, Any]:
    if os.path.lexists(out_dir):
        raise QualificationRefused(f"{out_dir} is occupied; the record is create-only")
    if R._inside(pathlib.Path(classes_root), R.REPO_ROOT):
        raise QualificationRefused(f"the class root must be OUTSIDE the repository: "
                                   f"{classes_root}")
    if os.path.lexists(classes_root):
        raise QualificationRefused(f"{classes_root} is occupied; compiles are create-only")
    os.makedirs(classes_root)
    try:
        checks, result, reason = _checks(classes_root), "CLEAN", None
    except _Stop as e:
        checks, result, reason = None, "STOP", str(e)
    except Exception as e:                        # noqa: BLE001 -- any failure is a STOP, recorded
        checks, result, reason = None, "STOP", f"{type(e).__name__}: {e}"
    record = {"record": "H4_PRODUCTION_QUALIFICATION", "result": result, "reason": reason,
              "checks": checks, "evidence_note": "NOT evidence of strength: no game was "
              "played and no outcome exists", "seed": SEED,
              "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    os.makedirs(out_dir)
    _write_create_only(os.path.join(out_dir, "record.json"), record)
    return record


def qualify(out_dir: str, classes_root: str) -> Dict[str, Any]:
    """PUBLIC ENTRY. The gate is read FIRST, before any file or process."""
    if not QAUTH.H4_PRODUCTION_QUALIFICATION_AUTHORIZED:
        raise QualificationUnauthorized("the H4 production qualification is UNAUTHORIZED. "
                                        "Nothing was loaded, compiled or written.")
    return _qualify(out_dir, classes_root)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """PUBLIC ENTRY."""
    import argparse
    ap = argparse.ArgumentParser(prog="h4_production_qualification",
                                 description="H4 3P-b qualification. NOT AUTHORIZED.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--classes-root", required=True)
    a = ap.parse_args(argv)
    if not QAUTH.H4_PRODUCTION_QUALIFICATION_AUTHORIZED:
        print("the H4 production qualification is UNAUTHORIZED. Nothing was loaded, "
              "compiled or written.", file=sys.stderr)
        return EXIT_UNAUTHORIZED
    try:
        rec = qualify(a.out_dir, a.classes_root)
    except QualificationRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"{rec['result']}: {os.path.join(a.out_dir, 'record.json')}")
    return EXIT_CLEAN if rec["result"] == "CLEAN" else EXIT_STOP


if __name__ == "__main__":                                                        # pragma: no cover
    raise SystemExit(main())
