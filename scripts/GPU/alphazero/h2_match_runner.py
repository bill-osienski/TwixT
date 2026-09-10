"""H2 — the execution wiring. THE GATE LIVES HERE AND IT IS SHUT.

The card is `docs/superpowers/2026-09-09-t1j-h2-deterministic-readout-card.md`
(frozen at 5aa5db4, amended through §2.2). This module holds the barriers, the
identity binding, the schedule and seed-position checks, the output safeguards,
the deadline, the cleanup, and the ordering that makes the degeneracy screen bind.

🔴 NOTHING HERE HAS RUN. `H2_EXECUTION_AUTHORIZED` is False, the seed block is
registered nowhere, and the play seam is injected -- the module loads no model,
starts no JVM and plays no game by itself.

THREE BARRIERS, and they are separate ON PURPOSE:
  1. the GATE below, a reviewed one-line change;
  2. `check_seed_registration`, which reads a registry it never writes;
  3. `check_output_paths`, which refuses before anything exists.
Registration does not open the gate and cannot: they are separate constants in
separate modules.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from . import e4_screen_reference as REF
from . import h2_match_plan as PLAN
from . import h2_match_rules as RULES


class H2Error(RuntimeError):
    """A refusal. Nothing was played."""


class H2VoidError(H2Error):
    """The instrument failed mid-run. No partial-cohort analysis is produced."""


#: 🔴 THE GATE. A reviewed one-line change plus a separate authorization, and
#: nothing reads an environment variable, a flag or a config file to reach it.
#: Both public entries read it, so gating one leaves the other reachable.
H2_EXECUTION_AUTHORIZED = False

#: Whole-run wall clock, and the per-call bound the helper already carries.
RUN_DEADLINE_S = 480 * 60                               # 8 hours, card §5
PER_CALL_TIMEOUT_S = 120

EXIT_OK = 0
EXIT_VOID = 3
EXIT_UNEXPECTED = 4
EXIT_UNAUTHORIZED = 5


def check_gate() -> None:
    """BARRIER 1. Read by every public entry, before anything else happens."""
    if not H2_EXECUTION_AUTHORIZED:
        raise H2Error(
            "H2 execution is UNAUTHORIZED. No model was loaded, no JVM started, no "
            "seed drawn, no game played and no file written. Opening this is a "
            "reviewed one-line change to H2_EXECUTION_AUTHORIZED plus a separate "
            "authorization.")


def check_seed_registration() -> None:
    """BARRIER 2. The reserved block must be REGISTERED before H2 draws from it.

    READS the registry; never writes one. Registering is a reviewed edit to
    `e4_screen_reference.ACCOUNTED_SEED_INTERVALS` and belongs to the H2 EXECUTION
    authorization -- a block reserved on paper and never authorized must cost
    nothing to abandon. A runtime mutation would make the registry something the
    run can grant itself, the same shape as a gate that opens its own gate.

    EVERY seed is checked, not the endpoints: a partial registration would
    otherwise pass and then draw an unaccounted seed halfway through.

    ⚠ Availability -- exposed, retired, consumed -- is a DIFFERENT question, asked
    per task by `REF.validate_schedule_executable`. That function does NOT ask the
    accounted question, which is exactly why this barrier exists separately.
    """
    lo, hi = RULES.H2_SEED_BLOCK
    missing = [s for s in range(lo, hi) if not REF.seed_is_accounted(s)]
    if missing:
        raise H2Error(
            f"the H2 seed block [{lo}, {hi}) is not registered: {len(missing)} of "
            f"{hi - lo} seeds are absent from ACCOUNTED_SEED_INTERVALS (first "
            f"{missing[0]}). Registering it is a reviewed edit to that registry, part "
            f"of the H2 EXECUTION authorization; nothing here writes a registry at "
            f"runtime.")


def _canonical(path: str) -> str:
    """Absolute, symlink-resolved, normalised. Two names for one file are one file."""
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def check_output_paths(results_path: str, trace_path: Optional[str]) -> None:
    """BARRIER 3. The outputs must not already exist, and must be two files.

    A path that already exists is a PRECONDITION failure -- nothing has run, no
    game was played, no seed drawn -- so it refuses BEFORE any output exists and
    writes no trace at all.
    """
    if not trace_path:
        raise H2Error(
            "H2 requires a trace path: the card freezes a create-only, non-analytic "
            "trace, and a match that cannot say how far it got is not the design that "
            "was preregistered. Nothing has been written.")
    if _canonical(results_path) == _canonical(trace_path):
        raise H2Error(
            f"the results and trace paths name ONE file ({results_path!r} and "
            f"{trace_path!r} canonicalise together); the trace would create it and the "
            f"recorder would then refuse it, turning a naming slip into a VOID that "
            f"spends every seed.")
    for label, path in (("results", results_path), ("trace", trace_path)):
        if os.path.exists(path):
            raise H2Error(
                f"the {label} path already exists: {path}. Outputs are create-only, so "
                f"an earlier run's file cannot be mistaken for this one's. Nothing has "
                f"run.")


#: The incumbent identity H2 requires, and the ONE field that differs from H1's.
def frozen_incumbent_identity() -> Dict[str, Any]:
    """The frozen incumbent, READ from the qualified path and never retyped, with
    the readout replaced by H2's.

    🔑 `selection_mode` IS THE STUDY. A recorded identity that still reads
    `opening_temperature` describes a run that did not make the change, and the
    binding below VOIDs it rather than reporting it as H2.

    The temperature settings are carried as NOT APPLICABLE rather than dropped or
    left looking active: a setting that no longer acts is not the same setting,
    and silently deleting it would hide which configuration ran.
    """
    from . import d1_probe as D1P
    base = D1P.frozen_incumbent_identity()
    cfg = dict(base["eval_config"])
    cfg["selection_mode"] = RULES.SELECTION_MODE
    inert = {k: cfg.pop(k) for k in RULES.INERT_UNDER_ARGMAX if k in cfg}
    out = dict(base)
    out["eval_config"] = cfg
    out["inert_under_argmax"] = inert
    out["readout_note"] = (
        "H2's ONE gameplay-rule change: the incumbent plays the visit-count maximum. "
        "The listed settings are INERT under argmax and are recorded as such, never "
        "as 'unchanged'.")
    return out


def check_incumbent_identity(identity: Mapping[str, Any]) -> None:
    """The recorded identity must BE H2's, field by field."""
    want = frozen_incumbent_identity()
    got_mode = (identity.get("eval_config") or {}).get("selection_mode")
    if got_mode != RULES.SELECTION_MODE:
        raise H2VoidError(
            f"the recorded incumbent identity carries selection_mode {got_mode!r}, not "
            f"{RULES.SELECTION_MODE!r}. H2 IS the readout change: a run recording the "
            f"old readout did not make it, and reporting it as H2 would attribute a "
            f"result to a configuration that never played.")
    for field in ("reference", "reference_sha1", "plan_sha256"):
        if identity.get(field) != want.get(field):
            raise H2VoidError(
                f"the recorded incumbent identity's {field!r} is {identity.get(field)!r}, "
                f"not the frozen {want.get(field)!r}")
    for field in ("mcts_sims", "board_size", "max_moves"):
        if (identity.get("eval_config") or {}).get(field) != want["eval_config"][field]:
            raise H2VoidError(
                f"the recorded eval_config's {field!r} disagrees with the frozen "
                f"configuration; this is not the H2 incumbent")


def check_schedule(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The supplied schedule must BE the frozen 736, in order, with its digest.

    🔑 A BUDGET IS A CEILING AND BOUNDS NOTHING BELOW. Without this a 700-task or
    a one-task schedule runs to completion and reports a verdict -- a short match
    wearing a finished run's clothes.
    """
    summary = PLAN.validate_h2_schedule(list(tasks))
    if summary["task_digest"] != RULES.H2_TASK_DIGEST:
        raise H2Error(
            f"the schedule digest is {summary['task_digest']} but the frozen H2 "
            f"schedule is {RULES.H2_TASK_DIGEST}. A different schedule is a different "
            f"experiment wearing this one's name.")
    REF.validate_schedule_executable(list(tasks))
    return summary


def run_h2(*, tasks: Sequence[Mapping[str, Any]], results_path: str, trace_path: str,
           play: Callable[..., Mapping[str, Any]],
           identity: Optional[Mapping[str, Any]] = None,
           deadline_s: float = RUN_DEADLINE_S) -> Dict[str, Any]:
    """THE PUBLIC ENTRY. Checks the gate FIRST, then every other barrier.

    `play` is the injected seam that actually plays one game. This module never
    imports one: the seam is what keeps the gate meaningful in tests, and a
    fixture that lifts an execution gate is the gate failing.
    """
    check_gate()
    return _run_h2_unguarded(tasks=tasks, results_path=results_path,
                             trace_path=trace_path, play=play, identity=identity,
                             deadline_s=deadline_s)


def _run_h2_unguarded(*, tasks, results_path, trace_path, play, identity=None,
                      deadline_s=RUN_DEADLINE_S) -> Dict[str, Any]:
    """Everything below the gate. PRIVATE, and never a way around `run_h2`.

    It exists so the machinery can be tested WITHOUT lifting the gate.
    """
    import json
    import time

    check_seed_registration()
    check_output_paths(results_path, trace_path)
    summary = check_schedule(tasks)
    ident = dict(identity) if identity is not None else frozen_incumbent_identity()
    check_incumbent_identity(ident)

    started = time.monotonic()
    results: List[Dict[str, Any]] = []
    per_game: List[Dict[str, Any]] = []
    tfd = os.open(trace_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    rfd = os.open(results_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(tfd, "w") as trace, os.fdopen(rfd, "w") as rec:
            def emit(fh, obj):
                fh.write(json.dumps(obj, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

            emit(trace, {"event": "run_start", "n_tasks": len(tasks),
                         "selection_mode": RULES.SELECTION_MODE})
            emit(rec, {"record_type": "header", "design": "H2",
                       "task_digest": summary["task_digest"],
                       "selection_mode": RULES.SELECTION_MODE, "identity": ident})
            for i, task in enumerate(tasks):
                if time.monotonic() - started > deadline_s:
                    emit(trace, {"event": "run_end", "verdict": "VOID",
                                 "games_completed": len(results)})
                    raise H2VoidError(
                        f"the {deadline_s / 60:.0f}-minute deadline expired at game {i} "
                        f"of {len(tasks)}; the run is VOID and no partial rate is "
                        f"reported -- a partial schedule is not a smaller design.")
                emit(trace, {"event": "task_start", "index": i})
                out = play(task=task, identity=ident, timeout_s=PER_CALL_TIMEOUT_S)
                row = dict(out["result"])
                results.append(row)
                emit(rec, {"record_type": "task_result", **row})
                t = RULES.transcript(out["plies"], row,
                                     opening_bound=out["opening_bound"],
                                     anchor_colour=task["anchor_colour"])
                per_game.append({"task_id": task["task_id"], "opening": task["opening"],
                                 "colour_arm": task["colour_arm"],
                                 "transcript_digest": RULES.transcript_digest(t)})
                emit(trace, {"event": "task_done", "index": i,
                             "games_completed": len(results)})
            emit(trace, {"event": "run_end", "verdict": "OK",
                         "games_completed": len(results)})
    except H2Error:
        raise
    except Exception as e:                                   # noqa: BLE001
        raise H2VoidError(f"{type(e).__name__}: {e}") from e

    if len(results) != RULES.N_GAMES:
        raise H2VoidError(
            f"{len(results)} of {RULES.N_GAMES} games completed; a partial schedule "
            f"produces no analysis.")
    # 🔑 THE SCREEN RUNS INSIDE THE REPORT, BEFORE THE INTERVAL. A screen that runs
    # after the number it guards is decoration -- see `h2_match_rules.h2_report`.
    return RULES.h2_report(results, list(tasks), per_game,
                           task_digest=summary["task_digest"])
