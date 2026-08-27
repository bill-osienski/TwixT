"""The L0 larger match's execution harness. THE MATCH IS NOT AUTHORIZED.

L1 wires the frozen 64-task plan to machinery that is ALREADY QUALIFIED and does
not rebuild any of it. Everything effectful is imported from the E4 screen's
harness, which was qualified over five attempts and is left BYTE-UNCHANGED here:

    play_task            the per-ply loop: opening bound first, one stateful agent
                         per colour per task, move validated before application,
                         both engines bound every ply, external ply cap
    Recorder             exclusive-create, flushed and fsynced per record
    AbortError/PHASE_*   the abort classification and its exit mapping
    _enforce_evaluator   reference-side-only evaluator identity
    _assert_not_scheduled, _refuse_* the fail-closed defaults

WHY A SEPARATE RUNNER AT ALL
The screen's `_run` is bound to the screen: 32 tasks, two endpoints, a band, and
an EARLY STOP. L0 is 64 tasks, one endpoint, no band and NO early stop. Widening
`_run` to serve both would put a published, frozen artifact's behaviour behind a
mode flag, and the screen's canonical run is history that must keep reproducing.
So the LOOP is reused and the SCHEDULE POLICY is written here.

WHAT IS DELIBERATELY ABSENT
No early stop of any kind. `l0_match_rules.may_stop_early` is a constant False and
this module never consults a band, a saturation rule or an incompleteness rule.
The screen's `early_in_band_forced`, `saturation_reachable`,
`cap_incompleteness_reachable`, `per_endpoint_decision` and `classify_joint` are
not imported, and a test asserts they are not.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import e4_screen_reference as REF
from . import e4_screen_runner as H
from . import l0_match_plan as PLAN
from . import l0_match_rules as RULES

#: PUBLIC modes. The match is NOT selectable here: it is reachable only through
#: the private path, whose one authorized caller is the L0 command.
MODES = ("qualify",)
MATCH_MODE = "match"
_ALL_MODES = MODES + (MATCH_MODE,)

#: The frozen no-rate outcome, MIRRORED from `l0_match_rules.match_report`.
#:
#: It is duplicated rather than imported because the rules module names it only
#: as a literal, and L1 is not permitted to modify the frozen statistical rules to
#: add a constant. The duplication is bound by
#: `test_the_cap_saturated_outcome_string_matches_what_the_rules_emit`, which runs
#: the real reporter over a cap-heavy vector and compares -- so a drift is a test
#: failure rather than a silently mislabelled outcome.
CAP_SATURATED_NO_RATE = "CAP_SATURATED_NO_RATE"


def _assert_match_seed(task: Dict[str, Any]) -> None:
    """Match mode permits exactly the reserved L0 block, and nothing else."""
    lo, hi = PLAN.L0_SEED_BLOCK
    seed = int(task["seed"])
    if not (lo <= seed < hi):
        raise H.HarnessError(
            f"{task['task_id']} carries seed {seed}, outside the reserved L0 block "
            f"[{lo}, {hi})")


def _assert_match_executable(tasks: Sequence[Dict[str, Any]]) -> None:
    """EXECUTION eligibility, asked only on the path that would actually play.

    Structure is not permission. The frozen plan stays verifiable forever; whether
    its seeds may still be drawn from is a different question, asked here.
    """
    try:
        REF.validate_schedule_executable(tasks)
    except REF.E4ReferenceError as e:
        raise H.HarnessError(f"the L0 schedule may not be executed: {e}") from None


def _verify_match_schedule(tasks: Sequence[Dict[str, Any]], plan: Dict[str, Any]) -> None:
    """The run must execute THE FROZEN 64, in order, unedited."""
    try:
        PLAN.validate_l0_schedule(tasks)
    except PLAN.L0PlanError as e:
        raise H.HarnessError(str(e)) from None
    digest = RULES.l0_task_digest(tasks)
    if digest != RULES.L0_TASK_DIGEST:
        raise H.HarnessError(
            f"task digest {digest} != pinned {RULES.L0_TASK_DIGEST}: the schedule has "
            f"been added to, removed from, reordered or edited")
    # Bound by FULL CONTENT against the plan, not by task_id: canonical NAMES
    # attached to synthetic CONTENT counted in an earlier workstream, and one of
    # its own tests proved exactly that.
    by_id = {t["task_id"]: t for t in plan["tasks"]}
    for t in tasks:
        c = by_id.get(t["task_id"])
        if c is None or any(t.get(k) != c.get(k) for k in RULES.L0_TASK_DIMENSIONS):
            raise H.HarnessError(
                f"{t.get('task_id')} does not match the frozen plan's task of that name")
    if len(tasks) != len(plan["tasks"]):
        raise H.HarnessError(
            f"the run schedules {len(tasks)} tasks, the plan freezes {len(plan['tasks'])}")


def run(plan_path: str, results_path: str, *, mode: str = "qualify") -> int:
    """PUBLIC ENTRY POINT. Paths only.

    No task, callable, evaluator, cleanup hook, reporter or schedule can be
    injected, and MATCH MODE IS NOT SELECTABLE HERE.
    """
    if mode not in MODES:
        raise H.HarnessError(
            f"mode {mode!r} is not permitted; the 64-game match is UNAUTHORIZED")
    return _run(plan_path, results_path, mode=mode)


def _run(plan_path: str, results_path: str, *, mode: str,
         _tasks: Optional[Sequence[Dict[str, Any]]] = None,
         _agent_factory: Optional[Callable] = None,
         _state_factory: Optional[Callable] = None,
         _binder: Optional[Callable] = None,
         _evaluator: Any = None,
         _cleanup: Optional[Callable] = None,
         _identity: Optional[Dict[str, Any]] = None,
         _setup: Optional[Callable] = None,
         _ply_cap: int = RULES.PLY_CAP,
         _ply_budget: Optional[int] = None) -> int:
    """PRIVATE. The underscore parameters exist ONLY to drive fail-closed tests."""
    if mode not in _ALL_MODES:
        raise H.HarnessError(
            f"mode {mode!r} is not permitted; the 64-game match is UNAUTHORIZED")
    plan = PLAN.load_l0_plan(plan_path)
    tasks = list(_tasks) if _tasks is not None else []
    match = mode == MATCH_MODE
    if match:
        _verify_match_schedule(tasks, plan)
        for t in tasks:
            _assert_match_seed(t)
        _assert_match_executable(tasks)
    else:
        # Qualification runs on synthetic or TEST-ONLY seeds. Test-only seeds are
        # drawn from freely by design and can never enter a schedule, so they are
        # legitimate here and refused by _assert_match_executable above.
        for t in tasks:
            H._assert_not_scheduled(t)

    binder = _binder or H._refuse_binder
    cleanup = _cleanup or H._default_cleanup
    rec = H.Recorder(results_path)
    results: List[Dict[str, Any]] = []
    cleanups = 0
    try:
        rec.emit({"record_type": "run_header", "mode": mode,
                  "harness": "l0_match_runner",
                  "plan_sha256": PLAN.L0_PLAN_SHA256,
                  "task_digest": RULES.L0_TASK_DIGEST,
                  "frozen_tasks": len(plan["tasks"]),
                  "scheduled_tasks": len(tasks),
                  "synthetic_tasks": 0 if match else len(tasks),
                  "ply_cap": _ply_cap, "ply_budget": _ply_budget,
                  "no_games": not match,
                  "early_stop": RULES.EARLY_STOP,
                  "n_games": RULES.N_GAMES,
                  # THE VERIFIED IDENTITIES, fsynced BEFORE any setup runs, so a
                  # setup failure still leaves a durable record of what this was.
                  "identity": _identity or {}})

        if _setup is not None:
            try:
                collaborators = _setup()
            except H.AbortError:
                raise
            except Exception as e:                            # noqa: BLE001
                raise H.AbortError(H.PHASE_SETUP, f"{type(e).__name__}: {e}") from None
            _agent_factory = collaborators["agent_factory"]
            _state_factory = collaborators["state_factory"]
            binder = collaborators["binder"]
            _evaluator = collaborators["evaluator"]
            cleanup = collaborators.get("cleanup", cleanup)
            rec.emit({"record_type": "setup_complete",
                      "artifacts": collaborators.get("artifacts", {})})

        for task in tasks:
            # NO EARLY STOP, AND NO SKIP PATH. Every scheduled task is played.
            rec.emit({"record_type": "task_start", "task_id": task["task_id"],
                      "seed": task["seed"], "opening": task["opening"],
                      "colour_arm": task["colour_arm"], "rep": task["rep"]})

            def agent_for(t, mover, _task=task):
                agent = (_agent_factory or H._refuse_factory)(_task, mover, _evaluator)
                H._enforce_evaluator(agent, _evaluator, _task, mover)
                return agent

            play_error = None
            outcome = None
            try:
                outcome = H.play_task(task=task, agent_for=agent_for,
                                      state_factory=_state_factory or H._refuse_state_factory,
                                      binder=binder, rec=rec, ply_cap=_ply_cap,
                                      ply_budget=_ply_budget)
            except BaseException as e:                        # noqa: BLE001
                play_error = e

            # A COMPLETED GAME IS PERSISTED AND COUNTED BEFORE CLEANUP RUNS, and a
            # failure to WRITE it must not skip cleanup either. Both orderings were
            # defects in the screen's harness before they were fixed there.
            record_error = None
            if play_error is None:
                row = {"record_type": "task_result", "task_id": task["task_id"],
                       "seed": task["seed"], **outcome}
                try:
                    rec.emit(row)
                    results.append(row)          # counted only once it is durable
                except Exception as e:                        # noqa: BLE001
                    record_error = e

            cleanup_error = None
            try:
                cleanup()
                cleanups += 1
            except Exception as e:                            # noqa: BLE001
                cleanup_error = e

            primary = play_error if play_error is not None else record_error
            if primary is not None:
                if cleanup_error is not None:
                    rec.emit_terminal({"record_type": "cleanup_failure_after_abort",
                                       "task_id": task["task_id"],
                                       "cleanup_error": f"{type(cleanup_error).__name__}: "
                                                        f"{cleanup_error}"})
                if primary is record_error:
                    raise H.AbortError(
                        H.PHASE_RECORD,
                        f"{task['task_id']} finished but its result could not be "
                        f"recorded: {type(record_error).__name__}: {record_error}")
                raise primary
            if cleanup_error is not None:
                raise H.AbortError(H.PHASE_CLEANUP,
                                   f"after {task['task_id']} (whose result is recorded): "
                                   f"{cleanup_error}")

        return _report(rec, plan, tasks, results, mode, cleanups)
    except H.AbortError as e:
        note = rec.emit_terminal({"record_type": "abort", "phase": e.phase,
                                  "message": e.message, "tasks_played": len(results)})
        if note:
            import sys
            print(f"WARNING: could not record the abort ({note})", file=sys.stderr)
        raise
    finally:
        rec.close()


def _report(rec, plan, tasks, results, mode, cleanups) -> int:
    """Post-run reporting, delegated ENTIRELY to the frozen L0 rules.

    This module computes no rate, no interval and no verdict of its own. It hands
    the durable rows and the FROZEN plan tasks to `l0_match_rules.match_report`.

    THE MODE DECIDES WHAT A REFUSAL MEANS, and an earlier version did not:
    every `reported=False` became a qualification receipt and exited 0, in EVERY
    mode. So an incomplete or malformed CANONICAL match would have failed
    reporting and still exited successfully, and the frozen CAP_SATURATED_NO_RATE
    outcome would have been mislabelled as a qualification artefact.

      qualify   a refusal is EXPECTED -- synthetic tasks are not the frozen
                design -- and is recorded as a receipt. Exit 0.
      match     CAP_SATURATED_NO_RATE is a FROZEN, PREREGISTERED OUTCOME: it is
                recorded as a match outcome in its own right and exits 0, because
                the match ran correctly and the preregistered rule says there is
                no rate. ANY OTHER refusal means the run did not produce the
                design it claimed to, and aborts under PHASE_CLASSIFY with a
                nonzero status.

    A `qualification_receipt` can therefore never appear in a match run.
    """
    try:
        report = RULES.match_report(results, plan["tasks"])
    except Exception as e:                                    # noqa: BLE001
        raise H.AbortError(H.PHASE_CLASSIFY, f"{type(e).__name__}: {e}") from None

    if report.get("reported"):
        rec.emit({"record_type": "match_report", **report,
                  "tasks_played": len(results), "cleanups": cleanups})
        return H.EXIT_OK

    if mode == MATCH_MODE:
        if report.get("outcome") == CAP_SATURATED_NO_RATE:
            rec.emit({"record_type": "match_outcome",
                      "outcome": report["outcome"], "reason": report.get("reason"),
                      "cap_terminations": report.get("cap_terminations"),
                      "games": report.get("games"),
                      "tasks_played": len(results), "cleanups": cleanups})
            return H.EXIT_OK
        raise H.AbortError(
            H.PHASE_CLASSIFY,
            f"the match produced no report and no preregistered outcome: "
            f"{report.get('reason')}")

    rec.emit({"record_type": "qualification_receipt", "mode": mode,
              "report_withheld": report.get("reason"),
              "outcome": report.get("outcome"),
              "tasks_scheduled": len(tasks), "tasks_played": len(results),
              "cleanups": cleanups})
    return H.EXIT_OK
