"""E4 endpoint-screen execution harness — the production play loop.

**THE 32-GAME SCREEN IS UNAUTHORIZED.** `--mode screen` is refused before
anything is opened, and no code path reaches it. Qualification runs this same
control flow with fake agents, fake states and a fake per-ply binder.

WHAT THIS MODULE OWNS. The loop that plays a task is here, not in a caller's
stub: it advances the state, alternates the two agents by colour, validates every
move BEFORE applying it, calls the per-ply binder after applying it, enforces the
external ply cap, scores the result, and stops an endpoint as soon as IN_BAND is
forced. An earlier version delegated the whole outcome to an injected opponent,
which meant the loop under test was the test's, not the harness's.

FAIL-CLOSED, EVERYWHERE.
  * the results path must NOT already exist -- an append would silently merge two runs;
  * every record is flushed AND fsynced before the next step, so a kill leaves a
    truthful prefix rather than a buffer;
  * every construction is checked against the ONE expected evaluator immediately,
    and a mismatch aborts the run rather than being counted;
  * failures are classified by phase, a terminal record is written when it can be,
    and a failure to write evidence never masks the error that caused it.

Exit codes are the gate: 0 ok, 2 precondition refused, 3 aborted mid-run, 4
unexpected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import e4_screen_reference as REF
from .e4_screen_rules import (
    LARGER_MATCH_PERMITTED, classify_joint, early_in_band_forced, per_endpoint_decision,
)

CANONICAL_PLAN_SHA256 = "10cd8c3156de7f8a6aec87b0b62318b6bb56c7a32fca56b76eb8448bfc065ac8"
CANONICAL_TASK_DIGEST = "f5a21395e67ad130a8e57d9219a435afce0be23cbc14828f56a22a70a842802b"
CANONICAL_N_TASKS = 32
CANONICAL_N_PER_ENDPOINT = 16
PLY_CAP = 280
BAND = (0.05, 0.95)

TASK_DIMENSIONS = ("task_id", "endpoint", "t1j_mdPly", "t1j_mdFixedPly", "opening",
                   "colour_arm", "anchor_colour", "reference", "reference_sha1",
                   "reference_colour", "seed")

MODES = ("qualify",)                 # "screen" is absent: it is UNAUTHORIZED

EXIT_OK, EXIT_PRECONDITION, EXIT_ABORT, EXIT_UNEXPECTED = 0, 2, 3, 4

PHASE_PRECONDITION = "precondition"
PHASE_FACTORY = "agent_construction"
PHASE_MOVE = "move"
PHASE_BIND = "per_ply_binding"
PHASE_RECORD = "result_recording"
PHASE_CLEANUP = "cleanup"
PHASE_CLASSIFY = "classification"


class HarnessError(Exception):
    """Refused before anything ran. Maps to EXIT_PRECONDITION."""


class AbortError(Exception):
    """Aborted mid-run. Carries the phase it died in. Maps to EXIT_ABORT."""

    def __init__(self, phase: str, message: str):
        super().__init__(f"[{phase}] {message}")
        self.phase = phase
        self.message = message


# ----------------------------------------------------------------- schedule

def task_digest(tasks: Sequence[Dict[str, Any]]) -> str:
    try:
        payload = json.dumps([[t[k] for k in TASK_DIMENSIONS] for t in tasks],
                             separators=(",", ":"))
    except KeyError as e:
        raise HarnessError(f"task is missing dimension {e}") from None
    return hashlib.sha256(payload.encode()).hexdigest()


def load_canonical_plan(plan_path: str) -> Dict[str, Any]:
    try:
        raw = open(plan_path, "rb").read()
    except OSError as e:
        raise HarnessError(f"cannot read the plan: {e}") from None
    got = hashlib.sha256(raw).hexdigest()
    if got != CANONICAL_PLAN_SHA256:
        raise HarnessError(f"plan sha256 {got} != pinned {CANONICAL_PLAN_SHA256}")
    plan = json.loads(raw)
    verify_tasks(plan.get("tasks", []))
    return plan


def verify_tasks(tasks: Sequence[Dict[str, Any]]) -> None:
    """Refuse additions, removals, reordering, duplicates and edits alike."""
    if len(tasks) != CANONICAL_N_TASKS:
        raise HarnessError(f"{len(tasks)} tasks, expected exactly {CANONICAL_N_TASKS}")
    ids = [t.get("task_id") for t in tasks]
    if len(set(ids)) != len(ids):
        raise HarnessError("duplicate task_id in the schedule")
    got = task_digest(tasks)
    if got != CANONICAL_TASK_DIGEST:
        raise HarnessError(
            f"task digest {got} != pinned {CANONICAL_TASK_DIGEST}: the schedule has been "
            f"added to, removed from, reordered or edited")
    REF.validate_schedule(tasks)


def _assert_not_scheduled(task: Dict[str, Any]) -> None:
    seed = int(task["seed"])
    if REF.seed_is_accounted(seed) or REF.seed_is_exposed(seed):
        raise HarnessError(
            f"seed {seed} is accounted or exposed; qualification runs on synthetic seeds only")


# ---------------------------------------------------------------- recording

class Recorder:
    """Exclusive-create, fsynced per record.

    Append mode would silently merge two runs into one file, and line buffering
    is not durability: it hands bytes to the OS, which may still lose them. Each
    record is flushed and fsynced before the run proceeds, so a kill leaves a
    truthful prefix.
    """

    def __init__(self, path: str):
        try:
            self._f = open(path, "x", buffering=1)          # x: refuse an existing file
        except FileExistsError:
            raise HarnessError(
                f"results path already exists: {path}. A run writes a NEW file; appending "
                f"would merge two runs.") from None
        except OSError as e:
            raise HarnessError(f"cannot create the results file: {e}") from None
        self.path = path
        self.n = 0

    def emit(self, record: Dict[str, Any]) -> None:
        self._f.write(json.dumps(record, sort_keys=True) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())
        self.n += 1

    def emit_terminal(self, record: Dict[str, Any]) -> Optional[str]:
        """Best-effort terminal record. NEVER masks the error that caused it."""
        try:
            self.emit(record)
            return None
        except Exception as e:                                # noqa: BLE001
            return f"{type(e).__name__}: {e}"

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:                                     # noqa: BLE001
            pass


# ------------------------------------------------------------- the play loop

def _refuse_binder(task, state, ply_index):
    raise AbortError(PHASE_BIND,
                     "no per-ply binder configured; the screen must bind both engines every ply")


def _bind(binder: Callable, task: Dict[str, Any], state, ply: int, where: str) -> None:
    """Every binder call goes through here, so NO binder failure escapes unclassified.

    An earlier version only handled AbortError, so a plain ValueError from a
    binder propagated as an unexpected error and left no durable abort record.
    """
    try:
        binder(task, state, ply)
    except AbortError:
        raise
    except Exception as e:                                    # noqa: BLE001
        raise AbortError(PHASE_BIND,
                         f"{task['task_id']} {where}: binder raised "
                         f"{type(e).__name__}: {e}") from None


def play_task(*, task: Dict[str, Any], agent_for: Callable, state_factory: Callable,
              binder: Callable, rec: Recorder, ply_cap: int = PLY_CAP) -> Dict[str, Any]:
    """Play ONE task to a terminal state. The loop lives here, not in a caller.

    THE OPENING IS BOUND FIRST. The scripted opening is a position both engines
    must already agree on; binding only from ply 7 would let a divergent opening
    run a whole game before anyone noticed. It is bound before the terminal check
    and before either agent is constructed.

    Per ply: pick the agent whose colour is to move, ask it for a move, VALIDATE
    the move before applying it, apply it, then bind both engines' states. The
    cap is external and applies to our own ply counter.
    """
    try:
        state = state_factory(task)
    except AbortError:
        raise
    except Exception as e:                                    # noqa: BLE001
        raise AbortError(PHASE_PRECONDITION, f"cannot build the opening state: {e}") from None

    _bind(binder, task, state, state.ply, "opening")          # BEFORE anything else
    rec.emit({"record_type": "opening_bound", "task_id": task["task_id"],
              "ply": state.ply, "opening": task.get("opening")})

    anchor_colour = task["anchor_colour"]
    # ONE agent per colour per task, built on first use and reused for the whole
    # game. SeededReferenceAgent is stateful by contract -- both RNG streams
    # advance across the game -- so rebuilding it each ply would silently reset
    # the seeding and make the per-task seed meaningless.
    agents: Dict[str, Any] = {}
    while True:
        winner = state.winner()
        if winner is not None:
            reason = "win"
            break
        if state.ply >= ply_cap:
            winner, reason = None, "cap"
            break

        mover = state.to_move
        if mover not in agents:
            try:
                agents[mover] = agent_for(task, mover)
            except AbortError:
                raise
            except Exception as e:                            # noqa: BLE001
                raise AbortError(PHASE_FACTORY,
                                 f"{task['task_id']} ply {state.ply}: {e}") from None
        agent = agents[mover]

        try:
            move = agent(state)
        except AbortError:
            raise
        except Exception as e:                                # noqa: BLE001
            raise AbortError(PHASE_MOVE,
                             f"{task['task_id']} ply {state.ply}: {mover} raised {e}") from None

        move = tuple(move) if move is not None else None
        if move is None or move not in set(state.legal_moves()):
            raise AbortError(PHASE_MOVE,
                             f"{task['task_id']} ply {state.ply}: {mover} returned {move}, "
                             f"which is not legal")
        state = state.apply_move(move)
        _bind(binder, task, state, state.ply, f"ply {state.ply}")
        rec.emit({"record_type": "ply", "task_id": task["task_id"], "ply": state.ply,
                  "mover": mover, "move": list(move)})

    points = 0.5 if reason == "cap" else (1.0 if winner == anchor_colour else 0.0)
    return {"winner": winner, "terminal_reason": reason, "plies": state.ply,
            "t1j_points": points, "agents_built": len(agents)}


# --------------------------------------------------------------- the runner

def classify_run(results: Sequence[Dict[str, Any]], *, n_per_endpoint: int,
                 band: Sequence[float]) -> Dict[str, Any]:
    """Per-endpoint decisions and the joint outcome. Partial results get no verdict."""
    out = {}
    for endpoint in ("weak", "strong"):
        rows = [r for r in results if r.get("endpoint") == endpoint]
        score = sum(float(r["t1j_points"]) for r in rows)
        caps = sum(1 for r in rows if r.get("terminal_reason") == "cap")
        out[endpoint] = {
            "played": len(rows), "score": score, "cap_terminations": caps,
            "decision": per_endpoint_decision(score, len(rows), n_per_endpoint,
                                              list(band), caps)}
    joint = classify_joint(out["weak"]["decision"], out["strong"]["decision"])
    return {"per_endpoint": out, "joint": joint,
            "larger_match_permitted": joint in LARGER_MATCH_PERMITTED}


def screen_verdict_allowed(canonical_tasks, tasks, results, skipped, stopped):
    """May a SCREEN VERDICT be drawn? It must bind to the CANONICAL SCHEDULE.

    An earlier version checked completeness against whatever task list it was
    handed, so ONE weak task and its result counted as a complete screen. A
    verdict now requires the run to have executed the verified canonical schedule
    itself -- both endpoints, the frozen identities -- with every task either
    played or skipped by a recorded early stop for its own endpoint.

    Synthetic qualification runs therefore get a RECEIPT, always: their
    identities are not the canonical ones, and that is the point.
    """
    if not tasks:
        return False, "no tasks were scheduled; a screen verdict needs a screen"
    canonical = [t["task_id"] for t in canonical_tasks]
    scheduled = [t["task_id"] for t in tasks]
    if len(set(scheduled)) != len(scheduled):
        return False, "the scheduled tasks contain duplicate identities"
    if len(scheduled) != len(canonical):
        return False, (
            f"the run executed {len(scheduled)} task(s), not the canonical schedule of "
            f"{len(canonical)}; a screen verdict binds to the frozen schedule only")
    # THE WHOLE SCHEDULE, ORDER AND CONTENTS. Comparing task_id SETS accepted the
    # 32 canonical names in reverse order, and accepted them with a seed edited --
    # both reproduced. verify_tasks re-runs the pinned ORDERED, dimension-projected
    # digest, so endpoint, depth, opening, colours, reference identity and seed are
    # all bound, not merely the names.
    try:
        verify_tasks(tasks)
    except HarnessError as e:
        return False, f"the run did not execute the canonical schedule: {e}"
    endpoints = {t["endpoint"] for t in tasks}
    if endpoints != {"weak", "strong"}:
        return False, f"a screen needs BOTH endpoints; this run covered {sorted(endpoints)}"
    got = [r["task_id"] for r in results] + [k["task_id"] for k in skipped]
    if len(set(got)) != len(got):
        return False, "a task identity appears more than once in the results"
    alien = sorted(set(got) - set(canonical))
    if alien:
        return False, f"results contain identities that are not in the schedule: {alien[:3]}"
    missing = sorted(set(canonical) - set(got))
    if missing:
        return False, f"{len(missing)} scheduled task(s) neither played nor skipped: {missing[:3]}"
    for k in skipped:
        if k["endpoint"] not in stopped:
            return False, f"{k['task_id']} was skipped but its endpoint recorded no early stop"
    return True, None


def run(plan_path: str, results_path: str, *, mode: str = "qualify") -> int:
    """PUBLIC ENTRY POINT. Paths only.

    No task, callable, evaluator, cleanup hook, classifier or schedule can be
    injected. The private `_run` carries those, for fail-closed tests only.
    """
    return _run(plan_path, results_path, mode=mode)


def _run(plan_path: str, results_path: str, *, mode: str,
         _tasks: Optional[Sequence[Dict[str, Any]]] = None,
         _agent_factory: Optional[Callable] = None,
         _state_factory: Optional[Callable] = None,
         _binder: Optional[Callable] = None,
         _evaluator: Any = None,
         _cleanup: Optional[Callable] = None,
         _n_per_endpoint: Optional[int] = None,
         _ply_cap: int = PLY_CAP,
         _band: Sequence[float] = BAND) -> int:
    """PRIVATE. The underscore parameters exist ONLY to drive fail-closed tests."""
    if mode not in MODES:
        raise HarnessError(
            f"mode {mode!r} is not permitted; the 32-game screen is UNAUTHORIZED")
    plan = load_canonical_plan(plan_path)          # verified, then inert
    tasks = list(_tasks) if _tasks is not None else []
    for t in tasks:
        _assert_not_scheduled(t)

    n_per = _n_per_endpoint or CANONICAL_N_PER_ENDPOINT
    binder = _binder or _refuse_binder
    cleanup = _cleanup or _default_cleanup
    rec = Recorder(results_path)
    results: List[Dict[str, Any]] = []
    stopped: Dict[str, str] = {}
    skipped: List[Dict[str, str]] = []
    cleanups = 0
    try:
        rec.emit({"record_type": "run_header", "mode": mode,
                  "plan_sha256": CANONICAL_PLAN_SHA256,
                  "task_digest": CANONICAL_TASK_DIGEST,
                  "canonical_tasks": len(plan["tasks"]), "canonical_tasks_executed": 0,
                  "synthetic_tasks": len(tasks), "ply_cap": _ply_cap,
                  "n_per_endpoint": n_per, "no_games": True})
        for task in tasks:
            endpoint = task["endpoint"]
            if endpoint in stopped:
                rec.emit({"record_type": "task_skipped", "task_id": task["task_id"],
                          "endpoint": endpoint, "reason": stopped[endpoint]})
                skipped.append({"task_id": task["task_id"], "endpoint": endpoint})
                continue
            rec.emit({"record_type": "task_start", "task_id": task["task_id"],
                      "endpoint": endpoint, "seed": task["seed"]})

            def agent_for(t, mover, _task=task):
                agent = (_agent_factory or _refuse_factory)(_task, mover, _evaluator)
                _enforce_evaluator(agent, _evaluator, _task, mover)
                return agent

            # cleanup runs ONCE PER STARTED TASK, including one that aborted --
            # an aborted game has already allocated whatever the cleanup releases
            play_error = None
            outcome = None
            try:
                outcome = play_task(task=task, agent_for=agent_for,
                                    state_factory=_state_factory or _refuse_state_factory,
                                    binder=binder, rec=rec, ply_cap=_ply_cap)
            except BaseException as e:                        # noqa: BLE001
                play_error = e
            # A COMPLETED GAME IS PERSISTED AND COUNTED BEFORE CLEANUP RUNS.
            # Recording it afterwards meant a cleanup failure erased a game that
            # had already finished: every ply in the log, then an abort, and
            # tasks_played = 0.
            # A failure to WRITE the result must not skip cleanup either: an
            # earlier version emitted the row outside the cleanup structure, so a
            # full disk or a bad descriptor left control immediately and cleanup
            # never ran, contradicting the once-per-started-task guarantee.
            record_error = None
            if play_error is None:
                row = {"record_type": "task_result", "task_id": task["task_id"],
                       "endpoint": endpoint, "seed": task["seed"], **outcome}
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
                    # recorded, never allowed to replace the failure that caused it
                    rec.emit_terminal({"record_type": "cleanup_failure_after_abort",
                                       "task_id": task["task_id"],
                                       "cleanup_error": f"{type(cleanup_error).__name__}: "
                                                        f"{cleanup_error}"})
                if primary is record_error:
                    raise AbortError(PHASE_RECORD,
                                     f"{task['task_id']} finished but its result could not be "
                                     f"recorded: {type(record_error).__name__}: {record_error}")
                raise primary
            if cleanup_error is not None:
                # the game record above is already durable; this is a RUN-level abort
                raise AbortError(PHASE_CLEANUP,
                                 f"after {task['task_id']} (whose result is recorded): "
                                 f"{cleanup_error}")

            rows = [r for r in results if r["endpoint"] == endpoint]
            score = sum(r["t1j_points"] for r in rows)
            caps = sum(1 for r in rows if r["terminal_reason"] == "cap")
            if early_in_band_forced(score, len(rows), n_per, list(_band), caps):
                stopped[endpoint] = (f"IN_BAND forced after {len(rows)} of {n_per} games "
                                     f"(score {score}, {caps} cap terminations)")
                rec.emit({"record_type": "early_stop", "endpoint": endpoint,
                          "played": len(rows), "score": score, "reason": stopped[endpoint]})

        allowed, why = screen_verdict_allowed(plan["tasks"], tasks, results, skipped, stopped)
        if not allowed:
            # A RECEIPT, NOT A VERDICT. An earlier version classified zero results
            # and emitted joint=INCONCLUSIVE with larger_match_permitted=true --
            # a screen conclusion drawn from no games at all.
            rec.emit({"record_type": "qualification_receipt", "mode": mode,
                      "verdict_withheld": why, "tasks_scheduled": len(tasks),
                      "tasks_played": len(results), "tasks_skipped": len(skipped),
                      "cleanups": cleanups})
            return EXIT_OK
        try:
            verdict = classify_run(results, n_per_endpoint=n_per, band=_band)
        except Exception as e:                                # noqa: BLE001
            raise AbortError(PHASE_CLASSIFY, str(e)) from None
        rec.emit({"record_type": "verdict", **verdict, "early_stopped": stopped,
                  "tasks_played": len(results), "cleanups": cleanups})
        return EXIT_OK
    except AbortError as e:
        note = rec.emit_terminal({"record_type": "abort", "phase": e.phase,
                                  "message": e.message, "tasks_played": len(results)})
        if note:
            print(f"WARNING: could not record the abort ({note})", file=sys.stderr)
        raise
    finally:
        rec.close()


def _enforce_evaluator(agent, expected, task, mover) -> None:
    """Identity, checked at construction, for the REFERENCE SIDE ONLY.

    T1j is a classical engine: it holds no MLX evaluator and never will, so
    applying this gate to both colours would abort on the first T1j construction.
    The gate belongs to the side that actually loads the network.
    """
    reference_colour = task.get("reference_colour")
    if reference_colour is None:
        raise AbortError(PHASE_FACTORY,
                         f"{task.get('task_id')}: the task names no reference_colour, so the "
                         f"reference side cannot be identified")
    if mover != reference_colour:
        return                                                # the classical anchor side
    if expected is None:
        # NO OFF SWITCH. Forgetting to load or pass the evaluator must not look
        # the same as deliberately disabling the check.
        raise AbortError(PHASE_FACTORY,
                         f"{task.get('task_id')} ({mover}, the reference side): no expected "
                         f"evaluator was supplied, so identity cannot be checked")
    got = _evaluator_of(agent, None)
    if got is not expected:
        raise AbortError(
            PHASE_FACTORY,
            f"{task['task_id']} ({mover}, the reference side): the agent holds "
            f"{'no evaluator' if got is None else 'a DIFFERENT evaluator object'} rather than "
            f"the one loaded for this run; the reference must be loaded once and reused")


def _evaluator_of(agent, fallback):
    mcts = getattr(agent, "mcts", None)
    if mcts is not None and hasattr(mcts, "evaluator"):
        return mcts.evaluator
    return getattr(agent, "evaluator", fallback)


def _default_cleanup() -> None:
    from .twixtbot_g3_reference import between_games_cleanup
    between_games_cleanup()


def _refuse_factory(task, mover, evaluator):
    raise AbortError(PHASE_FACTORY, "no agent factory: the public runner builds no agents")


def _refuse_state_factory(task):
    raise AbortError(PHASE_PRECONDITION, "no state factory: the public runner opens no games")


# --------------------------------------------------------------------- main

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command entry point. THE EXIT STATUS IS THE GATE."""
    p = argparse.ArgumentParser(
        prog="e4_screen_runner",
        description="E4 endpoint-screen harness. The 32-game screen is UNAUTHORIZED: "
                    "--mode accepts 'qualify' only, and no path reaches the screen.",
        epilog="exit codes: 0 ok, 2 precondition refused, 3 aborted mid-run, 4 unexpected")
    p.add_argument("--plan", required=True, help="path to the canonical plan (verified by hash)")
    p.add_argument("--results", required=True, help="results JSONL; MUST NOT already exist")
    p.add_argument("--mode", default="qualify",
                   help="only 'qualify' is permitted; anything else is refused")
    args = p.parse_args(argv)
    try:
        return run(args.plan, args.results, mode=args.mode)
    except HarnessError as e:
        print(f"PRECONDITION REFUSED: {e}", file=sys.stderr)
        return EXIT_PRECONDITION
    except AbortError as e:
        print(f"ABORTED: {e}", file=sys.stderr)
        return EXIT_ABORT
    except Exception as e:                                    # noqa: BLE001
        print(f"UNEXPECTED {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
