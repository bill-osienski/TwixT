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


def play_task(*, task: Dict[str, Any], agent_for: Callable, state_factory: Callable,
              binder: Callable, rec: Recorder, ply_cap: int = PLY_CAP) -> Dict[str, Any]:
    """Play ONE task to a terminal state. The loop lives here, not in a caller.

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
        binder(task, state, state.ply)                        # raises AbortError on divergence
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
                continue
            rec.emit({"record_type": "task_start", "task_id": task["task_id"],
                      "endpoint": endpoint, "seed": task["seed"]})

            def agent_for(t, mover, _task=task):
                agent = (_agent_factory or _refuse_factory)(_task, mover, _evaluator)
                _enforce_evaluator(agent, _evaluator, _task, mover)
                return agent

            outcome = play_task(task=task, agent_for=agent_for,
                                state_factory=_state_factory or _refuse_state_factory,
                                binder=binder, rec=rec, ply_cap=_ply_cap)
            row = {"record_type": "task_result", "task_id": task["task_id"],
                   "endpoint": endpoint, "seed": task["seed"], **outcome}
            rec.emit(row)
            results.append(row)
            try:
                cleanup()
            except Exception as e:                            # noqa: BLE001
                raise AbortError(PHASE_CLEANUP, f"after {task['task_id']}: {e}") from None

            rows = [r for r in results if r["endpoint"] == endpoint]
            score = sum(r["t1j_points"] for r in rows)
            caps = sum(1 for r in rows if r["terminal_reason"] == "cap")
            if early_in_band_forced(score, len(rows), n_per, list(_band), caps):
                stopped[endpoint] = (f"IN_BAND forced after {len(rows)} of {n_per} games "
                                     f"(score {score}, {caps} cap terminations)")
                rec.emit({"record_type": "early_stop", "endpoint": endpoint,
                          "played": len(rows), "score": score, "reason": stopped[endpoint]})

        try:
            verdict = classify_run(results, n_per_endpoint=n_per, band=_band)
        except Exception as e:                                # noqa: BLE001
            raise AbortError(PHASE_CLASSIFY, str(e)) from None
        rec.emit({"record_type": "verdict", **verdict, "early_stopped": stopped,
                  "tasks_played": len(results)})
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
    """Identity, checked at construction. A rebuild ABORTS; it is not counted."""
    if expected is None:
        return
    got = _evaluator_of(agent, None)
    if got is not expected:
        raise AbortError(
            PHASE_FACTORY,
            f"{task['task_id']} ({mover}): the agent holds a DIFFERENT evaluator object than "
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
