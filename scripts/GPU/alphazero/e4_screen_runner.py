"""E4 endpoint-screen execution harness.

QUALIFICATION ONLY. **The 32-game screen is UNAUTHORIZED.** `run` refuses any
mode that would execute the canonical schedule, and refuses to construct a
generator from any scheduled seed. The canonical plan is loaded and VERIFIED
here, but its tasks are inert data: nothing in qualify mode runs them.

THE PUBLIC ENTRY POINT ACCEPTS PATHS ONLY. No task list, callable, evaluator,
cleanup hook, classifier or schedule can be injected through it -- that is the
whole point of a harness whose schedule is supposed to be fixed. The private
`_run` carries injection points, and they exist for one reason: to drive the
fail-closed paths in tests. A test asserting `run`'s signature keeps the two
apart.

Recording is append-only and line-buffered: a record is durable when it happens,
never rewritten at the end. A run that aborts leaves exactly what it had done.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import e4_screen_reference as REF
from .e4_screen_rules import (              # frozen decision rules, imported not copied
    LARGER_MATCH_PERMITTED, classify_joint, per_endpoint_decision,
)

#: The canonical plan, pinned by content. Committed at fee7a3b.
CANONICAL_PLAN_SHA256 = "10cd8c3156de7f8a6aec87b0b62318b6bb56c7a32fca56b76eb8448bfc065ac8"
#: Digest over the ORDERED, dimension-projected task list. Order-sensitive, so a
#: reordering changes it; value-sensitive, so an edit does; length-sensitive, so
#: an addition, removal or duplicate does.
CANONICAL_TASK_DIGEST = "f5a21395e67ad130a8e57d9219a435afce0be23cbc14828f56a22a70a842802b"
CANONICAL_N_TASKS = 32
CANONICAL_N_PER_ENDPOINT = 16

TASK_DIMENSIONS = ("task_id", "endpoint", "t1j_mdPly", "t1j_mdFixedPly", "opening",
                   "colour_arm", "anchor_colour", "reference", "reference_sha1",
                   "reference_colour", "seed")

MODES = ("qualify",)          # "screen" is deliberately absent: it is unauthorized


class HarnessError(Exception):
    """The harness refused to proceed. Every refusal is fail-closed."""


def task_digest(tasks: Sequence[Dict[str, Any]]) -> str:
    try:
        payload = json.dumps([[t[k] for k in TASK_DIMENSIONS] for t in tasks],
                             separators=(",", ":"))
    except KeyError as e:
        raise HarnessError(f"task is missing dimension {e}") from None
    return hashlib.sha256(payload.encode()).hexdigest()


def load_canonical_plan(plan_path: str) -> Dict[str, Any]:
    """Load and VERIFY the plan. Any deviation is refused, never repaired."""
    raw = open(plan_path, "rb").read()
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
    """Qualification may never touch a seed the screen reserves."""
    seed = int(task["seed"])
    if REF.seed_is_accounted(seed) or REF.seed_is_exposed(seed):
        raise HarnessError(
            f"seed {seed} is accounted or exposed; qualification runs on synthetic seeds only")


class Recorder:
    """Append-only, line-buffered. Durable as it happens, never rewritten."""

    def __init__(self, path: str):
        self._f = open(path, "a", buffering=1)
        self.n = 0

    def emit(self, record: Dict[str, Any]) -> None:
        self._f.write(json.dumps(record, sort_keys=True) + "\n")
        self.n += 1

    def close(self) -> None:
        self._f.close()


def _default_cleanup() -> None:
    from .twixtbot_g3_reference import between_games_cleanup
    between_games_cleanup()


def classify_run(results: Sequence[Dict[str, Any]], *, n_per_endpoint: int,
                 band: Sequence[float]) -> Dict[str, Any]:
    """Per-endpoint decisions and the joint outcome, from recorded results only.

    PARTIAL RESULTS ARE REFUSED A VERDICT: an endpoint with fewer than
    `n_per_endpoint` resolved games is INCOMPLETE unless IN_BAND is already
    forced, and any INCOMPLETE endpoint makes the joint outcome INCONCLUSIVE.
    """
    out = {}
    for endpoint in ("weak", "strong"):
        rows = [r for r in results if r.get("endpoint") == endpoint]
        score = sum(float(r["t1j_points"]) for r in rows)
        caps = sum(1 for r in rows if r.get("terminal_reason") == "cap")
        out[endpoint] = {
            "played": len(rows), "score": score, "cap_terminations": caps,
            "decision": per_endpoint_decision(score, len(rows), n_per_endpoint,
                                              list(band), caps),
        }
    joint = classify_joint(out["weak"]["decision"], out["strong"]["decision"])
    return {"per_endpoint": out, "joint": joint,
            "larger_match_permitted": joint in LARGER_MATCH_PERMITTED}


def run(plan_path: str, results_path: str, *, mode: str = "qualify") -> int:
    """PUBLIC ENTRY POINT. Paths only.

    No task, callable, evaluator, cleanup hook, classifier or schedule may be
    passed in. The plan is loaded and verified by this function itself.
    Returns 0 on success; every failure raises. The 32-game screen is
    unauthorized and no mode reaches it.
    """
    return _run(plan_path, results_path, mode=mode)


def _run(plan_path: str, results_path: str, *, mode: str,
         _tasks: Optional[Sequence[Dict[str, Any]]] = None,
         _agent_factory: Optional[Callable] = None,
         _opponent: Optional[Callable] = None,
         _evaluator: Any = None,
         _cleanup: Optional[Callable] = None,
         _n_per_endpoint: Optional[int] = None,
         _band: Sequence[float] = (0.05, 0.95)) -> int:
    """PRIVATE. The underscore parameters exist ONLY to drive fail-closed tests."""
    if mode not in MODES:
        raise HarnessError(
            f"mode {mode!r} is not permitted; the 32-game screen is UNAUTHORIZED")

    plan = load_canonical_plan(plan_path)          # verified, then treated as inert
    tasks = list(_tasks) if _tasks is not None else []
    for t in tasks:
        _assert_not_scheduled(t)

    cleanup = _cleanup or _default_cleanup
    rec = Recorder(results_path)
    cleanups = 0
    # The OBJECTS, not their ids: CPython recycles id() for freed objects, so an
    # id-based count could report a rebuilt evaluator as a reused one -- exactly
    # the failure the reuse check exists to catch. Holding the references also
    # keeps them alive for the duration of the run.
    evaluators_seen: List[Any] = []
    results: List[Dict[str, Any]] = []
    try:
        rec.emit({"record_type": "run_header", "mode": mode,
                  "plan_sha256": CANONICAL_PLAN_SHA256,
                  "task_digest": CANONICAL_TASK_DIGEST,
                  "canonical_tasks": len(plan["tasks"]),
                  "canonical_tasks_executed": 0,
                  "synthetic_tasks": len(tasks), "no_games": True})
        for task in tasks:
            rec.emit({"record_type": "task_start", "task_id": task["task_id"],
                      "endpoint": task["endpoint"], "seed": task["seed"]})
            agent = (_agent_factory or _refuse_factory)(task, _evaluator)
            seen = _evaluator_of(agent, _evaluator)
            if not any(seen is e for e in evaluators_seen):
                evaluators_seen.append(seen)
            outcome = (_opponent or _refuse_opponent)(task, agent, rec)
            if outcome.get("abort"):
                rec.emit({"record_type": "abort", "task_id": task["task_id"],
                          "reason": outcome["abort"]})
                raise HarnessError(f"{task['task_id']}: {outcome['abort']}")
            row = {"record_type": "task_result", "task_id": task["task_id"],
                   "endpoint": task["endpoint"], "seed": task["seed"],
                   "t1j_points": outcome["t1j_points"],
                   "terminal_reason": outcome["terminal_reason"],
                   "plies": outcome.get("plies", 0)}
            rec.emit(row)
            results.append(row)
            cleanup()
            cleanups += 1
        verdict = classify_run(results, n_per_endpoint=_n_per_endpoint or CANONICAL_N_PER_ENDPOINT,
                               band=_band)
        rec.emit({"record_type": "verdict", **verdict,
                  "cleanups": cleanups,
                  "distinct_evaluators": len(evaluators_seen)})
        return 0
    finally:
        rec.close()


def _evaluator_of(agent, fallback):
    """The evaluator an agent actually holds. MCTS stores it as `self.evaluator`
    (mcts.py), so a real SeededReferenceAgent exposes it via `agent.mcts`."""
    mcts = getattr(agent, "mcts", None)
    if mcts is not None and hasattr(mcts, "evaluator"):
        return mcts.evaluator
    return getattr(agent, "evaluator", fallback)


def _refuse_factory(task, evaluator):
    raise HarnessError("no agent factory: the public runner builds no agents in qualify mode")


def _refuse_opponent(task, agent, rec):
    raise HarnessError("no opponent: qualify mode requires an injected stub")
