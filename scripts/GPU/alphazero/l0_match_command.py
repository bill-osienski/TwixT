"""The L0 64-game match command. IT IS NOT AUTHORIZED.

L1 wires the frozen plan to the qualified harness and stops there. The gate below
is L0's OWN, separate from the E4 screen's: closing one must not depend on the
other, and the screen's constant is not read, imported or consulted here.

ORDER IS THE POINT. Every precondition -- plan, schedule eligibility, repository
state, JDK components, jar, checkpoint, output path -- is checked FIRST and each
is IMMEDIATELY FATAL. Only when all seven have completed is the authorization gate
consulted. Nothing effectful happens before both: no model is loaded, no agent or
RNG constructed, no jvm started, no results file created.

The public entry point takes PATHS ONLY. The private `_run_match` carries seams
that exist to drive fail-closed tests, and no seam can enable the match.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional

from . import e4_screen_command as SCREEN_CMD
from . import e4_screen_integration as INT
from . import e4_screen_reference as REF
from . import e4_screen_runner as H
from . import l0_match_plan as PLAN
from . import l0_match_rules as RULES
from . import l0_match_runner as L0R

#: THE 64-GAME MATCH IS UNAUTHORIZED. Changing this is a reviewed code change.
#: Read directly, in ONE place. NO SUPPORTED OVERRIDE exists -- not argv, not the
#: environment, not a configuration file, not an import-time hook. It is not
#: claimed to be immutable: an importer could rebind any module global.
#:
#: This is L0's OWN gate. e4_screen_command.SCREEN_AUTHORIZED is a DIFFERENT
#: constant guarding a DIFFERENT experiment, and nothing here reads it: one gate
#: must never be openable by opening the other.
L0_EXECUTION_AUTHORIZED = False

#: Pinned artifacts, reused from the screen's command so there is ONE source.
JAR_SHA256 = SCREEN_CMD.JAR_SHA256
CHECKPOINT_SHA256 = SCREEN_CMD.CHECKPOINT_SHA256
CHECKPOINT_SHA1 = SCREEN_CMD.CHECKPOINT_SHA1
CANONICAL_CHECKPOINT_REL = SCREEN_CMD.CANONICAL_CHECKPOINT_REL

#: The fixed order. Every one is fatal; the authorization gate follows all seven.
PRECONDITIONS = ("plan", "schedule", "repository", "jdk", "jar", "checkpoint",
                 "output_path")

EXIT_OK = 0
EXIT_PRECONDITION = 2
EXIT_ABORT = 3
EXIT_UNEXPECTED = 4
EXIT_UNAUTHORIZED = 5


class PreconditionError(Exception):
    """A precondition failed. Immediately fatal; nothing effectful has run."""

    def __init__(self, which: str, message: str):
        super().__init__(f"[{which}] {message}")
        self.which = which
        self.message = message


class AuthorizationError(Exception):
    """The match is not authorized. Carries the preconditions that completed."""

    def __init__(self, completed):
        super().__init__("L0 MATCH NOT AUTHORIZED")
        self.completed_preconditions = list(completed)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------ preconditions

def check_plan(plan_path: str) -> Dict[str, Any]:
    """The frozen ordered 64-task schedule. RETAINS the verified plan."""
    try:
        plan = PLAN.load_l0_plan(plan_path)
    except (PLAN.L0PlanError, H.HarnessError, REF.E4ReferenceError) as e:
        # ALL THREE. A plan-path failure must exit 2 whichever module names it;
        # letting one escape as itself printed UNEXPECTED and exit 4 in the
        # screen's command until that was fixed there.
        raise PreconditionError("plan", str(e)) from None
    return {"plan_sha256": PLAN.L0_PLAN_SHA256, "task_digest": RULES.L0_TASK_DIGEST,
            "n_tasks": len(plan["tasks"]), "plan": plan}


def _verified_plan(records: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return records["plan"]["plan"]
    except KeyError:
        raise PreconditionError(
            "schedule", "the schedule check ran before the plan was verified") from None


def check_schedule(plan: Dict[str, Any]) -> Dict[str, Any]:
    """MAY THE VERIFIED SCHEDULE BE RUN? Asked separately from whether it parses.

    `plan` answers identity and keeps answering forever, so a completed match
    stays readable. This answers availability, and its answer CHANGES the moment
    the block is spent. A refusal here is a precondition, exit 2.
    """
    tasks = plan["tasks"]
    try:
        REF.validate_schedule_executable(tasks)
    except REF.E4ReferenceError as e:
        raise PreconditionError("schedule", str(e)) from None
    status = [REF.seed_status(int(t["seed"])) for t in tasks]
    return {"n_tasks": len(tasks),
            "exposed": sum(x["exposed"] for x in status),
            "retired": sum(x["retired"] for x in status),
            "test_only": sum(x["test_only"] for x in status),
            "executable": True}


def _delegate(which: str, fn, *args) -> Dict[str, Any]:
    """Run a screen-command check and TRANSLATE its refusal into L0's own.

    The reused checks raise `e4_screen_command.PreconditionError`, which is a
    DIFFERENT class from this module's. Without translation `main` did not catch
    it and a fully understood refusal printed `UNEXPECTED ... exit 4` -- caught on
    the very first end-to-end run, and the same defect shape as the screen's own
    E4ReferenceError escape. Reusing a check means adopting its failures too.
    """
    try:
        return fn(*args)
    except SCREEN_CMD.PreconditionError as e:
        raise PreconditionError(which, e.message) from None


def check_repository(repo_root: str, plan_path: str) -> Dict[str, Any]:
    """A match runs from a committed tree, and its plan must be that tree's."""
    return _delegate("repository", SCREEN_CMD.check_repository, repo_root, plan_path)


def check_jdk(jdk_home: str) -> Dict[str, Any]:
    return _delegate("jdk", SCREEN_CMD.check_jdk, jdk_home)


def check_jar(jar_path: str) -> Dict[str, Any]:
    return _delegate("jar", SCREEN_CMD.check_jar, jar_path)


def check_checkpoint(checkpoint_path: str, repo_root: str) -> Dict[str, Any]:
    return _delegate("checkpoint", SCREEN_CMD.check_checkpoint, checkpoint_path, repo_root)


def check_output_path(results_path: str) -> Dict[str, Any]:
    return _delegate("output_path", SCREEN_CMD.check_output_path, results_path)


# --------------------------------------------------------------- the command

def run_match(*, plan_path: str, results_path: str, repo_root: str, jdk_home: str,
              jar_path: str, checkpoint_path: str) -> int:
    """PUBLIC ENTRY POINT. Paths only: no task, callable, agent or evaluator."""
    return _run_match(plan_path=plan_path, results_path=results_path,
                      repo_root=repo_root, jdk_home=jdk_home, jar_path=jar_path,
                      checkpoint_path=checkpoint_path)


def _run_match(*, plan_path: str, results_path: str, repo_root: str, jdk_home: str,
               jar_path: str, checkpoint_path: str,
               _load_evaluator: Optional[Callable] = None,
               _build_agent: Optional[Callable] = None,
               _run_harness: Optional[Callable] = None,
               _cleanup: Optional[Callable] = None,
               _compile: Optional[Callable] = None,
               _state_factory: Optional[Callable] = None,
               _binder: Optional[Callable] = None,
               _trace: Optional[List[str]] = None) -> int:
    """PRIVATE. The seams drive fail-closed tests; none of them opens the gate."""
    trace = _trace if _trace is not None else []
    checks = {
        "plan": lambda: check_plan(plan_path),
        "schedule": lambda: check_schedule(_verified_plan(records)),
        "repository": lambda: check_repository(repo_root, plan_path),
        "jdk": lambda: check_jdk(jdk_home),
        "jar": lambda: check_jar(jar_path),
        "checkpoint": lambda: check_checkpoint(checkpoint_path, repo_root),
        "output_path": lambda: check_output_path(results_path),
    }
    records: Dict[str, Any] = {}
    for name in PRECONDITIONS:                 # fixed order, each immediately fatal
        records[name] = checks[name]()
        trace.append(name)

    # EVERY precondition has completed. Only now is authorization consulted, and
    # still nothing effectful has happened: no model, no agent, no RNG, no jvm,
    # and no results file.
    if not L0_EXECUTION_AUTHORIZED:
        raise AuthorizationError(trace)

    # ---- nothing below this line runs while the match is unauthorized --------
    return _execute_match(
        plan=records["plan"]["plan"], plan_path=plan_path, results_path=results_path,
        repo_root=repo_root, jdk_home=jdk_home, jar_path=jar_path,
        records=records, trace=trace,
        _load_evaluator=_load_evaluator, _build_agent=_build_agent,
        _run_harness=_run_harness, _cleanup=_cleanup, _compile=_compile,
        _state_factory=_state_factory, _binder=_binder)


def _execute_match(*, plan: Dict[str, Any], plan_path: str, results_path: str,
                   repo_root: str, jdk_home: str, jar_path: str,
                   records: Dict[str, Any], trace: List[str],
                   _load_evaluator: Optional[Callable] = None,
                   _build_agent: Optional[Callable] = None,
                   _run_harness: Optional[Callable] = None,
                   _cleanup: Optional[Callable] = None,
                   _compile: Optional[Callable] = None,
                   _state_factory: Optional[Callable] = None,
                   _binder: Optional[Callable] = None) -> int:
    """The authorized execution path. REACHED ONLY when the gate is True.

    Complete production wiring, dormant. Every collaborator is the QUALIFIED one:
    the T1j runtime, state factory and per-ply binder from the E3b-qualified
    integration module, the reference construction that delegates to the G3
    builder, and the L0 harness's own loop and recorder.
    """
    classes_dir = results_path + ".t1j_classes"

    def setup() -> Dict[str, Any]:
        """Everything effectful. Runs inside the harness, AFTER the identity
        header is fsynced and UNDER the harness's abort classification.

        Every collaborator is constructed EXACTLY as the qualified screen command
        constructs it. Three defects lived here until an enabled-path test reached
        them: T1jRuntime was called with the wrong keywords, make_state_factory
        was called without the openings it needs, and nothing was traced. A gate
        that is never opened hides its own branch.
        """
        if os.path.exists(classes_dir):
            raise H.AbortError(H.PHASE_SETUP,
                               f"the class directory already exists: {classes_dir}")
        os.makedirs(classes_dir)                   # exclusive: raises if it appears
        trace.append("compile")
        artifacts = (_compile or SCREEN_CMD._default_compile)(
            os.path.join(jdk_home, "bin", "javac"), jar_path, classes_dir)
        trace.append("t1j_runtime")
        runtime = INT.T1jRuntime(java=os.path.join(jdk_home, "bin", "java"),
                                 jar=jar_path, classes=classes_dir,
                                 ply_cap=RULES.PLY_CAP,
                                 timeout_s=SCREEN_CMD.T1J_TIMEOUT_S)
        ctx = INT.IntegrationContext()
        trace.append("load_evaluator")
        evaluator = (_load_evaluator or SCREEN_CMD._default_load_evaluator)(repo_root)
        trace.append("agent_factory")
        build = _build_agent or SCREEN_CMD._default_build_agent
        return {
            "state_factory": _state_factory or INT.make_state_factory(plan["openings"], ctx),
            "binder": _binder or INT.make_binder(runtime, ctx),
            "agent_factory": INT.make_agent_factory(
                runtime=runtime, ctx=ctx, evaluator=evaluator,
                reference_build=lambda task, evaluator: build(task, evaluator=evaluator)),
            "evaluator": evaluator,
            "cleanup": _cleanup or SCREEN_CMD._default_cleanup,
            "artifacts": dict(artifacts or {}, classes_dir=classes_dir),
        }

    trace.append("run_harness")
    run_harness = _run_harness or L0R._run
    rc = run_harness(
        plan_path, results_path, mode=L0R.MATCH_MODE,
        _tasks=plan["tasks"],                   # THE FROZEN 64, in order
        _identity=records,                      # verified identities, fsynced first
        _setup=setup,                           # effects, after the header
        _ply_cap=RULES.PLY_CAP,
        _ply_budget=None)                       # a match plays to a terminal state
    return _map_harness_exit(rc)


def _map_harness_exit(rc: int) -> int:
    """The harness's status is the command's, one for one."""
    return {H.EXIT_OK: EXIT_OK, H.EXIT_PRECONDITION: EXIT_PRECONDITION,
            H.EXIT_ABORT: EXIT_ABORT}.get(rc, EXIT_UNEXPECTED)


def main(argv: Optional[List[str]] = None) -> int:
    """Command entry point. THE EXIT STATUS IS THE GATE.

    There is deliberately no option that enables the match, and this function
    reads no environment variable and no configuration file.
    """
    p = argparse.ArgumentParser(
        prog="l0_match_command",
        description="The L0 64-game larger match. IT IS NOT AUTHORIZED: "
                    "L0_EXECUTION_AUTHORIZED is False and no option, environment "
                    "variable or configuration file can change that. Enabling it is a "
                    "reviewed code change, separate from the E4 screen's gate.",
        epilog="exit codes: 0 ok, 2 precondition refused, 3 aborted, 4 unexpected, "
               "5 match not authorized")
    p.add_argument("--plan", required=True)
    p.add_argument("--results", required=True, help="MUST NOT already exist")
    p.add_argument("--repo", required=True)
    p.add_argument("--jdk", required=True)
    p.add_argument("--jar", required=True)
    p.add_argument("--checkpoint", required=True)
    a = p.parse_args(argv)
    try:
        return run_match(plan_path=a.plan, results_path=a.results, repo_root=a.repo,
                         jdk_home=a.jdk, jar_path=a.jar, checkpoint_path=a.checkpoint)
    except PreconditionError as e:
        print(f"PRECONDITION REFUSED: {e}", file=sys.stderr)
        return EXIT_PRECONDITION
    except AuthorizationError as e:
        print("L0 MATCH NOT AUTHORIZED: L0_EXECUTION_AUTHORIZED is False.",
              file=sys.stderr)
        print(f"  preconditions completed first: {', '.join(e.completed_preconditions)}",
              file=sys.stderr)
        return EXIT_UNAUTHORIZED
    except H.AbortError as e:
        print(f"ABORTED [{e.phase}]: {e.message}", file=sys.stderr)
        return EXIT_ABORT
    except Exception as e:                                    # noqa: BLE001
        print(f"UNEXPECTED {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    raise SystemExit(main())
