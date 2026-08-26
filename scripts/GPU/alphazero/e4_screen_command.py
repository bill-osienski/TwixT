"""The E4 endpoint-screen command.

**THE SCREEN IS NOT AUTHORIZED.** `SCREEN_AUTHORIZED` below is `False`, and there
is **no supported override**: no command-line option, no environment variable, no
configuration file, no import-time hook. Stated precisely, because the stronger
claim would be false: a Python module global is technically rebindable by any code
that imports this module, so what is guaranteed is that no *supported* input
changes it. Enabling the screen is a reviewed change to the line below plus a
separate authorization.

The command refuses BEFORE it loads a model, constructs an agent or an RNG, starts
a jvm, or creates a results file.

ORDER IS THE POINT. Every precondition -- plan, repository state, JDK components,
jar, checkpoint, output path -- is checked FIRST and each is IMMEDIATELY FATAL.
Only when all six have completed is the authorization gate consulted. Nothing
effectful happens before both. The E4 integration script got this wrong in a way
that mattered: its JDK gate stopped, but its jar and checkpoint gates merely
accumulated, so a mismatch there would have loaded a model and spent seeds before
anything refused.

The public entry point takes PATHS ONLY. The private `_run_screen` carries seams
that record attempted effects; they exist so a qualification can prove the
ordering rather than assert it.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional

from . import e4_screen_integration as INT
from . import e4_screen_runner as H

#: THE SCREEN IS UNAUTHORIZED. Changing this is a reviewed code change.
#: Read directly, in one place. NO SUPPORTED OVERRIDE exists -- not argv, not the
#: environment, not a configuration file, not an import-time hook. It is not
#: claimed to be immutable: an importer could rebind any module global.
SCREEN_AUTHORIZED = False

#: Pinned artifacts. The jar and checkpoint are outside the repository.
JAR_SHA256 = "53ec95e421db2531758142e9ee8ae49030f5345f5dc0c57b2ddb103fbd44e9b7"
CHECKPOINT_SHA256 = "34c79c0d85a837f0281e90bb6a132c41535dc7830729fdd872af7df9612fcc26"
CHECKPOINT_SHA1 = "209cf2d4fd24a48553d259dd71b4954867b9473e"

#: The fixed order. Every one is fatal; the authorization gate follows all six.
PRECONDITIONS = ("plan", "repository", "jdk", "jar", "checkpoint", "output_path")

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
    """The screen is not authorized. Carries the preconditions that completed."""

    def __init__(self, completed: List[str]):
        super().__init__(
            "the canonical 32-game screen is NOT AUTHORIZED: SCREEN_AUTHORIZED is False. "
            "Enabling it is a reviewed code change plus a separate authorization, not a flag.")
        self.completed_preconditions = list(completed)


class EffectLog:
    """Records ATTEMPTED effects. A qualification substitutes this for the real
    things, so 'nothing ran before the gate' is measured, not asserted."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def _record(self, name: str) -> Callable:
        def _fn(*a, **kw):
            self.calls.append(name)
            raise AssertionError(f"effect {name!r} was reached; nothing effectful may run here")
        return _fn

    @property
    def load_evaluator(self) -> Callable:
        return self._record("load_evaluator")

    @property
    def build_agent(self) -> Callable:
        return self._record("build_agent")

    @property
    def run_harness(self) -> Callable:
        return self._record("run_harness")

    @property
    def compile(self) -> Callable:
        return self._record("compile")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------ preconditions

def check_plan(plan_path: str) -> Dict[str, Any]:
    """The canonical ordered 32-task schedule. No injection, no reshaping."""
    try:
        plan = H.load_canonical_plan(plan_path)
    except H.HarnessError as e:
        raise PreconditionError("plan", str(e)) from None
    return {"plan_sha256": H.CANONICAL_PLAN_SHA256,
            "task_digest": H.CANONICAL_TASK_DIGEST, "n_tasks": len(plan["tasks"])}


def check_repository(repo_root: str, plan_path: str) -> Dict[str, Any]:
    """A screen runs from a committed tree, and its plan must be that tree's."""
    def git(*a):
        r = subprocess.run(["git", "-C", repo_root, *a], capture_output=True, text=True)
        return r.returncode, r.stdout.strip()

    rc, head = git("rev-parse", "HEAD")
    if rc != 0:
        raise PreconditionError("repository", f"{repo_root} is not a git repository")
    rc, dirty = git("status", "--porcelain")
    if rc != 0:
        raise PreconditionError("repository", "cannot read the worktree state")
    if dirty:
        n = len(dirty.splitlines())
        raise PreconditionError("repository",
                                f"the worktree has {n} uncommitted change(s); a screen must run "
                                f"from a committed tree")
    rel = os.path.relpath(os.path.abspath(plan_path), os.path.abspath(repo_root))
    rc, committed = git("rev-parse", f"HEAD:{rel}")
    if rc != 0:
        raise PreconditionError("repository", f"the plan is not tracked at HEAD: {rel}")
    rc, on_disk = git("hash-object", plan_path)
    if rc != 0 or on_disk != committed:
        raise PreconditionError("repository",
                                f"the plan on disk ({on_disk}) is not the plan at HEAD ({committed})")
    return {"head": head, "worktree": "clean", "plan_blob": committed}


def check_jdk(jdk_home: str) -> Dict[str, Any]:
    try:
        seen = INT.verify_jdk_identity(jdk_home)
    except H.AbortError as e:
        raise PreconditionError("jdk", e.message) from None
    return {"components": seen}


def check_jar(jar_path: str) -> Dict[str, Any]:
    if not os.path.isfile(jar_path):
        raise PreconditionError("jar", f"missing: {jar_path}")
    got = _sha256(jar_path)
    if got != JAR_SHA256:
        raise PreconditionError("jar", f"sha256 {got} != pinned {JAR_SHA256}")
    return {"sha256": got}


def check_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Hashed, never opened as a model. Both digests, both fatal."""
    if not os.path.isfile(checkpoint_path):
        raise PreconditionError("checkpoint", f"missing: {checkpoint_path}")
    got256 = _sha256(checkpoint_path)
    if got256 != CHECKPOINT_SHA256:
        raise PreconditionError("checkpoint", f"sha256 {got256} != pinned {CHECKPOINT_SHA256}")
    got1 = _sha1(checkpoint_path)
    if got1 != CHECKPOINT_SHA1:
        raise PreconditionError("checkpoint", f"sha1 {got1} != pinned {CHECKPOINT_SHA1}")
    return {"sha256": got256, "sha1": got1}


def check_output_path(results_path: str) -> Dict[str, Any]:
    """Checked WITHOUT creating it. The file is created only once authorized."""
    if os.path.exists(results_path):
        raise PreconditionError("output_path",
                                f"already exists: {results_path}. A run writes a NEW file; "
                                f"appending would merge two runs.")
    parent = os.path.dirname(os.path.abspath(results_path))
    if not os.path.isdir(parent):
        raise PreconditionError("output_path", f"parent directory does not exist: {parent}")
    if not os.access(parent, os.W_OK):
        raise PreconditionError("output_path", f"parent directory is not writable: {parent}")
    return {"path": results_path, "created": False}


# --------------------------------------------------------------- the command

def run_screen(*, plan_path: str, results_path: str, repo_root: str, jdk_home: str,
               jar_path: str, checkpoint_path: str) -> int:
    """PUBLIC ENTRY POINT. Paths only: no task, callable, agent or evaluator."""
    return _run_screen(plan_path=plan_path, results_path=results_path, repo_root=repo_root,
                       jdk_home=jdk_home, jar_path=jar_path, checkpoint_path=checkpoint_path)


def _run_screen(*, plan_path: str, results_path: str, repo_root: str, jdk_home: str,
                jar_path: str, checkpoint_path: str,
                _load_evaluator: Optional[Callable] = None,
                _build_agent: Optional[Callable] = None,
                _run_harness: Optional[Callable] = None,
                _cleanup: Optional[Callable] = None,
                _compile: Optional[Callable] = None,
                _trace: Optional[List[str]] = None) -> int:
    """PRIVATE. The underscore seams record attempted effects, for qualification."""
    trace = _trace if _trace is not None else []
    checks = {
        "plan": lambda: check_plan(plan_path),
        "repository": lambda: check_repository(repo_root, plan_path),
        "jdk": lambda: check_jdk(jdk_home),
        "jar": lambda: check_jar(jar_path),
        "checkpoint": lambda: check_checkpoint(checkpoint_path),
        "output_path": lambda: check_output_path(results_path),
    }
    records: Dict[str, Any] = {}
    for name in PRECONDITIONS:                 # fixed order, each immediately fatal
        records[name] = checks[name]()
        trace.append(name)

    # EVERY precondition has completed. Only now is authorization consulted, and
    # still nothing effectful has happened: no model, no agent, no RNG, no jvm,
    # and no results file.
    if not SCREEN_AUTHORIZED:
        raise AuthorizationError(trace)

    # ---- nothing below this line runs while the screen is unauthorized -------
    return _execute_screen(
        plan=plan, plan_path=plan_path, results_path=results_path, repo_root=repo_root,
        jdk_home=jdk_home, jar_path=jar_path, records=records, trace=trace,
        _load_evaluator=_load_evaluator, _build_agent=_build_agent,
        _run_harness=_run_harness, _cleanup=_cleanup, _compile=_compile)


def _execute_screen(*, plan: Dict[str, Any], plan_path: str, results_path: str,
                    repo_root: str, jdk_home: str, jar_path: str,
                    records: Dict[str, Any], trace: List[str],
                    _load_evaluator: Optional[Callable] = None,
                    _build_agent: Optional[Callable] = None,
                    _run_harness: Optional[Callable] = None,
                    _cleanup: Optional[Callable] = None,
                    _compile: Optional[Callable] = None) -> int:
    """The authorized execution path. REACHED ONLY when SCREEN_AUTHORIZED is True.

    Complete production wiring, dormant: it builds the T1j runtime and context, the
    state factory, the E3b per-ply binder and the agent factory, loads the pinned
    reference ONCE, and hands the CANONICAL 32 TASKS to the qualified harness.

    THE HARNESS OWNS RECORDING. This function creates no Recorder of its own; it
    passes the path and lets the harness open it exclusively and fsync each record.

    NO RNG IS CREATED HERE. The only generators in a screen are the two each
    reference agent derives from its own bound task seed, inside
    SeededReferenceAgent -- there is no run-level RNG to seed.

    It is the one authorized caller of the harness's private `_run`: the screen
    must supply real collaborators, and this command's own public surface exposes
    none of them.
    """
    from . import e4_screen_integration as _INT
    from . import e4_screen_reference as _REF

    trace.append("compile_helper")
    classes_dir = os.path.join(os.path.dirname(os.path.abspath(results_path)), "t1j_classes")
    compile_fn = _compile or _default_compile
    compile_fn(os.path.join(jdk_home, "bin", "javac"), jar_path, classes_dir)

    trace.append("t1j_runtime")
    runtime = _INT.T1jRuntime(java=os.path.join(jdk_home, "bin", "java"), jar=jar_path,
                              classes=classes_dir, ply_cap=H.PLY_CAP)
    ctx = _INT.IntegrationContext()
    state_factory = _INT.make_state_factory(plan["openings"], ctx)
    binder = _INT.make_binder(runtime, ctx)

    trace.append("load_evaluator")
    evaluator = (_load_evaluator or _default_load_evaluator)(repo_root)

    trace.append("agent_factory")
    build = _build_agent or _default_build_agent
    agent_factory = _INT.make_agent_factory(
        runtime=runtime, ctx=ctx, evaluator=evaluator,
        reference_build=lambda task, evaluator: build(task, evaluator=evaluator))

    trace.append("run_harness")
    run_harness = _run_harness or H._run
    rc = run_harness(
        plan_path, results_path, mode="qualify",
        _tasks=plan["tasks"],                       # THE CANONICAL 32, in order
        _agent_factory=agent_factory,
        _state_factory=state_factory,
        _binder=binder,
        _evaluator=evaluator,
        _cleanup=_cleanup or _default_cleanup,
        _n_per_endpoint=H.CANONICAL_N_PER_ENDPOINT,
        _ply_cap=H.PLY_CAP,
        _ply_budget=None,                           # a screen plays to a terminal state
        _band=H.BAND)
    return _map_harness_exit(rc)


def _map_harness_exit(rc: int) -> int:
    """The harness's status is the command's, one for one."""
    return {H.EXIT_OK: EXIT_OK, H.EXIT_PRECONDITION: EXIT_PRECONDITION,
            H.EXIT_ABORT: EXIT_ABORT}.get(rc, EXIT_UNEXPECTED)


def _default_compile(javac: str, jar: str, out_dir: str):
    from . import t1j_adapter as A
    r = A.compile_helper(javac, jar, out_dir, sources=A.PREFLIGHT_SOURCES)
    if r.returncode != 0:
        raise H.AbortError("compile", f"javac exit {r.returncode}: {r.stderr.strip()[:200]}")
    return r


def _default_cleanup() -> None:
    from .twixtbot_g3_reference import between_games_cleanup
    between_games_cleanup()


def _default_load_evaluator(repo_root: str):
    from .twixtbot_g3_reference import load_reference_evaluator
    return load_reference_evaluator("calib020_0001", repo_root)


def _default_build_agent(task, *, evaluator):
    from . import e4_screen_reference as REF
    return REF.build(task, evaluator=evaluator)


# --------------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    """Command entry point. THE EXIT STATUS IS THE GATE.

    There is deliberately no option that enables the screen, and this function
    reads no environment variable and no configuration file.
    """
    p = argparse.ArgumentParser(
        prog="e4_screen_command",
        description="The E4 endpoint screen. IT IS NOT AUTHORIZED: SCREEN_AUTHORIZED is "
                    "False and no option, environment variable or configuration file can "
                    "change that. Enabling it is a reviewed code change.",
        epilog="exit codes: 0 ok, 2 precondition refused, 3 aborted, 4 unexpected, "
               "5 screen not authorized")
    p.add_argument("--plan", required=True)
    p.add_argument("--results", required=True, help="MUST NOT already exist")
    p.add_argument("--repo", required=True)
    p.add_argument("--jdk", required=True)
    p.add_argument("--jar", required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args(argv)
    try:
        return run_screen(plan_path=args.plan, results_path=args.results, repo_root=args.repo,
                          jdk_home=args.jdk, jar_path=args.jar, checkpoint_path=args.checkpoint)
    except PreconditionError as e:
        print(f"PRECONDITION REFUSED: {e}", file=sys.stderr)
        return EXIT_PRECONDITION
    except AuthorizationError as e:
        print(f"SCREEN NOT AUTHORIZED: {e}", file=sys.stderr)
        print(f"  preconditions completed first: {', '.join(e.completed_preconditions)}",
              file=sys.stderr)
        return EXIT_UNAUTHORIZED
    except H.AbortError as e:
        print(f"ABORTED: {e}", file=sys.stderr)
        return EXIT_ABORT
    except Exception as e:                                    # noqa: BLE001
        print(f"UNEXPECTED {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
