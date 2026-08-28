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

ORDER IS THE POINT. Every precondition -- plan, schedule eligibility, repository
state, JDK components, jar, checkpoint, output path -- is checked FIRST and each
is IMMEDIATELY FATAL. Only when all seven have completed is the authorization
gate consulted. Nothing
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
from . import e4_screen_reference as REF
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
#: The loader resolves the checkpoint relative to the repository. The verified
#: path must therefore BE that path, or a byte-identical copy elsewhere would
#: pass the precondition while different bytes were opened.
CANONICAL_CHECKPOINT_REL = "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"

#: The fixed order. Every one is fatal; the authorization gate follows all seven.
#:
#: `plan` and `schedule` are two questions, not one. `plan` asks whether the
#: canonical schedule parses and matches its pinned digests, and keeps answering
#: yes forever -- a spent schedule is still evidence. `schedule` asks whether it
#: may be RUN, and since 2026-08-26 the answer is no, because the block is spent.
#: That refusal is a precondition, exit 2, not an unexpected error.
#: The qualified per-invocation limit for a T1j process, from the E4 preflight:
#: docs/superpowers/2026-08-25-t1j-e4-preflight.md:40 -- "per-query timeout 120 s".
#: It bounds ONE call and bounds nothing about a whole run; that is a separate
#: limit and a separate decision. Named here because the preflight that measured
#: it is this command's, and read from here by anything else that needs it, so
#: there is one source rather than a number retyped per call site.
T1J_TIMEOUT_S = 120

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
    """The canonical ordered 32-task schedule. No injection, no reshaping.

    RETAINS the verified plan under "plan". An earlier version returned only
    metadata and the authorized branch then referred to a `plan` that did not
    exist -- flipping the constant would have raised NameError. The verified
    object is carried forward from here; it is never re-read later.
    """
    try:
        plan = H.load_canonical_plan(plan_path)
    except (H.HarnessError, REF.E4ReferenceError) as e:
        # BOTH. The harness converts what it sees, but a plan-path failure must
        # exit 2 whichever module names it -- an E4ReferenceError escaping to the
        # top made a fully understood refusal print UNEXPECTED and exit 4.
        raise PreconditionError("plan", str(e)) from None
    return {"plan_sha256": H.CANONICAL_PLAN_SHA256,
            "task_digest": H.CANONICAL_TASK_DIGEST, "n_tasks": len(plan["tasks"]),
            "plan": plan}


def _verified_plan(records: Dict[str, Any]) -> Dict[str, Any]:
    """The plan object check_plan verified. Never re-read, never re-parsed."""
    try:
        return records["plan"]["plan"]
    except KeyError:
        # Only reachable by reordering PRECONDITIONS. A bare KeyError here would
        # surface as UNEXPECTED/exit 4; this is a refusal, so it exits 2.
        raise PreconditionError(
            "schedule", "the schedule check ran before the plan was verified") from None


def check_schedule(plan: Dict[str, Any]) -> Dict[str, Any]:
    """MAY THE VERIFIED SCHEDULE BE RUN? Asked separately from whether it parses.

    check_plan answers identity and well-formedness, and keeps answering forever.
    This answers availability, and its answer CHANGES: once the canonical screen
    ran, its block became exposed-and-retired, so from then on this refuses --
    as a precondition, exit 2, which is what a fully understood refusal is. It is
    not an unexpected error, and it is not something to route around.
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
            "executable": True}


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


def check_checkpoint(checkpoint_path: str, repo_root: str) -> Dict[str, Any]:
    """Hashed, never opened as a model. Both digests, both fatal.

    AND it must be the very file the loader will open. `load_reference_evaluator`
    resolves `calib020_0001` relative to the repository, so verifying some other
    path -- even a byte-identical copy -- would check one file and load another.
    """
    expected = os.path.realpath(os.path.join(repo_root, CANONICAL_CHECKPOINT_REL))
    got_path = os.path.realpath(checkpoint_path)
    if got_path != expected:
        raise PreconditionError(
            "checkpoint",
            f"the supplied path is not the one the loader will open:\n"
            f"      supplied {got_path}\n      loader   {expected}")
    if not os.path.isfile(checkpoint_path):
        raise PreconditionError("checkpoint", f"missing: {checkpoint_path}")
    got256 = _sha256(checkpoint_path)
    if got256 != CHECKPOINT_SHA256:
        raise PreconditionError("checkpoint", f"sha256 {got256} != pinned {CHECKPOINT_SHA256}")
    got1 = _sha1(checkpoint_path)
    if got1 != CHECKPOINT_SHA1:
        raise PreconditionError("checkpoint", f"sha1 {got1} != pinned {CHECKPOINT_SHA1}")
    return {"sha256": got256, "sha1": got1, "path": got_path}


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
    if not SCREEN_AUTHORIZED:
        raise AuthorizationError(trace)

    # ---- nothing below this line runs while the screen is unauthorized -------
    return _execute_screen(
        plan=records["plan"]["plan"],               # the object check_plan verified
        plan_path=plan_path, results_path=results_path, repo_root=repo_root,
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
                    _compile: Optional[Callable] = None,
                    _state_factory: Optional[Callable] = None,
                    _binder: Optional[Callable] = None) -> int:
    """The authorized execution path. REACHED ONLY when SCREEN_AUTHORIZED is True.

    Complete production wiring, dormant: it builds the T1j runtime and context, the
    state factory, the E3b per-ply binder and the agent factory, loads the pinned
    reference ONCE, and hands the CANONICAL 32 TASKS to the qualified harness.

    THE HARNESS OWNS RECORDING. This function creates no Recorder of its own; it
    passes the path and lets the harness open it exclusively and fsync each record.

    NO RNG IS CREATED HERE. The only generators in a screen are the two each
    reference agent derives from its own bound task seed, inside
    SeededReferenceAgent -- there is no run-level RNG to seed.

    It is the one authorized caller of the harness's private `_run`, and the only
    caller that may use SCREEN mode: the screen must supply real collaborators,
    and this command's own public surface exposes none of them. Screen mode makes
    the harness verify the full canonical schedule by CONTENT, permit exactly the
    reserved live seed block, and write a header that says what it is --
    mode="screen", no_games=False, synthetic_tasks=0.
    """
    from . import e4_screen_integration as _INT

    # A RUN-UNIQUE, EXCLUSIVE class directory tied to the new results path. A
    # shared <parent>/t1j_classes could be overwritten by, or shared with, another
    # run -- and compile_helper's mkdir(exist_ok=True) would not notice.
    classes_dir = os.path.abspath(results_path) + ".t1j_classes"

    def setup() -> Dict[str, Any]:
        """Everything effectful. Runs inside the harness, AFTER the identity
        header is fsynced and UNDER the harness's abort classification."""
        if os.path.exists(classes_dir):
            raise H.AbortError(H.PHASE_SETUP,
                               f"the class directory already exists: {classes_dir}")
        os.makedirs(classes_dir)                   # exclusive: raises if it appears
        trace.append("compile")
        artifacts = (_compile or _default_compile)(
            os.path.join(jdk_home, "bin", "javac"), jar_path, classes_dir)
        trace.append("t1j_runtime")
        runtime = _INT.T1jRuntime(java=os.path.join(jdk_home, "bin", "java"), jar=jar_path,
                                  classes=classes_dir, ply_cap=H.PLY_CAP,
                                  timeout_s=T1J_TIMEOUT_S)
        ctx = _INT.IntegrationContext()
        trace.append("load_evaluator")
        evaluator = (_load_evaluator or _default_load_evaluator)(repo_root)
        trace.append("agent_factory")
        build = _build_agent or _default_build_agent
        return {
            "state_factory": _state_factory or _INT.make_state_factory(plan["openings"], ctx),
            "binder": _binder or _INT.make_binder(runtime, ctx),
            "agent_factory": _INT.make_agent_factory(
                runtime=runtime, ctx=ctx, evaluator=evaluator,
                reference_build=lambda task, evaluator: build(task, evaluator=evaluator)),
            "evaluator": evaluator,
            "cleanup": _cleanup or _default_cleanup,
            "artifacts": dict(artifacts or {}, classes_dir=classes_dir),
        }

    trace.append("run_harness")
    run_harness = _run_harness or H._run
    rc = run_harness(
        plan_path, results_path, mode=H.SCREEN_MODE,
        _tasks=plan["tasks"],                       # THE CANONICAL 32, in order
        _identity=records,                          # verified identities, fsynced first
        _setup=setup,                               # effects, after the header
        _n_per_endpoint=H.CANONICAL_N_PER_ENDPOINT,
        _ply_cap=H.PLY_CAP,
        _ply_budget=None,                           # a screen plays to a terminal state
        _band=H.BAND)
    return _map_harness_exit(rc)


def _map_harness_exit(rc: int) -> int:
    """The harness's status is the command's, one for one."""
    return {H.EXIT_OK: EXIT_OK, H.EXIT_PRECONDITION: EXIT_PRECONDITION,
            H.EXIT_ABORT: EXIT_ABORT}.get(rc, EXIT_UNEXPECTED)


def _default_compile(javac: str, jar: str, out_dir: str) -> Dict[str, Any]:
    """Compile, and RETURN the identities of what went in and what came out."""
    from . import t1j_adapter as A
    r = A.compile_helper(javac, jar, out_dir, sources=A.PREFLIGHT_SOURCES)
    if r.returncode != 0:
        raise H.AbortError(H.PHASE_SETUP,
                           f"javac exit {r.returncode}: {r.stderr.strip()[:200]}")
    sources = {p.name: _sha256(str(p)) for p in A.PREFLIGHT_SOURCES}
    classes = {}
    for root, _dirs, files in os.walk(out_dir):
        for f in sorted(files):
            if f.endswith(".class"):
                full = os.path.join(root, f)
                classes[os.path.relpath(full, out_dir)] = _sha256(full)
    return {"compile_inputs": sources, "compiled_classes": classes}


def _default_load_evaluator(repo_root: str):
    """Loads the checkpoint the precondition verified -- the same path, by
    construction: check_checkpoint refuses any other."""
    from .twixtbot_g3_reference import load_reference_evaluator
    return load_reference_evaluator("calib020_0001", repo_root)


def _default_cleanup() -> None:
    from .twixtbot_g3_reference import between_games_cleanup
    between_games_cleanup()


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
                    "change that. Enabling it is a reviewed code change. The canonical "
                    "schedule is also SPENT -- it ran once on 2026-08-26 -- so this "
                    "refuses at the `schedule` precondition with exit 2 before "
                    "authorization is ever consulted. The plan itself still loads: spent "
                    "prevents execution, not reading.",
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
