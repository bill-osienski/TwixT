"""The H1 viability screen's execution harness. THE MATCH IS NOT AUTHORIZED.

This module wires the frozen 224-task v3 plan to machinery that is ALREADY
QUALIFIED and rebuilds none of it. Everything effectful comes from the E4
screen's harness, unchanged, exactly as L0's runner reuses it: `play_task`,
`Recorder`, `AbortError`/`PHASE_*`, `_enforce_evaluator`, and the fail-closed
`_refuse_*` defaults.

🔴 TWO BARRIERS, AND THEY DID NOT EXIST BEFORE THIS MODULE
Until now H1 was safe because NO EXECUTABLE PATH EXISTED -- not because anything
protected one. An earlier comment claimed otherwise and was corrected. This
module creates the path, so it must create both barriers with it:

    1. `H1_EXECUTION_AUTHORIZED`, a gate defaulting to False;
    2. `check_seed_registration()`, an INDEPENDENT precondition requiring the H1
       block to be present in `ACCOUNTED_SEED_INTERVALS`.

They are independent on purpose. Opening the gate does not register the block and
registering the block does not open the gate, so neither one alone is enough, and
neither can be opened by opening the other.

🔑 BOTH FIRE BEFORE ANYTHING HAPPENS. Before helper compilation, before any model
load, before any JVM launch, and BEFORE THE RECORDER CREATES ITS FILE -- a
refusal that has already written output has not refused, and an empty results
file is indistinguishable from a run that produced nothing.

WHAT IS DELIBERATELY ABSENT
No early stop of any kind. `h1_viability_rules.may_stop_early` is a constant
False, `EARLY_STOP is None`, and this module consults no band, no saturation rule
and no incompleteness rule. The screen's decision functions named in
`RULES.MUST_NEVER_BE_CALLED_ON_AN_H1_RUN` are not imported, and a test asserts
each still exists so the prohibition cannot rot into a reference to nothing.

🔑 THE VOID RECORD CONTRACT, DECIDED AND STATED
A VOID produces NO REPORT: no rate, no interval, no viability verdict. It does
NOT delete what already happened. The results file keeps every `task_start`,
`ply` and `task_result` record written before the failure, DELIBERATELY:

  * they are the durable evidence of what actually ran, and destroying evidence
    on failure is a worse defect than keeping it;
  * a record already fsynced cannot be unwritten honestly.

So the trace is NOT the only surviving progress record, and this module does not
claim it is. What the trace uniquely guarantees is a NON-ANALYTIC one that
survives when the recorder itself is the thing that failed, or when SIGALRM ends
the process mid-write.

⚠ THE RETAINED ROWS MUST NOT BE ANALYSED, and that is enforced STRUCTURALLY
rather than asked for: `viability_report` binds through `bind_results`, which
refuses any vector that is not all 224 games. A partial vector cannot produce a
rate, an interval or a verdict -- so the retained rows are evidence and cannot
become a result. A test drives the real reporter over a truncated vector to
prove it.

THE VERDICT IS NOT COMPUTED HERE. Rates, intervals and the viability verdict are
`h1_viability_rules.viability_report`'s alone. This module hands it the durable
rows and the FROZEN plan tasks.
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import d1_probe as D1
from . import e4_screen_reference as REF
from . import e4_screen_runner as H
from . import h1_viability_plan as PLAN
from . import h1_viability_rules as RULES
from . import t1j_adapter as A
from . import void_trace as VT

#: 🔴 THE EXECUTION GATE. One line, its own commit when it is ever opened, and
#: restored to False immediately afterwards. It gates the MATCH path only:
#: qualification runs on test-only seeds and plays no scheduled game.
H1_EXECUTION_AUTHORIZED = False

#: 🔴 THE MATCH IS PUBLICLY REACHABLE, AND THE GATE IS WHAT REFUSES IT.
#:
#: An earlier version listed only "qualify" here, so `run(..., mode="match")` was
#: rejected by a MODE LIST before either barrier was consulted -- and every match
#: test had to reach the private `_run`. That is a gate whose branch is never
#: exercised on the public path, which is exactly how L0's command grew three
#: defects that only an enabled-path test found. Opening `H1_EXECUTION_AUTHORIZED`
#: and registering the block must be SUFFICIENT to start the match through the
#: public entry point; nothing else may stand in the way, and nothing else may
#: substitute for them.
MATCH_MODE = "match"
MODES = ("qualify", MATCH_MODE)
_ALL_MODES = MODES

#: 180 minutes, from the card: a measured projection of 107 minutes (L0 played 64
#: games in 30m31s = 28.6 s/game) plus 69% headroom. Exceeding it is a VOID, not
#: a truncated report.
RUN_DEADLINE_S = 180 * 60

#: 120 s per T1j call, the value §12.10 froze and E4/D1 already use.
PER_CALL_TIMEOUT_S = D1.PER_QUERY_TIMEOUT_S

#: Mirrored from the rules' reporter, and bound by a test that runs the real
#: reporter over a cap-heavy vector rather than trusting this literal.
CAP_SATURATED_NO_RATE = "CAP_SATURATED_NO_RATE"


class H1Error(Exception):
    """H1 refuses to run, or the run is not the frozen design."""


class H1VoidError(H1Error):
    """INSTRUMENT FAILURE. No report, no partial result, and the block retires.

    🔑 NOT a losing game. A COMPLETED GAME IS A RESULT whoever wins -- the
    distinction D1 destroyed by conflating FAIL with VOID, which is why the
    low-ply qualification had to exist at all.
    """


# ───────────────────────────── the two barriers ──────────────────────────────

def check_gate() -> None:
    """BARRIER 1. Refuse unless the match has been explicitly authorized."""
    if not H1_EXECUTION_AUTHORIZED:
        raise H1Error(
            "H1_EXECUTION_AUTHORIZED is False: the 224-game head-to-head match is "
            "NOT AUTHORIZED. Opening this gate is a separate, reviewed decision, "
            "and it does not register the seed block.")


def check_seed_registration() -> None:
    """BARRIER 2. The reserved block must be REGISTERED before H1 draws from it.

    READS the registry; never writes one. Registering is a reviewed edit to
    `e4_screen_reference.ACCOUNTED_SEED_INTERVALS` and belongs to the H1
    EXECUTION authorization -- a block reserved on paper and never authorized
    must cost nothing to abandon. A runtime mutation would make the registry
    something the run can grant itself, which is the same shape as a gate that
    opens its own gate.

    EVERY seed is checked, not the endpoints: a partial registration would
    otherwise pass and then draw an unaccounted seed halfway through.

    ⚠ Availability -- exposed, retired, already consumed -- is a DIFFERENT
    question, asked per task by `REF.validate_schedule_executable`. This asks
    only whether the block has been accounted for at all. `validate_task_executable`
    does NOT ask it: it inspects consumed/exposed/retired and accepts an
    unregistered seed, which is precisely why this barrier has to exist
    separately.
    """
    lo, hi = RULES.H1_SEED_BLOCK
    missing = [s for s in range(lo, hi) if not REF.seed_is_accounted(s)]
    if missing:
        raise H1Error(
            f"the H1 match seed block [{lo}, {hi}) is not registered: {len(missing)} "
            f"of {hi - lo} seeds are absent from ACCOUNTED_SEED_INTERVALS (first "
            f"{missing[0]}). Registering it is a reviewed edit to that registry, "
            f"part of the H1 EXECUTION authorization; nothing here writes a "
            f"registry at runtime.")


def _canonical(path: str) -> str:
    """Absolute, symlink-resolved, normalised. Two names for one file are one file."""
    return os.path.realpath(os.path.abspath(path))


def check_output_paths(results_path: str, trace_path: Optional[str] = None, *,
                       require_trace: bool = False) -> None:
    """BARRIER 3, in kind if not in name: the outputs must not already exist.

    🔴 THE EXCEPTION AND THE DURABLE VERDICT HAVE TO AGREE. Recorder construction
    used to happen after `run_start` was traced, so a results path that already
    existed wrote a VOID trace and then re-raised `H.HarnessError` unchanged: the
    file said the instrument failed mid-run while the exception said the run was
    never well formed. Both cannot be true.

    A path that already exists is a PRECONDITION failure -- nothing has run, no
    game was played, no seed drawn, no instrument engaged. It belongs with the
    gate and the registration check, refusing BEFORE any output exists, and it
    writes no trace at all. What remains inside the protected block is the
    genuinely mid-run recorder failure -- a full disk, a revoked permission --
    which IS a VOID and is translated to say so.
    """
    # 🔴 A MATCH WITHOUT A TRACE IS NOT THE FROZEN DESIGN. The card requires a
    # create-only VOID trace, and match mode accepted `trace_path=None` -- so a
    # run could reach the games with the one record a VOID depends on absent.
    if require_trace and not trace_path:
        raise H1Error(
            "match mode requires a trace path: the card freezes a create-only, "
            "non-analytic VOID trace, and a match that cannot say how far it got "
            "is not the design that was preregistered. Nothing has been written.")
    # 🔴 AND THE TWO MUST BE DIFFERENT FILES. Given one path twice, the trace
    # created it and the recorder then refused it -- turning a naming slip into a
    # VOID that retires all 224 seeds. Compared after canonicalisation, because
    # "r.jsonl", "./r.jsonl" and a symlink are the same file under three names.
    if trace_path and _canonical(results_path) == _canonical(trace_path):
        raise H1Error(
            f"the results and trace paths are the same file: {results_path!r} and "
            f"{trace_path!r} both resolve to {_canonical(results_path)}. Writing both "
            f"streams to one file would make an avoidable naming error a VOID, and a "
            f"VOID retires the whole seed block. Nothing has been written.")
    # 🔴 `lexists`, NOT `exists`. `os.path.exists` FOLLOWS the link, so a DANGLING
    # symlink reads as absent -- while the directory entry is already there and
    # create-exclusive open refuses it with FileExistsError. The precheck passed,
    # the trace opened, and recorder creation produced a VOID: another knowable
    # path condition turned into a whole-block retirement.
    for label, path in (("results", results_path), ("trace", trace_path)):
        if path and os.path.lexists(path):
            raise H1Error(
                f"the {label} path already exists: {path}"
                f"{' (a dangling symlink -- the directory entry is present)' if not os.path.exists(path) else ''}"
                f". A run writes NEW files; "
                f"appending would merge two runs, and overwriting would destroy the "
                f"record of one. Nothing has been written and no trace was opened.")


# ─────────────────────────────── the VOID trace ──────────────────────────────

TRACE_SCHEMA = "h1-void-trace/1"
TRACE_EVENTS = ("run_start", "task_start", "task_done", "run_end")
TRACE_VERDICTS = ("OK", "VOID")
TRACE_COUNTER_MAX = {
    "index": RULES.N_GAMES - 1,
    "n_games": RULES.N_GAMES,
    "games_completed": RULES.N_GAMES,
}
TRACE_EVENT_FIELDS = {
    "run_start": frozenset({"schema", "n_games"}),
    "task_start": frozenset({"index"}),
    "task_done": frozenset({"index", "games_completed"}),
    "run_end": frozenset({"verdict", "games_completed"}),
}

#: 🔑 NO IDENTITY STRINGS. No task_id, no opening, no colour arm, no seed, no
#: score. Tasks run in the frozen order, so the INDEX identifies the game without
#: recording anything about it -- and a count of games completed says how far the
#: run got without saying what happened in any of them. A trace that could carry
#: a result would be a partial-match report under a different filename.
H1_TRACE = VT.TraceSchema(
    schema=TRACE_SCHEMA, events=TRACE_EVENTS, event_fields=TRACE_EVENT_FIELDS,
    counter_max=TRACE_COUNTER_MAX, enums={"verdict": TRACE_VERDICTS},
    error=H1Error,
    extra_field_note=("The trace carries counters only; a free-form field would "
                      "make it a partial-match report under a different filename."))


@contextlib.contextmanager
def _trace_file(path: Optional[str]):
    """CREATE-ONLY, like the recorder. Reusing a trace file would let one run's
    progress be read as another's."""
    if path is None:
        yield None
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fh:
        yield fh


def _trace(fh, **fields: Any) -> None:
    if fh is not None:
        VT.write(H1_TRACE, fh, **fields)


# ────────────────────────────── schedule checks ──────────────────────────────

# 🔑 THERE IS NO SEPARATE SEED LOOP HERE, AND THE ONE FIRST WRITTEN COULD NEVER
# FIRE. `_verify_match_schedule` calls `validate_h1_schedule`, which already
# refuses any seed outside the block -- through the rules' predicate -- so a
# second loop below it was unreachable and an injected-defect control proved it:
# deleting it changed no test. One enforcement point, in the plan validator.


def _verify_match_schedule(tasks: Sequence[Dict[str, Any]],
                           plan: Dict[str, Any]) -> None:
    """The run must execute THE FROZEN 224, in order, unedited."""
    try:
        PLAN.validate_h1_schedule(tasks)
    except PLAN.H1PlanError as e:
        raise H1Error(str(e)) from None
    digest = RULES.L0.l0_task_digest(tasks)
    if digest != RULES.H1_TASK_DIGEST:
        raise H1Error(
            f"task digest {digest} != pinned {RULES.H1_TASK_DIGEST}: the schedule "
            f"has been added to, removed from, reordered or edited")
    # Bound by FULL CONTENT, not by task_id: canonical NAMES attached to synthetic
    # CONTENT counted in an earlier workstream, and a test there proved exactly that.
    by_id = {t["task_id"]: t for t in plan["tasks"]}
    for t in tasks:
        c = by_id.get(t["task_id"])
        # EVERY KEY, not just the digest dimensions. Restricted to those, this
        # check was strictly redundant with the digest comparison above it and
        # could never fail -- a control proved exactly that. Widened, it covers
        # what the digest does NOT: reference_sha256, reference_colour details
        # and the derived rng_streams.
        if c is None or any(t.get(k) != c.get(k) for k in set(t) | set(c)):
            raise H1Error(
                f"{t.get('task_id')} does not match the frozen plan's task of that name")
    if len(tasks) != RULES.N_GAMES:
        raise H1Error(f"the run schedules {len(tasks)} tasks, the design is "
                      f"{RULES.N_GAMES}")


class _PlyCountingRecorder:
    """The qualified Recorder, plus the ply the run is ATTEMPTING.

    Needed because the card requires a VOID to NAME THE POSITION IT DIED ON, and
    `play_task` reports the ply count only in its OUTCOME -- which a VOID never
    produces.

    🔴 COUNTING `ply` RECORDS ALONE WAS WRONG IN BOTH DIRECTIONS:
      * a failure on the FIRST searched move recorded `ply=None`, because no
        `ply` record exists yet -- only `opening_bound`;
      * a BINDER failure after a move applies happens BEFORE that move's `ply`
        record is emitted, so it reported the PREVIOUS ply.
    `opening_bound` is now read as well, and `note_bind` is called from a binder
    wrapper BEFORE the real binder runs, so the ply on record is the one being
    attempted rather than the last one that succeeded.
    """

    def __init__(self, path: str):
        self._rec = H.Recorder(path)
        self.last_ply: Optional[int] = None

    def note_bind(self, ply: Optional[int]) -> None:
        """The ply about to be bound. Called before the binder can fail."""
        self.last_ply = ply

    def emit(self, record: Dict[str, Any]) -> None:
        if record.get("record_type") in ("ply", "opening_bound"):
            self.last_ply = record.get("ply")
        self._rec.emit(record)

    def emit_terminal(self, record: Dict[str, Any]):
        return self._rec.emit_terminal(record)

    def close(self):
        return self._rec.close()

    @property
    def path(self):
        return self._rec.path


def _void_diagnostic(task: Optional[Dict[str, Any]], index: Optional[int],
                     ply: Optional[int], error: BaseException) -> Dict[str, Any]:
    """WHERE the run died, as FIELDS -- not buried in a message string.

    🔴 D1's VOID named a depth and an invocation and nothing else, so its cause
    could not be settled afterwards and still cannot be. A message a human must
    parse is not a diagnostic; every item the card requires is a key here, and a
    test asserts each is present rather than searching the prose.
    """
    d: Dict[str, Any] = {
        "task_id": task["task_id"] if task else None,
        "index": index,
        "opening": task["opening"] if task else None,
        "colour_arm": task["colour_arm"] if task else None,
        "rep": task["rep"] if task else None,
        "seed": task["seed"] if task else None,
        "ply": ply,
        "error_type": type(error).__name__,
    }
    d["helper_excerpt"] = _bounded_excerpt(error)
    return d


def _bounded_excerpt(error: BaseException) -> Optional[str]:
    """THE HELPER'S OWN WORDS, BOUNDED, however the failure was wrapped.

    🔴 READING `error.stdout` ALONE PRODUCED `null` FOR REAL FAILURES. Only an
    adapter parse or timeout error carries `stdout`; a postcondition failure, a
    binder divergence or an illegal move arrives as `H.AbortError`, which has no
    such attribute and puts the detail in `.message`. The earlier test made
    `play_task` raise `HelperOutputError` directly and so never met the wrapper
    that real failures pass through.

    Three sources, in order of fidelity, every one capped: the exception's own
    stdout; the stdout of anything in its `__cause__`/`__context__` chain; and
    finally the AbortError's own message, which is where the integration layer
    puts the POSTCOND line, the divergence or the rejected move.
    """
    seen, err = set(), error
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        stdout = getattr(err, "stdout", None)
        if stdout:
            return A.helper_failure_excerpt(stdout)
        err = err.__cause__ or err.__context__
    message = getattr(error, "message", None) or str(error)
    if not message:
        return None
    return message[:A.FAILURE_EXCERPT_CHARS]


# ─────────────────────── the production game setup, dormant ──────────────────

def _production_setup(results_path: str, deadline: "D1.Deadline") -> Callable[[], Dict[str, Any]]:
    """Everything effectful, built EXACTLY as the qualified commands build it.

    🔴 THIS DID NOT EXIST, AND ITS ABSENCE WAS INVISIBLE. On the real path
    `_setup` was None, so the refusing binder, agent factory and state factory
    stayed in place and a real match would have aborted on its first ply --
    while every successful test replaced `play_task` and therefore never reached
    them. A gate that is never opened hides its own branch, and L0's command grew
    three defects in exactly this function before an enabled-path test found them.

    Runs INSIDE the harness, after the identity header is fsynced and under the
    harness's abort classification, so a setup failure still leaves a durable
    record of what the run was.

    THE TOOLCHAIN IS RESOLVED, NOT SUPPLIED. `D1._default_compile` hashes the jar
    and every pinned JDK component through `t1j_toolchain.verified_paths`, which
    refuses a root under /tmp -- the failure that once deleted the whole
    toolchain -- and cross-checks the jar against E4's own qualification pin.
    """
    classes_dir = results_path + ".t1j_classes"

    def setup() -> Dict[str, Any]:
        from . import e4_screen_command as SCREEN_CMD
        from . import e4_screen_integration as INT
        from . import t1j_toolchain as TC

        tc = TC.verified_paths()
        java = os.path.join(tc["jdk_home"], "bin", "java")
        paths = D1.T1jPaths(java=java, jar=tc["jar"], classes=classes_dir,
                            ply_cap=RULES.PLY_CAP)
        # 🔴 THE RUN'S OWN DEADLINE OBJECT, not a new one. The first version
        # built `D1.Deadline(RUN_DEADLINE_S).start()` here, so compilation's
        # cooperative checks measured from a DIFFERENT ORIGIN than the alarm and
        # the run header -- reviving precisely the two-clock defect D1 was
        # corrected for. One deadline, one origin, one auditable window.
        artifacts = D1._default_compile(deadline, paths=paths)
        runtime = INT.T1jRuntime(java=java, jar=tc["jar"], classes=classes_dir,
                                 ply_cap=RULES.PLY_CAP,
                                 timeout_s=PER_CALL_TIMEOUT_S)
        ctx = INT.IntegrationContext()
        evaluator = SCREEN_CMD._default_load_evaluator(".")     # the incumbent, ONCE
        # 🔴 THE OPENINGS COME FROM THE SOURCE PLAN, NOT H1's. H1's frozen plan
        # records opening NAMES on its tasks; the MOVES live in the sha-pinned
        # screen plan, which is why `load_source_plan` exists. The first version
        # read `plan["openings"]` off H1's own plan and raised KeyError -- on the
        # real path, at setup, after the header was written. Only an enabled-path
        # test could reach it, which is the whole reason this setup is tested.
        openings = PLAN.load_source_plan()["openings"]
        return {
            "state_factory": INT.make_state_factory(openings, ctx),
            "binder": INT.make_binder(runtime, ctx),            # the E3b binder
            # 🔴 `t1j_timeout_s` IS A SEPARATE ARGUMENT AND DEFAULTS TO None.
            # `T1jRuntime.timeout_s` bounds REPLAY only; the agent's per-QUERY
            # timeout comes from here, and omitting it left the agent calling
            # `query(..., timeout_s=None)` -- an unbounded wait -- while the
            # runtime truthfully reported 120. Two timeouts, one supplied.
            "agent_factory": INT.make_agent_factory(
                runtime=runtime, ctx=ctx, evaluator=evaluator,
                t1j_timeout_s=PER_CALL_TIMEOUT_S,
                reference_build=lambda task, evaluator: (
                    SCREEN_CMD._default_build_agent(task, evaluator=evaluator))),
            "evaluator": evaluator,
            "cleanup": SCREEN_CMD._default_cleanup,
            "artifacts": dict(artifacts or {}, classes_dir=classes_dir,
                              per_call_timeout_s=PER_CALL_TIMEOUT_S),
        }

    return setup


# ──────────────────────────────── entry points ───────────────────────────────

def run(results_path: str, *, mode: str = "qualify",
        trace_path: Optional[str] = None) -> int:
    """PUBLIC ENTRY POINT. Paths only.

    No task, callable, evaluator, hook, reporter or schedule can be injected, no
    plan path can be supplied -- match mode loads ONLY the pinned v3 plan -- and
    MATCH MODE IS NOT SELECTABLE HERE.
    """
    if mode not in MODES:
        raise H1Error(f"mode {mode!r} is not permitted; modes are {list(MODES)}")
    # NOTE what is NOT here: no rejection of MATCH_MODE. The gate below refuses
    # it, and the gate is the thing a reviewer can see the state of.
    return _run(results_path, mode=mode, trace_path=trace_path,
                _setup_factory=_production_setup if mode == MATCH_MODE else None)


def _run(results_path: str, *, mode: str, trace_path: Optional[str] = None,
         _tasks: Optional[Sequence[Dict[str, Any]]] = None,
         _plan_path: Optional[str] = None,
         _agent_factory: Optional[Callable] = None,
         _state_factory: Optional[Callable] = None,
         _binder: Optional[Callable] = None,
         _evaluator: Any = None,
         _cleanup: Optional[Callable] = None,
         _identity: Optional[Dict[str, Any]] = None,
         _setup: Optional[Callable] = None,
         _setup_factory: Optional[Callable] = None,
         _deadline: Optional[D1.Deadline] = None,
         _supervise: bool = True,
         _ply_cap: int = RULES.PLY_CAP,
         _ply_budget: Optional[int] = None) -> int:
    """PRIVATE. The underscore parameters exist ONLY to drive fail-closed tests."""
    if mode not in _ALL_MODES:
        raise H1Error(f"mode {mode!r} is not permitted")
    match = mode == MATCH_MODE

    # 🔴 BOTH BARRIERS FIRST, BEFORE THE PLAN IS EVEN READ -- and so before any
    # compilation, model load, JVM launch or output. Order is the guarantee here:
    # a refusal that has already created a results file has not refused.
    if match:
        check_gate()
        check_seed_registration()
    # Authorization first, then the outputs: refuse an unauthorized run before
    # inspecting the filesystem at all.
    check_output_paths(results_path, trace_path, require_trace=match)

    # MATCH MODE LOADS ONLY THE PINNED PLAN. `_plan_path` is a test seam and is
    # refused on the match path, because a supplied plan is a supplied design.
    if match and _plan_path is not None:
        raise H1Error("match mode loads the pinned v3 plan only; a supplied plan "
                      "path is a supplied design")
    plan = PLAN.load_h1_plan(_plan_path or PLAN.H1_PLAN_REL)
    tasks = list(_tasks) if _tasks is not None else (
        list(plan["tasks"]) if match else [])

    if match:
        _verify_match_schedule(tasks, plan)
        try:
            REF.validate_schedule_executable(tasks)
        except REF.E4ReferenceError as e:
            raise H1Error(f"the H1 schedule may not be executed: {e}") from None
    else:
        for t in tasks:
            H._assert_not_scheduled(t)

    deadline = _deadline if _deadline is not None else D1.Deadline(RUN_DEADLINE_S)
    if not deadline.started:
        deadline.start()

    if _setup_factory is not None:
        # BOUND TO THE CLOCK THAT IS ALREADY RUNNING, never to a fresh one.
        _setup = _setup_factory(results_path, deadline)

    binder = _binder or H._refuse_binder
    cleanup = _cleanup or H._default_cleanup
    results: List[Dict[str, Any]] = []
    cleanups = 0
    completed = 0
    current = None                      # (index, task) in flight, for the VOID

    supervisor = D1._supervisor(deadline) if _supervise else contextlib.nullcontext()
    with _trace_file(trace_path) as tfh, supervisor:
        _trace(tfh, event="run_start", schema=TRACE_SCHEMA, n_games=len(tasks))
        # 🔴 CONSTRUCTED INSIDE THE PROTECTED BLOCK. It used to be built between
        # `run_start` and the `try`, so a results path that already existed left
        # a trace holding `run_start` AND NOTHING ELSE -- no `run_end`, no
        # diagnostic -- and the raw HarnessError escaped unclassified. The one
        # failure that destroys the results file is the one the trace exists for.
        rec = None
        try:
            try:
                rec = _PlyCountingRecorder(results_path)
            except H.HarnessError as e:
                # Past the preflight, so this is a full disk or a revoked
                # permission -- the instrument failing mid-run. VOID, and the
                # exception now says the same thing the trace does.
                raise H1VoidError(
                    f"the results file could not be created: {e} The H1 match is "
                    f"VOID: no viability report is produced and the seed block "
                    f"retires whole.") from None
            rec.emit({"record_type": "run_header", "mode": mode,
                      "harness": "h1_viability_runner",
                      "plan_sha256": PLAN.H1_PLAN_SHA256,
                      "task_digest": RULES.H1_TASK_DIGEST,
                      "frozen_tasks": len(plan["tasks"]),
                      "scheduled_tasks": len(tasks),
                      "synthetic_tasks": 0 if match else len(tasks),
                      "ply_cap": _ply_cap, "ply_budget": _ply_budget,
                      "no_games": not match,
                      "early_stop": RULES.EARLY_STOP,
                      "n_games": RULES.N_GAMES,
                      "per_call_timeout_s": PER_CALL_TIMEOUT_S,
                      "run_deadline_s": deadline.limit_s,
                      "seed_block": list(RULES.H1_SEED_BLOCK),
                      "identity": _identity or {}})

            if _setup is not None:
                _check_deadline(deadline, "setup")
                try:
                    collaborators = _setup()
                except H.AbortError:
                    raise
                except Exception as e:                        # noqa: BLE001
                    raise H.AbortError(H.PHASE_SETUP,
                                       f"{type(e).__name__}: {e}") from None
                _agent_factory = collaborators["agent_factory"]
                _state_factory = collaborators["state_factory"]
                binder = collaborators["binder"]
                _evaluator = collaborators["evaluator"]
                cleanup = collaborators.get("cleanup", cleanup)
                rec.emit({"record_type": "setup_complete",
                          "artifacts": collaborators.get("artifacts", {})})

            # THE BINDER IS WRAPPED HERE, after setup has supplied the real one.
            # `note_bind` runs BEFORE the binder can fail, so a binder failure --
            # which happens after a move applies but before that move's `ply`
            # record -- names the ply it was attempting, not the last that worked.
            _real_binder = binder

            def binder(task, state, ply, move=None, _b=_real_binder):   # noqa: F811
                rec.note_bind(ply)
                return _b(task, state, ply, move)

            for index, task in enumerate(tasks):
                current = (index, task)
                rec.last_ply = None
                # NO EARLY STOP AND NO SKIP PATH. Every scheduled game is played.
                _check_deadline(deadline, f"before game {index}")
                _trace(tfh, event="task_start", index=index)
                rec.emit({"record_type": "task_start", "task_id": task["task_id"],
                          "seed": task["seed"], "opening": task["opening"],
                          "colour_arm": task["colour_arm"], "rep": task["rep"]})

                def agent_for(t, mover, _task=task):
                    agent = (_agent_factory or H._refuse_factory)(_task, mover,
                                                                 _evaluator)
                    H._enforce_evaluator(agent, _evaluator, _task, mover)
                    return agent

                play_error = None
                outcome = None
                try:
                    outcome = H.play_task(
                        task=task, agent_for=agent_for,
                        state_factory=_state_factory or H._refuse_state_factory,
                        binder=binder, rec=rec, ply_cap=_ply_cap,
                        ply_budget=_ply_budget)
                except BaseException as e:                    # noqa: BLE001
                    play_error = e

                # A COMPLETED GAME IS PERSISTED AND COUNTED BEFORE CLEANUP RUNS,
                # and a failure to WRITE it must not skip cleanup either. Both
                # orderings were defects in the screen's harness before they were
                # fixed there.
                record_error = None
                if play_error is None:
                    row = {"record_type": "task_result", "task_id": task["task_id"],
                           "seed": task["seed"], **outcome}
                    try:
                        rec.emit(row)
                        results.append(row)      # counted only once it is durable
                    except Exception as e:                    # noqa: BLE001
                        record_error = e

                cleanup_error = None
                try:
                    cleanup()
                    cleanups += 1
                except Exception as e:                        # noqa: BLE001
                    cleanup_error = e

                primary = play_error if play_error is not None else record_error
                if primary is not None:
                    if cleanup_error is not None:
                        rec.emit_terminal(
                            {"record_type": "cleanup_failure_after_abort",
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
                    raise H.AbortError(
                        H.PHASE_CLEANUP,
                        f"after {task['task_id']} (whose result is recorded): "
                        f"{cleanup_error}")

                completed += 1
                _trace(tfh, event="task_done", index=index, games_completed=completed)

            code = _report(rec, plan, tasks, results, mode, cleanups)
            _trace(tfh, event="run_end", verdict="OK", games_completed=completed)
            return code
        except BaseException as e:                            # noqa: BLE001
            _trace(tfh, event="run_end", verdict="VOID", games_completed=completed)
            # 🔴 THE ASYNCHRONOUS ALARM RAISES D1's EXCEPTION. `_check_deadline`
            # translates only the COOPERATIVE breach; the reused supervisor fires
            # SIGALRM from inside whatever blocking call is running, and that
            # D1VoidError escaped this runner under the wrong experiment's type.
            # Translated here, at the boundary, so every deadline VOID -- however
            # it was detected -- leaves as H1VoidError.
            if isinstance(e, D1.D1VoidError) and not isinstance(e, H1VoidError):
                e = H1VoidError(f"{e} The H1 match is VOID: no viability report is "
                                f"produced and the seed block retires whole.")
            idx, tsk = (current if current is not None else (None, None))
            # `rec` is None only when the recorder itself could not be created --
            # exactly the case the trace above is the sole record of.
            if rec is not None:
                rec.emit_terminal({"record_type": "void_diagnostic",
                                   **_void_diagnostic(tsk, idx, rec.last_ply, e),
                                   "games_completed": completed,
                                   "tasks_played": len(results)})
                if isinstance(e, H.AbortError):
                    note = rec.emit_terminal({"record_type": "abort",
                                              "phase": e.phase, "message": e.message,
                                              "tasks_played": len(results)})
                    if note:
                        import sys
                        print(f"WARNING: could not record the abort ({note})",
                              file=sys.stderr)
            raise e from None
        finally:
            if rec is not None:
                rec.close()


def _check_deadline(deadline, where: str) -> None:
    """The COOPERATIVE clock check, between stages.

    🔑 IT DOES NOT TRANSLATE. It used to wrap `deadline.check` in its own
    D1VoidError -> H1VoidError conversion, and once the run boundary began
    translating the ASYNCHRONOUS alarm the two overlapped: the boundary caught
    the cooperative breach as well, so this conversion could never fire and an
    injected-defect control removing it was NOT CAUGHT. ONE translation point,
    at the boundary, covering both ways a breach is detected.

    The `Deadline` and its SIGALRM supervisor are REUSED, not copied: their
    ordering was wrong once -- the alarm armed before the clock started, giving
    the enforced and reported windows two different origins -- and was fixed
    once. A second copy would be a second chance to get that ordering wrong.
    """
    deadline.check(where)


def _report(rec, plan, tasks, results, mode, cleanups) -> int:
    """Post-run reporting, delegated ENTIRELY to the frozen H1 rules.

    This module computes no rate, no interval and no verdict of its own.

    THE MODE DECIDES WHAT A REFUSAL MEANS:
      qualify   a refusal is EXPECTED -- synthetic tasks are not the frozen
                design -- and is recorded as a receipt. Exit 0.
      match     CAP_SATURATED_NO_RATE is a FROZEN, PREREGISTERED OUTCOME: the
                match ran correctly and the rule says there is no rate and no
                verdict, so it is recorded as an outcome and exits 0. ANY OTHER
                refusal means the run did not produce the design it claimed to.
    """
    try:
        report = RULES.viability_report(results, plan["tasks"])
    except Exception as e:                                    # noqa: BLE001
        raise H.AbortError(H.PHASE_CLASSIFY, f"{type(e).__name__}: {e}") from None

    if report.get("reported"):
        rec.emit({"record_type": "viability_report", **report,
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
