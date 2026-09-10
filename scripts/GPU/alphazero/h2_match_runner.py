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
        # 🔴 `lexists`, NOT `exists` -- H1's own correction, which I repeated as a
        # defect. `os.path.exists` FOLLOWS the link, so a DANGLING symlink reads as
        # absent, `O_EXCL` then fails on the link itself, and a create-only
        # guarantee turns into a mid-run error.
        if os.path.lexists(path):
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


def _same(a: Any, b: Any, where: str) -> None:
    """RECURSIVE and TYPE-STRICT equality, refusing by path.

    🔑 `False == 0` and `6 == 6.0` are refusals here, as everywhere else in this
    programme: a recorded configuration that merely compares equal to the frozen
    one is not the frozen one.
    """
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if not (isinstance(a, Mapping) and isinstance(b, Mapping)):
            raise H2VoidError(f"{where}: {a!r} and {b!r} are not both mappings")
        extra, missing = set(a) - set(b), set(b) - set(a)
        if extra or missing:
            raise H2VoidError(
                f"{where}: the recorded identity has extra {sorted(extra)} and is "
                f"missing {sorted(missing)} against the frozen configuration")
        for k in sorted(b):
            _same(a[k], b[k], f"{where}.{k}")
        return
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if len(list(a or [])) != len(list(b or [])):
            raise H2VoidError(f"{where}: sequence lengths differ ({a!r} vs {b!r})")
        for i, (x, y) in enumerate(zip(list(a), list(b))):
            _same(x, y, f"{where}[{i}]")
        return
    if type(a) is not type(b) or a != b:
        raise H2VoidError(
            f"{where}: recorded {a!r} ({type(a).__name__}) but the frozen "
            f"configuration gives {b!r} ({type(b).__name__})")


def check_incumbent_identity(identity: Mapping[str, Any]) -> None:
    """The recorded identity must BE H2's -- THE WHOLE OF IT.

    🔴 An earlier version compared three top-level fields and three eval fields, so
    a changed evaluation batch size, stall-flush count, noise suppression, RNG mask,
    readout path or agent lifetime all passed. Everything the frozen identity
    carries is now compared recursively and type-strictly, and `selection_mode` is
    reported FIRST because it is the study.
    """
    want = frozen_incumbent_identity()
    got_mode = (identity.get("eval_config") or {}).get("selection_mode")
    if got_mode != RULES.SELECTION_MODE:
        raise H2VoidError(
            f"the recorded incumbent identity carries selection_mode {got_mode!r}, not "
            f"{RULES.SELECTION_MODE!r}. H2 IS the readout change: a run recording the "
            f"old readout did not make it, and reporting it as H2 would attribute a "
            f"result to a configuration that never played.")
    _same(dict(identity), want, "incumbent_identity")


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
    # 🔴 THE DIGEST COVERS THE DESIGN DIMENSIONS ONLY. A schedule with a forged
    # `reference_sha256` or forged `rng_streams` keeps the same digest and passed --
    # so every task is compared FIELD BY FIELD, recursively and type-strictly,
    # against the schedule this repository builds from the pinned source plan.
    canonical = PLAN.build_tasks(PLAN.load_source_plan())
    if len(canonical) != len(tasks):
        raise H2Error(f"{len(tasks)} tasks supplied, {len(canonical)} canonical")
    for i, (got, want) in enumerate(zip(tasks, canonical)):
        _same(dict(got), want, f"task[{i}]")
    REF.validate_schedule_executable(list(tasks))
    return summary


def run_h2(*, results_path: str, trace_path: str) -> Dict[str, Any]:
    """THE PUBLIC ENTRY. Takes the two OUTPUT PATHS and nothing else.

    🔴 EVERY OTHER INPUT IS RESOLVED HERE, so none can be supplied. An earlier
    version accepted `tasks`, `play`, `identity` and `deadline_s`, which meant
    opening the gate would have authorized CALLER-SUPPLIED GAMEPLAY through the
    API while the CLI could not run the real match at all: forged play with a
    real verdict on one side, no production path on the other.

    The schedule comes from the pinned source plan, the identity from the frozen
    configuration, the deadline from this module's constant, and the play seam is
    constructed here. The injection seams remain on the PRIVATE entry, for tests.
    """
    check_gate()
    tasks = PLAN.build_tasks(PLAN.load_source_plan())
    return _run_h2_unguarded(tasks=tasks, results_path=results_path,
                             trace_path=trace_path, play=_production_play(results_path),
                             identity=frozen_incumbent_identity(),
                             deadline_s=RUN_DEADLINE_S)


def _production_play(results_path: str) -> Callable[..., Dict[str, Any]]:
    """THE REAL PLAY SEAM, built exactly as the qualified commands build it.

    🔴 ITS ABSENCE WAS INVISIBLE, the defect H1 recorded in the same place: every
    passing test replaced the seam, so a missing production path could not fail a
    test. It is constructed lazily -- importing this module starts no JVM and
    loads no model -- and the collaborators are the qualified ones:
    `t1j_toolchain.verified_paths` for the pinned jar and JDK, `d1_probe`'s
    compile step, `e4_screen_integration` for the binder, state and agent
    factories, and `e4_screen_runner.play_task` for the game loop.

    🔑 THE ONE H2 DIFFERENCE IS THE CONFIG PASSED TO THE INCUMBENT'S BUILDER: an
    `EvalConfig` whose `selection_mode` is `argmax`, which `readout_from_eval_config`
    turns into `ReadoutConfig(mode=MODE_ARGMAX)`. That is the whole gameplay change,
    applied at the one place the readout is chosen.

    ⚠ NEVER EXERCISED END TO END. Only its construction and its refusals are
    tested; nothing in this repository has played a game with it.
    """
    def play(*, task: Mapping[str, Any], identity: Mapping[str, Any],
             timeout_s: float) -> Dict[str, Any]:
        from . import d1_probe as D1
        from . import e4_screen_command as SCREEN_CMD
        from . import e4_screen_integration as INT
        from . import e4_screen_runner as HARNESS
        from . import t1j_toolchain as TC
        from . import twixtbot_g3_reference as G3

        state = play._state
        if state is None:
            tc = TC.verified_paths()
            java = os.path.join(tc["jdk_home"], "bin", "java")
            classes = results_path + ".t1j_classes"
            paths = D1.T1jPaths(java=java, jar=tc["jar"], classes=classes,
                                ply_cap=RULES.PLY_CAP)
            D1._default_compile(D1.Deadline(RUN_DEADLINE_S), paths=paths)
            runtime = INT.T1jRuntime(java=java, jar=tc["jar"], classes=classes,
                                     ply_cap=RULES.PLY_CAP, timeout_s=timeout_s)
            ctx = INT.IntegrationContext()
            evaluator = SCREEN_CMD._default_load_evaluator(".")   # the incumbent, ONCE
            cfg = G3.eval_config()
            # THE GAMEPLAY-RULE CHANGE, at the one place the readout is chosen.
            argmax_cfg = cfg.__class__(**{**cfg.__dict__,
                                          "selection_mode": RULES.SELECTION_MODE})
            openings = PLAN.load_source_plan()["openings"]
            state = play._state = {
                "state_factory": INT.make_state_factory(openings, ctx),
                "binder": INT.make_binder(runtime, ctx),
                "agent_factory": INT.make_agent_factory(
                    runtime=runtime, ctx=ctx, evaluator=evaluator,
                    t1j_timeout_s=timeout_s,
                    reference_build=lambda t, evaluator: G3.build_reference_agent(
                        task=t, evaluator=evaluator,
                        colour=REF.reference_colour(t), config=argmax_cfg,
                        capture=True)),
                "harness": HARNESS,
            }
        cap = _CapturingRecorder()
        result = state["harness"].play_task(
            task=dict(task), agent_for=state["agent_factory"],
            state_factory=state["state_factory"], binder=state["binder"],
            rec=cap, ply_cap=RULES.PLY_CAP)
        plies = [r for r in cap.records if r.get("record_type") == "ply"]
        bounds = [r for r in cap.records if r.get("record_type") == "opening_bound"]
        if len(bounds) != 1:
            raise H2VoidError(
                f"{task['task_id']}: {len(bounds)} opening_bound records; the transcript's "
                f"first ply is anchored to exactly one, and without it the ply span "
                f"cannot be checked at all")
        return {"result": {"task_id": task["task_id"], "seed": task["seed"], **result},
                "plies": plies, "opening_bound": bounds[0]["ply"],
                "records": cap.records}

    play._state = None
    return play


class _CapturingRecorder:
    """The harness's recorder interface, captured IN MEMORY.

    The harness emits `opening_bound` and one `ply` record per move; H2 needs both
    to build a transcript, and the run needs them PERSISTED so the reported
    diversity can be recomputed by someone who was not there. This collects them
    and the caller writes them out.
    """

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def emit(self, obj: Mapping[str, Any]) -> None:
        self.records.append(dict(obj))


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
    # 🔴 CREATION HAPPENS INSIDE THE PROTECTED BLOCK. Opening the files before the
    # `try` meant a failure between the two opens leaked a descriptor and left the
    # trace file with no `run_end` -- the record a VOID depends on.
    trace = rec = None
    try:
        tfd = os.open(trace_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        rfd = os.open(results_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        trace, rec = os.fdopen(tfd, "w"), os.fdopen(rfd, "w")

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
                                 opening_bound=out["opening_bound"])
            digest = RULES.transcript_digest(t)
            # 🔴 THE TRANSCRIPT EVIDENCE IS PERSISTED, not just used. Only the task
            # result was written, so the reported diversity could not be recomputed
            # by anyone who was not there -- a screen whose input is unrecorded is a
            # number to be taken on trust.
            for r in out.get("records", out["plies"]):
                emit(rec, {"record_type": r.get("record_type", "ply"), **r})
            emit(rec, {"record_type": "transcript", "task_id": task["task_id"],
                       "opening": task["opening"], "colour_arm": task["colour_arm"],
                       "opening_bound": out["opening_bound"],
                       "n_plies": len(out["plies"]), "transcript_digest": digest})
            per_game.append({"task_id": task["task_id"], "opening": task["opening"],
                             "colour_arm": task["colour_arm"],
                             "transcript_digest": digest})
            emit(trace, {"event": "task_done", "index": i,
                         "games_completed": len(results)})
        emit(trace, {"event": "run_end", "verdict": "OK",
                     "games_completed": len(results)})
    except BaseException as e:                               # noqa: BLE001
        # 🔴 EVERY MID-RUN EXIT LEAVES A run_end/VOID. Only the deadline wrote one
        # before, so a refusal, a crash or an interrupt left a trace that stopped
        # mid-sentence and could not say the run was void.
        if trace is not None:
            try:
                trace.write(json.dumps(
                    {"event": "run_end", "verdict": "VOID",
                     "games_completed": len(results),
                     "error": type(e).__name__}, sort_keys=True) + "\n")
                trace.flush()
                os.fsync(trace.fileno())
            except Exception:                                # noqa: BLE001
                pass
        if isinstance(e, H2Error):
            raise
        if isinstance(e, Exception):
            raise H2VoidError(f"{type(e).__name__}: {e}") from e
        raise
    finally:
        for fh in (trace, rec):
            if fh is not None:
                try:
                    fh.close()
                except Exception:                            # noqa: BLE001
                    pass

    if len(results) != RULES.N_GAMES:
        raise H2VoidError(
            f"{len(results)} of {RULES.N_GAMES} games completed; a partial schedule "
            f"produces no analysis.")
    # 🔑 THE SCREEN RUNS INSIDE THE REPORT, BEFORE THE INTERVAL. A screen that runs
    # after the number it guards is decoration -- see `h2_match_rules.h2_report`.
    return RULES.h2_report(results, list(tasks), per_game,
                           task_digest=summary["task_digest"])
