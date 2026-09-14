"""H3 PILOT — the gated, supervised runner. CLOSED AND SEEDLESS.

Derived from the launch path attempt 3 proved end to end: a gate the entry reads
FIRST, a second gate check and a containment boundary at the production seam, a
create-only output precheck, and a supervised worker under an outer cap whose
wrapper restores the gate on every exit.

🔴 IT CANNOT RUN, AND NOT ONLY BECAUSE OF THE GATE.
  * `H3_PILOT_EXECUTION_AUTHORIZED` is False.
  * `PILOT_SEED_BLOCK` is **None**: no block has been reserved, proved disjoint or
    registered, so `check_seed_registration` refuses unconditionally. A gate can be
    opened by one reviewed edit; a seed block cannot be conjured by one.
Both are separate authorizations and neither is requested by this module.

⚠ NEVER EXERCISED END TO END. Only construction, wiring and refusals are tested --
the same state H2's runner was in before attempt 3, and the reason that attempt's
risk was recorded before it ran rather than after.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from . import h3_pilot_analysis as ANALYSIS
from . import h3_pilot_rules as RULES


class H3PilotRunError(RuntimeError):
    """A refusal. Never a verdict, and never a strength statement."""


class H3PilotVoidError(H3PilotRunError):
    """The run started and did not complete. NOT a verdict either."""


class H3PilotContainmentError(H3PilotRunError):
    """The production seam was reached from a test process."""


# ═══════════════════════ BARRIER 1: the gate ═══════════════════════════════
H3_PILOT_EXECUTION_AUTHORIZED = False

EXIT_UNAUTHORIZED = 5


def check_gate() -> None:
    """Read the gate, FIRST, before anything effectful.

    A control can delete this call -- one did, on 2026-09-12, and played 383 real
    games -- which is why the seam checks again and why the boundary exists.
    """
    if H3_PILOT_EXECUTION_AUTHORIZED is not True:
        raise H3PilotRunError(
            "the H3 pilot is NOT AUTHORIZED (H3_PILOT_EXECUTION_AUTHORIZED is "
            "False). No opening was generated, no model loaded, no JVM started, no "
            "seed drawn, no game played and no file written. Opening this is a "
            "reviewed one-line edit under its own authorization.")


# ═══════════════════ BARRIER 2: a seed block that does not exist ═══════════
#: 🔴 DELIBERATELY None. The card reserves nothing, and a runner that names a block
#: before one is reserved has spent it on paper. Setting this is a separate
#: authorization that must carry its own collision re-proof against the registries
#: as they stand.
PILOT_SEED_BLOCK: Optional[tuple] = None


def check_seed_registration() -> None:
    """Refuse while no block is reserved, and refuse again while it is unregistered.

    TWO DISTINCT REFUSALS, because they are different facts: no block chosen at
    all, versus a chosen block that the registry does not account for.
    """
    if PILOT_SEED_BLOCK is None:
        raise H3PilotRunError(
            "NO SEED BLOCK IS RESERVED for the H3 pilot. The card reserves none, "
            "and 40 games need 40 accounted seeds. Reserving one is a separate "
            "authorization carrying its own collision re-proof.")
    from . import e4_screen_reference as REF
    lo, hi = PILOT_SEED_BLOCK
    missing = [s for s in range(lo, hi) if not REF.seed_is_accounted(s)]
    if missing:
        raise H3PilotRunError(
            f"the pilot seed block [{lo}, {hi}) is not registered: {len(missing)} "
            f"of {hi - lo} seeds are absent from ACCOUNTED_SEED_INTERVALS "
            f"(first {missing[0]}).")


# ═══════════════════ BARRIER 3: create-only outputs ════════════════════════
def check_output_paths(results_path: str, trace_path: Optional[str],
                       report_path: str) -> None:
    """Three DISTINCT files, none of which may already exist.

    `lexists`, not `exists`: a DANGLING SYMLINK is a path that exists and would be
    followed on write, and `exists` cannot see one.
    """
    if not trace_path:
        raise H3PilotRunError("the pilot requires a trace path; a run with no "
                             "durable trace cannot say what it did")
    paths = [results_path, trace_path, report_path]
    if len({os.path.abspath(p) for p in paths}) != 3:
        raise H3PilotRunError(
            f"the three outputs must be THREE files, got {paths}")
    for p in paths:
        if os.path.lexists(p):
            raise H3PilotRunError(
                f"the output path already exists: {p}. Outputs are create-only, so "
                f"an earlier run's file cannot be mistaken for this one's. Nothing "
                f"has run.")


# ═══════════════════ the schedule, pinned ══════════════════════════════════
def check_schedule(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The schedule must BE the frozen 40, by digest."""
    tasks = [dict(t) for t in tasks]
    if len(tasks) != RULES.N_GAMES:
        raise H3PilotRunError(f"{len(tasks)} tasks, expected {RULES.N_GAMES}")
    got = RULES.task_digest(tasks)
    if got != RULES.TASK_DIGEST:
        raise H3PilotRunError(
            f"the schedule digest is {got} but the frozen pilot schedule is "
            f"{RULES.TASK_DIGEST}. A different schedule is a different experiment "
            f"wearing this one's name.")
    return {"n_tasks": len(tasks), "task_digest": got,
            "pairs": len({t["pair_id"] for t in tasks})}


# ═══════════════════ the production seam ═══════════════════════════════════
def _production_play(results_path: str, deadline: Any = None,
                     openings: Optional[Sequence[Dict[str, Any]]] = None
                     ) -> Callable[..., Dict[str, Any]]:
    """THE REAL PLAY SEAM, built as the qualified commands build it.

    Constructed LAZILY: importing this module starts no JVM and loads no model.
    The one pilot difference from H2 is the opening source -- the GENERATED set,
    keyed by name -- and that each game is TIMED on a MONOTONIC clock.
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
            # BOTH CHECKS, HERE, BEFORE ANY EFFECT -- not because the entry's is
            # redundant, but because a control can delete it, and did.
            check_gate()
            # 🔑 CALLED DIRECTLY IN THE SEAM, as H1 calls it -- not behind a private
            # wrapper. `test_NO_CONTROL_DELETES_AN_AUTHORIZATION_CHECK` reads the
            # seam's own call names to decide whether a gate-removal control is
            # admissible here, and a wrapper hides the boundary from it. ONE OWNER,
            # shared with H1 and H2: it grants nothing and blocks nothing a real run
            # does, but a control that deletes every authorization check still
            # cannot reach a toolchain, a javac, a model or a seed.
            try:
                SCREEN_CMD.assert_production_acts_are_inert(
                    "the H3 pilot's production seam", (
                        (TC, "verified_paths"),
                        (D1, "_default_compile"),
                        (SCREEN_CMD, "_default_load_evaluator"),
                        (HARNESS, "play_task")))
            except SCREEN_CMD.ContainmentError as e:
                raise H3PilotContainmentError(str(e)) from None
            tc = TC.verified_paths()
            java = os.path.join(tc["jdk_home"], "bin", "java")
            classes = results_path + ".t1j_classes"
            paths = D1.T1jPaths(java=java, jar=tc["jar"], classes=classes,
                                ply_cap=RULES.PLY_CAP if hasattr(RULES, "PLY_CAP")
                                else 280)
            if deadline is None or not deadline.started:
                raise H3PilotRunError(
                    "the production seam was given no STARTED deadline; compilation "
                    "checks a clock with no origin and would refuse after creating "
                    "the class directory")
            D1._default_compile(deadline, paths=paths)
            from . import h2_match_rules as H2R
            runtime = INT.T1jRuntime(java=java, jar=tc["jar"], classes=classes,
                                     ply_cap=H2R.PLY_CAP, timeout_s=timeout_s)
            ctx = INT.IntegrationContext()
            evaluator = SCREEN_CMD._default_load_evaluator(".")
            cfg = G3.eval_config()
            argmax_cfg = cfg.__class__(**{**cfg.__dict__,
                                          "selection_mode": H2R.SELECTION_MODE})
            from . import e4_screen_reference as REF
            if openings is None:
                raise H3PilotRunError(
                    "the production seam was given no openings; the pilot's "
                    "positions are GENERATED and pinned, never implied")
            state = play._state = {
                "state_factory": INT.make_state_factory(
                    RULES.openings_mapping(openings), ctx),
                "binder": INT.make_binder(runtime, ctx),
                "agent_factory": INT.make_agent_factory(
                    runtime=runtime, ctx=ctx, evaluator=evaluator,
                    t1j_timeout_s=timeout_s,
                    reference_build=lambda t, evaluator: G3.build_reference_agent(
                        task=t, evaluator=evaluator,
                        colour=REF.reference_colour(t), config=argmax_cfg,
                        capture=True)),
                "harness": HARNESS,
                "cleanup": SCREEN_CMD._default_cleanup,
            }
        check_gate()                      # EVERY game, not only the first
        # 🔑 MONOTONIC, not wall clock. A wall clock can step backwards over an
        # NTP correction and produce a negative duration; the analysis refuses
        # those, so a clock that can emit one would void a game for the weather.
        t0 = time.monotonic()
        try:
            from . import h2_match_rules as H2R
            out = state["harness"].play_task(
                task=dict(task), agent_for=state["agent_factory"],
                state_factory=state["state_factory"], binder=state["binder"],
                rec=None, ply_cap=H2R.PLY_CAP)
        finally:
            state["cleanup"]()
            play.cleanups += 1
        return {**dict(out), "elapsed_s": time.monotonic() - t0}

    play._state = None
    play.cleanups = 0
    return play


def run_pilot(*, results_path: str, trace_path: str,
              report_path: str) -> Dict[str, Any]:
    """THE PUBLIC ENTRY. Takes the three OUTPUT PATHS and nothing else.

    🔴 EVERY OTHER INPUT IS RESOLVED HERE, so none can be supplied. H2 records why:
    an entry that accepted `tasks`, `play` or `identity` would let an opened gate
    authorize CALLER-SUPPLIED GAMEPLAY through the API while the CLI could not run
    the real thing -- forged play with a real report on one side, no production
    path on the other. The injection seams live on the PRIVATE entry, for tests.
    """
    check_gate()
    from . import d1_probe as D1
    openings = RULES.generate_openings()
    if RULES.opening_set_digest(openings) != RULES.OPENING_SET_DIGEST:
        raise H3PilotRunError(
            f"the generated opening set does not match the frozen pin "
            f"{RULES.OPENING_SET_DIGEST}; the positions are not the ones the card "
            f"fixed and the pilot would answer a different question.")
    if PILOT_SEED_BLOCK is None:
        check_seed_registration()                  # refuses: no block is reserved
    tasks = RULES.build_tasks(openings, seed_interval=PILOT_SEED_BLOCK)
    deadline = D1.Deadline(RULES.RUN_DEADLINE_S)
    deadline.start()                    # ONE origin, before anything effectful
    return _run_pilot_unguarded(
        tasks=tasks, openings=openings, results_path=results_path,
        trace_path=trace_path, report_path=report_path,
        play=_production_play(results_path, deadline, openings),
        deadline_s=RULES.RUN_DEADLINE_S, _deadline=deadline)


def _run_pilot_unguarded(*, tasks, openings, results_path, trace_path,
                         report_path, play, deadline_s=None, _deadline=None,
                         _supervisor=None) -> Dict[str, Any]:
    """Everything below the gate. PRIVATE, and never a way around `run_pilot`.

    It exists so the machinery can be tested WITHOUT lifting the gate, with inert
    collaborators.

    🔑 THE ONE PLACE THIS DIFFERS FROM H2, AND IT IS THE CARD'S POINT. H2 raises
    VOID on a short schedule because its verdict needs all 736 games. THE PILOT'S
    OUTPUTS ARE DIAGNOSTICS, and diagnostics over completed pairs stay valid, so
    running out of time STOPS CLEANLY and still reports -- verdict PARTIAL, never
    OK. An EXCEPTION is still a VOID; a deadline is not.
    """
    import contextlib
    import json
    import time

    check_seed_registration()
    check_output_paths(results_path, trace_path, report_path)
    summary = check_schedule(tasks)
    deadline_s = RULES.RUN_DEADLINE_S if deadline_s is None else deadline_s

    from . import d1_probe as _D1
    deadline = _deadline if _deadline is not None else _D1.Deadline(deadline_s)
    if not deadline.started:
        deadline.start()

    games: list = []
    timed_out = False
    trace = rec = None
    stack = contextlib.ExitStack()
    t_start = time.monotonic()
    try:
        # ONE AT A TIME, EACH REGISTERED AS IT IS ACQUIRED: opening both before
        # tracking either leaks the first descriptor when the second open fails.
        trace = stack.enter_context(os.fdopen(
            os.open(trace_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "w"))
        rec = stack.enter_context(os.fdopen(
            os.open(results_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "w"))
        sup = _supervisor if _supervisor is not None else _D1._supervisor
        stack.enter_context(sup(deadline))

        def emit(fh, obj):
            fh.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

        emit(trace, {"event": "run_start", "n_tasks": len(tasks),
                     "pairs": summary["pairs"]})
        emit(rec, {"record_type": "header", "design": "H3_PILOT",
                   "task_digest": summary["task_digest"],
                   "opening_set_digest": RULES.opening_set_digest(openings)})

        for i, task in enumerate(tasks):
            if deadline.elapsed() > deadline_s:
                # 🔑 NOT A VOID. The card says so: a timeout still measures what
                # the completed games cost, and the diagnostics over completed
                # PAIRS stay valid. It stops, and the report says PARTIAL.
                timed_out = True
                emit(trace, {"event": "deadline_reached", "index": i,
                             "games_completed": len(games)})
                break
            emit(trace, {"event": "task_start", "index": i})
            out = play(task=task, identity={}, timeout_s=RULES.PER_CALL_TIMEOUT_S)
            row = dict(out["result"])
            t = ANALYSIS.transcript(out["plies"], row,
                                    opening_bound=out["opening_bound"])
            digest = ANALYSIS.transcript_digest(t)
            # THE TRANSCRIPT EVIDENCE IS PERSISTED, not merely used: a screen whose
            # input is unrecorded is a number to be taken on trust.
            for r in out.get("records", out["plies"]):
                emit(rec, {"record_type": r.get("record_type", "ply"), **r})
            emit(rec, {"record_type": "transcript", "task_id": task["task_id"],
                       "pair_id": task["pair_id"],
                       "incumbent_colour": task["incumbent_colour"],
                       "opening_bound": out["opening_bound"],
                       "n_plies": len(out["plies"]), "transcript_digest": digest})
            game = {"task_id": task["task_id"], "pair_id": task["pair_id"],
                    "incumbent_colour": task["incumbent_colour"],
                    "transcript_digest": digest,
                    "terminal_reason": row.get("terminal_reason"),
                    "winner": row.get("winner"),
                    "t1j_points": row.get("t1j_points"),
                    "plies": row.get("plies"),
                    "elapsed_s": out["elapsed_s"]}
            games.append(game)
            emit(rec, {"record_type": "task_result", **game})
            emit(trace, {"event": "task_done", "index": i,
                         "games_completed": len(games)})

        report = ANALYSIS.summarise(games, total_elapsed_s=time.monotonic() - t_start)
        report["timed_out"] = timed_out
        fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True, default=str)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        # THE TERMINAL RECORD COMES LAST, AFTER THE REPORT IS DURABLE -- H2's
        # lesson, where run_end/OK preceded the report and a write failure left
        # the trace and the exit code disagreeing about the same run.
        emit(trace, {"event": "run_end",
                     "verdict": "PARTIAL" if timed_out else "OK",
                     "games_completed": len(games),
                     "any_stop_rule_fired": report["any_fired"]})
        return report
    except BaseException as e:                               # noqa: BLE001
        verdict = "INTERRUPTED" if isinstance(e, KeyboardInterrupt) else "VOID"
        if trace is not None:
            try:
                trace.write(json.dumps(
                    {"event": "run_end", "verdict": verdict,
                     "games_completed": len(games),
                     "error": type(e).__name__}, sort_keys=True) + "\n")
                trace.flush()
                os.fsync(trace.fileno())
            except Exception:                                # noqa: BLE001
                pass
        if isinstance(e, H3PilotRunError):
            raise
        if isinstance(e, Exception):
            raise H3PilotVoidError(f"{type(e).__name__}: {e}") from e
        raise
    finally:
        stack.close()                     # every acquired descriptor, in reverse
