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
            result = state["harness"].play_task(
                task=dict(task), agent_for=state["agent_factory"],
                state_factory=state["state_factory"], binder=state["binder"],
                rec=None, ply_cap=H2R.PLY_CAP)
        finally:
            state["cleanup"]()
            play.cleanups += 1
        result = dict(result)
        result["elapsed_s"] = time.monotonic() - t0
        return result

    play._state = None
    play.cleanups = 0
    return play


def run_pilot(*, results_path: str, trace_path: str,
              report_path: str) -> Dict[str, Any]:
    """THE PUBLIC ENTRY. Gate FIRST, then the barriers, then nothing else yet.

    🔴 It refuses before it can reach a seam: the gate is shut and no seed block
    exists. The body beyond the barriers is deliberately unbuilt -- wiring a game
    loop that cannot be authorized would be code no test could exercise.
    """
    check_gate()
    check_seed_registration()
    check_output_paths(results_path, trace_path, report_path)
    raise H3PilotRunError(
        "the H3 pilot run body is not implemented: execution is a separate "
        "authorization and the seed block it needs does not exist.")
