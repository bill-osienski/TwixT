"""H3 FULL STUDY — the segment runner. GATE-SHUT AND SEEDLESS.

THREE BARRIERS, and none of them is the other's fault:
  1. `H3_STUDY_EXECUTION_AUTHORIZED` is False. One reviewed edit opens it.
  2. `STUDY_SEED_BLOCK` is None and no block is registered, so
     `check_seed_registration` refuses unconditionally. A gate can be opened by
     one edit; a seed block cannot be conjured by one.
  3. The production seam re-checks the gate and hits a containment boundary that
     refuses inside a test process.

Generating the co-produced stratum is a SEPARATE run behind a SEPARATE gate in
`h3_study_generator` — one switch for both would let a generation approval
authorize a match.

⚠ NEVER EXERCISED END TO END. Only construction, wiring and refusals are tested.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from . import h3_study_analysis as ANALYSIS
from . import h3_study_rules as RULES


class H3StudyRunError(RuntimeError):
    """A refusal. Never a verdict."""


class H3StudyVoidError(H3StudyRunError):
    """The segment started and did not complete. Not a verdict either."""


class H3StudyContainmentError(H3StudyRunError):
    """The production seam was reached from a test process."""


# ═══════════════════════ BARRIER 1: the gate ═══════════════════════════════
H3_STUDY_EXECUTION_AUTHORIZED = False

EXIT_UNAUTHORIZED = 5


def check_gate() -> None:
    """Read the gate, FIRST, before anything effectful."""
    if H3_STUDY_EXECUTION_AUTHORIZED is not True:
        raise H3StudyRunError(
            "the H3 full study is NOT AUTHORIZED "
            "(H3_STUDY_EXECUTION_AUTHORIZED is False). No segment may run. "
            "Opening this is a reviewed code change, and each of the four "
            "segments is authorized separately besides.")


# ═══════════════════════ BARRIER 2: the seeds ══════════════════════════════
#: 🔴 NO BLOCK IS RESERVED. 592 seeds, four contiguous 148-seed quarters, need a
#: fresh interval with its own collision re-proof -- which must ALSO cover the
#: generation ranges, because generation seeds derive streams too.
STUDY_SEED_BLOCK: Optional[tuple] = None


def check_seed_registration() -> None:
    """Refuse while no block is reserved, and again while it is unregistered."""
    if STUDY_SEED_BLOCK is None:
        raise H3StudyRunError(
            f"NO SEED BLOCK IS RESERVED for the H3 full study. The card reserves "
            f"none, and {RULES.N_GAMES} games need {RULES.N_GAMES} accounted "
            f"seeds. Reserving one is a separate authorization carrying its own "
            f"collision re-proof, which must cover the GENERATION ranges too.")
    from . import e4_screen_reference as REF
    lo, hi = STUDY_SEED_BLOCK
    missing = [s for s in range(lo, hi) if not REF.seed_is_accounted(s)]
    if missing:
        raise H3StudyRunError(
            f"the study seed block [{lo}, {hi}) is not registered: {len(missing)} "
            f"of {hi - lo} seeds are absent from ACCOUNTED_SEED_INTERVALS "
            f"(first {missing[0]}).")


# ═══════════════════════ BARRIER 3: the incumbent's identity ═══════════════
def frozen_argmax_config():
    """THE ONE construction of the configuration the incumbent PLAYS under.

    The seam carries THIS object and the identity is read off THIS object, so the
    record and the run cannot be two different configurations. The pilot learned
    that the hard way: same function, two objects, one of them describing a run
    that never happened.
    """
    from . import h2_match_rules as H2R
    from . import twixtbot_g3_reference as G3
    cfg = G3.eval_config()
    return cfg.__class__(**{**cfg.__dict__, "selection_mode": H2R.SELECTION_MODE})


def frozen_incumbent_identity(config) -> Dict[str, Any]:
    """H2's frozen argmax identity, INHERITED, read off THE OBJECT THAT WILL PLAY.

    `config` is REQUIRED and this function will not make one.
    """
    from . import e4_screen_reference as REF
    from . import h2_match_runner as H2RUN
    ident = dict(H2RUN.frozen_incumbent_identity())
    ident["argmax_config"] = {f: getattr(config, f)
                              for f in REF.frozen_settings()["eval_config"]}
    ident["design"] = "H3_FULL_STUDY"
    return ident


def check_incumbent_identity(identity: Mapping[str, Any], config) -> None:
    """Three questions in order: the cross-check between the object and the
    qualified path; `selection_mode`, which IS the study; then the whole recorded
    identity against the frozen one, type-strictly."""
    from . import h2_match_rules as H2R
    from . import h2_match_runner as H2RUN
    want = frozen_incumbent_identity(config)

    merged = {**want["eval_config"], **want["inert_under_argmax"]}
    try:
        H2RUN._same(want["argmax_config"], merged, "argmax_config")
    except H2RUN.H2VoidError as e:
        raise H3StudyRunError(
            f"the configuration the builder is handed disagrees with the "
            f"qualified path's frozen settings: {e}") from None

    got_mode = (identity.get("eval_config") or {}).get("selection_mode")
    if got_mode != H2R.SELECTION_MODE:
        raise H3StudyRunError(
            f"the recorded incumbent identity carries selection_mode {got_mode!r}, "
            f"not {H2R.SELECTION_MODE!r}. The study plays the frozen argmax "
            f"configuration; a record naming the old readout describes a run that "
            f"did not make the change.")
    try:
        H2RUN._same(dict(identity), want, "incumbent_identity")
    except H2RUN.H2VoidError as e:
        raise H3StudyRunError(str(e)) from None


def _require_seam_config(play) -> Any:
    """The object the seam will hand the builder — never a fresh derivation."""
    config = getattr(play, "config", None)
    if config is None:
        raise H3StudyRunError(
            "the play seam carries no configuration object, so the identity would "
            "describe something other than what plays. Every seam -- production "
            "or inert -- must declare the configuration it represents.")
    return config


# ═══════════════════════ BARRIER 4: create-only outputs ════════════════════
OUT_ROOT = "docs/superpowers/evidence"


def segment_out_dir(segment: int) -> str:
    """Each segment writes into ITS OWN directory, so no segment can overwrite
    another's record or an earlier experiment's."""
    _check_segment(segment)
    return f"{OUT_ROOT}/2026-09-15-t1j-h3-study-segment{segment}"


def check_output_paths(results_path: str, trace_path: Optional[str],
                       report_path: str) -> None:
    """CREATE-ONLY. `lexists`, not `exists`: a dangling symlink is a path that
    exists and would be followed on write."""
    if not trace_path:
        raise H3StudyRunError("a run with no durable trace cannot say what it did")
    paths = [results_path, trace_path, report_path]
    if len({os.path.abspath(p) for p in paths}) != 3:
        raise H3StudyRunError(f"the three outputs must be THREE files, got {paths}")
    for p in paths:
        if os.path.lexists(p):
            raise H3StudyRunError(
                f"the output path already exists: {p}. Outputs are create-only.")


# ═══════════════════════ the segment schedule ══════════════════════════════
def _check_segment(segment: Any) -> int:
    if type(segment) is not int or isinstance(segment, bool):
        raise H3StudyRunError(
            f"segment is {segment!r} ({type(segment).__name__}); an int is required")
    if not 0 <= segment < RULES.N_SEGMENTS:
        raise H3StudyRunError(
            f"segment {segment} is outside the planned 0..{RULES.N_SEGMENTS - 1}")
    return segment


def segment_schedule(tasks: Sequence[Mapping[str, Any]],
                     segment: int) -> List[Dict[str, Any]]:
    """The 148 tasks of one segment, in plan order."""
    _check_segment(segment)
    out = [dict(t) for t in tasks if t["segment"] == segment]
    if len(out) != RULES.GAMES_PER_SEGMENT:
        raise H3StudyRunError(
            f"segment {segment} holds {len(out)} tasks, expected "
            f"{RULES.GAMES_PER_SEGMENT}")
    return out


def segment_digest(tasks: Sequence[Mapping[str, Any]], segment: int) -> str:
    """THE SEGMENT'S OWN full-field digest. Each segment is its own one-shot
    schedule and so carries its own pin."""
    return RULES.task_digest(segment_schedule(tasks, segment))


def check_segment_schedule(tasks: Sequence[Mapping[str, Any]], segment: int,
                           want_digest: str) -> Dict[str, Any]:
    """The segment must BE its frozen 148, and its seeds must still be runnable."""
    _check_segment(segment)
    tasks = [dict(t) for t in tasks]
    if len(tasks) != RULES.GAMES_PER_SEGMENT:
        raise H3StudyRunError(
            f"{len(tasks)} tasks, expected {RULES.GAMES_PER_SEGMENT}")
    if {t["segment"] for t in tasks} != {segment}:
        raise H3StudyRunError(f"the tasks are not all from segment {segment}")
    got = RULES.task_digest(tasks)
    if got != want_digest:
        raise H3StudyRunError(
            f"the schedule digest is {got} but segment {segment}'s frozen "
            f"schedule is {want_digest}. A different schedule is a different "
            f"experiment wearing this one's name.")
    seedless = [t["task_id"] for t in tasks if t.get("seed") is None]
    if seedless:
        raise H3StudyRunError(
            f"{len(seedless)} of {len(tasks)} tasks carry no seed (first "
            f"{seedless[0]}). An unseeded schedule is a design identity, not a "
            f"runnable plan.")
    # 🔑 THE PIN IS NOT THE WHOLE ANSWER: it fixes the schedule forever, but the
    # REGISTRY MOVES -- these seeds become EXPOSED the moment the segment draws.
    from . import e4_screen_reference as REF
    try:
        REF.validate_schedule_executable(list(tasks))
    except REF.E4ReferenceError as e:
        raise H3StudyRunError(f"the registry refuses this schedule: {e}") from None
    return {"n_tasks": len(tasks), "segment": segment, "task_digest": got,
            "pairs": len({t["pair_id"] for t in tasks})}


# ═══════════════════════ combining segments (card §5.5, §5.6) ══════════════
def combine_segments(segments: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Whether the completed segments may support a verdict at all.

    🔴 TWO WAYS A VERDICT IS BLOCKED, and neither is about the numbers:

    * **A segment WITHHELD after any outcome was inspected.** The four runs are
      separately authorized operationally -- which is what makes a late failure
      survivable, and is also the opening for optional stopping. Stopping because
      the first segments "look decided" invalidates the interval.
    * **A VOIDed segment.** H2's attempt 3 died because one cell ran ~6x the mean:
      a timeout correlates with game length, which correlates with cap rate, which
      is an outcome. So a VOID is not a clean exclusion, the primary is FLAGGED,
      and no verdict issues without a separate review.
    """
    rows = [dict(s) for s in segments]
    for s in rows:
        if s.get("status") not in ("completed", "withheld", "void"):
            raise H3StudyRunError(f"segment status {s.get('status')!r}")
        if not isinstance(s.get("outcomes_inspected"), bool):
            raise H3StudyRunError(
                f"segment {s.get('segment')} does not say whether outcomes had "
                f"been inspected; the report must state it for every segment")
    inspected = any(s["outcomes_inspected"] for s in rows)
    withheld = [s for s in rows if s["status"] == "withheld"]
    voided = [s for s in rows if s["status"] == "void"]

    if withheld and inspected:
        return {"verdict_permitted": False, "flagged": True, "segments": rows,
                "why": ("a segment was WITHHELD after outcomes had been "
                        "INSPECTED; that is optional stopping and no strength "
                        "verdict may issue")}
    if voided:
        return {"verdict_permitted": False, "flagged": True, "segments": rows,
                "why": (f"{len(voided)} segment(s) VOIDed; a timeout correlates "
                        f"with game length and so with an outcome, so the primary "
                        f"rests on a non-random subset and needs a separate "
                        f"review before any verdict")}
    return {"verdict_permitted": True, "flagged": bool(withheld), "segments": rows,
            "why": ("every segment either completed or was withheld without any "
                    "outcome being inspected")}


# ═══════════════════════ the production seam ═══════════════════════════════
def _capturing_recorder():
    """H2's recorder, REUSED rather than copied."""
    from . import h2_match_runner as H2RUN
    return H2RUN._CapturingRecorder()


def _play_one(*, task: Mapping[str, Any], state: Mapping[str, Any],
              timeout_s: float, _count: Any = None) -> Dict[str, Any]:
    """ONE game through the real harness, returning the run body's contract."""
    from . import h2_match_rules as H2R
    t0 = time.monotonic()                      # MONOTONIC, never a wall clock
    cap = _capturing_recorder()
    try:
        result = state["harness"].play_task(
            task=dict(task), agent_for=state["agent_factory"],
            state_factory=state["state_factory"], binder=state["binder"],
            rec=cap, ply_cap=H2R.PLY_CAP)
    finally:
        state["cleanup"]()
        if _count is not None:
            _count.cleanups += 1
    elapsed = time.monotonic() - t0
    plies = [r for r in cap.records if r.get("record_type") == "ply"]
    bounds = [r for r in cap.records if r.get("record_type") == "opening_bound"]
    if len(bounds) != 1:
        raise H3StudyVoidError(
            f"{task['task_id']}: {len(bounds)} opening_bound records; the "
            f"transcript's first ply is anchored to exactly one")
    # 🔴 §6.2: the record carries its OWN seed, stratum and opening digest, so
    # accounting is READ. The pilot's dropped the seed and had to derive it.
    return {"result": {"task_id": task["task_id"], "seed": task.get("seed"),
                       "stratum": task.get("stratum"),
                       "segment": task.get("segment"),
                       "opening_digest": task.get("opening_digest"),
                       **result},
            "plies": plies, "opening_bound": bounds[0]["ply"],
            "records": cap.records, "elapsed_s": elapsed}


def _production_play(results_path: str, deadline: Any = None,
                     openings: Optional[Sequence[Dict[str, Any]]] = None,
                     config: Any = None) -> Callable[..., Dict[str, Any]]:
    """THE REAL PLAY SEAM. Constructed LAZILY; `config` is CARRIED, not made."""
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
            check_gate()                      # the gate, before anything effectful
            try:
                SCREEN_CMD.assert_production_acts_are_inert(
                    "H3 study's production seam", (
                        (TC, "verified_paths"),
                        (D1, "_default_compile"),
                        (SCREEN_CMD, "_default_load_evaluator"),
                        (HARNESS, "play_task")))
            except SCREEN_CMD.ContainmentError as e:
                raise H3StudyContainmentError(str(e)) from None
            tc = TC.verified_paths()
            java = os.path.join(tc["jdk_home"], "bin", "java")
            classes = results_path + ".t1j_classes"
            from . import h2_match_rules as H2R
            paths = D1.T1jPaths(java=java, jar=tc["jar"], classes=classes,
                                ply_cap=H2R.PLY_CAP)
            if deadline is None or not deadline.started:
                raise H3StudyRunError(
                    "the production seam was given no STARTED deadline")
            D1._default_compile(deadline, paths=paths)
            runtime = INT.T1jRuntime(java=java, jar=tc["jar"], classes=classes,
                                     ply_cap=H2R.PLY_CAP, timeout_s=timeout_s)
            ctx = INT.IntegrationContext()
            evaluator = SCREEN_CMD._default_load_evaluator(".")
            argmax_cfg = play.config
            if argmax_cfg is None:
                raise H3StudyRunError(
                    "the production seam was given no configuration object. The "
                    "identity records the object the builder is handed, so the "
                    "seam may not construct one of its own.")
            from . import e4_screen_reference as REF
            if openings is None:
                raise H3StudyRunError(
                    "the production seam was given no openings; the study's "
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
        return _play_one(task=task, state=state, timeout_s=timeout_s, _count=play)

    play._state = None
    play.cleanups = 0
    play.config = config
    return play


def load_opening_set(path: str) -> List[Dict[str, Any]]:
    """The PINNED opening set, both strata, from the generation run's artifact.

    🔴 IT DOES NOT EXIST YET, and this refuses rather than inventing one. The
    co-produced stratum comes from an authorized generation run (`h3_study_
    generator`), and until that has happened there is no population to play over.
    """
    import json
    if not os.path.lexists(path):
        raise H3StudyRunError(
            f"the pinned opening set {path} does not exist. The co-produced "
            f"stratum is produced by a separate AUTHORIZED GENERATION RUN; until "
            f"it has run there is no study population, and a stub may never stand "
            f"in for one.")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    openings = doc["openings"]
    RULES.check_opening_set(openings)          # refuses stubs, composition, orders
    got = RULES.opening_set_digest(openings)
    if got != doc.get("opening_set_digest"):
        raise H3StudyRunError(
            f"the opening set digest is {got} but the artifact says "
            f"{doc.get('opening_set_digest')}; the population has been edited")
    return openings


def run_segment(*, segment: int, results_path: str, trace_path: str,
                report_path: str) -> Dict[str, Any]:
    """THE PUBLIC ENTRY for ONE segment.

    🔴 EVERY OTHER INPUT IS RESOLVED HERE, so none can be supplied: an entry that
    accepted `tasks`, `play` or `identity` would let an opened gate authorize
    CALLER-SUPPLIED GAMEPLAY through the API while the CLI could not run the real
    thing. The injection seams live on the PRIVATE entry, for tests.
    """
    check_gate()
    _check_segment(segment)
    from . import d1_probe as D1
    openings = load_opening_set(OPENING_SET_PATH)
    tasks = RULES.build_tasks(openings, seed_interval=STUDY_SEED_BLOCK)
    deadline = D1.Deadline(RULES.SEGMENT_DEADLINE_S)
    deadline.start()                    # ONE origin, before anything effectful
    # ONE OBJECT, constructed here and handed to the seam; the run body reads the
    # identity off it before opening anything.
    argmax_cfg = frozen_argmax_config()
    return _run_segment_unguarded(
        segment=segment, tasks=segment_schedule(tasks, segment), openings=openings,
        results_path=results_path, trace_path=trace_path, report_path=report_path,
        play=_production_play(results_path, deadline, openings, argmax_cfg),
        want_digest=segment_digest(tasks, segment),
        deadline_s=RULES.SEGMENT_DEADLINE_S, _deadline=deadline)


def _run_segment_unguarded(*, segment: int, tasks, openings, results_path,
                           trace_path, report_path, play, want_digest,
                           deadline_s=None, identity=None, _deadline=None,
                           _supervisor=None) -> Dict[str, Any]:
    """Everything below the gate. PRIVATE, and never a way around `run_segment`.

    A DEADLINE STOPS CLEANLY and still reports — the segment's completed pairs
    stay valid evidence — with `timed_out` true. An EXCEPTION is a VOID, and
    §5.6 governs what a VOID does to the study's verdict.
    """
    import contextlib
    import json

    _check_segment(segment)
    check_seed_registration()
    check_output_paths(results_path, trace_path, report_path)
    summary = check_segment_schedule(tasks, segment, want_digest)

    # 🔑 THE SEAM'S OWN OBJECT, and BEFORE A SINGLE OUTPUT IS OPENED.
    config = _require_seam_config(play)
    ident = dict(identity) if identity is not None else frozen_incumbent_identity(config)
    check_incumbent_identity(ident, config)

    deadline_s = RULES.SEGMENT_DEADLINE_S if deadline_s is None else deadline_s
    from . import d1_probe as _D1
    deadline = _deadline if _deadline is not None else _D1.Deadline(deadline_s)
    if not deadline.started:
        deadline.start()

    games: list = []
    timed_out = False
    stack = contextlib.ExitStack()
    t_start = time.monotonic()
    try:
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

        emit(trace, {"event": "run_start", "segment": segment,
                     "n_tasks": len(tasks), "pairs": summary["pairs"],
                     "selection_mode": ident["eval_config"]["selection_mode"]})
        emit(rec, {"record_type": "header", "design": "H3_FULL_STUDY",
                   "segment": segment, "task_digest": summary["task_digest"],
                   "opening_set_digest": RULES.opening_set_digest(openings),
                   "selection_mode": ident["eval_config"]["selection_mode"],
                   "identity": ident})

        for i, task in enumerate(tasks):
            if deadline.elapsed() > deadline_s:
                timed_out = True
                emit(trace, {"event": "deadline", "index": i,
                             "games_completed": len(games)})
                break
            emit(trace, {"event": "task_start", "index": i})
            out = play(task=task, identity=ident,
                       timeout_s=RULES.PER_CALL_TIMEOUT_S)
            row = out["result"]
            t = ANALYSIS.transcript(out["plies"], row,
                                    opening_bound=out["opening_bound"])
            game = {"task_id": task["task_id"],
                    "seed": task["seed"],
                    "pair_id": task["pair_id"],
                    "segment": task["segment"],
                    "stratum": task["stratum"],
                    "incumbent_colour": task["incumbent_colour"],
                    "opening_digest": task["opening_digest"],
                    "transcript_digest": ANALYSIS.transcript_digest(t),
                    "terminal_reason": row.get("terminal_reason"),
                    "winner": row.get("winner"),
                    "plies": row.get("plies"),
                    "elapsed_s": out["elapsed_s"]}
            games.append(game)
            emit(rec, {"record_type": "transcript", "task_id": task["task_id"],
                       "n_plies": len(t), "opening_bound": out["opening_bound"]})
            emit(rec, {"record_type": "task_result", **game})
            emit(trace, {"event": "task_done", "index": i,
                         "games_completed": len(games)})

        report = ANALYSIS.summarise(games, total_elapsed_s=time.monotonic() - t_start)
        report["segment"] = segment
        report["timed_out"] = timed_out
        report["complete"] = (not timed_out) and len(games) == len(tasks)
        fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True, default=str)
        emit(trace, {"event": "run_end", "segment": segment,
                     "verdict": "PARTIAL" if timed_out else "OK",
                     "games_completed": len(games)})
        return report
    except H3StudyRunError:
        raise
    except Exception as e:                                     # noqa: BLE001
        raise H3StudyVoidError(f"{type(e).__name__}: {e}") from e
    finally:
        stack.close()


#: 🔴 WHERE THE PINNED OPENING SET WILL LIVE. It does not exist: producing it is a
#: separate authorized generation run.
OPENING_SET_PATH = (f"{OUT_ROOT}/2026-09-15-t1j-h3-study-openings/"
                    f"01_opening_set.json")
