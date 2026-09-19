"""H3 FULL STUDY — the segment runner. GATE-SHUT AND SEEDLESS.

THREE BARRIERS, and none of them is the other's fault:
  1. `H3_STUDY_EXECUTION_AUTHORIZED` is False. One reviewed edit opens it.
  2. The seeds are checked PER SEGMENT. `check_segment_seeds(k)` refuses if
     segment k's block is unregistered, overlaps a RETIRED block, or is already
     exposed/retired/test-only. A gate can be opened by one edit; an unspent
     registered block cannot be conjured by one.
  3. The production seam re-checks the gate and hits a containment boundary that
     refuses inside a test process.

The population is UNIFORM and ENGINE-FREE (Amendment 3), produced by
`h3_study_generator`, which has no execution gate because it runs no engine.
Playing the study IS a run and keeps this module's gate; freezing the population
has its own separate barrier. One switch for all three would let a population
approval authorize a match.

⚠ EXERCISED END TO END ONCE: segment 0 ran 2026-09-18 (148/148, exit 0) on
block [202628000, 202628148), which is now spent and refuses relaunch. Segments
1-3 have never run; for them only construction, wiring and refusals are tested.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from . import h3_study_analysis as ANALYSIS
from . import h3_study_generator as GEN
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
#: 🔴 FOUR SEGMENT BLOCKS, ONE PER SEGMENT. NOT one interval.
#:
#: The study had a single 592-seed block whose four quarters all had to be
#: unspent to construct ANY segment. Segment 0's quarter was retired on its VOID
#: of 2026-09-18 and the whole study became unbuildable -- segments 1-3 included,
#: though their seeds were untouched. **Segmenting exists so that one segment's
#: failure costs one segment**, and that coupling took the property away.
#:
#: segment 0  [202628000, 202628148)  FRESH, registered 2026-09-18 (proof v15)
#:            replacing [202626000, 202626148), RETIRED WHOLE on the VOID.
#:            No strength information was produced by that VOID -- 0 games, 0
#:            seeds exposed -- so replacing these seeds introduces no
#:            outcome-based selection.
#: segment 1  [202626148, 202626296)  unchanged, unspent
#: segment 2  [202626296, 202626444)  unchanged, unspent
#: segment 3  [202626444, 202626592)  unchanged, unspent
#:
#: 🔑 ACCOUNTED IS NOT EXPOSED AND NOT RETIRED: a reservation is not a draw.
SEGMENT_SEED_BLOCKS: tuple = (
    (202_628_000, 202_628_148),
    (202_626_148, 202_626_296),
    (202_626_296, 202_626_444),
    (202_626_444, 202_626_592),
)

#: 🔴 SEGMENT 0's RETIRED QUARTER, kept by name so nothing can quietly re-use it
#: and so the preflight can say WHY a retired block is refused.
RETIRED_SEGMENT_BLOCKS: tuple = ((202_626_000, 202_626_148),)

#: 🔴 THE SEEDED SCHEDULE'S PINS, RECOMPUTED AND FROZEN 2026-09-18.
#:
#: These fix the 592 tasks -- openings, colours, seeds, configuration, every
#: field -- over the population pinned as `OPENING_SET_DIGEST`. A schedule that
#: does not reproduce them is a different experiment wearing this one's name.
#:
#: 🔴 AND WITHOUT THEM THE SEGMENT CHECK COULD NOT FAIL. `run_segment` passed
#: `want_digest=segment_digest(tasks, segment)` -- the digest computed from the
#: very tasks it then handed to `check_segment_schedule` to verify. The two
#: sides came from one source, so they agreed unconditionally: a check that
#: existed and did not bind, which is this programme's recurring shape. The pin
#: is what gives the comparison a second, INDEPENDENT side.
SCHEDULE_DIGEST = (
    "158cc080da26d5bb39ae3a9dd3422234b842b962e7d65b40d2eb322fcdc8f272")

#: One per segment, in order. Each segment is its own one-shot schedule.
SEGMENT_DIGESTS = (
    # 🔴 SEGMENT 0's PIN MOVED, and ONLY segment 0's. Its block was replaced
    # after the VOID; segments 1-3 play the same openings, colours and seeds they
    # always did, so their digests are BYTE-IDENTICAL to the pre-repair ones and
    # a test asserts it. If any of the three had moved, the repair would have
    # reached further than segment 0 and the study's other segments would be
    # different experiments.
    "b82bbb08e8bff4df22284100ac6b3777d4a16ed75d77525d5c72f5c707091b23",  # 0, NEW
    "14aaefb3a01cc1d08bdfd7f0c4c21599414d0dc68fee8f51584e1e4a50660a41",  # 1, same
    "1764471c5b95042726b9a3bbeca92309ff703d734d73f448adf2f548d96a1855",  # 2, same
    "44576a71042ff6b7c9118d3e36f18d62d1d0dfc4d01aeb7eec318d0a4b8753f0",  # 3, same
)

#: the pins as they stood BEFORE segment 0's block was replaced, kept so the
#: "1-3 did not move" assertion has something to compare against.
SEGMENT_DIGESTS_BEFORE_SEG0_REPLACEMENT = (
    "cde6d04ce52e07088e437754e1ee60b750542de08c6b6e9fbc4744bc2c795d31",
    "14aaefb3a01cc1d08bdfd7f0c4c21599414d0dc68fee8f51584e1e4a50660a41",
    "1764471c5b95042726b9a3bbeca92309ff703d734d73f448adf2f548d96a1855",
    "44576a71042ff6b7c9118d3e36f18d62d1d0dfc4d01aeb7eec318d0a4b8753f0",
)


def check_schedule_digest(tasks: Sequence[Mapping[str, Any]]) -> str:
    """The whole 592-task schedule must BE the pinned one."""
    got = RULES.task_digest(tasks)
    if got != SCHEDULE_DIGEST:
        raise H3StudyRunError(
            f"the full schedule digest is {got} but the frozen schedule is "
            f"{SCHEDULE_DIGEST}. A different schedule is a different experiment "
            f"wearing this one's name.")
    return got


def check_seed_registration(segment: Optional[int] = None) -> None:
    """Refuse unless the seeds THIS LAUNCH needs are accounted and unspent.

    🔴 SEGMENT-LOCAL, AND THAT IS THE REPAIR. It used to demand the whole 592-seed
    interval be clean, so a retired segment blocked every other segment. Now it
    answers about ONE segment -- the one being launched -- and says nothing about
    the others.

    `segment=None` checks that every segment's block is REGISTERED, which is a
    planning question and never a launch one: it deliberately does NOT ask
    whether a block is spent, because a spent earlier segment must not block a
    later launch.
    """
    from . import e4_screen_reference as REF
    if not SEGMENT_SEED_BLOCKS or len(SEGMENT_SEED_BLOCKS) != RULES.N_SEGMENTS:
        raise H3StudyRunError(
            f"NO SEED BLOCKS ARE RESERVED for the H3 full study. Each of the "
            f"{RULES.N_SEGMENTS} segments needs its OWN {RULES.GAMES_PER_SEGMENT}"
            f"-seed block, with a collision re-proof covering the GENERATION "
            f"ranges too.")
    todo = range(RULES.N_SEGMENTS) if segment is None else [segment]
    for k in todo:
        _check_segment(k)
        lo, hi = SEGMENT_SEED_BLOCKS[k]
        if hi - lo != RULES.GAMES_PER_SEGMENT:
            raise H3StudyRunError(
                f"segment {k}'s block holds {hi - lo} seeds for "
                f"{RULES.GAMES_PER_SEGMENT} games")
        missing = [x for x in range(lo, hi) if not REF.seed_is_accounted(x)]
        if missing:
            raise H3StudyRunError(
                f"segment {k}'s block [{lo}, {hi}) is not registered: "
                f"{len(missing)} of {hi - lo} seeds are absent from "
                f"ACCOUNTED_SEED_INTERVALS (first {missing[0]}).")


def check_segment_seeds(segment: int) -> Dict[str, Any]:
    """🔴 THE LAUNCH CHECK, for ONE segment and no other.

    Accounted, and not exposed, retired or test-only. This is where "a spent
    block may not be revived" now lives: `build_tasks` no longer consults status,
    because construction is not a launch and a frozen schedule must exist whether
    or not a segment has since been spent.

    A RETIRED EARLIER SEGMENT MUST NOT REACH THIS FOR A LATER ONE -- and cannot,
    because it only ever looks at `SEGMENT_SEED_BLOCKS[segment]`.
    """
    from . import e4_screen_reference as REF
    _check_segment(segment)
    check_seed_registration(segment)
    lo, hi = SEGMENT_SEED_BLOCKS[segment]
    for retired_lo, retired_hi in RETIRED_SEGMENT_BLOCKS:
        if lo < retired_hi and retired_lo < hi:
            raise H3StudyRunError(
                f"segment {segment}'s block [{lo}, {hi}) overlaps the RETIRED "
                f"block [{retired_lo}, {retired_hi}). That block was spent on a "
                f"one-shot launch and may never be revived; a retry needs a "
                f"FRESH block with its own collision re-proof.")
    spent = [(x, REF.seed_status(x)) for x in range(lo, hi)]
    bad = [(x, st) for x, st in spent
           if st["exposed"] or st["retired"] or st["test_only"]]
    if bad:
        x, st = bad[0]
        raise H3StudyRunError(
            f"segment {segment}'s block [{lo}, {hi}) is SPENT: {len(bad)} of "
            f"{hi - lo} seeds are exposed, retired or test-only (first {x}: "
            f"exposed={st['exposed']} RETIRED={st['retired']} "
            f"test_only={st['test_only']}). A spent block may not be revived.")
    return {"segment": segment, "block": (lo, hi), "n": hi - lo}


# ═══════════════════ RESTORED: THE INCUMBENT IDENTITY BARRIER ══════════════
# 🔴 THESE FOUR WERE DELETED BY ACCIDENT and the preflight caught it.
# Rewriting the span between `check_seed_registration` and `segment_out_dir`
# swallowed the whole identity barrier along with `OUT_ROOT`. NOTHING FAILED AT
# IMPORT -- they are only referenced inside function bodies -- and the H3 test
# suite would have needed to reach them to notice. The preflight did, on its
# very next run, because it exercises the real builder.
#
# They are restored VERBATIM from the commit, not retyped: this barrier is what
# makes the recorded identity describe the object the builder actually received,
# and a paraphrase of it would be a different guard wearing its name.

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


#: 🔴 RESTORED. This constant sat between `check_seed_registration` and
#: `segment_out_dir` and was swallowed when that span was rewritten. Nothing
#: failed at import -- `OUT_ROOT` is only read inside an f-string at call time --
#: so it took actually CALLING `segment_out_dir` to find it. A module that
#: imports cleanly is not a module that works.
OUT_ROOT = "docs/superpowers/evidence"


def segment_out_dir(segment: int) -> str:
    """Each segment writes into ITS OWN directory, so no segment can overwrite
    another's record or an earlier experiment's."""
    _check_segment(segment)
    #: 🔴 SEGMENT 0's FIRST DESTINATION IS SPENT. The VOID of 2026-09-18
    #: consumed `2026-09-15-t1j-h3-study-segment0` -- consumed by ATTEMPT, not by
    #: success: the run died before creating it, but the authorization was spent
    #: and a retry must not write where a spent attempt was aimed. Its retry has
    #: its own place; segments 1-3 keep theirs.
    if segment == 0:
        return f"{OUT_ROOT}/2026-09-18-t1j-h3-study-segment0-retry"
    return f"{OUT_ROOT}/2026-09-15-t1j-h3-study-segment{segment}"


def ensure_parent_dirs(*paths: str) -> list:
    """Create each path's parent directory. Returns the directories it ensured.

    🔴 THE DEFECT THAT VOIDED SEGMENT 0. The run body opened its trace with
    `O_EXCL` and never created the parent, so the whole segment died in one
    second on `FileNotFoundError` -- spending an authorization and retiring a
    seed quarter for nothing. `h3_study_generator.write_artifact` had learned
    this; the segment runner had not.

    🔑 `exist_ok=True` DOES NOT WEAKEN CREATE-ONLY. The DIRECTORY may already
    exist; the FILES may not, and the `O_EXCL` opens are what say so.

    🔑 A NAMED FUNCTION, not four lines inline, so a test can exercise it
    directly. Inline, the only way to reach it was to drive the whole run body
    past its validation -- so the first test grepped the source instead, and the
    harness showed that disabling the guard left the text in place and the grep
    passing. A check that greps source is not a test.
    """
    made = []
    for p in paths:
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)
            made.append(d)
    return made


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
    """The 148 tasks (74 pairs) of one segment, in plan order."""
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
    """The FROZEN population, VALIDATED AND BOUND, from the pinned artifact.

    🔴 THIS USED TO TRUST THE FILE. It called `check_opening_set` and re-hashed
    the openings' DECLARED digests -- so an opening's `moves` could be rewritten
    while its `digest` and `opening_set_digest` stayed untouched, and the runner
    would accept and PLAY a different position than the one the study is defined
    over. A digest that is never recomputed from the thing it digests is a label.
    It also never called `validate_artifact` at all, so the artifact's schema,
    claim and generator identity were unchecked on the path that matters most.

    It now delegates to `GEN.validate_artifact`, which replays every opening's
    six moves through the real engine, recomputes each canonical digest, checks
    `seed == attempt_seed(base, index, attempts - 1)`, and re-derives the whole
    candidate from the PRNG. Nothing here re-implements those checks: a loader
    with its own copy of the rules checks the copy.

    🔑 AND IT STILL WILL NOT REGENERATE. Uniform generation is deterministic, so
    the set could be recomputed at start-up -- and that is exactly why it is not.
    A study must play the population that was FROZEN, not one rebuilt today that
    happens to match.
    """
    import json
    from . import h3_study_generator as GEN
    if not os.path.lexists(path):
        raise H3StudyRunError(
            f"the frozen opening set {path} does not exist. The population is "
            f"produced by a separate authorized POPULATION-FREEZE step "
            f"(`h3_freeze_command`); until that has run there is no study "
            f"population, and a recomputed set may never stand in for one.")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise H3StudyRunError(f"{path} does not hold an artifact object")
    try:
        bound = GEN.validate_artifact(doc)
    except GEN.H3GenerationError as e:
        raise H3StudyRunError(
            f"the frozen opening set {path} FAILED validation: {e}") from None
    openings = doc["openings"]
    RULES.check_opening_set(openings)      # the study's own rules, over the same set

    # 🔴 AND IT MUST BE THE PINNED POPULATION, not merely a self-consistent one.
    # `validate_artifact` proves the artifact is internally sound and re-derivable;
    # only OPENING_SET_DIGEST says it is THE one the study was defined over.
    pinned = RULES.expected_opening_set_digest()
    if bound["opening_set_digest"] != pinned:
        raise H3StudyRunError(
            f"{path} holds population {bound['opening_set_digest']}, but the "
            f"study is pinned to {pinned}. A different population is a different "
            f"study.")
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
    #: 🔴 SEGMENT-LOCAL. The seeds THIS launch needs must be accounted and
    #: unspent; the other three segments' status is not consulted, so a retired
    #: earlier segment cannot block a later one.
    check_segment_seeds(segment)
    from . import d1_probe as D1
    openings = load_opening_set(OPENING_SET_PATH)
    tasks = RULES.build_tasks(openings, seed_blocks=SEGMENT_SEED_BLOCKS)
    #: 🔴 AGAINST THE PIN, NOT AGAINST ITSELF. The digest handed to
    #: `check_segment_schedule` used to be computed from `tasks` two lines above
    #: it, so the comparison had one source and could not fail.
    check_schedule_digest(tasks)
    deadline = D1.Deadline(RULES.SEGMENT_DEADLINE_S)
    deadline.start()                    # ONE origin, before anything effectful
    # ONE OBJECT, constructed here and handed to the seam; the run body reads the
    # identity off it before opening anything.
    argmax_cfg = frozen_argmax_config()
    return _run_segment_unguarded(
        segment=segment, tasks=segment_schedule(tasks, segment), openings=openings,
        results_path=results_path, trace_path=trace_path, report_path=report_path,
        play=_production_play(results_path, deadline, openings, argmax_cfg),
        want_digest=SEGMENT_DIGESTS[segment],
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
    #: 🔴 THE DIRECTORY, BEFORE THE FIRST CREATE-ONLY WRITE. Without this the
    #: whole segment VOIDs in one second on `FileNotFoundError` at `os.open` --
    #: which is exactly what happened on 2026-09-18, spending an authorization
    #: and retiring a seed quarter for nothing. `h3_study_generator.write_artifact`
    #: had learned this; the segment runner had not.
    #:
    #: 🔑 `exist_ok=True` DOES NOT WEAKEN CREATE-ONLY. The directory may exist;
    #: the FILES may not, and `O_EXCL` below is what says so.
    ensure_parent_dirs(trace_path, results_path, report_path)
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


#: 🔴 WHERE THE FROZEN POPULATION WILL LIVE. It does not exist: freezing it is a
#: separate authorized step.
#:
#: 🔴 READ FROM THE WRITER, NEVER RETYPED. This was a literal naming attempt 1's
#: directory. It was correct when typed and became silently wrong the moment
#: Amendment 3 moved the destination -- pointing the runner at a SPENT directory
#: while every test still passed, because nothing compared the two.
OPENING_SET_PATH = GEN.DEFAULT_OUT
