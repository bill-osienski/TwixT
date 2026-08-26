"""The E4 endpoint screen's reference-agent construction. NO GAMES HERE.

Constructing an agent plays nothing and consumes no seed. This module exists so
that the E4 screen, if it is ever authorized, cannot invent its own reference
path: it delegates to `twixtbot_g3_reference.build_reference_agent`, the
construction already qualified for the twixtbot G3 calibration, and refuses any
task that does not carry the fields that path requires.

WHAT THE E4 PREFLIGHT ATTEMPT 2 GOT WRONG, AND THIS FIXES
---------------------------------------------------------
Attempt 2's schedule recorded a `seed` and stopped there. Nothing bound the seed
to the two RNG streams that actually decide our moves, and the tasks were missing
two fields `build_reference_agent` demands -- `reference_sha1` and
`anchor_colour` -- so every task would have been refused at construction time.

Attempt 2 also declared ``dirichlet_eps: 0.0`` as an override. THAT IS NOT HOW
NOISE IS DISABLED ON THIS PATH. `eval_runner.cfg_from` builds its `MCTSConfig`
without passing `dirichlet_eps` at all, so the field keeps its dataclass default
of 0.25; root noise is suppressed instead at the call site, by
``search_with_root(state, add_noise=False)`` in `SeededReferenceAgent.__call__`.
The declaration was cosmetic. It is replaced here by a statement of the real
mechanism, and by a check that reads the mechanism rather than the declaration.

TWO SEED REGISTRIES, AND TWO VALIDATION QUESTIONS
-------------------------------------------------
`EXPOSED_SEED_INTERVALS` records EXPERIMENTAL EXPOSURE: seeds drawn from OUTSIDE
the permanently unschedulable test namespace, so a seed a schedule could have used
has been struck off. `RETIRED_SEED_INTERVALS` records what MAY NOT BE USED -- a
rule about the future.
They are kept apart because merging them would overstate the record: the
canonical screen drew from 24 of its 32 seeds and skipped 8 undrawn, and calling
all 32 "exposed" claims a draw that never happened. Both refuse execution.

A THIRD list, `TEST_ONLY_SEED_INTERVALS`, is neither: it is a band reserved for
the seeds tests draw from, ineligible for any schedule by construction. Drawing
from it creates NO experimental exposure -- not because the draw is unreal, but
because no schedule may contain such a seed, so there is nothing to strike off.
Witnesses may be taken ONLY from it -- an ALLOWLIST, because the old denylist
silently drew from whatever nobody had thought to record, which is exactly what
happened to the ad hoc test seed 90000001.

Correspondingly there are TWO validation entry points, not one with a switch.
`validate_*_structure` asks whether a schedule is WELL FORMED and answers the
same way forever, so a completed run stays parseable, verifiable and analysable
as historical evidence. `validate_*_executable` asks whether it may RUN NOW, and
its answer changes the moment its seeds are spent. A `require_unspent=` keyword
would have made the second question defaultable, and a default that can be
switched off is exactly the gate-that-does-not-bind this workstream keeps
finding. Callers must name which question they are asking.

THE TWO STREAMS
---------------
`eval_runner.play_eval_game` derives two independent generators from one seed,
with colour-specific XOR masks, and the search generator must not be used for the
readout -- MCTS shares one `self.rng` across prior shuffle, PUCT tie-break and
readout, so drawing readout numbers from it would perturb every later search.
The masks are imported, never re-typed, so they cannot drift from that path.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple

from .twixtbot_g3_reference import SeededReferenceAgent, build_reference_agent, eval_config
from .twixtbot_g3_schedule import CONSUMED_SEEDS, REFERENCE_CHECKPOINTS

#: Every seed interval this workstream has reserved, consumed, or EXPOSED.
#: Half-open [start, end). These are seeds a schedule could use, so a draw from
#: one strikes it off even with no model and no game -- a witness may never be
#: taken from a seed inside these.
ACCOUNTED_SEED_INTERVALS = (
    (202608060, 202608124), (202608124, 202608188), (202608188, 202608988),
    (202608988, 202609388), (202609388, 202609788), (202609788, 202610188),
    (202611000, 202611400),          # twixtbot G3 declared block
    (202612000, 202612512),          # the E4 screen's reservation, see EXPOSED below
)

#: EXPERIMENTAL EXPOSURE: seeds drawn from OUTSIDE the permanently unschedulable
#: test namespace below. Half-open [start, end).
#:
#: Drawing alone is not what this records, and saying so would contradict the test
#: band: those are drawn from constantly and never appear here. What makes a draw
#: an exposure is that it happened to a seed A SCHEDULE COULD HAVE USED, which
#: strikes that seed off. A draw inside the test namespace strikes nothing off,
#: because nothing there was ever available to schedule.
EXPOSED_SEED_INTERVALS = (
    (202612128, 202612136),          # THE E4 CANONICAL SCREEN, strong endpoint,
    (202612144, 202612160),          # tasks 000-007, and the weak endpoint, tasks
                                     # 016-031: 24 tasks PLAYED on 2026-08-26 from
                                     # a8b3994. Their seeds drove real generators.
                                     # Tasks 008-015 (202612136..202612143) were
                                     # SKIPPED by the recorded early stop and were
                                     # never drawn from -- they are undrawn, and
                                     # are retired below rather than claimed here.
    (202612000, 202612032),          # E4 preflight attempt 3 witnesses
    (90002000, 90002004),            # E4 integration qualification ATTEMPT 2,
                                     # 2026-08-26: the corrective run, same four
                                     # tasks, same drawing. Spent.
    (90001000, 90001004),            # E4 INTEGRATION qualification, 2026-08-25:
                                     # four synthetic tasks each built a real
                                     # SeededReferenceAgent and drew from both
                                     # generators. Spent, model or no model.
    (90000001, 90000002),            # THE OLD `SYNTHETIC` TEST SEED, which was
                                     # SCHEDULABLE. Drawn from TWICE, both draws
                                     # preserved:
                                     #  1) 2026-08-25-t1j-e4-preflight-attempt4/
                                     #     06_endpoint_screen_plan.json, at
                                     #     seed_accounting.witness_demonstration
                                     #     -- an rng_witness frozen into the
                                     #     canonical plan, four values per stream.
                                     #  2) 2026-08-25-t1j-e4-harness-qualification/
                                     #     04_qualify.py.txt:26,50,70 binds it to
                                     #     the one real agent call, and
                                     #     02_qualification.txt:18-28 is that call
                                     #     RUNNING: completed, one move (14,13),
                                     #     "search RNG advanced" + "readout RNG
                                     #     advanced".
                                     # It was absent from this registry anyway, and
                                     # a test asserted it must stay usable -- the
                                     # exact reuse this registry exists to prevent.
                                     # Recorded 2026-08-26.
)

#: Seeds RESERVED PERMANENTLY FOR TESTS. Unit tests must draw from something, and
#: a draw from a schedulable seed strikes that seed off. Naming a fresh
#: "synthetic" seed each time is what put 90000001 above: it was schedulable, it
#: was drawn from repeatedly, and nobody recorded it.
#:
#: So the tests draw from a band that is INELIGIBLE FOR SCHEDULING BY
#: CONSTRUCTION. Draws here are just as real, and create NO experimental exposure,
#: because no schedule may contain one -- `validate_schedule_executable` refuses
#: them, and they lie outside the screen's reserved block, so the two barriers are
#: independent. They are therefore NOT listed as exposed: there is nothing to
#: strike off, not a draw to hide.
TEST_ONLY_SEED_INTERVALS = (
    (90009000, 90009100),
)

#: Seeds RETIRED ADMINISTRATIVELY. Not a claim that anything was drawn: a claim
#: that the block may not be used again. The canonical screen was a preregistered
#: ONE-SHOT schedule and it completed. Replaying its 8 undrawn seeds would select
#: exactly the tasks the early stop declined to play -- a result chosen after
#: seeing the first 24, which is selection bias however clean the RNG is. So the
#: WHOLE block retires together, drawn and undrawn alike.
RETIRED_SEED_INTERVALS = (
    (202612128, 202612160),          # the canonical screen's 32-seed block
)


def seed_is_accounted(seed: int) -> bool:
    return any(lo <= int(seed) < hi for lo, hi in ACCOUNTED_SEED_INTERVALS)


def seed_is_exposed(seed: int) -> bool:
    """Was this SCHEDULABLE seed drawn from, and so struck off? See the registry.

    Not simply "was it drawn from": test-namespace seeds are drawn from constantly
    and are never exposed, because they were never available to schedule.
    """
    return any(lo <= int(seed) < hi for lo, hi in EXPOSED_SEED_INTERVALS)


def seed_is_test_only(seed: int) -> bool:
    """Reserved for tests: drawable without limit, schedulable never.

    Drawing here is a real draw. It creates no exposure only because nothing in
    this band could ever have entered a schedule.
    """
    return any(lo <= int(seed) < hi for lo, hi in TEST_ONLY_SEED_INTERVALS)


def seed_is_retired(seed: int) -> bool:
    """Is this seed administratively withdrawn? A rule about the future."""
    return any(lo <= int(seed) < hi for lo, hi in RETIRED_SEED_INTERVALS)


def seed_is_unavailable(seed: int) -> bool:
    """Exposed OR retired: either way it may never be scheduled again."""
    return seed_is_exposed(seed) or seed_is_retired(seed)


def seed_status(seed: int) -> Dict[str, bool]:
    """The two registries reported SEPARATELY, never merged into one word."""
    return {"exposed": seed_is_exposed(seed), "retired": seed_is_retired(seed),
            "accounted": seed_is_accounted(seed), "test_only": seed_is_test_only(seed)}


#: Exactly the fields `build_reference_agent` reads off a task.
REQUIRED_TASK_FIELDS = ("seed", "reference", "reference_sha1", "anchor_colour")

#: How root noise is actually suppressed on this path. Not a config field.
NOISE_SUPPRESSION = "search_with_root(state, add_noise=False)"


class E4ReferenceError(Exception):
    """An E4 task cannot produce the qualified reference construction."""


def reference_colour(task: Dict[str, Any]) -> str:
    """The colour OUR side plays: the opposite of the anchor's."""
    anchor = task.get("anchor_colour")
    if anchor not in ("red", "black"):
        raise E4ReferenceError(f"anchor_colour must be red or black, got {anchor!r}")
    return "black" if anchor == "red" else "red"


def rng_stream_seeds(task: Dict[str, Any]) -> Dict[str, int]:
    """The two generator seeds this task will actually use.

    Derived with the SAME masks the qualified path uses, imported from it. This
    is what binds a scheduled seed to the moves our side plays.
    """
    colour = reference_colour(task)
    seed = int(task["seed"])
    return {
        "colour": colour,
        "search_seed": seed ^ SeededReferenceAgent.SEARCH_MASK[colour],
        "readout_seed": seed ^ SeededReferenceAgent.READOUT_MASK[colour],
    }


def rng_witness(task: Dict[str, Any], draws: int = 4) -> Dict[str, Any]:
    """First `draws` values of each stream, for a TEST-ONLY seed.

    This is a REAL DRAW: it constructs both generators and takes values from them.
    It creates no experimental exposure all the same, and not because the draw is
    somehow lesser -- only because `TEST_ONLY_SEED_INTERVALS` can never enter a
    schedule, so a draw there strikes nothing off. On a schedulable seed the same
    call would strike that seed off permanently, with no model and no game. E4
    preflight attempt 3 learned that the expensive way, by taking witnesses over
    its own 32 scheduled seeds.

    THIS IS AN ALLOWLIST, NOT A DENYLIST, AND THAT IS THE POINT. The old version
    refused seeds it had been told about -- accounted, exposed, retired -- so any
    schedulable seed nobody had thought to list was drawn from and struck off in
    silence. That is exactly how 90000001 was witnessed into the frozen plan, used
    for the first real agent call, and still left absent from the registry. Now a
    witness may be taken only from the band reserved for exactly this, and an
    unrecognised seed is REFUSED rather than quietly consumed.
    """
    seed = int(task["seed"])
    if not seed_is_test_only(seed):
        raise E4ReferenceError(
            f"refusing to draw from seed {seed}: a schedule could use it, so drawing "
            f"would strike it off. Witnesses may be taken ONLY from "
            f"TEST_ONLY_SEED_INTERVALS ({list(TEST_ONLY_SEED_INTERVALS)}), which no "
            f"schedule may contain. Do not name a fresh 'synthetic' seed -- that is how "
            f"90000001 was drawn from twice and never recorded.")
    s = rng_stream_seeds(task)
    return {
        **s,
        "search_first": _first(s["search_seed"], draws),
        "readout_first": _first(s["readout_seed"], draws),
    }


def _first(seed: int, n: int) -> List[float]:
    r = random.Random(seed)
    return [r.random() for _ in range(n)]


def validate_task_structure(task: Dict[str, Any]) -> None:
    """Is this task WELL FORMED? Fields, pinned reference identity, colour.

    STRUCTURE ONLY. It asks nothing about seed availability, so it answers the
    same way forever: a schedule that was valid when it ran is still valid to
    parse, verify, classify and analyse after its seeds are gone. This is what
    lets a completed screen stay readable as historical evidence.
    """
    missing = [f for f in REQUIRED_TASK_FIELDS if f not in task]
    if missing:
        raise E4ReferenceError(f"task is missing {missing}; build_reference_agent would refuse it")
    ref = task["reference"]
    if ref not in REFERENCE_CHECKPOINTS:
        raise E4ReferenceError(f"unknown reference {ref!r}")
    pinned = REFERENCE_CHECKPOINTS[ref]["sha1"]
    if task["reference_sha1"] != pinned:
        raise E4ReferenceError(
            f"task reference_sha1 {task['reference_sha1']} != pinned {pinned}")
    reference_colour(task)          # raises on a bad anchor_colour


def validate_task_executable(task: Dict[str, Any]) -> None:
    """May this task be PLAYED NOW? Structure, then seed availability.

    A SEPARATE REQUIRED FUNCTION, deliberately not a `require_unspent=True`
    keyword on the structural one. A switch defaults, and a default that can be
    switched off is the failure mode this workstream keeps finding: every caller
    here must name which question it is asking, and the name is the answer.
    """
    validate_task_structure(task)
    seed = int(task["seed"])
    if seed in CONSUMED_SEEDS:
        raise E4ReferenceError(f"seed {seed} is recorded as already consumed")
    if seed_is_exposed(seed):
        raise E4ReferenceError(
            f"seed {seed} was EXPOSED -- it has been drawn from -- and cannot be scheduled")
    if seed_is_retired(seed):
        raise E4ReferenceError(
            f"seed {seed} belongs to a RETIRED block: its one-shot schedule completed, so "
            f"reusing any part of it would select tasks after seeing the result")


def _injective_streams(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The seed->stream mapping must be injective, and structural.

    Two tasks sharing a generator stream would silently correlate two games that
    the schedule presents as independent. Nothing here draws: XOR only.
    """
    seeds = [int(t["seed"]) for t in tasks]
    if len(set(seeds)) != len(seeds):
        raise E4ReferenceError("duplicate task seeds")
    streams = [(rng_stream_seeds(t)["search_seed"], rng_stream_seeds(t)["readout_seed"])
               for t in tasks]
    if len(set(streams)) != len(streams):
        raise E4ReferenceError("two tasks derive the same generator streams")
    search_only = {s for s, _ in streams}
    readout_only = {r for _, r in streams}
    if search_only & readout_only:
        raise E4ReferenceError("a search stream collides with a readout stream")
    return {"n_tasks": len(tasks), "distinct_seeds": len(set(seeds)),
            "distinct_stream_pairs": len(set(streams)),
            "search_readout_disjoint": True}


def validate_schedule_structure(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Is this schedule WELL FORMED? Independent of what has since been spent."""
    if not tasks:
        raise E4ReferenceError("empty schedule")
    for t in tasks:
        validate_task_structure(t)
    return _injective_streams(tasks)


def validate_schedule_executable(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """May this schedule be RUN NOW? Structure, availability, and no test seeds.

    The test band is refused HERE rather than per task: building an agent on a
    test seed is fine and unit tests do it constantly, but a SCHEDULE is a set of
    games whose seeds must never have been drawn from -- and test seeds are drawn
    from by design. Scheduling is the operation that must refuse them.
    """
    if not tasks:
        raise E4ReferenceError("empty schedule")
    for t in tasks:
        validate_task_executable(t)
    for t in tasks:
        if seed_is_test_only(int(t["seed"])):
            raise E4ReferenceError(
                f"seed {t['seed']} is reserved for tests and is drawn from freely; it may "
                f"never appear in a schedule")
    return _injective_streams(tasks)


def build(task: Dict[str, Any], *, evaluator, config=None) -> SeededReferenceAgent:
    """The ONE construction path for the E4 screen. Delegates, never reimplements.

    AGENT CONSTRUCTION IS EXECUTION. It is the moment a seed becomes generators,
    so it asks the executable question, never the structural one.
    """
    validate_task_executable(task)
    return build_reference_agent(
        task=task, evaluator=evaluator, colour=reference_colour(task), config=config
    )


def frozen_settings() -> Dict[str, Any]:
    """The frozen research configuration, read from the qualified path itself."""
    cfg = eval_config()
    return {
        "eval_config": {f: getattr(cfg, f) for f in
                        ("board_size", "mcts_sims", "mcts_eval_batch_size",
                         "mcts_stall_flush_sims", "selection_mode",
                         "opening_temp_plies", "temp_high", "temp_low", "max_moves")},
        "noise_suppression": NOISE_SUPPRESSION,
        "noise_note": "cfg_from does NOT pass dirichlet_eps, so MCTSConfig keeps its "
                      "default 0.25; noise is suppressed at the call site instead. A "
                      "schedule that declares dirichlet_eps=0 is declaring nothing.",
        "search_mask": dict(SeededReferenceAgent.SEARCH_MASK),
        "readout_mask": dict(SeededReferenceAgent.READOUT_MASK),
        "readout_path": "eval_readout.select, never mcts.select_move",
        "agent_lifetime": "one instance per game; both streams advance across the game",
    }
