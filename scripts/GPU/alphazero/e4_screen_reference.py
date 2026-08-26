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
#: Half-open [start, end). `rng_witness` DRAWS from a generator, and under this
#: workstream's accounting drawing exposes the seed even with no model and no
#: game -- so a witness may never be taken from a seed inside these.
ACCOUNTED_SEED_INTERVALS = (
    (202608060, 202608124), (202608124, 202608188), (202608188, 202608988),
    (202608988, 202609388), (202609388, 202609788), (202609788, 202610188),
    (202611000, 202611400),          # twixtbot G3 declared block
    (202612000, 202612512),          # the E4 screen's reservation, see EXPOSED below
)

#: Seeds already burnt without a game. E4 preflight attempt 3 froze an rng witness
#: for all 32 scheduled tasks, which constructed both derived generators and drew
#: four values from each. No model ran and no game was played, but the seeds are
#: spent under this workstream's boundary and must never be scheduled.
EXPOSED_SEED_INTERVALS = (
    (202612000, 202612032),          # E4 preflight attempt 3 witnesses
    (90002000, 90002004),            # E4 integration qualification ATTEMPT 2,
                                     # 2026-08-26: the corrective run, same four
                                     # tasks, same drawing. Spent.
    (90001000, 90001004),            # E4 INTEGRATION qualification, 2026-08-25:
                                     # four synthetic tasks each built a real
                                     # SeededReferenceAgent and drew from both
                                     # generators. Spent, model or no model.
)


def seed_is_accounted(seed: int) -> bool:
    return any(lo <= int(seed) < hi for lo, hi in ACCOUNTED_SEED_INTERVALS)


def seed_is_exposed(seed: int) -> bool:
    return any(lo <= int(seed) < hi for lo, hi in EXPOSED_SEED_INTERVALS)


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
    """First `draws` values of each stream, for a SYNTHETIC seed only.

    This demonstrates the seed-to-generator derivation. It also SPENDS the seed:
    it constructs both real generators and draws from them, which under this
    workstream's accounting exposes that seed even though no model is loaded and
    no game is played. E4 preflight attempt 3 learned this the expensive way, by
    taking witnesses over its own 32 scheduled seeds and burning them.

    So this refuses any seed inside a reserved, consumed or exposed interval.
    Demonstrate the mechanism on a seed no schedule will ever use.
    """
    seed = int(task["seed"])
    if seed_is_accounted(seed) or seed_is_exposed(seed):
        raise E4ReferenceError(
            f"refusing to draw from scheduled seed {seed}: drawing spends it. "
            f"Use a synthetic seed outside ACCOUNTED_SEED_INTERVALS.")
    s = rng_stream_seeds(task)
    return {
        **s,
        "search_first": _first(s["search_seed"], draws),
        "readout_first": _first(s["readout_seed"], draws),
    }


def _first(seed: int, n: int) -> List[float]:
    r = random.Random(seed)
    return [r.random() for _ in range(n)]


def validate_task(task: Dict[str, Any]) -> None:
    """Everything `build_reference_agent` will check, checked BEFORE any model."""
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
    if int(task["seed"]) in CONSUMED_SEEDS:
        raise E4ReferenceError(f"seed {task['seed']} is recorded as already consumed")
    if seed_is_exposed(task["seed"]):
        raise E4ReferenceError(
            f"seed {task['seed']} was EXPOSED by an earlier witness and cannot be scheduled")
    reference_colour(task)          # raises on a bad anchor_colour


def validate_schedule(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate every task, and assert the seed->stream mapping is injective.

    Two tasks sharing a generator stream would silently correlate two games that
    the schedule presents as independent.
    """
    if not tasks:
        raise E4ReferenceError("empty schedule")
    for t in tasks:
        validate_task(t)
    seeds = [int(t["seed"]) for t in tasks]
    if len(set(seeds)) != len(seeds):
        raise E4ReferenceError("duplicate task seeds")
    # XOR only -- deriving the integer does NOT draw from a generator and does
    # not spend the seed. Never call rng_witness on a scheduled task.
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


def build(task: Dict[str, Any], *, evaluator, config=None) -> SeededReferenceAgent:
    """The ONE construction path for the E4 screen. Delegates, never reimplements."""
    validate_task(task)
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
