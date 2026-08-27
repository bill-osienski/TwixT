"""The L0 larger match's schedule: builder, digest and validation. NO EXECUTION.

L0 builds 64 tasks and freezes them. It loads no model, starts no jvm, constructs
no generator and draws from no seed. Deriving a task's stream integers is XOR
arithmetic on the already-qualified masks; it builds nothing.

WHERE THE PARAMETERS COME FROM
-------------------------------
Every parameter L0 shares with the E4 screen is READ FROM THE SCREEN'S FROZEN
PLAN, whose sha256 is pinned in `e4_screen_runner`, rather than retyped here.
That is the whole point: "preserve the qualified artifact, adapter, binder, ply
cap, scoring and recording paths unchanged" is a claim that can be checked only if
the values have a single source. Retyping eight openings would create a second
source that looks identical until it isn't.

THE DESIGN
----------
8 frozen openings x 2 colour arms x 4 independent repetitions = 64 games, all at
mdPly 6. The repetitions differ ONLY in the seed of our reference agent, so within
a cell they sample the reference agent's own sampling -- opening temperature over
the first 20 plies, and MCTS tie-breaks.

  CAREFUL, AND NOT CLAIMED: that does NOT make the repetitions a clean estimate of
  reference-agent variance alone. It would if T1j were deterministic, and E3a
  established determinism for ONE position at ONE ply, noting that `Zobrist` seeds
  itself from an unseeded `Random` per process. Cross-process agreement was
  observed there and explicitly not proven. So T1j's contribution to within-cell
  variation is UNKNOWN, not zero, and the plan says so.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence

from . import e4_screen_reference as REF
from . import e4_screen_runner as H
from . import l0_match_rules as RULES

#: The frozen E4 screen plan is L0's parameter source, pinned by the same sha256
#: the screen's own loader verifies.
SOURCE_PLAN_REL = ("docs/superpowers/evidence/2026-08-25-t1j-e4-preflight-attempt4/"
                   "06_endpoint_screen_plan.json")
SOURCE_PLAN_SHA256 = H.CANONICAL_PLAN_SHA256

#: Reserved 2026-08-26, UNSPENT, proved disjoint from every seed category and from
#: every derived RNG stream before reservation. Registered in
#: e4_screen_reference.ACCOUNTED_SEED_INTERVALS.
L0_SEED_BLOCK = (202613000, 202613064)

#: The frozen L0 plan, pinned once written. A loader must verify BOTH: the file's
#: sha256, and the ordered dimension-projected digest of the tasks inside it.
L0_PLAN_REL = ("docs/superpowers/evidence/2026-08-26-t1j-l0-larger-match/"
               "01_l0_match_plan.json")
L0_PLAN_SHA256 = "c8b9cba816852a6752bc9a8ae7f74fe06529dcb5af5a06ac4bb6bda716cf8a30"


COLOUR_ARMS = ("t1j_red", "t1j_black")

#: The frozen schedule identity is defined in the RULES layer and re-exported
#: here, so `bind_results` can verify the very digest this module pins. See
#: l0_match_rules.L0_TASK_DIMENSIONS for why it lives there.
L0_TASK_DIMENSIONS = RULES.L0_TASK_DIMENSIONS
L0_TASK_DIGEST = RULES.L0_TASK_DIGEST
l0_task_digest = RULES.l0_task_digest


class L0PlanError(Exception):
    """The L0 schedule is not the frozen one, or is not well formed."""


def load_source_plan(path: str = SOURCE_PLAN_REL) -> Dict[str, Any]:
    """The screen's frozen plan, sha256-verified before a single value is read."""
    try:
        raw = open(path, "rb").read()
    except OSError as e:
        raise L0PlanError(f"cannot read the source plan: {e}") from None
    got = hashlib.sha256(raw).hexdigest()
    if got != SOURCE_PLAN_SHA256:
        raise L0PlanError(f"source plan sha256 {got} != pinned {SOURCE_PLAN_SHA256}")
    return json.loads(raw)


def build_tasks(source_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The 64 tasks, deterministically ordered. Builds nothing and draws nothing."""
    openings = source_plan["openings"]
    names = list(openings.keys())
    if len(names) != RULES.N_OPENINGS:
        raise L0PlanError(f"source plan has {len(names)} openings, expected {RULES.N_OPENINGS}")
    ref = source_plan["reference"]
    lo, hi = L0_SEED_BLOCK
    if hi - lo != RULES.N_GAMES:
        raise L0PlanError(f"seed block holds {hi - lo} seeds, need {RULES.N_GAMES}")

    tasks: List[Dict[str, Any]] = []
    for opening in names:                          # frozen order, not sorted
        for arm in COLOUR_ARMS:
            for rep in range(RULES.N_REPS):
                i = len(tasks)
                anchor = "red" if arm == "t1j_red" else "black"
                t = {
                    "task_id": f"l0match-{i:03d}-strong{RULES.T1J_MDPLY}-{opening}-{arm}-r{rep}",
                    "endpoint": "strong",
                    "t1j_mdPly": RULES.T1J_MDPLY,
                    "t1j_mdFixedPly": True,
                    "opening": opening,
                    "colour_arm": arm,
                    "rep": rep,
                    "anchor_colour": anchor,
                    "reference": ref["name"],
                    "reference_sha1": ref["sha1"],
                    "reference_sha256": ref["sha256"],
                    "seed": lo + i,
                }
                t["reference_colour"] = REF.reference_colour(t)
                t["rng_streams"] = REF.rng_stream_seeds(t)     # XOR only
                tasks.append(t)
    if len(tasks) != RULES.N_GAMES:
        raise L0PlanError(f"built {len(tasks)} tasks, expected {RULES.N_GAMES}")
    return tasks


def validate_l0_schedule(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """STRUCTURE and DESIGN. Says nothing about whether the seeds may be run.

    Execution eligibility is `e4_screen_reference.validate_schedule_executable`,
    which is a separate required function for the reason the seed reconciliation
    settled: a spent schedule must stay parseable, and only scheduling asks
    whether it may run.
    """
    if len(tasks) != RULES.N_GAMES:
        raise L0PlanError(f"{len(tasks)} tasks, expected exactly {RULES.N_GAMES}")
    ids = [t["task_id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise L0PlanError("duplicate task_id")

    REF.validate_schedule_structure(tasks)         # fields, pins, injective streams

    lo, hi = L0_SEED_BLOCK
    for t in tasks:
        if not lo <= int(t["seed"]) < hi:
            raise L0PlanError(f"{t['task_id']} seed {t['seed']} outside [{lo}, {hi})")
        if t["t1j_mdPly"] != RULES.T1J_MDPLY:
            raise L0PlanError(f"{t['task_id']} is at mdPly {t['t1j_mdPly']}, not {RULES.T1J_MDPLY}")
        if t["t1j_mdFixedPly"] is not True:
            raise L0PlanError(f"{t['task_id']} does not fix the ply: mdFixedPly must be True")
        if t["colour_arm"] not in COLOUR_ARMS:
            raise L0PlanError(f"{t['task_id']} has colour arm {t['colour_arm']!r}")

    cells: Dict[Any, int] = {}
    for t in tasks:
        cells[(t["opening"], t["colour_arm"])] = cells.get((t["opening"], t["colour_arm"]), 0) + 1
    if len(cells) != RULES.N_OPENINGS * RULES.N_ARMS:
        raise L0PlanError(f"{len(cells)} opening/colour cells, expected "
                          f"{RULES.N_OPENINGS * RULES.N_ARMS}")
    wrong = {k: v for k, v in cells.items() if v != RULES.N_REPS}
    if wrong:
        raise L0PlanError(f"cells without exactly {RULES.N_REPS} repetitions: "
                          f"{sorted(wrong.items())[:3]}")
    for (opening, arm), _ in cells.items():
        reps = sorted(t["rep"] for t in tasks
                      if t["opening"] == opening and t["colour_arm"] == arm)
        if reps != list(range(RULES.N_REPS)):
            raise L0PlanError(f"cell ({opening}, {arm}) has repetitions {reps}")

    return {"n_tasks": len(tasks), "cells": len(cells), "reps_per_cell": RULES.N_REPS,
            "seed_block": list(L0_SEED_BLOCK), "task_digest": l0_task_digest(tasks)}


def load_l0_plan(path: str = L0_PLAN_REL) -> Dict[str, Any]:
    """The frozen L0 plan. STRUCTURAL, like the screen's loader.

    Verifies the file's sha256 AND the ordered task digest, then the design. It
    asks nothing about seed availability: that is
    `e4_screen_reference.validate_schedule_executable`, and keeping the two apart
    is what lets this plan stay readable after the match has been run.
    """
    try:
        raw = open(path, "rb").read()
    except OSError as e:
        raise L0PlanError(f"cannot read the L0 plan: {e}") from None
    got = hashlib.sha256(raw).hexdigest()
    if got != L0_PLAN_SHA256:
        raise L0PlanError(f"L0 plan sha256 {got} != pinned {L0_PLAN_SHA256}")
    plan = json.loads(raw)
    tasks = plan.get("tasks", [])
    digest = l0_task_digest(tasks)
    if digest != L0_TASK_DIGEST:
        raise L0PlanError(f"L0 task digest {digest} != pinned {L0_TASK_DIGEST}: the "
                          f"schedule has been added to, removed from, reordered or edited")
    validate_l0_schedule(tasks)
    return plan
