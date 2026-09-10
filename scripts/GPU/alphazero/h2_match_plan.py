"""H2 — the frozen 736-task schedule. BUILDS NOTHING AND DRAWS NOTHING.

Same shape as H1's plan module: the tasks are derived from the SAME pinned source
plan L0 and H1 use, so the openings cannot drift, and the only H2-specific inputs
are the repetition count and the seed block.

🔴 THE SEED BLOCK IS RESERVED ON PAPER AND REGISTERED NOWHERE. This module assigns
numbers; `h2_match_runner.check_seed_registration` refuses a run while they are
absent from `ACCOUNTED_SEED_INTERVALS`, and the gate refuses first.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

from . import e4_screen_reference as REF
from . import h2_match_rules as RULES
from . import l0_match_plan as L0PLAN


class H2PlanError(ValueError):
    """A refusal from H2's plan layer."""


#: The same source L0 and H1 use, pinned by the same sha256 their loaders verify.
SOURCE_PLAN_REL = L0PLAN.SOURCE_PLAN_REL

#: BOUND to the rules, never retyped: the module that states the seed abort rule
#: and the module that assigns the seeds must not be able to disagree.
H2_SEED_BLOCK = RULES.H2_SEED_BLOCK
COLOUR_ARMS = ("t1j_red", "t1j_black")

#: Where the built plan is written and read. The plan FILE does not exist until an
#: authorized preparation step writes it; nothing here creates it.
H2_PLAN_REL = ("docs/superpowers/evidence/2026-09-09-t1j-h2-deterministic-readout/"
               "01_h2_plan.json")


def load_source_plan(path: str = SOURCE_PLAN_REL) -> Dict[str, Any]:
    """The pinned source plan, through L0's own verifying loader."""
    return L0PLAN.load_source_plan(path)


def build_tasks(source_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The 736 tasks, deterministically ordered. Builds nothing and draws nothing.

    🔑 THE ONLY GAMEPLAY DIFFERENCE FROM H1 IS `selection_mode`. It is carried on
    every task so a schedule cannot be run under the readout it was not written
    for -- the runner binds it, and a task without it refuses.
    """
    openings = source_plan["openings"]
    names = list(openings.keys())                  # frozen order, not sorted
    if len(names) != RULES.N_OPENINGS:
        raise H2PlanError(f"source plan has {len(names)} openings, "
                          f"expected {RULES.N_OPENINGS}")
    ref = source_plan["reference"]
    lo, hi = H2_SEED_BLOCK
    if hi - lo != RULES.N_GAMES:
        raise H2PlanError(f"seed block holds {hi - lo} seeds, need {RULES.N_GAMES}")

    tasks: List[Dict[str, Any]] = []
    for opening in names:
        for arm in COLOUR_ARMS:
            for rep in range(RULES.N_REPS):
                i = len(tasks)
                t = {
                    "task_id": f"h2match-{i:03d}-strong{RULES.T1J_MDPLY}-"
                               f"{opening}-{arm}-r{rep}",
                    "endpoint": "strong",
                    "t1j_mdPly": RULES.T1J_MDPLY,
                    "t1j_mdFixedPly": True,
                    "opening": opening,
                    "colour_arm": arm,
                    "rep": rep,
                    "anchor_colour": "red" if arm == "t1j_red" else "black",
                    "reference": ref["name"],
                    "reference_sha1": ref["sha1"],
                    "reference_sha256": ref["sha256"],
                    "seed": lo + i,
                    "selection_mode": RULES.SELECTION_MODE,
                }
                t["reference_colour"] = REF.reference_colour(t)
                t["rng_streams"] = REF.rng_stream_seeds(t)     # XOR only
                tasks.append(t)
    if len(tasks) != RULES.N_GAMES:
        raise H2PlanError(f"built {len(tasks)} tasks, expected {RULES.N_GAMES}")
    return tasks


def validate_h2_schedule(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """STRUCTURE and DESIGN. Says nothing about whether the seeds may be RUN.

    Execution eligibility is `e4_screen_reference.validate_schedule_executable`,
    a separate required function, and registration is a third: a spent schedule
    must stay parseable, and only scheduling asks whether it may run.
    """
    if len(tasks) != RULES.N_GAMES:
        raise H2PlanError(f"{len(tasks)} tasks, expected exactly {RULES.N_GAMES}")
    ids = [t["task_id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise H2PlanError("duplicate task_id")

    REF.validate_schedule_structure(tasks)         # fields, pins, injective streams

    lo, hi = H2_SEED_BLOCK
    for i, t in enumerate(tasks):
        # THE RULES' PREDICATE, not a re-derivation of the bounds: the abort rule
        # stated in prose and the check that enforces it are one thing.
        if RULES.seed_is_outside_the_reserved_block(t["seed"]):
            raise H2PlanError(f"{t['task_id']} seed {t['seed']} outside [{lo}, {hi})")
        # 🔑 POSITIONAL, not merely inside: row i carries exactly lo + i. A block
        # every row draws from is not a block each row is assigned FROM, and one
        # seed shared by many rows would make their derived streams identical.
        if type(t["seed"]) is not int or t["seed"] != lo + i:
            raise H2PlanError(
                f"row {i} carries seed {t['seed']!r}; the frozen assignment gives "
                f"{lo + i}. Seeds are bound POSITIONALLY, so a count of seeds drawn "
                f"identifies WHICH seeds were drawn.")
        if t["t1j_mdPly"] != RULES.T1J_MDPLY:
            raise H2PlanError(f"{t['task_id']} is at mdPly {t['t1j_mdPly']}, "
                              f"not {RULES.T1J_MDPLY}")
        if t["t1j_mdFixedPly"] is not True:
            raise H2PlanError(f"{t['task_id']} does not fix the ply")
        if t["colour_arm"] not in COLOUR_ARMS:
            raise H2PlanError(f"{t['task_id']} has colour arm {t['colour_arm']!r}")
        # THE GAMEPLAY-RULE CHANGE, carried per task and checked per task.
        if t.get("selection_mode") != RULES.SELECTION_MODE:
            raise H2PlanError(
                f"{t['task_id']} carries selection_mode {t.get('selection_mode')!r}, "
                f"not {RULES.SELECTION_MODE!r}. H2 IS the readout change; a schedule "
                f"without it describes H1 with more games.")

    cells: Dict[Any, int] = {}
    for t in tasks:
        key = (t["opening"], t["colour_arm"])
        cells[key] = cells.get(key, 0) + 1
    if len(cells) != RULES.N_OPENINGS * RULES.N_ARMS:
        raise H2PlanError(f"{len(cells)} opening/colour cells, expected "
                          f"{RULES.N_OPENINGS * RULES.N_ARMS}")
    wrong = {k: v for k, v in cells.items() if v != RULES.N_REPS}
    if wrong:
        raise H2PlanError(f"cells without exactly {RULES.N_REPS} repetitions: "
                          f"{sorted(wrong.items())[:3]}")
    for (opening, arm) in cells:
        reps = sorted(t["rep"] for t in tasks
                      if t["opening"] == opening and t["colour_arm"] == arm)
        if reps != list(range(RULES.N_REPS)):
            raise H2PlanError(f"cell ({opening}, {arm}) has repetitions {reps}")

    return {"n_tasks": len(tasks), "cells": len(cells), "reps_per_cell": RULES.N_REPS,
            "seed_block": list(H2_SEED_BLOCK),
            "selection_mode": RULES.SELECTION_MODE,
            "task_digest": RULES.L0.l0_task_digest(tasks)}


def build_plan(source_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The whole plan object, validated. Writes nothing."""
    src = source_plan if source_plan is not None else load_source_plan()
    tasks = build_tasks(src)
    summary = validate_h2_schedule(tasks)
    return {"design": "H2", "source_plan_sha256": L0PLAN.SOURCE_PLAN_SHA256,
            "tasks": tasks, **summary}


def plan_sha256(plan: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
