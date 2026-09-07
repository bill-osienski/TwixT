"""The H1 viability screen's schedule: builder, digest and validation. NO EXECUTION.

H1 builds 224 tasks and freezes them. It loads no model, starts no jvm, constructs
no generator and draws from no seed. Deriving a task's stream integers is XOR
arithmetic on the already-qualified masks; it builds nothing.

WHERE THE PARAMETERS COME FROM
------------------------------
The openings and the reference identity are READ FROM THE E4 SCREEN'S FROZEN
PLAN, sha256-verified before a single value is read -- the same single source L0
reads, not a second hop through L0's derived artifact. Retyping eight openings
would create a source that looks identical until it isn't.

⚠ CORRECTED CLAIM. An earlier version of this docstring said
`validate_h1_schedule` requires the openings and reference to equal L0's. IT DOES
NOT, and never did -- that comparison lives in the test suite
(`test_the_openings_come_from_the_pinned_plan_and_match_what_L0_PLAYED`). A
docstring asserting a check the function does not perform is worse than no
docstring: it is the shape of a guard with nothing behind it.

WHAT ACTUALLY BINDS H1 TO L0 is upstream of validation and stronger for being
structural: `SOURCE_PLAN_REL` and `SOURCE_PLAN_SHA256` are TAKEN FROM L0's own
plan module and the screen's pinned constant, not retyped here, so H1 and L0
cannot read different files. Both then build from the same opening order and the
same `reference` object. The test confirms the built result, which is a check on
the construction rather than the substitute for one.

THE DESIGN
----------
8 frozen openings x 2 colour arms x 14 repetitions = 224 games, all at mdPly 6.
The repetitions differ ONLY in the seed of our reference agent.

  CAREFUL, AND NOT CLAIMED, exactly as L0 recorded: that does NOT make the
  repetitions a clean estimate of reference-agent variance alone. It would if T1j
  were deterministic, and E3a established determinism for ONE position at ONE ply
  while noting that `Zobrist` seeds itself from an unseeded `Random` per process.
  T1j's contribution to within-cell variation is UNKNOWN, not zero.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

from . import e4_screen_reference as REF
from . import e4_screen_runner as H
from . import h1_viability_rules as RULES
from . import l0_match_plan as L0PLAN

#: The same source L0 uses, pinned by the same sha256 the screen's loader verifies.
SOURCE_PLAN_REL = L0PLAN.SOURCE_PLAN_REL
SOURCE_PLAN_SHA256 = H.CANONICAL_PLAN_SHA256

#: THE CURRENT ATTEMPT'S BLOCK, defined in the rules layer (the seed abort rule
#: must be runnable, and this module imports the rules, so the rules cannot
#: import it back). ATTEMPT 2 (2026-09-07): [202617000, 202617224),
#: PAPER-RESERVED and DELIBERATELY UNREGISTERED -- see `h1_viability_rules`
#: for the proof and the gap policy. Attempt 1's spent block lives under
#: `RULES.H1_ATTEMPT1_SEED_BLOCK` with its own plan pins below.
#:
#: ⚠ THIS COMMENT HAS BEEN WRONG TWICE AND IS KEPT HONEST BY DATE. It first
#: claimed a barrier that did not exist; then it recorded attempt 1's block as
#: REGISTERED (true 2026-09-04) and that statement outlived the block, which
#: was drawn from and RETIRED on 2026-09-05. WHAT IS TRUE NOW (2026-09-07): the
#: runner holds BOTH barriers -- `H1_EXECUTION_AUTHORIZED` False and
#: `check_seed_registration`, which the attempt-2 block does NOT satisfy -- so
#: BOTH ARE UP. Registering the block is part of the EXECUTION authorization.
#: `validate_task_executable` still does not ask the accounted question, which
#: is why the registration barrier is a separate check.
H1_SEED_BLOCK = RULES.H1_SEED_BLOCK          # defined in the rules layer:
                                            # the seed abort rule must be runnable

#: The frozen H1 plan, pinned once written. A loader verifies BOTH the file's
#: sha256 AND the ordered dimension-projected digest of the tasks inside it --
#: the file hash alone would accept a correctly-hashed file whose tasks had been
#: rebuilt from a different design.
#: v3 after review. `01_h1_plan.json` and `06_h1_plan_v2.json` are SUPERSEDED and
#: deliberately PRESERVED rather than rewritten -- evidence is create-only, and
#: each earlier artifact is the record of what that version actually froze.
#: v2 corrected the abort rules, the denominators and cap saturation; v3 corrects
#: the NON-ABORT rules, which still described L0's protocol rather than H1's.
#: ATTEMPT 1 (v3): the schedule that RAN on 2026-09-05 and VOIDed at game 60.
#: PRESERVED, pinned under its own names, loadable as a record with
#: `load_h1_plan(H1_ATTEMPT1_PLAN_REL, sha256=..., task_digest=...)` and refused
#: for execution by the registry (every seed retired).
H1_ATTEMPT1_PLAN_REL = ("docs/superpowers/evidence/2026-08-31-t1j-h1-implementation/"
                        "10_h1_plan_v3.json")
H1_ATTEMPT1_PLAN_SHA256 = "57565530c961d250f02934de4ea3cd6b5c11398df0a3fcc85b2fe836346b0a14"

#: ATTEMPT 2 (v4, 2026-09-07): the SAME design, settings, threshold, bands,
#: rules, estimand and forbidden claims -- only the seed block (hence each task's
#: seed and derived streams) and the provenance fields differ; a test asserts
#: exactly that against v3. Built by a recorded script into the retry-prep
#: evidence directory; v1-v3 are SUPERSEDED and PRESERVED.
H1_PLAN_REL = ("docs/superpowers/evidence/2026-09-07-t1j-h1-retry-prep/"
               "06_h1_plan_v4.json")
H1_PLAN_SHA256 = "21cd2465d94d6c6a2e4834373c57f0e8e91a4e09d19f05c7c0db2bdf7ec9bc16"

COLOUR_ARMS = L0PLAN.COLOUR_ARMS


class H1PlanError(Exception):
    """The H1 schedule is not the frozen one, or is not well formed."""


def load_source_plan(path: str = SOURCE_PLAN_REL) -> Dict[str, Any]:
    """The screen's frozen plan, sha256-verified before a single value is read."""
    try:
        raw = open(path, "rb").read()
    except OSError as e:
        raise H1PlanError(f"cannot read the source plan: {e}") from None
    got = hashlib.sha256(raw).hexdigest()
    if got != SOURCE_PLAN_SHA256:
        raise H1PlanError(f"source plan sha256 {got} != pinned {SOURCE_PLAN_SHA256}")
    return json.loads(raw)


def build_tasks(source_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The 224 tasks, deterministically ordered. Builds nothing and draws nothing."""
    openings = source_plan["openings"]
    names = list(openings.keys())                  # frozen order, not sorted
    if len(names) != RULES.N_OPENINGS:
        raise H1PlanError(f"source plan has {len(names)} openings, "
                          f"expected {RULES.N_OPENINGS}")
    ref = source_plan["reference"]
    lo, hi = H1_SEED_BLOCK
    if hi - lo != RULES.N_GAMES:
        raise H1PlanError(f"seed block holds {hi - lo} seeds, need {RULES.N_GAMES}")

    tasks: List[Dict[str, Any]] = []
    for opening in names:
        for arm in COLOUR_ARMS:
            for rep in range(RULES.N_REPS):
                i = len(tasks)
                t = {
                    "task_id": f"h1match-{i:03d}-strong{RULES.T1J_MDPLY}-"
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
                }
                t["reference_colour"] = REF.reference_colour(t)
                t["rng_streams"] = REF.rng_stream_seeds(t)     # XOR only
                tasks.append(t)
    if len(tasks) != RULES.N_GAMES:
        raise H1PlanError(f"built {len(tasks)} tasks, expected {RULES.N_GAMES}")
    return tasks


def validate_h1_schedule(tasks: Sequence[Dict[str, Any]], *,
                         seed_block: Optional[tuple] = None) -> Dict[str, Any]:
    """STRUCTURE and DESIGN. Says nothing about whether the seeds may be run.

    Execution eligibility is `e4_screen_reference.validate_schedule_executable`,
    a separate required function: a spent schedule must stay parseable, and only
    scheduling asks whether it may run.

    `seed_block` defaults to the CURRENT attempt's block (the rules' predicate);
    a spent attempt's plan names its own so it can be validated AS A RECORD.
    """
    block = H1_SEED_BLOCK if seed_block is None else tuple(seed_block)
    if len(tasks) != RULES.N_GAMES:
        raise H1PlanError(f"{len(tasks)} tasks, expected exactly {RULES.N_GAMES}")
    ids = [t["task_id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise H1PlanError("duplicate task_id")

    REF.validate_schedule_structure(tasks)         # fields, pins, injective streams

    lo, hi = block
    for t in tasks:
        # THE RULES' PREDICATE for the current block, not a re-derivation of the
        # bounds: the abort rule that states this in prose and the check that
        # enforces it are one thing. A foreign (spent) block is checked by bounds.
        outside = (RULES.seed_is_outside_the_reserved_block(t["seed"])
                   if block == tuple(H1_SEED_BLOCK) else not lo <= int(t["seed"]) < hi)
        if outside:
            raise H1PlanError(f"{t['task_id']} seed {t['seed']} outside [{lo}, {hi})")
        if t["t1j_mdPly"] != RULES.T1J_MDPLY:
            raise H1PlanError(f"{t['task_id']} is at mdPly {t['t1j_mdPly']}, "
                              f"not {RULES.T1J_MDPLY}")
        if t["t1j_mdFixedPly"] is not True:
            raise H1PlanError(f"{t['task_id']} does not fix the ply: mdFixedPly must be True")
        if t["colour_arm"] not in COLOUR_ARMS:
            raise H1PlanError(f"{t['task_id']} has colour arm {t['colour_arm']!r}")

    cells: Dict[Any, int] = {}
    for t in tasks:
        key = (t["opening"], t["colour_arm"])
        cells[key] = cells.get(key, 0) + 1
    if len(cells) != RULES.N_OPENINGS * RULES.N_ARMS:
        raise H1PlanError(f"{len(cells)} opening/colour cells, expected "
                          f"{RULES.N_OPENINGS * RULES.N_ARMS}")
    wrong = {k: v for k, v in cells.items() if v != RULES.N_REPS}
    if wrong:
        raise H1PlanError(f"cells without exactly {RULES.N_REPS} repetitions: "
                          f"{sorted(wrong.items())[:3]}")
    # 🔑 RECIPROCITY NEEDS NO SEPARATE GUARD, and the one first written here could
    # never fire. 16 cells each holding exactly 14 ALREADY forces every opening to
    # appear 14 times in both arms: put an opening entirely in one arm and its
    # other cell disappears, so the 16-cell check above fires first (verified by
    # construction, not assumed). A guard that cannot fail is not protection; it
    # is a claim that something is checked when the check is elsewhere.
    #
    # What is NOT implied by the counts is that the repetition LABELS are the
    # frozen 0..N_REPS-1 in every cell, so that is checked, over BOTH arms.
    for (opening, arm) in cells:
        reps = sorted(t["rep"] for t in tasks
                      if t["opening"] == opening and t["colour_arm"] == arm)
        if reps != list(range(RULES.N_REPS)):
            raise H1PlanError(f"cell ({opening}, {arm}) has repetitions {reps}")

    return {"n_tasks": len(tasks), "cells": len(cells), "reps_per_cell": RULES.N_REPS,
            "seed_block": list(block),
            "task_digest": RULES.L0.l0_task_digest(tasks)}


def load_h1_plan(path: str = H1_PLAN_REL, *, sha256: Optional[str] = None,
                 task_digest: Optional[str] = None) -> Dict[str, Any]:
    """The frozen H1 plan. STRUCTURAL, like L0's loader.

    Verifies the file's sha256 AND the ordered task digest, then the design. It
    asks nothing about seed availability: that is
    `e4_screen_reference.validate_schedule_executable`, and keeping the two
    apart is what lets this plan stay readable after a match has been run.

    `sha256` / `task_digest` default to the CURRENT attempt's pins. A SPENT
    attempt's plan is loaded as a record by passing its own pins explicitly --
    both, because a caller who relocates the file must still name what it is.
    The design check below is the same for every attempt EXCEPT the seed-block
    membership, which is attempt-specific and is skipped only when both foreign
    pins are supplied (the registry, not this loader, refuses a spent block).
    """
    sha256 = H1_PLAN_SHA256 if sha256 is None else sha256
    task_digest = RULES.H1_TASK_DIGEST if task_digest is None else task_digest
    try:
        raw = open(path, "rb").read()
    except OSError as e:
        raise H1PlanError(f"cannot read the H1 plan: {e}") from None
    got = hashlib.sha256(raw).hexdigest()
    if got != sha256:
        raise H1PlanError(f"H1 plan sha256 {got} != pinned {sha256}")
    plan = json.loads(raw)
    tasks = plan.get("tasks", [])
    digest = RULES.L0.l0_task_digest(tasks)
    if digest != task_digest:
        raise H1PlanError(f"H1 task digest {digest} != pinned {task_digest}: "
                          f"the schedule has been added to, removed from, reordered "
                          f"or edited")
    current = sha256 == H1_PLAN_SHA256 and task_digest == RULES.H1_TASK_DIGEST
    validate_h1_schedule(tasks, seed_block=(H1_SEED_BLOCK if current else
                                            tuple(plan["seed_block"])))
    return plan
