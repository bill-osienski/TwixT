"""D1 position selection -- the FROZEN rule of plan 12.1-12.3. NO EXECUTION.

Nothing here loads a model, starts a JVM, queries T1j, draws or registers a
seed, or plays a game. It reads the published L0 record through D0's
digest-verified binding and recomputes deterministic board facts with our own
rules engine -- the same zero-inference footing D0 stands on.

WHY SELECTION LIVES OUTSIDE `d1_probe`. `d1_probe` is the gated execution
machinery; this is preparation, and preparation must be runnable and testable
without going anywhere near the execution gate. The same split the repository
already draws between `l0_match_plan` and `l0_match_command`.

THE RULE IS FROZEN, NOT INFERRED. Section 12.1 fixes the columns, the discovery
half, the incumbent-to-move restriction, the digest deduplication and the
per-cell cap of 3; 12.2 fixes the digest and the canonical prefix; 12.3 fixes
the matched controls. Nothing here chooses any of that. What this module DOES
choose -- because 12 does not state it -- is the order in which the 227 seeds
are assigned; that choice is named and defended at `SEED_ASSIGNMENT_ORDER`.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence, Tuple

from . import d0_postmortem as D0
from .fpu_state_hash import canonical_state_key

Pos = Tuple[int, int]


class D1SelectionError(Exception):
    """The frozen selection rule cannot be applied as written."""


def canonical_digest(state) -> str:
    """Plan 12.2's digest: sha256 over `to_move`, sorted pegs, sorted bridges.

    IMPLEMENTED LITERALLY, and deliberately NOT `fpu_state_hash`'s
    `canonical_state_sha1`. That helper is sha1 over a SUPERSET key
    (`board_size`, `active_size`, `max_plies_limit` as well), and across this
    single 24x24 cohort those three are constant -- so it would deduplicate
    identically while still not being the digest the preregistration froze.

    Its canonical SORTING is reused, because that is the part 12.2 and the
    helper agree on and a second sorting rule could drift. The three frozen
    fields are sliced out of the key; the payload is pinned by a test that
    rebuilds it from 12.2's wording, so a change to the key's shape fails
    loudly rather than silently hashing the wrong fields.

    THE DIGEST IS A DEDUPLICATION LABEL, NEVER REPLAY INPUT (12.2). The E3b
    adapter advances T1j only by replaying an ordered move sequence.
    """
    _board, _active, to_move, pegs, bridges, _limit = canonical_state_key(state)
    payload = json.dumps((to_move, pegs, bridges), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


#: The two gate-passing signatures, frozen in 12.1. No third may be added, and
#: no threshold is chosen: both columns are boolean, and `mover_more_fragmented`
#: is comparative precisely so D1 inherits no cutoff. The counts are the frozen
#: expectations; `select_all` recomputes them and refuses a mismatch rather than
#: trusting either the table or the code alone.
SIGNATURES = (
    {"name": "mover_fragmentation", "column": "mover_more_fragmented",
     "positions": 101, "controls": 60, "cells": 36},
    {"name": "created_threat", "column": "created_threat",
     "positions": 30, "controls": 36, "cells": 12},
)

#: 12.1: a cell is (opening x colour arm x phase), capped at 3.
PER_CELL_CAP = 3

#: 12.4: 227 positions, a hard ceiling fixed before any model or JVM load.
N_POSITIONS = 227

#: THE CANONICAL D1 SEED INTERVAL -- plan §14's reservation, 221 seeds, one per
#: §13 position. Defined HERE and nowhere else: `d1_probe` binds this object
#: rather than retyping the pair, because two literals of one fact drift the
#: moment either is edited and `==` cannot see the difference until they differ.
#:
#: RESERVED and UNSPENT. REGISTERED as ACCOUNTED on 2026-09-08 by the seed-
#: preparation authorization, so `d1_probe._check_seed_registration` is now
#: satisfied -- and NOT exposed, NOT retired, NOT consumed, because registering a
#: reservation is not drawing from it. Nothing in this module adds it to a
#: registry or draws from it; it assigns numbers on paper. The seed barrier is
#: down; `D1_EXECUTION_AUTHORIZED` is a separate gate and is still False.
#:
#: ⚠ This is NOT `RETIRED_SEED_INTERVAL` below, though the two were EQUAL until
#: the §14 handoff. They must never be edited together: a retirement guard
#: pointed at the new block would stop refusing the old one.
SEED_INTERVAL = (202615000, 202615221)

#: PLAN §13 AMENDMENT 2 -- the six already-retained rows the one authorized
#: low-ply qualification OBSERVED to fail the frozen completion condition.
#:
#: ENUMERATED, not the rule `ply >= 5`. The qualification supports excluding
#: these observed prefixes and no broader engine claim: nine prefixes in one
#: colour arm at three plies is not a threshold located. A rule would also have
#: excluded rows nobody tested.
#:
#: Applied ONLY AFTER §12.1-12.3 has run exactly as frozen. There is no
#: reselection, backfill, re-deduplication or re-capping -- replacing these six
#: would be a new selection rule written after the result was seen.
AMENDMENT2_EXCLUDED = (
    "487df111c9dbd7a3cd70c1ff0bd1316eee47cb950f7443fb15f5bff33927d2a7",  # ply 1
    "470721202fb36f18040bc00152ae6fbaf2ae576f8941ae2e6a6fcf99de676fde",  # ply 3
    "69c2875679f3eb1b128c42daccdc28122ee7ec2556092333f61fcf6e11d3a473",  # ply 1
    "4fba47bfca43d99f8b1c3fda801ec141d4013bdf7e135665fc4678a5047a002b",  # ply 3
    "7c326873cac4c1d7786dd2eb69b4ba4c4ba0c7631a24fe4855eee529beb0f6a4",  # ply 1
    "5e2c1f8ab19effb5487c8edba19c35a6075538ddcbb2110dd798364b010272e6",  # ply 3
)

#: §13.3's prospective totals. 227 - 6.
N_POSITIONS_AFTER_EXCLUSION = 227 - len(AMENDMENT2_EXCLUDED)          # 221

#: The revised per-cohort counts. All six removals fall in one cohort.
AMENDMENT2_COUNTS = {("mover_fragmentation", "position"): 101,
                     ("mover_fragmentation", "control"): 54,
                     ("created_threat", "position"): 30,
                     ("created_threat", "control"): 36}

#: The block D1 spent. RETIRED WHOLE after the 2026-08-28 VOID and never
#: reusable -- §13 reserves no replacement, so a run needs a fresh interval that
#: a separate authorization must choose and prove.
RETIRED_SEED_INTERVAL = (202614000, 202614227)

#: WHICH POSITION GETS WHICH SEED -- A CHOICE MADE HERE, NOT FROZEN IN 12.
#: Section 12.5 fixes the interval and "one per position" and stops there, so the
#: order is an execution decision and is recorded rather than left implicit. It
#: is the order 12 already uses everywhere else: the signature table's rows, each
#: signature's positions before its controls, and within a group the same
#: `(task_id, ply)` total order that drives selection and tie-breaking. No second
#: ordering rule exists that could disagree with the first.
SEED_ASSIGNMENT_ORDER = (("mover_fragmentation", "position"),
                         ("mover_fragmentation", "control"),
                         ("created_threat", "position"),
                         ("created_threat", "control"))


def cell(row: Dict[str, Any]) -> Tuple[str, str, str]:
    """12.1's cell: opening x colour arm x phase."""
    return (row["opening"], row["colour_arm"], row["phase"])


def discovery_plies(bound: Any) -> List[Dict[str, Any]]:
    """Every discovery ply, as D0 computes it, plus what D1 replay needs.

    Adds three keys and changes none: `digest` (12.2's label for the position
    FACED), `prefix` (the full ordered move sequence reaching it, which is the
    only thing the E3b adapter can consume) and `system` (D0's one definition of
    which engine moved).

    The confirmation half is unreachable from here: `D0.game_features` refuses
    it, and this walks `D0.discovery_task_ids`.
    """
    from .game.twixt_state import TwixtState

    out: List[Dict[str, Any]] = []
    for task_id in D0.discovery_task_ids(bound):
        rows = D0.game_features(bound, task_id)
        moves = D0.game_moves(bound, task_id)
        if len(rows) != len(moves):
            raise D1SelectionError(
                f"{task_id}: {len(rows)} feature rows for {len(moves)} moves")
        state = TwixtState()
        for i, (row, move) in enumerate(zip(rows, moves)):
            # The replay here and the one inside `game_features` must stay in
            # step; if they drift, the digest labels a position nobody faced.
            if row["ply"] != i or row["mover"] != state.to_move:
                raise D1SelectionError(
                    f"{task_id} ply {i}: feature row is ply {row['ply']} with "
                    f"{row['mover']} to move, replay has {state.to_move}")
            out.append({**row, "digest": canonical_digest(state),
                        "prefix": [tuple(m) for m in moves[:i]],
                        "system": D0.moved_by(row["colour_arm"], row["mover"])})
            state = state.apply_move(move)
    return out


def _dedup_and_cap(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """12.1 steps 2 and 3, in the ONE total order `(task_id, ply)`.

    Deduplication runs before the cap, so a duplicate never consumes a cell
    slot. Both keep the earliest, so a single pass over the sorted rows is the
    same result as two -- and there is no second tie-break rule to disagree
    with the first (12.2).
    """
    seen: set = set()
    per_cell: Dict[Tuple[str, str, str], int] = {}
    kept: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: (r["task_id"], r["ply"])):
        if row["digest"] in seen:
            continue
        seen.add(row["digest"])
        key = cell(row)
        if per_cell.get(key, 0) >= PER_CELL_CAP:
            continue
        per_cell[key] = per_cell.get(key, 0) + 1
        kept.append(row)
    return kept


def cohorts(plies: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The four groups: each signature's positions, then its matched controls.

    Selection reads D0 FEATURES ONLY. It consults no model output and no T1j
    answer -- neither exists when the rule is frozen, which is the point of
    freezing it in the plan rather than after the queries return.

    Controls are drawn from the cells the positions occupy, with the column
    False, under the identical dedup and cap (12.3). They are therefore matched
    on opening, colour arm and phase BY CONSTRUCTION, and capped by
    availability rather than padded.
    """
    out: List[Dict[str, Any]] = []
    for sig in SIGNATURES:
        col = sig["column"]
        ours = [r for r in plies if r["system"] == "ours"]
        positions = _dedup_and_cap([r for r in ours if bool(r[col]) is True])
        cells = {cell(r) for r in positions}
        controls = _dedup_and_cap(
            [r for r in ours if bool(r[col]) is False and cell(r) in cells])
        out.append({"signature": sig["name"], "role": "position", "column": col,
                    "rows": positions, "n_cells": len(cells)})
        out.append({"signature": sig["name"], "role": "control", "column": col,
                    "rows": controls, "n_cells": len({cell(r) for r in controls})})
    return out


#: MEASURED over the canonical record and PINNED by a test: the frozen rule
#: retains 227 positions covering 203 DISTINCT board states, because 12.1 and
#: 12.3 deduplicate within a cohort and nothing in 12 deduplicates across them.
#: 24 states are therefore retained twice -- once as one signature's position,
#: once as the other's control -- with two different seeds. That is what the
#: frozen counts contain: cross-cohort deduplication would have given 203, not
#: 227. Recorded here so the consequence is not discovered later: each of those
#: states is queried by four JVMs per depth rather than two, and 12.7's
#: determinism check compares only within a pair, never across the two pairs.
N_DISTINCT_STATES = 203
N_STATES_IN_TWO_COHORTS = 24


def _control_imbalance(frozen: Dict, kept: Dict) -> Dict[str, Any]:
    """§13.3's recorded consequence: the exclusion empties cells of controls.

    Backfilling would be reselection, so this is STATED rather than repaired --
    before any run, exactly as 12.3 stated its own 101-vs-60 imbalance "so it
    cannot later be mistaken for a filtered result". It widens a pre-existing
    gap; it does not create a new matching rule.
    """
    out: Dict[str, Any] = {}
    for sig in SIGNATURES:
        name = sig["name"]
        pos = {cell(r) for r in kept[(name, "position")]}
        cb = {cell(r) for r in frozen[(name, "control")]}
        ca = {cell(r) for r in kept[(name, "control")]}
        out[name] = {
            "position_cells": len(pos),
            "control_cells_before": len(cb), "control_cells_after": len(ca),
            "position_cells_without_control_before": len(pos - cb),
            "position_cells_without_control_after": len(pos - ca),
        }
    return out


def select_all(bound: Any,
               seed_interval: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """§12's frozen selection, THEN §13's enumerated exclusion.

    THE ORDER IS THE POINT. §12.1-12.3 runs untouched and its frozen counts are
    verified FIRST; only then are the six rows removed. Checking the counts after
    removal instead would let a drifted rule hide inside the exclusion.

    SEEDS ARE NOT ASSIGNED BY DEFAULT. §13 reserves no interval and the block D1
    spent is retired whole, so a prospective manifest carries `seed: None` until
    a separate authorization reserves and proves a fresh one. Supply
    `seed_interval` only when that has happened; it is size-checked against the
    cohort and refuses the retired block.
    """
    groups = cohorts(discovery_plies(bound))
    by_key = {(c["signature"], c["role"]): c for c in groups}
    if set(by_key) != set(SEED_ASSIGNMENT_ORDER):
        raise D1SelectionError(f"cohort keys {sorted(by_key)} are not the frozen four")

    for sig in SIGNATURES:
        for role, want in (("position", "positions"), ("control", "controls")):
            got = len(by_key[(sig["name"], role)]["rows"])
            if got != sig[want]:
                raise D1SelectionError(
                    f"{sig['name']} {role}s: the rule retained {got}, but 12.1 froze "
                    f"{sig[want]}. The budget is never raised or lowered to fit the "
                    f"data; the rule and the record must be reconciled instead.")

    # The four per-cohort checks above pin every count, so a separate check that
    # they sum to N_POSITIONS could not be reached alone -- a branch no test can
    # reach is a branch to delete. That SIGNATURES sums to N_POSITIONS is a
    # STATIC fact and is asserted as one, in the tests.
    total_frozen = sum(len(by_key[k]["rows"]) for k in SEED_ASSIGNMENT_ORDER)

    # ---- §12 has now been applied and verified EXACTLY as frozen. Only here
    #      does §13's enumerated exclusion run.
    frozen_rows = {k: list(by_key[k]["rows"]) for k in SEED_ASSIGNMENT_ORDER}
    excluded = set(AMENDMENT2_EXCLUDED)
    seen = {r["digest"] for rows in frozen_rows.values() for r in rows}
    missing = excluded - seen
    if missing:
        raise D1SelectionError(
            f"§13 names {len(missing)} rows the frozen selection did not retain: "
            f"{sorted(missing)}. An exclusion that removes nothing is not an exclusion.")
    for k in SEED_ASSIGNMENT_ORDER:
        by_key[k]["n_frozen"] = len(frozen_rows[k])
        by_key[k]["rows"] = [r for r in frozen_rows[k] if r["digest"] not in excluded]

    for (name, role), want in AMENDMENT2_COUNTS.items():
        got = len(by_key[(name, role)]["rows"])
        if got != want:
            raise D1SelectionError(
                f"§13 {name} {role}s: {got} after exclusion, but §13.3 froze {want}")
    total = sum(len(by_key[k]["rows"]) for k in SEED_ASSIGNMENT_ORDER)
    if total != N_POSITIONS_AFTER_EXCLUSION:
        raise D1SelectionError(
            f"{total} positions after exclusion, §13.3 froze {N_POSITIONS_AFTER_EXCLUSION}")

    if seed_interval is None:
        seeds: List[Optional[int]] = [None] * total
    else:
        lo, hi = seed_interval
        if tuple(seed_interval) == RETIRED_SEED_INTERVAL or (
                lo < RETIRED_SEED_INTERVAL[1] and hi > RETIRED_SEED_INTERVAL[0]):
            raise D1SelectionError(
                f"{tuple(seed_interval)} overlaps the retired block "
                f"{RETIRED_SEED_INTERVAL}, which the 2026-08-28 VOID consumed "
                f"administratively. It may not be revived or partially reused.")
        if hi - lo != total:
            raise D1SelectionError(
                f"the supplied interval holds {hi - lo} seeds for {total} positions")
        seeds = list(range(lo, hi))

    ordered: List[Dict[str, Any]] = []
    i = 0
    for key in SEED_ASSIGNMENT_ORDER:
        group = by_key[key]
        # A row can belong to one signature's positions AND the other's
        # controls, so the seed is written onto a COPY: assigning in place would
        # let the second group overwrite the first group's seed.
        group["rows"] = [dict(r, seed=seeds[i + n], signature=key[0], role=key[1])
                         for n, r in enumerate(group["rows"])]
        i += len(group["rows"])
        ordered.extend(group["rows"])

    return {"n_positions": total, "n_positions_frozen": total_frozen,
            "n_excluded": len(excluded),
            "excluded_digests": sorted(excluded),
            "seed_interval": None if seed_interval is None else list(seed_interval),
            "seed_assignment_order": [list(k) for k in SEED_ASSIGNMENT_ORDER],
            "control_imbalance": _control_imbalance(
                {k: frozen_rows[k] for k in SEED_ASSIGNMENT_ORDER},
                {k: by_key[k]["rows"] for k in SEED_ASSIGNMENT_ORDER}),
            "cohorts": [by_key[k] for k in SEED_ASSIGNMENT_ORDER],
            "positions": ordered}


#: Exactly what a D1 run needs on its input, and nothing else. The D0 feature
#: rows carry ~30 more columns; carrying them into the run record would invite a
#: later analysis to read a column the preregistration never named.
MANIFEST_FIELDS = ("task_id", "ply", "seed", "digest", "signature", "role",
                   "opening", "colour_arm", "phase")


def run_manifest(selection: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The run's INPUT: the probe's requirements plus 5.4's cohort labels.

    `prefix` is the only thing the E3b adapter can consume (12.2), `digest` is
    the label 12.7 re-checks it against, `seed` is this position's own, and
    `signature`/`role` are 5.4's "D0 structural signature and matched-control
    label". The two frozen signature columns travel too, so a reader can see
    which cohort a row belongs to without recomputing D0.

    JSON-ready: tuples become lists here rather than wherever the file is written.
    """
    columns = {sig["name"]: sig["column"] for sig in SIGNATURES}
    out: List[Dict[str, Any]] = []
    for row in selection["positions"]:
        entry = {field: row[field] for field in MANIFEST_FIELDS}
        entry["prefix"] = [[int(a), int(b)] for a, b in row["prefix"]]
        for column in columns.values():
            entry[column] = bool(row[column])
        out.append(entry)
    return out
