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

#: 12.5: RESERVED, UNSPENT, and deliberately NOT REGISTERED. Nothing in this
#: module adds it to a registry or draws from it; it assigns numbers on paper.
SEED_INTERVAL = (202614000, 202614227)

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


def select_all(bound: Any) -> Dict[str, Any]:
    """The frozen selection, with one reserved seed per retained position.

    Refuses on any departure from the frozen counts. A selection that silently
    came out at a different size would move the query budget with it, and 12.4
    fixes that ceiling before any model or JVM load.
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

    total = sum(len(by_key[k]["rows"]) for k in SEED_ASSIGNMENT_ORDER)
    if total != N_POSITIONS:
        raise D1SelectionError(f"{total} positions retained, 12.4 froze {N_POSITIONS}")

    lo, hi = SEED_INTERVAL
    if hi - lo != N_POSITIONS:
        raise D1SelectionError(f"the reserved block holds {hi - lo} seeds for {N_POSITIONS}")
    seed = lo
    ordered: List[Dict[str, Any]] = []
    for key in SEED_ASSIGNMENT_ORDER:
        group = by_key[key]
        # A row can belong to one signature's positions AND the other's
        # controls, so the seed is written onto a COPY: assigning in place would
        # let the second group overwrite the first group's seed.
        group["rows"] = [dict(r, seed=seed + i) for i, r in enumerate(group["rows"])]
        seed += len(group["rows"])
        ordered.extend(group["rows"])

    return {"n_positions": total, "seed_interval": list(SEED_INTERVAL),
            "seed_assignment_order": [list(k) for k in SEED_ASSIGNMENT_ORDER],
            "cohorts": [by_key[k] for k in SEED_ASSIGNMENT_ORDER],
            "positions": ordered}
