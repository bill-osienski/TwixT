"""Frozen G3 openings and the complete 128-task schedule.

Pilot card: docs/superpowers/2026-08-22-twixtbot-anchor-pilot-card.md.
PREPARATION ONLY. Nothing here plays a game or consumes a seed; it declares what
G3 would run. Pure: no filesystem, no engine, no model, no randomness, so the
schedule cannot be reshaped by anything on disk.

THE EIGHT OPENINGS are fixed move sequences replayed identically in both engines
before either agent moves. The SAME sequence is used for both colour arms; the
arms differ only in which agent is assigned which colour afterwards, so an
opening can never advantage one arm.

Openings alternate red, black, red, black from ply 0 (our TwixtState and
twixtbot both start with red/WHITE to move) and respect both engines' edge bars:
red may not sit on col 0 or 23, black may not sit on row 0 or 23.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Sequence, Tuple

BOARD_SIZE = 24
PLY_CAP = 280

TRIALS_LADDER: Tuple[int, ...] = (0, 100, 400, 1000)
REFERENCES: Tuple[str, ...] = ("0379", "calib020_0001")
COLOUR_ARMS: Tuple[str, ...] = ("anchor_red", "anchor_black")

#: SEEDS CONSUMED OUTSIDE A G3 GAME. These may never be used for a scheduled task.
#: 202611000 was spent on 2026-08-23 by the attempt-2 preflight, which played one
#: REAL 400-simulation reference move (both search and readout RNG streams) using
#: task 0's scheduled seed. The move was a smoke, not a game, but the seed is
#: burnt either way, and the attempt-2 evidence claim that "all 400 reserved seeds
#: are untouched" is FALSE. That package is preserved unaltered; this is the
#: correction. See docs/superpowers/evidence/2026-08-23-twixtbot-g3-preflight-attempt2/.
CONSUMED_SEEDS = (202611000,)

#: The schedule was re-frozen onto a block disjoint from every consumed seed.
SEED_BASE = 202611128
RESERVED_SEEDS = (202611000, 202611400)      # half-open, the reserved interval
SCHEDULE_SEEDS = (SEED_BASE, SEED_BASE + 128)   # [202611128, 202611256)

REFERENCE_CHECKPOINTS: Dict[str, Dict[str, str]] = {
    "0379": {
        "path": "checkpoints/alphazero-v2-staged/model_iter_0379.safetensors",
        "sha1": "8ad62ac432c35c6ea9b0630b8a2b8c572a0b03a1",
    },
    "calib020_0001": {
        "path": "checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors",
        "sha1": "209cf2d4fd24a48553d259dd71b4954867b9473e",
    },
}

ANCHOR = {
    "repo": "github.com/stevens68/twixtbot-ui",
    "commit": "83749f230a0bae1766b46a05bfde0ed87f0a9a0a",
    "model_dir": "model/pb",
    "weights_sha256": "1958f8476e9d56cbb87fa570db88f9cc9d389b30e20571fd66e8e779a4cefbab",
    "saved_model_sha256": "e0c2b882bc97c4661ac92af47ae5ef78443e57b42d0cb7c9c3a8cb7e8d01993b",
    "variables_index_sha256": "1d4073c4da30e515a984faef2c3cc9c8b8ba402353a01933c8bb80e1f279f8cd",
}

# Frozen search settings. Not parameters.
ANCHOR_SETTINGS = {
    "temperature": 0, "add_noise": 0, "rotation": "off",
    "allow_swap": 0, "allow_scl": False,
}
OUR_SETTINGS = {
    "mcts_sims": 400, "mcts_eval_batch_size": 14, "mcts_stall_flush_sims": 48,
    "selection_mode": "opening_temperature", "opening_temp_plies": 20,
    "temp_high": 1.0, "temp_low": 0.1, "max_moves": PLY_CAP, "workers": 1,
}

#: Eight openings, four plies each: red, black, red, black.
#: Chosen to spread the first contact across the board rather than to be strong.
OPENINGS: Tuple[Tuple[str, Tuple[Tuple[int, int], ...]], ...] = (
    ("O1_centre",        ((12, 12), (11, 10), (13, 13), (10, 11))),
    ("O2_centre_mirror", ((11, 11), (12, 13), (10, 12), (13, 10))),
    ("O3_high",          ((6, 12), (7, 10), (8, 13), (5, 11))),
    ("O4_low",           ((17, 11), (16, 13), (15, 10), (18, 12))),
    ("O5_left",          ((12, 6), (11, 8), (13, 5), (10, 7))),
    ("O6_right",         ((12, 17), (13, 15), (11, 18), (14, 16))),
    ("O7_diag",          ((8, 8), (9, 10), (10, 10), (7, 9))),
    ("O8_wide",          ((5, 5), (18, 18), (6, 7), (17, 16))),
)

OPENING_PLIES = 4
EXPECTED_TASKS = len(TRIALS_LADDER) * len(REFERENCES) * len(OPENINGS) * len(COLOUR_ARMS)


class ScheduleError(Exception):
    """A frozen invariant does not hold. Never swallowed."""


def opening_hash(moves: Sequence[Tuple[int, int]]) -> str:
    """Stable hash of a move sequence: order matters, formatting does not."""
    payload = json.dumps([[int(r), int(c)] for r, c in moves], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def seed_for(trial_index: int, reference_index: int, opening_index: int, colour_arm: int) -> int:
    """The frozen seed formula. Every task gets exactly one seed, and no two share.

    SEED_BASE + (((t*2 + r)*8 + o)*2 + c), which enumerates a 128-wide block
    exactly once. SEED_BASE moved from 202611000 to 202611128 after 202611000 was
    burnt by a preflight smoke; see CONSUMED_SEEDS.
    """
    for name, value, limit in (
        ("trial_index", trial_index, len(TRIALS_LADDER)),
        ("reference_index", reference_index, len(REFERENCES)),
        ("opening_index", opening_index, len(OPENINGS)),
        ("colour_arm", colour_arm, len(COLOUR_ARMS)),
    ):
        if not (0 <= value < limit):
            raise ScheduleError(f"{name}={value} outside [0,{limit})")
    return SEED_BASE + (((trial_index * 2 + reference_index) * 8 + opening_index) * 2 + colour_arm)


def enumerate_tasks() -> List[dict]:
    """All 128 task identities, in a fixed order. Pure."""
    tasks: List[dict] = []
    for t_i, trials in enumerate(TRIALS_LADDER):
        for r_i, reference in enumerate(REFERENCES):
            for o_i, (opening_id, moves) in enumerate(OPENINGS):
                for c_i, colour_arm in enumerate(COLOUR_ARMS):
                    tasks.append({
                        "task_index": len(tasks),
                        "trials": trials,
                        "reference": reference,
                        "reference_checkpoint": REFERENCE_CHECKPOINTS[reference]["path"],
                        "reference_sha1": REFERENCE_CHECKPOINTS[reference]["sha1"],
                        "opening_id": opening_id,
                        "opening_moves": [list(m) for m in moves],
                        "opening_sha256": opening_hash(moves),
                        "colour_arm": colour_arm,
                        "anchor_colour": "red" if colour_arm == "anchor_red" else "black",
                        "seed": seed_for(t_i, r_i, o_i, c_i),
                        "ply_cap": PLY_CAP,
                        "workers": 1,
                    })
    if len(tasks) != EXPECTED_TASKS:
        raise ScheduleError(f"enumerated {len(tasks)} tasks, expected {EXPECTED_TASKS}")
    return tasks


def schedule_invariants(tasks: Sequence[dict]) -> List[str]:
    """Every frozen property, as a checkable list. [] means all hold."""
    bad: List[str] = []
    n_ref, n_open = len(REFERENCES), len(OPENINGS)

    if len(tasks) != 128:
        bad.append(f"task count {len(tasks)} != 128")

    for trials in TRIALS_LADDER:
        per = [t for t in tasks if t["trials"] == trials]
        if len(per) != 32:
            bad.append(f"trials={trials}: {len(per)} tasks, expected 32")
        for ref in REFERENCES:
            pr = [t for t in per if t["reference"] == ref]
            if len(pr) != 16:
                bad.append(f"trials={trials} ref={ref}: {len(pr)} tasks, expected 16")
            for opening_id, _ in OPENINGS:
                po = [t for t in pr if t["opening_id"] == opening_id]
                if len(po) != 2:
                    bad.append(f"trials={trials} ref={ref} {opening_id}: {len(po)}, expected 2")
                if {t["colour_arm"] for t in po} != set(COLOUR_ARMS):
                    bad.append(f"trials={trials} ref={ref} {opening_id}: colour arms not balanced")

    for arm in COLOUR_ARMS:
        n = len([t for t in tasks if t["colour_arm"] == arm])
        if n != 64:
            bad.append(f"colour arm {arm}: {n} tasks, expected 64")

    seeds = [t["seed"] for t in tasks]
    if len(set(seeds)) != len(seeds):
        dupes = sorted({s for s in seeds if seeds.count(s) > 1})
        bad.append(f"duplicate seeds: {dupes}")
    lo, hi = RESERVED_SEEDS
    outside = [s for s in seeds if not (lo <= s < hi)]
    if outside:
        bad.append(f"{len(outside)} seeds outside the reserved interval: {outside[:5]}")
    if seeds and (min(seeds), max(seeds)) != (SEED_BASE, SEED_BASE + 127):
        bad.append(f"seed span {min(seeds)}..{max(seeds)}, expected {SEED_BASE}..{SEED_BASE+127}")
    if sorted(seeds) != list(range(*SCHEDULE_SEEDS)):
        bad.append(f"seeds are not exactly [{SCHEDULE_SEEDS[0]}, {SCHEDULE_SEEDS[1]})")
    burnt = sorted(set(seeds) & set(CONSUMED_SEEDS))
    if burnt:
        bad.append(f"schedule reuses already-consumed seed(s): {burnt}")

    # An opening must be byte-identical wherever it appears, both colour arms included.
    by_id: Dict[str, set] = {}
    for t in tasks:
        by_id.setdefault(t["opening_id"], set()).add(t["opening_sha256"])
    for opening_id, hashes in by_id.items():
        if len(hashes) != 1:
            bad.append(f"{opening_id} appears with {len(hashes)} different move sequences")

    return bad
