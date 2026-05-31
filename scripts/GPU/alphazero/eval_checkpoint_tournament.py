"""Checkpoint tournament: many pairings in one flat task list. Builder + CLI."""
from __future__ import annotations

import os

# short_id / resolve_checkpoint live in eval_runner (shared low-level module)
# and are re-exported here so callers/tests can import them from either place.
# This avoids a circular import (eval_summary also needs short_id).
from .eval_runner import build_pairing_tasks, short_id, resolve_checkpoint


def build_tournament_tasks(pairings, games: int, base_seed: int):
    """Flat task list across all pairings. pairings: list[(a_ckpt, b_ckpt)].

    Each pairing gets a distinct pairing_index, so task_ids and seeds never
    collide (stride = GAMES_PER_PAIRING_LIMIT).
    """
    tasks = []
    for idx, (a_ckpt, b_ckpt) in enumerate(pairings):
        pairing_id = f"{short_id(a_ckpt)}_vs_{short_id(b_ckpt)}"
        tasks.extend(build_pairing_tasks(pairing_id, a_ckpt, b_ckpt, games,
                                         base_seed, pairing_index=idx))
    return tasks
