"""Checkpoint match: one A-vs-B pairing. Builder + CLI (CLI added later)."""
from __future__ import annotations

from .eval_runner import build_pairing_tasks


def build_match_tasks(a_ckpt: str, b_ckpt: str, games: int, base_seed: int,
                      pairing_id: str):
    """Tasks for a single pairing (pairing_index fixed at 0)."""
    return build_pairing_tasks(pairing_id, a_ckpt, b_ckpt, games, base_seed,
                               pairing_index=0)
