"""Helpers for replaying manual game logs (logs/human-games/*.json).

The recorder appends to the moves[] array on every committed move, including
moves that were later undone. Each entry carries `state.ply_in_state` — the
actual game-state ply at the time of recording. When the user undoes and
plays a different move, the orphaned undone entry stays in the file with
the same ply_in_state as the replacement entry that follows it.

Naively iterating moves[] (treating it as append-only) replays the orphaned
moves and diverges from the actual game. Use `canonical_moves(log)` to
collapse the log to the canonical played line (last write wins per ply).
"""
from __future__ import annotations

from typing import Any, Dict, List


def canonical_moves(log: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the canonical played line from a manual-game log.

    Strategy: bucket entries by `state.ply_in_state`, last write wins per
    bucket. After undo+replay, the replacement move shares ply_in_state with
    the orphaned undone move; keeping the later append discards the undone
    one. Entries missing ply_in_state are treated as their position in the
    raw moves[] array (best-effort fallback for older logs).

    Args:
        log: Parsed manual-game JSON (the full document, not just moves[]).

    Returns:
        List of move dicts in ply_in_state order. Same shape as the input
        entries — no fields are dropped.
    """
    raw = log.get("moves") or []
    by_ply: Dict[int, Dict[str, Any]] = {}
    for fallback_ply, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        state = entry.get("state") or {}
        ply = state.get("ply_in_state")
        if not isinstance(ply, int):
            ply = fallback_ply
        by_ply[ply] = entry  # last write wins
    return [by_ply[k] for k in sorted(by_ply.keys())]


def starting_player(log: Dict[str, Any]) -> str:
    """Best-effort recovery of who moved first.

    Reads from the first canonical move's `to_move`; the recorder always
    sets to_move on every move entry.
    """
    moves = canonical_moves(log)
    if moves and moves[0].get("to_move") in ("red", "black"):
        return moves[0]["to_move"]
    raise ValueError("could not infer starting_player from log")
