"""Marathon-termination diagnostics. Pure-function analyzer surface
that computes no-progress windows, adjudication-coverage gate-block
distributions, resign-gate breakdown by game-length partition, and the
value-uncertain guard predicate used by termination knobs.

Spec: docs/superpowers/specs/2026-05-19-marathon-termination-tuning-design.md

All inputs are per-game data already on disk:
  - per-game `goal_completion_record` (dict)
  - per-game `goal_completion_diagnostics` (list of per-ply entries)
  - per-game `meta` (dict; carries adjudication_block_reason from
    self_play.py:1248-1249 via game_saver.py:146)

No self-play change required (Task 0 outcome A).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Optional, Tuple


NO_PROGRESS_WINDOW_SIZE = 15

# §3.1 — local-structural primary_class vocabulary for the no-progress
# window detector. Matches Spec 4's primary_class names + the
# goal_completion classifier vocabulary (completes_endpoint /
# reduces_total_goal_distance / redundant_reinforcement / off_chain
# from self_play._classify_argmax_against_gc, and the Spec 4 broader
# vocabulary including blocks_opponent_closeout).
_LOCAL_STRUCTURAL_CLASSES = frozenset({
    "redundant_reinforcement",
    "off_chain",
    "connects_to_existing_component",
    "improves_own_largest_component",
})


def _is_local_structural(entry: dict) -> bool:
    cls = (entry.get("selected_move_classification") or {}).get("primary_class")
    return cls in _LOCAL_STRUCTURAL_CLASSES


def _own_distance_reduced(entry: dict) -> bool:
    cls_info = entry.get("selected_move_classification") or {}
    before = cls_info.get("total_goal_distance_before")
    after = cls_info.get("total_goal_distance_after")
    if before is None or after is None:
        return False
    return int(after) < int(before)


def _completed_endpoint(entry: dict) -> bool:
    cls = (entry.get("selected_move_classification") or {}).get("primary_class")
    return cls == "completes_endpoint"


def _blocked_opponent(entry: dict) -> bool:
    """Opponent block (Spec §3.1, definition pinned).

    Primary signal: `selected_move_classification.primary_class ==
    "blocks_opponent_closeout"` (Spec 4 defense-classifier output).

    This is the SHARED definition with the recovery-retargeting
    diagnostic — both consult the same classifier output via
    `primary_class`, so the two diagnostics cannot diverge on what
    counts as a block.
    """
    cls = (entry.get("selected_move_classification") or {}).get("primary_class")
    return cls == "blocks_opponent_closeout"


def detect_no_progress_windows(diagnostics: list, *, side: str) -> int:
    """Count distinct sliding-window no-progress runs for `side`.

    A no-progress window of length NO_PROGRESS_WINDOW_SIZE (=15) ends at
    own-move ply t iff the trailing 15 own-moves for this side all satisfy:
      - moves are local-structural (primary_class in the structural set)
      - no own-move reduced own total_goal_distance
      - no own-move completed an endpoint
      - no own-move was an opponent block

    Overlapping windows anchored at distinct end-plies count separately
    only if there's at least 1 own-move gap between their end-plies;
    otherwise they collapse into the longest contiguous run.

    Concretely: count maximal-length runs of consecutive own-moves
    satisfying the four conditions, then sum floor(run_length / 15) for
    each run.
    """
    own_entries = [e for e in (diagnostics or []) if e.get("side_to_move") == side]
    if len(own_entries) < NO_PROGRESS_WINDOW_SIZE:
        return 0

    def is_no_progress(e: dict) -> bool:
        return (
            _is_local_structural(e)
            and not _own_distance_reduced(e)
            and not _completed_endpoint(e)
            and not _blocked_opponent(e)
        )

    windows = 0
    run_len = 0
    for e in own_entries:
        if is_no_progress(e):
            run_len += 1
        else:
            windows += run_len // NO_PROGRESS_WINDOW_SIZE
            run_len = 0
    windows += run_len // NO_PROGRESS_WINDOW_SIZE
    return windows
