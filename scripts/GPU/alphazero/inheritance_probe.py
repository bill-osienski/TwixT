"""Phase 0 inheritance preflight -- convergence atlas design, section 2.

Read-only characterization of how much of its search tree a shipped self-play
search inherits through ``MCTS.advance_root`` tree reuse.

This is a technical preflight, not evidence. Its single-game rows are serially
correlated and must not be used to characterize the inheritance distribution.

CPU-safe: stdlib only, no MLX, no scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


DECISION_POINT_SIMS: int = 320
POST_OPENING_MEDIAN_LIMIT: float = 0.10
OVERALL_P75_LIMIT: float = 0.20

PHASE_BOUNDS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("opening", 0, 30),
    ("early_mid", 31, 60),
    ("midgame", 61, 90),
    ("late", 91, None),
)
POST_OPENING_PHASES: Tuple[str, ...] = ("early_mid", "midgame", "late")


def phase_for_ply(ply: int) -> str:
    """Return the frozen phase bucket for a zero-based ply."""
    if ply < 0:
        raise ValueError(f"ply must be non-negative, got {ply}")
    for name, lower, upper in PHASE_BOUNDS:
        if ply >= lower and (upper is None or ply <= upper):
            return name
    raise AssertionError(f"unreachable: no phase for ply {ply}")


def inherited_fraction_320(starting_visits: int) -> float:
    """Return ``starting_visits / (starting_visits + 320)``."""
    if starting_visits < 0:
        raise ValueError(
            f"starting_visits must be non-negative, got {starting_visits}"
        )
    return starting_visits / (starting_visits + DECISION_POINT_SIMS)


@dataclass
class SearchRow:
    """One shipped search's start-of-search telemetry."""

    ply: int
    phase: str
    starting_visits: int
    starting_visited_children: int
    forced_count: int
    inherited_fraction_320: float
    played_child_visits: Optional[int] = None

    @classmethod
    def build(
        cls,
        ply: int,
        starting_visits: int,
        starting_visited_children: int,
        forced_count: int,
    ) -> "SearchRow":
        if starting_visited_children < 0:
            raise ValueError("starting_visited_children must be non-negative")
        if forced_count < 0:
            raise ValueError(f"forced_count must be non-negative, got {forced_count}")
        return cls(
            ply=ply,
            phase=phase_for_ply(ply),
            starting_visits=starting_visits,
            starting_visited_children=starting_visited_children,
            forced_count=forced_count,
            inherited_fraction_320=inherited_fraction_320(starting_visits),
        )
