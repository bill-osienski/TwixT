"""Phase 0 inheritance preflight -- convergence atlas design, section 2.

Read-only characterization of how much of its search tree a shipped self-play
search inherits through ``MCTS.advance_root`` tree reuse.

This is a technical preflight, not evidence. Its single-game rows are serially
correlated and must not be used to characterize the inheritance distribution.

CPU-safe: stdlib only, no MLX, no scipy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


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


@dataclass
class InheritanceProbeConfig:
    """Opt-in switch, mirroring the existing self-play tracker pattern."""

    enabled: bool = True


class InheritanceProbeTracker:
    """Record one strictly ordered three-call lifecycle per search."""

    def __init__(self, config: InheritanceProbeConfig) -> None:
        self.config = config
        self.rows: List[SearchRow] = []
        self._open_row: Optional[SearchRow] = None
        self._open_forced_before: Optional[int] = None
        self._awaiting_search_end = False

    def observe_search_start(
        self, ply: int, root: Any, forced_sims_total: int
    ) -> None:
        if not self.config.enabled:
            return
        if self._open_row is not None:
            raise RuntimeError(
                f"observe_search_start at ply {ply} with an unclosed row from "
                f"ply {self._open_row.ply}; observe_played_child was not called"
            )
        visited = sum(
            1
            for child in root.children.values()
            if getattr(child, "visit_count", 0) > 0
        )
        row = SearchRow.build(
            ply=ply,
            starting_visits=root.visit_count,
            starting_visited_children=visited,
            forced_count=0,
        )
        self.rows.append(row)
        self._open_row = row
        self._open_forced_before = forced_sims_total
        self._awaiting_search_end = True

    def observe_search_end(self, forced_sims_total: int) -> None:
        if not self.config.enabled:
            return
        if self._open_row is None or not self._awaiting_search_end:
            raise RuntimeError("observe_search_end called with no open search row")
        if self._open_forced_before is None:
            raise RuntimeError("observe_search_end missing its start counter")
        delta = forced_sims_total - self._open_forced_before
        if delta < 0:
            raise ValueError(
                "td1 forced-sims counter went backwards during ply "
                f"{self._open_row.ply} ({self._open_forced_before} -> "
                f"{forced_sims_total}); the telemetry counter was reset mid-search"
            )
        self._open_row.forced_count = delta
        self._awaiting_search_end = False

    def observe_played_child(self, visits: Optional[int]) -> None:
        if not self.config.enabled:
            return
        if self._open_row is None:
            raise RuntimeError("observe_played_child called with no open search row")
        if self._awaiting_search_end:
            raise RuntimeError(
                f"observe_played_child at ply {self._open_row.ply} before "
                "observe_search_end; forced_count would be left provisional"
            )
        self._open_row.played_child_visits = visits
        self._open_row = None
        self._open_forced_before = None

    def finalize_game(self) -> Dict[str, Any]:
        if self._open_row is not None:
            raise RuntimeError(
                f"finalize_game with an unclosed row at ply {self._open_row.ply}"
            )
        return {"rows": [row.__dict__.copy() for row in self.rows]}
