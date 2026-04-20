from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# Type aliases for clarity and reuse
Pos = Tuple[int, int]
Edge = Tuple[Pos, Pos]

# Components are immutable to prevent cache corruption
Component = Tuple[Pos, ...]
Components = Tuple[Component, ...]


@dataclass
class GameState:
    """Canonical TwixT game state.

    Includes both set-based bridges (for compatibility) and
    bitmask bridges (for O(1) crossing checks).
    """

    board_size: int = 24
    to_move: str = "red"  # Red moves first

    # (row, col) -> "red"|"black"
    pegs: Dict[Pos, str] = field(default_factory=dict)

    # Each bridge is stored as ((r1,c1),(r2,c2)) with endpoints ordered.
    # Kept for compatibility during Phase 2 transition.
    bridges: Set[Edge] = field(default_factory=set)

    move_history: List[Tuple[str, int, int]] = field(default_factory=list)

    # Phase 2: bitmask of placed bridges (bit i = 1 means edge i exists)
    # Kept in sync with bridges set during transition period.
    # In Phase 3, this becomes the canonical representation.
    bridge_mask: int = 0

    # ---- Connected Components / Adjacency caches ----
    cc_revision: int = 0

    # player -> (revision, components)
    _cc_cache: Dict[str, Tuple[int, Components]] = field(
        default_factory=dict, repr=False, compare=False
    )

    # (revision, {"red": adj, "black": adj})
    _adj_cache: Optional[Tuple[int, Dict[str, Dict[Pos, List[Pos]]]]] = field(
        default=None, repr=False, compare=False
    )

    def invalidate_cc_cache(self) -> None:
        """Call after any mutation to pegs/bridges/bridge_mask."""
        self.cc_revision += 1
        self._cc_cache.clear()
        self._adj_cache = None

    def copy(self) -> "GameState":
        """Create a copy with fresh (empty) caches."""
        return GameState(
            board_size=self.board_size,
            to_move=self.to_move,
            pegs=dict(self.pegs),
            bridges=set(self.bridges),
            move_history=list(self.move_history),
            bridge_mask=self.bridge_mask,  # int copy is O(1)
            # Revision carries forward; invalidate_cc_cache() will bump after mutation
            cc_revision=self.cc_revision,
            # Caches always start empty on a new state object
            _cc_cache={},
            _adj_cache=None,
        )
