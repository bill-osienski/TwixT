"""Sealed lane detection with collision-safe caching.

This module provides BFS-based reachability checking to determine if a player
can still reach their goal edge. Unlike the JS implementation's lossy cache key
(bbox+touches+size), we use a collision-safe key that captures all state
affecting BFS outcome.

Key design:
- LaneKey captures: player, ROI, target_edges, comp_mask, self_mask, opp_mask, bridges_sig
- ROI uses corridor strategy: primary direction extends to missing goal edges,
  orthogonal direction is component bbox ± KNIGHT_MARGIN
- BFS is bounded to the SAME ROI that the key captures
- CRITICAL: Check ROI bounds BEFORE accessing any state (occupancy/bridges_cross)
- All masks use exact bitmasks (collision-free), not hashes
"""
from __future__ import annotations

import struct
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from ..game.state import GameState
from ..game.board import is_valid_placement
from ..game.bridge import KNIGHT_OFFSETS, bridges_cross


def is_goal_edge_coordinate(player_str: str, row: int, col: int, board_size: int) -> bool:
    """Check if a coordinate is on the player's goal edge (not corners).

    Red's goal edges: row 0 and row board_size-1 (excluding corners)
    Black's goal edges: col 0 and col board_size-1 (excluding corners)
    """
    if player_str == "red":
        if row != 0 and row != board_size - 1:
            return False
        return 0 < col < board_size - 1
    else:
        if col != 0 and col != board_size - 1:
            return False
        return 0 < row < board_size - 1

# BFS can reach 2 squares beyond component bbox via knight moves
KNIGHT_MARGIN = 2
# Bridges can block paths up to 4 squares from their endpoints
BRIDGE_MARGIN = 4


@dataclass(frozen=True)
class LaneKey:
    """Collision-safe cache key for sealed lane detection.

    All fields that affect BFS outcome are captured:
    - player: which player we're checking reachability for
    - roi: the bounding region for BFS (r0, r1, c0, c1 inclusive)
    - target_edges: which goal edges we need to reach (bitmask)
    - comp_mask: the component pegs (BFS start set)
    - self_mask: ALL friendly pegs in ROI (BFS can walk through these)
    - opp_mask: opponent pegs in ROI (walls that block)
    - bridges_sig: hash of bridges that could block paths in ROI
    """
    player: int                     # 0=red, 1=black
    roi: Tuple[int, int, int, int]  # (r0, r1, c0, c1) inclusive
    target_edges: int               # bitmask: 1=top/left, 2=bottom/right
    comp_mask: bytes                # component pegs (start set)
    self_mask: bytes                # ALL friendly pegs in ROI
    opp_mask: bytes                 # opponent pegs in ROI (walls)
    bridges_sig: bytes              # hash of relevant bridges


class SealedLaneLRU:
    """LRU cache for sealed lane results, bounded to max_entries."""

    def __init__(self, max_entries: int = 50_000):
        self.max_entries = max_entries
        self._d: OrderedDict[LaneKey, bool] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: LaneKey) -> Optional[bool]:
        """Get cached result, updating LRU order. Returns None if not found."""
        if key in self._d:
            self._d.move_to_end(key)
            self.hits += 1
            return self._d[key]
        self.misses += 1
        return None

    def put(self, key: LaneKey, value: bool) -> None:
        """Store result, evicting oldest if at capacity."""
        if key in self._d:
            self._d.move_to_end(key)
            self._d[key] = value  # Update the value
        else:
            if len(self._d) >= self.max_entries:
                self._d.popitem(last=False)  # Remove oldest
            self._d[key] = value

    def get_or_compute(
        self,
        key: LaneKey,
        compute_fn: Callable[[], bool]
    ) -> bool:
        """Get cached result or compute and cache it."""
        result = self.get(key)
        if result is not None:
            return result
        result = compute_fn()
        self.put(key, result)
        return result

    def clear(self) -> None:
        """Clear all cached entries."""
        self._d.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        total = self.hits + self.misses
        return {
            "size": len(self._d),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }


def _compute_corridor_roi(
    player: int,
    component: List[Tuple[int, int]],
    touches_top_or_left: bool,
    touches_bottom_or_right: bool,
    board_size: int,
) -> Tuple[int, int, int, int]:
    """Compute corridor ROI for BFS.

    Strategy:
    - Primary direction (Red=rows, Black=cols): extend to missing goal edges
    - Orthogonal direction: component bbox ± KNIGHT_MARGIN
    - ALWAYS apply KNIGHT_MARGIN even when touching (BFS can step past component)

    Returns: (r0, r1, c0, c1) inclusive bounds
    """
    if not component:
        return (0, board_size - 1, 0, board_size - 1)

    rows = [r for r, c in component]
    cols = [c for r, c in component]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    if player == 0:  # Red: connects rows (top=0 to bottom=23)
        # Primary direction: rows
        if not touches_top_or_left:
            r0 = 0  # Need to reach top
        else:
            r0 = max(0, min_row - KNIGHT_MARGIN)  # Already touching, but allow margin

        if not touches_bottom_or_right:
            r1 = board_size - 1  # Need to reach bottom
        else:
            r1 = min(board_size - 1, max_row + KNIGHT_MARGIN)  # Already touching, but allow margin

        # Orthogonal direction: cols - bbox ± margin
        c0 = max(0, min_col - KNIGHT_MARGIN)
        c1 = min(board_size - 1, max_col + KNIGHT_MARGIN)
    else:  # Black: connects cols (left=0 to right=23)
        # Primary direction: cols
        if not touches_top_or_left:
            c0 = 0  # Need to reach left
        else:
            c0 = max(0, min_col - KNIGHT_MARGIN)

        if not touches_bottom_or_right:
            c1 = board_size - 1  # Need to reach right
        else:
            c1 = min(board_size - 1, max_col + KNIGHT_MARGIN)

        # Orthogonal direction: rows - bbox ± margin
        r0 = max(0, min_row - KNIGHT_MARGIN)
        r1 = min(board_size - 1, max_row + KNIGHT_MARGIN)

    return (r0, r1, c0, c1)


def _pegs_to_mask(pegs: Iterable[Tuple[int, int]], roi: Tuple[int, int, int, int]) -> bytes:
    """Convert peg positions to exact canonical bitmask within ROI.

    Each bit represents one cell in the ROI. This is collision-free.
    """
    r0, r1, c0, c1 = roi
    roi_width = c1 - c0 + 1
    roi_height = r1 - r0 + 1
    num_bits = roi_width * roi_height
    num_bytes = (num_bits + 7) // 8

    mask = bytearray(num_bytes)
    for r, c in pegs:
        if r0 <= r <= r1 and c0 <= c <= c1:
            # Map (r, c) to bit index within ROI
            bit_idx = (r - r0) * roi_width + (c - c0)
            byte_idx = bit_idx // 8
            bit_pos = bit_idx % 8
            mask[byte_idx] |= (1 << bit_pos)

    return bytes(mask)


def _bridge_signature(state: GameState, roi: Tuple[int, int, int, int]) -> bytes:
    """Compute exact signature of bridges that could affect paths in ROI.

    Uses bbox-intersection to catch bridges whose segment can cross paths in ROI,
    even if endpoints are outside. Returns sorted, packed endpoints (collision-free).
    """
    r0, r1, c0, c1 = roi

    exp_r0 = max(0, r0 - BRIDGE_MARGIN)
    exp_r1 = min(state.board_size - 1, r1 + BRIDGE_MARGIN)
    exp_c0 = max(0, c0 - BRIDGE_MARGIN)
    exp_c1 = min(state.board_size - 1, c1 + BRIDGE_MARGIN)

    relevant = []
    for (a_r, a_c), (b_r, b_c) in state.bridges:
        # Normalize endpoint order (smaller coord first)
        if (b_r, b_c) < (a_r, a_c):
            a_r, a_c, b_r, b_c = b_r, b_c, a_r, a_c

        br_min_r = a_r if a_r < b_r else b_r
        br_max_r = b_r if a_r < b_r else a_r
        br_min_c = a_c if a_c < b_c else b_c
        br_max_c = b_c if a_c < b_c else a_c

        # bbox intersects expanded ROI
        if br_max_r < exp_r0 or br_min_r > exp_r1:
            continue
        if br_max_c < exp_c0 or br_min_c > exp_c1:
            continue

        relevant.append((a_r, a_c, b_r, b_c))

    if not relevant:
        return b""

    relevant.sort()
    return struct.pack(f">{len(relevant) * 4}B", *[v for br in relevant for v in br])


def make_lane_key(
    state: GameState,
    player: int,
    component: List[Tuple[int, int]],
    touches_top_or_left: bool,
    touches_bottom_or_right: bool,
) -> LaneKey:
    """Build a collision-safe cache key for sealed lane detection.

    Args:
        state: Current game state
        player: 0=red, 1=black
        component: List of (row, col) pegs in the component
        touches_top_or_left: Component already reaches top (red) or left (black)
        touches_bottom_or_right: Component already reaches bottom (red) or right (black)

    Returns:
        LaneKey with all state captured
    """
    # Compute corridor ROI
    roi = _compute_corridor_roi(
        player, component,
        touches_top_or_left, touches_bottom_or_right,
        state.board_size
    )

    # Target edges: which goal edges we still need to reach
    target_edges = 0
    if not touches_top_or_left:
        target_edges |= 1
    if not touches_bottom_or_right:
        target_edges |= 2

    # Component mask (BFS start set)
    comp_mask = _pegs_to_mask(component, roi)

    # Self mask (ALL friendly pegs in ROI - BFS can walk through these)
    friendly_pegs = [pos for pos, p in state.pegs.items() if p == player]
    self_mask = _pegs_to_mask(friendly_pegs, roi)

    # Opponent mask (walls that block)
    opponent = 1 - player
    opp_pegs = [pos for pos, p in state.pegs.items() if p == opponent]
    opp_mask = _pegs_to_mask(opp_pegs, roi)

    # Bridge signature
    bridges_sig = _bridge_signature(state, roi)

    return LaneKey(
        player=player,
        roi=roi,
        target_edges=target_edges,
        comp_mask=comp_mask,
        self_mask=self_mask,
        opp_mask=opp_mask,
        bridges_sig=bridges_sig,
    )


def has_reachable_goal_edge_bounded(
    state: GameState,
    player: int,
    component: List[Tuple[int, int]],
    key: LaneKey,
) -> bool:
    """BFS to check if player can reach their goal edge, bounded to ROI.

    CRITICAL INVARIANTS:
    1. BFS is bounded to the same ROI captured in key
    2. Check ROI bounds BEFORE accessing any state
    3. Goal edge success requires is_goal_edge_coordinate() check
    4. Empty cells only expand if is_valid_placement() is true

    Args:
        state: Current game state
        player: 0=red, 1=black
        component: List of (row, col) pegs in the component (BFS start set)
        key: The LaneKey (used for ROI bounds)

    Returns:
        True if goal edge is reachable, False if lane is sealed
    """
    if not component:
        return False

    board_size = state.board_size
    r0, r1, c0, c1 = key.roi

    # Map player int to string for is_valid_placement
    player_str = "red" if player == 0 else "black"

    # Determine which goal edges we need to reach
    need_top_or_left = (key.target_edges & 1) != 0
    need_bottom_or_right = (key.target_edges & 2) != 0

    # If already touching both, lane is open
    if not need_top_or_left and not need_bottom_or_right:
        return True

    # BFS from component pegs
    # Track (row, col, is_peg) to handle empty vs peg semantics
    visited: Set[Tuple[int, int]] = set()
    queue: List[Tuple[int, int, bool]] = []  # (row, col, is_peg)

    # Initialize with component pegs that are in ROI
    for row, col in component:
        if r0 <= row <= r1 and c0 <= col <= c1:
            if (row, col) not in visited:
                visited.add((row, col))
                queue.append((row, col, True))  # True = is a peg

    head = 0
    while head < len(queue):
        row, col, is_peg = queue[head]
        head += 1

        # Check if we've reached a needed goal edge
        # Must be a legal goal edge coordinate (not corner, respects edge restrictions)
        if player == 0:  # Red: rows
            if need_top_or_left and row == 0:
                if is_goal_edge_coordinate(player_str, row, col, board_size):
                    if is_peg or is_valid_placement(state, player_str, row, col):
                        return True
            if need_bottom_or_right and row == board_size - 1:
                if is_goal_edge_coordinate(player_str, row, col, board_size):
                    if is_peg or is_valid_placement(state, player_str, row, col):
                        return True
        else:  # Black: cols
            if need_top_or_left and col == 0:
                if is_goal_edge_coordinate(player_str, row, col, board_size):
                    if is_peg or is_valid_placement(state, player_str, row, col):
                        return True
            if need_bottom_or_right and col == board_size - 1:
                if is_goal_edge_coordinate(player_str, row, col, board_size):
                    if is_peg or is_valid_placement(state, player_str, row, col):
                        return True

        # Explore knight-move neighbors
        for dr, dc in KNIGHT_OFFSETS:
            nr, nc = row + dr, col + dc

            # Skip if out of board
            if nr < 0 or nr >= board_size or nc < 0 or nc >= board_size:
                continue

            # CRITICAL: Check ROI bounds BEFORE reading any state
            if nr < r0 or nr > r1 or nc < c0 or nc > c1:
                continue

            # Skip if already visited
            if (nr, nc) in visited:
                continue

            # Now safe to check occupancy
            cell_owner = state.pegs.get((nr, nc))

            if cell_owner is not None:
                if cell_owner != player:
                    # Opponent peg blocks
                    continue
                # Friendly peg - can traverse
                next_is_peg = True
            else:
                # Empty cell - only expand if placeable
                if not is_valid_placement(state, player_str, nr, nc):
                    continue
                next_is_peg = False

            # Check if bridge crossing blocks this move
            if bridges_cross(state, row, col, nr, nc):
                continue

            visited.add((nr, nc))
            queue.append((nr, nc, next_is_peg))

    # Exhausted BFS without reaching goal edge
    return False


def sealed_lane_open_batch(
    cache: SealedLaneLRU,
    items: List[Tuple[GameState, int, List[Tuple[int, int]], bool, bool]],
) -> List[bool]:
    """Batch check sealed lanes with de-duplication.

    Groups items by LaneKey, computes BFS once per unique key.

    Args:
        cache: LRU cache to use
        items: List of (state, player, component, touches_top_left, touches_bottom_right)

    Returns:
        List of bool results, same order as items
    """
    if not items:
        return []

    # Build keys and group by unique key
    keys: List[LaneKey] = []
    key_to_indices: Dict[LaneKey, List[int]] = {}

    for i, (state, player, component, touches_tl, touches_br) in enumerate(items):
        key = make_lane_key(state, player, component, touches_tl, touches_br)
        keys.append(key)
        if key not in key_to_indices:
            key_to_indices[key] = []
        key_to_indices[key].append(i)

    # Compute results for unique keys
    key_results: Dict[LaneKey, bool] = {}
    for key, indices in key_to_indices.items():
        # Try cache first
        cached = cache.get(key)
        if cached is not None:
            key_results[key] = cached
        else:
            # Compute using first item with this key
            idx = indices[0]
            state, player, component, _, _ = items[idx]
            result = has_reachable_goal_edge_bounded(state, player, component, key)
            cache.put(key, result)
            key_results[key] = result

    # Map results back to original order
    results = [key_results[keys[i]] for i in range(len(items))]
    return results


def check_sealed_lane(
    state: GameState,
    player: int,
    component: List[Tuple[int, int]],
    touches_top_or_left: bool,
    touches_bottom_or_right: bool,
    cache: Optional[SealedLaneLRU] = None,
) -> bool:
    """Convenience wrapper for single sealed lane check.

    Args:
        state: Current game state
        player: 0=red, 1=black
        component: List of (row, col) pegs in the component
        touches_top_or_left: Component reaches top (red) or left (black)
        touches_bottom_or_right: Component reaches bottom (red) or right (black)
        cache: Optional LRU cache

    Returns:
        True if lane is OPEN (can reach goal), False if SEALED
    """
    key = make_lane_key(state, player, component, touches_top_or_left, touches_bottom_or_right)

    if cache is not None:
        return cache.get_or_compute(
            key,
            lambda: has_reachable_goal_edge_bounded(state, player, component, key)
        )

    return has_reachable_goal_edge_bounded(state, player, component, key)
