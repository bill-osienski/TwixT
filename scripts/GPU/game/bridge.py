"""Bridge management for TwixT game.

Optimized for TwixT's knight-move bridges:
- Phase 1: Bbox rejection + simplified intersection test
- Phase 2: O(1) bitmask crossing check via precomputed conflicts
"""

from __future__ import annotations

from typing import List, Tuple, Set

from .state import GameState
from .bridge_geom import _orient, _proper_intersect_knight
from .edge_index import (
    get_edge_to_idx,
    get_conflicts,
    get_all_edges,
    get_num_edges,
    Edge,
    BOARD_SIZE,
    KNIGHT_OFFSETS,
)


def normalize_edge(
    a: Tuple[int, int], b: Tuple[int, int]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Normalize bridge endpoints to canonical order: smaller tuple first."""
    return (a, b) if a <= b else (b, a)


# =============================================================================
# Phase 2: O(1) Crossing Check
# =============================================================================

def bridges_cross_fast(bridge_mask: int, edge: Edge) -> bool:
    """
    O(1) crossing check using precomputed conflict bitmasks.

    IMPORTANT: Requires canonical edge (use normalize_edge first).
    Returns False for invalid/non-canonical edges (silent failure).
    For safe external use, prefer check_crossing().

    Args:
        bridge_mask: Current state's bridge bitmask
        edge: Candidate bridge as canonical ((r1,c1), (r2,c2))

    Returns:
        True if candidate crosses any existing bridge
    """
    idx = get_edge_to_idx().get(edge)
    if idx is None:
        return False  # Invalid/non-canonical edge
    return (bridge_mask & get_conflicts()[idx]) != 0


def check_crossing(
    bridge_mask: int, p1: Tuple[int, int], p2: Tuple[int, int]
) -> bool:
    """
    Safe O(1) crossing check that normalizes before checking.

    Use this for external callers that may not have canonical edges.

    Args:
        bridge_mask: Current state's bridge bitmask
        p1, p2: Bridge endpoints (row, col) - order doesn't matter

    Returns:
        True if candidate crosses any existing bridge
    """
    edge = normalize_edge(p1, p2)
    return bridges_cross_fast(bridge_mask, edge)


# =============================================================================
# Phase 1: Geometry-based Crossing Check (Fallback)
# =============================================================================

def bridges_cross(state: GameState, r1: int, c1: int, r2: int, c2: int) -> bool:
    """
    Check if candidate bridge (r1,c1)-(r2,c2) crosses any existing bridge.

    Phase 1 version: O(n) iteration with bbox rejection.
    Kept as fallback for verification and non-standard board sizes.

    Uses x=col, y=row convention to match JS.
    """
    # Debug-only invariant: ensure this is a knight-edge
    if __debug__:
        dr = r2 - r1
        dc = c2 - c1
        adr = dr if dr >= 0 else -dr
        adc = dc if dc >= 0 else -dc
        assert (adr, adc) in ((1, 2), (2, 1)), \
            f"bridges_cross assumes knight-edge segments, got delta ({dr}, {dc})"

    bridges = state.bridges
    if not bridges:
        return False  # Early exit - no bridges means no crossings

    proper = _proper_intersect_knight  # Bind local for speed

    # x=col, y=row
    a1x, a1y = c1, r1
    a2x, a2y = c2, r2

    # Candidate bbox
    a_minx = a1x if a1x < a2x else a2x
    a_maxx = a2x if a1x < a2x else a1x
    a_miny = a1y if a1y < a2y else a2y
    a_maxy = a2y if a1y < a2y else a1y

    for (br1, bc1), (br2, bc2) in bridges:
        # Shared endpoint is legal (not a crossing)
        if ((r1 == br1 and c1 == bc1) or (r1 == br2 and c1 == bc2) or
            (r2 == br1 and c2 == bc1) or (r2 == br2 and c2 == bc2)):
            continue

        # Bbox reject (cheap - skips most bridges)
        b_minx = bc1 if bc1 < bc2 else bc2
        b_maxx = bc2 if bc1 < bc2 else bc1
        if b_maxx < a_minx or b_minx > a_maxx:
            continue

        b_miny = br1 if br1 < br2 else br2
        b_maxy = br2 if br1 < br2 else br1
        if b_maxy < a_miny or b_miny > a_maxy:
            continue

        # Proper intersection only (fast for knight edges)
        if proper(a1x, a1y, a2x, a2y, bc1, br1, bc2, br2):
            return True

    return False


# =============================================================================
# Bridge Creation (Phase 2 optimized)
# =============================================================================

def add_bridges_for_new_peg(
    state: GameState, player: str, row: int, col: int
) -> List[Edge]:
    """
    Create all legal non-crossing bridges from the new peg.

    Updates both state.bridges (set) and state.bridge_mask (bitmask).
    Uses O(1) bitmask crossing check for performance.

    Args:
        state: Current game state (modified in place)
        player: "red" or "black"
        row, col: Position of newly placed peg

    Returns:
        List of bridges that were created
    """
    # Board size assertion for Phase 2
    assert state.board_size == BOARD_SIZE, \
        f"Edge index assumes {BOARD_SIZE}x{BOARD_SIZE} board, got {state.board_size}"

    # Hoist lookups outside the loop (critical for performance)
    edge_to_idx = get_edge_to_idx()
    conflicts = get_conflicts()
    bridge_mask = state.bridge_mask
    pegs = state.pegs
    bridges = state.bridges
    board_size = state.board_size

    created: List[Edge] = []

    for dr, dc in KNIGHT_OFFSETS:
        r2, c2 = row + dr, col + dc

        # Bounds check
        if not (0 <= r2 < board_size and 0 <= c2 < board_size):
            continue

        # Same player's peg at other end?
        if pegs.get((r2, c2)) != player:
            continue

        # Normalize edge to canonical form
        edge = normalize_edge((row, col), (r2, c2))

        # Already exists?
        if edge in bridges:
            continue

        # Get edge index
        idx = edge_to_idx.get(edge)
        if idx is None:
            continue  # Invalid edge (shouldn't happen)

        # O(1) crossing check - INLINED for hot loop performance
        if bridge_mask & conflicts[idx]:
            continue

        # Add bridge to both representations (keeps them in sync)
        bridges.add(edge)
        bridge_mask |= (1 << idx)
        created.append(edge)

    # Write back the updated mask
    state.bridge_mask = bridge_mask

    return created


# =============================================================================
# Utility Functions
# =============================================================================

def rebuild_bridge_mask(bridges: Set[Edge]) -> int:
    """
    Reconstruct bridge_mask from bridges set.

    Use when loading saved games or migrating existing states.
    Defensively normalizes each edge in case of legacy un-normalized data.

    Args:
        bridges: Set of bridge edges (may or may not be canonical)

    Returns:
        Bitmask representing all bridges
    """
    edge_to_idx = get_edge_to_idx()
    mask = 0
    for edge in bridges:
        # Defensive normalization for legacy data
        normalized = normalize_edge(edge[0], edge[1])
        idx = edge_to_idx.get(normalized)
        if idx is not None:
            mask |= (1 << idx)
    return mask


def get_bridges_from_mask(bridge_mask: int) -> List[Edge]:
    """
    Reconstruct bridge list from bitmask.

    Useful for Phase 3 when mask becomes canonical,
    or for debugging/display purposes.

    Args:
        bridge_mask: Bitmask of placed bridges

    Returns:
        List of canonical bridge edges
    """
    all_edges = get_all_edges()
    num_edges = get_num_edges()

    result = []
    for i in range(num_edges):
        if bridge_mask & (1 << i):
            result.append(all_edges[i])
    return result


# =============================================================================
# Legacy: Full Segment Intersection (kept for reference/testing)
# =============================================================================

def _on_segment(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> bool:
    """Check if point C lies on segment AB (assuming collinear)."""
    return (
        min(ax, bx) <= cx <= max(ax, bx) and
        min(ay, by) <= cy <= max(ay, by)
    )


def segments_intersect(
    x1: int, y1: int, x2: int, y2: int,
    x3: int, y3: int, x4: int, y4: int
) -> bool:
    """
    Full segment intersection test (handles collinear cases).

    Kept for:
    - Testing/verification against _proper_intersect_knight
    - Non-knight edge use cases (if any)
    """
    o1 = _orient(x1, y1, x2, y2, x3, y3)
    o2 = _orient(x1, y1, x2, y2, x4, y4)
    o3 = _orient(x3, y3, x4, y4, x1, y1)
    o4 = _orient(x3, y3, x4, y4, x2, y2)

    # Proper intersection (exclude endpoint-only touching)
    if o1 != o2 and o3 != o4:
        endpoint_touch = (
            (o1 == 0 and _on_segment(x1, y1, x2, y2, x3, y3)) or
            (o2 == 0 and _on_segment(x1, y1, x2, y2, x4, y4)) or
            (o3 == 0 and _on_segment(x3, y3, x4, y4, x1, y1)) or
            (o4 == 0 and _on_segment(x3, y3, x4, y4, x2, y2))
        )
        return not endpoint_touch

    # Collinear overlaps
    for orient_val, px, py, qx, qy, rx, ry in [
        (o1, x1, y1, x2, y2, x3, y3),
        (o2, x1, y1, x2, y2, x4, y4),
        (o3, x3, y3, x4, y4, x1, y1),
        (o4, x3, y3, x4, y4, x2, y2),
    ]:
        if orient_val == 0 and _on_segment(px, py, qx, qy, rx, ry):
            shares = (rx == px and ry == py) or (rx == qx and ry == qy)
            if not shares:
                return True

    return False
