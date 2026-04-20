"""
Precomputed edge indexing and conflict bitmasks for O(1) crossing checks.

All data is lazily computed on first access to avoid import-time penalty.
Conflict matrix uses symmetric computation for ~2x faster precompute.

Edge count for 24x24: 2024 undirected knight edges.
Crossing count: 8752 (0.4% of all pairs).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Type alias for canonical edge: ((r1,c1), (r2,c2)) where (r1,c1) <= (r2,c2)
Edge = Tuple[Tuple[int, int], Tuple[int, int]]

BOARD_SIZE = 24
KNIGHT_OFFSETS = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
]


# =============================================================================
# Lazy-loaded module state
# =============================================================================

_ALL_EDGES: List[Edge] | None = None
_EDGE_TO_IDX: Dict[Edge, int] | None = None
_CONFLICTS: List[int] | None = None


# =============================================================================
# Edge Generation
# =============================================================================

def _generate_all_edges() -> List[Edge]:
    """
    Generate all canonical knight-edges on BOARD_SIZE x BOARD_SIZE.

    For 24x24: 2024 edges (directed would be 4048, divided by 2).
    Sorted for deterministic, stable indexing.
    """
    edges = set()
    for r1 in range(BOARD_SIZE):
        for c1 in range(BOARD_SIZE):
            for dr, dc in KNIGHT_OFFSETS:
                r2, c2 = r1 + dr, c1 + dc
                if 0 <= r2 < BOARD_SIZE and 0 <= c2 < BOARD_SIZE:
                    # Canonical order: smaller tuple first
                    if (r1, c1) <= (r2, c2):
                        edge = ((r1, c1), (r2, c2))
                    else:
                        edge = ((r2, c2), (r1, c1))
                    edges.add(edge)
    # Sort for deterministic ordering (critical for stable indexing)
    return sorted(edges)


def get_all_edges() -> List[Edge]:
    """Get list of all canonical edges (lazy-loaded)."""
    global _ALL_EDGES
    if _ALL_EDGES is None:
        _ALL_EDGES = _generate_all_edges()
    return _ALL_EDGES


def get_edge_to_idx() -> Dict[Edge, int]:
    """Get edge -> index mapping (lazy-loaded)."""
    global _EDGE_TO_IDX
    if _EDGE_TO_IDX is None:
        _EDGE_TO_IDX = {e: i for i, e in enumerate(get_all_edges())}
    return _EDGE_TO_IDX


def get_num_edges() -> int:
    """Get total number of edges (2024 for 24x24 board)."""
    return len(get_all_edges())


# =============================================================================
# Conflict Matrix (lazy-loaded, symmetric computation)
# =============================================================================

def _shares_endpoint(e1: Edge, e2: Edge) -> bool:
    """Check if two edges share an endpoint."""
    (a1, a2), (b1, b2) = e1, e2
    return a1 == b1 or a1 == b2 or a2 == b1 or a2 == b2


def _compute_conflicts() -> List[int]:
    """
    Precompute conflict bitmasks for all edges.

    conflicts[i] = bitmask where bit j=1 if edge j crosses edge i.

    Uses symmetric computation (j > i, set both bits) for ~2x speedup.
    Takes ~0.25s on first call, then cached.
    """
    # Import here to avoid circular import at module level
    from .bridge_geom import _proper_intersect_knight

    all_edges = get_all_edges()
    num_edges = len(all_edges)
    conflicts = [0] * num_edges

    for i, e1 in enumerate(all_edges):
        (r1, c1), (r2, c2) = e1
        # x=col, y=row convention
        x1, y1, x2, y2 = c1, r1, c2, r2

        # Only check j > i (symmetric optimization)
        for j in range(i + 1, num_edges):
            e2 = all_edges[j]

            # Shared endpoints are legal, not a crossing
            if _shares_endpoint(e1, e2):
                continue

            (r3, c3), (r4, c4) = e2
            x3, y3, x4, y4 = c3, r3, c4, r4

            if _proper_intersect_knight(x1, y1, x2, y2, x3, y3, x4, y4):
                # Set both bits (symmetric)
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)

    return conflicts


def get_conflicts() -> List[int]:
    """Get conflict bitmasks (lazy-loaded, ~0.25s first call)."""
    global _CONFLICTS
    if _CONFLICTS is None:
        _CONFLICTS = _compute_conflicts()
    return _CONFLICTS


# =============================================================================
# Convenience Functions
# =============================================================================

def edge_to_idx(edge: Edge) -> int | None:
    """Convert canonical edge to index. Returns None if invalid."""
    return get_edge_to_idx().get(edge)


def idx_to_edge(idx: int) -> Edge:
    """Convert index to canonical edge."""
    return get_all_edges()[idx]
