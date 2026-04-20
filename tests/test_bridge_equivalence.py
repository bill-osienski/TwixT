#!/usr/bin/env python3
"""
Bridge crossing equivalence tests.

Validates that optimized bridge crossing implementations match the original
geometry-based algorithm:
- Geometry optimization: _proper_intersect_knight vs segments_intersect
- Bitmask optimization: bridges_cross_fast vs bridges_cross (geometry)

These tests ensure the O(1) bitmask lookup produces identical results to the
full geometric intersection algorithm.
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "GPU"))

from game.state import GameState
from game.bridge import (
    bridges_cross,
    bridges_cross_fast,
    check_crossing,
    normalize_edge,
    rebuild_bridge_mask,
    get_bridges_from_mask,
    segments_intersect,
)
from game.bridge_geom import _proper_intersect_knight
from game.edge_index import get_all_edges, get_edge_to_idx, get_num_edges, get_conflicts

BOARD_SIZE = 24

# Knight-move offsets
KNIGHT_OFFSETS = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
]


def generate_all_knight_edges(board_size: int):
    """Generate all canonical knight-edges on the board."""
    edges = set()
    for r1 in range(board_size):
        for c1 in range(board_size):
            for dr, dc in KNIGHT_OFFSETS:
                r2, c2 = r1 + dr, c1 + dc
                if 0 <= r2 < board_size and 0 <= c2 < board_size:
                    edge = ((r1, c1), (r2, c2)) if (r1, c1) <= (r2, c2) else ((r2, c2), (r1, c1))
                    edges.add(edge)
    return list(edges)


def shares_endpoint(e1, e2):
    """Check if two edges share an endpoint."""
    (a1, a2), (b1, b2) = e1, e2
    return a1 == b1 or a1 == b2 or a2 == b1 or a2 == b2


# =============================================================================
# Geometry Equivalence Tests
# =============================================================================

@pytest.mark.slow
@pytest.mark.bridge
def test_geometry_equivalence():
    """
    Exhaustive test: _proper_intersect_knight vs segments_intersect.

    Verifies that for all knight-edge pairs on 24x24 (that don't share endpoints),
    the simplified _proper_intersect_knight gives the same crossing answer as
    the full segments_intersect.
    """
    edges = generate_all_knight_edges(BOARD_SIZE)
    print(f"Generated {len(edges)} canonical knight-edges on {BOARD_SIZE}x{BOARD_SIZE}")

    tested = 0
    crossings_found = 0
    mismatches = []

    for i, e1 in enumerate(edges):
        (r1, c1), (r2, c2) = e1
        x1, y1, x2, y2 = c1, r1, c2, r2

        for j, e2 in enumerate(edges):
            if i >= j:
                continue

            if shares_endpoint(e1, e2):
                continue

            (r3, c3), (r4, c4) = e2
            x3, y3, x4, y4 = c3, r3, c4, r4

            result_proper = _proper_intersect_knight(x1, y1, x2, y2, x3, y3, x4, y4)
            result_full = segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4)

            tested += 1
            if result_proper:
                crossings_found += 1

            if result_proper != result_full:
                mismatches.append({
                    'e1': e1,
                    'e2': e2,
                    'proper': result_proper,
                    'full': result_full,
                })

        if (i + 1) % 200 == 0:
            print(f"  Tested edges {i+1}/{len(edges)}...")

    print(f"\nResults:")
    print(f"  Total edge pairs tested: {tested:,}")
    print(f"  Crossings found: {crossings_found:,}")
    print(f"  Mismatches: {len(mismatches)}")

    assert len(mismatches) == 0, f"Found {len(mismatches)} mismatches"


# =============================================================================
# Bitmask Equivalence Tests
# =============================================================================

@pytest.mark.bridge
def test_empty_state():
    """No bridges means no crossings."""
    state = GameState()

    for edge in get_all_edges()[:100]:
        (r1, c1), (r2, c2) = edge
        assert not bridges_cross(state, r1, c1, r2, c2)
        assert not bridges_cross_fast(state.bridge_mask, edge)


@pytest.mark.slow
@pytest.mark.bridge
def test_random_configurations(num_configs: int = 50, bridges_per_config: int = 20):
    """
    Test many random bridge configurations.

    For each config:
    - Place random non-crossing bridges
    - Test all candidate edges against both geometry and bitmask
    - Verify they agree
    """
    all_edges = get_all_edges()
    edge_to_idx = get_edge_to_idx()

    total_tested = 0
    total_crossings = 0
    mismatches = 0

    for config_idx in range(num_configs):
        state = GameState()

        shuffled = list(all_edges)
        random.shuffle(shuffled)

        placed = 0
        for edge in shuffled:
            if placed >= bridges_per_config:
                break

            (r1, c1), (r2, c2) = edge
            if not bridges_cross(state, r1, c1, r2, c2):
                state.bridges.add(edge)
                idx = edge_to_idx[edge]
                state.bridge_mask |= (1 << idx)
                placed += 1

        test_edges = random.sample(all_edges, min(200, len(all_edges)))

        for edge in test_edges:
            if edge in state.bridges:
                continue

            (r1, c1), (r2, c2) = edge

            result_geom = bridges_cross(state, r1, c1, r2, c2)
            result_mask = bridges_cross_fast(state.bridge_mask, edge)

            total_tested += 1
            if result_geom:
                total_crossings += 1

            if result_geom != result_mask:
                print(f"MISMATCH config {config_idx}: {edge}")
                print(f"  bridges: {state.bridges}")
                print(f"  geom={result_geom} mask={result_mask}")
                mismatches += 1

        if (config_idx + 1) % 10 == 0:
            print(f"    Tested {config_idx + 1}/{num_configs} configs...")

    print(f"  [PASS] Random configs: {total_tested:,} tests, {total_crossings:,} crossings, {mismatches} mismatches")
    assert mismatches == 0, f"{mismatches} mismatches found"


@pytest.mark.bridge
def test_mask_sync():
    """Verify bridge_mask stays in sync during incremental adds."""
    all_edges = get_all_edges()
    edge_to_idx = get_edge_to_idx()

    state = GameState()

    shuffled = list(all_edges)
    random.shuffle(shuffled)

    placed = 0
    for edge in shuffled:
        if placed >= 30:
            break

        (r1, c1), (r2, c2) = edge
        if not bridges_cross(state, r1, c1, r2, c2):
            state.bridges.add(edge)
            idx = edge_to_idx[edge]
            state.bridge_mask |= (1 << idx)
            placed += 1

            reconstructed = rebuild_bridge_mask(state.bridges)
            assert state.bridge_mask == reconstructed, \
                f"Mask desync after adding {edge}"

    print(f"  [PASS] Mask sync verified for {placed} bridge additions")


@pytest.mark.bridge
def test_rebuild_with_normalization():
    """Verify rebuild_bridge_mask handles un-normalized edges."""
    non_canonical = [
        ((6, 7), (5, 5)),
        ((11, 12), (10, 10)),
    ]

    bridges = set(non_canonical)
    mask = rebuild_bridge_mask(bridges)

    edge_to_idx = get_edge_to_idx()

    for edge in non_canonical:
        normalized = normalize_edge(edge[0], edge[1])
        idx = edge_to_idx.get(normalized)
        assert idx is not None, f"Normalized edge {normalized} not found"
        assert mask & (1 << idx), f"Edge {normalized} not in rebuilt mask"

    print("  [PASS] rebuild_bridge_mask handles non-canonical edges")


@pytest.mark.bridge
def test_get_bridges_from_mask():
    """Verify round-trip: bridges -> mask -> bridges."""
    all_edges = get_all_edges()
    edge_to_idx = get_edge_to_idx()

    original_bridges = set(random.sample(all_edges, 15))

    mask = 0
    for edge in original_bridges:
        idx = edge_to_idx[edge]
        mask |= (1 << idx)

    reconstructed = set(get_bridges_from_mask(mask))

    assert original_bridges == reconstructed, \
        f"Round-trip failed: {original_bridges} != {reconstructed}"

    print("  [PASS] Round-trip bridges -> mask -> bridges")


@pytest.mark.bridge
def test_check_crossing_normalizes():
    """Verify check_crossing() works with non-canonical edges."""
    all_edges = get_all_edges()
    edge_to_idx = get_edge_to_idx()

    state = GameState()

    bridge = all_edges[100]
    state.bridges.add(bridge)
    idx = edge_to_idx[bridge]
    state.bridge_mask |= (1 << idx)

    crossing_edge = None
    for edge in all_edges:
        if edge in state.bridges:
            continue
        (r1, c1), (r2, c2) = edge
        if bridges_cross(state, r1, c1, r2, c2):
            crossing_edge = edge
            break

    if crossing_edge:
        (r1, c1), (r2, c2) = crossing_edge

        assert check_crossing(state.bridge_mask, (r1, c1), (r2, c2))
        assert check_crossing(state.bridge_mask, (r2, c2), (r1, c1))

        print("  [PASS] check_crossing() normalizes correctly")
    else:
        print("  [SKIP] No crossing edge found for check_crossing test")


@pytest.mark.bridge
def test_edge_count():
    """Verify expected edge count for 24x24 board."""
    num_edges = get_num_edges()
    assert num_edges == 2024, f"Expected 2024 edges, got {num_edges}"
    print(f"  [PASS] Edge count: {num_edges}")


@pytest.mark.bridge
def test_conflict_symmetry():
    """Verify conflict matrix is symmetric."""
    conflicts = get_conflicts()
    num_edges = get_num_edges()

    asymmetric = 0
    for i in range(num_edges):
        for j in range(i + 1, num_edges):
            i_has_j = bool(conflicts[i] & (1 << j))
            j_has_i = bool(conflicts[j] & (1 << i))
            if i_has_j != j_has_i:
                asymmetric += 1

    assert asymmetric == 0, f"Found {asymmetric} asymmetric conflict pairs"
    print("  [PASS] Conflict matrix is symmetric")
