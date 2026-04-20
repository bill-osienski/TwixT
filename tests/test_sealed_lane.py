"""Sanity tests for sealed lane detection.

Tests:
1. Cache vs no-cache: Same results, faster with cache
2. Batch vs naive: Same results for batch API
3. Basic semantics: Open lanes, sealed lanes, edge cases
"""
import sys
import time
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.GPU.game.state import GameState
from scripts.GPU.game.rules import apply_move
from scripts.GPU.ai.sealed_lane import (
    SealedLaneLRU,
    make_lane_key,
    has_reachable_goal_edge_bounded,
    check_sealed_lane,
    sealed_lane_open_batch,
)
from scripts.GPU.ai.heuristics import component_metrics


def create_test_state() -> GameState:
    """Create a fresh game state."""
    return GameState(board_size=24, to_move="red")


def apply_moves(state: GameState, moves: list) -> GameState:
    """Apply a sequence of moves to a state."""
    for row, col in moves:
        state = apply_move(state, row, col)
    return state


class TestSealedLaneLRU:
    """Tests for the LRU cache."""

    def test_basic_cache_operations(self):
        """Test get, put, hit/miss tracking."""
        cache = SealedLaneLRU(max_entries=100)

        # Create a dummy key
        state = create_test_state()
        state = apply_moves(state, [(12, 12)])  # Red plays center
        metrics = component_metrics(state, "red")
        key = make_lane_key(state, 0, metrics["largest_component"], False, False)

        # Miss on first access
        result = cache.get(key)
        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0

        # Put and get
        cache.put(key, True)
        result = cache.get(key)
        assert result is True
        assert cache.hits == 1

        print("  basic cache operations: PASS")

    def test_lru_eviction(self):
        """Test that oldest entries are evicted at capacity."""
        cache = SealedLaneLRU(max_entries=3)

        state = create_test_state()

        # Create 4 different keys by making different moves
        keys = []
        for i, pos in enumerate([(10, 10), (10, 12), (12, 10), (14, 10)]):
            s = apply_moves(state, [pos])
            metrics = component_metrics(s, "red")
            key = make_lane_key(s, 0, metrics["largest_component"], False, False)
            keys.append(key)
            cache.put(key, i % 2 == 0)  # Alternate True/False

        # Oldest key should be evicted
        stats = cache.stats()
        assert stats["size"] == 3

        # First key should be gone
        assert cache.get(keys[0]) is None
        # Others should still be there
        assert cache.get(keys[1]) is not None
        assert cache.get(keys[2]) is not None
        assert cache.get(keys[3]) is not None

        print("  LRU eviction: PASS")


class TestSealedLaneSemantics:
    """Tests for sealed lane detection correctness."""

    def test_empty_board_lane_open(self):
        """An empty component has no lane to check."""
        state = create_test_state()
        metrics = component_metrics(state, "red")

        # No pegs = lane is considered closed (can't reach goal)
        assert len(metrics["largest_component"]) == 0

        print("  empty board semantics: PASS")

    def test_single_peg_lane_open(self):
        """A single peg in the middle should have an open lane."""
        state = create_test_state()
        state = apply_moves(state, [(12, 12)])  # Red at center

        metrics = component_metrics(state, "red")
        assert len(metrics["largest_component"]) == 1

        # Check lane is open (can reach both edges via knight moves)
        result = check_sealed_lane(
            state, 0, metrics["largest_component"],
            False, False, None
        )
        assert result is True, "Single center peg should have open lane"

        print("  single peg lane open: PASS")

    def test_touching_one_edge(self):
        """A peg on row 0 should only need to reach row 23."""
        state = create_test_state()
        state = apply_moves(state, [(0, 12)])  # Red on top edge

        metrics = component_metrics(state, "red")
        touches_top = metrics["touches_top"]
        touches_bottom = metrics["touches_bottom"]

        assert touches_top is True
        assert touches_bottom is False

        # Lane should be open (can still reach bottom)
        result = check_sealed_lane(
            state, 0, metrics["largest_component"],
            touches_top, touches_bottom, None
        )
        assert result is True, "Peg on top edge should be able to reach bottom"

        print("  touching one edge: PASS")

    def test_spanning_both_edges(self):
        """Component touching both edges should return True immediately."""
        state = create_test_state()
        # Build a connected path from top to bottom (simplified)
        # Just place pegs on top and bottom - they won't be connected but
        # we can test the "touches both" case
        state = apply_moves(state, [(0, 12)])  # Red on top
        state = apply_move(state, 12, 12)  # Black
        state = apply_move(state, 23, 10)  # Red on bottom

        # Note: These pegs aren't bridge-connected, so largest component
        # is just one peg. Let's test the case where touches_both is True
        # via fake metrics
        component = [(0, 12), (2, 11), (4, 12), (6, 11), (8, 12),
                     (10, 11), (12, 12), (14, 11), (16, 12),
                     (18, 11), (20, 12), (22, 11), (23, 10)]

        # When already touching both edges, lane is open by definition
        result = check_sealed_lane(
            state, 0, component,
            True, True, None  # touches both
        )
        assert result is True, "Spanning both edges should be open"

        print("  spanning both edges: PASS")


class TestCacheVsNoCache:
    """Test that cached and uncached results match."""

    def test_results_match(self):
        """Cache should return same results as uncached computation."""
        state = create_test_state()
        # Build a small game position
        moves = [(12, 12), (12, 11), (14, 11), (10, 12), (16, 12)]
        for r, c in moves:
            state = apply_move(state, r, c)

        red_metrics = component_metrics(state, "red")
        component = red_metrics["largest_component"]
        touches_top = red_metrics["touches_top"]
        touches_bottom = red_metrics["touches_bottom"]

        # Compute without cache
        result_no_cache = check_sealed_lane(
            state, 0, component, touches_top, touches_bottom, None
        )

        # Compute with cache
        cache = SealedLaneLRU(max_entries=1000)
        result_with_cache = check_sealed_lane(
            state, 0, component, touches_top, touches_bottom, cache
        )

        assert result_no_cache == result_with_cache, \
            f"Cache mismatch: no_cache={result_no_cache}, with_cache={result_with_cache}"

        # Second call should hit cache
        result_cached = check_sealed_lane(
            state, 0, component, touches_top, touches_bottom, cache
        )
        assert result_cached == result_no_cache
        assert cache.hits >= 1

        print("  cache vs no-cache results match: PASS")

    def test_cache_is_faster(self):
        """Cache hits should be faster than recomputing."""
        state = create_test_state()
        moves = [(12, 12), (12, 11), (14, 11), (10, 12), (16, 12),
                 (18, 10), (8, 13), (20, 11)]
        for r, c in moves:
            state = apply_move(state, r, c)

        red_metrics = component_metrics(state, "red")
        component = red_metrics["largest_component"]
        touches_top = red_metrics["touches_top"]
        touches_bottom = red_metrics["touches_bottom"]

        cache = SealedLaneLRU(max_entries=1000)

        # First call populates cache
        check_sealed_lane(state, 0, component, touches_top, touches_bottom, cache)

        # Time uncached calls
        n_iterations = 100
        start = time.perf_counter()
        for _ in range(n_iterations):
            check_sealed_lane(state, 0, component, touches_top, touches_bottom, None)
        uncached_time = time.perf_counter() - start

        # Time cached calls
        start = time.perf_counter()
        for _ in range(n_iterations):
            check_sealed_lane(state, 0, component, touches_top, touches_bottom, cache)
        cached_time = time.perf_counter() - start

        # Cache should be significantly faster
        speedup = uncached_time / cached_time if cached_time > 0 else float('inf')
        print(f"  cache speedup: {speedup:.1f}x ({n_iterations} iterations)")
        print(f"    uncached: {uncached_time*1000:.2f}ms, cached: {cached_time*1000:.2f}ms")

        assert speedup > 2, f"Cache should be at least 2x faster, got {speedup:.1f}x"
        print("  cache is faster: PASS")


class TestBatchVsNaive:
    """Test that batch API matches naive single-item calls."""

    def test_batch_results_match_naive(self):
        """Batch API should return same results as single calls."""
        state = create_test_state()
        moves = [(12, 12), (12, 11), (14, 11), (10, 12)]
        for r, c in moves:
            state = apply_move(state, r, c)

        # Create multiple items to check
        items = []
        expected = []

        for player, player_int in [("red", 0), ("black", 1)]:
            metrics = component_metrics(state, player)
            component = metrics["largest_component"]
            if player == "red":
                touches_tl = metrics["touches_top"]
                touches_br = metrics["touches_bottom"]
            else:
                touches_tl = metrics["touches_left"]
                touches_br = metrics["touches_right"]

            items.append((state, player_int, component, touches_tl, touches_br))

            # Compute expected via naive method
            result = check_sealed_lane(
                state, player_int, component, touches_tl, touches_br, None
            )
            expected.append(result)

        # Run batch
        cache = SealedLaneLRU(max_entries=1000)
        batch_results = sealed_lane_open_batch(cache, items)

        assert batch_results == expected, \
            f"Batch mismatch: batch={batch_results}, expected={expected}"

        print("  batch results match naive: PASS")

    def test_batch_dedup(self):
        """Batch should deduplicate identical keys."""
        state = create_test_state()
        state = apply_move(state, 12, 12)  # Red at center

        metrics = component_metrics(state, "red")
        component = metrics["largest_component"]

        # Create 5 identical items
        items = [(state, 0, component, False, False) for _ in range(5)]

        cache = SealedLaneLRU(max_entries=1000)
        results = sealed_lane_open_batch(cache, items)

        # All should be the same
        assert len(set(results)) == 1, "All identical items should give same result"

        # Should only have 1 miss (dedup worked)
        assert cache.misses == 1, f"Expected 1 miss (dedup), got {cache.misses}"

        print("  batch dedup: PASS")
