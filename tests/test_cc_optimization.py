"""Tests for connected components optimization.

Validates:
1. Immutable return types (tuple-of-tuples)
2. Cache hit/miss behavior
3. Equivalence with legacy O(P*B) algorithm
4. Performance improvement from caching
"""

import pytest
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.game.state import GameState, Pos, Component, Components
from scripts.GPU.game.rules import apply_move
from scripts.GPU.game.bridge import add_bridges_for_new_peg
from scripts.GPU.ai.heuristics import find_connected_components, _get_player_adjacency


# =============================================================================
# Legacy implementation for comparison
# =============================================================================

def legacy_find_connected_components(state: GameState, player: str) -> List[List[Pos]]:
    """Original O(P*B) implementation for comparison."""
    player_pegs = [(r, c) for (r, c), p in state.pegs.items() if p == player]
    if not player_pegs:
        return []

    visited: Set[Pos] = set()
    components: List[List[Pos]] = []

    for peg in player_pegs:
        if peg in visited:
            continue

        component: List[Pos] = []
        stack = [peg]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            if current not in state.pegs:
                continue
            if state.pegs[current] != player:
                continue

            visited.add(current)
            component.append(current)

            # O(B) scan per node - the inefficiency we're fixing
            for bridge in state.bridges:
                (r1, c1), (r2, c2) = bridge

                if (r1, c1) == current:
                    other = (r2, c2)
                elif (r2, c2) == current:
                    other = (r1, c1)
                else:
                    continue

                if other in state.pegs and state.pegs[other] == player:
                    if other not in visited:
                        stack.append(other)

        if component:
            components.append(component)

    return components


def components_equivalent(optimized: Components, legacy: List[List[Pos]]) -> bool:
    """Check if optimized (immutable) and legacy (mutable) results are equivalent."""
    if len(optimized) != len(legacy):
        return False

    # Convert both to sorted sets-of-frozensets for comparison
    opt_set = {frozenset(c) for c in optimized}
    leg_set = {frozenset(c) for c in legacy}

    return opt_set == leg_set


# =============================================================================
# Immutability Tests
# =============================================================================

class TestImmutability:
    """Verify immutable return types prevent cache corruption."""

    def test_return_type_is_tuple(self, game_state):
        """find_connected_components returns tuple-of-tuples."""
        # Add a peg
        game_state.pegs[(5, 5)] = "red"
        result = find_connected_components(game_state, "red")

        assert isinstance(result, tuple), "Should return tuple"
        if result:
            assert isinstance(result[0], tuple), "Components should be tuples"

    def test_cannot_modify_returned_components(self, game_state):
        """Attempting to modify returned components raises TypeError."""
        game_state.pegs[(5, 5)] = "red"
        game_state.pegs[(7, 6)] = "red"
        game_state.bridges.add(((5, 5), (7, 6)))
        game_state.invalidate_cc_cache()

        result = find_connected_components(game_state, "red")

        # Tuples are immutable
        with pytest.raises(TypeError):
            result[0] = ((0, 0),)  # type: ignore

    def test_cache_not_corrupted_by_caller(self, game_state):
        """Caller cannot corrupt cache by modifying returned value."""
        game_state.pegs[(5, 5)] = "red"
        game_state.pegs[(7, 6)] = "red"
        game_state.bridges.add(((5, 5), (7, 6)))
        game_state.invalidate_cc_cache()

        result1 = find_connected_components(game_state, "red")
        # Second call should return same cached value
        result2 = find_connected_components(game_state, "red")

        # Should be the exact same object (from cache)
        assert result1 is result2, "Cache should return same object"


# =============================================================================
# Cache Behavior Tests
# =============================================================================

class TestCacheBehavior:
    """Verify caching works correctly."""

    def test_cache_hit_on_repeated_call(self, game_state):
        """Repeated calls return cached result without recomputation."""
        game_state.pegs[(5, 5)] = "red"
        game_state.pegs[(7, 6)] = "red"
        game_state.bridges.add(((5, 5), (7, 6)))
        game_state.invalidate_cc_cache()

        initial_revision = game_state.cc_revision

        result1 = find_connected_components(game_state, "red")
        result2 = find_connected_components(game_state, "red")

        # Revision should not change
        assert game_state.cc_revision == initial_revision
        # Same object from cache
        assert result1 is result2

    def test_cache_miss_after_invalidation(self, game_state):
        """Cache invalidation forces recomputation."""
        game_state.pegs[(5, 5)] = "red"

        result1 = find_connected_components(game_state, "red")
        rev1 = game_state.cc_revision

        # Invalidate cache
        game_state.invalidate_cc_cache()
        rev2 = game_state.cc_revision

        result2 = find_connected_components(game_state, "red")

        assert rev2 > rev1, "Revision should increment"
        # Results should be equivalent but potentially different objects
        assert result1 == result2

    def test_cache_per_player(self, game_state):
        """Each player has separate cache entry."""
        game_state.pegs[(5, 5)] = "red"
        game_state.pegs[(10, 10)] = "black"
        game_state.invalidate_cc_cache()

        red_result = find_connected_components(game_state, "red")
        black_result = find_connected_components(game_state, "black")

        assert len(red_result) == 1
        assert len(black_result) == 1
        assert red_result[0][0] == (5, 5)
        assert black_result[0][0] == (10, 10)

    def test_apply_move_invalidates_cache(self, game_state):
        """apply_move invalidates the CC cache."""
        game_state.pegs[(5, 5)] = "red"
        find_connected_components(game_state, "red")

        rev_before = game_state.cc_revision
        new_state = apply_move(game_state, 10, 10)
        rev_after = new_state.cc_revision

        assert rev_after > rev_before, "apply_move should bump revision"

    def test_adjacency_cache_shared_across_players(self, game_state):
        """Adjacency cache is built once for both players."""
        game_state.pegs[(5, 5)] = "red"
        game_state.pegs[(7, 6)] = "red"
        game_state.bridges.add(((5, 5), (7, 6)))
        game_state.pegs[(10, 10)] = "black"
        game_state.pegs[(12, 11)] = "black"
        game_state.bridges.add(((10, 10), (12, 11)))
        game_state.invalidate_cc_cache()

        # First call builds adjacency
        _get_player_adjacency(game_state, "red")
        cache_after_red = game_state._adj_cache

        # Second call should use cached adjacency
        _get_player_adjacency(game_state, "black")
        cache_after_black = game_state._adj_cache

        assert cache_after_red is cache_after_black, "Same cache object"


# =============================================================================
# Equivalence Tests
# =============================================================================

class TestEquivalence:
    """Verify optimized algorithm matches legacy."""

    def test_empty_board(self, game_state):
        """Empty board returns empty components."""
        result = find_connected_components(game_state, "red")
        legacy = legacy_find_connected_components(game_state, "red")

        assert result == ()
        assert legacy == []

    def test_single_peg(self, game_state):
        """Single peg forms singleton component."""
        game_state.pegs[(5, 5)] = "red"

        result = find_connected_components(game_state, "red")
        legacy = legacy_find_connected_components(game_state, "red")

        assert len(result) == 1
        assert components_equivalent(result, legacy)

    def test_two_disconnected_pegs(self, game_state):
        """Disconnected pegs form separate components."""
        game_state.pegs[(5, 5)] = "red"
        game_state.pegs[(15, 15)] = "red"

        result = find_connected_components(game_state, "red")
        legacy = legacy_find_connected_components(game_state, "red")

        assert len(result) == 2
        assert components_equivalent(result, legacy)

    def test_bridged_pegs(self, game_state):
        """Bridged pegs form single component."""
        game_state.pegs[(5, 5)] = "red"
        game_state.pegs[(7, 6)] = "red"  # Knight move away
        game_state.bridges.add(((5, 5), (7, 6)))

        result = find_connected_components(game_state, "red")
        legacy = legacy_find_connected_components(game_state, "red")

        assert len(result) == 1
        assert len(result[0]) == 2
        assert components_equivalent(result, legacy)

    def test_chain_of_bridges(self, game_state):
        """Chain of bridges forms single component."""
        # Create a chain: (5,5) -> (7,6) -> (9,7) -> (11,8)
        positions = [(5, 5), (7, 6), (9, 7), (11, 8)]
        for pos in positions:
            game_state.pegs[pos] = "red"
        for i in range(len(positions) - 1):
            game_state.bridges.add((positions[i], positions[i + 1]))

        result = find_connected_components(game_state, "red")
        legacy = legacy_find_connected_components(game_state, "red")

        assert len(result) == 1
        assert len(result[0]) == 4
        assert components_equivalent(result, legacy)

    def test_mixed_players(self, game_state):
        """Components are correctly separated by player."""
        game_state.pegs[(5, 5)] = "red"
        game_state.pegs[(7, 6)] = "red"
        game_state.bridges.add(((5, 5), (7, 6)))

        game_state.pegs[(10, 10)] = "black"
        game_state.pegs[(12, 11)] = "black"
        game_state.bridges.add(((10, 10), (12, 11)))

        red_result = find_connected_components(game_state, "red")
        red_legacy = legacy_find_connected_components(game_state, "red")

        black_result = find_connected_components(game_state, "black")
        black_legacy = legacy_find_connected_components(game_state, "black")

        assert components_equivalent(red_result, red_legacy)
        assert components_equivalent(black_result, black_legacy)
        assert len(red_result) == 1
        assert len(black_result) == 1

    def test_complex_graph(self, game_state):
        """Complex multi-component graph with bridges."""
        # Component 1: 3 pegs connected
        game_state.pegs[(2, 2)] = "red"
        game_state.pegs[(4, 3)] = "red"
        game_state.pegs[(6, 4)] = "red"
        game_state.bridges.add(((2, 2), (4, 3)))
        game_state.bridges.add(((4, 3), (6, 4)))

        # Component 2: 2 pegs connected
        game_state.pegs[(15, 15)] = "red"
        game_state.pegs[(17, 16)] = "red"
        game_state.bridges.add(((15, 15), (17, 16)))

        # Component 3: 1 isolated peg
        game_state.pegs[(10, 10)] = "red"

        result = find_connected_components(game_state, "red")
        legacy = legacy_find_connected_components(game_state, "red")

        assert len(result) == 3
        assert components_equivalent(result, legacy)

        # Check component sizes
        sizes = sorted(len(c) for c in result)
        assert sizes == [1, 2, 3]

    def test_orphan_bridges_ignored(self, game_state):
        """Bridges without matching pegs are safely ignored."""
        # Add a bridge with no pegs at endpoints
        game_state.bridges.add(((5, 5), (7, 6)))

        result = find_connected_components(game_state, "red")
        assert result == ()

        # Add one peg but not the other
        game_state.pegs[(5, 5)] = "red"
        game_state.invalidate_cc_cache()

        result = find_connected_components(game_state, "red")
        assert len(result) == 1
        assert len(result[0]) == 1  # Singleton, bridge ignored

    def test_mismatched_player_bridges_ignored(self, game_state):
        """Bridges connecting different players are ignored."""
        game_state.pegs[(5, 5)] = "red"
        game_state.pegs[(7, 6)] = "black"  # Different player!
        game_state.bridges.add(((5, 5), (7, 6)))

        red_result = find_connected_components(game_state, "red")
        black_result = find_connected_components(game_state, "black")

        # Each player has one singleton component
        assert len(red_result) == 1
        assert len(red_result[0]) == 1
        assert len(black_result) == 1
        assert len(black_result[0]) == 1


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """Verify caching provides performance benefit."""

    def test_cache_faster_than_recompute(self, game_state):
        """Cached calls are faster than initial computation."""
        # Build a moderately complex state
        for i in range(10):
            row, col = 2 + i * 2, 2 + i
            game_state.pegs[(row, col)] = "red"
            if i > 0:
                prev_row, prev_col = 2 + (i - 1) * 2, 2 + (i - 1)
                game_state.bridges.add(((prev_row, prev_col), (row, col)))

        game_state.invalidate_cc_cache()

        # Time first call (builds cache)
        start = time.perf_counter()
        for _ in range(100):
            game_state.invalidate_cc_cache()
            find_connected_components(game_state, "red")
        uncached_time = time.perf_counter() - start

        # Time cached calls
        game_state.invalidate_cc_cache()
        find_connected_components(game_state, "red")  # Prime cache
        start = time.perf_counter()
        for _ in range(100):
            find_connected_components(game_state, "red")
        cached_time = time.perf_counter() - start

        assert cached_time < uncached_time / 2, (
            f"Cached should be >2x faster: cached={cached_time:.4f}s, "
            f"uncached={uncached_time:.4f}s"
        )


# =============================================================================
# Opponent CC Invariance Tests
# =============================================================================

class TestOpponentCCInvariance:
    """Verify opponent components don't change when we make a move."""

    def test_opponent_cc_unchanged_after_our_move(self, game_state):
        """Opponent's connected components are invariant under our move."""
        # Set up: black has a chain of pegs
        game_state.pegs[(10, 10)] = "black"
        game_state.pegs[(12, 11)] = "black"
        game_state.pegs[(14, 12)] = "black"
        game_state.bridges.add(((10, 10), (12, 11)))
        game_state.bridges.add(((12, 11), (14, 12)))

        # Red has one peg
        game_state.pegs[(5, 5)] = "red"
        game_state.invalidate_cc_cache()

        # Get black's components BEFORE red's move
        black_cc_before = find_connected_components(game_state, "black")

        # Red makes a move
        child_state = apply_move(game_state, 7, 6)

        # Get black's components AFTER red's move
        black_cc_after = find_connected_components(child_state, "black")

        # They should be identical
        assert black_cc_before == black_cc_after, (
            f"Opponent CC changed: before={black_cc_before}, after={black_cc_after}"
        )

    def test_opponent_cc_unchanged_multiple_moves(self, game_state):
        """Opponent CC stays same across multiple of our moves."""
        # Black has two separate components
        game_state.pegs[(10, 10)] = "black"
        game_state.pegs[(12, 11)] = "black"
        game_state.bridges.add(((10, 10), (12, 11)))

        game_state.pegs[(18, 18)] = "black"  # Isolated peg

        game_state.invalidate_cc_cache()
        black_cc_original = find_connected_components(game_state, "black")

        # Red makes several moves
        state = game_state
        red_moves = [(5, 5), (7, 6), (9, 7), (11, 8)]
        for r, c in red_moves:
            state = apply_move(state, r, c)
            # Skip black's turn for this test
            state = GameState(
                board_size=state.board_size,
                to_move="red",
                pegs=dict(state.pegs),
                bridges=set(state.bridges),
                move_history=list(state.move_history),
            )

        black_cc_final = find_connected_components(state, "black")
        assert black_cc_original == black_cc_final

    def test_our_cc_does_change(self, game_state):
        """Sanity check: our own CC DOES change when we add a peg."""
        game_state.pegs[(5, 5)] = "red"
        game_state.invalidate_cc_cache()

        red_cc_before = find_connected_components(game_state, "red")
        assert len(red_cc_before) == 1
        assert len(red_cc_before[0]) == 1

        # Red makes a bridging move
        child_state = apply_move(game_state, 7, 6)
        red_cc_after = find_connected_components(child_state, "red")

        # Our CC should now have 2 pegs in one component
        assert len(red_cc_after) == 1
        assert len(red_cc_after[0]) == 2
