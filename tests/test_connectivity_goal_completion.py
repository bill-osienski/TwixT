"""Tests for connectivity goal-completion helpers (spec 2026-05-03 §6).

Phase 1: pure helpers in connectivity_diagnostics.py. No callers; tests
exercise the helpers directly via fixture replay.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.game.twixt_state import TwixtState
from scripts.GPU.alphazero.connectivity_diagnostics import (
    component_goal_distances,
)


def _state_after(moves, active_size=24, start_player="red"):
    """Replay a list of (r, c) moves through TwixtState.apply_move."""
    s = TwixtState(active_size=active_size, to_move=start_player)
    for m in moves:
        s = s.apply_move(m)
    return s


def _component_of(state, peg, player):
    """Wrapper around the engine's connected-component BFS (uses state.bridges)."""
    return frozenset(state._get_connected_component(peg, player))


def test_component_goal_distances_distance_zero_already_touching():
    """Red component containing a peg on row 0 → top distance = 0."""
    s = _state_after([(0, 5), (10, 10)], active_size=24)
    comp = _component_of(s, (0, 5), "red")
    d = component_goal_distances(s, "red", comp, max_depth=3)
    assert d["top"] == 0
    assert d["bottom"] is None or d["bottom"] >= 1


def test_component_goal_distances_distance_one_via_fresh_placement_on_goal_line():
    """Red peg at (2, 5); placing a peg at (0, 4) bridges to row 0 → top distance = 1."""
    s = _state_after([(2, 5), (10, 15)], active_size=24)
    comp = _component_of(s, (2, 5), "red")
    d = component_goal_distances(s, "red", comp, max_depth=3)
    assert d["top"] == 1


def test_component_goal_distances_distance_one_via_isolated_existing_goal_line_peg_with_bridgeable_connector():
    """Red at (3, 5); isolated red peg at (0, 6); placing a peg at (1, 4) bridges
    them so the extended component reaches row 0 → top distance = 1.
    Tests A1 absorption: existing same-color pegs become usable when a fresh
    placement creates the connecting bridges per apply_move()."""
    s = _state_after([(3, 5), (15, 15), (0, 6), (20, 20)], active_size=24)
    comp = _component_of(s, (3, 5), "red")
    d = component_goal_distances(s, "red", comp, max_depth=3)
    assert d["top"] == 1


def test_component_goal_distances_distance_two_two_hop_chain():
    """Red at (4, 5) with no peg at row 0–2; needs two fresh placements to reach top.
    E.g., place P1=(2, 4) bridging from (4, 5), then P2=(0, 3) bridging from (2, 4) to row 0.
    Distance = 2."""
    s = _state_after([(4, 5), (15, 15)], active_size=24)
    comp = _component_of(s, (4, 5), "red")
    d = component_goal_distances(s, "red", comp, max_depth=3)
    assert d["top"] == 2


def test_component_goal_distances_unreachable_within_max_depth_returns_none():
    """Red at (12, 5); reaching row 0 needs 4+ placements. max_depth=3 → top=None."""
    s = _state_after([(12, 5), (10, 10)], active_size=24)
    comp = _component_of(s, (12, 5), "red")
    d = component_goal_distances(s, "red", comp, max_depth=3)
    assert d["top"] is None


def test_component_goal_distances_blocked_by_intersecting_bridge_alt_route_used():
    """Red at (2, 5); black bridge (1, 5)-(3, 4) blocks the (2,5)↔(0,4) knight-bridge
    (the bridge segments properly intersect under the engine's crossing rule), so the
    (0, 4) route does NOT connect to (2, 5) when placed. The alternative knight-bridge
    via (0, 6) is unobstructed, so distance to top = 1.

    Engine-faithful: TwixtState.apply_move() refuses to add the (2,5)-(0,4) bridge
    when the proposed segment crosses an existing one, so the BFS sees (0, 4) as a
    placement that does NOT extend the component, while (0, 6) does.
    """
    s = _state_after([(2, 5), (1, 5), (10, 10), (3, 4)], active_size=24)
    comp = _component_of(s, (2, 5), "red")
    # Sanity: the existing black bridge is in place.
    assert ((1, 5), (3, 4)) in s.bridges
    # Sanity: hypothetically placing red at (0, 4) does NOT connect to (2, 5)
    # because the new bridge would cross the existing black bridge.
    from scripts.GPU.alphazero.connectivity_diagnostics import _apply_hypothetical
    s_blocked = _apply_hypothetical(s, "red", (0, 4))
    assert (2, 5) not in s_blocked._get_connected_component((0, 4), "red")
    # And placing at (0, 6) DOES connect to (2, 5) (alternate route is open).
    s_open = _apply_hypothetical(s, "red", (0, 6))
    assert (2, 5) in s_open._get_connected_component((0, 6), "red")
    # BFS should find the alternate route.
    d = component_goal_distances(s, "red", comp, max_depth=3)
    assert d["top"] == 1


def test_component_goal_distances_skips_invalid_placements_corner_and_blocked_force_distance_two():
    """Red at (2, 1) where the only one-hop knight neighbors on row 0 are (0, 0)
    [corner — illegal for any player] and (0, 2). A black bridge (1, 2)-(2, 0)
    properly intersects the (2, 1)↔(0, 2) candidate knight-edge, so the (0, 2)
    placement no longer connects to (2, 1). Both single-hop routes to row 0 are
    therefore unavailable: BFS must skip the corner candidate (rejected by
    is_valid_placement) AND skip the blocked one (apply_move accepts the placement
    but the resulting peg's component is disjoint from cur_comp). Distance to top
    is 2 via a two-hop chain (e.g., (1, 3) → (0, 1)).

    Engine-faithful: legality and crossing checks are both delegated to
    TwixtState; the helper just respects whatever they decide.
    """
    s = _state_after([(2, 1), (1, 2), (10, 10), (2, 0)], active_size=24)
    comp = _component_of(s, (2, 1), "red")
    # Sanity: black bridge in place.
    assert ((1, 2), (2, 0)) in s.bridges
    # Sanity: the corner (0, 0) is rejected by is_valid_placement for red.
    import dataclasses
    s_red = dataclasses.replace(s, to_move="red")
    assert s_red.is_valid_placement(0, 0) is False
    assert s_red.is_valid_placement(0, 2) is True
    # Sanity: red at (0, 2) does NOT connect to (2, 1) because of bridge crossing.
    from scripts.GPU.alphazero.connectivity_diagnostics import _apply_hypothetical
    s_after_02 = _apply_hypothetical(s, "red", (0, 2))
    assert (2, 1) not in s_after_02._get_connected_component((0, 2), "red")
    # Distance is 2 via a two-hop chain.
    d = component_goal_distances(s, "red", comp, max_depth=3)
    assert d["top"] == 2
