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
    compute_goal_completion_state,
    classify_selected_conversion_move,
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


def _make_state_interleaved(active_size, reds, blacks, start_player="red"):
    """Build a state by interleaving reds and blacks (red first by default).

    Used by Phase 1 synthetic fixtures: parity-respecting placements with
    arbitrary chains and filler pegs.
    """
    s = TwixtState(active_size=active_size, to_move=start_player)
    moves = []
    n = max(len(reds), len(blacks))
    for i in range(n):
        if i < len(reds):
            moves.append(reds[i])
        if i < len(blacks):
            moves.append(blacks[i])
    for m in moves:
        s = s.apply_move(m)
    return s


# ---------- Test 8: smaller-distance picking + tie-breaks ----------
def test_compute_goal_completion_state_picks_smallest_distance_then_largest_size():
    """Two red components on a 12x12 board with KNOWN different total distances:
    Comp A (5 pegs, total=2: top=1, bottom=1) and Comp B (4 pegs, total=3: top=1,
    bottom=2). Selection rule (smallest total) picks Comp A.
    Assertion: returned component_pegs == Comp A's pegs exactly.
    """
    reds = [(1, 3), (3, 4), (5, 3), (7, 4), (9, 3),    # Comp A: total=2
            (1, 8), (3, 7), (5, 8), (7, 7)]            # Comp B: total=3
    blacks = [(1, 0), (2, 0), (3, 0), (4, 0),
              (1, 11), (2, 11), (3, 11), (4, 11)]
    s = _make_state_interleaved(12, reds, blacks)
    res = compute_goal_completion_state(s, "red", max_depth=3, min_component_size=2)
    assert res is not None
    assert set(res["component_pegs"]) == {(1, 3), (3, 4), (5, 3), (7, 4), (9, 3)}
    assert res["total_goal_distance"] == 2


# ---------- Test 9: below min_component_size returns None ----------
def test_compute_goal_completion_state_returns_none_below_min_component_size():
    """Single isolated red peg (component size 1) with default min_component_size=8.
    Helper must return None.
    """
    s = _state_after([(5, 5), (2, 2)], active_size=10)
    res = compute_goal_completion_state(s, "red", max_depth=3, min_component_size=8)
    assert res is None


# ---------- Test 10: exact endpoint_completion_moves set ----------
def test_compute_goal_completion_state_endpoint_completion_moves_exact_set():
    """Reuse the Comp A fixture (12x12 chain (1,3)(3,4)(5,3)(7,4)(9,3); top=1,
    bottom=1; two_endpoint_closeout_2ply). Hand-verified completion moves are
    exactly {(0,1),(0,5),(11,2),(11,4)}: knight neighbors of (1,3) on row 0 are
    (0,1) and (0,5); knight neighbors of (9,3) on row 11 are (11,2) and (11,4).
    """
    reds = [(1, 3), (3, 4), (5, 3), (7, 4), (9, 3),
            (1, 8), (3, 7), (5, 8), (7, 7)]
    blacks = [(1, 0), (2, 0), (3, 0), (4, 0),
              (1, 11), (2, 11), (3, 11), (4, 11)]
    s = _make_state_interleaved(12, reds, blacks)
    res = compute_goal_completion_state(s, "red", max_depth=3, min_component_size=2)
    assert res is not None
    assert set(map(tuple, res["endpoint_completion_moves"])) == {
        (0, 1), (0, 5), (11, 2), (11, 4)
    }


# ---------- Test 11: completion ⊆ reducing invariant ----------
def test_compute_goal_completion_state_distance_reducing_is_superset_of_endpoint_completion():
    """Spec invariant: every completion move strictly reduces total_goal_distance.
    Use a fixture where BOTH sets are non-empty (the Comp A two_endpoint fixture).
    Assert completion ⊆ reducing.
    """
    reds = [(1, 3), (3, 4), (5, 3), (7, 4), (9, 3),
            (1, 8), (3, 7), (5, 8), (7, 7)]
    blacks = [(1, 0), (2, 0), (3, 0), (4, 0),
              (1, 11), (2, 11), (3, 11), (4, 11)]
    s = _make_state_interleaved(12, reds, blacks)
    res = compute_goal_completion_state(s, "red", max_depth=3, min_component_size=2)
    assert res is not None
    completion = set(map(tuple, res["endpoint_completion_moves"]))
    reducing = set(map(tuple, res["distance_reducing_moves"]))
    assert completion, "completion set must be non-empty for this invariant to mean something"
    assert reducing, "reducing set must be non-empty for this invariant to mean something"
    assert completion <= reducing, (
        f"completion not subset of reducing: extra={completion - reducing}"
    )


# ---------- Test 12: each category fires on a curated fixture ----------
def test_compute_goal_completion_state_categories_partition_correctly():
    """One fixture per category from spec §6.2.

    - already_won: 7-peg chain (0,2)..(11,1) touches both top and bottom.
    - one_move_win: 6-peg chain (0,2)..(10,3); top=0, bottom=1, total=1.
    - two_endpoint_closeout_2ply: Comp A from earlier — total=2, [1,1].
    - one_endpoint_distance_2: 5-peg chain (0,2)..(8,2); top=0, bottom=2.
    - broader_conversion: chain (1,8)..(7,7); top=1, bottom=2, total=3.
    """
    # already_won: 7 reds + 6 blacks for parity (red plays first)
    reds_a = [(0, 2), (2, 3), (4, 2), (6, 3), (8, 2), (10, 3), (11, 1)]
    blacks_a = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
    s_a = _make_state_interleaved(12, reds_a, blacks_a)
    res_a = compute_goal_completion_state(s_a, "red", max_depth=3, min_component_size=2)
    assert res_a is not None
    assert res_a["category"] == "already_won"
    assert res_a["total_goal_distance"] == 0

    # one_move_win
    reds_b = [(0, 2), (2, 3), (4, 2), (6, 3), (8, 2), (10, 3)]
    blacks_b = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    s_b = _make_state_interleaved(12, reds_b, blacks_b)
    res_b = compute_goal_completion_state(s_b, "red", max_depth=3, min_component_size=2)
    assert res_b is not None
    assert res_b["category"] == "one_move_win"
    assert res_b["total_goal_distance"] == 1

    # two_endpoint_closeout_2ply
    reds_c = [(1, 3), (3, 4), (5, 3), (7, 4), (9, 3),
              (1, 8), (3, 7), (5, 8), (7, 7)]
    blacks_c = [(1, 0), (2, 0), (3, 0), (4, 0),
                (1, 11), (2, 11), (3, 11), (4, 11)]
    s_c = _make_state_interleaved(12, reds_c, blacks_c)
    res_c = compute_goal_completion_state(s_c, "red", max_depth=3, min_component_size=2)
    assert res_c is not None
    assert res_c["category"] == "two_endpoint_closeout_2ply"
    assert sorted(res_c["endpoint_distances"].values()) == [1, 1]

    # one_endpoint_distance_2
    reds_d = [(0, 2), (2, 3), (4, 2), (6, 3), (8, 2)]
    blacks_d = [(1, 0), (2, 0), (3, 0), (4, 0)]
    s_d = _make_state_interleaved(12, reds_d, blacks_d)
    res_d = compute_goal_completion_state(s_d, "red", max_depth=3, min_component_size=2)
    assert res_d is not None
    assert res_d["category"] == "one_endpoint_distance_2"
    assert sorted(res_d["endpoint_distances"].values()) == [0, 2]

    # broader_conversion
    reds_e = [(1, 8), (3, 7), (5, 8), (7, 7)]
    blacks_e = [(1, 0), (2, 0), (3, 0)]
    s_e = _make_state_interleaved(12, reds_e, blacks_e)
    res_e = compute_goal_completion_state(s_e, "red", max_depth=3, min_component_size=2)
    assert res_e is not None
    assert res_e["category"] == "broader_conversion"
    assert res_e["total_goal_distance"] == 3


# ---------- Test 13: completes AND reduces → primary=completes ----------
def test_classify_selected_completes_and_reduces_both_true_primary_class_is_completes():
    """Comp A two_endpoint fixture; selected (0,1) is a completion move.
    Raw flags: completes=True, reduces=True. primary_class resolves to 'completes_endpoint'.
    """
    reds = [(1, 3), (3, 4), (5, 3), (7, 4), (9, 3),
            (1, 8), (3, 7), (5, 8), (7, 7)]
    blacks = [(1, 0), (2, 0), (3, 0), (4, 0),
              (1, 11), (2, 11), (3, 11), (4, 11)]
    s = _make_state_interleaved(12, reds, blacks)
    gs = compute_goal_completion_state(s, "red", max_depth=3, min_component_size=2)
    assert gs is not None
    res = classify_selected_conversion_move(s, "red", (0, 1), gs, max_depth=3)
    assert res["completes_endpoint"] is True
    assert res["reduces_total_goal_distance"] is True
    assert res["primary_class"] == "completes_endpoint"


# ---------- Test 14: reduces only (not completes) → primary=reduces ----------
def test_classify_selected_reduces_distance_only_primary_class_is_reduces():
    """broader_conversion fixture (top=1,bottom=2). The bottom-side reducing
    moves (9,6)/(9,8) drop bottom 2→1 but don't drop any endpoint to 0, so
    they're in reducing but not completion.
    """
    reds = [(1, 8), (3, 7), (5, 8), (7, 7)]
    blacks = [(1, 0), (2, 0), (3, 0)]
    s = _make_state_interleaved(12, reds, blacks)
    gs = compute_goal_completion_state(s, "red", max_depth=3, min_component_size=2)
    assert gs is not None
    completion = set(map(tuple, gs["endpoint_completion_moves"]))
    reducing = set(map(tuple, gs["distance_reducing_moves"]))
    candidates = reducing - completion
    assert candidates, "fixture must have at least one reducing-but-not-completion move"
    selected = sorted(candidates)[0]
    res = classify_selected_conversion_move(s, "red", selected, gs, max_depth=3)
    assert res["completes_endpoint"] is False
    assert res["reduces_total_goal_distance"] is True
    assert res["primary_class"] == "reduces_total_goal_distance"


# ---------- Test 15: redundant_reinforcement ----------
def test_classify_selected_redundant_reinforcement_bridgeable_to_component_no_distance_reduction():
    """one_move_win fixture (top=0, bottom=1). Selected (5,4) bridges into the
    chain via (4,2), but doesn't reduce total. primary_class='redundant_reinforcement'.
    """
    reds = [(0, 2), (2, 3), (4, 2), (6, 3), (8, 2), (10, 3)]
    blacks = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
    s = _make_state_interleaved(12, reds, blacks)
    assert s.to_move == "red"
    gs = compute_goal_completion_state(s, "red", max_depth=3, min_component_size=2)
    assert gs is not None
    res = classify_selected_conversion_move(s, "red", (5, 4), gs, max_depth=3)
    assert res["is_redundant_reinforcement"] is True
    assert res["reduces_total_goal_distance"] is False
    assert res["completes_endpoint"] is False
    assert res["primary_class"] == "redundant_reinforcement"


# ---------- Test 16: off_chain ----------
def test_classify_selected_off_chain_when_no_knight_neighbor_in_extended_component():
    """one_move_win fixture; selected (5,9) is far from chain — no knight
    neighbor in component AND not in reducing. primary_class='off_chain'.
    """
    reds = [(0, 2), (2, 3), (4, 2), (6, 3), (8, 2), (10, 3)]
    blacks = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
    s = _make_state_interleaved(12, reds, blacks)
    gs = compute_goal_completion_state(s, "red", max_depth=3, min_component_size=2)
    assert gs is not None
    # Sanity: (5,9) has no knight neighbor in the chain.
    knight = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
    chain = gs["component_pegs"]
    assert not any((5 + dr, 9 + dc) in chain for dr, dc in knight)
    res = classify_selected_conversion_move(s, "red", (5, 9), gs, max_depth=3)
    assert res["is_off_chain"] is True
    assert res["primary_class"] == "off_chain"


# ---------- Test 18: ambient bridge absorption guard ----------
def test_existing_same_color_goal_peg_requires_actual_or_new_bridge_connection():
    """An isolated red goal-line peg (0, 6) and a chain in mid-board must remain
    in DIFFERENT connected components when no bridge connects them. Only a fresh
    placement that bridges to BOTH may absorb the goal peg into the chain's
    component (per the engine's bridge-formation semantics).

    Guards against a future bug where compute_goal_completion_state might be
    tempted to treat any same-color peg as part of the dominant component
    "for free."
    """
    reds = [(3, 5), (5, 4), (7, 5), (9, 4), (0, 6)]
    blacks = [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
    s = _make_state_interleaved(12, reds, blacks)
    chain_comp = s._get_connected_component((3, 5), "red")
    lone_comp = s._get_connected_component((0, 6), "red")
    assert set(chain_comp).isdisjoint(lone_comp)
    gs = compute_goal_completion_state(s, "red", max_depth=3, min_component_size=2)
    assert gs is not None
    assert (0, 6) not in gs["component_pegs"]
    assert set(gs["component_pegs"]) == {(3, 5), (5, 4), (7, 5), (9, 4)}
    # And: a fresh placement at (1, 4) DOES bridge both the chain peg (3,5) and
    # the lone goal peg (0,6) — the only legitimate way for absorption to occur.
    from scripts.GPU.alphazero.connectivity_diagnostics import _apply_hypothetical
    s_after = _apply_hypothetical(s, "red", (1, 4))
    new_comp = s_after._get_connected_component((1, 4), "red")
    assert (3, 5) in new_comp and (0, 6) in new_comp


# ---------- Test 19: primary_class='other' ----------
def test_classify_selected_primary_class_other_for_adjacent_nonreducing_nonredundant_move():
    """Move with knight neighbor in component but its bridge is BLOCKED by an
    opposing bridge. Result: cand's new_comp is disjoint from chain (not bridgeable
    → not redundant), AND has knight neighbor in chain (not off_chain), AND no
    reduction. primary_class='other'.

    Layout: red chain (0,2)(2,3)(4,2)(6,3)(8,2)(10,3) — one_move_win shape.
    Black bridge (4,1)-(6,2) crosses red's would-be (5,1)-(6,3) bridge. So when
    red plays (5,1), the bridge to (6,3) doesn't form. (5,1)'s knight neighbor
    (6,3) is in the chain (so not off_chain), but (5,1) ends up in its own
    component (not bridgeable_to_component → not redundant), no reduction.
    """
    reds = [(0, 2), (2, 3), (4, 2), (6, 3), (8, 2), (10, 3)]
    blacks = [(4, 1), (6, 2), (1, 5), (2, 5), (3, 5), (7, 5)]
    s = _make_state_interleaved(12, reds, blacks)
    assert s.to_move == "red"
    # Sanity: red chain still intact (black bridge doesn't cross any red bridge).
    chain = s._get_connected_component((0, 2), "red")
    assert set(chain) == {(0, 2), (2, 3), (4, 2), (6, 3), (8, 2), (10, 3)}
    # Sanity: placing red at (5,1) — bridge to (6,3) is blocked.
    from scripts.GPU.alphazero.connectivity_diagnostics import _apply_hypothetical
    s_after = _apply_hypothetical(s, "red", (5, 1))
    new_comp = s_after._get_connected_component((5, 1), "red")
    assert (6, 3) not in new_comp, "if this fails, the blocking bridge isn't actually blocking"
    gs = compute_goal_completion_state(s, "red", max_depth=3, min_component_size=2)
    assert gs is not None
    res = classify_selected_conversion_move(s, "red", (5, 1), gs, max_depth=3)
    assert res["completes_endpoint"] is False
    assert res["reduces_total_goal_distance"] is False
    assert res["is_redundant_reinforcement"] is False
    assert res["is_off_chain"] is False  # (5,1) has knight-neighbor (6,3) in chain
    assert res["primary_class"] == "other"


def test_compute_goal_completion_state_game097_turn43_canonical():
    """Spec anchor: replay first 43 moves of iter_0108_game_097 (NOT 35 — the
    chain doesn't enter two_endpoint_closeout_2ply shape until Red plays (22, 4)
    at turn 43 to set up the bottom hop).

    At turn 43 the chain has top distance 1 (from (1,6) → (0,4) or (0,8)) AND
    bottom distance 1 (from (22,4) → (23,2) or (23,6)). Both endpoints are
    one fresh placement away. (0, 8) and (23, 6) are both completion moves
    under the spec §6.1 prose definition.
    """
    import json
    from pathlib import Path

    games_dir = Path(__file__).parent.parent / "scripts" / "GPU" / "logs" / "games"
    candidates = list(games_dir.glob("iter_0108_game_097*"))
    if not candidates:
        import pytest
        pytest.skip(
            "Game 097 anchor replay not present; synthetic Phase 1 tests still "
            "cover helper behavior. A green test run does NOT imply the canonical "
            "Game 097 closeout was validated."
        )
    record = json.loads(candidates[0].read_text())
    moves = [(int(m["row"]), int(m["col"])) for m in record["moves"][:43]]
    s = _state_after(moves, active_size=24,
                     start_player=record.get("starting_player", "red"))
    res = compute_goal_completion_state(s, "red", max_depth=3, min_component_size=8)
    assert res is not None, "Red must have a dominant-unclosed component at turn 43"
    assert res["total_goal_distance"] == 2, (
        f"Expected Red total_goal_distance=2 at turn 43, got {res['total_goal_distance']}"
    )
    assert res["endpoint_distances"]["top"] == 1
    assert res["endpoint_distances"]["bottom"] == 1
    assert res["category"] == "two_endpoint_closeout_2ply"
    assert (0, 8) in res["endpoint_completion_moves"]
    assert (23, 6) in res["endpoint_completion_moves"]
