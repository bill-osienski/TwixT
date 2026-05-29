"""Equivalence + cache-behavior tests for the _get_connected_component
adjacency optimization (spec 2026-05-29). The "legacy" reference below is a
verbatim copy of the pre-optimization full-bridge-scan algorithm; the engine's
output must match it exactly for every position in the corpus."""
import glob
import json
import os
import random
from collections import deque

import numpy as np
import pytest

from scripts.GPU.alphazero.game.twixt_state import TwixtState

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- reference: pre-optimization O(V*E) full-bridge-scan BFS ---------------
def _legacy_component(pegs, bridges, start, player):
    visited, component = set(), set()
    queue = deque([start])
    while queue:
        pos = queue.popleft()
        if pos in visited:
            continue
        if pegs.get(pos) != player:
            continue
        visited.add(pos)
        component.add(pos)
        for p1, p2 in bridges:
            if p1 == pos:
                nb = p2
            elif p2 == pos:
                nb = p1
            else:
                continue
            if pegs.get(p1) != player:  # mirrors original's check on p1
                continue
            if nb not in visited:
                queue.append(nb)
    return component


def _components_legacy(pegs, bridges, player):
    seen, comps = set(), set()
    for peg, p in pegs.items():
        if p != player or peg in seen:
            continue
        comp = _legacy_component(pegs, bridges, peg, player)
        comps.add(frozenset(comp))
        seen |= comp
    return comps


def _components_optimized(state, player):
    seen, comps = set(), set()
    for peg, p in state.pegs.items():
        if p != player or peg in seen:
            continue
        comp = state._get_connected_component(peg, player)
        comps.add(frozenset(comp))
        seen |= comp
    return comps


def _legacy_winner(state):
    active = state.active_size
    pegs, bridges = state.pegs, state.bridges
    for col in range(active):
        if pegs.get((0, col)) == "red":
            if any(r == active - 1 for (r, c) in _legacy_component(pegs, bridges, (0, col), "red")):
                return "red"
    for row in range(active):
        if pegs.get((row, 0)) == "black":
            if any(c == active - 1 for (r, c) in _legacy_component(pegs, bridges, (row, 0), "black")):
                return "black"
    return None


def _legacy_masks(state, player):
    active = state.active_size
    m_g1 = np.zeros((active, active), dtype=np.float32)
    m_g2 = np.zeros((active, active), dtype=np.float32)
    m_both = np.zeros((active, active), dtype=np.float32)
    if player == "red":
        on_g1, on_g2 = (lambda r, c: r == 0), (lambda r, c: r == active - 1)
    else:
        on_g1, on_g2 = (lambda r, c: c == 0), (lambda r, c: c == active - 1)
    for comp in _components_legacy(state.pegs, state.bridges, player):
        t1 = any(on_g1(r, c) for (r, c) in comp)
        t2 = any(on_g2(r, c) for (r, c) in comp)
        for (r, c) in comp:
            if t1:
                m_g1[r, c] = 1.0
            if t2:
                m_g2[r, c] = 1.0
            if t1 and t2:
                m_both[r, c] = 1.0
    return m_g1, m_g2, m_both


def _assert_position_equivalent(state):
    for player in ("red", "black"):
        assert _components_optimized(state, player) == _components_legacy(
            state.pegs, state.bridges, player
        ), f"component mismatch for {player} at ply {state.ply}"
        for opt, leg in zip(state.connectivity_masks(player), _legacy_masks(state, player)):
            assert np.array_equal(opt, leg), f"mask mismatch for {player} at ply {state.ply}"
    assert state.winner() == _legacy_winner(state), f"winner mismatch at ply {state.ply}"


def _random_game(seed, active_size=24, max_ply=160):
    """Play random legal moves, yielding the state after each move."""
    rng = random.Random(seed)
    state = TwixtState(active_size=active_size)
    for _ in range(max_ply):
        moves = state.legal_moves()
        if not moves:
            break
        state = state.apply_move(rng.choice(moves))
        yield state


def test_equivalence_synthetic_dense():
    for seed in (1, 2, 3):
        plies = list(_random_game(seed, active_size=24, max_ply=160))
        assert len(plies) >= 100, "synthetic game should reach a dense regime"
        for state in plies[::20] + [plies[-1]]:
            _assert_position_equivalent(state)


def test_equivalence_fixtures():
    # Empty board.
    _assert_position_equivalent(TwixtState(active_size=8))

    # Single red peg (singleton component). These fixtures construct-then-query,
    # so the cache builds lazily AFTER the mutations and needs no explicit
    # invalidation (that path is covered by test_invalidate_adj_picks_up_mutation).
    s = TwixtState(active_size=8)
    s.pegs[(3, 3)] = "red"
    _assert_position_equivalent(s)

    # Orphan bridge (endpoint without a peg) is ignored.
    s = TwixtState(active_size=8)
    s.pegs[(3, 3)] = "red"
    s.bridges.add(((3, 3), (5, 4)))  # (5,4) has no peg
    _assert_position_equivalent(s)
    assert s._get_connected_component((3, 3), "red") == {(3, 3)}

    # Cross-player bridge is ignored by both players.
    s = TwixtState(active_size=8)
    s.pegs[(3, 3)] = "red"
    s.pegs[(5, 4)] = "black"
    s.bridges.add(((3, 3), (5, 4)))
    _assert_position_equivalent(s)


def test_equivalence_real_replays():
    files = sorted(glob.glob(os.path.join(REPO_ROOT, "Replays", "**", "*.json"), recursive=True))
    if not files:
        pytest.skip("no Replays/ corpus present")
    checked = skipped = 0
    for path in files[:40]:  # bound runtime
        with open(path) as f:
            rec = json.load(f)
        moves = [(m["row"], m["col"]) for m in rec.get("moves", [])]
        if not moves:
            continue
        active = int(rec.get("meta", {}).get("board_size", 24))
        # Games may start black (mirror-prob); honor the recorded starting
        # player so apply_move's per-player legality holds during replay.
        start = rec.get("starting_player", "red")
        try:
            state = TwixtState(active_size=active, to_move=start)
            for mv in moves:
                state = state.apply_move(mv)
        except ValueError:
            skipped += 1  # a non-replayable record must not crash the suite
            continue
        _assert_position_equivalent(state)  # final (densest) position
        checked += 1
    print(f"[cc-adjacency] replays checked={checked} skipped={skipped} (capped at 40)")
    assert checked > 0, "expected at least one replayable game in Replays/"


def test_adj_is_none_on_fresh_state():
    s = TwixtState(active_size=8)
    assert s._adj is None


def test_adj_built_lazily_on_query():
    s = TwixtState(active_size=8)
    s.pegs[(3, 3)] = "red"
    assert s._adj is None  # setting a peg does not build the cache
    s._get_connected_component((3, 3), "red")
    assert isinstance(s._adj, dict)


def test_adj_not_carried_into_copy():
    s = TwixtState(active_size=8)
    s.pegs[(3, 3)] = "red"
    s._get_connected_component((3, 3), "red")  # build cache
    assert s._adj is not None
    child = s.copy()
    assert child._adj is None


def test_apply_move_child_has_fresh_cache():
    s = TwixtState(active_size=8)
    s._get_connected_component((0, 0), "red")  # build cache on parent
    child = s.apply_move((3, 3))  # red plays (3,3)
    assert child._adj is None
    assert child._get_connected_component((3, 3), "red") == {(3, 3)}


def test_invalidate_adj_picks_up_mutation():
    s = TwixtState(active_size=8)
    s.pegs[(3, 3)] = "red"
    s._get_connected_component((3, 3), "red")  # build cache (no bridges yet)
    s.pegs[(5, 4)] = "red"
    s.bridges.add(((3, 3), (5, 4)))
    s._invalidate_adj()
    assert s._get_connected_component((3, 3), "red") == {(3, 3), (5, 4)}


@pytest.mark.slow
def test_perf_smoke_dense_winner():
    """A dense position must resolve winner() quickly. With the old O(V*E)
    scan this loop would take many seconds; O(V+E) is well under the bound."""
    import time

    state = None
    for state in _random_game(7, active_size=24, max_ply=250):
        pass
    assert len(state.pegs) > 150, "expected a dense position"

    start = time.perf_counter()
    for _ in range(300):
        state._invalidate_adj()  # force rebuild + full traversal each call
        state.winner()
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0, f"dense winner() x300 took {elapsed:.2f}s (regression?)"
