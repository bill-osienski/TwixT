"""Atlas Stage 4, Task 0 -- producer closure.

Freezes what the pure analyses consume: the two-point D3 backup accounting, the
captures taken while their states exist, the online K(n+14) counters, and
amendment 4's two deep reference lines plus complete parent-visit maps.

CPU-only: FakeEvaluator at active_size=6. No reservoir, no checkpoint, no MLX.
"""
import random

import pytest

from scripts.GPU.alphazero.corpus_geometry import GameMeta
from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig
from scripts.GPU.alphazero.selection_tracer import SelectionTracer
from scripts.GPU.alphazero.warm_prefix_replay import (
    BatchSafeBoundaryObserver, capture_parent_visits, capture_tree_state,
    deep_reference_line, merge_reference_lines, replay_prefix,
    run_additive_ladder,
)

from tests.eval_fakes import FakeEvaluator

BASE = 20400000
SIZE = 6


def _edge(path, move, priors):
    """A reference-line edge as `deep_reference_line` emits it."""
    return {"parent_path": path, "move": move, "depth": len(path),
            "parent_priors": priors}


class _N:
    """Stand-in reaching the WHOLE capture, not just the D3 walker.

    capture_tree_state calls visit_leader_move(root), which reads each child's
    `.move`, and reads root.priors. A stand-in without them raises
    AttributeError before D3 is ever checked.
    """
    def __init__(self, visits, kids=None, move=0, priors=None):
        self.visit_count = visits
        self.children = kids or {}
        self.move = move
        self.priors = priors if priors is not None else {
            k: 1.0 / (i + 1) for i, k in enumerate(sorted(self.children))
        }


def test_D3_counts_nodes_at_depth_exactly_three():
    """Every backup reaching depth >=3 passes through exactly ONE depth-3 node,
    so summing visits there counts each such backup once."""
    root = _N(10, {0: _N(6, {0: _N(4, {0: _N(3, move=0), 1: _N(1, move=1)},
                                    move=0)}, move=0),
                   1: _N(4, {0: _N(2, {0: _N(2, move=0)}, move=0)}, move=1)})
    acc = capture_tree_state(root)
    assert acc["D3"] == 3 + 1 + 2          # depth-3 nodes only
    assert acc["n_visited_children"] == 2
    assert acc["one_visit_children"] == 0


def test_one_visit_children_is_counted_on_the_root_only():
    root = _N(5, {0: _N(3, move=0), 1: _N(1, move=1),
                  2: _N(1, move=2), 3: _N(0, move=3)})
    acc = capture_tree_state(root)
    assert acc["one_visit_children"] == 2
    assert acc["n_visited_children"] == 3        # the 0-visit child is excluded


def test_tracer_counts_the_lagged_bound_simultaneously():
    """Section 6a: K(n+14) is PRODUCED online, never supplied by a caller."""
    t = SelectionTracer()
    parent = type("P", (), {"priors": {i: 1.0 / (i + 1) for i in range(40)},
                            "children": {}})()
    rank20 = sorted(parent.priors.items(), key=lambda kv: (-kv[1], kv[0]))[19][0]
    t.on_select_child(parent=parent, selected_move=rank20, existing_child=None,
                      depth=1, parent_completed_visits=5, root_override=False,
                      within_forced_simulation=False)
    snap = t.snapshot()["by_shape"]["c4a05"]["overall"]
    # K(5) = 9 so rank 20 is outside; K(19) = 18 so it is still outside.
    assert snap["first_touch_outside_events"] == 1
    assert "lagged_first_touch_outside_events" in snap
    assert snap["lagged_first_touch_outside_events"] == 1


def test_the_lagged_counter_is_never_larger_than_the_unlagged_one():
    """K(n+14) >= K(n), so the lagged admitted set is wider and can only
    exclude fewer events."""
    t = SelectionTracer()
    parent = type("P", (), {"priors": {i: 1.0 / (i + 1) for i in range(60)},
                            "children": {}})()
    for rank in range(1, 40):
        mv = sorted(parent.priors.items(), key=lambda kv: (-kv[1], kv[0]))[rank - 1][0]
        t.on_select_child(parent=parent, selected_move=mv, existing_child=None,
                          depth=1, parent_completed_visits=5,
                          root_override=False, within_forced_simulation=False)
    o = t.snapshot()["by_shape"]["c4a05"]["overall"]
    assert o["lagged_first_touch_outside_events"] <= o["first_touch_outside_events"]


def _history(n, size=SIZE):
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    s = TwixtState(active_size=size, to_move="red")
    out = []
    for _ in range(n):
        lm = s.legal_moves()
        if not lm:
            break
        out.append(lm[0])
        s = s.apply_move(lm[0])
    return out


def test_features_are_frozen_at_the_boundary_not_after_the_ladder():
    """The decisive one: the root is mutated to 6,400 by leg 4, so a
    post-ladder read describes a different tree entirely."""
    hist = _history(4)
    meta = GameMeta(game_id=0, seed=BASE, n_moves=len(hist), start_player="red")
    m = MCTS(FakeEvaluator(value=0.0),
             MCTSConfig(n_simulations=1, eval_batch_size=14,
                        stall_flush_sims=48, pending_virtual_visits=8),
             random.Random(BASE))
    pre = replay_prefix(m, meta, hist, target_ply=2, active_size=SIZE)
    tracer = SelectionTracer()
    m._selection_observer = tracer
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I, threshold=40,
                                    leg_B=80, tracer=tracer)
    _legs, snaps = run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                                       boundary_observer=obs,
                                       target_tracer=tracer,
                                       increments=(80, 80, 80, 80))
    fb = snaps["captures"]["at_boundary"]
    f4 = snaps["captures"]["at_400"]
    assert fb is not None and f4 is not None

    # 1. The capture is anchored to the boundary instant, exactly.
    assert fb["root_visits"] == pre.inherited_I + obs.record.N_actual

    # 2. It must not have moved when later legs ran. Re-reading the returned
    #    dict after the ladder finished proves the snapshot was taken by value.
    assert snaps["captures"]["at_boundary"]["root_visits"] == fb["root_visits"]

    # 3. At least one field must have STRICTLY changed by the end, or `<=`
    #    comparisons would pass even if both reads happened at the end.
    after = capture_tree_state(pre.root)
    assert after["root_visits"] > fb["root_visits"]
    assert after["root_visits"] == pre.inherited_I + 320       # 4 x 80


def test_the_backup_invariant_is_asserted():
    """Section 6a: 0 <= D3(boundary) - D3(start) <= N_actual. A violation means
    the accounting is wrong and the row must fail, not be recorded."""
    from scripts.GPU.alphazero.warm_prefix_replay import check_backup_invariant
    assert check_backup_invariant(d3_start=10, d3_boundary=40, n_actual=326) is True
    with pytest.raises(ValueError):
        check_backup_invariant(d3_start=40, d3_boundary=10, n_actual=326)
    with pytest.raises(ValueError):
        check_backup_invariant(d3_start=0, d3_boundary=400, n_actual=326)


# -- amendment 4: two deep lines, and complete parent-visit maps --------------

def test_parent_visits_map_covers_every_path_through_depth_two():
    """A COMPLETE map, not a lookup of the line's paths: the union of required
    edges is not known until leg 4, and this map is taken during leg 1."""
    root = _N(10, {0: _N(6, {0: _N(4, {0: _N(3, move=0)}, move=0)}, move=0),
                   1: _N(4, move=1)})
    pv = capture_parent_visits(root)
    assert pv[()] == 10                      # the root is a parent, at depth 0
    assert pv[(0,)] == 6 and pv[(1,)] == 4
    assert pv[(0, 0)] == 4
    # Depth three is below every possible parent of a two-ply edge.
    assert (0, 0, 0) not in pv


def test_a_deep_line_is_a_list_of_edges_with_parent_paths():
    """Edges, not moves: the same move id under a different parent is a
    different edge, which is what makes agreement checkable at all."""
    root = _N(10, {5: _N(8, {3: _N(6, {1: _N(4, move=1)}, move=3)}, move=5),
                   9: _N(2, move=9)})
    line = deep_reference_line(root)
    assert [(e["parent_path"], e["move"]) for e in line["edges"]] == [
        ((), 5), ((5,), 3), ((5, 3), 1)]
    assert [e["depth"] for e in line["edges"]] == [0, 1, 2]
    # Parent PRIORS ride along, because ranks need them. The parent's visit
    # count deliberately does NOT: it would be a 6,400-era number, and its
    # presence is what let revision 1 measure against the wrong horizon.
    assert line["edges"][0]["parent_priors"] == root.priors
    assert "parent_effective_visits" not in line["edges"][0]


def test_the_union_keeps_both_replies_when_the_deep_lines_disagree():
    """Neither deep rung is truth, so a disagreement retains BOTH edges."""
    root_p = {7: 0.6, 8: 0.4}
    reply_p = {1: 0.5, 2: 0.5}
    l32 = {"edges": [_edge((), 7, root_p), _edge((7,), 1, reply_p)]}
    l64 = {"edges": [_edge((), 7, root_p), _edge((7,), 2, reply_p)]}
    m = merge_reference_lines(l32, l64)
    assert [(e["parent_path"], e["move"]) for e in m["required_edges"]] == [
        ((), 7), ((7,), 1), ((7,), 2)]
    assert m["required_edges"][0]["sources"] == (3200, 6400)   # collapsed
    assert m["required_edges"][1]["sources"] == (3200,)
    assert m["agreement"]["reply"]["state"] == "disagree"


def test_agreement_is_equality_of_the_complete_edge_not_the_move_id():
    """Move id 1 sits at depth 1 in both lines, under DIFFERENT parents. Scoring
    that as agreement would overstate how much the two rungs concur."""
    root_p = {7: 0.6, 8: 0.4}
    reply_p = {1: 0.5, 2: 0.5}
    l32 = {"edges": [_edge((), 7, root_p), _edge((7,), 1, reply_p)]}
    l64 = {"edges": [_edge((), 8, root_p), _edge((8,), 1, reply_p)]}
    m = merge_reference_lines(l32, l64)
    assert m["agreement"]["root"]["state"] == "disagree"
    assert m["agreement"]["reply"]["state"] == "disagree"
    assert len(m["required_edges"]) == 4          # nothing collapsed


def test_single_line_and_absent_both_are_DIFFERENT_missingness_states():
    """Amendment 4 calls a pair present in only one line "counted separately".
    A depth neither line reached is a different fact -- collapsing the two
    would report a comparison that was never even half-available. Neither
    enters the agreement denominator."""
    root_p = {7: 1.0}
    l32 = {"edges": [_edge((), 7, root_p), _edge((7,), 1, {1: 1.0})]}
    l64 = {"edges": [_edge((), 7, root_p)]}                 # shallower line
    m = merge_reference_lines(l32, l64)
    reply = m["agreement"]["reply"]
    assert reply["state"] == "single_line"
    assert reply["in_3200"] is True and reply["in_6400"] is False
    assert m["agreement"]["root"]["state"] == "agree"
    # Neither line reached two ply at all.
    assert m["agreement"]["two_ply"]["state"] == "absent_both"


def test_priors_equality_is_asserted_where_both_rungs_captured_a_parent():
    """add_noise=False, so priors cannot change between rungs. The assertion is
    keyed on the PARENT PATH, not the edge -- two different edges can share a
    parent, which is exactly the case worth checking."""
    l32 = {"edges": [_edge((), 7, {7: 0.6, 8: 0.4})]}
    same_parent = {"edges": [_edge((), 8, {7: 0.6, 8: 0.4})]}
    merge_reference_lines(l32, same_parent)                 # different move, OK
    drifted = {"edges": [_edge((), 8, {7: 0.9, 8: 0.1})]}
    with pytest.raises(ValueError, match="priors"):
        merge_reference_lines(l32, drifted)


def test_the_ladder_freezes_both_deep_lines_and_both_parent_visit_maps():
    hist = _history(4)
    meta = GameMeta(game_id=0, seed=BASE, n_moves=len(hist), start_player="red")
    m = MCTS(FakeEvaluator(value=0.0),
             MCTSConfig(n_simulations=1, eval_batch_size=14,
                        stall_flush_sims=48, pending_virtual_visits=8),
             random.Random(BASE))
    pre = replay_prefix(m, meta, hist, target_ply=2, active_size=SIZE)
    tracer = SelectionTracer()
    m._selection_observer = tracer
    obs = BatchSafeBoundaryObserver(inherited_I=pre.inherited_I, threshold=40,
                                    leg_B=80, tracer=tracer)
    _legs, snaps = run_additive_ladder(m, pre.root, pre.inherited_I, ply=2,
                                       boundary_observer=obs,
                                       target_tracer=tracer,
                                       increments=(80, 80, 80, 80))
    assert set(snaps["reference_lines"]) == {"at_3200", "at_6400", "merged"}
    assert set(snaps["parent_visits"]) == {"at_boundary", "at_400"}

    # Each map is anchored to its own instant: the root entry must equal the
    # tree capture taken at the same moment.
    pv_b, pv_4 = snaps["parent_visits"]["at_boundary"], snaps["parent_visits"]["at_400"]
    assert pv_b[()] == snaps["captures"]["at_boundary"]["root_visits"]
    assert pv_4[()] == snaps["captures"]["at_400"]["root_visits"]
    assert pv_4[()] >= pv_b[()]        # equal only if the boundary was the tail flush
    # ...and neither moved when legs 2-4 ran. The post-ladder tree is strictly
    # larger, so `>` here cannot be satisfied by two reads at the end.
    assert capture_parent_visits(pre.root)[()] > pv_4[()]

    # Every required edge carries the priors its rank will be computed from.
    assert all(e["parent_priors"]
               for e in snaps["reference_lines"]["merged"]["required_edges"])
