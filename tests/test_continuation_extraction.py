import pytest

from scripts.GPU.alphazero.continuation_extraction import (
    ContinuationSpec, case_path_token, continuation_case_id,
    extract_continuations, format_path_moves, path_moves_of,
    root_max_visit_share)
from scripts.GPU.alphazero.goal_line_trigger_probe_cases import position_state
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move
from tests.goal_line_probe_fixtures import legal_replay


def _child(parent, move_rc, visits, nn_value=0.1, expanded=True):
    node = MCTSNode(state=parent.state.apply_move(move_rc), parent=parent,
                    move=encode_move(*move_rc), visit_count=visits,
                    nn_value=nn_value if expanded else None,
                    priors={} if expanded else None)
    parent.children[node.move] = node
    return node


def _root():
    state = position_state(legal_replay(9, game_idx=1), 5, "black")
    return MCTSNode(state=state, visit_count=400, priors={})


def _tree_sharp():
    """PV chain a > b > c plus a small sibling."""
    root = _root()
    legal = root.state.legal_moves()
    a = _child(root, legal[0], 300, nn_value=-0.4)
    _child(root, legal[1], 100, nn_value=0.2)
    b = _child(a, a.state.legal_moves()[0], 200, nn_value=0.3)
    c = _child(b, b.state.legal_moves()[0], 120, nn_value=-0.2)
    _child(c, c.state.legal_moves()[0], 60, nn_value=0.0)
    return root, legal


def test_c_family_pv_depth_3():
    root, legal = _tree_sharp()
    specs = extract_continuations(root, "old_post_opening_retention")
    assert [s.source for s in specs] == ["pv", "pv", "pv"]
    assert [s.depth for s in specs] == [1, 2, 3]
    assert specs[0].path_moves == (legal[0],)
    assert specs[0].tree_visits == 300
    assert specs[0].tree_nn_value == pytest.approx(-0.4)
    assert len(specs[1].path_moves) == 2 and len(specs[2].path_moves) == 3
    # states are the node states (side alternates from black root)
    assert specs[0].state.to_move == "red"
    assert specs[1].state.to_move == "black"


def test_b_family_pv_depth_2():
    root, _ = _tree_sharp()
    specs = extract_continuations(root, "goal_line_retention")
    assert [s.depth for s in specs] == [1, 2]


def test_d_family_top_k_and_gated_child_pv():
    root = _root()
    legal = root.state.legal_moves()
    c1 = _child(root, legal[0], 150, nn_value=0.1)   # >= 40 -> child_pv allowed
    c2 = _child(root, legal[1], 30, nn_value=0.2)    # < 40  -> no child_pv
    c3 = _child(root, legal[2], 20, nn_value=0.3)
    _child(root, legal[3], 5, nn_value=0.4)          # rank 4 -> not in top-3
    g = _child(c1, c1.state.legal_moves()[0], 90, nn_value=-0.6)
    _child(c2, c2.state.legal_moves()[0], 25, nn_value=0.0)
    specs = extract_continuations(root, "red_predrop_retention")
    by_source = {}
    for s in specs:
        by_source.setdefault(s.source, []).append(s)
    assert len(by_source["top_child"]) == 3
    assert [s.tree_visits for s in by_source["top_child"]] == [150, 30, 20]
    assert len(by_source["child_pv"]) == 1            # only under the 150-visit child
    assert by_source["child_pv"][0].tree_visits == 90
    assert by_source["child_pv"][0].depth == 2


def test_unexpanded_and_terminal_nodes_are_skipped():
    root, legal = _tree_sharp()
    # deepest child unexpanded -> PV stops at depth reached so far
    deep = root.children[encode_move(*legal[0])]
    for _ in range(2):
        deep = max(deep.children.values(), key=lambda n: n.visit_count)
    deep.children.clear()
    deep.priors = None      # unexpanded
    deep.nn_value = None
    specs = extract_continuations(root, "old_post_opening_retention")
    assert [s.depth for s in specs] == [1, 2]         # depth-3 skipped


def test_max_per_root_hard_fails():
    root = _root()
    legal = root.state.legal_moves()
    for i in range(4):
        c = _child(root, legal[i], 100 - i, nn_value=0.0)
        _child(c, c.state.legal_moves()[0], 50, nn_value=0.0)
    with pytest.raises(ValueError, match="max_per_root"):
        extract_continuations(root, "red_predrop_retention",
                              d_top_k=4, max_per_root=6)


def test_path_helpers_and_case_id():
    root, legal = _tree_sharp()
    a = root.children[encode_move(*legal[0])]
    b = max(a.children.values(), key=lambda n: n.visit_count)
    path = path_moves_of(b)
    assert path[0] == legal[0] and len(path) == 2
    (r1, c1), (r2, c2) = path
    assert format_path_moves(path) == f"{r1}:{c1}>{r2}:{c2}"
    assert case_path_token(path) == f"{r1}-{c1}_{r2}-{c2}"
    spec = ContinuationSpec(path_moves=path, source="pv", depth=2,
                            tree_visits=b.visit_count, tree_nn_value=b.nn_value,
                            state=b.state)
    assert continuation_case_id("game_000433_ply_029", spec) == (
        f"game_000433_ply_029__cont_pv2_{case_path_token(path)}")


def test_root_max_visit_share():
    root, _ = _tree_sharp()
    assert root_max_visit_share(root) == pytest.approx(300 / 400)


def test_unknown_tag_raises():
    root, _ = _tree_sharp()
    with pytest.raises(ValueError, match="tag"):
        extract_continuations(root, "black_predrop_correction")


def test_module_has_no_mlx_import():
    import scripts.GPU.alphazero.continuation_extraction as m
    src = open(m.__file__).read()
    assert "import mlx" not in src
