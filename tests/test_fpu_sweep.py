from scripts.GPU.alphazero.mcts import MCTSNode, encode_move
from scripts.GPU.alphazero.diagnose_fpu_sweep import (
    gate_flags, n_visited_children, summarize)


def _child(parent, rc, visits):
    n = MCTSNode(state=None, parent=parent, move=encode_move(*rc),
                 visit_count=visits, value_sum=0.0)
    parent.children[n.move] = n
    return n


def test_gate_flags_use_the_gate_thresholds_not_zero():
    assert gate_flags(0.10) == (False, False)
    assert gate_flags(0.25) == (True, False)
    assert gate_flags(0.50) == (True, True)
    assert gate_flags(-0.30) == (False, False)


def test_n_visited_children_counts_only_visited():
    root = MCTSNode(state=None, visit_count=10)
    _child(root, (1, 1), 7)
    _child(root, (2, 2), 3)
    _child(root, (3, 3), 0)
    assert n_visited_children(root) == 2


def test_summarize_uses_gate_thresholds_and_reports_tree_shape():
    rows = [
        {"root_mcts_black_value": 0.60, "root_n_visited_children": 4,
         "top_child_n_visited_children": 300, "top_child_visit_share": 0.8},
        {"root_mcts_black_value": 0.30, "root_n_visited_children": 6,
         "top_child_n_visited_children": 200, "top_child_visit_share": 0.6},
        {"root_mcts_black_value": 0.10, "root_n_visited_children": 8,
         "top_child_n_visited_children": 100, "top_child_visit_share": 0.4},
        {"root_mcts_black_value": -0.40, "root_n_visited_children": 2,
         "top_child_n_visited_children": 400, "top_child_visit_share": 0.2},
    ]
    s = summarize(rows)
    assert s["n"] == 4
    assert abs(s["mean_black_value"] - 0.15) < 1e-9
    assert abs(s["over_pct_ge_0_25"] - 50.0) < 1e-9
    assert abs(s["severe_pct_ge_0_50"] - 25.0) < 1e-9
    assert abs(s["positive_pct_gt_0"] - 75.0) < 1e-9
    assert abs(s["root_children_mean"] - 5.0) < 1e-9
    assert abs(s["top_child_children_mean"] - 250.0) < 1e-9
    assert abs(s["top_child_visit_share_mean"] - 0.5) < 1e-9
    assert abs(s["min"] - (-0.40)) < 1e-9
    assert abs(s["max"] - 0.60) < 1e-9


def test_summarize_boundary_values_are_inclusive():
    rows = [{"root_mcts_black_value": 0.25, "root_n_visited_children": 1,
             "top_child_n_visited_children": 1, "top_child_visit_share": 0.5},
            {"root_mcts_black_value": 0.50, "root_n_visited_children": 1,
             "top_child_n_visited_children": 1, "top_child_visit_share": 0.5}]
    s = summarize(rows)
    assert s["over_pct_ge_0_25"] == 100.0
    assert s["severe_pct_ge_0_50"] == 50.0
