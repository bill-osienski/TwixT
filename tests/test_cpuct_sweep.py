from scripts.GPU.alphazero.mcts import MCTSNode, encode_move
from scripts.GPU.alphazero.diagnose_cpuct_sweep import (
    gate_flags, n_visited_children, summarize)


def _child(parent, rc, visits):
    n = MCTSNode(state=None, parent=parent, move=encode_move(*rc),
                 visit_count=visits, value_sum=0.0)
    parent.children[n.move] = n
    return n


def test_gate_flags_use_the_gate_thresholds_not_zero():
    # the gate is >= 0.25 / >= 0.50, NOT > 0
    assert gate_flags(0.10) == (False, False)   # positive but under the gate
    assert gate_flags(0.25) == (True, False)    # boundary is inclusive
    assert gate_flags(0.49) == (True, False)
    assert gate_flags(0.50) == (True, True)     # boundary is inclusive
    assert gate_flags(-0.30) == (False, False)


def test_n_visited_children_counts_only_visited():
    root = MCTSNode(state=None, visit_count=10)
    _child(root, (1, 1), 7)
    _child(root, (2, 2), 3)
    _child(root, (3, 3), 0)        # created at expansion, never visited
    assert n_visited_children(root) == 2


def test_n_visited_children_is_zero_for_a_leaf():
    assert n_visited_children(MCTSNode(state=None, visit_count=1)) == 0


def test_summarize_uses_gate_thresholds_and_reports_tree_shape():
    rows = [
        {"root_mcts_black_value": 0.60, "root_n_visited_children": 4,
         "top_child_n_visited_children": 300},
        {"root_mcts_black_value": 0.30, "root_n_visited_children": 6,
         "top_child_n_visited_children": 200},
        {"root_mcts_black_value": 0.10, "root_n_visited_children": 8,
         "top_child_n_visited_children": 100},
        {"root_mcts_black_value": -0.40, "root_n_visited_children": 2,
         "top_child_n_visited_children": 400},
    ]
    s = summarize(rows)
    assert s["n"] == 4
    assert abs(s["mean_black_value"] - 0.15) < 1e-9
    assert abs(s["over_pct_ge_0_25"] - 50.0) < 1e-9      # 0.60, 0.30
    assert abs(s["severe_pct_ge_0_50"] - 25.0) < 1e-9    # 0.60 only
    assert abs(s["positive_pct_gt_0"] - 75.0) < 1e-9     # 0.60, 0.30, 0.10
    assert abs(s["min"] - (-0.40)) < 1e-9
    assert abs(s["max"] - 0.60) < 1e-9
    assert abs(s["mean_root_n_visited_children"] - 5.0) < 1e-9
    assert abs(s["mean_top_child_n_visited_children"] - 250.0) < 1e-9


def test_summarize_over_and_positive_differ():
    # the distinction the earlier ad-hoc summarizer got wrong
    rows = [{"root_mcts_black_value": 0.10, "root_n_visited_children": 1,
             "top_child_n_visited_children": 1}]
    s = summarize(rows)
    assert s["over_pct_ge_0_25"] == 0.0 and s["positive_pct_gt_0"] == 100.0


def test_summarize_boundary_values_are_inclusive():
    # exactly on both gate thresholds; `>` instead of `>=` would drop them
    rows = [{"root_mcts_black_value": 0.25, "root_n_visited_children": 1,
             "top_child_n_visited_children": 1},
            {"root_mcts_black_value": 0.50, "root_n_visited_children": 1,
             "top_child_n_visited_children": 1}]
    s = summarize(rows)
    assert s["over_pct_ge_0_25"] == 100.0     # both are >= 0.25
    assert s["severe_pct_ge_0_50"] == 50.0    # only 0.50 is >= 0.50
