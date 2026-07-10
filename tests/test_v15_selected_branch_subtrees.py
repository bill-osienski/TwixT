import types

from scripts.GPU.alphazero.mcts import MCTSNode, encode_move
from scripts.GPU.alphazero.diagnose_v15_a_selected_branch_subtrees import (
    aggregate_by_depth, group_phase0_by_root, node_metrics, pv_chain,
    select_positive_branches, walk_subtree)


def _state(to_move, terminal=False):
    return types.SimpleNamespace(to_move=to_move,
                                 is_terminal=lambda: terminal)


def _node(parent, move_rc, visits, q, to_move="red", terminal=False):
    """Child of `parent` with the real backup relationship value_sum = q*visits."""
    n = MCTSNode(state=_state(to_move, terminal), parent=parent,
                 move=encode_move(*move_rc), visit_count=visits,
                 value_sum=q * visits)
    if parent is not None:
        parent.children[n.move] = n
    return n


def _phase0_row(root_case_id, child_move, contrib, pos_share, root_black,
                child_q=-0.5):
    return {"root_case_id": root_case_id, "child_move": child_move,
            "child_contribution_share": str(contrib),
            "positive_contribution_share": str(pos_share),
            "root_mcts_black_value": str(root_black),
            "child_q_value": str(child_q),
            "root_case_classification": "concentrated"}


# ---------- selection ----------

def test_select_positive_branches_skips_nonpositive_roots():
    rows = [_phase0_row("neg", "1:1", 0.3, 1.0, -0.19),
            _phase0_row("pos", "2:2", 0.4, 1.0, +0.42)]
    picked = select_positive_branches(group_phase0_by_root(rows))
    assert [cid for cid, _ in picked] == ["pos"]


def test_select_positive_branches_stops_at_cumulative_threshold():
    rows = [_phase0_row("r", "1:1", 0.50, 0.60, +0.5),
            _phase0_row("r", "2:2", 0.30, 0.35, +0.5),   # cum 0.95 >= 0.90 -> stop
            _phase0_row("r", "3:3", 0.04, 0.05, +0.5)]
    (_cid, picked), = select_positive_branches(group_phase0_by_root(rows))
    assert [p["child_move"] for p in picked] == ["1:1", "2:2"]


def test_select_positive_branches_caps_at_max_children():
    rows = [_phase0_row("r", f"{i}:{i}", 0.2, 0.2, +0.5) for i in range(1, 6)]
    (_cid, picked), = select_positive_branches(group_phase0_by_root(rows))
    assert len(picked) == 3            # cum never reaches 0.90; cap wins


def test_select_positive_branches_skips_root_with_zero_positive_mass():
    rows = [_phase0_row("r", "1:1", -0.1, 0.0, +0.01),
            _phase0_row("r", "2:2", -0.2, 0.0, +0.01)]
    assert select_positive_branches(group_phase0_by_root(rows)) == []


# ---------- walk ----------

def test_walk_subtree_visits_only_nodes_with_visits():
    root = MCTSNode(state=_state("black"), visit_count=100, value_sum=50.0)
    branch = _node(root, (1, 1), 90, -0.5)
    kept = _node(branch, (2, 2), 40, 0.4, to_move="black")
    _dropped = _node(branch, (3, 3), 0, 0.0, to_move="black")   # never visited
    deep = _node(kept, (4, 4), 10, -0.2)
    walked = walk_subtree(branch)
    assert set(id(n) for n in walked) == {id(branch), id(kept), id(deep)}


def test_walk_subtree_includes_the_branch_root():
    root = MCTSNode(state=_state("black"), visit_count=10, value_sum=1.0)
    branch = _node(root, (1, 1), 10, -0.3)
    assert walk_subtree(branch) == [branch]


# ---------- PV ----------

def test_pv_chain_marks_only_the_best_child_chain():
    root = MCTSNode(state=_state("black"), visit_count=100, value_sum=10.0)
    branch = _node(root, (1, 1), 90, -0.5)
    best = _node(branch, (2, 2), 70, 0.4, to_move="black")
    other = _node(branch, (3, 3), 20, 0.1, to_move="black")
    deep = _node(best, (4, 4), 60, -0.3)
    chain = pv_chain(branch)
    assert chain == {id(branch): 0, id(best): 1, id(deep): 2}
    assert id(other) not in chain


# ---------- node metrics ----------

def test_node_metrics_depths_shares_and_perspective():
    root = MCTSNode(state=_state("black"), visit_count=100, value_sum=10.0)
    branch = _node(root, (1, 1), 80, -0.5)               # red to move
    deep = _node(branch, (2, 2), 40, 0.25, to_move="black")

    m = node_metrics(deep, root, branch, pv_index=1)
    assert m["depth_from_root"] == 2
    assert m["depth_from_selected_child"] == 1
    assert m["move_from_parent"] == "2:2"
    assert m["path_moves"] == "1:1 2:2"
    assert abs(m["visit_share_from_parent"] - 40 / 80) < 1e-9
    assert abs(m["visit_share_from_root"] - 40 / 100) < 1e-9
    assert abs(m["q_value_node_perspective"] - 0.25) < 1e-9
    assert abs(m["q_value_root_perspective"] - 0.25) < 1e-9   # black to move: unchanged
    assert m["is_pv_path"] is True and m["pv_depth_index"] == 1

    mb = node_metrics(branch, root, branch, pv_index=None)
    assert mb["depth_from_selected_child"] == 0
    assert abs(mb["q_value_node_perspective"] + 0.5) < 1e-9
    assert abs(mb["q_value_root_perspective"] - 0.5) < 1e-9   # red to move: flipped
    assert mb["is_pv_path"] is False and mb["pv_depth_index"] == ""


def test_node_metrics_counts_unvisited_children_and_terminal():
    root = MCTSNode(state=_state("black"), visit_count=10, value_sum=1.0)
    branch = _node(root, (1, 1), 10, -0.3)
    _node(branch, (2, 2), 5, 0.1, to_move="black")
    _node(branch, (3, 3), 0, 0.0, to_move="black")
    m = node_metrics(branch, root, branch, pv_index=0)
    assert m["num_children"] == 2 and m["unvisited_children_count"] == 1
    assert m["is_terminal"] is False

    term = _node(branch, (4, 4), 3, 1.0, to_move="black", terminal=True)
    assert node_metrics(term, root, branch, pv_index=None)["is_terminal"] is True


# ---------- aggregate ----------

def _agg_row(depth, visit_share, raw, terminal=False):
    return {"depth_from_root": depth, "visit_share_from_root": visit_share,
            "raw_black_BASE": "" if terminal else raw,
            "raw_black_v14b": "" if terminal else raw,
            "unvisited_children_count": 0}


def test_aggregate_visit_mass_beats_node_count():
    # 1 raw-positive node holding 60% of the visit mass, 3 raw-negative nodes
    # holding 40% between them: pct_raw_positive=0.25 but the DECISION metric
    # pct_visit_mass_raw_positive=0.60. This is exactly why the two differ.
    rows = [_agg_row(3, 0.60, +0.4)] + [_agg_row(3, 0.40 / 3, -0.2)] * 3
    (rec,) = aggregate_by_depth(rows, "full_subtree")
    assert rec["scope"] == "full_subtree" and rec["depth_from_root"] == 3
    assert rec["nodes_count"] == 4 and rec["raw_scored_nodes_count"] == 4
    assert abs(rec["pct_raw_positive_BASE"] - 0.25) < 1e-9
    assert abs(rec["pct_visit_mass_raw_positive_BASE"] - 0.60) < 1e-9
    assert abs(rec["max_raw_black_BASE"] - 0.4) < 1e-9
    assert abs(rec["weighted_mean_raw_black_BASE"] - (0.6 * 0.4 + 0.4 * -0.2)) < 1e-9


def test_aggregate_excludes_terminal_nodes_from_raw_stats_but_keeps_visit_mass():
    rows = [_agg_row(2, 0.5, +0.3), _agg_row(2, 0.5, None, terminal=True)]
    (rec,) = aggregate_by_depth(rows, "pv_only")
    assert rec["nodes_count"] == 2 and rec["raw_scored_nodes_count"] == 1
    assert abs(rec["total_visit_share_from_root"] - 1.0) < 1e-9
    assert abs(rec["mean_raw_black_BASE"] - 0.3) < 1e-9          # over scored only
    assert abs(rec["pct_raw_positive_BASE"] - 1.0) < 1e-9        # over scored only
    # visit mass denominator is the SCORED mass, so the terminal node cannot
    # silently dilute the decision metric
    assert abs(rec["pct_visit_mass_raw_positive_BASE"] - 1.0) < 1e-9


def test_aggregate_handles_a_depth_with_no_scored_nodes():
    rows = [_agg_row(5, 0.2, None, terminal=True)]
    (rec,) = aggregate_by_depth(rows, "full_subtree")
    assert rec["raw_scored_nodes_count"] == 0
    assert rec["mean_raw_black_BASE"] == ""
    assert rec["pct_visit_mass_raw_positive_BASE"] == ""
