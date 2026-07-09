import random
from scripts.GPU.alphazero.mcts import MCTSNode, encode_move
from scripts.GPU.alphazero.diagnose_v15_a_continuation_concentration import (
    per_child_metrics, classify_concentration)


def _root_with_children(specs):
    # specs: list of (move_rc, visit_count, q_value). Builds a root whose
    # invariants (visit_count sum, value_sum = q*visits) match the real backup.
    root = MCTSNode(state=None)
    for (rc, vc, q) in specs:
        ch = MCTSNode(state=None, parent=root, move=encode_move(*rc),
                      visit_count=vc, value_sum=q * vc)
        root.children[ch.move] = ch
    root.visit_count = sum(vc for _, vc, _ in specs)
    root.value_sum = sum(-q * vc for _, vc, q in specs)   # single sign flip child->root
    return root


def test_contributions_sum_to_root_q():
    root = _root_with_children([((1, 1), 300, -0.9), ((2, 2), 60, 0.2), ((3, 3), 40, 0.1)])
    m = per_child_metrics(root)
    assert abs(sum(c["child_contribution_share"] for c in m) - root.q_value) < 1e-9
    assert abs(sum(c["visit_share"] for c in m) - 1.0) < 1e-9


def test_visit_share_and_contribution_values():
    # child (1,1): 300/400 visits, q=-0.9 -> contribution = 0.75 * 0.9 = +0.675 (root perspective)
    root = _root_with_children([((1, 1), 300, -0.9), ((2, 2), 100, 0.0)])
    m = {tuple(c["move"]): c for c in per_child_metrics(root)}
    assert abs(m[(1, 1)]["visit_share"] - 0.75) < 1e-9
    assert abs(m[(1, 1)]["child_contribution_share"] - 0.675) < 1e-9


def test_positive_contribution_share_normalizes_positive_mass():
    # one optimistic child (+contribution), one defensive (-contribution)
    root = _root_with_children([((1, 1), 300, -0.9), ((2, 2), 100, +0.5)])
    m = {tuple(c["move"]): c for c in per_child_metrics(root)}
    assert m[(2, 2)]["child_contribution_share"] < 0                      # defensive
    assert m[(2, 2)]["positive_contribution_share"] == 0.0               # -> 0 positive mass
    assert abs(m[(1, 1)]["positive_contribution_share"] - 1.0) < 1e-9    # carries all positive mass
    assert abs(sum(c["positive_contribution_share"] for c in m.values()) - 1.0) < 1e-9


def test_classify_concentrated():
    # top child carries ~96% of the positive backup mass
    root = _root_with_children([((1, 1), 380, -0.9), ((2, 2), 20, -0.1)])
    label, share = classify_concentration(per_child_metrics(root))
    assert label == "concentrated" and share >= 0.70


def test_classify_broad():
    # positive backup spread across many similar children
    specs = [((r, r), 40, -0.2) for r in range(1, 11)]     # 10 children, equal
    label, share = classify_concentration(per_child_metrics(_root_with_children(specs)))
    assert label == "broad" and share < 0.40
