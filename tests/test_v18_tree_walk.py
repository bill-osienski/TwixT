"""v18 read-only tree walker -- spec Sec 4.4 and Sec 10.1.1.

Constructed trees only: no evaluator, no GPU. `MCTSNode` needs a `state`
supporting `.is_terminal()`, which is all these tests exercise.
"""
import pytest

from scripts.GPU.alphazero.mcts import MCTSNode
from scripts.GPU.alphazero import v18_tree_walk as W


class FakeState:
    def __init__(self, terminal=False):
        self._terminal = terminal

    def is_terminal(self):
        return self._terminal


def node(nn_value=None, visits=0, value_sum=0.0, terminal=False, priors=None):
    n = MCTSNode(state=FakeState(terminal))
    n.nn_value = nn_value
    n.visit_count = visits
    n.value_sum = value_sum
    n.priors = {} if (priors is None and terminal) else (
        {0: 1.0} if priors is None else priors)
    return n


def attach(parent, child, move_id):
    parent.children[move_id] = child
    child.parent = parent
    child.move = move_id
    return child


def build_tree():
    """root -> two depth-1 children; the first carries three depth-2 leaves.

    Visit counts chosen so the terminating identity is hand-checkable:
      root 10; d1a 7, d1b 3; under d1a: leaf_hi 3, leaf_lo 2, leaf_term 1
      terminating(root)    = 10 - (7 + 3)     = 0
      terminating(d1a)     = 7  - (3 + 2 + 1) = 1
      terminating(leaf_hi) = 3  - 0           = 3

    d1a.nn_value = +0.087 (parent-to-move), so the baseline for its leaves is
    -0.087 and leaf_hi's residual is 0.793 - (-0.087) = 0.880 -- the measured A
    pattern from spec Sec 1.1.
    """
    root = node(nn_value=0.10, visits=10, value_sum=1.0)
    d1a = attach(root, node(nn_value=0.087, visits=7, value_sum=-1.0), 1)
    d1b = attach(root, node(nn_value=0.05, visits=3, value_sum=-0.2), 2)
    leaf_hi = attach(d1a, node(nn_value=0.793, visits=3, value_sum=2.379), 11)
    leaf_lo = attach(d1a, node(nn_value=0.10, visits=2, value_sum=0.20), 12)
    leaf_term = attach(d1a, node(visits=1, value_sum=1.0, terminal=True), 13)
    return root, d1a, d1b, leaf_hi, leaf_lo, leaf_term


def test_terminating_backups_identity():
    root, d1a, _d1b, leaf_hi, _lo, _t = build_tree()
    assert W.terminating_backups(root) == 0
    assert W.terminating_backups(d1a) == 1
    assert W.terminating_backups(leaf_hi) == 3


def test_depth_histogram_sums_to_root_visit_count():
    root, *_ = build_tree()
    assert sum(W.depth_terminating_histogram(root).values()) == root.visit_count


def test_eligible_pairs_exclude_terminal_leaves():
    root, d1a, _d1b, leaf_hi, leaf_lo, leaf_term = build_tree()
    leaves = [leaf for _p, leaf in W.eligible_depth2_pairs(root)]
    assert leaf_hi in leaves and leaf_lo in leaves and leaf_term not in leaves


def test_eligible_pairs_exclude_empty_prior_leaves():
    # Spec Sec 3.2: _expand_batch assigns a synthetic nn_value 0.0 to a node with
    # no legal moves; such a leaf is bypassed, not clipped.
    root, d1a, *_ = build_tree()
    synthetic = attach(d1a, node(nn_value=0.0, visits=1, priors={}), 14)
    assert synthetic not in [leaf for _p, leaf in W.eligible_depth2_pairs(root)]


def test_empty_parent_priors_raises_rather_than_skips():
    root = node(nn_value=0.0, visits=3)
    bad_parent = attach(root, node(nn_value=0.1, visits=2, priors={}), 1)
    attach(bad_parent, node(nn_value=0.5, visits=1), 11)
    with pytest.raises(ValueError):
        W.eligible_depth2_pairs(root)


def test_residual_uses_the_negated_parent_baseline():
    _root, d1a, _d1b, leaf_hi, *_ = build_tree()
    assert W.residual(d1a, leaf_hi) == pytest.approx(0.880, abs=1e-12)


def test_would_clip_is_cap_monotone():
    root, *_ = build_tree()
    # residuals present: leaf_hi 0.880, leaf_lo 0.187
    assert len(W.would_clip(root, 1.25)) == 0
    assert len(W.would_clip(root, 0.75)) == 1
    assert len(W.would_clip(root, 0.50)) == 1
    assert len(W.would_clip(root, 0.10)) == 2


def test_would_clip_is_defined_on_a_tree_that_never_clipped():
    """The point of the would_clip population: a shipped tree has zero ACTUAL
    clips, yet the counterfactual population is well defined and non-empty."""
    root, *_ = build_tree()
    assert W.would_clip(root, 0.50)


def test_leader_and_replies_include_terminals():
    root, d1a, *_ = build_tree()
    assert W.leader(root) is d1a
    # replies is the imported v17 breadth statistic: terminals INCLUDED.
    assert W.replies(d1a) == 3


def test_explored_replies_exclude_terminal_and_ineligible():
    # `MCTSNode` is a plain @dataclass, so eq=True sets __hash__ = None and the
    # type is UNHASHABLE: a set of nodes raises TypeError while building the
    # expected value, before the walker's return value is ever examined.
    # Compare identities instead -- that imposes neither hashability nor
    # ordering on the declared `list[MCTSNode]` return type.
    _root, d1a, _d1b, leaf_hi, leaf_lo, _t = build_tree()
    got = W.explored_replies(d1a)
    assert {id(n) for n in got} == {id(leaf_hi), id(leaf_lo)}


def test_follow_up_visits_counts_visits_after_first_touch():
    _root, d1a, *_ = build_tree()
    # leaf_hi 3 visits -> 2 follow-ups; leaf_lo 2 -> 1. Mean over 2 replies = 1.5
    assert W.follow_up_visits_per_explored_reply(d1a) == pytest.approx(1.5)


def test_follow_up_visits_empty_denominator_is_invalid_not_zero():
    with pytest.raises(ValueError):
        W.follow_up_visits_per_explored_reply(node(nn_value=0.1, visits=1))


def test_revisit_rate_over_would_clip_population():
    root, _d1a, _d1b, leaf_hi, *_ = build_tree()
    attach(leaf_hi, node(nn_value=-0.2, visits=1), 111)
    assert W.revisit_to_depth3_rate(root, 0.10) == pytest.approx(0.5)


def test_revisit_rate_empty_denominator_is_invalid_not_zero():
    root, *_ = build_tree()
    with pytest.raises(ValueError):
        W.revisit_to_depth3_rate(root, 1.25)


def test_sign_dominance_formula_and_zero_denominator():
    root, *_ = build_tree()
    pos, neg = W.positive_mass(root), W.negative_mass(root)
    assert pos > 0 and neg == 0
    assert W.sign_dominance(root) == pytest.approx(pos / (pos + neg))
    # A tree with no residual mass at all scores 0.0, never a ZeroDivisionError.
    flat_root = node(nn_value=0.0, visits=2)
    flat_d1 = attach(flat_root, node(nn_value=0.0, visits=1), 1)
    attach(flat_d1, node(nn_value=0.0, visits=1), 11)
    assert W.sign_dominance(flat_root) == 0.0


def test_exposed_positive_backup_mass_numerator_and_denominator():
    root, *_ = build_tree()
    num, den = W.exposed_positive_backup_mass(root, 0.50)
    # denominator sums terminating_backups * max(0, raw_leaf) over ALL eligible
    # leaves: leaf_hi 3*0.793 + leaf_lo 2*0.10 = 2.579
    assert den == pytest.approx(3 * 0.793 + 2 * 0.10, abs=1e-12)
    # numerator restricts to leaves that would clip at 0.50: leaf_hi only.
    assert num == pytest.approx(3 * 0.793, abs=1e-12)


def test_terminal_depth2_counts():
    root, *_ = build_tree()
    terminal, total = W.terminal_depth2_counts(root)
    assert terminal == 1
    assert total == 3          # leaf_hi, leaf_lo, leaf_term (all visited)


def test_walk_emits_every_documented_key_per_cap():
    root, *_ = build_tree()
    rec = W.walk(root, caps=(1.25, 0.50))
    for key in ("root_visit_count", "depth_terminating_histogram",
                "depth_ge3_backups", "depth_ge3_fraction", "leader_move",
                "replies", "explored_replies", "follow_up_visits_per_reply",
                "eligible_depth2_leaves", "positive_mass", "negative_mass",
                "sign_dominance", "terminal_depth2", "total_depth2", "per_cap"):
        assert key in rec, key
    assert set(rec["per_cap"]) == {"1.25", "0.5"}
    for cap_rec in rec["per_cap"].values():
        for key in ("would_clip_count", "clipped_amount_total", "positive_count",
                    "negative_count", "revisit_to_depth3_rate",
                    "contribution_weighted_positive_mass",
                    "exposed_positive_mass_numerator",
                    "exposed_positive_mass_denominator"):
            assert key in cap_rec, key
    # Cap 1.25 reaches no leaf on this tree, so its revisit rate has an empty
    # denominator. `walk` records null rather than fabricating 0.0 or aborting
    # the whole multi-cap record -- 0.0 would read as "clipped leaves were never
    # revisited", the opposite of "nothing was clipped".
    assert rec["per_cap"]["1.25"]["would_clip_count"] == 0
    assert rec["per_cap"]["1.25"]["revisit_to_depth3_rate"] is None


def test_walk_is_deterministic_and_json_safe():
    import json
    root, *_ = build_tree()
    a, b = W.walk(root, caps=(0.50,)), W.walk(root, caps=(0.50,))
    assert a == b
    json.dumps(a)          # no sets, no node objects


# --- Step 5: agreement with a REAL search tree ------------------------------
# Hand-built trees can lie about tree shape. This one is produced by shipped
# search: no v18 cap exists in `mcts.py`, and none is configured here.

def test_walker_agrees_with_a_real_shipped_search_tree():
    import json

    from tests.fpu_search_fixture import run_search

    n_sims = 200                                   # cfg.n_simulations
    _fp, root, _mcts = run_search(seed=7, n_sims=n_sims)

    assert sum(W.depth_terminating_histogram(root).values()) == root.visit_count
    assert root.visit_count == n_sims
    # Identity cap on a shipped tree: no leaf can clip.
    assert W.would_clip(root, 2.0) == []
    json.dumps(W.walk(root, caps=(0.50,)))
