"""v18 static first-order crossover analysis -- spec Sec 2.2.1."""
import pytest

from scripts.GPU.alphazero import v18_crossover as X
from tests.test_v18_tree_walk import attach, build_tree, node


def test_counterfactual_substitutes_the_initial_contribution_only():
    """Spec Sec 2.2.1: value_sum - nn_value + clipped_initial.

    leaf_hi: nn_value 0.793, visits 3, value_sum 2.379.
    Parent nn_value +0.087 -> baseline -0.087, residual 0.880.
    At cap 0.50 the clipped initial value is -0.087 + 0.50 = 0.413.
    Counterfactual sum = 2.379 - 0.793 + 0.413 = 1.999; q = 1.999 / 3.
    """
    _root, d1a, _d1b, leaf_hi, *_ = build_tree()
    assert X.counterfactual_child_q(d1a, leaf_hi, 0.50) == pytest.approx(
        1.999 / 3, abs=1e-12)


def test_counterfactual_equals_actual_q_when_the_cap_does_not_bind():
    _root, d1a, _d1b, leaf_hi, *_ = build_tree()
    assert X.counterfactual_child_q(d1a, leaf_hi, 2.0) == pytest.approx(
        leaf_hi.value_sum / leaf_hi.visit_count, abs=1e-15)


def test_synchronous_assertion_rejects_a_tree_with_the_wrong_visit_count():
    root, *_ = build_tree()          # root.visit_count == 10
    X.assert_synchronous_tree(root, 10, search_execution_mode="synchronous")
    with pytest.raises(ValueError):
        X.assert_synchronous_tree(root, 400, search_execution_mode="synchronous")


def test_synchronous_assertion_refuses_a_batched_tree_with_a_MATCHING_count():
    """The load-bearing case: identical visit count, wrong provenance.

    `search_from_root` backs up EVERY waiter on a pending leaf with the same
    expansion value (mcts.py:595-606), so a leaf's `value_sum` can hold
    k*nn_value while `counterfactual_child_q` substitutes exactly one -- the
    substitution is then wrong by (k-1)*(backup_value - nn_value). But the
    batched path still backs up one path per simulation, so
    `root.visit_count == expected_sims` holds on BOTH paths. The count proves
    the simulation BUDGET and never the provenance; only the mode does.
    """
    root, *_ = build_tree()
    with pytest.raises(ValueError):
        X.assert_synchronous_tree(root, 10, search_execution_mode="batched_waiter")


def test_synchronous_assertion_has_no_default_mode_and_rejects_unknown_modes():
    # A default of "synchronous" would reinstate exactly the hole this closes:
    # every caller that forgot the argument would silently assert the safe
    # value. The argument is required and keyword-only.
    root, *_ = build_tree()
    with pytest.raises(TypeError):
        X.assert_synchronous_tree(root, 10)
    for bad in ("Synchronous", "sync", "", None, True):
        with pytest.raises(ValueError):
            X.assert_synchronous_tree(root, 10, search_execution_mode=bad)


def test_identity_cap_predicts_no_change():
    root, *_ = build_tree()
    out = X.crossover_for_tree(root, cap=2.0, c_puct=1.5)
    assert out["predicted_reply_delta"] == 0
    assert out["predicted_reply_reduction"] == 0.0


def test_reduction_may_be_negative_for_a_negative_residual_population():
    """A large NEGATIVE residual lowers the counterfactual visited score, which
    makes unvisited replies relatively MORE attractive -- more scanning, not
    less. The plan must not clamp this away.

    The unvisited priors are load-bearing and must not be "tidied". The count
    can only move for an unvisited move whose score falls between the capped and
    the shipped best visited score:

        band            = (1.004309, 1.076532]
        unvisited score = c_puct * prior * sqrt(19 + 1) = 6.708204 * prior
        => only prior in (0.149714, 0.160480] can move the count

    So 13's prior is 0.155, and 12 absorbs the remainder to keep the priors
    summing to 1.0. With the obvious-looking {12: 0.3, 13: 0.2} BOTH unvisited
    moves already outscore the shipped best (2.012461 and 1.341641), the counts
    are 2 and 2, and the assertion `0.0 < 0.0` is unreachable by construction --
    the mechanism is real but the fixture cannot express it.
    """
    root = node(nn_value=0.0, visits=20, value_sum=0.0)
    d1 = attach(root, node(nn_value=-0.9, visits=19, value_sum=1.0,
                           priors={11: 0.5, 12: 0.345, 13: 0.155}), 1)
    # leaf raw -0.9 against baseline +0.9 -> residual -1.8, binds hard at 0.50.
    attach(d1, node(nn_value=-0.9, visits=18, value_sum=-16.2), 11)
    out = X.crossover_for_tree(root, cap=0.50, c_puct=1.5)
    assert out["predicted_reply_reduction"] < 0.0
    # Pin the exact outcome: a sign flip or any clamp fails HERE, rather than
    # silently degrading to the vacuous 0.0 == 0.0 the original fixture gave.
    assert out["predicted_shipped_replies"] == 1
    assert out["predicted_capped_replies"] == 2
    assert out["predicted_reply_delta"] == -1
    assert out["predicted_reply_reduction"] == -1.0


def test_reduction_is_the_documented_ratio():
    root, *_ = build_tree()
    out = X.crossover_for_tree(root, cap=0.50, c_puct=1.5)
    s, c = out["predicted_shipped_replies"], out["predicted_capped_replies"]
    assert out["predicted_reply_delta"] == s - c
    assert out["predicted_reply_reduction"] == pytest.approx((s - c) / s)


def test_zero_shipped_denominator_is_invalid_not_zero():
    lone = node(nn_value=0.0, visits=1)
    with pytest.raises(ValueError):
        X.crossover_for_tree(lone, cap=0.50, c_puct=1.5)


def test_crossover_excludes_terminal_and_synthetic_children():
    root, d1a, *_ = build_tree()
    attach(d1a, node(nn_value=0.0, visits=1, priors={}), 14)
    out = X.crossover_at_node(d1a, cap=0.50, c_puct=1.5)
    assert out["excluded_terminal"] >= 1
    assert out["excluded_synthetic"] >= 1


def test_crossover_is_deterministic():
    root, *_ = build_tree()
    assert (X.crossover_for_tree(root, cap=0.75, c_puct=1.5)
            == X.crossover_for_tree(root, cap=0.75, c_puct=1.5))
