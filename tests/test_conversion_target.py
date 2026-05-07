"""Per-position conversion target tests (Spec 2 §6.1)."""
import math
import numpy as np
from scripts.GPU.alphazero.conversion_loss import build_conversion_target


def test_target_normalizes_to_unit_sum():
    legal = [(0, 8), (5, 5), (22, 4)]
    completion = {(0, 8)}
    reducing = {(22, 4)}
    target = build_conversion_target(legal, completion, reducing,
                                     completion_weight=1.0, reducer_weight=0.35)
    assert target is not None
    assert math.isclose(target.sum(), 1.0, abs_tol=1e-6)


def test_target_assigns_completion_weight_to_completion_moves():
    legal = [(0, 8), (5, 5), (22, 4)]
    completion = {(0, 8)}
    reducing = {(22, 4)}
    target = build_conversion_target(legal, completion, reducing,
                                     completion_weight=1.0, reducer_weight=0.35)
    # (0,8) is completion → weight 1.0; total = 1.0 + 0.35 = 1.35
    assert math.isclose(target[0], 1.0 / 1.35, abs_tol=1e-6)


def test_target_assigns_reducer_weight_to_reducer_only_moves():
    legal = [(0, 8), (5, 5), (22, 4)]
    completion = {(0, 8)}
    reducing = {(22, 4)}
    target = build_conversion_target(legal, completion, reducing,
                                     completion_weight=1.0, reducer_weight=0.35)
    assert math.isclose(target[2], 0.35 / 1.35, abs_tol=1e-6)


def test_target_disjoint_mass_rule_completion_wins():
    """Move in BOTH sets gets completion_weight, not the sum."""
    legal = [(0, 8), (5, 5)]
    completion = {(0, 8)}
    reducing = {(0, 8)}     # same move in both sets
    target = build_conversion_target(legal, completion, reducing,
                                     completion_weight=1.0, reducer_weight=0.35)
    # Only one entry in target gets weight, normalized to 1.0
    assert math.isclose(target[0], 1.0, abs_tol=1e-6)
    assert math.isclose(target[1], 0.0, abs_tol=1e-6)


def test_target_zero_for_other_legal_moves():
    legal = [(0, 8), (5, 5), (22, 4)]
    completion = {(0, 8)}
    reducing = {(22, 4)}
    target = build_conversion_target(legal, completion, reducing,
                                     completion_weight=1.0, reducer_weight=0.35)
    assert target[1] == 0.0


def test_conversion_aux_target_aligns_with_legal_move_order():
    """ANCHOR (Spec 2 §11.3): exact alignment fixture from spec.

    legal=[(1,2),(3,4),(5,6)]
    completion=[(5,6)], reducer=[(1,2)]
    Expected target = [0.35/1.35, 0.0, 1.0/1.35]
    """
    legal = [(1, 2), (3, 4), (5, 6)]
    completion = {(5, 6)}
    reducing = {(1, 2)}
    target = build_conversion_target(legal, completion, reducing,
                                     completion_weight=1.0, reducer_weight=0.35)
    assert target is not None
    np.testing.assert_allclose(
        target,
        [0.35 / 1.35, 0.0, 1.0 / 1.35],
        atol=1e-6,
    )


def test_target_returns_none_when_no_completion_or_reducer_in_legal_moves():
    """If completion/reducer sets are non-empty but their moves are not in
    legal_moves (stale alignment), target is None — boundary defense."""
    legal = [(5, 5)]    # only this move legal
    completion = {(0, 8)}
    reducing = {(22, 4)}
    target = build_conversion_target(legal, completion, reducing,
                                     completion_weight=1.0, reducer_weight=0.35)
    assert target is None
