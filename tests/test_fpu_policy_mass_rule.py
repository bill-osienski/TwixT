"""Tests for the context-relative FPU (policy-mass) rule: the pure helpers
(`policy_mass_fpu`, `explored_policy_mass`) and the `_select_child` wiring
that consumes them.

Frozen design ref: docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md §1

Task 1 tests (below) are PURE: `policy_mass_fpu` and `MCTSConfig`
construction only. Task 2 tests add `explored_policy_mass` (also pure —
takes/reads an `MCTSNode` but does no search) plus `_select_child` tests
built on hand-built synthetic `MCTSNode` trees + a CPU stub value fn (same
pattern as `tests/test_fpu_value.py`) — no real search/GPU/MLX involved.
"""
import dataclasses
import math
import random

import pytest

from scripts.GPU.alphazero.mcts import (
    MCTS,
    MCTSConfig,
    MCTSNode,
    encode_move,
    explored_policy_mass,
    policy_mass_fpu,
)


# ---------------------------------------------------------------------------
# Formula
# ---------------------------------------------------------------------------

def test_formula_basic_triple():
    # parent_q=0.5, explored_mass=0.25, r=0.2 -> 0.5 - 0.2*sqrt(0.25) = 0.5 - 0.1 = 0.4
    assert policy_mass_fpu(0.5, 0.25, 0.2) == pytest.approx(0.4)


def test_formula_second_nontrivial_triple():
    # parent_q=-0.2, explored_mass=0.81, r=0.5 -> -0.2 - 0.5*sqrt(0.81) = -0.2 - 0.45 = -0.65
    assert policy_mass_fpu(-0.2, 0.81, 0.5) == pytest.approx(-0.65)


# ---------------------------------------------------------------------------
# Clamp
# ---------------------------------------------------------------------------

def test_clamp_below_zero_collapses_to_parent_q():
    # explored_mass < 0 clamps to 0 -> result == parent_q (sqrt(0) term vanishes)
    assert policy_mass_fpu(0.3, -5.0, 0.2) == 0.3


def test_clamp_above_one_collapses_to_parent_q_minus_r():
    # explored_mass > 1 clamps to 1 -> result == parent_q - r
    assert policy_mass_fpu(0.3, 5.0, 0.2) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Nonfinite reject — each of the three args, each of {nan, inf, -inf},
# with the other two args finite.
# ---------------------------------------------------------------------------

NONFINITE_VALUES = (math.nan, math.inf, -math.inf)


@pytest.mark.parametrize("bad", NONFINITE_VALUES)
def test_nonfinite_parent_q_rejected(bad):
    with pytest.raises(ValueError):
        policy_mass_fpu(bad, 0.5, 0.2)


@pytest.mark.parametrize("bad", NONFINITE_VALUES)
def test_nonfinite_explored_mass_rejected(bad):
    with pytest.raises(ValueError):
        policy_mass_fpu(0.5, bad, 0.2)


@pytest.mark.parametrize("bad", NONFINITE_VALUES)
def test_nonfinite_r_rejected(bad):
    with pytest.raises(ValueError):
        policy_mass_fpu(0.5, 0.5, bad)


# ---------------------------------------------------------------------------
# MCTSConfig field: default, 0.0 != None
# ---------------------------------------------------------------------------

def test_config_default_is_none():
    assert MCTSConfig().fpu_policy_mass_reduction is None


def test_config_zero_is_not_none_and_constructs():
    cfg = MCTSConfig(fpu_policy_mass_reduction=0.0)
    assert cfg.fpu_policy_mass_reduction == 0.0
    assert cfg.fpu_policy_mass_reduction is not None


# ---------------------------------------------------------------------------
# Guard: mutual exclusion with nonzero fpu_value; finite/>=0 requirement
# ---------------------------------------------------------------------------

def test_guard_rejects_nonzero_fpu_value_with_reduction_set():
    with pytest.raises(ValueError):
        MCTSConfig(fpu_value=-0.2, fpu_policy_mass_reduction=0.1)


def test_guard_rejects_negative_reduction():
    with pytest.raises(ValueError):
        MCTSConfig(fpu_policy_mass_reduction=-0.1)


def test_guard_rejects_infinite_reduction():
    with pytest.raises(ValueError):
        MCTSConfig(fpu_policy_mass_reduction=math.inf)


def test_guard_rejects_nan_reduction():
    with pytest.raises(ValueError):
        MCTSConfig(fpu_policy_mass_reduction=math.nan)


# ---------------------------------------------------------------------------
# Replace path OK
# ---------------------------------------------------------------------------

def test_replace_path_constructs_without_raising():
    cfg = dataclasses.replace(MCTSConfig(), fpu_policy_mass_reduction=0.2)
    assert cfg.fpu_value == 0.0
    assert cfg.fpu_policy_mass_reduction == 0.2


def test_config_zero_reduction_direct_construction_ok():
    cfg = MCTSConfig(fpu_policy_mass_reduction=0.0)
    assert cfg.fpu_policy_mass_reduction == 0.0


def test_config_positive_reduction_direct_construction_ok():
    cfg = MCTSConfig(fpu_policy_mass_reduction=0.3)
    assert cfg.fpu_policy_mass_reduction == 0.3


# ---------------------------------------------------------------------------
# explored_policy_mass -- completed-visit-only P_explored (Task 2, design §1
# "Completed-visits-only (safeguard 6a)"): sums prior mass only over children
# with a COMPLETED (backed-up) visit; zero-visit/pending/no-child moves are
# excluded.
# ---------------------------------------------------------------------------

def test_explored_policy_mass_counts_only_completed_visit_children():
    A, B, C = encode_move(0, 0), encode_move(1, 1), encode_move(2, 2)
    root = MCTSNode(state=None)
    root.priors = {A: 0.5, B: 0.3, C: 0.2}
    root.children[A] = MCTSNode(state=None, parent=root, move=A,
                                 visit_count=3, value_sum=1.5)   # completed
    root.children[B] = MCTSNode(state=None, parent=root, move=B,
                                 visit_count=0, value_sum=0.0)   # pending/unvisited
    # C: no child entry at all (equally unvisited).
    assert explored_policy_mass(root) == pytest.approx(0.5)


def test_explored_policy_mass_sums_across_multiple_completed_children():
    A, B, C = encode_move(0, 0), encode_move(1, 1), encode_move(2, 2)
    root = MCTSNode(state=None)
    root.priors = {A: 0.5, B: 0.3, C: 0.2}
    root.children[A] = MCTSNode(state=None, parent=root, move=A,
                                 visit_count=3, value_sum=1.5)   # completed
    root.children[B] = MCTSNode(state=None, parent=root, move=B,
                                 visit_count=0, value_sum=0.0)   # pending, excluded
    root.children[C] = MCTSNode(state=None, parent=root, move=C,
                                 visit_count=1, value_sum=0.4)   # completed
    assert explored_policy_mass(root) == pytest.approx(0.7)  # 0.5 + 0.2


def test_explored_policy_mass_no_visited_children_is_zero():
    A, B, C = encode_move(0, 0), encode_move(1, 1), encode_move(2, 2)
    root = MCTSNode(state=None)
    root.priors = {A: 0.5, B: 0.3, C: 0.2}
    root.children[B] = MCTSNode(state=None, parent=root, move=B,
                                 visit_count=0, value_sum=0.0)   # pending
    # A, C: no child entries at all.
    assert explored_policy_mass(root) == 0.0


# ---------------------------------------------------------------------------
# _select_child wiring -- the policy-mass FPU replaces `fpu_value` at the
# unvisited-child site only when `fpu_policy_mass_reduction is not None`;
# the off path (`None`) must reproduce the legacy `fpu_value` choice exactly
# on the SAME tree (see design §1 "Byte-identical-off proof").
# ---------------------------------------------------------------------------

def _stub_value_fn():
    def f(state):
        return {}, 0.0
    return f


def _synthetic_root_for_policy_mass():
    """Root with three candidate moves (same PUCT-arithmetic style as
    `tests/test_fpu_value.py::_synthetic_root`):
      X = a decent VISITED reply -- child q in the child's own perspective
          = -0.1, so the mover (parent) sees -(-0.1) = +0.1 -- prior 0.01,
          visited 100 times (COMPLETED).
      Z = a bad VISITED reply -- child q in its own perspective = +0.8, so
          the mover sees -0.8 -- prior 0.24, visited 50 times (COMPLETED).
          Z exists only to inflate P_explored; its score is the worst of the
          three in BOTH configs below and never wins.
      Y = UNVISITED (no child node) -- prior 0.01 -- the site under test.
    Root: visit_count=100, value_sum=50.0 -> q_value=0.5 (= Q_parent).

    Arithmetic (c_puct=1.5, sqrt_parent = sqrt(101) = 10.04987562112089;
    verified numerically, not just by hand):
      score_X = 0.1 + 1.5*0.01*sqrt_parent/(1+100)  = 0.10149255578531499
      score_Z = -0.8 + 1.5*0.24*sqrt_parent/(1+50)  = -0.7290597014979703  (always loses)
      u_Y     =       1.5*0.01*sqrt_parent/(1+0)    = 0.15074813431681333

    P_explored = prior(X) + prior(Z) = 0.01 + 0.24 = 0.25 (Y excluded,
    unvisited)  ->  sqrt(P_explored) = 0.5

    OFF (fpu_policy_mass_reduction=None -> legacy fpu_value=0.0 path):
      score_Y = 0.0 + u_Y = 0.15074813431681333 > score_X -> Y wins
      (the legacy "unvisited beats a decent visited reply" pathology, same
      shape as test_fpu_value.py's fpu=0.0 case). Margin ~0.0493.

    ON (fpu_policy_mass_reduction=r=1.5):
      FPU = policy_mass_fpu(0.5, 0.25, 1.5) = 0.5 - 1.5*0.5 = -0.25
      score_Y = -0.25 + u_Y = -0.09925186568318667 < score_X -> X wins
      Margin ~0.2007.

    Both margins are far above the 1e-8 tie epsilon, so the rng seed cannot
    affect either outcome.
    """
    root = MCTSNode(state=None, visit_count=100, value_sum=50.0)  # q_value=0.5
    X, Y, Z = encode_move(0, 0), encode_move(1, 1), encode_move(2, 2)
    root.priors = {X: 0.01, Y: 0.01, Z: 0.24}
    root.children[X] = MCTSNode(state=None, parent=root, move=X,
                                 visit_count=100, value_sum=-10.0)  # q=-0.1
    root.children[Z] = MCTSNode(state=None, parent=root, move=Z,
                                 visit_count=50, value_sum=40.0)    # q=0.8
    # Y intentionally has no child entry (unvisited) -- the site under test.
    return root, X, Y, Z


def test_select_child_applies_policy_mass_fpu_when_enabled():
    root, X, Y, Z = _synthetic_root_for_policy_mass()
    # Pin the formula inputs/output directly (ties the test to the actual
    # helpers, not just a hardcoded literal).
    assert explored_policy_mass(root) == pytest.approx(0.25)
    assert policy_mass_fpu(root.q_value, explored_policy_mass(root), 1.5) == pytest.approx(-0.25)

    cfg = MCTSConfig(n_simulations=1, c_puct=1.5, fpu_value=0.0,
                      fpu_policy_mass_reduction=1.5)
    m = MCTS(_stub_value_fn(), cfg, random.Random(0))
    chosen_move_id, _ = m._select_child(root)
    # ON: the reduced FPU drops Y below X -- differs from the OFF/legacy
    # choice asserted in test_select_child_off_path_reproduces_legacy_choice.
    assert chosen_move_id == X


def test_select_child_off_path_reproduces_legacy_choice():
    # Discriminator: same tree, fpu_policy_mass_reduction=None -> the legacy
    # fpu_value=0.0 absolute path is used, and the unvisited move Y wins
    # (would FAIL if the branch above had perturbed the off path).
    root, X, Y, Z = _synthetic_root_for_policy_mass()
    cfg = MCTSConfig(n_simulations=1, c_puct=1.5, fpu_value=0.0,
                      fpu_policy_mass_reduction=None)
    m = MCTS(_stub_value_fn(), cfg, random.Random(0))
    chosen_move_id, _ = m._select_child(root)
    assert chosen_move_id == Y
