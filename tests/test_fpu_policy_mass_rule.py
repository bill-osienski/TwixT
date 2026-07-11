"""Task 1 — pure tests for the context-relative FPU (policy-mass) rule helper
and its MCTSConfig plumbing.

Frozen design ref: docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md §1

PURE ONLY: this file tests `policy_mass_fpu` (a module-level function) and
`MCTSConfig` construction. No MCTS/search/GPU/MLX involved.
"""
import dataclasses
import math

import pytest

from scripts.GPU.alphazero.mcts import MCTSConfig, policy_mass_fpu


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
