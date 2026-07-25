"""v17 Task 2 -- the `fpu_shipped_policy_mass_reduction` config field.

Frozen design ref:
`docs/superpowers/specs/2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md`
(SHA-1 `944f358c0e3ef66503d2cbb56e31dabd145bafc2`) §2.1 "New field; retired
field remains distinct" and §2.3 "Explored mass semantics".

Scope is the CONFIG FIELD AND ITS VALIDATION ONLY. Task 2 deliberately does not
wire the selection site -- that is Task 3, and the Task 1 goldens
(`tests/test_fpu_v17_prechange_golden.py`) are what prove no search behaviour
changed here.

Retired-field behaviour (`fpu_policy_mass_reduction`, the Q_parent rule) is
covered by `tests/test_fpu_policy_mass_rule.py` and must stay untouched; the
checks below only pin that v17 did not disturb it.
"""
import dataclasses
import math

import pytest

from scripts.GPU.alphazero import mcts as mcts_mod
from scripts.GPU.alphazero.mcts import MCTSConfig, policy_mass_fpu


# ---------------------------------------------------------------------------
# The pure helper is REUSED, not reimplemented (design §2.3 "Do not add a
# duplicate formula helper"). v17 passes `config.fpu_value` as `parent_q`, and
# because §2.1 forces `fpu_value == 0.0` whenever the field is active, the
# operative v17 formula is -r*sqrt(clamp(P_explored, 0, 1)).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mass,r", [
    (0.0, 0.35), (0.25, 0.35), (1.0, 0.35),
    (0.5, 0.15), (0.5, 0.45), (0.64, 0.25), (1.0, 0.20),
])
def test_existing_helper_already_computes_the_v17_formula(mass, r):
    assert policy_mass_fpu(0.0, mass, r) == pytest.approx(-r * math.sqrt(mass))


def test_helper_clamp_and_nonfinite_behaviour_unchanged():
    # clamp to [0, 1] happens inside the helper, per §2.3
    assert policy_mass_fpu(0.0, 1.5, 0.35) == pytest.approx(-0.35)
    assert policy_mass_fpu(0.0, -0.5, 0.35) == 0.0
    # at zero explored mass the FPU is the shipped FPU (§2.3)
    assert policy_mass_fpu(0.0, 0.0, 0.45) == 0.0
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            policy_mass_fpu(0.0, bad, 0.35)
        with pytest.raises(ValueError):
            policy_mass_fpu(0.0, 0.5, bad)


def test_no_duplicate_formula_helper_was_added():
    """§2.3 forbids a second helper; v17 must call `policy_mass_fpu`."""
    dupes = [n for n in dir(mcts_mod)
             if "policy_mass" in n and n not in
             ("policy_mass_fpu", "explored_policy_mass")]
    assert dupes == []


# ---------------------------------------------------------------------------
# §2.1 configuration rules
# ---------------------------------------------------------------------------

def test_new_field_defaults_to_none():
    assert MCTSConfig().fpu_shipped_policy_mass_reduction is None


def test_zero_is_accepted_and_stays_distinct_from_none():
    cfg = MCTSConfig(fpu_shipped_policy_mass_reduction=0.0)
    assert cfg.fpu_shipped_policy_mass_reduction == 0.0
    # `0.0` and `None` must both take the shipped branch (§2.2), but the field
    # still records which was configured, so provenance can tell them apart.
    assert MCTSConfig().fpu_shipped_policy_mass_reduction is not 0.0  # noqa: F632


@pytest.mark.parametrize("r", [0.15, 0.20, 0.25, 0.35, 0.45])
def test_frozen_grid_coefficients_are_all_accepted(r):
    assert MCTSConfig(fpu_shipped_policy_mass_reduction=r).fpu_shipped_policy_mass_reduction == r


@pytest.mark.parametrize("bad", [-1e-9, -0.35, math.nan, math.inf, -math.inf])
def test_negative_or_nonfinite_rejected(bad):
    with pytest.raises(ValueError, match="fpu_shipped_policy_mass_reduction"):
        MCTSConfig(fpu_shipped_policy_mass_reduction=bad)


def test_mutually_exclusive_with_the_retired_parent_relative_field():
    with pytest.raises(ValueError, match="mutually exclusive"):
        MCTSConfig(fpu_shipped_policy_mass_reduction=0.35,
                   fpu_policy_mass_reduction=0.35)
    # also at the enabled-zero of each, which are not "off"
    with pytest.raises(ValueError, match="mutually exclusive"):
        MCTSConfig(fpu_shipped_policy_mass_reduction=0.0,
                   fpu_policy_mass_reduction=0.0)


@pytest.mark.parametrize("fpu_value", [-0.20, 0.20, -1e-12])
def test_non_none_new_field_rejected_unless_fpu_value_is_zero(fpu_value):
    """§2.1: 'Structurally reject a non-None new field unless fpu_value == 0.0'.
    This is what makes the operative formula -r*sqrt(P_explored)."""
    with pytest.raises(ValueError, match="fpu_value"):
        MCTSConfig(fpu_value=fpu_value, fpu_shipped_policy_mass_reduction=0.35)
    # including the enabled zero
    with pytest.raises(ValueError, match="fpu_value"):
        MCTSConfig(fpu_value=fpu_value, fpu_shipped_policy_mass_reduction=0.0)


def test_shipped_fpu_value_still_works_when_the_new_field_is_none():
    """The retired absolute path must be undisturbed: a nonzero `fpu_value`
    alone is still legal (that is how the shipped -0.20 experiments ran)."""
    cfg = MCTSConfig(fpu_value=-0.20)
    assert cfg.fpu_value == -0.20
    assert cfg.fpu_shipped_policy_mass_reduction is None


def test_retired_field_semantics_are_unchanged():
    """v17 must not rename, reinterpret, or reuse the Q_parent field (§2.1)."""
    cfg = MCTSConfig(fpu_policy_mass_reduction=0.0)
    assert cfg.fpu_policy_mass_reduction == 0.0
    assert cfg.fpu_shipped_policy_mass_reduction is None
    with pytest.raises(ValueError, match="fpu_policy_mass_reduction"):
        MCTSConfig(fpu_policy_mass_reduction=-0.1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        MCTSConfig(fpu_value=-0.20, fpu_policy_mass_reduction=0.35)


def test_field_survives_dataclasses_replace_and_asdict():
    """Provenance records the FULL effective config (§12), so the new field
    must round-trip through the dataclass helpers the fingerprints use."""
    cfg = dataclasses.replace(MCTSConfig(), fpu_shipped_policy_mass_reduction=0.25)
    assert cfg.fpu_shipped_policy_mass_reduction == 0.25
    assert dataclasses.asdict(cfg)["fpu_shipped_policy_mass_reduction"] == 0.25
    assert "fpu_shipped_policy_mass_reduction" in dataclasses.asdict(MCTSConfig())


# ===========================================================================
# Task 3 -- the selection branch. Design §2.2: the site must branch BEFORE
# calculating explored mass, so at `None` and `0.0` v17 does not call the
# formula helper, scan children for explored mass, alter RNG consumption,
# change tie behaviour, or add observer mutations.
#
# The golden checks in tests/test_fpu_v17_prechange_golden.py cover the RNG,
# tie and observer halves; the checks here cover the "no helper call, no mass
# scan" half, which is only observable at the call site itself.
# ===========================================================================
import random

from scripts.GPU.alphazero.mcts import MCTS, explored_policy_mass
from tests.test_fpu_policy_mass_rule import (
    _stub_value_fn, _synthetic_root_for_policy_mass,
)


def _select_with(**cfg_kwargs):
    root, X, Y, Z = _synthetic_root_for_policy_mass()
    cfg = MCTSConfig(n_simulations=1, c_puct=1.5, **cfg_kwargs)
    chosen, _child = MCTS(_stub_value_fn(), cfg, random.Random(0))._select_child(root)
    return chosen, X, Y, Z


def test_positive_v17_coefficient_changes_the_unvisited_child_choice():
    """Discriminator. On the pinned tree P_explored = 0.25, so a positive r
    lowers the unvisited move Y's score by r*sqrt(0.25) = r/2. Y beats the
    visited X by 0.04926, so r = 0.35 (a frozen grid point) must flip the
    choice to X. If the branch were not wired, Y would still win."""
    chosen, X, _Y, _Z = _select_with(fpu_shipped_policy_mass_reduction=0.35)
    assert chosen == X


def test_shipped_and_exact_zero_pick_the_unvisited_child_like_shipped():
    """`None` and `0.0` must both reproduce the shipped choice (Y)."""
    shipped_choice, _X, Y, _Z = _select_with()
    assert shipped_choice == Y
    assert _select_with(fpu_shipped_policy_mass_reduction=0.0)[0] == Y


def test_small_positive_coefficient_still_picks_the_unvisited_child():
    """Below the 0.0985 boundary the unvisited move still wins, so the flip
    above is the coefficient acting through the formula, not an on/off step."""
    chosen, _X, Y, _Z = _select_with(fpu_shipped_policy_mass_reduction=0.05)
    assert chosen == Y


class _Counter:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, *a, **k):
        self.calls += 1
        return self.fn(*a, **k)


def _call_counts(monkeypatch, **cfg_kwargs):
    """Count real calls to the two helpers during one `_select_child`."""
    formula = _Counter(mcts_mod.policy_mass_fpu)
    scan = _Counter(mcts_mod.explored_policy_mass)
    monkeypatch.setattr(mcts_mod, "policy_mass_fpu", formula)
    monkeypatch.setattr(mcts_mod, "explored_policy_mass", scan)
    _select_with(**cfg_kwargs)
    return formula.calls, scan.calls


def test_none_calls_neither_helper(monkeypatch):
    assert _call_counts(monkeypatch) == (0, 0)


def test_exact_zero_calls_neither_helper(monkeypatch):
    """§2.2: stronger than numerical equivalence -- `0.0` must take the shipped
    branch STRUCTURALLY, never computing explored mass and discarding it."""
    assert _call_counts(monkeypatch, fpu_shipped_policy_mass_reduction=0.0) == (0, 0)


def test_positive_coefficient_calls_both_helpers_once(monkeypatch):
    """Keeps the two checks above non-vacuous: the counters do observe calls
    when the positive branch runs, and mass is computed once per call, not
    once per child."""
    assert _call_counts(monkeypatch, fpu_shipped_policy_mass_reduction=0.35) == (1, 1)


def test_retired_field_still_calls_the_helper_with_q_parent(monkeypatch):
    """The v16 branch must be untouched: it still reduces from `node.q_value`,
    not from `fpu_value`. v17 must not have re-pointed the shared helper."""
    seen = {}

    def spy(parent_q, explored_mass, r):
        seen.update(parent_q=parent_q, explored_mass=explored_mass, r=r)
        return parent_q - r * math.sqrt(explored_mass)

    monkeypatch.setattr(mcts_mod, "policy_mass_fpu", spy)
    _select_with(fpu_policy_mass_reduction=1.5)
    assert seen["parent_q"] == 0.5           # node.q_value, the Q_parent rule
    assert seen["explored_mass"] == pytest.approx(0.25)
    assert seen["r"] == 1.5


def test_v17_reduces_from_fpu_value_not_q_parent(monkeypatch):
    """The whole point of v17: Q_parent does not participate. The helper must
    receive `fpu_value` (0.0 by validation), never the root's 0.5."""
    seen = {}

    def spy(parent_q, explored_mass, r):
        seen.update(parent_q=parent_q, explored_mass=explored_mass, r=r)
        return parent_q - r * math.sqrt(explored_mass)

    monkeypatch.setattr(mcts_mod, "policy_mass_fpu", spy)
    _select_with(fpu_shipped_policy_mass_reduction=0.35)
    assert seen["parent_q"] == 0.0           # fpu_value, NOT node.q_value (0.5)
    assert seen["explored_mass"] == pytest.approx(0.25)
    assert seen["r"] == 0.35


def test_explored_mass_semantics_unchanged_by_v17():
    """§2.3 reuses the existing completed-visit definition verbatim."""
    root, _X, _Y, _Z = _synthetic_root_for_policy_mass()
    assert explored_policy_mass(root) == pytest.approx(0.25)
