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
