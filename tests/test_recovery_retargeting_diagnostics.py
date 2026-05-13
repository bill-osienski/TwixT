"""Tests for Spec 4 recovery / re-targeting diagnostic."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    RecoveryRetargetingConfig,
    validate_config,
)


def test_config_defaults_match_spec():
    c = RecoveryRetargetingConfig()
    assert c.enabled is True
    assert c.collapse_value_threshold == -0.75
    assert c.severe_collapse_value_threshold == -0.90
    assert c.diffuse_root_top1_threshold == 0.20
    assert c.very_diffuse_root_top1_threshold == 0.15
    assert c.delta_threshold == 0.50
    assert c.delta_max_current_score == -0.30
    assert c.alternate_component_min_size == 4
    assert c.classify_defense is True
    assert c.max_sampled_moves_per_side == 32
    assert c.sample_all_moves is False


def test_validate_collapse_lt_delta_max_current_score():
    cfg = RecoveryRetargetingConfig(collapse_value_threshold=-0.30, delta_max_current_score=-0.30)
    with pytest.raises(ValueError, match="collapse_value_threshold"):
        validate_config(cfg)


def test_validate_severe_le_collapse():
    cfg = RecoveryRetargetingConfig(collapse_value_threshold=-0.75, severe_collapse_value_threshold=-0.50)
    with pytest.raises(ValueError, match="severe_collapse_value_threshold"):
        validate_config(cfg)


def test_validate_very_diffuse_le_diffuse():
    cfg = RecoveryRetargetingConfig(diffuse_root_top1_threshold=0.20, very_diffuse_root_top1_threshold=0.30)
    with pytest.raises(ValueError, match="very_diffuse_root_top1_threshold"):
        validate_config(cfg)


def test_validate_top1_range():
    with pytest.raises(ValueError, match="diffuse_root_top1_threshold"):
        validate_config(RecoveryRetargetingConfig(diffuse_root_top1_threshold=1.5))


def test_validate_delta_positive():
    with pytest.raises(ValueError, match="delta_threshold"):
        validate_config(RecoveryRetargetingConfig(delta_threshold=0.0))


def test_validate_alternate_component_min_size_positive():
    with pytest.raises(ValueError, match="alternate_component_min_size"):
        validate_config(RecoveryRetargetingConfig(alternate_component_min_size=0))


def test_validate_max_sampled_non_negative():
    with pytest.raises(ValueError, match="max_sampled_moves_per_side"):
        validate_config(RecoveryRetargetingConfig(max_sampled_moves_per_side=-1))


def test_validate_default_config_passes():
    validate_config(RecoveryRetargetingConfig())   # must not raise
