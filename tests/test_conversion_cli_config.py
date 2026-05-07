# tests/test_conversion_cli_config.py
"""CLI/config invariants for conversion auxiliary loss (Spec 2 §9)."""
import argparse
import pytest


def _build_parser():
    """Mirror scripts/GPU/alphazero/train.py argparse construction.
    Imports it after the new flags are added.
    """
    from scripts.GPU.alphazero.train import _build_parser_for_test
    return _build_parser_for_test()


def test_conversion_disabled_by_default_effective_weight_zero():
    """Spec 2 §11.3 anchor: default config has effective_loss_weight = 0.0."""
    p = _build_parser()
    args = p.parse_args([])
    assert args.conversion_policy_loss_enabled is False
    assert args.conversion_policy_loss_weight == 0.05  # configured default


def test_conversion_enabled_uses_configured_weight():
    p = _build_parser()
    args = p.parse_args(["--conversion-policy-loss-enabled"])
    assert args.conversion_policy_loss_enabled is True
    assert args.conversion_policy_loss_weight == 0.05


def test_conversion_enabled_with_zero_weight_errors():
    """Spec 2 §9.3: enabled + weight==0 must error."""
    p = _build_parser()
    from scripts.GPU.alphazero.train import _validate_conversion_args
    args = p.parse_args(["--conversion-policy-loss-enabled",
                         "--conversion-policy-loss-weight", "0.0"])
    with pytest.raises(SystemExit):
        _validate_conversion_args(p, args)


def test_reducer_weight_greater_than_completion_weight_errors():
    p = _build_parser()
    from scripts.GPU.alphazero.train import _validate_conversion_args
    args = p.parse_args(["--conversion-policy-loss-enabled",
                         "--conversion-completion-weight", "0.5",
                         "--conversion-reducer-weight", "0.8"])
    with pytest.raises(SystemExit):
        _validate_conversion_args(p, args)


def test_conversion_max_total_goal_distance_bounds():
    """Must be in [1, 3]."""
    p = _build_parser()
    from scripts.GPU.alphazero.train import _validate_conversion_args
    for bad in ["0", "4", "-1"]:
        args = p.parse_args(["--conversion-max-total-goal-distance", bad])
        with pytest.raises(SystemExit):
            _validate_conversion_args(p, args)
    for ok in ["1", "2", "3"]:
        args = p.parse_args(["--conversion-max-total-goal-distance", ok])
        _validate_conversion_args(p, args)  # should not raise


def test_sample_boost_without_loss_warns_and_tagging_stays_off(capsys):
    """Spec 2 §9.3: sample_boost > 1.0 with loss off should warn."""
    p = _build_parser()
    from scripts.GPU.alphazero.train import _validate_conversion_args
    args = p.parse_args(["--conversion-sample-boost", "2.0"])
    _validate_conversion_args(p, args)
    captured = capsys.readouterr()
    assert "[WARN]" in captured.out
    assert "--conversion-sample-boost" in captured.out
