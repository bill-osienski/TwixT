"""Tests for Spec 4 train.py CLI surface."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _build_parser():
    """Helper that recreates the train.py argparse with the new flags only."""
    from scripts.GPU.alphazero.train import _add_recovery_retargeting_args
    parser = argparse.ArgumentParser()
    _add_recovery_retargeting_args(parser)
    return parser


def test_default_flags_enable_diagnostic_off_by_disable_flag():
    p = _build_parser()
    args = p.parse_args([])
    assert args.recovery_retargeting_disabled is False
    assert args.recovery_retargeting_classify_defense is True


def test_disable_flag_turns_off():
    p = _build_parser()
    args = p.parse_args(["--recovery-retargeting-disabled"])
    assert args.recovery_retargeting_disabled is True


def test_no_classify_defense_flag_turns_off():
    p = _build_parser()
    args = p.parse_args(["--recovery-retargeting-no-classify-defense"])
    assert args.recovery_retargeting_classify_defense is False


def test_threshold_overrides_parse():
    p = _build_parser()
    args = p.parse_args([
        "--recovery-retargeting-collapse-value-threshold", "-0.60",
        "--recovery-retargeting-delta-threshold", "0.40",
    ])
    assert args.recovery_retargeting_collapse_value_threshold == -0.60
    assert args.recovery_retargeting_delta_threshold == 0.40
