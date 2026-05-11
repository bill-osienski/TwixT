"""Tests for Spec 3 Fix 1 CLI flag plumbing in train.py."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _build_parser():
    from scripts.GPU.alphazero.train import build_arg_parser
    return build_arg_parser()


def test_default_flags_disable_visit_forcing():
    p = _build_parser()
    args = p.parse_args([])
    assert args.closeout_td1_visit_forcing_enabled is False
    assert args.closeout_td1_min_visits == 8
    assert args.closeout_td1_max_forced_moves == 4
    assert args.closeout_td1_require_high_value is False
    assert abs(args.closeout_td1_high_value_threshold - 0.95) < 1e-9


def test_flags_parse_overrides():
    p = _build_parser()
    args = p.parse_args([
        "--closeout-td1-visit-forcing-enabled",
        "--closeout-td1-min-visits", "16",
        "--closeout-td1-max-forced-moves", "2",
        "--closeout-td1-require-high-value",
        "--closeout-td1-high-value-threshold", "0.9",
    ])
    assert args.closeout_td1_visit_forcing_enabled is True
    assert args.closeout_td1_min_visits == 16
    assert args.closeout_td1_max_forced_moves == 2
    assert args.closeout_td1_require_high_value is True
    assert abs(args.closeout_td1_high_value_threshold - 0.9) < 1e-9
