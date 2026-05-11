"""Unit tests for Spec 3 Fix 1 — td=1 root visit forcing (mcts side)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.mcts import MCTSConfig


def test_config_defaults_disable_visit_forcing():
    c = MCTSConfig()
    assert c.closeout_td1_visit_forcing_enabled is False
    assert c.closeout_td1_min_visits == 8
    assert c.closeout_td1_max_forced_moves == 4
    assert c.closeout_td1_require_high_value is False
    assert c.closeout_td1_high_value_threshold == 0.95


def test_config_accepts_overrides():
    c = MCTSConfig(closeout_td1_visit_forcing_enabled=True,
                   closeout_td1_min_visits=16,
                   closeout_td1_max_forced_moves=2,
                   closeout_td1_require_high_value=True,
                   closeout_td1_high_value_threshold=0.9)
    assert c.closeout_td1_visit_forcing_enabled is True
    assert c.closeout_td1_min_visits == 16
    assert c.closeout_td1_high_value_threshold == 0.9
