"""Tests for Spec 3 Fix 2 — narrow closeout selection tie-break."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.mcts import MCTS, MCTSConfig


def test_tiebreak_disabled_by_default():
    c = MCTSConfig()
    assert c.closeout_selection_tiebreak_enabled is False


def test_tiebreak_override_when_endpoint_in_topk():
    """argmax was redundant; endpoint is rank-3 in visits with share above floor → override."""
    cfg = MCTSConfig(
        closeout_selection_tiebreak_enabled=True,
        closeout_selection_tiebreak_max_distance=2,
        closeout_selection_tiebreak_topk=5,
        closeout_selection_tiebreak_min_value=0.95,
        closeout_selection_tiebreak_min_share=0.05,
    )
    visit_counts = {(0, 0): 100, (1, 1): 50, (2, 2): 80, (3, 3): 60}  # argmax = (0,0)
    gc_state = {
        "total_goal_distance": 2,
        "endpoint_completion_moves": [(2, 2)],
        "distance_reducing_moves": [(2, 2), (3, 3)],
    }
    root_q = 0.97
    selected_argmax_class = "redundant_reinforcement"
    updated_counts, record = MCTS.apply_closeout_selection_tiebreak(
        visit_counts=visit_counts, gc_state_full=gc_state,
        root_q=root_q, selected_argmax_class=selected_argmax_class, config=cfg,
    )
    new_argmax = max(updated_counts, key=updated_counts.get)
    assert new_argmax == (2, 2)
    assert record["overrode_to"] == "endpoint"


def test_tiebreak_skips_when_share_below_floor():
    cfg = MCTSConfig(closeout_selection_tiebreak_enabled=True,
                     closeout_selection_tiebreak_min_share=0.1)
    visit_counts = {(0, 0): 95, (2, 2): 5}  # endpoint share = 0.05 -> below 0.1 floor
    gc_state = {"total_goal_distance": 2, "endpoint_completion_moves": [(2, 2)],
                "distance_reducing_moves": [(2, 2)]}
    updated, rec = MCTS.apply_closeout_selection_tiebreak(
        visit_counts, gc_state, root_q=0.97,
        selected_argmax_class="off_chain", config=cfg,
    )
    assert updated == visit_counts
    assert rec.get("overrode_to") is None
