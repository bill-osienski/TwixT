"""Tests for Fix 0: td-before closeout breakdown (spec 2026-05-10 §3)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import aggregate_td_closeout_breakdown


def _ply_record(*, ply, side, det_side, td, q, sel_class, ec_p=None, ec_v=None, rd_p=None, rd_v=None):
    """Build a minimal per-ply diagnostic record matching schema in §3.1."""
    def _rank_block(p, v):
        if p is None and v is None:
            return {"best_policy_rank": None, "best_visit_rank": None}
        return {"best_policy_rank": p, "best_visit_rank": v}
    return {
        "ply": ply,
        "side_to_move": side,
        "root_summary": {"q_value": q},
        "goal_completion": {"total_goal_distance_before": td},
        "endpoint_completion_ranking": _rank_block(ec_p, ec_v),
        "distance_reducing_ranking": _rank_block(rd_p, rd_v),
        "selected_move_classification": {"primary_class": sel_class},
    }


def test_td_buckets_split_by_distance_and_classify_selection():
    # Build two td=1 rows (one redundant, one completes_endpoint) and
    # one td=2 row (off_chain). Detected player is "black" throughout.
    records = [
        _ply_record(ply=10, side="black", det_side="black", td=1, q=0.97,
                    sel_class="redundant_reinforcement", ec_p=33, ec_v=173, rd_p=33, rd_v=173),
        _ply_record(ply=12, side="black", det_side="black", td=1, q=0.98,
                    sel_class="completes_endpoint", ec_p=1, ec_v=1, rd_p=1, rd_v=1),
        _ply_record(ply=14, side="black", det_side="black", td=2, q=0.96,
                    sel_class="off_chain", ec_p=None, ec_v=None, rd_p=5, rd_v=4),
    ]

    out = aggregate_td_closeout_breakdown(records, detected_player="black",
                                          high_value_threshold=0.95)

    assert out["td=1"]["records"] == 2
    assert out["td=1"]["high_value_records"] == 2
    assert out["td=1"]["selected_redundant_rate"] == 0.5
    assert out["td=1"]["selected_completes_endpoint_rate"] == 0.5
    # endpoint exists in both td=1 rows
    assert out["td=1"]["endpoint_completion_exists_rate"] == 1.0
    # visit top-5: ranks 173 (no) and 1 (yes) → 0.5
    assert out["td=1"]["endpoint_visit_top5_rate"] == 0.5
    assert out["td=1"]["endpoint_visit_gt20_rate"] == 0.5

    assert out["td=2"]["records"] == 1
    assert out["td=2"]["selected_off_chain_rate"] == 1.0
    assert out["td=2"]["endpoint_completion_exists_rate"] == 0.0
    # reducer exists, in visit top-5
    assert out["td=2"]["distance_reducer_exists_rate"] == 1.0
    assert out["td=2"]["reducer_visit_top5_rate"] == 1.0


def test_records_for_other_side_to_move_are_excluded():
    records = [
        _ply_record(ply=10, side="red", det_side="black", td=1, q=0.97,
                    sel_class="completes_endpoint", ec_p=1, ec_v=1, rd_p=1, rd_v=1),
    ]
    out = aggregate_td_closeout_breakdown(records, detected_player="black",
                                          high_value_threshold=0.95)
    assert out["td=1"]["records"] == 0


def test_td_outside_1_2_3_is_ignored():
    records = [
        _ply_record(ply=10, side="black", det_side="black", td=4, q=0.97,
                    sel_class="off_chain", ec_p=None, ec_v=None, rd_p=None, rd_v=None),
    ]
    out = aggregate_td_closeout_breakdown(records, detected_player="black",
                                          high_value_threshold=0.95)
    assert out["td=1"]["records"] == 0
    assert out["td=2"]["records"] == 0
    assert out["td=3"]["records"] == 0
