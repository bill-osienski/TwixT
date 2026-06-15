import csv
import json
from pathlib import Path

import pytest

from scripts.GPU.alphazero.goal_line_trigger_probe_cases import (
    DEFAULT_SELECTION, EXPECTED_PROBLEM, case_id, position_state, select_cases, summarize,
)
from tests.goal_line_probe_fixtures import legal_replay

CANON_DIR = Path("logs/eval/loss_analysis_v2_1")
CANON_CANDIDATES = CANON_DIR / "goal_line_trigger_probe_candidates.csv"
CANON_MANIFEST = CANON_DIR / "goal_line_trigger_probe_manifest.json"


def _cand(**over):
    base = {
        "game_idx": "769", "rank": "4", "n_moves": "45",
        "collapse_type": "sharp_value_drop", "largest_drop_phase": "post_opening",
        "trigger_zone": "red_goal_band_3", "prev_black_ply": "39",
        "prev_black_row": "21", "prev_black_col": "21", "prev_black_value": "0.88",
        "prev_black_top1": "0.885", "trigger_red_ply": "40", "trigger_red_row": "22",
        "trigger_red_col": "22", "trigger_red_value": "0.65", "trigger_red_top1": "0.955",
        "drop_black_ply": "41", "drop_black_row": "18", "drop_black_col": "6",
        "drop_black_value": "-0.46", "drop_black_top1": "0.08", "drop_amount": "-1.34",
        "replay_path": "logs/eval/x_replays/game_000769.json",
    }
    base.update(over)
    return base


def test_candidate_to_case_field_mapping():
    case = select_cases([_cand()], DEFAULT_SELECTION)[0]
    assert case["game_idx"] == 769 and case["rank"] == 4
    assert case["position_ply"] == 39 and case["side_to_move"] == "black"
    assert case["expected_problem"] == EXPECTED_PROBLEM
    assert case["trigger_red_ply"] == 40
    assert case["trigger_red_move"] == {"row": 22, "col": 22}
    assert case["trigger_zone"] == "red_goal_band_3"
    assert case["baseline_black_prev_value"] == 0.88
    assert case["baseline_black_prev_top1"] == 0.885
    assert case["drop_black_ply"] == 41 and case["drop_amount"] == -1.34
    assert case["replay_path"] == "logs/eval/x_replays/game_000769.json"


def test_select_filters_each_knob_at_boundary():
    sel = DEFAULT_SELECTION
    assert select_cases([_cand(prev_black_value="0.24")], sel) == []
    assert len(select_cases([_cand(prev_black_value="0.25")], sel)) == 1
    assert select_cases([_cand(prev_black_top1="0.49")], sel) == []
    assert len(select_cases([_cand(prev_black_top1="0.5")], sel)) == 1
    assert select_cases([_cand(largest_drop_phase="opening")], sel) == []
    assert select_cases([_cand(trigger_zone="center")], sel) == []


def test_select_post_opening_only_can_be_disabled():
    sel = {**DEFAULT_SELECTION, "post_opening_only": False}
    assert len(select_cases([_cand(largest_drop_phase="opening")], sel)) == 1


def test_select_preserves_input_order():
    rows = [_cand(game_idx="3", rank="1"), _cand(game_idx="1", rank="2")]
    assert [c["game_idx"] for c in select_cases(rows, DEFAULT_SELECTION)] == [3, 1]


def test_case_id_format():
    case = select_cases([_cand(game_idx="15", prev_black_ply="19")], DEFAULT_SELECTION)[0]
    assert case_id(case) == "game_000015_ply_19"


def test_position_state_reconstructs_black_to_move():
    replay = legal_replay(8)                      # plies 0..7; ply 5 is black's turn
    state = position_state(replay, 5, "black")    # apply moves[0:5] -> black to move
    assert state.to_move == "black"


def test_position_state_position_ply_19_inside_opening_window():
    # Boundary: game-15-style case. Drop is post-opening but the black decision
    # ply is 19 (< opening_plies). position_state must reconstruct it normally.
    replay = legal_replay(22)
    assert replay["n_moves"] >= 20
    state = position_state(replay, 19, "black")   # 19 moves applied -> black to move
    assert state.to_move == "black"


def test_position_state_raises_on_out_of_range_ply():
    replay = legal_replay(8)
    with pytest.raises(ValueError, match="out of range"):
        position_state(replay, 99, "black")


def test_position_state_raises_on_side_to_move_mismatch():
    replay = legal_replay(8)
    # apply moves[0:4] -> red to move; claiming black must fail loud
    with pytest.raises(ValueError, match="side_to_move"):
        position_state(replay, 4, "black")


@pytest.mark.skipif(not CANON_CANDIDATES.exists() or not CANON_MANIFEST.exists(),
                    reason="canonical loss_analysis_v2_1 artifacts not present")
def test_real_candidates_reproduce_canonical_18():
    with CANON_CANDIDATES.open() as f:
        rows = list(csv.DictReader(f))
    manifest = json.loads(CANON_MANIFEST.read_text())
    got = select_cases(rows, manifest["selection"])
    got_keys = [(c["game_idx"], c["position_ply"]) for c in got]
    want_keys = [(c["game_idx"], c["position_ply"]) for c in manifest["cases"]]
    assert got_keys == want_keys          # exact set AND order
    assert len(got_keys) == manifest["num_cases"] == 18


def test_summarize_metrics_hand_computed():
    # values: two >= 0.5, one in [0.25,0.5), one below 0.25
    values = [0.8, 0.5, 0.3, -0.4]
    shares = [0.9, 0.8, 0.5, 0.2]
    s = summarize(values, shares)
    assert s["num_cases"] == 4
    assert s["mean_black_root_value"] == pytest.approx((0.8 + 0.5 + 0.3 - 0.4) / 4)
    assert s["median_black_root_value"] == pytest.approx(0.4)   # median(0.8,0.5,0.3,-0.4)
    assert s["black_overvalue_rate"] == 0.75                    # 3 of 4 >= 0.25
    assert s["severe_black_overvalue_rate"] == 0.5             # 2 of 4 >= 0.50
    assert s["mean_top1_share"] == pytest.approx((0.9 + 0.8 + 0.5 + 0.2) / 4)
    assert s["median_top1_share"] == pytest.approx(0.65)


def test_summarize_threshold_boundaries_inclusive():
    s = summarize([0.25, 0.50], [0.5, 0.5])
    assert s["black_overvalue_rate"] == 1.0      # 0.25 counts (>=)
    assert s["severe_black_overvalue_rate"] == 0.5  # only 0.50 counts


def test_summarize_empty_raises():
    with pytest.raises(ValueError, match="no cases"):
        summarize([], [])
