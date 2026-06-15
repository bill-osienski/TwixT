import csv
import json
from pathlib import Path

import pytest

from scripts.GPU.alphazero.goal_line_trigger_probe_cases import (
    DEFAULT_SELECTION, EXPECTED_PROBLEM, case_id, select_cases,
)

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
