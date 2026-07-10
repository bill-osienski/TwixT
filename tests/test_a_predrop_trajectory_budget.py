from scripts.GPU.alphazero.diagnose_a_predrop_trajectory_budget import (
    ply_window, side_for_ply, summarize_case)


def test_side_for_ply_parity():
    # red moves first (ply 0), so even plies are red-to-move
    assert side_for_ply(0) == "red"
    assert side_for_ply(19) == "black"
    assert side_for_ply(20) == "red"


def test_ply_window_spans_predrop_and_drop():
    # predrop=19, drop=21 -> {15,17,19} u {21,23,25}
    assert ply_window(19, 21, n_moves=49) == [15, 17, 19, 21, 23, 25]


def test_ply_window_clips_to_replay_length():
    # game 347: predrop=73, drop=75, n_moves=79 -> ply 79 is out of range
    assert ply_window(73, 75, n_moves=79) == [69, 71, 73, 75, 77]


def test_ply_window_clips_negative_plies_and_dedupes():
    # predrop=2, drop=4 -> {-2,0,2} u {4,6,8}; -2 dropped, nothing duplicated
    assert ply_window(2, 4, n_moves=7) == [0, 2, 4, 6]


def test_ply_window_dedupes_when_drop_overlaps_predrop_offsets():
    # a hypothetical drop only 2 plies later still yields 6 distinct plies,
    # but a larger predrop offset overlap must not double-count
    assert ply_window(10, 12, n_moves=100) == [6, 8, 10, 12, 14, 16]
    assert len(ply_window(10, 12, n_moves=100)) == 6


def test_summarize_case_splits_at_drop_ply():
    rows = [
        {"ply": 15, "root_black_value": 0.4},
        {"ply": 17, "root_black_value": 0.6},
        {"ply": 19, "root_black_value": 0.8},   # predrop, still pre
        {"ply": 21, "root_black_value": 0.0},   # drop_ply -> post
        {"ply": 23, "root_black_value": -0.3},
    ]
    s = summarize_case(rows, drop_ply=21)
    assert s["n_pre"] == 3 and s["n_post"] == 2
    assert abs(s["pre_drop_mean"] - 0.6) < 1e-9
    assert abs(s["post_drop_mean"] - (-0.15)) < 1e-9
    assert abs(s["drop_delta"] - (-0.75)) < 1e-9
    assert abs(s["max_pre_drop_value"] - 0.8) < 1e-9
    assert s["ply_of_max_pre_drop"] == 19


def test_summarize_case_handles_empty_post_side():
    rows = [{"ply": 5, "root_black_value": 0.2}]
    s = summarize_case(rows, drop_ply=7)
    assert s["n_post"] == 0
    assert s["post_drop_mean"] == "" and s["drop_delta"] == ""
    assert abs(s["pre_drop_mean"] - 0.2) < 1e-9
