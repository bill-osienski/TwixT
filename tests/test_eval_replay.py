import pytest

from scripts.GPU.alphazero.eval_replay import ply_record, REPLAY_SCHEMA_VERSION


def test_ply_record_fields():
    counts = {(4, 19): 124, (5, 5): 76, (1, 1): 200}
    rec = ply_record(0, "red", (4, 19), counts, root_value=0.12)
    assert rec == {
        "ply": 0, "player": "red", "row": 4, "col": 19,
        "root_value": 0.12,
        "root_top1_share": 200 / 400,
        "selected_visit_rank": 2,        # 200 > 124 > 76 -> (4,19) is rank 2
        "selected_visit_count": 124,
        "root_total_visits": 400,
        "n_legal": 3,
    }


def test_ply_record_rank_tiebreak_by_rowcol():
    # two moves tie at 100 visits; ascending (row,col) breaks the tie
    counts = {(2, 2): 100, (1, 9): 100, (0, 0): 50}
    # (1,9) and (2,2) tie at 100; (1,9) sorts before (2,2) -> ranks 1 and 2
    assert ply_record(0, "red", (1, 9), counts, 0.0)["selected_visit_rank"] == 1
    assert ply_record(0, "red", (2, 2), counts, 0.0)["selected_visit_rank"] == 2


def test_ply_record_top1_and_totals():
    counts = {(0, 0): 3, (0, 1): 7}
    rec = ply_record(5, "black", (0, 0), counts, -0.4)
    assert rec["root_total_visits"] == 10
    assert rec["root_top1_share"] == 0.7
    assert rec["selected_visit_count"] == 3
    assert rec["selected_visit_rank"] == 2


def test_ply_record_fails_on_empty_counts():
    with pytest.raises(ValueError, match="empty"):
        ply_record(0, "red", (4, 19), {}, 0.0)


def test_ply_record_fails_when_move_not_in_counts():
    with pytest.raises(ValueError, match="not in"):
        ply_record(0, "red", (9, 9), {(4, 19): 10}, 0.0)


def test_schema_version_is_one():
    assert REPLAY_SCHEMA_VERSION == 1
