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


from dataclasses import dataclass

from scripts.GPU.alphazero.eval_replay import build_replay_dict, replay_filename


@dataclass
class _FakeResult:
    pairing_id: str
    game_idx: int
    task_id: int
    red_checkpoint: str
    black_checkpoint: str
    winner: str
    winner_checkpoint: str
    reason: str
    n_moves: int


def test_build_replay_dict_shape():
    result = _FakeResult("0399_vs_0379", 3, 7, "A.safetensors", "B.safetensors",
                         "red", "A.safetensors", "win", 2)
    records = [
        {"ply": 0, "player": "red", "row": 4, "col": 19, "root_value": 0.1,
         "root_top1_share": 0.5, "selected_visit_rank": 1,
         "selected_visit_count": 5, "root_total_visits": 10, "n_legal": 3},
        {"ply": 1, "player": "black", "row": 1, "col": 1, "root_value": -0.1,
         "root_top1_share": 0.6, "selected_visit_rank": 1,
         "selected_visit_count": 6, "root_total_visits": 10, "n_legal": 2},
    ]
    d = build_replay_dict(result, seed=35791, board_size=24, records=records)
    assert d == {
        "schema_version": 1,
        "pairing_id": "0399_vs_0379",
        "game_idx": 3, "task_id": 7, "seed": 35791, "board_size": 24,
        "red_checkpoint": "A.safetensors", "black_checkpoint": "B.safetensors",
        "winner": "red", "winner_checkpoint": "A.safetensors",
        "reason": "win", "n_moves": 2,
        "moves": records,
    }


def test_replay_filename_zero_padded():
    assert replay_filename(0) == "game_000000.json"
    assert replay_filename(42) == "game_000042.json"
    assert replay_filename(123456) == "game_123456.json"


import json as _json
import os as _os

from scripts.GPU.alphazero.eval_replay import write_replay


def test_write_replay_roundtrip_and_relative_path(tmp_path):
    replay_dir = tmp_path / "m_replays"
    d = {"schema_version": 1, "game_idx": 5, "moves": []}
    path = write_replay(str(replay_dir), d)
    # returns a path relative to CWD, not absolute
    assert not _os.path.isabs(path)
    # file exists where expected and round-trips
    abs_path = replay_dir / "game_000005.json"
    assert abs_path.exists()
    assert _json.loads(abs_path.read_text()) == d


def test_write_replay_creates_dir_idempotently(tmp_path):
    replay_dir = tmp_path / "nested" / "replays"
    write_replay(str(replay_dir), {"game_idx": 0, "moves": []})
    # second write into the same (now-existing) dir must not raise
    write_replay(str(replay_dir), {"game_idx": 1, "moves": []})
    assert (replay_dir / "game_000000.json").exists()
    assert (replay_dir / "game_000001.json").exists()
