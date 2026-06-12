import pytest

from scripts.GPU.alphazero.eval_loss_replay_analysis import (
    Thresholds, side_plies, validate_replay,
)
from tests.eval_replay_fixtures import A, B, make_game


def test_thresholds_defaults_match_spec():
    th = Thresholds()
    assert th.bad_value == -0.25
    assert th.lost_value == -0.50
    assert th.sharp_drop == 0.40
    assert th.low_top1_share == 0.10
    assert th.low_visit_rank == 5
    assert th.opening_plies == 20


def test_side_plies_filters_one_side_in_order():
    _row, replay = make_game(0, a_is_black=True, n_moves=6)
    black = side_plies(replay, "black")
    red = side_plies(replay, "red")
    assert [m["ply"] for m in black] == [1, 3, 5]
    assert [m["ply"] for m in red] == [0, 2, 4]
    assert all(m["player"] == "black" for m in black)


def test_fixture_seats_a_by_color():
    row_b, _ = make_game(0, a_is_black=True)
    assert row_b["black_checkpoint"] == A and row_b["red_checkpoint"] == B
    row_r, _ = make_game(1, a_is_black=False, a_wins=True)
    assert row_r["red_checkpoint"] == A and row_r["winner"] == "red"


def test_validate_replay_accepts_consistent_pair():
    row, replay = make_game(3, n_moves=8)
    validate_replay(row, replay)  # no raise


def test_validate_replay_rejects_wrong_schema_version():
    row, replay = make_game(0)
    replay["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        validate_replay(row, replay)


def test_validate_replay_rejects_identity_mismatch():
    row, replay = make_game(0)
    replay["winner"] = "red" if replay["winner"] == "black" else "black"
    with pytest.raises(ValueError, match="winner"):
        validate_replay(row, replay)


def test_validate_replay_rejects_winner_checkpoint_mismatch():
    row, replay = make_game(0)
    replay["winner_checkpoint"] = "ckpts/other.safetensors"
    with pytest.raises(ValueError, match="winner_checkpoint"):
        validate_replay(row, replay)


def test_validate_replay_rejects_move_count_mismatch():
    row, replay = make_game(0, n_moves=10)
    replay["moves"] = replay["moves"][:-1]
    replay["n_moves"] = 10  # identity still matches the row
    with pytest.raises(ValueError, match="move records"):
        validate_replay(row, replay)


def test_validate_replay_rejects_broken_alternation():
    row, replay = make_game(0, n_moves=6)
    replay["moves"][2]["player"] = "black"  # ply 2 must be red
    with pytest.raises(ValueError, match="player"):
        validate_replay(row, replay)


def test_validate_replay_rejects_bad_ply_field():
    row, replay = make_game(0, n_moves=6)
    replay["moves"][4]["ply"] = 99
    with pytest.raises(ValueError, match="ply field"):
        validate_replay(row, replay)


def test_validate_replay_rejects_missing_ply_key():
    row, replay = make_game(0, n_moves=6)
    del replay["moves"][1]["root_value"]
    with pytest.raises(ValueError, match="missing keys"):
        validate_replay(row, replay)
