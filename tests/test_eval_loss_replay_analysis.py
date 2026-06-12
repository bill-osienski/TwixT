import pytest

from scripts.GPU.alphazero.eval_loss_replay_analysis import (
    Thresholds, side_plies,
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
