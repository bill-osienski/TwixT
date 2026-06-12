import json

import pytest

from scripts.GPU.alphazero.eval_loss_replay_analyzer import (
    parse_args, thresholds_from_args,
)
from tests.eval_replay_fixtures import A, B, make_game


def test_parse_args_defaults():
    args = parse_args(["--games-jsonl", "x_games.jsonl"])
    assert args.a_color == "black"
    assert (args.min_moves, args.max_moves) == (41, 80)
    assert args.opening_plies == 20 and args.opening_key_plies == 4
    assert args.review_queue == 50
    th = thresholds_from_args(args)
    assert th.bad_value == -0.25 and th.lost_value == -0.50
    assert th.sharp_drop == 0.40 and th.low_top1_share == 0.10
    assert th.low_visit_rank == 5 and th.opening_plies == 20


def test_parse_args_rejects_bad_value_not_above_lost_value():
    with pytest.raises(SystemExit) as e:
        parse_args(["--games-jsonl", "x", "--bad-value", "-0.6"])
    assert e.value.code == 2


def test_parse_args_rejects_nonpositive_sharp_drop():
    with pytest.raises(SystemExit) as e:
        parse_args(["--games-jsonl", "x", "--sharp-drop", "0"])
    assert e.value.code == 2
