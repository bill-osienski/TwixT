import pytest

from scripts.GPU.alphazero.eval_loss_replay_analysis import (
    Thresholds, side_plies, validate_replay, value_features,
    confidence_features, opening_key,
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


# A-as-black, n_moves=12 -> A plies at global plies 1,3,5,7,9,11 (6 A plies).
TRAJ = [0.5, 0.125, -0.125, -0.375, -0.625, -1.0]
# deltas: -0.375, -0.25, -0.25, -0.25, -0.375 (all binary-exact)


def _a_plies(values, n_moves=12):
    _row, replay = make_game(0, a_is_black=True, n_moves=n_moves, a_values=values)
    return side_plies(replay, "black")


def test_value_features_medians_mean_min():
    f = value_features(_a_plies(TRAJ), 12, Thresholds())
    assert f["initial_a_value"] == 0.125          # median(0.5, 0.125, -0.125)
    assert f["final_a_value"] == -0.625           # median(-0.375, -0.625, -1.0)
    assert f["mean_a_value"] == -0.25             # sum = -1.5 over 6
    assert f["min_a_value"] == -1.0


def test_value_features_largest_drop_with_tie_takes_earliest():
    f = value_features(_a_plies(TRAJ), 12, Thresholds())
    # ties at -0.375 (a_ply 1 and 5): earliest wins
    assert f["largest_a_value_drop"] == -0.375
    assert f["largest_drop_a_ply"] == 1
    assert f["largest_drop_ply"] == 3
    assert f["largest_drop_fraction"] == pytest.approx(3 / 11)


def test_value_features_first_crossings():
    f = value_features(_a_plies(TRAJ), 12, Thresholds())
    assert (f["first_a_value_below_0_ply"], f["first_a_value_below_0_a_ply"]) == (5, 2)
    assert f["first_a_value_below_0_fraction"] == pytest.approx(5 / 11)
    assert (f["first_a_value_below_bad_ply"], f["first_a_value_below_bad_a_ply"]) == (7, 3)
    assert (f["first_a_value_below_lost_ply"], f["first_a_value_below_lost_a_ply"]) == (9, 4)
    assert f["first_a_value_below_lost_fraction"] == pytest.approx(9 / 11)


def test_value_features_never_crossed_is_none():
    f = value_features(_a_plies([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), 12, Thresholds())
    assert f["first_a_value_below_0_ply"] is None
    assert f["first_a_value_below_lost_fraction"] is None


def test_value_features_single_ply_has_null_drop():
    _row, replay = make_game(0, a_is_black=True, n_moves=2, a_values=[-0.5])
    f = value_features(side_plies(replay, "black"), 2, Thresholds())
    assert f["largest_a_value_drop"] is None
    assert f["largest_drop_ply"] is None
    assert f["initial_a_value"] == -0.5           # median of the single value
    assert f["first_a_value_below_lost_ply"] == 1


def test_confidence_features_post_opening_only():
    # opening_plies=4 -> A (black) post plies are global 5,7,9,11 (a_ply 2..5)
    _row, replay = make_game(
        0, a_is_black=True, n_moves=12, a_values=TRAJ,
        a_top1=[0.5, 0.5, 0.08, 0.12, 0.3, 0.05],
        a_rank=[4, 1, 6, 2, 1, 7])
    th = Thresholds(opening_plies=4)
    f = confidence_features(side_plies(replay, "black"), th)
    assert f["n_a_plies"] == 6
    assert f["n_a_plies_post"] == 4
    assert f["mean_top1_share_post"] == pytest.approx((0.08 + 0.12 + 0.3 + 0.05) / 4)
    assert f["min_top1_share_post"] == 0.05
    assert f["median_selected_visit_rank_post"] == 4.0   # median(6, 2, 1, 7)
    assert f["max_selected_visit_rank_post"] == 7
    assert f["low_confidence_ply_count"] == 2            # ranks 6 and 7 >= 5
    assert f["diffuse_ply_fraction"] == 0.5              # 0.08, 0.05 <= 0.10
    assert f["mean_selected_visit_share_post"] == 0.5    # 200/400 everywhere
    assert f["mean_n_legal"] == 100


def test_confidence_features_all_opening_yields_nulls():
    _row, replay = make_game(0, a_is_black=True, n_moves=6)
    th = Thresholds(opening_plies=20)  # whole game inside the opening window
    f = confidence_features(side_plies(replay, "black"), th)
    assert f["n_a_plies_post"] == 0
    assert f["mean_top1_share_post"] is None
    assert f["median_selected_visit_rank_post"] is None
    assert f["low_confidence_ply_count"] is None
    assert f["diffuse_ply_fraction"] is None
    assert f["mean_n_legal"] == 100                      # all-plies metric survives


def test_confidence_features_boundary_ply_is_post_opening():
    # a_is_black=False -> A (red) plays plies 0,2,4,6,8,10; opening_plies=4
    # means ply 4 itself is post-opening (>=). A `>` regression would yield 3.
    _row, replay = make_game(
        0, a_is_black=False, a_wins=True, n_moves=12, a_values=TRAJ,
        a_top1=[0.5, 0.5, 0.3, 0.2, 0.1, 0.05])
    f = confidence_features(side_plies(replay, "red"), Thresholds(opening_plies=4))
    assert f["n_a_plies_post"] == 4                      # plies 4, 6, 8, 10
    assert f["mean_top1_share_post"] == pytest.approx((0.3 + 0.2 + 0.1 + 0.05) / 4)


def test_opening_key_first_k_plies():
    _row, replay = make_game(0, n_moves=6)
    # fixture rows/cols default to the ply number
    assert opening_key(replay, 4) == "r0c0|r1c1|r2c2|r3c3"
    assert opening_key(replay, 2) == "r0c0|r1c1"
