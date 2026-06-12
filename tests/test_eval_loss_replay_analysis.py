import pytest

from scripts.GPU.alphazero.eval_loss_replay_analysis import (
    Thresholds, side_plies, validate_replay, value_features,
    confidence_features, opening_key, classify_collapse,
    game_features, b_side_features,
    cohort_comparison_row, phase_of, phase_bucket_rows,
    cohens_d, effect_sizes,
    collapse_distribution, timing_distribution, secondary_contrast_summary,
    make_verdict,
    review_queue_rows, opening_cluster_rows,
    build_replay_summary, MIN_WIN_COHORT, OPENING_SAMPLING_NOTE,
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


def _feats(**over):
    base = {
        "initial_a_value": 0.0, "final_a_value": 0.0,
        "largest_a_value_drop": -0.1,
        "mean_top1_share_post": 0.5, "diffuse_ply_fraction": 0.0,
        "median_selected_visit_rank_post": 1, "low_confidence_ply_count": 0,
    }
    base.update(over)
    return base


def test_classify_already_bad_at_boundary():
    label, flags = classify_collapse(_feats(initial_a_value=-0.25), Thresholds())
    assert label == "already_bad" and flags["flag_already_bad"]


def test_classify_sharp_drop_at_boundary():
    label, _ = classify_collapse(_feats(largest_a_value_drop=-0.40), Thresholds())
    assert label == "sharp_value_drop"


def test_classify_gradual_decay_requires_healthy_start_and_no_cliff():
    label, _ = classify_collapse(
        _feats(initial_a_value=0.0, final_a_value=-0.40,
               largest_a_value_drop=-0.39), Thresholds())
    assert label == "gradual_decay"


def test_gradual_flag_suppressed_by_sharp():
    label, flags = classify_collapse(
        _feats(initial_a_value=0.0, final_a_value=-0.5,
               largest_a_value_drop=-0.45), Thresholds())
    assert label == "sharp_value_drop"
    assert flags["flag_gradual"] is False        # spec: "and not sharp"


def test_classify_diffusion_mean_or_fraction():
    label, _ = classify_collapse(_feats(mean_top1_share_post=0.15), Thresholds())
    assert label == "search_diffusion"
    label, _ = classify_collapse(_feats(diffuse_ply_fraction=0.25), Thresholds())
    assert label == "search_diffusion"


def test_classify_low_visit_median_or_count():
    label, _ = classify_collapse(
        _feats(median_selected_visit_rank_post=3), Thresholds())
    assert label == "low_visit_selection"
    label, _ = classify_collapse(
        _feats(low_confidence_ply_count=3), Thresholds())
    assert label == "low_visit_selection"


def test_classify_precedence_already_bad_beats_sharp_but_keeps_flag():
    label, flags = classify_collapse(
        _feats(initial_a_value=-0.3, largest_a_value_drop=-0.5), Thresholds())
    assert label == "already_bad"
    assert flags["flag_sharp"] is True           # multi-signal stays visible


def test_classify_no_clear_signal():
    label, flags = classify_collapse(_feats(), Thresholds())
    assert label == "no_clear_signal"
    assert not any(flags.values())


def test_classify_null_post_features_disable_those_rules():
    label, flags = classify_collapse(
        _feats(mean_top1_share_post=None, diffuse_ply_fraction=None,
               median_selected_visit_rank_post=None,
               low_confidence_ply_count=None), Thresholds())
    assert label == "no_clear_signal"
    assert flags["flag_diffusion"] is False and flags["flag_low_visit"] is False


def test_classify_just_outside_boundaries_do_not_fire():
    _label, flags = classify_collapse(
        _feats(initial_a_value=-0.24, largest_a_value_drop=-0.39,
               final_a_value=-0.39, mean_top1_share_post=0.16,
               diffuse_ply_fraction=0.24,
               median_selected_visit_rank_post=2,
               low_confidence_ply_count=2), Thresholds())
    assert not any(flags.values())


def test_game_features_merges_identity_value_confidence():
    row, replay = make_game(7, a_is_black=True, n_moves=12, a_values=TRAJ)
    f = game_features(row, replay, "black", Thresholds(opening_plies=4),
                      key_plies=2)
    assert f["game_idx"] == 7 and f["a_color"] == "black"
    assert f["replay_path"] == row["replay_path"]
    assert f["opening_key"] == "r0c0|r1c1"
    assert f["initial_a_value"] == 0.125          # value features present
    assert f["n_a_plies_post"] == 4               # confidence features present


def test_b_side_features_onsets_and_saw_it_first():
    # B is red: B plies at global 0,2,4,6,8,10. B's values rise to a win.
    _row, replay = make_game(
        0, a_is_black=True, n_moves=12,
        b_values=[0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
    th = Thresholds(opening_plies=4)
    f = b_side_features(replay, "red", th, a_first_below_lost_fraction=9 / 11)
    assert f["b_first_value_above_025_ply"] == 4         # 0.25 >= 0.25
    assert f["b_first_value_above_050_ply"] == 6         # 0.5 >= 0.50
    assert f["b_first_value_above_050_fraction"] == pytest.approx(6 / 11)
    assert f["b_saw_it_first"] is True                   # 6/11 < 9/11
    assert f["b_mean_value"] == pytest.approx((0.0 + 0.125 + 0.25 + 0.5 + 0.75 + 1.0) / 6)
    assert f["b_mean_top1_share_post"] == 0.5            # fixture default
    assert f["b_median_visit_rank_post"] == 1


def test_b_saw_it_first_false_when_either_onset_missing():
    _row, replay = make_game(0, a_is_black=True, n_moves=12)  # flat 0.0: no onset
    f = b_side_features(replay, "red", Thresholds(), a_first_below_lost_fraction=0.5)
    assert f["b_first_value_above_050_ply"] is None
    assert f["b_saw_it_first"] is False
    f2 = b_side_features(replay, "red", Thresholds(), a_first_below_lost_fraction=None)
    assert f2["b_saw_it_first"] is False


def test_cohort_comparison_row_pools_plies_across_games():
    th = Thresholds(opening_plies=4)
    games = []
    for i, vals in enumerate(([0.5, 0.25, -0.25, -0.5, -0.75, -1.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        _row, replay = make_game(i, a_is_black=True, n_moves=12, a_values=vals)
        games.append(side_plies(replay, "black"))
    r = cohort_comparison_row("loss", games, th.opening_plies)
    assert r["cohort"] == "loss" and r["games"] == 2 and r["plies"] == 12
    assert r["mean_root_value"] == pytest.approx(-1.75 / 12)
    assert r["mean_n_legal"] == 100
    # post pool: 4 post plies per game = 8
    assert r["mean_top1_share_post"] == 0.5


def test_phase_of_boundaries():
    # opening_plies=20, n_moves=80: post-opening span is plies 20..79
    assert phase_of(19, 80, 20) == "opening"
    assert phase_of(20, 80, 20) == "early_midgame"
    assert phase_of(34, 80, 20) == "early_midgame"   # f = 14/60 < 0.25
    assert phase_of(35, 80, 20) == "midgame"         # f = 15/60 = 0.25
    assert phase_of(79, 80, 20) == "pre_terminal"
    assert phase_of(40, 41, 20) == "pre_terminal"    # short game, last ply


def test_phase_bucket_rows_labels_opening_as_temperature():
    _row, replay = make_game(0, a_is_black=True, n_moves=12, a_values=TRAJ)
    rows = phase_bucket_rows("loss", [(side_plies(replay, "black"), 12)], 4)
    by_phase = {r["phase"]: r for r in rows}
    assert by_phase["opening"]["sampling"] == "temperature"
    assert all(r["sampling"] == "argmax" for p, r in by_phase.items()
               if p != "opening")
    assert by_phase["opening"]["plies"] == 2          # A plies 1, 3
    assert sum(r["plies"] for r in rows) == 6
    assert all(r["games"] == 1 for r in rows)
    assert "mean_root_value" in rows[0] and "median_selected_visit_rank" in rows[0]


def test_cohens_d_hand_computed():
    # means 2 vs 4, each var 1 (ddof=1), pooled sd 1 -> d = -2.0
    assert cohens_d([1, 2, 3], [3, 4, 5]) == pytest.approx(-2.0)


def test_cohens_d_degenerate_and_short():
    assert cohens_d([1.0, 1.0], [1.0, 1.0]) is None   # zero pooled variance
    assert cohens_d([1.0], [1.0, 2.0]) is None        # too few samples


def test_effect_sizes_sign_convention_and_nulls():
    loss = [{"final_a_value": -0.9, "largest_a_value_drop": -0.5,
             "initial_a_value": 0.0, "mean_top1_share_post": 0.2,
             "median_selected_visit_rank_post": 3},
            {"final_a_value": -0.7, "largest_a_value_drop": -0.4,
             "initial_a_value": 0.1, "mean_top1_share_post": 0.3,
             "median_selected_visit_rank_post": 2}]
    win = [{"final_a_value": 0.8, "largest_a_value_drop": -0.1,
            "initial_a_value": 0.1, "mean_top1_share_post": 0.5,
            "median_selected_visit_rank_post": 1},
           {"final_a_value": 0.6, "largest_a_value_drop": -0.2,
            "initial_a_value": 0.2, "mean_top1_share_post": 0.4,
            "median_selected_visit_rank_post": 1}]
    out = effect_sizes(loss, win)
    m = out["metrics"]
    assert "cohens_d" in out["formula"]
    assert m["final_a_value"]["d"] < 0                # lower in losses
    assert m["final_a_value"]["delta"] == pytest.approx(-0.8 - 0.7)
    assert m["median_selected_visit_rank_post"]["d"] > 0   # higher rank in losses
    # a metric that is all-None in one cohort yields nulls, not a crash
    for f in win:
        f["mean_top1_share_post"] = None
    out2 = effect_sizes(loss, win)
    assert out2["metrics"]["mean_top1_share_post"]["d"] is None
    assert out2["metrics"]["mean_top1_share_post"]["win_mean"] is None


def test_collapse_distribution_groups_failure_modes():
    labels = (["sharp_value_drop"] * 5 + ["gradual_decay"] * 2
              + ["search_diffusion"] * 2 + ["no_clear_signal"])
    d = collapse_distribution(labels)
    assert d["n"] == 10
    assert d["counts"]["sharp_value_drop"] == 5
    assert d["mode_shares"]["value-drop"] == pytest.approx(0.7)
    assert d["mode_shares"]["diffusion"] == pytest.approx(0.2)
    assert d["mode_shares"]["unexplained"] == pytest.approx(0.1)
    assert d["mode_shares"]["already-losing"] == 0.0


def test_timing_distribution_percentiles_and_never():
    feats = [{"first_a_value_below_0_fraction": x,
              "first_a_value_below_bad_fraction": None,
              "first_a_value_below_lost_fraction": None,
              "largest_drop_fraction": x}
             for x in (0.2, 0.4, 0.6)]
    feats.append({"first_a_value_below_0_fraction": None,
                  "first_a_value_below_bad_fraction": None,
                  "first_a_value_below_lost_fraction": None,
                  "largest_drop_fraction": None})
    t = timing_distribution(feats)
    assert t["first_a_value_below_0"]["p50"] == pytest.approx(0.4)
    assert t["first_a_value_below_0"]["p25"] == pytest.approx(0.3)
    assert t["first_a_value_below_0"]["never"] == 1
    assert t["first_a_value_below_lost"]["p50"] is None
    assert t["first_a_value_below_lost"]["never"] == 4
    assert t["largest_drop"]["p75"] == pytest.approx(0.5)


def test_secondary_contrast_summary_gap_and_share():
    f1 = {"mean_a_value": -0.5, "b_mean_value": 0.5,
          "mean_top1_share_post": 0.2, "b_mean_top1_share_post": 0.6,
          "median_selected_visit_rank_post": 3, "b_median_visit_rank_post": 1,
          "first_a_value_below_lost_fraction": 0.5,
          "b_first_value_above_050_fraction": 0.3, "b_saw_it_first": True}
    f2 = {"mean_a_value": -0.25, "b_mean_value": 0.25,
          "mean_top1_share_post": 0.4, "b_mean_top1_share_post": 0.5,
          "median_selected_visit_rank_post": 1, "b_median_visit_rank_post": 1,
          "first_a_value_below_lost_fraction": 0.6,
          "b_first_value_above_050_fraction": None, "b_saw_it_first": False}
    s = secondary_contrast_summary([f1, f2])
    assert s["games"] == 2
    assert s["b_saw_it_first_share"] == 0.5
    assert s["onset_gap_games"] == 1
    assert s["median_onset_gap_fraction"] == pytest.approx(0.2)   # 0.5 - 0.3
    assert s["a_mean_value"] == pytest.approx(-0.375)
    assert s["b_mean_value"] == pytest.approx(0.375)


def test_verdict_primary_and_secondary():
    labels = (["sharp_value_drop"] * 5 + ["gradual_decay"] * 2
              + ["search_diffusion"] * 2 + ["no_clear_signal"])
    v = make_verdict(labels, "A-as-black 41-80")
    assert v["primary"] == "value-drop"
    assert v["primary_share"] == pytest.approx(0.7)
    assert v["secondary"] == "diffusion"
    assert v["secondary_share"] == pytest.approx(0.2)
    assert "value-drop" in v["narrative"] and "A-as-black 41-80" in v["narrative"]


def test_verdict_mixed_when_no_mode_reaches_bar():
    labels = (["sharp_value_drop"] * 3 + ["search_diffusion"] * 3
              + ["low_visit_selection"] * 2 + ["already_bad"] * 2)
    v = make_verdict(labels, "X")
    assert v["primary"] == "mixed / no strong single signal"
    assert v["secondary"] is None


def test_verdict_mixed_when_unexplained_dominates():
    labels = ["no_clear_signal"] * 6 + ["sharp_value_drop"] * 4
    v = make_verdict(labels, "X")   # value-drop 0.4 >= bar, but unexplained 0.6 wins
    assert v["primary"] == "mixed / no strong single signal"


def test_verdict_no_secondary_below_bar():
    labels = ["sharp_value_drop"] * 8 + ["search_diffusion"] * 1 + ["no_clear_signal"]
    v = make_verdict(labels, "X")
    assert v["primary"] == "value-drop" and v["secondary"] is None


def _queue_feat(idx, drop, final, top1=0.5, rank=1):
    return {"game_idx": idx, "task_id": idx, "replay_path": f"r/{idx}.json",
            "a_color": "black", "winner": "red", "n_moves": 50,
            "collapse_type": "sharp_value_drop",
            "initial_a_value": 0.1, "final_a_value": final,
            "largest_a_value_drop": drop, "largest_drop_ply": 30,
            "largest_drop_fraction": 0.6,
            "first_a_value_below_lost_ply": 35,
            "first_a_value_below_lost_fraction": 0.7,
            "mean_top1_share_post": top1,
            "median_selected_visit_rank_post": rank, "opening_key": "k"}


def test_review_queue_composite_sort_and_limit():
    feats = [
        _queue_feat(1, -0.5, -0.2),          # mid drop, better final
        _queue_feat(2, -0.8, -0.9),          # sharpest drop -> rank 1
        _queue_feat(3, -0.5, -0.9),          # tie on drop -> worse final first
        _queue_feat(4, -0.1, -0.1),
    ]
    rows = review_queue_rows(feats, limit=3)
    assert [r["game_idx"] for r in rows] == [2, 3, 1]
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[0]["initial_a_value"] == 0.1            # spec: queue carries both
    assert rows[0]["final_a_value"] == -0.9
    assert "flag_sharp" not in rows[0]                  # queue is the curated view


def test_review_queue_null_drop_sorts_last():
    feats = [_queue_feat(1, None, -0.9), _queue_feat(2, -0.3, -0.1)]
    rows = review_queue_rows(feats, limit=10)
    assert [r["game_idx"] for r in rows] == [2, 1]


def test_opening_cluster_rows_grouping_and_sort():
    g0_row, g0 = make_game(0, a_is_black=True, a_wins=False, n_moves=12)
    g1_row, g1 = make_game(1, a_is_black=True, a_wins=True, n_moves=12)
    g2_row, g2 = make_game(2, a_is_black=True, a_wins=False, n_moves=14)
    for m in g2["moves"][:2]:
        m["row"], m["col"] = 9, 9           # distinct opening key
    rows = opening_cluster_rows(
        [(g0, "black", False), (g1, "black", True), (g2, "black", False)],
        key_plies=2, cohort_label="A_black_41_80_decisive", opening_plies=4)
    assert rows[0]["games"] == 2            # the shared key sorts first
    assert rows[0]["wins"] == 1 and rows[0]["losses"] == 1
    assert rows[0]["a_score_rate"] == 0.5
    assert rows[0]["cohort"] == "A_black_41_80_decisive"
    assert rows[0]["opening_plies"] == 2
    assert rows[1]["games"] == 1 and rows[1]["opening_key"] == "r9c9|r9c9"
    assert rows[1]["avg_moves"] == 14


def _summary_inputs(n_wins):
    # Values vary per game: identical dicts would give zero variance and a
    # (correctly) null Cohen's d, which is not what this test exercises.
    loss = [{"collapse_type": "sharp_value_drop",
             "final_a_value": -0.9 + 0.05 * i,
             "largest_a_value_drop": -0.6 - 0.01 * i,
             "initial_a_value": 0.0 + 0.01 * i,
             "mean_top1_share_post": 0.3 + 0.01 * i,
             "median_selected_visit_rank_post": 2 + (i % 2),
             "first_a_value_below_0_fraction": 0.4,
             "first_a_value_below_bad_fraction": 0.5,
             "first_a_value_below_lost_fraction": 0.6,
             "largest_drop_fraction": 0.55, "mean_a_value": -0.4,
             "b_mean_value": 0.4, "b_mean_top1_share_post": 0.5,
             "b_median_visit_rank_post": 1,
             "b_first_value_above_050_fraction": 0.5,
             "b_saw_it_first": True} for i in range(6)]
    win = [{"collapse_type": "no_clear_signal",
            "final_a_value": 0.8 - 0.05 * i,
            "largest_a_value_drop": -0.1 - 0.01 * i,
            "initial_a_value": 0.1 + 0.01 * i,
            "mean_top1_share_post": 0.5 - 0.01 * i,
            "median_selected_visit_rank_post": 1,
            "first_a_value_below_0_fraction": None,
            "first_a_value_below_bad_fraction": None,
            "first_a_value_below_lost_fraction": None,
            "largest_drop_fraction": 0.3, "mean_a_value": 0.5}
           for i in range(n_wins)]
    return loss, win


def _build(loss, win):
    return build_replay_summary(
        match="m", pairing_id="0399_vs_0379", a_ckpt=A, b_ckpt=B,
        filters={"a_color": "black"}, counts={"loss": len(loss), "win": len(win)},
        loss_feats=loss, win_feats=win,
        verdict=make_verdict([f["collapse_type"] for f in loss], "A-as-black"),
        cohort_rows=[{"cohort": "loss"}, {"cohort": "win"}],
        secondary=secondary_contrast_summary(loss))


def test_build_replay_summary_full_shape():
    loss, win = _summary_inputs(n_wins=6)
    s = _build(loss, win)
    assert s["match"] == "m" and s["a_checkpoint"] == A
    assert OPENING_SAMPLING_NOTE in s["notes"]
    assert s["primary_contrast"]["effect_sizes"]["metrics"]["final_a_value"]["d"] is not None
    assert s["primary_contrast"]["note"] is None
    assert s["collapse_type_distribution"]["mode_shares"]["value-drop"] == 1.0
    assert s["timing_distribution"]["first_a_value_below_lost"]["p50"] == pytest.approx(0.6)
    assert s["verdict"]["primary"] == "value-drop"
    assert s["secondary_contrast"]["b_saw_it_first_share"] == 1.0


def test_build_replay_summary_insufficient_contrast():
    loss, win = _summary_inputs(n_wins=MIN_WIN_COHORT - 1)
    s = _build(loss, win)
    assert s["primary_contrast"]["effect_sizes"] is None
    assert s["primary_contrast"]["note"] == "insufficient_contrast"
    assert s["verdict"]["primary"] == "value-drop"   # verdict still computed


def test_build_replay_summary_exactly_min_win_cohort_is_sufficient():
    loss, win = _summary_inputs(n_wins=MIN_WIN_COHORT)   # boundary: < not <=
    s = _build(loss, win)
    assert s["primary_contrast"]["effect_sizes"] is not None
    assert s["primary_contrast"]["note"] is None
