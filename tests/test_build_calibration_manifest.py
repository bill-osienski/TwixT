import csv
from scripts.GPU.alphazero.build_calibration_manifest import (
    derive_case, select_calibration_cases, load_holdout_game_idxs, main, OUTPUT_COLUMNS,
)

QUEUE_COLS = [
    "rank", "game_idx", "task_id", "replay_path", "a_color", "winner",
    "n_moves", "collapse_type", "initial_a_value", "final_a_value",
    "largest_a_value_drop", "largest_drop_ply", "largest_drop_fraction",
    "largest_drop_phase", "first_a_value_below_lost_ply",
    "first_a_value_below_lost_fraction", "mean_top1_share_post",
    "median_selected_visit_rank_post", "opening_key",
]


def _qrow(game_idx, drop_ply=41, collapse="sharp_value_drop",
          phase="post_opening", a_color="black", winner="red", rank=1):
    return {
        "rank": rank, "game_idx": game_idx, "task_id": game_idx,
        "replay_path": f"logs/eval/replays/game_{game_idx:06d}.json",
        "a_color": a_color, "winner": winner, "n_moves": 51,
        "collapse_type": collapse, "initial_a_value": 0.07,
        "final_a_value": -0.95, "largest_a_value_drop": -1.78,
        "largest_drop_ply": drop_ply, "largest_drop_fraction": 0.82,
        "largest_drop_phase": phase, "first_a_value_below_lost_ply": drop_ply,
        "first_a_value_below_lost_fraction": 0.82, "mean_top1_share_post": 0.45,
        "median_selected_visit_rank_post": 1, "opening_key": "r11c9",
    }


def test_derive_case_computes_position_ply_and_case_id():
    case = derive_case(_qrow(637, drop_ply=41), case_rank=1)
    assert case["game_idx"] == 637
    assert case["position_ply"] == 39          # drop_ply - 2
    assert case["drop_ply"] == 41
    assert case["side_to_move"] == "black"
    assert case["case_id"] == "game_000637_ply_039"
    assert case["case_rank"] == 1
    assert case["source_rank"] == 1            # original review-queue rank


def test_select_excludes_holdout_and_nonmatching():
    rows = [
        _qrow(637, rank=1),                                   # keep
        _qrow(277, rank=2),                                   # holdout -> drop
        _qrow(100, rank=3, collapse="gradual_decay"),         # wrong collapse -> drop
        _qrow(101, rank=4, phase="opening"),                  # wrong phase -> drop
        _qrow(102, rank=5, a_color="red"),                    # wrong color -> drop
        _qrow(103, rank=6, winner="black"),                   # wrong winner -> drop
        _qrow(200, rank=7),                                   # keep
        _qrow(999, rank=8, drop_ply=1),                       # position_ply = -1 -> drop
    ]
    cases = select_calibration_cases(rows, holdout_game_idxs={277})
    assert [c["game_idx"] for c in cases] == [637, 200]
    assert [c["case_rank"] for c in cases] == [1, 2]          # re-ranked 1..N
    assert [c["source_rank"] for c in cases] == [1, 7]        # original queue ranks


def test_main_writes_manifest_excluding_holdout(tmp_path):
    queue = tmp_path / "queue.csv"
    with queue.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QUEUE_COLS)
        w.writeheader()
        w.writerow(_qrow(637, rank=1))
        w.writerow(_qrow(277, rank=2))
        w.writerow(_qrow(200, rank=3))
    holdout = tmp_path / "frozen.csv"
    with holdout.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game_idx"])
        w.writeheader()
        w.writerow({"game_idx": 277})
    out = tmp_path / "train.csv"
    rc = main(["--queue", str(queue), "--holdout-manifest", str(holdout),
               "--out", str(out)])
    assert rc == 0
    with out.open(newline="") as f:
        assert f.readline().rstrip("\r\n") == ",".join(OUTPUT_COLUMNS)
    rows = list(csv.DictReader(out.open()))
    assert [int(r["game_idx"]) for r in rows] == [637, 200]
    assert load_holdout_game_idxs(holdout) == {277}
    # determinism: re-run yields identical bytes
    first = out.read_bytes()
    main(["--queue", str(queue), "--holdout-manifest", str(holdout),
          "--out", str(out)])
    assert out.read_bytes() == first
