import csv
import pytest

from scripts.GPU.alphazero.build_targeted_calibration_manifest import (
    UNIFIED_COLUMNS, correction_rows, assert_no_holdout_overlap, resolve_anchor_rows
)


def _rows(*labels):
    return [{"checkpoint": l, "case_id": f"c{i}"} for i, l in enumerate(labels)]


def test_resolve_exact_match():
    out = resolve_anchor_rows(_rows("0001", "0379", "0001"), "0001")
    assert [r["checkpoint"] for r in out] == ["0001", "0001"]


def test_resolve_unique_suffix_match():
    out = resolve_anchor_rows(_rows("alphazero-v2-calib020-from0409:0001", "x:0379"), "0001")
    assert [r["checkpoint"] for r in out] == ["alphazero-v2-calib020-from0409:0001"]


def test_resolve_ambiguous_suffix_raises():
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_anchor_rows(_rows("a:0001", "b:0001"), "0001")


def test_resolve_no_match_raises():
    with pytest.raises(ValueError, match="no checkpoint matches"):
        resolve_anchor_rows(_rows("0379", "0409"), "0001")


CORR_COLS = ["case_rank", "game_idx", "case_id", "replay_path", "position_ply",
             "drop_ply", "side_to_move", "a_color", "winner", "n_moves",
             "initial_a_value", "final_a_value", "largest_a_value_drop",
             "largest_drop_phase", "collapse_type"]


def _corr_row(game_idx, position_ply, case_rank=1):
    return {c: "" for c in CORR_COLS} | {
        "case_rank": case_rank, "game_idx": game_idx,
        "case_id": f"game_{game_idx:06d}_ply_{position_ply:03d}",
        "replay_path": f"logs/eval/replays/game_{game_idx:06d}.json",
        "position_ply": position_ply, "side_to_move": "black",
        "drop_ply": position_ply + 2, "largest_drop_phase": "post_opening",
        "collapse_type": "sharp_value_drop"}


def _write(tmp_path, name, cols, rows):
    p = tmp_path / name
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return p


def test_correction_rows_fields(tmp_path):
    p = _write(tmp_path, "train.csv", CORR_COLS, [_corr_row(637, 39), _corr_row(200, 41, 2)])
    rows = correction_rows(str(p), target=-0.35, weight=1.0)
    assert len(rows) == 2
    assert all(set(r) == set(UNIFIED_COLUMNS) for r in rows)
    assert rows[0]["tag"] == "black_predrop_correction"
    assert rows[0]["target_black_value"] == "-0.35"
    assert rows[0]["weight_scale"] == "1.0"
    assert rows[0]["replay_path"] == "logs/eval/replays/game_000637.json"
    assert rows[0]["side_to_move"] == "black"


def test_holdout_overlap_raises(tmp_path):
    corr = correction_rows(
        str(_write(tmp_path, "train.csv", CORR_COLS, [_corr_row(637, 39)])),
        target=-0.35, weight=1.0)
    holdout = _write(tmp_path, "frozen.csv", CORR_COLS, [_corr_row(637, 39)])  # same (path, ply)
    with pytest.raises(ValueError, match="leaks"):
        assert_no_holdout_overlap(corr, str(holdout))


def test_holdout_disjoint_ok(tmp_path):
    corr = correction_rows(
        str(_write(tmp_path, "train.csv", CORR_COLS, [_corr_row(637, 39)])),
        target=-0.35, weight=1.0)
    holdout = _write(tmp_path, "frozen.csv", CORR_COLS, [_corr_row(999, 39)])  # different game
    assert_no_holdout_overlap(corr, str(holdout))  # no raise


from scripts.GPU.alphazero.build_targeted_calibration_manifest import position_probe_retention_rows

PROBE_COLS = ["checkpoint", "game_idx", "case_id", "case_rank", "position_ply",
              "side_to_move", "probe_black_root_value", "probe_top1_share",
              "black_overvalue", "severe_black_overvalue", "replay_path", "drop_ply",
              "initial_a_value", "final_a_value", "largest_a_value_drop",
              "largest_drop_phase", "collapse_type"]


def _probe_row(ckpt, game_idx, value, side="red", position_ply=39):
    return {c: "" for c in PROBE_COLS} | {
        "checkpoint": ckpt, "game_idx": game_idx,
        "case_id": f"game_{game_idx:06d}_ply_{position_ply:03d}",
        "case_rank": 1, "position_ply": position_ply, "side_to_move": side,
        "probe_black_root_value": value,
        "replay_path": f"logs/eval/replays/game_{game_idx:06d}.json",
        "drop_ply": position_ply + 2, "largest_drop_phase": "post_opening",
        "collapse_type": "sharp_value_drop"}


def test_position_probe_retention_picks_anchor_only(tmp_path):
    p = _write(tmp_path, "red.csv", PROBE_COLS, [
        _probe_row("0001", 10, "-0.20"), _probe_row("0379", 10, "0.40"),
        _probe_row("0001", 11, "-0.10"), _probe_row("0409", 11, "0.50")])
    rows = position_probe_retention_rows(str(p), "0001", "red_predrop_retention", 0.5)
    assert [r["game_idx"] for r in rows] == ["10", "11"]
    assert rows[0]["tag"] == "red_predrop_retention"
    assert rows[0]["target_black_value"] == "-0.20"
    assert rows[0]["weight_scale"] == "0.5"
    assert rows[0]["anchor_checkpoint"] == "0001"
    assert rows[0]["replay_path"] == "logs/eval/replays/game_000010.json"
    assert rows[0]["side_to_move"] == "red"


def test_position_probe_retention_duplicate_case_id_raises(tmp_path):
    p = _write(tmp_path, "red.csv", PROBE_COLS,
               [_probe_row("0001", 10, "-0.20"), _probe_row("0001", 10, "-0.21")])
    with pytest.raises(ValueError, match="duplicate case_id"):
        position_probe_retention_rows(str(p), "0001", "red_predrop_retention", 0.5)
