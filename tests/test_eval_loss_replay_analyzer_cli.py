import json

import pytest

from scripts.GPU.alphazero.eval_loss_replay_analyzer import (
    main, parse_args, thresholds_from_args,
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


def _write_capture(tmp_path, games):
    """Write a games.jsonl + sidecars for (row, replay) pairs; returns jsonl path."""
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir(exist_ok=True)
    jsonl = tmp_path / "synth_games.jsonl"
    with jsonl.open("w") as fh:
        for row, replay in games:
            row = dict(row)
            path = replay_dir / f"game_{row['game_idx']:06d}.json"
            path.write_text(json.dumps(replay))
            row["replay_path"] = str(path)
            fh.write(json.dumps(row) + "\n")
    return jsonl


def _synth_games():
    """6 A-black losses + 6 A-black wins in a 41-80 window, 1 draw, 1 short."""
    games = []
    losing = [0.25] * 10 + [-0.125, -0.375, -0.625, -0.75, -0.875,
                            -0.875, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    winning = [0.25] * 10 + [0.375, 0.5, 0.5, 0.625, 0.625,
                             0.75, 0.75, 0.875, 0.875, 1.0, 1.0, 1.0]
    rising = [0.0] * 10 + [0.125, 0.25, 0.375, 0.5, 0.625, 0.75,
                           0.875, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(6):
        games.append(make_game(i, a_is_black=True, a_wins=False, n_moves=44,
                               a_values=losing, b_values=rising))
    for i in range(6, 12):
        games.append(make_game(i, a_is_black=True, a_wins=True, n_moves=44,
                               a_values=winning))
    games.append(make_game(12, a_is_black=True, reason="state_cap", n_moves=44))
    games.append(make_game(13, a_is_black=True, a_wins=False, n_moves=30,
                           a_values=[0.0] * 15))
    return games


def test_cli_end_to_end_writes_all_artifacts(tmp_path, capsys):
    jsonl = _write_capture(tmp_path, _synth_games())
    out = tmp_path / "out"
    rc = main(["--games-jsonl", str(jsonl), "--output-dir", str(out)])
    assert rc == 0
    stem = "synth"
    for suffix in ("replay_summary.json", "cohort_comparison.csv",
                   "phase_buckets.csv", "collapse_timing.csv",
                   "manual_review_queue.csv", "opening_clusters.csv"):
        assert (out / f"{stem}_{suffix}").exists(), suffix
    s = json.loads((out / f"{stem}_replay_summary.json").read_text())
    assert s["cohorts"] == {"focus_window_games": 13, "excluded_draws": 1,
                            "loss": 6, "win": 6}
    assert s["primary_contrast"]["effect_sizes"] is not None   # 6 wins >= 5
    assert s["verdict"]["primary"] == "value-drop"
    timing = (out / f"{stem}_collapse_timing.csv").read_text().splitlines()
    assert len(timing) == 13                                   # header + 12 games
    header = timing[0].split(",")
    assert "collapse_type" in header and "flag_sharp" in header
    assert "b_saw_it_first" in header and "cohort" in header
    console = capsys.readouterr().out
    assert "Phase B verdict:" in console
    assert "manual_review_queue.csv" in console


def test_cli_skips_v1_era_file_without_replay_path(tmp_path, capsys):
    jsonl = tmp_path / "old_games.jsonl"
    with jsonl.open("w") as fh:
        for row, _replay in _synth_games():
            row = dict(row)
            del row["replay_path"]
            fh.write(json.dumps(row) + "\n")
    rc = main(["--games-jsonl", str(jsonl), "--output-dir", str(tmp_path / "o")])
    assert rc == 0
    assert "no replay capture" in capsys.readouterr().out
    assert not (tmp_path / "o").exists()


def test_cli_null_replay_path_in_focus_window_raises(tmp_path):
    games = _synth_games()
    jsonl = _write_capture(tmp_path, games)
    rows = [json.loads(l) for l in jsonl.read_text().splitlines()]
    rows[0]["replay_path"] = None
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="replay_path"):
        main(["--games-jsonl", str(jsonl), "--output-dir", str(tmp_path / "o")])


def test_cli_empty_loss_cohort_raises(tmp_path):
    games = [g for g in _synth_games() if g[0]["winner_checkpoint"] == A
             or g[0]["winner"] is None]
    jsonl = _write_capture(tmp_path, games)
    with pytest.raises(ValueError, match="nothing to explain"):
        main(["--games-jsonl", str(jsonl), "--output-dir", str(tmp_path / "o")])
