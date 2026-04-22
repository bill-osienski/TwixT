"""Tests for the bootstrap probe suite generator.

Covers:
- CLI --help responds
- Deterministic byte-identical output on rerun
- No wall-clock fields in meta
- Only natural wins emitted
- Schema matches tests/probes/README.md expectations
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


def test_bootstrap_cli_help():
    """Bootstrap generator responds to --help."""
    result = subprocess.run(
        [".venv/bin/python", "scripts/build_bootstrap_probe_suite.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--out" in result.stdout
    assert "--source-iter-range" in result.stdout


def _write_fake_game(dir_path: Path, iteration: int, game_idx: int,
                     n_moves: int = 40, winner: str = "red",
                     reason: str = "win", board_size: int = 24):
    """Write a synthetic iter_NNNN_game_MMM.json matching the analyzer's
    replay format."""
    moves = [{"player": "red" if i % 2 == 0 else "black",
              "move": [(i * 7 + iteration * 3 + game_idx) % board_size,
                       (i * 11 + iteration * 5 + game_idx) % board_size]}
             for i in range(n_moves)]
    path = dir_path / f"iter_{iteration:04d}_game_{game_idx:03d}.json"
    path.write_text(json.dumps({
        "id": f"iter_{iteration:04d}_game_{game_idx:03d}",
        "meta": {"board_size": board_size, "iteration": iteration,
                 "game_idx": game_idx, "reason": reason, "n_moves": n_moves,
                 "starting_player": "red"},
        "moves": moves,
        "winner": winner,
        "starting_player": "red",
    }))
    return path


def test_bootstrap_deterministic_rerun(tmp_path):
    """Two consecutive runs with identical inputs produce byte-identical output."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    # 10 natural-win red games + 10 black at iter 30.
    for i in range(10):
        _write_fake_game(games_dir, iteration=30, game_idx=i, winner="red")
        _write_fake_game(games_dir, iteration=30, game_idx=100 + i, winner="black")

    out1 = tmp_path / "out1.json"
    out2 = tmp_path / "out2.json"
    for out in (out1, out2):
        result = subprocess.run(
            [".venv/bin/python", "scripts/build_bootstrap_probe_suite.py",
             "--input", str(games_dir),
             "--source-iter-range", "30", "30",
             "--out", str(out),
             "--max-probes", "20"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

    assert out1.read_bytes() == out2.read_bytes(), "byte-identity broken"


def test_bootstrap_no_wall_clock_fields(tmp_path):
    """Output meta contains no generated_at / timestamp / created_at etc."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(5):
        _write_fake_game(games_dir, iteration=30, game_idx=i, winner="red")
        _write_fake_game(games_dir, iteration=30, game_idx=100 + i, winner="black")

    out = tmp_path / "out.json"
    result = subprocess.run(
        [".venv/bin/python", "scripts/build_bootstrap_probe_suite.py",
         "--input", str(games_dir),
         "--source-iter-range", "30", "30",
         "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    meta = data.get("meta", {})
    for forbidden in ("generated_at", "timestamp", "created_at", "generation_time", "datetime"):
        assert forbidden not in meta, f"wall-clock field {forbidden!r} leaked into meta"


def test_bootstrap_only_natural_wins(tmp_path):
    """Resign/adjudicated/draw/timeout games produce zero probes."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    # Mix of natural-win and other termination reasons.
    for i in range(5):
        _write_fake_game(games_dir, iteration=30, game_idx=i, winner="red", reason="win")
    for i, bad in enumerate(("resign", "adjudicated", "timeout", "board_full")):
        _write_fake_game(games_dir, iteration=30, game_idx=200 + i,
                         winner="red", reason=bad)

    out = tmp_path / "out.json"
    subprocess.run(
        [".venv/bin/python", "scripts/build_bootstrap_probe_suite.py",
         "--input", str(games_dir),
         "--source-iter-range", "30", "30",
         "--out", str(out)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.read_text())
    sources = {p["source_game"] for p in data["probes"]}
    # Only natural-win source games (0..4) appear.
    for bad_idx in range(200, 204):
        assert f"iter_0030_game_{bad_idx:03d}" not in sources


def test_bootstrap_schema_fields(tmp_path):
    """Output conforms to tests/probes/README.md schema."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(10):
        _write_fake_game(games_dir, iteration=30, game_idx=i, winner="red")
        _write_fake_game(games_dir, iteration=30, game_idx=100 + i, winner="black")

    out = tmp_path / "out.json"
    subprocess.run(
        [".venv/bin/python", "scripts/build_bootstrap_probe_suite.py",
         "--input", str(games_dir),
         "--source-iter-range", "30", "30",
         "--out", str(out)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.read_text())
    assert "meta" in data and "probes" in data
    assert data["meta"]["type"] == "bootstrap_rule_selected"
    assert data["meta"]["not_gate_suite"] is True
    for p in data["probes"]:
        for required in ("id", "category", "confidence", "side_to_move",
                         "expected_value_sign", "active_size", "ply",
                         "move_history", "source_game", "source_ply"):
            assert required in p, f"probe missing {required!r}"
        assert p["confidence"] == "forced"
        assert p["active_size"] == 24
        assert p["category"] in ("near_win_red", "near_win_black")


def test_bootstrap_balance_ratio(tmp_path):
    """Majority class is capped to <= 2:1 vs minority class."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    # 50 red wins, 5 black wins - should truncate red to ~10 (2*5).
    for i in range(50):
        _write_fake_game(games_dir, iteration=30, game_idx=i, winner="red")
    for i in range(5):
        _write_fake_game(games_dir, iteration=30, game_idx=200 + i, winner="black")

    out = tmp_path / "out.json"
    subprocess.run(
        [".venv/bin/python", "scripts/build_bootstrap_probe_suite.py",
         "--input", str(games_dir),
         "--source-iter-range", "30", "30",
         "--out", str(out),
         "--max-probes", "100"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.read_text())
    red_count = sum(1 for p in data["probes"] if p["category"] == "near_win_red")
    black_count = sum(1 for p in data["probes"] if p["category"] == "near_win_black")
    assert red_count <= 2 * max(black_count, 1), (
        f"balance violated: red={red_count} black={black_count}"
    )


def test_bootstrap_balance_preserved_through_truncation(tmp_path):
    """Balance must hold even when max_probes forces truncation of a
    balanced pool — regression guard against the pre-fix behaviour where
    the ≤2:1 cap was applied before sort-and-truncate, so the truncation
    step could still skew the final subset toward one color when the most
    recent iters happened to favor it."""
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    # Balanced pool (10 red + 10 black at same iter / ply depth). Red game
    # basenames sort alphabetically before black (idx 0-9 vs 50-59), so a
    # naive sort-then-truncate would pick all red for the top max_probes.
    # Interleave-with-balance-cap must prevent that.
    for i in range(10):
        _write_fake_game(games_dir, iteration=30, game_idx=i, winner="red")
    for i in range(10):
        _write_fake_game(games_dir, iteration=30, game_idx=50 + i, winner="black")

    out = tmp_path / "out.json"
    subprocess.run(
        [".venv/bin/python", "scripts/build_bootstrap_probe_suite.py",
         "--input", str(games_dir),
         "--source-iter-range", "30", "30",
         "--out", str(out),
         "--max-probes", "6"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.read_text())
    red_count = sum(1 for p in data["probes"] if p["category"] == "near_win_red")
    black_count = sum(1 for p in data["probes"] if p["category"] == "near_win_black")
    assert red_count + black_count == 6, (
        f"expected 6 probes, got {red_count + black_count} (r={red_count} b={black_count})"
    )
    majority = max(red_count, black_count)
    minority = min(red_count, black_count)
    assert majority <= 2 * max(minority, 1), (
        f"balance violated under truncation: {red_count}r + {black_count}b "
        f"(majority={majority}, minority={minority})"
    )
