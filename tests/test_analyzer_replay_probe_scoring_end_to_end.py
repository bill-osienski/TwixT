"""End-to-end tests for the analyzer's new replay_probe_scoring and
value_calibration integrations, plus the checkpoint auto-discovery helper.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _write_fake_replay(dir_path: Path, iteration: int, game_idx: int,
                      n_moves: int = 30, winner: str = "red", reason: str = "win"):
    # Generate legal, non-overlapping moves on a 24x24 board. Reds must avoid
    # cols {0, 23} (and all corners); blacks must avoid rows {0, 23}. We
    # carve out disjoint interior regions for each color, offset by game_idx
    # so different games produce distinct move histories (keeps dedup from
    # collapsing them to a single probe).
    moves = []
    for i in range(n_moves):
        if i % 2 == 0:  # red
            idx = i // 2
            r = 2 + (idx + game_idx) % 10  # rows 2..11
            c = 2 + (idx * 3 + game_idx * 7) % 20  # cols 2..21 (avoid 0, 23)
        else:  # black
            idx = i // 2
            r = 14 + (idx + game_idx) % 8  # rows 14..21 (avoid 0, 23)
            c = 1 + (idx * 3 + game_idx * 7) % 22  # cols 1..22
        moves.append({"player": "red" if i % 2 == 0 else "black",
                      "move": [r, c]})
    path = dir_path / f"iter_{iteration:04d}_game_{game_idx:03d}.json"
    path.write_text(json.dumps({
        "id": f"iter_{iteration:04d}_game_{game_idx:03d}",
        "meta": {"board_size": 24, "iteration": iteration, "game_idx": game_idx,
                 "reason": reason, "n_moves": n_moves, "starting_player": "red"},
        "moves": moves,
        "winner": winner,
        "starting_player": "red",
    }))
    return path


def _write_fake_checkpoint(dir_path: Path, iteration: int, in_channels: int = 30):
    # Use create_network canonical defaults (hidden=128, n_blocks=6) so
    # probe_eval.load_network_for_scoring (which also uses those defaults)
    # can load the checkpoint without shape mismatches.
    from scripts.GPU.alphazero.network import create_network
    net = create_network(in_channels=in_channels)
    path = dir_path / f"model_iter_{iteration:04d}.safetensors"
    net.save_weights(str(path))
    return path


# ---------- Checkpoint auto-discovery helper ----------

def test_resolve_checkpoint_explicit_weights_wins(tmp_path):
    """When --weights is passed, it takes precedence over auto-discovery."""
    from scripts.twixt_replay_analyzer import _resolve_checkpoint_path
    explicit = tmp_path / "explicit.safetensors"
    explicit.write_bytes(b"fake")
    # Use a minimal args-like object.
    class Args:
        weights = str(explicit)
        calibrate_weights = None
        checkpoint_dir = None
    replays = [{"meta": {"iteration": 29}}]
    assert _resolve_checkpoint_path(Args(), replays) == str(explicit)


def test_resolve_checkpoint_auto_discover_from_max_iter(tmp_path):
    """Auto-discovery maps max(meta.iteration) → model_iter_{N+1}.safetensors."""
    from scripts.twixt_replay_analyzer import _resolve_checkpoint_path
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_path = _write_fake_checkpoint(ckpt_dir, iteration=30)
    class Args:
        weights = None
        calibrate_weights = None
        checkpoint_dir = str(ckpt_dir)
    replays = [{"meta": {"iteration": i}} for i in (27, 28, 29)]
    resolved = _resolve_checkpoint_path(Args(), replays)
    assert resolved == str(ckpt_path)


def test_resolve_checkpoint_not_found_returns_none(tmp_path):
    """When nothing is found, return None (analyzer will skip dependent sections)."""
    from scripts.twixt_replay_analyzer import _resolve_checkpoint_path
    class Args:
        weights = None
        calibrate_weights = None
        checkpoint_dir = str(tmp_path / "nonexistent")
    replays = [{"meta": {"iteration": 29}}]
    assert _resolve_checkpoint_path(Args(), replays) is None


def test_resolve_checkpoint_legacy_calibrate_weights_fallback(tmp_path):
    """Legacy --calibrate-weights path is honored when --weights not set."""
    from scripts.twixt_replay_analyzer import _resolve_checkpoint_path
    legacy = tmp_path / "legacy.safetensors"
    legacy.write_bytes(b"fake")
    class Args:
        weights = None
        calibrate_weights = str(legacy)
        checkpoint_dir = None
    replays = [{"meta": {"iteration": 29}}]
    assert _resolve_checkpoint_path(Args(), replays) == str(legacy)


def test_analyzer_emits_replay_probe_scoring(tmp_path):
    """End-to-end: analyzer with auto-discovered checkpoint populates
    summary['replay_probe_scoring'] with real counts."""
    import subprocess

    games_dir = tmp_path / "Replays" / "test_range"
    games_dir.mkdir(parents=True)
    for i in range(6):
        _write_fake_replay(games_dir, iteration=29, game_idx=i, winner="red")
        _write_fake_replay(games_dir, iteration=29, game_idx=100 + i, winner="black")

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _write_fake_checkpoint(ckpt_dir, iteration=30, in_channels=30)

    out_dir = tmp_path / "test_range_Replay"
    result = subprocess.run(
        [".venv/bin/python", "scripts/twixt_replay_analyzer.py",
         "--input", str(games_dir),
         "--out", str(out_dir),
         "--checkpoint-dir", str(ckpt_dir),
         "--calibration-disable",  # focus this test on probe scoring
         "--no-plots"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    suffix = "test_range"  # derived from --out basename minus _Replay suffix
    summary_path = out_dir / f"summary_{suffix}.json"
    assert summary_path.exists(), f"summary not produced: {list(out_dir.iterdir())}"
    summary = json.loads(summary_path.read_text())

    rps = summary.get("replay_probe_scoring")
    assert rps is not None, "replay_probe_scoring missing from summary"
    assert rps["source"] == "replay_derived"
    assert rps["probe_count"] > 0
    assert rps["n"] == rps["probe_count"]
    assert 0.0 <= rps["sign_correct_pct"] <= 1.0
    assert "by_category" in rps
