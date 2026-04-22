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
    moves = [{"player": "red" if i % 2 == 0 else "black",
              "move": [(i * 7 + game_idx) % 24, (i * 11 + game_idx) % 24]}
             for i in range(n_moves)]
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
    from scripts.GPU.alphazero.network import create_network
    net = create_network(in_channels=in_channels, hidden=8, n_blocks=1)
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
