"""Tests for per-move search_score + root_top1_share persistence (spec 2026-05-03 §5).

Covers Phase 0 saver-side behavior. Self-play and analyzer tests live in
adjacent files (test_self_play_per_move_capture.py, test_analyzer_per_move_stats.py).
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.game_saver import save_game_replay


def _basic_kwargs(games_dir: Path):
    """Minimal common kwargs for save_game_replay; tests vary the per-move lists."""
    return dict(
        games_dir=games_dir,
        iteration=0,
        game_idx=0,
        winner="red",
        move_history=((0, 1), (5, 5), (1, 2)),
        n_moves=3,
        active_size=24,
        simulations=400,
        start_player="red",
    )


def test_save_game_replay_writes_per_move_fields_when_lists_populated(tmp_path):
    """Per-move search_score and root_top1_share land in moves[i] when lists provided."""
    save_game_replay(
        **_basic_kwargs(tmp_path),
        move_root_values=[0.12, -0.34, 0.91],
        move_top1_shares=[0.42, 0.18, 0.77],
    )
    record = json.loads((tmp_path / "iter_0000_game_000.json").read_text())
    moves = record["moves"]
    assert len(moves) == 3
    assert moves[0]["search_score"] == 0.12
    assert moves[1]["search_score"] == -0.34
    assert moves[2]["search_score"] == 0.91
    assert moves[0]["root_top1_share"] == 0.42
    assert moves[1]["root_top1_share"] == 0.18
    assert moves[2]["root_top1_share"] == 0.77


def test_save_game_replay_per_move_fields_null_when_lists_absent(tmp_path):
    """When kwargs are absent, both per-move fields are explicit null in JSON."""
    save_game_replay(**_basic_kwargs(tmp_path))
    record = json.loads((tmp_path / "iter_0000_game_000.json").read_text())
    for m in record["moves"]:
        assert m["search_score"] is None
        assert m["root_top1_share"] is None


def test_save_game_replay_per_move_fields_handle_short_parallel_list(tmp_path):
    """Parallel list shorter than move_history: excess moves get null per-move fields."""
    save_game_replay(
        **_basic_kwargs(tmp_path),
        move_root_values=[0.50, -0.20],   # only 2 entries for 3 moves
        move_top1_shares=[0.80],          # only 1 entry for 3 moves
    )
    record = json.loads((tmp_path / "iter_0000_game_000.json").read_text())
    moves = record["moves"]
    assert moves[0]["search_score"] == 0.50
    assert moves[1]["search_score"] == -0.20
    assert moves[2]["search_score"] is None
    assert moves[0]["root_top1_share"] == 0.80
    assert moves[1]["root_top1_share"] is None
    assert moves[2]["root_top1_share"] is None


def test_save_game_replay_per_move_fields_ignores_long_parallel_list(tmp_path, capsys):
    """Parallel list longer than move_history: extras silently ignored, warning logged."""
    save_game_replay(
        **_basic_kwargs(tmp_path),
        move_root_values=[0.10, 0.20, 0.30, 0.40, 0.50],  # 5 entries for 3 moves
        move_top1_shares=[0.10, 0.20, 0.30, 0.40, 0.50],
    )
    record = json.loads((tmp_path / "iter_0000_game_000.json").read_text())
    moves = record["moves"]
    assert len(moves) == 3
    assert moves[0]["search_score"] == 0.10
    assert moves[1]["search_score"] == 0.20
    assert moves[2]["search_score"] == 0.30
    captured = capsys.readouterr()
    assert "move_root_values length 5" in captured.err
    assert "move_top1_shares length 5" in captured.err
