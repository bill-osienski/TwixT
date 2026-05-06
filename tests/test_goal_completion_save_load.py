"""IPC + saver plumbing for goal_completion_record (spec §9)."""
import json
import pickle
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.ipc_messages import GameComplete
from scripts.GPU.alphazero.self_play import GameRecord, PositionRecord
from scripts.GPU.alphazero.game_saver import save_game_replay


def test_game_record_has_goal_completion_record_field_default_none():
    rec = GameRecord(positions=[], winner="red", n_moves=0)
    assert hasattr(rec, "goal_completion_record")
    assert rec.goal_completion_record is None


def test_game_complete_has_goal_completion_record_field_default_none():
    gc = GameComplete(
        worker_id=0, winner="red", draw_reason=0, n_moves=0, n_positions=0,
        wall_time_s=0.0, nn_calls=0, expand_calls=0, nn_batches=0,
        total_backups=0, total_waiters=0, unique_leaves=0,
        max_waiters=0, flush_full=0, flush_stall=0, flush_tail=0,
    )
    assert hasattr(gc, "goal_completion_record")
    assert gc.goal_completion_record is None


def test_game_complete_pickle_roundtrip_preserves_goal_completion_record():
    record = {"version": 1, "outcome_class": 1, "winner": "red", "detected": True}
    gc = GameComplete(
        worker_id=0, winner="red", draw_reason=0, n_moves=21, n_positions=21,
        wall_time_s=1.0, nn_calls=0, expand_calls=0, nn_batches=0,
        total_backups=0, total_waiters=0, unique_leaves=0,
        max_waiters=0, flush_full=0, flush_stall=0, flush_tail=0,
        goal_completion_record=record,
    )
    payload = pickle.dumps(gc)
    gc2 = pickle.loads(payload)
    assert gc2.goal_completion_record == record


def test_save_game_replay_writes_top_level_goal_completion_record_when_present():
    record = {
        "version": 1, "outcome_class": 1, "game_id": "iter_0001_game_000",
        "winner": "red", "detected": True,
    }
    with tempfile.TemporaryDirectory() as tmp:
        games_dir = Path(tmp)
        path = save_game_replay(
            games_dir=games_dir,
            iteration=1, game_idx=0, winner="red",
            move_history=((0, 0),), n_moves=1,
            goal_completion_record=record,
        )
        with open(path) as f:
            payload = json.load(f)
        assert payload["goal_completion_record"] == record


def test_save_game_replay_omits_goal_completion_record_when_none():
    with tempfile.TemporaryDirectory() as tmp:
        games_dir = Path(tmp)
        path = save_game_replay(
            games_dir=games_dir,
            iteration=1, game_idx=0, winner="red",
            move_history=((0, 0),), n_moves=1,
            goal_completion_record=None,
        )
        with open(path) as f:
            payload = json.load(f)
        assert "goal_completion_record" not in payload


def test_save_game_replay_independent_of_other_goal_completion_keys():
    """All three keys are independent: any subset can be present."""
    with tempfile.TemporaryDirectory() as tmp:
        games_dir = Path(tmp)
        path = save_game_replay(
            games_dir=games_dir,
            iteration=1, game_idx=0, winner="red",
            move_history=((0, 0),), n_moves=1,
            goal_completion_record={"version": 1, "outcome_class": 3},
            goal_completion_diagnostics=None,
            goal_completion_diagnostics_meta=None,
        )
        with open(path) as f:
            payload = json.load(f)
        assert "goal_completion_record" in payload
        assert "goal_completion_diagnostics" not in payload
        assert "goal_completion_diagnostics_meta" not in payload
