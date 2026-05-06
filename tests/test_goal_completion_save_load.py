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


def test_save_game_from_record_injects_iteration():
    """The trainer overwrites iteration=0 placeholder with its actual iter."""
    from scripts.GPU.alphazero.trainer import _save_game_from_record

    # Minimal in-memory game saver double.
    saved = {}
    class _FakeSaver:
        def maybe_save_game(self, *args, **kwargs):
            saved.update(kwargs)
            return Path("/tmp/fake_path.json")

    rec = GameRecord(positions=[], winner="red", n_moves=1)
    rec.move_history = [(0, 0)]
    rec.start_player = "red"
    rec.draw_reason = None
    rec.goal_completion_record = {
        "version": 1, "outcome_class": 1, "winner": "red",
        "iteration": 0, "game_idx": 5, "game_id": "game_005",
    }

    fake = _FakeSaver()
    fake._current_iter = 112
    _save_game_from_record(fake, rec)
    rec_arg = saved["goal_completion_record"]
    assert rec_arg["iteration"] == 112


def test_save_game_from_ipc_injects_iteration():
    from scripts.GPU.alphazero.trainer import _save_game_from_ipc

    saved = {}
    class _FakeSaver:
        def maybe_save_game(self, *args, **kwargs):
            saved.update(kwargs)
            return Path("/tmp/fake_path.json")

    msg = GameComplete(
        worker_id=0, winner="red", draw_reason=0, n_moves=1, n_positions=1,
        wall_time_s=0.0, nn_calls=0, expand_calls=0, nn_batches=0,
        total_backups=0, total_waiters=0, unique_leaves=0,
        max_waiters=0, flush_full=0, flush_stall=0, flush_tail=0,
        move_history=((0, 0),),
        start_player="red",
        goal_completion_record={
            "version": 1, "outcome_class": 1, "winner": "red",
            "iteration": 0, "game_idx": 5, "game_id": "game_005",
        },
    )
    fake = _FakeSaver()
    fake._current_iter = 112
    _save_game_from_ipc(fake, msg)
    rec_arg = saved["goal_completion_record"]
    assert rec_arg["iteration"] == 112
