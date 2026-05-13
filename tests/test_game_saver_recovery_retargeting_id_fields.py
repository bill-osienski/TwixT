"""save_game_replay must overwrite recovery_retargeting_record's game_idx/game_id
with the saver's authoritative values, mirroring the 32c4966a6 fix for
goal_completion_record."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.game_saver import save_game_replay


def _read(path):
    with open(path) as f:
        return json.load(f)


def test_save_overrides_recovery_record_game_idx_and_game_id(tmp_path):
    bogus = {
        "version": 1,
        "iteration": 170,
        "game_idx": 13,             # dispatch-order, wrong
        "game_id": "game_013",
        "winner": "red", "loser": "black",
        "triggered_sides": ["black"],
        "side_records": {"red": {"triggered": False}, "black": {"triggered": True}},
    }
    out = save_game_replay(
        games_dir=tmp_path,
        iteration=170,
        game_idx=22,               # save-order, authoritative
        winner="red",
        move_history=((4, 19), (12, 12)),
        n_moves=2,
        recovery_retargeting_record=bogus,
    )
    saved = _read(out)
    assert saved["recovery_retargeting_record"]["game_idx"] == 22
    assert saved["recovery_retargeting_record"]["game_id"] == "game_022"
    assert saved["recovery_retargeting_record"]["triggered_sides"] == ["black"]


def test_save_does_not_mutate_caller_recovery_record(tmp_path):
    caller_view = {"iteration": 170, "game_idx": 13, "game_id": "game_013"}
    save_game_replay(
        games_dir=tmp_path, iteration=170, game_idx=22, winner="red",
        move_history=((4, 19),), n_moves=1,
        recovery_retargeting_record=caller_view,
    )
    assert caller_view["game_idx"] == 13
    assert caller_view["game_id"] == "game_013"


def test_save_recovery_record_none_is_unchanged(tmp_path):
    out = save_game_replay(
        games_dir=tmp_path, iteration=170, game_idx=22, winner=None,
        move_history=(), n_moves=0, recovery_retargeting_record=None,
    )
    saved = _read(out)
    assert "recovery_retargeting_record" not in saved
