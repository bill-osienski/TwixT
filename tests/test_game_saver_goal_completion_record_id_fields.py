"""save_game_replay must overwrite goal_completion_record's game_idx/game_id
with the saver's authoritative values.

Background: under parallel self-play, play_game writes the dispatch-order
counter into goal_completion_record.game_idx / game_id, while the saver
uses the save-order counter for the filename and meta. The two diverge,
making the worst-cases CSV's iteration/game_idx columns point to the wrong
files. The saver is the right place to reconcile because it's the source
of truth for the save-order counter.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.alphazero.game_saver import save_game_replay


def _read(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def test_save_overrides_record_game_idx_and_game_id(tmp_path):
    bogus = {
        "version": 1,
        "iteration": 169,
        "game_idx": 13,           # dispatch-order, wrong
        "game_id": "game_013",    # derived from the wrong counter
        "winner": "red",
        "detected": False,
    }
    out = save_game_replay(
        games_dir=tmp_path,
        iteration=169,
        game_idx=22,              # save-order, authoritative
        winner="red",
        move_history=((4, 19), (12, 12)),
        n_moves=2,
        goal_completion_record=bogus,
    )
    saved = _read(out)
    assert saved["meta"]["game_idx"] == 22
    assert saved["goal_completion_record"]["game_idx"] == 22
    assert saved["goal_completion_record"]["game_id"] == "game_022"
    # Non-identity fields are preserved.
    assert saved["goal_completion_record"]["winner"] == "red"
    assert saved["goal_completion_record"]["detected"] is False


def test_save_does_not_mutate_caller_record(tmp_path):
    """The trainer collects goal_completion_record into all_goal_completion_records
    BEFORE save_game_replay runs. Mutating the dict in place would back-propagate
    the override into the sidecar aggregation path. Use a defensive copy.
    """
    caller_view = {
        "iteration": 169,
        "game_idx": 13,
        "game_id": "game_013",
        "winner": "black",
    }
    save_game_replay(
        games_dir=tmp_path,
        iteration=169,
        game_idx=22,
        winner="black",
        move_history=((4, 19),),
        n_moves=1,
        goal_completion_record=caller_view,
    )
    assert caller_view["game_idx"] == 13
    assert caller_view["game_id"] == "game_013"


def test_save_record_none_is_unchanged(tmp_path):
    out = save_game_replay(
        games_dir=tmp_path,
        iteration=169,
        game_idx=22,
        winner=None,
        move_history=(),
        n_moves=0,
        goal_completion_record=None,
    )
    saved = _read(out)
    assert "goal_completion_record" not in saved
