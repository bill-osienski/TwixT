import copy
import hashlib
import json

import pytest

from scripts.GPU.alphazero import h2_match_rules as RULES
from scripts.GPU.alphazero import h2_replay_export as EXPORT


def _evidence(tmp_path):
    task_id = "h2match-test"
    result = {
        "record_type": "task_result", "task_id": task_id, "seed": 123,
        "winner": "red", "terminal_reason": "win", "plies": 7,
    }
    ply = {"record_type": "ply", "task_id": task_id, "ply": 7,
           "mover": "red", "move": [17, 11]}
    frozen = RULES.transcript([ply], result, opening_bound=6)
    records = [
        {"record_type": "header", "design": "H2",
         "identity": {"eval_config": {"mcts_sims": 400}}},
        result,
        {"record_type": "opening_bound", "task_id": task_id, "ply": 6,
         "opening": "o1_center"},
        ply,
        {"record_type": "transcript", "task_id": task_id,
         "opening": "o1_center", "colour_arm": "t1j_red",
         "opening_bound": 6, "n_plies": 1,
         "transcript_digest": RULES.transcript_digest(frozen)},
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_exact_recorded_moves_become_viewer_moves(tmp_path):
    source = _evidence(tmp_path)
    game = EXPORT.load_recorded_games(source)[0]
    record = EXPORT.viewer_record(game, source=source)

    assert record["seed"] == 123
    assert record["winner"] == "red"
    assert record["meta"]["opening_bound"] == 6
    assert record["meta"]["n_moves"] == 7
    assert [(m["turn"], m["player"], m["row"], m["col"])
            for m in record["moves"]][-2:] == [
                (6, "black", 14, 14), (7, "red", 17, 11)]


def test_digest_mismatch_refuses_instead_of_exporting_a_guess(tmp_path):
    source = _evidence(tmp_path)
    game = EXPORT.load_recorded_games(source)[0]
    game = copy.deepcopy(game)
    game["transcript"]["transcript_digest"] = hashlib.sha256(b"wrong").hexdigest()

    with pytest.raises(EXPORT.H2ReplayExportError, match="digest mismatch"):
        EXPORT.viewer_record(game, source=source)


def test_output_is_create_only(tmp_path):
    path = tmp_path / "replay.json"
    EXPORT.write_viewer_record({"moves": []}, path)

    with pytest.raises(EXPORT.H2ReplayExportError, match="refusing to replace"):
        EXPORT.write_viewer_record({"moves": []}, path)
