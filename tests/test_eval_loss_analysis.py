import pytest

from scripts.GPU.alphazero.eval_loss_analysis import (
    score_for_checkpoint, a_color, validate_rows,
)

A = "ckpts/model_iter_0399.safetensors"
B = "ckpts/model_iter_0379.safetensors"


def _row(game_idx, red, black, winner, reason="win", n=50, task_id=0,
         pairing_id="0399_vs_0379"):
    if winner == "red":
        rs, bs, wc = 1.0, 0.0, red
    elif winner == "black":
        rs, bs, wc = 0.0, 1.0, black
    else:
        rs, bs, wc = 0.5, 0.5, None
    return {
        "task_id": task_id, "pairing_id": pairing_id, "game_idx": game_idx,
        "red_checkpoint": red, "black_checkpoint": black,
        "winner": winner, "winner_checkpoint": wc, "reason": reason,
        "n_moves": n, "red_score": rs, "black_score": bs,
    }


def test_score_for_checkpoint_win_red_a():
    r = _row(0, A, B, "red")
    assert score_for_checkpoint(r, A) == 1.0
    assert score_for_checkpoint(r, B) == 0.0


def test_score_for_checkpoint_win_black_a():
    # A is seated as black this game and wins.
    r = _row(1, B, A, "black")
    assert score_for_checkpoint(r, A) == 1.0
    assert score_for_checkpoint(r, B) == 0.0


def test_score_for_checkpoint_draw_state_cap():
    r = _row(2, A, B, None, reason="state_cap", n=280)
    assert score_for_checkpoint(r, A) == 0.5
    assert score_for_checkpoint(r, B) == 0.5


def test_a_color_tracks_seat():
    assert a_color(_row(0, A, B, "red"), A) == "red"
    assert a_color(_row(1, B, A, "black"), A) == "black"


def test_validation_rejects_winner_checkpoint_mismatch():
    bad = _row(0, A, B, "red")
    bad["winner_checkpoint"] = B  # winner says red(A) but ckpt points at B
    with pytest.raises(ValueError, match="winner_checkpoint"):
        validate_rows([bad])


def test_validation_rejects_inconsistent_draw_scores():
    bad = _row(0, A, B, None, reason="state_cap", n=280)
    bad["red_score"] = 1.0  # draw must be 0.5/0.5
    with pytest.raises(ValueError, match="draw"):
        validate_rows([bad])


def test_validation_rejects_unknown_error():
    bad = _row(0, A, B, None, reason="unknown_error")
    bad["winner_checkpoint"] = None
    with pytest.raises(ValueError, match="unknown_error"):
        validate_rows([bad])


def test_validation_rejects_mixed_jsonl():
    rows = [_row(0, A, B, "red"), _row(1, A, "ckpts/model_iter_0123.safetensors", "red")]
    with pytest.raises(ValueError, match="mixed"):
        validate_rows(rows, A, B)
