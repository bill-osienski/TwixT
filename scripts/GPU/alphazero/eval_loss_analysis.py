"""Pure loss-shape analysis over checkpoint-eval *_games.jsonl rows.

No IO, no MLX: rows in (plain dicts), dicts/lists out (ready to serialize).
Scoring matches eval_summary: A/B keyed off winner_checkpoint, color off
red/black_checkpoint, draws (state_cap/board_full) = 0.5 for both sides.
"""
from __future__ import annotations

from statistics import mean

from scripts.GPU.alphazero.eval_elo import (
    score_rate, elo_diff, score_ci_trinomial, elo_ci, verdict,
)
from scripts.GPU.alphazero.eval_runner import short_id

LENGTH_BUCKETS_DEFAULT = (40, 60, 80, 120, 279, 280)

REQUIRED_KEYS = {
    "task_id", "pairing_id", "game_idx", "red_checkpoint", "black_checkpoint",
    "winner", "winner_checkpoint", "reason", "n_moves", "red_score", "black_score",
}
DRAW_REASONS = {"state_cap", "board_full"}
VALID_REASONS = {"win", "state_cap", "board_full", "unknown_error"}


def _require(i, cond, msg):
    if not cond:
        raise ValueError(f"row {i}: {msg}")


def validate_rows(rows, a_ckpt=None, b_ckpt=None):
    """Fail loud on any row that breaks the eval scoring invariants.

    When a_ckpt and b_ckpt are both given, also require every row to be
    between exactly those two checkpoints (catches a mixed/concatenated
    JSONL of more than one pairing).
    """
    if not rows:
        raise ValueError("no rows to analyze")
    ab = {a_ckpt, b_ckpt} if (a_ckpt is not None and b_ckpt is not None) else None
    for i, r in enumerate(rows):
        missing = REQUIRED_KEYS - r.keys()
        _require(i, not missing, f"missing keys {sorted(missing)}")
        reason = r["reason"]
        _require(i, reason in VALID_REASONS, f"bad reason {reason!r}")
        _require(i, reason != "unknown_error",
                 "reason 'unknown_error' not handled in V1 (none expected in current data)")
        winner = r["winner"]
        _require(i, winner in ("red", "black", None), f"bad winner {winner!r}")
        red, black = r["red_checkpoint"], r["black_checkpoint"]
        if winner == "red":
            _require(i, r["winner_checkpoint"] == red, "winner_checkpoint != red_checkpoint")
            _require(i, r["red_score"] == 1.0 and r["black_score"] == 0.0,
                     "red-win scores not 1.0/0.0")
        elif winner == "black":
            _require(i, r["winner_checkpoint"] == black, "winner_checkpoint != black_checkpoint")
            _require(i, r["red_score"] == 0.0 and r["black_score"] == 1.0,
                     "black-win scores not 0.0/1.0")
        else:  # draw
            _require(i, r["winner_checkpoint"] is None, "draw winner_checkpoint not None")
            _require(i, r["red_score"] == 0.5 and r["black_score"] == 0.5,
                     "draw scores not 0.5/0.5")
            _require(i, reason in DRAW_REASONS, f"draw reason {reason!r} not a draw reason")
        if ab is not None and {red, black} != ab:
            _require(i, False,
                     f"checkpoints {{{short_id(red)}, {short_id(black)}}} != resolved A/B "
                     "— mixed JSONL?")


def score_for_checkpoint(row, ckpt):
    """1.0 if ckpt won, 0.5 on a draw (no winner), else 0.0. Keyed off
    winner_checkpoint — never off color."""
    if row["winner_checkpoint"] == ckpt:
        return 1.0
    if row["winner_checkpoint"] is None:
        return 0.5
    return 0.0


def a_color(row, a_ckpt):
    """Which seat A played this game: 'red' or 'black'."""
    if row["red_checkpoint"] == a_ckpt:
        return "red"
    if row["black_checkpoint"] == a_ckpt:
        return "black"
    raise ValueError(f"A checkpoint {short_id(a_ckpt)} not in row {row['game_idx']}")
