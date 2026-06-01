import json
import os

import pytest

from scripts.GPU.alphazero.eval_runner import EvalConfig
from scripts.GPU.alphazero.eval_checkpoint_match import run_match

CKPT_DIR = "checkpoints/alphazero-v2-staged"
CKPT_0419 = os.path.join(CKPT_DIR, "model_iter_0419.safetensors")

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.path.exists(CKPT_0419), reason="0419 checkpoint absent")
def test_self_match_color_balance_is_near_even(tmp_path):
    """Sanity gate: a model vs itself, validating color-balanced bookkeeping.

    For a self-match the per-CHECKPOINT score is meaningless (a_ckpt == b_ckpt,
    so winner_checkpoint matches both and a_wins counts every decisive game).
    And red_win_rate is NOT 0.5 either — TwixT has a real first-move (red)
    advantage, so red legitimately wins >50% of decisive games.

    The meaningful, bug-catching metric is A's SIDE-AWARE score: A plays red on
    even game_idx and black on odd, so if color-balancing correctly cancels the
    first-move advantage, A's combined score over both roles must sit near 0.5.
    A gross deviation means a color-assignment / seed / bookkeeping bug.

    Deterministic (fixed base_seed) — pass/fail is stable run-to-run; the wide
    band only tolerates the 20-game sample size. This is a smoke gate, NOT proof
    (see the note below for the 100-200 game manual validation).
    """
    cfg = EvalConfig(board_size=24, mcts_sims=64, max_moves=280)
    out = tmp_path / "self.json"
    summary = run_match(
        a_ckpt=CKPT_0419, b_ckpt=CKPT_0419, games=20, base_seed=12345,
        config=cfg, workers=1, output=str(out),
    )
    assert summary["games"] == 20

    recs = [json.loads(line) for line
            in (tmp_path / "self_games.jsonl").read_text().splitlines()]
    assert len(recs) == 20
    # A's side: red on even game_idx, black on odd.
    a_side_score = sum(
        (r["red_score"] if r["game_idx"] % 2 == 0 else r["black_score"])
        for r in recs
    ) / len(recs)
    print(f"a_side_score={a_side_score:.3f}")
    assert 0.2 <= a_side_score <= 0.8, (
        f"self-match side-aware score {a_side_score:.3f} far from 0.5 — "
        f"suspect a color-assignment / seed / bookkeeping bug"
    )
