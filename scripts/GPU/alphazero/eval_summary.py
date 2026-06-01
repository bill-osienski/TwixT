"""Aggregate EvalGameResults into match / tournament summary dicts.

Pure: no MLX, no time, no git (the CLI stamps generated_at / git_commit).
"""
from __future__ import annotations

from statistics import mean

from .eval_elo import score_rate, elo_diff, score_ci_trinomial, elo_ci, verdict
from .eval_runner import short_id   # shared low-level module (no import cycle)

DRAW_SCORE_POLICY = "state_cap_and_board_full_score_0.5"


def _color_stats(results, model_ckpt, color):
    if color == "red":
        sub = [r for r in results if r.red_checkpoint == model_ckpt]
        wins = sum(1 for r in sub if r.winner == "red")
        losses = sum(1 for r in sub if r.winner == "black")
    else:
        sub = [r for r in results if r.black_checkpoint == model_ckpt]
        wins = sum(1 for r in sub if r.winner == "black")
        losses = sum(1 for r in sub if r.winner == "red")
    caps = sum(1 for r in sub if r.winner is None)
    n = len(sub)
    return {
        "games": n, "wins": wins, "losses": losses, "caps": caps,
        "score_rate": (score_rate(wins, caps, n) if n else None),
    }


def summarize_match(results, a_ckpt, b_ckpt, pairing_id, config) -> dict:
    if not results:
        # Empty here means a grouping bug (callers reject empty pairings
        # before running). Fail loud rather than emit a 0.0 placeholder.
        raise ValueError(f"no results for pairing {pairing_id}")

    self_match = (a_ckpt == b_ckpt)

    games = len(results)
    state_caps = sum(1 for r in results if r.reason == "state_cap")
    board_full = sum(1 for r in results if r.reason == "board_full")

    red_wins = sum(1 for r in results if r.winner == "red")
    black_wins = sum(1 for r in results if r.winner == "black")
    decisive = red_wins + black_wins

    color_bias = {
        "red_win_rate_decisive": (red_wins / decisive) if decisive else None,
    }

    base = {
        "pairing_id": pairing_id,
        "checkpoint_a": a_ckpt,
        "checkpoint_b": b_ckpt,
        "games": games,
        "state_caps": state_caps,
        "board_full": board_full,
        "self_match": self_match,
        "color_bias": color_bias,
        "avg_plies": mean(r.n_moves for r in results),
        "selection_mode": config.get("selection_mode") if config else None,
        "draw_score_policy": DRAW_SCORE_POLICY,
        "config": config,
    }

    if self_match:
        return {
            **base,
            "a_wins": None, "b_wins": None,
            "a_score": None,
            "a_score_rate": None,
            "elo_estimate": None,
            "elo_ci95": None,
            "score_rate_ci95": None,
            "verdict": None,
            "a_as_red": None,
            "a_as_black": None,
        }

    a_wins = sum(1 for r in results if r.winner_checkpoint == a_ckpt)
    b_wins = sum(1 for r in results if r.winner_checkpoint == b_ckpt)
    draws = state_caps + board_full
    a_score = a_wins + 0.5 * draws
    rate = score_rate(a_wins, draws, games)
    s_lo, s_hi = score_ci_trinomial(a_wins, draws, b_wins)
    e_lo, e_hi = elo_ci(a_wins, draws, b_wins)

    return {
        **base,
        "a_wins": a_wins, "b_wins": b_wins,
        "a_score": a_score,
        "a_score_rate": rate,
        "elo_estimate": elo_diff(rate, games),
        "elo_ci95": [e_lo, e_hi],
        "score_rate_ci95": [s_lo, s_hi],
        "verdict": verdict(rate),
        "a_as_red": _color_stats(results, a_ckpt, "red"),
        "a_as_black": _color_stats(results, a_ckpt, "black"),
    }


def summarize_tournament(results, pairings, config) -> dict:
    by_pairing: dict = {}
    for r in results:
        by_pairing.setdefault(r.pairing_id, []).append(r)

    pairing_summaries = []
    for a_ckpt, b_ckpt in pairings:
        pid = f"{short_id(a_ckpt)}_vs_{short_id(b_ckpt)}"
        group = by_pairing.get(pid, [])
        pairing_summaries.append(
            summarize_match(group, a_ckpt, b_ckpt, pid, config)
        )

    table = [
        {
            "pairing_id": s["pairing_id"],
            "a_score_rate": s["a_score_rate"],
            "elo_estimate": s["elo_estimate"],
            "elo_ci95": s["elo_ci95"],
            "verdict": s["verdict"],
        }
        for s in pairing_summaries
    ]
    table.sort(key=lambda t: t["pairing_id"])
    return {"pairings": pairing_summaries, "table": table, "config": config}
