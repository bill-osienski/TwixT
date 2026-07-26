"""Aggregate EvalGameResults into match / tournament summary dicts.

Pure: no MLX, no time, no git (the CLI stamps generated_at / git_commit).
"""
from __future__ import annotations

from statistics import mean

from .eval_elo import score_rate, elo_diff, score_ci_trinomial, elo_ci, verdict
from .eval_runner import short_id   # shared low-level module (no import cycle)

DRAW_SCORE_POLICY = "state_cap_and_board_full_score_0.5"


def _color_stats(results, ident, color, by_agent=False):
    """Per-color stats for one side.

    `by_agent` selects agent identity instead of checkpoint identity. Two
    agents can share a checkpoint, in which case matching on the checkpoint
    would pick up both sides of every game.
    """
    red_key, black_key = (("red_agent", "black_agent") if by_agent
                          else ("red_checkpoint", "black_checkpoint"))
    if color == "red":
        sub = [r for r in results if getattr(r, red_key) == ident]
        wins = sum(1 for r in sub if r.winner == "red")
        losses = sum(1 for r in sub if r.winner == "black")
    else:
        sub = [r for r in results if getattr(r, black_key) == ident]
        wins = sum(1 for r in sub if r.winner == "black")
        losses = sum(1 for r in sub if r.winner == "red")
    caps = sum(1 for r in sub if r.winner is None)
    n = len(sub)
    return {
        "games": n, "wins": wins, "losses": losses, "caps": caps,
        "score_rate": (score_rate(wins, caps, n) if n else None),
    }


def summarize_match(results, a_ckpt, b_ckpt, pairing_id, config,
                    a_agent=None, b_agent=None) -> dict:
    """Aggregate one pairing.

    `a_agent`/`b_agent` name two agents that are distinguishable independently
    of their checkpoints. Supplying them scores by agent identity, which is the
    only way to score two agents that share a checkpoint: the checkpoint-based
    path below cannot tell them apart and correctly refuses to try. Omitting
    them keeps the historical checkpoint-based behaviour exactly.
    """
    if not results:
        # Empty here means a grouping bug (callers reject empty pairings
        # before running). Fail loud rather than emit a 0.0 placeholder.
        raise ValueError(f"no results for pairing {pairing_id}")

    by_agent = a_agent is not None and b_agent is not None
    if by_agent:
        if a_agent == b_agent:
            raise ValueError(f"agents must be distinct, got {a_agent!r} twice")
        missing = [r.task_id for r in results
                   if r.red_agent is None or r.black_agent is None]
        if missing:
            raise ValueError(
                f"agent scoring requested but results {sorted(missing)} carry "
                f"no agent identity")
    elif a_agent is not None or b_agent is not None:
        raise ValueError("a_agent and b_agent must be supplied together")

    # Same checkpoint AND no agent identity => genuinely indistinguishable.
    self_match = (a_ckpt == b_ckpt) and not by_agent

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
    if by_agent:
        # Added only in agent mode, so legacy summary bytes are untouched.
        base["agent_a"] = a_agent
        base["agent_b"] = b_agent

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

    if by_agent:
        a_wins = sum(1 for r in results if r.winner_agent == a_agent)
        b_wins = sum(1 for r in results if r.winner_agent == b_agent)
    else:
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
        "a_as_red": _color_stats(results, a_agent if by_agent else a_ckpt,
                                 "red", by_agent),
        "a_as_black": _color_stats(results, a_agent if by_agent else a_ckpt,
                                   "black", by_agent),
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
