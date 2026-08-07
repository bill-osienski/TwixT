"""Aggregate EvalGameResults into match / tournament summary dicts.

Pure: no MLX, no time, no git (the CLI stamps generated_at / git_commit).
"""
from __future__ import annotations

from statistics import mean

from .eval_elo import score_rate, elo_diff, score_ci_trinomial, elo_ci, verdict
from .eval_runner import short_id, AGENT_COMPARISON_UNIT   # shared low-level module (no import cycle)

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


def _agent_color_stats(results, agent_id, color):
    """Per-colour stats INCLUDING a 95% interval.

    Spec section 8.1's colour-safety rule rejects only when a colour's own 95%
    UPPER bound falls below 50%, so the interval is a required decision input,
    not a nicety. Undefined on an empty colour -> None, never 0.
    """
    if color == "red":
        sub = [r for r in results if r.red_agent_id == agent_id]
        wins = sum(1 for r in sub if r.winner == "red")
        losses = sum(1 for r in sub if r.winner == "black")
    else:
        sub = [r for r in results if r.black_agent_id == agent_id]
        wins = sum(1 for r in sub if r.winner == "black")
        losses = sum(1 for r in sub if r.winner == "red")
    caps = sum(1 for r in sub if r.winner is None)
    n = len(sub)
    if not n:
        return {"games": 0, "wins": 0, "losses": 0, "caps": 0,
                "score_rate": None, "score_rate_ci95": None,
                "decisive_games": 0, "decisive_score_rate": None}
    lo, hi = score_ci_trinomial(wins, caps, losses)
    decisive = wins + losses
    return {
        "games": n, "wins": wins, "losses": losses, "caps": caps,
        "score_rate": score_rate(wins, caps, n),
        "score_rate_ci95": [lo, hi],
        "decisive_games": decisive,
        # Secondary per spec 8.1; None when there are no decisive games.
        "decisive_score_rate": (wins / decisive) if decisive else None,
    }


def _termination_distribution(results):
    """Counts by termination reason. Every reason that occurred appears."""
    out = {}
    for r in results:
        out[r.reason] = out.get(r.reason, 0) + 1
    return out


def summarize_match(results, a_ckpt, b_ckpt, pairing_id, config) -> dict:
    if not results:
        # Empty here means a grouping bug (callers reject empty pairings
        # before running). Fail loud rather than emit a 0.0 placeholder.
        raise ValueError(f"no results for pairing {pairing_id}")

    agent_rows = [r for r in results
                  if getattr(r, "comparison_unit", None) == AGENT_COMPARISON_UNIT]
    if agent_rows:
        raise ValueError(
            f"pairing {pairing_id}: {len(agent_rows)} of {len(results)} results "
            f"carry comparison_unit={AGENT_COMPARISON_UNIT!r}. Checkpoint keying "
            f"cannot score them -- use summarize_agent_match."
        )

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


def summarize_agent_match(results, agent_a_id, agent_b_id, pairing_id,
                          config) -> dict:
    """Aggregate a two-AGENT match. Agents may share a checkpoint.

    Everything is keyed on agent id. `winner_checkpoint` is deliberately
    unused: with one checkpoint on both sides it cannot identify a winner.
    """
    if not results:
        raise ValueError(f"no results for pairing {pairing_id}")
    for r in results:
        if getattr(r, "comparison_unit", None) != AGENT_COMPARISON_UNIT:
            raise ValueError(
                f"pairing {pairing_id}: result {r.task_id} is not an agent "
                f"comparison; use summarize_match")
    ids = {r.red_agent_id for r in results} | {r.black_agent_id for r in results}
    for aid in (agent_a_id, agent_b_id):
        if aid not in ids:
            raise ValueError(f"agent {aid!r} does not appear in these results; "
                             f"present: {sorted(ids)}")

    games = len(results)
    state_caps = sum(1 for r in results if r.reason == "state_cap")
    board_full = sum(1 for r in results if r.reason == "board_full")
    red_wins = sum(1 for r in results if r.winner == "red")
    black_wins = sum(1 for r in results if r.winner == "black")
    decisive = red_wins + black_wins

    a_wins = sum(1 for r in results if r.winner_agent_id == agent_a_id)
    b_wins = sum(1 for r in results if r.winner_agent_id == agent_b_id)
    draws = state_caps + board_full
    a_score = a_wins + 0.5 * draws
    rate = score_rate(a_wins, draws, games)
    s_lo, s_hi = score_ci_trinomial(a_wins, draws, b_wins)
    e_lo, e_hi = elo_ci(a_wins, draws, b_wins)

    def _readout_for(agent_id):
        for r in results:
            if r.red_agent_id == agent_id:
                return r.red_readout
            if r.black_agent_id == agent_id:
                return r.black_readout
        return None

    return {
        "pairing_id": pairing_id,
        "comparison_unit": AGENT_COMPARISON_UNIT,
        "agent_a": agent_a_id,
        "agent_b": agent_b_id,
        "checkpoint_a": results[0].red_checkpoint,
        "checkpoint_b": results[0].black_checkpoint,
        "same_checkpoint": all(r.same_checkpoint for r in results),
        # COMPLETE configs: mode alone cannot distinguish `tournament` from
        # `opening_then_argmax` (both mode "opening_temperature").
        "readout_a": _readout_for(agent_a_id),
        "readout_b": _readout_for(agent_b_id),
        "games": games,
        "state_caps": state_caps,
        "board_full": board_full,
        "color_bias": {
            "red_win_rate_decisive": (red_wins / decisive) if decisive else None,
        },
        "avg_plies": mean(r.n_moves for r in results),
        "draw_score_policy": DRAW_SCORE_POLICY,
        "config": config,
        "a_wins": a_wins, "b_wins": b_wins,
        "a_score": a_score,
        "a_score_rate": rate,
        "elo_estimate": elo_diff(rate, games),
        "elo_ci95": [e_lo, e_hi],
        "score_rate_ci95": [s_lo, s_hi],
        "verdict": verdict(rate),
        # SECONDARY per spec 8.1 -- reported, never decisive, because
        # excluding draws biases the comparison if the candidate changes draw
        # propensity. None when no game was decisive.
        "decisive_games": decisive,
        "a_decisive_score_rate": (a_wins / decisive) if decisive else None,
        # Operational reporting required by spec 8.3.
        "termination_distribution": _termination_distribution(results),
        "state_cap_rate": state_caps / games,
        "a_as_red": _agent_color_stats(results, agent_a_id, "red"),
        "a_as_black": _agent_color_stats(results, agent_a_id, "black"),
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
