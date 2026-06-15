"""Pure helpers for the goal-line trigger probe.

No MLX. Selection of fixed trigger cases from the candidates CSV, board
reconstruction at a case's position_ply, and per-checkpoint summary stats.
A case's position_ply is BLACK's decision point, one ply before red's goal-line
trigger move (trigger_red_ply = position_ply + 1). post_opening_only keys on the
drop's largest_drop_phase, NOT on position_ply (which may sit inside the opening
window — e.g. game 15, position_ply=19).
"""
from __future__ import annotations

from statistics import mean, median

from .game.twixt_state import TwixtState

EXPECTED_PROBLEM = "black_overvalues_red_goal_trigger"
OVERVALUE_THRESHOLD = 0.25
SEVERE_OVERVALUE_THRESHOLD = 0.50

DEFAULT_SELECTION = {
    "min_prev_black_value": 0.25,
    "min_prev_black_top1": 0.5,
    "post_opening_only": True,
    "trigger_zone_prefix": "red_goal",
}


def _candidate_to_case(r):
    """Map one candidates.csv row (string values) to a manifest case dict."""
    return {
        "game_idx": int(r["game_idx"]),
        "rank": int(r["rank"]),
        "replay_path": r["replay_path"],
        "position_ply": int(r["prev_black_ply"]),
        "side_to_move": "black",
        "expected_problem": EXPECTED_PROBLEM,
        "trigger_red_ply": int(r["trigger_red_ply"]),
        "trigger_red_move": {"row": int(r["trigger_red_row"]),
                             "col": int(r["trigger_red_col"])},
        "trigger_zone": r["trigger_zone"],
        "baseline_black_prev_value": float(r["prev_black_value"]),
        "baseline_black_prev_top1": float(r["prev_black_top1"]),
        "drop_black_ply": int(r["drop_black_ply"]),
        "drop_black_value": float(r["drop_black_value"]),
        "drop_amount": float(r["drop_amount"]),
    }


def select_cases(candidate_rows, selection):
    """Filter candidate rows by the selection criteria; map survivors to cases.

    Order is preserved (the candidates CSV is rank-sorted). The post_opening_only
    test reads largest_drop_phase, never position_ply.
    """
    out = []
    for r in candidate_rows:
        if float(r["prev_black_value"]) < selection["min_prev_black_value"]:
            continue
        if float(r["prev_black_top1"]) < selection["min_prev_black_top1"]:
            continue
        if selection["post_opening_only"] and r["largest_drop_phase"] != "post_opening":
            continue
        if not r["trigger_zone"].startswith(selection["trigger_zone_prefix"]):
            continue
        out.append(_candidate_to_case(r))
    return out


def case_id(case):
    return f"game_{case['game_idx']:06d}_ply_{case['position_ply']}"


def position_state(replay, position_ply, side_to_move):
    """Board at the side-to-move's decision point: apply moves[0:position_ply]
    to a fresh TwixtState. Fail loud if the ply is out of range or the
    reconstructed side to move disagrees with the manifest."""
    moves = replay["moves"]
    if not (0 <= position_ply < len(moves)):
        raise ValueError(
            f"position_ply {position_ply} out of range [0, {len(moves)})")
    state = TwixtState(active_size=replay["board_size"], to_move="red",
                       max_plies_limit=replay["n_moves"])
    for m in moves[:position_ply]:
        state = state.apply_move((m["row"], m["col"]))
    if state.to_move != side_to_move:
        raise ValueError(
            f"reconstructed to_move {state.to_move!r} != side_to_move "
            f"{side_to_move!r} at position_ply {position_ply}")
    return state
