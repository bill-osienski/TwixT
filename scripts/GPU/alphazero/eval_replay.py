"""Replay capture for checkpoint-eval games.

Pure per-ply / per-game record construction plus a single sidecar writer. No
game engine, no MLX. Coordinates are engine-native (row, col) — no x/y
conversion is performed in Phase A. A replay sidecar links from each
*_games.jsonl row via replay_path.
"""
from __future__ import annotations

import json
import os

REPLAY_SCHEMA_VERSION = 2


def _child_stat_dict(stat):
    """Serialize one eval_readout.ChildStat. Undefined means stay None, never
    0.0 -- MCTSNode.q_value returns 0.0 at zero visits, which is not a
    measurement."""
    return {
        "row": stat.move[0],
        "col": stat.move[1],
        "completed_visit_count": stat.visits,
        "q_value_child_perspective": stat.q_child,
        "q_value_root_perspective": stat.q_root,
    }


def ply_record(ply, player, move, counts, root_value, top2=None,
               overrode_leader=False):
    """One per-ply replay record.

    `move` is the selected (row, col). `counts` is the MCTS visit-count dict
    {(row, col): visits} over all legal moves at this root. `root_value` is
    root.q_value from the perspective of `player` (the side about to move),
    before the move is applied. `top2` is the top-two root children by
    completed visits (eval_readout.ChildStat), or None when not captured --
    None means "not captured", never "no children". Fail loud rather than
    emit a corrupt record.
    """
    if not counts:
        raise ValueError(f"ply {ply}: empty visit counts")
    if move not in counts:
        raise ValueError(f"ply {ply}: selected move {move} not in visit counts")
    total = sum(counts.values())
    # rank: descending visit count, ties broken by ascending (row, col).
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    rank = 1 + next(i for i, (m, _c) in enumerate(ranked) if m == move)
    row, col = move
    return {
        "ply": ply,
        "player": player,
        "row": row,
        "col": col,
        "root_value": root_value,
        "root_top1_share": max(counts.values()) / total,
        "selected_visit_rank": rank,
        "selected_visit_count": counts[move],
        "root_total_visits": total,
        "n_legal": len(counts),
        "top2": ([_child_stat_dict(s) for s in top2]
                 if top2 is not None else None),
        "readout_overrode_leader": bool(overrode_leader),
    }


def build_replay_dict(result, seed, board_size, records):
    """Assemble the replay sidecar dict from a finished EvalGameResult plus the
    per-ply records. Reads identity/outcome from `result`; `seed` and
    `board_size` complete the contract."""
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "pairing_id": result.pairing_id,
        "game_idx": result.game_idx,
        "task_id": result.task_id,
        "seed": seed,
        "board_size": board_size,
        "red_checkpoint": result.red_checkpoint,
        "black_checkpoint": result.black_checkpoint,
        # Agent identity, so a replay is self-describing and the preflight can
        # tell whose turns to score. All None on legacy checkpoint artifacts.
        # COMPLETE readout configs, not just the mode -- `tournament` and
        # `opening_then_argmax` are both mode "opening_temperature".
        "red_agent_id": getattr(result, "red_agent_id", None),
        "black_agent_id": getattr(result, "black_agent_id", None),
        "red_readout": getattr(result, "red_readout", None),
        "black_readout": getattr(result, "black_readout", None),
        "comparison_unit": getattr(result, "comparison_unit", None),
        "winner": result.winner,
        "winner_checkpoint": result.winner_checkpoint,
        "reason": result.reason,
        "n_moves": result.n_moves,
        "moves": records,
    }


def replay_filename(game_idx):
    return f"game_{game_idx:06d}.json"


def write_replay(replay_dir, replay_dict):
    """Write one game sidecar; return its path relative to the process CWD.

    Worker-safe: makedirs(exist_ok=True) tolerates concurrent creation by other
    worker processes writing into the same replay_dir.
    """
    os.makedirs(replay_dir, exist_ok=True)
    path = os.path.join(replay_dir, replay_filename(replay_dict["game_idx"]))
    with open(path, "w") as fh:
        json.dump(replay_dict, fh)
    # relpath raises ValueError on cross-drive paths (Windows); safe on macOS.
    return os.path.relpath(path)
