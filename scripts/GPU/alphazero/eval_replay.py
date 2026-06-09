"""Replay capture for checkpoint-eval games.

Pure per-ply / per-game record construction plus a single sidecar writer. No
game engine, no MLX. Coordinates are engine-native (row, col) — no x/y
conversion is performed in Phase A. A replay sidecar links from each
*_games.jsonl row via replay_path.
"""
from __future__ import annotations

import json
import os

REPLAY_SCHEMA_VERSION = 1


def ply_record(ply, player, move, counts, root_value):
    """One per-ply replay record.

    `move` is the selected (row, col). `counts` is the MCTS visit-count dict
    {(row, col): visits} over all legal moves at this root. `root_value` is
    root.q_value from the perspective of `player` (the side about to move),
    before the move is applied. Fail loud rather than emit a corrupt record.
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
