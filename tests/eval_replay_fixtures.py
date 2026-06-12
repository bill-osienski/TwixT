"""Shared synthetic Phase A capture builders for V2 Phase B tests. No MLX.

make_game builds a matched (games.jsonl row, replay sidecar dict) pair with
correct red/black alternation, A seated by color, and consistent identity
fields, so validate_rows (V1) and validate_replay (V2) both accept it.
"""

A = "ckpts/model_iter_0399.safetensors"
B = "ckpts/model_iter_0379.safetensors"
PAIRING = "0399_vs_0379"


def make_ply(ply, player, root_value, *, row=None, col=None, top1=0.5, rank=1,
             visits=200, total=400, n_legal=100):
    return {
        "ply": ply, "player": player,
        "row": ply if row is None else row, "col": ply if col is None else col,
        "root_value": root_value, "root_top1_share": top1,
        "selected_visit_rank": rank, "selected_visit_count": visits,
        "root_total_visits": total, "n_legal": n_legal,
    }


def make_game(game_idx, *, a_is_black=True, a_wins=False, n_moves=50,
              a_values=None, b_values=None, a_top1=0.5, a_rank=1,
              reason="win", task_id=None, replay_dir="replays"):
    """Build a (row, replay) pair.

    a_values / b_values: per-side root_value sequences (must match that
    side's ply count: n_moves // 2 plus the odd ply for red); default flat
    0.0. a_top1 / a_rank: scalar applied to every A ply, or a per-A-ply list.
    reason="state_cap" builds a draw (winner None, 0.5/0.5 scores).
    """
    red_ck, black_ck = (B, A) if a_is_black else (A, B)
    a_clr = "black" if a_is_black else "red"
    if reason == "win":
        winner = a_clr if a_wins else ("red" if a_is_black else "black")
        winner_ck = A if a_wins else B
        rs, bs = (1.0, 0.0) if winner == "red" else (0.0, 1.0)
    else:
        winner, winner_ck, rs, bs = None, None, 0.5, 0.5
    moves, ai, bi = [], 0, 0
    for ply in range(n_moves):
        player = "red" if ply % 2 == 0 else "black"
        if player == a_clr:
            v = a_values[ai] if a_values is not None else 0.0
            t1 = a_top1[ai] if isinstance(a_top1, (list, tuple)) else a_top1
            rk = a_rank[ai] if isinstance(a_rank, (list, tuple)) else a_rank
            moves.append(make_ply(ply, player, v, top1=t1, rank=rk))
            ai += 1
        else:
            v = b_values[bi] if b_values is not None else 0.0
            moves.append(make_ply(ply, player, v))
            bi += 1
    row = {
        "task_id": game_idx if task_id is None else task_id,
        "pairing_id": PAIRING, "game_idx": game_idx,
        "red_checkpoint": red_ck, "black_checkpoint": black_ck,
        "winner": winner, "winner_checkpoint": winner_ck, "reason": reason,
        "n_moves": n_moves, "red_score": rs, "black_score": bs,
        "replay_path": f"{replay_dir}/game_{game_idx:06d}.json",
    }
    replay = {
        "schema_version": 1, "pairing_id": PAIRING, "game_idx": game_idx,
        "task_id": row["task_id"], "seed": 1000 + game_idx, "board_size": 24,
        "red_checkpoint": red_ck, "black_checkpoint": black_ck,
        "winner": winner, "winner_checkpoint": winner_ck, "reason": reason,
        "n_moves": n_moves, "moves": moves,
    }
    return row, replay
