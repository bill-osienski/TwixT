"""Shared test fixtures for the goal-line trigger probe. No MLX.

legal_replay builds a synthetic replay whose moves are a legal TwixtState game
prefix, so position_state (which applies moves to a real TwixtState and validates
legality) can replay any prefix. Deterministic: always takes the first legal move.
"""
from scripts.GPU.alphazero.game.twixt_state import TwixtState


def legal_replay(n_plies, *, board_size=24, game_idx=0, winner="red", reason="win"):
    state = TwixtState(active_size=board_size, to_move="red",
                       max_plies_limit=board_size * board_size)
    moves = []
    for ply in range(n_plies):
        if state.winner() is not None:
            break
        legal = state.legal_moves()
        if not legal:
            break
        r, c = legal[0]
        moves.append({
            "ply": ply, "player": state.to_move, "row": r, "col": c,
            "root_value": 0.0, "root_top1_share": 0.5,
            "selected_visit_rank": 1, "selected_visit_count": 100,
            "root_total_visits": 100, "n_legal": len(legal),
        })
        state = state.apply_move((r, c))
    return {
        "schema_version": 1, "game_idx": game_idx, "board_size": board_size,
        "n_moves": len(moves), "winner": winner, "reason": reason, "moves": moves,
    }
