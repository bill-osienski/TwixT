from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from ..game.state import GameState


def order_moves(
    state: GameState,
    moves: Iterable[Tuple[int, int]],
    knobs: Dict[str, float],
    *,
    top_k: int = 32,
) -> List[Tuple[int, int]]:
    """Return top-K moves in best-first order.

    Stub: returns the first `top_k` legal moves.

    TODO: implement priority-based move ordering:
    - score all moves via vectorized heuristics
    - keep top-N
    """
    out: List[Tuple[int, int]] = []
    for m in moves:
        out.append(m)
        if len(out) >= top_k:
            break
    return out
