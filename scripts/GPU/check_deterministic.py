from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.game.state import GameState
from scripts.GPU.game.rules import apply_move
from scripts.GPU.ai.search import choose_move
from scripts.GPU.ai.heuristics import DEFAULT_KNOBS


def main() -> None:
    state = GameState()
    for mv in [
        ("red", 11, 11),
        ("black", 12, 13),
        ("red", 10, 12),
    ]:
        state.to_move = mv[0]
        state = apply_move(state, mv[1], mv[2])

    state.to_move = "black"

    r1 = choose_move(
        state,
        knobs=DEFAULT_KNOBS,
        depth=2,
        top_n=12,
        use_value_model=False,
        temperature=0.0,
        rng=None,
        mode="debug",
    )
    r2 = choose_move(
        state,
        knobs=DEFAULT_KNOBS,
        depth=2,
        top_n=12,
        use_value_model=False,
        temperature=0.0,
        rng=None,
        mode="debug",
    )

    if (r1.row, r1.col) != (r2.row, r2.col):
        raise SystemExit(f"Determinism check failed: {r1.row},{r1.col} vs {r2.row},{r2.col}")

    print(f"Determinism OK: move=({r1.row},{r1.col})")


if __name__ == "__main__":
    main()
