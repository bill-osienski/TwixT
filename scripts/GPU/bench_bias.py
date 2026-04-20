from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.ai.heuristics import DEFAULT_KNOBS
from scripts.GPU.selfplay.engine import TwixtSimulator, compute_bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure red/black bias in training mode.")
    parser.add_argument("--games", type=int, default=10, help="Number of games to run.")
    parser.add_argument("--depth", type=int, default=1, help="Search depth.")
    parser.add_argument("--board", type=int, default=24, help="Board size.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument("--max-moves", type=int, default=120, help="Max moves per game.")
    parser.add_argument("--stall-limit", type=int, default=40, help="Stall limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    knobs: Dict[str, float] = dict(DEFAULT_KNOBS)
    sim = TwixtSimulator(
        board_size=args.board,
        max_moves=args.max_moves,
        stall_limit=args.stall_limit,
    )

    outcomes = sim.play_batch(
        knobs,
        seeds=[args.seed + i for i in range(args.games)],
        depth=args.depth,
        mode="training",
    )

    red = sum(1 for o in outcomes if o.winner == "red")
    black = sum(1 for o in outcomes if o.winner == "black")
    draw = sum(1 for o in outcomes if o.winner == "draw")
    bias = compute_bias(outcomes)

    print(
        f"bias: {bias:+.3f}  red={red} black={black} draw={draw} "
        f"(games={len(outcomes)}, depth={args.depth})"
    )


if __name__ == "__main__":
    main()
