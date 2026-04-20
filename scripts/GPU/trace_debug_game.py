from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.ai.heuristics import DEFAULT_KNOBS
from scripts.GPU.selfplay.engine import TwixtSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single debug game with trace output.")
    parser.add_argument("--depth", type=int, default=2, help="Search depth.")
    parser.add_argument("--board", type=int, default=24, help="Board size.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--output", default="logs/debug-trace.json", help="Output file.")
    parser.add_argument("--max-plies", type=int, default=80, help="Max plies to log.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    knobs: Dict[str, float] = dict(DEFAULT_KNOBS)
    knobs["debug_trace"] = 1
    knobs["debug_sample_rate"] = 1.0
    knobs["debug_max_plies"] = int(args.max_plies)

    sim = TwixtSimulator(board_size=args.board)
    outcome = sim.play_one(
        knobs,
        seed=args.seed,
        depth=args.depth,
        mode="debug",
    )

    payload = {
        "seed": args.seed,
        "depth": args.depth,
        "winner": outcome.winner,
        "reason": outcome.reason,
        "total_moves": outcome.total_moves,
        "starting_player": outcome.starting_player,
        "stats": outcome.stats,
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote debug trace -> {out_path}")


if __name__ == "__main__":
    main()
