from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.GPU.ai.heuristics import DEFAULT_KNOBS
from scripts.GPU.selfplay.engine import TwixtSimulator
from scripts.GPU.selfplay.parallel import DEFAULT_WORKERS, run_games_parallel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Python/GPU self-play throughput.")
    parser.add_argument("--games", type=int, default=8, help="Number of games to run.")
    parser.add_argument("--depth", type=int, default=2, help="Search depth.")
    parser.add_argument("--board", type=int, default=24, help="Board size.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument("--max-moves", type=int, default=120, help="Max moves per game.")
    parser.add_argument("--stall-limit", type=int, default=40, help="Stall limit.")
    parser.add_argument("--mode", choices=("training", "debug"), default="training", help="Play mode.")
    parser.add_argument(
        "--log",
        default=str(Path(__file__).resolve().parents[2] / "logs" / "bench-selfplay.json"),
        help="Append results to this JSON file.",
    )
    parser.add_argument("--parallel", action="store_true", help="Use parallel workers.")
    parser.add_argument("--workers", type=int, default=0, help="Worker count (0=auto).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    knobs: Dict[str, float] = dict(DEFAULT_KNOBS)
    sim = TwixtSimulator(
        board_size=args.board,
        max_moves=args.max_moves,
        stall_limit=args.stall_limit,
    )

    t0 = time.time()
    if args.parallel:
        workers = args.workers if args.workers > 0 else DEFAULT_WORKERS
        outcomes = run_games_parallel(
            knobs=knobs,
            depth=args.depth,
            games=args.games,
            seed=args.seed,
            board_size=args.board,
            workers=workers,
            mode=args.mode,
        )
    else:
        outcomes = sim.play_batch(
            knobs,
            seeds=[args.seed + i for i in range(args.games)],
            depth=args.depth,
            mode=args.mode,
        )
    dt = time.time() - t0

    games = len(outcomes)
    gps = games / max(1e-6, dt)
    wins = {"red": 0, "black": 0, "draw": 0}
    reasons = {"win": 0, "stall": 0, "max_moves": 0, "no_moves": 0, "error": 0}
    total_moves = 0
    stagnation_max = 0
    progress_events = 0
    opening_random = 0
    score_vals = []
    for out in outcomes:
        wins[out.winner] = wins.get(out.winner, 0) + 1
        reasons[out.reason] = reasons.get(out.reason, 0) + 1
        total_moves += int(out.total_moves)
        stats = out.stats or {}
        stagnation_max += int(stats.get("stagnation_max", 0))
        progress_events += int(stats.get("progress_events", 0))
        opening_random += int(stats.get("opening_random_moves", 0))
        if stats.get("avg_search_score") is not None:
            score_vals.append(float(stats["avg_search_score"]))

    avg_moves = total_moves / max(1, games)
    avg_stagnation_max = stagnation_max / max(1, games)
    avg_progress_events = progress_events / max(1, games)
    avg_opening_random = opening_random / max(1, games)
    avg_search_score = sum(score_vals) / max(1, len(score_vals)) if score_vals else None
    summary = {
        "timestamp": time.time(),
        "mode": args.mode,
        "depth": args.depth,
        "games": games,
        "throughput_gps": gps,
        "avg_moves": avg_moves,
        "wins": wins,
        "reasons": reasons,
        "diagnostics": {
            "avg_stagnation_max": avg_stagnation_max,
            "avg_progress_events": avg_progress_events,
            "avg_opening_random_moves": avg_opening_random,
            "avg_search_score": avg_search_score,
        },
        "max_moves": args.max_moves,
        "stall_limit": args.stall_limit,
        "parallel": bool(args.parallel),
    }

    print(
        f"self-play throughput: {gps:.2f} games/sec "
        f"({games} games, depth={args.depth}, mode={args.mode})"
    )
    print(
        "outcomes: "
        f"red={wins['red']} black={wins['black']} draw={wins['draw']} "
        f"avg_moves={avg_moves:.1f}"
    )
    print(
        "reasons: "
        f"win={reasons['win']} stall={reasons['stall']} "
        f"max_moves={reasons['max_moves']} no_moves={reasons['no_moves']} "
        f"error={reasons['error']}"
    )
    print(
        "diagnostics: "
        f"avg_stagnation_max={avg_stagnation_max:.1f} "
        f"avg_progress_events={avg_progress_events:.1f} "
        f"avg_opening_random_moves={avg_opening_random:.1f} "
        f"avg_search_score={avg_search_score if avg_search_score is not None else 'n/a'}"
    )

    log_path = Path(args.log).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        existing = json.loads(log_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or "runs" not in existing:
            existing = {"runs": []}
    else:
        existing = {"runs": []}
    existing["runs"].append(summary)
    log_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote benchmark summary -> {log_path}")


if __name__ == "__main__":
    main()
