#!/usr/bin/env python3
"""
Analyze TwixT self-play per-game JSON logs to pick a sensible max move cap.

Expected per-game JSON structure (example):
  - meta.board_size (int)
  - meta.n_moves (int)   # number of plies / turns (1-based move count)
  - meta.reason (str)    # e.g., "win", "timeout", "board_full", "state_cap", "unknown"
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GameRec:
    path: str
    n_moves: int
    reason: str
    winner: str


def _safe_get(d: Dict[str, Any], path: Tuple[str, ...], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_games(root_dir: str, board_size: int) -> List[GameRec]:
    games: List[GameRec] = []
    for base, _, files in os.walk(root_dir):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            p = os.path.join(base, fn)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            bs = _safe_get(data, ("meta", "board_size"))
            if bs != board_size:
                continue

            n_moves = _safe_get(data, ("meta", "n_moves"))
            if n_moves is None:
                # fallback: derive from moves array length
                moves = data.get("moves", [])
                if isinstance(moves, list):
                    n_moves = len(moves)
            if not isinstance(n_moves, int) or n_moves <= 0:
                continue

            reason = _safe_get(data, ("meta", "reason"), default="unknown") or "unknown"
            winner = data.get("winner", "unknown") or "unknown"
            games.append(GameRec(path=p, n_moves=n_moves, reason=str(reason), winner=str(winner)))
    return games


def summarize_lengths(lengths: np.ndarray) -> Dict[str, float]:
    pct = [50, 75, 90, 95, 99]
    out = {
        "count": float(len(lengths)),
        "min": float(np.min(lengths)),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "max": float(np.max(lengths)),
    }
    for p in pct:
        out[f"p{p:02d}"] = float(np.percentile(lengths, p))
    return out


def summarize_by_reason(games: List[GameRec]) -> Dict[str, Dict[str, float]]:
    by: Dict[str, List[int]] = {}
    for g in games:
        by.setdefault(g.reason, []).append(g.n_moves)

    out: Dict[str, Dict[str, float]] = {}
    for reason, vals in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        arr = np.array(vals, dtype=np.int32)
        out[reason] = summarize_lengths(arr)
    return out


def cap_hit_rate(lengths: np.ndarray, cap: int) -> float:
    # "cap-hit" means game reached cap exactly or exceeded (depending on logging)
    # Using >= is robust if some logs record n_moves slightly beyond cap.
    return float(np.mean(lengths >= cap))


def pick_cap(
    lengths: np.ndarray,
    candidate_caps: List[int],
    target_cap_hit_rate: float,
) -> Tuple[int, Dict[int, float]]:
    rates = {cap: cap_hit_rate(lengths, cap) for cap in candidate_caps}
    p99 = float(np.percentile(lengths, 99))
    # choose smallest cap that is >= p99 and has acceptable cap-hit rate
    valid = [cap for cap in candidate_caps if cap >= p99 and rates[cap] <= target_cap_hit_rate]
    if valid:
        return min(valid), rates
    # fallback: choose cap with lowest hit rate, then smallest cap among ties
    best = min(candidate_caps, key=lambda c: (rates[c], c))
    return best, rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing per-game JSON files")
    ap.add_argument("--board-size", type=int, default=24, help="Board size to include (default: 24)")
    ap.add_argument(
        "--caps",
        default="180,240,300,360,420",
        help="Comma-separated candidate caps (plies). Default: 180,240,300,360,420",
    )
    ap.add_argument(
        "--target-cap-hit-rate",
        type=float,
        default=0.01,
        help="Desired max fraction of games that hit the cap (default: 0.01 = 1%%)",
    )
    args = ap.parse_args()

    caps = [int(x.strip()) for x in args.caps.split(",") if x.strip()]
    caps = sorted(set(caps))

    games = load_games(args.dir, board_size=args.board_size)
    if not games:
        raise SystemExit(f"No valid games found for board_size={args.board_size} in {args.dir}")

    lengths = np.array([g.n_moves for g in games], dtype=np.int32)

    overall = summarize_lengths(lengths)
    by_reason = summarize_by_reason(games)

    recommended, rates = pick_cap(lengths, caps, target_cap_hit_rate=args.target_cap_hit_rate)

    print(f"\n=== TwixT Game Length Summary (board_size={args.board_size}) ===")
    print(f"Games: {int(overall['count'])}")
    print(
        "Lengths (plies): "
        f"min={overall['min']:.0f}  mean={overall['mean']:.1f}  std={overall['std']:.1f}  "
        f"p50={overall['p50']:.0f}  p75={overall['p75']:.0f}  p90={overall['p90']:.0f}  "
        f"p95={overall['p95']:.0f}  p99={overall['p99']:.0f}  max={overall['max']:.0f}"
    )

    print("\nCap-hit rates (fraction of games with n_moves >= cap):")
    for cap in caps:
        print(f"  cap={cap:>4d}: {rates[cap]*100:5.2f}%")

    print(
        f"\nRecommended cap (rule: smallest cap >= p99 and cap-hit <= {args.target_cap_hit_rate*100:.1f}%): "
        f"{recommended}"
    )

    print("\nBreakdown by reason:")
    for reason, stats in by_reason.items():
        print(
            f"  {reason:>10s}: n={int(stats['count'])} "
            f"mean={stats['mean']:.1f} p95={stats['p95']:.0f} p99={stats['p99']:.0f} max={stats['max']:.0f}"
        )

    # Optional: show top 10 longest games for inspection
    top = sorted(games, key=lambda g: g.n_moves, reverse=True)[:10]
    print("\nTop 10 longest games:")
    for g in top:
        print(f"  n_moves={g.n_moves:>4d} reason={g.reason:<10s} winner={g.winner:<6s} file={os.path.basename(g.path)}")


if __name__ == "__main__":
    main()