"""Checkpoint tournament: many pairings in one flat task list. Builder + CLI."""
from __future__ import annotations

import os

# short_id / resolve_checkpoint live in eval_runner (shared low-level module)
# and are re-exported here so callers/tests can import them from either place.
# This avoids a circular import (eval_summary also needs short_id).
from .eval_runner import build_pairing_tasks, short_id, resolve_checkpoint


def build_tournament_tasks(pairings, games: int, base_seed: int):
    """Flat task list across all pairings. pairings: list[(a_ckpt, b_ckpt)].

    Each pairing gets a distinct pairing_index, so task_ids and seeds never
    collide (stride = GAMES_PER_PAIRING_LIMIT).
    """
    tasks = []
    for idx, (a_ckpt, b_ckpt) in enumerate(pairings):
        pairing_id = f"{short_id(a_ckpt)}_vs_{short_id(b_ckpt)}"
        tasks.extend(build_pairing_tasks(pairing_id, a_ckpt, b_ckpt, games,
                                         base_seed, pairing_index=idx))
    return tasks


import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone

from .eval_runner import EvalConfig, run_game_tasks
from .eval_summary import summarize_tournament


def parse_pairings(spec: str, checkpoints_dir: str):
    """Parse "A:B,A:C" into [(pathA, pathB), ...] resolving short ids."""
    pairings = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        a, b = token.split(":")
        pairings.append((resolve_checkpoint(a.strip(), checkpoints_dir),
                         resolve_checkpoint(b.strip(), checkpoints_dir)))
    if not pairings:
        raise ValueError("no pairings parsed")
    return pairings


def round_robin_pairings(checkpoints):
    """All C(n,2) pairs, preserving input order."""
    pairs = []
    for i in range(len(checkpoints)):
        for j in range(i + 1, len(checkpoints)):
            pairs.append((checkpoints[i], checkpoints[j]))
    return pairs


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def run_tournament(pairings, games, base_seed, config, workers, output_dir,
                   evaluator_factory=None):
    """Run all pairings through ONE flat task list / ONE pool. Returns the
    combined tournament summary and writes per-pairing + combined JSON."""
    tasks = build_tournament_tasks(pairings, games, base_seed)
    results = run_game_tasks(tasks, workers=workers, config=config,
                             evaluator_factory=evaluator_factory)
    config_dict = {**asdict(config), "base_seed": base_seed, "workers": workers}
    summary = summarize_tournament(results, pairings, config_dict)
    summary["git_commit"] = _git_commit()
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(output_dir, exist_ok=True)
    for ps in summary["pairings"]:
        with open(os.path.join(output_dir, f"{ps['pairing_id']}.json"), "w") as fh:
            json.dump(ps, fh, indent=2)
    with open(os.path.join(output_dir, "tournament.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _build_arg_parser():
    ap = argparse.ArgumentParser(description="Run a checkpoint tournament.")
    ap.add_argument("--checkpoints-dir", default="checkpoints/alphazero-v2-staged")
    ap.add_argument("--pairings", default=None,
                    help='e.g. "0419:0379,0419:0339"')
    ap.add_argument("--checkpoints", default=None,
                    help="comma list for --round-robin, e.g. 0419,0379,0339")
    ap.add_argument("--round-robin", action="store_true")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--board-size", type=int, default=24)
    ap.add_argument("--mcts-sims", type=int, default=400)
    ap.add_argument("--mcts-eval-batch-size", type=int, default=14)
    ap.add_argument("--mcts-stall-flush-sims", type=int, default=48)
    ap.add_argument("--selection-mode", default="opening_temperature",
                    choices=["opening_temperature", "argmax"])
    ap.add_argument("--opening-temp-plies", type=int, default=20)
    ap.add_argument("--temp-high", type=float, default=1.0)
    ap.add_argument("--temp-low", type=float, default=0.1)
    ap.add_argument("--max-moves", type=int, default=280)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--base-seed", type=int, default=12345)
    ap.add_argument("--output-dir", required=True)
    return ap


def _config_from_args(args) -> EvalConfig:
    return EvalConfig(
        board_size=args.board_size, mcts_sims=args.mcts_sims,
        mcts_eval_batch_size=args.mcts_eval_batch_size,
        mcts_stall_flush_sims=args.mcts_stall_flush_sims,
        selection_mode=args.selection_mode,
        opening_temp_plies=args.opening_temp_plies,
        temp_high=args.temp_high, temp_low=args.temp_low,
        max_moves=args.max_moves,
    )


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if args.round_robin:
        if not args.checkpoints:
            raise SystemExit("--round-robin requires --checkpoints")
        ckpts = [resolve_checkpoint(t.strip(), args.checkpoints_dir)
                 for t in args.checkpoints.split(",") if t.strip()]
        pairings = round_robin_pairings(ckpts)
    elif args.pairings:
        pairings = parse_pairings(args.pairings, args.checkpoints_dir)
    else:
        raise SystemExit("provide --pairings or --round-robin --checkpoints")

    for a, b in pairings:
        for p in (a, b):
            if not os.path.exists(p):
                raise SystemExit(f"checkpoint not found: {p}")

    summary = run_tournament(
        pairings=pairings, games=args.games, base_seed=args.base_seed,
        config=_config_from_args(args), workers=args.workers,
        output_dir=args.output_dir,
    )
    print(f"{'pairing':<16} {'a_rate':>7} {'elo':>7}  verdict")
    for row in summary["table"]:
        print(f"{row['pairing_id']:<16} {row['a_score_rate']:>7.4f} "
              f"{row['elo_estimate']:>7.1f}  {row['verdict']}")


if __name__ == "__main__":
    main()
