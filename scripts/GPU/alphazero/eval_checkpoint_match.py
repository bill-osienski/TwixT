"""Checkpoint match: one A-vs-B pairing.

Builds one pairing's balanced-color tasks, runs them through the shared
eval_runner pool, aggregates a summary, and writes per-game JSONL + summary
JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone

from .eval_runner import EvalConfig, build_pairing_tasks, run_game_tasks, short_id
from .eval_summary import summarize_match


def build_match_tasks(a_ckpt, b_ckpt, games, base_seed, pairing_id):
    """Tasks for a single pairing (pairing_index fixed at 0)."""
    return build_pairing_tasks(pairing_id, a_ckpt, b_ckpt, games, base_seed,
                               pairing_index=0)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _write_outputs(output, summary, results):
    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    stem, _ext = os.path.splitext(output)
    games_path = f"{stem}_games.jsonl"
    with open(games_path, "w") as fh:
        for r in results:  # already sorted by (pairing_id, game_idx)
            fh.write(json.dumps(asdict(r)) + "\n")
    with open(output, "w") as fh:
        json.dump(summary, fh, indent=2)


def run_match(a_ckpt, b_ckpt, games, base_seed, config, workers, output,
              pairing_id=None, evaluator_factory=None):
    """Run a full match and write outputs. Returns the summary dict."""
    if pairing_id is None:
        pairing_id = f"{short_id(a_ckpt)}_vs_{short_id(b_ckpt)}"
    tasks = build_match_tasks(a_ckpt, b_ckpt, games, base_seed, pairing_id)
    results = run_game_tasks(tasks, workers=workers, config=config,
                             evaluator_factory=evaluator_factory)
    config_dict = {**asdict(config), "base_seed": base_seed, "workers": workers}
    summary = summarize_match(results, a_ckpt, b_ckpt, pairing_id, config_dict)
    summary["git_commit"] = _git_commit()
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    if output:
        _write_outputs(output, summary, results)
    return summary


def _build_arg_parser():
    ap = argparse.ArgumentParser(description="Run a checkpoint A-vs-B match.")
    ap.add_argument("--checkpoint-a", required=True)
    ap.add_argument("--checkpoint-b", required=True)
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
    ap.add_argument("--output", required=True)
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
    for path in (args.checkpoint_a, args.checkpoint_b):
        if not os.path.exists(path):
            raise SystemExit(f"checkpoint not found: {path}")
    summary = run_match(
        a_ckpt=args.checkpoint_a, b_ckpt=args.checkpoint_b, games=args.games,
        base_seed=args.base_seed, config=_config_from_args(args),
        workers=args.workers, output=args.output,
    )
    print(f"{summary['pairing_id']}: a_score_rate={summary['a_score_rate']:.4f} "
          f"elo={summary['elo_estimate']:.1f} "
          f"CI95={summary['elo_ci95']} verdict={summary['verdict']}")


if __name__ == "__main__":
    main()
