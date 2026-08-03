"""Run the single-game Phase 0 inheritance technical preflight.

The resulting rows are serially correlated technical telemetry, not evidence.
The observed game must be excluded from the convergence-atlas corpus.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

from .inheritance_probe import (
    InheritanceProbeConfig,
    SearchRow,
    evaluate_verdict,
    summarize,
)
from .mcts import MCTSConfig
from .self_play import play_game


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _worktree_clean() -> bool:
    return subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ) == ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 0 inheritance preflight (one game)"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--max-moves", type=int, default=280)
    parser.add_argument("--active-size", type=int, default=24)
    parser.add_argument("--out", required=True, help="output JSON path (required)")
    args = parser.parse_args()

    # Import lazily to keep MLX out of module scope and GPU-free tests.
    # This builds one compile=True evaluator and reuses it for the full game.
    from .eval_runner import _default_evaluator_factory

    config = MCTSConfig(
        n_simulations=args.sims,
        eval_batch_size=14,
        stall_flush_sims=48,
        pending_virtual_visits=8,
    )
    record = play_game(
        evaluator=_default_evaluator_factory(args.checkpoint),
        mcts_config=config,
        rng=random.Random(args.seed),
        max_moves=args.max_moves,
        add_noise=False,
        active_size=args.active_size,
        game_id=args.seed,
        inheritance_probe_config=InheritanceProbeConfig(),
    )

    probe_record = record.inheritance_probe_record
    if probe_record is None:
        raise RuntimeError("inheritance probe produced no record")
    raw_rows = probe_record["rows"]
    rows = [SearchRow(**row) for row in raw_rows]
    summary = summarize(rows)
    verdict = evaluate_verdict(summary)

    artifact = {
        "rows": raw_rows,
        "summary": summary,
        "verdict": verdict,
        "provenance": {
            "git_head": _git_head(),
            "worktree_clean": _worktree_clean(),
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "n_simulations": args.sims,
            "batching": [
                config.eval_batch_size,
                config.stall_flush_sims,
                config.pending_virtual_visits,
            ],
            "add_noise": False,
            "note": (
                "TECHNICAL PREFLIGHT, NOT EVIDENCE. Exclude this game from "
                "the atlas corpus."
            ),
        },
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2, sort_keys=True))

    print(f"verdict: {verdict['verdict']}")
    for reason in verdict["reasons"]:
        print(f"  reason: {reason}")
    if not verdict["coverage_complete"]:
        print(
            "  COVERAGE INCOMPLETE -- unobserved post-opening phases: "
            + ", ".join(verdict["unobserved_post_opening_phases"])
            + " (an unobserved phase supplies no evidence either way)"
        )
    print(f"artifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
