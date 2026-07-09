#!/usr/bin/env python3
"""Train AlphaZero for TwixT.

Usage:
    python -m scripts.GPU.alphazero.train --iterations 10 --games-per-iter 5

    # Full training run
    python -m scripts.GPU.alphazero.train \\
        --iterations 100 \\
        --games-per-iter 25 \\
        --train-steps 100 \\
        --simulations 800 \\
        --checkpoint-dir checkpoints/alphazero

    # Resume from checkpoint
    python -m scripts.GPU.alphazero.train \\
        --resume checkpoints/alphazero/model_iter_0050.safetensors
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the trainer CLI parser. Importable from tests to assert
    defaults / validation behavior without invoking main()/parse_args().

    Single source of truth for the parser definition — main() calls this
    and then parse_args(); tests call this and parse_args(arglist).
    """
    parser = argparse.ArgumentParser(
        description="Train AlphaZero for TwixT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Training iterations
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)",
    )
    parser.add_argument(
        "--games-per-iter",
        type=int,
        default=25,
        help="Self-play games per iteration (default: 25)",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="Training steps per iteration (default: auto from table, 0 to skip)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )

    # MCTS settings
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="MCTS simulations per move (default: use SIMS_TABLE per board size)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum moves per game (default: 200)",
    )
    parser.add_argument(
        "--mcts-eval-batch-size",
        type=int,
        default=14,
        help="MCTS leaves per NN batch (default: 14)",
    )
    parser.add_argument(
        "--mcts-pending-virtual-visits",
        type=int,
        default=8,
        help="MCTS virtual visits for pending leaves (default: 8)",
    )
    parser.add_argument(
        "--mcts-stall-flush-sims",
        type=int,
        default=16,
        help="MCTS flush if no new pending leaf in N sims (0=disabled, default: 16)",
    )

    # MCTS exploration tuning
    parser.add_argument("--dirichlet-alpha", type=float, default=None,
        help="Dirichlet noise alpha (default: 0.3)")
    parser.add_argument("--dirichlet-eps", type=float, default=None,
        help="Dirichlet noise mixing weight 0-1 (default: 0.25)")
    parser.add_argument("--temp-high", type=float, default=None,
        help="Temperature for early game moves (default: 1.0)")
    parser.add_argument("--temp-low", type=float, default=None,
        help="Temperature for late game moves (default: 0.1)")
    parser.add_argument("--temp-threshold-ply", type=int, default=None,
        help="Ply at which temperature drops (default: 20)")

    # Network architecture
    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="Network hidden channels (default: 128)",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=6,
        help="Network residual blocks (default: 6)",
    )

    # Optimizer settings
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=1e-4,
        help="L2 regularization weight (default: 1e-4)",
    )
    parser.add_argument(
        "--value-lr-scale",
        type=float,
        default=0.1,
        help="Value head LR multiplier (default: 0.1)",
    )
    parser.add_argument(
        "--value-grad-max-norm",
        type=float,
        default=0.5,
        help="Max gradient norm for value head (default: 0.5)",
    )
    parser.add_argument(
        "--value-weight",
        type=float,
        default=None,
        help="Override value loss weight (default 0.5 from train())",
    )
    parser.add_argument(
        "--progress-weighted-value-loss",
        dest="progress_weighted",
        action="store_true",
        default=True,
        help="Use progress-weighted value loss (default ON)",
    )
    parser.add_argument(
        "--no-progress-weighted-value-loss",
        dest="progress_weighted",
        action="store_false",
        help="Disable progress-weighted value loss (use unweighted MSE)",
    )
    parser.add_argument(
        "--progress-weight-floor",
        type=float,
        default=0.25,
        help="Progress-weighted value loss floor [0, 1] (default 0.25)",
    )

    # Buffer settings
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="Replay buffer size (default: 100000)",
    )

    # Checkpoint settings
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/alphazero",
        help="Checkpoint directory (default: checkpoints/alphazero)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Load network weights only (no training state restore).",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Multi-process self-play
    default_workers = max(1, (os.cpu_count() or 4) - 2)
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,  # Single-process by default for stability
        help=f"Parallel self-play workers (default: 1, max recommended: {default_workers})",
    )

    # Game replay saving
    parser.add_argument(
        "--no-save-games",
        action="store_true",
        help="Disable saving game replays to scripts/GPU/logs/games/",
    )
    parser.add_argument(
        "--games-dir",
        type=str,
        default=None,
        help="Override games output directory (default: scripts/GPU/logs/games/)",
    )

    # Diagnostics
    parser.add_argument(
        "--opening-debug",
        action="store_true",
        help="Enable opening diagnostics (logged in self-play workers)",
    )

    # Mirror augmentation
    parser.add_argument(
        "--mirror-prob", type=float, default=0.5,
        help="Probability of mirror augmentation per position (default: 0.5, 0=off)",
    )

    # Opening exploration boost
    parser.add_argument("--opening-noise-ply", type=int, default=None,
        help="Ply count for opening noise boost (default: 0 = disabled)")
    parser.add_argument("--opening-dirichlet-alpha", type=float, default=None,
        help="Dirichlet alpha for opening noise boost (default: 1.0)")
    parser.add_argument("--opening-dirichlet-eps", type=float, default=None,
        help="Dirichlet eps for opening noise boost (default: 0.5)")

    # Edge-band prior penalty
    parser.add_argument("--root-edge-band-penalty", type=float, default=None,
        help="Edge-band prior penalty λ. If set, prior *= exp(-λ) for edge-band moves")
    parser.add_argument("--root-edge-band-penalty-ply", type=int, default=None,
        help="Apply edge-band penalty for ply < this value")
    parser.add_argument("--root-edge-band-width", type=int, default=None,
        help="Edge-band width in cells (MCTSConfig default: 2)")

    # Near-corner prior penalty
    parser.add_argument("--root-near-corner-penalty", type=float, default=None,
        help="Near-corner prior penalty λ. If set, prior *= exp(-λ) for near-corner moves")
    parser.add_argument("--root-near-corner-penalty-ply", type=int, default=None,
        help="Apply near-corner penalty for ply < this value")
    parser.add_argument("--root-near-corner-radius", type=int, default=None,
        help="Near-corner Chebyshev radius (MCTSConfig default: 2)")
    # Phase 2: early-only near-corner override — stronger penalty for ply 0 / 1
    # while the broader window uses --root-near-corner-penalty. Both args must
    # be > 0 for the override to take effect.
    parser.add_argument("--root-near-corner-penalty-early", type=float, default=None,
        help="Early-override near-corner penalty λ_early used for ply < "
             "--root-near-corner-penalty-early-plies. Replaces the baseline "
             "penalty during the early window; baseline applies after.")
    parser.add_argument("--root-near-corner-penalty-early-plies", type=int, default=None,
        help="Apply --root-near-corner-penalty-early for ply < this value "
             "(0 disables the override regardless of --root-near-corner-penalty-early).")

    # Curriculum learning
    parser.add_argument(
        "--curriculum-sizes",
        type=str,
        default="8,10,12,16,20,24",
        help="Comma-separated board sizes for curriculum (default: 8,10,12,16,20,24)",
    )
    parser.add_argument(
        "--curriculum-window",
        type=int,
        default=200,
        help="Games window for curriculum metrics (default: 200)",
    )
    parser.add_argument(
        "--curriculum-draw-threshold",
        type=float,
        default=0.3,
        help="Max draw rate for curriculum promotion (default: 0.3)",
    )
    parser.add_argument(
        "--curriculum-min-wins",
        type=int,
        default=5,
        help="Min wins per color for curriculum promotion (default: 5)",
    )

    # Resign parameters (conservative defaults = disabled)
    parser.add_argument("--resign-enabled", action="store_true",
        help="Enable automatic resign when position is hopeless")
    parser.add_argument("--resign-min-ply", type=int, default=80,
        help="Don't resign before this ply (default: 80)")
    parser.add_argument("--resign-threshold", type=float, default=-0.97,
        help="Resign when root value <= this (default: -0.97)")
    parser.add_argument("--resign-window", type=int, default=12,
        help="Sliding window size for resign check (default: 12)")
    parser.add_argument("--resign-k", type=int, default=8,
        help="Resign if K of last W checks meet condition (default: 8)")
    parser.add_argument("--resign-min-visits", type=int, default=200,
        help="Require root visits >= this to resign (default: 200)")
    parser.add_argument("--resign-min-top1-share", type=float, default=0.0,
        help="Require top move's visit share >= this to resign (default: 0 = disabled)")

    # Adjudication parameters (disabled by default)
    parser.add_argument("--adjudicate-enabled", action="store_true",
        help="Enable timeout adjudication (assign winner at max_moves using MCTS eval)")
    parser.add_argument("--adjudicate-min-ply", type=int, default=120,
        help="Don't adjudicate before this ply (default: 120)")
    parser.add_argument("--adjudicate-threshold", type=float, default=0.90,
        help="Adjudicate when |root_value| >= this (default: 0.90)")
    parser.add_argument("--adjudicate-min-visits", type=int, default=200,
        help="Require root visits >= this to adjudicate (default: 200)")
    parser.add_argument("--adjudicate-min-top1-share", type=float, default=0.0,
        help="Require top move's visit share >= this to adjudicate (default: 0 = disabled)")
    parser.add_argument("--adjudicate-debug", action="store_true",
        help="Print per-timeout ADJ_DEBUG lines showing gate results")

    # Phase 4: per-game replay contribution cap (disabled by default)
    parser.add_argument("--max-positions-per-game", type=int, default=0,
        help="Cap positions a single game contributes to replay (0 = disabled). "
             "Long games get sub-sampled so they do not dominate training.")
    parser.add_argument("--endgame-keep-positions", type=int, default=16,
        help="When capping a game's positions, keep this many tail positions "
             "unconditionally (protects endgame/conversion supervision). "
             "Only takes effect when --max-positions-per-game > 0. Default: 16")
    # Phase 2: inline forced-probe per-iter eval (additive observability)
    parser.add_argument("--probes-path", type=str, default="tests/probes/twixt_probes.json",
        help="Path to curated probe suite JSON (Phase 2). Per-iter forced-tier "
             "NN-only eval runs against this. If missing, the Probe block is "
             "silently skipped (Phase 0 of spec may not be done yet).")
    parser.add_argument("--probes-inline-disable", action="store_true",
        help="Disable per-iter inline forced-probe eval entirely. "
             "Use when probes file is intentionally absent or for max throughput.")

    # Spec 2: conversion auxiliary loss
    parser.add_argument("--conversion-policy-loss-enabled", action="store_true",
        help="Enable conversion auxiliary policy loss on closeout-eligible positions.")
    parser.add_argument("--conversion-policy-loss-weight", type=float, default=0.05,
        help="Weight λ for the conversion auxiliary loss term (default: 0.05).")
    parser.add_argument("--conversion-completion-weight", type=float, default=1.0,
        help="Target weight for endpoint_completion_moves (default: 1.0).")
    parser.add_argument("--conversion-reducer-weight", type=float, default=0.35,
        help="Target weight for distance_reducing_moves (default: 0.35). "
             "Must be <= --conversion-completion-weight.")
    parser.add_argument("--conversion-max-total-goal-distance", type=int, default=2,
        help="Eligibility threshold on total_goal_distance (default: 2). "
             "Range [1, 3]; first experiment uses 2, widens to 3 later.")
    # Track 2: sample boost
    parser.add_argument("--conversion-sample-boost", type=float, default=1.0,
        help="Multiplier on uniform-eligible expectation (default: 1.0 = pure uniform).")
    parser.add_argument("--conversion-max-batch-fraction", type=float, default=0.15,
        help="Hard cap on eligible fraction per batch (default: 0.15).")

    # Post-opening sharp-drop calibration (design 2026-06-16)
    parser.add_argument("--post-opening-calibration-enabled", action="store_true",
        help="Enable the post-opening sharp-drop value calibration aux loss.")
    parser.add_argument("--post-opening-calibration-manifest", type=str, default=None,
        help="Path to the calibration TRAIN manifest CSV (required when enabled).")
    parser.add_argument("--post-opening-calibration-target", type=float, default=-0.50,
        help="Soft value target (black perspective) for calibration positions "
             "(default: -0.50).")
    parser.add_argument("--post-opening-calibration-weight", type=float, default=0.02,
        help="Absolute coefficient on the calibration value-loss term "
             "(default: 0.02; NOT multiplied by value_weight).")
    parser.add_argument("--post-opening-calibration-batch-fraction", type=float, default=0.10,
        help="Calibration mini-batch size as a fraction of batch_size (default: 0.10).")
    parser.add_argument("--post-opening-calibration-tag-schedule", type=str, default=None,
        help="Tag-stratified calibration sampling schedule, e.g. "
             "'black_predrop_correction=2,goal_line_retention=1,"
             "old_post_opening_retention=2,red_predrop_retention=1'. When set, "
             "replaces uniform batch-fraction sampling (batch-fraction is ignored).")
    parser.add_argument("--post-opening-calibration-teacher-value-weight", type=float, default=1.0,
        help="v4: weight on the calibration value-MSE term (correction + teacher "
             "retention rows). Default 1.0.")
    parser.add_argument("--post-opening-calibration-teacher-policy-kl-weight", type=float, default=0.25,
        help="v4: weight on the teacher policy cross-entropy (KL) term on "
             "teacher_retention rows only. Default 0.25; 0.0 = value-only ablation.")
    parser.add_argument("--freeze-batchnorm-stats", action="store_true",
        help="Freeze BatchNorm running stats (momentum=0) for the run so they stay at the "
             "loaded base checkpoint; train-mode normalization still uses batch stats. Used "
             "by the v4 teacher-retention calibration (its eval-mode forward reads base "
             "stats so cached teacher targets stay reproducible) and as a frozen-BN control.")
    parser.add_argument("--train-value-head-only", action="store_true",
        help="v8: freeze encoder+policy_head (skip opt_main updates); "
             "only value_head.* tensors train. Pair with "
             "--freeze-batchnorm-stats.")
    parser.add_argument("--train-value-head-and-final-block", action="store_true",
        help="v9: train only value_head.* plus the final residual block "
             "encoder.blocks[last] (skip the whole-trunk opt_main update; "
             "apply just the final block). Mutually exclusive with "
             "--train-value-head-only. Pair with --freeze-batchnorm-stats.")
    parser.add_argument("--guardrail-margin", type=float, default=0.10,
        help="v12: tolerance band (black-value units) for the asymmetric "
             "guardrail hinge; penalize pro-black drift above BASE by more "
             "than this. Default 0.10.")
    parser.add_argument("--post-opening-calibration-gradient-projection",
        action="store_true",
        help="v13/v14b: project the A-correction gradient away from the "
             "guardrail hinge gradient on the caller-selected value-side "
             "surface when they conflict (dot<0). Requires a multi-component "
             "surface: --train-value-head-and-final-block (v13, value_head + "
             "final block) or --train-value-head-and-value-adapter (v14b, "
             "value_head + value_adapter). Off by default; byte-identical to "
             "v12b when off.")
    parser.add_argument("--post-opening-calibration-projection-strength", type=float,
        default=1.0,
        help="v13c: scale the gradient-conflict correction by folding this into "
             "the effective projection weight (strength * calibration weight). "
             "Only affects conflicting steps; 1.0 = v13 behavior.")
    parser.add_argument("--value-adapter", action="store_true",
        help="v14: build a value-only feature-correction adapter (1x1 bottleneck "
             "+ scalar gate init 0) between the encoder and the value head. Off "
             "by default (byte-identical). Required by "
             "--train-value-head-and-value-adapter.")
    parser.add_argument("--value-adapter-bottleneck-width", type=int, default=None,
        help="v14: adapter bottleneck width. Default (None) = hidden // 4.")
    parser.add_argument("--train-value-head-and-value-adapter", action="store_true",
        help="v14: train only value_head.* + value_adapter.* (skip the whole-trunk "
             "opt_main update; encoder/policy/final-block frozen). Mutually "
             "exclusive with --train-value-head-only / "
             "--train-value-head-and-final-block. Requires --value-adapter. Pair "
             "with --freeze-batchnorm-stats.")

    # Track 4: recovery / extreme-closeout-drift telemetry (default on; free)
    parser.add_argument("--recovery-bucket-enabled", action="store_true", default=True,
        help="Enable recovery / extreme-closeout-drift telemetry (default: on).")
    parser.add_argument("--no-recovery-bucket", dest="recovery_bucket_enabled",
        action="store_false",
        help="Disable recovery / extreme-closeout-drift telemetry.")
    parser.add_argument("--recovery-dominant-unavailable-threshold", type=int, default=10,
        help="DU-moves threshold for recovery bucket (default: 10).")
    parser.add_argument("--recovery-delay-threshold", type=int, default=20,
        help="conversion_delay_plies threshold for recovery bucket (default: 20).")

    # Spec 3 Fix 1: td=1 root visit forcing (closeout tail correction)
    parser.add_argument("--closeout-td1-visit-forcing-enabled", action="store_true",
        help="Enable td=1 endpoint-completion root visit forcing in MCTS (Spec 3 Fix 1).")
    parser.add_argument("--closeout-td1-min-visits", type=int, default=8,
        help="Forced visits per endpoint-completion candidate at td=1 (default: 8).")
    parser.add_argument("--closeout-td1-max-forced-moves", type=int, default=4,
        help="Cap on number of candidate endpoint-completion moves to force per position.")
    parser.add_argument("--closeout-td1-require-high-value", action="store_true",
        help="Gate Fix 1 on root.q_value >= --closeout-td1-high-value-threshold.")
    parser.add_argument("--closeout-td1-high-value-threshold", type=float, default=0.95,
        help="Root q threshold used when --closeout-td1-require-high-value is set.")

    # Spec 3 Fix 2: narrow closeout selection tie-break
    parser.add_argument("--closeout-selection-tiebreak-enabled", action="store_true",
        help="Enable Spec 3 Fix 2 closeout selection tie-break.")
    parser.add_argument("--closeout-selection-tiebreak-max-distance", type=int, default=2,
        help="Max total_goal_distance at which the tie-break may fire (default: 2).")
    parser.add_argument("--closeout-selection-tiebreak-topk", type=int, default=5,
        help="Visit-rank top-k window the closeout candidate must fall within (default: 5).")
    parser.add_argument("--closeout-selection-tiebreak-min-value", type=float, default=0.95,
        help="Root q_value gate for Fix 2 tie-break (default: 0.95).")
    parser.add_argument("--closeout-selection-tiebreak-min-share", type=float, default=0.05,
        help="Minimum visit-share floor for the candidate (default: 0.05).")

    # Spec 4: recovery / re-targeting diagnostic
    _add_recovery_retargeting_args(parser)

    return parser


def _add_recovery_retargeting_args(parser):
    """Spec 4 recovery / re-targeting diagnostic CLI flags."""
    parser.add_argument("--recovery-retargeting-disabled", action="store_true",
                        help="Disable the diagnostic. Default: enabled.")
    parser.add_argument("--recovery-retargeting-collapse-value-threshold", type=float, default=-0.75)
    parser.add_argument("--recovery-retargeting-severe-value-threshold", type=float, default=-0.90)
    parser.add_argument("--recovery-retargeting-diffuse-root-top1-threshold", type=float, default=0.20)
    parser.add_argument("--recovery-retargeting-very-diffuse-root-top1-threshold", type=float, default=0.15)
    parser.add_argument("--recovery-retargeting-delta-threshold", type=float, default=0.50)
    parser.add_argument("--recovery-retargeting-delta-max-current-score", type=float, default=-0.30)
    parser.add_argument("--recovery-retargeting-alternate-component-min-size", type=int, default=4)
    classify_group = parser.add_mutually_exclusive_group()
    classify_group.add_argument("--recovery-retargeting-classify-defense", dest="recovery_retargeting_classify_defense", action="store_true", default=True)
    classify_group.add_argument("--recovery-retargeting-no-classify-defense", dest="recovery_retargeting_classify_defense", action="store_false")
    parser.add_argument("--recovery-retargeting-max-sampled-moves-per-side", type=int, default=32)
    parser.add_argument("--recovery-retargeting-sample-all-moves", action="store_true", default=False)


# Backward-compat alias for callers/tests that imported the previous name.
_build_parser_for_test = build_arg_parser


def _validate_conversion_args(parser: argparse.ArgumentParser, args) -> None:
    """Validate Spec 2 conversion / recovery args. Raises SystemExit via parser.error."""
    if args.conversion_policy_loss_enabled and args.conversion_policy_loss_weight <= 0.0:
        parser.error(
            "--conversion-policy-loss-enabled requires "
            "--conversion-policy-loss-weight > 0.0. "
            "Omit --conversion-policy-loss-enabled to disable conversion entirely."
        )
    if args.conversion_completion_weight <= 0.0:
        parser.error("--conversion-completion-weight must be > 0.0")
    if args.conversion_reducer_weight < 0.0:
        parser.error("--conversion-reducer-weight must be >= 0.0")
    if args.conversion_reducer_weight > args.conversion_completion_weight:
        parser.error(
            "--conversion-reducer-weight must be <= --conversion-completion-weight "
            f"(got reducer={args.conversion_reducer_weight}, "
            f"completion={args.conversion_completion_weight})."
        )
    if not (1 <= args.conversion_max_total_goal_distance <= 3):
        parser.error("--conversion-max-total-goal-distance must be in [1, 3]")
    if args.conversion_sample_boost < 1.0:
        parser.error(
            "--conversion-sample-boost must be >= 1.0 "
            "(omit --conversion-policy-loss-enabled to disable conversion entirely)"
        )
    if not (0.0 <= args.conversion_max_batch_fraction <= 1.0):
        parser.error("--conversion-max-batch-fraction must be in [0.0, 1.0]")

    # Cross-flag warning (not error)
    if (not args.conversion_policy_loss_enabled
            and args.conversion_sample_boost > 1.0):
        print(
            "[WARN] --conversion-sample-boost > 1.0 has no effect when "
            "--conversion-policy-loss-enabled is off. Sample boost stays inactive "
            "and PositionRecord.conversion stays unpopulated."
        )

    if args.recovery_dominant_unavailable_threshold < 1:
        parser.error("--recovery-dominant-unavailable-threshold must be >= 1")
    if args.recovery_delay_threshold < 1:
        parser.error("--recovery-delay-threshold must be >= 1")


def _validate_closeout_td1_args(parser: argparse.ArgumentParser, args) -> None:
    """Validate Spec 3 Fix 1 closeout-td1 CLI args. Raises SystemExit via parser.error."""
    if args.closeout_td1_min_visits < 1:
        parser.error("--closeout-td1-min-visits must be >= 1")
    if args.closeout_td1_max_forced_moves < 1:
        parser.error("--closeout-td1-max-forced-moves must be >= 1")
    if not (0.0 <= args.closeout_td1_high_value_threshold <= 1.0):
        parser.error("--closeout-td1-high-value-threshold must be in [0.0, 1.0]")


def parse_calibration_tag_schedule(raw):
    """Parse 'tag=count,tag=count' into an ordered dict[str, int], or None.

    None/'' -> None (uniform batch-fraction sampling). Each count must be a
    non-negative int. Rejects entries missing '=', empty tags, duplicate tags,
    and an all-zero total.
    """
    if raw in (None, ""):
        return None
    out: dict = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid calibration tag schedule entry {part!r}")
        tag, value = part.split("=", 1)
        tag = tag.strip()
        if not tag:
            raise ValueError("calibration tag schedule contains an empty tag")
        n = int(value)
        if n < 0:
            raise ValueError(
                f"calibration tag schedule count must be >= 0 for {tag!r}")
        if tag in out:
            raise ValueError(f"duplicate calibration tag schedule entry {tag!r}")
        out[tag] = n
    if not out or sum(out.values()) <= 0:
        raise ValueError("calibration tag schedule must draw at least one sample")
    return out


def main():
    parser = build_arg_parser()

    args = parser.parse_args()
    _validate_conversion_args(parser, args)
    _validate_closeout_td1_args(parser, args)
    if sum([args.train_value_head_only,
            args.train_value_head_and_final_block,
            args.train_value_head_and_value_adapter]) > 1:
        parser.error("--train-value-head-only, --train-value-head-and-final-block, "
                     "and --train-value-head-and-value-adapter are mutually exclusive")
    if args.train_value_head_and_value_adapter and not args.value_adapter:
        parser.error("--train-value-head-and-value-adapter requires --value-adapter")

    # Propagate opening debug to workers via env var
    if args.opening_debug:
        os.environ["TWIXT_OPENING_DEBUG"] = "1"
    else:
        os.environ.pop("TWIXT_OPENING_DEBUG", None)

    # Validate and propagate mirror prob
    if not (0.0 <= args.mirror_prob <= 1.0):
        parser.error("--mirror-prob must be in [0, 1]")
    os.environ["TWIXT_MIRROR_PROB"] = str(args.mirror_prob)

    # Validate MCTS exploration tuning
    if args.dirichlet_alpha is not None and args.dirichlet_alpha <= 0:
        parser.error("--dirichlet-alpha must be > 0")
    if args.dirichlet_eps is not None and not (0 <= args.dirichlet_eps <= 1):
        parser.error("--dirichlet-eps must be in [0, 1]")
    if args.temp_high is not None and args.temp_high <= 0:
        parser.error("--temp-high must be > 0")
    if args.temp_low is not None and args.temp_low <= 0:
        parser.error("--temp-low must be > 0")
    if args.temp_threshold_ply is not None and args.temp_threshold_ply < 0:
        parser.error("--temp-threshold-ply must be >= 0")
    if (args.temp_high is not None and args.temp_low is not None
            and args.temp_low > args.temp_high):
        parser.error("--temp-low should be <= --temp-high")

    # Validate opening noise boost
    if args.opening_noise_ply is not None and args.opening_noise_ply < 0:
        parser.error("--opening-noise-ply must be >= 0")
    if args.opening_dirichlet_alpha is not None and args.opening_dirichlet_alpha <= 0:
        parser.error("--opening-dirichlet-alpha must be > 0")
    if args.opening_dirichlet_eps is not None and not (0 <= args.opening_dirichlet_eps <= 1):
        parser.error("--opening-dirichlet-eps must be in [0, 1]")

    # Validate edge-band penalty
    if args.root_edge_band_penalty is not None and args.root_edge_band_penalty < 0:
        parser.error("--root-edge-band-penalty must be >= 0")
    if args.root_edge_band_penalty_ply is not None and args.root_edge_band_penalty_ply < 0:
        parser.error("--root-edge-band-penalty-ply must be >= 0")
    if args.root_edge_band_width is not None and args.root_edge_band_width < 1:
        parser.error("--root-edge-band-width must be >= 1")
    if args.root_edge_band_width is not None and args.root_edge_band_width >= 12:
        parser.error("--root-edge-band-width must be < 12 for a 24x24 board")

    # Validate near-corner penalty
    if args.root_near_corner_penalty is not None and args.root_near_corner_penalty < 0:
        parser.error("--root-near-corner-penalty must be >= 0")
    if args.root_near_corner_penalty_ply is not None and args.root_near_corner_penalty_ply < 0:
        parser.error("--root-near-corner-penalty-ply must be >= 0")
    if args.root_near_corner_radius is not None and args.root_near_corner_radius < 1:
        parser.error("--root-near-corner-radius must be >= 1")
    if args.root_near_corner_radius is not None and args.root_near_corner_radius >= 12:
        parser.error("--root-near-corner-radius must be < 12 for a 24x24 board")

    # Phase 2: early-only near-corner override validation.
    # The two args travel together: either both >0 (active) or at least one 0
    # (inactive). We warn rather than error on half-set values so users can
    # script experiments by toggling a single flag.
    if args.root_near_corner_penalty_early is not None and args.root_near_corner_penalty_early < 0:
        parser.error("--root-near-corner-penalty-early must be >= 0")
    if args.root_near_corner_penalty_early_plies is not None and args.root_near_corner_penalty_early_plies < 0:
        parser.error("--root-near-corner-penalty-early-plies must be >= 0")
    _early_pen_set = (args.root_near_corner_penalty_early or 0) > 0
    _early_plies_set = (args.root_near_corner_penalty_early_plies or 0) > 0
    if _early_pen_set != _early_plies_set:
        print("[WARN] early near-corner override: one of "
              "--root-near-corner-penalty-early / "
              "--root-near-corner-penalty-early-plies is set to 0 — the "
              "override will have no effect. Set both > 0 to activate.")
    # Sanity: if the early window extends past the baseline window, the early
    # penalty effectively sets policy for the whole window. That is legal but
    # usually unintended — flag it.
    if (
        _early_pen_set and _early_plies_set
        and args.root_near_corner_penalty_ply is not None
        and args.root_near_corner_penalty_early_plies > args.root_near_corner_penalty_ply
    ):
        print(f"[WARN] early near-corner override window "
              f"({args.root_near_corner_penalty_early_plies}) is larger than "
              f"the baseline window ({args.root_near_corner_penalty_ply}) — "
              f"the baseline value will never apply.")

    # Validate resign parameters
    if args.resign_min_ply < 0:
        parser.error("--resign-min-ply must be >= 0")
    if args.resign_threshold > 0:
        parser.error("--resign-threshold must be <= 0 (negative means losing)")
    if args.resign_threshold < -1.0:
        parser.error("--resign-threshold must be >= -1.0 (value is tanh in [-1,1])")
    if args.resign_window < 1:
        parser.error("--resign-window must be >= 1")
    if args.resign_k < 1:
        parser.error("--resign-k must be >= 1")
    if args.resign_k > args.resign_window:
        parser.error("--resign-k must be <= --resign-window")
    if args.resign_min_visits < 1:
        parser.error("--resign-min-visits must be >= 1")
    if not (0.0 <= args.resign_min_top1_share <= 1.0):
        parser.error("--resign-min-top1-share must be in [0, 1]")

    # Validate adjudication parameters
    if args.adjudicate_min_ply < 0:
        parser.error("--adjudicate-min-ply must be >= 0")
    if not (0 <= args.adjudicate_threshold <= 1):
        parser.error("--adjudicate-threshold must be in [0, 1]")
    if args.adjudicate_min_visits < 1:
        parser.error("--adjudicate-min-visits must be >= 1")
    if not (0.0 <= args.adjudicate_min_top1_share <= 1.0):
        parser.error("--adjudicate-min-top1-share must be in [0, 1]")

    # Validate replay cap parameters (Phase 4)
    if args.max_positions_per_game < 0:
        parser.error("--max-positions-per-game must be >= 0 (0 = disabled)")
    if args.endgame_keep_positions < 0:
        parser.error("--endgame-keep-positions must be >= 0")
    if (args.max_positions_per_game > 0
            and args.endgame_keep_positions > args.max_positions_per_game):
        parser.error(
            "--endgame-keep-positions must be <= --max-positions-per-game "
            "when the cap is enabled"
        )

    # Mutual exclusion check
    if args.resume and args.load_weights:
        parser.error("Cannot use both --resume and --load-weights")

    # Parse curriculum sizes
    curriculum_sizes = tuple(int(x) for x in args.curriculum_sizes.split(","))

    # Import after arg parsing to avoid slow import for --help
    from scripts.GPU.alphazero.trainer import train

    print("=" * 60)
    print("ALPHAZERO TRAINING")
    print("=" * 60)
    print()
    print("Configuration:")
    print(f"  Iterations: {args.iterations}")
    print(f"  Games/iteration: {args.games_per_iter}")
    print(f"  Train steps/iteration: {args.train_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  MCTS simulations: {args.simulations}")
    print(f"  MCTS: eval_batch={args.mcts_eval_batch_size}, virtual_visits={args.mcts_pending_virtual_visits}, stall_flush={args.mcts_stall_flush_sims}")
    from scripts.GPU.alphazero.trainer import MAX_MOVES_TABLE
    max_moves_str = ", ".join(f"{s}:{MAX_MOVES_TABLE.get(s, '?')}" for s in curriculum_sizes)
    print(f"  Max moves/game: {max_moves_str} (per curriculum size)")
    print(f"  Network: hidden={args.hidden}, blocks={args.blocks}")
    print(f"  Learning rate: {args.lr}")
    print(f"  L2 weight: {args.l2}")
    print(f"  Buffer size: {args.buffer_size}")
    if args.max_positions_per_game > 0:
        print(f"  Replay cap: max {args.max_positions_per_game} positions/game "
              f"(endgame keep: {args.endgame_keep_positions})")
    else:
        print(f"  Replay cap: disabled (all positions from every game)")
    # Echo near-corner penalty windows (baseline + optional early override)
    if args.root_near_corner_penalty is not None or args.root_near_corner_penalty_early is not None:
        base_pen = args.root_near_corner_penalty or 0.0
        base_ply = args.root_near_corner_penalty_ply or 0
        early_pen = args.root_near_corner_penalty_early or 0.0
        early_plies = args.root_near_corner_penalty_early_plies or 0
        parts = [f"  Near-corner penalty:"]
        if base_pen > 0 and base_ply > 0:
            parts.append(f"baseline λ={base_pen} for ply<{base_ply}")
        if early_pen > 0 and early_plies > 0:
            parts.append(f"early λ={early_pen} for ply<{early_plies}")
        if len(parts) == 1:
            parts.append("inactive (half-set — see warning above)")
        print(" ".join(parts))
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Curriculum sizes: {curriculum_sizes}")
    print(f"  Workers: {args.n_workers}")
    print(f"  Save games: {not args.no_save_games}")
    if args.resume:
        print(f"  Resuming from: {args.resume}")
    if args.load_weights:
        print(f"  Loading weights from: {args.load_weights}")
    if args.seed is not None:
        print(f"  Random seed: {args.seed}")
    print()

    # Run training
    train_kwargs = dict(
        n_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        train_steps_per_iteration=args.train_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        checkpoint_dir=args.checkpoint_dir,
        mcts_simulations=args.simulations,
        learning_rate=args.lr,
        value_lr_scale=args.value_lr_scale,
        value_grad_max_norm=args.value_grad_max_norm,
        l2_weight=args.l2,
        progress_weighted=args.progress_weighted,
        progress_weight_floor=args.progress_weight_floor,
        hidden=args.hidden,
        n_blocks=args.blocks,
        max_moves=args.max_moves,
        resume_from=args.resume,
        load_weights_from=args.load_weights,
        seed=args.seed,
        # MCTS batching
        mcts_eval_batch_size=args.mcts_eval_batch_size,
        mcts_pending_virtual_visits=args.mcts_pending_virtual_visits,
        mcts_stall_flush_sims=args.mcts_stall_flush_sims,
        # Curriculum learning
        curriculum_sizes=curriculum_sizes,
        curriculum_window=args.curriculum_window,
        curriculum_draw_threshold=args.curriculum_draw_threshold,
        curriculum_min_wins=args.curriculum_min_wins,
        # Multi-process self-play
        n_workers=args.n_workers,
        # Game replay saving (default: enabled)
        save_games=not args.no_save_games,
        games_dir_override=args.games_dir,
        # MCTS exploration tuning (None = use MCTSConfig defaults)
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        temp_high=args.temp_high,
        temp_low=args.temp_low,
        temp_threshold_ply=args.temp_threshold_ply,
        # Opening exploration boost
        opening_noise_ply=args.opening_noise_ply,
        opening_dirichlet_alpha=args.opening_dirichlet_alpha,
        opening_dirichlet_eps=args.opening_dirichlet_eps,
        # Edge-band prior penalty
        root_edge_band_penalty=args.root_edge_band_penalty,
        root_edge_band_penalty_ply=args.root_edge_band_penalty_ply,
        root_edge_band_width=args.root_edge_band_width,
        # Near-corner prior penalty
        root_near_corner_penalty=args.root_near_corner_penalty,
        root_near_corner_penalty_ply=args.root_near_corner_penalty_ply,
        root_near_corner_radius=args.root_near_corner_radius,
        # Phase 2: early-only near-corner override
        root_near_corner_penalty_early=args.root_near_corner_penalty_early,
        root_near_corner_penalty_early_plies=args.root_near_corner_penalty_early_plies,
        # Resign parameters
        resign_enabled=args.resign_enabled,
        resign_min_ply=args.resign_min_ply,
        resign_threshold=args.resign_threshold,
        resign_window=args.resign_window,
        resign_k=args.resign_k,
        resign_min_visits=args.resign_min_visits,
        resign_min_top1_share=args.resign_min_top1_share,
        # Adjudication parameters
        adjudicate_enabled=args.adjudicate_enabled,
        adjudicate_min_ply=args.adjudicate_min_ply,
        adjudicate_threshold=args.adjudicate_threshold,
        adjudicate_min_visits=args.adjudicate_min_visits,
        adjudicate_min_top1_share=args.adjudicate_min_top1_share,
        adjudicate_debug=args.adjudicate_debug,
        # Phase 4: per-game replay contribution cap
        max_positions_per_game=(args.max_positions_per_game if args.max_positions_per_game > 0 else None),
        endgame_keep_positions=args.endgame_keep_positions,
        # Phase 2: inline forced-probe eval
        probes_path=args.probes_path,
        probes_inline_disable=args.probes_inline_disable,
    )
    train_kwargs.update(dict(
        conversion_policy_loss_enabled=args.conversion_policy_loss_enabled,
        conversion_policy_loss_weight=args.conversion_policy_loss_weight,
        conversion_completion_weight=args.conversion_completion_weight,
        conversion_reducer_weight=args.conversion_reducer_weight,
        conversion_max_total_goal_distance=args.conversion_max_total_goal_distance,
        conversion_sample_boost=args.conversion_sample_boost,
        conversion_max_batch_fraction=args.conversion_max_batch_fraction,
        recovery_bucket_enabled=args.recovery_bucket_enabled,
        recovery_dominant_unavailable_threshold=args.recovery_dominant_unavailable_threshold,
        recovery_delay_threshold=args.recovery_delay_threshold,
        # Spec 3 Fix 1: td=1 root visit forcing
        closeout_td1_visit_forcing_enabled=args.closeout_td1_visit_forcing_enabled,
        closeout_td1_min_visits=args.closeout_td1_min_visits,
        closeout_td1_max_forced_moves=args.closeout_td1_max_forced_moves,
        closeout_td1_require_high_value=args.closeout_td1_require_high_value,
        closeout_td1_high_value_threshold=args.closeout_td1_high_value_threshold,
        # Spec 3 Fix 2: narrow closeout selection tie-break
        closeout_selection_tiebreak_enabled=args.closeout_selection_tiebreak_enabled,
        closeout_selection_tiebreak_max_distance=args.closeout_selection_tiebreak_max_distance,
        closeout_selection_tiebreak_topk=args.closeout_selection_tiebreak_topk,
        closeout_selection_tiebreak_min_value=args.closeout_selection_tiebreak_min_value,
        closeout_selection_tiebreak_min_share=args.closeout_selection_tiebreak_min_share,
        # Post-opening sharp-drop calibration
        post_opening_calibration_enabled=args.post_opening_calibration_enabled,
        post_opening_calibration_manifest=args.post_opening_calibration_manifest,
        post_opening_calibration_target=args.post_opening_calibration_target,
        post_opening_calibration_weight=args.post_opening_calibration_weight,
        post_opening_calibration_batch_fraction=args.post_opening_calibration_batch_fraction,
        post_opening_calibration_tag_schedule=parse_calibration_tag_schedule(
            args.post_opening_calibration_tag_schedule),
        post_opening_calibration_teacher_value_weight=args.post_opening_calibration_teacher_value_weight,
        post_opening_calibration_teacher_policy_kl_weight=args.post_opening_calibration_teacher_policy_kl_weight,
        freeze_batchnorm_stats=args.freeze_batchnorm_stats,
        train_value_head_only=args.train_value_head_only,
        train_value_head_and_final_block=args.train_value_head_and_final_block,
        post_opening_guardrail_margin=args.guardrail_margin,
        post_opening_calibration_gradient_projection=args.post_opening_calibration_gradient_projection,
        post_opening_calibration_projection_strength=args.post_opening_calibration_projection_strength,
        value_adapter=args.value_adapter,
        value_adapter_bottleneck_width=args.value_adapter_bottleneck_width,
        train_value_head_and_value_adapter=args.train_value_head_and_value_adapter,
    ))
    train_kwargs.update(
        recovery_retargeting_enabled=not args.recovery_retargeting_disabled,
        recovery_retargeting_collapse_value_threshold=args.recovery_retargeting_collapse_value_threshold,
        recovery_retargeting_severe_value_threshold=args.recovery_retargeting_severe_value_threshold,
        recovery_retargeting_diffuse_root_top1_threshold=args.recovery_retargeting_diffuse_root_top1_threshold,
        recovery_retargeting_very_diffuse_root_top1_threshold=args.recovery_retargeting_very_diffuse_root_top1_threshold,
        recovery_retargeting_delta_threshold=args.recovery_retargeting_delta_threshold,
        recovery_retargeting_delta_max_current_score=args.recovery_retargeting_delta_max_current_score,
        recovery_retargeting_alternate_component_min_size=args.recovery_retargeting_alternate_component_min_size,
        recovery_retargeting_classify_defense=args.recovery_retargeting_classify_defense,
        recovery_retargeting_max_sampled_moves_per_side=args.recovery_retargeting_max_sampled_moves_per_side,
        recovery_retargeting_sample_all_moves=args.recovery_retargeting_sample_all_moves,
    )
    # Conditional override: None means "use default from train() (0.5)"
    if args.value_weight is not None:
        train_kwargs["value_weight"] = args.value_weight
    network = train(**train_kwargs)

    print()
    print("Training finished!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
