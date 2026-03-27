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
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
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
    import os
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

    args = parser.parse_args()

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
    print(f"  Max moves/game: {args.max_moves}")
    print(f"  Network: hidden={args.hidden}, blocks={args.blocks}")
    print(f"  Learning rate: {args.lr}")
    print(f"  L2 weight: {args.l2}")
    print(f"  Buffer size: {args.buffer_size}")
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
    network = train(
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
    )

    print()
    print("Training finished!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
