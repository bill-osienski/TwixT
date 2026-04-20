#!/usr/bin/env python3
"""Generate self-play games for AlphaZero training.

Usage:
    python -m scripts.GPU.alphazero.generate_games --n-games 10 --output games.json

    # With custom settings
    python -m scripts.GPU.alphazero.generate_games \\
        --n-games 100 \\
        --simulations 200 \\
        --seed 42 \\
        --output training_games.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Generate self-play games for AlphaZero training"
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=10,
        help="Number of games to generate (default: 10)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="MCTS simulations per move (default: 100, use 800 for quality)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum moves per game before draw (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="games.json",
        help="Output JSON file (default: games.json)",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Disable Dirichlet noise (for evaluation, not training)",
    )
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
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained weights (uses random init if not specified)",
    )

    args = parser.parse_args()

    # Import after arg parsing to avoid slow import for --help
    from scripts.GPU.alphazero.network import create_network
    from scripts.GPU.alphazero.mcts import MCTSConfig
    from scripts.GPU.alphazero.self_play import play_games
    from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator

    print("=" * 60)
    print("ALPHAZERO SELF-PLAY GAME GENERATION")
    print("=" * 60)
    print()

    # Create network
    print(f"Creating network (hidden={args.hidden}, blocks={args.blocks})...")
    network = create_network(hidden=args.hidden, n_blocks=args.blocks)

    if args.weights:
        print(f"Loading weights from {args.weights}...")
        network.load_weights(args.weights)
    else:
        print("Using random network weights (no trained model)")

    # Configure MCTS
    mcts_config = MCTSConfig(n_simulations=args.simulations)
    print(f"MCTS simulations per move: {args.simulations}")
    print(f"Maximum moves per game: {args.max_moves}")
    print(f"Dirichlet noise: {'disabled' if args.no_noise else 'enabled'}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print()

    # Generate games
    print(f"Generating {args.n_games} games...")
    start_time = time.time()

    def progress(i, game):
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        winner = game.winner or "draw"
        print(
            f"  Game {i+1}/{args.n_games}: {game.n_moves} moves, winner={winner} "
            f"({rate:.2f} games/sec)"
        )

    # Wrap network in LocalGPUEvaluator for MCTS
    evaluator = LocalGPUEvaluator(network)

    games = play_games(
        evaluator,
        n_games=args.n_games,
        mcts_config=mcts_config,
        seed=args.seed,
        max_moves=args.max_moves,
        add_noise=not args.no_noise,
        progress_callback=progress,
    )

    elapsed = time.time() - start_time
    print()
    print(f"Generated {len(games)} games in {elapsed:.1f}s")

    # Compute stats
    total_positions = sum(len(g.positions) for g in games)
    red_wins = sum(1 for g in games if g.winner == "red")
    black_wins = sum(1 for g in games if g.winner == "black")
    draws = sum(1 for g in games if g.winner is None)
    avg_moves = sum(g.n_moves for g in games) / len(games) if games else 0

    print()
    print("Statistics:")
    print(f"  Total positions: {total_positions}")
    print(f"  Average moves/game: {avg_moves:.1f}")
    print(f"  Red wins: {red_wins} ({100*red_wins/len(games):.1f}%)")
    print(f"  Black wins: {black_wins} ({100*black_wins/len(games):.1f}%)")
    print(f"  Draws: {draws} ({100*draws/len(games):.1f}%)")

    # Save to JSON
    print()
    print(f"Saving to {args.output}...")
    output_data = {
        "config": {
            "n_games": args.n_games,
            "simulations": args.simulations,
            "max_moves": args.max_moves,
            "seed": args.seed,
            "noise": not args.no_noise,
            "hidden": args.hidden,
            "blocks": args.blocks,
            "weights": args.weights,
        },
        "stats": {
            "total_positions": total_positions,
            "avg_moves": avg_moves,
            "red_wins": red_wins,
            "black_wins": black_wins,
            "draws": draws,
        },
        "games": [g.to_dict() for g in games],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(games)} games to {args.output}")
    print()
    print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
