"""Quality metrics for evaluating GPU move scoring model.

Metrics:
- Top-5 recall: % of heuristic's top-5 in model's top-5
- Top-1 agreement: % where model and heuristic pick same top move
- Blunder rate: % where model's top-1 falls outside heuristic's top-10
- Regret (mean/median): score difference between best and model's choice

Usage:
    python3 -m scripts.GPU.ai.quality_metrics --positions 100
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..game.state import GameState
from ..game.rules import generate_moves
from .heuristics import score_moves_batch, DEFAULT_KNOBS
from .training_data import generate_diverse_positions
from .tensor_repr import state_to_numpy
from ..utils.maybe_mlx import try_import_mlx


@dataclass
class QualityMetrics:
    """Quality metrics for model evaluation."""
    n_positions: int
    top_1_agreement: float
    top_5_recall: float
    top_10_recall: float
    blunder_rate: float  # Top-1 outside heuristic's top-10
    regret_mean: float
    regret_median: float
    regret_max: float


def compute_metrics(
    model_rankings: List[List[int]],
    heuristic_rankings: List[List[int]],
    heuristic_scores: List[List[float]],
) -> QualityMetrics:
    """Compute quality metrics comparing model to heuristic rankings.

    Args:
        model_rankings: For each position, list of move indices sorted by model score
        heuristic_rankings: For each position, list of move indices sorted by heuristic score
        heuristic_scores: For each position, heuristic scores for each move

    Returns:
        QualityMetrics dataclass
    """
    n_positions = len(model_rankings)

    top_1_matches = 0
    top_5_recalls = []
    top_10_recalls = []
    blunders = 0
    regrets = []

    for i in range(n_positions):
        model_rank = model_rankings[i]
        heur_rank = heuristic_rankings[i]
        scores = heuristic_scores[i]

        if len(model_rank) == 0 or len(heur_rank) == 0:
            continue

        # Top-1 agreement
        if model_rank[0] == heur_rank[0]:
            top_1_matches += 1

        # Top-5 recall
        model_top5 = set(model_rank[:5])
        heur_top5 = set(heur_rank[:5])
        recall_5 = len(model_top5 & heur_top5) / min(5, len(heur_rank))
        top_5_recalls.append(recall_5)

        # Top-10 recall
        model_top10 = set(model_rank[:10])
        heur_top10 = set(heur_rank[:10])
        recall_10 = len(model_top10 & heur_top10) / min(10, len(heur_rank))
        top_10_recalls.append(recall_10)

        # Blunder rate: model's top-1 outside heuristic's top-10
        if model_rank[0] not in heur_top10:
            blunders += 1

        # Regret: score(best) - score(model's choice)
        best_score = scores[heur_rank[0]]
        model_choice_score = scores[model_rank[0]]
        regret = best_score - model_choice_score
        regrets.append(regret)

    return QualityMetrics(
        n_positions=n_positions,
        top_1_agreement=top_1_matches / n_positions if n_positions > 0 else 0.0,
        top_5_recall=np.mean(top_5_recalls) if top_5_recalls else 0.0,
        top_10_recall=np.mean(top_10_recalls) if top_10_recalls else 0.0,
        blunder_rate=blunders / n_positions if n_positions > 0 else 0.0,
        regret_mean=np.mean(regrets) if regrets else 0.0,
        regret_median=np.median(regrets) if regrets else 0.0,
        regret_max=max(regrets) if regrets else 0.0,
    )


def evaluate_model(
    model,
    positions: List[GameState],
    knobs: Optional[Dict] = None,
    verbose: bool = True,
) -> QualityMetrics:
    """Evaluate model quality on a set of positions.

    Args:
        model: MoveRanker model to evaluate
        positions: List of game positions
        knobs: Heuristic knobs
        verbose: Print progress

    Returns:
        QualityMetrics
    """
    _mlx_env = try_import_mlx()
    if not _mlx_env.available:
        raise RuntimeError("MLX required for model evaluation")

    from .tensor_repr import state_to_tensor

    mx = _mlx_env.mx

    if knobs is None:
        knobs = DEFAULT_KNOBS

    model_rankings = []
    heuristic_rankings = []
    heuristic_scores_list = []

    for i, state in enumerate(positions):
        if verbose and (i + 1) % 20 == 0:
            print(f"  Evaluating position {i + 1}/{len(positions)}")

        moves = generate_moves(state)
        if len(moves) < 5:
            continue

        # Get heuristic scores
        scored = score_moves_batch(state, moves, knobs=knobs, return_children=False)

        # Create mapping from move to index
        move_to_idx = {m: i for i, m in enumerate(moves)}

        # Heuristic ranking (indices sorted by score, descending)
        heur_rank = [move_to_idx[m] for m, _ in scored]
        heuristic_rankings.append(heur_rank)

        # Heuristic scores in original order
        scores = [0.0] * len(moves)
        for (r, c), score in scored:
            scores[move_to_idx[(r, c)]] = score
        heuristic_scores_list.append(scores)

        # Get model scores
        board_tensor = state_to_tensor(state)
        model_logits = model.score_all_moves(board_tensor, moves)
        mx.eval(model_logits)

        # Model ranking
        model_order = mx.argsort(-model_logits).tolist()
        model_rankings.append(model_order)

    metrics = compute_metrics(
        model_rankings, heuristic_rankings, heuristic_scores_list
    )

    return metrics


def evaluate_random_baseline(
    positions: List[GameState],
    knobs: Optional[Dict] = None,
    seed: int = 42,
) -> QualityMetrics:
    """Evaluate random move selection as baseline.

    Args:
        positions: List of game positions
        knobs: Heuristic knobs
        seed: Random seed

    Returns:
        QualityMetrics for random selection
    """
    import random
    rng = random.Random(seed)

    if knobs is None:
        knobs = DEFAULT_KNOBS

    model_rankings = []
    heuristic_rankings = []
    heuristic_scores_list = []

    for state in positions:
        moves = generate_moves(state)
        if len(moves) < 5:
            continue

        # Get heuristic scores
        scored = score_moves_batch(state, moves, knobs=knobs, return_children=False)

        move_to_idx = {m: i for i, m in enumerate(moves)}
        heur_rank = [move_to_idx[m] for m, _ in scored]
        heuristic_rankings.append(heur_rank)

        scores = [0.0] * len(moves)
        for (r, c), score in scored:
            scores[move_to_idx[(r, c)]] = score
        heuristic_scores_list.append(scores)

        # Random ranking
        indices = list(range(len(moves)))
        rng.shuffle(indices)
        model_rankings.append(indices)

    return compute_metrics(model_rankings, heuristic_rankings, heuristic_scores_list)


def print_metrics(metrics: QualityMetrics, name: str = "Model") -> None:
    """Print quality metrics in a formatted way."""
    print(f"\n{name} Quality Metrics ({metrics.n_positions} positions):")
    print("-" * 50)
    print(f"  Top-1 Agreement:  {metrics.top_1_agreement:.1%}")
    print(f"  Top-5 Recall:     {metrics.top_5_recall:.1%}")
    print(f"  Top-10 Recall:    {metrics.top_10_recall:.1%}")
    print(f"  Blunder Rate:     {metrics.blunder_rate:.1%}")
    print(f"  Regret (mean):    {metrics.regret_mean:.0f}")
    print(f"  Regret (median):  {metrics.regret_median:.0f}")
    print(f"  Regret (max):     {metrics.regret_max:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPU move model quality")
    parser.add_argument("--positions", type=int, default=100, help="Number of positions")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("QUALITY METRICS EVALUATION")
    print("=" * 60)

    print(f"\nGenerating {args.positions} test positions...")
    positions = generate_diverse_positions(args.positions, seed=args.seed)
    print(f"Generated {len(positions)} positions")

    # Random baseline
    print("\nEvaluating random baseline...")
    random_metrics = evaluate_random_baseline(positions, seed=args.seed)
    print_metrics(random_metrics, "Random Baseline")

    # Model evaluation (if model provided)
    _mlx_env = try_import_mlx()
    if args.model and _mlx_env.available:
        from .move_model import load_model

        print(f"\nLoading model from {args.model}...")
        model = load_model(args.model)

        print("\nEvaluating model...")
        model_metrics = evaluate_model(model, positions)
        print_metrics(model_metrics, "GPU Model")

        # Comparison
        print("\n" + "=" * 60)
        print("COMPARISON (Model vs Random)")
        print("-" * 60)
        print(f"  Top-1 Agreement:  +{(model_metrics.top_1_agreement - random_metrics.top_1_agreement):.1%}")
        print(f"  Blunder Rate:     {(model_metrics.blunder_rate - random_metrics.blunder_rate):+.1%}")
        print(f"  Regret (mean):    {(model_metrics.regret_mean - random_metrics.regret_mean):+.0f}")
    elif not _mlx_env.available:
        print("\nMLX not available - skipping model evaluation")
    else:
        print("\nNo model specified - showing random baseline only")
        print("Use --model to evaluate a trained model")


if __name__ == "__main__":
    main()
