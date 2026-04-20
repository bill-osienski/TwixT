#!/usr/bin/env python3
"""Training script for GPU move scoring model.

Uses policy distillation with KL divergence to train the model
to match heuristic move ordering.

Usage:
    python3 scripts/GPU/ai/train_move_model.py --epochs 10 --positions 1000

Requires MLX for GPU training.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..utils.maybe_mlx import try_import_mlx
from .training_data import (
    TrainingPosition,
    TrainingDataGenerator,
    generate_training_batch,
)

# Check MLX availability
_mlx_env = try_import_mlx()

if _mlx_env.available:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from .move_model import MoveRanker, create_model


def log_softmax(x: "mx.array") -> "mx.array":
    """Numerically stable log-softmax: log(softmax(x)) = x - logsumexp(x)."""
    return x - mx.logsumexp(x)


def policy_distillation_loss(
    model_logits: "mx.array",
    heuristic_scores: "mx.array",
    temperature: float = 1.0,
) -> "mx.array":
    """KL divergence between model policy and heuristic policy.

    Numerically stable implementation:
    - Shifts scores before softmax to prevent overflow
    - Uses log_softmax for numerical stability

    Args:
        model_logits: (N,) raw logits from model
        heuristic_scores: (N,) scores from heuristic engine
        temperature: sharpness control (higher = softer distribution)

    Returns:
        KL divergence loss (scalar)
    """
    # Numerical stability: shift scores before softmax
    scores_shifted = heuristic_scores - mx.max(heuristic_scores)

    # Target distribution from heuristic scores
    target_logits = scores_shifted / temperature
    target_probs = mx.softmax(target_logits)

    # Model's predicted log-probabilities
    pred_log_probs = log_softmax(model_logits)

    # KL divergence: sum(p * (log(p) - log(q)))
    # Use log(p) directly for stability
    target_log_probs = log_softmax(target_logits)
    kl = mx.sum(target_probs * (target_log_probs - pred_log_probs))

    return kl


def top_k_policy_loss(
    model_logits: "mx.array",
    heuristic_scores: "mx.array",
    k: int = 50,
    temperature: float = 1.0,
) -> "mx.array":
    """Top-K policy distillation (renormalized within top-K).

    Computes KL on the top-K subset only, with softmax renormalized
    within top-K. This is NOT "KL vs full policy" - it's "match the
    conditional distribution over top-K moves."

    Args:
        model_logits: (N,) raw logits from model
        heuristic_scores: (N,) scores from heuristic engine
        k: Number of top moves to consider
        temperature: sharpness control

    Returns:
        KL divergence loss on top-K (scalar)
    """
    # Sort by heuristic score to get top-K indices
    order = mx.argsort(-heuristic_scores)
    top_k_idx = order[:k]

    # Extract top-K logits and scores
    top_k_logits = model_logits[top_k_idx]
    top_k_scores = heuristic_scores[top_k_idx]

    # Compute KL on top-K (renormalized within subset)
    return policy_distillation_loss(top_k_logits, top_k_scores, temperature)


class Trainer:
    """Training manager for MoveRanker model."""

    def __init__(
        self,
        model: "MoveRanker",
        learning_rate: float = 1e-3,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        """Initialize trainer.

        Args:
            model: MoveRanker model to train
            learning_rate: Optimizer learning rate
            temperature: KL divergence temperature
            top_k: If set, use top-K distillation (None = full KL)
        """
        if not _mlx_env.available:
            raise RuntimeError("MLX required for training")

        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.optimizer = optim.Adam(learning_rate=learning_rate)

        # Metrics tracking
        self.train_losses: List[float] = []
        self.epoch_times: List[float] = []

    def compute_loss(
        self,
        board_tensor: "mx.array",
        moves: List[tuple],
        heuristic_scores: "mx.array",
    ) -> "mx.array":
        """Compute loss for a single position.

        Args:
            board_tensor: (H, W, C) board tensor
            moves: List of (row, col) moves
            heuristic_scores: (N,) heuristic scores for moves

        Returns:
            Loss scalar
        """
        # Forward pass
        model_logits = self.model.score_all_moves(board_tensor, moves)

        # Compute loss
        if self.top_k is not None:
            loss = top_k_policy_loss(
                model_logits, heuristic_scores,
                k=self.top_k, temperature=self.temperature
            )
        else:
            loss = policy_distillation_loss(
                model_logits, heuristic_scores,
                temperature=self.temperature
            )

        return loss

    def train_step(
        self,
        batch: List[TrainingPosition],
    ) -> float:
        """Execute one training step on a batch.

        Note: Currently loops over positions (Phase 1 approach).
        Phase 2 optimization: bucket by move count and vectorize.

        Args:
            batch: List of TrainingPosition examples

        Returns:
            Average loss over batch
        """
        def loss_fn(model):
            total_loss = mx.array(0.0)

            for example in batch:
                board_tensor = mx.array(example.board_tensor)
                heuristic_scores = mx.array(example.heuristic_scores)

                loss = self.compute_loss(
                    board_tensor, example.moves, heuristic_scores
                )
                total_loss = total_loss + loss

            return total_loss / len(batch)

        # Compute loss and gradients
        loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model)

        # Update parameters
        self.optimizer.update(self.model, grads)

        # Force evaluation
        mx.eval(self.model.parameters(), self.optimizer.state, loss)

        return loss.item()

    def train_epoch(
        self,
        data_generator: TrainingDataGenerator,
        n_batches: int,
    ) -> float:
        """Train for one epoch.

        Args:
            data_generator: Generator yielding training batches
            n_batches: Number of batches per epoch

        Returns:
            Average loss over epoch
        """
        epoch_loss = 0.0
        start_time = time.perf_counter()

        for i in range(n_batches):
            batch = next(data_generator)
            loss = self.train_step(batch)
            epoch_loss += loss

            if (i + 1) % 10 == 0:
                avg_loss = epoch_loss / (i + 1)
                print(f"    Batch {i+1}/{n_batches}, Loss: {avg_loss:.4f}")

        epoch_time = time.perf_counter() - start_time
        self.epoch_times.append(epoch_time)

        avg_loss = epoch_loss / n_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def save_checkpoint(self, path: str, epoch: int) -> None:
        """Save model checkpoint.

        Args:
            path: Directory to save to
            epoch: Current epoch number
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model parameters using built-in save_weights
        model_path = path / f"model_epoch_{epoch:03d}.safetensors"
        self.model.save_weights(str(model_path))

        # Save training state
        state = {
            "epoch": epoch,
            "train_losses": self.train_losses,
            "epoch_times": self.epoch_times,
            "temperature": self.temperature,
            "top_k": self.top_k,
        }
        with open(path / f"state_epoch_{epoch:03d}.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"  Saved checkpoint to {path}")


def train_model(
    n_epochs: int = 10,
    positions_per_epoch: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    checkpoint_dir: str = "checkpoints/move_ranker",
    seed: Optional[int] = None,
) -> "MoveRanker":
    """Full training loop.

    Args:
        n_epochs: Number of training epochs
        positions_per_epoch: Positions to generate per epoch
        batch_size: Batch size for training
        learning_rate: Optimizer learning rate
        temperature: KL divergence temperature
        top_k: Top-K for distillation (None = full)
        checkpoint_dir: Directory for checkpoints
        seed: Random seed

    Returns:
        Trained MoveRanker model
    """
    if not _mlx_env.available:
        raise RuntimeError("MLX required for training")

    print("=" * 60)
    print("MOVE RANKER TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Positions/epoch: {positions_per_epoch}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-K: {top_k or 'full'}")

    # Create model
    print("\nInitializing model...")
    model = create_model()

    # Create trainer
    trainer = Trainer(
        model,
        learning_rate=learning_rate,
        temperature=temperature,
        top_k=top_k,
    )

    # Create data generator
    data_gen = TrainingDataGenerator(
        batch_size=batch_size,
        augment=True,
        seed=seed,
    )

    n_batches = positions_per_epoch // batch_size

    print(f"\nTraining for {n_epochs} epochs, {n_batches} batches each...")

    for epoch in range(n_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print("-" * 60)

        avg_loss = trainer.train_epoch(data_gen, n_batches)

        print(f"\n  Epoch loss: {avg_loss:.4f}")
        print(f"  Time: {trainer.epoch_times[-1]:.1f}s")

        # Save checkpoint every epoch
        trainer.save_checkpoint(checkpoint_dir, epoch + 1)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFinal loss: {trainer.train_losses[-1]:.4f}")
    print(f"Total time: {sum(trainer.epoch_times):.1f}s")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train GPU move scoring model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--positions", type=int, default=1000, help="Positions per epoch")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=1.0, help="KL temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-K distillation")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/move_ranker")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if not _mlx_env.available:
        print("ERROR: MLX is required for training but not available.")
        print("Install MLX with: pip install mlx")
        return

    train_model(
        n_epochs=args.epochs,
        positions_per_epoch=args.positions,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        top_k=args.top_k,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
