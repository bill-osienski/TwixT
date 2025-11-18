#!/usr/bin/env python3
"""Train a simple logistic regression value model from self-play traces.

Upgrades:
- --standardize : Z-score features using train-set mean/std (saved in model)
- --l2          : L2 regularization on weights (not on bias)
- --gamma       : temporal down-weighting by distance-to-terminal
- Skip draws entirely (games with no winner)
"""

import argparse
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("numpy is required: run `pip install numpy`") from exc

# Core heuristic feature keys coming from your self-play traces
FEATURE_KEYS = [
    "friendlyConnections",
    "opponentConnections",
    "friendlyDistance",
    "opponentDistance",
    "goalDistance",
    "centerBias",
    "isolatedBonus",
    "chainProximity",
    "frontierProximity",
    "frontierCapture",
    "connectorProximity",
    "connectorCapture",
    "trailingPenalty",
    "threatReduction",
    "noThreatReduction",
    "spanGain",
    "blackSpanComplete",
    "redSpanComplete",
    "opponentSpanReduction",
    "noSpanReductionPenalty",
    "blackSpanUpgradePenalty",
    "redSpanUpgradePenalty",
    "topBias",
    "aboveMinRowBonus",
    "belowMinRowPenalty",
    "bottomBias",
    "belowMaxRowBonus",
    "aboveMaxRowPenalty",
]

# Extra context features you already feed
EXTRA_FEATURES = ["player_is_red", "turn_normalized", "player_pegs", "opponent_pegs"]


def flatten_features(
    move: Dict[str, Any], summary: Dict[str, Any]
) -> Tuple[List[float], int]:
    """
    Extract a flat feature vector for a single move and a winner label (0/1).
    Label = 1 if the *player who took this move* eventually won the game, else 0.
    Assumes caller has filtered out draws.
    """
    heuristics = move.get("heuristics") or move.get("features") or {}
    feature_context = move.get("featureContext") or {}
    player = move.get("player") or feature_context.get("player")
    winner = summary.get("winner")
    label = 1 if winner and player and winner == player else 0

    # Base heuristics
    features = [float(heuristics.get(key, 0.0)) for key in FEATURE_KEYS]

    # ----- Extra features -----
    # player_is_red
    player_is_red = 1.0 if player == "red" else 0.0

    # turn_normalized (cap at 600 as in your trace shape)
    turn_value = move.get("turn")
    if turn_value is None:
        turn_value = feature_context.get("turn", 0)
    turn_normalized = float(turn_value or 0) / 600.0

    # player_pegs / opponent_pegs
    player_pegs = feature_context.get("playerPegCount")
    opponent_pegs = feature_context.get("opponentPegCount")

    # If counts absent, derive from board snapshot (if present)
    if player_pegs is None or opponent_pegs is None:
        board = move.get("board") or []
        pp = 0
        op = 0
        for row in board:
            for cell in row:
                if cell == player:
                    pp += 1
                elif cell and cell != player:
                    op += 1
        player_pegs = pp
        opponent_pegs = op

    features.extend(
        [
            player_is_red,
            turn_normalized,
            float(player_pegs or 0),
            float(opponent_pegs or 0),
        ]
    )
    # --------------------------

    return features, label


def load_dataset(
    path: str, gamma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load self-play JSON and return (X, y, w) where:
      - X: features (N x D)
      - y: winner labels (N,)
      - w: per-sample weights (N,)  [temporal down-weighting by distance-to-terminal]
    Draws are skipped entirely.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    games = data.get("games", [])
    X: List[List[float]] = []
    y: List[int] = []
    w: List[float] = []

    skipped_draws = 0

    for game in games:
        summary = game.get("summary", {}) or {}
        moves = game.get("moves", []) or []
        if not moves:
            continue

        # ---- Skip draws entirely (Option A) ----
        # winner must be exactly 'red' or 'black'
        winner = summary.get("winner")
        if winner not in ("red", "black"):
            skipped_draws += 1
            continue
        # (Also tolerate explicit flags if present)
        if summary.get("draw") is True:
            skipped_draws += 1
            continue
        meta = game.get("meta") or {}
        if meta.get("draw") is True:
            skipped_draws += 1
            continue
        # ---------------------------------------

        total_moves = summary.get("totalMoves")
        if not isinstance(total_moves, int) or total_moves <= 0:
            total_moves = len(moves)

        for ply_idx, move in enumerate(moves, start=1):
            feats, label = flatten_features(move, summary)
            X.append(feats)
            y.append(label)

            # temporal weight: down-weight early plies if gamma < 1.0
            dist_to_terminal = total_moves - ply_idx
            ww = (gamma**dist_to_terminal) if (gamma < 1.0) else 1.0
            w.append(float(ww))

    if skipped_draws:
        print(f"[train_value] Skipped {skipped_draws} draw game(s).")

    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)
    w_arr = np.array(w, dtype=np.float64)

    # Normalize weights so average weight ~ 1 (keeps LR schedule stable)
    if w_arr.size:
        s = w_arr.sum()
        if s > 0:
            w_arr *= len(w_arr) / s

    return X_arr, y_arr, w_arr


def weighted_logloss(y_true: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """Weighted binary cross-entropy."""
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(
        (-(w * (y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))).sum() / w.sum()
    )


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    epochs: int = 2000,
    lr: float = 0.01,
    l2: float = 0.0,
) -> np.ndarray:
    """
    Train logistic regression with bias using gradient descent.
    - X: (N, D) features (no bias column)
    - y: (N,) labels
    - w: (N,) sample weights (avg~1)
    Returns theta of shape (D+1,) with theta[0] = bias.
    """
    n_samples, n_features = X.shape
    # Intercept column
    X_aug = np.hstack([np.ones((n_samples, 1), dtype=X.dtype), X])
    theta = np.zeros(n_features + 1, dtype=X.dtype)

    # Precompute for speed
    w = w.astype(X.dtype, copy=False)
    norm = w.sum() if w.size else float(n_samples)

    for epoch in range(epochs):
        z = X_aug @ theta
        p = 1.0 / (1.0 + np.exp(-z))

        # Weighted residuals
        err = (p - y) * w  # (N,)

        # L2 on weights only (not bias at theta[0])
        reg = np.zeros_like(theta)
        if l2 > 0.0:
            reg[1:] = l2 * theta[1:]

        grad = (X_aug.T @ err) / norm + reg
        theta -= lr * grad

        if (epoch % 200) == 0 or epoch == epochs - 1:
            loss = weighted_logloss(y, p, w)
            print(f"Epoch {epoch:4d} | weighted logloss {loss:.5f}")

    return theta


def evaluate(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Unweighted accuracy for interpretability."""
    if X.size == 0:
        return 0.0
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    p = 1.0 / (1.0 + np.exp(-(X_aug @ theta)))
    preds = (p >= 0.5).astype(float)
    return float((preds == y).mean())


def main():
    parser = argparse.ArgumentParser(
        description="Train a simple value model from self-play traces."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to self-play JSON trace (from scripts/selfPlay.js).",
    )
    parser.add_argument(
        "--output",
        default="value-model.json",
        help="Where to save the trained model weights.",
    )
    parser.add_argument(
        "--epochs", type=int, default=2000, help="Training epochs (default: 2000)."
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate (default: 0.01)."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for data split."
    )

    # Upgrades
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Z-score features using train-set mean/std (stored in model).",
    )
    parser.add_argument(
        "--l2", type=float, default=0.0, help="L2 regularization strength (e.g., 1e-4)."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Temporal down-weighting: weight = gamma^(distance-to-terminal). Use 0.995–0.999.",
    )

    args = parser.parse_args()

    if args.l2 < 0:
        raise ValueError("--l2 must be >= 0")
    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("--gamma must be in (0, 1]")

    # Load dataset with temporal weights (draws skipped inside)
    X, y, w = load_dataset(args.input, gamma=args.gamma)
    if X.size == 0:
        raise SystemExit(
            "No training samples found in the input file (maybe all games were draws?)."
        )

    # Reproducible split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(0.80 * len(idx))
    tr, te = idx[:cut], idx[cut:]

    X_train, y_train, w_train = X[tr], y[tr], w[tr]
    X_test, y_test, w_test = X[te], y[te], w[te]

    # Optional standardization (fit on TRAIN only, apply to both)
    mu = None
    sigma = None
    if args.standardize:
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma

    # Train
    theta = train_logistic_regression(
        X_train, y_train, w_train, epochs=args.epochs, lr=args.lr, l2=args.l2
    )

    # Metrics
    def predict_proba(Xmat: np.ndarray) -> np.ndarray:
        if Xmat.size == 0:
            return np.zeros((0,), dtype=float)
        X_aug = np.hstack([np.ones((Xmat.shape[0], 1)), Xmat])
        return 1.0 / (1.0 + np.exp(-(X_aug @ theta)))

    p_train = predict_proba(X_train)
    p_test = predict_proba(X_test)

    train_acc = evaluate(theta, X_train, y_train)
    test_acc = evaluate(theta, X_test, y_test)
    train_logloss = weighted_logloss(y_train, p_train, w_train)
    test_logloss = weighted_logloss(y_test, p_test, w_test)

    # Save model (keep 'weights' for compatibility; add preproc + params)
    model = {
        "type": "logistic_regression",
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "feature_keys": FEATURE_KEYS + EXTRA_FEATURES,
        "weights": theta.tolist(),  # bias-first
        "preproc": {
            "standardize": bool(args.standardize),
            "mean": (mu.tolist() if mu is not None else None),
            "std": (sigma.tolist() if sigma is not None else None),
        },
        "params": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "l2": args.l2,
            "gamma": args.gamma,
            "seed": args.seed,
        },
        "metrics": {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "train_logloss": float(train_logloss),
            "test_logloss": float(test_logloss),
        },
    }

    output_path = os.path.abspath(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    print(f"Saved value model to {output_path}")
    print(
        f"Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f} | "
        f"Train logloss: {train_logloss:.5f} | Test logloss: {test_logloss:.5f}"
    )


if __name__ == "__main__":
    main()
