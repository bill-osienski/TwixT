"""Value calibration by position type — Phase 1 of the retrain design spec.

Bucket-wise value-head sanity stats: sign_agree, MSE, calibration-bin
reliability diagram, per bucket. Requires loading a checkpoint (not free);
gated behind --calibrate in the analyzer.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from .game.twixt_state import TwixtState
from .connectivity_diagnostics import compute_position_connectivity


def classify_position(state: TwixtState, ply: int, game_n_moves: int,
                     min_size: int = 8) -> str:
    """Assign a bucket label based on structural content + game phase."""
    stats = compute_position_connectivity(state)

    # Check "winning_structure" buckets — either color
    for color, prefix in (("red", "red"), ("black", "black")):
        largest = stats[f"{prefix}_largest_component_size"]
        n_touching = stats[f"{prefix}_n_goal_touching_components"]
        has_any_touch = stats[f"{prefix}_has_{'top' if color == 'red' else 'left'}_component"] or \
                         stats[f"{prefix}_has_{'bottom' if color == 'red' else 'right'}_component"]
        if has_any_touch and (largest >= min_size or n_touching >= 2):
            return f"{color}_winning_structure"

    # No winning structure: classify by game phase
    progress = ply / max(game_n_moves - 1, 1)
    if progress < 0.2:
        # Special case: empty / pre-game state → balanced_no_winning_structure
        if ply == 0:
            return "balanced_no_winning_structure"
        return "early_game"
    elif progress < 0.7:
        return "mid_game"
    else:
        return "late_game"


def compute_calibration_bins(preds: List[float], outcomes: List[float],
                             n_bins: int = 5) -> List[dict]:
    """Reliability-diagram bins: split preds into n_bins by value, compute
    mean pred and mean outcome per bin."""
    if not preds:
        return []
    # Bins over predicted value range [-1, 1]
    edges = np.linspace(-1, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = [(p, o) for (p, o) in zip(preds, outcomes) if lo <= p < hi]
        if i == n_bins - 1:  # last bin includes upper edge
            in_bin = [(p, o) for (p, o) in zip(preds, outcomes) if lo <= p <= hi]
        n = len(in_bin)
        if n == 0:
            bins.append({"lo": round(float(lo), 3), "hi": round(float(hi), 3),
                        "n": 0, "mean_pred": None, "mean_outcome": None})
        else:
            ps, os = zip(*in_bin)
            bins.append({
                "lo": round(float(lo), 3), "hi": round(float(hi), 3),
                "n": n,
                "mean_pred": round(sum(ps) / n, 4),
                "mean_outcome": round(sum(os) / n, 4),
            })
    return bins


def aggregate_calibration(samples: List[dict], n_bins: int = 5) -> dict:
    """samples is a list of {bucket, nn_value, outcome} dicts. Aggregates
    per bucket and globally."""
    from collections import defaultdict
    by_bucket: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        by_bucket[s["bucket"]].append(s)

    out = {"buckets": {}, "overall": {}}

    def _summary(rows):
        if not rows:
            return {"n": 0}
        preds = [r["nn_value"] for r in rows]
        outs = [r["outcome"] for r in rows]
        sign_agree_count = sum(1 for (p, o) in zip(preds, outs)
                               if (p > 0 and o > 0) or (p < 0 and o < 0) or
                                  (abs(p) < 0.1 and abs(o) < 0.1))
        return {
            "n": len(rows),
            "sign_agree": round(sign_agree_count / len(rows), 3),
            "mse": round(sum((p - o) ** 2 for (p, o) in zip(preds, outs)) / len(rows), 4),
            "pred_mean": round(sum(preds) / len(rows), 4),
            "outcome_mean": round(sum(outs) / len(rows), 4),
            "calibration_bins": compute_calibration_bins(preds, outs, n_bins),
        }

    for bucket, rows in by_bucket.items():
        out["buckets"][bucket] = _summary(rows)
    out["overall"] = _summary(samples)
    return out


def score_samples_against_checkpoint(
    replays: List[dict],
    network,
    samples_per_bucket: int = 200,
    max_total: int = 2000,
    min_size: int = 8,
) -> dict:
    """Phase-stratified calibration scoring.

    See spec §4.3 for full semantics. Pre-pass classifies every position in
    the replay pool; sample pass fills per-bucket caps in alphabetical
    order, halting when max_total binds; score pass runs NN forward for each
    sampled position and feeds the results to aggregate_calibration.
    """
    import random
    import numpy as np
    from .local_evaluator import LocalGPUEvaluator
    from .game.twixt_state import TwixtState

    # ---- Pre-pass: enumerate & classify every position ----
    # Positions are (game_idx, ply) pairs. classify_position needs TwixtState
    # + ply + game_n_moves.
    by_bucket_positions: dict[str, list[tuple[int, int]]] = {}
    natural_distribution: dict[str, int] = {}

    for g_idx, game in enumerate(replays):
        moves = game.get("moves") or []
        n_moves = len(moves)
        # Reconstruct state ply-by-ply so we classify each intermediate state.
        state = TwixtState(active_size=game.get("meta", {}).get("board_size", 24))
        for ply in range(n_moves):
            bucket = classify_position(state, ply, n_moves, min_size=min_size)
            natural_distribution[bucket] = natural_distribution.get(bucket, 0) + 1
            by_bucket_positions.setdefault(bucket, []).append((g_idx, ply))
            # Advance state for next ply.
            mv = moves[ply]["move"]
            state = state.apply_move((int(mv[0]), int(mv[1])))

    # ---- Sample pass: per-bucket caps, stable alphabetical order, halt on max_total ----
    rng = random.Random(42)  # deterministic sampling for reproducibility
    sampled_distribution: dict[str, int] = {b: 0 for b in natural_distribution}
    sampled_positions: list[tuple[str, int, int]] = []  # (bucket, game_idx, ply)
    cumulative = 0

    for bucket in sorted(natural_distribution.keys()):
        if cumulative >= max_total:
            break  # budget exhausted — remaining buckets get sampled=0
        bucket_pool = by_bucket_positions[bucket]
        cap = min(samples_per_bucket, len(bucket_pool), max_total - cumulative)
        if cap <= 0:
            continue
        chosen = rng.sample(bucket_pool, cap)
        for g_idx, ply in chosen:
            sampled_positions.append((bucket, g_idx, ply))
        sampled_distribution[bucket] = cap
        cumulative += cap

    # ---- Score pass: forward-pass each sampled position ----
    evaluator = LocalGPUEvaluator(network)
    samples: list[dict] = []

    for bucket, g_idx, ply in sampled_positions:
        game = replays[g_idx]
        moves = game["moves"]
        state = TwixtState(active_size=game.get("meta", {}).get("board_size", 24))
        for i in range(ply):
            mv = moves[i]["move"]
            state = state.apply_move((int(mv[0]), int(mv[1])))
        tensor = evaluator.build_input_tensor(state)
        tensor = np.transpose(tensor, (1, 2, 0))
        boards_np = np.expand_dims(tensor.astype(np.float32), axis=0)
        legal = state.legal_moves()
        move_rows_np = np.array([[m[0] for m in legal]], dtype=np.int32)
        move_cols_np = np.array([[m[1] for m in legal]], dtype=np.int32)
        move_mask_np = np.ones((1, len(legal)), dtype=np.float32)
        _, values_np = evaluator.infer(
            boards_np, move_rows_np, move_cols_np, move_mask_np, state.active_size
        )
        nn_value = float(values_np[0])
        # Red-perspective convention.
        if state.to_move == "black":
            nn_value = -nn_value
        # Outcome in red-perspective: +1 red wins, -1 black wins, 0 draw.
        winner = game.get("winner")
        if winner == "red":
            outcome = 1.0
        elif winner == "black":
            outcome = -1.0
        else:
            outcome = 0.0
        samples.append({"bucket": bucket, "nn_value": nn_value, "outcome": outcome})

    aggregate = aggregate_calibration(samples, n_bins=5)

    return {
        "samples_per_bucket_target": samples_per_bucket,
        "max_total": max_total,
        "natural_distribution": natural_distribution,
        "sampled_distribution": sampled_distribution,
        "stratified": True,
        "overall_note": "stratified aggregate, not population-weighted",
        "aggregate": aggregate,
    }
