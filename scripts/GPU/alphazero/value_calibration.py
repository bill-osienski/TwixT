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
