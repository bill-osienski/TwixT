from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..selfplay.results import GameSummary


@dataclass
class DepthResult:
    """Result for a single depth validation."""
    depth: int
    bias: float
    games: int
    passed: bool


@dataclass
class ValidationResult:
    """Result of validating across multiple depths."""
    per_depth: List[DepthResult]
    all_passed: bool  # True only if EVERY depth passed independently
    worst_bias: float  # Max absolute bias across depths
    combined_bias: float  # Weighted average (for backwards compat)


def validate_depths(
    summaries: List[Tuple[int, GameSummary]],
    pass_threshold: float = 0.02,
) -> ValidationResult:
    """Validate each depth INDEPENDENTLY.

    Unlike combine_depths which averages (allowing d2=+0.05, d3=-0.05 to cancel),
    this requires EACH depth to pass on its own.

    Args:
        summaries: List of (depth, GameSummary) tuples
        pass_threshold: Max absolute bias to pass (default 0.02 = 2%)

    Returns:
        ValidationResult with per-depth results and overall pass/fail
    """
    per_depth: List[DepthResult] = []

    for depth, summ in summaries:
        passed = abs(summ.bias) <= pass_threshold
        per_depth.append(DepthResult(
            depth=depth,
            bias=summ.bias,
            games=summ.games,
            passed=passed,
        ))

    all_passed = all(r.passed for r in per_depth)
    worst_bias = max(abs(r.bias) for r in per_depth) if per_depth else 0.0

    # Combined bias for backwards compat / logging
    total_games = sum(r.games for r in per_depth)
    if total_games > 0:
        combined_bias = sum(r.bias * r.games for r in per_depth) / total_games
    else:
        combined_bias = 0.0

    return ValidationResult(
        per_depth=per_depth,
        all_passed=all_passed,
        worst_bias=worst_bias,
        combined_bias=combined_bias,
    )


def combine_depths(summaries: List[Tuple[int, GameSummary]], depth_weights: Dict[int, float]) -> Tuple[float, float]:
    """Return (score, combined_bias).

    DEPRECATED: Use validate_depths() for proper per-depth validation.
    Kept for backwards compatibility with ranking.

    Score is now the WORST absolute bias (not weighted average).
    This ensures a config can't hide bad d2 with good d3.
    """
    if not summaries:
        return 999.0, 0.0

    # Score = worst bias across depths (stricter than average)
    worst_bias = max(abs(summ.bias) for _, summ in summaries)

    # Combined bias = weighted average (for logging)
    wsum = 0.0
    bias = 0.0
    for depth, summ in summaries:
        w = float(depth_weights.get(depth, 1.0))
        bias += w * summ.bias
        wsum += w

    combined_bias = (bias / wsum) if wsum > 0 else 0.0

    return worst_bias, combined_bias
