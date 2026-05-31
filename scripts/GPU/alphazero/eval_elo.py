"""Pure statistics for checkpoint-tournament results.

No game-engine or MLX dependencies — unit-testable in isolation. Scoring
counts a decisive win as 1 and a draw (state-cap or board-full) as 0.5.
"""
from __future__ import annotations

import math


def score_rate(wins: float, draws_plus_caps: float, total: int) -> float:
    """Score rate with draws/caps counting half. total must be > 0."""
    if total <= 0:
        raise ValueError("total must be > 0")
    return (wins + 0.5 * draws_plus_caps) / total


def _clamp_p(p: float, n: int) -> float:
    """Clamp a score rate away from {0, 1} so Elo stays finite.

    Bound is 1/(2N): a clean sweep maps to a large-but-finite Elo.
    """
    lo = 1.0 / (2 * n)
    hi = 1.0 - lo
    return min(max(p, lo), hi)


def elo_diff(p: float, n: int) -> float:
    """Elo difference implied by score rate p over n games (clamped)."""
    p = _clamp_p(p, n)
    return 400.0 * math.log10(p / (1.0 - p))


def score_ci_trinomial(w: int, d: int, l: int, z: float = 1.96) -> tuple[float, float]:
    """Draw-aware 95% CI on the score rate.

    Outcomes are {0, 0.5, 1}, so a Bernoulli/Wilson interval is the wrong
    model. Uses the trinomial score variance:
        var = [w(1-m)^2 + d(0.5-m)^2 + l(0-m)^2] / N,  SE = sqrt(var/N).
    w = wins, d = draws+caps, l = losses.
    """
    n = w + d + l
    if n <= 0:
        raise ValueError("no games")
    m = (w + 0.5 * d) / n
    var = (w * (1 - m) ** 2 + d * (0.5 - m) ** 2 + l * (0.0 - m) ** 2) / n
    se = math.sqrt(var / n)
    lo = max(0.0, m - z * se)
    hi = min(1.0, m + z * se)
    return lo, hi


def elo_ci(w: int, d: int, l: int, z: float = 1.96) -> tuple[float, float]:
    """95% Elo CI: trinomial score-rate endpoints mapped through elo_diff."""
    n = w + d + l
    lo, hi = score_ci_trinomial(w, d, l, z)
    return elo_diff(lo, n), elo_diff(hi, n)


def verdict(rate: float) -> str:
    """Strength verdict from score rate (spec thresholds)."""
    if rate >= 0.55:
        return "stronger"
    if rate >= 0.52:
        return "weak_signal"
    if rate >= 0.48:
        return "tied"
    return "worse"
