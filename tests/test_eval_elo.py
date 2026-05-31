import math
import pytest

from scripts.GPU.alphazero.eval_elo import (
    score_rate, elo_diff, score_ci_trinomial, elo_ci, verdict,
)


def test_score_rate_counts_draws_half():
    # 6 wins, 2 draws, 2 losses out of 10 -> (6 + 1)/10 = 0.7
    assert score_rate(6, 2, 10) == pytest.approx(0.7)


def test_score_rate_rejects_zero_total():
    with pytest.raises(ValueError):
        score_rate(0, 0, 0)


def test_elo_diff_60pct_is_about_plus_70():
    assert elo_diff(0.6, 400) == pytest.approx(70.4, abs=1.0)


def test_elo_diff_is_antisymmetric():
    # A scoring p and B scoring 1-p must give opposite Elo.
    assert elo_diff(0.6, 400) == pytest.approx(-elo_diff(0.4, 400), abs=1e-9)


def test_elo_diff_clamps_clean_sweep_to_finite():
    # p == 1.0 must not be +inf; clamp at 1 - 1/(2N).
    val = elo_diff(1.0, 400)
    assert math.isfinite(val)
    assert val == pytest.approx(elo_diff(1.0 - 1.0 / 800, 400))


def test_score_ci_trinomial_brackets_mean():
    lo, hi = score_ci_trinomial(223, 8, 169)
    m = (223 + 0.5 * 8) / 400
    assert lo < m < hi
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_elo_ci_endpoints_ordered():
    lo, hi = elo_ci(223, 8, 169)
    assert lo < hi


def test_verdict_thresholds():
    assert verdict(0.60) == "stronger"
    assert verdict(0.55) == "stronger"
    assert verdict(0.53) == "weak_signal"
    assert verdict(0.50) == "tied"
    assert verdict(0.40) == "worse"
