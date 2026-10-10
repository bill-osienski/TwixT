"""The replacement seed-selection rule (`seed_selection.candidate`). Synthetic values are computed from the
band's lower bound, so no band literal is spelled; the real-state tests read the live registries and the
pinned method without changing either."""
import importlib.util
import pathlib

import pytest

from scripts.GPU.alphazero import seed_selection as R

ROOT = pathlib.Path(__file__).resolve().parents[1]
LO = R.BAND[0]


def _pinned():
    spec = importlib.util.spec_from_file_location(
        "_tactical_survey_for_rule", ROOT / "docs/superpowers/evidence/tactical-candidate-seed-survey/01_seed_survey.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    return S


@pytest.mark.parametrize("highest_off,n,base_off", [
    (31, 32, 100),          # last seed +31 -> needs >= +64 -> +100
    (67, 32, 100),          # needs >= +100 exactly -> +100 (the margin is met with equality)
    (68, 32, 200),          # needs >= +101 -> +200
    (99, 1, 200),           # needs >= +101 -> +200
    (31, 400, 500),         # a larger block keeps a larger margin
])
def test_BASE_is_the_least_aligned_value_leaving_N_seeds_above_the_highest(highest_off, n, base_off):
    highest, base, block = R.candidate([(LO + 5, LO + highest_off + 1)], n)
    assert (highest, base, block) == (LO + highest_off, LO + base_off, (LO + base_off, LO + base_off + n))
    assert base % R.ALIGN == 0 and base - (highest + 1) >= n                   # aligned; the floor below holds
    assert base - R.ALIGN - (highest + 1) < n                                  # and it is the LEAST such


def test_the_block_lies_ABOVE_EVERY_in_band_interval_whatever_their_order():
    ivs = [(LO + 9_000, LO + 9_032), (LO + 100, LO + 132), (LO + 40_000, LO + 40_018), (LO + 5_000, LO + 5_001)]
    highest, base, block = R.candidate(ivs, 32)
    assert highest == LO + 40_017 and all(hi <= block[0] - 32 for _lo, hi in ivs)


def test_intervals_OUTSIDE_the_band_are_ignored_and_the_input_is_not_changed():
    ivs = [(-99, -1), (LO - 500, LO - 400), (R.BAND[1] + 7, R.BAND[1] + 9), (LO + 10, LO + 42)]
    before = list(ivs)
    assert R.candidate(ivs, 32)[2] == (LO + 100, LO + 132)
    assert ivs == before


def test_it_REFUSES_once_the_block_would_leave_the_band_and_never_wraps():
    near_end = R.BAND[1] - 150
    assert R.candidate([(near_end - 10, near_end)], 32)[2][1] <= R.BAND[1]       # control: still fits
    with pytest.raises(R.Exhausted, match="would leave the band"):
        R.candidate([(near_end - 10, near_end + 60)], 32)
    with pytest.raises(R.Exhausted):
        R.candidate([(LO, R.BAND[1])], 1)


@pytest.mark.parametrize("bad", [0, -3, 2.0, True, None])
def test_a_bad_block_size_is_refused(bad):
    with pytest.raises(ValueError, match="positive int"):
        R.candidate([(LO, LO + 1)], bad)


def test_no_in_band_interval_is_refused_not_guessed():
    with pytest.raises(ValueError, match="no registered"):
        R.candidate([(-9, -1)], 32)


def test_every_seed_it_can_choose_is_one_the_PINNED_scanner_SEES():
    """The rule stays in the band precisely so the pinned scanner (2026-prefixed nine-digit literals)
    covers every seed it could pick; a seed outside the band would be invisible to it."""
    S = _pinned()
    assert R.BAND == S.BAND
    for seed in (R.BAND[0], R.BAND[0] + 12_345, R.BAND[1] - 1):
        assert S.scan(f"x = {seed}\n") == {seed: 1}
    assert S.scan(f"x = {R.BAND[1]}\n") == {}                                   # just outside: unseen


# ─────────────────────────────── the real state ───────────────────────────────

def _reserved(S):
    registries, named, _gen, _masks, _consumed = S.prior_terms()
    return [iv for v in registries.values() for iv in v] + [iv for v in named.values() for iv in v]


def test_REAL_STATE_the_pinned_rule_is_exhausted_and_this_one_is_not():
    S = _pinned()
    reserved = _reserved(S)
    with pytest.raises(AssertionError, match="left the 2026 band"):
        S.candidate(reserved)
    highest, base, block = R.candidate(reserved, S.N)
    assert block[0] - (highest + 1) >= S.N and base % R.ALIGN == 0
    assert all(hi <= block[0] for lo, hi in reserved if R.BAND[0] <= lo < R.BAND[1])  # above every block
    assert all(not (lo < block[1] and block[0] < hi) for lo, hi in reserved)     # overlaps nothing at all


def test_REAL_STATE_the_closing_order_the_registries_rely_on_still_holds():
    """Every closed block was appended LAST and HIGHER; the next block this rule chooses is higher still,
    so appending it would keep EXPOSED and RETIRED in ascending order of their in-band entries."""
    from scripts.GPU.alphazero import e4_screen_reference as REG
    S = _pinned()
    _h, _b, block = R.candidate(_reserved(S), S.N)
    for name in ("EXPOSED_SEED_INTERVALS", "RETIRED_SEED_INTERVALS", "ACCOUNTED_SEED_INTERVALS"):
        last = [iv for iv in getattr(REG, name) if R.BAND[0] <= iv[0] < R.BAND[1]][-1]
        assert last[1] <= block[0], name
