"""THE SEED-SELECTION RULE that replaces the exhausted one (2026-10-10).

The pinned rule (`candidate` in evidence/tactical-candidate-seed-survey/01_seed_survey.py, imported unchanged
by every later survey) takes BASE = the first multiple of 10,000 strictly above the highest registered or
named-spent seed in the 2026 band. After the external-engine stage-3 block that BASE is the band's upper
end, so it refuses. The band holds only ten 10,000-aligned slots.

THIS RULE keeps everything else of the pinned method: the same band, so the pinned scanner (which sees only
nine-digit literals beginning 2026) still covers every seed it can choose; the same reserved intervals; the
same check, which decides CLEAN or stop and is never adjusted. It changes only how BASE is placed:

    BASE = the least multiple of ALIGN (100) that leaves at least N unused seeds above the highest seed
           (an interval's last seed is hi - 1) of every registered or named-spent interval lying in the
           band; block = [BASE, BASE + N).

  * Monotone, like the pinned rule: a new block always lies above every earlier one. No gap below the
    highest block is ever filled, so the closing tests' "only later, higher closings may follow" still holds.
  * The N-seed margin is the check's own gap floor (the block's size) against the interval below it.
  * Refused, never wrapped or searched, once the block would leave the band. Room left above the stage-3
    block is about 8,900 seeds (an H4 fixture literal sits near the band's end; the check's floor stops
    there). Beyond it a new band needs a new scanner, i.e. a new method and its own review.

Nothing here reads or writes a registry, reserves, exposes or retires a seed, or holds a seed literal.
"""
from typing import Iterable, Tuple

BAND = (2026 * 10 ** 5, 2027 * 10 ** 5)          # the nine-digit 2026 band, as pinned
ALIGN = 100


class Exhausted(Exception):
    """The block would leave the band: no seed is chosen."""


def candidate(intervals: Iterable[Tuple[int, int]], n: int) -> Tuple[int, int, Tuple[int, int]]:
    """(highest seed, BASE, block) for an `n`-seed block above every in-band interval of `intervals`."""
    if type(n) is not int or n < 1:
        raise ValueError(f"the block size must be a positive int, not {n!r}")
    seeds = [hi - 1 for lo, hi in intervals if BAND[0] <= lo < BAND[1]]
    if not seeds:
        raise ValueError("no registered or named-spent interval lies in the band")
    highest = max(seeds)
    base = -(-(highest + 1 + n) // ALIGN) * ALIGN            # the least multiple of ALIGN >= highest + 1 + n
    block = (base, base + n)
    if block[1] > BAND[1]:
        raise Exhausted(f"the {n}-seed block from BASE {base} would leave the band")
    return highest, base, block
