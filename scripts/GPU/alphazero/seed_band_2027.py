"""THE TWO-BAND SEED METHOD (2026-10-30): the scanner extended to the 2027 band, and the rule that places a block there.

`seed_selection.py` records that the nine-digit 2026 band's room ends at an H4 fixture literal near its top. The
combination pilot's 1,000-slot generation block no longer fits below it: the rule's block [BASE, BASE + 1,000) would
contain that fixture's values (ledger, addendum after entry 42). The user's review (2026-10-30) chose the 2027 band
[2027 * 10**5, 2028 * 10**5) for that block, with the scanner extended to see both bands. It is only a range of
numbers; nothing is scheduled in 2027.

THE SCANNER (`scan`): the pinned survey's (`evidence/tactical-candidate-seed-survey/01_seed_survey.py`), unchanged
except that a nine-digit literal counts when it begins 2026 OR 2027 -- plain or underscore-grouped (digits in groups
joined by `_`), with the same boundaries: no digit or `.` before it, no digit (or `_`, when grouped) after it, every
occurrence counted, the same exemption by (line, column). Values in neither band are not counted, as before.
`install(S)` rebinds the scanner of a PRIVATE copy of the pinned survey module, loaded by a stage's own survey script;
everything else of the pinned method is unchanged: the scope, the fail-closed reading, the exemptions, the prior terms
(registries, named-spent blocks, H3's generation ranges, CONSUMED_SEEDS), all four derived streams and the check
(direct overlap, streams, injectivity, gap floor = the block's own size). The pinned files are not edited.

THE RULE (`candidate`), fixed before any survey runs in the band, deterministic on its inputs:
    anchor = the largest of: the band's lower edge - 1; the last seed (hi - 1) of every registered or named-spent
             interval whose last seed lies at or above the band's lower edge; every repository literal the scan found
             at or above that edge;
    BASE   = the least multiple of ALIGN (100) >= anchor + 1 + N;  block = [BASE, BASE + N).
  * An EMPTY band (nothing registered and no literal in it): the anchor is the lower edge - 1, so BASE is the least
    multiple of 100 at least N above the edge -- the 2026 band's top is then at least N below the block.
  * Unlike `seed_selection.candidate`, the band's repository literals enter the anchor. The 2027 band already holds a
    few (synthetic test bases and one corpus value); a rule blind to them would pick a block that the check's gap
    floor is known to reject. Nothing is edited to make room.
  * Monotone: the block lies above every interval and literal in the band, so later closings still append higher.
  * Refused (`Exhausted`), never wrapped or searched, if the block would leave the band.
  The check then decides CLEAN or NOT CLEAN on the full prior terms; a NOT CLEAN block is returned for review, never
  adjusted.

Nothing here reads or writes a registry, reserves, exposes or retires a seed, or holds a seed literal.
"""
import bisect
import re
from typing import Iterable, Mapping, Optional, Tuple

BAND_2026 = (2026 * 10 ** 5, 2027 * 10 ** 5)     # the pinned band
BAND = (2027 * 10 ** 5, 2028 * 10 ** 5)          # the new band: where `candidate` places blocks
PREFIXES = ("2026", "2027")
ALIGN = 100

#: the pinned scanner's two patterns, with the prefix widened to both bands
PLAIN = re.compile(r"(?<![\d.])(202[67]\d{5})(?!\d)")
GROUPED = re.compile(r"(?<![\d.])(\d(?:_?\d)*)(?![\d_])")
#: what `install` expects to replace: the pinned module's own patterns
PINNED_PLAIN = r"(?<![\d.])(2026\d{5})(?!\d)"


class Exhausted(Exception):
    """The block would leave the band: no seed is chosen."""


def scan(text: str, exempt=frozenset(), *, grouped: bool = True) -> dict:
    """{value: count} of literals in either band at every occurrence (the pinned `scan` with both prefixes).
    `grouped=False` is the plain pattern alone, as in the pinned module."""
    starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
    hits = {}

    def keep(pos, value):
        line = bisect.bisect_right(starts, pos)
        if (line, pos - starts[line - 1]) not in exempt:
            hits[value] = hits.get(value, 0) + 1
    seen = set()
    for m in PLAIN.finditer(text):
        seen.add(m.start())
        keep(m.start(), int(m.group(1)))
    if grouped:
        for m in GROUPED.finditer(text):
            digits = m.group(1).replace("_", "")
            if "_" in m.group(1) and len(digits) == 9 and digits[:4] in PREFIXES and m.start() not in seen:
                keep(m.start(), int(digits))
    return hits


def install(S) -> None:
    """Rebind the scanner of S, a private copy of the pinned survey module, to `scan`. Its `repo_literals` looks the
    name up at call time, so the scope, reading and exemptions stay the pinned ones. Refuses an S whose scanner is not
    the pinned one (nothing else is touched)."""
    if S.scan is scan:
        return
    if S.V16.pattern != PINNED_PLAIN or S.GROUPED.pattern != GROUPED.pattern or S.BAND != BAND_2026:
        raise ValueError("not the pinned survey module: its scanner differs from the one this extends")
    S.scan = scan


def candidate(intervals: Iterable[Tuple[int, int]], literals: Optional[Mapping[int, object]] | Iterable[int],
              n: int) -> Tuple[Optional[int], int, Tuple[int, int]]:
    """(anchor seed or None for an empty band, BASE, block) for an `n`-seed block in the 2027 band, above every
    interval and literal in it."""
    if type(n) is not int or n < 1:
        raise ValueError(f"the block size must be a positive int, not {n!r}")
    found = [hi - 1 for _lo, hi in intervals if hi - 1 >= BAND[0]]
    found += [v for v in literals if type(v) is int and v >= BAND[0]]
    anchor = max(found, default=BAND[0] - 1)
    base = -(-(anchor + 1 + n) // ALIGN) * ALIGN           # the least multiple of ALIGN >= anchor + 1 + n
    block = (base, base + n)
    if block[1] > BAND[1]:
        raise Exhausted(f"the {n}-seed block from BASE {base} would leave the 2027 band")
    return (anchor if found else None), base, block
