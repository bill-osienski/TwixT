"""H4 STEP 4 -- READ-ONLY COLLISION SURVEY of five candidate seed blocks.

NOT THE COLLISION PROOF. The proof is re-run, against the registries AS THEY
STAND, in the registration commit itself (step 4b). This survey only shows the
candidates are viable before anything is reserved. Nothing is drawn, registered
or built: imports, XOR and set arithmetic.

Method: collision proof v15 (2026-09-18) unchanged -- every candidate seed and
its four XOR-mask derivations disjoint from every prior reservation (all four
registries, CONSUMED_SEEDS, the named spent blocks, the H3 generation ranges no
registry can see); derivations injective; the nearest other interval at least
the candidate's OWN size away. Added for H4: the five candidates must also be
disjoint from, and floor-separated from, EACH OTHER; and every nine-digit 2026
literal found in scripts/tests/docs outside a registered interval is a prior
term too (the repo scan found 202630000 in a test and 202699000 in the H4
fixture tests).

Pilot = 16 pairs = 32 seeds; each study segment = 74 pairs = 148 seeds (one per
game, the runner's schedule). T1j is unseeded; the seed reaches the incumbent only.
"""
import pathlib
import re

from scripts.GPU.alphazero.e4_screen_reference import (
    ACCOUNTED_SEED_INTERVALS, EXPOSED_SEED_INTERVALS,
    TEST_ONLY_SEED_INTERVALS, RETIRED_SEED_INTERVALS)
from scripts.GPU.alphazero.twixtbot_g3_schedule import CONSUMED_SEEDS
from scripts.GPU.alphazero.twixtbot_g3_reference import SeededReferenceAgent
from scripts.GPU.alphazero.d1_selection import SEED_INTERVAL as D1_PAPER
from scripts.GPU.alphazero.h1_viability_rules import H1_ATTEMPT1_SEED_BLOCK as H1_A1
from scripts.GPU.alphazero.h1_viability_rules import H1_SEED_BLOCK as H1_A2
from scripts.GPU.alphazero.h2_match_rules import H2_SEED_BLOCK as H2_A3
from scripts.GPU.alphazero.h2_match_rules import H2_ATTEMPT1_SEED_BLOCK as H2_A1
from scripts.GPU.alphazero.h2_match_rules import H2_ATTEMPT2_SEED_BLOCK as H2_A2
from scripts.GPU.alphazero.h3_pilot_runner import PILOT_SEED_BLOCK as H3_PILOT
from scripts.GPU.alphazero.h3_study_runner import SEGMENT_SEED_BLOCKS, RETIRED_SEGMENT_BLOCKS
from scripts.GPU.alphazero.h4_pilot_feasibility import PAIRS_PER_SEGMENT, PILOT, STUDY
from scripts.GPU.alphazero import h3_study_rules as H3R

ROOT = pathlib.Path(__file__).resolve().parents[4]

# ── the five candidates ───────────────────────────────────────────────────
PILOT_N, SEGMENT_N = 2 * PAIRS_PER_SEGMENT[PILOT], 2 * PAIRS_PER_SEGMENT[STUDY]
assert (PILOT_N, SEGMENT_N) == (32, 148)
CANDIDATES = {"H4_PILOT": (202_632_000, 202_632_000 + PILOT_N)}
for k in range(4):
    lo = 202_634_000 + 2_000 * k
    CANDIDATES[f"H4_STUDY_SEGMENT_{k}"] = (lo, lo + SEGMENT_N)

MASKS = sorted({*SeededReferenceAgent.SEARCH_MASK.values(),
                *SeededReferenceAgent.READOUT_MASK.values()})
assert len(MASKS) == 4, MASKS

GEN_RANGES = [(lo, hi) for lo, hi, _why in H3R.RETIRED_GENERATION_RANGES]
GEN_RANGES.append(H3R.generation_seed_range(H3R.GEN_SEED_UNIFORM))


def repo_literals():
    """Every nine-digit 2026 literal in scripts/tests/docs NOT inside a registered
    interval -- a seed someone wrote down that no registry holds."""
    acc = list(ACCOUNTED_SEED_INTERVALS)
    out = set()
    for base in ("scripts", "tests", "docs/superpowers"):
        for p in (ROOT / base).rglob("*"):
            if p.is_file() and p.suffix in (".py", ".md", ".json", ".jsonl", ".txt"):
                for m in re.finditer(r"(?<![\d.])(2026\d{5})(?!\d)",
                                     p.read_text(errors="ignore")):
                    v = int(m.group(1))
                    if not any(lo <= v < hi for lo, hi in acc):
                        out.add(v)
    return out


CATS = {
    "ACCOUNTED": {s for lo, hi in ACCOUNTED_SEED_INTERVALS for s in range(lo, hi)},
    "EXPOSED": {s for lo, hi in EXPOSED_SEED_INTERVALS for s in range(lo, hi)},
    "RETIRED": {s for lo, hi in RETIRED_SEED_INTERVALS for s in range(lo, hi)},
    "TEST_ONLY": {s for lo, hi in TEST_ONLY_SEED_INTERVALS for s in range(lo, hi)},
    "CONSUMED_SEEDS": set(CONSUMED_SEEDS),
    "NAMED_SPENT": {s for b in (D1_PAPER, H3_PILOT, H1_A1, H1_A2, H2_A1, H2_A2, H2_A3,
                                *SEGMENT_SEED_BLOCKS, *RETIRED_SEGMENT_BLOCKS)
                    for s in range(*b)},
    "REPO_LITERALS": repo_literals(),
}
for i, (lo, hi) in enumerate(GEN_RANGES):
    CATS[f"GEN_RANGE_{i}"] = set(range(lo, hi))

ALL_INTERVALS = (tuple(ACCOUNTED_SEED_INTERVALS) + tuple(EXPOSED_SEED_INTERVALS)
                 + tuple(RETIRED_SEED_INTERVALS) + tuple(TEST_ONLY_SEED_INTERVALS)
                 + (D1_PAPER, H3_PILOT, H1_A1, H1_A2, H2_A1, H2_A2, H2_A3,
                    *SEGMENT_SEED_BLOCKS, *RETIRED_SEGMENT_BLOCKS) + tuple(GEN_RANGES)
                 + tuple((s, s + 1) for s in CONSUMED_SEEDS)
                 + tuple((s, s + 1) for s in CATS["REPO_LITERALS"]))


def derivations(seeds):
    return {v for s in seeds for v in (s, *(s ^ m for m in MASKS))}


def check(block, others=()):
    """v15's check, with `others` -- the OTHER candidates -- as further priors."""
    ours = set(range(*block))
    cats = dict(CATS)
    cats["OTHER_CANDIDATES"] = {s for b in others for s in range(*b)}
    prior = set().union(*cats.values())
    vals = derivations(ours)
    floor = block[1] - block[0]
    gaps = [lo - block[1] if lo >= block[1] else block[0] - hi
            for lo, hi in ALL_INTERVALS + tuple(others)
            if not (lo < block[1] and block[0] < hi)]
    nearest = min(gaps)
    direct = {k: len(ours & v) for k, v in cats.items() if ours & v}
    streams = len(vals & derivations(prior))
    return {"block": block, "n": len(ours), "direct": direct, "streams": streams,
            "injective": len(vals) == len(ours) * 5, "nearest_gap": nearest,
            "floor": floor, "clean": not direct and not streams
            and len(vals) == len(ours) * 5 and nearest >= floor}


if __name__ == "__main__":
    print("H4 STEP 4 -- READ-ONLY COLLISION SURVEY (not the proof)")
    print("=" * 66)
    print(f"masks (imported): {[hex(m) for m in MASKS]}")
    for k, v in CATS.items():
        print(f"    {k:16s} {len(v):>9}")
    print(f"    repo literals outside every registered interval: "
          f"{sorted(CATS['REPO_LITERALS'])}")
    print()
    results = {}
    for name, block in CANDIDATES.items():
        others = tuple(b for n, b in CANDIDATES.items() if n != name)
        r = results[name] = check(block, others)
        print(f"{name:20s} {r['block']}  n={r['n']:>3}  direct={r['direct'] or 'NONE'}  "
              f"streams={r['streams']}  injective={r['injective']}  "
              f"gap={r['nearest_gap']} (floor {r['floor']})  CLEAN={r['clean']}")
    total = sum(r["n"] for r in results.values())
    print(f"\ntotal seeds: {total} = pilot {PILOT_N} + 4 x {SEGMENT_N}")
    print()
    print("NEGATIVE CONTROLS -- each must be REJECTED")
    print("-" * 66)
    seg0 = CANDIDATES["H4_STUDY_SEGMENT_0"]
    controls = [
        ("H3 segment-0 replacement block (spent)", SEGMENT_SEED_BLOCKS[0]),
        ("H3's retired segment-0 quarter", RETIRED_SEGMENT_BLOCKS[0]),
        ("the H3 pilot block", H3_PILOT),
        ("H2 attempt-3 block", H2_A3),
        ("D1's paper reservation", D1_PAPER),
        ("the TEST_ONLY band", tuple(TEST_ONLY_SEED_INTERVALS[0])),
        ("the literal 202630000 in tests/test_h1_viability.py", (202_630_000, 202_630_032)),
        ("the literal 202699000 in tests/test_h4_runner.py", (202_699_000, 202_699_032)),
        ("overlaps H4 segment 0 by one seed", (seg0[1] - 1, seg0[1] + 147)),
        ("starts where H4 segment 0 ends -- inside the floor", (seg0[1], seg0[1] + 148)),
        ("one seed inside segment 0's floor", (seg0[1] + 147, seg0[1] + 295)),
        ("inside the live H3 uniform generation range", (GEN_RANGES[-1][0],
                                                          GEN_RANGES[-1][0] + 148)),
    ]
    rejected = 0
    for why, block in controls:
        r = check(block, tuple(CANDIDATES.values()))
        ok = not r["clean"]
        rejected += ok
        print(f"  {'REJECTED' if ok else 'NOT REJECTED':12s} {why}  {block}")
    print(f"\ncontrols rejected: {rejected}/{len(controls)}")
    clean = all(r["clean"] for r in results.values())
    print(f"SURVEY VERDICT   : {'ALL FIVE CANDIDATES CLEAN' if clean else 'NOT CLEAN'}"
          f"{'' if rejected == len(controls) else ' -- BUT A CONTROL WAS NOT REJECTED'}")
