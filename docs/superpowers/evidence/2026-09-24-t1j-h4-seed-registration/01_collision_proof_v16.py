"""H4 SEED REGISTRATION -- COLLISION PROOF v16. READ-ONLY.

    python -m <this file as a path>  pre    # before registration: MUST be CLEAN
    python -m <this file as a path>  post   # after: each block excluded BY IDENTITY

The five H4 blocks of the step-4 card §2 (bound card), proved with v15's method
unchanged -- each candidate seed and its four XOR-mask derivations disjoint from
every prior term (all four registries, CONSUMED_SEEDS, the named spent H1/H2/H3/D1
blocks, the four H3 generation ranges no registry holds); derivations injective;
nearest other interval at least the candidate's OWN size away -- plus the two
additions of the 2026-09-24 survey: the five candidates are priors for EACH OTHER,
and every nine-digit 2026 literal in scripts/tests/docs outside a registered
interval is a prior term.

`pre` runs against the registries AS THEY STAND BEFORE REGISTRATION (step-4
card amendment 5): the blocks must be absent from every registry and CLEAN.
`post` runs after they are written into ACCOUNTED_SEED_INTERVALS: each block must
be registered as EXACTLY its own interval, neither exposed nor retired, and is
excluded from the priors BY IDENTITY for its own check -- never by subtracting its
seeds, which (2026-09-04) excused every control equal to an existing reservation.

Nothing is drawn, registered or built. Registries and masks are IMPORTED.
"""
import pathlib
import re
import sys

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
from scripts.GPU.alphazero import h3_study_rules as H3R

ROOT = pathlib.Path(__file__).resolve().parents[4]
MODE = sys.argv[1] if len(sys.argv) > 1 else ""
assert MODE in ("pre", "post"), "usage: pre | post"

#: The step-4 card §2 table, transcribed HERE -- the proof checks the card's
#: blocks, not whatever the code later calls them.
BLOCKS = {"H4_PILOT": (202_632_000, 202_632_032),
          "H4_STUDY_SEGMENT_0": (202_634_000, 202_634_148),
          "H4_STUDY_SEGMENT_1": (202_636_000, 202_636_148),
          "H4_STUDY_SEGMENT_2": (202_638_000, 202_638_148),
          "H4_STUDY_SEGMENT_3": (202_640_000, 202_640_148)}
assert [hi - lo for lo, hi in BLOCKS.values()] == [32, 148, 148, 148, 148]

MASKS = sorted({*SeededReferenceAgent.SEARCH_MASK.values(),
                *SeededReferenceAgent.READOUT_MASK.values()})
assert len(MASKS) == 4, MASKS
GEN_RANGES = [(lo, hi) for lo, hi, _why in H3R.RETIRED_GENERATION_RANGES]
GEN_RANGES.append(H3R.generation_seed_range(H3R.GEN_SEED_UNIFORM))
NAMED_SPENT = (D1_PAPER, H3_PILOT, H1_A1, H1_A2, H2_A1, H2_A2, H2_A3,
               *SEGMENT_SEED_BLOCKS, *RETIRED_SEGMENT_BLOCKS)


def repo_literals():
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


def intervals_set(ivs):
    return {s for lo, hi in ivs for s in range(lo, hi)}


REGISTRIES = {"ACCOUNTED": list(ACCOUNTED_SEED_INTERVALS),
              "EXPOSED": list(EXPOSED_SEED_INTERVALS),
              "RETIRED": list(RETIRED_SEED_INTERVALS),
              "TEST_ONLY": list(TEST_ONLY_SEED_INTERVALS)}
LITERALS = repo_literals()


def derivations(seeds):
    return {v for s in seeds for v in (s, *(s ^ m for m in MASKS))}


def check(block, others=(), *, own_registration=None):
    """`own_registration`: the interval this block is itself registered as,
    excluded BY IDENTITY from the registries and from the gap computation."""
    ours = set(range(*block))
    reg = {k: [iv for iv in v if tuple(iv) != own_registration] for k, v in REGISTRIES.items()}
    cats = {k: intervals_set(v) for k, v in reg.items()}
    cats.update({"CONSUMED_SEEDS": set(CONSUMED_SEEDS),
                 "NAMED_SPENT": intervals_set(NAMED_SPENT),
                 "REPO_LITERALS": set(LITERALS),
                 "OTHER_CANDIDATES": intervals_set(others)})
    for i, gr in enumerate(GEN_RANGES):
        cats[f"GEN_RANGE_{i}"] = set(range(*gr))
    prior = set().union(*cats.values())
    vals = derivations(ours)
    floor = block[1] - block[0]
    ivs = ([tuple(iv) for v in reg.values() for iv in v] + list(NAMED_SPENT)
           + list(GEN_RANGES) + list(others)
           + [(s, s + 1) for s in set(CONSUMED_SEEDS) | LITERALS])
    gaps = [lo - block[1] if lo >= block[1] else block[0] - hi
            for lo, hi in ivs if not (lo < block[1] and block[0] < hi)]
    nearest = min(gaps)
    direct = {k: len(ours & v) for k, v in cats.items() if ours & v}
    streams = len(vals & derivations(prior))
    return {"block": block, "n": len(ours), "direct": direct, "streams": streams,
            "injective": len(vals) == len(ours) * 5, "nearest_gap": nearest,
            "floor": floor, "clean": not direct and not streams
            and len(vals) == len(ours) * 5 and nearest >= floor}


if __name__ == "__main__":
    print(f"H4 SEED REGISTRATION -- COLLISION PROOF v16 -- MODE {MODE.upper()}")
    print("=" * 70)
    print(f"masks (imported): {[hex(m) for m in MASKS]}")
    print(f"registries: " + ", ".join(f"{k} {len(v)} intervals" for k, v in REGISTRIES.items()))
    print(f"repo literals outside every registered interval: {len(LITERALS)}")
    registered = {name: [k for k, v in REGISTRIES.items() if b in [tuple(x) for x in v]]
                  for name, b in BLOCKS.items()}
    status_ok = True
    for name, b in BLOCKS.items():
        overlapping = {k: sum(1 for lo, hi in v if lo < b[1] and b[0] < hi)
                       for k, v in REGISTRIES.items()}
        overlapping = {k: n for k, n in overlapping.items() if n}
        want = {} if MODE == "pre" else {"ACCOUNTED": 1}
        exact = registered[name] == ([] if MODE == "pre" else ["ACCOUNTED"])
        ok = overlapping == want and exact
        status_ok &= ok
        print(f"  registry state {name:20s} {b}: overlapping {overlapping or 'NONE'}"
              f"{'' if MODE == 'pre' else f', registered as exactly itself in {registered[name]}'}"
              f"  {'OK' if ok else 'WRONG'}")
    print()
    results = {}
    for name, block in BLOCKS.items():
        others = tuple(b for n, b in BLOCKS.items() if n != name)
        r = results[name] = check(block, others,
                                  own_registration=block if MODE == "post" else None)
        print(f"{name:20s} {r['block']}  n={r['n']:>3}  direct={r['direct'] or 'NONE'}  "
              f"streams={r['streams']}  injective={r['injective']}  "
              f"gap={r['nearest_gap']} (floor {r['floor']})  CLEAN={r['clean']}")
    print(f"\ntotal seeds: {sum(r['n'] for r in results.values())}")
    print("\nNEGATIVE CONTROLS -- each must be REJECTED")
    print("-" * 70)
    seg0 = BLOCKS["H4_STUDY_SEGMENT_0"]
    everything = tuple(BLOCKS.values())
    controls = [
        ("H3 segment-0 replacement block (spent)", SEGMENT_SEED_BLOCKS[0]),
        ("H3's retired segment-0 quarter", RETIRED_SEGMENT_BLOCKS[0]),
        ("the H3 pilot block", H3_PILOT),
        ("H2 attempt-3 block", H2_A3),
        ("D1's paper reservation", D1_PAPER),
        ("the TEST_ONLY band", tuple(TEST_ONLY_SEED_INTERVALS[0])),
        ("the stray literal 202630000", (202_630_000, 202_630_032)),
        ("the stray literal 202699000", (202_699_000, 202_699_032)),
        ("overlaps H4 segment 0 by one seed", (seg0[1] - 1, seg0[1] + 147)),
        ("starts where H4 segment 0 ends -- inside the floor", (seg0[1], seg0[1] + 148)),
        ("one seed inside segment 0's floor", (seg0[1] + 147, seg0[1] + 295)),
        ("EQUALS the H4 pilot block (an existing reservation, not its own)",
         BLOCKS["H4_PILOT"]),
        ("inside the live H3 uniform generation range",
         (GEN_RANGES[-1][0], GEN_RANGES[-1][0] + 148)),
    ]
    rejected = 0
    for why, block in controls:
        r = check(block, everything)
        ok = not r["clean"]
        rejected += ok
        print(f"  {'REJECTED' if ok else 'NOT REJECTED':12s} {why}  {block}")
    print(f"\ncontrols rejected: {rejected}/{len(controls)}")
    clean = all(r["clean"] for r in results.values()) and status_ok \
        and rejected == len(controls)
    print(f"VERDICT          : {'CLEAN' if clean else 'NOT CLEAN'}")
    sys.exit(0 if clean else 1)
