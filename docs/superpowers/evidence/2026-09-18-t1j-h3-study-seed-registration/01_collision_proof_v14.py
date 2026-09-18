"""H3 FULL STUDY match-seed collision proof for a FRESH 592-seed block. READ-ONLY.

v14 (2026-09-18, H3 STUDY pre-registration): the study needs 592 match seeds, one
per game, for 296 colour-reversed pairs across four 148-seed segment quarters.

🔴 FOUR GENERATION RANGES ARE PRIOR AND NO REGISTRY HOLDS ANY OF THEM. Three are
retired (the 148-opening uniform range, superseded; co-produced attempts 1 and 2,
both SPENT WHOLE on VOIDs) and one is LIVE -- it produced the population frozen on
2026-09-17 and pinned on 2026-09-18. A registry-only enumeration cannot see a
single one, so all four are added explicitly. This is the same term v10 had to add
by hand for D1's paper reservation, and the same one v13 added for the generation
ranges; it keeps having to be added because the registries are about MATCH seeds
and nothing else knows about generation.

v13 (2026-09-17) proved the uniform generation range. v10 (2026-09-14) is the
parent: gap floor = the candidate's OWN size, candidate excluded BY IDENTITY --
never by subtracting its seeds, which (2026-09-04) excused every control that
equalled an existing reservation.

⚠ THE GAP FLOOR HERE IS 592, the candidate's own size. It is a SMALLER number than
v13's 118,400 because this candidate is smaller; the policy is unchanged. The
ACTUAL nearest distance is reported alongside, so a narrow choice cannot hide
behind a small threshold.

Nothing is drawn, nothing is registered, no agent is built: XOR and set arithmetic
only. Registries and the four masks are IMPORTED, never retyped, so the proof
cannot drift from what the code enforces.
"""
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
from scripts.GPU.alphazero import h3_study_rules as R

# ── the candidate ─────────────────────────────────────────────────────────
CANDIDATE = (202_626_000, 202_626_592)
assert CANDIDATE[1] - CANDIDATE[0] == R.N_GAMES == 592, "one seed per game"
assert R.N_PAIRS * 2 == R.N_GAMES, "296 pairs, both colours"

MASKS = sorted({*SeededReferenceAgent.SEARCH_MASK.values(),
                *SeededReferenceAgent.READOUT_MASK.values()})
assert len(MASKS) == 4, MASKS

# ── the generation ranges, which NO registry can see ──────────────────────
GEN_RANGES = [(lo, hi, why.split(".")[0]) for lo, hi, why in R.RETIRED_GENERATION_RANGES]
_live_lo, _live_hi = R.generation_seed_range(R.GEN_SEED_UNIFORM)
GEN_RANGES.append((_live_lo, _live_hi,
                   "the LIVE uniform range -- it produced the population frozen "
                   "2026-09-17 and pinned 2026-09-18"))
assert len(GEN_RANGES) == 4, GEN_RANGES

CATS = {
    "ACCOUNTED": {s for lo, hi in ACCOUNTED_SEED_INTERVALS for s in range(lo, hi)},
    "EXPOSED": {s for lo, hi in EXPOSED_SEED_INTERVALS for s in range(lo, hi)},
    "RETIRED": {s for lo, hi in RETIRED_SEED_INTERVALS for s in range(lo, hi)},
    "TEST_ONLY": {s for lo, hi in TEST_ONLY_SEED_INTERVALS for s in range(lo, hi)},
    "CONSUMED_SEEDS": set(CONSUMED_SEEDS),
    # 🔑 D1's §14 block was registered, drawn and retired whole on 2026-09-08, so
    # it already sits in ACCOUNTED/EXPOSED/RETIRED. It is named here only so a
    # control can reach it by name; the set is a subset of those above.
    "D1_PAPER_S14": set(range(*D1_PAPER)),
    "H3_PILOT": set(range(*H3_PILOT)),
}
for i, (lo, hi, _why) in enumerate(GEN_RANGES):
    CATS[f"GEN_RANGE_{i}"] = set(range(lo, hi))
PRIOR = set().union(*CATS.values())

ALL_INTERVALS = (
    tuple(ACCOUNTED_SEED_INTERVALS) + tuple(EXPOSED_SEED_INTERVALS)
    + tuple(RETIRED_SEED_INTERVALS) + tuple(TEST_ONLY_SEED_INTERVALS)
    + (D1_PAPER, H3_PILOT, H1_A1, H1_A2, H2_A1, H2_A2, H2_A3)
    + tuple((lo, hi) for lo, hi, _ in GEN_RANGES)
    + tuple((s, s + 1) for s in CONSUMED_SEEDS)
)

#: 🔴 THE GAP FLOOR IS THE CANDIDATE'S OWN SIZE.
GAP_FLOOR = CANDIDATE[1] - CANDIDATE[0]


def derivations(seeds):
    """seed + one value per mask, both colours. XOR ONLY -- nothing is built."""
    return {v for s in seeds for v in (s, *(s ^ m for m in MASKS))}


def check(block, *, own_registration=None):
    """Is `block` disjoint from every OTHER reservation, directly and through the
    derived streams, and separated from all of them by the gap floor?

    ⚠ `own_registration` NAMES THE INTERVAL THIS BLOCK IS ITSELF REGISTERED AS and
    is excluded BY IDENTITY, in the OVERLAP check and in the GAP check alike.
    Without the second, a block re-proved after registration measures its distance
    to itself as zero and fails its own floor.

    🔴 EXCLUDING ONE NAMED INTERVAL IS THE NARROW THING MEANT. Subtracting the
    block's SEEDS from the prior set (the 2026-09-04 error) excuses any control
    that EQUALS an existing reservation, and every negative control went green at
    once.
    """
    ours = set(range(*block))
    cats = dict(CATS)
    if own_registration is not None:
        drop = set(range(*own_registration))
        cats = {k: (v - drop if set(v) >= drop else v) for k, v in CATS.items()}
    prior = set().union(*cats.values())
    prior_values = derivations(prior)
    vals = derivations(ours)
    n = len(ours)
    own = tuple(own_registration) if own_registration else None
    gaps = [lo - block[1] if lo >= block[1] else block[0] - hi
            for lo, hi in ALL_INTERVALS
            if (lo, hi) != own and not (lo < block[1] and block[0] < hi)]
    nearest = min(gaps) if gaps else None
    return {
        "block": tuple(block), "n": n,
        "direct": {k: len(ours & v) for k, v in cats.items() if ours & v},
        "stream_collisions": len(vals & prior_values),
        "own_values": len(vals), "injective": len(vals) == n * 5,
        "nearest_gap": nearest, "gap_floor": GAP_FLOOR,
        "gap_ok": nearest is not None and nearest >= GAP_FLOOR,
        "clean": (not (ours & prior) and not (vals & prior_values)
                  and len(vals) == n * 5
                  and nearest is not None and nearest >= GAP_FLOOR),
    }


if __name__ == "__main__":
    print("H3 FULL STUDY -- MATCH SEED BLOCK")
    print("=" * 66)
    print(f"masks (imported)  : {[hex(m) for m in MASKS]}")
    print(f"prior seeds       : {len(PRIOR)}")
    for k, v in CATS.items():
        print(f"    {k:16s} {len(v):>7}")
    print(f"prior values      : {len(derivations(PRIOR))} (incl. derivations)")
    print(f"gap floor         : {GAP_FLOOR} (the candidate's OWN size)")
    print()
    print("THE FOUR GENERATION RANGES NO REGISTRY CAN SEE:")
    for lo, hi, why in GEN_RANGES:
        print(f"    [{lo}, {hi})  {why[:56]}")
    print()
    r = check(CANDIDATE, own_registration=CANDIDATE)
    print(f"CANDIDATE {r['block']}  n={r['n']}")
    print(f"  direct overlap      : {r['direct'] or 'NONE'}")
    print(f"  stream collisions   : {r['stream_collisions']}")
    print(f"  own derivations     : {r['own_values']} (injective={r['injective']})")
    print(f"  nearest gap         : {r['nearest_gap']}  (floor {r['gap_floor']})"
          f"  ok={r['gap_ok']}")
    print(f"  CLEAN               : {r['clean']}")
    print()
    print("NEGATIVE CONTROLS -- a check that never rejects proves nothing")
    print("-" * 66)
    controls = [
        ("the SPENT H3 pilot block", H3_PILOT),
        ("the SPENT H1 attempt-1 block", H1_A1),
        ("the SPENT H1 attempt-2 block", H1_A2),
        ("the SPENT H2 attempt-1 block", H2_A1),
        ("the SPENT H2 attempt-2 block", H2_A2),
        ("the SPENT H2 attempt-3 block", H2_A3),
        ("D1's SPENT paper reservation", D1_PAPER),
        ("L0's spent block", (202613000, 202613064)),
        ("straddles the pilot block's END by exactly one seed",
         (H3_PILOT[1] - 1, H3_PILOT[1] - 1 + 592)),
        ("starts exactly where the pilot ends -- no overlap, INSIDE the floor",
         (H3_PILOT[1], H3_PILOT[1] + 592)),
        ("clear of everything but ONE SEED inside the gap floor",
         (H3_PILOT[1] + GAP_FLOOR - 1, H3_PILOT[1] + GAP_FLOOR - 1 + 592)),
        ("inside the LIVE uniform GENERATION range", (_live_lo, _live_lo + 592)),
        ("inside co-produced attempt 2's SPENT generation range",
         (20_261_400_000, 20_261_400_592)),
        ("inside the retired 148-opening uniform range",
         (20_261_000_000, 20_261_000_592)),
        ("the TEST_ONLY reservation", (90009000, 90009100)),
    ]
    rejected = 0
    for why, blk in controls:
        c = check(blk)
        ok = not c["clean"]
        rejected += ok
        reason = ("overlap " + str(c["direct"])) if c["direct"] else (
            f"gap {c['nearest_gap']} < floor {c['gap_floor']}" if not c["gap_ok"]
            else f"stream collisions {c['stream_collisions']}" if c["stream_collisions"]
            else "🔴 NOT REJECTED")
        print(f"  {'REJECTED' if ok else '🔴 ACCEPTED':>12}  {why}")
        print(f"                {tuple(blk)} -> {reason}")
    print()
    print(f"controls rejected : {rejected}/{len(controls)}")
    # 🔑 THE POSITIVE CONTROL: the candidate must STILL be clean after all that,
    # or the checks above reject everything and prove nothing either.
    print(f"VERDICT           : "
          f"{'CLEAN' if r['clean'] and rejected == len(controls) else '🔴 NOT CLEAN'}")
