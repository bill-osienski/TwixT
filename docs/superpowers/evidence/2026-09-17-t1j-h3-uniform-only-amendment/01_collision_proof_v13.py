"""H3 UNIFORM-ONLY collision proof for the AMENDMENT-3 generation allocation. READ-ONLY.

v13 (2026-09-17, Amendment 3): the co-produced stratum is CLOSED, so the study needs
ONE uniform generation range covering every candidate for all 296 openings:
296 x MAX_ATTEMPTS(400) = 118,400 seeds. The standing uniform range is 59,200 -- sized
for 148 openings -- and is INSUFFICIENT. This proves a single fresh contiguous range,
rather than combining the old one with a second: `attempt_seed` is
`base + i*MAX_ATTEMPTS + j`, a formula that is only total over a CONTIGUOUS base, and a
piecewise base is a seam this programme has been bitten by before.

🔴 THREE GENERATION RANGES ARE NOW PRIOR, AND NONE OF THEM IS IN A REGISTRY.
  * attempt 1's co-produced range -- SPENT WHOLE 2026-09-16 (VOID at opening 0, 0 accepted).
  * attempt 2's co-produced range -- SPENT WHOLE 2026-09-16 (VOID at opening 0, 0 accepted).
  * the standing 148-opening uniform range -- NOT spent by any authorized run, but
    RETIRED UNUSED by Amendment 3 rather than folded into the new allocation. It has been
    drawn on in-process by `tests/test_h3_study_rules.py` and
    `h3_study_prerun_verification.py` every time they run. Those draws built no agent and derived no search or readout stream (stratum A is
    engine-free), so this is conservatism, not necessity -- and it costs nothing, because
    the range is superseded either way.
Registry enumeration cannot see any of the three. They are added explicitly, exactly as
v10 had to add D1's paper reservation.

v10 (2026-09-14) is the parent: gap floor = the candidate's OWN size, candidate excluded
BY IDENTITY (never by subtracting its seeds), four masks and the registries IMPORTED and
never retyped. Nothing is drawn, nothing is registered, no generator is built.
"""
from scripts.GPU.alphazero.e4_screen_reference import (
    ACCOUNTED_SEED_INTERVALS, EXPOSED_SEED_INTERVALS,
    TEST_ONLY_SEED_INTERVALS, RETIRED_SEED_INTERVALS)
from scripts.GPU.alphazero.twixtbot_g3_schedule import CONSUMED_SEEDS
from scripts.GPU.alphazero.twixtbot_g3_reference import SeededReferenceAgent
from scripts.GPU.alphazero.d1_selection import SEED_INTERVAL as D1_PAPER
from scripts.GPU.alphazero.h3_pilot_runner import PILOT_SEED_BLOCK as H3_PILOT
from scripts.GPU.alphazero import h3_study_rules as R

# ── the prior generation ranges, IMPORTED so the proof cannot drift ───────────
OLD_UNIFORM = R.generation_seed_range(R.GEN_SEED_UNIFORM)
CO_ATTEMPT1 = R.SPENT_GENERATION_RANGES[0]
CO_ATTEMPT2 = R.generation_seed_range(R.GEN_SEED_CO_PRODUCED)
assert OLD_UNIFORM == (20_261_000_000, 20_261_059_200), OLD_UNIFORM
assert CO_ATTEMPT1 == (20_261_200_000, 20_261_259_200), CO_ATTEMPT1
assert CO_ATTEMPT2 == (20_261_400_000, 20_261_459_200), CO_ATTEMPT2
assert OLD_UNIFORM[1] - OLD_UNIFORM[0] == 148 * R.MAX_ATTEMPTS, "sized for 148 openings"

# ── the candidate: ONE range for all 296 openings x every attempt ─────────────
NEEDED = R.N_PAIRS * R.MAX_ATTEMPTS
assert NEEDED == 118_400, NEEDED
assert NEEDED > OLD_UNIFORM[1] - OLD_UNIFORM[0], "the standing range must be short, or v13 has no reason to exist"
GEN_SEED_UNIFORM_296 = 20_261_600_000
CANDIDATE = (GEN_SEED_UNIFORM_296, GEN_SEED_UNIFORM_296 + NEEDED)

MASKS = sorted({*SeededReferenceAgent.SEARCH_MASK.values(),
                *SeededReferenceAgent.READOUT_MASK.values()})
assert len(MASKS) == 4, MASKS

CATS = {
    "ACCOUNTED": {s for lo, hi in ACCOUNTED_SEED_INTERVALS for s in range(lo, hi)},
    "EXPOSED": {s for lo, hi in EXPOSED_SEED_INTERVALS for s in range(lo, hi)},
    "RETIRED": {s for lo, hi in RETIRED_SEED_INTERVALS for s in range(lo, hi)},
    "TEST_ONLY": {s for lo, hi in TEST_ONLY_SEED_INTERVALS for s in range(lo, hi)},
    "CONSUMED_SEEDS": set(CONSUMED_SEEDS),
    "D1_PAPER_S14": set(range(*D1_PAPER)),
    "H3_PILOT": set(range(*H3_PILOT)),
    # the three generation ranges no registry can see
    "GEN_UNIFORM_OLD": set(range(*OLD_UNIFORM)),
    "GEN_CO_ATTEMPT1": set(range(*CO_ATTEMPT1)),
    "GEN_CO_ATTEMPT2": set(range(*CO_ATTEMPT2)),
}
PRIOR = set().union(*CATS.values())


def derivations(seeds):
    """seed + one value per mask, both colours. XOR ONLY -- nothing is built."""
    return {v for s in seeds for v in (s, *(s ^ m for m in MASKS))}


# 🔴 THE GAP FLOOR IS THE CANDIDATE'S OWN SIZE: 118,400. The proof reports the ACTUAL
# nearest distance too, so a narrow choice cannot hide behind a threshold.
GAP_FLOOR = NEEDED
ALL_INTERVALS = (
    tuple(ACCOUNTED_SEED_INTERVALS) + tuple(EXPOSED_SEED_INTERVALS)
    + tuple(RETIRED_SEED_INTERVALS) + tuple(TEST_ONLY_SEED_INTERVALS)
    + (D1_PAPER, H3_PILOT, OLD_UNIFORM, CO_ATTEMPT1, CO_ATTEMPT2)
    + tuple((s, s + 1) for s in CONSUMED_SEEDS)
)


def check(block, *, own_registration=None):
    """Is `block` disjoint from every OTHER reservation, directly and through the streams?

    `own_registration` names the ONE interval this block is itself registered as and is
    excluded BY IDENTITY -- never by subtracting its seeds, which (2026-09-04) excused
    every control that equalled an existing reservation.
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
    gaps = [lo - block[1] if lo >= block[1] else block[0] - hi
            for lo, hi in ALL_INTERVALS
            if (lo, hi) != tuple(own_registration or ()) and not (lo < block[1] and block[0] < hi)]
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
    print("H3 AMENDMENT 3 -- UNIFORM-ONLY GENERATION ALLOCATION")
    print("=" * 64)
    print(f"masks (imported)  : {[hex(m) for m in MASKS]}")
    print(f"prior seeds       : {len(PRIOR)}  { {k: len(v) for k, v in CATS.items()} }")
    print(f"prior values      : {len(derivations(PRIOR))} (incl. derivations)")
    print(f"candidates needed : {R.N_PAIRS} openings x {R.MAX_ATTEMPTS} attempts = {NEEDED}")
    print(f"standing uniform  : {OLD_UNIFORM[1]-OLD_UNIFORM[0]}  -> SHORT BY {NEEDED-(OLD_UNIFORM[1]-OLD_UNIFORM[0])}")
    print()
    r = check(CANDIDATE, own_registration=CANDIDATE)
    print(f"CANDIDATE {r['block']}  n={r['n']}")
    print(f"  direct overlap    : {r['direct'] or 'NONE'}")
    print(f"  stream collisions : {r['stream_collisions']}")
    print(f"  own derivations   : {r['own_values']} (injective={r['injective']})")
    print(f"  nearest gap       : {r['nearest_gap']}  (floor {r['gap_floor']})  ok={r['gap_ok']}")
    print(f"  CLEAN             : {r['clean']}")
    print()
    print("NEGATIVE CONTROLS -- a check that never rejects proves nothing")
    print("-" * 64)
    controls = [
        ("attempt 1's SPENT co-produced range", CO_ATTEMPT1),
        ("attempt 2's SPENT co-produced range", CO_ATTEMPT2),
        ("the standing 148-opening uniform range, retired unused by Amendment 3", OLD_UNIFORM),
        ("the standing uniform range EXTENDED to 118,400 -- the naive fix", (OLD_UNIFORM[0], OLD_UNIFORM[0] + NEEDED)),
        ("straddles attempt 2's END by exactly one seed", (CO_ATTEMPT2[1] - 1, CO_ATTEMPT2[1] - 1 + NEEDED)),
        ("starts exactly where attempt 2 ends -- no overlap, INSIDE the gap floor", (CO_ATTEMPT2[1], CO_ATTEMPT2[1] + NEEDED)),
        ("clear of everything but 1 seed inside the gap floor", (CO_ATTEMPT2[1] + GAP_FLOOR - 1, CO_ATTEMPT2[1] + GAP_FLOOR - 1 + NEEDED)),
        ("the SPENT H3 pilot match block", H3_PILOT),
        ("D1's SPENT paper reservation", D1_PAPER),
        ("straddles the pilot block's END by exactly one seed", (H3_PILOT[1] - 1, H3_PILOT[1] - 1 + 40)),
    ]
    rejected = 0
    for why, blk in controls:
        c = check(blk)
        ok = not c["clean"]
        rejected += ok
        reason = ("overlap " + str(c["direct"])) if c["direct"] else (
            f"gap {c['nearest_gap']} < floor {c['gap_floor']}" if not c["gap_ok"] else
            f"stream collisions {c['stream_collisions']}" if c["stream_collisions"] else "NOT REJECTED")
        print(f"  {'REJECTED' if ok else '🔴 ACCEPTED':>12}  {why}")
        print(f"                {tuple(blk)}  -> {reason}")
    print()
    print(f"controls rejected : {rejected}/{len(controls)}")
    # 🔑 THE POSITIVE CONTROL: the candidate must still be CLEAN after all of that,
    # or the checks above are rejecting everything and prove nothing either.
    print(f"VERDICT           : {'CLEAN' if r['clean'] and rejected == len(controls) else '🔴 NOT CLEAN'}")
