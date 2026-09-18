# H3 FULL STUDY — MATCH SEED REGISTRATION, 2026-09-18

**`[202626000, 202626592)` — 592 seeds, one per game. ACCOUNTED only.**

Nothing was drawn. A reservation is not a draw, and registering opened nothing:
all ten gates and the population-freeze barrier are `False`.

| file | what it is |
|---|---|
| `01_collision_proof_v14.py` | the proof, run before the registry was touched |
| `02_collision_proof_v14_run.txt` | its output, verbatim |
| `03_preflight_pre_commit.txt` | the preflight at the moment of registration |

## What v14 proves

| check | result |
|---|---|
| prior seeds / values | **302,357 / 1,511,785** with derivations, exhaustive |
| direct overlap | **NONE**, across eleven categories |
| derived-stream collisions | **0** |
| own derivations injective | **2,960 = 592 × 5** |
| nearest actual boundary | **1,960** against a gap floor of **592** (its own size) |
| negative controls rejected | **15 / 15** |
| verdict | **CLEAN** |

## 🔴 Four generation ranges, and no registry holds one of them

```
[20261000000, 20261059200)   the 148-opening uniform range, RETIRED UNUSED
[20261200000, 20261259200)   co-produced attempt 1, SPENT WHOLE (VOID)
[20261400000, 20261459200)   co-produced attempt 2, SPENT WHOLE (VOID)
[20261600000, 20261718400)   LIVE — it produced the population frozen 2026-09-17
```

v14 adds all four by hand, and **three of its fifteen controls sit inside them**
so the addition is proved to matter: a registry-only enumeration would have
called an overlapping block clean.

## The controls that carry the weight

A check that never rejects proves nothing, so the boundary cases are the point:

* a block **straddling the pilot's end by one seed** → rejects on overlap;
* one **starting exactly where the pilot ends** — no overlap at all → rejects on
  gap 0;
* one **a single seed inside the floor** → rejects on gap 591 < 592.

The candidate is excluded **by identity** in the overlap check *and* the gap
check. Without the second, a block re-proved after registration measures its
distance to itself as zero and fails its own floor.

## The binding

Seeds bind **positionally**, row *i* → `lo + i`, across four contiguous
148-seed quarters:

```
segment 0  [202626000, 202626148)      segment 2  [202626296, 202626444)
segment 1  [202626148, 202626296)      segment 3  [202626444, 202626592)
```

A pair's two arms are **adjacent and share a segment**, so no pair can be split
across a VOIDed one.

## The schedule pins — and the check they replaced

```
schedule  e374a95f885656ea27caa88359437e4c035de0e0e0c027cf861ce5e613362350
segment 0 cde6d04ce52e07088e437754e1ee60b750542de08c6b6e9fbc4744bc2c795d31
segment 1 14aaefb3a01cc1d08bdfd7f0c4c21599414d0dc68fee8f51584e1e4a50660a41
segment 2 1764471c5b95042726b9a3bbeca92309ff703d734d73f448adf2f548d96a1855
segment 3 44576a71042ff6b7c9118d3e36f18d62d1d0dfc4d01aeb7eec318d0a4b8753f0
```

🔴 **`run_segment` used to pass `want_digest=segment_digest(tasks, segment)`** —
the digest computed from the very tasks it then handed to the checker. One
source, two sides, so they agreed unconditionally: **a check that existed and did
not bind.** The pins give the comparison a second, independent side.

## Scope

This registered a block. It authorizes **nothing**: `H3_STUDY_EXECUTION_AUTHORIZED`
is `False`, segment outputs are absent, and running segment 0 and pushing remain
unauthorized.
