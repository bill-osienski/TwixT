# E4 preflight — RAN, PASSED. No games.

**Date:** 2026-08-25 · **Status:** the E4 preflight **RAN and PASSED.** Driver gate exit **0** —
unit tests 0, sweep 0, determinism 0, freeze 0, preference controls 0, preference pair 0.
**No game was played, no seed consumed, no reference model loaded, no T1j source modified.**
· **The endpoint screen remains UNAUTHORIZED.**

Basis: `main` @ `ef81bbd`. Evidence: `evidence/2026-08-25-t1j-e4-preflight/` — 20 files plus a
**self-excluding manifest**. All nine prior evidence directories are **byte-unchanged**, and
`E3bDump.java` is asserted identical to its E3b attempt-3 bytes (`281664cf…`).

---

## The two corrections, bound

**Weak endpoint is `mdPly = 3`** — the deepening loop is `for (currentMaxPly = 3; currentMaxPly <=
maxPly; …)`, so below 3 the body never executes and no search runs at all.

**`mdPly = 4` is usable as a requested terminal depth.** The guard is
`if (currentMaxPly != 4 || currentMaxPly == maxPly)`, so depth 4 is skipped only as an *intermediate*
iteration. Executed ladders, recorded in the preregistration:

| requested `mdPly` | depths actually executed |
|---|---|
| 3 | {3} |
| 4 | {3, 4} |
| 5 | {3, 5} |
| 6 | {3, 5, 6} |
| 7 | {3, 5, 6, 7} |

Under `mdFixedPly = true` the loop's only `break` is guarded by `maxTime > 0` and `maxTime` is `−1`,
so it always runs to completion and exits at `maxPly + 1`. That gives an exact completion predicate:
**depth `d` completed ⟺ `usealphabeta` and `currentMaxPly == d + 1`** — using only the two fields E2
and E3a already qualified. No new reflection.

## Preregistration, frozen before measurement

`08_preregistration.json`, sha256 **`9bddafe3…`**, hashed into the transcript **before the first
query**. Every threshold is read from that file by the measuring code; none is inlined. Frozen:
per-query timeout **120 s**, sweep budget **3600 s**, determinism budget **900 s**,
`max_query_ms` **30000**, the selection rule, and the three dial-response classes.

## Positions — six, both regimes, both sides to move

`Evaluation.evaluatePosition` zeroes one side's value while `moveNr < 8`, so the preflight covers
both regimes. Every position is ≥ 6 plies, which is what keeps T1j's **unseeded**
`InitialMoves.firstMove()` unreached, and every one was verified legal and non-terminal in our engine
**before T1j was driven**, then bound ply-by-ply through the E3b adapter at `ply_cap = 280` with
**abort on the first divergence**. No divergence occurred.

| position | plies | to move | regime | bridges |
|---|---|---|---|---|
| `p06_e3a_center` | 6 | red | early (`moveNr<8`) | 1 |
| `p07_offset` | 7 | black | early | 1 |
| `p08_spread` | 8 | red | normal | 0 |
| `p09_contact` | 9 | black | normal | 4 |
| `p12_two_wings` | 12 | red | normal | 6 |
| `p14_bridged` | 14 | red | normal | 6 |

## The dial responds

**30 queries, 6 positions × 5 depths, every one completing its requested depth.** Classification:
**`observable_move_response`** — 14 (position, depth) pairs return a different move from that
position's depth-3 move, across **5 of the 6 positions**. `p14_bridged` returned `(20,12)` at every
depth.

> Recorded **descriptively**, as required. A changed move shows the setting reaches the search and
> alters its output. It is **not** evidence of strength monotonicity, and none is claimed.

## Cost, and the strong endpoint

Selection is **by cost only** — the rule takes no move argument, and a self-test asserts its
signature to keep it that way.

| depth | max engine ms | verdict |
|---|---:|---|
| 3 | 121 | qualifies |
| 4 | 203 | qualifies |
| 5 | 507 | qualifies |
| 6 | **2749** | **qualifies — selected** |
| 7 | 32055 | **rejected**: over the frozen `max_query_ms` of 30000 |

**Weak `mdPly = 3`, strong `mdPly = 6`.** The endpoints do not coincide. Depth 7 missed the frozen
threshold by 2.1 s — the preregistered number bit, rather than rubber-stamping the deepest option.

**Cost is dominated by position structure, not ply count.** At depth 7, the 8-ply `p08_spread` took
**32.1 s** while the 14-ply `p14_bridged` took **0.22 s** — 145× apart. Any model of "later position
⇒ slower" is contradicted by this data.

## Determinism at the selected endpoint

The E3a structure at `mdPly = 6`: **20 independent constructions in one JVM plus 5 fresh JVMs, 25
queries**, every position rebuilt and revalidated. **All 25 returned the identical move `(14,11)` at
identical completed depth 6**, all legal in T1j and in our engine, in 15.0 s of the 900 s budget.

Full suite at this tree: **2911 passed, 4 skipped, 0 failed** (`19_full_suite.txt`).

## Frozen, and deliberately not executed

`06_endpoint_screen_plan.json` — **8 balanced openings × 2 colour arms = 16 tasks**, each opening
6 plies (again to keep the unseeded path unreached) and verified legal here; deterministic task
identities; seed block **`[202612000, 202612512)` RESERVED and UNSPENT**, asserted disjoint from all
eight prior intervals; abort rules; an append-only JSONL result format; and stopping rules with band
`[0.05, 0.95]`, a saturation early-stop checked every 4 tasks, and **`INCONCLUSIVE` as a first-class
outcome** — a larger match only if the screen lands `IN_BAND` or is genuinely `INCONCLUSIVE`, never
after saturation.

## Runtime estimate — and why it is weak

The screen's cost is **not** safely extrapolable from this preflight. Measured per-move engine time
at `mdPly = 6` spans 94 ms – 2749 ms across six positions of 6–14 plies; a game runs far longer, and
the one clear trend is that **structure, not length, drives cost**.

Taking the measured span at face value — ~60 T1j moves per game × 16 games, T1j's side alone:

| per-move basis (depth 6) | value | screen total |
|---|---:|---:|
| minimum observed | 94 ms | **1.5 min** |
| median observed | 609 ms | **9.7 min** |
| maximum observed | 2749 ms | **44 min** |

Treat **44 min** as the planning number and the whole range as unvalidated: no position beyond 14
plies was measured at any depth, and the spread within a single depth is 29×.

---

## What this establishes, and what it does not

**Established:** `mdPly` reaches the search and changes its output on 5 of 6 frozen positions; every
requested depth 3–7 completed on every position; the cost curve is measured and steep, with depth 7
already past a preregistered ceiling; `mdPly = 6` is deterministic across 25 queries and 6 processes;
and the whole endpoint screen is frozen without a game being played.

**Not established, and not claimed:**

- **Strength monotonicity.** A changed move is a *response*, not an improvement. Timing says nothing
  about strength either.
- **That depth 6 is stronger than depth 3.** Nothing here compares outcomes.
- **Determinism beyond one position at one depth** — the determinism stage is `p06_e3a_center` at
  `mdPly = 6` only, and the Zobrist salt remains unseeded per process.
- **Cost at game length.** Nothing beyond 14 plies was measured.
- **Absolute placement.** The E0 caveat stands untouched: T1j is uncalibrated, so even a passing
  screen would yield an **ordering**, not a placement.

## Where the ladder stands

| gate | question | status |
|---|---|---|
| E3a | is the move stable? | PASSED — 25/25 at ply 3 |
| E3b | do its rules and state match ours? | PASSED (attempt 3 + correction) |
| **E4 preflight** | **is there a usable dial, and what does it cost?** | **PASSED — dial responds; endpoints 3 and 6** |
| E4 endpoint screen | is T1j in a usable strength band? | **unauthorized** |
