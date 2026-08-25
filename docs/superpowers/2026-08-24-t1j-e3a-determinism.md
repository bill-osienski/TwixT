# E3a — T1j determinism: RAN, PASSED

**Date:** 2026-08-24 · **Status:** E3a **RAN and PASSED.** Comparator gate exit **0**.
All 25 queries returned the **identical move**. Stopped after E3a as authorized.
· **E3b and E4 remain unauthorized.**

Basis: `main` @ `0c808b6`, clean, local == remote.
Predecessor: [`2026-08-24-t1j-e2-attempt4.md`](2026-08-24-t1j-e2-attempt4.md) (E2 PASSED).
Evidence: `evidence/2026-08-24-t1j-e3a/` — execution transcript, all six process stdouts, the
collected records, the probe source, the runtime comparator, the driver, exit statuses and stderr
counts (`50`–`64`), plus a post-run per-process structure checker and its run (`65`, `66`).

All five E1/E2 evidence directories are **byte-unchanged**.

---

## Result

**Every one of 25 queries returned `x=15, y=15`.** Zero distinct alternatives.

| gate requirement | result |
|---|---|
| Exactly 25 completed records | **PASS** — 20 shared + 5 fresh |
| Every query returns the identical numeric move | **PASS** — `(15,15)`, one distinct value |
| Every search completes fixed ply 3 | **PASS** — `currentMaxPly=4` on all 25 |
| Every position passes per-move coherence | **PASS** — `coherent=true` on all 25 |
| Zero `Window`/`Frame` at each process end | **PASS** — 0/0 in all six processes |
| Headless and preference checks | **PASS** — all six processes |
| No unexpected exception or unauthorized reflection | **PASS** — 0 failures, 0 bytes of stderr anywhere |

Reflection counts scaled exactly as the contract requires: the shared JVM recorded **20 writes /
40 reads**, each fresh JVM **1 write / 2 reads**, and every access was one of the three authorized
fields.

## Design

- **20 queries in one JVM.** Global tables (`CheckPattern.loadPattern()`, `Zobrist.initialize()`)
  initialized **once**; then a **fresh `Match` constructed and fully validated for every query** —
  boards cleared, resized and re-set-up, `nextPlayer` written, six alternating moves submitted
  through `setlastMove()`, and all six coherence axes checked per move before each search.
- **5 queries in five fresh JVMs**, one query each, each initializing once.
- Same pinned artifacts, same fixed six-move position, same fixed ply 3, same reflection contract
  as E2 attempt 4.
- **T1j's randomness was neither seeded, patched nor replaced.** The randomized first-move path is
  simply not reached: the fixture starts at `moveNr = 6`, past `InitialMoves`' range.

The identity gate **compared** rather than recorded — clone, JAR, JDK archive, and all four
extracted-JVM components against the values bound in E2 attempt 4, with a mismatch aborting the
run. That is the defect review caught in attempt 4, fixed here by construction.

## The comparator self-tests before it judges

It refuses to evaluate real data unless it first rejects every injected defect. All nine passed:

| injected defect | rejected on |
|---|---|
| changed move | moves not identical across queries |
| missing record | expected 25 records, got 24 |
| wrong group count | `{'fresh': 6, 'shared': 19}` != expected |
| wrong ply | `currentMaxPly 3 != 4` |
| failed postcondition | `windows=1 != 0` |
| wrong reflection count | `19/40 != 20/40` |
| incoherent position | `position not coherent` |
| missing `POSTCOND` | no POSTCOND records |
| baseline | accepted (the control — a comparator that rejects everything proves nothing) |

> **Gap found in review round 1, closed post-run.** The runtime comparator validated the **pooled**
> records and every `POSTCOND` it found, but **never required how many processes reported** — five
> valid `POSTCOND`s would have passed it. A post-run checker
> (`65_post_run_process_checker.py.txt`, run recorded in `66`) binds the process structure instead:
> the **exact file set**, **per-file record counts** (20 + 1×5), **query identities** (1..20 and 1),
> and **exactly one `POSTCOND` per file**, plus every condition the comparator already applied. It
> self-tests over nine cases including negative controls for a **missing process file**, a
> **duplicated/extra process file**, a **missing postcondition**, a **duplicated postcondition** and
> a **unanimous null sentinel `(-1,-1)`** — that last one added in review round 2, because the
> checker had omitted the comparator's sentinel rejection while this card claimed it reapplied
> every condition. A unanimous `(-1,-1)` is identical across processes but is not a move.
> **Result: PASS**, exit 0 — six processes, correct counts and identities throughout. E3a was **not
> rerun** and artifacts `50`–`64` are unchanged.

## Timing — descriptive only, never a gate

`elapsed_ms` min **10**, median **11**, max **37**. The shared JVM's first query took 33 ms and
settled to 10–11 ms by the tenth, while each cold fresh JVM took 32–37 ms — a shape **consistent
with JVM/JIT warm-up and cold-process overhead**, though timing alone does not isolate the cause.
**No timing value gates anything**, and none was used in the comparison.

## An observation worth recording — bounded

`Zobrist.initialize()` builds its hash table from an **unseeded `new Random()`**
(`Zobrist.java:31`). Each of the five fresh JVMs therefore **independently initialized an unseeded
table**, and all five returned the same move as each other and as the shared JVM.

> **Corrected in review round 1.** This first said the five JVMs "searched with five different
> Zobrist keyings" and that the result was shown "insensitive to the salt". **Neither is
> supported.** The probe never serialized or compared any Zobrist value, so **the keyings were not
> observed to differ** — independent unseeded initialization makes difference likely, not proven —
> and **nothing measured salt-insensitivity.** What the evidence shows is that five processes that
> each independently initialized an unseeded table agreed on the move.

---

## What this establishes, and what it does not

**Established, under the qualified runtime:** for **this one fixed position at fixed ply 3**, T1j's
move is **stable across 20 repetitions within a process and across five independent processes**,
with every position rebuilt and revalidated from scratch each time, and with each fresh process
**independently initializing an unseeded Zobrist table**.

**Not established, and not claimed:**

- **Determinism in general.** One position, one ply. Nothing here speaks to other positions, other
  plies, other board sizes, time-based search (`mdFixedPly=false`), or other machines and JVM
  builds. The randomized `InitialMoves` path was **bypassed, not disproved** — a driver that starts
  from an empty board would still meet it.
- **Strength.** No opponent, no game, no score.
- **Rules equivalence with our engine.** No cross-check against `TwixtState`; that is E3b's
  question and it is unauthorized.
- **Anything about repeated runs of *this* experiment.** E3a ran once.

## Where the ladder stands

| gate | question | status |
|---|---|---|
| E0 | is there a candidate? | T1j proposed |
| E1 | is the artifact identified? | official-release-qualified |
| E2 | can it be driven headlessly to a move? | **PASSED** — legal move, 4 attempts |
| **E3a** | **is the move stable?** | **PASSED — 25/25 identical** |
| E3b | does its state match ours? | **unauthorized** |
| E4 | is it in a usable strength band? | **unauthorized** |

E3a was the cheap half of E3 and it came back clean, so the adapter investment is not yet spent
against an engine that wanders. **The uncalibrated-anchor caveat from E0 still stands unchanged:**
even a fully qualified T1j yields an ordering, not a placement.
