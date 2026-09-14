# H3 — DESIGN ONLY: a paired-opening replacement for H2

**Status: DESIGN. Nothing is implemented, nothing is frozen, nothing has run, no
seed block is reserved, no gate exists, and no authorization is requested by this
document.** Implementation, the pilot, and any full match are **separate
authorizations**. The push stays held.

**H3 is not H2 continued.** It asks H2's question with a different evidence
structure. §7 forbids pooling its results with H1, either H2 attempt, or the 692
games attempt 3 played.

---

## 1. The question

Unchanged from H2, and inheriting its caveat:

> **PRIMARY QUESTION.** With the incumbent playing **visit-count argmax** after a
> fixed opening, is T1j stronger than our incumbent **in that configuration**,
> under otherwise matched conditions?

⚠ "Argmax" is not "the strongest deterministic use", and this card does not claim
it is. `T1J_STRONGER` means stronger than **this frozen configuration**.

---

## 2. What H2 attempt 3 established — and what it did not

It **VOIDED** on its 28,800 s deadline at game 692 of 736. **No strength verdict
exists and none is derived here.**

**The decisive fact does not depend on the VOID.** In **14 of the 15 completed
cells**, 46 repetitions produced **one distinct transcript** — against the
design's own floor of 42 of 46. Those cells are *complete*; finishing the
remaining 44 games could not have changed them. The repetition scheme failed its
diversity requirement on the evidence in hand.

⚠ **What that does NOT establish.** Deterministic move selection does not by
itself force identical repetitions. Argmax removes the **readout's** randomness,
not the **search's**, and the design expected **seeded search variation** to
differentiate games within a cell. The honest statement is narrower: *that
variation was insufficient in this configuration.* **Why is not known**, and this
design does not assume an answer — §6 measures it instead.

**Three consequences carried forward:**

1. **Repetition was not evidence.** 736 nominal games carried ~61 distinct ones.
   A Hoeffding interval computed on n = 736 described a sample that did not exist.
2. **Distinctness is necessary, not sufficient.** The one cell that cleared the
   floor — 46 distinct transcripts — hit the 280-ply cap in **every** game. Under
   the inherited rule a cap scores **0.5**, so that cell contributed 46 draws.
   Different moves gave no useful variation in **outcomes**.
3. **A degenerate cell does not merely add nothing — it pulls toward the primary
   comparison value.** The primary question is parity at 0.50, and repeated
   identical results and repeated caps both accumulate mass there. A design that
   counts them as fresh evidence can manufacture a parity result.

---

## 3. The shape of the replacement

**Diversity comes from the POSITIONS, not from repetition.**

| | H2 | **H3** |
|---|---|---|
| diversity source | 46 repetitions per opening | many **distinct openings** |
| openings | 8, hand-chosen | broad, **independently generated** |
| unit of evidence | the game | the **colour-reversed pair** |
| size fixed | in the card, before any measurement | **by a pilot**, from measured diversity and runtime |
| repeated identical games | counted as independent | **deduplicated before any rate** |

---

## 4. The opening set

* **Independently generated**, by a frozen, reproducible procedure — **not chosen
  by either engine and not hand-picked**, so neither side is favoured by the
  selection.
* Every position **legal, reachable, non-terminal**, and at a fixed ply depth
  recorded in the plan.
* **Deduplicated by canonical position digest** — reuse `d1_selection.
  canonical_digest`, which already freezes "two move orders, one position".
* The whole set is **pinned by a digest** and generated **before** any game.
* **Size is not fixed here.** §6 fixes it.

## 5. Colour reversal, and why it is the unit

Each opening is played **twice: once with our incumbent as red, once as black**,
from the *same* position.

* A single opening may favour one colour outright. Playing both ways puts that
  bias inside the pair instead of inside the result.
* **The pair is the independent unit** for §7. The two games in a pair share an
  opening and are **not** independent of each other.
* A pair yields a paired score; the analysis is over pairs, never over games.

## 6. The pilot — small, separate, and it can stop the programme

**Separately scoped and separately authorized.** Its size, seeds and deadline
belong to its own card; none are reserved here.

**It is not a strength measurement and produces no verdict.** It exists to
measure three things the full design cannot assume:

1. **Diversity across DISTINCT openings.** Do different openings give different
   games? H2 showed repetition does not. Whether *position* variation does is
   **untested**.
2. **Runtime, including the tail.** Per-game time, and the **cap rate** — H2's
   cap games ran ~6× the 51-ply mean and are what exhausted its deadline. A
   deadline must be **derived from measured runtime with the tail included**,
   never assumed.
3. **Whether outcomes vary at all**, not merely transcripts — the lesson of the
   cap cell.

**Declared STOP conditions, frozen before the pilot runs.** If distinct openings
still yield degenerate games, or caps dominate, **no full match is scheduled**
and H3 closes. The pilot is allowed to end the line; that is its point.

## 7. Analysis

* **Deduplicate by transcript digest before any rate is computed.** Identical
  games contribute **once**.
* Report **effective n (distinct) beside nominal n**, always. A gap between them
  is a finding, not a footnote.
* **Rates and intervals are over PAIRS**, under an independence model **stated as
  a model** and called nominal under it — H2's wording, kept deliberately.
* **Caps are declared in advance**: scored 0.5 under the inherited rule, *and*
  with a pre-set threshold above which the run reports **no rate**, because a
  rate over mostly-unresolved games measures the cap.
* **A per-pair degeneracy screen**, inherited in spirit from H2's per-cell one —
  a global rule passes a wholly collapsed subset.
* **No pooling** with H1, either H2 attempt, or attempt 3's 692 games.

## 8. Deliberately NOT fixed here

Full match size · the deadline · any seed block · the generation procedure's
parameters · the pilot's own size. Each is set by measurement or by its own
card, and **fixing them here is what H2 did wrong**.

## 9. What the H2 line did leave behind

The **authorized launch path is proven end to end**: pre-run verification 49/49,
a one-line gate, a supervised worker, deadline enforcement that fired correctly,
**gate restoration by the wrapper's `finally` rather than by hand** (exit 3, not
10), and no surviving worker, supervisor or JVM. H3 reuses it unchanged.

⚠ **Still untested: a full match reporting successfully.** No run has yet reached
the frozen reporting rules with a complete schedule.
