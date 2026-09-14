# H3 — DESIGN ONLY: a paired-opening replacement for H2

**Status: DESIGN. Nothing is implemented, nothing is frozen, nothing has run, no
seed block is reserved, no gate exists, and no authorization is requested by this
document.** Implementation, the pilot, and any full match are **separate
authorizations**. The push stays held.

**H3 is not H2 continued.** It asks H2's question with a different evidence
structure. §7 forbids pooling its results with H1, either H2 attempt, or the 692
games attempt 3 played.

**AMENDED 2026-09-13, before any pilot card was frozen, from design review. Two
analysis rules were wrong.**

**First amendment — deduplication was defined at the wrong level.** §7 said
"deduplicate by transcript digest", while §5 makes the **pair** the unit.
Dropping one game breaks its pair and leaves a half-observation with no paired
score. §7.1 now defines duplicate **pairs** and keeps every paired score intact:
**a game is never dropped on its own.**

**Second amendment — "whether outcomes vary at all" is withdrawn as a stop
condition.** It would stop the programme on its best possible result: if
independently generated openings give diverse games, few caps, and T1j wins every
pair, that is **strong evidence, not degeneracy**. What must be screened is
evidence made **uninformative** — by repeated play or by caps. Outcome uniformity
is now **reported and never a stop**; §6.3 states the screens.

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
3. **The outcome distribution** — reported, and see §6.3 for what it may and may
   not decide.

### 6.3 What is screened, and what is only reported

⚠ **Outcome uniformity is NOT a stop condition.** A unanimous result is what a
real strength difference looks like. If diverse openings give diverse games with
few caps and one side wins every pair, that is **evidence, not degeneracy**, and
stopping on it would discard the programme's best outcome. It is **reported**;
it decides nothing by itself, and the pilot is too small to weigh it.

**The screens are on evidence made UNINFORMATIVE**, which has exactly two known
causes and one practical one:

| screen | what it catches |
|---|---|
| **duplicate-pair rate** | repeated play — the same pair of games counted again |
| **cap rate** | a rate that measures the 280-ply cap rather than the players |
| **runtime tail** | a full study that cannot be scheduled inside any deadline |

**Numerical thresholds are frozen in the pilot's own card, before play begins.**
The pilot is allowed to end the line on those three; it is not allowed to end it
because the answer came out one-sided.

## 7. Analysis

### 7.1 Duplicates are detected and collapsed AT THE PAIR

**A game is never dropped on its own.** Dropping one game of a pair destroys the
paired score, which is the whole unit of evidence.

* Each game carries a **transcript digest**, frozen structurally as H2's §2.2
  froze it — the move sequence with its movers, never the whole record, or
  `seed`/`task_id`/`rep` would make every repetition look distinct.
* A pair's identity is the **ordered** pair
  `(digest(incumbent-as-red), digest(incumbent-as-black))`, keyed **by role**, so
  it does not depend on which game was played first.
* **Duplicate pairs** — equal pair keys — collapse to **one** observation. The
  surviving pair keeps **both** its games and its paired score intact.
* An **incomplete pair** (one game missing, VOID or unscoreable) is **excluded
  whole** and reported. Never half-counted.
* **Partial overlap** — two distinct pairs sharing a game digest — is **NOT**
  deduplicated: they remain distinct observations. It **is reported**, because
  they are not fully independent and the interval's model should be read knowing
  that. **Two counts, because they differ:** `partial_overlap_pairs` is the number
  of pairs involved in at least one overlap (observations affected, the headline);
  `partial_overlap_relations` is the number of unordered pair-to-pair couples that
  overlap (relationships). Computed among `pairs_distinct`, where each overlap
  shares exactly one game — sharing both would have collapsed as a duplicate, and
  the digest carries movers, so a red-slot game cannot equal a black-slot one.
* **Within-pair identity** — a pair whose two games have equal digests — is
  **flagged**: reversing the colours changed nothing observable, which is a
  degeneracy the pair-level key cannot otherwise show.

### 7.2 What is reported, always

`pairs_nominal` · `pairs_scored` · `pairs_informative` · `pairs_distinct` ·
`duplicate_pairs` · `partial_overlap_pairs` · `partial_overlap_relations` ·
`within_pair_identical` · `capped_games`.

**Effective n is `pairs_distinct`**, and it is reported **beside** the nominal
count every time. A gap between them is a finding, not a footnote.
* **Rates and intervals are over `pairs_distinct`**, under an independence model
  **stated as a model** and called nominal under it — H2's wording, kept
  deliberately. Partial overlap is a known departure from that model and is
  reported alongside.
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
