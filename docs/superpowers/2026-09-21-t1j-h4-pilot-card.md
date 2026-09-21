# H4 PILOT — CARD: does empty-board play produce games worth counting?

**DESIGN ONLY.** No seeds are chosen or reserved, no code is written, no gate
exists, nothing runs. This card exists to be argued with before any of that.

> **The question H4 would ask.** From an empty board, when both models choose
> every move themselves, which model scores better under the frozen
> configurations?

---

## 0. 🔴 READ THIS FIRST: the pilot as proposed would answer a question the existing evidence has already answered

The pilot was specified to measure "how many distinct trajectories the seeds
produce". **Under the frozen argmax configuration that H3 used, the answer is
already known, and it is two.**

Both agents are deterministic under that configuration:

* the incumbent plays `selection_mode: "argmax"` — H3's own records carry it,
  and `opening_temp_plies` / `temp_high` / `temp_low` are recorded as
  `inert_under_argmax`;
* T1j is deterministic — E3a put 25 queries through it, 20 in one JVM and 5 in
  fresh JVMs, and every one returned the same move.

Two deterministic agents and one starting position produce one game. The seed
has nothing to act on.

**This is not inference. H2's evidence measures it directly.** H2 played 692
games under argmax across 16 (opening, colour-arm) groups, 46 games per group,
each game with its own distinct seed:

| grouping | games per group | distinct seeds | distinct transcripts |
|---|---|---|---|
| 15 of the 16 groups | 46 | 46 | **1** |
| `o6_wide_right / t1j_red` | 46 | 46 | 46 — **and all 46 hit the 274-ply cap and never resolved** |

Forty-six different seeds, the same starting position, the identical game
forty-six times. The single group that varied produced variation only inside a
game that never terminated, which is noise rather than diversity.

An empty board is *one* starting position. Under argmax, H4 would therefore
produce:

* Game A (incumbent Red, moving first): **one** trajectory, repeated;
* Game B (T1j Red, moving first): **one** trajectory, repeated;
* 592 games → **2 distinct games**, and a Hoeffding interval over 296 identical
  observations that is arithmetically narrow and scientifically empty.

🔑 **This is the defect H3 already found once.** H3's co-produced opening
generator had no entropy for exactly this reason — argmax plus deterministic
T1j yielded 2 openings where the design wanted 148 — and the freeze caught it.
H4 as specified walks into the same trap from the other direction. The pilot
would spend 40 games rediscovering what 692 recorded games establish.

### What the pilot must actually decide first

Not "how many trajectories?" but **"where does the entropy come from, and is it
legitimate?"** That question has to be settled in this card, before a pilot is
worth running at all.

---

## 1. The entropy problem, and the three honest options

H1 supplies the counter-measurement. It played the same engines under
`selection_mode: "opening_temperature"` (temperature 1.0 for the first 20
plies, 0.1 after):

| configuration | evidence | games | distinct trajectories |
|---|---|---|---|
| `argmax` | H2 attempt 3 | 46 per group, 46 seeds | **1** |
| `opening_temperature` | H1 attempt 2 | 224 | **223** |

So the entropy exists and is available — it just lives in the selection mode,
not in the board.

**Option A — play H4 under `opening_temperature`.** Real diversity, genuinely
seeded, and the pair becomes a meaningful sample.
⚠️ **But it changes the agent.** H3's verdict is about the argmax
configuration; an H4 run under temperature measures a *different incumbent* and
is not comparable to it. The claim must say so, and the decision table in §7
must be read as being about the temperature configuration throughout.

**Option B — keep argmax and accept n = 2.** Then the honest design is two
games, reported as two games, with **no interval at all** — a distribution-free
interval over 296 copies of one observation is a gate that does not bind. This
is a legitimate, cheap experiment; it is simply not a 592-game study.

**Option C — a seeded opening segment, then argmax.** Sample the first *k*
plies at temperature, then switch to argmax for the rest. This keeps the
measured agent closest to H3's while letting the models choose their own
openings, which is nearer the stated intent than either A or B.
⚠️ It reintroduces an opening distribution — one produced *by the models*
rather than uniformly at random — and *k* would need to be frozen in advance
and justified, not tuned.

**Recommendation: Option C, with Option B first as a one-hour sanity run.**
Option B costs two games and establishes the deterministic head-to-head result
outright; if the incumbent wins both, that is already informative and cost
almost nothing. Option C then supplies the population for a real study.
**This choice is not mine to make and is the first thing this card needs
settled.**

---

## 2. What the pilot is for

Given a chosen entropy source, the pilot answers **feasibility only**:

1. **Diversity** — distinct transcripts per arm, and distinct *first* moves.
   The headline number; §0 is why.
2. **Caps and lengths** — empty-board games are longer than six-ply-seeded
   ones. H2's capped games ran to 274 plies. A cap rate that swamps the sample
   makes the primary meaningless, because caps score 0.5.
3. **Runtime distribution** — H3 averaged ~45 s/game from ply 6. From ply 0
   with more search per game, per-game cost could be several times that, which
   drives whether 592 games fit a segmented schedule at all.
4. **Colour dominance** — whether Red/first is so strong that the pair score is
   near-constant at 0.5 regardless of which engine holds it.
5. **Operability from `opening_bound = 0`** — see §3. This is a precondition,
   not a measurement.

🔴 **The pilot may not size the full study from its outcomes.** It reports
feasibility numbers; the sample size comes from the precision target alone
(§6), exactly as H3's 296 did. Choosing N after seeing who won is the defect
the §5.5 optional-stopping rule exists to prevent.

---

## 3. 🔴 T1j at low ply is a QUALIFICATION question, not an assumption

The low-ply qualification **RAN and FAILED**: T1j never enters alpha-beta at
plies 1 and 3; ply 5 searched, and only within 9 `t1j_red` prefixes. H3 began
at ply 6 partly for this reason. **H4 starts at ply 0 and walks straight into
the measured region.**

The instruction that T1j's low-ply behaviour is *part of the tested agent* and
is *recorded, not treated as a generator failure* is the right call, and this
card adopts it. But two distinct things must not be conflated:

* **Behaviour** — T1j not searching at low ply is a property of the opponent
  and belongs in the result.
* **Operability** — whether T1j *returns a legal move at all* from an empty
  board, through the qualified adapter, is a precondition. The low-ply record
  establishes that it does not *search*; it does not establish that it *moves*.

🔑 **Freeze against the qualification record, not against hope.** H3 froze a
schedule that placed T1j where it had been measured unable to search, and that
cost two ranges and two destinations. Before any H4 pilot is authorized there
must be a small, separately-reported operability check: T1j asked for a move
from the empty board and from a handful of two- and four-ply positions,
recording what it returns and by what path. If it cannot move, H4 does not
exist in this form.

⚠️ **And a consequence for interpretation.** In Game B, T1j moves first — at
ply 1, precisely where it was measured not to search. A T1j loss may therefore
be attributable to its unsearched opening play rather than to general weakness,
and **this design cannot separate the two.** That is the mirror of the user's
own reading that a T1j win would implicate our opening selection, and it
belongs in the claim.

---

## 4. The pair, and what it does and does not control

| | |
|---|---|
| Game A | incumbent plays **Red** and moves first; T1j plays Black |
| Game B | T1j plays **Red** and moves first; incumbent plays Black |
| pair score | incumbent's points over the two games ÷ 2 |
| points | win 1, loss 0, **cap 0.5** |
| seeds | fresh, assigned **positionally** — pair *i*, arm *j* → a fixed seed |

🔴 **THE PAIR MEANS SOMETHING DIFFERENT HERE THAN IN H3, AND THE WORD SHOULD
NOT CARRY THE OLD MEANING ACROSS.** H3's pair was one opening played twice with
the colours reversed: two views of the *same position*. H4's pair is two
*different games* that share only their index. What the pairing buys is the
balancing of the first-move advantage within the unit — real and worth having,
TwixT's first player being favoured — but it is **not** the within-position
control H3 had, and the variance reduction will be smaller. Any reuse of H3's
analysis code must not silently import the stronger interpretation.

---

## 5. What must be persisted — H3's omission, corrected

H3 persisted **no moves**: its transcript records carry only `n_plies` and
`opening_bound`, so no H3 game can ever be replayed. H4 fixes that. Every game
must record:

* **every move**, from ply 1 to termination (here the opening is empty, so the
  full game *is* the post-opening continuation);
* **player identity per colour** — which engine held Red, which held Black,
  with the frozen configuration identity of each;
* **viewer-compatible replay JSON**, consumable by `Replay.html` through the
  existing H2 exporter, carrying its own `evidence_note`;
* `seed`, `pair_id`, `arm`, `winner`, `terminal_reason`, `elapsed_s`, `plies`;
* **transcript digest** and the task result, so the record binds to itself.

🔑 The digest must be **recomputed** from the persisted moves during
verification, never read back from its own field.

---

## 6. If the pilot clears: the full study, reusing H3's proven machinery

Unchanged from H3, because the precision target is unchanged:

| | |
|---|---|
| size | **296 paired observations / 592 games** |
| interval | two-sided nominal 95% Hoeffding, half-width **≤ 0.08** |
| caps | score **0.5** in the primary |
| sensitivity | **cap-free** estimate reported beside the primary |
| seeds | a **fresh, collision-proved** block, ACCOUNTED before use |
| execution | segmented, one-shot per segment, supervised restoration |

Directly reusable: the segment runner and its per-segment seed isolation, the
supervised command wrapper with unconditional gate restoration, the create-only
destinations with atomic durable installs, the seed registries and collision
proof, `combine_segments`, the final-state cross-check, and the analysis's
pair-scoring and Hoeffding estimator. **The build should be far smaller than
H3's.** What is genuinely new is the empty-board runner path
(`opening_bound = 0`), full move persistence, and the replay export.

⚠️ **The interval remains nominal under the declared pair-independence model**,
and under Option A or C it is also conditional on the chosen selection mode and
on *k*.

---

## 7. The claim, and what it would and would not support

> Expected paired score when the frozen incumbent and T1j play complete games
> from an empty board, over the preregistered search-seed distribution.

Narrower and more practical than H3's uniform-random-position claim. It is
**still not** a claim about human play, about any other configuration, or about
either engine's behaviour outside this seed distribution — and under Option A
or C it is a claim about the **temperature-seeded** incumbent, not the argmax
one that H3 measured.

The decision table, adopted as stated, with the §3 caveat attached:

| outcome | reading |
|---|---|
| incumbent wins | promote it; T1j is no longer the stronger benchmark |
| T1j wins | our weakness is likely opening selection or early-game planning |
| near parity | H3's advantage is probably position handling and recovery rather than opening play |
| trajectories collapse | **report that directly** — hundreds of repeats of two games are not broad opening evidence, and saying so is the result |

---

## 8. What this card does not do

It reserves no seeds, writes no code, opens no gate and runs nothing. It does
not choose between Options A, B and C — that decision is the precondition for
everything else. It does not establish that T1j can move from an empty board
(§3). And it does not carry H3's verdict across: **H4 is a new study, not a
continuation or a reinterpretation of H3**, and nothing in H3's evidence may be
reused as evidence for it.
