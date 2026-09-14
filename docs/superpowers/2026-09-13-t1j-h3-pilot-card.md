# H3 PILOT — CARD: does a paired, independently generated opening give informative games?

**Status: CARD ONLY. Nothing is implemented, nothing has run, no seed block is
reserved, no gate exists, and no execution is authorized by this document.**
Implementation and execution are **separate authorizations**. The push stays held.

**IT CANNOT SUPPLY A STRENGTH VERDICT.** It computes **no rate, no interval, no
comparison with the incumbent**, and nothing it produces may be cited as evidence
about playing strength. Its only outcomes are: *authorize design work on a full
study*, or *close H3*.

---

## 1. What it is for

H2 failed because **repetition was not evidence**: 46 repetitions of one opening
gave one distinct game in 14 of 15 completed cells. H3 proposes to get diversity
from **independently generated openings** instead. **Nothing tests that
proposal.** This pilot does, and it is deliberately small enough that a
pathological result costs two hours rather than eight.

Three questions, in order of what would kill the design fastest:

1. Do **distinct openings** give **distinct pairs**? (H2 showed repetition does
   not; position variation is **untested**.)
2. Do caps make the outcomes uninformative?
3. What does a game actually **cost**, including the slow tail?

## 2. The opening-generation procedure — FROZEN HERE

**Target population.** Legal, non-terminal TwixT positions reachable from the
empty board at **exactly 6 plies**, on the 24×24 board — 6 because that is the
`opening_bound` every H1/H2 game used, so the pilot's positions sit at the depth
the programme has already played from.

**Procedure**, reproducible and engine-independent:

1. A `numpy` PCG64 generator seeded with the frozen constant **`H3_PILOT_OPENING_SEED = 20260913`**, declared here and **not drawn from any match seed block** — generation is not play.
2. From the empty board, play **6 plies of UNIFORMLY RANDOM LEGAL moves**, using `scripts/GPU/game/board.legal_moves_for_current`.
3. **Reject** and resample the whole position if it is terminal, if either side already holds a winning connection, or if its `d1_selection.canonical_digest` duplicates one already accepted **or** any of H1/H2's eight named openings (`o1_center` … `o8_contact`).
4. Repeat until **20 distinct positions** are accepted.
5. **Pin the accepted set by a digest** before any game is played, and record the full position list.

⚠ **The trade-off, stated rather than hidden.** Uniform random legal play is
**engine-independent** — neither side's preferences touch the selection, which is
the property H2's hand-chosen openings lacked. It is **not realistic**: these are
not positions a strong player would reach. The pilot therefore answers "do varied
positions give varied games", **not** "is the incumbent stronger in realistic
play". A full study may need a different population, and that is design work this
pilot is allowed to inform.

## 3. Size, pairing and seeds

* **20 openings × 2 colour assignments = 40 games**, one game per (opening,
  assignment). **No repetition** — repetition is the thing under suspicion.
* Each opening is played once with our incumbent as **red** and once as
  **black**. The **pair** is the unit (H3 §5).
* **40 seeds**, one per game, bound **positionally**. **No block is reserved
  here**; reserving one is its own authorization, with its own collision re-proof
  against the registries as they stand.
* The incumbent configuration is **H2's frozen one, unchanged**: visit-count
  argmax, 400 sims, board 24, `calib020_0001`.

## 4. Runtime limit — derived, not assumed

From attempt 3's own 692 games: **non-cap ≈ 32 s/game**, **capped ≈ 180 s/game**
(692 games in 28,800 s with 46 caps). H2's deadline was *assumed* and a cap
cluster exhausted it; this one is **sized so the pathological case still
completes**:

| | |
|---|---|
| expected (6.6 % caps) | ≈ 1,700 s (~28 min) |
| **worst case — every game caps** | 40 × 180 s = **7,200 s** |
| **whole-run deadline** | **7,200 s (2 h)** |
| per-call bound | 120 s, inherited |

Because the worst case fits, the pilot **yields a measurement whatever happens**.
That is the point: a pilot that can VOID on its own deadline measures nothing.

🔴 **Per-game duration MUST be recorded.** Attempt 3's `task_result` carries
`plies`, `seed`, `winner`, `terminal_reason`, `t1j_points` — **and no timing at
all**. "Runtime including the slow tail" cannot be reported from records like
that. The pilot's runner adds a per-game elapsed field; without it §6 is
unanswerable.

## 5. What it reports

Per H3 §7.2, plus the timing this pilot exists to gather:

`pairs_nominal` (20) · `pairs_scored` · **`pairs_distinct`** · `duplicate_pairs` ·
`partial_overlap_pairs` · `within_pair_identical` · `capped_games` ·
per-game elapsed **min / median / p90 / max** · total elapsed ·
the **outcome distribution over pairs**.

## 6. STOP RULES — numerical, frozen before play begins

Evaluated **after** the run, on the recorded numbers, in this order:

| # | rule | if it fires |
|---|---|---|
| **S1** | `pairs_distinct < 18` of 20 (**> 10 % duplicate pairs**) | **CLOSE H3.** Independently generated openings do not give distinct games either, and the diversity problem is not repetition-specific. |
| **S2** | `capped_games > 8` of 40 (**> 20 %**) | **CLOSE H3 in this configuration.** Above this a rate measures the cap. H2's overall cap rate was 6.6 %, so 20 % is a threefold worsening, not a marginal one. |
| **S3** | `within_pair_identical ≥ 1` | **CLOSE H3.** Reversing the colours changed nothing observable, so the pairing carries no information. |
| **S4** | p90 per-game elapsed **> 180 s** | **NO full study is scheduled** on this runtime. Report and re-scope: the tail, not the mean, is what sets a deadline. |

**If none fires:** the pilot **authorizes design work on a full study** — and
nothing more. It does not authorize a match, a seed block, or a size.

⚠ **Not a stop rule, by explicit decision:** the outcome distribution. A
one-sided result with high diversity and few caps is what a real strength
difference looks like; stopping there would discard the best available outcome.
It is reported (§5) and it decides nothing. At 20 pairs it could not, in any case.

## 7. What this card does not do

It reserves no seeds, opens no gate, fixes no full-study size, and asks for no
execution. It fixes the **procedure, the population, the size, the runtime limit
and the stop rules** — before play — because fixing them afterwards is how a
pilot becomes a result hunt.
