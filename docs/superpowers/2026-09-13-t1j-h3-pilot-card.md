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

## 4. Runtime limit — a CHOSEN limit, and the pilot may time out

🔴 **CORRECTED 2026-09-13. An earlier version of this section called 7,200 s a
derived worst-case bound. It is neither derived nor a bound**, and the error is
worth keeping visible because measuring runtime is one of this pilot's purposes.

### 4.1 What attempt 3's records actually support

**Neither `03_h2_results.jsonl` nor `04_h2_trace.jsonl` carries a timestamp of any
kind.** `task_result` holds `plies`, `seed`, `winner`, `terminal_reason`,
`t1j_points`, `agents_built`, `task_id`; the trace holds `event`, `index`,
`games_completed`. **There are no per-game durations to average.**

What is actually known:

| fact | source |
|---|---|
| 692 completed games in 28,800 s wall clock → **mean ≈ 41.6 s/game** | the run's own start/end and its deadline trip |
| 46 of those games hit the 280-ply cap | the records |
| plies: mean 59, median 42, max 280 | the records |

The earlier "non-cap ≈ 32 s, capped ≈ 180 s" split was **not** derived from those
records. The 180 s came from **interval polling of the trace while the run was in
progress** — 2–4 `task_done` events per 540 s window during the all-cap stretch,
so roughly 135–270 s per game — and the 32 s was then obtained by solving the
total *against that assumption*. That is a coarse live observation, not a
measurement, and presenting it as a derivation overstated it.

### 4.2 Why 40 × 180 is not a worst case either

An average is not a maximum. A single game is bounded only by
`PLY_CAP = 280` moves at `PER_CALL_TIMEOUT_S = 120 s` each — **hours for one
game**. No product of an observed average and a game count bounds anything.

### 4.3 The limit, and what a timeout means

| | |
|---|---|
| **whole-run limit** | **7,200 s (2 h) — CHOSEN, not derived** |
| basis | ≈ 4× the 1,700 s the observed mean suggests for 40 games; small enough that a pathological result costs two hours rather than eight |
| per-call bound | 120 s, inherited |
| **may it time out?** | **YES.** This is a real possibility, not a residual one. |

**A timeout still measures what the COMPLETED games cost.** That much is a real
measurement, and in that sense a timeout is informative. **It does NOT establish
the full schedule's cap rate or its runtime tail** — those are properties of all
40 games, and a timeout leaves some unplayed. §4.4 fixes the line exactly.

### 4.4 What a partial run may and may not report

🔴 **CORRECTED 2026-09-13, a second time.** The previous version said the
unfinished games "are precisely the slow ones", so the observed cap rate and p90
**understate** the truth, and that p90 is therefore a **lower bound**. **All of
that is wrong.** Unstarted games have **unknown** durations — they were never
reached, not observed to be slow. Only the single in-flight game at the timeout is
known to have been cut off. And **a quantile is not monotone under adding
observations**: the remaining games could be mostly fast and pull the full-sample
p90 **below** the truncated one. Nor is the truncation random — the schedule is
played in order, and H2's caps arrived in one contiguous cell — so the completed
games are not even an unbiased subsample.

**THE RULE THAT DECIDES EVERY CASE: a monotone COUNT may fire; a RATIO or a
QUANTILE may not.** An event already observed cannot be un-observed by playing
more games, so a count that has already crossed its threshold has crossed it for
the full schedule too. A ratio or a quantile can move in either direction.

| rule | quantity | on a timeout |
|---|---|---|
| **S1** `duplicate_pairs > 2` | monotone count | **MAY FIRE** conclusively once exceeded |
| **S2** `capped_games > 8` | monotone count | **MAY FIRE** conclusively once exceeded |
| **S3** `within_pair_identical > 2` | monotone count | **MAY FIRE** conclusively once exceeded |
| **S4a** `total elapsed > 3,600 s` | monotone, directly observed | **MAY FIRE** — trivially, since a timeout means 7,200 s was reached |
| **S4b** `p90 > 4 × median` | ratio of two quantiles | **UNRESOLVED.** Not evaluated unless the full schedule completes. |

**MAY**, over completed pairs only, with the completed count reported beside
`pairs_nominal`:

* every §5 count and the per-game timing statistics, **every one labelled
  PARTIAL** — they describe the games that finished and nothing beyond them;
* **S1, S2, S3 and S4a**, firing on absolute counts that already exceed their
  thresholds.

**MAY NOT:**

* **declare any rule CLEAR.** A count below its threshold may still cross it in
  the games that were never played. Firing is conclusive; not firing is not.
* **evaluate S4b at all.**
* **state a cap RATE or a runtime TAIL for the full schedule.** The cap *count*
  may fire S2; the *rate* over 40 games is not established by fewer than 40.
* **call any observed timing a bound**, in either direction.

**Floor.** Below **10 completed pairs** the run reports *"ran out of time"*, the
completed count, and the observed timings labelled partial — and **no stop-rule
evaluation at all** except S4a, which the timeout itself decides.

🔴 **Per-game elapsed MUST be recorded by the pilot's runner.** Attempt 3's
records make §4.1 unanswerable after the fact, and that gap is why this section
had to be corrected at all. Without a per-game duration field the pilot cannot
answer its own third question.

## 5. What it reports

Per H3 §7.2, plus the timing this pilot exists to gather:

`pairs_nominal` (20) · `pairs_scored` · `pairs_informative` ·
**`pairs_distinct`** · `duplicate_pairs` · `partial_overlap_pairs` ·
`within_pair_identical` · `capped_games` · per-game elapsed
**min / median / p90 / max** · total elapsed · the **outcome distribution over
pairs** · and, on a truncated run, the completed-pair count beside
`pairs_nominal`, with **every count and timing labelled PARTIAL** and no timing
called a bound in either direction (§4.4).

## 6. STOP RULES — numerical, frozen before play begins

Evaluated **after** the run, on the recorded numbers, in this order. **On a
truncated run §4.4 governs**: S1, S2, S3 and S4a may fire on absolute counts that
already exceed their thresholds, **S4b is not evaluated at all**, and **no rule
may be declared clear**.

| # | rule | if it fires |
|---|---|---|
| **S1** | `duplicate_pairs > 2` of 20 (**> 10 %**) | **CLOSE H3.** Independently generated openings do not give distinct games either, and the diversity problem is not repetition-specific. ⚠ Keyed on `duplicate_pairs`, **not** on `pairs_distinct` — the latter also falls when pairs are excluded as incomplete or within-pair-identical, and S1 must fire for duplication alone. |
| **S2** | `capped_games > 8` of 40 (**> 20 %**) | **CLOSE H3 in this configuration.** Above this a rate measures the cap. H2's overall cap rate was 6.6 %, so 20 % is a threefold worsening, not a marginal one. |
| **S3** | `within_pair_identical > 2` of 20 (**> 10 %**) | **CLOSE H3.** The pairing itself carries no information often enough to matter. Below the threshold, each affected pair is **excluded whole** (§6.1) and the count reported — it is not a closure. |
| **S4a** | **total elapsed > 3,600 s** for the 40 games | **NO full study is scheduled** on this runtime without re-scoping. Double the ≈1,700 s that attempt 3's observed mean suggests for 40 games — the one timing figure its records do support. |
| **S4b** | **p90 per-game > 4 × median per-game** | **NO full study is scheduled** without re-scoping. A heavy tail, not a slow mean, is what exhausted H2's deadline: 46 cap games clustered in one cell consumed it while the median game stayed fast. Stated as a SHAPE ratio because the pilot measures its own median; no absolute per-game threshold is available, the earlier 180 s figure having been withdrawn (§4.1). |

### 6.1 Within-pair identity: exclude the pair, do not close on one

🔴 **CORRECTED 2026-09-13.** S3 read `within_pair_identical ≥ 1` → close. That was
wrong: **one identical colour-reversed pair shows that THAT PAIR contributed no
discrimination. It says nothing about the other nineteen**, and closing a line on
a single position is not a justified inference.

Two separate things, fixed in advance:

1. **EXCLUSION, always.** A pair whose two games have equal transcript digests is
   **excluded whole** from the evidence — consistent with H3 §7.1, where a pair is
   kept or removed whole and **a game is never dropped on its own**. It carries no
   discrimination, so it is not counted as evidence. Its count is always reported.
2. **CLOSURE, only above a threshold.** More than **2 of 20 (> 10 %)** closes H3,
   matching S1's tolerance for duplicate pairs — the two screens ask the same
   question (is an observation a fresh one?) and there is no reason to hold them
   to different standards. Below it: report, exclude, continue.

**Order of operations**, so the counts are legible and nothing is excluded twice:

    pairs_nominal (20)
      → drop incomplete pairs            → pairs_scored
      → drop within-pair-identical pairs → pairs_informative
      → collapse duplicate pairs         → pairs_distinct   (= effective n)

Each step's count is reported separately. S1 is evaluated on `duplicate_pairs`
found at the third step, among `pairs_informative`; S3 on `within_pair_identical`
found at the second. Neither is read off `pairs_distinct`, which moves for all
three reasons at once.

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
