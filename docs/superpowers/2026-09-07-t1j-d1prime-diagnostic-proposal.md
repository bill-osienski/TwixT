# D1′ — Bounded, Plan-Only Diagnostic Proposal: is a specific weakness worth investigating?

**Revision 4, 2026-09-07 — after third review.** Revisions 2–3 stand; revision 4
removes the superseded fixed-weight estimand, **narrows the inferential claim**
(§4.2: T describes the frozen selected cohort; the resampling measures its
sensitivity to reweighting the observed games; the interval is an **empirical
stability interval**, not a confidence interval for repeated L0 designs), states
what `GO` means accordingly, and makes an undefined colour-arm statistic fail
confirmation explicitly.
Revision 2's corrections, named where they land: (1) neither `agree` nor `lprd`
logically implies the other; (2) the primary statistic is matched within common-support cells
with explicit weights; (3) the analysis is dependence-aware and no precision is
claimed that the design has not measured; (4) confirmation is fully frozen —
ceiling, overflow, cross-half duplicates, low-ply eligibility, floor-versus-power.

**Status: PROPOSAL FOR REVIEW. Nothing has run.** No diagnostic executed, no
model or Java loaded, no seed drawn, reserved or registered, no confirmation
diagnostic inspected, no historical selection changed, no game played, nothing
trained, no gate changed, nothing pushed. **Uncommitted** until reviewed.

**What it stands on.** H1 is closed: attempt 2 completed 224/224 with verdict
**INCONCLUSIVE** (T1j 151.5/224 = 0.6763, Hoeffding95 [0.5856, 0.7671] straddling
0.75). Not reopened, re-thresholded, pooled or extended here. Kept apart from the
verdict: under the tested settings T1j is the **stronger observed opponent**.
🔴 **T1j is an observed opponent, not ground truth.** No comparison below treats
T1j's move as correct. The measured quantity is a **disagreement**, named as such
(§3), and **even a replicated disagreement would not show that copying T1j's
move improves play** — that is a question for a later falsifier (T0), never for
this diagnostic.

**What it reuses, and what is new — stated exactly.**
- **Reused, UNCHANGED:** D0 (2026-08-27) and the frozen **D1 design** — §12.1–
  §12.7 and §13 of `plans/2026-08-27-t1j-sparring-postmortem-opponent-ladder.md`
  (selection rule, digest, matched controls, 221 positions / 1,105 queries, exact
  settings, integrity aborts) and §14's paper block; D2's confirmation structure
  (§6). The D1 instrument (`d1_selection`, `d1_probe`) is the instrument.
- **NEW, PROSPECTIVE — D1′'s analysis:** the decision-level measurement, the
  matched estimator, the dependence-aware uncertainty method, the decision rule,
  the eligibility floors, and the confirmation freeze in §6. §12.8 stated the
  outcomes (`GO`/`NO_GO`/`VOID`) without a metric; **this document supplies the
  metric before any query returns, and it is D1′'s, not "D1 unchanged".**

---

## 1. The question, and why structural statistics cannot answer it

D0 measured structural *outcome* frequencies per ply and found them
near-symmetric between the engines (`05_by_system.json`: `mover_more_fragmented`
0.330 vs 0.386; `created_threat` 0.050 vs 0.051). Two signatures pass §4.5's
well-formedness gate; **none identifies a weakness of ours.** A structural rate
says how often a *board pattern* follows a move, not whether the mover *chose*
differently from the same position. Two engines can share pattern rates while
disagreeing on every third decision.

**D1′'s question is decision-level, from the same positions (§12.0):**

> In positions of a frozen structural cohort where **our incumbent is to move**,
> does the incumbent's *own raw policy* rank the move the stronger opponent
> selects low, **more often than it does in matched control positions** of the
> same opening, colour arm and phase — by an amount an intervention could
> plausibly move, and on data never used to find it?

"Specific" = one named observable family (§7: A search/readout, B value,
C policy, D distribution). "Worth investigating" = it is large and stable enough
in the development cohort to justify a confirmation run, and then recurs on
held-out games at a pre-stated size. A `NO_GO` is a **successful** outcome
(§12.8).

## 2. Data audit — what has been looked at, and at what level

Counts from the JSONL records, computed 2026-09-07.

| Source | Games | Plies (incl. 6-ply openings) | Incumbent-to-move plies | Outcomes seen? | Diagnostics computed? | Positions selected? | Status for D1′ |
|---|---:|---:|---:|---|---|---|---|
| E4 canonical screen | early-stopped, two depths | — | — | yes | no | no | **excluded** (not the tested settings) |
| **L0 discovery half** (reps 0, 1) | 32 | 1,329 | ≈665 | yes | **yes — D0 §3** | **yes — D1's 221** | **DEVELOPMENT** |
| **L0 confirmation half** (reps 2, 3) | 32 | 1,271 | ≈635 | outcome **counts** only (D0 §2) | **NO — code-enforced: `game_features` refuses reps 2, 3** | no | **THE PREREGISTERED HOLDOUT (§4.2, §6). Closed.** |
| H1 attempt 1 (VOID at 60) | 60 | 2,428 | 1,209 | yes | no | no | development only; **excluded from confirmation** (5 of 16 cells) |
| **H1 attempt 2** (COMPLETED) | 224 | 8,139 | 4,049 | **yes — verdict, per-cell, per-opening** | no | no | **DEVELOPMENT, not a holdout** (per review) |

**Finding.** No set has both outcomes and diagnostics unseen. The only balanced,
preregistered, code-protected holdout is **L0's confirmation half** (32 games,
≈635 incumbent-to-move plies; its outcome *counts* were opened by §4.2, a
weaker exposure than a rate table, recorded not argued away). **D1′ names it as
the confirmation set.** Attempt 2 is development evidence only (§4.4). If the
confirmation half fails the eligibility floors of §6.5, the outcome is
`NO_GO — insufficient support`, and the only remedy is a **new preregistered
confirmation match on a fresh block** — a separate proposal.

## 3. The measurement — low-policy-rank disagreement (LPRD) — and the ONE primary hypothesis

Per position D1 captures (§5.2, `IncumbentReadout`): our chosen move (the
400-simulation readout), the **raw legal-move policy**, the root visit
distribution, the root value, the selected move's raw-policy rank and mass,
whether search overrode the raw-policy leader, and identities. T1j (§5.3) gives
one selected move per depth and no distribution, recorded as exactly that.

**Definitions, per position, frozen:**

- `t1j_move_6` — T1j's mdPly-6 move (its two invocations must agree or the run
  is already VOID by §12.7).
- `rank_raw(m)` — rank of move *m* in our raw legal-move policy, 1 = leader;
  **ties broken by the canonical move order `(row, col)`**, never by visits.
- **`lprd`** — `rank_raw(t1j_move_6) > 5`. **Low-policy-rank disagreement**: our
  raw policy does not place the opponent's choice in its top five. *k* = 5 is
  frozen; rank is used rather than mass because mass is dominated by the
  legal-move count, a phase proxy already used for matching.
- `agree` — our chosen move == `t1j_move_6`.

🔴 **Correction (1): neither `agree` nor `lprd` logically implies the other.**
(Statistical independence between them is unestablished and not claimed.) Our
400-simulation
search can select a move the raw policy ranks below fifth, and T1j can select
that same move; then `agree` = True **and** `lprd` = True. Revision 1's
"`agree ⇒ not blind`" was false and is withdrawn. `lprd` is defined purely from
the raw policy and `t1j_move_6`; `agree` is reported alongside as a separate
observation, and the joint
cell `agree ∧ lprd` ("search-rescued agreement") is reported descriptively
because it is the one case that separates policy from search. A **positive
test** with `rank_raw(t1j_move_6) = 7` and our move equal to it must yield
`lprd = True, agree = True`.

**Families and their signatures (from the same position):**

| Family | Signature | Role in D1′ |
|---|---|---|
| **C — policy** | `lprd` | **PRIMARY** |
| A — search/readout | visit share on `t1j_move_6`; `overrode_leader`; `agree ∧ lprd` | secondary, descriptive |
| B — value | root-value trajectory along the recorded continuation (§5.4) | secondary, descriptive |
| D — distribution | out of scope (needs training-data access) | — |

> **H-C (primary).** Within common-support cells of the `mover_fragmentation`
> cohort, the LPRD rate at signature positions exceeds the LPRD rate at matched
> control positions by at least **0.15** (matched estimator, §4.2).

Why family C: it is readable without search noise (the raw policy is a
deterministic function of position and evaluator), and it is the one family with
a concrete, bounded intervention shape that the do-not-repeat list has not
already retired. Why `mover_fragmentation`: the larger cohort, recurring in all
8 openings and both arms per D0; `created_threat` is analysed identically as
**secondary, non-deciding**.

⚠ **What H-C is not.** Not "T1j's move is right". A higher LPRD rate in
signature positions says our policy considers the opponent's choice less often
there than in matched positions. Whether considering it would help is T0's
question.

## 4. Development phase — the frozen D1 design, plus D1′'s prospective analysis

### 4.1 Design — UNCHANGED from §12/§13

Two D0 signatures; discovery half only; incumbent to move; dedup by canonical
digest, one canonical prefix; per-cell cap 3 (cell = opening × colour arm ×
phase); matched controls from the same cells; the six §13 exclusions; **221
positions = 101 / 54 / 30 / 36**; per position 1 incumbent readout + 2 × mdPly 3
+ 2 × mdPly 6 = **1,105 queries, all spent**; 120 s per query; 90 min whole run;
`[202615000, 202615221)` on paper, unregistered; all §12.7 aborts; `VOID` =
repair only. The low-ply result (§13.1) is inherited as the six enumerated rows
and **no ply rule is added to the development cohort**.

**Cell composition of the frozen cohort, computed 2026-09-07 from the discovery
half by the frozen rule (read-only; no selection changed):**

| Cohort | Cells | Common-support cells (≥1 position and ≥1 control) | Positions in common support | Controls in common support | Games contributing (positions / controls / either / both) | Max positions from one game |
|---|---:|---:|---:|---:|---|---:|
| `mover_fragmentation` | 36 | **21** | **62 of 101** | **54 of 54** | 17 / 13 / 20 / 10 of 32 | 14 |
| `created_threat` | 12 | 12 | 30 of 30 | 36 of 36 | 16 / 12 / 22 / 6 of 32 | 5 |

Every common-support cell holds at most 3 positions and 3 controls (the frozen
cap), so cell weights are small and near-equal (§4.2). **39 fragmentation
positions sit in 15 cells with no control**; §13 recorded this and D1′ does not
backfill.

### 4.2 D1′'s prospective analysis — NEW

🔴 **Correction (2): a genuinely matched primary statistic.** The pooled
101-vs-54 comparison of revision 1 mixed cell compositions; a pooled difference
could arise entirely from the 15 control-less cells. It is withdrawn.

**The statistic T, over the frozen selected cohort.** Over the
**common-support cells** *C* of the cohort (both roles present):

  T = Σ_{c∈C} w_c · [ p̂(lprd | signature, c) − p̂(lprd | control, c) ] / Σ_c w_c

with **Mantel–Haenszel weights** w_c = n_sig,c · n_ctl,c / (n_sig,c + n_ctl,c)
from the cohort's own counts (0.75–1.5 across the 21 development cells).
**What T describes: matched LPRD in the frozen selected cohort** — the
cell-matched excess of low-policy-rank disagreement at signature positions over
matched controls, among the 221 positions the frozen §12/§13 rule selected from
these games under these settings. It is a description of this cohort; it is not
presented as an estimate of a population parameter. **Unmatched rows** (positions
in cells without controls) are **retained and reported descriptively** (their
LPRD rate, by cell) and enter no statistic.

🔴 **Correction (3): dependence.** Positions are nested in games (up to 14
fragmentation positions from one game; median 9), and signature and control
positions share games (10 of 20 contributing games hold both). A standard
independent two-proportion interval assumes none of that and is **not used**.
Revision 3 completes the specification below.

**What the resampling measures — and what it does not.** 🔴 Revision 3's
"design expectation of T" is **withdrawn**. Resampling already-selected rows
does **not** reproduce repeated L0 designs followed by selection: deduplication
and earliest-row capping depend on which other games are present, so a
different draw of games could have retained positions that are absent from the
frozen 221, and no resampling of the 221 can see them. The narrowed claim:

> **T describes matched LPRD in the frozen selected cohort. The stratified
> cluster resampling measures T's sensitivity to reweighting the observed
> games. Its central 95% interval is an empirical stability interval, not an
> established confidence interval for repeated L0 designs.**

That is the whole inferential content. Everything below is the algorithm that
produces the stability interval, fixed so it can be reproduced and controlled.

*The resampling — stratified cluster bootstrap, frozen algorithm.*

1. **Strata:** the 16 opening × colour-arm cells of the L0 design, in the frozen
   opening order then arm order (`t1j_red`, `t1j_black`). Each stratum holds
   exactly **2 discovery games** (repetitions 0 and 1).
2. **Clusters:** games. A game is resampled **whole**: every already-selected
   row it contributed — **both roles, all phases, both cohorts** — travels with
   it. Only rows of the frozen 221 are ever resampled; nothing is re-derived.
3. **Draw:** for each stratum, draw 2 games **with replacement** from its 2. A
   replicate therefore always holds 32 game-draws, 2 per stratum, and each
   stratum realises one of three configurations (AA, AB, BB).
   ⚠ **Coarseness, stated:** the replicate space is 3¹⁶ configurations and each
   stratum's draw is a three-point distribution; this bootstrap is as coarse as
   the design is small, and the interval inherits that.
4. **Multiplicity retained:** a game drawn twice contributes its rows **twice**.
   There is **no re-deduplication, no re-capping and no re-selection** inside a
   replicate — the cap and dedup belong to selection, which happened once.
5. **Statistic per replicate:** T(replicate cohort) — common-support cells,
   counts and weights recomputed from the replicate's rows, exactly as in the
   full-sample value. A replicate is a reweighting of the observed games; its T
   says how T moves when some games count twice and others not at all.
6. **Undefined replicates, by rule, never discarded or redrawn:** a cell that
   lost a role in a replicate simply has no defined difference and does not
   enter that replicate's T (its weight would be zero); this is counted and the
   distribution of "defined cells per replicate" is reported. A replicate with
   **zero** defined cells makes T undefined; such replicates are **counted, kept
   in the record, and if their count is greater than zero the interval is
   reported as `UNDEFINED` and the outcome is `NO_GO — bootstrap undefined`.**
   No replicate is dropped, redrawn or imputed.
7. **PRNG:** NumPy `numpy.random.Generator` over the **`PCG64`** bit generator,
   constructed as `numpy.random.default_rng(20260907)` once per analysis; the
   per-stratum draws are `rng.integers(0, 2, size=2)` in the fixed stratum
   order, replicates b = 1 … **10,000** in sequence, no parallelism. The seed is
   an analysis-only constant, appears in no seed registry, and is not a game or
   search seed.
8. **Interval:** the **empirical stability interval** — the central 95% range,
   i.e. the empirical 2.5% and 97.5% quantiles of the 10,000 replicate values, `numpy.quantile(...,
   method="linear")` (Hyndman–Fan type 7). No bias correction, no acceleration;
   the convention is fixed here and stays fixed.
9. **Reproducibility:** the replicate sequence is a deterministic function of
   the frozen cohort file and the seed; a test asserts two runs produce
   identical replicate vectors.

**Why games are the resampling unit.** Positions are nested in games (up to 14
fragmentation positions from one game; median 9) and signature and control rows
share games (10 of 20). A reweighting that splits a game would treat dependent
rows as separately reweightable; whole-game reweighting within opening × arm
strata is the smallest unit that respects both the nesting and the design's
balance. This is a choice of *what stability is measured against*, not an
independence assumption about a population.

🔴 **What the stability interval does and does not establish.** It is the
central 95% range of T across 10,000 reweightings of the observed games. **It is
not a confidence interval for repeated L0 designs, its coverage of any
population quantity is not established, and no power is claimed.** A narrow
interval means T is stable under reweighting the games in hand; a wide one means
a few games carry it. A heavier model (cell-stratified conditional logistic with
cluster-robust errors) was considered and rejected: cells hold ≤ 3 per role.

**Decision rule, predeclared, over the `mover_fragmentation` cohort:**

| Outcome | Rule |
|---|---|
| **`GO`** to D2 | T ≥ **0.15** *and* the stability interval's lower bound > 0 |
| **`NO_GO`** | otherwise — including a lower bound ≤ 0 ("inconclusive" is `NO_GO`; there is no third verdict and no extension) |
| **`NO_GO — bootstrap undefined`** | any replicate with zero defined common-support cells (§4.2 step 6) |
| **`NO_GO — insufficient support`** | the eligibility floor (§4.3) not met; no backfill, no relaxation |
| **`VOID`** | any §12.7 abort; repair only; nothing is read |

`created_threat` is computed identically and reported; it cannot produce a
`GO`.

🔑 **What `GO` means here:** *sufficient effect and resampling stability to
justify spending a confirmation run* — **not** statistical proof of a population
effect. Confirmation (§6) then tests, independently and on games never used to
find the hypothesis, whether the pattern recurs.

### 4.3 Eligibility floor — a count rule, NOT a power claim

The primary statistic is **computed only if** the common support holds **≥ 8
cells, ≥ 40 signature positions, ≥ 30 controls, and ≥ 12 contributing games**.
The frozen development cohort meets it (21 / 62 / 54 / 20). 🔴 **This floor is
a minimum-count eligibility rule. It is not a power calculation and is not
described as one**: power for the frozen effect is unknown for this design and
this document makes no claim about it.

### 4.4 Development-only use of H1 attempt 2 — zero inference, no selection

D0's `ply_features` over attempt 2's 224 games, for one purpose: to report
whether the two signatures still **recur** with §4.5's well-formedness under
the tested settings. Descriptive context; **selects no positions**, forms no
hypothesis, enters no rule. ≈8,139 plies × ~2.7 ms ≈ 22 s; no model, no JVM.
Enlarging the D1 cohort with attempt-2 positions is **not proposed**; it would
be a new selection rule requiring its own amendment before any query.

## 5. Minimum actionable effect — a judgement, recorded as one

**T ≥ 0.15** in matched LPRD rate is the **decision threshold**, chosen by
judgement now, before data, and derived from nothing. It is **not** a claim that
smaller differences cannot improve gameplay, and not a claim about what any
intervention could move; it is the line below which this proposal declines to
spend a confirmation run.

## 6. Confirmation phase — D2 as frozen in §6, fully specified before it opens

🔴 **Correction (4).** Everything below is fixed now; nothing is decided after
the confirmation half is opened, and no branch depends on what a query returns.

### 6.1 When, and what is opened

Only after a development `GO` **and** review of this document as frozen: the
identical pipeline on **L0 repetitions 2 and 3**. `game_features`' refusal of
those reps is lifted by a **reviewed code change for the confirmation run only**
(a new function bound to reps 2–3 and refusing reps 0–1), never a flag.

### 6.2 Confirmation selection — frozen order of operations

1. D0 features on the confirmation half, exactly `ply_features`.
2. Signature rows (`mover_more_fragmented`, `created_threat`), incumbent to
   move — **§12.1 unchanged**.
3. **Low-ply eligibility, prospective and symmetric (§6.4): rows at ply < 5 are
   ineligible for BOTH roles**, applied here, before dedup and capping.
4. Dedup by canonical digest within the confirmation half; one canonical prefix,
   earliest by `(task_id, ply)` — §12.2.
5. **Cross-half duplicates:** any confirmation row whose digest is among the
   **221 development digests or the six §13-excluded digests** is **removed**
   (it has been queried, or was excluded, in development) and recorded as
   `seen_in_development`. 🔑 **Filter first, then select once:** steps 3–5 are
   filters applied to the candidate rows; the cap in step 6 then selects once
   from what remains. Later eligible rows can therefore enter the selection —
   that is the prospective rule, stated as such, and not "no replacement"
   relative to some earlier capped cohort, which never existed for this half.
6. Per-cell cap 3, earliest by `(task_id, ply)` — §12.1.
7. Matched controls: same cells, signature column False, steps 3–6 applied
   identically — §12.3.
8. Freeze the resulting counts and the common-support cells; compute the
   eligibility floor (§6.5) **before any seed is drawn or query issued**.

### 6.3 Hard ceiling and deterministic overflow

**Ceiling: 240 positions, 1,200 queries, 90 minutes.** The discovery half yielded
227 positions from 32 games under the same rule, so ≈200–230 is expected. **If
step 8 yields more than 240 positions, the confirmation run does not start**: it
is `REFUSED` at selection, before any seed or query, and the overflow is
resolved by a written amendment fixing a stricter cap **before** any of it runs.
There is no in-run drop rule, because any drop rule chooses positions, and
choosing positions after seeing the cohort is selection.

### 6.4 Previously unqualified low-ply positions

The one authorized low-ply qualification observed T1j **not searching at plies 1
and 3** and searching at ply 5, within nine `t1j_red` prefixes. Revision 1's
"qualify failures, then exclude them" was an outcome-dependent branch and is
**withdrawn**. The frozen rule is an **eligibility rule, applied before
selection, to both roles, independent of anything a query returns**:

> Confirmation rows at **ply < 5 are ineligible**. Ply-5 rows are eligible, as
> they are in the development cohort. The count of rows removed by this rule is
> recorded per cohort and per cell.

This is **not** a claim that T1j does not search below ply 5 (never
established); it is a decision to not spend confirmation queries where the
instrument is unqualified. Consequence, recorded now: opening-phase cells may
lose support in confirmation, which the eligibility floor (§6.5) then reports
honestly. Any T1j non-completion at an **eligible** ply remains a §12.7 abort —
`VOID`, instrument — exactly as in development.

### 6.5 Confirmation criteria — frozen

The **same statistic T** (§4.2), *k* = 5, applied to the **confirmation
cohort** — its own common support, its own MH weights — with the same stratified
cluster resampling over the 32 confirmation games (repetitions 2 and 3, two per
stratum; B = 10,000; seed `20260907`; conventions of §4.2 steps 1–9), same
direction. 🔴 **This is the same statistic on a different cohort**: support and
weights differ between halves by design. The confirmation claim is "the matched
LPRD pattern recurs, with sufficient size and stability, on games never used to
find it" — what §6 asks — and not "the same parameter was estimated twice".

**Per-arm statistics.** T is also computed within each colour arm separately
(cells of that arm only). **If either arm has no common-support cell, its T is
undefined and the confirmation outcome is `NO_GO — arm undefined`**; an
undefined arm is never treated as zero, as positive, or as absent from the
rule.

| Outcome | Rule |
|---|---|
| **`GO`** to D3 | T_conf ≥ 0.15 *and* the stability interval's lower bound > 0 *and* **T > 0 in each colour arm separately, both defined** (a one-arm artefact cannot confirm) |
| **`NO_GO`** | otherwise, including "smaller than 0.15", "one arm only", "lower bound ≤ 0" |
| **`NO_GO — insufficient support`** | fewer than 8 common-support cells, 40 signature positions, 30 controls, or 12 contributing games |
| **`NO_GO — bootstrap undefined`** | §4.2 step 6 |
| **`NO_GO — arm undefined`** | either colour arm has no common-support cell, so its T does not exist |
| **`VOID`** | §12.7 |

🔴 The floor is a **minimum-count rule, not demonstrated power.** The interval
is a stability interval, not a confidence interval; a wide interval with a large
T is a `NO_GO`, not a reason to extend.

### 6.6 Seeds

One fresh interval, one seed per selected position, **sized by the count frozen
in step 8**, proved read-only against every registry, D1's paper block and both
H1 blocks with the gap policy (≥ 224 from every prior boundary). **Not reserved
by this document.**

### 6.7 Stopping rules

One development run; one confirmation run. No retry on `NO_GO`, no second look
at the confirmation half, no re-thresholding, no cohort merge, no relaxation of
a floor. `VOID` licenses repair of the instrument and one re-run of the *same*
frozen cohort on a *fresh* interval.

## 7. After a confirmation `GO` — and what a `GO` does not mean

A D2 `GO` means: **the matched low-policy-rank disagreement found in the
development cohort recurred, with sufficient size and resampling stability, in a
cohort selected from games never used to find it.** It is not a population
estimate and not a proof. It does **not** mean the incumbent's policy is wrong
there, and it does **not** mean copying T1j's move would improve play. It opens
**one D3 card** for family C only, choosing the smallest intervention matching
the confirmed signature, checked by name against do-not-repeat `#1–#52` and R0;
**T0 (cheap falsifier) before T1 (bounded training)**; promotion is a **new
preregistration**, not a comparison against H1's record.

## 8. Budget

| Phase | Queries | Model loads | JVM launches | Wall cap | Seeds |
|---|---:|---:|---:|---|---|
| §4.4 features on attempt 2 | 0 | 0 | 0 | ≈ 1 min | none |
| Development D1 (frozen design) | 1,105 | 1 | 1 javac + 221 replays + 884 queries | 90 min | `[202615000, 202615221)`, registered only at execution authorization |
| Confirmation D2 | ≤ 1,200, frozen at §6.2 step 8 | 1 | ≤ 1 + 240 + 960 | 90 min | fresh, sized at step 8, proved then |

## 9. Controls the analysis code must pass before either run

Mocked readouts only; injected-defect controls in the harness.

- **`agree ∧ lprd` positive test**: `rank_raw(t1j_move_6) = 7`, our move equal
  to it → `lprd = True, agree = True`; a control that couples them is rejected.
- Symmetric synthetic cohort → `NO_GO`; injected excess of 0.30 at signature
  positions → `GO`; the same excess at controls → `NO_GO` (direction binds).
- **Common support binds**: a cohort whose excess lives only in control-less
  cells → `NO_GO`; a control that pools those cells is rejected.
- **MH weights**: a known-answer test with two hand-computed cells; a control
  replacing them with equal weights is rejected.
- **Stratified cluster bootstrap**: resampling by game within opening × arm
  strata, 2 draws per stratum — a control that resamples positions, or games
  without strata, is rejected by a known-answer test on a hand-built cohort;
  **multiplicity retained** (a game drawn twice counts twice; a control that
  re-deduplicates is rejected); a replicate that loses a role in a cell drops
  that cell and a replicate with zero defined cells is **counted, not dropped**
  (a control that redraws or discards it is rejected); `PCG64`, seed, draw
  order, B and the type-7 quantile convention pinned; identical replicate
  vectors on two runs.
- **Both-arm rule** in confirmation refuses a one-arm effect, and **an
  undefined arm (no common-support cell) yields `NO_GO — arm undefined`** — a
  control that treats an undefined arm as zero or skips it is rejected.
- Rank ties broken by `(row, col)`; a control breaking by visits is rejected.
- **Eligibility floor** refuses at 7 cells / 39 positions / 29 controls / 11
  games and accepts at the floor.
- **Confirmation selection order**: filters (ply < 5, within-half dedup,
  cross-half digests) run **before** the single cap (a control that caps first is
  rejected by a cell where a filtered row would have consumed a slot); ceiling
  240 refuses 241.
- Cohort labels bound type-strictly to the frozen rows; the analysis refuses any
  cohort other than the frozen 221 / the frozen confirmation count.

## 10. What this proposal does NOT do

It authorizes nothing. It runs no diagnostic, loads no model, launches no Java,
reserves or registers no seed, inspects no confirmation diagnostic, changes no
historical selection, plays no game, trains nothing. It leaves H1's verdict,
threshold and interval untouched and pools nothing. It treats T1j as an observed
opponent, not ground truth, and names its measurement a disagreement. Every
number is quoted from a named frozen record, computed read-only from the
development half by the frozen rule, or declared here as a judgement made before
data. **Review may reject it whole; nothing downstream exists yet to unwind.**
