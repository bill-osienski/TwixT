# H4 ANALYSIS — CARD (pilot feasibility · proceed/stop · confirmatory)

**Written 2026-09-23. DESIGN ONLY.** No analysis code, no tests, no gate, no
seed, no game, no aggregation, no outcome access, no push.

🔴 **THIS CARD IS THE AUTHORITY for every H4 analysis.** It is step 3 of the
agreed order, written **before** any analysis code so the analysis cannot be
shaped by the code, or by data. It rests on the replacement card
(`2026-09-21-t1j-h4-replacement-card.md`: §1 scientific core, §5 pilot, §6
study, §8 concentration, §9 interpretation) and the runner card
(`2026-09-22-t1j-h4-runner-persistence-card.md`: the records this analysis reads).

> **Why it is three things, not one.** A single "H4 analysis" command could read
> the pilot's winners while computing its feasibility, and break the blind by
> being run once with the wrong argument. So H4 analysis is **three separate
> entry points in separate modules**, each able to do only its own job:
> **pilot feasibility** (no outcomes), a **proceed/stop artifact** (decided by
> frozen rules, hash-bound, durable), and **confirmatory analysis** (outcomes,
> only after that artifact says PROCEED).

---

## 0. What is shared, and what is forbidden

**Shared, by import:** the runner's reader (`h4_runner.load_games`), which
re-derives every game — transcript digest, per-game record checks, position
digests from the moves — and refuses VOID, killed and fixture files; and its
record schema. Nothing else.

🔴 **H3 IS NOT USED IN ANY FORM.** No H3 analysis module is imported or called,
no H3 function is copied, no H3 constant is reused (`MAX_DUPLICATE_PAIRS`,
`MIN_DISTINCT_PER_CELL`, degeneracy screens), and **no duplicate rule of any
kind** exists. H3's `pair_identity` would degenerate to trajectory identity on
H4 and silently delete the mass H4 measures (replacement §1.3). H3 is closed and
published and is not edited.

---

## 1. PILOT FEASIBILITY — `h4_pilot_feasibility.py`

**Input:** exactly one pilot results file — header design **`H4_PILOT`**, **16
complete pairs / 32 games**, ending in `segment_end`. Anything else is refused:
fixture or study data, a different count, a file ending in `run_void` or in
neither.

### 1.1 What it reports — the complete list

| item | computed from |
|---|---|
| **operational failures** | whether the file is a clean `segment_end` run that the reader validates end to end — any `run_void`, reader refusal or record fault makes the pilot **VOID** (replacement §5.4: zero allowed) |
| **runtime projection** | `mean_game_s = (pilot_wall_s − setup_s) / 32`; `projected_segment_s = setup_s + 148 × mean_game_s`, from `segment_end.total_s` and `segment_end.setup_s` |
| **cap-affected pairs** | the number of pairs with **at least one** `terminal_reason == "cap"` game |
| **replay integrity** | the reader's pass or refusal, and per-game counts of re-derived digests |
| **process behaviour** | per game: query and replay counts, distinct pids, all processes accepted, none alive; answering-routine counts by ply and colour |
| **concentration** | §1.3 — **trajectory-only** measures |

`terminal_reason` appears **only as the categories `win` / `cap`**, which cap
feasibility needs and which carry no competitive outcome.

### 1.2 🔴 What it may NEVER report or compute

**No winner, no winner split, no points, no pair score, no per-arm or per-colour
outcome, no model advantage, no mean, no interval, no outcome pattern.**

⚠ **The blind is PROCEDURAL, and the card says so.** Moves determine outcomes, so
a list of moves could in principle be replayed to recover a winner. What the blind
guarantees is that **the feasibility module neither computes nor outputs one**:

* it works on a **blinded view** built immediately after the reader returns — a
  whitelist of fields (ids, arm, colours, plies, `terminal_reason` category,
  timings, process counts, moves, routines). `winner`, `t1j_points` and
  `incumbent_points` are **not copied into it**;
* the module **never reads** a blind field, **never calls** `winner()`, and
  **never imports** the confirmatory module;
* its report is **whitelist-checked before it is written** (the runner's
  `blinding_violations`); a report carrying any blind field is refused, not
  written.

### 1.3 Concentration — trajectory only

* the **pair-level trajectory tuple** is `(Arm A's move sequence, Arm B's move
  sequence)` — **moves only**. Identical move sequences imply identical outcomes,
  so this is equality of whole games without touching an outcome field;
* reported descriptively: unique complete games by arm, the maximum frequency of
  any game, first-move and first-*k*-ply prefix frequencies, identical pair-level
  tuples, answering-routine frequency by ply and colour (replacement §8);
* 🔴 **"unique outcome patterns" is OUTCOME-BEARING** and is therefore **not**
  reported by the pilot. It belongs to confirmatory analysis only.

### 1.4 The frozen pilot rules (replacement §5.4, transcribed, not changed)

| rule | outcome |
|---|---|
| any operational failure | **VOID** — no feasibility result |
| `projected_segment_s > 10,800 s` | **STOP_RUNTIME** — four-segment H4 reported **operationally infeasible**; redesign separately; **no re-segmenting** |
| cap-affected pairs **≥ 4** of 16 | **STOP_CAP** |
| **all 16** pair-level tuples identical | **STOP_COLLAPSE** |
| otherwise | **PROCEED** |

Precedence when several fire: **VOID**, then **STOP_RUNTIME**, **STOP_CAP**,
**STOP_COLLAPSE**; every rule that fired is **listed**, so no finding is hidden
by the headline.

🔴 **STOP_COLLAPSE says exactly this and nothing more:** *"empirically complete
concentration in the pilot"*. It **does not claim** the underlying distribution
has zero entropy — 16 draws cannot establish that. Anything short of complete
collapse is descriptive and stops nothing, and concentration **never**
invalidates, deduplicates or reweights an observation.

---

## 2. THE PROCEED/STOP ARTIFACT — the only door to confirmatory analysis

Written by a **separate** entry point from the feasibility report, which it
**reads back and re-verifies** rather than trusting:

| field | content |
|---|---|
| `decision` | `PROCEED` · `STOP_RUNTIME` · `STOP_CAP` · `STOP_COLLAPSE` · `VOID` — **computed by the §1.4 rules**, never typed by a person |
| `rules_fired` | every rule that fired, with its values |
| `pilot_results_sha256` | sha256 of the pilot results file |
| `feasibility_report_sha256` | sha256 of the feasibility report |
| `cards` | sha256 of this card, the runner card and the replacement card |
| `written_at` | timestamp |

* written **create-only**, into the pilot's evidence directory;
* 🔴 **durable means COMMITTED.** Confirmatory analysis treats the artifact as
  present only when it is **tracked by git at HEAD and unmodified in the working
  tree** — replacement §5.3: aggregation is not authorized until the decision is
  committed durably. A file on disk that was never committed is not a decision;
* proceeding **does not release the pilot into the study**: pilot games never
  enter any confirmatory estimate.

---

## 3. CONFIRMATORY ANALYSIS — `h4_confirmatory_analysis.py`

### 3.1 🔴 It refuses to run unless ALL of these hold

* its own gate is open — **`H4_STUDY_AGGREGATION_AUTHORIZED`** (created CLOSED
  when the code is written);
* the proceed/stop artifact exists, is **committed and unmodified** (§2), says
  **PROCEED**, and **both of its hashes recompute** against the pilot results
  file and the feasibility report as they stand now;
* its inputs are exactly the **four `H4_STUDY` segment files** — pilot and
  fixture data are refused — each complete, together **296 pairs / 592 games**.

**Anything else refuses, with no partial estimate.** An incomplete study is not
a smaller study (replacement §6: never shrunk after seeing results).

### 3.2 Pair formation and validation — exact

A valid pair is **exactly two games sharing a `pair_id`**, in which:

* one is **Arm A** (incumbent red) and one **Arm B** (incumbent black), with the
  colours on both records agreeing;
* they are **adjacent** (`game_index` 2k and 2k+1) and in the **same segment**;
* their seeds are **distinct**, and no seed or `task_id` appears anywhere else;
* both games are complete (`game_result` present) with `terminal_reason` in
  {`win`, `cap`};
* each game's points are **recomputed** from `winner` and the colours — win 1,
  loss 0, cap 0.5 — and must equal the recorded points.

🔴 **Any malformed pair is an integrity fault and the analysis REFUSES** — it
never drops the pair and continues (replacement §1.3).

### 3.3 Scoring, multiplicity and the interval

* **pair score** = (incumbent points, Arm A + Arm B) ÷ 2 ∈ {0, 0.25, 0.5, 0.75, 1};
* 🔴 **every valid pair is counted with its multiplicity.** Identical
  trajectories are observations, never duplicates: *k* identical pairs
  contribute *k* scores;
* **primary**: the mean pair score over **all** valid pairs, caps scoring 0.5;
* **interval**: two-sided **nominal** 95% Hoeffding, `mean ± sqrt(ln(2/0.05) /
  (2n))` — h(296) = **0.078938** — **nominal under a declared independence model**:
  independent seed bundles **and** isolated T1j process realizations. That
  independence is **an assumption, not a finding** (replacement §6), and the
  report states it beside the interval;
* **cap-free sensitivity**: excludes **whole pairs** containing any cap — never a
  single game — and reports its own mean, `n′` and interval beside the primary;
* **no other sensitivity** is frozen. Any further one requires an amendment
  **before** outcome access.

### 3.4 What it may say (replacement §9)

| result | wording |
|---|---|
| interval entirely **above 0.5** | incumbent stronger in empty-board direct play **under this joint execution protocol** |
| interval entirely **below 0.5** | T1j stronger under this protocol; H4 alone **cannot localize the cause** |
| interval contains 0.5 | unresolved or near parity; says nothing about why H3 differed |
| high cap rate | the protocol often fails to resolve games |
| high concentration | substantial mass on few games; **n is not reduced** to the unique count |

Confirmatory concentration adds the outcome-bearing measures §1.3 withholds —
unique outcome patterns — descriptively. No pooling with H1, H2, H3 or L0.

---

## 4. FAIL-CLOSED CONTROLS — each must be shown to REJECT

Required before any analysis code runs on real data, each against the **real
entry point**, with a clean baseline so a check tightened until nothing passes
cannot satisfy them:

| control | must reject |
|---|---|
| **blind leakage — output** | a feasibility report carrying `winner`, points, a pair score, an interval or an outcome pattern: refused before it is written |
| **blind leakage — computation** | the feasibility module reading a blind field, calling `winner()`, or importing the confirmatory module — an AST walk over executable code, with a clean-baseline control proving it sees a planted read |
| **blind leakage — view** | a blinded view carrying any blind field |
| **malformed pairs** | a missing arm; two Arm As; swapped colours; non-adjacent games; a pair across segments; a repeated seed or `task_id`; a duplicated `pair_id`; an incomplete game; recorded points disagreeing with the recomputed ones |
| **premature aggregation** | no artifact; a STOP or VOID decision; an artifact whose pilot or report hash no longer recomputes; an artifact not tracked by git, or modified; the aggregation gate closed; pilot or fixture input; fewer than four complete segments or ≠ 296 pairs |
| **accidental deduplication** | *k* identical pairs counted as fewer than *k*; the collapse rule computed over **unique** tuples instead of all 16 |
| **exact-collapse rule** | 16 identical tuples → STOP_COLLAPSE with the fixed wording and **no** zero-entropy claim; 15 identical and 1 different → **not** a stop |
| **feasibility boundaries** | projection exactly 10,800 s → proceed, just above → STOP_RUNTIME; 3 cap-affected pairs → proceed, 4 → STOP_CAP |
| **H3 isolation** | any import of an `h3_*` module, or any H3 constant or duplicate rule, in any H4 analysis module |

Controls use **synthetic fixture data only** (runner card §10.1): negative
seeds, `H4_SYNTHETIC_FIXTURE` headers, temporary directories — with the fixture
refusal deliberately relaxed **only inside tests**, and a control proving that
relaxation cannot be reached from either public entry point. The
injected-defect harness is extended with each check deleted in turn.

### Frozen names

```text
pilot feasibility   scripts/GPU/alphazero/h4_pilot_feasibility.py
proceed/stop        written by that module's separate entry point; read by confirmatory
confirmatory        scripts/GPU/alphazero/h4_confirmatory_analysis.py
confirmatory gate   H4_STUDY_AGGREGATION_AUTHORIZED   (created CLOSED with the code)
tests               tests/test_h4_pilot_feasibility.py, tests/test_h4_confirmatory_analysis.py
```

---

## 5. What this card does not do

It writes no code or tests, creates no gate, reserves no seed, plays no game,
reads no outcome, aggregates nothing, and pushes nothing. It does not edit H3.

**The next separately authorized action is implementing this card, with the
confirmatory gate created CLOSED and fixtures only.** Seeds (step 4) and the
pilot (step 5) remain separate authorizations after that.
