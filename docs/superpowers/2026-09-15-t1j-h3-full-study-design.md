# H3 FULL STUDY — DESIGN CARD

**DESIGN ONLY.** Nothing here is implemented, no seeds are reserved, no block is
registered, no opening set is generated, and no execution is authorized. Every
number below is fixed *before* any full-study game exists, which is the only time
it can be fixed honestly.

Predecessor: the H3 pilot, which ran once on 2026-09-15 and COMPLETED (40/40,
exit 0, all five preregistered stop rules CLEAR). Evidence
`docs/superpowers/evidence/2026-09-14-t1j-h3-pilot/`.

---

## AMENDMENT 1 — 2026-09-15, on review of the first draft

Six corrections, recorded here rather than silently folded in. **Four open
questions in §8 of the first draft were decided by the reviewer** and are now
design constants: keep stratum B, keep `h ≤ 0.08`, keep four segments, keep both
alternating orders.

| # | correction | where |
|---|---|---|
| 1 | **The sample size missed its own target.** `N ≥ 288.194` needs 289, not 288; 288 gives `h = 0.08003 > 0.08`. **N = 296**, which also divides cleanly by stratum and segment | §3 |
| 2 | **Precision was described as power.** The table read as if a *true* μ guaranteed a decisive interval. It is about the **observed mean**; and an INCONCLUSIVE result does **not** bound the true difference | §3.4 |
| 3 | **Independence was claimed, not modelled.** Distinct seeds and openings do not *prove* independence — rejection sampling, fixed strata, shared engines and deterministic software all couple things. The interval is **nominal under a declared model** | §2.1 |
| 4 | **The duplicate-pair rule could not see duplicate games.** H2's transcript excludes the opening, because H2 compared repetitions *within one fixed opening*. In H3 two different openings can share a post-opening digest and still be different games. **`opening_digest` enters game and pair identity**; exact duplicate pairs then become a **harness fault**, and equal continuations are reported, never collapsed | §4.1, §4.3, §4.6 |
| 5 | **Segmenting left room for outcome-driven stopping.** Withholding a later segment after inspecting earlier outcomes now **prevents a strength verdict** | §5.5 |
| 6 | **Sensitivity disagreement was defined for one analysis and one failure mode.** A joint rule now covers both sensitivities and both failure modes, including two decisive intervals pointing **opposite** ways | §4.7 |

A seventh, from the same review: **stratum B is "symmetrically co-produced", not
"neutral"** — the mirror balances each engine's role; it does not make the
openings engine-independent (§1.3).

---

## AMENDMENT 2 — 2026-09-15, on review of Amendment 1

`N = 296` and the balanced 37+37 segment composition are confirmed. Three
further changes, and one finding that came out of making them.

| # | change | where |
|---|---|---|
| 8 | **Shared continuations are DESCRIPTIVE ONLY — the `> 7` gate is removed.** Two games from different openings are distinct observations even when their move sequences match, because the opening changes the board those moves are played on. Nothing preregistered says identical continuations invalidate evidence, so nothing gates on them | §4.3, §4.6 |
| 9 | **Two verdict edge cases made explicit**: a sensitivity estimate of exactly 0.50 does **not** support the primary direction and suppresses the verdict; and if any required sensitivity **cannot be computed**, no strength verdict issues | §4.7 |
| 10 | **The CO-PRODUCED generation protocol is frozen** — configurations, PRNG streams, tree lifetime, rejection behaviour, alternating order, pinned artifacts | **§1.7 (new)** |

### 🔴 THE FINDING: THE CO-PRODUCED GENERATOR AS SPECIFIED PRODUCES **TWO**
### OPENINGS, NOT 148

Freezing the protocol immediately exposed why it had to be frozen.

* Under the **match** configuration the incumbent plays `selection_mode:
  "argmax"` — its move is a **deterministic function of the position**.
* **E3a established that T1j is deterministic** at fixed depth: 25 queries, 20
  in one JVM and 5 in fresh JVMs, all returned the same move.

An alternating generator built from those two players has **no entropy anywhere
in it.** Every incumbent-first opening would be the same opening, and every
T1j-first opening would be the same one. Stratum B would contain **2 distinct
positions repeated 74 times each**, and the whole study would rest on two
positions.

This is exactly the class of defect the reviewer's instruction anticipated: it
could not have been caught by inspection of the first draft, and it would have
surfaced as a bewildering result *during generation*, after implementation.

**The resolution is in §1.7.2**, and it changes what stratum B *is*, so it is
recorded as a design decision and not a detail.

---

## AMENDMENT 3 — 2026-09-17, THE CO-PRODUCED STRATUM IS CLOSED

**This amendment makes the study UNIFORM-ONLY.** Stratum B is removed, not
deferred. `N = 296` pairs / 592 games, `h ≤ 0.08`, the colour-reversed pair, the
four segments, the sensitivity rules, the deadlines and the optional-stopping
rule are all **unchanged**; what changes is the population those 296 pairs are
drawn from, and — necessarily — the claim they support.

### 🔴 THE FINDING THAT CLOSED IT: T1j CANNOT MOVE AT THE PLIES THE PROTOCOL NEEDS

Two authorized generation attempts ran, and both VOIDed at opening 0 with **zero
openings accepted**:

| attempt | range | destination | outcome |
|---|---|---|---|
| 1, 2026-09-16 | `[20261200000, 20261259200)` | `…-study-openings` | **VOID** — `KeyError: None`, a bare `IntegrationContext`. Repaired by `ctx.reset(...)` |
| 2, 2026-09-16 | `[20261400000, 20261459200)` | `…-study-openings-attempt2` | **VOID** — T1j `exit 3` at board-ply 1: `FAIL q1: requested depth 6 completed … failures=1` |

Attempt 2's failure is not a bug. It is the **2026-08-31 low-ply qualification**,
which is this programme's own recorded evidence: *T1j never enters alpha-beta at
plies 1 and 3* (12/12 fail at each); ply 5 completes (0/12 fail). §1.7.6 asked
T1j to move at 1/3/5 stones under `incumbent_first` and at 0/2/4 under
`t1j_first` — **precisely where it is known not to search.**

🔴 **This was discoverable from the programme's own evidence without running
anything.** Freezing §1.7 checked the generator for *entropy* (Amendment 2's
finding) and never checked it against the *qualification record of the engine it
scheduled*. Two ranges and two destinations were spent to learn something already
written down.

**Delaying T1j's first move to ply 5+ would not rescue the protocol** — it would
define a *different* opening population, which would need its own pilot. A third
attempt at this design is not authorized, and this amendment does not seek one.

### What changes

| # | change | where |
|---|---|---|
| 11 | **Stratum B is removed.** One population: 296 fresh, distinct, uniformly generated legal six-ply openings. No stratum comparison is computed, no `order` field exists, and no two-population language survives | §1, §1.3, §1.6, §1.7, §5.3, §6.2 |
| 12 | **The supported claim is narrowed explicitly** to performance over *uniformly random legal six-ply openings*. It says **nothing** about realistic or engine-reached positions, and §1.1's objection to that population now stands **unanswered and declared** rather than mitigated | §1.6 |
| 13 | **A generation allocation covering all `296 × 400 = 118,400` candidates**, separation re-proved directly and through the derived streams (v13). The standing 148-opening uniform range is **59,200 — short by exactly half** — and is retired unused rather than extended or combined | §1.7.3, §7.1 |
| 14 | **Generation is engine-free, and therefore is no longer a run.** No checkpoint, no JVM, no gate, no supervised wrapper. `GENERATION_DEADLINE_S` is retained as a value but its cost-based rationale is **void** and replaced | §1.5, §1.7.5 |
| 15 | **Both VOID attempts, their retirement records and their evidence directories are preserved unchanged.** Both ranges and both destinations stay spent whole | §7.1, §7.3 |

### What this amendment costs, stated plainly

The study loses the only reason it had for claiming relevance to play. §1.1
identified the pilot's uniform population as *unrealistic* — "those positions are
not ones either engine would reach" — and stratum B existed to answer exactly
that. Removing it does not answer the objection; it **accepts** it. What remains
is a well-powered, well-controlled measurement over a population nobody plays.
That is a real result and a narrow one, and §1.6 says so in the words the report
must use.

---


## 0. 🔴 WHAT THE PILOT MAY AND MAY NOT CONTRIBUTE

The pilot established two things and only two: **gameplay diversity** (40 games
→ 40 distinct transcripts; all 20 colour-reversed pairs differ internally; zero
capped games) and **runtime** (mean 41.22 s/game, median 40.5, p90 49.8, max
70.8).

| pilot quantity | may this design use it? |
|---|---|
| per-game timing | **YES** — §5 derives the deadline from it |
| distinct-transcript and distinct-pair counts | **YES** — evidence the design is viable at all |
| cap count | **YES** as viability evidence; **NO** as a rate input to any threshold |
| **the outcome distribution over pairs** | **NO. NOWHERE.** |

The pilot's pair outcomes (`win|win` 11, `win|loss` 7, `loss|loss` 1,
`loss|win` 1) are **descriptive pilot data**. They are not used to choose the
null, the decision boundary, the sample size, the direction of any comparison,
or any exclusion rule. Two structural guarantees, not just an intention:

1. **The sample size is derived from a distribution-free bound** (§3), which
   takes no variance and no effect-size estimate as input. There is nowhere for
   an outcome prior to enter the arithmetic.
2. **The pilot's 20 openings are excluded from the study population** (§1.4), and
   **the pilot's 40 games are never pooled into the study's analysis.** The pilot
   is a separate, closed artifact.

`is_strength_verdict` was `false` in the pilot report and stays false: the pilot
supplies no prior about who is stronger, because it never measured that.

### 0.1 🔴 WHAT AMENDMENT 3 CHANGES HERE — IT CUTS BOTH WAYS

Under the two-stratum design the pilot's population was **half** the study's.
Uniform-only makes it **the whole of it**: the pilot and the study now sample the
same distribution, by the same generator, under the same filters.

**One way that helps.** The pilot's diversity evidence — 40 games → 40 distinct
transcripts, 20/20 pairs differing internally, zero capped — was always the
weaker half of the case for stratum A and said nothing about stratum B. It is now
direct evidence about **the study's actual and only population.** The
viability question is answered for the whole study, not for half of it.

**One way it costs.** Every separation argument between the pilot and the study
now rests on **exactly two things**: the 20 pilot openings are excluded by
canonical digest (§1.4), and the match seeds are a fresh disjoint block (§7.1).
There is no longer a *population* difference doing any of that work. Those two
mechanisms were always the load-bearing ones; the difference is that nothing else
is standing behind them now, and a failure of either is no longer partially
absorbed by a second stratum the pilot never touched.

🔑 **The prohibition is unchanged and is if anything more important.** A shared
population is not a licence to pool: the pilot's 40 games stay out of the
study's analysis, and its outcome distribution stays out of every threshold.
Same population, separate experiments.

---

## 1. THE OPENING POPULATION, AND THE CLAIM IT SUPPORTS

> **Amendment 3 rewrote this whole section.** The two-stratum design it replaced
> is not deleted from the record — Amendments 1 and 2 above, and the two VOID
> attempts' evidence, are what it was and why it ended.

### 1.1 The population, and the objection it does not answer
The study uses **uniform random legal play to 6 plies** — the pilot's population,
now the only one. It is diverse and engine-neutral by construction, and it is
**unrealistic**: those positions are not ones either engine would reach.

🔴 **That objection stood in the first draft of this card, it was the entire
reason stratum B existed, and Amendment 3 does not answer it — it accepts it.**
The card said then that a result over these positions supports the claim

> over positions reachable by uniform random legal play at ply 6, …

and *not* the claim anyone actually wants, which is about positions that arise in
play. That remains exactly true. §1.6 is worded so the report cannot quietly
widen it back.

### 1.2 Why the realistic population is not available
There is no opening book. **R1 closed as NO_GO**: a read-only search for an
independent TwixT data source found none. So a realistic population cannot be
borrowed; it would have to be generated — and the two attempts at generating one
are recorded in the Amendment 3 header. T1j cannot move at the plies an
alternating generator requires, and moving its first move later would define a
*different* population needing its own pilot.

🔑 **The pairing does not rescue this either way.** Colour-reversed pairing
cancels a *colour* advantage. It does **not** cancel a *position-family*
advantage. Under Amendment 3 that matters less, because no engine chose these
positions — but it is the reason a single-engine-generated population was never
an acceptable substitute for the co-produced one.

### 1.3 The design: ONE population, 296 openings
| | **UNIFORM** |
|---|---|
| generation | uniform random legal play to 6 plies |
| who moves | nobody; the PRNG |
| standing | engine-neutral by construction |
| realism | **low, and declared** |
| pairs | **296** |

There is **no second stratum, no stratum comparison, and no alternating order.**
`stratum` survives in the durable record (§6.2) as the frozen constant
`"uniform"` — a record that names its own population is better provenance than
one that assumes it — but **no analysis may condition on it, split by it, or
compare across it.** There is nothing to compare it to.

**Admissibility filters**, stated in ENGINE-NEUTRAL structural terms — no
evaluator is consulted, because "the incumbent thinks this position is already
decided" is the incumbent's judgement and would import exactly the bias a uniform
population exists to avoid:

* the position is legal and not already won by either side;
* both colours have placed all three of their pegs (no side has passed);
* no immediate forced win exists for the side to move at depth 1;
* the position is distinct from every other opening in the study **up to the
  board's symmetry group**, by canonical digest;
* the position is distinct from every excluded earlier opening (§1.4), again up
  to symmetry.

A candidate failing any filter is rejected **whole** and regenerated —
whole-position rejection, as the pilot does, never per-move resampling, which
would distort the distribution it claims to draw from.

⚠ **Rejection sampling is itself a coupling.** It makes the realised population a
*conditioned* one, not the raw generator's, and the conditioning is shared across
all openings. This is one reason §2.1 treats independence as a model rather than
a fact. Attempt seeds, attempt caps and the recording of every attempt count are
frozen in §1.7.

### 1.4 Independence from every earlier experiment
The study's 296 openings must be disjoint, up to symmetry, from:

* the **20 pilot openings** — otherwise the pilot's games leak into the study;
* the **8 openings** H1, L0 and H2 played;
* **every opening produced by the two co-produced generation attempts.**

🔑 **The third exclusion is VACUOUS AS TO POSITIONS, and the card says so rather
than letting a reader infer coverage it does not have.** Both attempts VOIDed at
opening 0 and accepted **zero** openings; there is nothing to exclude. The
substantive residue of those attempts is in the **seed ranges**, which are spent
whole (§7.1) — not in the position set.

So the exclusion set is **28 positions**, not 30 and not 324. Checked at
generation time and pinned by `OPENING_SET_DIGEST` before any match seed exists.

### 1.5 🔴 GENERATION IS NO LONGER A RUN
Stratum B's generation loaded the incumbent's checkpoint and started a JVM, so it
was an execution and carried the full apparatus: its own gate, its own supervised
wrapper, its own pre-run verification, its own one-shot rule.

**Uniform generation does none of that.** It draws from a PRNG and applies
structural filters. No model is loaded, no JVM starts, no engine is consulted,
no move is requested. It is a deterministic computation over declared constants —
the pilot generated its 20 openings exactly this way, ungated, and both
`tests/test_h3_study_rules.py` and `h3_study_prerun_verification.py` have been
calling `generate_uniform_openings()` on every invocation for days.

**What it therefore does NOT need:** a gate, a supervised wrapper, a launch
receipt, a process-group cleanup, or an execution authorization.

**What it still DOES need, and these are not negotiable:**

* the output is **create-only** (`O_EXCL` plus `lexists`, so a dangling symlink
  cannot be written through);
* the opening set and its `OPENING_SET_DIGEST` are **frozen and committed before
  a single match seed is reserved**;
* every attempt count is recorded (§1.7);
* the generation constant is **declared, never drawn from a registry**, and
  asserted absent from all of them.

⚠ **This is a genuine reduction in apparatus, and it is worth naming as one.**
The gate, wrapper and receipt machinery built for stratum B is not deleted —
attempts 1 and 2 are the record of why it existed and it caught both failures
cleanly — it is simply not on this path, because there is no longer an engine on
this path to contain.

### 1.6 🔴 WHAT THE STUDY WILL SUPPORT — AND THE REPORT USES THESE WORDS

> Over **legal six-ply TwixT positions drawn uniformly at random** and filtered
> only by the structural admissibility rules of §1.3, played in the frozen
> deterministic argmax configuration: [the result].

**It will NOT support — and the report must not imply — any claim about:**

* positions either engine would actually reach;
* realistic, human-like, or tournament play;
* positions arising from an opening book, a repertoire, or any curated set;
* a co-produced, engine-neutral, or two-population population — **there is one
  population and it is uniform**;
* any configuration other than the frozen one.

🔴 **A uniformly random six-ply position is not a position anyone plays.** An
engine that is stronger over this population may be weaker over the positions it
actually reaches, and this study cannot distinguish those cases. That sentence
belongs in the report verbatim, next to the result, not in a limitations
appendix.

### 1.7 🔴 THE FROZEN GENERATION PROTOCOL
Fixed here, before implementation.

#### 1.7.1 No engine configuration
No engine participates in generation. There is no generator configuration to
freeze, no tree lifetime, no reset rule, and no alternating order — §1.7.1,
§1.7.2, §1.7.4 and §1.7.6 of the pre-amendment protocol are **void**, along with
the frozen research configuration they named. The incumbent's argmax match
configuration is unchanged and applies **only to play**.

#### 1.7.2 PRNG stream assignment — ONE CONTIGUOUS RANGE FOR ALL 296 × 400
* The generation seed is a **declared constant**, asserted absent from every
  registry — accounted, exposed, retired, test-only and consumed — by a test.
  **Generation never consumes a drawable seed** (the pilot's rule).
* Opening *i*, attempt *j*, uses `base + i × MAX_ATTEMPTS + j`. Attempts occupy
  **disjoint, pre-computable** ranges; a rejection never re-draws the seed that
  produced the rejected position.

🔴 **THE STANDING UNIFORM RANGE IS INSUFFICIENT AND IS NOT EXTENDED.**

```
candidates needed :  296 openings × 400 attempts  =  118,400
standing range    :  [20261000000, 20261059200)   =   59,200   (sized for 148)
                                                      SHORT BY 59,200
```

The allocation is **one fresh contiguous range of 118,400**:

```
GEN_SEED_UNIFORM_296 = 20_261_600_000
range                = [20261600000, 20261718400)
```

**Why one fresh range and not the standing one extended, nor two ranges
combined:**

* *extended* — `[20261000000, 20261118400)` overlaps the standing range by all
  59,200 of its seeds. It is the same range with a different end, not a new one.
  Rejected as control 4 of v13;
* *combined* — `attempt_seed` is `base + i × MAX_ATTEMPTS + j`, total only over a
  **contiguous** base. A piecewise base introduces a seam in the one function
  whose whole job is to make attempt seeds disjoint and pre-computable. This
  programme's recurring defect is a seam that looks like a gate; buying 59,200
  seeds with one is a bad trade when a clean range costs nothing;
* the standing range is therefore **retired unused**. No authorized run drew from
  it, but `tests/test_h3_study_rules.py` and `h3_study_prerun_verification.py`
  have drawn on it in-process on every invocation. Those draws built no agent and
  derived no search or readout stream — uniform generation is engine-free — so
  retiring rather than reusing it is conservatism, not necessity. It is free
  conservatism, because the range is superseded either way.

#### 1.7.3 The separation re-proof — v13, RUN
Collision proof **v13**, `docs/superpowers/evidence/2026-09-17-t1j-h3-uniform-only-amendment/`.
Registries and the four masks are **imported, never retyped**. Nothing is drawn,
nothing is registered, no generator is built: XOR and set arithmetic only.

**Three generation ranges are now prior and NO REGISTRY CAN SEE ANY OF THEM.**
They are added explicitly — the same term v10 had to add by hand for D1's paper
reservation, and a registry-only enumeration would call an overlapping candidate
clean:

| prior range | standing |
|---|---|
| `[20261000000, 20261059200)` | the 148-opening uniform range — **retired unused** by this amendment |
| `[20261200000, 20261259200)` | co-produced attempt 1 — **SPENT WHOLE**, VOID, 0 accepted |
| `[20261400000, 20261459200)` | co-produced attempt 2 — **SPENT WHOLE**, VOID, 0 accepted |

Result, against 183,957 prior seeds and 919,785 prior values:

| check | result |
|---|---|
| direct overlap | **NONE**, in all ten categories |
| derived-stream collisions | **0** over 592,000 derived values |
| injectivity of the candidate's own derivations | **592,000 = 118,400 × 5** |
| nearest gap to any other reservation | **140,800**, floor **118,400** (the candidate's own size) |
| negative controls rejected | **10 / 10** |
| **verdict** | **CLEAN** |

The two boundary controls are the ones that matter: a range **starting exactly
where attempt 2 ends** rejects at gap 0, and a range **one seed inside the floor**
rejects at gap 118,399 < 118,400. The actual nearest distance is reported, not
just the floor, so a narrow choice could not hide behind a small threshold.

🔑 **The derived-stream half of the proof is conservative, and deliberately so.**
Uniform generation feeds its seed to a PRNG and never builds a
`SeededReferenceAgent`, so it derives no search or readout stream and a collision
between a generation seed and a match seed's XOR image would be harmless today.
It is proved anyway, at no cost, so the range stays clean if any future
implementation on this path ever does construct an agent — and so the proof does
not depend on the "engine-free" claim being permanently true.

#### 1.7.4 Rejection behaviour
* **Whole-position rejection**: a candidate failing any §1.3 filter is discarded
  entire and regenerated from the next attempt seed. Never per-move resampling.
* `MAX_ATTEMPTS = 400` per opening is a **declared constant**. Exhausting it is a
  **hard failure that aborts generation** — it never yields fewer than 296
  openings, and it never relaxes a filter.
* **The attempt count for every opening is recorded** in the pinned artifact. A
  high rejection rate changes the conditioning of the population, so it is
  evidence about the population, not a private detail of the loop.
* A global cap on total attempts, also declared, aborts a generator that wanders.

#### 1.7.5 🔴 THE GENERATION DEADLINE IS RETAINED, ITS RATIONALE IS VOID
`GENERATION_DEADLINE_S = 10800` is **kept as a value**. Its recorded rationale is
**not** kept, and the correction is recorded rather than folded in silently:

> the old rationale derived the bound from the pilot's measured **41.22 s/game**,
> i.e. ~0.81 s per engine ply → ~4.85 s per six-ply *co-produced* attempt →
> ~718 s for 148 openings at one attempt each.

**Every term in that derivation is an engine cost, and there is no engine on this
path.** Uniform generation of 148 openings completes inside the test suite in
about a second. Reusing the co-produced rationale for uniform generation would be
a bound borrowed from a different quantity — precisely the defect Amendment 2's
review caught when the generation deadline was first borrowed from the match
segment.

**Its rationale now:** it is a **runaway guard, not a cost estimate.** The real
bound on generation work is `MAX_ATTEMPTS` × 296; the deadline exists only so a
generator that wanders cannot run unbounded. It is expected to be unreachable by
three orders of magnitude, and if it ever fires that is a **defect report**, not
a capacity result.

#### 1.7.6 The artifacts that get pinned
Frozen and committed **before a single match seed is reserved**:

1. **The opening set** — all 296 — with `OPENING_SET_DIGEST` over it.
2. **Per-opening provenance**: generation seed, **attempt count**, the move
   sequence, and the canonical (symmetry-reduced) digest. **No `order` field and
   no `stratum` discriminator** — there is one population, and `stratum` is the
   constant `"uniform"`.
3. **The generator's identity record** — the declared seed constant, the filter
   set, and the toolchain pins. No engine configuration, because no engine runs.
4. **The generation trace and exit code.**

A regenerated set that does not reproduce `OPENING_SET_DIGEST` is a different
population and may not be substituted for this one.

---


## 2. THE PAIRED SCORE AND THE PRIMARY RULE

### 2.1 The unit is the pair — and independence is a MODEL, not a fact
Each opening is played **twice**: once with the incumbent as red, once as black.
That pair is the unit of evidence.

🔴 **The interval is NOMINAL UNDER A DECLARED PAIR-INDEPENDENCE MODEL.** Distinct
openings and distinct PRNG seeds do not *prove* that pair scores are independent.
At least four things couple them:

* **rejection sampling** conditions the whole population jointly (§1.3);
* **fixed strata** make the composition deterministic, not a random draw;
* **the same two engines** play every game, so a systematic weakness of either
  affects every pair in the same direction;
* the software is **deterministic**: given the seed and the position, nothing is
  random in the probabilistic sense at all.

The model is a *declaration about how the interval is to be read*, and it is
stated in the report next to the interval. What the pairing does buy, concretely,
is that the two games of a pair are not treated as independent observations —
which is a real improvement over H1 and H2, whose intervals were nominal over
*games* that shared openings and repetitions.

### 2.2 The score
For pair *i*, let the incumbent's points across its two games be
`p_i ∈ {0, 0.5, 1, 1.5, 2}`, where each game contributes 1 for an incumbent win,
0 for a loss, and **0.5 for a capped game with no winner** (§4.4). The pair score
is

```
X_i = p_i / 2   ∈ [0, 1]
```

The estimand is **μ = E[X]** under the declared model, the incumbent's expected
paired score. `μ > 0.5` favours the incumbent; `μ < 0.5` favours T1j.

### 2.3 The primary rule: PARITY AT 0.50
The null is **μ = 0.5** — the two engines are equal. This comes from the question
("is T1j stronger than our incumbent in the deterministic argmax
configuration?"), exactly as H2's card fixed it, and from nothing observed.

A **two-sided 95% Hoeffding interval** on the mean of the pair scores:

```
X̄ ± h,     h = sqrt( ln(2/α) / (2N) ),   α = 0.05
```

Three outcomes, declared now:

| interval | conclusion |
|---|---|
| entirely **above** 0.5 | the incumbent is stronger over this population, at this precision |
| entirely **below** 0.5 | **T1j is stronger** over this population, at this precision |
| **straddles** 0.5 | **INCONCLUSIVE at this precision.** Not "no difference", and not a bound on the difference |

No other summary is a verdict. Per-stratum, per-colour and per-outcome-category
figures are **descriptive only**, as H1's per-colour split was.

---

## 3. SAMPLE SIZE, FROM PRECISION ALONE

### 3.1 Why Hoeffding, specifically
Hoeffding's bound needs only that `X_i ∈ [0,1]` and the independence model of
§2.1. It needs **no variance estimate and no anticipated effect size** — so the
pilot's outcome distribution has no route into the sample size. A variance-based
calculation would have needed exactly the number this design is forbidden to
use. The conservatism is the price of that guarantee, and it is worth paying.

### 3.2 The declared precision target
**h ≤ 0.08 at 95% confidence** — kept on review.

Its basis, fixed independently of any observation:
* a paired-score margin of 0.08 is roughly 0.58 vs 0.42 — the smallest
  difference the programme would act on, by decision-relevance;
* **H1 already failed at coarser precision**: 224 games gave a half-width of
  ~0.09 and straddled its bar, returning INCONCLUSIVE. A successor design that
  does not improve on the precision of the design that already failed has not
  addressed why it failed.

### 3.3 The arithmetic — corrected

```
ln(2/0.05) = ln 40 = 3.68888
N ≥ 3.68888 / (2 × 0.08²) = 288.194     →  N ≥ 289
```

🔴 **The first draft wrote 288 and claimed `h = 0.08003`, which is ABOVE the
target it had just declared.** Rounding 288.194 down does not satisfy `h ≤ 0.08`;
a target missed by the arithmetic meant to enforce it is exactly the defect class
this programme keeps finding. 289 satisfies it (`h = 0.07989`) and is prime — it
divides by neither stratum nor segment.

**N = 296 pairs = 592 games**, `h = sqrt(3.68888 / 592) = 0.07894 ≤ 0.08`, and it
preserves every balance without remainder:

| | |
|---|---|
| segments | **4 × 74 pairs** |
| games per segment | **148** |
| **total games and seeds** | **592** |

**Amendment 3 changes nothing here.** `N = 296` was chosen to satisfy `h ≤ 0.08`
and it still does; it divided cleanly by stratum *and* segment, and losing the
stratum constraint does not disturb the segment one — 296 = 4 × 74 either way.
The sample size was never derived from the number of strata.

### 3.4 What this can and cannot resolve — about the OBSERVED mean
An interval is decisive when `|X̄ − 0.5| > h = 0.07894`, i.e. `X̄ > 0.5789` or
`X̄ < 0.4211`.

| **observed** X̄ | interval at N = 296 | outcome |
|---|---|---|
| 0.60 | [0.521, 0.679] | decisive |
| 0.58 | [0.501, 0.659] | decisive, marginally |
| 0.57 | [0.491, 0.649] | straddles 0.5 |
| 0.55 | [0.471, 0.629] | straddles 0.5 |

🔴 **THIS IS PRECISION, NOT POWER.** The table says what an *observed* mean
implies. It does **not** say that a true μ of 0.58 will produce a decisive
interval — sampling variation can put the observed mean anywhere, and a true
difference can be missed at any size.

🔴 **AND AN INCONCLUSIVE RESULT BOUNDS NOTHING.** It does not establish that the
true difference is smaller than 0.08, nor that it is small, nor that there is
none. It establishes exactly one thing: **this study did not separate the paired
score from parity.** The first draft said the difference "is smaller than the
programme declared worth acting on" — that was a claim about the world made from
a failure to measure, and it is withdrawn.

---

## 4. IDENTITY, DEDUPLICATION, EXCLUSIONS, OVERLAP, CAPS

Inherited from the pilot card, which was reviewed and is in force; the
full-study specifics follow.

### 4.1 🔴 GAME AND PAIR IDENTITY MUST INCLUDE THE OPENING
H2's `transcript` — moves, movers, terminal reason, winner from `opening_bound`
onward — **excludes the opening**, because H2 compared **repetitions within one
fixed opening**, where the opening was a constant and carried no information.

**In H3 the opening is the variable.** Two *different* opening positions can
produce the same post-opening continuation and therefore the same transcript
digest, while being entirely different games. A rule keyed on the transcript
digest alone would collapse them.

```
game identity = ( opening_digest, incumbent_colour, transcript_digest )
pair identity = ( opening_digest, red_digest, black_digest )
```

### 4.2 Deduplication happens AT THE PAIR
Two pairs are duplicates when their **pair identity** is identical. Duplicates
collapse to one scored pair. **A game is never dropped alone** — dropping half a
pair destroys the colour balance that makes the pair the unit.

🔑 **AND WITH `opening_digest` IN THE IDENTITY, THIS SHOULD NEVER FIRE.** The 296
openings are distinct up to symmetry by construction (§1.3), so two distinct
pairs cannot share a pair identity. **A duplicate pair is therefore a HARNESS
FAULT** — the same opening scheduled or recorded twice — not a property of the
population. It is treated as one in §4.6.

### 4.3 Shared continuations are REPORTED, never collapsed
Two pairs from **different** openings whose games share a post-opening transcript
digest have converged to the same continuation from different starts. That is
informative and it is **not duplication**:

* `shared_continuation_pairs` — pairs sharing at least one transcript digest with
  a pair from a different opening;
* `shared_continuation_relations` — the number of such sharing relations.

🔴 **BOTH ARE DESCRIPTIVE ONLY, AND NEITHER GATES ANYTHING.** Two games played
from **different** openings are **distinct observations** even when their move
sequences match: the opening changes the board those moves are played on, so the
resulting positions differ. A matching continuation is a coincidence of move
coordinates, not a repeated game.

Amendment 1 put a `> 7` ceiling here. **It is removed.** Nothing preregistered
says that identical continuations invalidate evidence, and a gate without a
stated reason to fire is a gate that would have suppressed a sound result.
Neither count collapses a pair, neither is a duplicate, and neither can suppress
interpretation.

*(These replace the pilot's `partial_overlap_pairs` / `partial_overlap_relations`,
which were computed on transcript digests alone. The two counts stay distinct
because they answer different questions, and are never merged into one figure.)*

### 4.4 Exclusions, all counted and reported
Excluded before scoring: VOID games, unscoreable results, duplicated `task_id`,
malformed records (bad digest, reason, winner or ply count), records naming an
unknown `pair_id`, and any pair missing a colour. A pair with an excluded game is
excluded **whole**.

### 4.5 Caps and within-pair-identical pairs stay IN the primary
A capped game has no winner. **It scores 0.5 and stays in the primary.** A pair
whose two games have the same identity contributed no discrimination; it is
**also retained**. Excluding either would be an outcome-dependent exclusion —
cap frequency and transcript identity are properties of *play*, and removing them
selects on how the games went.

Reported: `capped_games`, `pairs_with_a_cap`, `within_pair_identical`.
Two preregistered sensitivity analyses, plus their intersection, in §4.7.

### 4.6 Degeneracy gates — declared now
The rule the pilot card fixed still governs a partial run: **a monotone COUNT may
fire; a RATIO or QUANTILE may not.**

| gate | threshold | what it means |
|---|---|---|
| **duplicate pairs** | **> 0** | 🔴 a **HARNESS FAULT** — the same opening twice. Not a population property. Any occurrence voids interpretation until explained |
| within-pair-identical pairs | > 7 | pairs contributing no discrimination |
| capped games | > 118 (20% of 592) | the configuration is not producing decisive games |
| completed pairs | < 148 (half) | below this, counts only — no interval interpretation |

A fired gate **suppresses interpretation, not the counts.** The counts are always
reported.

🔑 **Shared continuations are deliberately ABSENT from this table** (§4.3): they
are reported, and they gate nothing.

### 4.7 🔴 SENSITIVITY DISAGREEMENT — the joint rule, over BOTH analyses
Three analyses beside the primary **P** (all scoreable pairs):

* **C** — pairs containing no capped game;
* **I** — pairs whose two games are not identical;
* **J** — pairs satisfying both.

Each is reported with its own N, X̄ and Hoeffding interval at *its* N. Their
intervals are wider than P's simply because they drop pairs; that alone is not
disagreement.

**A strength verdict issues only if all four hold:**

1. **P excludes 0.5**; and
2. **every sensitivity is COMPUTABLE**; and
3. **every sensitivity's point estimate lies strictly on P's side of 0.5** —
   `sign(X̄ − 0.5)` is identical and non-zero for P, C, I and J; and
4. **no sensitivity's interval is decisive in the opposite direction to P.**

If any fails, the study **reports all four analyses and declares nothing.** Two
edge cases, made explicit because each would otherwise fall through a gap:

* 🔴 **A sensitivity estimate of EXACTLY 0.50 suppresses the verdict.** It does
  not "cross" to the other side, so a rule phrased as crossing would have let it
  pass — and an estimate sitting on the null supports neither direction. `sign`
  must be non-zero, not merely unchanged.
* 🔴 **A sensitivity that CANNOT BE COMPUTED suppresses the verdict.** If an
  exclusion set leaves no pairs, or too few to form the analysis at all, there is
  no estimate — and an absent check is not a passed check. The study reports the
  analyses it has and declares nothing. Fail-closed, as everywhere else here.

In particular:

* **two decisive intervals pointing opposite ways suppress the verdict** — this
  was undefined in the first draft, and it is the most dangerous case, because
  each looks conclusive alone;
* a sensitivity that merely widens to straddle 0.5 **while keeping P's sign** is
  reported as reduced precision from the dropped pairs, and does not by itself
  suppress the verdict — suppressing there would penalise the study for the
  mechanical loss of n rather than for a real disagreement.

---

## 5. RUNTIME AND DEADLINES, FROM MEASURED TIMING

### 5.1 The measurements (the pilot's, and the only pilot numbers used here)
mean **41.22 s/game**, median 40.5, p90 49.8, max 70.8; 40 games in 1,648.7 s.

### 5.2 Why a single 592-game run is the wrong shape
592 games ≈ **6.78 h** at the mean, **8.19 h** at p90. H2's attempt 3 VOIDed on
its own 28,800 s deadline at game **692 of 736** — 94% complete, and it produced
no verdict at all. A long single-shot schedule converts a late failure into total
loss.

### 5.3 Four segments, fixed in advance
**4 segments × 74 pairs = 148 games each.** Under Amendment 3 there is one
population, so a segment has no composition to balance: it is 74 consecutive
pairs of the frozen plan. The per-stratum and alternating-order allocations that
stood here — 37 + 37 per segment, 19/18/19/18 incumbent-first — are **void**.

| per segment | |
|---|---|
| expected | **1.69 h** at the mean, **2.05 h** at p90 |
| **deadline (CHOSEN)** | **10,800 s (3.0 h)** — 1.77× the mean estimate, 1.47× the p90 estimate |
| seeds | its own contiguous **148-seed** quarter of the block |
| outputs | its own create-only directory |
| schedule digest | its own |

### 5.4 The schedules are frozen before the first segment runs
🔑 **The segments, their order, their openings, their stratum composition and
their seeds are all fixed by the frozen plan, in one artifact, before any segment
executes.** Segments run in order; the analysis uses every completed segment; no
segment may be re-run.

### 5.5 🔴 WITHHOLDING A SEGMENT AFTER SEEING OUTCOMES PREVENTS A VERDICT
The four executions are separately authorized *operationally* — that is what
makes a late failure survivable — and that same separation is the opening for
outcome-driven stopping. So it is closed explicitly:

* **Early results must not inform whether a later segment runs.** Stopping
  because the first two segments "look decided" is optional stopping, and it
  invalidates the interval.
* **If any segment is withheld after its predecessors' outcomes were inspected,
  no strength verdict may issue.** The study reports counts and intervals,
  flagged, and declares nothing.
* Operational reasons to stop that are **independent of outcomes** — hardware
  loss, a toolchain failure, a withdrawal of authorization — are recorded with
  their cause, and §5.6 governs.
* The report states, for every segment: authorized, executed, completed or
  withheld, and **whether any outcome had been inspected at the time**.

### 5.6 🔴 A VOIDed segment is NOT a clean exclusion, and the card says so
If a segment VOIDs, the study reports the realized N with its **wider** interval,
recomputed — not the interval it planned for.

But a VOID is not guaranteed outcome-independent. **H2's attempt 3 died precisely
because one cell's games ran ~6× the mean** — a timeout correlates with game
length, which correlates with cap rate, which is an outcome. Therefore:

* any VOIDed segment is named in the report, with its cap and ply statistics up
  to the abort;
* the primary is reported **flagged** as resting on a non-random subset;
* if a VOID occurs, the study may report the interval but **may not declare a
  verdict** without a separate review of whether the loss was outcome-correlated.

A timeout is a reportable outcome. It is not a quiet exclusion.

---

## 6. DURABLE RECORDING — THE SEED GOES IN THE RECORD

### 6.1 The defect this fixes
The pilot's `task_result` **does not carry `seed`.** `_play_one` returns it and
the run body's projection drops it, so the pilot's exposure accounting had to be
*derived* — from the seeded schedule digest in the header plus the positional
binding row *i* → `lo + i`. H2's records carried the seed directly. That
derivation is sound and pinned, and it is still weaker provenance than a value in
the record.

### 6.2 The required `task_result` schema
Every game's record must carry, at minimum:

```
task_id, seed, pair_id, stratum, incumbent_colour, opening_digest,
transcript_digest, terminal_reason, winner, plies, elapsed_s
```

`seed`, `stratum` and `opening_digest` are the additions.

🔑 **`stratum` is now the frozen constant `"uniform"`, and it stays** — a record
that names its own population is better provenance than one that assumes it. But
it is a label, **not a factor**: no analysis may condition on it, split by it, or
compare across it, because there is nothing to compare it to. **There is no
`order` field**; the alternating-order machinery went with stratum B.

With these fields:
* **seed accounting is READ, not derived** — the exposure count comes from the
  records themselves;
* every game names the position it was played from, so a game is verifiable
  without reconstructing the plan — and §4.1's identity rule is computable **from
  a single record**, which is what makes it enforceable;
* the population is named in the record rather than inferred from the schedule.

### 6.3 The header keeps what the provenance repair added
The durable header carries the **whole** incumbent identity and
`selection_mode`, read off the configuration object the seam hands the builder —
one object, one origin, asserted by identity and not equality. That machinery
exists and is exercised; the full study inherits it unchanged.

---

## 7. SEED ACCOUNTING, OUTPUT PATHS, ONE-SHOT RULES

### 7.1 Fresh seeds, and nothing recycled
* **592 match seeds**, a fresh contiguous interval, never any part of
  `[202624000, 202624040)` (the pilot's — EXPOSED 40 / RETIRED WHOLE) or any
  other spent block.
* A **collision re-proof** against the registries as they then stand, with the
  candidate excluded **by identity** in the gap check as well as the overlap
  check, and the gap floor equal to the candidate's own size.
* Registered **ACCOUNTED only**. Registration is bookkeeping; permission is a
  separate review.
* **The generation PRNG constant is declared, not drawn**, and is asserted absent
  from every registry. **One constant now, not two.**

🔴 **THREE GENERATION RANGES ARE SPENT OR RETIRED, AND NO REGISTRY HOLDS ANY OF
THEM.** Any future re-proof must add them by hand, exactly as v13 does:

| range | standing | preserved at |
|---|---|---|
| `[20261000000, 20261059200)` | **retired unused** — the 148-opening uniform range, superseded by §1.7.2 | — |
| `[20261200000, 20261259200)` | **SPENT WHOLE** — co-produced attempt 1, VOID, 0 accepted | `evidence/2026-09-15-t1j-h3-study-openings/` |
| `[20261400000, 20261459200)` | **SPENT WHOLE** — co-produced attempt 2, VOID, 0 accepted | `evidence/2026-09-16-t1j-h3-study-openings-attempt2/` |

**Both VOID attempts, their receipts, traces and retirement records are preserved
unchanged.** Neither is amended, reinterpreted, or partially reclaimed by this
amendment. A range that was drawn on is spent whether or not an opening survived
— attempt 1 built the incumbent on its first seed and put a query to T1j from it;
attempt 2 got a T1j `exit 3` from its own.

* **The new uniform generation range `[20261600000, 20261718400)` is proved clean
  by v13** (§1.7.3), directly and through the derived streams, with the gap floor
  at its own size of 118,400 and 10/10 negative controls rejected.

### 7.2 Segment sub-blocks
Segment *k* uses `[lo + 148k, lo + 148(k+1))`, four blocks of **148 seeds**. On
start, that quarter is retired whole under the one-shot rule; exposure within it
is read from the records (§6.2). A VOIDed segment retires its own quarter and
leaves the others usable — which is the point of segmenting.

### 7.3 Output paths
A fresh evidence directory per segment, create-only (`O_EXCL` plus `lexists`, so
a dangling symlink cannot be written through), added to `SPENT_OUT_DIRS`
afterwards. No segment writes into another's directory, and none writes into any
earlier experiment's.

Both co-produced destinations remain in `SPENT_OUT_DIRS` and are **never reused
or reopened**, VOID or not.

### 7.4 One-shot execution
Each segment is authorized separately, subject to §5.5. No retry, no replacement
block, no re-run of a completed or VOIDed segment. A second full study would need
a fresh block and its own re-proof.

---

## 8. THE DECIDED QUESTIONS

Amendment 2's three changes and the §1.7 protocol were settled on review,
2026-09-15, together with the finding in the Amendment 2 header.

The first draft's four open questions were decided on review, 2026-09-15:

1. ~~**Keep stratum B.**~~ 🔴 **REVERSED BY AMENDMENT 3, 2026-09-17, on
   evidence rather than on reflection.** The decision was right when it was made
   and the reasoning still holds: without stratum B the study *does* retain the
   unrealistic-position limitation that motivated H3. It was not reversed because
   the cost was reconsidered — it was reversed because **the population cannot be
   built.** T1j cannot move at the plies the protocol requires, two authorized
   attempts VOIDed proving it, and a protocol that moved T1j later would define a
   different population needing its own pilot. **The limitation is now accepted
   and declared (§1.1, §1.6), not mitigated.**
2. **Keep `h ≤ 0.08`** — with the arithmetic corrected to 296 pairs (§3.3).
3. **Four segments.**
4. ~~**Keep both alternating orders.**~~ 🔴 **VOID UNDER AMENDMENT 3** — there is
   no alternating order, because there is no co-produced stratum. The narrowing
   this question produced (*symmetrically co-produced*, then narrowed again by
   the old §1.7.2 to *only one engine supplies any variation*) is kept in the
   record as the reasoning that led here, and applies to nothing in the current
   design.

---

## 9. SCOPE OF THIS DOCUMENT

No seed block is chosen, proved or registered; no OFFICIAL opening set is
generated; no gate is opened; no game is played. The pilot's block stays EXPOSED
40 / RETIRED WHOLE, **all TEN gates stay False** (§9.1 — earlier drafts of this
line said eight, and were wrong), and the push stays held.

✅ **AMENDMENT 3 IS NOW IMPLEMENTED, GATE-SHUT** (2026-09-17). The card and the
code agree again. `h3_study_rules` carries one population and one generation
range; `h3_study_generator` is engine-free; the analysis refuses a record from
any other population and a report that weakens the claim; the schedules are four
segments of 74 pairs with no composition to balance.

**The retired engine path is preserved and unreachable.**
`h3_coproduced_generator_retired`, `h3_generation_command` and
`h3_generation_preflight` keep the protocol, the entropy finding and both VOID
post-mortems. `check_gate` refuses unconditionally, and the retirement is
enforced by arithmetic rather than prose: both co-produced ranges are spent, so
`attempt_seed` refuses before a mover can exist. A test asserts no active module
imports any of the three, with a control proving the files are still there.

### 9.1 🔴 THE GATE COUNT IS TEN, AND MY REPORTS SAID EIGHT

Every report in this programme said "all EIGHT gates are False". **There were
ten.** The two omitted — `LOWPLY_QUALIFICATION_AUTHORIZED` and
`RUNTIME_REQUAL_AUTHORIZED` — belong to qualifications that had already run, so
nobody thought about them. A retired gate is still a closed authorization
constant in the source.

**The number was never the defect.** Three pre-run checkers each kept their own
hand-typed list and the three disagreed — H2's said **seven**, the pilot's said
**eight**, the study's said **ten** — and reports quoted whichever was nearest. A
hand-kept list cannot see a gate nobody remembered to add to it, which is the one
case a gate count exists to catch.

`gate_inventory` now **derives** the inventory from source by AST — parsed, never
imported, so taking an inventory cannot trip a gate — and all three checkers
import it. `tests/test_gate_inventory.py` pins the expected set, so adding or
removing a gate is a deliberate edit rather than a drift.

**Ten, and Amendment 3 SWAPPED one rather than removing it:** the generation
EXECUTION gate went, because engine-free generation runs nothing; the
**population-freeze barrier** took its place, guarding the one act that is still
irreversible — declaring a population THE population.

### 9.2 🔴 THREE INTEGRITY GAPS, FOUND ON REVIEW AND REPAIRED (2026-09-17)

The uniform-only structure was sound and the population was **not ready to
freeze**. Each gap was a control that existed and did not bind.

**1. The barrier was not one-shot.** `freeze_population()` accepted
`out_path`/`trace_path` and never restored the constant. Once opened it stayed
open, so repeated calls could write any number of candidate "official"
populations — and because generation is deterministic they would be
*byte-identical*, which is worse, not better: several indistinguishable
artifacts, none of them the one the study is defined over.

> The entry now takes **no arguments**. `h3_freeze_command` restores and
> **verifies** the barrier in a `finally` — verified by reading the source file
> back, because the imported module still holds the pre-rewrite value. A failed
> restoration **supersedes every other outcome** and has its own exit code: a
> population written with the barrier open is not a completed freeze.

**2. Generation happened outside the deadline and the terminal trace.** Python
evaluates arguments first, so `write_artifact(openings=build_population())` ran
the entire walk **before** the trace opened and the clock started. A hang or
crash inside generation left no terminal record and could not trip the runaway
guard §1.7.5 promises. **The guard was real, and it guarded only the part that
never takes any time.**

> `build` is now a callable invoked inside, after the trace exists and the clock
> runs; the deadline is checked once per **attempt** inside the candidate loop.
> The clock is `monotonic` — wall time steps backwards over an NTP correction.
> The artifact is **fsynced before `OK` is recorded**, because `OK` asserts the
> file is on disk.

**3. 🔴 THE DIGEST DID NOT BIND THE MOVES.** Both checks trusted each row's
**declared** digest and neither replayed the moves, so an opening could be
rewritten while its digest *and* `OPENING_SET_DIGEST` stayed valid — and the
runner would accept and play the altered position. `load_opening_set` never
called `validate_artifact` at all. **A digest that is never recomputed from the
thing it digests is a label, not a checksum.**

> Validation now binds, in order: exact artifact and opening schemas; index and
> segment from the **study's** plan, not the artifact's opinion; type-strict
> six-move replay through the real engine; **recomputed** canonical digest;
> `seed == attempt_seed(base, index, attempts − 1)`; the PRNG walk re-derived
> from that seed reproducing the exact moves; and the claim, generation note and
> generator identity exactly. The loader delegates to it, then requires the set
> to be **the pinned one** — self-consistent is not the same as pinned.

**And the generator identity now pins what reproduces the walk.** "No engine" is
not "no toolchain": it recorded the *design* — seed base, attempt ceiling, filter
names — and nothing that could reproduce anything. It now pins the interpreter,
NumPy, the bit generator by name, the three sources the walk depends on **by
content**, and the commit. The commit is recorded but **not enforced**: a frozen
population stays valid across later commits that do not touch the walk, and the
source pins refuse exactly when it changed.

### 9.3 🔴 THE PIN THAT INVALIDATED ITS OWN ARTIFACT (2026-09-17)

The freeze is two steps — write the artifact, then record its digest — and the
provenance pin covered the file the second step must edit.

```
freeze  →  the artifact pins h3_study_rules.py, OPENING_SET_DIGEST = None
           sha256 4f970cc44d8d751a…
pin     →  record the digest in h3_study_rules.py
           sha256 7d720a5cb47313c3…
play    →  load_opening_set REFUSES: source_pins drift
```

**The population invalidated itself on the one edit its own procedure required,
and nothing could ever have been played.** A pin that covers the file it will be
written into is a trap, not a pin.

**The rule, now enforced structurally: AN OUTPUT MAY NEVER LIVE INSIDE A PINNED
INPUT.** `OPENING_SET_DIGEST` was the instance that bit; the retired seed ranges
and the destination paths are the same shape — each moves after a run, and
pinning any of them would make a routine retirement invalidate every frozen
population.

`h3_generation_protocol.py` now holds everything that determines what the walk
**produces** and nothing that records what it produced. The pinned set is that
module, `game/twixt_state.py` and `d1_selection.py` — plus the resolved exclusion
set **by value**, because the population depends on *which* 28 digests are
excluded, not on the code that computed them.

🔑 **The spent-range refusal deliberately stayed out.** It lives in
`h3_study_rules.attempt_seed`, which delegates the arithmetic to the protocol, so
retiring a range changes a module no artifact pins. The loop takes the guard as
an injected callable: policy binds without joining the pinned surface.

**And the identity split is now a stated policy.** `python` and `numpy` were
recorded and silently unenforced while §9.2 claimed identity was checked exactly
and only `commit` was informational.

| field | status | why |
|---|---|---|
| `numpy` | **ENFORCED** | NumPy guarantees stream compatibility for the legacy `RandomState` and **not** for `Generator`/`PCG64`. A version bump may change every opening; refusing is fail-closed |
| `python` | recorded only | `legal_moves()` builds its list with nested `range` loops, so its order does not vary with the interpreter, and `PCG64` is NumPy's |
| `commit` | recorded only | a frozen population stays valid across commits that do not touch the walk; `source_pins` refuses exactly when it did |

A test asserts every identity field is in exactly one of the two sets, so a new
field cannot arrive unclassified.

### 9.4 ✅ THE POPULATION AND THE SEEDS ARE BOTH FIXED (2026-09-17 / 18)

| step | when | result |
|---|---|---|
| population **frozen** | 2026-09-17 | one attempt, exit 0, verdict OK, 296/296 in 0.667 s |
| digest **pinned** | 2026-09-18 | `35932b3f…f772e46`, read off the artifact |
| match block **registered** | 2026-09-18 | `[202626000, 202626592)`, ACCOUNTED only |

**The block**: 592 seeds, one per game, bound positionally (row *i* → `lo + i`)
across four contiguous 148-seed quarters. A pair's two arms are adjacent and
share a segment, so no pair can be split across a VOIDed one. Collision proof
**v14** — 0 direct, 0 derived-stream, injective, nearest boundary **1,960** vs a
gap floor of **592**, **15/15** controls rejected, candidate excluded by identity
in the gap check as well as the overlap check.

🔴 **Four generation ranges are prior and no registry holds one of them** — three
retired and one LIVE. v14 adds all four by hand, and three of its controls sit
inside them so the addition is proved to matter.

🔴 **Pinning the seeded schedule exposed a check that could not fail.**
`run_segment` passed `want_digest=segment_digest(tasks, segment)` — the digest
computed from the very tasks it then handed to the checker. One source, two
sides, agreeing unconditionally. `SCHEDULE_DIGEST` and `SEGMENT_DIGESTS` give the
comparison a second, independent side.

**Registration authorized nothing.** ACCOUNTED is not EXPOSED and not RETIRED; a
reservation is not a draw. All ten gates and the freeze barrier stay `False`.

### 9.5 What is still not done

| | |
|---|---|
| `OPENING_SET_DIGEST` | ✅ **pinned** |
| `STUDY_SEED_BLOCK` | ✅ **`[202626000, 202626592)`, ACCOUNTED** |
| the seeded schedule and per-segment pins | ✅ **recorded** |
| the four segment output directories | **absent** |
| all ten gates, the freeze barrier included | **False** |

**The only preparation left is the study gate itself**, which is a separate
authorization. Preflight reports **zero failures and zero pending items**.

🔑 **AND FREEZING IS TWO STEPS, NOT ONE.** `h3_freeze_command` writes the
artifact; recording its digest as `OPENING_SET_DIGEST` is a **separate reviewed
edit**, made after the artifact has been inspected. Freezing produces the
population; pinning is what makes the study play it. Both are unauthorized.

**Official opening generation, seed registration, study execution and the push
all remain separate and UNAUTHORIZED.** Building the population in memory is
free and the suite does it on every run; writing the official artifact and
fixing its digest is not.
