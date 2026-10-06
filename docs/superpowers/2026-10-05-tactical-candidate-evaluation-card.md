# Exact one-ply tactics as a candidate — evaluation design card (DESIGN ONLY)

**Date:** 2026-10-05. **Status:** design for review. **Authorizes nothing executable:** no code,
gate, corpus generation, model load, pilot, game, seed registration or push.

**Question.** Does exact one-ply tactical handling, applied to our agent's readout, make the agent
stronger against the unchanged incumbent? The closed H5 tactics diagnostic (`1ec87ac5`) answered a
narrower question: rule M1 changes reproduced baseline decisions to exact wins or defences, sometimes
choosing an unvisited move. It showed no strength.

**Programme.** Five checkpoints, each separately authorized:

1. correctness;
2. qualification;
3. a fresh tactical corpus;
4. feasibility;
5. one strength test.

It ends in **accept**, **reject** or **inconclusive**. No automatic deployment and no further tuning
follows any outcome.

---

## 1. The candidate — frozen, one change only

The checkpoint, network, 400-simulation search, priors, FPU and normal readout are unchanged. One
**default-off wrapper** acts **after the unchanged search has finished**, on the root visit counts:

| position (protocol semantics, cap 280) | candidate's move |
|---|---|
| an own immediate win exists | the most-visited winning move |
| no own win, and the opponent threatens | the most-visited move that removes **every** immediate winning reply |
| threats exist, but no move removes them all | the baseline move |
| no immediate threat | the baseline move |

* **Rule.** This is exactly M1 of the H5 tactics card (§3.1, §10 A1.2, §11 A2): `classify` → admissible
  set A → `m1_choice`, under **protocol** semantics only.
* **Tie-break.** The existing coordinate tie-break: visits descending, then (row, col).
* **Baseline already admissible.** If the baseline move is in A, it is preserved. It is then the most
  visited in A, so this follows from the rule.
* **Zero visits.** An admissible move with zero visits stays eligible. It is recorded as
  **unsearched**: tactically necessary, never examined by search.
* **Board.** Physical and playable board both 24×24, enforced exactly as in §11 A2.1: any other board is
  refused.
* **Cap.** The protocol cap is 280. The winner is checked first, so a winning 280th move wins. The
  search keeps its existing **board-only continuation**, unchanged.
* **Excluded:** M2, any training change, extra simulations, parameter tuning.
* **The wrapper is read-only.** It reads counts and the state. It draws from neither random stream, and
  changes neither the tree nor the configuration.
* **Identity.** The candidate's identity is the incumbent identity plus the wrapper module's hash.

**Proposed implementation reuse (decide at implementation):** import the qualified H5 rule code
(`classify`, `admissible`, `m1_choice`) unchanged rather than copy it. That module is evidence-bound and
stays unchanged.

---

## 2. Checkpoint 1 — implementation and synthetic correctness

### Synthetic controls

All on 24×24 at the real cap. Positions come from rules-engine-only playouts. Invalid boards appear
only in refusal tests. The controls:

* an own win takes precedence over defence;
* multiple threats with one common defence;
* alternative valid defences;
* undefendable and quiet positions are unchanged;
* zero-visit winning and saving moves (chosen, flagged unsearched);
* the real 279/280 boundary;
* invalid board sizes refused;
* the tree, the configuration and both random streams are untouched;
* a baseline move already in A is preserved.

### Independent checker (§2.1, point 5)

The optimized detector (`CC.immediate_wins`, `CC.threats_against`, `WR.saving_moves_of`) is checked
against an **exhaustive rules-engine scan**:

* for every legal move, apply it and test for a winner;
* for every legal mover move *m*, and every legal opponent reply *r* after *m*, test whether *r* wins.

The check must use no shared helper. It also confirms the detector's assumption that the mover's own
peg never creates a new opponent win.

* **Cost.** O(L²) per position, about 90,000 move trials at L ≈ 300. The freeze classified about 19,300
  positions in about 263 s, roughly 14 ms each for about 600 trials, so about 23 µs per trial. That gives
  an estimated **≈ 2 s per position**, to be measured.
* **Where it runs.** In **qualification only**, never in the production play loop.

### Regression and scope

* **Regression.** The H5 fixtures E1–E3 and CD1–CD2, with their M1 predictions, become regression
  tests. Passing them is **required** and is **not** fresh improvement evidence.
* **Untouched:** frozen H4/H5 code, evidence and the closed tactics diagnostic.

---

## 3. Implementation prerequisites — separate from the wrapper

**R1. A new two-agent runner.** The H4/H5 runner is built around the T1j adapter and cannot play
agent against agent. A new runner must provide:

* one fresh `SeededReferenceAgent` per colour per game;
* the seed coupling of §7.1;
* the 280-cap loop, winner first;
* full move records;
* an **override witness** at every candidate decision: ply, class, baseline move and visits, final move
  and visits, the unsearched flag, and both root-mover values;
* per-game digests;
* the run contract already proven in H5: create-only evidence, a one-shot claim, the hard watchdog
  killing the process group, a journal, a record on every exit, and the conclusion inside the recorded
  boundary;
* blinding: no score is printed until every required segment is committed;
* its **own gate(s)**, created closed.

**R2. A no-game production qualification**, separate from R1's implementation. On fixed positions
with the real model, and without playing games, it shows:

* the runner's agents reproduce the incumbent's pinned decisions;
* the candidate equals the incumbent wherever the rule is inactive;
* both random streams follow §7.1;
* the 279/280 boundary and the board refusal work on the production path.

**Proposed names (frozen at implementation authorization):**

| module | gate |
|---|---|
| `tactical_candidate.py` (wrapper, qualification corpus, qualification run) | `TACTICAL_CANDIDATE_QUALIFICATION_AUTHORIZED` |
| `candidate_match_runner.py` (pilot and study) | `CANDIDATE_MATCH_AUTHORIZED` |

Evidence would go under `evidence/tactical-candidate-*`. Gates 25 and 26 are counted by the inventory
test, not by hand.

---

## 4. Checkpoint 2 — a fresh tactical qualification corpus

* **Size.** At most **100 new 24×24 positions**, up to 20 per class: own immediate win; single
  defendable threat; multiple defendable threats; undefendable; quiet.
* **Generator** (point 6). Fixed, recorded and **rules-engine-only**. It is frozen in full before it
  runs:
  * stream names, which must be **new**. `draw24`, `tac24`, `undef24` and `tacwt24` produced the pinned
    test positions and are not reused. Proposed: `tq24-uniform:{seed}` and `tq24-pegs:{seed}`;
  * the seed range, i.e. the generation budget;
  * the move policy:
    * uniform random legal; and
    * a **frozen bias**: with probability 0.7, a random legal move a knight's step from one of the
      mover's own pegs. Without it, own-win-plus-threat positions are very rare;
  * the scan window, e.g. 1–20 plies before a win;
  * the ply bound, e.g. ≤ 270, so every reply lands inside the cap;
  * the class rule (protocol);
  * selection: one representative per (class, game), then a seeded sample of at most 20 games per class.
* **Shortages** are recorded, never filled by changing the procedure.
* **Exclusion by state digest** of every previously examined position:
  * the H5 fixtures and controls (10);
  * the H5 tactics corpus's 63 selected entries and its 23 fixture-equivalent positions;
  * every pinned test position.
* **Freeze.** The corpus is frozen and committed before any model load.
* **Run.** At each position, baseline and candidate use matching fresh-agent seeds. The searches must
  be identical: counts, tie set, fingerprint and both streams. Only the permitted readout may differ.
* **Reported:** tactical correctness, overrides, unsearched choices, timing.
* **Realism.** Random playouts give cluttered, late positions (plies 250+). The H5 threat positions were
  at plies 35–206. These positions establish **coverage**, **not** representative play or strength.

---

## 5. Checkpoint 3 — operational feasibility (pilot)

A small, separately authorized pilot of candidate against incumbent. **Pilot games never enter the
strength estimate.**

**Frozen before the pilot:**

* maximum runtime: per game, per segment, and the watchdog limit;
* acceptable tactical-scan overhead, as a share of decision time;
* recording, replay and process-integrity requirements;
* rules for operational failure;
* an excessive-cap threshold;
* the **sizing rule of §7.3**, complete.

**Measured:**

* **full-game runtime**, including both agents and the candidate's scans. The rough planning figure is
  about 2 minutes per game: about 1.7 s per decision in the H5 tactics run, times about 67 plies (H5 games
  averaged about 33 incumbent decisions). Self-play length may differ.
* the override firing rate, per game and per pair;
* the cap rate;
* replay integrity: §7.1's exact-replay condition holds on every no-override pair.

**The pilot decides only whether the study is practical.** It cannot:

* select or alter the candidate;
* resize the study after inspecting scores;
* drop the tactical scans to save time.

An unacceptable cost means **stop and review**.

---

## 6. Checkpoint 4 — scope (points 4 and 7)

* **Opponent.** This tests the candidate **against the incumbent**, not against T1j. The incumbent
  itself sometimes misses one-ply wins; the H5 weakness review found its misses were never punished. So
  a defence may be worth less in self-play than against an opponent that converts every win. Strength
  against T1j is a separate question.
* **Unsearched choices** (point 7). A zero-visit win or defence is chosen because the rules require it,
  not because search examined it. Every such choice is recorded as **unsearched**. The strength test is
  what prices that risk. Tactical correctness never implies stronger play.

---

## 7. Checkpoint 5 — one fresh strength comparison

**Games.** Candidate against the unchanged incumbent, from the empty board, in adjacent colour-reversed
pairs:

* **Arm A:** candidate Red. **Arm B:** candidate Black.
* Same checkpoint and 400 simulations per decision.
* Fresh, collision-checked seed bundles.
* Full move records and override witnesses.
* A capped game scores 0.5. Repeated trajectories keep their multiplicity.
* The sample size is frozen **before any evaluation game exists**. Scores stay blinded until every
  required segment is committed.
* An inconclusive study is **never** extended, never pooled with H4/H5 or the pilot, and never read as
  "tactical correctness ⇒ stronger".

### 7.1 Seed coupling — exact

The agent code fixes the streams. Search stream = `Random(seed ^ SEARCH_MASK[colour])`; readout
stream = `Random(seed ^ READOUT_MASK[colour])`. Under the argmax readout, `eval_readout.select` returns
`_argmax(counts)` and **never draws** from the readout stream. It is recorded as inert.

| follows | what |
|---|---|
| **colour** | the seed. Pair *k* has one bundle (*s_red(k)*, *s_black(k)*). In **both** games of the pair, whichever role plays red uses *s_red(k)*, and whichever plays black uses *s_black(k)*. |
| **game** | the agent instance. Each game builds one fresh agent per colour, so both streams restart at every game and advance across that game only. |
| **role** | **nothing.** Being candidate or incumbent never changes a seed, mask or stream. |

* **Before the first override.** The wrapper draws nothing and the evaluator is deterministic. (The H5
  fixture fingerprints reproduced exactly across processes; most recently they matched the `912b601f`
  pins in `1ec87ac5`.) So each game follows the **incumbent-only game *G₀(k)*** — incumbent red on
  *s_red(k)*, incumbent black on *s_black(k)* — move for move, until its first override:
  * game A deviates from *G₀(k)* at the first **red** decision where the rule changes the move;
  * game B deviates at the first **black** such decision.
* **Exact-replay condition.** If the rule changes no move in either game of pair *k*, then A = B =
  *G₀(k)* move for move, and the pair scores **exactly 0.5**.
  * This is **checked**, not assumed. A no-override pair whose games differ in any move means the
    determinism assumption has failed: **STOP**.
  * More generally, A and B must agree up to the earlier of their two first overrides.
* **Role-following seeds would break all of this**, for example the candidate always receiving the
  first seed. They are not allowed.

### 7.2 The primary estimand — candidate-vs-incumbent pairs against the matched counterpart (UNRESOLVED, an approval decision)

**(P) Pairs.** The pair score is the mean of the candidate's two game scores, in [0, 1]. The estimand is
its mean, against 0.5. This is the H4/H5 continuity design.

**(C) Matched incumbent-only counterpart.** Also play *G₀(k)* for every pair. Per arm:

* Δ_A = s_cand(A) − s_red(G₀);
* Δ_B = s_cand(B) − s_black(G₀).

Because s_red(G₀) + s_black(G₀) = 1 in every game (a cap is 0.5 each):

> (Δ_A + Δ_B) / 2 = pair score − 0.5 — **identically**, pair by pair.

What follows:

* **Pooled effect.** For the pooled effect, (C) is the *same statistic* as (P). It adds **no power**.
* **What (C) adds:**
  * per-colour attribution. Δ_A alone is the effect as red, controlled for red's advantage; (P) cannot
    separate a per-colour effect from colour advantage;
  * a direct replay check, every game against its counterpart.
* **What (C) costs:** **+50 % games**.
* **The power question** therefore lies in the statistical method (§7.3), not in (P) against (C).

**(K) Conditional on firing — LABELLED CONDITIONAL.**

* **Firing** = the rule changes at least one move in game A or game B. Whether a game fires is decided
  by *G₀(k)* alone (§7.1), before the candidate's play diverges.
* **Validity.** Conditioning on it is therefore valid inference about the **firing pairs**.
* **Relation to the overall effect.** Under exact replay, non-firing pairs contribute exactly 0, so
  overall − 0.5 = P(fire) × (conditional − 0.5): the two share their sign.
* **The rule.** A (K) result is reported **as conditional**. It never stands in for overall strength on
  its own. Any overall statement needs (P) or (C), with P(fire) and its uncertainty shown.

### 7.3 Sizing — a rule frozen before the pilot (an approval decision)

The firing rate alone cannot set power. A sizing rule needs, **all declared before the pilot**:

* **δ**, the effect worth detecting, on the pair-score scale;
* the **statistical method**;
* α = 0.05, two-sided;
* a **compute ceiling** in hours.

**Hoeffding, all pairs (continuity).**

* Half-width h = √(ln(2/α) / 2n), so n = ⌈ln 40 / (2δ²)⌉.
* This **does not depend on the firing rate**. The pilot cannot change n. It only checks
  n × 2 games × t_game ≤ ceiling; otherwise **stop and review**, never shrink.
* Hoeffding uses no variance, so it is conservative when many pairs score exactly 0.5.

| δ (half-width) | pairs n | games | rough compute at ~2 min/game (pilot measures) | with counterparts (C) |
|---|---|---|---|---|
| 0.079 | 296 | 592 | ~20 h | ~30 h |
| 0.050 | 738 | 1,476 | ~49 h | ~74 h |
| 0.040 | 1,153 | 2,306 | ~77 h | ~115 h |
| 0.030 | 2,050 | 4,100 | ~137 h | ~205 h |
| 0.020 | 4,612 | 9,224 | ~307 h | ~461 h |

> *Half-widths only — precision, not power. Superseded by A1.1; see §12 A2.3.*

**Dilution, illustrated.** If 10 % of pairs fire and the firing pairs average 0.70, the overall mean is
0.52. Even 1,153 pairs (±0.040) would very likely report **inconclusive**.

**Variance-adaptive alternative** (e.g. empirical Bernstein). It exploits the exact-0.5 pairs and can be
much tighter. It is **not** the H4/H5 method, so choosing it must be a deliberate decision. Its sizing
formula, and its dependence on the pilot's measured variance, must be frozen first.

**Conditional (K).** Play until **n_f firing pairs**, n_f = ⌈ln 40 / (2δ_c²)⌉ for a declared
conditional effect δ_c, or until the compute ceiling.

* The stopping count is outcome-blind, and "fired" is decided by *G₀*, so stopping on it is allowed.
  (H3 lesson: a monotone *count* may decide; a ratio may not.)
* If the ceiling comes first, the result is reported as **underpowered**, and the decision rules stay
  the same.

**Decision rules** (bounded-pair Hoeffding, independence of pairs assumed and stated):

* lower bound strictly above 0.5 → evidence of **improvement** under this protocol;
* upper bound strictly below 0.5 → evidence of **regression**;
* otherwise **inconclusive**.

### 7.4 Seeds

Fresh bundles are registered in the shared registry (`e4_screen_reference.py`), with a collision proof
by the established method: the v15/v16 provenance, extended for underscored literals. Registration is
its own authorization. Pilot and study seeds are disjoint.

---

## 8. Approval decisions this card leaves open

| # | decision | options set out here |
|---|---|---|
| D1 | the final comparison design | (P) pairs; (C) with counterparts; (K) only as a labelled conditional analysis alongside |
| D2 | the statistical method | Hoeffding (continuity) or a variance-adaptive bound |
| D3 | δ, hence n (or n_f), and the compute ceiling | the §7.3 table |
| D4 | runtime limits | per game, per segment, watchdog, scan-overhead share, cap-rate threshold |
| D5 | pilot size | small; never in the estimate |
| D6 | names and gates | §3 proposal |

## 9. Order of authorizations

1. This card's review.
2. Implementation: the wrapper, synthetic controls and the independent checker.
3. R1: the runner implementation.
4. R2: the no-game production qualification.
5. Freeze of the qualification corpus.
6. The qualification run (model load).
7. Decisions D1–D5 and the frozen sizing rule.
8. The pilot.
9. Seed registration.
10. The study segments.
11. Analysis → accept, reject or inconclusive.

Each step is separately authorized. Frozen H4/H5 code and evidence, curriculum support and `TwixtState`
stay untouched throughout.

## 10. Scope

This card authorizes nothing executable: no code, gate creation or opening, corpus generation, model
load, pilot, game, seed registration or push.

---

## 11. Amendment A1 — 2026-10-05, statistical corrections before implementation (user-decided)

Review of `513034dd` accepted:

* the seed coupling (§7.1);
* the counterpart algebra (§7.2);
* the module and gate names (§3).

It corrected three statistical points and §9's order, and recorded decisions.

### A1.1 — Precision is not power (supersedes §7.3's table and its double use of δ)

§7.3 used δ for two different quantities.

* **h(n)** is the interval half-width at n: precision. §7.3's table gives h(n), not power.
* **Δ** is the true overall effect worth detecting, on the pair-score scale: μ − 0.5.

At a true effect Δ = h(n), the lower bound clears 0.5 only about half the time. So a ±0.040 interval
is **not** an adequately powered design for a true +0.040. This also corrects this programme's earlier
1,153-pair suggestion.

**Power** = P(lower bound > 0.5 | μ = 0.5 + Δ), at α = 0.05 two-sided. The target is **80 %** (proposed;
part of D3). For **sizing only**, X̄ is treated as normal with standard deviation σ/√n, and the
empirical-Bernstein sample variance is planned at σ². No approximation enters a verdict.

* **Hoeffding:** n = ⌈((√(ln 40 / 2) + z·σ) / Δ)²⌉, with √(ln 40 / 2) = 1.3581 and z = z₀.₈₀ = 0.8416.
* **Empirical Bernstein (A1.5):** n = the smallest n ≥ 2 with
  Δ ≥ √(2σ² ln 80 / n) + 7 ln 80 / (3(n − 1)) + z·σ/√n.

**Computed for the worst case**, σ = 0.5, with no pilot input. The cost column assumes about 4 min per
pair (2 games × about 2 min). It is rough, and the pilot measures it.

| Δ | Hoeffding pairs | ≈ compute | EB pairs | ≈ compute |
|---|---|---|---|---|
| 0.079 | 508 | ~34 h | 818 | ~55 h |
| 0.050 | 1,266 | ~84 h | 1,832 | ~122 h |
| 0.040 | 1,978 | ~132 h | 2,747 | ~183 h |
| 0.030 | 3,517 | ~234 h | 4,673 | ~312 h |
| 0.020 | 7,912 | ~527 h | 10,032 | ~669 h |

> *Planning approximation (§12 A2.3): n is computed from a normal approximation for the mean of the pair scores (empirical Bernstein additionally plans Vₙ at σ²). It is **not** a distribution-free 80 % power guarantee, and valid interval coverage does not make the power calculation exact.*

### A1.2 — The pilot stays outcome-blind (supersedes "the pilot's measured variance" in §7.3)

* **Nothing outcome-derived is read.** No pilot score, winner, per-colour result or score variance is
  computed, printed or read. The records hold results; they are not read.
* **What the pilot report reads:**
  * per-game and per-pair wall time;
  * the termination type, win or cap — never the winner;
  * the override witnesses;
  * the move lists, for the exact-replay check.
* **Firing is outcome-blind.** A pair fires if, at any candidate decision in either game, the final move
  differs from the baseline move. This is read from the witnesses, not from any result.
* **Variance enters sizing only through an outcome-blind bound.**
  * Under exact replay (§7.1), a non-firing pair scores exactly 0.5. So
    Var(pair) ≤ E[(pair − 0.5)²] ≤ p/4, where p = P(a pair fires).
  * The pilot's firing count f out of 16 pairs gives **p_U**, the one-sided 95 % Clopper–Pearson upper
    bound. That accounts for the uncertainty in p.
  * σ_plan = min(0.5, √p_U / 2).
* **Preconditions:**
  * every non-firing pilot pair is move-identical (§7.1). Otherwise the bound is void: **STOP and
    review**, with no fallback chosen after the fact;
  * f ≥ 1. If f = 0, the candidate never acted in the pilot: **STOP and review** before sizing.

### A1.3 — A fixed sample means a fixed sample (supersedes §7.3's "(K) play until n_f … or ceiling" and its "underpowered" clause)

* **n is computed once**, by the frozen rule (A1.4), from the pilot firing count. It is committed before
  any study seed is registered and before any study game exists.
* **Stopping early means no verdict.** If the study stops before all n pairs are committed (compute,
  failure, interruption), the result is **INCOMPLETE: no confirmatory verdict**. That means no
  improvement, no regression, and not even "inconclusive"; only descriptive integrity reporting.
* **No interim look at scores.** Scores stay blinded until all n pairs are committed.
* **No time-uniform method is adopted.** Inference at arbitrary stopping times would need one (an
  anytime-valid confidence sequence; Howard et al., arXiv:1810.08240).
* **(K) has no stopping rule.** It is a **secondary**, **conditional** analysis, on the firing pairs of
  the complete study only, and labelled as such.
* **The compute ceiling is checked before the study starts** (A1.4 step 5). A study whose planned
  compute exceeds it is never started.

### A1.4 — The sizing rule, frozen before the pilot

Δ, the power target and the ceiling stay open (D3). Everything else is fixed:

1. **Declared before the pilot:**
   * Δ, overall, on the pair-score scale;
   * the power target, 80 % proposed;
   * α = 0.05, two-sided;
   * the method (D2);
   * the compute ceiling C, in hours.
2. **Pilot, 16 pairs:** f = the number of firing pairs; t̄ = the mean wall time per pilot pair, both
   games and the scans included.
3. p_U = the one-sided 95 % Clopper–Pearson upper bound for f of 16; σ_plan = min(0.5, √p_U / 2).
4. n = the A1.1 formula of the frozen method, at (Δ, σ_plan).
5. **The compute check:** n × t̄ × 1.25 ≤ C. Otherwise **STOP and review**. n is never reduced to fit.
6. **Preconditions (A1.2):** f ≥ 1, and every non-firing pilot pair replays exactly.
7. n is committed in a record before any study seed or game.

**Illustration** (computed; pairs needed, Hoeffding / EB):

| pilot f of 16 | p_U | σ_plan | Δ 0.05 | Δ 0.04 | Δ 0.03 | Δ 0.02 |
|---|---|---|---|---|---|---|
| 0 | 0.171 | 0.207 | (stop: f = 0) | — | — | — |
| 1 | 0.264 | 0.257 | 992 / 734 | 1,550 / 1,046 | 2,754 / 1,673 | 6,197 / 3,329 |
| 2 | 0.344 | 0.293 | 1,031 / 858 | 1,610 / 1,236 | 2,862 / 2,005 | 6,439 / 4,065 |
| 4 | 0.484 | 0.348 | 1,091 / 1,071 | 1,704 / 1,564 | 3,029 / 2,582 | 6,815 / 5,351 |
| 8 | 0.721 | 0.425 | 1,178 / 1,423 | 1,840 / 2,110 | 3,271 / 3,546 | 7,358 / 7,506 |
| 16 | 1.000 | 0.500 | 1,266 / 1,832 | 1,978 / 2,747 | 3,517 / 4,673 | 7,912 / 10,032 |

> *Planning approximation (§12 A2.3): n is computed from a normal approximation for the mean of the pair scores (empirical Bernstein additionally plans Vₙ at σ²). It is **not** a distribution-free 80 % power guarantee, and valid interval coverage does not make the power calculation exact.*

* A declared overall Δ with a small p implies a large effect per firing pair. For example, Δ = 0.04 at
  p = 0.26 needs firing pairs to average about 0.65.
* The rule does not judge whether Δ is plausible. That is the D3 decision.

### A1.5 — D2: the empirical-Bernstein interval, specified for review (Maurer & Pontil 2009, Theorem 4)

**The source** (arXiv:0907.3740 v1; read from the paper, Theorem 4 under the conditions of Theorem 1):

* Z, Z₁…Zₙ are **i.i.d.**, with values in **[0, 1]**; n ≥ 2; δ > 0.
* With probability at least 1 − δ:
  E Z − (1/n) Σ Zᵢ ≤ √(2 Vₙ(Z) ln(2/δ) / n) + 7 ln(2/δ) / (3(n − 1)).
* Vₙ(Z) = (1 / (n(n − 1))) Σ_{i<j} (Zᵢ − Zⱼ)². This equals the unbiased sample variance, with
  denominator n − 1.
* The bound is **one-sided**. The other side follows by replacing Z with 1 − Z.

**As applied:**

* Zᵢ = the pair score of pair i, one of {0, ¼, ½, ¾, 1}.
* Two-sided 95 %: each side at δ = 0.025, so ln(2/δ) = ln 80.
* 1 − Zᵢ has the same Vₙ. The interval is:

  > L = Z̄ − √(2 Vₙ ln 80 / n) − 7 ln 80 / (3(n − 1)); U = Z̄ + the same two terms.

* The decision rules are unchanged and apply **at the frozen n only**:
  * L > 0.5 → improvement;
  * U < 0.5 → regression;
  * otherwise inconclusive.
* **Stated assumptions:**
  * pairs i.i.d. — the same independence assumption as H4/H5, since the registered seeds are treated as
    an i.i.d. draw; *[corrected by §12 A2.5: identical distribution is an ADDITIONAL modelling assumption, not the H4/H5 one]*
  * scores bounded in [0, 1];
  * n fixed in advance (A1.3).
* **The estimand is unchanged:** E[pair score], design P.

**It is not automatically better** (computed, A1.1 and A1.4):

* With no firing information (σ = 0.5), EB needs **more** pairs than Hoeffding, because of its constants.
  For Δ = 0.04: 2,747 against 1,978.
* It needs fewer only when the planned variance is small: roughly f ≤ 4 of 16 pilot pairs firing.
* D2 remains the user's choice. An option, **not adopted:** pre-specify "the method with the smaller n
  under A1.4". It would be chosen from the outcome-blind firing count, before any study game.

### A1.6 — Decisions (supersedes §8's status)

| # | decision | status |
|---|---|---|
| D1 | **P**: all colour-reversed pairs, no counterpart games ((C) dropped); (K) secondary, conditional and labelled only | **decided** |
| D2 | the empirical-Bernstein interval of A1.5, or Hoeffding | **proposed for review**; see the trade-off in A1.5 |
| D3 | Δ, the power target and the compute ceiling C | **open** (user) |
| D4 | runtime limits: per game, per segment, watchdog, scan-overhead share, cap-rate threshold | **open** (user); no spending limit is inferred from earlier studies |
| D5 | pilot = **16 pairs** (32 games), excluded from the estimate | **decided** |
| D6 | the §3 module and gate names | **accepted** |

### A1.7 — Order of authorizations (supersedes §9)

1. Review of this card with A1.
2. Implementation: the wrapper, synthetic controls and the independent checker.
3. R1: the runner implementation.
4. **Freeze of the fresh qualification corpus** — rules-engine only, before any real-model load (§4).
5. R2, the no-game production qualification, and the tactical qualification run on the frozen corpus.
   These are the first real-model loads.
6. D2–D4 decided; the A1.4 sizing rule frozen, complete.
7. **Pilot seed registration.**
8. The pilot: 16 pairs, outcome-blind (A1.2).
9. n computed by the frozen rule, compute-checked and committed.
10. Study seed registration (separate).
11. Study segments, blinded.
12. Analysis → accept, reject or inconclusive — or **INCOMPLETE** (A1.3).

Each step is separately authorized.

### A1.8 — Scope

Design amendment only. No implementation, gates, corpus generation, model loading, seed registration,
games or push.

---

## 12. Amendment A2 — 2026-10-05, decisions D2–D4 and qualifications (user-decided)

These choices set **design limits, not permission to spend compute.** D1, D5 and D6 stand as decided
in A1.6.

### A2.1 — D2: a predeclared method-selection rule (supersedes A1.5's "D2 remains the user's choice")

Frozen before the pilot:

1. From the **outcome-blind** pilot inputs only — f, hence p_U and σ_plan (A1.2, A1.4) — compute the
   planned n for **Hoeffding** and for the **empirical-Bernstein interval of A1.5**. Use the A1.1
   formulas at Δ and σ_plan.
2. **Choose the smaller n**; a tie chooses **Hoeffding**.
3. **Commit** the chosen method and n before study seed registration.
4. **Never** choose between intervals after any study outcome exists.

This is the option A1.5 described as "not adopted", now adopted.

### A2.2 — D3: bounded scope

* **Effect worth detecting: Δ = 0.05**, i.e. a candidate pair-score mean of 0.55.
* **Planning power target: approximately 80 %** (A2.3).
* **Study compute ceiling: 80 hours**, *including* study setup and verification, across all segments.
* **The projection keeps the 25 % margin:**

  > projected = 1.25 × (n × t̄ + setup and verification, estimated from the pilot)

  If projected > 80 h: **stop before starting**. n is never shrunk.
* **Scope.** This first programme may miss smaller improvements. **Inconclusive is an acceptable
  result.**

**What this implies (computed).** At Δ = 0.05 and 80 % planning power, by pilot firing count f of 16.
The last column ignores setup and verification, so the real limit is lower.

| f | p_U | σ_plan | n Hoeffding | n EB | chosen (A2.1) | n | longest mean pair time t̄ that fits 80 h |
|---|---|---|---|---|---|---|---|
| 0 | 0.171 | 0.207 | — | — | **STOP** (A1.2: f = 0) | — | — |
| 1 | 0.264 | 0.257 | 992 | 734 | EB | 734 | 5.23 min |
| 2 | 0.344 | 0.293 | 1,031 | 858 | EB | 858 | 4.48 min |
| 3 | 0.417 | 0.323 | 1,063 | 969 | EB | 969 | 3.96 min |
| 4 | 0.484 | 0.348 | 1,091 | 1,071 | EB | 1,071 | 3.59 min |
| 5 | 0.548 | 0.370 | 1,116 | 1,167 | Hoeffding | 1,116 | 3.44 min |
| 8 | 0.721 | 0.425 | 1,178 | 1,423 | Hoeffding | 1,178 | 3.26 min |
| 12 | 0.910 | 0.477 | 1,239 | 1,700 | Hoeffding | 1,239 | 3.10 min |
| 16 | 1.000 | 0.500 | 1,266 | 1,832 | Hoeffding | 1,266 | 3.03 min |

> *Planning approximation (§12 A2.3): n is computed from a normal approximation for the mean of the pair scores (empirical Bernstein additionally plans Vₙ at σ²). It is **not** a distribution-free 80 % power guarantee, and valid interval coverage does not make the power calculation exact.*

**Read honestly:** at exactly 4 min per pair (the rough planning figure of §5), the study fits the
ceiling only if **f ≤ 2**. At f = 3, 969 × 4 min × 1.25 = 80.75 h, already over 80 h before setup. The
table's 3.96-minute limit for f = 3 is correct. Otherwise the study **stops before starting**, as A2.2
requires. The pilot's measured t̄ decides; nothing is shrunk to fit. *(Corrected on review: an earlier
version said f ≤ 3.)*

### A2.3 — The sizing tables are approximations (qualifies §7.3, A1.1, A1.4 and A2.2)

* Every planned n comes from a **normal approximation** for the mean of the pair scores. The
  empirical-Bernstein n additionally plans the sample variance at its planned value σ².
* **Neither is a distribution-free 80 % power guarantee.** For example, 1,978 pairs for Δ = 0.04 is
  correct under the A1.1 planning formula only.
* **Valid final interval coverage does not make the power calculation exact.**
* These qualifiers are annotated beside every sizing table above.

### A2.4 — D4: operational limits (fixed)

* **Per game: 15 minutes, hard-enforced** by the runner (R1). An overrun is an operational failure.
* **Per segment: 4 hours**, setup included.
* **The entire pilot: 4 hours.**
* **Tactical scans:** at most **25 % of the aggregate candidate decision time**, measured over the
  completed pilot. Above that: **stop and review**. Scans are never skipped to save time.
* **Caps:** **stop** if **4 or more of the 16 pilot pairs** contain a capped game. This uses the
  termination type only, never the winner (A1.2).
* **Any operational or integrity failure:** stop, retain the evidence, **no automatic retry.**
* **The 80-hour ceiling applies across all segments.** An unfinished study is INCOMPLETE, with no
  confirmatory verdict (A1.3).

### A2.5 — The i.i.d. assumption, stated (corrects A1.5)

* **Empirical Bernstein** (Maurer & Pontil, Theorem 4, under the conditions of their Theorem 1) requires
  **i.i.d.** pair scores.
* **Identical distribution is an additional modelling assumption.** It goes beyond the H4/H5
  independence assumption, and registering fresh seeds does **not** establish it. If the rule selects
  empirical Bernstein, the verdict holds under **i.i.d. pairs, assumed**, and says so.
* **Hoeffding**, in its general form for independent variables bounded in [0, 1] (Hoeffding 1963),
  needs independence only. Its interval then concerns the average of the pairs' expected scores.
* Whichever method the rule selects, the record states that method's assumption beside the verdict.

### A2.6 — Decisions (supersedes A1.6's statuses)

| # | decision | status |
|---|---|---|
| D1 | P: all colour-reversed pairs; (K) secondary, conditional only | decided (A1.6) |
| D2 | the method-selection rule of A2.1 | **decided** |
| D3 | Δ = 0.05; ≈ 80 % planning power; an 80 h ceiling including setup and verification; the 25 % margin | **decided** |
| D4 | the limits of A2.4 | **decided** |
| D5 | pilot = 16 pairs, excluded from the estimate | decided (A1.6) |
| D6 | the §3 names and gates | accepted (A1.6) |

### A2.7 — Scope

Design amendment only. No implementation, gate change, corpus generation, model loading, seed
registration, pilot, study or push.

---

## 13. Amendment A3 — 2026-10-05, the recording contract of R1 hardened (user-decided)

Review of R1 (`4c32c9f8`) found three gaps. Each is a gap in the recording contract, not evidence
about any real game.

### A3.1 — The 15-minute game limit is enforced by the PARENT (supersedes §12 A2.4's "hard-enforced by the runner")

* **The SIGALRM alarm was not hard.** It needs Python to run its handler, so a blocking native call can
  overrun it, and the parent's watchdog enforced only the 4-hour segment.
* **The child marks each game.** Before each game, the child journals an outcome-blind `game_start`
  (pair and arm only).
* **The parent enforces the limit.** It polls about every 0.5 s. When a game has been open for
  **15 minutes** — measured on the parent's own clock from when it first saw the `game_start` — it
  SIGKILLs the child's whole process group. It records which game breached, and why.
* **The bound** is 15 minutes plus about **two** polling intervals past the game's real start.
  * The parent first detects the `game_start` up to one interval after it is written, and its
    clock starts only then.
  * It detects the expiry up to one interval after the limit passes.

  At 0.5 s that is about 1 s under normal scheduling. *(Corrected on review: an earlier version
  said one interval.)*
* **The child's alarm stays**, set 30 s earlier (14.5 minutes). A responsive child therefore records a
  clean STOP before the parent's kill.

### A3.2 — Completed observations are durable before the next one starts

* **Each completed decision** is appended and fsynced to `games/<pair>-<arm>.partial.jsonl`: ply, move,
  and the candidate's compact witness where it moved. This is evidence that carries outcomes. It is
  kept **apart from the blind journal**.
* **Each completed game** is written create-only and fsynced to `games/<pair>-<arm>.json` (directory
  fsynced too) **before the next game starts**.
* **record.json** lists the games by file, sha256 and digest.
* **watchdog.json**, written by the parent, inventories every file under `games/` by name, size and
  sha256, and records `record.json`'s sha256. The parent reads no file content.
* **What a kill can lose** is the decision in progress, never a completed one.

### A3.3 — A predecessor segment is validated, not trusted (supersedes §12 A2's label-only check)

Before study segment *k* starts, every segment *j < k* must pass, outcome-blind:

* **Durability:** its claim, journal, record, watchdog summary and every listed game file are committed.
* **The claim** names the committed plan's file sha256, the study role and segment *j*.
* **watchdog.json:**
  * COMPLETE / COMPLETE;
  * its journal sha256 equals `log.jsonl`'s bytes;
  * its record sha256 equals `record.json`'s bytes.
* **record.json:**
  * card, mode `study`, status COMPLETE;
  * its plan and plan hash equal the committed plan;
  * its games are exactly the plan's (pair, arm) sequence;
  * each listed game file is present, committed, and of the recorded sha256;
  * its replay checks name every planned pair.

No move, terminal or witness content is parsed. Any failure refuses the later segment. The 80-hour
ceiling sums the validated segments' parent-measured elapsed times.

### A3.4 — Scope

Contract amendment, then the repairs, their synthetic failure controls, the regression and a local
commit. Corpus generation, model loading, qualifications, research seeds, gate opening, real games
and push remain unauthorized.

---

## 14. Amendment A4 — 2026-10-05, the fresh qualification corpus frozen in full (supersedes §4's examples)

§4 named the generator's settings by example. A4 fixes every one of them before any code exists.

**Unchanged:**

* the official 24×24 board, enforced (§11 A2.1), and protocol semantics with cap 280;
* at most **20 positions per class**, **100 in total**;
* shortages are recorded and **never** filled by changing the procedure.

### A4.1 — Streams, seeds, budget

| stream | random generator per game | seeds |
|---|---|---|
| `tq24-uniform` | `random.Random(f"tq24-uniform:{seed}")` | 0 … 3,999 |
| `tq24-pegs` | `random.Random(f"tq24-pegs:{seed}")` | 0 … 3,999 |

* **Budget:** 8,000 games, generated in a fixed order — uniform stream first, seeds ascending, then the
  pegs stream.
* **No early stop.** Every seed is processed even once quotas could be met, so the universe never
  depends on the order of processing.
* **Stream names are new.** `draw24`, `tac24`, `undef24` and `tacwt24` are never used.
* **Toolchain.** Python's string seeding and `random.choice` / `random.sample` are frozen at the
  recorded interpreter: CPython 3.14.7, from `.venv`. The manifest records the version, and a
  reproduction needs the same one.

### A4.2 — Move policies (exact draw order)

Each game starts from the empty official board. Every move computes `legal = sorted(state.legal_moves())`.

* **Uniform:** `rng.choice(legal)`.
* **Pegs:**
  * `own` = every cell (r + a, c + b) for each peg (r, c) of the mover's colour, over the knight steps
    (a, b) in the order (1,2) (2,1) (−1,2) (−2,1) (1,−2) (2,−1) (−1,−2) (−2,−1);
  * `near = [m for m in legal if m in own]`;
  * **if `near` is non-empty:** draw `u = rng.random()`, then `rng.choice(near)` if `u < 0.7`, else
    `rng.choice(legal)`;
  * **if `near` is empty:** `rng.choice(legal)`, and **no** `u` is drawn.

A game ends at the first winner, at ply 280, or when no legal move remains.

### A4.3 — Which positions are scanned

* **Only games that end in a win contribute**, with `end` = the ply of the final state. Games ending at
  the cap, or with no legal move, contribute nothing.
* **The scan window** is every position at plies **max(10, end − 20) … min(end − 1, 270)**.
  * **≥ 10:** no early-opening coincidence with any other process.
  * **≤ 270:** every reply lands well inside the cap, so the protocol and board-only classes coincide.
* **Classification.** Each window position is classified by `classify(state, semantics="protocol")`,
  imported unchanged from `h5_incumbent_tactics`, into OWN_WIN, DEFEND_SINGLE, DEFEND_MULTI,
  UNDEFENDABLE or QUIET.

### A4.4 — The prior-position exclusion inventory (complete, enumerated)

A position whose `state_digest` appears in the inventory is removed **before** selection, and counted
per class as `excluded_prior`. The inventory is written to `inventory.json`, each digest with its
source(s); its sha256 goes into the manifest. Its sources:

| # | source | positions |
|---|---|---|
| a | `fixture_digests` of the frozen H5 fixtures and controls (E1–E3, CT1–CT3, CI1–CI2, CD1–CD2) | 10 |
| b | the committed H5 tactics corpus, sha256 `214b958c…`: every `selected` digest and every `fixture_equivalent` digest | 63 entries (55 distinct) + 23 |
| c | the committed H5 tactics study record (`1ec87ac5`): every row's position, by replaying its moves | 60 |
| d | **every prefix** — empty board to full sequence — of every pinned move sequence in the test suite: `DRAW24`, `OWN_WIN_279`, `THREAT_279`, `UNDEFENDABLE_256`, `OWN_WIN_AND_THREAT_264`, `SINGLE_260`, `MULTI_269` (tests/test_h5_incumbent_tactics.py), and `ONE` and `TWO` (tests/test_h5_cap_completion.py) | about 2,230 |

**Not enumerated, stated:** positions that test code derives *on the fly* and does not pin — one-ply
extensions inside refutation trees, and the runner tests' stand-in games. No research conclusion rests
on them, and they come from other streams and policies. With eligible plies ≥ 10, a coincidence is not
expected — but it is **not excluded by digest**. *(Review point: the reviewer may require them too.)*

### A4.5 — Deterministic selection

* **One representative per game and class.** For each class and each game (stream, seed) with a
  position of that class in its window — after exclusion — take one ply:
  `random.Random(f"tq24-select:{class}:{stream}:{seed}").choice(sorted plies)`.
* **At most 20 games per class.** Over that class's games sorted by (stream, seed):
  `random.Random(f"tq24-select:{class}").sample(games, min(20, len(games)))`.
* **A shortage** takes every game, and records `quota_shortfall`.
* **Order.** Classes in the order OWN_WIN, DEFEND_SINGLE, DEFEND_MULTI, UNDEFENDABLE, QUIET; then
  (stream, seed, ply).
* **Repeats.** A position whose digest repeats one already selected is marked `repeat_of` and decided
  once. It still counts in its class.
* **The independent cross-check.** Every distinct selected position runs `cross_check`, the exhaustive
  engine-only checker of §2.1. **Any disagreement STOPS the freeze**, recorded, with no substitution:
  it would mean the detector is wrong.

### A4.6 — Destination, runtime limit, failure recording

* **Code:** a new module, `tactical_corpus.py`. This amends §3, which put the corpus in
  `tactical_candidate.py`: kept apart, the wrapper's recorded hash never changes because of generator
  code.
* **Gate:** the existing `TACTICAL_CANDIDATE_QUALIFICATION_AUTHORIZED`, opened for **one** freeze.
* **Destination,** one-shot and create-only: `docs/superpowers/evidence/tactical-candidate-corpus/`.
  * `claim.json`, child stdout and stderr;
  * `log.jsonl`: per completed game, the stream, seed and window count;
  * `candidates.jsonl`: per completed game, the stream, seed, `end`, and for each window position its
    ply, class and digest — appended and **fsynced before the next game**;
  * `inventory.json`;
  * `corpus.json` — **only if COMPLETE**;
  * `record.json`, on every exit;
  * `watchdog.json`, written by the parent.
* **Runtime limit:** **3 hours**, enforced by the **parent**, which kills the process group (as §13
  A3.1). The child's own alarm fires 60 s earlier.
  * **Estimate:** under 1.5 h. Earlier finders ran a uniform playout in about 45 ms. A window
    classification takes about 20–40 ms. An exhaustive cross-check took 1.4–5.5 s per position.
* **Failure:** STOP, INTERRUPTED, INCOMPLETE or TERMINATED writes **no** `corpus.json`. The evidence
  stays; there is no retry. Another attempt needs a new authorization, and an amendment naming a new
  destination.
* **corpus.json contains:**
  * the A4.1–A4.5 settings, the interpreter version, and the sha256 of `tactical_corpus.py`,
    `h5_incumbent_tactics.py` and `tactical_candidate.py`;
  * the inventory's sha256, and its count per source;
  * per class: universe positions and games, `excluded_prior`, the selection and `quota_shortfall`;
  * the repeats;
  * per selected position: stream, seed, ply, moves, `state_digest`, class dict, `repeat_of`, and the
    cross-check result.
* **Commit before any model load.** A COMPLETE freeze is committed with the gate restored.

### A4.7 — Scope

Design amendment only. Generator implementation, corpus generation, model loading, qualifications,
research seeds, gate opening, real games and push remain unauthorized.

---

## 15. Amendment A5 — 2026-10-05, exclusions completed; publication by the parent (user-decided)

Review of A4 (`aa91a26a`) required two corrections before any code.

### A5.1 — The exclusion inventory, completed (supersedes A4.4's "not enumerated" clause)

Sources (a)–(d) stand. Two are added:

| # | source | positions |
|---|---|---|
| e | **every prefix** of the four canonical stand-in games in `tests/test_candidate_match_runner.py`: pairs FIRED (seeds −3, −4) and QUIET (seeds −13, −14), arms A and B, as the stand-in plays them | about 1,000 |
| f | **every one-ply and two-ply extension** of each *base*, kept where its ply lies inside the eligible window (10–270) | about 1.4 million, computed |

* **The bases** are:
  * the final positions of the nine pinned sequences;
  * `ONE[:23]`, `DRAW24[:40]` (`QUIET_40`) and `DRAW24[:-1]`.
* **Why (f) is a superset.** Every hand-built tree in the tests reaches at most two plies past one of
  these bases. Covering the whole two-ply neighbourhood includes every node any test builds,
  whichever move a helper happens to choose (the first legal move, a saving set). Positions outside
  the eligible window can never be selected, so they are skipped.
* **The prior move lists are pinned in data.** The move lists of (d), (e) and the bases are committed
  once in `scripts/GPU/alphazero/tactical_corpus_prior.json`, with its sha256 pinned in the module.
  Tests check that it still equals the suite's pinned constants and the stand-in's current replays. So
  drift is caught, never silently absorbed into the inventory.
* **inventory.json** lists sources (a)–(e) digest by digest. Source (f) is recorded by base, rule,
  count and the sha256 of its sorted digest list — too many digests to list one by one.
* **Wording (corrects A4.3):** "≥ 10: no early-opening coincidence" becomes "≥ 10: **excludes the
  earliest opening plies**". **Novelty comes from the digest check** (A4.4, A5.1), not from the streams
  or the ply bound.

### A5.2 — Only the PARENT publishes the corpus (supersedes A4.6's `corpus.json` clause)

* **The child never writes `corpus.json`.** It writes `corpus.provisional.json` — create-only, every
  byte or a STOP, fsynced — and records the provisional file's sha256 in `record.json`.
* **The parent publishes `corpus.json` only when all of these hold:**
  * the child exited normally: rc 0, not terminated;
  * the record is COMPLETE, and the provisional file is present with the recorded sha256;
  * its own validation passes:
    * the frozen specification, and the recorded interpreter;
    * the quotas: at most 20 per class and 100 in total;
    * every selected position replays on the official board to its digest, at an eligible ply, with
      the recorded class;
    * every cross-check agreed;
    * `inventory.json`'s sha256 equals the provisional file's record of it.
* **Publishing** means writing `corpus.json` create-only with the provisional file's exact bytes. The
  published sha256 goes into `watchdog.json`.
* **Otherwise there is no `corpus.json`,** and `watchdog.json` says why. A completed provisional
  payload followed by termination is **never** published.
* **Qualification** (later, separately designed) requires all of these committed:
  * `watchdog.json` COMPLETE, with the published sha256 equal to `corpus.json`'s;
  * `record.json` COMPLETE, with the provisional sha256 equal to the same;
  * `corpus.json` itself.

### A5.3 — Scope

Card corrections, then the generator implementation with the gate closed, synthetic tests,
injected-defect controls, the regression and local commits. The 8,000-game generation, the corpus
freeze, model loading, qualifications, research seeds, gate opening, real games and push remain
unauthorized.
