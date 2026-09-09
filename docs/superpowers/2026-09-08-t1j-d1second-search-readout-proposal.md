# D1″ — Bounded, Plan-Only Diagnostic Proposal: does our SEARCH lose what our POLICY already found?

**Status: PLAN ONLY, FROZEN BEFORE INSPECTION. Nothing here has been run, and the fields this
proposal measures have not been looked at.** No model is loaded, no JVM is started, no seed is drawn
or registered, no game is played, no training is proposed, and no confirmation data is opened. The
implementation phase is a SEPARATE authorization and is not requested by this document.

**What has been read, and what has not.** To write this I read the *writer's source* —
`eval_replay.ply_record` and `d1_probe._Incumbent.__call__` — to learn which fields the D1 record
carries. I did **not** read a single value of `raw_policy`, `root_visits`, `selected_visit_rank`,
`root_top1_share`, `top2` or `readout_overrode_leader` from the acquired record, and no D1″ quantity
has been computed. That is the point of freezing first: a metric chosen after seeing the field it
measures is not a preregistration.

**AMENDED 2026-09-08, before implementation, documentation only.** One sentence of §1 claimed D1′
had shown T1j's moves are "not conspicuously missing from our policy's top choices". That
overstated a matched result: D1′ compared *roles*, so it constrains the fragmentation-associated
**excess** and not the shared level. The sentence is replaced in §1 and its consequence is drawn out
in §3. **No metric, denominator, statistic, threshold, resampling seed, secondary report or budget
changed** — the amendment removes an unsupported claim about D1′ and adds a caveat about what D1″
does not know in advance. The same overstatement stands uncorrected in commit `7f4a0f9`'s message and
in my summary to the reviewer; this note is the correction of record for both, and history is not
rewritten.

---

## 1. Where D1′ left the question

D1′ asked whether T1j's chosen move tends to sit **outside our raw policy's top five** more often at
positions our model marks as mover-fragmentation weaknesses than at their matched controls. The
answer was **NO_GO**: support was adequate, the matched excess was **small but positive** (T =
0.0695), below the preregistered 0.15 threshold, and unstable under whole-game reweighting
(−0.1944 to 0.1827). Per the frozen wording correction, that is *not* a null and *not* a finding of
"no excess" — it is an effect too small and too unstable to justify a confirmation run.

**Stated precisely, because a looser version of this sentence was wrong.** D1′ did **not** establish
that T1j's moves generally appear among our policy's top five. It established only that the
fragmentation cohort did not show a sufficiently large and stable **excess** of outside-top-five
moves *relative to its matched controls*. LPRD could still be common — even predominant — in **both**
roles; a matched design measures the difference between roles and is silent about the level shared by
them. Nothing in D1′ licenses a claim about how often our policy ranks T1j's move highly overall.

What D1′ leaves unexamined is a different mechanism, and it is the one worth asking next:

> **D1″'s question.** When our raw policy *does* rank T1j's move among its top choices, does the
> 400-simulation search then fail to carry that move through — never visiting it, or visiting it so
> little that it drops out of contention?

A policy that already knows the move and a search that discards it is a very different defect from a
policy that never knew the move, and it has a different remedy. D1′'s indicator and D1″'s are
**disjoint by construction** (§3), so this is a genuinely new question about the same rows, not a
re-test of the same one.

## 2. Data audit — what this uses, and what stays sealed

| | |
|---|---|
| **Uses** | the completed D1 record, `2026-09-08-t1j-d1-acquisition/06_d1_records.json`, 221 positions |
| **Acquisition needed** | **NONE.** Every field required already exists in that record |
| **Development half** | L0 reps (0, 1) — the same half D0 and D1′ used |
| **Confirmation half** | reps (2, 3) — **SEALED, not opened, not selected from, not counted** |
| **H1 attempt 2** | development evidence only; contributes no row and no inference |

**This is the second question asked of the same development half, and that is exactly what a
development half is for.** It is also why nothing here can confirm anything: the protection against
looking twice is that confirmation happens on data that has never been looked at once. No threshold,
floor or rule from D1′ is adjusted after seeing D1′'s result (§5).

**Feasibility rests on the schema, and is verified before any statistic is computed.** The writer
records, per position: `raw_policy` (mass over every legal move), `root_visits` (visit count over the
same legal set), `n_legal`, `selected_visit_rank`, `selected_visit_count`, `root_total_visits`,
`root_top1_share`, `top2`, `readout_overrode_leader`, `selected_policy_rank`, `selected_policy_mass`;
and per position a depth-6 T1j block from which `t1j_move_6` is read. D1 already VOIDs a record whose
policy and visit maps do not cover the same legal set, so the two rankings are always over the same
support. **Step one of implementation is to assert these fields are present and well formed for all
221 rows, and to refuse otherwise** — before computing anything.

## 3. The measurement — search suppression (SS) — and the ONE primary hypothesis

For each row, with `K = 5` **unchanged from D1′**:

- `rank_raw(m)` — rank of move `m` by descending raw-policy mass, ties by ascending `(row, col)`.
  Already implemented and tested as `d1prime_analysis.rank_raw`; reused, not reimplemented.
- `rank_visit(m)` — rank of `m` by descending root visit count, ties by ascending `(row, col)`.
  **The tie-break is canonical and takes no input from the policy**, so the visit ranking cannot
  inherit the ordering it is being compared against.

> **PRIMARY INDICATOR — search suppression:**
> `ss := rank_raw(t1j_move_6) ≤ 5 AND rank_visit(t1j_move_6) > 5`
>
> In words: our policy put T1j's move in its top five, and after 400 simulations the visit
> distribution did not.

**How many rows can even be suppressed is UNKNOWN and deliberately unmeasured.** `ss` is defined on
the complement of `lprd` — rows where `rank_raw ≤ 5` — and D1′ says nothing about the size of that
complement in either role, for the reason given in §1. It may be most of the cohort or little of it.
That is not a gap to be closed by peeking: the eligibility floors (§5) are exactly the instrument for
refusing a statistic computed over too little, and they are checked before any replicate is drawn. If
the complement is too small, the frozen answer is `NO_GO — insufficient support`, not a rescue.

**Denominator: every row in the cell, not only the policy-eligible ones.** A row where the policy
never ranked T1j's move highly cannot have been suppressed, and scores `False` — which is correct,
not a missing value. This keeps the denominator identical to D1′'s and the eligibility floors
meaningful; a conditional denominator would shrink cells and make the floors describe something else.

**Disjointness, stated so it can be checked:** `lprd` (D1′) is `rank_raw > 5`; `ss` requires
`rank_raw ≤ 5`. No row can score both. A test must assert this on the real rows.

**PRIMARY HYPOTHESIS (one, frozen):** in the `mover_fragmentation` cohort, `ss` occurs at a higher
rate at **positions** than at their matched **controls**.

**PRIMARY STATISTIC:** exactly D1′'s — the Mantel–Haenszel weighted difference in `ss` rate between
roles over common-support cells `(opening, colour_arm, phase)`, weights `w = n_pos·n_ctl/(n_pos+n_ctl)`
from the cohort's own counts, cells lacking either role retained and reported descriptively but
entering no statistic. **The machinery is reused unchanged**; only the indicator differs. That is
deliberate: a second question answered with a second bespoke statistic would be two chances to find
something.

**Why T1j's move is the reference, and the limit of that.** T1j is the stronger player in these
frozen conditions (H1 attempt 2: 0.6763, interval above parity), so its depth-6 choice is a
defensible proxy for "a move worth considering". It is a **proxy**, not ground truth: T1j is not
optimal, and a move it prefers may be no better than ours. D1″ measures *our search's treatment of a
strong opponent's move*, and claims nothing more.

## 4. Secondary, descriptive, and unable to change the outcome

Reported alongside, deciding nothing — the same standing D1′ gives its secondary cohort:

1. **Readout non-selection.** `readout_overrode_leader`, `selected_visit_rank` and
   `selected_policy_rank`, summarised by role. This is the *other* mechanism — the readout declining
   the move the search preferred — and it is deliberately **not** primary: one frozen decision rule,
   one metric.
2. **Strict variant.** `ss0 := rank_raw(t1j_move_6) ≤ 5 AND root_visits(t1j_move_6) == 0`, which needs
   no tie-break at all. Preregistered here as a **sensitivity report**, so that if the primary's
   zero-visit tie mass is later questioned, the answer already exists and was not chosen afterwards.
   It cannot change the outcome.
3. **The `created_threat` cohort**, same indicator. Reported; cannot produce GO.

## 5. Decision rule — frozen, and deliberately identical to D1′'s

| | |
|---|---|
| eligibility floors | **unchanged**: ≥ 8 common-support cells, ≥ 40 positions, ≥ 30 controls, ≥ 12 distinct games |
| stability | stratified **whole-game cluster** resampling, B = 10,000, PCG64, seed **20260908** |
| interval | central 95%, type-7 quantiles — an **empirical stability interval**, never a confidence interval |
| threshold | **T ≥ 0.15** |
| **GO** | floors met **AND** T ≥ 0.15 **AND** interval lower bound > 0 |
| **NO_GO** | anything else, named: `NO_GO — insufficient support`, `NO_GO — bootstrap undefined`, or `NO_GO` |

**The threshold stays 0.15, and that is a commitment, not an oversight.** D1′ returned 0.0695 against
0.15. Lowering the bar now, having seen a number just under it, would be threshold shopping dressed
as a new hypothesis. Raising it would be equally unprincipled. The judgement standard is unchanged;
only the question is new.

**The PRNG seed changes to 20260908** because reusing 20260907 would make D1″'s replicate draw a
deterministic repeat of D1′'s over the same games. It is an analysis-only seed, in no registry, drawn
from no experimental block.

**What GO would mean, in full:** enough effect and enough resampling stability to justify **proposing
one confirmation acquisition** on the sealed half. It would not be proof of a population effect, not
authorization to acquire anything, and not a training decision. A confirmation would require a fresh
seed block, a fresh registration, a fresh gate opening and its own review — the D1 block is spent
whole and may not be revived.

**What NO_GO would mean:** the search-suppression mechanism is not supported strongly enough to
spend the sealed half on it, and the fragmentation line closes on the evidence available.

## 6. Analysis-time refusals the implementation must carry

Inherited from D1′ and non-negotiable, because they are what makes the entry checkable rather than
merely careful:

1. **One public entry taking the D1 report and nothing else**, resolving cohort, design, reps, B and
   seed itself. No caller-supplied rows, cohort, tasks, reps, replicate count or seed.
2. **The record must be a completed D1 run**: acquisition contract, both identities bound to their
   frozen pins, `elapsed_s` a real number within the deadline, positions present and well formed.
3. **Type-strict comparison** throughout; `False == 0` and `6 == "6"` are refusals, not matches.
4. **Every row bound to its frozen manifest row** on all binding fields, and to the design on
   `(opening, colour_arm, rep)`.
5. **An undefined statistic is named, never zeroed**: no common-support cell means `T is None` and a
   named NO_GO; an undefined arm fails, and does not silently pass.
6. **`rank_visit` must be tested against ties, zero-visit moves, and a single-legal-move root**, and
   asserted to take no policy input.
7. **`ss` and `lprd` asserted disjoint on the real 221 rows.**
8. Injected-defect controls for each of the above, each proven to reject, over a clean baseline.

## 7. Budget

**Zero queries, zero games, zero seeds, zero JVMs, zero model loads.** D1″ is arithmetic over a file
that already exists. Its only cost is the resampling, which is bounded by B = 10,000 replicates and
runs in seconds.

## 8. What this proposal does NOT do

- It does not open, select from, count, or describe the confirmation half.
- It does not authorize its own implementation, let alone its own execution.
- It does not propose training, and no D1″ outcome can propose training by itself.
- It does not revisit D1′'s verdict, adjust its threshold, or pool with it.
- It does not treat T1j's move as ground truth.
- It does not ask a second question if this one returns NO_GO. At that point the fragmentation line
  is closed on the available evidence, and anything further is a new preregistration.
