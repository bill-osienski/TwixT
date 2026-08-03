# Convergence Atlas — Design

**Created:** 2026-08-03 · **Status:** design approved; Phase 0 frozen, atlas measurement NOT execution-frozen
· **Scope:** one fresh corpus, one multi-budget search ladder, three read-outs.

## Execution-freeze status

| Section | Status |
|---|---|
| §2 Phase 0 inheritance preflight | **FROZEN.** Decision rule fixed before the run. |
| §3–§12 atlas measurement | **NOT execution-frozen.** |

**Open items that must all be closed before §3–§12 are execution-frozen:**

| # | Item | Where |
|---|---|---|
| 1 | Root regime resolved by Phase 0; if warm-start, the exact warm-root producer specified. Replaying only the immediate parent does **not** reproduce trajectory-compounded inheritance. | §2, §12 |
| 2 | A batch-safe decision boundary defined, and the diagnostic producer matched to eventual prototype semantics. | §4 |
| 3 | The frozen convergence predicate signed off. | §7 |
| 4 | Read-out C's producer chosen — selection tracer or reduced scope. | §8 |
| 5 | Deterministic game-to-cell assignment and phase-geometry no-go frozen. | §3 |

Writing and committing this document authorizes no measurement, no GPU work and no
`mcts.py` change.

---

## §0 Context and standing constraints

### The durable objective

> Make unchanged `calib020_0001` play stronger at the same 400-simulation budget
> while preserving search reliability and collateral safety.

Success requires a statistically significant same-checkpoint, equal-400-simulation,
balanced-colour strength gain plus contemporaneous non-regression against `0379`.
Nothing in this document is a candidate, a prototype, or a step toward adoption.

### What is closed

The governing record is the experiment ledger. **On this branch it is
`docs/updated-v16a-ledger.md`, which ends at do-not-repeat entry 42 and contains no
v17 or v18 record.** Entries 43–46 and the v17 and v18 closeouts exist only on the
unmerged `v18-depth2-provisional-backup` branch, where the file is also renamed to
`docs/alphazero-value-search-experiment-ledger.md`. The constraints they encode are
reproduced below and are binding on this design regardless of which branch carries
the file. In summary:

- **Value calibration against the A signal (v2 → v14b).** The A "post-opening sharp
  value drop" is a 400-simulation search artifact: BASE A mean moves
  `+0.2570 → +0.0626 → −0.0451` at 400/1,600/6,400 simulations. Closed.
- **`c_puct`.** Falsified as an A fix; lowering it worsens the metric.
- **Absolute `fpu_value = −0.20`.** Rejected on the v16a game-held-out sample
  (late new-collapse 13/84 = 15.48%).
- **Policy-mass FPU, both formulations.** v16 parent-relative died at its own `r=0`
  prerequisite (27.5% lower-prior control flips vs a `<10%` gate); v17
  baseline-preserving ran the full preregistered grid and failed §7.2 safety at all
  five coefficients, with control flips flat at 31–50% while reply reduction varied
  threefold. Do-not-repeat #45.
- **v18 depth-2 provisional backup.** Rejected at shipped-only preflight:
  A-vs-matched-control AUC `0.5089` (bound `0.39`), sign dominance `0.78475 < 0.80`,
  and zero of 1,957 census rows satisfied the complete target predicate.
  Do-not-repeat #46. Its evidence is consumed and may inform mechanism design only.

The atlas does not resume, rescue or reuse any of the above. It uses a fresh corpus
and measures shipped search only.

### The pattern the atlas responds to

Every search intervention so far either fails to reach the mechanism, or reaches it
and carries collateral that does not scale down as the knob shrinks. v18 added a
third finding: an observable that *reaches* the phenomenon but does not *separate*
safe targets from collateral-sensitive positions.

Two structural observations motivate this diagnostic:

1. **The candidate set has never been touched.** `mcts.py::_select_child` iterates
   `node.priors.items()`, and `node.priors` is built in `_expand_batch` directly from
   `state.legal_moves()` with no truncation. At every node and every depth the search
   enumerates all legal moves — roughly 500 on a 24 board. c_puct, all three FPU
   forms and the v18 clip modulate the *value* of the unexplored or the value backed
   up from it. None changes how many options exist.
2. **The acceptance gates have never been validated against strength.** Every gate has
   the form "do not change shipped behaviour on positions you were not targeting."
   Applied to an intervention whose purpose is to change search behaviour, only a
   near-no-op passes. Four search interventions have been rejected and zero strength
   matches have been run.

### Two code findings that belong in scope

**Tree reuse is additive, and historical probes measured a different regime.**
`self_play.py:1145` advances the root into the played child
(`# TREE REUSE: advance root to chosen child`), and `search_from_root` runs
`for sim in range(remaining_sims)` where `remaining_sims = n_simulations − forced_count`.
Simulations are therefore **new**: a deployed nominal 400-simulation search adds 400
simulations to an inherited tree and ends at roughly `inherited + 400` root visits.

Every reconstructed-position diagnostic in this line — the A/B/C/D probes, v16a's 324
positions, v17's 32, v18's census — ran on fresh roots. Those probes had strictly
fewer accumulated visits than reused game roots. Calling them *weaker* is plausible
but not established; their scope is a **less-searched fresh-root regime**. Their
within-regime conclusions stand.

**Forced root visits consume the same budget.** `_maybe_force_root_td1_visits`
returns a `forced_count` that reduces the main-loop budget. Forced visits are real
selection events inside the fixed new-simulation budget. `forced_count` is resolved
inside `search_from_root` after the forcing check, so it is recorded as
**start-of-search telemetry**, not before the search call.

---

## §1 Hypotheses and stopping conditions

**Read-out A hypothesis.**

> Features available at the 320-completion prefix of a normal 400-simulation run can
> predict which ordinary 400-simulation searches materially disagree with a stable
> 3,200/6,400-simulation reference.

**Read-out C hypothesis.**

> A simple prior-ranked widening rule would restrict shallow allocation more often on
> misleading roots than on stable roots, while retaining the moves and replies that
> stable deeper search eventually prefers.

**Read-out B is calibration, not a hypothesis.** It asks whether the inherited
collateral gates fire on changes that move toward the stable deeper reference.

**The atlas fails operationally if** the corpus does not contain enough stable
positives and negatives, the 320-prefix detector does not separate them on
validation, or neither widening shape both retains the stable deep moves and
intervenes preferentially on misleading roots.

A capacity failure means "this atlas could not establish the signal," **not** "the
information cannot exist." This distinction is load-bearing and preserves the v18
closeout lesson.

---

## §2 Phase 0 — inheritance preflight (FROZEN)

### Why it exists

The atlas probes fresh roots. Deployment reuses trees. Read-out A's features are
inheritance-sensitive by construction: one-visit backup share, fraction of backups
reaching depth ≥3, and breadth below the leader all depend on how populated the tree
already is. The indirect estimate — v16a's mean top-child visit share of `0.4154`
implying roughly 166 inherited visits — is not adequate to freeze a regime on,
because opening temperature, non-leading played moves, phase and tree shape all
affect how many visits actually survive `advance_root`.

### Protocol

- Run **one** unchanged shipped self-play game with tree reuse.
- Record as start-of-search telemetry, per search: starting root visits, starting
  visited children, phase, and `forced_count`.
- Record per move, immediately before `advance_root`: the played child's visit count.
- Express deployment inheritance at the proposed decision point as:

  ```text
  inherited_fraction_320 = starting_visits / (starting_visits + 320)
  ```

- Summarize median and upper quartile by phase.
- The game is excluded from the atlas corpus and is never used to tune labels,
  features, gates or thresholds.

This is a **technical preflight, not scientific evidence.**

### Frozen decision rule

> If median `inherited_fraction_320 ≥ 0.10` in any post-opening phase, **or** overall
> p75 `≥ 0.20`, the atlas must use a warm-start regime. Otherwise fresh-root probing
> is acceptable, with the remaining mismatch stated.

If a warm-start regime is required, the exact warm-root producer must be specified
and frozen before §3–§12 become execution-frozen. See §12.

---

## §3 Corpus

### Source protocol (frozen at generation time)

- A new, non-overlapping contiguous half-open seed range, frozen **at the 400-game
  maximum** before any generation, so staged generation stays reproducible and no
  range is chosen after seeing data.
- A fixed game count chosen before generation, per the sizing rule below.
- **No top-up** after inspecting phase supply, labels or tree features.
- Unchanged shipped game generation, with tree reuse.
- Games completely disjoint from selected A, A/B/C/D, v16, v16a, v17 and the consumed
  v18 universe.

### Position sampling

- Four phases: opening `0–30`, early-mid `31–60`, midgame `61–90`, late `91+`.
- Equal representation by phase and side: eight phase×side cells, equal counts.
- At most **one** position per game.
- Selection uses **only** game identity, phase, side and the frozen sampling seed.
  No selection on values, residuals, entropy, branching, game outcome, or any other
  search result.
- Valid state-cap and winner-null games are kept, and identified in summaries.
- All three read-outs use the same resulting corpus.

### Staged sizing (frozen before the pilot)

Pilot: **24 positions from 24 distinct fresh games**, three per phase×side cell, run
on the full ladder, included in **discovery only** and never eligible for validation.

Let `p_m` = pilot misleading count / 24 and `p_s` = pilot stable-negative count / 24.
Retain a 60/40 discovery/validation split. Sizing targets carry a 20% margin over the
final validation minimums (`1.2 × 20 = 24` misleading, `1.2 × 25 = 30` stable-negative):

```text
N_required = max( 24 / (0.4 * p_m),  30 / (0.4 * p_s) )
```

Round upward to the next multiple of 40, so both splits stay exactly balanced across
eight phase×side cells. Allowed sizes: `200, 240, 280, 320, 360, 400`.

If either pilot frequency is zero, or the projected requirement exceeds 400, **stop
with a projected capacity no-go** rather than spending the full run on a design
expected to be underpowered.

After the formula selects `N`, generate the continuation block from the already-frozen
seed range. There is no second resizing step and no top-up after validation labels are
observed.

### `N` counts positions, not games

`N` is a required **position** count. With one position per game, `N` games are needed
*that can serve their assigned cell* — and not every game can. A game must survive to
ply 91+ to supply a late row, so the number of games generated necessarily exceeds `N`
by a factor the pilot measures. Only the **400-game maximum seed range** is frozen
before pilot generation; the exact continuation count is chosen afterwards by the
frozen sizing formula and the measured phase geometry.

This is the failure mode that has already bitten this project twice — v16's
`assign_split: cell capacity 0 < demand 60` and v18's `target|late capacity 0 <
demand 16`. Since top-up is forbidden, it must be prevented structurally:

- **Deterministic game-to-cell assignment**, frozen before generation, using only game
  identity, the frozen sampling seed, and game length. Game length is a game property,
  not a search result, so this does not violate blind sampling — but it is the direct
  mechanism behind the late-stratum bias recorded in §12, and the two must be read
  together.
- **Explicit phase-geometry no-go.** If a generated block cannot fill every phase×side
  cell at the required count under one-position-per-game, the atlas stops with a
  **geometry no-go**. It does not top up, does not rebalance cells, and does not
  relax the one-position-per-game rule.
- The pilot's 24 rows carry the same constraint: three late-black and three late-red
  rows require six distinct games reaching ply 91+. If the pilot block cannot supply
  them, that is a geometry no-go at the pilot, before the main generation is paid for.

Worked check of the rounding (8 cells, 60/40):

| `N` | per cell | discovery / cell | validation / cell |
|---:|---:|---:|---:|
| 200 | 25 | 15 | 10 |
| 240 | 30 | 18 | 12 |
| 320 | 40 | 24 | 16 |
| 400 | 50 | 30 | 20 |

### Final capacity gate

The completed validation split must contain **at least 20 misleading** and **at least
25 stable-negative** positions. If not, the atlas ends as an operational capacity
failure. Do not weaken labels, move ambiguous rows, or add games.

The stable-negative minimum is 25 rather than 40 on technical grounds:
`min(n_pos, n_neg)` binds the AUC standard error. At a true AUC of `0.75` with 20
positives, Hanley–McNeil gives SE `0.0753` at 25 negatives versus `0.0711` at 40 —
an improvement of about `0.004` for a 60% larger negative requirement.

---

## §4 Search ladder

```text
400 (features snapshotted at the 320-completion prefix), 1600, 3200, 6400
```

= **11,600 simulations per position.**

| budget | purpose |
|---|---|
| 320-completion prefix | deployable detector features |
| 400 | shipped comparison point |
| 1,600 | intermediate convergence, old-gate calibration |
| 3,200 and 6,400 | stable-reference test |

**The 320 prefix is a snapshot, not a rung.** Features are captured at the **320th
completed backup** inside the continuous batched 400-simulation run. It is described
as the "320-completion prefix of the 400 run" and is **not** claimed equivalent to an
independent `n_simulations=320` search — under batched evaluation, pending leaves mean
the 320th completed backup is not the 320th loop iteration. Any later prototype must
use the same continuous-run boundary semantics.

This choice removes a producer/consumer seam that a separate 320-simulation search
would have created.

**It does not yet make the boundary deployable, and this is an open item that must be
closed before atlas freeze.** `_flush_pending_batch` expands every pending leaf in a
batch and only then backs up all waiter paths, and `_observer_completed_count`
increments per backup inside `_backup`. A callback firing at the 320th backup
therefore sits *inside* a flush: it can already see expansions belonging to later
members of that batch, and up to `eval_batch_size` (14) simulations are queued and
cannot be redirected by any detector. The snapshot is a valid continuous-run
**diagnostic** boundary; it is **not** established that a deployed 320+80 controller
could observe and act at this exact point.

Before §3–§12 are execution-frozen, a **batch-safe decision boundary** must be defined
— for example the first backup completion at or after 320 that coincides with a flush
completion — and the diagnostic producer must be made to match the semantics the
eventual prototype would use. Until then, no claim of controller realizability may be
drawn from Read-out A.

**Cost.** The deep ladder dominates the analysis; the read-outs are nearly free once
the trees exist. Corpus generation dominates the ladder — approximately
`N × mean_plies × 400` simulations, which at `N=240` and ~160 mean plies is roughly
15M against the ladder's 2.8M. Estimate the full run from the **pilot's measured
rate**. Do not scale a smoke, and do not rely on an unrelated historical throughput
figure.

---

## §5 Labelling

All values are side-to-move perspective.

**Stable deep reference** when all hold:

- 3,200 and 6,400 selected moves agree.
- `|V3200 − V6400| ≤ 0.10`.
- Normalized 6,400 top-two visit margin ≥ `0.05`.

**Misleading at 400** when the reference is stable and either:

- `|V400 − V6400| ≥ 0.25`; or
- the 400 selected move differs from the stable deep move.

**Stable negative** when:

- the 400 selected move equals the stable deep move; and
- `|V400 − V6400| ≤ 0.10`.

Everything else is **ambiguous**. Ambiguous positions stay in the corpus and their
frequency is reported; they are never used as convenient positives or negatives.

The value and move components are reported separately. A detector that predicts value
correction but not move disagreement is useful mechanistic evidence but weaker support
for a playing-strength intervention.

---

## §6 Read-out A — convergence detector

Only features available at the 320-completion prefix may feed the deployable detector.

Fixed feature set:

- One-visit backup share.
- Fraction of backups reaching depth three or deeper.
- Normalized leader visit margin.
- Normalized root-policy entropy.
- Breadth below the current root leader.

Q dispersion, residual summaries, terminating-backup concentration and other tree
statistics may be reported descriptively but must not become an unrestricted feature
search.

One simple interpretable classifier — fixed-ridge logistic regression. Standardization
is learned on discovery only.

**Pass conditions:**

- Validation holds ≥20 misleading and ≥25 stable-negative positions.
- Validation AUC ≥ `0.75`.
- Bootstrap lower bound ≥ `0.60`.
- A discovery-frozen threshold flags ≤25% of validation positions at precision ≥ `0.60`.

**Power is thin at the bar and this is accepted in advance.** Hanley–McNeil,
one-sided 95%: at 20/25 a true AUC of `0.75` gives a lower bound of about `0.62`
(passes), while a true AUC of `0.70` gives about `0.57` (fails). The atlas can detect
only a strong detector. A miss near the bar is reported as underpowered, not as
evidence of no signal.

**Diagnostic contrast.** If the 400-tree features separate but the 320-prefix features
do not, the detector **fails** — a complete 400 tree cannot decide how to allocate the
last 80 simulations. That contrast is still reported, because "the information arrives
late" is informative for a different intervention point.

---

## §7 Read-out B — old-gate calibration

Compute the historical metrics at 400, 1,600 and 6,400:

- New collapse: top share crosses from below `0.95` to ≥ `0.95`.
- Flip to a lower-prior move.
- Effective-children reduction (`exp` of root visit entropy).
- Top-share increase.
- The historical compound narrowing condition where applicable.

### Frozen convergence predicate

The "needs review" rule below counts triggers that are *confirmed convergent*, so that
Boolean must be defined exactly. It is evaluated only on positions that have a stable
deep reference (§5). All four components are computed per position:

```text
move_convergent   := selected_400 != stable_deep_move AND selected_1600 == stable_deep_move
value_convergent  := |V1600 - V6400| <= |V400 - V6400| - 0.10
dist_convergent   := (top_share OR effective_children) closes >= 50% of its
                     400 -> 6400 gap, AND selected_1600 == stable_deep_move
persistent        := selected_1600 == selected_3200 == selected_6400

convergent := persistent AND (move_convergent OR value_convergent OR dist_convergent)
```

Persistence is a **joint requirement**, not a fourth alternative — a 1,600 change that
does not survive to 3,200 and 6,400 is not convergence. The `0.10` value threshold is
the same tolerance §5 uses for stable-reference agreement; the `50%` gap-closure is
carried unchanged from the original draft. No new numbers are introduced.

**This predicate is newly authored and is not yet signed off.** It must be approved
before Read-out B is implemented or interpreted.

Report the same metrics for 400→6,400 to show the scale of natural deeper-search
change. Those changes are the natural-convergence reference distribution; they are
**not** causal evidence that a same-budget intervention is safe.

**Frozen "needs review" rule.** A gate is marked *needs review* when all three hold:

1. At least 10 gate triggers.
2. At least 75% of those triggers are independently confirmed convergent.
3. The triggered convergence rate is at least **15 percentage points** above the base
   convergence rate among all stable-reference-eligible 400→1,600 rows.

Condition 3 exists because without a base-rate comparator, a gate that fires on
everything passes conditions 1 and 2 whenever 75% of all changes happen to be
convergent.

**"Needs review" means the gate structure must be reviewed and frozen before it judges
another prototype. It does not mean the gate is invalid, and it does not authorize
deleting or relaxing it.** Higher-budget fidelity is itself only a proxy, while the old
gates protect against collateral behaviour that fidelity does not measure.

Report overall and for late, flat-policy and near-even roots. Do not create a separate
acceptance gate per stratum.

---

## §8 Read-out C — progressive-widening coverage feasibility

### The rule under analysis

```text
K(n) = min(n_legal, max(1, ceil(C * n^alpha)))
```

`n` is the parent's **completed** visit count; moves are ordered by shipped adjusted
prior with deterministic move-ID tie-breaking.

Two shapes, chosen to differ in how fast breadth collapses below the root rather than
in overall permissiveness:

| n | `(C=4, α=0.5)` | `(C=13, α=0.3)` |
|---:|---:|---:|
| 400 (root) | 80 | 79 |
| 105 (top child) | 41 | 53 |
| 20 | 18 | 32 |
| 5 | 9 | 22 |
| 1 | 4 | 13 |

They match at the root and diverge 2–3× at low-visit descendants, which is the actual
design risk: whether widening becomes too aggressive below the root, where flat-policy
nodes make a top-k-by-prior admission close to arbitrary.

Theoretical admission visit for a move of prior rank `r` — defined as a search, not a
closed form:

```text
n_admit(r) = min { n >= 0 integer : K(n) >= r }
```

Compute it directly. The closed form `ceil((r/C)^(1/alpha))` is **wrong** because it
inverts `C*n^alpha >= r` and discards the `ceil` inside `K`. Pinned counterexample:
at `(C=4, alpha=0.5, r=9)` the closed form returns `6`, but
`K(5) = ceil(4*sqrt(5)) = ceil(8.944) = 9 >= 9`, so the true answer is `5`.

Note also that `n = 0` is admissible and rank 1 is always admitted there, because
`K(0) = min(n_legal, max(1, 0)) = 1` via the `max(1, ...)` floor. Any equivalent
closed form must be tested against a direct search over the full rank range for both
frozen shapes before use.

### Early static check (on the pilot)

Using stable 3,200/6,400 reference lines:

- Was the stable root move admitted by 320 and 400?
- Was the stable opponent reply under that move admitted given its parent's visits?
- For a fixed two-ply reference horizon, were the required moves admitted?

If both shapes clearly fail these retention checks on the pilot, **close progressive
widening without inventing another shape.**

### Full feasibility analysis

For actual shipped selection events through the 320-completion prefix and 400, record:

- How often the selected move's prior rank lies outside `K(n)`.
- Root and depth-1 exclusion rates.
- First-touch exclusions.
- Fraction of misleading roots meaningfully affected.
- Fraction of stable-negative roots meaningfully affected.
- Whether stable deep root and reply moves remain admitted.
- Excluded prior mass.

Aggregate online. Full event dumps are unnecessary.

### Producer gap — the existing observer cannot supply this

The existing `MCTSObserver` fires as `on_root_simulation(count, root, move,
visit_leader_move(root))` from inside `_backup`: it receives **post-backup, root-level
information only**. It cannot report individual `_select_child` events, their
selection-time parent visit counts, depth, or first-touch exclusions — which is exactly
what this read-out consumes.

Read-out C therefore requires either a **diagnostic-only selection tracer** emitting
per-`_select_child` events (depth, parent completed visits, selected move's prior rank,
`K(n)` membership, forced flag), or a **reduced scope** limited to what the root
observer can supply. That choice must be made before atlas freeze.

Whichever producer is chosen must be included in the observer-on/off batched identity
prerequisite of §9. Qualifying only the current root observer is insufficient: it would
prove nothing about the component that actually produces this read-out's data.

**Meaningful intervention** (frozen before the pilot): at least 10% of observed
first-touch selections lying outside the admitted set.

### Feasibility bar

- Retain ≥95% of stable deep root moves.
- Retain ≥90% of stable depth-1 replies.
- Meaningfully intervene on ≥50% of misleading roots.
- Meaningfully intervene on ≤25% of stable-negative roots.

### Frozen shape selection (lexicographic)

1. Root and reference-reply retention floors must pass.
2. The stable-root intervention ceiling must pass.
3. Among remaining shapes, choose the higher intervention on misleading roots.
4. Exact tie: choose the higher descendant reference retention.

Choose on discovery. Validate only the selected frozen shape. No capacity rescue and
no widening-shape substitution after validation.

### Completed-visit lag — directional, with a bound

Completed visits lag in-flight work by up to one batch (14). Smaller `n` gives a
narrower `K(n)`, so the lag is **conservative for retention** and
**anti-conservative for intervention**. Therefore:

- Evaluate **retention** using `K(n)` — the narrower, conservative admitted set.
- Require the **intervention** threshold to pass under `K(n+14)` as well, giving a
  conservative lower estimate of intervention after allowing for the maximum in-flight
  batch.
- If intervention passes only under `K(n)`, report it as **inconclusive, not a pass.**

### Forced root visits

Forced visits are recorded separately and **excluded from the ordinary PUCT-intervention
denominator.** Additionally, report whether the forced moves fall outside the
hypothetical admitted set, because a future widening implementation would need to
exempt or otherwise reconcile them explicitly.

### Descendant strata

Flat-policy status is **recomputed locally along the reference line**, not inherited
from the root. Report retention separately for:

- Late roots.
- Near-even roots, `|V_stm| ≤ 0.30`.
- Root-flat positions.
- Locally flat depth-1 and depth-2 parents.

Use the existing flat-policy definition — normalized policy entropy ≥ `0.90` and top
prior ≤ `0.025` — and the existing near-even definition, rather than inventing strata
after measurement.

This read-out is a **counterfactual coverage analysis**. It cannot prove progressive
widening would improve search, because applying widening changes the later tree.

---

## §9 Mechanical safeguards

### Observer identity prerequisite

The identity smoke must exercise the actual batched path:

- `search_from_root`.
- Batching `(14, 48, 8)`.
- `add_noise=false`.
- Unchanged checkpoint and shipped configuration.
- **Every** diagnostic producer off versus on — the existing root observer, the
  320-completion snapshot hook, and the §8 selection tracer if one is built. Each is
  qualified individually and all-on together.

Require exact equality of selected move, root value, visit counts, tree summary and
search-result hash. Synchronous CPU tests remain useful but cannot replace this batched
smoke, because they do not exercise pending leaves and virtual visits.

### Guard against

- Root/child perspective sign mistakes.
- Confusing the raw visit leader with the final shipped action after any selection
  override.
- Treating undefined tree metrics as zero. An undefined row-level statistic is `null`,
  never `0.0`, and never a reason to drop a row or abort a run.
- Reusing synchronous-only terminating-backup formulas without checking them on batched
  trees. The v18 tree walker remains useful but its terminating-backup arithmetic was
  scoped to synchronous search; any use on the batched shipped path must first pass
  explicit tree-accounting invariants.
- Persisting full trees unnecessarily. Compact summaries and online counters suffice.

### Operator rules

- Launch long jobs and wait for them in **separate** tool calls. `nohup` + `disown` is
  not enough when the launch and the wait share one call; a tool timeout SIGTERMs the
  whole process group. `setsid` does not exist on macOS.
- No source edit after qualification, and no commit between generation and
  qualification.
- Drive real producers into real consumers at least once. Consumers tested only against
  hand-written surrogates of their producers cost v18 four contract defects and three
  restarts.
- A provenance test must **construct** its negative case, never observe ambient worktree
  dirtiness.

---

## §10 What must be frozen

Before Phase 0: the §2 protocol and its decision rule. **Done.**

Before the pilot, one reviewed protocol document must freeze:

- Corpus source, seed range, exclusions and split.
- Search ladder, the 320-completion prefix boundary semantics, and shipped configuration.
- Stable-reference, misleading, stable-negative and ambiguous definitions.
- The five detector features and the detector pass bar.
- The old-gate "needs review" rule, including the base-rate comparator.
- Both widening shapes, the reference horizon and retention thresholds.
- The meaningful-intervention definition.
- The lexicographic shape-selection order.
- The staged sizing formula, allowed sizes and the capacity no-go rule.
- Pilot stop conditions.
- Bootstrap method and seed.
- No-top-up and no-post-hoc-rescue rules.

That is sufficient protection. This does not need a publication-grade artifact system
before a signal exists.

---

## §11 Decision tree

- **Detector passes; widening fails:** a bounded 320+80 verification prototype is
  eligible.
- **Widening passes; detector fails:** a small progressive-widening prototype is
  eligible.
- **Both pass:** compare the observed separation and implementation risk. Do **not**
  pre-commit to favouring the detector.
- **A gate is marked "needs review":** freeze any gate redesign before running or
  judging a prototype. Do not retroactively rescue v16 or v17.
- **Neither tree-local method separates misleading from stable roots:** close
  tree-local heuristics; consider high-budget distillation or direct playing-strength
  work.
- **Capacity fails:** record an operational no-go, not a claim that the information
  cannot exist.

Any prototype is still discovery work. It would need exact 400-simulation accounting,
fresh validation, collateral checks, A/B/C/D, and ultimately a same-checkpoint
balanced-colour strength match before adoption.

---

## §12 Stated limitations and open gaps

### Open specification gap

**If Phase 0 requires warm starts, the exact warm-root producer is not yet frozen.**
Replaying only the immediate parent does not reproduce a tree inherited across the full
game trajectory, because inheritance compounds: the subtree carried into ply `k+1` was
itself built on a tree inherited at ply `k`. Candidate producers, their cost and their
fidelity to real trajectory inheritance must be specified and frozen before §3–§12
become execution-frozen.

### Branch-level ledger gap

This branch is cut from `main`, whose ledger `docs/updated-v16a-ledger.md` ends at
do-not-repeat entry 42 and records neither v17 nor v18. The constraints from entries
43–46 — the policy-mass FPU closure and the v18 selectivity null — are binding on this
design and are reproduced in §0, so the spec is self-contained. But a reader on this
branch cannot verify them against the repository. Resolving this is a provenance
decision, not a design one, and it does not block Phase 0.

### Known biases and limits

- **Late-stratum composition.** Filling the late cells requires games that survive to
  ply 91+, so the late stratum over-represents long games. v16a saw this directly: 19
  state-cap 280-ply marathons supplied 47 of its 84 late positions. **Late results
  generalize to games that survive to ply 91+, not to the overall game population.**
  Report state-cap rate, winner-null rate and game-length distribution beside late
  results. No rebalancing after observing them.
- **6,400 is a stable proxy, not ground truth.** The 3,200/6,400 agreement requirement
  is what keeps this falsifiable.
- **Shared structure.** Labels and features come from the same searches, so label noise
  is shared. The stability requirement mitigates it; the discovery/validation split
  handles selection on features.
- **Fresh-root probes may not represent subtree-reusing game search.** This is what
  Phase 0 exists to resolve.
- **Phase 0 observes a single game trajectory.** Its per-search rows are serially
  correlated and one game's inheritance may be atypical, so the per-phase medians and
  p75 are a regime indicator rather than a population estimate. That is adequate for
  a frozen binary regime decision and is why Phase 0 is labelled a technical preflight
  rather than evidence; it is not adequate to characterize the inheritance
  distribution, and no later section may cite it as such.
- **Coverage is not performance.** Progressive-widening coverage must not be mistaken
  for intervention performance.
- **A value-only detector is weaker evidence** than one that also predicts move errors.
- **The pilot is for identity, runtime, class-frequency sizing and the frozen widening
  kill condition only.** It must not be used to adjust thresholds.
- **Ambiguous rows are reported, never discarded silently.**
- **Old collateral gates are not weakened merely because deeper search sometimes
  triggers them.**

---

## Related

- `docs/updated-v16a-ledger.md` — the governing record **as it exists on this branch**:
  do-not-repeat entries 1–42, no v17 or v18 record. Entries 43–46 and the v17/v18
  closeouts live on the unmerged `v18-depth2-provisional-backup` branch, where the
  file is renamed to `docs/alphazero-value-search-experiment-ledger.md`. §0 reproduces
  the constraints this design depends on, so the spec is self-contained, but the
  branch-level gap is real and is flagged in §12.
- `logs/eval/v18_depth2_provisional_backup/V18_FINAL_NULL.md` — the v18 closeout whose
  operational-no-go framing this design preserves. Gitignored: a local artifact, not a
  repository reference.
