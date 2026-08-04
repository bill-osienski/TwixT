# v18 Depth-2 Provisional Backup — Draft Design (revision 3)

**Status:** DRAFT — REVISED, NOT APPROVED, NOT FROZEN, AND NOT AN AUTHORIZATION
TO CHANGE CODE OR RUN SCIENTIFIC EVIDENCE.

Revision 2 (2026-07-29) incorporated a code audit of `mcts.py` and six design
decisions taken during review. Revision 3 (2026-07-29) resolves seven protocol
ambiguities found in review of revision 2. Items still marked **FREEZE BLOCKER**
must be resolved, reviewed, and incorporated into a clean frozen revision before
any MCTS source edit or scientific run.

v18 is not v17b. It does not rescue, refine, or extend either retired
policy-mass FPU formulation.

## Revision 2 change log

1. **Code surface corrected.** No v18 stage executes the batched waiter path;
   the rule has exactly one implementation call site (§4.3).
2. **Breadth gate reframed** from a flat `<10%` reply-reduction rejection to
   bounded loss plus proven depth conversion (§10.3, §10.4).
3. **Control roles split** into activation-negative identity witnesses and
   activation-positive flip controls (§9.2).
4. **Reuse strategy fixed** as a thin v18 layer over existing modules, with no
   extraction refactor (§5.1).
5. **Fidelity reference fixed** at 6,400 simulations, with 1,600 recorded as a
   descriptive trajectory diagnostic only (§10).
6. **Telemetry** derived from the final tree by a read-only walk rather than
   hot-path instrumentation (§4.4).

## Revision 3 change log

1. **Evidence-boundary contradiction resolved** by a narrow declared exception:
   historical unsafe operating points may impose a conservative ceiling on
   `R_max` only (§2.1.1).
2. **Cap-ladder routing completed** — every §10.3 and §10.4 outcome now has a
   defined consequence, and the prior-rank regression gate moved from §10.4 to
   §10.3 as a conservative stop rule (§7, §10.3, §10.4).
3. **Efficiency denominator guard corrected** — `R_min` is pooled over all
   targets and does not guarantee positive `lost_replies` on the stable-leader
   subset; the minimum-`lost_replies` requirement is the real guard, and it is
   now carried into held-out (§10.4.1, §11).
4. **6,400-simulation A move reference** does not exist in the historical
   artifact; a shipped-only capture is preregistered before the MCTS edit, or
   the move-agreement gate is dropped (§2.2.2, §12.1).
5. **Target exposure statistic** narrowed to a positive-residual formula with a
   sign-dominance requirement, because §1.3's directional prediction is a claim
   about the positive-residual population (§9.2.1).
6. **Exact formulas added** for every new conversion metric (§10.1.1).
7. **Three technical wording corrections** — evaluator call versus load, the
   synchronous prerequisite on the static substitution, and `tanh` range
   phrasing (§2.2.1, §3.1, §4.3).
8. **v16a reply-reduction anchor resolved** from its source artifact at
   `28.0273%` (§2.3).

Revision 3 review corrections (same revision, applied after review):

9. **`revisit_to_depth3_rate` given a shipped baseline** via a counterfactual
   `would_clip(arm, cap)` population evaluated in both arms, with an
   authentication invariant and a declared candidate-only fallback (§10.1.1).
10. **Aggregate reply reduction generalized** to the stage's frozen evaluation
    set, with `L_shipped` and `L_candidate` named explicitly (§10.1.1).
11. **Prior-rank stop rule reworded** as a conservative preregistered decision
    rather than a monotonicity claim §1.5 disclaims (§7, §10.3).
12. **All three historical anchors given exact figures and artifact SHA-1s**;
    rounded forms superseded (§2.3).
13. **Capture-script parameterization requirement recorded** for the A/6,400
    move capture, with dual-statistic authentication and a byte-identical
    determinism check (§2.2.2).

## 0. Decision context and objective

The durable objective is unchanged:

> Make unchanged `calib020_0001` play stronger at the same 400-simulation
> budget while preserving search reliability and collateral safety.

Success ultimately requires:

1. Fresh development safety and mechanism evidence.
2. Fresh held-out confirmation.
3. All established A/B/C/D acceptance gates.
4. A statistically significant same-checkpoint, equal-400-simulation,
   balanced-color strength gain over shipped search.
5. Contemporaneous non-regression against `0379`.

The final v17 result closes policy-mass FPU:

- The parent-relative v16 formula failed at its `r=0` prerequisite:
  `11/40 = 27.5%` lower-prior control flips.
- The baseline-preserving v17 formula preserved shipped behavior exactly at
  zero but every positive coefficient failed both development safety gates.
- v17 target new-collapse was `1–2/16` at every grid point, where zero was
  required.
- v17 control lower-prior flips were `5–8/16` at every grid point, where at
  most one was permitted.
- Reply reduction increased from about `0.23` to `0.79`, proving that the
  intended suppression mechanism was active; the unacceptable collateral did
  not disappear at smaller tested coefficients.

Therefore v18 must not change unexplored-child FPU, use explored policy mass,
replace the neutral point with `Q_parent`, relax the v17 gates, or extend the
v17 grid.

## 1. Scientific hypothesis

### 1.1 Evidence motivating a different intervention point

The selected-A diagnostics found:

- The 400-simulation excess was concentrated in the root's selected child,
  but not in a small PV or a few continuation states.
- At the top positive depth-1 child, raw black value averaged about `-0.087`
  while searched black value averaged about `+0.619`.
- Across the selected positive branches, `4,443` depth-2 nodes accounted for
  `77.3%` of leaf evaluations.
- Their mean raw black value was about `+0.793`, and `98.8%` were raw-positive.
- A median of `196` depth-2 nodes per root was required to cover 70% of
  positive raw mass.
- Increasing the budget from 400 to 1,600 and 6,400 simulations reduced the A
  mean from `+0.2570` to `+0.0626` and `-0.0451`, respectively.

The defect is therefore consistent with **premature confidence in a broad,
shallow, one-evaluation frontier**. More search eventually supplies deeper
evidence and removes most of the shallow bump.

FPU attacked this by suppressing selection of unexplored replies. That
reduced frontier width, but it also caused ordinary roots to concentrate,
collapse, and switch to lower-prior moves.

### 1.2 v18 hypothesis

v18 tests a different proposition:

> Preserve shipped selection and frontier breadth, but treat a newly expanded
> nonterminal leaf exactly two plies below the current root as provisional.
> Limit only an unusually large one-ply disagreement between that leaf's raw
> value and its expanded parent's raw value. If the branch is revisited,
> depth-3 and deeper evidence is backed up normally and may confirm or overturn
> the provisional value.

This mechanism acts after the leaf has already been selected and evaluated.
It does not decide whether an unexplored action may be visited.

The hypothesis is falsified if:

- the shallow A inflation is not associated with large sign-correct depth-2
  parent/leaf residuals;
- those residuals do not moderate when the branch is searched deeper;
- the same residual pattern is equally common in ordinary controls or
  legitimate tactical discovery positions;
- a positive v18 setting passes only by narrowing root search;
- fresh 400-simulation results do not move closer to the shipped 6,400-sim
  reference; or
- any safety, A/B/C/D, or strength gate fails.

### 1.3 Selection-level account — why this is not FPU relabelled

`_select_child` (`scripts/GPU/alphazero/mcts.py:1091-1114`) scores a visited
child as `q = -child.q_value` and an unvisited child as `q = fpu_value`, which
is `0.0` at the shipped setting. For the **observed positive-residual pattern**
in §1.1 — a depth-2 leaf backing up about `+0.79` in the leaf's own perspective
— clipping therefore acts in opposite directions at the two levels above it:

- **At the depth-1 node** (opponent to move) visited replies sit near `-0.79`
  against an unvisited score of `0.0`. Clipping lifts them toward `-0.41`,
  shrinking the gap, so a larger prior bonus is required before another
  unexplored reply outranks an already-visited one. Released budget goes to
  revisiting, i.e. to depth 3 and beyond.
- **At the root** (mover to move) the same clip lowers visited children from
  about `+0.79` toward `+0.41` against an unvisited score of `0.0`, reducing a
  visited child's advantage and permitting more root alternatives.

The preregistered aggregate signature on residual-exposed targets is therefore:

```text
root breadth              flat-or-up
depth-1 unique replies    modestly down
depth>=3 confirmation     up
```

FPU moved all three in the same direction. This signature is **falsifiable
evidence, not a structural guarantee of the formula**: the clip is symmetric,
and a large *negative* residual pushes each of these quantities the other way.
The claim is about the residual population v18 targets, and the development
sweep tests it rather than assuming it.

### 1.4 Selectivity observable and non-circularity

The runtime activation observable is only:

```text
relative search depth == 2
and leaf is nonterminal
and abs(sign-correct raw parent/leaf residual) > cap
```

Phase, side, game result, A membership, higher-budget value, eventual move
agreement, candidate result, policy-mass exposure, and whether the leaf later
proves right or wrong are not runtime inputs.

This is selective in application but still uses one global residual cap. It
therefore earns implementation only if the shipped-only preflight shows that
residual magnitude separates the measured shallow-horizon failure from matched
ordinary/tactical discovery positions. Merely showing that A has large
residuals is insufficient.

Fresh target selection may use shipped-only residual exposure to ensure the
mechanism is exercised. Candidate success may not use that same statistic. It
is judged by independent 6,400-simulation fidelity, move agreement, safety,
depth conversion, A/B/C/D, and strength.

### 1.5 Preregistered collateral-scaling prediction

The cap direction is the reverse of a penalty coefficient:

```text
cap 2.00  -> identity
cap 1.25  -> weakest proposed intervention
cap 1.00
cap 0.75
cap 0.50  -> strongest proposed intervention
```

As cap increases toward `2.00`, v18 predicts monotonically nonincreasing:

- unique clipped leaves;
- total absolute clipped amount; and
- direct per-leaf deviation from shipped backup.

This is the property the retired policy-mass family lacked. v17's control-flip
rate was flat at `0.31–0.50` across a grid whose reply reduction varied more
than threefold, so no coefficient was cheap. Here the direct effect vanishes
continuously as the cap rises, because a cap of `2.00` cannot bind on a
`tanh`-bounded value pair at all.

Search-level collateral such as move flips, collapse, breadth change, and
top-share change is not guaranteed pointwise monotonic because tree search is
discontinuous. Nevertheless, the preregistered expectation is that aggregate
collateral approaches shipped behavior at the weakest setting. If the weakest
setting already fails safety, the family stops before stronger settings run. If
observed aggregate collateral is grossly nonmonotonic, stop for review rather
than rationalizing the grid after the fact.

## 2. Evidence boundary

### 2.1 Existing evidence permitted for design

The following may motivate or falsify the mechanism but may not count as
fresh v18 acceptance evidence:

- v15 selected-A concentration and subtree artifacts.
- The shipped A 400/1,600/6,400 budget results.
- The v16 c_puct and selected-A absolute-FPU diagnostics.
- The v16a held-out result and its read-only postmortem.
- The v16 parent-relative production corpus and controls.
- The v17 development corpus, diagnostic rows, and null selection record.
- Established A/B/C/D artifacts only as historical acceptance definitions.
  B/C/D may not shape the v18 rule, cap grid, target predicate, or thresholds.

The v17 32-position corpus is consumed. It may be inspected read-only to
explain the completed v17 rejection, but no v18 cap, grid, threshold, or
candidate may be chosen from its candidate outcomes.

### 2.1.1 Narrow exception — historical unsafe operating points as a ceiling

The rule above and §10.3's use of historical anchors would otherwise conflict,
because v16a's and v17's reply-reduction figures *are* candidate outcomes. One
narrowly scoped exception is declared:

> A historical operating point that was **measured and rejected as unsafe** may
> impose a conservative **ceiling** on `R_max`, and nothing else.

Permitted use: `R_max` must lie strictly below the smallest historically
unsafe reply reduction. This can only make v18 harder to pass, and it encodes a
known failure region rather than fitting a threshold to a hoped-for result.

Forbidden use: these anchors may not determine `R_min`, the cap grid or its
spacing, the selector or any role predicate, the material-exposure floor, the
conversion-efficiency floor, the stable-leader minimum, or any pass bar. Those
come from the non-A discovery controls and budget logic per §2.2.3.

If this exception is not approved, remove the historical anchors from threshold
selection entirely and derive `R_max` from budget logic alone.

### 2.2 Permitted preflight before MCTS source edit

One shipped-only, read-only residual preflight is permitted:

- Reconstruct the historical selected-A searches with their authenticated
  shipped configuration and seeds.
- Reuse existing persisted Phase-0.5 data where it contains all required raw
  values and parent relationships; rerun shipped search only where necessary.
- Run shipped-only instrumentation on deterministic historical non-A
  discovery controls that are not established B/C/D acceptance positions.
- Record raw parent/leaf values, sign-correct temporal residuals, path depth,
  terminal status, raw policy context, completed visits, later revisit depth,
  and root contribution.
- Compute counterfactual clipping summaries from those immutable shipped
  values only. Do not run a v18 search configuration.
- Compute the static first-order crossover analysis of §2.2.1.
- Run exact-selector feasibility/sizing against authenticated historical
  shipped-only screen data after adding residual-exposure telemetry. This may
  size a future fresh reservoir only; it may not select a future v18 row.

#### 2.2.1 Static first-order crossover analysis

For each eligible depth-1 node in an authenticated shipped tree, and for each
proposed cap, compute the counterfactual selection scores that would have
applied at that node **holding the shipped tree fixed**.

A visited depth-2 child's `q_value` is a running mean, not its raw evaluation,
so the counterfactual must substitute the clipped initial contribution into the
accumulated sum. This is exact, because the expansion backup contributed
precisely `child.nn_value` to that child (`mcts.py:1145-1148`):

```text
clipped_initial =
    provisional_backup(child.nn_value, parent.nn_value, cap)

counterfactual_value_sum =
    child.value_sum - child.nn_value + clipped_initial

counterfactual_child_q =
    counterfactual_value_sum / child.visit_count

visited_score =
    -counterfactual_child_q
    + c_puct * prior * sqrt(parent.visit_count + 1)
      / (1 + child.visit_count)

unvisited_score =
    0.0
    + c_puct * prior * sqrt(parent.visit_count + 1)
```

Terminal children and synthetic no-legal-move children are excluded. Comparing
the best counterfactual visited score against the frozen unvisited-prior
distribution yields a static crossover estimate per node and cap.

**Prerequisite for exactness.** The substitution `value_sum - nn_value` is exact
only for a **synchronously** built tree in which the child's initial expansion
was backed up exactly once, so that precisely `nn_value` entered `value_sum` at
that expansion. All v18 evidence satisfies this (§4.3), but a tree produced by
the batched waiter path can back up one expansion to several waiters, and the
substitution would not be exact there. The preflight must assert the
synchronous provenance of every tree it analyses rather than assume it.

**What this analysis can support:** cap reach, plausible `R_min`/`R_max`, grid
spacing, selectivity thresholds, and the expected direction of the §1.3
signature.

**What it cannot support:** it does not reproduce sequential selection once the
candidate tree diverges from the shipped tree, and it therefore cannot
empirically derive candidate conversion efficiency. The §10.4 efficiency floor
is a **preregistered budget-conversion requirement informed by** this analysis,
never a candidate prediction recovered from a shipped tree.

#### 2.2.2 Shipped-only 6,400-simulation A move capture

The frozen historical 6,400-simulation A artifact records values and top shares
but **no selected move and no visit counts**. §12.1's move-agreement gate
therefore cannot be evaluated against it as written.

Preferred resolution, preregistered here and executed as part of the
shipped-only preflight, before any MCTS source edit:

- Re-run shipped search at 6,400 simulations on the 30 A rows with the
  authenticated historical configuration and seed rule, capturing selected move
  and full visit counts. Reuse the existing selected-move capture path
  (`capture_v17_abcd_selected_moves.py`) rather than writing a second one.
- **Parameterize, do not fork.** That script currently hard-codes
  `MCTS_SIMS, EVAL_BATCH_SIZE, STALL_FLUSH_SIMS, PENDING_VIRTUAL_VISITS =
  400, 14, 48, 8` at module scope (line 43), fixes all four gates in a `GATES`
  dict (line 51), and exposes only `--out`. The preflight parameterizes the
  simulation count and the gate subset so it can run A at 6,400, **while
  preserving its existing v17 default behavior byte-for-byte** — the existing
  batching assertion stays, and invoking it without the new arguments must
  reproduce the v17 capture exactly.
- **Authenticate by exact reproduction of both statistics:** the recomputed
  per-case root values **and** per-case top shares must reproduce the frozen
  historical 6,400-simulation artifact. Values alone are insufficient. A
  mismatch means the configuration or seed rule is not faithfully
  reconstructed, and the capture is discarded rather than adjusted.
- **Determinism check:** run the capture twice and require the two outputs to
  be byte-identical before either is bound.
- Bind the capture's bytes and SHA-1 into the v18 protocol as a frozen
  reference artifact.

This is a shipped-only measurement of an existing baseline. It runs no v18
configuration, selects no row, and sets no threshold.

**FREEZE BLOCKER:** if this capture is not authorized, or fails its exact
reproduction check, §12.1's move-agreement gate is **removed** from the A
requirements rather than evaluated against an unauthenticated reference. It is
not replaced by a 1,600-simulation or 400-simulation substitute.

#### 2.2.3 Where numeric thresholds may come from

Residual magnitude correlates with the statistic that *selected* the A rows, so
thresholds tuned on A would repeat the winner's-curse that closed the
calibration line.

- The selected-A set may be used **only** to demonstrate that the mechanism
  reaches the measured failure.
- Every numeric decision — exposure cut, material-exposure floor, `R_min`,
  `R_max`, efficiency floor, separation criterion, grid spacing — is derived
  from the matched non-A discovery controls and from budget logic.

#### 2.2.4 Preflight authority and limits

This preflight may:

- determine whether the proposed mechanism reaches the measured failure;
- confirm the terminal exemption and characterize tactical exposure using
  non-B/C/D discovery controls;
- size a provisional cap grid;
- size fresh development and held-out reservoirs for the final residual-based
  selector; and
- reject v18 before implementation.

It may not:

- count as development, held-out, A/B/C/D, or strength evidence;
- select a shipping candidate;
- change an established gate;
- use v17 candidate results to choose a cap; or
- authorize a source edit.

### 2.3 Preflight freeze blockers

Before approval, the preflight section of the frozen revision must record:

- exact input artifacts and SHA-1s;
- exact position counts and seeds;
- the perspective conversion used for every parent/leaf pair;
- depth-2 nonterminal residual distributions overall and by family;
- the fraction of positive A backup mass exposed to each proposed cap;
- how exposed branches behave when revisited or at higher budget;
- non-A discovery-control and terminal exposure;
- the §2.2.1 first-order crossover tables per cap;
- **the historical reply-reduction anchors, each traced to a source artifact.**
  All three are now resolved:

  | anchor | reply reduction | arithmetic | source artifact SHA-1 |
  |---|---|---|---|
  | v16a held-out `fpu_value=-0.20` | `0.28027286567513765` | `1 - 24583/34156`; 324 paired cases, 0 malformed | `6d15c7dd15bdc8e8a983700f536950bcc9830019` |
  | v17 weakest grid point `r=0.15` | `0.23358985966500678` | `516/2209` | `development_diagnostic.json` `af7778c84e1ea04f463febfc615e5363400d6aad` |
  | v16 selected-A FPU `-0.20` | `0.81836179163573375` | `1 - 734/4041 = 3307/4041` | `a_predrop_fpu_sweep_cases.csv` `f201f0f25b868e5c4c7103992054c7b4df5074d1` |

  Every anchor is now traced to an exact figure and an artifact SHA-1. The
  rounded forms that circulated earlier (`0.2336`, "about `0.818`",
  `134.7 -> 24.5`) are superseded by the exact values above and must not be
  used to set `R_max`.

  The v16a figure was recomputed from the case CSV rather than inferred; it is
  distinct from that run's effective-children reduction (`107.58 -> 70.92`,
  about `-34.1%`) and from its top-move flip rate (`27.16%`), which are
  different quantities that happen to sit near the same magnitude. Under
  §2.1.1 these anchors constrain `R_max` from above and nothing else;
- residual-target selector predicates, exact-selector trial counts, success
  criterion, and operational development/held-out sizes;
- the material-exposure floor for activation-positive flip controls;
- matched-control variables and tolerances, chosen without candidate or
  high-budget outcomes;
- monotonic direct-effect tables across the proposed cap ladder;
- the weakest-first run/stop logic and expected collateral scaling;
- the final cap grid and why each point exists; and
- a written PASS or FAIL against preregistered, numeric reach/separation
  criteria.

**FREEZE BLOCKER:** numeric preflight reach and family-separation thresholds
must be written before the preflight is run. The frozen revision must not
replace an unfavorable threshold after results are visible.

If the preflight fails, v18 ends without an MCTS source edit.

## 3. Proposed search rule

### 3.1 Perspective-safe formula

For a nonterminal leaf at relative search depth exactly 2:

```text
raw_leaf = leaf NN value, in leaf-to-move perspective
raw_parent = parent.nn_value, in parent-to-move perspective

parent_baseline_in_leaf_perspective = -raw_parent
residual = raw_leaf - parent_baseline_in_leaf_perspective

if residual > cap:
    backup_value = parent_baseline_in_leaf_perspective + cap
else if residual < -cap:
    backup_value = parent_baseline_in_leaf_perspective - cap
else:
    backup_value = raw_leaf
```

The final `else` must return the original `raw_leaf` object value rather than
reconstructing it as `baseline + residual`. This avoids a rounding change on
rows where the cap does not bind.

The comparison is strict (`>`), so `abs(residual) == cap` does not clip. §9.2's
identity-witness rule depends on this boundary being exact.

`backup_value` then enters the existing sign-alternating `_backup()` path
(`mcts.py:1130-1148`), which flips sign starting at the leaf. A depth-2 leaf
shares the root's side to move, so the residual baseline is expressed in the
same perspective as the root's accumulated value.

The value head is `tanh` (`scripts/GPU/alphazero/network.py:472`). For
implementation purposes raw values are treated as lying in `[-1, +1]` and the
residual in `[-2, +2]`, using closed intervals because saturated
floating-point `tanh` can return exactly `±1.0`. The identity guarantee at cap
`2.0` does **not** rest on this range: it is the structural branch of §4.3,
which returns before any parent read or residual computation.

### 3.2 Exact scope and eligibility

The rule applies only when all are true:

- the v18 field is active at a cap strictly below the identity sentinel;
- the search path is exactly `[root, depth1_parent, depth2_leaf]`;
- the depth-2 leaf is nonterminal;
- `leaf.priors` is non-empty; and
- `raw_leaf` and `raw_parent` are both finite.

It does not apply to:

- root expansion;
- depth-1 leaf expansion;
- depth 3 or deeper;
- any terminal result;
- any forced result or synthetic terminal value;
- a disabled or identity-sentinel configuration; or
- existing statistics during root reuse before a new leaf is expanded.

Three degenerate-input rules, distinguished because they have different causes:

- **Empty `leaf.priors`.** `_expand_batch` (`mcts.py:923-927`) assigns a
  synthetic `nn_value = 0.0` to a node with no legal moves, and `is_expanded`
  is `priors is not None` (`mcts.py:296`), so such a node is nominally
  expandable. On a 24 board a nonterminal position always has legal moves, so
  this is unreachable in practice. The rule nonetheless bypasses clipping,
  preserves the shipped `0.0`, and records `ineligible_no_legal_moves`.
- **Empty `parent.priors`.** Asserted invariant. Descent only continues through
  expanded nodes and `_select_child` cannot select from an empty prior map, so
  a parent with empty priors indicates corruption and must fail.
- **Nonfinite `raw_leaf` or `raw_parent`.** Fails. Silently bypassing an
  invalid value would conceal corruption.

Relative depth is measured from the current search root, not from game ply or
the lifetime of a reused tree. `advance_root()` detaches the new root; the
current `search_path` remains the authority.

### 3.3 Why depth exactly 2

Depth 2 is the measured horizon:

- depth-1 raw values were not the source of the selected-A searched excess;
- 77.3% of the selected-branch leaf evaluations were at depth 2;
- depth-3 and deeper evidence is the confirmation v18 intends to preserve;
  and
- applying the rule at all depths would be a broader value transformation not
  supported by the evidence.

Changing the scope to `depth >= 2`, game-ply bands, branching thresholds,
policy entropy, or an A-specific predicate is a different mechanism and
requires a new design.

### 3.4 Symmetry and terminal truth

The cap is symmetric in the sign-correct residual. It may limit an unusually
positive or unusually negative first estimate by the same magnitude. There is
no black-only, red-only, winning-only, or A-only branch.

Terminal values are ground truth and bypass the rule. Because the call site
sits inside the nonterminal branch (§4.3), this exemption is structural at
every backup site rather than a runtime flag. A candidate that needs terminal
clipping to pass is rejected rather than broadened.

## 4. Configuration and implementation contract

### 4.1 Field

Add one distinct optional MCTS field:

```python
depth2_provisional_backup_cap: float | None = None
```

Semantics:

- `None`: shipped search, with no new helper call or path inspection.
- `2.0`: explicit v18 identity sentinel; execute the same structural shipped
  branch as `None`.
- `0.0 < cap < 2.0`: enable the v18 depth-2 rule.
- Reject booleans, nonfinite values, `cap <= 0.0`, and `cap > 2.0`.

The identity sentinel is 2.0 because two `tanh` values can differ by at most
2.0. More importantly, the call site branches structurally before reading the
parent or calculating a residual, so identity does not depend on that range
assumption.

Boolean rejection is explicit because `bool` is a subclass of `int` in Python
and `True` would otherwise be accepted as `1.0`.

The field must not be named as FPU, policy mass, calibration, or value-head
training. It is a search backup rule.

### 4.2 Pure helper

Add one pure helper. It returns a compact result rather than a bare float, so
that §4.4's telemetry columns come from the same computation that produced the
backup value and the call site never recomputes a residual:

```python
def provisional_depth2_backup_value(
    raw_leaf_value: float,
    raw_parent_value: float,
    cap: float,
) -> ProvisionalBackup:      # backup_value, residual, clipped_amount,
    ...                      # clip_direction
```

The helper:

- accepts raw values in their native opposing perspectives;
- performs the sign conversion internally and documents unmistakably which
  perspective each argument uses;
- rejects nonfinite inputs;
- returns `raw_leaf_value` unchanged, with `clipped_amount == 0.0` and
  `clip_direction == 0`, when the residual is within the cap;
- is symmetric under simultaneous perspective reversal; and
- contains no node traversal, evaluator access, RNG, or state mutation.

There must be one formula implementation. The diagnostic, the screen, the
preflight, and the read-only walker all import it rather than copying it.

### 4.3 Single call site and batched refusal

The code audit establishes which paths matter. `_backup` has three call sites:

| site | file:line | reachable leaves |
|---|---|---|
| batched terminal | `mcts.py:667` | terminal only |
| synchronous | `mcts.py:831` | terminal or expanded |
| batched waiters | `mcts.py:1030` | nonterminal only |

The batched waiter path (`search_from_root` -> `_flush_pending_batch`) is
reached only from `self_play.py`. **Every v18 stage uses the synchronous path**
`search()` / `search_with_root()` -> `_run_single_simulation`: reservoir
generation (`eval_checkpoint_match` -> `eval_runner.py:246`), the corpus screen
(`fpu_dev_corpus_v2.py`), every diagnostic, every A/B/C/D probe
(`eval_position_probe.py:77`, `eval_goal_line_trigger_probe.py`), and both
strength matches. v18 does not adopt into self-play.

The contract is therefore:

```text
cap is None or 2.0:
    synchronous and batched paths retain shipped behavior

0.0 < cap < 2.0:
    synchronous _run_single_simulation supports v18
    batched search_from_root refuses
```

Required properties:

- The helper is called only in `_run_single_simulation`, immediately after
  nonterminal expansion and before `_backup`. There is no resolver method; one
  call site does not need one.
- `search_from_root` rejects an active v18 cap **before** root expansion, any
  evaluator **call**, RNG consumption, tree mutation, observer callback, and
  any search output. Two boundaries it cannot guarantee, stated so the claim is
  not overread: it cannot refuse before observer *attachment* or before
  evaluator *construction*, both of which precede the call in `MCTS.__init__`.
  Refusing before evaluator construction is an outer-runner responsibility and
  is covered separately by the §4.3 runner-routing validation.
- Batched waiter telemetry and batched equivalence requirements are removed
  from v18 entirely.
- Tests prove `None` and `2.0` leave the batched path byte-identical.
- A negative test proves an active cap reaches zero evaluator calls on the
  batched path.
- Every scientific v18 runner is validated to route through the synchronous
  path, and every protocol/artifact records
  `search_execution_mode=synchronous`.
- A mutation test kills removal or weakening of the batched refusal.

`_select_child()`, `policy_mass_fpu()`, `explored_policy_mass()`, priors, child
visit counts, pending virtual visits, and RNG tie behavior are outside the v18
change.

If v18 eventually passes every gate, self-play adoption is a separate
authorization boundary at which the waiter implementation is designed, tested,
and validated on its own terms — not smuggled into the initial experiment.

### 4.4 Raw-value and telemetry contract

The evaluator output remains authoritative raw data:

- `node.nn_value` stores the exact evaluator result and is never overwritten
  with the provisional backup.
- `_backup` credits the **leaf itself** as well as its ancestors
  (`mcts.py:1145-1148`). The first backup therefore contributes the *clipped*
  value to `leaf.value_sum`, so `leaf.q_value` reflects the provisional value
  while `leaf.nn_value` stays raw. This is the intended propagation path: later
  depth-3 evidence joins the leaf's running mean normally and can move it away
  from the provisional value.
- No diagnostic field may call the provisional value `nn_value`.

**All aggregate v18 telemetry is derived from the final tree by one shared
read-only walker, not by hot-path instrumentation.** `search_with_root` already
returns the root and the existing diagnostics already consume it
(`diagnose_fpu_policy_mass.py:1095`). The derivations are exact on the
synchronous path, which adds no virtual visits to `visit_count`:

```text
backups terminating at n =
    n.visit_count - sum(child.visit_count for child in n.children)

residual at a depth-2 node =
    n.nn_value - (-parent.nn_value)

clip events = one per clipped depth-2 node
    (each leaf is expanded exactly once on the synchronous path)
```

This yields, without touching the search hot path: the terminating-depth
histogram and its restriction to any root child's subtree, unique depth-3
descendants, follow-up completed visits per explored depth-2 reply, and the
revisit-to-depth-3 rate of provisionally backed-up leaves.

Per-row scientific columns: raw leaf, raw parent, sign-correct residual,
resolved backup, whether clipping occurred, clip direction, and absolute
clipped amount. Aggregate columns: eligible depth-2 leaves, clipped depth-2
leaves, positive/negative clip counts, and total/mean/max absolute clipped
amount.

Because there is one call site and one expansion per leaf, unique clipped
leaves and clip events coincide; no unique-versus-waiter distinction exists in
v18.

### 4.5 Batching and budget

All v18 runs record:

```text
eval_batch_size:        14
stall_flush_sims:       48
pending_virtual_visits: 8
```

Protocol validation checks the triple against the existing v17 authority
(`fpu_v17_provenance.py:42`) before evaluator load, and free CLI overrides are
rejected.

**Recorded status of this triple.** These three fields are read only by
`search_from_root` and by `_select_child`'s pending-visit penalty
(`mcts.py:1110`), which the synchronous path never triggers. They are therefore
binding provenance for reservoir generation and inert for the diagnostics and
matches. Comparability of v18 evidence to v17 evidence rests on the shared
synchronous path, seeds, budget, and `add_noise=false` — not on this triple.
The triple is still validated, because a mismatch would indicate an
unauthorized configuration change.

The v18 rule does not add evaluator calls. Every 400-simulation comparison
remains equal-budget by simulations and logical leaf evaluations. A
higher-budget shipped reference is a diagnostic reference, not the competing
agent in the final strength match.

## 5. Explicit reuse and non-reuse

### 5.1 Thin v18 layer, no extraction

The staging stack is already largely rule-agnostic:

| module | lines | v17-specific |
|---|---|---|
| `fpu_dev_reservoir_protocol.py` | 3,234 | 2 mentions |
| `fpu_provenance.py` | 118 | none |
| `fpu_state_hash.py` | 88 | none |
| `fpu_dev_corpus_v2.py` | 4,848 | role vocabulary + allocation |
| `fpu_v17_protocol.py` | 265 | genuinely v17 |
| `fpu_v17_provenance.py` | 352 | genuinely v17 |

Moving roughly 3,400 already-working lines because their module names mention
v17 would create substantial regression risk in the code that guards the
evidence chain, without improving the scientific test. The completed v17
evidence does not benefit from architectural cleanup.

The v18 rule is therefore:

- Import existing neutral helpers from their current modules, even where the
  names are historically awkward.
- Add only narrow, version-dispatched hooks needed for v18 residual roles.
  `fpu_dev_corpus_v2.py`'s `_ROLES` vocabulary and its `(role, phase)`
  allocation are already config-parsed and are the expected hook points.
- Preserve v17 defaults, schemas, constants, outputs, and call paths exactly.
- Create thin v18-specific modules for mechanism/provenance constants, protocol
  adaptation, residual telemetry and role definitions, and v18 gates and cap
  selection.
- Do not duplicate the selector or its qualification/binding logic.

Required protection:

- Existing v17 fixtures and the full suite remain green.
- Real v17 producer inputs still emit byte-identical outputs.
- v18 roles are rejected under v16/v17 schema and run identities.
- v18 role thresholds remain config-authoritative.
- A duplicate-definition audit proves one implementation each of: the selector,
  canonical position identity, corpus binding, gate metrics, the read-only tree
  walker, and the clip formula.

Full extraction happens only if a concrete v18 requirement cannot be expressed
through a narrow hook, and then as a separately reviewed refactor before the
scientific protocol is emitted.

Also reused by import or narrow parameterization: `MCTSNode.nn_value`,
`_expand_batch()`, `_backup()` and existing sign-alternating backup semantics;
the batching validator; canonical selected-move, prior-rank, collapse,
effective-children, p95, lower-prior-flip, seed, and pairing definitions
already used by v17; reservoir protocol creation, generation-command
derivation, qualification, qualification recheck, raw-policy/anchor screen,
post-screen qualification, and deterministic selector; corpus reconstruction
through `load_manifest()` and the source-index path; corpus binding by manifest
SHA-1, source-index SHA-1, and selected-replay-data SHA-1; canonical JSON/CSV
writing, exact schemas, artifact kind/version checks, tamper detection,
worktree-clean refusal, and source identity checks; same-checkpoint asymmetric
agent support in `eval_runner.py`, `eval_checkpoint_match.py`, and
`eval_summary.py`; A/B/C/D baseline authentication and exact selected-move
capture; and draw-aware trinomial confidence intervals with paired block
bootstrap.

### 5.2 Must remain distinct

Do not edit or reinterpret:

- `fpu_policy_mass_reduction`;
- `fpu_shipped_policy_mass_reduction`;
- v16/v17 formula IDs, design hashes, protocols, configs, results, or
  selection records;
- the v17 null or its gate table;
- historical reservoir, screen, corpus, and replay bytes; or
- any A/B/C/D frozen baseline.

Create v18-specific provenance and protocol identities under:

```text
logs/eval/v18_depth2_provisional_backup/
```

Proposed code modules:

```text
scripts/GPU/alphazero/v18_provisional_backup_provenance.py
scripts/GPU/alphazero/v18_provisional_backup_protocol.py
scripts/GPU/alphazero/diagnose_v18_provisional_backup.py
```

These names are provisional until plan review. Formula and gate constants must
have one source of truth.

### 5.3 Branch and ancestry

v18 branches from `fpu-v17-baseline-policy-mass`, which is where the imported
machinery lives. At revision time that branch is 34 commits ahead of `main`,
zero behind, with a clean tree apart from this untracked draft.

Recorded consequence:

> The v18 branch contains the complete v17 ancestry. Merging v18 into `main`
> will therefore also bring the unmerged v17 commits unless v17 is integrated
> first or the branch is deliberately rebased. v17's merge may be decided
> separately and at any time, but eventual v18 integration cannot pretend the
> ancestry is independent.

The execution constraint:

```text
finish source edits
-> commit
-> clean tree
-> emit protocol at that HEAD
-> generate
-> qualify
-> screen
-> select
```

No HEAD movement inside that chain. A source edit after qualification forces
re-qualification from the protocol down; a commit between generation and
qualification makes the chain non-requalifiable because conformance compares
the current HEAD against `generation_git_commit`.

## 6. Mechanistic predictions and principal risks

### 6.1 Predictions

- `None` and explicit cap `2.0` produce byte-identical search results, trees,
  observer callbacks, and RNG behavior.
- Positive v18 caps leave root and depth-1 raw backups unchanged.
- A binding cap changes only large depth-2 parent/leaf disagreements.
- On residual-exposed targets, the §1.3 signature holds in aggregate.
- Root breadth and top-child reply breadth remain materially closer to shipped
  than under rejected FPU candidates.
- On fresh targets, candidate 400-simulation values move closer to the shipped
  6,400-simulation reference.
- Persistent tactical evidence can still move the search because terminals and
  depth-3-or-deeper evaluations are unmodified.

### 6.2 Risks

- The parent raw value may itself be wrong, turning the cap into a horizon
  delay rather than a correction.
- Real tactical one-ply changes may be clipped before deeper confirmation.
- At 400 simulations against roughly 500 legal moves, most depth-2 leaves are
  never revisited, so for those leaves the "provisional" value is final. The
  §10.4 depth-conversion gates exist to measure how far this is true rather
  than to assume it away.
- Goal-line B may regress even with terminal truth exempt.
- The rule may change root moves indirectly despite not touching root
  selection.
- Apparent A progress may be value compression rather than better move choice.
- Candidate results may approach the 6,400-simulation root value while playing
  strength does not improve.
- A residual-target predicate may be too rare or unstable to support an
  adequately powered fresh corpus at an acceptable reservoir size.

Every risk has a stop gate below. A progress alone is never sufficient.

## 7. Provisional cap grid

**FREEZE BLOCKER:** the final grid is chosen only after the shipped-only
preflight and must be fixed before MCTS source edit.

The current draft grid is:

```text
identity: cap = 2.00
candidates: cap ∈ {1.25, 1.00, 0.75, 0.50}
```

Interpretation:

- `1.25` affects only extreme residuals and is the safety point.
- `1.00` is a modest bound over the full value range.
- `0.75` should reach the historical mean depth-1/depth-2 disagreement, which
  §1.1's numbers put near `0.88`.
- `0.50` is the strongest permitted test; it remains a provisional update, not
  replacement with the parent value.

No cap below `0.50` is proposed. A near-zero cap would approximate copying the
parent raw value and would test a materially different mechanism.

Selection rule: choose the **largest cap** passing every development safety and
mechanism gate, because it is the least intervention. If none pass, reject v18.
Do not interpolate, add caps, lower the floor, or consult later stages.

The frozen revision must replace the qualitative rationales above with
preflight-supported reach numbers.

Positive settings run sequentially, weakest first:

```text
1.25 -> 1.00 -> 0.75 -> 0.50
```

For each setting:

1. Complete and authenticate all paired rows.
2. Evaluate §10.3 safety before §10.4 mechanism.

Every outcome has exactly one defined consequence:

| outcome | consequence |
|---|---|
| **any** §10.3 safety gate fails | **reject v18 immediately**; run no stronger setting |
| safety passes, **any** §10.4 mechanism gate fails | **advance** to the next stronger preregistered cap |
| safety passes and all §10.4 gates pass | **select this cap immediately**; run no stronger setting |
| grid exhausted with no selection | **reject v18** |

The routing is by *gate section*, not by failure reason. Revision 2 said
"advance on insufficient reach", which left the consequences of a §10.4
fidelity, move-agreement, depth-conversion, efficiency, or stable-leader
failure undefined. The rule is now: **if a condition should terminate the
family rather than advance the ladder, it belongs in §10.3 and must be moved
there explicitly.** Revision 3 moved one such condition — target prior-rank
regression — from §10.4 to §10.3.

That placement is a **conservative preregistered stop rule, not a claim of
proven monotonicity.** §1.5 states that search-level collateral — move flips,
collapse, breadth change, top-share change — is *not* guaranteed pointwise
monotonic in the cap, because tree search is discontinuous, and prior-rank
regression is exactly that kind of quantity. The justification is
decision-theoretic, not mechanical: prior-rank regression is the signature of
the failure that killed v16, we have no basis for expecting a stronger cap to
repair it, and continuing the ladder after it fires would spend compute on
settings we would not adopt. Stopping is the safe default under uncertainty,
not a prediction about the next grid point.

This order prevents spending the full grid after the weakest intervention has
already demonstrated unacceptable collateral. It also prevents choosing a
stronger passing cap after a weaker cap has already qualified.

## 8. Stage 0 — software proof and tooling smoke

Stage 0 requires separate authorization after the design is frozen.

### 8.1 TDD and structural proofs

Before any positive scientific result:

- Pure-helper perspective, symmetry, boundary (`abs(residual) == cap` does not
  clip), finite-value, and no-op tests, including the compact-result fields.
- `None` and `2.0` structural shipped-branch tests.
- Exact-path tests for depth 0, 1, 2, 3, and terminal depth 2.
- Tests proving only depth exactly 2 can change.
- Tests proving `node.nn_value` remains raw while `leaf.q_value` reflects the
  clipped first backup.
- Empty-`leaf.priors` bypass test and empty-`parent.priors` failure test.
- Nonfinite raw value failure tests.
- Forced-root-visit path coverage.
- Tree-reuse tests proving relative depth is measured from the current root.
- Batched path: `None` and `2.0` byte-identical; active cap refuses with zero
  evaluator calls, no tree mutation, no RNG consumption, no observer callback,
  and no output.
- Read-only walker tests proving the terminating-depth identity
  `visit_count(n) - sum(child visit counts)` against a constructed tree, and
  proving the walker and the search agree on clip counts.
- Full CPU golden: visits, root value, tree signature, callback sequence, and
  RNG trace for shipped versus identity sentinel.
- Positive non-vacuity: at least one constructed depth-2 residual binds and
  changes the backed value by the exact expected amount.
- Tests proving `_select_child`, FPU helpers, priors, and visit-count updates
  are untouched.
- Import purity, exact schemas, protocol/config binding, tamper refusal, and
  clean-worktree refusal.
- Duplicate-definition audit (§5.1).
- Full repository suite.

Mutation tests must at least kill:

- applying at depth 1;
- applying at depth 3;
- clipping terminals;
- failing to negate the parent perspective;
- overwriting `nn_value`;
- reconstructing an unchanged raw value arithmetically;
- treating cap `2.0` as an active formula path;
- accepting an off-grid scientific cap;
- removing or weakening the batched refusal; and
- recomputing the residual at the call site instead of using the helper result.

### 8.2 Full-chain smoke

Reuse the real v17 reservoir chain with a new v18 output root and unused seeds.
The frozen revision must record a collision-audited range.

Proposed smoke:

```text
games:        32
board:        24
MCTS:         400 simulations
workers:      4
replays:      enabled
selection:    opening_temperature
opening temperature plies: 20
temperature:  1.0 early / 0.1 late
max moves:    280; timeout is a draw
root noise:   add_noise=false
run kind:     tooling_smoke
```

Run protocol emission/check, generation, qualification/check, screen,
post-screen qualification, and selector. Then run two selected controls under:

```text
shipped/None, identity cap=2.0, one middle positive cap
```

The smoke proves only real-path execution, exact identity, positive plumbing,
telemetry, schemas, deterministic bytes, and refusal behavior. Every output
must stamp `run_kind=tooling_smoke` and
`scientific_interpretation_forbidden=true`.

Run a deliberate wrong-depth/tampered-protocol failure and prove refusal before
evaluator load with no output.

### 8.3 Same-checkpoint runner smoke

Reuse the asymmetric same-checkpoint runner:

```text
shipped vs shipped: 4 games
shipped vs one positive cap: 4 games
colors/configs: exactly 2/2 per block
budget: 400 simulations for both
root noise: add_noise=false
run kind: tooling_smoke
```

Agent identity is `(agent_id, checkpoint, complete MCTS config)`. Both sides
must carry the frozen batching triple. Scores have no scientific meaning.

### 8.4 Runtime calibration before production authorization

v17 generation took about 12.6 hours and its shipped-only screen about 11.5
hours; earlier estimates were wrong by factors of roughly three and ten in
opposite directions. v18 therefore does not authorize production from an
arithmetic estimate alone.

Stage 0 records measured wall time and throughput for:

- generation games per hour;
- screen positions and shipped searches per hour;
- residual tree-walk telemetry overhead;
- each 400-simulation diagnostic row;
- each 1,600- and 6,400-simulation reference row; and
- same-checkpoint games per hour.

Before Stage 1 authorization, publish:

- the measured smoke timings;
- the exact selector-sized development and held-out counts;
- a low/central/high runtime estimate;
- the maximum single-stage uninterrupted runtime;
- expected artifact write points and restart semantics; and
- the durable execution/monitoring method, which is `nohup` plus `disown`
  (harness background tasks are killed, and `setsid` does not exist on macOS).

No partial reservoir, top-up, or resumed scientific screen is permitted unless
the frozen protocol explicitly implements and authenticates resume semantics.
Absent that support, interruption requires a clean full rerun of the affected
stage.

## 9. Stage 1 — fresh development reservoir and corpus

No v16 or v17 reservoir, screen, manifest, replay, or position is reused as
v18 scientific evidence.

### 9.1 Generation

Proposed settings reuse the qualified v17 production chain:

```text
checkpoint A: calib020_0001
checkpoint B: 0379
games:        provisional 1,600; final count set by exact-selector sizing
board:        24
MCTS:         400 simulations
workers:      4
replays:      enabled
top-up:       forbidden
selection:    opening_temperature
opening temperature plies: 20
temperature:  1.0 early / 0.1 late
max moves:    280; timeout is a draw
root noise:   add_noise=false
```

**FREEZE BLOCKER:** choose and collision-audit a new seed range. Do not reuse
v17 `[20310000, 20311600)`.

The generation matchup score is metadata only.

### 9.2 Deterministic development corpus

Reuse the authenticated v17 selector, qualification, and geometry machinery on
fresh bytes, but do **not** reuse v17's policy-mass target predicate.

The v18 shipped-only screen adds mechanism-specific telemetry, produced by the
§4.4 read-only walker:

- eligible nonterminal depth-2 unique leaves;
- positive/negative residual counts and quantiles;
- maximum absolute eligible depth-2 residual;
- counterfactual bind/excess counts for every frozen cap;
- contribution-weighted positive residual exposure;
- terminal count; and
- shipped root value, entropy, top prior, phase, side, and branching census.

Final role predicates are **FREEZE BLOCKERS**. They must be derived from the
shipped-only preflight per §2.2.3, frozen before fresh generation, and use no
candidate or higher-budget outcome.

#### 9.2.1 Four roles

```text
16  residual-exposed targets
 4  activation-negative identity witnesses
 4  activation-positive flip controls
16  representative phase controls
40  total
```

**Targets (16).** Late, satisfying the frozen near-even and
minimum-eligible-leaf rules, above the frozen **positive**-residual-exposure
threshold defined below.

**FREEZE BLOCKER — the exposure statistic must be exact.** "Residual exposure"
admits at least four incompatible readings: absolute-residual count,
positive-residual count, clipped-amount-weighted mass, and
root-contribution-weighted mass. They select different rows. §1.3's directional
prediction — root breadth flat-or-up, depth-1 replies down, depth `>=3` up — is
a claim about the **positive**-residual population specifically; a row loaded
with large *negative* residuals would push each of those quantities the other
way, so an absolute-value statistic can select rows that cancel or reverse the
predicted signature in aggregate. The frozen revision must fix, for the
strongest cap in the grid:

- the exact formula, over eligible nonterminal depth-2 leaves, in the form
  `positive_exposure = f({residual_i : residual_i > 0})`, naming whether `f` is
  a count above a threshold, a summed clipped amount, or a
  contribution-weighted mass;
- the numeric selection threshold on that statistic; and
- a **sign-dominance requirement** ensuring positive residual mass dominates
  negative residual mass on every selected target, with the exact ratio or
  margin.

Negative-residual exposure is recorded for every row in every role, and is
reported in the results, but does not select targets.

**Identity witnesses (4).** Matched to the same late, near-even, high-branching
regime, but with shipped `max|eligible depth-2 residual| <= 0.50`. Because the
clip comparison is strict, no candidate cap in the grid can bind. Requirement
under **every** positive cap: complete scientific-result identity — same
selected move and visit counts, same root value bits, same tree signature, same
canonical `search_result_sha1`, and zero clipped leaves. These can fail if v18
leaks outside its specified activation path; they are not collateral evidence
about an active cap.

**Activation-positive flip controls (4).** Matched to the same regime, with
shipped-only telemetry proving material exposure at the weakest cap `1.25` —
a preregistered minimum count of eligible leaves with `abs(residual) > 1.25`,
and/or a preregistered minimum counterfactual clipped amount at `1.25`. One
barely exposed leaf is technically nonzero but scientifically weak, so the floor
is a freeze blocker. Because they are exposed at `1.25` they remain exposed at
every stronger cap. Requirement: zero lower-prior selected-move flips. These
make the matched-control flip gate non-vacuous at every grid point.

**Representative phase controls (16).** Four per phase, satisfying the same
near-even rule but selected independently of residual magnitude after excluding
target rows. Their residual-exposure distribution is reported, not minimized.
They carry the pooled collateral gate.

#### 9.2.2 Side geometry

Exact per-role side balance, subject to exact-selector sizing:

```text
targets:                  8/8
identity witnesses:       2/2
exposed flip controls:    2/2
representative controls:  8/8
overall:                 20/20
```

#### 9.2.3 Shipped exposure versus actual activation

Role labels are established from shipped-only counterfactual exposure and are
part of the immutable corpus. Actual per-cap clip counts are recorded from
candidate runs and are used as **artifact/role authentication**:

- An identity witness that produces any clip, or any scientific-result
  difference, is an implementation or protocol failure.
- A flip control that produces zero actual clips at the weakest cap invalidates
  that candidate artifact/run.

In both cases the failure invalidates the candidate artifact or run — not the
immutable corpus bytes — and it never weakens the corresponding gate.

#### 9.2.4 Retained corpus invariants

- At most two positions per game, separated by at least 12 plies.
- Whole-game split isolation.
- Complete-state canonical uniqueness.
- Complete game and position disjointness from all v16/v17 evidence and
  established A/B/C/D.
- New collision-audited selector seed.
- Branching bands recorded, not post-hoc gated.

The existing anchor eligibility, qualification, screen, exact selector,
source-index join, and forbidden-hash loaders remain the authority. Extend the
role vocabulary and allocation through one config-authoritative v18 definition;
do not fork a second selector or encode thresholds in the diagnostic.

#### 9.2.5 Sizing

The 1,600-game v17 size is only an initial sizing tier. It cannot certify a new
residual-based predicate or the larger four-role geometry. Before freeze, run
deterministic whole-game exact-selector sizing on authenticated historical
shipped-only telemetry, using a preregistered resampling seed, ladder, trial
count, exact success criterion, and next-tier margin rule. The frozen design
records the resulting operational count. If the future exact selector cannot
fill the corpus, stop; do not top up or relax geometry.

## 10. Stage 2 — development sweep and references

Run on all 40 rows:

```text
shipped/None at 400 simulations
identity cap=2.0 at 400 simulations
each frozen positive cap at 400 simulations
```

On the 16 targets only, also run the shipped mechanism at both reference
budgets.

**Reference policy.** One formal reference throughout the protocol:

```text
formal reference:      shipped search at 6,400 simulations
trajectory diagnostic: shipped search at 1,600 simulations
```

Formal metrics:

```text
reference_value_error =
    abs(candidate_400_value - shipped_6400_value)

reference_move_agreement =
    candidate_400_move == shipped_6400_move
```

Cap selection and held-out transfer gates use only the 6,400-simulation
metrics. Gate A already uses the frozen 6,400-simulation `V_REF=-0.0451`, so
the protocol has a single notion of "closer to truth", and historical A was
materially different at 1,600 (`+0.0626`) versus 6,400 (`-0.0451`) — gating
against 1,600 could select a candidate that merely reproduces an intermediate
shallow-search state.

The 1,600-simulation result is descriptive only. It answers whether shipped
search moves consistently from 400 toward 6,400, whether candidate 400
resembles an intermediate stage or the deeper result, and whether an apparent
improvement is a smooth depth trajectory or a discontinuous value shift. **It
cannot pass, fail, rescue, select, exclude, or reclassify a position.**

Both shipped reference artifacts are completed and bound before any positive
cap runs. Position selection remains based solely on shipped 400-simulation
telemetry; neither reference may filter or replace rows afterward. If measured
runtime makes the 1,600 pass unjustifiable it is removed before freeze; it is
never promoted to the formal gate.

All paired 400-simulation searches use identical per-position seeds and
`add_noise=false`. Each reference budget uses its own explicitly derived and
recorded seed rule.

### 10.1 Reused metric semantics

Import without reinterpretation:

- canonical selected move and tie comparator;
- raw prior rank;
- top share and collapse at `>=0.95`;
- effective children from completed root visit counts;
- completed-visit reply count below the final root leader;
- lower-prior control flip;
- mover-value delta;
- sorted linear-interpolation p95;
- complete row pairing and finite-value validation.

Add v18-specific metrics, all from the §4.4 walker:

- `eligible_depth2_unique_leaves`;
- `clipped_depth2_unique_leaves`;
- positive and negative clip counts;
- total, mean, and max absolute clipped amount;
- root and final-leader reply breadth deltas;
- completed-backup depth histogram relative to the current root;
- the same histogram restricted to the root leader's subtree;
- fraction of completed backups originating at depth `>=3`;
- unique depth-3 descendants;
- follow-up completed visits per explored depth-2 reply;
- fraction of provisionally backed-up depth-2 leaves subsequently revisited to
  depth 3 or deeper;
- `reference_value_error` and `reference_move_agreement` against the 6,400-sim
  reference;
- the reference move's prior rank and final visit rank under each 400-sim
  config; and
- the 1,600-simulation trajectory columns, marked non-gating.

Reference error is a mechanism diagnostic, not ground truth. Final adoption
still requires playing strength.

#### 10.1.1 Exact formulas for the new conversion metrics

"Import without reinterpretation" (§10.1) covers the historical reply count and
the other v17 definitions. It does **not** cover the conversion metrics
introduced by v18, which have no prior definition and whose denominators change
what the gates mean. They are fixed here.

Let `L` be the root's final visit leader and `root.visit_count` the completed
simulation count, which equals `n_simulations` exactly on the synchronous path
(the existing diagnostics already assert this).

Each arm has its **own** final visit leader. Write `L_shipped` and
`L_candidate`; an unqualified `L` never appears in a formula.

```text
replies(L_arm) = #{child of L_arm : child.visit_count > 0}
                 (the imported v17 definition, unchanged)

aggregate reply reduction =
      1 - sum(candidate replies(L_candidate))
        / sum(shipped   replies(L_shipped))
    summed over the STAGE'S FROZEN EVALUATION SET -- 16 development
    targets, 24 held-out targets, or the 30 A rows -- not a fixed count.
    A zero shipped denominator is invalid, not an automatic pass.

    This statistic deliberately compares each arm under its own leader,
    so it remains defined when the leader changes. The same-leader
    restriction belongs only to Sec 10.4.1's conversion efficiency,
    which is a different question and must not reuse this denominator.

backups_terminating_at(n) =
    n.visit_count - sum(child.visit_count for child in n.children)

depth>=3 backup count =
    sum of backups_terminating_at(n) over all n at relative depth >= 3
    (whole tree, from the current root)

depth>=3 fraction = depth>=3 backup count / root.visit_count
    The denominator is the full simulation budget, identical for shipped
    and candidate, so count and fraction are equivalent up to a constant.

explored_replies(L_arm) =
    children of L_arm with visit_count > 0, EXCLUDING terminal children
    and children ineligible under Sec 3.2 (empty priors)

follow_up_visits_per_explored_reply =
      sum(child.visit_count - 1 for child in explored_replies(L_arm))
    / |explored_replies(L_arm)|
    Numerator counts visits after the first-touch expansion. An empty
    denominator is invalid, not zero.

would_clip(arm, cap) =
    depth-2 nodes in that arm's tree, eligible under Sec 3.2, whose
    OWN raw values satisfy abs(residual) > cap
    Defined identically in both arms from each arm's own tree. It is a
    counterfactual predicate in the shipped arm and an actual one in the
    candidate arm.

revisit_to_depth3_rate(arm, cap) =
      #{n in would_clip(arm, cap) : some child of n has visit_count > 0}
    / |would_clip(arm, cap)|
    An empty denominator is invalid, not zero.
```

**Why `would_clip` and not "clipped nodes".** Revision 3 defined the revisit
rate over *actually clipped* nodes, which is empty in the shipped arm by
construction — so §§10.4 and 11's requirement that the rate "increase versus
shipped" compared a candidate number against an undefined one. Evaluating the
same counterfactual predicate in both arms restores a real paired comparison.
The two arms' `would_clip` sets are not the same nodes, because the trees
diverge; that is inherent and is the same property `replies(L_arm)` already
has.

**Authentication invariant.** In the candidate arm, `would_clip(candidate, cap)`
must equal the set of nodes actually clipped, exactly. A discrepancy means the
runtime activation path and the analysis predicate disagree, which invalidates
the candidate artifact rather than adjusting either definition.

**Permitted fallback.** If the frozen revision finds the paired population
unstable at the corpus sizes chosen, it may instead preregister a
**candidate-only floor** on the revisit rate and drop the paired-increase
requirement in §§10.4 and 11. It may not keep the paired wording while
evaluating a candidate-only quantity.

**Terminal and ineligible exclusions.** Terminal and empty-priors children are
excluded from `explored_replies(L_arm)` in both numerator and denominator,
because a terminal reply can never be deepened and would bias a
depth-conversion metric downward regardless of the mechanism. They remain
**included** in `replies(L_arm)`,
which is the imported v17 breadth statistic and must not be redefined. The
excluded count is reported per row so the two denominators are reconcilable.

**Scope.** `replies(L_arm)`, `follow_up_visits_per_explored_reply` and
`revisit_to_depth3_rate` are leader-subtree scoped. The `depth>=3` count and
fraction are whole-tree, except inside §10.4.1 where conversion efficiency
restricts them to the leader subtree; that restriction is stated there and
applies nowhere else.

### 10.2 Identity prerequisite

Identity cap `2.0` must be byte-identical to shipped for:

- every scientific result field;
- full tree signature;
- selected move and visit counts;
- root value bits;
- callback sequence;
- RNG trace; and
- canonical `search_result_sha1`.

Only config/provenance labels may differ. Any mismatch rejects the
implementation before positive caps run. The two-phase v17 sequencing rule is
reused: complete and verify shipped/identity on all rows before starting any
positive search.

### 10.3 Development safety gates

For each positive cap versus shipped, reject if any:

- Target new-collapse rate `>=5%` (zero permitted on 16).
- Target p95 absolute mover-value delta `>=0.35`.
- Pooled representative-control lower-prior selected-move flip rate `>=10%`
  over the 16 representative controls (at most one permitted; reject at two).
  Identity witnesses are excluded from this pool: they cannot flip by
  construction, so including them would dilute the rate with structurally inert
  denominator.
- Any lower-prior selected-move flip on the 4 activation-positive flip
  controls.
- Any scientific-result difference on the 4 activation-negative identity
  witnesses.
- Control p95 absolute mover-value delta `>=0.35`, over the 20 controls capable
  of differing from shipped.
- Mean target root effective-children reduction `>=10%`.
- Mean target root top-share increase `>=0.05`.
- Aggregate target reply reduction `>= R_max`.
- More than one target changes to a move whose raw prior rank is worse by ten
  or more places. *(Moved here from §10.4 in revision 3 as a conservative
  preregistered stop rule — see §7. Not a monotonicity claim: §1.5 disclaims
  pointwise monotonicity for search-level collateral. The placement reflects
  that this is the v16 failure signature and that we have no basis for
  expecting a stronger cap to repair it.)*
- Any terminal-truth or exact-depth invariant is violated.
- Missing/nonfinite/incomplete rows, protocol mismatch, or scientific
  injection.

**The reply-reduction band replaces v17's flat `<10%` rejection.** v18's
benefit is transmitted *through* reduced depth-1 first-touch reply scanning;
treating that reduction as collateral would confuse the mechanism with its
failure mode. The distinction is:

- **Permitted:** fewer one-touch replies, with the released budget demonstrably
  becoming deeper confirmation.
- **Rejected:** fewer replies that merely concentrate the tree, raise top
  share, collapse roots, or fail to improve 6,400-simulation fidelity.

```text
R_min <= aggregate reply reduction < R_max      (R_min > 0)
```

**The two bounds are deliberately evaluated in different gates**, because they
mean different things and must route the §7 ladder differently:

| bound | gate | meaning | ladder consequence on failure |
|---|---|---|---|
| `< R_max` | §10.3 safety | did not lose too much breadth | **reject v18 immediately**; run no stronger cap |
| `>= R_min` | §10.4 mechanism | the mechanism actually acted | insufficient reach; **advance to the next stronger cap** |

Merging them into one gate would make an under-reaching weak cap trigger an
immediate family rejection, which is exactly the outcome the weakest-first
ladder exists to avoid.

**FREEZE BLOCKER:** both bounds are frozen from the shipped-only preflight,
budget accounting, and the §2.3 historical failure anchors — never from v18
candidate results. `R_max` must lie strictly below the smallest historically
unsafe reply reduction in §2.3's anchor table, whose exact figures and artifact
SHA-1s are authoritative; no rounded form of any anchor may be used here. Under
§2.1.1 the anchors may only lower `R_max`; they may not set `R_min` or any
other bar.

`R_min > 0` requires the mechanism to actually act. It does **not** protect
§10.4.1's efficiency denominator: `R_min` is pooled over all 16 targets, while
`lost_replies` is summed over the stable-leader subset only, and that subset can
show zero or negative loss even when the pooled reduction clears `R_min`. The
minimum-`lost_replies` requirement in §10.4.1 is the actual denominator guard.

**FREEZE BLOCKER:** include operating-characteristic and discreteness tables
for every count/rate gate before approval.

### 10.4 Development mechanism gates

A cap must satisfy all:

- At least one eligible depth-2 leaf is clipped on at least `8/16` targets.
- Aggregate 6,400-simulation absolute-value-error reduction is positive and at
  least **20%**:

  ```text
  1 - sum(candidate_reference_error) / sum(shipped_reference_error) >= 0.20
  ```

  A zero shipped-error denominator is invalid.
- At least `9/16` targets have strictly smaller reference-value error than
  shipped.
- Reference selected-move agreement is not worse than shipped by more than one
  target.
- Aggregate target reply reduction `>= R_min` (the lower bound only; the upper
  bound is a §10.3 safety rejection).
- Aggregate depth-`>=3` completed backups increase.
- At least `9/16` targets increase either the depth-`>=3` backup fraction or
  `revisit_to_depth3_rate` as defined in §10.1.1 — i.e. over the
  `would_clip(arm, cap)` population evaluated in both arms, not over actually
  clipped nodes.
- Mean follow-up completed visits per explored depth-2 reply is greater than
  shipped.
- Conversion efficiency (§10.4.1) clears its preregistered floor.

Root effective-children reduction and target prior-rank regression are **not**
repeated here: both are §10.3 safety rejections, so a failure terminates the
family rather than advancing the ladder. An *increase* in root breadth is
reported and is not a failure.

#### 10.4.1 Conversion efficiency on stable-leader targets

Conversion must be computed on comparable branches. A target whose root leader
changes between shipped and candidate has no comparable leader subtree, so it
cannot contribute conversion evidence:

```text
stable-leader targets =
    targets whose shipped and candidate root leader are identical

lost_replies =
    sum(shipped final-leader replies)
    - sum(candidate final-leader replies)
    over stable-leader targets

gained_deep_backups =
    sum(candidate depth>=3 completed backups)
    - sum(shipped depth>=3 completed backups)
    within those same leader subtrees

conversion_efficiency =
    gained_deep_backups / lost_replies
```

Requirements: a minimum stable-leader count, provisionally at least `12/16`;
`lost_replies` at or above a preregistered minimum count, so the ratio is not
computed on a one-or-two-reply denominator; positive `gained_deep_backups`; and
`conversion_efficiency` at or above its preregistered floor. Changed-leader
rows remain in every safety and fidelity gate but cannot create a conversion
pass.

**FREEZE BLOCKER:** the efficiency floor, the stable-leader minimum, and the
minimum `lost_replies` denominator are preregistered budget-conversion
requirements informed by §2.2.1, not candidate predictions recovered from a
shipped tree. The ratio is supporting evidence and
is never the sole gate — a ratio alone can look excellent while the search
narrows catastrophically, which is why it sits alongside the absolute reply
band, the breadth and concentration bounds, and the fidelity gates.

### 10.5 Selection and stop

Choose the largest cap passing §§10.3–10.4. Persist the whole preregistered
ladder with each point labeled `PASS`, `FAIL`, or `NOT_RUN_STOP`, plus null or
selected cap, gate records, development artifact SHA-1, and selection context.
If none pass, reject v18 and do not generate held-out evidence.

If one passes, stop for separate held-out authorization.

## 11. Stage 3 — fresh held-out confirmation

Run only the frozen development-selected cap.

Proposed held-out generation:

```text
games:        provisional 2,200; final count set by exact-selector sizing
all settings: identical to Stage 1
```

**FREEZE BLOCKER:** choose a new collision-audited seed range, selector seed,
and disjointness set before approval.

Provisional held-out geometry, using the same four frozen shipped-only role
rules:

```text
24  residual-exposed targets                    12/12
 6  activation-negative identity witnesses       3/3
 6  activation-positive flip controls            3/3
32  representative controls, eight per phase    16/16
68  total                                       34/34
```

**FREEZE BLOCKER:** exact-selector sizing must freeze exact per-role counts
before generation. The operator does not adapt allocation after observing
yield.

Retained: at most two positions per game with at least 12-ply separation;
whole-game split isolation; complete-state canonical uniqueness; complete game
and position disjointness from development and all historical evidence.

Run shipped and the frozen cap at 400 simulations on all 68 rows. Run the
shipped 6,400-simulation formal reference, and the 1,600-simulation trajectory
diagnostic, on the 24 targets.

Held-out rejection gates:

- Target new-collapse rate `>=5%`.
- Target p95 absolute mover-value delta `>=0.35`.
- Pooled representative-control lower-prior flip rate `>=10%` over the 32
  representative controls (at most three permitted; reject at four). Identity
  witnesses are excluded for the §10.3 reason.
- Any lower-prior flip on the 6 activation-positive flip controls.
- Any scientific-result difference on the 6 identity witnesses.
- Control p95 absolute mover-value delta `>=0.35`, over the 38 controls capable
  of differing from shipped.
- Mean target effective-children reduction `>=10%`.
- Mean target top-share increase `>=0.05`.
- Aggregate target reply reduction `>= R_max`.
- More than one target changes to a move whose raw prior rank is worse by ten
  or more places.
- Any missing, nonfinite, unbound, or incomplete evidence.

Mechanism transfer floor:

- aggregate target reply reduction `>= R_min`;
- aggregate 6,400-simulation reference-value-error reduction `>=10%`;
- at least `13/24` targets strictly improve reference error;
- reference move agreement not worse than shipped by more than one;
- at least `12/24` targets contain an actual depth-2 clip;
- aggregate depth-`>=3` completed backups increase;
- at least `13/24` targets increase depth-`>=3` backup share or
  `revisit_to_depth3_rate` over the §10.1.1 `would_clip` population;
- mean follow-up completed visits per explored depth-2 reply exceeds shipped;
  and
- conversion efficiency on stable-leader targets clears its floor, under the
  **complete** §10.4.1 guard set scaled to 24 targets: a stable-leader count of
  at least `18/24` (the same 75% as development's `12/16`), `lost_replies` at
  or above its preregistered minimum on that subset, and strictly positive
  `gained_deep_backups`. `R_min` is pooled over all targets and does not
  substitute for the `lost_replies` guard here either.

This is a transfer floor, not a repeated candidate-selection grid. Failure
rejects v18. Passing stops for separate A/B/C/D authorization.

## 12. Stage 4 — established A/B/C/D acceptance

Run shipped and the one frozen cap. No alternate cap runs.

Reuse v17's authenticated:

- exact A/B/C/D row sets and source SHA-1s;
- per-position seeds;
- selected-move comparator;
- shipped per-case values and selected moves; and
- exact baseline reproduction gate before candidate interpretation.

Expected shipped aggregates remain:

```text
A: n=30, mean +0.25702582687976244, over 15/30, severe 13/30
B: n=18, mean -0.24424638776811966, over 1/18, severe 0/18
C: n=30, mean +0.09857376916756039, over 10/30, severe 4/30
D: n=30, mean -0.18752245797826617, over 4/30, severe 0/30
```

### 12.1 A — historical mechanism and acceptance

Require:

- Mean black value `<=0.0`.
- Severe `<=5/30`.
- Progress toward frozen 6,400-sim `V_REF=-0.0451` `>=50%`.
- New-collapse count `<=2`.
- Mean root effective-children reduction `<10%`.
- Mean top-share increase `<0.05`.
- Aggregate reply reduction `< R_max`.
- Aggregate depth-`>=3` completed backups increase.
- Candidate selected-move agreement with the frozen 6,400-sim A move reference
  is not worse than shipped — **conditional on §2.2.2**. The historical 6,400
  A artifact records values and top shares but no selected move or visit
  counts, so this gate is evaluated only if the preregistered shipped-only
  capture is authorized and passes its exact-reproduction authentication. If
  it is not, this bullet is removed rather than evaluated against an
  unauthenticated or lower-budget substitute.

Do **not** require reply *suppression*, and do **not** apply `R_min` here.
Reply suppression was the rejected FPU mechanism, not the v18 objective. `R_min`
exists to prove the mechanism acted on rows *selected for residual exposure*;
A is a historical downstream probe whose rows were selected by an unrelated and
now-discredited statistic, so a low reply reduction on A is not evidence of
under-reach. Only the bounded-loss upper bound applies.

A remains a downstream historical acceptance/mechanism probe. It does not
select the cap.

### 12.2 B — goal-line

- Severe `=0/18`.
- Over `<=2/18`.
- No terminal value is ever clipped.

### 12.3 C — old broad post-opening

- Severe `<=4/30`.
- Over `<=10/30`.
- Mean `<=+0.099`.

### 12.4 D — red pre-drop

- Severe `=0/30`.
- Mean `<=0.0`.

All four must pass. Stop for separate strength authorization.

## 13. Stage 5 — primary same-checkpoint strength

Reuse the proven asymmetric same-checkpoint runner:

```text
candidate: calib020_0001 + frozen v18 cap
control:   calib020_0001 + shipped search
games:     800
colors:    exactly 400/400
budget:    400 simulations per move for both
selection: opening_temperature; 20 opening plies; 1.0 / 0.1
move cap:  280; timeout is a draw
root noise: add_noise=false
```

**FREEZE BLOCKER:** choose a collision-audited seed range.

All non-v18 fields, evaluator bytes, tasks, colors, move caps, and termination
rules are identical. Both agents use `(14, 48, 8)` and the synchronous search
path.

Pass only if the candidate draw-aware trinomial score interval has lower bound
`>0.5`. Report by-color score, draws, state caps, and Elo. No sequential
stopping, extension, or top-up.

## 14. Stage 6 — contemporaneous validation against `0379`

Run two separate, paired-seed, 800-game balanced-color matches:

```text
candidate v18 vs 0379
shipped search vs 0379
```

**FREEZE BLOCKER:** choose one shared collision-audited seed range.

Reuse v17's harness-validity rules:

- shipped protocol/checkpoint/settings identity matches the authenticated
  production anchor except for fresh seeds and the explicit v18-capable
  runner;
- shipped CI95 overlaps `0.588875`;
- shipped point estimate is within `0.05` of `0.588875`;
- shipped CI95 lower bound exceeds `0.5`.

Candidate requirements:

- CI95 lower bound versus `0379` exceeds `0.5`; and
- the deterministic paired two-game-block bootstrap, 100,000 replicates under
  a newly frozen bootstrap seed, gives a 95% lower bound for
  `(candidate score rate - shipped score rate) > -0.02`.

Historical `~+80 Elo` remains context, not pooled evidence.

## 15. Provenance and artifacts

Every v18 scientific artifact records:

- mechanism/formula ID and cap;
- whether identity or positive mode was active;
- exact depth scope and terminal exemption;
- `search_execution_mode=synchronous`;
- complete MCTS configs;
- raw and resolved value telemetry;
- per-cap actual clip counts per row, for role authentication (§9.2.3);
- checkpoint, manifest, source-index, selected-replay, config, protocol, and
  source-file SHA-1s;
- complete disjointness;
- per-position seeds and `add_noise=false`;
- match ranges and color allocation;
- git commit and clean-worktree state;
- runtime/MLX versions; and
- run kind.

Scientific kinds require a clean worktree and exact source identities before
evaluator load. Tooling smoke is always interpretation-forbidden.

Protocols bind the complete corpus triple for development and held-out. Later
stages bind the selected-cap rejection/authorization record and the development
artifact from which it was derived.

Diagnostic outputs are canonical and timestamp-free. Repeated builds and reruns
with identical inputs are byte-identical. Match operational timestamps are
excluded from scientific identity while the full JSONL remains bound.

No v16/v17 output is overwritten. Output root:

```text
logs/eval/v18_depth2_provisional_backup/
```

## 16. Stage and approval boundaries

Each arrow is a separate stop and authorization boundary:

```text
draft review
  -> shipped-only residual preflight
  -> final design/grid/threshold freeze
  -> implementation + software review
  -> tooling smoke
  -> fresh development reservoir/corpus
  -> development sweep
  -> fresh held-out reservoir/corpus
  -> held-out confirmation
  -> A/B/C/D
  -> same-checkpoint strength
  -> contemporaneous 0379 validation
  -> adoption decision
```

No stage authorizes the next.

All source edits and commits must precede generation protocol emission.
Generation, qualification, screen, and selection run at one unchanged HEAD. No
source edit or commit occurs inside that chain (§5.3).

## 17. Stop rules and non-goals

Stop at the first failed prerequisite.

Do not:

- alter FPU or policy mass;
- change `c_puct`;
- train or modify checkpoint weights;
- use A as the cap-selection corpus;
- clip terminal values;
- broaden the scope beyond relative depth exactly 2;
- implement the batched waiter path under v18;
- tune on v16a or v17 consumed evidence;
- derive any numeric threshold from the selected-A set (§2.2.3);
- relax collapse, control-flip, breadth, or concentration gates;
- widen `R_max` or lower `R_min` after seeing candidate results;
- accept improvement produced by FPU-scale breadth loss;
- promote the 1,600-simulation trajectory diagnostic to a gate;
- add/interpolate caps after development;
- run a stronger cap after a weaker cap fails safety;
- continue the cap ladder after the largest/weakest passing cap is selected;
- inspect held-out before cap selection is frozen;
- extend match seeds;
- treat reservoir matchup score as candidate evidence; or
- adopt into self-play before every gate passes.

If v18 rejects, keep `calib020_0001` and shipped search. A null closes this
depth-2 provisional-backup formulation; it does not authorize depth>=2,
one-sided clipping, entropy-adaptive caps, parent searched-Q baselines, or a
smaller-cap rescue.

## 18. Draft approval checklist

Resolved in revision 2:

- [x] Code surface audited; single synchronous call site with batched refusal.
- [x] Breadth constraint reframed as bounded loss plus proven conversion.
- [x] Control roles split into identity witnesses and exposed flip controls.
- [x] Reuse strategy fixed as a thin v18 layer with no extraction refactor.
- [x] Formal fidelity reference fixed at 6,400 simulations.
- [x] Telemetry derived from the final tree by a read-only walker.
- [x] Branch and ancestry consequence recorded.

Resolved in revision 3:

- [x] Cap-ladder routing defined for every §10.3/§10.4 outcome; prior-rank
      regression moved to §10.3.
- [x] Efficiency denominator guard corrected and carried into held-out.
- [x] Exact formulas fixed for every new conversion metric (§10.1.1).
- [x] Historical reply-reduction anchors traced to source artifacts, including
      v16a at `0.28027286567513765` (§2.3).
- [x] Evaluator-call, synchronous-prerequisite, and `tanh`-range wording
      corrected.

Approved 2026-07-29, **for the shipped-only preflight step only**:

- [x] Scientific hypothesis and depth-exactly-2 scope.
- [x] Raw parent NN value, with sign conversion, as the provisional baseline.
- [x] Terminal/depth-1/depth-3 exemptions and the §3.2 degenerate-input rules.
- [x] The §2.1.1 narrow exception permitting historically unsafe operating
      points to impose a ceiling on `R_max` only.
- [x] The shipped-only 6,400-simulation A move capture (§2.2.2), subject to its
      dual-statistic authentication and determinism check.
- [x] The terminal/ineligible exclusion decisions in §10.1.1.

These approvals authorize preflight planning. They do not authorize an
`mcts.py` edit, a scientific run, or a commit.

Unresolved and blocking freeze:

- [ ] **Freeze the exact positive-exposure formula, its threshold, and the
      sign-dominance requirement** for target selection (§9.2.1).
- [ ] Decide the §10.1.1 `revisit_to_depth3_rate` form: paired `would_clip`
      populations, or the declared candidate-only floor.
- [ ] Freeze numeric shipped-only preflight PASS/FAIL criteria **before** the
      preflight runs.
- [ ] Run and review the shipped-only preflight, including the §2.2.1
      first-order crossover analysis.
- [ ] Demonstrate that residual exposure separates selected-A-like shallow
      reversal from matched non-A discovery positions without using candidate
      or higher-budget outcomes as runtime inputs.
- [ ] Freeze `R_min`, `R_max`, and the conversion-efficiency floor from
      non-A discovery controls and budget logic.
- [ ] Freeze the stable-leader minimum and the minimum `lost_replies`
      denominator for conversion efficiency.
- [ ] Freeze the material-exposure floor for activation-positive flip controls.
- [ ] Approve weakest-first cap execution and immediate family rejection if
      the weakest positive cap fails safety.
- [ ] Replace the provisional cap grid with the final preflight-supported grid.
- [ ] Approve cap `2.0` as structural identity and largest-passing selection.
- [ ] Freeze exact development and held-out per-role counts and side geometry
      from exact-selector sizing.
- [ ] Freeze the 1,600-simulation trajectory pass, or drop it before freeze.
- [ ] Add count-gate operating-characteristic and discreteness tables.
- [ ] Collision-audit and freeze all generation, selector, diagnostic, match,
      and bootstrap seeds.
- [ ] Confirm the narrow-hook reuse plan and the duplicate-definition audit.
- [ ] Publish smoke-measured low/central/high compute estimates and durable
      execution semantics.
- [ ] Confirm complete source-file scope before implementation.
- [ ] Approve 800-game same-checkpoint strength and paired `0379` validation.
- [ ] Acknowledge expected low success probability and no rescue after null.

Until every applicable box is resolved in a reviewed frozen revision, this
document is a research draft only.
