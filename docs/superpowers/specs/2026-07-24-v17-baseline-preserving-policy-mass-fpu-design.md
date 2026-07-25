# v17 Baseline-Preserving Policy-Mass FPU — Preregistered Design

**Status:** APPROVED — FROZEN (2026-07-24), by explicit user approval after
zero-GPU preflight and review. This protocol's formula, grid, samples, gates,
stop rules, batching contract, and evidence boundary are now immutable;
observed results may not change them.

Freezing authorizes no work by itself. v17 source implementation, any
positive-coefficient result, the fresh development and held-out reservoirs,
the A/B/C/D probes, the strength matches, self-play adoption, and any commit
or push each require separate explicit authorization.

Freeze provenance:
`logs/eval/fpu_v17_baseline_policy_mass/frozen_preregistration.json`.

## 0. Decision and objective

The parent-relative v16 family is retired. Its neutral point,
`FPU = Q_parent`, changed the selected move to a lower-prior move on
`11/40 = 27.5%` of the tuning controls and failed the frozen `<10%` gate.
No nonzero v16 coefficient ran. The rejection is final.

v17 tests one new search mechanism:

```text
P_explored = Σ prior(a) over children with completed backed-up visits
FPU_v17    = shipped_FPU - r * sqrt(clamp(P_explored, 0, 1))
```

For the production baseline in this experiment, `shipped_FPU = fpu_value = 0`.
The mechanism is baseline-preserving: `r=0` must execute the shipped selection
branch exactly. `Q_parent` does not participate in v17 FPU.

The objective is unchanged `calib020_0001` playing stronger at the same
400-simulation budget while preserving collateral safety. Success requires:

1. All formal A/B/C/D gates, surpassing v14b's near-pass.
2. A statistically significant same-checkpoint win over shipped FPU.
3. Non-regression against `0379`, measured contemporaneously.

No self-play adoption is allowed until all three layers pass.

## 1. Evidence boundary

### 1.1 Evidence used to design v17

- Absolute `fpu_value=-0.20` reached the selected-A reply-scanning mechanism
  but failed fresh late collateral.
- v16 `r0` failed because replacing shipped FPU with `Q_parent` was itself
  behavior-changing and moved flipped controls from mean prior rank `1.27` to
  `9.00`.
- The v16 postmortem found the 11 flips concentrated in opening (`5/10`) and
  red-to-move (`8/20`) controls, with larger effective-children loss but no
  flip-specific increase in top share.

These findings motivate removing `Q_parent`; they do not estimate a v17
coefficient.

### 1.2 Consumed and forbidden scientific evidence

The following are immutable, consumed evidence and may not select or rescue a
v17 coefficient:

- The v16 production reservoir, screen, 120-row manifest, tuning split, and
  frozen split.
- The 40 v16 tuning controls and their 11-flip postmortem.
- The v16 nonzero grid, which was never run and remains retired with its
  parent-relative formula.
- The v16a 324-position neutral corpus as a coefficient-development sample.

Existing smoke replays may be used only for a labeled `tooling_smoke` that
cannot emit a scientific pass, select a coefficient, or justify escalation.
The established A/B/C/D manifests remain fixed acceptance benchmarks and run
only after a v17 coefficient has been frozen on fresh evidence.

Two read-only design preflights are explicitly permitted before freeze:

- Exact-selector feasibility/sizing may use the authenticated immutable v16
  production screen. It sizes fresh reservoirs only; it cannot select a v17
  position or coefficient and does not certify a future fresh reservoir.
- A shipped-FPU-only pass may measure `P_explored` at existing selected-A FPU
  application sites and at deterministic late flat-policy rows selected from
  the authenticated historical screen. It may establish formula/grid reach
  and root-safety exposure only; it cannot evaluate a positive v17
  coefficient, select a v17 position, or count as A/collateral acceptance
  evidence.

### 1.3 Completed preflight evidence

The shipped-only selected-A reach pass covered all 30 established A positions,
400 simulations, and `add_noise=false`. At the final root leader's reply node,
after its first completed reply and while unvisited replies remained,
`P_explored` was materially larger than the flat-root approximation:

```text
min 0.1299; p25 0.2069; median 0.2686; p75 0.3321; max 0.8147
```

The coefficient required to reach an effective `-0.20` at that application
site had median `0.3859`, p75 `0.4397`, and maximum `0.5549`. Therefore the old
upper bound `0.25` was underpowered at the median application site, but the
formula itself is not inherently null on selected A. The deterministic
preflight artifact is
`logs/eval/fpu_v17_baseline_policy_mass/preflight/selected_a_shipped_application_mass.json`,
SHA-1 `8854e02f33f210ed381e0cd44e8e1de0fb7f4b7b`. It is labeled
shipped-baseline mechanism reach, not a v17 result.

The complementary root-safety pass used the 24 deterministic late
flat-policy targets from the authenticated historical screen's held-out exact
witness, with the original shipped 400-simulation seeds and
`add_noise=false`. Final root explored mass was highly heterogeneous:

```text
min 0.0078; p25 0.0414; median 0.2376; p75 0.8644; p90 0.9289; max 1.0000
```

At final shipped mass, `r=0.45` would imply a median effective floor
`-0.2193`, p90 `-0.4337`, and maximum `-0.4500`; `r=0.55` would imply
`-0.2681`, `-0.5301`, and `-0.5500`. Only one of the 24 shipped roots was
already collapsed, so high mass cannot be dismissed as an artifact of roots
already excluded by the new-collapse definition. This establishes substantial
late-root safety exposure at the upper grid, without predicting the
candidate's visits or collapse outcome. The byte-deterministic artifact is
`logs/eval/fpu_v17_baseline_policy_mass/preflight/late_flat_root_shipped_mass.json`,
SHA-1 `9a18c7ff90ef27e4afa91b8bbf32d676ee655fad`.

Authenticated exact-selector sizing used 299 deterministic whole-game
resampling trials per game count, analysis seed `20260724`, selector seed
`20260725`, and the production screen only. The preregistered rule is all
`299/299` successes, whose exact one-sided 95% lower bound is
`0.9900308532 >= 0.99`.

- Development: 800 games passed only `191/299` (lower bound `0.5905`);
  1,400 was the first tested count passing `299/299`, and 1,600–2,000 also
  passed `299/299`.
- Held-out: 1,200 games passed only `211/299` (lower bound `0.6593`);
  2,000 was the first tested count passing `299/299`, and 2,200–2,400 also
  passed `299/299`.

The profile SHA-1s are
`2357f107a45debb4ee6b818016ef9f9806ee185d` (development) and
`fd9950ec886f99e00754d558e828ef749602cb01` (held-out). The deterministic
sizing artifact SHA-1s are
`b41089e2d5c34a551e6b6b9ebe95299afc8cf6bc` and
`35c0f2eb37d42492e768ee365a1d460c576c35c1`, respectively.
The preflight profiles use the existing tool's `tooling_smoke` run-kind solely
to mark their outputs non-scientific. Their allocation geometry and selector
seed are frozen here; the future v17 protocol must emit its proper
`development` or `held_out` run-kind rather than reuse that label.

v17 deliberately follows the v16 next-tier-up convention rather than choosing
the first passing count from correlated subsets of one historical reservoir.
The operational sizes are therefore 1,600 development games and 2,200
held-out games: one tested 200-game tier above the first `299/299` success.
This modest margin addresses lumpy per-game yield (`1,361/4,000` historical
games produced no kept row) and the sizing artifact's explicit warning that it
does not independently certify a fresh reservoir.

## 2. Search rule and implementation contract

### 2.1 New field; retired field remains distinct

Add a new optional MCTS field:

```python
fpu_shipped_policy_mass_reduction: float | None = None
```

Do not rename, reinterpret, or reuse the retired
`fpu_policy_mass_reduction` field. Its meaning remains
`Q_parent - r*sqrt(P_explored)` for artifact reproducibility.

Configuration rules:

- `None`: execute the existing shipped `fpu_value` branch.
- `0.0`: execute the exact same shipped branch as `None`.
- `>0`: use `fpu_value - r*sqrt(P_explored)`.
- Reject nonfinite or negative values.
- Reject simultaneous non-`None` use of the retired parent-relative field.
- Structurally reject a non-`None` new field unless `fpu_value == 0.0`.

### 2.2 Exact-zero identity is structural

The selection site must branch before calculating explored mass:

```python
r = config.fpu_shipped_policy_mass_reduction
if r is None or r == 0.0:
    q = config.fpu_value
else:
    q = policy_mass_fpu(
        config.fpu_value, explored_policy_mass(node), r)
```

This is stronger than numerical equivalence. At `r=0`, v17 must not call the
existing formula helper, scan children for explored mass, alter RNG consumption,
change tie behavior, or add observer mutations.

Required identity proofs:

- The full existing suite remains green.
- Synthetic-tree selected moves and selection traces match.
- Fixed CPU-search visit counts, root value, tree signature, and callback
  sequence match.
- Replay tooling-smoke scientific-result payloads for `None` and `0.0` are
  byte-identical and have the same canonical `search_result_sha1`.
  Required config/provenance labels remain distinct and are excluded from that
  result hash.

Any zero-identity failure is a tooling rejection. Do not run a positive
coefficient until it is fixed and reviewed.

### 2.3 Explored mass semantics

`P_explored` reuses the existing `explored_policy_mass(node)` completed-visit
definition:

- Count prior mass only when `child.visit_count > 0` after backup.
- Exclude pending and virtual visits.
- Use raw root priors with `add_noise=false` in diagnostics.
- Clamp only inside the existing pure `policy_mass_fpu` helper.

Do not add a duplicate formula helper. The existing
`policy_mass_fpu(parent_q, explored_mass, r)` already has the required finite
checks, clamp, and subtraction; v17 passes `config.fpu_value` as its first
argument. Because validation requires `fpu_value == 0.0` whenever the new
field is active, the positive v17 formula is operationally
`-r*sqrt(P_explored)`. “Baseline-preserving” refers to the structural `r=0`
identity, not to a supported nonzero base.

At full explored mass the maximum reduction is `r`. At zero explored mass the
FPU is shipped FPU.

### 2.4 Frozen batching contract

`P_explored` depends on completed backed-up visits, so batching and pending
leaf behavior are part of the v17 mechanism rather than performance-only
settings. Every shipped, zero, and positive v17 search in every stage must use:

```text
eval_batch_size:        14
stall_flush_sims:       48
pending_virtual_visits: 8
```

These values apply to diagnostics, tooling smoke, development, held-out,
A/B/C/D, same-checkpoint strength, and both `0379` validation agents. The
standalone `MCTSConfig.stall_flush_sims` default of `16` is not a permitted
v17 effective value; v17 must explicitly derive `48`. Free CLI overrides are
rejected. Before evaluator load, the protocol must assert equality of these
three fields across the diagnostic base config and each match-runner agent
config. Any mismatch invalidates the stage rather than creating a comparable
result at a different effective mechanism strength.

## 3. Mechanistic predictions and failure risks

Predictions:

- `r=0` produces zero move flips and byte-identical scientific search results;
  only the required config/provenance label differs.
- A positive `r` increasingly discourages first-touch reply scanning only
  after meaningful policy mass has been explored.
- The candidate should reduce opponent replies on fresh flat-policy targets
  and selected A without the immediate `Q_parent` move-order disruption.

Primary risks:

- Recreating the v16a late-collapse failure as explored mass approaches one.
- Excessive effective-children reduction without real strength improvement.
- Low-prior early lock-in after the penalty accumulates.
- Color-, phase-, or branching-specific collateral.
- Passing selected A by narrowing search rather than improving play.

The gates below reject these outcomes; A improvement alone is never enough.

## 4. Frozen coefficient grid

The amended v17 grid is:

```text
r ∈ {0.15, 0.20, 0.25, 0.35, 0.45}
```

Rationale:

- `0.15` is a low/safety probe.
- `0.20` is the direct absolute-`-0.20` analogue: because
  `P_explored <= 1`, its effective floor cannot be more negative than
  `-0.20`.
- `0.25` resolves the transition where 10/24 historical late roots would have
  at least `-0.20` final-floor magnitude under the shipped mass trajectory.
- `0.35` reaches approximately the selected-A p25 application site.
- `0.45` reaches approximately the selected-A p75 application site, but is
  explicitly the high-risk root-safety boundary: at shipped late-root mass its
  final p90 effective floor is about `-0.434`.

`0.55` is excluded before freeze. Its additional selected-A reach is not worth
the late-root exposure of a roughly `-0.530` p90 and `-0.550` maximum final
floor. The grid does not claim that `0.45` is likely to pass; it preserves one
upper point capable of testing whether the mechanism/safety window exists.

Selection rule: choose the smallest coefficient passing every fresh
development safety and mechanism gate. If none pass, reject v17. Do not
interpolate, extend the grid, rerun with a new grid, or consult later stages.

## 5. Stage 0 — TDD and tooling smoke

### 5.1 Unit/integration gates

Before positive replay evidence:

- Reuse tests for the existing pure formula helper; add v17 call-site tests.
- `None` and `0.0` exact shipped-branch tests.
- Completed-visit-only explored-mass tests.
- Mutual-exclusion and config-validation tests.
- Import-purity tests: no MLX on pure diagnostic import.
- Deterministic canonical JSON/CSV tests.
- Provenance tests covering the new module and MCTS source hashes.
- Same-checkpoint asymmetric-runner tests proving default-path identity,
  per-agent config separation, and color/config symmetry.
- Full repository suite.

### 5.2 Small full-chain tooling smoke

Before the development reservoir, exercise every real CLI stage on a separate
small batch:

```text
games:        32
board:        24
MCTS:         400 simulations
workers:      4
seed range:   [20309000, 20309032)
replays:      enabled
selection:    opening_temperature
opening temperature plies: 20
temperature:  1.0 early / 0.1 late
max moves:    280; timeout is a draw
root noise:   add_noise=false
run kind:     tooling_smoke
```

Run protocol creation/check, generation, qualification and qualification
recheck, raw-policy/anchor screening, post-screen qualification, and the
actual selector CLI. The smoke selector profile requests exactly two controls:
one opening and one early-mid, with no side constraint. It exists only to prove
that the real chain, arguments, output capture, progress reporting,
fingerprints, conflict/refusal behavior, and exit codes work end to end.

Run a deliberate controlled failure against a tampered copied config or
impossible smoke allocation and confirm nonzero exit with no scientific
artifact. Rerun the successful chain and confirm idempotent artifact hashes.
All outputs must stamp `scientific_interpretation_forbidden=true`; target yield,
match score, and selected positions have no scientific meaning. Failure blocks
Stage 1.

### 5.3 Replay tooling smoke

After §5.2 succeeds, use exactly its two selected controls as the replay-smoke
input; there is no dependency on the untracked `smoke_v1` directory. Run:

```text
shipped/None, v17 r=0.0, v17 r=0.35
```

The smoke may establish only state reconstruction, exact shipped/zero
scientific-result identity, positive-mode plumbing, schemas, hashes, seeds,
`add_noise=false`, and deterministic rerun bytes. It must stamp
`run_kind=tooling_smoke` and `scientific_interpretation_forbidden=true`.

### 5.4 Same-checkpoint runner tooling smoke

After the runner is implemented and the full software review passes, run eight
same-checkpoint tooling games with `calib020_0001` on both sides:

```text
shipped vs shipped: 4 games, seeds [20309100, 20309104)
shipped vs r=0.35:   4 games, seeds [20309104, 20309108)
colors/configs:      exactly 2/2 in each four-game block
budget:              400 simulations per move for both agents
root noise:          add_noise=false
run kind:            tooling_smoke
```

The shipped-vs-shipped block proves identical effective configs, deterministic
task construction, color/config swapping, and complete provenance; it has no
empirical 50% score expectation. The asymmetric block proves that distinct
per-agent MCTS configs reach the intended sides without leakage. Both blocks
must stamp `scientific_interpretation_forbidden=true`; their scores and game
outcomes have no scientific meaning.

## 6. Stage 1 — fresh development reservoir and corpus

This stage is separately operator-authorized only after Stage 0 passes.

### 6.1 Generation protocol

```text
checkpoint A: calib020_0001
checkpoint B: 0379
games:        1,600
board:        24
MCTS:         400 simulations
workers:      4
seed range:   [20310000, 20311600)
replays:      enabled
top-up:       forbidden
selection:    opening_temperature
opening temperature plies: 20
temperature:  1.0 early / 0.1 late
max moves:    280; timeout is a draw
root noise:   add_noise=false
```

The matchup score is generation metadata, not candidate-strength evidence.

### 6.2 Deterministic 32-position development corpus

Selection uses shipped FPU and raw policy only, before any positive v17 result:

- 16 late flat-policy targets:
  `normalized_entropy >= 0.90`, `top1_prior <= 0.025`,
  `abs(shipped root_value_stm) <= 0.25`.
- 16 concentrated-policy controls: four each from opening, early-mid,
  midgame, and late, with `normalized_entropy < 0.85` or
  `top1_prior >= 0.05`, and `abs(shipped root_value_stm) <= 0.25`.
- Exactly 16 red / 16 black overall. Report target/control side counts; no
  unimplemented per-role side-balance claim is made.
- At most two selected positions total per game, separated by at least 12
  plies.
- Complete-state canonical SHA-1 uniqueness.
- Zero overlap with all v16 production, v16a neutral, and A/B/C/D positions.
- Selector seed `20260725`; allocation geometry is the authenticated
  development preflight profile from §1.3.
- Branching bands are reported; no hard per-band minimum is imposed on this
  deliberately small discovery sample.

The role predicates are the existing `raw_policy_role` and `anchor_eligible`
definitions in `build_fpu_dev_corpus.py`; the spec does not restate a parallel
implementation. The cap, spacing, and 1,600-game size are preregistered from
the exact-selector preflight in §1.3, not capacity counts or v17 outcomes.
That analysis estimates subsets of the historical discovery reservoir and
does not independently certify this fresh run.

If the exact selector cannot fill this corpus, Stage 1 gate-fails. Do not top
up or relax the geometry.

## 7. Stage 2 — fresh development sweep

Run shipped, `r=0`, and the five positive coefficients on all 32 fresh rows
using identical per-position seeds and `add_noise=false`.

### 7.0 Frozen metric semantics

Reuse these existing diagnostic definitions exactly:

- Selected move / visit leader: greatest completed visit count, then the
  existing canonical move comparator for ties.
- Raw prior rank:
  `1 + count(other raw priors strictly greater than selected prior)`; exact
  prior ties do not worsen rank.
- Top share: visit leader's completed visits divided by completed root visits.
- Collapse: top share `>=0.95`; new collapse means candidate collapsed and
  shipped did not.
- Effective children: `exp(entropy(completed child visit counts))`; per-row
  reduction is `(shipped - candidate) / shipped`, or `0` when shipped is zero.
- Replies: number of completed-visit children below the root visit leader.
- Aggregate reply reduction:
  `1 - sum(candidate replies) / sum(shipped replies)`; a zero shipped
  denominator is invalid rather than automatically passing.
- Mover-value delta: candidate `root_value_stm - shipped root_value_stm`.
- p95: the existing sorted linear-interpolation percentile at rank
  `0.95 * (n - 1)`.
- Lower-prior control flip: selected move changed and the candidate move's raw
  prior is strictly less than the shipped move's raw prior.
- Early low-prior lock-in requires all five existing conditions:
  selected prior rank `>10`, selected prior `<0.01`, explored mass at final
  leader stabilization `<0.25`, stabilization simulation `<=100`, and final
  root top share `>=0.90`.
- A target's prior rank worsens by ten or more iff its selected move changes
  and `candidate_rank - shipped_rank >=10`.

Means of effective-children reductions and top-share increases are arithmetic
means of the paired per-position values. Missing, nonfinite, or
zero-denominator aggregate metrics fail the applicable gate.

### 7.1 Identity prerequisite

`r=0` must be byte-identical to shipped for every persisted scientific-result
field and the full per-position tree signature. The required config/provenance
label is the sole permitted difference and is excluded from the canonical
`search_result_sha1`. Any other mismatch rejects the implementation before
positive results are interpreted.

### 7.2 Development safety gates

For each positive coefficient versus shipped, reject if any:

- Target new-collapse rate `>=5%` (on 16 rows this requires zero new collapse).
- Target early low-prior lock-in count exceeds shipped by more than one.
- Target p95 absolute mover-value delta `>=0.35`.
- Mean effective-children reduction `>=50%` **and** mean top-share increase
  `>=0.15`.
- Control selected-move flip to lower prior rate `>=10%`.
- Control p95 absolute mover-value delta `>=0.35`.
- Any nonfinite metric, missing row, identity mismatch, or incomplete search.

Report, but do not add post-hoc gates for, phase, side, branching band, game,
collapse resolution, stabilization simulation, and explored mass.

The discreteness and operating characteristics are explicit. Under a binomial
model for 16 targets, the `>=5%` rule passes only with zero new collapses. Its
pass probability is `61.43%` at true rate `3%`, `18.53%` at `10%`, and `6.78%` at
`15.48%` (the measured v16a late rate). Thus it is a deliberately harsh
development screen, with `38.57%` false rejection at a benign `3%` rate.
The 16-control `<10%` flip gate permits at most one lower-prior flip and
rejects at two.

### 7.3 Development mechanism gates

A coefficient must satisfy all:

- Mean target reply-count reduction versus shipped `>=50%`, aligned with the
  downstream A reply-reduction gate.
- At least `8/16` targets have strictly fewer replies.
- Mean target effective-children reduction is positive but `<50%`.
- No more than one target changes to a move whose prior rank is worse by
  ten or more places.

Choose the smallest coefficient passing §§7.2–7.3. Persist the selected
coefficient and complete selection-context fingerprint. If none pass, stop.
If one passes, present the complete development result and stop for separate
held-out authorization.

## 8. Stage 3 — fresh held-out collateral

Run only the frozen Stage-2 coefficient. Generate no held-out games and inspect
no held-out artifact before coefficient selection is finalized.

### 8.1 Held-out generation

```text
games:        2,200
seed range:   [20312000, 20314200)
all other generation settings: identical to Stage 1
```

Select 56 positions by the same shipped-only rules:

- 24 late flat-policy targets.
- 32 controls, eight per phase.
- Exactly 28 red / 28 black overall. Report target/control side counts.
- At most two selected positions total per game, separated by at least 12
  plies.
- Complete game and position disjointness from Stage 1 and all forbidden
  historical corpora.

The exact-selector preflight in §1.3—not a capacity bound—made 2,000 the first
tested game count meeting the 299/299 reliability rule; the operational
held-out size is the next tested passing tier, 2,200.

### 8.2 Held-out gates

Relative to shipped FPU, reject if any:

- Target new-collapse rate `>=5%`.
- Target lock-in count exceeds shipped by more than two.
- Target p95 absolute mover-value delta `>=0.35`.
- Mean effective-children reduction `>=50%` **and** mean top-share increase
  `>=0.15`.
- Control lower-prior flip rate `>=10%`.
- Control p95 absolute mover-value delta `>=0.35`.

Mechanism confirmation requires mean target reply reduction `>=20%`; failure
rejects v17 even if collateral is otherwise safe.

This is deliberately a transfer floor, not a repeat of Stage 2's `>=50%`
selection gate. A held-out result from `20%` through less than `50%` is
classified as attenuated-but-present and may advance because Stage 3 is
collateral-primary, but it does not substitute for A's independent `>=50%`
mechanism confirmation.

Branching-band rates and census are reported only; 24 targets cannot support
the former `n>=20` per-band inference honestly. Under a binomial model for 24
targets, the `>=5%` collapse gate permits at most one new collapse and rejects
at two. Its pass probability is `83.88%` at true rate `3%`, `29.25%` at `10%`, and
`9.53%` at `15.48%`. The 32-control `<10%` flip gate permits at most three
lower-prior flips and rejects at four.

## 9. Stage 4 — established A/B/C/D acceptance

Run shipped FPU and the frozen candidate once each on the fixed established
probes. No other positive coefficient runs. These probes do not select or tune
the coefficient.

Before interpreting the candidate, a contemporaneous shipped-FPU run must
reproduce the frozen baseline artifact for every case within absolute
`1e-6`, with exact selected moves, over/severe counts, and case cardinality.
The expected aggregate baselines are:

```text
A: n=30, mean +0.25702582687976244, over 15/30, severe 13/30
B: n=18, mean -0.24424638776811966, over 1/18, severe 0/18
C: n=30, mean +0.09857376916756039, over 10/30, severe 4/30
D: n=30, mean -0.18752245797826617, over 4/30, severe 0/30
```

The canonical source artifacts are:

```text
A: logs/eval/calib020_0001_black_loss_post_opening_predrop_probe/position_probe_cases.csv
B: logs/eval/black_predrop_calib010_goal_line/goal_line_trigger_probe_cases.csv
C: logs/eval/calib020_post_opening_sweep/position_probe_cases.csv
D: logs/eval/calib020_0001_red_loss_post_opening_predrop_probe/position_probe_cases.csv
```

In each source, the frozen shipped row set is `checkpoint == "0001"`; other
checkpoint rows in a multi-checkpoint CSV are excluded.

For A and D, `over` and `severe` use the same black-perspective thresholds,
`probe_black_root_value >= +0.25` and `>= +0.50`, respectively. Recomputing
any gate mean in a different summation order may differ by one or two ULPs.
In the canonical sources, C may yield `+0.09857376916756037` instead of the
displayed `+0.09857376916756039`, and D may yield
`-0.18752245797826619` instead of the displayed
`-0.18752245797826617`. These differences are well inside the frozen `1e-6`
reproduction tolerance and are not drift.

The exact frozen per-case artifacts and SHA-1s are authenticated into the
protocol before any MCTS source edit. A mismatch makes Stage 4 invalid; it
does not rebase the gates.

### A — black pre-drop

Require all:

- Mean black value `<=0.0`.
- Severe rate materially below the shipped `13/30`; operationally `<=5/30`,
  matching or improving v14b's `5/30`.
- Progress toward the 6400-sim reference `>=50%`.
- Opponent reply reduction versus shipped `>=50%`.
- New-collapse count `<=2`.
- Mean top-share increase `<=0.15`.

Progress uses the existing frozen `V_REF=-0.0451` formula
`(mean_shipped - mean_candidate) / (mean_shipped - V_REF)`. Reply reduction
uses the aggregate definition in §7.0.

### B — goal-line

- Severe `=0/18`.
- Over `<=2/18`.

### C — old broad post-opening

- Severe `<=4/30`.
- Over `<=10/30`.
- Mean `<=+0.099`.

### D — red pre-drop

- Severe `=0/30`.
- Mean `<=0.0`.

All four must pass. A result merely closer to v14b is not success.

## 10. Stage 5 — primary same-checkpoint strength match

This is the causal strength endpoint:

```text
candidate: calib020_0001 + frozen v17 coefficient
control:   calib020_0001 + shipped FPU
games:     800
colors:    exactly 400/400
seeds:     [20320000, 20320800)
budget:    400 simulations per move for both
selection: opening_temperature; 20 opening plies; temperatures 1.0 / 0.1
move cap:  280; timeout is a draw
root noise: add_noise=false for both agents
```

All non-FPU settings, evaluator weights, task construction, selection mode,
move cap, and termination rules must be identical. The runner must distinguish
the two agents by full search-config identity even though their checkpoint
bytes are identical. For zero-based game index `g`, seed is `20320000 + g`;
the candidate is red for even `g` and black for odd `g`.

Pass only if the candidate's existing draw-aware trinomial interval
`score_ci_trinomial(w, d, l, z=1.96)` has lower bound `>0.5`. Report Elo and
by-color results. No sequential stopping, seed extension, or top-up is allowed.

## 11. Stage 6 — contemporaneous validation against `0379`

Run two separate 800-game, balanced-color matches with the same paired task
seeds:

```text
candidate v17 vs 0379: seeds [20330000, 20330800)
shipped FPU  vs 0379: seeds [20330000, 20330800)
root noise: add_noise=false in both runs
```

The two runs use identical non-FPU conditions and provide a contemporaneous
control. For zero-based game index `g`, both runs use seed `20330000 + g`;
the `calib020_0001` agent is red for even `g` and black for odd `g`.
Historical `~+80 Elo` is context, not pooled evidence.

Before candidate comparison, the shipped contemporaneous control must validate
the harness: its protocol/checkpoint/settings identity must match the
authenticated 4,000-game production match except for fresh seeds and the
explicit FPU-capable runner; its score-rate CI95 must overlap the frozen
production score rate `0.588875`, its point estimate must be within `0.05` of
that value, and its own CI95 lower bound must exceed `0.5`. Failure makes Stage
6 invalid rather than a candidate loss.

Require:

- Candidate v17 versus `0379` has
  `score_ci_trinomial(w, d, l, z=1.96)` lower bound `>0.5`.
- A deterministic paired bootstrap with `100,000` replicates and seed
  `20260724`, resampling with replacement the 400 two-game blocks
  `(g=2i, g=2i+1)` while keeping each game's candidate/shipped pair together,
  gives a percentile 95% lower bound for
  `(candidate score rate - shipped score rate) > -0.02`.
  This is the preregistered non-inferiority margin (about 14 Elo near 50%).
- Report whether candidate point Elo is at least the contemporaneous shipped
  Elo and whether it exceeds historical `~+80 Elo`; these are preferred
  outcomes, not substitutes for the two formal gates.

## 12. Provenance and deterministic artifacts

Every scientific artifact records:

- Formula ID and coefficient.
- Checkpoint identities.
- Complete effective MCTS configurations for both sides.
- Manifest, source-index, replay-data, config, and source-file SHA-1s.
- Complete game/position disjointness report.
- Per-position seeds and `add_noise=false` for diagnostics.
- Match seed range and exact color allocation.
- Git commit and clean-worktree state.
- Runtime and MLX versions.
- `run_kind ∈ {tooling_smoke, development, held_out, abcd, strength,
  external_validation}`.

Scientific run kinds (`development`, `held_out`, `abcd`, `strength`, and
`external_validation`) require `worktree_clean=true`; a dirty tree is a
pre-run refusal, not merely provenance to record. Pre-change goldens and
`tooling_smoke` may run on a dirty tree if they record that state and remain
scientifically forbidden. The repository must therefore be intentionally
resolved before the Stage-1 development protocol is emitted; this design
authorizes no automatic commit, stash, deletion, or cleanup.

Every artifact also records the effective `eval_batch_size`,
`stall_flush_sims`, and `pending_virtual_visits`. Protocol validation
re-derives and asserts the §2.4 triple `(14, 48, 8)` for diagnostics and both
match agents; differing triples are incomparable and fail before scientific
interpretation.

Diagnostic JSON/CSV is canonical, contains no timestamps, and must be
byte-identical across reruns. Match summaries may carry operational timestamps,
but scientific identity excludes them and binds the complete JSONL.

Outputs live under:

```text
logs/eval/fpu_v17_baseline_policy_mass/
```

No v16 production artifact may be overwritten or placed under a v16 root.

## 13. Stop rules and non-goals

Stop at the first failed prerequisite. After any scientific result:

- Do not edit the coefficient grid or thresholds.
- Do not add coefficients, interpolate, or extend seeds.
- Do not reuse development or held-out evidence to design v17b.
- Do not inspect a later stage after an earlier rejection.
- Do not treat a generation matchup as strength evidence.
- Do not change training, checkpoint weights, self-play, or promotion policy.

If v17 rejects, keep `calib020_0001` and shipped FPU. Any successor requires a
new name, mechanism, preregistration, and fresh development evidence.

The overall success expectation is deliberately low. The closest empirical
mechanism analogue, absolute `-0.20`, reduced selected-A replies by about 82%
but produced a `15.48%` late new-collapse rate; under the Stage-2 binomial
table, that collapse rate has only a `6.78%` chance to clear the 16-target
zero-collapse gate. The adaptive formula and fresh targets prevent assigning a
defensible numeric joint probability, and grid outcomes will be correlated,
so those marginal numbers must not be multiplied or treated as independent.
The honest prior is that the ≥50% reply-reduction and strict root-safety gates
may have no overlap. v17 is therefore a staged, high-risk falsification test
for a narrow window, not a high-probability promotion attempt.

Quantitatively, `r=0.15` and `0.20` reach at least `-0.20` final-floor
magnitude on `0/24` historical late roots; `r=0.25` and `0.35` do so on
`10/24`, and `r=0.45` on `14/24`. Meanwhile `r=0.35` reaches only about the
selected-A p25 application site. Stage 2 is therefore effectively a
three-point test (`0.25`, `0.35`, `0.45`) of whether the mechanism/safety
window exists, and a null is the expected outcome. Approval acknowledges that
expectation and the stop rule: a null closes v17 and does not authorize a grid
extension or rescue analysis.

The estimated authorized compute budget is approximately 48–53 GPU-hours:
about 0.30 hour for the 32-game chain and eight same-checkpoint tooling games,
12 hours for 1,600 development games, 16.5 hours for 2,200 held-out games, and
18 hours for the three 800-game matches, plus diagnostics/A–D and overhead.
This extrapolates from the approximately 30-hour 4,000-game production run.
Most cost occurs only after Stage 2; each operator gate authorizes only its
next incremental block.

## 14. Approval checklist

Before changing search code, explicitly approve or amend:

- [ ] Formula and distinct config-field semantics.
- [ ] Consumed/forbidden-evidence boundary and permitted design preflights.
- [ ] Exact-zero structural identity requirement.
- [ ] Frozen batching contract `(eval_batch_size, stall_flush_sims,
      pending_virtual_visits) = (14, 48, 8)`.
- [ ] Replay, 32-game full-chain, and eight-game same-checkpoint runner
      tooling-smoke requirements.
- [ ] Grid `{0.15,0.20,0.25,0.35,0.45}` and both shipped-only reach/safety
      bases.
- [ ] Fresh Stage-1 and Stage-3 sizes/seeds.
- [ ] Development and held-out gates, including the held-out transfer-floor
      interpretation.
- [ ] Exact count-based A/B/C/D criteria, including A severe `<=5/30`.
- [ ] 800-game same-checkpoint strength endpoint.
- [ ] `0379` non-inferiority margin.
- [ ] Expected-null interpretation, no grid extension after a null, and no
      self-play before every gate passes.
