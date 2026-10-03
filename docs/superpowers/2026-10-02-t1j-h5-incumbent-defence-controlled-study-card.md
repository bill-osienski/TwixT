# H5 INCUMBENT DEFENCE — CONTROLLED ONE-FACTOR STUDY — EXPLORATORY CARD (design only)

**Written 2026-10-02. DESIGN ONLY.** No implementation, model load, decision, gate, game or
push. Each later step needs its own authorization.

🔴 **EXPLORATORY; diagnostic fixtures.** This card follows the defence mechanism study
(card `2026-10-02-t1j-h5-incumbent-defence-mechanism-card.md`, run `912b601f`). It keeps
three aims apart, and only the first two are in scope:

1. **Explain** the fresh-agent searches at the five fixtures.
2. **Repair** the tactical omissions at those fixtures.
3. **Demonstrate stronger play** — **out of scope**. No result here bears on it (§5.5).

H4's and H5's strength results are unchanged. **FPU is not presumed to be the cause**: it is
one of three factors, and its hypothesis can be refuted (§5.2). No rejected approach is
revived (§1).

---

## 0. What the baseline run established (`912b601f`), in corrected wording

The run used fresh agents at seed −1 under the unchanged frozen configuration. All 15
preservation checks passed at every fixture, and parent and child were COMPLETE. Values are
given from the incumbent's perspective.

| fixture | root network value | root visit pattern | returned move | saving moves |
|---|---|---|---|---|
| E1 | −0.944 | **400 simulations → 400 different root children, 1 visit each** | (1,2), the (row, col) minimum of the 400-way tie | (16,9) and (17,9): **C** (1 visit each) |
| E2 | −0.813 | **400 × 1 visit** | (1,4), the tie minimum | (12,5): **C** |
| E3 | +0.678 | concentrated (79 / 38 / 22, then 261 × 1) | (11,10), the unique leader | (16,6), (17,7): **B** (prior ranks 415, 445 of 451) |
| CD1 | −0.974 | 398 × 1, and the block (16,9) × 2 | (16,9), the unique leader | (16,9) **D**, (17,9) C |
| CD2 | −0.396 | concentrated (20 children) | (12,5), 53 visits | (12,5) **D**, (12,4) B, (13,5) B |

Refutation categories at E1–E3: **R1** at all three. The opponent's winning reply W was
never visited under the returned move.

**Corrected wording** (user review, 2026-10-02):

* E1 and E2 are **choices** — deterministic ones, resolved by the coordinate tie-break.
* These trees explain the **fresh-agent** moves only. They do **not** explain the original
  games' searches, or the in-game follow-up (1,3); those trees were never recorded.
* A value-based tie-break would **not** have fixed E1 or E2:

  | fixture | best-valued visited move | value rank of the saving moves (of 400) | value rank of the returned move |
  |---|---|---|---|
  | E1 | (12,8), −0.363 | (16,9) 359, (17,9) 264 | 384 |
  | E2 | (15,13), −0.565 | (12,5) 55 | 372 |

* The FPU-sweep explanation is plausible; its causal contribution is **not** established. E3's
  low-prior starvation is supported, not proven causal.

## 1. Relation to the rejected lines — read from their original evidence (2026-10-02)

Each row below was read from its design document and its result artifact, not from a summary.

### 1.1 Absolute FPU `−0.20` (v16 selected-A diagnostic → v16a held-out validation)

**What was tested.** `MCTSConfig.fpu_value = −0.20` against `0.0`, at 400 simulations, on
`search_with_root` — the same synchronous path the incumbent uses. The test set was 324
game-held-out, non-selected positions (40 opening, 100 early-mid, 100 midgame, 84 late),
judged under a frozen decision table.

**Why it was rejected.** The automatic-reject rule (any stratum with n ≥ 20 at a new-collapse
rate ≥ 10%) fired:

| metric | value |
|---|---|
| late new-collapse rate | **13/84 = 15.48%** |
| late-red / late-black | 14.29% / 16.67% |
| overall new-collapse rate | 15/324 = 4.63% |
| top-move flips | 27.16% |
| effective children | −34.1% |

**The postmortem's account:** 12 of the 15 collapses committed to a network-dispreferred move
(prior rank #16 to #207), "chosen by **exploration order**". This happened at near-even,
flat-prior, high-branching roots, where the absolute floor sits near the node's value.

Evidence:
* `docs/superpowers/specs/2026-07-10-v16a-unbiased-fpu-validation-design.md`
* `docs/superpowers/specs/2026-07-10-v16a-fpu-reject-postmortem.md`
* `logs/eval/v16a_fpu_unbiased/neutral_fpu_sweep_by_stratum.csv` and
  `v16a_new_collapse_postmortem.csv` (both **tracked**)

### 1.2 Parent-relative policy-mass FPU (v16 successor)

**What was tested.** `FPU = Q_parent − r·√P_explored`, on `search_with_root`. Before any
nonzero `r` could run, the `r = 0` setting (`FPU = Q_parent`) had to pass the safety table
against the shipped FPU.

**Why it was rejected.** On the 80-row tuning split, 40 controls, it did not: lower-prior
control flips were **11/40 = 27.5%**, against a frozen limit of < 10%. That rejected the whole
family, and no nonzero `r` ever ran. The postmortem implicated replacing the shipped FPU with
the `Q_parent` baseline as the destabilizing step.

Evidence:
* `docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md` §3, §5, §6.2
* `logs/eval/fpu_v16_policy_mass_v2/diagnostic/production_v2_b400amend_4000g_seed20300000/tuning/controls_gate.json`:
  `r0_qualified: false`, `"control_flip_rate=0.2750>=0.1"`. **Untracked**: `logs/*` is
  gitignored, so this file exists only on this machine.

### 1.3 Baseline-preserving policy-mass FPU (v17)

**What was tested.** `FPU = −r·√P_explored`, with no `Q_parent`, on a fresh 1,600-game
reservoir and a 32-position corpus (16 targets, 16 controls). The grid was `r ∈ {0.15, 0.20,
0.25, 0.35, 0.45}`.

**Why it was rejected.** At `r = 0` it was identical to shipped (32/32). But **all five**
coefficients failed the §7.2 safety gates:

| failure | measured | limit |
|---|---|---|
| target new-collapse rate | 0.0625–0.125 | ≥ 0.05 rejects |
| control flips to a lower-prior move | 0.3125–0.50 | ≥ 0.10 rejects |
| reply reduction, at 0.15 and 0.20 | 0.23 and 0.37 | < 0.5 also fails the mechanism gate |

The selected coefficient was `null`. Per §13, a null closes v17: no grid extension.

Evidence:
* `docs/superpowers/specs/2026-07-24-v17-baseline-preserving-policy-mass-fpu-design.md` §4,
  §7.2, §7.3, §13
* `logs/eval/fpu_v17_baseline_policy_mass/development/selected_coefficient.json`
  (**untracked**; its development artifact has sha1 `af7778c8…`)

### 1.4 Adjacent closed lines that bound this card's other two factors

* **`c_puct`.** The v16 falsification showed lowering it worsens A and funnels visits; it is
  closed as a fix (ledger item 39).
* **Competitive readout** (closed 2026-08-08):
  * Candidate 1: all-ply argmax beat the tournament readout 54–10. That was a diagnostic only.
  * Candidate 2: a Hoeffding-LCB override returned a 408–388 null in 800 games.
  * Closure terms: no third formula, no rescue.
* **Convergence atlas** (closed 2026-08-05): the tree-local search axis is recorded as
  exhausted.
* **The ledger's own terms**, read from its closure statement: any further search-reliability
  proposal "requires a new written mechanism and fresh preregistered evidence". It forbids
  relaxing the control gate, extending the grid, and reusing consumed splits. The programme
  closed on 2026-08-12.

### 1.5 What this study adds, and what it does not

**Adds:** a fixture-level, move-level attribution in a regime none of those studies measured.

* The roots are **losing**: the incumbent's root network value is −0.81 to −0.97, and every
  visited root child is valued below the shipped FPU of 0.0.
* There, every simulation went to an unvisited child, and a coordinate tie decided the move.
* This is plausibly the same shipped-FPU breadth behaviour that v15 and v16 measured at depth 2
  as "first-touch opponent-reply scanning" (on selected A, replies scanned 134.7 → 24.5 under
  −0.20), now seen at the **root**.
* None of those studies observed whether an **immediate threat** was defended or missed.

**Does not add:**

* any candidate setting;
* any **collateral** evidence — the very property on which all three FPU lines failed;
* any strength evidence.

**No rejected rule is tested.** `Q_parent` and policy mass are not used. The absolute FPU levels
of §3.1 are **mechanism probes at five fixtures**, never settings:

* there is no "smallest passing level";
* no level may be proposed from this study;
* v16a's late-collapse failure cannot even be measured here.

## 2. The frozen baseline

**What is held fixed:**

* **The agent:** the unchanged incumbent — `calib020_0001` under `frozen_argmax_config()`
  (400 simulations, `search_with_root`, argmax readout, `c_puct` 1.5, FPU 0.0, every other
  `MCTSConfig` field at its default), bound to H5's identity.
* **Decisions:** a fresh agent for each decision, seed −1.
* **Fixtures:** the five fixtures, with their pins (frozen set sha256 `28c03f9f…`; incumbent
  card §9 A1.6).
* **Harness:** the existing `h5_incumbent_defence` harness — pass-through, fingerprint,
  reader, and the 15 preservation checks per decision (capture off, then on).

**B-repro — the reproduction control, run first.** The unchanged baseline is re-run at seed −1.
Its per-fixture fingerprints must **equal** those recorded in `912b601f` (5/5). This
establishes that the harness, model and trees are the ones already observed. Any mismatch is a
**STOP** before any factor runs.

**B-seed — baseline stability.** The unchanged baseline is also run at synthetic seeds −3 and
−5, all five fixtures. It is descriptive, and the interpretation rule in §5.2 uses it. No
research seed is involved.

## 3. One-factor comparisons — never combined

Each comparison changes **exactly one factor** from the baseline, at all five fixtures, at
seed −1. No two factors are ever changed together; there are no bundled fixes.

The production builder, its `_check_config` refusal and the agent class stay **unchanged**.
The varied agent is assembled study-locally. Its configuration is diffed field by field
against `cfg_from(frozen_argmax_config())`, and **only** the factor's fields may differ.

### 3.1 F — first-play value (`MCTSConfig.fpu_value`)

**Levels:** {−0.20, −0.50, −1.0}, against the baseline's 0.0.

The levels come from the baseline's **observed root values**, before any factor result.
Values are for the incumbent:

| fixture | visited-child value range | visited children above −0.20 | above −0.50 | above −1.0 |
|---|---|---|---|---|
| E1 | [−0.888, −0.363] | 0 | 4 | 400 |
| E2 | [−0.940, −0.565] | 0 | 0 | 400 |
| CD1 | [−0.967, −0.840] | 0 | 0 | 399 |
| E3 | [−0.704, +0.008] | 1 | 61 | 264 |
| CD2 | [−0.178, +0.078] | 20 | 20 | 20 |

* **−0.20** is v16a's historical value. It serves as an **anchor** whose predicted effect at
  E1, E2 and CD1 is *none* (§5.2).
* **−1.0** is the minimum value; with it, no unvisited child can be favoured over a visited
  one.
* FPU acts at **every** node, not only the root. Any attribution is therefore to the shipped
  knob as a whole. A root-only variant would be a new mechanism, and is excluded.

### 3.2 S — simulation budget

**Levels:** {1600, 6400}, against the baseline's 400. `EvalConfig.mcts_sims` and
`MCTSConfig.n_simulations` express one factor in two objects; `validate_ply` checks the budget.

* `c_puct` stays fixed, as do all other allocation levers:
  * `c_puct` is closed (§1.4);
  * closeout root-visit forcing exists only on the batched path;
  * widening was the convergence atlas.
* Once the budget exceeds the candidate count, the pigeonhole principle breaks a one-visit tie
  **by construction**. S therefore answers *repair with more search*, not the sweep mechanism.

### 3.3 R — readout tie resolution (post-tree, on the baseline trees)

**Variants:**

* **R-prior** — visit argmax, ties broken by prior (descending);
* **R-value** — visit argmax, ties broken by the incumbent's value of the visited child
  (descending).

Both are deterministic functions of the committed final trees, so they need **no decision**.

* **Contaminated.** At E1 and E2 both outcomes were already seen during review:

  | variant | E1 | E2 |
  |---|---|---|
  | R-value | (12,8), not saving | (15,13), not saving |
  | R-prior | (16,9), saving | (2,21), not saving |

* **Inert** at E3, CD1 and CD2, by construction: each has a unique visit leader.

R is therefore **recorded, not tested**. Because the readout line is closed, no readout
candidate may follow.

## 4. Controls — frozen before execution

* **CD1 and CD2 at every level.** A level under which a control no longer defends is reported
  as **control harm at that fixture**.
* **B-repro** (§2) and the **configuration diff** (§3), on every agent.
* **The 15 preservation checks** on every decision.
* **Absent:** collateral on positions other than the fixtures. This is stated, not assumed.
  Collateral is exactly where all three FPU lines failed, and this study cannot see it.

## 5. Pre-registered predictions and interpretation — frozen before execution

### 5.1 Definitions

* **Defends:** the returned move is in the fixture's pinned saving set.
* **Full sweep:** the maximum root-child visit count is 1, **and** the tie set equals the full
  set of visited children.
* **Categories:** A, B, B0, C, D and R0, R1, R2, as in the incumbent card (§2, §9 A1.2). Every
  value names its perspective.

### 5.2 H-FPU-SWEEP — the only factor with a mechanism prediction

**Hypothesis.** At a losing root, the shipped FPU of 0.0 exceeds every visited child's value.
Every simulation therefore goes to an unvisited child, until the budget runs out.

**Predicted root behaviour** — an approximate PUCT comparison: exploration terms are ignored,
and FPU also acts below the root.

| level | E1 | E2 | CD1 |
|---|---|---|---|
| −0.20 | sweep **persists** | **persists** | **persists** |
| −0.50 | sweep **breaks** | **persists** | **persists** |
| −1.0 | **breaks** | **breaks** | **breaks** |

E3 and CD2 were not full sweeps at baseline, so no sweep prediction is made for them.

**Rule.**

* The result is "**consistent with H-FPU-SWEEP at these fixtures**" only if:
  * all **nine** predicted cells match, **and**
  * B-seed shows a full sweep at E1, E2 and CD1 at all three seeds.
* Any mismatch means "**not supported as stated**".
* Even full agreement establishes **consistency at these fixtures under fresh-agent search**,
  not cause, and nothing beyond them.

### 5.3 Repair (aim 2)

* A level "**repairs** fixture X" iff X defends under that level, with every check passing.
* It is reported alongside the controls' status. A repair at E1–E3 that loses a control's
  defence is "**repair with control harm**".
* Levels are **not** ranked or selected.
* A repair at a fixture says nothing about collateral elsewhere.

### 5.4 Budget (S)

**Descriptive only**, with no predictions. For each fixture and level the record carries:

* whether the incumbent defends;
* the saving categories and the refutation category;
* the root visit histogram.

E3's specific question is whether W becomes visited under the returned move (R1 → R2) as the
budget grows.

### 5.5 Stronger play (aim 3) — out of scope

No result here may be stated as, or used as, strength evidence. A candidate would need, in
turn:

1. a new card with a new written mechanism (§1.4);
2. a fresh, preregistered corpus with collateral gates at least as strict as v16a's and v17's;
3. a separate strength match;
4. an explicit reopening of the closed programme.

### 5.6 Stop rules

* A B-repro mismatch: **STOP** before any factor runs.
* Any preservation or configuration-diff failure: **STOP** at that decision.
* After results are seen: no retry, no extra level, seed or fixture, and no combination of
  factors.

## 6. Size, contract, gate (for a later authorization)

**Decisions.** Each decision is an off/on pair, so there are 2 per fixture per arm.

| arm | decisions |
|---|---|
| B-repro | 10 |
| B-seed (2 seeds) | 20 |
| F (3 levels) | 30 |
| S (2 levels) | 20 |
| R | 0 (post-tree) |
| **total** | **80**, with **one model load** |

**Time.** An estimate, not a measurement: at the observed ~1 s per 400-simulation decision,
the run is under about 10 minutes, within the 1800 s parent watchdog.

**Seeds.** −1, −3 and −5, all synthetic (fixture mode). No research seed.

**Contract.** The established one (cap-completion card §7–§10; incumbent card §9 A1.7):
* a parent claim with a nonce;
* the child in its own session, under the hard watchdog;
* the child's streams captured to files;
* an fsynced journal;
* a record on every exit;
* `watchdog.json`;
* a process check that prints PID and command.

**Gate.** A new single-declaration gate, the 23rd, created **closed**.

**Next authorizable steps:**
1. implementation, with synthetic controls and the gate closed;
2. one run.

## 7. What it may conclude

| aim | may conclude | may not conclude |
|---|---|---|
| explain | whether the root behaviour at E1, E2 and CD1 matches H-FPU-SWEEP's nine predictions, under fresh-agent search | cause; anything about the in-game searches |
| repair | which single-factor levels make each fixture defend, and at what control cost, at these five fixtures | that a level is safe or desirable anywhere else |
| stronger play | nothing | anything |

## 8. Scope

Authorized by this card: **nothing executable.** No implementation, model load, decision,
gate, game, T1j call or push.
