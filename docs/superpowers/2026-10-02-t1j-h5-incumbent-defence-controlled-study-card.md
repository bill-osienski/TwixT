# H5 INCUMBENT DEFENCE — CONTROLLED ONE-FACTOR STUDY — EXPLORATORY CARD (design only)

⚠️ **Amended by §9 A1 (2026-10-03). Read §9 first.** It supersedes §1.5, §2, §3.1, §5.2 and
§6 where they conflict, and quotes each statement it replaces.

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

---

## 9. Amendment A1 — 2026-10-03, corrections before implementation (user-decided)

Review of `72e0eb08` found that the pass condition could never be met, plus four further
defects. Each superseded statement is quoted below and replaced.

**What stands unchanged:**

* the historical findings (§1.1–§1.4);
* the arms, factor levels and seeds;
* the budget-only S factor and the whole-tree FPU scope;
* the contaminated R factor;
* the 80 decisions and the separate gate;
* the run contract, stop rules and three aims.

**A1.1 — CD1 is a contrast and a control, not a sweep fixture.** This supersedes §5.2's table
and rule.

**What was wrong.** §5.2 required, in quoted text, "B-seed shows a full sweep at E1, E2 and CD1
at all three seeds", and it predicted CD1's sweep. CD1's committed baseline is 398 children
visited once plus the block (16,9) visited twice. That is **not** a full sweep, so the rule
could never pass. This was my error.

**CD1's role from now on:**

* It stays a **control**, because it defends at baseline.
* It stays a **contrast**: the same game and the same threat (17,9) as E1, four plies later.
* Under each level it is reported **descriptively**: its visit histogram, tie-set size, and
  whether it still defends. If it no longer defends, that is control harm.
* No sweep prediction is made for it.

**Revised predictions — six cells, E1 and E2 only, frozen now:**

| level | E1 | E2 |
|---|---|---|
| −0.20 | full sweep **persists** | **persists** |
| −0.50 | full sweep **breaks** | **persists** |
| −1.0 | **breaks** | **breaks** |

**Revised rule.** The result is "**consistent with the H-FPU-SWEEP conjecture at E1 and E2**"
iff all three hold:

1. B-repro passes;
2. B-seed shows a full sweep at E1 **and** E2, at seeds −3 **and** −5 (seed −1 is B-repro);
3. all **six** cells match.

Any mismatch means "**not supported as stated**". Full agreement establishes consistency at E1
and E2 under fresh-agent search — not cause, and nothing beyond them.

**A1.2 — Heuristic conjectures, not deductions.** This supersedes §3.1's rationale for the
levels, §5.2's hypothesis, and the word "losing" in §1.5.

**Why Q alone does not decide selection.** The PUCT score (`mcts.py:1076–1156`) has two parts:

| child | score |
|---|---|
| unvisited | `FPU + c·P·√(N+1)` |
| visited | `−Q(child) + c·P·√(N+1)/(1+n)` |

Two consequences:

* Visited values below the FPU do **not** guarantee a sweep: a visited child's exploration
  bonus can close the gap.
* An FPU of −1.0 does **not** stop an unvisited child winning a comparison: its own bonus can
  exceed the difference.

**Statements replaced:**

* §3.1, quoted: "−1.0 is the minimum value; with it, no unvisited child can be favoured over a
  visited one" — **wrong**.
* §5.2, quoted: "At a losing root, the shipped FPU of 0.0 exceeds every visited child's value.
  Every simulation therefore goes to an unvisited child" — **not a deduction**.

**H-FPU-SWEEP, restated as a heuristic conjecture.** When the network's estimates for the
root's visited children all lie well below the FPU, the FPU term *tends* to dominate the
comparison, so unvisited children *tend* to keep being selected. Lowering the FPU below part
or all of the visited-value range should *tend* to end a full sweep.

The six cells are predictions of this conjecture. They come from a comparison that ignores
the exploration bonus, and they can fail for that reason alone.

**"Losing roots"** means roots where the **network's estimates are negative**: −0.81 to −0.97
at E1, E2 and CD1. It does not mean established lost positions. §1.5's "The roots are
**losing**" is replaced by this.

**A1.3 — Arm-specific preservation checks.** This supersedes §2's and §4's application of "the
15 preservation checks" to every arm.

The existing module `h5_incumbent_defence` and its checks — the production of `912b601f` — are
**unchanged**. The new module defines its checks per arm.

**For every decision, in every arm — off/on equivalence.** The decision runs twice, from fresh
agents at the same seed, with capture off and then on. All of these must hold:

* the same move;
* the same search-stream and readout-stream states;
* the same counts and the same tie set;
* the same tree fingerprint;
* the reader leaves each tree's fingerprint unchanged and draws from neither stream;
* the same readings;
* the pass-through returns **the** objects;
* exactly one search per decision.

**For every decision — the budget.** The root visit count equals the arm's declared budget:
400 for B-repro, B-seed and F; 1600 or 6400 for S.

**For every decision — the configuration.** The agent's `EvalConfig` and `MCTSConfig` equal the
frozen ones (`frozen_argmax_config()`, `cfg_from(...)`) in every field except the arm's named
fields:

| arm | fields allowed to differ |
|---|---|
| B-repro, B-seed | none |
| F | `MCTSConfig.fpu_value` only |
| S | `EvalConfig.mcts_sims` and `MCTSConfig.n_simulations` only — one factor in two objects; they must equal each other |

The seed is the arm's own, and the colour is the mover.

**B-repro, additionally:**

* seed −1;
* the returned move equals the recorded move;
* the tree fingerprint equals the per-fixture fingerprint recorded in `912b601f` — five pinned
  values.

Any B-repro failure is a **STOP** before every other arm.

**B-seed:** seeds −3 and −5. No historical-move requirement.

**F and S:** seed −1. No historical-move requirement: a different move is a **result**, not a
failure.

Any check failure is a **STOP** naming the arm, the fixture, the seed or level, and the check.

**A1.4 — Refutation depends on the actual returned move.** This supersedes the incumbent card's
§2.2 as applied here, and §5.4's "R1 → R2" wording.

The rules engine examines the position **after the returned move *m***, before any R category
is applied:

| case | condition |
|---|---|
| **TERMINAL** | *m* ends the game: the incumbent wins |
| **NEUTRALIZED** | the opponent has **no** immediate winning move after *m*, i.e. *m* defends |
| otherwise | the **refutation set** is the opponent's immediate winning moves after *m*. It may differ from the frozen W. For each move in it, R0, R1 or R2 is applied under *m*'s node. |

This applies uniformly to all five fixtures and all arms. A defending control is NEUTRALIZED.

§5.4's E3 question becomes: under S, is the returned move NEUTRALIZED (a repair), or is its
refutation set visited (R2) or unvisited (R1)?

**A1.5 — An honest description of the reuse.** This supersedes §1.5's "**No rejected rule is
tested.**", which is **false for absolute FPU**.

* The F factor sets **absolute** `fpu_value` levels.
  * −0.20 is exactly the v16a candidate rejected as a general 400-simulation setting.
  * −0.50 and −1.0 belong to the same absolute family.
* This study **re-examines absolute FPU strictly as a diagnostic intervention at five
  fixtures**:
  * it does **not** reopen that rejected status as a general setting;
  * it measures **none** of the collateral that rejected it (late new-collapse);
  * no level may be proposed as a setting.
* The two **policy-mass** rules (`Q_parent − r·√P` and `−r·√P`) are **not** tested.

**A1.6 — Frozen names.**

| item | name |
|---|---|
| module | `scripts/GPU/alphazero/h5_incumbent_defence_factors.py` |
| gate (the 23rd) | `scripts/GPU/alphazero/h5_incumbent_defence_factors_authorization.py`, declaring `H5_INCUMBENT_DEFENCE_FACTORS_AUTHORIZED = False` |
| tests | `tests/test_h5_incumbent_defence_factors.py` |
| evidence | `docs/superpowers/evidence/t1j-h5-incumbent-defence-factors-EXPLORATORY/` |
| CLI | `python -m scripts.GPU.alphazero.h5_incumbent_defence_factors {run, run-claimed}` |

* The evidence directory is the frozen subdirectory of `h4_runner.EVIDENCE_ROOT`. It is a
  module constant and is create-only.
* There is **no `--out-dir`**: the module writes only there, and refuses any other path. Tests
  redirect only by patching `EVIDENCE_ROOT`, as in the earlier modules.
* `h5_incumbent_defence.py` is **reused by import and not edited**.

**A1.7 — Scope.** This amendment authorizes design changes and a local commit only. It
authorizes no implementation, model load, decision, gate, game, T1j call or push. The
80-decision run needs separate approval.
