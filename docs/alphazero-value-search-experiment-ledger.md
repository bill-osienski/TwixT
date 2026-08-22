# AlphaZero Value and Search — Experiment Ledger

**Created:** 2026-06-26 · **Status:** **CLOSED 2026-08-12** · **Scope:** the completed value-head calibration, search-reliability, competitive-readout and training-strength programme centered on `calib020_0001`.

**"Closed" means no further experiment is authorized; the record itself stays append-only.**
Results, errata and corrections are still added — nothing here is sealed or rewritten. Reopening
the programme would take a new scope and authorization, not an edit to this line.

The durable, append-only record of the connected calibration, search, readout and training
programme: what changed, the hypothesis and rationale, how each branch scored on its applicable
acceptance gates, what was concluded, and what **not** to retry. **Read this before proposing any
new local variant** — the programme is closed, and a change on the
[do-not-repeat](#do-not-repeat-prevents-going-in-circles) list (or another sweep of an exhausted
knob) is not a new hypothesis.

> **Next-session memory:** `docs/superpowers/2026-08-12-next-session-handoff.md` condenses this
> closeout, the latest frozen-parent plan and result, the surviving infrastructure, seed state and
> the separate product-model-alignment workstream. This ledger remains authoritative if they differ.

> **Colour-rule erratum (2026-08-10; applies to the competitive-readout and training matches below).** The preregistered rule that compared each candidate's per-colour 95% upper score bound with an absolute `0.50` null was mis-specified for a game with board-colour advantage. Equal agents need not score `0.50` in each colour. Moreover, estimating the colour null from the same two-agent match cannot identify candidate-specific colour interaction: candidate strength and board-colour effects are confounded, and the two adjusted deficits are equal by construction. Historical rule firings remain recorded as protocol facts, but they do **not** establish independent red/black rejection or black-specific harm. Every promotion verdict below is unchanged because its aggregate bar failed independently. Prospectively, retire the absolute `0.50` per-colour veto; use the aggregate criterion unless a future card preregisters an independently measured colour baseline.

> **Research-programme closeout (2026-08-12) — ALL EXPERIMENTAL LINES CLOSED; keep `calib020_0001`.**
>
> | field | durable conclusion |
> |---|---|
> | **Overall goal** | Find a checkpoint or same-budget search/readout policy with preregistered, statistically supported playing-strength improvement over the current best, without breaking the established safety/guardrail evidence. |
> | **Hypothesis families tested** | Value-head calibration and guardrails; adapter/projection cleanup; searched-continuation and depth-2 observables; c_puct and FPU/policy-mass search reliability; convergence/progressive-widening diagnostics; competitive move readout; ordinary training continuation with cold and warm replay; and frozen-parent opposition. |
> | **Conclusion** | **No tested candidate met its promotion bar. `calib020_0001` remains the best-supported checkpoint and the experimental programme is closed.** No experiment is running or authorized. |
> | **Primary reason** | Every credible branch reached a preregistered stop: collateral/safety failure, lack of selectivity or feasible corpus, null against the strength bar, clear regression, or bar-not-met parity interval. The last open mechanism, frozen-parent opposition, scored `0.46625` (CI95 `[0.4177, 0.5148]`), so it was not shown stronger and did not earn its conditional generalization match. |
> | **What the result does establish** | `calib020_0001` survived multiple independent challenges spanning model calibration, MCTS behavior, readout and training-data generation. The negative/null results sharply reduce the credible local search space and prevent repeating nearby variants under new names. |
> | **What it does not establish** | It does not prove `calib020_0001` is globally optimal, that all future training is futile, or that no fundamentally new architecture/data/objective could improve it. It establishes that the tested families and their nearby rescues are exhausted under their recorded evidence and decision rules. |
> | **Why not run more games or nearby variants** | Those continuations were frozen out before the results. Enlarging a match after observing parity, changing dose, inspecting intermediate checkpoints, extending grids or relaxing gates would be post-hoc selection rather than a new confirmatory experiment. See do-not-repeat entries `#1–#51`. |
> | **Infrastructure that survives** | The single Metal-owning inference arbiter, model-addressed routing, dual-root game seam, learner-only filtering, deterministic colour assignment and fail-closed worker/server paths are shipped and tested (`2,881` passed / `0` failed). The arbiter also completed the full fp6 workload—roughly 1,500 games at 400 simulations—without the driver abort that terminated fp5. `#50` prohibits the two-server design; the abort was its consequence, and the prohibition stands unchanged. |
> | **Process conclusion** | Preregistration plus hostile verification caught defects that ordinary green tests or source inspection missed: a circular oracle, a patch to the wrong symbol owner, tests whose names exceeded their assertions, a seed collision hidden by substring matching, and a self-falsifying provenance statement. Future work should retain independent oracles, behaviorally negative-constructed tests, exact-token identity checks and commit-bound gates. |
> | **Only open workstream** | **Product-model alignment, not research:** determine exactly what produced `server/model.onnx`, reproduce a verified ONNX export of `calib020_0001`, establish native-versus-ONNX numerical parity, and review deployment/rollback separately. This closeout authorizes none of those changes. |

> **Portfolio result, stated narrowly:** calibration, FPU policy mass, depth-2 provisional backup, the convergence atlas, competitive readout, ordinary continuation and frozen-parent opposition are closed. None promoted a replacement. This is a durable result about the tested search space—not evidence that every possible future idea has failed.

> **Key conclusion (updated 2026-07-29, post v17 null — the policy-mass FPU line is closed):** The A black-pre-drop calibration target is explained as a **400-sim search artifact**, not a stable value-head defect. v14 closed the adapter cleanup line; v15 showed the 400-sim backup came from broad depth-2 frontier optimism; and the budget/trajectory checks showed BASE A collapses with more search (**400/1600/6400 mean +0.2570 → +0.0626 → −0.0451**, gate-over **50.0% → 30.0% → 10.0%**, severe **43.3% → 6.7% → 3.3%**), with the apparent predrop “drop” mostly a selected shallow-search bump (**6400−400 = −0.573 at predrop, −0.001 at drop**). Therefore **value calibration against A remains unjustified**: no v15 Phase 1, no frontier hard-value correction, and no further adapter/projection/schedule cleanup. v16 then falsified c_puct and showed that negative FPU directly reaches the first-touch reply-scanning mechanism on the selected A set: `fpu_value=−0.20` moved mean **+0.2570 → −0.0344** and opponent replies **134.7 → 24.5**. However, the frozen v16a game-held-out test **rejected absolute `fpu_value=−0.20` as a general 400-sim setting**. Across 324 held-out positions it caused **15 new collapsed roots (4.63%)**, and the preregistered stratum reject gate fired in late play: **13/84 = 15.48%** new collapses, including **6/42 = 14.29% late-red** and **7/42 = 16.67% late-black**. Effective children fell **107.58 → 70.92** (−36.66; about −34.1%), top-move flips reached **27.16%**, and late collapsed roots rose **17/84 → 28/84**, despite small central value movement (mean mover delta **+0.0028**, median absolute **0.0180**, p95 absolute **0.2822**). The selected-A result remains valid **mechanistic evidence**, but the fixed absolute candidate does not generalize safely. The context-relative policy-mass successor was then implemented and taken through a fresh, fully fingerprinted production-v2 corpus, but it failed its prerequisite before any nonzero coefficient ran: `r=0` (`FPU=Q_parent`) changed the selected move to a lower-prior move on **11/40 tuning controls = 27.5%**, exceeding the frozen `<10%` collateral limit. Per the preregistered rule, this **rejects the entire parent-relative policy-mass family**; no candidate grid, frozen check, selected-A candidate gate, B/C/D, collateral, or strength match is authorized. The shipped-baseline successor **v17** (`FPU = −r·sqrt(P_explored)`, no `Q_parent`) was then implemented, preregistered and run end-to-end on entirely fresh evidence — a 1,600-game reservoir and a deterministic 32-position corpus disjoint from all prior sets — and returned a **null (2026-07-29)**: shipped vs `r=0` identity held exactly (32/32 rows byte-identical), every positive coefficient measurably changed search (32/32 positions), and **all five preregistered coefficients failed the §7.2 safety gates**. Target new-collapse was 1–2 of 16 where the rule permits zero, and control flips to lower-prior moves remained **31–50% across the tested grid** where two of sixteen already rejects. The intended reply-suppression mechanism is active (reply reduction 0.23 → 0.79 with `r`), but its control cost does not fall as `r` shrinks, so no coefficient is cheap. Per the preregistered §13 rule a null **closes v17**: no held-out generation, no A/B/C/D, no strength match, and **no grid extension**. **Both policy-mass FPU formulations — parent-relative and baseline-preserving — are now rejected on independent corpora with the same failure mode; the line is closed.** Keep `calib020_0001` and the shipped FPU. Any further search-reliability proposal requires a new written mechanism and fresh preregistered evidence; do not relax the observed control gate, extend the coefficient grid, or reuse the consumed tuning/frozen splits to rescue policy-mass FPU. The decisive success benchmark remains an equal-checkpoint, equal-400-sim, balanced-color, statistically significant head-to-head strength gain after collateral and guardrail checks pass.

> **v18 closeout (updates the key conclusion above through 2026-08-03):** The new depth-2 provisional-backup hypothesis was tested by a shipped-only, read-only preflight before any `mcts.py` implementation or positive-cap search. The proposed observable has **reach** on selected A (**0.5639 ≥ 0.50**) and a derivable candidate band, so this is not a no-op result. It nevertheless fails the properties required to justify implementation: A-vs-matched-control exposure is essentially chance (**AUC 0.5089**, bootstrap lower bound **0.39**, below **0.70/0.50**), sign dominance is **0.78475 < 0.80**, and the frozen selector has **zero target rows** and fails at every sizing tier, including the full authenticated 800-game universe. Therefore **v18 is closed at preflight**: no `mcts.py` edit, no positive cap, no Stage 0, no grid extension, no held-out generation, and no strength match. This failure means the required selectivity was not established; it does **not** prove no depth-2 provisional-backup effect exists. The consumed v18 evidence may motivate a genuinely new observable, but may not be reused as fresh confirmation or rescued by relaxing the frozen predicates after seeing the result. Keep `calib020_0001` and shipped search.

> **Convergence-atlas closeout (updates the search-reliability line through 2026-08-05):** The read-only warm-root atlas completed its authoritative 24/24 pilot and stopped on two preregistered findings. Stable-negative scarcity produced `PROJECTED_CAPACITY_NO_GO` (`1/24`, required `N=1800` versus the frozen maximum 400), so Read-out A's detector selectivity and Read-out B's gate calibration are **unanswered, not failed**. Independently, both frozen progressive-widening shapes failed the authoritative pilot check on retention alone: root retention was `1.0`, but depth-1 retention was `0.6842 < 0.90` over 12 stable-reference-eligible rows. Progressive widening and the broader tree-local heuristic line are closed; no continuation, prototype, strength match or search change ran. Keep shipped search. Full closeout: `docs/superpowers/2026-08-05-atlas-closeout.md`.

> **Competitive-readout line CLOSED 2026-08-08.** The successor to the closed atlas changed only the **final move readout** — the rule that picks a played move from a completed search — at unchanged `calib020_0001`, cold, 400 simulations. It was post-tree, so the historical A/B/C/D probes were **mathematically invariant**: they measure root value at frozen positions, and no readout can move that. Candidate 1's 54–10 all-ply-argmax diagnostic mostly established that the tournament readout carries a large aggregate sampling cost; it was not a prior for Candidate 2, whose agents sampled identically through ply 19. Candidate 2's frozen Hoeffding-LCB rule passed preflight at **6.08%** reach, then its 64-game screen lost 28–36 without triggering the one-sided futility stop. The authorized **800-game decisive match RAN and did not meet the research promotion bar**: **408–388 with four state caps, score rate `0.5125` (CI95 `0.4779–0.5471`), `+8.7` Elo (CI95 `−15.3` to `+32.8`)**. The lower bound was not above 50%. The historical absolute per-colour rule did not fire, but per the erratum no candidate-specific colour conclusion is supportable from it; integrity and search-identity evidence passed. This is a **null against the preregistered promotion bar, not a harm finding**. Per the frozen closeout, the competitive-readout line is finished: no third formula, relaxed threshold, larger match, replay-driven rescue, policy change, or product claim. Keep the existing policies. The next credible direction is a separately scoped training-line discovery, not another readout. Seeds: `docs/superpowers/2026-08-06-competitive-readout-seed-ledger.md`.

> **v16 postmortem closeout (2026-07-24):** The read-only tuning-control postmortem reproduced the exact **11/40** lower-prior flips with no frozen-check participation. Flips concentrated in opening (**5/10**) and red-to-move controls (**8/20**). On flipped rows, the mean selected-move prior rank moved **1.27→9.00**; effective children changed by **−20.10** versus **−7.35** on non-flips, while root-value movement stayed small (**+0.0080**) and mean top share fell slightly (**−0.0141**). Reply reduction was actually smaller on flipped rows (**−20.18** replies versus **−40.52** on non-flips), so the failures are not evidence that stronger reply suppression caused the move changes. The measured pattern implicates replacing shipped FPU with the `Q_parent` neutral baseline as the destabilizing step; it does **not** prove any shipped-baseline successor will work. Any successor must remove `Q_parent`, preserve shipped behavior exactly at coefficient zero, and use fresh preregistered evidence.

> **Competitive-readout closeout, 2026-08-08 — the line is CLOSED on a NULL.** Candidate 2's **800-game decisive match** returned **408–388, score rate `0.5125`, CI95 `[0.4779, 0.5471]`** — the 95% lower bound is `0.4779`, **not above `0.50`, so the promotion bar is NOT met**. Elo `+8.7` (CI95 `−15` to `+33`): the frozen Hoeffding-LCB override is **neither measurably stronger nor measurably weaker** than playing the visit leader. The match supports no candidate-specific colour conclusion, no §A integrity abort fired, and the search-identity trio passed. Notably the screen's adverse 28–36 **did not reproduce** — at 800 games the point estimate is slightly positive — which is exactly the irresolution that made the screen futility-only and the decisive match necessary. **Per the frozen Afterward: close the readout line. No third formula, no relaxed bar, no larger match, no replay-driven rescue, no policy or product change.** The 800 sidecars are archival. Candidate 2 was this line's last strength hypothesis; the next work is a separately scoped training-line discovery. Full closeout: `docs/superpowers/2026-08-08-competitive-readout-closeout.md`.

> **Training-continuation discovery, 2026-08-08 — the tested recipe is REJECTED on a clear regression.** Starting weights-only from `calib020_0001`, five iterations of calibration-free ordinary self-play training from an **empty replay buffer** (500 games and 800 training steps total) produced `model_iter_0005.safetensors`. In the frozen 400-game, equal-400-simulation match, the candidate scored **122 wins, 271 losses and 7 state caps: `0.31375`, CI95 `[0.2687, 0.3588]`, `−136.0` Elo, CI95 `[−173.9, −100.9]`**. Both colour-specific 95% upper bounds were below 50%, so the preregistered absolute per-colour rule mechanically fired for red and black; per the 2026-08-10 erratum, this does **not** establish independent colour harm. The aggregate result alone decisively rejects the recipe. The prediction of **null or mildly negative** was right in direction but materially too optimistic in magnitude. Reject this exact five-iteration, cold-buffer, calibration-free continuation recipe; keep `calib020_0001`. Do **not** evaluate iterations 1–4, extend the run, adjust nearby dose or learning-rate knobs, add games, or replicate a failed recipe. Because cold-buffer continuation and removal of the parent's calibration objective changed together, this run does not isolate the cause and is **not proof that all training has plateaued or that every new training mechanism is futile**.

> **Parent-replay bootstrap, 2026-08-09 — REJECTED, and ordinary continuation is now CLOSED.** The successor to the cold-buffer rejection changed exactly one thing: 500 games generated by the unchanged parent filled the replay buffer (44,578 positions) **before the first optimizer step**, then the same frozen five iterations ran with the calibration objective still disabled. In the 400-game parent match the candidate scored **170 wins, 224 losses and 6 state caps: `0.4325`, CI95 `[0.3843, 0.4807]`, `−47.2` Elo, CI95 `[−81.9, −13.4]`**. **The promotion bar was NOT met** — the whole score interval sits below `0.50`. The candidate scored `0.5000` as red (CI95 `[0.4318, 0.5682]`) and `0.3650` as black (CI95 `[0.2983, 0.4317]`); the preregistered absolute rule mechanically fired for black, but per the 2026-08-10 erratum this does **not** establish black-specific harm or a second independent rejection. The aggregate loss alone rejects. **The central forecast — substantial recovery without promotion — MATCHED.** Read the improvement carefully: `−136.0` → `−47.2` Elo is a **descriptive comparison across two separate runs** with different seeds and different evaluation intervals, **not a paired causal estimate**, and the stated 10–20% chance of a promotable gain **cannot be declared "correct" from a single failure**. The result **supports the parent-replay-bootstrap hypothesis — that the cold start contributed to the regression** — but **does not isolate the cold buffer causally**, since this run also carried a new training seed and 500 extra parent games. **Keep `calib020_0001`.** Per the frozen disposition: no warmup-size grid, no longer continuation, no dose or learning-rate change, no checkpoint shopping among iterations 1–4, and no replication (which a win alone would have authorized). Card: `docs/superpowers/2026-08-08-parent-replay-bootstrap-experiment-card.md`.

> **Frozen-parent opponent, 2026-08-10 — ABORTED BEFORE ITERATION 1, NO SCIENTIFIC RESULT.** The successor mechanism to closed ordinary continuation (learner plays the frozen best parent; only learner-to-move positions train) was implemented, countersigned and launched. It **aborted with exit `134`** on a **Metal driver assertion** — `AGXG15XFamilyCommandBuffer … 'A command encoder is already encoding to this command buffer'` — at the **first line of iteration 1**, immediately after the 500-game parent warmup completed normally on the ordinary single-network path. Two inference-server threads submitted concurrent work to the same device. **No checkpoint was produced, the provenance gate never ran, no evaluation was started, and nothing was retried.** This is an **implementation failure and is evidence neither for nor against frozen-parent training** — the mechanism never generated one training game, so the hypothesis and its `0.47–0.51` prediction remain **untested**. Training seed `20260810` and all `fp5` paths are spent; evaluation interval `[202609788, 202610188)` **drew zero seeds and is RELEASED UNUSED**. A successor is **not** another full run: it is a newly authorized, tiny real-GPU feasibility smoke using **one Metal-owning inference arbiter serving both networks**. See do-not-repeat **#50**.

> **Frozen-parent opponent, 2026-08-12 — BAR NOT MET, PARITY NOT RESOLVED, line CLOSED.** The successor mechanism to closed ordinary continuation (learner plays the frozen best parent; only learner-to-move positions train) finally RAN, from execution commit `13dd72f`, after an earlier attempt aborted on the two-server Metal defect. **184–211 with 5 state caps: score rate `0.46625`, CI95 `[0.4177, 0.5148]`, Elo `−23.5`, CI95 `[−57.7, +10.3]`.** The lower bound is not above `0.50`, so **the promotion bar was NOT met** — but unlike `cont5` and `warm5` this is **not a decisive rejection**: **both the score and Elo intervals include parity**, so the candidate is **not statistically distinguishable from the parent**. It was not shown stronger; it was also not shown weaker at this dose. **Do not write this as "equal".** The preregistered prediction (`0.47–0.51`, ~10% chance of clearing the bar), written before the implementation existed, matched **approximately in direction and magnitude** — `0.46625` falls just below the band's lower edge — and **a single non-pass cannot validate the 10% probability**. The `cont5 0.31375` → `warm5 0.4325` → `fp6 0.46625` progression is **descriptive across three separate runs** with different mechanisms, seeds and evaluation intervals, **not a paired causal estimate**. Per-colour figures are descriptive only per the 2026-08-10 erratum. Dose confirmed by measurement: 9,085–9,858 learner rows per iteration. **Keep `calib020_0001`.** See do-not-repeat **#51**; `#50` is unchanged and vindicated at scale — the single arbiter served ~1,500 games at 400 simulations without a driver abort.

> **Product-model alignment, Phase 3 — ANSWERED 2026-08-22 (updates the "only open workstream" row above).** The product-stack comparison ran and selected the candidate `c34b7ff3297c785a` as the served default; the baseline is retained as rollback. This is an engineering result about served bytes, not a research one — it authorizes nothing here and `calib020_0001` is unaffected. **Merged, deployed and verified 2026-08-22; the workstream is now CLOSED.** Full entry and deployment closeout: [Product-model alignment, Phase 3](#product-model-alignment-phase-3--candidate-selected-and-served-2026-08-22).

## Historical proposal check — programme now closed

This checklist remains as process history and a guard against relabelling an exhausted local
variant. It does **not** authorize a successor experiment. Any genuinely new programme would need
its own scope, hypothesis, evidence and authorization.

1. Check whether it is **only** a global-weight, retention-weight, or schedule-ratio sweep.
2. Check whether it uses **scalar-MSE retention only**.
3. Check whether it requires a **promotion match before A/B/C/D pass**.
4. **If yes to any of the above, stop and justify** why this is *not* repeating a rejected path — in writing, against the [experiment ledger](#experiment-ledger) row that already failed it.

## How to read this

The work targets one known weakness — **A, black pre-drop overvalue** — while holding three fragile "guardrail" families steady: **B goal-line**, **C old broad post-opening**, **D red pre-drop**. Each experiment is scored at 400-sim probes against the current best's baselines.

Per family we track **mean** (mean black-perspective value), **over** (% of positions overvalued), and **severe** (% severely overvalued). On the overvalued families lower is better; the gate pass bars are below.

### The four gates (pass bars vs current best `calib020_0001`)

| Gate | Family | Baseline (current best) | **Pass criteria** |
|---|---|---|---|
| **A** | black pre-drop (frozen-30, held out) | over 50.0% / severe 43.3% / mean +0.257 | mean ≤ 0.0 **and** severe materially below 43.3% |
| **B** | goal-line | over 5.6% / severe 0.0% | severe 0.0% **and** over ≤ 11.1% |
| **C** | old broad post-opening | over 33.3% / severe 13.3% / mean +0.099 | severe ≤ 13.3% **and** over ≤ 33.3% **and** mean ≤ +0.099 |
| **D** | red pre-drop | over 13.3% / severe 0.0% / mean −0.188 | severe = 0.0% **and** mean ≤ 0.0 |

**Promotion rule:** a checkpoint earns a **promotion match** (vs current best) only **after all four gates pass**. No branch below has earned one.

## Current best

**`calib020_0001`** — broad post-opening calibration from `0409`, selected early.

- Gate baselines (its own): A mean +0.257 / over 50.0% / severe 43.3% · B over 5.6% / severe 0.0% · C mean +0.099 / over 33.3% / severe 13.3% · D mean −0.188 / over 13.3% / severe 0.0%.
- **Match:** beat `0379` by **~+80 Elo**.
- **Decision: KEEP.** Strong overall, but a real **black pre-drop (A) weakness** — the thing every branch below tries to fix without breaking B/C/D.


### Implementation finding — BatchNorm calibration confound (2026-06-30)

During v4 gate-0 validation, train-mode BatchNorm was found to make calibration forwards batch-dependent. The v4 manifest initially failed real-checkpoint self-distillation until teacher caching and the teacher-retention training forward were aligned to eval-mode BatchNorm using frozen base running stats. Prior scalar-retention results remain valid for the implementation used, but BatchNorm batch-dependence is now a known confound that may have affected B/C/D retention behavior.

v4 and `v3-frozenBN-control` were both run with `--freeze-batchnorm-stats`. The control result shows BN freezing is required for clean calibration mechanics, but it is **not** sufficient to preserve B/C/D: v3 still passed A while failing B/C/D under frozen BN.

## Experiment ledger

| Experiment | Main change (knobs) | A — black pre-drop | B — goal-line | C — old post-opening | D — red pre-drop | Match | Decision / lesson |
|---|---|---|---|---|---|---|---|
| **calib010** — black-predrop v1 | Train **only** black-predrop correction, weight 0.01 | improved: over ~16–23%, severe ~6.7–10% | borderline / regressed | improved | regressed | Lost badly, **~−95 Elo** | **Reject.** Target fixed, broad play damaged. |
| **v2** — mixed pool | Correction + retention rows; retention_weight **0.5**, global **0.01**, **uniform** sampling | fixed A strongly | **fail** | **fail** | **hard fail** | no match | **Reject.** Correction worked, retention too weak. |
| **v2b** — mixed pool | v2 but retention_weight **2.0**, global 0.01 | fail/borderline @400: mean +0.038, severe 16.7% | **fail:** severe 5.6% | borderline fail: mean +0.109 | **fail:** severe 3.3% | no match | **Reject.** Stronger retention helped some; tradeoff remained. |
| **v3** — tag-stratified | Schedule **2:1:2:1**, retention_weight **1.0**, global 0.01 | **pass:** mean −0.047, severe 10.0% | **pass:** severe 0.0%, over 11.1% | **fail:** mean +0.180, over 40.0%, severe 23.3% | **fail:** severe 10.0% | no match | **Reject.** Stratification fixed mechanics + A/B, but C/D drifted. |
| **v3b** — tag-stratified, lower weight | Same schedule, global **0.005** | weak / fail-ish: mean −0.030, severe 20.0% | **fail:** severe 11.1% | **fail:** mean +0.113, severe 20.0% | **hard fail:** severe 23.3% | no match | **Reject.** Lower scalar weight didn't solve drift. **Stop scalar sweeps.** |
| **v4** — teacher-retention | raw-NN teacher value-MSE + teacher policy CE/KL on retention rows; global 0.01 / value 1.0 / policy 0.25; schedule 2:1:2:1; 1 iter (= v3); freeze_batchnorm_stats=true | **pass:** mean −0.305, over 13.3%, severe 6.7% | **fail:** over 16.7%, severe 11.1% | **fail:** mean +0.029, over 36.7%, severe 23.3% | **fail:** mean −0.038, over 36.7%, severe 16.7% | no match | **Reject.** Teacher-retention preserved clean Gate-0 self-distillation and fixed A, but B/C/D still drifted. No promotion. |
| **v3-frozenBN-control** — scalar-retention BN control | Same as v3 scalar per-row target setup; schedule 2:1:2:1; global 0.01; 1 iter; **freeze_batchnorm_stats=true** | **pass:** mean −0.106, over 20.0%, severe 13.3% | **fail:** over 16.7%, severe 5.6% | **fail:** mean +0.137, over 40.0%, severe 26.7% | **fail:** mean +0.013, over 40.0%, severe 16.7% | no match | **Reject.** Frozen-BN control shows v3 guardrail failure was not primarily a train-mode BatchNorm artifact. Scalar retention still damages B/C/D. |
| **v5** — MCTS-root-visit policy retention | raw teacher value anchor + BASE 400-sim root-visit policy CE on retention rows; global 0.01 / value 1.0 / policy-CE 0.25; schedule 2:1:2:1; freeze_batchnorm_stats=true | pass-ish / improved: mean −0.174, over 20.0%, severe 20.0% | **fail:** mean −0.288, over 16.7%, severe 5.6% | **fail:** mean +0.074, over 40.0%, severe 30.0% (mean passed) | **hard fail:** mean +0.046, over 40.0%, severe 36.7% | no match | **Reject.** Position-level root-visit anchors did not preserve B/C/D after A correction. Diagnose anchor-hold before any next design. |
| **v6** — searched-continuation retention | v5 source + BASE searched continuation/PV rows under B/C/D roots; value-only continuation rows; A hard correction unchanged; schedule 2:1:2:2; freeze_batchnorm_stats=true | **pass / improved:** mean −0.110, over 20.0%, severe 10.0% | **fail:** mean −0.321, over 16.7%, severe 0.0% | **fail:** mean +0.003, over 30.0%, severe 20.0% | **hard fail:** mean +0.150, over 53.3%, severe 30.0% | no match | **Reject.** Continuation rows existed for all failed roots, but D root raw values remained severe or drifted upward. Coverage was not the main problem. |
| **v6b** — D root + continuation hybrid | v6 manifest but schedule also drew `red_predrop_retention=1`, reintroducing D root teacher policy/root-visit CE; schedule 2:1:2:1:2; freeze_batchnorm_stats=true | **pass:** mean −0.308, over 3.3%, severe 3.3% | **fail:** mean −0.240, over 16.7%, severe 5.6% | **hard fail:** mean +0.118, over 56.7%, severe 23.3% | **hard fail:** mean −0.009, over 40.0%, severe 26.7% | no match | **Reject.** D root policy retention helped D only slightly but broke B/C. Root policy CE/KL is toxic as a mixed guardrail strategy. |
| **v6c** — D root value-only + continuation | v6c manifest added 30 depth-0 `red_predrop_root_value_retention` rows; D root rows value-only, no policy/root visits; schedule 2:1:2:1:2; freeze_batchnorm_stats=true | **fail / improved:** mean +0.006, over 30.0%, severe 23.3% | **fail:** mean −0.195, over 16.7%, severe 11.1% | **fail:** mean −0.007, over 36.7%, severe 16.7% | **fail:** mean +0.032, over 33.3%, severe 13.3% | no match | **Reject.** Value-only D root anchoring is less toxic than policy retention but still interferes with B/C and leaves D failing. |
| **v7** — sparse severe-D hard correction | Manifest-only: appended 8 `red_predrop_severe_root_correction` hard-value rows selected by BASE raw severe-overvalue (`target_black_value=-0.35`); v7 schedule 2:1:2:1:2; full-network training; freeze_batchnorm_stats=true | **pass / improved:** mean −0.065, over 26.7%, severe 13.3% | **fail:** mean −0.290, over 16.7%, severe 5.6% | **fail:** mean +0.002, over 30.0%, severe 20.0% | **hard fail:** mean +0.034, over 40.0%, severe 23.3% | no match | **Reject.** Sparse severe-D hard correction did not beat v6c and still broke B/C. Drift map showed nonlocal value-surface movement, not just wrong row selection. |
| **v8** — value-head-only on v7 manifest | Same v7 manifest/schedule, but `--train-value-head-only` skips encoder+policy updates; verifier proved all non-`value_head.*` tensors byte-identical and only 4 value-head tensors changed; freeze_batchnorm_stats=true | **fail / improved:** mean +0.068, over 33.3%, severe 20.0% | **pass:** mean −0.276, over 11.1%, severe 0.0% | **pass:** mean +0.024, over 23.3%, severe 10.0% | **pass:** mean −0.056, over 36.7%, severe 0.0% | no match | **Reject for promotion, but key positive result.** B/C/D passed with value-head-only; A undercorrected. Strong evidence full-network/trunk drift caused earlier guardrail failures. Next: v8b A draw pressure. |
| **v8b** — value-head-only, A draw pressure 3 | Same v7 manifest and value-head-only mechanics as v8, but A schedule raised `black_predrop_correction=2→3`; verifier passed with only 4 value-head tensors changed; freeze_batchnorm_stats=true | **fail / worse than v8:** mean +0.102, over 33.3%, severe 26.7% | **pass:** mean −0.286, over 5.6%, severe 0.0% | **pass / degraded vs v8:** mean +0.086, over 33.3%, severe 13.3% | **pass:** mean −0.096, over 26.7%, severe 0.0% | no match | **Reject.** Higher A draw pressure did not move A and made A/C worse. Raw-A diagnostic showed value-head-only barely moved the A family; the constraint is representational, not sampling. Do not run A=4/A=5 as the next step. |
| **v9** — value head + final residual block | Same v7 manifest and v8 schedule; `--train-value-head-and-final-block` updated only `value_head.*` plus final residual block `encoder.blocks.5` trainable tensors; strict verifier passed and all frozen tensors / BN running stats were byte-identical | **pass:** mean −0.089, over 30.0%, severe 16.7% | **fail:** mean −0.238, over 22.2%, severe 11.1% | **fail:** mean +0.067, over 46.7%, severe 30.0% | **fail:** mean −0.115, over 26.7%, severe 20.0% | no match | **Reject.** Final block gave enough flexibility to fix A, but immediately reintroduced B/C/D guardrail drift. Do not run v9b last-2 blocks as the next step; broader partial unfreeze is expected to worsen this failure mode. |
| **v10** — final block + root/continuation schedule | Config-only from v9: same v7 manifest, same `--train-value-head-and-final-block`, but enabled dormant B/C root-retention tags plus D root-value retention; schedule `2:1:1:1:2:1:2:1` (11 draws/step); telemetry clean (`calib_n_drawn_total=1760`, `n_teacher_retention_drawn=320`, policy CE/KL active) | **pass / near margin:** mean −0.004, over 20.0%, severe 16.7% | **fail / near-pass:** mean −0.195, over 11.1%, severe 5.6% | **pass:** mean +0.016, over 23.3%, severe 10.0% | **fail / near-pass:** mean −0.067, over 26.7%, severe 3.3% | no match | **Reject, but best near-pass.** Root+continuation schedule recovered C and preserved the A fix. Remaining blockers were narrow: B one severe row (`game_000015_ply_19` +0.6435) and D one barely-severe row (`red_loss_game_000752_predrop_ply_70_drop_72` +0.5003). |
| **v10b** — stronger B/D schedule | Config-only from v10, increased `goal_line_retention`, `red_predrop_root_value_retention`, and `red_predrop_severe_root_correction` from 1→2 (14 draws/step); telemetry/verifier clean (`calib_n_drawn_total=2240`, `n_teacher_retention_drawn=480`) | **fail / regressed:** mean +0.095, over 36.7%, severe 30.0% | **pass:** mean −0.310, over 11.1%, severe 0.0% | **fail / regressed:** mean +0.135, over 33.3%, severe 23.3% | **fail / regressed:** mean +0.043, over 23.3%, severe 13.3% | no match | **Reject.** Stronger B/D pressure fixed B but destabilized A/C/D. v10b caused many previously-safe rows to jump upward with high top1 concentration; broad schedule-count pressure is exhausted. If continuing, branch from v10 with surgical value-only rows, not from v10b. |
| **v11** — B surgical value-only root clones | Manifest-only from v10/v7: appended 2 `goal_line_root_value_retention` depth-0 value-only clones for v10 B blockers (`game_000015_ply_19`, `game_000327_ply_63`), replaced B root-policy CE with value-only B root pressure; same final-block update surface; telemetry/verifier clean (`calib_n_drawn_total=1920`, `n_teacher_retention_drawn=160`) | **pass:** mean −0.039, over 30.0%, severe 13.3% | **fail / worse than v10:** mean −0.060, over 22.2%, severe 16.7% | **fail:** mean +0.058, over 23.3%, severe 20.0% | **fail:** mean −0.109, over 30.0%, severe 6.7% | no match | **Reject.** B value-only root clones did not isolate/fix B; B worsened and C/D failed. The v10 B issue was not simply B root-policy CE or missing value-only root preservation. Close the v10/v11 row/schedule branch; next credible path requires a new constraint/objective, not more manifest tweaks. |
| **v12** — asymmetric one-sided guardrail hinge | New objective: `asymmetric_guardrail_retention` one-sided black-perspective hinge on B/C/D root guardrails; no policy CE; A hard correction unchanged; final-block update surface; manifest 136 rows; schedule `2:1:2:2`; telemetry/verifier clean (`calib_n_drawn_total=1120`, hinge active, policy CE/KL 0) | **fail by mean only / near-pass:** mean +0.005, over 20.0%, severe 13.3% | **pass:** mean −0.214, over 5.6%, severe 0.0% | **fail by severe:** mean +0.057, over 23.3%, severe 16.7% | **fail by severe:** mean −0.088, over 23.3%, severe 3.3% | no match | **Reject, but objective is promising.** B was fixed cleanly and the hinge path engaged as intended. A missed by only +0.005 mean, but C still had a broad severe repeat-offender cluster and D had one severe plus many high non-severe over rows. Root-only guardrails are insufficient for C/D; next branch is v12b continuation guardrails, not gradient projection yet. |
| **v12b** — continuation guardrail rows | Same v12 one-sided hinge objective, but loader extension allows `asymmetric_guardrail_retention` rows with `extra_moves_json` to reconstruct searched-continuation states; new builder emitted B/C/D root guardrails plus C/D continuation guardrails; no trainer.py change; manifest 353 rows; schedule `2:1:1:2:1:2`; telemetry/verifier clean (`calib_n_drawn_total=1440`, hinge active, policy CE/KL 0) | **pass / strong:** mean −0.137, over 30.0%, severe 13.3% | **fail by severe:** mean −0.302, over 5.6%, severe 5.6% | **fail by severe:** mean +0.028, over 33.3%, severe 23.3% | **hard fail:** mean −0.093, over 40.0%, severe 16.7% | no match | **Reject.** Continuation guardrails did not solve C/D and regressed B/D relative to v12. C failures were stable repeat offenders (`game_000505`, `000565`, `000619`, `000433`, `000065`, `000309`); D showed broad/diffuse severe drift (`000176`, `000278`, `000780`, `000456`, `000438`). Coverage is no longer the likely missing piece; next branch is v13 gradient-conflict handling/projection, not another schedule, margin, or row-coverage tweak. |
| **v13** — asymmetric gradient-conflict projection | Same v12b manifest/schedule/objective, but split A-correction and guardrail-hinge gradients on the applied surface and project A away from guardrail when `dot(g_A,g_G)<0`; `--freeze-batchnorm-stats --train-value-head-and-final-block --post-opening-calibration-gradient-projection`; telemetry-fixed rerun showed projection engaged (`conflict_rate=28.5%`) | **pass:** mean −0.117, over 23.3%, severe 20.0% | **pass:** mean −0.343, over 5.6%, severe 0.0% | **fail by severe:** mean −0.083, over 26.7%, severe 16.7% | **fail:** mean −0.151, over 36.7%, severe 13.3% | no match | **Reject, but directionally positive.** Projection engaged and fixed A/B while improving C/D shape versus v12b, but C/D severe remained. Initial v13 run had projection telemetry dropped from flattened JSON; telemetry fix made the run interpretable. |
| **v13b** — projection + lower guardrail margin | Same v13 projection mechanics and v12b schedule, but `--guardrail-margin 0.05` to activate guardrails earlier; projection activity rose (`conflict_rate=41.6%`, `active_frac=28.6%`) | **pass but weakened:** mean −0.017, over 36.7%, severe 20.0% | **pass:** mean −0.370, over 11.1%, severe 0.0% | **fail by severe:** mean −0.063, over 26.7%, severe 16.7% | **fail / worse severe:** mean −0.203, over 23.3%, severe 16.7% | no match | **Reject.** Lowering margin globally made more guardrail rows active, but did not solve C/D and weakened A. Margin-tightening is exhausted; do not run 0.025 or broader hinge activation. |
| **v13c** — projection-strength scalar 2.0 | Same v13 projection mechanics, margin restored to 0.10, added `--post-opening-calibration-projection-strength 2.0`; projection strength folds into effective projection weight only when conflict is detected; telemetry clean (`strength=2.0`, `conflict_rate=36.8%`, `removed_norm_avg=0.1292`) | **pass:** mean −0.052, over 13.3%, severe 10.0% | **pass:** mean −0.243, over 0.0%, severe 0.0% | **pass:** mean −0.076, over 6.7%, severe 6.7% | **fail by one severe row:** mean −0.073, over 16.7%, severe 3.3% | no match | **Reject by strict gate, best projection result.** A/B/C all passed; D had one barely-severe repeat outlier (`red_loss_game_000728_predrop_ply_48_drop_50` +0.5441). No promotion because D requires severe 0.0%. v13d is a tightly scoped arg-only D cleanup, not blind strength/margin tuning. |
| **v13d** — v13c + red root guardrail draw 2 | Same v13c mechanics (`projection_strength=2.0`, margin 0.10), same v12b manifest, but schedule changes only `red_predrop_guardrail_retention=1→2` while keeping `red_predrop_continuation_guardrail_retention=2`; telemetry/verifier clean (`calib_n_drawn_total=1600`, `conflict_rate=34.6%`, `removed_norm_avg=0.1802`) | **pass but weakened:** mean −0.083, over 26.7%, severe 23.3% | **fail:** mean −0.265, over 16.7%, severe 5.6% | **fail:** mean −0.013, over 33.3%, severe 20.0% | **hard fail:** mean −0.113, over 33.3%, severe 20.0% | no match | **Reject.** The single D-root cleanup did not clear D; it damaged B/C/D and weakened A relative to v13c. Close the v13 projection/cleanup line. Do not run more root draw pressure, margin tweaks, or projection-strength sweeps without a new design. |
| **v14** — gated value-adapter, projection OFF | New value-only adapter surface (`value_head.*` + `value_adapter.*`, scalar gate, bottleneck 32), encoder/policy/final block/BN frozen; same v12b manifest/schedule/objective, `guardrail_margin=0.10`, projection OFF; telemetry/verifier clean (`value_adapter_gate=0.003018`, `value_adapter_grad_norm=0.001381`, `calib_n_drawn_total=1440`) | **fail / improved:** mean +0.064, over 26.7%, severe 20.0% | **pass:** mean −0.272, over 5.6%, severe 0.0% | **pass:** mean +0.063, over 30.0%, severe 6.7% | **pass but degraded margin:** mean −0.079, over 23.3%, severe 0.0% | no match | **Reject.** Adapter surface is not a no-op: A moved substantially and B/C/D formally held, but A still missed mean ≤ 0.0 and D moved toward black within the pass band. This is not underfit; do not run width 64 next. Next branch: v14b projection ON over `value_head.*` + `value_adapter.*`. |
| **v14b** — value-adapter + projection strength 1.0 | Same v14 adapter surface/objective/schedule, but `--post-opening-calibration-gradient-projection` enabled over `value_head.*` + `value_adapter.*`; bottleneck 32; projection strength default 1.0; verifier passed (`value_head.*` + `value_adapter.*` only; final block byte-identical); telemetry clean after label fix (`conflict_steps=51`, `conflict_rate=39.8%`, `removed_norm_avg=0.0727`, `value_adapter_gate=0.001667`, `value_adapter_grad_norm=0.001518`, `calib_n_drawn_total=1440`) | **fail / near-pass:** mean +0.026, over 26.7%, severe 16.7% | **pass:** mean −0.254, over 11.1%, severe 0.0% | **pass:** mean +0.044, over 23.3%, severe 6.7% | **pass:** mean −0.047, over 23.3%, severe 0.0% | no match | **Reject / best adapter result so far.** Projection improved A while B/C/D stayed inside formal gates, but A still missed mean ≤ 0.0. No promotion. Since projection helped and guardrails held, the one justified follow-up is v14c: same setup with projection strength 2.0; do not redesign or widen before v14c gates. |
| **v14c** — value-adapter + projection strength 2.0 | Same v14b adapter projection setup, but added `--post-opening-calibration-projection-strength 2.0`; label fix confirmed `calib_projection_scope=value_head_and_value_adapter`; verifier passed (`value_head.*` + `value_adapter.*` only; final block byte-identical); telemetry clean (`strength=2.0`, `conflict_steps=49`, `conflict_rate=44.5%`, `removed_norm_avg=0.1329`, `value_adapter_gate=-0.001048`, `value_adapter_grad_norm=0.001371`, `calib_n_drawn_total=1440`) | **fail / regressed vs v14b:** mean +0.060, over 30.0%, severe 20.0% | **pass:** mean −0.261, over 5.6%, severe 0.0% | **pass but at severe cap:** mean +0.052, over 30.0%, severe 13.3% | **pass:** mean −0.056, over 26.7%, severe 0.0% | no match | **Reject.** Strength 2.0 did not push A through; it regressed A versus v14b and narrowed C/D margins. Do not run strength 3.0. Best adapter checkpoint remains v14b. One final easy-lift cleanup is v14d: revert to strength 1.0 and increase only A draw pressure (`black_predrop_correction=2→3`). |
| **v14d** — value-adapter + projection strength 1.0 + A draw 3 | Same v14b adapter projection setup (`projection_strength=1.0`, bottleneck 32, same v12b objective/surface), but increased only `black_predrop_correction=2→3`; all guardrail schedules unchanged; telemetry/verifier clean (`strength=1.0`, `conflict_steps=72`, `conflict_rate=54.5%`, `removed_norm_avg=0.0742`, `value_adapter_gate=-0.000585`, `value_adapter_grad_norm=0.001501`, `calib_n_drawn_total=1600`, A draws 480) | **fail:** mean +0.051, over 33.3%, severe 16.7% | **pass:** mean −0.299, over 5.6%, severe 0.0% | **pass:** mean +0.030, over 30.0%, severe 10.0% | **fail:** mean −0.049, over 40.0%, severe 3.3% | no match | **Reject.** The narrow A-pressure cleanup did not push A through and broke D severe. This closes the argument-only adapter cleanup line: do not run A=4/A=5, strength 3.0, width 64, or another schedule/projection tweak without a new written design. |
| **v15 diagnostics** — A searched-continuation concentration + selected-branch subtree walk | Read-only diagnostics only. Phase 0 walked immediate children under the 30 A roots; Phase 0.5 re-ran deterministic BASE MCTS on the 17 positive A roots, selected positive branches to 90%/max-3, and walked every expanded descendant with PV annotation. No manifest, no training, no checkpoint. | **mechanism clarified:** Phase 0: all 17 positive roots concentrated at selected-child level; Phase 0.5: depth-2 frontier produced broad raw optimism, not child/PV path | n/a | n/a | n/a | no match | **Reject v15 Phase 1 / close v15.** Do not build depth-1 child rows, shallow PV rows, or semi-PV continuation rows. The raw optimism is broad: 5,837 nodes walked, 5,745 raw-scored, median 196 depth-2 nodes/root needed for 70% positive raw mass, median PV share of positive raw mass 0.335%. Next branch must be v16 frontier/tree-level correction or search/prior intervention. |
| **post-v15 budget + trajectory diagnostics** — A search artifact confirmation | Read-only probe/trajectory checks on BASE `calib020_0001`; no checkpoint, no manifest, no training. Re-ran A at 400/1600/6400 sims and compared five high-A predrop→drop trajectories at 400 vs 6400. | **explained artifact:** A mean +0.2570 → +0.0626 → −0.0451 as sims increase; gate-over 50.0% → 30.0% → 10.0%; severe 43.3% → 6.7% → 3.3%. Predrop inflation 6400−400 averaged −0.573; drop-ply inflation −0.001. | n/a | n/a | n/a | no match | **Close A value-calibration line / reject v16 frontier correction before build.** The original “sharp value drop” was mostly a selected 400-sim predrop bump/winner's curse, not a stable value-head overvalue. Next branch is **v16 search reliability** (c_puct first; FPU only if needed), not value calibration. |
| **v16 c_puct falsification diagnostic** — c_puct is not the fix | Read-only search-reliability diagnostic on BASE A probe rows; no checkpoint, no manifest, no training. Integrity check passed: c_puct=1.5 reproduced Phase 0 per-case root values within 1e-6, then only `MCTSConfig.c_puct` varied. | **c_puct worsens A:** c_puct 1.5→0.25 raised mean +0.2570→+0.3778, gate-over 50.0%→60.0%, severe 43.3%→50.0%. Top-child visit share rose 0.474→0.642; top-child visited children rose 134.7→232.9; corr(top_child_n_visited_children, root value)=+0.943. | n/a | n/a | n/a | no match | **Reject c_puct as a fix / continue v16 only via FPU diagnostic.** Lower c_puct funnels visits into the selected root child and increases one-visit opponent-reply frontier scanning. Raising c_puct must not be used to pass A because it would lower the metric by spreading visits onto inferior root moves, not by repairing the frontier. Next search-code lever is an opt-in FPU field with default 0.0 preserving current behavior exactly. |
| **v16 FPU selected-A diagnostic** — first knob that reaches the measured mechanism | Search-code diagnostic on BASE A probe rows; no checkpoint, no manifest, no training. Added opt-in `MCTSConfig.fpu_value` with default 0.0, routing unvisited-child q through the field. Integrity check passed: fpu=0.0 reproduced Phase 0 within 1e-6 and the full suite passed, proving default behavior is byte-identical. | **FPU works on selected A:** fpu 0.0→−0.20 moved mean +0.2570→−0.0344, gate-over 50.0%→6.7%, severe 43.3%→6.7%, and opponent replies scanned 134.7→24.5. fpu −0.35/−0.50 reached severe 0.0% but with much narrower search. `−0.20` was frozen as the single held-out candidate because it was closest to the 6400 reference (mean −0.0451) while preserving more breadth than −0.35/−0.50. | n/a | n/a | n/a | no match | **Promising mechanism / not adoption.** FPU directly suppresses first-touch opponent-reply scanning, the mechanism c_puct could not reach. But the sample is biased by A selection, and FPU gets the value by narrowing search rather than searching deeper. The next rung was the frozen v16a comparison `0.0` vs `−0.20` on a game-held-out neutral sample; see the following row. |
| **v16a held-out FPU validation** — frozen `0.0` vs `−0.20` collateral screen | Search-code diagnostic only on a deterministic stratified, game-held-out, non-selected manifest: 324 positions from 252 games, buckets 40 opening / 100 early-mid / 100 midgame / 84 late, exactly 162 red / 162 black, and zero games shared with A discovery. The 19 winner-null 280-ply state-cap marathons were retained as stressed valid samples. No checkpoint or training changes. | **Not an A gate. Neutral result:** mean mover delta +0.0028, median absolute 0.0180, p95 absolute 0.2822; top-move flips 27.16%; effective children 107.58→70.92 (−36.66, about −34.1%); top-share +0.0716; 15 new collapses / 2 resolved. | n/a | n/a | **Late collateral failure:** 13/84 = 15.48% new collapse; late-red 6/42 = 14.29%, late-black 7/42 = 16.67%; late collapsed roots 17/84→28/84. | no match | **REJECT absolute `fpu_value=−0.20` as a general 400-sim setting.** The preregistered reject rule (any n≥20 stratum with new-collapse rate ≥10%) fired independently for late overall and both late-side strata. Do not proceed to B/C/D or strength evaluation with this candidate. Selected-A success remains mechanistic evidence only. Next: read-only postmortem of the 15 new-collapse cases, then design a new adaptive/parent-relative candidate on discovery data only. The v16a held-out manifest is consumed and must not be used for tuning. |
| **v16 policy-mass successor — production v2 control qualification** | Search-code successor on unchanged BASE `calib020_0001`: `FPU=Q_parent-r*sqrt(P_explored)`, with completed-visit policy mass. After v1 corpus infeasibility and the role-feasibility/selector-v2 repair, generated one immutable 4,000-game production reservoir (board 24, 400 sims, workers 4, seed `[20300000,20304000)`, no top-up), screened 20,464 proposals, and selected a fingerprinted 120-row whole-game-isolated corpus. | **not run.** Selected-A is downstream of the failed `r0` prerequisite. | **not run.** | **not run.** | **not run.** | no match | **REJECT parent-relative policy-mass family at the prerequisite (2026-07-24).** `r0` ran successfully on the untouched 80-row tuning split but failed development safety: lower-prior control flips `11/40 = 27.5%` vs required `<10%`. Target lock-ins improved `1→0` and mean top share fell `0.390125→0.295344`, but those descriptive improvements do not override the collateral gate. No nonzero `r`, frozen check, selected-A candidate gate, A/B/C/D, collateral, or strength result was run. |
| **v17** — baseline-preserving policy-mass FPU | `FPU = −r·sqrt(P_explored)`, NO Q_parent; grid {0.15,0.20,0.25,0.35,0.45}; fresh 1,600-game reservoir + 32-position corpus | not reached | not reached | not reached | not reached | no match | **Reject (null).** All five coefficients fail §7.2 safety at development. No held-out, no Task 12, no grid extension. |
| **v18** — depth-2 provisional backup, shipped-only preflight | Proposed cap on unusually large sign-correct raw parent/leaf residuals at newly expanded nonterminal depth-2 leaves; shipped selection/FPU unchanged; preflight only, no positive-cap search | **mechanism reach only:** pooled exposed positive backup mass 0.5639 passes; A-vs-matched-control AUC 0.5089 fails | not reached | not reached | not reached | no match | **Reject (preflight null).** Selectivity and sign-dominance fail; frozen selector produces 0 targets and sizing fails even at all 800 games. No implementation, positive cap, Stage 0, held-out, grid extension, or strength match. |
| **Convergence atlas** — warm-root convergence, gate-calibration and widening diagnostic | Read-only full-prefix replay with one additive 400/1,600/3,200/6,400 ladder and three read-outs on unchanged `calib020_0001`; authoritative 24-row pilot only | **not reached:** detector selectivity unanswered after `PROJECTED_CAPACITY_NO_GO` | **not reached:** old-gate calibration unanswered | **progressive widening rejected:** both shapes retain only 0.6842 of depth-1 stable-reference moves vs 0.90 floor | **controller feasibility only:** median remaining budget 66, no prototype justified | no match | **Close at valid pilot.** Stable-negative rate 1/24 implies required `N=1800 > 400`; no continuation or full atlas. Both widening shapes independently fail the preregistered pilot check. Keep shipped search and close tree-local heuristics. |
| **Candidate 1** — all-ply argmax readout diagnostic | Final move READOUT only: all-ply visit argmax vs the shipped tournament readout (temp `1.0` <20, then `0.1`). Unchanged `calib020_0001`, cold, 400 sims, 64 games, seeds `[202608060, 202608124)` | **invariant** — readout is post-tree; the probes measure root value at frozen positions and cannot fire | **invariant** | **invariant** | **invariant** | not a promotion match | **Large argmax win, 54–10, score rate 0.8438 (CI95 0.755–0.933), +293 Elo.** Per §7.3 this is confirmation only: **no 800-game follow-up**, no change to the tournament default. The contrast combines opening `T=1.0` and post-opening `T=0.1` sampling; this run does not attribute the effect between them and is NOT search-quality evidence. Must NOT become the prior for Candidate 2. |
| **Candidate 2 preflight** — frozen replay analysis | Read-only over Candidate 1's 64 sidecars; frozen Hoeffding-LCB rule and gates applied, not revisited. No GPU, no checkpoint, no games, no seed interval touched. | n/a | n/a | n/a | n/a | not a match | **PASS, all frozen gates.** Override rate **6.08%** (61/1,003) inside `[0.5%, 15%]`; single-game share 13.1% < 50%; 0 corrupt Q; 64/64 replays validated. Establishes **REACH and SCOPE only — NOT that the overrides are good moves.** No strength claim follows. Candidate 2 reaches its 64-game screen, which needs its own authorization. |
| **Candidate 2 screen** — 64-game mechanics screen | Frozen Hoeffding-LCB post-opening override vs post-opening argmax; both agents sample the opening identically at `T=1.0`, so only the override differs. Unchanged `calib020_0001`, cold, 400 sims, 64 games, seeds `[202608124, 202608188)` | **invariant** | **invariant** | **invariant** | **invariant** | screen only — cannot promote | **Futility NOT triggered; 800-game match becomes ELIGIBLE.** Lost 28–36, score rate 0.4375 (CI95 0.316–0.559), −43.7 Elo (CI95 −134 to +41). Upper bound 0.559 is not below 0.50, so the one gate did not fire. No §A abort; mechanics validated. **The adverse point estimate favours harm but does not resolve it** — the interval is compatible with both harm and null/noise. Eligible is not a pass, and expectations should be modest. |
| **Candidate 2 decisive** — 800-game promotion match | Same frozen isolation as the screen: Hoeffding-LCB post-opening override vs visit argmax, identical `T=1.0` opening. Unchanged `calib020_0001`, cold, 400 sims, 800 games, seeds `[202608188, 202608988)` | **invariant** | **invariant** | **invariant** | **invariant** | **408–388 + 4 caps; 0.5125, CI95 0.4779–0.5471; +8.7 Elo, CI95 −15.3 to +32.8** | **PROMOTION BAR NOT MET — close the readout line.** Primary lower bound 0.4779 was not above 0.50. The historical absolute per-colour rule did not fire but is non-diagnostic; integrity and search identity passed. **Null against the research bar, not harm:** no measurable aggregate gain or loss. No third formula, relaxed bar, larger match, replay rescue, policy change, or product claim. |
| **Candidate 2 decisive** — 800-game match | Frozen Hoeffding-LCB post-opening override vs post-opening argmax; identical opening sampling both sides. Unchanged `calib020_0001`, cold, 400 sims, 800 games, seeds `[202608188, 202608988)` | **invariant** | **invariant** | **invariant** | **invariant** | **decisive match RUN** | **NULL — promotion bar not met.** 408–388, score rate 0.5125, CI95 `[0.4779, 0.5471]`; lower bound `0.4779` not above `0.50`. Elo +8.7 (CI95 −15 to +33) — neither measurably better nor worse in aggregate. No candidate-specific colour conclusion is supportable. No §A abort; search identity passed. **Readout line CLOSED**: no third formula, relaxed bar, larger match, or replay rescue. |
| **Training continuation — cont5 cold-buffer, calibration-free** | Weights-only from `calib020_0001`; empty replay buffer; five iterations × 100 self-play games and 160 training steps; parent's post-opening calibration objective removed. Frozen `model_iter_0005`; 400-game parent match, equal 400 sims, balanced colours, seeds `[202608988, 202609388)` | **not run deliberately** — strength-first discovery; probes were reserved only for a promotable result | **not run deliberately** | **not run deliberately** | **not run deliberately** | **122–271 + 7 caps; 0.31375, CI95 0.2687–0.3588; −136.0 Elo, CI95 −173.9 to −100.9** | **REJECT the exact recipe.** Promotion failed decisively on the aggregate result. The preregistered absolute per-colour rule fired for both colours but, per the erratum, is not evidence of independent colour harm. The prediction of null or mild loss matched direction but understated the regression. No iteration-1–4 search, extension, nearby recipe tweak, added games or replication. This does not isolate cold-buffer versus objective-removal effects and does not prove all training is plateaued. |
| **Parent-replay bootstrap — warm5, 500-game parent bootstrap** | Same frozen recipe as cont5 with ONE change: `--replay-warmup-games 500` fills the buffer from the unchanged parent (44,578 positions) before the first optimizer step. Calibration objective still off. Fresh `--seed 20260809`; frozen `model_iter_0005`; 400-game parent match, equal 400 sims, balanced colours, seeds `[202609388, 202609788)` | **not run deliberately** — strength-first discovery; probes reserved for a promotable result | **not run deliberately** | **not run deliberately** | **not run deliberately** | **170–224 + 6 caps; 0.4325, CI95 0.3843–0.4807; −47.2 Elo, CI95 −81.9 to −13.4** | **REJECT — and CLOSE ordinary continuation, warm buffer included.** The aggregate bar was not met (whole interval below `0.50`). The absolute rule fired for black (`0.3650`, upper `0.4317`) but is retrospectively non-diagnostic of black-specific harm; red's raw `0.5000` likewise does not establish colour-specific neutrality. **The central forecast, substantial recovery without promotion, MATCHED.** `−136.0`→`−47.2` is **descriptive across separate runs, not paired causal**; the 10–20% promotion prior is **not validated by one failure**. Supports the bootstrap hypothesis without isolating the cold buffer (new seed + 500 extra parent games moved together). Keep `calib020_0001`; no warmup grid, longer run, dose/LR change, checkpoint shopping or replication. |
| **Frozen-parent opponent — fp6** | The learner plays the FROZEN parent instead of itself; only learner-to-move positions train. 500-game parent warmup, then 5 × 200 games (exactly 100 learner-as-red / 100 as black by game id) and 160 steps; seed `20260813`; frozen `model_iter_0005`; 400-game parent match, equal 400 sims, seeds `[202609788, 202610188)` | **not run** — strength-first | **not run** | **not run** | **not run** | **184–211 + 5 caps; 0.46625, CI95 0.4177–0.5148; −23.5 Elo, CI95 −57.7 to +10.3** | **BAR NOT MET — PARITY NOT RESOLVED — line CLOSED.** Lower bound `0.4177` not above `0.50`, so no promotion; but **both intervals include parity**, so the candidate is **not statistically distinguishable from the parent** — not shown stronger, not shown weaker at this dose. Prediction (`0.47–0.51`, ~10%) matched **approximately** in direction and magnitude; `0.46625` sits just below the band, and one non-pass **cannot validate** the 10% figure. Per-colour is descriptive (erratum). Keep `calib020_0001`; see #51. |
*(The current best `calib020_0001` is the baseline row — see [Current best](#current-best).)*

## v16 policy-mass successor — reservoir protocol v1 (historical; final outcome below)

This is the first context-relative successor to rejected v16a. It is a search experiment on unchanged `calib020_0001`, not a new trained checkpoint. The rule under test is:

```text
P_explored = sum(prior(a)) over children with completed backed-up visits
FPU        = Q_parent - r * sqrt(P_explored)
```

Purpose: retain absolute `-0.20`'s ability to suppress selected-A first-touch opponent-reply scanning without reproducing its held-out low-prior collapse failure at near-even, high-branching, flat-policy roots.

Frozen production decisions:

- Generation commit: `fca9c0dc563e47274b71059749ab451fb74e47f1`.
- Checkpoint A / screen anchor: `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`, SHA1 `209cf2d4fd24a48553d259dd71b4954867b9473e`.
- Checkpoint B / reservoir opponent: `checkpoints/alphazero-v2-staged/model_iter_0379.safetensors`, SHA1 `8ad62ac432c35c6ea9b0630b8a2b8c572a0b03a1`.
- Exactly 4,800 games; base seed `20270000`; half-open seed range `[20270000,20274800)`; no top-up.
- Board 24; 400 simulations; MCTS eval batch 14; stall flush 48; opening-temperature selection for 20 plies (`1.0 -> 0.1`); max moves 280; workers 4; replay capture required.
- Selection seed `20260712`.
- Phase-primary final corpus: target/control `180/60`; tuning/frozen `160/80`; each phase gets target `30/15` and control `10/5`.
- Late-target floors: `b300_399 >= 12`, `b200_299 >= 12`.
- Proposal enumerator: side-opposed pair per cell, minimum 12-ply gap, maximum 2 proposals per cell/game; final selection remains globally at most 2 rows/game with whole-game split and side balance.
- New-collapse sub-gate stratum: `ply_bucket`; branching band remains recorded.
- Forbidden/consumed: selected-A manifest and frozen v16a neutral manifest.

Artifact root:

```text
logs/eval/fpu_v16_policy_mass_v2/
```

Machine-authoritative protocol:

```text
logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/reservoir_protocol.json
```

Historical checkpoint status (recorded before reservoir v1 completed):

- Clean `main == origin/main` at `fca9c0d` was confirmed.
- `emit-protocol` succeeded.
- `emit-gen-command` reproduced the reviewed command exactly.
- Reservoir generation started 2026-07-16 and is expected to take approximately 39 hours at historical four-worker throughput.
- The match CLI is silent until completion and writes replay sidecars incrementally.
- **At this checkpoint, no scientific result existed yet.** Qualification, geometry, screen eligibility, coefficient safety, selected-A progress, frozen performance, B/C/D guardrails, and playing strength were pending; the later v1 outcome and final production-v2 decision are recorded below.

Frozen progression/stopping rule:

```text
reservoir qualification
  -> full persisted proposal screen
  -> deterministic 240-row select
  -> absolute_off/r0 tuning controls (r0 must qualify)
  -> frozen five-value grid; choose smallest safe passing r
  -> one isolated 80-position frozen check
  -> cross-matchup + fresh held-out collateral validation
  -> selected-A mechanism + B/C/D guardrails
  -> decisive same-checkpoint 400-sim strength match
```

If the reservoir faithfully matches the protocol but fails geometry, protocol v1 is retired; never append/top-up. If `r0` or every candidate fails, reject the formula family. No self-play adoption before the final strength match passes.

Full reproducible commands and artifact descriptions belong in `docs/post-game-analysis.md` section `v16 Context-Relative Policy-Mass FPU — Operator Runbook`.

## v17 baseline-preserving policy-mass FPU — REJECTED (null), 2026-07-29

**Rule.** `FPU = fpu_value − r·sqrt(clamp(P_explored, 0, 1))`, with validation forcing
`fpu_value == 0.0`, so the operative rule is `−r·sqrt(P_explored)`. **No `Q_parent`** —
this is what distinguishes v17 from the retired v16 parent-relative rule.

**Preregistration.** Design frozen 2026-07-24, SHA-1
`944f358c0e3ef66503d2cbb56e31dabd145bafc2`. Grid `{0.15, 0.20, 0.25, 0.35, 0.45}`,
fixed in advance; §13 forbids interpolating or extending it.

### Evidence

| Artifact | SHA-1 |
|---|---|
| `development_diagnostic.json` | `af7778c84e1ea04f463febfc615e5363400d6aad` |
| `selected_coefficient.json` (rejection witness) | `fad6ccb6fe678e86fc474c6d99234dc8841d0f2a` |
| reservoir protocol | `386d14f48a05380b94d252cf815e84121f06b0b7` |
| diagnostic protocol / config | `41f1612f4706c52c9393946b4319a67998cb955f` / `6d4fe3fbd7fc1fa54d6ed3f0ecd88962a6223378` |
| corpus manifest | `15b0228edc1ed605fea799694d4ca0eda3e3468b` |
| corpus source index | `960408f0b8e980730160112cd77f9661b4c41a10` |
| selected replays (16 games) | `0b8609cff6fb0c9ac66c7f008c73d3297309b3a4` |
| `mcts.py` at run time | `b60c983399dbc5ed292de9b15944b8850a1d8508` |

Fresh evidence throughout: 1,600 games at seeds `[20310000, 20311600)`, board 24,
400 sims, batching `(14, 48, 8)`, `add_noise=false`, no top-up; deterministic
32-position corpus (16 late flat-policy targets, 16 concentrated controls, four per
phase, 16 red / 16 black, ≤2 per game, ≥12-ply spacing), disjoint from all v16
production, v16a neutral and A/B/C/D positions.

**Identity prerequisite passed.** Shipped vs `r=0` byte-identical on **32/32** rows
including `search_result_sha1`. Each positive coefficient differs from shipped on
**32/32** positions, so the identity result is not vacuous.

### Gate table — all five coefficients fail

| r | target new-collapse (`<0.05`) | control flip to lower prior (`<0.10`) | reply reduction (`≥0.50`) | verdict |
|---|---|---|---|---|
| 0.15 | **0.0625** | **0.4375** | **0.2336** | FAIL |
| 0.20 | **0.0625** | **0.3125** | **0.3712** | FAIL |
| 0.25 | **0.1250** | **0.5000** | 0.5038 | FAIL |
| 0.35 | **0.1250** | **0.4375** | 0.6827 | FAIL |
| 0.45 | **0.0625** | **0.4375** | 0.7863 | FAIL |

Every coefficient fails **both** §7.2 safety gates. On 16 targets the `>=5%`
new-collapse rule passes only at zero collapses; each coefficient produced 1–2. The
16-control flip gate permits at most one flip and rejects at two; observed rates of
0.31–0.50 are **5–8 flips**.

**Selected coefficient: `null`.** The §7.3 reply-reduction mechanism gate is met at
`r ≥ 0.25`, but never becomes decisive because safety fails at every grid point.

### Interpretation

The intended reply-suppression mechanism is active — reply reduction rises
monotonically with `r` (0.23 → 0.79) — but it carries an unacceptable control cost at
**every** strength tested, and that cost does not fall as `r` shrinks: the control
flip rate remains **31–50% across the tested grid** while reply reduction varies more
than threefold. There is no region where the intended effect is obtained cheaply.

`selected_coefficient.json` is a **rejection witness**, not an authorization: it
records `coefficient: null` bound to the development artifact, gate table and
selection context. It does not permit advancing.

### Frozen-plan consequences

- **No held-out generation.** Task 11's stop condition is met.
- **No Task 12.** Held-out collateral is not authorized and must not be built.
- **No grid extension.** §13 pre-registers a null as closing v17. Reply reduction is
  still climbing at `r = 0.45`, which invites a larger coefficient — but control flips
  are already 7/16 there, and the preregistration forbids it. Trying `r > 0.45`
  would be exactly the post-hoc grid extension the freeze exists to prevent.
- **No strength match.** Nothing reached the strength stage.

## v18 depth-2 provisional backup — REJECTED at shipped-only preflight, 2026-08-03

### Hypothesis and why it was plausible

v18 was a new intervention point, not v17b and not a third policy-mass FPU formula.
The motivating evidence was that selected-A's 400-simulation inflation came from a
**broad, shallow depth-2 frontier**, while additional search removed the bump:

- the top positive depth-1 child had mean raw black value about `-0.087` but mean
  searched black value about `+0.619`;
- selected positive branches contained `4,443` depth-2 nodes accounting for `77.3%`
  of leaf evaluations, with mean raw black value about `+0.793` and `98.8%`
  raw-positive;
- a median `196` depth-2 nodes per root was needed to cover 70% of positive raw
  mass, ruling out a few-PV-row correction; and
- increasing search from 400 to 1,600 and 6,400 simulations reduced selected-A
  mean `+0.2570 -> +0.0626 -> -0.0451`.

The frozen hypothesis was therefore:

> Preserve shipped selection and frontier breadth, but treat a newly expanded
> nonterminal leaf exactly two plies below the root as provisional. Cap only an
> unusually large sign-correct disagreement between that leaf's raw value and its
> expanded parent's raw value; if revisited, back up depth-3-and-deeper evidence
> normally.

Unlike FPU, this would act **after** a leaf had already been selected and evaluated,
so it might redirect some depth-1 reply scanning into deeper confirmation without
preventing unexplored moves from receiving their shipped first visit. The proposal
earned implementation only if shipped-only residual exposure first separated the
selected-A phenomenon from matched ordinary/tactical positions and supported a
viable fresh corpus. Merely showing large A residuals was explicitly insufficient.

### Preregistration and execution boundary

- Design revision 3 SHA-1: `7be30d4eea9eccf0316fa5927757fc404ca83ecb`.
- Execution plan revision 45 SHA-1: `0f793ca0d53562dc5926c2425b3e5b15e61099e5`.
- Binding clean HEAD: `83f47465b2dd7da76d062dc5b162c9fe902d5d31`.
- Shipped search only: synchronous 400 simulations, `add_noise=false`, `c_puct=1.5`,
  batching `(14,48,8)`; 30 authenticated selected-A rows plus a residual-blind
  1,957-position census from exactly 800 authenticated games.
- The cap grid, exposure formula, matching variables, thresholds, selector geometry,
  sizing rule, winner's-curse guard and verdict vocabulary were frozen before the
  binding measurement.
- `mcts.py` stayed at SHA-1 `b60c983399dbc5ed292de9b15944b8850a1d8508`
  and has no v18 diff from the pinned branch point. No positive-cap search ran.

### Evidence of record

| Artifact | SHA-1 |
|---|---|
| `step5_verdict.json` | `13f1fb65c414d86b5e5c0d1887c84e05bde69a6f` |
| `step5_sizing.json` | `521892899820eaf619e44515569fb601bce72b05` |
| `step5_matched_cohort.json` | `01d79f3d030b93e27fbe1aaa37ab704ad8fba343` |
| `preflight_artifact.json` | `004e72a4b589c853713bf74e38e3cd148878d4eb` |
| `census_positions.csv` | `718f9abfe221a66b26c4e79517f6ea08f0802008` |
| `crossover_tables.csv` | `ee6040b7b83b4c2f4a97366abdd22014c4084b58` |
| `residual_rows.csv` | `4f0ea1783dac4d3c8302b65fde180ecbc354a742` |
| `frozen_preflight_criteria.json` | `87f645dd25ab71b86f4a3b70369e5eebb456dfee` |
| `frozen_source_universe.json` | `631876610c10127bd0678224ab9ee8d25198c8b4` |
| `step4_a6400_reference_bundle.json` | `60c9d7dd111e5ab34367fb4849cc0d872a8f5dca` |

The public evaluator authenticated and reproduced the census, crossover rows,
leaf-level reach evidence, matched cohort, production sizing ladder, criteria,
universe, runtime source identities and A/6,400 reference bundle before evaluating
the gates. Superseded and rejected diagnostics are retained separately and are not
part of this result.

### Outcome — `PREFLIGHT_FAIL`

| Gate | Frozen requirement | Observed | Result |
|---|---:|---:|---|
| A-vs-matched-control selectivity | AUC `>=0.70`; bootstrap lower bound `>=0.50` | AUC `0.5088888889`; lower bound `0.39` | **FAIL** |
| positive-residual sign dominance | `>=0.80` | `0.7847506370` | **FAIL** |
| selected-A reach | pooled `>=0.50` | `0.5638789580` | PASS |
| terminal depth-2 fraction | `<=0.10` | `0.0011669828` | PASS |
| selector sizing | a qualifying frozen tier | `0/299` successes at every 200–700 tier; `0/1` at all 800 games | **FAIL** |

The selectivity failure is decisive: `exposure_primary_0.50` was essentially chance
at separating 30 selected-A rows from 30 residual-blind matched controls despite
10,000 frozen-seed bootstrap replicates. The record's exact interpretation is:
**required A-vs-matched-control selectivity was not established**. It explicitly
does **not** mean that no depth-2 effect exists.

The sizing failure independently exposes the same problem. Of 1,957 census rows,
24 cleared the exposure cutoff (`0.2475800522`); all 24 also cleared sign dominance
and eligible-leaf count; only 2 were near-even; and both of those were flip controls.
Thus **0 rows satisfied the complete target predicate**. Frozen role counts were
target `0`, representative `16`, identity `411`, flip `95`, unassigned `1,435`.
The full-universe failure was `target|late capacity 0 < demand 16`, not a sampling
accident at a smaller tier.

### Interpretation

The preflight is a real negative scientific result, not merely a platform test.
It shows both of the following:

1. The proposed observable reaches the mechanism: selected-A pooled exposure passes,
   terminal contamination is negligible, and a formal candidate band is derivable
   (`R_min=0.0106029774`, `R_max=0.1167949298`).
2. The observable does not provide the required **selectivity or viable target
   geometry**. It cannot distinguish the hoped-for shallow-horizon population from
   collateral-sensitive positions well enough to justify changing shipped search.

The passed reach gate cannot rescue failed selectivity, and a derivable cap band
cannot create a target corpus that does not exist. Conversely, the null does not
prove that every depth-2 provisional-backup mechanism is ineffective; it closes
this preregistered residual observable, role geometry and cap-development path on
the consumed evidence.

### Decision and frozen consequences

- **Close v18 at preflight.** Keep `calib020_0001` and shipped MCTS.
- **No `mcts.py` implementation and no positive-cap search.** The intervention was
  deliberately stopped before it could affect search behavior.
- **No Stage 0, development grid, held-out generation, A/B/C/D acceptance run or
  strength match.** None is authorized by this result.
- **No threshold, role-predicate, near-even, flip-control or sizing relaxation after
  seeing the null.** Those rules were the protection against selecting a flattering
  corpus post hoc.
- **Evidence consumed.** The 800-game universe, 1,957-position census, matched cohort
  and selected-A measurement may inform a genuinely new mechanism or observable,
  but may not serve as fresh confirmatory evidence for it.
- Any successor must state a new selective observable and falsification criterion in
  writing, use fresh preregistered confirmation, and still earn the final benchmark:
  equal-checkpoint, equal-400-simulation, balanced-color significant strength gain
  after safety and guardrail checks pass.

## Convergence atlas — CLOSED at authoritative pilot, 2026-08-05

### Hypotheses and reasoning

The atlas was designed after v18's preflight null, without changing shipped search.
It targeted two untested questions. First, the candidate set was the last untouched
tree-local axis: shipped MCTS enumerates every legal move, while `c_puct`, absolute FPU,
both policy-mass FPU formulations and v18 changed selection values or backups rather
than move eligibility. Second, the inherited collateral gates had rejected four search
interventions but had never been calibrated against movement toward stable deeper
search; zero candidate had reached a strength match.

Phase 0 established a third fact before the atlas froze: deployment search inherits a
tree and then adds its nominal budget. A deployed 400-simulation search therefore ends
at `I + 400` visits, while the historical reconstructed-position probes were fresh.
The atlas used full-prefix replay so its 320-completion features and four-rung ladder
were measured in the warm-root regime that actually plays games.

The frozen questions were:

1. Can batch-safe 320-prefix convergence features identify 400-simulation searches
   that disagree with a stable 3,200/6,400 reference?
2. Do the old collateral gates reject changes that consistently move toward that
   reference?
3. Can either of two preregistered prior-ranked widening shapes affect misleading roots
   while retaining stable deeper root moves and replies?

### Expected outcomes and stopping rules

- A selective detector would make a bounded 320+80 verification prototype eligible.
- A widening shape with strong misleading-root coverage and stable-move retention would
  make a small progressive-widening prototype eligible.
- A gate that rejected consistently convergent changes would be redesigned and frozen
  before judging another prototype.
- Failure of both tree-local methods would close tree-local heuristics and pivot to
  direct strength or a separately designed distillation project.
- A corpus-capacity failure would stop the atlas as an **operational no-go**, not be
  misreported as proof that convergence information cannot exist.
- Genuine failure of both widening shapes on the pilot would close widening without a
  third shape, regardless of whether the full atlas proceeded.

### Preregistration and evidence of record

- Qualified code: `1332bcc`, full suite 2,636 passed / 4 skipped / 53 deselected.
- Binding run HEAD: `24847869540d6ae1611bfa8bb62ee8a53428f8ac`.
- Unchanged checkpoint: SHA-1 `209cf2d4fd24a48553d259dd71b4954867b9473e`.
- Fresh pilot: seeds `[20321000, 20321024)`, sampling seed `20260806`, trajectory-relative
  phase quarters from Amendment 5.
- Complete execution: geometry PASS 24/24; pilot `verdict=OK`, `authoritative=true`,
  assigned/measured 24/24, zero failed rows.
- Evidence hashes: protocol `2152eac7f17a411290fca35b8e12ce88fc9a8128`;
  block manifest `e0c2d6e49fbd76ba41c2c4e2ef85b2f465fbbeea`; pilot artifact
  `ce15927f3c162c901b5c8211e79b21ac737f41b1`.

The earlier `[20320000, 20320024)` block stopped at its absolute-phase geometry gate
and is retired as design evidence only. It contributed no atlas position.

### Outcome — `PROJECTED_CAPACITY_NO_GO` + widening `both_fail`

| Check | Observed | Decision |
|---|---:|---|
| pilot completeness | 24/24 measured, no failures | authoritative |
| misleading frequency | 11/24 | sufficient by itself (raw 130.9 → legal `N=160`) |
| stable-negative frequency | 1/24 | `required N=1800 > 400` — **PROJECTED_CAPACITY_NO_GO** |
| no stable 3,200/6,400 reference | 12/24 | half the pilot unclassifiable under frozen labels |
| widening root retention, both shapes | 1.0 vs 0.95 floor | pass |
| widening depth-1 retention, both shapes | 0.6842 vs 0.90 floor | **FAIL** |
| controller remaining budget | median 66; zero-budget fraction 0 | feasible, not justified |

The widening rejection rests on retention over 12 stable-reference-eligible rows, not
on sparse intervention denominators. Both shapes preserved every stable root move and
dropped roughly one-third of the depth-1 replies required by stable deeper search.
For scale, `9/24` stable-negative with the observed misleading rate would have sized to
exactly `N=200`; the observed `1/24` instead drove `N=1800`, so the capacity finding was
not marginal.

### Interpretation and frozen consequences

- **Close the atlas at its valid pilot.** No continuation block or full atlas is run.
- **Close progressive widening.** Do not add, tune or interpolate another shape.
- **Read-out A and Read-out B remain unanswered.** Capacity failure is not evidence
  that their information cannot exist.
- **Do not enlarge or relabel the corpus, loosen stable-reference criteria, top up or
  reuse either pilot block.** Both ranges are consumed for their recorded purposes.
- **No prototype, A/B/C/D acceptance run or strength match.** Keep shipped search.
- The remaining-budget pass establishes only that a controller could act before the
  400-simulation budget ends; it supplies no selective signal and authorizes nothing.
- Tree-local search heuristics are closed. Direct playing-strength work is the next
  default. A high-budget distillation successor would need its own stability rule,
  because 12/24 rows lacked a stable 3,200/6,400 reference.

## Candidate 1 — all-ply argmax readout diagnostic, RUN 2026-08-07

First result of the competitive-readout line, and the first new experimental result in
this ledger since the atlas closed.

### Hypothesis

Selecting the completed search's visit leader deterministically at every ply may play
more strongly than sampling from the same visit distribution under the tournament
readout. Because the network, search algorithm, simulation budget and completed tree are
identical, any game-level difference should come from the final move readout rather than
from stronger search.

### Why the hypothesis was plausible

The tournament control deliberately samples at `T=1.0` through ply 19 and continues with
`T=0.1` afterward, while Candidate 1 always chooses the visit leader. Sampling can select
a non-leader even though the search has already assigned more visits to another move.
The comparison therefore isolates the aggregate playing-strength cost or benefit of that
stochastic readout. It does **not** isolate opening from post-opening sampling and does
not test whether the underlying search tree is better.

### Predicted result and preregistered interpretation

No numeric effect size or positive pass threshold was preregistered. The qualitative
hypothesis made an argmax gain plausible, but the governing design explicitly treated a
near-null or a clear loss as credible outcomes rather than harness defects:

- a **large argmax win** would be useful confirmation, with no 800-game follow-up;
- a **near-null** would remain plausible and require mechanics inspection;
- a **clear argmax loss** would halt the line until visit-leader reliability was
  understood;
- an integrity failure would be a harness result, not candidate evidence.

The only 64-game decision boundary was one-sided futility. Candidate 1 was a diagnostic,
not a promotion experiment.

### What ran

Authorized by `docs/superpowers/2026-08-06-candidate-1-authorization.md`, countersigned
`bill-osienski` at `2026-08-07T02:36:42Z`. Executed from commit `d2aaf4f` with a clean
worktree; every frozen parameter pinned explicitly on the command line.

| | |
|---|---|
| checkpoint | `calib020_0001`, sha1 `209cf2d4…`, both agents (`same_checkpoint: true`) |
| candidate | `argmax` — all-ply visit argmax, deterministic canonical tie-break |
| control | `tournament` — temperature `1.0` through ply 19, then `0.1` |
| games / seeds | 64, `[202608060, 202608124)` half-open, priors `[]` |
| search | cold, 400 new simulations per move, asserted per ply |
| process exit | **0** · no §A integrity abort fired |

### Result

| | |
|---|---:|
| score rate | **0.8438**, CI95 `[0.755, 0.933]` |
| Elo | **+293.0**, CI95 `[195.3, 456.7]` |
| record | 54–10 |
| as red | 27–5, 0.8438, CI95 `[0.718, 0.970]` |
| as black | 27–5, 0.8438, CI95 `[0.718, 0.970]` |
| decisive | 64/64 · state caps **0** · board-full 0 |
| average plies | 51 |
| wall clock | **3,803.8 s** — 63 min, ~59 s/game at `--workers 1` |

The identical per-colour records are **not** a signal: with ten total losses, a 5/5 split
is the most central outcome available.

### Did the result match the prediction, and why?

It matched the qualitative direction of the hypothesis and landed in the preregistered
“large argmax win” branch. Removing stochastic selection from the completed visit
distribution produced a large aggregate gain in this harness. The run did **not** match
a numeric prediction, because none was registered, and the `+293` magnitude was not
predicted.

The mechanism-level explanation remains deliberately limited. Candidate and control
differed both during the opening (`argmax` versus `T=1.0`) and afterward (`argmax` versus
`T=0.1`). The result is consistent with a cost from stochastic non-leader selection, but
without the separately authorized phase-specific replay analysis it cannot say how much
came from either phase.

### Interpretation and consequences — narrower than the number looks

- This is evidence about **readout strength using `calib020_0001` in the Python
  evaluation harness**. It is **not** evidence that the checkpoint is stronger, and
  **not** evidence about the product, which serves a different network entirely
  (`docs/superpowers/2026-08-06-model-path-provenance-audit.md`, verdict `MISMATCH`).
- The contrast includes the control's deliberate opening exploration and its
  post-opening `T=0.1` sampling. This run does not decompose their contributions and is
  not evidence of a difference in search quality.
- **`+293` Elo must NOT become the prior for Candidate 2.** Candidate 2 keeps that same
  early sampling on *both* sides and changes only the post-opening readout. The two
  experiments do not share an effect size.
- Per §7.3 a large argmax win is confirmation only: **no 800-game follow-up**, and **no
  change to the checkpoint-tournament default** (policy 2). No product change and no
  default-policy change follows from this run.

### Measured cost, for planning

~59 s/game single-worker projects an 800-game match to roughly **13 hours** at this
configuration. That was one of the run's five stated purposes and is now on the record.

### What was captured but NOT analyzed

All 64 replay sidecars, with top-two root-child visits and both Q perspectives, are in
`logs/eval/candidate1_diagnostic_replays/`. **No telemetry analysis was performed** — no
preflight, no non-leader selection rates, no hand computation of either. That requires
its own authorization.

### Next sequence

1. ~~Record this result and mark the seed interval consumed.~~ Done here and in
   `docs/superpowers/2026-08-06-competitive-readout-seed-ledger.md`.
2. ~~Separately authorize descriptive analysis of the captured replays.~~ Done —
   `docs/superpowers/2026-08-07-replay-preflight-authorization.md`.
3. ~~Run the frozen Candidate 2 preflight — formula, population and gates unchanged.~~
   Done 2026-08-07: **PASS**, override rate 6.08%. See the preflight section below.
4. ~~Candidate 2's 64-game mechanics screen is now reachable, and needs its own
   countersigned authorization with a new reserved seed interval.~~ Ran 2026-08-07:
   futility not triggered, 28–36. See the screen section below.
5. ~~Run the separately authorized 800-game decisive match.~~ Done 2026-08-08: the
   promotion bar was **NOT MET**, 408–388 plus four state caps, score rate 0.5125 (CI95
   0.4779–0.5471). See the decisive section below.
6. **Competitive-readout line CLOSED.** Keep existing policies. Any continuation is a
   separately scoped training-line discovery, not another readout formula or rescue.

## Candidate 2 preflight — frozen replay analysis, RUN 2026-08-07

Read-only analysis of Candidate 1's 64 captured replay sidecars. No GPU, no checkpoint,
no games, no seed interval reserved or consumed. Authorized by
`docs/superpowers/2026-08-07-replay-preflight-authorization.md`, countersigned
`bill-osienski` 2026-08-07T13:02:37Z, executed from `5c50d04` against the frozen 64-file
digest `a4e2bfc6…`.

### Hypothesis

The frozen Hoeffding-LCB rule fires often enough on ordinary post-opening positions to be
worth testing on real play, without firing so often that it stops being the conservative
occasional override the design hypothesized, and without concentrating in a handful of
abnormal games.

### Why the question was open in both directions

The radius is deliberately conservative: `ε(190) = 0.197`, `ε(40) = 0.429`, so a
40-visit challenger must exceed a 190-visit leader by `0.232` in root-perspective Q. A
**near-no-op was a live and legitimate outcome** — the `< 0.5%` floor exists precisely to
close Candidate 2 cheaply in that case. Equally, at 400 simulations over a broad frontier
the second-choice child is often well visited, so a usable rate was plausible too.

### Predicted result and preregistered interpretation

**No numeric rate was predicted, and no strength prediction was made or implied.** The
gates frozen 2026-08-06, before any telemetry existed, were the whole prediction:

- override rate `< 0.5%` → close for insufficient reach;
- override rate `> 15%` → close for excessive scope;
- `> 50%` of overrides from one game → close for concentration;
- corrupt Q on a visited child, or an empty population → **no verdict**, taking
  precedence over any co-occurring rate failure;
- colour split → descriptive, never a gate.

Equality at a boundary passes, the frozen operators being strict.

### Result — `PASS`, all frozen gates

| statistic | observed | gate |
|---|---:|---|
| override rate | **6.08%** (61 / 1,003) | inside `[0.5%, 15%]` |
| single-game concentration | **13.1%** | under `50%` |
| corrupt Q on visited children | **0** | none |
| population | 1,003 post-opening argmax-agent plies, 898 eligible | non-empty |
| replays matched | **64 / 64** | all sidecars validated |

`failed_gates: []`, exit `0`. Overrides appeared in 35 of 64 games.

**Descriptive only, gating nothing:** colour split 59% red / 41% black; challenger visits
at override min 8 / median 87 / max 192; by ply bucket 5.80% (20–39) and 7.37% (40–69),
with 0% on the thin 70–109 and 110+ buckets (38 and 5 plies). The argmax agent played a
non-leader on 0 of 640 opening and 0 of 1,003 post-opening plies, which confirms the
readout behaved as configured and carries no attribution about Candidate 1's win.

### Did the result match the prediction, and why?

**There was no numeric prediction to match.** The qualitative expectation leaned toward a
near-no-op, given how conservative the radius is; the observed rate is an order of
magnitude above the floor, so that lean was wrong. The median challenger at override
carried 87 completed visits, so the rule is not scraping the `n_min = 8` floor on fringe
moves.

**What passing establishes is REACH and SCOPE, and nothing more.** It does **not**
establish that the overridden moves are better ones. A rule can fire at a usable rate and
still be wrong every time it fires. Whether these overrides help is exactly what the
64-game mechanics screen, and only after that a decisive match, exist to answer. No
strength claim of any kind follows from `6.08%`.

### Consequence

Candidate 2 reaches its 64-game mechanics screen. Passing this preflight **authorizes
nothing by itself**: that screen requires its own countersigned authorization, a new
reserved seed interval, and Candidate 1's consumed interval passed as a prior.

## Candidate 2 — 64-game mechanics screen, RUN 2026-08-07

Authorized by `docs/superpowers/2026-08-07-candidate-2-screen-authorization.md`,
countersigned `bill-osienski` 2026-08-07T13:25:37Z, executed from `0d9678a` with a clean
worktree. Suite measured green immediately before launch (2,795 passed / 0 failed).
Interval `[202608124, 202608188)`, prior `[202608060, 202608124)`.

### Hypothesis

The frozen Hoeffding-LCB override, exercised against a real model for the first time,
does not cause gross harm or operational failure.

### Why the question was open

Candidate 2 introduces root-aware selection code that had never played a game. The
preflight established **reach and scope only** — that the rule fires at 6.08% — and said
nothing about whether the moves it selects are better. The repository also carries
adverse precedent: `docs/superpowers/decisions/2026-05-19-reverted-closeout-experiments.md`
records a prior readout override whose relaxed value gate added false positives and
worsened play.

### Predicted result and preregistered interpretation

**No pass bar, and no numeric prediction.** Per §8.2 the 64-game screen may only *stop*;
it has no early-success path and takes no success decision. The single preregistered
boundary was one-sided futility — a 95% score-rate interval entirely below 50%, about
`0.378` at n=64 — and for Candidate 2, unlike Candidate 1, futility **closes the readout
line**. The 55%/45% gates were deliberately excluded: they would discard roughly two of
every three candidates capable of clearing the 800-game bar.

### Result — futility NOT triggered

| | |
|---|---:|
| score rate | **0.4375**, CI95 `[0.3160, 0.5590]` |
| Elo | **−43.7**, CI95 `[−134.2, +41.2]` |
| record | 28–36 |
| as red | 12–20, 0.375, CI95 `[0.207, 0.543]` |
| as black | 16–16, 0.500, CI95 `[0.327, 0.673]` |
| decisive | 64/64 · state caps **0** · avg 49.9 plies |
| wall clock | 3,705.7 s — 62 min, ~57.9 s/game at `workers=1` |
| integrity | **no §A abort**; `same_checkpoint: true`; isolation confirmed in the recorded readouts |

Futility test: the interval's upper bound is `0.5590`, not below `0.50`, so the gate did
not fire.

### Did the result match the prediction, and why?

**There was no numeric prediction to match**, and no pass bar to clear. The screen did
exactly its job: it exercised new selection code against a real model, produced no
integrity fault, and did not trip its one gate.

**The point estimate is adverse** — the candidate lost 28–36. But 64 games cannot resolve
an effect of this size: the Elo interval spans `−134` to `+41`, **compatible with both
harm and null/noise; the adverse point estimate favours harm but does not resolve it.**
That irresolution is precisely why the decisive match exists and why this screen was
designed futility-only.

The red/black split (12–20 versus 16–16) **does not establish colour-specific harm and is
non-decisive at this sample size**; the per-colour intervals overlap heavily.

### Consequence

The 800-game decisive match becomes **eligible**. Eligible is not a pass, and eligibility
is not obligation.

**Expectations should be modest.** The screen is adverse evidence, and promotion requires
an observed score rate around `0.535` against an observed `0.4375` here. But closing the
line on that point estimate would install a post-result gate the futility-only design
deliberately excluded, and would leave Candidate 2 unresolved after it passed every frozen
prerequisite. The decisive match exists to separate harm, null, and useful gain — which
this screen cannot.

Its projected cost is ~13 hours at this configuration, known before the screen ran. If
that cost is independently unacceptable, the correct record is **"decisive match declined
for resource cost,"** never "Candidate 2 failed."

## Candidate 2 — 800-game decisive match, RUN 2026-08-08

Authorized by `docs/superpowers/2026-08-07-candidate-2-decisive-authorization.md`,
countersigned `bill-osienski` 2026-08-07T16:02:15Z, executed from `2614795` with a clean
worktree. The full suite measured 2,795 passed / 0 failed immediately before launch; the
three targeted search-identity tests passed with exit 0, and shipped search remained
unchanged against `d5326a0`. Interval `[202608188, 202608988)`, with both earlier
competitive-readout intervals recorded as priors.

### Hypothesis

The frozen post-opening Hoeffding-LCB override improves playing strength relative to
post-opening visit argmax by enough to clear the preregistered research-promotion bar,
without convincing one-colour harm. Both agents use the same checkpoint, search,
simulation budget and opening sampling, so the only intended difference is the final
post-opening move readout when the override fires.

### Why the hypothesis was plausible — and why the prior was adverse

The preflight established that the rule was neither a no-op nor unbounded: it fired on
6.08% of the frozen post-opening population, across 35 of 64 games, and the median
challenger at an override had 87 completed visits. The rule therefore had enough reach to
affect games, and compared well-visited alternatives rather than only fringe children.

But reach was not merit. MCTS backups are correlated and adaptively sampled, so the
Hoeffding radius is a conservative ranking heuristic rather than a valid confidence
guarantee. The repository also carried adverse readout precedent, and the 64-game screen
lost 28–36. The authorization recorded that adverse prior explicitly and set modest
expectations rather than treating screen eligibility as evidence of strength.

### Predicted result and preregistered interpretation

The frozen scientific decision was categorical, not an adjustable forecast:

- **Research promotion** required the draw-inclusive 95% lower bound to exceed 50%, no
  colour's own 95% upper bound below 50%, zero integrity failures, and search-identity
  evidence. The per-colour clause is historical and was retired by the 2026-08-10 erratum.
- **Bar not met** meant close the readout line — no third formula, relaxed bar, larger
  match or post-result rescue.

Before the run, the authorization recorded an adverse prior from the 28–36 screen. An
informal forecast expected the promotion bar to fail, placed promotion below 10%, and
expected a neutral-to-negative point estimate. That forecast was not a gate and did not
alter the preregistered decision rule.

### Result — promotion bar NOT met

| | |
|---|---:|
| score rate | **0.5125**, CI95 `[0.4779, 0.5471]` |
| primary decision | lower bound `0.4779` is not above `0.50` — **NOT MET** |
| record | candidate 408 wins · control 388 wins · 4 state caps |
| Elo | **+8.7**, CI95 `[−15.3, +32.8]` |
| as red | 214–183 + 3 caps, 0.5387, CI95 `[0.4901, 0.5874]` |
| as black | 194–205 + 1 cap, 0.4863, CI95 `[0.4373, 0.5352]` |
| historical per-colour rule | not triggered — neither upper bound is below `0.50`; retrospectively non-diagnostic |
| decisive / caps | 796 decisive · 4 state caps (0.5%) · board-full 0 |
| average plies | 54.46 |
| wall clock | 49,530.1 s — 13.76 h, ~61.9 s/game at `workers=1` |
| integrity / identity | no §A abort · same checkpoint · targeted identity tests exit 0 |

The authoritative process exit was 0. All 800 replay sidecars were written and remain
unanalyzed.

### Did the result match the prediction, and why?

**The categorical forecast matched: the promotion bar was not met.** The measured effect
was much smaller than the roughly +24 Elo observed threshold the match was designed to
detect: +8.7 Elo, with a 95% interval spanning −15.3 to +32.8. The experiment therefore
did not establish that the override is stronger.

**The point forecast was too pessimistic.** The 64-game screen's adverse 28–36 did not
reproduce; the decisive point estimate was slightly positive rather than neutral-to-
negative. That is exactly the uncertainty the futility-only screen preserved for this
match to resolve. The result is compatible with a small benefit, a null, or a small harm,
but none clears the preregistered reliability bar. It is a null against that bar, not a
finding that the rule is harmful and not proof that its true effect is exactly zero.

The red/black point estimates differ, but neither colour triggered the frozen one-sided
harm rule. No colour-gap veto existed, and none is introduced after seeing the result.

### Consequence — competitive-readout line CLOSED

Per the frozen authorization, bar not met closes Candidate 2 and the competitive-readout
line. Do not run a third formula, relax the promotion or colour rules, enlarge or repeat
the match, or analyze the captured replays to rescue the candidate. Candidate 1's large
sampling contrast does not change this disposition, and A/B/C/D remain invariant because
the experiments changed no tree or root value.

No existing policy changes: no adoption into the product, the checkpoint-tournament
default, or policy 3. The finding is scoped to the frozen readout comparison at
`calib020_0001`, cold, 400 simulations, in the Python harness; it is not a checkpoint or
product-strength claim. Any next project begins separately as training-line discovery.

## Candidate 2 — 800-game decisive match, RUN 2026-08-08 — NULL, line closed

Authorized by `docs/superpowers/2026-08-07-candidate-2-decisive-authorization.md`,
countersigned `bill-osienski` 2026-08-07T16:02:15Z, executed from `2614795` with a clean
worktree. Suite measured green immediately before launch (2,795 passed / 0 failed);
search-identity trio passed `EXIT=0`. Interval `[202608188, 202608988)`, priors
`[202608060, 202608124)` and `[202608124, 202608188)`.

### Hypothesis

The frozen Hoeffding-LCB post-opening override produces a statistically significant
same-checkpoint strength gain at equal 400-simulation budget, against a control identical
to it in every respect except the override.

### Why it was plausible

The rule reached its screen having cleared every frozen prerequisite: a preflight
override rate of 6.08% — inside the band, with a median challenger of 87 completed visits
rather than fringe moves — and a mechanics screen whose one gate did not fire. Overriding
a visit leader on a conservative confidence comparison is a standard idea elsewhere, and
the readout was the last untouched axis after tree-local heuristics closed.

### Predicted result and preregistered interpretation

The bar was frozen before the run: **draw-inclusive score rate with a 95% lower bound
above 0.50**, i.e. an observed rate around `0.535` ≈ +24 Elo, with ~80% power at ~+35
Elo. The historical colour rule rejected only on a colour's own 95% **upper** bound below
0.50; the 2026-08-10 erratum later retired that rule. No colour-gap veto, no post-result additions.

The adverse prior was recorded in the authorization itself: the screen had lost 28–36,
and expectations were set to modest.

### Result — promotion bar NOT met

| | |
|---|---:|
| score rate | **0.5125**, CI95 `[0.4779, 0.5471]` |
| **lower bound** | **`0.4779` — not above `0.50`** |
| Elo | **+8.7**, CI95 `[−15.3, +32.8]` |
| record | 408–388 of 800 |
| as red | 214–183, 0.5387, CI95 `[0.4901, 0.5874]` |
| as black | 194–205, 0.4863, CI95 `[0.4373, 0.5352]` |
| decisive | 796 · state caps 4 (0.5%) · avg 54.5 plies |
| wall clock | 49,530 s = **13.76 h** (61.9 s/game, `workers=1`) |
| integrity | no §A abort; search identity `EXIT=0`; `same_checkpoint: true` |

The historical absolute per-colour rule did not trigger — neither upper bound (red
`0.5874`, black `0.5352`) falls below `0.50`. Per the erratum, that rule is non-diagnostic;
this match supports no candidate-specific colour conclusion.

### Did the result match the prediction, and why?

**Partly, and the distinction is worth keeping honest.**

The *decision-relevant* forecast was correct: the rule would not clear promotion, and it
did not. The *point* forecast was too pessimistic — the screen's 28–36 pointed
neutral-to-negative, while 800 games returned a slightly **positive** 0.5125.

**This is a NULL, not a harm finding.** The interval spans `−15` to `+33` Elo: the rule
is neither measurably better nor measurably worse than playing the visit leader. The
screen's adverse point estimate did not reproduce, which is exactly the irresolution that
made the screen futility-only and the decisive match necessary. Running it was right
despite the screen.

Read narrowly: evidence about a **readout rule** at fixed `calib020_0001`, cold, 400
simulations, in the Python harness. Not about the checkpoint, and not about the product.

### Frozen consequences

**Close the readout line.** Per the authorization's Afterward: no third readout formula,
no relaxed bar, no larger match, and no replay-driven rescue. The 800 sidecars are
archival. No policy or product change follows.

Candidate 2 was the line's last strength hypothesis. Closeout:
`docs/superpowers/2026-08-08-competitive-readout-closeout.md`.

## Training continuation — five-iteration cold-buffer test, RUN 2026-08-08 — REJECTED

Authorized by `docs/superpowers/2026-08-08-training-continuation-experiment-card.md`
and executed from `3c70fffe3d660451f0f151f6d769f25f1fd6edb5` with a clean worktree.
The parent was `calib020_0001` (SHA1
`209cf2d4fd24a48553d259dd71b4954867b9473e`); the frozen endpoint was
`checkpoints/alphazero-v2-cont5-from-calib020/model_iter_0005.safetensors`
(SHA1 `c8cac3971c483af3c94aee71e79fb0e157136c95`). Training and evaluation
exit files both recorded `0`. The 400-game evaluation consumed the half-open interval
`[202608988, 202609388)` recorded in
`docs/superpowers/2026-08-06-competitive-readout-seed-ledger.md`.

### Hypothesis

Five iterations of ordinary, calibration-free self-play training from the current-best
checkpoint could produce a large playing-strength gain without another calibration or
search intervention. The run loaded weights only, began with an empty replay buffer and
used game outcomes as the value target. Its frozen endpoint comprised 500 self-play games
and 800 training steps.

### Why the hypothesis was plausible

The previous calibration and search lines had exhausted their local mechanisms, while
the basic question of whether training still gained beyond the prior point had not been
answered directly. Ordinary self-play does not require deeper search to be treated as
truth, and a direct parent match avoided another unvalidated proxy-gate pipeline.

The dose was deliberately small enough to falsify cheaply but substantial per iteration:
100 games and 160 training steps for each of five iterations. The learning rate inherited
the documented parent-line value, but the run did **not** reproduce the parent's
post-opening calibration objective. That distinction, and the empty replay buffer, were
recorded before the run because either could affect the result.

### Predicted result and preregistered interpretation

The recorded forecast was **null or mildly negative**. A success required the candidate's
draw-inclusive 95% lower score bound to exceed 50%, no colour's own 95% upper bound below
50%, and clean provenance. The per-colour clause is historical and was retired by the
2026-08-10 erratum. Success would have authorized only a full retrain replication
from the parent with a new training seed and evaluation interval—not an extension to more
iterations.

Failure meant stop the exact recipe. The card deliberately omitted a 64-game screen and
A/B/C/D: the screen had little stopping value ahead of a cheap 400-game test, while the
probes were reserved for a checkpoint that first showed credible strength. Their omission
means this experiment does not measure behavioral regressions beyond the match and its
predeclared per-colour safety check.

### Result — clear regression; promotion bar NOT met

| | |
|---|---:|
| score rate | **0.31375**, CI95 `[0.2687, 0.3588]` |
| primary decision | lower bound `0.2687` is not above `0.50` — **NOT MET** |
| record | candidate 122 wins · parent 271 wins · 7 state caps |
| Elo | **−136.0**, CI95 `[−173.9, −100.9]` |
| as red | 66–131 + 3 caps, 0.3375, CI95 `[0.2725, 0.4025]` |
| as black | 56–140 + 4 caps, 0.2900, CI95 `[0.2279, 0.3521]` |
| historical per-colour rule | mechanically fired for both colours — upper bounds `0.4025` and `0.3521` are below the rule's absolute `0.50`; retrospectively non-diagnostic of independent colour harm |
| termination | 393 decisive · 7 state caps (1.75%) · board-full 0 |
| average plies | 62.78 |
| decisive red win rate | 0.5242 |
| training wall clock | 1 h 45 m 55 s at `n_workers=10` |
| evaluation wall clock | 3 h 11 m, ~28.7 s/game at `workers=4` |
| integrity | both exit files `0` · clean worktree · endpoint provenance gate passed |

The state-cap rate and absence of board-full endings give no indication that the loss is a
termination artifact. The loss appears in both raw colour assignments, but those views are
not independent rejections: without an independently measured board-colour null, this match
cannot separate candidate strength from colour advantage. The aggregate result is decisive.

### Did the result match the prediction, and why?

**It matched the predicted direction but not the predicted magnitude.** The candidate was
expected to be null or mildly negative; it was instead a clear regression of about 136
Elo, with the entire 95% Elo interval below zero and the entire score interval below 50%.
Calling the forecast simply “correct” would over-credit it: the preregistered expectation
was materially too optimistic about how much strength the recipe would retain.

The experiment does not identify a single cause. It simultaneously started from a cold
replay buffer and removed the post-opening calibration objective used in the parent's
lineage. The result is therefore compatible with damage from the cold start, objective
removal, or their interaction; no one of those explanations was isolated. Iterations 1–4
were neither probed nor strength-tested, so no claim is made about whether the trajectory
briefly improved before the frozen endpoint.

### Consequence — reject this recipe, not all training

Keep `calib020_0001` as current best. Do not evaluate iterations 1–4, extend this run to
10 or 20 iterations, adjust its learning rate or training dose, add match games, or run
the success-only replication. Those are nearby rescues of a decisively failed recipe,
not new hypotheses.

This result does **not** prove that checkpoint strength is globally plateaued or that all
training is futile. A future training project must name a genuinely different mechanism
and explain why it addresses this failure—for example, preservation of replay state or
parent behavior—then receive its own cheap falsification design. It may not be framed as
a top-up or parameter rescue of this run.

## Parent-replay bootstrap — warm-buffer continuation, RUN 2026-08-09 — REJECTED

Authorized by `docs/superpowers/2026-08-08-parent-replay-bootstrap-experiment-card.md`,
countersigned `bill-osienski` at `2026-08-09T14:45:54Z` on authorization basis `0a55edd`,
and executed from `bcf62e2187e5da5603eb8a8cc1090598e0645d94` with a clean worktree. Parent
`calib020_0001` (SHA1 `209cf2d4fd24a48553d259dd71b4954867b9473e`); frozen endpoint
`checkpoints/alphazero-v2-warm5-from-calib020/model_iter_0005.safetensors` (SHA1
`643b5464e779697028432664832ff997b7d0c75e`). Both exit files recorded `0`. The 400-game
evaluation consumed `[202609388, 202609788)`.

### Hypothesis

The `−136` Elo regression of the cold-buffer continuation was caused by the **empty replay
buffer**, not by dropping the parent's calibration objective and not by continuation itself.
Filling the buffer with parent-generated experience before the first optimizer step should
remove most of the collapse.

### Why the hypothesis was plausible

The rejected run changed two things together — empty buffer **and** no calibration objective —
so its own closeout named "preservation of replay state or parent behavior" as the acceptable
successor shape. Warm-starting is the smallest such mechanism: it changes what the optimizer
sees at step 1 and leaves dose, learning rate, iteration count, endpoint and match size
exactly as they were. It was also cheap to falsify — under a day, one training run and one
match.

Implementation was one opt-in flag, `--replay-warmup-games`, default `0` and in-memory only:
no buffer persistence, no replay format, no migration tooling, no analyzer. It fails closed if
self-play returns fewer games than requested, and it consumes the existing master RNG stream in
place so iteration 0 continues from its advanced state.

### Predicted result and preregistered interpretation

The recorded forecast was **"recovers much of the `−136` Elo collapse but still fails the
promotion bar," with a stated 10–20% chance of a genuinely promotable gain.** Success required
the draw-inclusive 95% lower score bound above 50%, with no colour's own 95% upper bound below
50%. The per-colour clause is historical and was retired by the 2026-08-10 erratum. The
frozen disposition decided all three outcomes in advance: **bar not met ⇒ close
ordinary continuation, warm buffer included**; bar met ⇒ replicate the whole run once with
fresh seeds before investing in durable buffer persistence; clear loss ⇒ close immediately.

### Result — substantial recovery, bar still not met

| | |
|---|---:|
| warmup | 500 parent games ⇒ **44,578 positions** before the first optimizer step, 4,333.8 s |
| score rate | **0.4325**, CI95 `[0.3843, 0.4807]` |
| primary decision | lower bound `0.3843` is not above `0.50` — **NOT MET** (whole interval below the bar) |
| record | candidate 170 wins · parent 224 wins · 6 state caps |
| Elo | **−47.2**, CI95 `[−81.9, −13.4]` |
| as red | 97–97 + 6 caps, 0.5000, CI95 `[0.4318, 0.5682]` — absolute rule does not fire; no independent colour-neutrality conclusion |
| as black | 73–127 + 0 caps, 0.3650, CI95 `[0.2983, 0.4317]` — absolute rule mechanically fires; no black-specific-harm conclusion |
| historical per-colour rule | fired for black under the preregistered absolute `0.50` null; retrospectively non-diagnostic, so the aggregate loss is the sole rejection basis |
| termination | 394 decisive · 6 state caps (1.5%) · board-full 0 |
| average plies | 61.29 |
| decisive red win rate | 0.5685 |
| training wall clock | 3 h 04 m at `n_workers=10` (of which warmup 1 h 12 m) |
| evaluation wall clock | 3 h 07 m, ~28.1 s/game at `workers=4` |
| integrity | both exit files `0` · clean worktree · provenance gate passed · match JSON records the execution commit |

Unlike the cold run, the raw deficit is visible only in the candidate-as-black cell, while
the candidate-as-red cell is exactly `0.5000`. That split cannot establish harm confined to
black: the same match supplies no independent board-colour null, and estimating one from its
pooled results makes adjusted red/black deficits equal by construction. The cap rate and
absence of board-full endings give no sign that the aggregate loss is a termination artifact.

### Did the result match the prediction, and why?

**The central forecast matched: substantial recovery without promotion.** That is the
decision-relevant claim and it was right — unlike the two preceding calls in this line.

Four limits on how far that can be pushed:

1. **`−136.0` → `−47.2` Elo is a descriptive comparison across two separate runs**, each
   against the same parent but with different training seeds and different evaluation
   intervals. It is **not a paired causal estimate**, and no confidence interval on the
   difference was computed or is available from this design.
2. **The 10–20% promotion probability cannot be declared "correct."** A single failure is
   consistent with almost any small probability; one draw does not validate a calibration.
3. **The result supports the parent-replay-bootstrap hypothesis without isolating the cold
   buffer causally.** This run also carried a fresh training seed and 500 additional
   parent-generated games; those moved together with the warm start by design, because the
   experiment was scoped to test a mechanism cheaply, not to decompose it.
4. **Recovery is not adoption.** A checkpoint still ~47 Elo below its parent is not a
   candidate for anything.

### Consequence — close ordinary continuation, including the replay bootstrap

Keep `calib020_0001` as current best. Per the frozen disposition, the bar-not-met branch closes
**ordinary continuation as a family**, warm buffer included: no 250 or 1,000 warmup games, no
ten- or twenty-iteration run, no dose or learning-rate change, no evaluation of iterations 1–4,
and no replication — replication was authorized only by a win. The `--replay-warmup-games` flag
remains in the trainer, default off and tested; it is working infrastructure, not an open line.

This does **not** prove that all training is futile or that the checkpoint has hit a capacity
ceiling. It does establish that two ordinary five-iteration continuations from this parent —
one cold, one warm — both land below it, and that the bootstrap run's point estimate recovered
much of the gap without closing it.

## Frozen-parent opposition — fp6, RUN 2026-08-11/12 — BAR NOT MET, PARITY NOT RESOLVED, LINE CLOSED

Authorized and closed by
`docs/superpowers/2026-08-11-frozen-parent-training-experiment-card.md`; countersigned by
`bill-osienski` and executed once from
`13dd72f6261f60e5256f25af5ce1c851dbd821cf` with a clean worktree. Parent
`calib020_0001` had SHA1 `209cf2d4fd24a48553d259dd71b4954867b9473e`; the frozen endpoint
`checkpoints/alphazero-v2-fp6-from-calib020/model_iter_0005.safetensors` had SHA1
`22f8d2196140aff5b04fac0b68e1e5fa955d5ad4`. Training and evaluation both exited `0`.
The evaluation consumed `[202609788, 202610188)`. Iterations 1–4 were never inspected,
probed, evaluated or selected from.

### Hypothesis

The ordinary continuations degraded because the learner trained against its own drifting play:
as the learner weakened, its opposition weakened with it and errors could compound. Playing every
training game against the **frozen best parent**, while training only on learner-to-move positions,
would hold opposition quality fixed and preserve a strong reference throughout the five-iteration
continuation.

### Why the hypothesis was plausible

The cold continuation lost about 136 Elo and the parent-bootstrap successor recovered to about
47 Elo below the parent without closing the gap. That left self-play co-drift as a specific,
untested mechanism. Frozen-parent opposition changed the data-generating process rather than
rescuing ordinary continuation through a nearby dose: the opponent never updated, learner colour
was fixed 100/100 per iteration by game id, and parent-to-move rows never entered training.

The mechanism also directly matched the causal story. If a moving opponent was part of the
problem, pinning it was the smallest intervention that removed that movement while leaving the
optimizer, five-iteration horizon, 400-simulation search and warm replay dose unchanged. The
earlier fp5 attempt supplied no scientific evidence because its obsolete two-server transport
aborted before the first frozen-parent game; fp6 was the mechanism's first completed test.

### Prediction and preregistered interpretation

The forecast, written before implementation and preserved through the infrastructure abort, was:
**material recovery relative to `warm5`, but no promotion; aggregate score `0.47–0.51`, roughly
equal to or slightly weaker than the parent, with about a 10% chance of clearing the bar.** The
reason was that fixed opposition addressed co-drift but did not change terminal outcome targets,
policy-dominated optimization, the short horizon or the risk of specializing against one parent.

Success required the candidate's aggregate 95% lower score bound to exceed `0.50` over the frozen
400-game match. Per-colour results were descriptive only under the 2026-08-10 erratum. The frozen
disposition was decisive: **bar not met closes frozen-parent opposition**; only a bar-met result
could authorize a separate 0379 generalization match, and even then no immediate promotion.

### Protocol and measured dose

| parameter | frozen value / observed result |
|---|---|
| learner start / opponent / match baseline | `calib020_0001`, identical checkpoint for all three roles |
| warmup | 500 ordinary parent games; **46,523 positions**, 5,481.6 s |
| training | 5 iterations × 200 mixed-agent games × 160 optimizer steps; batch 64; buffer 100,000 |
| learner colour | exactly 100 red / 100 black per iteration, derived from game id |
| search | 400 simulations for learner and frozen parent |
| learner rows by iteration | **9,611 · 9,085 · 9,382 · 9,858 · 9,701** |
| final buffer | 94,160 / 100,000; no eviction |
| training seed | `20260813` |
| match | 400 games, equal 400-simulation agents, base seed `202609788` |

The 200-game dose assumption was confirmed: learner-only filtering yielded roughly 9,000–9,900
rows per iteration, the exposure it was chosen to preserve relative to `warm5`'s 100 ordinary
self-play games. Training took 5 h 59 m and evaluation took 3 h 08 m. Iteration 1 took 8,114.9 s
versus 1,703.6–2,324.4 s for iterations 2–5; this unexplained timing is an **infrastructure
observation only**, not evidence about strength.

### Result — promotion bar not met; parity unresolved

| | |
|---|---:|
| score rate | **0.46625**, CI95 `[0.4177, 0.5148]` |
| primary decision | lower bound `0.4177` is not above `0.50` — **BAR NOT MET** |
| record | candidate 184 wins · parent 211 wins · 5 state caps |
| Elo | **−23.5**, CI95 `[−57.7, +10.3]` |
| as red | 102–94 + 4 caps, 0.5200, CI95 `[0.4515, 0.5885]` |
| as black | 82–117 + 1 cap, 0.4125, CI95 `[0.3444, 0.4806]` |
| termination | 395 decisive · 5 state caps (1.25%) · board-full 0 |
| average plies | 61.78 |
| decisive red win rate | 0.5544 |
| integrity | both exits `0` · clean execution commit · provenance gate passed · match JSON commit matched training commit |

The candidate was **not shown stronger**. It was also **not shown weaker at this dose**: both the
score interval and Elo interval include parity, so it is **not statistically distinguishable
from the parent**. That does not establish equality. The raw colour split decides nothing; red
won 55.4% of decisive games, and this same match has no independent colour baseline capable of
separating board advantage from candidate interaction.

### Did the result match the prediction, and why?

**Approximately in direction and magnitude, not precisely.** The observed `0.46625` sits just
below the forecast band's `0.47` lower edge, while both intervals include parity. The central
expectation—substantial recovery without a statistically supported promotion—therefore described
the result reasonably well. A single non-pass does **not** validate the stated 10% probability;
one outcome cannot calibrate a small prior probability.

The descriptive sequence `cont5 0.31375` → `warm5 0.4325` → `fp6 0.46625` is consistent with
each mechanism recovering part of the point-estimate deficit, but it is **not a paired causal
estimate**. The runs used different mechanisms, training seeds and evaluation intervals, and no
confidence interval on their differences was designed or computed. fp6 supports no claim that
fixed opposition caused a quantified improvement over warm replay.

### Conclusion, reason and frozen consequences

**Conclusion: BAR NOT MET / PARITY NOT RESOLVED / FROZEN-PARENT LINE CLOSED. Keep
`calib020_0001`.** The reason is the preregistered aggregate decision rule: a candidate had to
show a lower confidence bound above `0.50`, and fp6's lower bound was `0.4177`. An unresolved
interval is not evidence of improvement and does not trigger the success-only generalization
match.

Do not change the dose or warmup, add a second opponent or pool, extend the run, inspect
iterations 1–4, buy resolution with a larger match, or run the 0379 match. Those actions were
excluded before the result or conditional on success; doing them now would be post-hoc rescue.
See do-not-repeat `#51`.

The scientific line closed negatively, but the infrastructure succeeded. The single Metal-owning
arbiter, dual-root seam, learner-only filtering and fail-closed transport completed roughly 1,500
games at 400 simulations without the driver abort that terminated fp5. This validates the shipped
one-owner architecture at full-run scale while leaving the two-server prohibition in `#50`
unchanged.

### Evidence of record

| artifact | SHA1 |
|---|---|
| candidate `model_iter_0005.safetensors` | `22f8d2196140aff5b04fac0b68e1e5fa955d5ad4` |
| match JSON | `544d6e335a773a3ac0e410d3fda7950e04c545dc` |
| match games JSONL | `c6c4b76ccbd3005072c5fd512578fec939ac22ae` |
| evaluation stdout | `6dd40a97c312d224b829e5ea33369b6c65deea20` |
| evaluation exit | `09d2af8dd22201dd8d48e5dcfcaed281ff9422c7` |
| candidate provenance | `f58b49db67d0bcfa079bc433298a6ce99b6e7c23` |

## Product-model alignment, Phase 3 — CANDIDATE SELECTED AND SERVED, 2026-08-22

**A PRODUCT result, not a research one.** It changes which bytes the web app serves. It does not
reopen the closed programme, revalidate any rejected line, or bear on `calib020_0001`'s standing.

**Result.** A preregistered 200-game match in the product's own stack (Node ORT,
`server/mcts.js`, and the shipped readout), at **hard**, `P = 100`, analysed once by the frozen
§6 analyser:
**`ACCEPTED` / `CANDIDATE_STRONGER`**. Mean pair score **`0.8475`**, bootstrap 95%
**`[0.7975, 0.8950]`**, `t` 95% **`[0.7962, 0.8988]`**, both methods agreeing; **76 win / 20 draw /
4 loss** over 100 pairs. `P` was fixed from timing alone, and committed, before the first game
(`6.8919` games/h `< 8.8` → `100`).

**Selection.** `c34b7ff3297c785a` is the served `DEFAULT_MODEL_ID` as of commit **`879b67c`**.
`1d64027db521a50f` is **retained byte-identical as the rollback**; reverting is that one line.

**MCTS OOM — remedied and validated.** Lazy child-state materialization replaced the eager
build-per-legal-move. Falsification **`4490` → `8`** copies against a `≤ 8` gate, failing on eager
and passing on lazy as required. Equivalence **92/92 cases, 0 mismatches**, exact on `visit_counts`
(values and order), `root_value`, `selected_move` and `progress`. Default-heap probe: completes at
Node's default heap, maximum observed **84.91 MiB ≤ 512 MiB**.

**Boundary — what this does NOT establish.** `medium` is `DEFAULT_DIFFICULTY` and **was never
measured**; §5.2 forbids presenting a hard-arm result as evidence about the default user
experience, and that arm is blocked on seeded RNG (`server/mcts.js` calls bare `Math.random()` in
`selectMove`'s stochastic branch). There is **no external strength anchor**, so the result is
relative to *these served bytes in this stack*. What produced the incumbent artifact remains
**unknown** and was deliberately not pursued.

**Current state.** The switch and its test repair are committed and **pushed** on
`codex/product-stack-comparison-spec` (`879b67c`, `a4efff8`). **Not merged, not deployed.**

### Evidence of record

| artifact | path |
|---|---|
| spec (rev 6, frozen) | `docs/superpowers/2026-08-14-product-stack-comparison-specification.md` |
| decision memo | `docs/superpowers/2026-08-13-product-model-alignment-decision-memo.md` |
| the match, and its single analysis | `tests/product_match/match/aca5ca2/` |
| `P` decision, committed pre-match | `tests/product_match/p_decision.json` |
| OOM remediation design | `docs/superpowers/2026-08-16-mcts-memory-remediation-design.md` |
| golden corpora, eager and lazy | `tests/mcts_golden/golden/{841df60,lazy_aad5796}/` |
| falsification, eager and lazy | `tests/mcts_golden/falsification/{eager_481f9bd,lazy_3189187}/` |
| §6 heap probe | `tests/mcts_golden/heap_probe/5e2b372/` |
| switch audit | `docs/superpowers/2026-08-22-default-model-switch.md` |
| Phase 2 parity PASS | `docs/superpowers/2026-08-13-phase2-parity-specification.md` |

### Deployment closeout, 2026-08-22 — WORKSTREAM CLOSED

Deployed from `main@fd59619` through the established launcher (`npm start` →
`scripts/startServer.js`; game on `:5500`, AI on `:3001`). Startup clean: no fallback, no
validation warning, no stderr. The launcher and the inference server report the served model
independently, because the launcher hands the child `MODEL_MANIFEST` rather than its own verdict:

```
Model id:   c34b7ff3297c785a
Provenance: source_checkpoint=checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors
```

Manifest, graph, sidecar hashes, the structural graph-to-sidecar binding and the application
tensor contract all passed the normal loader. That is not a separate check to report: `resolveModel`
and `assertSessionContract` are fail-closed with 13 fatal codes and no fallback, so a listening
server printing that banner **is** the evidence they passed.

Service verification, ten checks, all green — `/api/health` reporting `modelLoaded`,
`/api/model-info` naming the candidate graph and not the baseline, deterministic
`/api/analyze-position` returning a finite `root_value` with every candidate legal in the queried
position, `/api/evaluate` finite, and `:5500` serving the app. Re-run after a restart with
byte-identical values.

**Rollback is unchanged and independently validated.** `1d64027db521a50f` is retained in full and
the loader accepts it through the documented `MODEL_MANIFEST` path, so the rollback target is
known-good rather than assumed. Reverting remains one constant.

**Manifest hygiene — the durable lesson.** Both manifests carried temporal claims that had silently
become false: "NOT the served model", parity unmeasured, "never been played in the product stack",
`DEFAULT_MODEL_ID` unchanged, and — in the rollback manifest — a strength claim of "never measured
in the product stack" when it had been the match's baseline arm. **An identity manifest must not
encode mutable programme or deployment state:** those facts change while the bytes stay identical,
so the file rots with no signal and nothing to catch it. Both now assert identity and provenance
only and link the reviewed records instead of restating them. Selection is described as determined
by repository and deployment configuration rather than by any single constant, because
`MODEL_MANIFEST` is a supported override.

**Deployment changes no evidence boundary.** Serving the candidate creates no evidence about
`medium`, which is `DEFAULT_DIFFICULTY` and remains unmeasured, and there is still no external
anchor. Deployment is an operational fact, not a strength result.

**The product-model alignment workstream is CLOSED.** Its successors — a seeded-RNG `medium` arm,
and recovering the incumbent's provenance — are not open, not authorized, and each would need its
own scope.

## What got better vs worse

**Improved — A (black pre-drop):** targeted correction is **real**. The strongest A correction so far is **v4 teacher-retention**: mean **−0.305**, over **13.3%**, severe **6.7%** (from baseline mean +0.257 / over 50.0% / severe 43.3%). This is an A-only success, not a promotion candidate, because B/C/D failed.

**Worsened / unstable — C (old post-opening):** regresses under every v2/v3 approach. Crucially, **v3 and v3b share 5 severe C cases** — the same positions break regardless of the scalar weight:

- `game_000065_ply_021`
- `game_000309_ply_019`
- `game_000433_ply_029`
- `game_000505_ply_037`
- `game_000565_ply_033`

That overlap means **C is not random eval variance** — it's a stable fragile family that scalar calibration keeps damaging. It needs **direct retention of the current-best behavior** on those positions.

**Worsened / diffuse — D (red pre-drop):** v3/v3b share only **1** severe case:

- `red_loss_game_000728_predrop_ply_48_drop_50`

Low overlap ⇒ D is likely a **broader value-head drift** problem, not a handful of hard positions. A few hardcoded D rows probably won't fix it; it may need broader retention.

## Do-not-repeat (prevents going in circles)

1. **Uniform mixed-pool sampling only.** v2/v2b showed uniform sampling can't reliably separate correction from retention.
2. **Only increasing `retention_weight`.** v2b (2.0) helped some guardrails but weakened A and still failed. → no `retention_weight 3.0` sweep; we've already seen the tradeoff.
3. **Only lowering global calibration weight.** v3b (0.005) did not preserve guardrails and weakened correction. → **stop scalar weight sweeps.**
4. **Promotion matches before A/B/C/D all pass.** Every rejected branch failed gates clearly enough that a match would be wasted compute.
5. **More scalar-MSE-only rows as the main strategy.** The C/D failures show scalar row anchors aren't enough to hold the guardrails.
6. **Rerunning scalar-retention v3 with frozen BatchNorm as the fix.** The `v3-frozenBN-control` still passed A but failed B/C/D, with B/D worse than original v3. BN freezing is required for clean calibration mechanics, but it does not solve guardrail retention.
7. **Another v4 teacher-retention weight/schedule tweak before raw-NN candidate scoring.** v4 included the shared fragile C/D rows, but raw-NN teacher-retention still failed to preserve 400-sim MCTS gate behavior. Before changing teacher value/policy weights or schedule ratios, inspect whether v4 actually matched the raw teacher values on the shared C/D rows. *(Scoring done 2026-07-01 — see #8.)*
8. **Any further raw teacher-retention weight/schedule sweep.** The 2026-07-01 raw-NN focus-row diagnostic shows v4 **mostly matched the raw teacher values** on the shared C/D rows (e.g. `game_000369_ply_051` delta +0.1127 raw vs a severe MCTS gate) while the 400-sim gates still failed — the **objective**, not its weighting, is wrong for the gate. The next branch must target MCTS-root/root-behavior retention.
9. **"Root-value-only retention" as a new branch.** It has already been run: v2/v3 retention rows' `target_black_value` came from `probe_black_root_value` (`build_targeted_calibration_manifest.py:105,137`), i.e. BASE's own 400-sim MCTS root values — and failed B/C/D, including under frozen BN (v3F). Any v5+ proposal whose only value signal is the BASE root value is a v3 rerun. The new signal must be the root **visit distribution** (or deeper tree/path structure), not the root scalar.
10. **Any root-policy weight/schedule sweep, or a new retention design, before the v5 anchor-hold diagnosis.** *(DISCHARGED 2026-07-02 — the path diagnostic ran: anchors HELD, continuations drifted; see the v5 path-diagnostic entry.)* The rule's successor is #11.
11. **Any further root-position-level anchoring as the primary retention strategy.** The v5 path diagnostic proves the mechanism: v5 held its root anchors (dominant moves + visit shares) on the fragile C rows yet stayed severe, because the drift lives in the **searched continuation/child values** one-plus plies below the anchored roots (D top-child NN values +0.03→+0.80). Adding more root rows, sharper root targets, or heavier root weights cannot reach it. Retention designs must anchor **continuation/PV states** (or deeper tree structure), i.e. v6's shape.
12. **Another full-network v6/v7 row-engineering branch as the primary fix.** v6, v6b, v6c, and v7 all used cleaner/more targeted row designs and still failed at least one guardrail. The v7 drift map showed nonlocal value-surface movement even when the selected rows were sensible. Do not add more continuation/root/severe rows under full-network training before changing the training mechanics.
13. **Broad D root retention or sparse severe-D hard correction as a standalone fix.** v6c (30 D root value-only rows) and v7 (8 severe-D hard rows) both moved D in the right mean direction but still failed D and/or broke B/C. D row pressure alone is not enough under full-network training.
14. **Assuming value-head-only is a promotion just because B/C/D pass.** v8 proved value-head-only protects B/C/D, but A still failed. Value-head-only is the active training-mechanics hypothesis, not a promotion candidate until A/B/C/D all pass.
15. **Increasing A draw pressure under value-head-only as the next fix.** v8b raised A schedule mass from 2→3 and made A worse (mean +0.068 → +0.102, severe 20.0% → 26.7%) while raw-A output barely changed. Do not run A=4/A=5 before changing the mechanism.
16. **Broader partial-trunk unfreeze as the next move.** v9 unfreezing just the final residual block fixed A but broke B/C/D. Last-2/final-N unfreeze is expected to increase the same nonlocal guardrail drift unless paired with a new constraint/objective; do not run v9b last-2 as a simple extension.
17. **Another broad v10/v10b schedule-count sweep as the next move.** v10 was a near-pass, but v10b showed that increasing B/D pressure broadly fixes B at the cost of A/C/D. The knobs are coupled; do not keep sweeping tag counts from this family.
18. **Using v10b as the branch point.** v10b is worse than v10 on three gates. Do not build on v10b.
19. **B-only value-root clone surgery as the next fix.** v11 added value-only B root clones for the v10 B blockers and reduced B policy-CE exposure (`n_teacher_retention_drawn` 320→160), but B worsened and C/D failed. The v10 B blocker was not simply B root-policy CE or missing B value-only root preservation.
20. **Any further v10/v11 schedule or manifest-row variant as the next move.** v10 was the near-pass, v10b proved broad pressure is destabilizing, and v11 proved surgical B root-value cloning is insufficient. Existing root/continuation/root-value/severe-row levers are exhausted under final-block training. Do not run v11b, more B clones, more D pressure, or another tag-count variant as the next step.
21. **Treating root-only asymmetric guardrails as sufficient for C/D.** v12 proved the one-sided hinge objective is useful and fixed B, but C still had a broad severe repeat-offender cluster and D retained a severe row plus high non-severe over rows. Do not keep testing root-only guardrail variants as the next step; C/D need continuation guardrails or a stronger constraint.
22. **Jumping directly to gradient projection before testing continuation guardrails.** v12 was too close and too diagnostic: B passed, A nearly passed, and the remaining failures were concentrated in C/D families already known to require searched-continuation coverage. v13 gradient-conflict handling is reserved for after v12b if C/D still break despite continuation guardrails.

23. **Another v12b schedule/margin/row-coverage tweak as the next move.** v12b already applied the one-sided hinge to C/D searched-continuation states and still failed B/C/D, with C stable repeat offenders and D broad severe drift. Do not run v12c with heavier continuation weights, added B continuation guardrails, more C/D rows, or a margin sweep as the primary next branch. The next credible step is gradient-conflict handling/projection.
24. **Rerunning projection variants without flattened projection telemetry.** The first v13 run produced valid gates but dropped `calib_projection_*` from `model_iter_*.json`, making conflict-rate/removed-norm interpretation impossible. Any future projection branch must persist telemetry in both the nested sidecar and flattened per-iteration row before being used for decision-making.
25. **Lowering guardrail margin as the next projection fix.** v13b margin 0.05 increased `guardrail_active_frac` and projection conflict rate but weakened A and did not improve C/D severe. Do not run margin 0.025 or another global hinge-activation sweep.
26. **Promoting a near-pass with one D severe row.** v13c passed A/B/C and missed D by one barely-severe row, but D's pass bar is explicitly `severe=0.0%`. Do not change promotion rules after seeing a near-pass result. One tightly scoped cleanup run was acceptable; promotion still requires all gates.
27. **More v13 projection cleanup after v13d.** v13d was the tightly scoped cleanup run and it failed broadly: B/C/D all failed and D severe rose to 20.0%. Do not keep tuning `red_predrop_guardrail_retention`, projection strength, margin, or tag schedule inside the v13 family. The projection/cleanup line is closed unless a new written design changes the mechanism.
28. **Treating v14 as a promotion candidate or an underfit result.** v14 improved A substantially (mean +0.257→+0.064, severe 43.3%→20.0%) and passed B/C/D by formal gates, but it still failed A by mean. Do not promote it, and do not classify it as no-move/underfit.
29. **Running width 64 immediately after v14.** Width is the underfit lever, but v14 did move A. The blocker was not lack of movement; it was that A remained positive while D's mean margin degraded. Projection over the adapter surface was the right next branch, not wider capacity.
30. **Treating v14b as promotable because B/C/D passed.** v14b is the best adapter result so far and B/C/D passed, but A still missed the formal mean gate (+0.026 > 0.0). No promotion match until A/B/C/D all pass.
31. **Redesigning or widening immediately after v14b before the strength-2 test.** v14b showed projection helps A and does not break formal guardrails. The next single-knob follow-up was v14c (`projection_strength=2.0`), not width 64, per-channel gates, margin changes, or objective changes.
32. **Running projection strength 3.0 after v14c.** v14c strength 2.0 regressed A versus v14b and narrowed C/D margins. Stronger projection is not the cleanup; stop projection-strength escalation.
33. **More than one A-pressure cleanup in the adapter line.** v14d (`black_predrop_correction=2→3` on top of v14b) was the deliberately narrow final cleanup because v14b missed only A mean by +0.026 while B/C/D passed. It failed: A still missed (mean +0.051) and D severe broke to 3.3%. Do not run A=4/A=5 or more guardrail-count tweaks inside the adapter line.
34. **Another argument-only adapter cleanup after v14d.** v14b remains the best adapter near-pass, but v14c (strength 2.0) and v14d (A draw 3) both failed. Do not run strength 3.0, width 64, per-channel gates, margin changes, or objective changes as an incremental tweak. Any continuation must be a new written design with a new mechanism and explicit acceptance/falsification criteria.
35. **Building v15 Phase 1 as depth-1 child rows or shallow PV/path rows.** Phase 0 made the A excess look targetable at the selected-child level, but Phase 0.5 falsified the few-row/PV interpretation. The positive raw mass that MCTS backs up is broad across the depth-2 frontier: median 196 depth-2 nodes per root are needed to cover 70% of positive raw mass, and the median PV share of positive raw mass is only 0.335%. Do not build `black_predrop_continuation_correction` as child/PV rows.
36. **Treating the A overvalue as another raw-root/value-adapter cleanup.** Raw A is already non-positive at BASE and more negative under v14b, while MCTS re-amplifies it from policy-selected frontier states. Do not run more A draw pressure, width/capacity tweaks, projection tweaks, or root raw hard-value rows. The next value-calibration branch must target the frontier distribution explicitly, or the work should redirect to search/prior behavior.
37. **Building v16 as policy-selected frontier hard-value correction.** The post-v15 budget/trajectory diagnostics falsified the remaining value-calibration interpretation. BASE A collapses with more search (400/1600/6400 mean +0.2570 → +0.0626 → −0.0451; gate-over 50.0% → 30.0% → 10.0%; severe 43.3% → 6.7% → 3.3%), and the trajectory check showed the selected predrop ply was inflated by 400-sim search while the drop ply was not. Do not train thousands of depth-2 `hard_value=-0.35` rows; that would gate-fit shallow-search noise.
38. **Treating v16 as another calibration/training branch.** v16 is reserved for search reliability only. No trainer/network/manifest/value-adapter changes under the v16 name.
39. **Using c_puct to fix or gate-pass A.** The v16 c_puct falsification diagnostic reproduced the gate at c_puct=1.5 and then showed lowering c_puct worsens A (mean +0.2570→+0.3778; over 50.0%→60.0%; severe 43.3%→50.0%) by increasing top-child visit share and top-child visited children. Raising c_puct is also disallowed as an A fix because it would reduce the metric by spreading visits onto inferior root moves, not by fixing the depth-2 frontier. c_puct is closed; the only allowed next search-code lever is opt-in FPU with default `0.0` preserving current behavior exactly.
40. **Treating selected-A FPU success as adoption.** The v16 FPU selected-A diagnostic is the first positive search-mechanism result, but it is still on a biased set selected by the flawed 400-sim A statistic. Do not adopt FPU into gates, self-play, or promotion rules based on selected-A results alone. Validate on an unbiased/non-selected sample first, then B/C/D under the same setting, then a head-to-head strength evaluation before any self-play adoption.
41. **Using absolute `fpu_value=−0.20` as a general 400-sim setting, or advancing it to B/C/D or a strength match.** v16a rejected it on the game-held-out sample: late new-collapse rate was 15.48% (13/84), with both late-red and late-black above the preregistered 10% stratum reject bar. The small mean value delta does not rescue a candidate that materially increases late root collapse.
42. **Tuning the replacement FPU candidate on the v16a held-out manifest.** The 324-position v16a sample has been observed and is consumed. Do not select absolute values, formulas, thresholds, or schedules against it. Diagnose the failure read-only, but design/tune any adaptive or parent-relative replacement using discovery data only; use a fresh or separately preregistered confirmatory holdout before adoption.
43. **Running or interpreting any nonzero policy-mass coefficient after the `r0` prerequisite failure, or relaxing the control-flip gate post hoc.** On the untouched production-v2 tuning split, `r0` (`FPU=Q_parent`) flipped the selected move to a lower-prior move on 11/40 controls = 27.5%, versus the frozen requirement `<10%`. The protocol explicitly makes `r0` qualification a prerequisite for the whole parent-relative family. Do not run the `{0.10,0.20,0.35,0.50,0.75}` grid, inspect the frozen split, or reuse these consumed splits to redesign the formula.
44. **Treating the v16 postmortem as a rescue of `Q_parent` or as positive evidence for a replacement coefficient.** The postmortem explains an already-final rejection: flips moved from mean prior rank 1.27 to 9.00 and concentrated in opening/red-to-move controls without a flip-specific top-share increase. It supports removing `Q_parent` from the neutral baseline; it does not validate `shipped_FPU-r*context`, select `r`, or authorize reuse of the consumed tuning/frozen positions.

45. **Any policy-mass FPU coefficient, in either formulation.** v16's parent-relative `Q_parent − r·sqrt(P_explored)` died at its own `r=0` prerequisite (27.5% control flips vs a <10% gate). v17's baseline-preserving `−r·sqrt(P_explored)` ran the full preregistered grid on fresh evidence and failed §7.2 safety at **all five** coefficients, with control flip rates of 0.31–0.50 that do **not** improve as `r` shrinks. Two independent formulations, two independent corpora, same failure mode: the policy-mass FPU line is closed. Do not propose `r > 0.45`, a finer grid, a relaxed control gate, or a third formulation of the same idea.

46. **Rescuing v18 by implementing the cap anyway, relaxing the preflight, or reusing its evidence.** v18 had real selected-A reach (`0.5639`) but failed its decisive A-vs-matched-control selectivity gate (AUC `0.5089`, lower bound `0.39`), missed sign dominance (`0.78475 < 0.80`), and produced **zero complete target rows**; selector sizing failed even on the full 800-game universe. Do not edit `mcts.py`, run a positive cap, extend or interpolate the grid, weaken the exposure/sign/near-even/flip/sizing predicates, or reuse the consumed census/cohort as fresh evidence to rescue this formulation. A successor needs a genuinely new selective observable, a new written hypothesis and fresh preregistered confirmation. This null means selectivity was not established; do not overstate it as proof that no depth-2 effect exists.
47. **Progressive widening in any shape, and enlarging or re-labelling the atlas corpus to rescue it.** The convergence atlas ran its authorized pilot (24 games, `[20321000, 20321024)`, checkpoint `209cf2d4…`) and returned two stopping findings. **Progressive widening: both frozen shapes FAIL on retention alone** — depth-1 retention `0.6842` against the `0.90` floor, identically for `(C=4, α=0.5)` and `(C=13, α=0.3)`, over 12 stable-reference-eligible rows, with root retention a perfect `1.0`. Widening keeps every stable root move and drops roughly a third of the depth-1 replies stable deeper search requires. Do **not** add, tune or interpolate a third shape — §8 preregistered this check as authoritative on the pilot alone. The intervention rates are **not** the basis and must not be quoted as if they were: their denominators were 1 and 10 rows with 17 and 2 inconclusive. **Separately `PROJECTED_CAPACITY_NO_GO`:** `p_s = 1/24` stable-negative drove required `N` to 1800 against a frozen maximum of 400, and **12 of 24 rows had no stable 3,200/6,400 reference at all**. Do not enlarge the corpus, loosen the stable-reference criteria, or reuse either the retired `[20320000, 20320024)` rows or these. **Read-out A's selectivity and Read-out B's gate calibration are UNANSWERED, not failed** — an operational capacity failure, not proof the information is absent. The `remaining` median of 66 establishes controller *feasibility* only and justifies no prototype. Closeout: `docs/superpowers/2026-08-05-atlas-closeout.md`. **Tree-local search heuristics are closed; the candidate set was the last untried axis.** A high-budget distillation successor must not inherit "deeper is truth" — half these rows disagreed with themselves between 3,200 and 6,400.

48. **Rescuing the rejected five-iteration cold-buffer continuation with nearby endpoints, dose changes or more match games.** The frozen iteration-5 checkpoint lost clearly to its parent over 400 games: score `0.31375`, CI95 `[0.2687, 0.3588]`, about `−136` Elo. The preregistered absolute per-colour rule mechanically fired for red and black, but the 2026-08-10 erratum makes that non-diagnostic of independent colour harm; the aggregate loss alone closes the recipe. Do not evaluate iterations 1–4, extend the same run to 10 or 20 iterations, alter its learning rate or training steps, add games, or invoke the success-only replication. The result rejects the exact calibration-free, cold-buffer continuation recipe; it does not isolate whether the cold replay buffer, removal of the parent's calibration objective, or their interaction caused the regression, and it does not prove all training has plateaued. A successor must introduce and justify a genuinely different training mechanism rather than relabeling a top-up as discovery.


49. **Ordinary self-play continuation from `calib020_0001` in ANY dose, warm buffer included — the family is closed.** Two runs now: cold buffer `0.31375` / `−136.0` Elo, and the parent-replay bootstrap `0.4325`, CI95 `[0.3843, 0.4807]`, `−47.2` Elo, CI95 `[−81.9, −13.4]`. The bootstrap's whole aggregate score interval sits below `0.50`, which is sufficient to close the family. Its preregistered absolute per-colour rule mechanically fired for black (`0.3650`, upper `0.4317`) but, per the 2026-08-10 erratum, this does **not** establish black-specific harm or an independent second rejection; the red raw score of `0.5000` (upper `0.5682`) likewise does not establish colour-specific neutrality. Do **not** run a warmup-size grid (250 / 1,000 / anything), extend to ten or twenty iterations, change the dose or learning rate, evaluate or select iterations 1–4, add match games, or run the replication — replication was authorized only by a win. **Its central forecast (substantial recovery, no promotion) MATCHED**, so the design was sound and the answer is negative; that is a result, not a reason to retune. Read the improvement precisely: `−136.0`→`−47.2` is **descriptive across two separate runs** with different seeds and evaluation intervals, **not a paired causal estimate**, and the 10–20% promotion prior is **not validated by one failure**. The run **supports the parent-replay-bootstrap hypothesis but does not isolate the cold buffer causally** — new seed and 500 extra parent games moved with it. Keep `calib020_0001`. `--replay-warmup-games` stays in the trainer as tested, default-off infrastructure; its presence authorizes nothing. A successor must name a genuinely different training mechanism, explain why it addresses a ~47 Elo shortfall that warm-starting did not close, and bring its own cheap falsification design and an external strength anchor if it intends to become an ongoing programme.

50. **Two independent inference-server threads submitting concurrent work to the same Metal device — in any experiment, for any purpose.** The frozen-parent opponent run aborted with **exit 134** (SIGABRT) at the first line of iteration 1, on a **Metal driver assertion**, not a Python exception: `AGXG15XFamilyCommandBuffer … failed assertion 'A command encoder is already encoding to this command buffer'`. The 500-game warmup had just completed normally **because it runs on the ordinary single-network path**; the crash arrived the moment the second server began serving alongside the first. Do **not** respond with a mutex around submission, a retry, a worker-count reduction, a `flush_ms` change, or a second process — a process-aborting driver defect is not a contention-tuning problem, and a lock that merely serialises two encoders leaves the same two owners racing at every seam. **Any successor that needs two networks served concurrently must route both through ONE Metal-owning inference arbiter**, and must first pass a **newly authorized, tiny real-GPU feasibility smoke** before any warmup or training budget is spent. **The lesson about evidence, not just about Metal:** every pre-accelerated gate item passed honestly, but the transport tests drive both servers with *stub* evaluators that never touch the GPU, so nothing in a 2,863-test suite exercised two servers doing real accelerated work at once. A stub-level integration test is not evidence about a device-level contract. **This entry records an implementation failure and is evidence neither for nor against frozen-parent training** — the mechanism never generated a single training game. Card (terminated): `docs/superpowers/2026-08-10-frozen-parent-opponent-experiment-card.md`.

**#50 addendum (2026-08-11) — one-thread/two-network feasibility PASSED; the two-server prohibition STANDS.** A countersigned probe (`docs/superpowers/2026-08-10-single-arbiter-feasibility-card.md`, execution commit `4808f324`) ran two arms of 400 synchronous `infer()` calls each: control, one thread with one network; treatment, one thread alternating **two independent resident network instances**. Both exited `0` with per-call shape and finiteness checks, stable digests, and agreeing digests across evaluators (`12c4e65bf4adf338`). **So a single Metal-owning thread CAN hold two networks at this dose** — the arbiter design is worth building. **Nothing about the prohibition changes:** two independent servers submitting concurrently remain forbidden, and this probe did not test that, did not show the original race is fixed, and did not exercise request routing or per-model batching. Reproducibility is to the six-decimal digest quantization, not bit-for-bit; timings show no observed slowdown at this dose, from one sample per arm. `--frozen-opponent-checkpoint` stays blocked at startup.

**#50 addendum (2026-08-11) — the implemented single arbiter SURVIVED real device load; the two-server prohibition is UNCHANGED.** The refactored transport (one `InferenceServer` thread, `model_id` routing, `(worker_id, model_id)` response queues, per-model batching) ran its countersigned real-GPU smoke at execution commit `7fb7f70b`: **402 synchronous GPU calls** — 2 reference + 400 routed across two workers — serving **two different checkpoints** (`209cf2d4…`, `8ad62ac4…`) with **exactly 1 observed inference thread**, per-model telemetry of **200 requests / 2,800 rows / 200 batches**, universal digest matches against an **independent** oracle (references computed by direct evaluator calls on the arbiter thread, never routed), both workers exiting `0`, and the server thread stopped. Exit `0` in 4.24 s. **So one Metal-owning thread can serve two networks in production shape.** Nothing about the prohibition changes: **two independent servers submitting concurrently remain forbidden**, and this smoke tested neither that, nor mixed-model grouping in one flush (`B` equalled the row cap, so every request was its own batch), nor **the composition with MCTS and real game generation** — `run_parallel_selfplay` over the dual-root `play_game` has been exercised only with stubs. `--frozen-opponent-checkpoint` stays blocked at startup; lifting it is a separate card gated on that composition smoke.

**#50 addendum (2026-08-11, second) — the single-owner arbiter survived REAL MCTS COMPOSITION; the two-server prohibition is UNCHANGED.** The first addendum recorded the arbiter passing synthetic load. The composition smoke (`docs/superpowers/2026-08-11-frozen-opponent-flag-lift-card.md`, execution commit `50324a45`) then drove the real path — `run_parallel_selfplay` → `self_play_worker` → dual-root `play_game` → MCTS → two `RemoteEvaluator`s → the arbiter — with real games: **4/4 generated, exactly one observed inference thread, learner-only buffered rows with an exact 2/2 colour split, server thread recorded and confirmed stopped, and `mixed_model_flushes = 38`.** That last figure matters most: **mixed-model grouping inside one flush has now actually executed on the device**, which neither the arbiter smoke (where `B` equalled the row cap, so every request was its own batch) nor any stub test could reach. Traffic was **ragged and partially coalesced** — 360 vs 320 requests, 2,231 vs 1,668 rows, batches below request counts — i.e. the real variable-size behaviour. **Nothing about the prohibition changes:** two independent servers remain forbidden, and this smoke tested neither that nor routing correctness (both instances loaded the SAME checkpoint, so a crossed response would be invisible — routing rests on the two-checkpoint arbiter smoke `9c5847a0`). `--frozen-opponent-checkpoint` remained blocked throughout and after; lifting it is a separate reviewed diff, and no training is authorized.

51. **Frozen-parent opposition from `calib020_0001`, in any dose or shape — the mechanism is CLOSED.** The `fp6` run (execution commit `13dd72f`, 2026-08-12) scored `0.46625`, CI95 `[0.4177, 0.5148]`, Elo `−23.5`, CI95 `[−57.7, +10.3]`. **The bar was not met.** Do **not** change the dose or warmup size, add a second opponent or an opponent pool, extend beyond five iterations, evaluate or select from iterations 1–4, run a larger match to "resolve" the parity, or run the **0379 generalization match** — that was conditional on a bar-met result and is not licensed by a non-pass. **Keep `calib020_0001`.** Read the result exactly: the candidate was **not shown stronger and not shown weaker at this dose** — both intervals include parity, so it is **not statistically distinguishable** from the parent, which is **not** the same as equal. The unresolved parity is **not** an invitation to buy resolution with more games; that would be a post-hoc power increase chosen after seeing the interval. The preregistered `0.47–0.51` / ~10% forecast matched **approximately**, and **one non-pass validates no probability**. The `cont5`→`warm5`→`fp6` improvement is **descriptive across separate runs, not paired or causal**. Iteration 1's anomalous 8,114.9 s is an **infrastructure observation** and carries no scientific weight. **`#50` is unchanged** — two independent inference servers on one Metal device remain forbidden; the single arbiter is what made this run possible and it held at full scale.

Also retired as *primary* strategies: global-weight sweeps, retention-weight sweeps, schedule-ratio sweeps, frozen-BN-as-the-fix reruns, raw-teacher weight/schedule tweaks, broad row-engineering, broader partial unfreeze, broad v10/v10b schedule-count sweeps, surgical B value-only root-clone manifest edits, projection-strength escalation, and adapter A-pressure cleanups. The active adapter-cleanup line is closed. The current default is to keep `calib020_0001`; any further calibration work requires a new written design.



## v14 adapter-projection cleanup status (2026-07-09)

### v14c — value-adapter projection strength 2.0 (RUN + REJECTED)

Checkpoint: `checkpoints/alphazero-v14c-value-adapter-projection-strength2-from-calib020-0001/model_iter_0001.safetensors`

Setup: same v14b value-adapter projection surface and v12b manifest/schedule/objective, but `--post-opening-calibration-projection-strength 2.0`. Telemetry/verifier were clean: `train_value_head_and_value_adapter=True`, `train_value_head_and_final_block=False`, `calib_projection_enabled=True`, `calib_projection_scope=value_head_and_value_adapter`, `calib_projection_strength=2.0`, `conflict_steps=49`, `conflict_rate=44.5%`, `removed_norm_avg=0.1329`, `value_adapter_gate=-0.001048`, `value_adapter_grad_norm=0.001371`, `calib_n_drawn_total=1440`, and the adapter verifier passed with only `value_head.*` + `value_adapter.*` changed.

Gate results:
- A black pre-drop: mean +0.060, over 30.0%, severe 20.0% — **FAIL**, and worse than v14b (mean +0.026, severe 16.7%).
- B goal-line: mean −0.261, over 5.6%, severe 0.0% — **PASS**.
- C old post-opening: mean +0.052, over 30.0%, severe 13.3% — **PASS**, but only at the severe cap.
- D red pre-drop: mean −0.056, over 26.7%, severe 0.0% — **PASS** by formal gate.

Decision: **REJECT / no promotion.** Projection strength 2.0 did not fix A and degraded the adapter-line shape versus v14b. Do **not** run strength 3.0. v14b remains the best adapter near-pass.

### v14d — one final narrow A-pressure cleanup (RUN + REJECTED)

Checkpoint: `checkpoints/alphazero-v14d-value-adapter-projection-a3-from-calib020-0001/model_iter_0001.safetensors`

Setup: same v14b value-adapter projection surface and v12b manifest/objective (`projection_strength=1.0`, bottleneck 32), but changed exactly one sampling knob: `black_predrop_correction=2→3`. All guardrail schedules, margin, projection mechanics, objective, and frozen surface were unchanged.

Telemetry/verifier:
- `train_value_head_and_value_adapter=True`, `train_value_head_and_final_block=False`.
- `calib_projection_enabled=True`, `calib_projection_scope=value_head_and_value_adapter`, `calib_projection_strength=1.0`.
- `conflict_steps=72`, `conflict_rate=54.5%`, `removed_norm_avg=0.0742`.
- `value_adapter_gate=-0.000585`, `value_adapter_grad_norm=0.001501`.
- `calib_n_drawn_total=1600`, `calib_n_drawn_per_step=10.0`, with A draws 480 and all guardrail draw counts unchanged from v14b.
- Adapter verifier passed: only `value_head.*` + `value_adapter.*` changed; frozen tensors byte-identical.

Gate results:
- A black pre-drop: mean +0.051, over 33.3%, severe 16.7% — **FAIL**; A did not cross mean ≤0.0.
- B goal-line: mean −0.299, over 5.6%, severe 0.0% — **PASS**.
- C old post-opening: mean +0.030, over 30.0%, severe 10.0% — **PASS**.
- D red pre-drop: mean −0.049, over 40.0%, severe 3.3% — **FAIL**; D severe must be 0.0%.

Decision: **REJECT / no promotion.** Extra A draw pressure did not push A through and broke D severe. This closes the argument-only v14 adapter cleanup line. Do not run A=4/A=5, projection strength 3.0, width 64, or another adapter schedule/projection tweak without a new written design.

## Next-step plan after v16a held-out rejection (2026-07-10)

**Default decision:** keep `calib020_0001` as current best. No v13/v14/v15 checkpoint earned a promotion match; v16/v16a were diagnostic-only and produced no model checkpoint.

**Closed lines:**
- v13 final-block projection cleanup is closed after v13d.
- v14 value-adapter cleanup is closed after v14c/v14d and the raw/MCTS drift diagnostic.
- v15 searched-continuation Phase 1 is closed before implementation: Phase 0.5 proved the optimism is broad depth-2 frontier mass, not a few child/PV states.
- The proposed v16 frontier value-correction design is rejected before build: budget/trajectory diagnostics show the A signal was selected shallow-search inflation, not a valid value-head target.
- c_puct is closed as an A/search-reliability fix.
- **Fixed absolute `fpu_value=−0.20` is closed as a general 400-sim setting.** It reached the selected-A mechanism but failed the preregistered held-out late-collapse gate.

**v16a result:** the frozen `0.0` vs `−0.20` test ran on 324 positions from 252 games, with 40/100/100/84 positions across opening/early-mid/midgame/late, exact red/black balance, and zero A-discovery game overlap. Overall new-collapse rate was 15/324 = 4.63%, just below the 5% overall reject line, but late play failed decisively: 13/84 = 15.48%, with late-red 6/42 = 14.29% and late-black 7/42 = 16.67%. Late collapsed roots rose from 17/84 to 28/84. The candidate also flipped the top move on 27.16% of positions and reduced effective children by 36.66 on average (107.58→70.92, about −34.1%). Central value movement stayed small (mean mover delta +0.0028; median absolute 0.0180; p95 absolute 0.2822), so the failure is search-shape concentration rather than broad mean-value drift.

**Immediate next step — read-only postmortem:** characterize the 15 new-collapse cases without changing the gate or selecting a replacement on this holdout. Report whether the cases cluster in 280-ply state-cap games; whether collapse coincides with top-move changes; baseline/candidate top share, effective children, mover value, root breadth, and opponent-reply count; and relevant legal-move/root-value context. This analysis may explain the failure mechanism but may not tune the next candidate.

**Next candidate design:** retire fixed absolute `−0.20`. A credible continuation should change the mechanism—most likely an adaptive or parent-relative FPU reduction—using the selected-A/discovery corpus and other non-v16a development data only. The v16a manifest is consumed and must not be used to choose the formula or its parameters. Do not run B/C/D or a strength match until a new candidate is frozen and passes an appropriate collateral screen.

**Decisive benchmark remains unchanged:** after collateral and B/C/D guardrail checks pass under an explicitly frozen ship-form search rule, run a same-checkpoint / same-400-sim / balanced-color head-to-head against FPU-off. Adoption requires a statistically significant strength gain; matching 6400-sim A values is mechanistic evidence, not success.

## Raw/MCTS drift diagnostic on A & D (RESOLVED 2026-07-09) — closes the v14 adapter line, sets v15

Ran the line-196 candidate: `eval_raw_nn_position_rows` (raw NN, no MCTS, eval-mode BN) on the A black-pre-drop + D red-pre-drop probe rows for BASE / v14b / v14d, juxtaposed with the 400-sim MCTS gate means. Raw CSV: `logs/eval/v15prep_raw_AD_drift_base_v14b_v14d.csv`.

**A — black pre-drop (want mean ≤ 0):**

| ckpt | raw mean | MCTS mean | search Δ (MCTS−raw) |
|---|---|---|---|
| BASE | −0.015 | +0.257 | +0.272 |
| v14b | −0.178 | +0.026 | +0.204 |
| v14d | −0.070 | +0.051 | +0.121 |

**D — red pre-drop (want mean ≤ 0, severe = 0):**

| ckpt | raw mean | MCTS mean | search Δ |
|---|---|---|---|
| BASE | +0.052 | −0.188 | −0.240 |
| v14b | +0.166 (raw severe 36.7%) | −0.047 | −0.213 |
| v14d | +0.051 | −0.049 | −0.100 |

**Conclusion — A is MCTS/search amplification, NOT raw-value undercorrection:**
- Raw A is already ≤0 at BASE (−0.015); the entire +0.257 gate overvalue is *added by the search* (+0.272). v14b over-corrected raw A to −0.178 yet the search still delivered +0.026 (re-amplified +0.204). The failed v14b/v14d gate is not raw capacity — the search backs up optimistic black continuations from the pre-drop roots.
- The search-Δ itself varies with training (v14b +0.204 → v14d +0.121), so the untapped lever is *reducing the search amplification*, not more raw correction (v14c/v14d proved that non-monotonic).
- D confirms the danger: raw D drift (v14b +0.166 / severe 36.7%) is *masked* by the search (MCTS −0.047) until the mask thins — exactly why "push A harder" broke D severe in v14d.

**Decision:** the v14 value-ADAPTER line did its job as a raw-surface experiment and is **CLOSED**. No width 64, per-channel gate, stronger projection, more A draws, or raw-adapter cleanup. **v15 = A searched-continuation correction on the v14b adapter+projection surface** — correct the child/PV states MCTS uses to produce the +0.20 backup, not the root raw value. Matches the line-291 tree/path hypothesis. First implementation step: a read-only A-continuation *concentration* diagnostic (is the +0.204 from a few child/PV states or broad?) to decide few-rows vs tree/path. Design: `docs/superpowers/specs/2026-07-09-targeted-value-calibration-v15-a-searched-continuation-correction-design.md`.

## v15 searched-continuation diagnostics (RESOLVED 2026-07-09) — closes v15 Phase 1

### Phase 0 — A-continuation concentration diagnostic

Design/plan: `docs/superpowers/specs/2026-07-09-targeted-value-calibration-v15-a-searched-continuation-correction-design.md` and `docs/superpowers/plans/2026-07-09-targeted-value-calibration-v15-phase0-concentration-diagnostic.md`.

Script: `scripts/GPU/alphazero/diagnose_v15_a_continuation_concentration.py`. Output: `logs/eval/v15prep_a_continuation_concentration.csv`.

Read: **concentrated at the selected-child level**. Of the 30 A roots, the 17 with `root_mcts_black_value > 0` were all concentrated. Globally, top-3 children carried 98.1% of all positive backup mass; a top-3-positive-per-root selection yielded 27 depth-1 child branches under the locked 90%/max-3 rule. Integrity checks passed: the sign invariant `sum(visit_share * -child.q_value) == root.q_value` held, and the 30-root MCTS mean reproduced the A gate mean (+0.2570).

Important finding: the depth-1 child raw values were already not the source of the backup. At the top-1 positive child of each overvaluing root, BASE raw black value averaged −0.087 while searched black value averaged +0.619. The +0.706 gap was search below the child, so depth-1 hard-value rows would only help if the value head generalized to deeper leaves.

### Phase 0.5 — selected-branch subtree diagnostic

Design/plan: `docs/superpowers/specs/2026-07-09-targeted-value-calibration-v15-phase05-selected-branch-subtree-diagnostic-design.md` and `docs/superpowers/plans/2026-07-09-targeted-value-calibration-v15-phase05-selected-branch-subtree-diagnostic.md`.

Script: `scripts/GPU/alphazero/diagnose_v15_a_selected_branch_subtrees.py`. Outputs:
- `logs/eval/v15prep_a_selected_branch_subtrees.csv`
- `logs/eval/v15prep_a_selected_branch_subtrees_by_depth_summary.csv`
- closure summaries: `logs/eval/v15prep_a_phase05_per_root_decomposition.csv`, `logs/eval/v15prep_a_phase05_depth2_coverage.csv`, `logs/eval/v15prep_a_phase05_pv_offpv_mass.csv`, `logs/eval/v15prep_a_phase05_closure_summary.txt`

Scope: reran deterministic BASE 400-sim MCTS on the 17 positive A roots, selected positive branches by cumulative positive share ≥0.90 / max 3, then walked every expanded descendant with `visit_count >= 1`, PV annotated. No depth cap, no manifest, no replay JSONs, no training. All 17 roots passed three fail-loud checks: tree reproduction against Phase 0 CSV, contribution invariant, and depth-1 cross-CSV perspective tie.

Result: **Case B — broad frontier optimism**.

- Nodes walked: 5,837.
- Raw-scored nodes: 5,745.
- Terminal nodes: 92.
- Depth-2 frontier: 4,443 nodes, 77.3% of leaf evaluations, mean raw_black(BASE) +0.793, 98.8% raw-positive.
- PV nodes: 139 / 5,745 leaf evaluations (2.4%), mean raw_black(BASE) −0.207.
- Off-PV mean raw_black(BASE): +0.618.
- Depth-2 nodes needed per root to cover 70% positive raw mass: p25 164, median 196, p75 205, max 238.
- PV share of positive raw mass: p25 0.176%, median 0.335%, p75 0.928%, max 23.53%.

Interpretation: Phase 0 proved the A excess is concentrated at which child branch MCTS selects, but Phase 0.5 proved the actual raw optimism being backed up is **not** concentrated on the child/PV path. It is broad across the policy-selected depth-2 frontier. A few-row child/PV continuation manifest would train the wrong states and miss nearly all positive raw mass.

**Decision:** v15 Phase 1 is **rejected before implementation**. Do not build depth-1 child rows, shallow PV/path rows, or semi-PV continuation rows. A temporary v16 frontier/tree-level value design was considered, but the later budget/trajectory diagnostics rejected it before build and redirected v16 to search reliability.

## Post-v15 search-budget + trajectory diagnostics (RESOLVED 2026-07-10) — closes A value calibration, sets v16

### A search-budget sweep on BASE

Ran the same A black-pre-drop probe on BASE / `calib020_0001` at 400, 1600, and 6400 sims. The corrected gate threshold is `over >= 0.25` and `severe >= 0.50`.

| sims | mean black value | gate over (>=0.25) | gate severe (>=0.50) |
|---:|---:|---:|---:|
| 400 | +0.2570 | 50.0% | 43.3% |
| 1600 | +0.0626 | 30.0% | 6.7% |
| 6400 | -0.0451 | 10.0% | 3.3% |

Result: the A gate signal collapses with more search. The earlier quick summary's `positive>0` column was not the gate overvalue metric; the true gate-over rate falls 50.0% -> 30.0% -> 10.0%.

### Predrop trajectory check

Checked five high-A loss games across the predrop->drop window. The 400-sim rerun reproduced the replay-stored `root_value` with max absolute diff 0.0000 across all checked plies, validating the pipeline and perspective. Comparing 6400 vs 400 showed the “sharp value drop” is mostly a predrop bump:

| position in window | mean (6400 - 400) |
|---|---:|
| predrop - 6 | -0.185 |
| predrop - 4 | -0.152 |
| predrop | -0.573 |
| drop ply | -0.001 |
| drop + 2 | +0.035 |
| drop + 4 | +0.012 |

At the predrop ply, four of five sampled high-A cases lost gate-over status at 6400 sims (`000281`, `000259`, `000127`, `000347`); `000611` remained overvalued and matches the full-30 result where only about 10% remain overvalued at 6400.

Interpretation: the original loss-analysis selection maximized `predrop_value - drop_value` using a 400-sim root statistic. Because 400-sim inflation is large at predrop and absent at the drop ply, the selection criterion picked the plies where shallow-search inflation was largest. The “drop” was mostly the selected 400-sim bump unwinding; at 6400 sims the curve is mostly a smooth decline or already non-overvalued.

**Decision:** value calibration against the A signal is not justified. The proposed v16 frontier hard-value correction is rejected before build. Do not train depth-2 frontier rows, child/PV rows, or any new A hard-value manifest from this signal. The remaining well-posed question is search reliability: can 400-sim MCTS be made to behave more like 6400 on these high-branching positions?

**v16:** the first v16 artifact was the read-only c_puct falsification sweep; c_puct is now closed as a fix. Continue only with a minimal opt-in FPU diagnostic if the search-reliability line proceeds.

## v16 c_puct falsification diagnostic (RESOLVED 2026-07-10) — c_puct is not the fix

Ran a read-only c_puct sweep on the 30 A black-pre-drop probe rows using BASE / `calib020_0001`. Integrity check passed: at `c_puct=1.5`, all 30 cases reproduced Phase 0 `root_mcts_black_value` within 1e-6. The sweep then varied only `MCTSConfig.c_puct`.

| c_puct | mean black value | gate over (>=0.25) | gate severe (>=0.50) | root children | top-child children | top-child visit share |
|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | +0.2570 | 50.0% | 43.3% | 80.4 | 134.7 | 0.474 |
| 1.0 | +0.3018 | 53.3% | 46.7% | 82.2 | 160.4 | 0.506 |
| 0.75 | +0.3027 | 53.3% | 40.0% | 82.0 | 175.8 | 0.552 |
| 0.5 | +0.3237 | 56.7% | 43.3% | 79.9 | 197.1 | 0.577 |
| 0.25 | +0.3778 | 60.0% | 50.0% | 79.1 | 232.9 | 0.642 |

Result: c_puct does not merely fail to help; lowering it actively worsens the A metric. The mean black value rose monotonically from +0.2570 to +0.3778 and moved farther away from the 6400-sim reference (mean -0.0451, over 10.0%, severe 3.3%).

Mechanism: lower c_puct increases root concentration (`top_child_visit_share` 0.474 -> 0.642), which sends more simulations into the selected root child. That child then scans more fresh opponent replies (`top_child_n_visited_children` 134.7 -> 232.9), exactly the depth-2 one-visit frontier mechanism implicated by Phase 0.5. Across all case/config rows, `top_child_n_visited_children` correlated with `root_mcts_black_value` at +0.943.

Important non-fix: raising c_puct must not be used to pass A. It can lower the metric by spreading root visits onto inferior black moves, not by repairing the frontier; that would optimize the gate statistic by degrading search.

**Decision:** c_puct is closed as a search-reliability fix. The next possible v16 code change is FPU/unvisited-child initialization, but only as an opt-in diagnostic with byte-identical default behavior and validation beyond the selected A set.

## v16 FPU selected-A diagnostic (RESOLVED 2026-07-10) — first positive search-mechanism result, not adoption

Added an opt-in FPU/unvisited-child initialization knob to MCTS with byte-identical default behavior: `MCTSConfig.fpu_value = 0.0`, and `_select_child` uses that value for unvisited-child q instead of hardcoded `0.0`. The default integrity check passed: `fpu_value=0.0` reproduced Phase 0 per-case root values within 1e-6, and the full suite passed.

Ran the selected A black-pre-drop probe rows at 400 sims across negative FPU values.

| fpu_value | mean black value | gate over (>=0.25) | gate severe (>=0.50) | opponent replies scanned | root moves |
|---:|---:|---:|---:|---:|---:|
| 0.0 | +0.2570 | 50.0% | 43.3% | 134.7 | 80.4 |
| -0.05 | +0.1725 | 40.0% | 26.7% | 102.4 | 67.9 |
| -0.10 | +0.0776 | 23.3% | 13.3% | 60.8 | 54.1 |
| -0.20 | -0.0344 | 6.7% | 6.7% | 24.5 | 33.5 |
| -0.35 | -0.1003 | 6.7% | 0.0% | 8.0 | 19.1 |
| -0.50 | -0.1261 | 3.3% | 0.0% | 6.2 | 10.4 |

Reference: 6400-sim BASE A was mean -0.0451, gate-over 10.0%, severe 3.3%.

Interpretation: FPU is the first tested knob that directly reaches the measured mechanism. Unlike c_puct, which increased top-child reply scanning, negative FPU collapses the opponent's first-touch reply scan and moves the 400-sim metric toward the 6400-sim reference. fpu≈−0.20 is the preferred next validation setting because it is closest to the 6400 mean while preserving more breadth than −0.35/−0.50.

Limitations:
- The sample is biased: these 30 roots were selected by the same 400-sim statistic now known to be inflated.
- FPU reaches the corrected value by narrowing first-touch exploration, while 6400 sims reaches it by deeper search; those are not automatically equivalent.
- The constant `fpu_value` form is diagnostic. A future ship-form may be parent-relative/reduction-based and must be evaluated separately.

**Decision at this stage:** selected-A FPU was promising but not adoptable. Freeze `−0.20` as the single candidate and proceed to a game-held-out v16a comparison against `0.0`; do not tune multiple values on the holdout. The completed v16a result below subsequently rejected the fixed absolute candidate.


## v16a stratified, game-held-out FPU validation (RESOLVED 2026-07-10) — absolute `−0.20` rejected

### Tooling and locked protocol

The v16a tooling generalized the FPU diagnostic without changing the trusted position-reconstruction path or legacy selected-A output. The selected-A path remained byte-identical through golden-output and old-vs-new fake-search comparisons; `fpu_value=0.0` remained the default MCTS behavior. Neutral mode enforced the frozen protocol `0.0` versus `−0.20`, used mover-perspective deltas as primary, and emitted paired search-shape summaries overall and by bucket / side / bucket×side.

The generated neutral manifest contained **324 positions from 252 held-out games**:

| bucket | positions |
|---|---:|
| opening | 40 |
| early-mid | 100 |
| midgame | 100 |
| late | 84 |

The sample was exactly balanced at **162 red / 162 black**, shared **zero games** with the selected-A discovery manifest, and reconstructed every retained row through `position_state`. The 19 `winner=null` games were valid 280-ply `state_cap` marathons and were retained as `game_result=unknown`; they contributed 47 of the 84 late positions and intentionally strengthened the stressed late-game screen.

### Overall paired result

| metric | `fpu=0.0` | `fpu=−0.20` / delta |
|---|---:|---:|
| positions | 324 | 324 |
| mean mover-value delta vs control | — | +0.0028 |
| median absolute mover-value delta | — | 0.0180 |
| p90 / p95 absolute mover-value delta | — | 0.2079 / 0.2822 |
| top-move flip rate | — | 27.16% |
| mean root entropy delta | — | −0.6106 |
| mean effective children | 107.58 | 70.92 (−36.66; about −34.1%) |
| mean visited root children | 132.49 | 90.81 (−41.68) |
| mean top-child reply count | 105.42 | 75.87 (−29.55) |
| stable-top reply-count delta | — | −24.56 |
| mean top-child visit share | 0.4154 | 0.4870 (+0.0716) |
| new / resolved collapses | — | 15 / 2 |
| collapsed-root rate | 7.72% | 11.73% |

The central value distribution did **not** show broad drift: the mean mover delta was nearly zero and the median absolute delta was only 0.018. The dominant effect was search narrowing and concentration.

### Preregistered gate result

The overall new-collapse rate was **15/324 = 4.63%**, narrowly below the 5% overall hard-reject threshold. The preregistered stratum rule nevertheless fired decisively: any stratum with `n ≥ 20` and new-collapse rate `≥10%` is an automatic reject.

| stratum | n | new collapses | rate | top-move flips | effective-children delta | top-share delta |
|---|---:|---:|---:|---:|---:|---:|
| opening | 40 | 0 | 0.00% | 35.00% | −20.39 | +0.0646 |
| early-mid | 100 | 2 | 2.00% | 29.00% | −45.09 | +0.0362 |
| midgame | 100 | 0 | 0.00% | 15.00% | −19.73 | +0.0227 |
| **late** | **84** | **13** | **15.48%** | **35.71%** | **−54.52** | **+0.1753** |
| **late-red** | **42** | **6** | **14.29%** | **38.10%** | **−80.09** | **+0.2099** |
| **late-black** | **42** | **7** | **16.67%** | **33.33%** | **−28.94** | **+0.1407** |

Late collapsed roots rose from **17/84 (20.24%)** under control to **28/84 (33.33%)** under `−0.20`: 13 new collapses, only 2 resolved, for a net increase of 11. Both late-side strata independently exceeded the reject threshold.

### Interpretation and decision

The selected-A mechanism finding remains valid: pessimistic FPU suppresses broad one-touch opponent-reply scanning, which c_puct could not reach. But a fixed absolute `−0.20` applies that suppression too aggressively in ordinary late-game positions, producing materially more near-single-line roots. Matching the 6400-sim A mean was therefore not evidence that the 400-sim search had become generally more reliable.

**Decision: REJECT absolute `MCTSConfig.fpu_value = −0.20` for general 400-sim search and self-play.** Do not run B/C/D or a head-to-head strength match with this candidate. Do not alter the preregistered threshold after observing the near-miss on the overall 5% line; the late-stratum reject is clear and independently repeated across red and black.

**Next:** perform a read-only case-level postmortem on the 15 new collapses, then design a new candidate on discovery data only. The likely direction is an adaptive or parent-relative FPU reduction rather than a fixed absolute value. The v16a manifest is consumed and must not be used to select that replacement.


## Severe-overlap findings (why the next step changes shape)

- **C — stable repeat offenders:** 5 of the severe cases repeat across v3/v3b (listed above). C should be treated as a fixed fragile family needing **direct retention of current-best behavior**, not as eval noise.
- **D — diffuse:** only 1 shared severe case across v3/v3b. D reads as **broad value-head drift**; unlikely to be solved by adding a few hard D rows.

## v4/v3-frozenBN severe-overlap follow-up

The post-v4 overlap check shows mixed failure structure:

- **B goal-line:** no severe-case overlap between v4 and `v3-frozenBN-control`. Treat B as a fragile guardrail, not a fixed-row problem yet.
- **C old post-opening:** 4 shared severe rows repeat across v4 and `v3-frozenBN-control`: `game_000065_ply_021`, `game_000369_ply_051`, `game_000505_ply_037`, `game_000619_ply_061`. This is the strongest stable fragile-family signal.
- **D red pre-drop:** only 1 shared severe row, `red_loss_game_000362_predrop_ply_52_drop_54`, but it shows strong value drift: baseline MCTS +0.198 → v4 MCTS +0.582 → `v3-frozenBN-control` MCTS +0.677. D remains mostly diffuse despite one common failure.

The shared fragile C/D rows were present in the v4 teacher-retention manifest, so v4 did **not** fail because the retention pool missed them. The stronger finding is that **raw-NN teacher retention did not preserve the 400-sim MCTS probe behavior**.

Key examples:
- `game_000065_ply_021`: teacher raw value +0.1105, base MCTS +0.480, v4 MCTS +0.758.
- `game_000369_ply_051`: teacher raw value −0.1389, base MCTS +0.334, v4 MCTS +0.765.
- `red_loss_game_000362_predrop_ply_52_drop_54`: teacher raw value −0.9379, base MCTS +0.198, v4 MCTS +0.582, `v3-frozenBN-control` MCTS +0.677.
- `game_000505_ply_037`: teacher raw value +0.9455 and base MCTS +0.856, so this row is already pro-black under the teacher/baseline and should not be treated as a clean retention failure.

Conclusion: before another branch, inspect raw-NN candidate values on these rows. If v4 matched raw teacher values but MCTS still drifted, the next design should move from raw-NN teacher retention to **MCTS-root retention** or another root-behavior retention objective.

## Resolved diagnostic after v4 and v3-frozenBN-control

Both completed follow-ups are rejects:

- `v3-frozenBN-control` passed A but failed B/C/D, proving that v3's guardrail failure was not primarily a train-mode BatchNorm artifact.
- v4 teacher-retention passed A strongly but failed B/C/D, even though the shared fragile C/D rows were present in the v4 manifest.

The next disciplined step is **not a new training branch**. First, score the shared C/D rows with raw NN-only evaluation for:

- `BASE = calib020_0001`
- `V4 = checkpoints/alphazero-v4-teacher-from-calib020-0001/model_iter_0001.safetensors`
- `V3F = checkpoints/alphazero-v3-frozenBN-control-from-calib020-0001/model_iter_0001.safetensors`

Focus rows:
- `game_000065_ply_021`
- `game_000369_ply_051`
- `game_000619_ply_061`
- `game_000505_ply_037` (diagnostic only; baseline/teacher already high)
- `red_loss_game_000362_predrop_ply_52_drop_54`

Decision value:
- If v4 matched the raw teacher values on these rows but MCTS still drifted, raw-NN teacher retention is the wrong objective for the gate and the next branch should use **MCTS-root/root-behavior retention**.
- If v4 did not match the raw teacher values, inspect loss weighting, masking, and gradient influence before designing a new branch.
- If raw-NN and MCTS disagree systematically on these rows, the gate must be treated as root-search behavior, not just value-head calibration.

Until this raw-NN candidate scoring is done, do **not** run another v3/v4 weight, policy-KL, or schedule sweep.

**→ RESOLVED 2026-07-01** — the scoring is done (next section). The first decision branch holds: **v4 matched the raw teacher values but the MCTS gate still drifted** ⇒ the next design is MCTS-root/root-behavior retention.

## Raw-NN focus-row diagnostic after v4/v3-frozenBN (2026-07-01)

Run via the new read-only `scripts/GPU/alphazero/eval_raw_nn_position_rows.py` CLI (raw NN forward only, no MCTS, eval-mode BatchNorm; plan `docs/superpowers/plans/2026-07-01-eval-raw-nn-position-rows-diagnostic.md`). The diagnostic scored BASE (`calib020_0001`), v4, and `v3-frozenBN-control` on the shared C/D severe rows. **BASE anchors reproduced exactly, validating reconstruction and eval-mode scoring.**

**Result: v4 mostly preserved the raw teacher values, while v3-frozenBN showed large raw drift on the key C rows.**

Key rows:

- `game_000369_ply_051`: BASE raw −0.1389, v4 raw −0.0262 (delta +0.1127, non-severe), v3F raw +0.6670 (delta +0.8059, severe). Yet v4's 400-sim MCTS gate was severe. **This is the cleanest evidence that raw retention held but MCTS/root behavior drifted.**
- `game_000065_ply_021`: BASE raw +0.1105, v4 raw +0.2697 (delta +0.1592, over but non-severe), v3F raw +0.5939 (delta +0.4834, severe).
- `red_loss_game_000362_predrop_ply_52_drop_54`: BASE raw stm −0.9379, v4 raw stm −0.8857 (delta +0.0522), v3F raw stm −0.8219 (delta +0.1160). v4 stayed close to the raw teacher despite failing the MCTS-root gate.
- `game_000619_ply_061` and `game_000505_ply_037` are already raw-severe under BASE, so they are useful diagnostics but not clean examples of newly-created raw drift.

**Conclusion:** v4 did not primarily fail because raw teacher-retention missed or ignored the fragile rows. It mostly preserved raw NN behavior, but that did not preserve the 400-sim MCTS gate behavior. The next branch should move to **MCTS-root/root-behavior retention** or an equivalent root-search preservation objective. Do **not** run another raw teacher-retention weight/schedule sweep as the next step.

## v5 design — MCTS-root-visit policy retention (LOCKED 2026-07-01 · RUN + REJECTED 2026-07-02, result below)

**Key correction that shaped v5 (code-verified):** root-value-only retention is **not new** — `build_targeted_calibration_manifest.py:105,137` set the v2/v3 retention rows' `target_black_value` from `probe_black_root_value`, i.e. **v2/v3 scalar retention already trained the raw value head toward BASE's 400-sim MCTS root values** — and failed B/C/D (twice, incl. frozen-BN control). So "root-value retention only" is v3 with a cleaner name: dead on arrival (do-not-repeat #9).

**The lineage that makes v5 the first genuinely new combination:**

| Branch | Value target | Policy target | Result |
|---|---|---|---|
| v3 / v3F | BASE MCTS-root value | none | failed B/C/D |
| v4 | BASE raw teacher value | BASE raw teacher priors | raw held, MCTS root still failed |
| **v5** | **BASE raw teacher value** | **BASE 400-sim MCTS root visit distribution** | **untested** |

The raw diagnostic showed v4 mostly held raw value, so pushing harder on raw value is not the missing signal; the missing signal is likely the **search-improved root policy**.

**Locked design shape:**

1. **New builder** `scripts/GPU/alphazero/build_mcts_root_retention_manifest.py` — input: source v4/v3-style stratified manifest + BASE checkpoint + gate MCTS config → output v5 manifest. Retention rows append `root_value_stm, root_black_value, root_visits_json, root_legal_moves_sha1, root_sims, root_base_checkpoint, root_seed, root_mcts_eval_batch_size, root_mcts_stall_flush_sims`; correction rows leave them blank.
2. **Root target generation** per retention row: reconstruct via `position_state` → BASE MCTS at 400 sims, `add_noise=False` → dense visit vector aligned to `state.legal_moves()` → normalize to sum 1.0 → dense JSON + legal-move sha1; `root_value_stm`/`root_black_value` stored as metadata. Builder asserts recomputed `root_black_value` ≈ the gate CSV's `probe_black_root_value` where available (the "did we match the gate setup?" check).
3. **Training semantics:** new `loss_mode = mcts_root_retention`, but **no new trainer loss path**: `calibration_pool.build_calibration_position()` parses the mode into the existing v4 teacher-retention tuple shape (`record.outcome` = raw teacher value stm, `record.visit_counts` = normalized BASE root visits, mask present) → the existing 14-tuple masked value + policy-CE path handles it. v2/v3/v4 paths byte-identical when unused.
4. **Value target = raw teacher value, not root value** — avoids repeating v3 and avoids amplifying MCTS root values into the raw head; the diagnostic says v4 held that anchor well, so keep it as a stabilizer.
5. **Policy target = dense normalized root visits, not top-k** — already aligned to legal moves, compatible with the v4 policy-CE machinery, zeros are informative at 400 sims, sha1 validation fits, no lossy top-k reconstruction.
6. **Gate-0 / smoke expectation:** do NOT expect v5 policy loss ≈ 0 at init (root visits are search-improved; raw priors should differ). The v5 smoke instead validates: builder target correctness (recomputed BASE root values match gate CSV values); training mechanics (value term starts ≈ 0 for raw teacher value; policy CE finite and mask-aligned; `legal_moves_sha1` matches; no NaN / shape mismatch / BN train-mode drift).

**Gate:** same A/B/C/D probes vs `calib020_0001`. No promotion unless all four pass.

**Important limitation (record in the v5 plan):** root-visit anchors constrain the candidate's raw policy **at the anchored root positions only**. If gate drift is caused by candidate value/prior changes deeper in the tree, root-visit retention may still fail. If v5 fails with raw value AND root policy held at the anchors, the next hypothesis becomes **tree-level/path-level retention**, not more anchored rows or stronger weights.

### v5 — MCTS-root-visit policy retention (RESULT, 2026-07-02)

Checkpoint: `checkpoints/alphazero-v5-mcts-root-from-calib020-0001/model_iter_0001.safetensors`

Setup: A hard-value correction rows unchanged; B/C/D retention rows used raw teacher value as the value anchor and BASE 400-sim MCTS root visit distribution as the masked policy-CE target. Training used `--freeze-batchnorm-stats`, global calibration weight 0.01, teacher value weight 1.0, root-policy CE weight 0.25, and the 2:1:2:1 tag schedule.

Gate results:
- A black pre-drop: PASS-ish / improved — mean −0.174, over 20.0%, severe 20.0% versus baseline mean +0.257, over 50.0%, severe 43.3%.
- B goal-line: FAIL — mean −0.288, over 16.7%, severe 5.6%. Pass requires severe 0.0% and over ≤ 11.1%.
- C old post-opening: FAIL — mean +0.074, over 40.0%, severe 30.0%. Mean passed, but over/severe failed.
- D red pre-drop: HARD FAIL — mean +0.046, over 40.0%, severe 36.7%. Pass requires severe 0.0% and mean ≤ 0.0%.

Decision: REJECT. No promotion match.

Lesson: Position-level root-visit policy retention did not preserve B/C/D after A correction. v5 tested the hypothesis that v4 failed because it preserved raw priors rather than search-improved root policy; that hypothesis is insufficient. The next step should not be a root-policy weight sweep. First diagnose whether v5 actually held the stored root-policy anchors on the retention rows. If held, the remaining failure points to deeper tree/path-level drift rather than root-row anchoring.

Run telemetry (provenance): `mode=mcts_root_retention`, draws_by_tag 320/160/320/160 (exact 2:1:2:1 over 160 steps), `n_teacher_retention_drawn=640`, `calib_policy_ce_avg_iter=3.83`, `calib_policy_kl_est_avg_iter=1.24` (vs v4's 0.19 — the root-visit target was genuinely non-trivial), `calib_value_term_avg_iter≈0.12`, `freeze_batchnorm_stats=true`.

### v5 path diagnostic — searched continuation drift (2026-07-02)

A gate-faithful path diagnostic was run on six representative failed v5 rows using the same synchronous `MCTS.search` path as the gates/builders. BASE root values matched the stored manifest values exactly, validating the diagnostic.

Findings:
- On C rows (`game_000433`, `game_000065`, `game_000565`), v5 preserved the same dominant root move and similar root visit share:
  - `game_000433`: BASE 19:9 share 0.9975, V5 19:9 share 0.9850.
  - `game_000065`: BASE 13:18 share 0.8800, V5 13:18 share 0.8650.
  - `game_000565`: BASE 21:5 share 1.0000, V5 21:5 share 0.9850.
  Despite this, v5 remained severe/overvalued, showing root-policy retention is insufficient.
- The child/continuation values shifted materially. Example: `game_000565` retained the same root move 21:5, but child NN value moved from BASE −0.4707 to V5 +0.4791.
- On D rows, BASE root visit distributions were diffuse, and v5 child NN values shifted strongly pro-black:
  - `red_loss_000780`: top child NN +0.0976 → +0.8258.
  - `red_loss_000362`: top child NN +0.0322 → +0.8013.
  - `red_loss_000176`: top child NN −0.1810 → +0.8613.

Conclusion: **v5 failed because root-level anchors do not constrain searched continuation values.** The next branch should be **v6 searched-continuation/PV retention**: add child/PV states from BASE MCTS under fragile rows and retain their raw teacher values, with policy retention only where distributions are sharp.

## v6/v7/v8 follow-up results (2026-07-03)

### v6 — searched-continuation/PV retention

Manifest: `logs/eval/targeted_calibration_v6_continuation_from_calib020_0001.csv` (381 rows: 50 hard-value, 78 old root-retention rows, 253 searched-continuation rows). Smoke passed with value-only continuations (`policy_ce=0.0`, 0 policy rows) and schedule draws `2:1:2:2` for A/B-cont/C-cont/D-cont.

Gate results vs `calib020_0001`:
- A black pre-drop: mean −0.110, over 20.0%, severe 10.0% — **PASS / improved**.
- B goal-line: mean −0.321, over 16.7%, severe 0.0% — **FAIL** on over cap.
- C old post-opening: mean +0.003, over 30.0%, severe 20.0% — **FAIL** on severe.
- D red pre-drop: mean +0.150, over 53.3%, severe 30.0% — **HARD FAIL**.

Diagnostic: every failed B/C/D root had continuation rows, so coverage was not the failure. Raw-NN diagnostics showed D root raw values remained severe or drifted upward. Continuation-only was too indirect for D.

### v6b — D root + continuation hybrid

Same v6 manifest, but training also scheduled `red_predrop_retention=1`, reactivating D root policy/root-visit retention. Telemetry confirmed `n_teacher_retention_drawn=160`, `policy_ce=3.809`, `policy_kl_est=1.297`.

Gate results:
- A: mean −0.308, over 3.3%, severe 3.3% — **PASS**.
- B: mean −0.240, over 16.7%, severe 5.6% — **FAIL**.
- C: mean +0.118, over 56.7%, severe 23.3% — **HARD FAIL**.
- D: mean −0.009, over 40.0%, severe 26.7% — **HARD FAIL**.

Lesson: D root policy retention slightly improved D versus v6 but broke B/C badly. Do not reintroduce root policy CE/KL as a mixed guardrail strategy.

### v6c — D root value-only + continuation

Manifest: `logs/eval/targeted_calibration_v6c_d_root_value_only_from_calib020_0001.csv` (411 rows: v6 + 30 depth-0 `red_predrop_root_value_retention` rows). Validation proved D root clones were value-only: `teacher_value` populated, `target_black_value` blank, `teacher_policy_json`/`root_visits_json` blank, `continuation_depth=0`, `continuation_source=root_value`. Smoke passed with `policy_ce=0.0`, 0 policy rows, and schedule `2:1:2:1:2`.

Gate results:
- A: mean +0.006, over 30.0%, severe 23.3% — **FAIL / improved**.
- B: mean −0.195, over 16.7%, severe 11.1% — **FAIL**.
- C: mean −0.007, over 36.7%, severe 16.7% — **FAIL**.
- D: mean +0.032, over 33.3%, severe 13.3% — **FAIL**.

Lesson: value-only D root anchoring is less toxic than policy retention, but still too broad and still interferes with B/C.

### v7 — sparse severe-D hard correction

Manifest-only branch using `logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv` (419 rows = v6c 411 + 8 hard-value rows). The 8 rows were selected by BASE raw severe-overvalue among D roots and assigned `target_black_value=-0.35` under tag `red_predrop_severe_root_correction`. Pool smoke passed: draws `2:1:2:1:2`, `policy_mask_sum=0.0`, no policy rows.

Selected severe-D rows: `red_loss_game_000752`, `000438`, `000362`, `000616`, `000408`, `000176`, `000456`, `000432` (BASE raw black approximately +0.62 to +0.97).

Gate results:
- A: mean −0.065, over 26.7%, severe 13.3% — **PASS / improved**.
- B: mean −0.290, over 16.7%, severe 5.6% — **FAIL**.
- C: mean +0.002, over 30.0%, severe 20.0% — **FAIL**.
- D: mean +0.034, over 40.0%, severe 23.3% — **HARD FAIL**.

Drift map: A and D means moved down, but D remained unstable (`up_0.25=12`, `down_0.25=12`) and B goal-line raw values moved upward on 9/18 rows (`mean_delta=+0.2015`). The problem was not just wrong row selection; full-network training moved the value surface nonlocally.

### v8 — value-head-only using v7 manifest

Implementation: `--train-value-head-only` skips the single `opt_main.update(main_module, main_grads)` call while always applying `opt_value.update(network.value_head, value_grads)`. Verifier CLI proved the trained checkpoint changed only the four `value_head.*` tensors; all 88 non-value-head tensors were byte-identical to BASE. Training telemetry: `train_value_head_only=True`, `freeze_batchnorm_stats=True`, `calib_n_drawn_total=1280`, `calib_n_drawn_per_step=8.0`, `policy_ce=0.0`, `n_teacher_retention_drawn=0`.

Gate results:
- A: mean +0.068, over 33.3%, severe 20.0% — **FAIL / improved but undercorrected**.
- B: mean −0.276, over 11.1%, severe 0.0% — **PASS**.
- C: mean +0.024, over 23.3%, severe 10.0% — **PASS**.
- D: mean −0.056, over 36.7%, severe 0.0% — **PASS**.

Decision: **Reject for promotion** because A failed, but this is the most informative positive result of the line: value-head-only protected B/C/D, supporting the hypothesis that full-network/trunk drift caused the v6/v7 guardrail failures.

### v8b — value-head-only, higher A draw pressure (RUN + REJECTED 2026-07-03)

Same v7 manifest + value-head-only mechanics, A draw pressure raised 2→3 (`black_predrop_correction=3,goal_line_continuation_retention=1,old_post_opening_continuation_retention=2,red_predrop_severe_root_correction=1,red_predrop_continuation_retention=2`). Telemetry/verifier were clean: `train_value_head_only=True`, `freeze_batchnorm_stats=True`, `calib_n_drawn_total=1440`, `calib_n_drawn_per_step=9.0`, `policy_ce=0.0`, `n_teacher_retention_drawn=0`, and only the four `value_head.*` tensors changed.

Gate results:
- A: mean +0.102, over 33.3%, severe 26.7% — **FAIL / worse than v8**.
- B: mean −0.286, over 5.6%, severe 0.0% — **PASS**.
- C: mean +0.086, over 33.3%, severe 13.3% — **PASS but worse than v8**.
- D: mean −0.096, over 26.7%, severe 0.0% — **PASS**.

Decision: **REJECT.** Higher A draw pressure did not help A and made A/C worse. This is not a simple "more A mass" problem.

### v8/v8b raw-A diagnostic (why value-head-only can't fix A)

On the 50 `black_predrop_correction` rows, **raw** value-head output barely moved:

| | raw mean | Δ vs BASE | severe raw overvalue |
|---|---|---|---|
| BASE | −0.2469 | — | 20.0% |
| v8 | −0.2533 | −0.0064 | 14.0% |
| v8b | −0.2433 | +0.0035 | 16.0% |

A did **not** fail because MCTS amplified an already-corrected raw value — the raw values themselves scarcely changed. A failed because value-head-only cannot substantially move the worst A raw values with the trunk frozen: `value_head` is a shallow MLP readout (`fc1→fc2`, no conv/BN) on frozen features.

**Conclusion:** v8 proved full-network drift was the main cause of B/C/D breakage (value-head-only preserved B/C/D). But value-head-only is too constrained to fix A. Next hypothesis is **partial unfreeze**: value head + the smallest late representation slice, starting with the final encoder/residual block.

## v9 — value head + final residual block partial unfreeze (RUN + REJECTED 2026-07-03)

Design spec: `docs/superpowers/specs/2026-07-03-targeted-value-calibration-v9-value-head-and-final-block-design.md`; implementation plan: `docs/superpowers/plans/2026-07-03-targeted-value-calibration-v9-value-head-and-final-block.md`.

Setup: same v7 manifest, v8 schedule `black_predrop_correction=2,goal_line_continuation_retention=1,old_post_opening_continuation_retention=2,red_predrop_severe_root_correction=1,red_predrop_continuation_retention=2`, global weight 0.01, `--freeze-batchnorm-stats`, and `--train-value-head-and-final-block`.

Implementation/acceptance proof: v9 skipped the whole-trunk `opt_main.update`, applied exactly one `opt_main.update(network.encoder.blocks[last], main_grads["encoder"]["blocks"][last])`, and always applied `opt_value.update(network.value_head, value_grads)`. Telemetry was clean: `train_value_head_and_final_block=True`, `unfrozen_block_index=5`, `freeze_batchnorm_stats=True`, `calib_n_drawn_total=1280`, `calib_n_drawn_per_step=8.0`, `policy_ce=0.0`, and `n_teacher_retention_drawn=0`. The strict tensor-diff verifier passed: only the 4 `value_head.*` tensors plus the 8 trainable tensors under `encoder.blocks.5.*` changed; all frozen tensors and all BN running stats stayed byte-identical.

Gate results:
- A black pre-drop: mean −0.089, over 30.0%, severe 16.7% — **PASS**.
- B goal-line: mean −0.238, over 22.2%, severe 11.1% — **FAIL**.
- C old post-opening: mean +0.067, over 46.7%, severe 30.0% — **FAIL**.
- D red pre-drop: mean −0.115, over 26.7%, severe 20.0% — **FAIL**.

Decision: **REJECT.** No promotion match.

Lesson: v9 gives the missing representational flexibility that v8 lacked — A passes — but unfreezing even the final residual block is enough to reintroduce the nonlocal B/C/D guardrail drift. The v8/v9 contrast identifies the tradeoff location: value-head-only protects B/C/D but cannot move A; value head + final block moves A but breaks B/C/D. Do **not** run v9b last-2 blocks as the next simple extension; broader partial unfreeze is expected to worsen the same failure mode unless a new constraint/objective is introduced.


## v10/v10b — guarded final block with root/search-path schedule (RUN + REJECTED 2026-07-05)

### v10 — schedule-only root + continuation retention

Design spec: `docs/superpowers/specs/2026-07-04-targeted-value-calibration-v10-final-block-root-continuation-schedule-design.md` (committed @ `91e14ec`).

Setup: same v7 manifest and same v9 update surface (`--freeze-batchnorm-stats --train-value-head-and-final-block`), but changed only `--post-opening-calibration-tag-schedule` to enable dormant root tags alongside already-scheduled continuation tags:

`black_predrop_correction=2,goal_line_retention=1,goal_line_continuation_retention=1,old_post_opening_retention=1,old_post_opening_continuation_retention=2,red_predrop_root_value_retention=1,red_predrop_continuation_retention=2,red_predrop_severe_root_correction=1`

Telemetry/verifier:
- `calib_n_drawn_total=1760`, `calib_n_drawn_per_step=11.0`.
- Draws by tag: A 320, B root 160, B cont 160, C root 160, C cont 320, D root-value 160, D cont 320, D severe 160.
- `n_teacher_retention_drawn=320`, `calib_policy_ce_avg_iter=3.8914`, `calib_policy_kl_est_avg_iter=1.0596`, proving B/C root policy-CE rows fired.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed; all frozen tensors byte-identical.

Gate results:
- A black pre-drop: mean −0.004, over 20.0%, severe 16.7% — **PASS**, but with thin mean margin.
- B goal-line: mean −0.195, over 11.1%, severe 5.6% — **FAIL** by one severe row.
- C old post-opening: mean +0.016, over 23.3%, severe 10.0% — **PASS**.
- D red pre-drop: mean −0.067, over 26.7%, severe 3.3% — **FAIL** by one barely-severe row.

Remaining v10 blockers:
- B severe: `game_000015_ply_19` value +0.6435, top1_share 0.8425. B also had one non-severe over row: `game_000327_ply_63` value +0.3538.
- D severe: `red_loss_game_000752_predrop_ply_70_drop_72` value +0.5003, top1_share 0.055. The row was barely above the severe threshold and diffuse.

Decision: **REJECT.** No promotion match.

Lesson: v10 is the best near-pass in the line. Adding B/C root pressure did **not** make B/C worse; it recovered C fully and left B close. It also preserved the A fix. However, D remained structurally fragile and B/D still had one severe blocker each.

### v10b — stronger B/D schedule

Setup: config-only from v10; increased `goal_line_retention`, `red_predrop_root_value_retention`, and `red_predrop_severe_root_correction` from 1→2:

`black_predrop_correction=2,goal_line_retention=2,goal_line_continuation_retention=1,old_post_opening_retention=1,old_post_opening_continuation_retention=2,red_predrop_root_value_retention=2,red_predrop_continuation_retention=2,red_predrop_severe_root_correction=2`

Telemetry/verifier:
- `calib_n_drawn_total=2240`, `calib_n_drawn_per_step=14.0`.
- Draws by tag: A 320, B root 320, B cont 160, C root 160, C cont 320, D root-value 320, D cont 320, D severe 320.
- `n_teacher_retention_drawn=480`, `calib_policy_ce_avg_iter=4.1862`, `calib_policy_kl_est_avg_iter=1.0937`.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed; all frozen tensors byte-identical.

Gate results:
- A black pre-drop: mean +0.095, over 36.7%, severe 30.0% — **FAIL**.
- B goal-line: mean −0.310, over 11.1%, severe 0.0% — **PASS**.
- C old post-opening: mean +0.135, over 33.3%, severe 23.3% — **FAIL**.
- D red pre-drop: mean +0.043, over 23.3%, severe 13.3% — **FAIL**.

Decision: **REJECT.** No promotion match.

Lesson: v10b fixed B but broke A/C/D. The extra broad B/D pressure did not cleanly solve the remaining blockers; it pushed many previously-safe rows upward and increased search confidence/top1 concentration. Examples:
- A: `black_loss_game_000291` −0.1924 → +0.6422, top1 0.2625 → 0.7775; `black_loss_game_000347` +0.0227 → +0.7694, top1 0.3175 → 0.9325.
- C: `game_000103` −0.3307 → +0.6406, top1 0.2600 → 0.9625; `game_000433` +0.0955 → +0.6995, top1 0.9625 → 0.9875.
- D: `red_loss_game_000362` +0.2579 → +0.6965; `red_loss_game_000578` +0.1619 → +0.5492; `red_loss_game_000780` +0.3815 → +0.5407.

Conclusion: **broad schedule-count pressure is exhausted.** The correct branch point, if continuing, is v10, not v10b. The only disciplined continuation is a surgical manifest edit with value-only rows for the specific v10 blockers (e.g. B `game_000015_ply_19`, possibly B `game_000327_ply_63`, and D `red_loss_game_000752_predrop_ply_70_drop_72`) while keeping the v10 schedule and avoiding any new policy-CE pressure.

## v11 — surgical B value-only root clones (RUN + REJECTED 2026-07-05)

### v11 manifest / training setup

Goal: test whether v10's B blocker was caused by B root-policy CE or missing value-only B root preservation. This was intentionally **manifest-only**, not trainer code and not new MCTS/inference.

Manifest script: `scripts/GPU/alphazero/build_v11_surgical_root_value_manifest.py` created `logs/eval/targeted_calibration_v11_surgical_root_value_from_v10_nearmiss.csv` from the v7 manifest by appending two depth-0 `goal_line_root_value_retention` clones:

- `game_000015_ply_19__root_value` with `teacher_value=0.0469`.
- `game_000327_ply_63__root_value` with `teacher_value=-0.8036`.

Both loaded through `calibration_pool` as `loss_mode=searched_continuation_retention`, `has_policy_target=False`, proving they are value-only and the copied SHA/root reconstruction path is valid.

Training setup: same v10 final-block mechanics (`--freeze-batchnorm-stats --train-value-head-and-final-block`) and a v10-shaped schedule, but with B root-policy CE replaced by B value-only root clones:

`black_predrop_correction=2,goal_line_root_value_retention=2,goal_line_continuation_retention=1,old_post_opening_retention=1,old_post_opening_continuation_retention=2,red_predrop_root_value_retention=1,red_predrop_continuation_retention=2,red_predrop_severe_root_correction=1`

Telemetry/verifier:
- `calib_n_drawn_total=1920`, `calib_n_drawn_per_step=12.0`.
- Draws by tag: A 320, B root-value 320, B continuation 160, C root 160, C continuation 320, D root-value 160, D continuation 320, D severe 160.
- `n_teacher_retention_drawn=160` (down from v10's 320), proving only C root policy-CE rows remained active; B root rows were value-only.
- `calib_policy_ce_avg_iter=2.9096`, `calib_policy_kl_est_avg_iter=1.1022`.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed; all frozen tensors byte-identical.

Gate results:
- A black pre-drop: mean −0.039, over 30.0%, severe 13.3% — **PASS**.
- B goal-line: mean −0.060, over 22.2%, severe 16.7% — **FAIL**, worse than v10.
- C old post-opening: mean +0.058, over 23.3%, severe 20.0% — **FAIL**.
- D red pre-drop: mean −0.109, over 30.0%, severe 6.7% — **FAIL**.

Decision: **REJECT.** No promotion match.

Lesson: v11 did not isolate/fix B. Replacing B root-policy CE with B value-only root clones made B worse and also lost C/D. Therefore v10's B failure was **not** simply caused by B root-policy CE or missing value-only B root preservation. This closes the v10/v11 schedule/manifest branch: existing root/continuation/root-value/severe-row levers cannot safely pass A/B/C/D under final-block training.

## v12 — asymmetric one-sided guardrail hinge (RUN + REJECTED 2026-07-06)

### v12 setup

Goal: introduce a new objective rather than another root/continuation schedule variant. v12 added `loss_mode=asymmetric_guardrail_retention`: a value-only, one-sided hinge that penalizes only candidate drift more pro-black than BASE by more than a margin. The hinge is computed in black perspective via a per-row sign: `relu(sign * (cb_values - cb_targets) - margin)^2`, where `sign=+1` for black-to-move and `-1` for red-to-move.

Manifest: `logs/eval/targeted_calibration_v12_guardrail_from_calib020_0001.csv` (136 rows: 58 hard-value rows + 78 B/C/D root guardrail clones). Guardrail tags: `goal_line_guardrail_retention` 18, `old_post_opening_guardrail_retention` 30, `red_predrop_guardrail_retention` 30.

Training setup: canonical calibration harness from the prior branches (`--iterations 1`, `--lr 0.0003`, `--curriculum-sizes 24`, `--games-per-iter 100`, `--simulations 400`, `--max-moves 280`, `--mcts-eval-batch-size 14`, `--mcts-pending-virtual-visits 8`, `--mcts-stall-flush-sims 48`, `--n-workers 10`, resign/adjudication settings, `--max-positions-per-game 280`) with `--freeze-batchnorm-stats --train-value-head-and-final-block`, target `-0.35`, weight `0.01`, and `--guardrail-margin 0.10`.

Schedule:

`black_predrop_correction=2,goal_line_guardrail_retention=1,old_post_opening_guardrail_retention=2,red_predrop_guardrail_retention=2`

Telemetry/verifier:
- `active_size=24`, `max_moves=280`, `games_per_iter=100`, `mcts_eval_batch_size=14`, `mcts_stall_flush_sims=48`.
- `calib_n_drawn_total=1120`, `calib_n_drawn_per_step=7.0`.
- Draws: A 320, B root guardrail 160, C root guardrail 320, D root guardrail 320.
- `guardrail_hinge_loss=0.02048`, `guardrail_active_frac=0.225`, `guardrail_margin=0.1`.
- `n_teacher_retention_drawn=0`, `calib_policy_ce_avg_iter=0.0`, `calib_policy_kl_est_avg_iter=0.0`, proving no policy CE / teacher-retention path was active.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed; all frozen tensors byte-identical.

### v12 gate results

- A black pre-drop: mean +0.005, over 20.0%, severe 13.3% — **FAIL by mean only** (pass requires mean ≤ 0.0).
- B goal-line: mean −0.214, over 5.6%, severe 0.0% — **PASS**.
- C old post-opening: mean +0.057, over 23.3%, severe 16.7% — **FAIL by severe**.
- D red pre-drop: mean −0.088, over 23.3%, severe 3.3% — **FAIL by severe**.

Decision: **REJECT.** No promotion match.

Lesson: v12 is the strongest evidence so far that the **objective shape matters**. The one-sided hinge fixed B cleanly without policy CE and avoided the v10/v11 B failure mode. However, root-only guardrails were not enough for C/D: C still showed a broad severe repeat-offender cluster (`game_000505`, `game_000565`, `game_000619`, `game_000065`, `game_000433`, etc.), and D had one severe row (`red_loss_game_000362...` +0.5257) plus many elevated non-severe over rows. A nearly passed by aggregate mean but still had severe top rows. The next branch should keep the v12 objective and add searched-continuation guardrail states for C/D before escalating to gradient projection.

## v12b — continuation guardrail rows (RUN + REJECTED 2026-07-06)

v12b reused the v12 objective unchanged and extended the guardrail state coverage. It did **not** change `trainer.py`, add a loss mode, add a CLI flag, change gates, or implement gradient projection.

Implementation:
- Merged/pushed on `origin/main` at `7335605` after v12 (`2cc4bd1`).
- Authoritative suite on merged main: 1354 passed, 0 failures.
- Loader change: in `build_calibration_position`, a guardrail row with non-empty `extra_moves_json` now walks `_apply_extra_moves`, so the hinge applies to the searched continuation board instead of the root. Root guardrail rows with blank `extra_moves_json` remain v12-compatible.
- Builder: `scripts/GPU/alphazero/build_v12b_continuation_guardrail_manifest.py`.
- Smoke: `scripts/GPU/alphazero/smoke_v12b_continuation_guardrail.py`.

Manifest:
- `logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv`
- 353 rows total: 58 hard-value, 78 root guardrail, 217 continuation guardrail.
- Counts: `goal_line_guardrail_retention=18`, `old_post_opening_guardrail_retention=30`, `red_predrop_guardrail_retention=30`, `old_post_opening_continuation_guardrail_retention=90`, `red_predrop_continuation_guardrail_retention=127`.
- Schema loaded as `asymmetric_guardrail_retention`.
- Smoke passed: `guardrail_hinge_loss=0.138`, `active_frac=0.429`, `guardrail_n=7`.

Training setup:
- Checkpoint: `checkpoints/alphazero-v12b-continuation-guardrail-from-calib020-0001/model_iter_0001.safetensors`.
- Canonical 24x24 harness, loaded `calib020_0001`, `--guardrail-margin 0.10`, `--freeze-batchnorm-stats`, `--train-value-head-and-final-block`.
- Schedule: `black_predrop_correction=2,goal_line_guardrail_retention=1,old_post_opening_guardrail_retention=1,old_post_opening_continuation_guardrail_retention=2,red_predrop_guardrail_retention=1,red_predrop_continuation_guardrail_retention=2`.

Telemetry/verifier:
- `active_size=24`, `max_moves=280`, `games_per_iter=100`, `mcts_eval_batch_size=14`, `mcts_stall_flush_sims=48`.
- `calib_n_drawn_total=1440`, `calib_n_drawn_per_step=9.0`.
- Draws by tag: A 320, B root guardrail 160, C root guardrail 160, C continuation guardrail 320, D root guardrail 160, D continuation guardrail 320.
- `guardrail_hinge_loss=0.01855`, `guardrail_active_frac=0.299`, `guardrail_margin=0.1`.
- `n_teacher_retention_drawn=0`, `calib_policy_ce_avg_iter=0.0`, `calib_policy_kl_est_avg_iter=0.0`.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed; all frozen tensors byte-identical.

Gate results:
- A black pre-drop: mean −0.137, over 30.0%, severe 13.3% — **PASS**.
- B goal-line: mean −0.302, over 5.6%, severe 5.6% — **FAIL** by severe.
- C old post-opening: mean +0.028, over 33.3%, severe 23.3% — **FAIL** by severe.
- D red pre-drop: mean −0.093, over 40.0%, severe 16.7% — **HARD FAIL**.

Failure-row readout:
- B had one severe row, `game_000299_ply_39` at +0.5807. The old v10/v11 blocker `game_000015_ply_19` was no longer severe, so B remains fragile but not fixed by targeting one historical blocker.
- C severe rows were dominated by stable repeat offenders: `game_000505_ply_037`, `game_000565_ply_033`, `game_000619_ply_061`, `game_000433_ply_029`, `game_000065_ply_021`, `game_000309_ply_019`, plus other high rows.
- D severe rows were broad/diffuse: `red_loss_game_000176`, `000278`, `000780`, `000456`, and `000438` were all severe/high. This matches the historical D pattern as broad value-surface drift, not a single fixed blocker.

Decision: **REJECT.** No promotion match.

Lesson: v12b falsifies the "root-only coverage was the main remaining problem" hypothesis. The one-sided hinge is useful, but adding searched-continuation guardrail coverage for C/D still did not protect B/C/D under value-head + final-block training. The remaining failure is more consistent with **gradient conflict in the shared final block**: A correction needs the final block to move, but the same update directions can still increase guardrail overvalue even when one-sided root and continuation guardrails are present.

## v13 — asymmetric gradient-conflict projection (RUN + REJECTED 2026-07-08)

v13 kept the useful v12b objective/state coverage and changed update mechanics. The core idea was to split calibration minibatch gradients into A correction and guardrail hinge pieces on the applied trainable surface (`value_head.*` + final residual block `encoder.blocks.5.*`). When the A gradient conflicted with the guardrail gradient (`dot(g_A,g_G)<0`), v13 projected the A component away from the guardrail direction before applying the combined surface update.

Setup:
- Base: `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` (`calib020_0001`).
- Manifest: `logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv`.
- Schedule: `black_predrop_correction=2,goal_line_guardrail_retention=1,old_post_opening_guardrail_retention=1,old_post_opening_continuation_guardrail_retention=2,red_predrop_guardrail_retention=1,red_predrop_continuation_guardrail_retention=2`.
- Training surface: `--freeze-batchnorm-stats --train-value-head-and-final-block --post-opening-calibration-gradient-projection`.
- Guardrail margin: 0.10.

Telemetry note: the first v13 checkpoint had projection fields missing from the flattened `model_iter_0001.json`. The code path was present and gates were valid, but conflict-rate/removed-norm could not be read. After fixing the two-site telemetry flattening mirror, v13 was rerun unchanged as `checkpoints/alphazero-v13-gradient-projection-telemetryfix-from-calib020-0001/model_iter_0001.safetensors`.

Telemetry/verifier for the telemetry-fixed run:
- `calib_projection_enabled=True`, `calib_projection_scope=value_head_and_final_block`.
- `calib_projection_conflict_steps=35`, `calib_projection_conflict_rate=0.2846`.
- `calib_projection_removed_norm_avg=0.0903`, `calib_projection_guardrail_grad_norm_avg=4.0431`, `calib_projection_a_grad_norm_avg=13.4087`.
- `guardrail_hinge_loss=0.01999`, `guardrail_active_frac=0.2339`, `guardrail_margin=0.1`.
- `calib_n_drawn_total=1440`, `calib_n_drawn_per_step=9.0`.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed; all frozen tensors byte-identical.

Gate results, telemetry-fixed run:
- A black pre-drop: mean −0.117, over 23.3%, severe 20.0% — **PASS**.
- B goal-line: mean −0.343, over 5.6%, severe 0.0% — **PASS**.
- C old post-opening: mean −0.083, over 26.7%, severe 16.7% — **FAIL** by severe.
- D red pre-drop: mean −0.151, over 36.7%, severe 13.3% — **FAIL**.

Decision: **REJECT.** No promotion match.

Lesson: projection engaged and was directionally useful: A/B passed and C/D improved in shape versus v12b. But C/D still had severe failures, so projection strength or protection needed a cleaner follow-up. The telemetry fix is mandatory for all later projection branches.

## v13b — projection with lower guardrail margin 0.05 (RUN + REJECTED 2026-07-08)

v13b was an arguments-only test of whether v13 failed because the guardrail hinge was not active enough. It kept v13 mechanics and schedule but changed `--guardrail-margin 0.10` to `--guardrail-margin 0.05`.

Telemetry/verifier:
- `guardrail_margin=0.05`, `guardrail_hinge_loss=0.02263`, `guardrail_active_frac=0.2857`.
- `calib_projection_strength` did not exist yet; projection used v13 strength 1.0.
- `calib_projection_conflict_steps=57`, `calib_projection_conflict_rate=0.4161`.
- `calib_projection_removed_norm_avg=0.0991`.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed.

Gate results:
- A black pre-drop: mean −0.017, over 36.7%, severe 20.0% — **PASS**, but weaker than v13.
- B goal-line: mean −0.370, over 11.1%, severe 0.0% — **PASS**.
- C old post-opening: mean −0.063, over 26.7%, severe 16.7% — **FAIL** by severe.
- D red pre-drop: mean −0.203, over 23.3%, severe 16.7% — **FAIL**, worse by severe.

Failure-row readout:
- A retained 19 severe rows, including high rows such as `black_loss_game_000281...` +0.8451 and `black_loss_game_000611...` +0.8037.
- C stayed in the same repeat-offender family (`game_000505`, `000565`, `000619`, `000433`, `000499`, `000065`, `000369`).
- D severe unique rose to 5 rows: `red_loss_game_000362`, `000728`, `000172`, `000176`, `000780`.

Decision: **REJECT.** No promotion match.

Lesson: lowering the margin increased guardrail/projection activity, but did not solve C/D and weakened A. This closes global margin-tightening as a projection fix.

## v13c — projection-strength scalar 2.0 (RUN + REJECTED 2026-07-08)

v13c added one CLI arg: `--post-opening-calibration-projection-strength`. The implementation folds the scalar into the effective projection weight while keeping the helper signature and geometric conflict primitive unchanged:

```
effective_projection_weight = post_opening_calibration_projection_strength * calibration_loss_weight
project_conflicting_gradient(..., weight=effective_projection_weight)
```

The test used `--post-opening-calibration-projection-strength 2.0`, restored `--guardrail-margin 0.10`, and otherwise kept the v13/v12b manifest, schedule, trainable surface, and projection mechanics unchanged.

Telemetry/verifier:
- `guardrail_margin=0.1`, `guardrail_hinge_loss=0.01345`, `guardrail_active_frac=0.2205`.
- `calib_projection_enabled=True`, `calib_projection_scope=value_head_and_final_block`, `calib_projection_strength=2.0`.
- `calib_projection_conflict_steps=46`, `calib_projection_conflict_rate=0.368`.
- `calib_projection_removed_norm_avg=0.1292`, up from v13's 0.0903, proving the stronger correction increased actual applied projection magnitude.
- `calib_projection_guardrail_grad_norm_avg=2.5302`, `calib_projection_a_grad_norm_avg=8.7761`.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed.

Gate results:
- A black pre-drop: mean −0.052, over 13.3%, severe 10.0% — **PASS**.
- B goal-line: mean −0.243, over 0.0%, severe 0.0% — **PASS**.
- C old post-opening: mean −0.076, over 6.7%, severe 6.7% — **PASS**.
- D red pre-drop: mean −0.073, over 16.7%, severe 3.3% — **FAIL** by one severe row.

Failure-row readout:
- Only D severe: `red_loss_game_000728_predrop_ply_48_drop_50` at +0.5441 (3 duplicate raw severe rows in the probe CSV).
- Next D rows were non-severe: `red_loss_game_000362...` +0.4838, `red_loss_game_000752...` +0.3901, `red_loss_game_000176...` +0.3302, `red_loss_game_000780...` +0.3102.

Decision: **REJECT by strict gate.** No promotion match.

Lesson: v13c is the best projection result and proves the projection-strength mechanism helped: A/B/C all passed and D missed by one barely-severe row. But D's gate requires `severe=0.0%`, so v13c cannot be promoted. The only justified continuation is a tightly scoped cleanup of D root guardrail sampling, not blind strength/margin tuning.

## v13d — v13c plus D root guardrail draw 2 (RUN + REJECTED 2026-07-08)

v13d was the only justified argument-only cleanup run after v13c. It kept:
- Base `calib020_0001`.
- v12b manifest.
- `projection_strength=2.0`.
- `guardrail_margin=0.10`.
- `--freeze-batchnorm-stats --train-value-head-and-final-block --post-opening-calibration-gradient-projection`.

Only the tag schedule changed:

```
red_predrop_guardrail_retention=1 -> 2
```

Full v13d schedule:

`black_predrop_correction=2,goal_line_guardrail_retention=1,old_post_opening_guardrail_retention=1,old_post_opening_continuation_guardrail_retention=2,red_predrop_guardrail_retention=2,red_predrop_continuation_guardrail_retention=2`

Checkpoint:
- `checkpoints/alphazero-v13d-projection-strength2-red-root-guardrail2-from-calib020-0001/model_iter_0001.safetensors`.

Telemetry/verifier:
- `guardrail_margin=0.1`, `guardrail_hinge_loss=0.023796`, `guardrail_active_frac=0.213281`.
- `calib_projection_enabled=True`, `calib_projection_scope=value_head_and_final_block`, `calib_projection_strength=2.0`.
- `calib_projection_conflict_steps=46`, `calib_projection_conflict_rate=0.3459`.
- `calib_projection_removed_norm_avg=0.1802`, `calib_projection_guardrail_grad_norm_avg=4.3081`, `calib_projection_a_grad_norm_avg=12.6490`.
- `calib_n_drawn_total=1600`, `calib_n_drawn_per_step=10.0`.
- Draws by tag: A 320, B 160, C root 160, C continuation 320, D root 320, D continuation 320.
- Strict verifier passed: only `value_head.*` and `encoder.blocks.5.*` trainable tensors changed; all frozen tensors byte-identical.

Gate results:
- A black pre-drop: mean −0.083, over 26.7%, severe 23.3% — **PASS**, but weakened versus v13c.
- B goal-line: mean −0.265, over 16.7%, severe 5.6% — **FAIL**.
- C old post-opening: mean −0.013, over 33.3%, severe 20.0% — **FAIL**.
- D red pre-drop: mean −0.113, over 33.3%, severe 20.0% — **HARD FAIL**.

Decision: **REJECT.** No promotion match.

Lesson: the one-row D-root cleanup did not clear D. It damaged B/C/D and weakened A relative to v13c. This closes the v13 projection/cleanup line: do not keep tuning root draw pressure, projection strength, margin, or tag schedule inside the final-block projection family without a new written design.

## v14 — gated value-adapter surface, projection OFF (RUN + REJECTED 2026-07-08)

v14 changed the training surface, not the objective. It tested whether a value-only adapter — more capacity than value-head-only but isolated from policy/trunk updates — could move A without the B/C/D guardrail drift caused by final-block training.

Setup:
- Base: `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` (`calib020_0001`).
- Checkpoint: `checkpoints/alphazero-v14-value-adapter-from-calib020-0001/model_iter_0001.safetensors`.
- Manifest: `logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv`.
- Schedule: `black_predrop_correction=2,goal_line_guardrail_retention=1,old_post_opening_guardrail_retention=1,old_post_opening_continuation_guardrail_retention=2,red_predrop_guardrail_retention=1,red_predrop_continuation_guardrail_retention=2`.
- Objective: v12b asymmetric one-sided guardrail hinge, `guardrail_margin=0.10`.
- Projection: OFF.
- Adapter: `--value-adapter --value-adapter-bottleneck-width 32`.
- Training surface: `--train-value-head-and-value-adapter`; encoder, policy head, final residual block, and BN running stats frozen.

Implementation note: the first gate-eval attempt exposed a scoring loader gap — `probe_eval.load_network_for_scoring()` instantiated a no-adapter network and rejected the v14 checkpoint's `value_adapter.*` keys. The fix was to detect `value_adapter.*` keys in the safetensors file and construct `create_network(..., value_adapter=True)` only for adapter checkpoints. This preserved base/v8-v13 loading behavior.

Telemetry/verifier:
- `train_value_head_and_value_adapter=True`.
- `train_value_head_only=False`, `train_value_head_and_final_block=False`.
- `freeze_batchnorm_stats=True`.
- `value_adapter_gate=0.003017987357452512`.
- `value_adapter_grad_norm=0.0013807759423798416`.
- `guardrail_hinge_loss=0.007941251490490764`, `guardrail_active_frac=0.25803572256118057`.
- `calib_n_drawn_total=1440`, `calib_n_drawn_per_step=9.0`.
- Draws by tag: A 320, B 160, C root 160, C continuation 320, D root 160, D continuation 320.
- `calib_value_term_avg_iter=0.0`, `calib_policy_ce_avg_iter=0.0`, `calib_policy_kl_est_avg_iter=0.0`.
- Verifier passed: 92 base tensors compared; shared frozen set byte-identical; only `value_head.*` and `value_adapter.*` changed; `value_adapter.gate` moved to 0.003018.

Gate results:
- A black pre-drop: baseline mean +0.257 / over 50.0% / severe 43.3%; v14 mean +0.064 / over 26.7% / severe 20.0% — **FAIL**, but strongly improved.
- B goal-line: baseline mean −0.244 / over 5.6% / severe 0.0%; v14 mean −0.272 / over 5.6% / severe 0.0% — **PASS**.
- C old post-opening: baseline mean +0.099 / over 33.3% / severe 13.3%; v14 mean +0.063 / over 30.0% / severe 6.7% — **PASS**.
- D red pre-drop: baseline mean −0.188 / over 13.3% / severe 0.0%; v14 mean −0.079 / over 23.3% / severe 0.0% — **PASS by formal gate**, but with degraded mean/overvalue margin.

Decision: **REJECT.** No promotion match.

Lesson: v14 is not an underfit/no-move result. The adapter surface moved A substantially while B/C/D formally held, so value-only adapter capacity is real and safer than final-block training. However, A still failed by mean and D's margin moved toward black even with the trunk/policy/final block frozen. Width 64 is not the next branch because A did move. The next justified branch is **v14b**: same adapter surface, same v12b objective, projection ON over the adapter value surface (`value_head.*` + `value_adapter.*`), bottleneck 32, projection strength 1.0 first.


## v14b — value-adapter surface + gradient projection, strength 1.0 (RUN + REJECTED 2026-07-09)

v14b tested the planned follow-up to v14: keep the value-only adapter surface and v12b objective, but enable the v13 A-yields-to-guardrail projection over `{value_head, value_adapter}` instead of the final residual block.

Setup:
- Base: `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` (`calib020_0001`).
- Checkpoint: `checkpoints/alphazero-v14b-value-adapter-projection-from-calib020-0001/model_iter_0001.safetensors`.
- Manifest/schedule/objective: same v12b continuation-guardrail manifest, tag schedule, asymmetric one-sided guardrail hinge, `guardrail_margin=0.10`.
- Adapter: `--value-adapter --value-adapter-bottleneck-width 32`.
- Training surface: `--train-value-head-and-value-adapter`; encoder, policy head, final residual block, and BN running stats frozen.
- Projection: `--post-opening-calibration-gradient-projection`, default `--post-opening-calibration-projection-strength 1.0`.

Implementation notes:
- The first v14b attempt failed at training step 0 because the projection guard still rejected the adapter surface. The implemented fix allowed projection with `--train-value-head-and-value-adapter`, selected the surface `{value_head, value_adapter}`, and left `project_conflicting_gradient` unchanged.
- Slot `[13]` telemetry was made self-describing: dict = projection telemetry (v13/v14b), float = v14 adapter grad norm only. Under v14b, `value_adapter_grad_norm` is folded into the projection dict and is the post-projection/applied adapter grad norm.
- The first completed v14b run exposed a telemetry-label bug: `calib_projection_scope` still said `value_head_and_final_block`. The tensor-diff verifier proved this was a label bug, not a surface leak; the final block stayed byte-identical. The label was fixed for subsequent runs.

Telemetry/verifier:
- `train_value_head_and_value_adapter=True`, `train_value_head_and_final_block=False`, `freeze_batchnorm_stats=True`.
- `value_adapter_gate=0.0016666483134031296`.
- `value_adapter_grad_norm=0.0015184665251581464`.
- `calib_projection_enabled=True`, `calib_projection_conflict_steps=51`, `calib_projection_conflict_rate=0.3984375`, `calib_projection_removed_norm_avg=0.07273300471135768`.
- `calib_projection_guardrail_grad_norm_avg=0.7594021144822769`, `calib_projection_a_grad_norm_avg=9.11618999313071`.
- `guardrail_hinge_loss=0.0032312683621622236`, `guardrail_active_frac=0.22500000698491932`.
- `calib_n_drawn_total=1440`, `calib_n_drawn_per_step=9.0`; draws by tag: A 320, B 160, C root 160, C continuation 320, D root 160, D continuation 320.
- Verifier passed: shared frozen set byte-identical; only `value_head.*` and `value_adapter.*` changed; `value_adapter.gate` moved to 0.001667.

Gate results:
- A black pre-drop: baseline mean +0.257 / over 50.0% / severe 43.3%; v14b mean +0.026 / over 26.7% / severe 16.7% — **FAIL**, but closer than v14.
- B goal-line: baseline mean −0.244 / over 5.6% / severe 0.0%; v14b mean −0.254 / over 11.1% / severe 0.0% — **PASS** at the over cap.
- C old post-opening: baseline mean +0.099 / over 33.3% / severe 13.3%; v14b mean +0.044 / over 23.3% / severe 6.7% — **PASS**.
- D red pre-drop: baseline mean −0.188 / over 13.3% / severe 0.0%; v14b mean −0.047 / over 23.3% / severe 0.0% — **PASS** by formal gate, with degraded mean margin.

Decision: **REJECT / near-pass.** No promotion match because A requires mean ≤ 0.0 and v14b remained positive at +0.026.

Lesson: v14b is the best adapter result so far. Projection helped A relative to v14 (`+0.064 → +0.026`) and B/C/D remained inside formal gates, so the adapter-projection mechanism is directionally correct. Because the only remaining blocker is A mean and projection did not break formal guardrails, the disciplined next step is **v14c**: same setup with `--post-opening-calibration-projection-strength 2.0`. Do not widen the adapter, change margin, or redesign the objective before v14c completes.

## Retired hypothesis — v6 searched-continuation/PV retention

**Working shape (2026-07-02, pre-design):** stop anchoring only the fragile root positions; anchor what search actually visits beneath them.

- **Retention rows:** for each fragile B/C/D row, run BASE MCTS (gate-faithful, 400 sims) and extract **child/PV states** (the searched continuations whose values drifted in v5). Each extracted state becomes its own retention row with a **raw teacher value** anchor (BASE eval-mode forward at that state — the v4/v5 value mechanism that provably holds). **Policy retention only where the visit distribution is sharp** (diffuse D-row distributions gave weak/noisy targets in v5).
- **Correction rows:** unchanged A hard-value family.
- **Manifest encoding:** prefer an `extra_moves_json` column on continuation rows. Reconstruct the source replay prefix with `replay_path + position_ply`, then apply `extra_moves_json` to reach the continuation state. Avoid new sidecar replay files unless `extra_moves_json` becomes too invasive or brittle.
- **State selection:** start conservative. For sharp C rows, keep the fragile root plus the top BASE child / PV line to depth 2–3. For diffuse D rows, add top-k BASE children only where no single PV dominates. Do not extract every visited child until row counts and dilution risk are understood.
- **Training:** should ride the existing retention machinery (`teacher_retention`-style rows over the masked 14-tuple path) — continuation rows are just additional positions.
- **Tags / schedule:** do not hide continuation rows under old root-retention tags. Use separate continuation tags so they can be scheduled and audited independently. Starting schedule candidate: `black_predrop_correction=2,goal_line_root_retention=1,old_post_opening_continuation_retention=2,red_predrop_continuation_retention=2`; adjust only after the builder reports final row counts and tag mass.
- **Gate:** same A/B/C/D probes vs `calib020_0001`. No promotion unless all four pass.


## Code / artifact pointers

- **v2** manifest builder + mixed-pool weighted loss: `scripts/GPU/alphazero/build_targeted_calibration_manifest.py`; operator guide `docs/post-game-analysis.md` §6.
- **v3** tag-stratified sampling: `--post-opening-calibration-tag-schedule` (commits `0c122cb` / `0e0fd24` / `282998d` / `b27d60b` on `main`); telemetry `state.calib_n_drawn_by_tag` + sidecar `post_opening_calibration.draws_by_tag`; operator guide `docs/post-game-analysis.md` §6 (tag-stratified block).
- **v4** teacher-retention builder/smoke/training path: `scripts/GPU/alphazero/build_teacher_calibration_manifest.py`, `scripts/GPU/alphazero/smoke_teacher_calibration_v4.py`, `--post-opening-calibration-teacher-value-weight`, `--post-opening-calibration-teacher-policy-kl-weight`, and `--freeze-batchnorm-stats`.
- **v3-frozenBN-control checkpoint/gates:** `checkpoints/alphazero-v3-frozenBN-control-from-calib020-0001/model_iter_0001.safetensors`, `logs/eval/v3_frozenBN_control_from_calib020_0001_gates_400s/`.
- **v4/v3F severe-overlap review:** `logs/eval/v3f_v4_severe_overlap_review.csv`.
- **Raw-NN focus-row diagnostic:** `scripts/GPU/alphazero/eval_raw_nn_position_rows.py` (+ `tests/test_eval_raw_nn_position_rows.py`, merged to main @ `7064621`); output `logs/eval/v3f_v4_raw_nn_focus_rows.csv`; plan `docs/superpowers/plans/2026-07-01-eval-raw-nn-position-rows-diagnostic.md`.
- **v5 root-retention:** builder `scripts/GPU/alphazero/build_mcts_root_retention_manifest.py` (+ `--gate-checkpoint-label` cross-check), smoke `scripts/GPU/alphazero/smoke_mcts_root_retention_v5.py`, `loss_mode=mcts_root_retention` in `calibration_pool.py`; manifest `logs/eval/targeted_calibration_v5_mcts_root_from_calib020_0001.csv`; checkpoint `checkpoints/alphazero-v5-mcts-root-from-calib020-0001/model_iter_0001.safetensors`; plan `docs/superpowers/plans/2026-07-01-targeted-value-calibration-v5-mcts-root-retention.md`; operator guide `docs/post-game-analysis.md` §8.
- **v6/v6c searched-continuation retention:** builder `scripts/GPU/alphazero/build_searched_continuation_retention_manifest.py`, smoke `scripts/GPU/alphazero/smoke_searched_continuation_retention_v6.py`, `loss_mode=searched_continuation_retention` in `calibration_pool.py`; manifests `logs/eval/targeted_calibration_v6_continuation_from_calib020_0001.csv` and `logs/eval/targeted_calibration_v6c_d_root_value_only_from_calib020_0001.csv`.
- **v7 severe-D hard correction:** manifest-only branch `logs/eval/targeted_calibration_v7_severe_d_root_correction_from_calib020_0001.csv`; severe-D rows selected via `scripts/GPU/alphazero/eval_raw_nn_position_rows.py`; no loader/trainer changes.
- **v8 value-head-only:** CLI flag `--train-value-head-only` in `scripts/GPU/alphazero/train.py` / guard in `scripts/GPU/alphazero/trainer.py`; verifier `scripts/GPU/alphazero/verify_value_head_only_checkpoint.py`; checkpoint `checkpoints/alphazero-v8-value-head-only-v7-manifest-from-calib020-0001/model_iter_0001.safetensors`.
- **v9 value head + final block:** CLI flag `--train-value-head-and-final-block` in `scripts/GPU/alphazero/train.py` / three-way update branch in `scripts/GPU/alphazero/trainer.py`; verifier `scripts/GPU/alphazero/verify_value_head_and_final_block_checkpoint.py`; checkpoint `checkpoints/alphazero-v9-value-head-and-final-block-v7-manifest-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v9_value_head_and_final_block_v7_manifest_from_calib020_0001_gates_400s`.
- **v10/v10b final-block schedule-only branches:** same v9 flag/verifier and v7 manifest; checkpoints `checkpoints/alphazero-v10-final-block-root-plus-cont-v7-manifest-from-calib020-0001/model_iter_0001.safetensors` and `checkpoints/alphazero-v10b-final-block-root-plus-cont-stronger-bd-v7-manifest-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v10_final_block_root_plus_cont_v7_manifest_from_calib020_0001_gates_400s` and `logs/eval/v10b_final_block_root_plus_cont_stronger_bd_v7_manifest_from_calib020_0001_gates_400s`; v10 design spec `docs/superpowers/specs/2026-07-04-targeted-value-calibration-v10-final-block-root-continuation-schedule-design.md`.
- **v11 surgical B value-only root clones:** manifest-copy script `scripts/GPU/alphazero/build_v11_surgical_root_value_manifest.py`; manifest `logs/eval/targeted_calibration_v11_surgical_root_value_from_v10_nearmiss.csv`; checkpoint `checkpoints/alphazero-v11-b-root-value-surgical-v10-schedule-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v11_b_root_value_surgical_v10_schedule_from_calib020_0001_gates_400s`.
- **v12 asymmetric one-sided guardrail hinge:** loss mode `asymmetric_guardrail_retention` in `scripts/GPU/alphazero/calibration_pool.py`; hinge/sign/13-tuple path in `scripts/GPU/alphazero/trainer.py`; CLI `--guardrail-margin` in `scripts/GPU/alphazero/train.py`; builder `scripts/GPU/alphazero/build_v12_guardrail_manifest.py`; smoke `scripts/GPU/alphazero/smoke_asymmetric_guardrail_v12.py`; manifest `logs/eval/targeted_calibration_v12_guardrail_from_calib020_0001.csv`; checkpoint `checkpoints/alphazero-v12-asymmetric-guardrail-hinge-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v12_asymmetric_guardrail_hinge_from_calib020_0001_gates_400s`.
- **v12b continuation guardrails:** loader gate in `scripts/GPU/alphazero/calibration_pool.py` for `asymmetric_guardrail_retention` rows with non-empty `extra_moves_json`; builder `scripts/GPU/alphazero/build_v12b_continuation_guardrail_manifest.py`; smoke `scripts/GPU/alphazero/smoke_v12b_continuation_guardrail.py`; manifest `logs/eval/targeted_calibration_v12b_continuation_guardrail_from_calib020_0001.csv`; checkpoint target `checkpoints/alphazero-v12b-continuation-guardrail-from-calib020-0001`; gates target `logs/eval/v12b_continuation_guardrail_from_calib020_0001_gates_400s`.
- **v13 gradient-conflict projection:** CLI `--post-opening-calibration-gradient-projection`; projection path in `scripts/GPU/alphazero/trainer.py`; projection telemetry in both `trainer.py` flattened row and `calibration_pool.py` sidecar; smoke `scripts/GPU/alphazero/smoke_v13_gradient_projection.py`; telemetry-fixed checkpoint `checkpoints/alphazero-v13-gradient-projection-telemetryfix-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v13_gradient_projection_telemetryfix_from_calib020_0001_gates_400s`.
- **v13b margin 0.05:** arg-only projection branch; checkpoint `checkpoints/alphazero-v13b-gradient-projection-margin005-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v13b_gradient_projection_margin005_from_calib020_0001_gates_400s`.
- **v13c projection-strength scalar:** CLI `--post-opening-calibration-projection-strength`; effective projection weight folds in `projection_strength * calibration_loss_weight`; mandatory telemetry `calib_projection_strength` in sidecar + flattened row; design `docs/superpowers/specs/2026-07-07-targeted-value-calibration-v13c-projection-strength-design.md`; checkpoint `checkpoints/alphazero-v13c-projection-strength-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v13c_projection_strength2_from_calib020_0001_gates_400s`.
- **v13d red-root cleanup:** arg-only v13c cleanup with `red_predrop_guardrail_retention=2`; checkpoint `checkpoints/alphazero-v13d-projection-strength2-red-root-guardrail2-from-calib020-0001/model_iter_0001.safetensors`; gates `logs/eval/v13d_projection_strength2_red_root_guardrail2_from_calib020_0001_gates_400s`; rejected and closes the v13 projection/cleanup line.
- **v14 adapter line:** value-adapter surface (`value_head.*` + `value_adapter.*`) with scalar gate/bottleneck 32; v14b best near-pass checkpoint `checkpoints/alphazero-v14b-value-adapter-projection-from-calib020-0001/model_iter_0001.safetensors`; v14c/v14d rejected and close cleanup.
- **v15 raw/MCTS + continuation diagnostics:** raw A/D drift CSV `logs/eval/v15prep_raw_AD_drift_base_v14b_v14d.csv`; Phase 0 concentration script `scripts/GPU/alphazero/diagnose_v15_a_continuation_concentration.py` output `logs/eval/v15prep_a_continuation_concentration.csv`; Phase 0.5 subtree script `scripts/GPU/alphazero/diagnose_v15_a_selected_branch_subtrees.py` outputs `logs/eval/v15prep_a_selected_branch_subtrees.csv` and `logs/eval/v15prep_a_selected_branch_subtrees_by_depth_summary.csv`; closure summary files `logs/eval/v15prep_a_phase05_*.csv` / `.txt`; decision: no v15 Phase 1.
- **v16/v16a search-reliability diagnostics:** FPU hook and sweep `scripts/GPU/alphazero/diagnose_fpu_sweep.py`; neutral-manifest builder `scripts/GPU/alphazero/build_v16a_neutral_position_manifest.py`; manifest/meta `logs/eval/v16a_fpu_unbiased/neutral_position_manifest.csv` and `.meta.json`; held-out outputs `neutral_fpu_sweep_cases.csv`, `neutral_fpu_sweep_summary.csv`, `neutral_fpu_sweep_by_stratum.csv`, and `operator_sweep.log` in the same directory; design `docs/superpowers/specs/2026-07-10-v16a-unbiased-fpu-validation-design.md`; result: fixed absolute `−0.20` rejected by the late-stratum collapse gate.
- **v16 context-relative policy-mass successor:** rule/observer/diagnostic `scripts/GPU/alphazero/mcts.py` + `scripts/GPU/alphazero/diagnose_fpu_policy_mass.py`; phase-primary corpus `scripts/GPU/alphazero/fpu_dev_corpus_v2.py`; immutable reservoir protocol/qualification `scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py`; historical discovery root `logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/`; final production-v2 root `logs/eval/fpu_v16_policy_mass_v2/production_v2_b400amend_4000g_seed20300000/`; diagnostic root `logs/eval/fpu_v16_policy_mass_v2/diagnostic/production_v2_b400amend_4000g_seed20300000/`; design `docs/superpowers/specs/2026-07-10-context-relative-fpu-policy-mass-design.md`; corpus design `docs/superpowers/specs/2026-07-12-fpu-dev-corpus-v2-phase-design.md`; qualification design `docs/superpowers/specs/2026-07-14-fpu-v2-reservoir-protocol-qualification-design.md`; operator guide `docs/post-game-analysis.md` §11; final status: **REJECTED at production tuning-control qualification on 2026-07-24; no nonzero coefficient ran.**

- **Plans:** `docs/superpowers/plans/2026-06-24-targeted-value-calibration-v2.md`, `docs/superpowers/plans/2026-06-25-targeted-value-calibration-v3-tag-stratified-sampling.md`, `docs/superpowers/plans/2026-06-29-targeted-value-calibration-v4-teacher-retention.md`.

## v16 policy-mass successor — reservoir v1 POST-SCREEN GATE-FAIL + role-feasibility repair (2026-07-18/19)

**Outcome of reservoir protocol v1 (updates the RUNNING status above): the 4,800-game reservoir generated and passed `qualify` + `screen`, but `select` GATE-FAILED post-screen.** Kept target capacity by phase: **opening 0 / early_mid 0 / midgame 0 / late 136** (155 kept late-target rows across 86 games under the ≤2/game rule) vs the 45-per-phase demand of the original 240-row allocation. Target geometry is late-only on this net. **No FPU coefficient was tested; no A/B/C/D or strength result exists.** reservoir_v1 artifacts are preserved untouched as immutable discovery evidence.

**Role-feasibility repair (branch `fpu-v2-role-feasibility-repair`, 22 commits fca9c0d→da4dc00; plan `docs/superpowers/plans/2026-07-18-fpu-v2-role-feasibility-repair.md`):** schema-2 config-authoritative `AllocationProfile` (late-only targets), controlled `post-screen-qualify` stage (PASS = exact-selector witness, never capacity bounds), protocol v2 + `run_kind` production/smoke isolation, per-split late-target band minima, authenticated discovery commands, `historical_screen_discovery_v1` identity policy (producer vs analyzer provenance; strict stages unchanged). Suite 2262/0; schema-1 byte-identity pinned by pre-repair goldens. Whole-branch review: ready to merge (0 Critical / 0 Important).

**Task 14 zero-GPU proof on the immutable v1 screen (final rerun r2 under source bytes @ `da4dc00`, clean tree):**

- **Production profile (120 rows) exact-selector PASS, exit 0, deterministic** (r1 pre-cell-order-fix run byte-identical on witness/qualification/profile; superseded r1 reports preserved on disk).
- Final frozen allocation: `target|late` 40 tuning + 20 frozen_check; `control|{opening,early_mid,midgame,late}` 10 tuning + 5 frozen_check each; totals 80/40 tuning/frozen_check, 60/60 target/control.
- Frozen late-target band minima: totals b400_plus ≥ 8 / b300_399 ≥ 12 / b200_299 ≥ 12; per-split tuning {b400 4, b300 8, b200 8} / frozen_check {b400 4, b300 5, b200 5} — **now backed by a constructive witness** (bands realized 11/23/26; per-split tuning 7/15/18, frozen_check 4/8/8).
- **The frozen_check b400_plus minimum is satisfied with ZERO witness slack (4 of 4)** — 11 of the 12 candidate b400 rows (12 games, 7 black/5 red on screen) were selected. This makes Task 16 sizing decisive for fresh-reservoir reliability.
- Old 240-row allocation on the same screen: **GATE_FAIL, exit 4** — all three non-late target cells capacity 0 < demand 45 (confirms 0/0/0/136).
- Committed evidence (`logs/eval/fpu_v16_policy_mass_v2/analysis/`, SHA-1): `production_profile.json` 378d3cdb61c9b113af6cb8d1cf5c8fb41e3f39e2 · `old_allocation_profile.json` d0287df2eb128e5c8fe595897dd203ec0d8a9012 · `production_feasibility_r2.json` 2bef7133939faa9c778dd576155ced3a68a6bf61 · `old_allocation_feasibility_r2.json` 6295e1af467dcfef8019b271947f85334ba5ad66.

**Task 15 smoke (2026-07-19): technical PASS** — full chain from frozen protocol (base_seed 20280000, disjoint seeds): generate 400g → qualify 0 → screen (2,032 proposals / 581 kept) → post-screen-qualify PASS → select 18 rows (12/6 splits, 6/12 roles, sides 10/8); idempotent byte-identical re-runs; production diagnostic rejects the smoke config. Plumbing only — not a scientific result. SHA-1s: protocol ee43e84763457660f6dc7e20eab0ac6d1f015065, report 040cccecc979be5c1f4c32ea10ce204b9c6c1b2c, manifest fbc87b198c86d18b0e96082e1ff3962ef7552d01.

**Task 16 sizing (2026-07-21, preregistered 299-trial finite-reservoir subsampling, seed 20260718; decision artifact `sizing_report.json` SHA-1 92c130b34f8b5c31bf606b8b20530f622515e444): NO tier ≤ 4,800 meets the 95% lower bound ≥ 0.99 criterion.** Success by tier: 1200 0/299 · 1800 0/299 · 2400 9/299 · 3000 67/299 · 3600 175/299 · 4200 255/299 (lb95 0.8149) · 4800 degenerate 1/1. Failure spectrum: ≤1800 total late-target capacity < 60; 2400 b400_plus band capacity < 8; ≥3000 increasingly selector-level (per-split b400 on the scarce pool). **Per the preregistered rule the production protocol was NOT emitted; reservoir size beyond the 4,800-game discovery screen is an open science decision.** The b400_plus scarcity (12 candidate rows in 4,800 games) is the binding resource, consistent with the zero-slack Task 14 witness.

**Selector v2 revision (2026-07-21): the sizing verdict above was PARTLY an allocator artifact.** Constraint-aware split assignment (`split_assignment_version: 2`; scarce-band pins + historical fallback, feasible set ⊇ v1's, 250×30-seed differential fuzz zero regressions, schema-1 byte-identical) rerun on the preregistered finer ladder: **4,600 games 299/299, lb95 0.9900 — MEETS the criterion; preregistered next-tier-up rule → production count 4,800** (`sizing_report_selector_v2.json` SHA-1 81499afc2ac5c04300a4c7c5f376cf8b65f7335f). smoke_v2 (400g, seed 20290000): controlled sampling GATE_FAIL (`target|late capacity 5 < demand 6`), select refused, idempotent, isolation verified — small-sample variance, machinery correct. Zero-GPU assigner exercise on the authenticated smoke_v2 screen: PASS, selector v2 executed, pins engaged (SHA-1 73a337f19a093e43f560f5e7c8639abbc09c96ef). **At that checkpoint, the 4,800-game production run remained unauthorized; this status was later superseded by the b400 amendment and explicit 4,000-game authorization below.**

**Amendment `b400-coverage-floor-v1` (2026-07-21, user decision): b400_plus floor 8→4 total (2+2 per split) — coverage-only rationale (n < DEV_BAND_MIN_N can never activate a per-band gate); everything else unchanged.** Frozen profile `production_profile_v2_b400amend.json` (SHA-1 eb1dd21648e49388bff92de8c7831eb0a1a3f6e8). Witness under 2+2 on the discovery screen: PASS, 120 rows, selector v2, frozen_check b400 3≥2 (slack restored). Preregistered decision sizing (ladder 3000–4000 step 200, 299 trials): 287/295/298/**299 @3,600 MEETS**/298/**299 @4,000 MEETS** → smallest qualifying 3,600 → **preregistered margin rule yields production count 3,800** (`sizing_report_b400amend.json` SHA-1 7ff76fcb8a5a9d58f1ca2227767142a3c2f307dd; noted transparently: 3,800's own draw 298/299, single selector failure, non-monotonic noise). **AUTHORIZATION (2026-07-21): Production reservoir authorized at 4,000 games, board 24, four workers, seed range `[20300000,20304000)`, `run_kind=production`, no top-up. This intentionally deviates from the preregistered 3,800 margin result because 4,000 independently met the unchanged criterion at 299/299, while 3,800 produced 298/299. All allocation, target, b300/b200, side, spacing, isolation, and amended b400 2+2 requirements remain unchanged.**

## v16 policy-mass successor — production v2 corpus PASS, `r0` prerequisite REJECT (2026-07-24)

**Production reservoir and qualification.** The one authorized production-v2 reservoir completed under frozen source commit `9e6a606ee6385d0dd34de3b09ba38bb1d5c721f1`: 4,000 games, board 24, 400 simulations, four workers, base seed `20300000` / half-open range `[20300000,20304000)`, replay capture, and no top-up. The reservoir opponent result (`calib020_0001` score rate 0.588875 vs `0379`) is generation metadata only, not the policy-mass endpoint. `qualify` and `qualify --check` both exited 0; conformance and summary binding passed, geometric preflight was feasible with no binding constraint, and the terminal histogram was 3,915 wins / 85 state-cap games. Frozen protocol SHA-1: `0ee3ad8d8a2973d127b9518af66002030f78cdd6`.

**GPU screen.** The authenticated screen completed with exit 0: 20,464 proposal rows, of which 5,065 were kept, 11,381 failed the anchor, 3,879 collided with forbidden/consumed positions, and 139 were role-ineligible. Screen CSV SHA-1: `05db59f8f0c96410d1e3dd4091c83ba39f09c44c`; meta SHA-1: `e62b136362131563a230d188e09c85e190838242`. The screen recorded `add_noise=false`, the correct anchor checkpoint identity, and the expected config/protocol/source-index/replay fingerprints.

**Post-screen qualification and exact selection.** `post-screen-qualify` PASSed with no binding constraint and selector v2 succeeded on assignment attempt 0. The 120-row witness exactly realized the frozen allocation: 80 tuning / 40 frozen_check; 60 target / 60 control; tuning sides 40 red / 40 black; frozen sides 20/20. Late-target band counts were b200=33, b300=20, b400=7; per split, tuning `{b200:21,b300:15,b400:4}` and frozen `{b200:12,b300:5,b400:3}`, clearing every amended minimum. Post-screen report SHA-1: `f59e531a6f405e2d28588117dbfbe1224ca5d269`. `select` then exited 0, hard-matched all 11 screen identities, cross-checked all rows, and wrote the production manifest: `fpu_dev_corpus_v2_manifest.csv` SHA-1 `84cdd4b45e089a2ebb292491c146ba00bff17ea9`; meta SHA-1 `5501e8b7aa220975d597c0245bf9d74c66d13035`. Independent audit: 120 rows from 69 selected games, at most 2/game, no <12-ply-gap violation, and no game shared between splits.

**Tuning control qualification — scientific stop.** The controls stage ran successfully and immutably on the untouched 80-row tuning split under the production fingerprint (`worktree_clean=true`, `add_noise=false`). `absolute_off` had 1 target lock-in and mean top share 0.390125; `r0` (`FPU=Q_parent`) had 0 target lock-ins and mean top share 0.29534375. Those descriptive improvements did **not** clear the full §6.2 safety table: `r0` changed the selected move to a lower-prior move on **11 of 40 controls = 27.5%**, versus the preregistered requirement `<10%`. Therefore `r0_qualified=false`, with reject reason `control_flip_rate=0.2750>=0.1`. Evidence SHA-1s: `controls_cases.csv` `306167f7e1bb0d3ba26842d5bd960419ac5317a7`; `controls_summary.csv` `d0b0941fa323362aeca1f5afd943fcf60bb629aa`; `controls_gate.json` `198e5387b81b6f513e557efe5df9828078bbba3c`.

**Decision: REJECT the parent-relative context-policy-mass formula family at its prerequisite.** The exit-0 controls run is a valid negative scientific result, not a tooling failure. Per the frozen protocol, do not run or interpret nonzero `r∈{0.10,0.20,0.35,0.50,0.75}`; do not inspect the frozen-check split; and do not run selected-A candidate, pooled collateral, A/B/C/D, or strength gates. No coefficient was selected, no FPU setting was promoted, and no self-play change is authorized. Preserve the production reservoir, screen, manifest, and controls artifacts unchanged as the completed evidence chain.

### Read-only 11-flip tuning-control postmortem (completed 2026-07-24)

**Scope and integrity.** The postmortem replayed the existing 80 tuning rows only, reported all 40 tuning controls, and used no nonzero coefficient. It reproduced exactly **11/40 lower-prior flips = 27.5%**, included **zero frozen-check rows**, and left the manifest, source index, replay data, production screen, and controls evidence byte-unchanged. Two complete runs emitted byte-identical canonical artifacts. The full repository suite after the analysis implementation was **2,274 passed / 4 skipped / 53 deselected**.

**Measured concentration.**

- Phase: opening **5/10**, early-mid **1/10**, midgame **3/10**, late **2/10**.
- Side to move: red **8/20**, black **3/20**.
- Branching band: b300–399 **2/4** and b400-plus **9/36**. The b300 rate is descriptive only because `n=4`.
- The 11 flips occurred across 11 separate games; no game contributed more than one flipped control.

**Flipped versus non-flipped controls.**

| Metric | Flipped (`n=11`) | Non-flipped (`n=29`) |
|---|---:|---:|
| Mean shipped selected-move prior rank | 1.2727 | 4.2759 |
| Mean `r0` selected-move prior rank | 9.0000 | 1.9655 |
| Mean effective-children delta (`r0-shipped`) | −20.0983 | −7.3547 |
| Mean reply-count delta (`r0-shipped`) | −20.1818 | −40.5172 |
| Mean root-value delta (`r0-shipped`) | +0.0080 | +0.0116 |
| Mean top-share delta (`r0-shipped`) | −0.0141 | +0.0106 |

**Interpretation boundary.** Measured: the flipped controls show a large shift away from initially high-prior moves and a larger effective-children loss, with small root-value movement and no flip-specific top-share increase. They do not show unusually strong reply reduction; non-flips reduced replies more. Inference: making `Q_parent` the FPU neutral point altered first-touch move ordering before any positive policy-mass coefficient existed. This justifies retiring the parent-relative baseline and requiring any successor's zero setting to dispatch through the shipped branch exactly. It does not establish that policy mass is beneficial, validate the proposed v17 formula, or provide coefficient-selection evidence.

**Reproduction command and evidence.**

```bash
.venv/bin/python -m scripts.GPU.alphazero.postmortem_fpu_policy_mass_controls
```

Artifact root:
`logs/eval/fpu_v16_policy_mass_v2/diagnostic/production_v2_b400amend_4000g_seed20300000/tuning/postmortem/`

- `control_flip_pairs.csv`: `cc423f5ed29af94be653017e0140edf3baa1aa68`
- `summary.json`: `63c22af1ca068e9c2c7e89ddfa7192299632dd8e`
- `provenance.json`: `cf98f99f1cdd1b5153d16a007003e993205b895c`
- `report.md`: `708401666be5712b5a7ca229f07c45431f7109eb`

---

*Append a new row to the [experiment ledger](#experiment-ledger) and update [do-not-repeat](#do-not-repeat-prevents-going-in-circles) whenever a branch is run and judged. Keep the key conclusion at the top current.*
