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

> **External anchor — SOUGHT, QUALIFIED, RUN AND REJECTED 2026-08-24 (updates the "external anchor" row of the benchmark hierarchy).** An independently trained engine (twixtbot, MIT) was configured to our exact rules and run over 128 games; it won **all of them**, scoring **1.000 against both `0379` and `calib020_0001` at every `trials` setting**, including with no search at all. No setting was unsaturated, so the anchor is **rejected** and **no usable external strength anchor exists**. This does not disturb any internal result: both checkpoints were decisively weaker than twixtbot **under the tested protocol**, an external ORDERING that sibling-vs-sibling Elo could not reveal — not a statement about absolute strength, since the anchor was never calibrated either. See do-not-repeat **#52** and [External anchor — twixtbot](#external-anchor--twixtbot-run-2026-08-24--rejected-no-usable-external-anchor-exists).

> **External-engine learning (twixtbot teacher → policy-head imitation) — RUN 2026-10-09/10; PROVISIONAL VIABILITY, STRENGTH UNRESOLVED.** A separately scoped three-stage trial (card `docs/superpowers/2026-10-08-external-engine-learning-viability.md`; result note `docs/superpowers/2026-10-10-external-engine-learning-result.md`; accepted at `92ab3040`) labelled 192 positions with twixtbot's moves, fitted `calib020_0001`'s policy head only (128 steps; 519 parameters in 8 tensors), then compared on four sealed held-out games and screened in 32 games. **Held-out imitation passed, narrowly:** per-game mean chosen-move CE candidate **3.211** vs scalar-temperature control **3.890** vs incumbent **4.566**; teacher top-1 agreement **25.7%** vs **20.5%**. Four games from one teacher are not broad generalization, and beating the control excludes only a global temperature explanation. **Teacher-rule limitation stands:** the teacher was trained with own-link crossings allowed; labels on positions legal under our rules do not show it chooses good moves under them. **Strength screen inconclusive:** 32/32 games, candidate **12–20**, mean pair score **0.375**, unclipped Hoeffding95 **[0.035, 0.715]** — an unfavourable point estimate whose interval cannot establish inferiority or improvement; NOT_VETOED means only that the disaster veto did not fire. **Keep `calib020_0001`;** the candidate is a research artifact only. Open before any future run: the cleanup exit-code ambiguity and the exhausted seed-selection rule (result note). See do-not-repeat **#53**.

> **Post-closure work, 2026-10-11 → 2026-10-16 — five lines, `calib020_0001` kept throughout** (full entries: [Post-closure work](#post-closure-work-2026-10-11--2026-10-16)). **Search distillation:** Stage A PASS (1,600 vs 400 simulations, mean pair 0.769); Stage B NON-PASS and closed (L2 18/40 jointly, L3 fail). **Search speed:** one-sync 7.8% (bar 10%) not adopted; batched B1600 NOT_FEASIBLE twice (1.945, then 1.445 fixed-shape) — batching path closed. **Implementation finding:** `eval_runner`'s default factory ran BatchNorm in TRAINING mode until `c8cb2aa6`; earlier results stand under their original protocol; the browser's eval-mode ONNX export was unaffected. **Inference-mode match:** corrected 20–12 legacy, mean pair 0.625, inconclusive, closed. **Self-play diagnostic (OPEN):** one missed defence; the losing side's search goes flat (one visit per child) once root value nears −1, because absolute FPU 0 outranks every visited move — the saving move was reached once and never searched deeper. No correction tested yet.

> **2026-10-19 additions (Post-closure entries 6–7).** **The diagnostic example was already lost:** a rules-engine-only check proved that after the save (21,6) red forces a win within three plies (four forcing red moves; every one of black's 486 replies answered by a verified red win). The flat search explains why the save was not deepened, but this example cannot show that a search correction would have saved the game. **`probe_eval._load_network` now returns eval mode** (`444269ac`), closing entry 3's open item. See do-not-repeat **#58**.

> **Mirror consistency (Post-closure entry 8, 2026-10-19, OPEN).** On four diagnostic positions, `calib020_0001` (eval mode) gave clearly different predictions for a position and its left-right reflection: value differences **0.045–0.637**, policy total variation **0.108–0.178**, top move disagreeing in 1 of 4 (tolerance 1e-5). This justifies *considering* an inference-time averaging prototype; it is not evidence of stronger play, and averaging would double inference cost.

> **Mirror-augmentation defect fixed (Post-closure entry 9, 2026-10-19).** Training's left-right mirror (`_mirror_position_lr`) flipped the black edge-connectivity channels 27/28 spatially but never swapped their left/right labels; fixed in `0c417882`. Whether this contributed to the incumbent's asymmetry (entry 8) is **unmeasured**; the incumbent's weights are unchanged and no retraining follows. Symmetry averaging stays an open hypothesis.

> **Mirror-averaging prototype — correctness PASS (Post-closure entry 10, 2026-10-19, OPEN).** A separate, unwired evaluator wrapper averaging the policy distributions and values of a position and its reflection matched an independent engine-rebuilt reference within 3e-8 on four positions (16 inference calls). Correctness only — **no evidence that averaging helps**; a cost-matched strength screen is not yet authorized.

> **Mirror-averaging match — STOP_CONFIGURATION (Post-closure entry 11, 2026-10-19, CLOSED).** 200 averaged vs 400 ordinary simulations, `calib020_0001`, 16 colour-reversed pairs: averaged **19–13** (no draws or caps), **19/32 points < the frozen bar 24**; mean pair score **0.594**, Hoeffding95 [0.254, 0.933] descriptive, pair independence assumed. Network evaluations 323,608 vs 323,983 (approximately matched); search time 696 s vs 840 s. This configuration is stopped. See do-not-repeat **#59**.

> **Evaluation reuse (Post-closure entry 12, 2026-10-19, OPEN).** Across three consecutive 400-simulation searches, 226 of 1,203 evaluator calls (all from an earlier search) repeated an exact input; their inference time was **12.8%** of search time, above the frozen 10% bar. That justifies only a small cache prototype — optimistic (cache overhead excluded), three positions, one timing run, not a route to stronger play.

> **Per-colour prediction cache — STOP_CONFIGURATION (Post-closure entry 13, 2026-10-19, CLOSED).** With each colour's cache kept to its own searches, only **4 of 802** later lookups hit (red; black 0): entry 12's duplicates were almost all **cross-colour**. Results were identical, but combined search time was **2.0% higher** (6.219 s vs 6.097 s), far from the 10% bar. See do-not-repeat **#60**.

> **Mirror-defect exposure (Post-closure entry 14, 2026-10-19, MEASURED — decision pending).** On the four diagnostic replays, the old and corrected training mirrors differ at **8 of 169** positions (4.7%), all in game 2 (the one black win), plies 34–41, once black's chain touched the right edge — about **1.6%** of a buffer at mirror probability 0.5. Exposure only, not benefit; four games. No training or match follows without authorization.

> **Mirror-repair pilot — NOT JUSTIFIED by current evidence (entry 14 decision, 2026-10-19, CLOSED route).** The proposed paired training-and-match pilot was not run: entry 14's small exposure gives too weak a reason for a ~5-hour trial whose null would be hard to interpret. Not a failed training experiment, and not evidence the defect was harmless. The fix (`0c417882`) stays and goes into any independently justified future training. See do-not-repeat **#61**.

> **Root-only sequential-halving variant (Post-closure entry 15, 2026-10-19, OPEN — built, CPU-checked, no games).** A fixed root-allocation prototype (16 candidates, halving rounds, `c_visit` 50, `c_scale` 1.0, zero Gumbel noise, from ply 20) under a narrow, recorded exception to #47; 13 focused CPU tests pass. Prediction on record: **fails the screen**. No model loaded, no seeds accounted, no match approved.

> **Root-only sequential halving — STOP_CONFIGURATION (Post-closure entry 15 result, 2026-10-19, CLOSED).** Halving vs ordinary search at 400 simulations each: **21–1–10**, **21.5/32** against the frozen bar of 24 (mean pair 0.672, interval includes 0.5) — the prediction "fails the screen" matched. Equal network cost (376k vs 378k evaluations). Not a strength finding either way. See do-not-repeat **#62**.

> **Adaptive per-move budget — oracle bookkeeping (Post-closure entry 16, 2026-10-20, MEASURED).** Rules-only chronological credit accounting over 1,135 recorded searches: of 126,800 credits freed by strongly evaluated positions, only **14.5%** reached contested positions before the game ended (+9% budget on average); 85.5% were unused. Oracle classification assumed; not a strength result. Attempt 1 was an execution error. Decision pending.

> **Banked-budget controller — DECLINED (entry 16 decision, 2026-10-20).** Not built or funded: contested moves would have gained only ~9% (436.7 simulations), not the proposed ~2×; the ~24% hypothetical reduction in simulated work is not measured runtime or play preservation. A declined proposal, not a failed experiment; adaptive budgeting generally stays untested. See do-not-repeat **#63**.

> **Training-target truncation — FIXED (Post-closure entry 17, 2026-10-20).** The ordinary training chain defaulted to 512 move slots, below the 528 legal moves of a 24×24 opening, dropping the tail moves' visits and renormalizing the rest; defaults now 576, explicit caps unchanged. Not a regression explanation (the parent inherited it); its effect on trained models is unmeasured.

> **Policy-head capacity pilot — isolated residual head built and CPU-checked (Post-closure entry 18, 2026-10-21, OPEN).** Arm R adds a zero-initialized residual path reading the move cell's 128 encoder features (8,840 trainable parameters); arm C trains only the existing 519. Six CPU checks pass after a float32 fix to one test assertion (first run 4 passed, 2 failed on that test bug). Prediction: none stated in advance. Nothing about capacity is measured; no checkpoint, corpus, search or training is authorized.

> **Policy-head capacity planted qualification — NOT QUALIFIED (Post-closure entry 18 addendum, 2026-10-21, CLOSED).** One run under card rev 3 (`9deaccb9`; evidence `08ced79e`), 26 s, COMPLETED: F1, F2, S2, S3 hold, but **S1 fails** — on the planted R-only target arm C reached KL 0.0496 and R 0.0210, a 0.0286-nat gap below δ = 0.05. The recipe therefore must not be used to measure capacity; no labels, no retuning. Operational only: nothing about whether the head's capacity binds, nothing about strength. The capacity hypothesis remains untested. See do-not-repeat **#65**.

> **Policy-head capacity proposal — SHELVED (Post-closure entry 18 addendum, 2026-10-22, decision).** No qualified, affordable test was established: the planted qualification was NOT QUALIFIED, and a read-only review found its planted target did not establish beforehand that a δ-sized gap between the arms had to exist; no suitable pre-fit guarantee of separation was identified. The capacity hypothesis remains **unresolved**; a frozen-encoder pilot would address only the narrower question of extracting information from existing features, and joint-training capacity and optimization remain open. No further spending; `calib020_0001` stays. See do-not-repeat **#66**.

> **Play-path timing on the new machine (Post-closure entry 19, 2026-10-21, MEASURED).** Mac Studio M5 Max: 1.74 ms per simulation, 0.70 s per ply at 400 simulations (eval mode, committed game positions, no games played, no seeds); about 41.7 s of search per game at entry 15's length, so the methods note's budgets become about 11.8 h (60%) and 47 h (55%) here (derived). Search time only, against entry 15's whole-game 66.6 s: no speed-up between machines is measured or claimed. Label lines derived (1.24 h / 0.62 h, constant cost per simulation assumed); the 205 s G-phase figure stays open (unexplained 3.6 vs 1.1 s per ply).

> **Throughput settings on the new machine (Post-closure entry 20, 2026-10-21, MEASURED).** Self-play: **18 workers at eval batch 14** measured fastest (median 1,207 games/h while all workers are busy; 12 workers 1,038; eval batch 16–48 no faster, 32 and 48 slower; no hang up to 48 in 3 short runs each, which does not show the old hang is gone). Whole-run wall time is dominated by one state-cap game finishing alone (140–484 s for the same setting), so the first sweep's timeouts and its "hang at 16" are void. Labelling: 10–18 workers equal (~3.25x serial); eval batch and stall-flush are never read by the labeller. Operational only; no defaults changed. See do-not-repeat **#64**.

> **Search distillation D2, Stage 1 live measurement (Post-closure entry 21, 2026-10-22, MEASURED).** Stage 0 built an offline trainer that trains `calib020_0001`'s encoder and policy head on its own captured search targets (value head and BatchNorm statistics frozen, value anchor). Stage 1 ran two incumbent self-play games once at 1,600 simulations through the serial capture driver: **COMPLETED in 124 s**; 52 and 54 plies; **1.16 s per ply**; 776 KB captured; the rebuilt records equal the learner rows, and 99 of 106 searches began with inherited child visits. Two games price generation (120 games ≈ 2 h at these lengths; ≤ 10.9 h even if every game reached the 280-ply cap, if the per-ply time holds) but cannot estimate the game-length distribution. Measurement only; the seeds are retired. Stages 2–5 unapproved. See #67.

> **Search distillation D2, Stage 2 corpus generation (Post-closure entry 22, 2026-10-22, COMPLETED).** One run of 120 fresh incumbent self-play games at 1,600 simulations, at Stage 1's exact configuration, in 12 phases of 10: **COMPLETED in 8,370 s (2.3 h) of a 12 h ceiling, 12/12 phases finalized**; 7,020 records (6,403 from the 108 training games, 617 from the 12 held-out games); every rebuilt record equal to the learner row; model digest unchanged. Game lengths 32–280 plies (median 53; two reached the 280-ply cap). The corpus (outside the repository, bound by `corpus.json`) is **eligible for training**. No claim about target quality or strength. Training, Stages 4–5, promotion and pushing unapproved. See #68.

> **Search distillation D2, Stage 3 training (Post-closure entry 23, 2026-10-22, STOP).** One run of the frozen recipe (401 updates of the encoder and policy head on the 6,403 training records; teacher, value head and BatchNorm statistics unchanged): held-out policy cross-entropy fell from 4.824 to 4.147, but absolute value drift from the teacher on the 617 held-out records was **mean 0.103 (limit 0.05) and p95 0.279 (limit 0.15)**, so the verdict is **STOP**. The candidate is kept as a research artifact only. Pre-clip gradient norms exceeded 10,000 on 86 of 401 updates; the cause was not derived. Stages 4–5 are not entered under this card. See #69.

> **Search distillation D2, gradient diagnostic (Post-closure entry 24, 2026-10-22, MISMATCH).** One instrumented replay of Stage 3's 401 updates, from the unchanged incumbent, on the 108 training games only: **updates 0–20 reproduced Stage 3's trace bit for bit; update 21, the first spike, did not** (policy term 4.37707 against 4.36954, anchor term bitwise equal, pre-clip norm 12,854 against 11,404), so the replay stopped there as its card requires. Measured on the replayed update 21: the policy head carried 96% of the squared pre-clip norm (`policy_head.fc.weight` alone 75%), against 2% on update 20. Stage 3's spikes therefore do not replay exactly; the cause of either the spikes or the divergence is not derived. No checkpoint written; no held-out game file opened. See #70.

> **Search distillation D2, reproducibility diagnostic (Post-closure entry 25, 2026-10-22, MEASURED).** Updates 0–21 replayed three times in one process: A (plain), B (attribution at 20 and 21), C (plain), each from a fresh load of the incumbent. **The GPU update path does not repeat.** Pass A reproduced Stage 3 bit for bit through update 21. B and C differed from A from update 0, with the same loss terms but a different pre-clip norm (16.63 against 12.45). They agreed with each other through update 20, and before update 21 they held identical model, Adam and input digests, yet their update-21 norms differed (3,867 against 3,512). At B's update 21, each record's own policy gradient had norm ≈ 3,867, and the 64 summed to about 64 times the batch gradient. The gradient of the stated loss is linear in the record weights (at update 20 the gap is 0.0), so that computed gradient is not the stated loss's gradient. Re-running the policy head step by step gave logits up to 11.5 away from the module's own on the same state. The mechanism is not derived. See #71.

> **Search distillation D2, gradient correctness check (Post-closure entry 26, 2026-10-22, MEASURED).** Two states: the incumbent, and a reconstructed pre-update-21 state whose digest equals entry 25's pass A. **CPU/GPU agreement: no.** The CPU is bitwise repeatable and additive. On the GPU, 59 of 60 gradient tensors fall outside the tolerance at both states; the GPU is not repeatable at S21 (norms 4.32, 834 and 2,900, against the CPU's 4.23); and weighted-record additivity fails. **Independent NumPy reference (policy head): the CPU agrees in all 6 cases and the GPU in none** (about 1e-3 off on small synthetic cases; grossly off on the real batches, where additivity misses by about 10⁴). **First localized discrepancy, from identical inputs:** the encoder's forward differs from the CPU's by 0.19 but repeats on the GPU, and which device is right is not established. Inside the policy head, the conv, BatchNorm, ReLU, gather, masks, log-softmax and CE agree to about 1e-7. The **fc layer (2→64) forward and input gradient**, and the **out layer (64→1) input gradient**, are outside the tolerance and not repeatable. The mechanism is not derived. See #72.

> *Correction to the entry 26 line (2026-10-22, the same day, on rechecking the evidence):* "agree to about 1e-7" should read "agree within 1.5e-5 of each tensor's scale". The real-batch figures are corrected in the entry's own correction note.

> **Search distillation D2, linear-layer reproducer (Post-closure entry 27, 2026-10-23, REPRODUCED IN ISOLATION).** The policy head's `fc` (2→64) and `out` (64→1) operations were run alone on the saved fixtures' float32 operands, against NumPy float64. **The failure reproduces in isolation** at all four original shapes, in all three implementations (`nn.Linear`, explicit matmul with autodiff, and explicit products), with the CPU clean. **The gross errors are confined to exactly the first N − 32,736 rows** (rows 0–863 of 33,600 at S0; rows 0–991 of 33,728 at S21), legal and padded rows alike. They appear in `fc`'s forward and input gradient and in `out`'s input gradient. In `fc`'s forward those rows equal the bias alone. **At 32,768 rows and below, nothing is gross.** The alternative layout and evaluation context reproduce too. Other rows carry a GPU error of about 1e-3 of scale (the CPU's is about 1e-7). The mechanism is not derived. See #73.

> **Search distillation D2, synthetic row-count probe — UNPLANNED GPU EXECUTION during mutation controls (Post-closure entry 28, 2026-10-23, INCIDENT; the approved run NOT made).** Under the injected defect "gate unchecked", a closed-gate test with no device or path isolation reached the real entry. The full frozen probe ran once on the GPU inside the disposable control worktree. Its evidence was deleted with that worktree. It left one read-only `compact_example.npz` (SHA-256 `7c412fd3…`) in `~/twixt-cache/search-distillation-d2/mlxrows-20261023T122440Z/`. That file is not evidence of the approved run and is not interpreted. Every test is now isolated, and the control harness stops if `~/twixt-cache` changes. The approved invocation waits for the user's decision. See #74.

> **Search distillation D2, historical exposure audit (Post-closure entry 29, 2026-10-23, READ-ONLY).** Until `ffe6b342` (2026-10-20), the trainer capped the policy-head move width at 512. Every `checkpoints/` run therefore had at most 64 × 512 = 32,768 rows, including every ancestor of the incumbent. Those runs are **insufficient records**: machine and MLX version per run were not recorded, and 32,768 was measured clean only on the M5 Max under 0.30.3. **Confirmed matching shapes:** D2 Stage 3 updates 0–21, 33,600–33,792 rows, derived from the recorded batch identities. **Possible exposure:** Stage 3 updates 22–400 and D2's evaluation `forward()`. A delegated width count read the held-out records' legal-move counts; this is disclosed and the figure is unused. Not a finding of corruption, nor a statement on measured strength. See #75.

> **Search distillation D2, synthetic row-count probe (Post-closure entry 30, 2026-10-23, COMPLETED; predictions 1 and 4 NOT MATCHED).** The first retained complete record, not the first execution: the same frozen probe ran once unrecorded in entry 28, and its stray example had already shown a gross GPU failure. On synthetic operands, 12 of 27 cases are gross on the GPU — fc, out and the wide 64→64 control at 32,769, 32,800, 33,600 and 33,728 rows — and none at ≤ 32,768 or on the CPU. The gross rows are the leading 32, 64, 864 and 992: **at 32,769 the band is 32 rows, not the predicted N − 32,736 = 33**. Gross outputs are **not repeatable** (repeat 0 differs from repeats 1–2); clean outputs repeat bitwise. dW, db and out Y stay correct. No universal boundary, safe chunk size or mechanism is established. Carries dated corrections to entries 28 (wide control: none stated) and 29 (the 576 cap permits, not forces, more than 32,768 rows). See #76.

> **Search distillation D2, full-float32 setting validation (Post-closure entry 31, 2026-10-23, COMPLETED; CANDIDATE for the isolated operations).** A fresh process launched with `MLX_ENABLE_TF32=0` ran the unchanged synthetic probe and entry 27's four saved cases. On the GPU: no evaluation outside the 1e-4 tolerance (Y and dX now ≤ 3.9e-7 of scale, from about 1e-3), no gross rows at any size, every output bitwise repeatable, in both sets. The CPU and the ungated GPU outputs (dW, db, out Y) are unchanged to the bit, as predicted. A candidate workaround for the isolated operations only: the policy head, full-model gradients and training are not validated, and the cause is not established. See #77.

> **Search distillation D2, full-float32 setting through the policy head and full gradient path (Post-closure entry 32, 2026-10-23, COMPLETED; FAIL at phase 2).** A fresh process with `MLX_ENABLE_TF32=0`. Phase 1, the policy head against the independent NumPy reference: PASS in all six cases (worst 0.05 of the tolerance). Phase 2, the full model at the incumbent against the CPU: the update-21 batch passes everywhere; on the update-0 batch, 2 of 60 gradient tensors (`encoder.blocks.1` and `.2` `conv2.weight`) differ from the CPU by 2.05 and 1.57 × entry 26's 1e-3 tolerance, repeatably. Everything else is within it, and entry 26's gross, non-repeatable GPU gradients are gone. FAIL under the frozen rule: phase 3 (21 updates) did not run. The setting is not validated for the full gradient path; which backend is nearer the exact gradient is not established. Both CPU predictions matched. See #78.
> **Search distillation D2, convolution-reference diagnostic (Post-closure entry 33, 2026-10-23, COMPLETED; both targets INHERITED).** Same setting, at the incumbent on the update-0 batch, no optimizer update. Entry 32's six failures reproduced bit for bit, and the operand capture was bitwise faithful (L2 included). Given identical operands, the CPU and GPU convolution weight-gradient kernels agree **bitwise**, within 0.006 of the tolerance of float64. Float64 alone, fed each device's own operands, reproduces the miss (2.05×, 1.57×): it is carried in by the upstream gradient (GPU against CPU 5.1× and 5.7× the tolerance, at 2 and 1 elements, all in channel 42), not made by either kernel. Which device's whole-model gradient is nearer the exact one is not established; entry 32's FAIL stands. See #79.
> **Search distillation D2, encoder branch trace (Post-closure entry 34, 2026-10-23, COMPLETED; BRANCH_CHANGE).** Same setting, incumbent and batch, no update. Reproduced bitwise; the trace was bitwise faithful; every local forward and backward step (convolutions, BN, residual, ReLU) is correct on both devices against float64 from their own operands. 14 ReLU elements with inputs within about 1e-6 of zero take different branches on the CPU and GPU, each following from its own device's operands; among them is block 2's outer ReLU at `[1, 13, 4, 42]` (CPU off, GPU on). Those flips alone account for both target gradients (0.006× and 0.004× the tolerance with them, 2.05× and 1.57× without). A numerical branch change, not an incorrect derivative; not shown harmless, and which device is more accurate is not established. Entry 32's FAIL stands; next is the user's qualification decision. See #80.
> **Full-float32 setting — qualification decision (decision note after Post-closure entry 34, 2026-10-23, the user's).** Entry 34's BRANCH_CHANGE accepted; no further localization. `MLX_ENABLE_TF32=0` is required for authorized GPU work on this machine and runtime (per process; no persistent change). Validated within the tested scope: isolated operations, the policy head, the incumbent-state local derivatives. **Training stays unqualified**: the updated-state check never ran, and neither the branch change's harmlessness nor training reliability is established. Entry 32 remains a FAIL. OPEN: a DRAFT card for completing the 21-update diagnostic prefix under a prospective criterion (`8b255801`), awaiting review; every gate closed. See #81.
> **Full-float32 setting — 21-update prefix card (addendum to the decision note, 2026-10-23).** Reviewed, amended and FROZEN; built in `ad873a12` (gate 44, closed; 39 + 16 tests; regression = the 58-id baseline). With both labels on the CPU, a full real-state run qualifies, with every reference passing. NOT RUN: execution awaits the user's review of the frozen implementation. Supersedes the decision note's "DRAFT" status line above. See #81.
> **Search distillation D2, full-float32 setting: the 21-update diagnostic prefix (Post-closure entry 35, 2026-10-23, COMPLETED; QUALIFIED_AT_U21).** The frozen card (`ad873a12`) ran once with `MLX_ENABLE_TF32=0` in 35 s. Entry 32's incumbent lines and teacher digest were reproduced bitwise (prediction 1 matched). Two GPU prefixes of 21 updates were equal at every update on scalars, model and Adam digests, all finite. At U21, on the update-0 and update-21 batches: repeatable, additive, forward agreement within the tolerance, every local, parameter (52, with L2) and policy-head reference passing. Batch 0's two failing native gradients (`blocks.3` conv1 1.49×, conv2 1.24×) were explained by near-zero branch flips; batch 21 had none. **Scope: that state and those two batches only.** Training remains unqualified; no branch change is shown harmless; entry 32 remains a FAIL. Supersedes the addendum's "NOT RUN" line above. See #82.
> **Search distillation D2 — bounded learning trial (decision note after Post-closure entry 35, 2026-10-23, the user's).** Entry 35 accepted; no further diagnostics are the next task. OPEN: a DRAFT successor card (`2026-10-23-search-distillation-d2-corrected-trial-card.md`) awaiting review. Its terms: unchanged incumbent and Stage 3 recipe and criteria, `MLX_ENABLE_TF32=0` in every process, 108 training games reused, 12 fresh held-out games, one candidate, no retries; strength evaluation only after a pass. Nothing generated, registered or trained; no gate. GPU-training reliability in general is not established. See #83.
> **Search distillation D2 — bounded learning trial: card FROZEN and built (addendum to the decision note after Post-closure entry 35, 2026-10-24).** Card `9f2c6096…`, build `d7d458ad` (gates 45 and 46, closed; regression = the 58-id baseline). The 12-seed held-out block [202692300, 202692312) is surveyed CLEAN, proven before and after registration, and registered in ACCOUNTED. Outcome prediction: None stated in advance. NOT RUN: generation awaits the user's review. Supersedes the "DRAFT" status in the line above. See #83.
> **Search distillation D2 — bounded trial CLOSED: held-out generation INTERRUPTED (Post-closure entry 36, 2026-10-25).** A macOS kernel panic rebooted the machine during game 1 of 12 (2 started, 1 finished). There is no wrapper record and the exit status is unknown; the set is incomplete and ineligible. Seeds: [202692300, 202692302) EXPOSED and the block RETIRED, through an amended reboot derivation (16/16). There is no learning result: the panic's cause is not established and the hypothesis is not rejected. Training was never run. Supersedes the line above. See #84.
> **Search distillation D2 — bounded trial ATTEMPT 2 prepared (decision note after Post-closure entry 36, 2026-10-25).** One fresh attempt, by the user's decision. Successor card FROZEN (`bde310a3…`; records and paths only). Build `e3f90078`, gates 45 and 46 closed. Fresh block [202692400, 202692412): surveyed CLEAN, proven before and after registration, registered. No new kernel panic found. A second panic stops GPU experiments pending a system-stability decision. Outcome prediction: None stated in advance. NOT RUN: generation awaits approval of the frozen plan. See #85.
> **Search distillation D2 — bounded trial ATTEMPT 2: held-out generation COMPLETED and eligible (Post-closure entry 37, 2026-10-25).** One approved run (gate 45 only): 12 games, 660 records, 1,086 s of the 2-hour ceiling; in-run reader and alignment checks passed; model digest unchanged; `MLX_ENABLE_TF32` "0" at start and end. Seeds [202692400, 202692412) EXPOSED and RETIRED whole (15/15). No new kernel panic before or after. Regression = the 58-id baseline. Training NOT RUN: gate 46 closed, awaiting approval. Outcome prediction: None stated in advance. See #86.
> **Search distillation D2 — bounded trial attempt 2: training STOP; the trial is CLOSED (Post-closure entry 38, 2026-10-25).** One approved run (gate 46 only), 401 updates under `MLX_ENABLE_TF32=0`, 46 s. Held-out policy CE 4.7878 → 4.1627 (strictly lower: met), but held-out value drift mean 0.1207 (limit 0.05) and p95 0.3433 (limit 0.15) both FAILED. The answer to the card's question is no: the unchanged recipe under TF32=0 did not pass Stage 3's held-out checks. No retry or tuning; no strength screen. `calib020_0001` stays the incumbent. Outcome prediction: None stated in advance. Supersedes the two lines above. See #87.
> **Dual-encoder policy branch — design-only feasibility card (decision note after Post-closure entry 38, 2026-10-26, the user's).** Entry 38's STOP and closeout accepted: it closes the recipe, not the goal of stronger play. Authorized: a design-only card for a composite agent (the frozen incumbent supplies values; a trainable encoder and policy-head copy supplies priors), so that the value function is preserved by construction. DRAFT `2026-10-26-dual-encoder-policy-branch-feasibility-card.md`: two forward passes per leaf (2.00× FLOPs); compute-fair comparison at matched network evaluations, composite 200 vs incumbent 400 simulations; recommends REJECTING on paper. Nothing loaded, generated, trained or gated. See #88.
> **Dual-encoder policy branch — REJECTED on paper (addendum to the decision note after Post-closure entry 38, 2026-10-26, the user's).** Step 1 not built; nothing executed (no model loaded; no seeds, games, training or gates). The value-preserving architecture's added inference cost is not justified by current evidence: this experiment has not earned its cost. This is not a result about dual encoders, value preservation, capacity, distillation or strength. The card is kept as the reviewed proposal (§8 records the decision). `calib020_0001` stays the incumbent; the stronger-play goal remains open with no supported next experiment. Supersedes the "DRAFT" line above. See #89.
> **Mirror averaging + root sequential halving — one fixed combination, design-only card (decision note after the dual-encoder rejection, 2026-10-26, the user's).** After a read-only review, the user authorized a DRAFT card (`2026-10-26-mirror-averaging-plus-root-halving-combination-card.md`). It proposes: the averaged evaluator at 200 simulations, ordinary opening through ply 19, the unchanged halving rule from ply 20, against the unchanged incumbent at 400, both under `MLX_ENABLE_TF32=0`, at approximately matched network evaluations (guards 1.05× evaluations, 1.10× search seconds); a composition-and-cost check, a 32-pair gross-regression screen and a 509-pair strength comparison (pass > 0.56020). It also proposes a narrow reopening of #47, #59 and #62 for this combination only. The two screens' scores are not current baselines and are not to be added. Nothing built, loaded, seeded or run. Prediction: None stated in advance. See #90.
> **Mirror averaging + root sequential halving — card FROZEN and step B built (addendum to the decision note, 2026-10-26).** The user approved the fixed combination, the narrow #47/#59/#62 exceptions (this candidate and its specified controls only) and step B; clarifications are in card §11. Build `0ad4de35`: a shared module (game loop copied verbatim from entry 15's match), the C1 entry with gate 47 and its 900-second wrapper, the screen/strength entry with gates 48 and 49 (their wrappers follow C1); 49 gates, all closed. 82 focused CPU tests pass. The CPU loop checks failed once on a defect in the check script (`0e134482`, kept) and passed on run 2 (`58924743`). Regression = the 58-id baseline. Nothing loaded, no GPU search, seeds or games. C1 awaits the user's review of the frozen build. Supersedes the "DRAFT" line above. See #91.
> **Mirror averaging + root sequential halving — step B accepted; C1 approved once (addendum, 2026-10-26, the user's).** Read-only review found no blocking defect. Accepted: deferring the screen/strength wrappers, plans and seed copies until after C1; the disclosed loop-check repair (no retry permission for C1 or later). C1 may run once (gate 47 only, TF32=0, 890 s + 10 s). A STOP, VOID or timeout closes the combination. The screen, strength comparison, seeds and games remain unauthorized. See #92.
> **Mirror averaging + root sequential halving — C1 PASS (Post-closure entry 39, 2026-10-26).** One run, 216 s of 900 s, gate 47 restored. Entry 10's check PASS; identity controls 40/40 and 64/64 bitwise; halving at all 64 positions from ply 20. Ratios of totals: evaluations 0.9961 (≤ 1.05), search seconds 0.8245 (≤ 1.10). Derived search-only projections at an assumed 59.8 plies: screen ≈ 0.65 h, strength ≈ 10.3 h. 🔴 Launched while NOT idle (load 4.0, Spotlight at ~313% CPU; check printed but not waited on), so the absolute timings may be inflated; not repeated. No strength claim. The screen, strength comparison and seeds remain unauthorized. Prediction: None stated in advance. See #93.
> **Mirror averaging + root sequential halving — screen/strength build only (addendum to entry 39, 2026-10-26).** C1 accepted with its disclosed idle deviation (timings neither rerun nor discounted). Ceilings: screen 3,496 s, strength comparison 55,601 s (TERM 3,486 / 55,591 + 10 s). Built (`b6632003`): an idle gate the launch depends on (fake launches: busy → no launch; idle → the exact command), both wrappers, and seed survey and closing copies (64 / 1,018 slots; blocks unselected, no plan committed). The match now logs `game_start` for closing. Nothing run; 49 gates closed. See #94.
> **Mirror averaging + root sequential halving — the screen's seed survey and pre-proof (addendum to entry 39, 2026-10-26).** Each run once in a clean worktree. The survey CLEAN (`28cabf0e`, 45/45): block [202692500, 202692564), 64 seeds, gap 82 (floor 64). The pre-proof HOLDS (`8c52583d`, 45/45). Not registered; no plan; nothing played. Awaits registration review. See #95.
> **Mirror averaging + root sequential halving — the screen block registered and proven (addendum to entry 39, 2026-10-26).** Registration review passed (`eeaac8cf`). [202692500, 202692564) appended last in ACCOUNTED (43 → 44) with the screen's `plan.json` = `build_plan` (`aa977403`). The post-registration proof, run once in a clean worktree, HOLDS (`4dc418db`, 45/45). Accounted, not exposed or retired; gate 48 closed; nothing played. Awaits screen approval. See #96.
> **Mirror averaging + root sequential halving — SCREEN CONTINUE (addendum to entry 39, 2026-10-26).** One run (`59bcd53d`), COMPLETED in 2,271 s of 3,486 s; gate 48 restored. 32 pairs / 64 games on [202692500, 202692564), all started: candidate 36, incumbent 27, 1 draw (state cap). Mean pair score 0.5703 ≥ 0.40, so CONTINUE: a gross-regression check only, not a strength result (descriptive Hoeffding95 [0.330, 0.810] includes 0.5). Guards held: evaluations 0.9956, search seconds 0.8158. Block not yet closed; awaits result and closing review. Prediction: None stated in advance. See #97.
> **Mirror averaging + root sequential halving — screen accepted and CLOSED (addendum to entry 39, 2026-10-26).** Result accepted: CONTINUE, encouraging but inconclusive; stronger play not demonstrated. K1 RECONCILES (`adbb31b2`, 15/15: all 64 started); K2 (`52d179da`) appended [202692500, 202692564) last to EXPOSED (41 → 42) and RETIRED (39 → 40), ACCOUNTED 44 unchanged. Lifecycle tests re-anchored (`00c0d7b4`, 184 focused passed). The strength comparison (ceiling 55,601 s) awaits a separate decision. See #98.
> **Mirror averaging + root sequential halving — the strength comparison's seed survey and pre-proof (addendum to entry 39, 2026-10-26).** Each run once in a clean worktree. Survey CLEAN (`2ccc9037`, 45/45): block [202693600, 202694618), 1,018 seeds, gap 1,030 (floor 1,018). Pre-proof HOLDS (`9631ec72`, 45/45). Not registered; no plan; gate 49 closed; nothing played. Awaits registration review. See #99.
> **Mirror averaging + root sequential halving — the strength block registered and proven (addendum to entry 39, 2026-10-26).** Registration review passed (`c665e938`). [202693600, 202694618) appended last in ACCOUNTED (44 → 45) with `plan.json` = `build_plan` (`0c1c18b5`). The post-registration proof, run once in a clean worktree, HOLDS (`5c4b60a7`, 45/45). Test re-anchored (`5e07bba4`, 184 focused passed). Accounted, not exposed or retired; gate 49 closed; nothing played. Awaits run approval. See #100.
> **Mirror averaging + root sequential halving — the strength comparison → TIMEOUT / VOID, no score (addendum to entry 39, 2026-10-27).** Run once (`264df589` → `28737707`): TERM at 55,591 s with 799 of 1,018 games ended; record VOID (terminated), verdict and guards null; outcomes not read. Pace 0.61 s/move until a BOINC client started mid-run (19:55 EDT) took the idle machine; then up to 2.2 s/move. Gate 49 restored and read back; 49 gates closed; no panic. Stronger play not demonstrated. Seed closing not yet authorized. See #101.
> **Mirror averaging + root sequential halving — the void strength block closed; one new attempt approved in principle (addendum to entry 39, 2026-10-27).** User's decision recorded (`376b75a3`). K1 RECONCILES once (`8ad03a26`, 15/15); K2 (`21a2c6f0`) EXPOSED [202693600, 202694400) (800 started), RETIRED [202693600, 202694618) whole; test re-anchored (`214063bb`, 184 passed). Partial games unscored. New attempt: same 509 pairs, fresh seeds, unchanged rule and guards, BOINC suspended until completion; its seeds, build and run each await approval. See #102.
> **Mirror averaging + root sequential halving — strength attempt 2 BUILT, for review (addendum to entry 39, 2026-10-27).** Approval `d687a1fc`; build `d39aa58e`. Pinned amendment; gate 50 closed (50 gates, all closed); `run strength2` on its own paths, otherwise the strength comparison unchanged. A read-only BOINC check (executable path or BOINC account; podman VM exempt) runs before launch and after every game: detection or a failed check refuses the launch or VOIDs. 40 new tests; 271 focused passed. No seeds chosen; nothing run. See #103.
> **Mirror averaging + root sequential halving — strength attempt 2 seeds prepared, ready to launch (addendum to entry 39, 2026-10-27).** Approved as one sequence (`b075bfaa`); every check passed. Survey CLEAN (`393c4e27`, 45/45): [202695700, 202696718), gap 1,076. Pre-proof HOLDS (`fa514913`). Registered last in ACCOUNTED, 45 → 46, with `plan.json` = `build_plan` (`827641cc`). Post-proof HOLDS (`828e39e4`). Test `7cd199b2`, 271 focused passed. Gate 50 closed; nothing played. Awaits launch approval. See #104.
> **Mirror averaging + root sequential halving — strength attempt 2 → COMPLETED, PASS (addendum to entry 39, 2026-10-28).** Run once (`babbd858` → `f1f4e2da`, approval `c04cde0b`): all 1,018 games on [202695700, 202696718) in 36,124 s; mean pair score 0.6793, Hoeffding lower 0.6191 > 0.5; guards 0.9975 (evaluations) and 0.8202 (search seconds), compute matched. 1,020 background checks, all CLEAR. Gate 50 restored and read back; no panic. Under this protocol, the fixed combination beats the incumbent within the declared cost allowances; this shows neither synergy nor either component's benefit. Seed closing, deployment, training and pushing not authorized. See #105.
> **Mirror averaging + root sequential halving — attempt 2's PASS accepted; block CLOSED (addendum to entry 39, 2026-10-28).** The user's independent recomputation matches (`7bacc265`): stronger play at comparable compute for this fixed combination under the tested protocol, assuming independent pairs; no synergy or component claim; no further match needed for the frozen question. K1 RECONCILES (`af838eb2`, 15/15: all 1,018 started); K2 (`b563792d`) appended [202695700, 202696718) last to EXPOSED (43 → 44) and RETIRED (41 → 42), ACCOUNTED 46 unchanged. Lifecycle test re-anchored (`3ffc9924`, 174 focused passed). Attempt 1 stays VOID and unscored. Deployment, training and pushing not authorized. See #106.
> **Combination policy pilot — card FROZEN; step 1 (build only) approved (decision note after entry 39, 2026-10-28).** One round of policy-only learning from the proven agent's own halving decisions: weights-only `calib020_0001`, encoder, value head and BatchNorm statistics frozen, arm R's residual policy branch trained on the CPU toward a completed-Q target within halving's 16 candidates, the objective the mirror-averaged policy. After two reviews the user froze the card (`docs/superpowers/2026-10-28-combination-policy-pilot-card.md`) and approved only the build, its focused CPU tests and a 20-update CPU timing on synthetic inputs, with a narrow #66 exception (reuse of the residual head, no capacity claim). A distinct recipe: continuation (#48, #49) and D2 (#69, #87) stay closed. The target is the main uncertainty and is not validated by the combination's PASS. Seeds, checkpoint, GPU, corpus, real-data training and games remain unauthorized. See do-not-repeat **#107**.
> **Combination policy pilot — step 1 BUILT and the CPU timing → COMPLETED (entry 40, 2026-10-28).** Build `b1ad8f00` + `c0bf2452` (52 focused CPU tests; three gates created closed, 50 → 53). The one-shot timing ran once at `c0bf2452` with BOINC quit and no BOINC process before or after: 20 synchronized updates at the real architecture on synthetic inputs, random weights, CPU, mean 0.487 s per update (range 0.481–0.498). Projected training (3,000 updates plus the pre-fit, held-out and value passes over the card's estimated 37,000 / 3,700 rows) ≈ 1,799 s; step 4's ceiling is therefore **45 min** (1.5×). Loading and the corpus replay are not timed. Synthetic timing only: no checkpoint, no real data, nothing about the target. See do-not-repeat **#108**.
> **Combination policy pilot — step 1 ACCEPTED; the throughput wrapper and its seed copies approved to build (decision note after entry 40, 2026-10-28).** The user's review found no material blocker in the build or the timing; the timing record stands (its load readings above 3 do not invalidate it under its declared checks; no rerun). Approved: build only the throughput check's wrapper and its seed-survey/proof and closing copies, with focused CPU checks, for review before any seed procedure or GPU work. The 144-in-15-minute rule stands: a timeout rejects this setup's operational feasibility, not policy learning. See do-not-repeat **#109**.
> **Combination policy pilot — the throughput check's wrapper and seed copies BUILT, for review (addendum to the decision note after entry 40, 2026-10-28).** Wrapper, survey/proof copy and closing copy, each a copy of strength attempt 2's with named differences: the 15-minute ceiling (TERM 890 s + 10 s), the run's process group counted for survivors (18 spawned workers do not carry the module's name), and the card's closing rule for parallel workers (started slots a prefix as a set, the claim counter equal to k, else the whole block exposed). 20 new focused CPU tests. Nothing run: no seed procedure, gate, checkpoint or GPU. See #109.
> **Combination policy pilot — the build reviewed; the throughput seed survey approved once; a supervisor repair approved (addendum, 2026-10-29, the user's decision).** The seed scripts are ready. One wrapper defect must be fixed before any launch: the supervisor cancels its alarm when the main process exits, so a worker that ignores TERM is never KILLed and the 900 s ceiling is not enforced. Approved: that repair with one fake-launch test, and the survey once in a clean worktree, its record committed for review before the pre-proof. Pre-proof, registration, post-proof, the gate, the checkpoint and the GPU remain unauthorized. See do-not-repeat **#110**.
> **Combination policy pilot — throughput seed survey → CLEAN; the supervisor repaired (addendum, 2026-10-29).** Survey run once (`cd4217a9`, clean worktree at `fea251fa`): block [202696900, 202697044), 144 seeds, controls 45/45, nearest gap 176 (floor 144); not registered, no plan. The supervisor now keeps the ceiling armed until the whole process group has gone, so a TERM-ignoring worker is KILLed; the new fake-launch test failed before the repair and passes after. Awaiting review; pre-proof and every later step unauthorized. See #111.
> **Combination policy pilot — survey accepted; pre-proof approved once; post-KILL wait to be bounded; TIMEOUT overrides COMPLETED (addendum, 2026-10-29, the user's decision).** The pre-registration proof runs once on the committed survey record and the unchanged block, its record returned before registration. The supervisor's wait after KILL must be bounded and report survivors or unconfirmed cleanup (minimal repair and one CPU check). An outer TIMEOUT closes the pilot even over a script COMPLETED record; it cannot admit generation (not yet enforced by `admitted`). Registration and later steps unauthorized. See #112.
> **Combination policy pilot — throughput pre-proof → HOLDS; the post-KILL wait bounded (addendum, 2026-10-29).** Pre-proof run once (`086d6e37`, clean worktree at `7aa0aea8`): bound to the survey record, block [202696900, 202697044) unregistered, no plan, controls 45/45. The supervisor stops watching 10 s after KILL and exits 125 (TIMEOUT, cleanup unconfirmed in survivors); the wrapper never waits past 910 s. The new test hung on the previous wrapper and passes now. Awaiting review; registration and later steps unauthorized. See #113.
> **Combination policy pilot — pre-proof and bounded supervisor accepted; registration with its plan, and the admission fix, approved (addendum, 2026-10-29, the user's decision).** Register exactly [202696900, 202697044) in ACCOUNTED with the plan from `build_plan`; separately, generation admission must require the committed `outer.json` too (zero exits, gate restored, survivors confirmed empty), refusing missing outer evidence, TIMEOUT or unconfirmed cleanup whatever the inner record says. Both return for review; the post-proof, gate, checkpoint, GPU, closing and pushing are unauthorized. See #114.
> **Combination policy pilot — throughput block REGISTERED with its plan; generation admission requires the outer record (addendum, 2026-10-29).** [202696900, 202697044) appended last to ACCOUNTED (`88c5d162`; AST: only ACCOUNTED changed), plan = `build_plan` bound to `dbe3e91c…`; registry tests unchanged from the parent (the same older failures, none new). `admitted("generate")` now also requires the committed `outer.json` (zero exits, gate restored, survivors empty, script record, same head); before, an inner COMPLETED alone admitted. Awaiting review; the post-proof and later steps unauthorized. See #115.
> **Combination policy pilot — registration, plan and admission fix accepted; the post-proof approved once (addendum, 2026-10-29, the user's decision).** The post-registration proof runs once in a clean worktree against the committed registration and unchanged plan; its create-only record is committed with the lifecycle-test update and focused CPU checks, then returns for review before the gate. The gate, checkpoint, GPU launch, closing and pushing remain unauthorized. See #116.
> **Combination policy pilot — the throughput post-proof HOLDS (addendum, 2026-10-29).** Run once at `d28f5baf` in a clean worktree (`20486339`): the block CLEAN, the state as expected (ACCOUNTED = the survey's + the block last, nothing else changed, plan = `build_plan`), controls 45/45, exit 0. The lifecycle test now expects it and pins the proved registry to the current one. Awaiting review; the gate, checkpoint, GPU launch, closing and pushing remain unauthorized. See #117.
> **Combination policy pilot — post-proof accepted; ONE throughput launch approved (addendum, 2026-10-29, the user's decision).** After a recorded approval, a fresh read-only panic/reboot check and a recheck that the registry and plan are unchanged, open only the throughput gate (one-line commit, clean worktree) and run the existing wrapper once, BOINC quit, both launch checks enforced; ceiling TERM 890 s / KILL 900 s / cleanup observation to 910 s. No retry, extension or top-up; outer TIMEOUT overrides inner COMPLETED and closes the pilot. Closing, corpus generation, real-data training and pushing remain unauthorized. See #118.
> **Combination policy pilot — the throughput check: outer TIMEOUT; the pilot CLOSED at step 2 (entry 41, 2026-10-29).** The one approved run hit the 890 s ceiling (inner 124, TERM sufficed; script record VOID "terminated"): 131 of 144 slots started, 113 finished, about 457 finished games/h against the ~576 the ceiling needed. Gate restored, no survivors, generation refused. No policy-learning result; the hypothesis is neither supported nor rejected. Seed closing awaits approval. See #119.
> **Combination policy pilot — TIMEOUT/VOID and closure at step 2 accepted; seed closing approved as one conditional sequence (addendum, 2026-10-29, the user's decision).** No policy-learning result; the combination's earlier strength PASS stands. K1 once in a clean worktree; only if it RECONCILES, all of [202696900, 202697044) to EXPOSED and RETIRED (no final `claims` event, so the frozen rule exposes all 144); then the lifecycle-test re-anchor and focused CPU checks. See #120.
> **Combination policy pilot — the throughput block CLOSED (addendum, 2026-10-29).** K1 RECONCILES 21/21 (`b69515d5`; no `claims` event); K2 (`e30e277c`) appends [202696900, 202697044) last to EXPOSED (44 → 45) and RETIRED (42 → 43), ACCOUNTED (47) and TEST_ONLY unchanged; the lifecycle test asserts the closed state (`3dee44a9`). The pilot is closed with no policy-learning result. Awaiting review; nothing further authorized. See #121.
> **Combination policy pilot — closeout ACCEPTED (addendum, 2026-10-29, the user's review).** The pilot and its seeds are fully closed. Policy learning remains untested; the combination's earlier strength PASS stands. Nothing further is authorized. See #121.
> **Combination policy pilot — ONE fresh attempt at the throughput check: amendment and build only (decision note after entry 41, 2026-10-29, the user's decision).** Attempt 1's 15-minute ceiling was too tight (113 games in ~890 s, roughly 19 minutes for 144). Attempt 2: the same 144 games, model, search, 18 workers and cap 14; TERM 1,790 s, KILL 1,800 s, observation to 1,810 s; fresh seeds, separate paths, gate 54 closed; generation admitted from attempt 2's committed inner and outer records only. Built for review; nothing run. See #122.
> **Combination policy pilot — attempt 2's build ACCEPTED; the seed survey and pre-proof approved once each (addendum, 2026-10-30, the user's decision).** "No further attempt" (amendment, #122) means one attempt now with no automatic retry; after a TIMEOUT or interruption the cause, cost and worth of another attempt are assessed, and any further attempt is the user's fresh decision (#123). The pinned amendment is unchanged. Registration, GPU work, games, training and pushing remain unauthorized.
> **Combination policy pilot — attempt 2's seed survey → CLEAN; pre-proof → HOLDS (addendum, 2026-10-30).** Survey run once (`df0c0ec9`, clean worktree at `a3a54485`): block [202697200, 202697344), 144 seeds, gap 150 (floor 144), controls 45/45. Pre-proof run once (`fcd3969d`, clean worktree at `df0c0ec9`), bound to the survey record (sha256 `26ef8470…`): the state HOLDS, the block CLEAN, 45/45. Not registered; awaiting review.
> **Combination policy pilot — attempt 2's survey and pre-proof ACCEPTED; registration, plan and post-proof approved as one conditional sequence (addendum, 2026-10-30, the user's decision).** Append exactly the surveyed block to ACCOUNTED and create the plan via `build_plan`; only if both are exactly as expected, run the post-proof once. Stop on any difference. The gate, GPU work, games, training and pushing remain unauthorized.
> **Combination policy pilot — attempt 2's block REGISTERED with its plan; post-proof → HOLDS (addendum, 2026-10-30).** [202697200, 202697344) appended last to ACCOUNTED (47 → 48), nothing else changed; plan = `build_plan` (`22f81be1`). Post-proof run once (`1c2f4220`), bound to the survey record: the state HOLDS, the block CLEAN, 45/45. Lifecycle test re-anchored; 243 passed. Gate closed; awaiting review before it is opened.
> **Combination policy pilot — attempt 2's registration and post-proof ACCEPTED; the ONE attempt-2 launch approved with its closing (addendum, 2026-10-30, the user's decision).** Panic/reboot check and registry/plan recheck; only gate 54 opened in a clean worktree; the existing wrapper once (idle + BOINC checks, TF32=0, TERM 1,790 s / KILL 1,800 s / observation to 1,810 s). After a started run: K1 once; only if it RECONCILES, its exposure verdict and the whole block retired. No automatic retry. Generation, training and pushing remain unauthorized. See #124.
> **Combination policy pilot — the throughput check, ATTEMPT 2: outer TIMEOUT; VOID; block closed (entry 42, 2026-10-30).** All 144 games started and 143 finished; slot 108 (started at 711 s) was in flight at the 1,790 s TERM (inner 124; record VOID; gate restored; no survivors). The bulk ran at attempt 1's pace; four 280-move state-cap games took 686–1,091 s, and that tail, not the mean, sank the run. K1 RECONCILES 21/21 (no claims event); K2 EXPOSES and RETIRES [202697200, 202697344) whole. No policy-learning result; no automatic retry (#123). See #125.
> **Combination policy pilot — generation without a completed throughput check: amendment, admission and wrapper BUILT; the seed survey NOT run (decision note + addendum after entry 42, 2026-10-30).** The user accepted attempt 2's TIMEOUT and closing and approved preparation for one 1,000-game generation run with a six-hour ceiling (amendment `11ced081…`, `7295e765`): admitted by the committed CPU timing alone; neither throughput record is read or relabelled; a timeout is VOID and assessed, with no automatic retry. The existing seed rule's 1,000-seed block [202698400, 202699400) contains the H4 fixture literals 202699000 and 202699032, so its survey could only be NOT CLEAN; it was not run. The seed method needs the user's decision. Nothing run; 54 gates closed. See #126.
> **Combination policy pilot — the two-band seed method and training's outer-record admission BUILT; the real survey NOT run (decision note + addendum, 2026-10-30).** The user accepted the generation build and the seed blocker and chose the 2027 band [202700000, 202800000) for generation's seeds. `seed_band_2027.py` (`f870de24`) widens the pinned scanner to both bands (plain and underscore-grouped) inside a private copy of the pinned module and places the block above every registered seed and repository literal in the 2027 band (an empty band: N above its edge); the gap floor stays 1,000 and the check is the pinned one. Training now also needs generation's committed `outer.json` to pass `outer_faults`. A three-mode rehearsal on a synthetic repository passes 55/55 controls per mode. Nothing run on the real repository; 54 gates closed. See #127.
> **Combination policy pilot — generation's seeds SURVEYED, PROVEN and REGISTERED in the 2027 band; the closing copy built (addendum, 2026-10-30).** Survey (`bb1549ea`) CLEAN 55/55: block [202739800, 202740800), anchored on the corpus hex fragment 202738776, gap 1,023 (floor 1,000). Pre-proof (`4701b1d3`) HOLDS 55/55; registration (`326004d0`) appends the block last to ACCOUNTED (48 → 49) with `plan.json` = `build_plan`; post-proof (`5a17ea92`) HOLDS 55/55 on the current registry. The closing copy (`e8ae01cc`) keeps the parallel-worker rule, 21/21. Nothing run on the GPU; 54 gates closed; awaiting launch review. See #128.
> **Combination policy pilot — CORPUS GENERATION: 1,000 games COMPLETED; block closed (entry 43, 2026-10-30).** The one approved run (gate opening `acf07d24`, evidence `56e6f386`) finished in 9,929 s of the six-hour ceiling (outer exit 0, record COMPLETED, gate restored, no survivors; 362.6 games/h, inside the card's 1.5–3 h estimate). 1,000 of 1,000 slots claimed, started and ended; 44,747 rows (40,589 training, 4,158 held out) in 1,000 files, 0 hash mismatches. 43 state-cap games took 664–1,779 s each. K1 RECONCILES 21/21 (`claims` 1,000); K2 EXPOSES and RETIRES [202739800, 202740800) whole. Training is now admitted by the committed records but not authorized; the 2,700 s training ceiling was priced from 37,000 estimated rows (1.45× the real-count projection). Awaiting the corpus and training-readiness review. See #129.
> **Combination policy pilot — generation ACCEPTED; training ceiling set at 2,820 s; the training wrapper to be prepared (decision note after entry 43, 2026-10-31, the user's decision).** The ceiling is the timing record's formula on the real counts (1,865.3 s) × 1.5, rounded up: TERM 2,810 s / KILL 2,820 s / observation to 2,830 s; the timing record (2,700 s) stays as written. Approved: the wrapper copy for CPU `run train` with recipe, checkpoint, admission and STOP rules unchanged, focused fake-launch checks, then launch review; and a limited repair of the external-learning closing tests' stale EXPOSED/RETIRED-tail assertions. Training and pushing remain unauthorized.
> **Combination policy pilot — the training wrapper BUILT for launch review; the external-learning closing tests repaired (addendum, 2026-10-31).** `combination-policy-train-run/01_wrapper.sh` (`f50465f8`) = generation's wrapper with the paths, the training gate (closed), `run train` and TERM 2,810 s / KILL 2,820 s / observation to 2,830 s only; the pilot is unchanged. 10 focused checks (ceiling re-derived from the committed records; fake launches) and four caught mutations; 158 focused tests pass. The stale EXPOSED/RETIRED-tail assertions repaired (`e7b35d5d`); entry 43's cause corrected (D2's held-out set, not stage 2). Training and pushing remain unauthorized. See #130.
> **Combination policy pilot — the ONE CPU training run APPROVED (addendum, 2026-10-31, the user's decision).** Panic/reboot check; idle and BOINC checks enforced; only the training gate opened by a one-line commit in a clean worktree; the wrapper once (calib020_0001 weights-only, CPU, the frozen recipe of 3,000 updates and STOP rules; TERM 2,810 s / KILL 2,820 s / observation to 2,830 s). A timeout is VOID with no automatic retry and does not permanently reject the method (#131). Strength games, deployment and pushing remain unauthorized, even on ELIGIBLE.
> **Combination policy pilot — CPU TRAINING: COMPLETED, verdict ELIGIBLE (entry 44, 2026-10-31).** The one approved run (gate opening `9b7479f7`, evidence `578577f2`) finished in 1,866 s of the 2,810 s TERM (outer exit 0, gate restored, no survivors). Pre-fit KL 1.434 nats (floor 0.01); probability discrepancy ≤ 1.03 × 10⁻⁵; held-out CE 3.862 → 3.645 (−0.217 nats); value outputs bitwise equal on 4,158 held-out rows, encoder and value-head digests unchanged; 3,000 finite updates. The card's directional prediction held. No strength result; the screen may be proposed, but strength games, deployment and pushing remain unauthorized. See #132.
> **Combination policy pilot — training ACCEPTED as ELIGIBLE; the 64-game screen approved to BUILD only (addendum, 2026-10-31, the user's review).** The user verified the head's hash and parameter digest, the unchanged source and evidence, the gate's restoration and all 54 gates closed. Learning toward the search targets is shown; whether it improves play is open. Build: the trained head (sha256 `ead156e2…`, bound to the successful inner and outer training records) against the frozen proven combination, both averaged, 200 simulations, halving from ply 20; the existing match loop and wrapper with a new closed gate, its own paths and fresh-seed survey and closing copies; STOP below a mean pair score of 0.40, otherwise CONTINUE, no strength claim; a fixed ceiling proposed from completed match timings. Seed procedures, checkpoint loading for evaluation, GPU games, further training, deployment and pushing remain unauthorized. See #133.
> **Combination policy pilot — the 64-game screen BUILT for review; ceiling PROPOSED at 3,480 s (addendum, 2026-10-31).** Build `6c908ca8`; nothing run. `run screen` behind a new closed gate (55 in all), admitted only by training's committed ELIGIBLE record and clean outer.json, bound to head.npz `ead156e2…` and its digests; the head installed before any inference; a probe must find the candidate's opening priors changed. Both sides averaged halving at 200 simulations through the strength match's loop; STOP below 0.40, else CONTINUE. Ceiling from completed match timings: 64 × 64.75 plies × 0.5508 s = 2,282.3 s × 1.5 → 3,480 s (TERM 3,470), room for 10 capped games against 2.75 expected. Survey and closing copies built, not run. Compiled values can differ from the parent's in the last bits (recorded, not gated). Awaiting review (#133).

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

## Benchmark reset for any future AlphaZero training programme, 2026-08-22

**Interpretive clarification, not a new experiment or authorization.** The research and product
workstreams answered different questions, and their sequence can otherwise make it appear that the
product comparison was the training hypothesis. It was not. The winning network already existed;
the product work established that its export was faithful, that the product stack could run it, and
that it was stronger than the previously served bytes at hard difficulty.

### What the completed sequence actually tested

| stage | hypothesis / reason for testing | outcome | durable consequence |
|---|---|---|---|
| `0409` → `calib020_0001` | The narrow goal-line repair in `0409` was insufficient because the broader post-opening sharp-drop family still failed. Broad post-opening calibration might repair that family without losing the goal-line fix. | `calib020_0001` passed the goal-line and broad post-opening gates, then beat prior promoted best `0379` in both 800-game directions: about `+80 Elo` each way. | This was the successful **training hypothesis**. `calib020_0001` became and remains the best-supported research checkpoint. |
| Successor programme centered on A/B/C/D | A remaining black pre-drop overvalue at 400 simulations might be correctable while preserving goal-line, broad post-opening and red-position guardrails. | No trained or search-policy successor met its promotion bar. Increasing search budget moved A mean `+0.2570 → +0.0626 → −0.0451` at 400/1600/6400 simulations, showing that much of A was a shallow-search selection artifact rather than a stable value-head defect. | The later experiments did **not** produce the winner; they established that nearby attempts to improve it should stop. The calibration, continuation, readout, FPU, tree-local and frozen-parent families listed in this ledger remain closed. |
| Product-model alignment | The verified ONNX export of the research winner might be stronger than the unknown-provenance artifact the product actually served, but the research result against `0379` could not answer that across a different runtime and search implementation. | Native↔ONNX parity passed; the Node MCTS OOM was remedied without a frozen-corpus behavior change; the 200-game hard match classified the candidate `CANDIDATE_STRONGER` with mean pair score `0.8475`; the candidate was deployed. | This was **deployment validation**, not another training hypothesis and not evidence about `medium` or absolute playing strength. |

### Benchmark hierarchy from this point forward

| role | frozen reference | rule |
|---|---|---|
| **Starting checkpoint** | `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`, SHA-1 `209cf2d4fd24a48553d259dd71b4954867b9473e` | A future trained successor starts from this checkpoint unless a new card explicitly justifies training from scratch. |
| **Primary research promotion benchmark** | The unchanged `calib020_0001` parent under the native MLX evaluation stack | Equal checkpoint-independent configuration, equal 400-simulation search, balanced colours, fresh frozen seeds, and a preregistered aggregate confidence rule. A candidate must beat the current best, not merely an older checkpoint. |
| **Historical/generalization control** | staged `model_iter_0379.safetensors`, SHA-1 `8ad62ac432c35c6ea9b0630b8a2b8c572a0b03a1` | `0379` is no longer a promotion benchmark. Beating it is insufficient because the current best already does so by about `+80 Elo`; it may be used only if a future card gives it a specific secondary role. |
| **Product rollback artifact** | `1d64027db521a50f` | Not a research checkpoint benchmark: its source checkpoint is unknown. It remains an operational rollback target only. |
| **Post-promotion product validation** | The model served when a research successor becomes eligible | Export parity and a separately reviewed product-stack comparison occur only after research promotion evidence; they do not substitute for the native parent match. |
| **External anchor** | none currently exists | Any reopened, ongoing strength programme must define an external anchor before making an absolute-progress claim. Internal head-to-heads establish relative improvement only. |

### What “continue training” may and may not mean

The existence of a proven checkpoint and a working four-worker MLX pipeline is **not** itself a
training hypothesis. Ordinary self-play continuation (cold or warm replay), nearby dose or
learning-rate changes, frozen-parent opposition, opponent pools as a rescue, and inspection of
previously rejected intermediate checkpoints remain closed under do-not-repeat `#48–#51`.
`0379` must not be restored as the primary bar merely because it is easier to beat.

A reopened training programme therefore requires, before code or compute:

1. a genuinely new mechanism — for example a justified architecture/capacity change or a new
   data/objective source, not another local continuation of a closed recipe;
2. a directional prediction explaining why that mechanism can improve on `calib020_0001` rather
   than repeat the observed collateral tradeoffs;
3. the smallest cheap falsification that can reject the mechanism before a full run;
4. a committed experiment card freezing starting checkpoint, code and data identity, budget,
   seeds, worker/arbiter configuration, checkpoint-selection rule, guardrails and stopping rules;
5. only after those gates pass, a balanced native-stack parent match against frozen
   `calib020_0001`; and
6. an external anchor if the intended conclusion is broader than “stronger than this parent in
   this internal stack.”

The historically proven generation configuration is **four workers behind one Metal-owning
inference arbiter**. That is a candidate configuration for a future card, not standing permission
to launch a run. **No training is active or authorized by this section.** The next activity, if the
programme is intentionally reopened, is hypothesis selection and preregistration—not training.

## External anchor — twixtbot, RUN 2026-08-24 — REJECTED; NO USABLE EXTERNAL ANCHOR EXISTS

**This closes the question the whole programme deferred.** Every Elo in this ledger is
sibling-vs-sibling; the closeout table says an ongoing programme needs an **external strength
anchor** before any absolute-progress claim. One was found, qualified and run. It failed.

**The anchor.** `github.com/stevens68/twixtbot-ui` @ `83749f230a0bae1766b46a05bfde0ed87f0a9a0a`
(MIT; wraps `BonyJordan/twixtbot`), an independently trained TwixT AlphaZero engine. Configured to
our exact rules — 24×24, automatic links, no link removal, own-link crossing **off**, swap **off**
— and driven through an adapter that asserts **per-ply state equivalence** (pegs, bridges as
order-independent endpoint pairs, side to move, full legal-move set, terminal status, winner) with
an external 280-ply cap applied identically to both engines.

**The run.** One authorized execution, repo `bcb72bf`, 3h 28m 49s, **128/128 games, 0 aborts**,
seeds `[202611128, 202611256)` each used exactly once. 8 openings × both colour arms × 4 `trials`
settings × 2 references, 16 games per cell.

| `trials` | vs `0379` | vs `calib020_0001` | non-saturated? |
|---|---|---|---|
| 0 — raw policy, **no search** | **1.000** (16) | **1.000** (16) | ✗ |
| 100 | **1.000** (16) | **1.000** (16) | ✗ |
| 400 | **1.000** (16) | **1.000** (16) | ✗ |
| 1000 | **1.000** (16) | **1.000** (16) | ✗ |

**The anchor won all 128 games**, 64/64 as red and 64/64 as black. Mean game length 35 plies
(min 27, max 49); the 280-ply cap was never approached. Pass condition was **non-saturation only**
— score rate within `[0.15, 0.85]` against BOTH references — deliberately *not* agreement with our
internal ordering, since requiring an external reference to reproduce our own result would make it
incapable of contradicting us. `passing_trials: []`, `selected_trials: null`.

**Verdict: the anchor is REJECTED, and no usable external anchor exists.** It cannot calibrate
against our checkpoints because it never loses to them — at `trials=0` it uses **raw policy with no
search at all** and still beat `calib020_0001` at 400 simulations 16 times out of 16.

**What this establishes.** Both checkpoints were decisively weaker than twixtbot under the tested protocol; sibling-vs-sibling Elo could not reveal that external ordering. It is an ORDERING under this frozen protocol, not a
statement about absolute strength: **the anchor itself was never calibrated against anything**, so
its own strength is unknown and our checkpoints inherit no absolute placement from losing to it.
The programme's internal results remain valid as *relative* statements and are unaffected —
`calib020_0001` still stands as the best-supported checkpoint by the evidence that selected it.

**What it does NOT establish.** No Elo, no absolute strength number, and no lower bound on the size
of the gap — a 1.000 score rate bounds nothing. No rules-matched comparison either: the anchor's
network was trained with own-link crossings **allowed** and played here with them **forbidden**, a
limitation recorded from the feasibility audit onward and never measured. And nothing about
`calib020_0001` versus `0379`: the anchor is **flat** at 1.000 against both, so the descriptive
ordering read-out returned no information.

**Seed accounting.** Reserved `[202611000, 202611400)` = 400. Consumed: `202611000` (a preflight
smoke that played one real 400-simulation move) and `[202611128, 202611256)` (G3). **271 remain
unused.** See do-not-repeat **#52**.

**Evidence.** `docs/superpowers/evidence/2026-08-24-twixtbot-g3/` — the runtime artifacts
byte-for-byte, including the untouched `g3_results.jsonl` (130 records: header, 128 games with full
move lists and complete anchor visit arrays, terminal verdict). Card:
`docs/superpowers/2026-08-22-twixtbot-anchor-pilot-card.md`. Six preflight packages under
`docs/superpowers/evidence/2026-08-23-twixtbot-g3-preflight*/`.

**Descriptive by-product — visit concentration.** Non-zero visit slots per anchor move averaged
**5.1 / 9.2 / 17.3** at `trials` 100/400/1000, while the top-move visit share stayed **~0.83** at
every setting. Concentration is real and *widens* with search. Recorded because it was an open
question; it is not a gate and did not affect the ladder or the verdict.


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

   *Narrow exception (2026-10-19, review).* One fixed, root-only sequential-halving allocation experiment (Post-closure entry 15) is excepted from the closing sentence above: search below the root is unchanged. Progressive widening, FPU variants and the archived evidence remain closed; the exception covers no other variant, parameter or shape.

48. **Rescuing the rejected five-iteration cold-buffer continuation with nearby endpoints, dose changes or more match games.** The frozen iteration-5 checkpoint lost clearly to its parent over 400 games: score `0.31375`, CI95 `[0.2687, 0.3588]`, about `−136` Elo. The preregistered absolute per-colour rule mechanically fired for red and black, but the 2026-08-10 erratum makes that non-diagnostic of independent colour harm; the aggregate loss alone closes the recipe. Do not evaluate iterations 1–4, extend the same run to 10 or 20 iterations, alter its learning rate or training steps, add games, or invoke the success-only replication. The result rejects the exact calibration-free, cold-buffer continuation recipe; it does not isolate whether the cold replay buffer, removal of the parent's calibration objective, or their interaction caused the regression, and it does not prove all training has plateaued. A successor must introduce and justify a genuinely different training mechanism rather than relabeling a top-up as discovery.


49. **Ordinary self-play continuation from `calib020_0001` in ANY dose, warm buffer included — the family is closed.** Two runs now: cold buffer `0.31375` / `−136.0` Elo, and the parent-replay bootstrap `0.4325`, CI95 `[0.3843, 0.4807]`, `−47.2` Elo, CI95 `[−81.9, −13.4]`. The bootstrap's whole aggregate score interval sits below `0.50`, which is sufficient to close the family. Its preregistered absolute per-colour rule mechanically fired for black (`0.3650`, upper `0.4317`) but, per the 2026-08-10 erratum, this does **not** establish black-specific harm or an independent second rejection; the red raw score of `0.5000` (upper `0.5682`) likewise does not establish colour-specific neutrality. Do **not** run a warmup-size grid (250 / 1,000 / anything), extend to ten or twenty iterations, change the dose or learning rate, evaluate or select iterations 1–4, add match games, or run the replication — replication was authorized only by a win. **Its central forecast (substantial recovery, no promotion) MATCHED**, so the design was sound and the answer is negative; that is a result, not a reason to retune. Read the improvement precisely: `−136.0`→`−47.2` is **descriptive across two separate runs** with different seeds and evaluation intervals, **not a paired causal estimate**, and the 10–20% promotion prior is **not validated by one failure**. The run **supports the parent-replay-bootstrap hypothesis but does not isolate the cold buffer causally** — new seed and 500 extra parent games moved with it. Keep `calib020_0001`. `--replay-warmup-games` stays in the trainer as tested, default-off infrastructure; its presence authorizes nothing. A successor must name a genuinely different training mechanism, explain why it addresses a ~47 Elo shortfall that warm-starting did not close, and bring its own cheap falsification design and an external strength anchor if it intends to become an ongoing programme.

50. **Two independent inference-server threads submitting concurrent work to the same Metal device — in any experiment, for any purpose.** The frozen-parent opponent run aborted with **exit 134** (SIGABRT) at the first line of iteration 1, on a **Metal driver assertion**, not a Python exception: `AGXG15XFamilyCommandBuffer … failed assertion 'A command encoder is already encoding to this command buffer'`. The 500-game warmup had just completed normally **because it runs on the ordinary single-network path**; the crash arrived the moment the second server began serving alongside the first. Do **not** respond with a mutex around submission, a retry, a worker-count reduction, a `flush_ms` change, or a second process — a process-aborting driver defect is not a contention-tuning problem, and a lock that merely serialises two encoders leaves the same two owners racing at every seam. **Any successor that needs two networks served concurrently must route both through ONE Metal-owning inference arbiter**, and must first pass a **newly authorized, tiny real-GPU feasibility smoke** before any warmup or training budget is spent. **The lesson about evidence, not just about Metal:** every pre-accelerated gate item passed honestly, but the transport tests drive both servers with *stub* evaluators that never touch the GPU, so nothing in a 2,863-test suite exercised two servers doing real accelerated work at once. A stub-level integration test is not evidence about a device-level contract. **This entry records an implementation failure and is evidence neither for nor against frozen-parent training** — the mechanism never generated a single training game. Card (terminated): `docs/superpowers/2026-08-10-frozen-parent-opponent-experiment-card.md`.

**#50 addendum (2026-08-11) — one-thread/two-network feasibility PASSED; the two-server prohibition STANDS.** A countersigned probe (`docs/superpowers/2026-08-10-single-arbiter-feasibility-card.md`, execution commit `4808f324`) ran two arms of 400 synchronous `infer()` calls each: control, one thread with one network; treatment, one thread alternating **two independent resident network instances**. Both exited `0` with per-call shape and finiteness checks, stable digests, and agreeing digests across evaluators (`12c4e65bf4adf338`). **So a single Metal-owning thread CAN hold two networks at this dose** — the arbiter design is worth building. **Nothing about the prohibition changes:** two independent servers submitting concurrently remain forbidden, and this probe did not test that, did not show the original race is fixed, and did not exercise request routing or per-model batching. Reproducibility is to the six-decimal digest quantization, not bit-for-bit; timings show no observed slowdown at this dose, from one sample per arm. `--frozen-opponent-checkpoint` stays blocked at startup.

**#50 addendum (2026-08-11) — the implemented single arbiter SURVIVED real device load; the two-server prohibition is UNCHANGED.** The refactored transport (one `InferenceServer` thread, `model_id` routing, `(worker_id, model_id)` response queues, per-model batching) ran its countersigned real-GPU smoke at execution commit `7fb7f70b`: **402 synchronous GPU calls** — 2 reference + 400 routed across two workers — serving **two different checkpoints** (`209cf2d4…`, `8ad62ac4…`) with **exactly 1 observed inference thread**, per-model telemetry of **200 requests / 2,800 rows / 200 batches**, universal digest matches against an **independent** oracle (references computed by direct evaluator calls on the arbiter thread, never routed), both workers exiting `0`, and the server thread stopped. Exit `0` in 4.24 s. **So one Metal-owning thread can serve two networks in production shape.** Nothing about the prohibition changes: **two independent servers submitting concurrently remain forbidden**, and this smoke tested neither that, nor mixed-model grouping in one flush (`B` equalled the row cap, so every request was its own batch), nor **the composition with MCTS and real game generation** — `run_parallel_selfplay` over the dual-root `play_game` has been exercised only with stubs. `--frozen-opponent-checkpoint` stays blocked at startup; lifting it is a separate card gated on that composition smoke.

**#50 addendum (2026-08-11, second) — the single-owner arbiter survived REAL MCTS COMPOSITION; the two-server prohibition is UNCHANGED.** The first addendum recorded the arbiter passing synthetic load. The composition smoke (`docs/superpowers/2026-08-11-frozen-opponent-flag-lift-card.md`, execution commit `50324a45`) then drove the real path — `run_parallel_selfplay` → `self_play_worker` → dual-root `play_game` → MCTS → two `RemoteEvaluator`s → the arbiter — with real games: **4/4 generated, exactly one observed inference thread, learner-only buffered rows with an exact 2/2 colour split, server thread recorded and confirmed stopped, and `mixed_model_flushes = 38`.** That last figure matters most: **mixed-model grouping inside one flush has now actually executed on the device**, which neither the arbiter smoke (where `B` equalled the row cap, so every request was its own batch) nor any stub test could reach. Traffic was **ragged and partially coalesced** — 360 vs 320 requests, 2,231 vs 1,668 rows, batches below request counts — i.e. the real variable-size behaviour. **Nothing about the prohibition changes:** two independent servers remain forbidden, and this smoke tested neither that nor routing correctness (both instances loaded the SAME checkpoint, so a crossed response would be invisible — routing rests on the two-checkpoint arbiter smoke `9c5847a0`). `--frozen-opponent-checkpoint` remained blocked throughout and after; lifting it is a separate reviewed diff, and no training is authorized.

51. **Frozen-parent opposition from `calib020_0001`, in any dose or shape — the mechanism is CLOSED.** The `fp6` run (execution commit `13dd72f`, 2026-08-12) scored `0.46625`, CI95 `[0.4177, 0.5148]`, Elo `−23.5`, CI95 `[−57.7, +10.3]`. **The bar was not met.** Do **not** change the dose or warmup size, add a second opponent or an opponent pool, extend beyond five iterations, evaluate or select from iterations 1–4, run a larger match to "resolve" the parity, or run the **0379 generalization match** — that was conditional on a bar-met result and is not licensed by a non-pass. **Keep `calib020_0001`.** Read the result exactly: the candidate was **not shown stronger and not shown weaker at this dose** — both intervals include parity, so it is **not statistically distinguishable** from the parent, which is **not** the same as equal. The unresolved parity is **not** an invitation to buy resolution with more games; that would be a post-hoc power increase chosen after seeing the interval. The preregistered `0.47–0.51` / ~10% forecast matched **approximately**, and **one non-pass validates no probability**. The `cont5`→`warm5`→`fp6` improvement is **descriptive across separate runs, not paired or causal**. Iteration 1's anomalous 8,114.9 s is an **infrastructure observation** and carries no scientific weight. **`#50` is unchanged** — two independent inference servers on one Metal device remain forbidden; the single arbiter is what made this run possible and it held at full scale.

52. **The twixtbot external-anchor pilot, in this shape — RUN AND REJECTED 2026-08-24; do not retry it.** One authorized 128-game run (repo `bcb72bf`, clone `83749f2`, seeds `[202611128, 202611256)`) returned **1.000 for the anchor against BOTH references at EVERY `trials` setting**, including `trials=0` where it uses raw policy and no search. `passing_trials: []`. Do **not** alter the `trials` ladder, relax the `[0.15, 0.85]` band, change or add openings, grant the anchor a rules change or handicap, add games, or hunt a weaker anchor configuration — every one of those was excluded before the result, and choosing any of them now is post-hoc selection against a recorded outcome. Do **not** re-run this pilot to "confirm" a 128/128 sweep. **Read the result exactly:** **both checkpoints were decisively weaker than twixtbot under the tested protocol, and sibling-vs-sibling Elo could not reveal that external ordering** — an ordering, NOT a statement about absolute strength, since the anchor was itself never calibrated; it gives **no Elo, no absolute number and no lower bound on the gap**, and it is **not** a rules-matched comparison, because the anchor's net was trained with own-link crossings allowed and played with them forbidden. It says **nothing** about `calib020_0001` versus `0379` — the anchor is flat against both. **`calib020_0001` is unaffected and still stands.** A future external anchor must be a *different* engine or a rules-matched one, and needs its own scope; the recorded fallback for strength work is the capacity-headroom probe on the existing replay corpus, which is unauthorized. Infrastructure that survives and is qualified: the adapter with per-ply state equivalence (`scripts/GPU/alphazero/twixtbot_adapter.py`), the seeded reference agent, and a fail-closed runner whose **exit status is the gate** (0 pass / 1 fail / 2 aborted / 3 precondition).

53. **The external-engine learning trial, as run — PROVISIONAL VIABILITY 2026-10-10; do not extend or rescue it.** Held-out imitation passed (CE **3.211 / 3.890 / 4.566**; top-1 **25.7% vs 20.5%**) and the 32-game screen was inconclusive (**12–20**, **[0.035, 0.715]**, NOT_VETOED). Do **not** add games or repeat the match to resolve the interval, refit beta, retrain, change the dose or rows, select another checkpoint, or promote the candidate — each was excluded before the result, and choosing one now is post-hoc selection. Do **not** read NOT_VETOED as non-inferiority, or teacher agreement as rules-correct strength. Any successor needs its own card and fresh seed accounting, and first a decision on the exhausted seed-selection rule and a distinct exit status for unverified process-group cleanup (result note `docs/superpowers/2026-10-10-external-engine-learning-result.md`). **`calib020_0001` stands.**

54. **Search distillation Stage B, as run — NON-PASS 2026-10-14; the pilot is closed.** Do **not** rerun Stage A or B, reuse their spent games, labels or seeds, retune on the 40 held-out games, or play strength games for the Stage B student or control without a new card. Stage A's PASS establishes stronger search at 1,600 simulations, nothing about distillation. The training-mode BatchNorm in Stage B's labels (Post-closure entry 3) is a possible confound, not a proven cause, and does not reopen the stage.

55. **Batched search (B1600) as a way to afford stronger search — CLOSED 2026-10-15.** Both screens were NOT_FEASIBLE (1.945; fixed-shape eval-mode 1.445). Do **not** try padding variants, batch-size or flush-setting searches, timing retries, lowered bars, or reruns under quieter load. The one-sync change (7.8% vs a 10% bar) is not adopted. The remaining cost lies outside inference but is not attributed to any component.

56. **The inference-mode match (corrected vs legacy BatchNorm) — inconclusive 2026-10-16; do not extend it.** 20–12, mean pair 0.625, interval includes 0.5. Do **not** add games or repeat the match to obtain significance, and run no further mode or batching experiment for this question. Keep the factory fix; future research uses eval-mode inference; historical results stay under their original protocols.

57. **The flat-search defence failure does NOT reopen FPU sweeps.** The 2026-10-16 probe shows absolute FPU 0 outranking every visited move once Q nears −1. That supports a mechanism; it does **not** revive the rejected absolute or policy-mass FPU settings (v17, context-relative FPU) or justify a parameter grid. Any correction must be a new, narrowly scoped test of this observed mechanism, chosen and reviewed on its own.

58. **The game-3 missed defence (ply index 41) is not evidence that a search correction saves games — 2026-10-19.** A rules-only check proved red forces a win within three plies after the save (21,6). Do **not** extend the horizon, rerun the check, or cite this example as a game a deeper search would have saved; it shows only the flat-search mechanism. A future correction needs examples where the save is not already lost within the checked horizon.

   *Correction to #58 (2026-10-19, review).* The requirement for new examples applies **only to corrections justified by this tactical criterion** (a missed one-move defence), not to every future search or strength idea: a bounded tactical proof is not a universal prerequisite for strength research. Conversely, failing to prove a loss within a horizon would not prove a position savable. Both inference fixes (`c8cb2aa6`, `444269ac`) and `calib020_0001` stay; do **not** expand this check, reopen FPU settings or start training from this example — the four diagnostic games supplied no evidence that warrants such a change.

59. **Mirror averaging at 200 vs 400 ordinary simulations — STOP_CONFIGURATION 2026-10-19.** 19/32 against a frozen bar of 24 (mean pair 0.594, interval includes 0.5). Do **not** add games or repeat the match to reach the bar, change the budgets (e.g. 250 or 400 averaged), weight the two orientations unequally, add other symmetries or orientations, or rescue it with another readout — each would be post-hoc tuning of a stopped configuration. A different symmetry-based idea needs its own justification and authorization. The prototype stays unwired; production is unchanged.

60. **A per-colour exact-input prediction cache — STOP_CONFIGURATION 2026-10-19.** Identical results, but 4 hits in 802 lookups and 2.0% *slower* search (bar: 10% faster). Do **not** retry it with more positions, longer sequences, faster hashing, a different key, or under quieter load to find a saving — there is almost nothing to reuse within one colour. Entry 12's 12.8% came from **cross-colour** duplicates; a cache shared between the two sides is a different question (between different agents it means one agent consuming the other's predictions) and needs its own justification and authorization.

   *Correction to #60 (2026-10-19, review).* The parenthesis ("one agent consuming the other's predictions") is wrong as stated: an exact-input prediction shared between two agents is valid **when both use the identical frozen evaluator and the inputs are exact** — it only avoids duplicate computation (entries 12 and 13 ran one checkpoint on both sides). Predictions from **different** models cannot be interchanged. This distinction does **not** authorize a shared-cache experiment. No deployment, retry or further experiment follows from #60.

61. **A standalone strength trial of the mirror-augmentation repair, justified by entry 14's exposure count — NOT JUSTIFIED 2026-10-19 (never run).** Do **not** run the paired control-vs-corrected training-and-match pilot on this evidence, and do **not** generate more games merely to seek a higher exposure count to justify it. This is a decision not to spend, not a failed experiment: 1.6% is an expected buffer share under these four games and the assumed mirror probability 0.5, not a measurement of the incumbent's training buffer; and the exposed positions are late, but it is **not established that they were already decided** — rare, important positions can affect strength disproportionately, so the count does not show the defect was harmless. Keep the fix `0c417882` and carry it into any independently justified future training experiment; a strength claim for the repair alone needs new evidence and its own authorization. #49 (ordinary continuation) stays closed.

62. **The root-only sequential-halving variant (Post-closure entry 15) — STOP_CONFIGURATION 2026-10-19.** 21.5/32 against a frozen bar of 24 (21–1–10; mean pair 0.672, interval includes 0.5). Do **not** add games, repeat the match, or treat the 21–10 lean as earning a confirmatory or strength comparison; and do **not** rescue it by changing the candidate count, `c_visit`, `c_scale`, the (q + 1)/2 mapping, the start ply, the tie-breaks or the final decision rule, by adding Gumbel noise, or by changing the simulation budget — each would be post-hoc tuning of a stopped configuration. The narrow #47 exception covered this one fixed variant and is spent; any other root-allocation idea needs its own justification, a new exception and its own authorization. The prototype stays unwired; production is unchanged.

63. **The banked-budget controller of entry 16 — DECLINED 2026-10-20 (never built or run).** Do **not** tune this policy — its thresholds, credit amounts, caps or borrowing rule — against the consumed match records (root-halving and mirror-averaging replays), run further accounting variants on them, or treat the oracle bookkeeping as strength evidence: it assumed a free classification and measured chronological opportunity only (14.5% of freed credit reached contested moves; ~9% extra budget). The ~24% hypothetical reduction in simulated work is not a measured runtime saving and not evidence that play is preserved. Other adaptive-budget mechanisms are untested and need an independent justification and their own authorization, not a rescue of this policy.

64. **Self-play throughput tuning on the M5 Max (Post-closure entry 20) — settled 2026-10-21.** Do **not** rank self-play settings on the total wall time of a few dozen games: one state-cap game finishing alone moves it by up to 3.5x. Rank on the rate while every worker is busy (`scripts/bench_knee.py` default). Do **not** read a watchdog kill as a Metal hang without progress output and stack dumps. Do **not** re-sweep eval batch above 14 hoping for speed on this machine (16–48 were no faster), and do **not** tune eval batch or stall-flush through `benchmark_phase2_knobs.py`: the labeller never reads them.

65. **The policy-head capacity planted qualification (entry 18 addendum) — NOT QUALIFIED 2026-10-21.** S1 failed (C − R on TR = 0.0286 < δ 0.05). Do **not** rerun it, rerun it with other seeds, or retune its recipe or design (steps, learning rate, batch, clipping, planted scale, δ, the 25% fit fraction, teacher shapes) and run again in search of a pass; do **not** generate real labels with this recipe or run the 200-game pilot on it; and do **not** read NOT QUALIFIED, or C's close fit of TR, as evidence that the head's capacity does or does not bind. A new recipe needs a new card, fresh teacher seeds and its own approval.

66. **The policy-head capacity proposal (entry 18) — SHELVED 2026-10-22.** Do **not** spend further on this proposal — no new planted recipe, synthetic separation check, real-label generation or 200-game pilot under it — unless a new design first establishes, before any fitting, a defensible KL separation between the arms it compares, and is separately approved. Do **not** cite the shelving, or the NOT QUALIFIED result, as evidence that the head's capacity does or does not limit learning: the hypothesis is unresolved.

67. **The D2 Stage 1 live measurement (entry 21) — MEASURED 2026-10-22.** Do **not** rerun it, add games to it, or use its two games (seeds `[202691900, 202691902)`, EXPOSED and RETIRED) as training, held-out or evaluation material: they are measurement material only. Do **not** read their lengths (52 and 54 plies, both decisive) as the corpus's game-length distribution — two games cannot estimate it — or compare their 1.16 s per ply (batched self-play search, 0.73 ms per simulation) with entry 19's 1.74 ms per simulation (the evaluation play path): different search paths. Nothing here is evidence about target quality or strength.

68. **The D2 Stage 2 corpus generation (entry 22) — COMPLETED 2026-10-22.** Do **not** rerun it, top it up, replace any game or phase, or add games to the corpus: it is fixed at 120 games in 12 phases. Do **not** move games between the training and held-out sets (held out: slot ≡ 9 mod 10, 12 games), train on the held-out games, or use any of the 120 games (seeds `[202692100, 202692220)`, EXPOSED and RETIRED) for strength or screen games. Do **not** read the run's 2.3 h, its 1.19 s per ply or its length distribution as a cost bound for other configurations, or anything here as evidence about target quality or strength. Training needs its own approval and a frozen Stage 3 card with the recipe fixed before training.

69. **The D2 Stage 3 training run (entry 23) — STOP 2026-10-22.** Do **not** rerun it, rerun it with other seeds, or retune its recipe (anchor weight, learning rate, updates, batch, clipping, L2, surface) and judge the result on the same 12 held-out games: they have served as this recipe's final check, and re-judging retuned recipes on them is selection. Do **not** use the candidate (`student-stage3-20261022T232207Z.safetensors`, state digest `9c73feb2…`) in a screen, a strength comparison or promotion, or as a starting point for further training. Do **not** read its lower held-out cross-entropy as evidence of stronger play, or the STOP as evidence that search distillation cannot work: the run shows only that this recipe moved the value output beyond the fixed limits. Do **not** name a cause for the gradient-norm spikes until one is derived. A new recipe needs a new card, fresh held-out data and its own approval.

70. **The D2 gradient diagnostic (entry 24) — MISMATCH 2026-10-22.** Do **not** rerun it, rerun it without the attribution passes, on the CPU, with other settings or from update 21, or extend it to more updates: it was one run under a frozen card and it stopped where its card says to stop. Do **not** read the replayed update 21's per-tensor norms as Stage 3's (the two runs differ at that update), or one spike as a description of all 86. Do **not** name a cause for the spikes or for the divergence at update 21 — GPU non-determinism, an effect of the attribution pass, large gathered policy features and the zero-variance BatchNorm channels are hypotheses, none measured as a cause — and do **not** read anything here as a cause of Stage 3's value drift. A next diagnostic needs its own card and approval.

71. **The D2 reproducibility diagnostic (entry 25) — MEASURED 2026-10-22.** Do **not** rerun it, extend it past update 21, add passes or reorder them: it was one run under a frozen card. Do **not** treat any GPU training result on this update path (`forward_padded` → masked policy cross-entropy → `nn.value_and_grad`, MLX 0.30.3, Metal) as a deterministic function of its inputs until that is re-established. Two passes from identical state and inputs produced different gradients, and one update's computed gradient violated the linearity of the stated loss. Do **not** read the Stage 3 spikes as properties of the data, the records or the BatchNorm statistics, and do **not** name the mechanism (which operation, why the first pass in a process differs from later ones) until one is derived. Do **not** start another learning recipe on this path before the gradient is checked against an independent reference. A next step needs its own card and approval.

72. **The D2 gradient correctness check (entry 26) — MEASURED 2026-10-22.** Do **not** rerun it, rerun it at other states or batches, or treat its fixtures as a test suite to iterate against without a new card. Do **not** train, fine-tune or evaluate a learning recipe on the MLX 0.30.3 Metal path (`forward_padded` → masked policy CE → autodiff) on the strength of its gradients: against the NumPy reference, the GPU policy-head gradients on real batches are wrong and not repeatable, and the CPU's are within tolerance. Do **not** call the CPU an oracle (it shares MLX's autodiff), or call the GPU encoder wrong or right: its 0.19 forward difference repeats on the GPU, and no independent reference covers it. Do **not** name the mechanism behind the fc and out discrepancies (kernel, shape, memory or graph-context dependence) until one is derived, or say whether earlier checkpoints' training or Stage 3's value drift was affected: both remain unresolved. Earlier match results remain evidence about those checkpoints. Backend upgrades and production fixes need their own approval.

73. **The D2 linear-layer reproducer (entry 27) — REPRODUCED 2026-10-23.** Do **not** rerun it, add row counts, layouts or contexts, or search for the exact threshold without a new card: one run under a frozen case list. Do **not** read the two failing sizes (33,600 and 33,728 rows), and the clean 32,768, as an established rule ("any matrix with more than 32,768 rows"). The rows-0…N−32,736 pattern was measured at two sizes only. Do **not** name the kernel, dispatch or indexing cause until it is derived. Do **not** read the `out` layer's "gross" bias-gradient flags (on the CPU too), or the 512-row `out` cases' "reproduces" flags that rest on them, as the failure: the cotangent there sums almost to zero, so the scale is tiny. Do **not** run any GPU learning step on a path with these shapes until the defect is avoided or fixed and checked against the saved expected results. Backend upgrades, workarounds in production code, bug reports and pushing need their own approval.

74. **The unplanned GPU execution of the D2 synthetic row-count probe during mutation controls (entry 28) — INCIDENT 2026-10-23.** Do **not** treat the stray `mlxrows-20261023T122440Z/compact_example.npz` as the probe's result, or count that execution as the approved invocation: it ran outside the gate procedure and left no record. Do **not** write a test that can reach a public entry point without remapping devices and redirecting outputs, **whatever the gate says**: a mutant that removes the gate check turns such a test into a real run. Do **not** run a control harness without a check that stops it when files outside the worktree change.

75. **The D2 historical exposure audit (entry 29) — READ-ONLY 2026-10-23.** Do **not** read "≤ 32,768 rows" as "unaffected": that count was measured clean only on the M5 Max under MLX 0.30.3, and the earlier runs' machine and version were not recorded. Do **not** read Stage 3's confirmed shapes as proof that the defect caused its value drift. Do **not** use the 27.85% width figure that included held-out records. Do **not** run the trainer with its current default cap (576) and B = 64 on the GPU until the defect is avoided or fixed and checked.

76. **The D2 synthetic row-count probe (entry 30) — COMPLETED 2026-10-23.** Do **not** rerun it, add row counts, layers or contexts, or search for the band's rule without a new card. Do **not** state N − 32,736, or the post-hoc 32 × ⌊(N − 32,736)/32⌋, as the rule, or 32,768 as a universal boundary or a safe chunk size: one MLX version, one machine, three shapes, nine sizes. Do **not** call the defect a narrow-operand defect: the wide 64→64 control fails too. Do **not** judge a fix or workaround on one GPU repeat: gross outputs differ between repeats. Do **not** treat the stray entry 28 file as an independent result. Validating a fix or workaround, package changes, training, bug-report submission and pushing need their own approval.

77. **The D2 full-float32 setting validation (entry 31) — CANDIDATE 2026-10-23.** Do **not** rerun it or extend it to other settings, shapes or sizes without a new card. Do **not** read CANDIDATE as a fix for training: it covers the isolated linear operations only, and the policy head and full-model gradient path must be validated against the saved fixtures before any GPU training resumes. Do **not** name the NAX kernel as the cause: it is the source reading, and no kernel was inspected. Do **not** make the setting persistent, or put it in production or training code, without that validation and its own approval. Do **not** read its timings as a benchmark.

78. **The D2 full-float32 setting path validation (entry 32) — FAIL at phase 2, 2026-10-23.** Do **not** rerun it, retry phase 2, run phase 3's 21 updates, loosen or re-pick the tolerance, or swap the comparison base after seeing the result, without a new card and approval. Do **not** describe the setting as validated for the gradient path or for training: two encoder conv2 weight gradients missed the frozen tolerance on the update-0 batch. Do **not** call the remaining difference a GPU error, or a CPU one: it is a disagreement between two MLX backends with no independent encoder reference, and no cause was measured. Do **not** read phase 1's pass or the vanished gross gradients as rehabilitating the Stage 3 candidate or explaining its value drift. Adopting the setting, package changes, chunking, training, bug-report submission and pushing need their own approval.

79. **The D2 convolution-reference diagnostic (entry 33) — both targets INHERITED, 2026-10-23.** Do **not** rerun it, extend it to other tensors, batches or states, or trace the upstream gradient further back, without a new card and approval. Do **not** read INHERITED as either backend being right: the whole-model gradients still differ, and no independent reference covers the upstream gradient. Do **not** read the bitwise kernel agreement as validating the setting for the gradient path or for training, or as relaxing or overturning entry 32's FAIL. Do **not** name a cause for the channel-42 concentration; it is an observation. Do **not** modify or delete the fixtures in `~/twixt-cache/search-distillation-d2/convref-fixtures-20261023T200104Z/`: they are this entry's evidence. Adopting the setting, training, package changes, bug-report submission and pushing need their own approval.

80. **The D2 encoder branch trace (entry 34) — BRANCH_CHANGE, 2026-10-23.** Do **not** rerun it, trace the operand differences further upstream, or extend it to other batches or states, without a new card and approval; the next step is a qualification decision, not another, more detailed diagnostic. Do **not** call the branch change harmless rounding: small pre-activations do not establish that, and it moves two gradients past the frozen criterion. Do **not** call either device's branch or gradient correct. Do **not** use it to relax, re-judge or overturn entry 32's FAIL, or to force CPU/GPU agreement (for example by perturbing or masking ReLU inputs). Keep the fixtures in `~/twixt-cache/search-distillation-d2/branchtrace-fixtures-20261023T210537Z/` unmodified. Adopting the setting, training, package changes, bug-report submission and pushing need their own approval.

81. **The full-float32 qualification decision (after entry 34), 2026-10-23.** Do **not** trace or localize entry 32's encoder discrepancy further: it is explained as a branch change. Do **not** re-judge entry 32 under the prefix card's new criterion; that criterion is prospective, for the updated state only. Do **not** run authorized GPU work on this machine and runtime without `MLX_ENABLE_TF32=0` in that process, and do **not** make the setting persistent without approval. Do **not** read the setting as qualified for training, or the branch change as harmless. Do **not** satisfy a failing native comparison by widening entry 26's tolerance; only the card's explicit accounting can explain one, and an unexplained discrepancy stops qualification without automatic follow-up. Do **not** freeze, build, gate or run the prefix card before the user reviews it. Training, a candidate checkpoint, strength claims, bug-report submission and pushing need their own approval.

82. **The D2 21-update diagnostic prefix (entry 35) — QUALIFIED_AT_U21, 2026-10-23.** Do **not** rerun it, extend it to more updates, other states, other batches or the failed candidate, or open gate 44 again, without a new card and approval. Do **not** read `QUALIFIED_AT_U21` as qualifying GPU training, as showing any branch change harmless, or as naming the more accurate device; it covers only the tested updated state and two batches. Do **not** use it to re-judge entry 32, which remains a FAIL, or to explain Stage 3's value drift or its differing logged scalars. Do **not** run authorized GPU work without `MLX_ENABLE_TF32=0` in that process. Training, a candidate checkpoint, held-out access, persistent environment changes, bug-report submission and pushing need their own approval.

83. **The decision to plan a bounded D2 learning trial (after entry 35), 2026-10-23.** Do **not** run further numerical diagnostics as the next step, or make the trial wait on proving the runtime correct for every state. Do **not** tune the recipe (anchor, learning rate, surface, steps, criteria) together with the numerical change. Do **not** open, train on or re-judge with the original 12 held-out games, or use the old Stage 3 candidate. Do **not** top up, replace or retry held-out games, training or a candidate: one candidate, and a STOP or VOID closes the trial. Do **not** judge the candidate by lower loss alone, or reuse a strength baseline measured under the old setting. Do **not** freeze, build, register seeds, generate, train, open a gate or push without the user's approval of each step.

84. **The D2 bounded trial's held-out generation (entry 36) — INTERRUPTED by a kernel panic, 2026-10-25; the trial is CLOSED.** Do **not** retry, resume, replace, extend or top up the generation, reuse the block [202692300, 202692312), or open gate 45 or 46 for this trial again. Do **not** train or score on the partial capture in `~/twixt-cache/search-distillation-d2/heldout-tf32off-20261025T133651Z/`, and keep it unchanged. Do **not** synthesize an `outer.json`, `record.json` or exit status for the run. Do **not** read the interruption as a learning result, as evidence against the learning hypothesis, or as establishing the panic's cause (the workload is neither blamed nor cleared). Do **not** run a stress test or reproduction of the panic without approval. Any new trial needs a new card, fresh seeds and the user's approval of each step.

85. **The decision to make one fresh attempt at the D2 bounded trial (after entry 36), 2026-10-25.** Do **not** make a third attempt automatically: if another kernel panic interrupts attempt 2, stop all further GPU experiments pending the user's system-stability decision. Do **not** tune the recipe, criteria, ceilings or numerical setting, or start a learning-method or panic investigation (stress tests, reproductions, system changes), before attempt 2 has had a completed test. Do **not** reuse attempt 1's seeds, records or partial capture, or admit attempt 1's held-out records into training. Do **not** schedule generation or training without a fresh read-only panic check and the user's approval of each step.

86. **The D2 bounded trial attempt 2's held-out generation (entry 37) — COMPLETED and eligible, 2026-10-25.** Do **not** regenerate, rerun, replace, extend or top up the held-out set, reuse the block [202692400, 202692412), or open gate 45 for attempt 2 again. Keep the capture in `~/twixt-cache/search-distillation-d2/heldout-tf32off-attempt2-20261025T214428Z/` unchanged; its games are held-out scoring data for attempt 2's one training run only, never training or strength-comparison data. Do **not** read the completed generation as a learning result or as evidence about the kernel panic's cause. Do **not** open gate 46 or schedule training without a fresh read-only panic check, an idle machine and the user's approval; it runs at most once, with the frozen recipe, criteria and 30-minute ceiling.

87. **The D2 bounded trial attempt 2's training run (entry 38) — STOP 2026-10-25; the corrected-configuration trial is CLOSED.** Do **not** rerun, retry or extend the training, run it with other seeds, or retune the recipe (anchor weight, learning rate, updates, batch, clipping, L2, surface, numerical setting) and judge it on the same corpus or held-out set. Do **not** open gate 46 again, re-score the candidate on new criteria, or select it after the fact. Do **not** screen, evaluate or promote the candidate (`~/twixt-cache/search-distillation-d2/student-tf32off-20261026T000538Z.safetensors`, read-only research artifact): a STOP does not permit the strength screen. Do **not** read the STOP as explaining the value drift, or the absence of gradient spikes as a qualification of GPU training in general. Any further distillation work needs a new written design and the user's approval.

88. **The dual-encoder policy-branch feasibility card (decision note after entry 38), 2026-10-26.** Do **not** build, load a model, account seeds, generate, train, open a gate, play games or push for it without the user's decision on the card and approval of each step. Do **not** judge the composite at equal simulations (400 vs 400) as a strength result: that gives it twice the network compute. Do **not** play or score the student file outside the composite: its own value head reads the trained encoder, which would be the closed D2 recipe with its anchor removed (#87). Do **not** reuse the old 108 training games or either spent held-out set (#69, #86, #87). Do **not** read the card's arithmetic (the log-linear Elo reading of Stage A, entry 11's cost precedent) as measurements on this machine and setting.

89. **The dual-encoder policy-branch proposal — REJECTED on paper, 2026-10-26 (nothing executed).** Do **not** build, load a model, account seeds, generate, train, gate or play for this card. Do **not** cite the rejection as evidence that dual encoders, value preservation, capacity or distillation fail, or as a strength result: nothing was measured. Do **not** present the card's +175 Elo as a measured requirement (it is an illustrative scenario: log-linear gains from the old protocol, target 0.60), entry 11's 0.594 as isolating the cost of halving simulations (that match also changed the evaluator), or the 5–10% figure as a calibrated forecast (a subjective assessment, not adopted). Reviving the design needs new evidence that justifies its inference cost, a new card and the user's approval.

90. **The mirror-averaging + root-halving combination card (DRAFT, 2026-10-26).** Do **not** build, load a model, account seeds, use the GPU or play games for it without the user's review of the card and approval of each step. Until the user approves §7's narrow reopening, #47, #59 and #62 stand unchanged; even then they open only for this one frozen combination, never for either component alone, another budget, candidate count, score, tie, start ply or orientation, a rerun or extension of entries 11 or 15, or a second combination. Do **not** use entry 11's or 15's scores as current baselines, add them, or present the hypothesis that averaging helps halving as measured. Do **not** treat simulation counts as proof of matched compute: the recorded network evaluations and runtime decide (card §4). A win would not establish synergy or attribute the gain to either component.

91. **The mirror-averaging + root-halving exceptions and frozen build (2026-10-26).** #47, #59 and #62 are reopened ONLY for card §2's frozen candidate and C1's two named identity controls (averaging alone at plies < 20; halving over a pass-through wrapper at plies ≥ 20; both at 200 simulations). Do **not** run either component in a match, change a budget or candidate parameter, or let C1's measurements change one: they only price later runs. Do **not** open gate 47 without the user's approval, an idle machine and a fresh panic check; it runs once under its 15-minute ceiling. Do **not** build or run the screen or strength wrappers, plans or seed blocks before C1's result is reviewed; their ceilings come from C1 by card §6. Do **not** edit the card or a pinned module without re-freezing its pins. Do **not** read the loop check's readout-difference counts (fake evaluators) as evidence about play.

92. **C1 of the mirror-averaging + root-halving combination (approved once, 2026-10-26).** Do **not** retry, rerun or extend C1, or open gate 47 a second time: a STOP, VOID or timeout closes the combination. The accepted loop-check repair is not a precedent for any retry. Do **not** open gates 48 or 49, build the screen/strength wrappers, plans or seed blocks, or play a game without the user's review of C1's record and a separate approval. Do **not** rerun the step-B regression; its log is preserved as evidence.

93. **C1's PASS (entry 39), 2026-10-26.** Do **not** rerun C1, open gate 47 again, or "correct" its timings under a quieter machine: it ran once, under recorded contention, and stands as recorded. Do **not** read the PASS or the 0.82 seconds ratio as evidence about playing strength, or the derived hours as whole-game measurements. Do **not** build the screen/strength wrappers, plans or seed blocks, open gates 48 or 49, or play a game without the user's review of entry 39 and a separate approval. Before any later GPU run, wait on the idle check (load below 3, no process above 50% CPU) rather than only printing it.

94. **The screen/strength build (addendum to entry 39), 2026-10-26.** Do **not** run a seed survey, proof, registration or closing, commit a `plan.json`, open gates 48 or 49, or launch either wrapper without the user's review of the build and a separate approval for each step. Do **not** bypass the idle gate (for example by launching the module directly or editing the wrapper's precondition), relax its limits, or change a ceiling: they are C1's recorded projection × 1.5, undiscounted. Do **not** change the candidate, budgets, guards or stopping rules. A busy refusal by the idle gate is not a run; any other STOP, VOID or timeout closes the combination without retry.

95. **The screen's seed survey and pre-proof (2026-10-26).** Do **not** rerun either mode, choose another block, or adjust [202692500, 202692564): the rule chose it and both records stand. Do **not** register it, commit the screen's `plan.json`, run the post-registration proof, open gate 48 or launch the screen without the user's registration review and separate approval of each step. Do **not** run any strength-comparison seed procedure yet: its block must be surveyed against the registry as it stands when that step is approved.

96. **The screen block's registration and post-proof (2026-10-26).** Do **not** rerun the post-registration proof, edit the screen's `plan.json`, or move, widen, expose or retire [202692500, 202692564) outside its closing procedure. Do **not** open gate 48 or launch the screen without the user's screen approval; the launch stays behind the idle gate. Do **not** start any strength-comparison seed procedure until it is separately approved.

97. **The screen (2026-10-26), CONTINUE.** Do **not** rerun, extend or rescore the screen, or read its 0.5703 (32 pairs) as a strength result or an effect estimate. Do **not** pool its games into the strength comparison, use them to choose a parameter, or change the candidate, budgets, guards or the strength verdict rule. Do **not** run its seed closing, any strength-comparison seed procedure, or open gate 49 without separate approval of each step. All 64 seeds were started, so the closing must expose the whole block.

98. **The screen's closing (2026-10-26).** Do **not** rerun K1, re-derive usage, or edit, move or reuse [202692500, 202692564) in any registry: it is EXPOSED and RETIRED whole. Do **not** retry the screen. Do **not** run strength-comparison seed work, open gate 49, train, or change the strength ceiling (55,601 s) without separate approval.

99. **The strength comparison's seed survey and pre-proof (2026-10-26).** Do **not** rerun either mode, choose another block, or adjust [202693600, 202694618). Do **not** register it, commit its `plan.json`, run the post-registration proof, open gate 49 or launch the strength comparison without the user's registration review and separate approval of each step. Do **not** change its 509 pairs, the Hoeffding rule, the guards or the 55,601 s ceiling, or pool the screen's games into it.

100. **The strength block's registration and post-proof (2026-10-26).** Do **not** rerun the post-registration proof, edit the strength `plan.json`, or move, widen, expose or retire [202693600, 202694618) outside its closing procedure. Do **not** open gate 49 or launch the strength comparison without the user's run approval; the launch stays behind the idle gate. Do **not** change the 509 pairs, the Hoeffding rule, the guards or the 55,601 s ceiling.
101. **The strength comparison → TIMEOUT / VOID (2026-10-27).** Do **not** retry, rerun, extend or top up the comparison, and do **not** score, pool or reuse its 799 partial games: an incomplete run receives no score. Do **not** reuse [202693600, 202694618); the block is consumed and awaits closing, which needs its own authorization. Any later GPU run on this machine must keep background compute (BOINC) out for its whole duration, not only at launch: the idle gate checks the launch only.
102. **The void strength block's closing (2026-10-27).** Do **not** rerun K1, edit the K2 intervals, or reuse any seed of [202693600, 202694618). Do **not** run the one new attempt on the old plan or evidence paths, or with any change to the 509 pairs, the Hoeffding rule, the two compute guards or the 55,601 s ceiling. Do **not** launch it unless BOINC is suspended for the whole run.
103. **The attempt-2 build (2026-10-27).** Do **not** run its seed procedures, register a block, create its plan, open gate 50 or launch it without the user's approval of each step. Do **not** change the amendment without re-pinning it under review, or weaken the BOINC check to match command lines (it voided on a mention) or to exempt more than `/opt/podman/bin/`. Do **not** make a third attempt: the exception covers this one only.
104. **Attempt 2's seed preparation (2026-10-27).** Do **not** rerun the survey or either proof, edit the strength2 `plan.json`, or move, widen, expose or retire [202695700, 202696718) outside its closing procedure. Do **not** open gate 50 or launch attempt 2 without the user's launch approval. The launch stays behind the idle gate and the BOINC check, with BOINC quit until the run finishes.
105. **Strength attempt 2 → COMPLETED, PASS (2026-10-28).** Do **not** rerun, extend, top up or re-score it, recompute its verdict under another rule, or pool attempt 1's 799 games or the screen's 64 into it. Do **not** reuse [202695700, 202696718); the block is consumed and awaits closing, which needs its own authorization. Do **not** read the PASS as evidence of synergy, of either component's benefit, of strength at other simulation counts, boards or opponents, or as a deployment decision. Do **not** tune the combination's parameters (16 candidates, ply 20, 200 vs 400) on the strength games.
106. **Attempt 2's closing (2026-10-28).** Do **not** rerun K1, edit the K2 intervals, or move, widen or reopen [202695700, 202696718); every seed is refused. Do **not** run another strength match, extension or replication to "confirm" the accepted PASS, re-score attempt 2 or pool attempt 1's 799 void games with it. Do **not** treat the PASS as a deployment, training or product decision, or as evidence for either component alone; each needs its own approval.
107. **The combination policy pilot — card frozen, step 1 build only, #66 exception (2026-10-28).** Do **not** run its throughput check, corpus generation, training, screen or strength comparison, run any of its seed procedures, or load the checkpoint for it, without the user's approval of each step. Do **not** change the frozen card -- the objective (the mirror-averaged policy), the target, the recipe and seeds, the stopping rules or the rule that all 144 throughput games finish inside 15 minutes -- without re-pinning it under review. The #66 exception covers reusing `ResidualPolicyHead` in this pilot only: do **not** cite the pilot or its outcome as evidence about the policy head's capacity, and do **not** extend the exception to any other use. Do **not** treat the combination's PASS as support for the pilot's target.
108. **The combination policy pilot's CPU timing (2026-10-28).** Do **not** rerun, repeat or extend the timing; it is one-shot and its record (`docs/superpowers/evidence/combination-policy-timing/`) prices step 4's ceiling at 45 min, which is never extended. Do **not** read it as a measurement of real-data training, of the GPU, or of anything about the target or the candidate's strength. Its approval did not authorize the throughput check, seed procedures, checkpoint loading, generation, training or games; each still needs its own approval.
109. **The pilot's step 1 accepted; the throughput launch build (2026-10-28).** Do **not** run the throughput check's seed survey, proofs, registration or closing, open its gate, load the checkpoint or launch the wrapper without the user's approval of each. Do **not** build the generation or training wrappers or their seed copies under this approval. Do **not** read a throughput timeout as evidence about whether policy learning works: it rejects only this setup's operational feasibility (and, as frozen, closes the pilot). Do **not** rerun the timing; its record stands as accepted.
110. **The throughput seed survey and the supervisor repair (2026-10-29).** Do **not** rerun the survey, adjust its block or select another one if it is not CLEAN; do **not** run the pre-proof, register the block, write the plan or run the post-proof without the user's approval of each. Do **not** open the throughput gate, load the checkpoint or launch the wrapper until the supervisor repair is reviewed: a supervisor that stops escalating when the main process exits does not enforce the ceiling, and a survivor found afterwards is not a substitute.
111. **The throughput seed survey record and the supervisor repair (2026-10-29).** Do **not** rerun the survey or edit its record (`02_seed_survey_run.txt`, sha256 `dbe3e91c…`); [202696900, 202697044) is surveyed, not registered, and becomes usable only through the separately approved pre-proof, registration with its plan, and post-proof. Do **not** remove the supervisor's group wait, or treat a group that outlived its main process past the ceiling as COMPLETED.
112. **The throughput pre-proof, the bounded post-KILL wait, and TIMEOUT over COMPLETED (2026-10-29).** Do **not** rerun the pre-proof, register the block, write the plan or run the post-proof without the user's approval of each. Do **not** launch with a supervisor whose wait after KILL is unbounded. Do **not** admit generation, or read the pilot as continuing, from a throughput run whose `outer.json` says TIMEOUT, whatever its `record.json` says; keep both records as written, and do **not** rewrite either to agree.
113. **The throughput pre-proof and the bounded wait (2026-10-29).** Do **not** rerun the pre-proof or edit its record (`03_collision_proof_pre_run.txt`, sha256 `5f60b891…`); register the block, write the plan or run the post-proof only on the user's approval. Do **not** remove the supervisor's bound or read exit 125 (cleanup unconfirmed) as anything but TIMEOUT with survivors; closing refuses it.
114. **The throughput registration and the admission fix (2026-10-29).** Do **not** register any block but [202696900, 202697044), change any registry list but ACCOUNTED, or write the plan by hand rather than through `build_plan`. Do **not** run the post-proof, open the gate, load the checkpoint, launch or close without the user's approval of each. Do **not** admit generation on the inner record alone, or weaken the outer checks (zero inner and outer exits, gate restored, survivors exactly empty). Do **not** push without approval.
115. **The throughput registration and the outer-record admission (2026-10-29).** Do **not** edit the plan, move or widen [202696900, 202697044), or register it again; the post-proof runs once, only on approval. Do **not** relax `outer_faults` or the committed reader's tracked-file check, or admit generation through any path that skips `outer.json`. Do **not** read the older registry-test failures as caused by this registration; they predate it.
116. **The throughput post-proof (2026-10-29).** Do **not** run the post-registration proof more than once, on a dirty tree, or against anything but the committed registration (`88c5d162`) and the unchanged plan; do **not** rerun or overwrite its record whatever its verdict. Do **not** open the gate, load the checkpoint, launch, close or push on this approval.
117. **The throughput post-proof record (2026-10-29).** Do **not** rerun the post-proof, or edit or regenerate `04_collision_proof_post_run.txt` (sha256 `03d04570…`). Do **not** change the registry or the plan before the run without a new proof: the lifecycle test pins the proved registry hash to the current file. Do **not** open the gate, load the checkpoint, launch, close or push without the user's approval of each.
118. **The one throughput launch (2026-10-29).** Do **not** run the throughput wrapper more than once, retry, extend, top up or rerun any part of a started run, or change its ceiling (TERM 890 s, KILL 900 s, observation to 910 s). Do **not** open any gate but the throughput gate, or open it if the registry or plan differ from what the post-proof proved: that returns for review, not to an automatic proof rerun. Do **not** count an outer TIMEOUT as a completed run, or use the 144 throughput games as training or strength evidence. Do **not** close, generate the corpus, train on real data or push without approval.
119. **The combination policy pilot, closed at the throughput check (2026-10-29).** Do **not** retry, extend, top up or rerun the throughput check, raise its 15-minute ceiling, reuse its block or its 144 games (never training or strength evidence), or proceed to generation or training: the pilot is closed by its own rule (card §5 step 2). Do **not** read the TIMEOUT as evidence about policy learning, or the descriptive games/h as the card's measurement (the record's `throughput` is null). A new pilot, a card amendment or seed closing needs the user's decision. Do **not** chain a merge after a commit whose failure a pipe can hide.
120. **The throughput seed closing (2026-10-29).** Do **not** run K1 more than once, or edit the registry if it does not RECONCILE. Do **not** expose only the 131 recorded starts: with no final `claims` event the frozen rule exposes the whole block. Do **not** change ACCOUNTED or TEST_ONLY. Do **not** retry, amend, profile, generate a corpus, train or push on this approval.
121. **The throughput block, closed (2026-10-29).** Do **not** reuse, unexpose, unretire or re-register any seed of [202696900, 202697044), rerun K1 or K2, or edit `02_derive_usage_run.txt` (sha256 `502e164a…`). Do **not** use the 113 finished throughput games for anything. Do **not** read the older registry-test failures (the same 16 failed / 69 errors before and after K2) as caused by this closing. A retry, amendment, profiling run, corpus generation, training or push needs a new decision.
122. **The throughput check, attempt 2 (2026-10-29).** Do **not** make a third attempt, retry, extend or top up attempt 2, or change its ceiling (TERM 1,790 s, KILL 1,800 s, observation to 1,810 s). Do **not** admit generation from attempt 1's records or paths, from an attempt-2 record without the pinned amendment's sha256, or without attempt 2's successful committed `outer.json`. Do **not** change the learning target, training recipe or evaluation criteria, repeat the CPU timing, profile or sweep parameters on this amendment. Do **not** edit the pinned amendment (sha256 `6dba686d…`) or attempt 1's evidence, closed seeds or registry entries. Seed procedures, checkpoint loading, GPU work, gameplay, training and pushing each need their own approval.
123. **Clarification of #122 (2026-10-30, the user's review).** #122's "a third attempt" and the amendment's "no further attempt, whatever happens" mean: one attempt is authorized now, with **no automatic** retry, extension or top-up. If attempt 2 times out or is interrupted, do **not** retry it on the spot; assess the cause, the likely cost and whether another attempt is worthwhile, and return that to the user, whose fresh decision any further attempt needs. Do **not** edit the pinned amendment to say this, and do **not** build a retry in advance.
124. **The one attempt-2 launch (2026-10-30).** Do **not** run the attempt-2 wrapper more than once, retry, extend, top up or rerun any part of a started run, or change its ceiling (TERM 1,790 s, KILL 1,800 s, observation to 1,810 s), workers, cap, checks or TF32 setting. An outer TIMEOUT overrides an inner COMPLETED record. Do **not** open any gate but gate 54, edit the registry before the closing derivation RECONCILES, or use the throughput games for training or strength evidence. After a TIMEOUT or interruption, assess before any further attempt (#123). Corpus generation, training and pushing need their own approval.
125. **The throughput check, attempt 2, VOID and closed (2026-10-30).** Do **not** reuse, unexpose, unretire or re-register any seed of [202697200, 202697344), rerun K1 or K2, edit attempt 2's run, closing or seed records, or use its games (143 finished) as training, throughput or strength evidence. Do **not** retry, extend or top up attempt 2. Do **not** read its 288 games/h (descriptive, VOID) as the pilot's measurement. Any further attempt is the user's fresh decision after assessing the cause (the 280-move state-cap tail: four games took 686–1,091 s), the cost and the worth (#123).
126. **Generation without a completed throughput check (2026-10-30).** Do **not** read either throughput record (each VOID under an outer TIMEOUT) as a pass, rewrite either, or admit generation through them; do **not** use either throughput run's games for training. Do **not** change the generation ceiling (TERM 21,590 s, KILL 21,600 s, observation to 21,610 s), the recipe, the target, the workers or the cap under this amendment. Do **not** run the generation seed survey under the existing rule (its 1,000-seed block holds in-band repository literals, so it cannot be CLEAN); do **not** adjust the rule, its gap floor or the block size, edit or remove the H4 fixture literals to make room, or move to a new band without a reviewed method. A generation timeout is VOID and is assessed by the user; no automatic retry, extension or top-up.
127. **The two-band seed method (2026-10-30).** Do **not** run the generation survey, either proof or the registration without the user's approval of each; each runs once. Do **not** adjust the rule, the band, the 1,000-seed gap floor or the block after the survey reads its literals: a NOT CLEAN or refused block is returned for review. Do **not** edit, add or remove repository literals in either band to steer the rule or the check, and never write a candidate seed of the 2027 band into `scripts/`, `tests/` or `docs/superpowers/` outside the survey's own records, the plan and the registry tuple. Do **not** rehearse the survey on the real repository: the synthetic rehearsal is a code check, not a survey. Do **not** admit training on generation's record alone or weaken `outer_faults`.
128. **Generation's seed block (2026-10-30).** Do **not** rerun the generation survey or either proof, edit their records (`02_*` sha256 `69ce5b87…`, `03_*` `295ce85c…`, `04_*` `f3c3d5dc…`), edit the plan, or move, widen or re-register [202739800, 202740800). Do **not** open the generation gate, load the checkpoint, launch, close the block, train or push without the user's approval of each. Do **not** write any seed of this block into `scripts/`, `tests/` or `docs/superpowers/` outside its own records, the plan and the registry tuple.
129. **The training corpus (2026-10-30).** Do **not** rerun, extend, top up or regenerate generation, or reuse [202739800, 202740800) (EXPOSED and RETIRED whole). Do **not** edit, move or delete the committed run evidence (`record.json` sha256 `df9ecaef…`, `games.jsonl` `e75ba513…`, `progress.jsonl` `db4aa7b9…`, `outer.json` `b6b2e7c8…`), K1's record (`ac143aba…`), or any of the 1,000 row files under `~/twixt-cache/combination-policy-pilot/generate/`: training verifies each against `games.jsonl` and refuses a changed one. Do **not** read the games' winners or reasons outside a reviewed training or analysis step. Do **not** open the training gate, train, change the recipe or the training ceiling, or push without the user's approval of each. These games are never strength evidence; the throughput runs' games stay excluded.
130. **The training wrapper and ceiling (2026-10-31).** Do **not** change the training ceiling (2,820 s: TERM 2,810 / KILL 2,820 / observation 2,830), relabel or edit the timing record, or rerun the timing. Do **not** open the training gate or launch without the user's approval; the wrapper runs once, and a run the ceiling stops is VOID and closes the pilot (card §5) — no retry, extension or top-up. Do **not** change the recipe, the starting checkpoint, admission or the STOP rules for this run, and do **not** push without approval.
131. **Clarification to #130 (2026-10-31, the user's wording).** A training run the ceiling stops is **VOID**, with **no automatic retry** — it does not permanently reject the learning method; its progress, cause and likely continuation cost go to the user for assessment. Do **not** retry, extend, top up or rerun it without a new approval. Do **not** run strength games, deploy, or push on an ELIGIBLE verdict without the user's approval of each.
132. **The trained head (2026-10-31).** Do **not** rerun, extend or continue training, change the recipe, or edit the run evidence (`record.json`, `progress.jsonl`, `head.npz` sha256 `ead156e2…`, `outer.json`). Do **not** read ELIGIBLE as a strength result: it is a held-out fit on the same agent's games. Do **not** run the screen, any strength games, deploy the head or push without the user's approval of each; the screen needs its own registered seed block and ceiling.
133. **The screen's preparation (2026-10-31).** Build only. Do **not** run the screen's seed survey, proofs, registration or closing, open its gate, load the checkpoint or the trained head for evaluation, play any GPU game, train further, deploy or push without the user's approval of each. Do **not** change the screen's frozen rule (STOP below a mean pair score of 0.40, otherwise CONTINUE) or read CONTINUE as a strength result.

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

## Post-closure work, 2026-10-11 → 2026-10-16

Five separately authorized lines run after the external-engine trial. None is a four-gate (A/B/C/D) calibration
branch, so the gate table above does not apply; each is recorded in this ledger's long-form fields. **Where no
prediction was stated before a run, the entry says so — none is backfilled.** `calib020_0001` remained the
incumbent throughout; nothing was pushed.

### 1. Search distillation — A1, Stage A PASS, Stage B NON-PASS (2026-10-11 → 2026-10-14)

**Test.** A1: one colour-reversed pair, `calib020_0001` at 1,600 vs 400 simulations (planning measurement and
wiring check). Stage A: 80 colour-reversed pairs (160 games), the same two budgets. Stage B: 200 fresh
self-play games at 400 vs 400; on 40 held-out games, the incumbent's eight policy-head tensors (519 parameters,
encoder, value head and BN running statistics frozen, eval mode) trained on 1,600-simulation visit
distributions (student), compared with the same procedure on 400-simulation targets (control), the unchanged
incumbent, and the incumbent with one fitted temperature.

**Hypothesis.** The incumbent's 1,600-simulation visit distributions are a policy target that makes the network
stronger when it plays at 400 simulations (proposal `docs/superpowers/2026-10-11-search-distillation-proposal.md`
§1).

**Why it was plausible.** Stronger search is the standard source of better policy targets; Stage A was designed
to establish first that 1,600 simulations actually play better than 400 under our exact engine and rules.

**Prediction on record.** Proposal §8: Stage A passes about 50%; the full chain passes Stage C at most 10%.
Frozen bars: Stage A PASS iff `mean − sqrt(ln(40)/(2·80)) > 0.5`; Stage B needs L1 and L2 each ≥ 26/40
held-out game wins and L3 student top-1 strictly above control and incumbent.

**What ran.** A1 `09741b3e`; Stage A run `aa5d2992`, analysis `480e0f9b`, closing `a46623b7`; Stage B phases
G `414afe5e`, P `ec639584`, L1 `a02f8a9a`, Q `1358bb0e` (QUALIFIED), L2 `2ccc7fca`, T `30da129a`, closing
`6299beaf`, closeout memo `c69224c4` (`docs/superpowers/2026-10-14-search-distillation-stage-b-closeout.md`).

**Result.** Stage A **PASS**: 1,600 simulations scored a mean pair score of **0.76875**, bound **0.61691**
(> 0.5; independence of pairs assumed). Stage B **NON-PASS**: L1 27/40 (held); L2 **18/40 jointly** (student vs
temperature alone 25/40, vs incumbent alone 25/40) — fail; L3 fail (top-1 student 0.140 < control 0.158 <
incumbent = temperature 0.181). Mean KL: student 1.510, control 1.565, temperature 1.556, incumbent 1.671.

**Did it match the prediction, and why?** Stage A passing matched a ~50% forecast; one outcome validates no
probability. Stage B: the student fitted the targets better on average (lower KL) but did not beat the
fitted-temperature and incumbent arms per game, and lost top-1 agreement — the fixed recipe improved average
fit without the per-game advantage the criteria required. Possible confound, **not a proven cause**: Stage B's
labels were produced through the evaluation factory, which ran the network in **training-mode BatchNorm**
(entry 3), while the fitted models ran in eval mode.

**Conclusion.** Stage A shows stronger search at 1,600 simulations. Stage B closed NON-PASS; no strength game was
played, so there is no strength evidence for the student either way. Candidates kept as research artifacts
(`runs/search-distillation-stage-b/`, untracked). See do-not-repeat **#54**.

### 2. Play-path search speed — profile, one-sync screen, batching screens (2026-10-14 → 2026-10-15)

**Test.** Make Stage A's stronger search affordable with the incumbent's weights fixed. Evidence:
`docs/superpowers/evidence/search-profile/`.

**Hypothesis.** (a) Removing one of the two GPU syncs per `infer` call cuts search time ≥ 10% with identical
results. (b) Batched search (`search_from_root`, batch 14) at 1,600 simulations (B1600) fits within the
sequential 400-simulation search's time (S400).

**Why it was plausible.** The profile (`b8cf2adf`) found `infer` at 71% of 3.16 ms/simulation, always batch 1,
with forward-through-first-sync 86.7% of `infer` (`a2e6e5cf`): batching amortizes exactly that cost.

**Prediction on record.** No probability forecast. Frozen bars: one-sync ≥ 10% reduction with exact matches;
B1600/S400 ≤ 1.00.

**What ran / result.**
| Screen | Commit | Result | Verdict |
|---|---|---|---|
| One sync instead of two | `77eff8a5` | 12/12 exact matches; 7.8% faster (per fixture 7.3 / 7.0 / 9.1%) | SCREEN_NON_PASS (bar 10%) |
| B1600 vs S400, variable shapes (training-mode BN) | `20badd1a` | 7,330 vs 3,768 ms, ratio **1.945** | NOT_FEASIBLE |
| B1600 fixed 14×528 vs S400, eval mode | `e7cd3674` | 4,933 vs 3,413 ms, ratio **1.445** (1.24 / 1.59 / 1.53) | NOT_FEASIBLE |

**Did it match, and why?** No. Padding removed the per-shape recompilation cost (padded calls ~9 ms each) and
B1600 spent *less* time in `infer` than S400, yet used about twice the process CPU: substantial time lies
**outside inference**. These measurements do not isolate how much belongs to selection, board operations or
input building.

**Conclusion.** The batching optimization path is CLOSED by its own closure condition (the fixed-shape
prototype also failed). The one-sync change was not adopted. See do-not-repeat **#55**.

### 3. Implementation finding — the evaluation factory ran BatchNorm in TRAINING mode (2026-10-15)

**Finding.** `eval_runner._default_evaluator_factory` loaded the network and never called `.eval()`. MLX modules
start in training mode, so every BatchNorm normalised each position with the statistics of the batch it was
given (in play, a batch of one) and assigned running statistics on each forward. Confirmed by
`dc34cac3` (14/14 BatchNorm, 42/42 modules in training mode; zero-padding a batch moved P1's value from 0.625 to
0.999). The trainer's self-play and inline evaluation already called `network.eval()`; the browser game uses an
ONNX export made in eval mode (`export_onnx.py`), so **the product was not affected**.

**Fix.** `c8cb2aa6`: the factory calls `net.eval()`. Verified by `0a7aa8e9` (0/42 modules in training mode;
padded vs ordinary within 1e-5; repeats bit-identical).

**Consequence for this ledger.** **Every result produced through `eval_runner`'s default factory before
`c8cb2aa6` ran in training-mode BatchNorm.** Those results stay valid **under their original protocol** and
are not re-scored. This is related to, but distinct from, the 2026-06-30 BatchNorm calibration confound above
(which concerned training-time forwards). Future research uses corrected inference.

**Still open (found 2026-10-16, not fixed):** `probe_eval.py`'s own entry point (`main`) also builds its
`LocalGPUEvaluator` from a network it never puts in eval mode, so measurements taken through that entry point
were training-mode too. Twelve other scripts that load through `load_network_for_scoring` do call `.eval()`.
Comparisons made under one mode on both sides are internally consistent; what does not transfer is a
pre-fix number reused as a baseline for an eval-mode run — re-measure the baseline instead.

### 4. Inference-mode match — corrected vs legacy, 32 games (2026-10-16)

**Test.** `calib020_0001` in eval mode (corrected) vs the same checkpoint in training mode (legacy), separate
network copies, 400 simulations each, Stage B protocol, 16 colour-reversed pairs, one fresh accounted 32-seed
block (survey → registration → closing).

**Hypothesis.** The inference correction helps the model play.

**Why it was plausible.** Training-mode BatchNorm makes each evaluation depend on its batch; eval mode is the
intended inference behaviour and what the trainer and browser already use.

**Prediction on record.** None stated in advance. Exploratory screen, no decision rule.

**What ran.** Scripts `9bed85bc`; run `4380bfbe` (outer exit 0, 2,093 s, COMPLETED, no survivors; modes held
at load and end); closing `a31b045d`.

**Result.** Corrected **20–12** (no draws, no caps); pair scores 6 × 1.0, 8 × 0.5, 2 × 0.0; mean pair score
**0.625**, Hoeffding 95% [0.285, 0.965] — descriptive, conditional on the assumed independence of the pairs,
which distinct seeds and colour reversal do not themselves establish.

**Conclusion.** Encouraging but **inconclusive**; the interval includes 0.5. Closed; keep the factory fix. See
do-not-repeat **#56**.

### 5. Self-play diagnostic and instrumented probe — a concrete defence failure (2026-10-16, OPEN FINDING)

**Test.** Four self-play games of `calib020_0001` in eval mode (400 simulations, Stage B protocol, fresh 4-seed
block, diagnostic material only), scanned with the H5 review's exact one-ply labels; then one instrumented cold
400-simulation search of the one hit.

**Hypothesis.** A small fresh sample shows at least one verifiable tactical mistake (missed immediate win or
avoidable one-move loss) that can be traced to a mechanism.

**Prediction on record.** None stated in advance.

**What ran.** Scripts `88a5f759`; run `ee718688` (outer exit 0, 172 s; four games of 41–43 plies); scan
`e8ef077c`; closing `8ed243fb`; probe script `beab7e9f`, probe `c8ad5050`.

**Result.** One **MISSED_DEFENCE** (game 3, ply index 41, black): red threatened (23,6) and (23,8); the only
saving move was (21,6); black played (14,19) (sampled at temperature 0.1, 1 visit) and red won next move.
In all four games, once the losing side's root value fell below about −0.82, its 400 visits went one per
child (23 positions). The probe (fresh cold search, test-only seed): **all 400 root selections chose an
unexplored move** — FPU 0 plus the exploration bonus (~0.005–0.053) beat the best visited score (~−0.93 to
−0.86) by 0.91–0.99 every time; 400 children × 1 visit, 88 never visited, no opponent reply searched anywhere.
(21,6): prior 0.0027 (rank 7 of a nearly flat policy, max 0.0031), reached at root visit 6, Q −0.959 — only
**5th best** of 400 visited; (22,7), (21,8), (22,6) and (20,8) were valued higher though each loses to an
immediate red win.

**Interpretation.** The save was **reached once and never examined further**: with absolute FPU 0, no root
move gets a second visit once visited Q is near −1, so the search never reaches the depth where red's winning
replies become terminal, and one network evaluation does not separate the save from immediate losses. One-ply
only: whether black's position was savable after (21,6) was not checked. Not a strength claim.

**Status.** OPEN — the finding guides the choice of the next correction; no correction has been tested. See
do-not-repeat **#57**.

### 6. Forced-loss check of the diagnostic example — rules engine only (2026-10-19)

**Test.** In entry 5's position (game 3, ply index 41), apply the save (21,6); can red force a win within three
further plies (red, black, red)? Exact checks only (`d0_postmortem.one_ply_threats`,
`h5_weakness_review.saving_moves_of`); no model, search, seed or game.

**Hypothesis.** None tested as a hypothesis: the question was whether this example can support a search
correction — avoiding the immediate loss does not by itself establish that black could save the game, and the
network's pessimism (Q −0.959, a value estimate, not a calibrated probability) might be justified.

**Prediction on record.** None stated in advance.

**What ran.** Script `fa11c06c`; one run, evidence `3c49042c`
(`docs/superpowers/evidence/self-play-diagnostic/forced-loss/`): `timeout -k 10 50`, exit 0, COMPLETED, 8.7 s.

**Result.** **FORCED_WIN_WITHIN_3.** After (21,6) red has no immediate win, but four red moves force one:
(21,3) threatening (23,2)/(23,4); (22,4) threatening (23,2)/(23,6); (22,6) threatening (23,4)/(23,8); (22,9)
threatening (23,7)/(23,11). For each, every one of black's 486 legal replies is answered by a red winning move,
each verified by applying it. Of the other 482 red moves, 476 leave red no immediate threat and 6 are answered
by a saving black reply (each confirmed by a full scan); none lets black win immediately.

**Did it match, and why?** No prediction. The position at ply index 41 was **already lost** within this horizon:
the save delays the loss by two moves.

**Conclusion / status.** Entry 5's mechanism stands (the search goes flat and never deepened the save), but
**this example does not show a search correction would have changed the result** — the network's pessimism was
justified here. The example is closed as evidence for a correction. See do-not-repeat **#58**.

### 7. Addendum to entry 3 — `probe_eval.py` loader fixed (2026-10-19)

`_load_network` now calls `net.eval()` after loading the weights (`444269ac`), covering
`load_network_for_scoring` and the probe CLI's direct call; one loader-mode test (red on the old loader, green on
the fix). `eval_runner`'s explicit call is kept; `LocalGPUEvaluator` and functions given a trainer-owned network
are unchanged. Also now eval mode: `twixt_replay_analyzer.py`'s shared scoring network, the one other caller that
did not call `.eval()` itself. No historical probe is rerun; earlier probe results stay under their recorded
protocol (training mode). The inference-mode match's `legacy_factory` loaded through this function, so that closed
one-shot script would now STOP on its own mode check rather than reproduce its legacy arm.

### 8. Mirror-consistency probe — position vs left-right reflection (2026-10-19, OPEN FINDING)

**Test.** Four positions from the four diagnostic games (entry 5; no held-out material): after 20 moves in games
0 and 2, after 21 in games 1 and 3. Each position and its left-right reflection — the complete move history
replayed with (r, c) → (r, 23 − c), so the engine rebuilds pegs and links — evaluated by `calib020_0001` in eval
mode, batch 1, exactly as `mcts.py` evaluates a leaf; reflected priors mapped back. Eight inference calls; no
search, game, seed or training.

**Hypothesis.** The network's predictions are not invariant under the left-right reflection, a symmetry of our
rules that preserves the true value; if so, averaging the two orientations might help.

**Why it was plausible.** Training uses horizontal-mirror augmentation (`self_play.py`, `_mirror_position_lr`),
which encourages but does not enforce invariance. No inference-time symmetry-averaging experiment is recorded.

**Prediction on record.** None stated in advance. Stop rule: if every difference were within 1e-5, stop the idea
on this sample.

**What ran.** Script `2077d2ef`; one run, evidence `d4081174`
(`docs/superpowers/evidence/self-play-diagnostic/mirror/`): `timeout -k 10 50`, exit 0, COMPLETED, 0.22 s,
0 modules in training mode, legal sets corresponding exactly under the map.

**Result.** **DISCREPANT.**
| Game | Moves | To move | Value | Value (reflected) | abs diff | Policy TV | Top move (reflected, mapped back) |
|---|---|---|---|---|---|---|---|
| 0 | 20 | red | 0.601 | 0.646 | 0.045 | 0.178 | (17,15) 0.122 vs (17,17) 0.107 — disagree |
| 1 | 21 | black | −0.213 | −0.849 | **0.637** | 0.122 | (17,11) both |
| 2 | 20 | red | −0.546 | −0.347 | 0.199 | 0.138 | (11,17) both |
| 3 | 21 | black | −0.978 | −0.842 | 0.136 | 0.108 | (4,11) both |

**Did it match, and why?** No prediction. The differences are four orders of magnitude above numerical tolerance,
so the network is materially orientation-dependent on these positions. Four positions from four games are not a
rate, and which orientation (or the average) is closer to the true value is not measured.

**Conclusion / status.** OPEN — the finding justifies *considering* an averaging prototype, not building one. Any
later comparison must account for averaging's doubled inference cost (e.g. equal wall-clock or equal network
evaluations). Not evidence of stronger play.

### 9. Implementation finding — training mirror augmentation kept the black edge-connectivity labels (2026-10-19)

**Finding (found in review).** `self_play._mirror_position_lr`, the training-time left-right mirror, permutes link
directions and swaps the black edge-*distance* channels (21/22), but did not swap the black edge-*connectivity*
channels **27** (black connected to the left edge) and **28** (right edge): it flipped their locations and kept
the wrong edge labels. So every mirrored 30-channel training sample with black edge-connected pegs carried
inconsistent connectivity labels.

**Fix.** `0c417882`: swap 27 ↔ 28 when the tensor has them; legacy 24-channel tensors are unchanged. Focused CPU
test (`tests/test_mirror.py::test_mirror_matches_engine_rebuilt_reflection_24x24`): a hand-built 24×24 position
with black components touching both goal edges, the helper's mirror compared channel by channel with the tensor
the engine builds from the reflected move history. Run once before the fix — failed on exactly channels
**[27, 28]**, every other channel matching — and once after — passed, the legacy 24-channel slice included. No
model, search, game or training.

**Consequence.** The incumbent was trained with this defect; its weights are unchanged. Whether the defect
contributed to entry 8's asymmetry is **unmeasured**. The probe in entry 8 did not use this helper (it rebuilt
mirrored states through the engine) and stands. No retraining or averaging strength trial follows from this fix;
symmetry averaging remains an open hypothesis.

### 10. Mirror-averaging prototype — one bounded correctness check (2026-10-19, OPEN)

**Test.** A fixed prototype with no parameters, `scripts/GPU/alphazero/mirror_averaging.py`
(`MirrorAveragingEvaluator`, imported by nothing in production): each `infer` evaluates the boards and their
tensor-mirrored counterparts (`_mirror_position_lr`, with entry 9's fix), reflected moves index-aligned, and returns
the mean of the two policy probability distributions and the mean of the two values. Weights, search and readout
unchanged. Checked on entry 8's four positions against a reference built independently from the original and the
**engine-rebuilt** reflected position (priors mapped back by move, then averaged).

**Hypothesis.** The prototype computes exactly the intended average (a correctness question, not a strength one).

**Prediction on record.** None stated in advance. Pass bar: |value| and |prior| differences ≤ 1e-5, finite
outputs, priors covering exactly the legal moves, non-negative and summing to 1 within 1e-5.

**What ran.** Code and check script `bf6d03ad`; one run in a clean linked worktree (fast-forwarded, removed),
evidence `3b11645a` (`docs/superpowers/evidence/mirror-averaging/run/`): `timeout -k 10 50`, exit 0, COMPLETED,
0.41 s; one checkpoint load, eval mode (0 modules in training mode), all 20 loaded repository modules committed;
**16** underlying inference calls.

**Result.** **PASS.** Max value difference **2.98e-08**, max prior difference **1.12e-08**; all outputs finite;
priors sum to 1 within 1e-7. Averaged values: game 0 0.623 (0.601 / 0.646), game 1 −0.531 (−0.213 / −0.849),
game 2 −0.447 (−0.546 / −0.347), game 3 −0.910 (−0.978 / −0.842).

**Did it match, and why?** The tensor mirror and the engine rebuild agree to float precision on these positions,
so the prototype averages what it claims to.

**Conclusion / status.** OPEN. Correctness only: it does not show averaging helps. Each prototype evaluation costs
two network evaluations, so any strength screen must be cost-matched (e.g. 200 averaged vs 400 ordinary
simulations, recording actual evaluations and runtime) and needs its own authorization.

### 11. Mirror-averaging match — 200 averaged vs 400 ordinary simulations, 32 games (2026-10-19, CLOSED)

**Test.** `calib020_0001` with `MirrorAveragingEvaluator` (entry 10) at 200 simulations ("averaged") vs the plain
evaluator at 400 ("ordinary"), each from its own eval-mode load; Stage B PROTOCOL otherwise (24×24, 280-ply cap,
opening-temperature readout, no root noise); 16 colour-reversed pairs, one fresh accounted 32-seed block (survey →
pre-proof → registration → post-proof → closing). Approximately inference-budget matched, not equal wall time.

**Hypothesis.** Averaging the network's predictions over the left-right reflection makes the incumbent play
better at about the same number of network evaluations.

**Why it was plausible.** Entry 8: predictions differ materially between orientations (value gaps up to 0.64);
entry 10: the prototype computes the intended average.

**Prediction on record.** None as a probability. **Frozen screening rule** (operational, not statistical proof):
averaged points ≥ 24 of 32 (mean pair ≥ 0.75) → consider a confirmatory study; below → stop this configuration
without changing budgets, averaging weights or orientations; an incomplete run gets no score.

**What ran.** Scripts `41620884`; survey `45ccc65f`, pre-proof `055693b1`, registration `3c8d5360`, post-proof
`b9acb1ab` (all CLEAN 45/45); run `7560d7bc` (outer exit 0, **1,540 s** of the 5,400 s cap, COMPLETED, no
survivors, both networks in eval mode at load and end); closing K1 `60a35d6e` (RECONCILES 15/15), K2 `a7b52290`
(block EXPOSED and RETIRED whole). Clean linked worktree, fast-forwarded and removed.

**Result.** Averaged **19–13** ordinary, no draws, no caps (games of 31–81 plies). Pair scores: 4 × 1.0, 10 × 0.5,
2 × 0.0 (in order 1, 1, .5, .5, .5, 1, 1, .5, 0, .5, 1, .5, 0, .5, .5, .5); mean **0.594**, unclipped Hoeffding 95%
**[0.254, 0.933]** — descriptive, pairs as the sampling unit, conditional on their assumed independence.
**19/32 < 24 → STOP_CONFIGURATION.** Cost per agent:
| Agent | Searches | Leaf calls | Network evaluations | Network s | Search s | Evaluations / search | s / search |
|---|---|---|---|---|---|---|---|
| averaged (200) | 824 | 161,804 | 323,608 | 542 | 696 | 392.7 | 0.845 |
| ordinary (400) | 821 | 323,983 | 323,983 | 553 | 840 | 394.6 | 1.023 |

**Did it match, and why?** The bar was not met. The point estimate favours averaging, but 10 of 16 pairs split
and the interval includes 0.5, so the screen cannot distinguish a small benefit from none. Network evaluations
were matched within 0.1%; averaging used about 17% less search time (half the tree operations), so the comparison
did not disadvantage it on runtime.

*Correction to entry 11 (2026-10-26):* the pair-score summary above ("4 × 1.0, 10 × 0.5, 2 × 0.0") and "10 of 16 pairs split" are wrong. The ordered list in the same entry, and the committed record's `.screen.pair_scores`, give **5 × 1.0, 9 × 0.5 (split), 2 × 0.0**, which totals the recorded 19 points. The mean pair score 0.594, the interval and the verdict are unchanged. Found during the 2026-10-26 combination review, which had repeated the error.

**Conclusion / status.** CLOSED for this configuration: no confirmatory study, no budget, weighting or
orientation change. Not evidence that averaging helps or hurts. `calib020_0001` and production defaults unchanged.
See do-not-repeat **#59**.

*Correction (2026-10-19, same day).* The pair-score counts above are miscounted: the record has **5 × 1.0, 9 × 0.5,
2 × 0.0** (the listed sequence is correct), so **9** of 16 pairs split, not 10. The mean (0.594), points (19) and
verdict are unchanged.

### 12. Evaluation-reuse measurement — exact duplicate evaluator inputs across consecutive searches (2026-10-19, OPEN)

**Test.** Game 0 of the self-play diagnostic corpus after 20, 21 and 22 moves (red, black, red to move); one
ordinary 400-simulation search of each (1,200 simulations) by `calib020_0001` in eval mode, Stage B search settings,
no root noise, each side its own MCTS over one shared evaluator as in `play_eval_game`, test-only RNG seed (a
diagnostic search, not a game). Every `infer` call keyed by the sha256 of its exact inputs (board tensor, move
rows, move columns, mask, active size) and timed; every inference ran normally — nothing cached or skipped.

**Hypothesis.** Consecutive searches re-evaluate identical inputs often enough that reusing predictions could make
existing search cheaper (no previous test of per-prediction reuse is recorded; existing caches reuse evaluator
objects or completed root searches).

**Prediction on record.** None stated in advance. **Frozen decision:** inference time on duplicates / search time
< 10% → stop; ≥ 10% → justifies only a small prototype (optimistic: cache overhead excluded).

**What ran.** Script `0cd47567`; one run in a clean linked worktree (fast-forwarded, removed), evidence `9bd5eeca`
(`docs/superpowers/evidence/eval-reuse/run/`): `timeout -k 10 50`, exit 0, COMPLETED, 3.9 s; 0 modules in
training mode; all 26 loaded repository modules committed.

**Result.** 1,203 calls (401 per search, batch 1), **977 distinct inputs, 226 exact duplicates — 0 within a
search, all 226 repeating an input from an earlier search** (28% of the 802 calls in searches 2 and 3). Inference
on duplicates **0.394 s** of **3.077 s** search time (hashing time, 0.057 s, subtracted) → **12.8% ≥ 10% →
JUSTIFIES_SMALL_PROTOTYPE.** Per search: 1.125 / 0.947 / 1.006 s search, 0.778 / 0.667 / 0.668 s inference (the
first includes one-time graph compilation, which only enlarges the denominator).

**Did it match, and why?** No prediction. With no transpositions inside one search, every duplicate is a
position the previous search(es) already expanded beneath the moves actually played — the subtree a reused tree
or prediction cache would keep.

**Limits.** Three positions from one game, one timing run, 2.8 points above the bar. The record does not split the
226 duplicates by which earlier search supplied them, so how many cross sides (red's search reused by black's,
which needs a cache shared by both agents — natural in self-play, not between different agents in a match) is
**unmeasured**. A cache's own lookup and memory cost is excluded. Not a route to stronger play by itself.

**Conclusion / status.** OPEN — clears the frozen bar, which justifies considering only a small prototype; nothing
is built or authorized. Batching and mirror averaging stay closed; `calib020_0001` unchanged.

*Follow-up (2026-10-19, review).* The small prototype this entry justified has since been **built, tested and stopped** — entry 13 (per-colour cache, STOP_CONFIGURATION, do-not-repeat #60). The "OPEN — nothing is built" status above is historical; do **not** treat it as an open invitation to build a cache. Correction to the Limits wording ("not between different agents in a match"): an exact-input prediction shared between two agents is valid **when both use the identical frozen evaluator and the inputs are exact** — it only avoids duplicate computation (entries 12 and 13 ran one checkpoint on both sides). Predictions from **different** models cannot be interchanged. This distinction does **not** authorize a shared-cache experiment.

### 13. Per-colour prediction cache — one bounded timing screen (2026-10-19, CLOSED)

**Test.** Entry 12's three positions, checkpoint and settings. A CACHE in the comparison script only
(`CachedEvaluator`): exact-input key as in entry 12, a miss runs the network and stores copies, a hit returns
copies; **each colour its own cache**, retained across that colour's searches in a sequence, never fed by the
opposing agent. Sequences (fresh MCTS objects, RNG streams and caches each) in the order **ordinary → cached →
cached → ordinary**, 4,800 search simulations, after an identical shared warm-up (one plain inference per root).

**Hypothesis.** Reusing a colour's own earlier predictions makes search cheaper after the cache's overhead.

**Why it was plausible.** Entry 12: 226 of 1,203 calls repeated an earlier input (12.8% of search time) — but
that count did not separate same-colour from cross-colour reuse.

**Prediction on record.** None stated in advance. **Frozen screen:** ≥ 10% lower combined search time (whole
`search_with_root`, hashing, lookup, storage and copying included) with every correctness check passing →
QUALIFIES; less → STOP_CONFIGURATION; any mismatch → STOP at once.

**What ran.** Script `f8c19d1d`; one run in a clean linked worktree (fast-forwarded, removed), evidence `938e35e4`
(`docs/superpowers/evidence/eval-reuse/cache-run/`): `timeout -k 10 50`, exit 0, COMPLETED, 12.7 s; eval mode; all
26 loaded repository modules committed.

**Result.** **Every check passed:** each search's visit counts, root value and most-visited move, and both colours'
final search-RNG states, identical to the first ordinary sequence. Cache, each cached sequence: **red 4 hits /
798 misses** (1.67 MB stored), **black 0 / 401** (0.84 MB). Search seconds by sequence: ordinary 3.151, cached
3.203, cached 3.017, ordinary 2.946; combined **ordinary 6.097 s vs cached 6.219 s → reduction −2.0% →
STOP_CONFIGURATION.**

**Did it match, and why?** No prediction. Entry 12's open question is answered: with no reuse inside a search
and almost none within one colour (4 of 802 later lookups), its duplicates were **almost entirely cross-colour** —
one side's search evaluating positions the other side had just expanded. A per-colour cache therefore has nearly
nothing to reuse and only adds overhead; the remaining difference is within the run-order drift visible between
the two ordinary sequences (3.151 vs 2.946 s).

**Conclusion / status.** CLOSED for this configuration. Cross-colour sharing (natural in self-play, but between
different agents in a match it means one agent using the other's predictions) was deliberately excluded and
was not tested. Not deployment or strength evidence; production unchanged. See do-not-repeat **#60**.

*Correction (2026-10-19, review).* The parenthesis above ("one agent using the other's predictions") is wrong as stated: an exact-input prediction shared between two agents is valid **when both use the identical frozen evaluator and the inputs are exact** — it only avoids duplicate computation (entries 12 and 13 ran one checkpoint on both sides). Predictions from **different** models cannot be interchanged. This distinction does **not** authorize a shared-cache experiment.


### 14. Mirror-defect exposure — how often old and corrected mirrored tensors differ (2026-10-19, MEASURED)

**Test.** Rules-only, no model, search, game, seed or training. Every pre-move position of the four committed
self-play diagnostic replays (games 0–3, 169 positions), tensor built as `play_game` records it. CORRECTED =
`self_play._mirror_position_lr` with entry 9's fix; OLD = CORRECTED with channels 27/28 swapped back (the swap is its
own inverse, so this is the pre-fix helper exactly). A position is EXPOSED iff OLD ≠ CORRECTED anywhere — its
mirrored training sample would have carried wrong black edge-connectivity labels. Integrity check: CORRECTED equals
the engine-rebuilt reflection at every position.

**Hypothesis.** None tested; this measures exposure to decide whether entry 9's repair deserves a controlled
training-and-strength trial (the proposed five-hour paired pilot was not authorized as written).

**Prediction on record.** None stated in advance. No decision rule was frozen; the count is descriptive.

**What ran.** Script `dc6c9a28`; one run in a clean linked worktree (fast-forwarded, removed), evidence `106a405c`
(`docs/superpowers/evidence/mirror-exposure/run/`): `timeout -k 10 50`, exit 0, COMPLETED, 0.22 s; integrity check
passed at all 169 positions; all 12 loaded repository modules committed.

**Result.** **8 of 169 positions exposed (4.7%)**: games 0, 1 and 3 (red wins) 0 of 43 / 41 / 43; **game 2 (the
black win) 8 of 42, plies 34–41**, every one with black pegs connected to the right edge (6–19 cells in channel 28)
and none to the left edge. Before ply 34 no position in any game had a black edge-connected peg. Self-play mirrors
each position with probability 0.5, so the expected exposed share of a training buffer is about **1.6%** of
samples (4.7% / 3).

**Did it match, and why?** No prediction. Channels 27/28 mark only black pegs whose component touches a goal edge,
which in these games happens late and only in the game black wins; the defect's exposure is concentrated in a
few late, decided positions rather than spread across play.

**Limits.** Four games from one diagnostic corpus (41–43 plies, three red wins), not a training-recipe corpus;
exposure in longer or black-favourable games may differ. Counts exposure, not effect on the network or on play.
The incumbent's actual launch settings (mirror probability) remain inferred from code defaults, not recorded.

**Conclusion / status.** MEASURED; the decision on a controlled trial rests with review. Nothing trained, played or
authorized; `calib020_0001` unchanged.

*Decision (2026-10-19, review).* **Proposed pilot not justified by current evidence — NO_GO.** The paired training-and-match pilot (one fresh corpus, control vs corrected mirror, candidate vs control) is not authorized on this exposure count; it was never run, so this is **not** a failed training experiment. Qualifications: 1.6% is an expected buffer share under these four games and the assumed mirror probability 0.5, not a measurement of the incumbent's training buffer; and the exposed positions are late, but it is **not established that they were already decided** — rare, important positions can affect strength disproportionately, so the count does not show the defect was harmless. The correctness fix `0c417882` stays and is carried into any independently justified future training experiment. This closes the route from this exposure count to that pilot, not the goal of stronger play.

*Correction (2026-10-19, review).* "Did it match" above calls the exposed positions "late, decided positions": they are late; that they were decided is **not established**.

### 15. Root-only sequential-halving variant — prototype and focused CPU checks (2026-10-19, OPEN)

**Test.** Build one fixed root-allocation variant and check it on the CPU before any gameplay. A subclass of `MCTS`
(`scripts/GPU/alphazero/root_sequential_halving.py`) overrides only `search_with_root`; descent below the root is
the existing PUCT through `_run_single_simulation(root, root_move_override=…)`. Called a **root-only
sequential-halving variant**, not full Gumbel AlphaZero (Danihelka et al., ICLR 2022).

**Frozen variant (no parameters, no grid).** Plies 0–19: ordinary search and the existing readout. From ply 20
(24×24 only, no root noise): the 16 highest-prior legal moves (fewer if fewer are legal), ties by (row, col); zero
Gumbel noise (as in the paper's perfect-information evaluation); score = log(max(prior, 1e-12)) + (`c_visit` 50 +
max visits) × `c_scale` 1.0 × (q + 1)/2, q the candidate's mean value from the root player's side — the fixed
(q + 1)/2 mapping replaces the paper's completed-Q rescaling (rescaling only the survivors could stretch a tiny
value gap to the full range); it is our adaptation and carries no published guarantee. Rank ties: higher prior,
then (row, col). Budget: ceil(log₂ m) visiting rounds; every round but the last gives each survivor
N ÷ (rounds × survivors) more visits (floor) and keeps the top ceil(survivors/2); the last round splits the rest
evenly, any remainder one visit each in rank order (prior order when nothing is visited yet); visits round-robin.
For 16 candidates: 6/12/25/54 more visits per round (cumulative 6/18/43/97); one candidate gets all 400. The move
played is the top-ranked final survivor and replaces the readout. Known limit, not patched: a move outside the 16
candidates is never searched.

**Scope exception to #47 (2026-10-19, review).** #47 closes progressive widening and ends "tree-local search
heuristics are closed". A **narrow exception** is recorded for this one fixed, root-only allocation experiment:
search below the root is unchanged, so #47's measured failure (depth-1 retention) does not apply; whether 16 root
candidates retain the moves deeper search needs is unmeasured. Progressive widening, FPU variants and the archived
evidence stay closed. The flat-root mechanism (#57) motivates it; the already-lost game-3 example (#58)
illustrates allocation behaviour and is **not** evidence this would save games.

**Hypothesis.** Concentrating the root budget on a few prior-ranked candidates, chosen by value and prior, plays
better than ordinary PUCT allocation at the same 400 simulations.

**Why it was plausible.** A published fixed-budget method for choosing among root moves, untested here (no prior
ledger or code record), aimed at the observed root failure: in losing positions FPU 0 outranks every visited move
and the root spreads flat (#57). 400 simulations over about 500 legal moves is a limited-budget setting, not
evidence against the approach; #39's overestimate finding does not establish that concentration generally weakens
play.

**Prediction on record.** The author's, stated with the proposal before implementation: **fails the screen.**

**What the comparison will measure.** Root allocation **plus its final decision rule** against ordinary search
with its readout — not allocation alone (identical opening procedures do not remove the later readout
difference). Intended screen, **not yet approved**: 32 fresh games (16 colour-reversed pairs, pairs the
statistical unit), 400 simulations per side, advance bar **24 points**, 90-minute cap; runtime and network
evaluations recorded, no timing veto.

**What ran.** CPU only, stub evaluator, no model: `tests/test_root_sequential_halving.py`, **13 passed** (2.6 s):
accounting for 1, 3, 7, 15 and 16 candidates (exact per-move counts, root total 400, remainder rule 120/119 for
7), full-board accounting (survivor sets 16 → 8 → 4 → 2, first 16 = top 16 by prior, played move legal and a
final survivor), value overriding a lower prior, the exact score formula and zero-prior floor, a terminal win
backed up as +1 and chosen, the pinned limit (a winning move outside the candidates is never searched), no PUCT at
the root and PUCT below it, plies 0 and 19 identical to `MCTS.search_with_root`, and the refusals (root noise, a
12×12 board, a 40-simulation budget). Four deliberate mutations (value sign, remainder order, floor instead of
ceil survivors, 17 candidates) each failed at least one test; the module was restored byte-identical.

**Result / did it match.** The implementation matches the frozen variant on these checks; no prediction concerns
them. No strength evidence.

**Conclusion / status.** OPEN — built and CPU-checked; the match script, its loop-equivalence check, model
loading, seed accounting and the screen await separate review. `calib020_0001` and production unchanged.

*Update (2026-10-19) — match script and loop-equivalence check.* The match script (`docs/superpowers/evidence/root-halving-match/01_match.py`) copies `eval_runner.play_eval_game` / `_play_and_build_result` with three marked differences only (each colour's MCTS class from its agent; the move is the search's `last_choice` when it supplies one, else the existing readout; an optional per-search timer); separate search/readout RNG streams, per-ply budget checks, colour binding and replay records are unchanged. Wrapper `02_wrapper.sh` (5,390 s + 10 s kill grace). **Loop-equivalence check** (`03_loop_check.py`, evidence `root-halving-match/loop-check/`): ordinary vs ordinary through both loops, one colour-reversed pair on test-only seeds, GPU-free fake evaluators (uniform priors, value 0), 16 simulations, 24×24, 280-ply cap; run once in a clean detached worktree at `e8df0da6`, exit 0, **PASS** in 10.6 s — both games identical in every result field but the replay path and in their replay files. Limits: equivalence was shown **with fake evaluators only**; both games ended at the 280-ply cap, so the win and board-full branches were not exercised; and the **halving choice branch (`last_choice`) was not exercised by this check**. The screen (32 games, 400 simulations per side, bar 24 points, 5,400 s) is now approved to run once after fresh seed preparation.

*Screen result (2026-10-19) — STOP_CONFIGURATION; CLOSED.* **What ran.** Seeds: survey `83fcf73f` (CLEAN, 45/45),
pre-proof `1dc80bc9`, registration of the 32-seed block in ACCOUNTED `66e80515` (only ACCOUNTED changed, 38 → 39;
registry tests 17 + 31 passed, exit 0), post-proof `af5c43a0` (state HOLDS, 45/45). One run through `02_wrapper.sh`
in a clean linked worktree on `codex/root-halving-run`: **outer exit 0, 2,130 s, COMPLETED, no survivors**; both
networks in evaluation mode at load and after the last game; all 27 loaded repository modules committed. Evidence
`d535cd75` (`docs/superpowers/evidence/root-halving-match/`). Closing: K1 `6817f7e3` (RECONCILES, 15/15; 32/32
started), K2 `9b6a8267` (the block EXPOSED and RETIRED whole, 36 → 37 and 34 → 35; ACCOUNTED unchanged; registry
tests 48 passed, exit 0).

**Result.** Halving **21–1–10** ordinary (one 280-ply cap draw); **21.5/32 points against the frozen bar of 24 →
STOP_CONFIGURATION.** Pair scores: six 1.0, one 0.75, eight 0.5, one 0.0; mean pair score **0.672**, unclipped
Hoeffding interval **[0.332, 1.011]** (descriptive, pairs the unit, conditional on assumed pair independence) —
includes 0.5. Halving won 9 games as red and 12 as black; ordinary 4 and 6. Games 30–88 plies apart from the cap.
Cost: network evaluations **376,168** (halving) vs **377,822** (ordinary); search seconds **1,065.5 vs 1,059.3**;
639 of halving's 959 searches used the halving allocation (ply ≥ 20), the rest the ordinary opening.

**Did it match the prediction?** **Yes — the author's prediction "fails the screen" matched**: the result is below
the advance bar. The 21–10 game count leans toward halving, but the interval includes 0.5 and this 32-game screen
cannot distinguish that lean from chance; it is neither a strength finding nor evidence that halving is weaker.
The comparison measured root allocation plus its final decision rule against ordinary search with its readout, at
equal simulations and essentially equal network cost.

**Conclusion / status.** CLOSED for this configuration (STOP_CONFIGURATION). No separately priced strength
comparison is earned. `calib020_0001`, ordinary search and production unchanged; the prototype stays unwired. See
do-not-repeat **#62**.


### Methods note — funding and powering future strength experiments (2026-10-20)

Recorded after entry 15, which applied a 24/32 knockout bar (a 75% observed score before any strength test). That
is a valid spending rule but poorly matched to the goal: it can discard worthwhile improvements (the older readout
plan warned of this). **This note reopens nothing previously closed** — halving (#62), mirror averaging (#59) and
every other closed line stay closed.

For the next genuinely new candidate:

* **The target effect is chosen before implementation.** The planning alternative is a **true mean pair score of
  60% against the incumbent**. It is the alternative the power budget is computed for — **not** a required observed
  score and **not** a promised effect.
* **The small screen catches broken behaviour and gross regression only**, against a threshold frozen before it
  runs; it does not demand an overwhelming win.
* **The full strength comparison is funded before launch.** Its statistic, independent sampling unit (colour-
  reversed pairs), regression threshold and **all** stopping rules are frozen in advance.
* **Reference budget** (conservative Hoeffding on independent pair scores in [0, 1], two-sided α 0.05, 80% power:
  n pairs with (√(ln 40 / 2) + √(ln 5 / 2)) / √n = δ): **60% → 509 pairs, 1,018 games** (about 19 hours at entry 15's
  66.6 s per game); **55% → 2,035 pairs, 4,070 games** (about 75 hours). Planning times are estimates, not
  guaranteed completion times.
* **Any sequential rule must control error across the whole procedure.** Adding an early-success test to a separate
  95% final test does not by itself preserve 95% validity; the combined procedure's error must be controlled (for
  example by an anytime-valid or properly spent design), and the power budget recalculated where needed.
* A normal-approximation figure of **about 130 games** for 60%, from entry 15's 16 pair scores (standard deviation
  about 0.29), is **illustrative only and is not an adopted budget**.


### 16. Adaptive per-move budget — oracle bookkeeping of chronological credit (2026-10-20, MEASURED)

**Test.** Rules-only accounting, no model calls, searches, games, training or seeds. The ordinary agent's recorded
400-simulation searches from ply 20 in the root-halving (entry 15) and mirror-averaging (entry 11) match replays —
64 games, 1,135 searches, the same records surveyed in chat on 2026-10-20. One illustrative policy, fixed by review
and not tuned: each game starts with 0 credits; +400 credits on each of the agent's turns from ply 20; a recorded
|root value| ≥ 0.8 spends 200; any other ("contested") position spends min(1,000, available credits); never borrow
from future turns. **This is oracle bookkeeping**: it assumes the recorded classification (from a completed
400-simulation search) is available cheaply, which is unestablished. It measures chronological opportunity, not
strength, and is not fresh confirmation (the records were already surveyed).

**Hypothesis.** Budget saved on strongly evaluated positions arrives in time to be spent on contested ones.

**Why it was plausible.** The pooled survey found 56% of these searches at |value| ≥ 0.8 with very flat roots; but
pooled percentages cannot show that savings arrive before they can be spent.

**Prediction on record.** None stated in advance; no decision rule frozen.

**What ran.** A script held outside the repository (scratchpad; not committed — only this ledger text is), run
under `timeout -k 5 55`. **Attempt 1: execution error, not a result** — exit 1 before any accounting output: it
read a context field `winner_agent_id` that the replay files do not carry (they record `winner` as a colour); no
accounting ran and no numbers were seen. **Attempt 2** (authorized), after a one-line schema fix (`winner` ==
the ordinary agent's colour) with the policy unchanged: exit 0, 0.014 s.

**Result.** 634 strongly evaluated and 501 contested positions. Strong positions freed **126,800** credits; only
**18,400 (14.5%)** reached contested positions before the game ended; **108,400 (85.5%) were unused at game end**.
Contested positions averaged **436.7** simulations (+36.7, about +9%); **19 of 501** reached 1,000. 28 of 64 games
moved any extra budget to a contested position; 12 had no contested position from ply 20; median unused credit at
game end 1,600; the top four games took 39% of all extra. Per-game figures are in the 2026-10-20 chat report.

**Did it match, and why?** No prediction. Strongly evaluated positions cluster at the end of games: once a game's
evaluation becomes strong it rarely returns to contested, so savings arrive after the contested moves they were
meant to fund.

*Correction (2026-10-20).* The explanation above (strong evaluations cluster at the end of games and rarely return to contested) is an **unmeasured interpretation**: the pass counted credits, not transitions between classes. The measured facts are the credit totals.

*Decision (2026-10-20, review).* **Proposal declined — the banked-budget controller is not built or funded on current evidence.** This is a declined proposal, **not** a failed gameplay experiment and **not** a rejection of adaptive budgeting generally. The proposed mechanism — roughly doubling the budget on contested moves — did not materialize: contested positions received **436.7** simulations on average, about **9%** extra; classification cost and reliability remain unresolved. The arithmetic's main effect is a **hypothetical** reduction of post-opening simulated work from **454,000 to 345,600** (about 24%) on fixed trajectories — not measured runtime savings, and not evidence that spending less preserves play. No further accounting variants or model runs follow. Other adaptive-budget mechanisms remain untested and need an independent justification, not a rescue of this policy. See do-not-repeat **#63**.

*Reporting correction (2026-10-20).* In the 2026-10-20 chat report's per-game figures, root-halving game 20 (the 280-ply cap) was labelled L for the ordinary agent; it was a capped **draw** (D). The accounting is unaffected.

**Limits.** Oracle classification (unestablished cost and reliability); |value| ≥ 0.8 means strongly evaluated, not
decided; one illustrative policy on 64 games from two matches; 400-simulation records only; says nothing about
strength. These records are now consumed for policy design: do not tune a budget policy against them.

**Conclusion / status.** MEASURED. Under this policy the central compute claim — that freed budget can be
reallocated to contested moves — largely fails chronologically (14.5% reached them). The decision on adaptive
budgeting rests with review; nothing is built or authorized. `calib020_0001` unchanged.

---

*Append a new row to the [experiment ledger](#experiment-ledger) and update [do-not-repeat](#do-not-repeat-prevents-going-in-circles) whenever a branch is run and judged. Keep the key conclusion at the top current.*

---

### 17. Implementation finding — the ordinary training chain truncated policy targets at 512 move slots (2026-10-20)

**Finding (found in review).** `trainer.make_padded_batch` keeps at most `max_moves_cap` legal moves per position:
it slices `legal_moves` and `visit_counts` to the first M slots and renormalizes the retained counts. The cap
defaulted to **512** there and in every function of the ordinary training chain that passes it through
(`_calibration_component_loss`, `summarize_value_sanity`, `alphazero_loss_batch`, `train_step`), and the
ordinary training loop's two `train_step` calls pass no cap. A 24×24 opening has **528** legal moves, so the last
16 moves and their visits were dropped and the target renormalized over the rest — e.g. 300 visits on the first
move and 100 on the last trained as 100% / 0% instead of 75% / 25%. Per the review, the historical continuation
source used the same default; the external-learning adapter already passed 576 explicitly
(`external_learning.MAX_MOVES_CAP`), so its protection never reached ordinary training.

**Fix.** The five defaults are now **576** (24 × 24, so no 24×24 position can be truncated); explicit
caller-supplied caps are unchanged. `tests/test_external_learning.py::test_a_chosen_move_beyond_slot_512_keeps_its_target`
relied on the old default to demonstrate the hazard; it now passes `max_moves_cap=512` explicitly (one line,
same assertion). That test's fixture plays fake-evaluator games, so it was **not run** under this authorization.

**Checks.** New focused CPU test `tests/test_padded_batch_cap.py` (MLX on CPU; no network, model, search, game
or training): the real 24×24 opening (`TwixtState` legal moves, 528) with 300 / 100 visits on slots 0 / 527 must
keep all 528 moves in legal order, every zero target and exactly 0.75 / 0.25; a two-position batch under 512
moves must match hand-computed rows, columns, mask and targets; an explicit cap of 512 must still truncate; and
all five defaults must be 576. Run once before the fix — 6 failed (the opening test and the five defaults), 2
passed (small batch, explicit cap), 0.24 s — and once after — 8 passed, 0.12 s.

**Consequence.** Not an explanation of any regression: the parent inherited the same truncation. The incumbent's
weights are unchanged; how many trained positions exceeded 512 legal moves, and what the truncation did to the
learned policy, are **unmeasured**. This is a correctness repair to carry into any future training proposal; no
training, retraining or match follows from it.

*Correction (2026-10-20, review).* The opening of **Consequence** above ("Not an explanation of any regression")
overstates what the evidence shows. That the parent inherited the same truncation does not rule out additional
damage from it during continuation; it means the existing evidence cannot isolate its contribution, which is
neither ruled in nor out. Continuation's disabled calibration objective and restarted value-weight warmup were
already recorded differences between parent and continuation, not newly discovered causes. The repair is a
training-data correctness fix with no demonstrated strength improvement; it is carried into any independently
justified future training, and the closed continuation experiment is not restarted to test it.

### 18. Policy-head capacity pilot: isolated residual head and CPU checks (2026-10-21, OPEN)

**Test.** Build, in isolation, the two training arms of a proposed policy-head capacity pilot and check them on the
CPU before any checkpoint, corpus, search or training. `scripts/GPU/alphazero/policy_head_capacity.py`:
`ResidualPolicyHead` subclasses `PolicyHead` and keeps the existing path unchanged, adding a second path that reads
the move cell's full 128 encoder features, logit = existing + `extra_out(relu(extra_fc(features at the move)))`,
`extra_fc` 128 → 64, `extra_out` 64 → 1, with `extra_out` zero-initialized so arm R starts exactly equal to the head
it copies. `install(network, "R" | "C")` installs the arm, puts the network in eval mode, freezes everything but the
arm's surface and refuses any other surface: **arm C trains only the existing head's 519 parameters; arm R trains
those 519 plus 8,321 new = 8,840.** Encoder, value head and every BatchNorm running statistic stay frozen. The head
refuses to be called by search. Not production and not wired into any caller.

**Hypothesis.** The user's, untested and not a found bottleneck: the policy head scores each move from only two
numbers per cell (1×1 conv 128 → 2, BN, ReLU, then 2 → 64 → 1; 519 parameters, `network.py:293`), which may limit
what the policy can learn. These CPU checks test no part of that hypothesis; they test only that the arms are
built as designed.

**Why it was plausible.** Every move logit passes through a two-channel bottleneck per cell, and the head has 519
parameters against an encoder of 128 channels per cell; a richer read of the same features is a direct way to ask
whether that bottleneck binds.

**Design (agreed in review before implementation).** Baselines: arm C, a fitted temperature T and the unchanged
incumbent I; R must beat each separately. Planted controls only qualify the fitting recipe (operational) and do not
isolate capacity; they run before any label generation, with a synthetic target scale frozen independently of real
targets. Development features may come from existing diagnostic positions, never spent held-out material. Repeat
labels measure reproducibility only, and Stage B's gate Q is not imported. The 59/100 sign-test power (82.5% at
p 0.63) is per comparison, not for the combined verdict. The training dose is frozen explicitly (1,600 positions ×
160 steps × batch 32 ≈ 3.2 passes, versus Stage B's 8; a short recipe failing proves nothing about capacity).
User decisions: δ = 0.05 nats accepted as a spending threshold; the 200-game, ~16 h pilot is **not** approved; no
Stage B variance computation; a later strength match would not need Stage A rerun.

**Prediction on record.** None stated in advance.

**What ran.** CPU only, MLX on CPU, random weights, no checkpoint, search, game or training:
`tests/test_policy_head_capacity.py`, six tests on real 24×24 positions from the rules engine (red's opening, 528
legal moves, and black to move after one red peg, padded to 528).
- *First run* (old machine, Python 3.14.7, by 2026-10-20; files untracked): **4 passed, 2 failed, 2.68 s.** Both
  failures were a **test bug at line 69**: the copied BN `running_mean` (float32) was compared against float64
  literals `[0.3, -0.2]` (actual 0.30000001 / −0.20000000; the copy was correct). Both arms' logits and value were
  bitwise equal to the incumbent head before that line.
- *Fix (approved 2026-10-21):* line 69 compares against `np.float32([0.3, -0.2])`; nothing else changed.
- *Corrected run* (2026-10-21, Mac Studio M5 Max, Python 3.14.8, run once, 60 s cap): **6 passed, 1.63 s.**
Committed with this entry (module sha256 `ab9418916d7f…`, test `f373a32d4aad…`).

**Result.** Both arms start bitwise equal to the incumbent head (logits and value), with the BN running statistics
copied. With random `extra_out` weights the residual path changes every legal logit while keeping full legal support
(finite logits on every legal move, −1e9 on padding, a proper distribution). After two Adam steps exactly the arm's
surface changed: no other parameter, no BN running statistic, and eval mode was kept. The refusals (unknown arm,
installing twice, calling the head from search) hold, and installing on a clone leaves the source untouched.

**Did it match the prediction.** No prediction was stated. These are construction checks, not a capacity result.

**Conclusion / status.** OPEN — built and CPU-checked; nothing about capacity is measured. The next step,
planted-target qualification of the fitting recipe, needs its own approval. No checkpoint loading, corpus
generation, search or real-target training is authorized.

**Addendum (2026-10-21): planted-target qualification of the fitting recipe — COMPLETED, NOT QUALIFIED (CLOSED).**

*Test.* Card `docs/superpowers/2026-10-21-policy-head-capacity-planted-qualification-card.md`: rev 2 (`28abdbfe`),
built and wrapped in `be18fd07`, amended to rev 3 in review before any run (`9deaccb9`: arm R's residual path seeded
(seed 15) instead of MLX's unseeded initializer; SIGTERM recorded as a VOID, and a record written by the wrapper
when the script cannot write one). Can the frozen recipe (400 Adam steps, batch 32, lr 1e-3, global-norm clip 1.0,
8 passes over 1,600 positions) recover a capacity gap planted on purpose on `calib020_0001`'s real encoder features
of real positions, without favouring R where no gap exists? TC is a perturbed copy of the existing head (inside both
arms' class); TR adds a fixed random 128 → 64 → 1 network of the move cell's features (inside R's class only); each
planted term has a 1.0-nat within-position scale. Five rules, all required: F1, F2 (each arm fits its exactly
representable target to ≤ 25% of the starting KL), S1 (R beats C on TR by ≥ δ = 0.05 nats), S2 (R beats a fitted
temperature on TR by ≥ δ), S3 (R does not beat C on TC by δ or more).

*Hypothesis.* Operational only: the recipe can expose a planted gap of this size.

*Prediction on record.* None stated in advance.

*What ran.* One run, approved 2026-10-21: gate opened by `8f0743e5`, `01_wrapper.sh` once, CPU only, 900 s
ceiling; inner exit 0, **26 s wall**, record written by the script, gate restored and read back, no survivors.
Evidence and the restored gate: `08ced79e` (`docs/superpowers/evidence/policy-head-capacity-planted/` record and
positions; `…-run/` outer.json and logs). Checkpoint SHA-1 verified, 0 modules in training mode; 64 games / 3,293
moves as frozen; pools 2,576 training / 584 evaluation (10 evaluation digests leaked into training and stayed
there); 1,600 + 400 positions. Feature cache (602 MB) outside the repository at `~/twixt-cache/policy-head-capacity-
planted/features-20261022T034928Z.npz`, re-hash matching the recorded digest; kept. TC α 0.6728 (scale 1.0000); TR
s 3.208. Both R fits started at the seed-15 residual (digest `ee796a7e…`).

*Result.* KL on the 400 evaluation positions (nats):

| target | start (incumbent I) | C | R | temperature (τ 1.187) |
|---|---|---|---|---|
| TC | 0.1715 | 0.00047 | 0.00057 | — |
| TR | 0.4712 | **0.0496** | **0.0210** | 0.4199 |

F1 holds (both ≤ 0.0429), F2 holds (0.0210 ≤ 0.1178), S2 holds (margin 0.399), S3 holds (C − R on TC = −0.0001).
**S1 fails: KL(C, TR) − KL(R, TR) = 0.0286 < 0.05.** Verdict **NOT_QUALIFIED** (status COMPLETED, not VOID).
Descriptive: chance baseline 6.217 nats; target entropy TC 2.584, TR 3.036; top-2 principal components hold 31.9% of
legal-cell feature variance; clipping active on 0.3–1.3% of steps. Phases (s): features 8.0, targets 7.4, each fit
2.0 (C) or 2.8 (R), temperature 0.2. These are this machine's first timings for this workload.

*Did it match the prediction.* No prediction was stated.

*Interpretation, kept to what was measured.* Arm C, with only the existing head's 519 parameters, brought KL on the
R-only target from 0.471 to 0.050 (to 10.5% of its start); R reached 0.021 (4.5%). At this planted size and dose the
recipe separated the arms by 0.029 nats, below the frozen δ, so by the card this recipe must not be used to measure
capacity. Why C fitted TR this closely was not measured and is not asserted. **This says nothing about whether the
incumbent head's capacity limits real learning, and nothing about strength.**

*Conclusion / status.* Planted qualification **CLOSED, NOT QUALIFIED.** Per card §6: no real labels are generated
and the recipe is not retuned; any new recipe needs a new card, fresh teacher seeds and its own approval. The
capacity hypothesis remains **untested**. The 200-game pilot, label generation and promotion remain unapproved.
See do-not-repeat **#65**.

**Addendum (2026-10-22): the policy-head capacity proposal is SHELVED (decision; documentation only).**

*Decision.* No further spending on this proposal. Basis: no qualified, affordable test has been established. The
planted qualification above was NOT QUALIFIED (S1: KL(C, TR) − KL(R, TR) = 0.02857 < δ 0.05). A follow-up read-only
review of the existing code and evidence (no fitting, inference, labels or games) found that the planted TR target,
although exactly representable by R, was not constructed so as to establish beforehand that C could not approximate it
within δ: the qualification measured the gap but did not establish that a gap of that size had to exist. No suitable
guarantee of a KL separation between these actual arms, specified before fitting, was identified in that review.

*Scope of what remains open.* A frozen-encoder pilot addresses a useful but narrower question: whether the existing
encoder's features carry move information that the current head does not extract. It does not address capacity or
optimization under joint training. That an encoder could in principle carry the final logit through one channel does
not show that this fixed, finite encoder could absorb a richer head without cost; joint-training capacity and
optimization remain open.

*Status.* The capacity hypothesis is **unresolved** (untested). `calib020_0001` remains the incumbent. Nothing new was
run for this addendum. The feature cache (`~/twixt-cache/policy-head-capacity-planted/`) is kept for now. No new
recipe, synthetic experiment, real labels, 200-game pilot or promotion is authorized. See do-not-repeat **#66**.

### 19. Play-path timing re-measured on the new machine (2026-10-21, MEASURED)

**Test.** Re-measure on this machine the timings that forward-looking documents rely on, without playing a game.
Hardware (recorded by the script from sysctl, system_profiler and sw_vers): **Mac Studio (Mac17,14), Apple M5 Max,
18 CPU cores (6 "Super" + 12 "Performance"), 40 GPU cores, 48 GB, macOS 27.0.1 (build 26A434)**; `.venv` Python
3.14.8, MLX 0.30.3. Script `docs/superpowers/evidence/play-path-timing/01_timing.py` (committed `55d3b9c1`):
`calib020_0001` (SHA-1 verified) through `eval_runner._default_evaluator_factory` (eval mode since `c8cb2aa6`; 0
modules in training mode, checked; compiled; Metal GPU); `MCTS.search_with_root(state, add_noise=False)` at
`cfg_from(EvalConfig(**PROTOCOL), 400)`, one MCTS per colour per game as `play_eval_game` builds them; no setting
changed; no seed (an unseeded `random.Random()`; the moves come from the record, so the stream only orders ties
inside a search). Positions: every position of one committed colour-reversed pair of inference-mode-match games,
the pair whose total length is closest to the median (pair 0, games 0 and 1: 43 + 61 plies); each recorded move
replayed and checked (ply, player, legal-move count, legality).

**Hypothesis.** None: a measurement, not an experiment.

**Why it was needed.** Every recorded timing came from the old machine (Apple M3 Pro). This machine has more GPU
cores, but the play path runs one position at a time and waits on GPU syncs, so no speed-up could be assumed.

**Prediction on record.** None stated in advance.

**What ran.** Script committed at `55d3b9c1`; run once on 2026-10-21 from a clean linked worktree, 10-minute cap, no
other pytest or heavy compute running; exit 0, 75 s wall. Record (committed with this entry):
`docs/superpowers/evidence/play-path-timing/run/record.json`.

**Result (measured).** 104 searches of 400 simulations. Excluding the first (758 ms, including warm-up): **mean 1.74
ms per simulation** (median 1.76, p90 1.85; the minimum, 0.10 ms, is one near-terminal search); **0.697 s per
ply**; search time per recorded game **30.2 s (43 plies) and 42.4 s (61 plies)**; process CPU about 60% of wall;
load average 1.2–1.7. Warm-up inference 1.53 s; load 0.02 s.

**Derived (not measured; assumptions stated).**
- At entry 15's mean of 59.8 plies per game (59.78125, counted from its committed records): **41.7 s per game,
  search time only** (game-loop overhead not included). The methods note's reference budgets become **about 11.8
  hours for 1,018 games (60%)** and **about 47 hours for 4,070 games (55%)** on this machine. The old machine's 19
  and 75 hours came from entry 15's 66.6 s, which was a **whole game** (one side the halving variant, loop
  included), while 41.7 s is search time only. **No speed-up between the machines was measured, and none is
  claimed.**
- The pilot's label lines: **training labels, 1,600 × 1,600 simulations ≈ 1.24 h** (old 2.9 h) and **held-out
  labels, 400 × 3,200 ≈ 0.62 h** (old 1.3 h). These **assume the cost per simulation measured at 400 simulations
  also holds at 1,600 and 3,200**; that was not measured.
- The generation line (200 × 205 s = 11.4 h on the old machine) is **not re-derived**. Stage B's G phase
  (`eval_runner.run_game_tasks`, a400 vs b400, mean 57 plies) implied about 3.6 s per ply against about 1.1 s per
  ply in the match games on the same path; that discrepancy is unexplained, so the figure stays **open** on this
  machine.

**Did it match the prediction.** No prediction was stated.

**Comparability.** The old profile's **3.16 ms per simulation** (`b8cf2adf`) is **not comparable** and is not a
baseline for 1.74 ms (CLAUDE.md): it came from three hand-built positions, with BatchNorm in training mode (before
`c8cb2aa6`), on the old machine.

**Conclusion / status.** MEASURED. Old-machine timings in this ledger stay as written (true for that machine);
planning on this machine uses the figures above, with the derived lines carrying their assumptions. Still open
here: the 205 s per G-phase game, and with it the pilot's total cost.

### 20. Self-play and labelling throughput settings for the new machine (2026-10-21, MEASURED)

**Test.** Find the fastest worker count and eval batch for (a) probe labelling (`build_probe_suite` phase 2) and
(b) self-play generation (`train.py`) on the Mac Studio M5 Max of entry 19 (18 CPU cores, 40 GPU cores, 48 GB).
The old values were tuned on the M3 Pro: `--label-workers 10`, training `--n-workers 10`, and eval batch capped at
14 since 16 caused Metal hangs there. Operational tuning, not a strength experiment: games are unseeded and no
outcome was read.

**Hypothesis.** None for worker count. For eval batch, see the prediction.

**Why it was plausible.** More cores and more GPU should move the worker optimum up. Whether the eval-batch-14 cap
was still needed on new hardware was open.

**Prediction on record.** Stated by the user before any eval-batch run: "we expect eval batch to cause it to crash
however this is a new CPU/GPU and more [memory] so I'd expect [us] to exceed the current safe limit of 12 or 14".
Nothing was predicted for worker count.

**What ran.** All runs from main with no other heavy job at start; evidence (reports, CSVs, records, and the
diagnostic's timestamped progress) in `docs/superpowers/evidence/selfplay-throughput/`.
- *Labelling* (`scripts/probes/benchmark_phase2_knobs.py`, 10 candidates, 500 sims, `model_iter_0059`): run A,
  workers 8–18, eval batch 12/14, stall-flush 8/16/32, 2 passes; run B, 12 workers, eval batch 14 and (unsafe)
  16/24/32/48.
- *Self-play* (`scripts/bench_knee.py`, rewritten for this, `calib020_0001`, 24x24, 400 sims, stall-flush 48,
  virtual visits 8, outputs outside the repository): first sweep at `6d888aa8` (36 games/run, 2 repeats, scored on
  whole-run wall time); diagnostic at `6864a16e` (12 workers, eval batch 14, 6 repeats, 30-minute watchdog,
  unbuffered timestamped output, CPU/GPU sampling); steady-state sweep at `5ddd7437` (48 games/run, 3 repeats,
  scored on games/h while every worker is busy, each run stopped at the first `Worker done`); one full run at 18
  workers, eval batch 12.

**Result.**
- *Labelling.* Serial 336.8 s; 8 workers 107.6 s (3.13x); 10–18 workers 102.9–103.8 s (3.24–3.27x), flat from 10
  up. Eval batch 12–48 and stall-flush 8–32 all within 0.6 s, with no hang. **Those two settings are inert for
  labelling:** the labeller calls `MCTS.search()` (`probe_eval.py:1038` → `search_with_root`), one leaf per network
  call; only `search_from_root()` (self-play) reads `eval_batch_size` (`mcts.py:632`) and `stall_flush_sims`
  (`mcts.py:717`). Run B therefore tested nothing about the cap. `docs/probe-suite-generation.md` presented them as
  labelling knobs and is corrected in the same commit as this entry.
- *First self-play sweep — invalid.* 6 of 12 runs were killed at a 409 s watchdog, eval batch 14 included, so its
  "first hang at eval batch 16" pick and the skipped 24/32/48 mean nothing.
- *Diagnostic.* All 6 runs completed; no stall. Wall time 140–484 s for identical settings, but the game rate
  while all workers were busy (games 5→25) was 0.280–0.306 games/s. The three slow runs each held exactly one
  state-cap draw (about 270 plies) that ran alone at the end: one worker at 100% CPU, GPU about 20%, for up to 378
  s. Outside CPU load stayed under 12% in the slow runs, so contention is ruled out. The first sweep's kills fit
  this pattern (such runs take 400–480 s), but their output was lost, so that is inferred, not shown.
- *Steady-state sweep* (median games/h, range over 3): at eval batch 14, workers 10 → 863 (776–998), 12 → 1,038
  (1,003–1,104), 14 → 1,068 (874–1,235), 16 → 1,136 (1,042–1,214), **18 → 1,207 (1,142–1,251)**. At 18 workers,
  eval batch **14 → 1,207**, 16 → 1,084 (989–1,088), 24 → 1,137 (1,056–1,195), 32 → 880 (858–1,199), 48 → 910
  (906–1,005). All 30 runs good; no timeout or stall at any eval batch. Spread within a setting reached about 30%
  (25–30 games per window, game-length mix, and another session's test suite adding up to 336% CPU in some runs).
- *Eval batch 12, 18 workers, one full run:* 1,219 games/h steady; average batch 10.5 vs 11.6 at 14 (the cap binds
  slightly more), speed unchanged.

**Did it match the prediction, and why.** Partly. No eval batch up to 48 hung, in either the labeller (where it is
never read, so that run says nothing) or self-play (3 short runs per value, about 2.5 minutes each). But going
above 14 gave **no speed-up**: 32 and 48 were slower. Batches already average about 11.6 of 14, so throughput is
limited by how fast the workers produce leaves, not by the cap. Three short clean runs per value do not show the
old hang is gone.

**Conclusion / status.** MEASURED; adopted by the user. Self-play: **18 workers, eval batch 14**, recorded in
`train.py --help`, `docs/train-cli.md` and `docs/analysis-metrics-guide.md` (`151f09c4`). Defaults are unchanged
(`--n-workers` stays 1, so pass 18 explicitly; eval batch stays 14). Labelling: `--label-workers 10` stays (within
1% of the best). Separate finding, left as is: in training, each iteration waits on its slowest game, so a
state-cap game costs up to about 5 minutes of near-idle machine per iteration; shortening such games would change
play, so it is not a throughput setting. See do-not-repeat **#64**.

### 21. Search distillation D2 — Stage 0 offline trainer and the Stage 1 live measurement (2026-10-22, MEASURED)

**Test.** Stage 0: an offline trainer (`scripts/GPU/alphazero/search_distillation_d2.py`) that rebuilds training
records from a finalized replay-capture phase and trains a student copy of `calib020_0001` on the encoder and
policy head only, in evaluation mode (BatchNorm running statistics frozen), with a value anchor to the unchanged
teacher; synthetic checks only. Stage 1 (card `docs/superpowers/2026-10-22-search-distillation-d2-stage1-card.md`,
pinned): exactly two incumbent self-play games at 1,600 simulations through the serial capture driver, at the
settings intended for corpus generation (`train()`'s other defaults, root noise on, mirror augmentation off, both
goal-completion diagnostics off), to price generation and to check the capture path with the live network.
Measurement only; no strength claim. The later stages are drafted in
`docs/superpowers/2026-10-22-search-distillation-d2-plan.md` and are not approved.

**Hypothesis.** None for the measurement. (The experiment's hypothesis, for Stages 2–5: distilling the incumbent's
1,600-simulation search into its whole policy path makes it stronger at 400 simulations.)

**Why it was plausible.** Stage A (entry 1) showed 1,600 simulations play much stronger than 400 (mean pair 0.769),
under the earlier protocol with training-mode BatchNorm; transfer to corrected inference is an assumption.

**Prediction on record.** No prediction for the measurement. The proposal's cost estimate, stated before the run:
about 2.8 s per ply (1.74 ms per simulation × 1,600), game length unmeasured.

**What ran.** Build `24327418` (13 trainer tests; 8 mutation controls caught after one was replaced because the
fake-evaluator games have no inherited child visits, so before- and after-search counts coincide there; that case
moved to a live check in Stage 1). Regression at `24327418`: the 58-id baseline exactly (3,272 passed). Seeds:
survey `c1f674e1` (CLEAN, 45/45), pre-proof `5c82af03`, registration in ACCOUNTED `51c51898` (39 → 40), post-proof
`4554d472`; registry readers 593 passed. Gate `SEARCH_DISTILLATION_D2_MEASURE_AUTHORIZED` (the thirty-second)
opened by `6cb69b40`; one run through `01_wrapper.sh` in a clean linked worktree; evidence and the restored gate
`e057761d`; closing K1 `818551f7` (RECONCILES, 15/15), K2 `09a540b7` (EXPOSED and RETIRED whole).

**Result.** Outer exit 0, **COMPLETED in 124 s** of a 1,800 s ceiling, record by the script, no survivors, gate
restored and read back. Network in evaluation mode at load and after the last game; model digest unchanged
(`ae0e71fc…`). Games (seeds 202691900, 202691901): **52 and 54 plies, both decisive**; 60.7 s and 62.6 s; **1.1635 s
per ply** overall (about 0.73 ms per simulation). Capture: 776,093 bytes, finalized; write 0.023 s, reopen and
validate 0.020 s. The Stage 0 reader rebuilt 106 records, **every one equal to the learner `PositionRecord`**
(board, mover, legal order, visit counts, outcome); **99 of 106** searches began with inherited child visits, so
the after-search versus before-search target distinction was exercised live.

**Did it match the prediction, and why.** The per-ply estimate was about 2.4 times too high. It extrapolated entry
19's evaluation-path figure (`search_with_root`, one leaf per network call); self-play uses the batched
`search_from_root` (eval batch 14). The two figures come from different search paths and are not comparable as a
machine speed.

**Conclusion / status.** MEASURED. The capture path holds with the live network for these two games, and the
Stage 0 reader reproduces the training rows exactly. Pricing for the plan's 120-game schedule: about 2.1 h at these
lengths, and at most about 10.9 h even if every game reached the 280-ply cap, provided the per-ply time holds in
long games (not measured beyond ply 54). Two games cannot estimate the length distribution, so the corpus's size in
positions (about 6,400 at these lengths) is also uncertain. Stages 2–5, promotion and pushing remain unapproved.
See do-not-repeat **#67**.

*Correction (2026-10-22, after the user's review of entry 21).* Entry 21 (and the build commit `24327418`) said the
missed mutant survived because "before- and after-search counts coincide" in the fake-evaluator games. That is
wrong. Without inherited visits the **before-counts are all zero and the after-counts equal the newly added
counts**; before and after do not coincide. What the missed mutant actually tested: it took the before-counts
**only where they had mass and fell back to the after-counts otherwise**. Every fake-game row had zero before-mass,
so it never substituted anything and was equivalent to the original on that data. A plain before-for-after
substitution was not run; the data could have caught it. The case the fake data cannot distinguish is **after
versus after-minus-before** (the newly added counts). A synthetic reader test now covers it. It is built on the
scripted capture fixture, where before, after and after-minus-before all differ on inherited rows; it also checks
the selection counts on the adjusted row. The test needs no game and no inference; no mutation campaign was
repeated. Stage 1's live check is unchanged: 99 of 106 rows there had inherited visits.

### 22. Search distillation D2 — Stage 2 corpus generation, 120 games (2026-10-22, COMPLETED)

**Test.** One bounded generation run under card `docs/superpowers/2026-10-22-search-distillation-d2-stage2-card.md`
(pinned by SHA-256 in `scripts/GPU/alphazero/search_distillation_d2_generate.py`): 120 fresh self-play games of
`calib020_0001` at 1,600 simulations, at Stage 1's exact configuration (entry 21), in 12 ordered phases of 10 games,
each phase its own finalized capture directory. Slot `s` uses seed `base + s`; slots with `s % 10 == 9` (one per
phase, 12 games) are held out, the other 108 are training games, fixed before generation. All 12 phases had to
finalize for the corpus to be eligible for training (the plan's six-phase fallback was replaced at approval). 12 h
ceiling including capture and validation; stop on the first failure, with no retry, replacement or top-up. After
each phase the Stage 0 reader rebuilt its records and every one had to equal the learner row. The approval asked for
120 games, not 240. Production of training material only; no strength claim.

**Hypothesis.** None for the generation. (The experiment's hypothesis is stated in entry 21.)

**Why it was plausible.** As for entry 21.

**Prediction on record.** None stated in advance about the corpus. The approved terms recorded the timing as an
estimate: about 2.1 h, projected from Stage 1's two games, with 10.9 h (every game capped) explicitly not a
demonstrated bound.

**What ran.** Build `b015a8fe` (gate `SEARCH_DISTILLATION_D2_GENERATE_AUTHORIZED`, the thirty-third; wrapper with TERM
at 43,190 s and KILL 10 s later; regression at the build equal to the 58-id baseline). Seeds `[202692100, 202692220)`:
survey `88209d76` (CLEAN, 45/45), pre-proof `0d77665b`, registration in ACCOUNTED `6aa0971e` (40 → 41), post-proof
`e10987bc`. Gate opened by `44390545`; one run through `01_wrapper.sh` in a clean linked worktree; evidence and the
restored gate `d78387f2`; closing K1 `985eccbd` (RECONCILES, 15/15), K2 `1d4d7eda` (EXPOSED and RETIRED whole);
registry readers 593 passed; seed, gate and D2 tests 312 passed.

**Result.** Outer exit 0, **COMPLETED in 8,370 s** (2.3 h) of a 43,200 s ceiling, record by the script, no survivors,
gate restored and read back, stderr empty. **12/12 phases finalized**; 120 games, **7,020 records**: 6,403 from the
108 training games and 617 from the 12 held-out games. Every phase's rebuilt records equal the learner rows; 6,728
of 7,020 searches began with inherited child visits. Model digest unchanged (`ae0e71fc…`); 0 of 42 modules in
training mode at load and at the end. Game lengths: 32 to 280 plies, median 53, mean 58.5, 90th percentile 73; four
games over 100 plies (103, 195, and two of 280 that ended at the state cap). Outcomes: 118 decisive, 2 state-cap
draws. Overall 1.19 s per ply (per game 1.00–1.44; the three longest games, 195–280 plies, ran slowest at 1.42–1.44 s per ply); 9 s
outside games. Capture: 50.0 MB in all, 3.5–5.8 MB per phase against an 84 MiB phase ceiling; write and reopen
under 0.24 s per phase. The corpus is at `~/twixt-cache/search-distillation-d2/corpus-20261022T162940Z/`, outside the
repository; `corpus.json` (SHA-256 `a9a425f2…`) lists the phases and their hashes and the training and held-out ids,
and each phase's context, receipts and completion marker are copied into the evidence directory. After the run the
whole corpus was reopened with the Stage 0 reader, independently of the script: 7,020 records, all game ids
distinct, and the context and completion-marker hashes match the record and the evidence copies.

**Did it match the prediction, and why.** No prediction was stated. Against the timing estimate: 2.3 h, not 2.1 h.
The games were longer on average than Stage 1's two (mean 58.5 against 53 plies), and per-ply time was higher in
long games (about 1.43 s per ply at 195–280 plies against 1.16 s in Stage 1), which Stage 1 had not measured
beyond ply 54.

**Conclusion / status.** COMPLETED. The corpus is fixed and eligible for training under the approved terms. It
is training material only and says nothing about target quality or strength. Training (Stage 3), Stages 4–5,
promotion and pushing remain unapproved. See do-not-repeat **#68**.

### 23. Search distillation D2 — Stage 3 training, one run (2026-10-22, STOP)

**Test.** One training run under card `docs/superpowers/2026-10-22-search-distillation-d2-stage3-card.md` (pinned by
SHA-256 in `scripts/GPU/alphazero/search_distillation_d2_train.py`). A student copy of `calib020_0001` was trained on the
Stage 2 corpus (entry 22; `corpus.json` SHA-256 `a9a425f2…` pinned, every phase reopened through the strict reader,
the corpus's split unchanged: 6,403 training and 617 held-out records). The update surface was the encoder and
policy head; the teacher, the value-head weights and every BatchNorm running statistic were held unchanged. The
recipe was frozen: 401 updates (four passes) at batch 64, Adam learning rate 1e-4 (MLX's resolved defaults recorded
and checked: betas 0.9 and 0.999, eps 1e-8, no bias correction), global-norm clip 1.0, value-anchor weight 1.0, L2
1e-4, draw-order seed 0. The held-out records were read only for the teacher baseline and for the final check on
the reloaded candidate. PASS needed finite metrics, held-out policy cross-entropy strictly below the teacher's, and
absolute value drift from the teacher on the held-out records with mean ≤ 0.05 and 95th percentile ≤ 0.15.
Otherwise the verdict is STOP. A pass would have shown fit and bounded value drift, not stronger play.

**Hypothesis.** Training the encoder and policy head on the incumbent's own 1,600-simulation targets, with a value
anchor, fits held-out targets better than the incumbent while keeping its value output within the limits.

**Why it was plausible.** The targets come from a stronger player (Stage A, entry 1, under the earlier protocol),
and the anchor penalizes value movement directly on every batch.

**Prediction on record.** None stated in advance for the outcome. The plan's cost estimate was under 0.5 h; a
timing check with a random-initialized network of the same architecture, on the training records only, put the
401 updates at about 45 s.

**What ran.** Build `1a92eb7c`: entry point, the gate `SEARCH_DISTILLATION_D2_TRAIN_AUTHORIZED` (the thirty-fourth), a
wrapper with TERM at 1,790 s and KILL 10 s later, and the card. It had 23 run-module and 17 wrapper tests. Of 14
injected defects, 12 were caught at first. The two misses were the loaded-teacher pin and the Adam-default check;
each got its own test, and both were then caught. Regression at the build equalled the 58-id baseline (3,272
passed). Gate opened by `24a3eaaf`; one run through `01_wrapper.sh` in a clean linked worktree; evidence and the
restored gate `1c489c5b`. No game seeds were drawn.

**Result.** Outer exit 6, **COMPLETED, verdict STOP**, 34 s in all (401 updates in 24.8 s on the GPU), record by the
script, no survivors, gate restored and read back, stderr empty.
- **Training checks:** every check passed. All 60 surface tensors changed, the 32 off-surface tensors were bitwise
  unchanged, and the teacher's digest (`ae0e71fc…`) was unchanged; every metric was finite.
- **Held-out policy cross-entropy:** 4.8243 for the teacher, 4.1474 for the candidate (KL to the targets 1.9714 →
  1.2945), strictly lower.
- **Held-out absolute value drift from the teacher:** **mean 0.1026 (limit 0.05), p95 0.2791 (limit 0.15)**, max 0.5795.
  Both limits failed.
- **Training-set diagnostics** (judging nothing): cross-entropy 4.6958 → 3.9801; drift mean 0.0679, p95 0.1975,
  max 0.5369.
- **Training log:** the anchor term's batch mean fell from 0.038 over the first 50 updates to 0.011 over the last 50.
  Every update was clipped: pre-clip gradient norms were all above 1, with a median of 4.46. On 86 of 401 updates
  they exceeded 10,000 (maximum 43,779 at update 361), in 68 separate runs between updates 21 and 389. The cause
  was not derived.
- **Candidate:** kept as a research artifact outside the repository, made read-only after the run, at
  `~/twixt-cache/search-distillation-d2/student-stage3-20261022T232207Z.safetensors`, SHA-256 `8552edb4…`, state digest
  `9c73feb2…`. The strict reload reproduced the state digest.

**Did it match the prediction, and why.** No outcome was predicted. The run was faster than the timing check (24.8 s
against about 45 s). On drift: the anchor weight of 1.0 did not hold the value output within the limits while the
encoder moved; how much each part of the recipe contributed was not measured.

**Conclusion / status.** STOP. The recipe fits the held-out targets better than the incumbent but moves its value
output beyond both fixed limits, so Stages 4–5 are not entered under this card, and the candidate is not used.
This says nothing about strength in either direction. The gradient-norm spikes are an observation without a
derived cause. Promotion and pushing remain unapproved. See do-not-repeat **#69**.

*Correction (2026-10-22, the same day, after rechecking entry 23's training-log figures).* Entry 23 says the 86
updates with pre-clip gradient norms above 10,000 came "in 68 separate runs between updates 21 and 389". The 68
runs counted the 88 updates above **100**. The 86 updates above 10,000 form **69** runs; their first and last
updates are also 21 and 389. Nothing else in the entry changes.

### 24. Search distillation D2 — gradient diagnostic, one instrumented replay (2026-10-22, MISMATCH)

**Test.** One instrumented replay under card `docs/superpowers/2026-10-22-search-distillation-d2-gradient-diagnostic-card.md`
(pinned by SHA-256 in `scripts/GPU/alphazero/search_distillation_d2_gradtrace.py`), approved as a diagnostic-only
exception to #69's replay restriction. It does not reopen the Stage 3 candidate. Replaying optimizer updates is
training computation, recorded here as a diagnostic replay.
- **Inputs:** the unchanged incumbent (`ae0e71fc…`), loaded as teacher and student. Stage 3's exact recipe and draw
  order: 401 updates, batch 64, Adam 1e-4 with resolved defaults checked, clip 1.0, anchor 1.0, L2 1e-4, seed 0.
- **Training games only:** each phase's context, receipts and completion marker were checked against the manifest,
  and only the 108 training games' files were opened, each against its receipt. The record lists the 144 files
  opened: 36 phase files and 108 training games. No held-out game file was opened. The 6,403 training records'
  identities matched a digest pinned in the card (`e90df4d3…`).
- **Reference:** Stage 3's record, by SHA-256 (`30b1d3ad…`).
- **Comparison:** every update's loss, policy term, anchor term and pre-clip norm must satisfy
  `|replayed − reference| ≤ 1e-8 + 1e-5 × |reference|`. The first update outside that stops the replay: MISMATCH.
- **Recorded:** per update, the batch, every surface tensor's pre-clip gradient norm, the clip scale, and each
  tensor's applied update. Detailed attribution (loss components, per-record gradients, BatchNorm taps) was planned
  at updates 20, 21 and 361 only.
- No checkpoint was written; nothing about the student was judged.

**Hypothesis.** None about a cause. The diagnostic's question was descriptive: which tensors, records and loss terms
carry Stage 3's spike norms.

**Why it was plausible.** The pre-run inspection, read-only and recorded in the card's §2:
- the logged norms were bimodal (303 below 10, 86 above 10,000) with ordinary loss values on spike updates;
- the draw order alone ruled out a fixed set of records that spike whenever drawn;
- 297 encoder BatchNorm channels hold a stored variance of exactly 0. Evaluation mode multiplies them by about 316.

These made a per-tensor and per-record measurement informative.

**Prediction on record.** From the card's §7:
- **P1:** the replay reproduces the trace within the tolerance.
- **P2:** on spike updates, one surface tensor carries more than half the squared pre-clip norm.
- None on which tensor, records or loss component, or on any link to the value drift.

**What ran.**
- **Build `1e26b8fa`:** the entry point; the gate `SEARCH_DISTILLATION_D2_GRADTRACE_AUTHORIZED` (the thirty-fifth); a
  wrapper with TERM at 590 s and KILL 10 s later; the card. It had 36 run-module and 17 wrapper tests.
  - The tests showed that `train_student`, the bare replay and the replay attributing every update leave the same
    student to the bit, and that the replay reproduces a Stage 3 fixture run's trace to the bit on the CPU.
  - Of 18 injected defects, 17 were caught at first. The miss was recording post-clip norms as pre-clip: the
    fixture's norms never reached the clip. A test with a binding clip was added, and it was then caught.
  - Regression equalled the 58-id baseline (3,272 passed).
- **Run:** gate opened by `959488e9`; one run through `01_wrapper.sh` in a clean linked worktree. Evidence and the
  restored gate are in `5f1a6a05`. No game seeds were drawn.

**Result.** Outer exit 6, **COMPLETED, trace MISMATCH**, 19 s wall clock. The record was written by the script;
there were no survivors; the gate was restored and read back; stderr was empty.
- **Updates 0–20:** all four logged values were **bitwise identical** to Stage 3's, including update 20, which ran
  after its own attribution pass.
- **Update 21** (Stage 3's first spike):
  - loss 4.500528 against 4.492997;
  - policy term 4.377072 against 4.369541;
  - anchor term bitwise identical;
  - pre-clip norm 12,854 against 11,404.

  The replay stopped there, with 22 updates traced.
- **Measured on the replayed update 21** (`trace.jsonl`, Stage 3's batch, the replay's own values):
  - the policy head carried 96.1% of the squared pre-clip norm: `policy_head.fc.weight` 75.1%, `policy_head.fc.bias`
    16.0%, `policy_head.bn.bias` 2.8%, `policy_head.conv.weight` 1.8%;
  - the encoder carried 3.9%, which is still a norm of about 2,500 against a total of 2–12 on updates 0–20;
  - the largest encoder tensor was `encoder.conv1.weight` at 1,413.
  - On updates 0–20 the policy head's share was between 0.5% and 7.9%.
  - The clip scaled update 21's gradient by 7.8e-5, yet its applied update had a total L2 of 0.116 against 0.128 on
    update 20. Adam's step is not proportional to the clipped gradient.
- **Lost diagnostics (a defect in the build):** the step-20 and step-21 attribution passes ran, but the record keeps
  attribution only for a completed replay, so the mismatch discarded them. The card required diagnostics to be
  kept, and the tests did not check that attribution survives a mismatch. The per-update trace for 0–21 was kept.

**Did it match the prediction, and why.**
- **P1 failed.** The replay is not bitwise reproducible at Stage 3's first spike, although it was for the 21 updates
  before it.
- **P2** held for the one spike measured (75%), but that spike is the replay's, not Stage 3's, and one update cannot
  describe all 86.
- **Measured attribution:**
  - the divergence appeared at the first spike update and nowhere earlier;
  - it changed the policy term but not the anchor term;
  - the replayed spike's norm sat mostly in the policy head's small fully connected layer.
- **Causal hypotheses, none tested:**
  - a non-deterministic GPU computation in the policy path on such batches;
  - a GPU-side effect of the attribution pass that ran just before update 21. Update 20 argues against this but does
    not exclude it;
  - very large gathered policy-head inputs in particular positions, possibly fed by the zero-variance BatchNorm
    channels. The BatchNorm taps that would have measured this were among the discarded attribution.
- None of this is shown to cause Stage 3's value drift.

**Conclusion / status.** MISMATCH, closed. The diagnostic showed that Stage 3's trace is reproducible up to its first
spike and not at it, and that the replayed spike sits in the policy head. It did not attribute the spikes to records
or loss terms, and it derived no cause. The Stage 3 candidate stays unused. Stages 4–5, further learning
experiments, promotion and pushing remain unapproved. See do-not-repeat **#70**.

### 25. Search distillation D2 — reproducibility diagnostic, three passes over updates 0–21 (2026-10-22, MEASURED)

**Test.** One run under card `docs/superpowers/2026-10-22-search-distillation-d2-reproducibility-card.md` (pinned by
SHA-256 in `scripts/GPU/alphazero/search_distillation_d2_gradrepeat.py`), approved after entry 24.
- **Three passes in one process, in this order, each on Stage 3's updates 0–21:**
  - **A**, plain;
  - **B**, with entry 24's attribution before updates 20 and 21;
  - **C**, plain.
- **Identical inputs for every pass:** a fresh load of the incumbent (digest checked), a fresh Adam, the same
  training records (training games only, read by entry 24's reader; no held-out game file opened), the same draw
  order and the same teacher values. Those values were computed once, and their digest recorded.
- **Recorded:**
  - every update's values;
  - at update 21, the model, Adam and input digests, before the attribution point, before the update and after it;
  - B's attribution, written and fsynced before each of its updates;
  - after each pass, at that pass's own pre-update-21 state reloaded into a fresh network, the scale of update 21's
    batch: policy-head inputs, hidden layer, legal-logit magnitude and spread, BatchNorm taps, and a recomputation
    of update 21.
- A difference was an observation, not a stop. No checkpoint was saved.

**Hypothesis.** None about a cause. The question was whether the update path repeats, and whether instrumentation
accompanies a difference.

**Why it was plausible.** Entry 24 matched Stage 3 bit for bit through update 20 and diverged at update 21. Matching
scalars did not establish matching weights or Adam state.

**Prediction on record.** None stated in advance.

**What ran.**
- **Build `aaf86625`:** the entry point; the gate `SEARCH_DISTILLATION_D2_GRADREPEAT_AUTHORIZED` (the thirty-sixth); a
  wrapper with TERM at 590 s and KILL 10 s later; the card.
  - It had 23 run-module and 16 wrapper tests. These include an attribution surviving a later non-finite stop and a
    SIGTERM, and the plain passes running nothing between their updates.
  - All 16 injected defects were caught. One of them restores entry 24's retention defect.
  - Regression equalled the 58-id baseline (3,272 passed).
- **Run:** gate opened by `3d59b75e`; one run through `01_wrapper.sh` in a clean linked worktree. Evidence and the
  restored gate are in `be47a5ea`. No game seeds were drawn.

**Result.** Outer exit 0, **COMPLETED**, 22 s wall clock. The record was written by the script; there were no
survivors; the gate was restored and read back; stderr was empty. All five planned files were written.

Pre-clip norms:

| Pass | Update 0 | Update 20 | Update 21 |
|---|---|---|---|
| A | 12.445 | 4.863 | **11,404.4** |
| B | 16.628 | 4.624 | **3,867.0** |
| C | 16.628 | 4.624 | **3,511.6** |
| Stage 3 | 12.445 | 4.863 | 11,404.4 |
| Entry 24 | 12.445 | 4.863 | 12,854.4 |

- **Pass A** equalled Stage 3 bit for bit on all four logged values at all 22 updates, update 21 included.
- **Passes B and C differed from A from update 0.** At update 0 all three had the same loss, policy and anchor terms
  (the starting state is the same), but B and C's pre-clip norm was 16.628 against A's 12.445.
  - B and C were bitwise identical to each other on updates 0–20.
  - Before update 21 they held **identical model, Adam and input digests**, at both digest points.
  - At update 21 their loss terms were bitwise identical, but their pre-clip norms differed (3,866.95 against
    3,511.61), and so did their states after the update.
- **B's attribution left the state unchanged.** In every pass the digests before and after the attribution point
  were equal.
- **Linearity:**
  - At update 20, B's per-record policy gradients summed to the batch gradient exactly (relative gap 0.0).
  - At update 21, **every one of the 64 records' own policy gradients had norm ≈ 3,867**: median 3,867.41,
    maximum 3,867.71. Each was computed with that record's weight alone set to 1/64. Their sum was about 64 times
    the batch gradient (relative gap 63.0).
  - The anchor gradients stayed linear (gap 0.0).
- **Recomputation at each pass's own pre-update-21 state.** The loss terms equalled the pass's bit for bit. The
  pre-clip norms did not: 3,075 against A's 11,404, and 192 against B's 3,867 and C's 3,512.
- **Two evaluations of the same logits disagreed.** The policy head re-run step by step gave legal-move logits differing from
  the module's own, on the same state, by up to 2.77 in A, 0 in B and 11.5 in C. The BatchNorm re-run matched to
  0 in every pass.
- **Scale of update 21's batch** (module values, all passes alike):
  - legal-logit magnitude at most 14.4 (median 11.6); spread at most 15.3 (median 6.3–6.4);
  - gathered policy-head inputs at most 2.7; hidden layer at most 2.2; encoder output at most 13.6.
  - Normalized inputs in the low-variance BatchNorm channels reached 55–61, against 27–28 in the other channels.
    At update 20 they reached 47.

**Did it match the prediction, and why.** None was stated.

**Measured attribution:**
- the update path is **not a deterministic function of state and inputs** here: the same digests gave different
  gradients;
- the first pass in each process (Stage 3, entry 24 and A) agreed with one another on updates 0–20, and the later
  passes in this process (B and C) agreed with each other there, but the two groups differed from update 0;
- the attribution pass did not change the state, and a pass without attribution (C) also differed;
- at update 21, the policy part of the computed gradient violates the linearity that the stated loss's gradient must
  have.
- The batch's forward values are of ordinary scale. No large policy-head input or logit appeared in this
  measurement, though the step-by-step policy-head forward was not repeatable either.

**Causal hypotheses, none tested:**
- an operation in the policy path (the gather at legal moves, its backward, or the masked log-softmax) reading or
  accumulating stale or uninitialized memory. This would fit results that depend on what ran earlier in the
  process;
- a different MLX or Metal defect on this path.

Whether Stage 3's 86 spikes share this property is not measured; only update 21 was. Whether it explains Stage 3's
value drift is not measured.

**Conclusion / status.** MEASURED, closed.
- The Stage 3 spikes at update 21 are not the gradient of the stated loss: the computed gradient there fails
  linearity, and it is not repeatable from identical state and inputs.
- Training on this GPU path cannot be taken as deterministic, and its gradients cannot be taken as correct, until
  they are checked against an independent reference.
- The mechanism is not derived.
- The Stage 3 candidate stays closed; `calib020_0001` stays the incumbent.
- Stages 4–5, further learning experiments, promotion and pushing remain unapproved. See do-not-repeat **#71**.

### 26. Search distillation D2 — gradient correctness check, CPU/GPU and an independent reference (2026-10-22, MEASURED)

**Test.** One run under card `docs/superpowers/2026-10-22-search-distillation-d2-gradient-check-card.md` (pinned by
SHA-256 in `scripts/GPU/alphazero/search_distillation_d2_gradcheck.py`), approved after entry 25 as a diagnostic-only
exception to the closed replay restriction.
- **Two fixed states:**
  - **S0**, the incumbent, on Stage 3's update-0 batch;
  - **S21**, a newly reconstructed pre-update-21 state, on the update-21 batch. It comes from ONE GPU prefix of
    updates 0–20, and its digest was compared with entry 25's recorded digests.
- **CPU/GPU comparison:** at each state, three evaluations per device of the training loss's forward outputs and all
  60 gradient tensors, each from fresh copies of the state and batch arrays with digests checked. No optimizer
  update. Plus weighted-record additivity.
- **An independent reference:** NumPy float64 with hand-derived derivatives for the policy head (1×1 conv,
  BatchNorm in evaluation mode, ReLU, gather with scatter-add, two linear layers, masks, masked cross-entropy). It is
  validated by fixed directional finite differences where no ReLU boundary is crossed. Cases: four synthetic
  (repeated indices, batch members, padding, weighted additivity) and the real batches at both states.
- **Stage-by-stage localization:** each stage's output and vector-Jacobian product on each device, three times, from
  the CPU chain's identical inputs and cotangents. A decomposition check compares the composed chain with autodiff.
- Every result was written as it was produced. Fixtures for disagreeing components were saved outside the
  repository; no encoder weights and no checkpoint were saved.

**Hypothesis.** None about a mechanism. The questions: do the CPU and GPU agree; does either agree with an
independent reference; where is the first discrepancy, given identical inputs.

**Why it was plausible.** Entry 25 measured gradients that differ from identical state and inputs, and a failure of
the linearity identity, without locating either.

**Prediction on record.** None stated in advance.

**What ran.**
- **Build `06c2a74d`:** the entry point; the gate `SEARCH_DISTILLATION_D2_GRADCHECK_AUTHORIZED` (the thirty-seventh); a
  wrapper with TERM at 590 s and KILL 10 s later; the card.
  - It had 24 run-module and 16 wrapper tests. These include injected faults at the gather, the masked log-softmax
    and the encoder, each located at that stage with its fixture saved; a stage that is not bitwise repeatable
    being flagged; and an every-coordinate finite-difference test of the NumPy derivatives.
  - All 13 injected defects were caught, including a non-accumulating scatter, a dropped BatchNorm scale and a
    policy-only encoder cotangent.
  - Regression equalled the 58-id baseline (3,272 passed).
- **Run:** gate opened by `84d7b00c`; one run through `01_wrapper.sh` in a clean linked worktree. Evidence and the
  restored gate are in `a30c852f`.
  - Ten fixtures, 64 MB, read-only, are at `~/twixt-cache/search-distillation-d2/gradcheck-fixtures-20261023T034401Z/`.
    Their hashes are in `fixtures.jsonl`.
  - No game seeds were drawn.

**Result.** Outer exit 0, **COMPLETED**, 29 s wall clock. The record was written by the script; there were no
survivors; the gate was restored and read back; stderr was empty. The teacher values' digest equalled entry 25's.
- **S21:** the reconstruction reproduced Stage 3's updates 0–20 bit for bit. Its digest (`91816a99…`) **equals pass
  A's recorded pre-update-21 digest**, not passes B and C's. It is a reconstruction, not the state entry 25 held.

***1. CPU/GPU agreement: they do not agree.***
- **CPU:** bitwise repeatable at both states (all 60 tensors and the forward outputs, three times each). Additivity
  held for every tensor.
- **GPU forward:**
  - logits differ from the CPU by at most 0.18 (S0) and 0.12 (S21), within the tolerance;
  - values differ by 0.038 and 0.016, outside it;
  - loss terms are bitwise repeatable on the GPU.
- **GPU gradients:**
  - 59 of 60 tensors are outside the tolerance in every evaluation at both states;
  - S0 norms are 16.63 three times, against the CPU's 13.71; the second evaluation's tensors were bitwise equal to
    the first for only 19 of 60;
  - **S21 norms are 4.32, 833.5 and 2,900.0**, against the CPU's 4.23 three times; only 4 of 60 tensors repeated
    bitwise;
  - additivity failed for 58 of 60 tensors at both states.
- **Context dependence of a GPU forward value:** at S0 on update 0's batch, the policy term was 4.576662 inside the
  training loop (Stage 3, pass A and this run's reconstruction), but 4.563246 in this run's standalone GPU
  evaluation of the same state and batch. The CPU gave 4.563363.

***2. Agreement with the independent reference (NumPy float64, policy head only).***
- The NumPy derivatives agreed with central finite differences in the four synthetic cases (relative 1e-8–1e-10).
  The real-batch directional checks were **skipped**: every step down to ε = 6.25e-7 crossed a ReLU boundary.
- **CPU MLX** agreed with NumPy within the tolerance in **all 6 cases**: every tensor and the features' gradient,
  bitwise repeatable, single-record gradients adding up.
- **GPU MLX** disagreed in **all 6 cases**:
  - **Synthetic cases:** relative errors of about 1e-3, repeatable, slightly beyond the tolerance. Directional
    derivatives were 0.3–0.6% off the finite difference, against the CPU's agreement to about 1e-6.
  - **Real batches:** off by up to 8% of scale in the conv, BatchNorm and features' gradients, not repeatable. The
    summed single-record gradients missed the full gradient by about 10⁴: for the features' gradient, 910 against a
    scale of 0.042 at S0 and 4,300 against 0.044 at S21; for the BatchNorm bias, 5.5e4 and 6.3e4.

***3. The first localized discrepancy (identical stage inputs and cotangents, GPU against the CPU chain).***
- **Decomposition:** the CPU chain reproduced CPU autodiff bitwise for all 60 tensors at both states.
- **Canonicalization:** bitwise identical on the GPU.
- **First in forward order: the encoder.** Its output differs by 0.19 (S0) and 0.17 (S21) against values up to
  about 13, and all 52 of its tensor gradients are outside the tolerance. It **repeats bitwise on the GPU**. No
  independent reference covers the encoder, so which device is right is not established.
- **Inside the policy head:**
  - **Agree within the tolerance and bitwise repeatable**, in forward and backward: the 1×1 conv, BatchNorm, ReLU,
    gather (its scatter-add backward included), fc ReLU, head mask, masked log-softmax, policy CE, value head and
    anchor. Differences are at most about 1e-6 absolute; the value head's 0.0011 at S21 is the exception, outside
    the tolerance.
  - **The fc layer** (Linear 2→64 on 33,600 and 33,728 rows): forward outside the tolerance by 0.36–1.7. Its
    gradient with respect to its input is off by 2.3 against a scale of 0.065, and **neither repeats bitwise**. Its
    weight and bias gradients are within the tolerance.
  - **The out layer** (Linear 64→1): forward within the tolerance (3.8e-6). Its **input gradient** is off by 1.7
    against a scale of 0.013, and is not repeatable at S0. **This is the first discrepancy in backward order.** Its
    weight and bias gradients are within the tolerance.

**Did it match the prediction, and why.** None was stated.

**Measured attribution:**
- the GPU path disagrees with the CPU and with an independent reference for the policy head, where the CPU agrees
  with the reference;
- from identical inputs, the policy head's discrepancies are confined to the two small linear layers: fc's forward
  and input gradient, and out's input gradient;
- the encoder differs from the CPU, repeatably on the GPU, and its correct value is not established;
- a GPU forward value depends on the computation it is evaluated within.

**Causal hypotheses, none tested:**
- a GPU defect in matrix products with a very narrow dimension (inner dimension 2 for fc's forward, 1 for out's
  input gradient, output width 2 for fc's input gradient) at about 33,000 rows;
- dependence on memory layout or buffer reuse, which would fit the context-dependent forward value and the passes in
  entry 25;
- the encoder's difference coming from ordinary float32 rounding amplified by the zero-variance BatchNorm channels
  (×316 per layer), rather than from a defect.

Whether earlier checkpoints' training, or Stage 3's value drift, was affected remains unresolved. Earlier match
results remain evidence about those checkpoints.

**Conclusion / status.** MEASURED, closed.
- On this machine's MLX 0.30.3 Metal path, the policy-head gradients on real training batches are wrong against an
  independent reference and not repeatable.
- The discrepancy localizes, from identical inputs, to the policy head's two small linear layers; the encoder's
  difference is unresolved.
- No learning recipe should run on this path until the defect is understood or avoided.
- The Stage 3 candidate stays closed; `calib020_0001` stays the incumbent.
- Stages 4–5, further learning experiments, backend upgrades, production fixes, promotion and pushing remain
  unapproved. See do-not-repeat **#72**.

*Correction (2026-10-22, the same day, on rechecking entry 26's figures against its evidence files).* Four figures in
entry 26 are wrong:
- **GPU policy-head gradients against NumPy on the real batches.** The entry says they are off by "up to 8% of
  scale in the conv, BatchNorm and features' gradients".
  - At S0 they are off by 7–8% of scale, and the fc weight and bias gradients are off by the same amount.
  - At S21 they are off by up to **27%** (the features' gradient), 12–14% for the BatchNorm weight and conv weight,
    and 8–10% for the fc tensors.
- **The NumPy derivatives against central finite differences** agreed to a relative 4e-10–2.3e-8, not "1e-8–1e-10".
- **The GPU's directional derivatives** were 0.1–0.3% off the finite difference, not "0.3–0.6%".
- **The policy-head stages that agree** differ from the CPU chain by at most 1.5e-5 of the tensor's scale (2.2e-5
  absolute, in the conv weight's gradient at S0), not "at most about 1e-6 absolute".

The conclusions do not change.

### 27. Search distillation D2 — linear-layer reproducer (2026-10-23, REPRODUCED IN ISOLATION)

**Test.** One run under card `docs/superpowers/2026-10-23-search-distillation-d2-linear-reproducer-card.md` (pinned by
SHA-256 in `scripts/GPU/alphazero/search_distillation_d2_linrepro.py`), approved after entry 26.
- **Operations:** the policy head's `fc` (Linear 2→64) and `out` (Linear 64→1), run in isolation.
- **Operands:** float32, from entry 26's saved fixtures, each checked against its recorded hash.
  - `out` uses its `stage_<S>_out` fixtures as saved.
  - `fc` has no fixture of its own, so its operands are derived from the `reference_real_<S>` fixtures by the NumPy
    reference and saved.
- **Expected results:** NumPy float64 accumulation of the same operands.
- **Implementations:** `nn.Linear` with `mx.vjp`; explicit `x @ w.T + b` with `mx.vjp`; and explicit products without
  autodiff. Each ran three times on the GPU and once on the CPU.
- **Judgement, frozen:** an element is outside the tolerance beyond 1e-4 of scale + 1e-6, and gross beyond 1e-2 of
  scale. Legal and padded rows are reported separately.
- **Targeted cases**, run only where the original shape reproduced: five row counts, one layout and one evaluation
  context.
- No network was loaded, nothing was trained and no checkpoint was written.

**Hypothesis.** None about a mechanism. The question: does one linear operation alone reproduce entry 26's
discrepancy, against an independently computed result?

**Why it was plausible.** Entry 26 localized the discrepancy, given identical stage inputs, to these two operations.

**Prediction on record.** None stated in advance.

**What ran.**
- **Build `9c874e2a`:** the entry point; the gate `SEARCH_DISTILLATION_D2_LINREPRO_AUTHORIZED` (the thirty-eighth); a
  wrapper with TERM at 590 s and KILL 10 s later, and no checkpoint argument; the card with its 32-case list.
  - It had 17 run-module and 16 wrapper tests.
  - All 15 injected defects were caught. One of them, a cotangent without the ReLU mask, was caught first by the
    clean-run test through a cancellation artifact; the operand test also catches it.
  - Regression equalled the 58-id baseline (3,272 passed).
- **Run:** gate opened by `12f0de35`; one run through `01_wrapper.sh` in a clean linked worktree. Evidence and the
  restored gate are in `024ca0fe`.
- **Artifacts:** 34 artifacts (1.6 GB, read-only) are at `~/twixt-cache/search-distillation-d2/linrepro-20261023T041634Z/`,
  with their SHA-256s in `artifacts.jsonl`.
  - The size comes from the frozen rule: every output outside the tolerance in every repeat was saved, including
    the GPU's ~1e-3 deviations in nearly every case.

**Result.** Outer exit 0, **COMPLETED**, 43 s wall clock. The record was written by the script; there were no
survivors; the gate was restored and read back; stderr was empty.
- **The failure reproduces in isolation** at all four original shapes (`fc` and `out`, S0 and S21), in **all three
  implementations**, so neither `nn.Linear` nor autodiff is required.
  - **CPU:** every output within the tolerance, up to 2e-6 of scale, except the `out` bias artifact below.
- **The gross rows are exactly the first N − 32,736**, in every repeat, every implementation, and the alternative
  layout and context:

| Case | Rows | Gross rows | Legal | Padded |
|---|---|---|---|---|
| S0 | 33,600 | 0–863 (864) | 848 | 16 |
| S21 | 33,728 | 0–991 (992) | — | — |

  - The same rows are affected in `fc`'s forward Y, `fc`'s input gradient dX and `out`'s input gradient dX.
  - In `fc`'s forward, those rows' outputs equal the bias alone to within 0.002–0.009, as if their X·Wᵀ term were
    missing.
  - In the two input gradients, the corrupted rows match neither zero nor the rows 32,736 further on. Their content
    is not characterized.
  - The weight and bias gradients, which sum over rows, are within the tolerance.
- **Row counts:** with the first 512, 2,048, 8,192, 16,384 or **32,768** rows, **no element is gross** in any
  implementation.
- **Elsewhere the GPU carries a tolerance-level error:** `fc`'s forward is off by about 1.1e-3 of scale and the
  input gradients by about 8e-4, against the CPU's about 1e-7.
- **Alternative layout** (X as a transposed view) and **alternative context** (each output evaluated alone): the same
  gross rows.
- **Repeats:** in `explicit_products` the corrupted rows were bitwise identical across the three repeats. Some
  `linear_autodiff` outputs did not repeat bitwise.
- **A judgement artifact, measured:** `out`'s bias gradient is flagged "gross" on the CPU too (6.2 of scale at S0).
  The cotangent sums nearly to zero, so the scale is tiny.
  - That flag alone is why the 512-row `out` cases are marked "reproduces".
  - Their input gradients are only at tolerance level (9e-4 of scale).

**Did it match the prediction, and why.** None was stated.

**Measured attribution:**
- a single float32 linear operation on the GPU, alone, from saved operands, gives wrong values in a contiguous block
  of leading rows when the operation has 33,600 or 33,728 rows, and none at 32,768;
- the error does not depend on `nn.Linear`, on autodiff, on X's layout or on whether outputs are evaluated jointly;
- in the forward pass, the corrupted rows look like rows whose X·Wᵀ term is missing.

**Causal hypotheses, none tested:**
- a GPU matrix-multiply dispatch or indexing defect for very narrow operands when the row count passes a
  power-of-two boundary (2¹⁵ = 32,768). The constant 32,736 = 2¹⁵ − 32 would fit a tiling of the row dimension;
- the ~1e-3 tolerance-level error coming from reduced-precision accumulation in the same kernels.

How this produced entry 25's linearity failure (every record's gradient carrying the same ~3,867 component) is not
derived. It would be consistent with corrupted rows that do not depend on the record weights.

**Conclusion / status.** REPRODUCED IN ISOLATION, closed.
- There is now a compact reproducer with an independently computed expected result: `fc_S0_original.npz`
  (SHA-256 `8b610769…`), containing operands `X` (33,600 × 2), `W` (64 × 2), `b` (64) and `C`, and
  `expected.Y = X·Wᵀ + b` in float64.
- It supports a precise bug report or fix, and serves as the check for validating a repair.
- The mechanism is not derived.
- Training on this path stays blocked.
- The Stage 3 candidate stays closed; `calib020_0001` stays the incumbent.
- Stages 4–5, further learning experiments, backend upgrades, production workarounds, bug reports and pushing
  remain unapproved. See do-not-repeat **#73**.

*Addendum (2026-10-23, the same day).* Entry 27's table left S21's legal/padded split blank. Of the 992 gross rows at
S21, 883 are legal and 109 padded, from `cases.jsonl`; at S0, 848 and 16. In both cases the split simply follows
where the padded slots fall within the leading rows.


### 28. Search distillation D2 — synthetic row-count probe: unplanned GPU execution during mutation controls (2026-10-23, INCIDENT)

**What was approved.** One synthetic-only probe under a single 10-minute ceiling, a bug-report draft (submission
separate), and a read-only historical exposure audit (entry 29).
- **Card:** `docs/superpowers/2026-10-23-search-distillation-d2-synthetic-rows-probe-card.md`.
- **Generator:** the standalone `scripts/GPU/alphazero/mlx_matmul_rows_probe.py` (standard library, NumPy and MLX
  only).
- **Gated entry:** `search_distillation_d2_mlxrows.py`, gate `SEARCH_DISTILLATION_D2_MLXROWS_AUTHORIZED` (the
  thirty-ninth).
- **Cases:** 27 — fc 2→64, out 64→1 and a wide 64→64 control at nine row counts.
- **Predictions on record**, card §8, stated before any run:
  - **fc Y, fc dX and out dX on the GPU:** no gross rows at N ≤ 32,768; exactly rows [0, N − 32,736) gross at
    32,769, 32,800, 33,600 and 33,728;
  - out Y, every dW and every db clean;
  - the CPU clean;
  - GPU repeats bitwise identical;
  - none for the wide control.

**What happened.**
- **Build `38deadf6`:** 17 generator, 9 entry and 16 wrapper tests. All 28 injected defects were caught, and
  regression equalled the 58-id baseline.
- **During the mutation-control pass, before that commit, under the defect "entry_gate_unchecked":**
  - The test `test_THE_GATE_IS_CLOSED_IN_THE_SOURCE_and_a_closed_gate_refuses` called the public entry **without**
    the device remap or the output redirection. Only the gate-open tests had them.
  - With the gate check removed, the entry ran the full frozen probe on the **real GPU**, inside the disposable
    control worktree, at 2026-10-23 12:24 UTC.
  - The test then failed, so the control counted as "caught".
  - The probe's evidence directory lay inside that worktree and was deleted with it. No `record.json`,
    `cases.jsonl` or environment survives.
  - Its example directory was the real `~/twixt-cache/search-distillation-d2/mlxrows-20261023T122440Z/`. It holds
    one read-only file, `compact_example.npz` (266,867 bytes, SHA-256
    `7c412fd35ab6b55c40d1232d918b66d90ed4d4a3f12e9e5e913ef26a5e80bdc9`).
  - Under the generator's rule, that file is written only when some case had gross GPU rows in Y or dX. Nothing else
    about that execution is known, and the file has not been opened.
- **Found** when the stray directory was seen during the idle check, after the gate-open commit `db2220d4` on a
  linked worktree and before any wrapper invocation.
- **Response:**
  - That worktree and branch were removed; all 39 gates were read back closed.
  - The approved invocation was **not** made.
  - **Fix `0c4b96bc`:** every entry test is now isolated by an autouse fixture (CPU devices, tmp outputs), and a test
    asserts it.
  - The control harness now stops if `~/twixt-cache` changes under any mutant. Rerun: 28 of 28 caught, cache
    unchanged.
- The earlier diagnostics' cache directories (one each for the gradient check and the linear reproducer) show no
  similar stray outputs.

**Result.** None. This is an execution without a record, not a measurement. The stray file is preserved untouched
and is not interpreted.

**Did it match the prediction, and why.** Not assessable: no record survives.

**Conclusion / status.** INCIDENT, closed.
- **The cause:** a test design that isolated only the gate-open path. A mutant that removes the gate check turns a
  closed-gate test into a real run.
- **Open decision for the user:** whether the approved single invocation may still be made, given this unrecorded
  GPU execution of the same frozen probe.
- See do-not-repeat **#74**.

### 29. Search distillation D2 — historical exposure audit (2026-10-23, READ-ONLY)

**Test.** A read-only audit, approved with the probe. Document:
`docs/superpowers/2026-10-23-search-distillation-d2-exposure-audit.md` (`93d52b46`).
- **The question:** which historical GPU trainings ran the policy head at the defect's shape, B × M rows above
  32,768, where M is the batch's padded move width.
- **The labels:** confirmed matching shapes, possible exposure, or insufficient records.
- **How it was done:** source and git history, sidecars, logs, records and `.venv` metadata were read, partly through
  a delegated read-only agent. The trainer cap, the batch sizes, the D2 widths and the `.venv` dates were re-checked
  directly.

**Hypothesis.** None. **Why it was plausible.** The incumbent's metadata records batch size 64. **Prediction on
record.** None stated in advance.

**Result.**
- **The trainer:** `make_padded_batch` sets M = min(batch maximum, `max_moves_cap`). The default cap was **512** until
  `ffe6b342` (2026-10-20, raised to 576), and no caller overrides it. All 542 sidecars record B = 64.
  - So every `checkpoints/` run, including every ancestor of the incumbent, ran at most **32,768** rows.
  - INFERENCE: essentially always exactly 32,768, since 25.4% of self-play positions exceed 512 legal moves.
  - Value-only and partial trainings still computed the policy head's forward and backward on the GPU.
- **Those runs are labelled insufficient records:**
  - their structure excludes the observed N > 32,768 failure;
  - their machine (INFERENCE: the old M3 Pro) and MLX version (unrecorded; 0.30.3 after 2026-05-09 by INFERENCE;
    unknown before) leave 32,768 itself unmeasured on that hardware.
- **D2 Stage 3, updates 0–21: confirmed matching shapes.** 64 × M = 33,600, 33,792, 33,664 and 33,728 rows, derived
  from the batch identities recorded in the gradient diagnostic's `trace.jsonl` and the training-game widths (training
  reader only), on MLX 0.30.3 and the M5 Max, where the defect reproduced.
- **Possible exposure:**
  - **Stage 3 updates 22–400:** identities not recorded; 27.5% of training positions exceed 512.
  - **D2's evaluation `forward()`:** chunks of 64, cap 576.
- **Inference paths:** self-play stays at or below 7,392 rows and play at or below 528.
- **Disclosure:** the delegated agent's width count (27.85% of 7,020 records) read legal-move counts from **all** D2
  corpus game files, held-out games included, and printed truncated fields of one game file's structure. The audit uses
  the training-only figure instead.

**Did it match the prediction, and why.** None was stated.

**Conclusion / status.** READ-ONLY AUDIT, closed.
- Only D2 Stage 3 is confirmed at the defect's shapes.
- No checkpoint is found corrupted.
- Measured playing-strength results stand as evidence about their checkpoints.
- Since `ffe6b342`, the trainer's default cap would put B = 64 training above 32,768 rows on the GPU.
- See do-not-repeat **#75**.

### 30. Search distillation D2 — synthetic row-count probe, the first retained complete record (2026-10-23, COMPLETED; predictions 1 and 4 NOT MATCHED)

**Test.** The frozen probe of entry 28, unchanged: card
`docs/superpowers/2026-10-23-search-distillation-d2-synthetic-rows-probe-card.md`, generator
`scripts/GPU/alphazero/mlx_matmul_rows_probe.py` (SHA-256 `1217bc8e…`, pinned), 27 cases (fc 2→64, out 64→1 and the
wide 64→64 control at 32,704; 32,736; 32,737; 32,767; 32,768; 32,769; 32,800; 33,600 and 33,728 rows), each on the GPU
three times and on the CPU once, against NumPy float64. Synthetic operands only; no repository model, fixture or corpus.

**This is the first retained complete record, not the first execution.**
- The same frozen probe ran once before, unrecorded, during mutation controls (entry 28; fix `0c4b96bc`). That
  execution left `~/twixt-cache/search-distillation-d2/mlxrows-20261023T122440Z/compact_example.npz` (266,867 bytes,
  SHA-256 `7c412fd35ab6b55c40d1232d918b66d90ed4d4a3f12e9e5e913ef26a5e80bdc9`). It is preserved unchanged and was not
  opened.
- **Disclosure:** that file's existence already showed, before this run, that at least one case had gross GPU rows in
  Y or dX. This run gives reproducible measurements; it is **not an independent first test**.
- The user approved this repeat explicitly after reviewing the isolation fix.

**Hypothesis.** The entry 27 failure (GPU, first N − 32,736 rows gross above 32,768 rows) is a property of the
backend at these shapes, not of the D2 fixtures' values. **Why it was plausible.** Entry 27 reproduced it in
isolation, in three implementations, with the CPU clean.

**Prediction on record** (card §8, stated before either execution, derived from entry 27 only):
1. fc Y, fc dX and out dX on the GPU: no gross rows at N ≤ 32,768; at N = 32,769, 32,800, 33,600 and 33,728 the gross
   rows are exactly [0, N − 32,736): 33, 64, 864 and 992 rows.
2. out Y, every dW and every db: no gross element at any N on either device (db under the new absolute scale).
3. CPU: nothing gross anywhere.
4. GPU repeats bitwise identical.
5. Wide control: **None stated in advance.**

**What ran.** Gate-open commit `a59e0f5f` in a linked worktree (machine idle by `ps`; load average 1.42 before),
`docs/superpowers/evidence/search-distillation-d2-mlxrows-run/01_wrapper.sh` once; evidence and the restored gate
`046dc41e` (main fast-forwarded; worktree and branch removed; the gate inventory test passes, all 39 gates closed;
no surviving process).
- **Outcome:** COMPLETED, inner exit 0, outer exit 0, 2.3 s in the script, 3 s wall, under the 600 s ceiling.
  `record_by: script`; stderr empty.
- **Environment** (`environment.json`): MLX 0.30.3, NumPy 2.4.1, Python 3.14.9, macOS 27.0.1, Apple M5 Max
  (`applegpu_g17s`), default device `Device(gpu, 0)`; 1-minute load average 4.15 at the script's start.
- **Evidence:** `docs/superpowers/evidence/search-distillation-d2-mlxrows/` (`environment.json`, `progress.jsonl`,
  `cases.jsonl`, `artifacts.jsonl`, `record.json`) and the run directory's `run.stdout.txt`, `run.stderr.txt`,
  `outer.json`.
- **Compact example** (outside the repository, read-only):
  `~/twixt-cache/search-distillation-d2/mlxrows-20261023T140758Z/compact_example.npz`, case `fc_32769`, output Y,
  full operands, rows 0–39, expected rows and GPU repeat 0's rows.

**Result.**
- **GPU gross cases: 12 of 27** — every layer at 32,769, 32,800, 33,600 and 33,728 rows. **None at ≤ 32,768**, for
  all three layers, including 32,737 and 32,767. **CPU: nothing gross in any case** (CPU max error on Y and dX ≤ 9.4e-6, at most 3.9e-7 of scale;
  scales 1–26).
- **Which outputs are gross:**
  - fc: Y and dX. out: dX only. wide: Y and dX.
  - **Every dW and every db is clean** on both devices; out Y is clean at every N.
- **Gross row ranges** (identical for every gross output and every repeat, except fc_33600 repeat 0):

  | N | gross rows | N − 32,736 |
  |---|---|---|
  | 32,769 | [0, 32) — 32 rows | 33 |
  | 32,800 | [0, 64) — 64 rows | 64 |
  | 33,600 | [0, 864) — 864 rows | 864 |
  | 33,728 | [0, 992) — 992 rows | 992 |

  - **fc_33600, GPU repeat 0:** 863 gross rows, Y [0, 192) + [193, 864) and dX [0, 543) + [544, 864). The largest
    error outside the gross rows in that repeat was 0.108 for Y (gross threshold 0.119) and 0.246 for dX (threshold
    0.259), against 0.010 and 0.021 in repeats 1–2. INFERENCE: rows 192 and 543 were wrong but just under the gross
    threshold; full outputs are not saved, so this is not checked.
  - **Row 32 at N = 32,769** (from the compact example, fc Y, repeat 0): error 0.0024 against a scale of 10.3, in the
    range of normal GPU rows. Rows 0–31 equal the bias exactly (max |observed − b| = 0).
- **Absolute errors:**
  - Inside the gross rows, the largest error per output was 5.5–11.2 for fc Y (scale 10–12), 13.8–21.3 for fc dX
    (scale 22–26), 3.2–4.7 for out dX (scale 1.2–1.3), 3.7–6.3 for wide Y (scale 6.7–8.1) and 3.3–6.6 for wide dX
    (scale 5.2–6.1).
  - Outside them, GPU Y and dX errors stayed at about 1e-3 of scale or below (fc Y ≤ 0.013, fc dX ≤ 0.023, out dX ≤
    0.0016, wide ≤ 0.0055), except fc_33600 repeat 0 above. The CPU's were at most 3.9e-7 of scale.
- **Repeatability:**
  - **All 15 clean cases:** the three GPU repeats are bitwise identical in every output.
  - **All 12 gross cases:** each gross output (Y and/or dX) of repeats 1 and 2 differs from repeat 0 (largest
    differences 2.5–6.5). Their dW, db and out Y are bitwise identical across repeats.
  - Repeats 1 and 2 agree with each other on every reported statistic. The record compares each repeat only with
    repeat 0, so their mutual bitwise identity is **not recorded**.
  - The number of rows outside the tolerance was the same in all three repeats of every case.
  - **The pattern in repeat 0:** at 32,769, 33,600 and 33,728, every gross Y row of fc and wide equals the bias alone
    (within 1e-2 × scale), and every gross wide dX row is near zero; at 32,800, none is near the bias. In repeats 1–2
    no gross Y row is near the bias. out dX's gross rows are never near zero.
- **Across the two executions:** the new compact example has the **same SHA-256** as the stray file from entry 28
  (`7c412fd3…`): the same case, operands, expected rows and GPU repeat-0 rows 0–39, byte for byte. The stray file
  itself was not opened; only the hash recorded in entry 28 was compared.

**Did it match the prediction, and why.**
1. **NOT MATCHED.**
   - No gross rows at N ≤ 32,768, for fc Y, fc dX and out dX: matched.
   - At 32,800 and 33,728: exactly [0, N − 32,736), matched. At 33,600: matched in repeats 1–2 and for out dX in all
     repeats; fc Y and dX in repeat 0 had 863 gross rows (above).
   - **At 32,769: 32 gross rows, [0, 32), not 33**, in every gross product and every repeat. The formula, seen at two
     sizes in entry 27, does not hold at the first size it extrapolated to.
   - A description fitted afterwards to the four sizes, **not tested and not a prediction**: the band is
     32 × ⌊(N − 32,736) / 32⌋ rows.
2. **MATCHED:** out Y, every dW and every db clean on both devices.
3. **MATCHED:** the CPU is clean everywhere.
4. **NOT MATCHED:** every gross output differs between repeat 0 and repeats 1–2, though clean outputs repeat bitwise.
   Entry 27's repeatability was measured on the fixtures' explicit products; why repeat 0 differs here is not derived.
5. **Wide control: none stated.** Observed: wide Y and dX fail at the same row bands as fc, and nothing is gross at
   ≤ 32,768.

**Conclusion / status.** COMPLETED; predictions 1 and 4 NOT MATCHED.
- **The defect reproduces on synthetic operands**, so it is not specific to the D2 fixtures' values. On this machine
  and MLX version, the leading rows of the row-indexed outputs are wrong at every probed size above 32,768 and at none
  at or below it. That includes the wide 64 → 64 control, so the defect is not limited to narrow operands. Reductions
  over the rows (dW, db) and out Y stay correct.
- **Not established:** a universal boundary, a safe chunk size, the band's rule, or the mechanism. One MLX version,
  one machine, three layer shapes, nine row counts.
- **Gross output is not repeatable**: a check for a repair must judge every repeat, not only the first.
- **Portable check for a repair:** the standalone generator with the frozen SPEC. A repaired backend should give
  COMPLETED with no GPU gross case and no non-repeatable case. Validating any fix or workaround needs its own approval.
- **Bug-report draft** `docs/superpowers/2026-10-23-mlx-matmul-rows-bug-report-draft.md`: updated with these results;
  still not submitted.
- See do-not-repeat **#76**.

**Dated corrections (2026-10-23, made with this entry).**
- **Entry 28, "none for the wide control":** the card states **"None stated in advance"** for the wide control. That
  line records the absence of a prediction, not a prediction of no gross rows. The report above is judged against
  the card's wording.
- **Entry 29, "the trainer's default cap would put B = 64 training above 32,768 rows on the GPU":** overstated. The
  576 cap **permits** batches above 32,768 rows; it does not make every batch exceed it. At B = 64 a batch exceeds
  32,768 rows only when its padded width M is greater than 512, that is, when some position in the batch has more than
  512 legal moves. The same correction is appended to the audit document (§8). #75's instruction stands.
- **Entry 29's disclosure:** the user's assessment is that the delegated width count's held-out access **exceeded the
  audit's training-only scope**. The disclosure stands. No further corpus access was needed or made for this probe.

**Dated correction to entry 30 (2026-10-23, later; the user's review).** "Clean" and "correct" were used there for
outputs that were only **free of gross errors**. Read with the frozen tolerance (|error| > 1e-4 × scale + 1e-6) kept
apart:
- **Outside the tolerance in every GPU evaluation, at every N** (27 of 27 each), gross rows or not: fc Y, fc dX,
  out dX, wide Y and wide dX, by up to about 1.2e-3 of scale outside the gross rows.
- **Within the tolerance in every GPU evaluation:** every dW, every db and out's Y.
- **The CPU:** within the tolerance everywhere.
- **Entry 27's fixtures, the same pattern:** every GPU Y and dX evaluation (fc) and dX (out) was outside the tolerance
  at every size, 512–33,728 rows; its CPU results were within it.
- "None at ≤ 32,768" in entry 30 means **no gross rows**, not results within the tolerance. The bug-report draft now
  keeps the two apart.

### 31. Search distillation D2 — full-float32 setting validation, MLX_ENABLE_TF32=0 (2026-10-23, COMPLETED; CANDIDATE for the isolated operations)

**Test.** One fresh process launched with `MLX_ENABLE_TF32=0`, same packages and application code. It ran:
- **Set A:** the unchanged synthetic probe of entry 30 (27 cases).
- **Set B:** entry 27's four original saved cases (fc and out at S0 and S21), with entry 27's three implementations.
- **Repeats:** each evaluation on the GPU three times and on the CPU once.

Card `docs/superpowers/2026-10-23-search-distillation-d2-tf32off-validation-card.md` (frozen before the run).

**Hypothesis.** MLX's documented reduced-precision float32 matmul path, which `MLX_ENABLE_TF32=0` turns off, carries
both the tolerance failures and the gross leading-row corruption of entries 27 and 30.
**Why it was plausible:**
- MLX's precision page documents the setting.
- In 0.30.3 (`mlx/backend/metal/matmul.cpp`), the setting decides between the NAX kernel and the standard fused GEMM
  for float32. It is read once per process (`mlx/utils.h`), and the installed binary contains it.
- In entry 30, the outputs within tolerance (dW, db, out Y) were exactly those on paths the setting does not gate.

**Prediction on record** (card §6, before the run):
1. The CPU's largest errors equal entries 30 and 27's, case by case.
2. GPU ungated outputs (every dW and db, and out's Y): largest errors equal to the earlier records'.
3. No GPU Y or dX row is outside the tolerance unless it holds a gross element.
4. Gross rows: none stated in advance.
5. Repeatability: none stated in advance.
6. Verdict: none stated in advance.

**Pass rule** (frozen):
- **Per set:** GPU T (evaluations outside the 1e-4 tolerance) = G (gross evaluations) = R (non-repeatable triples)
  = 0.
- **CANDIDATE:** both sets pass.
- **Not enough:** removing gross corruption alone.

**What ran:**
- **Build `ddbf5b80`:** 32 entry and 16 wrapper tests; 30 of 30 injected defects caught with `~/twixt-cache`
  unchanged; regression equal to the 58-id baseline (3,272 passed, 20 failed, 38 errors).
- **Gate 40:** opened `05248995` in a linked worktree after an idle check.
- **The run:** `docs/superpowers/evidence/search-distillation-d2-tf32off-run/01_wrapper.sh` once.
- **Evidence and the restored gate:** `5d12d5f2`. main fast-forwarded; worktree and branch removed; the gate
  inventory test passes; no surviving process.
- **Outcome:** COMPLETED, inner and outer exit 0, 3.2 s in the script (set B 0.9 s), 4 s wall. The setting read `"0"`
  at the start and the end; MLX 0.30.3, NumPy 2.4.1, Python 3.14.9, macOS 27.0.1, Apple M5 Max, default device GPU.
- **Evidence:** `docs/superpowers/evidence/search-distillation-d2-tf32off/`: `record.json`, `progress.jsonl`,
  `fixture_cases.jsonl`, and `probe/` (the probe's own record, cases, environment and progress). No compact example
  was written, since nothing was gross.

**Result.**
- **Verdict: CANDIDATE**, for the isolated operations.

  | Set | GPU evaluations | T | G | R | CPU T, G, R |
  |---|---|---|---|---|---|
  | A, synthetic | 324 | 0 | 0 | 0 | 0, 0, 0 (108) |
  | B, fixtures | 144 | 0 | 0 | 0 | 0, 0, 0 (48) |

- **Largest GPU error over scale:**
  - Y and dX: at most 3.9e-7 (synthetic) and 1.4e-7 (fixtures). In entry 30, without the setting, they were up to
    about 1.2e-3 outside the gross rows and 0.5–3.6 × scale inside them.
  - dW: at most 8.5e-6. db: at most 1e-8.
  - The tolerance is 1e-4 of scale.
- **Tolerance-only rows: 0** in every evaluation.
- **No gross rows at any size**, including 32,769–33,728 rows and the original fixture shapes (33,600 and 33,728).
- **Repeatability:** every GPU output was bitwise identical across its three repeats.
- **Comparisons with the earlier records:**
  - CPU: 108 of 108 (synthetic, entry 30) and 48 of 48 (fixtures, entry 27) largest errors equal.
  - GPU ungated: 189 of 189 and 90 of 90 equal.
  - GPU gated: 0 of 135 and 0 of 54 equal, all changed.
- **Not in the frozen criteria:** 183 of the 189 gated GPU evaluations now have exactly the CPU's largest error in the
  same case. The six that do not are `nn.Linear`'s forward Y at fc (`linear_autodiff`, S0 and S21, all repeats),
  still within the tolerance.

**Did it match the prediction, and why.**
1. **MATCHED:** the CPU is unaffected by the setting.
2. **MATCHED:** the ungated outputs are unchanged to the bit, as the source reading predicts. The setting changed only
   the gated products.
3. **MATCHED:** no tolerance-only rows.
4. Gross: none stated. Observed: none.
5. Repeatability: none stated. Observed: all bitwise repeatable.
6. Verdict: none stated. Observed: CANDIDATE.

**Conclusion / status.** COMPLETED; **CANDIDATE workaround for the isolated linear operations only.**
- With `MLX_ENABLE_TF32=0` in a fresh process, on this machine and MLX 0.30.3, these products give results within the
  frozen tolerance, with no gross rows and bitwise repeatable, in both sets.
- **Not established:**
  - **The cause.** The result is consistent with both defects lying on the path this setting disables (the NAX
    kernel, by the source reading), but no kernel was inspected.
  - **Behaviour through the policy head, the full model's gradients, or training.** That validation is the next
    step, and it needs its own approval.
  - **Behaviour on other machines, MLX versions, shapes or sizes.**
  - **Speed:** the timings are not a benchmark.
- **Not done:** no package change, no chunking, no persistent environment change, no training, no held-out access.
  The bug-report draft is updated with this result and is still not submitted. Nothing pushed.
- See do-not-repeat **#77**.

### 32. Search distillation D2 — full-float32 setting through the policy head and the full gradient path (2026-10-23, COMPLETED; FAIL at phase 2)

**Test.** One fresh process launched with `MLX_ENABLE_TF32=0`, same packages and application code, ran three
sequential checks. Each was judged on the GPU only, and a failing phase stops the later ones:
1. **Policy head:** entry 26's six saved reference cases (four synthetic; the real batch at S0 and at entry 26's S21)
   against entry 26's independent NumPy float64 reference, with MLX three times per device. Also single-record
   additivity and the directional central difference.
2. **Full model at the incumbent:** Stage 3's update-0 and update-21 batches. On each device three times, each from a
   fresh network and fresh batch arrays (digests checked): the loss terms, logits (legal moves; the padding bitwise),
   values and all 60 gradient tensors, against the CPU's first evaluation. Also half + half additivity.
3. **Updated state, only if 1 and 2 pass:** a prefix of updates 0–20 on the GPU (21 in-memory updates), then phase
   2's checks at that state.

Card `docs/superpowers/2026-10-23-search-distillation-d2-tf32off-path-validation-card.md` (frozen before the run).

**Hypothesis.** Entry 31's candidate setting also brings the policy head and the full model's gradients on the GPU
within entry 26's tolerances, at the incumbent and after 21 updates.
**Why it was plausible:**
- Entry 31: with the setting, every isolated linear product was within a 1e-4 tolerance, with no gross rows and
  bitwise repeatable.
- Entry 26 located the GPU disagreement in the encoder's forward and in fc and out, and entry 27 reproduced the fc and
  out failures in isolation.

**Prediction on record** (card §9, before the run):
1. Phase 1: the CPU's losses, and its largest differences from NumPy for every tensor, equal entry 26's in all six
   cases. NumPy's losses equal entry 26's too.
2. Phase 2: at S0 on the update-0 batch, the CPU's logits digest and policy term equal entry 26's.
3. The teacher values' digest: none stated in advance.
4. Each phase's result, and the verdict: none stated in advance.
5. The updated state's digest and the prefix's trace: none stated in advance.

**Pass rule** (frozen):
- **Tolerances:** entry 26's, unchanged. A tensor passes if max|a − b| ≤ 1e-3 × max|b| + 1e-6. Repeats must be
  bitwise equal. Finite differences use entry 26's two rules.
- **Phase 1:** every case on the GPU, against NumPy:
  - every gradient, the input-feature gradient and the loss within the tolerance;
  - repeats bitwise;
  - the summed single-record gradients within the tolerance;
  - the directional check within its tolerance wherever it ran.
- **The reference must validate itself** on the four synthetic cases. On the real batches, entry 26's directional
  check was skipped (a ReLU boundary at every ε), and a skip there is recorded, not a VOID.
- **Phases 2 and 3:** on both batches on the GPU:
  - every evaluation's loss terms, legal logits, values and 60 gradients within the tolerance of the CPU's first;
  - padded logits equal;
  - repeats bitwise;
  - half + half additivity within the tolerance.
- **Logits:** compared at legal moves only. Entry 26 compared the whole array, whose −1e9 padding set the scale, so
  that comparison could not fail.

**What ran:**
- **Build `1679b1c6`:** 40 entry and 16 wrapper tests.
  - The gate inventory has 41 gates.
  - One end-to-end test of the public entry runs in the real state (the real incumbent, corpus and fixtures, with both
    device labels on the CPU) and passes all three phases.
  - `~/twixt-cache` was unchanged.
  - The regression equals the 58-id baseline (3,272 passed, 20 failed, 38 errors).
  - Per the approval, no broad mutation campaign.
- **Corrected before the card was frozen:** my draft claimed entry 26's directional check was validated in all six
  cases. It was skipped on both real batches. The card states the actual rule.
- **Gate 41:** opened `69985900` in a linked worktree after an idle check (load about 1.5).
- **The run:** `docs/superpowers/evidence/search-distillation-d2-tf32path-run/01_wrapper.sh` once.
- **Evidence and the restored gate:** `2efeb5fd`. main fast-forwarded; worktree and branch removed; the gate
  inventory test passes; no surviving process.
- **Outcome:** COMPLETED, inner and outer exit 0, 16.6 s in the script, 17 s wall. The setting read `"0"` at the start
  and at the end; MLX 0.30.3.
- **Evidence:** `docs/superpowers/evidence/search-distillation-d2-tf32path/`: `record.json`, `progress.jsonl`,
  `reference/<case>.json` (six), `full.jsonl` and `full/S0_b0.json`, `full/S0_b21.json`. No array, model state or
  checkpoint was saved; held-out records read: 0.

**Result.**
- **Verdict: FAIL at phase 2** (full model at the incumbent). Phase 3 did not run: no optimizer update was made.
- **Phase 1: PASS.**
  - In all six cases, every GPU gradient was within the tolerance of NumPy, and the worst was 0.05 of it.
  - Largest relative difference: at most 1e-5 of scale.
  - Repeats bitwise; additivity at most 0.0084 of the tolerance.
  - The directional check passed on the four synthetic cases and was skipped on the two real batches, as in entry 26.
  - In entry 26, without the setting, the GPU disagreed with NumPy in all six cases.
- **Teacher values:** computed on the GPU in this process. Digest `abe2873a…`, not entry 26's `817aa7bf…`. An
  observation; no prediction was stated.
- **Phase 2, update-21 batch: every comparison passes.**
  - The worst gradient was 0.12 of the tolerance.
  - Logits within 5.3e-6 of scale; values within 9.2e-6.
  - Loss terms within the tolerance; padding equal; repeats bitwise; additivity at most 0.009 of the tolerance.
- **Phase 2, update-0 batch: two of 60 gradient tensors outside the tolerance,** in all three GPU evaluations:

  | Tensor | max abs difference | scale (CPU) | × tolerance | elements outside | entry 26 (no setting) |
  |---|---|---|---|---|---|
  | `encoder.blocks.1.conv2.weight` | 2.43e-4 | 0.1176 | 2.05 | 11 | 0.139 (1,169 ×) |
  | `encoder.blocks.2.conv2.weight` | 3.39e-4 | 0.2146 | 1.57 | 5 | 0.126 (581 ×) |

  - Everything else in this block is within the tolerance:
    - the other 58 tensors, with a median of 0.19 of the tolerance;
    - logits, within 3.3e-6 of scale;
    - values, within 1.25e-5;
    - the loss terms.
  - The padding is equal.
  - The GPU evaluations are bitwise repeatable, and the additivity is at most 0.005 of the tolerance.
  - The GPU's global gradient norm is 13.6991, against the CPU's 13.6988. In entry 26 it was 16.63, against 13.71.
- **The CPU's own checks** (repeats, additivity, against NumPy in phase 1) all pass.
- **The comparison base:** the failing comparison is between two MLX backends. It does not show which device is
  nearer the exact gradient: there is no independent reference for the encoder.

**Did it match the prediction, and why.**
1. **MATCHED:** the CPU's three losses and every largest difference from NumPy equal entry 26's, in all six cases.
   NumPy's losses equal too.
2. **MATCHED:** the CPU's logits digest and policy term at S0 on the update-0 batch equal entry 26's, in all three
   evaluations. As stated, the anchor and gradients differ, since the teacher values changed.
3. Teacher values: none stated. Observed: a new digest.
4. Verdict: none stated. Observed: phase 1 PASS, phase 2 FAIL.
5. Not run.

**Conclusion / status.** COMPLETED; **FAIL at phase 2 under the frozen rule.** `MLX_ENABLE_TF32=0` is **not
validated** for the full gradient path.
- **What the setting did, on this machine and MLX 0.30.3:**
  - The policy head agrees with the independent NumPy reference within the tolerance, by a wide margin.
  - At the incumbent, the full model's forward outputs and 58 of 60 gradients (all 60 on the update-21 batch) agree
    with the CPU within the tolerance, repeatably and additively.
  - Entry 26's gross and non-repeatable GPU gradients are gone in these two batches.
- **What remains:** two encoder conv2 weight gradients on one batch differ from the CPU by 1.6–2.1 × the frozen
  tolerance.
- **Not established:**
  - **Where that difference comes from,** and which backend is nearer the exact value. Nothing was measured to say.
  - **Behaviour after updates:** phase 3 never ran.
  - **Anything about training, the failed candidate, or the cause of Stage 3's value drift.**
- **No retry, and no other setting or tolerance.**
- **Not done:** no package change, no chunking, no persistent configuration change, no held-out access, no checkpoint
  or model state saved. Nothing submitted or pushed.
- See do-not-repeat **#78**.

### 33. Search distillation D2 — full-float32 setting: convolution-reference diagnostic for entry 32's two encoder gradients (2026-10-23, COMPLETED; CLASSIFIED: both INHERITED)

**Test.** One fresh process launched with `MLX_ENABLE_TF32=0`, same packages and application code, at the incumbent on
Stage 3's update-0 batch, with no optimizer update:
1. **Reproduce:** entry 32's `full_block`, unchanged, three evaluations per device. The GPU's failures must be exactly
   entry 32's six (`encoder.blocks.1.conv2.weight` and `encoder.blocks.2.conv2.weight` in each evaluation).
2. **Capture, on each device:**
   - the uninstrumented gradient, whose digest must equal step 1's;
   - the same loss with a zero probe added to each target convolution's output, so the probe's gradient is the
     upstream gradient `G`, and the convolution's input `X` is returned beside the loss. All 60 gradients must equal
     the uninstrumented ones bitwise;
   - the local reconstruction: the convolution's weight VJP from `X` and `G`, plus the separately calculated L2
     contribution, against the original gradient.
3. **Reference:** NumPy float64 weight gradients from the captured operands, after a layout check against MLX's
   forward convolution and an adjoint identity. Then each device's local gradient on **identical** operands (each
   device's set, copied to both), and the native operands separately.
4. **Classified** per target by three flags: `local_gpu` and `local_cpu` (that device's local gradient outside the
   tolerance of float64, for either operand set) and `inherited` (float64 of the GPU's operands against float64 of
   the CPU's, outside the tolerance).

Card `docs/superpowers/2026-10-23-search-distillation-d2-conv-reference-card.md` (frozen before the run).

**Hypothesis.** None was favoured. The question was whether the remaining difference is made by one backend's
convolution weight-gradient calculation itself (local), or carried into it by different operands from the rest of
the network (inherited).
**Why the check was plausible:**
- Entry 32's two misses were small (2.05× and 1.57×), repeatable and additive, on 11 and 5 of 147,456 elements.
- Every other output agreed, so a local kernel difference and a small upstream difference were both possible.

**Prediction on record** (card §9, before the run):
1. Reproduction: the teacher values' digest equals entry 32's (`abe2873a…`). On both devices, every evaluation's loss
   terms, logits digest and gradient digest equal entry 32's `full.jsonl` for `S0_b0`, and step 1 reproduces.
2. Capture fidelity: faithful on both devices, with the reconstruction **bitwise**. The basis was a CPU check on a
   random 3-block network.
3. The reference's self-checks pass.
4. Each target's flags and class, and the overall class: **None stated in advance.**

**Rule** (frozen):
- **Tolerances:**
  - comparisons use entry 26's `max|a − b| ≤ 1e-3 × max|b| + 1e-6`;
  - instrumented against uninstrumented: bitwise;
  - reconstruction: `≤ 1e-6 × max|orig|`;
  - adjoint identity: 1e-9 relative.
- **Stops:**
  - `NOT_REPRODUCED` if step 1 does not give exactly entry 32's six failures (and the CPU none of its own);
  - `CAPTURE_NOT_FAITHFUL` if any capture check fails;
  - VOID if the reference fails a self-check.
- **Entry 32's verdict is unchanged by any outcome.**

**What ran:**
- **Build `8cbfb852`,** then `6a9bb440` (entry 32's comparison recorded per key; a test expectation corrected) and
  `1a3a0517` (a second test expectation corrected).
  - 40 entry tests:
    - reference correctness: a brute-force loop, MLX's convolution on the CPU, and the adjoint check catching a
      flipped kernel and a 1e-6 scale error;
    - capture fidelity: the instrumentation adds no parameter and restores the class even on an exception, and
      omitting the L2 term or a non-zero probe is caught;
    - each stop, LOCAL_GPU and INHERITED injected on the rig, SIGTERM retention, refusals, and fresh-process
      setting checks;
    - two real-state tests: the public entry stops `NOT_REPRODUCED` when "gpu" is the CPU, and a capture plus
      reference on the real incumbent is bitwise faithful with its self-checks passing.
  - 16 wrapper tests; the gate inventory has 42 gates.
  - The regression equals the 58-id baseline (3,272 passed, 20 failed, 38 errors).
  - **A process slip:** `1a3a0517` was committed while its test was failing. The cause was the
    uncommitted-module safeguard, as before any build commit, and a pipe through `tail` hid pytest's exit code. The
    test passed once rerun after the commit.
- **Bug-report draft `1b23e19e`** (not submitted): entry 32 added, with the validated policy head kept apart from the
  unresolved encoder comparison.
- **Idle check:** the GPU's device utilization read 0% in six samples over 30 s (one earlier read showed 81%,
  transiently), and no Python process was busy. **The CPU was not idle:** macOS Spotlight indexing was running, and
  the load average was 5.6 at the start. I judged that acceptable for a GPU run of about a minute. Every reproduced
  value came out bitwise identical to entry 32's.
- **Gate 42:** opened `4c2c59a8` in a linked worktree.
- **The run:** `docs/superpowers/evidence/search-distillation-d2-convref-run/01_wrapper.sh` once.
- **Evidence and the restored gate:** `a90759b6`.
  - The gate is byte-identical to the pre-open commit, and no process survived.
  - main was fast-forwarded and the worktree and branch removed. The gate inventory test passes.
- **Outcome:** COMPLETED, inner and outer exit 0, 15.1 s in the script, 15 s wall. The setting read `"0"` at the start
  and at the end.
- **Evidence:** `docs/superpowers/evidence/search-distillation-d2-convref/`:
  - `record.json`, `progress.jsonl`;
  - `full.jsonl` and `full/S0_b0.json`;
  - `capture/{cpu,gpu}.json`, `reference/<target>.json`, `fixtures.jsonl`.
- **Fixtures** (outside the repository; the only change in `~/twixt-cache`, verified):
  - `~/twixt-cache/search-distillation-d2/convref-fixtures-20261023T200104Z/`;
  - four `.npz` of about 5.4 MB each, holding `X`, `G` and the local gradient per device and target;
  - SHA-256s in `fixtures.jsonl`, re-hashed after the run;
  - no weights, model state or checkpoint saved. Held-out records read: 0.

**Result.**
- **Step 1: reproduced, bit for bit.**
  - The teacher values' digest equals entry 32's.
  - On both devices, all three evaluations' loss terms and their gradient, logits and values digests equal entry
    32's, and `full/S0_b0.json` equals entry 32's.
  - The GPU's failures are exactly the six.
- **Step 2: faithful on both devices.**
  - The instrumented loss terms, logits, values and all 60 gradients equal the uninstrumented ones bitwise.
  - The reconstruction (local VJP plus the L2 contribution) equals the original gradient bitwise for both targets.
  - The L2 contribution is not negligible: without it, the miss is 1.1e-4 and 1.4e-4, about a thousand times the
    reconstruction tolerance.
- **Step 3: the reference passed its self-checks.**
  - Layout: at most 0.0004 of the tolerance.
  - Adjoint identity: about 1e-15 relative.

| | `blocks.1.conv2.weight` | `blocks.2.conv2.weight` |
|---|---|---|
| Identical operands: CPU kernel against float64 (CPU's set / GPU's set), × tolerance | 0.0026 / 0.0053 | 0.0026 / 0.0026 |
| Identical operands: GPU kernel against float64 | 0.0026 / 0.0053 | 0.0026 / 0.0026 |
| Identical operands: GPU kernel against CPU kernel | **bitwise equal** (both sets, all repeats) | **bitwise equal** |
| Each device's original against float64 of its own operands (with L2): CPU / GPU | 0.0026 / 0.0053 | 0.0027 / 0.0026 |
| Input `X`, GPU against CPU | 0.0037 | 0.0033 |
| **Upstream gradient `G`, GPU against CPU** | **5.15** (2 elements outside) | **5.66** (1 element) |
| **float64 of GPU operands against float64 of CPU operands** | **2.0475** (11 elements) | **1.5746** (5 elements) |
| Original GPU against CPU (entry 32's comparison) | 2.0487 (11) | 1.5741 (5) |
| Flags: local_gpu / local_cpu / inherited | no / no / **yes** | no / no / **yes** |
| **Class** | **INHERITED** | **INHERITED** |

- **Overall: INHERITED.** The kernel repeats were bitwise on both devices.
- **Where the differences sit** (observations, not a cause):
  - Every failed weight element of both targets is in **output channel 42**.
  - The largest `G` difference in both blocks is at the same position: record 1, row 13, column 4, channel 42.
    - Block 1: the CPU's `G` there is −1.08e-5 and the GPU's −2.74e-4.
    - Block 2: the CPU's is exactly −0.0 and the GPU's −2.92e-4.
- **Cancellation:** at the 16 failed elements, the sum of absolute terms from the CPU's operands is 2.5 to 40 times the
  element's float64 value, with a median of about 4.3.

**Did it match the prediction, and why.**
1. **MATCHED:** teacher values, every line on both devices and the block all equal entry 32's, and step 1 reproduced.
   The setting's GPU path repeated across processes as well as within one.
2. **MATCHED:** the capture was bitwise faithful on both devices, the reconstruction included.
3. **MATCHED:** both self-checks passed by wide margins.
4. None stated. Observed: INHERITED for both targets.

**Conclusion / status.** COMPLETED; **CLASSIFIED: both targets INHERITED.** Entry 32's FAIL at phase 2 stands.
- **Established, for these two tensors on this batch, on this machine and MLX 0.30.3:**
  - Neither backend's convolution weight-gradient calculation makes the difference. Given the same operands, the CPU
    and GPU kernels return bitwise identical results, within 0.006 of the tolerance of float64.
  - The miss is carried in by the operands, almost entirely by the upstream gradient `G`: float64 alone reproduces
    it from each device's operands, and `X` agrees closely.
- **Not established:**
  - Which device's upstream gradient, and so which **whole-model** gradient, is nearer the exact one (the card's §1
    limit).
  - Where downstream the two backends' backward passes diverge, or why. The concentration in one channel and one
    position is recorded, not explained.
  - Anything for other tensors, batches or states, for training, the failed candidate or Stage 3's value drift.
- **No retry,** and no localization further downstream without a new card.
- **Not done:** no parameter, package or persistent configuration change; no held-out access; no checkpoint or model
  state saved; nothing submitted or pushed. The bug-report draft does not yet include this entry.
- See do-not-repeat **#79**.

### 34. Search distillation D2 — full-float32 setting: encoder nonlinear-stage (branch) trace (2026-10-23, COMPLETED; BRANCH_CHANGE)

**Test.** One fresh process launched with `MLX_ENABLE_TF32=0`, at the unchanged incumbent on Stage 3's update-0 batch,
with no optimizer update:
1. **Reproduce:** on each device, the uninstrumented loss and gradient. The digests must equal entry 33's.
2. **Trace:** the encoder and its six blocks swapped, for the duration, to subclasses doing the same operations in the
   same order (`conv → BN → ReLU`, and per block `conv1 → BN1 → ReLU → conv2 → BN2 → + residual → ReLU`).
   - A zero probe sits at every conv output, BN output (= ReLU input), ReLU output and residual sum; their gradients
     are the loss's gradients there.
   - All 60 gradients and every output must be bitwise unchanged.
3. **Analysis in NumPy float64,** from each device's own captured operands:
   - every local forward and backward step: the convolutions, BN, the residual add and the ReLUs (the ReLU bitwise);
   - the ReLU masks and their flips, each quantified;
   - the focus element `[1, 13, 4, 42]`;
   - the backward walk (GPU against CPU, in backward order);
   - the accounting: a linear float64 propagation of the flips alone, from the encoder's output to the two target
     weight gradients.

Card `docs/superpowers/2026-10-23-search-distillation-d2-branch-trace-card.md` (frozen before the run).

**Hypothesis** (the user's, stated as a hypothesis, not an established defect). A ReLU takes a different branch on the
two devices at `[1, 13, 4, 42]`.
**Why it was plausible:** entry 33 put both blocks' largest upstream-gradient difference at that element, and block
2's CPU value there was exactly −0.0.

**Prediction on record** (card §9, before the run):
1. Reproduction bitwise on both devices, with the teacher values' digest equal to entry 33's.
2. The instrumentation bitwise faithful on both devices.
3. Block 2's outer ReLU flips at `[1, 13, 4, 42]`, with the CPU's mask 0 and the GPU's 1.
4. The outcome: **None stated in advance.**

**Rule** (frozen, card §7), in order:
1. `LOCAL_DERIVATIVE_ERROR` if any backward local check fails on either device.
2. `LOCAL_FORWARD_ERROR` if any forward local check fails.
3. `BRANCH_CHANGE` if there are flips in blocks 1–5 and, for both targets, the accounting is within the tolerance of
   the GPU's original gradient with the flips and outside it without.
4. `NOT_EXPLAINED` otherwise.

The tolerance is entry 26's, `1e-3 × max|b| + 1e-6`. Fidelity and the ReLU check are bitwise; the adjoint self-check
is 1e-9 relative. Entry 32's verdict is unchanged by any outcome.

**What ran:**
- **Build `2e202ece`:** 35 entry tests and 16 wrapper tests; the gate inventory has 43 gates.
  - Reference correctness: the transposed convolution and BN against a brute-force loop and MLX, and the adjoint
    check catching a flipped kernel.
  - Fidelity: no parameter added, every class restored on an exception, a perturbing trace caught.
  - On the rig: each outcome produced by an injected fault (a BN2 backward fault, a BN2 forward shift, and perturbed
    ReLU operands, which the accounting explains as a branch change); each stop, SIGTERM retention, refusals and
    fresh-process checks.
  - Two real-state tests (the public entry stops `NOT_REPRODUCED` when "gpu" is the CPU; a full trace on the real
    incumbent is bitwise faithful with every local check passing). Both run after the build commit; pytest's exit
    code was checked directly.
  - The regression equals the 58-id baseline (3,272 passed, 20 failed, 38 errors).
- **Bug-report draft `ae15f8df`** (not submitted): entry 33 added, saying explicitly that the convolution kernels
  passed on identical operands.
- **A process disclosure:** while designing the card, I checked MLX's ReLU derivative at zero with a five-element
  `mx.vjp` on the default GPU device, outside any gate and without the setting. It was trivial and was not used as
  evidence (the card's rule rests on the CPU test, and on the same convention seen on the GPU there). It was still GPU
  use outside the procedure.
- **Idle check:** the GPU's utilization read 0% in five of six samples over 30 s (one read 47%), and no Python
  process was busy. Spotlight was again using one CPU core; the load average was 2.3 at the start. As the user
  directed, Spotlight is not a reason to repeat entry 33. Entry 33's two disclosures (the Spotlight load and the
  committed failing test) stand as recorded there.
- **Gate 43:** opened `e20ae7c4` in a linked worktree.
- **The run:** `docs/superpowers/evidence/search-distillation-d2-branchtrace-run/01_wrapper.sh` once.
- **Evidence and the restored gate:** `c7b456c6`.
  - **Correction:** that commit's message says "18 ReLU flips". The record counts **14**; I mis-added the per-ReLU
    counts. The commit is left as it is, and this entry is the correction.
  - The gate is byte-identical to the pre-open commit, and no process survived.
  - main was fast-forwarded and the worktree and branch removed. The gate inventory test passes.
- **Outcome:** COMPLETED, inner and outer exit 0, 11.9 s in the script, 12 s wall. The time is observational, not a
  benchmark. The setting read `"0"` at the start and at the end.
- **Evidence:** `docs/superpowers/evidence/search-distillation-d2-branchtrace/`:
  - `record.json`, `progress.jsonl`;
  - `trace/{cpu,gpu}.json`, `stages/<relu>.json` (13);
  - `walk.json`, `focus.json`, `accounting.json`, `fixtures.jsonl`.
- **Fixtures:** `~/twixt-cache/search-distillation-d2/branchtrace-fixtures-20261023T210537Z/`, holding
  `{cpu,gpu}_record1.npz`.
  - About 12 MB each, re-hashed after the run.
  - They are the only change in `~/twixt-cache`, verified. No weights, model state or checkpoint saved; held-out
    records read: 0.

**Result.**
- **Reproduction:** bitwise on both devices. The teacher values, gradient digests, loss terms and the logits and
  values digests equal entry 33's. The targets are again 2.0487× and 1.5741×.
- **Fidelity:** on both devices, the traced loss terms, logits, values and all 60 gradients equal the uninstrumented
  ones bitwise.
- **Self-check:** the adjoint identity of the float64 transposed convolution held to 5.6e-17 absolute.
- **Every local step is correct on both devices.**
  - No forward or backward failure in the stem or any block.
  - Each ReLU's gradient equals its incoming gradient times its own mask, bitwise.
  - BN, the convolutions, the residual add and each block's input gradient are within the tolerance of float64 from
    that device's own operands.
- **14 ReLU flips** in the encoder:
  - block 0 outer: 1;
  - block 2: 1 inner and 3 outer;
  - block 4: 2 inner and 3 outer;
  - block 5: 3 inner and 1 outer;
  - none in the stem or blocks 1 and 3.
- **What every flip has in common:**
  - Its pre-activation is within 6e-6 of zero on both devices. 13 of the 14 are within 1.6e-6; the exception is
    block 5's inner ReLU at `[0, 0, 23, 33]` (CPU +2.7e-6, GPU −5.8e-6).
  - That is at most 0.21 of its stage's largest GPU−CPU pre-activation difference; 13 of the 14 are at most 0.09.
  - The float64 recomputation from each device's own operands lands on the same side of zero as that device's
    branch.
  - The incoming gradient is the same on both devices.
- **The focus element, block 2's outer ReLU at `[1, 13, 4, 42]`:**
  - Pre-activation: CPU −9.760e-7 (mask 0), GPU +9.015e-7 (mask 1). Float64 from its own operands gives −9.809e-7
    and +8.968e-7.
  - Its operands: the BN input is 0.10157866 on the CPU and 0.10158007 on the GPU; the residual is 0.0914961 on
    both, to 7e-9.
  - About 120–130 float32 ulps of the operands from zero, and 0.05 of the stage's largest pre-activation difference.
  - The incoming gradient is −2.178e-4 on both. The CPU's branch blocks it and the GPU's passes it: this is entry 33's
    upstream-gradient difference for block 2.
- **The backward walk:**
  - Everything downstream of block 4's outer ReLU is within the tolerance.
  - The first point outside it is block 4's outer ReLU, at `[52, 17, 7, 48]`: a flip (CPU −2.0e-7 off, GPU +3.4e-8
    on), with an incoming gradient of 2.1e-4.
  - Block 2's outer ReLU adds a new outside element, at the focus (`blocks.2.out` is within the tolerance and
    `blocks.2.s` is not).
  - Further up the encoder, points are outside the tolerance at 1 to 55 elements.

| Accounting | `blocks.1.conv2.weight` | `blocks.2.conv2.weight` |
|---|---|---|
| GPU original against CPU original (× tolerance) | 2.0486 | 1.5740 |
| float64 prediction **without** the flips, against GPU original | 2.0485 | 1.5747 |
| float64 prediction **with** the flips (all 14), against GPU original | **0.0059** | **0.0039** |
| predicted ΔG against actual ΔG | 0.094 | 0.085 |
| Explained | **yes** | **yes** |

**Did it match the prediction, and why.**
1. **MATCHED:** reproduction was bitwise, for the third process in a row.
2. **MATCHED:** the instrumentation was bitwise faithful.
3. **MATCHED:** block 2's outer ReLU flips at `[1, 13, 4, 42]`, CPU off and GPU on. It is not the only flip, nor the
   first point in the backward walk: that is block 4's.
4. None stated. Observed: `BRANCH_CHANGE`.

**Conclusion / status.** COMPLETED; **BRANCH_CHANGE.** Entry 32's FAIL at phase 2 stands; its criterion is not
relaxed.
- **Established, for this batch at the incumbent, on this machine and MLX 0.30.3:**
  - The two encoder gradient discrepancies are a **numerical branch change**, not an incorrect local derivative.
  - Every traced local forward and backward step is correct on both devices against float64 from their own operands.
  - 14 ReLU elements whose inputs lie within about 1e-6 of zero (at most 6e-6) take different branches. Each device's branch follows
    from its own operands.
  - Those flips alone account for both weight-gradient misses, to within 0.006 of the tolerance.
- **Not established:**
  - **That the branch change is harmless.** Small pre-activations alone do not establish it, and it moves two
    gradient tensors beyond the frozen criterion.
  - Which device's branch, or whole-model gradient, is the more accurate.
  - Where the forward operand differences of about 1e-6 come from (not traced; no recursive expansion).
  - Anything for other batches, states, training, the failed candidate or Stage 3's value drift.
- **No retry and no further trace.** The next step is the user's qualification decision from what entries 31–34
  establish. Training remains closed.
- **Not done:** no parameter, package or persistent configuration change; no held-out access; no checkpoint or model
  state saved; nothing submitted or pushed.
- See do-not-repeat **#80**.

### Decision note — full-float32 setting: qualification after entries 31–34 (2026-10-23, the user's decision)

**Not an experiment;** nothing ran. This records the user's decision on what entries 31–34 establish.

- **Entry 34's BRANCH_CHANGE accepted.** The two residual encoder discrepancies are explained, and further localization
  is not warranted. **Entry 32 remains a FAIL under its original rule.**
- **The decision:**
  - `MLX_ENABLE_TF32=0` is **required** for subsequent authorized GPU work on this machine and runtime (M5 Max, MLX
    0.30.3). It is set per process, as before; persistent environment changes remain unauthorized.
  - **Validated within the tested scope:** the isolated operations (entry 31), the policy head (entry 32, phase 1)
    and the incumbent-state local derivatives (entry 34).
  - **Training stays unqualified.** The updated-state check (entry 32's phase 3) never ran. Neither the branch
    change's harmlessness nor general training reliability is established.
- **A wording correction to my report of entry 34.** I wrote that a ReLU input near zero *will* make a CPU/GPU
  gradient comparison fail. The correct statement is that it **can**. That wording was in the report only, not in this
  ledger.
- **Authorized and done: a design-only card** for completing the 21-update diagnostic prefix:
  `docs/superpowers/2026-10-23-search-distillation-d2-prefix-qualification-card.md`, commit `8b255801`.
  - **Status:** DRAFT for review. Not frozen and not built, and no gate exists for it. Every gate stays closed.
  - **Its prospective criterion,** at the state after updates 0–20 (U21), on Stage 3's update-0 and update-21 batches:
    - finite results;
    - repeatability: two prefixes bitwise equal, evaluations bitwise equal;
    - additivity;
    - forward agreement on legal logits and pre-tanh values;
    - independent float64 checks of every local derivative and all 60 parameter gradients;
    - for any native gradient comparison that fails, entry 34's explicit branch accounting. Entry 26's tolerance is
      unchanged.
  - Anything unexplained stops qualification, with no automatic follow-up.
  - The prefix stays diagnostic: training records only, no candidate checkpoint, no strength claim.
  - Three questions are open for review in its §10.
- **Bug-report draft `bf6f5daa`** (not submitted): entry 34 added. It keeps the gross matmul defect apart from the
  explained branch change, and names no cause for the forward differences.
- **Not done:** nothing submitted or pushed; no persistent environment change; no run.
- See do-not-repeat **#81**.

**Addendum to the decision note (2026-10-23, later): the prefix card reviewed, frozen and built; not run.**
- **The user's review answers,** written into the card (§10):
  - two prefixes (42 diagnostic updates), each from a fresh incumbent and a fresh Adam, compared update by update on
    the scalars, the model digest and the Adam-state digest;
  - "raw values" are the tanh'd predictions the loss uses: the acceptance comparison, as in entry 32. Pre-tanh values
    are supplementary, and their capture must be bitwise faithful;
  - unexplained failures stop qualification, and the accounting is not extended. Internal encoder flips cannot excuse
    a discrepancy entering from the encoder-output gradient.
- **Two requirements the user added:**
  - every encoder parameter reference gates, L2 included;
  - the native policy-head gradients are validated per device from that device's own encoder output, apart from the
    common-input check.
- **Build `ad873a12`:** card frozen (sha `cdd900c2…`), `scripts/GPU/alphazero/search_distillation_d2_prefix.py`, gate
  44 (`SEARCH_DISTILLATION_D2_PREFIX_AUTHORIZED`, closed), the wrapper `search-distillation-d2-prefix-run/01_wrapper.sh`,
  39 entry and 16 wrapper tests.
  - The prefix loop's final state equals `gradcheck.reconstruct`'s bitwise. A changed Adam is caught by the digests at
    update 0, where that update's scalars agree.
  - On the rig, each criterion is produced by an injected fault, among them:
    - a weight-only gradient fault caught only by the parameter reference;
    - a policy-head fault caught by the native check and not by the common-input one;
    - an encoder-output gradient fault, unexplained;
    - near-zero branch flips, explained, which qualify;
    - flips large enough to show in the forward, a local forward error;
    - a whole-forward perturbation, unexplained despite its flips.
  - **Real state, both labels on the CPU:**
    - the public entry stops `NOT_REPRODUCED`, as it must when "gpu" is the CPU;
    - a full run with entry 32's digests replaced in the test only is `QUALIFIED_AT_U21`: the two 21-update prefixes
      bitwise equal, all 52 parameter references and the policy-head references passing on the real operands, with
      nothing flipping.
    - In that run the parameter references were at most 0.015 of the tolerance and the policy head at most 0.093
      (`out.bias`). It says nothing about the GPU.
  - The regression equals the 58-id baseline (3,272 passed, 20 failed, 38 errors). The new tests' exit code was checked
    directly.
- **Card wording settled during the build:**
  - the checks' running order: prefixes, then native evaluations, then batch by batch, judged in number order at each
    point;
  - a non-finite evaluation at the incumbent is `NOT_REPRODUCED`, not a VOID.
- **Not done:** no run, no GPU use, the gate never opened. Execution needs the user's approval after review of the
  frozen implementation. Nothing submitted or pushed.

### 35. Search distillation D2 — full-float32 setting: the 21-update diagnostic prefix (2026-10-23, COMPLETED; QUALIFIED_AT_U21)

**Test.** The frozen prefix card (`docs/superpowers/2026-10-23-search-distillation-d2-prefix-qualification-card.md`,
sha `cdd900c2…`), run once in a fresh process launched with `MLX_ENABLE_TF32=0`:
1. **Reproduce entry 32 at the incumbent:** the teacher-value digest; on each device, both batches' terms and digests.
2. **Two prefixes on the GPU, P1 and P2,** of Stage 3's updates 0–20. Each starts from a fresh incumbent and a fresh
   Adam, with identical batches and teacher targets: 42 in-memory updates in all. After every update the scalars, the
   model digest and the Adam-state digest are compared.
3. **At the updated state U21,** on Stage 3's update-0 and update-21 batches: finite, repeatable, additive, and forward
   agreement (loss terms, legal logits, tanh'd values) against the CPU.
4. **Batch by batch:**
   - trace and pre-tanh capture fidelity;
   - float64 local checks of every stage;
   - all 52 encoder parameter references with L2;
   - the native policy-head gradients (8) from each device's own encoder output;
   - the common-input policy-head reference;
   - ReLU flips and the encoder-output gradient;
   - entry 34's branch accounting for any failing native GPU gradient.

   Entry 26's tolerance is unchanged. Anything unexplained stops the run.

**Hypothesis.** Under the full-float32 setting, the GPU's updated state after 21 updates is reproducible and its
gradient path is correct or explained by branch changes, as at the incumbent (entries 32–34).

**Why it was plausible:** at the incumbent, every local step was correct on both devices, and the only native
failures were explained by near-zero ReLU branch changes (entry 34). Entry 26's gross, non-repeatable GPU gradients
were gone under the setting (entry 32).

**Prediction on record** (card §8, before freezing):
1. §3 reproduces bitwise, and the teacher values' digest equals entry 32's.
2. Everything else, the outcome included: None stated in advance.

**What ran:**
- Build `ad873a12`. The gate was opened by `d82d2a18` on a run branch in a separate worktree, and run once under the
  wrapper with its 600-second ceiling, on an idle machine.
- The opening commit was first made on main. I moved it to the run branch (main reset to `984beada`, not pushed)
  before anything ran, because the wrapper needs a clean tree and main holds two untracked files.
- Evidence and the gate's restoration are in `0e11ef10`, fast-forwarded onto main; the worktree and branch are
  removed.
- The wrapper exited 0 (COMPLETED) in 35 s. The record was written by the script, the gate read back equal to the
  pre-open commit, and no run process survived.
- No model state, arrays or checkpoint were saved. 0 held-out records were read.

**Result.** Recorded outcome **QUALIFIED_AT_U21**.
- **Reproduction:**
  - the teacher digest equals entry 32's (`abe2873a…`);
  - all four device/batch lines equal entry 32's.
- **Prefixes:**
  - P1 and P2 are equal at every one of the 21 updates: scalars, batch, model digest and Adam digest;
  - all values are finite;
  - the U21 digests are equal (`37ad56e1…`).
  - Stage 3's logged scalars, recorded beside them as an observation (card: the setting changes the path), differ at
    every update. At update 0 the loss is 4.6560 against 4.6693, and the gradient norm 13.70 against 12.45. No cause is
    derived here.
- **At U21:**
  - repeatable bitwise on both devices and batches;
  - all 60 additivity comparisons within tolerance;
  - padded logits equal.
- **GPU against the CPU, in tolerance units:**

  | Check | Batch 0 | Batch 21 |
  |---|---|---|
  | Legal logits | 0.003 | 0.004 |
  | Tanh'd values | 0.008 | 0.013 |
  | Pre-tanh values (supplementary) | 0.002 | 0.006 |
  | Encoder-output gradient | 0.001 | 0.002 |

- **Native GPU gradients:**
  - Batch 0: 2 of 60 tensors fail, in all three evaluations: `encoder.blocks.3.conv1.weight` at 1.49× and
    `.conv2.weight` at 1.24× the tolerance.
  - Batch 21: none fail; the worst is `encoder.blocks.1.conv1.weight` at 0.66×.
- **Fidelity:** trace and pre-tanh captures were bitwise faithful on both devices and batches (terms, logits, values,
  gradients).
- **Local checks,** both devices and batches: no forward or backward failure. Forward is at most 0.006× and backward at
  most 0.0007× the tolerance.
- **References, worst in tolerance units:**
  - all 52 parameter references with L2 pass on each device, at most 0.0095×;
  - the native policy head's 8 gradients pass, at most 0.037× (`out.bias`, batch 21, CPU);
  - the common-input reference passes, at most 0.044×. Its directional check was skipped on the real batches, as in the
    build.
- **ReLU flips:** 5 on batch 0 (blocks.1 outer 1, blocks.3 outer 1, blocks.5 outer 3) and 13 on batch 21.
- **Branch accounting,** batch 0: both failing tensors are explained.
  - Flips on their path: blocks.3 outer 1 and blocks.5 outer 3.
  - With the flips, conv1 is at 0.006× and conv2 at 0.004× the tolerance.
  - Without the flips, conv1 is at 1.49× (3 elements outside) and conv2 at 1.24× (5 outside).
  - The encoder-output gradient is within tolerance.
- **Run time:** 34.9 s inside the script.

**Did it match the prediction, and why.** Prediction 1: **matched**. The teacher digest and all four incumbent lines
equal entry 32's, as the same frozen data, code and setting should give. Prediction 2: None stated in advance.

**Conclusion / status.** COMPLETED; **QUALIFIED_AT_U21** under the card's prospective criterion.
- **Scope:** this updated state, after 21 updates from the incumbent, and these two batches only.
- **What it shows:** the GPU prefix was bitwise repeatable, and its U21 gradient path was locally correct. Its only
  native discrepancies were the two batch-0 tensors, each accounted for by near-zero branch changes as entry 34 found at
  the incumbent.
- **What it does not establish** (card §9):
  - that training on the GPU is reliable, or that any branch change is harmless;
  - which device is more accurate;
  - anything about other states, batches, more updates, the failed candidate, Stage 3's value drift, or another MLX
    version or machine.
- **Unchanged:** entry 32 remains a FAIL; training remains unqualified.
- **Next:** the user's decision. Nothing was submitted or pushed. See do-not-repeat **#82**.

### Decision note — after entry 35: a bounded learning trial under the corrected configuration (2026-10-23, the user's decision)

- **Accepted:** entry 35's `QUALIFIED_AT_U21`. In the user's words, the numerical evidence is enough to plan a bounded
  learning trial; further diagnostics are not the next task.
- **What the acceptance does not establish:** general GPU-training reliability, or an explanation of the old
  candidate's value drift.
- **Authorized:** a short, design-only successor card. Its terms:
  - unchanged `calib020_0001` with process-local `MLX_ENABLE_TF32=0`;
  - the 108 immutable training games reused; the original corpus unchanged and its 12 held-out games spent;
  - exactly 12 fresh, separately stored held-out games, with fresh seed accounting and no top-ups;
  - Stage 3's 401-update recipe fixed: anchor, learning rate and update surface unchanged, and no tuning alongside the
    numerical change;
  - Stage 3's held-out policy-improvement and value-drift criteria retained;
  - at most one candidate. A failure closes the trial, with no retry or selection;
  - the numerical configuration stated for generation, training and evaluation, including the provenance difference
    from the existing labels;
  - compute ceilings and stop conditions frozen, and the fresh strength evaluation outlined. Lower loss alone is not
    enough.
- **Not authorized:** generation, seed registration, training, gate opening and pushing.
- **Drafted:** `docs/superpowers/2026-10-23-search-distillation-d2-corrected-trial-card.md` (DRAFT, not frozen; no
  gate), awaiting review.
- **Prediction on record:** None stated in advance for this decision. The card proposes predictions (§9) for review.
- See do-not-repeat **#83**.

**Addendum to the decision note (2026-10-24): the trial card reviewed, frozen and built; seeds registered; nothing generated or trained.**
- **The user's review decisions,** written into the card (§11):
  - held-out labels generated with `MLX_ENABLE_TF32=0`;
  - the 2-hour and 30-minute ceilings approved, as estimates with no extension or top-up after a timeout;
  - "None stated in advance" for the outcome. The draft's proposed figures are withdrawn and are not a prediction on
    record.
- **Wording fixes,** in the card:
  - zero original held-out files and records read is recorded apart from access to the new set, and the new set is
    scored only for the baseline and the final check;
  - the old processes' setting is unknown, not asserted;
  - the row counts are flattened policy-head rows;
  - the strength threshold's exact formula belongs to its future card.
- **Build `d7d458ad`:** card frozen (sha `9f2c6096…`).
  - `search_distillation_d2_heldout` (gate 45, closed): a copy of Stage 2's loop for one phase of 12 held-out games.
  - `search_distillation_d2_trial` (gate 46, closed): a copy of Stage 3's loop, with gradtrace's training-only reader,
    admission of the new set only from a COMPLETED generation record under the card, the scorings ledger and the spike
    count.
  - Both refuse without the setting before MLX is imported, check the MLX, NumPy and Python versions, and pin the card
    and the reused modules.
  - Two wrappers (copies of Stage 2's and Stage 3's with the setting added), the 12-slot survey script (a copy of
    Stage 2's) and focused tests: 33 + 53 + 16 + 17, plus the gate inventory at 46.
  - Real-state tests, scoped:
    - the real corpus through the training-only reader: 6,403 records, 0 original held-out files opened;
    - the real incumbent loads twice as the pinned teacher;
    - the public entry refuses until a generated, committed held-out set exists;
    - the registered plan is read through the entry's own seams.
  - **No test trains the real incumbent.** A full real-state run would be the trial itself, about 6–7 minutes on the
    CPU.
- **Verification:**
  - the regression equals the 58-id baseline (3,272 passed, 20 failed, 38 errors), with the fingerprint unchanged;
  - the focused tests and their neighbours pass (198, exit 0), on the committed build.
- **Seeds:** in the method of Stage 2, in a worktree on a clean committed tree, each step run once:
  - S1, survey (`21d703ea`): THE D2 HELD-OUT BLOCK IS CLEAN, [202692300, 202692312), 45 of 45 controls, gap 74 against
    a floor of 12;
  - S2, pre-registration proof (`ede8faed`): CLEAN, the state HOLDS, 45 of 45;
  - S3, registration (`926aed49`): appended last to ACCOUNTED (41 → 42, the prefix identical), plus `plan.json`.
    Registry tests 48 passed; registry readers 593 passed;
  - S4, post-registration proof (`62f93fe3`): CLEAN, the state HOLDS, 45 of 45.

  The block is not exposed and not retired.
- **Not done:** no generation, training, GPU use or gate opening. Nothing submitted or pushed. Generation needs the
  user's approval after review of the frozen implementation, plan and proof records.

### 36. Search distillation D2 — bounded trial: the 12-game held-out generation (2026-10-25, INTERRUPTED; INCOMPLETE; CLOSED with no learning result)

- **Test:** the bounded trial's step 2 (card `docs/superpowers/2026-10-23-search-distillation-d2-corrected-trial-card.md`,
  sha `9f2c6096…`, §7). One invocation of the held-out generation wrapper: 12 fresh held-out games under
  `MLX_ENABLE_TF32=0`, from the frozen build `d7d458ad` and the registered plan [202692300, 202692312), under the
  2-hour ceiling.
- **Hypothesis:** none was being tested by this step itself. It only produces the new held-out set that the trial's
  training step (gate 46) is scored on. The trial's learning hypothesis is in the decision note after entry 35.
- **Why it was plausible:** Stage 2 (entry 22) generated 120 games with the same loop and finished cleanly.
- **Prediction on record:** None stated in advance (card §10).
- **What ran:**
  - `b887dd04`: the seed-usage derivation for closing, fixed before the run. It is a copy of Stage 2's; synthetic
    controls 15/15. The build had not included one, so it was added before the gate opened.
  - `98b4a031`: gate 45 opened by a one-line commit on a linked-worktree branch. Main's copy was never opened.
  - The wrapper started at 09:36 on 2026-10-25. `progress.jsonl` records `start` with the setting `"0"`, then
    `loaded`, `game_start` 0, `game_end` 0 and `game_start` 1.
  - **At 09:39:35 a macOS kernel panic rebooted the machine.** The report is
    `/Library/Logs/DiagnosticReports/panic-full-2026-10-25-093935.0002.panic`. The panicked task was `kernel_task`; the
    panic string begins "PMC0 Assert ID: 0x4c (PMC_ASSERT_TIMER_TRANSITION_LATENCY_TIMEOUT), rail:DCS".
- **Result:**
  - Two games started (slots 0 and 1); only slot 0 finished.
  - **The wrapper's exit trap never ran.** So there is no `outer.json`, `record.json`, `heldout.json` manifest or
    `finalized.json`, and none was synthesized. The runtime exit status is **unknown**.
  - After the reboot no run process remained. This was observed after the reboot, not verified by the wrapper.
  - `f4a10b89`: gate 45 restored by hand with the wrapper's own edit and read back equal to the pre-open commit. The
    partial evidence (`progress.jsonl`, the empty stdout and stderr) committed with a hand-written
    `02_interruption_note.txt`, which states that it is not a wrapper or script record. The worktree and branch were
    removed; all 46 gates are closed.
  - The partial capture is left unchanged in
    `~/twixt-cache/search-distillation-d2/heldout-tf32off-20261025T133651Z/` (`context.json` and `game_0.json`, with
    their sha256 in the note). **It is ineligible for training or scoring.**
- **Seed closing** (the user's option A, approved 2026-10-25):
  - The frozen derivation `01_*` needs `outer.json`, so it was left unchanged and not run.
  - `b588475a`: a dated amended derivation, `03_derive_usage_reboot.py`. It handles a reboot interruption explicitly:
    the exit status is unknown, and process absence is the post-reboot observation. It is bound by sha256 to the
    committed plan, `progress.jsonl` and the note. It requires `outer.json`, `record.json`, `heldout.json` and
    `finalized.json` all absent, and it reads event names and slots only.
  - `dae64eef`, K1: the amended derivation ran once and exited 0. It RECONCILES, with 16 of 16 controls passing:
    expose [202692300, 202692302) and retire [202692300, 202692312). The record is create-only.
  - `164774e2`, K2: those two intervals were appended last to EXPOSED (39 → 40) and RETIRED (37 → 38). ACCOUNTED is
    unchanged.
    - Registry tests: 48 passed. Registry readers: 593 passed.
    - Two real-state D2 tests were moved to the closed state; the D2 tests pass, 86 of 86.
- **Did it match the prediction?** There was none.
- **Conclusion / status: CLOSED — interrupted and incomplete, with no learning result.**
  - Under the user's rule (an incomplete set closes the trial), the trial ends here. Gate 46 was never opened; nothing
    was trained, scored or evaluated.
  - This entry **does not establish a cause for the panic**. It neither attributes the panic to the workload nor
    clears the workload.
  - It **does not reject the learning hypothesis**, which was never tested.
  - `calib020_0001` stays the incumbent (SHA-1 `209cf2d4…`, unchanged after the reboot).
  - See do-not-repeat **#84**.

### Decision note — after entry 36: one fresh attempt at the bounded trial (2026-10-25, the user's decision)

- **Decided:** exactly one fresh attempt is justified. The reboot prevented the learning test and gave no evidence for
  or against the recipe.
- **Terms:**
  - preserve the closed trial and its partial capture unchanged;
  - 12 fresh seeds and a separate held-out set;
  - the training corpus, recipe, `MLX_ENABLE_TF32=0`, acceptance criteria and compute ceilings unchanged;
  - a short successor card and fresh seed accounting, reusing the implementation with minimal changes for the new
    records and paths;
  - before scheduling, a read-only check for further kernel panics. No stress test or system change.
- **If another panic interrupts attempt 2:** further GPU experiments stop pending a system-stability decision, with no
  automatic third attempt.
- **Not to be done:** turn the interruption into another learning-method investigation, or tune the recipe before it
  has had a completed test.
- **Authorized:** preparation only. Generation and training need approval of the frozen plan.
- **Prediction on record:** None stated in advance.
- **Preparation done:**
  - **Panic check (read-only):** no panic since the 09:39 one in entry 36, and `last reboot` shows only that reboot
    since 2026-10-20. Correction to entry 36's citation: the report it cites
    (`/Library/Logs/DiagnosticReports/panic-full-2026-10-25-093935.0002.panic`) is no longer in that directory. Only
    the system's `.contents.panic` from 09:39 remains there.
  - **Card:** `docs/superpowers/2026-10-25-search-distillation-d2-corrected-trial-attempt2-card.md`, FROZEN (sha
    `bde310a3…`). The original card (`9f2c6096…`) stays the authority for everything it does not change. The new card
    changes only the records and paths. It adds the stability rule and states "None stated in advance".
  - **Build (`e3f90078`):**
    - both entries pin the new card;
    - the held-out evidence, plan and capture paths are renamed `attempt2`;
    - copies of attempt 1's wrapper (paths and header), seed survey (paths, titles, last-EXPOSED label) and closing
      derivation (paths and titles; synthetic controls 15/15);
    - gates 45 and 46 are reused, closed; the training entry's paths and wrapper are unchanged and were never used.
    - The training entry admits only a generation record whose card hash is the new card's, so attempt 1's records
      cannot be admitted.
    - The tests were retargeted. The focused D2 tests, the gate inventory and the tactical closing test pass on the
      committed build (163).
  - **Regression:** 59 ids, i.e. the 58-id baseline plus
    `tests/test_tactical_seed_closing.py::test_the_pilot_block_is_ACCOUNTED_EXPOSED_and_RETIRED_exactly_as_itself_and_LAST`.
    The fingerprint is unchanged.
    - That test broke at attempt 1's K2 (`164774e2`). It required every later EXPOSED entry to equal its RETIRED
      entry, and K2 exposed only the started prefix.
    - **Correction to entry 36:** its "registry checks" were the registry tests and the registry readers only. The
      regression was not run after K2.
    - `c96e205b` re-anchors the test: each later EXPOSED entry must start where its RETIRED entry does and end no
      later. Two mutants are caught, and the file passes 36 of 36. The full regression was not rerun after this
      re-anchor.
  - **Seeds:** in a clean linked worktree, each step run once, by attempt 1's method:
    - S1, survey (`0b195c4c`): THE D2 HELD-OUT BLOCK IS CLEAN, [202692400, 202692412), 45 of 45 controls, gap 82
      against a floor of 12;
    - S2, pre-registration proof (`45b836c1`): CLEAN, the state HOLDS, 45 of 45;
    - S3, registration (`7777b019`): appended last to ACCOUNTED (42 → 43), with `plan.json`. Registry tests 48
      passed; registry readers 593 passed; tactical closing 36 passed;
    - S4, post-registration proof (`0784a7be`): CLEAN, the state HOLDS, 45 of 45;
    - `cadebaf3`: a real-state test of the registered plan through the entry's own seams.

    The block is not exposed and not retired.
  - **Attempt 1 untouched:** its tracked evidence is unchanged since `bb0ff9b6`, and its partial capture's sha256
    still match the note.
- **Not done:** no generation, training, GPU use or gate opening. Nothing submitted or pushed.
- See do-not-repeat **#85**.

### 37. Search distillation D2 — bounded trial attempt 2: the 12-game held-out generation (2026-10-25, COMPLETED; ELIGIBLE)

- **Test:** attempt 2's step 2 (successor card
  `docs/superpowers/2026-10-25-search-distillation-d2-corrected-trial-attempt2-card.md`, sha `bde310a3…`; the original
  card `9f2c6096…` §4 governs). One invocation of the attempt 2 held-out wrapper: 12 fresh held-out games under
  `MLX_ENABLE_TF32=0`, from build `e3f90078` and the registered plan [202692400, 202692412), under the 2-hour ceiling.
- **Hypothesis:** none was being tested by this step itself. It produces the held-out set that the trial's training
  step (gate 46) is scored on. The learning hypothesis is in the decision note after entry 35.
- **Why it was plausible:** Stage 2 (entry 22) generated 120 games with the same loop and finished cleanly. Attempt 1
  (entry 36) was cut off by a kernel panic, with no evidence about the loop itself.
- **Prediction on record:** None stated in advance (attempt 2 card §6).
- **Approval:** the user approved the single generation run on 2026-10-25 (gate 45 only), and accepted the targeted
  re-anchor `c96e205b` with the post-fix full-suite status recorded as unverified at that point.
- **Before scheduling (read-only):** no new kernel panic. `/Library/Logs/DiagnosticReports/` held only the 09:39
  `.contents.panic`, and the reports newer than it were ordinary diag and cpu_resource files. `last reboot` showed no
  reboot after 09:39. The machine was idle: 1-minute load 1.6, top process 12% CPU, no python job other than the
  telemetry logger running since boot.
- **What ran:**
  - `1b55c1f0`: gate 45 opened by a one-line commit on a linked-worktree branch. Main's copy was never opened.
  - The wrapper ran from 17:44:28 to 18:02:34 (EDT).
- **Result:**
  - `outer.json`: inner exit 0, COMPLETED, outer exit 0, gate restored and read back equal to the pre-open commit
    `e38edbfd`, no survivors, record written by the script, 1,086 s wall against TERM at 7,190 s.
  - `record.json`: status COMPLETED, `eligible: true`, `card_sha256` the attempt 2 card's. 12 games, 660 plies and
    660 records, 4,793,599 bytes. The seeds are exactly [202692400, 202692412) in slot order, all held out.
  - **Alignment:** the run's VOID checks passed, so the 660 rebuilt records equal the learner rows (board, side to
    move, legal moves, visits, outcome) and the reader count matches.
  - **Model immutability:** model digest `ae0e71fc…` before and after; checkpoint SHA-1 `209cf2d4…`; all 42 modules
    (14 BatchNorm) in eval mode at load and at end.
  - **Setting:** `MLX_ENABLE_TF32` = `"0"` at start and end. Versions mlx 0.30.3, numpy 2.4.1, python 3.14.9.
  - **Capture:** `~/twixt-cache/search-distillation-d2/heldout-tf32off-attempt2-20261025T214428Z/`, with
    `context.json` `7d153267…` and `finalized.json` `4905dcbb…` matching the record. The manifest `heldout.json` is
    `da96b9e9…`, and the evidence copies of the three phase files are byte-identical to the capture.
  - `e358b3d7`: the evidence and the restored gate committed.
  - No game result was read for this entry; the held-out games are scoring data for the training step.
- **Seed closing:**
  - `3a68abb4`, K1: the frozen derivation `01_derive_usage.py` ran once and exited 0. It RECONCILES, with 15 of 15
    controls passing: 12 starts, so expose [202692400, 202692412) and retire [202692400, 202692412). The record is
    create-only.
  - `a6102976`, K2: the block appended last to EXPOSED (40 → 41) and RETIRED (38 → 39); AST check, prefixes
    identical; ACCOUNTED unchanged.
    - Registry tests: 48 passed. Registry readers: 593 passed. Gate inventory: 8 passed.
    - Three real-state D2 tests moved to the new state: the attempt 2 plan's seeds are now all refused; with the gate
      open, the trial entry admits the committed attempt 2 set and reaches the autouse stub boundary once (in process
      and in a fresh process; nothing trained). D2 held-out, trial, wrapper and seed-closing tests: 139 passed.
  - **Full regression at `a6102976`, in the main checkout:** exactly the 58-id baseline (3,272 passed, 20 failed, 38
    errors, 1,334 s). This also verifies the `c96e205b` re-anchor; the decision note's "not rerun" is now resolved.
    - A first run in the linked worktree gave 127 ids. The 69 extra were ERRORs in
      `tests/test_e4_screen_command.py` and `tests/test_l0_match_runner.py`, from a relative checkpoint path that does
      not exist in a worktree (`checkpoints/` lives only in the main checkout). That run is not a valid comparison and
      is superseded by the main-checkout run.
- **After the run (read-only):** no new panic report and no reboot since 09:39.
- **Attempt 1 untouched:** its tracked evidence is unchanged since `bb0ff9b6`, and its partial capture's sha256 still
  match the note.
- **Did it match the prediction?** There was none.
- **Conclusion / status: COMPLETED — the attempt 2 held-out set is eligible.**
  - Gate 45 was used once and is closed; the block is spent.
  - Gate 46 was never opened; nothing was trained, scored or evaluated. Training needs the user's approval, a fresh
    panic check and an idle machine.
  - This is not a learning result, and it says nothing about the cause of entry 36's panic.
  - `calib020_0001` stays the incumbent (SHA-1 `209cf2d4…`).
  - See do-not-repeat **#86**.

### 38. Search distillation D2 — bounded trial attempt 2: the one training run (2026-10-25, COMPLETED; STOP; the trial CLOSED)

- **Test:** attempt 2's step 4 (successor card `bde310a3…`; the original card `9f2c6096…` §5 governs). One invocation of
  the training wrapper: Stage 3's recipe unchanged (401 updates, batch 64, Adam 1e-4, clip 1.0, anchor 1.0, L2 1e-4,
  draw-order seed 0, encoder and policy head) under process-local `MLX_ENABLE_TF32=0`, on the 108 immutable training
  games, scored on attempt 2's new held-out set at baseline and final only, under the 30-minute ceiling.
- **Hypothesis (the card's question, §1):** the fixed D2 recipe, unchanged, can produce a candidate that passes Stage
  3's held-out checks when every process runs with `MLX_ENABLE_TF32=0`.
- **Why it was plausible:** entries 31–35 traced Stage 3's numerical faults to the GPU's reduced-precision path, and
  qualified the corrected setting within their tested scope (entry 35, `QUALIFIED_AT_U21`). Whether those faults
  caused Stage 3's value drift was never established.
- **Prediction on record:** None stated in advance (original card §10; attempt 2 card §6).
- **Approval:** the user approved the single training run on 2026-10-25 (gate 46 only), after reviewing entry 37's
  record, manifest, game hashes, receipts, metadata copies, seed closure, versions, checkpoint hash and code pins.
- **Before scheduling (read-only):** no new kernel panic and no reboot since 09:39. The machine was first busy
  (Spotlight's `corespotlightd` at about 295% CPU, 1-minute load 9.5); the run waited until the 1-minute load stayed
  under 3, with no process above 50% CPU, for a minute (20:05; load 1.9).
- **What ran:**
  - `3290a05d`: gate 46 opened by a one-line commit on a linked-worktree branch. Main's copy was never opened; gate 45
    stayed closed.
  - The wrapper ran from 20:05:37 to 20:06:24 (EDT).
- **Result:**
  - `outer.json`: inner exit 6, outcome STOP, outer exit 6, gate restored and read back equal to the pre-open commit
    `83636fb5`, no survivors, record written by the script, 46 s wall against TERM at 1,790 s. Stderr empty.
  - `record.json`: status COMPLETED, **verdict STOP**, `card_sha256` the attempt 2 card's, strength verdict none.
  - **Inputs:** 6,403 training records from the 108 training games (ids sha256 `e90df4d3…`); the original 12 held-out
    games' files opened 0, records read 0. The new set: 660 records, manifest `da96b9e9…`, record `d42de735…`, scored
    `baseline` and `final` only.
  - **Training checks:** 401 updates in 37.0 s on the GPU; all 60 surface tensors changed, the 32 off-surface tensors
    unchanged; teacher digest `ae0e71fc…` before and after; every metric finite. `MLX_ENABLE_TF32` = `"0"` at start
    and end; mlx 0.30.3; Adam's resolved defaults recorded (betas 0.9, 0.999; eps 1e-8; no bias correction).
  - **Held-out policy cross-entropy:** 4.7878 for the teacher, 4.1627 for the candidate (KL to the targets 2.0196 →
    1.3945): **strictly lower, met**.
  - **Held-out absolute value drift from the teacher: mean 0.1207 (limit 0.05), p95 0.3433 (limit 0.15)**, max 0.6339.
    **Both limits failed.**
  - **Training-set diagnostics** (judging nothing): cross-entropy 4.6965 → 3.9370; drift mean 0.0636, p95 0.1804, max
    0.4060.
  - **Training log:** every update was clipped (pre-clip gradient norm above 1 on all 401; median 3.38, maximum 13.70
    at update 0). **None exceeded the 10,000 spike norm** (Stage 3, entry 23: 86 of 401, maximum 43,779). The anchor
    term's batch mean fell from 0.036 over the first 50 updates to 0.009 over the last 50.
  - **Candidate:** saved create-only at
    `~/twixt-cache/search-distillation-d2/student-tf32off-20261026T000538Z.safetensors`, SHA-256 `71b76698…`; the
    strict reload reproduced its state digest `48565952…` with all 42 modules (14 BatchNorm) in eval mode. Made
    read-only after the run, as Stage 3's was. A research artifact only.
  - `bf0d7064`: the evidence and the restored gate committed. No game seeds were drawn.
- **After the run (read-only):** no new panic report and no reboot since 09:39; no run process left. Both gates closed;
  gate inventory and D2 tests 95 passed.
- **Did it match the prediction?** There was none. Compared with Stage 3 (entry 23, default setting, a different
  held-out set): the policy gain is similar (CE 4.8243 → 4.1474 there), the value drift is no smaller (mean 0.1026, p95
  0.2791 there), and the gradient-norm spikes are absent. Why the value output still moved this far was not derived.
- **Conclusion / status: STOP — the corrected-configuration trial is CLOSED.**
  - The answer to the card's question is **no**: the unchanged recipe under `MLX_ENABLE_TF32=0` did not pass Stage 3's
    held-out checks; it failed both value-drift limits.
  - Under the card, a STOP closes the trial, with no retry, tuning, rerun or selection. The strength screen is not
    permitted (only a PASS would have allowed review for one).
  - This does not establish why the value output drifts, does not qualify GPU training in general, and says nothing
    about the cause of entry 36's panic.
  - `calib020_0001` stays the incumbent (SHA-1 `209cf2d4…`). Nothing was evaluated, promoted, submitted or pushed.
  - See do-not-repeat **#87**.

### Decision note — after entry 38: the STOP accepted; a dual-encoder policy branch assessed on paper (2026-10-26, the user's decision)

- **Accepted (the user, after entry 38):** the STOP and the closeout. In the user's framing:
  - it closes the recipe, not the goal of stronger play;
  - because the held-out sets differ, the two trials do not show that the corrected setting worsened the drift or
    measure its effect. They show only that the drift persisted in the corrected run;
  - `calib020_0001` remains the incumbent;
  - the candidate stays a research artifact, with no re-scoring, tuning or strength screen.
- **The user's condition for a successor:** a distinct written rationale for preserving the value function. Adjusting
  only the anchor weight or the learning rate is not an authorized next step.
- **Wording agreed for the numerical work:**
  - "removed the gross GPU numerical errors in the tested scope; GPU training reliability remains unestablished";
  - no transfer of strength has been demonstrated (not "none occurred"); the latest candidate's strength is unknown.
  - Entry 32 remains a FAIL.
- **Authorized:** a short, design-only feasibility card for the user's proposal, a composite agent:
  - the incumbent's encoder and value head, frozen, supply values;
  - a trainable copy of the encoder and policy head supplies priors;
  - the value function is therefore preserved by construction, not by an anchor penalty.

  The card covers the frozen and trainable parameters, identity and preservation checks, the inference cost with a fair
  comparison, fresh data, one bounded trial and its stops. It reuses the existing infrastructure, with no new framework
  or parameter grid.
- **Not authorized:** model loading, games, training, gates and pushing.
- **Drafted:** `docs/superpowers/2026-10-26-dual-encoder-policy-branch-feasibility-card.md` (DRAFT).
  - **Cost:** two forward passes per leaf, about 2.00× FLOPs (the encoder is 99.97% of a forward pass); model size
    doubles.
  - **Fair comparison:** matched network evaluations, as in entry 11: the composite at 200 simulations against the
    incumbent at 400, 509 pairs, pass if the mean pair score exceeds 0.56020. The equal-simulation 400-vs-400 match
    is used only as a gross-regression screen.
  - **Fresh data:** a 120-game corpus generated under `MLX_ENABLE_TF32=0`. **Machine time:** roughly 17–25 h.
  - **Recommendation: reject on paper.** The composite would need a prior worth more than a doubling of search, plus
    the margin. On a log-linear reading of Stage A (an assumption, under the old protocol), that is about +175 Elo at
    equal simulations, about 84% of the whole 4× search gain, from one offline pass over about 6,400 positions.
    Entry 11 (0.594 at 200 vs 400, inconclusive) suggests halving may cost less. No distilled student has yet shown
    better play.
- **Prediction on record:** None stated in advance. The card's estimate (strength pass about 5–10%) is offered for the
  decision; it is not a prediction on record unless the user adopts it.
- **Status:** OPEN, awaiting the user's decision on the card. See do-not-repeat **#88**.

**Addendum to the decision note (2026-10-26): the proposal REJECTED on paper; documentation-only closeout.**
- **The user's decision:** reject on paper; do not build step 1. The architecture preserves value, but the current
  evidence does not justify its added inference cost. In the user's words, this particular experiment "has not earned
  its cost". That does not mean dual encoders cannot work or that preserving value cannot help.
- **The rationale, kept narrower than the card's estimates:**
  - +175 Elo is an illustrative scenario, conditional on log-linear search gains carried over from the old protocol
    and a target true score of 0.60. It is not a measured requirement.
  - Mirror averaging's 0.594 (entry 11) does not isolate the cost of halving simulations: it changed the evaluator too.
  - A non-pass could reveal harm or rule out a large benefit. The match is not designed to resolve modest improvements
    reliably.
  - The 5–10% figure is the assistant's subjective assessment, not adopted as a calibrated forecast.
- **What was executed:** nothing. No build, model loading, seed work, generation, training, games or gates. Not a
  negative experimental result about capacity, distillation or playing strength.
- **The card** is preserved unchanged as the reviewed proposal. Its §8 (appended) records the decision and the
  corrections to §6–7.
- **Prediction on record:** none (none was adopted).
- **Status: CLOSED on paper.** `calib020_0001` remains the incumbent. The broader stronger-play goal remains open, but
  this proposal does not provide a sufficiently supported next experiment. Nothing further is authorized. See
  do-not-repeat **#89**.

### Decision note — after the dual-encoder rejection: one fixed combination, mirror averaging + root sequential halving (2026-10-26, the user's decision)

- **Background:** a read-only review of the ledger and prototypes asked whether at most two existing, compatible
  interventions merit one combined experiment. It shortlisted mirror averaging (entry 11, #59) and root sequential
  halving (entry 15, #62).
  - **Why these two:** they are the only two strength screens of inference-level changes at matched network cost, both
    under eval-mode inference.
  - **Their strength:** both point estimates leaned positive, both intervals include 0.5, and neither established an
    advantage.
- **The user's corrections to the review:**
  - mirror averaging had 9 split pairs, not 10 (dated correction under entry 11);
  - halving's score is 0.672, unclipped interval [0.332, 1.011];
  - whether reflection consistency improves evaluation accuracy or halving's decisions is a hypothesis;
  - the combination's runtime is unmeasured;
  - the screens' machine and numerical setting differ from the proposed run, so their scores are not current baselines.
- **Authorized:** a design-only card for one candidate frozen without tuning:
  - the averaged evaluator at 200 simulations;
  - the ordinary opening search and readout through ply 19;
  - the existing halving rule from ply 20;
  - against the unchanged incumbent at 400, both under process-local `MLX_ENABLE_TF32=0`;
  - actual network evaluations and runtime recorded, with compute matching approximate.

  The card prices a composition-and-cost check, a gross-regression screen and a fully funded fresh strength
  comparison. It proposes reopening #47, #59 and #62 narrowly for this combination only; the individual
  configurations and old evidence stay closed. A win would not establish synergy, and no factorial study is needed to
  learn whether the combination wins.
- **Not authorized:** a build, model loading, seeds, GPU work, games and pushing.
- **Drafted:** `docs/superpowers/2026-10-26-mirror-averaging-plus-root-halving-combination-card.md` (DRAFT).
  - **Steps:** the build (CPU), C1 (≤ 15 min, no games), the screen (32 pairs, stop below 0.40), and the strength
    comparison (509 pairs; pass if the mean pair score exceeds 0.56020 and both matching guards hold).
  - **Machine time:** about 13–16 h estimated, from entry 19's derived rate. The combination's runtime and the TF32=0
    effect on play are unmeasured.
  - **The untested element:** halving's unchanged rule at 200 simulations (3/6/12/28 visits per round).
- **Prediction on record:** None stated in advance.
- **Status:** OPEN, awaiting the user's review of the card. See do-not-repeat **#90**.

**Addendum to the decision note (2026-10-26): the card approved and FROZEN; step B built and checked; C1 not run.**
- **The user's review:**
  - **Approved:** the fixed combination; the narrow #47, #59 and #62 exceptions, for this candidate and its specified
    controls only; step B.
  - **Not approved:** C1, the screen, the strength comparison, checkpoint loading, real GPU searches, seed
    registration and gameplay.
  - **Clarifications written into card §11:**
    - identical RNG states under the check-RNG convention (a fresh `random.Random(90009000)` per search, test-only
      band);
    - cost as ratios of totals over the fixed panel, with warm-up and the correctness and identity work excluded;
      guards unchanged at 1.05 and 1.10;
    - C1's hard ceiling of 15 minutes, its measurements pricing later runs without changing any budget or parameter;
    - a controlled case where `last_choice` differs from the readout;
    - component-only calls only as the named identity controls;
    - the exact Hoeffding formula; a non-pass "is not designed to reliably detect modest gains".
- **What ran:**
  - **Build `0ad4de35`:**
    - `scripts/GPU/alphazero/mirror_halving_combination.py`: the game loop copied verbatim from
      `root-halving-match/01_match.py` (a test checks the source), the composition, the pass-through control, the
      cost totals and both verdicts;
    - `mirror_halving_c1.py` with gate `MIRROR_HALVING_C1_AUTHORIZED` and
      `docs/superpowers/evidence/mirror-halving-c1-run/01_wrapper.sh` (TERM 890 s, KILL 10 s later);
    - `mirror_halving_match.py` with gates `MIRROR_HALVING_SCREEN_AUTHORIZED` and `MIRROR_HALVING_STRENGTH_AUTHORIZED`.
      It admits the screen only after a committed C1 PASS, and the strength comparison only after that and a screen
      CONTINUE;
    - the gate inventory 46 → 49, all closed.
  - **The prototypes are byte-identical** to the files entries 11 and 15 ran (SHA-256 checked against their records).
  - **Focused tests:** 20 (shared module), 27 (C1), 35 (match) passed; gate inventory 8 passed. Every public-entry test
    runs with its outputs redirected and the real seams replaced by fakes, closed-gate tests included.
  - **Loop checks (CPU, fake evaluators):**
    - **Run 1** (evidence in `0e134482`): STOP_ERROR. Part (a) had passed; part (b) stopped at its symmetry probe,
      before any game, because my check script's fake evaluator lacked the `.network` attribute the averaging wrapper
      reads. Kept as evidence; a one-line fix.
    - **Run 2** (`58924743`): COMPLETED, PASS.
      - (a) the copied loop reproduced `eval_runner`'s in both games.
      - (b) the combination at 200 against ordinary search at 400, with board-dependent fakes that make averaging
        change the outputs: two games to the 280-ply cap, 130 halving searches each. Every move from ply 20 was
        `last_choice`; none was supplied before ply 20 or by the incumbent. The readout would have chosen differently
        at 53 and 64 of those plies. These are fake evaluators, so this is not a signal about play.
  - **Regression, one run in the main checkout:** 3,272 passed / 20 failed / 38 errors, failure set equal to the
    58-id baseline.
- **Not done:** no injected-defect (mutation) controls for this build.
- **Executed against the model:** nothing. No checkpoint loaded, no GPU search, no seeds, no games; every gate is
  closed.
- **Prediction on record:** None stated in advance.
- **Status:** OPEN. The frozen build awaits the user's review before C1. See do-not-repeat **#91**.

**Addendum to the decision note (2026-10-26): step B accepted; C1 approved once.**
- **The user's review of step B:** accepted, with no blocking implementation defect found.
  - **Independently verified (read-only):**
    - the frozen card matches its pin, both prototypes match their historical hashes, and the copied loop functions
      are unchanged;
    - C1 implements identical starting RNG states, the fixed panel, excluded control and warm-up costs, ratios of
      totals and the unchanged guards;
    - the controlled `last_choice` test distinguishes the two move paths;
    - the loop evidence supports the reported equivalence and the 130 halving searches per game;
    - the regression log reports 3,272 passed / 20 failed / 38 errors, with the 58 ids equal to the baseline;
    - main `e04657d5`; 49 gates, all closed.
- **Accepted deviations:**
  - The screen and strength wrappers, plans and seed copies are deferred until after C1, whose measurements set
    their ceilings.
  - The loop check's disclosed CPU-harness repair and run 2, with run 1's failed evidence preserved. **This grants no
    retry permission for C1 or any later run.** No mutation campaign or additional regression is needed.
- **Evidence preserved, not rerun:** the step-B regression log and its failure-id comparison, create-only and
  read-only, in `docs/superpowers/evidence/mirror-halving-step-b-regression/`.
- **Approved: C1 only, once.** It follows a fresh read-only panic check and waits for an idle machine. It runs
  through its wrapper in a clean run worktree, opening gate 47 only, with:
  - process-local `MLX_ENABLE_TF32=0` and eval mode;
  - TERM at 890 s plus 10 s grace;
  - gate restoration and cleanup verified.
- **What closes the combination:** a C1 STOP, VOID or timeout, with no automatic retry.
- **The pinned card is unchanged.**
- **Still not authorized:** the screen, the strength comparison, seed registration, gameplay, training and pushing.
- See do-not-repeat **#92**.

### 39. Mirror averaging + root sequential halving — C1, the composition-and-cost check (2026-10-26, COMPLETED; PASS)

**Test.** Card `docs/superpowers/2026-10-26-mirror-averaging-plus-root-halving-combination-card.md` §4 and §5 C1, built
in `0ad4de35`. One bounded GPU run, no games and no schedulable seeds, on a fixed panel of 104 committed positions
(inference-mode-match games 0 and 1, as entry 19). Every search used a fresh `random.Random(90009000)` (test-only band).
- **Checks:**
  1. entry 10's averaging correctness check, run unchanged;
  2. control (a), plies < 20: the candidate equals averaging alone at 200 simulations;
  3. control (b), plies ≥ 20: halving over a pass-through wrapper equals halving over the plain evaluator at 200;
  4. cost: one search per side per position after one warm-up search each. The guards are ratios of totals: network
     evaluations ≤ 1.05× the incumbent's, search seconds ≤ 1.10×.
- **Sides:** the candidate (halving over the averaging wrapper, 200 simulations) and the incumbent (400).

**Hypothesis.** The composition works as specified (bitwise identities, halving at every position from ply 20), and
at 200 simulations the combination's cost is within the guards of the incumbent's at 400.

**Why it was plausible.** Each prototype ran in its own screen (entries 11 and 15). Entry 11 found the averaging side
cheaper in search time at matched evaluations, but on another machine and setting, and without halving.

**Prediction on record.** None stated in advance.

**What ran.**
- `92dbe110`: the approval was recorded; the read-only panic check found no new panic or reboot since 2026-10-25
  09:39.
- `7454e393`: gate 47 opened, in a clean run worktree.
- The wrapper:
  - inner and outer exit 0, COMPLETED; 216 s of the 900 s ceiling;
  - record written by the script;
  - no survivors; gate restored and read back.
- `e6a0b23a`: the evidence and the restored gate committed. Main fast-forwarded; the worktree and branch removed.
- `MLX_ENABLE_TF32` was `"0"` at start and end. Versions: MLX 0.30.3, NumPy 2.4.1, Python 3.14.9. 0 modules were in
  training mode at load and at the end, on both sides.
- **🔴 The idle condition was NOT met at launch, an execution lapse.** The pre-launch check was printed but not
  waited on.
  - At launch: load average 4.00, with Spotlight's `mds_stores` at about 313% CPU (the record's `loadavg_before` is
    3.997). At the end the load was 2.95.
  - The criterion used before GPU runs in this programme (load below 3, no process above 50% CPU) was therefore not
    satisfied.
  - The run is not repeated: the approval was one run, with no retry.
  - The two sides were interleaved position by position (the order alternating), so contention fell on both. Even so,
    **the absolute seconds and the projections derived from them may be inflated**, and the seconds ratio is not
    shown to be unaffected.

**Result.** **PASS**: every check held, and the failed-check list is empty.
- **Entry 10:** PASS, with 16 calls; the largest value difference was 3.0e-8 and the largest prior difference
  1.1e-8 (tolerance 1e-5).
- **Control (a):** 40 of 40 identical. **Control (b):** 64 of 64 identical. The candidate used its halving choice at
  all 64 positions from ply 20; the incumbent used none.
- **Totals (ratios of totals):**

| Side | Searches | Network evaluations | Network seconds | Search seconds |
|---|---|---|---|---|
| Candidate | 104 | 40,858 | 43.96 | 57.23 |
| Incumbent | 104 | 41,020 | 45.31 | 69.42 |

  - **Evaluations ratio 0.9961** (guard ≤ 1.05); **search-seconds ratio 0.8245** (guard ≤ 1.10).
  - The halves of the panel gave seconds ratios of 0.816 and 0.833.
  - Plies < 20: evaluations 1.0025, seconds 0.809. Plies ≥ 20: 0.9919 and 0.835.
  - Per-position seconds ratios ranged from 0.117 to 7.178 (median 0.815). Only the totals are gated.
- **DERIVED (search seconds only, at an assumed 59.8 plies per game, entry 15's mean):**
  - 0.550 s per search for the candidate and 0.667 s for the incumbent;
  - about 36.4 s of search per game;
  - screen about 0.65 h, strength comparison about 10.3 h (below the 30 h repricing line).
  - Whole-game time is unmeasured, and the contention above may inflate these figures.

**Did it match the prediction, and why?** There was none. The composition behaved as built: it consumed the
wrapper's outputs and its identities were exact. Its cost per search came in below the incumbent's, in line with
entry 11's direction (half the tree operations at equal network work). That is not shown to be the cause.

**Conclusion / status.** C1 PASS: the composition is verified and approximately compute-matched on this panel, under
the recorded contention.
- **Not shown:** anything about playing strength.
- The ceilings for the screen and the strength comparison may now be priced from these figures (card §6). Their
  wrappers, plans and seed blocks, the screen, the strength comparison and any game remain unauthorized pending the
  user's review.
- See do-not-repeat **#93**.

**Addendum to entry 39 (2026-10-26, the user's decision): C1 accepted with the disclosed deviation; screen and strength
ceilings priced; build-only approval.**
- **Accepted, qualified.** The user's read-only reconciliation:
  - the committed record shows correctness PASS, identities 40/40 and 64/64, evaluations ratio 0.99605 and seconds
    ratio 0.82445;
  - all 35 recorded module hashes match the execution commit;
  - the wrapper recorded exit 0, gate restoration and no survivors;
  - main `c36488fa`; 49 gates, all closed.
- **The idle requirement was violated.** Alternating the search order reduces exposure to timing drift, but it does
  **not** establish an equal slowdown or an unaffected ratio. The run is accepted for preparation with that
  qualification. **C1 is not rerun and its timings are not discounted:** there is no measured correction factor.
- **Ceilings:** entry 39's recorded projections, unchanged, × the frozen 1.5, rounded up to whole seconds. These are
  estimates from search time and an assumed game length, not measured whole-game durations.

| Step | Total ceiling | TERM, then 10 s grace |
|---|---:|---:|
| Screen, 32 pairs | 3,496 s (58 min 16 s) | 3,486 s |
| Strength comparison, 509 pairs | 55,601 s (15 h 26 min 41 s) | 55,591 s |

- **Approved: a build only.**
  - The screen and strength wrappers, copied from the existing ones.
  - The seed-procedure copies, with the plans prepared but the seed blocks unselected.
  - **Launch must depend on the idle check passing.** A printed check or a memory reminder is not sufficient. Focused
    CPU-only fake-launch checks must show that a busy machine prevents the launch and an idle one may reach the fake
    command.
  - The frozen candidate, budgets, guards and stopping rules are preserved. No additional full regression or mutation
    campaign.
  - C1's evidence and the pinned card are preserved.
- **Not authorized:** running the seed procedures, registration, checkpoint loading, GPU work, games, training and
  pushing. Gates 47–49 stay closed.

**Addendum to entry 39 (2026-10-26): the screen and strength build, built only (`b6632003`).**
- **The idle gate, `scripts/GPU/alphazero/idle_gate.py`:**
  - samples every 5 s; idle means a 1-minute load below 3 and no other process above 50% CPU, held 60 s in a row;
  - exit 0 once held; exit 1 if 30 minutes pass first.
- **The wrappers, `docs/superpowers/evidence/mirror-halving-{screen,strength}-run/01_wrapper.sh`:**
  - copies of C1's (`mirror-halving-c1-run/01_wrapper.sh`), with TERM at 3,486 s and 55,591 s plus 10 s grace (the
    pricing above);
  - the idle gate is their last precondition. The launch line is reachable only if it exits 0; otherwise the wrapper
    REFUSES (exit 1). Nothing then starts or is written, and the gate is restored;
  - once a run starts, the idle samples are kept, create-only, in `idle_check.txt`.
- **The seed-procedure copies:**
  - surveys `mirror-halving-{screen,strength}-seed-survey/01_seed_survey.py` (64 and 1,018 slots) and closings
    `…-seed-closing/01_derive_usage.py`, copied from attempt 2's with only the size, paths, titles, plan kind, slot
    lines and one control's label changed;
  - **the plans are prepared as each survey's `build_plan`, with no `plan.json` committed.** The pre-registration
    proof requires that no plan exists, and the match refuses a plan without an integer base; the seed blocks are
    unselected.
- **One addition to `mirror_halving_match.py`:** it now logs `game_start` (game = slot) before each game, because the
  copied closing derivation exposes the started prefix from those events, and the step-B build logged only
  `game_end`.
  - The candidate, budgets, guards and stopping rules are unchanged; no pinned file changed.
- **Tests:** 23 new focused CPU tests (22 for the launch, 1 for the match); 86 pass across the shared-module, match,
  launch and gate-inventory files.
  - **Fake launches of the REAL wrapper bytes,** both kinds, in throwaway git repositories with a stub idle gate and a
    stub match:
    - busy → exit 1, the stub never reached, nothing written, the gate restored;
    - idle → the stub reached with exactly `run <kind> --checkpoint <path>` and `MLX_ENABLE_TF32=0`, `idle_check.txt`
      kept, the gate restored.
  - The idle gate refuses entry 39's own launch conditions.
  - **Each ceiling is checked against C1's committed record:** ⌈1.5 × projection⌉ = TERM + 10.
- **Not done:** no additional full regression or mutation campaign, as approved.
- **Commits:** the C1 evidence and the pinned card are unchanged. A message-only amend corrected the build commit's
  test count before anything was pushed.
- **Not run:** no seed procedure, registration, checkpoint loading, GPU work or game. 49 gates, all closed.
- **Status:** the build awaits the user's review. See do-not-repeat **#94**.

**Addendum to entry 39 (2026-10-26, the user's decision): the screen/strength build accepted; the screen's seed survey and
pre-registration proof approved.**
- **Accepted, with no blocking finding.** The user's read-only checks:
  - both ceilings match C1's undiscounted projections (3,496 s and 55,601 s);
  - the launch depends on the idle gate, and the fake launches cover the busy refusal and the exact permitted
    command;
  - both seed-method pins match, the sizes are 64 and 1,018, and no plan or survey output exists;
  - the frozen card, the candidate's components and C1's evidence are unchanged; 49 gates, all closed.
- **Both deviations accepted:**
  - `build_plan` prepares the plan while meeting the pre-proof's no-plan requirement;
  - the `game_start` event is a necessary exposure-accounting correction that does not change play.
- **Approved, in order:**
  1. The screen's seed survey, once, under its fixed selection rule, from a clean committed checkout.
  2. Only if it is CLEAN with every control passing: commit its create-only evidence, then run the screen's
     pre-registration proof once.
  3. Return both records and the exact commits for registration review.

  If either fails: stop, with no block adjustment and no automatic retry. No additional regression or mutation
  testing.
- **Not authorized:** registration, plan creation, the post-registration proof, either gameplay run, the strength
  comparison's seed procedures, checkpoint loading, GPU work and pushing.

**Addendum to entry 39 (2026-10-26): the screen's seed survey and pre-registration proof, each run once.**
- **Where it ran:** a clean linked worktree (branch `mirror-halving-screen-seeds`, from `d4ceb641`); main
  fast-forwarded and the worktree removed afterwards.
  - The scope was this tree's and the main checkout's `scripts`, `tests` and `docs/superpowers`, fail-closed: 5,161
    and 47,852 regular files, no symbolic links, 5,248 band literal values.
- **The survey (`28cabf0e`):** exit 0; `02_seed_survey_run.txt`, sha256 `44b1fd0c…`, at `d4ceb641`, working tree
  clean.
  - The fixed rule: the highest registered or named-spent seed in the band is 202692411, so BASE = 202692500.
  - **Block [202692500, 202692564), 64 seeds:** no direct overlap, 0 derived streams, injective, nearest gap 82
    (floor 64). CLEAN.
  - Controls 45/45. Verdict: THE MIRROR-HALVING SCREEN BLOCK IS CLEAN.
- **The pre-registration proof (`8c52583d`):** exit 0; `03_collision_proof_pre_run.txt`, sha256 `631026a3…`, at
  `28cabf0e`, working tree clean.
  - It read the block from the committed survey record.
  - Expected state (pre) HOLDS: the registry unchanged, the block unregistered, no plan.
  - The block is CLEAN again; controls 45/45. Verdict: THE MIRROR-HALVING SCREEN BLOCK IS CLEAN AND THE STATE IS AS
    EXPECTED.
- **Not done:** no registration (the block appears nowhere in the registry), no plan, no post-registration proof, no
  game, no strength-comparison seed procedure, no checkpoint loading or GPU work. 49 gates, all closed.
- **Prediction on record:** none (a seed procedure).
- **Status:** awaiting the user's registration review. See do-not-repeat **#95**.

**Addendum to entry 39 (2026-10-26, the user's decision): registration review passed; the screen block's registration,
its plan and the post-registration proof approved.**
- **The user's read-only checks:**
  - both full record hashes, and 45 passing controls in each;
  - the pre-proof is bound to the survey;
  - the live registry still matches the surveyed hash, contains no overlapping interval, and has 43 ACCOUNTED
    blocks;
  - the fixed rule gives BASE 202692500;
  - no implementation changed since the reviewed build; 49 gates, all closed.
- **The memory-note slip is not a blocker;** the committed evidence is authoritative.
- **Approved, in order:**
  1. Append `(202692500, 202692564)` last in `ACCOUNTED_SEED_INTERVALS` (43 → 44), leaving all other registry state
     unchanged.
  2. Create and commit the screen plan exactly as `build_plan` specifies: kind `mirror-halving-screen-plan/1`, base
     202692500, n_slots 64, survey_record_sha256 `44b1fd0c2d612ffe2d1f7dac1cbc31b9015d6983a16514d0b2a4bd905fa4be98`.
  3. From a clean committed checkout, run post mode once, preserving its create-only evidence.
  4. Return the registration diff, the plan, the post-proof and the exact commits for screen approval.

  If the registry has changed before registration, or the post-proof fails: stop and return, with no block
  adjustment and no automatic retry.
- **Not authorized:** opening any gate, checkpoint loading, GPU work, gameplay (the screen itself), the strength
  comparison's seed work and pushing.

**Addendum to entry 39 (2026-10-26): the screen block registered; the post-registration proof run once.**
- **Before registering:** the registry's sha256 was `85f0a4cb…`, the hash the survey read; unchanged.
- **Registration and plan (`aa977403`):**
  - `(202692500, 202692564)` appended last in `ACCOUNTED_SEED_INTERVALS` with its comment block. An AST check
    against the parent: ACCOUNTED 43 → 44, the prefix identical, every other statement unchanged. EXPOSED and RETIRED
    are untouched.
  - `docs/superpowers/evidence/mirror-halving-screen-run/plan.json` is the survey script's
    `build_plan(202692500, sha256 of record 02)`, in the D2 attempt-2 plan's byte format: kind
    `mirror-halving-screen-plan/1`, base 202692500, n_slots 64, survey_record_sha256 `44b1fd0c…be98`.
- **The post-registration proof (`4dc418db`):** exit 0; `04_collision_proof_post_run.txt`, sha256 `8308569c…`, in a
  clean linked worktree at `aa977403` (branch `mirror-halving-screen-post`; main fast-forwarded and the worktree
  removed afterwards).
  - It read the block from survey record 02. Expected state (post) HOLDS: ACCOUNTED is the survey's plus exactly
    the block appended last, nothing else changed, and the plan is committed and equal to `build_plan`.
  - Direct overlap none, 0 derived streams, injective, nearest gap 82 (floor 64). Scope: 5,164 + 47,855 regular
    files, no symbolic links.
  - Controls 45/45. Verdict: THE MIRROR-HALVING SCREEN BLOCK IS CLEAN AND THE STATE IS AS EXPECTED.
- **Tests (CPU only; no regression was asked for):** `test_seed_selection`, `test_e4_screen_reference`,
  `test_mirror_halving_match` and `test_mirror_halving_launch` gave 105 passed and 1 failed.
  - The failure is `test_NO_PLAN_IS_COMMITTED_YET_the_seed_blocks_are_unselected[screen]`, the build-time guard
    that no screen plan or seed record exists. 🔴 Its record glob has been red since the survey record was committed
    at `28cabf0e`; I did not run it then and did not report it. Now the plan check fails as well.
  - `tests/test_search_distillation_d2_heldout.py`'s real-state attempt-2 test asserts that the D2 attempt-2 block
    is last in ACCOUNTED, so this registration makes that assertion false. It needs the checkpoint, so it was not
    run.
  - Neither test was edited: re-anchoring them is not authorized.
- **Not done:** no gate opened (49 closed), no checkpoint loaded, no GPU work, no game, no strength-comparison seed
  procedure, no push.
- **Prediction on record:** none (a seed procedure).
- **Status:** registered and proven; awaiting the user's screen approval. See do-not-repeat **#96**.

**Addendum to entry 39 (2026-10-26, the user's decision): registration accepted; two narrow test repairs and the screen,
once, approved.**
- **The user's read-only checks:**
  - the exact ACCOUNTED append, with the other registry statements unchanged;
  - the plan is correct, and the post-proof's full hash and 45 controls check out;
  - the block is unexposed and unretired; 49 gates, all closed; the candidate and the wrapper are unchanged.
- **Test maintenance (a separate, small commit):**
  - Replace the screen's obsolete "no plan or records" assertion with checks of its approved plan, survey binding
    and registration. Keep the strength block's unselected-state check.
  - Let the historical D2 attempt-2 block have later registrations while preserving its unique-registration,
    exposure, retirement and unavailability assertions. Do not simply remove the protection.
  - Run only the affected focused CPU tests, with existing isolation preserved. No full regression, mutation
    campaign or proof rerun. (The D2 test does not load a checkpoint; its model-loading helpers are deferred.)
- **The screen, once, limited to:**
  - 32 colour-reversed pairs (64 games) using exactly [202692500, 202692564);
  - the frozen candidate at 200 simulations versus the incumbent at 400, process-local TF32=0, eval mode;
  - a fresh read-only panic/reboot check, then the enforced idle gate;
  - a clean run worktree and a one-line opening of gate 48 only, through the reviewed wrapper (TERM at 3,486 s,
    KILL 10 s later);
  - the frozen verdict: STOP below a mean pair score of 0.40; otherwise CONTINUE, which permits review, not a
    strength claim.

  No further approval round before the launch if the test repairs and preflight pass.
- **Return:** the create-only run evidence, cost totals, gate restoration and cleanup verification, for result and
  closing review. A started-run STOP, VOID or timeout closes the combination without retry.
- **Not authorized:** strength-comparison seed work, gate 49, training and pushing.

**Addendum to entry 39 (2026-10-26): the screen, run once — COMPLETED, verdict CONTINUE.**
- **Test:** the card's §5 S, a gross-regression screen. The frozen combination (MirrorAveragingEvaluator around
  eval-mode calib020_0001 at 200 simulations, root sequential halving from ply 20, 16 candidates) against the
  unchanged incumbent at 400 simulations, in 32 colour-reversed pairs.
- **Hypothesis:** the combination is not grossly weaker than the incumbent at comparable compute. Why plausible:
  entries 11 and 15 (each component alone was above parity in 16 pairs, both intervals including parity) and C1's
  composition and cost check (entry 39).
- **Prediction on record:** None stated in advance. The 0.40 bar is a frozen stopping rule, not a prediction.
- **Before the run:**
  - test repairs at `dceac987` (23 focused CPU tests pass);
  - read-only panic/reboot check: no new panic and no reboot since 2026-10-25 09:39;
  - a clean run worktree from `dceac987`, with gate 48 alone opened by a one-line commit `0aaad193`.
- **The run (`59bcd53d`):** the reviewed wrapper, once.
  - **Idle gate:** busy at first (load 3.9, `mds` at 148% CPU); it held 60 s idle at t = 116 s and only then
    launched. Samples are in `idle_check.txt`.
  - **Outcome:** inner exit 0, COMPLETED, 2,271 s of the 3,486 s ceiling. The script wrote the record; no
    process survived; the gate was restored and read back equal to `dceac987`.
  - **Record:** `mirror-halving-screen/record.json`, sha256 `08342fe4…`, with `games.jsonl`, 64 replays and
    `progress.jsonl`.
  - **Integrity:**
    - TF32=0 at start and end;
    - both sides in eval mode at load and at the end (0 of 42 modules in training mode);
    - checkpoint sha1 `209cf2d4…`.
  - **Seeds:** base 202692500, 64 slots; `game_start` was logged for slots 0–63 (all started), and all 64 ended.
- **Result:**
  - **Games:** candidate 36 wins, incumbent 27, 1 draw (a state cap at 280 plies); 63 games ended by a connection.
    Mean length 57.5 plies (min 35).
  - **Pair scores:** 1.0 ×11, 0.75 ×1, 0.5 ×13, 0.0 ×7. Mean pair score **0.5703125** ≥ 0.40, so **CONTINUE**.
  - **Descriptive interval:** Hoeffding95 over 32 pairs is [0.330, 0.810], which includes 0.5. The screen was not
    designed to measure strength.
  - **Cost totals:**
    - candidate: 722,632 network evaluations, 1,018.0 s search, 1,843 searches (1,203 halving);
    - incumbent: 725,840 evaluations, 1,247.9 s search, 1,838 searches;
    - ratios: evaluations **0.9956** (≤ 1.05), search seconds **0.8158** (≤ 1.10). Both guards hold.
  - **Derived, not measured:** 35.5 s per game here suggests about 10.0 h for the strength comparison's 1,018 games,
    inside its 55,601 s ceiling. This assumes comparable game lengths (here 57.5 plies against the 59.8 assumed
    from C1).
- **Did it match the prediction?** No prediction was stated. The result passes the stopping rule and is consistent
  with C1's cost picture (C1: 0.9961 and 0.8245).
- **After the run (read-only):** no new panic and no reboot; no run process left. The worktree and branch were
  removed and main fast-forwarded. 49 gates, all closed (gate inventory 8 passed).
- **Not done:** no seed closing (all 64 seeds are started and must be exposed; the whole block retired), no
  strength-comparison seed work, gate 49 closed, no training, no push.
- **Conclusion / status:** CONTINUE. This is permission to review, not a strength claim. Awaits the user's result and
  closing review. See do-not-repeat **#97**.

**Addendum to entry 39 (2026-10-26, the user's decision): the screen's result accepted (CONTINUE); closing its seed
block approved.**
- **The result is encouraging but inconclusive:** stronger play has not been demonstrated.
- **The user's independent reconciliation:**
  - the mean pair score is 0.5703125, with interval [0.3302, 0.8104];
  - all 64 games started and completed, with correct seed and colour bindings;
  - the per-game costs sum exactly to the reported totals, and both guards held;
  - all 37 recorded module hashes match the execution commit;
  - the idle hold preceded the launch, the gate restoration matches, and 49 gates are closed.
- **Approved, in order:**
  1. Run the existing K1 usage derivation once, from a clean committed checkout, preserving its create-only evidence.
  2. Only if it RECONCILES with every control passing, append `(202692500, 202692564)` last to EXPOSED (41 → 42) and
     to RETIRED (39 → 40). Leave ACCOUNTED at 44 and all other registry state unchanged.
- **Also approved:** the small lifecycle-test updates this closure requires.
  - Assert the screen block's full exposure, retirement and unavailability.
  - Allow later, higher closures after the historical D2 block, while preserving its closure checks.
  - Run only the affected focused CPU tests: no full regression and no proof rerun.
- **Return:** the K1 record, the exact closing diff, the focused-test results and the commits.
- **Not authorized:** a screen retry, strength-comparison seed work, gate 49, training and pushing. The frozen
  strength ceiling remains 55,601 s.

**Addendum to entry 39 (2026-10-26): the screen seed block closed.**
- **K1, the usage derivation (`adbb31b2`):** exit 0, run once in a clean linked worktree at `35732676` (branch
  `mirror-halving-screen-close`; main fast-forwarded and the worktree removed afterwards).
  - Record: `mirror-halving-screen-seed-closing/02_derive_usage_run.txt`, sha256 `984afe63…`.
  - It read the committed, unmodified inputs: the plan, the progress log, `outer.json` and `record.json`. Outer
    exit 0, no survivors, status COMPLETED; the `game_start` slots were exactly 0..63. It read no score, winner or
    move.
  - Controls 15/15. Verdict: RECONCILES, so expose and retire the whole block. (Its verdict line prints the block
    with a closing bracket; the controls' tuples show the half-open interval.)
- **K2, the closing edit (`52d179da`):** `(202692500, 202692564)` appended last, with its comment block, to
  `EXPOSED_SEED_INTERVALS` (41 → 42) and `RETIRED_SEED_INTERVALS` (39 → 40).
  - An AST check against the parent: both prefixes identical; ACCOUNTED (44), TEST_ONLY and every other statement
    unchanged.
- **Lifecycle tests (`00c0d7b4`):**
  - The screen test asserts the block is EXPOSED and RETIRED exactly once each, with no other interval overlapping
    it, and that every seed is accounted, exposed, retired and unavailable.
  - The D2 attempt-2 test keeps its block in EXPOSED and RETIRED exactly once each, followed in both by the same
    number of later, higher entries; every seed is still unavailable.
  - Focused CPU tests (seed selection, the registry, mirror-halving launch and match, D2 held-out, tactical seed
    closing, the gate inventory): 184 passed. No regression and no proof rerun.
- **A process slip, caught before anything was wrong:** my first K2 and test commits used a shell variable for the
  git identity, which zsh does not split. Both commands failed, and I had hidden stderr. Nothing was committed; I
  saw the files still modified and recommitted with the flags written out.
- **Not done:** no screen retry, no strength-comparison seed work, gate 49 closed (49 gates closed), no training, no
  push.
- **Status:** the screen is closed: result accepted (CONTINUE, not a strength result) and its block exposed and
  retired. The strength comparison awaits a separate decision. See do-not-repeat **#98**.

**Addendum to entry 39 (2026-10-26, the user's decision): the screen's closing accepted; the strength comparison's
seed survey and pre-registration proof approved.**
- **The user's checks:**
  - the K1 hash and its 15 passing controls;
  - both exact registry appends, with the remaining registry statements unchanged;
  - 49 gates closed;
  - the lifecycle-test changes preserve the relevant protections;
  - the bracket display and the corrected commit-command slip do not invalidate the evidence.
- **What the screen supports:** it cleared its stopping rule at comparable measured cost. That supports proceeding
  with the planned strength test; it does not establish stronger play or justify changing its size.
- **Approved, in order:**
  1. The strength seed survey, once, for 1,018 fresh seeds under the existing fixed rule against the current registry.
  2. Only if it is CLEAN with every control passing: commit its create-only record, then run the pre-registration
     proof once.
  3. Return both records, their hashes and the exact commits for registration review.

  If either fails: stop, with no block adjustment and no automatic retry.
- **Also approved:** the narrow update to the strength test's obsolete "no survey records" assertion, preserving
  provenance and unregistered-state checks; focused CPU verification only.
- **Unchanged:** 509 pairs, the exact Hoeffding rule, both compute guards and the 55,601 s ceiling. Screen games must
  not be pooled into the strength result.
- **Not authorized:** registration, plan creation, the post-registration proof, gate 49, checkpoint loading, GPU
  work, gameplay, training and pushing.

**Addendum to entry 39 (2026-10-26): the strength comparison's seed survey and pre-registration proof, each run once.**
- **Where it ran:** a clean linked worktree (branch `mirror-halving-strength-seeds`, from `8e5665df`); main
  fast-forwarded and the worktree removed afterwards.
  - The scope was this tree's and the main checkout's `scripts`, `tests` and `docs/superpowers`, fail-closed: 5,237
    and 47,928 regular files, no symbolic links, 5,314 band literal values.
- **The survey (`2ccc9037`):** exit 0; `02_seed_survey_run.txt`, sha256 `7bdf6cf6…`, at `8e5665df`, working tree
  clean. It read the registry at sha256 `e39aa059…` (ACCOUNTED 44, EXPOSED 42, RETIRED 40).
  - The fixed rule: the highest registered or named-spent seed in the band is 202692563 (the screen's last), so
    BASE = 202693600.
  - **Block [202693600, 202694618), 1,018 seeds:** no direct overlap, 0 derived streams, injective, nearest gap
    1,030 (floor 1,018). CLEAN.
  - Controls 45/45. Verdict: THE MIRROR-HALVING STRENGTH BLOCK IS CLEAN.
- **The pre-registration proof (`9631ec72`):** exit 0; `03_collision_proof_pre_run.txt`, sha256 `b692d3b0…`, at
  `2ccc9037`, working tree clean.
  - It read the block from the committed survey record.
  - Expected state (pre) HOLDS: the registry unchanged, the block unregistered, no plan.
  - The block is CLEAN again; controls 45/45. Verdict: THE MIRROR-HALVING STRENGTH BLOCK IS CLEAN AND THE STATE IS
    AS EXPECTED.
- **Test update (`8d302a7f`):** the strength test's "no survey records" check now requires exactly records 02 and
  03. Its other conditions:
  - the survey is CLEAN, 45/45;
  - the pre-proof HOLDS, 45/45, bound to record 02's sha256 and block;
  - there is no plan, post-proof or closing record;
  - no registry interval overlaps the block.

  Focused CPU tests (mirror-halving launch and match): 58 passed.
- **Not done:** no registration, no plan, no post-registration proof, gate 49 closed (49 gates closed), no checkpoint
  loading or GPU work, no game, no training, no push.
- **Prediction on record:** none (a seed procedure).
- **Status:** awaiting the user's registration review. See do-not-repeat **#99**.

**Addendum to entry 39 (2026-10-26, the user's decision): the strength block's registration review passed; its
registration, plan and post-registration proof approved.**
- **The user's checks:**
  - both full record hashes, all 1,018 slot mappings, and 45 passing controls per record;
  - the method pins match, and the pre-proof is bound to the survey;
  - the live registry still matches the surveyed hash and contains no overlap;
  - the selection arithmetic is correct; 49 gates, all closed.
- **Approved, in order:**
  1. Append `(202693600, 202694618)` last to ACCOUNTED (44 → 45), leaving every other registry state unchanged.
  2. Create and commit the plan exactly as `build_plan` specifies: kind `mirror-halving-strength-plan/1`, base
     202693600, n_slots 1018, survey_record_sha256 `7bdf6cf6caf7c6b9b7f98f2c9aa4f6ffcc805130d7f7e200756ca1ba67a2c0d6`.
  3. Run post mode once from a clean committed checkout, preserving its create-only evidence.
  4. Return the registration diff, the plan, the post-proof and the exact commits for run approval.

  If the registry changes before registration or the post-proof fails: stop, with no block adjustment and no
  automatic retry.
- **Also approved:** the narrow lifecycle-test update for the registered state.
  - Preserve the survey and proof provenance.
  - Require unique registration, and confirm the block remains unexposed, unretired and schedulable.
  - Focused CPU verification only: no full regression and no proof rerun.
- **Unchanged:** the 509-pair design and the 55,601 s ceiling.
- **Not authorized:** gate 49, checkpoint loading, GPU work, gameplay, training and pushing; the strength run itself.

**Addendum to entry 39 (2026-10-26): the strength block registered; the post-registration proof run once.**
- **Before registering:** the registry's sha256 was `e39aa059…`, the hash the survey read; unchanged.
- **Registration and plan (`0c1c18b5`):**
  - `(202693600, 202694618)` appended last in `ACCOUNTED_SEED_INTERVALS` with its comment block. An AST check
    against the parent: ACCOUNTED 44 → 45, the prefix identical; EXPOSED (42), RETIRED (40), TEST_ONLY and every
    other statement unchanged.
  - `docs/superpowers/evidence/mirror-halving-strength-run/plan.json` is the survey script's
    `build_plan(202693600, sha256 of record 02)`, in the screen plan's byte format: kind
    `mirror-halving-strength-plan/1`, base 202693600, n_slots 1018, survey_record_sha256 `7bdf6cf6…c0d6`.
- **The post-registration proof (`5c4b60a7`):** exit 0; `04_collision_proof_post_run.txt`, sha256 `ca26c5a5…`, in a
  clean linked worktree at `0c1c18b5` (branch `mirror-halving-strength-post`; main fast-forwarded and the worktree
  removed afterwards).
  - It read the block from survey record 02. Expected state (post) HOLDS: ACCOUNTED is the survey's plus exactly
    the block appended last, nothing else changed, and the plan is committed and equal to `build_plan`.
  - Direct overlap none, 0 derived streams, injective, nearest gap 1,030 (floor 1,018). Scope: 5,240 + 47,931
    regular files, no symbolic links.
  - Controls 45/45. Verdict: THE MIRROR-HALVING STRENGTH BLOCK IS CLEAN AND THE STATE IS AS EXPECTED.
- **Lifecycle test (`5e07bba4`):** the strength test now requires records 02, 03 and 04, each proof bound to record 02's
  sha256. Its other conditions:
  - the plan equals `build_plan` of the surveyed block;
  - the block is in ACCOUNTED exactly once, with no EXPOSED, RETIRED or TEST_ONLY overlap;
  - every seed is schedulable by the entry's own predicates;
  - no closing record exists.

  Focused CPU tests: 184 passed. No regression and no proof rerun.
- **Not done:** gate 49 closed (49 gates closed), no checkpoint loading, no GPU work, no game, no training, no push.
- **Prediction on record:** none (a seed procedure).
- **Status:** registered and proven; awaiting the user's run approval. See do-not-repeat **#100**.

**Addendum to entry 39 (2026-10-26, the user's decision): the strength block's registration accepted; the strength
comparison approved, once, within the frozen protocol.**
- **The user's checks:**
  - the exact ACCOUNTED append, with the other registry statements unchanged;
  - the plan is correct, the post-proof's full hash checks out, and all 1,018 slot mappings and 45 controls pass;
  - the block is unexposed and unretired, and the C1 and screen admission records qualify;
  - the runtime code and the wrapper are unchanged; 49 gates, all closed.

  No further preparation or testing is required.
- **The run, once, limited to:**
  - 509 pairs (1,018 games) using exactly [202693600, 202694618), with no screen games pooled in;
  - the frozen candidate at 200 simulations versus the unchanged incumbent at 400, on 24×24 under the existing
    rules, with process-local TF32=0 and eval mode;
  - a fresh read-only panic/reboot check, then the enforced idle gate. A new panic stops GPU work pending a
    stability decision;
  - a clean run worktree, with gate 49 alone opened by a one-line commit, through the reviewed wrapper;
  - TERM at 55,591 s and KILL 10 s later: a total ceiling of 55,601 s.
- **The frozen judgement of a completed run:** PASS iff the mean pair score − √(ln(40)/(2×509)) > 0.5, with the
  candidate/incumbent ratios of total network evaluations ≤ 1.05 and of total search seconds ≤ 1.10. Clearing the
  score bound while failing a guard is NOT_COMPUTE_MATCHED.
- **Forbidden:** interim strength scoring, an early success decision, extension, top-up, retry and parameter changes.
  An incomplete run receives no score.
- **Return:** the create-only results, cost totals, wrapper diagnostics, gate restoration and cleanup evidence, for
  result and closing review.
- **Not authorized:** seed closing, deployment, training and pushing.

**Addendum to entry 39 (2026-10-27): the strength comparison, run once → TIMEOUT / VOID; no score.**
- **Test:** the frozen combination (MirrorAveragingEvaluator around eval-mode `calib020_0001` at 200 simulations,
  root sequential halving from ply 20 with 16 candidates) against the unchanged incumbent at 400 simulations, over
  509 colour-reversed pairs on seeds [202693600, 202694618), TF32=0 on both sides.
- **Hypothesis:** the combination plays more strongly than the incumbent at comparable compute.
- **Why it was plausible:** the screen's CONTINUE (mean pair score 0.5703 over 32 pairs, encouraging but
  inconclusive), with an evaluation ratio of 0.9956 and a search-seconds ratio of 0.8158.
- **Prediction on record:** None stated in advance. The Hoeffding rule is a frozen decision rule, not a prediction.
- **What ran:**
  - pre-run panic/reboot check clean (boot time unchanged since Oct 25 09:39:28; no new panic report);
  - clean run worktree, gate 49 opened alone (`264df589`, pre-open `396a1f1c`), through the reviewed wrapper;
  - idle gate: busy at t = 0 (mds 107 %), idle held 60 s at t = 65.6 s, then launched at load 2.0.
- **Result (`28737707`):**
  - wrapper outer exit 124, outcome TIMEOUT, wall 55,591 s;
  - record status VOID, step "terminated", "SIGTERM after 55590.9 s", written by the script;
  - strength_verdict and guards null; 800 games started and 799 ended of 1,018;
  - TF32=0 at start and end; survivors none; gate 49 restored by the wrapper and read back equal to `396a1f1c`.
- **Cause of the overrun (derived from timing only):**
  - games 0–349 ran at 0.61–0.63 s per move, matching the screen's 35 s per game;
  - from about t ≈ 14,000 s, 0.9 to 2.2 s per move; load average 14 at the end (2.0 at the start);
  - the BOINC client (Einstein@Home, NFS@Home, PrimeGrid) was started at 19:55 and 19:58 EDT, about 3.8 h into the
    run. It computes whenever there has been no keyboard or mouse input for 5 minutes;
  - its "suspend if non-BOINC CPU load exceeds 25 %" rule did not hold it off: a single-process run stays under that
    share of a many-core machine;
  - at TERM, eight or more nice'd BOINC tasks were each at about 99 % CPU;
  - the idle gate guards the launch only, as designed.
- **Not read:** no game outcome, pair score or cost ratio was read, during or after the run. An incomplete run receives
  no score under the frozen terms.
- **Did it match the prediction:** there was no prediction, and the run produced no score to compare.
- **Post-run checks:**
  - boot time unchanged; no panic report;
  - worktree committed by exact paths, main fast-forwarded to `28737707`, worktree and branch
    `mirror-halving-strength-run` removed;
  - 49 gates, all `False`.
- **Not done:** seed closing (block [202693600, 202694618) consumed in fact, not yet EXPOSED or RETIRED), deployment,
  training, push.
- **Status:** VOID by timeout, no strength verdict; stronger play not demonstrated. The frozen terms forbid retry,
  extension, top-up and scoring of the partial games, so the strength comparison cannot be rerun under this approval.
  Closing review is the user's. See do-not-repeat **#101**.

**Addendum to entry 39 (2026-10-27, the user's decision): close the void strength block; one new attempt approved in
principle because of the documented machine contention.**
- **Step 1, approved now:** close [202693600, 202694618) through its closing procedure. This means K1 (the
  `b6632003` strength seed-usage derivation) run once, then the K2 registry edit it reconciles to, plus the lifecycle
  tests re-anchored. The 799 partial games stay unscored.
- **Step 2, the user's decision:** one new attempt is explicitly approved because of the BOINC contention documented
  above. This is new text that narrows do-not-repeat #101 for exactly that one attempt. #101 is otherwise unchanged:
  the void run's games are never scored, pooled or extended.
- **Step 3, the new attempt's frozen terms:** the same fixed 509-pair comparison on fresh seeds, with the scoring rule
  and both compute guards unchanged, and BOINC suspended until the run completes.
- **Not yet done:** the new attempt's seeds (survey, pre-proof, registration, post-proof) and any gate opening, GPU
  work or run. Each follows the same step-by-step approvals. Deployment, training and pushing remain unauthorized.

**Addendum to entry 39 (2026-10-27): the void strength block closed.**
- **K1 (`8ad03a26`):** the `b6632003` strength seed-usage derivation, run once in a clean worktree at `376b75a3`.
  - It read only the plan, the progress events and slots, outer.json and the record's status.
  - Inputs: outer exit 124, no survivors, status VOID, game starts at slots 0..799 in order.
  - Controls 15/15. Verdict RECONCILES: expose [202693600, 202694400), the 800 started slots; retire the whole block
    [202693600, 202694618).
- **K2 (`21a2c6f0`):** EXPOSED 42 → 43 and RETIRED 40 → 41, each with the new interval appended last.
  - An AST check against the parent: only those two statements changed, with prefixes identical.
  - ACCOUNTED (45), TEST_ONLY and every other statement unchanged.
- **Lifecycle tests (`214063bb`):**
  - The strength test is renamed `…_REGISTERED_and_CLOSED`. It now asserts the K1 verdict line, the started prefix
    EXPOSED once, the whole block RETIRED once, and no other overlapping interval.
  - It also asserts every seed is accounted, retired and unavailable, and exposed exactly when it was started.
  - No duplicate test names. Focused CPU tests: 184 passed.
- **Not done:** the 799 games stay unscored. No outcome, pair score or cost ratio was read. No gate opening, GPU work,
  new seed work, training or push.
- **Prediction on record:** none (a seed procedure).
- **Status:** block closed. The one new attempt (approved in principle, above) needs fresh seeds and its own run
  paths, because the strength evidence and plan paths are create-only and now used. See do-not-repeat **#102**.

**Addendum to entry 39 (2026-10-27, the user's decision): the void run's closing accepted; the minimal attempt-2 build
approved, for review only.**
- **BOINC:** the user quits BOINC and keeps it quit until the new run finishes. Nothing suspends or restores it
  automatically.
- **The build, limited to:**
  - separate plan, evidence and closing paths, and gate 50, initially closed;
  - the existing search and scoring code, with the 509 pairs, simulations, guards and the 55,601 s ceiling unchanged;
  - copies of the wrapper and the seed procedures, with fresh seeds left unselected;
  - simple read-only checks for the BOINC client and workers, before launch and periodically during the run. Each
    observation is recorded; detection or a failed check prevents the launch or ends the attempt as VOID;
  - focused CPU tests with fake launches and temporary outputs, including BOINC absent and present, and the updated
    gate inventory. No full regression or mutation campaign.
- **Records:** a short dated attempt-2 amendment beside the original card, which stays unchanged, as does attempt 1's
  evidence.
- **The exception:** it permits one fresh attempt only. The original 799 games remain unscored.
- **Mid-build (2026-10-27), the user's decision:** BOINC's podman VM (`/opt/podman/bin/` gvproxy and krunkit) "is
  fine": it does none of the work that slowed attempt 1, so the check lists it as exempt.
- **Not authorized:** seed procedures, registration, checkpoint loading, GPU work, gameplay, opening gate 50 and
  pushing.

**Addendum to entry 39 (2026-10-27): the attempt-2 build (`d39aa58e`), for review; nothing run.**
- **Amendment:** `docs/superpowers/2026-10-27-mirror-halving-strength-attempt2-amendment.md`, beside the unchanged
  card. `mirror_halving_match` pins its sha256 `99fd9cb3…`; any other amendment refuses.
- **Gate 50:** `MIRROR_HALVING_STRENGTH2_AUTHORIZED`, closed, in its own module. The inventory is 49 → 50, all closed.
- **`run strength2`:**
  - the strength comparison's pairs (509), admission (C1 PASS, screen CONTINUE) and scoring, unchanged;
  - its own plan (`mirror-halving-strength2-run/plan.json`, not yet created) and evidence directory
    (`mirror-halving-strength2/`);
  - the record carries the amendment and its hash.
- **The BOINC check** (`background_compute.py`):
  - one read-only `ps -o pid,user,comm` listing;
  - BOINC means an executable path containing "boinc", or a BOINC account (`boinc_master`, `boinc_project`) running
    outside macOS's system directories;
  - the podman VM (`/opt/podman/bin/`) is exempt by the user's decision and listed in every observation;
  - a `ps` failure or an empty listing is a failed check.
- **Where it runs:**
  - before launch, the wrapper runs it after the idle gate; detection or a failed check refuses, and the observation
    goes to `bg_check.txt`;
  - in the run, before load, after every game and before the verdict, each logged as a `bg_check` event; detection or
    a failed check is a VOID at step `background_compute`;
  - the screen and attempt 1 never call it.
  - On this machine with BOINC quit it reads CLEAR among about 880 processes, with gvproxy and krunkit exempt.
- **A correction made during the build:** the first version matched the full command line. It flagged my own shell,
  whose arguments mentioned BOINC; in a run, a status command such as `pgrep boinc` would have voided the attempt. It
  now matches the executable only, and a test pins that.
- **Copies:**
  - the wrapper is attempt 1's plus the check, its record line, and the new paths, gate and run name;
  - the survey and closing scripts are attempt 1's with only the names changed, asserted byte for byte below the
    docstring.
- **Tests:**
  - 66 → 106 in the four touched files (40 new). They include fake launches of the real wrapper for BOINC absent,
    present and a failed check, mid-run and pre-load VOIDs, an unwired check, the amendment pin, and the gate
    inventory;
  - focused CPU set: 271 passed;
  - no full regression and no mutation campaign.
- **Records:** the build commit's first message said "52 new focused tests". That was not counted, so I corrected it
  before pushing anything (unpushed amend, `d39aa58e`).
- **Not done:** seed procedures, registration, checkpoint loading, GPU work, gameplay, opening gate 50, push.
- **Prediction on record:** none (a build).
- **Status:** built and awaiting review. See do-not-repeat **#103**.

**Addendum to entry 39 (2026-10-27, the user's decision): the attempt-2 build accepted; fresh-seed preparation
approved as one sequence.**
- **Accepted:** the build (`d39aa58e`), with no further BOINC work required. The podman exemption is acknowledged.
- **Approved:** the attempt-2 seed preparation, run once in order with no intermediate approval rounds:
  1. the survey;
  2. the pre-registration proof;
  3. registration of exactly the surveyed block, with its plan;
  4. the post-registration proof.
- **The condition:** each check must pass. Otherwise the sequence stops, with no adjustment or retry.
- **Then:** record the evidence and return once ready to launch.
- **Not authorized:** opening gate 50, checkpoint loading, GPU work, gameplay and pushing.

**Addendum to entry 39 (2026-10-27): attempt 2's fresh seeds prepared in one sequence; every check passed.**
- **Where:** a clean linked worktree (branch `mh-strength2-seeds`, from `b075bfaa`). Each record was committed before
  the next step.
- **Survey (`393c4e27`), run once:** CLEAN, controls 45/45, at a clean tree.
  - The rule's block is [202695700, 202696718): BASE 202695700, 1,018 seeds, above attempt 1's highest seed
    202694617.
  - Direct overlap none, 0 derived streams, injective, nearest gap 1,076 (floor 1,018). Record sha256 `38b7c008…`.
- **Pre-proof (`fa514913`), run once:** HOLDS, controls 45/45. Bound to that survey record; block unregistered, no plan.
- **Registration (`827641cc`):** exactly the surveyed block appended last in ACCOUNTED (45 → 46), with
  `mirror-halving-strength2-run/plan.json` = the survey's `build_plan(202695700, 38b7c008…)`.
  - An AST check against the parent: only ACCOUNTED changed, prefix identical.
  - EXPOSED (43), RETIRED (41), TEST_ONLY and every other statement unchanged.
- **Post-proof (`828e39e4`), run once:** HOLDS, controls 45/45, at a clean tree.
- **Lifecycle test (`7cd199b2`):** the "no seeds yet" test was replaced by the registered-and-schedulable test. It
  checks:
  - the records' provenance, with each proof bound to the survey's sha256;
  - the plan equals `build_plan`, and the block is ACCOUNTED once and last;
  - no EXPOSED, RETIRED or TEST_ONLY overlap, and every seed schedulable;
  - the block lies above attempt 1's;
  - gate 50 closed, and no closing record or run output.

  Focused CPU tests: 271 passed.
- **Ready to launch:**
  - main fast-forwarded and the worktree removed;
  - 50 gates, all closed;
  - the evidence directory `mirror-halving-strength2/` is absent, and the run directory holds only the wrapper and the
    plan;
  - the BOINC check reads CLEAR (846 processes; podman VM exempt); load 1.2.
- **Not done:** opening gate 50, checkpoint loading, GPU work, gameplay, push.
- **Prediction on record:** none (a seed procedure).
- **Status:** registered and proven; awaiting the user's launch approval. See do-not-repeat **#104**.

**Addendum to entry 39 (2026-10-27, the user's decision): attempt 2 approved for launch, once.**
- **The user's checks:**
  - all three seed records, 45/45 controls each, and all 1,018 slot mappings;
  - the exact registration and plan, and unchanged runtime code;
  - all 50 gates closed and no run output.

  No further build or testing is needed.
- **The run, once, limited to:**
  - a fresh read-only panic/reboot check, then a clean run worktree with gate 50 alone opened by a one-line commit;
  - the reviewed `run strength2` wrapper, with BOINC kept quit and the existing checks (idle gate, background-compute
    check before launch and during the run) unchanged;
  - 509 pairs (1,018 games) using exactly [202695700, 202696718), the candidate at 200 simulations versus the
    incumbent at 400, with TF32=0 and eval mode;
  - TERM at 55,591 s and KILL 10 s later.
- **The frozen judgement:** the Hoeffding rule and compute guards of the amendment, unchanged.
- **Forbidden:** interim scoring, an early decision, extension, top-up and any further retry. An incomplete run
  receives no score, and attempt 1's games remain excluded.
- **Return:** the results, wrapper diagnostics, gate restoration and cleanup evidence, for result and closing review.
- **Not authorized:** deployment, training and pushing.

**Addendum to entry 39 (2026-10-28): strength attempt 2, run once → COMPLETED, PASS.**
- **Test:** the frozen combination (MirrorAveragingEvaluator around eval-mode `calib020_0001` at 200 simulations,
  root sequential halving from ply 20 with 16 candidates) against the unchanged incumbent at 400 simulations, over
  509 colour-reversed pairs on the fresh seeds [202695700, 202696718), TF32=0 and eval mode on both sides, with
  BOINC kept quit (amendment `docs/superpowers/2026-10-27-mirror-halving-strength-attempt2-amendment.md`).
- **Hypothesis:** the combination plays more strongly than the incumbent at comparable compute.
- **Why it was plausible:** the screen's CONTINUE (mean pair score 0.5703 over 32 pairs, encouraging but
  inconclusive), with an evaluations ratio of 0.9956 and a search-seconds ratio of 0.8158. Attempt 1's 799 games were
  not read and played no part.
- **Prediction on record:** None stated in advance. The Hoeffding rule is a frozen decision rule, not a prediction.
- **What ran:**
  - approval recorded (`c04cde0b`); pre-run panic/reboot check clean (boot time unchanged since Oct 25 09:39:28; no
    new panic report); BOINC check CLEAR (898 processes; podman VM exempt);
  - clean run worktree, gate 50 opened alone (`babbd858`, pre-open `c04cde0b`), through the reviewed wrapper, once;
  - idle gate: busy at t = 0 (load 4.15; mds and mds_stores at about 50 % CPU), idle held 60 s at t = 257.5 s;
    launch BOINC check CLEAR (888 processes), kept in `bg_check.txt`.
- **Result (`f1f4e2da`):**
  - wrapper outer exit 0, outcome COMPLETED, record written by the script, wall 36,124 s (TERM was due at 55,591 s);
    survivors none; stderr empty;
  - all 1,018 games started and ended, each slot once, every pair colour-reversed; 1,003 wins and 15 state caps
    (draws);
  - candidate 684 wins, incumbent 319, 15 draws; pair scores 219 × 1.0, 10 × 0.75, 236 × 0.5, 5 × 0.25, 39 × 0.0;
  - mean pair score 0.67927; ε = √(ln(40)/(2×509)) = 0.06020; lower bound 0.61908 > 0.5 → **PASS**;
  - guards: network evaluations 11,572,676 / 11,602,032 = 0.9975 (≤ 1.05); search seconds 16,222 / 19,779 = 0.8202
    (≤ 1.10) → compute matched;
  - TF32=0 at start and end; eval mode at load and at end (0 of 42 modules training, 14 BatchNorm, both sides);
  - background checks: 1,020 (before load, after each of the 1,018 games, before the verdict), all CLEAR, no hits,
    no errors;
  - pace 0.604–0.632 s per move in every block of 100 games; load 1.3 before and 2.2 after.
- **Independent check:** recomputed from `games.jsonl`: the 509 pair scores equal the record's, the mean and bound
  match, and the cost totals summed per game equal the record's totals and ratios.
- **Did it match the prediction:** there was no prediction.
- **Reading:** under this protocol (24×24, the existing rules, 200 averaged-and-halved simulations against 400
  ordinary ones, the established opening-temperature readout), the fixed combination beats the unchanged incumbent
  within the declared cost allowances. The bound assumes independent pairs. The run shows neither synergy nor either
  component's benefit, and says nothing about other simulation counts, opponents or the browser product.
- **Post-run checks:**
  - boot time unchanged; no panic report; BOINC check CLEAR (854 processes);
  - gate 50 restored by the wrapper (`gate_restored` true) and equal to `c04cde0b`; 50 gates, all `False`;
  - worktree committed by exact paths, main fast-forwarded to `f1f4e2da`, worktree and branch `mh-strength2-run`
    removed;
  - focused tests: 105 passed and 1 failed: `test_ATTEMPT_2_S_PLAN_is_the_survey_s_build_plan_and_its_block_is_REGISTERED_and_SCHEDULABLE`
    pins the pre-run state (no run output). Expected, and left unedited: re-anchoring belongs with the closing, as
    for attempt 1.
- **Not done:** seed closing (block [202695700, 202696718) consumed, all 1,018 started; not yet EXPOSED or RETIRED),
  test re-anchoring, deployment, training, push.
- **Status:** PASS; the strength line's frozen question is answered. Result and closing review are the user's. See
  do-not-repeat **#105**.

**Addendum to entry 39 (2026-10-28, the user's decision): attempt 2's PASS accepted; the block's closing approved.**
- **Accepted:** the PASS, independently recomputed by the user: mean pair score 0.67927, Hoeffding lower bound
  0.61908 > 0.5, evaluations ratio 0.9975 and search-seconds ratio 0.8202. All 1,018 games, seed bindings, pair
  scores and cost totals reconcile. All 39 module hashes match the execution commit, the 1,020 background checks
  were clear, and all 50 gates are closed.
- **What it establishes:** stronger play at comparable compute for this fixed combination under the tested protocol,
  assuming independent pairs. It does not establish synergy or either component's benefit. No additional match is
  needed to answer the frozen question.
- **Approved, once:**
  1. the existing K1 derivation (`mirror-halving-strength2-seed-closing/01_derive_usage.py`), run once;
  2. only if it RECONCILES with every control passing: append (202695700, 202696718) last to EXPOSED (43 → 44) and
     RETIRED (41 → 42), with ACCOUNTED left at 46 and every other registry statement unchanged;
  3. re-anchor the lifecycle test to full exposure, retirement and unavailability, and run the affected focused CPU
     tests only.
- **Ledger:** the accepted PASS and the actual closing records are appended; attempt 1's void record is unchanged.
- **Not authorized:** deployment, training and pushing.

**Addendum to entry 39 (2026-10-28): attempt 2's block closed.**
- **What ran (each once, in a clean worktree at `7bacc265`):**
  - K1 (`af838eb2`): `mirror-halving-strength2-seed-closing/01_derive_usage.py` → RECONCILES, controls 15/15. It read
    outer exit 0, no survivors, status COMPLETED, and game starts 0–1,017 in order. Verdict: EXPOSE [202695700,
    202696718); RETIRE [202695700, 202696718). Record `02_derive_usage_run.txt` sha256
    73a71da8e738f2f7917c164c2cf01797afaf9dab168e679db55203ad32a780c6.
  - K2 (`b563792d`): (202695700, 202696718) appended last to EXPOSED (43 → 44) and RETIRED (41 → 42), each with a
    comment block. An AST check against the parent showed only those two statements changed, with identical
    prefixes; ACCOUNTED (46), TEST_ONLY and every other statement unchanged. Every seed of the block is now
    accounted, exposed, retired and unavailable.
  - Test (`3ffc9924`): `test_ATTEMPT_2_S_PLAN_…_REGISTERED_and_SCHEDULABLE` renamed `…_REGISTERED_and_CLOSED` and
    re-anchored to the K1 verdict line and controls, full exposure and retirement (once, last), every seed refused,
    and the run outputs and K1 record present. Affected focused CPU tests: 174 passed (registry, seed selection,
    the mirror-halving launch, match and combination tests, the background check and the gate inventory).
- **Prediction on record:** none (a closing procedure).
- **Unchanged:** attempt 1's void record, its closed block and its 799 unscored games; the PASS record and its
  evidence; 50 gates, all closed.
- **Not done:** deployment, training, push.
- **Status:** the strength line's frozen question is answered (PASS, accepted) and its seeds are closed. Any further
  use of the combination needs a new decision. See do-not-repeat **#106**.

### Decision note — after entry 39: a combination policy pilot, card frozen, step 1 built only (2026-10-28, the user's decision)

- **The proposal:** `docs/superpowers/2026-10-28-combination-policy-pilot-card.md`, first drafted as a proposal and
  amended once after the first review. One round of policy-only learning from the proven agent's own halving decisions:
  - weights-only `calib020_0001` (SHA-1 checked; no `--resume`), the encoder, value head and every BatchNorm running
    statistic frozen, so the value output is bit-identical by construction rather than by an anchor;
  - arm R's residual policy branch (8,840 parameters, `policy_head_capacity.ResidualPolicyHead`) trained on the CPU;
  - the target: within halving's 16 first-round candidates, π = P_C · softmax(log p + (50 + N_max)(q + 1)/2); outside
    them π = p (halving's visit counts are a schedule and are never the target);
  - 1,000 fresh self-play games of the proven agent on a registered block, then a 32-pair screen and a 509-pair strength
    comparison against the proven agent with identical search.
- **First review (amendments requested):** freeze one objective, the mirror-averaged policy, for the pre-fit
  measurement, the training and the held-out check; assign each game's seed by slot, independent of worker
  scheduling, keeping the match's four streams, with fresh registered blocks for the throughput check and generation;
  remove the claim that the 519-parameter head is too small; correct the CPU rationale (GPU training is unqualified;
  this recipe is not claimed to reproduce #75 under TF32=0); fix the initialization and sampling seeds (20261028,
  20261029); state hard stage ceilings before each execution approval.
- **Second review (this decision):** the amendments resolve both findings. Approved:
  - freezing the card;
  - step 1's build only: the minimal implementation and the proposed focused CPU tests, with temporary outputs and
    fake evaluators;
  - the 20-update CPU timing with synthetic inputs and random weights, with MLX evaluation synchronized so it measures
    completed updates;
  - a narrow **#66 exception**: reusing the residual head for this pilot, with no capacity claim.
- **Two clarifications, put into the frozen card without another review:** the comparison of p̄_θ₀ with the stored
  prior is a *probability discrepancy*, not a numerical floor on the KL (a maximum probability error does not bound the
  KL error); and all 144 throughput-check games must finish inside the 15-minute ceiling, or the run is VOID and the
  pilot closes, unless separately amended.
- **As accepted:** a distinct proposed learning recipe. The closed continuation (#48, #49) and D2 (#69, #87) trials
  remain closed. The constructed target is the main scientific uncertainty -- it can heavily favour a move eliminated
  after three visits -- and the combination's PASS does not validate it. 18 workers and arbiter cap 14 are reasonable
  starting settings, not established optima for this search.
- **Prediction on record (card §5, written before any run):** directional only -- the candidate's held-out
  cross-entropy falls, and it scores above 0.5 against the proven agent. No magnitude; low confidence.
- **Not requested:** additional experimental arms, a broad regression or mutation campaign, a numerical investigation.
- **Still unauthorized:** seed procedures, checkpoint loading, GPU work, corpus generation, real-data training, games
  and pushing. See do-not-repeat **#107**.

### 40. Combination policy pilot — step 1 built; the CPU timing (2026-10-28, COMPLETED)

- **Test:** step 1 of the frozen card (`docs/superpowers/2026-10-28-combination-policy-pilot-card.md`, sha256
  `237424bd…`): the build, its focused CPU tests, and the one-shot CPU timing of 20 training updates at the real
  architecture (30 / 128 / 6, arm R installed) on synthetic inputs with random weights, which prices step 4's ceiling.
- **Hypothesis:** none; this is a build and a cost measurement, not an experiment on the target.
- **Why it was needed:** the card's training-time estimate (§8: about 2–6 h for 3,000 × 64) rested on a CPU throughput
  that had never been measured, and step 4's hard ceiling must be stated before its approval.
- **Prediction on record:** the card's §8 cost estimate, about 2–6 h for CPU training, written before the timing. No
  per-update figure was stated in advance.
- **What was built** (`b1ad8f00`, plus the BOINC guard `c0bf2452`; card §10 lists the deviations from the code-change
  plan):
  - `halving_policy_target.py` -- the target, read from the root of the same search that chose the move;
  - `combination_selfplay.py` -- slot-seeded self-play that calls the strength match's own game loop (the four seed
    streams, the readout and `last_choice` are the match's code), through one inference server and spawned CPU
    workers; slot g plays seed base + g whichever worker claims it;
  - `combination_policy_pilot.py` -- `run throughput|generate|train` behind three new gates, all closed (50 → 53), and
    `time`, the timing, which refuses while BOINC runs and is VOID if BOINC appears by its end;
  - no change to `self_play.py`, `self_play_worker.py`, `trainer.py`, `train.py`, `mcts.py` or any frozen module.
- **Tests:** `tests/test_combination_policy_pilot.py`, 52 CPU tests with fake evaluators and temporary outputs, plus the
  gate inventory: 59 passed, 1 deselected (the opt-in end-to-end arbiter test, which passed once in 503 s). Earlier,
  with the other suites that import the gate inventory: 341 passed.
- **What ran:** `MLX_ENABLE_TF32=0 .venv/bin/python -B -m scripts.GPU.alphazero.combination_policy_pilot time`, once, at
  `c0bf2452`. A first launch attempt earlier the same day was not made: BOINC was running (16 processes, load about
  15), so the guard was added instead. On the run: the user quit BOINC; the background check reported no BOINC process
  (the podman VM exempt) before and after; the launch waited for the 1-minute load to fall below 3 (it read 3.13 at the
  start and 3.54 at the end; the 5- and 15-minute averages still carried BOINC's earlier run). 13.5 s wall.
- **Result** (`docs/superpowers/evidence/combination-policy-timing/record.json`, COMPLETED):
  - 20 synchronized updates (after 2 warm-up), each both orientations through the full network, the loss, the
    gradient and the Adam step: mean **0.487 s**, range 0.481–0.498 s (max / mean 1.02);
  - a forward pass of the averaged prediction: 7.51 ms per row (both orientations); `to_tensor` 0.117 ms per state;
  - projection over the card's estimated rows: updates 1,462 s, pre-fit and held-out passes 306 s, value check 28 s,
    corpus tensors 4 s; **total 1,799 s (30.0 min)**; **step 4's ceiling: 45 min** (1.5×, rounded up to the minute).
  - Not timed: checkpoint loading, the corpus replay with target re-verification, and the post-fit checks.
- **Did it match the prediction?** No: the projection (30 min) is well under the card's 2–6 h estimate. The estimate
  assumed a CPU throughput that was not measured; the timing replaces it, as §8 said it would. No cause beyond the
  measured rate is claimed.
- **Conclusion / status:** step 1 is complete and awaits the user's review. The 45-minute ceiling is the price for
  step 4 when that is requested; it rests on estimated row counts, and the untimed replay must fit inside its margin.
  Unchanged risk for step 2: all 144 throughput games must finish in 15 minutes (about 576 games/h against a 350–700
  estimate), or the pilot closes. Seeds, checkpoint loading, GPU work, generation, real-data training, games and
  pushing remain unauthorized. See do-not-repeat **#108**.

### Decision note — after entry 40: step 1 accepted; the throughput check's wrapper and seed copies approved to build (2026-10-28, the user's decision)

- **Review of step 1 (build `b1ad8f00`, `c0bf2452`; timing `ad3a4cb6`):** accepted, with no material blocker found in
  the implemented portion. The user verified: the frozen card's and the pinned components' hashes; that calling the
  proven match loop directly keeps its search and readout and leaves default self-play unchanged; that the averaged
  objective, the slot seeds and the policy-only training surface match the approved design; all 53 gates closed; and
  the timing arithmetic (1,799.35 s projected, so step 4's ceiling is 45 minutes).
- **The timing record stands.** Its 1-minute load readings above 3 (3.13 at the start, 3.54 at the end) do not
  invalidate it under its declared checks; no rerun. It measures synthetic CPU cost, not training reliability or
  improvement.
- **Approved now: build only** the throughput check's wrapper and its seed-survey (with the pre- and post-registration
  proofs) and closing copies, following the existing patterns, with focused CPU checks only. They return for review
  before any seed procedure is executed or any GPU work is launched.
- **The frozen requirement is unchanged:** 144 completed games within 15 minutes. A timeout would reject this setup's
  operational feasibility; it would say nothing about whether policy learning works.
- **Still unauthorized:** checkpoint loading, executing any seed procedure, GPU work, generation, real-data training,
  games and pushing. See do-not-repeat **#109**.

#### Addendum — the throughput check's launch, built for review (2026-10-28)

- **Built only** (no seed procedure run, no gate opened, no checkpoint loaded, no GPU), each a copy of strength attempt
  2's, its differences named in its header and pinned by a test:
  - `docs/superpowers/evidence/combination-policy-throughput-run/01_wrapper.sh` -- the paths, the gate (still closed),
    `run throughput`, the record kind, and **the card's 15-minute ceiling: TERM at 890 s, KILL 10 s later** (the
    precedent counts the ceiling to the KILL). **The run's process group:** the run spawns 18 workers whose command
    lines do not name the module, so attempt 2's name-only survivor search would miss one. The supervisor writes the
    group id, create-only, to `pgid.txt`; survivors are the named processes plus everything left in that group; a
    started run with no readable `pgid.txt` records survivors as unknown, so closing refuses. A supervisor that cannot
    write `pgid.txt` KILLs the group and exits 126 (CRASHED).
  - `combination-policy-throughput-seed-survey/01_seed_survey.py` -- 144 slots, the paths, the plan's kind, the
    titles and verdict lines, and slot lines without pairs (`SLOT g seed BASE + g`). The pinned method, the rule and
    the three modes are unchanged; its four XOR masks are exactly the game's four streams.
  - `combination-policy-throughput-seed-closing/01_derive_usage.py` -- card §5's one diff: the `game_start` slots
    (field `slot`) must form {0..k-1} as a set, each once, and the one `claims` event must equal k; otherwise k is not
    established and the whole block is exposed. A slot outside the block is a fault; a REFUSED run has k = 0. The
    whole block is retired whatever happened. 21 built-in controls.
- **Tests:** `tests/test_combination_policy_launch.py`, 20 CPU tests. The copies' diffs against attempt 2's files are
  exactly the named ones. Fake launches of the real wrapper bytes cover: busy, BOINC or a failed check refuse with
  nothing written and the gate restored; an idle launch reaches `run throughput --checkpoint <abs>` with TF32=0 and
  records the group; a process left in the group is reported as a survivor; with TERM_S changed to 2 in a test copy
  only, the ceiling ends the run and its worker and the wrapper writes a VOID timeout record; a lost `pgid.txt` gives
  unknown survivors. The closing copy reconciles the progress the real `_generate` writes through the real record
  boundary, both completed with starts out of order and VOID mid-run with no claims (whole block exposed). With the
  pilot, gate-inventory, mirror-halving launch and background-compute suites: 129 passed, 1 deselected (opt-in).
- **Defects found and fixed before commit:** a Perl scoping error in the new `pgid.txt` write (the supervisor would
  have crashed every launch; caught by the fake launch, which also showed the fail-closed path working), and a survivor
  list that would have read as non-empty when empty.
- **Still unauthorized:** the survey, the proofs, registration, the plan, opening the gate, the checkpoint, the GPU,
  the launch, closing, and every later stage. See do-not-repeat **#109**.

#### Addendum — the build reviewed; the throughput seed survey approved once; one supervisor repair approved (2026-10-29, the user's decision)

- **The review (of `fd683f52`):** the seed scripts are ready. The user checked that TERM at 890 s plus 10 s grace is
  consistent with the stated ceiling, that the survey's pinned methods match, that the parallel closing rule matches
  the card (including whole-block exposure when usage cannot be established), and that all 53 gates remain closed
  with no plan or run output present.
- **One wrapper defect, to be fixed before any launch.** In the supervisor, `waitpid` returning for the main process
  cancels the alarm at once. If the main process exits on TERM but a worker ignores TERM, the scheduled KILL never
  happens; detecting that survivor afterwards does not enforce the 900-second ceiling.
- **Approved: the repair and one focused CPU test.** The supervisor must keep escalation active while members of the
  run's group survive. One fake-launch case: the main process exits on TERM, its worker ignores TERM, and KILL removes
  that worker within the ceiling. No broader campaign.
- **Approved: the throughput seed survey, once,** in a clean worktree, with the fixed selection rule and create-only
  evidence. Its record is committed and returned for review before the pre-proof. The repair may proceed
  independently.
- **Still unauthorized:** the pre-proof, registration, the plan, the post-proof, opening the gate, checkpoint loading,
  the GPU launch, closing, and every later stage. See do-not-repeat **#110**.

#### Addendum — the throughput seed survey run once (CLEAN); the supervisor repaired (2026-10-29)

- **The survey (`cd4217a9`), run once** in a clean linked worktree (branch `cp-throughput-seeds`, from `fea251fa`;
  main fast-forwarded and the worktree removed afterwards):
  `.venv/bin/python -B docs/superpowers/evidence/combination-policy-throughput-seed-survey/01_seed_survey.py --mode
  survey`, exit 0. A first launch command was refused by the shell before Python started (a mistyped stderr path), so
  nothing ran and no record existed; the command was corrected and run once.
  - Verdict: **THE COMBINATION-POLICY THROUGHPUT BLOCK IS CLEAN**, controls **45/45**, clean tree at `fea251fa`; the
    main checkout scanned too (49,810 files; its 2 untracked files are not required clean).
  - The rule's block is **[202696900, 202697044)**: BASE 202696900, 144 seeds, above the highest registered seed
    202696717 (attempt 2's last). Direct overlap none, 0 derived streams, injective, nearest gap 176 (floor 144).
  - Slot g plays seed BASE + g, slots 0–143 listed. Record
    `combination-policy-throughput-seed-survey/02_seed_survey_run.txt`, sha256 `dbe3e91c…`.
  - **Not registered:** no proof run, ACCOUNTED unchanged, no plan.
- **The supervisor repair** (approved in the previous addendum): once it has reaped the main process, the supervisor
  now keeps its alarm armed and waits until no process of the run's group survives (`kill 0` to the group, every
  0.1 s). A worker that ignores TERM is therefore KILLed at the ceiling. A group that outlives its main process past
  890 s is a TIMEOUT (124, or 137 if KILL was needed), whatever the script recorded; the closing then still exposes
  the seeds the progress shows used and retires the block.
  - **The focused test** (fake launch, TERM_S 2 in a test copy only, so a 12 s ceiling): the main process dies of the
    group TERM, its worker ignores TERM, and KILL removes it; the wrapper returns 137 after 12 s or more, the worker
    is gone, outer TIMEOUT with no survivors, the wrapper's VOID timeout record. **It failed on the unrepaired
    wrapper** (the worker survived) and passes on the repaired one.
  - **Two existing tests changed, because the behaviour changed:** the wrapper-diff pin (one more added line), and
    the group-survivor test. The worker can now outlive the supervisor only if the supervisor itself is lost, so its
    stub now KILLs the supervisor; the wrapper's group search must still report the worker. The "nothing has run"
    test now expects the survey record and nothing after it.
  - `tests/test_combination_policy_launch.py`: 21 passed (three consecutive runs). With the pilot, gate-inventory,
    mirror-halving launch and background-compute suites: 130 passed, 1 deselected (opt-in).
  - **Not covered by a test:** the path where the main process exits normally while a worker lingers; it uses the
    same wait loop. A process that survives KILL (uninterruptible) would hold the supervisor, as an unkillable main
    process already would.
- **Prediction on record:** none (a seed procedure and a repair).
- **Status:** the survey record and the repaired wrapper await the user's review. The pre-proof, registration, plan,
  post-proof, gate, checkpoint, GPU launch and closing remain unauthorized. See do-not-repeat **#111**.

#### Addendum — the survey accepted; the pre-registration proof approved once; the post-KILL wait to be bounded (2026-10-29, the user's decision)

- **The survey accepted** (`cd4217a9`). The user verified the record's hash, all 144 slot lines, the 45/45 recorded
  controls, the pinned method hashes and the unchanged registry; no plan or proof exists; all 53 gates closed.
- **The supervisor repair accepted** (`50e10824`): it addresses the TERM-ignoring worker, and the new test covers
  that case.
- **Approved: the pre-registration proof, once,** on the committed survey record and the unchanged block
  [202696900, 202697044). Its create-only record is committed and returned before registration.
- **One disclosed issue to fix before launch, approved independently of the pre-proof:** the post-exit wait loop can
  wait forever after KILL. The observation must be bounded, and must then report the surviving processes or
  unconfirmed cleanup instead of hanging, so the wrapper stays bounded. The minimal repair and a focused CPU check.
- **Ruling: an outer TIMEOUT overrides an inner COMPLETED record.** Such a run closes the pilot and cannot admit
  generation. Both original records (the script's `record.json` and the wrapper's `outer.json`) are preserved.
  - **Not yet enforced in code:** `combination_policy_pilot.admitted("generate")` reads only the throughput
    `record.json` status. Changing it was not part of this approval; it is recorded here as an open item.
- **Still unauthorized:** registration, creating the plan, the post-proof, opening the gate, checkpoint loading, the
  GPU launch and closing. See do-not-repeat **#112**.

#### Addendum — the throughput pre-registration proof run once (HOLDS); the post-KILL wait bounded (2026-10-29)

- **The pre-proof (`086d6e37`), run once** in a clean linked worktree (branch `cp-throughput-pre`, from `7aa0aea8`; main
  fast-forwarded and the worktree removed afterwards): `--mode pre`, exit 0.
  - Verdict: **THE COMBINATION-POLICY THROUGHPUT BLOCK IS CLEAN AND THE STATE IS AS EXPECTED**, controls **45/45**,
    clean tree at `7aa0aea8`.
  - Bound to the committed survey record (sha256 `dbe3e91c…`, read at `fea251fa`); the block read from it,
    [202696900, 202697044), BASE 202696900; EXPECTED STATE (pre) HOLDS: the registry is the one the survey read
    (sha256 `06e442c0…`; ACCOUNTED 46, EXPOSED 44, RETIRED 42), the block unregistered, no plan.
  - Direct overlap none, 0 derived streams, injective, nearest gap 176 (floor 144). Record
    `combination-policy-throughput-seed-survey/03_collision_proof_pre_run.txt`, sha256 `5f60b891…`.
- **The bounded wait** (approved in the previous addendum):
  - **What a test could and could not reproduce:** no real process survives KILL here. On this macOS a group holding
    only an unreaped zombie answers `kill 0` with EPERM, which the supervisor reads as empty, so a zombie cannot
    stand in, and late-joining processes would race the KILL.
  - **The supervisor:** one polling loop now watches both the main process (`waitpid` WNOHANG, replacing the
    blocking `waitpid`, so an unkillable main process is bounded too) and its group. The alarm has a third stage:
    TERM at 890 s, KILL 10 s later, and 10 s after that the watch ends. If the group can still be signalled then, the
    supervisor exits **125**, which the wrapper maps to TIMEOUT. The wrapper never waits past 910 s.
  - **The wrapper's report:** survivors read `unconfirmed cleanup: process group <pgid> <pids>` when the supervisor
    gave up (125), or when the wrapper's own `kill -0` finds the group still there once the supervisor has gone (a
    lost supervisor, or the wrapper interrupted, whose trap never sees the supervisor's exit code). Closing refuses
    on any survivors, and no wrapper record is written over them.
  - **The focused test:** a copy whose TERM_S is 2 and whose group probe also counts a sentinel process outside the
    group, which KILL cannot reach: to the supervisor, a group that never empties. The wrapper returns 125 after about
    22 s; outer TIMEOUT, survivors `unconfirmed cleanup: process group <pgid>`, no wrapper record, the real group
    killed, the gate restored. **On the previous wrapper (`50e10824`) the same test hung until its 60 s timeout.** The
    lost-supervisor test now also checks the `unconfirmed cleanup` prefix, which covers the wrapper's own check.
  - The "nothing has run" test now expects the survey and pre-proof records. `tests/test_combination_policy_launch.py`:
    22 passed (three consecutive runs). With the pilot, gate-inventory, mirror-halving launch and background-compute
    suites: 131 passed, 1 deselected (opt-in). No test process left behind.
- **Still open:** the user's ruling that an outer TIMEOUT overrides an inner COMPLETED record is not enforced by
  `admitted("generate")` (it reads only `record.json`); changing it needs approval.
- **Prediction on record:** none (a seed procedure and a repair).
- **Status:** awaiting the user's review. Registration, the plan, the post-proof, the gate, the checkpoint, the GPU
  launch and closing remain unauthorized. See do-not-repeat **#113**.

#### Addendum — the pre-proof and the bounded supervisor accepted; registration with its plan, and the admission fix, approved (2026-10-29, the user's decision)

- **Accepted:** the pre-proof (`086d6e37`) and the bounded supervisor repair (`c3b2a6a2`). The user verified the
  proof's hash, its binding to the survey, all 144 slots, the 45/45 recorded controls and the unchanged registry; the
  block is unregistered, no plan exists, and all 53 gates are closed. The extra 10 s after KILL are cleanup
  observation, not run time.
- **Approved, two independent changes, each returned for review:**
  1. **Registration and plan:** append exactly [202696900, 202697044) to ACCOUNTED, leaving every other list
     unchanged, and create the plan through the existing `build_plan`, bound to the accepted survey hash
     (`dbe3e91c…`).
  2. **The admission fix, CPU tests only:** generation must require both the committed inner record and the
     committed `outer.json` to show successful completion: inner and outer exits zero, the gate restored, survivors
     confirmed empty. Missing outer evidence, a TIMEOUT or unconfirmed cleanup refuses admission, even when the inner
     record says COMPLETED. Focused tests cover those cases.
- **Git note from the review:** the local `origin/main` reflog shows a push through `50e10824` at 10:52 EDT on
  2026-10-29, which accounts for "3 ahead"; the review could not attribute its authorization. For the record: the user
  answered "push" in this session after being asked whether to run `git push origin main`, and that single
  fast-forward (`e5f3055e..50e10824`) was the push. This review authorizes no push.
- **Still unauthorized:** the post-proof, opening the gate, checkpoint loading, the GPU launch, closing and pushing.
  Existing evidence is preserved. See do-not-repeat **#114**.

#### Addendum — the throughput block registered with its plan; generation admission now requires the outer record (2026-10-29)

- **Registration (`88c5d162`),** in a clean linked worktree (branch `cp-throughput-reg`, from `158f12ba`; main
  fast-forwarded and the worktree removed afterwards):
  - The block READ from the accepted survey record (sha256 `dbe3e91c…` recomputed; the survey copy's own
    `read_record`), not typed: [202696900, 202697044), BASE 202696900.
  - ACCOUNTED: exactly that block appended last (46 → 47), with a comment in the precedent's form. An AST check
    against the parent: only ACCOUNTED changed, its prefix identical; EXPOSED (44), RETIRED (42), TEST_ONLY and every
    other statement unchanged; no list overlaps the block.
  - `combination-policy-throughput-run/plan.json` = the survey copy's `build_plan(202696900, dbe3e91c…)`, written
    create-only: kind `combination-policy-throughput-plan/1`, base 202696900, n_slots 144.
- **Tests re-anchored (`f20c47cf`):** the launch file's state test now checks the survey and pre-proof provenance
  (the proof bound to the survey's sha256), plan = `build_plan`, the block ACCOUNTED once and last, unexposed,
  unretired and every seed schedulable by the pilot's own predicate, no post-proof, the gate closed and no run output.
  Strength attempt 2's state test asserted its block was ACCOUNTED's last entry; it now requires every later entry to
  lie above it.
  - Every test file that reads the registry (41 files), run at the parent `158f12ba` and after registration: the
    same 16 failures and 69 errors in both, none new. They predate this work (tests needing the checkpoint, absent
    in a linked worktree, and older seed-closing tests asserting their block is the last EXPOSED/RETIRED entry).
- **The admission fix** (`combination_policy_pilot.py`):
  - `OUTER_ADMISSION`: generation also needs the throughput wrapper's `outer.json`. `outer_faults` refuses unless
    `inner_exit` 0, `outer_exit` 0, outcome COMPLETED, `gate_restored` true, survivors exactly empty and
    `record_by` "script", each of the right type. **Beyond the approval's list:** the outer `head` must equal the
    inner record's `head`, so both files come from one run.
  - Missing outer evidence refuses, whatever the inner record says. `run` now passes the real outer reader.
  - **The committed reader is tightened** for both files: it now also requires the file to be tracked, so an
    ignored, never-committed file no longer passes as committed (`git status` is silent about ignored files).
  - **Before the fix,** the old `admitted("generate")` admitted a COMPLETED inner record with no outer evidence at
    all (checked directly). Of the 25 new unit tests, 24 fail on the old code.
  - **Tests:** unit cases in `tests/test_combination_policy_pilot.py`:
    - success, plus 17 refusals: TIMEOUT 124/137, 125 unconfirmed, the group remaining, unknown or listed
      survivors, gate not restored, outer exit 5, a wrapper or missing record, non-integer exits, VOID, another or
      no head;
    - missing outer evidence;
    - only generation reads an outer record;
    - the committed reader on tracked, absent, untracked, ignored and modified files.

    In `tests/test_combination_policy_launch.py`, the real wrapper's own outputs go through the real committed
    readers:
    - a completed fake run with the script's COMPLETED record is refused until committed, then admitted;
    - a fake run whose script wrote COMPLETED and then outlived a 2 s ceiling (in a test copy only) ends 124
      TIMEOUT, both records stand as written, and admission refuses it;
    - the VOID, KILL, unconfirmed-cleanup, lost-supervisor and unknown-pgid outputs all fail `outer_faults`.

    Those two end-to-end tests fail on the old code only because the outer reader did not exist.
  - Results: 99 passed (three consecutive runs). With the gate-inventory, mirror-halving launch, background-compute,
    seed-selection and tactical-survey suites: 191 passed, 1 deselected (opt-in).
- **Prediction on record:** none (a seed procedure and a fix).
- **Status:** both await the user's review. The post-proof, opening the gate, checkpoint loading, the GPU launch,
  closing and pushing remain unauthorized. See do-not-repeat **#115**.

#### Addendum — registration, plan and admission fix accepted; the post-registration proof approved once (2026-10-29, the user's decision)

- **Accepted:** the registration and plan (`88c5d162`, tests `f20c47cf`) and the admission fix (`c38e9483`). The user
  independently verified that ACCOUNTED holds exactly the new block appended last with every other registry statement
  unchanged; that the plan equals `build_plan` bound to the accepted survey hash; that admission now refuses missing
  outer evidence, TIMEOUT and unconfirmed cleanup despite an inner COMPLETED record; that the added matching-head and
  tracked-file checks are appropriate; and that all 53 gates are closed with no post-proof or run output. The
  pre-existing registry-test failures need no repair for this step. The user thanked us for the push attribution.
- **Approved, once:** the post-registration proof (`--mode post` of the throughput survey copy), in a clean linked
  worktree, against the committed registration and the unchanged plan. Its create-only record is committed, with the
  necessary lifecycle-test update and focused CPU checks, and returns for review before the gate is opened.
- **Still unauthorized:** opening the gate, checkpoint loading, the GPU launch, closing and further pushing. Existing
  evidence is preserved. See do-not-repeat **#116**.

#### Addendum — the throughput post-registration proof HOLDS (2026-10-29)

- **What ran, once:** `01_seed_survey.py --mode post` of the throughput survey copy, with `.venv/bin/python -B`, in a
  clean linked worktree (branch `cp-throughput-post` at the decision commit `d28f5baf`); exit 0, stderr empty. The
  record was committed there (`20486339`), main fast-forwarded and the worktree removed.
- **Result** (`combination-policy-throughput-seed-survey/04_collision_proof_post_run.txt`, sha256 `03d04570…`):
  - HEAD `d28f5baf`, working tree clean; the main checkout scanned too.
  - Bound to the survey record (sha256 `dbe3e91c…` recomputed, read at `fea251fa`); the block read from it:
    [202696900, 202697044), BASE 202696900.
  - **EXPECTED STATE (post): HOLDS.** The registry it read (sha256 `71770f79…`, ACCOUNTED 47, EXPOSED 44, RETIRED 42,
    TEST_ONLY 1) is the one committed at `88c5d162`, and the registry and plan are unchanged since then: ACCOUNTED is
    the survey-time registry's with exactly the block appended last; every other list and statement is unchanged; the
    plan is committed and equals `build_plan`.
  - Direct overlap none, 0 derived streams, injective, nearest gap 176 (floor 144), CLEAN. Controls 45/45.
  - **PROOF VERDICT (post): THE COMBINATION-POLICY THROUGHPUT BLOCK IS CLEAN AND THE STATE IS AS EXPECTED.**
- **Lifecycle test** (`tests/test_combination_policy_launch.py`): the state test failed once the record existed
  ("one more item: 04_collision_proof_post_run.txt"), as it should. It is re-anchored and renamed
  (`…_REGISTERED_AND_PROVEN_…`): both proofs are bound to the survey hash, run on a clean tree, HOLD with 45/45 and end
  with the verdict; the post-proof names the plan's block; and the registry hash it proved must equal the current
  registry file's, so any registry change before the run fails the test until it is proven again. The gate closed, no
  closing record and no run output are still asserted.
- **Checks:** launch + pilot: 99 passed, 1 deselected (three consecutive runs). With the gate-inventory, mirror-halving
  launch, background-compute, seed-selection and tactical-survey suites: 191 passed, 1 deselected (opt-in). The
  gate inventory derives 53 gates, all closed. No test process left behind.
- **Observed, not changed:** as in the strength-2 precedent, the pilot's own launch admission does not read the
  post-proof; the lifecycle test is what guards it.
- **Prediction on record:** none (a seed procedure).
- **Status:** awaiting the user's review. Opening the gate, checkpoint loading, the GPU launch, closing and pushing
  remain unauthorized. See do-not-repeat **#117**.

#### Addendum — the post-proof accepted; one throughput launch approved (2026-10-29, the user's decision)

- **Accepted:** the post-proof (`20486339`) and the lifecycle-test update (`05774c93`). The user verified the record's
  hash, its binding to the survey, the 45/45 recorded controls and all 144 slots; the proved registry and plan match
  the current committed files; the card and component pins match; all 53 gates are closed.
- **Not a blocker for this explicitly reviewed run:** the launch code does not read the post-proof. Instead, the
  registry and plan are rechecked as unchanged before the gate opens; any change returns for review, not to an
  automatic proof rerun.
- **Approved, once:**
  - record this approval; a fresh read-only panic/reboot check; a clean linked worktree;
  - open **only the throughput gate** with the one-line commit;
  - run the existing wrapper once, BOINC quit and both launch checks (idle gate, background-compute check) enforced.
    This authorizes the checkpoint loading and GPU work for these **144 throughput games only**;
  - keep TERM at 890 s, KILL at 900 s and the bounded cleanup observation through 910 s;
  - restore the gate, verify cleanup, commit the exact evidence paths and return for review.
- **Rules:** a started run gets no retry, extension or top-up. An outer TIMEOUT overrides an inner COMPLETED record
  and closes the pilot. The throughput games stay excluded from training and strength evidence.
- **Still unauthorized:** closing, corpus generation, real-data training and pushing. See do-not-repeat **#118**.

### 41. Combination policy pilot — step 2, the throughput check (2026-10-29, outer TIMEOUT; VOID; the pilot CLOSED)

- **Test:** card §5 step 2 (`docs/superpowers/2026-10-28-combination-policy-pilot-card.md`): 144 slots on the
  registered block [202696900, 202697044), 18 workers, calib020_0001 on the GPU with `MLX_ENABLE_TF32=0`, inside a
  15-minute ceiling (TERM at 890 s, KILL at 900 s, cleanup observation to 910 s). All 144 games must finish inside
  it; a run the ceiling stops is VOID and closes the pilot. Its games are never used.
- **Hypothesis:** the combination self-play runs fast enough at 18 workers for step 3's 1,000-game generation to be
  priced from it.
- **Why it was plausible:** card §7 estimated about 350–700 games/h (strength 2: about 32 s per single-process
  combination game; the arbiter's batching at 18 workers unknown).
- **Prediction on record:** card §7's 350–700 games/h estimate (written before any run). The build's unresolved
  issues (card, "Unresolved issues found in the build"; entry 40) recorded that 144 games in 900 s needs about
  576 games/h, so the 15-minute rule might close the pilot on throughput alone.
- **What ran:** decision `bdb691dc` (#118). A fresh read-only panic/reboot check (no reboot since 2026-10-25 09:39; the
  only panic report is that one) and a recheck: registry sha256 `71770f79…` = the post-proof's, registry and run
  directory unchanged since `88c5d162`; BOINC check CLEAR (podman exempt); load about 1.2. Clean linked worktree
  `cp-throughput-run`; the one-line gate opening `60493787` (only `COMBINATION_POLICY_THROUGHPUT_AUTHORIZED`); the
  existing wrapper run once at 20:03:33 EDT. Idle gate held 60 s, background check CLEAR, then launch. Evidence
  `dba747bf`.
- **Result:**
  - **Wrapper:** inner exit **124** (the ceiling fired and TERM sufficed), outer exit 124, outcome **TIMEOUT**,
    `gate_restored` true, survivors empty, `record_by` script, wall 890 s. The gate file is identical to the pre-open
    commit; no pilot process remained; pgid 71921 was empty.
  - **Inner record:** status **VOID**, step "terminated", "SIGTERM after 889.9 s"; `throughput` null, verdict null. So
    the inner and outer records agree (no COMPLETED to override). TF32 "0" at start and end; the model loaded with 0
    of 42 modules in training mode; 114 BOINC checks, all clear; stderr empty.
  - **Slots:** 131 started (exactly slots 0–130, a prefix), 113 finished, 18 in flight at TERM (running 1–327 s);
    slots 131–143 never started.
  - **Descriptive timing, derived from the committed logs (not the card's measurement):** 113 finished games over
    889.9 s is about 457 games/h. Finished games took mean 130.3 s (median 123.5, range 70.2–695.5) and 58.5 moves
    (median 56, range 33–280); 2 ended at the state cap. Finished games alone account for 817.7 s per worker of the
    890 s, so the workers were busy almost throughout: at about 130 s per game, 144 games at 18 workers need roughly
    1,040 s of work before any tail.
  - **Admission:** `admitted("generate")` on the committed evidence REFUSES (the record is not COMPLETED).
  - **Post-run:** no new panic report and no reboot; the gate inventory derives 53 gates, all closed.
- **Did it match the prediction?** The descriptive rate (about 457 games/h) lies inside §7's 350–700 range but below
  the about 576 games/h the 15-minute rule needed, which is the risk the build recorded. Per-game time at 18 workers
  (about 130 s) is about four times strength 2's single-process 32 s; why is not established here (the arbiter's
  rows per call were not recorded, because the run was VOID).
- **Process slip, disclosed:** after the run, the first evidence commit failed (zsh passed the path list as one
  argument), a pipe hid the failure, and the chained `git merge --ff-only` fast-forwarded main to the gate-opening
  commit `60493787`. For about a minute the main checkout held the gate open; nothing ran. The worktree still held the
  evidence and the restored gate; it was committed (`dba747bf`) and main fast-forwarded to it, gate identical to
  `bdb691dc`. Nothing was pushed.
- **Lifecycle test:** `test_THE_THROUGHPUT_BLOCK_IS_REGISTERED_AND_PROVEN_…` now fails, as it should (it asserts no
  run output); re-anchoring it was not in this approval and awaits the user.
- **Conclusion / status:** the throughput check is **VOID**; by card §5 step 2 the **combination policy pilot is
  CLOSED** at step 2. There is no policy-learning result: the card's hypothesis (§4: the search-derived target
  improves the agent's prior; §1: one round of learning from the agent's own search makes it stronger) is neither
  supported nor rejected. Seed closing for the block (131 slots started), any amendment and any
  new design need the user's decision. Closing, corpus generation, real-data training and pushing remain
  unauthorized. See do-not-repeat **#119**.

#### Addendum — TIMEOUT/VOID and the closure at step 2 accepted; seed closing approved (2026-10-29, the user's decision)

- **Accepted:** the throughput check's TIMEOUT/VOID (entry 41) and the pilot's closure at step 2. It gives no result
  on policy learning and does not undo the combination's earlier strength PASS (entry 39 and the strength attempt 2).
  The user verified 131 unique starts, 113 finishes, 18 unfinished games, no final `claims` event and 114 clear BOINC
  checks; the gate matches its pre-open bytes, all 53 gates are closed, and the recorded module hashes match the
  launch commit.
- **The temporary gate opening on main** (entry 41's process slip) is confirmed by the reflog and repaired. The
  disclosure and all evidence are preserved.
- **Approved, closing only, as one conditional sequence:**
  1. run the existing K1 usage derivation once in a clean worktree;
  2. only if it RECONCILES: append all of [202696900, 202697044) to both EXPOSED and RETIRED. With no final `claims`
     event the frozen rule conservatively exposes all 144 seeds, not just the 131 recorded starts. ACCOUNTED and
     TEST_ONLY unchanged;
  3. re-anchor the lifecycle test to the closed state, keeping the historical registration and proof checks; focused
     CPU checks; the closeout recorded append-only.

  If K1 does not reconcile, stop before the registry edit.
- **Not approved:** any retry, amendment, profiling run, corpus generation, training or push. See do-not-repeat
  **#120**.

#### Addendum — the throughput block closed: K1 RECONCILES, K2 exposes and retires the whole block (2026-10-29)

- **K1, run once** (`01_derive_usage.py`, `.venv/bin/python -B`) in a clean linked worktree (`cp-throughput-k1` at the
  decision commit `3b6df508`); exit 0, stderr empty. Record `combination-policy-throughput-seed-closing/
  02_derive_usage_run.txt` (sha256 `502e164a…`), committed `b69515d5`:
  - all four inputs (plan, progress, outer.json, record.json) committed and unmodified; outer exit 124, survivors
    empty, status VOID;
  - game starts: the 131 slots 0–130, each once (logged out of order at the start, as the frozen rule allows);
    **claims events: none**, so k is not established;
  - controls 21/21;
  - **VERDICT: RECONCILES -- EXPOSE [202696900, 202697044]; RETIRE [202696900, 202697044]**: the frozen rule exposes
    the whole block, not just the 131 recorded starts.
- **K2** (`e30e277c`): [202696900, 202697044) appended last to EXPOSED and to RETIRED, each with a comment block in the
  precedent's form. An AST check against the parent: only EXPOSED (44 → 45) and RETIRED (42 → 43) changed, each the
  parent's tuple plus the block; no earlier entry overlaps it; ACCOUNTED (47), TEST_ONLY and every other statement
  unchanged.
- **Tests** (`3dee44a9`):
  - The lifecycle test is renamed `test_THE_THROUGHPUT_BLOCK_IS_REGISTERED_PROVEN_RUN_ONCE_AND_CLOSED_…` and keeps
    the registration and proof checks (survey and both proofs bound to the survey hash, clean trees, HOLDS 45/45,
    plan = `build_plan`). The post-proof's "proved = current registry" check becomes "proved = the registry at the
    launch commit `60493787`", since K2 has changed the current one. It now asserts the run (outer TIMEOUT, inner
    124, gate restored, no survivors, script record VOID "terminated", same head, `outer_faults` non-empty) and the
    closing (no claims event, 21/21, the RECONCILES verdict; the block EXPOSED and RETIRED once and last; ACCOUNTED
    once with every later entry above it; every seed refused).
  - Strength attempt 2's closed-state test asserted its block was the last EXPOSED/RETIRED entry; it now requires every
    later entry to lie above it.
- **Checks:**
  - launch + pilot: 99 passed, 1 deselected (three consecutive runs);
  - with the gate-inventory, mirror-halving launch, background-compute, seed-selection and tactical-survey suites:
    191 passed, 1 deselected (opt-in);
  - every test file that reads the registry (41), with the default marker exclusion, in linked worktrees: at the
    parent `b69515d5`, 17 failed and 69 errors (the older 16 + 69 plus the stale lifecycle test); after K2 alone, one
    more (strength 2's "last entry" assert); after the re-anchor, 16 failed and 69 errors, exactly this morning's older
    set (tests needing the checkpoint, absent in a linked worktree, and older "last entry" asserts). None new.
  - The gate inventory derives 53 gates, all closed.
- **A disclosed misstep:** the first comparison run passed `-o addopts=""`, which also drops the default
  `-m "not integration and not oracle and not parity"`. It was stopped within seconds. None of the 41 files carries
  those markers, so it selected no extra test; it was rerun with the exclusion stated explicitly.
- **Prediction on record:** none (a closing procedure).
- **Status:** the combination policy pilot is CLOSED at step 2 and its throughput block is closed. No result on policy
  learning; the combination's earlier strength PASS stands. No retry, amendment, profiling run, corpus generation,
  training or push is authorized. See do-not-repeat **#121**.

#### Addendum — the closeout accepted (2026-10-29, the user's review)

- **Accepted, with no further corrections or tests needed.** The user verified K1's committed record and hash
  (RECONCILES, 21/21); the whole block [202696900, 202697044) appended exactly once to EXPOSED and RETIRED; ACCOUNTED,
  TEST_ONLY and every other registry statement unchanged; the historical run, plan and proof evidence unchanged; ledger
  edits additions only; all 53 gates closed; the two unrelated untracked files untouched.
- **Status:** the combination policy pilot and its seeds are fully closed. Policy learning remains untested; the
  combination's earlier strength PASS stands. No retry, amendment, profiling, generation, training or push is
  authorized. See do-not-repeat **#121**.

### Decision note — after entry 41: one fresh attempt at the throughput check, amendment and build only (2026-10-29, the user's decision)

- **Reason (the user's):** attempt 1 completed 113 games in about 890 s, which suggests roughly 19 minutes for 144
  games. Its 15-minute deadline was too tight. This gives no result on policy learning.
- **Approved: documentation, implementation and focused CPU tests only,** for one fresh attempt:
  - the same 144 games, model, search, 18 workers and inference batch cap 14;
  - ceiling TERM at 1,790 s, KILL at 1,800 s, bounded cleanup observation through 1,810 s;
  - one attempt only, with no automatic retry, extension or top-up;
  - fresh seeds through the existing procedure, with separate create-only plan and evidence paths;
  - a separate gate, created closed;
  - the existing BOINC, idle, TF32, eval-mode and process-group cleanup checks preserved.
- **Also required:** a short prospective amendment granting this narrow exception to the pilot's closure; attempt 1's
  VOID record, closed seeds and historical evidence unchanged; the existing implementation reused with the smallest
  necessary changes; seed-procedure copies prepared but not executed; focused CPU checks for the new paths, the ceiling
  and admission. Future generation must require successful committed inner AND outer records from attempt 2; attempt 1
  must remain ineligible.
- **Not changed:** the learning target, the training recipe and the evaluation criteria. No repeat of the CPU timing,
  no profiling and no parameter sweep.
- **Not authorized:** seed-procedure execution, checkpoint loading, GPU work, gameplay, training and pushing.
- **Prediction on record:** none.

#### Addendum — attempt 2 of the throughput check, built for review (2026-10-29)

- **The amendment** `docs/superpowers/2026-10-29-combination-policy-throughput-attempt2-amendment.md` (sha256
  `6dba686d…`, pinned in the pilot): why, the one exception, what is unchanged, the five changes (ceiling, fresh seeds,
  separate paths, gate 54, generation admitted from attempt 2 only), attempt 1 preserved, and what it does not
  authorize. The card is unchanged.
- **The pilot** (`combination_policy_pilot.py`, 24 lines added, 5 removed):
  - `KINDS["throughput2"]`: attempt 1's 144 slots, no rows kept, with its own gate, evidence
    (`combination-policy-throughput2`), plan (`combination-policy-throughput2-run/plan.json`) and the pinned amendment.
    Admitted, as attempt 1 was, by the committed CPU timing.
  - `run` refuses `throughput2` unless the amendment file has the pinned sha256 (checked after the gate). Its record
    carries `amendment` and `amendment_sha256`. It prices generation exactly as attempt 1 would have.
  - **Generation's admission now reads attempt 2 only:** its committed record must be COMPLETED under the card with the
    pinned `amendment_sha256`, and its committed `outer.json` must pass `outer_faults` unchanged (zero inner and outer
    exits, COMPLETED, the gate restored, survivors empty, the script's record, the same head). Attempt 1's paths are
    never read.
  - Nothing else changed: 18 workers, the cap of 14, the recipe, the target, the timing, the checks.
- **The gate:** `combination_policy_throughput2_authorization.py`, `COMBINATION_POLICY_THROUGHPUT2_AUTHORIZED = False`,
  a single declaration. The gate inventory goes 53 → 54.
- **The wrapper** `combination-policy-throughput2-run/01_wrapper.sh`: attempt 1's with only `OUT_REL`, `RUN_REL`,
  `GATE_REL`, `GATE`, `TERM_S=1790` and `run throughput2` changed, plus a header paragraph before attempt 1's unchanged
  header. GRACE_S stays 10, so KILL comes at 1,800 s and the observation ends by 1,810 s. The idle gate, the BOINC
  check, the process-group supervisor, the survivors, the records and the gate restoration are byte-identical.
- **The seed copies** (built, not run): `combination-policy-throughput2-seed-survey/01_seed_survey.py` and
  `combination-policy-throughput2-seed-closing/01_derive_usage.py`, attempt 1's with only the paths, the plan's kind
  (`combination-policy-throughput2-plan/1`), the module names and the titles and verdict lines changed. Each docstring
  is attempt 2's paragraph followed by attempt 1's description unchanged. The method pins are unchanged.
- **Tests:**
  - `tests/test_combination_policy_pilot.py` (75 → 80):
    - the throughput pricing test now covers both attempts;
    - every pilot gate is checked closed;
    - generation's admission tests now read attempt 2's paths and need the amendment's sha256;
    - new: attempt 2 is attempt 1's check on its own paths, gate and pinned amendment; generation is never admitted
      from attempt 1 (a COMPLETED attempt-1 record with a perfect outer admits nothing, and its paths are never read),
      nor from an attempt-2 record without the pinned sha256; `run throughput2` refuses a changed amendment; attempt 1
      stays one-shot even with its gate opened in-process.
  - `tests/test_combination_policy_launch.py` (24 → 23): its two end-to-end admission tests assumed generation reads
    attempt 1. They are replaced by one: attempt 1's completed, committed fake run admits nothing. Their accept and
    TIMEOUT-over-COMPLETED cases move to attempt 2's file.
  - New `tests/test_combination_policy_throughput2_launch.py` (17):
    - nothing of attempt 2 has run, and attempt 1's evidence is unchanged since `e30e277c`;
    - the wrapper is attempt 1's with only the stated differences, line for line;
    - the ceiling is 1,790 / 1,800 / 1,810 s;
    - the wrapper serves the pilot's attempt-2 paths;
    - fake launches of the real attempt-2 wrapper: a busy machine, BOINC or a failed check refuses; an idle one reaches
      `run throughput2` with its exact arguments and `term_s` 1790; the ceiling (TERM_S 2 in a copy) ends the whole
      group and the VOID record lands at attempt 2's path;
    - through the real committed readers: attempt 2's completed run admits generation once committed; an outer TIMEOUT
      over its COMPLETED record refuses;
    - the seed copies differ from attempt 1's only as stated, serve attempt 2's paths, hold no seed literal, and their
      controls pass (12 and 21).
  - `tests/test_gate_inventory.py`: 54 gates, the stale-count guard extended.
- **Mutation checks** (each on a temporary edit, restored byte-equal): generation re-pointed at attempt 1 → 23 tests
  fail; the amendment pin check removed → 1 fails; attempt 1's TERM_S in the attempt-2 wrapper → 4 fail.
- **Results:** the three combination test files: 120 passed, 1 deselected (opt-in), three consecutive runs. With the
  gate-inventory, mirror-halving launch, background-compute, seed-selection and tactical-survey suites: 212 passed,
  1 deselected. No test process left behind.
- **Nothing run:** no seed procedure, registration, checkpoint, GPU, game, training or push. Attempt 1's evidence,
  closed seeds and registry entries are unchanged. All 54 gates are closed.
- **Status:** awaiting the user's review. See do-not-repeat **#122**.

#### Addendum — attempt 2's build accepted; "no further attempt" clarified; the seed survey and pre-proof approved once each (2026-10-30, the user's decision)

- **The build is accepted** (`d5340a0e`): no material implementation blocker. The 30-minute wrapper, separate paths and
  closed gate match the proposal; generation requires attempt 2's successful inner and outer records; attempt 1's
  evidence and registry entries are unchanged. All 54 gates are closed.
- **Clarification of the amendment's "no further attempt, whatever happens" and of #122's "a third attempt".** The
  authorization is for **one attempt now, with no automatic retry**. If attempt 2 times out or is interrupted, the cause,
  the likely cost and whether another attempt is worthwhile are assessed then, and any further attempt is a fresh
  decision by the user. Nothing is built for a retry now. The pinned amendment (sha256 `6dba686d…`) is **not** edited:
  this dated text clarifies it. See **#123**.
- **Approved:** the attempt-2 seed survey, once, in a clean linked worktree. **Only if it is CLEAN:** commit its
  create-only record, then run the pre-registration proof once against that committed record, commit its record, and
  return both results for review before registration.
- **Not authorized:** registration, the plan, the post-proof, opening the gate, checkpoint loading, GPU work, games,
  training and pushing.

#### Addendum — attempt 2's seed survey → CLEAN; its pre-registration proof → HOLDS (2026-10-30)

- **The survey (`df0c0ec9`), run once** in a clean linked worktree (branch `cp-throughput2-seeds`, from `a3a54485`;
  main fast-forwarded and the worktree removed afterwards):
  `.venv/bin/python -B docs/superpowers/evidence/combination-policy-throughput2-seed-survey/01_seed_survey.py --mode
  survey`, exit 0, nothing on stderr.
  - Verdict: **THE COMBINATION-POLICY THROUGHPUT ATTEMPT-2 BLOCK IS CLEAN**, controls **45/45**, clean tree at
    `a3a54485`; the main checkout scanned too (49,837 files; its 2 untracked files are not required clean).
  - Registries as read: ACCOUNTED 47, EXPOSED 45, RETIRED 43, TEST_ONLY 1 (registry sha256 `f04ae6c7…`).
  - The rule's block is **[202697200, 202697344)**: BASE 202697200, 144 seeds, above the highest registered seed
    202697043 (attempt 1's last, EXPOSED and RETIRED). Direct overlap none, 0 derived streams, injective, nearest gap
    150 (floor 144). Slot g plays seed BASE + g, slots 0–143 listed.
  - Record `combination-policy-throughput2-seed-survey/02_seed_survey_run.txt`, sha256 `26ef8470…`.
- **The pre-registration proof (`fcd3969d`), run once** in a fresh clean linked worktree (branch `cp-throughput2-pre`,
  from `df0c0ec9`; main fast-forwarded and the worktree removed afterwards): the same script, `--mode pre`, exit 0,
  nothing on stderr.
  - Bound to the committed survey record (sha256 `26ef8470…`, read at `a3a54485`); the block read from it.
  - **EXPECTED STATE (pre): HOLDS** (the survey's registry, the block unregistered, no plan). The block re-checked on
    the tree at `df0c0ec9`: direct none, 0 streams, injective, gap 150 (floor 144), CLEAN. Controls **45/45**.
  - Verdict: **THE COMBINATION-POLICY THROUGHPUT ATTEMPT-2 BLOCK IS CLEAN AND THE STATE IS AS EXPECTED.**
  - Record `03_collision_proof_pre_run.txt`, sha256 `803a93d9…`.
- **One test re-anchored:** the build-state test in `tests/test_combination_policy_throughput2_launch.py` asserted that
  no seed record exists, so it failed once both records were committed (the only failure). It now asserts the survey
  and pre-proof state: both records with their verdicts, 45/45 controls and clean trees; the proof bound to the survey
  record's sha256; the block, read from the record (the test holds no seed literal, since the post-proof scans
  `tests/`), in no registry; the gate closed; no plan, run output or closing record; attempt 1's evidence unchanged
  since `e30e277c`. The overlap check fires on attempt 1's block (all three registries). The three combination files
  and the gate inventory: 128 passed, 1 deselected.
- **Not done:** registration, the plan, the post-proof, the gate, checkpoint loading, GPU work, games, training and
  pushing. ACCOUNTED and the registry are unchanged; all 54 gates are closed.
- **Status:** awaiting the user's review before registration.

#### Addendum — attempt 2's survey and pre-proof accepted; registration, plan and post-proof approved as one conditional sequence (2026-10-30, the user's decision)

- **Accepted:** the survey (`df0c0ec9`) and pre-proof (`fcd3969d`). The user verified both committed hashes, their
  binding, all 144 slot lines, the 45/45 controls and the unchanged registry. The block is unregistered, all 54 gates
  are closed and attempt 1's evidence is unchanged. The lifecycle-test re-anchoring was routine upkeep of approved work
  and needed no separate approval. #123 matches the intended policy: any interruption is assessed before deciding what
  to do next.
- **Approved, as one conditional sequence:**
  1. Append exactly [202697200, 202697344) to ACCOUNTED and create the plan through the survey copy's `build_plan`,
     bound to the survey record's sha256. The other registry lists stay unchanged.
  2. Only if the registration and plan are exactly as expected: run the post-registration proof once, commit its
     record, and update the affected lifecycle test with focused CPU checks.
- **Stop and return** if the registration differs or the proof does not hold. Otherwise return the combined results
  before the gate is opened.
- **Not authorized:** opening the gate, checkpoint loading, GPU work, games, training and pushing.

#### Addendum — attempt 2's block registered with its plan; the post-registration proof → HOLDS (2026-10-30)

- **Registration (`22f81be1`),** in a clean linked worktree (branch `cp-throughput2-reg`, from `8b381f96`; main
  fast-forwarded and the worktree removed afterwards):
  - The block READ from the accepted survey record (sha256 `26ef8470…` recomputed; the survey copy's own
    `read_record`), checked equal to the approved [202697200, 202697344) before anything was written; the registry
    checked equal to the one the survey read.
  - ACCOUNTED: exactly that block appended last (47 → 48), with a comment in the precedent's form holding no seed
    literal. An AST check against the parent: only ACCOUNTED changed, its prefix identical; EXPOSED (45), RETIRED (43),
    TEST_ONLY (1) and every other statement unchanged; no list overlaps the block.
  - `combination-policy-throughput2-run/plan.json` = the survey copy's `build_plan(202697200, 26ef8470…)`, written
    create-only: kind `combination-policy-throughput2-plan/1`, base 202697200, n_slots 144.
- **The post-registration proof (`1c2f4220`), run once** in the same worktree on the clean committed tree at
  `22f81be1`: `--mode post`, exit 0, nothing on stderr.
  - Bound to the survey record (sha256 `26ef8470…`, read at `a3a54485`); registry sha256 `0aa5c3b0…`.
  - **EXPECTED STATE (post): HOLDS** (ACCOUNTED = the survey's with exactly the block appended last, every other list
    and statement unchanged, the plan committed and equal to `build_plan`). The block re-checked: direct none, 0
    streams, injective, gap 150 (floor 144), CLEAN. Controls **45/45**.
  - Verdict: **THE COMBINATION-POLICY THROUGHPUT ATTEMPT-2 BLOCK IS CLEAN AND THE STATE IS AS EXPECTED.**
  - Record `04_collision_proof_post_run.txt`, sha256 `8973a56d…`.
- **The lifecycle test re-anchored** (`tests/test_combination_policy_throughput2_launch.py`, no seed literal): the
  survey, pre- and post-proof records with their verdicts, 45/45 and clean trees, both proofs bound to the survey
  record's sha256; the plan equal to `build_plan`; the block (from the plan) the record's block, ACCOUNTED once and
  last, in no other list, every seed schedulable; the registry the post-proof proved is the registry now; the gate
  closed; no run output or closing record; attempt 1's evidence unchanged since `e30e277c`. Two temporary mutations
  (a registry comment appended; the plan's n_slots 143) each fail it, restored byte-equal.
- **Results:** the three combination files with the gate-inventory, mirror-halving launch, seed-selection, registry,
  background-compute and tactical-survey suites: 243 passed, 1 deselected. No test process left behind.
- **Not done:** opening the gate, checkpoint loading, GPU work, games, training and pushing. All 54 gates are closed.
- **Status:** awaiting the user's review before the gate is opened.

#### Addendum — attempt 2's registration and post-proof accepted; the ONE attempt-2 launch approved, with its closing as one conditional sequence (2026-10-30, the user's decision)

- **Accepted:** the registration (`22f81be1`) and post-proof (`1c2f4220`). The user verified the exact ACCOUNTED append,
  the unchanged other registry statements, the plan and its survey binding, the post-proof hash and its 45/45 controls;
  the card and amendment pins match, all 54 gates are closed, and no attempt-2 run output exists.
- **Approved sequence:**
  1. Record this decision; a fresh read-only panic/reboot check; recheck that the proved registry and the plan are
     unchanged.
  2. In a clean worktree, open **only gate 54** with the one-line commit and run the existing wrapper
     (`combination-policy-throughput2-run/01_wrapper.sh`) **once**. This authorizes checkpoint loading and GPU work for
     these 144 throughput games.
  3. The idle and BOINC checks, TF32=0 and the ceiling (TERM 1,790 s, KILL 1,800 s, cleanup observation to 1,810 s)
     stay as built.
  4. After a **started** run: the gate restored, then the existing seed-closing derivation
     (`combination-policy-throughput2-seed-closing/01_derive_usage.py`) once. If it RECONCILES, apply its exposure
     verdict, retire the full block and update the lifecycle test with focused CPU checks. If it does not reconcile,
     stop before editing the registry.
- Commit the evidence and the closeout, then return for review. A pre-launch refusal is not a started run. If the
  attempt times out or is interrupted, the cause and whether another attempt is worthwhile are assessed (#123); there is
  no automatic retry.
- **Not authorized:** corpus generation, training and pushing. See **#124**.

### 42. Combination policy pilot — step 2, the throughput check, ATTEMPT 2 (2026-10-30, outer TIMEOUT; VOID; closed; awaiting assessment)

- **Test:** amendment `docs/superpowers/2026-10-29-combination-policy-throughput-attempt2-amendment.md` (sha256
  `6dba686d…`) to card §5 step 2: the same 144 games (calib020_0001 in eval mode, mirror averaging, root sequential
  halving from ply 20, 24×24, `MLX_ENABLE_TF32=0`), 18 workers, inference cap 14, on the fresh registered block
  [202697200, 202697344), inside a 30-minute ceiling (TERM at 1,790 s, KILL at 1,800 s, cleanup observation to
  1,810 s). All 144 games must finish inside it; a run the ceiling stops is VOID. Its games are never used.
- **Hypothesis:** with the ceiling no longer binding at attempt 1's pace, the 144 games finish, and the run prices step
  3's generation.
- **Why it was plausible:** attempt 1 finished 113 games in about 890 s (entry 41), about 457 games/h, which projects
  roughly 19 minutes for 144.
- **Prediction on record:** the amendment's "At that pace 144 games need roughly 19 minutes" (written before the run).
  No prediction was stated about the tail of long games.
- **What ran:** decision `b560ad6e` (#124). A fresh read-only panic/reboot check (no reboot since 2026-10-25 09:39; the
  only panic report is that one; a JetsamEvent report at 2026-10-29 21:01, not a panic); load about 1.4. Recheck:
  registry sha256 `0aa5c3b0…` = the post-proof's; registry, run directory, pilot and amendment unchanged since
  `1c2f4220`; no gate open; BOINC check CLEAR (podman exempt); checkpoint present. Clean linked worktree
  `cp-throughput2-run`; the one-line gate opening `a01f17c0` (only `COMBINATION_POLICY_THROUGHPUT2_AUTHORIZED`); the
  existing wrapper run once at 12:35:28 EDT, ended 13:07:26. Idle gate held 60 s, background check CLEAR, then launch.
  Evidence `e95b0e51` (the worktree's evidence commit, with the restored gate, then main fast-forwarded to it).
- **Result:**
  - **Wrapper:** inner exit **124** (the ceiling fired and TERM sufficed), outer exit 124, outcome **TIMEOUT**,
    `gate_restored` true, survivors empty, `record_by` script, wall 1,791 s. The gate file is identical to the
    pre-open commit; no pilot process remained; pgid 12724 was empty.
  - **Inner record:** status **VOID**, step "terminated", "SIGTERM after 1789.9 s"; `throughput` null, verdict null.
    The amendment's sha256 recorded. TF32 "0" at start and end; the model loaded with 0 of 42 modules in training
    mode; 144 BOINC checks, all clear; stderr empty.
  - **Slots:** all **144 started** (slots 0–143, the last start at 1,015 s), **143 finished**; slot 108 was in flight
    at TERM, having started at 711 s and run 1,079 s.
  - **Descriptive timing, derived from the committed logs (not the card's measurement):** finished games took mean
    150.4 s (median 120.8, range 71.2–1,091.5) and 60.5 moves (median 54, range 33–280). 111 games had ended by 900 s,
    136 by 1,200 s, 141 by 1,500 s; the 143rd ended at 1,724 s. **Four games reached the 280-move state cap and took
    686, 696, 1,044 and 1,091 s**; every other finished game took at most 408 s. Finished-game time summed to 21,507 s,
    about 1,195 s per worker (range 711–1,724 s). Over the whole 1,790 s, 143 finished games is about 288 games/h;
    with no tail, the summed work alone would need about 20 minutes.
  - **Concurrent activity:** macOS diagnostic reports during the run: `appstoreagent` (12:54) and an `installd`
    CPU-resource report (12:57). The launch checks do not cover them; whether they slowed the run is not established.
  - **Admission:** `admitted("generate")` on the committed evidence REFUSES (the record is not COMPLETED with the
    amendment's sha256).
  - **Post-run:** no new panic report and no reboot; all 54 gates closed.
- **Did it match the prediction?** No. The bulk matched it: 111 games by 900 s, about attempt 1's pace, and the summed
  work is about 20 minutes. The run was lost to the tail: state-cap games take 700–1,100 s each, and a long game
  started late (slot 108 at 711 s) cannot finish inside 1,790 s. The amendment's projection used the mean game time
  and did not price that tail. Why the state-cap games take this long (about 4 s per move against about 2 s for the
  rest) is not established here.
- **Closing:**
  - **K1 (`f79174c4`), run once** in a clean linked worktree (`cp-throughput2-close`, from `e95b0e51`), exit 0:
    **RECONCILES**, controls **21/21**. Inputs committed and unmodified; outer exit 124, survivors '', status VOID; no
    `claims` event, so k is not established and the **whole block is EXPOSED**; the whole block is RETIRED. Record
    `combination-policy-throughput2-seed-closing/02_derive_usage_run.txt`, sha256 `9d020cb4…`.
  - **K2 (`ae361458`):** the verdict read from K1's committed record and checked against the plan; [202697200,
    202697344) appended last to EXPOSED (45 → 46) and to RETIRED (43 → 44), with comments holding no other seed
    literal. AST check against the parent: only those two lists changed, ACCOUNTED (48) and TEST_ONLY (1) unchanged.
    Every seed of the block is refused.
- **Tests re-anchored (`5a243133`):** the attempt-2 lifecycle test asserts the closed state (one run, outer TIMEOUT
  over a VOID record at 1,790 s, gate restored, no survivors, generation refused by the real committed readers; the
  proved registry = the launched one; K1 RECONCILES with no claims event; the whole block EXPOSED and RETIRED once and
  last, ACCOUNTED unchanged; every seed refused), keeping the survey and proof checks and no seed literal. Attempt 1's
  test now requires every later EXPOSED/RETIRED entry to lie above its block. A temporary deletion of the RETIRED entry
  fails the test (restored byte-equal). The focused suites: 243 passed, 1 deselected.
- **Conclusion / status:** the throughput check, attempt 2, is **VOID** and its block is **closed**. There is still no
  policy-learning result; the combination's earlier strength PASS stands. Per #123 there is no automatic retry: the
  cause (the state-cap tail), the likely cost and whether another attempt is worthwhile are for the user to assess.
  Corpus generation, training and pushing remain unauthorized. See do-not-repeat **#125**.

#### Correction to entry 42 (2026-10-30)

- Entry 42's "about 4 s per move against about 2 s for the rest" overstates the state-cap games' rate. From the
  committed logs: the four 280-move games ran **2.5–3.9 s per move** (686–1,091 s), against about **2.2 s per move**
  for a median game (120.8 s, 54 moves). The rest of entry 42 stands.

### Decision note — after entry 42: generation without a completed throughput check; preparation only (2026-10-30, the user's decision)

- **Accepted:** attempt 2's timeout and closing. The records support the report: all 144 games started and 143
  finished, and slot 108 was running at TERM; the gate was restored and the recorded survivors are empty; K1 reconciled
  and the whole block was correctly exposed and retired. All 54 gates are closed. The VOID run is **no formal
  throughput pass and no training result**; its elapsed-time observations help plan the next budget.
- **The reviewer's own correction:** the 30-minute recommendation was too optimistic. It extrapolated from average
  completions without allowing for a late, long game.
- **Decision:** move directly to preparing the 1,000-game training corpus, with a six-hour generation ceiling. Another
  144-game throughput run is unlikely to answer the learning question and could meet the same long-game tail.
- **Approved, for review before any GPU launch:**
  - a short prospective amendment that removes the requirement for a completed 144-game throughput check and sets six
    hours as the ceiling for one 1,000-game generation run. That is the card's existing maximum generation budget,
    not a guarantee that the corpus will finish;
  - the minimal changes needed for generation admission under that explicit exception. The old TIMEOUT records stay
    VOID, and the runner must not relabel them as a pass;
  - fresh seed preparation by the existing method: the survey, proofs and registration once each if their checks pass,
    the combined preparation returned for review;
  - focused CPU tests. The model, search, learning target, training recipe and strength-evaluation criteria stay
    unchanged.
- **If generation later times out,** progress and the cost of continuing are assessed; the learning question is not
  abandoned automatically.
- **Not authorized:** generation, checkpoint loading, training and pushing. The two throughput runs' games remain
  excluded from training.
- **Prediction on record:** none.

#### Addendum — generation's amendment, admission and wrapper built; the seed survey NOT run: the existing rule cannot place a clean 1,000-seed block (2026-10-30)

- **The amendment** `docs/superpowers/2026-10-30-combination-policy-generation-amendment.md` (sha256 `11ced081…`,
  pinned in the pilot), committed in `7295e765` with everything below:
  - why (entries 41 and 42), and the exception: no completed throughput check; generation is admitted by the committed
    CPU timing and the pinned amendment; neither throughput record is read or relabelled;
  - the six-hour ceiling: TERM at 21,590 s, KILL at 21,600 s, the cleanup observation through 21,610 s;
  - a timeout is VOID and its games are not used; the user then assesses progress and cost; no automatic retry;
  - what is unchanged, the paths, and what it does not authorize. It does not choose the seed method.
  - The card and the throughput amendment are unchanged.
- **The pilot** (`combination_policy_pilot.py`, 14 lines added, 8 removed):
  - `KINDS["generate"]` pins the amendment, by the mechanism attempt 2 used: `run generate` refuses a changed file,
    and its record carries `amendment` and `amendment_sha256`.
  - `ADMISSION["generate"]` is now the committed CPU timing record (COMPLETED under the card) only, as for the
    throughput kinds. `OUTER_ADMISSION` is empty: generation reads no throughput `record.json` or `outer.json`.
    `outer_faults` itself is kept.
  - Nothing else changed: the slots, rows, paths, gate, workers, cap, recipe, target and training admission.
  - On the real repository `admitted("generate")` now returns the timing alone. Generation still cannot start: its
    gate is closed and no plan exists.
- **The wrapper** `combination-policy-generate-run/01_wrapper.sh`: throughput attempt 2's, with only `OUT_REL`,
  `RUN_REL`, `GATE_REL`, `GATE` (the existing `COMBINATION_POLICY_GENERATE_AUTHORIZED`, still closed), `TERM_S=21590`
  and `run generate` changed, plus a header paragraph before attempt 2's unchanged header. No new gate (54).
- **Tests:**
  - `tests/test_combination_policy_pilot.py` (80 → 81): generation is admitted by the timing alone and reads no
    throughput record or outer.json, whatever those records say; no step reads an outer.json now; `outer_faults` keeps
    its 17 failure cases as a unit check; the amendment is pinned and nothing else moved; `run generate` refuses a
    changed amendment; the generation record carries the amendment.
  - `tests/test_combination_policy_launch.py` (23): attempt 1's completed fake run still admits nothing; the refusal
    now names the missing timing record.
  - `tests/test_combination_policy_throughput2_launch.py` (17): the closed-state test now asserts that generation's
    real admission reads only the timing record; attempt 2's completed fake run admits nothing; its TIMEOUT over
    COMPLETED keeps both records as written, and `outer_faults` rejects it.
  - New `tests/test_combination_policy_generate_launch.py` (14): nothing of generation has run, and both throughput
    runs stay VOID and unchanged since their closings; the wrapper is attempt 2's with only the stated differences;
    the ceiling is 21,590 / 21,600 / 21,610 s; it serves the pilot's generate paths; fake launches (busy, BOINC, a
    failed check refuse; an idle machine reaches `run generate` with its exact arguments; TERM_S 2 in a copy ends the
    whole group with the VOID record at generation's path); through the real committed readers, a committed COMPLETED
    timing admits generation without reading either throughput record, and an absent, uncommitted, VOID or other-card
    timing refuses.
- **Mutation checks** (each a temporary edit, restored byte-equal): generation re-pointed at attempt 2 → 8 failures;
  generation's outer read restored → 4; the amendment pin removed → 4; attempt 2's TERM_S in the generation wrapper → 4.
- **Results:** the four combination files and the gate inventory: 143 passed, 1 deselected. With the
  background-compute, mirror-halving launch, seed-selection and tactical-survey suites: 227 passed, 1 deselected. No
  test process left behind.
- **The seed survey was NOT run.** Before running it, the existing rule (`seed_selection.candidate`, ALIGN 100) was
  evaluated arithmetically on the current registry, writing nothing:
  - the highest registered seed is 202697343, so BASE = 202698400 and the block is [202698400, 202699400);
  - that block contains two in-band repository literals, 202699000 and 202699032: an H4 fixture in
    `tests/test_h4_runner.py`, also quoted in H4/H5 cards and in create-only H4 evidence records;
  - the pinned check counts repository literals as direct overlaps and in the gap floor (the block's own size), so
    the survey could only return NOT CLEAN. The largest block the rule can place clear of 202699000 by its own floor
    is 550 seeds;
  - `seed_selection.py`'s docstring already says the band's room ends at that fixture, and that going beyond it needs
    a new band, a new scanner and its own review.
  - Running the survey would have spent its one run on a known NOT CLEAN, so it was not run. No seed record,
    registration, plan or proof exists. The literals sit partly in create-only evidence, so removing them is not an
    option.
- **Open for the user's decision:**
  - the seed method for the 1,000-slot block, since the existing method is exhausted in the 2026 band;
  - flagged, not changed: training's admission reads only generation's `record.json`. The 2026-10-29 ruling (an outer
    TIMEOUT overrides an inner COMPLETED record) was made for the throughput → generation step and is not enforced for
    generation → training.
- **Not done:** seed procedures, registration, the plan, opening the gate, checkpoint loading, GPU work, generation,
  training and pushing. All 54 gates are closed.
- **Status:** awaiting the user's review. See do-not-repeat **#126**.

### Decision note — the generation build accepted; the 2027 band chosen; the seed method and training admission to be built (2026-10-30, the user's decision)

- **Accepted:** the generation build (`7295e765`) and the seed blocker. The existing rule selects [202698400,
  202699400), which holds the documented H4 fixture values, and the scanner recognizes only the 2026 band, so any other
  range needs the scanner extended. The user verified the amendment pin, the unchanged throughput evidence and all 54
  gates closed; no plan or generation output exists.
- **Decision:** generation's seeds come from the **2027 band** [202700000, 202800000). It is a range of numbers only;
  nothing is scheduled in 2027. No block is selected or proven yet.
- **Approved, built together with focused CPU tests, returned for review before the real survey:**
  - extend the existing scanner to plain and underscore-grouped literals of both the 2026 and the 2027 band, keeping
    the scan scope and the checks against registered and named-spent seeds and all four derived streams;
  - a deterministic selection rule, defined before the survey and covering an empty band, keeping the 1,000-seed gap
    floor and the collision checks. A failing block returns for review; it is never adjusted on the spot;
  - training's admission requires generation's committed `record.json` and `outer.json` to show successful completion
    and confirmed cleanup; an outer TIMEOUT refuses training even over a COMPLETED record.
- **Focused checks:** synthetic inputs, including old- and new-band literals and cross-band stream collisions.
- **Not authorized:** seed procedures, registration, checkpoint loading, GPU work, training and pushing.
- **Prediction on record:** none.

#### Addendum — the two-band seed method and training's outer-record admission built (2026-10-30)

Committed in `f870de24`; nothing run on the real repository.

- **`scripts/GPU/alphazero/seed_band_2027.py`** (new; holds no seed literal):
  - **The scanner** `scan`: the pinned survey's, with the prefix widened from 2026 to 2026 or 2027. Plain and
    underscore-grouped nine-digit literals, the same boundaries (no digit or `.` before, no digit after), every
    occurrence counted, the same (line, column) exemption. Other prefixes, ten-digit runs and decimals are not counted.
  - **`install(S)`** rebinds `scan` on a private copy of the pinned survey module, after checking that the copy's
    scanner is the pinned one. Its `repo_literals` looks the name up at call time, so the scope, fail-closed reading,
    exemptions, prior terms, four derived streams and the check (gap floor = the block's own size) stay the pinned
    ones. The pinned files are not edited.
  - **The rule** `candidate(intervals, literals, n)`, in the 2027 band:
    - anchor = the largest of: the band's lower edge − 1; the last seed of every registered or named-spent interval
      ending at or above the edge; every repository literal at or above the edge;
    - BASE = the least multiple of 100 ≥ anchor + 1 + N, and the block is [BASE, BASE + N);
    - **an empty band** gives BASE = the least multiple of 100 at least N above the edge, so the 2026 band's top lies
      at least N below the block;
    - it is refused (`Exhausted`) if the block would leave the band, and never wraps or searches.
  - **Why the rule reads literals (unlike `seed_selection`):** before writing the rule I listed the 2027-band literals
    in scope. There are four values: three synthetic test bases (202700000, 202700143, 202700144) and a hex-digest
    fragment in the tactical candidates corpus (202738776). A literal-blind rule would pick [202701000, 202702000),
    which the gap floor is known to reject against the test bases. Reading the literals avoids that without editing
    anything. On those four values the expected block is [202739800, 202740800). That is arithmetic on a grep, not a
    survey result: the survey decides.
  - **Cross-band streams:** computed exhaustively, no 2026-band value shares a derived stream with a 2027-band value
    under the four real masks. The pinned check is band-agnostic anyway, and it rejects a synthetic-mask cross-band
    collision.
- **The generation survey copy** `combination-policy-generate-seed-survey/01_seed_survey.py` is throughput attempt 2's
  with these changes:
  - N = 1000; the paths, the plan's kind, the module name, the titles and the verdicts;
  - the rule module `seed_band_2027`, installed into the pinned survey copy;
  - in survey mode, the scan runs first: once with no block, so the plan is never exempt for the rule. The rule reads
    those literals, and the check then scans as before;
  - ten two-band controls: both prefixes seen in plain and grouped form, the neighbours not seen, the install in
    effect, the rule on an empty band and above the literals, a differential cross-band stream collision rejected,
    and no real-mask cross-band stream.
  - The three modes, the state checks, the exemptions and the 45 copied controls are unchanged, so each mode now runs
    55 controls.
- **Training's admission** (`combination_policy_pilot.py`): `OUTER_ADMISSION["train"]` reads generation's committed
  `outer.json` (`combination-policy-generate-run/`) through `outer_faults`: zero inner and outer exits, COMPLETED, the
  gate restored, survivors exactly empty, the script's record and the same head. This is in addition to the generation
  and timing records.
- **Tests:**
  - `tests/test_seed_band_2027.py` (44):
    - both bands, plain and grouped; the neighbours are not counted;
    - occurrences and exemptions; a 300-text differential against the pinned scanner on 2026-only text;
    - the pinned scanner alone misses the 2027 band; the install, its refusal and that nothing else changes;
    - the rule: empty band, ignoring the 2026 band, above intervals and literals with the least margin, a straddling
      interval, the input left unchanged, `Exhausted`, a bad N;
    - cross-band stream collisions both ways (synthetic masks), none with the real masks, the gap floor at exactly
      1,000 either side;
    - nothing registered in the 2027 band yet; the module holds no literal. That check caught two example literals I
      had written in the module's docstring; they were removed before the commit.
  - `tests/test_combination_policy_generate_launch.py` (14 → 23):
    - training admitted by a completed, committed fake generation run; refused by an outer TIMEOUT over a COMPLETED
      record, and by an uncommitted `outer.json`;
    - the survey copy differs from attempt 2's only as stated, line by line; it serves the generation run; its code
      holds no literal in either band;
    - all its controls pass on a synthetic 2027 block (12 + 8 + 6 + 6 + 10); without the install, six plant controls
      fail;
    - **a three-mode rehearsal on a synthetic git repository** (the copy and the pinned method; a synthetic registry;
      one planted literal per band; the real prior terms): survey CLEAN with the block above the planted 2027
      literal, then pre HOLDS, a registration, and post HOLDS, each 55/55. It scans only the synthetic tree.
  - `tests/test_combination_policy_pilot.py` (81 → 82): only training reads an `outer.json`, and it is generation's.
    Training is admitted only with it, and each of the 17 outer failures refuses training over a COMPLETED record.
- **Mutation checks** (each a temporary edit, restored byte-equal):
  - the scanner sees only the 2026 prefix → 7 failures; the plain pattern only 2026 → 7;
  - the install a no-op → 4;
  - the rule ignoring literals → 6; the rule dropping the N margin → 9;
  - training reading no `outer.json` → 22.
- **Results:**
  - the combination files, the seed method and the gate inventory, with the background-compute, mirror-halving launch,
    seed-selection and tactical survey/proof suites: **312 passed**, 1 deselected;
  - no test process left behind.
- **Not built:** the generation closing copy (K1), which is needed only after a run.
- **Not done:** the real survey, the proofs, registration, the plan, the gate, checkpoint loading, GPU work,
  generation, training and pushing. All 54 gates are closed.
- **Status:** awaiting the user's review before the real survey. See do-not-repeat **#127**.

#### Addendum — the two-band method accepted; the generation seed preparation approved as one conditional sequence, with the closing copy (2026-10-30, the user's decision)

- **Accepted:** the two-band seed method and training's admission fix (`f870de24`). The rule may use repository
  literals for its anchor; that is outcome-blind bookkeeping, and counting the hex-digest fragment conservatively is
  acceptable. The user verified the pinned methods intact, the registry and historical evidence unchanged, all 54
  gates closed and no generation plan or output; training now needs generation's successful committed inner and outer
  records.
- **Approved, as one conditional sequence:**
  1. run the real generation seed survey once;
  2. only if it is CLEAN with 55/55 controls: commit its record, then run the pre-proof once;
  3. only if the pre-proof HOLDS: register exactly the surveyed block in ACCOUNTED and create the plan through
     `build_plan`, the other registry lists unchanged;
  4. run the post-proof once, then update the affected lifecycle checks and run focused CPU tests.
- **If any check fails:** stop and return the result; the block is never adjusted.
- **Also approved, alongside:** the generation seed-closing copy, by the reviewed parallel-worker rule, with focused
  CPU checks.
- **Not authorized:** checkpoint loading, GPU generation, training and pushing.
- **Prediction on record:** none.

#### Addendum — generation's seed block surveyed, proven and registered in the 2027 band; the closing copy built (2026-10-30)

Each procedure ran once in a fresh clean linked worktree, under `/Users/bill/twixt-worktrees/`, with the main
checkout's `.venv/bin/python -B`. Each record was committed in its worktree, main was fast-forwarded as a separate step,
and the worktree was removed.

- **The survey (`bb1549ea`, worktree `cp-gen-seeds` from `4fff63bc`):** `--mode survey`, exit 0, nothing on stderr.
  - The rule's anchor is 202738776 (the hex-digest fragment in the tactical candidates corpus; nothing is registered
    in the 2027 band), so BASE = 202739800 and the block is **[202739800, 202740800)**. The rule read the literals of
    this tree and of the main checkout (7,141 and 49,863 files).
  - The check: direct none, streams 0, injective, gap **1,023** (floor 1,000), CLEAN. Controls **55/55**, clean tree.
    Registry sha256 `d041e293…` (ACCOUNTED 48, EXPOSED 46, RETIRED 44, TEST_ONLY 1).
  - Verdict: **THE COMBINATION-POLICY GENERATION BLOCK IS CLEAN**. Record `02_seed_survey_run.txt`, sha256
    `69ce5b87…`; 1,000 slot lines.
- **The pre-registration proof (`4701b1d3`, worktree `cp-gen-pre` from `bb1549ea`):** `--mode pre`, exit 0, nothing on
  stderr.
  - Bound to the survey record (`69ce5b87…`, read at `4fff63bc`); the block read from it.
  - **EXPECTED STATE (pre): HOLDS**; CLEAN, gap 1,023; controls **55/55**. Record `03_collision_proof_pre_run.txt`,
    sha256 `295ce85c…`.
- **The registration (`326004d0`, worktree `cp-gen-register` from `4701b1d3`):**
  - A script checked everything before writing:
    - the survey record's hash, and the block read from it equal to the surveyed one;
    - the registry unchanged since the survey, and no plan;
    - a comment holding no literal of either band.
  - It appended [202739800, 202740800) last to ACCOUNTED (48 → 49) and wrote `combination-policy-generate-run/plan.json`
    = `build_plan(202739800, 69ce5b87…)`, create-only.
  - An AST comparison against the parent: only ACCOUNTED changed, by exactly that tuple; every other statement is
    unchanged. The plan equals `build_plan`.
  - Its first attempt stopped on a line-length assertion before anything was written; the comment was rewrapped and
    the script rerun.
- **The post-registration proof (`5a17ea92`, worktree `cp-gen-post` from `326004d0`):** `--mode post`, exit 0, nothing
  on stderr.
  - **EXPECTED STATE (post): HOLDS**; CLEAN, gap 1,023; controls **55/55**. The registry it proved has sha256
    `ee1070bb…`, the current file.
  - Record `04_collision_proof_post_run.txt`, sha256 `f3c3d5dc…`.
- **The closing copy** `combination-policy-generate-seed-closing/01_derive_usage.py` (`e8ae01cc`) is throughput attempt
  2's with only the paths, N_SLOTS 1000 and the title changed. The reviewed parallel-worker rule is unchanged:
  - the started slots must form {0..k−1} as a set, and the one final `claims` event must equal k; otherwise the whole
    block is exposed;
  - the whole block is retired.
  - Its 21 controls pass at the registered base. Built, not run.
- **Tests re-anchored (`e8ae01cc`):**
  - The generation state test now asserts:
    - the three records, their verdicts, 55/55 and clean trees, and both proofs bound to the survey hash;
    - the plan = `build_plan`, and the block read from the plan (no literal in tests): the first block in the 2027
      band, ACCOUNTED once and last, in no other list, every seed schedulable;
    - the proved registry is the current one;
    - nothing run: the gate closed, the run directory holding only the wrapper and the plan, no evidence, closing
      record or corpus;
    - both throughput runs unchanged.
  - The seed-method state test now expects exactly this block in the 2027 band. The synthetic rehearsal now plants its
    2027 literal above the registered block, since its prior terms are the real ones.
  - New closing-copy tests: the diff, the paths, no literal, and 21/21.
  - Temporary breakages (the registry entry removed; the closing N_SLOTS set to 144) each fail 2 tests; both were
    restored byte-equal.
  - The combination, seed-method, gate-inventory, background-compute, mirror-halving launch, seed-selection and
    tactical survey/proof suites: **315 passed**, 1 deselected.
- **Disclosed:** the pre-proof record's commit command sent its own output to `/dev/null`, but it printed the commit's
  exit code (0), and the commit is in the log. No failure was hidden; the other commits printed their output.
- **Not done:** opening the gate, checkpoint loading, GPU generation, closing, training and pushing. All 54 gates are
  closed.
- **Status:** the generation run's seed preparation is complete; awaiting the user's launch review. See do-not-repeat
  **#128**.

#### Addendum — the one 1,000-game generation run approved, with its conditional seed closing (2026-10-30, the user's decision)

- **Verified by the user:** all three seed records and their hashes, 55/55 recorded controls in each, the exact
  ACCOUNTED append, the plan binding and the current registry and scanner hashes; the closing copy preserves the
  reviewed rule; all 54 gates closed; no generation output.
- **Approved, as one sequence:**
  1. record this decision; a fresh read-only panic/reboot check; recheck the proved registry and the plan;
  2. in a clean worktree, open **only** the generation gate by a one-line commit and run the existing wrapper once,
     BOINC quit, the idle gate and the background-compute check enforced;
  3. the limits unchanged: TERM at 21,590 s, KILL at 21,600 s, cleanup observation through 21,610 s;
  4. after a started run: restore the gate and run K1 once. Only if it RECONCILES: apply its exposure verdict, retire
     the whole block, and update the lifecycle checks. If it does not reconcile, stop before any registry edit;
  5. commit the evidence and the closeout, and return for the corpus and training-readiness review.
- **If generation is incomplete:** preserve the partial capture, and bring back the progress, the cause and the likely
  cost of continuing for assessment. No automatic retry.
- **This run produces the actual training corpus.** Training and pushing remain unauthorized.
- **Prediction on record:** none.

### 43. Combination policy pilot — step 3, CORPUS GENERATION: 1,000 games (2026-10-30, COMPLETED; block closed; awaiting the corpus and training-readiness review)

- **Test:** amendment `docs/superpowers/2026-10-30-combination-policy-generation-amendment.md` (sha256 `11ced081…`) to
  card §5 step 3: 1,000 games of calib020_0001 playing itself (eval mode, mirror averaging, 200 simulations, root
  sequential halving from ply 20, 24×24, `MLX_ENABLE_TF32=0`), 18 workers, inference cap 14, slot g on seed BASE + g
  of the registered 2027-band block [202739800, 202740800), inside a six-hour ceiling (TERM at 21,590 s, KILL at
  21,600 s, cleanup observation to 21,610 s). Admitted by the committed CPU timing alone. A run the ceiling stops is
  VOID and assessed. These games are the pilot's training corpus.
- **Hypothesis:** the 1,000 games finish inside six hours and leave a complete, verifiable corpus outside the
  repository.
- **Why it was plausible:** attempt 2 (entry 42) finished 143 of 144 games at about attempt 1's pace; its summed
  game time (about 20 minutes for 144) projects roughly 2.3 h of work for 1,000 games over 18 workers, and the
  six-hour ceiling leaves room for a state-cap tail of the kind that sank attempt 2's 30-minute run.
- **Prediction on record:** the approval addendum stated none. The card's §7 "Estimated cost (unmeasured …)", written
  2026-10-28 before any run, gives "Generation, 1,000 games ~1.5–3 h (roughly 350–700 games/h)"; the amendment adds
  that six hours "does not guarantee that the corpus will finish". No prediction was stated for the row count beyond
  the card's §4 estimate ("about 37,000 rows").
- **What ran:** decision `92bdb38e`. A fresh read-only panic/reboot check (no reboot since 2026-10-25 09:39; the only
  panic report is that one; the newest report a `spotlightknowledged` CPU-resource report at 16:06, not a panic); load
  about 1.35; BOINC check CLEAR (podman exempt). Recheck: registry sha256 `ee1070bb…` = the post-proof's, unchanged
  since `326004d0`; plan unchanged; no gate open; checkpoint present; no generation output. Clean linked worktree
  `cp-gen-run`; the one-line gate opening `acf07d24` (only `COMBINATION_POLICY_GENERATE_AUTHORIZED`); the existing
  wrapper run once at 16:53:31 EDT, ended 19:41:01. The idle gate held 60 s (passed at 121 s, after load fell below
  3.0), the background check CLEAR, then launch. Evidence `56e6f386` (the worktree's evidence commit, with the
  restored gate; main fast-forwarded to it).
- **Result:**
  - **Wrapper:** inner exit **0**, outer exit 0, outcome **COMPLETED**, `gate_restored` true, survivors empty,
    `record_by` script, wall **9,929 s** (2.76 h) of the 21,590 s ceiling. `outer_faults` is empty. `outer.json`
    sha256 `b6b2e7c8…`.
  - **Inner record** (sha256 `df9ecaef…`): status **COMPLETED**, run `generate`, the amendment's sha256 recorded, seeds
    {base, 1,000 slots}. TF32 "0" at start and end; 0 of 42 modules in training mode at load and at end; 1,002 BOINC
    checks, all clear; stderr empty. Generation 9,928.7 s, **362.6 games/h**; 1,897,590 arbiter calls, 13.39 rows per
    call (cap 14).
  - **Slots:** telemetry claims, started and ended all **1,000**; `progress.jsonl` (sha256 `db4aa7b9…`) has 1,000
    `game_start`, 1,000 `game_end` and one `claims` event. `games.jsonl` (sha256 `e75ba513…`) has one line per slot,
    slot g on seed BASE + g.
  - **Corpus** (read for slot, seed, file, hash, row and move counts only — never a winner or reason): 1,000
    create-only files in `~/twixt-cache/combination-policy-pilot/generate/` (314 MB), exactly the files
    `games.jsonl` names, **0 sha256 mismatches**. **44,747 rows** (ply ≥ 20; 9–260 per game, no game without rows):
    **40,589 training** rows from 900 games and **4,158 held-out** rows from the 100 games with slot ≡ 9 (mod 10).
    The record's summary: 0 duplicate rows.
  - **Descriptive timing, derived from the committed logs (not a measurement the card defines):** games took mean
    153.3 s (median 121.7, range 63.8–1,778.6) and 64.7 moves (median 54, range 29–280). **43 games reached the
    280-move state cap**, taking 664–1,779 s each (median 696 s, 32,897 s in all, 21% of the 153,275 s of game time);
    every other game took at most 524 s. Per-worker busy time 8,182–9,928 s.
  - **Concurrent activity:** a scheduled job outside this repository (launchd coalition
    `com.user.autotrade.weekly-review`) wrote a disk-writes report for 17:30–17:35, and `spotlightknowledged` a
    CPU-resource report at 17:32. The launch checks do not cover them; whether they slowed the run is not established.
  - **Admission:** through the real committed readers, `admitted("train")` now admits (the timing record, the
    generation record and its `outer.json`, each COMPLETED). The training gate is closed.
  - **Post-run:** no new panic report and no reboot; all 54 gates closed.
- **Did it match the prediction?** Yes for time: 2.76 h at 362.6 games/h, inside the card's 1.5–3 h range, near its
  slow end, and 37% of the ceiling. The state-cap tail recurred (43 games, 4.3%, against 4 of 143 in attempt 2) and
  cost about a fifth of the game time, but with 1,000 slots the long games overlapped the rest of the work instead of
  ending the run. The rows exceeded the card's estimate: 44,747 against about 37,000, because games ran longer (mean
  64.7 moves against strength2's 56.8, the basis of that estimate).
- **Closing:**
  - **K1 (`4cf54bac`), run once** in a clean linked worktree (`cp-gen-k1`, from `56e6f386`), exit 0: **RECONCILES**,
    controls **21/21**. Inputs committed and unmodified; outer exit 0, survivors '', status COMPLETED; the 1,000
    started slots form {0..999} as a set and the one `claims` event is 1,000, so k = 1,000 and the **whole block is
    EXPOSED**; the whole block is RETIRED. Record `combination-policy-generate-seed-closing/02_derive_usage_run.txt`,
    sha256 `ac143aba…`.
  - **K2 (`569ba2a4`):** the verdict read from K1's committed record and checked against the plan; [202739800,
    202740800) appended last to EXPOSED (46 → 47) and to RETIRED (44 → 45), with comments holding no seed literal. AST
    check against the parent: only those two lists changed, ACCOUNTED (49) and TEST_ONLY (1) unchanged. Every seed of
    the block is refused; its neighbours are not. Registry sha256 now `10f76e3b…`.
- **Tests re-anchored (6e8032ab):** the generation lifecycle test now asserts the run-and-closed state (one run from
  the one-line gate opening on the post-proof's state, outer COMPLETED with `outer_faults` empty, record COMPLETED with
  the amendment's sha256 and every slot claimed, started and ended, TF32 0 and eval mode throughout, the proved registry
  = the launched one, games.jsonl's slots and seeds read without outcomes, K1 RECONCILES with `claims` 1,000, the whole
  block EXPOSED and RETIRED once and last, every seed refused, the run evidence unchanged since `56e6f386`), keeping the
  survey and proof checks and no seed literal. The two-band real-state test requires the block to be the 2027 band's
  only entry in ACCOUNTED, EXPOSED and RETIRED; attempt 2's test requires every later EXPOSED/RETIRED entry to lie
  above its block. The focused suites: 296 passed, 1 deselected. The full fast suite was run in a linked worktree alongside a baseline run at 92bdb38e (the pre-run main): 174 failure ids common to both are pre-existing (ignored files absent from a linked worktree: the checkpoint, .venv and game logs; the frozen guards of the 58-id regression baseline; load-sensitive fake launches; and the external-learning seed-closing tests, which fail at baseline too because stage 2's EXPOSED entry differs from its RETIRED entry). Every id that differed between the two (fake launches under the doubled load, and real-state checks refusing the then-uncommitted test edits) passed when its 8 files were rerun alone after the commit: 235 passed.
- **For the readiness review (derived, not measured):** the committed training ceiling, 2,700 s, was priced from the
  card's estimated counts (37,000 rows, 3,700 held out). The timing record's own formula on the real counts (44,747
  and 4,158; the 3,000 updates do not scale with rows) gives about 1,865 s, so the ceiling is about 1.45× that
  projection rather than the card's 1.5×; the card's rule on the real counts would give 2,820 s. Loading and replay
  are not timed. Whether to keep the committed ceiling is the user's decision.
- **Conclusion / status:** corpus generation **COMPLETED** and its block is **closed**. The corpus is complete and
  verifiable against the committed hashes. There is still no policy-learning result. Training, checkpoint loading for
  training and pushing remain unauthorized; the corpus and training-readiness review is next. See do-not-repeat
  **#129**.

### Decision note — after entry 43: generation and its closing accepted; the training ceiling set at 2,820 s; the training wrapper to be prepared (2026-10-31, the user's decision)

- **Accepted:** generation (entry 43) and its seed closing. The user's own read-only review confirmed all 1,000 file
  hashes, the 40,589 / 4,158 training–held-out split, complete slot accounting, unchanged evidence and all 54 gates
  closed. This establishes data readiness only; learning is untested.
- **The training ceiling (card §5 step 4, "stated before approval"): 2,820 s (47 minutes).** The timing record's own
  formula on the real corpus counts (44,747 rows, 4,158 held out; the 3,000 updates do not scale with rows) gives
  1,865.3 s; the card's 1.5× rule, rounded up to the minute, gives 2,820 s. The user derived 1,865.3 s independently.
  The wrapper enforces it: TERM at 2,810 s, KILL at 2,820 s, cleanup observation through 2,830 s. The timing record
  (`combination-policy-timing/record.json`, `ceiling_seconds` 2,700, priced from the card's estimated 37,000 / 3,700)
  is kept as written: nothing edits or relabels it, and no new timing run is made.
- **Approved now: minimal training-wrapper preparation.** A copy of the reviewed generation wrapper for the CPU
  `run train`, with that ceiling. The frozen recipe, the starting checkpoint (calib020_0001), the admission checks and
  the STOP rules are unchanged. Focused fake-launch checks are sufficient. The build returns for launch review.
- **Approved separately: a limited repair of the external-learning seed-closing tests' stale assertions.** They
  require the later portions of EXPOSED and RETIRED to match, but an interrupted run legitimately exposes a prefix and
  retires its whole block (stage 2: 2 seeds exposed, 12 retired, each exactly once). The repair keeps the
  seed-unavailability checks and leaves the registry and all evidence untouched. It does not block training.
- **No further full-suite run is needed.** The next experiment is CPU training on this corpus — not another generation
  or throughput run.
- **Not authorized:** opening the training gate, training and pushing.
- **Prediction on record:** the card's, unchanged ("directional only. The candidate's held-out cross-entropy falls
  (step 4) …"); none added here.

#### Correction to entry 43 (2026-10-31)

- Entry 43 says the external-learning seed-closing tests fail "because stage 2's EXPOSED entry differs from its
  RETIRED entry". That is wrong. Stage 2's seed is exposed and retired exactly once. The differing entry is **D2's
  held-out set** (K2 `164774e2`, 2026-10-25: [202692300, 202692302) exposed, the 12-seed block [202692300, 202692312)
  retired). All three tests (stages 1, 2 and 3) compared the two lists' portions after their own block, so every
  later partial exposure broke them. The rest of entry 43 stands.

#### Addendum — the training wrapper built; the external-learning closing tests repaired (2026-10-31)

- **Training wrapper (`f50465f8`), built only:** `combination-policy-train-run/01_wrapper.sh` is generation's
  wrapper with only the paths (`combination-policy-train-run/`, `combination-policy-train/`), the training gate
  (`COMBINATION_POLICY_TRAIN_AUTHORIZED`, closed), `run train` and the ceiling changed: TERM_S 2,810, GRACE_S 10, so
  KILL at 2,820 s and the cleanup observation through 2,830 s. The idle gate, the background-compute check, the
  process-group supervisor, the survivors, the records and the gate restoration are byte-identical. It refuses unless
  HEAD is the one-line gate opening. The pilot module is not edited, so the recipe, the starting checkpoint
  (calib020_0001 from the main checkout), the admission checks and the STOP rules are unchanged; the timing record is
  kept as written.
- **Its checks (`tests/test_combination_policy_train_launch.py`, 10):** the built-not-run state (gate closed, the run
  directory holds the wrapper only, no training evidence; `admitted("train")` through the real committed readers
  admits on the timing, generation's record and its `outer.json`; the timing record still says 2,700 s, unchanged);
  the line-by-line diff against generation's wrapper; the ceiling re-derived from the committed timing record and
  the committed corpus counts (44,747 and 4,158 → 1,865.3 s → 2,820 s); the paths; and fake launches of the real
  wrapper's bytes: a busy machine, BOINC or a failed check refuses with nothing launched; a gate opening that also
  changes another file refuses; an idle machine reaches exactly `run train --checkpoint <calib020_0001>` with
  TF32=0, TERM_S 2,810 and a clean outer record; a shortened ceiling ends the whole process group with the wrapper's
  VOID record in training's evidence directory. Four wrapper mutations (the ceiling, the subcommand, the idle gate,
  the gate check) were each caught. Focused suites: 158 passed, 1 deselected.
- **External-learning test repair (`e7b35d5d`):** the three stale assertions now require only that nothing at or
  below each block follows it in EXPOSED or RETIRED; every other check is kept (exactly once in ACCOUNTED, EXPOSED
  and RETIRED, no overlap, nothing just outside, every seed accounted, exposed, retired and not test-only). The
  registry and all evidence are untouched. 60 passed; a later low entry in either list fails all three.
- **Not done:** opening the training gate, training and pushing. All 54 gates closed. Awaiting launch review.

#### Addendum — the ONE CPU training run approved; a clarification to #130 (2026-10-31, the user's decision)

- **Verified by the user:** the wrapper has exactly the six executable-line changes reported; the 2,820 s budget
  checks out; the recipe, the admission checks and the evidence are unchanged; all 54 gates are closed. The limited
  test repairs are accepted. The user ran no tests or experiments.
- **Approved, as one sequence:**
  1. record this approval; a fresh read-only panic/reboot check; the existing idle and BOINC checks enforced;
  2. in a clean worktree, open **only** `COMBINATION_POLICY_TRAIN_AUTHORIZED` by a one-line commit and run the
     wrapper once;
  3. the frozen weights-only starting checkpoint (calib020_0001), CPU-only training and the unchanged recipe: 3,000
     updates, subject to the existing STOP rules; TERM at 2,810 s, KILL at 2,820 s, cleanup observation through
     2,830 s;
  4. restore and verify the gate, commit the evidence, re-anchor the lifecycle test with focused checks, and return
     the initial KL, the probability discrepancy, the held-out CE before and after, the value-preservation checks,
     the verdict and the cleanup results.
- **Clarification to #130 and to the wrapper header's "closes the pilot":** a timeout makes this run **VOID**, with
  **no automatic retry**. It does not permanently reject the learning method. If the run is interrupted, the progress,
  the cause and the likely cost of continuing come back for the user's assessment. (The wrapper's bytes are not
  edited for this; its header is read with this clarification.) See #131.
- **No further timing or full-suite run is needed.**
- **Not authorized, even if training reports ELIGIBLE:** strength games (the screen included), deployment and
  pushing.
- **Prediction on record:** the card's, unchanged ("The candidate's held-out cross-entropy falls (step 4)"; directional
  only; confidence low); none added here.

### 44. Combination policy pilot — step 4, CPU TRAINING, then the check (2026-10-31, COMPLETED; verdict ELIGIBLE; awaiting review)

- **Test:** card §5 step 4 on the committed corpus (entry 43): calib020_0001 weights-only, on the CPU, arm R's residual
  policy branch installed (init_seed 20261028) and checked bit-equal to the parent; card §4's one objective, the
  mirror-averaged policy p̄; 3,000 Adam updates of batch 64 (lr 1e-3, l2 1e-4, order from PCG64(20261029)) on the
  40,589 training rows; then the STOP rules in order: pre-fit KL(π ‖ p̄_θ₀) below 0.01 nats; any non-finite loss or
  gradient; value outputs not bitwise equal on the held-out rows, or the encoder or value-head digests changed;
  held-out CE(π, p̄_θ) not below CE(π, p̄_θ₀). Ceiling 2,820 s (TERM 2,810 s; decision note after entry 43).
- **Hypothesis:** the corpus holds a learnable policy signal: the policy head, trained toward the halving target,
  moves the averaged policy toward it on held-out games while the value path stays exactly the parent's.
- **Why it was plausible:** generation's summary showed the target differs from the prior (mean KL(target ‖ prior)
  about 1.43 nats, entry 43's record), and only the policy head's surface is trained, so the value path cannot move.
- **Prediction on record:** the card's: "directional only. The candidate's held-out cross-entropy falls (step 4)"; no
  magnitude; confidence low (§8). Nothing was added before the run.
- **What ran:** approval `9d6d48bf` (#131). A fresh read-only panic/reboot check (no reboot since 2026-10-25 09:39;
  the only panic report is that one; a JetsamEvent report at 10:20 that morning, not a panic); load about 1.0;
  BOINC check CLEAR (podman exempt). Nothing in `scripts/` or `docs/superpowers/` changed since the build
  (`f50465f8`); no gate open; no training evidence; checkpoint SHA-1 `209cf2d4…`. Clean linked worktree
  `cp-train-run`; the one-line gate opening `9b7479f7` (only `COMBINATION_POLICY_TRAIN_AUTHORIZED`); the wrapper run
  once at 11:42:02 EDT, ended 12:15:04. The idle gate held 60 s (passed at 116 s), the background check CLEAR, then
  launch. Evidence `578577f2` (the worktree's evidence commit, with the restored gate; main fast-forwarded to it).
- **Result:**
  - **Wrapper:** inner exit **0**, outer exit 0, outcome **COMPLETED**, `gate_restored` true, survivors empty,
    `record_by` script, wall **1,866 s** of the 2,810 s TERM. `outer_faults` is empty; stderr empty; no pilot
    process remained.
  - **Record:** status **COMPLETED**, verdict **ELIGIBLE**, no STOP. Device CPU; TF32 "0" at start and end; versions
    the card's; checkpoint SHA-1 = calib020_0001's; admission on the timing record, generation's record and its
    `outer.json`. Corpus 1,000 games, 44,747 rows: 40,589 training, 4,158 held out (slot % 10 == 9). Surface: the
    policy head's 12 arrays, 8,840 parameters.
  - **Pre-fit (the parent, θ₀):** training rows KL(π ‖ p̄) **1.4339** nats (floor 0.01), CE 3.9333; held-out KL
    1.4113, CE **3.8620**. Probability discrepancy |p̄ − stored prior| over legal moves, the largest over the rows:
    **1.03 × 10⁻⁵** (training) and **4.74 × 10⁻⁶** (held-out): the recomputed parent policy agrees with the prior
    the corpus stored.
  - **Fit:** 3,000 updates, every loss and gradient finite, 1,458.4 s. Batch loss mean 3.782 over the first 100
    updates and 3.712 over the last 100 (descriptive; the batch loss is noisy and includes the l2 term).
  - **Value preserved:** value outputs **bitwise equal** to the parent's on all 4,158 held-out rows (max |Δ| 0.0);
    the encoder and value-head digests unchanged; the policy head's digest changed.
  - **Held-out after:** CE **3.6454** (from 3.8620: −0.2166 nats, −5.6%); KL 1.1947 (from 1.4113, the same −0.2166,
    since the targets' entropy is fixed). Probability discrepancy 0.696 against the stored (parent) prior:
    descriptive; the head has moved, and the measure does not bound the KL.
  - **Output:** `head.npz` (sha256 `ead156e2…`), the trained head's arrays.
  - **Phase times:** corpus verification 12.7 s, load 15.8 s, pre-fit measurement 333.0 s, fit 1,458.4 s, check
    46.2 s. The timing record's projection for the updates (1,461.6 s) matched the fit.
  - **Post-run:** no new panic report and no reboot; all 54 gates closed.
- **Did it match the prediction?** Yes, in direction: the held-out cross-entropy fell, by 0.217 nats, with the value
  path bit-identical. No magnitude was predicted. This is a held-out fit to the halving target on games from the
  same agent and distribution; it says nothing yet about playing strength.
- **Tests re-anchored (`1d5c3080`):** the training lifecycle test asserts the ran-once state from the committed
  records (outer COMPLETED inside TERM from the one-line opening; COMPLETED/ELIGIBLE; CPU, checkpoint and recipe
  unchanged; real counts; pre-fit KL at or above the floor; 3,000 finite losses; values bitwise equal and the
  encoder and value-head digests unchanged, the policy head changed; held-out CE below pre-fit; the head's sha256;
  the timing record unchanged at 2,700 s). Focused suites: 158 passed, 1 deselected.
- **Conclusion / status:** step 4 is **ELIGIBLE**: by the card, the screen (step 5) may be proposed. No seed was used.
  There is no strength result. Strength games (the screen included), deployment and pushing remain unauthorized
  (#131). See do-not-repeat **#132**.

#### Addendum — training ACCEPTED as ELIGIBLE; build-only preparation of the 64-game screen approved (2026-10-31, the user's review)

- **Accepted:** the training result, as **ELIGIBLE** (entry 44). The held-out CE fell by 0.2166 nats and the recorded
  value checks passed. The user independently verified the saved head's hash and parameter digest, that the source
  and evidence are unchanged, the gate's restoration, and that all 54 gates are closed. No blocking finding.
- **The user's reading:** the pilot has now shown learning toward the search targets. Whether it improves play is the
  next unanswered question.
- **Approved: build-only preparation of the card's 64-game screen (§5 step 5):**
  - the trained head against the frozen proven combination, both with mirror averaging, 200 simulations and halving
    from ply 20;
  - the candidate bound to this exact saved head (`head.npz`, sha256 `ead156e2…`) and to the successful inner and
    outer training records;
  - the existing match loop and wrapper reused, with a new closed gate, separate evidence paths and fresh-seed survey
    and closing copies; the build to confirm that search reaches the residual head through `forward_padded`;
  - the frozen rule kept: a mean pair score below 0.40 is STOP, otherwise CONTINUE; the screen makes no strength
    claim;
  - a fixed ceiling proposed from completed match timings, allowing for long capped games; no additional timing or
    profiling run is needed;
  - focused CPU tests and fake launches; the build returned for review.
- **Not authorized:** seed procedures, checkpoint loading for evaluation, GPU games, further training, deployment and
  pushing.
- **Prediction on record:** the card's, unchanged ("it scores above 0.5 against the proven agent (steps 5–6)";
  directional only; confidence low); none added here. The screen's own rule is gross regression only.

#### Addendum — the 64-game screen BUILT for review; its ceiling proposed (2026-10-31)

Build `6c908ca8`, as approved in the addendum above. Nothing was run: no seed procedure, no checkpoint or head loaded for
evaluation, no GPU, no game. All 55 gates are closed.

- **The entry:** `combination_policy_pilot run screen` (card §5 step 5), behind a new closed gate,
  `combination_policy_screen_authorization` (the fifty-fifth). Its own paths: `combination-policy-screen` (the
  script's) and `combination-policy-screen-run` (the wrapper's, with the plan the seed steps would add).
- **Admission and the head binding.** The screen is admitted only by the training run's committed `record.json`
  (COMPLETED, verdict ELIGIBLE, this card) and its wrapper's committed `outer.json`, through the existing
  `outer_faults` (inner and outer exit 0, COMPLETED, gate restored, no survivors, record by the script, the same head).
  `TRAINED_HEAD` pins the head:
  - `head.npz` sha256 `ead156e2…`, and the record's digests after training (policy head `f653aca2…`; encoder
    `92498b02…` and value head `576391a7…`, both also the record's digests before training);
  - the record must name exactly these, and the committed file must be these bytes;
  - the bytes checked are the bytes loaded.
- **The candidate and the reference.**
  - calib020_0001 is loaded twice, with the strength match's loader (`eval_runner._default_evaluator_factory`,
    eval mode, compiled).
  - One copy gets arm R's head (the recipe's init_seed) with every saved array loaded over it. Its digests must equal
    the pinned ones, or the run is VOID at step "head".
  - The other copy must keep the parent's encoder and value-head digests and a different policy head, or the run is
    VOID at step "reference".
  - The head is installed before any inference, so the compiled forward pass is traced with it.
- **Search reaches the residual head through `forward_padded` (card §10.10), confirmed by the build.**
  - The evaluator calls `network.forward_padded`, which calls `ResidualPolicyHead.forward_padded`. The head's
    `__call__` (the list API) refuses.
  - In the tests on tiny random CPU networks, every underlying evaluation of a real 200-simulation halving search at
    ply 20, made through the averaging wrapper, went through `ResidualPolicyHead.forward_padded`.
  - The compiled evaluator's averaged prior equals training's own prediction, exp(log p̄), on positions from several
    plies.
  - At run time, a probe at the empty board must find the candidate's priors different from the reference's, or the
    run is VOID.
- **The games:** 32 colour-reversed pairs, with game g on seed plan.base + g, through the strength match's loop
  (`mirror_halving_combination.play_and_build`). Both sides are `MirrorAveragingEvaluator` +
  `RootSequentialHalvingMCTS` at 200 simulations, with halving from ply 20. The candidate is red in even games.
  - `game_start` is logged before each game.
  - The background-compute check runs before loading, after every game and before the verdict.
  - Each game's searches must account for its moves, with both sides searching.
  - The frozen rule: `screen_verdict`, STOP below a mean pair score of 0.40, otherwise CONTINUE. There is no strength
    claim, and the cost ratios are descriptive only.
- **Found in the build (measured on tiny CPU networks, not on the real model):** the compiled evaluator's value output
  can differ from the parent's in the last bits (1.45 × 10⁻⁷ there). Compilation fuses the graph differently once the
  residual branch exists; the parent's own compiled and uncompiled values differ by the same amount. Uncompiled, the
  values are bitwise equal. The run records the opening value difference (`head_probe.value_abs_diff`) and does not
  gate on it. Entry 44's bitwise check was uncompiled, on the CPU, and stands.
- **The wrapper:** `combination-policy-screen-run/01_wrapper.sh` is a copy of training's. It differs only in its paths,
  its gate, `run screen`, TERM_S 3470 and its header. The idle gate, the BOINC check, the supervisor, its bounded
  observation, the records and the gate restoration are byte-identical.
- **Proposed ceiling, for the user's review: 3,480 s** (TERM 3,470 s, KILL 3,480 s, cleanup observation through
  3,490 s). The card prices the screen from the throughput check, which never completed. Per the user's instruction,
  it is priced instead from completed match timings, reading per-side costs and move counts only:
  - per search: 0.5508 s per combination search, strength2's candidate side (29,453 searches; the same loop, loader,
    machine and TF32=0);
  - per game: 64.75 plies, from generation's 1,000 games of this agent against itself (strength2's games against the
    incumbent averaged 57.7);
  - projection: 64 games × 64.75 plies × 0.5508 s = 2,282.3 s; × 1.5, rounded up to the minute, gives 3,480 s;
  - capped games: a 280-ply game at the slowest per-search rate of any capped game in the two completed combination
    matches (0.6285 s) takes about 176 s;
  - room: with the other games at generation's uncapped mean (55.1 plies), TERM leaves room for 10 capped games of
    the 64; 2.75 are expected at generation's 4.3% rate, and the chance of 11 or more at that rate is about 8 × 10⁻⁵;
  - a candidate that lengthens games more than that would time out: a VOID, with no automatic retry (#131's
    reading).
- **The seed copies, built and never run:**
  - **Survey** (`combination-policy-screen-seed-survey/01_seed_survey.py`): a copy of generation's (the two-band
    method). It differs in the block size (64), the paths, the plan kind, the module name, the titles and verdict
    lines, and the slot listing's heading. One control's expected value is corrected: on an empty band the rule gives
    the least multiple of 100 at least 64 above the edge, not the edge + 64 that the copied control wrote out for
    1,000. The rule itself is unchanged. The three modes were rehearsed on a synthetic repository (55/55 controls).
  - **Closing** (`combination-policy-screen-seed-closing/01_derive_usage.py`): a copy of strength2's, not
    generation's. The screen plays its games in order in one process and logs the match's events, so the sequential
    rule applies: the started games must be the prefix 0..k−1 in order, and only those are exposed. It differs in
    the block size (64), the paths and the title. All 15 controls pass, and it reconciles the screen's own events
    (complete: the whole block exposed; VOID in game 9: games 0–9 exposed; retired whole either way).
- **Tests:** `tests/test_combination_policy_screen.py` (new). The pilot's tests are updated: the kind list, the
  outer-record admission, the usage case and the isolation. The gate inventory is now 55. Focused suites, run serially (all six combination-pilot files, seed band, gate inventory, the mirror-halving match and launch, the registry): 390 passed, 1 deselected; after one added test and line wrapping, the screen and pilot files: 148 passed, 1 deselected. No full-suite run.
- **Status:** built, for review. Seed procedures, checkpoint loading for evaluation, GPU games, further training,
  deployment and pushing remain unauthorized (#133). The ceiling is a proposal until the user accepts it.

#### Addendum — the screen build and its 3,480 s ceiling ACCEPTED; the screen's seed preparation approved as one conditional sequence (2026-10-31, the user's review)

- **Accepted:** the screen build (`6c908ca8`) and the ceiling, fixed at **TERM 3,470 s, KILL 3,480 s, cleanup
  observation through 3,490 s**. The user's read-only review found no blocking issue: the head binding, the loading
  order, both agents' search settings, the wrapper and the seed copies match the approved scope; all 55 gates are
  closed and prior evidence is unchanged.
- **The user's reading of the budget:** the arithmetic checks out; its long-game allowance is an estimate based on the
  parent's games, not a completion guarantee for the trained candidate.
- **The compiled-value caveat, accepted as disclosed:** entry 44's CPU bitwise-preservation result stands; the
  tiny-network observations do not establish exact equality on the real compiled GPU model.
- **Approved, as one conditional sequence:**
  1. run the screen's seed survey once; proceed only if it is CLEAN with 55/55 controls;
  2. commit its record, then run the pre-proof once; require HOLDS;
  3. register exactly that block in ACCOUNTED and create its plan through `build_plan`;
  4. run the post-proof once; require HOLDS; then update the lifecycle test with focused CPU checks.
- **If any check fails:** return the result; the block and the method are never adjusted. No additional engineering,
  profiling or full-suite run.
- **Not authorized:** the launch (the registered plan and the post-proof are returned for that decision), checkpoint
  loading for evaluation, GPU games, further training, deployment and pushing.
- **Prediction on record:** none added.
