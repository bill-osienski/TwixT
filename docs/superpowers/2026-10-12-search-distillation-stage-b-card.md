# Search distillation — Stage B design card: the smallest meaningful learning pilot (DESIGN ONLY, rev 2, 2026-10-12)

**Status.** This is revision 2, a draft for review. It is not committed.

**It authorizes nothing.** Each of these needs its own explicit approval:
* infrastructure;
* seed accounting;
* model loading;
* games, position selection, labels and training;
* strength games.

`calib020_0001` remains the incumbent in every outcome.

**Revision 2 changes.**
* L2 now requires a gain over the incumbent as well as over the temperature arm.
* Position selection counts as seed use, and exposure is the union of used slots.
* Four positions per game, not 16.
* Q is an operational qualification gate, and its edge cases are defined.
* "Repeat ceiling" is now "repeat agreement".
* The state hash and the full legal support are frozen.
* The interpretation is tightened.
* 6.11 ms is an observed-rate scenario, not a bound.
* `chosen_ce` is fed soft targets unchanged.
* Accepted with two corrections: phase P has its own 5-minute cap and L1 has 85 minutes (together the
  former 90; the total stays 17.5 h), and the label call builds `EvalConfig(**PROTOCOL)` before
  `cfg_from`.

**Basis.** Stage A passed under its stated independence assumption (`480e0f9b`): mean pair score 0.768750,
lower bound 0.616910, for the incumbent at 1,600 simulations against itself at 400. That is a playing
advantage for the label-producing search under that protocol. It does not show that its visit
distributions can be learned, or that learning them helps. Stage B asks only whether they can be learned.

**Nothing from A1 or Stage A is reused.** No game, position, move or outcome from either is used for
any choice here. Their only use is the timing in §10: wall seconds and ply counts from
`progress.jsonl`, which holds no outcomes.

## 1. The question

On held-out games, does training the incumbent's eight policy-head tensors on 1,600-simulation visit
distributions predict held-out 1,600 distributions better than each of these?
* **(a)** The specified training procedure applied to 400-simulation targets for the same positions.
* **(b)** The incumbent, unchanged.
* **(c)** The incumbent with one fitted temperature.

The question is asked only after the targets pass an operational qualification gate (§6).

**What each comparison does and does not control.**
* (a) controls only for this procedure trained on these 400 targets. It does not control for every
  form of fine-tuning.
* (c) compares against this one fitted temperature. It does not compare against every calibration.

**The corpus is incumbent play.** It is 400-v-400 incumbent games. A student that plays differently
may meet a different distribution of positions. Stage B plays no strength game.

## 2. Frozen design at a glance

| Item | Frozen value |
|---|---|
| Corpus | 200 fresh games, incumbent at 400 vs itself at 400, otherwise the Stage A protocol (§3) |
| Split | By game: held out if and only if g % 5 == 4. That gives 40 held-out games (the sampling units) and 160 training games. |
| Positions | Up to **4** per game, drawn uniformly without replacement from non-terminal plies 1 … n−1. Selected for all 200 games in phase P, before any label (§4). |
| Targets | Normalized root visits from `search_with_root(state, add_noise=False)` over the **full legal move list**: 1,600 (student) and 400 (control), same position, same label seed (§5) |
| Qualification | Gate Q on the held-out labels, which include a second 1,600 label from an independent seed. Q runs before any training label (§6). |
| Trainable | Exactly `policy_head.{conv,bn,fc,out}.{weight,bias}`, 519 parameters. Eval mode throughout (§7). |
| Compute | 160 steps × batch 32: about **eight passes** over at most 640 training positions. Adam lr 1e-3, gradient clipping 1.0. Same index schedule for both arms (§7). |
| Learning criteria | L1–L3 on the 40 held-out games, all required (§8) |
| Seeds | One new block of 441 slots. Exposure is the union of used slots across phases (§9). |
| Price | About 6.0 h planned; about 11.6 h in the observed-rate scenario; five caps totalling 17.5 h (§10) |

**Fixed design choices.** These are not evidence of strength transfer:
* lr 1e-3;
* the fitted-temperature arm;
* 200 games with 40 held out;
* four positions per game;
* 160 steps;
* the 0.5 ratio in Q2.

**After a non-pass:** no automatic expansion. A larger pilot would need its own card.

## 3. Corpus and splits

* **Games.** 200 games of `calib020_0001` at 400 vs `calib020_0001` at 400.
* **Protocol.** Otherwise the Stage A protocol:
  * `PROTOCOL` from `external_learning_stage3`: batch 14, stall flush 48, opening temperature 1.0 for
    20 plies then 0.1, max 280 moves;
  * `search_with_root` without noise;
  * four independently seeded streams per game.
* **What a game contributes.** Only its replayed move list. No Stage B step reads its winner.
* **The split is fixed by slot before any game.**
  * Held-out: g % 5 == 4, giving 40 games.
  * Training: the other 160.
* **Fresh seeds.** No game is from A1 or Stage A. Every seed comes from a new block (§9).

## 4. Position selection (phase P) and the state hash

**Phase P.** It runs for all 200 games, after the corpus is committed and before any label. The
leakage rule (below) compares held-out positions with training positions, so the training positions
must exist before Q.

**For each game g, in slot order:**
1. Write a `select_start` event naming slot g's label seed to the progress log. This is written
   **before** the draw, because the draw uses L_g (§9).
2. Rebuild the game's states from its move list in our engine.
3. Draw `random.Random(L_g).sample(range(1, n), min(4, n − 1))`, where n is the number of moves.
   * Position p is the state before the move at ply p.
   * Ply 0, the empty board, is excluded.
   * The final state is excluded.
4. Write the sorted plies and each position's state digest.

Phase P reads no outcome and no label.

**State digest, frozen.**
* `TwixtState` has exactly seven constructor fields. The digest covers all of them:
  * `board_size`, `active_size`, `to_move`, `ply`, `max_plies_limit`;
  * pegs and bridges, serialized below.
* The derived adjacency cache is excluded.
* There is no swap or pie context in this engine.
* Serialization:
  ```python
  obj = {"board_size": s.board_size, "active_size": s.active_size, "to_move": s.to_move, "ply": s.ply,
         "max_plies_limit": s.max_plies_limit,
         "pegs": sorted([r, c, colour] for (r, c), colour in s.pegs.items()),
         "bridges": sorted([r1, c1, r2, c2] for (r1, c1), (r2, c2) in s.bridges)}
  digest = hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  ```
* Bridges are taken in the engine's canonical tuple order. A refactor that adds a constructor field
  must fail the build's field-set check rather than silently weaken the digest.

**Leakage rule.**
* A held-out position whose digest occurs among the training positions is excluded from every held-out
  measure. It stays in training.
* The count excluded is recorded.
* Duplicate positions within a split are kept, as independent draws.
* A held-out game that loses all its positions is an **empty game** (§6, §8).

## 5. Labels: the call, the randomness and the full legal support

**The call.**
* For position x, construct a fresh `MCTS(evaluator, cfg_from(EvalConfig(**PROTOCOL), sims), random.Random(seed ^ 0xA5A5A5))`
  and call `search_with_root(x, add_noise=False)`.
* `cfg_from` takes an `EvalConfig`, not the `PROTOCOL` dict. The config object is built first, as
  `external_learning_stage3.eval_config()` does.
* This is the evaluation harness's red-search construction. Its stream is one of the four masks the
  collision method already checks.

**Full legal support.**
* A label is defined over `x.legal_moves()`: every legal move in the engine's sorted row-major order,
  including moves with zero visits.
* The target is visits divided by `sims`.
* **Storage may be sparse**: the nonzero `(row, col, visits)` entries, with the position digest, the
  legal-list digest, the seed slot and `sims`.
* **Batching and evaluation must restore the full support.** Rebuild the full legal list from the
  state; check that its digest matches the stored one, and that the visits sum to `sims`; fill zeros.
* `make_padded_batch(max_moves_cap=576)` never truncates. The mask covers every legal move, so every
  softmax normalizes over every legal move.

**The search is random.** `MCTS` uses its random stream in two places:
* shuffling each expansion's prior order;
* breaking selection ties at random.

Two seeds can therefore give different targets.

**Common random numbers.**
* The 1,600 target and the 400 control target for a position use the same label seed L_g.
* With a deterministic evaluator, the 400 search is exactly the first 400 simulations of the 1,600
  search.

**Determinism check D.**
* On each game's first drawn position, repeat the 400 call with the same seed. The counts must be
  identical.
* A mismatch is a STOP with NO VERDICT.

## 6. Gate Q: operational qualification of the targets (before any training label)

**Measurements.**
* Every held-out position gets a second 1,600 label, T′, from the game's repeat seed R_h, using the
  same call.
* Per position:
  * D = TV(T1600, T400);
  * R = TV(T1600, T′).
  * TV is computed over the full legal support.
* Per non-empty held-out game: D̄_g and R̄_g, the means over its positions.

**D and R are not a signal/noise decomposition.**
* D compares two searches that **share a prefix**: the 400 search is the first 400 simulations of the
  1,600 one. So D measures how much the extra 1,200 simulations move one search's distribution.
* R compares two **independent** 1,600 searches.
* Q is therefore a deliberately coupled contrast: "does the budget change move the target more than an
  independent rerun does?"

**Q passes only if both conditions hold.**
* **Q1.** R̄_g < D̄_g in at least **26 of 40** held-out games.
  * This is the exact one-sided sign test at α = 0.05: P(X ≥ 26 | 40, ½) = 0.0403.
  * An empty game fails Q1.
  * A tie (R̄_g = D̄_g) fails Q1.
  * n stays 40.
* **Q2.** Σ R̄_g / Σ D̄_g ≤ **0.5**, summed over non-empty games.
  * If Σ D̄_g = 0 (no measurable budget change), Q2 fails, and the record says why.
  * If every game is empty, Q2 fails.

**If Q fails: QUALIFICATION NON-PASS.**
* The record names which condition failed (Q1, Q2 or both) and gives both statistics.
* Stage B stops before the training labels.
* A non-pass does not show that the 1,600 target is unstable or unlearnable. Perfectly stable targets
  that are identical at 400 and 1,600 would fail too, because D = R = 0.
* It means only that this pilot has nothing it was designed to measure.

**Descriptive only:**
* the median R and median D;
* **repeat agreement**: the share of held-out positions where the argmax sets of T1600 and T′
  intersect.

**Repeat agreement is not an upper bound** on any arm's agreement. A learner that predicts the dominant
move can exceed it.

## 7. Training: fixed compute, two arms

**Starting point.** A verified copy of `calib020_0001` (SHA-1 `209cf2d4…`), in eval mode throughout.
* #53 stage 2's `head_only` is reused: freeze everything, unfreeze the policy head, then freeze the
  head's BN running statistics again.
* The trainable set must be exactly the eight tensors, 519 parameters.
* The encoder, the value path and every BN running statistic stay frozen.

**Arms.** The two arms are identical except for the targets.
* Student: T1600.
* Control: T400 for the same positions.

**Loss.** #53 stage 2's `chosen_ce`, **unchanged**.
* It already computes the mean over the batch of −Σ target · masked log-softmax(logits).
* It is fed soft targets with full legal support (§5) in place of one-hot targets.
* There is no other term.

**Schedule.**
* 160 steps × batch 32 = 5,120 draws. Over at most 640 training positions (160 games × 4), that is
  about **eight passes**.
* Indices are drawn uniformly with replacement from `numpy.random.Generator(PCG64(T))`.
* The **same index sequence** is used for both arms.
* Adam with MLX defaults (betas 0.9/0.999, eps 1e-8, no bias correction), lr 1e-3, global-norm clipping
  1.0, no weight decay.
* Only the final step is used: no checkpoint selection, no early stopping.

**Integrity checks, as #53 stage 2.** Any of these is a STOP with NO VERDICT:
* a non-finite step;
* leaving eval mode;
* a gradient outside the head;
* a saved candidate that differs byte-wise from the incumbent file in any tensor other than the eight;
* a candidate in which none of the eight tensors changed.

**Temperature arm.** No gradient steps.
* A single inverse temperature β is applied to the incumbent's logits.
* β is fitted on the **training** positions' T1600 targets, using #53's convex bisection
  (`fit_beta`) generalized to soft targets. The derivative is Σ p·z − Σ T·z over the legal support.
* Logits are re-masked after scaling.
* A non-finite or unbracketed fit is a STOP.

**Cap.** 30 minutes for both arms, the fit, the saves and the reloads.

## 8. Held-out learning criteria (all three required)

**The KL measure.**
* For each held-out position after the leakage rule:
  KL(T1600 ‖ p) = Σ_{m: T_m > 0} T_m (log T_m − log p_m).
* p is the arm's softmax over the full legal support.
* The four arms are the student, the control, the temperature arm and the incumbent.
* Each arm gets a per-game mean over the game's positions. Every game is weighted equally, and the game
  is the unit.

| Criterion | A held-out game is a student win iff | Requirement |
|---|---|---|
| **L1** (against the specified 400-target procedure) | KL_student < KL_control | **≥ 26 of 40** games |
| **L2** (against the unchanged incumbent and the fitted temperature) | KL_student < **min(KL_incumbent, KL_temperature)** | **≥ 26 of 40** games |
| **L3** (top-1, interpretable) | — | The student's top-1 agreement with T1600 is **strictly above** both the control's and the incumbent's |

**Top-1 agreement (L3).**
* Per game: the share of the game's positions where the arm's top-1 move lies in T1600's argmax set.
* Overall: the mean of the per-game shares over all **40** games.
* It is also reported for the temperature arm, beside the repeat agreement from §6, which is not a
  ceiling.

**Ties and empty games, frozen.**
* In L1 and L2, equality counts as a student loss, and n stays 40.
* An **empty game** is a student loss in L1 and L2. It contributes **zero** top-1 agreement for **every**
  arm in L3.
* An arm's top-1 is the argmax of its masked logits. Ties go to the lowest encoded move id.
* The target's argmax set is every move with the maximum visit count.
* In L3, equal agreement is not "strictly above".

**Independence and power.**
* The sign tests assume the 40 held-out games are independent. This is an assumption, not a result.
* The 26/40 threshold gives about 80.7% power at a 70% per-game win probability **for each comparison
  separately**. That is not the power of the whole pipeline: Q, L1, L2 and L3 must all pass.

**Descriptive only:**
* every arm's mean KL;
* (KL_incumbent − KL_student) / KL_incumbent;
* β;
* the per-game tables.

**Verdicts.**
* **PASS:** Q, L1, L2 and L3 all hold.
* **NON-PASS:** Q holds and any of L1–L3 fails.
* **QUALIFICATION NON-PASS:** Q fails (§6).
* **NO VERDICT:** any of the following:
  * a cap;
  * a determinism mismatch;
  * an integrity STOP;
  * a survivor;
  * a schedule fault.

## 9. Seeds and exposure

**One block of 441 slots** from `seed_selection.candidate(reserved, 441)`, under the established
method:
* the pinned survey and proof scripts;
* a two-tree scan;
* survey → pre-proof → registration → post-proof;
* no probe block.

| Slots | Seed | Used by | Streams |
|---|---|---|---|
| 0–199 | G_g | corpus game g | `play_eval_game`'s four masks |
| 200–399 | L_g | phase P draw for game g; its 1,600, 400 and D label calls | `Random(L_g)` direct (draw); `Random(L_g ^ 0xA5A5A5)` (searches) |
| 400–439 | R_h | the T′ calls of held-out game g = 5h + 4 | `Random(R_h ^ 0xA5A5A5)` |
| 440 | T | the training index schedule | `PCG64(T)` direct |

**Exposure.**
* Exposure is the **union of used slots across phases**, not only the slots whose searches started.
* Each use is logged as a progress event **before** the seed is first used:
  * G_g: `game_start` (G);
  * L_g: `select_start` (P), or any label start for game g (L1, L2);
  * R_h: `repeat_start` (L1);
  * T: `train_start` (T).
* A slot is EXPOSED if any phase logged its use.

**Closing.**
* The whole block is RETIRED whatever happens.
* EXPOSED is the union above, derived from the committed outcome-blind progress logs of every phase
  that ran.

## 10. Pricing from measured search costs

**Why repriced.** The proposal's 2.46 ms per simulation came from a 400-only game, and is replaced.

**Rates.** Rates come from Stage A and A1 wall seconds and ply counts (outcome-blind progress):

| Source | ms per simulation |
|---|---|
| Stage A, all 160 games (31,028 s / 9.85 M simulations) | **3.15**, the planning rate |
| Stage A, per-game median (IQR) | 3.00 (2.83–3.29) |
| Stage A, slowest game | 4.81 |
| A1 game 0 (339.5 s, 55,600 simulations) | **6.11**, the slowest rate observed: the observed-rate scenario, **not a bound** |

**Limits of the rate data.**
* The data cannot separate the per-simulation cost of a 1,600-simulation search from that of a
  400-simulation search, because the two counts move together in every game.
* Nothing here bounds the true rate; a slower run is possible.
* The caps below are what limit wall time.

**Label work is fixed by design**, at 4 positions per game, whatever the game lengths. The maximum is
**1.936 M** simulations, down from 7.504 M at 16 positions.

| Phase | Simulations (max) | At 3.15 ms | At 6.11 ms (scenario) | Cap |
|---|---|---|---|---|
| G: 200 corpus games, 400 v 400 (≈ 61.4 plies, Stage A's mean; 400-v-400 length unmeasured) | ≈ 4.91 M | 4.3 h | 8.3 h | **12 h** |
| P: position selection (engine replay only, no search) | 0 | minutes | — | **5 min** |
| L1: held-out labels: 160 × (1,600 + 400) + 160 × 1,600 T′ + 40 × 400 D | 0.592 M | 0.52 h | 1.0 h | **85 min** |
| L2: training labels: 640 × (1,600 + 400) + 160 × 400 D | 1.344 M | 1.18 h | 2.28 h | **3.5 h** |
| T: training, β fit, held-out evaluation | ~0 (forward/backward) | minutes | — | **30 min** |
| **Total** | ≈ 6.85 M | **≈ 6.0 h** | **≈ 11.6 h** | **17.5 h** |

**Phase G is the least certain.** 400-v-400 game length is unmeasured. The accepted #53 figure of 61.9 s
per game would give 3.4 h.

**Caps.**
* Each phase runs under `timeout -k 10`, with termination confirmed.
* A cap is never extended.
* Reaching a cap gives NO VERDICT.

## 11. Sequence (each step separately approved)

1. **B0, the build.**
   * A corpus runner: the Stage A runner at 400/400 with the 200-game schedule.
   * A phase-P selector, with the field-set check on the state digest.
   * A label script that writes sparse labels and restores the full support.
   * A training and evaluation script reusing #53 stage 2's `head_only`, `train_head` and
     `chosen_ce` unchanged with soft targets, and `surface_check`.
   * The soft-target β fit.
   * Analysis and closing scripts, each with synthetic controls only.
2. **Seeds:** survey → pre-proof → registration and plan → post-proof.
3. **G:** corpus run (detached, capped), then commit its outputs.
4. **P:** selection for all 200 games; commit.
5. **L1:** held-out labels, T′ and D; commit. Then run Q once on the committed L1 outputs. If Q fails:
   QUALIFICATION NON-PASS, go to closing.
6. **L2:** training labels and D; commit.
7. **T:** both arms, the β fit, the surface checks, then held-out evaluation and the verdict, once.
8. **Closing:** the exposure derivation over every phase that ran, then the closing edit.

**Stop rules.**
* Any non-zero exit, failed control, red test or timeout stops for review.
* Nothing is retried, extended or replaced.
* A non-pass does not trigger a larger pilot.

## 12. Strength evaluation: priced separately, not part of Stage B

* **Student against incumbent.** The student at 400 vs the incumbent at 400 under the same protocol.
  * N = 200 pairs: 400 games, about 9.8 M simulations.
  * 8.6 h at 3.15 ms; 16.7 h in the 6.11 ms scenario.
  * The pass rule `mean − sqrt(ln(40)/(2N)) > 0.5` needs an observed mean above 0.596.
* **Attributing a gain to the deeper target** needs the trained control evaluated the same way: another
  8.6–16.7 h.
* **Not requested.** Each needs its own card and approval, and only after a Stage B PASS.

## 13. What Stage B does not establish

* **Strength.** No strength game is played.
* **Target optimality.** Stage A measured a playing margin. Q measures only a coupled budget-change
  contrast against independent reruns.
* **Generalization beyond incumbent 400-v-400 play.** The student's own play may shift the position
  distribution.
* **The source of a gain beyond the two stated comparisons.**
  * L1 rules out only the specified 400-target procedure.
  * L2 rules out only the unchanged incumbent and the one fitted temperature.
* **Anything about a non-pass beyond the criterion that failed.**
