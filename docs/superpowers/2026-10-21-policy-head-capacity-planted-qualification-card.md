# Policy-head capacity pilot — planted-target qualification of the fitting recipe (FROZEN CARD, rev 2, 2026-10-21)

**Status.** Revision 2, frozen for review. Not committed; nothing is built or run.

**It authorizes nothing.** Each of these needs its own explicit approval: committing this card; building the
qualification script and its focused CPU tests; opening the gate and running it once.

**Basis.** Ledger post-closure entry 18 (`c2dc3e32`): `scripts/GPU/alphazero/policy_head_capacity.py` builds arm R
(existing head + zero-initialized residual path over the move cell's 128 encoder features; 8,840 trainable
parameters) and arm C (the existing head's 519 parameters only), frozen elsewhere, eval mode; 6/6 CPU checks pass.

**Review changes in rev 1 (2026-10-21).** T0 (null target) dropped: both arms start at its optimum with zero
gradients, so it cannot fail. TC separation rule added (§6, S3). Top-2 principal-component variance fraction
reported, descriptive only (§7). Dose frozen at 400 Adam steps (§5). Each fit's starting KL and the chance baseline
reported (§7). Feature cache kept outside the repository; only its digest is recorded (§4).

**Review changes in rev 2 (2026-10-21).** Each planted term's 1.0-nat scale is measured within position: the
position's mean added logit is subtracted before the SD over training legal moves is taken, for TC and TR alike
(§4). Gradient clipping is stated explicitly: global norm clipped at 1.0, as external-learning stage 2 (§5). The
temperature is fitted on the training positions only and scored on the evaluation positions (§5, §6). Teacher seeds
listed explicitly (§2). The replay VOID check and the `~/twixt-cache/` location were accepted.

## 1. The question, and what an answer means

Can the frozen fitting recipe (§5) recover a capacity difference that was **planted** on purpose, on the real
encoder's features of real positions, while not favouring R where no capacity difference exists?

**All results are operational qualification only.** A QUALIFIED verdict says the recipe can expose a planted gap
of the planted size; it says nothing about real targets, about whether the incumbent head's capacity binds, or
about strength. A NOT QUALIFIED verdict says this recipe should not be used to measure capacity; it proves nothing
about capacity either.

## 2. Frozen design at a glance

| item | frozen value |
|---|---|
| network | `calib020_0001`, `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`, SHA-1 `209cf2d4fd24a48553d259dd71b4954867b9473e` |
| device | CPU only (`mx.set_default_device(mx.cpu)`, asserted); float32 model computation |
| positions | 1,600 training + 400 evaluation, 24×24, from two closed screens' committed game records (§3) |
| targets | TC (in both arms' class), TR (in R's class only) (§4) |
| fits | C on TC, R on TC, C on TR, R on TR; plus a fitted temperature on TR (§5) |
| dose | 400 Adam steps, batch 32, lr 1e-3, frozen draw order — 12,800 samples = 8 passes (matches Stage B) |
| verdict | QUALIFIED only if all five rules F1, F2, S1, S2, S3 hold (§6) |
| cap | 15 minutes wall for the whole run (§8) |
| seeds | small integers outside the seed registry (no games are played), all distinct: **11** position selection; **12** TC teacher (perturbation noise ε); **13** TR teacher (weights of g); **14** draw order |

## 3. Positions: source, split, digest, leakage

**Source.** The committed game records of two closed screens:
`docs/superpowers/evidence/inference-mode-match/games/` and `docs/superpowers/evidence/mirror-averaging-match/games/`
(32 games each, all `board_size` 24, all ended `win`; 3,293 moves in total). These are records of finished screens,
not reserved held-out material. Excluded and untouched: Stage B's held-out games (spent), the root-halving match and
the self-play diagnostic (left for the real pilot).

**Replay.** Each game is replayed from `TwixtState(active_size=24, to_move="red", max_plies_limit=280)` with its
recorded moves in ply order. The candidate positions are the states **before** each recorded move (plies 0 to n−1);
the terminal state after the last move is not a position. At every ply the replayed state's legal-move count must
equal the record's `n_legal` and the recorded move must be legal (integrity, §8).

**Split by game.** Evaluation games: `game_idx % 5 == 4` (6 per screen, 12 games, 611 moves); training games: the
other 52 (2,682 moves). No game contributes to both splits.

**Digest.** Stage B's frozen state digest (card `docs/superpowers/2026-10-12-search-distillation-stage-b-card.md`
§4), unchanged: SHA-256 of the canonical JSON of the seven `TwixtState` constructor fields (`board_size`,
`active_size`, `to_move`, `ply`, `max_plies_limit`, sorted pegs, sorted bridges), with the same field-set check.

**Pools and leakage.** Within each split, positions are deduplicated by digest (first occurrence in (screen,
game_idx, ply) order kept). Following Stage B's direction, an evaluation digest that also occurs in the training pool
is removed from the evaluation pool and stays in training; the count removed is recorded.

**Selection.** `numpy.random.default_rng(11)`: 1,600 positions drawn without replacement from the training pool and
400 from the evaluation pool, each pool first listed in (screen, game_idx, ply) order. Fewer than 1,600 or 400 unique
positions available is a VOID (§8).

**Recording.** The evidence lists every selected position by digest, screen, game_idx, ply, to-move and legal count,
so these positions can be excluded **by identity** from any later held-out set.

## 4. Features and planted targets

**Loading.** Verify the checkpoint's SHA-1 against §2, load it with `probe_eval.load_network_for_scoring`, call
`network.eval()`, and require 0 of the network's modules in training mode and the loader's input-channel count to
equal the tensor channels of `TwixtState.to_tensor()`. Any mismatch is a VOID.

**Features.** For each selected position, `features = network.encoder(board)` (24×24×128, float32) with
`board = state.to_tensor().transpose(1, 2, 0)`. Features are computed once and cached **outside the repository** at
`~/twixt-cache/policy-head-capacity-planted/features-<run-id>.npz` (training and evaluation arrays, with the legal
rows, columns and masks padded to 528). The evidence records only the cache file's SHA-256; the file is re-hashed at
the end of the run and must match (§8). The encoder and value head are never run again after this.

**Incumbent logits I.** The existing `PolicyHead.forward_padded` on the cached features, legal moves only.

**Scale (frozen, independent of real targets), measured within position.** Each teacher adds a term a to I. For
each training position, the position's mean added logit over its legal moves is subtracted (a constant per position
does not change the softmax); the standard deviation of these centred values, pooled over **all training legal
moves**, must be **1.0 nat**. The same measure is used for TC and TR. The scale is fixed by the teacher's own output
on the training features only; no evaluation position, real label, search output or real-target entropy enters.

**TC — inside both arms' class.** Teacher seed **12**. The incumbent head with all eight weight and bias arrays (519
parameters) perturbed: `w_k + α · σ_k · ε_k`, with ε_k ~ N(0, 1) from `default_rng(12)` in a fixed array order, σ_k
the standard deviation of the incumbent array k, and BatchNorm running statistics unchanged. The added term is
a = TC logits − I. The scalar α is found by bisection (60 iterations on [0, 10]) so that the within-position scale of
a is 1.0 (±0.001). Both C and R can represent TC exactly (R with its residual path at zero).

**TR — inside R's class only.** Teacher seed **13**. `I + s · g(features at the move)`, where g is a fresh
128 → 64 → ReLU → 1 network with He-normal weights and zero biases from `default_rng(13)`; the added term is
a = s · g, and s = 1.0 / (the within-position scale of g), which is exact because a is linear in s. R represents
TR exactly (existing head unchanged, residual path = s · g); C in general cannot.

**Targets.** p_target = softmax of the teacher logits over each position's legal moves, computed in float64 and
stored as float32 rows that sum to 1 over legal moves and are 0 on padding.

## 5. Fitting recipe (frozen; identical for every fit)

- **Arms.** A fresh copy of the loaded network per fit; `policy_head_capacity.install(network, "C" | "R")`.
  Before the first step each arm's evaluation logits must equal I bitwise (entry 18's tested property); otherwise
  VOID.
- **Loss.** Soft-target cross-entropy as in `tests/test_policy_head_capacity.py`:
  `−Σ p_target · log_softmax(logits)` over the batch's padded moves, divided by the batch size; computed from the
  cached features through `network.policy_head.forward_padded` only.
- **Optimizer and clipping, as external-learning stage 2** (`scripts/GPU/alphazero/external_learning_stage2.py`:
  `LEARNING_RATE, CLIP_NORM, WEIGHT_DECAY = 1e-3, 1.0, 0.0`; `ADAM` betas 0.9 / 0.999, eps 1e-8,
  `bias_correction=False`). `mlx.optimizers.Adam(learning_rate=1e-3, betas=[0.9, 0.999], eps=1e-8,
  bias_correction=False)`; constant rate, weight decay 0, no early stopping. Every step's gradient is clipped at
  **global norm 1.0** with `trainer.clip_grad_norm(grads["policy_head"], max_norm=1.0)`, as stage 2 calls it: the L2
  norm over the arm's whole policy-head gradient tree, scaled by min(1, 1.0 / (norm + 1e-8)). `trainer` imports
  `self_play` at module level, so it is imported inside the fit function (as stage 2 does), and the script's tests
  and run happen only in a clean linked worktree (§9).
- **Dose.** 400 steps at batch 32 on the 1,600 training positions = 8 passes.
- **Draw order.** `default_rng(14)`: eight permutations of the 1,600 training indices, each cut into 50 batches of
  32, in order → exactly 400 batches. The same order is used for all four fits.
- **Evaluation.** Once, after step 400. No intermediate checkpoint is evaluated or kept.
- **Temperature fit (TR only).** No gradient steps: logits I/τ with τ = e^u, u found by golden-section search on
  [−3, 3] (60 iterations) minimizing cross-entropy against TR on the **1,600 training positions only**. The fitted τ
  is then scored, like every fit, on the **400 evaluation positions only** (§6); no evaluation position enters the
  fit.

## 6. Verdict rules (all five required for QUALIFIED)

KL = mean over the 400 evaluation positions of KL(p_target ‖ softmax(model logits)) over legal moves, in nats,
computed in float64. KL0(target) = the same quantity for the incumbent I (both arms' starting point). δ = 0.05 nats.
Every fit, the temperature included, is trained or fitted on the training positions only and scored here on the
evaluation positions only.

| rule | condition |
|---|---|
| **F1** fit TC | KL(C, TC) ≤ 0.25 · KL0(TC) **and** KL(R, TC) ≤ 0.25 · KL0(TC) |
| **F2** fit TR | KL(R, TR) ≤ 0.25 · KL0(TR) |
| **S1** separation on TR | KL(C, TR) − KL(R, TR) ≥ δ |
| **S2** beats temperature on TR | KL(τ, TR) − KL(R, TR) ≥ δ |
| **S3** no optimisation advantage on TC | KL(C, TC) − KL(R, TC) < δ |

S3 guards against reading R's TR advantage as capacity when it might come from optimisation: on TC, where both
arms can represent the target exactly, R must not beat C by δ or more.

**QUALIFIED** if all five hold. **NOT QUALIFIED** if the rules are computed and any one fails: no labels are
generated and the recipe is not retuned; any new recipe needs a new card, fresh teacher seeds and its own approval.
All outcomes are operational qualification only (§1).

## 7. Reported, descriptive only (no rule reads these)

- Each fit's starting KL (= KL0 of its target) and final KL; final training cross-entropy.
- The chance baseline: mean ln(number of legal moves) over the 400 evaluation positions.
- Each target's mean entropy over the evaluation positions.
- The fraction of legal-cell feature variance in the top 2 principal components: over every legal cell of the
  1,600 training positions, the 128-dimensional features centred, eigenvalues of their covariance (float64),
  (λ₁ + λ₂) / Σλ.
- The bisection α and the scale s; the fitted τ; the leakage count; pool sizes; wall time of each phase.
- Per fit, the pre-clip gradient norm at every step and the fraction of steps on which clipping was active.

## 8. Integrity: VOID rules

Any of these makes the run **VOID** (not NOT QUALIFIED). A record is still written; nothing is rerun without
approval:

- a non-finite loss or gradient norm at any step, or non-finite KL;
- after a fit, any parameter outside the arm's surface changed, or any BatchNorm running statistic changed;
- the network or its policy head leaves eval mode;
- checkpoint SHA-1 mismatch, or the feature cache's re-hash does not match its recorded digest;
- a non-24×24 position; a replayed `n_legal` or move legality mismatch; fewer unique positions than §3 requires;
  any count differing from the frozen numbers;
- the 15-minute wall cap reached (enforced by the wrapper's alarm and checked by the script between phases);
- any exception.

**Record boundary.** The verdict is computed inside the same handler that writes `record.json`, so every exit,
including VOID, leaves a record.

## 9. Execution (each step separately approved)

1. **Build.** `scripts/GPU/alphazero/policy_head_capacity_planted.py`: a closed single-declaration gate, refusing by
   default; evidence create-only under `docs/superpowers/evidence/policy-head-capacity-planted/`. Focused CPU tests
   with stub features and no checkpoint, run in a clean linked worktree (the fit imports `trainer`, hence
   `self_play`): each teacher is exactly representable by its arm (copying the planted weights gives KL 0), the
   within-position scale procedure hits 1.0 nat, clipping matches `clip_grad_norm`, the digest equals Stage B's,
   split and leakage, the five rules on hand-computed numbers, and each VOID path fires.
2. **Run once.** In a clean linked worktree: a one-line commit opens the gate; the wrapper runs the script under
   `.venv/bin/python` with a 900-second alarm; the gate is restored and committed with the evidence; main is
   fast-forwarded and the worktree removed.
3. **Report and ledger.** The verdict and §7's numbers, then a ledger entry before the line is reported done.

**Time.** Unmeasured on this machine; old-machine timings are not used. The run records each phase's wall time,
which become this machine's first measurements for this workload.
