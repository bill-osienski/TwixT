# Combination policy pilot — card (FROZEN 2026-10-28)

**Status.** Frozen on the user's second review (2026-10-28), which approved this card and step 1's build only, with a
narrow #66 exception for reusing the residual head in this pilot (no capacity claim). Two clarifications from that
review are applied in place (§4 the probability discrepancy; §5 step 2 the 144 games inside 15 minutes). §10 records
how step 1 was built; where §3 and §10 differ, §10 describes the code. Seed procedures, checkpoint loading, GPU work,
corpus generation, real-data training and games remain unauthorized; each later step needs its own approval.

The text below is the reviewed proposal (amendment 1) otherwise unchanged. It was written before the build, at `main`
`752a723d`.

**Amendment 1 (after the first review):**
- The objective is now frozen as the mirror-averaged policy, used for measurement, training and the held-out check
  alike (§4).
- Each game's seed is now assigned by slot, independent of the worker that plays it. The proven agent's four
  streams are preserved, and every game block comes from a fresh registered block (§3, §4).
- The 519-parameter claim is removed (§2).
- The CPU rationale is corrected (§4).
- The initialization and sampling seeds are fixed (§4), and each stage gets a hard ceiling before its approval (§5).
- The target's main uncertainty is stated as a hypothesis (§4, §8).
- The direction is supported in principle. The build and every execution remain unapproved.

## 1. Question

Can one round of learning from the proven agent's own search make that agent stronger?

- **The proven agent** (ledger entry 39, PASS accepted `7bacc265`):
  - `MirrorAveragingEvaluator` around eval-mode `calib020_0001`, at 200 simulations;
  - ordinary search and readout through ply 19;
  - unchanged `RootSequentialHalvingMCTS` from ply 20;
  - 24×24, existing rules, TF32=0.
- **The pilot's candidate:** the same agent and search, over a network whose **policy pathway alone** has learned from
  that agent's halving decisions.

The proven agent stays frozen as the comparison reference.

## 2. Why this is not a closed line (#48, #49, #51, #69, #87, #89)

| Closed line | What it trained | What this pilot does instead |
|---|---|---|
| cont5 / warm5 (#48, #49): ordinary continuation, −136 / −47 Elo | Whole network; outcome value targets; 400-sim PUCT visit counts with root noise | Nothing in the encoder or value head moves; no value target; halving-derived policy target |
| fp5 / fp6 (#51): frozen-parent opponent | Whole network against the parent | No opponent mechanism |
| D2 Stage 3 / attempt 2 (#69, #87): search distillation | Encoder + policy head to 1,600-sim visit counts, value anchored (weight 1.0); value drift 0.1207 / 0.3433 | Encoder frozen, so the value cannot drift; no anchor |
| Dual encoder (#89) | Rejected on paper | Not used |

**The rationale for preserving the value function is distinct:** preservation is **by construction**, not by penalty.

- D2 froze the value head and anchored the value output. The value still drifted (mean 0.1207 against the 0.05
  limit), because the shared encoder moved. An anchor on the training rows did not hold on held-out rows (training
  drift 0.0636).
- This pilot freezes the encoder, the value head and every BatchNorm running statistic. Only a policy-only branch is
  trained. The value output is therefore **bit-identical** to the parent's, and a CPU check (§6) verifies that.
- This matters for this agent in particular: from ply 20, halving ranks its candidates by
  `log prior + (50 + max N)·(q + 1)/2`. The q terms come from the parent's value, which the PASS has shown to work
  inside this search.
- **On the ~47 Elo shortfall (#49):** warm5's regression was never attributed between value, trunk and policy. This
  design removes the value and trunk channels entirely. Whatever caused warm5's loss through those channels cannot
  act here. The policy change that remains is measured directly, before any game (§5, step 4).

**What the policy branch can improve:**
- The halving search considers only the 16 highest-prior moves. A good move outside them is never searched (the
  card's known limit).
- The log-prior term also enters every ranking.
- Better priors therefore change which moves get searched, not just tie-breaks.

**What it cannot improve:** the value. This is a deliberate limit of the pilot.

**Exception needed:** the trainable branch reuses `policy_head_capacity.ResidualPolicyHead` (arm R: the existing 519
head parameters plus 8,321 zero-initialized residual parameters).
- #66 shelved the capacity proposal. This pilot makes no capacity claim and compares no arms.
- Reusing the component needs a narrow, explicit exception to #66: for this pilot only, with no claim about capacity.
- The ledger's small head did learn the planted targets substantially. Whether the existing head has enough capacity
  remains unresolved, and this pilot does not test it.

## 3. Training-path audit (current code)

**What self-play does now:**
- `self_play.play_game` (`:704`) builds `MCTS(evaluator, …)`.
- Each ply it calls `search_from_root(root, add_noise=True, ply=…)` (`:940`): batched leaves, virtual visits, a reused
  subtree, and Dirichlet noise at the root.
- It plays `select_move` (temperature over the counts, `:1186`), then `advance_root` (tree reuse, `:1269`).
- The position is recorded with raw visit counts as the policy target (`:1119`). Mirror augmentation is
  probabilistic, `TWIXT_MIRROR_PROB`, default 0.5.

**What the proven agent does:**
- `RootSequentialHalvingMCTS.search_with_root` takes a fresh root, no noise (it raises `ValueError` otherwise),
  synchronous one-leaf simulations, and a `last_choice`.
- The caller plays `last_choice` from ply 20. Before that it uses `eval_readout.select` (opening temperature 1.0
  through ply 19).
- Swapping the class therefore changes nothing: `search_from_root` is not overridden and would run ordinary batched
  PUCT.

**A latent fault found in the audit:** `MirrorAveragingEvaluator.__init__` reads `base.network`. Self-play's
`RemoteEvaluator` (`remote_evaluator.py:21`) has no such attribute, so wrapping it would raise `AttributeError`.

- Board layout is not a problem: MCTS transposes to (H, W, C) before `infer` (`mcts.py:992`) on both paths, which is
  what `_mirror_position_lr` expects.

**Minimal opt-in changes.** The default path stays byte-identical, and the frozen components are not edited (#59, #62).

1. **`self_play.play_game(…, combination_agent=False)`**, a keyword appended last. When True:
   - wrap the evaluator in `MirrorAveragingEvaluator`;
   - build `RootSequentialHalvingMCTS` with `eval_runner.cfg_from(PROTOCOL, 200)`, the match's exact config;
   - each ply, run `search_with_root(state, add_noise=False)` on a fresh root, with no `advance_root` reuse;
   - play `last_choice` if it is set, else `eval_readout.select` with the protocol readout, on the side's readout RNG
     stream (§3.4);
   - record only ply ≥ 20 rows, with the §4 target.
   - **Refused:** `add_noise`, active size ≠ 24, max moves ≠ 280, resign, adjudication, td1 forcing, the closeout
     tie-break, frozen opponent, training capture, non-zero `TWIXT_MIRROR_PROB`, and a passed `rng` (§3.4). No row
     is mirror-augmented: the frozen objective is mirror-invariant (§4).
2. **`self_play_worker`:** in this mode, set `evaluator.network = None` on the worker's `RemoteEvaluator` instance
   before wrapping. That is an instance attribute; neither class is edited.
3. **`trainer.run_parallel_selfplay`:** pass the flag through to the workers. One `InferenceServer` in the main
   process owns Metal, and the workers stay CPU-only (#50).
4. **Seed assignment by slot** (combination mode only; the default path is unchanged).
   - **The current defect:** each worker receives its own seed from `master_rng.randint` (`trainer.py:2102`), and
     derives a game's seed as `worker_seed ^ (gid · 0x9E3779B1)` (`self_play_worker.py:210`). Workers claim game ids
     dynamically, so the worker that happens to claim a game decides its seed. The comment there ("reproducibility
     regardless of scheduling") is false. Default self-play keeps this behaviour; it is noted, not changed.
   - **The fix:** in this mode the entry passes a registered `slot_base` instead of worker seeds. Game slot `g` plays
     seed `s = slot_base + g`, whichever worker claims it.
   - **The four streams:** `play_game` derives them from `s` exactly as `mirror_halving_combination.play_game` does
     (`:92`–`:95`). Red's search uses `Random(s ^ 0xA5A5A5)` and black's `Random(s ^ 0x5A5A5A)`. Red's readout uses
     `Random(s ^ 0xC3C3C3)` and black's `Random(s ^ 0x3C3C3C)`. There are two search objects, one per colour, as in
     the match. The worker's `game_rng` is not passed in this mode, and `play_game` refuses one.
   - **Logging for closing:** before playing a claimed slot, the worker sends a start notice on the existing stats
     queue, and the main process logs `game_start` with the slot. At the end the main process logs the final claim
     counter.
   - Every seed is checked as ACCOUNTED, available and schedulable before launch, with the same `seed_ok` as
     `mirror_halving_match.py:304`.
5. **A small `halving_policy_target.py`:** the §4 target, computed from the returned root using the halving module's
   own `score()` and `last_rounds[0]`. It is imported, never modified.
6. **One gated pilot entry** (`generate`, `train`, `check` subcommands; new gates; create-only evidence; wrappers
   copied from the strength run's, keeping the idle gate and the BOINC check).
   - The `train()` loop and `train.py` are **not** used and not changed. There are no iterations, curriculum,
     `--resume` or replay buffer across rounds.

**Compatibility:**
- **Batching:** `search_with_root` ignores the worker-side leaf batch. Each worker has one leaf in flight, and the
  averaging issues two sequential one-row requests per leaf.
  - The arbiter still batches across the 18 workers, capped at `max_batch_rows = mcts_eval_batch_size = 14`.
  - So `--n-workers 18` applies directly, and `14` survives only as the arbiter's row cap per GPU call. Both are the
    entry-20 measured starting configuration, for a different search; neither is "optimal" here.
  - Throughput for this request pattern is **unmeasured** (§7). No sweep is proposed (#64).
- **Tree reuse:** none. Halving's schedule assumes a fresh root, and the proven agent never reused trees.
- **Exploration:** no root noise, because halving forbids it and the proven agent has none. Diversity comes from the
  opening temperature (1.0 through ply 19) and the per-slot seeds. Duplicate positions are counted and reported.
- **MCTS evaluation batch versus optimizer minibatch:** the former is rows per GPU inference call (14). The latter is
  rows per gradient step, 64 here as in every training line. They are unrelated settings.

## 4. The fixed recipe

**Starting state:**
- Weights-only load of `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`; the SHA-1 must equal
  `209cf2d4fd24a48553d259dd71b4954867b9473e` or the entry refuses.
- Not `--resume`: no iteration, curriculum, freeze state or optimizer state is read.
- `policy_head_capacity.install(network, "R", init_seed=20261028)`. `extra_out` starts at zero, so at step 0 the
  candidate **is** the proven agent, bitwise. `extra_fc`'s starting values come from that seed alone, never from
  MLX's global random state.
- Network in eval mode throughout. Neither D2 student and no random initialization is used.

**Data (fresh only):**
- G = **1,000** games of the proven agent playing itself, on a fresh registered block of 1,000 slots (§5, step 3).
- Never used: the screen, strength, void-run or throughput-check games; D2's games or held-out sets; Stage A/B; T1j;
  twixtbot.
- Rows: ply ≥ 20 only, about 37 per game (strength2 mean 56.8 plies), so about 37,000 rows.
- Games in slots ≡ 9 (mod 10) are held out (about 3,700 rows). The split is by slot and fixed before generation.
- Each row stores the generation-time mirror-averaged prior `p` (what the agent used) and the target π. The policy
  width cap is 576 (`ffe6b342`).
- **No mirror augmentation.** Under the frozen objective below the loss is mirror-invariant: a mirrored row (fix
  `0c417882`) with its mirrored target gives the same loss as the original. It would duplicate rows without adding
  information, so it is omitted. That leaves about 33,300 distinct training rows.

**Policy target (deliberate, not visit counts):**
- Halving's visit counts are a schedule: at 200 simulations each of the 16 candidates gets 3, 6, 12 or 28 more visits
  per round, by survival. They express the schedule, not a preference.
- The target therefore uses halving's own ranking quantity, applied within the candidate set. Let:
  - C be the 16 first-round candidates;
  - p the mirror-averaged prior (no noise);
  - q(a) the candidate's mean value from the root side (every candidate has at least 3 visits);
  - N_max the largest child visit count.
- Then:
  - **for a ∈ C:** π(a) = P_C · softmax_C(log p(a) + (50 + N_max)·(q(a) + 1)/2), with P_C = Σ_{b∈C} p(b);
  - **for a ∉ C:** π(a) = p(a).
- So moves the search never examined keep exactly their prior mass. Within the candidates, mass shifts as the search
  judged. This follows Gumbel AlphaZero's completed-Q improvement target, restricted to the searched set, with this
  repository's fixed (q + 1)/2 mapping. That mapping is an adaptation with no published guarantee.
- **The hypothesis, stated plainly:** this target improves the agent's prior. It can put heavy mass on a candidate
  eliminated after only 3 visits, whose q is noisy. The combination's PASS validates the search, **not this target**.
  The pilot tests the target; nothing earlier supports it.

**The objective (frozen; one definition everywhere):**
- The prediction is the **mirror-averaged policy**, the prior the agent actually uses:
  p̄_θ(s) = ½ [p_θ(s) + M⁻¹ p_θ(M s)]. Here M is the left-right mirror, and each p_θ is the network's masked
  probability distribution. This is the same probability mean as `MirrorAveragingEvaluator` (`mirror_averaging.py:30`).
- The raw network policy is never the prediction. A raw policy could reduce its loss by learning symmetry even where
  the target equals the averaged prior. The averaged prediction cannot.
- The same p̄_θ is used in three places:
  - the pre-fit measurement KL(π ‖ p̄_θ₀);
  - the training loss, CE(π, p̄_θ);
  - the held-out check, CE(π, p̄_θ) after training against p̄_θ₀.
- p̄_θ₀ is recomputed on the CPU from the parent. Its largest absolute difference from the stored generation-time
  `p`, over legal moves, is reported as the **probability discrepancy**. It is descriptive only: a maximum probability
  error does not itself bound the error in the KL.

**Loss, surface and optimizer:**
- Loss: CE(π, p̄_θ) over legal moves, plus 1e-4 × L2 on the trainable arrays. No value loss: nothing on the value
  path trains.
- Surface: arm R's 8,840 parameters. Encoder, value head and all BatchNorm running statistics frozen.
- Optimizer: a **fresh** Adam (zero moments), lr 1e-3, batch 64, **3,000 updates**. That is about 5.8 passes over
  the about 33,300 distinct rows; it is the same amount of information as the draft's 2.9 passes over mirror-doubled
  rows.
  - Minibatch order comes from `numpy.random.Generator(PCG64(20261029))`, a fresh permutation per pass.
  - These two training seeds (20261028 and 20261029) seed no game and are not registry seeds.
  - No schedule, no early stopping and no checkpoint selection: only the final state is evaluated.
- **Device: CPU** for the whole training step, because GPU training remains unqualified (#81, #82).
  - This recipe is not claimed to reproduce the #75 defect under TF32=0; that is untested.
  - Batch 64 does exceed 32,768 rows in the head's matmuls (64 × 576) and in the encoder's convolutions. That is the
    shape class #75 measured, so the GPU path would first need its own qualification, which this pilot does not seek.
  - With the encoder frozen, the backward pass touches only the small head. The averaged prediction needs two CPU
    encoder forwards per row, and that is the main cost (§7).

## 5. Steps, checks and stopping rules

Each step needs its own approval, as before.

**Seeds, by the existing procedure.** Every game block is fresh and goes through survey → pre-proof → registration →
post-proof before its run, and K1 → K2 closing after it:
- the throughput check: 144 slots;
- generation: 1,000 slots;
- the screen: 64 slots;
- the strength comparison: 1,018 slots.

The closing derivation is a copy of `mirror-halving-strength2-seed-closing/01_derive_usage.py`, with one diff: with
parallel workers, `game_start` events arrive out of order, so the started slots must form the prefix {0..k−1} **as a
set**. The final claim counter must equal k. If the record cannot establish k, the whole block is exposed. Every
block is retired whole, whatever happened.

**Hard ceilings.** Each stage's wall-clock ceiling is stated in its own approval request before it runs. Each
wrapper enforces it (TERM, then KILL 10 s later). No ceiling is ever extended. A stage that hits its ceiling is VOID
and closes the pilot.

1. **Build** (CPU tests only, §6; all new gates closed). This includes a CPU timing of 20 training updates at the real
   architecture on synthetic inputs, with random weights and no checkpoint. That timing prices step 4's ceiling.
2. **Throughput check** (GPU, 144 slots on its own registered block; ceiling 15 minutes; its games are never used).
   - **All 144 games must finish inside the 15-minute ceiling.** A run that has not finished them is TERMinated at
     the ceiling, is VOID, and closes the pilot, as every stage that hits its ceiling does. Changing that needs a
     separate amendment.
   - Measures games/h and the arbiter's mean rows per call at 18 workers. Step 3's ceiling is then set at 1.5× the
     projection, and never above 6 h.
   - STOP if a worker or the arbiter fails, the TF32 or eval-mode checks fail, or two slots' games depend on the
     worker that played them.
3. **Generation** (G = 1,000 on its own registered block; BOINC kept quit, idle gate, BOINC check per game, TERM at
   the ceiling).
   - An incomplete run is VOID and is not used.
4. **CPU training, then the check** (ceiling: 1.5× the build's timing projection, stated before approval):
   - **Pre-fit falsification:** if the mean KL(π ‖ p̄_θ₀) over training rows is below **0.01 nats**, STOP: there is
     nothing to learn. This is meaningful only because the prediction is the averaged policy (§4).
   - **Value preserved:** on the held-out rows the candidate's value outputs must equal the parent's bitwise, and the
     encoder and value-head digests must be unchanged. Otherwise STOP.
   - **Policy learned:** the held-out CE(π, p̄_θ) after 3,000 updates must be below CE(π, p̄_θ₀). Otherwise STOP,
     with no games.
   - STOP on any non-finite loss or gradient.
5. **Screen**: 32 colour-reversed pairs.
   - The trained candidate against the proven agent, with identical search configuration on both sides (both
     averaged, halving, 200 simulations).
   - On its own registered block; ceiling priced from the throughput check.
   - STOP below a mean pair score of 0.40. Otherwise CONTINUE. No strength claim either way.
6. **Strength comparison**: 509 pairs on a fresh registered block; ceiling priced from the screen.
   - PASS iff the mean pair score − √(ln 40/(2·509)) > 0.5, with guards: evaluations ≤ 1.05 and search seconds ≤ 1.10
     (expected ≈ 1.0: same architecture plus a residual of about 0.4% of the forward work).
   - No interim look, extension, top-up or retry. Any STOP, VOID or timeout closes the pilot.

**Prediction to record if approved:** directional only. The candidate's held-out cross-entropy falls (step 4), and it
scores above 0.5 against the proven agent (steps 5–6). No magnitude is predicted; confidence is low (§8).

## 6. Focused CPU checks (step 1)

These use fake evaluators and temporary outputs only, with no checkpoint and no MLX GPU.

- **Default path unchanged:** `play_game` without the flag produces a GameRecord identical to the current code's on a
  fixed fake evaluator and seed.
- **The agent is really the proven agent:** with a deterministic fake evaluator and identical RNG streams, a
  combination-mode self-play game plays exactly the move sequence of `mirror_halving_combination.play_game` (the
  match loop). Every search is `search_with_root` on a fresh root without noise. Every leaf makes two underlying
  calls, the second mirrored. The played move is `last_choice` from ply 20 and the readout before.
- **Seeds independent of scheduling:** with a fake in-process server and fake evaluator, the same slots run with 1,
  2 and 3 workers, and with an injected per-worker delay, produce identical games per slot. Each slot's game equals
  `mirror_halving_combination.play_game` at seed `slot_base + g`. The four streams are the match's XOR constants. A
  `game_rng` passed in this mode is refused. Every slot logs `game_start` before its game, and the claim counter is
  logged at the end. A seed that is not ACCOUNTED refuses the launch.
- **The objective:** on the same input, the training prediction p̄_θ equals `MirrorAveragingEvaluator`'s prior over
  the same network on the CPU (to float tolerance). A row and its mirrored row with the mirrored target give the
  same loss. The pre-fit KL, the training loss and the held-out check call one shared function.
- **The remote path:** `MirrorAveragingEvaluator` over a `RemoteEvaluator`, with a fake in-process server, returns the
  same arrays as over a local fake. Without the attribute shim it raises (pinning the audit's finding).
- **Refusals:** each refused option (§3.1) raises, and halving with `add_noise` still raises.
- **The target:** it sums to 1; non-candidate mass equals the prior; within C it is proportional to
  p·exp(σ) from `root_sequential_halving.score`; it is invariant to the mirror (the mirrored row's target is the
  mirrored target); no ply < 20 row is emitted.
- **Training:**
  - the weights-only start refuses a SHA-1 mismatch;
  - Adam starts with no state; no `.json` training state is read;
  - the R install with `init_seed=20261028` starts bit-equal to the parent, and two installs are identical;
  - the minibatch order from seed 20261029 is reproducible;
  - after updates, the value outputs and the encoder and value-head digests are bitwise unchanged;
  - the device is CPU;
  - each STOP rule fires on a constructed case.
- **Environment:** the entry refuses unless `MLX_ENABLE_TF32` is exactly `"0"` and no `mlx` is loaded at start. A
  spawned worker sees `"0"`. Nothing is persisted.
- **Gates:** the gate inventory is updated, and fake launches cover refusal while busy, refusal on detected BOINC,
  and the exact command.

## 7. Estimated cost (unmeasured; step 2 measures generation)

| Step | Estimate | Basis |
|---|---|---|
| Throughput check | ≤ 15 min | ceiling |
| Generation, 1,000 games | **~1.5–3 h** (roughly 350–700 games/h) | Strength2 single-process: about 32 s per combination game, about 22,700 underlying evaluations per game. With 18 workers the arbiter's batching is the unknown. |
| CPU training, 3,000 × 64 | **~2–6 h** | Two frozen-encoder forwards of about 1 GMAC each per row (the averaged prediction), at a CPU throughput not measured here. The build's timing replaces this estimate before the training approval. |
| Check | minutes | |
| Seed procedures (four blocks) | minutes each | as before |
| Screen, 64 games | ~40 min | about 32 s per game with both sides the combination |
| Strength, 1,018 games | **~9 h** (ceiling about 13.5 h at 1.5×) | same |

The total is roughly 15–22 machine hours, over separately approved runs. The engineering is about 350–450 lines plus
about 30 focused tests.

## 8. Unresolved material issues

1. **Is the mechanism new enough?** The argument is in §2. If the review judges it a continuation of #49 or #87 in
   other clothes, the pilot should not proceed.
2. **The exception to #66** for reusing the residual head, narrow and with no capacity claim. Whether the existing
   head alone could learn this target is unresolved.
3. **The target is the main scientific uncertainty.** With N_max = 49 at 200 simulations, σ spans about 99·Δq̂.
   Candidates eliminated after 3 visits carry noisy q and can draw much of the mass. That this target improves the
   prior is a hypothesis; the combination's PASS does not validate it. The definition is frozen without tuning. Its
   entropy, its mass on the played move, and its mass on candidates eliminated in the first round will be reported.
4. **The pilot can only move the policy.** If the agent's limits lie in the value, the pilot cannot find that out or
   fix it.
5. **Throughput** of synchronous halving plus two one-row requests per leaf through the arbiter is unmeasured. Step 2
   measures it. Sending both orientations in one two-row request would change the frozen evaluator's call pattern,
   and is not proposed.
6. **CPU training speed** is unmeasured until the build's timing. CPU is chosen because GPU training is unqualified,
   not because this recipe is known to hit the defect.
7. **Diversity without root noise:** the opening temperature is the only source. Duplicate counts are reported, not
   gated.
8. **Default self-play's seed defect** (§3.4) stays in the default path. This pilot does not depend on it. Fixing it
   would change default self-play and is not proposed.
9. **One round only.** A PASS would show a single amortization step under this protocol. It would show nothing about
   iterating, other simulation counts, deployment or the browser.

## 9. Approval boundaries

- **Requested now:** review of this proposal only.
- **Each needing separate approval:** the build (step 1); then for each of the four blocks its seeds, its run (with
  its stated ceiling) and its closing; and training with its check (with its stated ceiling).
- **Exceptions needing explicit approval:** reuse of the residual head under #66.
- **Not requested:** deployment, `--resume`, iterative training, value-head or encoder training, any GPU gradient, any
  sweep, and pushing.

## 10. As built (step 1, 2026-10-28)

**New:** `halving_policy_target.py`, `combination_selfplay.py`, `combination_policy_pilot.py`, and three gates,
created closed: `combination_policy_throughput_authorization`, `combination_policy_generate_authorization` and
`combination_policy_train_authorization`. Tests: `tests/test_combination_policy_pilot.py`, with the doubles in
`tests/_combination_fakes.py`.

**Unchanged:** `self_play.py`, `self_play_worker.py`, `trainer.py`, `train.py` and `mcts.py`. The four frozen modules
are pinned by sha256 in the entry: `mirror_halving_combination`, `mirror_averaging`, `root_sequential_halving` and
`policy_head_capacity`.

**Where the build differs from §3:**
1. **No flag through self-play.** §3 changes 1–4 proposed a `combination_agent` keyword on `self_play.play_game`, the
   shim in `self_play_worker` and a pass-through in `run_parallel_selfplay`.
   - Those paths carry the curriculum, the replay buffer, `GameComplete` statistics and visit-count
     `PositionRecord`s. This pilot uses none of them, and their rows cannot carry a float target.
   - `combination_selfplay` reuses their parts directly instead: one `InferenceServer` thread owns the device
     (`max_batch_rows` 14, flush 2 ms); spawned CPU workers use `RemoteEvaluator`; a shared claim counter assigns
     slots; and the queue sizes are `run_parallel_selfplay`'s.
   - Default self-play is untouched: the three files are byte-identical.
2. **The game is the match's loop, called rather than copied.**
   - `play_slot` calls `mirror_halving_combination.play_game` with the combination agent on both sides, at 200
     simulations under PROTOCOL.
   - Its only addition is `capturing`: a subclass of the halving search that hands each completed halving search's
     root to the target code.
   - The four streams, the readout, `last_choice` and the refusal of root noise are therefore the match's own code.
   - §3.1's refusal list (resign, adjudication, td1, tie-break, opponent, capture, `TWIXT_MIRROR_PROB`) names options
     of `self_play.play_game` that this path does not have. It checks instead that PROTOCOL is 24×24 with 280 plies,
     that the halving constants are the card's (`check_variant`), and that the seed is an int; it accepts no RNG.
3. **The remote shim** sets `network = None` on each worker's `RemoteEvaluator` instance, as §3.2 proposed, in
   `combination_selfplay.worker_main`.
4. **The arbiter's evaluator** is `LocalGPUEvaluator(network)` with `compile=False`, as self-play's arbiter uses. The
   strength match used `compile=True` in one process.
   - With the arbiter, a call's batch composition depends on worker timing. So a slot's game is not expected to be
     bitwise reproducible on the GPU, even though its seed is fixed.
   - §5 step 2's STOP "two slots' games depend on the worker that played them" is therefore checked two ways, not by
     replaying games on the GPU:
     - on the record: each game's recorded seed equals base + slot, and its rows match its halving plies;
     - by the CPU check with a batch- and padding-invariant fake evaluator (§6).
5. **Logging for closing.**
   - Each worker reports `start` with its slot before playing. The entry logs `game_start` (slot, worker) as it
     arrives, and `claims` (the final counter) after the last game.
   - Each worker reports its `MLX_ENABLE_TF32` value and whether MLX is loaded. Anything but "0" with MLX not loaded
     fails the run.
6. **Rows.**
   - One create-only `.npz` per game goes under `~/twixt-cache/combination-policy-pilot/generate/`, outside the
     repository. Its sha256 sits in the committed `games.jsonl`, beside the game's moves.
   - The throughput check writes no rows.
   - Each row keeps the prior, the target, the candidates (with their q, visits and rounds survived), N_max and the
     played move.
7. **Training as built.**
   - **Features:** the frozen encoder's features, canonicalized as `forward_padded` does, are computed outside the
     gradient for each orientation.
   - **Prediction:** log p̄_θ = logaddexp(log p_θ(s), log p_θ(M s) mapped back) − log 2. That is the log of the
     probability mean.
   - **Optimizer:** MLX's Adam with betas (0.9, 0.999), eps 1e-8 and bias_correction False. The L2 term covers all
     8,840 trainable parameters. There is no gradient clipping.
   - **Corpus check:** the corpus is re-verified file by file (sha256) and row by row: the legal-move count on replay,
     and the stored target against `halving_target` recomputed.
   - **Architecture:** the run refuses any architecture other than the timed one (30 channels, 128 hidden, 6 blocks).
   - **Output:** a run that reaches the held-out check writes the head's arrays to `head.npz` in its evidence
     directory. Only an ELIGIBLE verdict admits that head to a screen proposal.
8. **The timing** (`time`) runs without a gate, as part of step 1. It writes
   `docs/superpowers/evidence/combination-policy-timing/record.json`.
   - **What it times:** 2 warm-up and 20 timed updates, each synchronized with `mx.eval` on the parameters and the
     optimizer state; 5 forward-only batches; and `to_tensor` per row.
   - **Inputs:** the recipe's architecture with random weights (seed 20261030), on synthetic inputs (500 legal of 576
     cells).
   - **Projection:** it uses §4's estimated row counts (37,000, with 3,700 held out). The training ceiling is 1.5× the
     projection, rounded up to the minute.
9. **Admission.** The throughput check needs the committed timing record. Generation needs the committed throughput
   record (COMPLETED). Training needs both the generation and the timing records.
10. **Not built in step 1:**
    - the stage wrappers, and the seed-survey and closing copies. Each is built with its stage's seed preparation, as
      a copy of the strength-2 one with the differences named;
    - the screen and strength entry for the trained candidate, a later build. `ResidualPolicyHead.__call__` refuses
      the list API, so that build must confirm search reaches the head only through `forward_padded`.

**Unresolved issues found in the build:**
- **The 15-minute throughput rule may close the pilot on throughput alone.**
  - 144 games in 900 s needs about 576 games/h; §7 estimated 350–700.
  - Whole-run wall time is set by the slowest worker's last games. In entry 20, one state-cap game could finish
    alone for minutes.
  - As frozen, failing to finish is VOID.
- **GPU reproducibility:** see item 4 above.
