# Higher-budget search distillation — learning proposal (DESIGN ONLY, 2026-10-11)

**Status.** Design only. **Stage A is proposed; Stages B and C are provisional.** Nothing in this
document authorizes implementation, model loading, games, corpus generation, training, seed
registration, gate opening or a push. Each step below needs its own explicit approval.

**Why this document exists.** The training-replay capture checkpoints (scripted persistence, serial
hooks, a real-search counter probe) established capture *feasibility* only. Review then ruled
NO_GO for a learning experiment until a qualified learning mechanism and target source exist, and
paused capture development. This proposal names one mechanism and the qualification step that
must come first. It does not justify learning yet. `calib020_0001` remains the incumbent in every
outcome.

## 1. Hypothesis and target source

**Hypothesis.** The visit distributions from the incumbent's search at 1,600 simulations are a policy
target that makes the network stronger when it plays at 400 simulations.

**Target source.** `calib020_0001` searched by `MCTS.search_with_root` with a fresh root, no Dirichlet
noise, and the batch and stall settings of the #53 Stage 3 protocol (`PROTOCOL` in
`scripts/GPU/alphazero/external_learning_stage3.py`). Only the simulation count differs: 1,600 instead
of 400. This is the call the evaluation harness already makes at every ply, and it plays under our
exact rules and engine.

## 2. Objective and exact parameters changed

* **Trainable:** only the eight `policy_head.{conv,bn,fc,out}.{weight,bias}` tensors, 519 parameters
  in all.
* **Frozen:** the encoder, the value head, and every batch-norm running statistic. The forward pass
  runs in eval mode.
* **Loss:** policy cross-entropy against the normalized 1,600-simulation visit distribution, with no
  other term.
* **Schedule:** one shot from `calib020_0001`; 2 epochs; learning rate 0.0003; the final step only,
  with no checkpoint selection. This schedule is a design choice, to be frozen before any corpus
  exists.
* **Control arm:** the identical procedure, with labels from the same call at 400 simulations on the
  same positions. It controls for this particular fine-tuning procedure only (see §5).

## 3. Why it differs from the closed experiments

| Closed line | How this differs |
|---|---|
| #49 and #51 (cold and warm continuation, frozen-parent opposition) | Those were iterative self-play that updated the whole network, with same-budget visit targets and game-outcome value targets, and no measured teacher margin. This is one shot, policy head only, with a higher-budget target whose playing margin is measured before any corpus is built. |
| #53 (external-engine imitation) and the 2026-10-10 search-distribution memo | That teacher plays under different rules, and agreement with it is not strength under our rules. Here the teacher is our own search, under our rules. #53's training mechanics may be reused; nothing of #53 is extended or rescued. |
| Value calibration (v2–v14b) | Those trained the value head. Here the whole value path is frozen. |
| Convergence atlas, FPU, competitive readout | Those changed the search with the weights fixed. Here the search is unchanged and only the policy head learns. |

The atlas closeout asked that a high-budget distillation successor bring its own **stability rule**,
because 12 of its 24 rows lacked a stable 3,200/6,400 reference. Stage A does **not** supply one. A
playing margin establishes aggregate playing performance under the label protocol; it does not
establish per-position target stability or optimality. **The stability criterion is unresolved and
must be settled before any Stage B approval.**

## 4. Harness audit (read-only, 2026-10-11)

The standard checkpoint-match path cannot run Stage A as it stands:

1. `play_eval_game` builds both sides' searches from one `cfg_from(config)`, so red and black always
   share one simulation count.
2. `validate_ply` requires every root to finish with exactly `config.mcts_sims` visits. A 1,600-sim
   side would raise `IntegrityError` on its first move.
3. `AgentSpec` holds only `agent_id`, `checkpoint` and `readout`. Two agents may share a checkpoint
   (`same_checkpoint` is recorded), but nothing binds a simulation count to an agent.
4. The evaluation search is `search_with_root`: synchronous, a fresh root each ply, no noise. Self-play
   uses the batched `search_from_root` with tree reuse. The capture probe qualified only the batched
   path, so its counter results do not carry over to these labels.
5. Opening variety comes only from the opening-temperature readout, which draws from its own random
   stream and does not disturb the search's.
6. Evaluation replays keep the top two root children only, not full visit counts, so they cannot
   supply labels.

The repair (step A0 below) is small but is a harness change and needs its own approval.

## 5. Stage A — PROPOSED: does the label-producing agent beat the 400-sim agent?

**Agents.** The incumbent at 1,600 simulations against the incumbent at 400. Both use the label call of
§1 and the #53 opening-temperature readout: temperature 1.0 for the first 20 plies, then 0.1. For the
first 20 plies the 1,600 agent approximately samples its normalized visits; the existing readout adds a
`1e-8` floor to every count before the log-softmax, and that implementation stays unchanged. After
ply 20 it plays close to argmax.

**Unit and sampling.**
* A pair is two games with colours reversed, each independently seeded. **A pair's score is the mean
  of its two game scores** for the 1,600 agent (win 1, draw 0.5, loss 0).
* Pairs must be sampled independently, with both agents and the whole protocol frozen before the
  first game. Shared openings are unnecessary.
* N = 80 pairs, 160 games.

**Pass rule (exact).**

    mean − sqrt(ln(40) / (2N)) > 0.5

At N = 80 the subtracted term is 0.152, so a pass needs an observed mean above 0.652. This is the
two-sided 95% Hoeffding lower bound on pair means in [0, 1].

**Power.** A true effect of +25.2 points is a **conservative sufficient effect for 80% power** under
this method; it is not a necessary effect.

**A non-pass** is an operational stop for this line. It does not show that the 1,600 agent is no
stronger, and it does not show that distillation is futile.

**Steps before Stage A, each separately approved:**
* **A0 — harness change.** Per-agent simulation budgets, correct colour binding, per-side integrity
  checks and recorded budgets, with focused CPU tests. Legacy defaults stay unchanged. Full-visit
  export, workers and training are out of scope.
* **A1 — planning measurement.** One colour-reversed pair at 1,600 vs 400. It is not counted toward
  the result. After it, an explicit runtime cap with headroom is frozen; **reaching the cap before
  completion gives no verdict.**
* **Seeds.** Fresh accounting under the replacement seed-selection rule (ca9cb1eb).

**Planning runtime.** About 6.9 h, by linear extrapolation from the accepted 61.9 s per 400-vs-400 game
(one game at a time): a 1,600-vs-400 game is about 2.5 times that, and 160 games follow. This is an
extrapolation, not a measurement or a bound. The cap comes from A1, not from this figure.

## 6. Stage B — PROVISIONAL: corpus and offline learning, no strength games

* **Corpus.** 300 incumbent games at 400 vs 400 under the Stage A protocol, which matches the
  distribution the student will meet. Up to 16 positions per game, split by game: 240 games for
  training, 60 held out. Labels from the §1 call at both 1,600 and 400 simulations.
* **Planning cost (extrapolated).** Games about 5.2 h. Labels about 6.6 h: 9.6 million simulations at
  about 2.46 ms each (61.9 s over about 63 plies at 400).
* **Gate.**
  * The student's held-out KL to the 1,600 target, averaged per held-out game, must be lower than the
    control student's in at least 37 of 60 held-out games (exact one-sided sign test, α = 0.05; the
    game is the unit).
  * The student's top-1 agreement with the 1,600 argmax must be strictly above the control's.
* **Limits.** Neither check excludes a temperature-only explanation; that would need a fitted
  scalar-temperature arm, as in #53, with logits re-masked after scaling.
* **To freeze before any Stage B approval:** the stability criterion (§3), position selection,
  label-search randomness, and held-out tie handling.

## 7. Stage C — PROVISIONAL: strength

* The student at 400 against the incumbent at 400, same protocol, N = 200 pairs scored as in §5.
* Pass rule `mean − sqrt(ln(40) / (2N)) > 0.5`: the subtracted term is 0.096, so a pass needs an observed
  mean above 0.596. A true effect of +15.9 points is a conservative sufficient effect for 80% power.
* Planning runtime about 6.9 h (400 games at 61.9 s), extrapolated; the cap is frozen from a
  measurement first.
* A pass would show the student stronger than the incumbent under this protocol. It would **not**
  attribute the gain to deeper targets; that needs the trained control evaluated too, which is not
  budgeted here.

## 8. Forecast on record

* Stage A passes: about 50%.
* The full chain passes Stage C: at most 10%.

One non-pass validates no probability. `calib020_0001` stays the incumbent in every outcome, and
capture development stays paused.
