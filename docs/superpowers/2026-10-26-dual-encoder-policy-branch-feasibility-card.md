# Dual-encoder policy branch — design-only feasibility card

**Date:** 2026-10-26

**Status:** DRAFT, for the user's decision.
- **Authorized:** this design-only card.
- **Not authorized:** model loading, games, training, seed accounting, gates, a build and pushing.

**Question for the decision:** does this mechanism merit one experiment? It is not a way to rescue the closed D2
recipe (#87). **§7 recommends rejecting it on paper.**

## 1. Mechanism, and what is new

The agent is a **composite** of two networks with `calib020_0001`'s architecture:

- **Value:** the unchanged incumbent's encoder and value head, frozen.
- **Priors:** a trainable copy of the incumbent's encoder and policy head (the "policy branch"), distilled toward the
  incumbent's 1,600-simulation visit counts.

The value function is preserved **by construction**, not by an anchor penalty: no optimizer ever sees the value path.

Distinct from the closed lines:

| Line | What it did | Difference here |
|---|---|---|
| D2 recipe (#69, #87) | Trained the shared encoder; an anchor tried to hold the value | The value path is a separate, untrained network |
| Stage B (#54) | Trained 519 policy-head parameters on a frozen shared encoder | The whole encoder copy is trainable |
| Head capacity (#65, #66) | Asked whether a larger policy head could fit planted targets | No capacity question; judged by play at matched compute |
| Frozen-parent opposition (#51) | Self-play training against a frozen opponent | The frozen network is part of the agent; offline distillation |
| Mirror averaging (#59) | Two network calls per leaf, no training | Borrows only its cost-matching convention (§4) |

Nothing here shows that the policy branch would play better.

## 2. Frozen and trainable parameters

Architecture: 30 input channels, 128 hidden, 6 residual blocks (`network.py`).

| Part | Tensors | Parameters | Status |
|---|---|---|---|
| Value network: encoder | 52 + 26 BN running statistics | 1,809,024 | Frozen: the pinned incumbent file, loaded unchanged |
| Value network: value head | 4 | 66,049 | Frozen |
| Value network: policy head | 8 + 2 BN statistics | 519 | Computed by the evaluator, output discarded |
| Policy branch: encoder | 52 + 26 BN statistics | 1,809,024 | **Trainable** (BN statistics frozen; eval mode) |
| Policy branch: policy head | 8 + 2 BN statistics | 519 | **Trainable** (BN statistics frozen) |
| Policy branch: value head | 4 | 66,049 | Off-surface, unchanged, output discarded |

- **Trainable surface:** exactly `train_student`'s existing surface on the student (60 tensors, 1,809,543
  parameters).
- **Composite checkpoint:** two ordinary files, the pinned incumbent and one student safetensors. No new format.
- **Every evaluation path refuses** to load the student file outside the composite.
  - **Why:** played alone, the student's own value head would read the trained encoder. That is the closed recipe with
    its anchor removed.

## 3. Identity and value-preservation checks

**I1 — build tests (no model):** with fake evaluators, the composite:
- takes its priors only from the policy evaluator and its values only from the value evaluator;
- hands both evaluators the identical input arrays;
- never mixes the two.

**I2 — one bounded identity-and-cost run (the first model-loading step; needs approval).** The policy branch is a
fresh load of the incumbent file, with equal digests.
- **Positions:** the committed positions of the play-path timing run (entry 19).
- **Inference identity:** the composite's priors and values must equal the plain incumbent evaluator's **bitwise**.
- **Search identity:** 400-simulation searches with the same seeds must give identical moves and visit counts, as in
  the one-sync screen's exact-match check.
- **Measured at the same time:** the composite's and the plain evaluator's seconds per simulation under
  `MLX_ENABLE_TF32=0`, for §4.
- **Stop:** any difference stops for review. There is no tolerance.

**V1 — after training, in the training process:**
- The composite's values must equal the incumbent's **bitwise** on every fresh held-out and training record.
- The value network's digest must be unchanged before and after.
- Any difference is a VOID (construction failure).
- **No drift limits:** the drift is zero by construction, and V1 verifies it.
- The student's own value-head drift is reported as a diagnostic only.

**V2 — every evaluation run:**
- The value network's digest is checked at load and at the end.
- At the start, values on the I2 positions must equal a standalone incumbent's bitwise.

## 4. Inference cost and a fair comparison

**Cost.**
- One forward pass is about 1.039 GMAC; the encoder is 99.97% of it.
- The composite runs two forward passes per leaf: about 2.079 GMAC, a ratio of **2.00**.
- Model size doubles (1.88 M → 3.75 M parameters).
- **Precedent, entry 11** (old M3 Pro, default numerical setting; not re-measured here):
  - Two network calls per leaf cost the same per network evaluation (542 s for 323,608 calls vs 553 s for 323,983).
  - At matched network evaluations, the two-call side used **17% less search time**: half the leaves, so half the
    tree work.

**Comparison: matched network evaluations.**
- The composite plays at **200** simulations against the incumbent at **400** (its production budget).
- This is entry 11's convention.
  - **FLOPs:** the budgets are equal.
  - **Wall time:** on the precedent's measurement, the convention does not favour the composite.
- **Recorded per side**, as in entry 11's table: searches, leaf calls, network evaluations, network seconds and
  search seconds.
- **Stop and reprice:** if I2 measures the composite above 2.0× the incumbent's seconds per simulation. Matching
  evaluations would then favour it on wall time.
- **Not used:** an equal-simulation match (400 vs 400). It gives the composite twice the compute, so beating that
  cheaper baseline would not show better use of compute.
- **Deployment:** a pass would mean the composite at 200 simulations could replace the incumbent at 400 for the same
  network cost.

## 5. Fresh data, one bounded trial, stop conditions

**Fresh corpus.** 120 incumbent self-play games at Stage 2's configuration:
- 1,600 simulations, eval batch 14, root noise on, mirror off, serial capture;
- `MLX_ENABLE_TF32=0`;
- 12 phases of 10; slot 9 of each phase held out (108 training / 12 held-out);
- all 12 phases must finalize; a fresh 120-seed block.

**Why fresh:**
- #87 forbids judging a changed surface on the old corpus or held-out set.
- The old 108 games' numerical setting was never recorded.
- Both held-out sets are spent (#69, #86).

**Generation estimate:** attempt 2 ran at 1.65 s per ply (Stage 2: 1.19). At Stage 2's 58.5-ply mean, that is about
3.2 h under a 12-hour ceiling. These are estimates.

**Training (one run, 30-minute ceiling):** `search_distillation_d2.train_student`, unchanged, with:
- `ceil(4 × N_train / 64)` updates;
- batch 64, Adam at learning rate 1e-4, clip 1.0, L2 1e-4, seed 0;
- **anchor 0**.

The anchor is 0 because no played output reads the student's value. This is not an anchor tuned on the closed recipe,
and the refusal in §2 enforces it. Exactly one candidate is saved create-only and reloaded strictly.

**Steps, each run at most once and each with its own approval:**

| Step | Content | Stop |
|---|---|---|
| 1. Build | Composite evaluator, factory, tests | Regression differs from the 58-id baseline |
| 2. I2 | Identity and cost | Any difference; cost above 2.0× |
| 3. Generation | Fresh corpus | Any VOID; any phase unfinalized |
| 4. Training | Fit check | Non-finite; failed `train_student` check; held-out CE not strictly below the teacher's; V1 |
| 5. Screen | 32 pairs at **400 vs 400**: gross-regression check only (twice the compute, no strength claim) | Mean pair < 0.40 |
| 6. Strength | **509 pairs, composite 200 vs incumbent 400**, Stage B protocol, fresh seeds, both arms `MLX_ENABLE_TF32=0` | Pass iff mean − √(ln 40 / (2·509)) > 0.5, i.e. mean > **0.56020** |

**Rules for every step:**
- A STOP, VOID or timeout closes the trial: no retry, top-up, extension, budget change or selection.
- A kernel panic stops all GPU experiments, under attempt 2's rule.
- The strength size gives conservative 80% power at a true mean pair score of 0.60.

**Machine time:**
- I2: < 15 min.
- Generation: ~3.2 h.
- Training: < 1 min of GPU.
- Screen: ~1 h.
- Strength: about the cost of a 400-vs-400 match, ~11.8 h of search alone at 1.74 ms per simulation, which was
  measured under the default setting. The setting's effect on the play path is unmeasured.
- **Total: roughly 17–25 h**, plus the build.

## 6. What a pass would and would not show

- **A pass would show:** better play than the incumbent at equal network cost, under this protocol.
- **It would not show:**
  - that GPU training is reliable (#81, #82);
  - why the D2 value output drifted;
  - anything about T1j or twixtbot.
- **Promotion** would be a separate decision. It needs the browser export of a two-network model.
- **A non-pass** is powered only against 0.60, so it would not separate "no gain" from a small one.

## 7. Assessment and recommendation

**What the composite must achieve.** At half the incumbent's simulations, it must reach a mean pair score above
0.5602 (design target 0.60). Its prior must therefore be worth at least what doubling the search is worth, plus a
margin.

**What halving costs the incumbent is not measured.** The record holds two readings:

- **Stage A (entry 1):** 1,600 vs 400 scored 0.769, about +209 Elo for two doublings, under the old training-mode
  protocol.
  - If the gain is log-linear (an assumption), one doubling is about +104 Elo.
  - The 0.60 target is about +70 Elo.
  - The distilled policy would then need to be worth **about +175 Elo at equal simulations**. That is about 84% of
    the whole 4× search gain, from one offline pass over about 6,400 positions.
- **Entry 11:** a two-call method at 200 scored 19–13 (0.594, inconclusive) against 400.
  - So halving may cost less than the log-linear reading implies.
  - A smaller budget leans harder on the prior, which helps a better policy.

**What the record says about distilled students.** No distilled student has shown better play here. Stage B's fitted
its targets better and had no per-game advantage. Neither D2 candidate was played.

**My estimate**, offered for this decision. It is not a prediction on record unless the user adopts it:
- training passes: about 85%;
- the strength step passes: **about 5–10%**;
- most likely mean pair score: 0.40–0.55.

**Recommendation: reject on paper.**
- Machine time is not the obstacle (about 17–25 h).
- The compute-fair bar asks for a gain of a size nothing in the record supports.
- A non-pass would be weakly informative (§6).
- If the user runs it anyway, §5 is the whole experiment, and any change is a new card.

**What would change this recommendation:** evidence that halving simulations costs the eval-mode incumbent little.
Measuring that is a separate experiment and is not proposed here.

**Not proposed:** branching only the top blocks (a shared frozen trunk) would cut the cost ratio below 2. It would also
reopen the frozen-feature question that #66 shelved, so it needs its own rationale.

## 8. The user's decision (2026-10-26): REJECTED on paper

**Status:** this section supersedes the "DRAFT" status at the top. The card is kept unchanged above as the reviewed
proposal. Nothing in it was built or run: no model was loaded, and no seeds, games, training or gates were used.

**Decision:** reject on paper; step 1 is not built. The architecture preserves value, but the current evidence does not
justify its added inference cost. **This experiment has not earned its cost.** The decision does not say that dual
encoders cannot work, or that preserving value cannot help.

**The rationale, narrower than §7's estimates** (the user's corrections):
- **+175 Elo is an illustrative scenario, not a measured requirement.** It assumes log-linear search gains, carried over
  from the old protocol, and a target true score of 0.60.
- **Mirror averaging's 0.594 does not isolate the cost of halving simulations:** that match changed the evaluator too.
  §7's sentence "So halving may cost less" claims more than the result supports.
- **A non-pass would not be uninformative.** It could reveal harm or rule out a large benefit. The match is simply not
  designed to resolve modest improvements reliably. §6's and §7's wording ("weakly informative") understated this.
- **The 5–10% figure is my subjective assessment.** It was not adopted as a calibrated forecast.

**Not to be read as** a negative experimental result about capacity, distillation or playing strength. Nothing was
measured.

**Status after the decision:**
- `calib020_0001` remains the incumbent.
- The broader goal of stronger play remains open. This proposal does not provide a sufficiently supported next
  experiment.
- No build, model loading, seed work, training, games or pushing is authorized.
