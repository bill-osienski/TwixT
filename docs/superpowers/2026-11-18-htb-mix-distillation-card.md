# Experiment card H-MX: mixed-data distillation from the current `hard` (draft)

**Status:** FROZEN at revision 1, 2026-11-18. The user approved the card, the narrow permissions in §2 and the
**model-free build only**. Nothing is loaded, trained, exported or played under this card. Before freezing, the only
work done for it was the read-only feasibility checks in §4–§5 and the planning computation in §10. It follows H-OP
(ledger entries 69 and 72, #206 and #209).

**Approvals:** the build returns once for launch review. Each runtime step in §14 needs its own approval.

**Revision 1 (2026-11-18, the user's review), before any build:**
- **Kept as drafted:** the dose (7,500 updates), the 16 + 16 data mix, the cached validation sets and the 400-pair
  match.
- **The exact sign test is adopted, with two corrections:**
  - §9 now states what the test establishes and the assumptions behind its 5% bound. The mean-above-0.50 condition
    is an additional guard, not part of that bound.
  - The power table and minimum pair count in §10 are recomputed from the specified inputs. The draft's 82% at 400
    pairs had used the planning model's own decisive rate at +70 (0.120) in place of the specified 0.114. The
    corrected value is **about 80% (0.7998)**. Every Elo conversion, and the retrospective H-OP sensitivity estimate,
    is tied explicitly to the planning model.
- **The cached validation sets are one-time fit screens,** with the disclosure that their earlier results informed
  this design (§7).
- **Recorded:** the narrow data-reuse exceptions and the planning-only use of H-OP's match summary (§2). The open
  questions are resolved (§17).

## 1. Question, hypothesis and mechanism

**What H-OP showed (entries 69 and 72).**
- The H-OP fit moved the current student strongly toward the teacher on held-out positions from `hard`'s self-play:
  KL 0.7073 → 0.4176 (−41.0%), top-1 0.5761 → 0.6789, value MSE 0.02578 → 0.01971.
- On the first fit's 24,000 validation rows (the original H-TB distribution) it moved away: KL +21.8%
  (0.5745 → 0.6996), top-1 0.6514 → 0.6101, value MSE 0.0703 → 0.1147 (+63.1%). That was descriptive and never gated.
- Its challenger then scored a mean pair score of 0.4982 against the current `hard` over 272 colour-reversed pairs.
  The verdict was NON_PASS.

**The specific change.** H-OP trained only on H-OP rows. This card trains on a fixed 50/50 mix of the two
teacher-labelled training sets that already exist: 16 H-OP rows and 16 H-TB rows in every 32-row batch. Everything
else in H-OP's recipe stays: the warm start, the update count, the architecture, the optimizer and the losses (§6).

**Hypothesis H-MX.** A challenger warm-started from the current student and fitted to the 50/50 mix is **stronger
than the current `hard`** under the same 200-simulation protocol. Operationally, stronger means it wins more decisive
colour-reversed pairs than it loses over the planned opening sample (§9).

**Why it is plausible:**
- **Interference, and rehearsal as its remedy.** Fine-tuning a network on a new distribution alone can degrade it on
  the old one. Rehearsing old data alongside the new is the standard countermeasure. H-OP measured such a
  degradation on the old rows (above).
- **Coverage plus new positions.** The H-TB rows cover the corpus distribution the current student was trained on.
  The H-OP rows cover positions `hard` now reaches. A mixed fit may keep the first while learning the second.
- **Cheap to test.** Both label sets exist and are verified (§4). No generation, labelling or teacher load is
  needed. A fit gate costing about 1.8 h decides whether the 8.5 h match is proposed at all.

**What could make it fail, and what each outcome cannot separate:**
- **The old distribution may matter little for play.** Match games start from random 4-ply openings and are played by
  `hard`-like players, so they are closer to the H-OP distribution than to the corpus. If so, preserving old coverage
  buys little, and mixing mostly dilutes the new data.
- **Dose.** At 7,500 updates × 16 rows, each H-OP row is seen **once**; H-OP saw each one twice. Half of the H-TB
  training rows are seen once; the current student has already seen all of them twice. No outcome here separates the
  effect of mixing from the effect of the halved H-OP dose (kept by the user's decision, §17).
- **H-OP's match may not have been sensitive enough.** Under the planning model of §10 (a model, not a
  measurement), H-OP's pass rule had little power below about +225 Elo-equivalent. H-OP's NON_PASS therefore says
  little about whether its challenger gained strength, and nothing about whether the retention loss mattered. The
  hypothesis that it did is **a conjecture, not a measured cause**.
- **The same capacity and label-quality limits as H-OP** (H-OP card §1) still apply.
- **The gate passes and the match does not:** the mix kept the old distribution and learned the new one, without a
  strength gain the match detected. That is not evidence of no gain (§10).
- **A PASS** shows that the mixed update wins more decisive pairs than it loses against the current `hard`, over
  the planned opening sample (§9). It does not show that mixing, as
  opposed to continued training on teacher labels at this dose, caused the gain. The challenger is not compared with
  H-OP's challenger.

**This card is not** a change to the teacher, its configuration or its role. The teacher is not loaded and plays no
game. It is not a deployment step either.

## 2. Ledger items this card touches, and the narrow permissions it needs

**Approved 2026-11-18 (the user's review), recorded here and in the ledger:** items 1–5 below, **for this card only**:
its model-free build, and the runs in §14 once each, each run with its own approval. None of them covers a second
round, another mix ratio, another recipe, another dose, or any rescue after a result.

1. **#53 (twixtbot as a training source) and #87 (distillation).** #199 says H-OP's exceptions cannot be cited for
   "another recipe". **Approved:** new narrow #53 and #87 exceptions for H-MX.
2. **#177 and #178 (the H-TB labels and the first fit).**
   - **Approved:** read the 240,000 H-TB training labels (`ec28e768…`) as a training input to this one fit, as
     written. No row is edited, relabelled or reordered on disk.
   - **Approved, as a one-time fit screen:** an exception to #178's "reuse the 24,000 validation rows to tune or
     select a later method". The rows (`5d3dcee9…`) are scored once, inside the one fit, for screens M4–M6 (§7), to
     stop or pass this challenger. They are never re-scored to revisit the verdict.
   - **Disclosure:** these rows were scored before, for the first fit's F1–F3 and for H-OP's descriptive retention
     (entry 69). **Those earlier results informed this design:** the hypothesis and the M4–M6 bars, which are set
     from H-OP's measured regression on these rows. Nothing was tuned on them.
3. **#205 (the H-OP labels).**
   - **Approved:** read the 120,000 H-OP training labels (`b0ea21cb…`) as written, as training input.
   - **Approved, as a one-time fit screen:** score the 12,000 H-OP validation labels (`66cfcaf2…`) for screens M1–M3,
     once, inside the one fit.
   - **Disclosure:** these rows were H-OP's I1–I3 gate set, and their results (entry 69) informed this design. M1–M3
     copy I1–I3.
   - Nothing on disk changes. The 18,021 / 3,130 positions H-OP's selection left out stay unused.
4. **#206 (the H-OP fit is spent: "do not … fit again with another recipe … or row set").** This card is a fit with
   another row set. **Approved:** an exception for this card's one fit. H-OP's challenger (`5178a47c…`) is not used,
   loaded or compared.
5. **#209 (H-OP is spent).**
   - No H-OP stage is rerun. The H-OP match is not re-scored or re-analysed for a gain.
   - **Recorded, planning-only use:** two summary figures from the H-OP match were read **only as planning inputs**
     to §10: the decisive-pair rate (31/272 = 0.114) and the pair-score SD (0.165). They set the planning model and
     the power table. No win/loss split, colour result or other figure from that match enters this card.
   - The use is not a re-analysis. It changes nothing about H-OP's verdict, and it supports no claim about H-OP's
     challenger.
   - #209 asks that a further attempt name a specific change and why it could improve play. §1 does that.

**Unchanged constraints, no exception needed:**
- **#184 and #188.** The student `0e1cb621…` and the pair `78c16a31c3dba37f` stay unchanged. The challenger is a new
  file, warm-started from a copy.
- **#186 and #187.** Python ORT parity runs only after the two-conv canary passes. `onnxruntime-node` stays at exactly
  1.23.2.
- **#194 and #195.** E1–E3 keys are excluded from the training mix (§5). The 11 H-TB rows they hit are dropped.
- **#200.** `htb_pilot.py`, `htb_onpolicy.py`, `onpolicy_match.mjs` and `onpolicy_match_analyse.mjs` are **not
  edited**. H-MX code lives in new files that import them read-only.
- **#165** is not revived. **#52** is not needed: the teacher plays no game.

## 3. Identities (fixed before any work)

| Role | Identity |
|---|---|
| Current student (MLX), warm start and reference S | `~/twixt-cache/htb-pilot/student.safetensors`, sha256 `0e1cb621…`, state digest `586dca4f…` |
| Current `hard` (product) | ONNX pair `78c16a31c3dba37f` (onnx `bc68a274…`, exported from that student), `protocolMove`: mirror averaging, 200 simulations, T 1.0 through ply 19, root sequential halving from ply 20 |
| H-TB labels | `~/twixt-cache/htb-pilot/labels-{train,val}`; manifests `ec28e768…` (240,000) / `5d3dcee9…` (24,000); selection `161babe3…`; label record committed at `docs/superpowers/evidence/htb-label/`; states rebuilt from the C0 corpus, refused unless it reproduces C0's eligible digest |
| H-OP labels | `~/twixt-cache/htb-onpolicy/labels-{train,val}`; manifests `b0ea21cb…` (120,000) / `66cfcaf2…` (12,000); selection `ecde6637…`; states rebuilt from the committed `htb-onpolicy-gen/games.jsonl` (5,197 games) |
| Research runtime | project `.venv`: Python 3.14.10, MLX 0.30.3, NumPy 2.4.1, `MLX_ENABLE_TF32=0`, CPU |
| Product runtime | Node as pinned for H-OP, `onnxruntime-node` 1.23.2 |
| Challenger | created by §6, saved create-only to `~/twixt-cache/htb-mix/challenger.safetensors`; its export in §8 to `~/twixt-cache/htb-mix/export/<id>/` |
| Namespaces | fit order `htb-mix-fit/1`; match `htb-mix/20261118/…` |

## 4. Data reuse: both cached datasets, as written (read-only checks, 2026-11-18)

All four label sets were reloaded through the frozen loader `P.load_label_compact`. It verifies the manifest hash,
every shard's sha256, and every shard's row-list hash. Then real examples were built through `P.ExampleStore.take`,
which replays each game and **re-derives every row's key**.

| Set | Manifest (matches the committed record) | Rows | Distinct keys | Rows equal the committed selection, in order | Plies | Games |
|---|---|---|---|---|---|---|
| H-TB train | `ec28e768…` ✓ | 240,000 | 240,000 | ✓ | 1–247 | 15,590 |
| H-TB val | `5d3dcee9…` ✓ | 24,000 | 24,000 | ✓ | 2–279 | 1,899 |
| H-OP train | `b0ea21cb…` ✓ | 120,000 | 120,000 | ✓ | 4–571 | 4,678 |
| H-OP val | `66cfcaf2…` ✓ | 12,000 | 12,000 | ✓ | 4–570 | 519 |

- **Shapes and format:** both are `htb-labels/1` shards, with targets (n, 576) float32 and values (n,) float64. The
  same `ExampleStore` serves both.
- **States:** the C0 corpus pool reproduces its eligible digest (17,489 games, 18.5 s). The H-OP games read from the
  committed file (5,197 games).
- **Examples built and keys re-derived** on a 2,048-row hashed sample of each set: 1.88 / 1.44 / 1.33 / 0.74 ms per
  row. That is negligible next to an update (≈ 0.7 s).
- **Memory:** compact arrays total about 0.9 GB. H-OP's 12 GiB launch check is kept.
- **Conclusion:** both datasets are reusable **as written**. Nothing is rewritten, re-sharded or relabelled. The fit
  admits each set only through its committed record and manifest hash.

## 5. Leakage and overlap (read-only checks, 2026-11-18)

**Key:** the first fit's canonical key, used unchanged by H-OP: sha256 of the smaller of the state's serialization
and its left–right reflection's. **Equal keys therefore cover mirrored positions.** Colour-swap transposition is not a
symmetry this key captures, as in every earlier card.

**Pairwise key overlap** across all four sets (all six pairs):

| Pair | Shared keys |
|---|---|
| H-TB train × H-TB val | **0** |
| H-TB train × H-OP val | **0** |
| H-OP train × H-TB val | **0** |
| H-OP train × H-OP val | **0** |
| H-TB train × H-OP train | 0 |
| H-TB val × H-OP val | 0 |

The combined training set (any subset of H-TB train ∪ H-OP train) shares **no key, direct or mirrored,** with either
validation set.

**E1–E3** (benchmark, G3, Diagnostic O), from H-OP's `build_exclusions` on committed evidence (31,299 / 10,079 / 949
keys):
- They hit **11 H-TB training rows**, all at plies 1–3 (early opening positions).
- They hit no H-TB validation row. The H-OP rows were already net of E1–E3.
- **The 11 rows are dropped from the H-TB pool by key** (§6), so the training mix is net of E1–E3, as H-OP's was.

**The fit re-runs every check above at launch** and STOPs on any non-zero overlap. That covers both validation sets
against the actual training rows used, E1–E3, duplicate keys, and the manifest and corpus digests.

## 6. Fit: H-OP's recipe, with the mix

| | H-OP fit (entry 69) | This card |
|---|---|---|
| Initialization | current student `0e1cb621…`; a second untouched load is S | **same** |
| Data | 120,000 H-OP rows | **H-OP 120,000 rows + H-TB 120,000 rows** (below) |
| Batch | 32 rows × 2 orientations, by `htb-onpolicy-fit/1` order | **16 H-OP + 16 H-TB rows × 2 orientations** |
| Updates | 2 epochs = 7,500 | **7,500** (one pass over the H-OP rows; one pass over the H-TB subset) |
| Architecture, inputs | 128 × 6 | **same** |
| Loss | masked CE on both orientations + 1.0 × value MSE + 1e-4 × L2, mean over the batch | **same**, so each source gets equal weight |
| Optimizer | Adam lr 1e-4 (defaults read back), clip 1.0 | **same** |
| Device, BatchNorm | MLX CPU, eval mode throughout, running statistics bitwise unchanged | **same** |
| Start check | KL(T ‖ S_MA) ≥ 0.01 on the first 2,048 rows | **same**, on the first 64 mixed batches (1,024 rows of each source) |

**Fixed batch composition:**
- **H-OP rows:** all 120,000, ordered by sha256(`"htb-mix-fit/1|op|<key>"`).
- **H-TB rows:** the 240,000 training rows minus the 11 E1–E3 rows, leaving 239,989. Ordered by
  sha256(`"htb-mix-fit/1|tb|<key>"`); the **first 120,000** are taken.
- **Batch b:** H-OP rows 16b to 16b+15, then H-TB rows 16b to 16b+15.
- **No duplicate key** across the 240,000 (checked).

**Build:** a new `fit_mixed` in a new module, `scripts/GPU/alphazero/htb_mix.py`. It is `htb_onpolicy.fit` line for
line except for batch assembly. A model-free test shows that with one source it reproduces `htb_onpolicy.fit`
bitwise.

## 7. The fit gate: a cheap stop before any export or game

S = the current student; C = the mixed challenger. Both are scored with the pilot's mirror-averaged definitions
(`P.score`) on both validation sets, inside the fit run.

**Improvement on the H-OP validation set (12,000 rows)**, the same bars as H-OP's I1–I3:
- **M1:** KL_C ≤ **0.90** × KL_S. With S = 0.70735, the bar is ≤ 0.636615.
- **M2:** TOP_C > TOP_S (strict). The bar is > 0.57608.
- **M3:** MSE_C ≤ MSE_S. The bar is ≤ 0.025782.

**Limited regression on the original H-TB validation set (24,000 rows):**
- **M4:** KL_C ≤ **1.05** × KL_S. With S = 0.57449, the bar is ≤ 0.603215.
- **M5:** TOP_C ≥ TOP_S − **0.010**. The bar is ≥ 0.64142.
- **M6:** MSE_C ≤ **1.15** × MSE_S. With S = 0.070317, the bar is ≤ 0.080865.

The bars are computed in the run from S's full-precision scores. The values above use entry 69's rounded figures.

**One-time fit screens, not fresh held-out estimates.**
- Both validation sets are scored **once**, inside the one fit, for S and C. The M1–M6 verdict is final, and the sets
  are not re-scored to revisit it.
- Neither set's rows train anything (§5).
- **Disclosure:** both sets have been scored before, and those results informed this design.
  - The H-OP set was H-OP's I1–I3 gate set; M1–M3 copy those bars.
  - H-OP's retention scores on the H-TB set (entry 69) motivated the hypothesis and set the M4–M6 bars.
- So M1–M6 screen this challenger against bars chosen with knowledge of these sets. They are not unbiased estimates
  of how it generalizes to new positions. Only the fresh match (§9) tests playing strength.

**Why these bars.** They are chosen, not derived.
- M1–M3 keep H-OP's requirement that the fit move toward the teacher where `hard` plays.
- M4–M6 allow **at most about a quarter of H-OP's measured regression** on each metric, rounded down: KL +21.8% → 5%;
  top-1 −0.0413 → 0.010; value MSE +63.1% → 15%. Continued training of any kind moves these numbers a little, so a
  zero-regression bar would stop on noise-sized drift.
- **H-OP's challenger would fail M4, M5 and M6 by wide margins** (0.6996, 0.6101, 0.1147), while passing M1–M3. This
  gate is the one H-OP did not have.

**S must reproduce H-OP's measurement.** S's six scores must equal those in H-OP's fit record (`8fd862d8…`, entry 69)
to within 1e-9 relative. Otherwise STOP: the scoring path or the data changed. If the build finds MLX CPU scoring is
not bitwise repeatable, this becomes 1e-6, fixed before launch.

**Verdicts:**
- **ELIGIBLE** needs M1–M6, every check in §5–§6, and unchanged BatchNorm statistics.
- If any of M1–M6 fails, the verdict is **STOP: the mixed challenger missed the fit gate**. All twelve values are
  reported, with C − S, C / S and the direction for each metric. A failed integrity check is a STOP for that reason.
- **Descriptive, never gated:** the same metrics beside H-OP's challenger's recorded values (entry 69), for context.

**Cost:** about 15 min of scoring inside the fit. The gate measures imitation on two position distributions. It says
nothing about playing strength.

## 8. Export and parity, before any game

H-OP's §9, unchanged:
- The unmodified exporter with the August invocation. The export is held outside the product manifest, and
  `HARD_MODEL_ID` is unchanged.
- V2-R surfaces and tolerances, and V3 mirror-averaged parity, each Python ORT surface after the two-conv canary. The
  Node runtime canary passes.
- **Ops and shapes identical to `78c16a31c3dba37f`'s.** Any failure STOPs.

## 9. The test of improvement: a fresh colour-reversed match against the current `hard`

- **Players:** the mixed challenger's export and `78c16a31c3dba37f`, both through `protocolMove` at 200 simulations,
  in one Node process. This is H-OP's driver pattern, in new files that import H-OP's frozen helpers read-only.
- **Openings:** fresh 4-ply openings, `htb-mix/20261118/match opening <i>`, by H-OP's rule. An opening is skipped if
  it equals, directly or reflected:
  - any benchmark or G3 opening;
  - any H-OP generation or generation-smoke opening;
  - any H-OP match or match-smoke opening;
  - an earlier opening of this card.
- **Pairs: 400** (800 games). Pair *p* plays opening *p* twice: game 0 with C as Red, game 1 with C as Black.
- **Seeds:** sha256(`"htb-mix/20261118/match/pair <p>/game <g>/ply <k>"`). Both players draw from the same per-ply
  seed.
- **Pair score:** C's mean over its two games (win 1, draw 0.5, loss 0). Game end by the product's rule only.
- **Pass rule: an exact one-sided sign test over decisive pairs, α = 0.05, plus a mean-score guard.**
  - A pair is a **pair win** if its score is > 0.5, a **pair loss** if it is < 0.5, and **uninformative** at 0.5.
  - Let W be the pair wins, L the pair losses, and m = W + L the decisive pairs.
  - **The test:** reject H0 iff m ≥ 1 and P(Binomial(m, ½) ≥ W) ≤ 0.05. Examples of the threshold: m = 40 needs
    W ≥ 26; m = 46 needs W ≥ 30; m = 60 needs W ≥ 37.
  - **The additional guard:** the observed mean pair score must exceed 0.50.
  - **PASS** iff the test rejects **and** the guard holds. Anything else that completes and passes the compute guards
    is **NON_PASS**.
- **What the test establishes.** It evaluates whether **the challenger wins more decisive pairs than it loses**.
  Formally:
  - H0: among decisive pairs, P(C wins the pair) ≤ ½.
  - H1: P(C wins the pair) > ½.
  - Both are stated over **the planned opening sample**: fresh 4-ply openings by the frozen rule (above), each played
    twice with colours reversed, under this protocol.
  - A PASS rejects H0 at one-sided α = 0.05. It does not estimate the size of the advantage. It does not cover other
    openings, the empty board, other time controls or other opponents.
  - It holds for the whole update: data mix, dose and continued training together.
- **Assumptions behind the 5% bound:**
  - **(a) Independent pairs.** Each pair's outcome depends only on its own opening and its own per-ply seeds. The
    openings and seeds are fixed by hash rules before any game. The players carry no state from one game to the next.
    `protocolMove` is a stateless function of the position, the policy and the seeds, and the inference objects keep
    no evaluation cache between calls (read 2026-11-18: `server/protocol_move.js`, `server/inference.js`,
    `server/mirror_inference.js`). Nothing learns during the match.
  - **(b) Exchangeable, fair signs under the null.** At P(C wins a decisive pair) = ½, the win/loss signs of the
    decisive pairs are independent and exchangeable. Given m, W ~ Binomial(m, ½). At P < ½, W is stochastically
    smaller, so the bound still holds. The exact conditional binomial then gives P(reject | m) ≤ 0.05 for every m, so
    the unconditional false-PASS rate is ≤ 0.05 whatever the decisive-pair rate.
  - **(c) A fixed design.** The 400 openings, the pair count and the rule are frozen before play. There is no
    sequential look, top-up, discarding or reweighting of pairs. A VOID or INVALID run is unscored as a whole, never
    truncated and scored.
  - **(d) Scope.** The opening sample is drawn by the planned rule with its exclusions, so the bound and the claim
    apply to that opening distribution only.
- **The mean guard is not part of the 5% bound.** It can only turn a test rejection into NON_PASS, so it never raises
  the false-PASS rate. It rules out a PASS with W > L but a mean ≤ 0.50, which is possible only through 0.75/0.25
  pairs (games with draws).
- **Why a sign test rather than H-OP's Hoeffding bound** (§10): split pairs say nothing about which model is stronger.
  Under this protocol most pairs split, and Hoeffding's range-based bound then wastes almost all its power.
- **Reported, descriptive:**
  - W, L, m and the 95% Clopper–Pearson interval for W / m;
  - the mean pair score with its two-sided 95% Hoeffding interval (half-width 0.0679 at 400 pairs);
  - results by colour, draws, game lengths and the count of distinct games.
- **Guards** (H-OP's §10 guards; any failure means INVALID, which is not scored):
  - one row per played ply;
  - a 200-simulation budget **and** 200 root visits on both sides at every move;
  - C's total search seconds ≤ **1.10** × the current `hard`'s;
  - both models' sha256 and the runtime versions checked before load and in the record;
  - the run bound to its launch commit and process group, with the gate restored.
- **Smoke:** 2 pairs (4 games), `htb-mix/20261118/match smoke/…`. Replay-validated, never read or scored, and used
  only for timing.

## 10. The smallest gain worth detecting, and the power plan

**Planning input** (§2 item 5, planning-only use): two summary figures from the H-OP match. 241 of 272 pairs split
1–1, so the decisive-pair rate was 31/272 = 0.114. The pair-score SD was 0.165. H-OP's power plan had assumed SD 0.50.

**Power, from specified inputs only.** The calculation needs nothing but n, a decisive-pair rate d and a win
probability among decisive pairs q:
- m ~ Binomial(n, d), and W | m ~ Binomial(m, q);
- PASS by the test iff P(Binomial(m, ½) ≥ W) ≤ 0.05;
- draws are ignored (2 of 544 H-OP games), so the mean guard is not modelled.

| Win probability among decisive pairs, q | d = 0.08 | **d = 0.114** (H-OP's rate) | d = 0.16 |
|---|---|---|---|
| 0.500 (null: false PASS) | 0.036 | 0.037 | 0.039 |
| 0.640 | 0.42 | 0.55 | 0.69 |
| **0.691** | **0.65** | **0.80** (0.7998) | **0.91** |
| 0.760 | 0.89 | 0.97 | 0.99 |

All at **n = 400 pairs**.

- **Minimum pair count for ≥ 80% power at q = 0.691:** **401** at d = 0.114 (0.8008; 400 gives 0.7998); 571 at
  d = 0.08; 285 at d = 0.16.
- **The card keeps 400 pairs** (the user's decision, 2026-11-18): about 80% at the specified inputs.
- These power figures are conditional on the match completing and clearing its guards. The false-PASS bound (≤ 0.05)
  rests on §9's assumptions (a)–(d), not on these inputs.
- No sequential look, no top-up and no pooling with the smoke.

**Smallest gain worth detecting: q = 0.691,** the challenger winning about 69% of decisive pairs.

**Elo conversions: planning model only.** Every Elo figure in this card comes from the following planning model, not
from a measurement:
- Each opening carries its own Red advantage F_i ~ Normal(μ, s) on the logistic scale. A strength edge of D Elo adds
  to whichever side C plays. Draws are ignored.
- Calibrated so that at D = 0, P(Red wins) = 0.864 and P(decisive pair) = 0.114. This gives μ ≈ 725 and s ≈ 583 in
  Elo-equivalent units.
- **Under that model only,** q depends on D alone: q = 0.640 ↔ +50, **q = 0.691 ↔ +70**, q = 0.760 ↔ +100.
  - The model's mean pair score at +70 is about 0.523.
  - The model's own decisive rate rises slightly with D, to 0.120 at +70. The table holds d fixed, as specified.
- "+70" keeps H-OP's intent that a smaller gain would not justify an integration cycle. The observable target is
  q = 0.691; the Elo figure is that model's translation of it.

**Retrospective sensitivity of H-OP's rule, under the planning model only.** These figures assume the calibrated
model above; they are not measurements and are not a re-analysis of H-OP's outcome.
- H-OP's rule needed a mean pair score above 0.5742. Under the model, that corresponds to about +225 Elo-equivalent.
- Under the model, the rule had about 1% power at +145 and 74% at +250.
- H-OP's §11 equated 0.60 with "about +70 Elo" and assumed SD 0.50. Under this model, neither holds for this
  protocol's first-move advantage.
- H-OP's NON_PASS stands as written. Under the model, it is weaker evidence against a gain than H-OP's §11 figures
  implied.

**Levers considered:**
- **H-OP's Hoeffding rule:** it would need far more than 18 h of games to have power at q = 0.691. Rejected.
- **544 pairs:** more power, but the 1.5× projection is about 17.4 h, against the 18 h ceiling. Not chosen.
- **Balanced openings:** these would raise the decisive rate. But selecting them needs an evaluator, which biases the
  opening set, and it moves the protocol away from product play. Not proposed.

## 11. Compute guards and ceilings

- **Every stage runs under a wrapper with a fixed ceiling** (TERM 10 s before KILL), a one-line gate-open commit in a
  clean linked worktree, the idle gate, BOINC quit and Unreal Editor closed. The gate is restored on every exit.
- The gates are new: `HTB_MIX_FIT_AUTHORIZED`, `HTB_MIX_EXPORT_AUTHORIZED`, `MIX_MATCH_SMOKE_AUTHORIZED` and
  `MIX_MATCH_AUTHORIZED`. The gate inventory test derives the count.
- A timeout or failure is **VOID**: no automatic retry, and no partial run is scored or used.
- The closing commit is verified before any fast-forward, which is always its own separate step (#207).

| Stage | Fixed ceiling | Launch rule |
|---|---|---|
| Fit + gate (both validation sets) | 11,700 s | 12 GiB memory check; all §4–§5 checks re-run at start |
| Export + parity | 3,600 s | ELIGIBLE fit, committed |
| Match smoke (4 games) | 1,800 s | PARITY_PASS, committed |
| Match (400 pairs) | **18 h (64,800 s)** | LAUNCH_OK iff 1.5 × the smoke's projection ≤ 64,800 s |

**Matched compute in the match:** identical graphs (§8), 200 simulations on both sides enforced per move, and the
1.10 search-seconds guard. H-OP's match ran at a ratio of 0.99987.

## 12. Total cost (planning estimates from measured paces)

| Step | Estimate | Pace it rests on |
|---|---|---|
| Build (model-free, behind closed gates) | not machine time | — |
| Fit, gate and scoring | **~1.8 h** | H-OP fit, 6,526 s: the same 7,500 updates and the same 12,000 + 24,000 scored rows |
| Export and parity | < 5 min | H-OP export, 12 s |
| Match smoke | ~3 min | H-OP smoke |
| Match | **~8.5 h** (30,600 s; 1.5 × = 12.8 h) | H-OP match: 76.5 s per pair (38.3 s per game) |
| **Total** | **~10.5 h of CPU machine time**, in three approved runs | no generation, no labelling, no GPU, no teacher |

- **If the gate STOPs, the experiment costs about 1.8 h.**
- The H-OP smoke underestimated the match's per-game time by 19% (32.0 s against 38.3 s). The 1.5× launch margin
  covers that.

## 13. Decision rule (fixed before any work)

| Outcome | Reading | What it permits |
|---|---|---|
| Any stage VOID or INVALID | no result | nothing; no automatic retry |
| Overlap or integrity check fails at fit start | no result | STOP; a reviewed decision; no silent repair |
| **STOP: missed the fit gate** (any of M1–M6) | the mix did not both learn the H-OP positions and keep the H-TB positions to the stated bars; dose and mix are not separated | the line closes; no export or match |
| **ELIGIBLE** | the challenger cleared all six bars on these positions, and nothing about play | proposing §8 (approval), then §9 (approval) |
| Match **PASS** | the exact sign test rejects (C wins more decisive pairs than it loses over the planned opening sample, one-sided α = 0.05, under §9's assumptions) and the mean pair score exceeds 0.50; for the whole update | proposing an integration decision (export pin, G2/G3-style checks), and nothing more |
| Match **NON_PASS** | superiority in decisive pairs not shown; not evidence of no gain (about 80% power at q = 0.691, d = 0.114) | the current `hard` stays; the line closes; no top-up or re-score |

PASS authorizes no deployment, push or further training. Another mix ratio, dose or round would need its own card.

## 14. Steps; each needs the user's approval

1. This card and the §2 permissions.
2. The build (**approved 2026-11-18**):
   - `htb_mix.py`: the fit config, `fit_mixed`, the launch-time overlap and E1–E3 checks, gate M1–M6 and the
     S-reproduction check.
   - Export glue, the match driver and the sign-test analyser, in new files.
   - The four gates, closed, with model-free tests.
   - Returned once for launch review.
3. The fit with its gate (CPU, ceiling 11,700 s).
4. On ELIGIBLE: export and parity.
5. On PARITY_PASS: the match smoke, then the match only on LAUNCH_OK, then the analysis.

Steps 4–5 may be approved together if you prefer.

## 15. Prediction on record

Low confidence, stated before any build:
- **Fit gate: ELIGIBLE, about a 55% chance.** M6 (H-TB value MSE) is the likeliest failure, then M4.
  - H-OP validation: KL_C about 20–35% below KL_S, smaller than H-OP's 41% at twice the H-OP dose.
  - H-TB validation: KL within −5% to +5%; top-1 within ±0.01; value MSE +0% to +20%.
- **Match, if played: mean pair score about 0.51** (plausible range 0.49–0.54). C wins about 50–65% of decisive
  pairs. **PASS: about a 15% chance.**
  - Why low: H-OP's large imitation gain produced no detectable strength gain.
  - Nothing measured shows that H-OP's retention loss was the obstacle.

## 16. Ledger record (proposal fields; the long-form entries follow each judged run)

- **Test:** H-MX, mixed-data distillation (§6–§9).
  - Warm-start from the current student `0e1cb621…`, the current `hard`'s source.
  - 7,500 CPU updates under H-OP's recipe on a fixed 50/50 mix: 16 H-OP + 16 H-TB teacher-labelled rows per 32-row
    batch. The cached label sets are reused as written (manifests `b0ea21cb…` and `ec28e768…`, 120,000 rows each),
    with E1–E3 rows dropped.
  - One-time fit screens M1–M6 on both cached validation sets.
  - On ELIGIBLE: export, parity, and a fresh 400-pair colour-reversed match against `78c16a31c3dba37f` at 200
    simulations.
- **Hypothesis:** training on both distributions keeps more of the original coverage while learning from positions
  `hard` now reaches. The resulting challenger wins more decisive colour-reversed pairs than it loses against the
  current `hard`.
- **Why it is plausible:**
  - H-OP cut teacher KL by 41.0% on held-out self-play positions, while KL rose 21.8% (value MSE +63.1%) on the
    original validation set (entry 69). Its challenger scored 0.4982 over 272 pairs (entry 72).
  - Rehearsing old data is the standard remedy for that kind of interference. Whether it improves play is the
    question.
  - The link between H-OP's retention loss and its match result is unmeasured.
- **Prediction on record (§15, stated before any build):** gate ELIGIBLE, about 55%. Match mean pair score about
  0.51 (0.49–0.54), C winning 50–65% of decisive pairs. PASS about 15%.
- **Decision rule (§13):**
  - **Screens on H-OP validation:** M1 KL ≤ 0.90 × S, M2 top-1 > S, M3 value MSE ≤ S.
  - **Screens on H-TB validation:** M4 KL ≤ 1.05 × S, M5 top-1 ≥ S − 0.010, M6 value MSE ≤ 1.15 × S.
  - Any screen missed: STOP, and the line closes.
  - **Match:** PASS iff the exact one-sided sign test over 400 pairs rejects (P(Binomial(m, ½) ≥ W) ≤ 0.05) and the
    mean pair score exceeds 0.50.
  - **Power:** about 80% at q = 0.691 and d = 0.114 (planning model: about +70 Elo-equivalent).
  - **Compute guards:** 200 simulations and 200 visits on both sides; search-seconds ratio ≤ 1.10; identical graphs.
  - NON_PASS keeps the current `hard`, with no top-up.
- **Permissions recorded (§2):**
  - narrow #53/#87 exceptions;
  - #177/#178: the H-TB training labels as input, and the 24,000 H-TB validation rows as a one-time fit screen;
  - #205/#206: the H-OP labels in a fit with another row set, and the H-OP validation rows as a one-time fit screen;
  - #209: two H-OP match summary figures used for planning only.
- **What ran / result / match to prediction / conclusion:** to be written after each judged run.

## 17. Review decisions (2026-11-18, the user's review)

- **Q1, dose:** keep 7,500 updates (each H-OP row seen once). The dose confound with H-OP stays disclosed (§1).
- **Q2, H-TB screen rows:** use the cached 24,000 rows as a one-time fit screen, with the disclosure in §2 and §7. No
  fresh labelling.
- **Q3, match size:** 400 pairs (about 80% at the specified inputs; the minimum for ≥ 80% is 401).
- **Q4, pass rule:** the exact sign test, with its claim and assumptions stated (§9) and the mean condition as an
  additional guard.

## 18. What this card does not authorize

Building, loading any model, fitting, exporting, playing, opening any gate, labelling, deploying or pushing. Each needs
the user's approval (§14). It does not reopen #53, #87, #177, #178, #205, #206 or #209 beyond the permissions
requested in §2. It does not use H-OP's challenger, H-OP's match games, or any benchmark, G3 or Diagnostic O position
except as exclusions.
