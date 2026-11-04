# Source comparison and next-hypothesis proposal (2026-11-03)

**Status:** PROPOSAL for independent review. Paper only. Nothing is built, loaded, seeded, played or trained, and
nothing is authorized. This is a source review, not an experiment or a result.
**Reference to beat:** the original proven combination: `calib020_0001` (SHA-1 `209cf2d4…`) with mirror averaging,
200 simulations, ordinary search through ply 19, and root sequential halving from ply 20 (entry 39 addenda, #105–#106).
**Builds on:** the first-pass source note at the end of the ledger, dated 2026-11-03. Do-not-repeat #164–#165 stand.

Source identities used throughout. Git blob identity was read from the GitHub API listings. No file was executed and
no model artifact was downloaded.

- **U** = upstream `BonyJordan/twixtbot` @ `f0b8d10f444f46787cf4cd03f8d67fc68d1f9ddb`
- **UI** = evaluated `stevens68/twixtbot-ui` @ `83749f230a0bae1766b46a05bfde0ed87f0a9a0a`
- **Ours** = this repo @ `bc37487e`

---

## 1. Verified differences

| Axis | Ours | Upstream source | Reference |
|---|---|---|---|
| Inputs | 30 planes. Pegs, 8+8 link directions, to-move, 4 edge planes, phase, 6 hand-made edge-connectivity planes. The position is canonicalized by a 90° rotation for black. | 2 peg planes, 8 link planes and 2 **coordinate-ramp** planes (x/S, y/S), plus an optional recent-moves plane. The position is swapped/transposed for black. No hand-made connectivity planes. | Ours `network.py:32-49`; U `naf.py:197-224`, `naf.py:447-457`, `naf.py:49-59` |
| First layer | One 3×3 convolution over all inputs. | Separate projections summed: locations 1×1, pegs 5×5, links 4×4. | Ours `network.py:269`; U `mkbig.py:35-51` |
| Trunk | 6 residual blocks, 128 channels, 3×3 kernels, ReLU, activation after the add. About 1.88 M parameters. | Builder defaults: 12 blocks, 40 filters, 5×5 kernels. The activation is **`abs(x)`**, applied *before* the add. About 1.08 M weights at the defaults, by arithmetic from the source. | Ours `network.py:230-290`; U `mkbig.py:12-13`, `:26-27`, `:53-65` |
| Theoretical receptive field of a policy logit | ±13 cells: one 3×3 convolution plus twelve more 3×3s, then a 1×1 head. On a 24-cell board, an edge cell cannot see the far side. | ±50 cells at the defaults; still ±26 with 6 blocks. | Arithmetic from the kernel sizes above |
| Policy head | Per cell: 1×1 to 2 channels → BN → ReLU → MLP 2→64→1, evaluated only at legal moves (519 parameters). | Per cell: 1×1 to 2 channels → BN → abs → 1×1 2→1, reshaped over S·(S−2) cells. **Narrower than ours.** | Ours `network.py:293-385`; U `mkbig.py:83-92` |
| Value head | Masked global average and max pooling → FC 256 → soft clamp → tanh. Translation-invariant summary. | `pwin_reductions` (default 2) strided 5×5 VALID convolutions → flatten → dense 80 → BN → abs → dense → tanh (or a 3-way softmax). **Spatially aware.** | Ours `network.py:387-476`; U `mkbig.py:94-115` |
| Loss | Visit-count cross-entropy + value MSE + 1e-4·L2, over all parameters, in ordinary training. | Cross-entropy + MSE (or 3-way cross-entropy) + 1e-4·L2, over all trainable variables; Adam or SGD. | Ours `trainer.py:1324-1343`; U `mkbig.py:142-170` |
| Policy target | Visit counts normalized, i.e. temperature 1. | Default temperature 0.5: **N² normalized** (sharper). At temperature 0: one-hot + ε. | Ours `trainer.py:1161-1179`; U `train.py:25`, `:170-186` |
| Symmetry augmentation | A left-right mirror with probability 0.5 (`train.py:264` default). A defect in black's edge planes was fixed in `0c417882` (entry 9). | One of 4 symmetries drawn uniformly per row: identity, h-flip, v-flip, both. | Ours `self_play.py:1133-1150`; U `train.py:145-149`, `naf.py:125-131` |
| Batch / learning rate | 64 / 1e-3 | Trainer defaults: 256 / 0.01 | Ours `train.py:63-66`, `:129-132`; U `train.py:21-23` |
| Recent fitting (combination pilots) | Policy head only. Encoder, value head and BN statistics frozen. | Not applicable. The upstream trainer optimizes the whole network. | Entries 43–51; U `mkbig.py:160-170` |

The two-channel reduction is common to both policy heads, and upstream's is narrower than ours, so that reduction
cannot by itself explain the gap. None of the rows above is an established cause of strength.

## 2. The evaluated external model: known, inferred and unknown

**Established from source and repository metadata:**
- **The weights we played against are upstream's own published "best" network, byte for byte.** Our harness pinned UI
  `model/pb` (`twixtbot_g3_schedule.py:55-61`). In the UI tree, `model/pb/variables/variables.data-00000-of-00001` and
  `.index` have the same Git blob IDs as U `models/six-917000.data-00000-of-00001` (`310672e9…`, 10,270,828 bytes) and
  `.index` (`10123d89…`). UI `model/meta/*` is identical to U `models/*`, `.meta` included (`ab695734…`).
- The UI's conversion to TensorFlow 2 changed graph metadata only. `convert/meta2pbtxt.py` re-exports the checkpoint
  and asks for a manual edit of 94 `FusedBatchNormGrad` nodes; the variable bytes are unchanged. UI README:65 says the
  network was taken from twixtbot in December 2020.
- U README:20-21 calls `six-917000` "my best one", which "can probably be improved with more training and/or a bigger
  net".
- **The network was trained with own-link crossings allowed** (UI README:67). We play with them forbidden.
- The trainer imports whatever graph the `.meta` holds (U `train.py:51`), so the construction arguments live in the
  model, not in the trainer. The model arrived in U's first commit (`fa54da43`, 2020-06-28). The builder's
  `pwin_*` and `init_stdev` options were added later that same day (`4994d0b1`), so the builder **at this revision may
  not be the one that built the model**.

**Inferred, not established:**
- **About 917,000 optimizer steps.** The trainer saves `model_name-<global_step>` (U `train.py:295`, `:301`), so the
  suffix 917000 is the global step, assuming the counter was never reset. The batch size is unknown; the default is 256.
- **Our lineage, for scale:** about 409 iterations × 100 games at 400 simulations, batch 64. The per-row optimizer
  steps were not recorded (`train_steps_per_iteration` is blank or `None` in the surviving metrics and sidecars). The
  table default at 24×24 is 160 (`trainer.py:308-315`), and the first staged row shows 200. That puts the lineage at
  roughly **65–82 K steps**: about 11–14× fewer optimizer steps than upstream, and an unknown ratio of samples.
- **The builder defaults are probably not this model's configuration.** If the checkpoint stores raw float32 tensors,
  the defaults predict 4.32 MB of data under SGD or 12.95 MB under Adam (weights plus slots), against 10.27 MB actual.
  A brute-force scan over blocks 1–40, filters 8–256 and the head options found no configuration that reproduces
  10,270,828 bytes exactly. This is weak evidence: the serialization assumption is unverified.

**Unknown:** the blocks, filters, value-head options and recent-moves flag of `six-917000`; its number of self-play
games; its search trials per move; its noise schedule (U README describes 25% random noise in self-play); and its
learning-rate history.

**The one cheap closure, not taken:** the 8,323-byte `variables.index` lists every variable's name and shape, and
would settle the architecture in a single metadata read. It is a model artifact, so it was not fetched. It is
**not needed** for the proposal in §4. It becomes useful only if that proposal's fit check fails.

## 3. What was tested, what is unresolved, what needs reopening

| Hypothesis family | Status | Record |
|---|---|---|
| Ordinary self-play continuation from `calib020_0001`, cold or warm buffer | **Tested, closed**: 0.31375 and 0.4325 | #48–#49 |
| Frozen-parent opposition | **Tested, closed**: 0.46625 | #51 |
| Head-only imitation of twixtbot's chosen moves (192 positions, 519 parameters) | **Tested.** Held-out CE 4.566 → 3.211 passed; the 32-game screen was inconclusive (0.375, [0.035, 0.715]). Do not extend or rescue it. | #53 |
| Search distillation: Stage B, and D2 (encoder + policy on our own search targets, value frozen) | **Tested, closed.** D2 stopped on value drift even under TF32=0. | #54, #69, #87 |
| Head-only learning from the combination's halving targets, two cycles | **Tested, closed**: NON_PASS 0.5309 and 0.5201, separate and unpooled. The recipe stops. | #138–#139, #164–#165 |
| Policy-head capacity (residual head) | **Unresolved**: the planted check was NOT QUALIFIED; the proposal is shelved | #65–#66 |
| Dual-encoder value-preserving branch | **Rejected on paper for cost**; not a result | #88–#89 |
| Trunk architecture: depth, width, kernel, receptive field, activation, spatial value head | **Never tested.** Every surviving recorded run is 128×6 (capacity C0). C0 closed as a feasibility NO-GO; C1 was refused. | C0 card 2026-08-24 |
| Training scale (optimizer steps or samples per position) | **Never tested directly** | — |
| Sharpened policy targets (temperature 0.5) | **Never tested** | — |
| Mirror-defect repair alone | **Not justified**; the fix carries into any future training | #61 |

**Requires explicit reopening:**
- Any use of twixtbot as a training source. #53 requires a new card, fresh seed accounting, and decisions on its
  seed-selection rule and cleanup exit status. Both appear to have been superseded by the 2027 band method and the
  exit-125 supervisor, but the card must confirm that.
- Any distillation, under #87's requirement of a new written design.
- Capacity, under #66, only with a pre-fit KL separation.
- Dual encoders, under #89.

T1j stays evaluation-only and is not used in anything proposed here.

## 4. Best next hypothesis

**H-TB: the 128×6 network, trained as a whole to match twixtbot's raw-network policy *and* value on a large position
set, plays stronger than `calib020_0001` inside the unchanged proven combination.**

**The specific change:** whole-network distillation. The teacher is UI `model/pb` with no search. The policy target is
the teacher's softmax restricted to our legal moves and renormalized. The value target is the teacher's value. The
student is warm-started from `calib020_0001` with every parameter trainable, including the value head, and uses the
mirror fix. Search, evaluator and agent stay exactly as in the proven combination. The student has the same
architecture as the reference, so inference cost is matched by construction.

**Why it could improve play:**
1. **The largest headroom on record.** twixtbot's raw network, with **no search** (`trials=0`), won every G3 game
   against `calib020_0001`, which did search (#52). That is an ordering, not an Elo figure, and not a rules-matched
   result, but no internal line has come close to that margin.
2. **The signal is learnable even at a tiny dose.** 519 parameters fitted on 192 chosen-move labels cut held-out CE by
   1.36 nats (#53). The proposal raises the dose by 3+ orders of magnitude: the full network, soft targets for both
   heads, about 10⁵–10⁶ positions.
3. **It sidesteps D2's failure mode.** D2 stopped because training the encoder moved the *frozen* value head. Here the
   value head has its own teacher target, so drift from the incumbent's value is the goal, not a failure.
4. **It answers the question this source review cannot.** It separates representation from training scale. If our
   architecture absorbs the teacher's function, the gap was training, and we gain a stronger network directly. If it
   cannot, that is direct evidence for the architecture rows of §1. A distilled student of another architecture is then
   a cheap way to compare architectures, because no self-play is needed.

**Positions:** the existing eligible corpus (capacity C0): 38,770 games and 2.1 M positions, content-pinned by
`eligible.digest dc7713c6…`, with a frozen 80/10/10 split by game. It was never used as held-out scoring data
(C1 was refused). No new games and no seeds are needed for labelling, because raw-network labels are deterministic.
Its generator provenance is limited, but it supplies only positions; the teacher supplies the labels.

**Known risks:**
- **Teacher rules mismatch.** Its evaluations assume own-link crossings are possible. UI README:67 calls the effect
  small "in most cases", which is a claim, not a measurement.
- **Off-distribution positions.** The corpus comes from our own weaker play.
- **Training device.** The GPU path is not qualified for general training (#81–#82). CPU fitting is trusted but has
  not been timed for the whole network.

## 5. Falsification, cost and strength test

**Stage 1 — fit check (CPU or a qualified device; no games).** Train on about 300 K positions from the corpus's train
games. Score on about 30 K positions from its validation games, each position labelled once. Freeze the criteria
before labelling:
- F1: the mean KL(teacher‖student) is ≤ 0.5 × the mean KL(teacher‖`calib020_0001`).
- F2: top-1 agreement with the teacher is strictly higher than `calib020_0001`'s.
- F3: the value MSE against the teacher is ≤ 0.5 × `calib020_0001`'s.

Any failure is a **STOP**. A STOP is evidence, not proof, of a representation or fit limit; it would then justify the
`variables.index` read and an architecture-student design. The 0.5 factors are proposed for review; they are not
derived.

**Stage 2 — screen.** 32 colour-reversed pairs: the student in the proven combination against the reference. STOP if
the mean pair score is below 0.50. At a true score of 0.65, the chance of a false stop is about 1–5%. The screen's
games are never pooled with Stage 3.

**Stage 3 — strength test, sized for a worthwhile gain.** The target is a **mean pair score of 0.65** (about +108
Elo). A distilled network that gains only a few points would not justify depending on a rules-mismatched external
teacher. The 509-pair design is not reused.
- **Size:** 110 fresh pairs (220 games).
- **PASS rule:** a one-sided Hoeffding bound at α = 0.05: PASS if mean − √(ln 20 / 220) > 0.5, which means a mean
  above **0.6167**.
- **False PASS:** ≤ 5% at a true 0.5, assuming independent pairs and completion.
- **Power at a true 0.65:** about 76–84%, for a pair-score standard deviation between 0.5 and 0.35. At a true 0.60,
  power is only about 33%; the test is deliberately insensitive below +10 points.
- **Guards, unchanged:** evaluations ≤ 1.05× and search seconds ≤ 1.10×.
- **Conditions:** TF32=0, a fresh 2027-band block, and the idle and BOINC launch gates.

**Estimated total cost.** These are estimates, not funding, and each needs its own timing step before a ceiling is set.

| Item | Estimate | Note |
|---|---|---|
| Engineering | 2–3 review rounds | Reuse the `external_learning.py` teacher path (`:895`), the D2 trainer pieces and the existing match stack; add gates and tests |
| Dependency | Separate approval | Re-acquire the UI checkout and the TensorFlow runtime; the October checkout is absent |
| Labelling | ~1–2 h CPU | Unmeasured; about 330 K raw-network evaluations |
| Fit | ~1–2 h GPU-equivalent, or several hours on CPU | 600 K samples; the recorded 181 positions/s at 128×6 is a GPU-era figure |
| Screen | ~0.6 h | 64 games |
| Strength test | ~2.2 h | 220 games at the 35.3 s/game of entry 51 |
| **Total compute** | **~5–12 h** | About one 509-pair comparison |

**Prediction proposed for the record:** directional only, with low-to-moderate confidence: F1–F3 pass, and the
student scores above 0.5 against the reference. No calibrated probability is offered.

**Decisions this needs from the user before any build:**
1. Reopen twixtbot as a training source under a new card (#53).
2. Accept the distillation design under #87.
3. Approve the dependency re-acquisition.
4. Choose the training device.
5. Optionally, allow the one-file `variables.index` metadata read. It is not needed unless Stage 1 stops.

**If this is declined:** no other line on this page currently earns an experiment. Architecture changes trained from
scratch through self-play would cost on the order of 100 h of generation, and any result would be confounded with the
unknown upstream training scale.
