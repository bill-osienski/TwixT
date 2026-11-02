# Combination policy: second policy-improvement cycle (amendment, draft for review, 2026-11-02)

**Status:** design only, not frozen and not pinned. It authorizes no build, seed procedure, model loading, generation,
training, evaluation, deployment or push.

This amends `docs/superpowers/2026-10-28-combination-policy-pilot-card.md` and
`docs/superpowers/2026-10-30-combination-policy-generation-amendment.md` for one new experiment. Those files stay
unchanged. The first pilot's NON_PASS, head and evidence also stay unchanged. Whatever this amendment does not change
still applies. The implementation plan and cost estimate are in
`docs/superpowers/plans/2026-11-02-combination-policy-second-cycle-plan.md`.

## 1. Question and hypothesis

**Question:** does one more self-play and policy-learning cycle produce an agent stronger than the frozen proven
combination at comparable compute?

**Hypothesis:** repeating policy improvement helps. This does not assume the first head is already stronger; that head
learned its targets but did not establish stronger play. Using it as the teacher is an explicit experimental choice,
and its changed priors may improve the next targets or reinforce mistakes.

**The specific new improvement (#140):** halving searches only the teacher's 16 highest-prior moves, and the target
leaves every other move at its prior. The first corpus therefore holds no search evidence about moves outside the
parent's candidates, and refitting it could not add any. A teacher with different priors searches a different set of
candidates, so its corpus can contain evaluations the first cycle never made. Nothing yet shows that those evaluations
improve play.

**The only deliberate changes:** the teacher and the student's starting policy. The search, target formula, trainable
surface, optimizer recipe and evaluation criteria stay fixed.

## 2. Starting state and teacher

| Item | Binding |
|---|---|
| Weights-only checkpoint | `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` |
| Checkpoint SHA-1 | `209cf2d4fd24a48553d259dd71b4954867b9473e` |
| First-cycle head | `docs/superpowers/evidence/combination-policy-train/head.npz` |
| Head SHA-256 | `ead156e2eab2ebc26f96879edf9d5705c7150bb0a6f8beca74d7e63d6a4d4947` |
| Digests after loading | encoder `92498b02…`, policy head `f653aca2…`, value head `576391a7…` (the pilot's `TRAINED_HEAD`) |

Before any inference: load the weights, check the SHA-1, install arm R, restore every saved head array, and check the
three digests. A mismatch refuses or voids the run.

The student starts from these same teacher weights, on a fresh Adam. The learned residual branch is not reset, no
optimizer state is resumed, and the first head's file is never written to.

## 3. Generate one fresh corpus

**1,000 self-play games**, with the frozen teacher on both sides:

- 24×24, unchanged rules, 280-ply cap;
- mirror averaging, 200 simulations;
- ordinary opening search and readout through ply 19;
- unchanged root sequential halving from ply 20, playing `last_choice`;
- fresh roots and the existing opening randomness;
- eval mode and process-local `MLX_ENABLE_TF32=0`.

The run uses **18 CPU workers, one GPU inference server and a batch cap of 14**. These are starting settings, not
claimed optima. Slot `g` plays seed `BASE + g`, independent of worker scheduling, with the existing four RNG streams.

Rows cover ply 20 onward, using the unchanged halving target formula, and record the teacher's priors, the targets and
the existing provenance fields. Games with `slot % 10 == 9` are held out, fixed before generation. There is no mirror
duplication.

Only this corpus is used for fitting. The first corpus and the training, screen, strength and VOID-run games stay
excluded. These games are never strength evidence, and their winners and reasons are not read outside a reviewed step.

**A known difference in the corpus:** the first head was fitted only on rows from ply 20 onward, yet it also changes
earlier priors (the screen's probe found a largest change of 0.114 at the empty board). The teacher's openings
therefore use priors never fitted there, and the corpus's positions will shift from the first corpus's. This is
reported through the existing summary and duplicate counts; it is not a gate.

## 4. Fit the second-cycle student

Fitting runs entirely on the **CPU**, in eval mode. It trains only the existing **8,840 policy-head parameters**. The
encoder, value head and all BatchNorm running statistics stay frozen.

The first-cycle recipe is unchanged:

- 3,000 updates, minibatch 64;
- fresh Adam, learning rate `0.001`, betas `(0.9, 0.999)`, epsilon `1e-8`, bias correction disabled;
- L2 coefficient `0.0001`;
- the existing minibatch-order seed and permutation procedure;
- only the final state is eligible for evaluation.

The prediction remains the **mean of the two mirrored probability distributions**. Initial KL, training loss and
held-out CE all use that one objective. Each corpus file is verified against its committed hash, and its stored targets
are recomputed, before fitting.

## 5. Learning checks

| Condition | Result |
|---|---|
| Initial training-row KL below `0.01` nats | STOP |
| Non-finite loss or gradient | STOP |
| Any change outside the permitted surface | STOP |
| Encoder or value-head digest changed, or held-out CPU value outputs not bitwise preserved | STOP |
| Final held-out CE not strictly below its initial value | STOP |

The CPU-versus-generation prior discrepancy is reported descriptively. It is not a KL error bound.

ELIGIBLE permits proposing evaluation. It does not authorize games or establish stronger play.

## 6. Evaluate against the proven reference

The final second-cycle candidate plays the **original proven combination**, not the first trained head. Both sides use
mirror averaging, halving from ply 20 and 200 simulations.

1. **Screen:** 32 colour-reversed pairs on fresh seeds. STOP below a mean pair score of `0.40`, otherwise CONTINUE. No
   strength claim either way.
2. **Strength, if separately approved:** 509 fresh colour-reversed pairs. PASS only if
   `mean pair score − sqrt(ln(40) / 1018) > 0.5`, the network-evaluation ratio is ≤ `1.05` and the search-seconds ratio
   is ≤ `1.10`, with ratios over whole-run totals.

There is no interim outcome inspection, checkpoint selection, or pooling with earlier matches. A PASS would establish
that this candidate beats the proven reference under this protocol. It would not isolate whether the second cycle
improved on the first head.

## 7. Runtime and incomplete runs

The experiment reuses the reviewed generation, training, match, wrapper and seed machinery, with its own evidence paths
and closed gates. Each ceiling is proposed from completed measurements before its runtime approval:

- **Generation:** the six-hour cap (TERM 21,590 s, KILL 21,600 s, observation through 21,610 s).
- **CPU fitting:** the existing timing formula on the new corpus's actual counts, × 1.5, rounded up to the minute.
- **Screen:** completed search timings and the new generation's move counts, × 1.5, rounded up to the minute.
- **Strength:** the completed new screen scaled to 1,018 games, × 1.5, rounded up to the minute.

TERM and KILL enforcement, bounded cleanup, gate restoration, BOINC checks and the required preflight checks stay.
Fresh seed blocks follow the existing survey, proof, registration and closing procedures.

An interrupted or timed-out run is incomplete and receives no score. Its progress, cause and continuation cost return
for assessment, with no automatic retry.

## 8. Additions to the reviewed specification

These come from auditing the code, and they do not change the search, target, recipe or criteria.

1. **A teacher-reach probe at generation.** Generation serves the network through the uncompiled evaluator, while only
   the compiled path has played a trained head. Both call `forward_padded`, so I expect the head to be reached.
   However, the prior discrepancy is descriptive and could not stop a corpus generated without the teacher. Before any
   worker starts, the run compares the teacher's and the parent's opening priors, and it is VOID if they are equal.
2. **The fit's start check changes.** The first cycle checked that arm R started bit-equal to the parent, which cannot
   hold here. It is replaced by checking that the student's three digests are the teacher's. The value baseline stays
   the parent's held-out CPU outputs, taken before the head is installed.
3. **STOP and VOID as built.** A gradient outside the surface is currently a VOID, and a changed array is a STOP. Both
   admit no head, so the code stays as it is.

## 9. Exceptions needing explicit approval

- **#66 and #107:** the exception for reusing `ResidualPolicyHead` covers the first pilot only and says not to extend
  it. This experiment needs a new narrow exception, again with no capacity claim.
- **#132, #139 and #140:** these bar further training and new seed procedures without the user's approval. Approval
  here would cover this one cycle, step by step. The first head's own confirmation stays unfunded.
- **Card §8.9 and §9:** the first pilot was one round, and iterative training was not requested. This amendment is that
  separate request.
- **#49:** the closed family trained the whole network with ordinary self-play targets. This keeps the card §2
  distinction: the encoder and value path do not move, and the target is the halving target.

## 10. Prediction proposed for the record

Directional only, with low confidence and no magnitude: the held-out CE falls, and the candidate scores above 0.5
against the proven reference.

## 11. Completion

This funds **one additional cycle only**. A completed NON_PASS or scientific STOP ends the experiment, and another
cycle needs a new justification and decision. Results go into dated, append-only ledger entries, with the hypothesis,
bindings, budgets and approvals.
