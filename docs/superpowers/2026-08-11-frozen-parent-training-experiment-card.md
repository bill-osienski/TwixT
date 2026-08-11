# Frozen-Parent Training — Experiment Card

**Date:** 2026-08-11 · **Status:** DRAFT, not countersigned · **Scope: one five-iteration
training run and one 400-game evaluation. Nothing else.**

**Prior:** `cf76982f` — the closed acceptance card, which accepted the frozen-opponent
implementation at `95987ffa` as the enabled baseline and explicitly left its **use**
unauthorized. This card authorizes that use, once.

This is the **third attempt at the same hypothesis**. The first (cold buffer) was rejected on
strength. The second (frozen parent) **aborted before iteration 1** on the two-server Metal
defect and produced no scientific result. The hypothesis has never been tested.

## Hypothesis

Short continuations degrade because the learner trains on its own drifting play: as it weakens,
so does its opposition, and errors compound. Pinning the opponent to the **frozen best parent**
holds opposition quality fixed, so the training signal stays anchored to a strong reference.

## Prediction, on record — PRESERVED, not restated

**Carried verbatim from the aborted card, where it was recorded before any implementation
existed and has never been examined:**

> **Central forecast: material recovery relative to `warm5`, but no promotion.** Expect the
> frozen endpoint to finish roughly equal to or slightly weaker than the parent, with an
> aggregate point score around **`0.47–0.51`**. Estimated chance of clearing the promotion bar
> is **about 10%**.
>
> Holding opposition strength fixed directly addresses self-play co-drift, so improvement over
> the descriptive `warm5` result of `0.4325` is plausible. However, the mechanism does not
> change the terminal outcome targets, policy-dominated optimization, short five-iteration
> horizon, or risk of specializing against one opponent. Therefore parity is more likely than a
> statistically significant gain.

**It must not be softened, re-derived or re-anchored after any result is seen.** Its value is
that it predates the implementation entirely.

## The frozen budget — unchanged from the aborted card

| choice | value |
|---|---|
| parent warmup | **500 games**, ordinary single-network path |
| games per iteration | **200** (learner-only filtering halves usable rows) |
| colour split | exactly **100 learner-as-red, 100 learner-as-black**, by game id |
| iterations | **5** |
| optimizer | **160 steps**/iteration, batch **64**, buffer **100,000** |
| simulations | **400**, both agents |
| game count | **fixed** — no adaptive target, no top-up |
| per-iteration reporting | actual learner positions added |

**Nothing about the dose changes.** Only the paths, the seed and the interval differ from the
aborted attempt — because those are spent, not because the recipe is being retuned. Retuning
after an infrastructure abort would silently convert a repeat into a new experiment.

## New paths and seeds — the old ones are spent

| item | value |
|---|---|
| checkpoint dir | `checkpoints/alphazero-v2-fp6-from-calib020` |
| games dir | `logs/selfplay/fp6_from_calib020` |
| training logs | `logs/eval/fp6_train.{stdout,exit}` |
| provenance | `logs/eval/fp6_candidate_provenance.txt` |
| evaluation | `logs/eval/fp6_vs_calib020.{json,stdout,exit}` + `_games.jsonl` |
| training seed | **`20260812`** — `20260810` is spent |
| evaluation interval | **`[202609788, 202610188)`** — interval 6, **RELEASED UNUSED**, to be **explicitly re-reserved** at countersignature |

Interval 6 drew **zero** seeds: the aborted run died before the evaluation started, and
`eval_checkpoint_match` is its only consumer. Re-reserving it is legitimate and must be
explicit in the ledger, flipping `RELEASED UNUSED` → `RESERVED`.

## Parent

```
checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors
sha1  209cf2d4fd24a48553d259dd71b4954867b9473e
```

Learner start, frozen opponent, and evaluation baseline are all this checkpoint.

## Frozen endpoint

**Only `checkpoints/alphazero-v2-fp6-from-calib020/model_iter_0005.safetensors` is evaluated.**
Iterations 1–4 are never evaluated, probed or selected from, under any result.

## Decision rule — aggregate only

**Success:** the candidate's overall **95% lower bound above 50%** over 400 games against
`calib020_0001`.

**No per-colour veto.** The 2026-08-10 erratum retired the absolute `0.50` per-colour rule:
equal agents need not score `0.50` in each colour, and a colour null estimated from the same
match is confounded with candidate strength. Per-colour figures are **reported and descriptive**.

## Frozen disposition

| outcome | disposition |
|---|---|
| **Bar not met** | Close frozen-opponent training. No dose change, no warmup-size change, no second opponent, no opponent pool, no extension. |
| **Bar met** | **Do not promote.** Run the **0379 generalization match** under its own authorization: beating the parent while failing 0379 means the learner exploited one opponent rather than becoming broadly stronger — the central risk of frozen-opponent training. |
| **Clear loss** | Close immediately. |

## Infrastructure risk, stated honestly

The aborted attempt died at the first line of iteration 1. The arbiter and composition smokes
(`9c5847a0`, `d312caa2`) now cover that failure mode, but **at doses orders of magnitude below
this run**: 402 synthetic calls and 4 games at 32 simulations, versus ~1,500 games at 400
simulations here. Sustained contention, long-run memory behaviour and throughput cost are
**unprobed**.

**A second infrastructure abort is a real possibility and is not a scientific result.** If it
happens, it is recorded as such — paths and training seed spent, interval consumed only if the
evaluation actually began — and the hypothesis remains untested for a third time.

## Cost

Warmup ≈ **1 h 12 m** (measured). Training self-play ≈ **3 h 35 m** interpolated from measured
like-for-like work, plus whatever the arbiter costs — per-model batching may reduce effective
batch size under balanced demand, and **that cost is unmeasured**. Evaluation ≈ **3 h 07 m**.
**Roughly 8 h, with the arbiter overhead unknown.** A planning estimate, **not a timeout**:
exceeding it triggers no retry, no parameter change and no abort.

## Deliberate omissions

- **No 64-game screen**, no A/B/C/D, no iterations 1–4, no checkpoint sweep, no analyzer.
- **No external strength anchor** — still conditional on this becoming an ongoing programme.

## A positive result is research evidence only

It promotes no checkpoint. The 0379 generalization match precedes any promotion discussion, and
adoption is a separate decision with its own record.

---

## Countersignature

**Unsigned, this card authorizes nothing.**

```
authorizer          : ____________________
timestamp (UTC)     : ____________________
authorization basis : ____________________   # the reviewed commit this signature approves
execution commit    : the commit containing this completed countersignature block
approved scope      : the exact FOUR command blocks (to be written into this card
                      before signature), unmodified:
                        1. training run              [GPU, state-changing]
                        2. provenance gate           [no GPU; writes ONE artifact]
                        3. evaluation match          [GPU, state-changing]
                        4. per-colour reporting      [read-only, descriptive]
```

**Conditions:**

- Both runs execute from the **execution commit** with a clean worktree.
- On signature, flip interval 6 `[202609788, 202610188)` from `RELEASED UNUSED` to `RESERVED`
  in the seed ledger, **in the same commit**.
- Approval covers **one** training run and **one** evaluation. No re-run, parameter change,
  extra iteration, or retry after failure — including after an infrastructure abort.
- The prediction above is frozen. Amend and re-sign **before** running, never after a result.
