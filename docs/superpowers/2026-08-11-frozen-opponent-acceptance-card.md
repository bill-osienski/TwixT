# Frozen-Opponent Implementation — Acceptance Card

**Date:** 2026-08-11 · **Status:** DRAFT, not countersigned · **Scope: DOCUMENTATION ONLY.**
Accepts the implementation at `95987ffa` as the **enabled baseline**. **No command execution,
no training, no evaluation, no seed reservation.**

This card runs nothing. Its entire effect is to record that the frozen-opponent implementation
is accepted, so that later work can cite one acceptance rather than re-deriving the chain.

## Status of the flag, stated precisely

`--frozen-opponent-checkpoint` is **enabled at HEAD**. The startup refusal was removed in
`95987ffa`; the flag is part of the baseline. **What is not authorized is its USE** — no
training run may invoke it without its own countersigned card.

That distinction has been muddled before and is worth keeping sharp: the *lift* is done; the
*run* is not authorized.

## What is accepted

| element | evidence |
|---|---|
| single Metal-owning inference arbiter, `model_id` routing, `(worker_id, model_id)` response queues, per-model batching and telemetry | `9c5847a0` — arbiter smoke closeout |
| dual-root `play_game` seam, learner-only training rows, id-derived colour split, worker fail-loud | shipped and tested; suite green throughout |
| the composition: real MCTS over the arbiter, mixed-model grouping on the device | `d312caa2` — composition smoke closeout |
| flag enabled with fail-fast recipe validation | `95987ffa` — validations precede Metal configuration, directory creation and both networks |

**Suite at acceptance: 2,881 passed / 4 skipped / 53 deselected / 0 failed**, measured against
the exact pushed tree at `95987ffa`.

## The evidence chain, in order

1. **`#50`** — two independent `InferenceServer` threads on one Metal device abort the process
   (`exit 134`, 2026-08-10, at the first line of iteration 1).
2. **Feasibility probe** (`18eb7211`) — one thread can hold two resident networks: control and
   treatment both passed, 400 calls each.
3. **Arbiter smoke** (`9c5847a0`) — 402 GPU calls, **one** observed inference thread, correct
   per-model routing against an **independent** oracle using two *different* checkpoints, exact
   telemetry.
4. **Composition smoke** (`d312caa2`) — real self-play over the arbiter: 4/4 games, one
   inference thread, learner-only rows, exact 2/2 colour split, and **38 mixed-model flushes**,
   the first time mixed grouping executed on the device.
5. **Lift** (`95987ffa`) — refusal removed, validations hoisted, both directions tested.

## What acceptance does NOT establish

- **Routing correctness under composition.** The composition smoke used one checkpoint twice,
  so a crossed response would have been invisible there. Routing rests on step 3's
  two-checkpoint oracle.
- **Load beyond the smoke doses.** 402 synthetic calls and 4 games at 32 simulations. A
  five-iteration training run is orders of magnitude larger, and nothing has probed sustained
  contention, long-run memory behaviour or throughput cost.
- **Any strength claim.** The frozen-parent hypothesis is **untested**: its preregistered
  prediction of an aggregate score around **`0.47–0.51`** with roughly a **10%** chance of
  clearing the promotion bar has stood since the aborted run and remains unexamined.
- **`#50` is unchanged.** Two independent servers submitting concurrently remain forbidden. The
  arbiter satisfies the prohibition; it does not repeal it.

## Constraints inherited by any successor training card

- **`fp5` paths are spent** — `checkpoints/alphazero-v2-fp5-from-calib020`,
  `logs/selfplay/fp5_from_calib020`, `logs/eval/fp5_train.*`. New paths required.
- **Training seed `20260810` is spent.** A new `--seed` is required.
- **Evaluation interval 6 `[202609788, 202610188)` is RELEASED UNUSED** and may be **explicitly
  re-reserved** at countersignature — it drew zero seeds, because the evaluation never started.
- **The `0.47–0.51` / ~10% prediction is preserved as preregistered** and must not be restated,
  softened or re-derived after seeing any result.

## No ledger entries

No results-table row and no seed-ledger entry. Acceptance of an implementation is not a
scientific result: no strength claim, no checkpoint, no evaluated games, no interval consumed.

---

## Countersignature

**Unsigned, this card authorizes nothing.** Signed, it authorizes **only** the recording of
acceptance — there is no command to run.

```
authorizer          : ____________________
timestamp (UTC)     : ____________________
authorization basis : ____________________   # the reviewed commit this signature approves
execution commit    : the commit containing this completed countersignature block
approved scope      : record acceptance of the frozen-opponent implementation at
                      95987ffa as the enabled baseline. DOCUMENTATION ONLY --
                      no command execution, no training, no evaluation, no seed
                      reservation.
```

**Conditions:**

- Acceptance authorizes **no invocation** of `--frozen-opponent-checkpoint`.
- A frozen-parent training run requires its **own** countersigned card with new paths, a new
  training seed, an explicitly re-reserved evaluation interval, and its own disposition.
- `#50` remains in force unchanged.
