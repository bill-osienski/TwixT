# Frozen-Opponent Flag Lift — Composition Card

**Date:** 2026-08-11 · **Status:** DRAFT, not countersigned · **Scope: one tiny real-GPU
self-play composition smoke, and — only if it passes — removal of the
`--frozen-opponent-checkpoint` startup refusal. NO TRAINING.**

**Durable prior: `9c5847a0`** — the closed arbiter card
(`docs/superpowers/2026-08-11-inference-arbiter-refactor-card.md`, real-GPU smoke PASS at
execution commit `7fb7f70b`).

## What is already established, and what is not

The arbiter smoke proved, on the device: one server thread serving two different checkpoints
across 402 GPU calls, **exactly one observed inference thread**, correct per-model routing
against an independent oracle, and exact per-model telemetry.

**It proved nothing about the composition.** Those 402 calls were synthetic requests issued by a
purpose-built harness. The path that actually matters —
`run_parallel_selfplay` → `self_play_worker` → dual-root `play_game` → MCTS → two
`RemoteEvaluator`s → the arbiter — **has only ever run with stub evaluators**. Stubs return
constants and never touch Metal; per #50's standing lesson, that is not evidence about a
device-level contract.

Between the arbiter and a training run sits real MCTS: variable batch sizes below the row cap
(so **mixed-model grouping can finally occur**), tree reuse, backpressure, and games that end at
different plies. None of it has met the GPU. (`active_size` is **not** among them — this dose
fixes it at 24.)

## The gate: a tiny real-GPU composition smoke

**Real `run_parallel_selfplay` with a real frozen opponent, at the smallest dose that exercises
the real path.** Not synthetic requests — actual games.

| item | value |
|---|---|
| learner | `calib020_0001`, sha1 `209cf2d4fd24a48553d259dd71b4954867b9473e` |
| frozen opponent | the **same** checkpoint, as in iteration 1 of the real recipe |
| games | **4** (even, so the colour split is exact) |
| workers | 2 |
| board / sims | `active_size=24`, **32 simulations** — enough for real tree reuse, small enough to be minutes |
| `max_batch_rows` / `flush_ms` | 14 / 2, as production |
| max moves | 60, to bound the run |
| master RNG seed | **`20260812`**, pinned |
| `TWIXT_MIRROR_PROB` | **`0.0`**, set in the environment **before** `self_play` is imported (it is read at import time) |
| artifacts | `logs/eval/composition_smoke.json`, `.exit` — refuse if either exists |
| both SHA-1s | verified before any evaluator is built |

**Mixed-model grouping must be measured, not inferred.** "A flush with more than one request"
proves nothing: both requests could be the learner's. The smoke instruments the **pending flush
before grouping** and reports two separate counters — `multi_request_flushes` (any flush with
≥2 requests) and **`mixed_model_flushes`** (a flush containing **two distinct model ids**).
Only the second is evidence.

**Pass requires all of:** exit `0`; **4 games generated**; **exactly 2 learner-as-red and 2
learner-as-black**; **only learner-to-move positions**; **one observed inference thread**;
**both models served** with non-zero per-model telemetry; **`mixed_model_flushes >= 1`**; no
worker error; all workers exit `0`; server thread stopped.

**Learner-only and the colour split are checked on the streamed chunks**, non-vacuously: the
smoke passes a recording buffer to `run_parallel_selfplay`, so it sees each chunk as the worker
sends it. **Every chunk must contain exactly one `to_move` colour**, and **exactly four
game-start chunks** — those beginning at learner ply `0` or `1`, depending on who moved first —
must exist, **two of each colour**. Plies increase within a game, so a later chunk cannot
masquerade as a game start.

## Three outcomes, distinguished

| outcome | reading | consequence |
|---|---|---|
| **PASS** (all conditions, `mixed_model_flushes >= 1`) | the composition works on the device | authorizes **drafting** the refusal-removal diff for separate review — nothing more |
| **NO EXPOSURE / STOP** (everything else passes but `mixed_model_flushes == 0`) | scheduling never put both models in one pending flush, so the central condition **was not exercised**. **This is not an arbiter failure and not a Metal failure.** | **Refusal stays.** Re-scope the exposure — timing, request rate, worker count — under a new authorization. Do not reinterpret an unexercised condition as a passed one. |
| **FAIL** (SIGABRT, wrong routed results, worker error, non-zero exit) | a genuine failure of the composition | **Stop for redesign**, not patch-and-retry. Refusal stays. |

## Only then: the lift — under its own authorization

A PASS authorizes **drafting** the refusal removal in `trainer.py` and the `--help` correction,
**for separate review**. It does **not** authorize committing that change under this signature:
the diff does not exist yet, and this card will not pre-authorize unseen code. The lift lands
only after its own review, against a closed version of this card.

**This card does NOT authorize training.** After the lift, a frozen-parent training run needs
its **own** countersigned card with its own seed interval, prediction and disposition — the
prediction from the aborted card (`0.47–0.51`, ~10%) is still on record and still untested.

## Prediction, on record

**The composition smoke passes.** The arbiter is proven on the device and the seam is proven in
tests; what remains is their conjunction. Moderate confidence, lower than for the arbiter smoke:
this is the first time variable-size batches, mixed-model grouping and tree reuse meet real
Metal together, and mixed grouping in particular has never executed on the device at all.

**The likeliest non-pass is NO EXPOSURE, not failure.** Four games at 32 simulations across two
workers may simply never queue both models into one pending flush — small doses make thin
traffic. That is why it is a distinct outcome with its own consequence rather than being folded
into either success or breakage.

## Explicitly NOT authorized

- Any training run, warmup, evaluation, checkpoint or seed-interval reservation.
- Removing the refusal before the composition smoke passes.
- Any change to the dual-root game seam or the arbiter, beyond what a smoke failure would force
  — and such a failure is a stop for redesign, not a patch-and-retry.

---

## Countersignature

Execution is authorized only when this block is filled in, committed, and pushed.
**Unsigned, this card authorizes nothing.** The composition smoke script must be written and
committed **unrun** before signature, as the previous two were.

```
authorizer          : ____________________
timestamp (UTC)     : ____________________
authorization basis : ____________________   # the reviewed commit this signature approves
execution commit    : the commit containing this completed countersignature block
approved scope      : ONE execution of the committed composition smoke script,
                      and reporting it against the three-outcome table.
                      NOTHING ELSE. A PASS authorizes only DRAFTING the
                      refusal-removal diff for separate review; it does not
                      authorize committing it under this signature. No training,
                      no evaluation, no seed interval.
```

**Conditions:**

- A failing composition smoke is a **stop**. The refusal stays.
- `mixed_model_flushes == 0` is **NO EXPOSURE**, not a pass and not a failure. The refusal stays.
- The lift diff is written, reviewed and authorized **separately** from this signature.
- No outcome of this card authorizes a training run.
