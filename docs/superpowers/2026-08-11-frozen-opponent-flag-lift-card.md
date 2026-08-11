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
(so **mixed-model grouping finally occurs**), variable `active_size`, tree reuse, backpressure,
and games that end at different plies. None of it has met the GPU.

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
| artifacts | `logs/eval/composition_smoke.json`, `.exit` — refuse if either exists |
| both SHA-1s | verified before any evaluator is built |

**Pass requires all of:** exit `0`; **4 games generated**; **exactly 2 learner-as-red and 2
learner-as-black**; **only learner-to-move positions** in the buffer; **one observed inference
thread**; **both models served** with non-zero per-model telemetry; **at least one flush
containing more than one request** — the first real evidence of mixed-model grouping on the
device; no worker error; all workers exit `0`; server thread stopped.

**Any non-zero result is a stop.** In particular a SIGABRT here would mean the arbiter is fine
in isolation but breaks under real MCTS load, which is a finding, not a retry.

## Only then: the lift

If and only if the composition smoke passes, remove the startup refusal in `trainer.py` and
correct the `--help` text. That is a **small, separate diff** reviewed on its own, and it
authorizes nothing by itself.

**This card does NOT authorize training.** After the lift, a frozen-parent training run needs
its **own** countersigned card with its own seed interval, prediction and disposition — the
prediction from the aborted card (`0.47–0.51`, ~10%) is still on record and still untested.

## Prediction, on record

**The composition smoke passes.** The arbiter is proven on the device and the seam is proven in
tests; what remains is their conjunction. Moderate confidence, lower than for the arbiter smoke:
this is the first time variable-size batches, mixed-model grouping and tree reuse meet real
Metal together, and mixed grouping in particular has never executed on the device at all.

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
approved scope      : run the committed composition smoke ONCE; then, only if it
                      passes, commit the refusal removal as a separate reviewed
                      diff. No training, no evaluation, no seed interval.
```

**Conditions:**

- A failing composition smoke is a **stop**. The refusal stays.
- The lift diff is reviewed separately from the smoke result.
- No outcome of this card authorizes a training run.
