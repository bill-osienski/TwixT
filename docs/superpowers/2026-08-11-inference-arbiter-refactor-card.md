# Single Inference Arbiter — Refactor Card

**Date:** 2026-08-11 · **Status:** DRAFT, not countersigned · **Scope: one refactor of the
inference transport. No training, no evaluation, no seed interval, and no lifting of the
`--frozen-opponent-checkpoint` block.**

**Durable prior: `18eb7211`** — the closed single-arbiter feasibility probe
(`docs/superpowers/2026-08-10-single-arbiter-feasibility-card.md`, PASS/PASS at execution
commit `4808f324`). That probe established, at its dose, that **one thread can hold two
resident networks**. It established nothing about request routing or per-model batching, which
is exactly what this card builds.

## The problem this fixes

Do-not-repeat **#50**: two independent `InferenceServer` threads submitting concurrent work to
one Metal device abort the process (`exit 134`, 2026-08-10). Frozen-parent training needs two
networks served at once, so it needs a transport that has exactly **one** device owner.

## The change

**One `InferenceServer` thread owning the device, serving N networks.** Requests carry a model
id; the server routes each to the matching evaluator and batches **per model**, since distinct
networks cannot share a batch.

| file | change |
|---|---|
| `ipc_messages.py` | `InferenceRequest` gains `model_id` |
| `inference_server.py` | `evaluator` → `evaluators: Dict[str, evaluator]`; pending requests grouped by `model_id`; one flush per model |
| `remote_evaluator.py` | `RemoteEvaluator` carries a `model_id` and stamps every request |
| `trainer.py` | one server, one request queue; learner and opponent `RemoteEvaluator`s differ only by `model_id` |
| `self_play_worker.py` | second evaluator built from the **same** queue pair, different `model_id` |

**Deleted, not kept alongside:** the second server, the second request queue, the second
response-queue set, and their symmetric start/stop/join/cleanup. Two owners must become
unreachable, not merely unused.

## Non-negotiable invariants

1. **Exactly one thread ever calls a `LocalGPUEvaluator`.** Asserted by test, not by convention.
2. **Default path unchanged** — one model registered, deterministic behavioural equivalence,
   no extra queues, threads or RNG draws.
3. **Cross-model isolation:** a batch never mixes model ids, and a response never reaches a
   requester that asked a different model.
4. **Fail closed:** an unknown `model_id` raises; it never falls back to a default evaluator.
   The existing `server_error` / `worker_error` paths keep working for both models.

## Stop rule

Roughly **two working days and ~250 production lines**, counted across the five files. If
`inference_server.py`'s batching loop cannot absorb per-model grouping without a rewrite, stop
and re-scope rather than growing it.

## Tests required before any GPU work

- Single-thread invariant: one owner, asserted structurally and by instrumentation.
- Per-model routing: two models, interleaved requests, each response matches its own model's
  expected output; never crossed.
- Unknown model id raises rather than defaulting.
- Batching groups by model and never mixes ids in one flush.
- Default-off equivalence for the one-model case.
- The existing 2,865-test suite green, with the two-server tests **removed or rewritten** — they
  describe a design that no longer exists.

**Stub evaluators are insufficient on their own.** #50's standing lesson is that a stub-level
integration test says nothing about a device-level contract, so this card also requires a
**small real-GPU round trip**: two real networks through the arbiter via real workers, enough
calls to be meaningful, verified the way the probe verified — per-call shapes, finiteness,
digest stability, digests agreeing where the weights are identical.

## What this card does NOT authorize

- Lifting the `--frozen-opponent-checkpoint` startup refusal. That is a **separate** card, after
  this one lands and its real-GPU round trip passes.
- Any warmup, training run, evaluation, checkpoint, or seed-interval reservation.
- Re-running the aborted frozen-parent recipe in any form.
- Changing the dual-root game seam, which is already tested and unaffected.

## Prediction, on record

**The refactor lands and its real-GPU round trip passes.** The probe removed the main doubt —
one thread can hold two resident networks — and what remains is ordinary plumbing. Moderate
confidence: per-model batching halves effective batch size when both models are hot, so the
throughput cost is real and unmeasured, and the probe's "no observed slowdown" was one sample at
a dose with no queueing at all.

---

## Countersignature

Execution is authorized only when this block is filled in, committed, and pushed.
**Unsigned, this card authorizes nothing.** The refactor may be implemented before signature;
**no real-GPU round trip may run before it.**

```
authorizer          : ____________________
timestamp (UTC)     : ____________________
authorization basis : ____________________   # the reviewed commit this signature approves
execution commit    : the commit containing this completed countersignature block
approved scope      : implement the arbiter across the five files named above,
                      pass the required tests, and run the small real-GPU round
                      trip once. Nothing else.
```

**Conditions:**

- The two-server code paths are **deleted**, not left dormant behind a flag.
- A failing real-GPU round trip is a **stop**, not a retry or a tuning exercise.
- No result of this card authorizes training. The flag stays blocked until its own card.
