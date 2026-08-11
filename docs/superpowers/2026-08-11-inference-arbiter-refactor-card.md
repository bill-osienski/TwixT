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
| `inference_server.py` | `evaluator` → `evaluators: Dict[str, evaluator]`; pending grouped by `model_id`; one flush per model; responses routed by `(worker_id, model_id)` |
| `remote_evaluator.py` | `RemoteEvaluator` carries a `model_id`, stamps every request, and keeps its **own** response queue |
| `trainer.py` | one server, **one** request queue, response queues keyed `(worker_id, model_id)` in a single routing map |
| `self_play_worker.py` | second evaluator on the **same request queue**, its **own** response queue, different `model_id` |

### Response identity — frozen, because the obvious design is broken

An earlier draft said both evaluators "share one queue pair." **That is a silent
cross-model-contamination bug**, not merely an ambiguity. Verified in the current code:
`InferenceRequest` carries `worker_id` + `request_id` but **no model id**;
`InferenceResponse` carries **only `request_id`**; the server routes by
`self.response_queues[req.worker_id]` (`inference_server.py:218`); and each `RemoteEvaluator`
starts `itertools.count(1)` with its **own** `_mailbox` (`remote_evaluator.py:35-36`). Two
evaluators on one response queue would both mint `request_id` 1, 2, 3…, and either could
dequeue the other's response, match it as its own, and **return the wrong model's priors and
values** — undetectably.

**Frozen design: one request queue, one server, one device owner — and DISTINCT response
queues addressed by `(worker_id, model_id)` in a single routing map.** Each evaluator owns its
queue, so its counter and mailbox are unambiguous by construction and `InferenceResponse` needs
no new field. Extra response queues **do not** violate the one-owner rule: they carry no device
work. The rejected alternative — a model id on responses plus a genuinely shared counter and
mailbox — is more moving parts for the same guarantee.

**Deleted, not kept alongside:** the second server, the second request queue, and the separate
`opp_response_queues` **collection**, together with their symmetric start/stop/join/cleanup.
Two owners must become unreachable, not merely unused.

**Not deleted — relocated:** opponent-addressed response queues still exist. They live *inside*
the single `(worker_id, model_id)` routing map rather than in a second parallel collection. The
thing being removed is the duplicate server and its duplicate queue *set*, not per-model
addressing.

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
- Per-model routing: two models, interleaved requests, each response matches **its own** model;
  never crossed.
- Unknown model id raises rather than defaulting.
- Batching groups by model and never mixes ids in one flush.
- Response-queue addressing: `(worker_id, model_id)` is the key; no two evaluators share a queue.
- Default-off equivalence for the one-model case.

### Test replacement map — required, not an estimate

The existing two-server tests describe a design that will no longer exist. **Do not delete
coverage; map it.** Before countersignature the implementation must include a
**behaviour-by-behaviour table**: each retired test, the behaviour it asserted, and the test
that now asserts that behaviour — or an explicit note that the behaviour itself is gone with
the second server.

**Do not trust a count made in advance.** An earlier draft said "five tests"; a crude scan then
flagged over twenty, but that scan over-reports because its window bleeds into neighbouring
functions. The affected set — including shared helpers `_tiny_run`, `_balanced_call`, `_drain`
and the `StubEvaluator`/`ExplodingEvaluator` companions — must be established during
implementation and **the final suite-count delta reported** against the 2,865 baseline.

### Real-GPU smoke — frozen dose

Stubs cannot discharge this (#50). The smoke drives the **real arbiter** with **real workers**.

| item | value |
|---|---|
| model A | `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`, sha1 `209cf2d4fd24a48553d259dd71b4954867b9473e` |
| model B | `checkpoints/alphazero-v2-staged/model_iter_0379.safetensors`, sha1 `8ad62ac432c35c6ea9b0630b8a2b8c572a0b03a1` |
| **why two DIFFERENT checkpoints** | identical weights would make a crossed response **invisible** — both models would return the same digest. Distinct weights are what make routing falsifiable. |
| workers | 2 |
| requests | 100 per model per worker ⇒ **400 routed** |
| **true dose** | **402 synchronous GPU calls** — 2 direct reference calls + 400 routed requests |
| batch | `B=14`, `M=64`, `C=30`, `active_size=24` |
| inputs | deterministic, seed `20260811`, **the same batch to both models**, so each model's digest is a fixed reference |
| batching | `max_batch_rows=14`, `flush_ms=2`, both pinned |
| schedule | alternating and **opposed**: worker 0 issues A/B, worker 1 issues B/A |
| timeout | `900 s` via `SIGALRM` ⇒ exit `142` |
| artifacts | `logs/eval/arbiter_smoke.json`, `logs/eval/arbiter_smoke.exit` — **refuse if either exists** (exit `3`) |
| both SHA-1s | verified **before** any evaluator is built or Metal touched ⇒ exit `4` on mismatch |

**The reference digests are computed on the arbiter-owning thread**, the same thread that
later serves requests — never on the harness or main thread. Computing them anywhere else would
put a second thread on the device and violate invariant 1 while trying to verify it.

**What this smoke does and does not exercise.** With `B=14` and `max_batch_rows=14`, **every
request becomes its own device batch**, so mixed-model grouping inside one flush never occurs
here. The GPU smoke therefore tests **device ownership and routing**; **mixed-request grouping
is discharged separately by deterministic non-GPU tests**. Saying otherwise would overclaim what
400 single-request batches can show.

**Digest agreement is NOT the criterion here.** The probe could compare digests because both its
instances loaded one checkpoint; ordinary self-play sends *different* board states to the two
models, so identical weights alone cannot produce agreeing digests. This smoke instead sends an
**identical controlled batch to two different models** and requires **each model's every
response to match that model's own reference digest**, computed in-process before the round
trip. Crossing then fails loudly.

**Pass requires all of:** exact per-model request counts; every response shape `priors (14, 64)`
and `values (14,)`; every response finite; every response matching its own model's reference
digest; `model_A_digest != model_B_digest` (proving the two models are genuinely distinct, so
the check is not vacuous); and both SHA-1s verified pre-GPU.

**Per-model telemetry is mandatory and its expected values are exact**, not merely "non-zero":

| per model | expected |
|---|---:|
| requests served | **200** |
| rows processed | **2,800** (200 × `B=14`) |
| device batches flushed | **200** (one per request, since `B` equals the row cap) |

Any deviation is a **fail**. A model with zero batches is a fail, not a pass; so is a model with
more batches than requests, which would mean the row cap is not behaving as pinned.

**Exit semantics:** `0` pass · `1` verification fail · `3` artifact exists · `4` SHA mismatch ·
`142` timeout · `-6`/`134` SIGABRT · anything else invalid. **Any non-zero result is a stop, not
a retry.** Exact command blocks are written into this card once the smoke script exists and is
committed unrun, exactly as the probe was.

## What this card does NOT authorize

- Lifting the `--frozen-opponent-checkpoint` startup refusal. That is a **separate** card, after
  this one lands and its real-GPU round trip passes.
- Any warmup, training run, evaluation, checkpoint, or seed-interval reservation.
- Re-running the aborted frozen-parent recipe in any form.
- Changing the dual-root game seam, which is already tested and unaffected.

## Prediction, on record

**The refactor lands and its real-GPU round trip passes.** The probe removed the main doubt —
one thread can hold two resident networks — and what remains is ordinary plumbing. Moderate
confidence: per-model batching **may roughly halve** effective batch size **under balanced
simultaneous demand**, so a throughput cost is plausible and unmeasured, and the probe's "no
observed slowdown" was one sample at a dose with no queueing at all. Real demand is unlikely to
be perfectly balanced, which is why the smoke reports per-model telemetry rather than assuming
a ratio.

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
