# Corrections to FAILURE.md

`da3d2c9` recorded the failed golden capture. Review found three claims in that memo that the
evidence does not support. `da3d2c9` is **not rewritten**; the corrections are recorded here and
applied to `FAILURE.md`, so the original assertion and its retraction both remain visible.

The **raw evidence is unaffected** — `partial/G_P01_baseline_s1.json` and `capture.log` are
byte-identical to what the run produced, and their hashes are unchanged.

## 1. "the complete, unedited console output" — WRONG

`capture.log` is the complete **orchestrator** log. On a worker failure, `capture.mjs` printed
only `proc.stderr` and **discarded `proc.stdout`**, so the worker's own standard output for the
failing case was never recorded and cannot be recovered.

This matters beyond wording: the missing stdout is precisely the observation that would say
whether the post-`runCase` `console.log` executed, which is the sharpest available localisation
of the abort. The evidence gap is a **defect in the orchestrator**, not a property of the
failure — and it is fixed going forward.

## 2. "`runCase` returned normally" — NOT SUPPORTED

A complete artifact and an absent `.tmp` prove that the search ran, the trace was constructed and
the artifact was **atomically published**. They do not prove that `runCase`'s promise resolved,
that `console.log` ran, or that `process.exit` was reached: a native thread can abort
asynchronously at any point, including immediately after publication.

The supportable statement is the narrower one — **the abort came after publication** — and the
memo now says that.

## 3. "`mutex lock failed` is that teardown failing", and "not in ONNX Runtime" — OVERSTATED

No observation isolates `process.exit` as the aborting mechanism, and a failure arising only on
the ORT path cannot be ruled out as an ORT defect without a controlled probe. None was run.

The memo now says the message is **consistent with an unsafe forced teardown**, and describes the
defect in terms of directly observable harness behaviour: **the worker does not release the
`InferenceSession` before forcing exit.**

What the evidence *does* support is retained: the failure is **not** a failure to produce an MCTS
result, and the eager-expansion memory defect is an implausible explanation at one simulation.

## Remedy shipped alongside these corrections

Construction only — no capture, no probe, no `server/mcts.js` change:

- `worker.mjs` releases the session in `finally`, via `InferenceSession.release()`
  (`onnxruntime-common/.../inference-session.d.ts:437`), on both the success and failure paths;
- forced `process.exit()` is replaced by `process.exitCode`, so the event loop drains instead of
  the process being torn down under it;
- `capture.mjs` preserves worker `status`, `signal`, `error`, `stdout` **and** `stderr` on every
  failure, so the next failure cannot lose the evidence this one did;
- lifecycle and ordering tests using fakes, including that the session is released even when the
  search throws.

**The hypothesis remains untested.** The next gate is one real, single-case worker lifecycle
probe — not a 92-case recapture.
