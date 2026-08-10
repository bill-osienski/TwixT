# Single-Arbiter Metal Feasibility — Probe Card

**Date:** 2026-08-10 · **Status:** DRAFT, not countersigned · **Scope: one standalone
diagnostic probe. No production change, no training, no evaluation, no seed interval.**

Successor step to the aborted frozen-parent run
(`docs/superpowers/2026-08-10-frozen-parent-opponent-experiment-card.md`, exit `134`,
do-not-repeat **#50**).

## This is not a research experiment

It produces **no strength claim, no checkpoint, no evaluated games, and draws no seed
interval.** It is an engineering feasibility gate whose only output is a yes/no about the
device. Nothing in the results table or the seed ledger changes because of it.

## The question

**Can a single Metal-owning thread serve two distinct networks under sustained interleaved
load without a driver abort?**

That is the premise of the arbiter design #50 prescribes — one thread owning the device,
routing requests to per-model evaluators. **The premise is currently untested.** If it is
false, the arbiter refactor is dead before it is written and the frozen-parent mechanism needs
a different shape entirely (separate processes, or one network evaluated per batch).

## Why probe before building

The arbiter is a real change to `inference_server.py` and `remote_evaluator.py` — a model id on
`InferenceRequest`, per-model batching, both `RemoteEvaluator`s sharing one request queue.
That is worth writing **only if** one thread can hold two networks safely. A ~100-line
standalone probe answers it in minutes; the refactor would cost a day and could reach the same
abort from a different direction.

**Nothing in the existing suite can answer this.** Per #50, the transport tests drive both
servers with stub evaluators that never touch the GPU. This probe exists precisely because a
stub-level test is not evidence about a device-level contract.

## Design — two arms, each in its own subprocess

A standalone script, run manually, writing an exit code per arm. Neither arm imports or alters
the training path.

**Arm A — control, reproduce the failure.** Two threads, one network each, submitting
concurrently. **Expected: abort (SIGABRT / exit 134).** Its job is to prove the diagnosis in
#50 is right. Deliberately crashing a subprocess is the point; it is isolated and expected.

**Arm B — treatment, the arbiter premise.** **One** thread alternating inference between two
distinct loaded networks, same total volume as arm A. **Expected: completes, exit 0.**

Both arms use real `LocalGPUEvaluator`s on real checkpoints at the pinned network shape, run a
bounded number of interleaved batches, and print a per-arm exit code. Each arm runs in its own
subprocess so arm A's abort cannot take arm B with it.

## Preregistered decision table

| arm A | arm B | reading | consequence |
|---|---|---|---|
| aborts | exit 0 | Diagnosis confirmed **and** the arbiter premise holds | The arbiter refactor is justified. It needs its **own** authorization; this card does not grant it. |
| aborts | aborts | Diagnosis confirmed, **premise false** | The arbiter is dead. Two networks cannot share one device in-process at all. Frozen-parent needs a different shape or is abandoned — record it and stop. |
| exit 0 | exit 0 | **We do not understand the 2026-08-10 failure** | Stop. Do **not** proceed on a diagnosis that failed to reproduce. Re-diagnose from the original log first. |
| exit 0 | aborts | Incoherent — the safer arm failed | Stop and re-diagnose. Treat the probe itself as suspect. |

**No arm's result authorizes a training run.**

## Prediction, on record before the run

**Arm A aborts, arm B completes.** The 2026-08-10 assertion names concurrent encoding on one
command buffer, which is a *concurrency* fault rather than a two-networks fault, and MLX
evaluates many distinct modules on one thread routinely. Confidence is moderate, not high — the
non-reproduction row exists in the table because a heisenbug is a live possibility, and the
"premise false" row exists because two large resident graphs may interact in ways a single
network does not.

## Cost and stop rule

**Minutes of GPU time**, both arms, at a bounded batch count. If the probe cannot be written in
roughly **100 lines and half a day**, stop and re-scope rather than growing it.

## Explicitly NOT authorized by this card

- Implementing the arbiter, or any change to `inference_server.py`, `remote_evaluator.py`,
  `trainer.py`, `self_play.py` or `self_play_worker.py`.
- Lifting the `--frozen-opponent-checkpoint` startup refusal.
- Any warmup, training run, evaluation, checkpoint or seed-interval reservation.
- Re-running the aborted frozen-parent recipe in any form.

## Preconditions

1. Clean worktree; probe run from a committed HEAD.
2. The probe writes only to `logs/eval/arbiter_probe_*` — new paths, refuse if present.
3. Suite green before the run (the probe adds no production code, so this is cheap).

---

## Countersignature

Execution is authorized only when this block is filled in, committed, and pushed.
**Unsigned, this card authorizes nothing.**

```
authorizer          : ____________________
timestamp (UTC)     : ____________________
authorization basis : ____________________   # the reviewed commit this signature approves
execution commit    : the commit containing this completed countersignature block
approved scope      : write the standalone probe, run arm A and arm B once each,
                      and report both exit codes against the decision table.
                      Nothing else.
```

**Conditions:**

- The probe is standalone. If writing it requires touching production code, **stop** — that
  itself is a finding and needs re-scoping.
- Approval covers **one run of each arm**. A non-reproducing arm A is a stop, not a retry.
- No result of this probe authorizes the arbiter, the flag, or any training. Each needs its own
  card.
