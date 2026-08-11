# Single-Arbiter Metal Feasibility — Probe Card

**Date:** 2026-08-10 · **Status:** DRAFT, not countersigned · **Scope: run an
already-written, already-reviewed probe script. No production change, no training, no
evaluation, no seed interval.**

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
a different shape entirely — **not** separate processes, which #50 rejects as a successor
shape; the remaining option would be one network evaluated per batch, or abandonment.

## Why probe before building

The arbiter is a real change to `inference_server.py` and `remote_evaluator.py` — a model id on
`InferenceRequest`, per-model batching, both `RemoteEvaluator`s sharing one request queue.
That is worth writing **only if** one thread can hold two networks safely. A ~100-line
standalone probe answers it in minutes; the refactor would cost a day and could reach the same
abort from a different direction.

**Nothing in the existing suite can answer this.** Per #50, the transport tests drive both
servers with stub evaluators that never touch the GPU. This probe exists precisely because a
stub-level test is not evidence about a device-level contract.

## Design — two arms, one thread in both, identical device work

**There is no two-thread arm.** #50 forbids two owners submitting concurrently *"in any
experiment, for any purpose"*, and an earlier draft of this card proposed exactly that as a
"control" — a direct violation. It would also have been worthless: a race that fails to
reproduce in one short run falsifies nothing about a timing-sensitive fault.

| arm | evaluators | calls | thread |
|---|---|---|---|
| **control** | one | `2N` on that one | one |
| **treatment** | **two independent instances** | `N` each, strictly alternating | one |

Total device work is identical; the only difference is whether two networks are resident and
interleaved. That isolates the premise without ever creating a second device owner.

Both instances load the **same** checkpoint, which is faithful to iteration 1 and yields a free
correctness check: their digests must agree.

## Frozen dose — every value pinned

| item | value |
|---|---|
| checkpoint | `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` |
| SHA-1 | `209cf2d4fd24a48553d259dd71b4954867b9473e`, re-hashed at run time and recorded |
| network | `hidden=128`, `blocks=6`, two **independent** `create_network` + `load_weights` instances |
| `compile` | **`False`**, explicit — the training path never compiles |
| board | `active_size=24` |
| batch dims | `B=14` (matches `--mcts-eval-batch-size`), `M=64` move slots, `C=30` channels |
| calls | `N=200` per evaluator ⇒ **400 `infer()` calls per arm** |
| inputs | deterministic, `numpy` seed `20260810`, one batch reused for every call |
| timeout | `900 s` via `SIGALRM` ⇒ exit `142` |
| outputs | `logs/eval/arbiter_probe_control.json`, `logs/eval/arbiter_probe_treatment.json` (refuse if present) |

**`infer()` synchronously exercises Metal.** `LocalGPUEvaluator.infer()` calls `mx.eval`
(`local_evaluator.py:115` and `:119`), so each call forces execution rather than queuing a lazy
graph. Recorded in the report as `mx_eval_forces_sync`.

## Verification — exit 0 alone is not a pass

Each arm writes a JSON report and passes **only** if all of:

- `calls_completed` equals the expected count **per evaluator** (`[400]` control,
  `[200, 200]` treatment);
- output shapes are exactly `priors (14, 64)` and `values (14,)`;
- every output is **finite**;
- a stable **digest** is present for each evaluator, and in treatment the two **agree** (same
  checkpoint, same inputs);
- the re-hashed checkpoint SHA-1 matches the pinned value.

**Exit-code semantics.** `0` = completed; **SIGABRT = subprocess return code `-6`, or `134`
when normalised by a shell** — the failure under test; `142` = timeout. **Any other nonzero
result is invalid: stop and diagnose, do not interpret it as a device finding.**

## Preregistered decision table

| control | treatment | reading | consequence |
|---|---|---|---|
| PASS | PASS | One thread holds two resident networks safely | **The arbiter premise holds.** The refactor is justified and needs its **own** authorization; this card does not grant it. |
| PASS | SIGABRT | Two resident networks abort a single owner | **The arbiter is dead.** Frozen-parent needs one-network-per-batch or abandonment. Record and stop. |
| SIGABRT | either | The baseline itself aborts | **Stop.** The device fails on ordinary single-network load, so nothing here is about two networks. Re-diagnose from first principles. |
| timeout / other nonzero / verification fail | any | Result invalid | **Stop.** Do not read a device finding out of an inconclusive run. |

**No outcome authorizes a training run**, and no outcome lifts the
`--frozen-opponent-checkpoint` refusal.

## Prediction, on record before the run

**Both arms pass.** The 2026-08-10 assertion names concurrent encoding on one command buffer,
which is a *concurrency* fault rather than a two-networks fault, and MLX evaluates many distinct
modules on one thread routinely. Confidence is moderate, not high: the "arbiter is dead" row
exists because two large resident graphs may exhaust or interleave device resources in ways one
does not, and this probe deliberately cannot tell us whether the original race is fixed — only
whether the proposed replacement is viable at all.

## Cost and stop rule

**Minutes of GPU time** — 400 `infer()` calls per arm, two arms. The script is already written
and committed (see the preconditions); its size is the record of the ~100-line stop rule.

## Explicitly NOT authorized by this card

- Implementing the arbiter, or any change to `inference_server.py`, `remote_evaluator.py`,
  `trainer.py`, `self_play.py` or `self_play_worker.py`.
- Lifting the `--frozen-opponent-checkpoint` startup refusal.
- Any warmup, training run, evaluation, checkpoint or seed-interval reservation.
- Re-running the aborted frozen-parent recipe in any form.

## Preconditions

1. **The probe script exists, is committed, and has been reviewed** —
   `scripts/GPU/alphazero/probe_single_arbiter_metal.py`. It was written and committed
   **without being run**, so the countersignature approves reviewed code rather than an
   intention to write some.
2. Clean worktree; the run executes from the countersigned commit.
3. Both output paths absent — the script refuses if either exists (exit `3`).
4. Suite green before the run. The probe imports the training path but changes none of it.

## Command block 1 — control `[GPU, read-only apart from its report]`

```bash
bash -c '.venv/bin/python -m scripts.GPU.alphazero.probe_single_arbiter_metal --arm control
rc=$?
printf "%s\n" "$rc" > logs/eval/arbiter_probe_control.exit
exit "$rc"'
```

## Command block 2 — treatment `[GPU, read-only apart from its report]`

```bash
bash -c '.venv/bin/python -m scripts.GPU.alphazero.probe_single_arbiter_metal --arm treatment
rc=$?
printf "%s\n" "$rc" > logs/eval/arbiter_probe_treatment.exit
exit "$rc"'
```

Run **control first**. If control does not pass, treatment is not run.

---

## Countersignature

Execution is authorized only when this block is filled in, committed, and pushed.
**Unsigned, this card authorizes nothing.**

```
authorizer          : ____________________
timestamp (UTC)     : ____________________
authorization basis : ____________________   # the reviewed commit this signature approves
execution commit    : the commit containing this completed countersignature block
approved scope      : run the ALREADY-COMMITTED, ALREADY-REVIEWED script
                      scripts/GPU/alphazero/probe_single_arbiter_metal.py
                      once per arm via the two command blocks below, and report
                      both reports and exit codes against the decision table.
                      No edit to that script after signing. Nothing else.
```

**Conditions:**

- The script is frozen at signature. Editing it voids the signature; amend and re-sign first.
- Approval covers **one run of each arm**. Any invalid result is a stop, not a retry.
- No result of this probe authorizes the arbiter, the flag, or any training. Each needs its own
  card.
