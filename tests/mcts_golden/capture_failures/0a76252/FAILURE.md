# Golden capture FAILURE — native abort after artifact publication

**Date:** 2026-08-16 · **Outcome: FAILED at case 1 of 92. NO VALID CORPUS EXISTS.**

The single authorized 92-case golden capture
(`docs/superpowers/2026-08-16-mcts-memory-remediation-design.md` §4) aborted on its **first**
case. No golden corpus was produced, `tests/mcts_golden/golden/` was **not** created, and nothing
may be compared against this run.

## What was run

| | |
|---|---|
| command | `node tests/mcts_golden/capture.mjs capture runs/mcts_golden_eager_0a76252` |
| capture commit | `0a76252c5d6d3d1c0ad9c7bdb19d0417757bb060` |
| pinned surface commit | `74dca6e1535ee1e36d640dae3ba644c6c2ed2e5e` |
| execution-surface sha256 | `228f57b55448f44136ffd41d6f092c9da904ca469a1e7bc4055656ffd8ef77bd` |
| worktree at launch | clean (preflight enforces it) |
| output directory | `runs/mcts_golden_eager_0a76252` (absent beforehand, created by the run) |
| started (UTC) | `2026-08-16T20:03:23Z` |
| node | `v26.7.0` |
| onnxruntime-node | `1.23.2` |
| failing case | `G_P01_baseline_s1` — P01, baseline `1d64027db521a50f`, 1 simulation |
| worker pid | `63321` |

## The failure

```
capture commit 0a76252c5d6d3d1c0ad9c7bdb19d0417757bb060
mode           capture

[ 1/92] G_P01_baseline_s1  FAILED exit=null
libc++abi: terminating due to uncaught exception of type std::__1::system_error:
    mutex lock failed: Invalid argument

stopped at G_P01_baseline_s1; the corpus is incomplete and is not a corpus
```

`capture.log` (sha256 `b1b69c5dcea48c327ae9a3d2709a0380ba8e994db04379056ee163769aa27f5f`) is the
complete, unedited **orchestrator** log — all seven lines of it.

> **Correction (see `CORRECTIONS.md`).** An earlier revision of this memo called it "the complete
> console output". It is not. On a worker failure `capture.mjs` printed only `proc.stderr` and
> **discarded `proc.stdout`**, so the worker's own standard output for this case **was never
> captured and no longer exists**. In particular the evidence **cannot** say whether the
> post-`runCase` `console.log` executed — which is exactly the observation that would most
> sharply localise the abort. That gap is a defect in the orchestrator, not a property of the
> failure.

`exit=null` from `spawnSync` means the worker was **killed by a signal**, not that it returned a
non-zero status. The message is a native C++ abort, not a JavaScript error and **not an
out-of-memory condition**.

## The decisive detail: the work had already finished

`partial/G_P01_baseline_s1.json` (sha256
`f67fd34d0f8a4a71c3f043201df0a37d0644fbb1b5cfdb55994df902d62fc461`, 23,974 bytes) is **complete
and semantically well-formed**:

| field | value |
|---|---|
| `status` | `captured` |
| `trace.visit_counts` | **524 entries**, matching `fixture.n_legal` |
| sum of visit counts | **1**, matching `n_simulations: 1` |
| `trace.root_value` | `-0.09874988347291946` |
| `trace.selected_move` | `6,17` |
| `trace.progress` | 1 entry, `{done: 1, total: 1, valueEstimate: -0.0987…}` |

The output directory contains **exactly one file and no `.tmp` residue**, so
`writeAtomicNoClobber` ran to completion — temp created, linked to the final path, temp unlinked.

**What that establishes, precisely:** the search ran, the trace was constructed, and the artifact
was atomically published. The abort therefore came **after publication**, and the failure is not
a failure to produce an MCTS result.

**What it does NOT establish:** that `runCase`'s promise resolved, or that `console.log` ran, or
that `process.exit` was ever reached. A native thread can abort asynchronously at any moment,
including immediately after publication and before the surrounding JavaScript continues. An
earlier revision of this memo asserted "`runCase` returned normally"; that is not supported by
the evidence and is withdrawn.

## Leading hypothesis — and it is a hypothesis

Three facts, all recorded rather than re-derived:

1. The **dry-run** path passed **92/92** twice. Dry-run reaches ONNX Runtime only through the
   dynamic `import()` inside `captureTrace`, so it never loads a model.
2. The **first case that loads a model** aborted.
3. It aborted **after the artifact was published**.

The code between publication and process end is `console.log(...)` followed by
**`process.exit(EXIT_OK)`** in `worker.mjs::main`, and the worker **never released the
`InferenceSession`** it opened. `mutex lock failed: Invalid argument` is **consistent with an
unsafe forced teardown** — `process.exit()` terminating the process while ORT's native thread
pool is still live and the session is unreleased.

**Stated as the defect: the harness does not release the session before forcing exit.** That is a
description of the harness's behaviour, which is directly observable, rather than a claim about
which native call aborts.

**What the evidence does NOT support:**

- It does **not** isolate `process.exit` as the aborting mechanism. No observation distinguishes
  it from another asynchronous native fault on the ORT path.
- It does **not** exonerate ONNX Runtime. A failure that arises only on the ORT path cannot be
  ruled out as an ORT defect without a controlled probe, and none was run.
- It does **not** identify the precise native fault. No profiler, debugger or re-run was used,
  because none is authorized.

**What the evidence does support:** the failure is **not** a failure to produce an MCTS result
(the trace is complete and well-formed), and the eager-expansion memory defect is an
**implausible** explanation — the failing case ran a single simulation, nowhere near the scale at
which that defect bites.

The models, the fixtures and `server/mcts.js` are likewise unimplicated by the trace being whole,
though "unimplicated" is weaker than "exonerated".

## Consequences

- **No valid corpus exists.** `tests/mcts_golden/golden/` was not created and must not be
  created from this run.
- **The one artifact is not a golden trace.** It is evidence about a crash. It happens to be
  well-formed, and it is preserved for exactly that reason — it is what proves the work finished
  before the abort — but 1 of 92 is not a corpus (§4.2 fixes the count at 92, and §9 makes any
  other count a stop).
- **The harness guards behaved correctly.** The orchestrator stopped at the first failure rather
  than continuing, and refused to declare success. Nothing was overwritten or deleted.
- **No rerun was attempted**, per the capture authorization.

## Status

The 92-case capture is **blocked** pending review. A remedy would be a change to
`tests/mcts_golden/worker.mjs` and `tests/mcts_golden/capture.mjs` — neither is an
execution-surface file, so the pinned digest `228f57b5…` is unaffected and a fixed harness could
still capture traces describing the same `74dca6e` search code.

Because the hypothesis is not established, the correct next step is **one real, single-case
worker lifecycle probe**, not a 92-case recapture: a corpus is not the instrument for testing
whether a process can exit cleanly.

Nothing here authorizes that change, a re-capture, an `server/mcts.js` edit, a falsification run,
a heap measurement, a timing smoke, a `P` decision, or a match.
