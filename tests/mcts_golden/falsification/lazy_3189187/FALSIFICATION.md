# Lazy falsification — GATE SATISFIED

**Date:** 2026-08-17 · **Outcome: exit `0`, `COPY-COUNT GATE SATISFIED`, `copy_count 8`.**

The second half of design §5. The same falsification that **violated** its gate against the eager
implementation (`../eager_481f9bd/`, `copy_count 4490`) now **satisfies** it against the lazy one,
at the same position, the same simulation count and the same gate.

## What was run

```
node tests/mcts_golden/falsify.mjs --stage lazy
```

| | |
|---|---|
| execution commit | `3189187615941b8e2fff2a9914d4cf779dbb4059` (pushed before the run) |
| execution surface sha256 | `d7fb6bc3fbc722e306940accadc2b8bdda6c92d125710b9b22c32d31dac4c769` |
| stage | `lazy` — pinned to that surface, checked before **and** after the measurement |
| worktree | clean, re-checked after the measurement |
| position | `P11` — `timing_02_opening_202.json` @ prefix 28, `n_legal` **500** |
| model requested / loaded | `1d64027db521a50f` / `1d64027db521a50f` |
| simulations | `8`, `cPuct 1.5` |
| gate | `copyCount <= 8` |
| started / ended (UTC) | `2026-08-17T14:09:47Z` / `2026-08-17T14:09:47Z` |
| node | `v26.7.0` · onnxruntime-node `1.23.2` |

## Result

| observation | value |
|---|---|
| **exit status** | **`0`** — the code reserved for a satisfied gate |
| **signal** | **`null`** — returned, not signalled |
| stdout | `stdout.txt`, 924 bytes, sha256 `fc497b32a2b8452c8711d3c38aa2636c39588d725cdce231f2c0a6885744099c` |
| stderr | **0 bytes**, sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| **`copy_count`** | **8** |
| `satisfied` | `true` |
| `required_outcome` for this stage | `satisfied` |
| verdict | `COPY-COUNT GATE SATISFIED` |

Status was captured directly, with no pipeline able to substitute another command's status.
Empty stderr means no `SESSION_RELEASE_FAILED`, no `SESSION_RELEASE_UNAVAILABLE`, no `SECONDARY`
line and no abort.

## The two runs side by side

| | eager (`481f9bd`) | lazy (`3189187`) |
|---|---:|---:|
| execution surface | `228f57b5…` | `d7fb6bc3…` |
| `copy_count` | **4490** | **8** |
| against the gate of 8 | 561.3× over | at it |
| exit status | `1` (violated) | `0` (satisfied) |

Position, model, simulation count, `cPuct` and gate are identical across both runs; only the
execution surface differs. **561× fewer state constructions.**

## `copy_count` landed exactly on the structural ceiling

Design §3 argues the root is not copied and each simulation materializes **at most one** child,
making `S` an exact ceiling rather than an approximation. The observed count is **8 for S = 8** —
at the ceiling, not merely under it. Every one of the eight simulations materialized exactly one
new child: none terminated early at an already-materialized node and none reached a terminal leaf.

That is a stronger observation than `≤ 8` alone. It is **not** a proof that the ceiling is
attained everywhere: it describes this position at this simulation count.

## What this establishes

- The falsification **binds and now passes**: it failed against the pre-change code and succeeds
  against the changed code, which is what makes the pass non-vacuous.
- For this position and simulation count, state construction is bounded by the simulation count
  rather than scaling with the 500 legal moves.

## What this does NOT establish

- **Nothing about behavioural equivalence.** This measures allocation only. Whether the lazy
  implementation reproduces the eager traces is the golden comparison (§4), which has not been
  run.
- **Nothing about the §6 heap criterion.** That is a separate measurement at 800 simulations
  against a preregistered ceiling, and has not been run.
- **Nothing about the other 15 positions, the other four simulation counts, the candidate model,
  or the abort cases.**
- **No scaling law.** One position at one simulation count cannot establish one; §3's argument is
  structural.

## Status

§5 is now complete in both directions. The lazy golden capture and its exact comparison against
the eager corpus remain **unauthorized**, as do the heap measurement, the timing smoke, `P`
selection and the match.
