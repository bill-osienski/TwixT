# Eager falsification — GATE VIOLATED, as required

**Date:** 2026-08-17 · **Outcome: exit `1`, `COPY-COUNT GATE VIOLATED`, `copy_count 4490`.**

Design §5 requires the falsification to be **demonstrated failing against the pre-change code
before the change is made** — a falsification first observed after a fix is not evidence that it
binds. This is that demonstration.

## What was run

```
node tests/mcts_golden/falsify.mjs --stage eager
```

| | |
|---|---|
| execution commit | `481f9bd55743d4a453812c6d8581e31aae5f76b9` (pushed before the run) |
| execution surface sha256 | `228f57b55448f44136ffd41d6f092c9da904ca469a1e7bc4055656ffd8ef77bd` |
| stage | `eager` — pinned to that surface, checked before **and** after the measurement |
| worktree | clean, re-checked after the measurement |
| position | `P11` — `timing_02_opening_202.json` @ prefix 28 |
| model requested / loaded | `1d64027db521a50f` / `1d64027db521a50f` |
| simulations | `8`, `cPuct 1.5` |
| gate | `copyCount <= 8` |
| started / ended (UTC) | `2026-08-17T01:03:10Z` / `2026-08-17T01:03:10Z` |
| node | `v26.7.0` · onnxruntime-node `1.23.2` |

## Result

| observation | value |
|---|---|
| **exit status** | **`1`** — the code reserved for a gate violation |
| **signal** | **`null`** — returned, not signalled |
| stdout | `stdout.txt`, 963 bytes, sha256 `03091c4bf621f1c79084834890969795f3211d65d02b9a8a804524ef2a2f4b8d` |
| stderr | **0 bytes**, sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `n_legal` at P11 | **500** |
| **`copy_count`** | **4490** — `561.3×` the gate |
| `satisfied` | `false` |
| verdict | `COPY-COUNT GATE VIOLATED` |

Status was captured directly, with no pipeline able to substitute another command's status.
Empty stderr means no `SESSION_RELEASE_FAILED`, no `SESSION_RELEASE_UNAVAILABLE`, no `SECONDARY`
line and no abort.

## What this establishes

**The falsification binds.** Against the eager implementation it fails, and fails by a wide
margin, so a later passing run would not be vacuous. That is the whole purpose of running it
first.

Design §5's *guaranteed* claim is confirmed: root expansion alone must contribute at least
`528 − 28 = 500` copies at this position, and the observed total is far above that.

## What this does NOT establish

- **Nothing about the lazy implementation.** It does not exist yet. This run says only what the
  eager code did.
- **No scaling law.** One position at one simulation count cannot establish how allocation scales.
  That argument is structural (design §3), not measured here — which is why the verdict wording is
  a copy-count gate and nothing more.
- **Nothing about the other 15 positions, the other four simulation counts, the candidate model,
  or the abort cases.**

## A detail that confirms an earlier correction

The naive `(1 + S) × L = 9 × 500 = 4500` overestimates the observed **4490** by 10.

Design §5 was corrected in review to withdraw `(1+S)×L` as a structural claim, on the grounds
that deeper expansions face different legal-move counts and terminal leaves are never expanded.
The 10-copy shortfall is that correction showing up in real data: the deeper expansions each had
fewer legal moves than the root. The preregistered claim rested only on the root's ≥ 500, which
holds regardless.

## Status

The §5 prerequisite is satisfied. This authorizes nothing further: not the `server/mcts.js`
change, not a lazy-stage falsification, not the equivalence verification, not the heap
measurement, not timing, `P` or a match.
