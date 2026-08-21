# §6 default-heap probe — PASSED

**Date:** 2026-08-21 · **Outcome: `M1` completed, `M2` 84.91 MiB ≤ 512 MiB, exit `0`.**

The measurement design §6 preregistered: one 800-simulation search on the lazy implementation, at
Node's default heap, against a ceiling fixed before any of this was built.

## What was run

```
npm run heap-probe        # node tests/mcts_golden/heap_probe.mjs
```

| | |
|---|---|
| execution commit | `5e2b37217ce179aa37670853b8d17a9f2f3c6a89` (pushed before the run) |
| execution surface | `d7fb6bc3fbc722e306940accadc2b8bdda6c92d125710b9b22c32d31dac4c769` |
| stage | `lazy` — pinned to that surface |
| worktree | clean |
| position | `P11` — `timing_02_opening_202.json` @ prefix 28, `n_legal` **500** |
| model requested / loaded | `1d64027db521a50f` / `1d64027db521a50f` |
| simulations · `cPuct` | **800** · 1.5 |
| ceiling | `512 × 1024² = 536,870,912` bytes |
| launch configuration | `execArgv []`, `NODE_OPTIONS null` — **recorded, not assumed** |
| v8 `heap_size_limit` | `4,395,630,592` bytes (4.09 GiB) — what "default heap" was on this machine |
| started (UTC) | `2026-08-21T04:39:21Z` |
| node · onnxruntime-node | `v26.7.0` · `1.23.2` |

## Result

| observation | value |
|---|---|
| **exit status** | **`0`** · signal `null` |
| stdout | `stdout.txt`, 1,528 bytes, sha256 `9131ff4f6903987970f195b30b0d52e009417c92296dc1e990d04776cf2f969a` |
| stderr | **0 bytes**, sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| **M1** — completed without OOM | **true** |
| **M2** — max observed ≤ ceiling | **true** — `89,031,752` bytes = **84.91 MiB** |
| usage | **16.6%** of the 512 MiB ceiling — **427.09 MiB (83.4%) left unused**; **2.03%** of the 4.09 GiB default heap |

## The sample was complete

| seam | count |
|---|---:|
| H1 (before search) | 1 |
| H2.before / H2.after (each `evaluate`) | 801 / 801 |
| H3 (each progress callback) | 800 |
| H4 (after search) | 1 |
| **total observations** | **2,404** = `1 + 1 + 800 + 2 × 801` |

The completeness gate ran before any verdict, so this maximum is over a full sample rather than a
partial one.

**H2 sat at its ceiling of `1 + nSimulations`.** Unlike a copy count, an evaluation count *does*
constrain traversal here: `evaluate` is reached only from `_expand`, and a simulation ending on a
terminal leaf never expands. 801 evaluations therefore means one root expansion plus **800 leaf
expansions — every simulation reached a non-terminal leaf.**

## What this establishes

For **this position at 800 simulations on the lazy surface**, an 800-simulation product search
completes at Node's default heap and its maximum observed `heapUsed` is well inside the
preregistered ceiling. The ceiling was fixed in the design before the implementation existed and
was not revised.

For contrast, the eager implementation exhausted the same default heap at ~4,080 MiB
(`tests/product_match/timing_failures/74dca6e/`).

## What this does NOT establish

- **Not heap safety in general.** One position, one simulation count, one search.
- **Not that a full game fits.** The original failure was a **game** — dozens of searches in one
  process — not a single search. §6 specifies one search and this satisfies it as written; the
  ten-game timing smoke is what actually exercises many searches in one process.
- **Nothing about throughput or timing.** No wall-clock measurement was taken, and `P` cannot be
  derived from anything here.
- **Nothing about playing strength**, which this programme has never measured on either side.
- **"Maximum observed", not "peak".** A timer cannot observe the true peak: the selection and
  expansion loops block the event loop. The figure is the maximum at the H1–H4 seams.

## Status

§6 is satisfied. The next step is a **fresh ten-game timing smoke at this surface** — the original
gate, still unpassed — then `P`, then the match. `tests/product_match/p_decision.json` remains
absent and `selected_p` remains none.
