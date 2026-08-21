# Ten-game timing smoke — COMPLETED. `P = 100`

**Date:** 2026-08-21 · **Outcome: 10 / 10 games, exit `0`, throughput `6.891897` games/hour,
`selected_p = 100`.**

The §7.3 gate that has blocked Phase 3 since 2026-08-16, when the same smoke aborted at
`74dca6e` with a heap OOM after four games (`../timing_failures/74dca6e/`). It now completes.

**`tests/product_match/p_decision.json` was written by the runner but is NOT committed** — see
"The decision is not yet binding" below.

## What was run

```
node tests/product_match/timing.mjs runs/timing_ff400a4
```

| | |
|---|---|
| execution commit | `ff400a49ab94138a1f403f31b977e9483f119967` |
| execution surface sha256 | `d7fb6bc3fbc722e306940accadc2b8bdda6c92d125710b9b22c32d31dac4c769` |
| worktree at launch | clean |
| configuration | product ORT defaults, no session options; default Node heap, no flags |
| execution | one process, sequential, no concurrency |
| started / finished (UTC) | `2026-08-21T04:48:22Z` / `2026-08-21T06:16:01Z` |
| node · onnxruntime-node | `v26.7.0` · `1.23.2` |

| observation | value |
|---|---|
| **exit status** | **`0`** · signal `null` |
| stdout | `timing.log`, 501 bytes, sha256 `93dc6bf8553ca8d09ffa16c7b746c23f3655e83719ee9157db908bebe8001d04` |
| stderr | **0 bytes**, sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| sidecars | **10**, one per reserved opening `200…209` |

## The measurement

| | |
|---|---|
| `total_sequential_wall_ms` | **5,223,525.365** = 87.06 min |
| `games_per_hour` | **6.891897231172262** |
| threshold | `8.8` |
| **`selected_p`** | **100** |

`6.8919 < 8.8`, so `P = 100`. Recomputed independently from the recorded wall time:
`10 × 3,600,000 / 5,223,525.365 = 6.8919`. **Derived mechanically; never chosen.**

## Per game

| sidecar | arm | plies | result | elapsed | s/search | eager s/search |
|---|---|---:|---|---:|---:|---:|
| `00_opening_200` | baseline self play | 39 | red | 135.8 s | 3.88 | 4.74 |
| `01_opening_201` | baseline self play | 51 | red | 184.0 s | 3.91 | 5.01 |
| `02_opening_202` | baseline self play | 57 | red | 209.2 s | 3.95 | 5.14 |
| `03_opening_203` | baseline self play | 54 | black | 278.1 s | 5.56 | 5.02 |
| `04_opening_204` | baseline self play | 572 | draw | 3402.5 s | 5.99 | — |
| `05_opening_205` | candidate self play | 94 | black | 321.0 s | 3.57 | — |
| `06_opening_206` | candidate self play | 51 | red | 162.3 s | 3.45 | — |
| `07_opening_207` | candidate self play | 62 | black | 207.1 s | 3.57 | — |
| `08_opening_208` | candidate self play | 49 | red | 156.1 s | 3.47 | — |
| `09_opening_209` | candidate self play | 52 | black | 167.4 s | 3.49 | — |

## One game accounts for two thirds of the wall clock

`opening_204` ran **572 plies** to a `no_legal_moves` draw — **568 searches, 56.7 minutes, 65.1%
of the total**. The other nine games took 30.4 minutes between them.

**That is the game the eager run died on.** The previous smoke aborted during game 5 of 10, which
is this one; §7.1 had anticipated the shape ("a pathological game could reach ~44 min") and it
reached 56.7.

**This is an explanation of the number, not a reason to revise it.** §7.3 fixes `P` from the
whole ten-game span, and recomputing without an outlier after seeing it would be exactly the
post-hoc selection the specification exists to prevent. Recorded only so the figure is
understood: the nine other games alone would run at 17.79 games/hour.

## Behavioural cross-check, beyond the frozen corpus

For the four openings that also have an eager sidecar (`200…203`), the lazy run reproduced the
**identical move sequence, result and ply count** — 35, 47, 53 and 50 searches respectively.

That is a full-game agreement check on games the 92-case golden corpus does not cover, since the
corpus compares single searches at fixed prefixes. It is consistent with the exact-match
comparison; it does not extend that result's preregistered scope.

## Per-search cost changed, but not uniformly

`3.45–3.95 s` on seven games versus the eager `4.74–5.14 s`, **but `opening_203` was slower**
(`5.56` vs `5.02`) and the 572-ply game averaged `5.99`. Per-search cost is not uniformly lower
under the lazy implementation, and no claim is made that it is.

## The decision is not yet binding

The runner wrote `tests/product_match/p_decision.json` (3,391 bytes, sha256
`5250a2ab3df0295fe44a051fffad870d11611997a08a4037603f198d1c14fa6c`) as its final step. **It is
untracked and uncommitted**, because committing `P` was not authorized for this run.

Per §7.3 and `p_decision.mjs`, the decision is read as a **git blob** — an uncommitted file is not
a commitment, since its bytes can change after any check. Both production entry points therefore
still refuse, and `selected_p` is **not yet fixed for the match**.

## What `P = 100` costs and buys

At the measured throughput, 100 pairs = 200 games ≈ **29.0 hours**.

§7.2's preregistered characteristics for `P = 100`, none of them revisable now:

| | `P = 100` |
|---|---:|
| planning resolution threshold (observed `s̄`) | `0.598` |
| **true** score giving 80% planning power | `0.640` |
| planning power at a **true** `0.57` | **29%** |

The match would detect only a **large** difference, and at a true `0.57` it would fail roughly
seven times in ten. §6.2 already fixes what follows: a near-tie is reported unresolved and left
there, and **an unresolved result does not authorize a larger match.**

## Scope

This measures **throughput only**. It says nothing about which model plays better: both arms are
self-play by construction (§7.3), so the ten games carry **no comparative information**, and the
per-game results above must not be read as evidence about baseline versus candidate.
