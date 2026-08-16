# Timing smoke FAILURE — heap exhaustion at `74dca6e`

**Date:** 2026-08-16 · **Outcome: FAILED. `selected_p: none`.**

Phase 3 step §11.4 of `docs/superpowers/2026-08-14-product-stack-comparison-specification.md`
(rev 6). The single authorized ten-game timing invocation aborted after four games. **No `P`
was selected, no `p_decision.json` was written, and none may be derived from this run.**

## What was run

| | |
|---|---|
| command | `node tests/product_match/timing.mjs runs/timing` |
| execution commit | `74dca6e1535ee1e36d640dae3ba644c6c2ed2e5e` |
| worktree at launch | clean (`executionCommit()` would have refused otherwise) |
| execution-surface sha256 | `228f57b55448f44136ffd41d6f092c9da904ca469a1e7bc4055656ffd8ef77bd` |
| opening pool sha256 | `f14e80163910694bc48b04552dad529465f2fafb0aaeb4b49e522281d9c65a10` |
| launched (UTC) | `2026-08-16T14:08:56Z` |
| node | `v26.7.0` |
| onnxruntime-node | `1.23.2` |
| ORT configuration | product default — `InferenceSession.create(path)`, no options |
| execution mode | one process, sequential, no concurrency |
| PID | `13582` |

Both models loaded and both session contracts asserted successfully before the clock started;
that part of the run is not implicated.

## What completed

Four of ten games. All four are from the **baseline self-play arm** (openings `200…204`). The
candidate self-play arm (`205…209`) **never ran**.

| sidecar | opening | arm | result | termination | plies | elapsed | sha256 |
|---|---:|---|---|---|---:|---:|---|
| `timing_00_opening_200.json` | 200 | baseline self-play | red | win | 39 | 165,947 ms | `0a63df9b9fc5b5b8d28660b92277910d9b6299a2f40dbd6ce428d1d7665122f8` |
| `timing_01_opening_201.json` | 201 | baseline self-play | red | win | 51 | 235,322 ms | `900f1c5c67337e27ae970ce946d510247cea8026c205547689c25c66e920fc36` |
| `timing_02_opening_202.json` | 202 | baseline self-play | red | win | 57 | 272,157 ms | `960a72869a72ccf93fe44faf73b34b9e4c6f23347cb02a527f5f10802a0436ce` |
| `timing_03_opening_203.json` | 203 | baseline self-play | black | win | 54 | 250,883 ms | `4d51789f3ec226b9c084f8bd5ed14a2223264e5f350ad4932546d4e986df3cc9` |

Sum of the four `elapsed_ms`: **924,309 ms = 15.41 min**. Per-search cost rose across the four
games, `4.26 → 4.61 → 4.77 → 4.65` s.

The fifth game (opening `204`, baseline self-play) began and never finished.

`oom_crash.log` (sha256 `5d9ea238911b60073de4c3d6befdbf94a40b720c81f4140f3596035aad91a2e7`)
is the complete unedited process output.

## How it failed

The process aborted **2,303,279 ms ≈ 38 min 23 s** after start — so roughly **23 minutes** were
spent inside game 5 without completing it, against 15.41 min for the four games that did.

```
<--- Last few GCs --->

[13582:0x86140c000]  2303025 ms: Mark-Compact 4080.0 (4100.1) -> 4076.5 (4100.1) MB,
    pooled: 2.5 MB, 213.96 / 0.00 ms  (average mu = 0.140, current mu = 0.009)
    allocation failure; scavenge might not succeed
[13582:0x86140c000]  2303279 ms: Mark-Compact 4080.5 (4100.6) -> 4077.1 (4101.4) MB,
    pooled: 1.2 MB, 251.80 / 0.00 ms  (average mu = 0.071, current mu = 0.007)
    allocation failure; scavenge might not succeed

FATAL ERROR: Ineffective mark-compacts near heap limit
             Allocation failed - JavaScript heap out of memory
```

`current mu = 0.007` means over 99% of the time was being spent in garbage collection.
**"Ineffective mark-compacts" means full compaction could not reclaim the space** — the ~4 GB
was live and reachable, not uncollected garbage. Node's default old-space limit is ~4 GB and the
heap sat at 4,080 MB against it.

## Mechanism

Established by static reading of the committed source at `74dca6e`, and consistent with the
observed heap size to an order of magnitude. **No heap snapshot was taken**, so additional
contributors are not excluded.

1. `server/mcts.js:178-183` — on every `_expand()`, the `// Create children (unexpanded)` loop
   iterates **every legal move**, calling `node.state.applyMove(move)` and constructing an
   `MCTSNode` for each.
2. `server/gameLogic.js:377` — `applyMove()` calls `this.copy()`.
3. `server/gameLogic.js:172-181` — `copy()` deep-copies the peg map and bridge set
   (`new Map(this.pegs)`, `new Set(this.bridges)`).

One search performs 1 root expansion + up to 800 leaf expansions. At plies 40–60 there are
roughly 470–490 legal moves, so a single 800-simulation search materializes on the order of
`801 × ~480 ≈ 385,000` fully deep-copied states, all reachable from the live root for the
duration of that search.

**This is algorithmic peak retention within one search, not a leak across moves or games.**
Ruled out by reading:

- `MCTS.search()` allocates a fresh root per call and returns only a `Map` of visit counts and a
  number, so no tree survives a search;
- `playGame()` constructs its two `MCTS` objects per game and drops them at return;
- `AlphaZeroInference.evaluateRaw()` allocates per call and retains nothing on the instance.

The two `AlphaZeroInference` sessions are the only objects living across all games, and neither
accumulates.

## Consequences

- **`selected_p: none`.** No `P` exists. `tests/product_match/p_decision.json` was never
  written, and both production entry points therefore still refuse, correctly.
- **`P` may not be derived from this run.** §7.3 defines throughput as one wall-clock span over
  ten games; four games interrupted by GC thrashing is not that measurement, and the per-search
  times are contaminated by heap pressure.
- **No comparative inference of any kind is available from this evidence.** Both arms of the
  smoke are self-play by design (§7.3), and only the baseline arm ran at all. These four games
  say nothing about baseline versus candidate, and must never be cited as if they did.
- **The defect is not specific to timing.** The same search runs in the match, so the
  ~30-hour Arm A run would have failed the same way — likely around game 5 of 400. It also
  affects the shipped server, which runs the same `server/mcts.js` at `nSims: 800` on hard.
- The worktree remained clean throughout; the failed run wrote only into gitignored `runs/`.

## Status and what is not authorized

Phase 3 is **blocked** pending remediation. Explicitly **not** authorized by this memo or by the
run it records: rerunning the smoke, raising Node's heap limit, generating `p_decision.json`,
inferring `P`, taking a heap snapshot, running a profiler, editing source, or implementing a
remedy.

Raising `--max-old-space-size` was considered and **rejected as a workaround**: it treats a
symptom, and a run that only completes because it was given more headroom is not the
configuration that ships.

## Conditions any remedy must meet

Recorded here so they are fixed before implementation, not fitted to it. The likely remedy is
**lazy child-state materialization** — retain priors for every legal move, but construct a child
`TwixtState` only when PUCT first selects that move. Any implementation must preregister and
verify:

- exact equivalence of visit counts, move selection and tie-breaks against current behaviour;
- child-state count bounded approximately by the simulation count, not by
  simulations × legal moves;
- no change to the 800 simulations, the ORT configuration, the readout policy, the models, or
  the statistical design;
- a successful bounded-memory 800-simulation product search, demonstrated before any further
  ten-game timing authorization.

Note that `server/mcts.js` is one of the ten execution-surface files. Changing it moves the
execution-surface digest away from `228f57b5…`, which is correct and expected: the timing
measurement must then be taken afresh against the new surface.
