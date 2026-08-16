# MCTS Memory Remediation — PREREGISTERED DESIGN

**Date:** 2026-08-16 · **Basis:** `74dca6e` (Phase 3 implementation) ·
**Failure record:** `tests/product_match/timing_failures/74dca6e/FAILURE.md` (`7510660`, `14e233a`)

**Status: DESIGN. Preregistered before implementation. It authorizes nothing** — see §10.

Phase 3 of `docs/superpowers/2026-08-13-product-model-alignment-decision-memo.md` is **blocked**:
the ten-game timing smoke aborted after four games with a JavaScript heap OOM, so
`selected_p: none` and no `p_decision.json` exists. This document fixes, in advance, what a
remedy must do and how it will be judged — so the acceptance criteria cannot be fitted to
whatever the implementation happens to produce.

---

## 1. The mechanism being remedied, and its limits

### 1.1 Established

By static reading of the committed source at `74dca6e`:

1. `server/mcts.js:178-183` — every `_expand()` runs a `// Create children (unexpanded)` loop
   over **every legal move**, calling `node.state.applyMove(move)` and constructing an
   `MCTSNode` for each.
2. `server/gameLogic.js:377` — `applyMove()` calls `this.copy()`.
3. `server/gameLogic.js:172-181` — `copy()` deep-copies the peg map and bridge set
   (`new Map(this.pegs)`, `new Set(this.bridges)`).

One search performs 1 root expansion plus up to `nSimulations` leaf expansions. At plies 40–60
there are roughly 470–490 legal moves, so an 800-simulation search materializes on the order of
`801 × ~480 ≈ 385,000` deep-copied states, all reachable from the live root for the duration of
that search. This is the right order of magnitude for the 4,080 MB observed at failure.

**It is peak retention inside a single search, not a leak across moves or games.** `search()`
allocates a fresh root per call and returns only a `Map` and a number; `playGame()` drops its two
`MCTS` objects per game; `AlphaZeroInference.evaluate()` does not accumulate on the inference
instance.

### 1.2 Explicitly NOT established

**Additional contributors are not excluded.** No heap snapshot was taken. The arithmetic above is
an order-of-magnitude argument, not a measurement of what occupied the heap. Other objects
outlive an individual game (the opening pool, the timing schedule, the harness's written-file and
evidence lists), and the `priors` Map returned by `evaluate()` is retained by each expanding node
and forms part of the same per-search peak.

**Consequence for this design:** §6's verification is a *measurement against a preregistered
criterion*, not a formality. Removing eager expansion is necessary; this document does not assume
it is sufficient. If §6 fails, §9 applies.

---

## 2. The remedy — lazy child-state materialization

### 2.1 Shape

Keep `node.priors` covering **every** legal move, exactly as today. Change `node.children` from
"every legal move, pre-materialized" to "only those moves PUCT has actually selected". A child
`TwixtState` is constructed the first time selection descends into that move, and never
otherwise.

### 2.2 Why this can be exactly equivalent, not approximately

This is the load-bearing argument and the reason a cheaper-looking approximation is not needed.

`_selectChild` (`server/mcts.js:189-227`) scores each child as:

```
q     = child.visitCount > 0 ? -child.qValue : 0
u     = (cPuct * prior * sqrt(node.visitCount + 1)) / (1 + child.visitCount)
score = q + u
```

For an **unvisited** child, `visitCount === 0`, so `q === 0` and `u === cPuct * prior *
sqrtParent`. Both terms are derivable from `node.priors` alone. **`child.state` is never read
while scoring.** An unmaterialized child and a materialized-but-unvisited child therefore produce
bit-identical scores, and selection is unchanged.

Materialization is required only at the moment descent enters the child — precisely when the
current code would first read its `state`.

### 2.3 Invariants the implementation must preserve

| # | invariant | why it is at risk |
|---|---|---|
| I1 | **Iteration order** over candidate moves is the insertion order of `node.priors`, which is the order `legalMoves()` returned | today `_selectChild` iterates `node.children`; iterating a different collection can reorder, and the tie-break is order-sensitive |
| I2 | **Tie-break** stays `moveKey < bestMove`, a **lexicographic string** comparison on `"row,col"` — *not* numeric (`"10,5" < "9,3"` is true) | a "cleanup" to numeric ordering silently changes played moves |
| I3 | `score > bestScore` stays **strictly** greater, with equality falling through to I2 | `>=` would keep the last tied move instead of the lexicographically smallest |
| I4 | **Simulation count** unchanged: 1 root expansion + `nSimulations` iterations | a lazy path must not consume an iteration to materialize |
| I5 | **Root visit-count output includes every legal move, zero-count entries included** | today `visitCounts` is built from `root.children`, which holds all legal moves; built from materialized children alone it would silently drop zeros |
| I6 | **Abort semantics** unchanged: abort after root expand returns `{ visitCounts: new Map(), rootValue: 0 }`; a mid-loop abort `break`s and reports partial counts | these are distinct observable behaviours |
| I7 | **Progress callback** unchanged: `{done, total, elapsed, valueEstimate}` with `valueEstimate` clamped from `root.valueSum / root.visitCount` | |
| I8 | **Terminal-leaf handling** unchanged: terminal nodes take the explicit ±1/0 value and are not expanded | |
| I9 | `_backup` unchanged — it walks `searchPath`, never `parent` | |
| I10 | `selectMove` / `selectMoveDeterministic` untouched | `nSims`, `cPuct`, readout and models are out of scope by §7 |

**I5 is the invariant most likely to be missed and most consequential.** `server/index.js:92`
iterates `out.visitCounts` into the `visits` payload that is cached and returned to the client, so
the zero-count entries are observable outside MCTS, not an internal detail.

### 2.4 Intended blast radius

`server/mcts.js` only. `server/gameLogic.js` is **not** expected to change: `copy()` is correct,
it is simply called far too often. Any change beyond `server/mcts.js` widens the execution
surface being altered and requires re-review before implementation continues.

---

## 3. Structural bound

**Criterion.** Over one `search(rootState, {nSimulations: S})`, the number of `TwixtState`
objects retained by the search tree must be **≤ `1 + S`**, not `≈ (1 + S) × L` for `L` legal
moves.

The bound follows from the descent rule rather than from tuning. The selection loop runs
`while (node.isExpanded && !node.state.isTerminal())`. A newly materialized child is by
construction unexpanded, so descent stops on it. **At most one new child is materialized per
simulation**, plus the root. `1 + S` is therefore a structural ceiling, not an empirical average.

At `S = 800` this is ≤ 801 retained states against the current ~385,000 — a reduction of roughly
480×, the same factor as the legal-move count, which is what makes the default heap plausible
again.

"Approximately" in the authorization is honoured as: **≤ `1 + S` exactly** for the tree, with any
excess to be explained and re-reviewed rather than absorbed into a tolerance.

---

## 4. Behavioural equivalence — fixtures and golden traces

### 4.1 What is captured, and from where

Golden traces are captured from the **unmodified eager implementation at `74dca6e`**, before any
edit, and committed. The lazy implementation must reproduce them.

There is no `server/test_mcts.js` today — MCTS is exercised only indirectly by
`server/test_parity.js` and `server/test_server.js` — so this harness is new work, not an
extension of an existing one.

### 4.2 Fixture set

| axis | values |
|---|---|
| positions | ≥ 12 drawn from the four committed timing sidecars (`timing_failures/74dca6e/`), spanning opening / early-mid / midgame plies, plus ≥ 1 near-terminal and ≥ 1 position with a forced win available |
| model | baseline `1d64027db521a50f` (candidate `c34b7ff3297c785a` for ≥ 2 positions, to prove the trace is not model-specific) |
| `nSimulations` | `1, 2, 8, 64, 800` |
| `cPuct` | `1.5` (product value; not varied) |
| readout | `temperature = 0` only — `selectMove` above `0.01` calls `Math.random()` (`mcts.js:289`) and is not reproducible |

Small `nSimulations` values are included deliberately: divergence in selection order shows up at
simulation 2–8, where a 800-simulation aggregate could mask it.

**The positions are derived from already-committed evidence**, so the fixture set is fixed by
this document and cannot be re-drawn after seeing a result.

### 4.3 Recorded per fixture

- the **complete** `visitCounts` Map — every key in iteration order with its count, **including
  every zero-count legal move** (I5);
- `rootValue`;
- the move `selectMoveDeterministic(visitCounts)` returns;
- the ordered sequence of `onProgress` payloads at `progressEvery: 1`, which exposes
  `root.visitCount` and `valueEstimate` after every simulation and is the cheapest available
  proxy for per-simulation descent order (I1–I3);
- for the abort fixtures: the exact return value for (a) abort before the first simulation and
  (b) abort mid-loop (I6).

### 4.4 Acceptance

**Exact equality on every field of every fixture.** Counts are integers and `rootValue` is a
deterministic float from identical arithmetic on identical inputs, so this is byte equality, not
tolerance-based. Any "close enough" comparison is a failure of the test, not a pass of the
remedy.

---

## 5. Falsification

**Requirement:** a test that **fails against the eager implementation** and passes only if
allocation scales with simulations rather than with simulations × legal moves. A test that passes
both is worthless — it would be the "gate that does not bind" this project keeps finding.

**Design.** In a test-local scope, count `TwixtState` construction by spying on
`TwixtState.prototype.copy` (the sole path — `applyMove` → `copy`, §1.1). No production
instrumentation, no counter shipped in `server/`.

```
position with L legal moves ≥ 400
S = 8 simulations
assert copyCount <= 2 * (1 + S)          # i.e. <= 18
```

| implementation | `copyCount` | outcome |
|---|---:|---|
| eager (`74dca6e`) | ≈ `(1 + 8) × ~480 ≈ 4,320` | **FAILS**, by ~240× |
| lazy | ≤ `9` | passes |

`S = 8` keeps it to nine network evaluations, so it is cheap enough for the ordinary suite while
still separating the two implementations by more than two orders of magnitude.

**The falsification must be demonstrated to fail on the pre-change code before the change is
made** — run it against `74dca6e`, record the failure, then implement. A falsification first
observed after the fix is not evidence that it binds.

---

## 6. Default-heap verification, with the criterion fixed now

**Procedure.** One `search()` at `nSimulations: 800`, `cPuct: 1.5`, baseline model, product ORT
configuration, **Node's default heap — no `--max-old-space-size`**, from a midgame fixture
position (≥ 400 legal moves).

**Preregistered criteria, both required:**

| # | criterion |
|---|---|
| M1 | the search **completes** without OOM at the default heap |
| M2 | peak `process.memoryUsage().heapUsed` sampled during the search is **≤ 512 MB** |

`512 MB` is a ceiling chosen to sit decisively below the 4,080 MB failure and comfortably above
the ~801 retained states plus their `priors` Maps that §3 predicts. **It is not a tuned target**,
and it may not be revised after a measurement. Missing M2 while passing M1 is a §9 stop, not a
pass — it would mean retention is bounded by something other than the mechanism in §2.

The measured peak is reported whatever it is.

---

## 7. Required checks beyond equivalence

| area | requirement |
|---|---|
| regression | `npm run test:server`, `npm run test:match`, `npm run test:parity` and `npm test` all green; existing tests unmodified — a test edited to accommodate the change is a §9 stop unless the edit is itself reviewed as a deliberate contract change |
| determinism | each fixture produces identical output across repeated calls in one process **and** across separate process invocations |
| product stack | the server path still returns a `visits` payload containing **every legal move** (`server/index.js:92`), including zeros (I5); `server/test_server.js` and `server/test_readout_policy.js` green |
| readout seam | `selectMoveForRequest` remains the only route to `selectMove`; `server/test_readout_policy.js`'s direct-call ban still passes |
| resumption | the harness's pair-level resume still reproduces a sidecar **exactly** on replay. **A run started before the remedy may not be resumed after it** — the §10 run fingerprint pins `execution_commit`, and a clean restart at a different commit is a new run, not a resume |
| unchanged by construction | `nSimulations` 800, `cPuct` 1.5, ORT configuration, `readout_policy.js`, both model artifacts, the opening pool, and the entire statistical design of the comparison specification |

---

## 8. Consequence for the execution surface and for `P`

`server/mcts.js` is one of the ten files in `EXECUTION_SURFACE_FILES`
(`tests/product_match/p_decision.mjs:77-88`). **Any remedy moves the execution-surface digest off
`228f57b55448f44136ffd41d6f092c9da904ca469a1e7bc4055656ffd8ef77bd`.** That is correct and
intended, and it has three consequences:

1. **The ten-game timing smoke must be run again**, in full, under its own fresh authorization,
   against the new surface. The four sidecars in `timing_failures/74dca6e/` are a failure record;
   they are not partial timing evidence and may never be completed by a later run.
2. **`P` may not be derived from anything measured before the change.** `selected_p` remains
   `none` until a complete ten-game smoke succeeds at the new commit.
3. The remedy also changes **timing itself** — lazy materialization removes ~385,000 deep copies
   per search, so per-search cost will differ from the 4.74 / 5.01 / 5.14 / 5.02 s observed at
   `74dca6e`. Throughput must be re-measured, never extrapolated. (Note those figures use
   `ply_count − 4` searches, since `elapsed_ms` starts after the opening replay,
   `tests/product_match/harness.mjs:157`.)

---

## 9. Stop conditions

Any of the following **stops the work and returns it for review**. None may be resolved by
adjusting the criterion, the fixture set or the tolerance:

- any golden-trace mismatch on any fixture, including a difference confined to zero-count entries
  or to move ordering;
- the falsification in §5 passing against the eager implementation at `74dca6e`;
- the §3 bound exceeded;
- M1 or M2 in §6 not met — including M2 missed while M1 passes, which indicates a contributor §1.2
  did not exclude;
- any regression, determinism, product-stack or resumption check in §7 failing;
- a change required outside `server/mcts.js` (§2.4);
- an existing test needing modification to pass.

Explicitly **not** permitted as responses: raising `--max-old-space-size`, lowering
`nSimulations`, altering `cPuct`, relaxing the equivalence comparison to a tolerance, re-drawing
the fixture set, or narrowing §6 to a smaller position.

---

## 10. What this document authorizes

**Nothing.** Not the implementation, not the fixture or golden-trace capture (which requires
running the current code), not a profiler, not a heap snapshot, not a heap-limit change, not the
timing smoke, not `P` selection, not the match, not deployment, not promotion, not any change to
`DEFAULT_MODEL_ID`, not training.

Each step needs its own explicit authorization. The earliest is capturing the §4 golden traces
from the unmodified `74dca6e`, which is a *measurement* step and separate from implementing the
change in §2.
