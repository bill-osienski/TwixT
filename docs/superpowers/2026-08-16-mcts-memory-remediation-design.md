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

That product is an **estimate, not a formula**: it assumes a uniform legal-move count, whereas
deeper expansions face fewer legal moves and terminal leaves are never expanded at all. It is
used to show the mechanism is large enough to explain the observed heap, never as a quantity to
be checked against. §5 states separately what is structurally guaranteed.

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
objects retained by the search tree must be **≤ `1 + S`** — a quantity that does **not** scale
with the legal-move count `L`, as the current implementation's does.

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

### 4.2 The corpus, frozen exactly

"At least 12 positions across four strata" does not determine a corpus — it leaves a reviewer
free to redraw after seeing an inconvenient result while still satisfying the document. The exact
positions, matrix and count are therefore fixed here.

**Positions.** Each is a **prefix of the `moves` array of a named committed sidecar**, replayed
through `TwixtState` from the empty board. Prefix rule, applied uniformly:
`{4, 16, 28, ply_count − 1}`.

| id | sidecar (in `tests/product_match/timing_failures/74dca6e/`) | prefix plies |
|---|---|---:|
| P01–P04 | `timing_00_opening_200.json` (`ply_count` 39) | 4, 16, 28, **38** |
| P05–P08 | `timing_01_opening_201.json` (`ply_count` 51) | 4, 16, 28, **50** |
| P09–P12 | `timing_02_opening_202.json` (`ply_count` 57) | 4, 16, 28, **56** |
| P13–P16 | `timing_03_opening_203.json` (`ply_count` 54) | 4, 16, 28, **53** |

**16 positions.** Ply 4 is the opening itself, 16 early-mid, 28 midgame, `ply_count − 1`
near-terminal.

**"An immediate winning move exists" — mechanically, not by judgement.** Every sidecar records
`termination: "win"`, and the four games have been replayed legally to their recorded terminal
results. Therefore at prefix `ply_count − 1` the side to move has a move that immediately wins:
the move actually recorded at that ply. This is derivable from committed evidence without running
anything, and it replaces the earlier unverifiable "forced win available". **P04, P08, P12 and
P16 are the immediate-win positions**, and they are also the near-terminal stratum.

**Execution matrix, frozen:**

| model | positions | `nSimulations` | cases |
|---|---|---|---:|
| baseline `1d64027db521a50f` | all 16 | `1, 2, 8, 64, 800` | 80 |
| candidate `c34b7ff3297c785a` | **P02 and P11 only** | `1, 2, 8, 64, 800` | 10 |

`cPuct` is `1.5` and is not varied. Readout is `temperature = 0` only — `selectMove` above `0.01`
calls `Math.random()` (`server/mcts.js:289`) and is not reproducible.

Small `nSimulations` values are deliberate: a selection-order divergence shows at simulation 2–8,
where an 800-simulation aggregate can mask it.

**Abort fixtures, with exact trigger points** (both on **P07**, baseline, `nSimulations: 64`):

| id | trigger | expected |
|---|---|---|
| A1 | the `AbortSignal` is **already aborted** when `search()` is called | returns `{ visitCounts: new Map(), rootValue: 0 }` after the root expansion |
| A2 | abort fired from the `onProgress` callback at `progressEvery: 1` when **`done === 5`** | loop `break`s; partial counts returned |

**Expected fixture count: exactly 92** — `80 + 10 + 2`. A capture producing any other number is a
§9 stop, not a corpus to be adjusted.

### 4.3 Recorded per fixture, and what equality means

**Compared for exact equality:**

- the **complete** `visitCounts` Map — every key in iteration order with its count, **including
  every zero-count legal move** (I5);
- `rootValue`;
- the move `selectMoveDeterministic(visitCounts)` returns;
- the ordered sequence of `onProgress` payloads at `progressEvery: 1`, restricted to
  **`done`, `total` and `valueEstimate`** — this exposes `root.visitCount` and the running value
  after every simulation, and is the cheapest available proxy for per-simulation descent order
  (I1–I3);
- for A1 and A2, the exact return value (I6).

**Recorded but NOT compared for equality: `elapsed`.** `search()` derives it from `Date.now()`
(`server/mcts.js:73`, emitted as `elapsed: now - t0` at `server/mcts.js:142`), so it is
wall-clock and cannot reproduce across runs. **Requiring it would
make a correct implementation fail**, which is a defect in the test, not in the code. It is
checked only as metadata: present, a finite number, `≥ 0`, and non-decreasing across the payload
sequence within a run. Nothing else about it is asserted.

If exact reproduction of `elapsed` is ever wanted, it needs an injected clock seam added to
`search()` **before both captures** — not a post-hoc relaxation of the comparison.

### 4.4 Acceptance

**Exact equality on every compared field of all 92 fixtures.** Counts are integers and
`rootValue` is a deterministic float produced by identical arithmetic on identical inputs, so this
is exact equality, not tolerance-based. Any "close enough" comparison on a compared field is a
failure of the test, not a pass of the remedy.

### 4.5 Capture protocol

**Each of the 92 cases is captured in a freshly spawned process that runs that one case and
exits.** The eager implementation is the one whose per-search retention is the problem, so
running several cases in one process would let one case's heap pressure contaminate the next —
and would reproduce, in the capture harness, the very confound that makes the original failure
hard to attribute.

**The naive instruction "run at `74dca6e` with a clean worktree" is impossible, and saying so is
the point of this subsection.** At `74dca6e` neither the capture harness (new work, §4.1) nor
these timing-failure sidecars (committed later, at `7510660`) exists. A clean worktree at that
commit cannot execute this corpus by itself. The protocol must therefore name *two* things: the
code whose behaviour is captured, and the commit supplying the harness and fixtures.

**Frozen protocol — clean descendant with a byte-identical execution surface:**

1. The capture harness and any fixture helpers are **committed and reviewed first**, as their own
   step, on a descendant of `74dca6e`.
2. The capture runs from that descendant, with a **clean worktree**.
3. **Before any case is captured**, the runner asserts
   `executionSurfaceDigest(HEAD) === 228f57b55448f44136ffd41d6f092c9da904ca469a1e7bc4055656ffd8ef77bd`.
   Because that digest is taken over the ten execution-surface blobs, equality *is* the statement
   "every execution-surface file is byte-identical to `74dca6e`" — it is not a proxy for it.
4. Both the capture commit and the pinned surface digest are recorded in every captured artifact.

This works precisely because the capture harness is **not** an execution-surface file: adding it
changes `HEAD` without changing the digest, exactly as the evidence and design commits already
did. If a future need ever forces the surface itself to differ, capture from that commit is
invalid regardless, and the fallback applies: import the execution code from a **separate clean
`74dca6e` worktree** while reading fixtures from the harness commit, recording both commits.

**Consequence for sequencing:** building and reviewing the capture harness is a **separate,
earlier gate** than any measurement. Measurement cannot be authorized first — there would be
nothing to run it with. §10 is written accordingly.

**A single eager 800-simulation search is expected to be capturable**: the four completed timing
games performed `35 + 47 + 53 + 50 = 185` such searches without OOM, so the per-search peak fits
the default heap on its own. That expectation is not a guarantee — see §9.

---

## 5. Falsification

**Requirement:** a test that **fails against the eager implementation** and passes only if
allocation scales with simulations rather than with simulations × legal moves. A test that passes
both is worthless — it would be the "gate that does not bind" this project keeps finding.

**Design.** In a test-local scope, count `TwixtState` construction by spying on
`TwixtState.prototype.copy` — the sole path, since `applyMove` → `copy` (§1.1). No production
instrumentation, no counter shipped in `server/`.

**Frozen protocol.** Every element below is fixed so the test cannot be loosened into passing:

| element | frozen value |
|---|---|
| position | **P11** — `timing_02_opening_202.json`, prefix **28** plies |
| model | baseline `1d64027db521a50f` |
| `S` | **8** simulations, `cPuct` 1.5 |
| spy installed | **after** the fixture state is fully constructed (prefix replay complete), **before** `search()` is called |
| spy removed | immediately after `search()` returns |
| counted | every `TwixtState.prototype.copy` invocation while the spy is installed, and nothing else |

Replaying the 28-ply prefix itself performs copies; those are excluded by construction because
the spy is installed afterwards. The root state is supplied by the fixture, so `search()` inherits
it without a copy.

**Threshold — the bound the algorithm actually claims:**

```
assert copyCount <= S            # i.e. <= 8
```

Not `2 × (1 + S)`. Under §3 the root is not copied and each simulation materializes **at most
one** child, so `S` is the exact ceiling; a bound of 18 would silently permit more than twice the
copies the design claims and would still pass an implementation that materializes a spare child
per simulation. Every copy the implementation performs must be one this bound accounts for; any
excess must be explained and re-reviewed, not absorbed.

**What the eager side is claimed to do — the sufficient claim, not a formula.** `(1 + S) × L` is
**not** structural and is not asserted here: deeper expansions face different legal-move counts,
and terminal leaves are never expanded at all, so no single `L` characterises the whole search.

The preregistered claim is confined to what is structurally guaranteed, and it is already more
than enough:

> The **root expansion alone** copies once per legal move at P11. Legal moves on this board
> satisfy `n_legal ≥ 528 − ply`, so at ply 28 the root contributes **≥ 500** copies, before any
> simulation runs.

| implementation | `copyCount` | outcome |
|---|---:|---|
| eager (`74dca6e`) | **≥ 500** from the root expansion alone | **FAILS** the `≤ 8` gate by **≥ 62.5×** |
| lazy | ≤ `8` | passes |

Copies from the eight subsequent expansions are **reported as measured evidence** and are
expected to add several thousand more, but the gate does not depend on them and no fixed
multiplier is claimed. The actual `L` at P11 and the total `copyCount` are both recorded at
capture time.

`S = 8` keeps the test to at most nine network evaluations, cheap enough for the ordinary suite
while separating the two implementations by more than an order of magnitude on the guaranteed
part alone.

**The falsification must be demonstrated to fail on the pre-change code before the change is
made** — run it against `74dca6e`, record the failure, then implement. A falsification first
observed after the fix is not evidence that it binds.

---

## 6. Default-heap verification, with the criterion fixed now

**Procedure.** One `search()` at `nSimulations: 800`, `cPuct: 1.5`, baseline model
`1d64027db521a50f`, product ORT configuration, **Node's default heap — no
`--max-old-space-size`**, from position **P11** (`timing_02_opening_202.json`, prefix 28), in a
**freshly spawned process that performs this one search and exits**.

**Measurement protocol, frozen.** "Sampled during the search" is not reproducible: a timer cannot
observe the peak, because the expansion and selection loops are synchronous and block the event
loop for the interval that matters. The measurement is therefore taken at **named seams**, all
test-local, with **no production instrumentation**:

| # | seam | how it is reached without touching `server/` |
|---|---|---|
| H1 | immediately before `search()` is called | test code |
| H2 | immediately **before** and **after** every `evaluate()` call | the `inference` object handed to `MCTS` is a **test-local proxy** that samples, delegates to the real `AlphaZeroInference`, samples again, and returns its result unchanged |
| H3 | at every `onProgress` callback, `progressEvery: 1` | the callback is test-supplied |
| H4 | immediately after `search()` returns | test code |

Each observation is `process.memoryUsage().heapUsed`. The reported figure is the **maximum
observed `heapUsed` across H1–H4**, and the protocol is stated in those terms rather than as a
"peak", which the process cannot actually observe.

**What each seam does and does not bracket.** H2 does **not** bracket eager expansion, and an
earlier draft claimed it did. In `_expand()`, `evaluate()` is awaited at `server/mcts.js:173` and
returns *before* the `// Create children (unexpanded)` loop at `:178-183` — so **both** H2 samples
fall on the near side of that expansion's allocations. Stated correctly:

| seam | what it actually observes |
|---|---|
| H2 | the **inference envelope** — the tensors and `priors` Map allocated per evaluation — plus any **lazy materialization performed during descent** before that evaluation |
| H3 | **retained state after a completed expansion**, since the progress callback fires at the end of each simulation. This is the first named seam positioned after the child-creation loop |

The H1–H4 protocol stands; only that justification was wrong. Under the lazy design H2 and H3
converge anyway, because at most one child is materialized per simulation and there is no
expansion-time allocation burst left to miss.

**Evaluation count: up to `1 + S` = up to 801**, not exactly 801. A simulation reaching a
terminal leaf takes the explicit ±1/0 value and never calls `_expand()`, so it performs no
evaluation (I8).

**Preregistered criteria, both required:**

| # | criterion |
|---|---|
| M1 | the search **completes** without OOM at the default heap |
| M2 | the maximum observed `heapUsed` across H1–H4 is **≤ 512 MB** |

`512 MB` is a ceiling chosen to sit decisively below the 4,080 MB failure and comfortably above
the ~801 retained states plus their `priors` Maps that §3 predicts. **It is not a tuned target**,
and it may not be revised after a measurement. Missing M2 while passing M1 is a §9 stop, not a
pass — it would mean retention is bounded by something other than the mechanism in §2, which §1.2
does not exclude.

The maximum observed value is reported whatever it is, together with `L` at P11 and the
observation count.

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
- **any eager golden capture failing or OOMing.** Preserve that failure as evidence and stop. Do
  **not** substitute a different position, drop the case, shorten the prefix, lower
  `nSimulations`, or re-run it with a larger heap. An eager capture that cannot complete at the
  default heap is itself a finding about the mechanism in §1 and must be reported as one, not
  routed around. The corpus is 92 cases (§4.2); a capture yielding any other number stops here
  too;
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

Each step needs its own explicit authorization, and §4.5 changes what the earliest one is:

1. **Build and review the capture harness** — construction only, no measurement. This must come
   first, because a clean worktree at `74dca6e` contains neither the harness nor the fixtures and
   so cannot run the corpus at all.
2. **Capture the 92 golden cases** from the pinned execution surface (§4.5) — a measurement step,
   separate from step 1 and from step 3.
3. **Implement the §2 change**, then satisfy §§3–7.

Authorizing measurement before step 1 would authorize running something that does not exist.
