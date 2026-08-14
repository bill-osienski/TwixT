# Product-Stack Comparison Specification — DRAFT FOR REVIEW (rev 6)

**Date:** 2026-08-14 · **Basis:** `bdbbeca` (closed Phase 2 head)
**Status: DRAFT. NOT PREREGISTERED, and it authorizes nothing.**

Phase 3 of `docs/superpowers/2026-08-13-product-model-alignment-decision-memo.md`, whose branch
was chosen as **stronger shipped play**.

**Scope: the hard-difficulty arm only.** Medium is deferred to its own specification (§5.2).

> ## Revision 2 — 2026-08-14, before any approval, harness, opening or game
>
> | # | change | reason |
> |---|---|---|
> | 1 | §6.1 now fixes game scoring, the pair formula, the CI procedure and pair win/draw/loss | Draws make a pair score `{0, .25, .5, .75, 1}`, not `{0, .5, 1}`, and rev 1 never named a CI method at all. |
> | 2 | §7 separates **resolution threshold** from **power** | Rev 1 called `0.069` a power figure. It is a worst-case CI half-width. At `P=200` a true `0.57` has ~50% chance of clearing the bar; 80% power needs ~`0.60`. |
> | 3 | §5.1 terminates through `state.isTerminal()`; the invented 400-ply cap is gone, and ORT runs at the product's own configuration | `MAX_PLIES = 600` is the product's draw rule and feeds tensor channel 23 (`ply/600`). A 400-ply cap plays a **different game**. `AlphaZeroInference.load()` passes no session options, so thread tuning would not be the product. |
> | 4 | §4/§9 fix a 210-opening pool with the seed named here; timing uses 10 reserved openings, outcome-blind | Rev 1 generated openings after `P` was known and let the smoke play the real matchup, which could reveal comparative strength before `P` was locked. |
> | 5 | Arm B deferred to a separate specification, with its promotion consequence stated | Rev 1 required seeded RNG "before any match", then said hard did not need it, then proposed Arm A only — three statements that could not all hold. |
> | 6 | §8 adds deterministic resumption, and separates integrity failure from interruption | Rev 1 said "across two sessions" and "any failure aborts" in the same document. |
>
> Rev 1 is preserved in git history at `654d191`.
>
> ## Revision 3 — 2026-08-14, still before any approval, harness, opening or game
>
> | # | change | reason |
> |---|---|---|
> | 1 | §3 and §9 require `assertSessionContract` per model; §9's claim that `resolveModel` checks the contract is **withdrawn** | **Factual error.** `resolveModel` returns at `model_manifest.js:643` having validated manifest, hashes, sizes and external-data binding — it never touches the tensor contract. The product calls `assertSessionContract` **separately** after `inference.load()` (`server/index.js:598`). A harness copying rev 2 would have skipped a check the product performs. |
> | 2 | §6.1 fixes the percentile convention and the resampling index algorithm; §6.2 requires `t` agreement on **both** directional branches; §7.2 relabels `0.569` a planning threshold and adds the `P=100` characteristics | A seed and a replicate count do not define endpoints. Requiring cross-method agreement only for "stronger" was asymmetric — it made a harm finding easier to declare than a benefit. |
> | 3 | §7.3 gives timing its own output namespace, freezes one-process sequential execution, fixes the opening mapping, and requires `P` committed before the first match game | The smoke plays self-play games, which §9 said invalidate the run — the two rules contradicted. Throughput also depends on a concurrency choice rev 2 never fixed. |
> | 4 | New §10 freezes the evidence schema; §8 replaces "discard" with **quarantine and verify** | Rev 1's explicit field list was lost in rev 2. And silently deleting the only surviving evidence of a half-finished pair destroys exactly what would prove a nondeterminism defect. |
>
> Rev 2 is preserved at `190aad6`.
>
> ## Revision 4 — 2026-08-14, still before any approval, harness, opening or game
>
> | # | change | reason |
> |---|---|---|
> | 1 | §10 cardinality restated per **two sidecars per pair** | Rev 3's rule was **impossible to satisfy**: it required each `pair_index` exactly once when a pair emits two games. `opening_id` uniqueness likewise had to be stated across pairs, not across sidecars. |
> | 2 | §10 requires the analyser to **re-derive** result, termination, ply count and `candidate_score`, and to replay `moves` through `TwixtState` | `candidate_score` was consumed on trust, so a self-consistent but wrong sidecar — a mislabelled colour, a stale result — would have entered the statistic unchallenged. The whole comparison rests on that number. |
> | 3 | §6.1 freezes both `t` critical values; §7.2 labels its figures **planning** quantities; §7.3 defines throughput as one wall-clock span | A library's `t` precision, a power figure read as exact for a composite rule, and `elapsed_ms`-sum versus whole-run timing could each move a decision at a boundary. |
> | 4 | §8/§10 place `quarantine/` outside `match/` | A superseded sidecar must not be reachable as evidence. |
>
> Rev 3 is preserved at `07457a2`.
>
> ## Revision 5 — 2026-08-14, still before any approval, harness, opening or game
>
> | # | change | reason |
> |---|---|---|
> | 1 | §10 requires `opening_id === pair_index` and a distinct opening set of **exactly `0…P-1`** | Rev 4 accepted "`P` unique openings from `0…199`", which permits `100…199` at `P=100` — the right *size*, the wrong *set*. That is a selected sample wearing a cardinality rule. |
> | 2 | §10 adds a **run fingerprint** invariant across every sidecar and every resume | Nothing stopped a `~30`-hour job being restarted at a different `execution_commit` and finishing a match whose halves were played by different code. |
> | 3 | `candidate_score`, `ort_config` and the timing span reworded | The `candidate_score` entry still said the analyser "never re-derives", contradicting the re-derivation §10 gained in rev 4. `{}` misdescribes a one-argument call as an empty-but-present options object. "First move to last move" left search and sidecar write ambiguously inside or outside the span. |
>
> Rev 4 is preserved at `6f16de9`.
>
> ## Revision 6 — 2026-08-14, still before any approval, harness, opening or game
>
> | # | change | reason |
> |---|---|---|
> | 1 | The §10 fingerprint uses colour-independent **role** fields `baseline_model_id` / `candidate_model_id`, added to the schema; colour assignment is validated separately | Rev 5's fingerprint said "both model IDs", but the stored fields are `red_model_id`/`black_model_id`, which **swap by design** between the two games of a pair. Read literally, the invariant rejected **every valid pair** — unsatisfiable, not strict. |
>
> Rev 5 is preserved at `e8e3952`. No statistical, model or measurement change.

---

## 1. The question, and the question this is not

**Asked:** in the product's own stack — Node ONNX Runtime, `server/mcts.js`, the shipped readout
policy — does the candidate play measurably stronger than the artifact currently served, at
**hard** difficulty?

**Not asked:** whether either model is good in absolute terms. Both are internal to this
project; there is still **no external strength anchor**, so this is a relative result only.

**Not asked:** whether the *default* user experience improves. The default is medium, and this
document does not measure it. See §5.2.

**Why the existing record cannot answer it.** `calib020_0001` beat `0379` by about `+80` Elo
under the Python/MLX stack, against a different opponent, at 400 simulations. The served
baseline's provenance is **unknown** — the audit excluded `calib020_0001` conclusively and
identified `model_iter_0193` only circumstantially. Phase 2's parity PASS is a correctness
result about one model's export. None of that says which artifact plays better here.

## 2. The two artifacts

| | baseline | candidate |
|---|---|---|
| `model_id` | `1d64027db521a50f` | `c34b7ff3297c785a` |
| graph sha256 | `f1b4411a…` | `9df19e08…` |
| data sha256 | `111546445e…` | `fc1ffaac…` |
| source | **unknown** | `calib020_0001`, SHA-1 `209cf2d4…` |
| parity | never measured | **PASS**, 126 positions |
| currently served | **yes** | no |

Neither is modified. `DEFAULT_MODEL_ID` is untouched by the match.

## 3. Prerequisite — the match harness (does not exist)

**Nothing in this repository plays a game in the product stack.** The server answers per-move
requests; the research harness is MLX/Python and cannot drive ONNX Runtime. This must be built
and reviewed as its own step.

Requirements: alternate two `MCTS` instances over one `TwixtState` sequence; use the shipped
`readout_policy.js` for both sides; terminate **only** via `state.isTerminal()`; write one
atomic JSON sidecar per game conforming to §10; fail loud rather than substitute a model; never
write to either model directory.

**Model loading must reproduce the product's full startup path, which is two calls, not one:**

1. `resolveModel({ MODEL_MANIFEST: … })` — validates the manifest, both file hashes and sizes,
   and the graph-to-sidecar external-data binding.
2. **`assertSessionContract(manifest, session, inference.maxMoves)`** after `inference.load()`.

`resolveModel` **does not** validate the tensor contract; it returns before any session exists
(`server/model_manifest.js:643`). The product performs the second call separately at
`server/index.js:598`. A harness that stopped at step 1 would run a check *weaker* than the
server it is measuring — so both models get both calls, and either failing aborts.

**Seeded RNG is NOT a prerequisite for this document.** Hard is `moveTemp: 0`, which never
reaches the `Math.random()` branch at `server/mcts.js:289`. It is a prerequisite for medium,
which is why medium is deferred.

## 4. Openings

**Hard play is fully deterministic** — `moveTemp: 0`, no Dirichlet noise, argmax with
lexicographic tie-break. A given (position, colour assignment) yields *exactly one* game,
forever. Playing 400 games from the empty board would play **one** game 400 times. Diverse
openings are structural, not a refinement.

| property | value |
|---|---|
| pool size | **210** openings, generated once and committed |
| match set | openings `0…199` |
| timing set | openings `200…209`, reserved for §7 and **never** used in the match |
| depth | 4 plies (2 moves per side) |
| PRNG | mulberry32, **seed `20260814`**, fixed here |
| uniqueness | canonical key over sorted pegs, sorted bridges, side to move |
| exclusions | terminal, or fewer than 2 legal moves |
| rejections | every rejected seed recorded with its reason |

The pool is generated and committed **before** `P` is known, so `P` selects a prefix of an
already-fixed list and cannot reshape the sample.

**Pairing.** Each opening is played twice from the identical position: once with the candidate
as red, once as black. Board-colour advantage cancels **by construction**, which is why the
retired absolute per-colour `0.50` veto (2026-08-10 erratum) is not reintroduced — per-colour
splits are **descriptive only**.

**Depth 4 is a judgement call**: deep enough to diverge, shallow enough to stay near-balanced.
Recorded as a threat in §11.

## 5. Arms

### 5.1 Arm A — hard, and the whole of this document

| parameter | value |
|---|---|
| difficulty | `hard` — `nSims: 800`, `moveTemp: 0` |
| identical both sides | simulations, `cPuct` 1.5, readout policy, termination |
| termination | **`state.isTerminal()` only** — win, no legal moves, or the product's `MAX_PLIES = 600` forced draw |
| ORT configuration | **exactly the product's**: `InferenceSession.create(path)` with no options, as `server/inference.js:36` does |

No invented ply cap: `MAX_PLIES = 600` is the product's own rule and also feeds tensor channel
23 (`ply / MAX_PLIES`), so a shorter cap would play a different game *and* feed the network a
different phase signal near the end. No thread tuning: the product passes no session options,
so a tuned session would not be the product.

### 5.2 Arm B — medium, DEFERRED to a separate specification

Deferred rather than half-specified. It needs its own sample size, seed schedule, statistic and
decision rule, and it is blocked on seeded RNG injection — `server/mcts.js:289` calls bare
`Math.random()`, so medium is not reproducible today.

**Consequence, stated so it cannot be glossed later:** `medium` is `DEFAULT_DIFFICULTY`, so
**Arm A cannot support any claim about the default user experience.** A hard-arm pass makes the
candidate *eligible*; the separate switch review must decide whether medium evidence is required
before changing the default, and must record that decision. A hard-arm pass is not medium
evidence and may not be presented as such.

## 6. Statistic and decision rule

### 6.1 Definitions, fixed in advance

**Game score**, from the candidate's perspective: win `1.0`, draw `0.5`, loss `0.0`. A draw is
any terminal state with no winner — no legal moves, or the 600-ply forced draw.

**Pair score** = the mean of the candidate's two game scores in that pair, so
`s ∈ {0, 0.25, 0.5, 0.75, 1}`. Five values, not three: a pair containing one draw scores
`0.25` or `0.75`.

**Pair outcome**, for reporting: **win** if `s > 0.5`, **draw** if `s = 0.5`, **loss** if
`s < 0.5`.

**Primary statistic** = the mean pair score `s̄` over the `P` pairs, with pairs as the
independent unit. Per-game binomial intervals are **wrong here**: the two games in a pair share
an opening and are not independent.

**Confidence interval — primary: two-sided 95% percentile bootstrap over pairs.** Chosen because
pair scores are discrete, bounded and possibly skewed, where a normal approximation is least
trustworthy.

A seed and a replicate count do **not** determine the endpoints, so the algorithm is fixed here
completely. Two correct implementations must produce bit-identical bounds.

| element | fixed value |
|---|---|
| replicates `B` | `10000` |
| RNG | mulberry32, seeded `20260814` **once**, drawn as a single continuous stream |
| draw order | replicate `b = 0…B-1`, and within each, `P` indices drawn in order `i = 0…P-1` |
| index formula | `idx = Math.floor(rand() * P)`, clamped to `P-1` should `rand()` ever return exactly `1` |
| replicate statistic | mean of the `P` resampled **pair** scores (whole pairs, never games) |
| ordering | replicate means sorted **ascending**, ties kept, 0-indexed as `r[0…9999]` |
| lower bound | `r[250]` |
| upper bound | `r[9749]` |

The endpoints are order statistics of the sorted replicates — **no interpolation**, no quantile
variant. `r[250]` and `r[9749]` follow from `floor(0.025 × 10000)` and
`ceil(0.975 × 10000) − 1`; they are written literally so no library's default convention can
substitute a different pair.

**Confidence interval — secondary cross-check:** Student `t` interval,
`s̄ ± t₀.₉₇₅,ₚ₋₁ · sd(s)/√P`, with `sd` the sample standard deviation (denominator `P−1`).

Only two `P` values are possible, so both critical values are frozen here rather than left to a
library's precision or a table lookup:

| `P` | `df` | `t₀.₉₇₅` |
|---:|---:|---|
| 100 | 99 | `1.9842169515` |
| 200 | 199 | `1.9719565442` |

Verified independently by Cornish-Fisher expansion to all ten digits shown.

**Both methods must agree on the decision, in either direction.** If they disagree — whichever
way — the outcome is **unresolved** and the disagreement is reported. Requiring agreement only
for "stronger" would make a harm finding easier to declare than a benefit, which is not a
defensible asymmetry.

The observed `sd(s)` is reported, since §7's arithmetic assumes a worst case.

### 6.2 Decision rule

| outcome | decision |
|---|---|
| bootstrap 95% **lower** bound `> 0.50` **and the `t` interval agrees** | **candidate stronger.** Eligible for a separate, reviewed switch. Not automatic. |
| bootstrap 95% **upper** bound `< 0.50` **and the `t` interval agrees** | **candidate weaker.** Keep the baseline. |
| interval contains `0.50`, **or the two methods disagree in either direction** | **unresolved.** Keep the baseline. |

**An unresolved result does not authorize a larger match.** Buying resolution after seeing the
interval is a post-hoc power increase; do-not-repeat `#51` closed a research line for exactly
that.

**A pass is eligibility, not a switch.** Deployment remains a separate reviewed action with
before/after hashes, backup, rollback and a startup check.

### 6.3 What a null leaves behind, stated honestly

An unresolved result leaves the product serving an artifact of **unknown provenance** while a
fully provenanced, parity-verified candidate sits committed beside it. Uncomfortable, and
correct under the chosen branch: *stronger shipped play* switches on demonstrated strength, and
a null is not that.

Switching on provenance hygiene would be a **different justification** needing its own
authorization. It is not licensed here and must not be smuggled in as a consolation reading of a
null.

## 7. Sample size, power and cost

### 7.1 Measured cost, at the product's own configuration

One 800-simulation search, baseline model, `server/mcts.js`, default session options:
**4.41 s**.

For context only — **not a licensed configuration** — the same search takes 9.70 s at
`intraOpNumThreads: 2` and 17.92 s at 1. ORT already parallelises across the 6 performance cores
(28.4 s CPU per 4.7 s wall), so extra processes contend rather than multiply. The match uses the
product's default regardless, because that is what ships.

At the fp6 measured average of ~62 plies per game: **≈ 4.6 min per game**, ≈ **13 games/hour**.

| pairs `P` | games | estimated wall clock |
|---:|---:|---:|
| 100 | 200 | ~15 h |
| **200** | **400** | **~30 h** |

The tail is bounded by the 600-ply forced draw; a pathological game could reach ~44 min. Total
runtime is therefore an estimate, not a guarantee.

### 7.2 Resolution threshold versus power — different quantities

These are different quantities and rev 1 conflated them.

| | `P = 100` | `P = 200` |
|---|---:|---:|
| planning resolution threshold (observed `s̄`) | `0.598` | `0.569` |
| **true** score giving 80% planning power | `0.640` | `0.599` |
| planning power at a **true** `0.57` | **29%** | **51%** |

**Every figure in this table is a normal-approximation planning quantity.** None is the exact
operating characteristic of the §6.2 rule, which is composite — it requires **both** the
bootstrap and the `t` interval to agree — and therefore has power no greater than the weaker of
the two, on an empirical distribution these formulas do not model. The table sizes the run; it
does not predict the decision.

**Resolution threshold** is a normal-approximation **planning** figure: `0.5 + 1.96·sd/√P` at
worst-case `sd = 0.5`. It says roughly where an observed mean would need to land. It is **not** a
power statement, and it is **not a guarantee about the bootstrap rule** in §6.1 — the bootstrap
is computed from the empirical distribution and may place its lower bound above or below this
approximation. Only §6.2, applied to the actual bootstrap output, decides anything.

**Power** is the probability of clearing the bar given a true effect:
`δ = √((z₀.₉₇₅ + z₀.₈)²·sd²/P)`.

Read the table plainly: at `P = 200` a true `0.57` is a **coin flip**, and at `P = 100` it fails
**seven times in ten**. This design detects a **large** difference — the plausible case, since
the candidate is the best-supported research checkpoint and the baseline is probably ~200
iterations earlier in a different line. It cannot resolve a modest one, and per §6.2 a near-tie
is reported unresolved and left there.

All figures use worst-case `sd = 0.5`. Pairing and draws will likely give a smaller `sd`,
improving every column; the observed value is reported and **does not retroactively change the
rule**.

### 7.3 Timing smoke, and how `P` is fixed

Before the match: **10 games**, one per reserved opening, in this exact mapping.

| openings | pairing | games |
|---|---|---:|
| `200…204` | baseline vs baseline | 5 |
| `205…209` | candidate vs candidate | 5 |

**Outcome-blind by construction:** a model playing itself yields no comparative information, so
`P` cannot be chosen with any knowledge of the matchup.

**Separate output namespace.** Timing sidecars are written to `timing/`, never to the match
output directory, and carry `"kind": "timing"`. The match analyser reads **only** `match/` and
**rejects** any sidecar whose `kind` is not `"match"`. This resolves rev 2's contradiction: §9
invalidates a *match* game that does not show both model IDs, and a self-play timing game is not
a match game. Timing evidence can never enter the statistic.

**Execution is frozen for both the smoke and the match: one process, sequential, no
concurrency.** Otherwise measured throughput would describe an unspecified concurrency choice
rather than the configuration the match will actually use. Both run at the product's default ORT
configuration (§5.1).

**Throughput is defined exactly**, so the `8.8` boundary cannot turn on an arithmetic choice:

```
games_per_hour = 10 × 3,600,000 / total_sequential_wall_ms
```

`total_sequential_wall_ms` is **one wall-clock measurement**, from **immediately before the first
MCTS search of game 1** to **the completion of the atomic rename of game 10's sidecar**. Stated
that precisely because "first move to last move" leaves it ambiguous whether search and sidecar
write are inside the span; they are, since the match pays both.

It is **not** a sum of per-game `elapsed_ms`, which would silently exclude inter-game overhead
and could land on the other side of the `8.8` boundary.

**Both sessions are loaded and both contracts asserted before the clock starts**, so one-off
model-load cost is excluded while still being performed.

| measured throughput | `P` |
|---|---:|
| **≥ 8.8 games/hour** | 200 |
| **< 8.8 games/hour** | 100 |

`8.8` is the `13.2` estimate degraded by 1.5×. The choice is made from **timing alone**, and the
resulting `P` — with the measured games/hour and the raw `total_sequential_wall_ms` that produced
it — is **committed to the repository before the first match game is played**. Both `P` branches
have their planning characteristics preregistered in §7.2, so neither is a surprise.

**No futility screen.** Candidate 2's 64-game screen produced an adverse `28–36` that did not
reproduce at 800 games, so a small screen here would mostly buy noise at real cost.

## 8. Execution, interruption and resumption

A ~30-hour run will be interrupted. Determinism is what makes that safe: hard play is fully
deterministic, so a resumed run produces **exactly** what an uninterrupted one would.

- **Fixed order.** Pairs run in opening-index order `0…P-1`; within a pair, candidate-as-red
  then candidate-as-black. The order is a property of the opening file, not of the run.
- **Atomic sidecars.** Each game is written to a temporary file and renamed, so a sidecar is
  never partially observed.
- **Resume unit is the pair.** On restart, pairs with both sidecars are complete and skipped. A
  pair with one sidecar is replayed in full — but the existing sidecar is **moved to
  `quarantine/`, never deleted**, and the replayed game must reproduce it **exactly**: identical
  move sequence, result, termination reason and ply count. `quarantine/` sits **outside**
  `match/` (§10), so a superseded sidecar can never be read as evidence.
- **A replay mismatch is an integrity failure, not a retry.** Hard play is deterministic, so a
  divergence means determinism does not hold — the most important defect this run could
  surface. Deleting the only prior evidence would destroy the proof; the run aborts and both
  copies are kept for diagnosis.
- **Intentional pause is allowed** and is not a defect: stop the process, restart later.
- **Integrity failure is different from interruption.** A model-identity mismatch, hash
  mismatch, contract failure or colour-assignment error **aborts and invalidates the run** — it
  cannot be resumed past. A crash, power loss or deliberate stop is an interruption and is
  resumable, after the §9 checks pass again on restart.

This replaces rev 1's contradictory "across two sessions" plus "any failure aborts; none may be
repaired".

## 9. Integrity checks

Re-verified at every process start, including each resume:

- both model file hashes match their manifests, and the manifests match §2;
- both models pass **both** loading calls — `resolveModel` for hashes and external-data binding,
  **and** `assertSessionContract` after session creation for names, types, shapes and
  `maxMoves`. `resolveModel` alone does **not** check the contract (§3);
- every **match** sidecar (`kind: "match"`) carries the two role fields and a colour assignment
  whose set equals them (§10); any match game not showing exactly `1d64027db521a50f` and
  `c34b7ff3297c785a`, one per colour, invalidates the run. Timing sidecars (`kind: "timing"`) are
  self-play by design, live in a separate namespace, and are never read by the analyser (§7.3);
- both sides share simulations, `cPuct`, readout policy and termination — the **only**
  difference is the model;
- colour assignment is derived from the opening index, never drawn; the 50/50 split is asserted
  before analysis;
- execution commit recorded, worktree clean at launch;
- after the run, both model pairs re-hashed to prove neither was touched.

## 10. Evidence schema — frozen before implementation

Fixed here so the harness is written to a known contract rather than the schema being back-fitted
to whatever the harness happened to emit.

**One sidecar per game**, written atomically (temp file, then rename):

| field | content |
|---|---|
| `kind` | `"match"` or `"timing"` — the analyser reads only `"match"` |
| `schema` | `twixt-product-match/1` |
| `opening_id` | index into the committed pool, `0…209` |
| `opening_sha256` | hash of that opening's move list, binding the game to the frozen pool |
| `pair_index` | `0…P-1`; absent for timing games |
| `game_in_pair` | `0` (candidate as red) or `1` (candidate as black); absent for timing |
| `baseline_model_id`, `candidate_model_id` | the two **roles**, colour-independent and therefore **constant across every sidecar in the run** — `1d64027db521a50f` and `c34b7ff3297c785a` |
| `red_model_id`, `black_model_id` | the **colour assignment**, which by design **swaps** between the two games of a pair |
| `moves` | the complete move sequence, opening included |
| `result` | `"red"`, `"black"` or `"draw"` |
| `candidate_score` | `1.0` / `0.5` / `0.0`. Stored **for legibility only** and **independently recomputed** by the analyser; never consumed on trust |
| `termination` | `"win"`, `"no_legal_moves"` or `"max_plies"` |
| `ply_count` | final ply |
| `n_simulations`, `c_puct`, `move_temp` | `800`, `1.5`, `0` |
| `ort_version`, `ort_config` | `onnxruntime-node` version, and `"no options supplied"` — the product calls `InferenceSession.create(path)` with a single argument, so there is no options object to record, and writing `{}` would misdescribe it as an empty-but-present configuration |
| `execution_commit` | git HEAD, with the worktree asserted clean |
| `elapsed_ms` | for throughput reporting |

**Directory layout.** Three sibling namespaces; `quarantine/` is **outside** the analyser's
input, not a subdirectory of it:

```
<run_dir>/
  match/        kind: "match"    — the ONLY input the analyser reads
  timing/       kind: "timing"   — self-play smoke (§7.3)
  quarantine/   superseded sidecars from resumed half-pairs (§8)
```

**Analyser acceptance, all required.** There are **two sidecars per pair**, so the cardinality
rules are stated per that fact:

- exactly `2P` match sidecars, forming exactly `P` pairs;
- each `pair_index` in `0…P-1` occurs **exactly twice**;
- within a `pair_index`, `game_in_pair` is `0` **once** and `1` **once**;
- the two games of a pair share the **same** `opening_id` and carry **opposite** colour
  assignments;
- for every sidecar, the **set** `{red_model_id, black_model_id}` equals exactly
  `{baseline_model_id, candidate_model_id}` — the two roles, one per colour, neither repeated;
- `game_in_pair` **determines** the assignment, and is checked rather than assumed:
  `0` ⇒ `red_model_id === candidate_model_id`, `1` ⇒ `black_model_id === candidate_model_id`;
- **`opening_id === pair_index` for every sidecar**, and the distinct opening set is **exactly
  `0…P-1`** — not merely `P` unique values drawn from `0…199`;
- every `opening_sha256` matches the committed pool;
- every sidecar carries `kind: "match"`.

**Why the prefix is pinned, not just the count.** "`P` unique openings from `0…199`" would accept
`100…199` at `P=100` — a set of the specified size that is nonetheless **not the frozen prefix**,
and therefore a selected sample. Since `P` is chosen from timing alone (§7.3), the openings it
selects must follow mechanically from `P`, with no freedom left in *which* ones.

**Run fingerprint — one implementation, or no result.** These fields must be **byte-identical
across every match sidecar in the run**, including across every resume:

`execution_commit` · `schema` · `ort_version` · `ort_config` · `n_simulations` · `c_puct` ·
`move_temp` · `baseline_model_id` · `candidate_model_id`

**The fingerprint uses the colour-independent ROLE fields, never `red_model_id`/`black_model_id`.**
Those two swap between the games of a pair by design, so requiring them byte-identical across
sidecars would reject **every valid pair** — the invariant would be unsatisfiable rather than
strict. Colour assignment is validated separately, below.

Any variation invalidates the run. A `~30`-hour job will be restarted, and the tempting failure
is benign-looking: fix an unrelated bug, restart, and finish a match whose first half was played
by different code. **A clean restart at a different `execution_commit` is not a resume — it is a
new run**, and the completed pairs from the old commit may not be reused. The fingerprint is
recorded once when the run starts and re-asserted at every process start; a mismatch aborts
rather than continues.

**The analyser trusts nothing it can re-derive.** For every game it independently:

- **replays `moves` through `TwixtState`**, asserting each move was legal when played;
- checks the first 4 plies equal the named opening, and that `opening_sha256` matches it;
- re-derives `result`, `termination` and `ply_count` from the replayed terminal state and
  requires equality with the stored fields;
- **recomputes `candidate_score`** from the re-derived `result`, the per-colour model IDs and the
  colour assignment, and requires equality with the stored value.

`candidate_score` is stored for legibility, never consumed on trust. Without this, a sidecar that
is internally self-consistent but wrong — a mislabelled colour, a stale result — would enter the
statistic unchallenged, and the whole comparison rests on that one number.

Anything else — a duplicate pair, a stray sidecar, a partial pair, an unexpected `kind`, a
re-derivation mismatch — is a **hard reject of the analysis**, not a row to skip. A run that
cannot present exactly `P` complete, self-verifying, unique pairs has no result.

## 11. Procedure order

1. Commit this specification; obtain separate review and explicit approval.
2. Build and review the §3 harness. **No games.**
3. Generate and commit the 210-opening pool; verify its constraints.
4. Run the §7.3 timing smoke; fix `P` from timing alone and record it.
5. Run Arm A.
6. Analyse strictly per §6; report.

## 12. Threats to validity

- **Random openings are not human openings.** Both engines face identical positions, so the
  comparison is fair, but external validity to real play is an assumption.
- **No external anchor.** Relative result only.
- **One machine, CPU only.** Different hardware or an ORT GPU provider is a different surface.
- **Hard is not the default.** Medium is, and it is not measured here (§5.2).
- **Underpowered for a near-tie**, by design and by cost. A null means "not resolved", never
  "equal".
- **The baseline's identity remains unknown.** The result is about *the served bytes*, not about
  any named checkpoint.
- **Opening depth 4** is a judgement call, not a measured optimum.

## 13. What this document authorizes

**Nothing.** Not the harness, not the opening pool, not the timing smoke, not a game, not a
switch of `DEFAULT_MODEL_ID`, not deployment, not training, not research-seed use. Each step in
§11 needs its own approval, and the earliest is step 2.
