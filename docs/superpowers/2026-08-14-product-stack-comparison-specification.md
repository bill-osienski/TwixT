# Product-Stack Comparison Specification — DRAFT FOR REVIEW (rev 2)

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

Requirements: load both models through `resolveModel`/`MODEL_MANIFEST`; alternate two `MCTS`
instances over one `TwixtState` sequence; use the shipped `readout_policy.js` for both sides;
terminate **only** via `state.isTerminal()`; write one atomic JSON sidecar per game; fail loud
rather than substitute a model; never write to either model directory.

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

**Confidence interval — primary:** two-sided 95% **percentile bootstrap** over pairs,
**10,000 resamples**, resampling whole pairs with replacement, RNG mulberry32 seeded
`20260814`. Chosen because pair scores are discrete, bounded and possibly skewed, where a normal
approximation is least trustworthy. The seed makes the interval exactly reproducible.

**Confidence interval — secondary cross-check:** Student `t` interval,
`s̄ ± t₀.₉₇₅,ₚ₋₁ · sd(s)/√P`. Reported alongside. **If the two methods disagree on the decision,
that is a reported finding and the outcome is treated as unresolved** — not an invitation to
pick the friendlier one.

The observed `sd(s)` is reported, since the §7 power arithmetic assumes a worst case.

### 6.2 Decision rule

| outcome | decision |
|---|---|
| bootstrap 95% **lower** bound `> 0.50` (and the `t` interval agrees) | **candidate stronger.** Eligible for a separate, reviewed switch. Not automatic. |
| interval contains `0.50`, or the two methods disagree | **unresolved.** Keep the baseline. |
| bootstrap 95% **upper** bound `< 0.50` | **candidate weaker.** Keep the baseline. |

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

**Resolution threshold.** With worst-case `sd = 0.5` at `P = 200`, the 95% half-width is
`1.96 × 0.5/√200 ≈ 0.069`, so an **observed** `s̄ ≥ 0.569` would clear the bar. This is a
property of the data once seen, **not** a power statement.

**Power.** For 80% power at one-sided `α = 0.025`,
`δ = √((1.96 + 0.842)² · 0.25 / 200) ≈ 0.099` — a **true** score of about **`0.60`**.

At a true `0.57`, power at `P = 200` is only about **50%**: a coin flip. `P = 200` is powered to
detect a **large** difference, which is the plausible case given the candidate is the
best-supported research checkpoint and the baseline is probably ~200 iterations earlier in a
different line. It is **not** powered to resolve a modest one, and per §6.2 a near-tie will be
reported unresolved and left there.

Both figures use worst-case `sd = 0.5`. Pairing and draws will likely give a smaller `sd`,
improving both; the observed value is reported and **does not retroactively change the rule**.

### 7.3 Timing smoke, and how `P` is fixed

Before the match: **10 games** on the reserved openings `200…209` — 5 baseline-versus-baseline
and 5 candidate-versus-candidate.

**Outcome-blind by construction:** a model playing itself yields no comparative information, so
`P` cannot be chosen with any knowledge of the matchup.

| measured throughput | `P` |
|---|---:|
| **≥ 8.8 games/hour** | 200 |
| **< 8.8 games/hour** | 100 |

`8.8` is the estimate degraded by 1.5×. The choice is made from **timing alone, before any match
game is played**, and is recorded before the match starts.

**No futility screen.** Candidate 2's 64-game screen produced an adverse `28–36` that did not
reproduce at 800 games, so a small screen here would mostly buy noise at real cost.

## 8. Execution, interruption and resumption

A ~30-hour run will be interrupted. Determinism is what makes that safe: hard play is fully
deterministic, so a resumed run produces **exactly** what an uninterrupted one would.

- **Fixed order.** Pairs run in opening-index order `0…P-1`; within a pair, candidate-as-red
  then candidate-as-black. The order is a property of the opening file, not of the run.
- **Atomic sidecars.** Each game is written to a temporary file and renamed, so a sidecar is
  never partially observed.
- **Resume unit is the pair.** On restart, pairs with both sidecars are complete and skipped; a
  pair with one sidecar has that sidecar **discarded and the pair replayed in full**. A pair is
  the statistical unit and must never be half-counted.
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
- both models load through `resolveModel`, so external-data binding and the application contract
  are checked;
- every game sidecar records both `model_id`s; any game not showing exactly
  `1d64027db521a50f` and `c34b7ff3297c785a` invalidates the run;
- both sides share simulations, `cPuct`, readout policy and termination — the **only**
  difference is the model;
- colour assignment is derived from the opening index, never drawn; the 50/50 split is asserted
  before analysis;
- execution commit recorded, worktree clean at launch;
- after the run, both model pairs re-hashed to prove neither was touched.

## 10. Procedure order

1. Commit this specification; obtain separate review and explicit approval.
2. Build and review the §3 harness. **No games.**
3. Generate and commit the 210-opening pool; verify its constraints.
4. Run the §7.3 timing smoke; fix `P` from timing alone and record it.
5. Run Arm A.
6. Analyse strictly per §6; report.

## 11. Threats to validity

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

## 12. What this document authorizes

**Nothing.** Not the harness, not the opening pool, not the timing smoke, not a game, not a
switch of `DEFAULT_MODEL_ID`, not deployment, not training, not research-seed use. Each step in
§10 needs its own approval, and the earliest is step 2.
