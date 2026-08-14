# Product-Stack Comparison Specification — DRAFT FOR REVIEW

**Date:** 2026-08-14 · **Branch:** `codex/product-model-alignment-phase2`
**Basis:** `bdbbeca` (pushed Phase 2 head)
**Status: DRAFT. NOT PREREGISTERED YET, and it authorizes no games.**

Phase 3 of `docs/superpowers/2026-08-13-product-model-alignment-decision-memo.md`, whose branch
was chosen as **stronger shipped play**. That choice set the evidence standard; this document
proposes the design that would meet it.

**Nothing here is authorized.** No game may be played until this specification is committed,
separately reviewed, and explicitly approved — and two prerequisites in §3 must be built and
reviewed before a match is even possible. On approval this document becomes preregistered and
its numbers stop being editable.

---

## 1. The question, and the question this is not

**Asked:** in the product's own stack — Node ONNX Runtime, `server/mcts.js`, the shipped readout
policy — does the candidate play measurably stronger than the artifact currently served?

**Not asked, and not answerable here:** whether either model is good in absolute terms. Both
sides are internal to this project. There is still **no external strength anchor**, so this
produces a relative result only.

**Why the research record cannot answer it.** `calib020_0001` beat `0379` by about `+80` Elo,
but that was measured under the Python/MLX search stack, against a different opponent, at 400
simulations. The served baseline's provenance is **unknown**; the audit excluded
`calib020_0001` conclusively and identified `alphazero-v2-staged/model_iter_0193` only
circumstantially. So the delta that would matter to a user — *this served artifact* versus
*this candidate*, under *this* search implementation — has never been measured.

**Phase 2 does not answer it either.** Parity established that the candidate's export is
numerically faithful to its native checkpoint. That is a correctness result about one model; it
says nothing about which model plays better.

## 2. The two artifacts

Both are committed, content-addressed, and load through the same validated path.

| | baseline | candidate |
|---|---|---|
| `model_id` | `1d64027db521a50f` | `c34b7ff3297c785a` |
| graph sha256 | `f1b4411a…` | `9df19e08…` |
| data sha256 | `111546445e…` | `fc1ffaac…` |
| source | **unknown** (audit: not `calib020_0001`; `0193` circumstantial only) | `calib020_0001`, SHA-1 `209cf2d4…` |
| parity measured | never | **PASS**, 126 positions (`bdbbeca`) |
| currently served | **yes** (`DEFAULT_MODEL_ID`) | no |

Neither file is modified by this work. `DEFAULT_MODEL_ID` is not touched by the match; changing
it would be a separate, later, reviewed action.

## 3. Prerequisites — both must be built and reviewed BEFORE any match

These are not part of the match. They are code that does not exist yet, and each needs its own
review.

### 3.1 A product-stack match harness (does not exist)

**Nothing in this repository plays a game in the product stack.** The server answers per-move
requests; there is no self-play or match loop on the Node side at all. The research harness is
MLX/Python and cannot drive ONNX Runtime.

The harness must: load both models through `resolveModel`/`MODEL_MANIFEST`; alternate two `MCTS`
instances over one `TwixtState` sequence; apply the shipped `readout_policy.js` for both sides;
terminate on win or a preregistered ply cap; and write one JSON sidecar per game recording the
move list, both `model_id`s, colours, simulation count, readout temperature, termination reason,
and ply count. It must fail loud rather than substitute a model, and it must never write to
either model directory.

### 3.2 Seeded RNG injection (blocks the medium arm entirely)

`server/mcts.js:289` calls bare `Math.random()` inside `selectMove`'s temperature branch, so
**medium play is not reproducible today**. A medium result produced now could not be
regenerated, audited, or replicated.

Required: an injectable RNG on `MCTS`, defaulting to `Math.random` so shipped behaviour is
**byte-identical when unseeded** — the pattern the research programme used for `fpu_value`
(opt-in, default preserves behaviour exactly). Acceptance requires a test proving the default
path is unchanged, and a test proving that a fixed seed reproduces an identical game.

**Until §3.2 ships and is reviewed, the medium arm in §5.2 may not run.** The hard arm does not
depend on it.

## 4. Openings — why they are mandatory, and how they are built

**Hard play is fully deterministic.** `readout_policy.js` sets `hard` to `moveTemp: 0`;
`server/mcts.js:6` states there is no Dirichlet noise, and the only `Math.random()` is in the
stochastic branch, which `moveTemp: 0` never reaches. Selection is argmax with a lexicographic
tie-break.

**Consequence:** a given (starting position, colour assignment) yields *exactly one* game,
forever. Playing "400 games" from the empty board would play **one** game 400 times. Diverse
openings are not a refinement; without them the match has a sample size of one.

**Construction.** A committed JSON file of opening move sequences, generated once by seeded
pseudo-random legal play and then frozen — the same shape as the parity corpus, which is
regenerable byte-identically and independently verifiable.

| property | value |
|---|---|
| opening depth | 4 plies (2 moves per side) |
| count | `P` openings (see §7) |
| uniqueness | canonical key over sorted pegs, sorted bridges, side to move |
| exclusions | terminal, or fewer than 2 legal moves |
| PRNG | mulberry32, seed recorded, every rejected seed recorded |

**Pairing.** Every opening is played **twice**: once with the candidate as red, once as black,
from the identical position. So each opening contributes a matched pair, and board-colour
advantage cancels **by construction**. This is why the retired absolute per-colour `0.50` veto
(2026-08-10 erratum) is not reintroduced: per-colour splits will be reported as **descriptive
only**.

**Depth 4 is a judgement call.** Deep enough to diverge, shallow enough that positions remain
near-balanced. Deeper openings would add diversity but increasingly decide the game before the
engines play. Recorded as a threat in §10.

## 5. Arms

### 5.1 Arm A — hard, primary

The strongest configuration the product offers, and the cleanest signal: deterministic, so every
game is exactly reproducible from its opening and colour assignment.

| parameter | value |
|---|---|
| difficulty | `hard` — `nSims: 800`, `moveTemp: 0` |
| identical for both sides | simulation count, `cPuct` 1.5, readout, ply cap |
| ply cap | 400, recorded as a non-decisive termination |

### 5.2 Arm B — medium, secondary, GATED on §3.2

`medium` is `DEFAULT_DIFFICULTY`, so it is what most users actually meet — but
`readout_policy.js` says outright that easy and medium **sacrifice strength by design**
(`moveTemp: 0.5` at every ply). A model difference can therefore be swamped by sampling noise
that is deliberate.

Arm B answers a *different* question — "does the default experience change?" — and its result
may not be substituted for Arm A's. It runs only after §3.2 ships, with seeds recorded per game.

**If Arm A and Arm B disagree, that is a finding to report, not a conflict to resolve by picking
the friendlier arm.**

## 6. Statistic and decision rule

### 6.1 Primary statistic — pair-level, not per-game

Each opening contributes one observation: the candidate's share of its two games, in
`{0, 0.5, 1}`.

Per-game binomial intervals would be **wrong here**: the two games in a pair share an opening and
are not independent, so a per-game interval understates uncertainty. Pair-level observations are
i.i.d. across openings, which is the assumption a confidence interval actually needs.

Reported: mean pair score, its 95% interval, the win/draw/loss split over pairs, and — as
descriptive context only — the raw per-game record and per-colour split.

### 6.2 Decision rule, fixed in advance

| outcome | decision |
|---|---|
| pair-level 95% **lower** bound `> 0.50` | **candidate is stronger.** Eligible for a separate, reviewed switch of `DEFAULT_MODEL_ID`. Not automatic. |
| interval **contains** `0.50` | **unresolved.** Keep the baseline. |
| pair-level 95% **upper** bound `< 0.50` | **candidate is weaker.** Keep the baseline; do not switch on provenance grounds either. |

**An unresolved result does not authorize a larger match.** Buying resolution after seeing the
interval is a post-hoc power increase — do-not-repeat `#51` closed a research line for exactly
that reason, and the same discipline applies here.

**A pass is eligibility, not a switch.** Deployment stays a separate reviewed action with
before/after hashes, backup, rollback and a startup check.

### 6.3 What a null would leave behind, stated honestly

If the result is unresolved, the product keeps serving an artifact **whose provenance is
unknown** while a fully-provenanced, parity-verified candidate sits committed beside it. That is
an uncomfortable end state, and it is the correct one under the chosen branch: *stronger shipped
play* means switching on demonstrated strength, and a null is not demonstrated strength.

Switching on provenance hygiene instead would be a **different justification** requiring its own
authorization. It is not licensed by this document, and it must not be smuggled in as a
consolation reading of a null.

## 7. Sample size and cost — the binding constraint

**Measured on this machine** (baseline model, single 800-simulation search, `server/mcts.js`):

| configuration | per search | notes |
|---|---:|---|
| `intraOpNumThreads` 6 (default) | **4.41 s** | ORT already parallelises: 28.4 s CPU / 4.7 s wall |
| `intraOpNumThreads` 2 | 9.70 s | |
| `intraOpNumThreads` 1 | 17.92 s | |

Machine: 12 logical cores, **6 performance** + 6 efficiency.

**ORT already saturates the performance cores**, so running more game processes does *not*
multiply throughput. Because thread scaling is sublinear, many-processes-few-threads is
modestly better in principle — 6 × 1-thread ≈ 0.33 searches/s versus 1 × 6-thread ≈ 0.23 — but
that ignores contention, so treat ~1.5× as an upper bound, not a plan.

**Derived cost**, at ~62 plies per game (the fp6 measured average) and ~3.3 s per move amortized:

| pairs `P` | games | estimated wall clock |
|---:|---:|---:|
| 100 | 200 | ~11 h |
| 200 | 400 | ~23 h |
| 400 | 800 | ~46 h |

**Proposal: `P = 200` (400 games), Arm A only, run overnight across two sessions.**

**Power, stated honestly.** At `P = 200` the worst-case 95% half-width is about `0.069`, so the
design can detect a pair-level score of roughly `0.57` or better. It is powered to detect a
**large** difference — which is the plausible case, since the candidate is the best-supported
research checkpoint and the baseline is probably ~200 iterations earlier in a different line. It
is **not** powered to resolve a near-tie, and per §6.2 a near-tie will be reported as unresolved
and left there.

**Throughput smoke, required before `P` is fixed.** Ten games measured end to end, reporting
games/hour under the chosen process/thread configuration. If measured throughput misses this
estimate by more than 1.5×, `P` drops to 100 — a decision made from **timing only, before any
game is scored**, which cannot bias the outcome.

**A futility screen is deliberately NOT included.** Candidate 2's 64-game screen produced an
adverse `28–36` that did not reproduce at 800 games, so a small screen here would mostly buy
noise at real cost.

## 8. Integrity checks

Any failure aborts the match; none may be repaired after the fact.

- Every game sidecar records both `model_id`s; any game not showing exactly
  `1d64027db521a50f` and `c34b7ff3297c785a` invalidates the run.
- Both models load through `resolveModel`, so hashes, external-data binding and the application
  contract are verified per process.
- The two sides share simulation count, `cPuct`, readout policy and ply cap; the **only**
  difference is the model.
- Colour assignment is derived from the opening index, not drawn, and the 50/50 split is
  asserted before analysis.
- Execution commit recorded and the worktree clean at launch.
- Baseline and candidate file hashes re-verified after the run, proving neither was touched.

## 9. Procedure order

1. Commit this specification; obtain separate review and explicit approval.
2. Build and review the §3.1 harness. **No games.**
3. Build and review the §3.2 seeded RNG. Required for Arm B only.
4. Generate and commit the openings file; verify its constraints.
5. Run the §7 throughput smoke; fix `P` from timing alone.
6. Run Arm A. Analyse strictly per §6.
7. Report. Arm B only if §3.2 shipped and Arm A is reported first.

## 10. Threats to validity, acknowledged in advance

- **Random openings are not human openings.** Both engines face identical positions, so the
  comparison is fair, but external validity to real play is an assumption, not a measurement.
- **No external anchor.** A relative result only; it cannot say either model is good.
- **One machine, CPU only.** Different hardware or an ORT GPU provider is a different surface.
- **Hard is not the default.** Arm A measures the strongest configuration; the default is
  medium, which Arm B addresses only if §3.2 ships.
- **Underpowered for a near-tie**, by design and by cost. Stated up front so a null is read as
  "not resolved", never as "equal".
- **The baseline's identity remains unknown.** A result is about *the served bytes*, not about
  any named checkpoint.

## 11. What this document authorizes

**Nothing.** Not the harness, not the RNG work, not the openings file, not the smoke, not a
game, not a switch of `DEFAULT_MODEL_ID`, not deployment, not training, not research-seed use.

Each numbered step in §9 needs its own approval, and the earliest of them is step 2.
