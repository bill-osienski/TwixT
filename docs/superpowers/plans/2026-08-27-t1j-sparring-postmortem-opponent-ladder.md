# T1j Sparring Postmortem and External-Opponent Ladder Plan

**Status:** PLAN ONLY. This document authorizes no execution, model load, JVM,
inference, new game, seed use, training, checkpoint selection, or push.

**Amendment 1** (2026-08-27, plan-only). This document supersedes the original as
received — sha256 `1f0d0046f2a16276b1665775136c5c74952d76881fb98c62c0bf4f86924615b0`,
retained unmodified outside the repository. Nine corrections were applied at their
points of use after a review of the components the plan names; each is marked
**[A1]** in place and listed in §11. No file outside this document changed, and D0
remains unstarted.

**Goal:** Turn the already completed 64-game T1j match into a disciplined source
of hypotheses about `calib020_0001`, test only hypotheses that repeat on held-out
games, and—only if one survives—design a separately authorized improvement. T1j
becomes the first development opponent in a growing external-opponent ladder.

**Current facts:**

- The canonical match is immutable historical evidence: 64 games, 2,216 bound
  plies, zero engine-state divergences, T1j score `38/64 = 0.594`, with both
  reported intervals containing parity.
- T1j is independently implemented and uses classical alpha-beta search. It is
  useful precisely because it does not share our model lineage.
- The durable match record contains positions, moves, outcomes, openings,
  colours and task identities. It does **not** contain our complete raw policy,
  MCTS visit distribution or value trajectory at every position, and it does
  not expose a T1j policy distribution. Those facts must not be reconstructed
  from the winner.
- The R0 research decision is `NO_GO`; `calib020_0001` remains incumbent. This
  plan does not reopen training by itself.
- **[A1]** The local R1 report **has already been corrected** to the bounded
  statement “the surveyed sources produced no qualifying corpus.” The amendment
  is unpushed at local `HEAD`; `origin/main` does not yet carry it. Its withdrawn
  universal SGF/π claims are not a premise of this plan.

---

## 1. Role change, stated explicitly

T1j was initially kept pristine as an external strength anchor. This plan uses
it as a **development sparring partner** instead.

That choice has consequences:

1. The original 64-game result remains valid historical evidence because it
   predates any T1j-informed development.
2. Once a model or mechanism is selected using T1j positions or moves, future
   T1j results are no longer untouched external validation. They measure
   performance against a known development opponent.
3. Held-out T1j openings and games can still test generalization *within T1j*,
   but cannot restore full independence.
4. T1j must never be described as ground truth. Its move is an independently
   generated alternative, not proof that our move was wrong.
5. A later independently built opponent becomes the next external test. Every
   earlier opponent stays in the regression ladder.

This is an intentional trade: for a niche game, extracting useful disagreement
from a qualified opponent is more valuable than preserving the only opponent as
a permanently untouched referee.

---

## 2. Programme shape

| Phase | Question | Effectful work permitted by this plan? | Possible result |
|---|---|---:|---|
| P0 | Is the record and scope clean enough to begin? | no | ready / blocked |
| D0 | What patterns exist in the recorded 64 games? | no model/JVM/games | bounded inventory |
| D1 | What do both systems prefer from the same positions? | separately authorized only | disagreement dataset |
| D2 | Does a candidate weakness repeat on held-out games? | separately authorized only | `GO` / `NO_GO` |
| D3 | Is there a new, ledger-distinct intervention? | design only | training card / close |
| T0 | Does the intervention pass its cheap falsifier? | separately authorized only | run / reject |
| T1 | Does a trained candidate improve? | separately authorized only | promote / reject |
| O1 | Is another external opponent ready to join? | separately authorized survey | add / reject |

No phase inherits authorization from the phase before it.

---

## 3. P0 — close the record before analysis

- [ ] **[A1] VERIFY — do not re-amend —** that R1 says only what its bounded
  survey established: confirm SGF’s standard TwixT profile defining no π property
  is no longer presented as proof that SGF or every external corpus cannot carry
  π. The correction already stands, unpushed, at local `HEAD`. Amending a second
  time would rewrite a correction rather than check it, and would leave no record
  of which version P0 actually verified.
- [ ] Preserve the canonical L0 match directory byte-for-byte.
- [ ] **[A1]** Bind the analysis to the published match JSONL and its manifest by
  digest **before any reconstruction reads a move**. The record is self-binding:
  `run_header` carries `plan_sha256` and `task_digest` alongside the whole frozen
  plan, so the opening prefix §4 replays is trustworthy only once those digests
  are checked. This is a precondition of §4, not a parallel task.
- [ ] Record the exact repository commit, checkpoint hashes, T1j JAR/JDK hashes,
  and the 64 canonical task identities.
- [ ] Reconcile the role change in the ledger: “T1j development opponent after
  L0,” while leaving the pre-development L0 result untouched.

**P0 stop:** any missing or altered canonical artifact blocks the programme.

---

## 4. D0 — zero-inference postmortem of the existing match

### 4.1 Scope

D0 reads the existing JSONL only. It may reconstruct our `TwixtState` from the
recorded openings and moves to compute deterministic board facts.

**[A1] Where the moves are.** The `ply` records begin at ply 7 in all
64 tasks; the 6 opening plies are **not** in the ply stream. The
sequence D0 replays is the embedded frozen plan's opening prefix
(`run_header.identity.plan.plan.openings[opening]`) followed by the recorded `ply`
rows in order. This reconstruction is verified consistent in 64 of
64 tasks by `opening_bound.ply + len(ply records) == task_result.plies`,
and is legitimate only under the digest binding required by §3.

D0 must not:

- load `calib020_0001`;
- launch Java or query T1j;
- generate a move or game;
- draw a seed;
- infer a counterfactual result;
- call a move “bad” solely because the mover later lost;
- **[A1]** import `eval_loss_replay_analysis`, or otherwise adopt D1 telemetry
  vocabulary.

**[A1] Why that last one is a scope rule, not a style rule.** That module's
features are `root_value`, `selected_visit_rank` and `root_top1_share` — exactly
the observables this record does not contain and §4.4 forbids D0 from claiming.
Importing the vocabulary is how a forbidden claim gets made by accident. Its
feature-agnostic arithmetic (`phase_of`, `cohens_d`, `effect_sizes`) may be reused
only if lifted free of the telemetry features; the module itself belongs to D1,
whose §5.2 requires that visit rank, value perspective and policy alignment keep a
single definition.

### 4.2 Freeze a discovery/confirmation split before inspecting diagnostics

The 64 tasks contain four repetitions in each of 16 opening/colour cells.

- **Discovery:** repetitions `0` and `1` — 32 games, two per cell.
- **Confirmation:** repetitions `2` and `3` — 32 games, two per cell.

D0 may inventory both halves for integrity and outcome counts, but all
hypothesis formation uses discovery games only. Confirmation-game diagnostic
features remain unopened until D2 has frozen a hypothesis and test.

### 4.3 Required D0 outputs

For every ply in the discovery half, derive only facts available from the
recorded move sequence and rules engine:

- task, opening, colour arm, repetition, mover and eventual winner;
- board ply, legal-move count and remaining empty holes;
- peg and bridge counts by colour;
- connected-component counts and largest component size by colour;
- **[A1]** minimum boundary distance of each component to its target sides — **a
  new derivation, with no existing helper**. Tensor channels 19–22 are per-cell
  geometric edge distance (`1.0 - r / max_idx`), and `connectivity_masks` reports
  only whether a component touches a goal, giving no distance for one that touches
  neither. Define it once, in D0, and state in the definition whether the metric
  is graph distance or geometric;
- newly created bridges, blocked bridge opportunities and immediate wins;
- **[A1]** whether the move created, answered or ignored a one-ply terminal
  threat — **cost must be measured before this feature is assumed cheap**. It
  needs an `apply_move` + `winner` over every legal move at every ply, across
  1,137 recorded discovery plies (1,329 including opening
  prefixes) against several hundred legal moves in the early game. Measure on a
  small sample first and record a stop rule; if measured cost exceeds it, narrow
  the feature explicitly rather than dropping it silently;
- distance from the terminal ply.

Aggregate by opening, colour arm, winner and coarse game phase. Every aggregate
must retain its denominator; per-opening and per-colour results are descriptive.

### 4.4 What D0 may conclude

D0 may say that a structural pattern **recurs**. It may not assign the pattern to
policy, value or search, because those observables were not captured.

Examples of legitimate D0 hypotheses:

- losses repeatedly follow an unanswered immediate threat;
- one colour accumulates disconnected local components late in losses;
- losses concentrate after bridge-blocking contact positions;
- the losing side reaches terminal positions with many locally plausible moves.

Examples of forbidden D0 claims:

- “the policy missed the winning move”;
- “the value head was overconfident”;
- “MCTS was too shallow”;
- “T1j’s move was objectively better.”

### 4.5 D0 gate

`GO` to D1 only if at least one precisely defined structural signature:

1. appears in more than one opening;
2. appears in both colour arms or is explicitly scoped as colour-specific;
3. can be computed identically on the held-out half; and
4. maps to a named observable that D1 can measure from both systems.

Otherwise D0 returns `NO_GO`: the existing match provides a score but no
actionable repeated weakness.

---

## 5. D1 — same-position interrogation, only after a separate authorization

D1 does not play games. It replays a frozen set of existing positions and asks
both systems what they would do from the **same** state.

### 5.1 Position selection

- Select from discovery games using a rule frozen from D0, never by looking at
  model/T1j answers.
- Include matched controls from the same opening, colour and phase where the D0
  signature is absent.
- **[A1]** Carry a deterministic ordered move prefix with every selected
  position. The E3b adapter advances T1j only by replaying an ordered sequence
  through `setlastMove()`; it cannot convert a bare `TwixtState`. Deduplicate
  identical positions by a canonical state digest, but **the digest is a
  deduplication label, never replay input** — collapsing distinct move orders onto
  one state discards the only thing the adapter can consume. Retain exactly one
  canonical prefix per surviving digest, chosen by a rule frozen with the
  selection rule.
- Fix a hard query budget before any model or JVM load.
- Use a newly registered diagnostic seed interval for our search/readout. Never
  reuse L0’s retired seeds.

### 5.2 Capture from our system

At the exact incumbent configuration (`calib020_0001`, 400 simulations,
noise off), persist:

- chosen move;
- raw legal-move policy;
- 400-simulation root visit distribution;
- root value from the side-to-move perspective;
- selected move’s raw-policy and visit ranks;
- top children with visit count and root-perspective Q;
- exact evaluator, MCTS configuration and RNG identities.

Reuse existing replay/telemetry machinery where its contract fits; do not create
a second definition of visit rank, value perspective or policy alignment.

### 5.3 Capture from T1j

At the qualified fixed depths `3` and `6`, persist:

- selected move;
- requested and completed depth;
- legality and searched-position dump;
- exact reflection/postcondition surface;
- whether depth 3 and depth 6 agree.

T1j supplies no policy distribution. Its single selected move must be recorded
as exactly that—not expanded into a synthetic π.

### 5.4 Per-position comparisons

- exact move agreement among our move, T1j-depth-3 and T1j-depth-6;
- rank and mass assigned by our raw policy to each T1j move;
- rank and visit share assigned by our search to each T1j move;
- whether our search promotes or suppresses the raw-policy leader;
- root-value trajectory along the recorded continuation;
- the D0 structural signature and matched-control label.

These are disagreement measurements, not move-quality labels.

### 5.5 D1 integrity

- Reuse the E3b binder for every replayed position.
- Abort on the first state, legality, history, terminal or postcondition mismatch.
- One evaluator instance; compilation enabled; no rebuilding per query.
- Append-only, exclusive-create, flushed and fsynced records.
- No training file is emitted from D1.
- **[A1]** Pass `ply_cap` explicitly at every call. `play_task` declares
  `ply_cap: int = PLY_CAP` with `PLY_CAP = 280`, so an omitted cap is **silently
  defaulted, not refused** — the hazard is silence, not absence. The no-default
  protections sit further down the stack, in `t1j_adapter.replay`,
  `t1j_adapter.terminal_with_cap` and `T1jRuntime.__init__`, each of which refuses
  a missing cap; a caller that stops at `play_task` never reaches them.

---

## 6. D2 — falsify the weakness on held-out games

Before opening confirmation diagnostics, freeze:

1. one primary weakness hypothesis;
2. one primary metric and direction;
3. eligibility and exclusion rules;
4. the minimum effect considered practically actionable;
5. a power/precision calculation appropriate to that metric;
6. missing-data and integrity-abort handling.

Run the identical feature/query pipeline on repetitions `2` and `3` without
changing the hypothesis.

### D2 outcomes

- **`GO`:** the preregistered signature repeats in the stated direction, with
  the required coverage and integrity checks.
- **`NO_GO`:** it does not repeat, is too small to act on, or only survives by
  changing the cohort, metric or interpretation.
- **`VOID`:** instrumentation or identity failed. A void licenses repair of the
  instrument only, never reinterpretation of the observed data.

Only a D2 `GO` may open an intervention design.

---

## 7. D3 — choose the smallest intervention that matches the confirmed cause

The intervention must be selected from the measured failure mode, not from a
pre-existing wishlist. It must be checked by name against do-not-repeat
`#1–#52` and R0.

Possible intervention families, each requiring its own new card:

### A. Search or readout defect

Use when the raw policy contains the alternative but search systematically
suppresses it, or when a depth/horizon signature repeats.

- Cheap falsifier: replay the frozen diagnostic positions under exactly one
  search change.
- Training data: none initially.
- Promotion still requires a separate match; replay success is not strength.

### B. Value defect

Use when preregistered value trajectories are confidently wrong in a repeated,
held-out structural cohort.

- Cheap falsifier: value-only evaluation on frozen positions and legal
  continuations.
- Any calibration/training proposal must be genuinely distinct from the closed
  value-calibration families.

### C. Policy blind spot

Use when the strong T1j alternative repeatedly has negligible incumbent policy
mass and the signature survives holdout.

Two possible labels must remain distinct:

- **T1j move as one-hot expert action:** behavioural cloning, not AlphaZero π.
- **Our own fresh MCTS visit distribution on the position:** AlphaZero-style π,
  generated by our search and labelled as such—not attributed to T1j.

Training on T1j actions formally retires T1j as an untouched anchor. The card
must say so and must include safeguards against learning only T1j’s style.

**[A1]** This remains a **deferred, explicit decision, taken at D3 and nowhere
else**. Because it changes T1j's standing status, it must never be reached by
inheritance from an earlier phase. D0 does not take it, does not depend on it, and
does not presuppose its outcome.

### D. Position-distribution gap

Use when T1j cross-play reaches a repeated state family rarely represented in
our data, without a single move-level defect.

- First response: generate a bounded diagnostic corpus, not a training run.
- Prefer T1j-versus-our-model cross-play over T1j self-play because cross-play
  directly exposes interaction failures.
- T1j self-play is secondary and descriptive; it primarily samples T1j’s own
  style.
- **[A1]** Any cross-play starting from an **empty board** must separately close
  the unseeded-opening issue first. T1j's `InitialMoves.firstMove()` selects from a
  seven-entry table using an unseeded `Random`, and is reached only below `moveNr`
  6; every qualified game so far started at ply 6 precisely to avoid it, so the
  path is **bypassed, not disproved**. Neither D0 nor D1 exercises it — D1 replays
  positions from games that already began at ply 6 — so neither may be cited as
  evidence about it.

### D3 gate

The card must name:

- the confirmed D2 result;
- the ledger entries it is and is not;
- the cheap falsifier;
- the required data and how labels are produced;
- compute cost, stop rule and contamination consequences;
- a minimum meaningful effect plus a benchmark sized for power before training.

If no intervention satisfies those conditions, return `NO_GO` and keep the
incumbent.

---

## 8. Training and evaluation, if later authorized

### T0 — cheap falsifier

Run the smallest test capable of disproving the mechanism. No rescue grid, dose
change, additional cohort or post-hoc threshold. A failed falsifier closes that
intervention.

### T1 — bounded training

Only after T0 passes:

- freeze the exact training corpus, labels, checkpoint initialization, optimizer,
  step budget and selection rule;
- preserve an incumbent control;
- train one candidate family under a preregistered stop;
- never select using the final held-out T1j benchmark.

### T2 — promotion evaluation

Use three distinct surfaces:

1. **Internal regression:** candidate versus incumbent and all established
   product/engine correctness gates.
2. **T1j development test:** unseen openings/seeds, both colours, paired incumbent
   and candidate measurements. This tests improvement against the known sparring
   partner, not untouched external validity. **[A1]** If any of these games starts
   from an empty board rather than a scripted opening, §7-D's unseeded-opening
   precondition applies here too, and must be closed before the surface is scored.
3. **Opponent-ladder regression:** every previously admitted opponent, under its
   frozen protocol, to catch style-specific regressions.

“Mastered T1j” must be defined before T2 by a rate/effect target and a powered
sample size. It is not one winning match, a point estimate above parity or a
post-hoc claim based on the most favorable opening subset.

---

## 9. Adding the next external opponent

After a candidate clears T1j, survey for the next opponent. It need not be
AlphaZero; algorithmic diversity is desirable.

### Admission gates

1. **Identity:** pinned source/artifact/license and reproducible build/runtime.
2. **Rules:** exact match, or an explicit adapter with state equivalence proven
   ply-by-ply. A semantic rules conversion is not silently accepted.
3. **Automation:** headless, bounded, durable and fail-closed.
4. **Determinism:** measured at the selected setting, or randomness explicitly
   controlled and recorded.
5. **Strength dial:** at least one usable setting that is neither trivial nor
   saturated against the incumbent.
6. **State binder:** pegs, links, legal moves, side, ply, history and terminal
   winner agree with our engine.
7. **Frozen benchmark:** openings, colours, seeds, sample size, scoring and stop
   rules published before play.

### Candidate order

- A different classical engine is useful even if it is not neural.
- A neural/MCTS system such as `twixtbot` is attractive for style diversity, but
  its crossing/swap rules must be reconciled before it can enter the ladder.
- Human games can inform qualitative review, but are not an executable opponent
  and do not provide AlphaZero π.

Each admitted opponent is frozen as a new ladder rung. It is never removed merely
because a later model learns to beat it.

---

## 10. Immediate next authorization

The next authorization should cover **D0 only**:

- read the published 64-game JSONL;
- build a pure deterministic postmortem over the discovery half;
- inventory, reconstruct and verify both halves without opening confirmation
  diagnostics;
- write tests and a docs/evidence package;
- commit locally, do not push;
- no model, JVM, T1j query, inference, new game, seed or training;
- **[A1]** read the §3–§8 corrections as written. D0 inherits no authorization
  from Amendment 1, which changed no file but this one.

The D0 result may be `NO_GO`. That is a successful outcome if the recorded match
does not contain a repeated, measurable weakness.

---

## 11. Amendment 1 — the nine corrections

Applied 2026-08-27, plan-only. Each is marked **[A1]** at its point of use;
this list is an audit surface, not the authority. Where they disagree, the
point-of-use text governs.

| # | Correction | Section |
|---|---|---|
| 1 | P0 **verifies** the corrected R1; it does not amend it again | Current facts, §3 |
| 2 | `play_task` defaults `ply_cap` to `PLY_CAP = 280` — the hazard is silent defaulting, not absence; the no-default refusals are in the adapter/runtime paths | §5.5 |
| 3 | D0 reconstructs from the embedded frozen plan's opening prefix **plus** recorded plies, with digest binding established first | §3, §4.1 |
| 4 | Per-component boundary distance is a **new derivation**; no existing helper supplies it | §4.3 |
| 5 | One-ply terminal-threat detection needs a **measured** cost/stop check before being assumed cheap | §4.3 |
| 6 | D0 must not import D1 telemetry vocabulary or `eval_loss_replay_analysis` | §4.1 |
| 7 | D1 positions retain a deterministic ordered move prefix; a state digest is a deduplication label, not replay input | §5.1 |
| 8 | Empty-board cross-play must separately close the unseeded-opening issue; D0 and D1 do not exercise it | §7-D, §8-T2 |
| 9 | Training on T1j actions stays a deferred, explicit **D3** decision that changes T1j's status; D0 does not take it | §7-C |

**Derived at generation, not typed:** 64 tasks; ply streams begin at ply
7; 6 opening plies per game; reconstruction consistent in
64/64 tasks; discovery half 32 games / 1,137
recorded plies (1,329 with prefixes).
