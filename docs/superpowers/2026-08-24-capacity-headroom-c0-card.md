# C0 card — capacity-headroom probe on the existing replay corpus

**Date:** 2026-08-24 · **Status: CLOSED 2026-08-24 — C0 complete, and its result is a
FEASIBILITY NO-GO. The proposed C1 design is UNQUALIFIED and UNAUTHORIZED.**
· **Scope: read-only. Nothing was trained, loaded, inferred or generated.**

Basis: `main` @ `d8ea093`, clean, local == remote.
Predecessor: [`2026-08-22-twixtbot-anchor-pilot-card.md`](2026-08-22-twixtbot-anchor-pilot-card.md),
CLOSED — the external anchor was rejected, and this probe is its recorded fallback.
Governing constraint: `docs/alphazero-value-search-experiment-ledger.md`, `Status: CLOSED`,
do-not-repeat **#1–#52**.

---

## CORRECTIONS — review round 1, 2026-08-24

This card was first written claiming C0 was a feasibility **go** with three narrowings. Review
rejected four claims, all correctly. The corrections are recorded here and applied in place below;
**no evidence file and no line of `capacity_c0_inventory.py` was changed** — they are byte-identical
to what the review verified.

| # | claim as first written | why it was wrong | what replaced it |
|---|---|---|---|
| 1 | "an **authenticated** corpus does exist" | the digest authenticates the *bytes present now*, and the timestamp suffix supports structural coherence, but **no game identifies the generating network or run** and overwriting is known to have occurred | **content-pinned but provenance-limited, not authenticated** — §4. Under the original C0 null condition, **generator provenance remains unavailable** |
| 2 | "no policy target exists **anywhere on disk**" | contradicted by this card's own §3 table: the atlas artifact holds **96 legs with visit dictionaries** | the supported and sufficient finding: **no eligible replay game corpus contains π**, and the isolated probe artifacts that do are retired and cannot supply a training corpus — §3 |
| 3 | "**96 is the positive control and it is binding**" | fewer parameters guarantee lower *nominal* capacity, not worse held-out loss. `L(96) ≈ L(128)` is consistent with the task saturating below 128 — **scientific evidence, not instrument failure** | the ladder has **no qualified positive control**; a binding one needs independently guaranteed signal destruction with legality preserved, e.g. an input ablation — §5 |
| 4 | a compute ceiling of "≈4.45 GPU-hours per 1,000 steps" and a confound-separation claim | the LR grid is **itself training** and is absent from the 45.375-equivalent estimate, which prices only the final three-seed ladder. Nothing binds which seeds, pool fraction, budget, checkpoint-selection rule or tie-break pick the LR. The shuffled-target control has **no legality-preserving shuffle and no defined chance baseline** | the ceiling is **withdrawn as invalid** and the confound separation is **not established** — §6, §8 |

**Consequent decisions, taken by the reviewer:** do not run the value-plus-behaviour-cloning probe;
set no compute ceiling; do not choose or trim the ladder, because its gate is not qualified. The
appropriate next authorization is **read-only discovery of a genuinely different external anchor**,
which is a separate scope and is not granted by this card.

### Review round 2, 2026-08-24 — historical universal claims bounded

Two claims reached backwards past their evidence. The current writer plus the files that survive
establish facts about **the present**; they cannot establish what every historical version of this
project once wrote — a distinction that matters most in exactly this card, which has just
documented overwrites and absent provenance.

| # | claim as written | bounded to |
|---|---|---|
| 5 | "no game written by the self-play path **has ever** carried π" | "**None of the 44,900 extant replay games carries π, and the current `GameSaver` cannot write it.**" — §3 |
| 6 | "every `metrics.csv` the trainer **has ever written** … there is no exception", and its summary "every run this project has ever done is 128×6" | every **surviving** recorded run — 39 files, 655 rows — with `checkpoints/` untracked, so these are the runs whose metrics survive, not necessarily every run performed — §2, §13 |

Two further sentences carried the same unbounded form and were brought into line with them: §1's
"network capacity has never been varied here" → *no surviving record shows it varied*, and §4's
"the information was never written" → *nothing that survives carries it and the current writers do
not emit it*. Correcting the summaries while leaving their originating claims unbounded would have
left the card contradicting itself.

Round 2 is **prose only**. No evidence file and no line of `capacity_c0_inventory.py` changed.

---

## C0 VERDICT — FEASIBILITY NO-GO

**C0 produced a durable corpus inventory and a working corpus-pinning gate. It did not produce a
runnable capacity experiment.** Four findings, each established by measurement below, and together
they close the line:

1. **The corpus is content-pinned but provenance-limited, not authenticated.** Its bytes are fixed
   by digest and its iteration ordering is structurally coherent, but **no game names the network
   or run that produced it**, and a later run has demonstrably overwritten part of it. The original
   C0 null condition asked whether a *suitable authenticated corpus* exists; **generator provenance
   is unavailable**, so that clause is not satisfied. (§4)
2. **No eligible replay game corpus contains the AlphaZero policy target π.** It survives on disk
   only in retired, isolated probe artifacts. The probe could use the terminal outcome and the move
   the search played — neither of which is π, and neither of which tests the training objective.
   (§3)
3. **The proposed design has no qualified positive control**, so it cannot distinguish a genuine
   null from an instrument that never had the sensitivity to see anything. (§5)
4. **The compute ceiling is invalid and the confound separation is unestablished**, because the
   learning-rate selection work is neither specified nor priced. (§6, §8)

Layered on top: the result would have carried **no strength claim in either direction** — there is
still no external anchor — so it would not have addressed the objective the fallback was meant to
serve. (§10)

**C1 is not authorized. The design below is retained as a record of what was considered and why it
did not qualify, not as a specification to be executed.**

---

## 1. Hypothesis — the question that was considered, and will not be tested

> Sections 1 and 2 record what the probe would have asked and what is already known. Sections 5–9
> describe a design that **did not qualify**. Nothing from §5 onward is authorized.

**H1.** On the pinned corpus, with matched optimization and matched data, held-out predictive loss
falls materially as trunk width rises above 128 at fixed depth 6.

**H0.** It does not — 128×6 already sits on the flat part of the capacity curve for the targets
this corpus can supply.

**Why this is not on the do-not-repeat list.** #1–#52 are value-head calibration, search
heuristics, readout formulas, training doses, continuation recipes and anchor hunts. Every one of
them held the architecture fixed at 128×6. No surviving record shows capacity ever being varied
here — §2 establishes that over every surviving recorded run, not from recollection.

**Directional prediction, recorded before anything is fitted.** Widening to 2× and 4× buys a small
but real reduction in behaviour-cloning cross-entropy and essentially nothing in value MSE, with
the curve already flattening between 128 and 181. Confidence **low**. The competing expectation —
that the search's move choice is a much harder function than 1.9M parameters can represent, so the
curve is still descending at 256 — is not excluded by anything measured here.

## 2. What is already known about capacity in this repository

**Every surviving `metrics.csv` — 39 files, 655 iteration rows — records `network_hidden=128,
network_blocks=6`, with no exception.** Bounded to what survives: `checkpoints/` is untracked, so
these are the runs whose metrics files are still on disk, not necessarily every run the project has
ever performed.
Evidence: `evidence/2026-08-24-capacity-c0/architecture_and_throughput.txt`.

`train.py` already exposes `--hidden` (default 128) and `--blocks` (default 6), and
`AlphaZeroNetwork(in_channels, hidden, n_blocks)` threads both through the encoder and both heads.
**Instantiating a wider network needs no change to model or training code.**

Parameter counts, derived from `network.py` by arithmetic — no network is instantiated — and
cross-checked against the byte size of the shipped `calib020_0001` checkpoint to within its
8,669-byte safetensors header:

| hidden | blocks | params | × params | × trunk FLOPs |
|---|---|---|---|---|
| 96 | 6 | 1,075,108 | 0.573 | 0.562 |
| **128** | **6** | **1,875,588** | **1.000** | **1.000** |
| 181 | 6 | 3,687,923 | 1.966 | 2.000 |
| 256 | 6 | 7,289,348 | 3.886 | 4.000 |

## 3. No eligible replay game corpus contains the AlphaZero policy target

> **Corrected in review round 1.** This section first claimed π exists "nowhere on disk". That is
> false and this card's own table below disproves it: the atlas artifact holds 96 legs with visit
> dictionaries. The claim has been narrowed to what the evidence supports — which is still enough
> to settle the question, because a training corpus is what the probe needed and none exists.

`PositionRecord.visit_counts` (`self_play.py:297`, built at `self_play.py:1090`) is the policy
target. It lives in `ReplayBuffer` (`trainer.py:1793`) and is consumed by `train_step`. **There is
no serialisation path for the buffer** — no `save_buffer`, no pickle, no `np.savez`, no caller of
`GameRecord.to_dict`. It dies with the run.

`GameSaver` writes per move: `turn, player, row, col, bridges_created, heuristics, search_score,
root_top1_share`. No distribution. **None of the 44,900 extant replay games carries π, and the
current `GameSaver` cannot write it.**

That is bounded on purpose. The writer as it stands today plus the files that survive today cannot
speak for what every historical version of the self-play path once emitted — least of all in a
directory this card has just shown to be overwritten and provenance-free. What the evidence
supports is a statement about the present contents and the present writer, and that is sufficient:
the probe needed a corpus it could read now.

Verified by opening every game file in all five replay corpora — 44,900 games — and counting those
carrying any visit key: **0**.

π does survive on disk, in exactly one place, and it cannot serve as a corpus:

| holder | what it has | why it cannot supply a training corpus |
|---|---|---|
| `logs/eval/atlas_pilot2/.../pilot_artifact.json` | **96 legs carrying real `visit_counts` dicts of 3–509 entries** — genuine π | derived from **24 games**, from the atlas pilot recorded `PHASE_GEOMETRY_NO_GO` and **retired as geometry-design evidence only**. Twenty-four games is not a corpus, and retired evidence may not be reused as fresh confirmatory evidence |
| `logs/eval/*_replays` (928 games) | `selected_visit_count`, `root_total_visits`, `n_legal`, `top2` | exactly **2** children per ply across all 43,567 plies, out of ~500 legal — a 2-point summary, not a distribution |
| `iter_NNNN_stats.json` (414) | matches only the metric *name* `best_by_visit` | per-ply region percentages aggregated over 100 games |
| `runs/mcts_golden_*` (279 files) | golden determinism fixtures | 16 fixed probe positions, not games |

**The supported finding:** no eligible replay game corpus contains π; the only artifacts that do
are isolated and retired. Evidence: `evidence/2026-08-24-capacity-c0/policy_target_search.txt`,
which reports the atlas legs explicitly and always did.

**Consequence.** The available targets are:

- **value** — `z ∈ {+1, −1}` from the side to move, from the terminal result of games that ended in
  a win under the rules. Ground truth, no model in the loop.
- **behaviour cloning** — the move the 400-simulation search actually played, as a one-hot target.
  This is a temperature sample from π, not π. Its cross-entropy has an irreducible floor equal to
  the entropy of the sampling distribution — but that floor is common to every architecture, so it
  cancels in a comparison between them.

The board tensor itself is reconstructable: replay the move list through `TwixtState.apply_move`
and call `state.to_tensor()`, the pattern already used at `probe_eval.py:865`.

## 4. Corpus identity — content-pinned, provenance-limited, NOT authenticated

> **Corrected in review round 1.** The C0 verdict first said "an authenticated corpus does exist".
> It does not, and the distinction is the whole point of this section.

**What the digest establishes.** `directory_digest` and `eligible.digest` fix the *bytes present
now*. Re-running `verify` proves the corpus has not changed since the inventory was taken. That is
content pinning, and it is real.

**What nothing establishes.** No extant game carries any field naming the network, checkpoint, run
id or configuration that generated it — not in the game JSON, not in the `_stats.json` sidecars.
The directories are untracked (`.gitignore`), and `game_saver.py` names every file by iteration and
game index alone, so runs sharing a directory share one flat `iter_NNNN_game_NNN.json` namespace
and **a later, shorter run silently overwrites the low iterations of an earlier one — which is not
a hypothetical here; it has already happened.**

The timestamp suffix in `intact_suffix()` recovers *structural coherence*: a range of iterations
written once, in order, by something. It does **not** recover *generator identity*: it cannot say
which network produced those games, and it would not detect a rewrite that preserved timestamp
ordering. **The corpus is therefore content-pinned but provenance-limited.**

**Consequence for the C0 null condition.** The condition was "if no suitable **authenticated**
corpus exists, record that and stop." Generator provenance is unavailable and cannot be recovered
from anything on disk, so that clause is not satisfied — and no further read-only work would
recover it, because nothing that survives carries it and the current writers do not emit it.

In `scripts/GPU/logs/games`, iterations 0–14 carry timestamps from 2026-06-21, 06-23, 06-25 and
07-09, while iterations 15–413 form one increasing sweep from 2026-04-21 to 2026-06-19. The
original curriculum warm-up survives only at iterations 15–19 (board 16 and 20, 150–200 sims);
iterations 0–14 are 24×24/400-sim games left by later runs. **1,500 games are foreign.**

`capacity_c0_inventory.py` does not hardcode that boundary. `intact_suffix()` derives it: the
longest run of iterations, ending at the highest, whose timestamp windows do not run backwards. It
returns `[15, 413]` from the data alone, matching a hand analysis done independently.

### Inventory of every replay corpus

| corpus | games | positions | policy targets | intact range | recorded use |
|---|---|---|---|---|---|
| `scripts/GPU/logs/games` | 41,400 | 2,378,243 | 0 | **15–413** (1,500 games foreign) | the AlphaZero-v2 training run; also read by the probe-suite, opening, goal-completion, recovery-retargeting and marathon diagnostics |
| `logs/selfplay/cont5_from_calib020` | 500 | 30,761 | 0 | 0–4 | training-continuation line, CLOSED, ledger #49 |
| `logs/selfplay/warm5_from_calib020` | 1,000 | 61,601 | 0 | 0–4 | parent-replay bootstrap, CLOSED, ledger #49 |
| `logs/selfplay/fp5_from_calib020` | 500 | 32,497 | 0 | 0–0 | frozen-parent opponent, CLOSED, ledger #51 |
| `logs/selfplay/fp6_from_calib020` | 1,500 | 96,005 | 0 | 0–4 | frozen-parent training, CLOSED, ledger #51 |

`Replays/` holds derived CSV analyses and summaries, not games. `runs/` holds MCTS golden fixtures.
Neither is a corpus. Prior-use evidence: `evidence/2026-08-24-capacity-c0/prior_use.txt`.

**The four continuation corpora are not usable.** Together they are 3,500 games, they were produced
by and consumed in lines the ledger records as closed, and they carry no policy target either.
The probe uses `scripts/GPU/logs/games` only.

### Frozen eligibility predicate

`reason == "win"` **and** `board_size == 24` **and** iteration inside the derived intact suffix.

Within the intact range of 39,900 games, `reason == "win"` removes 634 (**1.589%**): 542 whose
outcome label is adjudicated, timed out or state-capped — labels produced by a network, which
would be circular for a fit probe — and 92 resignations, whose label is also value-triggered.
`board_size == 24` removes the 500-game small-board curriculum. Four games fail both clauses, so
1,130 are removed in total and **38,770** remain.

| | value |
|---|---|
| eligible games | **38,770** |
| eligible positions | **2,095,652** |
| iterations present | 394, spanning 20–413 |
| winner balance | red 19,427 · black 19,343 |
| plies per game | min 25 · p05 38 · median 53 · p95 74 · max 341 |
| **`eligible.digest`** | `dc7713c601ae6fc861b47b21fbfa7939ecdb6a361e05eef2f8ff09671ec22832` |
| **`directory_digest`** | `b496637685ceacd141551ceecd802ab9588d2e095a50b2ed6f2d5020b746a1a9` |

Structural integrity over all 41,400 games: 0 duplicate ids, 0 duplicate move sequences (41,400
distinct), 0 games where `meta.n_moves` disagrees with the move list, 0 games whose turn numbering
is not `1..n`, 0 unreadable files.

### Outcome-blind whole-game split

`split = sha1(game_id) mod 100`, `< 80` train, `< 90` val, else test. The game id is
`iter_NNNN_game_NNN` — derived from position in the run and nothing else, so **the key cannot
encode the winner**. Whole games only; no position is ever split across arms.

| arm | games | positions | red / black | iterations present | min games per iteration |
|---|---|---|---|---|---|
| train | 31,026 | 1,676,453 | 15,539 / 15,487 | 394 | 63 |
| val | 3,823 | 207,336 | 1,930 / 1,893 | 394 | 1 |
| test | 3,921 | 211,863 | 1,958 / 1,963 | 394 | 3 |

Colour balance is within 0.006 of even in every arm and all 394 iterations appear in all three, so
the arms share the corpus's iteration mixture without explicit stratification.

### Duplication risk, bounded on both sides

Games are distinct, but openings repeat, so a whole-game split still shares *positions*. Both
bounds were measured:

- **Lower bound** — exact move prefix, order-sensitive; a match is a genuine repeat:
  **19,420 / 211,863 = 9.166%** of test positions.
- **Upper bound** — placed pegs by colour, order-blind. Bridges are created greedily in knight
  offset order and blocked by crossing (`game/bridge.py`), so equal peg sets can carry different
  link sets; treating them as identical over-counts: **20,338 / 211,863 = 9.600%**.

All of it is the shared opening book. Under the upper bound the last ply with any overlap is 10;
**at ply ≥ 11 the overlap is zero under both bounds**, and `0 / 149,127` at ply ≥ 16.

**Frozen consequence:** the primary metric is computed on the **leak-free slice, ply ≥ 11** —
168,732 test positions. Full-slice numbers are reported as a secondary. This boundary is chosen
now, from a structural property of a frozen split, with no model having produced a single
prediction.

## 5. Design as proposed — RETAINED AS A RECORD, NOT A SPECIFICATION

> **The remainder of this card describes the design that was considered and did NOT qualify.**
> §5 has no valid positive control and §6/§8 do not price or bind their own selection step. It is
> kept so a future proposal can see what was tried and where it broke, not so it can be run.

Four widths at fixed `blocks = 6`: **96, 128, 181, 256** — 0.5625×, 1×, 2×, 4× trunk FLOPs.

Width only. Depth is held fixed so the comparison cannot be confounded by depth-specific
optimization behaviour; that is a limitation, recorded in §10, not an oversight.

### The 96 arm is NOT a positive control — corrected in review round 1

This card first asserted "**96 is the positive control and it is binding**", and made
`L(96) − L(128) < δ` a VOID condition. **That was wrong.**

Fewer parameters guarantee lower **nominal** capacity. They do not guarantee **worse held-out
loss** on this task. If `L(96) ≈ L(128)`, the most natural reading is that the task saturates at or
below width 96 — which is **a scientific result about capacity, and one of the most interesting
outcomes the ladder could produce**. Declaring it an instrument failure would have thrown away the
finding and, worse, would have made the design unable to report the very null it was built to test.

A binding positive control needs a manipulation whose effect is **guaranteed independently of the
hypothesis** — signal destruction that must degrade the metric if the pipeline works at all, with
**legality preserved** so the model is still solving a well-formed problem. An input ablation
(masking or permuting the input planes that carry the position, while the legal-move set and the
target space stay valid and unchanged) is the shape that would qualify. **No such control was
designed, so the ladder has none, and rule 1 in §7 does not bind.**

## 6. Controls as proposed — the separation is NOT established

> **Corrected in review round 1.** Two of these controls are underspecified in ways that void the
> claim that this table separates capacity from optimization, and void §8's cost model with it.
>
> **(a) The learning-rate selection is unspecified and unpriced.** Choosing an LR per width *is
> training* — a full extra sweep — and §8's 45.375 incumbent-equivalents price only the final
> three-seed ladder, so the ceiling omits it entirely. Beyond cost, nothing here binds **which
> seeds** the selection runs use, **which pool fraction**, **what step budget**, **which checkpoint
> within a selection run is scored**, or **how ties are broken**. Every one of those is a
> researcher degree of freedom sitting directly on the optimization axis this control claims to
> close. A valid version has to freeze all five and price the sweep.
>
> **(b) The shuffled-target control has no legality-preserving shuffle and no defined chance
> baseline.** Shuffling the played move across positions will hand many positions a target that is
> **not legal there**, so a model can drive the loss down by learning legality alone and the arm
> stops being a label-leak detector. The shuffle must be **within the legal move set of the same
> position**, and "chance" must be written down in advance as an explicit number — for a uniform
> draw over `n_legal`, the mean of `ln(n_legal)` over the evaluated slice — not left as a
> qualitative expectation.
>
> **Therefore the compute ceiling in §8 is invalid and the confound separation below is not
> established.** The table is retained as written for the record.

| confound | control |
|---|---|
| **optimization** | learning rate chosen per width from the fixed grid `{3e-4, 1e-3, 3e-3}` on the **validation** arm. Identical optimizer, batch size, schedule, weight decay and canonicalization across widths. The test arm is read exactly once per arm, at the end. |
| **seed / run-to-run noise** | 3 seeds per (width, selected LR). The spread across the three **incumbent** seeds defines the null band — "material" is measured, not asserted. |
| **data volume** | the whole ladder runs at two training-pool sizes, `P` and `P/2`, the half drawn as an outcome-blind whole-game subsample of the train arm. If the incumbent's `P/2 → P` gain exceeds the `128 → 256` gain at `P`, the corpus is data-limited relative to the capacity effect and the result is **DATA_LIMITED**, not a capacity finding. |
| **training duration** | fixed optimizer-step budget `S` for every arm, with validation loss logged throughout. If any width has not reached a validation plateau by `S` — plateau defined as no improvement over the final 20% of steps beyond the seed band — the comparison is duration-confounded and the run is **VOID**. |
| **label leakage in the pipeline** | one extra incumbent-width arm trained on **shuffled targets** must reach chance loss. If it does not, the pipeline is leaking and nothing else is reportable. |
| **corpus drift mid-run** | `capacity_c0_inventory verify` runs before the first step and after the last. Non-zero exit aborts and voids. |

## 7. Primary metric and the single binding rule

**Primary metric.** Held-out **behaviour-cloning cross-entropy in nats per position**, on the
leak-free test slice (ply ≥ 11), from the early-stopped checkpoint at the per-width selected LR,
averaged over the 3 seeds. Written `L(w)`.

Behaviour cloning rather than value: across all 652 recorded training iterations that log both
terms, median `avg_policy_loss` is **3.254** against median `avg_value_loss` **0.316** — the policy
channel carries **10.3×** the loss mass, and is where representational demand actually lives. Value
MSE is reported alongside, and is **not** binding.

**The rule, fixed here, before any prediction has been computed.**

Let `s` = sample standard deviation of `L(128)` across its 3 seeds.
Let **`δ = max(3s, 0.02 × L(128))`**.

The 2% floor exists so that an unusually tight seed band cannot manufacture a significant result
from a change too small to matter. It is a formula, not a number chosen against an observed effect.

1. ~~**VOID** if `L(96) − L(128) < δ` — the positive control failed.~~ **STRUCK, review round 1:
   96 is not a positive control (§5). `L(96) ≈ L(128)` is a capacity result, not an instrument
   failure, and this rule would have suppressed it. Nothing replaces it, so the design has no
   sensitivity check at all.**
2. **VOID** if the duration, shuffled-label or corpus-verify controls fail. **The shuffled-label
   control as written does not qualify — no legality-preserving shuffle, no numeric chance
   baseline (§6b).**
3. **DATA_LIMITED** if the data-volume control fires.
4. Otherwise **HEADROOM** iff `L(128) − L(256) ≥ δ` **and** `L(181) ≤ L(128) + δ` and
   `L(181) ≥ L(256) − δ` — the improvement must be real *and* monotone across the ladder.
5. Otherwise **NO HEADROOM**.

One evaluation. No second look, no relaxed margin, no extra width, no extra seed, no rerun at a
different budget. A `VOID` is a failure of the instrument and licenses repair of the instrument
only — never a re-read of the same arms.

## 8. Compute cost — the ceiling below is WITHDRAWN as invalid

> **Corrected in review round 1.** The `45.375` figure prices **only the final three-seed ladder**.
> It omits the learning-rate selection sweep entirely, and that sweep is training: at three LRs per
> width it is of the same order as the ladder itself, and its true size cannot even be stated
> because §6 never binds the selection runs' seeds, pool fraction or step budget. **Every number in
> this section is therefore a lower bound on an unknown total, not a ceiling.** No compute ceiling
> was set, and the reviewer has set none.
>
> The measured throughput below stands on its own and is the one durable quantity here.

Measured, from three independent August 2026 runs, 15 clean iteration rows, all 128×6 at batch 64:
**median 2.833 optimizer steps/s = 181.3 positions/s.** (A sixteenth row records `wall = 0.00 s`;
the timer bracketed no work and it is excluded, a decision about the timer, not about the rate.)

Final-ladder cost only, in incumbent-equivalents:
`(0.5625 + 1 + 2 + 4) × 3 seeds × 2 data fractions = 45.375`. At 2.833 steps/s that is `16.0 s` per
step of budget, or `≈ 4.45 GPU-hours per 1,000 steps` — **excluding selection, and therefore not a
ceiling.**

The table below is retained **only** to show the order of magnitude of the data scale the hardware
permits. Its "ceiling" column is not a ceiling — it prices the final ladder alone — and no row of
it is authorized.

| final-ladder budget | step budget `S` | positions per arm | pool for 4 epochs | pool for 2 epochs |
|---|---|---|---|---|
| 12 h | 2,700 | 172,800 | ~43,000 pos (~800 games) | ~86,000 (~1,600 games) |
| 24 h | 5,400 | 345,600 | ~86,000 (~1,600 games) | ~173,000 (~3,200 games) |
| 48 h | 10,800 | 691,200 | ~173,000 (~3,200 games) | ~346,000 (~6,400 games) |

**Against a train arm of 31,026 games this is a few percent** — and that is before the omitted
selection sweep takes its share of any real budget. Whatever budget were chosen, the
probe fits on a subsample; it does not fit the corpus. That is the honest shape of the finding, and
§10 records it as a limit on the claim.

Two things in the table are projections, not measurements, and **C1's first act must be to replace
them with measurements**: that throughput scales with trunk FLOPs, and that batch 64 is the right
operating point. There is a named reason the measured figure may understate a purpose-built fitting
loop — `train_step` calls `.item()` on every loss term on every step (`trainer.py:1737–1774`),
forcing an MLX synchronisation per step, and batch 64 is small for this GPU. A throughput
calibration is a measurement of the machine, not of the science; sizing `S` and `P` from it is
legitimate, and is not the same act as choosing a threshold after seeing a loss.

**The sizing rule first written here — "choose the largest pool `P` for which the full ladder fits
the ceiling at ≥ 4 epochs" — is void**, because it is stated against a ceiling that does not price
the selection sweep. It is not replaced. No `S`, no `P`, no schedule is frozen, and none may be.

## 9. What C1 would have built — NOT AUTHORIZED, nothing here may be built

- One fitting script that reads the pinned corpus, reconstructs board tensors through the existing
  `TwixtState` replay path, and fits `AlphaZeroNetwork(hidden=w, n_blocks=6)` at a given width,
  LR, seed and pool size. It writes one JSONL row per logging interval and one terminal row.
- A test-arm guard whose accessor raises unless the run is in its single final evaluation, with a
  self-test proving it raises when called earlier.
- A preflight that re-runs `capacity_c0_inventory verify`, re-checks zero overlap at ply ≥ 11, and
  refuses to start on any mismatch.
- A runner whose **exit status is the verdict**: `0` a decided result, `2` precondition, `3`
  VOID, `4` DATA_LIMITED.

**Not built:** no change to `network.py`, `trainer.py`, `train.py`, `mcts.py`, `self_play.py`, the
product service or any archive. No self-play. No new games. No seed reservation — the probe
consumes none, because it generates nothing.

## 10. Interpretation limits — binding on any report of the result

- **No playing-strength claim, no Elo, no promotion, no product or deployment claim, in either
  direction.** There is still no external anchor; the twixtbot line was rejected on 2026-08-24.
  A fit result is not a strength result and may not be described as one.
- The targets are the terminal outcome and a **behaviour clone of the search's played move**. They
  are not π. A result about them is not a result about the AlphaZero training objective.
- The measurement is on a **subsample** — order 10³ games of 38,770 — at a fixed step budget. It
  does not establish what any architecture would do trained properly on everything.
- **Width only, depth 6 fixed.** Nothing here speaks to deeper networks.
- The corpus was generated by a non-stationary 128×6 lineage across 394 iterations, and the
  incumbent architecture was itself settled on while looking at this data. The comparison is
  fair between widths; the corpus is not neutral ground.
- `HEADROOM` would license *proposing* a successor programme with its own authorization. It would
  license no training, no checkpoint, no promotion, and would say nothing about `calib020_0001`.
- `NO HEADROOM` would bound the tested widths on the tested targets at the tested data scale. It
  would not show that architecture is a dead end.

## 11. Stopping rules, cleanup, evidence

**Stop and report immediately on:** `capacity_c0_inventory verify` non-zero at either end; any NaN
or non-finite loss; the shuffled-label control failing to reach chance; ~~the width-96 positive
control failing;~~ the wall-clock ceiling being reached. (The width-96 clause is struck — see §5.
The shuffled-label clause does not bind as written — see §6b.)

Failure at a frozen setting **is the result**. Nothing may be lowered, widened or retried to obtain
a decided run.

**Cleanup.** Intermediate checkpoints are deleted once the single test pass has written its row.
Nothing is written into any corpus directory, ever. No file under `scripts/GPU/logs/games`,
`logs/selfplay`, `Replays` or `checkpoints` is created, modified or removed.

**Evidence.** Create-only, under `docs/superpowers/evidence/2026-08-24-capacity-c0/`. Every attempt
is preserved unaltered; corrections are made by adding a file, never by editing one.
**Review round 1 changed none of it, and changed no line of `capacity_c0_inventory.py`** — the
files below and the tool are byte-identical to what the reviewer independently verified against all
five corpora. Only this card was amended, and its corrections are logged at the top rather than
applied silently. The C0 evidence:

| file | what it is |
|---|---|
| `inventory_games.json` | full structural inventory + digests of the training corpus |
| `inventory_{cont5,warm5,fp5,fp6}.json` | the same for the four continuation corpora |
| `gate_controls.txt` | **attempt 1, FAILED** — every `verify` exited 1 on a `TypeError`. Preserved. |
| `gate_controls_attempt2.txt` | attempt 2, 13 controls as specified, but under the system interpreter, where the MLX-leak control was vacuous. Preserved. |
| `gate_controls_attempt3_venv.txt` | attempt 3, 14 controls under `.venv/bin/python`, MLX-leak control now binding |
| `architecture_and_throughput.txt` | 655-row architecture history, throughput, parameter arithmetic |
| `policy_target_search.txt` | where π lives — including the 96 atlas legs that do carry it — and why no eligible game corpus does |
| `prior_use.txt` | every tracked reference to each corpus path |

## 12. The gate that pins the corpus, and proof that it rejects

`scripts/GPU/alphazero/capacity_c0_inventory.py` — stdlib only; the control run confirms it imports
no MLX, numpy, torch, onnx or game engine. `emit` writes an inventory and refuses to overwrite one;
`verify` recomputes it from the live corpus and compares. There is one code path: `verify` calls
the same `inventory()` that `emit` wrote from, so the check cannot drift from the thing checked.

**Attempt 1 failed, and is preserved in `gate_controls.txt`.** Every `verify` exited 1 on
`TypeError: '<' not supported between instances of 'int' and 'str'` — JSON has no integer dict
keys, so the live inventory and the one read back were not comparable. A gate that crashes is not
a gate. Fixed by canonicalising through a JSON round trip inside `inventory()`, with a regression
assertion added to `selftest`.

**Attempt 2 passed but one of its controls was vacuous.** It ran under the system interpreter,
which has no `mlx` installed — so "imports no MLX" could not have failed and proved nothing. The
same interpreter cannot even collect the test suite (30 import errors). Attempt 2 is preserved as
run; **attempt 3 re-ran every control under `.venv/bin/python`**, the interpreter the project uses,
where `mlx` is importable and the check binds.

Attempt 3, each a fresh subprocess under the project interpreter, exit code stated in advance:

| # | control | expected | got |
|---|---|---|---|
| P1 | true corpus vs its own inventory | 0 | 0 |
| N1 | eligible-game count incremented by one | 3 | 3 |
| N2 | directory digest replaced | 3 | 3 |
| N3 | provenance range widened over the clobbered iterations | 3 | 3 |
| N4 | a whole section deleted from the inventory | 3 | 3 |
| N5 | leakage understated to zero | 3 | 3 |
| N6 | policy-target count falsified to "all 41,400 files have one" | 3 | 3 |
| N7 | correct inventory pointed at a different corpus | 3 | 3 |
| N8 | cont5 inventory pointed at the fp5 corpus | 3 | 3 |
| N9 | nonexistent root | 2 | 2 |
| N10 | corrupt inventory file | 2 | 2 |
| N11 | `emit` over existing evidence | 2 | 2 |
| P2 | `selftest` | 0 | 0 |
| P3 | all four continuation corpora vs their own inventories | 0 | 0 |

The import-leak check now carries its own negative case: the module leaks nothing, while importing
`network.py` in that same interpreter does pull `mlx.core`, so the check is capable of failing.
Suite collection under the project interpreter: **2,885 collected, 0 errors**.

**Stated at the width the evidence supports.** N1–N6 mutate the *stored* side. That exercises the
same comparison a drifted corpus would trip, and it is the only side that can be mutated without
writing into a corpus, which is excluded. It is not a demonstration that the tool was run against
an actually-altered corpus, and it is not described as one.

## 13. Decisions — SETTLED, review round 1, 2026-08-24

The three questions this section opened have been answered by the reviewer. Recorded as taken:

1. **Do not run the value-plus-behaviour-cloning probe.** It does not test π, does not establish
   strength, and does not address the external-baseline objective the fallback exists to serve.
2. **No compute ceiling is set. C1 is not authorized.**
3. **Do not choose or trim the ladder.** Its gate is not qualified (§5, §6), so there is nothing to
   size.

**Status: this line is closed as a feasibility no-go.** What C0 leaves behind and what it does not:

| kept | value |
|---|---|
| `capacity_c0_inventory.py` | a working, negative-controlled corpus-pinning gate, independently verified by the reviewer against all five corpora |
| the five inventories | content pins for corpora that had no identity at all before, including the record that 1,500 games at iterations 0–14 are foreign |
| the throughput and architecture record | 181 positions/s at 128×6; every **surviving recorded run** — 39 `metrics.csv`, 655 rows — is 128×6 |
| the leakage bounds | 9.166–9.600%, zero at ply ≥ 11 — reusable by any future work that splits this corpus |

| NOT established | |
|---|---|
| corpus provenance | no game names its generator, and this cannot be recovered read-only |
| that capacity is separable here | the design that claimed it had no qualified positive control and no priced selection step |
| anything about strength | there is still no external anchor |

**The appropriate next authorization is read-only discovery of a genuinely different external
anchor** — a separate scope, not granted by this card. Nothing in this card licenses training, a
checkpoint, a promotion, or any claim about `calib020_0001`.
