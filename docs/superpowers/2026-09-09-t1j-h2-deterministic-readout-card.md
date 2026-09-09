# H2 — Plan-Only Preregistration: is T1j stronger than our incumbent in the **deterministic visit-count-argmax configuration**?

**Status: PLAN ONLY. Nothing here has run.** No code, no runner, no gate, no model load, no JVM, no
game, no seed registration, no training, no confirmation inspection, no push. The implementation and
the execution are **separate authorizations** and neither is requested by this document.

**H2 is not H1 continued.** It asks a different question of a **different incumbent configuration**
with **different seeds**, and §8 forbids pooling or rate-comparison inference with either H1 attempt.

**AMENDED TWICE, 2026-09-09, before implementation, plan only.**

**Third amendment.** §2.2 narrowed twice more: the terminal reason is exactly `win` or `cap` (this
protocol has **no resignation**, which my first draft listed), and the ply sequence must be exactly
`opening_bound + 1 … task_result.plies` with each mover **equal to its expected colour** — contiguity
cannot see a missing first or final record, and alternation cannot see every mover flipped. Three
controls added.

**Second amendment.** §2.2 freezes what a **transcript** is, structurally. Without it the per-cell
screen is vacuous: hashing the whole record makes `seed`, `task_id` and `rep` give 46 distinct
transcripts for a cell that played one game 46 times. Four controls are added with it. Nothing else
changes — the parity rule, the 736 games and the per-cell threshold stand.

**First amendment.** Five corrections from pre-implementation
review, none touching the parity rule or the 736-game count: the title and question no longer claim
argmax is the *strongest* deterministic use, only that it is *this* frozen configuration; the
degeneracy screen is now **per cell** (≥42 of 46), because a global 90% rule passes a wholly collapsed
cell, and the blanket "effective n = 16" claim is withdrawn; the independence Hoeffding requires is
now **stated as a model** and the interval called **nominal under it**, with the transcript screen
explicitly not a validity proof; the sentence attributing H1's 0.6763 to the readout is **removed** as
the very inference §8 forbids; and "exactly one change" is narrowed to **one gameplay-rule change**.

---

## 1. The question, and why it comes before any training

Every measurement so far used the incumbent's **temperature-sampling** readout, in which the played
move is drawn from the visit distribution rather than taken as its maximum. The D1″ reporting
correction established, from the stored histograms, that under that readout a **non-leader move was
played in most positions examined** — some with a visit rank in the hundreds. Whether that costs
playing strength is untested, and it is a cheap thing to change: the same model, the same search, one
different selection rule.

> **PRIMARY QUESTION.** With the incumbent playing **visit-count argmax** after each fixed opening, is
> T1j stronger than our incumbent **in that configuration**, under otherwise matched conditions?

⚠ **"Argmax" is not "the strongest deterministic use", and this card does not claim it is.** Visit-
count argmax is a clear, already-qualified deterministic rule; **nothing establishes it as the best
one**, and another deterministic rule (an LCB readout, a policy-weighted rule, a deeper search) could
do better. `T1J_STRONGER` therefore means **stronger than this frozen configuration** — never
stronger than every deterministic use of the model.

**Why this is the shortest path.** A `NOT_STRONGER` outcome would establish that **H1 did not show
our model to be inferior under argmax** — the premise for training against T1j would rest on a
configuration we had already improved on, and would weaken sharply. 🔴 **It would NOT make H1's 0.6763
attributable to the readout**: the readout *and* the seeds differ between the two runs, so that
attribution is confounded by construction and §8 forbids it. A `T1J_STRONGER` outcome would establish
that T1j beats this configuration, and a training question would become worth designing. Either
answer is worth more than another diagnostic on the same games.

## 2. The design — H1's, with one GAMEPLAY-RULE change

| | |
|---|---|
| openings | the same **8**: `o1_center`, `o2_offcenter`, `o3_low`, `o4_high`, `o5_wide_left`, `o6_wide_right`, `o7_diagonal`, `o8_contact` |
| colours | **reciprocal**: every opening played with T1j as red and as black (`t1j_red`, `t1j_black`) |
| T1j | **mdPly 6**, the pinned jar and JDK, unchanged |
| incumbent | `calib020_0001`, **400 simulations**, board 24, batch 14, stall-flush 48 — unchanged |
| ply cap | **280** |
| **THE GAMEPLAY-RULE CHANGE** | incumbent readout `selection_mode`: `opening_temperature` → **`argmax`** |

⚠ **"One change" means one change to how a game is PLAYED.** Four other things necessarily differ and
are listed so the phrase cannot mislead: the **sample size** (736, not 224), the **seed block** (a
fresh paper reservation), the **incumbent identity** (which records the new mode), and the **output
locations**. Only the readout changes what either side does at the board.

`eval_readout.MODE_ARGMAX` already exists and is one of the three qualified modes. Under it,
`temp_high`, `temp_low` and `opening_temp_plies` become inert; the card records them as **not
applicable**, never as "unchanged", because a setting that no longer acts is not the same setting.

🔑 **The incumbent identity therefore DIFFERS from L0's and H1's, and must say so.** The recorded
identity must carry `selection_mode: "argmax"`. A run whose identity still reads
`opening_temperature` has not made the change this study is about, and must VOID.

### 2.1 A design check this change makes necessary

T1j is deterministic at fixed depth (E3a: 25 identical queries). With argmax, our readout is
deterministic too. **The remaining source of variation between repetitions of one cell is the search
stream** — the seeded RNG driving prior shuffle and PUCT tie-breaks — which differs per game seed.

If that were not so, the repetitions in a cell would be one game repeated.

🔴 **THE SCREEN IS PER CELL, because the variation it checks is per cell.** A global 90% rule does not
bind: one completely collapsed cell beside fifteen fully distinct ones gives 691/736 = **93.9%
distinct and passes**, while a sixteenth of the design carries no information at all. The run must
record, and the report must state, the count of **distinct game transcripts WITHIN EACH of the 16
opening × colour cells**.

**Frozen rule:** every cell must hold **at least 42 distinct transcripts of its 46** (the same 90%,
applied where it binds). If any cell falls below that, the primary interval is **not computed** and
the outcome is **`INCONCLUSIVE — DEGENERATE DESIGN`**, naming the offending cells.

### 2.2 What a TRANSCRIPT IS — frozen structurally, because otherwise the gate is vacuous

🔴 **THE FAILURE THIS CLOSES.** If the implementation hashes the whole game record, `seed`, `task_id`
and `rep` make **all 46 games unique even when the gameplay is byte-identical** — the screen would
report 46 of 46 distinct for a cell that played one game 46 times, and would pass every design it was
built to catch. A screen whose diversity comes from the schedule rather than the play is not a screen.

> **Within a cell, a transcript is the type-strict ordered sequence of post-opening `(mover, row,
> col)` moves, followed by the terminal reason and the winner. It is EXACTLY that and nothing else.**

**Included:** each post-opening ply's `mover` and its `(row, col)`, in played order; the terminal
reason; the winner.

**The terminal reason is one of exactly two values: `win` or `cap`.** `l0_match_rules.TERMINAL_REASONS`
is `("win", "cap")` and refuses anything else. **This protocol has no resignation** — an earlier draft
of this section listed one, which would have written a state the runner cannot produce into the
definition of a transcript. A third value appearing in a record is a **refusal**, not a new
transcript.

**Excluded, explicitly:** `task_id`, `rep`, `seed` and every derived stream; timestamps and any
timing; every diagnostic (visit counts, root values, policy ranks, override flags, `top2`); and all
provenance (identities, digests, file paths, engine or plan versions). The **opening is excluded too**
— it is constant within a cell, so including it could only add sameness, never diversity.

**Type-strict**, like every other comparison in this programme: `mover` is one of the frozen colour
values, `row` and `col` are `int`, and `"11"` is not `11`. A record whose moves arrive as strings
describes a different thing and must refuse rather than silently compare unequal.

**Malformed ply records REFUSE; they never manufacture diversity.**

🔴 **"Contiguous from the first post-opening ply" was not enough**, and the gap is at both ends: a
sequence missing its FIRST post-opening record is still contiguous from whatever record survives, and
one missing its LAST is still contiguous up to there. Contiguity is a property of the interior; it
cannot see a truncated boundary.

**Frozen instead — the sequence must be EXACTLY:**

> `opening_bound.ply + 1` … `task_result.plies`

— every index present, once, in increasing order, with **both endpoints anchored to records the run
itself wrote**. The first index is fixed by the game's own `opening_bound` and the last by its own
`task_result`, so a missing record at either end is a **count mismatch against a declared boundary**,
not a judgement call.

**Each mover must EQUAL THE COLOUR EXPECTED FOR ITS PLY**, derived from the task's colour arm and the
ply's parity — not merely alternate. Alternation is preserved by flipping every mover in a game, which
would swap which side played every move while passing an alternation check, and would silently turn
one transcript into a different one.

An undefined transcript **VOIDs the run** — it is never hashed into a novel value that inflates the
distinct count.

⚠ **No blanket "effective n = 16" claim is made.** That would hold only if **all sixteen** cells
collapsed completely; a partial collapse reduces the effective sample by an amount this design does
not quantify, which is exactly why the screen refuses rather than adjusts.

## 3. Sample size, and what it can resolve

| | |
|---|---|
| repetitions per cell | **46** |
| cells | 16 (8 openings × 2 colour arms) |
| **exact task count** | **736 games**, one seed each |
| primary interval | **Hoeffding 95%, two-sided**, distribution-free, **nominal under the independence model of §3.1** |
| half-width at n=736 | **±0.0501** |

The score resolves parity only if it falls **below 0.4499** or **above 0.5501** — the interval must
clear 0.50 by its own half-width. That is the honest reach of this design and it is stated **before**
the run: **any score between those two bounds is INCONCLUSIVE**, and no larger n is added afterwards
to rescue it.

**Why 46 and not 14.** H1's 224 games gave ±0.0907, which cannot separate parity from a moderate edge
— exactly the ambiguity that left H1 inconclusive. 736 games buy ±0.05 at a cost of roughly six
hours. Hoeffding is distribution-free, so this reach does not depend on any variance assumption,
including the lower per-game variance argmax is likely to produce.

### 3.1 The independence assumption, stated rather than assumed away

🔴 **Hoeffding needs bounded outcomes AND INDEPENDENT ones.** It is distribution-free about the
*shape* of the outcome, not about dependence between games. The games here are driven by
**predetermined pseudorandom seed streams** from one generator, against a deterministic opponent, over
a design that repeats each opening 46 times. **Independence is a MODEL we are relying on, not a fact
the design establishes.** The interval is therefore reported as **nominal under that model** — every
statement of it must carry that phrase — and the outcome names a decision under the model, not a
proven coverage guarantee.

⚠ **The per-cell transcript screen (§2.1) does NOT validate this assumption.** Distinct transcripts
detect gross schedule degeneracy — the same game replayed — and nothing more. Games can be entirely
distinct and still be dependent, for instance through a shared opening, a shared model, or correlated
seed derivations. The screen is a **degeneracy check**, kept deliberately separate from the
independence model, and neither substitutes for the other.

## 4. The primary rule — parity, frozen

Let **S** = T1j's score (win 1, draw or cap-termination ½, loss 0) divided by 736, and let
**[lo, hi]** be its two-sided Hoeffding 95% interval, **nominal under §3.1's independence model**.

| outcome | condition |
|---|---|
| **T1J_STRONGER** | `lo > 0.50` |
| **NOT_STRONGER** | `hi < 0.50` |
| **INCONCLUSIVE** | the interval contains 0.50 |
| **INCONCLUSIVE — DEGENERATE DESIGN** | **any** cell holds < 42 distinct transcripts of its 46 (§2.1) |
| **VOID** | any abort, cap breach, identity mismatch, or incomplete schedule |

**One rule, one threshold, decided in advance.** No secondary statistic may override it, and a
narrower interval may not be substituted after seeing the result — **Wilson cannot replace Hoeffding
after the fact**, the specific error H1's closure named.

### 4.1 The 0.75 threshold is REPORTED, and decides nothing here

H1's "worth targeted investment" band at 0.75 may be reported for continuity. **It does not replace
the parity question and cannot change the outcome.** They are different questions: 0.50 asks whether
T1j is stronger at all; 0.75 asked whether it is far enough ahead to justify targeted work.

## 5. Execution bounds

| | |
|---|---|
| whole-run wall clock | **480 minutes**, breached ⇒ VOID |
| per-query timeout | **120 s**, unchanged |
| expected duration | ≈ **5 h 41 m** (H1 attempt 2 measured 27.82 s/game; an ESTIMATE from a different readout, not a promise) |
| cap terminations | scored ½, counted, and reported; a count above 368 (half) is reported as a design warning, not a rule |
| **no early stop** | one preregistered one-shot schedule. **No interim look, no stopping on a lead, no extension.** A partial run is VOID and yields no analysis. |
| retries | **none without a new authorization** |

## 6. Seeds — a fresh interval, PAPER-ONLY

**Proposed:** `[202618000, 202618736)` — 736 seeds, one per game, bound **positionally** (row *i*
carries `202618000 + i`).

**Reserved on paper and REGISTERED NOWHERE by this document.** Registration is a separate, reviewed
edit under its own authorization, so a block that is never authorized costs nothing to abandon.

**The collision-proof procedure, frozen** (to run at registration time, not now):

1. Enumerate every seed in `ACCOUNTED`, `EXPOSED`, `RETIRED`, `TEST_ONLY`, `CONSUMED_SEEDS`, **and
   every paper-only reservation**, each named by its **own constant**, never by an alias.
2. Exclude the candidate from the prior set **by identity** if it is itself already a paper
   reservation — never by subtracting its seeds.
3. Check direct overlap **and** every derived RNG stream (the four frozen masks, both colours).
4. Apply the gap policy: **≥ 224 seeds** from every prior boundary, with the same identity exclusion
   applied to the gap check as to the overlap check.
5. Run **negative controls** that must all reject: each spent block, an interval straddling a boundary
   by one seed, a derived-stream collision with no direct overlap, and the candidate itself without
   the identity exclusion.
6. The enumeration must be **exhaustive**, not sampled, and the widest interval must be small enough
   for that to hold.

## 7. Outputs — locations frozen

Under `docs/superpowers/evidence/<run-date>-t1j-h2-deterministic-readout/`, create-only, with a
`A_MANIFEST.sha256.txt` over every entry and **no footer comment**, so `shasum -c` verifies cleanly:

| file | content |
|---|---|
| `01_h2_plan.json` | the 736 frozen tasks, sha256-pinned |
| `02_prerun_verification.txt` | gates, seed state, identity, output paths absent |
| `03_h2_results.jsonl` | one record per game |
| `04_h2_trace.jsonl` | predeclared event schema, fsynced per line |
| `05_run_command.txt`, `06_stdout.txt`, `07_stderr.txt`, `08_exit.txt` | the invocation and its outcome |
| `09_report.json` | score, interval (labelled nominal), outcome, **all 16 per-cell distinct-transcript counts**, cap terminations, by-cell table |
| `10_accounting_from_records.txt` | seeds drawn, counted from the records |
| `11_outcome.md` | the verdict and what it does not establish |

## 8. What this may NOT be used for

- **No pooling** with H1 attempt 1 or attempt 2, or with L0. Different readout, different seeds,
  different question.
- **No causal claim from comparing H2's rate with H1's.** The readout differs *and* the seeds differ,
  so any difference is confounded by construction; a comparison may be *displayed* beside its
  confound, never interpreted as the effect of the readout.
- **No training decision.** A `T1J_STRONGER` outcome makes a training question worth *designing*; it
  authorizes nothing.
- **No claim about the readout's effect on our model's strength**, which would need a matched
  argmax-vs-temperature comparison this design does not run.
- **No claim beyond these frozen conditions** — 8 openings, mdPly 6, 400 simulations, ply cap 280.
- **No claim that argmax is the best deterministic readout.** `T1J_STRONGER` and `NOT_STRONGER` are
  statements about *this* configuration; another deterministic rule could beat it, and comparing them
  is a different study.
- **No coverage guarantee.** The interval is nominal under §3.1's independence model; the design does
  not establish that model, and the transcript screen does not test it.

## 9. Refusals the implementation must carry (for its own authorization)

1. The gate is a reviewed one-line change; the runner reads it and the CLI reads it, and neither
   accepts a flag, an environment variable or a config file.
2. The schedule must be the **exact 736-task frozen plan** by digest; a short schedule must refuse,
   because a budget is a ceiling and bounds nothing below.
3. Every seed **registered** before any is drawn; row *i* carries `202618000 + i`.
4. The recorded incumbent identity must carry `selection_mode: "argmax"`; anything else VOIDs.
5. The gate is restored **immediately and unconditionally** after the process exits, whatever the
   outcome, and process-tree cleanup is verified.
6. The per-cell transcript screen of §2.1 is computed **before** the primary interval, and a cell
   below 42 distinct **prevents** the interval from being computed at all — a screen that runs after
   the number it guards is decoration. A control must prove that a single collapsed cell refuses,
   including the case where the global distinct rate still exceeds 90%.
6a. **The transcript is built from §2.2's fields and no others**, with controls that prove:
   - **46 games with identical gameplay but different `seed`, `task_id` and `rep` count as ONE
     transcript, and the cell FAILS** — the vacuity this definition exists to prevent;
   - **changing a single played move creates a distinct transcript**, so the comparison is not inert
     in the other direction;
   - a **missing, duplicated or out-of-order ply record REFUSES by name**, and does not become a
     novel transcript that inflates the count — with separate controls that **remove the FIRST ply**,
     **remove the FINAL ply**, and **flip every mover while preserving alternation**. All three must
     VOID; the first two are the cases plain contiguity cannot see, and the third is the one an
     alternation check cannot see;
   - a terminal reason outside `("win", "cap")` **refuses**;
   - the report carries **all 16 per-cell distinct counts**, not a global figure and not only the
     minimum, so a reader can see which cell was thin.
7. Every reported interval carries "nominal under the independence model" **in the report itself**,
   not only in this card.
8. Injected-defect controls for each of the above, each proven to reject, over a clean baseline.

## 10. What this document does NOT do

It does not implement, authorize, register, run, or push anything; it does not open confirmation data;
and it does not revisit the closed D1 line, whose verdicts stand.
