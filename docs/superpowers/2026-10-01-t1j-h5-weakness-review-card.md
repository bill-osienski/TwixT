# H5 POST-HOC WEAKNESS REVIEW — EXPLORATORY CARD (design only)

**Written 2026-10-01. DESIGN ONLY.** No individual H5 game has been inspected for this
card; no analysis has been run. The analysis itself waits for review of this card and
its own authorization.

🔴 **EXPLORATORY.** Every artifact this review produces carries that label. Nothing here
amends, qualifies, re-tests or adds to H5's confirmatory claim (directional
replication, 0.5946, nominal 95% [0.5157, 0.6735], `8ca7278`) or to H4's. **No H4–H5
pooling.** No model change, no strength rerun, no new confirmatory claim, no p-value,
interval or significance test. A finding here is a lead for a *separately carded*
experiment, never a result.

---

## 0. Inputs and what was already seen

**Data — H5 only, read-only:** the four committed segment results
(`docs/superpowers/evidence/t1j-h5-study-segment{0..3}/results.jsonl`, sha256
`e9c54d5f…`, `7e1ca0b9…`, `d13a3bd8…`, `4ff64341…`) and the committed study manifest
(`e08c5f35…`), each checked durable and hash-equal before use. H4's games are not used.

**Disclosed — seen before this card was written** (all aggregate, from the committed
H5 estimate `73600cf2…`; no individual game, loss list or trajectory):
per-pair outcome-pattern counts; 40 cap-affected pairs / 42 capped games; per-segment
means; per-arm opening-prefix frequencies (plies 1–6) and unique-game counts (Arm A 52
of 296 — the incumbent opened 18,19 in every Arm-A game; Arm B 125 of 296).

**Sequencing.** This card is committed **before** the H5 replay export (authorized with
it). The export's viewer checks display 16 individual games for *rendering*
verification only; they are not part of this review, and nothing seen there may change
these questions.

## 1. Terms (fixed here)

| term | meaning |
|---|---|
| **ours / incumbent** | `calib020_0001`, the manifest's `incumbent_identity` |
| **Arm A / Arm B** | the manifest's arms: A = incumbent **red** (moves first), T1j black; B = incumbent **black**, T1j red |
| **incumbent loss** | `winner` is T1j's colour |
| **incumbent win** | `winner` is the incumbent's colour |
| **capped** | `terminal_reason == "cap"`: **UNRESOLVED** — never a loss, never a win, never pooled with either |
| **trajectory** | the full move sequence from the empty board |
| **distinct trajectory** | exact move-sequence identity; its **frequency** = the number of games that share it |
| **occurrence-weighted** | every game counts once (frequency-weighted over distinct trajectories) |

Every count is reported **twice — occurrence-weighted and distinct — and separately by
arm**, and every distinct trajectory keeps its frequency. Neither denominator is
reported alone (D0's rule; here, with a deterministic incumbent, the two can differ by
a factor of several).

## 2. What D0 established, and what this adds

**D0** (`93457e4`, `2026-08-27-t1j-d0-postmortem.md`): 32 discovery games of the L0
match — 8 **fixed 6-ply openings**, T1j at mdPly 6 — structural ply features aggregated
by opening, arm, winner and phase; **no capped game**; no trajectory inspected one by
one; no move labelled good or bad. Verdict: the gate returned GO, but **no signature
was asymmetric between the engines** (`created_threat` 0.050 vs 0.051;
`mover_more_fragmented` 0.330 vs 0.386 — we were the *less* fragmented side), so there
was **no actionable weakness**; late-game fragmentation is a property of *losing at
TwixT*, not of our model. Its `unanswered_immediate_threat` restated the ending (every
hit at the penultimate ply).

**This review adds:**
1. **Empty-board play** under the H4/H5 protocol, 296 games per arm — D0 never saw the
   empty board; every D0 game started from a scripted 6-ply opening.
2. **Caps** — 42 capped games exist here; D0 had none.
3. **Recurring trajectories.** The incumbent is deterministic (argmax), so the same
   game can recur many times. A recurring *loss trajectory* is a concrete, repeatable
   situation that can be replayed and inspected, which an aggregate signature is not.
4. **A stricter definition of a demonstrated defensive error** (§4, Q2): D0's
   `ignored_threat` counted a ply as "ignored" whenever the opponent still had a
   one-ply win afterwards, **without checking that any move could have prevented it**.
   Here an error is *demonstrated* only if a saving move existed.

**Inherited from D0, applied prospectively:** both denominators; **a category whose
every hit sits at one fixed distance from the terminal ply is flagged as restating the
ending** (D0's fifth condition — its user ruling allows adding it prospectively, never
retroactively); **no attribution to policy, value or search** (H5's ply records carry
none of those — only actor, mover, move, ply and timing); **every finding is cut by
which engine moved**, because D0's decisive result was that its signatures were
symmetric.

## 3. The frozen questions

> **Q1.** Which empty-board opening and response patterns recur in incumbent losses?
>
> **Q2.** Where can replay demonstrate a missed connection or defensive opportunity?
> Distinguish demonstrated errors from hypotheses.
>
> **Q3.** What recurring situations produce caps? Capped games are unresolved, not
> losses.

## 4. Procedure, in order

**Step 1 — full outcome inventory (all 592 games).** Re-derived from the committed
results with H4's reader (`h4_runner.load_games`; integrity as at aggregation). Per
arm: incumbent wins / losses / caps; game-length distribution; terminal reasons;
distinct trajectories with frequencies — every count occurrence-weighted **and**
distinct. Cross-check: the per-arm totals must reproduce the committed estimate's
aggregates exactly, or the review stops.

**Step 2 — Q1, prefix tables.** Per arm, for prefix depths **2, 4, 6, 8, 10 and 12
plies**: every prefix that occurs, with (games, wins, losses, caps) occurrence-weighted
and distinct. A pattern **recurs in losses** if it appears in **at least two distinct
loss trajectories**; it is reported with its win and cap counts beside it, so a
prefix common to wins and losses is visibly not loss-specific. The prefix *responses*
are labelled by which engine played each ply.

**Step 3 — Q2, mechanical checks, then inspection.** For every ply of **every distinct
trajectory** — losses, wins and caps, so each check has a comparison — computed with
our rules engine on the replayed position (D0's `one_ply_threats`, reused):

| label | definition (all one-ply; DEMONSTRATED) |
|---|---|
| **MISSED_WIN** | the mover had a legal move that wins immediately, and played a different move that does not win |
| **MISSED_DEFENCE** | the opponent had ≥ 1 immediate win; **at least one legal move of the mover leaves the opponent with none**; the mover played a move that leaves at least one |
| **UNDEFENDABLE** | the opponent had ≥ 1 immediate win and **no** legal move leaves it with none — *not an error at this ply*; reported so it cannot be mistaken for one |

Each hit records: trajectory id and frequency, arm, ply, **which engine moved**, the
move played, the saving/winning moves available, and its distance from the terminal
ply. Rates are reported for **both engines** side by side, occurrence-weighted and
distinct; the fixed-distance flag (§2) is applied per label. **Cost is measured on a
3-trajectory sample before the full run** (the MISSED_DEFENCE existence check scans
every legal reply); if projected over 2 h the run stops and returns for a decision.

Then **inspection**: every distinct loss trajectory and every distinct capped trajectory,
in descending frequency, is stepped through in its exported replay. Anything not one of
the labels above — a slow plan, a bridge left cuttable two plies later, an "inferior"
shape — is written down as a **HYPOTHESIS**, with file and ply, and **never** counted
as demonstrated. Hypotheses carry no rate.

**Step 4 — Q3, caps.** For the capped trajectories (distinct and occurrence-weighted,
by arm): the prefix tables of Step 2 restricted to caps, and the position at the cap
described by D0's structural features (pegs, bridges, components, nearest
goal distances per colour) — descriptive only. A **recurring cap situation** is one
shared by at least two distinct capped trajectories (same prefix at some depth, or the
same feature profile at the cap); each is reported with its frequency. Caps are never
added to losses or wins anywhere.

## 5. Outputs

A create-only evidence directory,
`docs/superpowers/evidence/<date>-t1j-h5-weakness-review-EXPLORATORY/`:
`01_inventory`, `02_prefix_tables`, `03_mechanical_checks`, `04_caps`,
`05_hypotheses`, and a report whose title and every section header say
**EXPLORATORY**, plus the exact inputs' sha256. Read-only code in its own module and
tests, outside every bound `code` list; it opens no gate (none is needed: no model,
JVM, T1j query, game or seed).

## 6. What this review may and may not conclude

It **may** say: a trajectory or prefix recurs among incumbent losses (with both counts
and its frequency); a one-ply error is *demonstrated* at a given ply, by which engine;
a situation recurs among caps.

It **may not**: call a move bad because the mover later lost (D0 §4.1); attribute
anything to policy, value or search; treat a cap as a loss; pool with H4; report an
interval, test or p-value; describe a difference between engines as established; or
change any model, card, bound file or result. A weakness that looks actionable becomes
the hypothesis of a **new, separately carded** experiment.

**Next separately authorized action:** review of this card; then the analysis.

---

## 7. Clarifications — 2026-10-01, committed BEFORE any diagnostic runs (user-decided)

**C1 — Terminal precedence and the three labels (replaces §4 Step 3's table).** At a
pre-move position P, mover X, opponent Y: **W** = X's immediate wins; **T** = Y's
immediate wins *were Y to move* — **board threats**, hypothetical while X is to move.
**An immediate winning move counts as a saving move**, and no opponent reply after a win
is ever examined. Labels are exclusive per ply, in this precedence:

| label | condition |
|---|---|
| **MISSED_WIN** | W ≠ ∅ and the played move ∉ W. Witness: every move of W (and T, if non-empty) |
| **MISSED_DEFENCE** | W = ∅, T ≠ ∅, the played move leaves Y ≥ 1 immediate win, and **some legal move leaves Y none**. Witness: every such saving move, and Y's wins the played move left |
| **UNDEFENDABLE** | W = ∅, T ≠ ∅, and **neither an immediate win nor any move removing every immediate Y win exists**. Not an error |
| *(DEFENDED, context only)* | W = ∅, T ≠ ∅, and the played move leaves Y no immediate win |

**Board threats vs replies.** Every hit records T (board threats before the move), the
Y wins the played move **left** (Y's actual replies), whether that reply was
**available** — false only when the game **capped** at this ply, so Y never moved — and
whether Y **won on the next ply**. A miss whose reply was unavailable because of the cap
is reported apart from one whose reply was available.

**Exact reduction, disclosed.** An X peg can only remove Y connectivity, so Y's immediate
wins after any X move are a **subset of T**. The defence check therefore tests only the
moves of T after each candidate. This is (a) verified against the full scan in the
synthetic tests, and (b) **asserted on every played move**: the Y wins left by the move
(the next ply's full W scan, or a full scan at a capped final ply) must be ⊆ T — a
violation stops the run. W and T themselves are always full scans (D0's
`one_ply_threats`).

**C2 — Denominators.** Each label's rate = hits / **inspected pre-move positions**,
separately by **engine** (incumbent, T1j) and **arm**; **occurrence-weighted** (every
game's positions) and **distinct** (each distinct trajectory's positions once);
numerator and denominator always shown. Every pre-move position of every distinct
trajectory is inspected — no sampling.

**C3 — Timing sample and the frozen projection.** Distinct trajectories are keyed by
(arm, moves); **trajectory digest** = sha256 of the canonical JSON
`{"arm": …, "moves": [[r, c], …]}`. Sorted by (length, digest): the sample is the first
(**shortest**), the element at index (n − 1) // 2 (**median-length**) and the last
(**longest**). Projection, frozen before timing:

```text
projected_s = (t_short + t_median + t_long) / (n_short + n_median + n_long) × N
N = total pre-move positions over all distinct trajectories
```

If projected_s > 7,200 the full run does not start; during the full run, it stops at
7,200 s elapsed and reports what completed. No silent sampling, no pruning.
