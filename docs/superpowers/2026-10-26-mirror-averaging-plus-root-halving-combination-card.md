# Mirror averaging + root sequential halving — one fixed combination (design-only card)

**Date:** 2026-10-26

**Status:** DRAFT, for the user's review.
- **Authorized:** this design-only card, including a narrowly scoped proposal to reopen #47, #59 and #62 for this
  combination only (§7).
- **Not authorized:** a build, model loading, seeds, GPU work, games and pushing.

## 1. Question

Does one fixed combination beat the unchanged `calib020_0001` at approximately matched network evaluations under
`MLX_ENABLE_TF32=0`? The combination is mirror averaging plus root sequential halving.

**This card tests whether the combination wins, not why.** A win would not establish synergy, attribute the gain to
either component, or show that either component helps alone. No factorial study is needed to answer the question.

## 2. The candidate, frozen without tuning

| Element | Setting |
|---|---|
| Evaluator | `MirrorAveragingEvaluator` (`scripts/GPU/alphazero/mirror_averaging.py`) over an eval-mode `calib020_0001`: mean of the masked priors and of the values of a position and its left-right reflection, two network calls per leaf |
| Budget | **200 simulations** |
| Plies 0–19 | Ordinary search and the existing opening readout, unchanged |
| From ply 20 | `RootSequentialHalvingMCTS` (`scripts/GPU/alphazero/root_sequential_halving.py`), unchanged: 16 highest-prior candidates; the same score, ties and allocation formula; the top-ranked final survivor is played |
| Opponent | The unchanged incumbent at **400 simulations**, ordinary search and readout |
| Both sides | Process-local `MLX_ENABLE_TF32=0`; eval mode; Stage B protocol (24×24, 280-ply cap, opening-temperature readout, no root noise) |

**What changes at 200 simulations.** The frozen rule allocates 3/6/12/28 visits per survivor (cumulative 3/9/21/49),
instead of the 6/12/25/54 it used at 400. This follows from the unchanged formula, and it has never been tested. It is
the combination's main untested element.

**Known limit, unchanged.** A move outside the 16 candidates is never searched from ply 20.

**Why test it** (a hypothesis, not a measured fact). The two components act at different layers:
- Averaging enforces reflection consistency in each evaluation. The orientations' values differed by up to 0.64 on
  four positions (entry 8).
- Halving changes root allocation and the final choice.

Whether averaging improves evaluation accuracy, or the decisions halving makes from few visits, is not measured.

## 3. The evidence, stated as recorded

| | Mirror averaging (entry 11, #59) | Root halving (entry 15, #62) |
|---|---|---|
| Comparison | 200 averaged vs 400 ordinary | 400 vs 400 ordinary |
| Score | 19–13, 16 pairs: **5 × 1.0, 9 split, 2 × 0.0**; mean pair 0.594, Hoeffding95 [0.254, 0.933] | 21–1–10, 16 pairs: 6 × 1.0, 1 × 0.75, 8 × 0.5, 1 × 0.0; mean pair **0.672**, unclipped [0.332, 1.011] |
| Verdict | STOP_CONFIGURATION (bar 24/32); no advantage established | STOP_CONFIGURATION (bar 24/32); no advantage established |
| Network evaluations | 323,608 vs 323,983 | 376,168 vs 377,822 |
| Search seconds | 696 vs 840 | 1,065.5 vs 1,059.3 |
| Configuration | Eval mode; old M3 Pro; Python 3.14.7, MLX 0.30.3; default TF32 setting | The same day, protocol and opponent |

**The configuration differs from this card's run** (M5 Max, Python 3.14.9, `MLX_ENABLE_TF32=0`). Neither score can
serve as a current baseline, and the two estimates are not to be added together.

**The combination's runtime is unmeasured.** Averaging alone was cheaper in search time than its opponent, but that
does not establish that adding halving will also be cheaper.

## 4. Fair comparison

- **The currency is network evaluations,** as in entry 11. The combination spends 200 simulations × 2 calls; the
  incumbent spends 400 × 1. Simulation counts alone do not guarantee the match: terminal leaves, tree reuse and
  halving's round-robin can all shift the totals.
- **Recorded per side,** in entry 11's fields: searches, halving searches, leaf infer calls, network infer calls,
  network seconds and search seconds.
- **Matching guards, frozen here as judgement calls:**
  - the combination's network evaluations ≤ 1.05 × the incumbent's;
  - its search seconds ≤ 1.10 × the incumbent's.

  §5 applies them at C1 and at the strength step.
- **Not a fair comparison:** an equal-simulation match (combination at 400 vs 400), which would give the combination
  twice the network compute.

## 5. Steps, each run at most once and each with its own approval

**B. Build (CPU only).**
- One match script, a copy of `docs/superpowers/evidence/root-halving-match/01_match.py` with exactly one added
  difference: the candidate agent's evaluator is wrapped as in `.../mirror-averaging-match/01_match.py` (lines
  175–200). Cost fields as in both scripts.
- Wrapper, seed survey and closing: copies of the existing ones. No new framework.
- **Focused tests (stub evaluators):**
  - halving receives the wrapper's averaged priors and values;
  - plies 0–19 of the combination equal averaging alone;
  - halving over a pass-through wrapper equals halving alone (visits and choice);
  - halving's frozen allocation at 200 simulations gives the counts stated in §2.
- **Loop-equivalence check:** a copy of `.../root-halving-match/03_loop_check.py`, extended to exercise the
  `last_choice` branch, which entry 15's check never did. It runs with fake evaluators at 200 and 400 simulations.
- **Regression:** must equal the 58-id baseline, run in the main checkout.

**C1. Composition-and-cost check: one bounded GPU run, no games, no seeds, about 15 minutes.**
- **Positions:** the committed positions timed in entry 19, which span plies before and after 20.
- **Checks:**
  - entry 10's averaging correctness check, re-run unchanged against its independent reference and tolerance;
  - the combination equals averaging alone at plies < 20 (bitwise);
  - halving over a pass-through wrapper equals plain halving (bitwise).
- **Measured on the same positions:** each side's network evaluations and search seconds per search. These also
  price §6.
- **Recorded:** `MLX_ENABLE_TF32` `"0"` at start and end, versions, and both networks in eval mode.
- **STOP** on any failed check, or if either matching guard (§4) is exceeded.

**S. Gross-regression screen.**
- 32 fresh colour-reversed pairs (64 games), combination 200 vs incumbent 400.
- **STOP if the mean pair score is below 0.40.** This catches broken behaviour only; it makes no strength claim.

**T. Strength comparison, funded in full before launch.**
- 509 fresh pairs (1,018 games), a fixed size, with no early look and no extension.
- **Pass** iff mean − √(ln 40 / (2·509)) > 0.5, i.e. a mean pair score above **0.56020**, **and** both matching
  guards (§4) hold over the whole run.
  - A score above the bound that fails a guard is reported as **NOT_COMPUTE_MATCHED**, not as a pass.
- **Statistics:** state caps count as draws, and pairs are assumed independent. The size gives conservative 80%
  power at a true mean pair score of 0.60.
- **A non-pass** can rule out a large benefit or reveal harm, but cannot resolve modest effects.

**Seeds.** Fresh blocks for S (64) and T (1,018), each through survey → pre-proof → registration → post-proof →
closing (K1, K2). No old block is reused.

**Stops for every step.**
- A STOP, VOID or timeout closes the combination, with no retry, top-up, extension, budget change or selection.
- An incomplete run gets no score.
- Before each GPU step: a read-only kernel-panic check and an idle machine. A new panic stops all GPU experiments,
  under attempt 2's rule.

## 6. Cost (estimates; the combination's runtime is unmeasured)

| Step | Estimate | Basis |
|---|---|---|
| C1 | ≤ 15 min | A few dozen searches per side |
| S | ~0.75 h | 64 games × 41.7 s of search per game at 400 vs 400 (entry 19, derived, default setting) |
| T | ~11.8 h search alone | 1,018 games at the same derived rate (methods note, entry 19) |
| Total | **~13–16 h** of machine time, plus the build | Whole-game time and the TF32=0 effect on the play path unmeasured |

- **These assume** the combination's search time per game is comparable to the incumbent's. C1 measures this.
- **Ceilings for S and T:** fixed from C1's measured seconds per search before S is approved, at 1.5× the projected
  time. A timeout is a VOID.
- **If C1 projects T beyond 30 h,** the card is repriced and put forward again, not silently resized.
- **Resolving a true score near 0.55** would need 2,035 pairs (~47 h derived). This card does not fund that.

## 7. Proposed reopening: this combination only

| Ruling | What it forbids | Proposed narrow exception |
|---|---|---|
| **#47** | Tree-local search heuristics. The exception for one halving variant is spent | A new exception for exactly §2's unchanged halving rule, inside this combination only |
| **#59** | Changing averaging's budgets, or rescuing it with another readout | Averaging at its tested 200 simulations, with halving's choice from ply 20 as the readout, inside this combination only |
| **#62** | Changing halving's simulation budget, or treating the 21–10 lean as earning a strength comparison | Halving's unchanged rule at 200 simulations, and one strength comparison, for this combination only |

**Still closed:**
- either component run alone, and any rerun or extension of entries 11 or 15;
- any other budget, candidate count, score, tie, start ply or orientation;
- re-scoring or reusing the old evidence or seed blocks;
- any second combination.

## 8. What a pass would and would not show

- **A pass would show:** this fixed combination beat the incumbent at approximately matched network evaluations, under
  this protocol, machine and setting.
- **It would not show:**
  - synergy, or which component contributed;
  - that either component helps alone;
  - anything about training, T1j or twixtbot.
- **Production** is unchanged, and any deployment is a separate decision. The browser would need both components
  reimplemented: a second ONNX call per leaf, and halving in its search.

## 9. Prediction on record

**None stated in advance.** The two screens' estimates are not a forecast for the combination.

## 10. Corrections to the read-only review that proposed this card (2026-10-26)

- Mirror averaging had **9 split pairs, not 10**. The review copied entry 11's summary, which the committed record
  (`.screen.pair_scores`) contradicts; the ledger carries a dated correction.
- That averaging improves evaluation accuracy, or halving's decisions, is a **hypothesis**.
- The combination's runtime is **unmeasured**. Averaging alone being cheaper does not establish it.
- Both screens ran in eval mode, but on a different machine and setting. **Their scores are not current baselines.**
