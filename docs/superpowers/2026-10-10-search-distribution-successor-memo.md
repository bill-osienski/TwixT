# Search-distribution successor — decision memo (DESIGN ONLY, 2026-10-10)

**Question.** Is a successor that trains on the teacher's root **visit distributions** (instead of
one chosen move) worth attempting as a way to transfer playing strength?

**Decision: not as a successor card now.** The distributions are available under our rules, but at the
budget they exist they add little beyond the one-hot label. Nothing yet links a richer target from this
teacher to stronger play. At most, a teacher-only information probe could be considered later (below).
No experiment is authorized by this memo.

> **Corrected on review (2026-10-10) — read §6 first.** Three interpretations in §2–§4 overreach; §6
> supersedes them. The decision stands, and so does withholding the probe.

Context: the closed trial (`2026-10-10-external-engine-learning-result.md`, ledger #53) improved held-out
imitation but left strength unresolved at an unfavourable 12–20. `calib020_0001` remains the incumbent.

## 1. Can we export root visit distributions under our exact rules? — Yes; they already were

Read from the pinned teacher source (`twixtbot-ui` @ `83749f23`, `src/backend/`) and the committed stage-1
evidence:

* `NeuralMCTS.mcts()` returns `root.N`, the full visit vector over all 528 policy indices.
  `create_response` sorts it into `Y` with `MAXBEST = 24**24`, i.e. every index, not a head slice.
* Stage 1's capture (`twixtbot_g3_harness.anchor_move`) recorded the full `Y` and its move order for
  **all 192 teacher moves**, in `evidence/external-engine-learning-stage1/games/*.json` (`visits`,
  `moves_order`). Settings: 100 trials, no noise, temperature 0, no rotation, no swap, `smart_root` 0.
* **Rules.** Search plays in place on a `Game` built with `allow_scl=False`. The draw-detection scratch
  board (`twixt.py`, `_has_end_to_end_path`) builds `Game(True)` but its crossing checks read the live
  game's `allow_scl`. The teacher's legal mask is the mover's empty points. In the 120 usable training
  rows, **no visit fell outside our legal set**.
* **Caveats for any export:**
  * A proven root returns a move, not a distribution. That happened for 8 training rows, each carrying a
    single visit.
  * Moves proven losing are dropped from the legal mask but keep their visits, so they must be masked.
  * `compute_root` reuses the subtree within a game, so a root can inherit visits. None appeared (all 120
    totals are exactly 100), but an export must use a fresh root or record the inherited counts.

## 2. Are they informative? — Mostly one-hot at this budget

Training rows only (120 usable of 128; the sealed held-out rows were not inspected):

| per row | lower quartile | median | upper quartile |
|---|---|---|---|
| moves with any visit | 1 | 2 | 3 |
| chosen move's share of visits | 0.62 | 0.835 | 1.0 |
| effective number of moves (1/Σp²) | 1.0 | 1.39 | 1.93 |
| entropy (nats; uniform over legal ≈ 6.2) | 0.0 | 0.46 | 0.83 |

* **Mostly near one-hot:** 45 rows are exactly one-hot, and 60 more spread over just 2–3 moves.
* **The diffuse rows look like noise:** 12 rows spread visits over 11–99 moves, some at 2–10 visits per
  move. All 12 are teacher-as-black at plies 5–17. That is consistent with a search in which every move
  looks alike (not verified). As targets they would pull our policy toward uniform. Dropping them would be
  a filter chosen after seeing them, so it would have to be preregistered on fresh rows.
* **The visits largely echo the teacher's own prior:** at 100 PUCT trials, mostly that is what they
  reflect. The teacher's network was trained with own-link crossings allowed, so the rule mismatch carries
  straight into the target.

## 3. Would it transfer strength? — No argument yet

* The closed trial already moved our policy toward this teacher (top-1 20.5% → 25.7%) and scored 12–20.
  A soft target from the same teacher at the same budget mostly repeats that signal; it does not explain
  why agreement did not become strength.
* The AlphaZero precedent for visit targets is self-play: the distribution comes from the learner's own
  search under its own rules and improves on its prior. Here it comes from a different network trained
  under different rules, at a shallow budget.
* **A richer target is possible but has the same flaw:** for searched moves the teacher's full prior
  (`resp["P"]`) is returned and was discarded by the capture. Logit distillation would carry more
  information than visits, but even more directly inherits the rule mismatch.
* More steps, rows or unfrozen layers are excluded as arguments (as you noted), and do-not-repeat #53
  forbids retraining the closed trial's candidate.

## 4. What would justify a card, and the cheapest falsifier

A successor card is warranted only if a **teacher-only information probe** first shows that a deeper
search changes the target materially. The probe would work like this:

* Re-query a fixed set of committed stage-1 positions at higher trial counts (for example 400 and 1000).
* Use a fresh root per query, noise off, deterministic.
* Play no games and draw no seeds.
* Measure whether entropy and alternative mass rise and stay stable across trial counts, and whether the
  diffuse black-opening rows persist.

**Cheapest falsifier:** if visits at 1000 trials still sit near one-hot (for example median effective
number below about 2) or stay diffuse in the same positions, the visit-target line is dead with no
training and no games.

Even a positive probe establishes information content only, not strength. A strength claim needs a
match far larger than 32 games, as the frozen-parent line's 400 games showed (interval width about ±0.05).
Any such card must state that cost up front.

## 5. Prerequisites before any future run (your step 1; not designed here)

1. **Unverified cleanup needs its own failure status.** An exception escaping `EL.end_group` must map to
   a distinct exit status, not the 1 it shares with STOP_HELDOUT. It must keep the cleanup-failed record
   and still refuse to finalize. `PermissionError` must not be read as "group gone".
2. **A replacement seed-selection rule.** The pinned rule's next base is the 2026 band's upper end. A
   replacement must preserve every existing reservation, exposure and retirement, and needs its own review.

Both need focused synthetic tests with all gates closed, and each needs separate authorization.

## Recommendation

Keep the candidate archived. Do **not** write a search-distribution successor card now. If you want to
keep the option open, the only step worth authorizing is the teacher-only probe in §4, after the two
prerequisites, judged by its falsifier before any training is designed.

## 6. Review corrections (2026-10-10) — supersede §2–§4 where they conflict

The reviewer independently confirmed the training-only figures: 120 usable rows, 8 proven roots, zero
visits outside our legal set, 45 one-hot rows, median effective move count 1.39.

1. **Concentration is not a falsifier on its own.** A strong search can be confidently concentrated on one
   move, and higher entropy can mean uncertainty rather than useful information. §4's rule ("median
   effective count below about 2 means the line is dead") is **withdrawn as unsupported**. A probe needs a
   criterion that measures whether search changed the target **usefully**, not how soft the target is.
2. **Diffuse visits do not establish noise, and concentrated visits do not establish prior imitation.**
   §2's readings (the 12 diffuse rows as "noise"; visits that "largely echo the teacher's own prior") are
   **hypotheses only**. The capture kept no raw priors, so how much search changed them cannot be
   measured from this data.
3. **The 12–20 result supplies caution, not an explanation.** It neither establishes failed strength
   transfer nor identifies why transfer might fail; §3's reading that a softer target "mostly repeats that
   signal" is a hypothesis. Any future match must be priced with its own statistic and interval method:
   the frozen-parent line's 400-game interval in §4 is **not** a transferable precision guarantee.

**Net:** there is no qualified argument for another training experiment yet. The teacher-only probe in §4
also stays **unapproved** until it has a criterion that measures useful search changes rather than target
softness. No run, card, training or push is authorized.
