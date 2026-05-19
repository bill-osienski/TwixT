# Marathon Termination Tuning — Design Spec

**Date:** 2026-05-19
**Status:** Drafted (awaiting user review)
**Predecessors:**
- [2026-05-19-long-tail-bucket-classifier-design.md](2026-05-19-long-tail-bucket-classifier-design.md) — produced the 80.8% marathon concentration that motivates this spec
- [2026-05-12-recovery-retargeting-diagnostic-design.md](2026-05-12-recovery-retargeting-diagnostic-design.md), [2026-05-13-recovery-retargeting-filtered-view-design.md](2026-05-13-recovery-retargeting-filtered-view-design.md)
- [2026-05-10-closeout-tail-correction-design.md](2026-05-10-closeout-tail-correction-design.md)

**Successor:** None planned. If the diagnostics in §5 surface a clear failure pattern, a single follow-up termination/adjudication intervention; not a Spec-5-style training-target change.

---

## 1. Goal & Scope

### 1.1 Goal

Reduce the volume of training-uninformative marathon and state-cap games by classifying their no-progress shape and applying a targeted termination / adjudication fix. **Diagnostic-first**: before changing any termination rule, add the no-progress diagnostics that will tell us *which* marathon shape is dominant (resign-gate-blocked, adjudication-blocked, or genuinely-ambiguous).

### 1.2 Why this, not closeout

The 190-219 long-tail bucket counts (n=52 games over ~3000, 1.7% of all training games):

| Bucket | Count | Share |
|---|---:|---:|
| marathon_or_state_cap | 42 | **80.8%** |
| td2_alt_in_top5 | 5 | 9.6% |
| td2_reducer_buried | 4 | 7.7% |
| dominant_unavailable_contested | 1 | 1.9% |
| td3_drift | 0 | 0% |

80% of the remaining tail is structural marathon, not closeout-selection failure. The two closeout-side experiments already tried both regressed:

- **Fix 2 gate 0.95 → 0.90**: more overrides, worse tail.
- **conversion-policy-loss weight 0.05 → 0.075**: worse td=2 ranking, more state-cap pressure.

The marathon population is too rare (1.7%) and too concentrated (80%) to justify broad training-loss changes. A termination/adjudication-focused fix is much more likely to reduce wasted compute and bad replay tail without distorting the policy.

### 1.3 In scope

- **Rollback** of two failed experiments to their prior defaults (§2).
- **Four new diagnostics** measuring no-progress patterns, resign-gate behavior on hopeless positions, and adjudication coverage (§3).
- **One follow-up termination intervention** selected from §3 results — likely either an early-state-cap rule, a relaxed adjudication threshold, or a relaxed resign top1 gate. Specifics deferred to a follow-up spec once §3 results land.
- **Analyzer-side wiring** for the new diagnostics: report section + per-iter CSV.
- **Tests** for the diagnostic computation (pure-function logic, sidecar-driven where possible).

### 1.4 Out of scope

- Any change to MCTS, training targets, conversion-aware loss, or closeout selection rules (Fix 1 / Fix 2 thresholds).
- Spec 5 (recovery-aware training) — explicitly NOT motivated by the bucket counts.
- New aux losses or policy targets.
- Per-position MCTS modifications.

---

## 2. Rollback of failed closeout experiments

Apply on the next training block, before any new diagnostic ships:

| Flag | Recently | Roll back to | Notes |
|---|---|---|---|
| `--closeout-selection-tiebreak-min-value` | 0.90 | **0.95** | 0.90 produced more overrides but worse tail metrics. The model isn't reliable enough at q=0.90 to make tie-break overrides net-positive. |
| `--conversion-policy-loss-weight` | 0.075 | **0.05** | 0.075 worsened td=2 ranking and increased state-cap pressure. Stronger conversion signal apparently pulls the policy away from healthy chain-extension during normal play. |

Both flags revert to the values used in the 190-219 baseline. No code change required — just the launch command flags. Memory should be updated to flag both experiments as **tried and reverted**, so they don't get re-attempted under a different framing.

---

## 3. The four no-progress diagnostics

All four are computed per-game from existing `goal_completion_diagnostics` per-ply entries and the game's `goal_completion_record`. No new self-play hook needed; the underlying data is already collected.

### 3.1 No-progress window detector (per-side)

For each side, slide a window of the last N=15 own-moves over the game and flag a "no-progress window" if **all** of:

```
no_goal_distance_reduction   : no own-move in the window reduced
                               total_goal_distance for this side
no_endpoint_completion       : no own-move in the window completed an endpoint
no_opponent_block            : no own-move in the window was a defensive block
                               (definition below)
moves_are_local_structural   : all moves in the window had primary_class in
                               {redundant_reinforcement, off_chain,
                                connects_to_existing_component,
                                improves_own_largest_component}
```

**"Opponent block" definition (pinned):** Use the same rule as the recovery-retargeting defense classifier (Spec 4 §3, `blocks_opponent_closeout`). For an own-move at ply t, the move is an opponent block iff:

```
opponent_total_goal_distance_before <= 2
AND opponent_total_goal_distance_after > opponent_total_goal_distance_before
```

If `goal_completion_diagnostics` entries already carry a `selected_move_classification.primary_class == "blocks_opponent_closeout"` value (Spec 4 vocabulary), prefer that flag — it's the same rule already computed. Otherwise compute inline from the entry's `goal_completion.total_goal_distance_before` and the next opponent-eval's `total_goal_distance_before` (which equals "after" from the own-move's perspective). The implementation MUST share a helper with `recovery_retargeting_diagnostics.classify_move` so the two diagnostics cannot diverge on what counts as a block.

Per-game metric: count of distinct no-progress windows per side (overlapping windows count once, anchored to the last ply of the longest such window).

### 3.2 Adjudication coverage diagnostic

For each game that ended at n_moves == 280 with reason == state_cap, compute:

```
ply_of_first_eligible_adjudication      : first ply >= adjudicate_min_ply
                                          where adjudicate eligibility held
ply_of_first_threshold_crossing         : first ply at which the would-be
                                          winner's value crossed adjudicate_threshold
gate_blocked_by                         : one of the taxonomy values below
plies_after_first_threshold_crossing    : 280 - ply_of_first_threshold_crossing
```

**Adjudication gate taxonomy (full enumeration):**

```
not_attempted          : ply >= adjudicate_min_ply was reached but no
                         adjudication attempt was logged for the game
                         (likely a coverage hole; signals a self-play bug
                         OR that adjudication only fires on a sparse
                         schedule we need to characterize)
value_below_threshold  : at least one attempt was logged, but no ply
                         had a side's value cross adjudicate_threshold
min_top1_share         : value crossed threshold AND visits met, but
                         root_top1_share was below adjudicate_min_top1_share
                         at every attempt
min_visits             : value crossed threshold AND top1 met, but visit
                         count was below adjudicate_min_visits
missing_signal         : per-ply data missing for ply >= adjudicate_min_ply
                         (no search_score or no top1_share); cannot classify
would_have_passed      : all gates passed at some ply but adjudication
                         still wasn't applied — bug indicator
```

When multiple gates blocked across the game's plies, classify by the **last-blocking** gate (i.e., the closest-to-passing). Ties resolved by precedence `min_top1_share > min_visits > value_below_threshold` (top1 typically tightens last).

Range-level rollup: distribution of `gate_blocked_by` across 280-ply state_cap games. Tells us which adjudication gate to relax (if any). The `not_attempted` and `would_have_passed` counts should both be 0 in a healthy pipeline; non-zero values flag a telemetry/wiring bug.

**Implementation pre-check (Task 1):** confirm whether per-ply adjudication-block reasons are already persisted in the existing `adjudication` sidecar block or per-game record. The current sidecar tracks aggregate `attempts` and `checks` counts but may not surface per-attempt gate-block causes. If absent, add a minimal self-play telemetry hook (one append to the adjudication telemetry per attempt: the gate that blocked it) **before** writing the analyzer-side aggregator. This is the only allowed self-play-side change in this spec; see §6 step 0.

### 3.3 Resign-gate-on-hopeless diagnostic

For each game's losing side (when winner is known), in the last 40 plies, compute these four counts explicitly (separating "no value signal" from "value signal blocked by top1" is the point):

```
value_hits           : count of own-plies where search_score < resign_threshold
                       (i.e., the value signal qualified for resign on its own)
eligible_hits        : count of value_hits that ALSO satisfied
                       min_visits AND >= resign_min_ply
                       (i.e., all gates met EXCEPT top1)
blocked_by_top1      : count of eligible_hits where root_top1_share was
                       below resign_min_top1_share (so resign was gated)
final_eval_below_thr : at the last own-ply for the loser,
                       search_score < resign_threshold (sanity check —
                       confirms the loser actually had a losing signal)
```

Derived rates (per game):

```
top1_block_rate_over_value_hits     : blocked_by_top1 / value_hits
                                      (was the value signal frequently
                                      present but mostly blocked by top1?)
top1_block_rate_over_eligible_hits  : blocked_by_top1 / eligible_hits
                                      (of resigns that ONLY needed top1
                                      to pass, what fraction were blocked?
                                      this is the cleanest "is top1 the
                                      problem?" rate)
```

Range-level rollup: **both** rates above, partitioned by game length:

```
short games (n_moves <= 100)    : sanity baseline
mid games   (100 < n_moves <= 200)
long games  (n_moves > 200)     : the ones we care about
```

If `top1_block_rate_over_eligible_hits` in long games is significantly higher than in short games (e.g. 2x), the top1 gate is too conservative late in hopeless games. The two-rate split avoids confusing "no losing value signal" (low `value_hits`) with "top1 gate prevented resignation" (high `top1_block_rate_over_eligible_hits`).

### 3.4 Stagnation-rate per-iter trend

Per-iter aggregation of §3.1: how often does a no-progress window of length 15 occur per game on average? Tracks whether the population is improving block-over-block, independent of the §3.2/§3.3 termination questions.

---

## 4. Output

### 4.1 Per-game record extension

Append to existing `goal_completion_record`:

```
no_progress_windows_red        : int — count from §3.1
no_progress_windows_black      : int — count from §3.1
adjudication_coverage          : dict from §3.2 (only populated when game
                                 ended in 280-ply state_cap)
resign_top1_gate_block_rate    : float from §3.3
```

### 4.2 New CSV `marathon_termination_by_iter_<range>.csv`

Long format, one row per iter, columns:

```
iteration,
games_total,
state_cap_280_games,
mean_no_progress_windows_per_game,
adjudication_gate_blocked_by_min_visits,
adjudication_gate_blocked_by_min_top1_share,
adjudication_gate_blocked_by_value_below_threshold,
adjudication_gate_never_blocked,
mean_resign_top1_block_rate_short_games,
mean_resign_top1_block_rate_mid_games,
mean_resign_top1_block_rate_long_games
```

Plus one range-total row with `iteration = -1` (matching the long-tail-buckets CSV convention).

### 4.3 Report section "Marathon termination diagnostics"

Inserted after the long-tail bucket counts section. Format:

```
Marathon termination diagnostics (<range>)
==========================================
state_cap 280-ply games: N
  adjudication gate blocked by:
    min_visits:                  N
    min_top1_share:              N
    value_below_threshold:       N
    never_blocked (eligible):    N

No-progress windows (length 15 own-moves, structural-only):
  mean per game: X.XX
  per-iter trend: ...

Resign top1-gate block rate (losing-side, last 40 plies):
  short games (n<=100):  XX.X%
  mid   games (n<=200):  XX.X%
  long  games (n>200):   XX.X%

Suggested termination action:
  (computed from the above; see spec §5)
```

The "Suggested termination action" line is the decision-rule output from §5.

---

## 5. Decision rule (after first run)

The diagnostic output feeds a single termination-knob choice, in this priority order:

1. **If `adjudication_gate_blocked_by_value_below_threshold` dominates** (>50% of 280-ply games): the model never crosses `adjudicate_threshold=0.20` even at ply 280. The position is genuinely ambiguous; **don't terminate** — the state-cap is appropriate. Consider whether `adjudicate_threshold` should be lowered (e.g., to 0.10) for very late plies (>270).

2. **If `adjudication_gate_blocked_by_min_top1_share` dominates**: relax `--adjudicate-min-top1-share` from 0.13 → 0.08 in late-game (>=260 ply) only, OR add a separate late-game adjudication tier with looser top1.

3. **If `adjudication_gate_blocked_by_min_visits` dominates**: investigate why visits are below 200 at the relevant plies — likely an MCTS budget or stall flush issue, not adjudication.

4. **If `mean_resign_top1_block_rate_long_games` is >2x `short_games`**: relax `--resign-min-top1-share` from 0.102 → 0.05 for late-game (>=200 ply) — top1 gating in hopeless positions is too strict.

5. **If `mean_no_progress_windows_per_game` is high (>2 per game)** AND none of (1)-(4) trigger: the candidate action is an **early state-cap** rule that ends the game after K consecutive no-progress windows for either side. K=2 is the starting point.

   **The early state-cap action is diagnostic-gated and default-off.** It is NOT enabled by the diagnostic rollout itself. It is enabled in a follow-up treatment run only if §5 selects it as the dominant remedy AND the user explicitly approves the treatment.

6. **If none of (1)-(5) clearly dominates**: the marathon shape is heterogeneous; defer to a hand-review of 5-10 representative cases before another knob change.

Each branch produces at most one CLI/config change. No simultaneous tuning of multiple termination knobs.

### 5.1 Value-uncertain guard (applies to any termination knob)

Any termination/adjudication change selected by §5 — especially early state-cap — MUST include a value-uncertain guard before terminating:

```
Do not terminate early if both sides' recent root values (window
of last 10 own-plies, both sides) remain near neutral (|search_score| < 0.30)
or oscillatory (sign-flip count >= 3), unless the chosen decision rule
explicitly selected state-cap-by-stagnation AND the required consecutive
no-progress window count K has been met.
```

Rationale: we do not want to cut off genuinely contested games just because they are long. A game where both sides are still searching meaningfully (mid-range root values, no stable assessment) is the OPPOSITE of a marathon — it's a hard contested position that the model is correctly treating as uncertain. Terminating those would corrupt the training signal.

The guard is enforced at the termination call-site, not the diagnostic call-site. The diagnostic still measures and reports; the termination knob is what consults the guard before acting.

---

## 6. Implementation order

0. **Pre-check (Task 0)**: confirm whether per-ply adjudication-block reasons are already persisted in the existing `adjudication` sidecar block or per-game record (§3.2). If absent, add a minimal self-play telemetry hook (one append per adjudication attempt: the gate that blocked) **before** any analyzer-side work. This is the only allowed self-play-side change in this spec. Until Task 0 resolves, §3.2 numbers will be partially observable; document the partial-observability state explicitly in any pre-Task-0 report.
1. **Rollback the two failed experiments** in the next training-launch command. No code change. Update memory: both should be flagged as "tried and reverted" to prevent re-attempts under a different framing.
2. New module `scripts/GPU/alphazero/marathon_termination_diagnostics.py` with four pure-function diagnostics from §3 + unit tests. The opponent-block helper (§3.1) MUST be imported from / share a helper with `recovery_retargeting_diagnostics.classify_move` to prevent definitional drift.
3. Hook the diagnostics into the analyzer's per-game pass (reads `goal_completion_record` and `goal_completion_diagnostics`, no self-play change beyond Task 0). Sidecar fields and CSV-writer.
4. Report-section formatter with the decision-rule output line. Tests.
5. End-to-end smoke on existing 190-219 data. The 42 marathon games already on disk are the input — no retraining required for diagnostic-side work (Task 0 may require a fresh training block for the new telemetry to be populated; pre-Task-0 the report just shows partial-observability state).
6. Apply §5 decision rule against the resulting numbers; ship the single chosen termination change in the following training block — **with the §5.1 value-uncertain guard enforced at the termination call-site**.

Each step is analyzer-only (except Task 0 if needed). The training-side change in step 6 is gated on the diagnostic output, not pre-committed.

---

## 7. Tests

In `tests/test_marathon_termination_diagnostics.py`:

1. `test_no_progress_window_detects_pure_structural_run` — fixture with 15 consecutive redundant_reinforcement moves → 1 window detected.
2. `test_no_progress_window_breaks_on_distance_reduction` — 14 redundant + 1 reduces_total_goal_distance → 0 windows.
3. `test_no_progress_window_breaks_on_endpoint_completion` — 14 redundant + 1 completes_endpoint → 0 windows.
4. `test_no_progress_window_breaks_on_opponent_block` — 14 redundant + 1 blocks_opponent_closeout → 0 windows.
5. `test_no_progress_window_window_size_15` — exactly 14 redundant → 0 windows; 15 → 1.
6. `test_no_progress_window_opponent_block_uses_shared_helper` — fixture where the only "block" move passes Spec 4's defense rule but NOT a stricter local test → still counted as a block (confirms the shared helper).
7. `test_adjudication_coverage_blocked_by_min_top1` — synthetic per-ply data with high value, low top1 → `'min_top1_share'`.
8. `test_adjudication_coverage_blocked_by_value_below_threshold` — value never crosses 0.20 → `'value_below_threshold'`.
9. `test_adjudication_coverage_blocked_by_min_visits` — value + top1 met, visits below 200 → `'min_visits'`.
10. `test_adjudication_coverage_not_attempted` — eligible ply reached but no attempts logged → `'not_attempted'`.
11. `test_adjudication_coverage_would_have_passed` — all gates passed at some ply, no adjudication applied → `'would_have_passed'` (bug indicator).
12. `test_adjudication_coverage_missing_signal` — per-ply data missing search_score / top1 → `'missing_signal'`.
13. `test_adjudication_coverage_last_blocking_gate_precedence` — game where multiple gates blocked across plies: classify by last-blocking, with `min_top1_share > min_visits > value_below_threshold` precedence on ties.
14. `test_adjudication_coverage_skipped_for_non_state_cap_games` — game ending in win → `adjudication_coverage` is None/absent.
15. `test_resign_top1_gate_block_rate_partitions_by_n_moves` — three synthetic games (short / mid / long) → correct partitioned rates for BOTH `top1_block_rate_over_value_hits` and `top1_block_rate_over_eligible_hits`.
16. `test_resign_separates_no_value_signal_from_blocked_by_top1` — game with low `value_hits` (no losing signal) vs game with high `value_hits` and high `blocked_by_top1` → distinguishable in the report (the point of edit 5).
17. `test_value_uncertain_guard_blocks_termination_when_neutral` — predicate: last 10 own-plies for both sides with |score|<0.30 → guard returns "do not terminate".
18. `test_value_uncertain_guard_blocks_termination_when_oscillatory` — last 10 own-plies with >=3 sign-flips → guard returns "do not terminate".
19. `test_value_uncertain_guard_allows_termination_when_stable_losing` — last 10 own-plies stably below -0.30 for the loser → guard allows.
20. `test_aggregate_marathon_termination_per_iter_and_range` — synthetic 3-game / 2-iter fixture → correct per-iter rows + range-total row.
21. `test_format_marathon_termination_report_renders_section_with_decision_suggestion` — agg with min_top1_share dominant → suggestion line mentions "relax --adjudicate-min-top1-share".
22. `test_format_marathon_termination_report_partial_observability_when_taxonomy_incomplete` — pre-Task-0 state (gate-block reasons absent): report shows "partial observability — Task 0 not yet completed" caveat instead of definitive numbers.

In `tests/test_analyzer_marathon_termination.py`:

13. `test_analyzer_writes_marathon_termination_csv` — analyzer integration smoke.
14. `test_analyzer_report_includes_marathon_section` — section header + decision line appear.

---

## 8. Open questions

- Window size N=15 in §3.1 is a starting point; may need calibration on first run.
- Game-length partitioning in §3.3 uses 100 / 200 / >200 ply bins; revisit if the long-bin sample size is < 30 across a single range.

(The adjudication-attempt logging granularity question from the original draft has been promoted to Task 0 in §6 — it's a pre-check rather than an open question.)
