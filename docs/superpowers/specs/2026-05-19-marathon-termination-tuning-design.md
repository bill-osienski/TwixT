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
no_opponent_block            : no own-move in the window was classified as
                               blocks_opponent_closeout (Spec 4 vocabulary)
moves_are_local_structural   : all moves in the window had primary_class in
                               {redundant_reinforcement, off_chain,
                                connects_to_existing_component,
                                improves_own_largest_component}
```

Per-game metric: count of distinct no-progress windows per side (overlapping windows count once, anchored to the last ply of the longest such window).

### 3.2 Adjudication coverage diagnostic

For each game that ended at n_moves == 280 with reason == state_cap, compute:

```
ply_of_first_eligible_adjudication      : first ply >= adjudicate_min_ply
                                          where adjudicate eligibility held
ply_of_first_threshold_crossing         : first ply at which the would-be
                                          winner's value crossed adjudicate_threshold
gate_blocked_by                         : which gate blocked: 'min_visits',
                                          'min_top1_share', 'value_below_threshold',
                                          or 'never_blocked' (eligible but no
                                          attempt — likely a bug)
plies_after_first_threshold_crossing    : 280 - ply_of_first_threshold_crossing
```

Range-level rollup: distribution of `gate_blocked_by` across 280-ply state_cap games. Tells us which adjudication gate to relax (if any).

### 3.3 Resign-gate-on-hopeless diagnostic

For each game's losing side (when winner is known), in the last 40 plies:

```
resign_value_hits_blocked_by_top1       : count of plies where the loser's
                                          search_score was below resign_threshold
                                          AND root_top1_share was below
                                          resign_min_top1_share (so resign was gated)
final_eval_was_below_threshold          : at the last ply for the loser,
                                          search_score < resign_threshold
losing_side_total_value_hits            : total value-hits across the game
losing_side_top1_gate_block_rate        : value_hits_blocked_by_top1 /
                                          losing_side_total_value_hits
```

Range-level rollup: mean `top1_gate_block_rate` for losing sides, partitioned by game length:

```
short games (n_moves <= 100)    : sanity baseline
mid games   (100 < n_moves <= 200)
long games  (n_moves > 200)     : these are the ones we care about
```

If long-game block-rate is significantly higher than short-game (e.g. 2x), the top1 gate is too conservative late in hopeless games.

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

5. **If `mean_no_progress_windows_per_game` is high (>2 per game)** AND none of (1)-(4) trigger: introduce an **early state-cap** rule that ends the game after K consecutive no-progress windows for either side. K=2 is the starting point.

6. **If none of (1)-(5) clearly dominates**: the marathon shape is heterogeneous; defer to a hand-review of 5-10 representative cases before another knob change.

Each branch produces at most one CLI/config change. No simultaneous tuning of multiple termination knobs.

---

## 6. Implementation order

1. **Rollback the two failed experiments** in the next training-launch command. No code change. Update memory.
2. New module `scripts/GPU/alphazero/marathon_termination_diagnostics.py` with four pure-function diagnostics from §3 + unit tests.
3. Hook the diagnostics into the analyzer's per-game pass (reads `goal_completion_record` and `goal_completion_diagnostics`, no self-play change). Sidecar fields and CSV-writer.
4. Report-section formatter with the decision-rule output line. Tests.
5. End-to-end smoke on existing 190-219 data. The 42 marathon games already on disk are the input — no retraining required for diagnostic-side work.
6. Apply §5 decision rule against the resulting numbers; ship the single chosen termination change in the following training block.

Each step is analyzer-only and ships independently. The training-side change in step 6 is gated on the diagnostic output, not pre-committed.

---

## 7. Tests

In `tests/test_marathon_termination_diagnostics.py`:

1. `test_no_progress_window_detects_pure_structural_run` — fixture with 15 consecutive redundant_reinforcement moves → 1 window detected.
2. `test_no_progress_window_breaks_on_distance_reduction` — 14 redundant + 1 reduces_total_goal_distance → 0 windows.
3. `test_no_progress_window_breaks_on_endpoint_completion` — 14 redundant + 1 completes_endpoint → 0 windows.
4. `test_no_progress_window_breaks_on_opponent_block` — 14 redundant + 1 blocks_opponent_closeout → 0 windows.
5. `test_no_progress_window_window_size_15` — exactly 14 redundant → 0 windows; 15 → 1.
6. `test_adjudication_coverage_blocked_by_min_top1` — synthetic per-ply data with high value, low top1 → `gate_blocked_by = 'min_top1_share'`.
7. `test_adjudication_coverage_blocked_by_value_below_threshold` — value never crosses 0.20 → `'value_below_threshold'`.
8. `test_adjudication_coverage_never_blocked` — eligible but no attempt logged → `'never_blocked'`.
9. `test_adjudication_coverage_skipped_for_non_state_cap_games` — game ending in win → `adjudication_coverage` is None/absent.
10. `test_resign_top1_gate_block_rate_partitions_by_n_moves` — three synthetic games (short / mid / long) → correct partitioned rates.
11. `test_aggregate_marathon_termination_per_iter_and_range` — synthetic 3-game / 2-iter fixture → correct per-iter rows + range-total row.
12. `test_format_marathon_termination_report_renders_section_with_decision_suggestion` — agg with min_top1_share dominant → suggestion line mentions "relax --adjudicate-min-top1-share".

In `tests/test_analyzer_marathon_termination.py`:

13. `test_analyzer_writes_marathon_termination_csv` — analyzer integration smoke.
14. `test_analyzer_report_includes_marathon_section` — section header + decision line appear.

---

## 8. Open questions

- Window size N=15 in §3.1 is a starting point; may need calibration on first run.
- Game-length partitioning in §3.3 uses 100 / 200 / >200 ply bins; revisit if the long-bin sample size is < 30 across a single range.
- Adjudication-attempt logging granularity: the diagnostic in §3.2 needs to know *which* gate blocked. The existing `adjudication` sidecar block tracks aggregate attempts/checks but may not surface per-ply gate-block reasons. If that data isn't present, §3.2 will be partially observable and may need a small self-play hook addition (one-line append to existing adjudication telemetry). To be confirmed in step 3 implementation.
