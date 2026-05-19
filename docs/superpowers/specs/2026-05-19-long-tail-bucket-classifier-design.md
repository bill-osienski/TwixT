# Long-Tail Bucket Classifier — Design Spec

**Date:** 2026-05-19
**Status:** Drafted (awaiting user review)
**Predecessors:**
- [2026-05-13-recovery-retargeting-filtered-view-design.md](2026-05-13-recovery-retargeting-filtered-view-design.md) (Spec 4 v1.1 calibrated filter)
- [2026-05-10-closeout-tail-correction-design.md](2026-05-10-closeout-tail-correction-design.md) (Spec 3 Fix 1 + Fix 2)

**Successor:** Targeted closeout / training interventions, choice gated on bucket counts produced by this classifier.

---

## 1. Goal & Scope

### 1.1 Goal

Replace the current manual triage of long-tail games (delay ≥ 20, state-cap, dominant unavailable, board cap) with an analyzer-side classifier that assigns each qualifying game to exactly one of five mutually exclusive failure buckets. The five-bucket count table is the next decision input: which knob to turn (relax Fix 2 value gate, raise conversion-policy-loss weight, address state-cap rules, etc.) should be driven by the relative bucket sizes, not by re-doing the manual review each block.

### 1.2 Motivation

The 190-219 manual triage (3 blocks × 25 worst-cases each = 75 rows, 24 unique games after dedupe) split cleanly into:
- 14 state-cap / 280-ply marathons (mutual structural drift)
- 6 td=2 games where the correct closeout move was in visit top-5 but Fix 2's `min_value=0.95` gate blocked the override (q at the failure ply was 0.7-0.9, not 0.95+)
- 4 td=2 games where the correct move was visit-rank 88+ (policy training tail)

The first cluster needs state-cap / adjudication tuning, the second is a Fix 2 calibration question, the third is a training-signal question — three different interventions. Without an automated classifier we have to redo this analysis for each new block of 10 iters, which is both time-consuming and error-prone.

### 1.3 In scope

- New function `classify_long_tail_bucket(record, diagnostics) -> str` in a new module `scripts/GPU/alphazero/long_tail_bucket_classifier.py`
- New column `long_tail_bucket` on existing `goal_completion_worst_cases_<range>.csv`
- New CSV `goal_completion_long_tail_buckets_<range>.csv` with one row per bucket + per-iter trend rows
- New report section "Long-tail bucket counts" with per-bucket counts and percentages
- Unit tests for the classifier (one per bucket + priority-order tests)

### 1.4 Out of scope

- Any change to MCTS, self-play, training targets, or Fix 1 / Fix 2 thresholds (those are the *decisions* this classifier is meant to inform)
- Changes to the `goal_completion_diagnostics` or `recovery_retargeting` modules
- Per-position classification (this is per-game)
- Re-classification of games outside the long-tail filter (filter scope is unchanged from existing worst-cases CSV)

---

## 2. Bucket definitions

Each long-tail game is assigned to exactly one bucket, by **priority order** (first match wins). This avoids ambiguity for games that satisfy multiple criteria.

### 2.1 Priority 1 — `marathon_or_state_cap`

```
record.reason == "state_cap"  OR  record.n_moves == 280
```

Covers both pure state-cap games (no winner, board cap) and 280-ply marathons with eventual winner (board cap + late win). Highest priority because these are structurally different from per-position closeout failures — neither side ever resolves the position.

### 2.2 Priority 2 — `dominant_unavailable_contested`

```
record.winner_moves_with_dominant_unavailable >= 10
```

The winner's dominant chain became unavailable for ≥ 10 moves during the watch window. Threshold 10 chosen from the 190-219 data: distribution had p50=0, p90=19, max=112. A floor of 10 captures the contested-chain cases without picking up noise (a few transient unavailable plies during normal play).

### 2.3 Priority 3 — `td3_drift`

```
record.first_total_goal_distance >= 3
```

The closeout-eligible position was detected at total_goal_distance ≥ 3. These positions legitimately require multi-move plans; they're not Fix 2's design target. Separating them prevents misattributing td=3 delays to Fix 2 calibration issues.

### 2.4 Priority 4 — `td2_alt_in_top5`

```
record.first_total_goal_distance == 2
AND >= 50% of redundant-pick plies in diagnostics had an
endpoint-completion or distance-reducing move in visit top-5
```

This is Fix 2's design territory where the right move was visible but Fix 2 didn't act. The 50% threshold is a tunable parameter — anchored at 50% because if ≥ half of the failure plies had an alternative in top-5, the bottleneck is the Fix 2 gate / preconditions, not policy quality.

A "redundant-pick ply" is a position where `selected_move_classification.primary_class == "redundant_reinforcement"`.

An "alternative in top-5" means `endpoint_completion_ranking.any_in_visit_top5 == True` OR `distance_reducing_ranking.any_in_visit_top5 == True`.

### 2.5 Priority 5 — `td2_reducer_buried`

```
record.first_total_goal_distance == 2  AND  none of priorities 1-4 apply
```

The td=2 game's correct move was outside visit top-5 at most failure plies — policy training tail. No tie-break can help; needs stronger training signal (conversion-aware policy loss, etc.).

### 2.6 Fallback — `unclassified`

A long-tail game that doesn't fit any of the five buckets above. Expected to be ~0% on real data given the priority structure (priority 1 covers all state-cap/280-ply, priority 3+4+5 cover all td≥2, priority 2 covers high-dom-unavail). Emitted only as a defensive check — if `unclassified` count grows above ~5%, the classifier's logic needs revision before drawing conclusions from the table.

---

## 3. Filter scope (which games get classified)

Same as the existing `write_goal_completion_worst_cases_csv` input set: top-K (default 25) goal-completion worst-cases per range, sorted by the existing CSV's sort key. Plus, for the new bucket-count CSV, ALL games matching:

```
record.conversion_delay_plies >= 20
OR record.reason == "state_cap"
OR record.winner_moves_with_dominant_unavailable >= 20
OR record.n_moves == 280
```

The `>= 20` for delay and dom_unavailable matches the manual triage filter and the top-K CSV's natural slope. (Lower thresholds add noise without changing the bucket distribution materially.)

---

## 4. Output

### 4.1 Extension of existing `goal_completion_worst_cases_<range>.csv`

Add one new column at the end (after `search_score_coverage_in_watch_window`):

```
long_tail_bucket
```

Value is one of the five bucket names from §2, or `unclassified`. Games that don't qualify for the long-tail filter (delay < 20, etc.) and are in the worst-cases CSV for other reasons get `not_long_tail`.

### 4.2 New `goal_completion_long_tail_buckets_<range>.csv`

Long format, one row per (iteration × bucket). Aggregated over ALL games matching the long-tail filter (not just top-K worst-cases). Empty buckets emit a row with count=0.

```
iteration,
bucket,                  # one of the five names + 'unclassified'
games,                   # number of long-tail games in this bucket this iter
total_long_tail_games,   # denominator for share computation (iter-level)
share                    # games / total_long_tail_games (0.0 if denom is 0)
```

Plus one summary row per bucket aggregated across the range, written with `iteration = -1` as the sentinel (chosen because real iterations are always positive, making the sentinel both type-stable and trivially filterable in pandas / awk).

Same columns; `total_long_tail_games` = sum across iterations.

### 4.3 New report section "Long-tail bucket counts"

Inserted in the report after the existing "Goal-completion worst cases" / closeout-related sections. Format:

```
Long-tail bucket counts (<range>)
=================================
Long-tail filter: delay >= 20 OR state_cap OR dom_unavail >= 20 OR n_moves == 280
Total long-tail games in range: N

Bucket                                games   share   next-action hint
marathon_or_state_cap                   N    XX.X%   state-cap / adjudication / resign tuning
dominant_unavailable_contested          N    XX.X%   recovery dynamics (rare; usually overlaps Spec 4)
td3_drift                               N    XX.X%   broader closeout / planning depth
td2_alt_in_top5                         N    XX.X%   Fix 2 calibration (value gate / preconditions)
td2_reducer_buried                      N    XX.X%   policy training tail (conversion-aware loss)
unclassified                            N    XX.X%   classifier review needed if > 5%

Per-iter trend (range):
iter    marathon  contested  td3   td2_top5  td2_buried  uncl  total
NNN          N          N    N         N           N       N      N
...
```

The "next-action hint" column is hard-coded text, not derived — it's a documentation aid embedded in the report so the reader can act on the table without remembering the spec.

---

## 5. Classifier API

In `scripts/GPU/alphazero/long_tail_bucket_classifier.py`:

```python
LONG_TAIL_BUCKETS = (
    "marathon_or_state_cap",
    "dominant_unavailable_contested",
    "td3_drift",
    "td2_alt_in_top5",
    "td2_reducer_buried",
    "unclassified",
)

NOT_LONG_TAIL = "not_long_tail"

def matches_long_tail_filter(record: dict) -> bool:
    """True if the game qualifies for long-tail classification.
    delay >= 20 OR state_cap OR dom_unavail >= 20 OR n_moves == 280."""

def classify_long_tail_bucket(record: dict, diagnostics: list) -> str:
    """Return the long-tail bucket for a single game.

    record: per-game goal_completion_record dict
    diagnostics: per-game goal_completion_diagnostics list (per-ply entries)

    Returns one of LONG_TAIL_BUCKETS. Returns NOT_LONG_TAIL if the game
    does not match the long-tail filter.

    Priority order:
      1. marathon_or_state_cap
      2. dominant_unavailable_contested
      3. td3_drift
      4. td2_alt_in_top5
      5. td2_reducer_buried
      6. unclassified (defensive fallback)
    """

def aggregate_long_tail_buckets(records_with_diagnostics: list) -> dict:
    """Aggregate per-game classifications into a per-iter and per-range table.

    records_with_diagnostics: list of (record, diagnostics) tuples.

    Returns:
      {
        "per_iter": {iter: {bucket: count, ...}, ...},
        "range_total": {bucket: count},
        "total_long_tail_games_per_iter": {iter: count},
        "total_long_tail_games_range": int,
      }
    """
```

---

## 6. Tests

In `tests/test_long_tail_bucket_classifier.py`:

1. `test_matches_long_tail_filter_delay_threshold` — delay=19 → False, delay=20 → True.
2. `test_matches_long_tail_filter_state_cap` — `reason="state_cap"` → True.
3. `test_matches_long_tail_filter_dom_unavail_threshold` — 19 → False, 20 → True.
4. `test_matches_long_tail_filter_n_moves_280` — n_moves=279 → False, 280 → True.
5. `test_bucket_marathon_or_state_cap_state_cap_reason` — state_cap game → bucket 1.
6. `test_bucket_marathon_or_state_cap_280_ply_with_winner` — n_moves=280, winner set → bucket 1.
7. `test_bucket_dominant_unavailable_contested_threshold` — dom_un=10 → bucket 2, dom_un=9 (but delay=30) → not bucket 2.
8. `test_bucket_td3_drift` — first_total_goal_distance=3, normal game → bucket 3.
9. `test_bucket_td2_alt_in_top5_majority_have_top5_alt` — td=2 with 3/3 redundant picks having top-5 alt → bucket 4.
10. `test_bucket_td2_reducer_buried_majority_have_no_top5_alt` — td=2 with 0/3 redundant picks having top-5 alt → bucket 5.
11. `test_bucket_td2_exactly_50_percent_alt_goes_to_top5` — boundary: 2/4 with top-5 alt → bucket 4 (uses `>=`).
12. `test_priority_marathon_over_contested` — game with state_cap AND dom_un=50 → bucket 1 (marathon wins).
13. `test_priority_contested_over_td_buckets` — game with td=2, dom_un=20 → bucket 2 (contested wins).
14. `test_unclassified_defensive_fallback` — synthetic record that doesn't match any bucket → "unclassified".
15. `test_aggregate_long_tail_buckets_per_iter_and_range_totals` — 3 games across 2 iters → correct per-iter and range totals.
16. `test_aggregate_long_tail_buckets_share_computation` — bucket count / total_long_tail = share, rounded to 3 places.

In `tests/test_analyzer_long_tail_buckets.py` (new):

17. `test_analyzer_writes_long_tail_buckets_csv` — synthetic records produce the expected per-iter + range_total rows.
18. `test_worst_cases_csv_has_long_tail_bucket_column` — existing CSV gains the new column; rows have correct bucket labels.

---

## 7. Implementation order

1. Module skeleton + `matches_long_tail_filter` + tests 1-4.
2. `classify_long_tail_bucket` for buckets 1, 2, 3 (no diagnostics needed) + tests 5-8.
3. `classify_long_tail_bucket` for buckets 4, 5 (uses diagnostics) + tests 9-11.
4. Priority-order tests 12-14.
5. `aggregate_long_tail_buckets` + tests 15-16.
6. Wire into analyzer:
   - Extend `write_goal_completion_worst_cases_csv` with `long_tail_bucket` column + test 18.
   - New `write_goal_completion_long_tail_buckets_csv` + test 17.
   - New `format_long_tail_bucket_report` section.
   - Main-path wiring.
7. End-to-end smoke on 190-219 to verify the table matches the manual triage from 2026-05-17.

Each step ships independently. No self-play / no trainer-side change.

---

## 8. Decision rule (after first run)

The bucket-count table is the input; the action is one of:

- `marathon_or_state_cap` dominant (>= 50% of long-tail) → look at state-cap/adjudication thresholds, not closeout.
- `td2_alt_in_top5` dominant (>= 30%) → Fix 2 calibration is the highest-leverage knob (relax `min_value` gate, broaden top-K).
- `td2_reducer_buried` dominant (>= 30%) → policy training signal needs strengthening (raise `conversion_policy_loss_weight`, consider new aux target).
- `td3_drift` dominant (>= 30%) → broader planning depth or training-tail issue distinct from td=2.
- `dominant_unavailable_contested` non-trivial (>= 10%) → cross-reference with Spec 4 recovery diagnostic; may be a recovery overlap pattern.
- `unclassified` > 5% → classifier needs revision before drawing conclusions.

Thresholds are starting points; recalibrate after first 1-2 runs.

---

## 9. Open questions

None blocking implementation. The 50% threshold for "majority redundant picks have top-5 alt" (priority 4) and the 10-move threshold for `dominant_unavailable_contested` are calibrated against 190-219 manual data and should be re-checked after first analyzer run on a fresh range.
