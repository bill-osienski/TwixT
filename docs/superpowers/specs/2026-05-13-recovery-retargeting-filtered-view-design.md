# Recovery / Re-targeting — Filtered Side-Split View — Design Spec

**Date:** 2026-05-13
**Status:** Drafted (awaiting user review)
**Predecessors:**
- [2026-05-12-recovery-retargeting-diagnostic-design.md](2026-05-12-recovery-retargeting-diagnostic-design.md) (Spec 4: shipped 2026-05-12; 170-179 production data complete)
- [2026-05-10-closeout-tail-correction-design.md](2026-05-10-closeout-tail-correction-design.md) (Spec 3 Fix 1 + Fix 2 shipped)

**Successor:** Spec 5 (Recovery-aware training) only if §7 decision rule, applied to the filtered view, signals a systemic, addressable failure shape.

---

## 1. Goal & Scope

### 1.1 Goal

Add an analyzer-only side-outcome split and an actionable-collapse filter on top of the existing Spec 4 telemetry, so the §11 Spec-5-gating decision rule can be applied to the population of side windows that plausibly represent **true failed re-targeting** rather than the broad raw trigger.

Re-runnable on existing per-game records (no self-play required, no trainer changes, no sidecar schema changes).

### 1.2 Why this exists

Spec 4 production data from iters 169-178 (1000 games, all three diagnostics on):

- `trigger_rate` 100.0% (1000/1000) — every game has at least one triggered side
- `triggered_winner_side_per_triggered_game` 34.2% — exceeds the spec §11 rule 6 bar of 20%, signalling thresholds-too-loose
- Pooled rollup: 23.3% constructive recovery, 0.5% defense, 54.2% structural connection, 22.0% local drift / unclear

The raw trigger is intentionally broad and window-based (steady_state caught 11422 of 12166 in-window plies). Five hand-reviewed seeds confirmed defense is essentially zero in the qualitative cases. The pooled rollup mixes loser-side, winner-side, and draw/state-cap windows, including many winner-side windows that are likely spurious early-trigger artifacts that stay open the rest of the game.

This view separates the structural shape (raw side-outcome split) from the calibrated decision input (filtered actionable-collapse view) without modifying or replacing the existing pooled summary. Both are needed: the raw split shows what the trigger is doing; the filtered view shows what likely represents true failed re-targeting.

### 1.3 In scope

- New aggregator functions in `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py` (additive — existing `aggregate_recovery_retargeting_records` unchanged)
- Pure-predicate actionable filter, used by both the filtered aggregator and the worst-cases CSV per-row annotation (single source of truth)
- New analyzer report sections (raw side-outcome split + filtered actionable-collapse view + filter summary)
- New CSV `recovery_retargeting_side_split_by_iter_<range>.csv` (long format, six rows per iter)
- Three new columns on existing `recovery_retargeting_worst_cases_<range>.csv`
- Unit and integration tests

### 1.4 Out of scope

- Trainer-side sidecar emission — `recovery_retargeting_summary` in `iter_NNNN_stats.json` keeps its existing shape; backward-compatible
- Self-play, MCTS, training targets, or move selection
- CLI flags for filter thresholds — defaults baked in for v1; add later if threshold sweeps become useful
- Resignation/adjudication tuning
- Spec 5 itself — this spec only provides the gating input

### 1.5 Relationship to Spec 4

Spec 4 ships the raw triggered population. This spec provides two structured analyzer views over the same per-side data: the raw side-outcome split (triggered population grouped by eventual game outcome) and the filtered actionable-collapse view (a calibrated subset for Spec 5 decisions).

---

## 2. Side-bucket assignment

Every triggered side in a per-game record is assigned to exactly one of three buckets:

```
eventual_loser     : record.loser is not None AND side == record.loser
eventual_winner    : record.winner is not None AND side == record.winner
state_cap_or_draw  : record.winner is None  (record.loser is None too — they co-occur)
```

Verified against 170-178 data: of 900 records with `recovery_retargeting_record`, 892 have both winner and loser set, 8 have neither. No partial-classification cases exist.

A single game can contribute up to two triggered sides (one per colour). They land in different buckets when the game has a winner; both land in `state_cap_or_draw` when it doesn't.

**Bucket-level invariant:** sum of triggered sides across the three buckets equals `triggered_loser_side + triggered_winner_side` from the existing pooled summary, plus state-cap/draw side counts (which the existing pooled summary doesn't surface but is present in the per-game records).

---

## 3. Actionable-collapse filter

A pure predicate applied per triggered side. All five clauses must hold for the side to pass.

| Clause | Default | Meaning |
|---|---|---|
| `in_window_own_moves >= min_in_window_own_moves` | 20 | Window long enough for "re-targeting" to be a meaningful concept |
| `triggered_own_moves >= min_triggered_own_moves` | 3 | More than a one-off trigger |
| `mean_search_score_triggered_plies <= max_mean_search_score_triggered_plies` | -0.85 | Sustained collapse, not a brief dip |
| `constructive_recovery_rate < max_constructive_recovery_rate` | 0.30 | Side did not actually re-target |
| `(structural_connection_rate + local_drift_rate) >= min_structural_plus_local_rate` | 0.60 | Failure mode is "kept playing local moves", not e.g. defensive moves that didn't pay off |

Note the absence of an isolated `local_drift_rate >= 40%` clause: the five hand-reviewed seeds show many failures are structural-only (extending existing components without reducing goal distance), not pure local drift.

Filter reasons (stable identifiers, used in `failed_reason_counts` and `filter_reasons_failed`):

```
in_window_below_min
triggered_below_min
mean_score_above_max
constructive_recovery_above_max
structural_plus_local_below_min
```

A side can fail multiple clauses — all failed reasons are reported, so `sum(failed_reason_counts.values())` may exceed `side_views_failed`.

---

## 4. Aggregator API

All in `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`.

### 4.1 New public functions

```python
def aggregate_recovery_retargeting_with_side_split(
    records: list[dict],
    *,
    games_total: int,
    config: dict | None = None,
) -> dict: ...

def apply_actionable_filter(
    side_view: dict,
    *,
    min_in_window_own_moves: int = 20,
    min_triggered_own_moves: int = 3,
    max_mean_search_score_triggered_plies: float = -0.85,
    max_constructive_recovery_rate: float = 0.30,
    min_structural_plus_local_rate: float = 0.60,
) -> tuple[bool, list[str]]: ...

def aggregate_recovery_retargeting_filtered(
    records: list[dict],
    *,
    games_total: int,
    config: dict | None = None,
    filter_config: dict | None = None,
) -> dict: ...
```

### 4.2 Private helpers (single source of truth for the math)

```python
def _iter_triggered_side_views(records: list[dict]) -> Iterator[dict]:
    """Yield one normalized side-view per triggered side across all records.
    Each view carries: side_bucket, in_window_own_moves, triggered_own_moves,
    severe_collapse_moves, very_diffuse_moves, classified_in_window_moves,
    selected_class_counts, mean_search_score_triggered_plies,
    min_search_score_triggered_plies, max_search_score_triggered_plies,
    mean_root_top1_share_triggered_plies, trigger_reason_counts,
    derived constructive_recovery_rate / defensive_rate /
    structural_connection_rate / local_drift_rate."""

def _side_bucket_for_record(record: dict, side: str) -> str:
    """Return 'eventual_loser' | 'eventual_winner' | 'state_cap_or_draw'."""

def _compute_side_rollup(side_views: Iterable[dict]) -> dict:
    """Shared rollup math used by all three public aggregators."""

def _empty_side_rollup() -> dict:
    """Stable zero schema for empty buckets."""
```

### 4.3 Side-rollup shape (returned per bucket)

```
{
  "sides": int,                                       # number of side views in this bucket
  "in_window_own_moves_total": int,
  "triggered_own_moves_total": int,
  "severe_collapse_moves_total": int,
  "very_diffuse_moves_total": int,
  "classified_in_window_moves_total": int,
  "selected_class_counts_total": {<class>: int, ...},
  "selected_class_rates_total":  {<class>: float, ...},
  "trigger_reason_counts_total": {"delta_precursor": int, "steady_state": int, "both": int},
  "constructive_recovery_rate":  float,
  "defensive_rate":              float,
  "structural_connection_rate":  float,
  "local_drift_rate":            float,
  "mean_search_score_triggered_plies": float | None,
  "min_search_score_triggered_plies":  float | None,
  "max_search_score_triggered_plies":  float | None,
  "mean_root_top1_share_triggered_plies": float | None,
}
```

### 4.4 Bucket-level aggregation rules for score statistics

Means are pooled, weighted by per-side `triggered_own_moves`:

```
mean_search_score_triggered_plies =
  sum(s.mean_search_score_triggered_plies * s.triggered_own_moves)
  / sum(s.triggered_own_moves)

mean_root_top1_share_triggered_plies =
  sum(s.mean_root_top1_share_triggered_plies * s.triggered_own_moves)
  / sum(s.triggered_own_moves)
```

Min/max are taken across per-side mins/maxes:

```
min_search_score_triggered_plies = min(s.min_search_score_triggered_plies for s in views)
max_search_score_triggered_plies = max(s.max_search_score_triggered_plies for s in views)
```

If `sum(triggered_own_moves)` is zero, all four score fields are `None`.

Rationale: the rollup describes the population of triggered plies (matches the existing pooled-count semantics for class rates), not the average side. Long collapse windows should contribute proportionally more weight than short one-off triggers. Unweighted per-side means are intentionally not emitted in v1.

### 4.5 `aggregate_recovery_retargeting_with_side_split` return

```
{
  "version": 1,
  "view": "raw_side_split",
  "config": <as passed in>,
  "games_total": int,
  "games_triggered": int,
  "eventual_loser":     <side rollup>,
  "eventual_winner":    <side rollup>,
  "state_cap_or_draw":  <side rollup>,
  "schema_integrity": {
    "skipped_unknown_version_count": int,
    "skipped_config_mismatch_count": int,
    "classifier_error_count_total":  int,
  },
}
```

### 4.6 `aggregate_recovery_retargeting_filtered` return

Same shape as §4.5 with `view = "filtered_actionable_collapse"`, plus:

```
"filter_summary": {
  "filter_config": {
    "min_in_window_own_moves": 20,
    "min_triggered_own_moves": 3,
    "max_mean_search_score_triggered_plies": -0.85,
    "max_constructive_recovery_rate": 0.30,
    "min_structural_plus_local_rate": 0.60
  },
  "side_views_total":   int,
  "side_views_passed":  int,
  "side_views_failed":  int,
  "failed_reason_counts": {
    "in_window_below_min":             int,
    "triggered_below_min":             int,
    "mean_score_above_max":            int,
    "constructive_recovery_above_max": int,
    "structural_plus_local_below_min": int
  }
}
```

`failed_reason_counts` may sum greater than `side_views_failed` because a side can fail multiple clauses.

### 4.7 Empty input handling

Both new aggregators accept an empty `records` list and return a fully-shaped result with all three side buckets populated by `_empty_side_rollup()` (counts zero, score fields `None`). `aggregate_recovery_retargeting_filtered` additionally returns a `filter_summary` with `side_views_total = side_views_passed = side_views_failed = 0` and an all-zero `failed_reason_counts`. This applies both to per-iter calls on iterations with zero triggered games and to the range-level call when no iters in scope produced records.

### 4.8 Backward compatibility

`aggregate_recovery_retargeting_records` is unchanged in signature, behavior, and return shape. The trainer-side sidecar (`recovery_retargeting_summary` in `iter_NNNN_stats.json`) continues to use it. No existing test of the aggregator needs modification.

---

## 5. Analyzer changes

In `scripts/twixt_replay_analyzer.py`.

### 5.1 Report rendering

`format_recovery_retargeting_report()` is extended (not replaced) to take the existing pooled summary plus the new side-split and filtered summaries. Output structure:

```
Recovery / Re-targeting Diagnostics
===================================
<existing pooled section, unchanged>

Raw side-outcome split
----------------------
eventual_loser:
  triggered sides:        N
  constructive recovery:  X%
  defense:                X%
  structural connection:  X%
  local drift / unclear:  X%
  mean trigger score:     -0.XX  (in [min, max])
  mean trigger top1:       0.XX

eventual_winner:
  ...

state_cap_or_draw:
  ...

Filtered actionable-collapse view
---------------------------------
Filter: in_window>=20, triggered>=3, mean_score<=-0.85,
        constructive<30%, structural+local>=60%

eventual_loser:
  eligible sides:         N
  constructive recovery:  X%
  defense:                X%
  structural connection:  X%
  local drift / unclear:  X%
  mean trigger score:     -0.XX
  mean trigger top1:       0.XX

eventual_winner:
  eligible sides:         N   [threshold sanity only — not used as Spec 5 intervention target]
  ...

state_cap_or_draw:
  ...

Filter summary:
  side views total:    N
  side views passed:   N
  side views failed:   N
  failed reasons:
    in_window_below_min:             N
    triggered_below_min:             N
    mean_score_above_max:            N
    constructive_recovery_above_max: N
    structural_plus_local_below_min: N
```

The pooled section is retained as-is for continuity with prior reports (e.g. 160-169) and to preserve the `Worst cases: ...` reference line at its existing location.

### 5.2 New CSV: `recovery_retargeting_side_split_by_iter_<range>.csv`

Long format. Up to 6 rows per iteration (3 side buckets × 2 views). Empty buckets still emit a row: integer columns are `0`; score columns (`mean_search_score_triggered_plies`, `mean_root_top1_share_triggered_plies`) are written as the empty string. Iterations with zero triggered games still emit all 6 rows (all-zero / empty-score).

Columns:

```
iteration,
view,                      # "raw" | "filtered"
side_bucket,               # "eventual_loser" | "eventual_winner" | "state_cap_or_draw"
sides,
in_window_own_moves_total,
triggered_own_moves_total,
mean_search_score_triggered_plies,
mean_root_top1_share_triggered_plies,
constructive_recovery_rate,
defensive_rate,
structural_connection_rate,
local_drift_rate,
redundant_local_reinforcement_rate,
off_plan_or_unclear_rate
```

Single `sides` column for both views (semantics depend on `view`): in `raw` rows it's the count of triggered sides in the bucket; in `filtered` rows it's the count of sides passing the filter.

### 5.3 Worst-cases CSV extension

`write_recovery_retargeting_worst_cases_csv()` gets three additional columns appended, applied per-row using `apply_actionable_filter` (single source of truth):

```
side_bucket               # "eventual_loser" | "eventual_winner" | "state_cap_or_draw"
passes_actionable_filter  # "true" | "false"
filter_reasons_failed     # semicolon-separated reason ids; empty when passes
```

Existing sort key (`local_drift_moves DESC, in_window_own_moves DESC, min_search_score_triggered_plies ASC`) is preserved. Existing `top_k` truncation is preserved.

### 5.4 Wiring

Inside the `recovery_retargeting` block at the analyzer's main path (currently around `twixt_replay_analyzer.py:4975-5040`):

```python
rr_summary  = aggregate_recovery_retargeting_records(rr_records, games_total=..., config=...)
rr_split    = aggregate_recovery_retargeting_with_side_split(rr_records, games_total=..., config=...)
rr_filtered = aggregate_recovery_retargeting_filtered(rr_records, games_total=..., config=...)

lines.extend(format_recovery_retargeting_report(rr_summary, rr_split, rr_filtered))

write_recovery_retargeting_by_iter_csv(...)            # unchanged
write_recovery_retargeting_side_split_csv(...)         # new
write_recovery_retargeting_worst_cases_csv(...)        # extended (3 new columns)
```

Per-iter CSVs need per-iter split/filtered summaries — collected during the existing per-iter loop alongside `per_iter_rr`.

The `summary["recovery_retargeting"]` JSON gets two new sibling keys: `"recovery_retargeting_side_split"` and `"recovery_retargeting_filtered"`.

---

## 6. Tests

Extend `tests/test_recovery_retargeting_diagnostics.py`:

1. `test_side_bucket_eventual_loser` — record with winner=red, loser=black: black-side view → `eventual_loser`.
2. `test_side_bucket_eventual_winner` — same record: red-side view → `eventual_winner`.
3. `test_side_bucket_state_cap_or_draw` — record with winner=None, loser=None: both sides → `state_cap_or_draw`.
4. `test_side_split_rollup_recombines_to_pooled_counts` — for a synthetic record set, the sum of `in_window_own_moves_total` and `classified_in_window_moves_total` across the three side-split buckets equals the corresponding totals from `aggregate_recovery_retargeting_records`. Pins the math against drift between aggregators.
5. `test_apply_actionable_filter_passes_clean_failure_case` — side view satisfying all five clauses returns `(True, [])`.
6. `test_apply_actionable_filter_reports_all_failed_reasons` — side view failing three clauses returns `(False, [<all three reason ids>])`.
7. `test_filtered_aggregator_counts_only_passing_sides` — aggregator receives a record set with mixed pass/fail sides; `sides` per filtered bucket equals the predicate's true count.
8. `test_filter_summary_counts_multiple_failed_reasons` — for a record with sides failing multiple clauses, `failed_reason_counts` sums above `side_views_failed`.
9. `test_bucket_score_aggregation_uses_pooled_weighted_mean` — synthetic two-side bucket (one with 120 triggered plies @ −0.95, one with 3 triggered plies @ −0.40); bucket `mean_search_score_triggered_plies` ≈ `(−0.95×120 + −0.40×3) / 123` ≈ `−0.937`. Confirms the weighting scheme of §4.4.

Extend `tests/test_analyzer_recovery_retargeting.py`:

10. `test_analyzer_writes_side_split_csv_with_six_rows_per_iter` — for a two-iter test fixture with both winner/loser/state-cap representation, the new CSV contains 12 rows in long format.
11. `test_worst_cases_csv_uses_same_filter_predicate` — for a fixture, every row's `passes_actionable_filter` matches the result of calling `apply_actionable_filter` on the same per-side data. Pins the single-source-of-truth invariant.

---

## 7. Decision rule for Spec 5 gating

Apply to the **filtered actionable-collapse view**, not the pooled or raw-split views.

**Spec 5 (recovery-aware training) is justified when all of:**
- `(filtered.eventual_loser.sides + filtered.state_cap_or_draw.sides) / filter_summary.side_views_total >= 30%` (placeholder; denominator is the count of raw triggered side views before filtering)
- For those two buckets combined: `(structural_connection_rate + local_drift_rate)` remains high (placeholder ≥ 60%, matching the filter's lower bound — i.e. not just admitted, but representative)
- For those two buckets combined: `(constructive_recovery_rate + defensive_rate)` remains low (placeholder ≤ 35%)

**Spec 5 not justified when any of:**
- Filtered `eventual_loser.sides + state_cap_or_draw.sides` is small (rule 1 above is violated) — most triggers are spurious / winner-side artifacts
- Filtered loser+draw `constructive_recovery_rate` ≥ 40% — sides do re-target after collapse; not a systemic failure
- Filtered `eventual_winner.sides` remains > 20% of total filtered sides — filter is still too permissive; tighten before deciding

**Always do before applying the rule:**
- Verify `schema_integrity.classifier_error_count_total` is 0 (or < 1% of `classified_in_window_moves_total`)
- Hand-review the top 3 `eventual_loser` rows from the worst-cases CSV with `passes_actionable_filter=true` to confirm the bucket assignments match the qualitative read

Concrete numbers will be calibrated on the first run against the 170-179 data; the placeholders above are starting points, not final thresholds.

---

## 8. Implementation order

1. Diagnostics module: private helpers (`_side_bucket_for_record`, `_iter_triggered_side_views`, `_compute_side_rollup`, `_empty_side_rollup`).
2. Diagnostics module: `aggregate_recovery_retargeting_with_side_split`. Tests 1–4.
3. Diagnostics module: `apply_actionable_filter`. Tests 5–6.
4. Diagnostics module: `aggregate_recovery_retargeting_filtered`. Tests 7–9.
5. Analyzer: `format_recovery_retargeting_report` extension. Manual smoke against 170-179 data.
6. Analyzer: `write_recovery_retargeting_side_split_csv`. Test 10.
7. Analyzer: extend `write_recovery_retargeting_worst_cases_csv` with three new columns. Test 11.
8. Analyzer: wiring in main path; per-iter collection updates.
9. Re-run analyzer end-to-end on 170-179 data; eyeball report and CSVs.
10. Apply §7 decision rule against the resulting filtered view.

Each step ships independently and is verifiable in isolation. The per-iter sidecar (and therefore the trainer / IPC plumbing) is never touched.

---

## 9. Open questions

None blocking implementation. The §7 decision-rule thresholds are intentionally placeholders to be calibrated on first-run data.

---

## 10. v1.1 calibration update (2026-05-14)

After shipping v1.0 and applying §7 to the 170-179 data, the read-out was:

| Rule | Outcome | Value |
|---|---|---|
| Rule 1: share ≥ 30% | FAIL | 7.4% |
| Rule 6: winner_share ≤ 20% | FAIL | 23.7% |

A 31-case hand-review of the filter-passing winner-side cases (median in_window=25, just above the 20-cutoff; median first_trigger_ply at 25% into the game; top class `connects_to_existing_component` at 61% pooled) confirmed those were Category B (trigger artifacts on transient bad MCTS reads), not genuine recoveries.

A filter sweep on the same data tested four levers (raise `min_in_window_own_moves`, raise `min_triggered_own_moves`, tighten `max_mean_search_score_triggered_plies`, and combinations). Findings:

- Rule 1 fails in **every** tested config — the actionable-collapse signal is genuinely rare in 170-179 (not hidden behind a too-loose filter).
- Tightening `min_in_window_own_moves` is bad for the loser-side signal (in=30 drops loser sides 95→13; real late-game collapses have short windows).
- Tightening `max_mean_search_score_triggered_plies` from -0.85 to -0.90 (Option C) is the cleanest single-knob fix for Rule 6: winner_share 23.7% → 17.5%, loser+state_cap 100 → 47 sides (still hand-reviewable), structural+local stays at 85.7%, constructive+defense stays at 14.3%.

**v1.1 default change (shipped 2026-05-14):**

```python
_DEFAULT_FILTER_CONFIG["max_mean_search_score_triggered_plies"] = -0.90  # was -0.85 in v1
apply_actionable_filter(..., max_mean_search_score_triggered_plies: float = -0.90)  # was -0.85
```

No other defaults changed. The runtime trigger (`collapse_value_threshold = -0.75` in self-play) is unchanged — this is an analyzer filter calibration only, not a self-play diagnostic trigger change.

**§7 read-out under v1.1:**
- Rule 1 still FAILS (share 3.5%) — confirms Spec 5 is not justified by current data.
- Rule 6 PASSES (winner_share 17.5%) — filter is no longer too permissive.
- The "Spec 5 not justified" verdict now has a clean basis: **the failure shape exists and is real, but it is rare**, rather than "maybe the filter is too permissive".

**Backward compatibility for re-runs against v1 reports:** pass `filter_config={"max_mean_search_score_triggered_plies": -0.85}` to `aggregate_recovery_retargeting_filtered` to restore v1 behavior. Tested by `test_filter_config_override_can_restore_v1_threshold`.
