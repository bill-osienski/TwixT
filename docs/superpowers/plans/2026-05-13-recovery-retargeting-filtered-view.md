# Recovery / Re-targeting Filtered Side-Split View — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an analyzer-only raw three-way side-outcome split and a calibrated actionable-collapse filter on top of the existing Spec 4 recovery/re-targeting telemetry, so the §11 Spec-5-gating decision rule can be applied to a population of side windows that plausibly represent true failed re-targeting.

**Architecture:** All changes are additive. The existing `aggregate_recovery_retargeting_records` and the trainer-side sidecar emission stay unchanged (backward compatible). Two new module-level aggregators (`aggregate_recovery_retargeting_with_side_split`, `aggregate_recovery_retargeting_filtered`) live in `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py` and share six private helpers (`_filter_and_canonicalize`, `_side_bucket_for_record`, `_side_view_for_record_side`, `_iter_triggered_side_views`, `_compute_side_rollup`, `_empty_side_rollup`) so the rollup math has a single source of truth. The same `_side_view_for_record_side` helper is reused by the worst-cases CSV writer for per-row annotation, so the filtered aggregator and the worst-cases CSV cannot construct different side-view shapes. A pure-predicate `apply_actionable_filter` is reused by both the filtered aggregator and the per-row annotation, so the filtered report and the worst-cases CSV cannot disagree on what passes. Analyzer (`scripts/twixt_replay_analyzer.py`) calls all three aggregators per-iter and at range-level, formats three report sections (pooled / raw side split / filtered actionable-collapse view + filter summary), writes a new `recovery_retargeting_side_split_by_iter_<range>.csv`, and adds three columns to `recovery_retargeting_worst_cases_<range>.csv`. No self-play, no trainer change, no MCTS change.

**Tech Stack:** Python 3.14, pytest, pure-stdlib (csv module). All work re-runnable on existing per-game records in `scripts/GPU/logs/games/`.

**Spec:** [`docs/superpowers/specs/2026-05-13-recovery-retargeting-filtered-view-design.md`](../specs/2026-05-13-recovery-retargeting-filtered-view-design.md)

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py` | Modify | Add `_filter_and_canonicalize`, `_side_bucket_for_record`, `_side_view_for_record_side`, `_iter_triggered_side_views`, `_compute_side_rollup`, `_empty_side_rollup`, `aggregate_recovery_retargeting_with_side_split`, `apply_actionable_filter`, `aggregate_recovery_retargeting_filtered`. Existing `aggregate_recovery_retargeting_records`, `_bucket_rollup`, `_AGG_COUNT_KEYS`, `PRIMARY_CLASSES`, tracker classes — untouched. |
| `scripts/twixt_replay_analyzer.py` | Modify | Extend `format_recovery_retargeting_report` (3-arg signature), add `write_recovery_retargeting_side_split_csv`, extend `write_recovery_retargeting_worst_cases_csv` with 3 new columns, wire all three aggregators into the main path's recovery-retargeting block. |
| `tests/test_recovery_retargeting_diagnostics.py` | Modify | Add 9 tests (Tests 1–9 from spec §6). Reuses existing `_record()` fixture, adds two new factories for multi-side and state-cap records. |
| `tests/test_analyzer_recovery_retargeting.py` | Modify | Add 2 tests (Tests 10–11 from spec §6). Reuses existing `_summary()` factory, adds factories for the new split/filtered summaries. |

---

## Conventions

- Run all tests via the project venv: `.venv/bin/python -m pytest <path> -v`.
- All new public function signatures use keyword-only arguments after the first positional `records` / `side_view`, matching the existing aggregator's style.
- Filter reason identifiers are the stable strings: `in_window_below_min`, `triggered_below_min`, `mean_score_above_max`, `constructive_recovery_above_max`, `structural_plus_local_below_min`.
- Side-bucket identifiers: `eventual_loser`, `eventual_winner`, `state_cap_or_draw`.
- All commits use conventional commit prefixes (`feat:`, `test:`, `refactor:`) plus the scope tag pattern in recent history (e.g., `feat(diagnostics): ...`).

---

## Task 1: Side-bucket helper

**Files:**
- Modify: `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py` — append new section after the existing `aggregate_recovery_retargeting_records` (around line 823).
- Test: `tests/test_recovery_retargeting_diagnostics.py` — append at end of file.

- [ ] **Step 1: Add the import to the test file (if not already present)**

The file already imports `aggregate_recovery_retargeting_records` and `PRIMARY_CLASSES` at line 664-667. Add `_side_bucket_for_record` to the same import block:

```python
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    aggregate_recovery_retargeting_records,
    _side_bucket_for_record,
)
```

- [ ] **Step 2: Write the three failing tests at the end of `tests/test_recovery_retargeting_diagnostics.py`**

```python
def test_side_bucket_eventual_loser():
    rec = {"winner": "red", "loser": "black"}
    assert _side_bucket_for_record(rec, "black") == "eventual_loser"


def test_side_bucket_eventual_winner():
    rec = {"winner": "red", "loser": "black"}
    assert _side_bucket_for_record(rec, "red") == "eventual_winner"


def test_side_bucket_state_cap_or_draw():
    rec = {"winner": None, "loser": None}
    assert _side_bucket_for_record(rec, "red") == "state_cap_or_draw"
    assert _side_bucket_for_record(rec, "black") == "state_cap_or_draw"
```

- [ ] **Step 3: Run tests to verify they fail with ImportError**

Run: `.venv/bin/python -m pytest tests/test_recovery_retargeting_diagnostics.py::test_side_bucket_eventual_loser tests/test_recovery_retargeting_diagnostics.py::test_side_bucket_eventual_winner tests/test_recovery_retargeting_diagnostics.py::test_side_bucket_state_cap_or_draw -v`

Expected: collection error or `ImportError: cannot import name '_side_bucket_for_record'`.

- [ ] **Step 4: Add the helper section header and the helper itself to `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`**

Append after the existing `aggregate_recovery_retargeting_records` function (after line 823):

```python
# ---------------------------------------------------------------------------
# Side-split aggregation (Spec 2026-05-13 filtered side-split view)
# ---------------------------------------------------------------------------


def _side_bucket_for_record(record: dict, side: str) -> str:
    """Map a triggered side to its eventual-game-outcome bucket.

    Returns one of: 'eventual_loser', 'eventual_winner', 'state_cap_or_draw'.

    Buckets are by *eventual game outcome*, not by side at the trigger ply.
    Draws and state-caps land in 'state_cap_or_draw' for both sides.
    """
    winner = record.get("winner")
    loser = record.get("loser")
    if winner is None:
        # winner and loser co-occur in production data; either being None
        # implies the game ended state-cap or draw.
        return "state_cap_or_draw"
    if side == loser:
        return "eventual_loser"
    if side == winner:
        return "eventual_winner"
    # Defensive fallback: side is neither winner nor loser in a classified
    # game. Should not occur with current self-play; treat as state_cap_or_draw.
    return "state_cap_or_draw"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_recovery_retargeting_diagnostics.py::test_side_bucket_eventual_loser tests/test_recovery_retargeting_diagnostics.py::test_side_bucket_eventual_winner tests/test_recovery_retargeting_diagnostics.py::test_side_bucket_state_cap_or_draw -v`

Expected: 3 passed.

- [ ] **Step 6: Run the whole test file to confirm no regression**

Run: `.venv/bin/python -m pytest tests/test_recovery_retargeting_diagnostics.py -v`

Expected: all existing tests still pass + 3 new ones.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/recovery_retargeting_diagnostics.py tests/test_recovery_retargeting_diagnostics.py
git commit -m "$(cat <<'EOF'
feat(diagnostics): _side_bucket_for_record helper (Spec filtered-view §2)

Maps a triggered side to one of eventual_loser / eventual_winner /
state_cap_or_draw based on the per-game record's winner/loser fields.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Side-split aggregator with shared rollup helpers

**Files:**
- Modify: `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py` — add private helpers and the side-split aggregator after Task 1's `_side_bucket_for_record`.
- Test: `tests/test_recovery_retargeting_diagnostics.py` — append after Task 1's tests.

This task introduces five private helpers (`_filter_and_canonicalize`, `_side_view_for_record_side`, `_iter_triggered_side_views`, `_empty_side_rollup`, `_compute_side_rollup`) and the first public aggregator that uses them. Together with `_side_bucket_for_record` from Task 1, this completes the six side-split helpers listed in the Architecture section. Test 4 (recombination) and Test 9 (weighted-mean math) live here.

- [ ] **Step 1: Extend the import in the test file**

```python
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    aggregate_recovery_retargeting_records,
    aggregate_recovery_retargeting_with_side_split,
    _side_bucket_for_record,
)
```

- [ ] **Step 2: Add a new test fixture for two-side records at the end of `tests/test_recovery_retargeting_diagnostics.py`**

```python
def _record_two_sides(
    *,
    winner="red",
    loser_classes=None,
    winner_classes=None,
    loser_in_window=20, loser_triggered=10,
    winner_in_window=10, winner_triggered=4,
    loser_score_mean=-0.90, winner_score_mean=-0.80,
):
    """Record where both sides triggered. Used for split/filter tests."""
    loser = "black" if winner == "red" else "red"
    loser_classes = loser_classes or {"redundant_local_reinforcement": 10}
    winner_classes = winner_classes or {"redundant_local_reinforcement": 4}

    def _side(triggered, in_window, classes, score_mean, top1_mean=0.15, mins=None, maxs=None):
        counts = {c: 0 for c in PRIMARY_CLASSES}
        counts.update(classes)
        classified = sum(counts.values())
        return {
            "triggered": True,
            "in_window_own_moves": in_window,
            "triggered_own_moves": triggered,
            "non_triggered_in_window_moves": in_window - triggered,
            "missing_signal_moves": 0,
            "severe_collapse_moves": triggered // 2,
            "very_diffuse_moves": triggered,
            "trigger_reason_counts": {"delta_precursor": 1, "steady_state": triggered - 1, "both": 0},
            "classified_in_window_moves": classified,
            "selected_class_counts": counts,
            "mean_search_score_triggered_plies": score_mean,
            "min_search_score_triggered_plies": mins if mins is not None else score_mean - 0.05,
            "max_search_score_triggered_plies": maxs if maxs is not None else score_mean + 0.05,
            "mean_root_top1_share_triggered_plies": top1_mean,
            "classifier_error_count": 0,
        }

    return {
        "version": 1,
        "iteration": 170, "game_idx": 0, "game_id": "game_000",
        "winner": winner, "loser": loser,
        "triggered_sides": ["red", "black"],
        "side_records": {
            loser: _side(loser_triggered, loser_in_window, loser_classes, loser_score_mean),
            winner: _side(winner_triggered, winner_in_window, winner_classes, winner_score_mean),
        },
        "classifier_error_count": 0,
        "config": {
            "collapse_value_threshold": -0.75,
            "severe_collapse_value_threshold": -0.90,
            "diffuse_root_top1_threshold": 0.20,
            "very_diffuse_root_top1_threshold": 0.15,
            "delta_threshold": 0.50,
            "delta_max_current_score": -0.30,
            "alternate_component_min_size": 4,
            "classify_defense": True,
        },
    }


def _record_state_cap(triggered_sides=("red", "black"), classes=None, in_window=25, triggered=5, score_mean=-0.95):
    """Record where the game ended state-cap/draw (winner=None)."""
    classes = classes or {"redundant_local_reinforcement": 5}
    counts = {c: 0 for c in PRIMARY_CLASSES}
    counts.update(classes)
    classified = sum(counts.values())

    def _side():
        return {
            "triggered": True,
            "in_window_own_moves": in_window,
            "triggered_own_moves": triggered,
            "non_triggered_in_window_moves": in_window - triggered,
            "missing_signal_moves": 0,
            "severe_collapse_moves": triggered // 2,
            "very_diffuse_moves": triggered,
            "trigger_reason_counts": {"delta_precursor": 0, "steady_state": triggered, "both": 0},
            "classified_in_window_moves": classified,
            "selected_class_counts": counts,
            "mean_search_score_triggered_plies": score_mean,
            "min_search_score_triggered_plies": score_mean - 0.05,
            "max_search_score_triggered_plies": score_mean + 0.05,
            "mean_root_top1_share_triggered_plies": 0.15,
            "classifier_error_count": 0,
        }

    side_records = {}
    for s in ("red", "black"):
        if s in triggered_sides:
            side_records[s] = _side()
        else:
            side_records[s] = {"triggered": False, "classifier_error_count": 0}

    return {
        "version": 1,
        "iteration": 170, "game_idx": 0, "game_id": "game_000",
        "winner": None, "loser": None,
        "triggered_sides": list(triggered_sides),
        "side_records": side_records,
        "classifier_error_count": 0,
        "config": {
            "collapse_value_threshold": -0.75,
            "severe_collapse_value_threshold": -0.90,
            "diffuse_root_top1_threshold": 0.20,
            "very_diffuse_root_top1_threshold": 0.15,
            "delta_threshold": 0.50,
            "delta_max_current_score": -0.30,
            "alternate_component_min_size": 4,
            "classify_defense": True,
        },
    }
```

- [ ] **Step 3: Write Test 4 (recombination) and Test 9 (weighted-mean math) at the end of the test file**

```python
def test_side_split_rollup_recombines_to_pooled_counts():
    """Spec §6 Test 4. Sum of counts across the three side-split buckets
    must equal the corresponding totals from the pooled aggregator."""
    recs = [
        _record(side="black", in_window=10, triggered=8, classified=10),
        _record_two_sides(loser_in_window=20, loser_triggered=10, winner_in_window=8, winner_triggered=3),
        _record_state_cap(in_window=25, triggered=5),
    ]
    pooled = aggregate_recovery_retargeting_records(recs, games_total=100)
    split  = aggregate_recovery_retargeting_with_side_split(recs, games_total=100)

    split_in_window = (
        split["eventual_loser"]["in_window_own_moves_total"]
        + split["eventual_winner"]["in_window_own_moves_total"]
        + split["state_cap_or_draw"]["in_window_own_moves_total"]
    )
    split_classified = (
        split["eventual_loser"]["classified_in_window_moves_total"]
        + split["eventual_winner"]["classified_in_window_moves_total"]
        + split["state_cap_or_draw"]["classified_in_window_moves_total"]
    )
    assert split_in_window == pooled["in_window_own_moves_total"]
    assert split_classified == pooled["classified_in_window_moves_total"]


def test_bucket_score_aggregation_uses_pooled_weighted_mean():
    """Spec §6 Test 9. Bucket mean_search_score_triggered_plies =
    sum(per_side_mean * per_side_triggered) / sum(per_side_triggered).

    Two records, both with winner=red+loser=black, so both black sides
    land in eventual_loser. This exercises multi-side aggregation
    *within* the same bucket — the actual invariant the weighting rule
    is meant to enforce. (A fixture with one side per bucket would only
    test that the bucket holds a single side's mean unchanged.)"""
    rec_long = _record(
        side="black", in_window=120, triggered=120, classified=120,
        classes={"redundant_local_reinforcement": 120},
    )
    rec_long["side_records"]["black"]["mean_search_score_triggered_plies"]    = -0.95
    rec_long["side_records"]["black"]["min_search_score_triggered_plies"]     = -0.97
    rec_long["side_records"]["black"]["max_search_score_triggered_plies"]     = -0.93
    rec_long["side_records"]["black"]["mean_root_top1_share_triggered_plies"] = 0.10

    rec_short = _record(
        side="black", in_window=3, triggered=3, classified=3,
        classes={"redundant_local_reinforcement": 3},
    )
    rec_short["side_records"]["black"]["mean_search_score_triggered_plies"]    = -0.40
    rec_short["side_records"]["black"]["min_search_score_triggered_plies"]     = -0.42
    rec_short["side_records"]["black"]["max_search_score_triggered_plies"]     = -0.38
    rec_short["side_records"]["black"]["mean_root_top1_share_triggered_plies"] = 0.30
    # _record() sets game_idx=0 for both — keep distinct to avoid identity confusion.
    rec_short["game_idx"] = 1
    rec_short["game_id"]  = "game_001"

    split = aggregate_recovery_retargeting_with_side_split(
        [rec_long, rec_short], games_total=2,
    )
    bucket = split["eventual_loser"]
    assert bucket["sides"] == 2
    expected_score = (-0.95 * 120 + -0.40 * 3) / 123
    expected_top1  = (0.10 * 120 + 0.30 * 3) / 123
    # Aggregator rounds to 3 places; assert with same tolerance.
    assert abs(bucket["mean_search_score_triggered_plies"] - round(expected_score, 3)) < 1e-9
    assert abs(bucket["mean_root_top1_share_triggered_plies"] - round(expected_top1,  3)) < 1e-9
    # min/max are min-of-mins / max-of-maxes (not weighted).
    assert bucket["min_search_score_triggered_plies"] == -0.97
    assert bucket["max_search_score_triggered_plies"] == -0.38


def test_side_view_for_record_side_matches_iter_triggered_side_views():
    """Helper invariant: _side_view_for_record_side and _iter_triggered_side_views
    must produce equivalent views for the same triggered sides. Pins the
    'single source of truth' contract used by Task 7's worst-cases CSV writer."""
    from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
        _side_view_for_record_side, _iter_triggered_side_views,
    )
    rec = _record_two_sides(winner="red", loser_in_window=20, loser_triggered=10,
                            winner_in_window=8, winner_triggered=4)
    iterator_views = list(_iter_triggered_side_views([rec]))
    helper_views = [
        _side_view_for_record_side(rec, "red"),
        _side_view_for_record_side(rec, "black"),
    ]
    helper_views = [v for v in helper_views if v is not None]
    # Sort both by side_bucket for stable comparison.
    iterator_views.sort(key=lambda v: v["side_bucket"])
    helper_views.sort(key=lambda v: v["side_bucket"])
    assert iterator_views == helper_views


def test_side_view_derives_rates_when_missing():
    """Spec filtered-view §4.2: _side_view_for_record_side derives the four
    bucket rates from selected_class_counts when the underlying side record
    omits them. Protects the filter from defaulting to zero on records that
    pre-date the rate fields or come from a parallel implementation."""
    from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
        _side_view_for_record_side,
    )
    rec = _record(side="black", in_window=20, triggered=10, classified=10,
                  classes={"connects_to_existing_component": 4,
                           "redundant_local_reinforcement": 6})
    # Strip the derived rate fields the test factory set.
    for k in ("constructive_recovery_rate", "defensive_rate",
              "structural_connection_rate", "local_drift_rate"):
        rec["side_records"]["black"].pop(k, None)
    view = _side_view_for_record_side(rec, "black")
    assert view is not None
    # 4 structural / 10 classified = 0.4; 6 local_drift / 10 = 0.6.
    assert view["structural_connection_rate"] == 0.4
    assert view["local_drift_rate"]           == 0.6
    assert view["constructive_recovery_rate"] == 0.0
    assert view["defensive_rate"]             == 0.0


def test_side_split_schema_for_empty_records():
    """Spec filtered-view §4.7. Empty records list returns a fully-shaped
    summary with all three buckets present (zero counts, None scores)."""
    out = aggregate_recovery_retargeting_with_side_split([], games_total=10)
    assert out["games_total"] == 10
    assert out["games_triggered"] == 0
    for bucket in ("eventual_loser", "eventual_winner", "state_cap_or_draw"):
        b = out[bucket]
        assert b["sides"] == 0
        assert b["in_window_own_moves_total"] == 0
        assert b["mean_search_score_triggered_plies"] is None
        assert b["constructive_recovery_rate"] == 0.0
    assert out["schema_integrity"]["classifier_error_count_total"] == 0


def test_bucket_score_aggregation_ignores_missing_mean_values():
    """A side view with triggered_own_moves > 0 but a None mean_search_score
    must NOT contribute its weight to the score denominator. Otherwise the
    pooled mean is biased toward zero. Same rule applies to mean_root_top1_share."""
    rec_with = _record(
        side="black", in_window=10, triggered=10, classified=10,
        classes={"redundant_local_reinforcement": 10},
    )
    rec_with["side_records"]["black"]["mean_search_score_triggered_plies"]    = -0.90
    rec_with["side_records"]["black"]["mean_root_top1_share_triggered_plies"] = 0.10

    rec_without = _record(
        side="black", in_window=10, triggered=10, classified=10,
        classes={"redundant_local_reinforcement": 10},
    )
    rec_without["game_idx"] = 1
    rec_without["game_id"]  = "game_001"
    rec_without["side_records"]["black"]["mean_search_score_triggered_plies"]    = None
    rec_without["side_records"]["black"]["mean_root_top1_share_triggered_plies"] = None
    rec_without["side_records"]["black"]["min_search_score_triggered_plies"]     = None
    rec_without["side_records"]["black"]["max_search_score_triggered_plies"]     = None

    split = aggregate_recovery_retargeting_with_side_split(
        [rec_with, rec_without], games_total=2,
    )
    bucket = split["eventual_loser"]
    assert bucket["sides"] == 2
    # Pooled score should equal the side-with-mean's value, not (-0.90 + 0)/2.
    assert bucket["mean_search_score_triggered_plies"] == -0.9
    assert bucket["mean_root_top1_share_triggered_plies"] == 0.1


def test_split_schema_integrity_matches_existing_pooled_behavior():
    """Parity test: _filter_and_canonicalize must reproduce the existing
    aggregator's accepted-record / skipped-counts behavior on the same
    inputs. Catches future drift if either filter implementation changes."""
    a = _record(side="black")
    b = _record(side="black")
    b["version"] = 99             # unknown version
    c = _record(side="black")
    c["config"] = dict(c["config"])
    c["config"]["collapse_value_threshold"] = -0.50  # config mismatch

    pooled = aggregate_recovery_retargeting_records([a, b, c], games_total=10)
    split  = aggregate_recovery_retargeting_with_side_split([a, b, c], games_total=10)

    assert pooled["games_triggered"] == split["games_triggered"]
    p_si = pooled["schema_integrity"]
    s_si = split["schema_integrity"]
    assert p_si["skipped_unknown_version_count"] == s_si["skipped_unknown_version_count"]
    assert p_si["skipped_config_mismatch_count"] == s_si["skipped_config_mismatch_count"]
    assert p_si["classifier_error_count_total"]  == s_si["classifier_error_count_total"]
```

- [ ] **Step 4: Run tests to verify they fail with ImportError**

Run: `.venv/bin/python -m pytest tests/test_recovery_retargeting_diagnostics.py -k "side_split or bucket_score or side_view or split_schema" -v`

Expected: ImportError on `aggregate_recovery_retargeting_with_side_split`.

- [ ] **Step 5: Add the five private helpers to `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`**

Insert after `_side_bucket_for_record` (added in Task 1). The helpers — in dependency order: `_filter_and_canonicalize`, `_side_view_for_record_side`, `_iter_triggered_side_views`, `_empty_side_rollup`, `_compute_side_rollup`:

```python
_SIDE_BUCKETS = ("eventual_loser", "eventual_winner", "state_cap_or_draw")

_RATE_KEYS = (
    "constructive_recovery_rate",
    "defensive_rate",
    "structural_connection_rate",
    "local_drift_rate",
)


def _filter_and_canonicalize(records, *, config):
    """Apply version + config-mismatch filtering. Mirrors the logic in
    aggregate_recovery_retargeting_records so all aggregators see the
    same accepted-record set semantics."""
    skipped_unknown_version = 0
    skipped_config_mismatch = 0
    accepted = []
    canonical_config = config
    for rec in records:
        if rec is None:
            continue
        if rec.get("version") != 1:
            skipped_unknown_version += 1
            continue
        cfg = rec.get("config") or {}
        if canonical_config is None:
            canonical_config = cfg
        elif cfg != canonical_config:
            skipped_config_mismatch += 1
            continue
        accepted.append(rec)
    return accepted, canonical_config, skipped_unknown_version, skipped_config_mismatch


def _side_view_for_record_side(record, side):
    """Build a normalized side-view dict for a single (record, side) pair.

    Returns None if the side did not trigger. The view carries 'side_bucket'
    plus all per-side stats needed by both _compute_side_rollup and
    apply_actionable_filter. If derived rates (constructive_recovery_rate,
    defensive_rate, structural_connection_rate, local_drift_rate) are missing
    from the underlying side record, they are computed from selected_class_counts
    so the filter never silently sees zeros.

    Single source of truth for side-view construction — used by
    _iter_triggered_side_views (filtered/split aggregation) and by the
    worst-cases CSV writer (per-row annotation)."""
    sr = (record.get("side_records") or {}).get(side) or {}
    if not sr.get("triggered"):
        return None
    view = dict(sr)
    view["side_bucket"] = _side_bucket_for_record(record, side)
    # Derive any of the four rollup rates that are missing — if even one
    # is absent, recompute all four from selected_class_counts so the
    # filter never silently sees a default-zero where a partial record
    # has it set inconsistently.
    if any(k not in view for k in _RATE_KEYS):
        counts = view.get("selected_class_counts") or {}
        classified = sum(counts.values())
        denom = classified if classified > 0 else 1
        rollup = _bucket_rollup(counts, denom=denom)
        view["constructive_recovery_rate"] = rollup["constructive_recovery_rate"]
        view["defensive_rate"]              = rollup["defensive_rate"]
        view["structural_connection_rate"]  = rollup["structural_connection_rate"]
        view["local_drift_rate"]            = rollup["local_drift_rate"]
    return view


def _iter_triggered_side_views(records):
    """Yield one normalized side-view dict per triggered side across all
    records. Delegates per-side construction to _side_view_for_record_side."""
    for rec in records:
        for side in ("red", "black"):
            view = _side_view_for_record_side(rec, side)
            if view is not None:
                yield view


def _empty_side_rollup() -> dict:
    """Stable zero schema for an empty side bucket."""
    return {
        "sides": 0,
        "in_window_own_moves_total": 0,
        "triggered_own_moves_total": 0,
        "non_triggered_in_window_moves_total": 0,
        "missing_signal_moves_total": 0,
        "severe_collapse_moves_total": 0,
        "very_diffuse_moves_total": 0,
        "classified_in_window_moves_total": 0,
        "selected_class_counts_total": {c: 0 for c in PRIMARY_CLASSES},
        "selected_class_rates_total":  {c: 0.0 for c in PRIMARY_CLASSES},
        "trigger_reason_counts_total": {"delta_precursor": 0, "steady_state": 0, "both": 0},
        "constructive_recovery_rate":  0.0,
        "defensive_rate":              0.0,
        "structural_connection_rate":  0.0,
        "local_drift_rate":            0.0,
        "mean_search_score_triggered_plies":    None,
        "min_search_score_triggered_plies":     None,
        "max_search_score_triggered_plies":     None,
        "mean_root_top1_share_triggered_plies": None,
    }


def _compute_side_rollup(side_views) -> dict:
    """Roll up a list of side views into a single bucket summary.

    Counts are summed. Class rates are pooled (count / classified_total).
    Score means are pooled triggered-ply means weighted by triggered_own_moves.
    Min/max are taken across per-side mins/maxes. If no side views, returns
    _empty_side_rollup()."""
    views = list(side_views)
    if not views:
        return _empty_side_rollup()

    out = _empty_side_rollup()
    out["sides"] = len(views)

    count_keys = (
        "in_window_own_moves",
        "triggered_own_moves",
        "non_triggered_in_window_moves",
        "missing_signal_moves",
        "severe_collapse_moves",
        "very_diffuse_moves",
        "classified_in_window_moves",
    )
    count_to_total = {
        "in_window_own_moves": "in_window_own_moves_total",
        "triggered_own_moves": "triggered_own_moves_total",
        "non_triggered_in_window_moves": "non_triggered_in_window_moves_total",
        "missing_signal_moves": "missing_signal_moves_total",
        "severe_collapse_moves": "severe_collapse_moves_total",
        "very_diffuse_moves": "very_diffuse_moves_total",
        "classified_in_window_moves": "classified_in_window_moves_total",
    }
    for v in views:
        for k in count_keys:
            out[count_to_total[k]] += int(v.get(k, 0) or 0)
        for cls, c in (v.get("selected_class_counts") or {}).items():
            if cls in out["selected_class_counts_total"]:
                out["selected_class_counts_total"][cls] += int(c or 0)
        for reason, c in (v.get("trigger_reason_counts") or {}).items():
            if reason in out["trigger_reason_counts_total"]:
                out["trigger_reason_counts_total"][reason] += int(c or 0)

    classified = out["classified_in_window_moves_total"]
    denom = classified if classified > 0 else 1
    out["selected_class_rates_total"] = {
        cls: round(c / denom, 3)
        for cls, c in out["selected_class_counts_total"].items()
    }
    rollup = _bucket_rollup(out["selected_class_counts_total"], denom=denom)
    out["constructive_recovery_rate"] = rollup["constructive_recovery_rate"]
    out["defensive_rate"]              = rollup["defensive_rate"]
    out["structural_connection_rate"]  = rollup["structural_connection_rate"]
    out["local_drift_rate"]            = rollup["local_drift_rate"]

    # Score statistics: pooled means weighted by triggered_own_moves; min of mins,
    # max of maxes. Separate denominators for score vs top1 — a side with a
    # triggered count but a missing mean must NOT contribute its weight to the
    # denominator (otherwise the pooled mean gets biased toward zero).
    weighted_score = 0.0
    weighted_top1 = 0.0
    score_weight_sum = 0
    top1_weight_sum = 0
    mins = []
    maxs = []
    for v in views:
        w = int(v.get("triggered_own_moves", 0) or 0)
        if w <= 0:
            continue
        ms = v.get("mean_search_score_triggered_plies")
        mt = v.get("mean_root_top1_share_triggered_plies")
        mn = v.get("min_search_score_triggered_plies")
        mx = v.get("max_search_score_triggered_plies")
        if ms is not None:
            weighted_score += float(ms) * w
            score_weight_sum += w
        if mt is not None:
            weighted_top1 += float(mt) * w
            top1_weight_sum += w
        if mn is not None:
            mins.append(float(mn))
        if mx is not None:
            maxs.append(float(mx))

    if score_weight_sum > 0:
        out["mean_search_score_triggered_plies"]    = round(weighted_score / score_weight_sum, 3)
    if top1_weight_sum > 0:
        out["mean_root_top1_share_triggered_plies"] = round(weighted_top1  / top1_weight_sum, 3)
    out["min_search_score_triggered_plies"] = round(min(mins), 3) if mins else None
    out["max_search_score_triggered_plies"] = round(max(maxs), 3) if maxs else None

    return out
```

- [ ] **Step 6: Add `aggregate_recovery_retargeting_with_side_split` immediately after the helpers**

```python
def aggregate_recovery_retargeting_with_side_split(
    records,
    *,
    games_total: int,
    config: Optional[dict] = None,
) -> dict:
    """Three-way side-outcome split aggregator. Spec 2026-05-13 §4.

    Returns a dict with eventual_loser / eventual_winner / state_cap_or_draw
    rollups (each shape produced by _compute_side_rollup), plus games_total,
    games_triggered, schema_integrity, and the canonical config block.

    Existing aggregate_recovery_retargeting_records is unchanged; this is
    a sibling, not a replacement.
    """
    accepted, canonical_config, skipped_version, skipped_config = (
        _filter_and_canonicalize(records, config=config)
    )
    games_triggered = len(accepted)
    classifier_error_total = sum(int(r.get("classifier_error_count", 0)) for r in accepted)

    by_bucket: Dict[str, list] = {b: [] for b in _SIDE_BUCKETS}
    for view in _iter_triggered_side_views(accepted):
        by_bucket[view["side_bucket"]].append(view)

    return {
        "version": 1,
        "view": "raw_side_split",
        "enabled": True,
        "config": canonical_config or {},
        "games_total": games_total,
        "games_triggered": games_triggered,
        "eventual_loser":     _compute_side_rollup(by_bucket["eventual_loser"]),
        "eventual_winner":    _compute_side_rollup(by_bucket["eventual_winner"]),
        "state_cap_or_draw":  _compute_side_rollup(by_bucket["state_cap_or_draw"]),
        "schema_integrity": {
            "skipped_unknown_version_count": skipped_version,
            "skipped_config_mismatch_count": skipped_config,
            "classifier_error_count_total":  classifier_error_total,
        },
    }
```

- [ ] **Step 7: Run the two tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_recovery_retargeting_diagnostics.py::test_side_split_rollup_recombines_to_pooled_counts tests/test_recovery_retargeting_diagnostics.py::test_bucket_score_aggregation_uses_pooled_weighted_mean -v`

Expected: 2 passed.

- [ ] **Step 8: Run the whole test file to confirm no regression**

Run: `.venv/bin/python -m pytest tests/test_recovery_retargeting_diagnostics.py -v`

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add scripts/GPU/alphazero/recovery_retargeting_diagnostics.py tests/test_recovery_retargeting_diagnostics.py
git commit -m "$(cat <<'EOF'
feat(diagnostics): aggregate_recovery_retargeting_with_side_split + helpers

Adds the raw three-way side-outcome split aggregator (eventual_loser /
eventual_winner / state_cap_or_draw) plus the shared side-split helpers
(_filter_and_canonicalize, _side_view_for_record_side,
_iter_triggered_side_views, _empty_side_rollup, _compute_side_rollup).
Bucket-level mean_search_score and mean_root_top1_share are pooled
triggered-ply means weighted by triggered_own_moves; min/max are
min-of-mins / max-of-maxes (Spec filtered-view §4.4).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Actionable-collapse filter predicate

**Files:**
- Modify: `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py` — append after `aggregate_recovery_retargeting_with_side_split`.
- Test: `tests/test_recovery_retargeting_diagnostics.py` — append after Task 2's tests.

- [ ] **Step 1: Extend the import in the test file**

```python
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    aggregate_recovery_retargeting_records,
    aggregate_recovery_retargeting_with_side_split,
    apply_actionable_filter,
    _side_bucket_for_record,
)
```

- [ ] **Step 2: Write Tests 5 and 6 at the end of the test file**

```python
def test_apply_actionable_filter_passes_clean_failure_case():
    """Spec §6 Test 5. A side view passing all five clauses returns (True, [])."""
    side_view = {
        "in_window_own_moves": 30,
        "triggered_own_moves": 10,
        "mean_search_score_triggered_plies": -0.92,
        "constructive_recovery_rate": 0.10,
        "structural_connection_rate": 0.40,
        "local_drift_rate": 0.30,
    }
    passes, reasons = apply_actionable_filter(side_view)
    assert passes is True
    assert reasons == []


def test_apply_actionable_filter_reports_all_failed_reasons():
    """Spec §6 Test 6. A side view failing 3 clauses returns (False, [<3 ids>])."""
    side_view = {
        "in_window_own_moves": 5,             # fails in_window_below_min
        "triggered_own_moves": 1,             # fails triggered_below_min
        "mean_search_score_triggered_plies": -0.50,  # fails mean_score_above_max
        "constructive_recovery_rate": 0.10,
        "structural_connection_rate": 0.40,
        "local_drift_rate": 0.30,
    }
    passes, reasons = apply_actionable_filter(side_view)
    assert passes is False
    assert "in_window_below_min" in reasons
    assert "triggered_below_min" in reasons
    assert "mean_score_above_max" in reasons
    assert len(reasons) == 3
```

- [ ] **Step 3: Run tests to verify they fail with ImportError**

Run: `.venv/bin/python -m pytest tests/test_recovery_retargeting_diagnostics.py::test_apply_actionable_filter_passes_clean_failure_case tests/test_recovery_retargeting_diagnostics.py::test_apply_actionable_filter_reports_all_failed_reasons -v`

Expected: ImportError on `apply_actionable_filter`.

- [ ] **Step 4: Add the predicate to `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`**

Insert after `aggregate_recovery_retargeting_with_side_split`:

```python
def apply_actionable_filter(
    side_view: dict,
    *,
    min_in_window_own_moves: int = 20,
    min_triggered_own_moves: int = 3,
    max_mean_search_score_triggered_plies: float = -0.85,
    max_constructive_recovery_rate: float = 0.30,
    min_structural_plus_local_rate: float = 0.60,
):
    """Pure predicate. Returns (passes: bool, reasons_failed: list[str]).

    Five clauses, all must hold for passes=True. Each failed clause appends
    one stable reason id. A side view can fail multiple clauses.
    """
    reasons = []
    if int(side_view.get("in_window_own_moves", 0) or 0) < min_in_window_own_moves:
        reasons.append("in_window_below_min")
    if int(side_view.get("triggered_own_moves", 0) or 0) < min_triggered_own_moves:
        reasons.append("triggered_below_min")
    mean_score = side_view.get("mean_search_score_triggered_plies")
    if mean_score is None or float(mean_score) > max_mean_search_score_triggered_plies:
        reasons.append("mean_score_above_max")
    if float(side_view.get("constructive_recovery_rate", 0.0) or 0.0) >= max_constructive_recovery_rate:
        reasons.append("constructive_recovery_above_max")
    structural_plus_local = (
        float(side_view.get("structural_connection_rate", 0.0) or 0.0)
        + float(side_view.get("local_drift_rate", 0.0) or 0.0)
    )
    if structural_plus_local < min_structural_plus_local_rate:
        reasons.append("structural_plus_local_below_min")
    return (len(reasons) == 0, reasons)
```

- [ ] **Step 5: Run the two tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_recovery_retargeting_diagnostics.py::test_apply_actionable_filter_passes_clean_failure_case tests/test_recovery_retargeting_diagnostics.py::test_apply_actionable_filter_reports_all_failed_reasons -v`

Expected: 2 passed.

- [ ] **Step 6: Run the whole test file to confirm no regression**

Run: `.venv/bin/python -m pytest tests/test_recovery_retargeting_diagnostics.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/recovery_retargeting_diagnostics.py tests/test_recovery_retargeting_diagnostics.py
git commit -m "$(cat <<'EOF'
feat(diagnostics): apply_actionable_filter predicate (Spec filtered-view §3)

Pure predicate with five clauses (in_window>=20, triggered>=3,
mean_score<=-0.85, constructive_recovery<30%, structural+local>=60%).
Returns (passes, reasons_failed) where reasons_failed contains the
stable id of every clause that failed. Will be used both by the
filtered aggregator and by the worst-cases CSV row annotation so the
two views cannot disagree on what passes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Filtered side-split aggregator

**Files:**
- Modify: `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py` — append after `apply_actionable_filter`.
- Test: `tests/test_recovery_retargeting_diagnostics.py` — append after Task 3's tests.

- [ ] **Step 1: Extend the import in the test file**

```python
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    aggregate_recovery_retargeting_records,
    aggregate_recovery_retargeting_with_side_split,
    aggregate_recovery_retargeting_filtered,
    apply_actionable_filter,
    _side_bucket_for_record,
)
```

- [ ] **Step 2: Write Tests 7 and 8 at the end of the test file**

```python
def test_filtered_aggregator_counts_only_passing_sides():
    """Spec §6 Test 7. `sides` per filtered bucket equals the predicate's
    true count over the records' triggered sides in that bucket.

    This test intentionally does NOT pre-set the four bucket rate fields
    on the per-side records — it relies on _side_view_for_record_side
    deriving them from selected_class_counts. That's the contract used in
    production by the analyzer when it loads per-game JSONs."""
    # Passing loser side (structural 40%, local 60%, sum=100%, constructive 0%).
    passing_loser = _record(
        side="black", in_window=30, triggered=10,
        classified=10,
        classes={"connects_to_existing_component": 4,
                 "redundant_local_reinforcement": 3,
                 "off_plan_or_unclear": 3},
    )
    passing_loser["side_records"]["black"]["mean_search_score_triggered_plies"] = -0.92
    passing_loser["side_records"]["black"]["min_search_score_triggered_plies"]  = -0.95
    passing_loser["side_records"]["black"]["max_search_score_triggered_plies"]  = -0.86
    passing_loser["side_records"]["black"]["mean_root_top1_share_triggered_plies"] = 0.12
    # Strip the rate fields the test factory sets — force the helper to derive them.
    for k in ("constructive_recovery_rate", "defensive_rate",
              "structural_connection_rate", "local_drift_rate"):
        passing_loser["side_records"]["black"].pop(k, None)

    # Failing loser side (in_window=5 -> fails in_window_below_min).
    failing_loser = _record(
        side="black", in_window=5, triggered=2,
        classified=2,
        classes={"redundant_local_reinforcement": 2},
    )
    failing_loser["side_records"]["black"]["mean_search_score_triggered_plies"] = -0.92
    for k in ("constructive_recovery_rate", "defensive_rate",
              "structural_connection_rate", "local_drift_rate"):
        failing_loser["side_records"]["black"].pop(k, None)

    out = aggregate_recovery_retargeting_filtered(
        [passing_loser, failing_loser], games_total=10,
    )
    assert out["eventual_loser"]["sides"] == 1
    assert out["eventual_winner"]["sides"] == 0
    assert out["state_cap_or_draw"]["sides"] == 0
    assert out["filter_summary"]["side_views_total"] == 2
    assert out["filter_summary"]["side_views_passed"] == 1
    assert out["filter_summary"]["side_views_failed"] == 1


def test_filter_summary_counts_multiple_failed_reasons():
    """Spec §6 Test 8. failed_reason_counts may sum above side_views_failed
    because a side can fail multiple clauses."""
    # Single side failing three clauses: in_window=5, triggered=1, mean=-0.50
    bad = _record(side="black", in_window=5, triggered=1, classified=1)
    bad["side_records"]["black"]["mean_search_score_triggered_plies"] = -0.50
    bad["side_records"]["black"]["constructive_recovery_rate"]  = 0.10
    bad["side_records"]["black"]["structural_connection_rate"]  = 0.50
    bad["side_records"]["black"]["local_drift_rate"]            = 0.20

    out = aggregate_recovery_retargeting_filtered([bad], games_total=1)
    fs = out["filter_summary"]
    assert fs["side_views_total"] == 1
    assert fs["side_views_failed"] == 1
    # in_window_below_min + triggered_below_min + mean_score_above_max
    # + structural_plus_local_below_min = 4 reasons, sum > 1
    counts = fs["failed_reason_counts"]
    assert counts["in_window_below_min"] == 1
    assert counts["triggered_below_min"] == 1
    assert counts["mean_score_above_max"] == 1
    assert counts["structural_plus_local_below_min"] == 1
    assert sum(counts.values()) > fs["side_views_failed"]


def test_filtered_schema_for_empty_records():
    """Spec filtered-view §4.7. Empty records list returns a fully-shaped
    summary with all three buckets zeroed AND filter_summary present
    with all-zero counts and the default filter_config."""
    out = aggregate_recovery_retargeting_filtered([], games_total=10)
    assert out["games_total"] == 10
    assert out["games_triggered"] == 0
    for bucket in ("eventual_loser", "eventual_winner", "state_cap_or_draw"):
        assert out[bucket]["sides"] == 0
    fs = out["filter_summary"]
    assert fs["side_views_total"] == 0
    assert fs["side_views_passed"] == 0
    assert fs["side_views_failed"] == 0
    assert all(v == 0 for v in fs["failed_reason_counts"].values())
    assert fs["filter_config"]["min_in_window_own_moves"] == 20
```

- [ ] **Step 3: Run tests to verify they fail with ImportError**

Run: `.venv/bin/python -m pytest tests/test_recovery_retargeting_diagnostics.py::test_filtered_aggregator_counts_only_passing_sides tests/test_recovery_retargeting_diagnostics.py::test_filter_summary_counts_multiple_failed_reasons tests/test_recovery_retargeting_diagnostics.py::test_filtered_schema_for_empty_records -v`

Expected: ImportError on `aggregate_recovery_retargeting_filtered`.

- [ ] **Step 4: Add the filtered aggregator to `scripts/GPU/alphazero/recovery_retargeting_diagnostics.py`**

Insert after `apply_actionable_filter`:

```python
_DEFAULT_FILTER_CONFIG = {
    "min_in_window_own_moves": 20,
    "min_triggered_own_moves": 3,
    "max_mean_search_score_triggered_plies": -0.85,
    "max_constructive_recovery_rate": 0.30,
    "min_structural_plus_local_rate": 0.60,
}

_FILTER_REASON_KEYS = (
    "in_window_below_min",
    "triggered_below_min",
    "mean_score_above_max",
    "constructive_recovery_above_max",
    "structural_plus_local_below_min",
)


def aggregate_recovery_retargeting_filtered(
    records,
    *,
    games_total: int,
    config: Optional[dict] = None,
    filter_config: Optional[dict] = None,
) -> dict:
    """Filtered side-split aggregator. Spec 2026-05-13 §4.

    Same shape as aggregate_recovery_retargeting_with_side_split, but only
    counts side views that pass apply_actionable_filter. Adds a top-level
    'filter_summary' with per-clause failed counts.
    """
    fcfg = dict(_DEFAULT_FILTER_CONFIG)
    if filter_config:
        fcfg.update(filter_config)

    accepted, canonical_config, skipped_version, skipped_config = (
        _filter_and_canonicalize(records, config=config)
    )
    games_triggered = len(accepted)
    classifier_error_total = sum(int(r.get("classifier_error_count", 0)) for r in accepted)

    by_bucket: Dict[str, list] = {b: [] for b in _SIDE_BUCKETS}
    side_views_total = 0
    side_views_passed = 0
    side_views_failed = 0
    failed_reason_counts = {k: 0 for k in _FILTER_REASON_KEYS}

    for view in _iter_triggered_side_views(accepted):
        side_views_total += 1
        passes, reasons = apply_actionable_filter(view, **fcfg)
        if passes:
            by_bucket[view["side_bucket"]].append(view)
            side_views_passed += 1
        else:
            side_views_failed += 1
            for r in reasons:
                if r in failed_reason_counts:
                    failed_reason_counts[r] += 1

    return {
        "version": 1,
        "view": "filtered_actionable_collapse",
        "enabled": True,
        "config": canonical_config or {},
        "games_total": games_total,
        "games_triggered": games_triggered,
        "eventual_loser":     _compute_side_rollup(by_bucket["eventual_loser"]),
        "eventual_winner":    _compute_side_rollup(by_bucket["eventual_winner"]),
        "state_cap_or_draw":  _compute_side_rollup(by_bucket["state_cap_or_draw"]),
        "filter_summary": {
            "filter_config":         dict(fcfg),
            "side_views_total":      side_views_total,
            "side_views_passed":     side_views_passed,
            "side_views_failed":     side_views_failed,
            "failed_reason_counts":  failed_reason_counts,
        },
        "schema_integrity": {
            "skipped_unknown_version_count": skipped_version,
            "skipped_config_mismatch_count": skipped_config,
            "classifier_error_count_total":  classifier_error_total,
        },
    }
```

- [ ] **Step 5: Run the three tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_recovery_retargeting_diagnostics.py::test_filtered_aggregator_counts_only_passing_sides tests/test_recovery_retargeting_diagnostics.py::test_filter_summary_counts_multiple_failed_reasons tests/test_recovery_retargeting_diagnostics.py::test_filtered_schema_for_empty_records -v`

Expected: 3 passed.

- [ ] **Step 6: Run the whole test file to confirm no regression**

Run: `.venv/bin/python -m pytest tests/test_recovery_retargeting_diagnostics.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/GPU/alphazero/recovery_retargeting_diagnostics.py tests/test_recovery_retargeting_diagnostics.py
git commit -m "$(cat <<'EOF'
feat(diagnostics): aggregate_recovery_retargeting_filtered (Spec filtered-view §4.6)

Filtered side-split aggregator. Same three-bucket shape as the raw split,
but only counts sides where apply_actionable_filter returns True. Adds a
filter_summary with side_views_total/passed/failed and per-clause
failed_reason_counts. failed_reason_counts may sum above side_views_failed
because a side can fail multiple clauses (documented behavior).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Extend report formatter with side-split + filtered sections

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` — change `format_recovery_retargeting_report` signature and add new section rendering (around line 2481-2550).
- Test: `tests/test_analyzer_recovery_retargeting.py` — add a smoke test that calls the formatter with all three summaries and checks that the new section headers appear.

- [ ] **Step 1: Read the existing formatter signature and main-path call site**

Run: `grep -n "format_recovery_retargeting_report\|def format_recovery_retargeting" /Users/bill/projects/TwixT_Game/scripts/twixt_replay_analyzer.py`

Confirm the function is at line 2481 and the main-path call is at line 5018.

- [ ] **Step 2: Add a smoke test at the end of `tests/test_analyzer_recovery_retargeting.py`**

```python
def _split_summary(**overrides):
    """Minimal raw_side_split summary fixture."""
    base = {
        "version": 1,
        "view": "raw_side_split",
        "enabled": True,
        "config": _summary()["config"],
        "games_total": 1000, "games_triggered": 1000,
        "eventual_loser":    _empty_split_bucket(sides=989, in_window=10000, triggered=8000,
                                                 mean_score=-0.92, constructive=0.20, defensive=0.01,
                                                 structural=0.55, local=0.24),
        "eventual_winner":   _empty_split_bucket(sides=342, in_window=2500, triggered=1500,
                                                 mean_score=-0.78, constructive=0.30, defensive=0.00,
                                                 structural=0.50, local=0.20),
        "state_cap_or_draw": _empty_split_bucket(sides=8, in_window=200, triggered=120,
                                                 mean_score=-0.93, constructive=0.18, defensive=0.02,
                                                 structural=0.55, local=0.25),
        "schema_integrity": {
            "skipped_unknown_version_count": 0,
            "skipped_config_mismatch_count": 0,
            "classifier_error_count_total": 0,
        },
    }
    base.update(overrides)
    return base


def _filtered_summary(**overrides):
    """Minimal filtered_actionable_collapse summary fixture."""
    base = _split_summary()
    base["view"] = "filtered_actionable_collapse"
    base["filter_summary"] = {
        "filter_config": {
            "min_in_window_own_moves": 20,
            "min_triggered_own_moves": 3,
            "max_mean_search_score_triggered_plies": -0.85,
            "max_constructive_recovery_rate": 0.30,
            "min_structural_plus_local_rate": 0.60,
        },
        "side_views_total":  1339,
        "side_views_passed": 412,
        "side_views_failed": 927,
        "failed_reason_counts": {
            "in_window_below_min":             120,
            "triggered_below_min":             210,
            "mean_score_above_max":            300,
            "constructive_recovery_above_max": 250,
            "structural_plus_local_below_min": 90,
        },
    }
    base.update(overrides)
    return base


def _empty_split_bucket(*, sides=0, in_window=0, triggered=0, mean_score=None,
                        constructive=0.0, defensive=0.0, structural=0.0, local=0.0):
    return {
        "sides": sides,
        "in_window_own_moves_total": in_window,
        "triggered_own_moves_total": triggered,
        "non_triggered_in_window_moves_total": in_window - triggered,
        "missing_signal_moves_total": 0,
        "severe_collapse_moves_total": triggered // 2,
        "very_diffuse_moves_total": triggered,
        "classified_in_window_moves_total": triggered,
        "selected_class_counts_total": {
            "blocks_opponent_closeout": 0, "reduces_own_goal_distance": 0,
            "starts_or_extends_alternate_component": 0,
            "connects_to_existing_component": 0, "improves_own_largest_component": 0,
            "redundant_local_reinforcement": 0, "off_plan_or_unclear": 0,
        },
        "selected_class_rates_total": {
            "blocks_opponent_closeout": 0.0, "reduces_own_goal_distance": 0.0,
            "starts_or_extends_alternate_component": 0.0,
            "connects_to_existing_component": 0.0, "improves_own_largest_component": 0.0,
            "redundant_local_reinforcement": 0.0, "off_plan_or_unclear": 0.0,
        },
        "trigger_reason_counts_total": {"delta_precursor": 0, "steady_state": triggered, "both": 0},
        "constructive_recovery_rate": constructive,
        "defensive_rate":             defensive,
        "structural_connection_rate": structural,
        "local_drift_rate":           local,
        "mean_search_score_triggered_plies":    mean_score,
        "min_search_score_triggered_plies":     (mean_score - 0.05) if mean_score is not None else None,
        "max_search_score_triggered_plies":     (mean_score + 0.05) if mean_score is not None else None,
        "mean_root_top1_share_triggered_plies": 0.15 if mean_score is not None else None,
    }


def test_format_report_renders_three_sections():
    pooled = _summary()
    split = _split_summary()
    filtered = _filtered_summary()
    lines = format_recovery_retargeting_report(pooled, split, filtered)
    text = "\n".join(lines)
    assert "Raw side-outcome split" in text
    assert "Filtered actionable-collapse view" in text
    assert "Filter summary" in text
    assert "eventual_loser" in text
    assert "eventual_winner" in text
    assert "state_cap_or_draw" in text


def test_format_report_backward_compat_pooled_only():
    """When split and filtered are None, render only the existing pooled section."""
    pooled = _summary()
    lines = format_recovery_retargeting_report(pooled, None, None)
    text = "\n".join(lines)
    assert "Recovery / Re-targeting Diagnostics" in text
    assert "Raw side-outcome split" not in text
    assert "Filtered actionable-collapse view" not in text
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_analyzer_recovery_retargeting.py::test_format_report_renders_three_sections tests/test_analyzer_recovery_retargeting.py::test_format_report_backward_compat_pooled_only -v`

Expected: TypeError (formatter currently takes 1 arg, called with 3).

- [ ] **Step 4: Extend `format_recovery_retargeting_report` in `scripts/twixt_replay_analyzer.py`**

**Conservative extension — minimize blast radius.** Do NOT rewrite the existing pooled rendering logic line-by-line. The two changes are:

1. Change the function signature from `format_recovery_retargeting_report(summary)` to `format_recovery_retargeting_report(summary, split_summary=None, filtered_summary=None)`.
2. After the existing `lines.append("Worst cases: recovery_retargeting_worst_cases.csv")` line, add the new section blocks gated on `if split_summary:` and `if filtered_summary:`.

The full extended function is shown below for reference, but the diff should be a signature change + an append at the end + a new `_format_side_split_block` helper. Anything in between (the existing pooled-rendering block) stays byte-identical:

```python
def format_recovery_retargeting_report(
    summary,
    split_summary=None,
    filtered_summary=None,
):
    """Format the recovery / re-targeting telemetry section.

    Spec 4 §6.5 (pooled, original) + Spec 2026-05-13 §5.1 (raw side-outcome
    split + filtered actionable-collapse view + filter summary).
    """
    if not summary:
        return []
    cfg = summary.get("config") or {}
    classify_defense_on = bool(cfg.get("classify_defense", True))

    def _pct(x):
        return f"{(x or 0.0) * 100.0:.1f}%"

    lines = []
    lines.append("Recovery / Re-targeting Diagnostics")
    lines.append("===================================")
    iters = summary.get("iters_covered") or []
    if iters:
        lines.append(
            f"Iters covered: {min(iters)}-{max(iters)}  enabled={summary.get('enabled')}  "
            f"defense_classifier={'on' if classify_defense_on else 'off'}"
        )
    lines.append(
        f"Config: collapse_value<={cfg.get('collapse_value_threshold')}  "
        f"diffuse_root_top1<={cfg.get('diffuse_root_top1_threshold')}  "
        f"delta>={cfg.get('delta_threshold')} with current<={cfg.get('delta_max_current_score')}"
    )
    games_total = summary.get("games_total", 0)
    games_triggered = summary.get("games_triggered", 0)
    lines.append(f"Triggered games:           {games_triggered} / {games_total} ({_pct(summary.get('trigger_rate'))})")
    lines.append(f"  side was eventual loser: {summary.get('triggered_loser_side', 0)} / {games_triggered} ({_pct(summary.get('triggered_loser_side_per_triggered_game'))})")
    lines.append(f"  side was eventual winner:{summary.get('triggered_winner_side', 0):4d} / {games_triggered} ({_pct(summary.get('triggered_winner_side_per_triggered_game'))})")
    in_window = summary.get("in_window_own_moves_total", 0)
    lines.append(f"In-window own moves:       {in_window}")
    lines.append(f"  triggered:               {summary.get('triggered_own_moves_total', 0)}")
    lines.append(f"  non-triggered in-window: {summary.get('non_triggered_in_window_moves_total', 0)}")
    lines.append(f"  missing-signal:          {summary.get('missing_signal_moves_total', 0)}")
    lines.append("Severity:")
    lines.append(f"  severe collapse:         {summary.get('severe_collapse_moves_total', 0)} plies")
    lines.append(f"  very diffuse root:       {summary.get('very_diffuse_moves_total', 0)} plies")
    trc = summary.get("trigger_reason_counts_total") or {}
    lines.append("Trigger composition:")
    lines.append(f"  delta_precursor:         {trc.get('delta_precursor', 0)}")
    lines.append(f"  steady_state:            {trc.get('steady_state', 0)}")
    lines.append(f"  both:                    {trc.get('both', 0)}")
    counts = summary.get("selected_class_counts_total") or {}
    rates = summary.get("selected_class_rates_total") or {}
    lines.append("Move-class composition (denominator: classified in-window):")
    for cls, label in (
        ("blocks_opponent_closeout",              "blocks opponent closeout:"),
        ("reduces_own_goal_distance",             "reduces own goal distance:"),
        ("starts_or_extends_alternate_component", "starts/extends alternate component:"),
        ("connects_to_existing_component",        "connects to existing component:"),
        ("improves_own_largest_component",        "improves own largest component:"),
        ("redundant_local_reinforcement",         "redundant local reinforcement:"),
        ("off_plan_or_unclear",                   "off-plan or unclear:"),
    ):
        lines.append(f"  {label:42s} {_pct(rates.get(cls)):>6s}   ({counts.get(cls, 0)})")
    lines.append("Rollup:")
    lines.append(f"  constructive recovery:                 {_pct(summary.get('constructive_recovery_rate'))}")
    if classify_defense_on:
        lines.append(f"  defense:                               {_pct(summary.get('defensive_rate'))}")
    else:
        lines.append("  defense:                  N/A (defense classification disabled — local drift may include defensive moves)")
    lines.append(f"  structural connection:                 {_pct(summary.get('structural_connection_rate'))}")
    lines.append(f"  local drift / unclear:                 {_pct(summary.get('local_drift_rate'))}")
    si = summary.get("schema_integrity") or {}
    lines.append("Schema integrity:")
    lines.append(f"  classifier_error_count:                {si.get('classifier_error_count_total', 0)}")
    lines.append(f"  records skipped (unknown version):     {si.get('skipped_unknown_version_count', 0)}")
    lines.append(f"  records skipped (config mismatch):     {si.get('skipped_config_mismatch_count', 0)}")
    lines.append("Worst cases: recovery_retargeting_worst_cases.csv")

    if split_summary:
        lines.append("")
        lines.append("Raw side-outcome split")
        lines.append("----------------------")
        lines.extend(_format_side_split_block(split_summary, label="triggered sides"))

    if filtered_summary:
        lines.append("")
        lines.append("Filtered actionable-collapse view")
        lines.append("---------------------------------")
        fc = (filtered_summary.get("filter_summary") or {}).get("filter_config") or {}
        lines.append(
            f"Filter: in_window>={fc.get('min_in_window_own_moves')}, "
            f"triggered>={fc.get('min_triggered_own_moves')}, "
            f"mean_score<={fc.get('max_mean_search_score_triggered_plies')}, "
            f"constructive<{int(round(fc.get('max_constructive_recovery_rate', 0)*100))}%, "
            f"structural+local>={int(round(fc.get('min_structural_plus_local_rate', 0)*100))}%"
        )
        lines.extend(_format_side_split_block(
            filtered_summary, label="eligible sides",
            winner_note="[threshold sanity only — not used as Spec 5 intervention target]",
        ))
        fs = filtered_summary.get("filter_summary") or {}
        lines.append("")
        lines.append("Filter summary:")
        lines.append(f"  side views total:    {fs.get('side_views_total', 0)}")
        lines.append(f"  side views passed:   {fs.get('side_views_passed', 0)}")
        lines.append(f"  side views failed:   {fs.get('side_views_failed', 0)}")
        lines.append("  failed reasons:")
        for reason in (
            "in_window_below_min", "triggered_below_min", "mean_score_above_max",
            "constructive_recovery_above_max", "structural_plus_local_below_min",
        ):
            lines.append(f"    {reason+':':36s} {(fs.get('failed_reason_counts') or {}).get(reason, 0)}")

    return lines


def _format_side_split_block(side_split_summary, *, label, winner_note=""):
    """Render the three side buckets within a side-split or filtered summary."""
    def _pct(x):
        return f"{(x or 0.0) * 100.0:.1f}%"
    out = []
    for bucket_key, bucket_label in (
        ("eventual_loser",    "eventual_loser:"),
        ("eventual_winner",   "eventual_winner:"),
        ("state_cap_or_draw", "state_cap_or_draw:"),
    ):
        b = side_split_summary.get(bucket_key) or {}
        out.append(bucket_label)
        sides_line = f"  {label}:        {b.get('sides', 0)}"
        if bucket_key == "eventual_winner" and winner_note:
            sides_line += f"   {winner_note}"
        out.append(sides_line)
        out.append(f"  constructive recovery:  {_pct(b.get('constructive_recovery_rate'))}")
        out.append(f"  defense:                {_pct(b.get('defensive_rate'))}")
        out.append(f"  structural connection:  {_pct(b.get('structural_connection_rate'))}")
        out.append(f"  local drift / unclear:  {_pct(b.get('local_drift_rate'))}")
        ms = b.get("mean_search_score_triggered_plies")
        mn = b.get("min_search_score_triggered_plies")
        mx = b.get("max_search_score_triggered_plies")
        mt = b.get("mean_root_top1_share_triggered_plies")
        if ms is not None:
            score_str = f"{ms:.3f}"
            if mn is not None and mx is not None:
                score_str += f"  (in [{mn:.3f}, {mx:.3f}])"
            out.append(f"  mean trigger score:     {score_str}")
        if mt is not None:
            out.append(f"  mean trigger top1:       {mt:.3f}")
        out.append("")
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_analyzer_recovery_retargeting.py -v`

Expected: all tests pass (existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_recovery_retargeting.py
git commit -m "$(cat <<'EOF'
feat(analyzer): three-section recovery-retargeting report (Spec filtered-view §5.1)

format_recovery_retargeting_report now takes (pooled, split, filtered)
and renders three sections: pooled (unchanged) + Raw side-outcome split
+ Filtered actionable-collapse view + Filter summary block. Backward
compatible: passing None for split/filtered yields the original pooled-only
output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: New side-split CSV writer

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` — append `write_recovery_retargeting_side_split_csv` after the existing `write_recovery_retargeting_by_iter_csv` (around line 2606).
- Test: `tests/test_analyzer_recovery_retargeting.py` — append after Task 5's tests.

- [ ] **Step 1: Extend the import in the test file**

```python
from scripts.twixt_replay_analyzer import (
    format_recovery_retargeting_report,
    write_recovery_retargeting_side_split_csv,
)
```

- [ ] **Step 2: Write Test 10 at the end of `tests/test_analyzer_recovery_retargeting.py`**

```python
def test_analyzer_writes_side_split_csv_with_six_rows_per_iter(tmp_path):
    """Spec §6 Test 10. For two iters, the new CSV contains 12 rows
    (6 per iter: 3 buckets x 2 views)."""
    per_iter_split = {170: _split_summary(), 171: _split_summary()}
    per_iter_filtered = {170: _filtered_summary(), 171: _filtered_summary()}
    out_path = tmp_path / "side_split.csv"
    write_recovery_retargeting_side_split_csv(
        str(out_path), per_iter_split, per_iter_filtered,
    )
    import csv
    with open(out_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    iters = sorted({int(r["iteration"]) for r in rows})
    assert iters == [170, 171]
    views = {r["view"] for r in rows}
    assert views == {"raw", "filtered"}
    buckets = {r["side_bucket"] for r in rows}
    assert buckets == {"eventual_loser", "eventual_winner", "state_cap_or_draw"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_analyzer_recovery_retargeting.py::test_analyzer_writes_side_split_csv_with_six_rows_per_iter -v`

Expected: ImportError on `write_recovery_retargeting_side_split_csv`.

- [ ] **Step 4: Add the CSV writer to `scripts/twixt_replay_analyzer.py`**

Insert after `write_recovery_retargeting_by_iter_csv` (after line 2606):

```python
def write_recovery_retargeting_side_split_csv(
    out_path: str,
    per_iter_split_summaries: dict,
    per_iter_filtered_summaries: dict,
) -> str:
    """Write recovery_retargeting_side_split_by_iter.csv. Spec 2026-05-13 §5.2.

    Long format, 6 rows per iteration: 3 side buckets x 2 views (raw, filtered).
    Empty buckets emit a row with zero counts and empty score fields.
    """
    import csv
    fields = [
        "iteration",
        "view",                # "raw" | "filtered"
        "side_bucket",         # "eventual_loser" | "eventual_winner" | "state_cap_or_draw"
        "sides",
        "in_window_own_moves_total",
        "triggered_own_moves_total",
        "mean_search_score_triggered_plies",
        "mean_root_top1_share_triggered_plies",
        "constructive_recovery_rate",
        "defensive_rate",
        "structural_connection_rate",
        "local_drift_rate",
        "redundant_local_reinforcement_rate",
        "off_plan_or_unclear_rate",
    ]
    iters = sorted(set(per_iter_split_summaries.keys()) | set(per_iter_filtered_summaries.keys()))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in iters:
            for view, summaries in (("raw", per_iter_split_summaries),
                                    ("filtered", per_iter_filtered_summaries)):
                s = summaries.get(it) or {}
                for bucket in ("eventual_loser", "eventual_winner", "state_cap_or_draw"):
                    b = s.get(bucket) or {}
                    rates = b.get("selected_class_rates_total") or {}
                    ms = b.get("mean_search_score_triggered_plies")
                    mt = b.get("mean_root_top1_share_triggered_plies")
                    w.writerow({
                        "iteration": it,
                        "view": view,
                        "side_bucket": bucket,
                        "sides": int(b.get("sides", 0) or 0),
                        "in_window_own_moves_total":  int(b.get("in_window_own_moves_total", 0) or 0),
                        "triggered_own_moves_total":  int(b.get("triggered_own_moves_total", 0) or 0),
                        "mean_search_score_triggered_plies":    "" if ms is None else ms,
                        "mean_root_top1_share_triggered_plies": "" if mt is None else mt,
                        "constructive_recovery_rate": b.get("constructive_recovery_rate", 0.0),
                        "defensive_rate":             b.get("defensive_rate", 0.0),
                        "structural_connection_rate": b.get("structural_connection_rate", 0.0),
                        "local_drift_rate":           b.get("local_drift_rate", 0.0),
                        "redundant_local_reinforcement_rate": rates.get("redundant_local_reinforcement", 0.0),
                        "off_plan_or_unclear_rate":           rates.get("off_plan_or_unclear", 0.0),
                    })
    return out_path
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_analyzer_recovery_retargeting.py::test_analyzer_writes_side_split_csv_with_six_rows_per_iter -v`

Expected: 1 passed.

- [ ] **Step 6: Run all analyzer tests**

Run: `.venv/bin/python -m pytest tests/test_analyzer_recovery_retargeting.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_recovery_retargeting.py
git commit -m "$(cat <<'EOF'
feat(analyzer): write_recovery_retargeting_side_split_csv (Spec filtered-view §5.2)

Long-format CSV: 6 rows per iteration (3 side buckets x 2 views).
Single 'sides' column whose semantics depend on 'view'. Empty buckets
emit a row with zero counts and empty-string score fields.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Worst-cases CSV: add side_bucket / passes_actionable_filter / filter_reasons_failed

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` — extend `write_recovery_retargeting_worst_cases_csv` (around line 2608-2680+).
- Test: `tests/test_analyzer_recovery_retargeting.py` — append.

- [ ] **Step 1: Read the existing worst-cases writer to confirm the row-construction site**

Run: `grep -n "def write_recovery_retargeting_worst_cases_csv\|rows.append\|fields = list" /Users/bill/projects/TwixT_Game/scripts/twixt_replay_analyzer.py`

Confirm `rows.append(...)` is the per-row construction site and `fields = list(rows[0].keys())` derives the header.

- [ ] **Step 2: Extend imports in the test file**

```python
from scripts.twixt_replay_analyzer import (
    format_recovery_retargeting_report,
    write_recovery_retargeting_side_split_csv,
    write_recovery_retargeting_worst_cases_csv,
)
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import apply_actionable_filter
```

- [ ] **Step 3: Write Test 11 at the end of `tests/test_analyzer_recovery_retargeting.py`**

```python
def _worst_cases_record(*, side="black", winner="red", in_window=30, triggered=10,
                        mean_score=-0.92, classes=None,
                        constructive=0.0, defensive=0.0, structural=0.4, local=0.6):
    """Per-game record fixture suitable for the worst-cases CSV writer."""
    classes = classes or {"redundant_local_reinforcement": 6, "connects_to_existing_component": 4}
    counts = {
        "blocks_opponent_closeout": 0, "reduces_own_goal_distance": 0,
        "starts_or_extends_alternate_component": 0,
        "connects_to_existing_component": 0, "improves_own_largest_component": 0,
        "redundant_local_reinforcement": 0, "off_plan_or_unclear": 0,
    }
    counts.update(classes)
    other = "red" if side == "black" else "black"
    return {
        "version": 1,
        "iteration": 170, "game_idx": 0, "game_id": "game_000",
        "winner": winner, "loser": side, "n_moves": 65, "reason": "win",
        "triggered_sides": [side],
        "side_records": {
            other: {"triggered": False, "classifier_error_count": 0},
            side: {
                "triggered": True,
                "first_trigger_ply": 44,
                "first_trigger_reason": "steady_state",
                "in_window_own_moves": in_window,
                "triggered_own_moves": triggered,
                "non_triggered_in_window_moves": in_window - triggered,
                "missing_signal_moves": 0,
                "severe_collapse_moves": triggered // 2,
                "very_diffuse_moves": triggered,
                "trigger_reason_counts": {"delta_precursor": 0, "steady_state": triggered, "both": 0},
                "classified_in_window_moves": sum(counts.values()),
                "selected_class_counts": counts,
                "constructive_recovery_moves": counts["reduces_own_goal_distance"] + counts["starts_or_extends_alternate_component"],
                "defensive_moves":             counts["blocks_opponent_closeout"],
                "structural_connection_moves": counts["connects_to_existing_component"] + counts["improves_own_largest_component"],
                "local_drift_moves":           counts["redundant_local_reinforcement"] + counts["off_plan_or_unclear"],
                "constructive_recovery_rate":  constructive,
                "defensive_rate":              defensive,
                "structural_connection_rate":  structural,
                "local_drift_rate":            local,
                "mean_search_score_triggered_plies":    mean_score,
                "min_search_score_triggered_plies":     mean_score - 0.05,
                "max_search_score_triggered_plies":     mean_score + 0.05,
                "mean_root_top1_share_triggered_plies": 0.12,
                "classifier_error_count": 0,
            },
        },
        "classifier_error_count": 0,
    }


def test_worst_cases_csv_uses_same_filter_predicate(tmp_path):
    """Spec §6 Test 11. Each row's passes_actionable_filter matches
    apply_actionable_filter on the same per-side data."""
    passing = _worst_cases_record(in_window=30, triggered=10, mean_score=-0.92,
                                  constructive=0.0, structural=0.4, local=0.6)
    failing = _worst_cases_record(in_window=5, triggered=1, mean_score=-0.50,
                                  constructive=0.10, structural=0.50, local=0.20)

    out_path = tmp_path / "worst.csv"
    write_recovery_retargeting_worst_cases_csv(
        str(out_path), [passing, failing], top_k=25,
    )
    import csv
    with open(out_path) as f:
        rows = list(csv.DictReader(f))

    assert "side_bucket" in rows[0]
    assert "passes_actionable_filter" in rows[0]
    assert "filter_reasons_failed" in rows[0]

    for row in rows:
        rec = passing if int(row["in_window_own_moves"]) == 30 else failing
        sr = rec["side_records"][row["triggered_side"]]
        passes, reasons = apply_actionable_filter(sr)
        assert (row["passes_actionable_filter"] == "true") == passes
        if not passes:
            assert set(row["filter_reasons_failed"].split(";")) == set(reasons)
        else:
            assert row["filter_reasons_failed"] == ""
        # side_bucket matches: loser == 'eventual_loser' here (winner=red, loser=black)
        assert row["side_bucket"] == "eventual_loser"


def test_worst_cases_csv_filter_reasons_empty_when_passes(tmp_path):
    """For a row that passes apply_actionable_filter, filter_reasons_failed
    must be the empty string (not 'none', not absent, not whitespace)."""
    passing = _worst_cases_record(in_window=30, triggered=10, mean_score=-0.92,
                                  constructive=0.0, structural=0.4, local=0.6)
    out_path = tmp_path / "worst.csv"
    write_recovery_retargeting_worst_cases_csv(str(out_path), [passing], top_k=25)
    import csv
    with open(out_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["passes_actionable_filter"] == "true"
    assert rows[0]["filter_reasons_failed"] == ""
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_analyzer_recovery_retargeting.py::test_worst_cases_csv_uses_same_filter_predicate tests/test_analyzer_recovery_retargeting.py::test_worst_cases_csv_filter_reasons_empty_when_passes -v`

Expected: KeyError on `side_bucket` (column not yet emitted) for both.

- [ ] **Step 5: Extend `write_recovery_retargeting_worst_cases_csv` in `scripts/twixt_replay_analyzer.py`**

Locate the function (around line 2608). Two structural changes:
1. Import `apply_actionable_filter` and `_side_view_for_record_side` from the diagnostics module.
2. For each row, build the side view via `_side_view_for_record_side(rec, side)` (same helper the filtered aggregator uses) and pass that to `apply_actionable_filter`. This guarantees the row's `passes_actionable_filter` value cannot disagree with the filtered aggregator's classification of the same side. The row's data fields are still pulled from `sr` (the per-side record) for compatibility with the existing column set.

```python
def write_recovery_retargeting_worst_cases_csv(
    out_path: str, records: list, *, top_k: int = 25,
) -> str:
    """Write recovery_retargeting_worst_cases.csv. Spec 4 §6.7 + Spec 2026-05-13 §5.3.

    One row per triggered side; sorted by (local_drift_moves DESC,
    in_window_own_moves DESC, min_search_score_triggered_plies ASC).

    Spec 2026-05-13 adds three columns: side_bucket, passes_actionable_filter,
    filter_reasons_failed (semicolon-separated; empty when passes). The filter
    is applied to the side view produced by _side_view_for_record_side — the
    same helper the filtered aggregator uses — so per-row annotations cannot
    drift from the filtered report.
    """
    import csv
    from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
        apply_actionable_filter, _side_view_for_record_side,
    )
    rows = []
    for rec in records:
        if not rec:
            continue
        for side in rec.get("triggered_sides") or []:
            sr = (rec.get("side_records") or {}).get(side) or {}
            counts = sr.get("selected_class_counts") or {}
            view = _side_view_for_record_side(rec, side)
            if view is None:
                # Defensive: triggered_sides included a side whose side_record
                # has triggered=False. Skip rather than emit a malformed row.
                continue
            passes, reasons = apply_actionable_filter(view)
            rows.append({
                "iteration": rec.get("iteration"),
                "game_idx": rec.get("game_idx"),
                "game_id": rec.get("game_id"),
                "winner": rec.get("winner"),
                "loser": rec.get("loser"),
                "reason": rec.get("reason"),
                "n_moves": rec.get("n_moves"),
                "triggered_side": side,
                "side_bucket": view["side_bucket"],
                "first_trigger_ply": sr.get("first_trigger_ply"),
                "first_trigger_reason": sr.get("first_trigger_reason"),
                "in_window_own_moves": sr.get("in_window_own_moves", 0),
                "triggered_own_moves": sr.get("triggered_own_moves", 0),
                "severe_collapse_moves": sr.get("severe_collapse_moves", 0),
                "very_diffuse_moves": sr.get("very_diffuse_moves", 0),
                "classified_in_window_moves": sr.get("classified_in_window_moves", 0),
                "missing_signal_moves": sr.get("missing_signal_moves", 0),
                "blocks_opponent_closeout_moves":              counts.get("blocks_opponent_closeout", 0),
                "reduces_own_goal_distance_moves":             counts.get("reduces_own_goal_distance", 0),
                "starts_or_extends_alternate_component_moves": counts.get("starts_or_extends_alternate_component", 0),
                "connects_to_existing_component_moves":        counts.get("connects_to_existing_component", 0),
                "improves_own_largest_component_moves":        counts.get("improves_own_largest_component", 0),
                "redundant_local_reinforcement_moves":         counts.get("redundant_local_reinforcement", 0),
                "off_plan_or_unclear_moves":                   counts.get("off_plan_or_unclear", 0),
                "constructive_recovery_moves": sr.get("constructive_recovery_moves", 0),
                "defensive_moves":             sr.get("defensive_moves", 0),
                "structural_connection_moves": sr.get("structural_connection_moves", 0),
                "local_drift_moves":           sr.get("local_drift_moves", 0),
                "local_drift_rate":            sr.get("local_drift_rate", 0.0),
                "constructive_recovery_rate":  sr.get("constructive_recovery_rate", 0.0),
                "mean_search_score_triggered_plies": sr.get("mean_search_score_triggered_plies"),
                "min_search_score_triggered_plies":  sr.get("min_search_score_triggered_plies"),
                "max_search_score_triggered_plies":  sr.get("max_search_score_triggered_plies"),
                "mean_root_top1_share_triggered_plies": sr.get("mean_root_top1_share_triggered_plies"),
                "passes_actionable_filter": "true" if passes else "false",
                "filter_reasons_failed":    ";".join(reasons),
            })
    rows.sort(
        key=lambda r: (
            -int(r["local_drift_moves"] or 0),
            -int(r["in_window_own_moves"] or 0),
            float(r["min_search_score_triggered_plies"]) if r.get("min_search_score_triggered_plies") is not None else 0.0,
        )
    )
    rows = rows[:max(0, int(top_k))]
    fields = list(rows[0].keys()) if rows else [
        "iteration","game_idx","game_id","winner","loser","reason","n_moves",
        "triggered_side","side_bucket","first_trigger_ply","first_trigger_reason",
        "in_window_own_moves","triggered_own_moves",
        "severe_collapse_moves","very_diffuse_moves",
        "classified_in_window_moves","missing_signal_moves",
        "blocks_opponent_closeout_moves","reduces_own_goal_distance_moves",
        "starts_or_extends_alternate_component_moves",
        "connects_to_existing_component_moves","improves_own_largest_component_moves",
        "redundant_local_reinforcement_moves","off_plan_or_unclear_moves",
        "constructive_recovery_moves","defensive_moves",
        "structural_connection_moves","local_drift_moves",
        "local_drift_rate","constructive_recovery_rate",
        "mean_search_score_triggered_plies","min_search_score_triggered_plies",
        "max_search_score_triggered_plies","mean_root_top1_share_triggered_plies",
        "passes_actionable_filter","filter_reasons_failed",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return out_path
```

(If the existing function ends with a different `with open(...)` write block, preserve that block — the only changes are the three new dict keys, the import, the empty-fields list, and the `apply_actionable_filter` call.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_analyzer_recovery_retargeting.py::test_worst_cases_csv_uses_same_filter_predicate tests/test_analyzer_recovery_retargeting.py::test_worst_cases_csv_filter_reasons_empty_when_passes -v`

Expected: 2 passed.

- [ ] **Step 7: Run all analyzer tests**

Run: `.venv/bin/python -m pytest tests/test_analyzer_recovery_retargeting.py -v`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_recovery_retargeting.py
git commit -m "$(cat <<'EOF'
feat(analyzer): worst-cases CSV side_bucket + filter annotation (Spec filtered-view §5.3)

Adds three columns to recovery_retargeting_worst_cases.csv:
  side_bucket               (eventual_loser / eventual_winner / state_cap_or_draw)
  passes_actionable_filter  (true / false)
  filter_reasons_failed     (semicolon-separated reason ids; empty when passes)

Uses the same apply_actionable_filter predicate as the filtered aggregator,
so the report's filtered view and the per-row CSV annotation cannot disagree
on what passes. Sort key and top_k truncation preserved.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Wire all three aggregators into the analyzer's main path

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` — extend the recovery-retargeting block (around line 4975-5040).

This is the integration step. Per-iter aggregator calls happen alongside the existing `per_iter_rr` collection; the range-level calls produce the inputs for the extended formatter and new CSV writer.

- [ ] **Step 1: Read the current recovery-retargeting block**

Run: `sed -n '4970,5045p' /Users/bill/projects/TwixT_Game/scripts/twixt_replay_analyzer.py`

Identify (a) where `aggregate_recovery_retargeting_records` is called for the range, (b) where the per-iter `per_iter_rr` dict is populated, (c) where `format_recovery_retargeting_report` is called, (d) where the CSV writers are called.

- [ ] **Step 2: Extend the import block at the top of that block**

Find the existing import:

```python
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    aggregate_recovery_retargeting_records,
)
```

Replace with:

```python
from scripts.GPU.alphazero.recovery_retargeting_diagnostics import (
    aggregate_recovery_retargeting_records,
    aggregate_recovery_retargeting_with_side_split,
    aggregate_recovery_retargeting_filtered,
)
```

- [ ] **Step 3: Add the per-iter split/filtered aggregation alongside per_iter_rr**

Locate the per-iter loop that populates `per_iter_rr`. Right after the line that assigns the per-iter pooled summary, add per-iter calls to the two new aggregators. The exact pattern depends on the surrounding code; match the same loop variable names. Pseudocode:

```python
per_iter_rr = {}
per_iter_rr_split = {}
per_iter_rr_filtered = {}
for iter_idx, iter_records in <existing per-iter loop>:
    per_iter_rr[iter_idx] = aggregate_recovery_retargeting_records(
        iter_records, games_total=<existing>, config=<existing>,
    )
    per_iter_rr_split[iter_idx] = aggregate_recovery_retargeting_with_side_split(
        iter_records, games_total=<existing>, config=<existing>,
    )
    per_iter_rr_filtered[iter_idx] = aggregate_recovery_retargeting_filtered(
        iter_records, games_total=<existing>, config=<existing>,
    )
```

If the existing loop already builds `per_iter_rr` differently (e.g., from sidecar `recovery_retargeting_summary` blocks rather than from per-game records), you'll need to ensure the same per-iter records are available for the new aggregators. The new aggregators need per-game records (`recovery_retargeting_record`), not the sidecar pooled summary. If only sidecar summaries are kept per-iter, refactor the loop to also retain per-iter `rr_records` slices — name the slice `per_iter_rr_records: dict[int, list[dict]]` and call the new aggregators against `per_iter_rr_records[iter_idx]`.

- [ ] **Step 4: Replace the range-level calls**

Find the range-level call:

```python
rr_summary = aggregate_recovery_retargeting_records(
    rr_records, games_total=..., config=...,
)
```

Replace with:

```python
rr_summary  = aggregate_recovery_retargeting_records(
    rr_records, games_total=..., config=...,
)
rr_split    = aggregate_recovery_retargeting_with_side_split(
    rr_records, games_total=..., config=...,
)
rr_filtered = aggregate_recovery_retargeting_filtered(
    rr_records, games_total=..., config=...,
)
```

- [ ] **Step 5: Pass split + filtered to the formatter**

Find:

```python
lines.extend(format_recovery_retargeting_report(rr_summary))
```

Replace with:

```python
lines.extend(format_recovery_retargeting_report(rr_summary, rr_split, rr_filtered))
```

- [ ] **Step 6: Add the new CSV writer call alongside the existing writers**

Find the existing `write_recovery_retargeting_by_iter_csv(...)` call. Add the new CSV writer call immediately after:

```python
rr_split_path = os.path.join(out_dir, _suffixed("recovery_retargeting_side_split_by_iter", "csv", suffix))
write_recovery_retargeting_side_split_csv(
    rr_split_path, per_iter_rr_split, per_iter_rr_filtered,
)
```

(Use whatever existing helper builds the suffixed path — the existing call uses `_suffixed("recovery_retargeting_by_iter", "csv", suffix)`. Mirror that.)

- [ ] **Step 7: Add the two new sibling JSON keys to summary**

Find:

```python
summary["recovery_retargeting"] = rr_summary
```

Add two more lines immediately after:

```python
summary["recovery_retargeting_side_split"]  = rr_split
summary["recovery_retargeting_filtered"]    = rr_filtered
```

- [ ] **Step 8: Run all tests to confirm nothing regressed**

Run: `.venv/bin/python -m pytest tests/test_recovery_retargeting_diagnostics.py tests/test_analyzer_recovery_retargeting.py -v`

Expected: all tests pass.

- [ ] **Step 9: Smoke test the end-to-end analyzer on the existing 170-179 data**

Stage and run (this is read-only — re-runs the analyzer over existing per-game records, overwriting only outputs in `Replays/170-179_Replay/`):

```bash
ls Replays/170-179/ 2>/dev/null | head -3
# If Replays/170-179/ does not exist, recreate it from the handoff prompt's staging script:
mkdir -p Replays/170-179
for f in scripts/GPU/logs/games/iter_0169_game_*.json \
         scripts/GPU/logs/games/iter_017?_game_*.json; do
  ln -sf "../../$f" "Replays/170-179/$(basename $f)" 2>/dev/null
done
for f in scripts/GPU/logs/games/iter_0169_stats.json \
         scripts/GPU/logs/games/iter_017?_stats.json; do
  ln -sf "../../$f" "Replays/170-179/$(basename $f)" 2>/dev/null
done

.venv/bin/python ./scripts/twixt_replay_analyzer.py \
  --input Replays/170-179 --out Replays/170-179_Replay
```

Expected: analyzer completes without error. Inspect the new artefacts:

```bash
ls Replays/170-179_Replay/recovery_retargeting_side_split_by_iter_170-179.csv
head -2 Replays/170-179_Replay/recovery_retargeting_side_split_by_iter_170-179.csv
grep -A 60 "Raw side-outcome split" Replays/170-179_Replay/report_170-179.txt | head -70
grep -A 30 "Filtered actionable-collapse view" Replays/170-179_Replay/report_170-179.txt | head -40
head -1 Replays/170-179_Replay/recovery_retargeting_worst_cases_170-179.csv
```

The first command should show 14 columns (iteration, view, side_bucket, sides, ...); the second/third should show the new report sections; the fourth should include `side_bucket`, `passes_actionable_filter`, `filter_reasons_failed`.

- [ ] **Step 10: Commit**

```bash
git add scripts/twixt_replay_analyzer.py
git commit -m "$(cat <<'EOF'
feat(analyzer): wire side-split + filtered aggregators into main path

Calls aggregate_recovery_retargeting_with_side_split and
aggregate_recovery_retargeting_filtered alongside the existing pooled
aggregator for both per-iter slices and the range-level summary. Passes
all three to format_recovery_retargeting_report; writes new
recovery_retargeting_side_split_by_iter_<range>.csv; emits two new
sibling keys on the summary JSON.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Re-run analyzer on 170-179 data and apply §7 decision rule

This task is verification and decision, not new code. It produces the artifact the spec was written to enable — a calibrated read of the filtered view against the §7 placeholders.

- [ ] **Step 1: Re-run the analyzer on the 170-179 dataset**

```bash
.venv/bin/python ./scripts/twixt_replay_analyzer.py \
  --input Replays/170-179 --out Replays/170-179_Replay
```

- [ ] **Step 2: Read the three new report sections**

```bash
sed -n '/Recovery \/ Re-targeting Diagnostics/,/Recovery \/ dominant-component-lost/p' \
  Replays/170-179_Replay/report_170-179.txt
```

- [ ] **Step 3: Inspect the new side-split CSV (range row totals only)**

```bash
.venv/bin/python -c "
import csv
from collections import defaultdict
totals = defaultdict(lambda: defaultdict(int))
with open('Replays/170-179_Replay/recovery_retargeting_side_split_by_iter_170-179.csv') as f:
    for row in csv.DictReader(f):
        key = (row['view'], row['side_bucket'])
        totals[key]['sides'] += int(row['sides'])
        totals[key]['triggered_own_moves_total'] += int(row['triggered_own_moves_total'])
for key, vals in sorted(totals.items()):
    print(f'{key[0]:9s}  {key[1]:18s}  sides={vals[\"sides\"]:>5d}  triggered_moves={vals[\"triggered_own_moves_total\"]:>6d}')
"
```

- [ ] **Step 4: Apply the §7 decision rule (manual)**

From the report's filter summary block + the filtered side-bucket sections, compute:

```
denom = filter_summary.side_views_total
share = (filtered.eventual_loser.sides + filtered.state_cap_or_draw.sides) / denom

For the loser+state_cap_or_draw combined population:
  combined_structural_plus_local_rate
  combined_constructive_plus_defensive_rate

Decision (per spec §7):
  Justify Spec 5 if all of:
    share >= 30%   AND
    combined_structural_plus_local_rate >= 60%   AND
    combined_constructive_plus_defensive_rate <= 35%

  Block Spec 5 if any of:
    share is small (rule 1 violated)
    OR filtered eventual-loser+state_cap constructive_recovery_rate >= 40%
    OR filtered eventual_winner.sides > 20% of total filtered sides
       (filter still too permissive; tighten before deciding)

  Always-do checks:
    schema_integrity.classifier_error_count_total < 1% of classified_in_window_moves_total
    Hand-review top 3 worst-cases CSV rows where side_bucket=eventual_loser AND
      passes_actionable_filter=true; confirm bucket assignments match qualitative read
```

Record the resulting Spec-5 decision in the project memory file (`spec4_recovery_retargeting_diagnostic.md`) under a new "Filtered-view §7 read-out (170-179)" section. No code changes in this step.

- [ ] **Step 5: No commit needed unless memory was updated**

If memory was updated, commit it:

```bash
# Memory lives outside the project repo (in ~/.claude/projects/...) — no git action.
```

---

## Self-Review

After completing all tasks, verify:

**Spec coverage** — every numbered section of the spec has at least one task implementing it:
- §2 side-bucket assignment → Task 1
- §3 actionable-collapse filter → Task 3
- §4.1 public functions → Tasks 2, 3, 4
- §4.2 private helpers → Task 2 (`_side_view_for_record_side` is the single source of truth for side-view construction; reused by Task 7)
- §4.3 side-rollup shape → Task 2 (`_empty_side_rollup`)
- §4.4 weighted score aggregation → Task 2 (Test 9 — multi-side-in-one-bucket fixture)
- §4.5/§4.6 return shapes → Tasks 2, 4
- §4.7 empty-input handling → Task 2 (`test_side_split_schema_for_empty_records`), Task 4 (`test_filtered_schema_for_empty_records`)
- §4.8 backward compatibility → Task 2 (untouched existing function + `test_split_schema_integrity_matches_existing_pooled_behavior` parity test), Task 5 (formatter back-compat test)
- §5.1 report rendering → Task 5 (conservative extension — signature change + appended sections only, existing pooled lines byte-identical)
- §5.2 new CSV → Task 6
- §5.3 worst-cases extension → Task 7 (uses `_side_view_for_record_side` so the per-row annotation cannot drift from the filtered aggregator)
- §5.4 wiring → Task 8
- §6 tests 1–11 → Tasks 1, 2, 3, 4, 5, 6, 7 (plus 5 additional tests added during plan review)
- §7 decision rule → Task 9
- §8 implementation order → followed by Task 1 → 9

**Tests added during plan review (beyond spec §6's eleven):**
- `test_side_view_for_record_side_matches_iter_triggered_side_views` (Task 2) — pins single-source-of-truth contract
- `test_side_view_derives_rates_when_missing` (Task 2) — robustness against records missing rate fields
- `test_side_split_schema_for_empty_records` (Task 2) — §4.7 invariant
- `test_split_schema_integrity_matches_existing_pooled_behavior` (Task 2) — parity with existing aggregator
- `test_filtered_schema_for_empty_records` (Task 4) — §4.7 invariant
- `test_worst_cases_csv_filter_reasons_empty_when_passes` (Task 7) — explicit empty-string contract

**Type/name consistency:**
- `_side_bucket_for_record` / `_side_view_for_record_side` / `_iter_triggered_side_views` / `_compute_side_rollup` / `_empty_side_rollup` / `_filter_and_canonicalize` / `_SIDE_BUCKETS` / `_DEFAULT_FILTER_CONFIG` / `_FILTER_REASON_KEYS` — used consistently across Tasks 1–4 and reused in Task 7
- `aggregate_recovery_retargeting_with_side_split` / `aggregate_recovery_retargeting_filtered` / `apply_actionable_filter` — three public function names used consistently in Tasks 2, 3, 4, 5, 6, 7, 8
- `write_recovery_retargeting_side_split_csv` introduced in Task 6, called in Task 8
- Side-bucket strings: `eventual_loser`, `eventual_winner`, `state_cap_or_draw` — used everywhere
- Filter reason strings: 5 stable ids — used in Tasks 3, 4, 7

**Placeholder scan:** No `TBD` / `TODO` / "implement later" in any step. Each step contains either complete code or an exact verification command.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-13-recovery-retargeting-filtered-view.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
