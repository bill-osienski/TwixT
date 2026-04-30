# Replay Analyzer Per-Game Stats Surfacing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the eight new per-game persistence fields (plus pre-existing `n_moves` and `reason`) in `scripts/twixt_replay_analyzer.py`'s `summary.json` (new top-level `per_game_stats` block) and `report.txt` (new triage section near the existing `Compute:` line).

**Architecture:** Two new functions in one existing file, plus two integration call sites. **Function 1:** `aggregate_per_game_stats(replays) -> dict` — pure function, single pass over replays, returns the `per_game_stats` block per spec §4. Built incrementally across Tasks 1–3 (each task extends it). **Function 2:** `format_per_game_stats_report(per_game_stats) -> List[str]` — pure function, renders the report.txt section per spec §5. **Integration:** Task 5 adds the two call sites (summary dict literal + report `lines.append`).

**Tech Stack:** Python 3.14, pytest, numpy (already imported in analyzer for percentile computation). All work in `scripts/twixt_replay_analyzer.py` and one new test file. No new modules.

**Spec:** `docs/superpowers/specs/2026-04-29-analyzer-per-game-stats-design.md`

---

## File Structure

| File | Type | Responsibility |
|---|---|---|
| `scripts/twixt_replay_analyzer.py` | modify | Add `aggregate_per_game_stats(replays)` near `aggregate_sidecars` (~line 340). Add `format_per_game_stats_report(per_game_stats)` near other `format_*_report` functions (~line 948). Add two call sites: `summary["per_game_stats"]` (~line 1864) and `lines.extend(...)` (~line 2180). |
| `tests/test_analyzer_per_game_stats.py` | create | All 20 unit tests (tests 1–20 from spec §9). Test fixture helper `_make_replay(...)` defined inline using a sentinel for "field absent" vs "field present and null". |

---

## Task 1: Aggregation foundation — game_length, outcomes, coverage skeleton

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` — add `aggregate_per_game_stats(replays)` function near `aggregate_sidecars` (~line 340). At this stage the function returns `null` for all persistence-era distribution blocks (filled in by Tasks 2–3).
- Create: `tests/test_analyzer_per_game_stats.py` — module setup + `_make_replay` fixture helper + tests 1, 2, 9, 10.

### Step 1: Create the test file with fixture helper and the four foundation tests

- [ ] Create `tests/test_analyzer_per_game_stats.py` with this content:

```python
"""Tests for aggregate_per_game_stats and format_per_game_stats_report.

Spec: docs/superpowers/specs/2026-04-29-analyzer-per-game-stats-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest  # for pytest.approx in tests with non-trivial float arithmetic

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Sentinel distinguishes "field is absent from meta" (old schema) from
# "field is present and explicitly null" (e.g., worker_id=None for in-process).
_OMIT = object()


def _make_replay(
    *,
    n_moves=100,
    reason="win",
    worker_id=_OMIT,
    wall_time_s=_OMIT,
    final_root_value=_OMIT,
    final_top1_share=_OMIT,
    leaf_evals=_OMIT,
    backups=_OMIT,
    nn_batches=_OMIT,
    include_compute=True,
    omit_n_moves=False,
    omit_reason=False,
    omit_meta=False,
):
    """Construct a minimal replay record for aggregate_per_game_stats tests.

    Sentinel _OMIT means the meta key is absent (old-schema). Passing
    `None` means the key is present and explicitly null. Passing a value
    means the key is present with that value.
    """
    if omit_meta:
        return {}
    meta = {}
    if not omit_n_moves:
        meta["n_moves"] = n_moves
    if not omit_reason:
        meta["reason"] = reason
    if worker_id is not _OMIT:
        meta["worker_id"] = worker_id
    if wall_time_s is not _OMIT:
        meta["wall_time_s"] = wall_time_s
    if final_root_value is not _OMIT:
        meta["final_root_value"] = final_root_value
    if final_top1_share is not _OMIT:
        meta["final_top1_share"] = final_top1_share
    if include_compute:
        compute = {}
        if leaf_evals is not _OMIT:
            compute["leaf_evals"] = leaf_evals
        if backups is not _OMIT:
            compute["backups"] = backups
        if nn_batches is not _OMIT:
            compute["nn_batches"] = nn_batches
        if compute:
            meta["compute"] = compute
    return {"meta": meta}


# -------------------------------------------------------------------------
# Test 1: empty replays
# -------------------------------------------------------------------------

def test_aggregate_returns_zero_coverage_for_empty_replays():
    """aggregate_per_game_stats([]) returns the documented zero-coverage shape."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    out = aggregate_per_game_stats([])

    assert out["n_games_total"] == 0
    assert out["n_games_with_any_stats"] == 0
    # Coverage map present with all zeros, including pre-existing fields
    cov = out["coverage"]
    for key in ("wall_time_s", "worker_id", "final_root_value", "final_top1_share",
                "compute.leaf_evals", "compute.backups", "compute.nn_batches",
                "n_moves", "reason"):
        assert cov[key] == 0, f"coverage[{key!r}] should be 0"
    # Distribution blocks all null
    assert out["game_length"] is None
    assert out["wall_time_s"] is None
    assert out["final_root_value"] is None
    assert out["final_top1_share"] is None
    assert out["compute_per_game"] is None
    # Outcomes always present, all zero
    assert out["outcomes"] == {"decisive": 0, "resign": 0, "adjudicated": 0,
                                "timeout": 0, "draw_other": 0}
    # Worker balance shape
    wb = out["worker_balance"]
    assert wb["by_worker"] == {}
    assert wb["in_process_count"] == 0
    assert wb["max_min_wall_time_ratio"] is None
    assert wb["max_min_games_ratio"] is None
    assert wb["wall_time_cv"] is None


# -------------------------------------------------------------------------
# Test 2: old-schema only (no persistence-era fields)
# -------------------------------------------------------------------------

def test_aggregate_returns_zero_coverage_for_old_schema_only():
    """Replays without any persistence fields → all persistence blocks null,
    but game_length and outcomes still populated from pre-existing fields."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        _make_replay(n_moves=100, reason="win"),
        _make_replay(n_moves=120, reason="resign"),
        _make_replay(n_moves=80,  reason="timeout_selfplay"),
    ]
    out = aggregate_per_game_stats(replays)

    assert out["n_games_total"] == 3
    assert out["n_games_with_any_stats"] == 0
    # Persistence-era coverage all zero
    for key in ("wall_time_s", "worker_id", "final_root_value", "final_top1_share",
                "compute.leaf_evals", "compute.backups", "compute.nn_batches"):
        assert out["coverage"][key] == 0
    # Pre-existing fields fully covered
    assert out["coverage"]["n_moves"] == 3
    assert out["coverage"]["reason"] == 3
    # Persistence blocks all null
    assert out["wall_time_s"] is None
    assert out["final_root_value"] is None
    assert out["final_top1_share"] is None
    assert out["compute_per_game"] is None
    # game_length and outcomes populated
    assert out["game_length"] is not None
    assert out["game_length"]["max"] == 120
    assert out["game_length"]["min"] == 80
    assert out["outcomes"]["decisive"] == 1
    assert out["outcomes"]["resign"] == 1
    assert out["outcomes"]["timeout"] == 1


# -------------------------------------------------------------------------
# Test 9: outcomes categorize meta.reason correctly
# -------------------------------------------------------------------------

def test_aggregate_outcomes_categorizes_meta_reason():
    """meta.reason values map to the five outcome categories per spec §4.

    timeout and timeout_selfplay both → outcomes.timeout
    Unrecognized reasons → outcomes.draw_other
    """
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        _make_replay(reason="win"),                # decisive
        _make_replay(reason="win"),                # decisive
        _make_replay(reason="resign"),             # resign
        _make_replay(reason="adjudicated"),        # adjudicated
        _make_replay(reason="timeout"),            # timeout
        _make_replay(reason="timeout_selfplay"),   # timeout
        _make_replay(reason="board_full"),         # draw_other
        _make_replay(reason="state_cap"),          # draw_other
        _make_replay(reason="unknown"),            # draw_other
        _make_replay(reason="something_weird"),    # draw_other (unrecognized)
    ]
    out = aggregate_per_game_stats(replays)

    assert out["outcomes"]["decisive"]    == 2
    assert out["outcomes"]["resign"]      == 1
    assert out["outcomes"]["adjudicated"] == 1
    assert out["outcomes"]["timeout"]     == 2
    assert out["outcomes"]["draw_other"]  == 4
    # Counts sum to n_games_total (mutually exclusive categories invariant)
    assert sum(out["outcomes"].values()) == out["n_games_total"] == 10


# -------------------------------------------------------------------------
# Test 10: game_length uses meta.n_moves and computes percentiles
# -------------------------------------------------------------------------

def test_aggregate_game_length_uses_meta_n_moves():
    """game_length stats computed from meta.n_moves; percentiles correct."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    # Use 10 evenly-spaced values so percentiles are easy to check.
    n_moves_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    replays = [_make_replay(n_moves=n) for n in n_moves_values]
    out = aggregate_per_game_stats(replays)

    gl = out["game_length"]
    assert gl["min"] == 10
    assert gl["max"] == 100
    assert gl["mean"] == 55.0
    # numpy.percentile linear interpolation on this set:
    # p50 = 55, p90 = 91, p95 ≈ 95.5, p99 ≈ 99.1
    # (p95 and p99 use abs-epsilon because numpy returns 95.49999999999999
    #  / 99.10000000000001 due to IEEE 754 linear-interpolation rounding.)
    assert gl["p50"] == 55.0
    assert gl["p90"] == 91.0
    assert abs(gl["p95"] - 95.5) < 1e-9
    assert abs(gl["p99"] - 99.1) < 1e-9
    # coverage reflects all 10 replays carried n_moves
    assert out["coverage"]["n_moves"] == 10
```

### Step 2: Run the tests to verify they fail

Run: `.venv/bin/python -m pytest tests/test_analyzer_per_game_stats.py -v`

Expected: **FAIL** with `ImportError: cannot import name 'aggregate_per_game_stats' from 'scripts.twixt_replay_analyzer'`.

### Step 3: Implement the foundation of aggregate_per_game_stats

- [ ] In `scripts/twixt_replay_analyzer.py`, locate `aggregate_sidecars` (line 340). Just **above** that function (so the new function sits next to it logically), add:

```python
def aggregate_per_game_stats(replays: List[dict]) -> dict:
    """Aggregate per-game stats from loaded replay records.

    Reads:
      - meta.n_moves (pre-existing) → game_length distribution
      - meta.reason (pre-existing) → outcomes breakdown
      - meta.{worker_id, wall_time_s, final_root_value, final_top1_share,
        compute.{leaf_evals, backups, nn_batches}} (new in 2026-04-29
        persistence change) → distribution blocks + worker_balance + compute_per_game

    Old replays lacking persistence fields are silently excluded from
    those per-stat aggregates. Per-field coverage is recorded in
    coverage.{...} so consumers can tell exactly which stats are reliable.
    A missing meta.compute subkey is treated as MISSING (excluded from
    that subkey's stats), not as zero.

    Worker identity comes solely from meta.worker_id; never inferred
    from filename, sidecar, or any other source.

    Pure function. Does not mutate replays.

    Spec: docs/superpowers/specs/2026-04-29-analyzer-per-game-stats-design.md
    """
    n_games_total = len(replays)

    # --- Coverage counters (always present in output) ---
    coverage = {
        "wall_time_s": 0,
        "worker_id": 0,
        "final_root_value": 0,
        "final_top1_share": 0,
        "compute.leaf_evals": 0,
        "compute.backups": 0,
        "compute.nn_batches": 0,
        "n_moves": 0,
        "reason": 0,
    }

    # --- Accumulators for distribution blocks (Tasks 2 and 3 fill these) ---
    n_moves_arr = []
    outcomes = {"decisive": 0, "resign": 0, "adjudicated": 0, "timeout": 0, "draw_other": 0}

    # --- n_games_with_any_stats: counted in Task 2 (and updated in Task 3 for worker_id).
    # Until then the only persistence-era data we track is none, so it stays 0. ---
    n_games_with_any_stats = 0

    # --- One pass over replays ---
    for rp in replays:
        meta = rp.get("meta") or {}

        # n_moves (pre-existing field, treated like other fields: missing != 0)
        n_moves = meta.get("n_moves")
        if n_moves is not None:
            coverage["n_moves"] += 1
            n_moves_arr.append(int(n_moves))

        # reason (pre-existing field) → outcomes
        reason = meta.get("reason")
        if reason is not None:
            coverage["reason"] += 1
        # Categorize: missing or unrecognized → draw_other
        if reason == "win":
            outcomes["decisive"] += 1
        elif reason == "resign":
            outcomes["resign"] += 1
        elif reason == "adjudicated":
            outcomes["adjudicated"] += 1
        elif reason in ("timeout", "timeout_selfplay"):
            outcomes["timeout"] += 1
        else:
            outcomes["draw_other"] += 1

    # --- game_length distribution ---
    if coverage["n_moves"] > 0:
        arr = np.asarray(n_moves_arr, dtype=np.float64)
        game_length = {
            "mean": float(arr.mean()),
            "p50":  float(np.percentile(arr, 50)),
            "p90":  float(np.percentile(arr, 90)),
            "p95":  float(np.percentile(arr, 95)),
            "p99":  float(np.percentile(arr, 99)),
            "max":  int(arr.max()),
            "min":  int(arr.min()),
        }
    else:
        game_length = None

    # --- Build output (persistence-era blocks filled in Tasks 2 and 3) ---
    return {
        "n_games_total": n_games_total,
        "n_games_with_any_stats": n_games_with_any_stats,
        "coverage": coverage,
        "game_length": game_length,
        "outcomes": outcomes,
        "wall_time_s": None,           # Task 2 fills this in
        "worker_balance": {            # Task 3 fills the body of this
            "by_worker": {},
            "in_process_count": 0,
            "max_min_wall_time_ratio": None,
            "max_min_games_ratio": None,
            "wall_time_cv": None,
        },
        "final_root_value": None,      # Task 2 fills this in
        "final_top1_share": None,      # Task 2 fills this in
        "compute_per_game": None,      # Task 2 fills this in
    }
```

Note: `numpy` is imported as `np` at the top of the analyzer (search for `import numpy as np` to confirm). `List` and `dict` are also imported via the existing `from typing import ...` block.

### Step 4: Run the tests to verify they pass

Run: `.venv/bin/python -m pytest tests/test_analyzer_per_game_stats.py -v`

Expected: **PASS** — all 4 tests (test_aggregate_returns_zero_coverage_for_empty_replays, test_aggregate_returns_zero_coverage_for_old_schema_only, test_aggregate_outcomes_categorizes_meta_reason, test_aggregate_game_length_uses_meta_n_moves).

### Step 5: Run the existing analyzer regression to confirm no breakage

Run: `.venv/bin/python -m pytest tests/test_analyzer_phase2_sidecar_fields.py tests/test_analyzer_phase2_smoke.py tests/test_analyzer_replay_probe_scoring_end_to_end.py tests/test_strong_advantage_analyzer_aggregation.py -v`

Expected: All existing analyzer tests still pass (we only added a new function; no existing code modified).

### Step 6: Commit

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_per_game_stats.py
git commit -m "feat(analyzer): aggregate_per_game_stats foundation (game_length, outcomes, coverage)

Adds the new aggregate_per_game_stats(replays) function with the coverage
map shape, n_games_with_any_stats counter, game_length distribution
(from meta.n_moves), and outcomes breakdown (from meta.reason). The
persistence-era distribution blocks (wall_time_s, final_root_value,
final_top1_share, compute_per_game, worker_balance internals) are
returned as null/empty placeholders — they get filled in by Tasks 2 and
3 of the implementation plan.

Adds tests/test_analyzer_per_game_stats.py with the _make_replay
fixture helper (sentinel-based, distinguishes 'field absent' from
'field present and null') and the four foundation tests (1, 2, 9, 10
from spec §9).

Spec: docs/superpowers/specs/2026-04-29-analyzer-per-game-stats-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Distribution aggregation (wall_time, final_root_value, final_top1_share, compute_per_game)

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` — extend `aggregate_per_game_stats` to compute the four persistence-era distribution blocks.
- Test: `tests/test_analyzer_per_game_stats.py` — append tests 3, 4, 8.

### Step 1: Append test 3 (full coverage populates all blocks) to the test file

- [ ] Append to `tests/test_analyzer_per_game_stats.py`:

```python
# -------------------------------------------------------------------------
# Test 3: full coverage populates all blocks
# -------------------------------------------------------------------------

def test_aggregate_full_coverage_populates_all_blocks():
    """5 replays, every persistence-era field populated → all blocks non-null."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = []
    for i in range(5):
        replays.append(_make_replay(
            n_moves=100 + i,
            worker_id=i % 2,
            wall_time_s=10.0 + i,
            final_root_value=0.1 * i,
            final_top1_share=0.2 + 0.1 * i,
            leaf_evals=1000 + i * 100,
            backups=2000 + i * 100,
            nn_batches=50 + i * 5,
        ))
    out = aggregate_per_game_stats(replays)

    assert out["n_games_total"] == 5
    assert out["n_games_with_any_stats"] == 5
    # Every coverage entry == 5
    for key in ("wall_time_s", "worker_id", "final_root_value", "final_top1_share",
                "compute.leaf_evals", "compute.backups", "compute.nn_batches",
                "n_moves", "reason"):
        assert out["coverage"][key] == 5, f"coverage[{key!r}] should be 5"
    # Distribution blocks non-null
    assert out["wall_time_s"] is not None
    assert out["wall_time_s"]["mean"] == 12.0           # mean of [10,11,12,13,14] — exact int math
    assert out["wall_time_s"]["min"] == 10.0
    assert out["wall_time_s"]["max"] == 14.0
    assert out["wall_time_s"]["total"] == 60.0          # sum
    # Decimal arithmetic on [0, 0.1, 0.2, 0.3, 0.4] is not byte-exact in IEEE 754.
    assert out["final_root_value"]["mean"]     == pytest.approx(0.2)
    assert out["final_root_value"]["abs_mean"] == pytest.approx(0.2)  # all values >= 0 here
    assert out["final_top1_share"]["mean"]     == pytest.approx(0.4)  # mean of [0.2, 0.3, 0.4, 0.5, 0.6]
    assert out["final_top1_share"]["min"]      == pytest.approx(0.2)
    assert out["compute_per_game"] is not None
    assert out["compute_per_game"]["leaf_evals"]["mean"] == 1200.0  # mean of [1000,1100,1200,1300,1400] — exact int math
    assert out["compute_per_game"]["backups"]["mean"] == 2200.0
    assert out["compute_per_game"]["nn_batches"]["mean"] == 60.0


# -------------------------------------------------------------------------
# Test 4: per-field coverage counts independently
# -------------------------------------------------------------------------

def test_aggregate_per_field_coverage_counts_independently():
    """Mixed coverage: 8 have wall_time_s, 5 have final_top1_share, 7 have nn_batches."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = []
    for i in range(10):
        kw = {"n_moves": 50 + i, "reason": "win"}
        if i < 8:
            kw["wall_time_s"] = 1.0 * (i + 1)
        if i < 5:
            kw["final_top1_share"] = 0.5
        if i < 7:
            kw["leaf_evals"] = 100
            kw["backups"]   = 200
            kw["nn_batches"] = 10
        # else: include_compute=True but no compute keys means meta.compute absent
        replays.append(_make_replay(**kw))

    out = aggregate_per_game_stats(replays)

    assert out["n_games_total"] == 10
    assert out["coverage"]["wall_time_s"] == 8
    assert out["coverage"]["final_top1_share"] == 5
    assert out["coverage"]["compute.nn_batches"] == 7
    assert out["coverage"]["compute.leaf_evals"] == 7
    assert out["coverage"]["compute.backups"] == 7
    assert out["coverage"]["final_root_value"] == 0
    assert out["coverage"]["worker_id"] == 0
    # Distribution blocks computed only over their covering games
    assert out["wall_time_s"] is not None
    assert out["wall_time_s"]["total"] == 36.0          # 1+2+...+8
    assert out["final_top1_share"] is not None
    assert out["final_top1_share"]["mean"] == 0.5       # all five are 0.5
    assert out["compute_per_game"]["nn_batches"]["mean"] == 10.0
    # final_root_value has zero coverage → null
    assert out["final_root_value"] is None


# -------------------------------------------------------------------------
# Test 8: missing compute subkey is excluded, not zero
# -------------------------------------------------------------------------

def test_aggregate_compute_subkey_missing_is_excluded_not_zero():
    """meta.compute = {leaf_evals: 100, backups: 200} (no nn_batches) →
    coverage.compute.nn_batches == 0; nn_batches block is null;
    leaf_evals/backups stats reflect actual values, not depressed by phantom zeros.
    """
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        _make_replay(leaf_evals=100, backups=200),  # no nn_batches
        _make_replay(leaf_evals=300, backups=400),  # no nn_batches
    ]
    out = aggregate_per_game_stats(replays)

    assert out["coverage"]["compute.leaf_evals"] == 2
    assert out["coverage"]["compute.backups"] == 2
    assert out["coverage"]["compute.nn_batches"] == 0
    assert out["compute_per_game"] is not None
    assert out["compute_per_game"]["leaf_evals"]["mean"] == 200.0  # (100+300)/2
    assert out["compute_per_game"]["backups"]["mean"] == 300.0     # (200+400)/2
    assert out["compute_per_game"]["nn_batches"] is None


# -------------------------------------------------------------------------
# Test 8b: empty meta.compute does not count as carrying any persistence stats
# -------------------------------------------------------------------------

def test_empty_compute_object_does_not_count_as_any_stats():
    """A replay with meta.compute = {} (key present but no subkeys) must NOT
    increment n_games_with_any_stats — we count actual populated fields, not
    just key presence."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    # _make_replay with no compute kwargs and include_compute=True → meta.compute
    # is OMITTED entirely. To get an explicit empty {} we construct directly.
    replay = {"meta": {"n_moves": 100, "reason": "win", "compute": {}}}
    out = aggregate_per_game_stats([replay])

    assert out["n_games_total"] == 1
    assert out["n_games_with_any_stats"] == 0  # empty compute does not count
    assert out["coverage"]["compute.leaf_evals"] == 0
    assert out["coverage"]["compute.backups"] == 0
    assert out["coverage"]["compute.nn_batches"] == 0
    assert out["compute_per_game"] is None
```

### Step 2: Run the new tests to verify they fail

Run: `.venv/bin/python -m pytest tests/test_analyzer_per_game_stats.py -v -k "full_coverage or per_field_coverage or compute_subkey"`

Expected: **FAIL** — all three assert that distribution blocks are non-null, but the Task 1 implementation returns `None` for those blocks.

### Step 3: Extend aggregate_per_game_stats with the four distribution blocks

- [ ] In `scripts/twixt_replay_analyzer.py`, locate the `aggregate_per_game_stats` function added in Task 1. Find the comment line `# --- Accumulators for distribution blocks (Tasks 2 and 3 fill these) ---` and replace the surrounding section. The full new accumulator + extraction + computation block:

  Find this section in the function:
  ```python
      # --- Accumulators for distribution blocks (Tasks 2 and 3 fill these) ---
      n_moves_arr = []
      outcomes = {"decisive": 0, "resign": 0, "adjudicated": 0, "timeout": 0, "draw_other": 0}
  ```

  Replace with:
  ```python
      # --- Accumulators for distribution blocks ---
      n_moves_arr = []
      wall_time_arr = []
      final_root_value_arr = []
      final_top1_share_arr = []
      leaf_evals_arr = []
      backups_arr = []
      nn_batches_arr = []
      outcomes = {"decisive": 0, "resign": 0, "adjudicated": 0, "timeout": 0, "draw_other": 0}
  ```

- [ ] Within the per-replay loop, after the outcomes categorization block, add the persistence-era field extraction. Find this section in the function:

  ```python
          else:
              outcomes["draw_other"] += 1
  ```

  Replace with:

  ```python
          else:
              outcomes["draw_other"] += 1

          # Persistence-era field extraction (each field tracked independently).
          # has_any_persistence_stat starts False per replay, set True only when a
          # persistence-era field is actually populated (not just key-present-but-empty,
          # like meta["compute"] == {}). Worker_id case is added in Task 3 — that one
          # treats explicit null as a persistence-era signal (in-process game).
          has_any_persistence_stat = False

          wt = meta.get("wall_time_s")
          if wt is not None:
              coverage["wall_time_s"] += 1
              wall_time_arr.append(float(wt))
              has_any_persistence_stat = True

          frv = meta.get("final_root_value")
          if frv is not None:
              coverage["final_root_value"] += 1
              final_root_value_arr.append(float(frv))
              has_any_persistence_stat = True

          fts = meta.get("final_top1_share")
          if fts is not None:
              coverage["final_top1_share"] += 1
              final_top1_share_arr.append(float(fts))
              has_any_persistence_stat = True

          # meta.compute subkeys are tracked independently — missing != 0
          comp = meta.get("compute") or {}
          le = comp.get("leaf_evals")
          if le is not None:
              coverage["compute.leaf_evals"] += 1
              leaf_evals_arr.append(int(le))
              has_any_persistence_stat = True
          bk = comp.get("backups")
          if bk is not None:
              coverage["compute.backups"] += 1
              backups_arr.append(int(bk))
              has_any_persistence_stat = True
          nb = comp.get("nn_batches")
          if nb is not None:
              coverage["compute.nn_batches"] += 1
              nn_batches_arr.append(int(nb))
              has_any_persistence_stat = True

          if has_any_persistence_stat:
              n_games_with_any_stats += 1
  ```

- [ ] Below the `game_length` computation (already added in Task 1), and ABOVE the `return {...}` statement, add the four distribution blocks computation:

  ```python
      # --- Persistence-era distribution blocks ---
      def _percentiles_block(arr, *, include_total=False, int_minmax=False):
          a = np.asarray(arr, dtype=np.float64)
          block = {
              "mean": float(a.mean()),
              "p50":  float(np.percentile(a, 50)),
              "p90":  float(np.percentile(a, 90)),
              "p95":  float(np.percentile(a, 95)),
              "p99":  float(np.percentile(a, 99)),
              "max":  int(a.max()) if int_minmax else float(a.max()),
              "min":  int(a.min()) if int_minmax else float(a.min()),
          }
          if include_total:
              block["total"] = float(a.sum())
          return block

      wall_time_block = _percentiles_block(wall_time_arr, include_total=True) if wall_time_arr else None

      if final_root_value_arr:
          a = np.asarray(final_root_value_arr, dtype=np.float64)
          final_root_value_block = {
              "mean":     float(a.mean()),
              "p10":      float(np.percentile(a, 10)),
              "p50":      float(np.percentile(a, 50)),
              "p90":      float(np.percentile(a, 90)),
              "abs_mean": float(np.abs(a).mean()),
          }
      else:
          final_root_value_block = None

      if final_top1_share_arr:
          a = np.asarray(final_top1_share_arr, dtype=np.float64)
          final_top1_share_block = {
              "mean": float(a.mean()),
              "p10":  float(np.percentile(a, 10)),
              "p50":  float(np.percentile(a, 50)),
              "p90":  float(np.percentile(a, 90)),
              "min":  float(a.min()),
          }
      else:
          final_top1_share_block = None

      def _compute_subblock(arr):
          if not arr:
              return None
          a = np.asarray(arr, dtype=np.float64)
          return {
              "mean": float(a.mean()),
              "p50":  int(np.percentile(a, 50)),
              "p90":  int(np.percentile(a, 90)),
              "max":  int(a.max()),
          }

      compute_per_game_block = None
      if leaf_evals_arr or backups_arr or nn_batches_arr:
          compute_per_game_block = {
              "leaf_evals": _compute_subblock(leaf_evals_arr),
              "backups":    _compute_subblock(backups_arr),
              "nn_batches": _compute_subblock(nn_batches_arr),
          }
  ```

- [ ] Update the `return {...}` statement to use the new block variables. Replace the previous return with:

  ```python
      return {
          "n_games_total": n_games_total,
          "n_games_with_any_stats": n_games_with_any_stats,
          "coverage": coverage,
          "game_length": game_length,
          "outcomes": outcomes,
          "wall_time_s": wall_time_block,
          "worker_balance": {            # Task 3 fills the body of this
              "by_worker": {},
              "in_process_count": 0,
              "max_min_wall_time_ratio": None,
              "max_min_games_ratio": None,
              "wall_time_cv": None,
          },
          "final_root_value": final_root_value_block,
          "final_top1_share": final_top1_share_block,
          "compute_per_game": compute_per_game_block,
      }
  ```

### Step 4: Run the new tests to verify they pass

Run: `.venv/bin/python -m pytest tests/test_analyzer_per_game_stats.py -v`

Expected: **PASS** — all 8 tests so far (4 from Task 1 + 4 new in Task 2: tests 3, 4, 8, 8b).

### Step 5: Commit

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_per_game_stats.py
git commit -m "feat(analyzer): wall_time, final_root, final_top1, compute distributions

Extends aggregate_per_game_stats with the four persistence-era
distribution blocks. Each block computed only when its coverage > 0;
otherwise null. compute subkeys are tracked independently — missing
subkey is excluded (not zeroed), so per-game compute averages are not
silently depressed by old-schema replays.

Adds tests 3, 4, 8 (full coverage, per-field coverage, missing-subkey).

Spec §6.1 behavior, §7 edge cases.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Worker balance aggregation

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` — extend `aggregate_per_game_stats` with worker_balance computation.
- Test: `tests/test_analyzer_per_game_stats.py` — append tests 5, 6, 7, 11, 12, 13.

### Step 1: Append the six worker tests to the test file

- [ ] Append to `tests/test_analyzer_per_game_stats.py`:

```python
# -------------------------------------------------------------------------
# Test 5: worker_balance groups by worker_id and computes ratios + CV
# -------------------------------------------------------------------------

def test_aggregate_worker_balance_groups_by_worker_id():
    """4 replays from 2 workers with different wall_time_s → all metrics correct."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        # Worker 0: 2 games, total wall_time 30s
        _make_replay(worker_id=0, wall_time_s=10.0, n_moves=100),
        _make_replay(worker_id=0, wall_time_s=20.0, n_moves=120),
        # Worker 1: 2 games, total wall_time 60s
        _make_replay(worker_id=1, wall_time_s=25.0, n_moves=110),
        _make_replay(worker_id=1, wall_time_s=35.0, n_moves=130),
    ]
    out = aggregate_per_game_stats(replays)

    wb = out["worker_balance"]
    assert wb["by_worker"]["0"]["games"] == 2
    assert wb["by_worker"]["0"]["wall_time_total_s"] == 30.0
    assert wb["by_worker"]["0"]["wall_time_mean_s"] == 15.0
    assert wb["by_worker"]["1"]["games"] == 2
    assert wb["by_worker"]["1"]["wall_time_total_s"] == 60.0
    assert wb["by_worker"]["1"]["wall_time_mean_s"] == 30.0
    assert wb["in_process_count"] == 0
    # max/min ratios
    assert wb["max_min_wall_time_ratio"] == 2.0   # 60 / 30
    assert wb["max_min_games_ratio"] == 1.0       # 2 / 2
    # CV: per-worker totals = [30, 60], mean=45, stddev (ddof=0) = sqrt(((30-45)^2 + (60-45)^2)/2) = 15
    # CV = 15/45 = 0.333...
    assert abs(wb["wall_time_cv"] - (15.0 / 45.0)) < 1e-9


# -------------------------------------------------------------------------
# Test 6: per-worker n_moves_total
# -------------------------------------------------------------------------

def test_aggregate_worker_balance_includes_n_moves_per_worker():
    """by_worker[w]["n_moves_total"] is sum of meta.n_moves across that worker's games."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        _make_replay(worker_id=0, n_moves=100, wall_time_s=1.0),
        _make_replay(worker_id=0, n_moves=200, wall_time_s=1.0),
        _make_replay(worker_id=1, n_moves=150, wall_time_s=1.0),
    ]
    out = aggregate_per_game_stats(replays)

    assert out["worker_balance"]["by_worker"]["0"]["n_moves_total"] == 300
    assert out["worker_balance"]["by_worker"]["1"]["n_moves_total"] == 150


# -------------------------------------------------------------------------
# Test 7: in-process games counted separately from worker games
# -------------------------------------------------------------------------

def test_aggregate_in_process_games_counted_separately():
    """worker_id=None (in-process) increments in_process_count, not by_worker.
    worker_id=0 is a legitimate worker key, not conflated with null.
    """
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        _make_replay(worker_id=0,    wall_time_s=1.0),  # worker 0
        _make_replay(worker_id=1,    wall_time_s=1.0),  # worker 1
        _make_replay(worker_id=None, wall_time_s=1.0),  # in-process
        _make_replay(worker_id=None, wall_time_s=1.0),  # in-process
    ]
    out = aggregate_per_game_stats(replays)

    wb = out["worker_balance"]
    assert set(wb["by_worker"].keys()) == {"0", "1"}
    assert "None" not in wb["by_worker"]
    assert wb["in_process_count"] == 2
    # coverage["worker_id"] counts ALL games where the field is present (incl. explicit null)
    assert out["coverage"]["worker_id"] == 4
    # All four games carry persistence-era fields (worker_id present is sufficient,
    # even when explicitly null — that's a meaningful "in-process new schema" signal).
    assert out["n_games_with_any_stats"] == 4


# -------------------------------------------------------------------------
# Test 11: single worker yields null ratios and null CV
# -------------------------------------------------------------------------

def test_aggregate_single_worker_yields_null_ratios():
    """One distinct worker → all three imbalance metrics are None."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        _make_replay(worker_id=0, wall_time_s=10.0),
        _make_replay(worker_id=0, wall_time_s=20.0),
    ]
    out = aggregate_per_game_stats(replays)

    wb = out["worker_balance"]
    assert wb["max_min_wall_time_ratio"] is None
    assert wb["max_min_games_ratio"] is None
    assert wb["wall_time_cv"] is None
    # by_worker still populated
    assert wb["by_worker"]["0"]["games"] == 2


# -------------------------------------------------------------------------
# Test 12: worker with zero wall_time_total excluded from ratio
# -------------------------------------------------------------------------

def test_aggregate_workers_with_zero_wall_time_excluded_from_ratio():
    """Per spec §7: worker with wall_time_total_s == 0 is excluded from
    max_min_wall_time_ratio computation; if fewer than 2 workers remain, ratio is None.
    """
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    # Two workers, but one has wall_time_total_s == 0 (no wall_time_s on its replays)
    replays = [
        _make_replay(worker_id=0, wall_time_s=10.0),
        _make_replay(worker_id=0, wall_time_s=20.0),
        _make_replay(worker_id=1),  # no wall_time_s
    ]
    out = aggregate_per_game_stats(replays)

    wb = out["worker_balance"]
    # by_worker still has both, but worker 1's wall_time_total_s == 0
    assert wb["by_worker"]["0"]["wall_time_total_s"] == 30.0
    assert wb["by_worker"]["1"]["wall_time_total_s"] == 0.0
    # Only 1 worker remains after excluding zero-time worker → ratio is None
    assert wb["max_min_wall_time_ratio"] is None
    assert wb["wall_time_cv"] is None
    # max_min_games_ratio is well-defined (1 game vs 2 games)
    assert wb["max_min_games_ratio"] == 2.0


# -------------------------------------------------------------------------
# Test 13: uniform per-worker wall_time yields ratio=1.0, cv=0.0
# -------------------------------------------------------------------------

def test_aggregate_uniform_per_worker_wall_time_yields_unity_ratio_zero_cv():
    """All per-worker wall_time_total equal → max_min_ratio=1.0, cv=0.0."""
    from scripts.twixt_replay_analyzer import aggregate_per_game_stats

    replays = [
        _make_replay(worker_id=0, wall_time_s=10.0),
        _make_replay(worker_id=1, wall_time_s=10.0),
        _make_replay(worker_id=2, wall_time_s=10.0),
    ]
    out = aggregate_per_game_stats(replays)

    wb = out["worker_balance"]
    assert wb["max_min_wall_time_ratio"] == 1.0
    assert wb["wall_time_cv"] == 0.0
    assert wb["max_min_games_ratio"] == 1.0
```

### Step 2: Run the new tests to verify they fail

Run: `.venv/bin/python -m pytest tests/test_analyzer_per_game_stats.py -v -k "worker"`

Expected: **FAIL** — all 6 worker tests fail because `worker_balance.by_worker` is hardcoded to `{}` in the current implementation.

### Step 3: Extend aggregate_per_game_stats with worker_balance computation

- [ ] In `scripts/twixt_replay_analyzer.py`, locate the section in `aggregate_per_game_stats` that extracts persistence-era fields (added in Task 2). The block currently ends with the nn_batches extraction immediately followed by the `if has_any_persistence_stat: n_games_with_any_stats += 1` increment. Find this section:

  ```python
          nb = comp.get("nn_batches")
          if nb is not None:
              coverage["compute.nn_batches"] += 1
              nn_batches_arr.append(int(nb))
              has_any_persistence_stat = True

          if has_any_persistence_stat:
              n_games_with_any_stats += 1
  ```

  Insert the worker accumulation block BETWEEN the nn_batches block and the increment, so worker-id presence (including explicit null) is reflected in `n_games_with_any_stats`:

  ```python
          nb = comp.get("nn_batches")
          if nb is not None:
              coverage["compute.nn_batches"] += 1
              nn_batches_arr.append(int(nb))
              has_any_persistence_stat = True

          # worker_id: integer → bucket; explicit null → in-process; absent → not counted.
          # Explicit null IS a persistence-era signal (in-process game in new schema),
          # so we set has_any_persistence_stat regardless of whether wid is None or int.
          if "worker_id" in meta:
              coverage["worker_id"] += 1
              has_any_persistence_stat = True
              wid = meta["worker_id"]
              if wid is None:
                  in_process_count += 1
              else:
                  key = str(int(wid))
                  bucket = by_worker.setdefault(key, {
                      "games": 0,
                      "n_moves_total": 0,
                      "wall_time_total_s": 0.0,
                  })
                  bucket["games"] += 1
                  if n_moves is not None:
                      bucket["n_moves_total"] += int(n_moves)
                  if wt is not None:
                      bucket["wall_time_total_s"] += float(wt)

          if has_any_persistence_stat:
              n_games_with_any_stats += 1
  ```

- [ ] Above the `# --- Accumulators for distribution blocks ---` block, add the worker accumulators initialization. Find this section:

  ```python
      # --- Coverage counters (always present in output) ---
      coverage = {
          ...
      }

      # --- Accumulators for distribution blocks ---
  ```

  Replace with (inserting two new accumulators before the existing accumulator block):

  ```python
      # --- Coverage counters (always present in output) ---
      coverage = {
          "wall_time_s": 0,
          "worker_id": 0,
          "final_root_value": 0,
          "final_top1_share": 0,
          "compute.leaf_evals": 0,
          "compute.backups": 0,
          "compute.nn_batches": 0,
          "n_moves": 0,
          "reason": 0,
      }

      # --- Worker balance accumulators ---
      by_worker = {}              # str(worker_id) → {games, n_moves_total, wall_time_total_s}
      in_process_count = 0

      # --- Accumulators for distribution blocks ---
  ```

- [ ] Below the `compute_per_game_block` computation (added in Task 2), and ABOVE the `return {...}` statement, add the worker_balance derived metrics:

  ```python
      # --- Worker balance derived metrics ---
      # Add wall_time_mean_s to each bucket
      for bucket in by_worker.values():
          if bucket["games"] > 0 and bucket["wall_time_total_s"] > 0:
              bucket["wall_time_mean_s"] = bucket["wall_time_total_s"] / bucket["games"]
          else:
              bucket["wall_time_mean_s"] = 0.0

      # max/min games ratio: needs >= 2 distinct workers
      if len(by_worker) >= 2:
          games_list = [b["games"] for b in by_worker.values()]
          gmax = max(games_list)
          gmin = min(games_list)
          max_min_games_ratio = float(gmax) / float(gmin) if gmin > 0 else None
      else:
          max_min_games_ratio = None

      # max/min wall-time ratio + CV: only over workers with wall_time_total_s > 0,
      # and only when >= 2 such workers remain.
      timed_workers = [b for b in by_worker.values() if b["wall_time_total_s"] > 0]
      if len(timed_workers) >= 2:
          wt_totals = np.asarray([b["wall_time_total_s"] for b in timed_workers], dtype=np.float64)
          wt_max = float(wt_totals.max())
          wt_min = float(wt_totals.min())
          max_min_wall_time_ratio = wt_max / wt_min  # wt_min > 0 by filter above
          wt_mean = float(wt_totals.mean())
          if wt_mean > 0:
              wall_time_cv = float(wt_totals.std(ddof=0)) / wt_mean
          else:
              wall_time_cv = None
      else:
          max_min_wall_time_ratio = None
          wall_time_cv = None
  ```

- [ ] Update the `return {...}` statement to use the new worker_balance values. Replace:

  ```python
          "worker_balance": {            # Task 3 fills the body of this
              "by_worker": {},
              "in_process_count": 0,
              "max_min_wall_time_ratio": None,
              "max_min_games_ratio": None,
              "wall_time_cv": None,
          },
  ```

  With:

  ```python
          "worker_balance": {
              "by_worker": by_worker,
              "in_process_count": in_process_count,
              "max_min_wall_time_ratio": max_min_wall_time_ratio,
              "max_min_games_ratio": max_min_games_ratio,
              "wall_time_cv": wall_time_cv,
          },
  ```

### Step 4: Run the worker tests to verify they pass

Run: `.venv/bin/python -m pytest tests/test_analyzer_per_game_stats.py -v`

Expected: **PASS** — all 14 tests so far (4 from Task 1 + 4 from Task 2 + 6 worker tests).

### Step 5: Commit

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_per_game_stats.py
git commit -m "feat(analyzer): worker balance aggregation in aggregate_per_game_stats

Extends aggregate_per_game_stats with the worker_balance block:
by_worker (keyed by str(worker_id) with games / n_moves_total /
wall_time_total_s / wall_time_mean_s), in_process_count for null
worker_id, plus three imbalance metrics: max_min_wall_time_ratio,
max_min_games_ratio, wall_time_cv (coefficient of variation).

Worker identity is taken solely from meta.worker_id — never inferred
from filename or sidecar. Workers with wall_time_total_s == 0 are
excluded from the wall-time ratio/CV (defended in test 12).

Adds tests 5, 6, 7, 11, 12, 13.

Spec §6.1 worker block, §7 edge cases.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Format function for report.txt

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` — add `format_per_game_stats_report(per_game_stats)` near other `format_*_report` functions (~line 948), plus a small helper `_format_duration_human(seconds)`.
- Test: `tests/test_analyzer_per_game_stats.py` — append tests 14–20.

### Step 1: Append the seven format tests

- [ ] Append to `tests/test_analyzer_per_game_stats.py`:

```python
# -------------------------------------------------------------------------
# Test 14: zero-coverage short message
# -------------------------------------------------------------------------

def test_format_renders_zero_coverage_short_message():
    """n_games_with_any_stats == 0 → game_length and outcomes lines render,
    then the short 'no games carry new persistence fields' message.
    """
    from scripts.twixt_replay_analyzer import (
        aggregate_per_game_stats, format_per_game_stats_report,
    )

    # Old-schema-only replays
    replays = [_make_replay(n_moves=100, reason="win") for _ in range(3)]
    pgs = aggregate_per_game_stats(replays)
    lines = format_per_game_stats_report(pgs)
    text = "\n".join(lines)

    # game_length and outcomes are still rendered (they use pre-existing fields)
    assert "Game length:" in text
    assert "Outcomes:" in text
    # The short fallback message
    assert "no games carry new persistence fields" in text
    # No persistence-era stat lines
    assert "Wall time:" not in text
    assert "Workers:" not in text
    assert "Final root:" not in text
    assert "Final top1:" not in text
    assert "Compute/game:" not in text


# -------------------------------------------------------------------------
# Test 15: full block rendering with uniform coverage suppresses Coverage line
# -------------------------------------------------------------------------

def test_format_renders_full_block():
    """Fully-populated per_game_stats → expected lines in expected order.
    Uniform coverage → no separate 'Coverage:' line printed.
    """
    from scripts.twixt_replay_analyzer import (
        aggregate_per_game_stats, format_per_game_stats_report,
    )

    replays = [
        _make_replay(worker_id=0, wall_time_s=10.0, final_root_value=0.5,
                     final_top1_share=0.4, leaf_evals=1000, backups=2000,
                     nn_batches=50, n_moves=100),
        _make_replay(worker_id=1, wall_time_s=20.0, final_root_value=-0.3,
                     final_top1_share=0.6, leaf_evals=1500, backups=2500,
                     nn_batches=80, n_moves=120),
    ]
    pgs = aggregate_per_game_stats(replays)
    lines = format_per_game_stats_report(pgs)
    text = "\n".join(lines)

    # Headers and stat lines all present
    assert "Per-game stats" in text
    assert "Game length:" in text
    assert "Outcomes:" in text
    assert "Wall time:" in text
    assert "Workers:" in text
    assert "Final root:" in text
    assert "Final top1:" in text
    assert "Compute/game:" in text
    # Uniform coverage → no Coverage: line
    assert "Coverage:" not in text


# -------------------------------------------------------------------------
# Test 16: partial coverage prints the Coverage: line
# -------------------------------------------------------------------------

def test_format_renders_coverage_line_on_partial_coverage():
    """Non-uniform per-field coverage → Coverage: line is printed."""
    from scripts.twixt_replay_analyzer import (
        aggregate_per_game_stats, format_per_game_stats_report,
    )

    # 3 replays: all have wall_time_s, only 2 have final_top1_share
    replays = [
        _make_replay(worker_id=0, wall_time_s=10.0, final_top1_share=0.5),
        _make_replay(worker_id=0, wall_time_s=15.0, final_top1_share=0.6),
        _make_replay(worker_id=0, wall_time_s=20.0),  # no final_top1_share
    ]
    pgs = aggregate_per_game_stats(replays)
    lines = format_per_game_stats_report(pgs)
    text = "\n".join(lines)

    assert "Coverage:" in text


# -------------------------------------------------------------------------
# Test 17: zero-coverage field is omitted entirely (no "n/a" line)
# -------------------------------------------------------------------------

def test_format_omits_lines_for_zero_coverage_fields():
    """When a field has zero coverage, omit its line entirely."""
    from scripts.twixt_replay_analyzer import (
        aggregate_per_game_stats, format_per_game_stats_report,
    )

    # Replays carry wall_time_s but not final_top1_share
    replays = [
        _make_replay(worker_id=0, wall_time_s=10.0),
        _make_replay(worker_id=0, wall_time_s=20.0),
    ]
    pgs = aggregate_per_game_stats(replays)
    lines = format_per_game_stats_report(pgs)
    text = "\n".join(lines)

    assert "Wall time:" in text
    assert "Final top1:" not in text
    # Specifically: we never render "Final top1: n/a" (we omit the whole line instead).
    # Generic "n/a" might appear in unrelated future text, so be precise.
    assert "Final top1: n/a" not in text
    assert "Final root: n/a" not in text


# -------------------------------------------------------------------------
# Test 18: single worker line shape
# -------------------------------------------------------------------------

def test_format_handles_single_worker():
    """One distinct worker → 'Workers: 1 active; games=N; wall-time mean=Xs (in-process: M)'."""
    from scripts.twixt_replay_analyzer import (
        aggregate_per_game_stats, format_per_game_stats_report,
    )

    replays = [
        _make_replay(worker_id=0, wall_time_s=10.0),
        _make_replay(worker_id=0, wall_time_s=20.0),
    ]
    pgs = aggregate_per_game_stats(replays)
    lines = format_per_game_stats_report(pgs)
    text = "\n".join(lines)

    assert "1 active" in text
    assert "ratio" not in text   # no ratios printed when only 1 worker
    assert "cv=" not in text     # no CV either


# -------------------------------------------------------------------------
# Test 19: in-process only (worker_id all null)
# -------------------------------------------------------------------------

def test_format_handles_in_process_only():
    """All worker_id == null → 'Workers: 0 active; in-process: N'."""
    from scripts.twixt_replay_analyzer import (
        aggregate_per_game_stats, format_per_game_stats_report,
    )

    replays = [
        _make_replay(worker_id=None, wall_time_s=10.0),
        _make_replay(worker_id=None, wall_time_s=20.0),
        _make_replay(worker_id=None, wall_time_s=30.0),
    ]
    pgs = aggregate_per_game_stats(replays)
    lines = format_per_game_stats_report(pgs)
    text = "\n".join(lines)

    assert "0 active" in text
    assert "in-process: 3" in text


# -------------------------------------------------------------------------
# Test 20: human-readable duration formatting
# -------------------------------------------------------------------------

def test_format_human_readable_duration():
    """_format_duration_human handles the three cases per spec §5.2."""
    from scripts.twixt_replay_analyzer import _format_duration_human

    # < 60s → 'X.Xs'
    assert _format_duration_human(0.0) == "0.0s"
    assert _format_duration_human(30.0) == "30.0s"
    assert _format_duration_human(59.9) == "59.9s"
    # < 1h → 'Xm Ys'
    assert _format_duration_human(60.0)  == "1m 0s"
    assert _format_duration_human(145.0) == "2m 25s"
    assert _format_duration_human(3599.0) == "59m 59s"
    # >= 1h → 'XhYm'
    assert _format_duration_human(3600.0)   == "1h0m"
    assert _format_duration_human(17852.4)  == "4h57m"
```

### Step 2: Run the format tests to verify they fail

Run: `.venv/bin/python -m pytest tests/test_analyzer_per_game_stats.py -v -k "format"`

Expected: **FAIL** — `ImportError` for `format_per_game_stats_report` and `_format_duration_human`.

### Step 3: Add _format_duration_human helper and format_per_game_stats_report

- [ ] In `scripts/twixt_replay_analyzer.py`, locate any of the existing `format_*_report` functions (e.g., `format_replay_cap_report` at line 761, or `format_connectivity_diagnostics_report` at line 948). Just **above** one of them (place near `format_connectivity_diagnostics_report` at ~line 948 for proximity to other report formatters), add the helper and the format function:

```python
def _format_duration_human(seconds: float) -> str:
    """Render a duration in seconds as a human-readable string per spec §5.2."""
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        m = int(seconds // 60)
        s = int(round(seconds - m * 60))
        return f"{m}m {s}s"
    h = int(seconds // 3600)
    m = int((seconds - h * 3600) // 60)
    return f"{h}h{m}m"


def format_per_game_stats_report(per_game_stats: dict) -> List[str]:
    """Render the per-game stats block as report.txt lines.

    - Suppresses lines for fields with zero coverage (per spec §5.1).
    - Suppresses the per-field 'Coverage:' line when coverage is uniform
      across all persistence-era fields.
    - Falls back to a single short message when n_games_with_any_stats == 0.

    Spec: docs/superpowers/specs/2026-04-29-analyzer-per-game-stats-design.md §5
    """
    lines: List[str] = []
    pgs = per_game_stats
    n_total = pgs.get("n_games_total", 0)
    n_with = pgs.get("n_games_with_any_stats", 0)

    # Header
    if n_total == 0:
        lines.append("Per-game stats: no replays loaded.")
        lines.append("")
        return lines
    lines.append(f"Per-game stats (n={n_with} / {n_total} games carry new fields):")

    # game_length (always rendered when n_total > 0 and coverage > 0)
    gl = pgs.get("game_length")
    if gl is not None:
        lines.append(
            f"  Game length:  mean={gl['mean']:.1f} p50={int(gl['p50'])} "
            f"p90={int(gl['p90'])} p95={int(gl['p95'])} max={int(gl['max'])}"
        )

    # outcomes (always rendered when n_total > 0)
    o = pgs["outcomes"]
    lines.append(
        f"  Outcomes:     decisive={o['decisive']} resign={o['resign']} "
        f"adjudicated={o['adjudicated']} timeout={o['timeout']} draw_other={o['draw_other']}"
    )

    # Short fallback when no persistence-era data
    if n_with == 0:
        lines.append("Per-game stats: no games carry new persistence fields (all replays predate persistence change).")
        lines.append("")
        return lines

    # Wall time
    wt = pgs.get("wall_time_s")
    if wt is not None:
        lines.append(
            f"  Wall time:    mean={wt['mean']:.1f}s p50={wt['p50']:.1f}s "
            f"p90={wt['p90']:.1f}s p95={wt['p95']:.1f}s max={wt['max']:.1f}s "
            f"(total={_format_duration_human(wt['total'])})"
        )

    # Workers
    wb = pgs["worker_balance"]
    n_workers = len(wb["by_worker"])
    in_proc = wb["in_process_count"]
    if n_workers == 0:
        lines.append(f"  Workers:      0 active; in-process: {in_proc}")
    elif n_workers == 1:
        # Single-worker line: no ratios
        only_w = next(iter(wb["by_worker"].values()))
        wt_mean_str = f"; wall-time mean={only_w['wall_time_mean_s']:.1f}s" if only_w['wall_time_mean_s'] > 0 else ""
        lines.append(f"  Workers:      1 active; games={only_w['games']}{wt_mean_str} (in-process: {in_proc})")
    else:
        # Multi-worker line: include ratios + CV when defined
        games_list = [b["games"] for b in wb["by_worker"].values()]
        gmin = min(games_list); gmax = max(games_list)
        parts = [f"  Workers:      {n_workers} active",
                 f"games min/max={gmin}/{gmax}"]
        if wb["max_min_games_ratio"] is not None:
            parts.append(f"ratio={wb['max_min_games_ratio']:.2f}")
        if wb["max_min_wall_time_ratio"] is not None:
            parts.append(f"wall-time ratio={wb['max_min_wall_time_ratio']:.2f}")
        if wb["wall_time_cv"] is not None:
            parts.append(f"cv={wb['wall_time_cv']:.2f}")
        line = parts[0] + "; " + "; ".join(parts[1:]) + f" (in-process: {in_proc})"
        lines.append(line)

    # Final root
    frv = pgs.get("final_root_value")
    if frv is not None:
        lines.append(
            f"  Final root:   mean={frv['mean']:.2f} p50={frv['p50']:.2f} "
            f"p10={frv['p10']:.2f} p90={frv['p90']:.2f} (|abs| mean={frv['abs_mean']:.2f})"
        )

    # Final top1
    fts = pgs.get("final_top1_share")
    if fts is not None:
        lines.append(
            f"  Final top1:   mean={fts['mean']:.2f} p50={fts['p50']:.2f} "
            f"p10={fts['p10']:.2f} p90={fts['p90']:.2f} min={fts['min']:.2f}"
        )

    # Compute per game (each subkey rendered only when non-null)
    cpg = pgs.get("compute_per_game")
    if cpg is not None:
        sub_parts = []
        for key in ("leaf_evals", "backups", "nn_batches"):
            sub = cpg.get(key)
            if sub is not None:
                sub_parts.append(f"{key} p50={int(sub['p50'])} p90={int(sub['p90'])} max={int(sub['max'])}")
        if sub_parts:
            lines.append(f"  Compute/game: " + " | ".join(sub_parts))

    # Coverage line: print only when coverage is non-uniform across persistence-era fields
    cov = pgs["coverage"]
    persistence_keys = ("wall_time_s", "worker_id", "final_root_value", "final_top1_share",
                        "compute.leaf_evals", "compute.backups", "compute.nn_batches")
    cov_values = [cov[k] for k in persistence_keys]
    if len(set(cov_values)) > 1:
        # Format compactly
        compute_subs = ", ".join(
            f"{k.split('.')[1]}={cov[k]}" for k in
            ("compute.leaf_evals", "compute.backups", "compute.nn_batches")
        )
        lines.append(
            f"  Coverage:     wall_time_s={cov['wall_time_s']} worker_id={cov['worker_id']} "
            f"final_root_value={cov['final_root_value']} final_top1_share={cov['final_top1_share']} "
            f"compute={{{compute_subs}}}"
        )

    lines.append("")
    return lines
```

### Step 4: Run the format tests to verify they pass

Run: `.venv/bin/python -m pytest tests/test_analyzer_per_game_stats.py -v`

Expected: **PASS** — all 21 tests now pass (4 + 4 + 6 + 7 = tests 1, 2, 9, 10 + 3, 4, 8, 8b + 5, 6, 7, 11, 12, 13 + 14–20).

### Step 5: Run the existing analyzer regression to confirm no breakage

Run: `.venv/bin/python -m pytest tests/test_analyzer_phase2_sidecar_fields.py tests/test_analyzer_phase2_smoke.py tests/test_analyzer_replay_probe_scoring_end_to_end.py tests/test_strong_advantage_analyzer_aggregation.py -v`

Expected: All pass.

### Step 6: Commit

```bash
git add scripts/twixt_replay_analyzer.py tests/test_analyzer_per_game_stats.py
git commit -m "feat(analyzer): format_per_game_stats_report renders triage section

Adds format_per_game_stats_report() and the _format_duration_human()
helper. Renders a compact triage section for report.txt: header with
coverage ratio, then game length, outcomes, wall time, workers (with
3-metric imbalance line), final root, final top1, compute/game, and
optionally a Coverage: line when per-field coverage is non-uniform.

Suppresses lines for zero-coverage fields (no 'n/a' filler). Single-
worker case omits ratios. In-process-only case shows 'Workers: 0 active'.

Adds tests 14-20.

Spec §5 rendering, §5.1 partial-coverage rules, §5.2 number formatting.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Integration into summary.json + report.txt builder

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py` — two integration call sites: one in the summary dict literal (~line 1864), one in the report-text builder (~line 2180).

### Step 1: Read the integration sites for accurate placement

- [ ] `Read scripts/twixt_replay_analyzer.py:1860-1880` to confirm the `summary = { ... }` dict literal location and the surrounding context.
- [ ] `Read scripts/twixt_replay_analyzer.py:2175-2195` to confirm the `Compute:` line location and what follows it.

### Step 2: Add the aggregate call before the summary dict literal

- [ ] In `scripts/twixt_replay_analyzer.py`, just **before** the line `summary = {` (~line 1864), add:

```python
    # Per-game stats persistence surfacing (spec 2026-04-29).
    per_game_stats_val = aggregate_per_game_stats(replays)
```

### Step 3: Add per_game_stats to the summary dict literal

- [ ] In the `summary = { ... }` dict literal, find the line:

```python
        "compute": compute_val,
```

Add immediately after it:

```python
        "compute": compute_val,
        # Per-game stats persistence surfacing (spec 2026-04-29).
        # Distributions complement the sidecar-derived `compute` totals above.
        "per_game_stats": per_game_stats_val,
```

### Step 4: Add the format call to the report builder

- [ ] In `scripts/twixt_replay_analyzer.py`, locate the existing `Compute:` rendering line (~line 2180):

```python
        comp = summary["compute"]
        lines.append(f"Compute: buffer_size={comp['buffer_size']}, backups={comp['backups']}, leaf_evals={comp['leaf_evals']}, nn_batches={comp['nn_batches']}")
        lines.append("")
```

Add immediately after the blank-line append:

```python
        comp = summary["compute"]
        lines.append(f"Compute: buffer_size={comp['buffer_size']}, backups={comp['backups']}, leaf_evals={comp['leaf_evals']}, nn_batches={comp['nn_batches']}")
        lines.append("")

        # Per-game stats triage section (spec 2026-04-29).
        lines.extend(format_per_game_stats_report(summary["per_game_stats"]))
```

### Step 5: Manual end-to-end smoke test against the test_game_saver_per_game_fields fixtures

- [ ] Run the analyzer's existing tests + the new analyzer per-game-stats tests together:

```bash
.venv/bin/python -m pytest tests/test_analyzer_per_game_stats.py tests/test_analyzer_phase2_sidecar_fields.py tests/test_analyzer_phase2_smoke.py tests/test_analyzer_replay_probe_scoring_end_to_end.py tests/test_strong_advantage_analyzer_aggregation.py -v
```

Expected: All pass (20 new + the four existing analyzer test files).

### Step 6: Lightweight synthetic-replay end-to-end smoke

This is the primary integration smoke: lighter than spinning up MLX self-play, environment-independent, and exercises the same code path (load_replays → aggregate_per_game_stats → format_per_game_stats_report).

- [ ] Create `/tmp/analyzer_smoke.py` with this content:

```python
"""Lightweight analyzer smoke: write one synthetic new-schema replay JSON to
a temp dir, then call load_replays + aggregate + format directly."""
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path("/Users/bill/Desktop/TwixT_Game")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.twixt_replay_analyzer import (
    load_replays, aggregate_per_game_stats, format_per_game_stats_report,
)

# A minimal new-schema replay record matching what game_saver writes.
SYNTHETIC = {
    "id": "iter_0000_game_000",
    "timestamp": "2026-04-29T00:00:00+00:00",
    "config_hash": "alphazero",
    "depth": 200,
    "seed": 0,
    "winner": "red",
    "starting_player": "red",
    "moves": [
        {"turn": 1, "player": "red",   "row": 0, "col": 1, "bridges_created": [], "heuristics": {}, "search_score": None},
        {"turn": 2, "player": "black", "row": 1, "col": 0, "bridges_created": [], "heuristics": {}, "search_score": None},
        {"turn": 3, "player": "red",   "row": 2, "col": 2, "bridges_created": [], "heuristics": {}, "search_score": None},
    ],
    "meta": {
        "board_size": 24,
        "mode": "alphazero",
        "reason": "win",
        "iteration": 0,
        "game_idx": 0,
        "simulations": 200,
        "n_moves": 3,
        "starting_player": "red",
        "worker_id": 2,
        "wall_time_s": 14.27,
        "adjudication_block_reason": None,
        "final_root_value": 0.83,
        "final_top1_share": 0.62,
        "compute": {"leaf_evals": 17400, "backups": 17400, "nn_batches": 850},
    },
}

with tempfile.TemporaryDirectory(prefix="analyzer_smoke_") as tmp:
    tmp_dir = Path(tmp)
    (tmp_dir / "iter_0000_game_000.json").write_text(json.dumps(SYNTHETIC))
    replays = load_replays([str(tmp_dir)])
    pgs = aggregate_per_game_stats(replays)
    print("=== per_game_stats ===")
    print(json.dumps(pgs, indent=2))
    print()
    print("=== format_per_game_stats_report output ===")
    for line in format_per_game_stats_report(pgs):
        print(line)
```

Run: `.venv/bin/python /tmp/analyzer_smoke.py`

Expected:
- `per_game_stats` block printed with `n_games_total == 1`, `n_games_with_any_stats == 1`, every persistence-era coverage entry == 1, all distribution blocks non-null. `worker_balance.by_worker == {"2": {"games": 1, "n_moves_total": 3, "wall_time_total_s": 14.27, "wall_time_mean_s": 14.27}}` and `in_process_count == 0`. Single-worker → ratios all None.
- Format report shows: Game length, Outcomes (decisive=1), Wall time (mean=14.3s, total=14.3s), Workers (1 active, no ratios), Final root (mean=0.83), Final top1 (mean=0.62), Compute/game (leaf_evals p50=17400, etc.).

### Step 7: (Optional) Production-save-path smoke

If the local environment has MLX/Metal available and the upstream persistence pipeline is healthy, also run an end-to-end smoke that exercises `play_game → _save_game_from_record → analyzer`. Skip this step if MLX import fails or if the local environment is otherwise constrained — Step 6 already verifies the analyzer side of the integration.

- [ ] (Optional) Create `/tmp/analyzer_prod_smoke.py`:

```python
"""Production-path analyzer smoke (requires MLX). Spins up one short
real self-play game, saves via the production trainer helper, runs the
analyzer over it. Use only when MLX/Metal is healthy locally."""
import json, random, sys, tempfile
from pathlib import Path

PROJECT_ROOT = Path("/Users/bill/Desktop/TwixT_Game")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import mlx.core as mx
from scripts.GPU.alphazero.mcts import MCTSConfig
from scripts.GPU.alphazero.network import create_network
from scripts.GPU.alphazero.local_evaluator import LocalGPUEvaluator
from scripts.GPU.alphazero.self_play import play_game
from scripts.GPU.alphazero.game_saver import GameSaver
from scripts.GPU.alphazero.trainer import _save_game_from_record
from scripts.twixt_replay_analyzer import (
    load_replays, aggregate_per_game_stats, format_per_game_stats_report,
)

np.random.seed(7); mx.random.seed(7)
net = create_network(hidden=64, n_blocks=2)
ev = LocalGPUEvaluator(net)
cfg = MCTSConfig(n_simulations=40)
game = play_game(evaluator=ev, mcts_config=cfg, rng=random.Random(7),
                 max_moves=30, active_size=11, start_player="red")

with tempfile.TemporaryDirectory(prefix="analyzer_prod_smoke_") as tmp:
    tmp_dir = Path(tmp)
    saver = GameSaver(games_dir=tmp_dir, max_games_per_iter=1, simulations=40, active_size=11)
    saver.set_iteration(0)
    _save_game_from_record(saver, game)
    replays = load_replays([str(tmp_dir)])
    pgs = aggregate_per_game_stats(replays)
    print("=== per_game_stats ===")
    print(json.dumps(pgs, indent=2))
    print()
    print("=== format_per_game_stats_report output ===")
    for line in format_per_game_stats_report(pgs):
        print(line)
```

Run (only if environment supports it): `.venv/bin/python /tmp/analyzer_prod_smoke.py`

Expected: similar shape to Step 6 but with `worker_balance.in_process_count == 1` (in-process path) and real measured wall_time / final_root / final_top1 values.

### Step 8: Cleanup the smoke scripts

- [ ] Run: `rm -f /tmp/analyzer_smoke.py /tmp/analyzer_prod_smoke.py`

### Step 9: Commit

```bash
git add scripts/twixt_replay_analyzer.py
git commit -m "feat(analyzer): wire per_game_stats into summary.json and report.txt

Adds two call sites: aggregate_per_game_stats(replays) emits the new
per_game_stats top-level block in summary.json (sibling of compute),
and format_per_game_stats_report(...) appends the triage section to
report.txt right after the existing Compute: line.

End-to-end verified by manual smoke (1 new-schema game saved via the
production trainer helper → analyzer reads it → per_game_stats block
correct, format renders the full section). Existing analyzer regression
suite still passes.

Spec §6.3 integration points.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

After all five tasks committed:

```bash
# 1. New + existing analyzer tests
.venv/bin/python -m pytest tests/test_analyzer_per_game_stats.py tests/test_analyzer_phase2_sidecar_fields.py tests/test_analyzer_phase2_smoke.py tests/test_analyzer_replay_probe_scoring_end_to_end.py tests/test_strong_advantage_analyzer_aggregation.py -v

# 2. Phase 1 candidate-mining regression (consumer of game JSONs)
.venv/bin/python -m pytest tests/test_strong_advantage_probe_suite.py -v

# 3. Manual analyzer run over the saved game JSONs (if any exist post-persistence-change)
.venv/bin/python scripts/twixt_replay_analyzer.py scripts/GPU/logs/games --out /tmp/analyzer_full_smoke
cat /tmp/analyzer_full_smoke/summary.json | python -m json.tool | grep -A 40 per_game_stats
grep -A 8 "Per-game stats" /tmp/analyzer_full_smoke/report.txt
```

Expected: steps 1–2 green; step 3 shows the `per_game_stats` block with non-zero `n_games_with_any_stats` (assuming any post-persistence games are on disk) and the new section visible in `report.txt`.

---

## Self-review

- **Spec coverage:**
  - §4 schema → Task 1 (foundation), Task 2 (distributions), Task 3 (worker_balance) — all blocks covered.
  - §4.1 contracts → tests 1, 2, 3, 4, 8, 8b, 11, 12, 13 cover the type/null contracts.
  - §5 report rendering → Task 4 + tests 14–20.
  - §5.1 partial coverage → tests 14, 16, 17, 18, 19.
  - §5.2 number formatting → tests 15 (mean/percentile rendering), 20 (duration helper).
  - §6.1 aggregate behavior → Tasks 1–3.
  - §6.2 format function → Task 4.
  - §6.3 integration call sites → Task 5.
  - §7 edge cases → tests 1, 2, 7, 8, 8b, 11, 12, 13 (every row in the edge-case table has a test or is structurally implied by the implementation).
  - §9 test plan → tests 1–20 implemented; bonus test 8b (empty meta.compute = {} sanity); optional test 21 deferred per spec §10.
- **Placeholder scan:** no TBD/TODO/etc. Every step has concrete code.
- **Type consistency:** `aggregate_per_game_stats(replays) -> dict`, `format_per_game_stats_report(per_game_stats) -> List[str]`, `_format_duration_human(seconds) -> str` — names consistent across all tasks and tests. `_make_replay` test helper consistent across all tests. `has_any_persistence_stat` is the canonical per-replay flag set in Tasks 2 (per-stat extraction) and Task 3 (worker_id extraction); the `if has_any_persistence_stat: n_games_with_any_stats += 1` increment lives at the END of the per-replay loop body so worker_id presence (including explicit null) is reflected.
