# Analyzer Opening Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the replay analyzer to read, validate, aggregate, and export opening penalty diagnostics from both per-game JSON files and per-iteration sidecar files.

**Architecture:** A new helper module (`scripts/opening_diagnostics_analyzer.py`) handles all opening diagnostics processing. The main analyzer imports from it and calls into it at well-defined integration points. The helper tries to import the canonical aggregation from `scripts/GPU/alphazero/opening_diagnostics.py` (single source of truth) with a graceful fallback if the import fails.

**Tech Stack:** Python, csv, json (standard library only in the helper)

---

## File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `scripts/opening_diagnostics_analyzer.py` | **NEW** — Extraction, validation, aggregation, CSV writing, report formatting | Create |
| `scripts/twixt_replay_analyzer.py` | Integration: call helper, add to summary/CSV/report | Modify (~30 lines) |

---

### Task 1: Create helper module skeleton with canonical import

**Files:**
- Create: `scripts/opening_diagnostics_analyzer.py`

- [ ] **Step 1: Create the module with imports and fallback**

```python
"""Opening penalty diagnostics — analyzer helpers.

Reads opening_diagnostics from per-game JSON and opening_penalty_diagnostics
from per-iteration sidecar files. Validates, aggregates, and exports to
CSV and report text.

Import strategy: tries to reuse canonical aggregation from the training
module (single source of truth, zero drift). Falls back to a local
implementation if the training module is unavailable.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# --- Canonical import with fallback ---
# Try to reuse the training module's aggregation (single source of truth).
# Two import paths: depends on how the analyzer is launched.
_CANONICAL_AGGREGATE = False
_canonical_aggregate = None
try:
    import sys as _sys
    _scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if _scripts_dir not in _sys.path:
        _sys.path.insert(0, _scripts_dir)
    from GPU.alphazero.opening_diagnostics import (
        aggregate_opening_diagnostics as _canonical_aggregate,
    )
    _CANONICAL_AGGREGATE = True
except ImportError:
    try:
        from scripts.GPU.alphazero.opening_diagnostics import (
            aggregate_opening_diagnostics as _canonical_aggregate,
        )
        _CANONICAL_AGGREGATE = True
    except ImportError:
        _CANONICAL_AGGREGATE = False
        _canonical_aggregate = None


REGIONS = ("near_corner", "edge_band", "interior")
VALID_REGIONS = {"near_corner", "edge_band", "interior"}
MASS_TOLERANCE = 0.05
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile scripts/opening_diagnostics_analyzer.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/opening_diagnostics_analyzer.py
git commit -m "feat: create opening_diagnostics_analyzer module skeleton"
```

---

### Task 2: Per-game extraction

**Files:**
- Modify: `scripts/opening_diagnostics_analyzer.py`

- [ ] **Step 1: Add extraction function**

```python
def extract_per_game_diagnostics(replays: List[dict]) -> Tuple[List[dict], List[List[dict]]]:
    """Extract opening_diagnostics from per-game replay JSONs.

    Args:
        replays: List of loaded replay dicts (from load_replays)

    Returns:
        (per_game_records, all_game_diag_lists)
        per_game_records: List of dicts with source, iteration, game_id, meta, diagnostics
        all_game_diag_lists: List of per-game diagnostic lists (for aggregation)
    """
    per_game_records: List[dict] = []
    all_game_diag_lists: List[List[dict]] = []

    for rp in replays:
        od = rp.get("opening_diagnostics")
        odm = rp.get("opening_diagnostics_meta")
        if od is None or not isinstance(od, list) or len(od) == 0:
            continue
        per_game_records.append({
            "source": rp.get("_source_name", ""),
            "iteration": (rp.get("meta") or {}).get("iteration", -1),
            "game_id": rp.get("id", ""),
            "meta": odm or {},
            "diagnostics": od,
        })
        all_game_diag_lists.append(od)

    return per_game_records, all_game_diag_lists
```

- [ ] **Step 2: Add sidecar extraction function**

```python
def extract_sidecar_diagnostics(sidecars: Dict[int, dict]) -> Dict[int, dict]:
    """Extract opening_penalty_diagnostics from sidecar dicts.

    Returns: Dict mapping iteration -> opening_penalty_diagnostics sub-dict.
    """
    result: Dict[int, dict] = {}
    for it, sc in sidecars.items():
        opd = sc.get("opening_penalty_diagnostics")
        if opd and isinstance(opd, dict):
            result[it] = opd
    return result
```

- [ ] **Step 3: Verify syntax**

Run: `python3 -m py_compile scripts/opening_diagnostics_analyzer.py`

- [ ] **Step 4: Commit**

```bash
git add scripts/opening_diagnostics_analyzer.py
git commit -m "feat: add per-game and sidecar extraction for opening diagnostics"
```

---

### Task 3: Aggregation (sidecar multi-iteration merge + replay fallback)

**Files:**
- Modify: `scripts/opening_diagnostics_analyzer.py`

- [ ] **Step 1: Add sidecar multi-iteration aggregation**

```python
def aggregate_sidecar_opening_diagnostics(opd_by_iter: Dict[int, dict]) -> dict:
    """Merge opening_penalty_diagnostics from multiple iteration sidecars.

    Single iteration: passthrough (no recomputation).
    Multiple iterations: weighted merge of by_ply entries.
    """
    if not opd_by_iter:
        return {}
    if len(opd_by_iter) == 1:
        return next(iter(opd_by_iter.values()))

    # Collect all by_ply entries keyed by (ply_str, color) with weights
    merged_by_ply: Dict[str, Dict[str, dict]] = {}
    rollup_by_color: Dict[str, List[Tuple[int, dict]]] = {}  # color -> [(n, entry)]
    total_games = 0
    total_games_with_od = 0
    latest_it = max(opd_by_iter.keys())
    latest_meta = {}
    # Config consistency check across iterations
    config_values = set()

    for it in sorted(opd_by_iter.keys()):
        opd = opd_by_iter[it]
        total_games += opd.get("games_total", 0)
        total_games_with_od += opd.get("games_with_opening_diagnostics", 0)

        # Track config for consistency check
        config_values.add((
            opd.get("diagnostic_end_ply"),
            opd.get("extra_plies_after_penalty"),
            opd.get("floor_min_ply"),
        ))

        if it == latest_it:
            latest_meta = {
                "diagnostic_end_ply": opd.get("diagnostic_end_ply"),
                "extra_plies_after_penalty": opd.get("extra_plies_after_penalty"),
                "floor_min_ply": opd.get("floor_min_ply"),
                "used_floor": opd.get("used_floor"),
            }

    config_mismatch = len(config_values) > 1
    if config_mismatch:
        print(f"[WARN] Opening diagnostics config differs across iterations: {config_values}")

        for ply_str, colors in opd.get("by_ply", {}).items():
            if ply_str not in merged_by_ply:
                merged_by_ply[ply_str] = {}
            for color, entry in colors.items():
                n = entry.get("n", 0)
                if n <= 0:
                    continue
                if color not in merged_by_ply[ply_str]:
                    merged_by_ply[ply_str][color] = _empty_ply_entry()
                _accumulate_ply_entry(merged_by_ply[ply_str][color], entry, n)
                rollup_by_color.setdefault(color, []).append((n, entry))

    # Finalize weighted averages
    by_ply_final: Dict[str, Dict[str, dict]] = {}
    for ply_str in sorted(merged_by_ply.keys(), key=lambda s: int(s)):
        by_ply_final[ply_str] = {}
        for color, acc in merged_by_ply[ply_str].items():
            by_ply_final[ply_str][color] = _finalize_ply_entry(acc)

    # Build all_diagnostic_plies rollup
    all_diag = {}
    for color, entries in rollup_by_color.items():
        total_n = sum(n for n, _ in entries)
        if total_n == 0:
            continue
        acc = _empty_ply_entry()
        for n, entry in entries:
            _accumulate_ply_entry(acc, entry, n)
        all_diag[color] = _finalize_ply_entry(acc)

    result = {
        "version": 1,
        "games_total": total_games,
        "games_with_opening_diagnostics": total_games_with_od,
        "config_mismatch": config_mismatch,
        "all_diagnostic_plies": all_diag,
        "by_ply": by_ply_final,
    }
    result.update(latest_meta)
    # Note: rebound_vs_last_active is intentionally omitted from multi-iteration
    # merges — it is only meaningful within a single iteration's penalty window.
    return result


def _empty_ply_entry() -> dict:
    """Create an empty accumulator for weighted ply entry merging."""
    return {
        "total_n": 0,
        "penalties_active": None,
        "raw_mass": {r: 0.0 for r in REGIONS},
        "pen_mass": {r: 0.0 for r in REGIONS},
        "vis_mass": {r: 0.0 for r in REGIONS},
        "shift": {r: 0.0 for r in REGIONS},
        "raw_top1": {r: 0.0 for r in REGIONS},
        "pen_top1": {r: 0.0 for r in REGIONS},
        "vis_top1": {r: 0.0 for r in REGIONS},
        "legal": {r: 0.0 for r in REGIONS},
    }


def _accumulate_ply_entry(acc: dict, entry: dict, n: int) -> None:
    """Accumulate a weighted ply entry into the accumulator."""
    acc["total_n"] += n
    if acc["penalties_active"] is None:
        acc["penalties_active"] = entry.get("penalties_active", {})
    for r in REGIONS:
        acc["raw_mass"][r] += entry.get("mean_raw_mass", {}).get(r, 0.0) * n
        acc["pen_mass"][r] += entry.get("mean_penalized_mass", {}).get(r, 0.0) * n
        acc["vis_mass"][r] += entry.get("mean_visit_mass", {}).get(r, 0.0) * n
        acc["shift"][r] += entry.get("mean_penalty_shift", {}).get(r, 0.0) * n
        acc["raw_top1"][r] += entry.get("raw_top1_region_pct", {}).get(r, 0.0) * n
        acc["pen_top1"][r] += entry.get("penalized_top1_region_pct", {}).get(r, 0.0) * n
        acc["vis_top1"][r] += entry.get("visit_top1_region_pct", {}).get(r, 0.0) * n
        acc["legal"][r] += entry.get("mean_legal_counts", {}).get(r, 0.0) * n


def _finalize_ply_entry(acc: dict) -> dict:
    """Finalize a weighted accumulator into a standard ply entry."""
    n = acc["total_n"]
    if n <= 0:
        return {"n": 0}
    return {
        "n": n,
        "penalties_active": acc["penalties_active"] or {},
        "mean_raw_mass": {r: round(acc["raw_mass"][r] / n, 4) for r in REGIONS},
        "mean_penalized_mass": {r: round(acc["pen_mass"][r] / n, 4) for r in REGIONS},
        "mean_visit_mass": {r: round(acc["vis_mass"][r] / n, 4) for r in REGIONS},
        "mean_penalty_shift": {r: round(acc["shift"][r] / n, 4) for r in REGIONS},
        "raw_top1_region_pct": {r: round(acc["raw_top1"][r] / n, 3) for r in REGIONS},
        "penalized_top1_region_pct": {r: round(acc["pen_top1"][r] / n, 3) for r in REGIONS},
        "visit_top1_region_pct": {r: round(acc["vis_top1"][r] / n, 3) for r in REGIONS},
        "mean_legal_counts": {r: round(acc["legal"][r] / n, 1) for r in REGIONS},
    }
```

- [ ] **Step 2: Add replay fallback aggregation**

```python
def aggregate_replay_opening_diagnostics(
    per_game_records: List[dict],
    all_game_diag_lists: List[List[dict]],
    total_games: int,
) -> dict:
    """Aggregate opening diagnostics from per-game records (fallback when sidecars unavailable).

    Uses canonical aggregation function if available, otherwise falls back to
    local implementation.
    """
    if not per_game_records:
        return {}

    meta = per_game_records[0].get("meta", {})

    if _CANONICAL_AGGREGATE and _canonical_aggregate is not None:
        return _canonical_aggregate(
            all_game_diagnostics=all_game_diag_lists,
            diagnostic_end_ply=meta.get("diagnostic_end_ply", 6),
            extra_plies=meta.get("extra_plies_after_penalty", 2),
            floor_min_ply=meta.get("floor_min_ply", 4),
            used_floor=meta.get("used_floor", False),
            games_total_iter=total_games,
        )

    # Fallback: use local sidecar-style aggregation on per-game data
    # Build the same structure as sidecar would produce
    print("[INFO] Using analyzer-local aggregation (canonical import unavailable)")
    by_ply_color: Dict[Tuple[int, str], List[dict]] = {}
    for diag_list in all_game_diag_lists:
        for rec in diag_list:
            key = (rec["ply"], rec["side_to_move"])
            by_ply_color.setdefault(key, []).append(rec)

    last_active_ply: Dict[str, int] = {}
    by_ply: Dict[str, Dict[str, dict]] = {}
    rollup_by_color: Dict[str, List[dict]] = {}

    for (ply, color), recs in sorted(by_ply_color.items()):
        n = len(recs)
        ply_key = str(ply)
        if ply_key not in by_ply:
            by_ply[ply_key] = {}

        is_active = recs[0].get("penalties_active", {}).get("edge_band", False) or recs[0].get("penalties_active", {}).get("near_corner", False)
        if is_active:
            last_active_ply[color] = ply

        mean_raw = {r: sum(rec.get("raw_mass", {}).get(r, 0.0) for rec in recs) / n for r in REGIONS}
        mean_pen = {r: sum(rec.get("penalized_mass", {}).get(r, 0.0) for rec in recs) / n for r in REGIONS}
        mean_vis = {r: sum(rec.get("visit_mass", {}).get(r, 0.0) for rec in recs) / n for r in REGIONS}

        raw_top1_pct = {r: 0 for r in REGIONS}
        pen_top1_pct = {r: 0 for r in REGIONS}
        vis_top1_pct = {r: 0 for r in REGIONS}
        for rec in recs:
            raw_top1_pct[rec.get("raw_top1", {}).get("primary_region", "interior")] += 1
            pen_top1_pct[rec.get("penalized_top1", {}).get("primary_region", "interior")] += 1
            vis_top1_pct[rec.get("visit_top1", {}).get("primary_region", "interior")] += 1

        entry = {
            "n": n,
            "penalties_active": recs[0].get("penalties_active", {}),
            "mean_raw_mass": {r: round(mean_raw[r], 4) for r in REGIONS},
            "mean_penalized_mass": {r: round(mean_pen[r], 4) for r in REGIONS},
            "mean_visit_mass": {r: round(mean_vis[r], 4) for r in REGIONS},
            "mean_penalty_shift": {r: round(mean_pen[r] - mean_raw[r], 4) for r in REGIONS},
            "raw_top1_region_pct": {r: round(v / n, 3) for r, v in raw_top1_pct.items()},
            "penalized_top1_region_pct": {r: round(v / n, 3) for r, v in pen_top1_pct.items()},
            "visit_top1_region_pct": {r: round(v / n, 3) for r, v in vis_top1_pct.items()},
            "mean_legal_counts": {r: round(sum(rec.get("legal_move_counts", {}).get(r, 0) for rec in recs) / n, 1) for r in REGIONS},
        }
        by_ply[ply_key][color] = entry
        rollup_by_color.setdefault(color, []).append(entry)

    # Pass 2: rebound
    for (ply, color), recs in sorted(by_ply_color.items()):
        is_active = recs[0].get("penalties_active", {}).get("edge_band", False) or recs[0].get("penalties_active", {}).get("near_corner", False)
        if is_active:
            continue
        last_ply = last_active_ply.get(color)
        if last_ply is None:
            continue
        last_entry = by_ply.get(str(last_ply), {}).get(color)
        this_entry = by_ply.get(str(ply), {}).get(color)
        if last_entry and this_entry:
            this_entry["rebound_vs_last_active"] = {
                "near_corner_mass_delta": round(this_entry["mean_visit_mass"]["near_corner"] - last_entry["mean_visit_mass"]["near_corner"], 4),
                "edge_band_mass_delta": round(this_entry["mean_visit_mass"]["edge_band"] - last_entry["mean_visit_mass"]["edge_band"], 4),
            }

    # Build all_diagnostic_plies rollup
    all_diag = {}
    for color, entries in rollup_by_color.items():
        total_n = sum(e["n"] for e in entries)
        if total_n == 0:
            continue
        all_diag[color] = {
            "n": total_n,
            "mean_raw_mass": {r: round(sum(e["mean_raw_mass"][r] * e["n"] for e in entries) / total_n, 4) for r in REGIONS},
            "mean_penalized_mass": {r: round(sum(e["mean_penalized_mass"][r] * e["n"] for e in entries) / total_n, 4) for r in REGIONS},
            "mean_visit_mass": {r: round(sum(e["mean_visit_mass"][r] * e["n"] for e in entries) / total_n, 4) for r in REGIONS},
            "mean_penalty_shift": {r: round(sum(e["mean_penalty_shift"][r] * e["n"] for e in entries) / total_n, 4) for r in REGIONS},
            "raw_top1_region_pct": {r: round(sum(e["raw_top1_region_pct"][r] * e["n"] for e in entries) / total_n, 3) for r in REGIONS},
            "penalized_top1_region_pct": {r: round(sum(e["penalized_top1_region_pct"][r] * e["n"] for e in entries) / total_n, 3) for r in REGIONS},
            "visit_top1_region_pct": {r: round(sum(e["visit_top1_region_pct"][r] * e["n"] for e in entries) / total_n, 3) for r in REGIONS},
            "mean_legal_counts": {r: round(sum(e["mean_legal_counts"][r] * e["n"] for e in entries) / total_n, 1) for r in REGIONS},
        }

    return {
        "version": 1,
        "diagnostic_end_ply": meta.get("diagnostic_end_ply", 6),
        "extra_plies_after_penalty": meta.get("extra_plies_after_penalty", 2),
        "floor_min_ply": meta.get("floor_min_ply", 4),
        "used_floor": meta.get("used_floor", False),
        "games_total": total_games,
        "games_with_opening_diagnostics": len(per_game_records),
        "all_diagnostic_plies": all_diag,
        "by_ply": by_ply,
    }
```

- [ ] **Step 3: Verify syntax**

Run: `python3 -m py_compile scripts/opening_diagnostics_analyzer.py`

- [ ] **Step 4: Commit**

```bash
git add scripts/opening_diagnostics_analyzer.py
git commit -m "feat: add aggregation functions (sidecar merge + replay fallback)"
```

---

### Task 4: Validation

**Files:**
- Modify: `scripts/opening_diagnostics_analyzer.py`

- [ ] **Step 1: Add validation functions**

```python
def validate_per_game_record(rec: dict) -> List[str]:
    """Validate a single per-ply diagnostic record. Returns list of warnings."""
    warnings = []
    for stage in ("raw_mass", "penalized_mass", "visit_mass"):
        vals = rec.get(stage, {})
        total = sum(vals.get(r, 0.0) for r in REGIONS)
        if abs(total - 1.0) > MASS_TOLERANCE:
            warnings.append(f"ply={rec.get('ply')} {stage} sum={total:.4f} (expected ~1.0)")

    lmc = rec.get("legal_move_counts", {})
    lmt = rec.get("legal_moves_total", 0)
    lmc_sum = sum(lmc.get(r, 0) for r in REGIONS)
    if lmt != lmc_sum:
        warnings.append(f"ply={rec.get('ply')} legal_moves_total={lmt} != sum(counts)={lmc_sum}")

    for stage in ("raw_top1", "penalized_top1", "visit_top1"):
        pr = rec.get(stage, {}).get("primary_region", "")
        if pr not in VALID_REGIONS:
            warnings.append(f"ply={rec.get('ply')} {stage}.primary_region={pr!r} invalid")

    return warnings


def validate_sidecar_entry(entry: dict) -> List[str]:
    """Validate a single by_ply[ply][color] sidecar entry. Returns warnings."""
    warnings = []
    n = entry.get("n", 0)
    if n <= 0:
        warnings.append(f"n={n} (expected > 0)")
        return warnings

    for field in ("raw_top1_region_pct", "penalized_top1_region_pct", "visit_top1_region_pct"):
        vals = entry.get(field, {})
        total = sum(vals.get(r, 0.0) for r in REGIONS)
        if abs(total - 1.0) > MASS_TOLERANCE:
            warnings.append(f"{field} sum={total:.3f} (expected ~1.0)")

    for field in ("mean_raw_mass", "mean_penalized_mass", "mean_visit_mass"):
        vals = entry.get(field, {})
        total = sum(vals.get(r, 0.0) for r in REGIONS)
        if abs(total - 1.0) > MASS_TOLERANCE:
            warnings.append(f"{field} sum={total:.4f} (expected ~1.0)")

    return warnings


def validate_all(
    per_game_records: List[dict],
    opd_aggregate: dict,
) -> List[str]:
    """Run all validation checks. Returns aggregated warnings."""
    warnings = []

    # Per-game validation (sample first 10 games to avoid excessive output)
    for gr in per_game_records[:10]:
        for rec in gr.get("diagnostics", []):
            for w in validate_per_game_record(rec):
                warnings.append(f"game={gr.get('game_id', '?')} {w}")

    # Sidecar validation
    for ply_str, colors in opd_aggregate.get("by_ply", {}).items():
        for color, entry in colors.items():
            for w in validate_sidecar_entry(entry):
                warnings.append(f"sidecar ply={ply_str} {color}: {w}")

    return warnings
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile scripts/opening_diagnostics_analyzer.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/opening_diagnostics_analyzer.py
git commit -m "feat: add opening diagnostics validation functions"
```

---

### Task 5: Summary dict builders

**Files:**
- Modify: `scripts/opening_diagnostics_analyzer.py`

- [ ] **Step 1: Add summary builders**

```python
def build_opening_diagnostics_summary(
    opd_aggregate: dict,
    total_games: int,
    games_with_diagnostics: int,
    source_mode: str,
) -> dict:
    """Build the opening_diagnostics_summary for the analyzer's summary.json."""
    return {
        "source": source_mode,
        "aggregation_impl": "canonical" if _CANONICAL_AGGREGATE else "analyzer_fallback",
        "coverage": {
            "games_total": total_games,
            "games_with_diagnostics": games_with_diagnostics,
            "coverage_pct": round(games_with_diagnostics / max(total_games, 1) * 100, 1),
        },
        "diagnostic_end_ply": opd_aggregate.get("diagnostic_end_ply"),
        "extra_plies_after_penalty": opd_aggregate.get("extra_plies_after_penalty"),
        "floor_min_ply": opd_aggregate.get("floor_min_ply"),
        "used_floor": opd_aggregate.get("used_floor"),
        "all_diagnostic_plies": opd_aggregate.get("all_diagnostic_plies", {}),
    }


def build_opening_diagnostics_by_ply(opd_aggregate: dict) -> dict:
    """Extract the by_ply dict for the analyzer's summary.json."""
    return opd_aggregate.get("by_ply", {})
```

- [ ] **Step 2: Verify syntax and commit**

```bash
python3 -m py_compile scripts/opening_diagnostics_analyzer.py
git add scripts/opening_diagnostics_analyzer.py
git commit -m "feat: add opening diagnostics summary builders"
```

---

### Task 6: CSV exports

**Files:**
- Modify: `scripts/opening_diagnostics_analyzer.py`

- [ ] **Step 1: Add opening_summary.csv writer**

```python
def write_opening_summary_csv(out_dir: str, opd_summary: dict, iteration_info: dict) -> str:
    """Write opening_summary.csv — one row with all-ply rollup per color."""
    path = os.path.join(out_dir, "opening_summary.csv")
    all_plies = opd_summary.get("all_diagnostic_plies", {})
    cov = opd_summary.get("coverage", {})

    header = ["iteration", "iteration_min", "iteration_max",
              "games_total", "games_with_diagnostics", "diagnostic_end_ply"]
    for color in ("red", "black"):
        for stage in ("raw", "penalized", "visit"):
            for r in REGIONS:
                header.append(f"{color}_{stage}_{r}")
        for stage in ("raw", "penalized", "visit"):
            for r in REGIONS:
                header.append(f"{color}_{stage}_top1_{r}")

    row = [
        iteration_info.get("iteration", ""),
        iteration_info.get("iteration_min", ""),
        iteration_info.get("iteration_max", ""),
        cov.get("games_total", 0),
        cov.get("games_with_diagnostics", 0),
        opd_summary.get("diagnostic_end_ply", ""),
    ]
    for color in ("red", "black"):
        data = all_plies.get(color, {})
        for stage, key in [("raw", "mean_raw_mass"), ("penalized", "mean_penalized_mass"), ("visit", "mean_visit_mass")]:
            for r in REGIONS:
                row.append(data.get(key, {}).get(r, ""))
        for stage, key in [("raw", "raw_top1_region_pct"), ("penalized", "penalized_top1_region_pct"), ("visit", "visit_top1_region_pct")]:
            for r in REGIONS:
                row.append(data.get(key, {}).get(r, ""))

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)
    return path
```

- [ ] **Step 2: Add opening_by_ply.csv writer**

```python
def write_opening_by_ply_csv(out_dir: str, by_ply: dict) -> str:
    """Write opening_by_ply.csv — one row per (ply, color)."""
    path = os.path.join(out_dir, "opening_by_ply.csv")
    header = [
        "ply", "color", "n", "penalties_active_edge", "penalties_active_corner",
    ]
    for prefix in ("mean_raw", "mean_penalized", "mean_visit", "mean_shift"):
        for r in REGIONS:
            header.append(f"{prefix}_{r}")
    for prefix in ("raw_top1", "penalized_top1", "visit_top1"):
        for r in REGIONS:
            header.append(f"{prefix}_{r}")
    header.extend(["rebound_near_corner_delta", "rebound_edge_band_delta"])

    rows = []
    for ply_str in sorted(by_ply.keys(), key=lambda s: int(s)):
        for color in ("red", "black"):
            entry = by_ply[ply_str].get(color)
            if not entry or entry.get("n", 0) <= 0:
                continue
            pa = entry.get("penalties_active", {})
            row = [
                ply_str, color, entry["n"],
                pa.get("edge_band", False), pa.get("near_corner", False),
            ]
            for key in ("mean_raw_mass", "mean_penalized_mass", "mean_visit_mass", "mean_penalty_shift"):
                for r in REGIONS:
                    row.append(entry.get(key, {}).get(r, ""))
            for key in ("raw_top1_region_pct", "penalized_top1_region_pct", "visit_top1_region_pct"):
                for r in REGIONS:
                    row.append(entry.get(key, {}).get(r, ""))
            rebound = entry.get("rebound_vs_last_active", {})
            row.append(rebound.get("near_corner_mass_delta", ""))
            row.append(rebound.get("edge_band_mass_delta", ""))
            rows.append(row)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path
```

- [ ] **Step 3: Add opening_per_game.csv writer (debug CSV)**

```python
def write_opening_per_game_csv(out_dir: str, per_game_records: List[dict]) -> str:
    """Write opening_per_game.csv — one row per (game, ply). Debug detail."""
    path = os.path.join(out_dir, "opening_per_game.csv")
    header = [
        "iteration", "game_id", "ply", "side_to_move",
        "penalty_active_edge", "penalty_active_corner",
        "raw_near_corner", "raw_edge_band", "raw_interior",
        "penalized_near_corner", "penalized_edge_band", "penalized_interior",
        "visit_near_corner", "visit_edge_band", "visit_interior",
        "raw_top1_region", "penalized_top1_region", "visit_top1_region",
        "legal_moves_total",
    ]
    rows = []
    for gr in per_game_records:
        for rec in gr.get("diagnostics", []):
            pa = rec.get("penalties_active", {})
            row = [
                gr.get("iteration", ""), gr.get("game_id", ""),
                rec.get("ply", ""), rec.get("side_to_move", ""),
                pa.get("edge_band", False), pa.get("near_corner", False),
            ]
            for key in ("raw_mass", "penalized_mass", "visit_mass"):
                for r in REGIONS:
                    row.append(rec.get(key, {}).get(r, ""))
            row.append(rec.get("raw_top1", {}).get("primary_region", ""))
            row.append(rec.get("penalized_top1", {}).get("primary_region", ""))
            row.append(rec.get("visit_top1", {}).get("primary_region", ""))
            row.append(rec.get("legal_moves_total", ""))
            rows.append(row)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path
```

- [ ] **Step 4: Verify syntax and commit**

```bash
python3 -m py_compile scripts/opening_diagnostics_analyzer.py
git add scripts/opening_diagnostics_analyzer.py
git commit -m "feat: add opening diagnostics CSV export functions"
```

---

### Task 7: Report text section

**Files:**
- Modify: `scripts/opening_diagnostics_analyzer.py`

- [ ] **Step 1: Add report formatting function**

```python
def format_opening_diagnostics_report(
    opd_summary: dict,
    by_ply: dict,
    warnings: List[str],
) -> List[str]:
    """Format opening diagnostics section for report.txt."""
    lines = []
    lines.append("Opening Penalty Diagnostics")
    lines.append("=" * 30)

    cov = opd_summary.get("coverage", {})
    lines.append(f"Source: {opd_summary.get('source', 'unknown')}")
    lines.append(f"Aggregation: {opd_summary.get('aggregation_impl', 'unknown')}")
    lines.append(f"Coverage: {cov.get('games_with_diagnostics', 0)}/{cov.get('games_total', 0)} "
                 f"games ({cov.get('coverage_pct', 0)}%)")
    end_ply = opd_summary.get('diagnostic_end_ply', '?')
    extra = opd_summary.get('extra_plies_after_penalty', '?')
    floor = opd_summary.get('floor_min_ply', '?')
    used_floor = opd_summary.get('used_floor', '?')
    lines.append(f"Diagnostic window: end_ply={end_ply}, extra={extra}, floor={floor}, used_floor={used_floor}")
    lines.append("")

    # All-ply rollup
    all_plies = opd_summary.get("all_diagnostic_plies", {})
    if all_plies:
        lines.append("All-ply rollup (weighted averages):")
        for color in ("red", "black"):
            data = all_plies.get(color, {})
            if not data:
                continue
            lines.append(f"  {color} (n={data.get('n', 0)}):")
            for label, key in [("raw ", "mean_raw_mass"), ("pen ", "mean_penalized_mass"), ("visit", "mean_visit_mass"), ("shift", "mean_penalty_shift")]:
                vals = data.get(key, {})
                parts = " ".join(f"{r[:2]}={vals.get(r, 0):.3f}" for r in REGIONS)
                lines.append(f"    {label}: {parts}")
            for label, key in [("raw_top1 ", "raw_top1_region_pct"), ("pen_top1 ", "penalized_top1_region_pct"), ("visit_top1", "visit_top1_region_pct")]:
                vals = data.get(key, {})
                parts = " ".join(f"{r[:2]}={vals.get(r, 0):.1%}" for r in REGIONS)
                lines.append(f"    {label}: {parts}")
        lines.append("")

    # Per-ply detail (only penalty-active plies + first rebound ply)
    if by_ply:
        lines.append("Per-ply detail:")
        for ply_str in sorted(by_ply.keys(), key=lambda s: int(s)):
            for color in ("red", "black"):
                entry = by_ply[ply_str].get(color)
                if not entry or entry.get("n", 0) <= 0:
                    continue
                pa = entry.get("penalties_active", {})
                active_str = []
                if pa.get("edge_band"):
                    active_str.append("edge")
                if pa.get("near_corner"):
                    active_str.append("corner")
                active_label = "+".join(active_str) if active_str else "inactive"

                lines.append(f"  ply={ply_str} {color} (n={entry['n']}, {active_label}):")
                for label, key in [("raw ", "mean_raw_mass"), ("pen ", "mean_penalized_mass"), ("visit", "mean_visit_mass")]:
                    vals = entry.get(key, {})
                    parts = " ".join(f"{r[:2]}={vals.get(r, 0):.3f}" for r in REGIONS)
                    lines.append(f"    {label}: {parts}")
                shift = entry.get("mean_penalty_shift", {})
                parts = " ".join(f"{r[:2]}={shift.get(r, 0):+.3f}" for r in REGIONS)
                lines.append(f"    shift: {parts}")

                rebound = entry.get("rebound_vs_last_active")
                if rebound:
                    lines.append(f"    rebound: nc={rebound.get('near_corner_mass_delta', 0):+.4f} "
                                 f"eb={rebound.get('edge_band_mass_delta', 0):+.4f}")
        lines.append("")

    # Warnings
    if warnings:
        lines.append(f"Validation warnings ({len(warnings)}):")
        for w in warnings[:20]:
            lines.append(f"  - {w}")
        if len(warnings) > 20:
            lines.append(f"  ... and {len(warnings) - 20} more")
    else:
        lines.append("Validation: OK (no warnings)")
    lines.append("")

    return lines
```

- [ ] **Step 2: Verify syntax and commit**

```bash
python3 -m py_compile scripts/opening_diagnostics_analyzer.py
git add scripts/opening_diagnostics_analyzer.py
git commit -m "feat: add opening diagnostics report text formatter"
```

---

### Task 8: Integrate into main analyzer

**Files:**
- Modify: `scripts/twixt_replay_analyzer.py`

- [ ] **Step 1: Add import (near top, after line 34)**

After the existing `from typing import ...` line, add:

```python
try:
    from opening_diagnostics_analyzer import (
        extract_per_game_diagnostics,
        extract_sidecar_diagnostics,
        aggregate_sidecar_opening_diagnostics,
        aggregate_replay_opening_diagnostics,
        validate_all,
        build_opening_diagnostics_summary,
        build_opening_diagnostics_by_ply,
        write_opening_summary_csv,
        write_opening_by_ply_csv,
        write_opening_per_game_csv,
        format_opening_diagnostics_report,
    )
    _HAS_OD_ANALYZER = True
except ImportError:
    _HAS_OD_ANALYZER = False
```

- [ ] **Step 2: Add opening diagnostics processing in analyze()**

Inside `analyze()`, after the sidecar coverage validation block (after `sc_agg = aggregate_sidecars(...)`) and before the summary dict construction, add:

```python
    # --- Opening diagnostics ---
    od_summary_dict = {}
    od_by_ply_dict = {}
    od_warnings = []
    od_per_game_records = []

    if _HAS_OD_ANALYZER:
        od_per_game_records, od_all_diag_lists = extract_per_game_diagnostics(replays)
        od_sidecar_data = extract_sidecar_diagnostics(relevant_sidecars) if use_sidecar else {}

        if use_sidecar and od_sidecar_data:
            od_aggregate = aggregate_sidecar_opening_diagnostics(od_sidecar_data)
            od_source = "sidecar"
            games_with_od = od_aggregate.get("games_with_opening_diagnostics", 0)
        elif od_per_game_records:
            od_aggregate = aggregate_replay_opening_diagnostics(
                od_per_game_records, od_all_diag_lists, n)
            od_source = "replay_fallback"
            games_with_od = len(od_per_game_records)
        else:
            od_aggregate = {}
            od_source = "none"
            games_with_od = 0

        if od_aggregate:
            od_warnings = validate_all(od_per_game_records, od_aggregate)
            if od_warnings:
                print(f"[WARN] Opening diagnostics: {len(od_warnings)} validation warning(s)")
            od_summary_dict = build_opening_diagnostics_summary(
                od_aggregate, n, games_with_od, od_source)
            od_by_ply_dict = build_opening_diagnostics_by_ply(od_aggregate)
```

- [ ] **Step 3: Add to summary dict**

In the summary dict construction, after the `"notes"` entry, add:

```python
        "opening_diagnostics_summary": od_summary_dict,
        "opening_diagnostics_by_ply": od_by_ply_dict,
```

- [ ] **Step 4: Write CSV files after summary.json**

After the `json.dump(summary, ...)` call, add:

```python
    if _HAS_OD_ANALYZER and od_summary_dict:
        _od_iter_info = {
            "iteration": summary.get("iteration"),
            "iteration_min": summary.get("iteration_min"),
            "iteration_max": summary.get("iteration_max"),
        }
        _od_csv1 = write_opening_summary_csv(out_dir, od_summary_dict, _od_iter_info)
        _od_csv2 = write_opening_by_ply_csv(out_dir, od_by_ply_dict)
        print(f"[OK] wrote: {_od_csv1}")
        print(f"[OK] wrote: {_od_csv2}")
        if od_per_game_records:
            _od_csv3 = write_opening_per_game_csv(out_dir, od_per_game_records)
            print(f"[OK] wrote: {_od_csv3}")
```

- [ ] **Step 5: Add report.txt section**

Before `lines.append("Outputs:")`, add:

```python
    if _HAS_OD_ANALYZER and od_summary_dict:
        lines.extend(format_opening_diagnostics_report(od_summary_dict, od_by_ply_dict, od_warnings))
```

- [ ] **Step 6: Verify syntax**

Run: `python3 -m py_compile scripts/twixt_replay_analyzer.py`

- [ ] **Step 7: Commit**

```bash
git add scripts/twixt_replay_analyzer.py
git commit -m "feat: integrate opening diagnostics into replay analyzer output"
```

---

## Verification

1. **Syntax check both files:**
   ```
   python3 -m py_compile scripts/opening_diagnostics_analyzer.py
   python3 -m py_compile scripts/twixt_replay_analyzer.py
   ```

2. **Backward compat (old game files, no diagnostics):**
   ```
   python scripts/twixt_replay_analyzer.py --input Replays/860-869 --out /tmp/test_old --no-plots
   ```
   Verify: no errors, `opening_diagnostics_summary` and `opening_diagnostics_by_ply` are empty dicts in summary.json.

3. **With diagnostics (after training run with penalties):**
   Run analyzer against game directory with diagnostics-enabled game files + sidecars.
   Verify: all three CSVs produced, report.txt has opening diagnostics section, summary.json has populated opening sections.

4. **Check CSV columns match spec:**
   Verify `opening_by_ply.csv` has all columns from the user's spec: ply, color, n, penalties_active, mass columns, shift columns, top1 columns, rebound deltas.

5. **Validation output:**
   Check that mass sums are ~1.0 in the validation output (no warnings for well-formed data).
