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
        aggregate_root_child_details as _canonical_root_child_aggregate,
        build_early_override_summary as _canonical_early_override_summary,
    )
    _CANONICAL_AGGREGATE = True
except ImportError:
    try:
        from scripts.GPU.alphazero.opening_diagnostics import (
            aggregate_opening_diagnostics as _canonical_aggregate,
            aggregate_root_child_details as _canonical_root_child_aggregate,
            build_early_override_summary as _canonical_early_override_summary,
        )
        _CANONICAL_AGGREGATE = True
    except ImportError:
        _CANONICAL_AGGREGATE = False
        _canonical_aggregate = None
        _canonical_root_child_aggregate = None
        _canonical_early_override_summary = None


REGIONS = ("near_corner", "edge_band", "interior")
VALID_REGIONS = {"near_corner", "edge_band", "interior"}
MASS_TOLERANCE = 0.05


def _suffixed(name: str, ext: str, suffix: str) -> str:
    """Compose a filename as `{name}_{suffix}.{ext}` (or `{name}.{ext}` if no suffix).

    Kept local to each analyzer module (opening_diagnostics_analyzer +
    twixt_replay_analyzer) so neither module needs to import the other just
    for this one-liner.
    """
    if suffix:
        return f"{name}_{suffix}.{ext}"
    return f"{name}.{ext}"

# Phase 1: root-child diagnostics metric names (must match trainer-side aggregator)
ROOT_CHILD_METRICS = (
    "best_by_penalized_prior",
    "best_by_visit",
    "best_by_score",
    "best_by_q",
    "best_by_u",
)


# --- Extraction ---

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


# --- Aggregation helpers ---

def _empty_ply_entry() -> dict:
    """Create an empty accumulator for weighted ply entry merging."""
    return {
        "total_n": 0,
        "penalties_active": None,
        # Phase 2: effective penalty + source at this ply. Deterministic per
        # (ply, config), so a single value captured on first encounter is the
        # authoritative value for the whole merge. `first_non_none` logic in
        # `_accumulate_ply_entry` lets older sidecars (missing these keys)
        # pass through without polluting newer sidecars' values.
        "effective_near_corner_penalty": None,
        "near_corner_penalty_source": None,
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
    # Phase 2: first non-None wins. Any sidecar that carried the Phase 2
    # fields is authoritative for this (ply, color) — older sidecars lacking
    # the keys simply don't overwrite the value.
    if acc["effective_near_corner_penalty"] is None:
        v = entry.get("effective_near_corner_penalty")
        if v is not None:
            acc["effective_near_corner_penalty"] = v
    if acc["near_corner_penalty_source"] is None:
        v = entry.get("near_corner_penalty_source")
        if v is not None:
            acc["near_corner_penalty_source"] = v
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
        "effective_near_corner_penalty": acc.get("effective_near_corner_penalty"),
        "near_corner_penalty_source": acc.get("near_corner_penalty_source"),
        "mean_raw_mass": {r: round(acc["raw_mass"][r] / n, 4) for r in REGIONS},
        "mean_penalized_mass": {r: round(acc["pen_mass"][r] / n, 4) for r in REGIONS},
        "mean_visit_mass": {r: round(acc["vis_mass"][r] / n, 4) for r in REGIONS},
        "mean_penalty_shift": {r: round(acc["shift"][r] / n, 4) for r in REGIONS},
        "raw_top1_region_pct": {r: round(acc["raw_top1"][r] / n, 3) for r in REGIONS},
        "penalized_top1_region_pct": {r: round(acc["pen_top1"][r] / n, 3) for r in REGIONS},
        "visit_top1_region_pct": {r: round(acc["vis_top1"][r] / n, 3) for r in REGIONS},
        "mean_legal_counts": {r: round(acc["legal"][r] / n, 1) for r in REGIONS},
    }


# --- Sidecar multi-iteration aggregation ---

def aggregate_sidecar_opening_diagnostics(opd_by_iter: Dict[int, dict]) -> dict:
    """Merge opening_penalty_diagnostics from multiple iteration sidecars.

    Single iteration: passthrough (no recomputation).
    Multiple iterations: weighted merge of by_ply entries.
    """
    if not opd_by_iter:
        return {}
    if len(opd_by_iter) == 1:
        return next(iter(opd_by_iter.values()))

    merged_by_ply: Dict[str, Dict[str, dict]] = {}
    rollup_by_color: Dict[str, List[Tuple[int, dict]]] = {}
    total_games = 0
    total_games_with_od = 0
    latest_it = max(opd_by_iter.keys())
    latest_meta = {}
    # Phase 2: run-level near-corner config echo (required #1). Walk the
    # sidecars newest-to-oldest and take the first `run_config` that carries
    # the Phase 2 keys, so multi-sidecar merges still surface the early
    # override config in the final summary.
    run_config: Dict = {}
    _fallback_run_config: Dict = {}
    for _it in sorted(opd_by_iter.keys(), reverse=True):
        _rc = opd_by_iter[_it].get("run_config") or {}
        if not _rc:
            continue
        if not _fallback_run_config:
            _fallback_run_config = dict(_rc)
        if "near_corner_penalty_early" in _rc:
            run_config = dict(_rc)
            break
    if not run_config:
        run_config = _fallback_run_config
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
        # Bumped: matches the GPU-side canonical aggregator's v2 shape
        # (run_config + effective_near_corner_penalty per (ply, color)).
        "version": 2,
        "games_total": total_games,
        "games_with_opening_diagnostics": total_games_with_od,
        "config_mismatch": config_mismatch,
        "run_config": run_config,
        "all_diagnostic_plies": all_diag,
        "by_ply": by_ply_final,
    }
    result.update(latest_meta)
    # Note: rebound_vs_last_active is intentionally omitted from multi-iteration
    # merges — it is only meaningful within a single iteration's penalty window.
    return result


# --- Replay fallback aggregation ---

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

    # Pass 2: rebound (based on visit_mass — compares post-search behavior)
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


# --- Validation ---

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

    # Sidecar/aggregate validation
    for ply_str, colors in opd_aggregate.get("by_ply", {}).items():
        for color, entry in colors.items():
            for w in validate_sidecar_entry(entry):
                warnings.append(f"sidecar ply={ply_str} {color}: {w}")

    return warnings


# --- Summary builders ---

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


# --- CSV exports ---

def write_opening_summary_csv(
    out_dir: str,
    opd_summary: dict,
    iteration_info: dict,
    suffix: str = "",
) -> str:
    """Write opening_summary.csv — one row with all-ply rollup per color."""
    path = os.path.join(out_dir, _suffixed("opening_summary", "csv", suffix))
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


def write_opening_by_ply_csv(out_dir: str, by_ply: dict, suffix: str = "") -> str:
    """Write opening_by_ply.csv — one row per (ply, color)."""
    path = os.path.join(out_dir, _suffixed("opening_by_ply", "csv", suffix))
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


def write_opening_per_game_csv(
    out_dir: str,
    per_game_records: List[dict],
    suffix: str = "",
) -> str:
    """Write opening_per_game.csv — one row per (game, ply). Debug detail."""
    path = os.path.join(out_dir, _suffixed("opening_per_game", "csv", suffix))
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


# --- Report text ---

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

    # Per-ply detail
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


# =============================================================================
# Phase 1: Root-child diagnostics (ply 0–1 deep inspection)
# =============================================================================
#
# These helpers read the `root_child_diagnostics` block the trainer writes into
# per-iteration sidecars (emitted by
# scripts.GPU.alphazero.opening_diagnostics.aggregate_root_child_details).
# They are fully optional — older sidecars without this block will yield an
# empty aggregate and no CSV/report output, letting the analyzer keep working.


def extract_sidecar_root_child_diagnostics(sidecars: Dict[int, dict]) -> Dict[int, dict]:
    """Extract `root_child_diagnostics` from sidecar dicts.

    Args:
        sidecars: Dict mapping iteration -> parsed iter_NNNN_stats.json

    Returns:
        Dict iteration -> root_child_diagnostics block. Iterations without the
        block (older runs) are silently skipped.
    """
    result: Dict[int, dict] = {}
    for it, sc in sidecars.items():
        rcd = sc.get("root_child_diagnostics")
        if rcd and isinstance(rcd, dict):
            result[it] = rcd
    return result


def _empty_root_child_entry() -> dict:
    """Accumulator for weighted merge of a (ply, color) root-child entry."""
    acc: dict = {
        "total_n": 0,
        "score_tie_count_sum": 0.0,       # weighted by n
        "score_tie_rate_sum": 0.0,        # weighted by n
        "nn_value_sum": 0.0,              # weighted by n (skipping None)
        "nn_value_weight": 0,             # n that contributed a non-null nn_value
    }
    for metric in ROOT_CHILD_METRICS:
        acc[f"{metric}_region_sum"] = {r: 0.0 for r in REGIONS}
    return acc


def _accumulate_root_child_entry(acc: dict, entry: dict, n: int) -> None:
    """Weight-accumulate one iteration's (ply, color) entry into the rollup."""
    if n <= 0:
        return
    acc["total_n"] += n
    acc["score_tie_count_sum"] += float(entry.get("score_tie_count_mean", 0.0)) * n
    acc["score_tie_rate_sum"] += float(entry.get("score_tie_rate", 0.0)) * n

    nn_val = entry.get("nn_value_mean")
    if nn_val is not None:
        acc["nn_value_sum"] += float(nn_val) * n
        acc["nn_value_weight"] += n

    for metric in ROOT_CHILD_METRICS:
        pcts = entry.get(f"{metric}_region_pct", {}) or {}
        for r in REGIONS:
            acc[f"{metric}_region_sum"][r] += float(pcts.get(r, 0.0)) * n


def _finalize_root_child_entry(acc: dict) -> dict:
    """Normalize weighted sums back into a sidecar-style per-entry dict."""
    n = acc["total_n"]
    if n <= 0:
        return {"n": 0}
    out: dict = {
        "n": n,
        "score_tie_count_mean": round(acc["score_tie_count_sum"] / n, 3),
        "score_tie_rate": round(acc["score_tie_rate_sum"] / n, 3),
        "nn_value_mean": (
            round(acc["nn_value_sum"] / acc["nn_value_weight"], 4)
            if acc["nn_value_weight"] > 0 else None
        ),
    }
    for metric in ROOT_CHILD_METRICS:
        out[f"{metric}_region_pct"] = {
            r: round(acc[f"{metric}_region_sum"][r] / n, 3) for r in REGIONS
        }
    return out


def aggregate_replay_root_child_diagnostics(
    all_game_diag_lists: List[List[dict]],
    child_detail_max_ply: int = 2,
) -> dict:
    """Fallback aggregation from per-game records when sidecars don't carry
    the `root_child_diagnostics` block.

    This is the same path the trainer uses — reuses the canonical
    `aggregate_root_child_details` from the training module. When the canonical
    import isn't available (minimal install), returns an empty dict.

    Useful when older runs' sidecars predate the block but per-game JSONs
    already carry `root_summary` + `top_children` on early plies.
    """
    if not all_game_diag_lists:
        return {}
    if _canonical_root_child_aggregate is None:
        print("[INFO] root_child per-game fallback unavailable "
              "(canonical aggregator not importable)")
        return {}
    return _canonical_root_child_aggregate(
        all_game_diagnostics=all_game_diag_lists,
        child_detail_max_ply=child_detail_max_ply,
    )


def aggregate_sidecar_root_child_diagnostics(rcd_by_iter: Dict[int, dict]) -> dict:
    """Merge `root_child_diagnostics` blocks from multiple iteration sidecars.

    Single iteration → pass-through (no recomputation).
    Multiple iterations → weighted merge by `n` for each (ply, color).
    """
    if not rcd_by_iter:
        return {}
    if len(rcd_by_iter) == 1:
        return next(iter(rcd_by_iter.values()))

    merged: Dict[str, Dict[str, dict]] = {}
    for it in sorted(rcd_by_iter.keys()):
        rcd = rcd_by_iter[it]
        for ply_str, colors in (rcd.get("by_ply") or {}).items():
            if ply_str not in merged:
                merged[ply_str] = {}
            for color, entry in colors.items():
                n = entry.get("n", 0)
                if n <= 0:
                    continue
                if color not in merged[ply_str]:
                    merged[ply_str][color] = _empty_root_child_entry()
                _accumulate_root_child_entry(merged[ply_str][color], entry, n)

    by_ply_final: Dict[str, Dict[str, dict]] = {}
    for ply_str in sorted(merged.keys(), key=lambda s: int(s)):
        by_ply_final[ply_str] = {}
        for color, acc in merged[ply_str].items():
            by_ply_final[ply_str][color] = _finalize_root_child_entry(acc)

    return {
        "metrics": list(ROOT_CHILD_METRICS),
        "by_ply": by_ply_final,
    }


def compute_disagreement_metrics(entry: dict) -> Dict[str, float]:
    """Derived per-entry columns that expose the root-cause signal.

    For each region `r` and each non-baseline metric (visit/score/q/u), compute:
        {metric}_minus_penalized_{r} =
            best_by_{metric}_region_pct[r] - best_by_penalized_prior_region_pct[r]

    Positive values on near_corner mean that component is *re-introducing*
    near-corner moves relative to the penalized prior baseline.
    """
    baseline = entry.get("best_by_penalized_prior_region_pct", {}) or {}
    out: Dict[str, float] = {}
    for metric in ("best_by_visit", "best_by_score", "best_by_q", "best_by_u"):
        other = entry.get(f"{metric}_region_pct", {}) or {}
        short = metric.replace("best_by_", "")  # visit, score, q, u
        for r in REGIONS:
            out[f"{short}_minus_penalized_{r}"] = round(
                float(other.get(r, 0.0)) - float(baseline.get(r, 0.0)), 3
            )
    return out


# --- CSV exports ---

def write_root_child_by_ply_csv(
    out_dir: str,
    rcd_aggregate: dict,
    suffix: str = "",
) -> Optional[str]:
    """Write root_child_by_ply.csv — one row per (ply, color).

    Returns the file path, or None if there is no root_child data to write
    (older run without the block — caller can skip the section cleanly).
    """
    by_ply = rcd_aggregate.get("by_ply") or {}
    if not by_ply:
        return None

    path = os.path.join(out_dir, _suffixed("root_child_by_ply", "csv", suffix))

    header = ["ply", "color", "n"]
    # Raw region percentages per metric
    for metric in ROOT_CHILD_METRICS:
        for r in REGIONS:
            header.append(f"{metric}_{r}_pct")
    # Derived disagreement columns (vs penalized_prior)
    for metric in ("visit", "score", "q", "u"):
        for r in REGIONS:
            header.append(f"{metric}_minus_penalized_{r}")
    header.extend(["score_tie_count_mean", "score_tie_rate", "nn_value_mean"])

    rows = []
    for ply_str in sorted(by_ply.keys(), key=lambda s: int(s)):
        for color in ("red", "black"):
            entry = by_ply[ply_str].get(color)
            if not entry or entry.get("n", 0) <= 0:
                continue
            row = [ply_str, color, entry["n"]]
            for metric in ROOT_CHILD_METRICS:
                pcts = entry.get(f"{metric}_region_pct", {}) or {}
                for r in REGIONS:
                    row.append(pcts.get(r, ""))
            disagree = compute_disagreement_metrics(entry)
            for metric in ("visit", "score", "q", "u"):
                for r in REGIONS:
                    row.append(disagree.get(f"{metric}_minus_penalized_{r}", ""))
            row.append(entry.get("score_tie_count_mean", ""))
            row.append(entry.get("score_tie_rate", ""))
            nn = entry.get("nn_value_mean")
            row.append("" if nn is None else nn)
            rows.append(row)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def write_root_child_per_game_csv(
    out_dir: str,
    per_game_records: List[dict],
    suffix: str = "",
) -> Optional[str]:
    """Write per-game per-child detail (ply 0 / 1 only) — optional, bulky.

    One row per (game, ply, child). Pulls from per-root records that carry
    `root_summary` + `top_children` (only ply < CHILD_DETAIL_PLIES; older
    records / later plies are silently skipped).

    Returns the file path, or None if no game carries child details at all.
    """
    any_child = False
    for gr in per_game_records:
        for rec in gr.get("diagnostics", []):
            if "top_children" in rec:
                any_child = True
                break
        if any_child:
            break
    if not any_child:
        return None

    path = os.path.join(out_dir, _suffixed("root_child_per_game", "csv", suffix))
    header = [
        "iteration", "game_id", "ply", "side_to_move",
        "root_visits", "root_q", "root_nn_value", "root_score_tie_count",
        "rank_by_visits", "move_r", "move_c", "region",
        "prior_raw", "prior_penalized",
        "visit_count", "visit_share",
        "q_value_child", "q", "u", "score", "in_score_tie",
        "best_by_penalized_prior_region", "best_by_visit_region",
        "best_by_score_region", "best_by_q_region", "best_by_u_region",
    ]

    rows = []
    for gr in per_game_records:
        iteration = gr.get("iteration", "")
        game_id = gr.get("game_id", "")
        for rec in gr.get("diagnostics", []):
            top = rec.get("top_children")
            rs = rec.get("root_summary")
            if not top or not rs:
                continue
            best_regions = {}
            for metric in ROOT_CHILD_METRICS:
                bm = rs.get(metric) or {}
                best_regions[metric] = bm.get("region", "")
            for rank, ch in enumerate(top, start=1):
                mv = ch.get("move", [None, None])
                rows.append([
                    iteration, game_id, rec.get("ply", ""), rec.get("side_to_move", ""),
                    rs.get("visit_count", ""), rs.get("q_value", ""),
                    rs.get("nn_value", ""), rs.get("score_tie_count", ""),
                    rank,
                    mv[0] if len(mv) > 0 else "",
                    mv[1] if len(mv) > 1 else "",
                    ch.get("region", ""),
                    ch.get("prior_raw", ""), ch.get("prior_penalized", ""),
                    ch.get("visit_count", ""), ch.get("visit_share", ""),
                    ch.get("q_value_child", ""), ch.get("q", ""),
                    ch.get("u", ""), ch.get("score", ""),
                    ch.get("in_score_tie", ""),
                    best_regions["best_by_penalized_prior"],
                    best_regions["best_by_visit"],
                    best_regions["best_by_score"],
                    best_regions["best_by_q"],
                    best_regions["best_by_u"],
                ])

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


# --- Summary + report ---

# Thresholds for calling out divergences in the text report. These are
# chosen conservatively — they trigger only on material gaps, not noise.
_DIVERGENCE_PCT = 0.15   # |visit − penalized_prior| >= 15% in a region
_TIE_RATE_FLAG = 0.10    # >=10% of roots ended in a score tie


def build_root_child_summary(rcd_aggregate: dict) -> dict:
    """Compact summary dict for summary.json, including top-line divergences."""
    by_ply = rcd_aggregate.get("by_ply") or {}
    if not by_ply:
        return {"present": False}

    divergences: List[dict] = []
    for ply_str in sorted(by_ply.keys(), key=lambda s: int(s)):
        for color in ("red", "black"):
            entry = by_ply[ply_str].get(color)
            if not entry or entry.get("n", 0) <= 0:
                continue
            disagree = compute_disagreement_metrics(entry)
            # Flag near_corner divergence between penalized_prior and visit
            v_nc = disagree.get("visit_minus_penalized_near_corner", 0.0)
            if v_nc >= _DIVERGENCE_PCT:
                # Attribute to whichever component is carrying the weight
                by_comp = {
                    "q": disagree.get("q_minus_penalized_near_corner", 0.0),
                    "u": disagree.get("u_minus_penalized_near_corner", 0.0),
                    "score": disagree.get("score_minus_penalized_near_corner", 0.0),
                }
                dominant = max(by_comp, key=lambda k: by_comp[k])
                divergences.append({
                    "ply": int(ply_str),
                    "color": color,
                    "n": entry["n"],
                    "visit_minus_penalized_near_corner": v_nc,
                    "q_minus_penalized_near_corner": by_comp["q"],
                    "u_minus_penalized_near_corner": by_comp["u"],
                    "score_minus_penalized_near_corner": by_comp["score"],
                    "dominant_component": dominant,
                    "score_tie_rate": entry.get("score_tie_rate", 0.0),
                })

    return {
        "present": True,
        "divergence_threshold_pct": _DIVERGENCE_PCT,
        "tie_rate_flag": _TIE_RATE_FLAG,
        "divergences": divergences,
        "by_ply": by_ply,
    }


def build_early_override_summary(
    opd_aggregate: dict,
    rcd_aggregate: Optional[dict] = None,
    early_plies: int = 2,
) -> dict:
    """Analyzer-side builder — delegates to the canonical helper when importable.

    When the training module isn't importable (rare — minimal install), falls
    back to returning an empty dict so callers can still produce a report.
    """
    if _canonical_early_override_summary is None:
        return {}
    return _canonical_early_override_summary(
        opd_aggregate=opd_aggregate,
        rcd_aggregate=rcd_aggregate,
        early_plies=early_plies,
    )


def format_early_override_report(eo_summary: dict) -> List[str]:
    """Format the Phase 2 early-override summary section for report.txt.

    Graceful fallback when the summary is empty (pre-Phase-2 runs) — emits a
    one-line note instead of the full section.
    """
    lines: List[str] = []
    lines.append("Early Override Summary (ply 0–1)")
    lines.append("=" * 32)

    by_ply = (eo_summary or {}).get("by_ply") or {}
    if not by_ply:
        lines.append(
            "Not available (sidecars from this run predate the early-override "
            "summary block, or the run had no opening-diagnostics coverage at "
            "ply 0–1)."
        )
        lines.append("")
        return lines

    # Header line: echo the run config so reviewers don't have to cross-reference
    cfg = eo_summary.get("config") or {}
    base_pen = cfg.get("near_corner_penalty")
    base_ply = cfg.get("near_corner_penalty_ply")
    early_pen = cfg.get("near_corner_penalty_early")
    early_plies = cfg.get("near_corner_penalty_early_plies")
    cfg_parts = []
    if base_pen and base_ply:
        cfg_parts.append(f"baseline λ={base_pen} for ply<{base_ply}")
    if early_pen and early_plies:
        cfg_parts.append(f"early λ={early_pen} for ply<{early_plies}")
    if cfg_parts:
        lines.append("Config: " + " | ".join(cfg_parts))
        lines.append("")

    for ply_str in sorted(by_ply.keys(), key=lambda s: int(s)):
        for color in ("red", "black"):
            entry = (by_ply[ply_str] or {}).get(color)
            if not entry:
                continue
            n = entry.get("n", 0)
            eff = entry.get("effective_near_corner_penalty")
            src = entry.get("near_corner_penalty_source", "?")
            eff_str = f"{eff:.3f}" if isinstance(eff, (int, float)) else str(eff)
            lines.append(
                f"  ply={ply_str} {color:>5} n={n:<5} eff_pen={eff_str} ({src})"
            )
            # Mass signal (always present when this section has data)
            raw = entry.get("raw_near_corner_mass", 0.0)
            pen = entry.get("penalized_near_corner_mass", 0.0)
            vis = entry.get("visit_near_corner_mass", 0.0)
            vmp_mass = entry.get("visit_minus_penalized_near_corner_mass", 0.0)
            lines.append(
                f"    mass: raw={raw:.3f} pen={pen:.3f} visit={vis:.3f}  "
                f"Δ(visit−pen)={vmp_mass:+.3f}"
            )
            # Best-by-* signal (only when root-child block was present)
            if "best_by_visit_near_corner_pct" in entry:
                q_mp = entry.get("q_minus_penalized_near_corner", 0.0)
                u_mp = entry.get("u_minus_penalized_near_corner", 0.0)
                s_mp = entry.get("score_minus_penalized_near_corner", 0.0)
                v_mp = entry.get("visit_minus_penalized_near_corner_pct", 0.0)
                tie = entry.get("score_tie_rate")
                tie_str = f" tie_rate={tie:.1%}" if isinstance(tie, (int, float)) else ""
                nn = entry.get("nn_value_mean")
                nn_str = f" nn_v={nn:+.3f}" if isinstance(nn, (int, float)) else ""
                lines.append(
                    f"    best-by near_corner %: visit={entry['best_by_visit_near_corner_pct']:.2f} "
                    f"score={entry['best_by_score_near_corner_pct']:.2f} "
                    f"q={entry['best_by_q_near_corner_pct']:.2f} "
                    f"u={entry['best_by_u_near_corner_pct']:.2f}"
                )
                lines.append(
                    f"    Δ vs penalized_prior: visit={v_mp:+.2f} score={s_mp:+.2f} "
                    f"q={q_mp:+.2f} u={u_mp:+.2f}{tie_str}{nn_str}"
                )
    lines.append("")
    return lines


def format_root_child_report(rcd_summary: dict) -> List[str]:
    """Format the root-child-diagnostics section for report.txt."""
    lines: List[str] = []
    lines.append("Root-Child Diagnostics (ply 0–1)")
    lines.append("=" * 32)

    if not rcd_summary.get("present"):
        lines.append(
            "Not available (sidecars from this run predate root_child_diagnostics, "
            "or the opening-diagnostics window does not cover ply 0–1)."
        )
        lines.append("")
        return lines

    by_ply = rcd_summary.get("by_ply") or {}
    divergences = rcd_summary.get("divergences") or []
    thr = rcd_summary.get("divergence_threshold_pct", _DIVERGENCE_PCT)
    tie_flag = rcd_summary.get("tie_rate_flag", _TIE_RATE_FLAG)

    # Headline — divergences first
    if divergences:
        lines.append(
            f"Divergences (visit vs penalized_prior in near_corner >= {thr:.0%}):"
        )
        for d in divergences:
            lines.append(
                f"  ply={d['ply']} {d['color']:>5} n={d['n']:<5} "
                f"visit−pen={d['visit_minus_penalized_near_corner']:+.2f} "
                f"q−pen={d['q_minus_penalized_near_corner']:+.2f} "
                f"u−pen={d['u_minus_penalized_near_corner']:+.2f} "
                f"score−pen={d['score_minus_penalized_near_corner']:+.2f}  "
                f"=> dominated by {d['dominant_component']}  "
                f"tie_rate={d['score_tie_rate']:.1%}"
            )
        lines.append("")
    else:
        lines.append(
            f"No near-corner divergence above {thr:.0%}. "
            f"Search visits are following the penalized prior baseline."
        )
        lines.append("")

    # Per-ply / per-color detail
    lines.append("Per-ply breakdown (near_corner %):")
    for ply_str in sorted(by_ply.keys(), key=lambda s: int(s)):
        for color in ("red", "black"):
            entry = by_ply[ply_str].get(color)
            if not entry or entry.get("n", 0) <= 0:
                continue
            pp = (entry.get("best_by_penalized_prior_region_pct") or {}).get("near_corner", 0.0)
            vv = (entry.get("best_by_visit_region_pct") or {}).get("near_corner", 0.0)
            ss = (entry.get("best_by_score_region_pct") or {}).get("near_corner", 0.0)
            qq = (entry.get("best_by_q_region_pct") or {}).get("near_corner", 0.0)
            uu = (entry.get("best_by_u_region_pct") or {}).get("near_corner", 0.0)
            tie = entry.get("score_tie_rate", 0.0)
            tie_marker = " *tie*" if tie >= tie_flag else ""
            nn = entry.get("nn_value_mean")
            nn_str = f" nn_v={nn:+.3f}" if nn is not None else ""
            lines.append(
                f"  ply={ply_str} {color:>5} n={entry['n']:<5} "
                f"pen={pp:.2f} visit={vv:.2f} score={ss:.2f} q={qq:.2f} u={uu:.2f} "
                f"tie_rate={tie:.1%}{tie_marker}{nn_str}"
            )
    lines.append("")

    return lines
