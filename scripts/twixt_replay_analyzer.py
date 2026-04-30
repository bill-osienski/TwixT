#!/usr/bin/env python3
"""
Twixt AlphaZero replay analyzer

Reads one or more replay JSON files (or a .zip containing JSONs) and produces:
- Summary CSV/JSON
- Opening-move and opening-sequence frequency tables
- Heatmaps of peg placements by player and by ply buckets
- "Diversity" / "stuck opening" indicators (entropy, top-k concentration, KL drift between windows)
- Checkpoint-backed signals when a checkpoint is resolvable (see below):
    * `replay_probe_scoring` — end-of-chunk sign-correct probe scoring
      against forced probes extracted from the replay range.
    * `value_calibration` — phase-stratified calibration of the network's
      value head against winning-structure / early / mid / late buckets.

Replay schema expected (minimal):
{
  "winner": "red"|"black"|"draw"|...,
  "starting_player": "red"|"black",
  "moves": [{"turn": int, "player": "red"|"black", "row": int, "col": int, ...}, ...],
  "meta": {"board_size": int, "iteration": int, ...}
}
Extra fields are ignored.

Usage examples:
  python twixt_replay_analyzer.py --input /path/to/games_dir --out out/replay_report
  python twixt_replay_analyzer.py --input Replay3.zip --out out/replay_report
  python twixt_replay_analyzer.py --input "logs/games/iter_0192_game_*.json" --out out/report

New CLI flags (all optional, introduced by the probes-and-calibration-closure work):
  --weights <path>                      Explicit checkpoint for probe scoring + calibration.
                                        Skips auto-discovery.
  --checkpoint-dir <path>               Directory to search for auto-discovered checkpoint
                                        (overrides the checkpoints/<single-subdir>/ convention).
  --probe-scoring-disable               Skip replay_probe_scoring entirely.
  --calibration-disable                 Skip value_calibration entirely.
  --calibration-samples-per-bucket N    Target samples per phase-stratified bucket (default 200).
  --calibration-max-total N             Safety cap on total calibration forward passes (default 2000).

Legacy flags retained for backwards compatibility (superseded by --weights + auto-discovery):
  --calibrate                           Old on/off trigger; the new path auto-runs calibration
                                        whenever a checkpoint resolves and --calibration-disable
                                        is absent.
  --calibrate-weights <path>            Old explicit-weights flag. If --weights is absent this
                                        path is honored as a fallback.
  --calibration-sample N                Legacy single-sample count — ignored by the new path.

Checkpoint auto-discovery order (when --weights is omitted):
  1. --calibrate-weights <path>         (legacy fallback)
  2. model_iter_{max(meta.iteration) + 1:04d}.safetensors in:
       a. --checkpoint-dir if given
       b. checkpoints/<single-subdir>/ if exactly one subdir exists under checkpoints/
       c. current working directory
  First existing file wins. If no match is found, the analyzer emits a one-line warning and
  skips the checkpoint-backed sections; all other outputs are unaffected.

New summary keys emitted in summary_<suffix>.json (when a checkpoint is resolved and the
respective -disable flag is absent):
  - replay_probe_scoring — {source, weights, checkpoint_in_channels, selection_rules,
    probe_count, n, sign_correct, sign_correct_pct, median_abs_v, by_category}.
    Sourced from extract_forced_probes_from_games(replays) + run_forced_probes_inline.
  - value_calibration — {weights, samples_per_bucket_target, max_total,
    natural_distribution, sampled_distribution, stratified, overall_note, aggregate}.
    Sourced from score_samples_against_checkpoint. Stratified: per-bucket cap honored,
    alphabetical halt when max_total binds, no redistribution across buckets.

Companion CSVs emitted under --out:
  - replay_probe_per_probe_<suffix>.csv  — one row per scored probe.
  - value_calibration_by_bucket_<suffix>.csv — one row per position-type bucket.

Notes:
- Heatmaps default to board_size from meta.board_size; fallback to --board-size.
- "Corner" is any move with row/col in {0,1,board-2,board-1}. You can tune with --edge-pad.

Reference: docs/superpowers/specs/2026-04-21-probes-and-calibration-closure-design.md
           docs/analysis-metrics-guide.md (Analyzer knobs section)
"""
from __future__ import annotations

import argparse, csv, glob, io, json, math, os, re, sys, zipfile
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple, Optional

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
        # Phase 1: root-child diagnostics (optional — absent on older sidecars)
        extract_sidecar_root_child_diagnostics,
        aggregate_sidecar_root_child_diagnostics,
        aggregate_replay_root_child_diagnostics,
        write_root_child_by_ply_csv,
        write_root_child_per_game_csv,
        build_root_child_summary,
        format_root_child_report,
        # Phase 2: early-override summary (optional — pre-Phase-2 data yields empty)
        build_early_override_summary,
        format_early_override_report,
    )
    _HAS_OD_ANALYZER = True
except ImportError:
    _HAS_OD_ANALYZER = False

# Phase 1 (connectivity-retrain): connectivity diagnostics + value calibration.
# Import lazily; fall back gracefully if the modules are unavailable so the
# analyzer still runs on environments that haven't pulled the new modules.
try:
    # Resolve via repo-root package path so the analyzer works when run as a
    # bare script (sys.path has `scripts/` rather than the repo root).
    import sys as _sys
    import os as _os
    _REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _REPO_ROOT not in _sys.path:
        _sys.path.insert(0, _REPO_ROOT)
    from scripts.GPU.alphazero.connectivity_diagnostics import (
        aggregate_connectivity_by_ply,
        compute_position_connectivity,
    )
    from scripts.GPU.alphazero.value_calibration import (
        aggregate_calibration,
        classify_position,
    )
    _HAS_PHASE1_DIAG = True
except ImportError:
    _HAS_PHASE1_DIAG = False

import numpy as np
try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except Exception:
    plt = None  # type: ignore
    _HAS_MPL = False


TIER_NAMES = ("forced", "strong_advantage")


def _read_tier_summary(sc: dict, tier: str):
    """Read a per-iter sidecar's summary for `tier`. Prefers the new
    `probe_summary.<tier>` shape; falls back to the legacy
    `forced_probe_summary` field for tier == "forced".
    """
    ps = sc.get("probe_summary") or {}
    if tier in ps and ps[tier] is not None:
        return ps[tier]
    if tier == "forced":
        return sc.get("forced_probe_summary")
    return None


# -----------------------------
# Loading utilities
# -----------------------------

def _resolve_checkpoint_path(args, replays: List[dict]) -> Optional[str]:
    """Resolve a checkpoint path for probe scoring + calibration.

    Resolution order (spec §6.2):
      1. args.weights if given
      2. args.calibrate_weights (legacy fallback)
      3. Auto-discover model_iter_{max(meta.iteration) + 1:04d}.safetensors in:
         a. args.checkpoint_dir if given
         b. checkpoints/<single-subdir>/ if exactly one subdir exists
         c. current working directory
      4. Return None if nothing found.
    """
    import os
    from pathlib import Path

    # 1. Explicit --weights wins.
    explicit = getattr(args, "weights", None)
    if explicit:
        return explicit if os.path.exists(explicit) else None

    # 2. Legacy --calibrate-weights fallback.
    legacy = getattr(args, "calibrate_weights", None)
    if legacy:
        return legacy if os.path.exists(legacy) else None

    # 3. Auto-discover from replays.
    if not replays:
        return None
    iters = [r.get("meta", {}).get("iteration") for r in replays
             if isinstance(r.get("meta", {}).get("iteration"), int)]
    if not iters:
        return None
    max_iter = max(iters)
    target_name = f"model_iter_{max_iter + 1:04d}.safetensors"

    candidate_dirs: List[str] = []
    explicit_dir = getattr(args, "checkpoint_dir", None)
    if explicit_dir:
        candidate_dirs.append(explicit_dir)
    else:
        # Single-subdir convention.
        ckpt_root = Path("checkpoints")
        if ckpt_root.is_dir():
            subdirs = [p for p in ckpt_root.iterdir() if p.is_dir()]
            if len(subdirs) == 1:
                candidate_dirs.append(str(subdirs[0]))
        candidate_dirs.append(".")  # cwd fallback

    for d in candidate_dirs:
        full = os.path.join(d, target_name)
        if os.path.exists(full):
            return full
    return None


def _derive_out_suffix(out_dir: str, override: Optional[str] = None) -> str:
    """Compute the filename suffix applied to all output artifacts.

    Convention: the analyzer writes one set of files per output dir. Naming
    them with the iteration range (e.g. `summary_945-949.json`) makes it
    trivial to keep multiple runs side-by-side and grep by range.

    Precedence:
      - If `override` is provided (from `--out-suffix`), use it verbatim
        (stripped of surrounding whitespace and underscores). An empty
        override means "no suffix" (files land as `summary.json`, etc.).
      - Otherwise derive from the basename of out_dir. A trailing `_Replay`
        (case-insensitive) is stripped so `Replays/945-949_Replay/` yields
        suffix `945-949` — matching the user's historical naming.

    Returns an empty string to mean "no suffix".
    """
    if override is not None:
        return override.strip().strip("_")
    base = os.path.basename(os.path.normpath(out_dir)) if out_dir else ""
    if base.lower().endswith("_replay"):
        base = base[: -len("_replay")]
    return base


def _suffixed(name: str, ext: str, suffix: str) -> str:
    """Compose a filename as `{name}_{suffix}.{ext}` (or `{name}.{ext}` if no suffix)."""
    if suffix:
        return f"{name}_{suffix}.{ext}"
    return f"{name}.{ext}"


def _iter_json_blobs_from_path(path: str) -> Iterable[Tuple[str, bytes]]:
    """
    Yield (name, bytes) for JSON files found at:
      - a single .json file
      - a directory (recursively)
      - a glob pattern
      - a .zip containing .json files
    """
    if os.path.isfile(path) and path.lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            for n in zf.namelist():
                if n.lower().endswith(".json") and not n.endswith("/"):
                    yield n, zf.read(n)
        return

    # If it's a single json file
    if os.path.isfile(path) and path.lower().endswith(".json"):
        with open(path, "rb") as f:
            yield os.path.basename(path), f.read()
        return

    # Directory
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fn in files:
                if fn.lower().endswith(".json"):
                    fp = os.path.join(root, fn)
                    with open(fp, "rb") as f:
                        yield fp, f.read()
        return

    # Glob
    for fp in glob.glob(path):
        if os.path.isfile(fp) and fp.lower().endswith(".json"):
            with open(fp, "rb") as f:
                yield fp, f.read()


def load_replays(inputs: List[str]) -> List[dict]:
    replays: List[dict] = []
    for p in inputs:
        for name, blob in _iter_json_blobs_from_path(p):
            if re.search(r"iter_\d{4,}_stats\.json$", name):
                continue   # skip stats sidecars
            try:
                obj = json.loads(blob.decode("utf-8"))
                obj["_source_name"] = name
                replays.append(obj)
            except Exception as e:
                print(f"[WARN] failed to parse {name}: {e}")
    return replays


def load_sidecars(inputs: List[str]) -> Dict[int, dict]:
    """Load iter_NNNN_stats.json sidecar files from input paths.
    Returns dict mapping iteration number -> sidecar data.
    Precedence on duplicates: last discovered file wins.
    """
    _pat = re.compile(r"iter_(\d{4,})_stats\.json$")
    sidecars: Dict[int, dict] = {}
    for p in inputs:
        paths: List[str] = []
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in sorted(files):
                    if _pat.search(fn):
                        paths.append(os.path.join(root, fn))
        elif os.path.isfile(p) and _pat.search(p):
            paths.append(p)
        elif "*" in p or "?" in p:
            for fp in sorted(glob.glob(p)):
                if os.path.isfile(fp) and _pat.search(fp):
                    paths.append(fp)
        for fp in paths:
            m = _pat.search(os.path.basename(fp))
            if m:
                it = int(m.group(1))
                if it in sidecars:
                    print(f"[WARN] duplicate sidecar for iteration {it}: {fp} overwriting previous")
                try:
                    with open(fp, encoding="utf-8") as f:
                        sidecars[it] = json.load(f)
                except Exception as e:
                    print(f"[WARN] failed to parse sidecar {fp}: {e}")
    return sidecars


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

    # --- Worker balance accumulators ---
    by_worker = {}              # str(worker_id) → {games, n_moves_total, wall_time_total_s}
    in_process_count = 0

    # --- Accumulators for distribution blocks ---
    n_moves_arr = []
    wall_time_arr = []
    final_root_value_arr = []
    final_top1_share_arr = []
    leaf_evals_arr = []
    backups_arr = []
    nn_batches_arr = []
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

    # --- Build output ---
    return {
        "n_games_total": n_games_total,
        "n_games_with_any_stats": n_games_with_any_stats,
        "coverage": coverage,
        "game_length": game_length,
        "outcomes": outcomes,
        "wall_time_s": wall_time_block,
        "worker_balance": {
            "by_worker": by_worker,
            "in_process_count": in_process_count,
            "max_min_wall_time_ratio": max_min_wall_time_ratio,
            "max_min_games_ratio": max_min_games_ratio,
            "wall_time_cv": wall_time_cv,
        },
        "final_root_value": final_root_value_block,
        "final_top1_share": final_top1_share_block,
        "compute_per_game": compute_per_game_block,
    }


def aggregate_sidecars(sidecars: Dict[int, dict]) -> dict:
    """Aggregate per-iteration sidecar dicts into one summary.

    Always normalizes (even for single sidecar) so field semantics
    are consistent: summed counts, weighted avg_plies, latest-snapshot
    for balance/percentile fields.
    """
    if not sidecars:
        return {}

    agg = {
        "iteration_min": min(sidecars.keys()),
        "iteration_max": max(sidecars.keys()),
        "iterations_count": len(sidecars),
        "games_per_iter": 0,   # latest iteration's value
        "games_total": 0,      # sum across all iterations
        # Summed
        "results": {"red_wins": 0, "black_wins": 0, "draws": 0},
        "draw_breakdown": {"timeout": 0, "board_full": 0, "state_cap": 0, "unknown": 0},
        "termination": {"win": 0, "resign": 0, "adjudicated": 0, "timeout": 0},
        "termination_by_winner": {
            "red": {"win": 0, "resign": 0, "adjudicated": 0},
            "black": {"win": 0, "resign": 0, "adjudicated": 0},
            "draw": {"timeout": 0},
        },
        "targets": {"z_pos": 0, "z_zero": 0, "z_neg": 0},
        "adjudication": {
            "attempts": 0, "adjudicated": 0, "red_wins": 0, "black_wins": 0, "remaining_timeouts": 0,
            "blocks": {"ply": 0, "threshold": 0, "visits": 0, "top1": 0},
            "stats": {},  # latest snapshot
        },
        "resign": {"total": 0, "by_red": 0, "by_black": 0},
        "resign_gate": {
            "checks": 0, "red_checks": 0, "black_checks": 0,
            "value_hits": 0, "red_value_hits": 0, "black_value_hits": 0,
            "blocked_by_top1": 0, "red_blocked_by_top1": 0, "black_blocked_by_top1": 0,
            "eligible_hits": 0, "red_eligible_hits": 0, "black_eligible_hits": 0,
            "top1_share_on_value_hits": {},  # latest snapshot
            "min_top1_share": 0.0,           # latest snapshot
        },
        "compute": {"buffer_size": 0, "backups": 0, "leaf_evals": 0, "nn_batches": 0},
        # Weighted
        "avg_plies": 0.0,
        # Latest snapshot
        "balance": {},
        # Phase 2 (2026-04-19 connectivity retrain): per-iter value-head sign-agree
        # by connectivity bucket + inline forced-probe summary. Collected as
        # per-iter rows (for CSV + trend reports) + latest snapshot.
        "sanity_by_connectivity_by_iter": [],   # list of per-iter rows
        "sanity_by_connectivity_latest": {},    # latest iter's snapshot dict
        # Phase 2: per-tier inline probe aggregates. Initialized for every tier
        # in TIER_NAMES so summary.json and downstream consumers see a stable
        # set of keys regardless of whether any sidecar carries the data yet.
        **{f"{tier}_probe_by_iter": [] for tier in TIER_NAMES},
        **{f"{tier}_probe_latest": {} for tier in TIER_NAMES},
    }

    total_plies_w = 0.0
    total_games_w = 0
    latest_it = max(sidecars.keys())

    for it in sorted(sidecars.keys()):
        sc = sidecars[it]
        gpi = sc.get("games_per_iter", 0)
        agg["games_total"] += gpi

        for key in ("results", "draw_breakdown", "termination", "targets"):
            for k, v in sc.get(key, {}).items():
                if isinstance(v, (int, float)):
                    agg[key][k] = agg[key].get(k, 0) + v
        for winner in ("red", "black", "draw"):
            for reason, count in sc.get("termination_by_winner", {}).get(winner, {}).items():
                agg["termination_by_winner"][winner][reason] = agg["termination_by_winner"][winner].get(reason, 0) + count
        for k in ("total", "by_red", "by_black"):
            agg["resign"][k] += sc.get("resign", {}).get(k, 0)
        adj = sc.get("adjudication", {})
        for k in ("attempts", "adjudicated", "red_wins", "black_wins", "remaining_timeouts"):
            agg["adjudication"][k] += adj.get(k, 0)
        for k in ("ply", "threshold", "visits", "top1"):
            agg["adjudication"]["blocks"][k] += adj.get("blocks", {}).get(k, 0)
        rg = sc.get("resign_gate", {})
        for k in ("checks", "red_checks", "black_checks", "value_hits", "red_value_hits",
                   "black_value_hits", "blocked_by_top1", "red_blocked_by_top1",
                   "black_blocked_by_top1", "eligible_hits", "red_eligible_hits", "black_eligible_hits"):
            agg["resign_gate"][k] += rg.get(k, 0)
        comp = sc.get("compute", {})
        for k in ("backups", "leaf_evals", "nn_batches"):
            agg["compute"][k] += comp.get(k, 0)

        total_plies_w += sc.get("avg_plies", 0.0) * gpi
        total_games_w += gpi

        # Phase 2: per-iter connectivity-bucketed sanity (optional — absent on
        # pre-Phase-2 sidecars, None on 24-channel checkpoints).
        sbc = sc.get("sanity_by_connectivity")
        if sbc:
            ws = sbc.get("winning_structure", {}) or {}
            nws = sbc.get("no_winning_structure", {}) or {}
            agg["sanity_by_connectivity_by_iter"].append({
                "iteration": it,
                "winning_n": ws.get("n"),
                "winning_sign_agree": ws.get("sign_agree"),
                "winning_median_abs_v": ws.get("median_abs_v"),
                "no_winning_n": nws.get("n"),
                "no_winning_sign_agree": nws.get("sign_agree"),
                "no_winning_median_abs_v": nws.get("median_abs_v"),
                "winning_size_threshold": sbc.get("winning_size_threshold"),
            })

        # Phase 2: per-iter inline probes, parameterized over all known tiers.
        # Reads `probe_summary.<tier>` first (forward path) and falls back to
        # the legacy `forced_probe_summary` for tier == "forced".
        for tier in TIER_NAMES:
            tps = _read_tier_summary(sc, tier)
            if not tps:
                continue
            agg[f"{tier}_probe_by_iter"].append({
                "iteration": it,
                "n": tps.get("n"),
                "n_skipped_size": tps.get("n_skipped_size"),
                "sign_correct": tps.get("sign_correct"),
                "sign_correct_pct": tps.get("sign_correct_pct"),
                "median_abs_v": tps.get("median_abs_v"),
                "delta_sign_correct_pct": tps.get("delta_sign_correct_pct"),
                "delta_median_abs_v": tps.get("delta_median_abs_v"),
                "rolling5_sign_correct_pct": tps.get("rolling5_sign_correct_pct"),
                "rolling5_median_abs_v": tps.get("rolling5_median_abs_v"),
            })

        if it == latest_it:
            agg["games_per_iter"] = gpi
            agg["balance"] = {"window": sc.get("balance", {}).get("window", "n/a")}
            agg["compute"]["buffer_size"] = comp.get("buffer_size", 0)
            agg["adjudication"]["stats"] = adj.get("stats", {})
            agg["resign_gate"]["top1_share_on_value_hits"] = rg.get("top1_share_on_value_hits", {})
            agg["resign_gate"]["min_top1_share"] = rg.get("min_top1_share", 0.0)
            # Phase 2: latest-iter snapshots of the new blocks
            if sbc:
                agg["sanity_by_connectivity_latest"] = sbc
            for tier in TIER_NAMES:
                tps_latest = _read_tier_summary(sc, tier)
                if tps_latest:
                    agg[f"{tier}_probe_latest"] = tps_latest

    agg["avg_plies"] = round(total_plies_w / total_games_w, 1) if total_games_w > 0 else 0.0

    # Recompute balance percentages from aggregated totals (not snapshot)
    _rw = agg["results"]["red_wins"]
    _bw = agg["results"]["black_wins"]
    _dw = agg["results"]["draws"]
    _decisive = _rw + _bw
    _total = _rw + _bw + _dw
    agg["balance"]["red_pct"] = round(_rw / _decisive * 100, 1) if _decisive > 0 else 0.0
    agg["balance"]["black_pct"] = round(_bw / _decisive * 100, 1) if _decisive > 0 else 0.0
    agg["balance"]["draw_pct"] = round(_dw / _total * 100, 1) if _total > 0 else 0.0
    agg["balance"]["decisive_games"] = _decisive

    return agg


# -----------------------------
# Phase 4: Replay-cap helpers (sidecar `replay_cap` block)
# -----------------------------
#
# The trainer writes a per-iteration `replay_cap` block when per-game replay
# contribution capping is enabled (and emits a disabled marker block when it is
# not). The block is fully optional — older sidecars won't have it. These
# helpers extract, aggregate across iterations, produce a CSV, and format a
# short report section.


def extract_sidecar_replay_cap(sidecars: Dict[int, dict]) -> Dict[int, dict]:
    """Extract `replay_cap` blocks from sidecars (iteration -> block).

    Older sidecars without the block are silently skipped.
    """
    out: Dict[int, dict] = {}
    for it, sc in sidecars.items():
        blk = sc.get("replay_cap")
        if blk and isinstance(blk, dict):
            out[it] = blk
    return out


def _bucket_label(edges: List[int], idx: int) -> str:
    """Pretty label for a length bucket given its edges list.

    edges = [40, 80, 120, 160, 200] → labels for 6 buckets:
        "0-39", "40-79", "80-119", "120-159", "160-199", "200+"
    """
    if idx == 0:
        return f"0-{edges[0]-1}" if edges else "0+"
    if idx >= len(edges):
        return f"{edges[-1]}+"
    return f"{edges[idx-1]}-{edges[idx]-1}"


def aggregate_replay_cap(rcap_by_iter: Dict[int, dict]) -> dict:
    """Roll replay-cap blocks up into a single dict for the report + summary.

    Sums counts across iterations; takes the latest iteration's cap config
    (enabled / max / endgame_keep) as the "current" setting — this mirrors how
    `aggregate_sidecars` treats latest-snapshot fields.
    """
    if not rcap_by_iter:
        return {}

    # Find a bucket-edges vector (prefer the latest iteration's)
    latest_it = max(rcap_by_iter.keys())
    latest = rcap_by_iter[latest_it]
    edges = list((latest.get("by_length_bucket") or {}).get("edges_ply") or [])
    n_buckets = len(edges) + 1 if edges else 0

    total_games = 0
    total_games_capped = 0
    total_orig = 0
    total_kept = 0
    any_enabled = False
    bucket_games = [0] * n_buckets
    bucket_orig = [0] * n_buckets
    bucket_kept = [0] * n_buckets
    edge_variants: set = set()
    # Phase 1 (2026-04-19): termination-type + length-split accumulators.
    total_positions_by_termination = {"win": 0, "resign": 0, "adjudicated": 0, "timeout": 0}
    total_positions_in_short_games = 0
    total_positions_in_long_games = 0

    for it in sorted(rcap_by_iter.keys()):
        blk = rcap_by_iter[it]
        if blk.get("enabled"):
            any_enabled = True
        total_games += int(blk.get("games_total", 0) or 0)
        total_games_capped += int(blk.get("games_capped", 0) or 0)
        total_orig += int(blk.get("total_positions_original", 0) or 0)
        total_kept += int(blk.get("total_positions_kept", 0) or 0)
        bt = blk.get("positions_by_termination") or {}
        for term in total_positions_by_termination:
            total_positions_by_termination[term] += int(bt.get(term, 0) or 0)
        total_positions_in_short_games += int(blk.get("positions_in_short_games", 0) or 0)
        total_positions_in_long_games += int(blk.get("positions_in_long_games", 0) or 0)
        blb = blk.get("by_length_bucket") or {}
        blb_edges = tuple(blb.get("edges_ply") or ())
        if blb_edges:
            edge_variants.add(blb_edges)
        g = blb.get("games") or []
        o = blb.get("positions_original") or []
        k = blb.get("positions_kept") or []
        # Align buckets with the latest-iteration edge vector. If edges shifted
        # across iterations, drop the mismatched ones and flag it.
        if tuple(blb_edges) == tuple(edges) and len(g) == n_buckets:
            for i in range(n_buckets):
                bucket_games[i] += int(g[i] or 0)
                bucket_orig[i] += int(o[i] or 0)
                bucket_kept[i] += int(k[i] or 0)

    edges_mismatch = len(edge_variants) > 1

    return {
        "enabled_latest": bool(latest.get("enabled")),
        "any_enabled": any_enabled,
        "max_positions_per_game_latest": int(latest.get("max_positions_per_game", 0) or 0),
        "endgame_keep_positions_latest": int(latest.get("endgame_keep_positions", 0) or 0),
        "edges_mismatch_across_iters": edges_mismatch,
        "games_total": total_games,
        "games_capped": total_games_capped,
        "capped_rate": round(total_games_capped / total_games, 4) if total_games else 0.0,
        "total_positions_original": total_orig,
        "total_positions_kept": total_kept,
        "kept_fraction": round(total_kept / total_orig, 4) if total_orig else 1.0,
        "total_positions_by_termination": total_positions_by_termination,
        "total_positions_in_short_games": total_positions_in_short_games,
        "total_positions_in_long_games": total_positions_in_long_games,
        "by_length_bucket": {
            "edges_ply": edges,
            "labels": [_bucket_label(edges, i) for i in range(n_buckets)],
            "games": bucket_games,
            "positions_original": bucket_orig,
            "positions_kept": bucket_kept,
            "kept_fraction_per_bucket": [
                round(bucket_kept[i] / bucket_orig[i], 4) if bucket_orig[i] else 1.0
                for i in range(n_buckets)
            ],
        },
    }


def write_replay_cap_by_iter_csv(
    out_dir: str,
    rcap_by_iter: Dict[int, dict],
    suffix: str = "",
) -> Optional[str]:
    """Write replay_cap_by_iter.csv — one row per iteration.

    Args:
        suffix: if non-empty, output is `replay_cap_by_iter_{suffix}.csv`
                (enables side-by-side comparison of multiple ranges).

    Returns the file path, or None if no iteration carries a replay_cap block
    (older run — caller can skip the section silently).
    """
    if not rcap_by_iter:
        return None

    # Use the latest iteration's bucket edges to decide column layout (the
    # aggregator already flags cross-iteration edge drift; we just keep a
    # consistent header).
    latest_it = max(rcap_by_iter.keys())
    latest = rcap_by_iter[latest_it]
    edges = list((latest.get("by_length_bucket") or {}).get("edges_ply") or [])
    n_buckets = len(edges) + 1 if edges else 0
    labels = [_bucket_label(edges, i) for i in range(n_buckets)]

    header = [
        "iteration", "enabled", "max_positions_per_game", "endgame_keep_positions",
        "games_total", "games_capped", "capped_rate",
        "total_positions_original", "total_positions_kept",
        "mean_positions_original", "mean_positions_kept", "kept_fraction",
    ]
    for lb in labels:
        header.append(f"bucket_games_{lb}")
    for lb in labels:
        header.append(f"bucket_orig_{lb}")
    for lb in labels:
        header.append(f"bucket_kept_{lb}")

    path = os.path.join(out_dir, _suffixed("replay_cap_by_iter", "csv", suffix))
    rows = []
    for it in sorted(rcap_by_iter.keys()):
        blk = rcap_by_iter[it]
        blb = blk.get("by_length_bucket") or {}
        g = blb.get("games") or []
        o = blb.get("positions_original") or []
        k = blb.get("positions_kept") or []
        # Only line-up buckets when this iter's edges match the header edges
        aligned = tuple(blb.get("edges_ply") or ()) == tuple(edges) and len(g) == n_buckets
        row = [
            it,
            int(bool(blk.get("enabled"))),
            int(blk.get("max_positions_per_game", 0) or 0),
            int(blk.get("endgame_keep_positions", 0) or 0),
            int(blk.get("games_total", 0) or 0),
            int(blk.get("games_capped", 0) or 0),
            blk.get("capped_rate", ""),
            int(blk.get("total_positions_original", 0) or 0),
            int(blk.get("total_positions_kept", 0) or 0),
            blk.get("mean_positions_original", ""),
            blk.get("mean_positions_kept", ""),
            blk.get("kept_fraction", ""),
        ]
        for i in range(n_buckets):
            row.append(g[i] if aligned and i < len(g) else "")
        for i in range(n_buckets):
            row.append(o[i] if aligned and i < len(o) else "")
        for i in range(n_buckets):
            row.append(k[i] if aligned and i < len(k) else "")
        rows.append(row)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def write_replay_probe_per_probe_csv(
    out_dir: str, suffix: str, probes: list, scoring_result: dict,
) -> None:
    """Emit replay_probe_per_probe_<suffix>.csv (one row per probe)."""
    import csv
    path = os.path.join(out_dir, _suffixed("replay_probe_per_probe", "csv", suffix))
    nn_values = scoring_result.get("nn_values") or []
    expected_signs = scoring_result.get("expected_signs") or []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "id", "category", "source_game", "source_ply",
            "expected_value_sign", "nn_value", "sign_correct", "nn_magnitude",
        ])
        w.writeheader()
        for p, v, s in zip(probes, nn_values, expected_signs):
            correct = int((s > 0 and v > 0) or (s < 0 and v < 0)
                          or (s == 0 and abs(v) < 0.1))
            w.writerow({
                "id": p["id"],
                "category": p["category"],
                "source_game": p["source_game"],
                "source_ply": p["source_ply"],
                "expected_value_sign": s,
                "nn_value": round(v, 4),
                "sign_correct": correct,
                "nn_magnitude": round(abs(v), 4),
            })


def write_value_calibration_by_bucket_csv(
    out_dir: str, suffix: str, cal_summary: dict,
) -> None:
    """Emit value_calibration_by_bucket_<suffix>.csv (one row per bucket)."""
    import csv
    path = os.path.join(out_dir, _suffixed("value_calibration_by_bucket", "csv", suffix))
    natural = cal_summary.get("natural_distribution") or {}
    sampled = cal_summary.get("sampled_distribution") or {}
    buckets_stats = (cal_summary.get("aggregate") or {}).get("buckets") or {}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "bucket", "natural_count", "sampled_count",
            "sign_agree", "mse", "pred_mean", "outcome_mean",
        ])
        w.writeheader()
        for bucket in sorted(natural.keys()):
            stats = buckets_stats.get(bucket, {})
            w.writerow({
                "bucket": bucket,
                "natural_count": natural.get(bucket, 0),
                "sampled_count": sampled.get(bucket, 0),
                "sign_agree": stats.get("sign_agree", ""),
                "mse": stats.get("mse", ""),
                "pred_mean": stats.get("pred_mean", ""),
                "outcome_mean": stats.get("outcome_mean", ""),
            })


def format_replay_cap_report(rcap_summary: dict) -> List[str]:
    """Format a concise replay-cap section for report.txt.

    Empty input (no sidecar had the block) → one "not available" line.
    """
    lines: List[str] = []
    lines.append("Replay-cap Engagement (Phase 4)")
    lines.append("=" * 31)
    if not rcap_summary:
        lines.append(
            "Not available (sidecars from this run predate the replay_cap block)."
        )
        lines.append("")
        return lines

    if not rcap_summary.get("any_enabled"):
        lines.append(
            "Replay cap was disabled across all iterations in this range. "
            "Every game contributed every position — long games still dominate."
        )
        lines.append("")
        return lines

    lines.append(
        f"Cap (latest iter): max_positions_per_game="
        f"{rcap_summary['max_positions_per_game_latest']}, "
        f"endgame_keep={rcap_summary['endgame_keep_positions_latest']}"
    )
    lines.append(
        f"Totals: games={rcap_summary['games_total']:,} "
        f"capped={rcap_summary['games_capped']:,} "
        f"({rcap_summary['capped_rate']:.1%}) | "
        f"positions produced={rcap_summary['total_positions_original']:,} "
        f"kept={rcap_summary['total_positions_kept']:,} "
        f"({rcap_summary['kept_fraction']:.1%})"
    )

    blb = rcap_summary.get("by_length_bucket") or {}
    labels = blb.get("labels") or []
    games = blb.get("games") or []
    orig = blb.get("positions_original") or []
    kept = blb.get("positions_kept") or []
    kfpb = blb.get("kept_fraction_per_bucket") or []
    if labels:
        lines.append("By game-length bucket (ply count):")
        lines.append(
            "  " + f"{'bucket':>12}  {'games':>8}  {'orig pos':>10}  "
                   f"{'kept pos':>10}  {'kept_frac':>10}"
        )
        for i, lb in enumerate(labels):
            g = games[i] if i < len(games) else 0
            o = orig[i] if i < len(orig) else 0
            k = kept[i] if i < len(kept) else 0
            kf = kfpb[i] if i < len(kfpb) else 1.0
            lines.append(
                "  " + f"{lb:>12}  {g:>8,}  {o:>10,}  {k:>10,}  "
                       f"{kf:>10.1%}"
            )
    if rcap_summary.get("edges_mismatch_across_iters"):
        lines.append(
            "  NOTE: bucket edges differ across iterations in this range — "
            "bucket rows aggregate only iterations matching the latest edge vector."
        )
    lines.append("")
    return lines


# -----------------------------
# Feature extraction
# -----------------------------

def _board_size(replay: dict, override: Optional[int]) -> int:
    m = replay.get("meta") or {}
    bs = m.get("board_size")
    if isinstance(bs, int) and bs > 0:
        return bs
    if override is not None:
        return int(override)
    # Fall back to 24 (your common setting)
    return 24


def _winner(replay: dict) -> str:
    w = (replay.get("winner") or "").lower().strip()
    if w in ("red", "black", "draw"):
        return w
    return w or "unknown"


def _starting_player(replay: dict) -> str:
    s = (replay.get("starting_player") or (replay.get("meta") or {}).get("starting_player") or "").lower().strip()
    return s if s in ("red","black") else "unknown"


def _moves(replay: dict) -> List[dict]:
    ms = replay.get("moves") or []
    if not isinstance(ms, list):
        return []
    return [m for m in ms if isinstance(m, dict) and "row" in m and "col" in m and "player" in m]


def _edge_or_corner(row: int, col: int, n: int, pad: int) -> Tuple[bool, bool]:
    # corner-ish: within pad of BOTH edges
    corner = (row < pad or row >= n-pad) and (col < pad or col >= n-pad)
    edge = (row < pad or row >= n-pad or col < pad or col >= n-pad)
    return edge, corner



def _is_exact_edge(row: int, col: int, n: int) -> bool:
    return row == 0 or row == n - 1 or col == 0 or col == n - 1


def _is_edge_band(row: int, col: int, n: int, band: int) -> bool:
    # band=2 => r in {0,1} or {n-2,n-1} or c in {0,1} or {n-2,n-1}
    return (row < band) or (row >= n - band) or (col < band) or (col >= n - band)


def _is_near_corner(row: int, col: int, n: int, radius: int) -> bool:
    # Chebyshev distance to any corner <= radius
    corners = ((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1))
    for rr, cc in corners:
        if max(abs(row - rr), abs(col - cc)) <= radius:
            return True
    return False


def _ply_buckets(spec: str) -> List[Tuple[int,int,str]]:
    """
    Parse bucket spec like: "1-4,5-10,11-20,21-999"
    Returns (lo,hi,label) inclusive bounds.
    """
    buckets = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
        if not m:
            raise ValueError(f"Bad bucket '{part}' (expected like 1-4)")
        lo = int(m.group(1)); hi = int(m.group(2))
        if lo <= 0 or hi < lo:
            raise ValueError(f"Bad bucket range '{part}'")
        buckets.append((lo, hi, f"{lo}-{hi}"))
    return buckets


def _bucket_for_ply(ply: int, buckets: List[Tuple[int,int,str]]) -> str:
    for lo,hi,label in buckets:
        if lo <= ply <= hi:
            return label
    return "other"


def _opening_sequence_key(moves: List[dict], k: int) -> Tuple[Tuple[str,int,int], ...]:
    seq = []
    for i in range(min(k, len(moves))):
        m = moves[i]
        seq.append((m["player"], int(m["row"]), int(m["col"])))
    return tuple(seq)


def _pos_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


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


# -----------------------------
# Phase 1 (connectivity-retrain) report formatters
# -----------------------------

def format_connectivity_diagnostics_report(summary: dict, rows: list) -> List[str]:
    """Render the connectivity-diagnostics section for the text report.

    Emits a small preview (first few rows) and points at the CSV for the full
    table. Always renders a section header so section presence is a stable
    grep target (e.g. the E2E smoke test checks for "Connectivity Diagnostics").
    """
    lines = []
    lines.append("Connectivity Diagnostics (Phase 1)")
    lines.append("=" * 40)
    if not rows:
        lines.append("  (not available — Phase 1 diagnostics require connectivity_diagnostics module)")
        lines.append("")
        return lines
    lines.append(f"  Rows: {len(rows)}")
    # Print first few rows as a summary; keep it small
    for row in rows[:8]:
        lines.append(f"  - {row}")
    if len(rows) > 8:
        lines.append(f"  ... {len(rows) - 8} more rows in connectivity_by_ply.csv")
    lines.append("")
    return lines


def format_sanity_by_connectivity_report(by_iter: list, latest: dict) -> List[str]:
    """Render the per-iter value-head sign-agree by connectivity-bucket section.

    Source: `sanity_by_connectivity` block written by the trainer per iteration
    (from Phase 2 connectivity-retrain). Shows latest snapshot + trend over
    recent iters. Degrades silently if no sidecar carries the block (pre-Phase-2
    data or disabled inline eval).
    """
    lines = []
    lines.append("Value Head Sanity by Connectivity Bucket (Phase 2)")
    lines.append("=" * 55)
    if not by_iter and not latest:
        lines.append("  (not available — no sanity_by_connectivity data in sidecars)")
        lines.append("")
        return lines

    # Latest-snapshot summary
    ws = (latest or {}).get("winning_structure", {}) or {}
    nws = (latest or {}).get("no_winning_structure", {}) or {}
    thr = (latest or {}).get("winning_size_threshold", 8)
    lines.append(f"  Latest iter (threshold: largest_component>={thr} OR n_goal_touching>=2):")
    def _fmt_bucket(b):
        n = b.get("n", 0)
        if n == 0:
            return "(n=0)"
        sa = b.get("sign_agree")
        mv = b.get("median_abs_v")
        sa_s = f"{sa:.1%}" if sa is not None else "n/a"
        mv_s = f"{mv:.3f}" if mv is not None else "n/a"
        return f"(n={n}): sign_agree={sa_s}, median |v|={mv_s}"
    lines.append(f"    winning_structure    {_fmt_bucket(ws)}")
    lines.append(f"    no_winning_structure {_fmt_bucket(nws)}")

    # Trend across iters (last 10 in a compact table)
    if by_iter:
        trend = by_iter[-10:]
        lines.append("")
        lines.append(f"  Trend (last {len(trend)} iters):")
        lines.append(f"    {'iter':>5} {'win_n':>6} {'win_SA':>8} {'win_|v|':>8} {'nw_n':>5} {'nw_SA':>8} {'nw_|v|':>8}")
        for row in trend:
            def _p(x, pct=False):
                if x is None:
                    return "n/a"
                return f"{x:.1%}" if pct else f"{x:.3f}"
            lines.append(
                f"    {row['iteration']:>5} "
                f"{row.get('winning_n') or 0:>6} "
                f"{_p(row.get('winning_sign_agree'), pct=True):>8} "
                f"{_p(row.get('winning_median_abs_v')):>8} "
                f"{row.get('no_winning_n') or 0:>5} "
                f"{_p(row.get('no_winning_sign_agree'), pct=True):>8} "
                f"{_p(row.get('no_winning_median_abs_v')):>8}"
            )
        lines.append(f"  ... full per-iter table: sanity_by_connectivity_by_iter.csv")
    lines.append("")
    return lines


def format_tier_probe_report(tier: str, by_iter: list, latest: dict) -> List[str]:
    """Render the per-iter inline probe section for `tier` (forced or
    strong_advantage). Same shape as the previous forced-only formatter.
    """
    title = {
        "forced": "Forced-Tier Probe Sign-Agree (Phase 2)",
        "strong_advantage": "Strong-Advantage Probe Sign-Agree (deep-MCTS labeled)",
    }.get(tier, f"{tier} Probe Sign-Agree")

    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    if not by_iter and not latest:
        lines.append(f"  (not available - no probe_summary.{tier} data in sidecars;")
        lines.append("   either probes file absent or inline eval disabled)")
        lines.append("")
        return lines

    n = (latest or {}).get("n")
    sc = (latest or {}).get("sign_correct")
    sc_pct = (latest or {}).get("sign_correct_pct")
    mv = (latest or {}).get("median_abs_v")
    r5_pct = (latest or {}).get("rolling5_sign_correct_pct")
    r5_mv = (latest or {}).get("rolling5_median_abs_v")
    lines.append("  Latest iter:")
    if n and n > 0:
        sc_pct_s = f"{sc_pct:.1%}" if sc_pct is not None else "n/a"
        mv_s = f"{mv:.3f}" if mv is not None else "n/a"
        lines.append(f"    n={n}, sign_correct={sc}/{n} ({sc_pct_s}), median |v|={mv_s}")
        if r5_pct is not None:
            r5_mv_s = f"{r5_mv:.3f}" if r5_mv is not None else "n/a"
            lines.append(f"    rolling(5 prior): sign={r5_pct:.1%}, median |v|={r5_mv_s}")
    else:
        lines.append(f"    n=0 (no probes matched active_size at this iter)")

    if by_iter:
        trend = by_iter[-10:]
        lines.append("")
        lines.append(f"  Trend (last {len(trend)} iters):")
        lines.append(f"    {'iter':>5} {'n':>4} {'sc':>4} {'sc%':>8} {'|v|':>8} {'delta_sc%':>10} {'rolling5_sc%':>13}")
        for row in trend:
            def _p(x, pct=False):
                if x is None: return "n/a"
                return f"{x:.1%}" if pct else f"{x:.3f}"
            def _d(x):
                if x is None: return "n/a"
                return f"{x*100:+.1f}pp"
            lines.append(
                f"    {row['iteration']:>5} "
                f"{row.get('n') or 0:>4} "
                f"{row.get('sign_correct') or 0:>4} "
                f"{_p(row.get('sign_correct_pct'), pct=True):>8} "
                f"{_p(row.get('median_abs_v')):>8} "
                f"{_d(row.get('delta_sign_correct_pct')):>10} "
                f"{_p(row.get('rolling5_sign_correct_pct'), pct=True):>13}"
            )
        lines.append(f"  ... full per-iter table: {tier}_probe_by_iter.csv")
    lines.append("")
    return lines


def format_forced_probe_report(by_iter: list, latest: dict) -> List[str]:
    """Backward-compat shim. Use format_tier_probe_report('forced', ...) instead."""
    return format_tier_probe_report("forced", by_iter, latest)


def format_value_calibration_report(summary: dict) -> List[str]:
    """Render the value-calibration section for the text report."""
    lines = []
    lines.append("Value Head Calibration by Position Type (Phase 1)")
    lines.append("=" * 50)
    if not summary:
        lines.append("  (not available — pass --weights <path> or place checkpoint")
        lines.append("   under checkpoints/<subdir>/ matching max(meta.iteration)+1)")
        lines.append("")
        return lines
    lines.append(f"  Weights: {summary.get('weights', '?')}")
    lines.append(f"  Stratified: True (per-bucket target N={summary.get('samples_per_bucket_target', '?')})")
    lines.append("  NOTE: per-bucket calibration is phase-stratified; 'overall' row is a")
    lines.append("        stratified aggregate, NOT population-weighted.")
    lines.append("")
    lines.append("  Natural vs. sampled distribution:")
    natural = summary.get("natural_distribution") or {}
    sampled = summary.get("sampled_distribution") or {}
    lines.append(f"    {'bucket':<34}{'natural':>10}{'sampled':>10}")
    for bucket in sorted(natural.keys()):
        lines.append(f"    {bucket:<34}{natural[bucket]:>10}{sampled.get(bucket, 0):>10}")
    lines.append("")
    lines.append("  Per-bucket stats:")
    buckets_stats = (summary.get("aggregate") or {}).get("buckets") or {}
    lines.append(f"    {'bucket':<34}{'n':>6}{'sign_agree':>12}{'mse':>10}")
    for bucket in sorted(buckets_stats.keys()):
        s = buckets_stats[bucket]
        n = s.get("n", 0)
        sa = s.get("sign_agree", "")
        mse = s.get("mse", "")
        lines.append(f"    {bucket:<34}{n:>6}{sa!s:>12}{mse!s:>10}")
    lines.append("")
    return lines


def format_replay_probe_scoring_report(summary: dict) -> List[str]:
    """Render the replay_probe_scoring section for the text report."""
    lines = []
    lines.append("Replay-Derived Probe Scoring (end-of-chunk snapshot)")
    lines.append("=" * 50)
    if not summary:
        lines.append("  (not available — pass --weights <path> or place checkpoint")
        lines.append("   under checkpoints/<subdir>/ matching max(meta.iteration)+1)")
        lines.append("")
        return lines
    if summary.get("probe_count", 0) == 0:
        lines.append(f"  (no probes extracted — {summary.get('skipped_reason', 'unknown')})")
        lines.append("")
        return lines
    lines.append(f"  Source: {summary.get('source', '?')} (NOT spec §7 curated gate suite)")
    lines.append(f"  Weights: {summary.get('weights', '?')}")
    lines.append(f"  Checkpoint in_channels: {summary.get('checkpoint_in_channels', '?')}")
    lines.append(f"  Probe count: {summary.get('probe_count', 0)}")
    n = summary.get("n", 0)
    sc = summary.get("sign_correct", 0)
    sc_pct = summary.get("sign_correct_pct", 0.0)
    mv = summary.get("median_abs_v", None)
    mv_s = f"{mv:.3f}" if mv is not None else "n/a"
    lines.append(f"  Overall: sign_correct={sc}/{n} ({sc_pct:.1%}), median |v|={mv_s}")
    lines.append("")
    lines.append("  By category:")
    for cat in sorted((summary.get("by_category") or {}).keys()):
        c = summary["by_category"][cat]
        pct = c.get("sign_correct_pct") or 0.0
        lines.append(f"    {cat:<20} n={c.get('n',0):>5}  sign_correct={pct:.1%}  median |v|={c.get('median_abs_v','?')}")
    lines.append("")
    return lines


# -----------------------------
# Main analysis
# -----------------------------

@dataclass
class ReplayRow:
    source: str
    iteration: int
    winner: str
    reason: str
    starting: str
    n_moves: int
    red_first: str
    black_first: str

    # Exact-corner only (true corners)
    red_first_is_corner: int
    black_first_is_corner: int

    # Opening-geometry additions (kept separate from --edge-pad logic)
    red_first_is_near_corner_r2: int
    black_first_is_near_corner_r2: int
    red_first_is_edge_band_b1: int
    black_first_is_edge_band_b1: int
    red_first_is_edge_band_b2: int
    black_first_is_edge_band_b2: int
    red_first_is_exact_edge: int
    black_first_is_exact_edge: int


def analyze(replays: List[dict],
            out_dir: str,
            board_size_override: Optional[int],
            edge_pad: int,
            opening_k: int,
            opening_geom_kmax: int,
            near_corner_radius: int,
            edge_band_width: int,
            buckets_spec: str,
            window: int,
            run_config: Optional[dict] = None,
            meta: Optional[dict] = None,
            sidecars: Optional[Dict[int, dict]] = None,
            no_plots: bool = False,
            dump_root_child_per_game: bool = False,
            out_suffix: Optional[str] = None,
            calibrate: bool = False,
            calibrate_weights: Optional[str] = None,
            no_connectivity: bool = False,
            args: Optional[argparse.Namespace] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Compute once — every output artifact shares this suffix.
    suffix = _derive_out_suffix(out_dir, override=out_suffix)
    buckets = _ply_buckets(buckets_spec)

    rows: List[ReplayRow] = []
    win_counts = Counter()
    start_counts = Counter()
    reason_counts = Counter()  # meta.reason: "resign", "win", "timeout_selfplay", etc.
    reason_by_winner = defaultdict(Counter)  # winner -> Counter[reason]

    # Per-ply placement counts (overall and per player)
    # heat[player][bucket] = NxN int
    heat = defaultdict(lambda: defaultdict(lambda: None))
    # Also track "opening move" distributions
    opening_pos = defaultdict(Counter)   # player -> Counter[(r,c)]
    opening_seq = Counter()             # key tuple -> count

    # Corner/edge tendency in early game
    early_edge = defaultdict(int)   # player -> count of early moves on edge
    early_corner = defaultdict(int) # player -> count of early moves in corner pad
    early_total = defaultdict(int)  # player -> total early moves considered

    # Opening-geometry additions (fixed definitions independent of --edge-pad)
    # We keep these separate so old metrics remain comparable.
    geom_kmax = int(opening_geom_kmax)
    geom_kmax = max(1, geom_kmax)
    R = int(near_corner_radius)
    B = int(edge_band_width)

    # counts[player][k] where k is 1..geom_kmax (ply cutoff)
    geom_total = defaultdict(lambda: defaultdict(int))
    geom_near_corner = defaultdict(lambda: defaultdict(int))
    geom_edge_band = defaultdict(lambda: defaultdict(int))
    geom_edge_band_b1 = defaultdict(lambda: defaultdict(int))
    geom_edge_band_b2 = defaultdict(lambda: defaultdict(int))

    # For drift / diversity
    per_game_opening = []  # list of ((player,r,c)...) k moves

    for rp in replays:
        bs = _board_size(rp, board_size_override)
        ms = _moves(rp)
        w = _winner(rp)
        s = _starting_player(rp)

        it = (rp.get("meta") or {}).get("iteration")
        it = int(it) if isinstance(it, int) else -1
        reason = ((rp.get("meta") or {}).get("reason") or "unknown").lower().strip()

                # First move per player (for opening-geometry tracking)
        red_first = ""
        black_first = ""
        for m0 in ms:
            pl0 = (m0.get("player") or "").lower()
            try:
                rr0 = int(m0.get("row"))
                cc0 = int(m0.get("col"))
            except Exception:
                continue
            if pl0 == "red" and not red_first:
                red_first = f"{rr0},{cc0}"
            elif pl0 == "black" and not black_first:
                black_first = f"{rr0},{cc0}"
            if red_first and black_first:
                break

        # Corner coordinates for this board size (exact corners only)
        corners = {(0, 0), (0, bs - 1), (bs - 1, 0), (bs - 1, bs - 1)}

        def _parse_rc(rc: str) -> Optional[Tuple[int, int]]:
            if not rc:
                return None
            try:
                rr_s, cc_s = rc.split(",", 1)
                return int(rr_s), int(cc_s)
            except Exception:
                return None

        def _is_corner_exact(rc: str) -> int:
            pt = _parse_rc(rc)
            return 1 if (pt is not None and pt in corners) else 0

        def _is_near_corner_r2(rc: str) -> int:
            pt = _parse_rc(rc)
            return 1 if (pt is not None and _is_near_corner(pt[0], pt[1], bs, 2)) else 0

        def _is_edge_band_b1(rc: str) -> int:
            pt = _parse_rc(rc)
            return 1 if (pt is not None and _is_edge_band(pt[0], pt[1], bs, 1)) else 0

        def _is_edge_band_b2(rc: str) -> int:
            pt = _parse_rc(rc)
            return 1 if (pt is not None and _is_edge_band(pt[0], pt[1], bs, 2)) else 0

        # NOTE: do not shadow the module-level `_is_exact_edge(row, col, n)`.
        def _rc_is_exact_edge(rc: str) -> int:
            pt = _parse_rc(rc)
            return 1 if (pt is not None and _is_exact_edge(pt[0], pt[1], bs)) else 0

        red_first_is_corner = _is_corner_exact(red_first)
        black_first_is_corner = _is_corner_exact(black_first)

        rows.append(ReplayRow(
            source=rp.get("_source_name", ""),
            iteration=it,
            winner=w,
            reason=reason,
            starting=s,
            n_moves=len(ms),
            red_first=red_first,
            black_first=black_first,
            red_first_is_corner=red_first_is_corner,
            black_first_is_corner=black_first_is_corner,
            red_first_is_near_corner_r2=_is_near_corner_r2(red_first),
            black_first_is_near_corner_r2=_is_near_corner_r2(black_first),
            red_first_is_edge_band_b1=_is_edge_band_b1(red_first),
            black_first_is_edge_band_b1=_is_edge_band_b1(black_first),
            red_first_is_edge_band_b2=_is_edge_band_b2(red_first),
            black_first_is_edge_band_b2=_is_edge_band_b2(black_first),
            red_first_is_exact_edge=_rc_is_exact_edge(red_first),
            black_first_is_exact_edge=_rc_is_exact_edge(black_first),
        ))
        win_counts[w] += 1
        start_counts[s] += 1
        reason_counts[reason] += 1
        reason_by_winner[w][reason] += 1

        # init heat arrays lazily
        for pl in ("red","black"):
            for _,_,lab in buckets:
                if heat[pl][lab] is None:
                    heat[pl][lab] = np.zeros((bs, bs), dtype=np.int32)
            if heat[pl]["all"] is None:
                heat[pl]["all"] = np.zeros((bs, bs), dtype=np.int32)

        # opening stats
        if ms:
            opening_pos[ms[0]["player"]][(int(ms[0]["row"]), int(ms[0]["col"]))] += 1
        opening_seq[_opening_sequence_key(ms, opening_k)] += 1
        per_game_opening.append(_opening_sequence_key(ms, opening_k))

        # placements by bucket / all
        for m in ms:
            ply = int(m.get("turn", 0))
            pl = (m.get("player") or "").lower()
            if pl not in ("red","black"):
                continue
            r = int(m["row"]); c = int(m["col"])
            if not (0 <= r < bs and 0 <= c < bs):
                continue
            lab = _bucket_for_ply(ply, buckets)
            heat[pl][lab][r, c] += 1
            heat[pl]["all"][r, c] += 1

            # edge/corner bias in early plies (use first bucket upper bound as "early")
            early_max = buckets[0][1]
            if ply <= early_max:
                edge, corner = _edge_or_corner(r, c, bs, edge_pad)
                early_total[pl] += 1
                if edge:
                    early_edge[pl] += 1
                if corner:
                    early_corner[pl] += 1

                # Opening-geometry additions for k=1..geom_kmax (plies counted from 1)
                if ply <= geom_kmax:
                    is_nc = _is_near_corner(r, c, bs, R)
                    is_eb = _is_edge_band(r, c, bs, B)
                    is_eb1 = _is_edge_band(r, c, bs, 1)
                    is_eb2 = _is_edge_band(r, c, bs, 2)
                    for kk in range(ply, geom_kmax + 1):
                        geom_total[pl][kk] += 1
                        if is_nc:
                            geom_near_corner[pl][kk] += 1
                        if is_eb:
                            geom_edge_band[pl][kk] += 1
                        if is_eb1:
                            geom_edge_band_b1[pl][kk] += 1
                        if is_eb2:
                            geom_edge_band_b2[pl][kk] += 1

    # -----------------------------
    # Write summary CSV
    # -----------------------------
    summary_csv = os.path.join(out_dir, _suffixed("replay_summary", "csv", suffix))
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "iteration", "winner", "reason", "starting_player", "n_moves",
            "red_first", "black_first",
            "red_first_is_corner", "black_first_is_corner",
            "red_first_is_exact_edge", "black_first_is_exact_edge",
            "red_first_is_near_corner_r2", "black_first_is_near_corner_r2",
            "red_first_is_edge_band_b1", "black_first_is_edge_band_b1",
            "red_first_is_edge_band_b2", "black_first_is_edge_band_b2"])
        for r in rows:
            # Pre-existing bug: the row-write previously omitted the `_b1`
            # pair, so the CSV misaligned by two columns — the `_b1` header
            # carried `_b2` values and the `_b2` columns came out blank.
            # Row order MUST exactly match the header (18 fields).
            w.writerow([r.source, r.iteration, r.winner, r.reason, r.starting, r.n_moves,
            r.red_first, r.black_first,
            r.red_first_is_corner, r.black_first_is_corner,
            r.red_first_is_exact_edge, r.black_first_is_exact_edge,
            r.red_first_is_near_corner_r2, r.black_first_is_near_corner_r2,
            r.red_first_is_edge_band_b1, r.black_first_is_edge_band_b1,
            r.red_first_is_edge_band_b2, r.black_first_is_edge_band_b2])

    # -----------------------------
    # Aggregate metrics
    # -----------------------------
    n = len(rows)
    avg_len = sum(r.n_moves for r in rows) / max(1, n)
    it_vals = [r.iteration for r in rows if r.iteration >= 0]
    it_min = min(it_vals) if it_vals else None
    it_max = max(it_vals) if it_vals else None

    # Opening diversity
    open_ent = {pl: _pos_entropy(opening_pos[pl]) for pl in opening_pos}
    open_top = {}
    for pl, cnt in opening_pos.items():
        if cnt:
            (pos, c) = cnt.most_common(1)[0]
            open_top[pl] = {"pos": pos, "share": c / sum(cnt.values())}
        else:
            open_top[pl] = {"pos": None, "share": 0.0}

    # Drift: KL between consecutive windows of opening-1 distribution (both players combined)
    # (Helpful to see if you're "stuck": KL stays ~0 AND entropy is low.)
    drift = []
    if n >= 2 * window:
        # Build combined first-move histograms per game order
        combined = []
        for rp in replays:
            ms = _moves(rp)
            if not ms:
                combined.append(None)
                continue
            combined.append((ms[0]["player"], int(ms[0]["row"]), int(ms[0]["col"])))
        # Map positions into index space per player separately then concatenate
        # Simpler: represent as string key
        keys = []
        for x in combined:
            if x is None:
                continue
            keys.append(f"{x[0]}@{x[1]},{x[2]}")
        vocab = {k:i for i,k in enumerate(sorted(set(keys)))}
        if vocab:
            arr = np.zeros((len(combined), len(vocab)), dtype=np.float32)
            for i,x in enumerate(combined):
                if x is None:
                    continue
                k = f"{x[0]}@{x[1]},{x[2]}"
                arr[i, vocab[k]] = 1.0
            # sliding windows
            for start in range(0, len(combined) - 2*window + 1, window):
                p = arr[start:start+window].sum(axis=0)
                q = arr[start+window:start+2*window].sum(axis=0)
                drift.append({
                    "window_start": start,
                    "kl_opening1": _kl(p, q),
                    "support": int((p>0).sum()),
                })

    # Edge / corner rates
    edge_rates = {}
    for pl in ("red","black"):
        tot = early_total.get(pl, 0)
        edge_rates[pl] = {
            "early_moves_considered": tot,
            "edge_rate": (early_edge.get(pl, 0) / tot) if tot else 0.0,
            "corner_rate": (early_corner.get(pl, 0) / tot) if tot else 0.0,
        }

    
    # Opening-geometry additions (k=1..geom_kmax)
    opening_geometry = {
        "board_size": board_size_override or 24,
        "k_max": geom_kmax,
        "near_corner_radius": R,
        "edge_band_width": B,
        "edge_band_width_b1": 1,
        "edge_band_width_b2": 2,
        "per_player": {},
    }
    for pl in ("red", "black"):
        ks = {}
        for k in range(1, geom_kmax + 1):
            tot = geom_total[pl].get(k, 0)
            ks[str(k)] = {
                "moves_considered": tot,
                "near_corner_rate": (geom_near_corner[pl].get(k, 0) / tot) if tot else 0.0,
                "edge_band_rate": (geom_edge_band[pl].get(k, 0) / tot) if tot else 0.0,
                "edge_band_rate_b1": (geom_edge_band_b1[pl].get(k, 0) / tot) if tot else 0.0,
                "edge_band_rate_b2": (geom_edge_band_b2[pl].get(k, 0) / tot) if tot else 0.0,
            }
        opening_geometry["per_player"][pl] = {
            "k": ks,
            "near_corner_rate_first_k": ks[str(geom_kmax)]["near_corner_rate"],
            "edge_band_rate_first_k": ks[str(geom_kmax)]["edge_band_rate"],
            "edge_band_rate_first_k_b1": ks[str(geom_kmax)]["edge_band_rate_b1"],
            "edge_band_rate_first_k_b2": ks[str(geom_kmax)]["edge_band_rate_b2"],
        }

# Aggregate first-move corner bias
    n_rows = len(rows)
    red_first_corner_rate = (sum(r.red_first_is_corner for r in rows) / n_rows) if n_rows else 0.0
    black_first_corner_rate = (sum(r.black_first_is_corner for r in rows) / n_rows) if n_rows else 0.0
    either_first_corner_rate = (sum((1 if (r.red_first_is_corner or r.black_first_is_corner) else 0) for r in rows) / n_rows) if n_rows else 0.0

    # --- Sidecar coverage validation ---
    replay_iters = set(r.iteration for r in rows if r.iteration >= 0)
    use_sidecar = False
    relevant_sidecars: Dict[int, dict] = {}
    partial_coverage = False

    if sidecars and replay_iters:
        # Only use sidecars matching replay iterations
        relevant_sidecars = {it: sidecars[it] for it in replay_iters if it in sidecars}
        missing = replay_iters - set(relevant_sidecars.keys())
        if missing:
            partial_coverage = True
            print(f"[WARN] Partial sidecar coverage: missing iterations {sorted(missing)}. "
                  f"Falling back to replay-derived stats.")
        elif relevant_sidecars:
            use_sidecar = True

    sc_agg = aggregate_sidecars(relevant_sidecars) if use_sidecar else {}

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

    # --- Phase 1: root-child diagnostics --------------------------------------
    # Prefer per-game records when present — they're the authoritative source
    # and side-step two failure modes of the sidecar path:
    #   (a) pre-IPC-fix runs emit per-game records but no sidecar block
    #   (b) partial / stale sidecars (e.g. a small test block) would otherwise
    #       shadow a large pool of per-game data
    # Only fall back to the sidecar aggregate when per-game records are absent
    # (e.g. analyzer fed only a zipped summary).
    rcd_aggregate: dict = {}
    rcd_summary_dict: dict = {}
    rcd_source: str = "none"
    if _HAS_OD_ANALYZER:
        if od_all_diag_lists:
            rcd_aggregate = aggregate_replay_root_child_diagnostics(
                od_all_diag_lists, child_detail_max_ply=2,
            )
            if rcd_aggregate:
                rcd_source = "replay"
        if not rcd_aggregate and relevant_sidecars:
            rcd_by_iter = extract_sidecar_root_child_diagnostics(relevant_sidecars)
            if rcd_by_iter:
                rcd_aggregate = aggregate_sidecar_root_child_diagnostics(rcd_by_iter)
                rcd_source = "sidecar"
        if rcd_aggregate:
            rcd_summary_dict = build_root_child_summary(rcd_aggregate)
            rcd_summary_dict["source"] = rcd_source

    # --- Phase 4: replay-cap engagement (sidecar-only, optional) ------------
    # NOT gated on use_sidecar: see the comment above. Per-iteration replay_cap
    # blocks are self-describing and useful even under partial coverage.
    rcap_by_iter: Dict[int, dict] = {}
    rcap_summary_dict: dict = {}
    if relevant_sidecars:
        rcap_by_iter = extract_sidecar_replay_cap(relevant_sidecars)
        if rcap_by_iter:
            rcap_summary_dict = aggregate_replay_cap(rcap_by_iter)

    # --- Phase 2: early-override summary (combines mass + best-by-* signals)
    # Built only when we have at least the opening-diagnostics aggregate —
    # the best-by-* columns fill in only if the root-child aggregate is also
    # present. Pre-Phase-2 runs get an empty dict, which the formatter handles.
    early_override_summary_dict: dict = {}
    if _HAS_OD_ANALYZER and od_aggregate:
        early_override_summary_dict = build_early_override_summary(
            opd_aggregate=od_aggregate,
            rcd_aggregate=rcd_aggregate if rcd_aggregate else None,
            early_plies=2,
        )

    # --- Phase 1 (connectivity-retrain): connectivity diagnostics ----------
    # Replays the move history of every game, computes per-position
    # connectivity stats (goal-touching components, largest component size),
    # and aggregates by (ply bucket, outcome). Runs unconditionally when the
    # module is importable; gated by --no-connectivity for cost-sensitive runs.
    connectivity_rows: List[dict] = []
    connectivity_summary: dict = {}
    value_calibration_summary: dict = {}

    if _HAS_PHASE1_DIAG and not no_connectivity:
        # Ply buckets are intentionally distinct from --ply-buckets (heatmaps):
        # wider buckets yield tighter per-bucket confidence at the cost of
        # finer phase resolution. Matches the cadence in the design spec.
        phase1_ply_buckets = [
            (1, 40, "ply_1_40"),
            (41, 80, "ply_41_80"),
            (81, 120, "ply_81_120"),
            (121, 10_000, "ply_121+"),
        ]
        try:
            connectivity_rows = aggregate_connectivity_by_ply(
                replays, phase1_ply_buckets,
            )
            connectivity_summary = {"n_rows": len(connectivity_rows)}
        except Exception as _e:
            # Defensive: one malformed replay should not take down the whole
            # analyzer. Report an empty summary so downstream writers skip.
            connectivity_rows = []
            connectivity_summary = {"n_rows": 0, "error": str(_e)}

    # --- Phase 1 (connectivity-retrain): replay-derived probe scoring ------
    # Spec §6.2: resolve checkpoint once, shared across probe scoring + calibration.
    resolved_weights = _resolve_checkpoint_path(args, replays) if args is not None else None
    shared_network = None
    _in_ch = None
    if resolved_weights is not None:
        try:
            from scripts.GPU.alphazero.probe_eval import load_network_for_scoring
            shared_network, _in_ch, _h, _nb = load_network_for_scoring(
                resolved_weights, verbose=False
            )
        except Exception as _e:
            print(f"[analyzer] WARNING: failed to load checkpoint {resolved_weights}: {_e}",
                  file=sys.stderr)
            resolved_weights = None
            shared_network = None

    # Spec §6.3: replay-derived probe scoring.
    replay_probe_scoring: dict = {}
    probes_for_scoring: list = []
    scoring_result: dict = {}
    if (resolved_weights is not None
            and shared_network is not None
            and args is not None
            and not args.probe_scoring_disable):
        from scripts.GPU.alphazero.probe_eval import (
            extract_forced_probes_from_games, run_forced_probes_inline,
        )
        probes_for_scoring = extract_forced_probes_from_games(
            replays,
            active_size=24,
            k_plies=2,
            winner_reasons=frozenset({"win"}),
            dedupe_exact=True,
            dedupe_mirror=True,
            max_probes=None,
        )
        if not probes_for_scoring:
            replay_probe_scoring = {
                "source": "replay_derived",
                "weights": os.path.abspath(resolved_weights),
                "probe_count": 0,
                "skipped_reason": "no_natural_wins",
            }
        else:
            scoring_result = run_forced_probes_inline(
                shared_network, probes_for_scoring, active_size=24
            )
            n = scoring_result["n"]
            sign_correct = scoring_result["sign_correct"]
            # Category breakdown.
            by_cat: dict = {}
            nn_values_iter = iter(scoring_result["nn_values"])
            exp_signs_iter = iter(scoring_result["expected_signs"])
            for p in probes_for_scoring:
                v = next(nn_values_iter)
                s = next(exp_signs_iter)
                correct = int((s > 0 and v > 0) or (s < 0 and v < 0)
                              or (s == 0 and abs(v) < 0.1))
                cat = p["category"]
                cat_bucket = by_cat.setdefault(cat, {
                    "n": 0, "sign_correct": 0, "abs_v_sum": 0.0,
                })
                cat_bucket["n"] += 1
                cat_bucket["sign_correct"] += correct
                cat_bucket["abs_v_sum"] += abs(v)
            by_category = {
                cat: {
                    "n": b["n"],
                    "sign_correct_pct": round(b["sign_correct"] / b["n"], 4)
                                       if b["n"] else None,
                    "median_abs_v": round(b["abs_v_sum"] / b["n"], 4)
                                   if b["n"] else None,
                }
                for cat, b in by_cat.items()
            }
            replay_probe_scoring = {
                "source": "replay_derived",
                "weights": os.path.abspath(resolved_weights),
                "checkpoint_in_channels": _in_ch,
                "selection_rules": {
                    "k_plies": 2,
                    "winner_reasons": ["win"],
                    "dedup": "exact + 4-form-mirror-canonical",
                },
                "probe_count": len(probes_for_scoring),
                "n": n,
                "sign_correct": sign_correct,
                "sign_correct_pct": scoring_result["sign_correct_pct"],
                "median_abs_v": scoring_result["median_abs_v"],
                "by_category": by_category,
            }

    # Spec §6.4: real calibration scoring (replaces former stub).
    if (_HAS_PHASE1_DIAG
            and resolved_weights is not None
            and shared_network is not None
            and not args.calibration_disable):
        from scripts.GPU.alphazero.value_calibration import (
            score_samples_against_checkpoint,
        )
        try:
            cal = score_samples_against_checkpoint(
                replays,
                network=shared_network,
                samples_per_bucket=args.calibration_samples_per_bucket,
                max_total=args.calibration_max_total,
                min_size=args.winning_structure_min_size,
            )
            cal["weights"] = os.path.abspath(resolved_weights)
            value_calibration_summary = cal
        except Exception as _e:
            print(f"[analyzer] WARNING: calibration scoring failed: {_e}",
                  file=sys.stderr)
            value_calibration_summary = {}

    # --- Build summary from sidecar or fallback ---
    if use_sidecar:
        results_val = sc_agg["results"]
        draw_breakdown_val = sc_agg["draw_breakdown"]
        termination_val = sc_agg["termination"]
        termination_by_winner_val = sc_agg["termination_by_winner"]
        avg_plies_val = sc_agg["avg_plies"]
        balance_val = sc_agg["balance"]
        resign_val = sc_agg["resign"]
        targets_val = sc_agg.get("targets", {"z_pos": 0, "z_zero": 0, "z_neg": 0})
        adjudication_val = sc_agg.get("adjudication", {})
        resign_gate_val = sc_agg.get("resign_gate", {})
        compute_val = sc_agg.get("compute", {})
        stats_source = {
            "mode": "sidecar",
            "sidecar_iterations": sorted(relevant_sidecars.keys()),
            "replay_iterations": sorted(replay_iters),
            "partial_sidecar_coverage": False,
        }
    else:
        # Backward compat: derive from game files, normalized to target schema
        results_val = {
            "red_wins": win_counts.get("red", 0),
            "black_wins": win_counts.get("black", 0),
            "draws": win_counts.get("draw", 0),
        }
        _rw = results_val["red_wins"]
        _bw = results_val["black_wins"]
        _dw = results_val["draws"]
        _decisive = _rw + _bw
        _total = _rw + _bw + _dw
        draw_breakdown_val = {
            "timeout": sum(v for k, v in reason_by_winner.get("draw", {}).items() if "timeout" in k),
            "board_full": reason_by_winner.get("draw", {}).get("board_full", 0),
            "state_cap": reason_by_winner.get("draw", {}).get("state_cap", 0),
            "unknown": sum(v for k, v in reason_by_winner.get("draw", {}).items() if k in ("unknown", "draw")),
        }
        termination_val = {
            "win": reason_counts.get("win", 0),
            "resign": reason_counts.get("resign", 0),
            "adjudicated": reason_counts.get("adjudicated", 0),
            "timeout": draw_breakdown_val["timeout"],
        }
        termination_by_winner_val = {
            "red": {
                "win": reason_by_winner.get("red", {}).get("win", 0),
                "resign": reason_by_winner.get("red", {}).get("resign", 0),
                "adjudicated": reason_by_winner.get("red", {}).get("adjudicated", 0),
            },
            "black": {
                "win": reason_by_winner.get("black", {}).get("win", 0),
                "resign": reason_by_winner.get("black", {}).get("resign", 0),
                "adjudicated": reason_by_winner.get("black", {}).get("adjudicated", 0),
            },
            "draw": {"timeout": draw_breakdown_val["timeout"]},
        }
        avg_plies_val = round(avg_len, 1)
        balance_val = {
            "red_pct": round(_rw / _decisive * 100, 1) if _decisive > 0 else 0.0,
            "black_pct": round(_bw / _decisive * 100, 1) if _decisive > 0 else 0.0,
            "draw_pct": round(_dw / _total * 100, 1) if _total > 0 else 0.0,
            "decisive_games": _decisive,
            "window": "n/a",
        }
        resign_val = {
            "total": reason_counts.get("resign", 0),
            "by_red": sum(1 for r in rows if r.reason == "resign" and r.winner == "black"),
            "by_black": sum(1 for r in rows if r.reason == "resign" and r.winner == "red"),
        }
        targets_val = {"z_pos": 0, "z_zero": 0, "z_neg": 0}
        adjudication_val = {
            "attempts": 0, "adjudicated": reason_counts.get("adjudicated", 0),
            "red_wins": reason_by_winner.get("red", {}).get("adjudicated", 0),
            "black_wins": reason_by_winner.get("black", {}).get("adjudicated", 0),
            "remaining_timeouts": draw_breakdown_val["timeout"],
            "blocks": {"ply": 0, "threshold": 0, "visits": 0, "top1": 0},
            "stats": {},
        }
        resign_gate_val = {
            "checks": 0, "red_checks": 0, "black_checks": 0,
            "value_hits": 0, "red_value_hits": 0, "black_value_hits": 0,
            "blocked_by_top1": 0, "red_blocked_by_top1": 0, "black_blocked_by_top1": 0,
            "eligible_hits": 0, "red_eligible_hits": 0, "black_eligible_hits": 0,
            "top1_share_on_value_hits": {}, "min_top1_share": 0.0,
        }
        compute_val = {"buffer_size": 0, "backups": 0, "leaf_evals": 0, "nn_batches": 0}
        stats_source = {
            "mode": "replay_fallback",
            "sidecar_iterations": sorted(relevant_sidecars.keys()) if relevant_sidecars else [],
            "replay_iterations": sorted(replay_iters),
            "partial_sidecar_coverage": partial_coverage,
        }

    # Per-game stats persistence surfacing (spec 2026-04-29).
    per_game_stats_val = aggregate_per_game_stats(replays)

    summary = {
        "iteration": sc_agg.get("iteration_max", it_max) if use_sidecar else it_max,
        "iteration_min": sc_agg.get("iteration_min", it_min) if use_sidecar else it_min,
        "iteration_max": sc_agg.get("iteration_max", it_max) if use_sidecar else it_max,
        "iterations_count": sc_agg.get("iterations_count", len(replay_iters)) if use_sidecar else len(replay_iters),
        "games_per_iter": sc_agg.get("games_per_iter", 0) if use_sidecar else 0,
        "games_total": sc_agg.get("games_total", n) if use_sidecar else n,
        "stats_source": stats_source,
        # --- New fields (additions) ---
        "results": results_val,
        "draw_breakdown": draw_breakdown_val,
        "termination": termination_val,
        "termination_by_winner": termination_by_winner_val,
        "avg_plies": avg_plies_val,
        "balance": balance_val,
        "targets": targets_val,
        "adjudication": adjudication_val,
        "resign": resign_val,
        "resign_gate": resign_gate_val,
        "compute": compute_val,
        # Per-game stats persistence surfacing (spec 2026-04-29).
        # Distributions complement the sidecar-derived `compute` totals above.
        "per_game_stats": per_game_stats_val,
        # --- Original fields (preserved) ---
        "analyzer": {"name": "twixt_replay_analyzer", "version": "0.4"},
        "run_config": (run_config or {}),
        "meta": (meta or {}),
        "first_move_corner_rate": {
            "red": red_first_corner_rate,
            "black": black_first_corner_rate,
            "either": either_first_corner_rate,
        },
        "n_games": n,
        "wins": dict(win_counts),
        "termination_reason": dict(reason_counts),
        "termination_reason_by_winner": {k: dict(v) for k, v in reason_by_winner.items()},
        "starting_player": dict(start_counts),
        "avg_game_length_moves": avg_len,
        "opening": {
            "k": opening_k,
            "first_move_entropy_nats": open_ent,
            "first_move_top": open_top,
            "sequence_top10": [
                {"count": c, "seq": list(seq)} for seq, c in opening_seq.most_common(10)
            ],
            # New additions inside opening
            "first_move_corner_rate": {
                "red": red_first_corner_rate,
                "black": black_first_corner_rate,
                "either": either_first_corner_rate,
            },
            "early_edge_corner": {
                "red": {"edge_rate": edge_rates.get("red", {}).get("edge_rate", 0.0), "corner_rate": edge_rates.get("red", {}).get("corner_rate", 0.0)},
                "black": {"edge_rate": edge_rates.get("black", {}).get("edge_rate", 0.0), "corner_rate": edge_rates.get("black", {}).get("corner_rate", 0.0)},
            },
        },
        "early_edge_corner": edge_rates,
        "opening_geometry": opening_geometry,
        "opening_drift": drift,
        "notes": [
            "Low entropy + high top-share + low KL drift => 'stuck opening' risk.",
            "If corner_rate is high early, you're likely still corner/edge-biased.",
        ],
        "opening_diagnostics_summary": od_summary_dict,
        "opening_diagnostics_by_ply": od_by_ply_dict,
        # Phase 1: root-child diagnostics (ply 0–1). Empty when sidecars predate
        # the feature or no record carries `root_summary`.
        "root_child_diagnostics_summary": rcd_summary_dict,
        # Phase 4: replay-cap engagement. Empty when sidecars predate the feature.
        "replay_cap_summary": rcap_summary_dict,
        # Phase 2: compact ply 0-1 view combining mass and best-by-* signals
        # with the run's near-corner config echoed inline.
        "early_override_summary": early_override_summary_dict,
        # Phase 1 (connectivity-retrain): connectivity diagnostics summary.
        # Empty dict when the module is unavailable or --no-connectivity is set.
        "connectivity_diagnostics": connectivity_summary,
        # Phase 1 (connectivity-retrain): replay-derived probe scoring.
        # Empty dict when checkpoint unavailable, scoring disabled, or no
        # natural-win replays in the input.
        "replay_probe_scoring": replay_probe_scoring,
        # Phase 1 (connectivity-retrain): value-calibration summary.
        # Populated when a checkpoint resolves and --calibration-disable is
        # not set. Empty dict otherwise (including scoring-exception path).
        "value_calibration": value_calibration_summary,
        # Phase 2: per-iter sanity-by-connectivity trend + latest snapshot.
        # Empty if no sidecar carried sanity_by_connectivity.
        "sanity_by_connectivity": {
            "by_iter": sc_agg.get("sanity_by_connectivity_by_iter", []) if use_sidecar else [],
            "latest": sc_agg.get("sanity_by_connectivity_latest", {}) if use_sidecar else {},
        },
        # Phase 2: inline per-tier probe trend + latest snapshot.
        # Each tier's `by_iter` is empty if probes file was absent or eval
        # disabled in the run. Per-tier human-friendly keys preserved during
        # the legacy/forward dual-emit window.
        **{
            f"{tier}_probe": {
                "by_iter": sc_agg.get(f"{tier}_probe_by_iter", []) if use_sidecar else [],
                "latest": sc_agg.get(f"{tier}_probe_latest", {}) if use_sidecar else {},
            }
            for tier in TIER_NAMES
        },
    }

    summary_json = os.path.join(out_dir, _suffixed("summary", "json", suffix))
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    if _HAS_OD_ANALYZER and od_summary_dict:
        _od_iter_info = {
            "iteration": summary.get("iteration"),
            "iteration_min": summary.get("iteration_min"),
            "iteration_max": summary.get("iteration_max"),
        }
        _od_csv1 = write_opening_summary_csv(out_dir, od_summary_dict, _od_iter_info, suffix=suffix)
        _od_csv2 = write_opening_by_ply_csv(out_dir, od_by_ply_dict, suffix=suffix)
        print(f"[OK] wrote: {_od_csv1}")
        print(f"[OK] wrote: {_od_csv2}")
        if od_per_game_records:
            _od_csv3 = write_opening_per_game_csv(out_dir, od_per_game_records, suffix=suffix)
            print(f"[OK] wrote: {_od_csv3}")

    # --- Phase 1: root_child_by_ply.csv (+ optional per-game dump) ---
    if _HAS_OD_ANALYZER and rcd_aggregate:
        _rcd_csv = write_root_child_by_ply_csv(out_dir, rcd_aggregate, suffix=suffix)
        if _rcd_csv:
            print(f"[OK] wrote: {_rcd_csv}")
        if dump_root_child_per_game and od_per_game_records:
            _rcd_pg = write_root_child_per_game_csv(out_dir, od_per_game_records, suffix=suffix)
            if _rcd_pg:
                print(f"[OK] wrote: {_rcd_pg}")

    # --- Phase 1 (connectivity-retrain): connectivity_by_ply.csv ---
    if connectivity_rows:
        connectivity_csv_path = os.path.join(
            out_dir,
            _suffixed("connectivity_by_ply", "csv", suffix),
        )
        with open(connectivity_csv_path, "w", newline="") as f:
            keys = list(connectivity_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in connectivity_rows:
                w.writerow(row)
        print(f"[OK] wrote: {connectivity_csv_path}")

    # --- Phase 2: per-iter sidecar-sourced connectivity/probe trends ---
    if use_sidecar:
        sbc_rows = sc_agg.get("sanity_by_connectivity_by_iter", [])
        if sbc_rows:
            sbc_csv_path = os.path.join(
                out_dir,
                _suffixed("sanity_by_connectivity_by_iter", "csv", suffix),
            )
            with open(sbc_csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(sbc_rows[0].keys()))
                w.writeheader()
                for row in sbc_rows:
                    w.writerow(row)
            print(f"[OK] wrote: {sbc_csv_path}")

        for tier in TIER_NAMES:
            tier_rows = sc_agg.get(f"{tier}_probe_by_iter", [])
            if tier_rows:
                tier_csv_path = os.path.join(
                    out_dir,
                    _suffixed(f"{tier}_probe_by_iter", "csv", suffix),
                )
                with open(tier_csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(tier_rows[0].keys()))
                    w.writeheader()
                    for row in tier_rows:
                        w.writerow(row)
                print(f"[OK] wrote: {tier_csv_path}")

    # --- Phase 4: replay_cap_by_iter.csv ---
    if rcap_by_iter:
        _rc_csv = write_replay_cap_by_iter_csv(out_dir, rcap_by_iter, suffix=suffix)
        if _rc_csv:
            print(f"[OK] wrote: {_rc_csv}")

    # Spec §6.5: new CSVs for replay probe scoring + value calibration.
    if replay_probe_scoring and replay_probe_scoring.get("probe_count", 0) > 0:
        write_replay_probe_per_probe_csv(
            out_dir, suffix,
            probes=probes_for_scoring,
            scoring_result=scoring_result,
        )
    if value_calibration_summary:
        write_value_calibration_by_bucket_csv(
            out_dir, suffix, value_calibration_summary
        )

    # -----------------------------
    # Heatmap figures
    # -----------------------------
    def save_heatmap(arr: np.ndarray, title: str, path: str) -> None:
        if no_plots or (not _HAS_MPL):
            return
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(arr, origin="upper")
        ax.set_title(title)
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    for pl in ("red","black"):
        for lab, arr in heat[pl].items():
            if arr is None:
                continue
            fn = f"heat_{pl}_{lab}.png"
            save_heatmap(arr, f"{pl} placements (ply bucket: {lab})", os.path.join(out_dir, fn))

    # -----------------------------
    # Text report
    # -----------------------------
    report_path = os.path.join(out_dir, _suffixed("report", "txt", suffix))
    lines = []
    lines.append("Twixt Replay Analyzer Report")
    lines.append("="*30)
    lines.append(f"Games analyzed: {n}")
    if it_min is not None:
        lines.append(f"Iteration range: {it_min} .. {it_max}")
    lines.append(f"Avg game length (moves): {avg_len:.1f}")
    lines.append("")
    lines.append("Win counts:")
    for k,v in win_counts.items():
        lines.append(f"  {k}: {v} ({v/max(1,n)*100:.1f}%)")
    lines.append("")
    lines.append("Termination reason:")
    for k,v in reason_counts.most_common():
        lines.append(f"  {k}: {v} ({v/max(1,n)*100:.1f}%)")
    lines.append("")
    lines.append("Termination reason by winner:")
    for winner in ("red", "black", "draw"):
        if winner in reason_by_winner:
            parts = ", ".join(f"{r}={c}" for r, c in reason_by_winner[winner].most_common())
            lines.append(f"  {winner}: {parts}")
    lines.append("")
    lines.append("Starting player counts:")
    for k,v in start_counts.items():
        lines.append(f"  {k}: {v} ({v/max(1,n)*100:.1f}%)")
    lines.append("")
    lines.append("Opening first-move diversity:")
    for pl in sorted(open_ent.keys()):
        top = open_top.get(pl, {})
        lines.append(f"  {pl}: entropy={open_ent[pl]:.3f} nats, top={top.get('pos')} share={top.get('share',0.0):.2%}")
    lines.append("")
    lines.append("Early edge/corner rates (plies <= first bucket hi):")
    for pl, d in edge_rates.items():
        lines.append(f"  {pl}: edge={d['edge_rate']:.2%} corner={d['corner_rate']:.2%} (n={d['early_moves_considered']})")
    lines.append("")
    lines.append("Top opening sequences:")
    for item in summary["opening"]["sequence_top10"]:
        lines.append(f"  x{item['count']}: {item['seq']}")
    lines.append("")
    if drift:
        lines.append("Opening drift (KL between windows):")
        for d in drift:
            lines.append(f"  start={d['window_start']}: KL={d['kl_opening1']:.4f} (support={d['support']})")
    else:
        lines.append("Opening drift: not computed (need >= 2*window games).")
    lines.append("")

    # --- New summary sections ---
    lines.append(f"Stats source: {summary['stats_source']['mode']}")
    lines.append("")

    lines.append("Results:")
    r = summary["results"]
    lines.append(f"  Red={r['red_wins']}, Black={r['black_wins']}, Draws={r['draws']}")
    lines.append("")

    db = summary["draw_breakdown"]
    lines.append(f"Draw breakdown: timeout={db['timeout']}, board_full={db['board_full']}, state_cap={db['state_cap']}, unknown={db['unknown']}")
    lines.append("")

    rs = summary["resign"]
    lines.append(f"Resign: {rs['total']} (by_red={rs['by_red']}, by_black={rs['by_black']})")
    lines.append("")

    bal = summary["balance"]
    lines.append(f"Balance: red={bal['red_pct']}%, black={bal['black_pct']}%, draw={bal['draw_pct']}% "
                 f"(decisive={bal['decisive_games']}, window={bal['window']})")
    lines.append("")

    lines.append(f"Avg plies: {summary['avg_plies']}")
    lines.append("")

    if use_sidecar:
        adj = summary["adjudication"]
        lines.append(f"Adjudicated: {adj['adjudicated']} (red_wins={adj['red_wins']}, black_wins={adj['black_wins']}, remaining_timeouts={adj['remaining_timeouts']})")
        blk = adj["blocks"]
        lines.append(f"Adjudication blocks: ply={blk['ply']} thr={blk['threshold']} visits={blk['visits']} top1={blk['top1']} (attempts={adj['attempts']})")
        if adj.get("stats"):
            arv = adj["stats"].get("abs_root_value", {})
            t1s = adj["stats"].get("top1_share", {})
            lines.append(f"Adj stats: abs_rv p50={arv.get('p50', 'n/a')} p90={arv.get('p90', 'n/a')} top1 p50={t1s.get('p50', 'n/a')} p10={t1s.get('p10', 'n/a')}")
        lines.append("")

        rg = summary["resign_gate"]
        lines.append(f"Resign gate:")
        lines.append(f"  checks={rg['checks']} (red={rg['red_checks']}, black={rg['black_checks']})")
        lines.append(f"  value_hits={rg['value_hits']} (red={rg['red_value_hits']}, black={rg['black_value_hits']})")
        lines.append(f"  blocked_by_top1={rg['blocked_by_top1']} (red={rg['red_blocked_by_top1']}, black={rg['black_blocked_by_top1']})  [min_top1={rg['min_top1_share']}]")
        lines.append(f"  eligible_hits={rg['eligible_hits']} (red={rg['red_eligible_hits']}, black={rg['black_eligible_hits']})")
        tp = rg.get("top1_share_on_value_hits", {})
        if tp:
            lines.append(f"  top1_share_on_value_hits: p50={tp.get('p50', 'n/a')} p90={tp.get('p90', 'n/a')} p99={tp.get('p99', 'n/a')}")
        lines.append("")

        tgt = summary["targets"]
        lines.append(f"Targets: z_pos={tgt['z_pos']}, z_zero={tgt['z_zero']}, z_neg={tgt['z_neg']}")
        lines.append("")

        comp = summary["compute"]
        lines.append(f"Compute: buffer_size={comp['buffer_size']}, backups={comp['backups']}, leaf_evals={comp['leaf_evals']}, nn_batches={comp['nn_batches']}")
        lines.append("")

        # Per-game stats triage section (spec 2026-04-29).
        lines.extend(format_per_game_stats_report(summary["per_game_stats"]))

    if _HAS_OD_ANALYZER and od_summary_dict:
        lines.extend(format_opening_diagnostics_report(od_summary_dict, od_by_ply_dict, od_warnings))

    # Phase 1 (root-child at ply 0–1). Emits a graceful "not available" if
    # the sidecars in this range pre-date root_child_diagnostics.
    if _HAS_OD_ANALYZER:
        lines.extend(format_root_child_report(rcd_summary_dict))

    # Phase 2 (early-override summary at ply 0–1). The compact "is the early
    # override working?" view — mass + best-by-* disagreement deltas with the
    # run config echoed inline.
    if _HAS_OD_ANALYZER:
        lines.extend(format_early_override_report(early_override_summary_dict))

    # Phase 1 (connectivity-retrain): connectivity diagnostics.
    # Always on when the module is available; gated on --no-connectivity to
    # skip the compute path on cost-sensitive runs. Canonical report order:
    # root-child → early-override → connectivity → sanity-by-connectivity →
    # forced-probe → replay-cap.
    if _HAS_PHASE1_DIAG:
        lines.extend(format_connectivity_diagnostics_report(connectivity_summary, connectivity_rows))

    # Phase 2: per-iter value-head sign-agree by connectivity bucket + inline
    # forced-probe trends from sidecars. Rendered unconditionally (stubs
    # gracefully when sidecars lack the new blocks).
    if use_sidecar:
        lines.extend(format_sanity_by_connectivity_report(
            sc_agg.get("sanity_by_connectivity_by_iter", []),
            sc_agg.get("sanity_by_connectivity_latest", {}),
        ))
        for tier in TIER_NAMES:
            lines.extend(format_tier_probe_report(
                tier,
                sc_agg.get(f"{tier}_probe_by_iter", []),
                sc_agg.get(f"{tier}_probe_latest", {}),
            ))

    # Phase 1 (connectivity-retrain): value calibration + replay-derived
    # probe scoring. Both render unconditionally (formatter stubs gracefully
    # when the summary dict is empty, e.g. no checkpoint resolved).
    if _HAS_PHASE1_DIAG:
        lines.extend(format_value_calibration_report(value_calibration_summary))
        lines.extend(format_replay_probe_scoring_report(replay_probe_scoring))

    # Phase 4 (replay-cap engagement). Same backward-compat behavior.
    lines.extend(format_replay_cap_report(rcap_summary_dict))

    lines.append("Outputs:")
    lines.append(f"  - {os.path.abspath(summary_csv)}")
    lines.append(f"  - {os.path.abspath(summary_json)}")
    lines.append(f"  - heatmaps: heat_<player>_<bucket>.png")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[OK] wrote: {summary_csv}")
    print(f"[OK] wrote: {summary_json}")
    print(f"[OK] wrote: {report_path}")
    print(f"[OK] heatmaps saved in: {os.path.abspath(out_dir)}")


def main():
    ap = argparse.ArgumentParser(description="Analyze TwixT self-play replay JSONs and produce summary + plots.")
    ap.add_argument("--input", nargs="+", required=True,
                    help="Input path(s): .json files, directories, or .zip archives.")
    ap.add_argument("--out", required=True, help="Output directory for report artifacts.")
    ap.add_argument("--board-size", type=int, default=None, help="Fallback board size if meta.board_size missing.")
    ap.add_argument("--edge-pad", type=int, default=2,
                    help="How many squares from each edge counts as 'edge' (default 2).")
    ap.add_argument("--opening-k", type=int, default=6,
                    help="How many plies count as 'opening' for some stats (default 6).")

    # Opening-geometry additions (independent of --edge-pad)
    ap.add_argument("--opening-geom-kmax", type=int, default=4,
                    help="Compute additional opening-geometry rates for k=1..K plies (default 4).")
    ap.add_argument("--near-corner-radius", type=int, default=2,
                    help="Chebyshev radius for near-corner vicinity (default 2).")
    ap.add_argument("--edge-band-width", type=int, default=2,
                    help="Edge-band width in squares from edge (default 2).")

    ap.add_argument("--ply-buckets", type=str, default="1-4,5-10,11-20,21-999",
                    help="Comma-separated ply buckets for heatmaps, like '1-4,5-10,11-20,21-999'.")
    ap.add_argument("--window", type=int, default=50,
                    help="Window size (games) for KL drift calculation (default 50).")

    # Optional run metadata/config for reproducibility in summary.json
    ap.add_argument("--run-config", dest="run_config", default=None,
                    help="Path to JSON file containing run configuration. Merged into summary.json under run_config.")
    ap.add_argument("--meta", action="append", default=[],
                    help="Additional metadata key=value (repeatable). Merged into summary.json under meta.")
    ap.add_argument("--meta-json", dest="meta_json", default=None,
                    help="Additional metadata as JSON object string, or @path/to.json. Merged into summary.json under meta.")
    ap.add_argument("--no-plots", dest="no_plots", action="store_true",
                    help="Disable PNG plot generation (still writes replay_summary.csv and summary.json).")
    ap.add_argument("--dump-root-child-per-game", dest="dump_root_child_per_game",
                    action="store_true",
                    help="Additionally emit root_child_per_game.csv (large — one row "
                         "per (game, ply<2, child)). Useful for case-by-case inspection.")
    ap.add_argument("--out-suffix", dest="out_suffix", default=None,
                    help="Suffix appended to all output filenames (e.g. "
                         "`945-949` yields summary_945-949.json, report_945-949.txt, "
                         "replay_summary_945-949.csv, ...). Default: basename of "
                         "--out with a trailing `_Replay` stripped. Pass an empty "
                         "string to disable suffixing.")

    # Phase 1 (connectivity-retrain) CLI flags — spec Section 9.5.
    ap.add_argument("--probes", default=None,
                    help="Path to probe-eval sidecar data (reserved — not yet wired).")
    ap.add_argument("--calibrate", action="store_true", default=False,
                    help="Run value-calibration by position type "
                         "(requires --calibrate-weights). Currently a scaffold — "
                         "full scoring loop is a follow-up task.")
    ap.add_argument("--calibrate-weights", dest="calibrate_weights", default=None,
                    help="Explicit weights path for --calibrate. Superseded "
                         "by --weights + auto-discovery; retained for "
                         "backwards compatibility with existing scripts.")
    ap.add_argument("--calibration-sample", dest="calibration_sample", type=int, default=1000,
                    help="Number of positions to sample for calibration (default 1000).")
    ap.add_argument("--calibration-bins", dest="calibration_bins", type=int, default=5,
                    help="Reliability-diagram bin count (default 5).")
    ap.add_argument("--winning-structure-min-size", dest="winning_structure_min_size",
                    type=int, default=8,
                    help="Threshold for classify_position winning-structure bucket (default 8).")
    ap.add_argument("--no-connectivity", dest="no_connectivity", action="store_true", default=False,
                    help="Skip connectivity diagnostics (saves time on large runs).")

    # Probes / calibration — spec §6.1 new flags.
    ap.add_argument("--weights", default=None,
                    help="Explicit checkpoint path for probe scoring + "
                         "calibration. Skips auto-discovery. When omitted, "
                         "the analyzer auto-discovers model_iter_{max+1}"
                         ".safetensors in --checkpoint-dir or "
                         "checkpoints/<single-subdir>/.")
    ap.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None,
                    help="Directory to search for auto-discovered checkpoint. "
                         "When omitted, uses checkpoints/<single-subdir>/ if "
                         "exactly one subdirectory exists under checkpoints/.")
    ap.add_argument("--probe-scoring-disable", dest="probe_scoring_disable",
                    action="store_true", default=False,
                    help="Skip replay_probe_scoring entirely.")
    ap.add_argument("--calibration-disable", dest="calibration_disable",
                    action="store_true", default=False,
                    help="Skip value_calibration entirely.")
    ap.add_argument("--calibration-samples-per-bucket",
                    dest="calibration_samples_per_bucket", type=int, default=200,
                    help="Target samples per phase-stratified bucket (spec §4.3).")
    ap.add_argument("--calibration-max-total", dest="calibration_max_total",
                    type=int, default=2000,
                    help="Safety cap on total calibration forward passes.")

    args = ap.parse_args()

    # Enforce --calibrate requires --calibrate-weights (design spec §9.2).
    # Non-zero exit via ap.error — keeps error handling consistent with argparse.
    if args.calibrate and not args.calibrate_weights:
        ap.error("--calibrate requires --calibrate-weights <path> for formal runs")

    run_config = None
    if args.run_config:
        with open(args.run_config, "r", encoding="utf-8") as f:
            run_config = json.load(f)
        if not isinstance(run_config, dict):
            raise SystemExit("--run-config JSON must be an object")

    meta: dict = {}

    # key=value metadata
    for item in (args.meta or []):
        if "=" not in item:
            raise SystemExit(f"--meta must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        meta[k.strip()] = v.strip()

    # JSON metadata (string or @file)
    if args.meta_json:
        mj = args.meta_json
        if mj.startswith("@"):
            with open(mj[1:], "r", encoding="utf-8") as f:
                meta2 = json.load(f)
        else:
            meta2 = json.loads(mj)
        if not isinstance(meta2, dict):
            raise SystemExit("--meta-json must be a JSON object")
        meta.update(meta2)

    no_plots = bool(args.no_plots)


    replays = load_replays(args.input)
    if not replays:
        raise SystemExit("No replay game files found in input path(s). "
                         "The analyzer requires game files for opening/geometry analysis.")

    sidecars = load_sidecars(args.input)
    if sidecars:
        print(f"Found {len(sidecars)} stats sidecar(s) (iterations {min(sidecars)}..{max(sidecars)})")

    analyze(
        replays=replays,
        out_dir=args.out,
        board_size_override=args.board_size,
        edge_pad=args.edge_pad,
        opening_k=args.opening_k,
        opening_geom_kmax=args.opening_geom_kmax,
        near_corner_radius=args.near_corner_radius,
        edge_band_width=args.edge_band_width,
        buckets_spec=args.ply_buckets,
        window=args.window,
        run_config=run_config,
        meta=meta,
        sidecars=sidecars,
        no_plots=no_plots,
        dump_root_child_per_game=bool(args.dump_root_child_per_game),
        out_suffix=args.out_suffix,
        calibrate=bool(args.calibrate),
        calibrate_weights=args.calibrate_weights,
        no_connectivity=bool(args.no_connectivity),
        args=args,
    )


if __name__ == "__main__":
    main()