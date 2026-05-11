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
        compute_goal_completion_state,
        classify_selected_conversion_move,
    )
    from scripts.GPU.alphazero.game.twixt_state import TwixtState
    from scripts.GPU.alphazero.value_calibration import (
        aggregate_calibration,
        classify_position,
    )
    from scripts.GPU.alphazero.goal_completion_aggregator import (
        aggregate_goal_completion_records,
        _summarize_main_population as _aggregator_summarize_main_population,
        _summarize_capped_population as _aggregator_summarize_capped_population,
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
    `<tier>_probe_summary` field during the one-release dual-emit window
    (spec 2026-04-28 §6 + 2026-05-03 §9.2).
    """
    ps = sc.get("probe_summary") or {}
    if tier in ps and ps[tier] is not None:
        return ps[tier]
    # Legacy fallback: <tier>_probe_summary (e.g., forced_probe_summary,
    # strong_advantage_probe_summary).
    return sc.get(f"{tier}_probe_summary")


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
    outcomes = {"decisive": 0, "resign": 0, "adjudicated": 0,
                "timeout": 0, "state_cap": 0, "draw_other": 0}

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
        elif reason in ("state_cap", "terminal_state_cap"):
            outcomes["state_cap"] += 1
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


# ---------------------------------------------------------------------------
# Goal-completion / conversion diagnostics (spec 2026-05-03 §7)
# ---------------------------------------------------------------------------

# Outcome-class taxonomy (spec §7.1).
# Class 1 (decisive): winner-only scope, counted in main metrics.
# Class 2 (capped): both-sides scope, counted in bad_cases only.
# Class 3 (excluded): not counted.
_CLASS1_REASONS = frozenset({"win", "resign", "adjudicated"})
_CLASS2_REASONS = frozenset({"state_cap", "timeout", "timeout_selfplay", "board_full",
                             "terminal_state_cap"})


def _replay_int_field(replay: dict, field: str, default: int = 0) -> int:
    """Extract an int field from a replay dict.

    The saver writes some identity fields (iteration, game_idx) under
    `meta`, NOT at the top level. Look at top-level first, then meta.
    Returns `default` when the field is absent or None in both places.
    """
    val = replay.get(field)
    if val is None:
        meta = replay.get("meta") or {}
        val = meta.get(field)
    if val is None:
        return default
    return int(val)


def _surface_phase3_diagnostics(replays: list, n_decisive: int) -> dict:
    """Aggregate Phase 3 closeout-diagnostics records across replays.

    Reads per-game `goal_completion_diagnostics` (list of per-ply records)
    and `goal_completion_diagnostics_meta` (per-game meta) from each replay
    JSON. Pure dict iteration — no BFS, no replay walking.

    Returns:
        {"diagnostics_coverage": dict, "policy_mcts_summary": dict | None}
    """
    games_with_diag = 0
    total_records = 0
    total_error_count = 0
    total_resign_dropped = 0
    total_skipped_priors = 0
    total_records_dropped_cap = 0
    diagnostic_version = 1
    all_records: list = []

    for replay in replays:
        diag_array = replay.get("goal_completion_diagnostics")
        diag_meta = replay.get("goal_completion_diagnostics_meta")
        if diag_array:
            games_with_diag += 1
            total_records += len(diag_array)
            all_records.extend(diag_array)
        if diag_meta:
            total_error_count += diag_meta.get("error_count", 0) or 0
            total_resign_dropped += diag_meta.get("resign_dropped_partial_count", 0) or 0
            total_skipped_priors += diag_meta.get("skipped_missing_priors_count", 0) or 0
            total_records_dropped_cap += diag_meta.get("records_dropped_by_cap", 0) or 0
            v = diag_meta.get("diagnostic_version")
            if v is not None:
                diagnostic_version = v

    coverage_pct = (games_with_diag / n_decisive * 100.0) if n_decisive else 0.0
    diagnostics_coverage = {
        "games_with_diagnostics":            games_with_diag,
        "total_records":                     total_records,
        "coverage_pct_of_decisive_games":    coverage_pct,
        "error_count":                       total_error_count,
        "resign_dropped_partial_count":      total_resign_dropped,
        "skipped_missing_priors_count":      total_skipped_priors,
        "records_dropped_by_cap":            total_records_dropped_cap,
        "version":                           diagnostic_version,
    }

    policy_mcts_summary = _summarize_policy_mcts(all_records) if all_records else None

    return {
        "diagnostics_coverage": diagnostics_coverage,
        "policy_mcts_summary": policy_mcts_summary,
    }


def _aggregate_td_breakdown_multi_side(
    records: list,
    detected_sides: list,
    high_value_threshold: float = 0.95,
) -> dict:
    """Variant of aggregate_td_closeout_breakdown that takes per-record
    detected_player labels. Used when aggregating across games where
    detected_player differs game-to-game.

    Internally re-implements aggregation directly on raw counts to avoid
    losing precision when combining rates across games.
    """
    def _empty():
        return {
            "records": 0, "high_value_records": 0,
            "selected_completes_endpoint": 0, "selected_reduces_distance": 0,
            "selected_redundant": 0, "selected_off_chain": 0, "selected_other": 0,
            "endpoint_exists": 0,
            "endpoint_policy_top1": 0, "endpoint_policy_top5": 0,
            "endpoint_policy_top20": 0, "endpoint_policy_gt20": 0,
            "endpoint_visit_top1": 0, "endpoint_visit_top5": 0,
            "endpoint_visit_top20": 0, "endpoint_visit_gt20": 0,
            "reducer_exists": 0,
            "reducer_policy_top1": 0, "reducer_policy_top5": 0,
            "reducer_policy_top20": 0, "reducer_policy_gt20": 0,
            "reducer_visit_top1": 0, "reducer_visit_top5": 0,
            "reducer_visit_top20": 0, "reducer_visit_gt20": 0,
        }

    def _bucket_rank(rank, c, prefix):
        if rank is None:
            return
        if rank <= 1:
            c[f"{prefix}_top1"] += 1; c[f"{prefix}_top5"] += 1; c[f"{prefix}_top20"] += 1
        elif rank <= 5:
            c[f"{prefix}_top5"] += 1; c[f"{prefix}_top20"] += 1
        elif rank <= 20:
            c[f"{prefix}_top20"] += 1
        else:
            c[f"{prefix}_gt20"] += 1

    buckets = {"td=1": _empty(), "td=2": _empty(), "td=3": _empty()}
    for rec, det in zip(records, detected_sides):
        if not isinstance(rec, dict) or rec.get("side_to_move") != det:
            continue
        gc = rec.get("goal_completion") or {}
        td = gc.get("total_goal_distance_before")
        if td not in (1, 2, 3):
            continue
        b = buckets[f"td={td}"]
        b["records"] += 1
        q = (rec.get("root_summary") or {}).get("q_value")
        if isinstance(q, (int, float)) and q >= high_value_threshold:
            b["high_value_records"] += 1
        cls_name = ((rec.get("selected_move_classification") or {}).get("primary_class")) or ""
        cls_field = {
            "completes_endpoint": "selected_completes_endpoint",
            "reduces_total_goal_distance": "selected_reduces_distance",
            "redundant_reinforcement": "selected_redundant",
            "off_chain": "selected_off_chain",
            "other": "selected_other",
        }.get(cls_name)
        if cls_field:
            b[cls_field] += 1
        ec = rec.get("endpoint_completion_ranking") or {}
        if ec.get("best_policy_rank") is not None or ec.get("best_visit_rank") is not None:
            b["endpoint_exists"] += 1
            _bucket_rank(ec.get("best_policy_rank"), b, "endpoint_policy")
            _bucket_rank(ec.get("best_visit_rank"),  b, "endpoint_visit")
        rd = rec.get("distance_reducing_ranking") or {}
        if rd.get("best_policy_rank") is not None or rd.get("best_visit_rank") is not None:
            b["reducer_exists"] += 1
            _bucket_rank(rd.get("best_policy_rank"), b, "reducer_policy")
            _bucket_rank(rd.get("best_visit_rank"),  b, "reducer_visit")

    def _rate(num, den):
        return (num / den) if den > 0 else 0.0

    out = {}
    for key, b in buckets.items():
        n = b["records"]; e = b["endpoint_exists"]; r = b["reducer_exists"]
        out[key] = {
            "records": n,
            "high_value_records": b["high_value_records"],
            "selected_completes_endpoint_rate": _rate(b["selected_completes_endpoint"], n),
            "selected_reduces_distance_rate":   _rate(b["selected_reduces_distance"], n),
            "selected_redundant_rate":          _rate(b["selected_redundant"], n),
            "selected_off_chain_rate":          _rate(b["selected_off_chain"], n),
            "selected_other_rate":              _rate(b["selected_other"], n),
            "endpoint_completion_exists_rate":  _rate(e, n),
            "endpoint_policy_top1_rate":  _rate(b["endpoint_policy_top1"], e),
            "endpoint_policy_top5_rate":  _rate(b["endpoint_policy_top5"], e),
            "endpoint_policy_top20_rate": _rate(b["endpoint_policy_top20"], e),
            "endpoint_policy_gt20_rate":  _rate(b["endpoint_policy_gt20"], e),
            "endpoint_visit_top1_rate":   _rate(b["endpoint_visit_top1"], e),
            "endpoint_visit_top5_rate":   _rate(b["endpoint_visit_top5"], e),
            "endpoint_visit_top20_rate":  _rate(b["endpoint_visit_top20"], e),
            "endpoint_visit_gt20_rate":   _rate(b["endpoint_visit_gt20"], e),
            "distance_reducer_exists_rate":   _rate(r, n),
            "reducer_policy_top1_rate":   _rate(b["reducer_policy_top1"], r),
            "reducer_policy_top5_rate":   _rate(b["reducer_policy_top5"], r),
            "reducer_policy_top20_rate":  _rate(b["reducer_policy_top20"], r),
            "reducer_policy_gt20_rate":   _rate(b["reducer_policy_gt20"], r),
            "reducer_visit_top1_rate":    _rate(b["reducer_visit_top1"], r),
            "reducer_visit_top5_rate":    _rate(b["reducer_visit_top5"], r),
            "reducer_visit_top20_rate":   _rate(b["reducer_visit_top20"], r),
            "reducer_visit_gt20_rate":    _rate(b["reducer_visit_gt20"], r),
        }
    return out


def aggregate_td_closeout_breakdown(
    per_ply_records: list,
    detected_player: str,
    high_value_threshold: float = 0.95,
) -> dict:
    """Bucket strict closeout per-ply records by total_goal_distance_before.

    Spec 2026-05-10 §3. Reads records of the form emitted in
    goal_completion_diagnostics (see closeout_diagnostics.build_*).

    Returns a dict keyed "td=1" / "td=2" / "td=3" with the metric set
    described in spec §3.1.
    """
    def _empty():
        return {
            "records": 0,
            "high_value_records": 0,
            "selected_completes_endpoint": 0,
            "selected_reduces_distance": 0,
            "selected_redundant": 0,
            "selected_off_chain": 0,
            "selected_other": 0,
            "endpoint_exists": 0,
            "endpoint_policy_top1": 0,
            "endpoint_policy_top5": 0,
            "endpoint_policy_top20": 0,
            "endpoint_policy_gt20": 0,
            "endpoint_visit_top1": 0,
            "endpoint_visit_top5": 0,
            "endpoint_visit_top20": 0,
            "endpoint_visit_gt20": 0,
            "reducer_exists": 0,
            "reducer_policy_top1": 0,
            "reducer_policy_top5": 0,
            "reducer_policy_top20": 0,
            "reducer_policy_gt20": 0,
            "reducer_visit_top1": 0,
            "reducer_visit_top5": 0,
            "reducer_visit_top20": 0,
            "reducer_visit_gt20": 0,
        }

    buckets = {"td=1": _empty(), "td=2": _empty(), "td=3": _empty()}

    def _bucket_rank(rank, c):
        if rank is None:
            return
        if rank <= 1:
            c["_top1"] += 1; c["_top5"] += 1; c["_top20"] += 1
        elif rank <= 5:
            c["_top5"] += 1; c["_top20"] += 1
        elif rank <= 20:
            c["_top20"] += 1
        else:
            c["_gt20"] += 1

    for rec in per_ply_records or []:
        if not isinstance(rec, dict):
            continue
        if rec.get("side_to_move") != detected_player:
            continue
        gc = rec.get("goal_completion") or {}
        td = gc.get("total_goal_distance_before")
        if td not in (1, 2, 3):
            continue
        key = f"td={td}"
        b = buckets[key]
        b["records"] += 1
        q = (rec.get("root_summary") or {}).get("q_value")
        if isinstance(q, (int, float)) and q >= high_value_threshold:
            b["high_value_records"] += 1
        cls_name = ((rec.get("selected_move_classification") or {}).get("primary_class")) or ""
        cls_field = {
            "completes_endpoint": "selected_completes_endpoint",
            "reduces_total_goal_distance": "selected_reduces_distance",
            "redundant_reinforcement": "selected_redundant",
            "off_chain": "selected_off_chain",
            "other": "selected_other",
        }.get(cls_name)
        if cls_field is not None:
            b[cls_field] += 1
        # Endpoint completion ranking buckets (denominator: endpoint_exists)
        ec = rec.get("endpoint_completion_ranking") or {}
        epr = ec.get("best_policy_rank")
        evr = ec.get("best_visit_rank")
        if epr is not None or evr is not None:
            b["endpoint_exists"] += 1
            tmp_p = {"_top1": 0, "_top5": 0, "_top20": 0, "_gt20": 0}
            _bucket_rank(epr, tmp_p)
            tmp_v = {"_top1": 0, "_top5": 0, "_top20": 0, "_gt20": 0}
            _bucket_rank(evr, tmp_v)
            for k in ("_top1", "_top5", "_top20", "_gt20"):
                b[f"endpoint_policy{k}"] += tmp_p[k]
                b[f"endpoint_visit{k}"] += tmp_v[k]
        # Distance reducer ranking buckets
        rd = rec.get("distance_reducing_ranking") or {}
        rpr = rd.get("best_policy_rank")
        rvr = rd.get("best_visit_rank")
        if rpr is not None or rvr is not None:
            b["reducer_exists"] += 1
            tmp_p = {"_top1": 0, "_top5": 0, "_top20": 0, "_gt20": 0}
            _bucket_rank(rpr, tmp_p)
            tmp_v = {"_top1": 0, "_top5": 0, "_top20": 0, "_gt20": 0}
            _bucket_rank(rvr, tmp_v)
            for k in ("_top1", "_top5", "_top20", "_gt20"):
                b[f"reducer_policy{k}"] += tmp_p[k]
                b[f"reducer_visit{k}"] += tmp_v[k]

    # Convert raw counts to rates
    def _rate(num, den):
        return (num / den) if den > 0 else 0.0

    out = {}
    for key, b in buckets.items():
        n = b["records"]
        e_exists = b["endpoint_exists"]
        r_exists = b["reducer_exists"]
        out[key] = {
            "records": n,
            "high_value_records": b["high_value_records"],
            "selected_completes_endpoint_rate": _rate(b["selected_completes_endpoint"], n),
            "selected_reduces_distance_rate":   _rate(b["selected_reduces_distance"], n),
            "selected_redundant_rate":          _rate(b["selected_redundant"], n),
            "selected_off_chain_rate":          _rate(b["selected_off_chain"], n),
            "selected_other_rate":              _rate(b["selected_other"], n),
            "endpoint_completion_exists_rate":  _rate(e_exists, n),
            "endpoint_policy_top1_rate":  _rate(b["endpoint_policy_top1"], e_exists),
            "endpoint_policy_top5_rate":  _rate(b["endpoint_policy_top5"], e_exists),
            "endpoint_policy_top20_rate": _rate(b["endpoint_policy_top20"], e_exists),
            "endpoint_policy_gt20_rate":  _rate(b["endpoint_policy_gt20"], e_exists),
            "endpoint_visit_top1_rate":   _rate(b["endpoint_visit_top1"], e_exists),
            "endpoint_visit_top5_rate":   _rate(b["endpoint_visit_top5"], e_exists),
            "endpoint_visit_top20_rate":  _rate(b["endpoint_visit_top20"], e_exists),
            "endpoint_visit_gt20_rate":   _rate(b["endpoint_visit_gt20"], e_exists),
            "distance_reducer_exists_rate":   _rate(r_exists, n),
            "reducer_policy_top1_rate":   _rate(b["reducer_policy_top1"], r_exists),
            "reducer_policy_top5_rate":   _rate(b["reducer_policy_top5"], r_exists),
            "reducer_policy_top20_rate":  _rate(b["reducer_policy_top20"], r_exists),
            "reducer_policy_gt20_rate":   _rate(b["reducer_policy_gt20"], r_exists),
            "reducer_visit_top1_rate":    _rate(b["reducer_visit_top1"], r_exists),
            "reducer_visit_top5_rate":    _rate(b["reducer_visit_top5"], r_exists),
            "reducer_visit_top20_rate":   _rate(b["reducer_visit_top20"], r_exists),
            "reducer_visit_gt20_rate":    _rate(b["reducer_visit_gt20"], r_exists),
        }
    return out


def aggregate_recovery_events(replays: list) -> list:
    """Build per-event rows for the recovery diagnostic (spec §6).

    Event criterion (§6.1): a replay contributes an event when any of
    - winner_moves_with_dominant_unavailable >= 10
    - meta.reason == "state_cap" AND record.detected == True (or
      winner_moves_in_watch_window > 0 as a proxy for detection)
    - meta.reason == "adjudicated" AND winner_moves_with_dominant_unavailable >= 5
    """
    events = []
    for replay in replays or []:
        rec = replay.get("goal_completion_record")
        if not isinstance(rec, dict):
            continue
        meta = replay.get("meta") or {}
        reason = meta.get("reason")
        dom_unavail = rec.get("winner_moves_with_dominant_unavailable") or 0
        in_window = rec.get("winner_moves_in_watch_window") or 0
        detected = (in_window or 0) > 0

        triggered = (
            (dom_unavail or 0) >= 10
            or (reason == "state_cap" and detected)
            or (reason == "adjudicated" and (dom_unavail or 0) >= 5)
        )
        if not triggered:
            continue

        # Optional per-ply walk to find first_unavailable_ply (first detected-side
        # ply where total_goal_distance_before > 2).
        det_side = rec.get("detected_player")
        first_unavailable_ply = None
        q_at_first_unavailable = None
        diag = replay.get("goal_completion_diagnostics") or []
        for r in diag:
            if not isinstance(r, dict):
                continue
            if r.get("side_to_move") != det_side:
                continue
            gc = r.get("goal_completion") or {}
            td = gc.get("total_goal_distance_before")
            if td is not None and td > 2:
                first_unavailable_ply = r.get("ply")
                q_at_first_unavailable = (r.get("root_summary") or {}).get("q_value")
                break
        # latest fields from last detected-side row in diag
        latest_largest = None; latest_td = None
        for r in diag:
            if isinstance(r, dict) and r.get("side_to_move") == det_side:
                gc = r.get("goal_completion") or {}
                latest_largest = gc.get("largest_component_size")
                latest_td = gc.get("total_goal_distance_before")
        sel_class_counts = {"completes_endpoint": 0, "reduces_total_goal_distance": 0,
                            "redundant_reinforcement": 0, "off_chain": 0, "other": 0}
        if first_unavailable_ply is not None:
            for r in diag:
                if not isinstance(r, dict):
                    continue
                if r.get("side_to_move") != det_side:
                    continue
                if (r.get("ply") or 0) < first_unavailable_ply:
                    continue
                cls = ((r.get("selected_move_classification") or {}).get("primary_class"))
                if cls in sel_class_counts:
                    sel_class_counts[cls] += 1

        q_at_terminal = meta.get("final_root_value")
        outcome = "win" if reason == "win" else (
            "state_cap" if reason == "state_cap" else (
                "adjudicated" if reason == "adjudicated" else (reason or "other")
            )
        )

        delay_winner = rec.get("conversion_delay_winner_moves") or 0
        # Bucket assignment (priority order, §6.3)
        recovered_later_to_le2 = any(
            (r.get("side_to_move") == det_side
             and (r.get("goal_completion") or {}).get("total_goal_distance_before") is not None
             and (r.get("goal_completion") or {}).get("total_goal_distance_before") <= 2
             and (first_unavailable_ply is not None and (r.get("ply") or 0) > first_unavailable_ply))
            for r in diag if isinstance(r, dict)
        )
        if outcome == "win" and (dom_unavail or 0) >= 10 and recovered_later_to_le2:
            bucket = "lost_then_recovered"
        elif outcome == "win" and delay_winner >= 30:
            bucket = "lost_then_won_late"
        elif outcome == "state_cap":
            bucket = "lost_then_state_cap"
        elif (q_at_first_unavailable is not None and q_at_first_unavailable >= 0.9
              and (q_at_terminal or 0) <= 0.5):
            bucket = "lost_and_value_collapsed"
        elif (q_at_first_unavailable is not None and q_at_first_unavailable >= 0.9
              and (q_at_terminal or 0) >= 0.9):
            bucket = "lost_but_value_stayed_high"
        else:
            bucket = "lost_other"

        events.append({
            "iteration": meta.get("iteration"),
            "game_id": rec.get("game_id"),
            "winner": rec.get("winner"),
            "detected_player": det_side,
            "first_detection_ply": rec.get("first_dominant_unclosed_ply"),
            "first_unavailable_ply": first_unavailable_ply,
            "dominant_unavailable_moves": dom_unavail,
            "conversion_delay_winner_moves": delay_winner,
            "latest_largest_component_size": latest_largest,
            "latest_total_goal_distance": latest_td,
            "q_at_first_unavailable": q_at_first_unavailable,
            "q_at_terminal": q_at_terminal,
            "selected_class_counts_after_first_unavailable": sel_class_counts,
            "eventual_outcome": outcome,
            "recovery_class": bucket,
        })
    return events


def aggregate_goal_completion_diagnostics_from_records(
    replays: list, sidecar_summaries: dict, config: dict,
) -> dict:
    """Default analyzer path (spec §11.1).

    Per-game records are canonical. Sidecar summaries are held for
    validation / iteration telemetry but the cross-iteration roll-up
    is recomputed from records.

    Emits aggregated warnings for missing records, sidecar/replay
    coverage mismatches, and version drift.
    """
    per_game_records = [replay.get("goal_completion_record") for replay in replays]
    # Variable convention here: `rec` is the per-game record (which may be None),
    # `replay` is the full replay JSON dict. Names kept distinct to avoid the
    # "r is None then r['game_id'] crashes" bug class flagged in spec §11.3 review.
    missing = [
        (idx, replay) for idx, (rec, replay) in enumerate(zip(per_game_records, replays))
        if rec is None
    ]
    n_missing = len(missing)
    n_total = len(replays)

    if n_missing == n_total and n_total > 0:
        print(
            f"[WARN] {n_missing}/{n_total} replays missing "
            f"goal_completion_record. Goal-completion report skipped. "
            f"Run with --goal-completion-recompute or rerun training "
            f"with goal_completion_record_enabled=True.",
            file=sys.stderr,
        )
    elif n_missing > 0:
        examples = []
        for _, replay in missing[:3]:
            # Inline record is by definition None here, so we read identity
            # off the replay JSON directly. Synthesized id matches the
            # iter_NNNN_game_NNN naming the saver writes.
            gid = (
                f"iter_{_replay_int_field(replay, 'iteration'):04d}"
                f"_game_{_replay_int_field(replay, 'game_idx'):03d}"
            )
            examples.append(gid)
        print(
            f"[WARN] {n_missing}/{n_total} replays missing "
            f"goal_completion_record. Examples: {', '.join(examples)}.",
            file=sys.stderr,
        )

    # Sidecar / replay reconciliation per iteration.
    if sidecar_summaries:
        per_iter_record_counts: dict = {}
        for replay, rec in zip(replays, per_game_records):
            if rec is None:
                continue
            it = _replay_int_field(replay, "iteration", default=-1)
            if it >= 0:
                per_iter_record_counts[it] = per_iter_record_counts.get(it, 0) + 1
        for it, summary in sidecar_summaries.items():
            sidecar_n = (summary.get("diagnostics_coverage") or {}).get("games_with_record")
            replay_n = per_iter_record_counts.get(it, 0)
            if sidecar_n is not None and sidecar_n != replay_n:
                print(
                    f"[WARN] Goal-completion sidecar/replay mismatch for "
                    f"iter {it:04d}: sidecar games_with_record={sidecar_n}, "
                    f"replay records found={replay_n}. Using per-game records "
                    f"as canonical analyzer source.",
                    file=sys.stderr,
                )
            sidecar_version = summary.get("version")
            if sidecar_version is not None and sidecar_version != 1:
                print(
                    f"[WARN] Goal-completion version mismatch for iter "
                    f"{it:04d}: sidecar version={sidecar_version}, "
                    f"per-game records canonical (treating as v1).",
                    file=sys.stderr,
                )

    result = aggregate_goal_completion_records(
        per_game_records, config=config, games_total=n_total,
    )
    # Fix 0 (spec 2026-05-10 §3): bulk td-before breakdown across decisive
    # winners. Each replay's per-ply diagnostic records carry the
    # winner's perspective; pair them with the per-game record's
    # detected_player so we can filter to side_to_move == detected.
    td_breakdown_records = []
    td_breakdown_detected = []
    for replay, rec in zip(replays, per_game_records):
        if rec is None:
            continue
        det = rec.get("detected_player")
        if not det:
            continue
        diag = replay.get("goal_completion_diagnostics") or []
        for r in diag:
            if isinstance(r, dict):
                td_breakdown_records.append(r)
                td_breakdown_detected.append(det)
    result["td_closeout_breakdown"] = _aggregate_td_breakdown_multi_side(
        td_breakdown_records, td_breakdown_detected, high_value_threshold=0.95,
    )
    # Surface Phase 3 detailed-record aggregation alongside the compact
    # record summary. The default path was missing this in Spec 1.5; the
    # data IS in the per-game JSONs, just not aggregated here.
    n_decisive = (result.get("main_population") or {}).get("n", 0) or (
        result.get("main_population") or {}
    ).get("games", 0)
    phase3 = _surface_phase3_diagnostics(replays, n_decisive)
    # Phase 3's diagnostics_coverage is DIFFERENT from the Spec 1.5
    # record-coverage block (which lives at result["diagnostics_coverage"]).
    # Spec 1's analyzer used the same key name for the Phase 3 block;
    # preserve that contract so format_policy_mcts_closeout_report finds it.
    # We move the Spec 1.5 record-coverage to a separate key first.
    if "diagnostics_coverage" in result:
        result["record_coverage"] = result.pop("diagnostics_coverage")
    result["diagnostics_coverage"] = phase3["diagnostics_coverage"]
    result["policy_mcts_summary"] = phase3["policy_mcts_summary"]
    return result


def _merge_inline_with_recomputed(
    inline: list, recomputed: list,
) -> list:
    """Mixed-corpus merge (spec §13.2). Inline records preferred; gaps
    filled by recomputed records."""
    return [
        ir if ir is not None else rr
        for ir, rr in zip(inline, recomputed)
    ]


def aggregate_goal_completion_diagnostics(
    replays: List[dict],
    max_depth: int = 3,
    min_component_size: int = 8,
    detection_threshold: int = 2,
    high_value_threshold: float = 0.9,
    high_value_delay_threshold_plies: int = 10,
    worst_cases_top_k: int = 25,
) -> dict:
    """Per-game goal-completion analysis bucketed by outcome class.

    Phase 2 Task 8 scaffolding: builds Class 1 (decisive winner) per-game
    records and pools them into the main_population summary block. Class 2
    capped + Class 3 excluded populations are scaffolded with zero-value
    defaults; Task 9 fills in their detection logic.

    Pure function. Returns the goal_completion summary block per
    spec 2026-05-03 §7.4.

    Spec: docs/superpowers/specs/2026-05-03-goal-completion-diagnostics-design.md
    """
    config = {
        "max_depth": int(max_depth),
        "min_component_size": int(min_component_size),
        "detection_threshold": int(detection_threshold),
        "high_value_threshold": float(high_value_threshold),
        "high_value_delay_threshold_plies": int(high_value_delay_threshold_plies),
        "worst_cases_top_k": int(worst_cases_top_k),
    }

    class1_records: List[dict] = []
    capped_pop: dict = {
        "games": 0,
        "games_with_dominant_unclosed": 0,
        "detected_before_cap": 0,
        "per_game_records": [],
    }
    excluded_count = 0

    for rp in replays:
        meta = rp.get("meta") or {}
        reason = meta.get("reason")
        winner = rp.get("winner")
        moves = rp.get("moves") or []

        # Replays with no moves are excluded from goal-completion analysis (spec §7.7).
        if not moves:
            excluded_count += 1
            continue

        # Determine outcome class.
        if reason in _CLASS1_REASONS:
            # "reason == win" but winner == null is a corrupt record -> Class 3.
            if winner not in ("red", "black"):
                excluded_count += 1
                continue
            try:
                rec = _build_class1_per_game_record(
                    rp,
                    max_depth=max_depth,
                    min_component_size=min_component_size,
                    detection_threshold=detection_threshold,
                    high_value_threshold=high_value_threshold,
                    high_value_delay_threshold_plies=high_value_delay_threshold_plies,
                )
            except Exception:
                # Defensive: corrupt move history etc. -> exclude.
                excluded_count += 1
                continue
            class1_records.append(rec)
        elif reason in _CLASS2_REASONS:
            try:
                record = _build_class2_per_game_record(
                    rp,
                    max_depth=max_depth,
                    min_component_size=min_component_size,
                    detection_threshold=detection_threshold,
                )
            except Exception:
                # Defensive: corrupt move history etc. -> exclude.
                excluded_count += 1
                continue
            capped_pop["games"] += 1
            if record["ever_distance_le_3"]:
                capped_pop["games_with_dominant_unclosed"] += 1
            if record["detected"]:
                capped_pop["detected_before_cap"] += 1
            capped_pop["per_game_records"].append(record)
        else:
            excluded_count += 1

    main_population = _summarize_main_population(
        class1_records,
        config=config,
        detection_threshold=detection_threshold,
        high_value_threshold=high_value_threshold,
        high_value_delay_threshold_plies=high_value_delay_threshold_plies,
    )

    capped_population = _summarize_capped_population(capped_pop)

    excluded_population = {"games": excluded_count}

    # Phase 3 surfacing — extracted to _surface_phase3_diagnostics for reuse
    # across legacy and default analyzer paths.
    n_decisive = main_population.get("games", 0) if main_population else 0
    phase3 = _surface_phase3_diagnostics(replays, n_decisive)
    diagnostics_coverage = phase3["diagnostics_coverage"]
    policy_mcts_summary = phase3["policy_mcts_summary"]

    return {
        "config": config,
        "main_population": main_population,
        "capped_population": capped_population,
        "excluded_population": excluded_population,
        "diagnostics_coverage": diagnostics_coverage,
        "policy_mcts_summary": policy_mcts_summary,
    }


def _summarize_policy_mcts(records: list) -> dict:
    """Pool closeout-diagnostic records into the policy_mcts_summary block.

    Args:
        records: list of per-ply closeout records (from
            replay["goal_completion_diagnostics"], pooled across replays).
    Returns:
        Dict with n_records, endpoint+distance-reducing ranking pools (with
        n_rankable denominators), primary_class rates, high_value_delayed
        counter, and by_distance buckets (le_2 / eq_3).
    """
    n_records = len(records)
    primary_counts = {k: 0 for k in (
        "completes_endpoint", "reduces_total_goal_distance",
        "redundant_reinforcement", "off_chain", "other"
    )}
    high_value_delayed = 0
    for r in records:
        cls = r.get("selected_move_classification") or {}
        pc = cls.get("primary_class")
        if pc in primary_counts:
            primary_counts[pc] += 1
        rs = r.get("root_summary") or {}
        gc = r.get("goal_completion") or {}
        if (
            (rs.get("q_value") or 0.0) >= 0.9
            and pc in ("redundant_reinforcement", "off_chain", "other")
            and (gc.get("total_goal_distance_before") or 99) <= 2
        ):
            high_value_delayed += 1

    def _ranking_pool(records, key):
        rankable = [r.get(key) for r in records if r.get(key) is not None]
        n = len(rankable)
        if n == 0:
            return {"n_rankable": 0, "policy_top1_rate": 0.0, "policy_top5_rate": 0.0,
                    "visit_top1_rate": 0.0, "visit_top5_rate": 0.0}
        return {
            "n_rankable": n,
            "policy_top1_rate": sum(
                1 for b in rankable if (b.get("best_policy_rank") or 99) == 1
            ) / n,
            "policy_top5_rate": sum(
                1 for b in rankable if b.get("any_in_policy_top5", False)
            ) / n,
            "visit_top1_rate": sum(
                1 for b in rankable if (b.get("best_visit_rank") or 99) == 1
            ) / n,
            "visit_top5_rate": sum(
                1 for b in rankable if b.get("any_in_visit_top5", False)
            ) / n,
        }

    by_distance = {"distance_le_2": [], "distance_eq_3": []}
    for r in records:
        gc = r.get("goal_completion") or {}
        total = gc.get("total_goal_distance_before")
        if total is None:
            continue
        if total <= 2:
            by_distance["distance_le_2"].append(r)
        elif total == 3:
            by_distance["distance_eq_3"].append(r)

    return {
        "n_records": n_records,
        "endpoint_completion_ranking": _ranking_pool(records, "endpoint_completion_ranking"),
        "distance_reducing_ranking":   _ranking_pool(records, "distance_reducing_ranking"),
        "selected_primary_class_rates": {
            k: (v / max(n_records, 1)) for k, v in primary_counts.items()
        },
        "high_value_delayed_closeouts": high_value_delayed,
        "by_distance": {
            "distance_le_2": {"n": len(by_distance["distance_le_2"])},
            "distance_eq_3": {"n": len(by_distance["distance_eq_3"])},
        },
    }


def _build_class1_per_game_record(
    replay: dict,
    *,
    max_depth: int,
    min_component_size: int,
    detection_threshold: int,
    high_value_threshold: float,
    high_value_delay_threshold_plies: int,
) -> dict:
    """Construct a Class 1 per-game record by replaying the game (spec §7.2).

    Replays moves through TwixtState, computes per-ply
    compute_goal_completion_state(winner), records the first ply where
    total_goal_distance <= detection_threshold, and classifies subsequent
    winner-perspective moves via classify_selected_conversion_move.

    Watch window opens AFTER the detection ply on the in-scope side's
    subsequent moves. Non-winner plies are skipped. Detection ply itself
    is not classified.
    """
    meta = replay.get("meta") or {}
    moves = replay.get("moves") or []
    winner = replay["winner"]
    starting_player = (
        replay.get("starting_player")
        or meta.get("starting_player")
        or "red"
    )
    board_size = int(meta.get("board_size", 24))

    n_moves = meta.get("n_moves")
    if n_moves is None:
        n_moves = len(moves)
    n_moves = int(n_moves)

    # Pass 1: replay the game, tracking per-ply goal-completion states for the
    # winner. Captures pre-state at each ply for downstream classification.
    state = TwixtState(active_size=board_size, to_move=starting_player)
    pre_states: List[TwixtState] = []     # state immediately before move i (0-indexed)
    goal_states_after: List[Optional[dict]] = []  # goal_state for winner immediately after move i

    ever_le_2 = False
    ever_le_3 = False
    min_total = None  # min total_goal_distance ever seen for the winner
    first_dominant_unclosed_ply: Optional[int] = None  # 1-indexed ply where total <= detection_threshold
    first_total_goal_distance: Optional[int] = None
    first_category: Optional[str] = None

    for i, m in enumerate(moves):
        pre_states.append(state)
        try:
            state = state.apply_move((int(m["row"]), int(m["col"])))
        except Exception:
            # Corrupt move history -> bubble up; caller will exclude this game.
            raise
        # Per-ply detection only needs total_goal_distance + category;
        # skip the expensive endpoint_completion_moves / distance_reducing_moves
        # enumeration. Watch-window classification (below) re-computes with
        # enumerate_moves=True only on winner moves that actually need it.
        gs = compute_goal_completion_state(
            state,
            winner,
            max_depth=max_depth,
            min_component_size=min_component_size,
            enumerate_moves=False,
        )
        goal_states_after.append(gs)
        if gs is not None:
            total = gs.get("total_goal_distance")
            if total is not None:
                if min_total is None or total < min_total:
                    min_total = total
                if total <= 2:
                    ever_le_2 = True
                if total <= 3:
                    ever_le_3 = True
                if first_dominant_unclosed_ply is None and total <= detection_threshold:
                    first_dominant_unclosed_ply = i + 1  # 1-indexed
                    first_total_goal_distance = total
                    first_category = gs.get("category")

    detected = first_dominant_unclosed_ply is not None

    actual_terminal_ply = n_moves
    actual_win_ply = n_moves if winner in ("red", "black") else None

    # Watch-window classification (spec §7.2).
    winner_moves_in_watch_window = 0
    winner_moves_with_dominant_component = 0
    winner_moves_with_dominant_unavailable = 0
    primary_class_counts = {
        "completes_endpoint": 0,
        "reduces_total_goal_distance": 0,
        "redundant_reinforcement": 0,
        "off_chain": 0,
        "other": 0,
    }

    # search_score is winner-perspective at winner plies (state.to_move == winner); no sign flip needed
    search_scores_after_detection: List[float] = []  # only over winner moves with non-null search_score
    high_value_after_detection_plies = 0

    if detected:
        # Watch window: winner moves at ply > first_dominant_unclosed_ply,
        # through actual_terminal_ply (inclusive).
        for i, m in enumerate(moves):
            ply_1based = i + 1
            if ply_1based <= first_dominant_unclosed_ply:
                continue
            if ply_1based > actual_terminal_ply:
                break
            if m.get("player") != winner:
                continue

            winner_moves_in_watch_window += 1
            pre_state = pre_states[i]
            gs_before = compute_goal_completion_state(
                pre_state,
                winner,
                max_depth=max_depth,
                min_component_size=min_component_size,
            )
            if gs_before is None:
                winner_moves_with_dominant_unavailable += 1
            else:
                winner_moves_with_dominant_component += 1
                cls = classify_selected_conversion_move(
                    pre_state,
                    winner,
                    (int(m["row"]), int(m["col"])),
                    gs_before,
                    max_depth=max_depth,
                    min_component_size=min_component_size,
                )
                primary = cls.get("primary_class", "other")
                if primary in primary_class_counts:
                    primary_class_counts[primary] += 1
                else:
                    primary_class_counts["other"] += 1

            # search_score: only at winner plies, only if populated.
            ss = m.get("search_score")
            if ss is not None:
                search_scores_after_detection.append(float(ss))
                if float(ss) >= high_value_threshold:
                    high_value_after_detection_plies += 1

    # search_score-derived summaries (null when no coverage in watch window).
    if search_scores_after_detection:
        max_search_score_after_detection = float(max(search_scores_after_detection))
        mean_search_score_after_detection = float(
            sum(search_scores_after_detection) / len(search_scores_after_detection)
        )
        search_score_coverage_in_watch_window = len(search_scores_after_detection)
    else:
        max_search_score_after_detection = None
        mean_search_score_after_detection = None
        search_score_coverage_in_watch_window = 0

    # root_value_high_but_delayed (spec §7.2):
    # Class 1 + detected + >= 1 high-value post-detection winner ply +
    # conversion_delay_plies >= high_value_delay_threshold_plies.
    if detected:
        conversion_delay_plies = actual_terminal_ply - first_dominant_unclosed_ply
        # Count of winner moves strictly after detection through terminal.
        conversion_delay_winner_moves = winner_moves_in_watch_window
    else:
        conversion_delay_plies = None
        conversion_delay_winner_moves = None

    root_value_high_but_delayed = bool(
        detected
        and high_value_after_detection_plies >= 1
        and conversion_delay_plies is not None
        and conversion_delay_plies >= high_value_delay_threshold_plies
    )

    return {
        "game_id": replay.get("id"),
        "iteration": meta.get("iteration"),
        "game_idx": meta.get("game_idx"),
        "winner": winner,
        "starting_player": starting_player,
        "n_moves": n_moves,
        "reason": meta.get("reason"),
        "outcome_class": 1,
        "scope": "winner",
        "detected_player": winner,

        "ever_distance_le_2": ever_le_2,
        "ever_distance_le_3": ever_le_3,
        "min_total_goal_distance": min_total,

        "detected": detected,
        "first_dominant_unclosed_ply": first_dominant_unclosed_ply,
        "first_total_goal_distance": first_total_goal_distance,
        "first_category": first_category,
        "actual_terminal_ply": actual_terminal_ply,
        "actual_win_ply": actual_win_ply,

        "conversion_delay_plies": conversion_delay_plies,
        "conversion_delay_winner_moves": conversion_delay_winner_moves,

        "winner_moves_in_watch_window": winner_moves_in_watch_window,
        "winner_moves_with_dominant_component": winner_moves_with_dominant_component,
        "winner_moves_with_dominant_unavailable": winner_moves_with_dominant_unavailable,
        "primary_class_counts": primary_class_counts,

        "max_search_score_after_detection": max_search_score_after_detection,
        "mean_search_score_after_detection": mean_search_score_after_detection,
        "high_value_after_detection_plies": high_value_after_detection_plies,
        "root_value_high_but_delayed": root_value_high_but_delayed,
        "search_score_coverage_in_watch_window": search_score_coverage_in_watch_window,
    }


def _build_class2_per_game_record(
    replay: dict,
    *,
    max_depth: int,
    min_component_size: int,
    detection_threshold: int,
) -> dict:
    """Class 2 (capped/timeout/board_full): both-sides scope. Detection
    triggered by either side reaching dominant-unclosed before terminal.

    Spec: docs/superpowers/specs/2026-05-03-goal-completion-diagnostics-design.md §7.2.
    """
    meta = replay.get("meta") or {}
    moves = replay.get("moves") or []
    starting_player = (
        replay.get("starting_player")
        or meta.get("starting_player")
        or "red"
    )
    active = int(meta.get("board_size", 24))
    n_moves = meta.get("n_moves")
    if n_moves is None:
        n_moves = len(moves)
    n_moves = int(n_moves)

    state = TwixtState(active_size=active, to_move=starting_player)
    first_detected_ply: Optional[int] = None
    first_detected_player: Optional[str] = None
    first_total: Optional[int] = None
    first_category: Optional[str] = None
    ever_le_2 = False
    ever_le_3 = False
    min_total: Optional[int] = None

    for ply_idx, m in enumerate(moves):
        state = state.apply_move((int(m["row"]), int(m["col"])))
        # Check both sides at this ply. Tie-break: lower total wins; on equality
        # prefer red over black (loop order). For first_detected_ply we keep the
        # FIRST side seen at the first qualifying ply.
        for player in ("red", "black"):
            # Class 2 detection: same per-ply optimization as Class 1.
            gc = compute_goal_completion_state(
                state, player,
                max_depth=max_depth, min_component_size=min_component_size,
                enumerate_moves=False,
            )
            if gc is None or gc.get("total_goal_distance") is None:
                continue
            t = gc["total_goal_distance"]
            if min_total is None or t < min_total:
                min_total = t
            if t <= 2:
                ever_le_2 = True
            if t <= 3:
                ever_le_3 = True
            if t <= detection_threshold and first_detected_ply is None:
                first_detected_ply = ply_idx + 1
                first_detected_player = player
                first_total = t
                first_category = gc.get("category")

    cap_delay = (
        n_moves - first_detected_ply if first_detected_ply is not None else None
    )

    return {
        "game_id": replay.get("id"),
        "iteration": meta.get("iteration"),
        "game_idx": meta.get("game_idx"),
        "winner": None,
        "starting_player": starting_player,
        "n_moves": n_moves,
        "reason": meta.get("reason"),
        "outcome_class": 2,
        "scope": "both_sides",
        "detected_player": first_detected_player,
        "ever_distance_le_2": ever_le_2,
        "ever_distance_le_3": ever_le_3,
        "min_total_goal_distance": min_total,
        "detected": first_detected_ply is not None,
        "first_dominant_unclosed_ply": first_detected_ply,
        "first_total_goal_distance": first_total,
        "first_category": first_category,
        "actual_terminal_ply": n_moves,
        "actual_win_ply": None,
        "conversion_delay_plies": None,
        "conversion_delay_winner_moves": None,
        "cap_delay_after_detection_plies": cap_delay,
        "winner_moves_in_watch_window": None,
        "winner_moves_with_dominant_component": None,
        "winner_moves_with_dominant_unavailable": None,
        "primary_class_counts": None,
        "max_search_score_after_detection": None,
        "mean_search_score_after_detection": None,
        "high_value_after_detection_plies": None,
        "root_value_high_but_delayed": None,
        "search_score_coverage_in_watch_window": 0,
    }


def _summarize_main_population(
    records: List[dict],
    *,
    config: dict,
    detection_threshold: int,
    high_value_threshold: float,
    high_value_delay_threshold_plies: int,
) -> dict:
    """Pool Class 1 per-game records into the main_population summary.

    DEPRECATED: this is a thin delegating shim around
    scripts.GPU.alphazero.goal_completion_aggregator._summarize_main_population
    (which is the canonical implementation post-Spec 1.5). Bridges the
    legacy (kwarg-thresholds) signature to the new (config-bundled) one.
    Adds the legacy `_per_game_records_internal` key for backward compat
    with any consumer still using it.
    """
    merged_config = dict(config) if config else {}
    merged_config.setdefault("detection_threshold", detection_threshold)
    merged_config.setdefault("high_value_threshold", high_value_threshold)
    merged_config.setdefault("high_value_delay_threshold_plies", high_value_delay_threshold_plies)
    out = dict(_aggregator_summarize_main_population(records, merged_config))
    # Legacy back-compat: the aggregator returns {"n": 0} for empty input, but
    # legacy tests assert on "games" and "detected" keys; ensure they are present.
    out.setdefault("games", out.get("n", 0))
    out.setdefault("detected", 0)
    # Legacy back-compat: preserve the field that the old worst-cases CSV
    # writer used to read. Spec 1.5's new writer reads from per-game records
    # directly, but the legacy aggregate_goal_completion_diagnostics flow
    # may still expect this.
    out["_per_game_records_internal"] = records
    return out


def _summarize_capped_population(capped_pop: dict) -> dict:
    """Roll up Class 2 per-game records into the capped_population summary.

    DEPRECATED: thin delegating shim around
    scripts.GPU.alphazero.goal_completion_aggregator._summarize_capped_population
    (canonical post-Spec 1.5). Bridges the legacy
    `{"per_game_records": [...]}` input shape to the new flat-list shape.
    """
    records = capped_pop.get("per_game_records") or []
    out = dict(_aggregator_summarize_capped_population(records))
    # Legacy back-compat: the aggregator returns {"n": 0} for empty input, but
    # legacy tests assert on "games" key; ensure it is always present.
    out.setdefault("games", out.get("n", 0))
    # Legacy back-compat: tests assert on _per_game_records_internal to inspect
    # individual per-game fields after aggregation.
    out["_per_game_records_internal"] = records
    return out


def aggregate_per_move_stats(replays: List[dict]) -> dict:
    """Aggregate per-move search_score and root_top1_share distributions across replays.

    Reads moves[i].search_score and moves[i].root_top1_share, treating absent
    keys or null values as not-covered (move-count denominators, not
    game-count). Returns the per_move_stats summary block per spec 2026-05-03 §5.4.
    """
    import numpy as np

    n_games_total = len(replays)
    n_moves_total = 0
    search_score_vals: List[float] = []
    top1_share_vals: List[float] = []

    for replay in replays:
        moves = replay.get("moves") or []
        for m in moves:
            n_moves_total += 1
            ss = m.get("search_score")
            if ss is not None:
                search_score_vals.append(float(ss))
            ts = m.get("root_top1_share")
            if ts is not None:
                top1_share_vals.append(float(ts))

    def _stats(vals: List[float]) -> Optional[dict]:
        if not vals:
            return None
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean":     float(np.mean(arr)),
            "p50":      float(np.percentile(arr, 50)),
            "p90":      float(np.percentile(arr, 90)),
            "p95":      float(np.percentile(arr, 95)),
            "min":      float(np.min(arr)),
            "max":      float(np.max(arr)),
        }

    ss_block = _stats(search_score_vals)
    if ss_block is not None:
        ss_block["mean_abs"] = float(np.mean(np.abs(np.array(search_score_vals))))

    return {
        "n_games_total": n_games_total,
        "n_moves_total": n_moves_total,
        "coverage": {
            "search_score":    len(search_score_vals),
            "root_top1_share": len(top1_share_vals),
        },
        "search_score":    ss_block,
        "root_top1_share": _stats(top1_share_vals),
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


def format_conversion_training_trend_report(sidecar_summaries: dict) -> list:
    """Read-only roll-up of conversion_training blocks across iters.

    sidecar_summaries: {iteration_int: conversion_training_block_dict}
    Returns a list of report-line strings.
    """
    lines = ["── Conversion-training trend ─────────────────────────────────"]
    if not sidecar_summaries:
        lines.append("  (no conversion_training data)")
        lines.append("──────────────────────────────────────────────────────────────")
        return lines
    iters = sorted(sidecar_summaries.keys())
    lines.append(f"Iters covered:   {iters[0]}-{iters[-1]}")

    weights = sorted({
        sidecar_summaries[i]["config"].get("effective_loss_weight", 0.0)
        for i in iters
    })
    if len(weights) == 1:
        lines.append(f"Aux loss weight: {weights[0]} (constant)")
    else:
        lines.append(f"Aux loss weight: varies ({weights})")

    aux_losses = [sidecar_summaries[i]["loss"]["aux_loss_avg_iter"] for i in iters]
    lines.append("Aux loss (avg):  " + " → ".join(f"{x:.2f}" for x in aux_losses))
    coverages = [sidecar_summaries[i]["loss"]["aux_target_coverage_rate"] for i in iters]
    lines.append("Coverage rate:   " + " → ".join(f"{x*100:.1f}%" for x in coverages))
    matches = [sidecar_summaries[i]["consistency"].get("drawn_vs_seen_match", None) for i in iters]
    avail = [sidecar_summaries[i]["consistency"].get("available", False) for i in iters]
    if all(avail) and all(m is True for m in matches):
        consistency_summary = "✓ all iters consistent"
    elif not any(avail):
        consistency_summary = "(consistency check unavailable — Phase 2 only)"
    else:
        consistency_summary = "✗ DIVERGENCE — check WARNs"
    lines.append(f"Drawn vs seen:   {consistency_summary}")
    lines.append("──────────────────────────────────────────────────────────────")
    return lines


def format_recovery_or_extreme_closeout_drift_report(sidecar_summaries: dict) -> list:
    """Read-only roll-up of recovery blocks across iters."""
    lines = ["── Recovery / extreme-closeout-drift (telemetry only) ────────"]
    if not sidecar_summaries:
        lines.append("  (no recovery data)")
        lines.append("──────────────────────────────────────────────────────────────")
        return lines
    iters = sorted(sidecar_summaries.keys())
    lines.append(f"Iters covered:        {iters[0]}-{iters[-1]}")
    counts = [sidecar_summaries[i]["count"] for i in iters]
    rates = [sidecar_summaries[i]["rate"] for i in iters]
    p90s = [sidecar_summaries[i]["dominant_unavailable_moves"]["p90"] for i in iters]
    lines.append("Recovery count/iter:  " + " → ".join(str(x) for x in counts))
    lines.append("Recovery rate:        " + " → ".join(f"{x*100:.1f}%" for x in rates))
    lines.append("DU moves p90:         " + " → ".join(str(x) for x in p90s))
    lines.append("──────────────────────────────────────────────────────────────")
    return lines


def write_conversion_training_by_iter_csv(
    sidecar_summaries: dict, output_dir, suffix: str = "",
) -> str:
    """Write conversion_training_by_iter.csv — one row per iter."""
    import csv
    from pathlib import Path
    fieldnames = [
        "iteration", "cnv_enabled", "cnv_loss_weight", "cnv_aux_loss_avg",
        "cnv_aux_coverage", "cnv_aux_seen", "cnv_eligible_in_buf",
        "cnv_eligible_at_size", "cnv_drawn_total", "cnv_drawn_vs_seen_ok",
    ]
    out_path = Path(output_dir) / f"conversion_training_by_iter{suffix}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for it in sorted(sidecar_summaries.keys()):
            s = sidecar_summaries[it]
            cons = s.get("consistency", {})
            w.writerow({
                "iteration": it,
                "cnv_enabled": int(s.get("enabled", False)),
                "cnv_loss_weight": s.get("config", {}).get("effective_loss_weight", 0.0),
                "cnv_aux_loss_avg": s.get("loss", {}).get("aux_loss_avg_iter", 0.0),
                "cnv_aux_coverage": s.get("loss", {}).get("aux_target_coverage_rate", 0.0),
                "cnv_aux_seen": s.get("loss", {}).get("aux_positions_seen_in_training", 0),
                "cnv_eligible_in_buf": s.get("buffer", {}).get("eligible_positions_in_buffer", 0),
                "cnv_eligible_at_size": s.get("buffer", {}).get("eligible_positions_at_active_size", 0),
                "cnv_drawn_total": s.get("sample_stats", {}).get("eligible_drawn_total", 0),
                "cnv_drawn_vs_seen_ok": int(cons.get("drawn_vs_seen_match") is True),
            })
    return str(out_path)


def write_recovery_or_extreme_closeout_drift_by_iter_csv(
    sidecar_summaries: dict, output_dir, suffix: str = "",
) -> str:
    """Write recovery_or_extreme_closeout_drift_by_iter.csv."""
    import csv
    from pathlib import Path
    fieldnames = ["iteration", "rcv_count", "rcv_rate", "rcv_du_p90"]
    out_path = Path(output_dir) / f"recovery_or_extreme_closeout_drift_by_iter{suffix}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for it in sorted(sidecar_summaries.keys()):
            s = sidecar_summaries[it]
            w.writerow({
                "iteration": it,
                "rcv_count": s.get("count", 0),
                "rcv_rate": s.get("rate", 0.0),
                "rcv_du_p90": s.get("dominant_unavailable_moves", {}).get("p90", 0),
            })
    return str(out_path)


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


def format_per_move_stats_report(per_move_stats: dict) -> List[str]:
    """Render the per-move stats block as report.txt lines.

    Suppresses the Coverage line only when both fields have FULL coverage
    over all moves (n_moves_with_any_stats == n_moves_total and per-field
    coverage equals n_moves_total). Falls back to a short message when
    no moves carry any per-move stats.
    """
    n_total = per_move_stats.get("n_moves_total", 0)
    cov = per_move_stats.get("coverage") or {}
    cov_ss = cov.get("search_score", 0)
    cov_ts = cov.get("root_top1_share", 0)
    n_with_any = max(cov_ss, cov_ts)
    lines: List[str] = []
    if n_with_any == 0:
        lines.append(
            "Per-move stats: no moves carry new fields "
            "(all replays predate persistence change)."
        )
        lines.append("")
        return lines

    header_n = f"n={n_with_any:,} / {n_total:,}"
    lines.append(f"Per-move stats ({header_n} moves carry new fields):")

    ss = per_move_stats.get("search_score")
    if ss is not None:
        lines.append(
            f"  search_score:    mean={ss['mean']:.2f} p50={ss['p50']:.2f} "
            f"p90={ss['p90']:.2f} p95={ss['p95']:.2f} "
            f"(range [{ss['min']:.2f}, {ss['max']:.2f}], "
            f"mean_abs={ss['mean_abs']:.2f})"
        )
    ts = per_move_stats.get("root_top1_share")
    if ts is not None:
        lines.append(
            f"  root_top1_share: mean={ts['mean']:.2f} p50={ts['p50']:.2f} "
            f"p90={ts['p90']:.2f} p95={ts['p95']:.2f} "
            f"min={ts['min']:.2f}"
        )

    # Coverage line only when not uniform full coverage.
    is_uniform_full = (cov_ss == n_total) and (cov_ts == n_total)
    if not is_uniform_full:
        lines.append(
            f"  Coverage:        search_score={cov_ss}/{n_total} "
            f"root_top1_share={cov_ts}/{n_total}"
        )
    lines.append("")
    return lines


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
        f"adjudicated={o['adjudicated']} timeout={o['timeout']} "
        f"state_cap={o['state_cap']} draw_other={o['draw_other']}"
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


def format_goal_completion_report(gc: dict) -> List[str]:
    """Render the goal_completion summary block as report.txt lines (spec §7.5)."""
    lines: List[str] = []
    cfg = gc.get("config") or {}
    main = gc.get("main_population") or {}
    capped = gc.get("capped_population") or {}
    excluded = gc.get("excluded_population") or {}

    lines.append("Goal-Completion / Conversion Diagnostics")
    lines.append("========================================")

    if main.get("games", 0) == 0 and capped.get("games", 0) == 0:
        lines.append("No decisive or capped games in this run.")
        lines.append("")
        return lines

    if main.get("detected", 0) == 0 and capped.get("detected_before_cap", 0) == 0:
        lines.append(
            f"Config: detection<={cfg.get('detection_threshold')} / "
            f"max_depth={cfg.get('max_depth')} / "
            f"min_component={cfg.get('min_component_size')} / "
            f"high_value>={cfg.get('high_value_threshold')}"
        )
        lines.append(
            f"Population split: {main.get('games', 0)} decisive / "
            f"{capped.get('games', 0)} capped / "
            f"{excluded.get('games', 0)} excluded"
        )
        lines.append("No dominant-unclosed positions detected this run.")
        lines.append("")
        return lines

    lines.append(
        f"Config: detection<={cfg.get('detection_threshold')} / "
        f"max_depth={cfg.get('max_depth')} / "
        f"min_component={cfg.get('min_component_size')} / "
        f"high_value>={cfg.get('high_value_threshold')}"
    )
    lines.append(
        f"Population split: {main.get('games', 0)} decisive / "
        f"{capped.get('games', 0)} capped / "
        f"{excluded.get('games', 0)} excluded"
    )
    lines.append("")

    # Main population
    lines.append("Main (decisive wins, winner-only):")
    n_dom = main.get("games_with_dominant_unclosed", 0)
    n_games = main.get("games", 0)
    pct_dom = (n_dom / n_games * 100.0) if n_games else 0.0
    lines.append(
        f"  Dominant-unclosed reached: {n_dom} / {n_games} ({pct_dom:.1f}%)"
    )
    lines.append(
        f"    Strict closeout (<=2): {main.get('games_with_total_distance_le_2', 0)}    "
        f"Broader (<=3): {main.get('games_with_total_distance_le_3', 0)}"
    )
    lines.append(
        f"  Detected (gate=<={cfg.get('detection_threshold')}): "
        f"{main.get('detected', 0)}"
    )
    cd = main.get("conversion_delay_plies")
    if cd:
        lines.append("  Conversion delay:")
        lines.append(
            f"    plies:        p50={cd['p50']:.0f} p90={cd['p90']:.0f} "
            f"p95={cd['p95']:.0f} max={cd['max']:.0f} mean={cd['mean']:.1f}"
        )
        cdw = main.get("conversion_delay_winner_moves") or {}
        if cdw:
            lines.append(
                f"    winner moves: p50={cdw['p50']:.0f} p90={cdw['p90']:.0f} "
                f"max={cdw['max']:.0f} mean={cdw['mean']:.1f}"
            )
    mq = main.get("move_quality_after_detection") or {}
    if mq:
        lines.append("  Move quality after detection (pooled):")
        lines.append(f"    endpoint completion: {mq.get('completes_endpoint_rate', 0)*100:.1f}%")
        lines.append(f"    distance reducing:    {mq.get('reduces_total_goal_distance_rate', 0)*100:.1f}%")
        lines.append(f"    redundant reinforce: {mq.get('redundant_reinforcement_rate', 0)*100:.1f}%")
        lines.append(f"    off-chain:           {mq.get('off_chain_rate', 0)*100:.1f}%")
        lines.append(f"    other:                {mq.get('other_rate', 0)*100:.1f}%")
        lines.append(f"    dominant unavailable: {mq.get('dominant_unavailable_rate', 0)*100:.1f}%")
    hv = main.get("high_value_diagnostics") or {}
    cov = hv.get("search_score_coverage_pct", 0.0)
    if cov > 0:
        lines.append("  High value after detection:")
        max_p = hv.get("max_search_score_after_detection") or {}
        mean_p = hv.get("mean_search_score_after_detection") or {}
        if max_p:
            lines.append(
                f"    max search_score:  p50={max_p['p50']:.2f} "
                f"p90={max_p['p90']:.2f} max={max_p['max']:.2f}"
            )
        if mean_p:
            lines.append(
                f"    mean search_score: p50={mean_p['p50']:.2f} "
                f"p90={mean_p['p90']:.2f} max={mean_p['max']:.2f}"
            )
    bc = main.get("bad_cases") or {}
    if bc:
        lines.append("  Bad cases:")
        lines.append(f"    delay >=10 plies:               {bc.get('delay_ge_10_plies', 0)}")
        lines.append(f"    delay >=20 plies:                {bc.get('delay_ge_20_plies', 0)}")
        if cov > 0:
            lines.append(f"    high value but delayed:         {bc.get('root_value_high_but_delayed', 0)}")

    # Capped population
    if capped.get("games", 0) > 0:
        lines.append("")
        lines.append("Capped (state_cap / timeout / board_full):")
        lines.append(f"  Games:                              {capped.get('games', 0)}")
        lines.append(f"  Dominant unclosed before cap:       {capped.get('detected_before_cap', 0)}")
        cdcap = capped.get("cap_delay_after_detection_plies")
        if cdcap:
            lines.append("  Cap delay after detection:")
            lines.append(
                f"    plies: p50={cdcap['p50']:.0f} p90={cdcap['p90']:.0f} "
                f"max={cdcap['max']:.0f}"
            )
        cbc = capped.get("bad_cases") or {}
        lines.append("  Bad cases:")
        lines.append(f"    state_cap after detection:        {cbc.get('state_cap_after_detection', 0)}")
        lines.append(f"    timeout after detection:          {cbc.get('timeout_after_detection', 0)}")
        lines.append(f"    board_full after detection:       {cbc.get('board_full_after_detection', 0)}")

    lines.append("")
    return lines


def format_td_closeout_breakdown_report(breakdown: dict) -> list:
    """Format the td_closeout_breakdown section for report_<range>.txt.

    Spec 2026-05-10 §3.2. `breakdown` is the dict returned by
    aggregate_td_closeout_breakdown().
    """
    def _pct(x):
        return f"{(x or 0.0) * 100.0:.1f}%"

    lines = []
    lines.append("Closeout breakdown by total_goal_distance")
    lines.append("=========================================")
    for key in ("td=1", "td=2", "td=3"):
        b = breakdown.get(key) or {}
        n = b.get("records", 0)
        hv = b.get("high_value_records", 0)
        lines.append(f"{key}:  records={n}  high_value={hv}")
        if n == 0:
            lines.append("  (no records)")
            continue
        lines.append(
            "  selected: complete=" + _pct(b.get("selected_completes_endpoint_rate"))
            + "  reduce=" + _pct(b.get("selected_reduces_distance_rate"))
            + "  redundant=" + _pct(b.get("selected_redundant_rate"))
            + "  off-chain=" + _pct(b.get("selected_off_chain_rate"))
            + "  other=" + _pct(b.get("selected_other_rate"))
        )
        lines.append(
            "  endpoint exists: " + _pct(b.get("endpoint_completion_exists_rate"))
            + "  policy top5=" + _pct(b.get("endpoint_policy_top5_rate"))
            + "  visit top5=" + _pct(b.get("endpoint_visit_top5_rate"))
            + "  visit >20=" + _pct(b.get("endpoint_visit_gt20_rate"))
        )
        lines.append(
            "  reducer  exists: " + _pct(b.get("distance_reducer_exists_rate"))
            + "  policy top5=" + _pct(b.get("reducer_policy_top5_rate"))
            + "  visit top5=" + _pct(b.get("reducer_visit_top5_rate"))
            + "  visit >20=" + _pct(b.get("reducer_visit_gt20_rate"))
        )
    return lines


def format_policy_mcts_closeout_report(gc_block: dict) -> List[str]:
    """Render the policy/MCTS closeout behavior section per spec §8.7.

    Reads from gc_block (the goal_completion summary dict) — both
    diagnostics_coverage and policy_mcts_summary live there.
    """
    lines: List[str] = []
    coverage = (gc_block.get("diagnostics_coverage") or {})
    pms = gc_block.get("policy_mcts_summary")
    n_decisive_games = (gc_block.get("main_population") or {}).get("games", 0)

    if not pms or pms.get("n_records", 0) == 0:
        lines.append(
            f"Coverage: {coverage.get('games_with_diagnostics', 0)} / "
            f"{n_decisive_games} decisive games "
            f"({coverage.get('coverage_pct_of_decisive_games', 0):.1f}%); "
            f"{coverage.get('error_count', 0)} capture errors. "
            f"No closeout records captured this run."
        )
        lines.append("")
        return lines

    n_records = pms["n_records"]
    games_with = coverage.get("games_with_diagnostics", 0)
    pct = coverage.get("coverage_pct_of_decisive_games", 0)

    lines.append(f"Policy/MCTS closeout behavior (n={n_records} records across {games_with} games):")
    lines.append(
        f"  Coverage:                        {games_with} / {n_decisive_games} "
        f"decisive games ({pct:.1f}%); {coverage.get('error_count', 0)} capture errors"
    )
    er = pms.get("endpoint_completion_ranking") or {}
    if er.get("n_rankable", 0) > 0:
        lines.append(f"  Endpoint-completion ranking (n_rankable={er['n_rankable']}):")
        lines.append(
            f"    best completion in policy top1: {er['policy_top1_rate']*100:.1f}%   "
            f"policy top5: {er['policy_top5_rate']*100:.1f}%"
        )
        lines.append(
            f"    best completion in visit top1:  {er['visit_top1_rate']*100:.1f}%   "
            f"visit top5:  {er['visit_top5_rate']*100:.1f}%"
        )
    rr = pms.get("distance_reducing_ranking") or {}
    if rr.get("n_rankable", 0) > 0:
        lines.append(f"  Distance-reducing ranking (n_rankable={rr['n_rankable']}):")
        lines.append(
            f"    best reducer in policy top1:    {rr['policy_top1_rate']*100:.1f}%   "
            f"policy top5: {rr['policy_top5_rate']*100:.1f}%"
        )
        lines.append(
            f"    best reducer in visit top1:     {rr['visit_top1_rate']*100:.1f}%   "
            f"visit top5:  {rr['visit_top5_rate']*100:.1f}%"
        )
    rates = pms.get("selected_primary_class_rates") or {}
    lines.append("  Selected (primary class):")
    lines.append(f"    completes endpoint:    {rates.get('completes_endpoint', 0)*100:.1f}%")
    lines.append(f"    reduces distance:       {rates.get('reduces_total_goal_distance', 0)*100:.1f}%")
    lines.append(f"    redundant:             {rates.get('redundant_reinforcement', 0)*100:.1f}%")
    lines.append(f"    off-chain:             {rates.get('off_chain', 0)*100:.1f}%")
    lines.append(f"    other:                  {rates.get('other', 0)*100:.1f}%")
    lines.append(f"  High-value delayed closeouts:    {pms.get('high_value_delayed_closeouts', 0)}")
    by_dist = pms.get("by_distance") or {}
    le2 = by_dist.get("distance_le_2", {})
    eq3 = by_dist.get("distance_eq_3", {})
    if le2 or eq3:
        lines.append("  By distance:")
        if le2.get("n", 0):
            lines.append(f"    le_2 (n={le2['n']}): see policy_mcts_summary.by_distance for details")
        if eq3.get("n", 0):
            lines.append(f"    eq_3 (n={eq3['n']}): see policy_mcts_summary.by_distance for details")
    lines.append("")
    return lines


def write_goal_completion_worst_cases_csv(
    out_path: str, replays: list, top_k: int = 25,
) -> None:
    """Write worst-cases CSV from per-game goal_completion_records.

    Sort key: conversion_delay_plies for Class 1, cap_delay_proxy_plies
    for Class 2; replays without a record are skipped silently.
    """
    def _sort_delay(rec):
        if rec is None:
            return -1
        oc = rec.get("outcome_class")
        if oc == 1:
            return int(rec.get("conversion_delay_plies") or 0)
        if oc == 2:
            return int(rec.get("cap_delay_proxy_plies") or 0)
        return -1

    pairs = []
    for replay in replays:
        rec = replay.get("goal_completion_record")
        if rec is None:
            continue
        pairs.append((rec, replay))
    pairs.sort(key=lambda p: -_sort_delay(p[0]))
    top = pairs[:top_k]

    fieldnames = [
        "iteration", "game_idx", "game_id", "winner", "starting_player",
        "n_moves", "reason", "outcome_class", "scope",
        "detected_player", "first_dominant_unclosed_ply",
        "first_total_goal_distance", "first_category",
        "actual_terminal_ply", "actual_win_ply",
        "conversion_delay_plies", "conversion_delay_winner_moves",
        "cap_delay_proxy_plies",
        "primary_class_completes_endpoint",
        "primary_class_reduces_total_goal_distance",
        "primary_class_redundant_reinforcement",
        "primary_class_off_chain",
        "primary_class_other",
        "winner_moves_in_watch_window",
        "winner_moves_with_dominant_component",
        "winner_moves_with_dominant_unavailable",
        "max_search_score_after_detection",
        "mean_search_score_after_detection",
        "high_value_after_detection_plies",
        "root_value_high_but_delayed",
        "search_score_coverage_in_watch_window",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec, replay in top:
            pcc = rec.get("primary_class_counts") or {}
            row = {
                "iteration": rec.get("iteration"),
                "game_idx": rec.get("game_idx"),
                "game_id": rec.get("game_id") or (
                    f"iter_{_replay_int_field(replay, 'iteration'):04d}"
                    f"_game_{_replay_int_field(replay, 'game_idx'):03d}"
                ),
                "winner": rec.get("winner"),
                "starting_player": rec.get("starting_player"),
                "n_moves": rec.get("n_moves"),
                "reason": rec.get("reason"),
                "outcome_class": rec.get("outcome_class"),
                "scope": rec.get("scope"),
                "detected_player": rec.get("detected_player"),
                "first_dominant_unclosed_ply": rec.get("first_dominant_unclosed_ply"),
                "first_total_goal_distance": rec.get("first_total_goal_distance"),
                "first_category": rec.get("first_category"),
                "actual_terminal_ply": rec.get("actual_terminal_ply"),
                "actual_win_ply": rec.get("actual_win_ply"),
                "conversion_delay_plies": rec.get("conversion_delay_plies"),
                "conversion_delay_winner_moves": rec.get("conversion_delay_winner_moves"),
                "cap_delay_proxy_plies": rec.get("cap_delay_proxy_plies"),
                "primary_class_completes_endpoint": pcc.get("completes_endpoint") if pcc else None,
                "primary_class_reduces_total_goal_distance": pcc.get("reduces_total_goal_distance") if pcc else None,
                "primary_class_redundant_reinforcement": pcc.get("redundant_reinforcement") if pcc else None,
                "primary_class_off_chain": pcc.get("off_chain") if pcc else None,
                "primary_class_other": pcc.get("other") if pcc else None,
                "winner_moves_in_watch_window": rec.get("winner_moves_in_watch_window"),
                "winner_moves_with_dominant_component": rec.get("winner_moves_with_dominant_component"),
                "winner_moves_with_dominant_unavailable": rec.get("winner_moves_with_dominant_unavailable"),
                "max_search_score_after_detection": rec.get("max_search_score_after_detection"),
                "mean_search_score_after_detection": rec.get("mean_search_score_after_detection"),
                "high_value_after_detection_plies": rec.get("high_value_after_detection_plies"),
                "root_value_high_but_delayed": rec.get("root_value_high_but_delayed"),
                "search_score_coverage_in_watch_window": rec.get("search_score_coverage_in_watch_window"),
            }
            w.writerow(row)


def write_goal_completion_td_breakdown_csv(path: str, breakdown: dict) -> None:
    """Write one row per td_before bucket. Spec 2026-05-10 §3.3."""
    fields = [
        "td_before", "records", "high_value_records",
        "selected_completes_endpoint_rate", "selected_reduces_distance_rate",
        "selected_redundant_rate", "selected_off_chain_rate", "selected_other_rate",
        "endpoint_completion_exists_rate",
        "endpoint_policy_top1_rate", "endpoint_policy_top5_rate",
        "endpoint_policy_top20_rate", "endpoint_policy_gt20_rate",
        "endpoint_visit_top1_rate", "endpoint_visit_top5_rate",
        "endpoint_visit_top20_rate", "endpoint_visit_gt20_rate",
        "distance_reducer_exists_rate",
        "reducer_policy_top1_rate", "reducer_policy_top5_rate",
        "reducer_policy_top20_rate", "reducer_policy_gt20_rate",
        "reducer_visit_top1_rate", "reducer_visit_top5_rate",
        "reducer_visit_top20_rate", "reducer_visit_gt20_rate",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in ("td=1", "td=2", "td=3"):
            b = breakdown.get(key) or {}
            row = {"td_before": key.split("=", 1)[1]}
            for k in fields[1:]:
                row[k] = b.get(k, 0)
            w.writerow(row)


def write_recovery_events_csv(path: str, events: list) -> None:
    """One row per recovery event (spec §6.4)."""
    fields = [
        "iteration", "game_id", "winner", "detected_player",
        "first_detection_ply", "first_unavailable_ply", "dominant_unavailable_moves",
        "conversion_delay_winner_moves",
        "latest_largest_component_size", "latest_total_goal_distance",
        "q_at_first_unavailable", "q_at_terminal",
        "sel_completes_endpoint", "sel_reduces_distance",
        "sel_redundant_reinforcement", "sel_off_chain", "sel_other",
        "eventual_outcome", "recovery_class",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in events:
            sc = e.get("selected_class_counts_after_first_unavailable") or {}
            row = {
                "iteration": e.get("iteration"),
                "game_id": e.get("game_id"),
                "winner": e.get("winner"),
                "detected_player": e.get("detected_player"),
                "first_detection_ply": e.get("first_detection_ply"),
                "first_unavailable_ply": e.get("first_unavailable_ply"),
                "dominant_unavailable_moves": e.get("dominant_unavailable_moves"),
                "conversion_delay_winner_moves": e.get("conversion_delay_winner_moves"),
                "latest_largest_component_size": e.get("latest_largest_component_size"),
                "latest_total_goal_distance": e.get("latest_total_goal_distance"),
                "q_at_first_unavailable": e.get("q_at_first_unavailable"),
                "q_at_terminal": e.get("q_at_terminal"),
                "sel_completes_endpoint": sc.get("completes_endpoint", 0),
                "sel_reduces_distance": sc.get("reduces_total_goal_distance", 0),
                "sel_redundant_reinforcement": sc.get("redundant_reinforcement", 0),
                "sel_off_chain": sc.get("off_chain", 0),
                "sel_other": sc.get("other", 0),
                "eventual_outcome": e.get("eventual_outcome"),
                "recovery_class": e.get("recovery_class"),
            }
            w.writerow(row)


def format_recovery_events_report(events: list) -> list:
    """Format the recovery section for report_<range>.txt (spec §6.4)."""
    lines = []
    lines.append("Recovery / dominant-component-lost diagnostics")
    lines.append("===============================================")
    lines.append(f"Events: {len(events)}")
    if not events:
        return lines
    counts = {}
    for e in events:
        b = e.get("recovery_class") or "lost_other"
        counts[b] = counts.get(b, 0) + 1
    lines.append("By outcome:")
    for k in ("lost_then_recovered", "lost_then_won_late", "lost_then_state_cap",
              "lost_and_value_collapsed", "lost_but_value_stayed_high", "lost_other"):
        if k in counts:
            lines.append(f"  {k:30s} {counts[k]}")
    dom = sorted(int(e.get("dominant_unavailable_moves") or 0) for e in events)
    delays = sorted(int(e.get("conversion_delay_winner_moves") or 0) for e in events)
    def _median(xs):
        return xs[len(xs)//2] if xs else 0
    lines.append(f"Median dominant_unavailable_moves: {_median(dom)}")
    lines.append(f"Median delay (winner_moves):       {_median(delays)}")
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
    # --goal-completion-recompute-validate implies --goal-completion-recompute.
    if getattr(args, "goal_completion_recompute_validate", False):
        if hasattr(args, "goal_completion_recompute"):
            args.goal_completion_recompute = True
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

    # --- Spec 2 §8.8: conversion_training + recovery per-iter dicts ----------
    # Read-only: extracted from sidecars, no recomputation.
    conversion_training_by_iter: Dict[int, dict] = {}
    recovery_by_iter: Dict[int, dict] = {}
    if relevant_sidecars:
        conversion_training_by_iter = {
            it: sidecar_dict.get("conversion_training")
            for it, sidecar_dict in relevant_sidecars.items()
            if sidecar_dict.get("conversion_training") is not None
        }
        recovery_by_iter = {
            it: sidecar_dict.get("recovery_or_extreme_closeout_drift")
            for it, sidecar_dict in relevant_sidecars.items()
            if sidecar_dict.get("recovery_or_extreme_closeout_drift") is not None
        }

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
    per_move_stats_val = aggregate_per_move_stats(replays)
    per_game_stats_val = aggregate_per_game_stats(replays)
    if getattr(args, "goal_completion_recompute", False):
        # Recompute path (spec §11.5): walk replays via the new module
        # using pre-move detection semantics, merge with any inline
        # records, and aggregate via the shared aggregator.
        from scripts.GPU.alphazero.goal_completion_recompute import (
            recompute_goal_completion_records_from_replays,
        )
        per_game_inline = [r.get("goal_completion_record") for r in replays]
        recomputed = recompute_goal_completion_records_from_replays(
            replays,
            config={
                "detection_threshold": getattr(args, "goal_completion_detection_threshold", 2) if args else 2,
                "max_depth": getattr(args, "goal_completion_max_depth", 3) if args else 3,
                "min_component_size": getattr(args, "goal_completion_min_component_size", 8) if args else 8,
                "high_value_threshold": getattr(args, "goal_completion_high_value_threshold", 0.9) if args else 0.9,
                "high_value_delay_threshold_plies": 6,
            },
        )
        merged = _merge_inline_with_recomputed(per_game_inline, recomputed)
        if getattr(args, "goal_completion_recompute_validate", False):
            from scripts.GPU.alphazero.goal_completion_recompute import (
                compare_records_for_validation,
            )
            n_diverge = 0
            for inline_rec, rec_rec, replay in zip(per_game_inline, recomputed, replays):
                div = compare_records_for_validation(inline_rec, rec_rec)
                if div:
                    n_diverge += 1
                    gid = (inline_rec or rec_rec or {}).get("game_id") or (
                        f"iter_{_replay_int_field(replay, 'iteration'):04d}"
                        f"_game_{_replay_int_field(replay, 'game_idx'):03d}"
                    )
                    print(f"[VALIDATE] {gid}: {len(div)} fields diverge",
                          file=sys.stderr)
                    for fname, (a, b) in div.items():
                        print(f"    {fname}: inline={a!r}  recomputed={b!r}",
                              file=sys.stderr)
            if n_diverge == 0:
                print(
                    f"[VALIDATE] All {len(replays)} replays match between "
                    f"inline and recomputed paths.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[VALIDATE] {n_diverge}/{len(replays)} replays diverge.",
                    file=sys.stderr,
                )
        goal_completion_val = aggregate_goal_completion_records(
            merged,
            config={
                "detection_threshold": getattr(args, "goal_completion_detection_threshold", 2) if args else 2,
                "emit_threshold": 3,
                "high_value_threshold": getattr(args, "goal_completion_high_value_threshold", 0.9) if args else 0.9,
                "high_value_delay_threshold_plies": 6,
                "max_depth": getattr(args, "goal_completion_max_depth", 3) if args else 3,
                "min_component_size": getattr(args, "goal_completion_min_component_size", 8) if args else 8,
            },
            games_total=len(replays),
        )
    else:
        goal_completion_val = aggregate_goal_completion_diagnostics_from_records(
            replays,
            sidecar_summaries={
                it: sc.get("goal_completion_summary")
                for it, sc in (relevant_sidecars or {}).items()
                if sc.get("goal_completion_summary") is not None
            },
            config={
                "detection_threshold": getattr(args, "goal_completion_detection_threshold", 2) if args else 2,
                "emit_threshold": 3,
                "high_value_threshold": getattr(args, "goal_completion_high_value_threshold", 0.9) if args else 0.9,
                "high_value_delay_threshold_plies": 6,
                "max_depth": getattr(args, "goal_completion_max_depth", 3) if args else 3,
                "min_component_size": getattr(args, "goal_completion_min_component_size", 8) if args else 8,
            },
        )

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
        "per_move_stats": per_move_stats_val,
        "per_game_stats": per_game_stats_val,
        "goal_completion": goal_completion_val,
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

    # --- Spec 2 §8.8: conversion_training + recovery per-iter CSVs ---
    if conversion_training_by_iter:
        _cnv_csv = write_conversion_training_by_iter_csv(
            conversion_training_by_iter, out_dir, suffix=suffix
        )
        print(f"[OK] wrote: {_cnv_csv}")
    if recovery_by_iter:
        _rcv_csv = write_recovery_or_extreme_closeout_drift_by_iter_csv(
            recovery_by_iter, out_dir, suffix=suffix
        )
        print(f"[OK] wrote: {_rcv_csv}")

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

    # Goal-completion worst cases CSV
    if not getattr(args, "goal_completion_recompute", False):
        write_goal_completion_worst_cases_csv(
            os.path.join(out_dir, _suffixed("goal_completion_worst_cases", "csv", suffix)),
            replays,
            top_k=getattr(args, "goal_completion_worst_cases_top_k", 25) if args else 25,
        )
    # Recompute path's own CSV writer is preserved; Task 13 wires it.

    # Spec 2026-05-10 §3.3 — td_closeout breakdown CSV. Always emitted
    # so downstream tooling has a stable filename; empty buckets render as
    # zero rows.
    write_goal_completion_td_breakdown_csv(
        os.path.join(out_dir, _suffixed("goal_completion_td_breakdown", "csv", suffix)),
        (summary.get("goal_completion") or {}).get("td_closeout_breakdown") or {},
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
    # `n` counts every loaded replay file (entire input directory).
    # For users running on a multi-iteration logs/games dir, this is
    # often higher than the in-range count derived from sidecars.
    in_range_count = len([r for r in rows if it_min is not None
                          and it_min <= r.iteration <= it_max])
    if in_range_count and in_range_count != n:
        lines.append(f"Games analyzed: {in_range_count} in iter range "
                     f"({n} total loaded from input)")
    else:
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

    # Per-game stats triage section (spec 2026-04-29). Rendered for both
    # sidecar and replay_fallback modes — the section's own null-handling
    # covers the all-old-schema case per spec §7 / §5.1.
    lines.extend(format_per_move_stats_report(summary["per_move_stats"]))
    lines.extend(format_per_game_stats_report(summary["per_game_stats"]))
    lines.extend(format_goal_completion_report(summary["goal_completion"]))
    # Spec 2026-05-10 §3.2 — closeout breakdown by total_goal_distance.
    td_breakdown = (summary.get("goal_completion") or {}).get("td_closeout_breakdown")
    if td_breakdown:
        lines.append("")
        lines.extend(format_td_closeout_breakdown_report(td_breakdown))
    lines.extend(format_policy_mcts_closeout_report(summary["goal_completion"]))
    lines.extend(format_conversion_training_trend_report(conversion_training_by_iter))
    lines.extend(format_recovery_or_extreme_closeout_drift_report(recovery_by_iter))
    # Spec 2026-05-10 §6 — per-game recovery event classification + CSV.
    recovery_events = aggregate_recovery_events(replays)
    write_recovery_events_csv(
        os.path.join(out_dir, _suffixed("recovery_events", "csv", suffix)),
        recovery_events,
    )
    lines.append("")
    lines.extend(format_recovery_events_report(recovery_events))

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

    # Goal-completion diagnostics (Phase 2, spec 2026-05-03 §7).
    ap.add_argument("--goal-completion-detection-threshold", type=int, default=2,
                    help="Phase 2 detection threshold for total_goal_distance (default: 2)")
    ap.add_argument("--goal-completion-high-value-threshold", type=float, default=0.9,
                    help="search_score threshold for high-value bad-case detection (default: 0.9)")
    ap.add_argument("--goal-completion-worst-cases-top-k", type=int, default=25,
                    help="Top-K worst cases to write to CSV (default: 25)")
    ap.add_argument("--goal-completion-max-depth", type=int, default=3,
                    help="Max BFS depth for endpoint distance computation (default: 3)")
    ap.add_argument("--goal-completion-min-component-size", type=int, default=8,
                    help="Min component size to qualify as dominant-unclosed (default: 8)")
    ap.add_argument("--goal-completion-recompute", action="store_true",
                    default=False,
                    help="Use the recompute walker for goal-completion (pre-move semantics). "
                         "Default: read pre-computed records from per-game JSONs.")
    ap.add_argument("--goal-completion-recompute-validate", action="store_true",
                    default=False,
                    help="With --goal-completion-recompute, also load inline "
                         "records and report per-field divergence. Implies "
                         "--goal-completion-recompute. Intentionally expensive.")

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