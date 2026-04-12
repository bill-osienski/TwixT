# Opening Penalty Diagnostics Design

## Problem

Opening geometry has regressed — edge/corner play is increasing — but we cannot determine *why* from existing metrics. Current replay analysis shows *that* edge/corner rates changed, but not whether:
- the model itself prefers edge/corner moves (raw prior problem)
- penalties are too weak to overcome that preference (penalty strength problem)
- search undoes the penalty effect (search/value interaction problem)
- penalties stop too early and behavior rebounds (penalty window problem)

## Goal

Add instrumentation that captures the full diagnostic pipeline at each opening root search — what the network wanted, what penalties changed, and what search chose — so we can identify the failure mode and make targeted fixes.

## Architecture

**B is source of truth, sidecar gets compact aggregates.**

- **Per-game JSON files** store raw per-root diagnostic records for every ply in the diagnostic window
- **Per-iteration sidecar** stores pre-aggregated summaries broken out by ply and color
- Aggregation happens in the trainer after all games complete (not in the self-play hot path)

### Data Flow

```
self_play.py (game loop)
  After each root search (ply < diagnostic_end_ply):
    Capture root.priors_raw (model output) and root.priors (post-root-adjustment)
    Capture visit_counts (post-search)
    Classify each legal move into regions
    Build per-root diagnostic record
    Attach to GameRecord

GameRecord -> trainer via IPC (opening_diagnostics field on GameResult)

game_saver.py
  Writes opening_diagnostics + opening_diagnostics_meta into per-game JSON

trainer.py (after all games complete)
  Aggregates per-game diagnostics into sidecar opening_penalty_diagnostics
  Writes to iter_NNNN_stats.json
```

### Files Modified

| File | Change |
|------|--------|
| `scripts/GPU/alphazero/self_play.py` | Build per-root diagnostic records in game loop, attach to GameRecord |
| `scripts/GPU/alphazero/ipc_messages.py` | Add `opening_diagnostics` field to GameResult message |
| `scripts/GPU/alphazero/game_saver.py` | Write `opening_diagnostics` and `opening_diagnostics_meta` to game JSON |
| `scripts/GPU/alphazero/trainer.py` | Aggregate per-game diagnostics into sidecar `opening_penalty_diagnostics` |
| `scripts/GPU/alphazero/mcts.py` | No changes needed — `root.priors_raw` and `root.priors` already available |

## Diagnostic Window

```python
diagnostic_end_ply = max(
    mcts_config.root_edge_band_penalty_ply,
    mcts_config.root_near_corner_penalty_ply,
    4  # minimum floor even if penalties disabled (baseline measurement)
) + 2
```

- Records all plies where penalties are active
- Records 2 extra plies after penalties end (rebound detection)
- Floor of 4 ensures diagnostics even when penalties are disabled

## Regional Classification

### Boolean flags (per-game records — full truth)

Each legal move gets:
- `is_edge_band`: true if within `root_edge_band_width` of any board edge
- `is_near_corner`: true if within Chebyshev `root_near_corner_radius` of any corner

A move can be both. Uses the same definitions as the penalty code in `mcts.py`.

### Exclusive buckets (mass summaries — clean aggregates)

Priority: `near_corner` > `edge_band` > `interior`

- If `is_near_corner` -> bucket as `near_corner`
- Else if `is_edge_band` -> bucket as `edge_band`
- Else -> `interior`

Mass always sums to ~1.0 across the three buckets.

## Three Diagnostic Stages

| Stage | Source | Meaning |
|-------|--------|---------|
| **raw** | `root.priors_raw` | What the neural network output (no noise, no penalties) |
| **penalized** | `root.priors` | Post-root-adjustment priors (includes Dirichlet noise + penalties). In code comments: "post-root-adjustment priors" |
| **visit** | `visit_counts` | What MCTS search chose (final visit distribution) |

Note: "penalized" includes both noise and penalties. Noise is random and washes out in aggregation. The diagnostic value is comparing model output vs search input vs search output.

## Per-Game JSON Schema

### Metadata (one per game)

```json
"opening_diagnostics_meta": {
  "version": 1,
  "diagnostic_end_ply": 18,
  "extra_plies_after_penalty": 2,
  "floor_min_ply": 4,
  "used_floor": false
}
```

### Per-Root Records (array, one per ply in window)

```json
"opening_diagnostics": [
  {
    "ply": 0,
    "side_to_move": "red",
    "penalties_active": {"edge_band": true, "near_corner": true},
    "config": {
      "edge_band_width": 2,
      "edge_band_penalty": 1.5,
      "near_corner_radius": 3,
      "near_corner_penalty": 2.0
    },
    "legal_move_counts": {"near_corner": 8, "edge_band": 14, "interior": 120},
    "raw_mass": {"near_corner": 0.35, "edge_band": 0.20, "interior": 0.45},
    "penalized_mass": {"near_corner": 0.10, "edge_band": 0.12, "interior": 0.78},
    "visit_mass": {"near_corner": 0.08, "edge_band": 0.15, "interior": 0.77},
    "raw_top1": {
      "move": [2, 19],
      "share": 0.15,
      "is_edge_band": true,
      "is_near_corner": false,
      "primary_region": "edge_band"
    },
    "penalized_top1": {
      "move": [10, 12],
      "share": 0.09,
      "is_edge_band": false,
      "is_near_corner": false,
      "primary_region": "interior"
    },
    "visit_top1": {
      "move": [10, 12],
      "share": 0.22,
      "is_edge_band": false,
      "is_near_corner": false,
      "primary_region": "interior"
    }
  }
]
```

## Per-Iteration Sidecar Schema

### Opening Penalty Diagnostics Section

```json
"opening_penalty_diagnostics": {
  "version": 1,
  "diagnostic_end_ply": 18,
  "extra_plies_after_penalty": 2,
  "floor_min_ply": 4,
  "used_floor": false,
  "games_total": 100,
  "all_diagnostic_plies": {
    "red": {
      "n": 520,
      "mean_raw_mass": {"near_corner": 0.30, "edge_band": 0.18, "interior": 0.52},
      "mean_penalized_mass": {"near_corner": 0.08, "edge_band": 0.10, "interior": 0.82},
      "mean_visit_mass": {"near_corner": 0.06, "edge_band": 0.12, "interior": 0.82},
      "mean_penalty_shift": {"near_corner": -0.22, "edge_band": -0.08, "interior": 0.30},
      "raw_top1_region_pct": {"near_corner": 0.25, "edge_band": 0.35, "interior": 0.40},
      "penalized_top1_region_pct": {"near_corner": 0.04, "edge_band": 0.12, "interior": 0.84},
      "visit_top1_region_pct": {"near_corner": 0.02, "edge_band": 0.10, "interior": 0.88},
      "mean_legal_counts": {"near_corner": 8, "edge_band": 14, "interior": 120}
    },
    "black": { "..." : "same structure" }
  },
  "by_ply": {
    "0": {
      "red": {
        "n": 52,
        "penalties_active": {"edge_band": true, "near_corner": true},
        "mean_raw_mass": {"near_corner": 0.31, "edge_band": 0.18, "interior": 0.51},
        "mean_penalized_mass": {"near_corner": 0.09, "edge_band": 0.11, "interior": 0.80},
        "mean_visit_mass": {"near_corner": 0.07, "edge_band": 0.13, "interior": 0.80},
        "mean_penalty_shift": {"near_corner": -0.22, "edge_band": -0.07, "interior": 0.29},
        "raw_top1_region_pct": {"near_corner": 0.25, "edge_band": 0.35, "interior": 0.40},
        "penalized_top1_region_pct": {"near_corner": 0.04, "edge_band": 0.12, "interior": 0.84},
        "visit_top1_region_pct": {"near_corner": 0.02, "edge_band": 0.10, "interior": 0.88},
        "mean_legal_counts": {"near_corner": 8, "edge_band": 14, "interior": 120}
      },
      "black": { "..." : "same structure" }
    },
    "1": { "..." : "same structure per color" },
    "16": {
      "red": {
        "n": 52,
        "penalties_active": {"edge_band": false, "near_corner": false},
        "mean_raw_mass": {"near_corner": 0.28, "edge_band": 0.22, "interior": 0.50},
        "mean_penalized_mass": {"near_corner": 0.28, "edge_band": 0.22, "interior": 0.50},
        "mean_visit_mass": {"near_corner": 0.25, "edge_band": 0.20, "interior": 0.55},
        "mean_penalty_shift": {"near_corner": 0.0, "edge_band": 0.0, "interior": 0.0},
        "rebound_vs_last_active": {
          "near_corner_mass_delta": 0.17,
          "edge_band_mass_delta": 0.09
        },
        "raw_top1_region_pct": {"near_corner": 0.30, "edge_band": 0.30, "interior": 0.40},
        "penalized_top1_region_pct": {"near_corner": 0.30, "edge_band": 0.30, "interior": 0.40},
        "visit_top1_region_pct": {"near_corner": 0.28, "edge_band": 0.25, "interior": 0.47},
        "mean_legal_counts": {"near_corner": 8, "edge_band": 14, "interior": 120}
      },
      "black": { "..." : "same structure" }
    }
  }
}
```

### Rebound Metric

For plies where `penalties_active` is false (post-penalty window), include:

```json
"rebound_vs_last_active": {
  "near_corner_mass_delta": 0.17,
  "edge_band_mass_delta": 0.09
}
```

Computed as: `mean_visit_mass[region] at this ply` minus `mean_visit_mass[region] at last active ply`.

Positive values indicate mass flowing back to edge/corner after penalties stop (rebound).

### Aggregation Rules

| Field | Method |
|-------|--------|
| `n` | Count of games where that player moved at that ply |
| `mean_*_mass` | Arithmetic mean across games |
| `mean_penalty_shift` | `mean_penalized_mass - mean_raw_mass` per region |
| `*_top1_region_pct` | Fraction of games where top-1 move was in that region (sums to 1.0, denominator = n) |
| `mean_legal_counts` | Arithmetic mean of legal move counts per region |
| `rebound_vs_last_active` | Difference in mean_visit_mass vs last penalty-active ply |
| `all_diagnostic_plies` | Aggregate of all per-ply entries (weighted by n) |

## Diagnostic Interpretation Guide

| Observation | Diagnosis | Response |
|------------|-----------|----------|
| Raw mass heavily edge/corner | Model preference problem | Strengthen penalties |
| Raw mass OK, penalized mass still edge/corner | Penalties too weak or narrow | Raise penalty strength, widen band/radius |
| Penalized mass OK, visit mass collapses back | Search/value interaction | Increase penalties further (not surviving search) |
| Active plies OK, post-penalty plies rebound | Penalty window too short | Extend penalty ply window |
| Only certain plies regress | Ply-specific issue | Tune window start/end |

## Scope Control

- Only record diagnostics for plies within the diagnostic window
- Root summaries only (not every MCTS child node)
- No per-move detail beyond top-1 (top-k is a future refinement)
- Aggregation happens in trainer after all games complete (not in self-play hot path)
